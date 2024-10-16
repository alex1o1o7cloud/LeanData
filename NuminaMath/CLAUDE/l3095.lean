import Mathlib

namespace NUMINAMATH_CALUDE_solutions_of_absolute_value_equation_l3095_309583

theorem solutions_of_absolute_value_equation :
  {x : ℝ | |x - 2| + |x - 3| = 1} = Set.Icc 2 3 := by sorry

end NUMINAMATH_CALUDE_solutions_of_absolute_value_equation_l3095_309583


namespace NUMINAMATH_CALUDE_smallest_term_is_fifth_l3095_309532

def a (n : ℕ) : ℤ := 3 * n^2 - 28 * n

theorem smallest_term_is_fifth : 
  ∀ k : ℕ, k ≠ 0 → a 5 ≤ a k :=
sorry

end NUMINAMATH_CALUDE_smallest_term_is_fifth_l3095_309532


namespace NUMINAMATH_CALUDE_problem_statement_l3095_309541

theorem problem_statement (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) : 
  (∀ x y, x > 0 ∧ y > 0 ∧ x + y = 1 → x * y ≤ a * b) ∧
  (a^2 + b^2 ≥ 1/2) ∧
  (4/a + 1/b ≥ 9) := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l3095_309541


namespace NUMINAMATH_CALUDE_number_of_nieces_l3095_309596

def hand_mitts_price : ℚ := 14
def apron_price : ℚ := 16
def utensils_price : ℚ := 10
def knife_price : ℚ := 2 * utensils_price
def discount_rate : ℚ := 1/4
def total_spending : ℚ := 135

def discounted_price (price : ℚ) : ℚ :=
  price * (1 - discount_rate)

def gift_set_price : ℚ :=
  discounted_price hand_mitts_price +
  discounted_price apron_price +
  discounted_price utensils_price +
  discounted_price knife_price

theorem number_of_nieces :
  total_spending / gift_set_price = 3 := by sorry

end NUMINAMATH_CALUDE_number_of_nieces_l3095_309596


namespace NUMINAMATH_CALUDE_larger_number_problem_l3095_309580

theorem larger_number_problem (x y : ℝ) : 
  x + y = 84 → y = 3 * x → max x y = 63 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_problem_l3095_309580


namespace NUMINAMATH_CALUDE_quadratic_equation_solutions_l3095_309565

theorem quadratic_equation_solutions :
  {x : ℝ | x^2 - Real.sqrt 2 * x = 0} = {0, Real.sqrt 2} := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solutions_l3095_309565


namespace NUMINAMATH_CALUDE_computer_price_comparison_l3095_309553

theorem computer_price_comparison (price1 : ℝ) (discount1 : ℝ) (discount2 : ℝ) (price_diff : ℝ) : 
  price1 = 950 ∧ 
  discount1 = 0.06 ∧ 
  discount2 = 0.05 ∧ 
  price_diff = 19 →
  ∃ (price2 : ℝ), 
    price2 * (1 - discount2) = price1 * (1 - discount1) + price_diff ∧ 
    price2 = 960 := by
  sorry

end NUMINAMATH_CALUDE_computer_price_comparison_l3095_309553


namespace NUMINAMATH_CALUDE_baseball_team_size_l3095_309568

/-- Calculates the number of players on a team given the total points, 
    points scored by one player, and points scored by each other player -/
def team_size (total_points : ℕ) (one_player_points : ℕ) (other_player_points : ℕ) : ℕ :=
  (total_points - one_player_points) / other_player_points + 1

/-- Theorem stating that for the given conditions, the team size is 6 -/
theorem baseball_team_size : 
  team_size 68 28 8 = 6 := by
  sorry

end NUMINAMATH_CALUDE_baseball_team_size_l3095_309568


namespace NUMINAMATH_CALUDE_coprime_2013_in_32nd_group_l3095_309505

def is_coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

def group_size (n : ℕ) : ℕ := 2 * n - 1

def cumulative_group_size (n : ℕ) : ℕ := n^2

def coprime_count (n : ℕ) : ℕ := n - (n.div 2 + n.div 503 - n.div 1006)

theorem coprime_2013_in_32nd_group :
  ∃ k : ℕ, k = 32 ∧
    coprime_count 2012 < cumulative_group_size (k - 1) ∧
    coprime_count 2012 + 1 ≤ cumulative_group_size k ∧
    is_coprime 2013 2012 := by
  sorry

end NUMINAMATH_CALUDE_coprime_2013_in_32nd_group_l3095_309505


namespace NUMINAMATH_CALUDE_unique_function_satisfying_condition_l3095_309545

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, n = k * k

def satisfies_condition (f : ℕ → ℕ) : Prop :=
  ∀ a b : ℕ, is_perfect_square (f a * f (a + b) - a * b)

theorem unique_function_satisfying_condition :
  ∃! f : ℕ → ℕ, satisfies_condition f ∧ ∀ x : ℕ, f x = x :=
sorry

end NUMINAMATH_CALUDE_unique_function_satisfying_condition_l3095_309545


namespace NUMINAMATH_CALUDE_thursday_coffee_consumption_l3095_309552

/-- Represents the relationship between coffee consumption, sleep, and preparation time -/
def coffee_relation (k : ℝ) (c h p : ℝ) : Prop :=
  c * (h + p) = k

theorem thursday_coffee_consumption 
  (k : ℝ)
  (c_wed h_wed p_wed : ℝ)
  (h_thu p_thu : ℝ)
  (hw : coffee_relation k c_wed h_wed p_wed)
  (wed_data : c_wed = 3 ∧ h_wed = 8 ∧ p_wed = 2)
  (thu_data : h_thu = 5 ∧ p_thu = 3) :
  ∃ c_thu : ℝ, coffee_relation k c_thu h_thu p_thu ∧ c_thu = 15/4 := by
  sorry

end NUMINAMATH_CALUDE_thursday_coffee_consumption_l3095_309552


namespace NUMINAMATH_CALUDE_min_side_length_triangle_l3095_309533

theorem min_side_length_triangle (a b c : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (∃ (h1 h2 h3 : ℝ), h1 > 0 ∧ h2 > 0 ∧ h3 > 0 ∧
    h1 * a = h2 * b ∧ h2 * b = h3 * c ∧
    h1 = 3 ∧ h2 = 4 ∧ h3 = 5) →
  min a (min b c) ≥ 12 :=
by sorry

end NUMINAMATH_CALUDE_min_side_length_triangle_l3095_309533


namespace NUMINAMATH_CALUDE_oil_change_time_is_15_minutes_l3095_309536

/-- Represents the time in minutes for various car maintenance tasks -/
structure CarMaintenanceTimes where
  washTime : ℕ
  oilChangeTime : ℕ
  tireChangeTime : ℕ

/-- Represents the number of tasks performed -/
structure TasksCounts where
  carsWashed : ℕ
  oilChanges : ℕ
  tireChanges : ℕ

/-- Calculates the total time spent on tasks -/
def totalTime (times : CarMaintenanceTimes) (counts : TasksCounts) : ℕ :=
  times.washTime * counts.carsWashed +
  times.oilChangeTime * counts.oilChanges +
  times.tireChangeTime * counts.tireChanges

/-- The main theorem to prove -/
theorem oil_change_time_is_15_minutes 
  (times : CarMaintenanceTimes)
  (counts : TasksCounts)
  (h1 : times.washTime = 10)
  (h2 : times.tireChangeTime = 30)
  (h3 : counts.carsWashed = 9)
  (h4 : counts.oilChanges = 6)
  (h5 : counts.tireChanges = 2)
  (h6 : totalTime times counts = 4 * 60) :
  times.oilChangeTime = 15 := by
  sorry


end NUMINAMATH_CALUDE_oil_change_time_is_15_minutes_l3095_309536


namespace NUMINAMATH_CALUDE_a_11_value_l3095_309569

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem a_11_value (a : ℕ → ℝ) 
    (h_arithmetic : arithmetic_sequence a)
    (h_a1 : a 1 = 1)
    (h_diff : ∀ n : ℕ, a (n + 2) - a n = 6) :
  a 11 = 31 := by
sorry

end NUMINAMATH_CALUDE_a_11_value_l3095_309569


namespace NUMINAMATH_CALUDE_min_distance_circle_to_line_l3095_309581

theorem min_distance_circle_to_line :
  let circle := {(x, y) : ℝ × ℝ | x^2 + y^2 - 2*x - 2*y + 1 = 0}
  let line := {(x, y) : ℝ × ℝ | 3*x + 4*y + 8 = 0}
  (∀ p ∈ circle, ∃ q ∈ line, ∀ r ∈ line, dist p q ≤ dist p r) ∧
  (∃ p ∈ circle, ∃ q ∈ line, dist p q = 2) ∧
  (∀ p ∈ circle, ∀ q ∈ line, dist p q ≥ 2) :=
by sorry

where
  dist : ℝ × ℝ → ℝ × ℝ → ℝ := λ (x₁, y₁) (x₂, y₂) => Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2)

end NUMINAMATH_CALUDE_min_distance_circle_to_line_l3095_309581


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3095_309556

def A : Set ℝ := {x | 2 * x ≤ 1}
def B : Set ℝ := {-1, 0, 1}

theorem intersection_of_A_and_B : A ∩ B = {-1, 0} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3095_309556


namespace NUMINAMATH_CALUDE_fraction_evaluation_l3095_309566

theorem fraction_evaluation :
  let x : ℚ := 4/3
  let y : ℚ := 5/7
  (3*x + 7*y) / (21*x*y) = 9/140 := by
sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l3095_309566


namespace NUMINAMATH_CALUDE_senior_citizen_discount_l3095_309501

/-- Calculates the senior citizen discount percentage given the number of shorts and shirts bought,
    their respective prices, and the total amount paid. -/
theorem senior_citizen_discount
  (num_shorts num_shirts : ℕ)
  (price_shorts price_shirts : ℚ)
  (total_paid : ℚ)
  (h1 : num_shorts = 3)
  (h2 : num_shirts = 5)
  (h3 : price_shorts = 15)
  (h4 : price_shirts = 17)
  (h5 : total_paid = 117) :
  (1 - total_paid / (num_shorts * price_shorts + num_shirts * price_shirts)) * 100 = 10 := by
  sorry

end NUMINAMATH_CALUDE_senior_citizen_discount_l3095_309501


namespace NUMINAMATH_CALUDE_range_of_a_l3095_309574

open Real

noncomputable def f (a x : ℝ) : ℝ := exp x * (2 * x - 1) - 2 * a * x + 2 * a

theorem range_of_a (a : ℝ) :
  (a < 1) →
  (∃! (x₀ : ℤ), f a (x₀ : ℝ) < 0) →
  a ∈ Set.Icc (3 / (4 * exp 1)) (1 / 2) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l3095_309574


namespace NUMINAMATH_CALUDE_three_subset_M_l3095_309578

def M : Set ℤ := {x | ∃ n : ℤ, x = 4 * n - 1}

theorem three_subset_M : {3} ⊆ M := by
  sorry

end NUMINAMATH_CALUDE_three_subset_M_l3095_309578


namespace NUMINAMATH_CALUDE_two_digit_property_three_digit_property_l3095_309589

/-- Two-digit positive integer -/
def TwoDigitInt (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

/-- Three-digit positive integer -/
def ThreeDigitInt (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

/-- Converts a two-digit number to its digits -/
def toDigits2 (n : ℕ) : ℕ × ℕ := (n / 10, n % 10)

/-- Converts a three-digit number to its digits -/
def toDigits3 (n : ℕ) : ℕ × ℕ × ℕ := (n / 100, (n / 10) % 10, n % 10)

theorem two_digit_property (n : ℕ) (h : TwoDigitInt n) :
  let (a, b) := toDigits2 n
  (a + 1) * (b + 1) = n + 1 ↔ b = 9 := by sorry

theorem three_digit_property (n : ℕ) (h : ThreeDigitInt n) :
  let (a, b, c) := toDigits3 n
  (a + 1) * (b + 1) * (c + 1) = n + 1 ↔ b = 9 ∧ c = 9 := by sorry

end NUMINAMATH_CALUDE_two_digit_property_three_digit_property_l3095_309589


namespace NUMINAMATH_CALUDE_min_value_of_sum_l3095_309559

theorem min_value_of_sum (a b : ℝ) : 
  a > 0 → b > 0 → 
  (∀ x y : ℝ, a * x + 2 * b * y - 2 = 0) → 
  (∀ x y : ℝ, x^2 + y^2 - 4*x - 2*y - 8 = 0) → 
  (∀ x y : ℝ, a * x + 2 * b * y - 2 = 0 → 
    (x - 2)^2 + (y - 1)^2 = 9) → 
  (1 / (2 * a) + 1 / b) ≥ (3 + 2 * Real.sqrt 2) / 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_sum_l3095_309559


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l3095_309598

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x, (1 - 2*x)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  a₁ + a₂ + a₃ + a₄ + a₅ = -2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l3095_309598


namespace NUMINAMATH_CALUDE_intersections_for_12_6_l3095_309515

/-- The maximum number of intersection points in the first quadrant -/
def max_intersections (x_points y_points : ℕ) : ℕ :=
  Nat.choose x_points 2 * Nat.choose y_points 2

/-- Theorem stating the maximum number of intersection points for 12 x-axis points and 6 y-axis points -/
theorem intersections_for_12_6 :
  max_intersections 12 6 = 990 := by
  sorry

end NUMINAMATH_CALUDE_intersections_for_12_6_l3095_309515


namespace NUMINAMATH_CALUDE_smallest_difference_in_triangle_l3095_309599

theorem smallest_difference_in_triangle (PQ PR QR : ℕ) : 
  PQ + PR + QR = 2021 →  -- Perimeter condition
  PQ < PR →              -- PQ < PR condition
  PR = (3 * PQ) / 2 →    -- PR = 1.5 × PQ condition
  PQ > 0 ∧ PR > 0 ∧ QR > 0 →  -- Positive side lengths
  PQ + QR > PR ∧ PR + QR > PQ ∧ PQ + PR > QR →  -- Triangle inequality
  PR - PQ ≥ 204 :=
by sorry

end NUMINAMATH_CALUDE_smallest_difference_in_triangle_l3095_309599


namespace NUMINAMATH_CALUDE_linear_function_property_l3095_309503

theorem linear_function_property (x : ℝ) : ∃ x > -1, -2 * x + 2 ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_linear_function_property_l3095_309503


namespace NUMINAMATH_CALUDE_eli_calculation_l3095_309586

theorem eli_calculation (x : ℝ) (h : (8 * x - 7) / 5 = 63) : (5 * x - 7) / 8 = 24.28125 := by
  sorry

end NUMINAMATH_CALUDE_eli_calculation_l3095_309586


namespace NUMINAMATH_CALUDE_prob_at_least_one_red_correct_l3095_309523

/-- Represents a bag containing balls of different colors -/
structure Bag where
  red : ℕ
  yellow : ℕ

/-- Calculate the probability of drawing at least one red ball from two bags -/
def probAtLeastOneRed (bagA bagB : Bag) : ℚ :=
  1 - (bagA.yellow * bagB.yellow : ℚ) / ((bagA.red + bagA.yellow) * (bagB.red + bagB.yellow))

theorem prob_at_least_one_red_correct :
  let bagA : Bag := ⟨1, 1⟩
  let bagB : Bag := ⟨2, 1⟩
  probAtLeastOneRed bagA bagB = 5/6 := by
  sorry

#eval probAtLeastOneRed ⟨1, 1⟩ ⟨2, 1⟩

end NUMINAMATH_CALUDE_prob_at_least_one_red_correct_l3095_309523


namespace NUMINAMATH_CALUDE_parallel_transitive_perpendicular_from_line_l3095_309550

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations between lines and planes
variable (parallel : Plane → Plane → Prop)
variable (perpendicular : Plane → Plane → Prop)
variable (line_parallel : Line → Plane → Prop)
variable (line_perpendicular : Line → Plane → Prop)
variable (line_in_plane : Line → Plane → Prop)

-- Theorem for proposition ①
theorem parallel_transitive (α β γ : Plane) :
  parallel α β → parallel α γ → parallel γ β := by sorry

-- Theorem for proposition ③
theorem perpendicular_from_line (m : Line) (α β : Plane) :
  line_perpendicular m α → line_parallel m β → perpendicular α β := by sorry

end NUMINAMATH_CALUDE_parallel_transitive_perpendicular_from_line_l3095_309550


namespace NUMINAMATH_CALUDE_jane_age_ratio_l3095_309535

/-- Represents the ages of Jane and her children at two different times -/
structure FamilyAges where
  J : ℝ  -- Jane's current age
  M : ℝ  -- Years ago
  younger_sum : ℝ  -- Sum of ages of two younger children
  oldest : ℝ  -- Age of oldest child

/-- The conditions given in the problem -/
def satisfies_conditions (ages : FamilyAges) : Prop :=
  ages.J > 0 ∧ 
  ages.M > 0 ∧
  ages.J = 2 * ages.younger_sum ∧
  ages.J = ages.oldest / 2 ∧
  ages.J - ages.M = 3 * (ages.younger_sum - 2 * ages.M) ∧
  ages.J - ages.M = ages.oldest - ages.M

theorem jane_age_ratio (ages : FamilyAges) 
  (h : satisfies_conditions ages) : ages.J / ages.M = 10 := by
  sorry

end NUMINAMATH_CALUDE_jane_age_ratio_l3095_309535


namespace NUMINAMATH_CALUDE_range_m_f_less_than_one_solution_sets_f_geq_mx_range_m_f_nonnegative_in_interval_l3095_309585

/-- The function f(x) defined in the problem -/
def f (m : ℝ) (x : ℝ) : ℝ := (m + 1) * x^2 - (m - 1) * x + m - 1

/-- Theorem for the range of m when f(x) < 1 for all x in ℝ -/
theorem range_m_f_less_than_one :
  ∀ m : ℝ, (∀ x : ℝ, f m x < 1) ↔ m < (1 - 2 * Real.sqrt 7) / 3 :=
sorry

/-- Theorem for the solution sets of f(x) ≥ (m+1)x -/
theorem solution_sets_f_geq_mx (m : ℝ) :
  (m = -1 ∧ {x : ℝ | x ≥ 1} = {x : ℝ | f m x ≥ (m + 1) * x}) ∨
  (m > -1 ∧ {x : ℝ | x ≤ (m - 1) / (m + 1) ∨ x ≥ 1} = {x : ℝ | f m x ≥ (m + 1) * x}) ∨
  (m < -1 ∧ {x : ℝ | 1 ≤ x ∧ x ≤ (m - 1) / (m + 1)} = {x : ℝ | f m x ≥ (m + 1) * x}) :=
sorry

/-- Theorem for the range of m when f(x) ≥ 0 for all x in [-1/2, 1/2] -/
theorem range_m_f_nonnegative_in_interval :
  ∀ m : ℝ, (∀ x : ℝ, x ∈ Set.Icc (-1/2) (1/2) → f m x ≥ 0) ↔ m ≥ 1 :=
sorry

end NUMINAMATH_CALUDE_range_m_f_less_than_one_solution_sets_f_geq_mx_range_m_f_nonnegative_in_interval_l3095_309585


namespace NUMINAMATH_CALUDE_geometric_mean_max_l3095_309546

theorem geometric_mean_max (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) 
  (h_arithmetic_mean : (a + b) / 2 = 4) : 
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ (x + y) / 2 = 4 ∧ 
  Real.sqrt (x * y) = 4 ∧ 
  ∀ (c d : ℝ), c > 0 → d > 0 → (c + d) / 2 = 4 → Real.sqrt (c * d) ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_geometric_mean_max_l3095_309546


namespace NUMINAMATH_CALUDE_function_translation_l3095_309549

-- Define a function type for f
def FunctionType := ℝ → ℝ

-- Define the translation vector
def TranslationVector := ℝ × ℝ

-- State the theorem
theorem function_translation
  (f : FunctionType)
  (a : TranslationVector) :
  (∀ x y : ℝ, y = f (2*x - 1) + 1 ↔ 
              y + a.2 = f (2*(x + a.1) - 1) + 1) →
  (∀ x y : ℝ, y = f (2*x + 1) - 1 ↔ 
              y = f (2*(x + a.1) + 1) - 1) →
  a = (1, -2) :=
sorry

end NUMINAMATH_CALUDE_function_translation_l3095_309549


namespace NUMINAMATH_CALUDE_cube_volume_7cm_l3095_309548

-- Define the edge length of the cube
def edge_length : ℝ := 7

-- Define the volume of a cube
def cube_volume (edge : ℝ) : ℝ := edge ^ 3

-- Theorem statement
theorem cube_volume_7cm :
  cube_volume edge_length = 343 := by sorry

end NUMINAMATH_CALUDE_cube_volume_7cm_l3095_309548


namespace NUMINAMATH_CALUDE_ashok_pyarelal_capital_ratio_l3095_309509

/-- Given a total loss and Pyarelal's loss, prove the ratio of Ashok's capital to Pyarelal's capital -/
theorem ashok_pyarelal_capital_ratio 
  (total_loss : ℕ) 
  (pyarelal_loss : ℕ) 
  (h1 : total_loss = 1200) 
  (h2 : pyarelal_loss = 1080) : 
  ∃ (a p : ℕ), a ≠ 0 ∧ p ≠ 0 ∧ a * 9 = p * 1 := by
  sorry

end NUMINAMATH_CALUDE_ashok_pyarelal_capital_ratio_l3095_309509


namespace NUMINAMATH_CALUDE_greater_solution_of_quadratic_l3095_309543

theorem greater_solution_of_quadratic (x : ℝ) : 
  x^2 + 20*x - 96 = 0 → x ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_greater_solution_of_quadratic_l3095_309543


namespace NUMINAMATH_CALUDE_library_book_distribution_l3095_309531

/-- Represents the number of books in a library -/
def total_books : ℕ := 6

/-- Calculates the number of ways to distribute books between library and checked-out status -/
def distribution_ways (n : ℕ) : ℕ :=
  if n ≥ 2 then n - 1 else 0

/-- Theorem stating that the number of ways to distribute the books is 5 -/
theorem library_book_distribution :
  distribution_ways total_books = 5 := by sorry

end NUMINAMATH_CALUDE_library_book_distribution_l3095_309531


namespace NUMINAMATH_CALUDE_number_multiplication_l3095_309500

theorem number_multiplication : ∃ n : ℝ, n * 40 = 173 * 240 ∧ n = 1038 := by
  sorry

end NUMINAMATH_CALUDE_number_multiplication_l3095_309500


namespace NUMINAMATH_CALUDE_sugar_solution_percentage_l3095_309502

theorem sugar_solution_percentage (original_percentage : ℝ) : 
  (3/4 : ℝ) * original_percentage + (1/4 : ℝ) * 28 = 16 → 
  original_percentage = 12 := by
sorry

end NUMINAMATH_CALUDE_sugar_solution_percentage_l3095_309502


namespace NUMINAMATH_CALUDE_division_remainder_l3095_309514

theorem division_remainder : 
  let dividend : ℕ := 220020
  let divisor : ℕ := 555 + 445
  let quotient : ℕ := 2 * (555 - 445)
  dividend % divisor = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_division_remainder_l3095_309514


namespace NUMINAMATH_CALUDE_dollar_composition_30_l3095_309595

/-- The dollar function as defined in the problem -/
noncomputable def dollar (N : ℝ) : ℝ := 0.75 * N + 2

/-- The statement to be proved -/
theorem dollar_composition_30 : dollar (dollar (dollar 30)) = 17.28125 := by
  sorry

end NUMINAMATH_CALUDE_dollar_composition_30_l3095_309595


namespace NUMINAMATH_CALUDE_logarithm_difference_equals_three_l3095_309593

theorem logarithm_difference_equals_three :
  (Real.log 320 / Real.log 4) / (Real.log 80 / Real.log 4) -
  (Real.log 640 / Real.log 4) / (Real.log 40 / Real.log 4) = 3 := by
  sorry

end NUMINAMATH_CALUDE_logarithm_difference_equals_three_l3095_309593


namespace NUMINAMATH_CALUDE_tamara_height_l3095_309551

/-- Given the heights of Tamara, Kim, and Gavin, prove Tamara's height is 95 inches -/
theorem tamara_height (kim : ℝ) : 
  let tamara := 3 * kim - 4
  let gavin := 2 * kim + 6
  (3 * kim - 4) + kim + (2 * kim + 6) = 200 → tamara = 95 := by
sorry

end NUMINAMATH_CALUDE_tamara_height_l3095_309551


namespace NUMINAMATH_CALUDE_inequality_proof_l3095_309542

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 1) :
  2 * (a + b + c) + 9 / ((a * b + b * c + c * a) ^ 2) ≥ 7 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3095_309542


namespace NUMINAMATH_CALUDE_equality_of_reciprocals_l3095_309516

theorem equality_of_reciprocals (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_eq : (3 : ℝ) ^ a = (4 : ℝ) ^ b ∧ (4 : ℝ) ^ b = (6 : ℝ) ^ c) : 
  2 / c = 2 / a + 1 / b :=
sorry

end NUMINAMATH_CALUDE_equality_of_reciprocals_l3095_309516


namespace NUMINAMATH_CALUDE_arithmetic_sequence_mean_median_l3095_309588

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) :=
  ∀ n, a (n + 1) = a n + d

def geometric_sequence (a b c : ℝ) :=
  b * b = a * c

theorem arithmetic_sequence_mean_median
  (a : ℕ → ℝ)
  (d : ℝ)
  (h_arith : arithmetic_sequence a d)
  (h_d_nonzero : d ≠ 0)
  (h_a3 : a 3 = 8)
  (h_geom : geometric_sequence (a 1) (a 3) (a 7)) :
  let mean := (a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 + a 10) / 10
  let median := (a 5 + a 6) / 2
  mean = 13 ∧ median = 13 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_mean_median_l3095_309588


namespace NUMINAMATH_CALUDE_range_of_a_l3095_309590

-- Define the function f
def f (a x : ℝ) : ℝ := x^2 - 2*a*x + 1

-- State the theorem
theorem range_of_a (a : ℝ) :
  (∀ x ∈ Set.Ici (-1 : ℝ), f a x ≥ a) →
  -2 ≤ a ∧ a ≤ (-1 + Real.sqrt 5) / 2 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l3095_309590


namespace NUMINAMATH_CALUDE_distance_between_sasha_and_kolya_l3095_309525

-- Define the race distance
def race_distance : ℝ := 100

-- Define the runners' speeds
variable (v_S v_L v_K : ℝ)

-- Define the conditions
axiom positive_speeds : 0 < v_S ∧ 0 < v_L ∧ 0 < v_K
axiom lyosha_behind_sasha : v_L / v_S = 0.9
axiom kolya_behind_lyosha : v_K / v_L = 0.9

-- Define the theorem
theorem distance_between_sasha_and_kolya :
  let t_S := race_distance / v_S
  let d_K := v_K * t_S
  race_distance - d_K = 19 := by sorry

end NUMINAMATH_CALUDE_distance_between_sasha_and_kolya_l3095_309525


namespace NUMINAMATH_CALUDE_positive_integer_solutions_for_mn_equation_l3095_309563

theorem positive_integer_solutions_for_mn_equation :
  ∀ m n : ℕ+,
  m^(n : ℕ) = n^((m : ℕ) - (n : ℕ)) →
  ((m = 9 ∧ n = 3) ∨ (m = 8 ∧ n = 2)) :=
by sorry

end NUMINAMATH_CALUDE_positive_integer_solutions_for_mn_equation_l3095_309563


namespace NUMINAMATH_CALUDE_expression_evaluation_l3095_309561

theorem expression_evaluation : 
  60 + (105 / 15) + (25 * 16) - 250 + (324 / 9)^2 = 1513 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3095_309561


namespace NUMINAMATH_CALUDE_lindseys_money_is_36_l3095_309518

/-- Calculates the remaining money for Lindsey given her savings and spending. -/
def lindseys_remaining_money (sept_savings oct_savings nov_savings mom_bonus_threshold mom_bonus video_game_cost : ℕ) : ℕ :=
  let total_savings := sept_savings + oct_savings + nov_savings
  let with_bonus := total_savings + if total_savings > mom_bonus_threshold then mom_bonus else 0
  with_bonus - video_game_cost

/-- Proves that Lindsey's remaining money is $36 given her savings and spending. -/
theorem lindseys_money_is_36 :
  lindseys_remaining_money 50 37 11 75 25 87 = 36 := by
  sorry

#eval lindseys_remaining_money 50 37 11 75 25 87

end NUMINAMATH_CALUDE_lindseys_money_is_36_l3095_309518


namespace NUMINAMATH_CALUDE_christines_stickers_l3095_309508

theorem christines_stickers (total_needed : ℕ) (more_needed : ℕ) (h1 : total_needed = 30) (h2 : more_needed = 19) :
  total_needed - more_needed = 11 := by
  sorry

end NUMINAMATH_CALUDE_christines_stickers_l3095_309508


namespace NUMINAMATH_CALUDE_circle_and_line_intersection_l3095_309529

-- Define the circle C
def circle_C (x y : ℝ) (a : ℝ) : Prop :=
  x^2 + y^2 + 4*x - 2*y + a = 0

-- Define the line l
def line_l (x y : ℝ) : Prop :=
  x - y - 3 = 0

-- Define the line m
def line_m (x y : ℝ) : Prop :=
  x + y + 1 = 0

-- Define the origin O
def origin : ℝ × ℝ := (0, 0)

-- Define perpendicularity of vectors
def perpendicular (v1 v2 : ℝ × ℝ) : Prop :=
  v1.1 * v2.1 + v1.2 * v2.2 = 0

theorem circle_and_line_intersection (a : ℝ) :
  (∃ (x y : ℝ), circle_C x y a ∧ line_l x y) →
  (∃ (x₁ y₁ x₂ y₂ : ℝ),
    circle_C x₁ y₁ a ∧ line_l x₁ y₁ ∧
    circle_C x₂ y₂ a ∧ line_l x₂ y₂ ∧
    perpendicular (x₁, y₁) (x₂, y₂)) →
  (∀ (x y : ℝ), line_m x y ↔ (x = -2 ∧ y = 1) ∨ (x + y + 1 = 0)) ∧
  a = -18 := by sorry

end NUMINAMATH_CALUDE_circle_and_line_intersection_l3095_309529


namespace NUMINAMATH_CALUDE_pencil_distribution_l3095_309570

def colored_pencils : ℕ := 14
def black_pencils : ℕ := 35
def siblings : ℕ := 3
def kept_pencils : ℕ := 10

theorem pencil_distribution :
  (colored_pencils + black_pencils - kept_pencils) / siblings = 13 :=
by sorry

end NUMINAMATH_CALUDE_pencil_distribution_l3095_309570


namespace NUMINAMATH_CALUDE_inequality_proof_l3095_309534

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h_condition : a * b * c * (a + b + c) = a * b + b * c + c * a) :
  5 * (a + b + c) ≥ 7 + 8 * a * b * c := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3095_309534


namespace NUMINAMATH_CALUDE_count_satisfying_numbers_is_45_l3095_309575

/-- A function that checks if a three-digit number satisfies the condition -/
def satisfiesCondition (n : Nat) : Bool :=
  let a := n / 100
  let b := (n / 10) % 10
  let c := n % 10
  b = a + c ∧ 100 ≤ n ∧ n ≤ 999

/-- The count of three-digit numbers satisfying the condition -/
def countSatisfyingNumbers : Nat :=
  (List.range 900).map (· + 100)
    |>.filter satisfiesCondition
    |>.length

/-- Theorem stating that the count of satisfying numbers is 45 -/
theorem count_satisfying_numbers_is_45 : countSatisfyingNumbers = 45 := by
  sorry

end NUMINAMATH_CALUDE_count_satisfying_numbers_is_45_l3095_309575


namespace NUMINAMATH_CALUDE_penguins_to_feed_l3095_309557

theorem penguins_to_feed (total_penguins : ℕ) (fed_penguins : ℕ) 
  (h1 : total_penguins = 36) 
  (h2 : fed_penguins = 19) : 
  total_penguins - fed_penguins = 17 := by
  sorry

end NUMINAMATH_CALUDE_penguins_to_feed_l3095_309557


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l3095_309587

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n

theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) (h_geometric : is_geometric_sequence a) 
  (h_condition : 8 * a 2 + a 5 = 0) :
  ∃ q : ℝ, (∀ n : ℕ, a (n + 1) = q * a n) ∧ q = -2 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l3095_309587


namespace NUMINAMATH_CALUDE_pqr_sum_fraction_prime_l3095_309539

theorem pqr_sum_fraction_prime (p q r : ℕ) : 
  Nat.Prime p → Nat.Prime q → Nat.Prime r → 
  (∃ k : ℕ, p * q * r = k * (p + q + r)) → 
  Nat.Prime (p * q * r / (p + q + r)) :=
by sorry

end NUMINAMATH_CALUDE_pqr_sum_fraction_prime_l3095_309539


namespace NUMINAMATH_CALUDE_log_greater_than_reciprocal_l3095_309560

theorem log_greater_than_reciprocal (x : ℝ) (h : x > 0) : Real.log (1 + x) > 1 / (x + 1) := by
  sorry

end NUMINAMATH_CALUDE_log_greater_than_reciprocal_l3095_309560


namespace NUMINAMATH_CALUDE_sum_of_squares_first_50_even_integers_l3095_309537

theorem sum_of_squares_first_50_even_integers :
  (Finset.range 50).sum (fun i => (2 * (i + 1))^2) = 171700 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_first_50_even_integers_l3095_309537


namespace NUMINAMATH_CALUDE_original_price_of_discounted_shoes_l3095_309555

/-- Given a pair of shoes sold at a 20% discount for $480, prove that its original price was $600. -/
theorem original_price_of_discounted_shoes (discount_rate : ℝ) (discounted_price : ℝ) : 
  discount_rate = 0.20 → discounted_price = 480 → (1 - discount_rate) * 600 = discounted_price := by
  sorry

end NUMINAMATH_CALUDE_original_price_of_discounted_shoes_l3095_309555


namespace NUMINAMATH_CALUDE_union_condition_equiv_range_l3095_309512

def A (a : ℝ) : Set ℝ := {x | a - 1 < x ∧ x < 2 * a + 1}
def B : Set ℝ := {x | 0 < x ∧ x < 5}

theorem union_condition_equiv_range (a : ℝ) :
  A a ∪ B = B ↔ a ≤ -2 ∨ (1 ≤ a ∧ a ≤ 2) := by
  sorry

end NUMINAMATH_CALUDE_union_condition_equiv_range_l3095_309512


namespace NUMINAMATH_CALUDE_more_girls_than_boys_l3095_309521

/-- The number of girls in the school -/
def num_girls : ℕ := 739

/-- The number of boys in the school -/
def num_boys : ℕ := 337

/-- The difference between the number of girls and boys -/
def difference : ℕ := num_girls - num_boys

theorem more_girls_than_boys : difference = 402 := by
  sorry

end NUMINAMATH_CALUDE_more_girls_than_boys_l3095_309521


namespace NUMINAMATH_CALUDE_toys_needed_l3095_309594

theorem toys_needed (available : ℕ) (people : ℕ) (per_person : ℕ) : 
  available = 68 → people = 14 → per_person = 5 → 
  (people * per_person - available : ℕ) = 2 := by
  sorry

end NUMINAMATH_CALUDE_toys_needed_l3095_309594


namespace NUMINAMATH_CALUDE_square_field_side_length_l3095_309530

theorem square_field_side_length (area : Real) (side_length : Real) :
  area = 196 ∧ area = side_length ^ 2 → side_length = 14 := by
  sorry

end NUMINAMATH_CALUDE_square_field_side_length_l3095_309530


namespace NUMINAMATH_CALUDE_smallest_three_digit_multiple_of_13_l3095_309507

theorem smallest_three_digit_multiple_of_13 : 
  ∀ n : ℕ, n ≥ 100 ∧ n < 1000 ∧ 13 ∣ n → n ≥ 104 :=
by sorry

end NUMINAMATH_CALUDE_smallest_three_digit_multiple_of_13_l3095_309507


namespace NUMINAMATH_CALUDE_angle_sum_is_pi_over_two_l3095_309517

theorem angle_sum_is_pi_over_two (a b : Real) : 
  0 < a ∧ a < π/2 →
  0 < b ∧ b < π/2 →
  5 * (Real.sin a)^2 + 3 * (Real.sin b)^2 = 2 →
  4 * Real.sin (2*a) + 3 * Real.sin (2*b) = 3 →
  2*a + b = π/2 := by
  sorry

end NUMINAMATH_CALUDE_angle_sum_is_pi_over_two_l3095_309517


namespace NUMINAMATH_CALUDE_quilt_material_requirement_l3095_309567

/-- Given that 7 quilts can be made with 21 yards of material,
    prove that 12 quilts require 36 yards of material. -/
theorem quilt_material_requirement : 
  (7 : ℚ) * (36 : ℚ) = (12 : ℚ) * (21 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_quilt_material_requirement_l3095_309567


namespace NUMINAMATH_CALUDE_increments_theorem_l3095_309591

/-- The function z(x, y) = xy -/
def z (x y : ℝ) : ℝ := x * y

/-- The initial point M₀ -/
def M₀ : ℝ × ℝ := (1, 2)

/-- The point M₁ -/
def M₁ : ℝ × ℝ := (1.1, 2)

/-- The point M₂ -/
def M₂ : ℝ × ℝ := (1, 1.9)

/-- The point M₃ -/
def M₃ : ℝ × ℝ := (1.1, 2.2)

/-- The increment of z from M₀ to another point -/
def increment (M : ℝ × ℝ) : ℝ := z M.1 M.2 - z M₀.1 M₀.2

theorem increments_theorem :
  increment M₁ = 0.2 ∧ increment M₂ = -0.1 ∧ increment M₃ = 0.42 := by
  sorry

end NUMINAMATH_CALUDE_increments_theorem_l3095_309591


namespace NUMINAMATH_CALUDE_nail_decoration_time_l3095_309592

def base_coat_time : ℕ := 20
def paint_coat_time : ℕ := 20
def glitter_coat_time : ℕ := 20
def drying_time : ℕ := 20
def pattern_time : ℕ := 40

def total_decoration_time : ℕ :=
  base_coat_time + drying_time +
  paint_coat_time + drying_time +
  glitter_coat_time + drying_time +
  pattern_time

theorem nail_decoration_time :
  total_decoration_time = 160 :=
by sorry

end NUMINAMATH_CALUDE_nail_decoration_time_l3095_309592


namespace NUMINAMATH_CALUDE_cube_surface_area_l3095_309538

theorem cube_surface_area (volume : ℝ) (h : volume = 1331) :
  let side := (volume ^ (1/3 : ℝ))
  6 * side^2 = 726 := by
sorry

end NUMINAMATH_CALUDE_cube_surface_area_l3095_309538


namespace NUMINAMATH_CALUDE_inequality_solution_l3095_309528

-- Define the inequality function
def f (a x : ℝ) : ℝ := x^2 - a*x + a - 1

-- Define the solution set for a > 2
def solution_set_gt2 (a : ℝ) : Set ℝ := 
  {x | x < 1 ∨ x > a - 1}

-- Define the solution set for a = 2
def solution_set_eq2 : Set ℝ := 
  {x | x < 1 ∨ x > 1}

-- Define the solution set for a < 2
def solution_set_lt2 (a : ℝ) : Set ℝ := 
  {x | x < a - 1 ∨ x > 1}

-- Theorem statement
theorem inequality_solution (a : ℝ) :
  (∀ x, f a x > 0 ↔ 
    (a > 2 ∧ x ∈ solution_set_gt2 a) ∨
    (a = 2 ∧ x ∈ solution_set_eq2) ∨
    (a < 2 ∧ x ∈ solution_set_lt2 a)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l3095_309528


namespace NUMINAMATH_CALUDE_intersection_sum_reciprocal_constant_l3095_309558

/-- The curve C representing the locus of centers of the moving circle M -/
def curve_C (x y : ℝ) : Prop :=
  x > 0 ∧ x^2 / 4 - y^2 / 12 = 1

/-- A point on the curve C -/
structure PointOnC where
  x : ℝ
  y : ℝ
  on_curve : curve_C x y

/-- The origin point O -/
def O : ℝ × ℝ := (0, 0)

/-- Distance squared between two points -/
def dist_squared (p q : ℝ × ℝ) : ℝ :=
  (p.1 - q.1)^2 + (p.2 - q.2)^2

theorem intersection_sum_reciprocal_constant
  (P Q : PointOnC)
  (h_perp : (P.x * Q.x + P.y * Q.y = 0)) : -- OP ⊥ OQ condition
  1 / dist_squared O (P.x, P.y) + 1 / dist_squared O (Q.x, Q.y) = 1/6 :=
sorry

end NUMINAMATH_CALUDE_intersection_sum_reciprocal_constant_l3095_309558


namespace NUMINAMATH_CALUDE_fruit_arrangement_count_l3095_309526

/-- The number of ways to arrange fruits with constraints -/
def fruitArrangements (apples oranges bananas : ℕ) : ℕ :=
  (Nat.factorial (apples + oranges + bananas)) / 
  (Nat.factorial apples * Nat.factorial oranges * Nat.factorial bananas) * 
  (Nat.choose (apples + bananas) apples)

/-- Theorem stating the number of fruit arrangements -/
theorem fruit_arrangement_count :
  fruitArrangements 4 2 2 = 18900 := by
  sorry

end NUMINAMATH_CALUDE_fruit_arrangement_count_l3095_309526


namespace NUMINAMATH_CALUDE_hyperbola_foci_distance_l3095_309510

/-- The distance between the foci of a hyperbola defined by x^2 - y^2 = 1 is 2√2 -/
theorem hyperbola_foci_distance :
  ∃ (f₁ f₂ : ℝ × ℝ), 
    (∀ (x y : ℝ), x^2 - y^2 = 1 → (x - f₁.1)^2 + (y - f₁.2)^2 = (x - f₂.1)^2 + (y - f₂.2)^2) ∧
    dist f₁ f₂ = 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_foci_distance_l3095_309510


namespace NUMINAMATH_CALUDE_pasta_preference_ratio_l3095_309562

theorem pasta_preference_ratio (total_students : ℕ) (ravioli_preference : ℕ) (tortellini_preference : ℕ)
  (h1 : total_students = 800)
  (h2 : ravioli_preference = 300)
  (h3 : tortellini_preference = 150) :
  (ravioli_preference : ℚ) / tortellini_preference = 2 := by
  sorry

end NUMINAMATH_CALUDE_pasta_preference_ratio_l3095_309562


namespace NUMINAMATH_CALUDE_square_difference_l3095_309554

theorem square_difference (x y : ℝ) 
  (h1 : x + y = 12) 
  (h2 : 3 * x + y = 18) : 
  x^2 - y^2 = -72 := by
sorry

end NUMINAMATH_CALUDE_square_difference_l3095_309554


namespace NUMINAMATH_CALUDE_remainder_theorem_l3095_309540

-- Define the polynomial Q(x)
variable (Q : ℝ → ℝ)

-- Define the conditions
axiom Q_remainder_15 : ∃ P : ℝ → ℝ, ∀ x, Q x = P x * (x - 15) + 10
axiom Q_remainder_12 : ∃ P : ℝ → ℝ, ∀ x, Q x = P x * (x - 12) + 2

-- Theorem statement
theorem remainder_theorem :
  ∃ R : ℝ → ℝ, ∀ x, Q x = R x * ((x - 12) * (x - 15)) + (8/3 * x - 30) :=
sorry

end NUMINAMATH_CALUDE_remainder_theorem_l3095_309540


namespace NUMINAMATH_CALUDE_range_of_a_l3095_309564

theorem range_of_a (a : ℝ) : 
  (∃ x₁ x₂ x₃ : ℤ, 
    (∀ x : ℝ, (x > 2*a - 3 ∧ 2*x ≥ 3*(x-2) + 5) ↔ (x = x₁ ∨ x = x₂ ∨ x = x₃)) ∧
    (∀ x : ℤ, x ≠ x₁ ∧ x ≠ x₂ ∧ x ≠ x₃ → ¬(x > 2*a - 3 ∧ 2*x ≥ 3*(x-2) + 5))) →
  (1/2 : ℝ) ≤ a ∧ a < 1 := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l3095_309564


namespace NUMINAMATH_CALUDE_negation_equivalence_l3095_309519

-- Define the predicate P
def P (k : ℝ) : Prop := ∃ x y : ℝ, y = k * x + 1 ∧ x^2 + y^2 = 2

-- State the theorem
theorem negation_equivalence :
  (¬ ∀ k : ℝ, P k) ↔ (∃ k₀ : ℝ, ¬ P k₀) :=
sorry

end NUMINAMATH_CALUDE_negation_equivalence_l3095_309519


namespace NUMINAMATH_CALUDE_third_term_value_l3095_309582

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem third_term_value
  (a : ℕ → ℤ)
  (h_arith : arithmetic_sequence a)
  (h_first : a 1 = -11)
  (h_sum : a 4 + a 6 = -6) :
  a 3 = -7 :=
sorry

end NUMINAMATH_CALUDE_third_term_value_l3095_309582


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_one_l3095_309520

/-- A geometric sequence with negative terms and a specific sum condition has a common ratio of 1. -/
theorem geometric_sequence_common_ratio_one 
  (a : ℕ+ → ℝ) 
  (h_geometric : ∀ n : ℕ+, a (n + 1) = a n * q) 
  (h_negative : ∀ n : ℕ+, a n < 0) 
  (h_sum : a 3 + a 7 ≥ 2 * a 5) : 
  q = 1 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_one_l3095_309520


namespace NUMINAMATH_CALUDE_andrea_rhinestone_ratio_l3095_309506

/-- Proves that the ratio of rhinestones Andrea bought to the total rhinestones needed is 1:3 -/
theorem andrea_rhinestone_ratio :
  let total_needed : ℕ := 45
  let found_in_supplies : ℕ := total_needed / 5
  let still_needed : ℕ := 21
  let bought : ℕ := total_needed - found_in_supplies - still_needed
  (bought : ℚ) / total_needed = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_andrea_rhinestone_ratio_l3095_309506


namespace NUMINAMATH_CALUDE_plant_branches_l3095_309584

theorem plant_branches (x : ℕ) 
  (h1 : x > 0)
  (h2 : 1 + x + x * x = 31) : x = 5 := by
  sorry

end NUMINAMATH_CALUDE_plant_branches_l3095_309584


namespace NUMINAMATH_CALUDE_factory_temporary_stats_l3095_309504

/-- Represents the different employee categories in the factory -/
inductive EmployeeCategory
  | Technician
  | SkilledLaborer
  | Manager
  | Administrative

/-- Represents the employment status of an employee -/
inductive EmploymentStatus
  | Permanent
  | Temporary

/-- Structure to hold information about each employee category -/
structure CategoryInfo where
  category : EmployeeCategory
  percentage : Float
  permanentPercentage : Float
  weeklyHours : Nat

def factory : List CategoryInfo := [
  { category := EmployeeCategory.Technician, percentage := 0.4, permanentPercentage := 0.6, weeklyHours := 45 },
  { category := EmployeeCategory.SkilledLaborer, percentage := 0.3, permanentPercentage := 0.5, weeklyHours := 40 },
  { category := EmployeeCategory.Manager, percentage := 0.2, permanentPercentage := 0.8, weeklyHours := 50 },
  { category := EmployeeCategory.Administrative, percentage := 0.1, permanentPercentage := 0.9, weeklyHours := 35 }
]

def totalEmployees : Nat := 100

/-- Calculate the percentage of temporary employees -/
def calculateTemporaryPercentage (factoryInfo : List CategoryInfo) : Float :=
  factoryInfo.foldl (fun acc info => 
    acc + info.percentage * (1 - info.permanentPercentage)) 0

/-- Calculate the total weekly hours worked by temporary employees -/
def calculateTemporaryHours (factoryInfo : List CategoryInfo) (totalEmp : Nat) : Float :=
  factoryInfo.foldl (fun acc info => 
    acc + (info.percentage * totalEmp.toFloat * (1 - info.permanentPercentage) * info.weeklyHours.toFloat)) 0

theorem factory_temporary_stats :
  calculateTemporaryPercentage factory = 0.36 ∧ 
  calculateTemporaryHours factory totalEmployees = 1555 := by
  sorry


end NUMINAMATH_CALUDE_factory_temporary_stats_l3095_309504


namespace NUMINAMATH_CALUDE_sub_committee_count_l3095_309513

/-- The number of people in the committee -/
def totalPeople : ℕ := 8

/-- The size of each sub-committee -/
def subCommitteeSize : ℕ := 2

/-- The number of people who cannot be in the same sub-committee -/
def restrictedPair : ℕ := 1

/-- The number of valid two-person sub-committees -/
def validSubCommittees : ℕ := 27

theorem sub_committee_count :
  (Nat.choose totalPeople subCommitteeSize) - restrictedPair = validSubCommittees :=
sorry

end NUMINAMATH_CALUDE_sub_committee_count_l3095_309513


namespace NUMINAMATH_CALUDE_square_sum_of_integers_l3095_309571

theorem square_sum_of_integers (x y : ℕ+) 
  (h1 : x * y + x + y = 117)
  (h2 : x^2 * y + x * y^2 = 1512) : 
  x^2 + y^2 = 549 := by
sorry

end NUMINAMATH_CALUDE_square_sum_of_integers_l3095_309571


namespace NUMINAMATH_CALUDE_euro_problem_l3095_309524

-- Define the € operation
def euro (x y : ℝ) : ℝ := 2 * x * y

-- State the theorem
theorem euro_problem (n : ℝ) :
  euro 8 (euro 4 n) = 640 → n = 5 := by
  sorry

end NUMINAMATH_CALUDE_euro_problem_l3095_309524


namespace NUMINAMATH_CALUDE_max_2x2_squares_5x7_grid_l3095_309547

/-- Represents the dimensions of the grid -/
structure GridDimensions where
  rows : ℕ
  cols : ℕ

/-- Represents the different types of pieces that can be cut from the grid -/
inductive PieceType
  | Square2x2
  | LShape
  | Strip1x3

/-- Represents a configuration of pieces cut from the grid -/
structure Configuration where
  square2x2Count : ℕ
  lShapeCount : ℕ
  strip1x3Count : ℕ

/-- Checks if a configuration is valid for the given grid dimensions -/
def isValidConfiguration (grid : GridDimensions) (config : Configuration) : Prop :=
  4 * config.square2x2Count + 3 * config.lShapeCount + 3 * config.strip1x3Count = grid.rows * grid.cols

/-- Theorem: The maximum number of 2x2 squares in a valid configuration for a 5x7 grid is 5 -/
theorem max_2x2_squares_5x7_grid :
  ∃ (maxSquares : ℕ),
    maxSquares = 5 ∧
    (∃ (config : Configuration),
      isValidConfiguration ⟨5, 7⟩ config ∧
      config.square2x2Count = maxSquares) ∧
    (∀ (config : Configuration),
      isValidConfiguration ⟨5, 7⟩ config →
      config.square2x2Count ≤ maxSquares) :=
by
  sorry

end NUMINAMATH_CALUDE_max_2x2_squares_5x7_grid_l3095_309547


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l3095_309576

theorem complex_modulus_problem (z : ℂ) : 
  z = ((1 - I) * (2 - I)) / (1 + 2*I) → Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l3095_309576


namespace NUMINAMATH_CALUDE_min_value_problem_l3095_309527

theorem min_value_problem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 4 * x + y = 1) :
  (∀ x' y' : ℝ, x' > 0 → y' > 0 → 4 * x' + y' = 1 → 1 / x' + 4 / y' ≥ 1 / x + 4 / y) →
  1 / x + 4 / y = 16 :=
by sorry

end NUMINAMATH_CALUDE_min_value_problem_l3095_309527


namespace NUMINAMATH_CALUDE_no_five_linked_country_with_46_airlines_l3095_309544

theorem no_five_linked_country_with_46_airlines :
  ¬ ∃ (n : ℕ), n > 0 ∧ (5 * n) / 2 = 46 := by
  sorry

end NUMINAMATH_CALUDE_no_five_linked_country_with_46_airlines_l3095_309544


namespace NUMINAMATH_CALUDE_shooting_competition_probabilities_l3095_309522

/-- Probability of a single shooter hitting the target -/
structure Shooter where
  prob_hit : ℚ
  prob_hit_nonneg : 0 ≤ prob_hit
  prob_hit_le_one : prob_hit ≤ 1

/-- A team of two shooters -/
structure ShootingTeam where
  shooter_a : Shooter
  shooter_b : Shooter

/-- Probability of exactly one hit in a single round -/
def prob_one_hit (team : ShootingTeam) : ℚ :=
  team.shooter_a.prob_hit * (1 - team.shooter_b.prob_hit) +
  (1 - team.shooter_a.prob_hit) * team.shooter_b.prob_hit

/-- Probability of exactly three hits in three rounds -/
def prob_three_hits_three_rounds (team : ShootingTeam) : ℚ :=
  (1 - team.shooter_a.prob_hit)^3 * team.shooter_b.prob_hit^3 +
  3 * team.shooter_a.prob_hit * (1 - team.shooter_a.prob_hit)^2 * team.shooter_b.prob_hit^2 * (1 - team.shooter_b.prob_hit) +
  3 * team.shooter_a.prob_hit^2 * (1 - team.shooter_a.prob_hit) * team.shooter_b.prob_hit * (1 - team.shooter_b.prob_hit)^2 +
  team.shooter_a.prob_hit^3 * (1 - team.shooter_b.prob_hit)^3

theorem shooting_competition_probabilities
  (team : ShootingTeam)
  (h_a : team.shooter_a.prob_hit = 1/2)
  (h_b : team.shooter_b.prob_hit = 2/3) :
  prob_one_hit team = 1/2 ∧ prob_three_hits_three_rounds team = 7/24 := by
  sorry

end NUMINAMATH_CALUDE_shooting_competition_probabilities_l3095_309522


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3095_309573

theorem quadratic_inequality_solution_set 
  (a b c : ℝ) 
  (h1 : a < 0) 
  (h2 : Set.Ioo (-2 : ℝ) 3 = {x | a * x^2 + b * x + c > 0}) : 
  Set.Ioo (-(1/2) : ℝ) (1/3) = {x | c * x^2 + b * x + a < 0} := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3095_309573


namespace NUMINAMATH_CALUDE_geometry_problem_l3095_309511

-- Define the points
def A : ℝ × ℝ := (1, 3)
def B : ℝ × ℝ := (3, 1)
def C : ℝ × ℝ := (-1, 0)

-- Define the line equation
def line_AB (x y : ℝ) : Prop := x + y - 4 = 0

-- Define the circle equation
def circle_C (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 25/2

theorem geometry_problem :
  -- The equation of line AB
  (∀ x y : ℝ, (x - A.1) * (B.2 - A.2) = (y - A.2) * (B.1 - A.1) ↔ line_AB x y) ∧
  -- The circle with center C is tangent to line AB
  (∃ x y : ℝ, line_AB x y ∧ circle_C x y ∧
    ∀ x' y' : ℝ, line_AB x' y' → ((x' - C.1)^2 + (y' - C.2)^2 ≥ 25/2)) :=
by sorry

end NUMINAMATH_CALUDE_geometry_problem_l3095_309511


namespace NUMINAMATH_CALUDE_number_solution_l3095_309572

theorem number_solution : ∃ x : ℚ, x + (3/5) * x = 240 ∧ x = 150 := by sorry

end NUMINAMATH_CALUDE_number_solution_l3095_309572


namespace NUMINAMATH_CALUDE_eight_N_plus_nine_is_perfect_square_l3095_309577

theorem eight_N_plus_nine_is_perfect_square (n : ℕ) : 
  let N := 2^(4*n + 1) - 4^n - 1
  (∃ k : ℤ, N = 9 * k) → 
  ∃ m : ℕ, 8 * N + 9 = m^2 := by
sorry

end NUMINAMATH_CALUDE_eight_N_plus_nine_is_perfect_square_l3095_309577


namespace NUMINAMATH_CALUDE_more_wins_probability_correct_l3095_309597

/-- The number of matches played by the team -/
def num_matches : ℕ := 5

/-- The probability of winning, losing, or tying a single match -/
def match_probability : ℚ := 1/3

/-- The probability of ending with more wins than losses -/
def more_wins_probability : ℚ := 16/243

theorem more_wins_probability_correct :
  (∀ (outcome : Fin num_matches → Fin 3),
    (∃ (wins losses : ℕ),
      wins > losses ∧
      wins + losses ≤ num_matches ∧
      (∀ i, outcome i = 0 → wins > 0) ∧
      (∀ i, outcome i = 1 → losses > 0))) →
  ∃ (favorable_outcomes : ℕ),
    favorable_outcomes = 16 ∧
    (favorable_outcomes : ℚ) / (3 ^ num_matches) = more_wins_probability :=
sorry

end NUMINAMATH_CALUDE_more_wins_probability_correct_l3095_309597


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l3095_309579

/-- The speed of a boat in still water, given its downstream and upstream distances in one hour -/
theorem boat_speed_in_still_water (downstream upstream : ℝ) 
  (h_downstream : downstream = 11) 
  (h_upstream : upstream = 5) : 
  (downstream + upstream) / 2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l3095_309579
