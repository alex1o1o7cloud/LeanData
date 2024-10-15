import Mathlib

namespace NUMINAMATH_CALUDE_complex_power_approximation_l1879_187903

/-- Prove that (3 * cos(30°) + 3i * sin(30°))^8 is approximately equal to -3281 - 3281i * √3 -/
theorem complex_power_approximation :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 1 ∧ 
  Complex.abs ((3 * Complex.cos (30 * π / 180) + 3 * Complex.I * Complex.sin (30 * π / 180))^8 - 
               (-3281 - 3281 * Complex.I * Real.sqrt 3)) < ε :=
by sorry

end NUMINAMATH_CALUDE_complex_power_approximation_l1879_187903


namespace NUMINAMATH_CALUDE_silver_division_representation_l1879_187911

/-- Represents the problem of dividing silver among guests -/
structure SilverDivision where
  guests : ℕ      -- number of guests
  silver : ℕ      -- total amount of silver in taels

/-- The conditions of the silver division problem are satisfied -/
def satisfiesConditions (sd : SilverDivision) : Prop :=
  (7 * sd.guests = sd.silver - 4) ∧ 
  (9 * sd.guests = sd.silver + 8)

/-- The system of equations correctly represents the silver division problem -/
theorem silver_division_representation (sd : SilverDivision) : 
  satisfiesConditions sd ↔ 
  (∃ x y : ℕ, 
    sd.guests = x ∧ 
    sd.silver = y ∧ 
    7 * x = y - 4 ∧ 
    9 * x = y + 8) :=
sorry

end NUMINAMATH_CALUDE_silver_division_representation_l1879_187911


namespace NUMINAMATH_CALUDE_balanced_split_theorem_l1879_187929

/-- A finite collection of positive real numbers is balanced if each number
    is less than the sum of the others. -/
def IsBalanced (s : Finset ℝ) : Prop :=
  ∀ x ∈ s, x < (s.sum id - x)

/-- A finite collection of positive real numbers can be split into three parts
    with the property that the sum of the numbers in each part is less than
    the sum of the numbers in the two other parts. -/
def CanSplitIntoThreeParts (s : Finset ℝ) : Prop :=
  ∃ (a b c : Finset ℝ), a ∪ b ∪ c = s ∧ a ∩ b = ∅ ∧ b ∩ c = ∅ ∧ a ∩ c = ∅ ∧
    a.sum id < b.sum id + c.sum id ∧
    b.sum id < a.sum id + c.sum id ∧
    c.sum id < a.sum id + b.sum id

/-- The main theorem -/
theorem balanced_split_theorem (m : ℕ) (hm : m ≥ 3) :
  (∀ (s : Finset ℝ), s.card = m → IsBalanced s → CanSplitIntoThreeParts s) ↔ m ≠ 4 :=
sorry

end NUMINAMATH_CALUDE_balanced_split_theorem_l1879_187929


namespace NUMINAMATH_CALUDE_washing_machines_removed_l1879_187959

/-- Represents a shipping container with crates, boxes, and washing machines -/
structure ShippingContainer where
  num_crates : ℕ
  boxes_per_crate : ℕ
  initial_machines_per_box : ℕ
  machines_removed_per_box : ℕ

/-- Calculates the total number of washing machines removed from a shipping container -/
def total_machines_removed (container : ShippingContainer) : ℕ :=
  container.num_crates * container.boxes_per_crate * container.machines_removed_per_box

/-- Theorem stating the number of washing machines removed from the specific shipping container -/
theorem washing_machines_removed : 
  let container : ShippingContainer := {
    num_crates := 10,
    boxes_per_crate := 6,
    initial_machines_per_box := 4,
    machines_removed_per_box := 1
  }
  total_machines_removed container = 60 := by
  sorry


end NUMINAMATH_CALUDE_washing_machines_removed_l1879_187959


namespace NUMINAMATH_CALUDE_choose_3_from_13_l1879_187963

theorem choose_3_from_13 : Nat.choose 13 3 = 286 := by sorry

end NUMINAMATH_CALUDE_choose_3_from_13_l1879_187963


namespace NUMINAMATH_CALUDE_trig_identity_l1879_187928

theorem trig_identity (x y z : ℝ) : 
  Real.sin (x - y + z) * Real.cos y - Real.cos (x - y + z) * Real.sin y = Real.sin x := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l1879_187928


namespace NUMINAMATH_CALUDE_smallest_special_number_proof_l1879_187912

/-- A function that returns true if a natural number uses exactly four different digits -/
def uses_four_different_digits (n : ℕ) : Prop :=
  (Finset.card (Finset.image (λ d => d % 10) (Finset.range 4))) = 4

/-- The smallest natural number greater than 3429 that uses exactly four different digits -/
def smallest_special_number : ℕ := 3450

theorem smallest_special_number_proof :
  smallest_special_number > 3429 ∧
  uses_four_different_digits smallest_special_number ∧
  ∀ n : ℕ, n > 3429 ∧ n < smallest_special_number → ¬(uses_four_different_digits n) :=
sorry

end NUMINAMATH_CALUDE_smallest_special_number_proof_l1879_187912


namespace NUMINAMATH_CALUDE_sqrt_ax_cube_l1879_187902

theorem sqrt_ax_cube (a x : ℝ) (ha : a < 0) : 
  Real.sqrt (a * x^3) = -x * Real.sqrt (a * x) :=
sorry

end NUMINAMATH_CALUDE_sqrt_ax_cube_l1879_187902


namespace NUMINAMATH_CALUDE_x0_value_l1879_187989

-- Define the function f
def f (x : ℝ) : ℝ := 13 - 8*x + x^2

-- Define the derivative of f
def f' (x : ℝ) : ℝ := -8 + 2*x

-- Theorem statement
theorem x0_value (x₀ : ℝ) (h : f' x₀ = 4) : x₀ = 6 := by
  sorry

end NUMINAMATH_CALUDE_x0_value_l1879_187989


namespace NUMINAMATH_CALUDE_area_condition_implies_parallel_to_KL_l1879_187974

/-- A quadrilateral with non-parallel sides AB and CD -/
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)
  (not_parallel : ¬ (B.1 - A.1) * (D.2 - C.2) = (B.2 - A.2) * (D.1 - C.1))

/-- The area of a triangle given by three points -/
noncomputable def triangleArea (P Q R : ℝ × ℝ) : ℝ := sorry

/-- The area of a quadrilateral -/
noncomputable def quadrilateralArea (q : Quadrilateral) : ℝ := sorry

/-- The intersection point of lines AB and CD -/
noncomputable def intersectionPoint (q : Quadrilateral) : ℝ × ℝ := sorry

/-- Point K on the extension of AB such that OK = AB -/
noncomputable def pointK (q : Quadrilateral) : ℝ × ℝ := sorry

/-- Point L on the extension of CD such that OL = CD -/
noncomputable def pointL (q : Quadrilateral) : ℝ × ℝ := sorry

/-- Check if three points are collinear -/
def collinear (P Q R : ℝ × ℝ) : Prop := sorry

/-- Check if a point is inside a quadrilateral -/
def isInside (X : ℝ × ℝ) (q : Quadrilateral) : Prop := sorry

/-- Check if two lines are parallel -/
def parallel (P Q R S : ℝ × ℝ) : Prop := sorry

theorem area_condition_implies_parallel_to_KL (q : Quadrilateral) (X : ℝ × ℝ) :
  isInside X q →
  triangleArea q.A q.B X + triangleArea q.C q.D X = (quadrilateralArea q) / 2 →
  ∃ P Q : ℝ × ℝ, collinear P Q X ∧ parallel P Q (pointK q) (pointL q) :=
sorry

end NUMINAMATH_CALUDE_area_condition_implies_parallel_to_KL_l1879_187974


namespace NUMINAMATH_CALUDE_candies_sum_l1879_187996

/-- The number of candies Linda has -/
def linda_candies : ℕ := 34

/-- The number of candies Chloe has -/
def chloe_candies : ℕ := 28

/-- The total number of candies Linda and Chloe have together -/
def total_candies : ℕ := linda_candies + chloe_candies

theorem candies_sum : total_candies = 62 := by
  sorry

end NUMINAMATH_CALUDE_candies_sum_l1879_187996


namespace NUMINAMATH_CALUDE_carol_has_62_pennies_l1879_187993

/-- The number of pennies Alex currently has -/
def alex_pennies : ℕ := sorry

/-- The number of pennies Carol currently has -/
def carol_pennies : ℕ := sorry

/-- If Alex gives Carol two pennies, Carol will have four times as many pennies as Alex has -/
axiom condition1 : carol_pennies + 2 = 4 * (alex_pennies - 2)

/-- If Carol gives Alex two pennies, Carol will have three times as many pennies as Alex has -/
axiom condition2 : carol_pennies - 2 = 3 * (alex_pennies + 2)

/-- Carol has 62 pennies -/
theorem carol_has_62_pennies : carol_pennies = 62 := by sorry

end NUMINAMATH_CALUDE_carol_has_62_pennies_l1879_187993


namespace NUMINAMATH_CALUDE_banana_cost_proof_l1879_187923

/-- The cost of Tony's purchase in dollars -/
def tony_cost : ℚ := 7

/-- The number of dozen apples Tony bought -/
def tony_apples : ℕ := 2

/-- The cost of Arnold's purchase in dollars -/
def arnold_cost : ℚ := 5

/-- The number of dozen apples Arnold bought -/
def arnold_apples : ℕ := 1

/-- The number of bunches of bananas each person bought -/
def bananas : ℕ := 1

/-- The cost of a bunch of bananas in dollars -/
def banana_cost : ℚ := 3

theorem banana_cost_proof :
  banana_cost = tony_cost - arnold_cost - (tony_apples - arnold_apples) * (tony_cost - arnold_cost) :=
by
  sorry

end NUMINAMATH_CALUDE_banana_cost_proof_l1879_187923


namespace NUMINAMATH_CALUDE_jamie_bathroom_theorem_l1879_187915

/-- The amount of liquid (in ounces) that triggers the need to use the bathroom -/
def bathroom_threshold : ℕ := 32

/-- The amount of liquid (in ounces) in a cup -/
def cup_ounces : ℕ := 8

/-- The amount of liquid (in ounces) in a pint -/
def pint_ounces : ℕ := 16

/-- The amount of liquid Jamie consumed before the test -/
def pre_test_consumption : ℕ := cup_ounces + pint_ounces

/-- The maximum amount Jamie can drink during the test without needing the bathroom -/
def max_test_consumption : ℕ := bathroom_threshold - pre_test_consumption

theorem jamie_bathroom_theorem : max_test_consumption = 8 := by
  sorry

end NUMINAMATH_CALUDE_jamie_bathroom_theorem_l1879_187915


namespace NUMINAMATH_CALUDE_arithmetic_sequence_kth_term_l1879_187939

/-- Given an arithmetic sequence where the sum of the first n terms is 3n^2 + 2n,
    this theorem proves that the k-th term is 6k - 1. -/
theorem arithmetic_sequence_kth_term 
  (S : ℕ → ℝ) -- S represents the sum function of the sequence
  (h : ∀ n : ℕ, S n = 3 * n^2 + 2 * n) -- condition that sum of first n terms is 3n^2 + 2n
  (k : ℕ) -- k represents the index of the term we're looking for
  : S k - S (k-1) = 6 * k - 1 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_kth_term_l1879_187939


namespace NUMINAMATH_CALUDE_min_value_of_sum_of_roots_l1879_187916

theorem min_value_of_sum_of_roots (x : ℝ) : 
  Real.sqrt (x^2 + 4*x + 20) + Real.sqrt (x^2 + 2*x + 10) ≥ 5 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_sum_of_roots_l1879_187916


namespace NUMINAMATH_CALUDE_distinct_prime_factors_of_2310_l1879_187909

theorem distinct_prime_factors_of_2310 : Nat.card (Nat.factors 2310).toFinset = 5 := by
  sorry

end NUMINAMATH_CALUDE_distinct_prime_factors_of_2310_l1879_187909


namespace NUMINAMATH_CALUDE_largest_four_digit_perfect_square_l1879_187965

theorem largest_four_digit_perfect_square : 
  ∀ n : ℕ, 1000 ≤ n ∧ n ≤ 9999 → (∃ m : ℕ, n = m^2) → n ≤ 9261 :=
by sorry

end NUMINAMATH_CALUDE_largest_four_digit_perfect_square_l1879_187965


namespace NUMINAMATH_CALUDE_chandler_saves_49_weeks_l1879_187949

/-- The number of weeks it takes Chandler to save for a mountain bike --/
def weeks_to_save : ℕ :=
  let bike_cost : ℕ := 620
  let birthday_money : ℕ := 70 + 40 + 20
  let weekly_earnings : ℕ := 18
  let weekly_spending : ℕ := 8
  let weekly_savings : ℕ := weekly_earnings - weekly_spending
  ((bike_cost - birthday_money) + weekly_savings - 1) / weekly_savings

theorem chandler_saves_49_weeks :
  weeks_to_save = 49 :=
sorry

end NUMINAMATH_CALUDE_chandler_saves_49_weeks_l1879_187949


namespace NUMINAMATH_CALUDE_exam_failure_marks_l1879_187919

theorem exam_failure_marks (T : ℕ) (passing_mark : ℕ) : 
  (60 * T / 100 - 20 = passing_mark) →
  (passing_mark = 160) →
  (passing_mark - 40 * T / 100 = 40) :=
by sorry

end NUMINAMATH_CALUDE_exam_failure_marks_l1879_187919


namespace NUMINAMATH_CALUDE_circle_center_l1879_187973

/-- Given a circle with equation x^2 + y^2 - 2mx - 3 = 0, where m < 0 and radius 2, 
    prove that its center is (-1, 0) -/
theorem circle_center (m : ℝ) (h1 : m < 0) :
  let eq := fun (x y : ℝ) ↦ x^2 + y^2 - 2*m*x - 3 = 0
  let r : ℝ := 2
  ∃ (C : ℝ × ℝ), C = (-1, 0) ∧ 
    (∀ (x y : ℝ), eq x y ↔ (x - C.1)^2 + (y - C.2)^2 = r^2) := by
  sorry

end NUMINAMATH_CALUDE_circle_center_l1879_187973


namespace NUMINAMATH_CALUDE_fraction_of_books_sold_l1879_187953

/-- Given a collection of books where some were sold and some remained unsold,
    this theorem proves the fraction of books sold. -/
theorem fraction_of_books_sold
  (price_per_book : ℝ)
  (unsold_books : ℕ)
  (total_revenue : ℝ)
  (h1 : price_per_book = 3.5)
  (h2 : unsold_books = 40)
  (h3 : total_revenue = 280.00000000000006) :
  (total_revenue / price_per_book) / ((total_revenue / price_per_book) + unsold_books : ℝ) = 2/3 := by
  sorry

#eval (280.00000000000006 / 3.5) / ((280.00000000000006 / 3.5) + 40)

end NUMINAMATH_CALUDE_fraction_of_books_sold_l1879_187953


namespace NUMINAMATH_CALUDE_bobby_candy_consumption_l1879_187901

/-- The number of candy pieces Bobby ate first -/
def first_eaten : ℕ := 34

/-- The number of candy pieces Bobby ate later -/
def later_eaten : ℕ := 18

/-- The total number of candy pieces Bobby ate -/
def total_eaten : ℕ := first_eaten + later_eaten

theorem bobby_candy_consumption :
  total_eaten = 52 := by
  sorry

end NUMINAMATH_CALUDE_bobby_candy_consumption_l1879_187901


namespace NUMINAMATH_CALUDE_race_result_l1879_187955

/-- Represents a runner in a race -/
structure Runner where
  speed : ℝ
  time : ℝ

/-- Calculates the distance traveled by a runner given time -/
def distance (r : Runner) (t : ℝ) : ℝ := r.speed * t

theorem race_result (a b : Runner) 
  (h1 : a.time = 240)
  (h2 : b.time = a.time + 10)
  (h3 : distance a a.time = 1000) :
  distance a a.time - distance b a.time = 40 := by
  sorry

end NUMINAMATH_CALUDE_race_result_l1879_187955


namespace NUMINAMATH_CALUDE_father_twice_son_age_l1879_187966

/-- Represents the age difference between father and son when the father's age becomes more than twice the son's age -/
def AgeDifference : ℕ → Prop :=
  λ x => ∃ (y : ℕ), (27 + x = 2 * (((27 - 3) / 3) + x) + y) ∧ y > 0

/-- Theorem stating that it takes 11 years for the father's age to be more than twice the son's age -/
theorem father_twice_son_age : AgeDifference 11 := by
  sorry

end NUMINAMATH_CALUDE_father_twice_son_age_l1879_187966


namespace NUMINAMATH_CALUDE_investment_principal_l1879_187932

/-- 
Given two investments with the same principal and interest rate:
1. Peter's investment yields $815 after 3 years
2. David's investment yields $850 after 4 years
3. Both use simple interest

This theorem proves that the principal invested is $710
-/
theorem investment_principal (P r : ℚ) : 
  (P + P * r * 3 = 815) →
  (P + P * r * 4 = 850) →
  P = 710 := by
sorry

end NUMINAMATH_CALUDE_investment_principal_l1879_187932


namespace NUMINAMATH_CALUDE_bus_driver_max_regular_hours_l1879_187900

/-- Represents the problem of finding the maximum regular hours for a bus driver -/
theorem bus_driver_max_regular_hours 
  (regular_rate : ℝ) 
  (overtime_rate_factor : ℝ) 
  (total_compensation : ℝ) 
  (total_hours : ℝ) 
  (h1 : regular_rate = 16)
  (h2 : overtime_rate_factor = 1.75)
  (h3 : total_compensation = 1116)
  (h4 : total_hours = 57) :
  ∃ (max_regular_hours : ℝ),
    max_regular_hours * regular_rate + 
    (total_hours - max_regular_hours) * (regular_rate * overtime_rate_factor) = 
    total_compensation ∧ 
    max_regular_hours = 40 :=
by sorry

end NUMINAMATH_CALUDE_bus_driver_max_regular_hours_l1879_187900


namespace NUMINAMATH_CALUDE_set_intersection_empty_implies_a_range_l1879_187975

def A (a : ℝ) := {x : ℝ | a - 1 < x ∧ x < 2*a + 1}
def B := {x : ℝ | 0 < x ∧ x < 1}

theorem set_intersection_empty_implies_a_range (a : ℝ) :
  A a ∩ B = ∅ ↔ a ≤ -1/2 ∨ a ≥ 2 :=
sorry

end NUMINAMATH_CALUDE_set_intersection_empty_implies_a_range_l1879_187975


namespace NUMINAMATH_CALUDE_sqrt_eight_minus_sqrt_two_equals_sqrt_two_l1879_187967

theorem sqrt_eight_minus_sqrt_two_equals_sqrt_two :
  Real.sqrt 8 - Real.sqrt 2 = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_eight_minus_sqrt_two_equals_sqrt_two_l1879_187967


namespace NUMINAMATH_CALUDE_min_value_sum_and_sqrt_l1879_187934

theorem min_value_sum_and_sqrt (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  1/a + 1/b + 2 * Real.sqrt (a * b) ≥ 4 ∧
  (1/a + 1/b + 2 * Real.sqrt (a * b) = 4 ↔ a = b) := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_and_sqrt_l1879_187934


namespace NUMINAMATH_CALUDE_no_function_satisfies_condition_l1879_187908

-- Define the function g
def g : ℝ → ℝ := sorry

-- Properties of g
axiom g_integer : ∀ n : ℤ, g n = (-1) ^ n
axiom g_affine : ∀ n : ℤ, ∀ x : ℝ, n ≤ x → x ≤ n + 1 → 
  ∃ a b : ℝ, ∀ y : ℝ, n ≤ y → y ≤ n + 1 → g y = a * y + b

-- Theorem statement
theorem no_function_satisfies_condition : 
  ¬ ∃ f : ℝ → ℝ, ∀ x y : ℝ, f (x + f y) = f x + g y := by sorry

end NUMINAMATH_CALUDE_no_function_satisfies_condition_l1879_187908


namespace NUMINAMATH_CALUDE_min_non_isosceles_2008gon_l1879_187969

/-- The number of ones in the binary representation of a natural number -/
def binary_ones (n : ℕ) : ℕ := sorry

/-- A regular polygon -/
structure RegularPolygon where
  sides : ℕ
  sides_pos : sides > 0

/-- A triangulation of a regular polygon -/
structure Triangulation (p : RegularPolygon) where
  diagonals : ℕ
  diag_bound : diagonals ≤ p.sides - 3

/-- The number of non-isosceles triangles in a triangulation -/
def non_isosceles_count (p : RegularPolygon) (t : Triangulation p) : ℕ := sorry

theorem min_non_isosceles_2008gon :
  ∀ (p : RegularPolygon) (t : Triangulation p),
    p.sides = 2008 →
    t.diagonals = 2005 →
    non_isosceles_count p t ≥ 5 :=
sorry

end NUMINAMATH_CALUDE_min_non_isosceles_2008gon_l1879_187969


namespace NUMINAMATH_CALUDE_milkman_profit_is_51_l1879_187905

/-- Represents the milkman's problem --/
structure MilkmanProblem where
  total_milk : ℝ
  total_water : ℝ
  first_mixture_milk : ℝ
  first_mixture_water : ℝ
  second_mixture_water : ℝ
  pure_milk_cost : ℝ
  first_mixture_price : ℝ
  second_mixture_price : ℝ

/-- Calculate the total profit for the milkman --/
def calculate_profit (p : MilkmanProblem) : ℝ :=
  let second_mixture_milk := p.total_milk - p.first_mixture_milk
  let first_mixture_volume := p.first_mixture_milk + p.first_mixture_water
  let second_mixture_volume := second_mixture_milk + p.second_mixture_water
  let total_cost := p.pure_milk_cost * p.total_milk
  let total_revenue := p.first_mixture_price * first_mixture_volume + 
                       p.second_mixture_price * second_mixture_volume
  total_revenue - total_cost

/-- Theorem stating that the milkman's profit is 51 --/
theorem milkman_profit_is_51 : 
  let p : MilkmanProblem := {
    total_milk := 50,
    total_water := 15,
    first_mixture_milk := 30,
    first_mixture_water := 8,
    second_mixture_water := 7,
    pure_milk_cost := 20,
    first_mixture_price := 17,
    second_mixture_price := 15
  }
  calculate_profit p = 51 := by sorry


end NUMINAMATH_CALUDE_milkman_profit_is_51_l1879_187905


namespace NUMINAMATH_CALUDE_solve_for_x_l1879_187946

theorem solve_for_x (x y : ℝ) (h1 : x + 2*y = 100) (h2 : y = 25) : x = 50 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_x_l1879_187946


namespace NUMINAMATH_CALUDE_paper_clip_distribution_l1879_187922

theorem paper_clip_distribution (total_clips : ℕ) (num_boxes : ℕ) (clips_per_box : ℕ) : 
  total_clips = 81 → num_boxes = 9 → clips_per_box = total_clips / num_boxes → clips_per_box = 9 := by
  sorry

end NUMINAMATH_CALUDE_paper_clip_distribution_l1879_187922


namespace NUMINAMATH_CALUDE_complement_union_theorem_l1879_187942

-- Define the universal set U
def U : Finset Char := {'a', 'b', 'c', 'd'}

-- Define set A
def A : Finset Char := {'a', 'b'}

-- Define set B
def B : Finset Char := {'b', 'c', 'd'}

-- Theorem statement
theorem complement_union_theorem :
  (U \ A) ∪ (U \ B) = {'a', 'c', 'd'} := by sorry

end NUMINAMATH_CALUDE_complement_union_theorem_l1879_187942


namespace NUMINAMATH_CALUDE_john_uber_profit_l1879_187938

/-- John's profit from driving Uber --/
def uber_profit (earnings depreciation : ℕ) : ℕ :=
  earnings - depreciation

/-- Depreciation of John's car --/
def car_depreciation (purchase_price trade_in_value : ℕ) : ℕ :=
  purchase_price - trade_in_value

theorem john_uber_profit :
  let earnings : ℕ := 30000
  let purchase_price : ℕ := 18000
  let trade_in_value : ℕ := 6000
  uber_profit earnings (car_depreciation purchase_price trade_in_value) = 18000 := by
sorry

end NUMINAMATH_CALUDE_john_uber_profit_l1879_187938


namespace NUMINAMATH_CALUDE_stating_prob_no_adjacent_same_dice_rolls_l1879_187998

/-- The number of people sitting around the circular table -/
def n : ℕ := 5

/-- The number of sides on the die -/
def die_sides : ℕ := 8

/-- The probability of no two adjacent people rolling the same number -/
def prob_no_adjacent_same : ℚ := 637 / 2048

/-- 
Theorem stating that the probability of no two adjacent people
rolling the same number on an eight-sided die when 5 people
sit around a circular table is 637/2048.
-/
theorem prob_no_adjacent_same_dice_rolls 
  (h1 : n = 5)
  (h2 : die_sides = 8) :
  prob_no_adjacent_same = 637 / 2048 := by
  sorry


end NUMINAMATH_CALUDE_stating_prob_no_adjacent_same_dice_rolls_l1879_187998


namespace NUMINAMATH_CALUDE_triangle_area_l1879_187957

theorem triangle_area (A B C : ℝ) (a b c : ℝ) :
  B = π / 3 →
  a = 2 →
  b = Real.sqrt 3 →
  (1 / 2) * a * b * Real.sin B = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l1879_187957


namespace NUMINAMATH_CALUDE_math_competition_non_participants_l1879_187956

theorem math_competition_non_participants (total_students : ℕ) 
  (participation_ratio : ℚ) (h1 : total_students = 89) 
  (h2 : participation_ratio = 3/5) : 
  total_students - (participation_ratio * total_students).floor = 35 := by
  sorry

end NUMINAMATH_CALUDE_math_competition_non_participants_l1879_187956


namespace NUMINAMATH_CALUDE_alice_number_puzzle_l1879_187936

theorem alice_number_puzzle (x : ℝ) : 
  (((x + 3) * 3 - 3) / 3 = 10) → x = 8 := by sorry

end NUMINAMATH_CALUDE_alice_number_puzzle_l1879_187936


namespace NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l1879_187987

theorem quadratic_two_distinct_roots (k : ℝ) : 
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 - 3*k*x₁ - 2 = 0 ∧ x₂^2 - 3*k*x₂ - 2 = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l1879_187987


namespace NUMINAMATH_CALUDE_election_winner_percentage_l1879_187907

theorem election_winner_percentage : 
  ∀ (total_votes : ℕ) (winner_votes loser_votes : ℕ),
  winner_votes = 864 →
  winner_votes - loser_votes = 288 →
  total_votes = winner_votes + loser_votes →
  (winner_votes : ℚ) / (total_votes : ℚ) = 3/5 :=
by sorry

end NUMINAMATH_CALUDE_election_winner_percentage_l1879_187907


namespace NUMINAMATH_CALUDE_articles_produced_l1879_187941

/-- Given that x men working x hours a day for 2x days produce 2x³ articles,
    prove that y men working 2y hours a day for y days produce 2y³ articles. -/
theorem articles_produced (x y : ℕ) (h : x * x * (2 * x) = 2 * x^3) :
  y * (2 * y) * y = 2 * y^3 := by
  sorry

end NUMINAMATH_CALUDE_articles_produced_l1879_187941


namespace NUMINAMATH_CALUDE_set_as_interval_l1879_187960

def S : Set ℝ := {x : ℝ | -12 ≤ x ∧ x < 10 ∨ x > 11}

theorem set_as_interval : S = Set.Icc (-12) 10 ∪ Set.Ioi 11 := by sorry

end NUMINAMATH_CALUDE_set_as_interval_l1879_187960


namespace NUMINAMATH_CALUDE_boys_neither_happy_nor_sad_l1879_187997

/-- Given a group of children with various emotional states, prove the number of boys who are neither happy nor sad. -/
theorem boys_neither_happy_nor_sad
  (total_children : ℕ)
  (happy_children : ℕ)
  (sad_children : ℕ)
  (neither_children : ℕ)
  (total_boys : ℕ)
  (total_girls : ℕ)
  (happy_boys : ℕ)
  (sad_girls : ℕ)
  (h1 : total_children = 60)
  (h2 : happy_children = 30)
  (h3 : sad_children = 10)
  (h4 : neither_children = 20)
  (h5 : total_boys = 22)
  (h6 : total_girls = 38)
  (h7 : happy_boys = 6)
  (h8 : sad_girls = 4)
  (h9 : total_children = happy_children + sad_children + neither_children)
  (h10 : total_children = total_boys + total_girls) :
  total_boys - happy_boys - (sad_children - sad_girls) = 10 :=
by sorry

end NUMINAMATH_CALUDE_boys_neither_happy_nor_sad_l1879_187997


namespace NUMINAMATH_CALUDE_portias_university_students_l1879_187986

theorem portias_university_students :
  ∀ (p l c : ℕ),
  p = 4 * l →
  c = l / 2 →
  p + l + c = 4500 →
  p = 3273 :=
by
  sorry

end NUMINAMATH_CALUDE_portias_university_students_l1879_187986


namespace NUMINAMATH_CALUDE_cubic_roots_sum_cubes_l1879_187930

theorem cubic_roots_sum_cubes (p q r : ℝ) : 
  (p^3 - 2*p^2 + 7*p - 1 = 0) → 
  (q^3 - 2*q^2 + 7*q - 1 = 0) → 
  (r^3 - 2*r^2 + 7*r - 1 = 0) → 
  (p + q + r = 2) →
  (p * q * r = 1) →
  (p + q - 2)^3 + (q + r - 2)^3 + (r + p - 2)^3 = -3 := by
sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_cubes_l1879_187930


namespace NUMINAMATH_CALUDE_triangle_pcd_area_l1879_187964

/-- Given points P(0, 18), D(3, 18), and C(0, q) in a Cartesian coordinate system,
    where PD and PC are perpendicular sides of triangle PCD,
    prove that the area of triangle PCD is equal to 27 - (3/2)q. -/
theorem triangle_pcd_area (q : ℝ) : 
  let P : ℝ × ℝ := (0, 18)
  let D : ℝ × ℝ := (3, 18)
  let C : ℝ × ℝ := (0, q)
  -- PD and PC are perpendicular
  (D.1 - P.1) * (C.2 - P.2) = 0 →
  -- Area of triangle PCD
  (1/2) * (D.1 - P.1) * (P.2 - C.2) = 27 - (3/2) * q := by
  sorry

end NUMINAMATH_CALUDE_triangle_pcd_area_l1879_187964


namespace NUMINAMATH_CALUDE_sum_of_a_equals_2673_l1879_187961

def a (n : ℕ) : ℕ :=
  if n % 15 = 0 ∧ n % 18 = 0 then 15
  else if n % 18 = 0 ∧ n % 12 = 0 then 16
  else if n % 12 = 0 ∧ n % 15 = 0 then 17
  else 0

theorem sum_of_a_equals_2673 :
  (Finset.range 3000).sum (fun n => a (n + 1)) = 2673 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_a_equals_2673_l1879_187961


namespace NUMINAMATH_CALUDE_expand_and_simplify_l1879_187990

theorem expand_and_simplify (x : ℝ) (h : x ≠ 0) :
  (3 / 7) * (7 / x^2 + 15 * x^3 - 4 * x) = 3 / x^2 + 45 * x^3 / 7 - 12 * x / 7 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_l1879_187990


namespace NUMINAMATH_CALUDE_count_integer_values_is_ten_l1879_187906

/-- The number of integer values of n for which 8000 * (2/5)^n is an integer --/
def count_integer_values : ℕ := 10

/-- Predicate to check if a given number is an integer --/
def is_integer (x : ℚ) : Prop := ∃ (k : ℤ), x = k

/-- The main theorem stating that there are exactly 10 integer values of n
    for which 8000 * (2/5)^n is an integer --/
theorem count_integer_values_is_ten :
  (∃! (s : Finset ℤ), s.card = count_integer_values ∧
    ∀ n : ℤ, n ∈ s ↔ is_integer (8000 * (2/5)^n)) :=
sorry

end NUMINAMATH_CALUDE_count_integer_values_is_ten_l1879_187906


namespace NUMINAMATH_CALUDE_unique_prime_digit_l1879_187951

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def number (B : ℕ) : ℕ := 303160 + B

theorem unique_prime_digit :
  ∃! B : ℕ, B < 10 ∧ is_prime (number B) :=
sorry

end NUMINAMATH_CALUDE_unique_prime_digit_l1879_187951


namespace NUMINAMATH_CALUDE_isabellas_hair_length_l1879_187950

/-- Given Isabella's hair length at the end of the year and the amount it grew,
    prove that her initial hair length is equal to the final length minus the growth. -/
theorem isabellas_hair_length (final_length growth : ℕ) (h : final_length = 24 ∧ growth = 6) :
  final_length - growth = 18 := by
  sorry

end NUMINAMATH_CALUDE_isabellas_hair_length_l1879_187950


namespace NUMINAMATH_CALUDE_min_cost_for_48_students_l1879_187948

/-- The minimum cost to purchase tickets for a group of students. -/
def min_ticket_cost (num_students : ℕ) (single_price : ℕ) (group_price : ℕ) : ℕ :=
  min
    ((num_students / 10) * group_price + (num_students % 10) * single_price)
    ((num_students / 10 + 1) * group_price)

/-- The minimum cost to purchase tickets for 48 students is 350 yuan. -/
theorem min_cost_for_48_students :
  min_ticket_cost 48 10 70 = 350 := by
  sorry

#eval min_ticket_cost 48 10 70

end NUMINAMATH_CALUDE_min_cost_for_48_students_l1879_187948


namespace NUMINAMATH_CALUDE_select_students_theorem_l1879_187920

/-- Represents the number of students in each category for a group -/
structure GroupComposition :=
  (male : ℕ)
  (female : ℕ)

/-- Calculates the number of ways to select students from two groups with exactly one female -/
def selectStudentsWithOneFemale (groupA groupB : GroupComposition) : ℕ :=
  let selectOneFromA := groupA.female * groupA.male * (groupB.male.choose 2)
  let selectOneFromB := groupB.female * groupB.male * (groupA.male.choose 2)
  selectOneFromA + selectOneFromB

/-- The main theorem stating the number of ways to select students -/
theorem select_students_theorem (groupA groupB : GroupComposition) : 
  groupA.male = 5 → groupA.female = 3 → groupB.male = 6 → groupB.female = 2 →
  selectStudentsWithOneFemale groupA groupB = 345 := by
  sorry

#eval selectStudentsWithOneFemale ⟨5, 3⟩ ⟨6, 2⟩

end NUMINAMATH_CALUDE_select_students_theorem_l1879_187920


namespace NUMINAMATH_CALUDE_sara_apples_l1879_187927

theorem sara_apples (total : ℕ) (ali_factor : ℕ) (sara_apples : ℕ) 
  (h1 : total = 80)
  (h2 : ali_factor = 4)
  (h3 : total = sara_apples + ali_factor * sara_apples) :
  sara_apples = 16 := by
  sorry

end NUMINAMATH_CALUDE_sara_apples_l1879_187927


namespace NUMINAMATH_CALUDE_smallest_candy_count_l1879_187952

theorem smallest_candy_count : ∃ (n : ℕ), 
  (100 ≤ n ∧ n < 1000) ∧ 
  (n + 6) % 9 = 0 ∧ 
  (n - 9) % 6 = 0 ∧
  (∀ m : ℕ, (100 ≤ m ∧ m < n) → (m + 6) % 9 ≠ 0 ∨ (m - 9) % 6 ≠ 0) ∧
  n = 111 := by
sorry

end NUMINAMATH_CALUDE_smallest_candy_count_l1879_187952


namespace NUMINAMATH_CALUDE_certain_number_proof_l1879_187940

theorem certain_number_proof : 
  ∃ (x : ℝ), x / 1.45 = 17.5 → x = 25.375 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l1879_187940


namespace NUMINAMATH_CALUDE_shinyoung_read_most_l1879_187977

theorem shinyoung_read_most (shinyoung seokgi woong : ℚ) : 
  shinyoung = 1/3 ∧ seokgi = 1/4 ∧ woong = 1/5 → 
  shinyoung > seokgi ∧ shinyoung > woong := by
  sorry

end NUMINAMATH_CALUDE_shinyoung_read_most_l1879_187977


namespace NUMINAMATH_CALUDE_second_place_wins_l1879_187979

/-- Represents a hockey team's performance --/
structure TeamPerformance where
  wins : ℕ
  ties : ℕ

/-- Calculates points for a team based on wins and ties --/
def calculatePoints (team : TeamPerformance) : ℕ := 2 * team.wins + team.ties

/-- Represents the hockey league --/
structure HockeyLeague where
  firstPlace : TeamPerformance
  secondPlace : TeamPerformance
  elsasTeam : TeamPerformance

theorem second_place_wins (league : HockeyLeague) : 
  league.firstPlace = ⟨12, 4⟩ →
  league.elsasTeam = ⟨8, 10⟩ →
  league.secondPlace.ties = 1 →
  (calculatePoints league.firstPlace + calculatePoints league.secondPlace + calculatePoints league.elsasTeam) / 3 = 27 →
  league.secondPlace.wins = 13 := by
  sorry

#eval calculatePoints ⟨13, 1⟩  -- Expected output: 27

end NUMINAMATH_CALUDE_second_place_wins_l1879_187979


namespace NUMINAMATH_CALUDE_inequality_range_l1879_187980

theorem inequality_range (k : ℝ) : 
  (∀ x : ℝ, 2*k*x^2 + k*x - 3/8 < 0) ↔ k ∈ Set.Ioc (-3) 0 :=
sorry

end NUMINAMATH_CALUDE_inequality_range_l1879_187980


namespace NUMINAMATH_CALUDE_domain_intersection_is_closed_open_interval_l1879_187945

-- Define the domains of the two functions
def domain_sqrt (x : ℝ) : Prop := 4 - x^2 ≥ 0
def domain_ln (x : ℝ) : Prop := 4 - x > 0

-- Define the intersection of the domains
def domain_intersection (x : ℝ) : Prop := domain_sqrt x ∧ domain_ln x

-- Theorem statement
theorem domain_intersection_is_closed_open_interval :
  ∀ x, domain_intersection x ↔ x ∈ Set.Ici (-2) ∩ Set.Iio 1 :=
sorry

end NUMINAMATH_CALUDE_domain_intersection_is_closed_open_interval_l1879_187945


namespace NUMINAMATH_CALUDE_defective_draws_count_l1879_187937

/-- The number of ways to draw at least 3 defective products out of 5 from a batch of 50 products containing 4 defective ones -/
def defective_draws : ℕ := sorry

/-- Total number of products in the batch -/
def total_products : ℕ := 50

/-- Number of defective products in the batch -/
def defective_products : ℕ := 4

/-- Number of products drawn -/
def drawn_products : ℕ := 5

theorem defective_draws_count : defective_draws = 4186 := by sorry

end NUMINAMATH_CALUDE_defective_draws_count_l1879_187937


namespace NUMINAMATH_CALUDE_crayfish_yield_theorem_l1879_187962

theorem crayfish_yield_theorem (last_year_total : ℝ) (this_year_total : ℝ) 
  (yield_difference : ℝ) (h1 : last_year_total = 4800) 
  (h2 : this_year_total = 6000) (h3 : yield_difference = 60) : 
  ∃ (x : ℝ), x = 300 ∧ this_year_total / x = last_year_total / (x - yield_difference) :=
by sorry

end NUMINAMATH_CALUDE_crayfish_yield_theorem_l1879_187962


namespace NUMINAMATH_CALUDE_fraction_subtraction_equality_l1879_187926

theorem fraction_subtraction_equality : (3 + 6 + 9) / (2 + 5 + 8) - (2 + 5 + 8) / (3 + 6 + 9) = 11 / 30 := by
  sorry

end NUMINAMATH_CALUDE_fraction_subtraction_equality_l1879_187926


namespace NUMINAMATH_CALUDE_A_intersect_B_equals_unit_interval_l1879_187968

-- Define the sets A and B
def A : Set ℝ := {y | ∃ x, y = Real.sin x}
def B : Set ℝ := {y | ∃ x, y = Real.sqrt (-x^2 + 4*x - 3)}

-- State the theorem
theorem A_intersect_B_equals_unit_interval :
  A ∩ B = Set.Icc 0 1 := by sorry

end NUMINAMATH_CALUDE_A_intersect_B_equals_unit_interval_l1879_187968


namespace NUMINAMATH_CALUDE_triangle_inequality_l1879_187918

-- Define a triangle ABC in 2D space
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define a point M
def Point := ℝ × ℝ

-- Function to calculate distance between two points
def distance (p q : ℝ × ℝ) : ℝ := sorry

-- Function to check if a point is inside a triangle
def isInside (t : Triangle) (p : Point) : Prop := sorry

-- Function to calculate the perimeter of a triangle
def perimeter (t : Triangle) : ℝ := sorry

-- Theorem statement
theorem triangle_inequality (t : Triangle) (M : Point) 
  (h : isInside t M) : 
  min (distance M t.A) (min (distance M t.B) (distance M t.C)) + 
  distance M t.A + distance M t.B + distance M t.C < 
  perimeter t := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l1879_187918


namespace NUMINAMATH_CALUDE_ellipse_constant_product_l1879_187935

-- Define the ellipse C
def ellipse_C (x y : ℝ) : Prop := x^2 / 6 + y^2 / 2 = 1

-- Define the moving line
def moving_line (k x y : ℝ) : Prop := y = k * (x - 2)

-- Define the dot product of vectors
def dot_product (x1 y1 x2 y2 : ℝ) : ℝ := x1 * x2 + y1 * y2

-- Theorem statement
theorem ellipse_constant_product :
  ∃ (E : ℝ × ℝ),
    E.2 = 0 ∧
    ∀ (k : ℝ) (A B : ℝ × ℝ),
      k ≠ 0 →
      ellipse_C A.1 A.2 →
      ellipse_C B.1 B.2 →
      moving_line k A.1 A.2 →
      moving_line k B.1 B.2 →
      dot_product (A.1 - E.1) (A.2 - E.2) (B.1 - E.1) (B.2 - E.2) = -5/9 :=
sorry

end NUMINAMATH_CALUDE_ellipse_constant_product_l1879_187935


namespace NUMINAMATH_CALUDE_number_of_valid_paths_l1879_187988

-- Define the grid dimensions
def grid_width : ℕ := 8
def grid_height : ℕ := 4

-- Define the blocked segments
def blocked_segments : List (ℕ × ℕ × ℕ × ℕ) := [(6, 2, 6, 3), (8, 2, 8, 3)]

-- Define a function to calculate valid paths
def valid_paths (width : ℕ) (height : ℕ) (blocked : List (ℕ × ℕ × ℕ × ℕ)) : ℕ :=
  sorry

-- Theorem statement
theorem number_of_valid_paths :
  valid_paths grid_width grid_height blocked_segments = 271 :=
sorry

end NUMINAMATH_CALUDE_number_of_valid_paths_l1879_187988


namespace NUMINAMATH_CALUDE_square_areas_sum_l1879_187983

theorem square_areas_sum (a b c : ℕ) (ha : a = 2) (hb : b = 3) (hc : c = 6) :
  a^2 + b^2 + c^2 = 7^2 := by
  sorry

end NUMINAMATH_CALUDE_square_areas_sum_l1879_187983


namespace NUMINAMATH_CALUDE_white_white_pairs_coincide_l1879_187913

/-- Represents a half of the geometric figure -/
structure Half where
  red : ℕ
  blue : ℕ
  white : ℕ

/-- Represents the coinciding pairs when the halves are folded -/
structure CoincidingPairs where
  red_red : ℕ
  blue_blue : ℕ
  red_white : ℕ

/-- The main theorem statement -/
theorem white_white_pairs_coincide 
  (half : Half) 
  (coinciding : CoincidingPairs) 
  (h1 : half.red = 4) 
  (h2 : half.blue = 7) 
  (h3 : half.white = 10) 
  (h4 : coinciding.red_red = 3) 
  (h5 : coinciding.blue_blue = 4) 
  (h6 : coinciding.red_white = 3) : 
  ∃ (white_white : ℕ), white_white = 7 ∧ 
    white_white = half.white - coinciding.red_white := by
  sorry

end NUMINAMATH_CALUDE_white_white_pairs_coincide_l1879_187913


namespace NUMINAMATH_CALUDE_santa_gift_combinations_l1879_187992

theorem santa_gift_combinations (n : ℤ) : 30 ∣ (n^5 - n) := by
  sorry

end NUMINAMATH_CALUDE_santa_gift_combinations_l1879_187992


namespace NUMINAMATH_CALUDE_tan_alpha_implies_sin_2alpha_plus_pi_half_l1879_187994

theorem tan_alpha_implies_sin_2alpha_plus_pi_half (α : Real) 
  (h : Real.tan α = -Real.cos α / (3 + Real.sin α)) : 
  Real.sin (2 * α + π / 2) = 7 / 9 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_implies_sin_2alpha_plus_pi_half_l1879_187994


namespace NUMINAMATH_CALUDE_bicycle_installation_problem_l1879_187910

/-- The number of bicycles a skilled worker can install per day -/
def x : ℕ := sorry

/-- The number of bicycles a new worker can install per day -/
def y : ℕ := sorry

/-- The number of skilled workers -/
def a : ℕ := sorry

/-- The number of new workers -/
def b : ℕ := sorry

/-- Theorem stating the conditions and expected results -/
theorem bicycle_installation_problem :
  (2 * x + 3 * y = 44) ∧
  (4 * x = 5 * y) ∧
  (25 * (a * x + b * y) = 3500) →
  (x = 10 ∧ y = 8) ∧
  ((a = 2 ∧ b = 15) ∨ (a = 6 ∧ b = 10) ∨ (a = 10 ∧ b = 5)) :=
by sorry

end NUMINAMATH_CALUDE_bicycle_installation_problem_l1879_187910


namespace NUMINAMATH_CALUDE_circle_existence_theorem_l1879_187921

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A circle in a 2D plane -/
structure Circle where
  center : Point
  radius : ℝ

/-- Check if a point is inside a circle -/
def isInside (p : Point) (c : Circle) : Prop :=
  (p.x - c.center.x)^2 + (p.y - c.center.y)^2 < c.radius^2

/-- Check if a point is on a circle -/
def isOn (p : Point) (c : Circle) : Prop :=
  (p.x - c.center.x)^2 + (p.y - c.center.y)^2 = c.radius^2

/-- Check if three points are collinear -/
def areCollinear (p1 p2 p3 : Point) : Prop :=
  (p2.y - p1.y) * (p3.x - p1.x) = (p3.y - p1.y) * (p2.x - p1.x)

/-- The main theorem -/
theorem circle_existence_theorem (n : ℕ) (points : Fin n → Point) 
    (h1 : n ≥ 3) 
    (h2 : ∃ p1 p2 p3 : Fin n, ¬areCollinear (points p1) (points p2) (points p3)) :
    ∃ (c : Circle) (p1 p2 p3 : Fin n), 
      p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 ∧
      isOn (points p1) c ∧ isOn (points p2) c ∧ isOn (points p3) c ∧
      ∀ (p : Fin n), p ≠ p1 → p ≠ p2 → p ≠ p3 → ¬isInside (points p) c :=
  sorry

end NUMINAMATH_CALUDE_circle_existence_theorem_l1879_187921


namespace NUMINAMATH_CALUDE_wood_measurement_correct_l1879_187970

/-- Represents the length of the wood in feet -/
def wood_length : ℝ := sorry

/-- Represents the length of the rope in feet -/
def rope_length : ℝ := sorry

/-- The system of equations for the wood measurement problem -/
def wood_measurement_equations : Prop :=
  (rope_length - wood_length = 4.5) ∧ (wood_length - 1/2 * rope_length = 1)

/-- Theorem stating that the system of equations correctly represents the wood measurement problem -/
theorem wood_measurement_correct : wood_measurement_equations :=
sorry

end NUMINAMATH_CALUDE_wood_measurement_correct_l1879_187970


namespace NUMINAMATH_CALUDE_arithmetic_sequence_a10_l1879_187933

/-- An arithmetic sequence {a_n} -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_a10
  (a : ℕ → ℝ)
  (h_arithmetic : arithmetic_sequence a)
  (h_sum : a 6 + a 8 = 16)
  (h_a4 : a 4 = 1) :
  a 10 = 15 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_a10_l1879_187933


namespace NUMINAMATH_CALUDE_monotonic_increasing_cubic_linear_l1879_187954

/-- The function f(x) = x^3 - ax is monotonically increasing over ℝ if and only if a ≤ 0 -/
theorem monotonic_increasing_cubic_linear (a : ℝ) :
  (∀ x : ℝ, Monotone (fun x => x^3 - a*x)) ↔ a ≤ 0 := by sorry

end NUMINAMATH_CALUDE_monotonic_increasing_cubic_linear_l1879_187954


namespace NUMINAMATH_CALUDE_sum_series_result_l1879_187991

def double_factorial : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | n + 2 => (n + 2) * double_factorial n

def sum_series (n : ℕ) : ℚ :=
  (Finset.range n).sum (λ i => (↑(double_factorial (2*i+1)) / ↑(double_factorial (2*i+2)) + 1 / 2^(i+1)))

theorem sum_series_result : 
  ∃ (a b : ℕ), b % 2 = 1 ∧ 
    (∃ (num : ℕ), sum_series 2023 = num / (2^a * b : ℚ)) ∧
    a * b / 10 = 4039 / 10 := by
  sorry

end NUMINAMATH_CALUDE_sum_series_result_l1879_187991


namespace NUMINAMATH_CALUDE_permutation_element_selection_l1879_187917

theorem permutation_element_selection (n : ℕ) (hn : n ≥ 10) :
  (Finset.range n).card.choose 3 = Nat.choose (n - 7) 3 :=
by sorry

end NUMINAMATH_CALUDE_permutation_element_selection_l1879_187917


namespace NUMINAMATH_CALUDE_ascending_order_l1879_187984

theorem ascending_order (a b c : ℝ) (h1 : a > b) (h2 : b > c) (h3 : c > 1) :
  b * c < a * c ∧ a * c < a * b ∧ a * b < a * b * c := by
  sorry

end NUMINAMATH_CALUDE_ascending_order_l1879_187984


namespace NUMINAMATH_CALUDE_restaurant_bill_calculation_l1879_187976

theorem restaurant_bill_calculation (adults children meal_cost : ℕ) 
  (h1 : adults = 2) 
  (h2 : children = 5) 
  (h3 : meal_cost = 3) : 
  (adults + children) * meal_cost = 21 := by
  sorry

end NUMINAMATH_CALUDE_restaurant_bill_calculation_l1879_187976


namespace NUMINAMATH_CALUDE_ladder_rungs_count_ladder_rungs_count_proof_l1879_187999

theorem ladder_rungs_count : ℕ → Prop :=
  fun n =>
    let middle_rung := n / 2
    let final_position := middle_rung + 5 - 7 + 8 + 7
    (n % 2 = 1) ∧ (final_position = n) → n = 27

-- The proof is omitted
theorem ladder_rungs_count_proof : ladder_rungs_count 27 := by sorry

end NUMINAMATH_CALUDE_ladder_rungs_count_ladder_rungs_count_proof_l1879_187999


namespace NUMINAMATH_CALUDE_man_crossing_bridge_l1879_187947

/-- Proves that a man walking at 6 km/hr will take 15 minutes to cross a bridge of 1500 meters in length. -/
theorem man_crossing_bridge (walking_speed : Real) (bridge_length : Real) (crossing_time : Real) : 
  walking_speed = 6 → bridge_length = 1500 → crossing_time = 15 → 
  crossing_time * (walking_speed * 1000 / 60) = bridge_length := by
  sorry

#check man_crossing_bridge

end NUMINAMATH_CALUDE_man_crossing_bridge_l1879_187947


namespace NUMINAMATH_CALUDE_num_workers_is_500_l1879_187904

/-- The number of workers who raised money by equal contribution -/
def num_workers : ℕ := sorry

/-- The original contribution amount per worker in rupees -/
def contribution_per_worker : ℕ := sorry

/-- The total contribution is 300,000 rupees -/
axiom total_contribution : num_workers * contribution_per_worker = 300000

/-- If each worker contributed 50 rupees extra, the total would be 325,000 rupees -/
axiom total_with_extra : num_workers * (contribution_per_worker + 50) = 325000

/-- Theorem: The number of workers is 500 -/
theorem num_workers_is_500 : num_workers = 500 := by sorry

end NUMINAMATH_CALUDE_num_workers_is_500_l1879_187904


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l1879_187958

theorem imaginary_part_of_z (z : ℂ) (h : (2 - Complex.I) * z = 5) : 
  Complex.im z = 1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l1879_187958


namespace NUMINAMATH_CALUDE_prop_negation_false_l1879_187944

theorem prop_negation_false (p q : Prop) : 
  ¬(¬(p ∧ q)) → (p ∧ q) := by
  sorry

end NUMINAMATH_CALUDE_prop_negation_false_l1879_187944


namespace NUMINAMATH_CALUDE_fish_catch_theorem_l1879_187982

def mike_rate : ℕ := 30
def jim_rate : ℕ := 2 * mike_rate
def bob_rate : ℕ := (3 * jim_rate) / 2

def total_fish_caught (mike_rate jim_rate bob_rate : ℕ) : ℕ :=
  let fish_40_min := (mike_rate + jim_rate + bob_rate) * 2 / 3
  let fish_20_min := jim_rate * 1 / 3
  fish_40_min + fish_20_min

theorem fish_catch_theorem :
  total_fish_caught mike_rate jim_rate bob_rate = 140 := by
  sorry

end NUMINAMATH_CALUDE_fish_catch_theorem_l1879_187982


namespace NUMINAMATH_CALUDE_movie_ticket_distribution_l1879_187925

theorem movie_ticket_distribution (n : ℕ) (k : ℕ) (h1 : n = 10) (h2 : k = 3) :
  (n.descFactorial k) = 720 :=
sorry

end NUMINAMATH_CALUDE_movie_ticket_distribution_l1879_187925


namespace NUMINAMATH_CALUDE_a_squared_gt_b_squared_necessary_not_sufficient_l1879_187972

theorem a_squared_gt_b_squared_necessary_not_sufficient (a b : ℝ) :
  (∀ a b : ℝ, a^3 > b^3 ∧ b^3 > 0 → a^2 > b^2) ∧
  (∃ a b : ℝ, a^2 > b^2 ∧ ¬(a^3 > b^3 ∧ b^3 > 0)) :=
by sorry

end NUMINAMATH_CALUDE_a_squared_gt_b_squared_necessary_not_sufficient_l1879_187972


namespace NUMINAMATH_CALUDE_crayons_added_l1879_187914

theorem crayons_added (initial : ℕ) (final : ℕ) (added : ℕ) : 
  initial = 9 → final = 12 → added = final - initial → added = 3 := by sorry

end NUMINAMATH_CALUDE_crayons_added_l1879_187914


namespace NUMINAMATH_CALUDE_series_convergence_l1879_187943

/-- The sum of the infinite series ∑(n=1 to ∞) [(n³+4n²+8n+8) / (3ⁿ·(n³+5))] converges to 1/2. -/
theorem series_convergence : 
  let f : ℕ → ℝ := λ n => (n^3 + 4*n^2 + 8*n + 8) / (3^n * (n^3 + 5))
  ∑' n, f n = 1/2 := by sorry

end NUMINAMATH_CALUDE_series_convergence_l1879_187943


namespace NUMINAMATH_CALUDE_no_prime_common_multiple_under_70_l1879_187931

theorem no_prime_common_multiple_under_70 : ¬ ∃ n : ℕ, 
  (10 ∣ n) ∧ (15 ∣ n) ∧ (n < 70) ∧ Nat.Prime n :=
by sorry

end NUMINAMATH_CALUDE_no_prime_common_multiple_under_70_l1879_187931


namespace NUMINAMATH_CALUDE_tabithas_age_l1879_187978

/-- Tabitha's hair color problem -/
theorem tabithas_age :
  ∀ (current_year : ℕ) (start_year : ℕ) (start_colors : ℕ) (future_year : ℕ) (future_colors : ℕ),
  start_year = 15 →
  start_colors = 2 →
  future_year = current_year + 3 →
  future_colors = 8 →
  future_colors = start_colors + (future_year - start_year) →
  current_year = 18 :=
by sorry

end NUMINAMATH_CALUDE_tabithas_age_l1879_187978


namespace NUMINAMATH_CALUDE_two_enchiladas_five_tacos_cost_l1879_187924

/-- The price of an enchilada in dollars -/
def enchilada_price : ℝ := sorry

/-- The price of a taco in dollars -/
def taco_price : ℝ := sorry

/-- The condition that one enchilada and four tacos cost $3.50 -/
axiom condition1 : enchilada_price + 4 * taco_price = 3.50

/-- The condition that four enchiladas and one taco cost $4.20 -/
axiom condition2 : 4 * enchilada_price + taco_price = 4.20

/-- The theorem stating that two enchiladas and five tacos cost $5.04 -/
theorem two_enchiladas_five_tacos_cost : 
  2 * enchilada_price + 5 * taco_price = 5.04 := by sorry

end NUMINAMATH_CALUDE_two_enchiladas_five_tacos_cost_l1879_187924


namespace NUMINAMATH_CALUDE_nail_positions_symmetry_l1879_187971

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents the shape of the flag -/
structure FlagShape where
  width : ℝ
  height : ℝ
  -- Additional parameters could be added to describe the specific shape

/-- Predicate to check if a nail position allows the flag to cover the hole -/
def covers (hole : Point) (nail : Point) (flag : FlagShape) : Prop :=
  -- This would involve checking if the hole is within the bounds of the flag
  -- when placed at the nail position
  sorry

/-- The set of all valid nail positions for a given hole and flag shape -/
def validNailPositions (hole : Point) (flag : FlagShape) : Set Point :=
  {nail : Point | covers hole nail flag}

theorem nail_positions_symmetry (hole : Point) (flag : FlagShape) :
  ∃ (center : Point), ∀ (nail : Point),
    nail ∈ validNailPositions hole flag →
    ∃ (symmetricNail : Point),
      symmetricNail ∈ validNailPositions hole flag ∧
      center.x = (nail.x + symmetricNail.x) / 2 ∧
      center.y = (nail.y + symmetricNail.y) / 2 :=
  sorry

end NUMINAMATH_CALUDE_nail_positions_symmetry_l1879_187971


namespace NUMINAMATH_CALUDE_intersection_distance_l1879_187985

-- Define the quadratic function
def f (x : ℝ) : ℝ := 3 * x^2 + 2 * x - 5

-- Define the horizontal line
def g (x : ℝ) : ℝ := 2

-- Define the intersection points
def intersection_points : Set ℝ := {x : ℝ | f x = g x}

-- Theorem statement
theorem intersection_distance :
  ∃ (x₁ x₂ : ℝ), x₁ ∈ intersection_points ∧ x₂ ∈ intersection_points ∧ x₁ ≠ x₂ ∧
  |x₁ - x₂| = 2 * Real.sqrt 22 / 3 :=
sorry

end NUMINAMATH_CALUDE_intersection_distance_l1879_187985


namespace NUMINAMATH_CALUDE_retailer_profit_calculation_l1879_187995

theorem retailer_profit_calculation 
  (cost_price : ℝ) 
  (markup_percentage : ℝ) 
  (discount_percentage : ℝ) 
  (actual_profit_percentage : ℝ) 
  (h1 : markup_percentage = 60) 
  (h2 : discount_percentage = 25) 
  (h3 : actual_profit_percentage = 20) : 
  markup_percentage = 60 := by
sorry

end NUMINAMATH_CALUDE_retailer_profit_calculation_l1879_187995


namespace NUMINAMATH_CALUDE_lg_equation_l1879_187981

-- Define lg as the base-10 logarithm
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem lg_equation : (lg 5)^2 + lg 2 * lg 50 = 1 := by
  sorry

end NUMINAMATH_CALUDE_lg_equation_l1879_187981
