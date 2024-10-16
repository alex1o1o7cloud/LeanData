import Mathlib

namespace NUMINAMATH_CALUDE_winter_clothing_count_l2027_202731

theorem winter_clothing_count (num_boxes : ℕ) (scarves_per_box : ℕ) (mittens_per_box : ℕ) : 
  num_boxes = 4 → scarves_per_box = 2 → mittens_per_box = 6 →
  num_boxes * scarves_per_box + num_boxes * mittens_per_box = 32 := by
  sorry

end NUMINAMATH_CALUDE_winter_clothing_count_l2027_202731


namespace NUMINAMATH_CALUDE_right_triangle_area_l2027_202799

theorem right_triangle_area (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  b = (2/3) * a →
  b = (2/3) * c →
  a^2 + b^2 = c^2 →
  (1/2) * a * b = 32/9 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_area_l2027_202799


namespace NUMINAMATH_CALUDE_wrench_force_problem_l2027_202757

/-- The force required to loosen a nut varies inversely with the length of the wrench handle -/
def inverse_variation (force : ℝ) (length : ℝ) : Prop :=
  ∃ k : ℝ, k > 0 ∧ force * length = k

theorem wrench_force_problem (force₁ : ℝ) (length₁ : ℝ) (force₂ : ℝ) (length₂ : ℝ) :
  inverse_variation force₁ length₁ →
  inverse_variation force₂ length₂ →
  force₁ = 300 →
  length₁ = 12 →
  length₂ = 18 →
  force₂ = 200 := by
  sorry

end NUMINAMATH_CALUDE_wrench_force_problem_l2027_202757


namespace NUMINAMATH_CALUDE_second_year_interest_rate_l2027_202771

/-- Proves that given an initial investment of $15,000 with a 10% simple annual interest rate
    for the first year, and a final amount of $17,325 after two years, the interest rate of
    the second year's investment is 5%. -/
theorem second_year_interest_rate
  (initial_investment : ℝ)
  (first_year_rate : ℝ)
  (final_amount : ℝ)
  (h1 : initial_investment = 15000)
  (h2 : first_year_rate = 0.1)
  (h3 : final_amount = 17325)
  : ∃ (second_year_rate : ℝ),
    final_amount = initial_investment * (1 + first_year_rate) * (1 + second_year_rate) ∧
    second_year_rate = 0.05 := by
  sorry

end NUMINAMATH_CALUDE_second_year_interest_rate_l2027_202771


namespace NUMINAMATH_CALUDE_max_value_of_s_l2027_202711

theorem max_value_of_s (p q r s : ℝ) 
  (sum_eq : p + q + r + s = 10)
  (sum_prod_eq : p*q + p*r + p*s + q*r + q*s + r*s = 20) :
  s ≤ (5 * (1 + Real.sqrt 21)) / 2 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_s_l2027_202711


namespace NUMINAMATH_CALUDE_henrys_age_l2027_202710

/-- Given that the sum of Henry and Jill's present ages is 41, and 7 years ago Henry was twice the age of Jill, prove that Henry's present age is 25. -/
theorem henrys_age (h_age j_age : ℕ) 
  (sum_condition : h_age + j_age = 41)
  (past_condition : h_age - 7 = 2 * (j_age - 7)) :
  h_age = 25 := by
  sorry

end NUMINAMATH_CALUDE_henrys_age_l2027_202710


namespace NUMINAMATH_CALUDE_initial_gasoline_percentage_l2027_202701

/-- Proves that the initial gasoline percentage is 95% given the problem conditions -/
theorem initial_gasoline_percentage
  (initial_volume : ℝ)
  (initial_ethanol_percentage : ℝ)
  (optimal_ethanol_percentage : ℝ)
  (added_ethanol : ℝ)
  (h1 : initial_volume = 36)
  (h2 : initial_ethanol_percentage = 0.05)
  (h3 : optimal_ethanol_percentage = 0.10)
  (h4 : added_ethanol = 2)
  (h5 : optimal_ethanol_percentage * (initial_volume + added_ethanol) =
        initial_ethanol_percentage * initial_volume + added_ethanol) :
  initial_volume * (1 - initial_ethanol_percentage) / initial_volume = 0.95 := by
  sorry

#check initial_gasoline_percentage

end NUMINAMATH_CALUDE_initial_gasoline_percentage_l2027_202701


namespace NUMINAMATH_CALUDE_alvin_age_l2027_202749

/-- Alvin's age -/
def A : ℕ := 30

/-- Simon's age -/
def S : ℕ := 10

/-- Theorem stating that Alvin's age is 30, given the conditions -/
theorem alvin_age : 
  (S + 5 = A / 2) → A = 30 := by
  sorry

end NUMINAMATH_CALUDE_alvin_age_l2027_202749


namespace NUMINAMATH_CALUDE_missing_element_is_five_l2027_202748

/-- Represents a 2x2 matrix --/
structure Matrix2x2 where
  a11 : ℤ
  a12 : ℤ
  a21 : ℤ
  a22 : ℤ

/-- Calculates the sum of diagonal products for a 2x2 matrix --/
def diagonalProductSum (m : Matrix2x2) : ℤ :=
  m.a11 * m.a22 + m.a12 * m.a21

/-- Theorem stating that for a matrix with given conditions, the missing element must be 5 --/
theorem missing_element_is_five (m : Matrix2x2) 
  (h1 : m.a11 = 2)
  (h2 : m.a12 = 6)
  (h3 : m.a21 = 1)
  (h4 : diagonalProductSum m = 16) :
  m.a22 = 5 := by
  sorry


end NUMINAMATH_CALUDE_missing_element_is_five_l2027_202748


namespace NUMINAMATH_CALUDE_remaining_onions_l2027_202741

/-- Given that Sally grew 5 onions, Fred grew 9 onions, and they gave Sara 4 onions,
    prove that Sally and Fred have 10 onions now. -/
theorem remaining_onions (sally_onions fred_onions given_onions : ℕ)
    (h1 : sally_onions = 5)
    (h2 : fred_onions = 9)
    (h3 : given_onions = 4) :
  sally_onions + fred_onions - given_onions = 10 := by
  sorry

end NUMINAMATH_CALUDE_remaining_onions_l2027_202741


namespace NUMINAMATH_CALUDE_unique_four_digit_number_l2027_202794

theorem unique_four_digit_number : ∃! (abcd : ℕ), 
  (1000 ≤ abcd ∧ abcd < 10000) ∧  -- 4-digit number
  (abcd % 11 = 0) ∧  -- multiple of 11
  (((abcd / 1000) * 10 + ((abcd / 100) % 10)) % 7 = 0) ∧  -- ac is multiple of 7
  ((abcd / 1000) + ((abcd / 100) % 10) + ((abcd / 10) % 10) + (abcd % 10) = (abcd % 10)^2) ∧  -- sum of digits equals square of last digit
  abcd = 3454 := by
sorry

end NUMINAMATH_CALUDE_unique_four_digit_number_l2027_202794


namespace NUMINAMATH_CALUDE_larger_circle_radius_l2027_202750

/-- The radius of a circle that is internally tangent to four externally tangent circles of radius 2 -/
theorem larger_circle_radius : ℝ := by
  -- Define the radius of the smaller circles
  let small_radius : ℝ := 2

  -- Define the number of smaller circles
  let num_small_circles : ℕ := 4

  -- Define the angle between the centers of adjacent smaller circles
  let angle_between_centers : ℝ := 360 / num_small_circles

  -- Define the radius of the larger circle
  let large_radius : ℝ := small_radius * (1 + Real.sqrt 2)

  -- Prove that the radius of the larger circle is 2(1 + √2)
  sorry

end NUMINAMATH_CALUDE_larger_circle_radius_l2027_202750


namespace NUMINAMATH_CALUDE_split_eggs_into_groups_l2027_202756

/-- The number of groups created when splitting eggs -/
def number_of_groups (total_eggs : ℕ) (eggs_per_group : ℕ) : ℕ :=
  total_eggs / eggs_per_group

/-- Theorem: Splitting 9 eggs into groups of 3 creates 3 groups -/
theorem split_eggs_into_groups :
  number_of_groups 9 3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_split_eggs_into_groups_l2027_202756


namespace NUMINAMATH_CALUDE_max_sum_in_t_grid_l2027_202795

/-- A T-shaped grid represented as a list of 5 integers -/
def TGrid := List Int

/-- Check if a T-shaped grid is valid (contains exactly the numbers 2, 5, 8, 11, 14) -/
def isValidTGrid (grid : TGrid) : Prop :=
  grid.length = 5 ∧ grid.toFinset = {2, 5, 8, 11, 14}

/-- Calculate the vertical sum of a T-shaped grid -/
def verticalSum (grid : TGrid) : Int :=
  match grid with
  | [a, b, c, _, _] => a + b + c
  | _ => 0

/-- Calculate the horizontal sum of a T-shaped grid -/
def horizontalSum (grid : TGrid) : Int :=
  match grid with
  | [_, b, _, d, e] => b + d + e
  | _ => 0

/-- Check if a T-shaped grid satisfies the sum condition -/
def satisfiesSumCondition (grid : TGrid) : Prop :=
  verticalSum grid = horizontalSum grid

/-- The main theorem: The maximum sum in a valid T-shaped grid is 33 -/
theorem max_sum_in_t_grid :
  ∀ (grid : TGrid),
    isValidTGrid grid →
    satisfiesSumCondition grid →
    (verticalSum grid ≤ 33 ∧ horizontalSum grid ≤ 33) :=
by sorry

end NUMINAMATH_CALUDE_max_sum_in_t_grid_l2027_202795


namespace NUMINAMATH_CALUDE_quadratic_always_positive_l2027_202779

theorem quadratic_always_positive (a : ℝ) : 
  (∀ x : ℝ, (5 - a) * x^2 - 6 * x + a + 5 > 0) ↔ -4 < a ∧ a < 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_always_positive_l2027_202779


namespace NUMINAMATH_CALUDE_ice_machine_cubes_l2027_202712

/-- The number of ice chests -/
def num_chests : ℕ := 7

/-- The number of ice cubes per chest -/
def cubes_per_chest : ℕ := 42

/-- The total number of ice cubes in the ice machine -/
def total_cubes : ℕ := num_chests * cubes_per_chest

/-- Theorem stating that the total number of ice cubes is 294 -/
theorem ice_machine_cubes : total_cubes = 294 := by
  sorry

end NUMINAMATH_CALUDE_ice_machine_cubes_l2027_202712


namespace NUMINAMATH_CALUDE_sum_of_digits_3n_l2027_202700

/-- Sum of decimal digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Given a natural number n where the sum of its digits is 100 and
    the sum of digits of 44n is 800, prove that the sum of digits of 3n is 300 -/
theorem sum_of_digits_3n (n : ℕ) 
  (h1 : sumOfDigits n = 100) 
  (h2 : sumOfDigits (44 * n) = 800) : 
  sumOfDigits (3 * n) = 300 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_3n_l2027_202700


namespace NUMINAMATH_CALUDE_power_64_five_sixths_l2027_202765

theorem power_64_five_sixths : (64 : ℝ) ^ (5/6) = 32 := by sorry

end NUMINAMATH_CALUDE_power_64_five_sixths_l2027_202765


namespace NUMINAMATH_CALUDE_robes_savings_l2027_202762

/-- Calculates Robe's initial savings given the repair costs and remaining savings --/
def initial_savings (repair_fee : ℕ) (corner_light_cost : ℕ) (brake_disk_cost : ℕ) (remaining_savings : ℕ) : ℕ :=
  remaining_savings + repair_fee + corner_light_cost + 2 * brake_disk_cost

theorem robes_savings :
  let repair_fee : ℕ := 10
  let corner_light_cost : ℕ := 2 * repair_fee
  let brake_disk_cost : ℕ := 3 * corner_light_cost
  let remaining_savings : ℕ := 480
  initial_savings repair_fee corner_light_cost brake_disk_cost remaining_savings = 630 := by
  sorry

#eval initial_savings 10 20 60 480

end NUMINAMATH_CALUDE_robes_savings_l2027_202762


namespace NUMINAMATH_CALUDE_graphing_calculator_count_l2027_202705

theorem graphing_calculator_count :
  ∀ (S G : ℕ),
    S + G = 45 →
    10 * S + 57 * G = 1625 →
    G = 25 :=
by
  sorry

end NUMINAMATH_CALUDE_graphing_calculator_count_l2027_202705


namespace NUMINAMATH_CALUDE_sum_of_digits_Q_is_six_l2027_202774

-- Define R_k as a function that takes k and returns the integer with k ones in base 10
def R (k : ℕ) : ℕ := (10^k - 1) / 9

-- Define Q as R_30 / R_5
def Q : ℕ := R 30 / R 5

-- Function to calculate the sum of digits of a natural number
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

-- Theorem stating that the sum of digits of Q is 6
theorem sum_of_digits_Q_is_six : sum_of_digits Q = 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_Q_is_six_l2027_202774


namespace NUMINAMATH_CALUDE_angie_necessities_contribution_l2027_202797

def salary : ℕ := 80
def taxes : ℕ := 20
def leftover : ℕ := 18

theorem angie_necessities_contribution :
  salary - taxes - leftover = 42 := by sorry

end NUMINAMATH_CALUDE_angie_necessities_contribution_l2027_202797


namespace NUMINAMATH_CALUDE_baker_cake_difference_l2027_202773

/-- Given the initial number of cakes, number of cakes sold, and number of cakes bought,
    prove that the difference between cakes bought and sold is 63. -/
theorem baker_cake_difference (initial : ℕ) (sold : ℕ) (bought : ℕ)
  (h1 : initial = 13)
  (h2 : sold = 91)
  (h3 : bought = 154) :
  bought - sold = 63 := by
  sorry

end NUMINAMATH_CALUDE_baker_cake_difference_l2027_202773


namespace NUMINAMATH_CALUDE_quadratic_root_theorem_l2027_202703

-- Define the quadratic equation
def quadratic (x k : ℝ) : ℝ := x^2 + 2*x + 3 - k

-- Define the condition for distinct real roots
def has_distinct_real_roots (k : ℝ) : Prop :=
  ∃ α β : ℝ, α ≠ β ∧ quadratic α k = 0 ∧ quadratic β k = 0

-- Define the relationship between k and the roots
def root_relationship (k α β : ℝ) : Prop :=
  k^2 = α * β + 3 * k

-- Theorem statement
theorem quadratic_root_theorem (k : ℝ) :
  has_distinct_real_roots k ∧ (∃ α β : ℝ, root_relationship k α β) → k = 3 :=
sorry

end NUMINAMATH_CALUDE_quadratic_root_theorem_l2027_202703


namespace NUMINAMATH_CALUDE_min_additional_squares_for_symmetry_l2027_202726

/-- Represents a position on the grid -/
structure Position where
  row : Nat
  col : Nat

/-- Represents the grid -/
def Grid := List Position

/-- The initially shaded squares -/
def initial_shaded : Grid := 
  [⟨1, 2⟩, ⟨3, 1⟩, ⟨4, 4⟩, ⟨6, 1⟩]

/-- Function to check if a grid has both horizontal and vertical symmetry -/
def has_symmetry (g : Grid) : Bool := sorry

/-- Function to count the number of additional squares needed for symmetry -/
def additional_squares_needed (g : Grid) : Nat := sorry

/-- Theorem stating that 8 additional squares are needed for symmetry -/
theorem min_additional_squares_for_symmetry :
  additional_squares_needed initial_shaded = 8 := by sorry

end NUMINAMATH_CALUDE_min_additional_squares_for_symmetry_l2027_202726


namespace NUMINAMATH_CALUDE_upstream_downstream_time_ratio_l2027_202755

/-- The speed of the boat in still water in kmph -/
def boat_speed : ℝ := 57

/-- The speed of the stream in kmph -/
def stream_speed : ℝ := 19

/-- The time taken to row upstream -/
def time_upstream : ℝ := sorry

/-- The time taken to row downstream -/
def time_downstream : ℝ := sorry

/-- The distance traveled (assumed to be the same for both upstream and downstream) -/
def distance : ℝ := sorry

theorem upstream_downstream_time_ratio :
  time_upstream / time_downstream = 2 := by sorry

end NUMINAMATH_CALUDE_upstream_downstream_time_ratio_l2027_202755


namespace NUMINAMATH_CALUDE_oliver_gave_janet_ten_pounds_l2027_202766

/-- The amount of candy Oliver gave to Janet -/
def candy_given_to_janet (initial_candy : ℕ) (remaining_candy : ℕ) : ℕ :=
  initial_candy - remaining_candy

/-- Proof that Oliver gave Janet 10 pounds of candy -/
theorem oliver_gave_janet_ten_pounds :
  candy_given_to_janet 78 68 = 10 := by
  sorry

end NUMINAMATH_CALUDE_oliver_gave_janet_ten_pounds_l2027_202766


namespace NUMINAMATH_CALUDE_four_students_arrangement_l2027_202780

/-- The number of ways to arrange n students in a line -/
def lineArrangements (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to arrange n students in a line with one specific student at either end -/
def arrangementsWithOneAtEnd (n : ℕ) : ℕ := 
  2 * lineArrangements (n - 1)

/-- Theorem: There are 12 ways to arrange 4 students in a line with one specific student at either end -/
theorem four_students_arrangement : arrangementsWithOneAtEnd 4 = 12 := by
  sorry

end NUMINAMATH_CALUDE_four_students_arrangement_l2027_202780


namespace NUMINAMATH_CALUDE_yogurt_production_cost_l2027_202754

/-- The price of fruit per kilogram that satisfies the yogurt production constraints -/
def fruit_price : ℝ := 2

/-- The cost of milk per liter -/
def milk_cost : ℝ := 1.5

/-- The number of liters of milk needed for one batch of yogurt -/
def milk_per_batch : ℝ := 10

/-- The number of kilograms of fruit needed for one batch of yogurt -/
def fruit_per_batch : ℝ := 3

/-- The cost to produce three batches of yogurt -/
def cost_three_batches : ℝ := 63

theorem yogurt_production_cost :
  fruit_price * fruit_per_batch * 3 + milk_cost * milk_per_batch * 3 = cost_three_batches :=
sorry

end NUMINAMATH_CALUDE_yogurt_production_cost_l2027_202754


namespace NUMINAMATH_CALUDE_discount_problem_l2027_202758

/-- Given a purchase with a 25% discount where the discount amount is $40, 
    prove that the total amount paid is $120. -/
theorem discount_problem (original_price : ℝ) (discount_rate : ℝ) (discount_amount : ℝ) (total_paid : ℝ) : 
  discount_rate = 0.25 →
  discount_amount = 40 →
  discount_amount = discount_rate * original_price →
  total_paid = original_price - discount_amount →
  total_paid = 120 := by
sorry

end NUMINAMATH_CALUDE_discount_problem_l2027_202758


namespace NUMINAMATH_CALUDE_sales_volume_function_correct_profit_at_95_yuan_max_profit_at_110_yuan_max_profit_value_l2027_202715

/-- Represents the weekly sales volume as a function of selling price -/
def sales_volume (x : ℝ) : ℝ := -10 * x + 1500

/-- Represents the weekly profit as a function of selling price -/
def profit (x : ℝ) : ℝ := (x - 80) * (sales_volume x)

/-- The cost price of each shirt -/
def cost_price : ℝ := 80

/-- The minimum allowed selling price -/
def min_price : ℝ := 90

/-- The maximum allowed selling price -/
def max_price : ℝ := 110

theorem sales_volume_function_correct :
  ∀ x, sales_volume x = -10 * x + 1500 := by sorry

theorem profit_at_95_yuan :
  profit 95 = 8250 := by sorry

theorem max_profit_at_110_yuan :
  ∀ x, min_price ≤ x ∧ x ≤ max_price → profit x ≤ profit 110 := by sorry

theorem max_profit_value :
  profit 110 = 12000 := by sorry

end NUMINAMATH_CALUDE_sales_volume_function_correct_profit_at_95_yuan_max_profit_at_110_yuan_max_profit_value_l2027_202715


namespace NUMINAMATH_CALUDE_no_real_solutions_l2027_202743

theorem no_real_solutions : ¬∃ (x : ℝ), (x^3 - x^2 - 4*x)/(x^2 + 5*x + 6) + 2*x = -6 := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_l2027_202743


namespace NUMINAMATH_CALUDE_square_side_length_from_voice_range_l2027_202763

/-- The side length of a square ground, given the area of a quarter circle
    representing the range of a trainer's voice from one corner. -/
theorem square_side_length_from_voice_range (r : ℝ) (area : ℝ) 
    (h1 : r = 140)
    (h2 : area = 15393.804002589986)
    (h3 : area = (π * r^2) / 4) : 
  ∃ (s : ℝ), s^2 = r^2 ∧ s = 140 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_from_voice_range_l2027_202763


namespace NUMINAMATH_CALUDE_smallest_number_divisible_l2027_202778

theorem smallest_number_divisible (n : ℕ) : n = 386105 ↔ 
  (∀ m : ℕ, m < n → ¬(∃ k : ℕ, m - 5 = 27 * k)) ∧
  (∀ m : ℕ, m < n → ¬(∃ k : ℕ, m - 5 = 36 * k)) ∧
  (∀ m : ℕ, m < n → ¬(∃ k : ℕ, m - 5 = 44 * k)) ∧
  (∀ m : ℕ, m < n → ¬(∃ k : ℕ, m - 5 = 52 * k)) ∧
  (∀ m : ℕ, m < n → ¬(∃ k : ℕ, m - 5 = 65 * k)) ∧
  (∃ k₁ k₂ k₃ k₄ k₅ : ℕ, n - 5 = 27 * k₁ ∧ n - 5 = 36 * k₂ ∧ n - 5 = 44 * k₃ ∧ n - 5 = 52 * k₄ ∧ n - 5 = 65 * k₅) :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_divisible_l2027_202778


namespace NUMINAMATH_CALUDE_gcd_of_45_and_75_l2027_202783

theorem gcd_of_45_and_75 : Nat.gcd 45 75 = 15 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_45_and_75_l2027_202783


namespace NUMINAMATH_CALUDE_quadratic_unique_solution_l2027_202740

theorem quadratic_unique_solution (c : ℝ) : 
  (c ≠ 0 ∧ 
   ∃! b : ℝ, b > 0 ∧ 
   ∃! x : ℝ, x^2 + (b^2 + 1/b^2) * x + c = 0) ↔ 
  (c = (1 + Real.sqrt 2) / 2 ∨ c = (1 - Real.sqrt 2) / 2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_unique_solution_l2027_202740


namespace NUMINAMATH_CALUDE_class_composition_unique_l2027_202788

/-- Represents a pair of numbers written by a student -/
structure Answer :=
  (classmates : Nat)
  (girls : Nat)

/-- Represents the class composition -/
structure ClassComposition :=
  (boys : Nat)
  (girls : Nat)

/-- Checks if an answer is valid given the actual class composition -/
def isValidAnswer (actual : ClassComposition) (answer : Answer) : Prop :=
  (answer.classmates = actual.boys + actual.girls - 1 ∧ 
   (answer.girls = actual.girls ∨ answer.girls = actual.girls + 4 ∨ answer.girls = actual.girls - 4)) ∨
  (answer.girls = actual.girls ∧ 
   (answer.classmates = actual.boys + actual.girls - 1 ∨ 
    answer.classmates = actual.boys + actual.girls + 3 ∨ 
    answer.classmates = actual.boys + actual.girls - 5))

theorem class_composition_unique :
  ∃! comp : ClassComposition,
    isValidAnswer comp ⟨15, 18⟩ ∧
    isValidAnswer comp ⟨15, 10⟩ ∧
    isValidAnswer comp ⟨12, 13⟩ ∧
    comp.boys = 16 ∧
    comp.girls = 14 := by sorry

end NUMINAMATH_CALUDE_class_composition_unique_l2027_202788


namespace NUMINAMATH_CALUDE_length_AE_l2027_202737

/-- Represents a point on a line -/
structure Point where
  x : ℝ

/-- Calculates the distance between two points -/
def distance (p q : Point) : ℝ := abs (p.x - q.x)

/-- Theorem: Length of AE given specific conditions -/
theorem length_AE (a b c d e : Point) 
  (consecutive : a.x < b.x ∧ b.x < c.x ∧ c.x < d.x ∧ d.x < e.x)
  (bc_eq_3cd : distance b c = 3 * distance c d)
  (de_eq_7 : distance d e = 7)
  (ab_eq_5 : distance a b = 5)
  (ac_eq_11 : distance a c = 11) :
  distance a e = 18 := by
  sorry

end NUMINAMATH_CALUDE_length_AE_l2027_202737


namespace NUMINAMATH_CALUDE_inequality_proof_l2027_202744

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  1 / (a * (1 + b)) + 1 / (b * (1 + c)) + 1 / (c * (1 + a)) ≥ 3 / (1 + a * b * c) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2027_202744


namespace NUMINAMATH_CALUDE_cos_1275_degrees_l2027_202772

theorem cos_1275_degrees :
  Real.cos (1275 * π / 180) = -(Real.sqrt 2 + Real.sqrt 6) / 4 := by
  sorry

end NUMINAMATH_CALUDE_cos_1275_degrees_l2027_202772


namespace NUMINAMATH_CALUDE_omega_range_for_four_zeros_l2027_202709

/-- Given a function f(x) = cos(ωx) - 1 with ω > 0, if f has exactly 4 zeros 
    in the interval [0, 2π], then 3 ≤ ω < 4. -/
theorem omega_range_for_four_zeros (ω : ℝ) (h_pos : ω > 0) : 
  (∃! (s : Finset ℝ), s.card = 4 ∧ 
    (∀ x ∈ s, 0 ≤ x ∧ x ≤ 2 * Real.pi ∧ Real.cos (ω * x) = 1) ∧
    (∀ x, 0 ≤ x ∧ x ≤ 2 * Real.pi ∧ Real.cos (ω * x) = 1 → x ∈ s)) →
  3 ≤ ω ∧ ω < 4 := by
sorry

end NUMINAMATH_CALUDE_omega_range_for_four_zeros_l2027_202709


namespace NUMINAMATH_CALUDE_percentage_calculation_l2027_202708

theorem percentage_calculation (x : ℝ) (h : 0.035 * x = 700) : 0.024 * (1.5 * x) = 720 := by
  sorry

end NUMINAMATH_CALUDE_percentage_calculation_l2027_202708


namespace NUMINAMATH_CALUDE_increasing_magnitude_l2027_202702

theorem increasing_magnitude (x : ℝ) (h : 0.85 < x ∧ x < 1.1) :
  x ≤ x + Real.sin x ∧ x + Real.sin x < x^(x^x) := by
  sorry

end NUMINAMATH_CALUDE_increasing_magnitude_l2027_202702


namespace NUMINAMATH_CALUDE_smallest_four_digit_multiple_of_112_l2027_202733

theorem smallest_four_digit_multiple_of_112 : ∃ n : ℕ, 
  (n = 1008) ∧ 
  (n ≥ 1000) ∧ 
  (n < 10000) ∧ 
  (n % 112 = 0) ∧ 
  (∀ m : ℕ, m ≥ 1000 ∧ m < 10000 ∧ m % 112 = 0 → m ≥ n) :=
sorry

end NUMINAMATH_CALUDE_smallest_four_digit_multiple_of_112_l2027_202733


namespace NUMINAMATH_CALUDE_parallelogram_circles_theorem_l2027_202728

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a circle in 2D space -/
structure Circle :=
  (center : Point) (radius : ℝ)

/-- Represents a parallelogram -/
structure Parallelogram :=
  (A B C D : Point)

/-- Checks if four points form a parallelogram -/
def is_parallelogram (p : Parallelogram) : Prop :=
  -- Add parallelogram conditions here
  sorry

/-- Checks if a circle passes through four points -/
def circle_passes_through (c : Circle) (p1 p2 p3 p4 : Point) : Prop :=
  -- Add circle condition here
  sorry

/-- Calculates the distance between two points -/
def distance (p1 p2 : Point) : ℝ :=
  -- Add distance calculation here
  sorry

theorem parallelogram_circles_theorem (ABCD : Parallelogram) (E F : Point) (ω1 ω2 : Circle) :
  is_parallelogram ABCD →
  distance ABCD.A ABCD.B > distance ABCD.B ABCD.C →
  circle_passes_through ω1 ABCD.A ABCD.D E F →
  circle_passes_through ω2 ABCD.B ABCD.C E F →
  ∃ (X Y : Point),
    distance ABCD.B X = 200 ∧
    distance X Y = 9 ∧
    distance Y ABCD.D = 80 →
    distance ABCD.B ABCD.C = 51 :=
sorry

end NUMINAMATH_CALUDE_parallelogram_circles_theorem_l2027_202728


namespace NUMINAMATH_CALUDE_min_total_time_for_three_students_l2027_202767

/-- Represents a student with their bucket filling time -/
structure Student where
  name : String
  fillTime : Real

/-- Calculates the minimum total time for students to fill their buckets -/
def minTotalTime (students : List Student) : Real :=
  sorry

/-- Theorem stating the minimum total time for the given scenario -/
theorem min_total_time_for_three_students :
  let students := [
    { name := "A", fillTime := 1.5 },
    { name := "B", fillTime := 0.5 },
    { name := "C", fillTime := 1.0 }
  ]
  minTotalTime students = 5 := by sorry

end NUMINAMATH_CALUDE_min_total_time_for_three_students_l2027_202767


namespace NUMINAMATH_CALUDE_fibonacci_100_mod_7_l2027_202776

def fibonacci : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fibonacci n + fibonacci (n + 1)

def fibonacci_mod (n : ℕ) (m : ℕ) : ℕ := fibonacci n % m

theorem fibonacci_100_mod_7 :
  ∃ (period : ℕ), period > 0 ∧
  (∀ (k : ℕ), fibonacci_mod (k + period) 7 = fibonacci_mod k 7) ∧
  period = 16 →
  fibonacci_mod 100 7 = 3 := by
sorry

end NUMINAMATH_CALUDE_fibonacci_100_mod_7_l2027_202776


namespace NUMINAMATH_CALUDE_cos_pi_minus_2alpha_l2027_202736

theorem cos_pi_minus_2alpha (α : ℝ) (h : Real.cos (π / 2 - α) = Real.sqrt 2 / 3) : 
  Real.cos (π - 2 * α) = -5 / 9 := by
  sorry

end NUMINAMATH_CALUDE_cos_pi_minus_2alpha_l2027_202736


namespace NUMINAMATH_CALUDE_find_other_number_l2027_202793

theorem find_other_number (a b : ℤ) (h1 : a - b = 8) (h2 : a = 16) : b = 8 := by
  sorry

end NUMINAMATH_CALUDE_find_other_number_l2027_202793


namespace NUMINAMATH_CALUDE_number_puzzle_l2027_202724

theorem number_puzzle : ∃ x : ℝ, ((x - 50) / 4) * 3 + 28 = 73 ∧ x = 110 := by
  sorry

end NUMINAMATH_CALUDE_number_puzzle_l2027_202724


namespace NUMINAMATH_CALUDE_garrison_reinforcement_size_l2027_202722

/-- Calculates the size of reinforcement given garrison provisions information -/
theorem garrison_reinforcement_size
  (initial_size : ℕ)
  (initial_duration : ℕ)
  (initial_consumption : ℚ)
  (time_before_reinforcement : ℕ)
  (new_consumption : ℚ)
  (additional_duration : ℕ)
  (h1 : initial_size = 2000)
  (h2 : initial_duration = 40)
  (h3 : initial_consumption = 3/2)
  (h4 : time_before_reinforcement = 20)
  (h5 : new_consumption = 2)
  (h6 : additional_duration = 10) :
  ∃ (reinforcement_size : ℕ),
    reinforcement_size = 1500 ∧
    (initial_size * initial_consumption * initial_duration : ℚ) =
    (initial_size * initial_consumption * time_before_reinforcement +
     (initial_size * initial_consumption + reinforcement_size * new_consumption) * additional_duration : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_garrison_reinforcement_size_l2027_202722


namespace NUMINAMATH_CALUDE_multiplication_puzzle_l2027_202707

def is_valid_digit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

def are_distinct (a b c d e : ℕ) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧
  c ≠ d ∧ c ≠ e ∧
  d ≠ e

def matches_pattern (a b c d e : ℕ) : Prop :=
  let abba := a * 1000 + b * 100 + b * 10 + a
  let cdea := c * 1000 + d * 100 + e * 10 + a
  let product := abba * cdea
  ∃ (x y z : ℕ),
    product = z * 100000 + b * 1000 + b * 100 + e * 10 + e ∧
    z = x * 10000 + y * 1000 + c * 100 + e * 10 + e

theorem multiplication_puzzle :
  ∀ (a b c d e : ℕ),
    is_valid_digit a → is_valid_digit b → is_valid_digit c → is_valid_digit d → is_valid_digit e →
    are_distinct a b c d e →
    matches_pattern a b c d e →
    a = 3 ∧ b = 0 ∧ c = 7 ∧ d = 2 ∧ e = 9 :=
sorry

end NUMINAMATH_CALUDE_multiplication_puzzle_l2027_202707


namespace NUMINAMATH_CALUDE_expression_simplification_l2027_202753

theorem expression_simplification (x : ℝ) (h : x = Real.sqrt 2 - 1) :
  (1 + 4 / (x - 3)) / ((x^2 + 2*x + 1) / (2*x - 6)) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2027_202753


namespace NUMINAMATH_CALUDE_joan_money_proof_l2027_202742

def dimes_to_dollars (jacket_dimes shorts_dimes : ℕ) : ℚ :=
  (jacket_dimes + shorts_dimes) * (10 : ℚ) / 100

theorem joan_money_proof (jacket_dimes shorts_dimes : ℕ) 
  (h1 : jacket_dimes = 15) (h2 : shorts_dimes = 4) : 
  dimes_to_dollars jacket_dimes shorts_dimes = 1.90 := by
  sorry

end NUMINAMATH_CALUDE_joan_money_proof_l2027_202742


namespace NUMINAMATH_CALUDE_existence_of_four_integers_l2027_202796

theorem existence_of_four_integers : ∃ (a b c d : ℤ),
  (abs a > 1000000) ∧
  (abs b > 1000000) ∧
  (abs c > 1000000) ∧
  (abs d > 1000000) ∧
  (1 / a + 1 / b + 1 / c + 1 / d : ℚ) = 1 / (a * b * c * d) :=
by sorry

end NUMINAMATH_CALUDE_existence_of_four_integers_l2027_202796


namespace NUMINAMATH_CALUDE_three_from_eight_committee_l2027_202751

/-- The number of ways to select k items from n items without replacement and where order doesn't matter. -/
def combinations (n k : ℕ) : ℕ := (n.factorial) / ((k.factorial) * ((n - k).factorial))

/-- Theorem: There are 56 ways to select 3 people from a group of 8 people where order doesn't matter. -/
theorem three_from_eight_committee : combinations 8 3 = 56 := by
  sorry

end NUMINAMATH_CALUDE_three_from_eight_committee_l2027_202751


namespace NUMINAMATH_CALUDE_min_A_over_C_l2027_202764

theorem min_A_over_C (x A C : ℝ) (hx : x > 0) (hA : A > 0) (hC : C > 0)
  (hdefA : x^2 + 1/x^2 = A) (hdefC : x + 1/x = C) :
  ∃ (m : ℝ), m = 2 * Real.sqrt 2 ∧ ∀ y, y = A / C → y ≥ m := by
  sorry

end NUMINAMATH_CALUDE_min_A_over_C_l2027_202764


namespace NUMINAMATH_CALUDE_sequence_sum_l2027_202769

theorem sequence_sum (S : ℝ) (a b : ℝ) : 
  (S - a) / 100 = 2022 →
  (S - b) / 100 = 2023 →
  (a + b) / 2 = 51 →
  S = 202301 := by
sorry

end NUMINAMATH_CALUDE_sequence_sum_l2027_202769


namespace NUMINAMATH_CALUDE_intersection_complement_equal_l2027_202718

def U : Set Int := Set.univ

def A : Set Int := {-2, -1, 1, 2}

def B : Set Int := {1, 2}

theorem intersection_complement_equal : A ∩ (Set.compl B) = {-2, -1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_complement_equal_l2027_202718


namespace NUMINAMATH_CALUDE_volume_Q_3_l2027_202725

/-- Recursive definition of polyhedron volumes -/
def Q : ℕ → ℚ
  | 0 => 8
  | (n + 1) => Q n + 4 * (1 / 27)^n

/-- The volume of Q₃ is 5972/729 -/
theorem volume_Q_3 : Q 3 = 5972 / 729 := by sorry

end NUMINAMATH_CALUDE_volume_Q_3_l2027_202725


namespace NUMINAMATH_CALUDE_merry_go_round_revolutions_l2027_202782

/-- The number of revolutions needed for the second horse to cover the same distance as the first horse on a merry-go-round. -/
theorem merry_go_round_revolutions (r₁ r₂ : ℝ) (n₁ : ℕ) (h₁ : r₁ = 30) (h₂ : r₂ = 10) (h₃ : n₁ = 25) :
  (r₁ * n₁ : ℝ) / r₂ = 75 := by
  sorry

end NUMINAMATH_CALUDE_merry_go_round_revolutions_l2027_202782


namespace NUMINAMATH_CALUDE_kelly_wendy_ratio_l2027_202786

def scholarship_problem (kelly wendy nina : ℕ) : Prop :=
  let total := 92000
  wendy = 20000 ∧
  ∃ n : ℕ, kelly = n * wendy ∧
  nina = kelly - 8000 ∧
  kelly + nina + wendy = total

theorem kelly_wendy_ratio :
  ∀ kelly wendy nina : ℕ,
  scholarship_problem kelly wendy nina →
  kelly / wendy = 2 :=
sorry

end NUMINAMATH_CALUDE_kelly_wendy_ratio_l2027_202786


namespace NUMINAMATH_CALUDE_binomial_12_3_l2027_202714

theorem binomial_12_3 : Nat.choose 12 3 = 220 := by
  sorry

end NUMINAMATH_CALUDE_binomial_12_3_l2027_202714


namespace NUMINAMATH_CALUDE_sqrt_10_factorial_div_210_l2027_202777

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem sqrt_10_factorial_div_210 : 
  Real.sqrt (factorial 10 / 210) = 72 * Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_sqrt_10_factorial_div_210_l2027_202777


namespace NUMINAMATH_CALUDE_quadratic_factorization_l2027_202729

theorem quadratic_factorization (x : ℝ) : 
  x^2 - 6*x - 6 = 0 ↔ (x - 3)^2 = 15 := by sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l2027_202729


namespace NUMINAMATH_CALUDE_inequality_properties_l2027_202719

theorem inequality_properties (a b : ℝ) (h : (1 / a) < (1 / b) ∧ (1 / b) < 0) :
  (a + b < a * b) ∧ (abs a < abs b) ∧ (b / a + a / b > 2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_properties_l2027_202719


namespace NUMINAMATH_CALUDE_total_apples_packed_l2027_202792

/-- Calculates the total number of apples packed in two weeks under specific conditions -/
theorem total_apples_packed (apples_per_box : ℕ) (boxes_per_day : ℕ) (days_per_week : ℕ) (reduced_apples : ℕ) : 
  apples_per_box = 40 →
  boxes_per_day = 50 →
  days_per_week = 7 →
  reduced_apples = 500 →
  (apples_per_box * boxes_per_day * days_per_week) + 
  ((apples_per_box * boxes_per_day - reduced_apples) * days_per_week) = 24500 := by
sorry

end NUMINAMATH_CALUDE_total_apples_packed_l2027_202792


namespace NUMINAMATH_CALUDE_number_of_elements_in_set_l2027_202739

theorem number_of_elements_in_set
  (initial_average : ℚ)
  (misread_number : ℚ)
  (correct_number : ℚ)
  (correct_average : ℚ)
  (h1 : initial_average = 18)
  (h2 : misread_number = 26)
  (h3 : correct_number = 36)
  (h4 : correct_average = 19) :
  ∃ (n : ℕ), (n : ℚ) * initial_average - misread_number = (n : ℚ) * correct_average - correct_number ∧ n = 10 :=
by sorry

end NUMINAMATH_CALUDE_number_of_elements_in_set_l2027_202739


namespace NUMINAMATH_CALUDE_square_area_from_vertices_l2027_202738

/-- The area of a square with adjacent vertices at (0,3) and (3,-4) is 58 -/
theorem square_area_from_vertices : 
  let p1 : ℝ × ℝ := (0, 3)
  let p2 : ℝ × ℝ := (3, -4)
  let side_length := Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)
  side_length^2 = 58 := by sorry

end NUMINAMATH_CALUDE_square_area_from_vertices_l2027_202738


namespace NUMINAMATH_CALUDE_exists_bijection_open_to_closed_unit_interval_l2027_202704

open Set Function Real

-- Define the open interval (0, 1)
def open_unit_interval : Set ℝ := Ioo 0 1

-- Define the closed interval [0, 1]
def closed_unit_interval : Set ℝ := Icc 0 1

-- Statement: There exists a bijective function from (0, 1) to [0, 1]
theorem exists_bijection_open_to_closed_unit_interval :
  ∃ f : ℝ → ℝ, Bijective f ∧ (∀ x, x ∈ open_unit_interval ↔ f x ∈ closed_unit_interval) :=
sorry

end NUMINAMATH_CALUDE_exists_bijection_open_to_closed_unit_interval_l2027_202704


namespace NUMINAMATH_CALUDE_perpendicular_vectors_imply_k_l2027_202706

/-- Given vectors a, b, and c in R², prove that if (a - 2b) is perpendicular to c, then k = -3 -/
theorem perpendicular_vectors_imply_k (a b c : ℝ × ℝ) (h1 : a = (Real.sqrt 3, 1))
    (h2 : b = (0, -1)) (h3 : c = (k, Real.sqrt 3)) 
    (h4 : (a - 2 • b) • c = 0) : k = -3 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_imply_k_l2027_202706


namespace NUMINAMATH_CALUDE_sum_xyz_equality_l2027_202720

theorem sum_xyz_equality (x y z : ℝ) 
  (h1 : x^2 + y^2 + z^2 = 1) 
  (h2 : x + 2*y + 3*z = Real.sqrt 14) : 
  x + y + z = (3 * Real.sqrt 14) / 7 := by
sorry

end NUMINAMATH_CALUDE_sum_xyz_equality_l2027_202720


namespace NUMINAMATH_CALUDE_yellow_balls_count_l2027_202747

theorem yellow_balls_count (total : ℕ) (red blue green yellow : ℕ) : 
  total = 500 ∧ 
  red = (total / 3 : ℕ) ∧ 
  blue = ((total - red) / 5 : ℕ) ∧ 
  green = ((total - red - blue) / 4 : ℕ) ∧ 
  yellow = total - red - blue - green →
  yellow = 201 := by
sorry

end NUMINAMATH_CALUDE_yellow_balls_count_l2027_202747


namespace NUMINAMATH_CALUDE_parallel_vectors_k_value_l2027_202730

/-- Given two vectors a and b in ℝ³, if k * a + b is parallel to 2 * a - b, then k = -2 -/
theorem parallel_vectors_k_value (a b : ℝ × ℝ × ℝ) (k : ℝ) 
    (h1 : a = (1, 1, 0)) 
    (h2 : b = (-1, 0, -2)) 
    (h_parallel : ∃ (t : ℝ), t • (k • a + b) = 2 • a - b) : 
  k = -2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_k_value_l2027_202730


namespace NUMINAMATH_CALUDE_f_composition_of_three_l2027_202727

def f (x : ℝ) : ℝ := 3 * x + 2

theorem f_composition_of_three : f (f (f 3)) = 107 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_of_three_l2027_202727


namespace NUMINAMATH_CALUDE_wedding_catering_calculation_l2027_202723

/-- Calculates the total number of items needed for a wedding reception -/
theorem wedding_catering_calculation 
  (bridgette_guests : ℕ) 
  (alex_guests : ℕ) 
  (extra_plates : ℕ) 
  (tomatoes_per_salad : ℕ) 
  (asparagus_regular : ℕ) 
  (asparagus_large : ℕ) 
  (large_portion_percent : ℚ) 
  (blueberries_per_dessert : ℕ) 
  (raspberries_per_dessert : ℕ) 
  (blackberries_per_dessert : ℕ) 
  (h1 : bridgette_guests = 84)
  (h2 : alex_guests = (2 * bridgette_guests) / 3)
  (h3 : extra_plates = 10)
  (h4 : tomatoes_per_salad = 5)
  (h5 : asparagus_regular = 8)
  (h6 : asparagus_large = 12)
  (h7 : large_portion_percent = 1/10)
  (h8 : blueberries_per_dessert = 15)
  (h9 : raspberries_per_dessert = 8)
  (h10 : blackberries_per_dessert = 10) :
  ∃ (cherry_tomatoes asparagus_spears blueberries raspberries blackberries : ℕ),
    cherry_tomatoes = 750 ∧ 
    asparagus_spears = 1260 ∧ 
    blueberries = 2250 ∧ 
    raspberries = 1200 ∧ 
    blackberries = 1500 := by
  sorry


end NUMINAMATH_CALUDE_wedding_catering_calculation_l2027_202723


namespace NUMINAMATH_CALUDE_find_number_l2027_202791

theorem find_number : ∃ x : ℝ, 0.5 * x = 0.4 * 120 + 180 ∧ x = 456 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l2027_202791


namespace NUMINAMATH_CALUDE_priyanka_value_l2027_202732

/-- A system representing the values of individuals --/
structure ValueSystem where
  Neha : ℕ
  Sonali : ℕ
  Priyanka : ℕ
  Sadaf : ℕ
  Tanu : ℕ

/-- The theorem stating Priyanka's value in the given system --/
theorem priyanka_value (sys : ValueSystem) 
  (h1 : sys.Sonali = 15)
  (h2 : sys.Priyanka = 15)
  (h3 : sys.Sadaf = sys.Neha)
  (h4 : sys.Tanu = sys.Neha) :
  sys.Priyanka = 15 := by
    sorry

end NUMINAMATH_CALUDE_priyanka_value_l2027_202732


namespace NUMINAMATH_CALUDE_fixed_point_of_power_plus_one_l2027_202713

/-- The function f(x) = x^n + 1 has a fixed point at (1, 2) for any positive integer n. -/
theorem fixed_point_of_power_plus_one (n : ℕ+) :
  let f : ℝ → ℝ := fun x ↦ x^(n : ℕ) + 1
  f 1 = 2 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_of_power_plus_one_l2027_202713


namespace NUMINAMATH_CALUDE_coprime_35_58_in_base_l2027_202745

/-- Two natural numbers are coprime if their greatest common divisor is 1. -/
def Coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

/-- A numeral system base is valid if it's greater than 1. -/
def ValidBase (base : ℕ) : Prop := base > 1

theorem coprime_35_58_in_base (base : ℕ) (h : ValidBase base) (h_base : base > 8) :
  Coprime 35 58 := by
  sorry

#check coprime_35_58_in_base

end NUMINAMATH_CALUDE_coprime_35_58_in_base_l2027_202745


namespace NUMINAMATH_CALUDE_trapezoid_lines_parallel_or_concurrent_l2027_202734

/-- A point in the Euclidean plane -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- A line in the Euclidean plane -/
structure Line :=
  (a : ℝ) (b : ℝ) (c : ℝ)

/-- Trapezoid ABCD with diagonals intersecting at E -/
structure Trapezoid :=
  (A B C D E : Point)
  (AB_parallel_CD : Line)
  (AC_diagonal : Line)
  (BD_diagonal : Line)
  (E_on_AC_and_BD : Prop)

/-- P is the foot of altitude from A to BC -/
def altitude_foot_P (trap : Trapezoid) : Point :=
  sorry

/-- Q is the foot of altitude from B to AD -/
def altitude_foot_Q (trap : Trapezoid) : Point :=
  sorry

/-- F is the intersection of circumcircles of CEQ and DEP -/
def point_F (trap : Trapezoid) (P Q : Point) : Point :=
  sorry

/-- Line through two points -/
def line_through (P Q : Point) : Line :=
  sorry

/-- Check if three lines are parallel or concurrent -/
def parallel_or_concurrent (l₁ l₂ l₃ : Line) : Prop :=
  sorry

theorem trapezoid_lines_parallel_or_concurrent (trap : Trapezoid) :
  let P := altitude_foot_P trap
  let Q := altitude_foot_Q trap
  let F := point_F trap P Q
  let AP := line_through trap.A P
  let BQ := line_through trap.B Q
  let EF := line_through trap.E F
  parallel_or_concurrent AP BQ EF :=
sorry

end NUMINAMATH_CALUDE_trapezoid_lines_parallel_or_concurrent_l2027_202734


namespace NUMINAMATH_CALUDE_complex_cube_root_l2027_202721

theorem complex_cube_root (a b : ℕ+) (h : (↑a + Complex.I * ↑b) ^ 3 = 2 + 11 * Complex.I) :
  ↑a + Complex.I * ↑b = 2 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_cube_root_l2027_202721


namespace NUMINAMATH_CALUDE_total_wax_required_l2027_202761

/-- Given the amount of wax already available and the additional amount needed,
    calculate the total wax required for the feathers. -/
theorem total_wax_required 
  (wax_available : ℕ) 
  (wax_needed : ℕ) 
  (h1 : wax_available = 331) 
  (h2 : wax_needed = 22) : 
  wax_available + wax_needed = 353 := by
  sorry

end NUMINAMATH_CALUDE_total_wax_required_l2027_202761


namespace NUMINAMATH_CALUDE_tan_neg_seven_pi_sixths_l2027_202768

theorem tan_neg_seven_pi_sixths : 
  Real.tan (-7 * π / 6) = -Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_neg_seven_pi_sixths_l2027_202768


namespace NUMINAMATH_CALUDE_Q_when_b_is_one_Q_subset_P_iff_b_in_range_l2027_202717

-- Define the sets P and Q
def P : Set ℝ := {x | x^2 - 5*x + 4 ≤ 0}
def Q (b : ℝ) : Set ℝ := {x | x^2 - (b+2)*x + 2*b ≤ 0}

-- Theorem 1: When b = 1, Q = {x | 1 ≤ x ≤ 2}
theorem Q_when_b_is_one : Q 1 = {x : ℝ | 1 ≤ x ∧ x ≤ 2} := by sorry

-- Theorem 2: Q ⊆ P if and only if b ∈ [1, 4]
theorem Q_subset_P_iff_b_in_range : ∀ b : ℝ, Q b ⊆ P ↔ 1 ≤ b ∧ b ≤ 4 := by sorry

end NUMINAMATH_CALUDE_Q_when_b_is_one_Q_subset_P_iff_b_in_range_l2027_202717


namespace NUMINAMATH_CALUDE_max_k_inequality_l2027_202781

theorem max_k_inequality (x y : ℝ) (hx : x ≥ 1) (hy : y ≥ 1) :
  (∃ k : ℝ, ∀ k' : ℝ, 
    (x^2 / (1 + x) + y^2 / (1 + y) + (x - 1) * (y - 1) ≥ k' * x * y) → k' ≤ k) ∧
  (x^2 / (1 + x) + y^2 / (1 + y) + (x - 1) * (y - 1) ≥ ((13 - 5 * Real.sqrt 5) / 2) * x * y) :=
by sorry

end NUMINAMATH_CALUDE_max_k_inequality_l2027_202781


namespace NUMINAMATH_CALUDE_range_of_a_for_three_roots_l2027_202785

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := -x^3 + 3*x + a

-- State the theorem
theorem range_of_a_for_three_roots (a : ℝ) :
  (∃ m n p : ℝ, m ≠ n ∧ n ≠ p ∧ m ≠ p ∧ 
    f a m = 2024 ∧ f a n = 2024 ∧ f a p = 2024) →
  2022 < a ∧ a < 2026 := by
sorry

end NUMINAMATH_CALUDE_range_of_a_for_three_roots_l2027_202785


namespace NUMINAMATH_CALUDE_quadratic_maximum_l2027_202798

/-- The quadratic function we're analyzing -/
def f (x : ℝ) : ℝ := -2 * x^2 - 8 * x + 10

/-- The point where the maximum occurs -/
def x_max : ℝ := -2

theorem quadratic_maximum :
  ∀ x : ℝ, f x ≤ f x_max :=
sorry

end NUMINAMATH_CALUDE_quadratic_maximum_l2027_202798


namespace NUMINAMATH_CALUDE_second_graders_count_l2027_202784

/-- The number of second graders wearing blue shirts -/
def second_graders : ℕ := sorry

/-- The cost of a blue shirt for second graders -/
def blue_shirt_cost : ℚ := 560 / 100

/-- The number of kindergartners -/
def kindergartners : ℕ := 101

/-- The cost of an orange shirt for kindergartners -/
def orange_shirt_cost : ℚ := 580 / 100

/-- The number of first graders -/
def first_graders : ℕ := 113

/-- The cost of a yellow shirt for first graders -/
def yellow_shirt_cost : ℚ := 500 / 100

/-- The number of third graders -/
def third_graders : ℕ := 108

/-- The cost of a green shirt for third graders -/
def green_shirt_cost : ℚ := 525 / 100

/-- The total amount spent on all shirts -/
def total_spent : ℚ := 231700 / 100

/-- Theorem stating that the number of second graders wearing blue shirts is 107 -/
theorem second_graders_count : second_graders = 107 := by
  sorry

end NUMINAMATH_CALUDE_second_graders_count_l2027_202784


namespace NUMINAMATH_CALUDE_consecutive_integers_sqrt_l2027_202760

theorem consecutive_integers_sqrt (x y : ℤ) : 
  (y = x + 1) →  -- x and y are consecutive integers
  (x < Real.sqrt 30) →  -- x < √30
  (Real.sqrt 30 < y) →  -- √30 < y
  Real.sqrt (2 * x + y) = 4 ∨ Real.sqrt (2 * x + y) = -4 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_sqrt_l2027_202760


namespace NUMINAMATH_CALUDE_quadratic_sum_of_solutions_l2027_202789

theorem quadratic_sum_of_solutions : ∃ a b : ℝ, 
  (∀ x : ℝ, x^2 - 6*x + 11 = 25 ↔ (x = a ∨ x = b)) ∧ 
  a ≥ b ∧ 
  3*a + 2*b = 15 + Real.sqrt 92 / 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_of_solutions_l2027_202789


namespace NUMINAMATH_CALUDE_product_xyz_l2027_202770

theorem product_xyz (x y z : ℝ) (h1 : x + 2 / y = 2) (h2 : y + 2 / z = 2) (h3 : z ≠ 0) : x * y * z = -4 := by
  sorry

end NUMINAMATH_CALUDE_product_xyz_l2027_202770


namespace NUMINAMATH_CALUDE_unique_arrangement_l2027_202775

/-- Represents the three types of people in the problem -/
inductive PersonType
  | TruthTeller
  | Liar
  | Diplomat

/-- Represents the three positions -/
inductive Position
  | Left
  | Middle
  | Right

/-- A person's statement about another person's type -/
structure Statement where
  speaker : Position
  subject : Position
  claimedType : PersonType

/-- The arrangement of people -/
structure Arrangement where
  left : PersonType
  middle : PersonType
  right : PersonType

def isConsistent (arr : Arrangement) (statements : List Statement) : Prop :=
  ∀ s ∈ statements,
    (s.speaker = Position.Left ∧ arr.left = PersonType.TruthTeller) ∨
    (s.speaker = Position.Left ∧ arr.left = PersonType.Diplomat) ∨
    (s.speaker = Position.Middle ∧ arr.middle = PersonType.Liar) ∨
    (s.speaker = Position.Right ∧ arr.right = PersonType.TruthTeller) →
      ((s.subject = Position.Middle ∧ s.claimedType = arr.middle) ∨
       (s.subject = Position.Right ∧ s.claimedType = arr.right))

def problemStatements : List Statement :=
  [ ⟨Position.Left, Position.Middle, PersonType.TruthTeller⟩,
    ⟨Position.Middle, Position.Middle, PersonType.Diplomat⟩,
    ⟨Position.Right, Position.Middle, PersonType.Liar⟩ ]

theorem unique_arrangement :
  ∃! arr : Arrangement,
    arr.left = PersonType.Diplomat ∧
    arr.middle = PersonType.Liar ∧
    arr.right = PersonType.TruthTeller ∧
    isConsistent arr problemStatements :=
  sorry

end NUMINAMATH_CALUDE_unique_arrangement_l2027_202775


namespace NUMINAMATH_CALUDE_tennis_balls_order_l2027_202746

theorem tennis_balls_order (white yellow : ℕ) (h1 : white = yellow)
  (h2 : (white : ℚ) / (yellow + 70 : ℚ) = 8 / 13) :
  white + yellow = 224 := by
  sorry

end NUMINAMATH_CALUDE_tennis_balls_order_l2027_202746


namespace NUMINAMATH_CALUDE_gcd_12547_23791_l2027_202752

theorem gcd_12547_23791 : Nat.gcd 12547 23791 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_12547_23791_l2027_202752


namespace NUMINAMATH_CALUDE_concatenated_numbers_problem_l2027_202790

theorem concatenated_numbers_problem : 
  ∃! (x y : ℕ), 
    100 ≤ x ∧ x < 1000 ∧ 
    100 ≤ y ∧ y < 1000 ∧ 
    1000 * x + y = 7 * x * y ∧
    x = 143 ∧ y = 143 := by
  sorry

end NUMINAMATH_CALUDE_concatenated_numbers_problem_l2027_202790


namespace NUMINAMATH_CALUDE_restaurant_bill_proof_l2027_202759

theorem restaurant_bill_proof : 
  ∀ (total_bill : ℝ),
  (∃ (individual_share : ℝ),
    -- 9 friends initially splitting the bill equally
    individual_share = total_bill / 9 ∧ 
    -- 8 friends each paying an extra $3.00
    8 * (individual_share + 3) = total_bill) →
  total_bill = 216 := by
sorry

end NUMINAMATH_CALUDE_restaurant_bill_proof_l2027_202759


namespace NUMINAMATH_CALUDE_octahedron_edge_length_is_four_l2027_202716

/-- A regular octahedron circumscribed around four identical balls -/
structure OctahedronWithBalls where
  /-- The radius of each ball -/
  ball_radius : ℝ
  /-- The edge length of the octahedron -/
  edge_length : ℝ
  /-- The condition that three balls are touching each other on the floor -/
  balls_touching : ball_radius = 2
  /-- The condition that the fourth ball rests on top of the other three -/
  fourth_ball_on_top : True

/-- The theorem stating that the edge length of the octahedron is 4 units -/
theorem octahedron_edge_length_is_four (o : OctahedronWithBalls) : o.edge_length = 4 := by
  sorry

end NUMINAMATH_CALUDE_octahedron_edge_length_is_four_l2027_202716


namespace NUMINAMATH_CALUDE_no_linear_term_iff_m_eq_two_l2027_202787

/-- The expression (2x-m)(x+1) does not contain a linear term of x if and only if m = 2 -/
theorem no_linear_term_iff_m_eq_two (x m : ℝ) : 
  (2 * x - m) * (x + 1) = 2 * x^2 - m ↔ m = 2 :=
by sorry

end NUMINAMATH_CALUDE_no_linear_term_iff_m_eq_two_l2027_202787


namespace NUMINAMATH_CALUDE_solution_set_of_even_increasing_function_l2027_202735

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def is_increasing_on_nonneg (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 ≤ x → 0 ≤ y → x ≤ y → f x ≤ f y

theorem solution_set_of_even_increasing_function
  (f : ℝ → ℝ)
  (h_even : is_even_function f)
  (h_increasing : is_increasing_on_nonneg f) :
  {x : ℝ | f x > f 1} = {x : ℝ | x > 1 ∨ x < -1} := by
  sorry

end NUMINAMATH_CALUDE_solution_set_of_even_increasing_function_l2027_202735
