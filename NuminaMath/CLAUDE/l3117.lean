import Mathlib

namespace NUMINAMATH_CALUDE_middle_card_is_four_l3117_311761

/-- Represents a valid triple of card numbers -/
def ValidTriple (a b c : ℕ) : Prop :=
  0 < a ∧ a < b ∧ b < c ∧ a + b + c = 15

/-- Predicate for uncertainty about other numbers given the left card -/
def LeftUncertain (a : ℕ) : Prop :=
  ∃ b₁ c₁ b₂ c₂, b₁ ≠ b₂ ∧ ValidTriple a b₁ c₁ ∧ ValidTriple a b₂ c₂

/-- Predicate for uncertainty about other numbers given the right card -/
def RightUncertain (c : ℕ) : Prop :=
  ∃ a₁ b₁ a₂ b₂, a₁ ≠ a₂ ∧ ValidTriple a₁ b₁ c ∧ ValidTriple a₂ b₂ c

/-- Predicate for uncertainty about other numbers given the middle card -/
def MiddleUncertain (b : ℕ) : Prop :=
  ∃ a₁ c₁ a₂ c₂, a₁ ≠ a₂ ∧ ValidTriple a₁ b c₁ ∧ ValidTriple a₂ b c₂

theorem middle_card_is_four :
  ∀ a b c : ℕ,
    ValidTriple a b c →
    (∀ x, ValidTriple x b c → LeftUncertain x) →
    (∀ z, ValidTriple a b z → RightUncertain z) →
    MiddleUncertain b →
    b = 4 := by
  sorry

end NUMINAMATH_CALUDE_middle_card_is_four_l3117_311761


namespace NUMINAMATH_CALUDE_mrs_petersons_tumblers_l3117_311714

/-- The number of tumblers bought given the price per tumbler, 
    the amount paid, and the change received. -/
def number_of_tumblers (price_per_tumbler : ℕ) (amount_paid : ℕ) (change : ℕ) : ℕ :=
  (amount_paid - change) / price_per_tumbler

/-- Theorem stating that Mrs. Petersons bought 10 tumblers -/
theorem mrs_petersons_tumblers : 
  number_of_tumblers 45 500 50 = 10 := by
  sorry

end NUMINAMATH_CALUDE_mrs_petersons_tumblers_l3117_311714


namespace NUMINAMATH_CALUDE_pet_store_profit_percentage_l3117_311727

-- Define the types of animals
inductive AnimalType
| Gecko
| Parrot
| Tarantula

-- Define the structure for animal sales
structure AnimalSale where
  animalType : AnimalType
  quantity : Nat
  purchasePrice : Nat

-- Define the bulk discount function
def bulkDiscount (quantity : Nat) : Rat :=
  if quantity ≥ 5 then 0.1 else 0

-- Define the selling price function
def sellingPrice (animalType : AnimalType) (purchasePrice : Nat) : Nat :=
  match animalType with
  | AnimalType.Gecko => 3 * purchasePrice + 5
  | AnimalType.Parrot => 2 * purchasePrice + 10
  | AnimalType.Tarantula => 4 * purchasePrice + 15

-- Define the profit percentage calculation
def profitPercentage (sales : List AnimalSale) : Rat :=
  let totalCost := sales.foldl (fun acc sale =>
    acc + sale.quantity * sale.purchasePrice * (1 - bulkDiscount sale.quantity)) 0
  let totalRevenue := sales.foldl (fun acc sale =>
    acc + sale.quantity * sellingPrice sale.animalType sale.purchasePrice) 0
  let profit := totalRevenue - totalCost
  (profit / totalCost) * 100

-- Theorem statement
theorem pet_store_profit_percentage :
  let sales := [
    { animalType := AnimalType.Gecko, quantity := 6, purchasePrice := 100 },
    { animalType := AnimalType.Parrot, quantity := 3, purchasePrice := 200 },
    { animalType := AnimalType.Tarantula, quantity := 10, purchasePrice := 50 }
  ]
  abs (profitPercentage sales - 227.67) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_pet_store_profit_percentage_l3117_311727


namespace NUMINAMATH_CALUDE_inverse_proportion_ordering_l3117_311708

theorem inverse_proportion_ordering :
  ∀ y₁ y₂ y₃ : ℝ,
  y₁ = 1 / (-1) →
  y₂ = 1 / (-2) →
  y₃ = 1 / 3 →
  y₃ > y₂ ∧ y₂ > y₁ :=
by sorry

end NUMINAMATH_CALUDE_inverse_proportion_ordering_l3117_311708


namespace NUMINAMATH_CALUDE_min_value_when_a_neg_one_range_of_expression_when_a_neg_nine_l3117_311772

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + x * |x - 2*a|

-- Part 1: Minimum value when a = -1
theorem min_value_when_a_neg_one :
  ∃ (m : ℝ), m = -1/2 ∧ ∀ (x : ℝ), f (-1) x ≥ m :=
sorry

-- Part 2: Range of x₁/x₂ + x₁ when a = -9
theorem range_of_expression_when_a_neg_nine :
  ∀ (x₁ x₂ : ℝ), x₁ < x₂ → f (-9) x₁ = f (-9) x₂ →
  (x₁/x₂ + x₁ < -16 ∨ x₁/x₂ + x₁ ≥ -4) :=
sorry

end NUMINAMATH_CALUDE_min_value_when_a_neg_one_range_of_expression_when_a_neg_nine_l3117_311772


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l3117_311749

open Set

-- Define the universal set U as ℝ
def U : Set ℝ := univ

-- Define set A
def A : Set ℝ := { x | 0 ≤ x ∧ x ≤ 2 }

-- Define set B
def B : Set ℝ := { y | ∃ x ∈ A, y = x + 1 }

-- Statement to prove
theorem intersection_A_complement_B : A ∩ (U \ B) = Icc 0 1 := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l3117_311749


namespace NUMINAMATH_CALUDE_kate_money_left_l3117_311757

def march_savings : ℕ := 27
def april_savings : ℕ := 13
def may_savings : ℕ := 28
def keyboard_cost : ℕ := 49
def mouse_cost : ℕ := 5

def total_savings : ℕ := march_savings + april_savings + may_savings
def total_spent : ℕ := keyboard_cost + mouse_cost
def money_left : ℕ := total_savings - total_spent

theorem kate_money_left : money_left = 14 := by
  sorry

end NUMINAMATH_CALUDE_kate_money_left_l3117_311757


namespace NUMINAMATH_CALUDE_friend_ate_two_slices_l3117_311713

/-- Calculates the number of slices James's friend ate given the initial number of slices,
    the number James ate, and the fact that James ate half of the remaining slices. -/
def friend_slices (total : ℕ) (james_ate : ℕ) : ℕ :=
  total - 2 * james_ate

theorem friend_ate_two_slices :
  let total := 8
  let james_ate := 3
  friend_slices total james_ate = 2 := by
  sorry

end NUMINAMATH_CALUDE_friend_ate_two_slices_l3117_311713


namespace NUMINAMATH_CALUDE_least_multiple_945_l3117_311799

-- Define a function to check if a number is a multiple of 45
def isMultipleOf45 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 45 * k

-- Define a function to get the digits of a number
def digits (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
    if m = 0 then acc else aux (m / 10) ((m % 10) :: acc)
  aux n []

-- Define a function to calculate the product of a list of numbers
def productOfList (l : List ℕ) : ℕ :=
  l.foldl (· * ·) 1

-- Define the main theorem
theorem least_multiple_945 :
  (isMultipleOf45 945) ∧
  (isMultipleOf45 (productOfList (digits 945))) ∧
  (∀ n : ℕ, n > 0 ∧ n < 945 →
    ¬(isMultipleOf45 n ∧ isMultipleOf45 (productOfList (digits n)))) :=
sorry

end NUMINAMATH_CALUDE_least_multiple_945_l3117_311799


namespace NUMINAMATH_CALUDE_repeating_decimal_division_l3117_311790

/-- Represents a repeating decimal with a two-digit repetend -/
def RepeatingDecimal (a b : ℕ) : ℚ := (a * 10 + b : ℚ) / 99

/-- The fraction representation of 0.overline(63) -/
def frac63 : ℚ := RepeatingDecimal 6 3

/-- The fraction representation of 0.overline(21) -/
def frac21 : ℚ := RepeatingDecimal 2 1

/-- Proves that the division of 0.overline(63) by 0.overline(21) equals 3 -/
theorem repeating_decimal_division : frac63 / frac21 = 3 := by sorry

end NUMINAMATH_CALUDE_repeating_decimal_division_l3117_311790


namespace NUMINAMATH_CALUDE_leading_coefficient_is_negative_eleven_l3117_311778

def polynomial (x : ℝ) : ℝ := 2 * (x^5 - 3*x^4 + 2*x^2) + 5 * (x^5 + x^4) - 6 * (3*x^5 + x^3 - x + 1)

theorem leading_coefficient_is_negative_eleven :
  ∃ (f : ℝ → ℝ), (∀ x, polynomial x = f x) ∧ 
  (∃ (a : ℝ) (g : ℝ → ℝ), a ≠ 0 ∧ (∀ x, f x = a * x^5 + g x) ∧ a = -11) :=
by sorry

end NUMINAMATH_CALUDE_leading_coefficient_is_negative_eleven_l3117_311778


namespace NUMINAMATH_CALUDE_negative_four_cubed_inequality_l3117_311748

theorem negative_four_cubed_inequality : (-4)^3 ≠ -4^3 := by
  -- Define the left-hand side
  have h1 : (-4)^3 = (-4) * (-4) * (-4) := by sorry
  -- Define the right-hand side
  have h2 : -4^3 = -(4 * 4 * 4) := by sorry
  -- Prove the inequality
  sorry

end NUMINAMATH_CALUDE_negative_four_cubed_inequality_l3117_311748


namespace NUMINAMATH_CALUDE_intersection_complement_equality_l3117_311704

open Set

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 3, 4}

theorem intersection_complement_equality : A ∩ (U \ B) = {1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_complement_equality_l3117_311704


namespace NUMINAMATH_CALUDE_min_value_quadratic_l3117_311726

theorem min_value_quadratic (x : ℝ) : x^2 + 10*x ≥ -25 ∧ ∃ y : ℝ, y^2 + 10*y = -25 := by
  sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l3117_311726


namespace NUMINAMATH_CALUDE_sum_of_divisors_143_l3117_311744

def sum_of_divisors (n : ℕ) : ℕ := sorry

theorem sum_of_divisors_143 : sum_of_divisors 143 = 168 := by sorry

end NUMINAMATH_CALUDE_sum_of_divisors_143_l3117_311744


namespace NUMINAMATH_CALUDE_desktop_revenue_is_12000_l3117_311785

/-- The revenue generated from the sale of desktop computers in Mr. Lu's store --/
def desktop_revenue (total_computers : ℕ) (laptop_price netbook_price desktop_price : ℕ) : ℕ :=
  let laptop_count := total_computers / 2
  let netbook_count := total_computers / 3
  let desktop_count := total_computers - laptop_count - netbook_count
  desktop_count * desktop_price

/-- Theorem stating the revenue from desktop computers --/
theorem desktop_revenue_is_12000 :
  desktop_revenue 72 750 500 1000 = 12000 := by
  sorry

end NUMINAMATH_CALUDE_desktop_revenue_is_12000_l3117_311785


namespace NUMINAMATH_CALUDE_lcm_18_60_l3117_311754

theorem lcm_18_60 : Nat.lcm 18 60 = 180 := by
  sorry

end NUMINAMATH_CALUDE_lcm_18_60_l3117_311754


namespace NUMINAMATH_CALUDE_gcd_180_270_l3117_311784

theorem gcd_180_270 : Nat.gcd 180 270 = 90 := by
  sorry

end NUMINAMATH_CALUDE_gcd_180_270_l3117_311784


namespace NUMINAMATH_CALUDE_smallest_integer_solution_inequality_l3117_311734

theorem smallest_integer_solution_inequality (x : ℤ) :
  (∀ y : ℤ, 3 * y ≥ y - 5 → y ≥ -2) ∧
  (3 * (-2) ≥ -2 - 5) :=
sorry

end NUMINAMATH_CALUDE_smallest_integer_solution_inequality_l3117_311734


namespace NUMINAMATH_CALUDE_min_abs_ab_for_perpendicular_lines_l3117_311745

theorem min_abs_ab_for_perpendicular_lines (a b : ℝ) : 
  (∀ x y : ℝ, x + a^2 * y + 1 = 0 ∧ (a^2 + 1) * x - b * y + 3 = 0 → 
    (1 : ℝ) + a^2 * (-b) = 0) → 
  ∃ m : ℝ, m = 2 ∧ ∀ k : ℝ, k = |a * b| → k ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_abs_ab_for_perpendicular_lines_l3117_311745


namespace NUMINAMATH_CALUDE_find_N_l3117_311718

theorem find_N : ∃! (N : ℕ), N > 0 ∧ 18^2 * 45^2 = 15^2 * N^2 ∧ N = 81 := by sorry

end NUMINAMATH_CALUDE_find_N_l3117_311718


namespace NUMINAMATH_CALUDE_divisible_by_nine_l3117_311712

theorem divisible_by_nine : ∃ (B : ℕ), B < 10 ∧ (7000 + 600 + 20 + B) % 9 = 0 := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_nine_l3117_311712


namespace NUMINAMATH_CALUDE_distance_to_specific_line_l3117_311732

/-- Polar coordinates of a point -/
structure PolarPoint where
  r : ℝ
  θ : ℝ

/-- Polar equation of a line -/
structure PolarLine where
  equation : ℝ → ℝ → Prop

/-- Distance from a point to a line -/
def distanceToLine (p : PolarPoint) (l : PolarLine) : ℝ := sorry

theorem distance_to_specific_line :
  let A : PolarPoint := ⟨2, 7 * π / 4⟩
  let L : PolarLine := ⟨fun ρ θ ↦ ρ * Real.sin (θ + π / 4) = Real.sqrt 2 / 2⟩
  distanceToLine A L = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_specific_line_l3117_311732


namespace NUMINAMATH_CALUDE_total_cost_of_suits_l3117_311779

/-- The total cost of two suits, given the cost of an off-the-rack suit and the pricing rule for a tailored suit. -/
theorem total_cost_of_suits (off_the_rack_cost : ℕ) : 
  off_the_rack_cost = 300 →
  off_the_rack_cost + (3 * off_the_rack_cost + 200) = 1400 := by
sorry

end NUMINAMATH_CALUDE_total_cost_of_suits_l3117_311779


namespace NUMINAMATH_CALUDE_f_inequality_range_l3117_311758

-- Define the function f
def f (x : ℝ) : ℝ := |x - 1| + |x + 1|

-- State the theorem
theorem f_inequality_range (a : ℝ) :
  (∀ x : ℝ, f x - a ≤ 0) ↔ a ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_f_inequality_range_l3117_311758


namespace NUMINAMATH_CALUDE_terminal_zeros_25_times_240_l3117_311767

/-- The number of terminal zeros in a positive integer -/
def terminalZeros (n : ℕ) : ℕ := sorry

/-- The prime factorization of 25 -/
def primeFactor25 : ℕ → ℕ
| 5 => 2
| _ => 0

/-- The prime factorization of 240 -/
def primeFactor240 : ℕ → ℕ
| 2 => 4
| 3 => 1
| 5 => 1
| _ => 0

theorem terminal_zeros_25_times_240 : 
  terminalZeros (25 * 240) = 3 := by sorry

end NUMINAMATH_CALUDE_terminal_zeros_25_times_240_l3117_311767


namespace NUMINAMATH_CALUDE_factorization_equality_l3117_311738

theorem factorization_equality (a x y : ℝ) :
  a^2 * (x - y) + 9 * (y - x) = (x - y) * (a + 3) * (a - 3) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l3117_311738


namespace NUMINAMATH_CALUDE_positive_x_axis_line_m_range_l3117_311743

/-- A line passing through the positive half-axis of the x-axis -/
structure PositiveXAxisLine where
  m : ℝ
  equation : ℝ → ℝ
  equation_def : ∀ x, equation x = 2 * x + m - 3
  passes_positive_x : ∃ x > 0, equation x = 0

/-- The range of m for a line passing through the positive half-axis of the x-axis -/
theorem positive_x_axis_line_m_range (line : PositiveXAxisLine) : line.m < 3 := by
  sorry


end NUMINAMATH_CALUDE_positive_x_axis_line_m_range_l3117_311743


namespace NUMINAMATH_CALUDE_arccos_one_half_l3117_311770

theorem arccos_one_half : Real.arccos (1/2) = π/3 := by
  sorry

end NUMINAMATH_CALUDE_arccos_one_half_l3117_311770


namespace NUMINAMATH_CALUDE_sophie_shopping_budget_l3117_311736

def initial_budget : ℚ := 260
def shirt_cost : ℚ := 18.5
def num_shirts : ℕ := 2
def trouser_cost : ℚ := 63
def num_additional_items : ℕ := 4

theorem sophie_shopping_budget :
  let total_spent := shirt_cost * num_shirts + trouser_cost
  let remaining_budget := initial_budget - total_spent
  remaining_budget / num_additional_items = 40 := by
  sorry

end NUMINAMATH_CALUDE_sophie_shopping_budget_l3117_311736


namespace NUMINAMATH_CALUDE_min_value_of_quadratic_l3117_311722

theorem min_value_of_quadratic (x : ℝ) :
  let z := 5 * x^2 - 20 * x + 45
  ∀ y : ℝ, z ≥ 25 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_quadratic_l3117_311722


namespace NUMINAMATH_CALUDE_number_puzzle_l3117_311762

theorem number_puzzle : ∃ x : ℝ, 3 * (2 * x + 9) = 81 := by
  sorry

end NUMINAMATH_CALUDE_number_puzzle_l3117_311762


namespace NUMINAMATH_CALUDE_sum_of_non_visible_numbers_l3117_311710

/-- Represents a standard six-sided die -/
def StandardDie : Type := Fin 6

/-- The sum of numbers on a standard six-sided die -/
def sumOfDie : ℕ := 21

/-- The total number of faces on four dice -/
def totalFaces : ℕ := 24

/-- The number of visible faces -/
def visibleFaces : ℕ := 9

/-- The list of visible numbers -/
def visibleNumbers : List ℕ := [1, 2, 3, 3, 4, 5, 5, 6, 6]

/-- The theorem stating the sum of non-visible numbers -/
theorem sum_of_non_visible_numbers :
  (4 * sumOfDie) - (visibleNumbers.sum) = 49 := by sorry

end NUMINAMATH_CALUDE_sum_of_non_visible_numbers_l3117_311710


namespace NUMINAMATH_CALUDE_gym_membership_duration_is_three_years_l3117_311760

/-- Calculates the duration of a gym membership in years given the monthly cost,
    down payment, and total cost. -/
def gym_membership_duration (monthly_cost : ℚ) (down_payment : ℚ) (total_cost : ℚ) : ℚ :=
  ((total_cost - down_payment) / monthly_cost) / 12

/-- Proves that given the specific costs, the gym membership duration is 3 years. -/
theorem gym_membership_duration_is_three_years :
  gym_membership_duration 12 50 482 = 3 := by
  sorry

#eval gym_membership_duration 12 50 482

end NUMINAMATH_CALUDE_gym_membership_duration_is_three_years_l3117_311760


namespace NUMINAMATH_CALUDE_unique_triplet_solution_l3117_311795

theorem unique_triplet_solution : 
  ∃! (m n k : ℕ), 3^n + 4^m = 5^k :=
by
  sorry

end NUMINAMATH_CALUDE_unique_triplet_solution_l3117_311795


namespace NUMINAMATH_CALUDE_power_equality_l3117_311783

theorem power_equality : 32^2 * 4^4 = 2^18 := by
  sorry

end NUMINAMATH_CALUDE_power_equality_l3117_311783


namespace NUMINAMATH_CALUDE_common_internal_tangent_length_l3117_311752

theorem common_internal_tangent_length 
  (distance_between_centers : ℝ) 
  (radius1 : ℝ) 
  (radius2 : ℝ) 
  (h1 : distance_between_centers = 50) 
  (h2 : radius1 = 7) 
  (h3 : radius2 = 10) : 
  ∃ (tangent_length : ℝ), tangent_length = Real.sqrt 2211 := by
  sorry

end NUMINAMATH_CALUDE_common_internal_tangent_length_l3117_311752


namespace NUMINAMATH_CALUDE_original_cat_count_l3117_311771

theorem original_cat_count (original_dogs original_cats current_dogs current_cats : ℕ) :
  original_dogs = original_cats / 2 →
  current_dogs = original_dogs + 20 →
  current_dogs = 2 * current_cats →
  current_cats = 20 →
  original_cats = 40 :=
by
  sorry

end NUMINAMATH_CALUDE_original_cat_count_l3117_311771


namespace NUMINAMATH_CALUDE_quadratic_symmetry_l3117_311709

theorem quadratic_symmetry 
  (a b c x₁ x₂ x₃ x₄ x₅ : ℝ) 
  (ha : a ≠ 0) 
  (hx : x₁ ≠ x₂ + x₃ + x₄ + x₅) 
  (hy : a * x₁^2 + b * x₁ + c = 5 ∧ a * (x₂ + x₃ + x₄ + x₅)^2 + b * (x₂ + x₃ + x₄ + x₅) + c = 5) :
  a * (x₁ + x₂)^2 + b * (x₁ + x₂) + c = a * (x₃ + x₄ + x₅)^2 + b * (x₃ + x₄ + x₅) + c :=
by sorry

end NUMINAMATH_CALUDE_quadratic_symmetry_l3117_311709


namespace NUMINAMATH_CALUDE_largest_number_with_equal_quotient_and_remainder_l3117_311764

theorem largest_number_with_equal_quotient_and_remainder :
  ∀ (A B C : ℕ),
    (A = 7 * B + C) →
    (B = C) →
    (C < 7) →
    A ≤ 48 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_number_with_equal_quotient_and_remainder_l3117_311764


namespace NUMINAMATH_CALUDE_min_value_expression_min_value_achievable_l3117_311787

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  ((3*a*b - 6*b + a*(1 - a))^2 + (9*b^2 + 2*a + 3*b*(1 - a))^2) / (a^2 + 9*b^2) ≥ 4 :=
by sorry

theorem min_value_achievable :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
  ((3*a*b - 6*b + a*(1 - a))^2 + (9*b^2 + 2*a + 3*b*(1 - a))^2) / (a^2 + 9*b^2) = 4 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_min_value_achievable_l3117_311787


namespace NUMINAMATH_CALUDE_karen_cookie_distribution_l3117_311765

/-- Calculates the number of cookies each person in Karen's class receives -/
def cookies_per_person (total_cookies : ℕ) (kept_cookies : ℕ) (grandparents_cookies : ℕ) (class_size : ℕ) : ℕ :=
  (total_cookies - kept_cookies - grandparents_cookies) / class_size

/-- Theorem stating that each person in Karen's class receives 2 cookies -/
theorem karen_cookie_distribution :
  cookies_per_person 50 10 8 16 = 2 := by
  sorry

#eval cookies_per_person 50 10 8 16

end NUMINAMATH_CALUDE_karen_cookie_distribution_l3117_311765


namespace NUMINAMATH_CALUDE_secret_spread_days_l3117_311723

/-- The number of people who know the secret after n days -/
def secret_spread (n : ℕ) : ℕ := (3^(n+1) - 1) / 2

/-- The number of days required for 3280 students to know the secret -/
theorem secret_spread_days : ∃ n : ℕ, secret_spread n = 3280 ∧ n = 7 := by
  sorry

end NUMINAMATH_CALUDE_secret_spread_days_l3117_311723


namespace NUMINAMATH_CALUDE_spiders_in_playground_sami_spiders_l3117_311720

theorem spiders_in_playground (ants : ℕ) (initial_ladybugs : ℕ) (departed_ladybugs : ℕ) (total_insects : ℕ) : ℕ :=
  by
  sorry

-- Definitions and conditions
def ants : ℕ := 12
def initial_ladybugs : ℕ := 8
def departed_ladybugs : ℕ := 2
def total_insects : ℕ := 21

-- Theorem statement
theorem sami_spiders : spiders_in_playground ants initial_ladybugs departed_ladybugs total_insects = 3 := by
  sorry

end NUMINAMATH_CALUDE_spiders_in_playground_sami_spiders_l3117_311720


namespace NUMINAMATH_CALUDE_exists_word_with_multiple_associations_l3117_311717

-- Define the alphabet A
def A : Type := Char

-- Define the set of all words over A
def A_star : Type := List A

-- Define the transducer T'
def T' : A_star → Set A_star := sorry

-- Define the property of a word having multiple associations
def has_multiple_associations (v : A_star) : Prop :=
  ∃ (w1 w2 : A_star), w1 ∈ T' v ∧ w2 ∈ T' v ∧ w1 ≠ w2

-- Theorem statement
theorem exists_word_with_multiple_associations :
  ∃ (v : A_star), has_multiple_associations v := by sorry

end NUMINAMATH_CALUDE_exists_word_with_multiple_associations_l3117_311717


namespace NUMINAMATH_CALUDE_reflection_of_circle_center_l3117_311798

/-- Reflects a point (x, y) about the line y = -x -/
def reflect_about_y_eq_neg_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.2, -p.1)

theorem reflection_of_circle_center :
  let original_center : ℝ × ℝ := (8, -3)
  let reflected_center : ℝ × ℝ := reflect_about_y_eq_neg_x original_center
  reflected_center = (3, -8) := by sorry

end NUMINAMATH_CALUDE_reflection_of_circle_center_l3117_311798


namespace NUMINAMATH_CALUDE_mathematicians_set_l3117_311789

-- Define the set of famous people
inductive FamousPerson
| BillGates
| Gauss
| YuanLongping
| Nobel
| ChenJingrun
| HuaLuogeng
| Gorky
| Einstein

-- Define a function to determine if a person is a mathematician
def isMathematician : FamousPerson → Prop
| FamousPerson.Gauss => True
| FamousPerson.ChenJingrun => True
| FamousPerson.HuaLuogeng => True
| _ => False

-- Define the set of all famous people
def allFamousPeople : Set FamousPerson :=
  {FamousPerson.BillGates, FamousPerson.Gauss, FamousPerson.YuanLongping,
   FamousPerson.Nobel, FamousPerson.ChenJingrun, FamousPerson.HuaLuogeng,
   FamousPerson.Gorky, FamousPerson.Einstein}

-- Theorem: The set of mathematicians is equal to {Gauss, Chen Jingrun, Hua Luogeng}
theorem mathematicians_set :
  {p ∈ allFamousPeople | isMathematician p} =
  {FamousPerson.Gauss, FamousPerson.ChenJingrun, FamousPerson.HuaLuogeng} :=
by sorry

end NUMINAMATH_CALUDE_mathematicians_set_l3117_311789


namespace NUMINAMATH_CALUDE_zeros_in_square_of_near_power_of_ten_l3117_311766

theorem zeros_in_square_of_near_power_of_ten : 
  (∃ n : ℕ, n = (10^12 - 5)^2 ∧ 
   ∃ m : ℕ, m > 0 ∧ n = m * 10^12 ∧ m % 10 ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_zeros_in_square_of_near_power_of_ten_l3117_311766


namespace NUMINAMATH_CALUDE_first_number_in_sum_l3117_311707

theorem first_number_in_sum (a b c : ℝ) 
  (sum_eq : a + b + c = 3.622) 
  (b_eq : b = 0.014) 
  (c_eq : c = 0.458) : 
  a = 3.15 := by
sorry

end NUMINAMATH_CALUDE_first_number_in_sum_l3117_311707


namespace NUMINAMATH_CALUDE_binomial_product_l3117_311719

theorem binomial_product (x : ℝ) : (5 * x - 3) * (2 * x + 4) = 10 * x^2 + 14 * x - 12 := by
  sorry

end NUMINAMATH_CALUDE_binomial_product_l3117_311719


namespace NUMINAMATH_CALUDE_shopkeeper_decks_l3117_311728

theorem shopkeeper_decks (total_red_cards : ℕ) (cards_per_deck : ℕ) (colors_per_deck : ℕ) (red_suits_per_deck : ℕ) (cards_per_suit : ℕ) : 
  total_red_cards = 182 →
  cards_per_deck = 52 →
  colors_per_deck = 2 →
  red_suits_per_deck = 2 →
  cards_per_suit = 13 →
  (total_red_cards / (red_suits_per_deck * cards_per_suit) : ℕ) = 7 := by
sorry

end NUMINAMATH_CALUDE_shopkeeper_decks_l3117_311728


namespace NUMINAMATH_CALUDE_girls_in_school_l3117_311716

/-- Proves that in a school with 1600 students, if a stratified sample of 200 students
    contains 10 fewer girls than boys, then the total number of girls in the school is 760. -/
theorem girls_in_school (total_students : ℕ) (sample_size : ℕ) (girls_in_sample : ℕ) :
  total_students = 1600 →
  sample_size = 200 →
  girls_in_sample = sample_size / 2 - 5 →
  (girls_in_sample : ℚ) / (total_students : ℚ) = (sample_size : ℚ) / (total_students : ℚ) →
  girls_in_sample * (total_students / sample_size) = 760 :=
by sorry

end NUMINAMATH_CALUDE_girls_in_school_l3117_311716


namespace NUMINAMATH_CALUDE_sum_of_integers_l3117_311702

theorem sum_of_integers (m n p q : ℕ+) : 
  m ≠ n ∧ m ≠ p ∧ m ≠ q ∧ n ≠ p ∧ n ≠ q ∧ p ≠ q →
  (7 - m) * (7 - n) * (7 - p) * (7 - q) = 4 →
  m + n + p + q = 28 := by
sorry

end NUMINAMATH_CALUDE_sum_of_integers_l3117_311702


namespace NUMINAMATH_CALUDE_outfits_count_l3117_311742

/-- The number of shirts available --/
def num_shirts : ℕ := 8

/-- The number of pairs of pants available --/
def num_pants : ℕ := 5

/-- The number of ties available --/
def num_ties : ℕ := 4

/-- The number of hats available --/
def num_hats : ℕ := 2

/-- The total number of outfit combinations --/
def total_outfits : ℕ := num_shirts * num_pants * (num_ties + 1) * (num_hats + 1)

/-- Theorem stating that the total number of outfits is 600 --/
theorem outfits_count : total_outfits = 600 := by
  sorry

end NUMINAMATH_CALUDE_outfits_count_l3117_311742


namespace NUMINAMATH_CALUDE_log_base_value_l3117_311769

theorem log_base_value (f : ℝ → ℝ) (a : ℝ) :
  (∀ x > 0, f x = Real.log x / Real.log a) →  -- Definition of f as logarithm base a
  a > 0 →                                     -- Condition: a > 0
  a ≠ 1 →                                     -- Condition: a ≠ 1
  f 9 = 2 →                                   -- Condition: f(9) = 2
  a = 3 :=                                    -- Conclusion: a = 3
by sorry

end NUMINAMATH_CALUDE_log_base_value_l3117_311769


namespace NUMINAMATH_CALUDE_region_area_l3117_311786

/-- The area of a region bounded by three circular arcs -/
theorem region_area (r : ℝ) (θ : ℝ) : 
  r > 0 → 
  θ = π / 4 → 
  let sector_area := θ / (2 * π) * π * r^2
  let triangle_area := 1 / 2 * r^2 * Real.sin θ
  3 * (sector_area - triangle_area) = 24 * π - 48 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_region_area_l3117_311786


namespace NUMINAMATH_CALUDE_darius_bucket_count_l3117_311701

/-- Represents the number of ounces in each of Darius's water buckets -/
def water_buckets : List ℕ := [11, 13, 12, 16, 10]

/-- The total amount of water in the first large bucket -/
def first_large_bucket : ℕ := 23

/-- The total amount of water in the second large bucket -/
def second_large_bucket : ℕ := 39

theorem darius_bucket_count :
  ∃ (bucket : ℕ) (remaining : List ℕ),
    bucket ∈ water_buckets ∧
    remaining = water_buckets.filter (λ x => x ≠ bucket ∧ x ≠ 10) ∧
    bucket + 10 = first_large_bucket ∧
    remaining.sum = second_large_bucket ∧
    water_buckets.length = 5 := by
  sorry

end NUMINAMATH_CALUDE_darius_bucket_count_l3117_311701


namespace NUMINAMATH_CALUDE_second_box_capacity_l3117_311703

/-- Represents the dimensions and capacity of a rectangular box -/
structure Box where
  height : ℝ
  width : ℝ
  length : ℝ
  capacity : ℝ

/-- Calculates the volume of a box -/
def boxVolume (b : Box) : ℝ := b.height * b.width * b.length

theorem second_box_capacity (box1 box2 : Box) : 
  box1.height = 1.5 ∧ 
  box1.width = 4 ∧ 
  box1.length = 6 ∧ 
  box1.capacity = 72 ∧
  box2.height = 3 * box1.height ∧
  box2.width = 2 * box1.width ∧
  box2.length = 0.5 * box1.length →
  box2.capacity = 216 := by
  sorry

end NUMINAMATH_CALUDE_second_box_capacity_l3117_311703


namespace NUMINAMATH_CALUDE_irrationality_of_pi_l3117_311768

-- Define rational numbers
def isRational (x : ℝ) : Prop := ∃ (p q : ℤ), q ≠ 0 ∧ x = p / q

-- Define irrational numbers as the complement of rational numbers
def isIrrational (x : ℝ) : Prop := ¬(isRational x)

-- Theorem statement
theorem irrationality_of_pi :
  isIrrational π ∧ isRational 0 ∧ isRational (22/7) ∧ isRational (Real.rpow 8 (1/3)) := by
  sorry


end NUMINAMATH_CALUDE_irrationality_of_pi_l3117_311768


namespace NUMINAMATH_CALUDE_parallelogram_base_length_l3117_311705

theorem parallelogram_base_length 
  (area : ℝ) 
  (height : ℝ) 
  (h1 : area = 108) 
  (h2 : height = 9) :
  area / height = 12 := by
sorry

end NUMINAMATH_CALUDE_parallelogram_base_length_l3117_311705


namespace NUMINAMATH_CALUDE_impossibleEggDivision_l3117_311792

/-- Represents the number of eggs of each type -/
structure EggCounts where
  typeA : ℕ
  typeB : ℕ
  typeC : ℕ

/-- Represents the ratio of eggs in each group -/
structure EggRatio where
  typeA : ℕ
  typeB : ℕ
  typeC : ℕ

/-- Function to check if it's possible to divide eggs into groups with a given ratio -/
def canDivideEggs (counts : EggCounts) (ratio : EggRatio) (numGroups : ℕ) : Prop :=
  ∃ (groupSize : ℕ),
    counts.typeA = numGroups * groupSize * ratio.typeA ∧
    counts.typeB = numGroups * groupSize * ratio.typeB ∧
    counts.typeC = numGroups * groupSize * ratio.typeC

/-- Theorem stating that it's impossible to divide the given eggs into 5 groups with the specified ratio -/
theorem impossibleEggDivision : 
  let counts : EggCounts := ⟨15, 12, 8⟩
  let ratio : EggRatio := ⟨2, 3, 1⟩
  let numGroups : ℕ := 5
  ¬(canDivideEggs counts ratio numGroups) := by
  sorry


end NUMINAMATH_CALUDE_impossibleEggDivision_l3117_311792


namespace NUMINAMATH_CALUDE_nested_sqrt_simplification_l3117_311777

theorem nested_sqrt_simplification : 
  Real.sqrt (9 * Real.sqrt (27 * Real.sqrt 81)) = 9 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_nested_sqrt_simplification_l3117_311777


namespace NUMINAMATH_CALUDE_jellyfish_cost_l3117_311747

theorem jellyfish_cost (jellyfish_cost eel_cost : ℝ) : 
  eel_cost = 9 * jellyfish_cost →
  jellyfish_cost + eel_cost = 200 →
  jellyfish_cost = 20 := by
sorry

end NUMINAMATH_CALUDE_jellyfish_cost_l3117_311747


namespace NUMINAMATH_CALUDE_work_completion_time_l3117_311724

/-- If a group of people can complete a work in 8 days, then twice the number of people can complete half the work in 2 days. -/
theorem work_completion_time 
  (P : ℕ) -- Number of people
  (W : ℝ) -- Amount of work
  (h : P > 0) -- Assumption that there is at least one person
  (completion_time : ℝ) -- Time to complete the work
  (h_completion : completion_time = 8) -- Given that the work is completed in 8 days
  : (2 * P) * (W / 2) / W * completion_time = 2 :=
by sorry

end NUMINAMATH_CALUDE_work_completion_time_l3117_311724


namespace NUMINAMATH_CALUDE_classroom_students_l3117_311730

theorem classroom_students (T : ℕ) (S : ℕ) (n : ℕ) : 
  (T = S / n + 24) →  -- Teacher's age is 24 years more than average student age
  (T = (T + S) / (n + 1) + 20) →  -- Teacher's age is 20 years more than average age of everyone
  (n = 5) := by  -- Number of students is 5
sorry

end NUMINAMATH_CALUDE_classroom_students_l3117_311730


namespace NUMINAMATH_CALUDE_jason_potato_eating_time_l3117_311700

/-- Given that Jason eats 3 potatoes in 20 minutes, prove that it takes him 3 hours to eat 27 potatoes. -/
theorem jason_potato_eating_time :
  let potatoes_per_20_min : ℚ := 3
  let total_potatoes : ℚ := 27
  let minutes_per_session : ℚ := 20
  let hours_to_eat_all : ℚ := (total_potatoes / potatoes_per_20_min) * (minutes_per_session / 60)
  hours_to_eat_all = 3 := by
sorry

end NUMINAMATH_CALUDE_jason_potato_eating_time_l3117_311700


namespace NUMINAMATH_CALUDE_quadratic_function_range_l3117_311794

/-- A quadratic function with a positive coefficient for the squared term -/
structure PositiveQuadraticFunction where
  f : ℝ → ℝ
  is_quadratic : ∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c
  positive_coeff : ∃ a b c : ℝ, (∀ x, f x = a * x^2 + b * x + c) ∧ a > 0

/-- The main theorem -/
theorem quadratic_function_range
  (f : PositiveQuadraticFunction)
  (h_symmetry : ∀ x : ℝ, f.f (2 + x) = f.f (2 - x))
  (h_inequality : ∀ x : ℝ, f.f (1 - 2*x^2) < f.f (1 + 2*x - x^2)) :
  {x : ℝ | -2 < x ∧ x < 0} = {x : ℝ | f.f (1 - 2*x^2) < f.f (1 + 2*x - x^2)} := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_range_l3117_311794


namespace NUMINAMATH_CALUDE_difference_of_squares_l3117_311729

theorem difference_of_squares (a : ℝ) : a^2 - 100 = (a + 10) * (a - 10) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l3117_311729


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l3117_311737

def is_geometric_sequence (a : ℕ+ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ+, a (n + 1) = q * a n

def satisfies_condition (a : ℕ+ → ℝ) : Prop :=
  ∀ n : ℕ+, a n * a (n + 3) = a (n + 1) * a (n + 2)

theorem geometric_sequence_property :
  (∀ a : ℕ+ → ℝ, is_geometric_sequence a → satisfies_condition a) ∧
  (∃ a : ℕ+ → ℝ, satisfies_condition a ∧ ¬is_geometric_sequence a) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l3117_311737


namespace NUMINAMATH_CALUDE_bottles_purchased_l3117_311706

/-- The number of large bottles purchased -/
def large_bottles : ℕ := 1380

/-- The cost of a large bottle in dollars -/
def large_bottle_cost : ℚ := 175/100

/-- The number of small bottles purchased -/
def small_bottles : ℕ := 690

/-- The cost of a small bottle in dollars -/
def small_bottle_cost : ℚ := 135/100

/-- The average price per bottle in dollars -/
def average_price : ℚ := 16163438256658595/10000000000000000

theorem bottles_purchased :
  (large_bottles * large_bottle_cost + small_bottles * small_bottle_cost) / 
  (large_bottles + small_bottles : ℚ) = average_price := by
  sorry

end NUMINAMATH_CALUDE_bottles_purchased_l3117_311706


namespace NUMINAMATH_CALUDE_answer_key_combinations_l3117_311725

/-- Represents the number of answer choices for a multiple-choice question -/
def multipleChoiceOptions : ℕ := 4

/-- Represents the number of true-false questions -/
def trueFalseQuestions : ℕ := 5

/-- Represents the number of multiple-choice questions -/
def multipleChoiceQuestions : ℕ := 2

/-- Calculates the number of valid true-false combinations -/
def validTrueFalseCombinations : ℕ := 2^trueFalseQuestions - 2

/-- Calculates the number of multiple-choice combinations -/
def multipleChoiceCombinations : ℕ := multipleChoiceOptions^multipleChoiceQuestions

/-- Theorem: The number of ways to create an answer key for the quiz is 480 -/
theorem answer_key_combinations : 
  validTrueFalseCombinations * multipleChoiceCombinations = 480 := by
  sorry

end NUMINAMATH_CALUDE_answer_key_combinations_l3117_311725


namespace NUMINAMATH_CALUDE_circle_chord_length_l3117_311793

theorem circle_chord_length (a : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 + 2*x - 2*y + a = 0 → (∃ x₀ y₀ : ℝ, x₀ + y₀ + 4 = 0 ∧ 
    (x - x₀)^2 + (y - y₀)^2 = 2^2)) → 
  a = -7 := by
sorry


end NUMINAMATH_CALUDE_circle_chord_length_l3117_311793


namespace NUMINAMATH_CALUDE_negative_square_cubed_l3117_311775

theorem negative_square_cubed (a : ℝ) : (-a^2)^3 = -a^6 := by
  sorry

end NUMINAMATH_CALUDE_negative_square_cubed_l3117_311775


namespace NUMINAMATH_CALUDE_congruence_solution_l3117_311797

theorem congruence_solution (x : ℤ) :
  x ≡ 6 [ZMOD 17] → 15 * x + 2 ≡ 7 [ZMOD 17] := by
  sorry

end NUMINAMATH_CALUDE_congruence_solution_l3117_311797


namespace NUMINAMATH_CALUDE_basketball_league_female_fraction_l3117_311751

theorem basketball_league_female_fraction :
  -- Define variables
  let last_year_males : ℕ := 30
  let last_year_total : ℝ := last_year_males + last_year_females
  let this_year_total : ℝ := 1.2 * last_year_total
  let this_year_males : ℝ := 1.1 * last_year_males
  let this_year_females : ℝ := 1.25 * last_year_females
  -- Conditions
  (this_year_total = this_year_males + this_year_females) →
  -- Conclusion
  (this_year_females / this_year_total = 25 / 36) := by
sorry

end NUMINAMATH_CALUDE_basketball_league_female_fraction_l3117_311751


namespace NUMINAMATH_CALUDE_frog_hop_probability_l3117_311759

/-- Represents a position on the 4x4 grid -/
inductive Position
| Inner : Fin 2 → Fin 2 → Position
| Edge : Fin 4 → Fin 4 → Position

/-- Represents a possible hop direction -/
inductive Direction
| Up | Down | Left | Right

/-- The grid size -/
def gridSize : Nat := 4

/-- The maximum number of hops -/
def maxHops : Nat := 5

/-- Function to determine if a position is on the edge -/
def isEdge (p : Position) : Bool :=
  match p with
  | Position.Edge _ _ => true
  | _ => false

/-- Function to perform a single hop -/
def hop (p : Position) (d : Direction) : Position :=
  sorry

/-- Function to calculate the probability of reaching an edge within n hops -/
def probReachEdge (start : Position) (n : Nat) : Rat :=
  sorry

/-- The starting position (second square in the second row) -/
def startPosition : Position := Position.Inner 1 1

/-- The main theorem to prove -/
theorem frog_hop_probability :
  probReachEdge startPosition maxHops = 94 / 256 := by
  sorry

end NUMINAMATH_CALUDE_frog_hop_probability_l3117_311759


namespace NUMINAMATH_CALUDE_ratio_antecedent_proof_l3117_311781

theorem ratio_antecedent_proof (ratio_antecedent ratio_consequent consequent : ℚ) : 
  ratio_antecedent = 4 →
  ratio_consequent = 6 →
  consequent = 75 →
  (ratio_antecedent / ratio_consequent) * consequent = 50 := by
sorry

end NUMINAMATH_CALUDE_ratio_antecedent_proof_l3117_311781


namespace NUMINAMATH_CALUDE_jesse_room_area_l3117_311735

/-- Calculates the area of a rectangle given its length and width -/
def rectangleArea (length width : ℝ) : ℝ := length * width

/-- Represents an L-shaped room with two rectangular parts -/
structure LShapedRoom where
  length1 : ℝ
  width1 : ℝ
  length2 : ℝ
  width2 : ℝ

/-- Calculates the total area of an L-shaped room -/
def totalArea (room : LShapedRoom) : ℝ :=
  rectangleArea room.length1 room.width1 + rectangleArea room.length2 room.width2

/-- Theorem: The total area of Jesse's L-shaped room is 120 square feet -/
theorem jesse_room_area :
  let room : LShapedRoom := { length1 := 12, width1 := 8, length2 := 6, width2 := 4 }
  totalArea room = 120 := by
  sorry

end NUMINAMATH_CALUDE_jesse_room_area_l3117_311735


namespace NUMINAMATH_CALUDE_f_is_integer_valued_l3117_311756

/-- The polynomial f(x) = (1/5)x^5 + (1/2)x^4 + (1/3)x^3 - (1/30)x -/
def f (x : ℚ) : ℚ := (1/5) * x^5 + (1/2) * x^4 + (1/3) * x^3 - (1/30) * x

/-- Theorem stating that f(x) is an integer-valued polynomial -/
theorem f_is_integer_valued : ∀ (x : ℤ), ∃ (y : ℤ), f x = y := by
  sorry

end NUMINAMATH_CALUDE_f_is_integer_valued_l3117_311756


namespace NUMINAMATH_CALUDE_gwen_zoo_pictures_l3117_311753

/-- The number of pictures Gwen took at the zoo -/
def zoo_pictures : ℕ := sorry

/-- The number of pictures Gwen took at the museum -/
def museum_pictures : ℕ := 29

/-- The number of pictures Gwen deleted -/
def deleted_pictures : ℕ := 15

/-- The number of pictures Gwen had after deleting -/
def remaining_pictures : ℕ := 55

/-- Theorem stating that the number of pictures Gwen took at the zoo is 41 -/
theorem gwen_zoo_pictures :
  zoo_pictures = 41 :=
by
  have h1 : zoo_pictures + museum_pictures - deleted_pictures = remaining_pictures :=
    sorry
  sorry

end NUMINAMATH_CALUDE_gwen_zoo_pictures_l3117_311753


namespace NUMINAMATH_CALUDE_joan_apples_l3117_311763

/-- The number of apples Joan picked -/
def apples_picked : ℕ := 43

/-- The number of apples Joan gave to Melanie -/
def apples_given : ℕ := 27

/-- The number of apples Joan has now -/
def apples_remaining : ℕ := apples_picked - apples_given

theorem joan_apples : apples_remaining = 16 := by
  sorry

end NUMINAMATH_CALUDE_joan_apples_l3117_311763


namespace NUMINAMATH_CALUDE_m_less_than_two_necessary_not_sufficient_l3117_311731

/-- The condition for the quadratic inequality x^2 + mx + 1 > 0 to have ℝ as its solution set -/
def has_real_solution_set (m : ℝ) : Prop :=
  ∀ x, x^2 + m*x + 1 > 0

/-- The statement that m < 2 is a necessary but not sufficient condition -/
theorem m_less_than_two_necessary_not_sufficient :
  (∀ m, has_real_solution_set m → m < 2) ∧
  ¬(∀ m, m < 2 → has_real_solution_set m) := by sorry

end NUMINAMATH_CALUDE_m_less_than_two_necessary_not_sufficient_l3117_311731


namespace NUMINAMATH_CALUDE_polynomial_division_theorem_l3117_311721

-- Define the polynomial
def P (x a b : ℝ) : ℝ := 2*x^4 - 3*x^3 + a*x^2 + 7*x + b

-- Define the divisor
def D (x : ℝ) : ℝ := x^2 + x - 2

-- Theorem statement
theorem polynomial_division_theorem (a b : ℝ) :
  (∀ x, ∃ q, P x a b = D x * q) →
  a / b = -2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_theorem_l3117_311721


namespace NUMINAMATH_CALUDE_count_numbers_greater_than_1_1_l3117_311796

theorem count_numbers_greater_than_1_1 : 
  let numbers : List ℚ := [1.4, 9/10, 1.2, 0.5, 13/10]
  (numbers.filter (λ x => x > 1.1)).length = 3 := by
  sorry

end NUMINAMATH_CALUDE_count_numbers_greater_than_1_1_l3117_311796


namespace NUMINAMATH_CALUDE_quadratic_max_value_l3117_311773

theorem quadratic_max_value :
  let f : ℝ → ℝ := fun x ↦ -3 * x^2 + 6 * x + 4
  ∃ (max : ℝ), ∀ (x : ℝ), f x ≤ max ∧ ∃ (x_max : ℝ), f x_max = max ∧ max = 7 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_max_value_l3117_311773


namespace NUMINAMATH_CALUDE_max_sum_of_seventh_powers_l3117_311741

theorem max_sum_of_seventh_powers (a b c d : ℝ) (h : a^6 + b^6 + c^6 + d^6 = 64) :
  ∃ (m : ℝ), m = 128 ∧ a^7 + b^7 + c^7 + d^7 ≤ m ∧ 
  ∃ (a' b' c' d' : ℝ), a'^6 + b'^6 + c'^6 + d'^6 = 64 ∧ a'^7 + b'^7 + c'^7 + d'^7 = m :=
by sorry

end NUMINAMATH_CALUDE_max_sum_of_seventh_powers_l3117_311741


namespace NUMINAMATH_CALUDE_product_of_digits_not_divisible_by_five_l3117_311739

def numbers : List Nat := [4825, 4835, 4845, 4855, 4865]

def is_divisible_by_five (n : Nat) : Prop :=
  n % 5 = 0

def units_digit (n : Nat) : Nat :=
  n % 10

def tens_digit (n : Nat) : Nat :=
  (n / 10) % 10

theorem product_of_digits_not_divisible_by_five :
  ∃ n ∈ numbers,
    ¬is_divisible_by_five n ∧
    ∀ m ∈ numbers, m ≠ n → is_divisible_by_five m ∧
    units_digit n * tens_digit n = 30 :=
  sorry

end NUMINAMATH_CALUDE_product_of_digits_not_divisible_by_five_l3117_311739


namespace NUMINAMATH_CALUDE_trees_planted_specific_plot_l3117_311746

/-- Calculates the number of trees planted around a rectangular plot -/
def trees_planted (length width spacing : ℕ) : ℕ :=
  let perimeter := 2 * (length + width)
  let total_intervals := perimeter / spacing
  total_intervals - 4

/-- Theorem stating the number of trees planted around the specific rectangular plot -/
theorem trees_planted_specific_plot :
  trees_planted 60 30 6 = 26 :=
by
  sorry

end NUMINAMATH_CALUDE_trees_planted_specific_plot_l3117_311746


namespace NUMINAMATH_CALUDE_intersection_M_N_l3117_311791

-- Define the sets M and N
def M : Set ℝ := {y | ∃ x, y = x^2 + 1}
def N : Set ℝ := {y | ∃ x, y = x + 1}

-- State the theorem
theorem intersection_M_N : M ∩ N = {y | y ≥ 1} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l3117_311791


namespace NUMINAMATH_CALUDE_max_sum_with_negative_l3117_311776

def S : Finset Int := {-7, -5, -3, 0, 2, 4, 6}

def is_valid_selection (a b c : Int) : Prop :=
  a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ (a < 0 ∨ b < 0 ∨ c < 0)

theorem max_sum_with_negative :
  ∃ (a b c : Int), is_valid_selection a b c ∧
    a + b + c = 7 ∧
    ∀ (x y z : Int), is_valid_selection x y z → x + y + z ≤ 7 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_with_negative_l3117_311776


namespace NUMINAMATH_CALUDE_seven_people_six_seats_l3117_311740

/-- The number of ways to seat 6 people from a group of 7 at a circular table -/
def seating_arrangements (n : ℕ) (k : ℕ) : ℕ :=
  n * Nat.factorial (k - 1)

/-- Theorem stating the number of seating arrangements for 7 people at a circular table with 6 seats -/
theorem seven_people_six_seats :
  seating_arrangements 7 6 = 840 := by
  sorry

end NUMINAMATH_CALUDE_seven_people_six_seats_l3117_311740


namespace NUMINAMATH_CALUDE_a_negative_sufficient_not_necessary_l3117_311733

def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 2 * x + 1

theorem a_negative_sufficient_not_necessary :
  (∀ a : ℝ, a < 0 → ∃ x : ℝ, x < 0 ∧ f a x = 0) ∧
  (∃ a : ℝ, a ≥ 0 ∧ ∃ x : ℝ, x < 0 ∧ f a x = 0) := by
  sorry

end NUMINAMATH_CALUDE_a_negative_sufficient_not_necessary_l3117_311733


namespace NUMINAMATH_CALUDE_complex_number_simplification_l3117_311715

theorem complex_number_simplification :
  (6 - 3*Complex.I) + 3*(2 - 7*Complex.I) = 12 - 24*Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_number_simplification_l3117_311715


namespace NUMINAMATH_CALUDE_only_B_incorrect_l3117_311750

-- Define the polynomial
def polynomial (a : ℝ) : ℝ := 8 * a

-- Define the options
def optionA (a : ℝ) : ℝ := 8 * a
def optionB (a : ℝ) : ℝ := 0.92 * a
def optionC (a : ℝ) : ℝ := 8 * a
def optionD (a : ℝ) : ℝ := 8 * a

-- Theorem stating that only option B is incorrect
theorem only_B_incorrect (a : ℝ) : 
  optionA a = polynomial a ∧
  optionB a ≠ polynomial a ∧
  optionC a = polynomial a ∧
  optionD a = polynomial a :=
by sorry

end NUMINAMATH_CALUDE_only_B_incorrect_l3117_311750


namespace NUMINAMATH_CALUDE_tangent_line_passes_through_fixed_point_l3117_311711

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define the line l
def line_l (x : ℝ) : Prop := x = 2

-- Define a point P on line l
def point_P (t : ℝ) : ℝ × ℝ := (2, t)

-- Define the equation of the common chord AB
def common_chord (t x y : ℝ) : Prop := 2*x + t*y = 1

-- Theorem statement
theorem tangent_line_passes_through_fixed_point :
  ∀ t : ℝ, ∃ A B : ℝ × ℝ,
    circle_C A.1 A.2 ∧
    circle_C B.1 B.2 ∧
    common_chord t A.1 A.2 ∧
    common_chord t B.1 B.2 ∧
    common_chord t (1/2) 0 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_passes_through_fixed_point_l3117_311711


namespace NUMINAMATH_CALUDE_speed_upstream_l3117_311788

theorem speed_upstream (boat_speed : ℝ) (current_speed : ℝ) (h1 : boat_speed = 60) (h2 : current_speed = 17) :
  boat_speed - current_speed = 43 := by
  sorry

end NUMINAMATH_CALUDE_speed_upstream_l3117_311788


namespace NUMINAMATH_CALUDE_parabola_directrix_l3117_311782

/-- The equation of a parabola -/
def parabola_eq (x y : ℝ) : Prop := y = 4 * x^2 + 4 * x + 1

/-- The equation of the directrix -/
def directrix_eq (y : ℝ) : Prop := y = 11/16

/-- Theorem stating that the given directrix is correct for the parabola -/
theorem parabola_directrix : 
  ∀ x y : ℝ, parabola_eq x y → ∃ d : ℝ, directrix_eq d ∧ 
  (∀ p : ℝ × ℝ, p.1 = x ∧ p.2 = y → 
   ∃ f : ℝ × ℝ, (p.1 - f.1)^2 + (p.2 - f.2)^2 = (p.2 - d)^2) :=
sorry

end NUMINAMATH_CALUDE_parabola_directrix_l3117_311782


namespace NUMINAMATH_CALUDE_quadratic_factorization_l3117_311755

theorem quadratic_factorization (E F : ℤ) :
  (∀ y : ℝ, 15 * y^2 - 82 * y + 48 = (E * y - 16) * (F * y - 3)) →
  E * F + E = 20 := by
sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l3117_311755


namespace NUMINAMATH_CALUDE_julia_spent_114_on_animal_food_l3117_311780

/-- The total amount spent on animal food --/
def total_spent (weekly_total : ℕ) (rabbit_food_cost : ℕ) (rabbit_weeks : ℕ) (parrot_weeks : ℕ) : ℕ :=
  (weekly_total - rabbit_food_cost) * parrot_weeks + rabbit_food_cost * rabbit_weeks

/-- Proof that Julia spent $114 on animal food --/
theorem julia_spent_114_on_animal_food :
  total_spent 30 12 5 3 = 114 := by
  sorry

end NUMINAMATH_CALUDE_julia_spent_114_on_animal_food_l3117_311780


namespace NUMINAMATH_CALUDE_smallest_fraction_between_l3117_311774

theorem smallest_fraction_between (p q : ℕ+) : 
  (3 : ℚ) / 5 < (p : ℚ) / q ∧ 
  (p : ℚ) / q < (2 : ℚ) / 3 ∧ 
  (∀ (p' q' : ℕ+), (3 : ℚ) / 5 < (p' : ℚ) / q' ∧ (p' : ℚ) / q' < (2 : ℚ) / 3 → q ≤ q') →
  q - p = 3 := by
sorry

end NUMINAMATH_CALUDE_smallest_fraction_between_l3117_311774
