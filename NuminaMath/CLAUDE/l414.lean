import Mathlib

namespace NUMINAMATH_CALUDE_total_machines_is_five_l414_41480

/-- Represents the production scenario with new and old machines -/
structure ProductionScenario where
  totalProduction : ℕ
  newMachineProduction : ℕ
  oldMachineProduction : ℕ
  totalMachines : ℕ

/-- Represents the conditions of the production problem -/
def productionProblem : ProductionScenario → Prop
  | s => s.totalProduction = 9000 ∧
         s.oldMachineProduction = s.newMachineProduction / 2 ∧
         s.totalProduction = (s.totalMachines - 1) * s.newMachineProduction + s.oldMachineProduction

/-- Represents the scenario if the old machine is replaced -/
def replacedScenario (s : ProductionScenario) : ProductionScenario :=
  { totalProduction := s.totalProduction
  , newMachineProduction := s.newMachineProduction - 200
  , oldMachineProduction := s.newMachineProduction - 200
  , totalMachines := s.totalMachines }

/-- The main theorem stating that the total number of machines is 5 -/
theorem total_machines_is_five :
  ∃ s : ProductionScenario, productionProblem s ∧
    productionProblem (replacedScenario s) ∧
    s.totalMachines = 5 := by
  sorry


end NUMINAMATH_CALUDE_total_machines_is_five_l414_41480


namespace NUMINAMATH_CALUDE_expression_value_at_five_l414_41400

theorem expression_value_at_five :
  let x : ℚ := 5
  (x^2 + x - 12) / (x - 4) = 18 :=
by sorry

end NUMINAMATH_CALUDE_expression_value_at_five_l414_41400


namespace NUMINAMATH_CALUDE_A_intersect_B_eq_unit_interval_l414_41415

-- Define set A
def A : Set ℝ := {x | x + 1 > 0}

-- Define set B
def B : Set ℝ := {y | ∃ x, y = Real.sqrt (1 - 2^x)}

-- Theorem stating that the intersection of A and B is [0, 1)
theorem A_intersect_B_eq_unit_interval :
  A ∩ B = Set.Icc 0 1 := by sorry

end NUMINAMATH_CALUDE_A_intersect_B_eq_unit_interval_l414_41415


namespace NUMINAMATH_CALUDE_local_minimum_implies_b_range_l414_41407

-- Define the function f(x)
def f (b : ℝ) (x : ℝ) : ℝ := x^3 - 3*b*x + 3*b

-- State the theorem
theorem local_minimum_implies_b_range :
  ∀ b : ℝ, (∃ x : ℝ, x ∈ (Set.Ioo 0 1) ∧ IsLocalMin (f b) x) → 0 < b ∧ b < 1 :=
by sorry

end NUMINAMATH_CALUDE_local_minimum_implies_b_range_l414_41407


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l414_41413

/-- The equation of a hyperbola -/
def hyperbola_equation (x y : ℝ) : Prop :=
  y^2 / 4 - x^2 / 9 = 1

/-- The equation of the asymptotes -/
def asymptote_equation (x y : ℝ) : Prop :=
  y = 2/3 * x ∨ y = -2/3 * x

/-- Theorem: The asymptotes of the given hyperbola are y = ±(2/3)x -/
theorem hyperbola_asymptotes :
  ∀ x y : ℝ, hyperbola_equation x y → asymptote_equation x y :=
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l414_41413


namespace NUMINAMATH_CALUDE_new_arithmetic_mean_l414_41472

def original_count : ℕ := 60
def original_mean : ℝ := 45
def removed_numbers : List ℝ := [48, 58, 62]

theorem new_arithmetic_mean :
  let original_sum : ℝ := original_count * original_mean
  let removed_sum : ℝ := removed_numbers.sum
  let new_count : ℕ := original_count - removed_numbers.length
  let new_sum : ℝ := original_sum - removed_sum
  new_sum / new_count = 44.42 := by sorry

end NUMINAMATH_CALUDE_new_arithmetic_mean_l414_41472


namespace NUMINAMATH_CALUDE_apple_eating_duration_l414_41452

/-- The number of apples Eva needs to buy -/
def total_apples : ℕ := 14

/-- The number of days in a week -/
def days_per_week : ℕ := 7

/-- The number of weeks Eva should eat an apple every day -/
def weeks_to_eat_apples : ℚ := total_apples / days_per_week

theorem apple_eating_duration : weeks_to_eat_apples = 2 := by
  sorry

end NUMINAMATH_CALUDE_apple_eating_duration_l414_41452


namespace NUMINAMATH_CALUDE_friends_coming_over_l414_41463

theorem friends_coming_over (sandwiches_per_friend : ℕ) (total_sandwiches : ℕ) 
  (h1 : sandwiches_per_friend = 3) 
  (h2 : total_sandwiches = 12) : 
  total_sandwiches / sandwiches_per_friend = 4 :=
by sorry

end NUMINAMATH_CALUDE_friends_coming_over_l414_41463


namespace NUMINAMATH_CALUDE_largest_even_from_powerful_digits_l414_41438

/-- A natural number is powerful if n + (n+1) + (n+2) has no carrying over --/
def isPowerful (n : ℕ) : Prop :=
  ∀ d : ℕ, d > 0 → (n / 10^d % 10 + (n+1) / 10^d % 10 + (n+2) / 10^d % 10) < 10

/-- The set of powerful numbers less than 1000 --/
def powerfulSet : Set ℕ := {n | n < 1000 ∧ isPowerful n}

/-- The set of digits from powerful numbers less than 1000 --/
def powerfulDigits : Set ℕ := {d | ∃ n ∈ powerfulSet, ∃ k, n / 10^k % 10 = d}

/-- An even number formed by non-repeating digits from powerfulDigits --/
def validNumber (n : ℕ) : Prop :=
  n % 2 = 0 ∧ 
  (∀ d, d ∈ powerfulDigits → (∃! k, n / 10^k % 10 = d)) ∧
  (∀ k, n / 10^k % 10 ∈ powerfulDigits)

theorem largest_even_from_powerful_digits :
  ∃ n, validNumber n ∧ ∀ m, validNumber m → m ≤ n ∧ n = 43210 :=
sorry

end NUMINAMATH_CALUDE_largest_even_from_powerful_digits_l414_41438


namespace NUMINAMATH_CALUDE_greg_earnings_l414_41401

/-- Represents the rates and walking details for Greg's dog walking business -/
structure DogWalkingRates where
  small_base : ℝ := 15
  small_per_minute : ℝ := 1
  medium_base : ℝ := 20
  medium_per_minute : ℝ := 1.25
  large_base : ℝ := 25
  large_per_minute : ℝ := 1.5
  small_dogs : ℕ := 3
  small_minutes : ℕ := 12
  medium_dogs : ℕ := 2
  medium_minutes : ℕ := 18
  large_dogs : ℕ := 1
  large_minutes : ℕ := 25

/-- Calculates Greg's total earnings from dog walking -/
def calculateEarnings (rates : DogWalkingRates) : ℝ :=
  (rates.small_base * rates.small_dogs + rates.small_per_minute * rates.small_dogs * rates.small_minutes) +
  (rates.medium_base * rates.medium_dogs + rates.medium_per_minute * rates.medium_dogs * rates.medium_minutes) +
  (rates.large_base * rates.large_dogs + rates.large_per_minute * rates.large_dogs * rates.large_minutes)

/-- Theorem stating that Greg's total earnings are $228.50 -/
theorem greg_earnings (rates : DogWalkingRates) : calculateEarnings rates = 228.5 := by
  sorry

end NUMINAMATH_CALUDE_greg_earnings_l414_41401


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l414_41419

theorem imaginary_part_of_z (z : ℂ) (h : (1 + Complex.I) * z = 1 - Complex.I) : 
  Complex.im z = -1 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l414_41419


namespace NUMINAMATH_CALUDE_min_sum_of_squares_l414_41469

theorem min_sum_of_squares (x₁ x₂ x₃ : ℝ) 
  (h_pos : x₁ > 0 ∧ x₂ > 0 ∧ x₃ > 0) 
  (h_sum : x₁ + 3*x₂ + 4*x₃ = 100) : 
  ∀ y₁ y₂ y₃ : ℝ, y₁ > 0 ∧ y₂ > 0 ∧ y₃ > 0 → y₁ + 3*y₂ + 4*y₃ = 100 → 
  x₁^2 + x₂^2 + x₃^2 ≤ y₁^2 + y₂^2 + y₃^2 ∧ 
  ∃ z₁ z₂ z₃ : ℝ, z₁ > 0 ∧ z₂ > 0 ∧ z₃ > 0 ∧ z₁ + 3*z₂ + 4*z₃ = 100 ∧ 
  z₁^2 + z₂^2 + z₃^2 = 5000/13 := by
sorry

end NUMINAMATH_CALUDE_min_sum_of_squares_l414_41469


namespace NUMINAMATH_CALUDE_rotated_semicircle_area_l414_41435

/-- The area of a shaded figure formed by rotating a semicircle -/
theorem rotated_semicircle_area (R : ℝ) (h : R > 0) :
  let α : ℝ := 20 * π / 180  -- Convert 20° to radians
  let semicircle_area : ℝ := π * R^2 / 2
  let sector_area : ℝ := 2 * R^2 * α / 2
  sector_area = 2 * π * R^2 / 9 :=
by sorry

end NUMINAMATH_CALUDE_rotated_semicircle_area_l414_41435


namespace NUMINAMATH_CALUDE_sum_of_roots_l414_41417

theorem sum_of_roots (p q r s : ℝ) : 
  p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s →
  (∀ x : ℝ, x^2 - 12*p*x - 8*q = 0 ↔ x = r ∨ x = s) →
  (∀ x : ℝ, x^2 - 12*r*x - 8*s = 0 ↔ x = p ∨ x = q) →
  p + q + r + s = 1248 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_l414_41417


namespace NUMINAMATH_CALUDE_tips_fraction_is_one_third_l414_41461

/-- Represents the income of a waitress -/
structure WaitressIncome where
  salary : ℚ
  tips : ℚ

/-- The fraction of income that comes from tips -/
def tipFraction (income : WaitressIncome) : ℚ :=
  income.tips / (income.salary + income.tips)

/-- Theorem: If tips are 2/4 of salary, then 1/3 of income is from tips -/
theorem tips_fraction_is_one_third
  (income : WaitressIncome)
  (h : income.tips = (2 : ℚ) / 4 * income.salary) :
  tipFraction income = 1 / 3 := by
  sorry

#eval (1 : ℚ) / 3  -- To check the result

end NUMINAMATH_CALUDE_tips_fraction_is_one_third_l414_41461


namespace NUMINAMATH_CALUDE_simultaneous_inequalities_l414_41444

theorem simultaneous_inequalities (a b : ℝ) :
  (a < b ∧ 1 / a < 1 / b) ↔ a < 0 ∧ 0 < b := by
  sorry

end NUMINAMATH_CALUDE_simultaneous_inequalities_l414_41444


namespace NUMINAMATH_CALUDE_transport_speed_problem_l414_41485

/-- Proves that given two transports traveling in opposite directions for 2.71875 hours,
    with one transport traveling at 68 mph, and ending up 348 miles apart,
    the speed of the other transport must be 60 mph. -/
theorem transport_speed_problem (speed_b : ℝ) (time : ℝ) (distance : ℝ) (speed_a : ℝ) : 
  speed_b = 68 →
  time = 2.71875 →
  distance = 348 →
  (speed_a + speed_b) * time = distance →
  speed_a = 60 := by sorry

end NUMINAMATH_CALUDE_transport_speed_problem_l414_41485


namespace NUMINAMATH_CALUDE_integer_triple_divisibility_l414_41453

theorem integer_triple_divisibility : 
  {(a, b, c) : ℤ × ℤ × ℤ | 1 < a ∧ a < b ∧ b < c ∧ (abc - 1) % ((a - 1) * (b - 1) * (c - 1)) = 0} = 
  {(3, 5, 15), (2, 4, 8)} := by sorry

end NUMINAMATH_CALUDE_integer_triple_divisibility_l414_41453


namespace NUMINAMATH_CALUDE_cube_division_theorem_l414_41493

/-- Represents the volume of the remaining solid after removal of marked cubes --/
def remaining_volume (k : ℕ) : ℚ :=
  if k % 2 = 0 then 1/2
  else (k+1)^2 * (2*k-1) / (4*k^3)

/-- Represents the surface area of the remaining solid after removal of marked cubes --/
def remaining_surface_area (k : ℕ) : ℚ :=
  if k % 2 = 0 then 3*(k+1) / 2
  else 3*(k+1)^2 / (2*k)

theorem cube_division_theorem (k : ℕ) :
  (k ≥ 65 → remaining_surface_area k > 100) ∧
  (k % 2 = 0 → remaining_volume k ≤ 1/2) :=
sorry

end NUMINAMATH_CALUDE_cube_division_theorem_l414_41493


namespace NUMINAMATH_CALUDE_valid_outfit_count_l414_41499

/-- Represents the colors available for clothing items -/
inductive Color
  | Tan
  | Black
  | Blue
  | Gray
  | Green
  | White
  | Yellow

/-- Represents a clothing item -/
structure ClothingItem where
  color : Color

/-- Represents an outfit -/
structure Outfit where
  shirt : ClothingItem
  pants : ClothingItem
  hat : ClothingItem

def is_valid_outfit (o : Outfit) : Prop :=
  o.shirt.color ≠ o.pants.color ∧ o.hat.color ≠ o.pants.color

def num_shirts : Nat := 8
def num_pants : Nat := 5
def num_hats : Nat := 8

def pants_colors : List Color := [Color.Tan, Color.Black, Color.Blue, Color.Gray, Color.Green]

theorem valid_outfit_count :
  (∃ (valid_outfits : List Outfit),
    (∀ o ∈ valid_outfits, is_valid_outfit o) ∧
    valid_outfits.length = 255) :=
  sorry


end NUMINAMATH_CALUDE_valid_outfit_count_l414_41499


namespace NUMINAMATH_CALUDE_min_toothpicks_to_remove_l414_41479

/-- Represents a square lattice made of toothpicks -/
structure SquareLattice where
  size : Nat
  total_toothpicks : Nat
  boundary_toothpicks : Nat
  internal_grid_toothpicks : Nat
  diagonal_toothpicks : Nat

/-- Theorem: Minimum number of toothpicks to remove to eliminate all squares and triangles -/
theorem min_toothpicks_to_remove (lattice : SquareLattice) 
  (h1 : lattice.size = 3)
  (h2 : lattice.total_toothpicks = 40)
  (h3 : lattice.boundary_toothpicks = 12)
  (h4 : lattice.internal_grid_toothpicks = 4)
  (h5 : lattice.diagonal_toothpicks = 12)
  (h6 : lattice.boundary_toothpicks + lattice.internal_grid_toothpicks + lattice.diagonal_toothpicks = lattice.total_toothpicks) :
  ∃ (n : Nat), n = lattice.boundary_toothpicks + lattice.internal_grid_toothpicks ∧ 
               n = 16 ∧
               (∀ m : Nat, m < n → ∃ (square : Bool) (triangle : Bool), square ∨ triangle) :=
by sorry


end NUMINAMATH_CALUDE_min_toothpicks_to_remove_l414_41479


namespace NUMINAMATH_CALUDE_triangle_cosine_theorem_l414_41423

theorem triangle_cosine_theorem (A : ℝ) (a m : ℝ) (θ : ℝ) : 
  A = 24 →
  a = 12 →
  m = 5 →
  A = (1/2) * a * m * Real.sin θ →
  Real.cos θ = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_triangle_cosine_theorem_l414_41423


namespace NUMINAMATH_CALUDE_cone_volume_from_half_sector_l414_41495

theorem cone_volume_from_half_sector (r : ℝ) (h : r = 6) :
  let sector_arc_length := π * r
  let cone_base_radius := sector_arc_length / (2 * π)
  let cone_slant_height := r
  let cone_height := Real.sqrt (cone_slant_height^2 - cone_base_radius^2)
  let cone_volume := (1/3) * π * cone_base_radius^2 * cone_height
  cone_volume = 3 * π * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_cone_volume_from_half_sector_l414_41495


namespace NUMINAMATH_CALUDE_no_infinite_sequence_exists_l414_41492

theorem no_infinite_sequence_exists : 
  ¬ (∃ (x : ℕ → ℝ), (∀ n : ℕ, x n > 0) ∧ 
    (∀ n : ℕ, x (n + 2) = Real.sqrt (x (n + 1)) - Real.sqrt (x n))) := by
  sorry

end NUMINAMATH_CALUDE_no_infinite_sequence_exists_l414_41492


namespace NUMINAMATH_CALUDE_october_order_theorem_l414_41437

/-- Represents the order quantities for a specific month -/
structure MonthOrder where
  clawHammers : ℕ
  ballPeenHammers : ℕ
  sledgehammers : ℕ

/-- Calculates the next month's order based on the pattern -/
def nextMonthOrder (current : MonthOrder) : MonthOrder := sorry

/-- Calculates the total number of hammers in an order -/
def totalHammers (order : MonthOrder) : ℕ :=
  order.clawHammers + order.ballPeenHammers + order.sledgehammers

/-- Applies the seasonal increase to the total order -/
def applySeasonalIncrease (total : ℕ) (increase : Rat) : ℕ := sorry

/-- The order data for June, July, August, and September -/
def juneOrder : MonthOrder := ⟨3, 2, 1⟩
def julyOrder : MonthOrder := ⟨4, 3, 2⟩
def augustOrder : MonthOrder := ⟨6, 7, 3⟩
def septemberOrder : MonthOrder := ⟨9, 11, 4⟩

/-- The seasonal increase percentage -/
def seasonalIncrease : Rat := 7 / 100

theorem october_order_theorem :
  let octoberOrder := nextMonthOrder septemberOrder
  let totalBeforeIncrease := totalHammers octoberOrder
  let finalTotal := applySeasonalIncrease totalBeforeIncrease seasonalIncrease
  finalTotal = 32 := by sorry

end NUMINAMATH_CALUDE_october_order_theorem_l414_41437


namespace NUMINAMATH_CALUDE_anthony_friend_house_distance_l414_41424

/-- Given the distances between various locations, prove the distance to Anthony's friend's house -/
theorem anthony_friend_house_distance 
  (distance_to_work : ℝ) 
  (distance_to_gym : ℝ) 
  (distance_to_grocery : ℝ) 
  (distance_to_friend : ℝ) : 
  distance_to_work = 10 ∧ 
  distance_to_gym = (distance_to_work / 2) + 2 ∧
  distance_to_grocery = 4 ∧
  distance_to_grocery = 2 * distance_to_gym ∧
  distance_to_friend = 3 * (distance_to_gym + distance_to_grocery) →
  distance_to_friend = 63 := by
  sorry


end NUMINAMATH_CALUDE_anthony_friend_house_distance_l414_41424


namespace NUMINAMATH_CALUDE_max_min_s_values_l414_41432

theorem max_min_s_values (x y : ℝ) (h : 4 * x^2 - 5 * x * y + 4 * y^2 = 5) :
  let s := x^2 + y^2
  (∀ a b : ℝ, 4 * a^2 - 5 * a * b + 4 * b^2 = 5 → a^2 + b^2 ≤ 10/3) ∧
  (∃ c d : ℝ, 4 * c^2 - 5 * c * d + 4 * d^2 = 5 ∧ c^2 + d^2 = 10/3) ∧
  (∀ a b : ℝ, 4 * a^2 - 5 * a * b + 4 * b^2 = 5 → a^2 + b^2 ≥ 10/13) ∧
  (∃ e f : ℝ, 4 * e^2 - 5 * e * f + 4 * f^2 = 5 ∧ e^2 + f^2 = 10/13) :=
by sorry

end NUMINAMATH_CALUDE_max_min_s_values_l414_41432


namespace NUMINAMATH_CALUDE_greatest_common_divisor_of_180_and_n_l414_41462

/-- Given two positive integers 180 and n that share exactly five positive divisors,
    the greatest of these five common divisors is 27. -/
theorem greatest_common_divisor_of_180_and_n : 
  ∀ n : ℕ+, 
  (∃! (s : Finset ℕ+), s.card = 5 ∧ (∀ d ∈ s, d ∣ 180 ∧ d ∣ n)) → 
  (∃ (s : Finset ℕ+), s.card = 5 ∧ (∀ d ∈ s, d ∣ 180 ∧ d ∣ n) ∧ 27 ∈ s ∧ ∀ x ∈ s, x ≤ 27) :=
by sorry


end NUMINAMATH_CALUDE_greatest_common_divisor_of_180_and_n_l414_41462


namespace NUMINAMATH_CALUDE_remainder_problem_l414_41486

theorem remainder_problem (x : ℤ) (h : x % 82 = 5) : (x + 13) % 41 = 18 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l414_41486


namespace NUMINAMATH_CALUDE_otimes_calculation_l414_41498

-- Define the ⊗ operation
def otimes (a b : ℚ) : ℚ := a^3 / b^2

-- State the theorem
theorem otimes_calculation :
  let x := otimes (otimes 2 4) (otimes 1 3)
  let y := otimes 2 (otimes 4 3)
  x - y = 1215 / 512 := by sorry

end NUMINAMATH_CALUDE_otimes_calculation_l414_41498


namespace NUMINAMATH_CALUDE_bill_throws_21_objects_l414_41429

/-- The number of sticks Ted throws -/
def ted_sticks : ℕ := 10

/-- The number of rocks Ted throws -/
def ted_rocks : ℕ := 10

/-- The number of sticks Bill throws -/
def bill_sticks : ℕ := ted_sticks + 6

/-- The number of rocks Bill throws -/
def bill_rocks : ℕ := ted_rocks / 2

/-- The total number of objects Bill throws -/
def bill_total : ℕ := bill_sticks + bill_rocks

theorem bill_throws_21_objects : bill_total = 21 := by
  sorry

end NUMINAMATH_CALUDE_bill_throws_21_objects_l414_41429


namespace NUMINAMATH_CALUDE_tax_amount_l414_41465

def gross_pay : ℝ := 1120
def retirement_rate : ℝ := 0.25
def net_pay : ℝ := 740

theorem tax_amount : 
  gross_pay * (1 - retirement_rate) - net_pay = 100 := by
  sorry

end NUMINAMATH_CALUDE_tax_amount_l414_41465


namespace NUMINAMATH_CALUDE_largest_fraction_l414_41442

theorem largest_fraction : 
  let f1 := 5 / 11
  let f2 := 6 / 13
  let f3 := 19 / 39
  let f4 := 101 / 203
  let f5 := 152 / 303
  let f6 := 80 / 159
  (f6 > f1) ∧ (f6 > f2) ∧ (f6 > f3) ∧ (f6 > f4) ∧ (f6 > f5) := by
  sorry

end NUMINAMATH_CALUDE_largest_fraction_l414_41442


namespace NUMINAMATH_CALUDE_unique_three_digit_number_l414_41443

/-- A three-digit number is represented as 100 * a + 10 * b + c, where a, b, c are single digits -/
def three_digit_number (a b c : ℕ) : Prop :=
  a ≠ 0 ∧ a < 10 ∧ b < 10 ∧ c < 10

/-- The number is 12 times the sum of its digits -/
def twelve_times_sum (a b c : ℕ) : Prop :=
  100 * a + 10 * b + c = 12 * (a + b + c)

theorem unique_three_digit_number :
  ∃! (a b c : ℕ), three_digit_number a b c ∧ twelve_times_sum a b c ∧
    100 * a + 10 * b + c = 108 :=
by sorry

end NUMINAMATH_CALUDE_unique_three_digit_number_l414_41443


namespace NUMINAMATH_CALUDE_quadratic_two_zeros_a_range_l414_41457

theorem quadratic_two_zeros_a_range (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ a * x₁^2 + 4 * x₁ - 2 = 0 ∧ a * x₂^2 + 4 * x₂ - 2 = 0) →
  a ∈ Set.Ioo (-2 : ℝ) 0 ∪ Set.Ioi 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_two_zeros_a_range_l414_41457


namespace NUMINAMATH_CALUDE_sum_of_four_integers_l414_41474

theorem sum_of_four_integers (a b c d : ℤ) :
  (a + b + c) / 3 + d = 8 ∧
  (a + b + d) / 3 + c = 12 ∧
  (a + c + d) / 3 + b = 32 / 3 ∧
  (b + c + d) / 3 + a = 28 / 3 →
  a + b + c + d = 30 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_four_integers_l414_41474


namespace NUMINAMATH_CALUDE_investment_proof_l414_41431

/-- Compound interest function -/
def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time

theorem investment_proof : 
  let principal : ℝ := 1000
  let rate : ℝ := 0.06
  let time : ℕ := 8
  let final_balance : ℝ := 1593.85
  compound_interest principal rate time = final_balance := by
sorry

end NUMINAMATH_CALUDE_investment_proof_l414_41431


namespace NUMINAMATH_CALUDE_other_factor_of_60n_l414_41408

theorem other_factor_of_60n (x : ℕ+) (h : ∀ n : ℕ, n ≥ 8 → (∃ k m : ℕ, 60 * n = x * k ∧ 60 * n = 8 * m)) :
  x ≥ 60 := by
  sorry

end NUMINAMATH_CALUDE_other_factor_of_60n_l414_41408


namespace NUMINAMATH_CALUDE_alloy_price_example_l414_41410

/-- The price of the alloy per kg when two metals are mixed in equal proportions -/
def alloy_price (price1 price2 : ℚ) : ℚ :=
  (price1 + price2) / 2

/-- Theorem: The price of an alloy made from two metals costing 68 and 96 per kg, 
    mixed in equal proportions, is 82 per kg -/
theorem alloy_price_example : alloy_price 68 96 = 82 := by
  sorry

end NUMINAMATH_CALUDE_alloy_price_example_l414_41410


namespace NUMINAMATH_CALUDE_justin_reading_ratio_l414_41464

/-- Proves that the ratio of pages read each day in the remaining 6 days to the first day is 2:1 -/
theorem justin_reading_ratio : ∀ (pages_first_day : ℕ) (total_pages : ℕ) (days_remaining : ℕ),
  pages_first_day = 10 →
  total_pages = 130 →
  days_remaining = 6 →
  (days_remaining * (pages_first_day * (total_pages - pages_first_day) / (pages_first_day * days_remaining)) = total_pages - pages_first_day) →
  (total_pages - pages_first_day) / (pages_first_day * days_remaining) = 2 := by
  sorry

end NUMINAMATH_CALUDE_justin_reading_ratio_l414_41464


namespace NUMINAMATH_CALUDE_student_A_received_A_grade_l414_41482

-- Define the students
inductive Student : Type
| A : Student
| B : Student
| C : Student

-- Define the grade levels
inductive Grade : Type
| A : Grade
| B : Grade
| C : Grade

-- Define a function to represent the actual grades received
def actual_grade : Student → Grade := sorry

-- Define a function to represent the correctness of predictions
def prediction_correct : Student → Prop := sorry

-- Theorem statement
theorem student_A_received_A_grade :
  -- Only one student received an A grade
  (∃! s : Student, actual_grade s = Grade.A) →
  -- A's prediction: C can only get a B or C
  (actual_grade Student.C ≠ Grade.A) →
  -- B's prediction: B will get an A
  (actual_grade Student.B = Grade.A) →
  -- C's prediction: C agrees with A's prediction
  (actual_grade Student.C ≠ Grade.A) →
  -- Only one prediction was inaccurate
  (∃! s : Student, ¬prediction_correct s) →
  -- Student A received an A grade
  actual_grade Student.A = Grade.A :=
sorry

end NUMINAMATH_CALUDE_student_A_received_A_grade_l414_41482


namespace NUMINAMATH_CALUDE_fruit_weights_l414_41409

/-- Represents the fruits on the table -/
inductive Fruit
| orange
| banana
| mandarin
| peach
| apple

/-- Assigns weights to fruits -/
def weight : Fruit → ℕ
| Fruit.orange => 280
| Fruit.banana => 170
| Fruit.mandarin => 100
| Fruit.peach => 200
| Fruit.apple => 150

/-- The set of all possible weights -/
def weights : Set ℕ := {100, 150, 170, 200, 280}

theorem fruit_weights :
  (∀ f : Fruit, weight f ∈ weights) ∧
  (weight Fruit.peach < weight Fruit.orange) ∧
  (weight Fruit.apple < weight Fruit.banana) ∧
  (weight Fruit.banana < weight Fruit.peach) ∧
  (weight Fruit.mandarin < weight Fruit.banana) ∧
  (weight Fruit.apple + weight Fruit.banana > weight Fruit.orange) ∧
  (∀ w : ℕ, w ∈ weights → ∃! f : Fruit, weight f = w) :=
by sorry

end NUMINAMATH_CALUDE_fruit_weights_l414_41409


namespace NUMINAMATH_CALUDE_min_cos_C_in_triangle_l414_41422

theorem min_cos_C_in_triangle (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) 
  (triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b)
  (side_relation : a^2 + b^2 = (5/2) * c^2) : 
  ∃ (cos_C : ℝ), cos_C = (a^2 + b^2 - c^2) / (2*a*b) ∧ cos_C ≥ 3/5 :=
sorry

end NUMINAMATH_CALUDE_min_cos_C_in_triangle_l414_41422


namespace NUMINAMATH_CALUDE_rectangle_ratio_l414_41414

/-- Configuration of rectangles around a square -/
structure RectangleConfig where
  /-- Side length of the smaller square -/
  s : ℝ
  /-- Shorter side of the rectangle -/
  y : ℝ
  /-- Longer side of the rectangle -/
  x : ℝ
  /-- The side length of the smaller square is 1 -/
  h1 : s = 1
  /-- The side length of the larger square is s + 2y -/
  h2 : s + 2*y = 2*s
  /-- The side length of the larger square is also x + s -/
  h3 : x + s = 2*s

/-- The ratio of the longer side to the shorter side of the rectangle is 2 -/
theorem rectangle_ratio (config : RectangleConfig) : x / y = 2 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_ratio_l414_41414


namespace NUMINAMATH_CALUDE_quadratic_equivalence_l414_41428

/-- The quadratic function in general form -/
def f (x : ℝ) : ℝ := x^2 - 2*x + 3

/-- The quadratic function in vertex form -/
def g (x : ℝ) : ℝ := (x - 1)^2 + 2

/-- Theorem stating that f and g are equivalent -/
theorem quadratic_equivalence : ∀ x : ℝ, f x = g x := by sorry

end NUMINAMATH_CALUDE_quadratic_equivalence_l414_41428


namespace NUMINAMATH_CALUDE_joes_kids_haircuts_l414_41411

-- Define the time it takes for each type of haircut
def womens_haircut_time : ℕ := 50
def mens_haircut_time : ℕ := 15
def kids_haircut_time : ℕ := 25

-- Define the number of women's and men's haircuts
def num_womens_haircuts : ℕ := 3
def num_mens_haircuts : ℕ := 2

-- Define the total time spent cutting hair
def total_time : ℕ := 255

-- Define a function to calculate the number of kids' haircuts
def num_kids_haircuts (w m k : ℕ) : ℕ :=
  (total_time - (w * womens_haircut_time + m * mens_haircut_time)) / k

-- Theorem statement
theorem joes_kids_haircuts :
  num_kids_haircuts num_womens_haircuts num_mens_haircuts kids_haircut_time = 3 := by
  sorry

end NUMINAMATH_CALUDE_joes_kids_haircuts_l414_41411


namespace NUMINAMATH_CALUDE_purse_percentage_l414_41473

/-- The value of a penny in cents -/
def penny_value : ℕ := 1

/-- The value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- The value of a dime in cents -/
def dime_value : ℕ := 10

/-- The value of a quarter in cents -/
def quarter_value : ℕ := 25

/-- The number of pennies in Samantha's purse -/
def num_pennies : ℕ := 2

/-- The number of nickels in Samantha's purse -/
def num_nickels : ℕ := 3

/-- The number of dimes in Samantha's purse -/
def num_dimes : ℕ := 1

/-- The number of quarters in Samantha's purse -/
def num_quarters : ℕ := 2

/-- The total value of coins in Samantha's purse in cents -/
def total_cents : ℕ := 
  num_pennies * penny_value + 
  num_nickels * nickel_value + 
  num_dimes * dime_value + 
  num_quarters * quarter_value

/-- The percentage of one dollar in Samantha's purse -/
theorem purse_percentage : (total_cents : ℚ) / 100 = 77 / 100 := by
  sorry

end NUMINAMATH_CALUDE_purse_percentage_l414_41473


namespace NUMINAMATH_CALUDE_A_intersect_B_eq_A_l414_41456

-- Define sets A and B
def A : Set ℝ := {x | 0 < x ∧ x ≤ 1}
def B : Set ℝ := {x | -Real.sqrt 2 ≤ x ∧ x ≤ Real.sqrt 2}

-- Theorem statement
theorem A_intersect_B_eq_A : A ∩ B = A := by sorry

end NUMINAMATH_CALUDE_A_intersect_B_eq_A_l414_41456


namespace NUMINAMATH_CALUDE_quadratic_inverse_unique_solution_l414_41434

/-- A quadratic function with its inverse -/
structure QuadraticWithInverse where
  a : ℝ
  b : ℝ
  c : ℝ
  f : ℝ → ℝ
  f_inv : ℝ → ℝ
  h_f : ∀ x, f x = a * x^2 + b * x + c
  h_f_inv : ∀ x, f_inv x = c * x^2 + b * x + a
  h_inverse : (∀ x, f (f_inv x) = x) ∧ (∀ x, f_inv (f x) = x)

/-- Theorem stating the unique solution for a, b, and c -/
theorem quadratic_inverse_unique_solution (q : QuadraticWithInverse) :
  q.a = -1 ∧ q.b = 1 ∧ q.c = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inverse_unique_solution_l414_41434


namespace NUMINAMATH_CALUDE_sum_of_solutions_is_pi_over_six_l414_41421

theorem sum_of_solutions_is_pi_over_six :
  ∃ (x : ℝ), 0 ≤ x ∧ x ≤ π ∧
  (1 / Real.sin x) + (1 / Real.cos x) = 2 * Real.sqrt 3 ∧
  (∀ (y : ℝ), 0 ≤ y ∧ y ≤ π ∧
    (1 / Real.sin y) + (1 / Real.cos y) = 2 * Real.sqrt 3 →
    y = x) ∧
  x = π / 6 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_solutions_is_pi_over_six_l414_41421


namespace NUMINAMATH_CALUDE_perfect_square_m_l414_41488

theorem perfect_square_m (k m n : ℕ) (h1 : k > 0) (h2 : m > 0) (h3 : n > 0) 
  (h4 : Odd k) (h5 : (2 + Real.sqrt 3)^k = 1 + m + n * Real.sqrt 3) : 
  ∃ (q : ℕ), m = q^2 := by
sorry

end NUMINAMATH_CALUDE_perfect_square_m_l414_41488


namespace NUMINAMATH_CALUDE_tony_bought_seven_swords_l414_41470

/-- Represents the purchase of toys by Tony -/
structure ToyPurchase where
  lego_cost : ℕ
  sword_cost : ℕ
  dough_cost : ℕ
  lego_sets : ℕ
  doughs : ℕ
  total_paid : ℕ

/-- Calculates the number of toy swords bought given a ToyPurchase -/
def calculate_swords (purchase : ToyPurchase) : ℕ :=
  let lego_total := purchase.lego_cost * purchase.lego_sets
  let dough_total := purchase.dough_cost * purchase.doughs
  let sword_total := purchase.total_paid - lego_total - dough_total
  sword_total / purchase.sword_cost

/-- Theorem stating that Tony bought 7 toy swords -/
theorem tony_bought_seven_swords : 
  ∀ (purchase : ToyPurchase), 
    purchase.lego_cost = 250 →
    purchase.sword_cost = 120 →
    purchase.dough_cost = 35 →
    purchase.lego_sets = 3 →
    purchase.doughs = 10 →
    purchase.total_paid = 1940 →
    calculate_swords purchase = 7 := by
  sorry

end NUMINAMATH_CALUDE_tony_bought_seven_swords_l414_41470


namespace NUMINAMATH_CALUDE_fourth_side_length_l414_41466

/-- A quadrilateral inscribed in a circle -/
structure InscribedQuadrilateral where
  /-- The radius of the circumscribed circle -/
  radius : ℝ
  /-- The lengths of the four sides of the quadrilateral -/
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  side4 : ℝ
  /-- Condition that the quadrilateral is inscribed in the circle -/
  inscribed : True -- This is a placeholder for the actual condition

/-- Theorem stating that given the specific conditions, the fourth side has length 500 -/
theorem fourth_side_length (q : InscribedQuadrilateral) 
  (h_radius : q.radius = 300 * Real.sqrt 2)
  (h_side1 : q.side1 = 300)
  (h_side2 : q.side2 = 300)
  (h_side3 : q.side3 = 400) :
  q.side4 = 500 := by
  sorry


end NUMINAMATH_CALUDE_fourth_side_length_l414_41466


namespace NUMINAMATH_CALUDE_polynomial_division_l414_41439

theorem polynomial_division (x y : ℝ) (h : y ≠ 0) : (3 * x * y + y) / y = 3 * x + 1 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_l414_41439


namespace NUMINAMATH_CALUDE_two_zeros_iff_k_range_l414_41450

/-- The function f(x) = xe^x - k has exactly two zeros if and only if -1/e < k < 0 -/
theorem two_zeros_iff_k_range (k : ℝ) :
  (∃! (a b : ℝ), a ≠ b ∧ a * Real.exp a - k = 0 ∧ b * Real.exp b - k = 0) ↔
  -1 / Real.exp 1 < k ∧ k < 0 := by sorry

end NUMINAMATH_CALUDE_two_zeros_iff_k_range_l414_41450


namespace NUMINAMATH_CALUDE_fraction_equality_l414_41478

theorem fraction_equality (x y : ℝ) (h : x / y = 2) : (x - y) / x = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l414_41478


namespace NUMINAMATH_CALUDE_distance_between_points_l414_41404

/-- The distance between two points when two people walk towards each other --/
theorem distance_between_points (speed_a speed_b : ℝ) (midpoint_offset : ℝ) : 
  speed_a = 70 →
  speed_b = 60 →
  midpoint_offset = 80 →
  (speed_a - speed_b) * ((2 * midpoint_offset) / (speed_a - speed_b)) = speed_a + speed_b →
  (speed_a + speed_b) * ((2 * midpoint_offset) / (speed_a - speed_b)) = 2080 :=
by
  sorry

#check distance_between_points

end NUMINAMATH_CALUDE_distance_between_points_l414_41404


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l414_41420

theorem sufficient_not_necessary_condition (a b : ℝ) : 
  (b < -4 → ∀ a, |a| + |b| > 4) ∧ 
  (∃ a b, |a| + |b| > 4 ∧ b ≥ -4) := by
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l414_41420


namespace NUMINAMATH_CALUDE_sector_arc_length_l414_41403

/-- Given a circular sector with area 2 cm² and central angle 4 radians,
    the length of the arc of the sector is 6 cm. -/
theorem sector_arc_length (area : ℝ) (angle : ℝ) (arc_length : ℝ) :
  area = 2 →
  angle = 4 →
  arc_length = 6 :=
by sorry

end NUMINAMATH_CALUDE_sector_arc_length_l414_41403


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_number_l414_41412

theorem purely_imaginary_complex_number (a : ℝ) : 
  (Complex.I * (a + 1) = (a^2 - 2*a - 3) + Complex.I * (a + 1)) → a = 3 := by
  sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_number_l414_41412


namespace NUMINAMATH_CALUDE_sin_cos_difference_equality_l414_41458

theorem sin_cos_difference_equality : 
  Real.sin (7 * π / 180) * Real.cos (37 * π / 180) - 
  Real.sin (83 * π / 180) * Real.sin (37 * π / 180) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_difference_equality_l414_41458


namespace NUMINAMATH_CALUDE_vector_problem_l414_41427

/-- Given vectors in R^2 -/
def OA : Fin 2 → ℝ := ![3, -4]
def OB : Fin 2 → ℝ := ![6, -3]
def OC (m : ℝ) : Fin 2 → ℝ := ![5 - m, -3 - m]

/-- A, B, and C are collinear if and only if the cross product of AB and AC is zero -/
def collinear (m : ℝ) : Prop :=
  let AB := OB - OA
  let AC := OC m - OA
  AB 0 * AC 1 = AB 1 * AC 0

/-- ABC is a right-angled triangle if and only if one of its angles is 90 degrees -/
def right_angled (m : ℝ) : Prop :=
  let AB := OB - OA
  let BC := OC m - OB
  let AC := OC m - OA
  AB • BC = 0 ∨ BC • AC = 0 ∨ AC • AB = 0

/-- Main theorem -/
theorem vector_problem (m : ℝ) :
  (collinear m → m = 1/2) ∧
  (right_angled m → m = 7/4 ∨ m = -3/4 ∨ m = (1 + Real.sqrt 5)/2 ∨ m = (1 - Real.sqrt 5)/2) :=
sorry

end NUMINAMATH_CALUDE_vector_problem_l414_41427


namespace NUMINAMATH_CALUDE_two_children_gender_combinations_l414_41455

-- Define the Gender type
inductive Gender
  | Male
  | Female

-- Define a type for a pair of children
def ChildPair := (Gender × Gender)

-- Define the set of all possible gender combinations
def allGenderCombinations : Set ChildPair :=
  {(Gender.Male, Gender.Male), (Gender.Male, Gender.Female),
   (Gender.Female, Gender.Male), (Gender.Female, Gender.Female)}

-- Theorem statement
theorem two_children_gender_combinations :
  ∀ (family : Set ChildPair),
  (∀ pair : ChildPair, pair ∈ family ↔ pair ∈ allGenderCombinations) ↔
  family = allGenderCombinations :=
by sorry

end NUMINAMATH_CALUDE_two_children_gender_combinations_l414_41455


namespace NUMINAMATH_CALUDE_compare_quadratic_expressions_compare_fraction_sum_and_sum_l414_41481

-- Statement 1
theorem compare_quadratic_expressions (m : ℝ) :
  3 * m^2 - m + 1 > 2 * m^2 + m - 3 := by
  sorry

-- Statement 2
theorem compare_fraction_sum_and_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  a^2 / b + b^2 / a ≥ a + b := by
  sorry

end NUMINAMATH_CALUDE_compare_quadratic_expressions_compare_fraction_sum_and_sum_l414_41481


namespace NUMINAMATH_CALUDE_coefficient_x_cubed_expansion_l414_41496

theorem coefficient_x_cubed_expansion : 
  let f : Polynomial ℚ := (X - 1) * (2 * X + 1)^5
  (f.coeff 3) = -40 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x_cubed_expansion_l414_41496


namespace NUMINAMATH_CALUDE_only_valid_pair_is_tiger_and_leopard_l414_41494

/-- Represents the animals that can participate in the forest competition. -/
inductive Animal : Type
  | Lion : Animal
  | Tiger : Animal
  | Leopard : Animal
  | Elephant : Animal

/-- Represents a pair of animals. -/
structure AnimalPair :=
  (first : Animal)
  (second : Animal)

/-- Checks if the given animal pair satisfies all the competition rules. -/
def satisfiesRules (pair : AnimalPair) : Prop :=
  -- If a lion is sent, a tiger must also be sent
  (pair.first = Animal.Lion ∨ pair.second = Animal.Lion) → 
    (pair.first = Animal.Tiger ∨ pair.second = Animal.Tiger) ∧
  -- If a leopard is not sent, a tiger cannot be sent
  (pair.first ≠ Animal.Leopard ∧ pair.second ≠ Animal.Leopard) → 
    (pair.first ≠ Animal.Tiger ∧ pair.second ≠ Animal.Tiger) ∧
  -- If a leopard participates, the elephant is not willing to go
  (pair.first = Animal.Leopard ∨ pair.second = Animal.Leopard) → 
    (pair.first ≠ Animal.Elephant ∧ pair.second ≠ Animal.Elephant)

/-- The theorem stating that the only valid pair is Tiger and Leopard. -/
theorem only_valid_pair_is_tiger_and_leopard :
  ∀ (pair : AnimalPair), 
    satisfiesRules pair ↔ 
      ((pair.first = Animal.Tiger ∧ pair.second = Animal.Leopard) ∨
       (pair.first = Animal.Leopard ∧ pair.second = Animal.Tiger)) :=
by sorry

end NUMINAMATH_CALUDE_only_valid_pair_is_tiger_and_leopard_l414_41494


namespace NUMINAMATH_CALUDE_real_axis_length_is_six_l414_41460

/-- The hyperbola C with equation x²/a² - y²/b² = 1 -/
structure Hyperbola where
  a : ℝ
  b : ℝ

/-- The line l with equation 4x + 3y - 20 = 0 -/
def line_l (x y : ℝ) : Prop := 4*x + 3*y - 20 = 0

/-- The line l passes through one focus of the hyperbola C -/
def passes_through_focus (C : Hyperbola) : Prop :=
  ∃ (x y : ℝ), line_l x y ∧ x^2 - C.a^2 = C.b^2

/-- The line l is parallel to one of the asymptotes of the hyperbola C -/
def parallel_to_asymptote (C : Hyperbola) : Prop :=
  C.b / C.a = 4 / 3

/-- The theorem stating that the length of the real axis of the hyperbola C is 6 -/
theorem real_axis_length_is_six (C : Hyperbola)
  (h1 : passes_through_focus C)
  (h2 : parallel_to_asymptote C) :
  2 * C.a = 6 := by sorry

end NUMINAMATH_CALUDE_real_axis_length_is_six_l414_41460


namespace NUMINAMATH_CALUDE_system_solution_l414_41446

/-- Given a system of equations x * y = a and x^5 + y^5 = b^5, this theorem states the solutions
    for different cases of a and b. -/
theorem system_solution (a b : ℝ) :
  (∀ x y : ℝ, x * y = a ∧ x^5 + y^5 = b^5 →
    (a = 0 ∧ b = 0 ∧ ∃ t : ℝ, x = t ∧ y = -t) ∨
    ((16 * b^5 ≤ a^5 ∧ a^5 < 0) ∨ (0 < a^5 ∧ a^5 ≤ 16 * b^5) ∧
      ((x = a/2 + Real.sqrt (Real.sqrt ((a^5 + 4*b^5)/(5*a))/2 - a^2/4) ∧
        y = a/2 - Real.sqrt (Real.sqrt ((a^5 + 4*b^5)/(5*a))/2 - a^2/4)) ∨
       (x = a/2 - Real.sqrt (Real.sqrt ((a^5 + 4*b^5)/(5*a))/2 - a^2/4) ∧
        y = a/2 + Real.sqrt (Real.sqrt ((a^5 + 4*b^5)/(5*a))/2 - a^2/4))))) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l414_41446


namespace NUMINAMATH_CALUDE_remaining_time_formula_l414_41425

/-- Represents the exam scenario for Jessica -/
structure ExamScenario where
  totalTime : ℕ  -- Total time for the exam in minutes
  totalQuestions : ℕ  -- Total number of questions
  answeredQuestions : ℕ  -- Number of questions answered so far
  timeUsed : ℕ  -- Time used so far in minutes
  penaltyPerIncorrect : ℕ  -- Time penalty for each incorrect answer in minutes

/-- Calculates the remaining time after penalties -/
def remainingTimeAfterPenalties (scenario : ExamScenario) (incorrectAnswers : ℕ) : ℤ :=
  scenario.totalTime - scenario.timeUsed - 
  (scenario.totalQuestions - scenario.answeredQuestions) * 
  (scenario.timeUsed / scenario.answeredQuestions) -
  incorrectAnswers * scenario.penaltyPerIncorrect

/-- Theorem stating that the remaining time after penalties is 15 - 2x -/
theorem remaining_time_formula (incorrectAnswers : ℕ) : 
  remainingTimeAfterPenalties 
    { totalTime := 90
    , totalQuestions := 100
    , answeredQuestions := 20
    , timeUsed := 15
    , penaltyPerIncorrect := 2 } 
    incorrectAnswers = 15 - 2 * incorrectAnswers :=
by sorry

end NUMINAMATH_CALUDE_remaining_time_formula_l414_41425


namespace NUMINAMATH_CALUDE_square_sum_equals_ten_l414_41441

theorem square_sum_equals_ten : 2^2 + 1^2 + 0^2 + (-1)^2 + (-2)^2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_equals_ten_l414_41441


namespace NUMINAMATH_CALUDE_max_trigonometric_product_l414_41459

theorem max_trigonometric_product (x y z : ℝ) : 
  (Real.sin (2 * x) + Real.sin (3 * y) + Real.sin (4 * z)) * 
  (Real.cos (2 * x) + Real.cos (3 * y) + Real.cos (4 * z)) ≤ 4.5 := by
  sorry

end NUMINAMATH_CALUDE_max_trigonometric_product_l414_41459


namespace NUMINAMATH_CALUDE_f_seven_equals_negative_two_l414_41416

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem f_seven_equals_negative_two
  (f : ℝ → ℝ)
  (h_odd : is_odd f)
  (h_periodic : ∀ x, f (x + 4) = f x)
  (h_f_one : f 1 = 2) :
  f 7 = -2 := by
sorry

end NUMINAMATH_CALUDE_f_seven_equals_negative_two_l414_41416


namespace NUMINAMATH_CALUDE_special_function_half_l414_41471

/-- A function satisfying the given properties -/
def special_function (f : ℝ → ℝ) : Prop :=
  f 1 = 2 ∧ ∀ x y : ℝ, f (x * y + f x) = x * f y + f x

/-- The main theorem stating the value of f(1/2) -/
theorem special_function_half (f : ℝ → ℝ) (h : special_function f) : f (1/2) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_special_function_half_l414_41471


namespace NUMINAMATH_CALUDE_larger_number_proof_l414_41449

/-- Given two positive integers with specific HCF and LCM conditions, prove the larger one is 230 -/
theorem larger_number_proof (a b : ℕ+) (h1 : Nat.gcd a b = 23) 
  (h2 : Nat.lcm a b = 23 * 9 * 10) (h3 : a > b) : a = 230 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_proof_l414_41449


namespace NUMINAMATH_CALUDE_six_player_four_games_tournament_l414_41406

/-- Represents a chess tournament --/
structure ChessTournament where
  numPlayers : Nat
  gamesPerPlayer : Nat

/-- Calculates the total number of games in a chess tournament --/
def totalGames (t : ChessTournament) : Nat :=
  (t.numPlayers * t.gamesPerPlayer) / 2

/-- Theorem: In a tournament with 6 players where each plays 4 others, there are 10 games total --/
theorem six_player_four_games_tournament :
  ∀ (t : ChessTournament),
    t.numPlayers = 6 →
    t.gamesPerPlayer = 4 →
    totalGames t = 10 := by
  sorry

#check six_player_four_games_tournament

end NUMINAMATH_CALUDE_six_player_four_games_tournament_l414_41406


namespace NUMINAMATH_CALUDE_cheap_gym_cost_l414_41426

/-- Represents the monthly cost of gym memberships and related calculations -/
def gym_costs (cheap_monthly : ℝ) : Prop :=
  let expensive_monthly := 3 * cheap_monthly
  let cheap_signup := 50
  let expensive_signup := 4 * expensive_monthly
  let cheap_yearly := cheap_signup + 12 * cheap_monthly
  let expensive_yearly := expensive_signup + 12 * expensive_monthly
  cheap_yearly + expensive_yearly = 650

theorem cheap_gym_cost : ∃ (cheap_monthly : ℝ), gym_costs cheap_monthly ∧ cheap_monthly = 10 := by
  sorry

end NUMINAMATH_CALUDE_cheap_gym_cost_l414_41426


namespace NUMINAMATH_CALUDE_school_boys_count_l414_41489

theorem school_boys_count (total : ℕ) (diff : ℕ) (boys : ℕ) : 
  total = 1443 →
  boys + (boys - diff) = total →
  diff = 141 →
  boys = 792 := by
sorry

end NUMINAMATH_CALUDE_school_boys_count_l414_41489


namespace NUMINAMATH_CALUDE_complex_fraction_squared_l414_41402

theorem complex_fraction_squared (i : ℂ) (hi : i^2 = -1) :
  ((1 - i) / (1 + i))^2 = -1 := by sorry

end NUMINAMATH_CALUDE_complex_fraction_squared_l414_41402


namespace NUMINAMATH_CALUDE_find_common_ratio_l414_41454

/-- Given a table of n^2 (n ≥ 4) positive numbers arranged in n rows and n columns,
    where each row forms an arithmetic sequence and each column forms a geometric sequence
    with the same common ratio q, prove that q = 1/2 given the specified conditions. -/
theorem find_common_ratio (n : ℕ) (a : ℕ → ℕ → ℝ) (q : ℝ) 
    (h_n : n ≥ 4)
    (h_positive : ∀ i j, 1 ≤ i ∧ i ≤ n ∧ 1 ≤ j ∧ j ≤ n → a i j > 0)
    (h_arithmetic_row : ∀ i k, 1 ≤ i ∧ i ≤ n ∧ 1 ≤ k ∧ k < n → 
      a i (k + 1) - a i k = a i (k + 2) - a i (k + 1))
    (h_geometric_col : ∀ i j, 1 ≤ i ∧ i < n ∧ 1 ≤ j ∧ j ≤ n → 
      a (i + 1) j = q * a i j)
    (h_a26 : a 2 6 = 1)
    (h_a42 : a 4 2 = 1/8)
    (h_a44 : a 4 4 = 3/16) :
  q = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_find_common_ratio_l414_41454


namespace NUMINAMATH_CALUDE_population_after_two_years_l414_41430

def initial_population : ℕ := 10000
def first_year_rate : ℚ := 1.05
def second_year_rate : ℚ := 0.95

theorem population_after_two_years :
  (↑initial_population * first_year_rate * second_year_rate).floor = 9975 := by
  sorry

end NUMINAMATH_CALUDE_population_after_two_years_l414_41430


namespace NUMINAMATH_CALUDE_certain_number_value_l414_41445

theorem certain_number_value : ∃ x : ℝ, (35 / 100) * x = (20 / 100) * 700 ∧ x = 400 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_value_l414_41445


namespace NUMINAMATH_CALUDE_min_garden_cost_l414_41433

-- Define the regions and their areas
def region1_area : ℕ := 10  -- 5x2
def region2_area : ℕ := 9   -- 3x3
def region3_area : ℕ := 20  -- 5x4
def region4_area : ℕ := 2   -- 2x1
def region5_area : ℕ := 7   -- 7x1

-- Define the flower costs
def aster_cost : ℚ := 1
def begonia_cost : ℚ := 2
def canna_cost : ℚ := 2
def dahlia_cost : ℚ := 3
def easter_lily_cost : ℚ := 2.5

-- Define the total garden area
def total_area : ℕ := region1_area + region2_area + region3_area + region4_area + region5_area

-- Theorem statement
theorem min_garden_cost : 
  ∃ (aster_count begonia_count canna_count dahlia_count easter_lily_count : ℕ),
    aster_count + begonia_count + canna_count + dahlia_count + easter_lily_count = total_area ∧
    aster_count * aster_cost + 
    begonia_count * begonia_cost + 
    canna_count * canna_cost + 
    dahlia_count * dahlia_cost + 
    easter_lily_count * easter_lily_cost = 81.5 ∧
    ∀ (a b c d e : ℕ),
      a + b + c + d + e = total_area →
      a * aster_cost + b * begonia_cost + c * canna_cost + d * dahlia_cost + e * easter_lily_cost ≥ 81.5 :=
by sorry

end NUMINAMATH_CALUDE_min_garden_cost_l414_41433


namespace NUMINAMATH_CALUDE_win_sector_area_l414_41451

theorem win_sector_area (r : ℝ) (p : ℝ) (A_win : ℝ) :
  r = 8 →
  p = 1 / 4 →
  A_win = p * π * r^2 →
  A_win = 16 * π :=
by sorry

end NUMINAMATH_CALUDE_win_sector_area_l414_41451


namespace NUMINAMATH_CALUDE_olympiad_1958_l414_41477

theorem olympiad_1958 (n : ℤ) : 1155^1958 + 34^1958 ≠ n^2 := by
  sorry

end NUMINAMATH_CALUDE_olympiad_1958_l414_41477


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l414_41467

/-- Given a hyperbola with imaginary axis length 2 and focal distance 2√3,
    prove that the equation of its asymptotes is y = ±(√2/2)x -/
theorem hyperbola_asymptotes 
  (b : ℝ) 
  (c : ℝ) 
  (h1 : b = 1)  -- half of the imaginary axis length
  (h2 : c = Real.sqrt 3)  -- half of the focal distance
  : ∃ (k : ℝ), k = Real.sqrt 2 / 2 ∧ 
    (∀ (x y : ℝ), (y = k * x ∨ y = -k * x) ↔ 
      (x^2 / (c^2 - b^2) - y^2 / b^2 = 1)) := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l414_41467


namespace NUMINAMATH_CALUDE_light_bulb_probability_l414_41476

/-- The probability of selecting a light bulb from Factory A and it passing the quality test -/
theorem light_bulb_probability (p_A : ℝ) (p_B : ℝ) (pass_A : ℝ) (pass_B : ℝ) 
  (h1 : p_A = 0.7) 
  (h2 : p_B = 0.3) 
  (h3 : p_A + p_B = 1) 
  (h4 : pass_A = 0.95) 
  (h5 : pass_B = 0.8) :
  p_A * pass_A = 0.665 := by
  sorry

end NUMINAMATH_CALUDE_light_bulb_probability_l414_41476


namespace NUMINAMATH_CALUDE_population_average_age_l414_41484

theorem population_average_age
  (ratio_women_men : ℚ)
  (avg_age_women : ℚ)
  (avg_age_men : ℚ)
  (h_ratio : ratio_women_men = 7 / 5)
  (h_women_age : avg_age_women = 40)
  (h_men_age : avg_age_men = 30) :
  (ratio_women_men * avg_age_women + avg_age_men) / (ratio_women_men + 1) = 215 / 6 :=
by sorry

end NUMINAMATH_CALUDE_population_average_age_l414_41484


namespace NUMINAMATH_CALUDE_money_left_l414_41483

def initial_amount : ℕ := 20
def num_items : ℕ := 4
def cost_per_item : ℕ := 2

theorem money_left : initial_amount - (num_items * cost_per_item) = 12 := by
  sorry

end NUMINAMATH_CALUDE_money_left_l414_41483


namespace NUMINAMATH_CALUDE_internet_cost_proof_l414_41440

/-- The daily cost of internet service -/
def daily_cost : ℝ := 0.28

/-- The number of days covered by the payment -/
def days_covered : ℕ := 25

/-- The amount paid -/
def payment : ℝ := 7

/-- The maximum allowed debt -/
def max_debt : ℝ := 5

theorem internet_cost_proof :
  daily_cost * days_covered = payment ∧
  daily_cost * (days_covered + 1) > payment + max_debt :=
by sorry

end NUMINAMATH_CALUDE_internet_cost_proof_l414_41440


namespace NUMINAMATH_CALUDE_range_of_a_l414_41418

def A (a : ℝ) : Set ℝ := {x | a + 1 ≤ x ∧ x ≤ 2*a - 1}
def B : Set ℝ := {x | x ≤ 3 ∨ x > 5}
def f (a x : ℝ) : ℝ := x^2 - 2*a*x + 1

def p (a : ℝ) : Prop := A a ⊆ B
def q (a : ℝ) : Prop := ∀ x > (1/2 : ℝ), Monotone (f a)

theorem range_of_a :
  ∀ a : ℝ, (¬(p a ∧ q a) ∧ (p a ∨ q a)) → ((1/2 < a ∧ a ≤ 2) ∨ a > 4) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l414_41418


namespace NUMINAMATH_CALUDE_min_x_prime_factorization_l414_41448

theorem min_x_prime_factorization (x y : ℕ+) (a b : ℕ) (c d : ℕ) 
  (h1 : 4 * x^7 = 13 * y^17)
  (h2 : x = a^c * b^d)
  (h3 : Nat.Prime a)
  (h4 : Nat.Prime b)
  (h5 : ∀ (w z : ℕ+) (e f : ℕ) (p q : ℕ), 
        4 * w^7 = 13 * z^17 → 
        w = p^e * q^f → 
        Nat.Prime p → 
        Nat.Prime q → 
        w ≤ x) : 
  a + b + c + d = 19 := by
sorry

end NUMINAMATH_CALUDE_min_x_prime_factorization_l414_41448


namespace NUMINAMATH_CALUDE_intersection_A_B_l414_41436

def A : Set ℝ := {x | -3 ≤ 2*x - 1 ∧ 2*x - 1 < 3}
def B : Set ℝ := {x | ∃ k : ℤ, x = 2*k + 1}

theorem intersection_A_B : A ∩ B = {-1, 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_B_l414_41436


namespace NUMINAMATH_CALUDE_sum_of_z_values_l414_41468

-- Define the function g
def g (x : ℝ) : ℝ := (4 * x)^2 - (4 * x) + 2

-- State the theorem
theorem sum_of_z_values (z : ℝ) : 
  (∃ z₁ z₂, g z₁ = 8 ∧ g z₂ = 8 ∧ z₁ ≠ z₂ ∧ z₁ + z₂ = 1/16) :=
sorry

end NUMINAMATH_CALUDE_sum_of_z_values_l414_41468


namespace NUMINAMATH_CALUDE_youngest_child_age_l414_41475

/-- Represents a family with its members and ages -/
structure Family where
  members : Nat
  total_age : Nat

/-- The problem setup -/
def initial_family : Family := { members := 4, total_age := 96 }

/-- The current state of the family -/
def current_family : Family := { members := 6, total_age := 144 }

/-- The time passed since the initial state -/
def years_passed : Nat := 10

/-- The age difference between the two new children -/
def age_difference : Nat := 2

/-- Theorem stating that the youngest child's age is 3 years -/
theorem youngest_child_age :
  let youngest_age := (current_family.total_age - (initial_family.total_age + years_passed * initial_family.members)) / 2
  youngest_age = 3 := by sorry

end NUMINAMATH_CALUDE_youngest_child_age_l414_41475


namespace NUMINAMATH_CALUDE_quadratic_increases_iff_l414_41487

/-- The quadratic function y = 2x^2 - 4x - 1 increases for x > a iff a ≥ 1 -/
theorem quadratic_increases_iff (a : ℝ) : 
  (∀ x₁ x₂ : ℝ, x₁ > a ∧ x₂ > x₁ → (2*x₂^2 - 4*x₂ - 1) > (2*x₁^2 - 4*x₁ - 1)) ↔ 
  a ≥ 1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_increases_iff_l414_41487


namespace NUMINAMATH_CALUDE_motorcyclist_wait_time_l414_41490

/-- Given a hiker and a motorcyclist with specified speeds, prove the time it takes for the
    motorcyclist to cover the distance the hiker walks in 48 minutes. -/
theorem motorcyclist_wait_time (hiker_speed : ℝ) (motorcyclist_speed : ℝ) 
    (hiker_walk_time : ℝ) (h1 : hiker_speed = 6) (h2 : motorcyclist_speed = 30) 
    (h3 : hiker_walk_time = 48) :
    (hiker_speed * hiker_walk_time) / motorcyclist_speed = 9.6 := by
  sorry

#check motorcyclist_wait_time

end NUMINAMATH_CALUDE_motorcyclist_wait_time_l414_41490


namespace NUMINAMATH_CALUDE_percentage_of_120_to_80_l414_41447

theorem percentage_of_120_to_80 : ∃ (p : ℝ), p = (120 : ℝ) / 80 * 100 ∧ p = 150 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_120_to_80_l414_41447


namespace NUMINAMATH_CALUDE_no_consec_nat_prod_equals_consec_even_prod_l414_41497

theorem no_consec_nat_prod_equals_consec_even_prod : 
  ¬∃ (m n : ℕ), m * (m + 1) = 4 * n * (n + 1) := by
sorry

end NUMINAMATH_CALUDE_no_consec_nat_prod_equals_consec_even_prod_l414_41497


namespace NUMINAMATH_CALUDE_b_value_proof_l414_41405

theorem b_value_proof (b : ℚ) (h : b + b/4 = 5/2) : b = 2 := by
  sorry

end NUMINAMATH_CALUDE_b_value_proof_l414_41405


namespace NUMINAMATH_CALUDE_menu_choices_l414_41491

/-- The number of ways to choose one menu for lunch and one for dinner -/
def choose_menus (lunch_chinese : Nat) (lunch_japanese : Nat) (dinner_chinese : Nat) (dinner_japanese : Nat) : Nat :=
  (lunch_chinese + lunch_japanese) * (dinner_chinese + dinner_japanese)

theorem menu_choices : choose_menus 5 4 3 5 = 72 := by
  sorry

end NUMINAMATH_CALUDE_menu_choices_l414_41491
