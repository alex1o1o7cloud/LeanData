import Mathlib

namespace NUMINAMATH_CALUDE_water_evaporation_period_l3271_327107

theorem water_evaporation_period (initial_water : ℝ) (daily_evaporation : ℝ) (evaporation_percentage : ℝ) :
  initial_water = 10 →
  daily_evaporation = 0.012 →
  evaporation_percentage = 0.06 →
  (initial_water * evaporation_percentage) / daily_evaporation = 50 :=
by sorry

end NUMINAMATH_CALUDE_water_evaporation_period_l3271_327107


namespace NUMINAMATH_CALUDE_correct_sqrt_calculation_l3271_327142

theorem correct_sqrt_calculation :
  (∃ (x y z : ℝ), x = Real.sqrt 2 ∧ y = Real.sqrt 3 ∧ z = Real.sqrt 6 ∧ x * y = z) ∧
  (∀ (x y z : ℝ), x = Real.sqrt 2 ∧ y = Real.sqrt 3 ∧ z = Real.sqrt 5 → x + y ≠ z) ∧
  (∀ (x y : ℝ), x = Real.sqrt 3 ∧ y = Real.sqrt 2 → x - y ≠ 1) ∧
  (∀ (x y : ℝ), x = Real.sqrt 4 ∧ y = Real.sqrt 2 → x / y ≠ 2) :=
by sorry


end NUMINAMATH_CALUDE_correct_sqrt_calculation_l3271_327142


namespace NUMINAMATH_CALUDE_probability_one_white_two_red_l3271_327188

def white_balls : ℕ := 4
def red_balls : ℕ := 5
def total_balls : ℕ := white_balls + red_balls
def drawn_balls : ℕ := 3

def favorable_outcomes : ℕ := (Nat.choose white_balls 1) * (Nat.choose red_balls 2)
def total_outcomes : ℕ := Nat.choose total_balls drawn_balls

theorem probability_one_white_two_red : 
  (favorable_outcomes : ℚ) / total_outcomes = 10 / 21 := by
  sorry

end NUMINAMATH_CALUDE_probability_one_white_two_red_l3271_327188


namespace NUMINAMATH_CALUDE_exists_divisible_by_digit_sum_in_sequence_l3271_327132

/-- Given a natural number, return the sum of its digits -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- Check if a number is divisible by the sum of its digits -/
def is_divisible_by_digit_sum (n : ℕ) : Prop :=
  n % digit_sum n = 0

/-- Theorem: In any sequence of 18 consecutive three-digit numbers, 
    at least one number is divisible by the sum of its digits -/
theorem exists_divisible_by_digit_sum_in_sequence :
  ∀ (start : ℕ), 100 ≤ start → start + 17 < 1000 →
  ∃ (k : ℕ), k ∈ Finset.range 18 ∧ is_divisible_by_digit_sum (start + k) :=
sorry

end NUMINAMATH_CALUDE_exists_divisible_by_digit_sum_in_sequence_l3271_327132


namespace NUMINAMATH_CALUDE_d11d_divisible_by_5_l3271_327139

/-- Represents a base-7 digit -/
def Base7Digit := {d : ℕ // d < 7}

/-- Converts a base-7 number of the form d11d to its decimal equivalent -/
def toDecimal (d : Base7Digit) : ℕ := 344 * d.val + 56

/-- A base-7 number d11d_7 is divisible by 5 if and only if d = 1 -/
theorem d11d_divisible_by_5 (d : Base7Digit) : 
  5 ∣ toDecimal d ↔ d.val = 1 := by sorry

end NUMINAMATH_CALUDE_d11d_divisible_by_5_l3271_327139


namespace NUMINAMATH_CALUDE_unique_alpha_beta_pair_l3271_327143

theorem unique_alpha_beta_pair :
  ∃! (α β : ℝ), α > 0 ∧ β > 0 ∧
  (∀ (x y z w : ℝ), x > 0 → y > 0 → z > 0 → w > 0 →
    x + y^2 + z^3 + w^6 ≥ α * (x*y*z*w)^β) ∧
  (∃ (x y z w : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧ w > 0 ∧
    x + y^2 + z^3 + w^6 = α * (x*y*z*w)^β) ∧
  α = 2^(4/3) * 3^(1/4) ∧ β = 1/2 := by
sorry

end NUMINAMATH_CALUDE_unique_alpha_beta_pair_l3271_327143


namespace NUMINAMATH_CALUDE_cube_root_125_times_fourth_root_256_times_sqrt_16_l3271_327178

theorem cube_root_125_times_fourth_root_256_times_sqrt_16 :
  (125 : ℝ) ^ (1/3) * (256 : ℝ) ^ (1/4) * (16 : ℝ) ^ (1/2) = 80 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_125_times_fourth_root_256_times_sqrt_16_l3271_327178


namespace NUMINAMATH_CALUDE_set_operations_l3271_327187

open Set

def U : Set ℝ := univ

def A : Set ℝ := {x : ℝ | 0 < x ∧ x ≤ 2}

def B : Set ℝ := {x : ℝ | x < -3 ∨ x > 1}

theorem set_operations :
  (A ∩ B = {x : ℝ | 1 < x ∧ x ≤ 2}) ∧
  (A ∪ B = {x : ℝ | x < -3 ∨ x > 0}) ∧
  ((Aᶜ) ∩ (Bᶜ) = {x : ℝ | -3 ≤ x ∧ x ≤ 0}) := by sorry

end NUMINAMATH_CALUDE_set_operations_l3271_327187


namespace NUMINAMATH_CALUDE_y_to_x_equals_25_l3271_327138

theorem y_to_x_equals_25 (x y : ℝ) (h : |x - 2| + (y + 5)^2 = 0) : y^x = 25 := by
  sorry

end NUMINAMATH_CALUDE_y_to_x_equals_25_l3271_327138


namespace NUMINAMATH_CALUDE_expected_sixes_is_one_third_l3271_327171

/-- The probability of rolling a 6 on a standard die -/
def prob_six : ℚ := 1 / 6

/-- The probability of not rolling a 6 on a standard die -/
def prob_not_six : ℚ := 1 - prob_six

/-- The expected number of 6's when rolling two standard dice -/
def expected_sixes : ℚ := 
  0 * (prob_not_six ^ 2) + 
  1 * (2 * prob_six * prob_not_six) + 
  2 * (prob_six ^ 2)

theorem expected_sixes_is_one_third : expected_sixes = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_expected_sixes_is_one_third_l3271_327171


namespace NUMINAMATH_CALUDE_max_known_cards_l3271_327189

/-- A strategy for selecting cards and receiving information -/
structure CardStrategy where
  selectCards : Fin 2013 → Finset (Fin 2013)
  receiveNumber : Finset (Fin 2013) → Fin 2013

/-- The set of cards for which we know the numbers after applying a strategy -/
def knownCards (s : CardStrategy) : Finset (Fin 2013) :=
  sorry

/-- The theorem stating that 1986 is the maximum number of cards we can guarantee to know -/
theorem max_known_cards :
  (∃ (s : CardStrategy), (knownCards s).card = 1986) ∧
  (∀ (s : CardStrategy), (knownCards s).card ≤ 1986) :=
sorry

end NUMINAMATH_CALUDE_max_known_cards_l3271_327189


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3271_327164

theorem complex_equation_solution (z : ℂ) 
  (h : 18 * Complex.normSq z = 2 * Complex.normSq (z + 3) + Complex.normSq (z^2 + 2) + 48) : 
  z + 12 / z = -3 := by
sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3271_327164


namespace NUMINAMATH_CALUDE_youngest_child_age_l3271_327166

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

end NUMINAMATH_CALUDE_youngest_child_age_l3271_327166


namespace NUMINAMATH_CALUDE_symmetric_difference_of_A_and_B_l3271_327167

-- Define the set difference operation
def setDifference (M N : Set ℝ) : Set ℝ := {x | x ∈ M ∧ x ∉ N}

-- Define the symmetric difference operation
def symmetricDifference (M N : Set ℝ) : Set ℝ := (setDifference M N) ∪ (setDifference N M)

-- Define sets A and B
def A : Set ℝ := {x | x ≥ -9/4}
def B : Set ℝ := {x | x < 0}

-- State the theorem
theorem symmetric_difference_of_A_and_B :
  symmetricDifference A B = {x | x < -9/4 ∨ x ≥ 0} :=
by sorry

end NUMINAMATH_CALUDE_symmetric_difference_of_A_and_B_l3271_327167


namespace NUMINAMATH_CALUDE_continuous_stripe_probability_l3271_327199

/-- A regular tetrahedron with painted stripes on each face -/
structure StripedTetrahedron :=
  (faces : Fin 4 → Fin 2)

/-- The probability of a specific stripe configuration -/
def stripe_probability : ℚ := 1 / 16

/-- A continuous stripe encircles the tetrahedron -/
def has_continuous_stripe (t : StripedTetrahedron) : Prop := sorry

/-- The number of stripe configurations that result in a continuous stripe -/
def continuous_stripe_count : ℕ := 2

theorem continuous_stripe_probability :
  (continuous_stripe_count : ℚ) * stripe_probability = 1 / 8 := by sorry

end NUMINAMATH_CALUDE_continuous_stripe_probability_l3271_327199


namespace NUMINAMATH_CALUDE_unique_number_from_dialogue_l3271_327114

/-- Represents a two-digit natural number -/
def TwoDigitNumber (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

/-- Calculates the sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Calculates the number of divisors of a natural number -/
def numberOfDivisors (n : ℕ) : ℕ := sorry

/-- Checks if the number satisfies the dialogue conditions -/
def satisfiesDialogueConditions (n : ℕ) : Prop :=
  TwoDigitNumber n ∧
  (∀ m : ℕ, TwoDigitNumber m → sumOfDigits m = sumOfDigits n → m ≠ n) ∧
  (numberOfDivisors n ≠ 2 ∧ numberOfDivisors n ≠ 12) ∧
  (∀ m : ℕ, TwoDigitNumber m → 
    sumOfDigits m = sumOfDigits n → 
    numberOfDivisors m = numberOfDivisors n → 
    m = n)

theorem unique_number_from_dialogue :
  ∃! n : ℕ, satisfiesDialogueConditions n ∧ n = 30 := by
  sorry

end NUMINAMATH_CALUDE_unique_number_from_dialogue_l3271_327114


namespace NUMINAMATH_CALUDE_figure_404_has_2022_squares_l3271_327140

/-- The number of squares in the nth figure of the sequence -/
def squares_in_figure (n : ℕ) : ℕ := 7 + (n - 1) * 5

theorem figure_404_has_2022_squares :
  squares_in_figure 404 = 2022 := by
  sorry

end NUMINAMATH_CALUDE_figure_404_has_2022_squares_l3271_327140


namespace NUMINAMATH_CALUDE_certain_number_proof_l3271_327177

theorem certain_number_proof (a n : ℕ) (h1 : a = 105) (h2 : a^3 = 21 * 25 * n * 49) : n = 5 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l3271_327177


namespace NUMINAMATH_CALUDE_six_integers_mean_double_mode_l3271_327144

def is_valid_list (l : List Int) : Prop :=
  l.length = 6 ∧ l.all (λ x => x > 0 ∧ x ≤ 150)

def mean (l : List Int) : Rat :=
  (l.sum : Rat) / l.length

def mode (l : List Int) : Int :=
  l.foldl (λ acc x => if l.count x > l.count acc then x else acc) 0

theorem six_integers_mean_double_mode :
  ∀ y z : Int,
    let l := [45, 76, y, y, z, z]
    is_valid_list l →
    mean l = 2 * (mode l : Rat) →
    y = 49 ∧ z = 21 := by
  sorry

end NUMINAMATH_CALUDE_six_integers_mean_double_mode_l3271_327144


namespace NUMINAMATH_CALUDE_quadratic_integer_root_set_characterization_l3271_327125

/-- The set of positive integers a for which the quadratic equation
    ax^2 + 2(2a-1)x + 4(a-3) = 0 has at least one integer root -/
def QuadraticIntegerRootSet : Set ℕ+ :=
  {a | ∃ x : ℤ, a * x^2 + 2*(2*a-1)*x + 4*(a-3) = 0}

/-- Theorem stating that the QuadraticIntegerRootSet contains exactly 1, 3, 6, and 10 -/
theorem quadratic_integer_root_set_characterization :
  QuadraticIntegerRootSet = {1, 3, 6, 10} := by
  sorry

#check quadratic_integer_root_set_characterization

end NUMINAMATH_CALUDE_quadratic_integer_root_set_characterization_l3271_327125


namespace NUMINAMATH_CALUDE_solve_system_l3271_327181

theorem solve_system (B C : ℝ) (eq1 : 5 * B - 3 = 32) (eq2 : 2 * B + 2 * C = 18) :
  B = 7 ∧ C = 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_system_l3271_327181


namespace NUMINAMATH_CALUDE_triangle_side_range_l3271_327185

-- Define the triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def hasTwoSolutions (t : Triangle) : Prop :=
  t.b * Real.sin t.A < t.a ∧ t.a < t.b

-- Theorem statement
theorem triangle_side_range (t : Triangle) 
  (h1 : t.b = 2)
  (h2 : t.A = π / 6)
  (h3 : hasTwoSolutions t) :
  1 < t.a ∧ t.a < 2 :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_side_range_l3271_327185


namespace NUMINAMATH_CALUDE_candy_theorem_l3271_327133

/-- The total number of candy pieces caught by four friends -/
def total_candy (tabitha stan julie carlos : ℕ) : ℕ :=
  tabitha + stan + julie + carlos

/-- Theorem: Given the conditions, the friends caught 72 pieces of candy in total -/
theorem candy_theorem (tabitha stan julie carlos : ℕ) 
  (h1 : tabitha = 22)
  (h2 : stan = 13)
  (h3 : julie = tabitha / 2)
  (h4 : carlos = 2 * stan) :
  total_candy tabitha stan julie carlos = 72 := by
  sorry

end NUMINAMATH_CALUDE_candy_theorem_l3271_327133


namespace NUMINAMATH_CALUDE_odd_z_has_4n_minus_1_divisor_l3271_327120

theorem odd_z_has_4n_minus_1_divisor (x y : ℕ+) (z : ℤ) 
  (hz : z = (4 * x * y : ℤ) / (x + y : ℤ)) 
  (hodd : Odd z) : 
  ∃ (d : ℤ), d ∣ z ∧ ∃ (n : ℕ+), d = 4 * n - 1 := by
  sorry

end NUMINAMATH_CALUDE_odd_z_has_4n_minus_1_divisor_l3271_327120


namespace NUMINAMATH_CALUDE_prob_three_primes_l3271_327194

def num_dice : ℕ := 6
def sides_per_die : ℕ := 12
def prob_prime : ℚ := 5/12

theorem prob_three_primes :
  let choose_three := Nat.choose num_dice 3
  let prob_three_prime := (prob_prime ^ 3 : ℚ)
  let prob_three_non_prime := ((1 - prob_prime) ^ 3 : ℚ)
  choose_three * prob_three_prime * prob_three_non_prime = 312500/248832 := by
sorry

end NUMINAMATH_CALUDE_prob_three_primes_l3271_327194


namespace NUMINAMATH_CALUDE_continuous_piecewise_function_l3271_327148

-- Define the piecewise function g(x)
noncomputable def g (x a b : ℝ) : ℝ :=
  if x > 1 then b * x + 1
  else if -3 ≤ x ∧ x ≤ 1 then 2 * x - 4
  else 3 * x - a

-- State the theorem
theorem continuous_piecewise_function (a b : ℝ) :
  Continuous g → a + b = -2 :=
by sorry

end NUMINAMATH_CALUDE_continuous_piecewise_function_l3271_327148


namespace NUMINAMATH_CALUDE_sin_cos_sum_squared_l3271_327169

theorem sin_cos_sum_squared (x : Real) : 
  (Real.sin x + Real.cos x = Real.sqrt 2 / 2) → 
  (Real.sin x)^4 + (Real.cos x)^4 = 7/8 := by
sorry

end NUMINAMATH_CALUDE_sin_cos_sum_squared_l3271_327169


namespace NUMINAMATH_CALUDE_clock_angle_at_13_20_clock_angle_at_13_20_is_80_l3271_327104

/-- The angle between the hour and minute hands of a clock at 13:20 (1:20 PM) --/
theorem clock_angle_at_13_20 : ℝ :=
  let hour := 1
  let minute := 20
  let degrees_per_hour := 360 / 12
  let degrees_per_minute := 360 / 60
  let hour_hand_angle := hour * degrees_per_hour + (minute / 60) * degrees_per_hour
  let minute_hand_angle := minute * degrees_per_minute
  |minute_hand_angle - hour_hand_angle|

/-- The angle between the hour and minute hands of a clock at 13:20 (1:20 PM) is 80 degrees --/
theorem clock_angle_at_13_20_is_80 : clock_angle_at_13_20 = 80 := by
  sorry

end NUMINAMATH_CALUDE_clock_angle_at_13_20_clock_angle_at_13_20_is_80_l3271_327104


namespace NUMINAMATH_CALUDE_correct_number_of_pitchers_l3271_327183

/-- The number of glasses each pitcher can serve -/
def glasses_per_pitcher : ℕ := 5

/-- The total number of glasses served -/
def total_glasses_served : ℕ := 30

/-- The number of pitchers prepared -/
def pitchers_prepared : ℕ := total_glasses_served / glasses_per_pitcher

theorem correct_number_of_pitchers : pitchers_prepared = 6 := by
  sorry

end NUMINAMATH_CALUDE_correct_number_of_pitchers_l3271_327183


namespace NUMINAMATH_CALUDE_points_on_angle_bisector_l3271_327129

/-- Given two points A and B, proves that if they lie on the angle bisector of the first and third quadrants, their coordinates satisfy specific conditions. -/
theorem points_on_angle_bisector 
  (a b : ℝ) 
  (A : ℝ × ℝ) 
  (B : ℝ × ℝ) 
  (h1 : A = (a - 1, 2)) 
  (h2 : B = (-3, b + 1)) 
  (h3 : (a - 1) = 2 ∧ (b + 1) = -3) : 
  a = 3 ∧ b = -4 := by
  sorry

end NUMINAMATH_CALUDE_points_on_angle_bisector_l3271_327129


namespace NUMINAMATH_CALUDE_storks_and_birds_difference_l3271_327130

theorem storks_and_birds_difference : 
  ∀ (initial_birds initial_storks additional_storks : ℕ),
    initial_birds = 4 →
    initial_storks = 3 →
    additional_storks = 6 →
    (initial_storks + additional_storks) - initial_birds = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_storks_and_birds_difference_l3271_327130


namespace NUMINAMATH_CALUDE_prob_at_least_one_X_correct_l3271_327128

/-- Represents the probability of selecting at least one person who used model X
    when randomly selecting 2 people from a group of 5, where 3 used model X and 2 used model Y. -/
def prob_at_least_one_X : ℚ := 9 / 10

/-- The total number of people in the experience group -/
def total_people : ℕ := 5

/-- The number of people who used model X bicycles -/
def model_X_users : ℕ := 3

/-- The number of people who used model Y bicycles -/
def model_Y_users : ℕ := 2

/-- The number of ways to select 2 people from the group -/
def total_selections : ℕ := total_people.choose 2

/-- The number of ways to select 2 people who both used model Y -/
def both_Y_selections : ℕ := model_Y_users.choose 2

theorem prob_at_least_one_X_correct :
  prob_at_least_one_X = 1 - (both_Y_selections : ℚ) / total_selections :=
by sorry

end NUMINAMATH_CALUDE_prob_at_least_one_X_correct_l3271_327128


namespace NUMINAMATH_CALUDE_combined_pastures_capacity_l3271_327145

/-- Represents the capacity of a pasture -/
structure Pasture where
  area : ℝ
  cattleCapacity : ℕ
  daysCapacity : ℕ

/-- Calculates the total grass units a pasture can provide -/
def totalGrassUnits (p : Pasture) : ℝ :=
  p.area * (p.cattleCapacity : ℝ) * (p.daysCapacity : ℝ)

/-- Theorem: Combined pastures can feed 250 cattle for 28 days -/
theorem combined_pastures_capacity 
  (pastureA : Pasture)
  (pastureB : Pasture)
  (h1 : pastureA.area = 3)
  (h2 : pastureB.area = 4)
  (h3 : pastureA.cattleCapacity = 90)
  (h4 : pastureA.daysCapacity = 36)
  (h5 : pastureB.cattleCapacity = 160)
  (h6 : pastureB.daysCapacity = 24)
  (h7 : totalGrassUnits pastureA + totalGrassUnits pastureB = 
        (pastureA.area + pastureB.area) * 250 * 28) :
  ∃ (combinedPasture : Pasture), 
    combinedPasture.area = pastureA.area + pastureB.area ∧
    combinedPasture.cattleCapacity = 250 ∧
    combinedPasture.daysCapacity = 28 :=
  sorry

end NUMINAMATH_CALUDE_combined_pastures_capacity_l3271_327145


namespace NUMINAMATH_CALUDE_calculate_unknown_leak_rate_l3271_327123

/-- Calculates the unknown leak rate after the first repair --/
theorem calculate_unknown_leak_rate (initial_capacity : ℕ) (first_leak_rate : ℕ) (first_leak_duration : ℕ)
  (second_leak_duration : ℕ) (fill_rate : ℕ) (fill_duration : ℕ) (final_deficit : ℕ) :
  initial_capacity = 350000 →
  first_leak_rate = 32000 →
  first_leak_duration = 5 →
  second_leak_duration = 10 →
  fill_rate = 40000 →
  fill_duration = 3 →
  final_deficit = 140000 →
  ∃ (x : ℕ),
    initial_capacity - (first_leak_rate * first_leak_duration) - (x * second_leak_duration) + (fill_rate * fill_duration) =
    initial_capacity - final_deficit ∧
    x = 10000 :=
by sorry

end NUMINAMATH_CALUDE_calculate_unknown_leak_rate_l3271_327123


namespace NUMINAMATH_CALUDE_arithmetic_sequence_length_l3271_327195

theorem arithmetic_sequence_length : 
  ∀ (a₁ : ℕ) (aₙ : ℕ) (d : ℕ),
    a₁ = 4 →
    aₙ = 130 →
    d = 2 →
    (aₙ - a₁) / d + 1 = 64 :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_length_l3271_327195


namespace NUMINAMATH_CALUDE_solution_difference_l3271_327175

theorem solution_difference (r s : ℝ) : 
  r ≠ s →
  (6 * r - 18) / (r^2 + 2*r - 15) = r + 3 →
  (6 * s - 18) / (s^2 + 2*s - 15) = s + 3 →
  r > s →
  r - s = 8 := by sorry

end NUMINAMATH_CALUDE_solution_difference_l3271_327175


namespace NUMINAMATH_CALUDE_winnie_lollipops_l3271_327162

/-- The number of lollipops Winnie keeps for herself -/
def lollipops_kept (cherry wintergreen grape shrimp friends : ℕ) : ℕ :=
  (cherry + wintergreen + grape + shrimp) % friends

theorem winnie_lollipops :
  lollipops_kept 36 125 8 241 13 = 7 := by
  sorry

end NUMINAMATH_CALUDE_winnie_lollipops_l3271_327162


namespace NUMINAMATH_CALUDE_prism_volume_l3271_327153

/-- The volume of a right rectangular prism with given face areas -/
theorem prism_volume (side_area front_area bottom_area : ℝ) 
  (h_side : side_area = 18)
  (h_front : front_area = 12)
  (h_bottom : bottom_area = 8) :
  ∃ x y z : ℝ, 
    x * y = side_area ∧ 
    y * z = front_area ∧ 
    x * z = bottom_area ∧ 
    x * y * z = 24 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_prism_volume_l3271_327153


namespace NUMINAMATH_CALUDE_problem_solution_l3271_327127

def A : Set ℝ := {x | x^2 - x - 6 < 0}
def B : Set ℝ := {x | x > -3 ∧ x ≤ 3}

theorem problem_solution :
  (A = {x | -2 < x ∧ x < 3}) ∧
  (Set.compl (A ∩ B) = {x | x ≤ -2 ∨ x ≥ 3}) ∧
  ((Set.compl A) ∩ B = {x | -3 < x ∧ x ≤ -2 ∨ x = 3}) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3271_327127


namespace NUMINAMATH_CALUDE_thirteen_people_evaluations_l3271_327115

/-- The number of evaluations for a group of people, where each pair is categorized into one of three categories. -/
def num_evaluations (n : ℕ) : ℕ := n.choose 2 * 3

/-- Theorem: For a group of 13 people, where each pair is categorized into one of three categories, the total number of evaluations is 234. -/
theorem thirteen_people_evaluations : num_evaluations 13 = 234 := by
  sorry

end NUMINAMATH_CALUDE_thirteen_people_evaluations_l3271_327115


namespace NUMINAMATH_CALUDE_A_intersect_B_l3271_327146

def A : Set ℝ := {x | (2*x - 6) / (x + 1) ≤ 0}
def B : Set ℝ := {-2, -1, 0, 3, 4}

theorem A_intersect_B : A ∩ B = {0, 3} := by sorry

end NUMINAMATH_CALUDE_A_intersect_B_l3271_327146


namespace NUMINAMATH_CALUDE_tips_fraction_is_one_third_l3271_327190

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

end NUMINAMATH_CALUDE_tips_fraction_is_one_third_l3271_327190


namespace NUMINAMATH_CALUDE_house_rent_calculation_l3271_327163

def salary : ℚ := 170000

def food_fraction : ℚ := 1/5
def clothes_fraction : ℚ := 3/5
def remaining_amount : ℚ := 17000

def house_rent_fraction : ℚ := 1/10

theorem house_rent_calculation :
  house_rent_fraction * salary + food_fraction * salary + clothes_fraction * salary + remaining_amount = salary :=
by sorry

end NUMINAMATH_CALUDE_house_rent_calculation_l3271_327163


namespace NUMINAMATH_CALUDE_complement_subset_l3271_327131

-- Define the set M
def M : Set ℝ := {x | 0 < x ∧ x < 2}

-- Define the set N
def N : Set ℝ := {x | x^2 + x - 6 ≤ 0}

-- Theorem statement
theorem complement_subset : Set.compl N ⊆ Set.compl M := by
  sorry

end NUMINAMATH_CALUDE_complement_subset_l3271_327131


namespace NUMINAMATH_CALUDE_prob_adjacent_vertices_octagon_l3271_327159

/-- An octagon is a polygon with 8 vertices -/
def Octagon : Type := Unit

/-- The number of vertices in an octagon -/
def num_vertices (o : Octagon) : ℕ := 8

/-- The number of adjacent vertices for any vertex in an octagon -/
def num_adjacent (o : Octagon) : ℕ := 2

/-- The probability of choosing two distinct adjacent vertices in an octagon -/
def prob_adjacent_vertices (o : Octagon) : ℚ :=
  (num_adjacent o : ℚ) / ((num_vertices o - 1) : ℚ)

theorem prob_adjacent_vertices_octagon :
  ∀ o : Octagon, prob_adjacent_vertices o = 2 / 7 := by
  sorry

end NUMINAMATH_CALUDE_prob_adjacent_vertices_octagon_l3271_327159


namespace NUMINAMATH_CALUDE_range_of_f_l3271_327116

-- Define the function f
def f (x : ℝ) : ℝ := -x^2 + 4*x

-- Define the domain
def domain : Set ℝ := { x | 0 ≤ x ∧ x ≤ 2 }

-- Theorem statement
theorem range_of_f :
  { y | ∃ x ∈ domain, f x = y } = { y | 0 ≤ y ∧ y ≤ 4 } := by sorry

end NUMINAMATH_CALUDE_range_of_f_l3271_327116


namespace NUMINAMATH_CALUDE_fruit_bowl_oranges_l3271_327111

theorem fruit_bowl_oranges (bananas apples oranges : ℕ) : 
  bananas = 2 → 
  apples = 2 * bananas → 
  bananas + apples + oranges = 12 → 
  oranges = 6 := by
sorry

end NUMINAMATH_CALUDE_fruit_bowl_oranges_l3271_327111


namespace NUMINAMATH_CALUDE_arithmetic_sequence_tenth_term_l3271_327117

/-- An arithmetic sequence is a sequence where the difference between 
    any two consecutive terms is constant. -/
def ArithmeticSequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_tenth_term
  (a : ℕ → ℚ)
  (h_arithmetic : ArithmeticSequence a)
  (h_third_term : a 3 = 5)
  (h_seventh_term : a 7 = 13) :
  a 10 = 19 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_tenth_term_l3271_327117


namespace NUMINAMATH_CALUDE_brandys_trail_mix_raisins_l3271_327112

/-- The weight of raisins in a trail mix -/
def weight_of_raisins (weight_of_peanuts weight_of_chips total_weight : Real) : Real :=
  total_weight - (weight_of_peanuts + weight_of_chips)

/-- Theorem stating the weight of raisins in Brandy's trail mix -/
theorem brandys_trail_mix_raisins : 
  weight_of_raisins 0.17 0.17 0.42 = 0.08 := by
  sorry

end NUMINAMATH_CALUDE_brandys_trail_mix_raisins_l3271_327112


namespace NUMINAMATH_CALUDE_sum_of_roots_l3271_327106

noncomputable def f (x : ℝ) : ℝ := x^2 - 2*|x + 4| - 27

theorem sum_of_roots : ∃ (r₁ r₂ : ℝ), f r₁ = 0 ∧ f r₂ = 0 ∧ r₁ + r₂ = 6 - Real.sqrt 20 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_l3271_327106


namespace NUMINAMATH_CALUDE_three_digit_number_relation_l3271_327179

theorem three_digit_number_relation (h t u : ℕ) : 
  h ≥ 1 ∧ h ≤ 9 ∧  -- h is a single digit
  t ≥ 0 ∧ t ≤ 9 ∧  -- t is a single digit
  u ≥ 0 ∧ u ≤ 9 ∧  -- u is a single digit
  h = t + 2 ∧      -- hundreds digit is 2 more than tens digit
  h + t + u = 27   -- sum of digits is 27
  → ∃ (r : ℕ → ℕ → Prop), r t u  -- there exists some relation r between t and u
:= by sorry

end NUMINAMATH_CALUDE_three_digit_number_relation_l3271_327179


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l3271_327141

theorem quadratic_inequality_range (m : ℝ) :
  (∀ x : ℝ, m * x^2 - m * x - 1 < 0) ↔ -4 < m ∧ m ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l3271_327141


namespace NUMINAMATH_CALUDE_line_through_point_with_given_segment_length_l3271_327155

-- Define the angle BAC
def Angle (A B C : ℝ × ℝ) : Prop := sorry

-- Define a point on the angle bisector
def OnAngleBisector (D A B C : ℝ × ℝ) : Prop := sorry

-- Define a line passing through two points
def Line (P Q : ℝ × ℝ) : Set (ℝ × ℝ) := sorry

-- Define the length of a segment
def SegmentLength (P Q : ℝ × ℝ) : ℝ := sorry

-- Define a point being on a line
def OnLine (P : ℝ × ℝ) (L : Set (ℝ × ℝ)) : Prop := sorry

theorem line_through_point_with_given_segment_length 
  (A B C D : ℝ × ℝ) (l : ℝ) 
  (h1 : Angle A B C) 
  (h2 : OnAngleBisector D A B C) 
  (h3 : l > 0) : 
  ∃ (E F : ℝ × ℝ), 
    OnLine E (Line A B) ∧ 
    OnLine F (Line A C) ∧ 
    OnLine D (Line E F) ∧ 
    SegmentLength E F = l := 
sorry

end NUMINAMATH_CALUDE_line_through_point_with_given_segment_length_l3271_327155


namespace NUMINAMATH_CALUDE_honor_distribution_proof_l3271_327118

/-- The number of ways to distribute honors among people -/
def distribute_honors (num_honors num_people : ℕ) (incompatible_pair : Bool) : ℕ :=
  sorry

/-- The number of ways to distribute honors in the specific problem -/
def problem_distribution : ℕ :=
  distribute_honors 5 3 true

theorem honor_distribution_proof :
  problem_distribution = 114 := by sorry

end NUMINAMATH_CALUDE_honor_distribution_proof_l3271_327118


namespace NUMINAMATH_CALUDE_trig_inequality_and_equality_condition_l3271_327192

theorem trig_inequality_and_equality_condition (α β : Real) 
  (h1 : 0 < α) (h2 : α < π/2) (h3 : 0 < β) (h4 : β < π/2) :
  (1 / (Real.cos α)^2 + 1 / ((Real.sin α)^2 * (Real.sin β)^2 * (Real.cos β)^2) ≥ 9) ∧
  (1 / (Real.cos α)^2 + 1 / ((Real.sin α)^2 * (Real.sin β)^2 * (Real.cos β)^2) = 9 ↔ 
    α = Real.arctan (Real.sqrt 2) ∧ β = π/4) :=
by sorry

end NUMINAMATH_CALUDE_trig_inequality_and_equality_condition_l3271_327192


namespace NUMINAMATH_CALUDE_choose_computers_l3271_327108

theorem choose_computers (n : ℕ) : 
  (Nat.choose 3 2 * Nat.choose 3 1) + (Nat.choose 3 1 * Nat.choose 3 2) = 18 :=
by sorry

end NUMINAMATH_CALUDE_choose_computers_l3271_327108


namespace NUMINAMATH_CALUDE_prime_pairs_divisibility_l3271_327197

theorem prime_pairs_divisibility (p q : ℕ) : 
  Prime p → Prime q → p ≤ q → (p * q) ∣ ((5^p - 2^p) * (7^q - 2^q)) →
  ((p = 3 ∧ q = 5) ∨ (p = 3 ∧ q = 3) ∨ (p = 5 ∧ q = 37) ∨ (p = 5 ∧ q = 83)) :=
by sorry

end NUMINAMATH_CALUDE_prime_pairs_divisibility_l3271_327197


namespace NUMINAMATH_CALUDE_train_length_calculation_l3271_327170

/-- The length of a train that passes a tree in 12 seconds while traveling at 90 km/hr is 300 meters. -/
theorem train_length_calculation (passing_time : ℝ) (speed_kmh : ℝ) : 
  passing_time = 12 → speed_kmh = 90 → passing_time * (speed_kmh * (1000 / 3600)) = 300 := by
  sorry

end NUMINAMATH_CALUDE_train_length_calculation_l3271_327170


namespace NUMINAMATH_CALUDE_find_n_l3271_327186

theorem find_n : ∃ n : ℕ, 
  50 ≤ n ∧ n ≤ 120 ∧ 
  n % 8 = 0 ∧ 
  n % 12 = 4 ∧ 
  n % 7 = 4 ∧
  n = 88 := by
  sorry

end NUMINAMATH_CALUDE_find_n_l3271_327186


namespace NUMINAMATH_CALUDE_gcd_power_two_minus_one_l3271_327176

theorem gcd_power_two_minus_one (a b : ℕ+) :
  Nat.gcd ((2 : ℕ) ^ a.val - 1) ((2 : ℕ) ^ b.val - 1) = (2 : ℕ) ^ (Nat.gcd a.val b.val) - 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_power_two_minus_one_l3271_327176


namespace NUMINAMATH_CALUDE_odd_square_mod_eight_l3271_327193

theorem odd_square_mod_eight (k : ℤ) : ∃ m : ℤ, (2 * k + 1)^2 = 8 * m + 1 := by
  sorry

end NUMINAMATH_CALUDE_odd_square_mod_eight_l3271_327193


namespace NUMINAMATH_CALUDE_unique_k_solution_l3271_327124

theorem unique_k_solution : ∃! k : ℕ, 10^k - 1 = 9*k^2 := by sorry

end NUMINAMATH_CALUDE_unique_k_solution_l3271_327124


namespace NUMINAMATH_CALUDE_power_5_2023_mod_17_l3271_327152

theorem power_5_2023_mod_17 : (5 : ℤ) ^ 2023 % 17 = 2 := by sorry

end NUMINAMATH_CALUDE_power_5_2023_mod_17_l3271_327152


namespace NUMINAMATH_CALUDE_sharks_score_l3271_327109

theorem sharks_score (total_points eagles_points sharks_points : ℕ) : 
  total_points = 60 → 
  eagles_points = sharks_points + 18 → 
  eagles_points + sharks_points = total_points → 
  sharks_points = 21 := by
sorry

end NUMINAMATH_CALUDE_sharks_score_l3271_327109


namespace NUMINAMATH_CALUDE_go_match_probability_l3271_327160

/-- The probability that two more games will conclude a Go match given the specified conditions -/
theorem go_match_probability (p_a : ℝ) (p_b : ℝ) : 
  p_a = 0.6 →
  p_b = 0.4 →
  p_a + p_b = 1 →
  (p_a ^ 2 + p_b ^ 2 : ℝ) = 0.52 := by
  sorry

end NUMINAMATH_CALUDE_go_match_probability_l3271_327160


namespace NUMINAMATH_CALUDE_max_y_value_l3271_327149

theorem max_y_value (x y : ℝ) :
  (Real.log (x + y) / Real.log (x^2 + y^2) ≥ 1) →
  y ≤ 1/2 + Real.sqrt 2 / 2 :=
by sorry

end NUMINAMATH_CALUDE_max_y_value_l3271_327149


namespace NUMINAMATH_CALUDE_inequality_bound_l3271_327156

theorem inequality_bound (m : ℝ) : 
  (∀ x : ℝ, |x - 1| + |x + 2| ≥ m) → m ≤ 3 :=
by
  sorry

end NUMINAMATH_CALUDE_inequality_bound_l3271_327156


namespace NUMINAMATH_CALUDE_complement_intersection_MN_l3271_327110

-- Define the universe set U
def U : Set ℕ := {1, 2, 3, 4}

-- Define set M
def M : Set ℕ := {1, 2}

-- Define set N
def N : Set ℕ := {2, 3}

-- Theorem statement
theorem complement_intersection_MN : 
  (M ∩ N)ᶜ = {1, 3, 4} :=
by sorry

end NUMINAMATH_CALUDE_complement_intersection_MN_l3271_327110


namespace NUMINAMATH_CALUDE_no_consec_nat_prod_equals_consec_even_prod_l3271_327180

theorem no_consec_nat_prod_equals_consec_even_prod : 
  ¬∃ (m n : ℕ), m * (m + 1) = 4 * n * (n + 1) := by
sorry

end NUMINAMATH_CALUDE_no_consec_nat_prod_equals_consec_even_prod_l3271_327180


namespace NUMINAMATH_CALUDE_quadratic_no_real_roots_l3271_327165

theorem quadratic_no_real_roots (a : ℝ) : 
  (¬ ∃ x : ℝ, x^2 - a*x + 1 = 0) ↔ -2 < a ∧ a < 2 :=
sorry

end NUMINAMATH_CALUDE_quadratic_no_real_roots_l3271_327165


namespace NUMINAMATH_CALUDE_stewart_farm_sheep_count_l3271_327134

def sheep_horse_ratio : ℚ := 2 / 7
def horse_food_per_day : ℕ := 230
def total_horse_food : ℕ := 12880

theorem stewart_farm_sheep_count :
  ∃ (sheep horses : ℕ),
    sheep / horses = sheep_horse_ratio ∧
    horses * horse_food_per_day = total_horse_food ∧
    sheep = 16 := by sorry

end NUMINAMATH_CALUDE_stewart_farm_sheep_count_l3271_327134


namespace NUMINAMATH_CALUDE_final_balance_percentage_l3271_327182

def starting_balance : ℝ := 125
def initial_increase : ℝ := 0.25
def first_usd_to_eur : ℝ := 0.85
def decrease_in_eur : ℝ := 0.20
def eur_to_usd : ℝ := 1.15
def increase_in_usd : ℝ := 0.15
def decrease_in_usd : ℝ := 0.10
def final_usd_to_eur : ℝ := 0.88

theorem final_balance_percentage (starting_balance initial_increase first_usd_to_eur
  decrease_in_eur eur_to_usd increase_in_usd decrease_in_usd final_usd_to_eur : ℝ) :
  let initial_eur := starting_balance * (1 + initial_increase) * first_usd_to_eur
  let after_decrease_eur := initial_eur * (1 - decrease_in_eur)
  let back_to_usd := after_decrease_eur * eur_to_usd
  let after_increase_usd := back_to_usd * (1 + increase_in_usd)
  let after_decrease_usd := after_increase_usd * (1 - decrease_in_usd)
  let final_eur := after_decrease_usd * final_usd_to_eur
  let starting_eur := starting_balance * first_usd_to_eur
  (final_eur / starting_eur) * 100 = 104.75 :=
by sorry

end NUMINAMATH_CALUDE_final_balance_percentage_l3271_327182


namespace NUMINAMATH_CALUDE_train_speed_train_speed_is_72_l3271_327119

/-- Given a train that crosses a platform and a stationary man, calculate its speed in km/h -/
theorem train_speed (platform_length : ℝ) (platform_time : ℝ) (man_time : ℝ) : ℝ :=
  let train_speed_mps := platform_length / (platform_time - man_time)
  let train_speed_kmph := train_speed_mps * 3.6
  train_speed_kmph

/-- The speed of the train is 72 km/h -/
theorem train_speed_is_72 : 
  train_speed 260 31 18 = 72 := by sorry

end NUMINAMATH_CALUDE_train_speed_train_speed_is_72_l3271_327119


namespace NUMINAMATH_CALUDE_unique_prime_base_1021_l3271_327172

theorem unique_prime_base_1021 : ∃! (n : ℕ), n ≥ 2 ∧ Nat.Prime (n^3 + 2*n + 1) :=
sorry

end NUMINAMATH_CALUDE_unique_prime_base_1021_l3271_327172


namespace NUMINAMATH_CALUDE_parallel_vectors_imply_m_value_l3271_327101

def a (m : ℝ) : Fin 2 → ℝ := ![1, m]
def b : Fin 2 → ℝ := ![2, 5]
def c (m : ℝ) : Fin 2 → ℝ := ![m, 3]

theorem parallel_vectors_imply_m_value :
  ∀ m : ℝ,
  (∃ k : ℝ, k ≠ 0 ∧ (a m + c m) = k • (a m - b)) →
  (m = (3 + Real.sqrt 17) / 2 ∨ m = (3 - Real.sqrt 17) / 2) :=
by sorry

end NUMINAMATH_CALUDE_parallel_vectors_imply_m_value_l3271_327101


namespace NUMINAMATH_CALUDE_like_terms_value_l3271_327102

theorem like_terms_value (m n : ℕ) (a b c : ℝ) : 
  (∃ k : ℝ, 3 * a^m * b * c^2 = k * (-2 * a^3 * b^n * c^2)) → 
  3^2 * n - (2 * m * n^2 - 2 * (m^2 * n + 2 * m * n^2)) = 51 :=
by sorry

end NUMINAMATH_CALUDE_like_terms_value_l3271_327102


namespace NUMINAMATH_CALUDE_greatest_common_divisor_of_180_and_n_l3271_327191

/-- Given two positive integers 180 and n that share exactly five positive divisors,
    the greatest of these five common divisors is 27. -/
theorem greatest_common_divisor_of_180_and_n : 
  ∀ n : ℕ+, 
  (∃! (s : Finset ℕ+), s.card = 5 ∧ (∀ d ∈ s, d ∣ 180 ∧ d ∣ n)) → 
  (∃ (s : Finset ℕ+), s.card = 5 ∧ (∀ d ∈ s, d ∣ 180 ∧ d ∣ n) ∧ 27 ∈ s ∧ ∀ x ∈ s, x ≤ 27) :=
by sorry


end NUMINAMATH_CALUDE_greatest_common_divisor_of_180_and_n_l3271_327191


namespace NUMINAMATH_CALUDE_ellipse_chord_theorem_l3271_327158

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 25 + y^2 / 16 = 1

-- Define the foci
def left_focus : ℝ × ℝ := (-3, 0)
def right_focus : ℝ × ℝ := (3, 0)

-- Define a chord passing through the left focus
def chord_through_left_focus (x1 y1 x2 y2 : ℝ) : Prop :=
  is_on_ellipse x1 y1 ∧ is_on_ellipse x2 y2 ∧
  ∃ t : ℝ, (1 - t) * x1 + t * x2 = -3 ∧ (1 - t) * y1 + t * y2 = 0

-- Define the incircle circumference condition
def incircle_circumference_2pi (x1 y1 x2 y2 : ℝ) : Prop :=
  ∃ r : ℝ, r * (Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2) +
               Real.sqrt ((x1 - 3)^2 + y1^2) +
               Real.sqrt ((x2 - 3)^2 + y2^2)) = 10 ∧
           2 * Real.pi * r = 2 * Real.pi

theorem ellipse_chord_theorem (x1 y1 x2 y2 : ℝ) :
  chord_through_left_focus x1 y1 x2 y2 →
  incircle_circumference_2pi x1 y1 x2 y2 →
  |y1 - y2| = 10 / 3 := by sorry

end NUMINAMATH_CALUDE_ellipse_chord_theorem_l3271_327158


namespace NUMINAMATH_CALUDE_ding_score_is_97_l3271_327147

-- Define the average score of Jia, Yi, and Bing
def avg_three : ℝ := 89

-- Define Ding's score
def ding_score : ℝ := 97

-- Define the average score of all four people
def avg_four : ℝ := avg_three + 2

-- Theorem statement
theorem ding_score_is_97 :
  ding_score = 4 * avg_four - 3 * avg_three :=
by sorry

end NUMINAMATH_CALUDE_ding_score_is_97_l3271_327147


namespace NUMINAMATH_CALUDE_car_overtake_distance_l3271_327105

/-- Proves that the initial distance between two cars is equal to the product of their relative speed and the overtaking time. -/
theorem car_overtake_distance (v_red v_black : ℝ) (t : ℝ) (h1 : 0 < v_red) (h2 : v_red < v_black) (h3 : 0 < t) :
  (v_black - v_red) * t = (v_black - v_red) * t :=
by sorry

/-- Calculates the initial distance between two cars given their speeds and overtaking time. -/
def initial_distance (v_red v_black t : ℝ) : ℝ :=
  (v_black - v_red) * t

#check car_overtake_distance
#check initial_distance

end NUMINAMATH_CALUDE_car_overtake_distance_l3271_327105


namespace NUMINAMATH_CALUDE_trigonometric_identities_l3271_327135

open Real

theorem trigonometric_identities (α : ℝ) :
  (tan α = 1 / 3) →
  (1 / (2 * sin α * cos α + cos α ^ 2) = 2 / 3) ∧
  (tan (π - α) * cos (2 * π - α) * sin (-α + 3 * π / 2)) / (cos (-α - π) * sin (-π - α)) = -1 :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l3271_327135


namespace NUMINAMATH_CALUDE_function_geq_square_for_k_geq_4_l3271_327121

def is_increasing_square (f : ℕ+ → ℝ) : Prop :=
  ∀ k : ℕ+, f k ≥ k^2 → f (k + 1) ≥ (k + 1)^2

theorem function_geq_square_for_k_geq_4
  (f : ℕ+ → ℝ)
  (h_increasing : is_increasing_square f)
  (h_f4 : f 4 = 25) :
  ∀ k : ℕ+, k ≥ 4 → f k ≥ k^2 :=
sorry

end NUMINAMATH_CALUDE_function_geq_square_for_k_geq_4_l3271_327121


namespace NUMINAMATH_CALUDE_center_cell_value_l3271_327151

theorem center_cell_value (a b c d e f g h i : ℝ) : 
  (a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧ f > 0 ∧ g > 0 ∧ h > 0 ∧ i > 0) →
  (a * b * c = 1) →
  (d * e * f = 1) →
  (g * h * i = 1) →
  (a * d * g = 1) →
  (b * e * h = 1) →
  (c * f * i = 1) →
  (a * b * d * e = 2) →
  (b * c * e * f = 2) →
  (d * e * g * h = 2) →
  (e * f * h * i = 2) →
  e = 1 :=
by sorry

end NUMINAMATH_CALUDE_center_cell_value_l3271_327151


namespace NUMINAMATH_CALUDE_octahedron_tetrahedron_combination_l3271_327157

/-- Represents a regular octahedron --/
structure RegularOctahedron :=
  (edge_length : ℝ)

/-- Represents a regular tetrahedron --/
structure RegularTetrahedron :=
  (edge_length : ℝ)

/-- Theorem stating that it's possible to combine six regular octahedrons and eight regular tetrahedrons
    to form a larger regular octahedron with twice the edge length --/
theorem octahedron_tetrahedron_combination
  (small_octahedrons : Fin 6 → RegularOctahedron)
  (tetrahedrons : Fin 8 → RegularTetrahedron)
  (h1 : ∀ i j, small_octahedrons i = small_octahedrons j)  -- All small octahedrons are congruent
  (h2 : ∀ i, (tetrahedrons i).edge_length = (small_octahedrons 0).edge_length)  -- Tetrahedron edges equal octahedron edges
  : ∃ (large_octahedron : RegularOctahedron),
    large_octahedron.edge_length = 2 * (small_octahedrons 0).edge_length :=
by sorry

end NUMINAMATH_CALUDE_octahedron_tetrahedron_combination_l3271_327157


namespace NUMINAMATH_CALUDE_four_greater_than_sqrt_fifteen_l3271_327168

theorem four_greater_than_sqrt_fifteen : 4 > Real.sqrt 15 := by
  sorry

end NUMINAMATH_CALUDE_four_greater_than_sqrt_fifteen_l3271_327168


namespace NUMINAMATH_CALUDE_decagon_diagonal_intersection_probability_l3271_327122

/-- A regular decagon is a 10-sided polygon with all sides and angles equal -/
def RegularDecagon : Type := Unit

/-- The number of diagonals in a regular decagon -/
def num_diagonals (d : RegularDecagon) : ℕ := 35

/-- The number of ways to choose three diagonals that intersect at a single point -/
def num_intersecting_diagonals (d : RegularDecagon) : ℕ := 840

/-- The total number of ways to choose three diagonals -/
def total_diagonal_choices (d : RegularDecagon) : ℕ := 6545

/-- The probability that three randomly chosen diagonals in a regular decagon
    intersect at a single point inside the decagon -/
theorem decagon_diagonal_intersection_probability (d : RegularDecagon) :
  (num_intersecting_diagonals d : ℚ) / (total_diagonal_choices d : ℚ) = 840 / 6545 := by
  sorry

end NUMINAMATH_CALUDE_decagon_diagonal_intersection_probability_l3271_327122


namespace NUMINAMATH_CALUDE_pizza_group_composition_l3271_327113

theorem pizza_group_composition :
  ∀ (boys girls : ℕ),
  (∀ (b : ℕ), b ≤ boys → 6 ≤ b ∧ b ≤ 7) →
  (∀ (g : ℕ), g ≤ girls → 2 ≤ g ∧ g ≤ 3) →
  49 ≤ 6 * boys + 2 * girls →
  7 * boys + 3 * girls ≤ 59 →
  boys = 8 ∧ girls = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_pizza_group_composition_l3271_327113


namespace NUMINAMATH_CALUDE_nina_total_spending_l3271_327126

/-- The total cost of Nina's purchases --/
def total_cost (toy_price toy_quantity card_price card_quantity shirt_price shirt_quantity : ℕ) : ℕ :=
  toy_price * toy_quantity + card_price * card_quantity + shirt_price * shirt_quantity

/-- Theorem stating that Nina's total spending is $70 --/
theorem nina_total_spending :
  total_cost 10 3 5 2 6 5 = 70 := by
  sorry

end NUMINAMATH_CALUDE_nina_total_spending_l3271_327126


namespace NUMINAMATH_CALUDE_islander_group_composition_l3271_327154

/-- Represents the type of islander: knight or liar -/
inductive IslanderType
| Knight
| Liar

/-- Represents an islander's statement about the group composition -/
inductive Statement
| MoreLiars
| MoreKnights
| Equal

/-- A function that returns the true statement about group composition -/
def trueStatement (knights liars : Nat) : Statement :=
  if knights > liars then Statement.MoreKnights
  else if liars > knights then Statement.MoreLiars
  else Statement.Equal

/-- A function that determines what an islander would say based on their type and the true group composition -/
def islanderStatement (type : IslanderType) (knights liars : Nat) : Statement :=
  match type with
  | IslanderType.Knight => trueStatement knights liars
  | IslanderType.Liar => 
    match trueStatement knights liars with
    | Statement.MoreLiars => Statement.MoreKnights
    | Statement.MoreKnights => Statement.MoreLiars
    | Statement.Equal => Statement.MoreLiars  -- Arbitrarily chosen, could be MoreKnights as well

theorem islander_group_composition 
  (total : Nat) 
  (h_total : total = 10) 
  (knights liars : Nat) 
  (h_sum : knights + liars = total) 
  (h_five_more_liars : ∃ (group : Finset IslanderType), 
    group.card = 5 ∧ 
    ∀ i ∈ group, islanderStatement i knights liars = Statement.MoreLiars) :
  knights = liars ∧ 
  ∃ (other_group : Finset IslanderType), 
    other_group.card = 5 ∧ 
    ∀ i ∈ other_group, islanderStatement i knights liars = Statement.Equal :=
sorry


end NUMINAMATH_CALUDE_islander_group_composition_l3271_327154


namespace NUMINAMATH_CALUDE_shaded_area_of_folded_rectangle_l3271_327136

/-- The area of the shaded region formed by folding a rectangular sheet along its diagonal -/
theorem shaded_area_of_folded_rectangle (length width : ℝ) (h_length : length = 12) (h_width : width = 18) :
  let rectangle_area := length * width
  let diagonal := Real.sqrt (length^2 + width^2)
  let triangle_area := (1 / 2) * diagonal * diagonal * (2 / 3)
  rectangle_area - triangle_area = 138 := by sorry

end NUMINAMATH_CALUDE_shaded_area_of_folded_rectangle_l3271_327136


namespace NUMINAMATH_CALUDE_binary_division_remainder_l3271_327173

theorem binary_division_remainder (n : ℕ) (h : n = 0b111001011110) : n % 4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_binary_division_remainder_l3271_327173


namespace NUMINAMATH_CALUDE_income_calculation_l3271_327174

/-- Represents a person's financial situation -/
structure FinancialSituation where
  income : ℕ
  expenditure : ℕ
  savings : ℕ

/-- Theorem: Given a person's financial situation where the income to expenditure ratio
    is 3:2 and savings are 7000, the income is 21000 -/
theorem income_calculation (fs : FinancialSituation) 
  (h1 : fs.income = 3 * (fs.expenditure / 2))
  (h2 : fs.savings = 7000)
  (h3 : fs.income = fs.expenditure + fs.savings) : 
  fs.income = 21000 := by
  sorry

end NUMINAMATH_CALUDE_income_calculation_l3271_327174


namespace NUMINAMATH_CALUDE_soda_consumption_per_person_l3271_327161

/- Define the problem parameters -/
def people_attending : ℕ := 5 * 12  -- five dozens
def cans_per_box : ℕ := 10
def cost_per_box : ℕ := 2
def family_members : ℕ := 6
def payment_per_member : ℕ := 4

/- Define the theorem -/
theorem soda_consumption_per_person :
  let total_payment := family_members * payment_per_member
  let boxes_bought := total_payment / cost_per_box
  let total_cans := boxes_bought * cans_per_box
  total_cans / people_attending = 2 := by sorry

end NUMINAMATH_CALUDE_soda_consumption_per_person_l3271_327161


namespace NUMINAMATH_CALUDE_flower_purchase_cost_katie_flower_purchase_cost_l3271_327184

/-- The cost of buying roses and daisies at a fixed price per flower -/
theorem flower_purchase_cost 
  (price_per_flower : ℕ) 
  (num_roses : ℕ) 
  (num_daisies : ℕ) : 
  price_per_flower * (num_roses + num_daisies) = 
    price_per_flower * num_roses + price_per_flower * num_daisies :=
by sorry

/-- The total cost of Katie's flower purchase -/
theorem katie_flower_purchase_cost : 
  (5 : ℕ) + 5 = 10 ∧ 6 * 10 = 60 :=
by sorry

end NUMINAMATH_CALUDE_flower_purchase_cost_katie_flower_purchase_cost_l3271_327184


namespace NUMINAMATH_CALUDE_zero_discriminant_implies_geometric_progression_l3271_327198

/-- Given a quadratic equation ax^2 + 3bx + c = 0 with zero discriminant,
    prove that a, b, and c form a geometric progression. -/
theorem zero_discriminant_implies_geometric_progression
  (a b c : ℝ) (h : 9 * b^2 - 4 * a * c = 0) :
  ∃ r : ℝ, b = a * r ∧ c = b * r :=
sorry

end NUMINAMATH_CALUDE_zero_discriminant_implies_geometric_progression_l3271_327198


namespace NUMINAMATH_CALUDE_probability_of_black_ball_l3271_327150

theorem probability_of_black_ball
  (p_red : ℝ) (p_white : ℝ) (p_black : ℝ)
  (h1 : p_red = 0.52)
  (h2 : p_white = 0.28)
  (h3 : p_red + p_white + p_black = 1) :
  p_black = 0.2 := by
sorry

end NUMINAMATH_CALUDE_probability_of_black_ball_l3271_327150


namespace NUMINAMATH_CALUDE_keith_card_spending_l3271_327196

/-- Represents the total amount spent on trading cards -/
def total_spent (digimon_packs pokemon_packs yugioh_packs magic_packs : ℕ) 
  (digimon_price pokemon_price yugioh_price magic_price baseball_price : ℚ) : ℚ :=
  digimon_packs * digimon_price + 
  pokemon_packs * pokemon_price + 
  yugioh_packs * yugioh_price + 
  magic_packs * magic_price + 
  baseball_price

/-- Theorem stating the total amount Keith spent on cards -/
theorem keith_card_spending :
  total_spent 4 3 6 2 4.45 5.25 3.99 6.75 6.06 = 77.05 := by
  sorry

end NUMINAMATH_CALUDE_keith_card_spending_l3271_327196


namespace NUMINAMATH_CALUDE_trig_identity_proof_l3271_327103

theorem trig_identity_proof (θ : ℝ) : 
  Real.sin (θ + Real.pi / 180 * 75) + Real.cos (θ + Real.pi / 180 * 45) - Real.sqrt 3 * Real.cos (θ + Real.pi / 180 * 15) = 0 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_proof_l3271_327103


namespace NUMINAMATH_CALUDE_four_from_seven_l3271_327100

def choose (n k : ℕ) : ℕ :=
  if k > n then 0
  else Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem four_from_seven :
  choose 7 4 = 35 := by
  sorry

end NUMINAMATH_CALUDE_four_from_seven_l3271_327100


namespace NUMINAMATH_CALUDE_min_value_theorem_l3271_327137

theorem min_value_theorem (x : ℝ) (h : x > 0) :
  x^4 + 16*x + 256/x^6 ≥ 56 ∧
  (x^4 + 16*x + 256/x^6 = 56 ↔ x = 2) := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3271_327137
