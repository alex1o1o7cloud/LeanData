import Mathlib

namespace NUMINAMATH_CALUDE_max_hearts_desire_desire_fulfilled_l1455_145579

/-- Represents a four-digit natural number M = 1000a + 100b + 10c + d -/
structure FourDigitNumber where
  a : Nat
  b : Nat
  c : Nat
  d : Nat
  h1 : 1 ≤ a ∧ a ≤ 9
  h2 : 1 ≤ b ∧ b ≤ 9
  h3 : 1 ≤ c ∧ c ≤ 9
  h4 : 1 ≤ d ∧ d ≤ 9
  h5 : c > d

/-- Calculates the value of M given its digits -/
def FourDigitNumber.value (n : FourDigitNumber) : Nat :=
  1000 * n.a + 100 * n.b + 10 * n.c + n.d

/-- Checks if a number is a "heart's desire" and "desire fulfilled" number -/
def isHeartsDesireAndDesireFulfilled (n : FourDigitNumber) : Prop :=
  (10 * n.b + n.c) / (n.a + n.d) = 11

/-- Calculates F(M) -/
def F (n : FourDigitNumber) : Nat :=
  10 * (n.a + n.b) + 3 * n.c

/-- Main theorem statement -/
theorem max_hearts_desire_desire_fulfilled :
  ∃ (M : FourDigitNumber),
    isHeartsDesireAndDesireFulfilled M ∧
    F M % 7 = 0 ∧
    M.value = 5883 ∧
    (∀ (N : FourDigitNumber),
      isHeartsDesireAndDesireFulfilled N ∧
      F N % 7 = 0 →
      N.value ≤ M.value) := by
  sorry

end NUMINAMATH_CALUDE_max_hearts_desire_desire_fulfilled_l1455_145579


namespace NUMINAMATH_CALUDE_werewolf_identity_l1455_145573

-- Define the inhabitants
inductive Inhabitant : Type
| A : Inhabitant
| B : Inhabitant
| C : Inhabitant

-- Define the possible states
inductive State
| Knight : State
| Liar : State
| Werewolf : State

def is_knight (i : Inhabitant) (state : Inhabitant → State) : Prop :=
  state i = State.Knight

def is_liar (i : Inhabitant) (state : Inhabitant → State) : Prop :=
  state i = State.Liar

def is_werewolf (i : Inhabitant) (state : Inhabitant → State) : Prop :=
  state i = State.Werewolf

-- A's statement: At least one of us is a knight
def A_statement (state : Inhabitant → State) : Prop :=
  ∃ i : Inhabitant, is_knight i state

-- B's statement: At least one of us is a liar
def B_statement (state : Inhabitant → State) : Prop :=
  ∃ i : Inhabitant, is_liar i state

-- Theorem to prove
theorem werewolf_identity (state : Inhabitant → State) :
  -- At least one is a werewolf
  (∃ i : Inhabitant, is_werewolf i state) →
  -- None are both knight and werewolf
  (∀ i : Inhabitant, ¬(is_knight i state ∧ is_werewolf i state)) →
  -- A's statement is true if A is a knight, false if A is a liar
  ((is_knight Inhabitant.A state → A_statement state) ∧
   (is_liar Inhabitant.A state → ¬A_statement state)) →
  -- B's statement is true if B is a knight, false if B is a liar
  ((is_knight Inhabitant.B state → B_statement state) ∧
   (is_liar Inhabitant.B state → ¬B_statement state)) →
  -- C is the werewolf
  is_werewolf Inhabitant.C state :=
by sorry

end NUMINAMATH_CALUDE_werewolf_identity_l1455_145573


namespace NUMINAMATH_CALUDE_triangle_area_from_perimeter_and_inradius_l1455_145569

/-- The area of a triangle given its perimeter and inradius -/
theorem triangle_area_from_perimeter_and_inradius
  (perimeter : ℝ) (inradius : ℝ) (angle_smallest_sides : ℝ)
  (h_perimeter : perimeter = 36)
  (h_inradius : inradius = 2.5)
  (h_angle : angle_smallest_sides = 75) :
  inradius * (perimeter / 2) = 45 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_from_perimeter_and_inradius_l1455_145569


namespace NUMINAMATH_CALUDE_find_m_l1455_145560

theorem find_m (w x y z m : ℝ) 
  (h : 9 / (w + x + y) = m / (w + z) ∧ m / (w + z) = 15 / (z - x - y)) : 
  m = 24 := by
sorry

end NUMINAMATH_CALUDE_find_m_l1455_145560


namespace NUMINAMATH_CALUDE_divisor_problem_l1455_145519

theorem divisor_problem (n m : ℕ) (h1 : n = 3830) (h2 : m = 5) : 
  (∃ d : ℕ, d > 0 ∧ (n - m) % d = 0 ∧ 
   ∀ k < m, ¬((n - k) % d = 0)) → 
  (n - m) % 15 = 0 ∧ 15 > 0 ∧ 
  ∀ k < m, ¬((n - k) % 15 = 0) :=
sorry

end NUMINAMATH_CALUDE_divisor_problem_l1455_145519


namespace NUMINAMATH_CALUDE_largest_solution_of_equation_l1455_145524

theorem largest_solution_of_equation (x : ℝ) : 
  (6 * (12 * x^2 + 12 * x + 11) = x * (12 * x - 44)) → x ≤ -1 :=
by sorry

end NUMINAMATH_CALUDE_largest_solution_of_equation_l1455_145524


namespace NUMINAMATH_CALUDE_power_product_equals_five_l1455_145586

theorem power_product_equals_five (a : ℝ) : a^2 * a^3 = a^5 := by
  sorry

end NUMINAMATH_CALUDE_power_product_equals_five_l1455_145586


namespace NUMINAMATH_CALUDE_fish_in_tank_fish_in_tank_proof_l1455_145599

theorem fish_in_tank : ℕ → Prop :=
  fun total_fish =>
    ∃ (blue_fish : ℕ),
      blue_fish = total_fish / 3 ∧
      blue_fish / 2 = 10 ∧
      total_fish = 60

-- The proof is omitted
theorem fish_in_tank_proof : fish_in_tank 60 := by
  sorry

end NUMINAMATH_CALUDE_fish_in_tank_fish_in_tank_proof_l1455_145599


namespace NUMINAMATH_CALUDE_unique_k_for_inequality_l1455_145505

theorem unique_k_for_inequality :
  ∃! k : ℝ, ∀ t : ℝ, t ∈ Set.Ioo (-1) 1 →
    (1 + t) ^ k * (1 - t) ^ (1 - k) ≤ 1 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_k_for_inequality_l1455_145505


namespace NUMINAMATH_CALUDE_root_sum_product_l1455_145598

theorem root_sum_product (p q : ℝ) : 
  (Complex.I : ℂ).re = 0 ∧ (Complex.I : ℂ).im = 1 →
  (2 : ℂ) * (-3 + 2 * Complex.I)^2 + p * (-3 + 2 * Complex.I) + q = 0 →
  p + q = 38 := by sorry

end NUMINAMATH_CALUDE_root_sum_product_l1455_145598


namespace NUMINAMATH_CALUDE_margaret_fraction_of_dollar_l1455_145509

-- Define the amounts for each person
def lance_cents : ℕ := 70
def guy_cents : ℕ := 50 + 10  -- Two quarters and a dime
def bill_cents : ℕ := 6 * 10  -- Six dimes
def total_cents : ℕ := 265

-- Define Margaret's amount
def margaret_cents : ℕ := total_cents - (lance_cents + guy_cents + bill_cents)

-- Theorem to prove
theorem margaret_fraction_of_dollar : 
  (margaret_cents : ℚ) / 100 = 3 / 4 := by sorry

end NUMINAMATH_CALUDE_margaret_fraction_of_dollar_l1455_145509


namespace NUMINAMATH_CALUDE_number_exceeding_percentage_l1455_145543

theorem number_exceeding_percentage (x : ℝ) : x = 0.16 * x + 42 → x = 50 := by
  sorry

end NUMINAMATH_CALUDE_number_exceeding_percentage_l1455_145543


namespace NUMINAMATH_CALUDE_quadratic_equation_properties_l1455_145520

theorem quadratic_equation_properties (k : ℝ) :
  let f : ℝ → ℝ := λ x ↦ x^2 + 3*x + k - 2
  (∃ x₁ x₂ : ℝ, f x₁ = 0 ∧ f x₂ = 0 ∧ x₁ ≠ x₂) →
  (k ≤ 17/4 ∧
   (∀ x₁ x₂ : ℝ, f x₁ = 0 → f x₂ = 0 → x₁ ≠ x₂ → (x₁ - 1)*(x₂ - 1) = -1 → k = -3)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_properties_l1455_145520


namespace NUMINAMATH_CALUDE_range_of_2cos_squared_l1455_145554

theorem range_of_2cos_squared (x : ℝ) : 0 ≤ 2 * (Real.cos x)^2 ∧ 2 * (Real.cos x)^2 ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_2cos_squared_l1455_145554


namespace NUMINAMATH_CALUDE_modified_fibonacci_series_sum_l1455_145504

/-- Modified Fibonacci sequence -/
def F : ℕ → ℕ
  | 0 => 1
  | 1 => 2
  | (n + 2) => F (n + 1) + F n

/-- The sum of the series F_n / 5^n from n = 0 to infinity -/
noncomputable def seriesSum : ℝ := ∑' n, (F n : ℝ) / 5^n

theorem modified_fibonacci_series_sum : seriesSum = 35 / 18 := by
  sorry

end NUMINAMATH_CALUDE_modified_fibonacci_series_sum_l1455_145504


namespace NUMINAMATH_CALUDE_pizza_distribution_l1455_145508

/-- Calculates the number of slices each person gets in a group pizza order -/
def slices_per_person (num_people : ℕ) (num_pizzas : ℕ) (slices_per_pizza : ℕ) : ℕ :=
  (num_pizzas * slices_per_pizza) / num_people

/-- Proves that given 18 people, 6 pizzas with 9 slices each, each person gets 3 slices -/
theorem pizza_distribution :
  slices_per_person 18 6 9 = 3 := by
  sorry

#eval slices_per_person 18 6 9

end NUMINAMATH_CALUDE_pizza_distribution_l1455_145508


namespace NUMINAMATH_CALUDE_pagoda_lamps_l1455_145501

theorem pagoda_lamps (n : ℕ) (total : ℕ) (h1 : n = 7) (h2 : total = 381) : 
  (∃ a : ℕ, a * (2^n - 1) = total) → 3 * (2^n - 1) = total := by
sorry

end NUMINAMATH_CALUDE_pagoda_lamps_l1455_145501


namespace NUMINAMATH_CALUDE_product_of_integers_l1455_145522

theorem product_of_integers (w x y z : ℤ) : 
  0 < w → w < x → x < y → y < z → w + z = 5 → w * x * y * z = 36 := by
  sorry

end NUMINAMATH_CALUDE_product_of_integers_l1455_145522


namespace NUMINAMATH_CALUDE_tan_domain_shift_l1455_145529

theorem tan_domain_shift (x : ℝ) :
  (∃ k : ℤ, x = k * π / 2 + π / 12) ↔ (∃ k : ℤ, 2 * x + π / 3 = k * π + π / 2) :=
by sorry

end NUMINAMATH_CALUDE_tan_domain_shift_l1455_145529


namespace NUMINAMATH_CALUDE_toy_selling_price_l1455_145550

/-- Calculates the total selling price of toys given the number of toys sold,
    the number of toys whose cost price was gained, and the cost price per toy. -/
def totalSellingPrice (numToysSold : ℕ) (numToysGained : ℕ) (costPrice : ℕ) : ℕ :=
  numToysSold * costPrice + numToysGained * costPrice

/-- Theorem stating that for the given conditions, the total selling price is 27300. -/
theorem toy_selling_price :
  totalSellingPrice 18 3 1300 = 27300 := by
  sorry

end NUMINAMATH_CALUDE_toy_selling_price_l1455_145550


namespace NUMINAMATH_CALUDE_equation_is_quadratic_l1455_145546

theorem equation_is_quadratic : ∃ (a b c : ℝ), a ≠ 0 ∧ 
  ∀ x, 3 * (x + 1)^2 = 2 * (x - 2) ↔ a * x^2 + b * x + c = 0 :=
by sorry

end NUMINAMATH_CALUDE_equation_is_quadratic_l1455_145546


namespace NUMINAMATH_CALUDE_gensokyo_tennis_club_meeting_day_l1455_145547

/-- The Gensokyo Tennis Club problem -/
theorem gensokyo_tennis_club_meeting_day :
  let total_players : ℕ := 2016
  let total_courts : ℕ := 1008
  let reimu_start : ℕ := 123
  let marisa_start : ℕ := 876
  let winner_move (court : ℕ) : ℕ := if court > 1 then court - 1 else 1
  let loser_move (court : ℕ) : ℕ := if court < total_courts then court + 1 else total_courts
  let reimu_path (day : ℕ) : ℕ := if day < reimu_start then reimu_start - day else 1
  let marisa_path (day : ℕ) : ℕ :=
    if day ≤ (total_courts - marisa_start) then
      marisa_start + day
    else
      total_courts - (day - (total_courts - marisa_start))
  ∃ (n : ℕ), n > 0 ∧ reimu_path n = marisa_path n ∧ 
    ∀ (m : ℕ), m > 0 ∧ m < n → reimu_path m ≠ marisa_path m :=
by
  sorry

end NUMINAMATH_CALUDE_gensokyo_tennis_club_meeting_day_l1455_145547


namespace NUMINAMATH_CALUDE_composite_5n_plus_3_l1455_145502

theorem composite_5n_plus_3 (n : ℕ) (h1 : ∃ x : ℕ, 2 * n + 1 = x^2) (h2 : ∃ y : ℕ, 3 * n + 1 = y^2) : 
  ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ 5 * n + 3 = a * b :=
by sorry

end NUMINAMATH_CALUDE_composite_5n_plus_3_l1455_145502


namespace NUMINAMATH_CALUDE_vectors_in_same_plane_l1455_145555

variable (V : Type*) [AddCommGroup V] [Module ℝ V]
variable (a b c : V)

def is_basis (a b c : V) : Prop :=
  LinearIndependent ℝ ![a, b, c] ∧ Submodule.span ℝ {a, b, c} = ⊤

def coplanar (u v w : V) : Prop :=
  ∃ (x y z : ℝ), x • u + y • v + z • w = 0 ∧ (x ≠ 0 ∨ y ≠ 0 ∨ z ≠ 0)

theorem vectors_in_same_plane (h : is_basis V a b c) :
  coplanar V (2 • a + b) (a + b + c) (7 • a + 5 • b + 3 • c) :=
sorry

end NUMINAMATH_CALUDE_vectors_in_same_plane_l1455_145555


namespace NUMINAMATH_CALUDE_min_value_fraction_l1455_145584

theorem min_value_fraction (x : ℝ) (h : x > -1) : 
  x^2 / (x + 1) ≥ 0 ∧ ∃ y > -1, y^2 / (y + 1) = 0 := by sorry

end NUMINAMATH_CALUDE_min_value_fraction_l1455_145584


namespace NUMINAMATH_CALUDE_projection_matrix_values_l1455_145596

def is_projection_matrix (P : Matrix (Fin 2) (Fin 2) ℚ) : Prop :=
  P * P = P

theorem projection_matrix_values :
  ∀ (a c : ℚ),
  let P : Matrix (Fin 2) (Fin 2) ℚ := !![a, 18/45; c, 27/45]
  is_projection_matrix P →
  a = 1/5 ∧ c = 2/5 := by
sorry

end NUMINAMATH_CALUDE_projection_matrix_values_l1455_145596


namespace NUMINAMATH_CALUDE_exactly_one_even_negation_l1455_145548

/-- Represents the property of a natural number being even -/
def IsEven (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k

/-- Represents the property of a natural number being odd -/
def IsOdd (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k + 1

/-- States that exactly one of three natural numbers is even -/
def ExactlyOneEven (a b c : ℕ) : Prop :=
  (IsEven a ∧ IsOdd b ∧ IsOdd c) ∨
  (IsOdd a ∧ IsEven b ∧ IsOdd c) ∨
  (IsOdd a ∧ IsOdd b ∧ IsEven c)

/-- States that at least two of three natural numbers are even or all are odd -/
def AtLeastTwoEvenOrAllOdd (a b c : ℕ) : Prop :=
  (IsEven a ∧ IsEven b) ∨
  (IsEven a ∧ IsEven c) ∨
  (IsEven b ∧ IsEven c) ∨
  (IsOdd a ∧ IsOdd b ∧ IsOdd c)

theorem exactly_one_even_negation (a b c : ℕ) :
  ¬(ExactlyOneEven a b c) ↔ AtLeastTwoEvenOrAllOdd a b c :=
sorry

end NUMINAMATH_CALUDE_exactly_one_even_negation_l1455_145548


namespace NUMINAMATH_CALUDE_last_three_digits_l1455_145565

/-- A function that generates the list of positive integers with first digit 2 in increasing order -/
def digit2List : ℕ → ℕ 
| 0 => 2
| (n + 1) => 
  let prev := digit2List n
  if prev < 10 then 20
  else if prev % 10 = 9 then prev + 11
  else prev + 1

/-- The 998th digit in the digit2List -/
def digit998 : ℕ := sorry

/-- The 999th digit in the digit2List -/
def digit999 : ℕ := sorry

/-- The 1000th digit in the digit2List -/
def digit1000 : ℕ := sorry

/-- Theorem stating that the 998th, 999th, and 1000th digits form the number 216 -/
theorem last_three_digits : 
  digit998 * 100 + digit999 * 10 + digit1000 = 216 := by sorry

end NUMINAMATH_CALUDE_last_three_digits_l1455_145565


namespace NUMINAMATH_CALUDE_sequence_range_l1455_145593

theorem sequence_range (a : ℝ) : 
  (∀ n : ℕ+, (fun n => if n < 6 then (1/2 - a) * n + 1 else a^(n - 5)) n > 
             (fun n => if n < 6 then (1/2 - a) * n + 1 else a^(n - 5)) (n + 1)) → 
  (1/2 < a ∧ a < 7/12) := by
  sorry

end NUMINAMATH_CALUDE_sequence_range_l1455_145593


namespace NUMINAMATH_CALUDE_lcm_of_15_25_35_l1455_145557

theorem lcm_of_15_25_35 : Nat.lcm (Nat.lcm 15 25) 35 = 525 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_15_25_35_l1455_145557


namespace NUMINAMATH_CALUDE_krishans_money_krishan_has_4046_l1455_145590

/-- Given the ratios of money between Ram, Gopal, and Krishan, and Ram's amount, 
    calculate Krishan's amount. -/
theorem krishans_money 
  (ram_gopal_ratio : ℚ) 
  (gopal_krishan_ratio : ℚ) 
  (ram_money : ℕ) : ℕ :=
  let gopal_money := (ram_money * 17) / 7
  let krishan_money := (gopal_money * 17) / 7
  krishan_money

/-- Prove that Krishan has Rs. 4046 given the problem conditions. -/
theorem krishan_has_4046 :
  krishans_money (7/17) (7/17) 686 = 4046 := by
  sorry

end NUMINAMATH_CALUDE_krishans_money_krishan_has_4046_l1455_145590


namespace NUMINAMATH_CALUDE_quadratic_inequality_always_negative_l1455_145581

theorem quadratic_inequality_always_negative : ∀ x : ℝ, -6 * x^2 + 2 * x - 8 < 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_always_negative_l1455_145581


namespace NUMINAMATH_CALUDE_function_zeros_and_monotonicity_l1455_145514

theorem function_zeros_and_monotonicity (a : ℝ) : 
  a ≠ 0 →
  (∃! x : ℝ, x ∈ (Set.Ioo 0 1) ∧ 2 * a * x^2 - x - 1 = 0) →
  ¬(∀ x y : ℝ, x > 0 → y > 0 → x < y → x^(2-a) > y^(2-a)) →
  1 < a ∧ a ≤ 2 := by
sorry

end NUMINAMATH_CALUDE_function_zeros_and_monotonicity_l1455_145514


namespace NUMINAMATH_CALUDE_subset_cardinality_inequality_l1455_145597

theorem subset_cardinality_inequality (n m : ℕ) (A : Fin m → Finset (Fin n)) :
  (∀ i : Fin m, ¬ (30 ∣ (A i).card)) →
  (∀ i j : Fin m, i ≠ j → (30 ∣ (A i ∩ A j).card)) →
  2 * m - m / 30 ≤ 3 * n :=
by sorry

end NUMINAMATH_CALUDE_subset_cardinality_inequality_l1455_145597


namespace NUMINAMATH_CALUDE_solve_linear_equation_l1455_145588

theorem solve_linear_equation (x : ℝ) : (3 * x - 8 = -2 * x + 17) → x = 5 := by
  sorry

end NUMINAMATH_CALUDE_solve_linear_equation_l1455_145588


namespace NUMINAMATH_CALUDE_solution_set_equiv_solution_values_l1455_145559

-- Part I
def solution_set (x : ℝ) : Prop := |x + 3| < 2*x + 1

theorem solution_set_equiv : ∀ x : ℝ, solution_set x ↔ x > 2 := by sorry

-- Part II
def has_solution (t : ℝ) : Prop := 
  t ≠ 0 ∧ ∃ x : ℝ, |x - t| + |x + 1/t| = 2

theorem solution_values : ∀ t : ℝ, has_solution t → t = 1 ∨ t = -1 := by sorry

end NUMINAMATH_CALUDE_solution_set_equiv_solution_values_l1455_145559


namespace NUMINAMATH_CALUDE_distance_after_one_hour_l1455_145515

/-- The distance between two people moving in opposite directions for 1 hour -/
def distance_between (speed1 speed2 : ℝ) : ℝ :=
  speed1 + speed2

theorem distance_after_one_hour :
  let riya_speed : ℝ := 21
  let priya_speed : ℝ := 22
  distance_between riya_speed priya_speed = 43 := by
  sorry

end NUMINAMATH_CALUDE_distance_after_one_hour_l1455_145515


namespace NUMINAMATH_CALUDE_green_face_prob_five_eighths_l1455_145544

/-- A regular octahedron with colored faces -/
structure ColoredOctahedron where
  green_faces : ℕ
  purple_faces : ℕ
  total_faces : ℕ
  face_sum : green_faces + purple_faces = total_faces
  is_octahedron : total_faces = 8

/-- The probability of rolling a green face on a colored octahedron -/
def green_face_probability (o : ColoredOctahedron) : ℚ :=
  o.green_faces / o.total_faces

/-- Theorem: The probability of rolling a green face on a regular octahedron 
    with 5 green faces and 3 purple faces is 5/8 -/
theorem green_face_prob_five_eighths :
  ∀ (o : ColoredOctahedron), 
    o.green_faces = 5 → 
    o.purple_faces = 3 → 
    green_face_probability o = 5/8 :=
by
  sorry

end NUMINAMATH_CALUDE_green_face_prob_five_eighths_l1455_145544


namespace NUMINAMATH_CALUDE_vertical_angles_equal_parallel_lines_corresponding_angles_equal_l1455_145572

-- Define the concept of an angle
def Angle : Type := ℝ

-- Define the concept of a line
def Line : Type := Unit

-- Define vertical angles
def are_vertical (a b : Angle) : Prop := sorry

-- Define parallel lines
def are_parallel (l1 l2 : Line) : Prop := sorry

-- Define corresponding angles
def are_corresponding (a b : Angle) (l1 l2 : Line) : Prop := sorry

-- Theorem: Vertical angles are equal
theorem vertical_angles_equal (a b : Angle) : 
  are_vertical a b → a = b := by sorry

-- Theorem: If two lines are parallel, then corresponding angles are equal
theorem parallel_lines_corresponding_angles_equal (a b : Angle) (l1 l2 : Line) :
  are_parallel l1 l2 → are_corresponding a b l1 l2 → a = b := by sorry

end NUMINAMATH_CALUDE_vertical_angles_equal_parallel_lines_corresponding_angles_equal_l1455_145572


namespace NUMINAMATH_CALUDE_sum_of_cubes_equals_square_l1455_145558

-- Define the pattern function
def pattern (n : ℕ) : ℕ := n * (n + 1) / 2

-- State the theorem
theorem sum_of_cubes_equals_square :
  (1^3 + 2^3 = pattern 2^2) →
  (1^3 + 2^3 + 3^3 = pattern 3^2) →
  (1^3 + 2^3 + 3^3 + 4^3 = pattern 4^2) →
  (1^3 + 2^3 + 3^3 + 4^3 + 5^3 + 6^3 = pattern 6^2) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_equals_square_l1455_145558


namespace NUMINAMATH_CALUDE_point_inside_circle_l1455_145552

/-- A line in 2D space defined by the equation ax + by = 1 -/
structure Line where
  a : ℝ
  b : ℝ

/-- A circle in 2D space defined by the equation x² + y² = 1 -/
def Circle := {p : ℝ × ℝ | p.1^2 + p.2^2 = 1}

/-- Predicate to check if a point is inside a circle -/
def is_inside_circle (p : ℝ × ℝ) : Prop := p.1^2 + p.2^2 < 1

/-- Predicate to check if a line and a circle have no intersection points -/
def no_intersection (l : Line) (c : Set (ℝ × ℝ)) : Prop :=
  ∀ p : ℝ × ℝ, (l.a * p.1 + l.b * p.2 = 1) → p ∉ c

theorem point_inside_circle (l : Line) (h : no_intersection l Circle) :
  is_inside_circle (l.b, l.a) := by
  sorry

end NUMINAMATH_CALUDE_point_inside_circle_l1455_145552


namespace NUMINAMATH_CALUDE_number_puzzle_l1455_145595

theorem number_puzzle : ∃ x : ℝ, 3 * (2 * x + 9) = 63 := by
  sorry

end NUMINAMATH_CALUDE_number_puzzle_l1455_145595


namespace NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_17_l1455_145562

theorem greatest_three_digit_multiple_of_17 : ∃ n : ℕ, 
  n = 986 ∧ 
  17 ∣ n ∧
  100 ≤ n ∧ 
  n ≤ 999 ∧
  ∀ m : ℕ, (17 ∣ m ∧ 100 ≤ m ∧ m ≤ 999) → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_17_l1455_145562


namespace NUMINAMATH_CALUDE_circle_center_correct_l1455_145576

/-- The equation of a circle in the form ax² + bx + cy² + dy + e = 0 -/
structure CircleEquation where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  e : ℝ

/-- The center of a circle -/
structure CircleCenter where
  x : ℝ
  y : ℝ

/-- Given a circle equation, return its center -/
def findCenter (eq : CircleEquation) : CircleCenter :=
  sorry

theorem circle_center_correct :
  let eq := CircleEquation.mk 1 (-6) 1 2 (-75)
  findCenter eq = CircleCenter.mk 3 (-1) := by sorry

end NUMINAMATH_CALUDE_circle_center_correct_l1455_145576


namespace NUMINAMATH_CALUDE_smallest_number_600_times_prime_divisors_l1455_145545

theorem smallest_number_600_times_prime_divisors :
  ∃ (N : ℕ), N > 1 ∧
  (∀ p : ℕ, Nat.Prime p → p ∣ N → N ≥ 600 * p) ∧
  (∀ M : ℕ, M > 1 → (∀ q : ℕ, Nat.Prime q → q ∣ M → M ≥ 600 * q) → M ≥ N) ∧
  N = 1944 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_600_times_prime_divisors_l1455_145545


namespace NUMINAMATH_CALUDE_x_squared_minus_five_is_quadratic_l1455_145507

/-- Definition of a quadratic equation -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The equation x² - 5 = 0 -/
def f (x : ℝ) : ℝ := x^2 - 5

/-- Theorem: x² - 5 = 0 is a quadratic equation -/
theorem x_squared_minus_five_is_quadratic : is_quadratic_equation f := by
  sorry

end NUMINAMATH_CALUDE_x_squared_minus_five_is_quadratic_l1455_145507


namespace NUMINAMATH_CALUDE_a_not_in_A_l1455_145561

def A : Set ℝ := {x | x ≤ 4}

theorem a_not_in_A : 3 * Real.sqrt 3 ∉ A := by sorry

end NUMINAMATH_CALUDE_a_not_in_A_l1455_145561


namespace NUMINAMATH_CALUDE_vector_expression_simplification_l1455_145534

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

theorem vector_expression_simplification (a b : V) :
  2 • (a - b) - 3 • (a + b) = -a - 5 • b :=
by sorry

end NUMINAMATH_CALUDE_vector_expression_simplification_l1455_145534


namespace NUMINAMATH_CALUDE_negation_of_existence_l1455_145574

theorem negation_of_existence (m : ℝ) :
  (¬ ∃ x₀ : ℝ, x₀ > 0 ∧ x₀^2 + m*x₀ - 2 > 0) ↔
  (∀ x : ℝ, x > 0 → x^2 + m*x - 2 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_l1455_145574


namespace NUMINAMATH_CALUDE_salmon_migration_l1455_145533

theorem salmon_migration (male_salmon female_salmon : ℕ) 
  (h1 : male_salmon = 712261) 
  (h2 : female_salmon = 259378) : 
  male_salmon + female_salmon = 971639 := by
  sorry

end NUMINAMATH_CALUDE_salmon_migration_l1455_145533


namespace NUMINAMATH_CALUDE_polynomial_roots_l1455_145513

theorem polynomial_roots : ∃ (x₁ x₂ x₃ : ℝ), 
  (x₁ = 1 ∧ x₂ = 2 ∧ x₃ = -1) ∧ 
  (∀ x : ℝ, x^4 - 4*x^3 + 3*x^2 + 4*x - 4 = 0 ↔ (x = x₁ ∨ x = x₂ ∨ x = x₃)) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_roots_l1455_145513


namespace NUMINAMATH_CALUDE_rectangle_side_length_l1455_145535

/-- Given a rectangle with its bottom side on the x-axis from (-a, 0) to (a, 0),
    its top side on the parabola y = x^2, and its area equal to 81,
    prove that the length of its side parallel to the x-axis is 2∛(40.5). -/
theorem rectangle_side_length (a : ℝ) : 
  (2 * a * a^2 = 81) →  -- Area of the rectangle
  (2 * a = 2 * (40.5 : ℝ)^(1/3)) := by
  sorry

end NUMINAMATH_CALUDE_rectangle_side_length_l1455_145535


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l1455_145512

def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_function_properties
  (a b c : ℝ)
  (h_odd : ∀ x, f a b c (-x) = -f a b c x)
  (h_f1 : f a b c 1 = b + c)
  (h_f2 : f a b c 2 = 4 * a + 2 * b + c) :
  (a = 2 ∧ b = -3 ∧ c = 0) ∧
  (∀ x, x > 0 → ∀ y, y > x → f a b c y < f a b c x) ∧
  (∃ m, m = 2 ∧ ∀ x, x > 0 → f a b c x ≥ m) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l1455_145512


namespace NUMINAMATH_CALUDE_sugar_recipe_reduction_l1455_145551

theorem sugar_recipe_reduction : 
  ∀ (original_sugar : ℚ) (reduced_sugar : ℚ),
  original_sugar = 17/3 →
  reduced_sugar = (1/3) * original_sugar →
  reduced_sugar = 17/9 :=
by
  sorry

#check sugar_recipe_reduction

end NUMINAMATH_CALUDE_sugar_recipe_reduction_l1455_145551


namespace NUMINAMATH_CALUDE_yellow_square_area_l1455_145518

-- Define the cube's edge length
def cube_edge : ℝ := 15

-- Define the total amount of purple paint
def total_purple_paint : ℝ := 900

-- Define the number of faces on a cube
def num_faces : ℕ := 6

-- Theorem statement
theorem yellow_square_area :
  let total_surface_area := num_faces * (cube_edge ^ 2)
  let purple_area_per_face := total_purple_paint / num_faces
  let yellow_area_per_face := cube_edge ^ 2 - purple_area_per_face
  yellow_area_per_face = 75 := by sorry

end NUMINAMATH_CALUDE_yellow_square_area_l1455_145518


namespace NUMINAMATH_CALUDE_parallel_planes_condition_l1455_145583

-- Define the basic types
variable (Point : Type) (Line : Type) (Plane : Type)

-- Define the relations
variable (on_plane : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)
variable (intersect : Line → Line → Prop)
variable (planes_parallel : Plane → Plane → Prop)

-- State the theorem
theorem parallel_planes_condition 
  (α β : Plane) (m n l₁ l₂ : Line)
  (h1 : on_plane m α)
  (h2 : on_plane n α)
  (h3 : m ≠ n)
  (h4 : on_plane l₁ β)
  (h5 : on_plane l₂ β)
  (h6 : intersect l₁ l₂)
  (h7 : parallel m l₁)
  (h8 : parallel n l₂) :
  planes_parallel α β :=
sorry

end NUMINAMATH_CALUDE_parallel_planes_condition_l1455_145583


namespace NUMINAMATH_CALUDE_quadratic_equation_q_value_l1455_145539

theorem quadratic_equation_q_value 
  (p q : ℝ) 
  (h : ∃ x : ℂ, 3 * x^2 + p * x + q = 0 ∧ x = 4 + 3*I) : 
  q = 75 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_q_value_l1455_145539


namespace NUMINAMATH_CALUDE_virus_infected_computers_office_virus_scenario_l1455_145577

/-- Represents the state of computers in an office before and after a virus infection. -/
structure ComputerNetwork where
  total : ℕ             -- Total number of computers
  infected : ℕ          -- Number of infected computers
  initialConnections : ℕ -- Number of initial connections per computer
  finalConnections : ℕ  -- Number of final connections per uninfected computer
  disconnectedCables : ℕ -- Number of cables disconnected due to virus

/-- The theorem stating the number of infected computers given the network conditions -/
theorem virus_infected_computers (network : ComputerNetwork) : 
  network.initialConnections = 5 ∧ 
  network.finalConnections = 3 ∧ 
  network.disconnectedCables = 26 →
  network.infected = 8 := by
  sorry

/-- Main theorem proving the number of infected computers in the given scenario -/
theorem office_virus_scenario : ∃ (network : ComputerNetwork), 
  network.initialConnections = 5 ∧
  network.finalConnections = 3 ∧
  network.disconnectedCables = 26 ∧
  network.infected = 8 := by
  sorry

end NUMINAMATH_CALUDE_virus_infected_computers_office_virus_scenario_l1455_145577


namespace NUMINAMATH_CALUDE_total_blue_balloons_l1455_145564

/-- The number of blue balloons Joan and Melanie have in total is 81, 
    given that Joan has 40 and Melanie has 41. -/
theorem total_blue_balloons (joan_balloons melanie_balloons : ℕ) 
  (h1 : joan_balloons = 40) 
  (h2 : melanie_balloons = 41) : 
  joan_balloons + melanie_balloons = 81 := by
  sorry

end NUMINAMATH_CALUDE_total_blue_balloons_l1455_145564


namespace NUMINAMATH_CALUDE_tangerine_persimmon_ratio_l1455_145589

theorem tangerine_persimmon_ratio :
  let apples : ℕ := 24
  let tangerines : ℕ := 6 * apples
  let persimmons : ℕ := 8
  tangerines = 18 * persimmons :=
by
  sorry

end NUMINAMATH_CALUDE_tangerine_persimmon_ratio_l1455_145589


namespace NUMINAMATH_CALUDE_basketball_tryouts_l1455_145525

theorem basketball_tryouts (girls boys called_back : ℕ) 
  (h1 : girls = 39)
  (h2 : boys = 4)
  (h3 : called_back = 26) :
  girls + boys - called_back = 17 := by
sorry

end NUMINAMATH_CALUDE_basketball_tryouts_l1455_145525


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l1455_145592

theorem regular_polygon_sides (n : ℕ) (exterior_angle : ℝ) :
  n > 2 →
  exterior_angle = 40 * Real.pi / 180 →
  exterior_angle = (2 * Real.pi) / n →
  n = 9 := by
sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l1455_145592


namespace NUMINAMATH_CALUDE_like_terms_implies_value_l1455_145511

-- Define the condition for like terms
def are_like_terms (m n : ℕ) : Prop := m = 3 ∧ n = 2

-- State the theorem
theorem like_terms_implies_value (m n : ℕ) :
  are_like_terms m n → (-n : ℤ)^m = -8 := by
  sorry

end NUMINAMATH_CALUDE_like_terms_implies_value_l1455_145511


namespace NUMINAMATH_CALUDE_pizza_toppings_l1455_145567

theorem pizza_toppings (total_slices : ℕ) (pepperoni_slices : ℕ) (mushroom_slices : ℕ) 
  (h1 : total_slices = 24)
  (h2 : pepperoni_slices = 15)
  (h3 : mushroom_slices = 20)
  (h4 : ∀ slice, slice ≤ total_slices → (slice ≤ pepperoni_slices ∨ slice ≤ mushroom_slices)) :
  ∃ both_toppings : ℕ, 
    both_toppings = pepperoni_slices + mushroom_slices - total_slices ∧
    both_toppings = 11 := by
  sorry

end NUMINAMATH_CALUDE_pizza_toppings_l1455_145567


namespace NUMINAMATH_CALUDE_max_z_value_l1455_145575

theorem max_z_value : 
  (∃ (z : ℝ), ∀ (w : ℝ), 
    (∃ (x y : ℝ), 4*x^2 + 4*y^2 + z^2 + x*y + y*z + x*z = 8) → 
    (∃ (x y : ℝ), 4*x^2 + 4*y^2 + w^2 + x*y + y*w + x*w = 8) → 
    w ≤ z) ∧ 
  (∃ (x y : ℝ), 4*x^2 + 4*y^2 + 3^2 + x*y + y*3 + x*3 = 8) :=
by sorry

end NUMINAMATH_CALUDE_max_z_value_l1455_145575


namespace NUMINAMATH_CALUDE_speaker_arrangement_count_l1455_145530

-- Define the number of speakers
def n : ℕ := 6

-- Theorem statement
theorem speaker_arrangement_count :
  (n.factorial / 2 : ℕ) = (n.factorial / 2 : ℕ) := by
  sorry

end NUMINAMATH_CALUDE_speaker_arrangement_count_l1455_145530


namespace NUMINAMATH_CALUDE_sqrt_8_same_type_as_sqrt_2_l1455_145532

-- Define what it means for two square roots to be of the same type
def same_type (a b : ℝ) : Prop :=
  ∃ (q : ℚ), a = q * b

-- State the theorem
theorem sqrt_8_same_type_as_sqrt_2 :
  same_type (Real.sqrt 8) (Real.sqrt 2) :=
by
  sorry

end NUMINAMATH_CALUDE_sqrt_8_same_type_as_sqrt_2_l1455_145532


namespace NUMINAMATH_CALUDE_intersection_distance_product_l1455_145537

/-- Given a line L and a circle C, prove that the product of distances from a point on the line to the intersection points of the line and circle is 1/4. -/
theorem intersection_distance_product (P : ℝ × ℝ) (α : ℝ) (C : Set (ℝ × ℝ)) : 
  P = (1/2, 1) →
  α = π/6 →
  C = {(x, y) | x^2 + y^2 = x + y} →
  ∃ (A B : ℝ × ℝ), A ∈ C ∧ B ∈ C ∧ 
    (∃ (t₁ t₂ : ℝ), 
      A = (1/2 + (Real.sqrt 3)/2 * t₁, 1 + 1/2 * t₁) ∧
      B = (1/2 + (Real.sqrt 3)/2 * t₂, 1 + 1/2 * t₂)) ∧
    Real.sqrt ((A.1 - P.1)^2 + (A.2 - P.2)^2) * 
    Real.sqrt ((B.1 - P.1)^2 + (B.2 - P.2)^2) = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_intersection_distance_product_l1455_145537


namespace NUMINAMATH_CALUDE_triangle_inequality_theorem_l1455_145549

/-- Given a triangle ABC with side lengths a, b, c, and a point M inside it,
    Ra, Rb, Rc are distances from M to sides BC, CA, AB respectively,
    da, db, dc are perpendicular distances from vertices A, B, C to the line through M parallel to the opposite sides. -/
def triangle_inequality (a b c Ra Rb Rc da db dc : ℝ) : Prop :=
  a * Ra + b * Rb + c * Rc ≥ 2 * (a * da + b * db + c * dc)

/-- M is the orthocenter of triangle ABC -/
def is_orthocenter (M : Point) (A B C : Point) : Prop := sorry

theorem triangle_inequality_theorem 
  (A B C M : Point) (a b c Ra Rb Rc da db dc : ℝ) :
  triangle_inequality a b c Ra Rb Rc da db dc ∧ 
  (triangle_inequality a b c Ra Rb Rc da db dc = (a * Ra + b * Rb + c * Rc = 2 * (a * da + b * db + c * dc)) ↔ 
   is_orthocenter M A B C) := by sorry

end NUMINAMATH_CALUDE_triangle_inequality_theorem_l1455_145549


namespace NUMINAMATH_CALUDE_bookmark_difference_l1455_145503

/-- The price of a bookmark in cents -/
def bookmark_price : ℕ := sorry

/-- The number of fifth graders who bought bookmarks -/
def fifth_graders : ℕ := sorry

/-- The number of fourth graders who bought bookmarks -/
def fourth_graders : ℕ := 20

theorem bookmark_difference : 
  bookmark_price > 0 ∧ 
  bookmark_price * fifth_graders = 225 ∧ 
  bookmark_price * fourth_graders = 260 →
  fourth_graders - fifth_graders = 7 := by sorry

end NUMINAMATH_CALUDE_bookmark_difference_l1455_145503


namespace NUMINAMATH_CALUDE_real_part_of_i_times_one_plus_i_l1455_145582

theorem real_part_of_i_times_one_plus_i (i : ℂ) :
  i * i = -1 →
  Complex.re (i * (1 + i)) = -1 :=
by sorry

end NUMINAMATH_CALUDE_real_part_of_i_times_one_plus_i_l1455_145582


namespace NUMINAMATH_CALUDE_system_solution_l1455_145568

theorem system_solution :
  ∃ (k m : ℚ),
    (3 * k - 4) / (k + 7) = 2/5 ∧
    2 * m + 5 * k = 14 ∧
    k = 34/13 ∧
    m = 6/13 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l1455_145568


namespace NUMINAMATH_CALUDE_parallel_lines_in_parallel_planes_parallel_line_to_intersecting_planes_l1455_145594

-- Define the basic types
variable (Point : Type) (Line : Type) (Plane : Type)

-- Define the relationships between geometric objects
variable (perpendicular : Line → Line → Prop)
variable (perpendicular_plane : Plane → Plane → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)
variable (parallel_plane : Plane → Plane → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (intersect : Plane → Plane → Line → Prop)
variable (in_plane : Line → Plane → Prop)

-- Theorem for proposition 2
theorem parallel_lines_in_parallel_planes
  (α β γ : Plane) (m n : Line) :
  parallel_plane α β →
  intersect α γ m →
  intersect β γ n →
  parallel m n :=
sorry

-- Theorem for proposition 4
theorem parallel_line_to_intersecting_planes
  (α β : Plane) (m n : Line) :
  intersect α β m →
  parallel m n →
  ¬in_plane n α →
  ¬in_plane n β →
  parallel_line_plane n α ∧ parallel_line_plane n β :=
sorry

end NUMINAMATH_CALUDE_parallel_lines_in_parallel_planes_parallel_line_to_intersecting_planes_l1455_145594


namespace NUMINAMATH_CALUDE_at_least_one_positive_l1455_145517

theorem at_least_one_positive (x y z : ℝ) : 
  (x^2 - 2*y + Real.pi/2 > 0) ∨ 
  (y^2 - 2*z + Real.pi/3 > 0) ∨ 
  (z^2 - 2*x + Real.pi/6 > 0) := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_positive_l1455_145517


namespace NUMINAMATH_CALUDE_fourth_divisor_l1455_145506

theorem fourth_divisor (n : Nat) (h1 : n = 9600) (h2 : n % 15 = 0) (h3 : n % 25 = 0) (h4 : n % 40 = 0) :
  ∃ m : Nat, m = 16 ∧ n % m = 0 ∧ ∀ k : Nat, k > m → n % k = 0 → (k % 15 = 0 ∨ k % 25 = 0 ∨ k % 40 = 0) :=
by sorry

end NUMINAMATH_CALUDE_fourth_divisor_l1455_145506


namespace NUMINAMATH_CALUDE_special_function_value_l1455_145510

/-- A function satisfying the given conditions -/
def special_function (f : ℝ → ℝ) : Prop :=
  (∀ x y : ℝ, f (x + y) = f x + f y + 6 * x * y) ∧
  (f (-1) * f 1 ≥ 9)

/-- Theorem stating that for any function satisfying the special conditions,
    f(2/3) = 4/3 -/
theorem special_function_value (f : ℝ → ℝ) (h : special_function f) :
  f (2/3) = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_special_function_value_l1455_145510


namespace NUMINAMATH_CALUDE_golden_ratio_equation_l1455_145566

theorem golden_ratio_equation : 
  let x : ℝ := (Real.sqrt 5 + 1) / 2
  let y : ℝ := (Real.sqrt 5 - 1) / 2
  x^3 * y + 2 * x^2 * y^2 + x * y^3 = 5 := by sorry

end NUMINAMATH_CALUDE_golden_ratio_equation_l1455_145566


namespace NUMINAMATH_CALUDE_point_on_x_axis_l1455_145580

/-- A point in the 2D coordinate plane -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- The x-axis in the 2D coordinate plane -/
def xAxis : Set Point2D := {p : Point2D | p.y = 0}

/-- Theorem: A point P(x,0) lies on the x-axis -/
theorem point_on_x_axis (x : ℝ) : 
  Point2D.mk x 0 ∈ xAxis := by sorry

end NUMINAMATH_CALUDE_point_on_x_axis_l1455_145580


namespace NUMINAMATH_CALUDE_smallest_candy_count_l1455_145527

theorem smallest_candy_count : ∃ n : ℕ, 
  (100 ≤ n ∧ n ≤ 999) ∧ 
  (n + 7) % 9 = 0 ∧ 
  (n - 10) % 6 = 0 ∧
  (∀ m : ℕ, (100 ≤ m ∧ m < n ∧ (m + 7) % 9 = 0 ∧ (m - 10) % 6 = 0) → False) ∧
  n = 146 := by
sorry

end NUMINAMATH_CALUDE_smallest_candy_count_l1455_145527


namespace NUMINAMATH_CALUDE_sequence_term_proof_l1455_145536

def sequence_sum (n : ℕ) : ℕ := 2^n

def sequence_term (n : ℕ) : ℕ :=
  if n = 1 then 2 else 2^(n-1)

theorem sequence_term_proof :
  ∀ n : ℕ, n ≥ 1 →
    sequence_term n = (if n = 1 then sequence_sum 1 else sequence_sum n - sequence_sum (n-1)) :=
by sorry

end NUMINAMATH_CALUDE_sequence_term_proof_l1455_145536


namespace NUMINAMATH_CALUDE_fourth_separation_at_136pm_l1455_145570

-- Define the distance between cities
def distance_between_cities : ℝ := 300

-- Define the start time
def start_time : ℝ := 6

-- Define the time of first 50 km separation
def first_separation_time : ℝ := 8

-- Define the distance of separation
def separation_distance : ℝ := 50

-- Define the function to calculate the fourth separation time
def fourth_separation_time : ℝ := start_time + 7.6

-- Theorem statement
theorem fourth_separation_at_136pm 
  (h1 : distance_between_cities = 300)
  (h2 : start_time = 6)
  (h3 : first_separation_time = 8)
  (h4 : separation_distance = 50) :
  fourth_separation_time = 13.6 := by sorry

end NUMINAMATH_CALUDE_fourth_separation_at_136pm_l1455_145570


namespace NUMINAMATH_CALUDE_line_and_points_l1455_145526

-- Define the line equation
def line_equation (x y : ℝ) : Prop := y = -2

-- Define the points
def point_A : ℝ × ℝ := (2, -2)
def point_B : ℝ × ℝ := (3, 2)

-- Theorem statement
theorem line_and_points :
  (∀ x y : ℝ, line_equation x y → y = -2) ∧  -- Line equation is y = -2
  (∀ x : ℝ, line_equation x (-2))            -- Line is parallel to x-axis
  ∧ line_equation point_A.1 point_A.2        -- Point A lies on the line
  ∧ ¬line_equation point_B.1 point_B.2 :=    -- Point B does not lie on the line
by sorry

end NUMINAMATH_CALUDE_line_and_points_l1455_145526


namespace NUMINAMATH_CALUDE_prob_full_house_is_one_third_l1455_145587

/-- Represents the outcome of rolling five six-sided dice -/
structure DiceRoll where
  pairs : Fin 6 × Fin 6
  odd : Fin 6

/-- The probability of getting a full house after rerolling the odd die -/
def prob_full_house_after_reroll (roll : DiceRoll) : ℚ :=
  2 / 6

/-- Theorem stating the probability of getting a full house after rerolling the odd die -/
theorem prob_full_house_is_one_third (roll : DiceRoll) :
  prob_full_house_after_reroll roll = 1 / 3 := by
  sorry

#check prob_full_house_is_one_third

end NUMINAMATH_CALUDE_prob_full_house_is_one_third_l1455_145587


namespace NUMINAMATH_CALUDE_division_equations_l1455_145528

theorem division_equations (h : 40 * 60 = 2400) : 
  (2400 / 40 = 60) ∧ (2400 / 60 = 40) := by
  sorry

end NUMINAMATH_CALUDE_division_equations_l1455_145528


namespace NUMINAMATH_CALUDE_pyramid_volume_l1455_145578

/-- The volume of a pyramid with a rectangular base and equal edge lengths from apex to base corners -/
theorem pyramid_volume (base_length : ℝ) (base_width : ℝ) (edge_length : ℝ) :
  base_length = 5 →
  base_width = 7 →
  edge_length = 15 →
  let base_area := base_length * base_width
  let base_diagonal := Real.sqrt (base_length^2 + base_width^2)
  let height := Real.sqrt (edge_length^2 - (base_diagonal / 2)^2)
  (1 / 3 : ℝ) * base_area * height = (35 * Real.sqrt 188) / 3 :=
by sorry

end NUMINAMATH_CALUDE_pyramid_volume_l1455_145578


namespace NUMINAMATH_CALUDE_no_multiple_of_five_l1455_145571

theorem no_multiple_of_five : ∀ C : ℕ, C < 10 → ¬(∃ k : ℕ, 200 + 10 * C + 4 = 5 * k) := by
  sorry

end NUMINAMATH_CALUDE_no_multiple_of_five_l1455_145571


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l1455_145556

/-- Given a geometric sequence {a_n} with common ratio q = 2 and sum of first n terms S_n,
    prove that S_4 / a_2 = 15/2 -/
theorem geometric_sequence_ratio (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n, a (n + 1) = 2 * a n) →  -- Common ratio q = 2
  (∀ n, S n = (a 1) * (1 - 2^n) / (1 - 2)) →  -- Sum formula for geometric sequence
  S 4 / (a 2) = 15 / 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l1455_145556


namespace NUMINAMATH_CALUDE_abs_neg_2022_l1455_145521

theorem abs_neg_2022 : |(-2022 : ℤ)| = 2022 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_2022_l1455_145521


namespace NUMINAMATH_CALUDE_unique_solution_ab_minus_a_minus_b_eq_one_l1455_145563

theorem unique_solution_ab_minus_a_minus_b_eq_one :
  ∃! (a b : ℕ), a * b - a - b = 1 ∧ a > b ∧ b > 0 ∧ a = 3 ∧ b = 2 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_ab_minus_a_minus_b_eq_one_l1455_145563


namespace NUMINAMATH_CALUDE_triangle_area_theorem_l1455_145516

theorem triangle_area_theorem (x : ℝ) (h1 : x > 0) :
  (1/2) * x * (3*x) = 96 → x = 8 := by sorry

end NUMINAMATH_CALUDE_triangle_area_theorem_l1455_145516


namespace NUMINAMATH_CALUDE_max_blocks_fit_l1455_145585

/-- Represents the dimensions of a rectangular box or block -/
structure Dimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the volume of a box or block given its dimensions -/
def volume (d : Dimensions) : ℕ :=
  d.length * d.width * d.height

/-- Represents the box and block dimensions -/
def box : Dimensions := ⟨4, 3, 2⟩
def block : Dimensions := ⟨1, 1, 2⟩

/-- Calculates the maximum number of blocks that can fit in the box based on volume -/
def max_blocks_by_volume : ℕ :=
  volume box / volume block

/-- Calculates the maximum number of blocks that can fit in the box based on physical arrangement -/
def max_blocks_by_arrangement : ℕ :=
  (box.length / block.length) * (box.width / block.width)

theorem max_blocks_fit :
  max_blocks_by_volume = 12 ∧ max_blocks_by_arrangement = 12 :=
sorry

end NUMINAMATH_CALUDE_max_blocks_fit_l1455_145585


namespace NUMINAMATH_CALUDE_at_least_two_inequalities_false_l1455_145540

theorem at_least_two_inequalities_false (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
  ¬(((x + y > 0) ∧ (y + z > 0) ∧ (z + x > 0) ∧ (x + 2*y < 0) ∧ (y + 2*z < 0)) ∨
    ((x + y > 0) ∧ (y + z > 0) ∧ (z + x > 0) ∧ (x + 2*y < 0) ∧ (z + 2*x < 0)) ∨
    ((x + y > 0) ∧ (y + z > 0) ∧ (z + x > 0) ∧ (y + 2*z < 0) ∧ (z + 2*x < 0)) ∨
    ((x + y > 0) ∧ (y + z > 0) ∧ (x + 2*y < 0) ∧ (y + 2*z < 0) ∧ (z + 2*x < 0)) ∨
    ((x + y > 0) ∧ (z + x > 0) ∧ (x + 2*y < 0) ∧ (y + 2*z < 0) ∧ (z + 2*x < 0)) ∨
    ((y + z > 0) ∧ (z + x > 0) ∧ (x + 2*y < 0) ∧ (y + 2*z < 0) ∧ (z + 2*x < 0))) :=
by
  sorry


end NUMINAMATH_CALUDE_at_least_two_inequalities_false_l1455_145540


namespace NUMINAMATH_CALUDE_closest_option_is_150000_l1455_145538

/-- Represents the population of the United States in 2020 --/
def us_population : ℕ := 331000000

/-- Represents the total area of the United States in square miles --/
def us_area : ℕ := 3800000

/-- Represents the number of square feet in one square mile --/
def sq_feet_per_sq_mile : ℕ := 5280 * 5280

/-- Calculates the average number of square feet per person --/
def avg_sq_feet_per_person : ℚ :=
  (us_area * sq_feet_per_sq_mile) / us_population

/-- List of given options for the average square feet per person --/
def options : List ℕ := [30000, 60000, 90000, 120000, 150000]

/-- Theorem stating that 150000 is the closest option to the actual average --/
theorem closest_option_is_150000 :
  ∃ (x : ℕ), x ∈ options ∧ 
  ∀ (y : ℕ), y ∈ options → |avg_sq_feet_per_person - x| ≤ |avg_sq_feet_per_person - y| :=
by
  sorry

end NUMINAMATH_CALUDE_closest_option_is_150000_l1455_145538


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l1455_145500

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

def is_increasing_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n < a (n + 1)

theorem geometric_sequence_common_ratio
  (a : ℕ → ℝ)
  (h_geom : is_geometric_sequence a)
  (h_incr : is_increasing_sequence a)
  (h_pos : a 1 > 0)
  (h_eq : ∀ n : ℕ, 2 * (a n + a (n + 2)) = 5 * a (n + 1)) :
  ∃ q : ℝ, (∀ n : ℕ, a (n + 1) = a n * q) ∧ q = 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l1455_145500


namespace NUMINAMATH_CALUDE_book_pages_theorem_l1455_145541

def pages_read_day1 (total : ℕ) : ℕ :=
  total / 4 + 20

def pages_left_day1 (total : ℕ) : ℕ :=
  total - pages_read_day1 total

def pages_read_day2 (total : ℕ) : ℕ :=
  (pages_left_day1 total) / 3 + 25

def pages_left_day2 (total : ℕ) : ℕ :=
  pages_left_day1 total - pages_read_day2 total

def pages_read_day3 (total : ℕ) : ℕ :=
  (pages_left_day2 total) / 2 + 30

def pages_left_day3 (total : ℕ) : ℕ :=
  pages_left_day2 total - pages_read_day3 total

theorem book_pages_theorem :
  pages_left_day3 480 = 70 := by
  sorry

end NUMINAMATH_CALUDE_book_pages_theorem_l1455_145541


namespace NUMINAMATH_CALUDE_sufficient_condition_for_inequality_l1455_145553

theorem sufficient_condition_for_inequality (m : ℝ) (h1 : m ≠ 0) :
  (m > 2 → m + 4 / m > 4) ∧ ¬(m + 4 / m > 4 → m > 2) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_condition_for_inequality_l1455_145553


namespace NUMINAMATH_CALUDE_workshop_average_salary_l1455_145531

theorem workshop_average_salary
  (total_workers : ℕ)
  (technicians : ℕ)
  (avg_salary_technicians : ℕ)
  (avg_salary_rest : ℕ)
  (h1 : total_workers = 35)
  (h2 : technicians = 7)
  (h3 : avg_salary_technicians = 16000)
  (h4 : avg_salary_rest = 6000) :
  (technicians * avg_salary_technicians + (total_workers - technicians) * avg_salary_rest) / total_workers = 8000 :=
by
  sorry

end NUMINAMATH_CALUDE_workshop_average_salary_l1455_145531


namespace NUMINAMATH_CALUDE_minimum_condition_range_l1455_145542

/-- Given a function f with derivative f'(x) = a(x+1)(x-a), 
    if f attains its minimum at x = a, then a < -1 or a > 0 -/
theorem minimum_condition_range (f : ℝ → ℝ) (a : ℝ) 
  (h_deriv : ∀ x, deriv f x = a * (x + 1) * (x - a))
  (h_min : IsLocalMin f a) :
  a < -1 ∨ a > 0 := by
sorry

end NUMINAMATH_CALUDE_minimum_condition_range_l1455_145542


namespace NUMINAMATH_CALUDE_train_length_is_415_l1455_145523

/-- Represents the problem of calculating a train's length -/
def TrainProblem (speed : ℝ) (tunnelLength : ℝ) (time : ℝ) : Prop :=
  let speedMPS := speed * 1000 / 3600
  let totalDistance := speedMPS * time
  totalDistance = tunnelLength + 415

/-- Theorem stating that given the conditions, the train length is 415 meters -/
theorem train_length_is_415 :
  TrainProblem 63 285 40 := by
  sorry

#check train_length_is_415

end NUMINAMATH_CALUDE_train_length_is_415_l1455_145523


namespace NUMINAMATH_CALUDE_sum_congruence_l1455_145591

theorem sum_congruence : (1 + 22 + 333 + 4444 + 55555 + 666666 + 7777777 + 88888888) % 9 = 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_congruence_l1455_145591
