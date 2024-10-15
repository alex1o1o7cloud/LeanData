import Mathlib

namespace NUMINAMATH_CALUDE_g_difference_l3263_326373

/-- Given a function g(n) = 1/2 * n^2 * (n+3), prove that g(s) - g(s-1) = 1/2 * (3s - 2) for any real number s. -/
theorem g_difference (s : ℝ) : 
  let g : ℝ → ℝ := λ n => (1/2) * n^2 * (n + 3)
  g s - g (s - 1) = (1/2) * (3*s - 2) := by
sorry

end NUMINAMATH_CALUDE_g_difference_l3263_326373


namespace NUMINAMATH_CALUDE_three_number_average_l3263_326300

theorem three_number_average (a b c : ℝ) 
  (h1 : a = 2 * b) 
  (h2 : a = 3 * c) 
  (h3 : a - c = 96) : 
  (a + b + c) / 3 = 88 := by
  sorry

end NUMINAMATH_CALUDE_three_number_average_l3263_326300


namespace NUMINAMATH_CALUDE_circle_in_square_area_ratio_l3263_326371

/-- The ratio of the area of a circle inscribed in a square 
    (where the circle's diameter is equal to the square's side length) 
    to the area of the square is π/4. -/
theorem circle_in_square_area_ratio : 
  ∀ s : ℝ, s > 0 → (π * (s/2)^2) / (s^2) = π/4 := by
sorry

end NUMINAMATH_CALUDE_circle_in_square_area_ratio_l3263_326371


namespace NUMINAMATH_CALUDE_factor_expression_l3263_326398

theorem factor_expression : ∀ x : ℝ, 75 * x + 45 = 15 * (5 * x + 3) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l3263_326398


namespace NUMINAMATH_CALUDE_total_legs_in_room_l3263_326360

/-- Represents the count of furniture items with their respective leg counts -/
structure FurnitureCount where
  four_leg_tables : ℕ
  four_leg_sofas : ℕ
  four_leg_chairs : ℕ
  three_leg_tables : ℕ
  one_leg_tables : ℕ
  two_leg_rocking_chairs : ℕ

/-- Calculates the total number of legs in the room -/
def total_legs (fc : FurnitureCount) : ℕ :=
  4 * fc.four_leg_tables +
  4 * fc.four_leg_sofas +
  4 * fc.four_leg_chairs +
  3 * fc.three_leg_tables +
  1 * fc.one_leg_tables +
  2 * fc.two_leg_rocking_chairs

/-- The given furniture configuration in the room -/
def room_furniture : FurnitureCount :=
  { four_leg_tables := 4
  , four_leg_sofas := 1
  , four_leg_chairs := 2
  , three_leg_tables := 3
  , one_leg_tables := 1
  , two_leg_rocking_chairs := 1 }

/-- Theorem stating that the total number of legs in the room is 40 -/
theorem total_legs_in_room : total_legs room_furniture = 40 := by
  sorry

end NUMINAMATH_CALUDE_total_legs_in_room_l3263_326360


namespace NUMINAMATH_CALUDE_polynomial_factorization_l3263_326347

theorem polynomial_factorization (t : ℝ) :
  ∃ (a b c d : ℝ), ∀ (x : ℝ),
    x^4 + t*x^2 + 1 = (x^2 + a*x + b) * (x^2 + c*x + d) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l3263_326347


namespace NUMINAMATH_CALUDE_solve_exponential_equation_l3263_326393

theorem solve_exponential_equation :
  ∃ x : ℝ, (64 : ℝ)^(3*x) = (16 : ℝ)^(4*x - 5) ∧ x = -10 := by
  sorry

end NUMINAMATH_CALUDE_solve_exponential_equation_l3263_326393


namespace NUMINAMATH_CALUDE_analysis_method_seeks_sufficient_conditions_l3263_326313

/-- The analysis method in mathematical proofs -/
def analysis_method (method : String) : Prop :=
  method = "starts from the conclusion to be proven and progressively seeks conditions that make the conclusion valid"

/-- The type of conditions sought by a proof method -/
inductive ConditionType
  | Sufficient
  | Necessary
  | NecessaryAndSufficient
  | Equivalent

/-- The conditions sought by a proof method -/
def seeks_conditions (method : String) (condition_type : ConditionType) : Prop :=
  analysis_method method → condition_type = ConditionType.Sufficient

theorem analysis_method_seeks_sufficient_conditions :
  ∀ (method : String),
  analysis_method method →
  seeks_conditions method ConditionType.Sufficient :=
by
  sorry


end NUMINAMATH_CALUDE_analysis_method_seeks_sufficient_conditions_l3263_326313


namespace NUMINAMATH_CALUDE_smallest_n_square_cube_l3263_326338

theorem smallest_n_square_cube : ∃ (n : ℕ), n > 0 ∧ 
  (∃ (k : ℕ), 4 * n = k^2) ∧ 
  (∃ (m : ℕ), 5 * n = m^3) ∧ 
  (∀ (x : ℕ), x > 0 ∧ x < n → ¬(∃ (y : ℕ), 4 * x = y^2) ∨ ¬(∃ (z : ℕ), 5 * x = z^3)) ∧
  n = 100 :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_square_cube_l3263_326338


namespace NUMINAMATH_CALUDE_library_book_purchase_l3263_326369

/-- The library's book purchase problem -/
theorem library_book_purchase :
  let total_spent : ℕ := 4500
  let total_books : ℕ := 300
  let price_zhuangzi : ℕ := 10
  let price_confucius : ℕ := 20
  let price_mencius : ℕ := 15
  let price_laozi : ℕ := 28
  let price_sunzi : ℕ := 12
  ∀ (num_zhuangzi num_confucius num_mencius num_laozi num_sunzi : ℕ),
    num_zhuangzi + num_confucius + num_mencius + num_laozi + num_sunzi = total_books →
    num_zhuangzi * price_zhuangzi + num_confucius * price_confucius + 
    num_mencius * price_mencius + num_laozi * price_laozi + 
    num_sunzi * price_sunzi = total_spent →
    num_zhuangzi = num_confucius →
    num_sunzi = 4 * num_laozi + 15 →
    num_sunzi = 195 :=
by sorry

end NUMINAMATH_CALUDE_library_book_purchase_l3263_326369


namespace NUMINAMATH_CALUDE_quadratic_equation_two_distinct_roots_l3263_326333

theorem quadratic_equation_two_distinct_roots :
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧
  (∀ x : ℝ, (x - 1) * (x + 5) = 3 * x + 1 ↔ x = x₁ ∨ x = x₂) :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_two_distinct_roots_l3263_326333


namespace NUMINAMATH_CALUDE_range_of_a_l3263_326392

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x^2 + (a + 2) * x + 1 ≥ 0) → -4 ≤ a ∧ a ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l3263_326392


namespace NUMINAMATH_CALUDE_lcm_of_18_28_45_65_l3263_326339

theorem lcm_of_18_28_45_65 : Nat.lcm 18 (Nat.lcm 28 (Nat.lcm 45 65)) = 16380 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_18_28_45_65_l3263_326339


namespace NUMINAMATH_CALUDE_binomial_20_10_l3263_326362

theorem binomial_20_10 (h1 : Nat.choose 18 8 = 43758) 
                       (h2 : Nat.choose 18 9 = 48620) 
                       (h3 : Nat.choose 18 10 = 43758) : 
  Nat.choose 20 10 = 184756 := by
  sorry

end NUMINAMATH_CALUDE_binomial_20_10_l3263_326362


namespace NUMINAMATH_CALUDE_corrected_mean_l3263_326315

def number_of_observations : ℕ := 100
def original_mean : ℝ := 125.6
def incorrect_observation1 : ℝ := 95.3
def incorrect_observation2 : ℝ := -15.9
def correct_observation1 : ℝ := 48.2
def correct_observation2 : ℝ := -35.7

theorem corrected_mean (n : ℕ) (om : ℝ) (io1 io2 co1 co2 : ℝ) :
  n = number_of_observations ∧
  om = original_mean ∧
  io1 = incorrect_observation1 ∧
  io2 = incorrect_observation2 ∧
  co1 = correct_observation1 ∧
  co2 = correct_observation2 →
  (n : ℝ) * om - (io1 + io2) + (co1 + co2) / n = 124.931 := by
  sorry

end NUMINAMATH_CALUDE_corrected_mean_l3263_326315


namespace NUMINAMATH_CALUDE_coin_problem_l3263_326331

theorem coin_problem :
  let total_coins : ℕ := 56
  let total_value : ℕ := 440
  let coins_of_one_type : ℕ := 24
  let x : ℕ := total_coins - coins_of_one_type  -- number of 10-peso coins
  let y : ℕ := coins_of_one_type  -- number of 5-peso coins
  (x + y = total_coins) ∧ (10 * x + 5 * y = total_value) → y = 24 :=
by
  sorry

end NUMINAMATH_CALUDE_coin_problem_l3263_326331


namespace NUMINAMATH_CALUDE_smallest_angle_measure_l3263_326343

theorem smallest_angle_measure (ABC ABD : ℝ) (h1 : ABC = 40) (h2 : ABD = 30) :
  ∃ (CBD : ℝ), CBD = ABC - ABD ∧ CBD = 10 ∧ ∀ (x : ℝ), x ≥ 0 → x ≥ CBD :=
by sorry

end NUMINAMATH_CALUDE_smallest_angle_measure_l3263_326343


namespace NUMINAMATH_CALUDE_min_value_expression_equality_achievable_l3263_326357

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  8 * a^3 + 6 * b^3 + 27 * c^3 + 9 / (8 * a * b * c) ≥ 18 :=
by sorry

theorem equality_achievable :
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
    8 * a^3 + 6 * b^3 + 27 * c^3 + 9 / (8 * a * b * c) = 18 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_equality_achievable_l3263_326357


namespace NUMINAMATH_CALUDE_married_men_fraction_l3263_326386

theorem married_men_fraction (total_women : ℕ) (total_people : ℕ) 
  (h1 : total_women > 0)
  (h2 : total_people > total_women)
  (h3 : (3 : ℚ) / 7 = (total_women - (total_people - total_women)) / total_women) :
  (total_people - total_women : ℚ) / total_people = 4 / 11 := by
sorry

end NUMINAMATH_CALUDE_married_men_fraction_l3263_326386


namespace NUMINAMATH_CALUDE_largest_after_erasing_100_l3263_326354

/-- Concatenates numbers from 1 to n as a string -/
def concatenateNumbers (n : ℕ) : String :=
  (List.range n).map (fun i => toString (i + 1)) |> String.join

/-- Checks if a number is the largest possible after erasing digits -/
def isLargestAfterErasing (original : String) (erased : ℕ) (result : String) : Prop :=
  result.length = original.length - erased ∧
  ∀ (other : String), other.length = original.length - erased →
    other.toNat! ≤ result.toNat!

theorem largest_after_erasing_100 :
  isLargestAfterErasing (concatenateNumbers 60) 100 "99999785960" := by
  sorry

end NUMINAMATH_CALUDE_largest_after_erasing_100_l3263_326354


namespace NUMINAMATH_CALUDE_smallest_x_absolute_value_exists_smallest_x_smallest_x_value_l3263_326397

theorem smallest_x_absolute_value (x : ℝ) : 
  (|5 * x - 3| = 32) → x ≥ -29/5 :=
by sorry

theorem exists_smallest_x : 
  ∃ x : ℝ, |5 * x - 3| = 32 ∧ ∀ y : ℝ, |5 * y - 3| = 32 → y ≥ x :=
by sorry

theorem smallest_x_value : 
  ∃ x : ℝ, x = -29/5 ∧ |5 * x - 3| = 32 ∧ ∀ y : ℝ, |5 * y - 3| = 32 → y ≥ x :=
by sorry

end NUMINAMATH_CALUDE_smallest_x_absolute_value_exists_smallest_x_smallest_x_value_l3263_326397


namespace NUMINAMATH_CALUDE_base_seven_sum_l3263_326384

/-- Given A, B, C are distinct digits in base 7 and ABC_7 + BCA_7 + CAB_7 = AAA1_7,
    prove that B + C = 6 (in base 7) if A = 1, or B + C = 12 (in base 7) if A = 2. -/
theorem base_seven_sum (A B C : ℕ) : 
  A < 7 → B < 7 → C < 7 → 
  A ≠ B → B ≠ C → A ≠ C →
  (7^2 * A + 7 * B + C) + (7^2 * B + 7 * C + A) + (7^2 * C + 7 * A + B) = 
    7^3 * A + 7^2 * A + 7 * A + 1 →
  (A = 1 ∧ B + C = 6) ∨ (A = 2 ∧ B + C = 12) :=
by sorry

end NUMINAMATH_CALUDE_base_seven_sum_l3263_326384


namespace NUMINAMATH_CALUDE_subtraction_divisibility_implies_sum_l3263_326317

/-- Represents a three-digit number in the form xyz --/
structure ThreeDigitNumber where
  x : Nat
  y : Nat
  z : Nat
  x_nonzero : x ≠ 0
  digits_bound : x < 10 ∧ y < 10 ∧ z < 10

/-- Converts a ThreeDigitNumber to its numerical value --/
def ThreeDigitNumber.toNat (n : ThreeDigitNumber) : Nat :=
  100 * n.x + 10 * n.y + n.z

theorem subtraction_divisibility_implies_sum (a b : Nat) :
  ∃ (num1 num2 : ThreeDigitNumber),
    num1.toNat = 407 + 10 * a ∧
    num2.toNat = 304 + 10 * b ∧
    830 - num1.toNat = num2.toNat ∧
    num2.toNat % 7 = 0 →
    a + b = 2 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_divisibility_implies_sum_l3263_326317


namespace NUMINAMATH_CALUDE_largest_n_for_trig_inequality_l3263_326340

theorem largest_n_for_trig_inequality : 
  (∃ (n : ℕ), n > 0 ∧ ∀ (x : ℝ), (Real.sin x)^n + (Real.cos x)^n ≥ 2/n) ∧ 
  (∀ (n : ℕ), n > 2 → ∃ (x : ℝ), (Real.sin x)^n + (Real.cos x)^n < 2/n) :=
by sorry

end NUMINAMATH_CALUDE_largest_n_for_trig_inequality_l3263_326340


namespace NUMINAMATH_CALUDE_average_b_c_is_fifty_l3263_326365

theorem average_b_c_is_fifty (a b c : ℝ) 
  (h1 : (a + b) / 2 = 45)
  (h2 : c - a = 10) :
  (b + c) / 2 = 50 := by
sorry

end NUMINAMATH_CALUDE_average_b_c_is_fifty_l3263_326365


namespace NUMINAMATH_CALUDE_trapezoid_perimeter_is_230_l3263_326377

/-- Represents a trapezoid ABCD with given properties -/
structure Trapezoid where
  BC : ℝ
  AP : ℝ
  DQ : ℝ
  AB : ℝ
  CD : ℝ
  AD_longer_than_BC : BC < AP + BC + DQ

/-- Calculates the perimeter of the trapezoid -/
def perimeter (t : Trapezoid) : ℝ :=
  t.AB + t.BC + t.CD + (t.AP + t.BC + t.DQ)

/-- Theorem stating that the perimeter of the given trapezoid is 230 -/
theorem trapezoid_perimeter_is_230 (t : Trapezoid) 
    (h1 : t.BC = 60)
    (h2 : t.AP = 24)
    (h3 : t.DQ = 11)
    (h4 : t.AB = 40)
    (h5 : t.CD = 35) : 
  perimeter t = 230 := by
  sorry

#check trapezoid_perimeter_is_230

end NUMINAMATH_CALUDE_trapezoid_perimeter_is_230_l3263_326377


namespace NUMINAMATH_CALUDE_orange_banana_relationship_l3263_326388

/-- The cost of fruits at Frank's Fruit Market -/
structure FruitCost where
  banana_to_apple : ℚ  -- ratio of bananas to apples
  apple_to_orange : ℚ  -- ratio of apples to oranges

/-- Given the cost ratios, calculate how many oranges cost as much as 24 bananas -/
def oranges_for_24_bananas (cost : FruitCost) : ℚ :=
  24 * (cost.banana_to_apple * cost.apple_to_orange)

/-- Theorem stating the relationship between banana and orange costs -/
theorem orange_banana_relationship (cost : FruitCost)
  (h1 : cost.banana_to_apple = 4 / 3)
  (h2 : cost.apple_to_orange = 5 / 2) :
  oranges_for_24_bananas cost = 36 / 5 := by
  sorry

#eval oranges_for_24_bananas ⟨4/3, 5/2⟩

end NUMINAMATH_CALUDE_orange_banana_relationship_l3263_326388


namespace NUMINAMATH_CALUDE_mango_purchase_proof_l3263_326309

def grape_quantity : ℕ := 11
def grape_price : ℕ := 98
def mango_price : ℕ := 50
def total_payment : ℕ := 1428

def mango_quantity : ℕ := (total_payment - grape_quantity * grape_price) / mango_price

theorem mango_purchase_proof : mango_quantity = 7 := by
  sorry

end NUMINAMATH_CALUDE_mango_purchase_proof_l3263_326309


namespace NUMINAMATH_CALUDE_no_divisible_seven_digit_numbers_l3263_326323

/-- A function that checks if a number uses each of the digits 1-7 exactly once. -/
def usesDigits1To7Once (n : ℕ) : Prop :=
  ∃ (a b c d e f g : ℕ),
    n = a * 1000000 + b * 100000 + c * 10000 + d * 1000 + e * 100 + f * 10 + g ∧
    ({a, b, c, d, e, f, g} : Finset ℕ) = {1, 2, 3, 4, 5, 6, 7}

/-- Theorem stating that there are no two seven-digit numbers formed using
    digits 1-7 once each where one divides the other. -/
theorem no_divisible_seven_digit_numbers :
  ¬∃ (m n : ℕ), m ≠ n ∧ 
    usesDigits1To7Once m ∧ 
    usesDigits1To7Once n ∧ 
    m ∣ n :=
by sorry

end NUMINAMATH_CALUDE_no_divisible_seven_digit_numbers_l3263_326323


namespace NUMINAMATH_CALUDE_ellipse_t_squared_range_l3263_326303

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

-- Define the focus
def focus : ℝ × ℝ := (1, 0)

-- Define point H
def H : ℝ × ℝ := (3, 0)

-- Define the condition for points A and B
def intersects_ellipse (A B : ℝ × ℝ) : Prop :=
  ellipse A.1 A.2 ∧ ellipse B.1 B.2 ∧ ∃ k : ℝ, A.2 - B.2 = k * (A.1 - B.1) ∧ A.2 = k * (A.1 - H.1) + H.2

-- Define the condition for point P
def P_condition (P : ℝ × ℝ) : Prop := ellipse P.1 P.2

-- Define the vector relation
def vector_relation (O A B P : ℝ × ℝ) (t : ℝ) : Prop :=
  (A.1 - O.1, A.2 - O.2) + (B.1 - O.1, B.2 - O.2) = t • (P.1 - O.1, P.2 - O.2)

-- Define the distance condition
def distance_condition (P A B : ℝ × ℝ) : Prop :=
  ((P.1 - A.1)^2 + (P.2 - A.2)^2)^(1/2) - ((P.1 - B.1)^2 + (P.2 - B.2)^2)^(1/2) < Real.sqrt 3

theorem ellipse_t_squared_range :
  ∀ (O A B P : ℝ × ℝ) (t : ℝ),
    intersects_ellipse A B →
    P_condition P →
    vector_relation O A B P t →
    distance_condition P A B →
    20 - Real.sqrt 283 < t^2 ∧ t^2 < 4 :=
sorry

end NUMINAMATH_CALUDE_ellipse_t_squared_range_l3263_326303


namespace NUMINAMATH_CALUDE_bus_empty_seats_after_second_stop_l3263_326353

/-- Represents the state of the bus at different stages --/
structure BusState where
  total_seats : ℕ
  occupied_seats : ℕ

/-- Calculates the number of empty seats in the bus --/
def empty_seats (state : BusState) : ℕ :=
  state.total_seats - state.occupied_seats

/-- Updates the bus state after passenger movement --/
def update_state (state : BusState) (board : ℕ) (leave : ℕ) : BusState :=
  { total_seats := state.total_seats,
    occupied_seats := state.occupied_seats + board - leave }

theorem bus_empty_seats_after_second_stop :
  let initial_state : BusState := { total_seats := 23 * 4, occupied_seats := 16 }
  let first_stop := update_state initial_state 15 3
  let second_stop := update_state first_stop 17 10
  empty_seats second_stop = 57 := by sorry


end NUMINAMATH_CALUDE_bus_empty_seats_after_second_stop_l3263_326353


namespace NUMINAMATH_CALUDE_integral_equals_minus_eight_implies_a_equals_four_l3263_326367

theorem integral_equals_minus_eight_implies_a_equals_four (a : ℝ) :
  (∫ (x : ℝ) in -a..a, (2 * x - 1)) = -8 → a = 4 := by
  sorry

end NUMINAMATH_CALUDE_integral_equals_minus_eight_implies_a_equals_four_l3263_326367


namespace NUMINAMATH_CALUDE_fraction_simplification_l3263_326316

theorem fraction_simplification (a : ℝ) (h1 : a ≠ 2) (h2 : a ≠ -2) :
  (2 * a) / (a^2 - 4) - 1 / (a - 2) = 1 / (a + 2) := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3263_326316


namespace NUMINAMATH_CALUDE_student_age_l3263_326329

theorem student_age (student_age man_age : ℕ) : 
  man_age = student_age + 26 →
  man_age + 2 = 2 * (student_age + 2) →
  student_age = 24 := by
sorry

end NUMINAMATH_CALUDE_student_age_l3263_326329


namespace NUMINAMATH_CALUDE_multiply_powers_l3263_326301

theorem multiply_powers (a : ℝ) : 2 * a^3 * 3 * a^2 = 6 * a^5 := by
  sorry

end NUMINAMATH_CALUDE_multiply_powers_l3263_326301


namespace NUMINAMATH_CALUDE_sqrt_of_sqrt_81_l3263_326328

theorem sqrt_of_sqrt_81 : Real.sqrt (Real.sqrt 81) = 3 := by sorry

end NUMINAMATH_CALUDE_sqrt_of_sqrt_81_l3263_326328


namespace NUMINAMATH_CALUDE_equation_solution_l3263_326336

theorem equation_solution (x : ℝ) : 5 * x^2 + 4 = 3 * x + 9 → (10 * x - 3)^2 = 109 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3263_326336


namespace NUMINAMATH_CALUDE_arithmetic_expressions_correctness_l3263_326322

theorem arithmetic_expressions_correctness :
  (∀ a b c : ℚ, (a + b) + c = a + (b + c)) ∧
  (∃ a b c : ℚ, (a - b) - c ≠ a - (b - c)) ∧
  (∃ a b c : ℚ, (a + b) / c ≠ a + (b / c)) ∧
  (∃ a b c : ℚ, (a / b) / c ≠ a / (b / c)) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_expressions_correctness_l3263_326322


namespace NUMINAMATH_CALUDE_x_value_proof_l3263_326383

theorem x_value_proof : ∃ x : ℝ, x = 70 * (1 + 11/100) ∧ x = 77.7 := by
  sorry

end NUMINAMATH_CALUDE_x_value_proof_l3263_326383


namespace NUMINAMATH_CALUDE_cylinder_max_volume_ratio_l3263_326321

/-- Given a rectangle with perimeter 12 that forms a cylinder, prove that the ratio of the base circumference to height is 2:1 when volume is maximized -/
theorem cylinder_max_volume_ratio (l w : ℝ) : 
  l > 0 → w > 0 → 
  2 * l + 2 * w = 12 → 
  let r := l / (2 * Real.pi)
  let h := w
  let V := Real.pi * r^2 * h
  (∀ l' w', l' > 0 → w' > 0 → 2 * l' + 2 * w' = 12 → 
    let r' := l' / (2 * Real.pi)
    let h' := w'
    Real.pi * r'^2 * h' ≤ V) →
  l / w = 2 := by
sorry

end NUMINAMATH_CALUDE_cylinder_max_volume_ratio_l3263_326321


namespace NUMINAMATH_CALUDE_toyota_not_less_than_honda_skoda_combined_l3263_326389

/-- Proves that the number of Toyotas is not less than the number of Hondas and Skodas combined in a parking lot with specific conditions. -/
theorem toyota_not_less_than_honda_skoda_combined 
  (C T H S X Y : ℕ) 
  (h1 : C - H = (3 * (C - X)) / 2)
  (h2 : C - S = (3 * (C - Y)) / 2)
  (h3 : C - T = (X + Y) / 2) :
  T ≥ H + S := by sorry

end NUMINAMATH_CALUDE_toyota_not_less_than_honda_skoda_combined_l3263_326389


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l3263_326320

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 (a > 0, b > 0),
    if the distance from (4, 0) to its asymptote is √2,
    then its eccentricity is (2√14)/7 -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (4 * b / Real.sqrt (a^2 + b^2) = Real.sqrt 2) →
  (Real.sqrt (a^2 + b^2) / a = 2 * Real.sqrt 14 / 7) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l3263_326320


namespace NUMINAMATH_CALUDE_function_is_constant_one_l3263_326352

def is_valid_function (f : ℕ → ℕ) : Prop :=
  (∀ a b : ℕ, a > 0 ∧ b > 0 → f (a^2 + b^2) = f a * f b) ∧
  (∀ a : ℕ, a > 0 → f (a^2) = (f a)^2)

theorem function_is_constant_one (f : ℕ → ℕ) (h : is_valid_function f) :
  ∀ n : ℕ, n > 0 → f n = 1 :=
by sorry

end NUMINAMATH_CALUDE_function_is_constant_one_l3263_326352


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l3263_326319

theorem complex_fraction_simplification :
  let i : ℂ := Complex.I
  (2 : ℂ) / (1 + i) = 1 - i :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l3263_326319


namespace NUMINAMATH_CALUDE_coin_toss_probability_l3263_326382

theorem coin_toss_probability (n : ℕ) : (∀ k : ℕ, k < n → 1 - (1/2)^k < 15/16) ∧ 1 - (1/2)^n ≥ 15/16 → n = 4 := by
  sorry

end NUMINAMATH_CALUDE_coin_toss_probability_l3263_326382


namespace NUMINAMATH_CALUDE_team_size_is_five_l3263_326304

/-- The length of the relay race in meters -/
def relay_length : ℕ := 150

/-- The distance each team member runs in meters -/
def member_distance : ℕ := 30

/-- The number of people on the team -/
def team_size : ℕ := relay_length / member_distance

theorem team_size_is_five : team_size = 5 := by
  sorry

end NUMINAMATH_CALUDE_team_size_is_five_l3263_326304


namespace NUMINAMATH_CALUDE_price_reduction_theorem_l3263_326324

/-- Proves that a price reduction resulting in an 80% increase in sales and a 26% increase in total revenue implies a 30% price reduction -/
theorem price_reduction_theorem (P S : ℝ) (x : ℝ) 
  (h1 : x > 0) 
  (h2 : x < 100) 
  (h3 : P > 0) 
  (h4 : S > 0) 
  (h5 : P * (1 - x / 100) * (S * 1.8) = P * S * 1.26) : 
  x = 30 := by
  sorry

end NUMINAMATH_CALUDE_price_reduction_theorem_l3263_326324


namespace NUMINAMATH_CALUDE_isosceles_triangles_same_perimeter_l3263_326356

-- Define the properties of the triangles
def isIsosceles (a b c : ℝ) : Prop := (a = b) ∨ (b = c) ∨ (a = c)
def perimeter (a b c : ℝ) : ℝ := a + b + c

-- Define the theorem
theorem isosceles_triangles_same_perimeter 
  (c d : ℝ) 
  (h1 : isIsosceles 7 7 10) 
  (h2 : isIsosceles c c d) 
  (h3 : c ≠ d) 
  (h4 : perimeter 7 7 10 = 24) 
  (h5 : perimeter c c d = 24) :
  d = 2 := by sorry

end NUMINAMATH_CALUDE_isosceles_triangles_same_perimeter_l3263_326356


namespace NUMINAMATH_CALUDE_hex_to_decimal_l3263_326390

/-- Given a hexadecimal number 10k5₍₆₎ where k is a positive integer,
    if this number equals 239 when converted to decimal, then k = 3. -/
theorem hex_to_decimal (k : ℕ+) : (1 * 6^3 + k * 6 + 5 = 239) → k = 3 := by
  sorry

end NUMINAMATH_CALUDE_hex_to_decimal_l3263_326390


namespace NUMINAMATH_CALUDE_hanging_spheres_mass_ratio_l3263_326363

/-- Given two hanging spheres with masses m₁ and m₂, where the tension in the upper string
    is twice the tension in the lower string, prove that the ratio of masses m₁/m₂ = 1 -/
theorem hanging_spheres_mass_ratio (m₁ m₂ : ℝ) (g : ℝ) (h : g > 0) : 
  (m₁ * g + m₂ * g = 2 * (m₂ * g)) → m₁ / m₂ = 1 := by
  sorry

end NUMINAMATH_CALUDE_hanging_spheres_mass_ratio_l3263_326363


namespace NUMINAMATH_CALUDE_first_day_is_thursday_l3263_326308

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a day in a month -/
structure MonthDay where
  day : Nat
  dayOfWeek : DayOfWeek

/-- Function to get the next day of the week -/
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

/-- Function to get the previous day of the week -/
def prevDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Saturday
  | DayOfWeek.Monday => DayOfWeek.Sunday
  | DayOfWeek.Tuesday => DayOfWeek.Monday
  | DayOfWeek.Wednesday => DayOfWeek.Tuesday
  | DayOfWeek.Thursday => DayOfWeek.Wednesday
  | DayOfWeek.Friday => DayOfWeek.Thursday
  | DayOfWeek.Saturday => DayOfWeek.Friday

/-- Theorem: If the 24th day of a month is a Saturday, then the 1st day of that month is a Thursday -/
theorem first_day_is_thursday (m : MonthDay) (h : m.day = 24 ∧ m.dayOfWeek = DayOfWeek.Saturday) :
  ∃ (firstDay : MonthDay), firstDay.day = 1 ∧ firstDay.dayOfWeek = DayOfWeek.Thursday :=
by sorry

end NUMINAMATH_CALUDE_first_day_is_thursday_l3263_326308


namespace NUMINAMATH_CALUDE_license_plate_increase_l3263_326391

theorem license_plate_increase : 
  let old_plates := 26 * 10^3
  let new_plates := 26^2 * 10^4
  new_plates / old_plates = 260 := by
sorry

end NUMINAMATH_CALUDE_license_plate_increase_l3263_326391


namespace NUMINAMATH_CALUDE_difference_calculation_l3263_326342

theorem difference_calculation (x y : ℝ) (hx : x = 497) (hy : y = 325) :
  2/5 * (3*x + 7*y) - 3/5 * (x * y) = -95408.6 := by
  sorry

end NUMINAMATH_CALUDE_difference_calculation_l3263_326342


namespace NUMINAMATH_CALUDE_square_side_length_average_l3263_326361

theorem square_side_length_average (a b c : ℝ) (ha : a = 25) (hb : b = 64) (hc : c = 121) :
  (a.sqrt + b.sqrt + c.sqrt) / 3 = 8 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_average_l3263_326361


namespace NUMINAMATH_CALUDE_smallest_BD_is_five_l3263_326395

/-- Represents a quadrilateral with side lengths and an angle -/
structure Quadrilateral :=
  (AB BC CD DA : ℝ)
  (angleBDA : ℝ)

/-- The smallest possible integer value of BD in the given quadrilateral -/
def smallest_integer_BD (q : Quadrilateral) : ℕ :=
  sorry

/-- Theorem stating the smallest possible integer value of BD -/
theorem smallest_BD_is_five (q : Quadrilateral) 
  (h1 : q.AB = 7)
  (h2 : q.BC = 15)
  (h3 : q.CD = 7)
  (h4 : q.DA = 11)
  (h5 : q.angleBDA = 90) :
  smallest_integer_BD q = 5 :=
sorry

end NUMINAMATH_CALUDE_smallest_BD_is_five_l3263_326395


namespace NUMINAMATH_CALUDE_smallest_number_divisible_l3263_326349

theorem smallest_number_divisible (n : ℕ) : n ≥ 58 →
  (∃ k : ℕ, n - 10 = 24 * k) →
  (∀ m : ℕ, m < n → ¬(∃ k : ℕ, m - 10 = 24 * k)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_divisible_l3263_326349


namespace NUMINAMATH_CALUDE_chess_tournament_games_l3263_326306

/-- The number of games in a chess tournament where each player plays twice against every other player. -/
def tournament_games (n : ℕ) : ℕ := n * (n - 1)

/-- Theorem: In a chess tournament with 17 players, where each player plays twice against every other player, the total number of games played is 272. -/
theorem chess_tournament_games :
  tournament_games 17 = 272 := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_games_l3263_326306


namespace NUMINAMATH_CALUDE_least_K_inequality_l3263_326335

theorem least_K_inequality (K : ℝ) : (∀ x y : ℝ, (1 + 20 * x^2) * (1 + 19 * y^2) ≥ K * x * y) ↔ K ≤ 8 * Real.sqrt 95 := by
  sorry

end NUMINAMATH_CALUDE_least_K_inequality_l3263_326335


namespace NUMINAMATH_CALUDE_inequality_proof_l3263_326387

theorem inequality_proof (x y : ℝ) : (1 / 2) * (x^2 + y^2) - x * y ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3263_326387


namespace NUMINAMATH_CALUDE_chocolate_bars_per_box_l3263_326348

theorem chocolate_bars_per_box (total_bars : ℕ) (total_boxes : ℕ) (bars_per_box : ℕ) :
  total_bars = 475 →
  total_boxes = 19 →
  total_bars = total_boxes * bars_per_box →
  bars_per_box = 25 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_bars_per_box_l3263_326348


namespace NUMINAMATH_CALUDE_greatest_four_digit_multiple_of_17_l3263_326326

theorem greatest_four_digit_multiple_of_17 :
  ∀ n : ℕ, 1000 ≤ n ∧ n ≤ 9999 ∧ 17 ∣ n → n ≤ 9996 :=
by sorry

end NUMINAMATH_CALUDE_greatest_four_digit_multiple_of_17_l3263_326326


namespace NUMINAMATH_CALUDE_probability_of_white_marble_l3263_326318

/-- Given a box of marbles with four colors, prove the probability of drawing a white marble. -/
theorem probability_of_white_marble (total_marbles : ℕ) 
  (p_green p_red_or_blue p_white : ℝ) : 
  total_marbles = 100 →
  p_green = 1/5 →
  p_red_or_blue = 0.55 →
  p_green + p_red_or_blue + p_white = 1 →
  p_white = 0.25 := by
  sorry

#check probability_of_white_marble

end NUMINAMATH_CALUDE_probability_of_white_marble_l3263_326318


namespace NUMINAMATH_CALUDE_single_point_ellipse_l3263_326345

/-- 
If the graph of 3x^2 + 4y^2 + 6x - 8y + c = 0 consists of a single point, then c = 7.
-/
theorem single_point_ellipse (c : ℝ) : 
  (∃! p : ℝ × ℝ, 3 * p.1^2 + 4 * p.2^2 + 6 * p.1 - 8 * p.2 + c = 0) → c = 7 := by
  sorry

end NUMINAMATH_CALUDE_single_point_ellipse_l3263_326345


namespace NUMINAMATH_CALUDE_yellow_candles_count_l3263_326396

/-- The number of yellow candles on a birthday cake --/
def yellow_candles (total_candles red_candles blue_candles : ℕ) : ℕ :=
  total_candles - (red_candles + blue_candles)

/-- Theorem: The number of yellow candles is 27 --/
theorem yellow_candles_count :
  yellow_candles 79 14 38 = 27 := by
  sorry

end NUMINAMATH_CALUDE_yellow_candles_count_l3263_326396


namespace NUMINAMATH_CALUDE_quadratic_roots_difference_l3263_326385

theorem quadratic_roots_difference (x : ℝ) : 
  let f : ℝ → ℝ := λ x => x^2 + 42*x + 384
  let roots := {x : ℝ | f x = 0}
  ∃ (r₁ r₂ : ℝ), r₁ ∈ roots ∧ r₂ ∈ roots ∧ |r₁ - r₂| = 8 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_difference_l3263_326385


namespace NUMINAMATH_CALUDE_complex_modulus_l3263_326378

theorem complex_modulus (z : ℂ) (h : (1 + Complex.I) * z = 1 - 2 * Complex.I^3) : 
  Complex.abs z = Real.sqrt 10 / 2 := by
sorry

end NUMINAMATH_CALUDE_complex_modulus_l3263_326378


namespace NUMINAMATH_CALUDE_correct_operation_l3263_326302

theorem correct_operation (a : ℝ) : (-a + 2) * (-a - 2) = a^2 - 4 := by
  sorry

end NUMINAMATH_CALUDE_correct_operation_l3263_326302


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3263_326341

theorem complex_equation_solution (z : ℂ) :
  z * (1 - Complex.I) = 2 * Complex.I → z = -1 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3263_326341


namespace NUMINAMATH_CALUDE_fraction_sum_l3263_326374

theorem fraction_sum (a b : ℚ) (h : a / b = 1 / 3) : (a + b) / b = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_l3263_326374


namespace NUMINAMATH_CALUDE_problem_1_l3263_326305

theorem problem_1 : -36 * (3/4 - 1/6 + 2/9 - 5/12) + |(-21/5) / (7/25)| = 61 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l3263_326305


namespace NUMINAMATH_CALUDE_domain_of_function_1_l3263_326355

theorem domain_of_function_1 (x : ℝ) : 
  (x ≥ 1 ∨ x < -1) ↔ (x ≠ -1 ∧ (x - 1) / (x + 1) ≥ 0) :=
sorry

#check domain_of_function_1

end NUMINAMATH_CALUDE_domain_of_function_1_l3263_326355


namespace NUMINAMATH_CALUDE_value_of_expression_l3263_326344

theorem value_of_expression (m n : ℤ) (h : m - n = 2) : 
  (n - m)^3 - (m - n)^2 + 1 = -11 := by
sorry

end NUMINAMATH_CALUDE_value_of_expression_l3263_326344


namespace NUMINAMATH_CALUDE_square_area_triple_l3263_326346

/-- Given a square I with diagonal 2a, prove that a square II with triple the area of square I has an area of 6a² -/
theorem square_area_triple (a : ℝ) :
  let diagonal_I : ℝ := 2 * a
  let area_I : ℝ := (diagonal_I ^ 2) / 2
  let area_II : ℝ := 3 * area_I
  area_II = 6 * a ^ 2 := by
sorry

end NUMINAMATH_CALUDE_square_area_triple_l3263_326346


namespace NUMINAMATH_CALUDE_interval_length_implies_k_l3263_326337

theorem interval_length_implies_k (k : ℝ) : 
  k > 0 → 
  (Set.Icc (-3) 3 : Set ℝ) = {x : ℝ | x^2 + k * |x| ≤ 2019} → 
  k = 670 := by
sorry

end NUMINAMATH_CALUDE_interval_length_implies_k_l3263_326337


namespace NUMINAMATH_CALUDE_triangle_properties_l3263_326327

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  a * Real.sin B = Real.sqrt 3 * b * Real.cos A →
  Real.cos C = 1 / 3 →
  c = 4 * Real.sqrt 2 →
  A = π / 3 ∧ 
  (1 / 2 * a * c * Real.sin B) = 4 * Real.sqrt 3 + 3 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l3263_326327


namespace NUMINAMATH_CALUDE_special_rectangle_AB_length_l3263_326364

/-- Represents a rectangle with specific properties -/
structure SpecialRectangle where
  AB : ℝ
  BC : ℝ
  PQ : ℝ
  XY : ℝ
  equalAreas : Bool
  PQparallelAB : Bool
  XYequation : Bool

/-- The theorem statement -/
theorem special_rectangle_AB_length
  (rect : SpecialRectangle)
  (h1 : rect.BC = 19)
  (h2 : rect.PQ = 87)
  (h3 : rect.equalAreas)
  (h4 : rect.PQparallelAB)
  (h5 : rect.XYequation) :
  rect.AB = 193 := by
  sorry

end NUMINAMATH_CALUDE_special_rectangle_AB_length_l3263_326364


namespace NUMINAMATH_CALUDE_division_remainder_problem_l3263_326350

theorem division_remainder_problem : ∃ (r : ℕ), 15968 = 179 * 89 + r ∧ r < 179 := by
  sorry

end NUMINAMATH_CALUDE_division_remainder_problem_l3263_326350


namespace NUMINAMATH_CALUDE_tangent_is_simson_line_l3263_326370

/-- A parabola in a 2D plane. -/
structure Parabola where
  -- Add necessary fields to define a parabola

/-- A triangle in a 2D plane. -/
structure Triangle where
  -- Add necessary fields to define a triangle

/-- The Simson line of a triangle with respect to a point. -/
def SimsonLine (t : Triangle) (p : ℝ × ℝ) : Set (ℝ × ℝ) :=
  sorry

/-- The tangent line to a parabola at a given point. -/
def TangentLine (p : Parabola) (point : ℝ × ℝ) : Set (ℝ × ℝ) :=
  sorry

/-- The vertex of a parabola. -/
def Vertex (p : Parabola) : ℝ × ℝ :=
  sorry

/-- Given three tangent lines to a parabola, find their intersection points forming a triangle. -/
def TriangleFromTangents (p : Parabola) (t1 t2 t3 : Set (ℝ × ℝ)) : Triangle :=
  sorry

theorem tangent_is_simson_line (p : Parabola) (t1 t2 t3 : Set (ℝ × ℝ)) :
  TangentLine p (Vertex p) = SimsonLine (TriangleFromTangents p t1 t2 t3) (Vertex p) :=
sorry

end NUMINAMATH_CALUDE_tangent_is_simson_line_l3263_326370


namespace NUMINAMATH_CALUDE_hill_climbing_time_l3263_326312

theorem hill_climbing_time 
  (descent_time : ℝ) 
  (average_speed_total : ℝ) 
  (average_speed_climbing : ℝ) 
  (h1 : descent_time = 2)
  (h2 : average_speed_total = 3.5)
  (h3 : average_speed_climbing = 2.625) :
  ∃ (climb_time : ℝ), 
    climb_time = 4 ∧ 
    average_speed_total = (2 * average_speed_climbing * climb_time) / (climb_time + descent_time) := by
  sorry

end NUMINAMATH_CALUDE_hill_climbing_time_l3263_326312


namespace NUMINAMATH_CALUDE_prob_red_then_black_is_three_fourths_l3263_326311

/-- A deck of cards with red and black cards -/
structure Deck :=
  (total_cards : ℕ)
  (red_cards : ℕ)
  (black_cards : ℕ)
  (h_total : total_cards = red_cards + black_cards)
  (h_equal : red_cards = black_cards)

/-- The probability of drawing a red card first and a black card second -/
def prob_red_then_black (d : Deck) : ℚ :=
  (d.red_cards : ℚ) * d.black_cards / (d.total_cards * (d.total_cards - 1))

/-- Theorem: For a deck with 64 cards, half red and half black,
    the probability of drawing a red card first and a black card second is 3/4 -/
theorem prob_red_then_black_is_three_fourths (d : Deck) 
    (h_total : d.total_cards = 64) : prob_red_then_black d = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_prob_red_then_black_is_three_fourths_l3263_326311


namespace NUMINAMATH_CALUDE_max_candy_pieces_l3263_326334

theorem max_candy_pieces (n : ℕ) (avg : ℕ) (min_pieces : ℕ) :
  n = 30 →
  avg = 7 →
  min_pieces = 1 →
  ∃ (max_pieces : ℕ), max_pieces = n * avg - (n - 1) * min_pieces ∧
                       max_pieces = 181 :=
by sorry

end NUMINAMATH_CALUDE_max_candy_pieces_l3263_326334


namespace NUMINAMATH_CALUDE_trout_calculation_l3263_326375

/-- The number of people fishing -/
def num_people : ℕ := 2

/-- The number of trout each person gets after splitting -/
def trout_per_person : ℕ := 9

/-- The total number of trout caught -/
def total_trout : ℕ := num_people * trout_per_person

theorem trout_calculation : total_trout = 18 := by
  sorry

end NUMINAMATH_CALUDE_trout_calculation_l3263_326375


namespace NUMINAMATH_CALUDE_quadratic_roots_l3263_326332

theorem quadratic_roots : ∃ (x₁ x₂ : ℝ), x₁ = 3 ∧ x₂ = -1 ∧ 
  (x₁^2 - 2*x₁ - 3 = 0) ∧ (x₂^2 - 2*x₂ - 3 = 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_l3263_326332


namespace NUMINAMATH_CALUDE_completing_square_equivalence_l3263_326379

theorem completing_square_equivalence :
  ∀ x : ℝ, (x^2 + 8*x + 9 = 0) ↔ ((x + 4)^2 = 7) := by
  sorry

end NUMINAMATH_CALUDE_completing_square_equivalence_l3263_326379


namespace NUMINAMATH_CALUDE_range_of_m_l3263_326399

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, -1 < x ∧ x < 4 → x > 2 * m^2 - 3) ∧ 
  (∃ x : ℝ, x > 2 * m^2 - 3 ∧ (x ≤ -1 ∨ x ≥ 4)) → 
  -1 ≤ m ∧ m ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l3263_326399


namespace NUMINAMATH_CALUDE_quadratic_real_root_condition_l3263_326307

/-- A quadratic equation x^2 + bx + 25 = 0 has at least one real root
    if and only if b ∈ (-∞, -10] ∪ [10, ∞) -/
theorem quadratic_real_root_condition (b : ℝ) :
  (∃ x : ℝ, x^2 + b*x + 25 = 0) ↔ b ≤ -10 ∨ b ≥ 10 := by sorry

end NUMINAMATH_CALUDE_quadratic_real_root_condition_l3263_326307


namespace NUMINAMATH_CALUDE_triangle_incenter_distance_l3263_326368

/-- A triangle with sides a, b, and c, and incenter J -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  J : ℝ × ℝ

/-- The incircle of a triangle -/
structure Incircle where
  center : ℝ × ℝ
  radius : ℝ

/-- Given a triangle PQR with sides PQ = 30, PR = 29, and QR = 31,
    and J as the intersection of internal angle bisectors (incenter),
    prove that QJ = √(226 - r²), where r is the radius of the incircle -/
theorem triangle_incenter_distance (T : Triangle) (I : Incircle) :
  T.a = 30 ∧ T.b = 29 ∧ T.c = 31 ∧ 
  I.center = T.J ∧
  I.radius = r →
  ∃ (QJ : ℝ), QJ = Real.sqrt (226 - r^2) :=
sorry

end NUMINAMATH_CALUDE_triangle_incenter_distance_l3263_326368


namespace NUMINAMATH_CALUDE_group_size_l3263_326366

/-- The number of people in a group, given certain weight changes. -/
theorem group_size (avg_increase : ℝ) (old_weight new_weight : ℝ) (h1 : avg_increase = 1.5)
    (h2 : new_weight - old_weight = 6) : ℤ :=
  4

#check group_size

end NUMINAMATH_CALUDE_group_size_l3263_326366


namespace NUMINAMATH_CALUDE_work_hours_theorem_l3263_326325

/-- Calculates the total hours worked given the number of days and hours per day -/
def total_hours (days : ℝ) (hours_per_day : ℝ) : ℝ :=
  days * hours_per_day

/-- Proves that working 2 hours per day for 4 days results in 8 total hours -/
theorem work_hours_theorem :
  let days : ℝ := 4
  let hours_per_day : ℝ := 2
  total_hours days hours_per_day = 8 := by
  sorry

end NUMINAMATH_CALUDE_work_hours_theorem_l3263_326325


namespace NUMINAMATH_CALUDE_companion_pair_expression_zero_l3263_326310

/-- Definition of companion number pairs -/
def is_companion_pair (a b : ℚ) : Prop :=
  a / 2 + b / 3 = (a + b) / 5

/-- Theorem: For any companion number pair (m,n), 
    the expression 14m-5n-[5m-3(3n-1)]+3 always evaluates to 0 -/
theorem companion_pair_expression_zero (m n : ℚ) 
  (h : is_companion_pair m n) : 
  14*m - 5*n - (5*m - 3*(3*n - 1)) + 3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_companion_pair_expression_zero_l3263_326310


namespace NUMINAMATH_CALUDE_cone_height_ratio_l3263_326394

/-- Proves the ratio of heights for a cone with reduced height and constant base --/
theorem cone_height_ratio (original_height : ℝ) (base_circumference : ℝ) (shorter_volume : ℝ) :
  original_height = 15 →
  base_circumference = 10 * Real.pi →
  shorter_volume = 50 * Real.pi →
  ∃ (shorter_height : ℝ),
    (1 / 3) * Real.pi * (base_circumference / (2 * Real.pi))^2 * shorter_height = shorter_volume ∧
    shorter_height / original_height = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_cone_height_ratio_l3263_326394


namespace NUMINAMATH_CALUDE_midpoint_octagon_area_ratio_l3263_326358

/-- A regular octagon -/
structure RegularOctagon where
  vertices : Fin 8 → ℝ × ℝ
  is_regular : sorry  -- Additional properties to ensure the octagon is regular

/-- The octagon formed by joining the midpoints of a regular octagon's sides -/
def midpointOctagon (o : RegularOctagon) : RegularOctagon :=
  sorry

/-- The area of a regular octagon -/
def area (o : RegularOctagon) : ℝ :=
  sorry

/-- The theorem stating that the area of the midpoint octagon is 3/4 of the original octagon -/
theorem midpoint_octagon_area_ratio (o : RegularOctagon) :
  area (midpointOctagon o) = (3/4) * area o :=
sorry

end NUMINAMATH_CALUDE_midpoint_octagon_area_ratio_l3263_326358


namespace NUMINAMATH_CALUDE_simplify_expression_l3263_326376

theorem simplify_expression (z : ℝ) : 3 * (4 - 5 * z) - 2 * (2 + 3 * z) = 8 - 21 * z := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3263_326376


namespace NUMINAMATH_CALUDE_quadratic_roots_condition_l3263_326372

theorem quadratic_roots_condition (a : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ a * x^2 - 4 * x + 2 = 0 ∧ a * y^2 - 4 * y + 2 = 0) ↔ 
  (a ≤ 2 ∧ a ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_condition_l3263_326372


namespace NUMINAMATH_CALUDE_minimum_employees_needed_l3263_326330

/-- Represents the set of employees monitoring water pollution -/
def W : Finset Nat := sorry

/-- Represents the set of employees monitoring air pollution -/
def A : Finset Nat := sorry

/-- Represents the set of employees monitoring land pollution -/
def L : Finset Nat := sorry

theorem minimum_employees_needed : 
  (Finset.card W = 95) → 
  (Finset.card A = 80) → 
  (Finset.card L = 50) → 
  (Finset.card (W ∩ A) = 30) → 
  (Finset.card (A ∩ L) = 20) → 
  (Finset.card (W ∩ L) = 15) → 
  (Finset.card (W ∩ A ∩ L) = 10) → 
  Finset.card (W ∪ A ∪ L) = 170 := by
  sorry

end NUMINAMATH_CALUDE_minimum_employees_needed_l3263_326330


namespace NUMINAMATH_CALUDE_estimate_fish_population_l3263_326359

/-- Estimates the number of fish in a pond using the mark-recapture method. -/
theorem estimate_fish_population
  (initially_tagged : ℕ)
  (second_catch : ℕ)
  (tagged_in_second_catch : ℕ)
  (h1 : initially_tagged = 100)
  (h2 : second_catch = 300)
  (h3 : tagged_in_second_catch = 15) :
  (initially_tagged * second_catch) / tagged_in_second_catch = 2000 := by
  sorry

#check estimate_fish_population

end NUMINAMATH_CALUDE_estimate_fish_population_l3263_326359


namespace NUMINAMATH_CALUDE_probability_three_white_balls_l3263_326380

/-- The probability of drawing three white balls from a box containing 7 white balls and 8 black balls is 1/13. -/
theorem probability_three_white_balls (white_balls black_balls : ℕ) 
  (h1 : white_balls = 7) (h2 : black_balls = 8) : 
  (Nat.choose white_balls 3 : ℚ) / (Nat.choose (white_balls + black_balls) 3) = 1 / 13 := by
  sorry

#eval Nat.choose 7 3
#eval Nat.choose 15 3
#eval (35 : ℚ) / 455

end NUMINAMATH_CALUDE_probability_three_white_balls_l3263_326380


namespace NUMINAMATH_CALUDE_min_value_of_linear_function_l3263_326381

theorem min_value_of_linear_function :
  ∃ (m : ℝ), ∀ (x y : ℝ), 2*x + 3*y ≥ m ∧ ∃ (x₀ y₀ : ℝ), 2*x₀ + 3*y₀ = m :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_linear_function_l3263_326381


namespace NUMINAMATH_CALUDE_wire_length_ratio_l3263_326314

/-- The ratio of wire lengths in cube frame construction -/
theorem wire_length_ratio (bonnie_wire_pieces : ℕ) (bonnie_wire_length : ℝ) 
  (roark_wire_length : ℝ) : 
  bonnie_wire_pieces = 12 →
  bonnie_wire_length = 8 →
  roark_wire_length = 0.5 →
  (bonnie_wire_length ^ 3) * (roark_wire_length ^ 3)⁻¹ * 
    (12 * roark_wire_length) * (bonnie_wire_pieces * bonnie_wire_length)⁻¹ = 256 →
  (bonnie_wire_pieces * bonnie_wire_length) * 
    ((bonnie_wire_length ^ 3) * (roark_wire_length ^ 3)⁻¹ * (12 * roark_wire_length))⁻¹ = 1 / 256 := by
  sorry

end NUMINAMATH_CALUDE_wire_length_ratio_l3263_326314


namespace NUMINAMATH_CALUDE_initial_population_proof_l3263_326351

/-- The population change function over 5 years -/
def population_change (P : ℝ) : ℝ :=
  P * 0.9 * 1.1 * 0.9 * 1.15 * 0.75

/-- Theorem stating the initial population given the final population -/
theorem initial_population_proof : 
  ∃ P : ℕ, population_change (P : ℝ) = 4455 ∧ P = 5798 :=
sorry

end NUMINAMATH_CALUDE_initial_population_proof_l3263_326351
