import Mathlib

namespace NUMINAMATH_CALUDE_power_three_mod_eight_l2874_287464

theorem power_three_mod_eight : 3^20 % 8 = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_three_mod_eight_l2874_287464


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l2874_287486

-- Problem 1
theorem problem_1 (a : ℝ) (h : a = -1) : 
  (1 : ℝ) * (a + 3)^2 + (3 + a) * (3 - a) = 12 := by sorry

-- Problem 2
theorem problem_2 (x y : ℝ) (hx : x = 2) (hy : y = 3) : 
  (x - 2*y) * (x + 2*y) - (x + 2*y)^2 + 8*y^2 = -24 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l2874_287486


namespace NUMINAMATH_CALUDE_probability_theorem_l2874_287467

def shirts : ℕ := 6
def pants : ℕ := 7
def socks : ℕ := 8
def total_articles : ℕ := shirts + pants + socks
def selected_articles : ℕ := 4

def probability_two_shirts_one_pant_one_sock : ℚ :=
  (Nat.choose shirts 2 * Nat.choose pants 1 * Nat.choose socks 1) /
  Nat.choose total_articles selected_articles

theorem probability_theorem :
  probability_two_shirts_one_pant_one_sock = 40 / 285 := by
  sorry

end NUMINAMATH_CALUDE_probability_theorem_l2874_287467


namespace NUMINAMATH_CALUDE_charity_ticket_sales_l2874_287466

theorem charity_ticket_sales (total_tickets : ℕ) (total_revenue : ℕ) 
  (h : total_tickets = 180 ∧ total_revenue = 2800) :
  ∃ (full_price : ℕ) (half_price_count : ℕ),
    full_price > 0 ∧
    half_price_count + (total_tickets - half_price_count) = total_tickets ∧
    half_price_count * (full_price / 2) + (total_tickets - half_price_count) * full_price = total_revenue ∧
    half_price_count = 328 := by
  sorry

end NUMINAMATH_CALUDE_charity_ticket_sales_l2874_287466


namespace NUMINAMATH_CALUDE_cube_difference_not_divisible_l2874_287432

theorem cube_difference_not_divisible (a b : ℤ) 
  (ha : Odd a) (hb : Odd b) (hab : a ≠ b) : 
  ¬ (2 * (a - b) ∣ (a^3 - b^3)) := by
  sorry

end NUMINAMATH_CALUDE_cube_difference_not_divisible_l2874_287432


namespace NUMINAMATH_CALUDE_all_statements_false_l2874_287416

theorem all_statements_false : 
  (¬ ∀ (x : ℚ), -x ≠ |x|) ∧ 
  (¬ ∀ (x y : ℚ), x = -y → x = y⁻¹) ∧ 
  (¬ ∀ (x y : ℚ), |x| = |y| → x = y) := by
  sorry

end NUMINAMATH_CALUDE_all_statements_false_l2874_287416


namespace NUMINAMATH_CALUDE_last_two_digits_product_l2874_287405

/-- Given an integer n, returns the tens digit -/
def tensDigit (n : ℤ) : ℤ := (n / 10) % 10

/-- Given an integer n, returns the units digit -/
def unitsDigit (n : ℤ) : ℤ := n % 10

/-- Theorem: For any integer divisible by 5 with the sum of its last two digits being 12,
    the product of its last two digits is 35 -/
theorem last_two_digits_product (n : ℤ) : 
  n % 5 = 0 → 
  tensDigit n + unitsDigit n = 12 → 
  tensDigit n * unitsDigit n = 35 := by
  sorry

end NUMINAMATH_CALUDE_last_two_digits_product_l2874_287405


namespace NUMINAMATH_CALUDE_xyz_value_l2874_287411

theorem xyz_value (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 49)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 19)
  (h3 : x + y + z = 7) : 
  x * y * z = 10 := by
sorry

end NUMINAMATH_CALUDE_xyz_value_l2874_287411


namespace NUMINAMATH_CALUDE_prob_two_blue_balls_l2874_287414

/-- The probability of drawing two blue balls from an urn --/
theorem prob_two_blue_balls (total : ℕ) (blue : ℕ) (h1 : total = 10) (h2 : blue = 5) :
  (blue.choose 2 : ℚ) / total.choose 2 = 2 / 9 := by
  sorry

end NUMINAMATH_CALUDE_prob_two_blue_balls_l2874_287414


namespace NUMINAMATH_CALUDE_marilyn_bottle_caps_l2874_287462

/-- Given that Marilyn starts with 51 bottle caps and shares 36 with Nancy, 
    prove that she ends up with 15 bottle caps. -/
theorem marilyn_bottle_caps 
  (start : ℕ) 
  (shared : ℕ) 
  (h1 : start = 51) 
  (h2 : shared = 36) : 
  start - shared = 15 := by
sorry

end NUMINAMATH_CALUDE_marilyn_bottle_caps_l2874_287462


namespace NUMINAMATH_CALUDE_sara_letters_ratio_l2874_287412

theorem sara_letters_ratio (january february total : ℕ) 
  (h1 : january = 6)
  (h2 : february = 9)
  (h3 : total = 33) :
  (total - january - february) / january = 3 := by
  sorry

end NUMINAMATH_CALUDE_sara_letters_ratio_l2874_287412


namespace NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_smallest_primes_l2874_287452

/-- The five smallest prime numbers -/
def smallest_primes : List Nat := [2, 3, 5, 7, 11]

/-- A number is four-digit if it's between 1000 and 9999 inclusive -/
def is_four_digit (n : Nat) : Prop := 1000 ≤ n ∧ n ≤ 9999

/-- A number is divisible by all numbers in a list if it's divisible by their product -/
def divisible_by_all (n : Nat) (list : List Nat) : Prop :=
  n % (list.prod) = 0

theorem smallest_four_digit_divisible_by_smallest_primes :
  (2310 = (smallest_primes.prod)) ∧
  (is_four_digit 2310) ∧
  (divisible_by_all 2310 smallest_primes) ∧
  (∀ m : Nat, m < 2310 → ¬(is_four_digit m ∧ divisible_by_all m smallest_primes)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_smallest_primes_l2874_287452


namespace NUMINAMATH_CALUDE_curve_to_line_equation_l2874_287448

/-- Proves that the curve parameterized by (x,y) = (3t + 6, 5t - 8) 
    can be expressed as the line equation y = (5/3)x - 18 -/
theorem curve_to_line_equation : 
  ∀ (t x y : ℝ), x = 3*t + 6 ∧ y = 5*t - 8 → y = (5/3)*x - 18 := by
  sorry

end NUMINAMATH_CALUDE_curve_to_line_equation_l2874_287448


namespace NUMINAMATH_CALUDE_f_properties_l2874_287430

noncomputable def f (x : ℝ) := 2 * (Real.cos (x / 2))^2 + Real.sqrt 3 * Real.sin x

theorem f_properties :
  (∃ (M : ℝ), ∀ x, f x ≤ M ∧ (∃ x, f x = M) ∧ M = 3) ∧
  (∀ k : ℤ, f (Real.pi / 3 + 2 * k * Real.pi) = 3) ∧
  (∀ α : ℝ, Real.tan (α / 2) = 1 / 2 → f α = (8 + 4 * Real.sqrt 3) / 5) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l2874_287430


namespace NUMINAMATH_CALUDE_problem_statement_l2874_287417

def P : Set ℝ := {-1, 1}
def Q (a : ℝ) : Set ℝ := {x : ℝ | a * x = 1}

theorem problem_statement (a : ℝ) : P ∪ Q a = P → a ∈ ({-1, 0, 1} : Set ℝ) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2874_287417


namespace NUMINAMATH_CALUDE_amanda_keeps_22_candy_bars_l2874_287489

/-- The number of candy bars Amanda keeps for herself given the initial amount, 
    the amount given to her sister initially, the amount bought later, 
    and the multiplier for the second giving. -/
def amanda_candy_bars (initial : ℕ) (first_given : ℕ) (bought : ℕ) (multiplier : ℕ) : ℕ :=
  initial - first_given + bought - (multiplier * first_given)

/-- Theorem stating that Amanda keeps 22 candy bars for herself 
    given the specific conditions in the problem. -/
theorem amanda_keeps_22_candy_bars : 
  amanda_candy_bars 7 3 30 4 = 22 := by sorry

end NUMINAMATH_CALUDE_amanda_keeps_22_candy_bars_l2874_287489


namespace NUMINAMATH_CALUDE_weekly_egg_supply_l2874_287413

/-- The number of eggs in a dozen -/
def dozen : ℕ := 12

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The number of dozens of eggs supplied to the first store daily -/
def store1_supply : ℕ := 5

/-- The number of eggs supplied to the second store daily -/
def store2_supply : ℕ := 30

/-- Theorem: The total number of eggs supplied to both stores in a week is 630 -/
theorem weekly_egg_supply : 
  (store1_supply * dozen + store2_supply) * days_in_week = 630 := by
  sorry

end NUMINAMATH_CALUDE_weekly_egg_supply_l2874_287413


namespace NUMINAMATH_CALUDE_largest_common_value_l2874_287492

theorem largest_common_value (n m : ℕ) : 
  (∃ n m : ℕ, 479 = 2 + 3 * n ∧ 479 = 3 + 7 * m) ∧ 
  (∀ k : ℕ, k < 500 → k > 479 → ¬(∃ p q : ℕ, k = 2 + 3 * p ∧ k = 3 + 7 * q)) := by
sorry

end NUMINAMATH_CALUDE_largest_common_value_l2874_287492


namespace NUMINAMATH_CALUDE_volume_change_specific_l2874_287463

/-- Represents the change in volume of a rectangular parallelepiped -/
def volume_change (a b c : ℝ) (da db dc : ℝ) : ℝ :=
  b * c * da + a * c * db + a * b * dc

/-- Theorem stating the change in volume for specific dimensions and changes -/
theorem volume_change_specific :
  let a : ℝ := 8
  let b : ℝ := 6
  let c : ℝ := 3
  let da : ℝ := 0.1
  let db : ℝ := 0.05
  let dc : ℝ := -0.15
  volume_change a b c da db dc = -4.2 := by
  sorry

#eval volume_change 8 6 3 0.1 0.05 (-0.15)

end NUMINAMATH_CALUDE_volume_change_specific_l2874_287463


namespace NUMINAMATH_CALUDE_six_digit_divisibility_theorem_l2874_287453

/-- Represents a 6-digit number in the form 739ABC -/
def SixDigitNumber (a b c : Nat) : Nat :=
  739000 + 100 * a + 10 * b + c

/-- Checks if a number is divisible by 7, 8, and 9 -/
def isDivisibleBy789 (n : Nat) : Prop :=
  n % 7 = 0 ∧ n % 8 = 0 ∧ n % 9 = 0

/-- The main theorem stating the possible values for A, B, and C -/
theorem six_digit_divisibility_theorem :
  ∀ a b c : Nat,
  a < 10 ∧ b < 10 ∧ c < 10 →
  isDivisibleBy789 (SixDigitNumber a b c) →
  (a = 3 ∧ b = 6 ∧ c = 8) ∨ (a = 8 ∧ b = 7 ∧ c = 2) :=
by sorry

end NUMINAMATH_CALUDE_six_digit_divisibility_theorem_l2874_287453


namespace NUMINAMATH_CALUDE_max_reciprocal_sum_l2874_287470

theorem max_reciprocal_sum (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 1) :
  (∀ m : ℝ, (1 / a + 1 / b ≥ m) → m ≤ 4) ∧ 
  (∃ a b : ℝ, 0 < a ∧ 0 < b ∧ a + b = 1 ∧ 1 / a + 1 / b = 4) :=
sorry

end NUMINAMATH_CALUDE_max_reciprocal_sum_l2874_287470


namespace NUMINAMATH_CALUDE_expression_simplification_l2874_287498

theorem expression_simplification (a b : ℝ) 
  (ha : a = Real.sqrt 3 + 2) 
  (hb : b = Real.sqrt 3 - 2) : 
  (a^2 / (a^2 + 2*a*b + b^2) - a / (a + b)) / (a^2 / (a^2 - b^2) - b / (a - b) - 1) = 2 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2874_287498


namespace NUMINAMATH_CALUDE_equal_sampling_most_representative_l2874_287480

/-- Represents a school in the survey --/
inductive School
| A
| B
| C
| D

/-- Represents a survey method --/
structure SurveyMethod where
  schools : List School
  studentsPerSchool : Nat

/-- Defines the representativeness of a survey method --/
def representativeness (method : SurveyMethod) : ℝ :=
  sorry

/-- The number of schools in the survey --/
def totalSchools : Nat := 4

/-- The survey method that samples from all schools equally --/
def equalSamplingMethod : SurveyMethod :=
  { schools := [School.A, School.B, School.C, School.D],
    studentsPerSchool := 150 }

/-- Theorem stating that the equal sampling method is the most representative --/
theorem equal_sampling_most_representative :
  ∀ (method : SurveyMethod),
    method.schools.length = totalSchools →
    representativeness equalSamplingMethod ≥ representativeness method :=
  sorry

end NUMINAMATH_CALUDE_equal_sampling_most_representative_l2874_287480


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l2874_287410

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^3 - x^2 + 1 ≤ 0) ↔ (∃ x : ℝ, x^3 - x^2 + 1 > 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l2874_287410


namespace NUMINAMATH_CALUDE_smallest_n_l2874_287409

/-- The smallest three-digit positive integer n such that n + 7 is divisible by 9 and n - 10 is divisible by 6 -/
theorem smallest_n : ∃ n : ℕ, 
  (100 ≤ n ∧ n ≤ 999) ∧ 
  (9 ∣ (n + 7)) ∧ 
  (6 ∣ (n - 10)) ∧ 
  (∀ m : ℕ, (100 ≤ m ∧ m < n ∧ (9 ∣ (m + 7)) ∧ (6 ∣ (m - 10))) → False) ∧
  n = 118 := by
sorry

end NUMINAMATH_CALUDE_smallest_n_l2874_287409


namespace NUMINAMATH_CALUDE_sons_age_l2874_287443

/-- Given a man and his son, where the man is 32 years older than his son,
    and in two years the man's age will be twice the age of his son,
    prove that the present age of the son is 30 years. -/
theorem sons_age (son_age man_age : ℕ) : 
  man_age = son_age + 32 →
  man_age + 2 = 2 * (son_age + 2) →
  son_age = 30 := by
sorry

end NUMINAMATH_CALUDE_sons_age_l2874_287443


namespace NUMINAMATH_CALUDE_mystery_book_price_l2874_287457

theorem mystery_book_price (biography_price : ℝ) (total_discount : ℝ) 
  (biography_quantity : ℕ) (mystery_quantity : ℕ) (total_discount_rate : ℝ) 
  (mystery_discount_rate : ℝ) :
  biography_price = 20 →
  total_discount = 19 →
  biography_quantity = 5 →
  mystery_quantity = 3 →
  total_discount_rate = 0.43 →
  mystery_discount_rate = 0.375 →
  ∃ (mystery_price : ℝ),
    mystery_price * mystery_quantity * mystery_discount_rate + 
    biography_price * biography_quantity * (total_discount_rate - mystery_discount_rate) = 
    total_discount ∧
    mystery_price = 12 :=
by sorry

end NUMINAMATH_CALUDE_mystery_book_price_l2874_287457


namespace NUMINAMATH_CALUDE_quadratic_solution_l2874_287476

theorem quadratic_solution (p q : ℝ) (hp : p ≠ 0) (hq : q ≠ 0)
  (h1 : p^2 + 2*p*p + q = 0)
  (h2 : q^2 + 2*p*q + q = 0) :
  p = 1 ∧ q = -3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_l2874_287476


namespace NUMINAMATH_CALUDE_fraction_equality_l2874_287429

theorem fraction_equality : (1998 - 998) / 1000 = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2874_287429


namespace NUMINAMATH_CALUDE_union_condition_intersection_condition_l2874_287441

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}
def B (m : ℝ) : Set ℝ := {x | m - 2 ≤ x ∧ x ≤ m + 2}

-- Theorem for the first part
theorem union_condition (m : ℝ) : A ∪ B m = A → m = 1 := by sorry

-- Theorem for the second part
theorem intersection_condition (m : ℝ) : A ∩ B m = {x | 0 ≤ x ∧ x ≤ 3} → m = 2 := by sorry

end NUMINAMATH_CALUDE_union_condition_intersection_condition_l2874_287441


namespace NUMINAMATH_CALUDE_sum_of_squares_l2874_287431

theorem sum_of_squares (x y : ℝ) (h1 : x + y = 8) (h2 : x * y = 28) : x^2 + y^2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_l2874_287431


namespace NUMINAMATH_CALUDE_angle_measure_theorem_l2874_287436

theorem angle_measure_theorem (x : ℝ) : 
  (180 - x = 4 * (90 - x)) → x = 60 := by
  sorry

end NUMINAMATH_CALUDE_angle_measure_theorem_l2874_287436


namespace NUMINAMATH_CALUDE_f_1993_of_3_eq_one_fifth_l2874_287439

-- Define the function f
def f (x : ℚ) : ℚ := (1 + x) / (1 - 3 * x)

-- Define the iterated function f_n recursively
def f_n : ℕ → (ℚ → ℚ)
  | 0 => id
  | n + 1 => f ∘ (f_n n)

-- State the theorem
theorem f_1993_of_3_eq_one_fifth : f_n 1993 3 = 1/5 := by sorry

end NUMINAMATH_CALUDE_f_1993_of_3_eq_one_fifth_l2874_287439


namespace NUMINAMATH_CALUDE_sin_2A_eq_sin_2B_neither_sufficient_nor_necessary_l2874_287485

/-- A triangle ABC -/
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  angle_sum : A + B + C = π

/-- Definition of an isosceles triangle -/
def is_isosceles (t : Triangle) : Prop :=
  t.A = t.B ∨ t.B = t.C ∨ t.C = t.A

/-- The condition sin 2A = sin 2B -/
def condition (t : Triangle) : Prop :=
  Real.sin (2 * t.A) = Real.sin (2 * t.B)

/-- The main theorem to prove -/
theorem sin_2A_eq_sin_2B_neither_sufficient_nor_necessary :
  ¬(∀ t : Triangle, condition t → is_isosceles t) ∧
  ¬(∀ t : Triangle, is_isosceles t → condition t) := by
  sorry

end NUMINAMATH_CALUDE_sin_2A_eq_sin_2B_neither_sufficient_nor_necessary_l2874_287485


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_theorem_l2874_287491

noncomputable def inscribed_circle_radius (side_length : ℝ) (A₁ A₂ : ℝ) : ℝ :=
  sorry

theorem inscribed_circle_radius_theorem :
  let side_length : ℝ := 4
  let A₁ : ℝ := 8
  let A₂ : ℝ := 8
  -- Square circumscribes both circles
  side_length ^ 2 = A₁ + A₂ →
  -- Arithmetic progression condition
  A₁ + A₂ / 2 = (A₁ + (A₁ + A₂)) / 2 →
  -- Radius calculation
  inscribed_circle_radius side_length A₁ A₂ = 2 * Real.sqrt (2 / Real.pi)
  := by sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_theorem_l2874_287491


namespace NUMINAMATH_CALUDE_equation_one_solutions_l2874_287472

theorem equation_one_solutions (x : ℝ) : (5*x + 2) * (4 - x) = 0 ↔ x = -2/5 ∨ x = 4 := by
  sorry

#check equation_one_solutions

end NUMINAMATH_CALUDE_equation_one_solutions_l2874_287472


namespace NUMINAMATH_CALUDE_stacy_heather_walk_l2874_287458

/-- The problem of determining the time difference between Stacy and Heather's start times -/
theorem stacy_heather_walk (total_distance : ℝ) (heather_speed : ℝ) (stacy_speed : ℝ) 
  (heather_distance : ℝ) :
  total_distance = 25 →
  heather_speed = 5 →
  stacy_speed = heather_speed + 1 →
  heather_distance = 10.272727272727273 →
  ∃ (time_diff : ℝ), 
    time_diff * 60 = 24 ∧ 
    heather_distance / heather_speed = 
      (total_distance - heather_distance) / stacy_speed - time_diff :=
by sorry


end NUMINAMATH_CALUDE_stacy_heather_walk_l2874_287458


namespace NUMINAMATH_CALUDE_canoe_oar_probability_l2874_287494

theorem canoe_oar_probability (p_row : ℝ) (h_p_row : p_row = 0.84) : 
  ∃ (p_right : ℝ), 
    (p_right = 1 - Real.sqrt (1 - p_row)) ∧ 
    (p_right = 0.6) := by
  sorry

end NUMINAMATH_CALUDE_canoe_oar_probability_l2874_287494


namespace NUMINAMATH_CALUDE_fraction_equivalence_l2874_287475

theorem fraction_equivalence :
  ∀ (n : ℚ), (4 + n) / (7 + n) = 3 / 4 ↔ n = 5 := by sorry

end NUMINAMATH_CALUDE_fraction_equivalence_l2874_287475


namespace NUMINAMATH_CALUDE_male_democrat_ratio_l2874_287419

theorem male_democrat_ratio (total_participants : ℕ) 
  (female_democrats : ℕ) (h1 : total_participants = 840) 
  (h2 : female_democrats = 140) 
  (h3 : female_democrats * 2 ≤ total_participants) : 
  (total_participants / 3 - female_democrats) * 4 = 
  (total_participants - female_democrats * 2) := by
  sorry

#check male_democrat_ratio

end NUMINAMATH_CALUDE_male_democrat_ratio_l2874_287419


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2874_287407

/-- An arithmetic sequence is a sequence where the difference between any two consecutive terms is constant. -/
def IsArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The problem statement -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h_arith : IsArithmeticSequence a) 
  (h_eq : 4 * a 3 + a 11 - 3 * a 5 = 10) : 
  a 4 = 5 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2874_287407


namespace NUMINAMATH_CALUDE_product_of_roots_l2874_287495

theorem product_of_roots (x : ℝ) : (x + 3) * (x - 4) = 22 → 
  ∃ y : ℝ, (y + 3) * (y - 4) = 22 ∧ x * y = -34 :=
by
  sorry

end NUMINAMATH_CALUDE_product_of_roots_l2874_287495


namespace NUMINAMATH_CALUDE_election_outcomes_l2874_287433

/-- The number of students participating in the election -/
def total_students : ℕ := 4

/-- The number of students eligible for the entertainment committee member role -/
def eligible_for_entertainment : ℕ := total_students - 1

/-- The number of positions available for each role -/
def positions_per_role : ℕ := 1

/-- Theorem: The number of ways to select a class monitor and an entertainment committee member
    from 4 students, where one specific student cannot be the entertainment committee member,
    is equal to 9. -/
theorem election_outcomes :
  (eligible_for_entertainment.choose positions_per_role) *
  (eligible_for_entertainment.choose positions_per_role) = 9 := by
  sorry

end NUMINAMATH_CALUDE_election_outcomes_l2874_287433


namespace NUMINAMATH_CALUDE_bakery_rolls_combinations_l2874_287497

theorem bakery_rolls_combinations :
  let n : ℕ := 8  -- total number of rolls
  let k : ℕ := 4  -- number of roll types
  let remaining : ℕ := n - k  -- remaining rolls after putting one in each category
  (Nat.choose (remaining + k - 1) (k - 1)) = 35 := by
  sorry

end NUMINAMATH_CALUDE_bakery_rolls_combinations_l2874_287497


namespace NUMINAMATH_CALUDE_sphere_radius_from_shadows_l2874_287404

/-- Given a sphere and a cone on a horizontal field with parallel sun rays,
    if the sphere's shadow extends 20 meters from its base,
    and a 3-meter-high cone casts a 5-meter-long shadow,
    then the radius of the sphere is 12 meters. -/
theorem sphere_radius_from_shadows (sphere_shadow : ℝ) (cone_height cone_shadow : ℝ)
  (h_sphere_shadow : sphere_shadow = 20)
  (h_cone_height : cone_height = 3)
  (h_cone_shadow : cone_shadow = 5) :
  sphere_shadow * (cone_height / cone_shadow) = 12 :=
by sorry

end NUMINAMATH_CALUDE_sphere_radius_from_shadows_l2874_287404


namespace NUMINAMATH_CALUDE_remainder_9876543210_mod_101_l2874_287445

theorem remainder_9876543210_mod_101 : 9876543210 % 101 = 31 := by
  sorry

end NUMINAMATH_CALUDE_remainder_9876543210_mod_101_l2874_287445


namespace NUMINAMATH_CALUDE_triangle_minimum_value_l2874_287427

theorem triangle_minimum_value (A B C : Real) (a b c : Real) :
  -- Triangle ABC with sides a, b, c opposite to angles A, B, C
  (0 < B) → (B < Real.pi / 2) →
  (Real.cos B)^2 + (1/2) * Real.sin (2 * B) = 1 →
  -- |BC + AB| = 3
  b = 3 →
  -- Minimum value of 16b/(ac)
  (∀ x y z : Real, x > 0 → y > 0 → z > 0 →
    (Real.cos x)^2 + (1/2) * Real.sin (2 * x) = 1 →
    y = 3 →
    16 * y / (z * x) ≥ 16 * (2 - Real.sqrt 2) / 3) ∧
  (∃ x y z : Real, x > 0 ∧ y > 0 ∧ z > 0 ∧
    (Real.cos x)^2 + (1/2) * Real.sin (2 * x) = 1 ∧
    y = 3 ∧
    16 * y / (z * x) = 16 * (2 - Real.sqrt 2) / 3) :=
by sorry

end NUMINAMATH_CALUDE_triangle_minimum_value_l2874_287427


namespace NUMINAMATH_CALUDE_percent_relationship_l2874_287473

theorem percent_relationship (a b : ℝ) (h : a = 1.2 * b) : 4 * b = (10 / 3) * a := by
  sorry

end NUMINAMATH_CALUDE_percent_relationship_l2874_287473


namespace NUMINAMATH_CALUDE_a_2022_eq_674_l2874_287471

def a : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | 2 => 1
  | n+3 => (n+3) / (a n * a (n+1) * a (n+2))

theorem a_2022_eq_674 : a 2022 = 674 := by
  sorry

end NUMINAMATH_CALUDE_a_2022_eq_674_l2874_287471


namespace NUMINAMATH_CALUDE_july_rainfall_l2874_287402

theorem july_rainfall (march april may june : ℝ) (h1 : march = 3.79) (h2 : april = 4.5) 
  (h3 : may = 3.95) (h4 : june = 3.09) (h5 : (march + april + may + june + july) / 5 = 4) : 
  july = 4.67 := by
  sorry

end NUMINAMATH_CALUDE_july_rainfall_l2874_287402


namespace NUMINAMATH_CALUDE_ginos_white_bears_l2874_287482

theorem ginos_white_bears :
  ∀ (brown_bears white_bears black_bears total_bears : ℕ),
    brown_bears = 15 →
    black_bears = 27 →
    total_bears = 66 →
    total_bears = brown_bears + white_bears + black_bears →
    white_bears = 24 := by
  sorry

end NUMINAMATH_CALUDE_ginos_white_bears_l2874_287482


namespace NUMINAMATH_CALUDE_car_speed_second_hour_l2874_287449

/-- Proves that given a car's speed of 95 km/h in the first hour and an average speed of 77.5 km/h over two hours, the speed in the second hour is 60 km/h. -/
theorem car_speed_second_hour 
  (speed_first_hour : ℝ) 
  (average_speed : ℝ) 
  (h1 : speed_first_hour = 95)
  (h2 : average_speed = 77.5) : 
  ∃ (speed_second_hour : ℝ), 
    speed_second_hour = 60 ∧ 
    average_speed = (speed_first_hour + speed_second_hour) / 2 := by
  sorry


end NUMINAMATH_CALUDE_car_speed_second_hour_l2874_287449


namespace NUMINAMATH_CALUDE_sqrt_eight_same_type_as_sqrt_two_l2874_287487

/-- Two real numbers are of the same type if one is a rational multiple of the other -/
def same_type (a b : ℝ) : Prop := ∃ q : ℚ, a = q * b

/-- √2 is a real number -/
axiom sqrt_two : ℝ

/-- √8 is a real number -/
axiom sqrt_eight : ℝ

/-- The statement to be proved -/
theorem sqrt_eight_same_type_as_sqrt_two : same_type sqrt_eight sqrt_two := by sorry

end NUMINAMATH_CALUDE_sqrt_eight_same_type_as_sqrt_two_l2874_287487


namespace NUMINAMATH_CALUDE_gcd_of_B_is_five_l2874_287450

def B : Set ℕ := {n : ℕ | ∃ x : ℕ, x > 0 ∧ n = (x-2) + (x-1) + x + (x+1) + (x+2)}

theorem gcd_of_B_is_five :
  ∃ d : ℕ, d > 0 ∧ (∀ n ∈ B, d ∣ n) ∧ (∀ m : ℕ, (∀ n ∈ B, m ∣ n) → m ∣ d) ∧ d = 5 := by
sorry

end NUMINAMATH_CALUDE_gcd_of_B_is_five_l2874_287450


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_exponential_inequality_l2874_287437

theorem negation_of_existence (P : ℝ → Prop) :
  (¬∃ x > 0, P x) ↔ (∀ x > 0, ¬P x) :=
by sorry

theorem negation_of_exponential_inequality :
  (¬∃ x > 0, 3^x < x^2) ↔ (∀ x > 0, 3^x ≥ x^2) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_exponential_inequality_l2874_287437


namespace NUMINAMATH_CALUDE_fraction_equality_l2874_287421

theorem fraction_equality (a b : ℝ) (ha : a ≠ 0) (h : (a + 2*b) / a = 4) : 
  a / (b - a) = 2 := by
sorry

end NUMINAMATH_CALUDE_fraction_equality_l2874_287421


namespace NUMINAMATH_CALUDE_small_triangles_in_large_triangle_l2874_287406

theorem small_triangles_in_large_triangle :
  let large_side : ℝ := 15
  let small_side : ℝ := 3
  let area (side : ℝ) := (Real.sqrt 3 / 4) * side^2
  let num_small_triangles := (area large_side) / (area small_side)
  num_small_triangles = 25 := by sorry

end NUMINAMATH_CALUDE_small_triangles_in_large_triangle_l2874_287406


namespace NUMINAMATH_CALUDE_intersection_value_l2874_287454

/-- Given a proportional function y = kx (k ≠ 0) and an inverse proportional function y = -5/x
    intersecting at points A(x₁, y₁) and B(x₂, y₂), the value of x₁y₂ - 3x₂y₁ is equal to 10. -/
theorem intersection_value (k : ℝ) (x₁ y₁ x₂ y₂ : ℝ) : 
  k ≠ 0 →
  y₁ = k * x₁ →
  y₁ = -5 / x₁ →
  y₂ = k * x₂ →
  y₂ = -5 / x₂ →
  x₁ * y₂ - 3 * x₂ * y₁ = 10 := by
  sorry

end NUMINAMATH_CALUDE_intersection_value_l2874_287454


namespace NUMINAMATH_CALUDE_triangle_special_sequence_equilateral_l2874_287456

/-- A triangle with angles forming an arithmetic sequence and reciprocals of side lengths forming an arithmetic sequence is equilateral. -/
theorem triangle_special_sequence_equilateral (A B C : ℝ) (a b c : ℝ) :
  -- Angles form an arithmetic sequence
  ∃ (d : ℝ), (B = A + d ∧ C = B + d) →
  -- Reciprocals of side lengths form an arithmetic sequence
  ∃ (k : ℝ), (1/b = 1/a + k ∧ 1/c = 1/b + k) →
  -- Angles sum to 180°
  A + B + C = 180 →
  -- Side lengths are positive
  a > 0 ∧ b > 0 ∧ c > 0 →
  -- Triangle inequality
  a + b > c ∧ b + c > a ∧ c + a > b →
  -- Conclusion: The triangle is equilateral
  A = 60 ∧ B = 60 ∧ C = 60 := by
sorry

end NUMINAMATH_CALUDE_triangle_special_sequence_equilateral_l2874_287456


namespace NUMINAMATH_CALUDE_spherical_cap_area_ratio_l2874_287496

/-- Given two concentric spheres and a spherical cap area on the smaller sphere,
    calculate the corresponding spherical cap area on the larger sphere. -/
theorem spherical_cap_area_ratio (R₁ R₂ A₁ : ℝ) (h₁ : R₁ > 0) (h₂ : R₂ > 0) (h₃ : A₁ > 0) :
  let A₂ := A₁ * (R₂ / R₁)^2
  R₁ = 4 → R₂ = 6 → A₁ = 17 → A₂ = 38.25 := by
  sorry

end NUMINAMATH_CALUDE_spherical_cap_area_ratio_l2874_287496


namespace NUMINAMATH_CALUDE_top100_top10_difference_l2874_287468

/-- Represents the number of songs in different categories --/
structure SongCounts where
  total : Nat
  top10 : Nat
  unreleased : Nat

/-- Calculates the number of songs on the top 100 charts --/
def top100Count (s : SongCounts) : Nat :=
  s.total - s.top10 - s.unreleased

/-- Theorem stating the difference between top 100 and top 10 hits --/
theorem top100_top10_difference (s : SongCounts) 
  (h1 : s.total = 80)
  (h2 : s.top10 = 25)
  (h3 : s.unreleased = s.top10 - 5) :
  top100Count s - s.top10 = 10 := by
  sorry

#eval top100Count { total := 80, top10 := 25, unreleased := 20 } - 25

end NUMINAMATH_CALUDE_top100_top10_difference_l2874_287468


namespace NUMINAMATH_CALUDE_range_of_a_l2874_287401

open Set

def A : Set ℝ := {x | -5 < x ∧ x < 1}
def B : Set ℝ := {x | -2 < x ∧ x < 8}
def C (a : ℝ) : Set ℝ := {x | x < a}

theorem range_of_a (a : ℝ) (h : A ∩ B ⊆ C a) : a ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2874_287401


namespace NUMINAMATH_CALUDE_small_triangle_perimeter_l2874_287415

/-- Represents a triangle divided into smaller triangles -/
structure DividedTriangle where
  large_perimeter : ℝ
  num_small_triangles : ℕ
  small_perimeter : ℝ

/-- The property that the sum of 6 small triangle perimeters minus 3 small triangle perimeters
    equals the large triangle perimeter -/
def perimeter_property (dt : DividedTriangle) : Prop :=
  6 * dt.small_perimeter - 3 * dt.small_perimeter = dt.large_perimeter

/-- Theorem stating that for a triangle with perimeter 120 divided into 9 equal smaller triangles,
    each small triangle has a perimeter of 40 -/
theorem small_triangle_perimeter
  (dt : DividedTriangle)
  (h1 : dt.large_perimeter = 120)
  (h2 : dt.num_small_triangles = 9)
  (h3 : perimeter_property dt) :
  dt.small_perimeter = 40 := by
  sorry

end NUMINAMATH_CALUDE_small_triangle_perimeter_l2874_287415


namespace NUMINAMATH_CALUDE_polynomial_problems_l2874_287484

theorem polynomial_problems :
  (∀ x y, ∃ k, (2 - b) * x^2 + (a + 3) * x + (-6) * y + 7 = k) →
  (a - b)^2 = 25 ∧
  (∀ x y, ∃ k, (-1 - n) * x^2 + (-m + 6) * x + (-18) * y + 5 = k) →
  n = -1 ∧ m = 6 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_problems_l2874_287484


namespace NUMINAMATH_CALUDE_mildreds_oranges_l2874_287422

/-- The number of oranges Mildred's father gave her -/
def oranges_given (initial final : ℕ) : ℕ := final - initial

theorem mildreds_oranges : oranges_given 77 79 = 2 := by
  sorry

end NUMINAMATH_CALUDE_mildreds_oranges_l2874_287422


namespace NUMINAMATH_CALUDE_difference_of_squares_l2874_287479

theorem difference_of_squares (m : ℝ) : m^2 - 1 = (m + 1) * (m - 1) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l2874_287479


namespace NUMINAMATH_CALUDE_smallest_rectangular_block_l2874_287460

theorem smallest_rectangular_block (l m n : ℕ) : 
  (l - 1) * (m - 1) * (n - 1) = 462 → 
  l * m * n ≥ 672 ∧ 
  ∃ (l' m' n' : ℕ), (l' - 1) * (m' - 1) * (n' - 1) = 462 ∧ l' * m' * n' = 672 :=
by sorry

end NUMINAMATH_CALUDE_smallest_rectangular_block_l2874_287460


namespace NUMINAMATH_CALUDE_teacher_distribution_l2874_287447

/-- The number of ways to distribute teachers to schools -/
def distribute_teachers (total_teachers : ℕ) (female_teachers : ℕ) (schools : ℕ) : ℕ :=
  sorry

/-- The number of ways to distribute teachers to schools with constraints -/
def distribute_teachers_constrained (total_teachers : ℕ) (female_teachers : ℕ) (schools : ℕ) : ℕ :=
  sorry

theorem teacher_distribution :
  distribute_teachers 4 2 3 = 36 ∧
  distribute_teachers_constrained 4 2 3 = 30 :=
sorry

end NUMINAMATH_CALUDE_teacher_distribution_l2874_287447


namespace NUMINAMATH_CALUDE_unique_base_representation_l2874_287420

/-- The fraction we're considering -/
def fraction : ℚ := 8 / 65

/-- The repeating digits in the base-k representation -/
def repeating_digits : List ℕ := [2, 4]

/-- 
Given a positive integer k, this function should return true if and only if
the base-k representation of the fraction is 0.24242424...
-/
def is_correct_representation (k : ℕ) : Prop :=
  k > 0 ∧ 
  fraction = (2 / k + 4 / k^2) / (1 - 1 / k^2)

/-- The theorem to be proved -/
theorem unique_base_representation : 
  ∃! k : ℕ, is_correct_representation k ∧ k = 18 :=
sorry

end NUMINAMATH_CALUDE_unique_base_representation_l2874_287420


namespace NUMINAMATH_CALUDE_sum_of_xyz_l2874_287483

theorem sum_of_xyz (x y z : ℝ) 
  (h1 : x*z/(x+y) + y*x/(y+z) + z*y/(z+x) = -5)
  (h2 : y*z/(x+y) + z*x/(y+z) + x*y/(z+x) = 7) :
  x + y + z = 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_xyz_l2874_287483


namespace NUMINAMATH_CALUDE_coffee_shop_spending_coffee_shop_spending_proof_l2874_287493

theorem coffee_shop_spending : ℝ → ℝ → Prop :=
  fun b d =>
    (d = 0.6 * b) →  -- David spent 40 cents less for each dollar Ben spent
    (b = d + 14) →   -- Ben paid $14 more than David
    (b + d = 56)     -- Their total spending

-- The proof is omitted
theorem coffee_shop_spending_proof : ∃ b d : ℝ, coffee_shop_spending b d := by sorry

end NUMINAMATH_CALUDE_coffee_shop_spending_coffee_shop_spending_proof_l2874_287493


namespace NUMINAMATH_CALUDE_evaluate_complex_fraction_l2874_287459

theorem evaluate_complex_fraction : 
  (3 * Real.sqrt 7) / (Real.sqrt 3 + Real.sqrt 5 + Real.sqrt 7 + Real.sqrt 11) = 
  -(1/6) * (Real.sqrt 21 + Real.sqrt 35 - Real.sqrt 77) - 7/3 := by
sorry

end NUMINAMATH_CALUDE_evaluate_complex_fraction_l2874_287459


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2874_287499

theorem inequality_solution_set (x : ℝ) : 
  (abs (2*x - 1) + abs (2*x + 3) < 5) ↔ (-3/2 ≤ x ∧ x < 3/4) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2874_287499


namespace NUMINAMATH_CALUDE_grandfathers_age_l2874_287465

/-- Given the conditions about a family's ages, prove the grandfather's age 5 years ago. -/
theorem grandfathers_age (father_age : ℕ) (h1 : father_age = 58) :
  ∃ (son_age grandfather_age : ℕ),
    father_age - son_age = son_age ∧ 
    (son_age - 5) * 2 = grandfather_age ∧
    grandfather_age = 48 :=
by sorry

end NUMINAMATH_CALUDE_grandfathers_age_l2874_287465


namespace NUMINAMATH_CALUDE_triangle_angles_l2874_287442

theorem triangle_angles (a b c : ℝ) (ha : a = 3) (hb : b = Real.sqrt 11) (hc : c = 2 + Real.sqrt 5) :
  ∃ (A B C : ℝ), 
    (0 < A ∧ A < π) ∧ 
    (0 < B ∧ B < π) ∧ 
    (0 < C ∧ C < π) ∧ 
    A + B + C = π ∧
    B = C ∧
    A = π - 2*B := by
  sorry

end NUMINAMATH_CALUDE_triangle_angles_l2874_287442


namespace NUMINAMATH_CALUDE_colored_ngon_at_most_two_colors_l2874_287428

/-- A regular n-gon with colored sides and diagonals -/
structure ColoredNGon where
  n : ℕ
  vertices : Fin n → Point
  colors : ℕ
  coloring : (Fin n × Fin n) → Fin colors

/-- The coloring satisfies the first condition -/
def satisfies_condition1 (R : ColoredNGon) : Prop :=
  ∀ c : Fin R.colors, ∀ A B : Fin R.n,
    (R.coloring (A, B) = c) ∨
    (∃ C : Fin R.n, R.coloring (A, C) = c ∧ R.coloring (B, C) = c)

/-- The coloring satisfies the second condition -/
def satisfies_condition2 (R : ColoredNGon) : Prop :=
  ∀ A B C : Fin R.n,
    (R.coloring (A, B) ≠ R.coloring (B, C)) →
    (R.coloring (A, C) = R.coloring (A, B) ∨ R.coloring (A, C) = R.coloring (B, C))

/-- Main theorem: If a ColoredNGon satisfies both conditions, then it has at most 2 colors -/
theorem colored_ngon_at_most_two_colors (R : ColoredNGon)
  (h1 : satisfies_condition1 R) (h2 : satisfies_condition2 R) :
  R.colors ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_colored_ngon_at_most_two_colors_l2874_287428


namespace NUMINAMATH_CALUDE_pattern_boundary_length_l2874_287446

theorem pattern_boundary_length (square_area : ℝ) (num_points : ℕ) : square_area = 144 ∧ num_points = 4 →
  ∃ (boundary_length : ℝ), boundary_length = 18 * Real.pi + 36 := by
  sorry

end NUMINAMATH_CALUDE_pattern_boundary_length_l2874_287446


namespace NUMINAMATH_CALUDE_candidate_vote_percentage_l2874_287488

/-- Calculates the percentage of valid votes a candidate received in an election. -/
theorem candidate_vote_percentage
  (total_votes : ℕ)
  (invalid_vote_percentage : ℚ)
  (candidate_valid_votes : ℕ)
  (h1 : total_votes = 560000)
  (h2 : invalid_vote_percentage = 15 / 100)
  (h3 : candidate_valid_votes = 380800) :
  (candidate_valid_votes : ℚ) / ((1 - invalid_vote_percentage) * total_votes) = 80 / 100 := by
sorry


end NUMINAMATH_CALUDE_candidate_vote_percentage_l2874_287488


namespace NUMINAMATH_CALUDE_not_perfect_square_123_ones_l2874_287469

def number_with_ones (n : ℕ) : ℕ :=
  (10^n - 1) * 10^n + 123

theorem not_perfect_square_123_ones :
  ∀ n : ℕ, ∃ k : ℕ, (number_with_ones n) ≠ k^2 := by
  sorry

end NUMINAMATH_CALUDE_not_perfect_square_123_ones_l2874_287469


namespace NUMINAMATH_CALUDE_multiplication_table_odd_fraction_l2874_287478

theorem multiplication_table_odd_fraction :
  let table_size : ℕ := 16
  let total_products : ℕ := table_size * table_size
  let odd_numbers : ℕ := (table_size + 1) / 2
  let odd_products : ℕ := odd_numbers * odd_numbers
  (odd_products : ℚ) / total_products = 1 / 4 := by
sorry

end NUMINAMATH_CALUDE_multiplication_table_odd_fraction_l2874_287478


namespace NUMINAMATH_CALUDE_jerome_contacts_l2874_287424

/-- Calculates the total number of contacts on Jerome's list --/
def total_contacts (classmates : ℕ) (family_members : ℕ) : ℕ :=
  classmates + (classmates / 2) + family_members

/-- Theorem stating that Jerome's contact list has 33 people --/
theorem jerome_contacts : total_contacts 20 3 = 33 := by
  sorry

end NUMINAMATH_CALUDE_jerome_contacts_l2874_287424


namespace NUMINAMATH_CALUDE_joels_age_when_dad_twice_as_old_l2874_287440

theorem joels_age_when_dad_twice_as_old (joel_current_age dad_current_age : ℕ) 
  (h1 : joel_current_age = 12) 
  (h2 : dad_current_age = 47) : 
  ∃ (years : ℕ), dad_current_age + years = 2 * (joel_current_age + years) ∧ 
                 joel_current_age + years = 35 := by
  sorry

end NUMINAMATH_CALUDE_joels_age_when_dad_twice_as_old_l2874_287440


namespace NUMINAMATH_CALUDE_max_value_quadratic_l2874_287438

theorem max_value_quadratic (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x^2 - 2*x*y + 3*y^2 = 10) : 
  ∃ (max_val : ℝ), (∀ (x' y' : ℝ), x' > 0 → y' > 0 → x'^2 - 2*x'*y' + 3*y'^2 = 10 
    → x'^2 + 2*x'*y' + 3*y'^2 ≤ max_val) ∧ max_val = 20 + 10 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_max_value_quadratic_l2874_287438


namespace NUMINAMATH_CALUDE_lisa_decorative_spoons_l2874_287455

/-- The number of children Lisa has -/
def num_children : ℕ := 4

/-- The number of baby spoons each child had -/
def baby_spoons_per_child : ℕ := 3

/-- The number of large spoons in the new cutlery set -/
def new_large_spoons : ℕ := 10

/-- The number of teaspoons in the new cutlery set -/
def new_teaspoons : ℕ := 15

/-- The total number of spoons Lisa has now -/
def total_spoons : ℕ := 39

/-- The number of decorative spoons Lisa created -/
def decorative_spoons : ℕ := total_spoons - (new_large_spoons + new_teaspoons) - (num_children * baby_spoons_per_child)

theorem lisa_decorative_spoons : decorative_spoons = 2 := by
  sorry

end NUMINAMATH_CALUDE_lisa_decorative_spoons_l2874_287455


namespace NUMINAMATH_CALUDE_division_result_l2874_287481

theorem division_result : (210 : ℚ) / (15 + 12 * 3 - 6) = 14 / 3 := by
  sorry

end NUMINAMATH_CALUDE_division_result_l2874_287481


namespace NUMINAMATH_CALUDE_identity_function_satisfies_conditions_l2874_287400

def is_identity_function (f : ℕ → ℕ) : Prop :=
  ∀ m : ℕ, f m = m

theorem identity_function_satisfies_conditions (f : ℕ → ℕ) 
  (h1 : ∀ m : ℕ, f m = 1 ↔ m = 1)
  (h2 : ∀ m n : ℕ, f (m * n) = f m * f n / f (Nat.gcd m n))
  (h3 : ∀ m : ℕ, (f^[2012]) m = m) :
  is_identity_function f :=
sorry

end NUMINAMATH_CALUDE_identity_function_satisfies_conditions_l2874_287400


namespace NUMINAMATH_CALUDE_rectangle_length_l2874_287434

theorem rectangle_length (w l : ℝ) (h1 : w > 0) (h2 : l > 0) : 
  (2*l + 2*w) / w = 5 → l * w = 150 → l = 15 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_length_l2874_287434


namespace NUMINAMATH_CALUDE_towel_loads_l2874_287408

theorem towel_loads (towels_per_load : ℕ) (total_towels : ℕ) (h1 : towels_per_load = 7) (h2 : total_towels = 42) :
  total_towels / towels_per_load = 6 := by
sorry

end NUMINAMATH_CALUDE_towel_loads_l2874_287408


namespace NUMINAMATH_CALUDE_age_difference_proof_l2874_287490

theorem age_difference_proof (hurley_age richard_age : ℕ) : 
  hurley_age = 14 →
  hurley_age + 40 + (richard_age + 40) = 128 →
  richard_age - hurley_age = 20 := by
sorry

end NUMINAMATH_CALUDE_age_difference_proof_l2874_287490


namespace NUMINAMATH_CALUDE_max_circumference_in_standard_parabola_l2874_287474

/-- Represents a circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a parabola in the form x^2 = 4y -/
def standardParabola : Set (ℝ × ℝ) :=
  {p | p.1^2 = 4 * p.2}

/-- Checks if a circle passes through the vertex of the standard parabola -/
def passesVertexStandardParabola (c : Circle) : Prop :=
  c.center.1^2 + c.center.2^2 = c.radius^2

/-- Checks if a circle is entirely inside the standard parabola -/
def insideStandardParabola (c : Circle) : Prop :=
  ∀ (x y : ℝ), (x - c.center.1)^2 + (y - c.center.2)^2 ≤ c.radius^2 → x^2 ≤ 4 * y

/-- The maximum circumference theorem -/
theorem max_circumference_in_standard_parabola :
  ∃ (c : Circle),
    passesVertexStandardParabola c ∧
    insideStandardParabola c ∧
    (∀ (c' : Circle),
      passesVertexStandardParabola c' ∧
      insideStandardParabola c' →
      2 * π * c'.radius ≤ 2 * π * c.radius) ∧
    2 * π * c.radius = 4 * π :=
sorry

end NUMINAMATH_CALUDE_max_circumference_in_standard_parabola_l2874_287474


namespace NUMINAMATH_CALUDE_floor_equation_solutions_l2874_287403

theorem floor_equation_solutions (x y : ℝ) :
  (∀ n : ℕ+, x * ⌊n * y⌋ = y * ⌊n * x⌋) ↔
  (x = y ∨ x = 0 ∨ y = 0 ∨ (∃ a b : ℤ, x = a ∧ y = b)) :=
sorry

end NUMINAMATH_CALUDE_floor_equation_solutions_l2874_287403


namespace NUMINAMATH_CALUDE_total_age_proof_l2874_287418

/-- Given three people a, b, and c, where a is two years older than b, 
    b is twice as old as c, and b is 20 years old, 
    prove that the total of their ages is 52 years. -/
theorem total_age_proof (a b c : ℕ) : 
  a = b + 2 → 
  b = 2 * c → 
  b = 20 → 
  a + b + c = 52 := by
  sorry

end NUMINAMATH_CALUDE_total_age_proof_l2874_287418


namespace NUMINAMATH_CALUDE_function_inequality_relation_l2874_287477

theorem function_inequality_relation (f : ℝ → ℝ) (a b : ℝ) :
  (∀ x, f x = 3 * x + 1) →
  a > 0 →
  b > 0 →
  (∀ x, |x - 1| < b → |f x - 4| < a) →
  a ≥ 3 * b :=
sorry

end NUMINAMATH_CALUDE_function_inequality_relation_l2874_287477


namespace NUMINAMATH_CALUDE_tan_product_simplification_l2874_287435

theorem tan_product_simplification :
  (1 + Real.tan (10 * π / 180)) * (1 + Real.tan (35 * π / 180)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_tan_product_simplification_l2874_287435


namespace NUMINAMATH_CALUDE_exists_nonparallel_quadrilateral_from_identical_triangles_l2874_287425

/-- A triangle in 2D space --/
structure Triangle :=
  (a b c : ℝ × ℝ)

/-- A quadrilateral in 2D space --/
structure Quadrilateral :=
  (a b c d : ℝ × ℝ)

/-- Check if two line segments are parallel --/
def are_parallel (p1 p2 q1 q2 : ℝ × ℝ) : Prop :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := q1
  let (x4, y4) := q2
  (x2 - x1) * (y4 - y3) = (y2 - y1) * (x4 - x3)

/-- Check if a quadrilateral has parallel sides --/
def has_parallel_sides (q : Quadrilateral) : Prop :=
  are_parallel q.a q.b q.c q.d ∨ are_parallel q.a q.d q.b q.c

/-- Check if a quadrilateral is convex --/
def is_convex (q : Quadrilateral) : Prop := sorry

/-- Function to construct a quadrilateral from four triangles --/
def construct_quadrilateral (t1 t2 t3 t4 : Triangle) : Quadrilateral := sorry

/-- Theorem: There exists a convex quadrilateral formed by four identical triangles that does not have parallel sides --/
theorem exists_nonparallel_quadrilateral_from_identical_triangles :
  ∃ (t : Triangle) (q : Quadrilateral),
    q = construct_quadrilateral t t t t ∧
    is_convex q ∧
    ¬has_parallel_sides q :=
sorry

end NUMINAMATH_CALUDE_exists_nonparallel_quadrilateral_from_identical_triangles_l2874_287425


namespace NUMINAMATH_CALUDE_cubic_coefficient_in_product_l2874_287423

/-- The coefficient of x^3 in the expansion of (3x^3 + 2x^2 + 4x + 5)(4x^3 + 3x^2 + 5x + 6) -/
def cubic_coefficient : ℤ := 40

/-- The first polynomial in the product -/
def polynomial1 (x : ℚ) : ℚ := 3 * x^3 + 2 * x^2 + 4 * x + 5

/-- The second polynomial in the product -/
def polynomial2 (x : ℚ) : ℚ := 4 * x^3 + 3 * x^2 + 5 * x + 6

/-- The theorem stating that the coefficient of x^3 in the expansion of the product of polynomial1 and polynomial2 is equal to cubic_coefficient -/
theorem cubic_coefficient_in_product : 
  ∃ (a b c d e f g : ℚ), 
    polynomial1 x * polynomial2 x = a * x^6 + b * x^5 + c * x^4 + cubic_coefficient * x^3 + d * x^2 + e * x + f :=
by sorry

end NUMINAMATH_CALUDE_cubic_coefficient_in_product_l2874_287423


namespace NUMINAMATH_CALUDE_rectangle_from_right_triangle_l2874_287426

theorem rectangle_from_right_triangle (a b c : ℝ) (h_right_triangle : a^2 + b^2 = c^2) 
  (h_positive : a > 0 ∧ b > 0) :
  ∃ x y : ℝ, 
    x + y = c ∧ 
    x * y = a * b / 2 ∧
    x = (c + a - b) / 2 ∧ 
    y = (c - a + b) / 2 := by
  sorry


end NUMINAMATH_CALUDE_rectangle_from_right_triangle_l2874_287426


namespace NUMINAMATH_CALUDE_valid_numbers_count_l2874_287461

def is_valid_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99 ∧
  100 ≤ n^2 ∧ n^2 < 1000 ∧
  100 ≤ (10 * (n % 10) + n / 10)^2 ∧ (10 * (n % 10) + n / 10)^2 < 1000 ∧
  n^2 = (10 * (n % 10) + n / 10)^2 % 10 * 100 + ((10 * (n % 10) + n / 10)^2 / 10 % 10) * 10 + (10 * (n % 10) + n / 10)^2 / 100

theorem valid_numbers_count :
  ∃ (S : Finset ℕ), S.card = 4 ∧ (∀ n, n ∈ S ↔ is_valid_number n) :=
sorry

end NUMINAMATH_CALUDE_valid_numbers_count_l2874_287461


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_lengths_l2874_287451

def IsoscelesTriangle (a b c : ℝ) : Prop :=
  (a = b ∧ a + b + c = 5 ∧ (a = 2 ∨ b = 2 ∨ c = 2)) ∧
  (a + b > c) ∧ (b + c > a) ∧ (c + a > b)

theorem isosceles_triangle_base_lengths :
  ∃ (a b c : ℝ), IsoscelesTriangle a b c ∧ (c = 1.5 ∨ c = 2) := by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_lengths_l2874_287451


namespace NUMINAMATH_CALUDE_trigonometric_simplification_max_value_cosine_function_l2874_287444

open Real

theorem trigonometric_simplification (α : ℝ) :
  (sin (2 * π - α) * tan (π - α) * cos (-π + α)) / (sin (5 * π + α) * sin (π / 2 + α)) = tan α :=
sorry

theorem max_value_cosine_function :
  let f : ℝ → ℝ := λ x ↦ 2 * cos x - cos (2 * x)
  ∃ (max_value : ℝ), max_value = 3 / 2 ∧
    ∀ x, f x ≤ max_value ∧
    ∀ k : ℤ, f (π / 3 + 2 * π * ↑k) = max_value ∧ f (-π / 3 + 2 * π * ↑k) = max_value :=
sorry

end NUMINAMATH_CALUDE_trigonometric_simplification_max_value_cosine_function_l2874_287444
