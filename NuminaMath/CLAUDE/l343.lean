import Mathlib

namespace NUMINAMATH_CALUDE_max_q_minus_r_for_1027_l343_34376

theorem max_q_minus_r_for_1027 :
  ∃ (q r : ℕ+), 1027 = 23 * q + r ∧ 
  ∀ (q' r' : ℕ+), 1027 = 23 * q' + r' → q' - r' ≤ q - r ∧ q - r = 29 := by
sorry

end NUMINAMATH_CALUDE_max_q_minus_r_for_1027_l343_34376


namespace NUMINAMATH_CALUDE_nested_circles_radius_l343_34332

theorem nested_circles_radius (A₁ A₂ : ℝ) : 
  A₁ > 0 → 
  A₂ > 0 → 
  (∃ d : ℝ, A₂ = A₁ + d ∧ A₁ + 2*A₂ = A₂ + d) → 
  A₁ + 2*A₂ = π * 5^2 → 
  ∃ r : ℝ, r > 0 ∧ A₁ = π * r^2 ∧ r = Real.sqrt 5 :=
sorry

end NUMINAMATH_CALUDE_nested_circles_radius_l343_34332


namespace NUMINAMATH_CALUDE_odd_function_property_l343_34369

/-- A function is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem odd_function_property (f : ℝ → ℝ) 
    (h_odd : IsOdd f) 
    (h_period : ∀ x, f (x + 2) = -f x) : 
  f (-2) = 0 := by
sorry

end NUMINAMATH_CALUDE_odd_function_property_l343_34369


namespace NUMINAMATH_CALUDE_book_sale_revenue_l343_34336

/-- Calculates the total amount received from book sales given the number of unsold books -/
def totalAmountReceived (unsoldBooks : ℕ) : ℚ :=
  let totalBooks := unsoldBooks * 3
  let soldBooks := totalBooks * 2 / 3
  soldBooks * 5

/-- Proves that the total amount received from book sales is $500 when 50 books remain unsold -/
theorem book_sale_revenue : totalAmountReceived 50 = 500 := by
  sorry

end NUMINAMATH_CALUDE_book_sale_revenue_l343_34336


namespace NUMINAMATH_CALUDE_number_relations_l343_34309

theorem number_relations : 
  (∃ n : ℤ, 28 = 4 * n) ∧ 
  (∃ n : ℤ, 361 = 19 * n) ∧ 
  (∀ n : ℤ, 63 ≠ 19 * n) ∧ 
  (∃ n : ℤ, 45 = 15 * n) ∧ 
  (∃ n : ℤ, 30 = 15 * n) ∧ 
  (∃ n : ℤ, 144 = 12 * n) := by
sorry

end NUMINAMATH_CALUDE_number_relations_l343_34309


namespace NUMINAMATH_CALUDE_equation_solution_l343_34343

theorem equation_solution :
  ∃! x : ℝ, (9 - 3*x) * (3^x) - (x - 2) * (x^2 - 5*x + 6) = 0 ∧ x = 3 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l343_34343


namespace NUMINAMATH_CALUDE_wilson_theorem_plus_one_l343_34315

theorem wilson_theorem_plus_one (p : Nat) (hp : Nat.Prime p) (hodd : p % 2 = 1) :
  (p - 1).factorial + 1 ∣ p := by
  sorry

end NUMINAMATH_CALUDE_wilson_theorem_plus_one_l343_34315


namespace NUMINAMATH_CALUDE_proposition_p_and_q_true_l343_34382

theorem proposition_p_and_q_true : 
  (∃ α : ℝ, Real.cos (π - α) = Real.cos α) ∧ (∀ x : ℝ, x^2 + 1 > 0) := by
  sorry

end NUMINAMATH_CALUDE_proposition_p_and_q_true_l343_34382


namespace NUMINAMATH_CALUDE_franks_breakfast_cost_l343_34373

/-- The cost of Frank's breakfast shopping -/
def breakfast_cost (bun_price : ℚ) (bun_quantity : ℕ) (milk_price : ℚ) (milk_quantity : ℕ) (egg_price_multiplier : ℕ) : ℚ :=
  bun_price * bun_quantity + milk_price * milk_quantity + (milk_price * egg_price_multiplier)

/-- Theorem stating that Frank's breakfast shopping costs $11 -/
theorem franks_breakfast_cost :
  breakfast_cost 0.1 10 2 2 3 = 11 := by
  sorry

end NUMINAMATH_CALUDE_franks_breakfast_cost_l343_34373


namespace NUMINAMATH_CALUDE_exactly_one_girl_and_exactly_two_girls_mutually_exclusive_but_not_contrary_l343_34379

/-- Represents the possible outcomes when selecting 2 students from a group of 2 boys and 2 girls -/
inductive Outcome
  | TwoBoys
  | OneGirlOneBoy
  | TwoGirls

/-- The sample space of all possible outcomes -/
def SampleSpace : Set Outcome := {Outcome.TwoBoys, Outcome.OneGirlOneBoy, Outcome.TwoGirls}

/-- The event "Exactly 1 girl" -/
def ExactlyOneGirl : Set Outcome := {Outcome.OneGirlOneBoy}

/-- The event "Exactly 2 girls" -/
def ExactlyTwoGirls : Set Outcome := {Outcome.TwoGirls}

/-- Theorem stating that "Exactly 1 girl" and "Exactly 2 girls" are mutually exclusive but not contrary -/
theorem exactly_one_girl_and_exactly_two_girls_mutually_exclusive_but_not_contrary :
  (ExactlyOneGirl ∩ ExactlyTwoGirls = ∅) ∧
  (ExactlyOneGirl ∪ ExactlyTwoGirls ≠ SampleSpace) := by
  sorry

end NUMINAMATH_CALUDE_exactly_one_girl_and_exactly_two_girls_mutually_exclusive_but_not_contrary_l343_34379


namespace NUMINAMATH_CALUDE_sequence_equality_l343_34344

/-- Given two sequences of real numbers (xₙ) and (yₙ) defined as follows:
    x₁ = y₁ = 1
    xₙ₊₁ = (xₙ + 2) / (xₙ + 1)
    yₙ₊₁ = (yₙ² + 2) / (2yₙ)
    Prove that yₙ₊₁ = x₂ⁿ holds for n = 0, 1, 2, ... -/
theorem sequence_equality (x y : ℕ → ℝ) 
    (hx1 : x 1 = 1)
    (hy1 : y 1 = 1)
    (hx : ∀ n : ℕ, x (n + 1) = (x n + 2) / (x n + 1))
    (hy : ∀ n : ℕ, y (n + 1) = (y n ^ 2 + 2) / (2 * y n)) :
  ∀ n : ℕ, y (n + 1) = x (2 ^ n) := by
  sorry


end NUMINAMATH_CALUDE_sequence_equality_l343_34344


namespace NUMINAMATH_CALUDE_problem_solution_l343_34325

theorem problem_solution : 
  (Real.sqrt (7^2 + 24^2)) / (Real.sqrt (49 + 16)) = (5 * Real.sqrt 65) / 13 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l343_34325


namespace NUMINAMATH_CALUDE_otimes_properties_l343_34374

-- Define the operation
def otimes (a b : ℝ) : ℝ := a * (1 - b)

-- Statement of the theorem
theorem otimes_properties :
  (otimes 2 (-2) = 6) ∧ 
  (∃ a b : ℝ, otimes a b ≠ otimes b a) ∧
  (∀ a b : ℝ, a + b = 0 → otimes a a + otimes b b = 2 * a * b) ∧
  (∃ a b : ℝ, otimes a b = 0 ∧ a ≠ 0) := by
  sorry


end NUMINAMATH_CALUDE_otimes_properties_l343_34374


namespace NUMINAMATH_CALUDE_fruit_arrangements_l343_34393

/-- The number of distinct arrangements of 9 items, where there are 4 indistinguishable items of type A, 3 indistinguishable items of type B, and 2 indistinguishable items of type C. -/
def distinct_arrangements (total : Nat) (a : Nat) (b : Nat) (c : Nat) : Nat :=
  Nat.factorial total / (Nat.factorial a * Nat.factorial b * Nat.factorial c)

/-- Theorem stating that the number of distinct arrangements of 9 items, 
    where there are 4 indistinguishable items of type A, 
    3 indistinguishable items of type B, and 2 indistinguishable items of type C, 
    is equal to 1260. -/
theorem fruit_arrangements : distinct_arrangements 9 4 3 2 = 1260 := by
  sorry

end NUMINAMATH_CALUDE_fruit_arrangements_l343_34393


namespace NUMINAMATH_CALUDE_divisibility_17_and_289_l343_34307

theorem divisibility_17_and_289 (n : ℤ) :
  (∃ k : ℤ, n^2 - n - 4 = 17 * k) ↔ (∃ m : ℤ, n = 17 * m - 8) ∧
  ¬(∃ l : ℤ, n^2 - n - 4 = 289 * l) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_17_and_289_l343_34307


namespace NUMINAMATH_CALUDE_books_from_first_shop_l343_34395

theorem books_from_first_shop (total_spent : ℕ) (second_shop_books : ℕ) (avg_price : ℕ) :
  total_spent = 768 →
  second_shop_books = 22 →
  avg_price = 12 →
  ∃ first_shop_books : ℕ,
    first_shop_books = 42 ∧
    total_spent = avg_price * (first_shop_books + second_shop_books) :=
by
  sorry

end NUMINAMATH_CALUDE_books_from_first_shop_l343_34395


namespace NUMINAMATH_CALUDE_range_of_m_l343_34394

-- Define set A
def A : Set ℝ := {y | ∃ x > 0, y = 1 / x}

-- Define set B
def B : Set ℝ := {x | ∃ y, y = Real.log (2 * x - 4)}

-- Theorem statement
theorem range_of_m (m : ℝ) (h1 : m ∈ A) (h2 : m ∉ B) : m ∈ Set.Ioo 0 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l343_34394


namespace NUMINAMATH_CALUDE_second_car_speed_l343_34342

/-- Given two cars on a circular track, prove the speed of the second car. -/
theorem second_car_speed
  (track_length : ℝ)
  (first_car_speed : ℝ)
  (total_time : ℝ)
  (h1 : track_length = 150)
  (h2 : first_car_speed = 60)
  (h3 : total_time = 2)
  (h4 : ∃ (second_car_speed : ℝ),
    (first_car_speed + second_car_speed) * total_time = 2 * track_length) :
  ∃ (second_car_speed : ℝ), second_car_speed = 90 :=
by
  sorry


end NUMINAMATH_CALUDE_second_car_speed_l343_34342


namespace NUMINAMATH_CALUDE_point_with_distance_6_l343_34339

def distance_from_origin (x : ℝ) : ℝ := |x|

theorem point_with_distance_6 (A : ℝ) :
  distance_from_origin A = 6 ↔ A = 6 ∨ A = -6 := by
  sorry

end NUMINAMATH_CALUDE_point_with_distance_6_l343_34339


namespace NUMINAMATH_CALUDE_sum_using_splitting_terms_l343_34348

/-- The sum of (-2017⅔) + 2016¾ + (-2015⅚) + 16½ using the method of splitting terms -/
theorem sum_using_splitting_terms :
  (-2017 - 2/3) + (2016 + 3/4) + (-2015 - 5/6) + (16 + 1/2) = -2000 - 1/4 := by
  sorry

end NUMINAMATH_CALUDE_sum_using_splitting_terms_l343_34348


namespace NUMINAMATH_CALUDE_grid_shading_theorem_l343_34334

/-- Represents a square on the grid -/
structure Square where
  row : Fin 6
  col : Fin 6

/-- Determines if a square is shaded based on its position -/
def is_shaded (s : Square) : Prop :=
  (s.row % 2 = 0 ∧ s.col % 2 = 1) ∨ (s.row % 2 = 1 ∧ s.col % 2 = 0)

/-- The total number of squares in the grid -/
def total_squares : Nat := 36

/-- The number of shaded squares in the grid -/
def shaded_squares : Nat := 21

/-- The fraction of shaded squares in the grid -/
def shaded_fraction : Rat := 7 / 12

theorem grid_shading_theorem :
  (shaded_squares : Rat) / total_squares = shaded_fraction := by
  sorry

end NUMINAMATH_CALUDE_grid_shading_theorem_l343_34334


namespace NUMINAMATH_CALUDE_total_pears_picked_l343_34384

/-- Represents a person who picks pears -/
structure PearPicker where
  name : String
  morning : Bool

/-- Calculates the number of pears picked on Day 2 -/
def day2Amount (day1 : ℕ) (morning : Bool) : ℕ :=
  if morning then day1 / 2 else day1 * 2

/-- Calculates the number of pears picked on Day 3 -/
def day3Amount (day1 day2 : ℕ) : ℕ :=
  (day1 + day2 + 1) / 2  -- Adding 1 for rounding up

/-- Calculates the total pears picked by a person over three days -/
def totalPears (day1 : ℕ) (morning : Bool) : ℕ :=
  let day2 := day2Amount day1 morning
  let day3 := day3Amount day1 day2
  day1 + day2 + day3

/-- The main theorem stating the total number of pears picked -/
theorem total_pears_picked : 
  let jason := PearPicker.mk "Jason" true
  let keith := PearPicker.mk "Keith" true
  let mike := PearPicker.mk "Mike" true
  let alicia := PearPicker.mk "Alicia" false
  let tina := PearPicker.mk "Tina" false
  let nicola := PearPicker.mk "Nicola" false
  totalPears 46 jason.morning +
  totalPears 47 keith.morning +
  totalPears 12 mike.morning +
  totalPears 28 alicia.morning +
  totalPears 33 tina.morning +
  totalPears 52 nicola.morning = 747 := by
  sorry

end NUMINAMATH_CALUDE_total_pears_picked_l343_34384


namespace NUMINAMATH_CALUDE_function_inequality_implies_upper_bound_l343_34381

theorem function_inequality_implies_upper_bound (a : ℝ) : 
  (∀ x₁ ∈ Set.Icc (1/2 : ℝ) 3, ∃ x₂ ∈ Set.Icc (2 : ℝ) 3, 
    x₁ + 4/x₁ ≥ 2^x₂ + a) → a ≤ 0 := by sorry

end NUMINAMATH_CALUDE_function_inequality_implies_upper_bound_l343_34381


namespace NUMINAMATH_CALUDE_quadratic_minimum_value_l343_34351

/-- Given a quadratic function f(x) = ax^2 + bx + c that is always non-negative
    and a < b, prove that (3a-2b+c)/(b-a) ≥ 1 -/
theorem quadratic_minimum_value (a b c : ℝ) 
    (h1 : ∀ x : ℝ, a * x^2 + b * x + c ≥ 0)
    (h2 : a < b) : 
    (3*a - 2*b + c) / (b - a) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_minimum_value_l343_34351


namespace NUMINAMATH_CALUDE_original_price_l343_34301

theorem original_price (p q : ℝ) (h : p ≠ 0 ∧ q ≠ 0) :
  let x := (20000 : ℝ) / (10000^2 - (p^2 + q^2) * 10000 + p^2 * q^2)
  let final_price := x * (1 + p/100) * (1 + q/100) * (1 - q/100) * (1 - p/100)
  final_price = 2 := by sorry

end NUMINAMATH_CALUDE_original_price_l343_34301


namespace NUMINAMATH_CALUDE_correct_answer_l343_34300

theorem correct_answer (x : ℝ) (h : 2 * x = 60) : x / 2 = 15 := by
  sorry

end NUMINAMATH_CALUDE_correct_answer_l343_34300


namespace NUMINAMATH_CALUDE_f_negative_a_l343_34356

noncomputable def f (x : ℝ) : ℝ := 1 + Real.tan x

theorem f_negative_a (a : ℝ) (h : f a = 3) : f (-a) = -1 := by
  sorry

end NUMINAMATH_CALUDE_f_negative_a_l343_34356


namespace NUMINAMATH_CALUDE_prob_same_group_l343_34314

/-- The number of interest groups -/
def num_groups : ℕ := 3

/-- The probability of a student joining any specific group -/
def prob_join_group : ℚ := 1 / num_groups

/-- The total number of possible outcomes for two students joining groups -/
def total_outcomes : ℕ := num_groups * num_groups

/-- The number of outcomes where both students join the same group -/
def same_group_outcomes : ℕ := num_groups

theorem prob_same_group :
  (same_group_outcomes : ℚ) / total_outcomes = 1 / 3 :=
sorry

end NUMINAMATH_CALUDE_prob_same_group_l343_34314


namespace NUMINAMATH_CALUDE_min_attempts_eq_n_l343_34345

/-- Represents a binary code of length n -/
def BinaryCode (n : ℕ) := Fin n → Bool

/-- Feedback from an attempt -/
inductive Feedback
| NoClick
| Click
| Open

/-- Function representing an attempt to open the safe -/
def attempt (n : ℕ) (secretCode : BinaryCode n) (tryCode : BinaryCode n) : Feedback :=
  sorry

/-- Minimum number of attempts required to open the safe -/
def minAttempts (n : ℕ) : ℕ :=
  sorry

/-- Theorem stating that the minimum number of attempts is n -/
theorem min_attempts_eq_n (n : ℕ) : minAttempts n = n :=
  sorry

end NUMINAMATH_CALUDE_min_attempts_eq_n_l343_34345


namespace NUMINAMATH_CALUDE_number_division_problem_l343_34370

theorem number_division_problem (x : ℚ) : (x / 5 = 75 + x / 6) → x = 2250 := by
  sorry

end NUMINAMATH_CALUDE_number_division_problem_l343_34370


namespace NUMINAMATH_CALUDE_sin_equality_proof_l343_34308

theorem sin_equality_proof (n : ℤ) : 
  -90 ≤ n ∧ n ≤ 90 ∧ Real.sin (n * π / 180) = Real.sin (782 * π / 180) → 
  n = 62 ∨ n = -62 := by
  sorry

end NUMINAMATH_CALUDE_sin_equality_proof_l343_34308


namespace NUMINAMATH_CALUDE_discount_card_problem_l343_34333

/-- Proves that given a discount card that costs 20 yuan and provides a 20% discount,
    if a customer saves 12 yuan by using the card, then the original price of the purchase
    before the discount was 160 yuan. -/
theorem discount_card_problem (card_cost discount_rate savings original_price : ℝ)
    (h1 : card_cost = 20)
    (h2 : discount_rate = 0.2)
    (h3 : savings = 12)
    (h4 : card_cost + (1 - discount_rate) * original_price = original_price - savings) :
    original_price = 160 :=
  sorry

end NUMINAMATH_CALUDE_discount_card_problem_l343_34333


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l343_34337

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_product (a : ℕ → ℝ) :
  geometric_sequence a →
  a 2 * a 4 = 16 →
  a 1 * a 3 * a 5 = 64 ∨ a 1 * a 3 * a 5 = -64 :=
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l343_34337


namespace NUMINAMATH_CALUDE_qin_jiushao_v3_l343_34326

/-- Qin Jiushao algorithm for polynomial evaluation -/
def qin_jiushao (f : ℤ → ℤ) (x : ℤ) : ℕ → ℤ
| 0 => 1
| 1 => qin_jiushao f x 0 * x + 47
| 2 => qin_jiushao f x 1 * x + 0
| 3 => qin_jiushao f x 2 * x - 37
| _ => 0

/-- The polynomial f(x) = x^5 + 47x^4 - 37x^2 + 1 -/
def f (x : ℤ) : ℤ := x^5 + 47*x^4 - 37*x^2 + 1

theorem qin_jiushao_v3 : qin_jiushao f (-1) 3 = 9 := by sorry

end NUMINAMATH_CALUDE_qin_jiushao_v3_l343_34326


namespace NUMINAMATH_CALUDE_rachel_earnings_l343_34392

/-- Rachel's earnings as a waitress in one hour -/
theorem rachel_earnings (hourly_wage : ℝ) (people_served : ℕ) (tip_per_person : ℝ) 
  (h1 : hourly_wage = 12)
  (h2 : people_served = 20)
  (h3 : tip_per_person = 1.25) :
  hourly_wage + (people_served : ℝ) * tip_per_person = 37 := by
  sorry

end NUMINAMATH_CALUDE_rachel_earnings_l343_34392


namespace NUMINAMATH_CALUDE_fraction_simplification_l343_34363

theorem fraction_simplification :
  1 / (1 / (1/2)^1 + 1 / (1/2)^2 + 1 / (1/2)^3) = 1 / 14 := by sorry

end NUMINAMATH_CALUDE_fraction_simplification_l343_34363


namespace NUMINAMATH_CALUDE_binomial_coeff_not_arithmetic_progression_l343_34310

theorem binomial_coeff_not_arithmetic_progression (n k : ℕ) (h1 : k ≤ n - 3) :
  ¬∃ (a d : ℤ), 
    (Nat.choose n k : ℤ) = a ∧
    (Nat.choose n (k + 1) : ℤ) = a + d ∧
    (Nat.choose n (k + 2) : ℤ) = a + 2*d ∧
    (Nat.choose n (k + 3) : ℤ) = a + 3*d :=
by sorry

end NUMINAMATH_CALUDE_binomial_coeff_not_arithmetic_progression_l343_34310


namespace NUMINAMATH_CALUDE_correct_relative_pronoun_l343_34341

/-- Represents a relative pronoun -/
inductive RelativePronoun
| When
| That
| Where
| Which

/-- Represents the context of an opportunity -/
structure OpportunityContext where
  universal : Bool
  independentOfAge : Bool
  independentOfProfession : Bool
  independentOfReligion : Bool
  independentOfBackground : Bool

/-- Represents the function of a relative pronoun in a sentence -/
structure PronounFunction where
  modifiesNoun : Bool
  describesCircumstances : Bool
  introducesAdjectiveClause : Bool

/-- Determines if a relative pronoun is correct for the given sentence -/
def isCorrectPronoun (pronoun : RelativePronoun) (context : OpportunityContext) (function : PronounFunction) : Prop :=
  context.universal ∧
  context.independentOfAge ∧
  context.independentOfProfession ∧
  context.independentOfReligion ∧
  context.independentOfBackground ∧
  function.modifiesNoun ∧
  function.describesCircumstances ∧
  function.introducesAdjectiveClause ∧
  pronoun = RelativePronoun.Where

theorem correct_relative_pronoun (context : OpportunityContext) (function : PronounFunction) :
  isCorrectPronoun RelativePronoun.Where context function :=
by sorry

end NUMINAMATH_CALUDE_correct_relative_pronoun_l343_34341


namespace NUMINAMATH_CALUDE_fourth_quadrant_condition_l343_34350

def complex_number (b : ℝ) : ℂ := (1 + b * Complex.I) * (2 + Complex.I)

def in_fourth_quadrant (z : ℂ) : Prop := z.re > 0 ∧ z.im < 0

theorem fourth_quadrant_condition (b : ℝ) : 
  in_fourth_quadrant (complex_number b) ↔ b < -1/2 := by sorry

end NUMINAMATH_CALUDE_fourth_quadrant_condition_l343_34350


namespace NUMINAMATH_CALUDE_shaded_area_theorem_l343_34330

theorem shaded_area_theorem (total_area : ℝ) (total_triangles : ℕ) (shaded_triangles : ℕ) : 
  total_area = 64 → 
  total_triangles = 64 → 
  shaded_triangles = 28 → 
  (shaded_triangles : ℝ) * (total_area / total_triangles) = 28 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_theorem_l343_34330


namespace NUMINAMATH_CALUDE_lewis_earnings_theorem_l343_34386

/-- Calculates Lewis's earnings per week without overtime during harvest season. -/
def lewis_earnings_without_overtime (weeks : ℕ) (overtime_pay : ℚ) (total_earnings : ℚ) : ℚ :=
  let total_overtime := overtime_pay * weeks
  let earnings_without_overtime := total_earnings - total_overtime
  earnings_without_overtime / weeks

/-- Proves that Lewis's earnings per week without overtime is approximately $27.61. -/
theorem lewis_earnings_theorem (weeks : ℕ) (overtime_pay : ℚ) (total_earnings : ℚ)
    (h1 : weeks = 1091)
    (h2 : overtime_pay = 939)
    (h3 : total_earnings = 1054997) :
    ∃ (ε : ℚ), ε > 0 ∧ ε < 1 ∧ 
    |lewis_earnings_without_overtime weeks overtime_pay total_earnings - 27.61| < ε :=
  sorry

end NUMINAMATH_CALUDE_lewis_earnings_theorem_l343_34386


namespace NUMINAMATH_CALUDE_fraction_addition_l343_34317

theorem fraction_addition (a : ℝ) (ha : a ≠ 0) : 3 / a + 2 / a = 5 / a := by
  sorry

end NUMINAMATH_CALUDE_fraction_addition_l343_34317


namespace NUMINAMATH_CALUDE_prime_sum_2019_power_l343_34396

theorem prime_sum_2019_power (p q : ℕ) : 
  Prime p → Prime q → p + q = 2019 → (p - 1)^(q - 1) = 1 ∨ (p - 1)^(q - 1) = 2016 := by
  sorry

end NUMINAMATH_CALUDE_prime_sum_2019_power_l343_34396


namespace NUMINAMATH_CALUDE_not_multiple_of_121_l343_34346

theorem not_multiple_of_121 (n : ℤ) : ¬ ∃ k : ℤ, n^2 + 2*n + 12 = 121*k := by
  sorry

end NUMINAMATH_CALUDE_not_multiple_of_121_l343_34346


namespace NUMINAMATH_CALUDE_sams_remaining_seashells_l343_34340

-- Define the initial number of seashells Sam found
def initial_seashells : ℕ := 35

-- Define the number of seashells Sam gave to Joan
def given_away : ℕ := 18

-- Theorem stating how many seashells Sam has now
theorem sams_remaining_seashells : 
  initial_seashells - given_away = 17 := by sorry

end NUMINAMATH_CALUDE_sams_remaining_seashells_l343_34340


namespace NUMINAMATH_CALUDE_mary_paper_problem_l343_34347

/-- Represents the initial state of Mary's paper pieces -/
structure InitialState where
  squares : ℕ
  triangles : ℕ
  total_pieces : ℕ
  total_pieces_eq : squares + triangles = total_pieces

/-- Represents the final state after cutting some squares -/
structure FinalState where
  initial : InitialState
  squares_cut : ℕ
  total_vertices : ℕ
  squares_cut_constraint : squares_cut ≤ initial.squares

theorem mary_paper_problem (state : InitialState) (final : FinalState)
  (h_initial_pieces : state.total_pieces = 10)
  (h_squares_cut : final.squares_cut = 3)
  (h_final_pieces : state.total_pieces + final.squares_cut = 13)
  (h_final_vertices : final.total_vertices = 42)
  : state.triangles = 4 := by
  sorry

end NUMINAMATH_CALUDE_mary_paper_problem_l343_34347


namespace NUMINAMATH_CALUDE_rearrangement_theorem_l343_34322

/-- The number of terms in the expansion of ((x^(1/2) + x^(1/3))^12) -/
def total_terms : ℕ := 13

/-- The number of terms with positive integer powers of x in the expansion -/
def integer_power_terms : ℕ := 3

/-- The number of terms without positive integer powers of x in the expansion -/
def non_integer_power_terms : ℕ := total_terms - integer_power_terms

/-- The number of ways to rearrange the terms in the expansion of ((x^(1/2) + x^(1/3))^12)
    so that the terms containing positive integer powers of x are not adjacent to each other -/
def rearrangement_count : ℕ := (Nat.factorial non_integer_power_terms) * (Nat.factorial (non_integer_power_terms + 1) / (Nat.factorial (non_integer_power_terms - 2)))

theorem rearrangement_theorem : 
  rearrangement_count = (Nat.factorial 10) * (Nat.factorial 11 / (Nat.factorial 8)) :=
sorry

end NUMINAMATH_CALUDE_rearrangement_theorem_l343_34322


namespace NUMINAMATH_CALUDE_smallest_sum_B_plus_c_l343_34357

theorem smallest_sum_B_plus_c : ∃ (B c : ℕ),
  (0 ≤ B ∧ B ≤ 4) ∧
  (c > 6) ∧
  (31 * B = 4 * (c + 1)) ∧
  (∀ (B' c' : ℕ), (0 ≤ B' ∧ B' ≤ 4) ∧ (c' > 6) ∧ (31 * B' = 4 * (c' + 1)) → B + c ≤ B' + c') ∧
  B + c = 34 :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_B_plus_c_l343_34357


namespace NUMINAMATH_CALUDE_penny_frog_count_l343_34371

/-- The number of tree frogs Penny counted -/
def tree_frogs : ℕ := 55

/-- The number of poison frogs Penny counted -/
def poison_frogs : ℕ := 10

/-- The number of wood frogs Penny counted -/
def wood_frogs : ℕ := 13

/-- The total number of frogs Penny counted -/
def total_frogs : ℕ := tree_frogs + poison_frogs + wood_frogs

theorem penny_frog_count : total_frogs = 78 := by
  sorry

end NUMINAMATH_CALUDE_penny_frog_count_l343_34371


namespace NUMINAMATH_CALUDE_car_travel_distance_l343_34329

/-- Proves that a car can travel 500 miles before refilling given specific journey conditions. -/
theorem car_travel_distance (fuel_cost : ℝ) (journey_distance : ℝ) (food_ratio : ℝ) (total_spent : ℝ)
  (h1 : fuel_cost = 45)
  (h2 : journey_distance = 2000)
  (h3 : food_ratio = 3/5)
  (h4 : total_spent = 288) :
  journey_distance / (total_spent / ((1 + food_ratio) * fuel_cost)) = 500 := by
  sorry

end NUMINAMATH_CALUDE_car_travel_distance_l343_34329


namespace NUMINAMATH_CALUDE_compute_expression_l343_34359

theorem compute_expression : 6^3 - 4*5 + 2^4 = 212 := by
  sorry

end NUMINAMATH_CALUDE_compute_expression_l343_34359


namespace NUMINAMATH_CALUDE_third_term_is_27_l343_34312

/-- A geometric sequence with six terms where the fifth term is 81 and the sixth term is 243 -/
def geometric_sequence (a b c d : ℝ) : Prop :=
  ∃ (r : ℝ), r ≠ 0 ∧ 
  b = a * r ∧
  c = b * r ∧
  d = c * r ∧
  81 = d * r ∧
  243 = 81 * r

/-- The third term of the geometric sequence a, b, c, d, 81, 243 is 27 -/
theorem third_term_is_27 (a b c d : ℝ) (h : geometric_sequence a b c d) : c = 27 := by
  sorry

end NUMINAMATH_CALUDE_third_term_is_27_l343_34312


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l343_34397

theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) :
  (∀ n : ℕ, a n > 0) →  -- positive terms
  (∀ n : ℕ, a (n + 1) = a n * q) →  -- geometric sequence
  a 1 = 3 →  -- first term
  a 1 + a 2 + a 3 = 21 →  -- sum of first three terms
  a 3 + a 4 + a 5 = 84 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l343_34397


namespace NUMINAMATH_CALUDE_f_has_root_in_interval_l343_34354

def f (x : ℝ) := x^3 - 3*x - 3

theorem f_has_root_in_interval :
  ∃ (x : ℝ), x ∈ Set.Ioo 2 3 ∧ f x = 0 := by
  sorry

end NUMINAMATH_CALUDE_f_has_root_in_interval_l343_34354


namespace NUMINAMATH_CALUDE_back_squat_increase_calculation_l343_34391

/-- Represents the increase in John's back squat in kg -/
def back_squat_increase : ℝ := sorry

/-- John's original back squat weight in kg -/
def original_back_squat : ℝ := 200

/-- The ratio of John's front squat to his back squat -/
def front_squat_ratio : ℝ := 0.8

/-- The ratio of a triple to John's front squat -/
def triple_ratio : ℝ := 0.9

/-- The total weight moved in three triples in kg -/
def total_triple_weight : ℝ := 540

theorem back_squat_increase_calculation :
  3 * (triple_ratio * front_squat_ratio * (original_back_squat + back_squat_increase)) = total_triple_weight ∧
  back_squat_increase = 50 := by sorry

end NUMINAMATH_CALUDE_back_squat_increase_calculation_l343_34391


namespace NUMINAMATH_CALUDE_solution_satisfies_system_l343_34316

theorem solution_satisfies_system :
  let x : ℚ := 130/161
  let y : ℚ := 76/23
  let z : ℚ := 3
  (7 * x - 3 * y + 2 * z = 4) ∧
  (4 * y - x - 5 * z = -3) ∧
  (3 * x + 2 * y - z = 7) := by
  sorry

end NUMINAMATH_CALUDE_solution_satisfies_system_l343_34316


namespace NUMINAMATH_CALUDE_contrapositive_odd_sum_even_l343_34331

def is_odd (n : ℤ) : Prop := ∃ k, n = 2*k + 1

def is_even (n : ℤ) : Prop := ∃ k, n = 2*k

theorem contrapositive_odd_sum_even :
  (∀ a b : ℤ, (is_odd a ∧ is_odd b) → is_even (a + b)) ↔
  (∀ a b : ℤ, ¬is_even (a + b) → ¬(is_odd a ∧ is_odd b)) :=
sorry

end NUMINAMATH_CALUDE_contrapositive_odd_sum_even_l343_34331


namespace NUMINAMATH_CALUDE_scientific_notation_of_9560000_l343_34362

theorem scientific_notation_of_9560000 :
  9560000 = 9.56 * (10 : ℝ) ^ 6 :=
by sorry

end NUMINAMATH_CALUDE_scientific_notation_of_9560000_l343_34362


namespace NUMINAMATH_CALUDE_largest_divisor_of_n_squared_div_360_l343_34355

theorem largest_divisor_of_n_squared_div_360 (n : ℕ+) (h : 360 ∣ n^2) :
  ∃ (t : ℕ), t = 60 ∧ t ∣ n ∧ ∀ (k : ℕ), k ∣ n → k ≤ t :=
sorry

end NUMINAMATH_CALUDE_largest_divisor_of_n_squared_div_360_l343_34355


namespace NUMINAMATH_CALUDE_remainder_of_repeating_number_l343_34368

def repeating_number (n : ℕ) : ℕ := 
  (12 : ℕ) * ((100 ^ n - 1) / 99)

theorem remainder_of_repeating_number : 
  repeating_number 150 % 99 = 18 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_repeating_number_l343_34368


namespace NUMINAMATH_CALUDE_determine_dracula_status_l343_34360

/-- Represents the types of Transylvanians -/
inductive TransylvanianType
| Truthful
| Liar

/-- Represents the possible answers to a yes/no question -/
inductive Answer
| Yes
| No

/-- Represents Dracula's status -/
inductive DraculaStatus
| Alive
| NotAlive

/-- A Transylvanian's response to the question -/
def response (t : TransylvanianType) (d : DraculaStatus) : Answer :=
  match t, d with
  | TransylvanianType.Truthful, DraculaStatus.Alive => Answer.Yes
  | TransylvanianType.Truthful, DraculaStatus.NotAlive => Answer.No
  | TransylvanianType.Liar, DraculaStatus.Alive => Answer.Yes
  | TransylvanianType.Liar, DraculaStatus.NotAlive => Answer.No

/-- The main theorem: The question can determine Dracula's status -/
theorem determine_dracula_status :
  ∀ (t : TransylvanianType) (d : DraculaStatus),
    response t d = Answer.Yes ↔ d = DraculaStatus.Alive :=
by sorry

end NUMINAMATH_CALUDE_determine_dracula_status_l343_34360


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l343_34323

-- Define the sets A and B
def A : Set ℝ := {x | -Real.sqrt 2 ≤ x ∧ x ≤ Real.sqrt 2}
def B : Set ℝ := {x | -1 < x ∧ x < 2}

-- Define the theorem
theorem intersection_A_complement_B :
  A ∩ (Set.univ \ B) = {x | -Real.sqrt 2 ≤ x ∧ x ≤ -1} := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l343_34323


namespace NUMINAMATH_CALUDE_water_one_eighth_after_three_pourings_l343_34375

def water_remaining (n : ℕ) : ℚ :=
  (1 : ℚ) / 2^n

theorem water_one_eighth_after_three_pourings :
  water_remaining 3 = (1 : ℚ) / 8 := by
  sorry

#check water_one_eighth_after_three_pourings

end NUMINAMATH_CALUDE_water_one_eighth_after_three_pourings_l343_34375


namespace NUMINAMATH_CALUDE_toms_total_cost_is_48_l343_34399

/-- Represents the fruit prices and quantities --/
structure FruitPurchase where
  lemon_price : ℝ
  papaya_price : ℝ
  mango_price : ℝ
  orange_price : ℝ
  apple_price : ℝ
  pineapple_price : ℝ
  lemon_qty : ℕ
  papaya_qty : ℕ
  mango_qty : ℕ
  orange_qty : ℕ
  apple_qty : ℕ
  pineapple_qty : ℕ

/-- Calculates the total cost after all discounts --/
def totalCostAfterDiscounts (purchase : FruitPurchase) (customer_number : ℕ) : ℝ :=
  sorry

/-- Theorem stating that Tom's total cost after all discounts is $48 --/
theorem toms_total_cost_is_48 :
  let purchase : FruitPurchase := {
    lemon_price := 2,
    papaya_price := 1,
    mango_price := 4,
    orange_price := 3,
    apple_price := 1.5,
    pineapple_price := 5,
    lemon_qty := 8,
    papaya_qty := 6,
    mango_qty := 5,
    orange_qty := 3,
    apple_qty := 8,
    pineapple_qty := 2
  }
  totalCostAfterDiscounts purchase 7 = 48 := by sorry

end NUMINAMATH_CALUDE_toms_total_cost_is_48_l343_34399


namespace NUMINAMATH_CALUDE_marble_prism_weight_l343_34319

/-- Represents the properties of a rectangular prism -/
structure RectangularPrism where
  height : ℝ
  baseLength : ℝ
  density : ℝ

/-- Calculates the weight of a rectangular prism -/
def weight (prism : RectangularPrism) : ℝ :=
  prism.height * prism.baseLength * prism.baseLength * prism.density

/-- Theorem: The weight of the specified marble rectangular prism is 86400 kg -/
theorem marble_prism_weight :
  let prism : RectangularPrism := {
    height := 8,
    baseLength := 2,
    density := 2700
  }
  weight prism = 86400 := by
  sorry

end NUMINAMATH_CALUDE_marble_prism_weight_l343_34319


namespace NUMINAMATH_CALUDE_scale_length_calculation_l343_34349

/-- Calculates the total length of a scale given the number of equal parts and the length of each part. -/
def totalScaleLength (numParts : ℕ) (partLength : ℝ) : ℝ :=
  numParts * partLength

/-- Theorem: The total length of a scale with 5 equal parts, each 25 inches long, is 125 inches. -/
theorem scale_length_calculation :
  totalScaleLength 5 25 = 125 := by
  sorry

end NUMINAMATH_CALUDE_scale_length_calculation_l343_34349


namespace NUMINAMATH_CALUDE_stream_speed_l343_34353

theorem stream_speed 
  (downstream_distance : ℝ) 
  (upstream_distance : ℝ) 
  (downstream_time : ℝ) 
  (upstream_time : ℝ) 
  (downstream_wind : ℝ) 
  (upstream_wind : ℝ) 
  (h1 : downstream_distance = 110) 
  (h2 : upstream_distance = 85) 
  (h3 : downstream_time = 5) 
  (h4 : upstream_time = 6) 
  (h5 : downstream_wind = 3) 
  (h6 : upstream_wind = 2) : 
  ∃ (boat_speed stream_speed : ℝ), 
    downstream_distance = (boat_speed + stream_speed + downstream_wind) * downstream_time ∧ 
    upstream_distance = (boat_speed - stream_speed + upstream_wind) * upstream_time ∧ 
    stream_speed = 3.4 := by
  sorry

end NUMINAMATH_CALUDE_stream_speed_l343_34353


namespace NUMINAMATH_CALUDE_cubic_sum_theorem_l343_34365

theorem cubic_sum_theorem (p q r : ℝ) (h_distinct : p ≠ q ∧ q ≠ r ∧ p ≠ r) 
  (h_eq : (p^3 + 10) / p = (q^3 + 10) / q ∧ (q^3 + 10) / q = (r^3 + 10) / r) : 
  p^3 + q^3 + r^3 = -30 := by
  sorry

end NUMINAMATH_CALUDE_cubic_sum_theorem_l343_34365


namespace NUMINAMATH_CALUDE_family_weight_problem_l343_34364

/-- Given a family with a grandmother, daughter, and child, prove their weights satisfy certain conditions and the combined weight of the daughter and child is 60 kg. -/
theorem family_weight_problem (grandmother daughter child : ℝ) : 
  grandmother + daughter + child = 110 →
  child = (1 / 5) * grandmother →
  daughter = 50 →
  daughter + child = 60 := by
sorry

end NUMINAMATH_CALUDE_family_weight_problem_l343_34364


namespace NUMINAMATH_CALUDE_cube_root_of_negative_twenty_seven_l343_34352

-- Define the cube root function for real numbers
noncomputable def cubeRoot (x : ℝ) : ℝ := Real.rpow x (1/3)

-- State the theorem
theorem cube_root_of_negative_twenty_seven :
  cubeRoot (-27) = -3 := by sorry

end NUMINAMATH_CALUDE_cube_root_of_negative_twenty_seven_l343_34352


namespace NUMINAMATH_CALUDE_solution_set_l343_34388

theorem solution_set (x y : ℝ) : 
  x - 2*y = 1 → x^3 - 8*y^3 - 6*x*y = 1 → y = (x-1)/2 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_l343_34388


namespace NUMINAMATH_CALUDE_demand_increase_factor_l343_34390

def demand (p : ℝ) : ℝ := 150 - p

def supply (p : ℝ) : ℝ := 3 * p - 10

def new_demand (α : ℝ) (p : ℝ) : ℝ := α * (150 - p)

theorem demand_increase_factor (α : ℝ) :
  (∃ p_initial p_new : ℝ,
    demand p_initial = supply p_initial ∧
    new_demand α p_new = supply p_new ∧
    p_new = 1.25 * p_initial) →
  α = 1.4 := by sorry

end NUMINAMATH_CALUDE_demand_increase_factor_l343_34390


namespace NUMINAMATH_CALUDE_hyperbola_equation_l343_34380

/-- The standard equation of a hyperbola with given focus and conjugate axis endpoint -/
theorem hyperbola_equation (f : ℝ × ℝ) (e : ℝ × ℝ) :
  f = (-10, 0) →
  e = (0, 4) →
  ∀ x y : ℝ, (x^2 / 84 - y^2 / 16 = 1) ↔ 
    (∃ a b c : ℝ, a^2 = 84 ∧ b^2 = 16 ∧ c^2 = a^2 + b^2 ∧
      x^2 / a^2 - y^2 / b^2 = 1 ∧
      c = 10 ∧ 
      (x - f.1)^2 + (y - f.2)^2 - ((x + 10)^2 + y^2) = 4 * a^2) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l343_34380


namespace NUMINAMATH_CALUDE_completing_square_sum_l343_34304

theorem completing_square_sum (x : ℝ) : 
  (∃ m n : ℝ, (x^2 - 6*x = 1) ↔ ((x - m)^2 = n)) → 
  (∃ m n : ℝ, (x^2 - 6*x = 1) ↔ ((x - m)^2 = n) ∧ m + n = 13) :=
by sorry

end NUMINAMATH_CALUDE_completing_square_sum_l343_34304


namespace NUMINAMATH_CALUDE_optimal_difference_optimal_difference_four_stars_l343_34389

/-- Represents the state of the game board -/
structure GameBoard where
  n : ℕ
  minuend : List ℕ
  subtrahend : List ℕ

/-- Represents a player in the game -/
inductive Player
  | First
  | Second

/-- Represents a move in the game -/
structure Move where
  digit : ℕ
  position : ℕ
  player : Player

/-- The game state after a sequence of moves -/
def gameState (initial : GameBoard) (moves : List Move) : GameBoard :=
  sorry

/-- The difference between minuend and subtrahend on the game board -/
def boardDifference (board : GameBoard) : ℕ :=
  sorry

/-- Optimal strategy for the first player -/
def firstPlayerStrategy (board : GameBoard) : Move :=
  sorry

/-- Optimal strategy for the second player -/
def secondPlayerStrategy (board : GameBoard) (digit : ℕ) : Move :=
  sorry

/-- The main theorem stating the optimal difference -/
theorem optimal_difference (n : ℕ) :
  ∀ (moves : List Move),
    boardDifference (gameState (GameBoard.mk n [] []) moves) ≤ 4 * 10^(n-1) ∧
    boardDifference (gameState (GameBoard.mk n [] []) moves) ≥ 4 * 10^(n-1) :=
  sorry

/-- Corollary for the specific case of n = 4 -/
theorem optimal_difference_four_stars :
  ∀ (moves : List Move),
    boardDifference (gameState (GameBoard.mk 4 [] []) moves) = 4000 :=
  sorry

end NUMINAMATH_CALUDE_optimal_difference_optimal_difference_four_stars_l343_34389


namespace NUMINAMATH_CALUDE_anthony_transaction_percentage_l343_34398

theorem anthony_transaction_percentage (mabel_transactions cal_transactions jade_transactions : ℕ)
  (anthony_transactions : ℕ) :
  mabel_transactions = 90 →
  cal_transactions = (2 : ℚ) / 3 * anthony_transactions →
  jade_transactions = cal_transactions + 14 →
  jade_transactions = 80 →
  (anthony_transactions - mabel_transactions : ℚ) / mabel_transactions * 100 = 10 := by
sorry

end NUMINAMATH_CALUDE_anthony_transaction_percentage_l343_34398


namespace NUMINAMATH_CALUDE_arithmetic_geometric_progression_sine_l343_34372

theorem arithmetic_geometric_progression_sine (x y z : ℝ) :
  let α := Real.arccos (-1/5)
  (∃ d, x = y - d ∧ z = y + d ∧ d = α) →
  (∃ r ≠ 1, (2 + Real.sin x) * (2 + Real.sin z) = (2 + Real.sin y)^2 ∧ 
             (2 + Real.sin y) = r * (2 + Real.sin x) ∧
             (2 + Real.sin z) = r * (2 + Real.sin y)) →
  Real.sin y = -1 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_progression_sine_l343_34372


namespace NUMINAMATH_CALUDE_yoojeong_drank_most_l343_34378

def yoojeong_milk : ℚ := 7/10
def eunji_milk : ℚ := 1/2
def yuna_milk : ℚ := 6/10

theorem yoojeong_drank_most : 
  yoojeong_milk > eunji_milk ∧ yoojeong_milk > yuna_milk := by
  sorry

end NUMINAMATH_CALUDE_yoojeong_drank_most_l343_34378


namespace NUMINAMATH_CALUDE_orange_bucket_difference_l343_34321

/-- Proves that the difference between the number of oranges in the second and first buckets is 17 -/
theorem orange_bucket_difference :
  ∀ (second_bucket : ℕ),
  22 + second_bucket + (second_bucket - 11) = 89 →
  second_bucket - 22 = 17 :=
by
  sorry

end NUMINAMATH_CALUDE_orange_bucket_difference_l343_34321


namespace NUMINAMATH_CALUDE_five_students_left_l343_34305

/-- Calculates the number of students who left during the year. -/
def students_who_left (initial_students new_students final_students : ℕ) : ℕ :=
  initial_students + new_students - final_students

/-- Proves that 5 students left during the year given the problem conditions. -/
theorem five_students_left : students_who_left 31 11 37 = 5 := by
  sorry

end NUMINAMATH_CALUDE_five_students_left_l343_34305


namespace NUMINAMATH_CALUDE_share_distribution_l343_34311

theorem share_distribution (total : ℚ) (a b c : ℚ) 
  (h1 : total = 578)
  (h2 : a = (2/3) * b)
  (h3 : b = (1/4) * c)
  (h4 : a + b + c = total) : 
  c = 408 := by
  sorry

end NUMINAMATH_CALUDE_share_distribution_l343_34311


namespace NUMINAMATH_CALUDE_least_c_for_triple_f_l343_34318

def f (x : ℤ) : ℤ :=
  if x % 2 = 1 then x + 5 else x / 2

def is_odd (n : ℤ) : Prop := n % 2 = 1

theorem least_c_for_triple_f (b : ℤ) :
  ∃ c : ℤ, is_odd c ∧ f (f (f c)) = b ∧ ∀ d : ℤ, is_odd d ∧ f (f (f d)) = b → c ≤ d :=
sorry

end NUMINAMATH_CALUDE_least_c_for_triple_f_l343_34318


namespace NUMINAMATH_CALUDE_heather_bicycle_distance_l343_34367

-- Define the speed in kilometers per hour
def speed : ℝ := 8

-- Define the time in hours
def time : ℝ := 5

-- Define the distance formula
def distance (s t : ℝ) : ℝ := s * t

-- Theorem to prove
theorem heather_bicycle_distance :
  distance speed time = 40 := by
  sorry

end NUMINAMATH_CALUDE_heather_bicycle_distance_l343_34367


namespace NUMINAMATH_CALUDE_largest_root_range_l343_34366

theorem largest_root_range (b₀ b₁ b₂ : ℝ) 
  (h₀ : |b₀| < 3) (h₁ : |b₁| < 3) (h₂ : |b₂| < 3) :
  ∃ r : ℝ, 3.5 < r ∧ r < 5 ∧
  (∀ x : ℝ, x > 0 → x^4 + x^3 + b₂*x^2 + b₁*x + b₀ = 0 → x ≤ r) ∧
  (∃ x : ℝ, x > 0 ∧ x^4 + x^3 + b₂*x^2 + b₁*x + b₀ = 0 ∧ x = r) :=
by sorry

end NUMINAMATH_CALUDE_largest_root_range_l343_34366


namespace NUMINAMATH_CALUDE_cube_sum_equals_275_l343_34361

theorem cube_sum_equals_275 (a b : ℝ) (h1 : a + b = 11) (h2 : a * b = 32) :
  a^3 + b^3 = 275 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_equals_275_l343_34361


namespace NUMINAMATH_CALUDE_line_tangent_to_circle_l343_34358

-- Define the line l
def line_l (c : ℝ) : ℝ → ℝ → Prop :=
  fun x y => x + 2*y + c = 0

-- Define the circle C
def circle_C : ℝ → ℝ → Prop :=
  fun x y => x^2 + y^2 + 2*x - 4*y = 0

-- Define the translated line l'
def line_l_prime (c : ℝ) : ℝ → ℝ → Prop :=
  fun x y => x + 2*y + c + 5 = 0

-- Define the tangency condition
def is_tangent (l : ℝ → ℝ → Prop) (C : ℝ → ℝ → Prop) : Prop :=
  ∃ x y, l x y ∧ C x y ∧ ∀ x' y', l x' y' ∧ C x' y' → (x = x' ∧ y = y')

theorem line_tangent_to_circle (c : ℝ) :
  is_tangent (line_l_prime c) circle_C → c = -3 ∨ c = -13 :=
by sorry

end NUMINAMATH_CALUDE_line_tangent_to_circle_l343_34358


namespace NUMINAMATH_CALUDE_smaller_two_digit_number_l343_34306

theorem smaller_two_digit_number 
  (x y : ℕ) 
  (h1 : x < y) 
  (h2 : x ≥ 10 ∧ x < 100) 
  (h3 : y ≥ 10 ∧ y < 100) 
  (h4 : x + y = 88) 
  (h5 : (100 * y + x) - (100 * x + y) = 3564) : 
  x = 26 := by
sorry

end NUMINAMATH_CALUDE_smaller_two_digit_number_l343_34306


namespace NUMINAMATH_CALUDE_set_operations_l343_34387

def A : Set ℕ := {1, 3, 5, 7, 9}
def B : Set ℕ := {3, 4, 5}

theorem set_operations :
  (A ∪ B = {1, 3, 4, 5, 7, 9}) ∧
  (A ∩ B = {3, 5}) ∧
  ({x | x ∈ A ∧ x ∉ B} = {1, 7, 9}) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_l343_34387


namespace NUMINAMATH_CALUDE_negative_abs_negative_five_l343_34313

theorem negative_abs_negative_five : -|-5| = -5 := by
  sorry

end NUMINAMATH_CALUDE_negative_abs_negative_five_l343_34313


namespace NUMINAMATH_CALUDE_vector_decomposition_l343_34385

def x : Fin 3 → ℝ := ![3, -1, 2]
def p : Fin 3 → ℝ := ![2, 0, 1]
def q : Fin 3 → ℝ := ![1, -1, 1]
def r : Fin 3 → ℝ := ![1, -1, -2]

theorem vector_decomposition :
  x = p + q :=
by sorry

end NUMINAMATH_CALUDE_vector_decomposition_l343_34385


namespace NUMINAMATH_CALUDE_melissa_family_theorem_l343_34377

/-- The number of Melissa's daughters and granddaughters who have no daughters -/
def num_without_daughters (total_descendants : ℕ) (num_daughters : ℕ) (daughters_with_children : ℕ) (granddaughters_per_daughter : ℕ) : ℕ :=
  (num_daughters - daughters_with_children) + (daughters_with_children * granddaughters_per_daughter)

theorem melissa_family_theorem :
  let total_descendants := 50
  let num_daughters := 10
  let daughters_with_children := num_daughters / 2
  let granddaughters_per_daughter := 4
  num_without_daughters total_descendants num_daughters daughters_with_children granddaughters_per_daughter = 45 := by
sorry

end NUMINAMATH_CALUDE_melissa_family_theorem_l343_34377


namespace NUMINAMATH_CALUDE_parallelogram_sides_l343_34338

/-- Represents a parallelogram with side lengths a and b -/
structure Parallelogram where
  a : ℝ
  b : ℝ
  a_positive : 0 < a
  b_positive : 0 < b

/-- The perimeter of a parallelogram -/
def perimeter (p : Parallelogram) : ℝ := 2 * (p.a + p.b)

/-- The difference between perimeters of adjacent triangles formed by diagonals -/
def triangle_perimeter_difference (p : Parallelogram) : ℝ := abs (p.b - p.a)

theorem parallelogram_sides (p : Parallelogram) 
  (h_perimeter : perimeter p = 44)
  (h_diff : triangle_perimeter_difference p = 6) :
  p.a = 8 ∧ p.b = 14 := by
  sorry

#check parallelogram_sides

end NUMINAMATH_CALUDE_parallelogram_sides_l343_34338


namespace NUMINAMATH_CALUDE_exactly_two_sets_l343_34328

/-- A structure representing a set of consecutive positive integers -/
structure ConsecutiveSet where
  start : ℕ+
  length : ℕ+

/-- The sum of a set of consecutive positive integers -/
def sum_consecutive (s : ConsecutiveSet) : ℕ :=
  (s.length : ℕ) * (2 * (s.start : ℕ) + s.length - 1) / 2

/-- Predicate for a valid set of consecutive integers summing to 256 -/
def is_valid_set (s : ConsecutiveSet) : Prop :=
  s.length ≥ 2 ∧ sum_consecutive s = 256

theorem exactly_two_sets :
  ∃! (sets : Finset ConsecutiveSet), sets.card = 2 ∧ ∀ s ∈ sets, is_valid_set s :=
sorry

end NUMINAMATH_CALUDE_exactly_two_sets_l343_34328


namespace NUMINAMATH_CALUDE_total_money_l343_34303

/-- The total amount of money A, B, and C have between them is 700, given:
  * A and C together have 300
  * B and C together have 600
  * C has 200 -/
theorem total_money (A B C : ℕ) : 
  A + C = 300 → B + C = 600 → C = 200 → A + B + C = 700 := by
  sorry

end NUMINAMATH_CALUDE_total_money_l343_34303


namespace NUMINAMATH_CALUDE_juku_exit_position_l343_34335

/-- Represents the state of Juku on the escalator -/
structure EscalatorState where
  time : ℕ
  position : ℕ

/-- The escalator system with Juku's movement -/
def escalator_system (total_steps : ℕ) (start_position : ℕ) : ℕ → EscalatorState
| 0 => ⟨0, start_position⟩
| t + 1 => 
  let prev := escalator_system total_steps start_position t
  let new_pos := 
    if t % 3 == 0 then prev.position - 1
    else if t % 3 == 1 then prev.position + 1
    else prev.position - 2
  ⟨t + 1, new_pos⟩

/-- Theorem: Juku exits at the 23rd step relative to the ground -/
theorem juku_exit_position : 
  ∃ (t : ℕ), (escalator_system 75 38 t).position + (t / 2) = 23 := by
  sorry

#eval (escalator_system 75 38 45).position + 45 / 2

end NUMINAMATH_CALUDE_juku_exit_position_l343_34335


namespace NUMINAMATH_CALUDE_centroid_property_l343_34302

/-- Definition of a point in 2D space -/
structure Point2D where
  x : ℚ
  y : ℚ

/-- Definition of a triangle in 2D space -/
structure Triangle where
  A : Point2D
  B : Point2D
  C : Point2D

/-- Calculate the centroid of a triangle -/
def centroid (t : Triangle) : Point2D :=
  { x := (t.A.x + t.B.x + t.C.x) / 3,
    y := (t.A.y + t.B.y + t.C.y) / 3 }

/-- The main theorem -/
theorem centroid_property :
  let t := Triangle.mk
    (Point2D.mk (-1) 4)
    (Point2D.mk 5 2)
    (Point2D.mk 3 10)
  let c := centroid t
  10 * c.x + c.y = 86 / 3 := by
  sorry

end NUMINAMATH_CALUDE_centroid_property_l343_34302


namespace NUMINAMATH_CALUDE_igloo_bottom_row_bricks_l343_34320

/-- Represents the structure of an igloo --/
structure Igloo where
  total_rows : ℕ
  top_row_bricks : ℕ
  total_bricks : ℕ

/-- Calculates the number of bricks in each row of the bottom half of the igloo --/
def bottom_row_bricks (igloo : Igloo) : ℕ :=
  let bottom_rows := igloo.total_rows / 2
  let top_bricks := bottom_rows * igloo.top_row_bricks
  (igloo.total_bricks - top_bricks) / bottom_rows

/-- Theorem stating that for the given igloo specifications, 
    the number of bricks in each row of the bottom half is 12 --/
theorem igloo_bottom_row_bricks :
  let igloo : Igloo := { total_rows := 10, top_row_bricks := 8, total_bricks := 100 }
  bottom_row_bricks igloo = 12 := by
  sorry


end NUMINAMATH_CALUDE_igloo_bottom_row_bricks_l343_34320


namespace NUMINAMATH_CALUDE_part_one_part_two_l343_34327

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := m - |x - 3|

-- Part 1
theorem part_one (m : ℝ) : 
  (∀ x, f m x > 2 ↔ 2 < x ∧ x < 4) → m = 3 := by sorry

-- Part 2
theorem part_two (a : ℝ) :
  (∀ x, |x - a| ≥ f 3 x) → (a ≥ 6 ∨ a ≤ 0) := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l343_34327


namespace NUMINAMATH_CALUDE_alpha_plus_beta_eq_128_l343_34324

theorem alpha_plus_beta_eq_128 :
  ∀ α β : ℝ, (∀ x : ℝ, (x - α) / (x + β) = (x^2 - 96*x + 2209) / (x^2 + 66*x - 3969)) →
  α + β = 128 := by
sorry

end NUMINAMATH_CALUDE_alpha_plus_beta_eq_128_l343_34324


namespace NUMINAMATH_CALUDE_tom_dance_lessons_l343_34383

theorem tom_dance_lessons 
  (cost_per_lesson : ℕ) 
  (free_lessons : ℕ) 
  (total_paid : ℕ) :
  cost_per_lesson = 10 →
  free_lessons = 2 →
  total_paid = 80 →
  (total_paid / cost_per_lesson) + free_lessons = 10 :=
by sorry

end NUMINAMATH_CALUDE_tom_dance_lessons_l343_34383
