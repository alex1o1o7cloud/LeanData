import Mathlib

namespace NUMINAMATH_CALUDE_function_value_at_negative_pi_fourth_l1504_150478

theorem function_value_at_negative_pi_fourth 
  (f : ℝ → ℝ) 
  (a b : ℝ) 
  (h1 : ∀ x, f x = a * Real.tan x - b * Real.sin x + 1) 
  (h2 : f (π/4) = 7) : 
  f (-π/4) = -5 := by
sorry

end NUMINAMATH_CALUDE_function_value_at_negative_pi_fourth_l1504_150478


namespace NUMINAMATH_CALUDE_special_polynomial_sum_l1504_150425

theorem special_polynomial_sum (d₁ d₂ d₃ d₄ e₁ e₂ e₃ e₄ : ℝ) 
  (h : ∀ x : ℝ, x^8 - x^7 + x^6 - x^5 + x^4 - x^3 + x^2 - x + 1 = 
    (x^2 + d₁*x + e₁)*(x^2 + d₂*x + e₂)*(x^2 + d₃*x + e₃)*(x^2 + d₄*x + e₄)) : 
  d₁*e₁ + d₂*e₂ + d₃*e₃ + d₄*e₄ = -1 := by
  sorry

end NUMINAMATH_CALUDE_special_polynomial_sum_l1504_150425


namespace NUMINAMATH_CALUDE_greatest_x_with_lcm_l1504_150483

/-- Given that the least common multiple of x, 15, and 21 is 105, 
    the greatest possible value of x is 105. -/
theorem greatest_x_with_lcm (x : ℕ) : 
  Nat.lcm x (Nat.lcm 15 21) = 105 → x ≤ 105 ∧ ∃ y : ℕ, y > 105 → Nat.lcm y (Nat.lcm 15 21) > 105 :=
by sorry

end NUMINAMATH_CALUDE_greatest_x_with_lcm_l1504_150483


namespace NUMINAMATH_CALUDE_one_third_percent_of_180_l1504_150427

theorem one_third_percent_of_180 : (1 / 3 : ℚ) / 100 * 180 = 0.6 := by sorry

end NUMINAMATH_CALUDE_one_third_percent_of_180_l1504_150427


namespace NUMINAMATH_CALUDE_draw_three_with_red_standard_deck_l1504_150429

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : Nat)
  (suits : Nat)
  (cards_per_suit : Nat)
  (black_suits : Nat)
  (red_suits : Nat)

/-- Calculate the number of ways to draw three cards with at least one red card -/
def draw_three_with_red (d : Deck) : Nat :=
  d.total_cards * (d.total_cards - 1) * (d.total_cards - 2) - 
  (d.black_suits * d.cards_per_suit) * (d.black_suits * d.cards_per_suit - 1) * (d.black_suits * d.cards_per_suit - 2)

/-- Theorem: The number of ways to draw three cards with at least one red from a standard deck is 117000 -/
theorem draw_three_with_red_standard_deck :
  let standard_deck : Deck := {
    total_cards := 52,
    suits := 4,
    cards_per_suit := 13,
    black_suits := 2,
    red_suits := 2
  }
  draw_three_with_red standard_deck = 117000 := by
  sorry

end NUMINAMATH_CALUDE_draw_three_with_red_standard_deck_l1504_150429


namespace NUMINAMATH_CALUDE_sum_of_roots_l1504_150411

def f (x : ℝ) : ℝ := x^3 + 3*x^2 + 6*x + 14

theorem sum_of_roots (a b : ℝ) (ha : f a = 1) (hb : f b = 19) : a + b = -2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_l1504_150411


namespace NUMINAMATH_CALUDE_work_ratio_l1504_150454

theorem work_ratio (a b : ℝ) (ha : a = 8) (hab : 1/a + 1/b = 0.375) : b/a = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_work_ratio_l1504_150454


namespace NUMINAMATH_CALUDE_arcsin_cos_arcsin_plus_arccos_sin_arccos_l1504_150409

theorem arcsin_cos_arcsin_plus_arccos_sin_arccos (x : ℝ) : 
  Real.arcsin (Real.cos (Real.arcsin x)) + Real.arccos (Real.sin (Real.arccos x)) = π / 2 := by
  sorry

end NUMINAMATH_CALUDE_arcsin_cos_arcsin_plus_arccos_sin_arccos_l1504_150409


namespace NUMINAMATH_CALUDE_nested_fraction_equality_l1504_150463

theorem nested_fraction_equality : 
  (1 : ℚ) / (3 - 1 / (3 - 1 / (3 - 1 / (3 - 1 / 3)))) = 21 / 55 := by
  sorry

end NUMINAMATH_CALUDE_nested_fraction_equality_l1504_150463


namespace NUMINAMATH_CALUDE_fraction_inequality_l1504_150486

theorem fraction_inequality (x : ℝ) : (x + 6) / (x^2 + 2*x + 7) ≥ 0 ↔ x ≥ -6 := by
  sorry

end NUMINAMATH_CALUDE_fraction_inequality_l1504_150486


namespace NUMINAMATH_CALUDE_arithmetic_sequence_30th_term_l1504_150422

/-- A sequence {a_n} with sum S_n satisfying the given conditions -/
def ArithmeticSequence (a : ℕ → ℚ) (S : ℕ → ℚ) : Prop :=
  (∀ n : ℕ, 3 * S n / n + n = 3 * a n + 1) ∧ (a 1 = -1/3)

/-- Theorem stating that the 30th term of the sequence is 19 -/
theorem arithmetic_sequence_30th_term
  (a : ℕ → ℚ) (S : ℕ → ℚ) (h : ArithmeticSequence a S) :
  a 30 = 19 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_30th_term_l1504_150422


namespace NUMINAMATH_CALUDE_professor_count_l1504_150415

theorem professor_count (p : ℕ) 
  (h1 : 6480 % p = 0)  -- 6480 is divisible by p
  (h2 : 11200 % (p + 3) = 0)  -- 11200 is divisible by (p + 3)
  (h3 : (6480 : ℚ) / p < (11200 : ℚ) / (p + 3))  -- grades per professor increased
  : p = 5 := by
  sorry

end NUMINAMATH_CALUDE_professor_count_l1504_150415


namespace NUMINAMATH_CALUDE_point_in_quadrant_iv_l1504_150426

/-- Given a system of equations x - y = a and 6x + 5y = -1, where x = 1,
    prove that the point (a, y) is in Quadrant IV -/
theorem point_in_quadrant_iv (a : ℚ) : 
  let x : ℚ := 1
  let y : ℚ := -7/5
  (x - y = a) → (6 * x + 5 * y = -1) → (a > 0 ∧ y < 0) := by
  sorry

#check point_in_quadrant_iv

end NUMINAMATH_CALUDE_point_in_quadrant_iv_l1504_150426


namespace NUMINAMATH_CALUDE_solve_for_k_l1504_150419

-- Define the polynomials
def p (x y k : ℝ) : ℝ := x^3 - 2*k*x*y
def q (x y : ℝ) : ℝ := y^2 + 4*x*y

-- Define the condition that the difference doesn't contain xy term
def no_xy_term (k : ℝ) : Prop :=
  ∀ x y, ∃ a b c, p x y k - q x y = a*x^3 + b*y^2 + c

-- State the theorem
theorem solve_for_k :
  ∃ k : ℝ, no_xy_term k ∧ k = -2 :=
sorry

end NUMINAMATH_CALUDE_solve_for_k_l1504_150419


namespace NUMINAMATH_CALUDE_integer_roots_imply_n_values_l1504_150452

theorem integer_roots_imply_n_values (n : ℤ) : 
  (∃ x y : ℤ, x ≠ y ∧ x^2 - 6*x - 4*n^2 - 32*n = 0 ∧ y^2 - 6*y - 4*n^2 - 32*n = 0) →
  (n = 10 ∨ n = 0 ∨ n = -8 ∨ n = -18) :=
by sorry

end NUMINAMATH_CALUDE_integer_roots_imply_n_values_l1504_150452


namespace NUMINAMATH_CALUDE_student_money_proof_l1504_150417

/-- The amount of money (in rubles) the student has after buying 11 pens -/
def remaining_after_11 : ℝ := 8

/-- The additional amount (in rubles) needed to buy 15 pens -/
def additional_for_15 : ℝ := 12.24

/-- The cost of one pen in rubles -/
noncomputable def pen_cost : ℝ :=
  (additional_for_15 + remaining_after_11) / (15 - 11)

/-- The initial amount of money the student had in rubles -/
noncomputable def initial_amount : ℝ :=
  11 * pen_cost + remaining_after_11

theorem student_money_proof :
  initial_amount = 63.66 := by sorry

end NUMINAMATH_CALUDE_student_money_proof_l1504_150417


namespace NUMINAMATH_CALUDE_three_digit_number_divisible_by_seven_l1504_150441

theorem three_digit_number_divisible_by_seven (a b : ℕ) 
  (h1 : a ≥ 1 ∧ a ≤ 9) 
  (h2 : b ≥ 0 ∧ b ≤ 9) 
  (h3 : (a + b + b) % 7 = 0) : 
  ∃ k : ℕ, (100 * a + 10 * b + b) = 7 * k :=
sorry

end NUMINAMATH_CALUDE_three_digit_number_divisible_by_seven_l1504_150441


namespace NUMINAMATH_CALUDE_min_value_on_circle_l1504_150496

theorem min_value_on_circle (x y : ℝ) (h : (x + 5)^2 + (y - 12)^2 = 14^2) :
  ∃ (m : ℝ), (∀ (a b : ℝ), (a + 5)^2 + (b - 12)^2 = 14^2 → x^2 + y^2 ≤ a^2 + b^2) ∧ m = 1 :=
by sorry

end NUMINAMATH_CALUDE_min_value_on_circle_l1504_150496


namespace NUMINAMATH_CALUDE_correct_statements_l1504_150412

theorem correct_statements (a b c d : ℝ) :
  (ab > 0 ∧ bc - ad > 0 → c / a - d / b > 0) ∧
  (a > b ∧ c > d → a - d > b - c) :=
by sorry

end NUMINAMATH_CALUDE_correct_statements_l1504_150412


namespace NUMINAMATH_CALUDE_flame_time_calculation_l1504_150406

/-- Represents the duration of one minute in seconds -/
def minute_duration : ℕ := 60

/-- Represents the interval between weapon fires in seconds -/
def fire_interval : ℕ := 15

/-- Represents the duration of each flame shot in seconds -/
def flame_duration : ℕ := 5

/-- Calculates the total time spent shooting flames in one minute -/
def flame_time_per_minute : ℕ := (minute_duration / fire_interval) * flame_duration

theorem flame_time_calculation :
  flame_time_per_minute = 20 := by sorry

end NUMINAMATH_CALUDE_flame_time_calculation_l1504_150406


namespace NUMINAMATH_CALUDE_tomatoes_calculation_l1504_150420

/-- The number of tomato plants -/
def num_plants : ℕ := 50

/-- The number of tomatoes produced by each plant -/
def tomatoes_per_plant : ℕ := 15

/-- The fraction of tomatoes that are dried -/
def dried_fraction : ℚ := 2 / 3

/-- The fraction of remaining tomatoes used for marinara sauce -/
def marinara_fraction : ℚ := 1 / 2

/-- The number of tomatoes left after drying and making marinara sauce -/
def tomatoes_left : ℕ := 125

theorem tomatoes_calculation :
  (num_plants * tomatoes_per_plant : ℚ) * (1 - dried_fraction) * (1 - marinara_fraction) = tomatoes_left := by
  sorry

end NUMINAMATH_CALUDE_tomatoes_calculation_l1504_150420


namespace NUMINAMATH_CALUDE_trapezoid_area_l1504_150480

/-- A trapezoid bounded by y = 2x, y = 12, y = 8, and the y-axis -/
structure Trapezoid where
  /-- The line y = 2x -/
  line : ℝ → ℝ
  /-- The upper bound y = 12 -/
  upper_bound : ℝ
  /-- The lower bound y = 8 -/
  lower_bound : ℝ
  /-- The line is y = 2x -/
  line_eq : ∀ x, line x = 2 * x
  /-- The upper bound is 12 -/
  upper_eq : upper_bound = 12
  /-- The lower bound is 8 -/
  lower_eq : lower_bound = 8

/-- The area of the trapezoid -/
def area (t : Trapezoid) : ℝ := sorry

/-- Theorem: The area of the specified trapezoid is 20 square units -/
theorem trapezoid_area : ∀ t : Trapezoid, area t = 20 := by sorry

end NUMINAMATH_CALUDE_trapezoid_area_l1504_150480


namespace NUMINAMATH_CALUDE_andrew_remaining_vacation_days_l1504_150459

/-- Calculates the remaining vacation days for an employee given their work days and vacation days taken. -/
def remaining_vacation_days (work_days : ℕ) (march_vacation : ℕ) : ℕ :=
  let earned_days := work_days / 10
  let taken_days := march_vacation + 2 * march_vacation
  earned_days - taken_days

/-- Theorem stating that Andrew has 15 remaining vacation days. -/
theorem andrew_remaining_vacation_days :
  remaining_vacation_days 300 5 = 15 := by
  sorry

#eval remaining_vacation_days 300 5

end NUMINAMATH_CALUDE_andrew_remaining_vacation_days_l1504_150459


namespace NUMINAMATH_CALUDE_early_arrival_speed_l1504_150470

/-- Represents the travel scenario for Mrs. Early --/
structure TravelScenario where
  speed : ℝ
  timeDifference : ℝ  -- in hours, positive for early, negative for late

/-- Calculates the required speed to arrive exactly on time --/
def exactTimeSpeed (scenario1 scenario2 : TravelScenario) : ℝ :=
  -- Implementation details omitted
  sorry

/-- Theorem stating the correct speed for Mrs. Early to arrive on time --/
theorem early_arrival_speed : 
  let scenario1 : TravelScenario := { speed := 50, timeDifference := -1/15 }
  let scenario2 : TravelScenario := { speed := 70, timeDifference := 1/12 }
  let requiredSpeed := exactTimeSpeed scenario1 scenario2
  57 < requiredSpeed ∧ requiredSpeed < 58 := by
  sorry

end NUMINAMATH_CALUDE_early_arrival_speed_l1504_150470


namespace NUMINAMATH_CALUDE_number_problem_l1504_150418

theorem number_problem (N : ℝ) (h : (1/4) * (1/3) * (2/5) * N = 25) : 
  0.40 * N = 300 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l1504_150418


namespace NUMINAMATH_CALUDE_max_min_f_on_interval_l1504_150464

def f (x : ℝ) := x^3 - 12*x + 8

theorem max_min_f_on_interval :
  ∃ (max min : ℝ),
    (∀ x ∈ Set.Icc (-3) 3, f x ≤ max) ∧
    (∃ x ∈ Set.Icc (-3) 3, f x = max) ∧
    (∀ x ∈ Set.Icc (-3) 3, min ≤ f x) ∧
    (∃ x ∈ Set.Icc (-3) 3, f x = min) ∧
    max = 24 ∧ min = -6 := by
  sorry

end NUMINAMATH_CALUDE_max_min_f_on_interval_l1504_150464


namespace NUMINAMATH_CALUDE_tangent_line_at_one_max_value_of_f_l1504_150414

/-- The function f(x) defined as 2a ln x - x^2 --/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2 * a * Real.log x - x^2

/-- Theorem stating the equation of the tangent line when a = 2 --/
theorem tangent_line_at_one (a : ℝ) (h : a = 2) :
  ∃ m b : ℝ, ∀ x y : ℝ, y = m * x + b ↔ 2 * x - y - 3 = 0 :=
sorry

/-- Theorem stating the maximum value of f(x) when a > 0 --/
theorem max_value_of_f (a : ℝ) (h : a > 0) :
  ∃ x_max : ℝ, x_max = Real.sqrt 2 ∧
    ∀ x : ℝ, x > 0 → f a x ≤ f a x_max ∧ f a x_max = Real.log 2 - 2 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_at_one_max_value_of_f_l1504_150414


namespace NUMINAMATH_CALUDE_prove_equation_l1504_150493

theorem prove_equation (c d : ℝ) 
  (h1 : 5 + c = 6 - d) 
  (h2 : 3 + d = 8 + c) : 
  5 - c = 7 := by
sorry

end NUMINAMATH_CALUDE_prove_equation_l1504_150493


namespace NUMINAMATH_CALUDE_total_pictures_calculation_l1504_150442

/-- The number of pictures that can be contained in one album -/
def pictures_per_album : ℕ := 20

/-- The number of albums needed -/
def albums_needed : ℕ := 24

/-- The total number of pictures -/
def total_pictures : ℕ := pictures_per_album * albums_needed

theorem total_pictures_calculation :
  total_pictures = 480 :=
by sorry

end NUMINAMATH_CALUDE_total_pictures_calculation_l1504_150442


namespace NUMINAMATH_CALUDE_quadratic_equation_properties_l1504_150434

theorem quadratic_equation_properties (k : ℝ) :
  let f : ℝ → ℝ := λ x => x^2 - (k + 1) * x - 6
  (∃ x y : ℝ, x ≠ y ∧ f x = 0 ∧ f y = 0) ∧
  (f 2 = 0 → k = -2 ∧ f (-3) = 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_properties_l1504_150434


namespace NUMINAMATH_CALUDE_revolution_volume_formula_l1504_150450

/-- Region P in the coordinate plane -/
def P : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | |6 - p.1| + p.2 ≤ 8 ∧ 4 * p.2 - p.1 ≥ 20}

/-- The line around which P is revolved -/
def revolveLine (x y : ℝ) : Prop := 4 * y - x = 20

/-- The volume of the solid formed by revolving P around the line -/
noncomputable def revolutionVolume : ℝ := sorry

theorem revolution_volume_formula :
  revolutionVolume = 24 * Real.pi / (85 * Real.sqrt 3741) := by sorry

end NUMINAMATH_CALUDE_revolution_volume_formula_l1504_150450


namespace NUMINAMATH_CALUDE_point_A_coordinates_l1504_150495

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define the translation operation
def translate (p : Point2D) (dx dy : ℝ) : Point2D :=
  { x := p.x + dx, y := p.y + dy }

theorem point_A_coordinates : 
  ∀ A : Point2D, 
  let B := translate (translate A 0 (-3)) 2 0
  B = Point2D.mk (-1) 5 → A = Point2D.mk (-3) 8 := by
sorry

end NUMINAMATH_CALUDE_point_A_coordinates_l1504_150495


namespace NUMINAMATH_CALUDE_mikes_pears_l1504_150421

/-- Given that Jason picked 7 pears and the total number of pears picked was 15,
    prove that Mike picked 8 pears. -/
theorem mikes_pears (jason_pears total_pears : ℕ) 
    (h1 : jason_pears = 7)
    (h2 : total_pears = 15) :
    total_pears - jason_pears = 8 := by
  sorry

end NUMINAMATH_CALUDE_mikes_pears_l1504_150421


namespace NUMINAMATH_CALUDE_admission_price_is_two_l1504_150448

/-- Calculates the admission price for adults given the total number of people,
    admission price for children, total admission receipts, and number of adults. -/
def admission_price_for_adults (total_people : ℕ) (child_price : ℚ) 
                               (total_receipts : ℚ) (num_adults : ℕ) : ℚ :=
  (total_receipts - (total_people - num_adults : ℚ) * child_price) / num_adults

/-- Proves that the admission price for adults is $2 given the specific conditions. -/
theorem admission_price_is_two :
  admission_price_for_adults 610 1 960 350 = 2 := by sorry

end NUMINAMATH_CALUDE_admission_price_is_two_l1504_150448


namespace NUMINAMATH_CALUDE_tan_405_degrees_l1504_150455

theorem tan_405_degrees : Real.tan (405 * π / 180) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_405_degrees_l1504_150455


namespace NUMINAMATH_CALUDE_product_sum_of_digits_77_sevens_77_threes_l1504_150494

/-- Represents a string of digits repeated n times -/
def RepeatedDigitString (digit : Nat) (n : Nat) : Nat :=
  -- Definition omitted for brevity
  sorry

/-- Calculates the sum of digits in a natural number -/
def sumOfDigits (n : Nat) : Nat :=
  -- Definition omitted for brevity
  sorry

/-- The main theorem to prove -/
theorem product_sum_of_digits_77_sevens_77_threes :
  sumOfDigits (RepeatedDigitString 7 77 * RepeatedDigitString 3 77) = 231 := by
  sorry

end NUMINAMATH_CALUDE_product_sum_of_digits_77_sevens_77_threes_l1504_150494


namespace NUMINAMATH_CALUDE_recurring_decimal_equals_fraction_l1504_150474

/-- The decimal representation of 3.127̄ as a rational number -/
def recurring_decimal : ℚ := 3 + 127 / 999

/-- The fraction 3124/999 -/
def target_fraction : ℚ := 3124 / 999

/-- Theorem stating that the recurring decimal 3.127̄ is equal to the fraction 3124/999 -/
theorem recurring_decimal_equals_fraction : recurring_decimal = target_fraction := by
  sorry

end NUMINAMATH_CALUDE_recurring_decimal_equals_fraction_l1504_150474


namespace NUMINAMATH_CALUDE_negation_of_implication_l1504_150468

theorem negation_of_implication (A B : Set α) :
  ¬(A ∪ B = A → A ∩ B = B) ↔ (A ∪ B = A ∧ A ∩ B ≠ B) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_implication_l1504_150468


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_l1504_150431

theorem quadratic_roots_sum (x₁ x₂ : ℝ) : 
  x₁^2 - 3*x₁ + 1 = 0 → x₂^2 - 3*x₂ + 1 = 0 → x₁^2 + 3*x₁*x₂ + x₂^2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_l1504_150431


namespace NUMINAMATH_CALUDE_circle_ratio_l1504_150445

theorem circle_ratio (r R : ℝ) (h : r > 0) (H : R > 0) 
  (area_condition : π * R^2 - π * r^2 = 4 * (π * r^2)) : 
  r / R = 1 / Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_circle_ratio_l1504_150445


namespace NUMINAMATH_CALUDE_books_in_pile_A_l1504_150446

/-- Given three piles of books with the following properties:
  - The total number of books is 240
  - Pile A has 30 more than three times the books in pile B
  - Pile C has 15 fewer books than pile B
  Prove that pile A contains 165 books. -/
theorem books_in_pile_A (total : ℕ) (books_B : ℕ) (books_A : ℕ) (books_C : ℕ) : 
  total = 240 →
  books_A = 3 * books_B + 30 →
  books_C = books_B - 15 →
  books_A + books_B + books_C = total →
  books_A = 165 := by
sorry

end NUMINAMATH_CALUDE_books_in_pile_A_l1504_150446


namespace NUMINAMATH_CALUDE_museum_entrance_cost_l1504_150477

theorem museum_entrance_cost (group_size : ℕ) (ticket_price : ℚ) (tax_rate : ℚ) : 
  group_size = 25 →
  ticket_price = 35.91 →
  tax_rate = 0.05 →
  (group_size : ℚ) * ticket_price * (1 + tax_rate) = 942.64 := by
sorry

end NUMINAMATH_CALUDE_museum_entrance_cost_l1504_150477


namespace NUMINAMATH_CALUDE_a_squared_gt_one_sufficient_not_necessary_l1504_150413

-- Define the equation
def is_ellipse (a : ℝ) : Prop :=
  ∃ x y : ℝ, x^2 / a^2 + y^2 = 1 ∧ (x ≠ 0 ∨ y ≠ 0)

-- Theorem statement
theorem a_squared_gt_one_sufficient_not_necessary :
  (∀ a : ℝ, a^2 > 1 → is_ellipse a) ∧
  (∃ a : ℝ, is_ellipse a ∧ a^2 ≤ 1) :=
sorry

end NUMINAMATH_CALUDE_a_squared_gt_one_sufficient_not_necessary_l1504_150413


namespace NUMINAMATH_CALUDE_total_money_proof_l1504_150462

/-- The total amount of money p, q, and r have among themselves -/
def total_amount (r_amount : ℚ) (r_fraction : ℚ) : ℚ :=
  r_amount / r_fraction

theorem total_money_proof (r_amount : ℚ) (h1 : r_amount = 2000) 
  (r_fraction : ℚ) (h2 : r_fraction = 2/3) : 
  total_amount r_amount r_fraction = 5000 := by
  sorry

#check total_money_proof

end NUMINAMATH_CALUDE_total_money_proof_l1504_150462


namespace NUMINAMATH_CALUDE_meadow_grazing_l1504_150465

/-- Represents the amount of grass one cow eats per day -/
def daily_cow_consumption : ℝ := sorry

/-- Represents the amount of grass that grows on the meadow per day -/
def daily_grass_growth : ℝ := sorry

/-- Represents the initial amount of grass in the meadow -/
def initial_grass : ℝ := sorry

/-- Condition: 9 cows will graze the meadow empty in 4 days -/
axiom condition1 : initial_grass + 4 * daily_grass_growth = 9 * 4 * daily_cow_consumption

/-- Condition: 8 cows will graze the meadow empty in 6 days -/
axiom condition2 : initial_grass + 6 * daily_grass_growth = 8 * 6 * daily_cow_consumption

/-- The number of cows that can graze continuously in the meadow -/
def continuous_grazing_cows : ℕ := 6

theorem meadow_grazing :
  daily_grass_growth = continuous_grazing_cows * daily_cow_consumption :=
sorry

end NUMINAMATH_CALUDE_meadow_grazing_l1504_150465


namespace NUMINAMATH_CALUDE_fraction_equality_l1504_150476

theorem fraction_equality (p q s u : ℚ) 
  (h1 : p / q = 5 / 6) 
  (h2 : s / u = 7 / 18) : 
  (5 * p * s - 6 * q * u) / (7 * q * u - 10 * p * s) = -473 / 406 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1504_150476


namespace NUMINAMATH_CALUDE_mn_value_l1504_150444

theorem mn_value (m n : ℤ) (h : |3*m - 6| + (n + 4)^2 = 0) : m * n = -8 := by
  sorry

end NUMINAMATH_CALUDE_mn_value_l1504_150444


namespace NUMINAMATH_CALUDE_high_school_student_distribution_l1504_150424

theorem high_school_student_distribution :
  ∀ (freshmen sophomores juniors seniors : ℕ),
    freshmen + sophomores + juniors + seniors = 800 →
    juniors = 216 →
    sophomores = 200 →
    seniors = 160 →
    freshmen - sophomores = 24 := by
  sorry

end NUMINAMATH_CALUDE_high_school_student_distribution_l1504_150424


namespace NUMINAMATH_CALUDE_cans_per_carton_l1504_150403

theorem cans_per_carton (total_cartons : ℕ) (loaded_cartons : ℕ) (remaining_cans : ℕ) :
  total_cartons = 50 →
  loaded_cartons = 40 →
  remaining_cans = 200 →
  (total_cartons - loaded_cartons) * (remaining_cans / (total_cartons - loaded_cartons)) = remaining_cans :=
by sorry

end NUMINAMATH_CALUDE_cans_per_carton_l1504_150403


namespace NUMINAMATH_CALUDE_caroline_lassi_production_l1504_150430

/-- Given that Caroline can make 15 lassis out of 3 mangoes, 
    prove that she can make 75 lassis out of 15 mangoes. -/
theorem caroline_lassi_production :
  (∃ (lassis_per_3_mangoes : ℕ), lassis_per_3_mangoes = 15) →
  (∃ (lassis_per_15_mangoes : ℕ), lassis_per_15_mangoes = 75) :=
by
  sorry

end NUMINAMATH_CALUDE_caroline_lassi_production_l1504_150430


namespace NUMINAMATH_CALUDE_matrix_commutation_l1504_150492

open Matrix

theorem matrix_commutation (A B : Matrix (Fin 2) (Fin 2) ℝ) 
  (h1 : A + B = A * B) 
  (h2 : A * B = ![![5, 1], ![-2, 2]]) : 
  B * A = A * B := by sorry

end NUMINAMATH_CALUDE_matrix_commutation_l1504_150492


namespace NUMINAMATH_CALUDE_coefficient_x_squared_in_expansion_l1504_150437

theorem coefficient_x_squared_in_expansion :
  (Finset.range 6).sum (fun k => (Nat.choose 5 k) * 2^k * (if k = 2 then 1 else 0)) = 40 :=
sorry

end NUMINAMATH_CALUDE_coefficient_x_squared_in_expansion_l1504_150437


namespace NUMINAMATH_CALUDE_circle_equation_l1504_150473

/-- The line to which the circle is tangent -/
def tangent_line (x y : ℝ) : Prop := x + y - 2 = 0

/-- A circle centered at the origin -/
def circle_at_origin (x y r : ℝ) : Prop := x^2 + y^2 = r^2

/-- The circle is tangent to the line -/
def is_tangent (r : ℝ) : Prop := ∃ x y : ℝ, tangent_line x y ∧ circle_at_origin x y r

theorem circle_equation : 
  ∃ r : ℝ, is_tangent r → circle_at_origin x y 2 :=
sorry

end NUMINAMATH_CALUDE_circle_equation_l1504_150473


namespace NUMINAMATH_CALUDE_min_value_theorem_l1504_150489

theorem min_value_theorem (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h : x * y * z = 27) :
  ∃ (min : ℝ), min = 30 ∧ ∀ (a b c : ℝ), a > 0 → b > 0 → c > 0 → a * b * c = 27 → 
    a^2 + 3*b + 6*c ≥ min := by
  sorry

#check min_value_theorem

end NUMINAMATH_CALUDE_min_value_theorem_l1504_150489


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1504_150460

theorem inequality_solution_set (x : ℝ) :
  (3 * x - 1) / (x - 2) > 0 ↔ x < 1/3 ∨ x > 2 :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1504_150460


namespace NUMINAMATH_CALUDE_drops_used_proof_l1504_150433

/-- Represents the number of drops used to test a single beaker -/
def drops_per_beaker : ℕ := 3

/-- Represents the total number of beakers -/
def total_beakers : ℕ := 22

/-- Represents the number of beakers with copper ions -/
def copper_beakers : ℕ := 8

/-- Represents the number of beakers without copper ions that were tested -/
def tested_non_copper : ℕ := 7

theorem drops_used_proof :
  drops_per_beaker * (copper_beakers + tested_non_copper) = 45 := by
  sorry

end NUMINAMATH_CALUDE_drops_used_proof_l1504_150433


namespace NUMINAMATH_CALUDE_inequalities_problem_l1504_150453

theorem inequalities_problem (a b : ℝ) (h : 1/a < 1/b ∧ 1/b < 0) :
  (a + b < a * b) ∧
  (b/a + a/b > 2) ∧
  (a > b) ∧
  (abs a < abs b) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_problem_l1504_150453


namespace NUMINAMATH_CALUDE_smallest_number_l1504_150432

/-- Given three numbers A, B, and C with the following properties:
  - A is 38 greater than 18
  - B is 26 less than A
  - C is the quotient of B divided by 3
  Prove that C is the smallest among A, B, and C. -/
theorem smallest_number (A B C : ℤ) 
  (h1 : A = 18 + 38)
  (h2 : B = A - 26)
  (h3 : C = B / 3) :
  C ≤ A ∧ C ≤ B := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_l1504_150432


namespace NUMINAMATH_CALUDE_soda_bottle_difference_l1504_150435

theorem soda_bottle_difference :
  let diet_soda : ℕ := 4
  let regular_soda : ℕ := 83
  regular_soda - diet_soda = 79 :=
by sorry

end NUMINAMATH_CALUDE_soda_bottle_difference_l1504_150435


namespace NUMINAMATH_CALUDE_point_A_in_transformed_plane_l1504_150482

/-- The similarity transformation coefficient -/
def k : ℚ := 1/2

/-- The original plane equation: 4x - 3y + 5z - 10 = 0 -/
def plane_a (x y z : ℚ) : Prop := 4*x - 3*y + 5*z - 10 = 0

/-- The transformed plane equation: 4x - 3y + 5z - 5 = 0 -/
def plane_a' (x y z : ℚ) : Prop := 4*x - 3*y + 5*z - 5 = 0

/-- Point A -/
def point_A : ℚ × ℚ × ℚ := (1/4, 1/3, 1)

/-- Theorem: Point A belongs to the image of plane a under the similarity transformation -/
theorem point_A_in_transformed_plane :
  plane_a' point_A.1 point_A.2.1 point_A.2.2 :=
by sorry

end NUMINAMATH_CALUDE_point_A_in_transformed_plane_l1504_150482


namespace NUMINAMATH_CALUDE_jewelry_sweater_difference_l1504_150423

theorem jewelry_sweater_difference (sweater_cost initial_fraction remaining : ℚ) :
  sweater_cost = 40 →
  initial_fraction = 1/4 →
  remaining = 20 →
  let initial_money := sweater_cost / initial_fraction
  let jewelry_cost := initial_money - sweater_cost - remaining
  jewelry_cost - sweater_cost = 60 := by
  sorry

end NUMINAMATH_CALUDE_jewelry_sweater_difference_l1504_150423


namespace NUMINAMATH_CALUDE_circle_c_properties_l1504_150428

-- Define the circle C
structure CircleC where
  center : ℝ × ℝ
  radius : ℝ

-- Define the line l
structure LineL where
  b : ℝ

-- Define point N
def pointN : ℝ × ℝ := (0, 3)

-- Define the theorem
theorem circle_c_properties (c : CircleC) (l : LineL) :
  -- Condition 1: Circle C's center is on the line x - 2y = 0
  c.center.1 = 2 * c.center.2 →
  -- Condition 2: Circle C is tangent to the positive half of the y-axis
  c.center.2 > 0 →
  -- Condition 3: The chord obtained by intersecting the x-axis is 2√3 long
  2 * Real.sqrt 3 = 2 * Real.sqrt (c.radius^2 - c.center.2^2) →
  -- Condition 4: Line l intersects circle C at two points
  ∃ (A B : ℝ × ℝ), A ≠ B ∧ 
    (A.1 - c.center.1)^2 + (A.2 - c.center.2)^2 = c.radius^2 ∧
    (B.1 - c.center.1)^2 + (B.2 - c.center.2)^2 = c.radius^2 ∧
    A.2 = -2 * A.1 + l.b ∧ B.2 = -2 * B.1 + l.b →
  -- Condition 5: The circle with AB as its diameter passes through the origin
  ∃ (A B : ℝ × ℝ), A ≠ B ∧ 
    (A.1 - c.center.1)^2 + (A.2 - c.center.2)^2 = c.radius^2 ∧
    (B.1 - c.center.1)^2 + (B.2 - c.center.2)^2 = c.radius^2 ∧
    A.2 = -2 * A.1 + l.b ∧ B.2 = -2 * B.1 + l.b ∧
    A.1 * B.1 + A.2 * B.2 = 0 →
  -- Condition 6-9 are implicitly included in the structure of CircleC
  -- Prove:
  -- 1. The standard equation of circle C is (x - 2)² + (y - 1)² = 4
  ((c.center = (2, 1) ∧ c.radius = 2) ∨
  -- 2. The value of b in the equation y = -2x + b is (5 ± √15) / 2
   (l.b = (5 + Real.sqrt 15) / 2 ∨ l.b = (5 - Real.sqrt 15) / 2)) ∧
  -- 3. The y-coordinate of the center of circle C is in the range (0, 2]
   (0 < c.center.2 ∧ c.center.2 ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_circle_c_properties_l1504_150428


namespace NUMINAMATH_CALUDE_line_passes_through_circle_center_l1504_150467

-- Define the line equation
def line_equation (x y m : ℝ) : Prop := x - y + m = 0

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 2*y = 0

-- Define the center of the circle
def circle_center (x y : ℝ) : Prop := circle_equation x y ∧ ∀ x' y', circle_equation x' y' → (x - x')^2 + (y - y')^2 ≤ (x' - x)^2 + (y' - y)^2

-- Theorem statement
theorem line_passes_through_circle_center :
  ∃ x y : ℝ, circle_center x y ∧ line_equation x y (-3) :=
sorry

end NUMINAMATH_CALUDE_line_passes_through_circle_center_l1504_150467


namespace NUMINAMATH_CALUDE_mobile_payment_probability_l1504_150479

def group_size : ℕ := 10

def mobile_payment_prob (p : ℝ) : ℝ := p

def is_independent (p : ℝ) : Prop := true

def num_mobile_users (X : ℕ) : ℕ := X

def variance (X : ℕ) (p : ℝ) : ℝ := group_size * p * (1 - p)

def prob_X_eq (k : ℕ) (p : ℝ) : ℝ :=
  (Nat.choose group_size k : ℝ) * p^k * (1 - p)^(group_size - k)

theorem mobile_payment_probability :
  ∀ p : ℝ,
    0 ≤ p ∧ p ≤ 1 →
    is_independent p →
    variance (num_mobile_users X) p = 2.4 →
    prob_X_eq 4 p < prob_X_eq 6 p →
    p = 0.6 := by
  sorry

end NUMINAMATH_CALUDE_mobile_payment_probability_l1504_150479


namespace NUMINAMATH_CALUDE_sin_cos_pi_12_l1504_150481

theorem sin_cos_pi_12 : Real.sin (π / 12) * Real.cos (π / 12) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_pi_12_l1504_150481


namespace NUMINAMATH_CALUDE_eleven_points_form_120_triangles_l1504_150458

/-- The number of triangles formed by 11 points on two segments -/
def numTriangles (n m : ℕ) : ℕ :=
  n * m * (m - 1) / 2 + m * n * (n - 1) / 2 + (n * (n - 1) * (n - 2)) / 6

/-- Theorem stating that 11 points on two segments (7 on one, 4 on another) form 120 triangles -/
theorem eleven_points_form_120_triangles :
  numTriangles 7 4 = 120 := by
  sorry

end NUMINAMATH_CALUDE_eleven_points_form_120_triangles_l1504_150458


namespace NUMINAMATH_CALUDE_scientific_notation_equivalence_l1504_150466

theorem scientific_notation_equivalence :
  ∃ (a : ℝ) (n : ℤ), 
    27017800000000 = a * (10 : ℝ) ^ n ∧ 
    1 ≤ a ∧ a < 10 ∧
    n = 13 ∧
    a = 2.70178 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_equivalence_l1504_150466


namespace NUMINAMATH_CALUDE_max_at_2_implies_c_6_l1504_150447

/-- The function f(x) = x(x-c)² has a maximum value at x = 2 -/
def has_max_at_2 (c : ℝ) : Prop :=
  let f := fun x => x * (x - c)^2
  ∀ x, f x ≤ f 2

/-- Theorem: If f(x) = x(x-c)² has a maximum value at x = 2, then c = 6 -/
theorem max_at_2_implies_c_6 : 
  ∀ c : ℝ, has_max_at_2 c → c = 6 := by
  sorry

end NUMINAMATH_CALUDE_max_at_2_implies_c_6_l1504_150447


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l1504_150410

/-- An arithmetic sequence with sum of first n terms S_n -/
structure ArithmeticSequence where
  S : ℕ → ℚ
  is_arithmetic : ∀ n : ℕ, 2 * (S (n + 1) - S n) = S (n + 2) - S n

/-- Theorem: If S_5 : S_10 = 2 : 3, then S_15 : S_5 = 3 : 2 -/
theorem arithmetic_sequence_ratio 
  (seq : ArithmeticSequence) 
  (h : seq.S 5 / seq.S 10 = 2 / 3) : 
  seq.S 15 / seq.S 5 = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l1504_150410


namespace NUMINAMATH_CALUDE_decimal_25_to_binary_binary_to_decimal_25_l1504_150497

/-- Represents a binary digit (0 or 1) -/
inductive BinaryDigit
| zero : BinaryDigit
| one : BinaryDigit

/-- Represents a binary number as a list of binary digits -/
def BinaryNumber := List BinaryDigit

/-- Converts a decimal number to its binary representation -/
def decimalToBinary (n : ℕ) : BinaryNumber :=
  sorry

/-- Converts a binary number to its decimal representation -/
def binaryToDecimal (b : BinaryNumber) : ℕ :=
  sorry

theorem decimal_25_to_binary :
  decimalToBinary 25 = [BinaryDigit.one, BinaryDigit.one, BinaryDigit.zero, BinaryDigit.zero, BinaryDigit.one] :=
by sorry

theorem binary_to_decimal_25 :
  binaryToDecimal [BinaryDigit.one, BinaryDigit.one, BinaryDigit.zero, BinaryDigit.zero, BinaryDigit.one] = 25 :=
by sorry

end NUMINAMATH_CALUDE_decimal_25_to_binary_binary_to_decimal_25_l1504_150497


namespace NUMINAMATH_CALUDE_average_age_decrease_l1504_150407

/-- Proves that the average age of a class decreases by 4 years when new students join --/
theorem average_age_decrease (original_strength original_average new_students new_average : ℕ) :
  original_strength = 12 →
  original_average = 40 →
  new_students = 12 →
  new_average = 32 →
  let total_age_before := original_strength * original_average
  let total_age_new := new_students * new_average
  let total_age_after := total_age_before + total_age_new
  let new_strength := original_strength + new_students
  let new_average_age := total_age_after / new_strength
  original_average - new_average_age = 4 := by
  sorry

end NUMINAMATH_CALUDE_average_age_decrease_l1504_150407


namespace NUMINAMATH_CALUDE_island_population_theorem_l1504_150472

/-- Represents the number of turtles and rabbits on an island -/
structure IslandPopulation where
  turtles : ℕ
  rabbits : ℕ

/-- Represents the populations of the four islands -/
structure IslandSystem where
  happy : IslandPopulation
  lonely : IslandPopulation
  serene : IslandPopulation
  tranquil : IslandPopulation

/-- Theorem stating the conditions and the result to be proven -/
theorem island_population_theorem (islands : IslandSystem) : 
  (islands.happy.turtles = 120) →
  (islands.happy.rabbits = 80) →
  (islands.lonely.turtles = islands.happy.turtles / 3) →
  (islands.lonely.rabbits = islands.lonely.turtles) →
  (islands.serene.rabbits = 2 * islands.lonely.rabbits) →
  (islands.serene.turtles = 3 * islands.lonely.rabbits / 4) →
  (islands.tranquil.turtles = islands.tranquil.rabbits) →
  (islands.tranquil.turtles = 
    (islands.happy.turtles - islands.serene.turtles) + 5) →
  (islands.happy.turtles + islands.lonely.turtles + 
   islands.serene.turtles + islands.tranquil.turtles = 285) ∧
  (islands.happy.rabbits + islands.lonely.rabbits + 
   islands.serene.rabbits + islands.tranquil.rabbits = 295) := by
  sorry

end NUMINAMATH_CALUDE_island_population_theorem_l1504_150472


namespace NUMINAMATH_CALUDE_sum_of_digits_of_product_of_repeated_digits_l1504_150487

/-- The integer consisting of n repetitions of digit d in base 10 -/
def repeatedDigit (d : ℕ) (n : ℕ) : ℕ :=
  d * (10^n - 1) / 9

/-- The sum of digits of a natural number in base 10 -/
def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

theorem sum_of_digits_of_product_of_repeated_digits :
  let a := repeatedDigit 6 1000
  let b := repeatedDigit 7 1000
  sumOfDigits (9 * a * b) = 19986 := by
sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_product_of_repeated_digits_l1504_150487


namespace NUMINAMATH_CALUDE_charge_account_interest_l1504_150499

/-- Calculates the total amount owed after one year with simple interest -/
def total_amount_owed (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal + principal * rate * time

/-- Proves that the total amount owed after one year on a $75 charge with 7% simple annual interest is $80.25 -/
theorem charge_account_interest : total_amount_owed 75 0.07 1 = 80.25 := by
  sorry

end NUMINAMATH_CALUDE_charge_account_interest_l1504_150499


namespace NUMINAMATH_CALUDE_birthday_friends_count_l1504_150400

theorem birthday_friends_count : ∃ (n : ℕ), 
  (12 * (n + 2) = 16 * n) ∧ 
  (∀ m : ℕ, 12 * (m + 2) = 16 * m → m = n) :=
by sorry

end NUMINAMATH_CALUDE_birthday_friends_count_l1504_150400


namespace NUMINAMATH_CALUDE_solution_set_f_range_of_m_l1504_150491

-- Define the function f
def f (x : ℝ) : ℝ := |x - 1|

-- Part I
theorem solution_set_f (x : ℝ) : 
  (|f x - 3| ≤ 4) ↔ (-6 ≤ x ∧ x ≤ 8) := by sorry

-- Part II
theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, f x + f (x + 3) ≥ m^2 - 2*m) ↔ (-1 ≤ m ∧ m ≤ 3) := by sorry

end NUMINAMATH_CALUDE_solution_set_f_range_of_m_l1504_150491


namespace NUMINAMATH_CALUDE_triangle_area_from_perimeter_and_inradius_l1504_150485

/-- The area of a triangle with given perimeter and inradius -/
theorem triangle_area_from_perimeter_and_inradius 
  (perimeter : ℝ) (inradius : ℝ) (area : ℝ) 
  (h_perimeter : perimeter = 20) 
  (h_inradius : inradius = 3) : 
  area = 30 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_from_perimeter_and_inradius_l1504_150485


namespace NUMINAMATH_CALUDE_first_negative_term_is_14th_l1504_150438

/-- The index of the first negative term in the arithmetic sequence -/
def first_negative_term_index : ℕ := 14

/-- The first term of the arithmetic sequence -/
def a₁ : ℤ := 51

/-- The common difference of the arithmetic sequence -/
def d : ℤ := -4

/-- The general term of the arithmetic sequence -/
def aₙ (n : ℕ) : ℤ := a₁ + (n - 1 : ℤ) * d

theorem first_negative_term_is_14th :
  (∀ k < first_negative_term_index, aₙ k ≥ 0) ∧
  aₙ first_negative_term_index < 0 := by
  sorry

#eval aₙ first_negative_term_index

end NUMINAMATH_CALUDE_first_negative_term_is_14th_l1504_150438


namespace NUMINAMATH_CALUDE_daniel_video_game_collection_l1504_150439

/-- The number of video games Daniel bought for $12 each -/
def games_at_12 : ℕ := 80

/-- The price of the first group of games -/
def price_1 : ℕ := 12

/-- The price of the second group of games -/
def price_2 : ℕ := 7

/-- The price of the third group of games -/
def price_3 : ℕ := 3

/-- The total amount Daniel spent on all games -/
def total_spent : ℕ := 2290

/-- Theorem stating the total number of video games in Daniel's collection -/
theorem daniel_video_game_collection :
  ∃ (games_at_7 games_at_3 : ℕ),
    games_at_7 = games_at_3 ∧
    games_at_12 * price_1 + games_at_7 * price_2 + games_at_3 * price_3 = total_spent ∧
    games_at_12 + games_at_7 + games_at_3 = 346 :=
by sorry

end NUMINAMATH_CALUDE_daniel_video_game_collection_l1504_150439


namespace NUMINAMATH_CALUDE_total_red_stripes_l1504_150484

/-- Calculates the number of red stripes in Flag A -/
def red_stripes_a (total_stripes : ℕ) : ℕ :=
  1 + (total_stripes - 1) / 2

/-- Calculates the number of red stripes in Flag B -/
def red_stripes_b (total_stripes : ℕ) : ℕ :=
  total_stripes / 3

/-- Calculates the number of red stripes in Flag C -/
def red_stripes_c (total_stripes : ℕ) : ℕ :=
  let full_patterns := total_stripes / 9
  let remaining_stripes := total_stripes % 9
  2 * full_patterns + min remaining_stripes 2

/-- The main theorem stating the total number of red stripes -/
theorem total_red_stripes :
  let flag_a_count := 20
  let flag_b_count := 30
  let flag_c_count := 40
  let flag_a_stripes := 30
  let flag_b_stripes := 45
  let flag_c_stripes := 60
  flag_a_count * red_stripes_a flag_a_stripes +
  flag_b_count * red_stripes_b flag_b_stripes +
  flag_c_count * red_stripes_c flag_c_stripes = 1310 := by
  sorry

end NUMINAMATH_CALUDE_total_red_stripes_l1504_150484


namespace NUMINAMATH_CALUDE_hostel_problem_l1504_150449

/-- Calculates the number of men who left a hostel given the initial conditions and the new duration of provisions. -/
def men_who_left (initial_men : ℕ) (initial_days : ℕ) (new_days : ℕ) : ℕ :=
  initial_men - (initial_men * initial_days) / new_days

/-- Proves that 50 men left the hostel under the given conditions. -/
theorem hostel_problem : men_who_left 250 48 60 = 50 := by
  sorry

end NUMINAMATH_CALUDE_hostel_problem_l1504_150449


namespace NUMINAMATH_CALUDE_billy_ice_trays_l1504_150451

theorem billy_ice_trays (ice_cubes_per_tray : ℕ) (total_ice_cubes : ℕ) 
  (h1 : ice_cubes_per_tray = 9)
  (h2 : total_ice_cubes = 72) :
  total_ice_cubes / ice_cubes_per_tray = 8 := by
  sorry

end NUMINAMATH_CALUDE_billy_ice_trays_l1504_150451


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l1504_150461

-- Define the universal set U as ℝ
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x | x ≥ 1}

-- Define set B
def B : Set ℝ := {x | x ≥ 2}

-- Theorem statement
theorem intersection_A_complement_B :
  A ∩ (Set.compl B) = {x | 1 ≤ x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l1504_150461


namespace NUMINAMATH_CALUDE_closest_integer_to_seven_times_three_fourths_l1504_150490

theorem closest_integer_to_seven_times_three_fourths : 
  ∃ (n : ℤ), ∀ (m : ℤ), |n - (7 * 3 / 4)| ≤ |m - (7 * 3 / 4)| ∧ n = 5 :=
by sorry

end NUMINAMATH_CALUDE_closest_integer_to_seven_times_three_fourths_l1504_150490


namespace NUMINAMATH_CALUDE_subtraction_value_l1504_150475

theorem subtraction_value (x y : ℝ) : 
  (x - 5) / 7 = 7 → (x - y) / 8 = 6 → y = 6 := by
sorry

end NUMINAMATH_CALUDE_subtraction_value_l1504_150475


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l1504_150488

/-- A sequence of real numbers -/
def Sequence := ℕ → ℝ

/-- A sequence is arithmetic if the difference between consecutive terms is constant -/
def is_arithmetic (a : Sequence) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

/-- Points lie on the line y = x + 1 -/
def points_on_line (a : Sequence) : Prop :=
  ∀ n : ℕ+, a n = n + 1

theorem sufficient_not_necessary :
  (∀ a : Sequence, points_on_line a → is_arithmetic a) ∧
  (∃ a : Sequence, is_arithmetic a ∧ ¬points_on_line a) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l1504_150488


namespace NUMINAMATH_CALUDE_kai_born_in_1995_l1504_150443

/-- Kai's birth year, given his 25th birthday is in March 2020 -/
def kais_birth_year : ℕ := sorry

/-- The year of Kai's 25th birthday -/
def birthday_year : ℕ := 2020

/-- Kai's age at his birthday in 2020 -/
def kais_age : ℕ := 25

theorem kai_born_in_1995 : kais_birth_year = 1995 := by
  sorry

end NUMINAMATH_CALUDE_kai_born_in_1995_l1504_150443


namespace NUMINAMATH_CALUDE_marble_problem_l1504_150457

theorem marble_problem : 
  ∀ (x : ℚ), x > 0 →
  let bag1 := x
  let bag2 := 2 * x
  let bag3 := 3 * x
  let green1 := (1 / 2) * bag1
  let green2 := (1 / 3) * bag2
  let green3 := (1 / 4) * bag3
  let total_green := green1 + green2 + green3
  let total_marbles := bag1 + bag2 + bag3
  (total_green / total_marbles) = 23 / 72 :=
by
  sorry

end NUMINAMATH_CALUDE_marble_problem_l1504_150457


namespace NUMINAMATH_CALUDE_polynomial_negative_l1504_150440

theorem polynomial_negative (a : ℝ) (x : ℝ) (h : 0 < x ∧ x < a) : 
  (a - x)^6 - 3*a*(a - x)^5 + (5/2)*a^2*(a - x)^4 - (1/2)*a^4*(a - x)^2 < 0 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_negative_l1504_150440


namespace NUMINAMATH_CALUDE_largest_common_divisor_414_345_l1504_150498

theorem largest_common_divisor_414_345 : Nat.gcd 414 345 = 69 := by
  sorry

end NUMINAMATH_CALUDE_largest_common_divisor_414_345_l1504_150498


namespace NUMINAMATH_CALUDE_spider_reachable_points_l1504_150471

/-- A cube with edge length 1 -/
structure Cube where
  edge_length : ℝ
  edge_length_pos : edge_length = 1

/-- A point on the surface of a cube -/
structure CubePoint (c : Cube) where
  x : ℝ
  y : ℝ
  z : ℝ
  on_surface : (x = 0 ∨ x = c.edge_length ∨ y = 0 ∨ y = c.edge_length ∨ z = 0 ∨ z = c.edge_length) ∧
               0 ≤ x ∧ x ≤ c.edge_length ∧
               0 ≤ y ∧ y ≤ c.edge_length ∧
               0 ≤ z ∧ z ≤ c.edge_length

/-- The distance between two points on the surface of a cube -/
def surface_distance (c : Cube) (p1 p2 : CubePoint c) : ℝ :=
  sorry -- Definition of surface distance calculation

/-- The set of points reachable by the spider in 2 seconds -/
def reachable_points (c : Cube) (start : CubePoint c) : Set (CubePoint c) :=
  {p : CubePoint c | surface_distance c start p ≤ 2}

/-- Theorem: The set of points reachable by the spider in 2 seconds
    is equivalent to the set of points within 2 cm of the starting vertex -/
theorem spider_reachable_points (c : Cube) (start : CubePoint c) :
  reachable_points c start = {p : CubePoint c | surface_distance c start p ≤ 2} :=
by sorry

end NUMINAMATH_CALUDE_spider_reachable_points_l1504_150471


namespace NUMINAMATH_CALUDE_blue_cap_cost_l1504_150405

/-- The cost of items before applying a discount --/
structure PreDiscountCost where
  tshirt : ℕ
  backpack : ℕ
  bluecap : ℕ

/-- The total cost after applying a discount --/
def total_after_discount (cost : PreDiscountCost) (discount : ℕ) : ℕ :=
  cost.tshirt + cost.backpack + cost.bluecap - discount

/-- The theorem stating the cost of the blue cap --/
theorem blue_cap_cost (cost : PreDiscountCost) (discount : ℕ) :
  cost.tshirt = 30 →
  cost.backpack = 10 →
  discount = 2 →
  total_after_discount cost discount = 43 →
  cost.bluecap = 5 := by
  sorry

#check blue_cap_cost

end NUMINAMATH_CALUDE_blue_cap_cost_l1504_150405


namespace NUMINAMATH_CALUDE_arithmetic_sequence_1005th_term_l1504_150456

/-- An arithmetic sequence with specific conditions -/
def ArithmeticSequence (p r : ℚ) : ℕ → ℚ
  | 0 => p
  | 1 => 10
  | 2 => 4 * p - r
  | 3 => 4 * p + r
  | n + 4 => ArithmeticSequence p r 3 + (n + 1) * (ArithmeticSequence p r 3 - ArithmeticSequence p r 2)

/-- The 1005th term of the arithmetic sequence is 5480 -/
theorem arithmetic_sequence_1005th_term (p r : ℚ) :
  ArithmeticSequence p r 1004 = 5480 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_1005th_term_l1504_150456


namespace NUMINAMATH_CALUDE_sum_of_solutions_quadratic_l1504_150408

theorem sum_of_solutions_quadratic (x : ℝ) : 
  (x^2 - 6*x + 5 = 2*x - 8) → 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 - 6*x₁ + 5 = 2*x₁ - 8 ∧ x₂^2 - 6*x₂ + 5 = 2*x₂ - 8 ∧ x₁ + x₂ = 8) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_solutions_quadratic_l1504_150408


namespace NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l1504_150436

-- Define condition p
def condition_p (x y : ℝ) : Prop := x > 2 ∧ y > 3

-- Define condition q
def condition_q (x y : ℝ) : Prop := x + y > 5 ∧ x * y > 6

-- Theorem stating that p is sufficient but not necessary for q
theorem p_sufficient_not_necessary_for_q :
  (∀ x y : ℝ, condition_p x y → condition_q x y) ∧
  ¬(∀ x y : ℝ, condition_q x y → condition_p x y) :=
by sorry

end NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l1504_150436


namespace NUMINAMATH_CALUDE_sin_30_degrees_l1504_150402

theorem sin_30_degrees : Real.sin (π / 6) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_30_degrees_l1504_150402


namespace NUMINAMATH_CALUDE_broken_crayons_percentage_l1504_150469

theorem broken_crayons_percentage (total : ℕ) (slightly_used : ℕ) :
  total = 120 →
  slightly_used = 56 →
  (total / 3 : ℚ) + slightly_used + (total / 5 : ℚ) = total →
  (total / 5 : ℚ) / total * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_broken_crayons_percentage_l1504_150469


namespace NUMINAMATH_CALUDE_three_zeros_l1504_150404

noncomputable def f (a b x : ℝ) : ℝ := (1/2) * a * x^2 - (a^2 + a + 2) * x + (2*a + 2) * Real.log x + b

theorem three_zeros (a b : ℝ) (ha : a > 3) (hb : a^2 + a + 1 < b) (hb' : b < 2*a^2 - 2*a + 2) :
  ∃! (s : Finset ℝ), s.card = 3 ∧ ∀ x ∈ s, f a b x = 0 :=
sorry

end NUMINAMATH_CALUDE_three_zeros_l1504_150404


namespace NUMINAMATH_CALUDE_carpet_length_proof_l1504_150416

theorem carpet_length_proof (length width diagonal : ℝ) : 
  length > 0 ∧ width > 0 ∧
  length * width = 60 ∧
  diagonal + length = 5 * width ∧
  diagonal^2 = length^2 + width^2 →
  length = 2 * Real.sqrt 30 :=
by sorry

end NUMINAMATH_CALUDE_carpet_length_proof_l1504_150416


namespace NUMINAMATH_CALUDE_unique_valid_quintuple_l1504_150401

/-- A quintuple of nonnegative real numbers satisfying the given conditions -/
structure ValidQuintuple where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  e : ℝ
  nonneg_a : 0 ≤ a
  nonneg_b : 0 ≤ b
  nonneg_c : 0 ≤ c
  nonneg_d : 0 ≤ d
  nonneg_e : 0 ≤ e
  condition1 : a^2 + b^2 + c^3 + d^3 + e^3 = 5
  condition2 : (a + b + c + d + e) * (a^3 + b^3 + c^2 + d^2 + e^2) = 25

/-- There exists exactly one valid quintuple -/
theorem unique_valid_quintuple : ∃! q : ValidQuintuple, True :=
  sorry

end NUMINAMATH_CALUDE_unique_valid_quintuple_l1504_150401
