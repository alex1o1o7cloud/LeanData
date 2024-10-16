import Mathlib

namespace NUMINAMATH_CALUDE_group_frequency_problem_l1804_180453

/-- A problem about frequency and relative frequency in a grouped sample -/
theorem group_frequency_problem (total_sample : ℕ) (num_groups : ℕ) 
  (group_frequencies : Fin 8 → ℕ) :
  total_sample = 100 →
  num_groups = 8 →
  group_frequencies 0 = 10 →
  group_frequencies 1 = 13 →
  group_frequencies 3 = 14 →
  group_frequencies 4 = 15 →
  group_frequencies 5 = 13 →
  group_frequencies 6 = 12 →
  group_frequencies 7 = 9 →
  group_frequencies 2 = 14 ∧ 
  (group_frequencies 2 : ℚ) / total_sample = 14 / 100 :=
by sorry

end NUMINAMATH_CALUDE_group_frequency_problem_l1804_180453


namespace NUMINAMATH_CALUDE_rate_categories_fractions_l1804_180451

/-- Represents the three rate categories for electricity usage --/
inductive RateCategory
  | A
  | B
  | C

/-- Total hours in a week --/
def hoursInWeek : ℕ := 7 * 24

/-- Hours that Category A applies in a week --/
def categoryAHours : ℕ := 12 * 5

/-- Hours that Category B applies in a week --/
def categoryBHours : ℕ := 10 * 2

/-- Hours that Category C applies in a week --/
def categoryCHours : ℕ := hoursInWeek - (categoryAHours + categoryBHours)

/-- Function to get the fraction of the week a category applies to --/
def categoryFraction (c : RateCategory) : ℚ :=
  match c with
  | RateCategory.A => categoryAHours / hoursInWeek
  | RateCategory.B => categoryBHours / hoursInWeek
  | RateCategory.C => categoryCHours / hoursInWeek

theorem rate_categories_fractions :
  categoryFraction RateCategory.A = 5 / 14 ∧
  categoryFraction RateCategory.B = 5 / 42 ∧
  categoryFraction RateCategory.C = 11 / 21 ∧
  categoryFraction RateCategory.A + categoryFraction RateCategory.B + categoryFraction RateCategory.C = 1 := by
  sorry


end NUMINAMATH_CALUDE_rate_categories_fractions_l1804_180451


namespace NUMINAMATH_CALUDE_insufficient_info_for_both_correct_evans_class_test_l1804_180487

theorem insufficient_info_for_both_correct (total_students : ℕ) 
  (q1_correct : ℕ) (absent : ℕ) (q2_correct : ℕ) : Prop :=
  total_students = 40 ∧ 
  q1_correct = 30 ∧ 
  absent = 10 ∧
  q2_correct ≥ 0 ∧ q2_correct ≤ (total_students - absent) →
  ∃ (both_correct₁ both_correct₂ : ℕ), 
    both_correct₁ ≠ both_correct₂ ∧
    both_correct₁ ≥ 0 ∧ both_correct₁ ≤ q1_correct ∧
    both_correct₂ ≥ 0 ∧ both_correct₂ ≤ q1_correct ∧
    both_correct₁ ≤ q2_correct ∧ both_correct₂ ≤ q2_correct

theorem evans_class_test : insufficient_info_for_both_correct 40 30 10 q2_correct :=
sorry

end NUMINAMATH_CALUDE_insufficient_info_for_both_correct_evans_class_test_l1804_180487


namespace NUMINAMATH_CALUDE_employee_count_sum_l1804_180447

theorem employee_count_sum : 
  (Finset.sum (Finset.filter (fun s => 200 ≤ s ∧ s ≤ 300 ∧ (s - 1) % 7 = 0) (Finset.range 301)) id) = 3493 :=
by sorry

end NUMINAMATH_CALUDE_employee_count_sum_l1804_180447


namespace NUMINAMATH_CALUDE_prob_not_all_same_five_eight_sided_dice_l1804_180416

/-- The number of sides on each die -/
def n : ℕ := 8

/-- The number of dice rolled -/
def k : ℕ := 5

/-- The probability of not getting all the same numbers when rolling k n-sided dice -/
def prob_not_all_same (n k : ℕ) : ℚ :=
  1 - (n : ℚ) / (n ^ k : ℚ)

/-- Theorem: The probability of not getting all the same numbers when rolling 
    five fair 8-sided dice is 4095/4096 -/
theorem prob_not_all_same_five_eight_sided_dice :
  prob_not_all_same n k = 4095 / 4096 := by sorry

end NUMINAMATH_CALUDE_prob_not_all_same_five_eight_sided_dice_l1804_180416


namespace NUMINAMATH_CALUDE_connors_garage_wheels_l1804_180455

/-- The number of wheels in Connor's garage -/
def total_wheels (bicycles cars motorcycles : ℕ) : ℕ :=
  2 * bicycles + 4 * cars + 2 * motorcycles

/-- Theorem: The total number of wheels in Connor's garage is 90 -/
theorem connors_garage_wheels :
  total_wheels 20 10 5 = 90 := by
  sorry

end NUMINAMATH_CALUDE_connors_garage_wheels_l1804_180455


namespace NUMINAMATH_CALUDE_painting_scheme_combinations_l1804_180464

def number_of_color_choices : ℕ := 10
def colors_to_choose : ℕ := 2
def number_of_texture_choices : ℕ := 3
def textures_to_choose : ℕ := 1

theorem painting_scheme_combinations :
  (number_of_color_choices.choose colors_to_choose) * (number_of_texture_choices.choose textures_to_choose) = 135 := by
  sorry

end NUMINAMATH_CALUDE_painting_scheme_combinations_l1804_180464


namespace NUMINAMATH_CALUDE_student_selection_plans_l1804_180423

/-- The number of students in the group -/
def total_students : ℕ := 5

/-- The number of students to be selected -/
def selected_students : ℕ := 4

/-- The number of competitions -/
def num_competitions : ℕ := 4

/-- The number of competitions student A cannot participate in -/
def restricted_competitions : ℕ := 2

/-- The number of different plans to select students for competitions -/
def num_plans : ℕ := 72

theorem student_selection_plans :
  (Nat.choose total_students selected_students * Nat.factorial selected_students) +
  (Nat.choose (total_students - 1) (selected_students - 1) *
   Nat.choose (num_competitions - restricted_competitions) 1 *
   Nat.factorial (selected_students - 1)) = num_plans :=
sorry

end NUMINAMATH_CALUDE_student_selection_plans_l1804_180423


namespace NUMINAMATH_CALUDE_kevins_calculation_l1804_180488

theorem kevins_calculation (k : ℝ) : 
  (20 + 1) * (6 + k) = 20 + 1 * 6 + k → 20 + 1 * 6 + k = 21 :=
by
  sorry

end NUMINAMATH_CALUDE_kevins_calculation_l1804_180488


namespace NUMINAMATH_CALUDE_percent_commutation_l1804_180465

theorem percent_commutation (x : ℝ) (h : 0.3 * 0.4 * x = 36) :
  0.4 * 0.3 * x = 0.3 * 0.4 * x :=
by
  sorry

end NUMINAMATH_CALUDE_percent_commutation_l1804_180465


namespace NUMINAMATH_CALUDE_largest_digit_divisible_by_six_l1804_180497

theorem largest_digit_divisible_by_six :
  ∀ M : ℕ, M ≤ 9 →
    (54320 + M).mod 6 = 0 →
    ∀ N : ℕ, N ≤ 9 → N > M →
      (54320 + N).mod 6 ≠ 0 →
    M = 4 :=
by sorry

end NUMINAMATH_CALUDE_largest_digit_divisible_by_six_l1804_180497


namespace NUMINAMATH_CALUDE_negative_four_cubed_equality_l1804_180440

theorem negative_four_cubed_equality : (-4)^3 = -(4^3) := by sorry

end NUMINAMATH_CALUDE_negative_four_cubed_equality_l1804_180440


namespace NUMINAMATH_CALUDE_subtraction_addition_problem_l1804_180473

theorem subtraction_addition_problem :
  ∃! x : ℤ, 3005 - x + 10 = 2705 :=
by sorry

end NUMINAMATH_CALUDE_subtraction_addition_problem_l1804_180473


namespace NUMINAMATH_CALUDE_unique_positive_solution_l1804_180401

/-- The polynomial function f(x) = x^8 + 3x^7 + 6x^6 + 2023x^5 - 2000x^4 -/
def f (x : ℝ) : ℝ := x^8 + 3*x^7 + 6*x^6 + 2023*x^5 - 2000*x^4

/-- The theorem stating that f(x) = 0 has exactly one positive real solution -/
theorem unique_positive_solution : ∃! x : ℝ, x > 0 ∧ f x = 0 := by sorry

end NUMINAMATH_CALUDE_unique_positive_solution_l1804_180401


namespace NUMINAMATH_CALUDE_congruence_square_implies_congruence_or_negative_l1804_180452

theorem congruence_square_implies_congruence_or_negative (x y : ℤ) :
  x^2 ≡ y^2 [ZMOD 239] → (x ≡ y [ZMOD 239] ∨ x ≡ -y [ZMOD 239]) := by
  sorry

end NUMINAMATH_CALUDE_congruence_square_implies_congruence_or_negative_l1804_180452


namespace NUMINAMATH_CALUDE_eugene_sunday_swim_time_l1804_180403

/-- Represents the swim times for Eugene over three days -/
structure SwimTimes where
  sunday : ℕ
  monday : ℕ
  tuesday : ℕ

/-- Calculates the average swim time over three days -/
def averageSwimTime (times : SwimTimes) : ℚ :=
  (times.sunday + times.monday + times.tuesday : ℚ) / 3

theorem eugene_sunday_swim_time :
  ∃ (times : SwimTimes),
    times.monday = 30 ∧
    times.tuesday = 45 ∧
    averageSwimTime times = 34 ∧
    times.sunday = 27 :=
  sorry

end NUMINAMATH_CALUDE_eugene_sunday_swim_time_l1804_180403


namespace NUMINAMATH_CALUDE_min_value_of_a2_plus_b2_l1804_180489

theorem min_value_of_a2_plus_b2 (a b : ℝ) :
  (∃ x : ℝ, x^4 + a*x^3 + b*x^2 + a*x + 1 = 0) →
  (∀ a' b' : ℝ, (∃ x : ℝ, x^4 + a'*x^3 + b'*x^2 + a'*x + 1 = 0) → a'^2 + b'^2 ≥ 4/5) ∧
  (∃ a' b' : ℝ, (∃ x : ℝ, x^4 + a'*x^3 + b'*x^2 + a'*x + 1 = 0) ∧ a'^2 + b'^2 = 4/5) :=
by sorry


end NUMINAMATH_CALUDE_min_value_of_a2_plus_b2_l1804_180489


namespace NUMINAMATH_CALUDE_dog_hare_speed_ratio_challenging_terrain_l1804_180481

/-- Represents the ratio of dog leaps to hare leaps -/
def dogHareLeapRatio : ℚ := 10 / 2

/-- Represents the ratio of dog leap distance to hare leap distance -/
def dogHareDistanceRatio : ℚ := 2 / 1

/-- Represents the reduction factor of dog's leap distance on challenging terrain -/
def dogReductionFactor : ℚ := 3 / 4

/-- Represents the reduction factor of hare's leap distance on challenging terrain -/
def hareReductionFactor : ℚ := 1 / 2

/-- Theorem stating the speed ratio of dog to hare on challenging terrain -/
theorem dog_hare_speed_ratio_challenging_terrain :
  (dogHareDistanceRatio * dogReductionFactor) / hareReductionFactor = 3 / 1 := by
  sorry

end NUMINAMATH_CALUDE_dog_hare_speed_ratio_challenging_terrain_l1804_180481


namespace NUMINAMATH_CALUDE_consecutive_integers_product_l1804_180411

theorem consecutive_integers_product (a b c d e : ℕ) : 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧
  b = a + 1 ∧ c = b + 1 ∧ d = c + 1 ∧ e = d + 1 ∧
  a * b * c * d * e = 15120 →
  e = 9 := by sorry

end NUMINAMATH_CALUDE_consecutive_integers_product_l1804_180411


namespace NUMINAMATH_CALUDE_last_digit_of_2_to_2024_l1804_180470

def last_digit (n : ℕ) : ℕ := n % 10

def power_of_two_last_digit (n : ℕ) : ℕ :=
  match n % 4 with
  | 1 => 2
  | 2 => 4
  | 3 => 8
  | 0 => 6
  | _ => 0  -- This case should never occur, but Lean requires it for completeness

theorem last_digit_of_2_to_2024 :
  last_digit (2^2024) = power_of_two_last_digit 2024 :=
by sorry

end NUMINAMATH_CALUDE_last_digit_of_2_to_2024_l1804_180470


namespace NUMINAMATH_CALUDE_reciprocal_greater_than_one_l1804_180463

theorem reciprocal_greater_than_one (x : ℝ) : 
  (x ≠ 0 ∧ (1 / x) > 1) ↔ (0 < x ∧ x < 1) :=
by sorry

end NUMINAMATH_CALUDE_reciprocal_greater_than_one_l1804_180463


namespace NUMINAMATH_CALUDE_geometric_sequence_single_digit_numbers_l1804_180433

theorem geometric_sequence_single_digit_numbers :
  ∃! (a b c : ℕ),
    a < 10 ∧ b < 10 ∧ c < 10 ∧
    ∃ (q : ℚ),
      b = a * q ∧
      (10 * a + c : ℚ) = a * q^2 ∧
      (10 * c + b : ℚ) = a * q^3 ∧
      a = 1 ∧ b = 4 ∧ c = 6 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_single_digit_numbers_l1804_180433


namespace NUMINAMATH_CALUDE_sum_remainder_mod_nine_l1804_180458

theorem sum_remainder_mod_nine : (88000 + 88002 + 87999 + 88001 + 88003 + 87998) % 9 = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_remainder_mod_nine_l1804_180458


namespace NUMINAMATH_CALUDE_correct_operation_l1804_180476

theorem correct_operation (a b : ℝ) : 3 * a^2 * b - 3 * b * a^2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_correct_operation_l1804_180476


namespace NUMINAMATH_CALUDE_correct_calculation_l1804_180422

theorem correct_calculation (x : ℤ) (h : x - 32 = 33) : x + 32 = 97 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l1804_180422


namespace NUMINAMATH_CALUDE_alice_bob_difference_zero_l1804_180490

/-- Represents the vacation expenses problem -/
def vacation_expenses (alice_paid bob_paid charlie_paid : ℝ) (a b : ℝ) : Prop :=
  let total_paid := alice_paid + bob_paid + charlie_paid
  let equal_share := total_paid / 3
  -- Alice's balance after giving 'a' to Charlie
  (alice_paid - a = equal_share) ∧
  -- Bob's balance after giving 'b' to Charlie
  (bob_paid - b = equal_share) ∧
  -- Charlie's balance after receiving 'a' from Alice and 'b' from Bob
  (charlie_paid + a + b = equal_share)

/-- Theorem stating that the difference between what Alice and Bob give to Charlie is zero -/
theorem alice_bob_difference_zero 
  (alice_paid bob_paid charlie_paid : ℝ) 
  (h_alice : alice_paid = 180) 
  (h_bob : bob_paid = 240) 
  (h_charlie : charlie_paid = 120) :
  ∃ a b : ℝ, vacation_expenses alice_paid bob_paid charlie_paid a b ∧ a - b = 0 :=
sorry

end NUMINAMATH_CALUDE_alice_bob_difference_zero_l1804_180490


namespace NUMINAMATH_CALUDE_quadratic_increasing_implies_a_bound_l1804_180468

/-- A quadratic function f(x) = x^2 + bx + c with b = 2a-1 -/
def QuadraticFunction (a : ℝ) : ℝ → ℝ := λ x => x^2 + (2*a - 1)*x + 3

/-- The function is increasing on the interval (1, +∞) -/
def IsIncreasingOn (f : ℝ → ℝ) : Prop :=
  ∀ x y, 1 < x ∧ x < y → f x < f y

theorem quadratic_increasing_implies_a_bound (a : ℝ) :
  IsIncreasingOn (QuadraticFunction a) → a ≥ -1/2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_increasing_implies_a_bound_l1804_180468


namespace NUMINAMATH_CALUDE_function_equality_condition_l1804_180466

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then a * x + 1 - 4 * a else x^2 - 3 * a * x

theorem function_equality_condition (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f a x₁ = f a x₂) ↔ a ∈ Set.Ioi (2/3) ∪ Set.Iic 0 :=
sorry

end NUMINAMATH_CALUDE_function_equality_condition_l1804_180466


namespace NUMINAMATH_CALUDE_smallest_y_in_arithmetic_series_l1804_180471

theorem smallest_y_in_arithmetic_series (x y z : ℝ) : 
  x > 0 → y > 0 → z > 0 →  -- all terms are positive
  ∃ d : ℝ, x = y - d ∧ z = y + d →  -- arithmetic series
  x * y * z = 216 →  -- product is 216
  y ≥ 6 ∧ ∃ (x₀ y₀ z₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ z₀ > 0 ∧ 
    (∃ d₀ : ℝ, x₀ = y₀ - d₀ ∧ z₀ = y₀ + d₀) ∧ 
    x₀ * y₀ * z₀ = 216 ∧ y₀ = 6 :=
by sorry

end NUMINAMATH_CALUDE_smallest_y_in_arithmetic_series_l1804_180471


namespace NUMINAMATH_CALUDE_paityn_blue_hats_l1804_180446

theorem paityn_blue_hats (paityn_red : ℕ) (paityn_blue : ℕ) (zola_red : ℕ) (zola_blue : ℕ) 
  (h1 : paityn_red = 20)
  (h2 : zola_red = (4 : ℕ) * paityn_red / 5)
  (h3 : zola_blue = 2 * paityn_blue)
  (h4 : paityn_red + paityn_blue + zola_red + zola_blue = 2 * 54) :
  paityn_blue = 24 := by
  sorry

end NUMINAMATH_CALUDE_paityn_blue_hats_l1804_180446


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_l1804_180407

/-- Given a triangle ABC where b = a * sin(C) and c = a * cos(B), prove that ABC is an isosceles right triangle -/
theorem isosceles_right_triangle (a b c : ℝ) (A B C : ℝ) 
  (h1 : b = a * Real.sin C) 
  (h2 : c = a * Real.cos B) 
  (h3 : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h4 : 0 < A ∧ 0 < B ∧ 0 < C) 
  (h5 : A + B + C = π) : 
  A = π / 2 ∧ b = c := by
  sorry

end NUMINAMATH_CALUDE_isosceles_right_triangle_l1804_180407


namespace NUMINAMATH_CALUDE_sally_total_spent_l1804_180437

/-- The amount Sally paid for peaches after applying a coupon -/
def peaches_price : ℚ := 12.32

/-- The amount of the coupon applied to the peaches purchase -/
def coupon_amount : ℚ := 3

/-- The amount Sally paid for cherries -/
def cherries_price : ℚ := 11.54

/-- The theorem stating that the total amount Sally spent is $23.86 -/
theorem sally_total_spent : peaches_price + cherries_price = 23.86 := by
  sorry

end NUMINAMATH_CALUDE_sally_total_spent_l1804_180437


namespace NUMINAMATH_CALUDE_diophantine_equation_solution_l1804_180499

theorem diophantine_equation_solution :
  ∀ x y z : ℤ, 2*x^2 + 2*x^2*z^2 + z^2 + 7*y^2 - 42*y + 33 = 0 ↔
  (x = 1 ∧ y = 5 ∧ z = 0) ∨
  (x = -1 ∧ y = 5 ∧ z = 0) ∨
  (x = 1 ∧ y = 1 ∧ z = 0) ∨
  (x = -1 ∧ y = 1 ∧ z = 0) :=
by sorry

end NUMINAMATH_CALUDE_diophantine_equation_solution_l1804_180499


namespace NUMINAMATH_CALUDE_remaining_painting_time_l1804_180477

/-- Calculates the remaining painting time for a building -/
def remaining_time (total_rooms : ℕ) (hours_per_room : ℕ) (painted_rooms : ℕ) : ℕ :=
  (total_rooms - painted_rooms) * hours_per_room

/-- Theorem: The remaining time to finish all painting work is 155 hours -/
theorem remaining_painting_time : 
  let building1 := remaining_time 12 7 5
  let building2 := remaining_time 15 6 4
  let building3 := remaining_time 10 5 2
  building1 + building2 + building3 = 155 := by
  sorry

end NUMINAMATH_CALUDE_remaining_painting_time_l1804_180477


namespace NUMINAMATH_CALUDE_vector_norm_inequality_l1804_180414

theorem vector_norm_inequality (a₁ a₂ b₁ b₂ : ℝ) :
  Real.sqrt (a₁^2 + a₂^2) + Real.sqrt (b₁^2 + b₂^2) ≥ Real.sqrt ((a₁ - b₁)^2 + (a₂ - b₂)^2) := by
  sorry

end NUMINAMATH_CALUDE_vector_norm_inequality_l1804_180414


namespace NUMINAMATH_CALUDE_odd_product_units_digit_l1804_180496

def is_odd (n : ℕ) : Prop := n % 2 = 1

def is_between (n a b : ℕ) : Prop := a < n ∧ n < b

def units_digit (n : ℕ) : ℕ := n % 10

theorem odd_product_units_digit :
  ∃ (prod : ℕ),
    (∀ n : ℕ, is_odd n ∧ is_between n 20 130 → n ∣ prod) ∧
    units_digit prod = 5 :=
by sorry

end NUMINAMATH_CALUDE_odd_product_units_digit_l1804_180496


namespace NUMINAMATH_CALUDE_expected_digits_icosahedral_die_l1804_180498

def icosahedral_die := Finset.range 20

theorem expected_digits_icosahedral_die :
  let digits_function := fun n => if n < 10 then 1 else 2
  let expected_value := (icosahedral_die.sum fun i => digits_function (i + 1)) / icosahedral_die.card
  expected_value = 31 / 20 := by
sorry

end NUMINAMATH_CALUDE_expected_digits_icosahedral_die_l1804_180498


namespace NUMINAMATH_CALUDE_first_day_over_200_is_thursday_l1804_180402

def paperclips (n : Nat) : Nat := 5 * 3^n

def days : List String := ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]

theorem first_day_over_200_is_thursday :
  days[4] = "Thursday" ∧
  (∀ k < 4, paperclips k ≤ 200) ∧
  paperclips 4 > 200 := by
sorry

end NUMINAMATH_CALUDE_first_day_over_200_is_thursday_l1804_180402


namespace NUMINAMATH_CALUDE_inequality_proof_l1804_180408

theorem inequality_proof (p q : ℝ) (m n : ℕ) 
  (h_pos_p : p > 0) (h_pos_q : q > 0) (h_sum : p + q = 1) (h_pos_m : m > 0) (h_pos_n : n > 0) : 
  (1 - p^m)^n + (1 - q^n)^m ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1804_180408


namespace NUMINAMATH_CALUDE_expression_meets_requirements_l1804_180420

/-- Represents an algebraic expression -/
inductive AlgebraicExpression
  | constant (n : ℚ)
  | variable (name : String)
  | product (e1 e2 : AlgebraicExpression)
  | power (base : AlgebraicExpression) (exponent : ℕ)
  | fraction (numerator denominator : AlgebraicExpression)
  | negation (e : AlgebraicExpression)

/-- Checks if an algebraic expression meets the standard writing requirements -/
def meetsWritingRequirements (e : AlgebraicExpression) : Prop :=
  match e with
  | AlgebraicExpression.constant _ => true
  | AlgebraicExpression.variable _ => true
  | AlgebraicExpression.product e1 e2 => meetsWritingRequirements e1 ∧ meetsWritingRequirements e2
  | AlgebraicExpression.power base exponent => meetsWritingRequirements base ∧ exponent > 0
  | AlgebraicExpression.fraction num den => meetsWritingRequirements num ∧ meetsWritingRequirements den
  | AlgebraicExpression.negation e => meetsWritingRequirements e

/-- The expression -1/3 * x^2 * y -/
def expression : AlgebraicExpression :=
  AlgebraicExpression.negation
    (AlgebraicExpression.fraction
      (AlgebraicExpression.constant 1)
      (AlgebraicExpression.constant 3))

theorem expression_meets_requirements :
  meetsWritingRequirements expression :=
sorry


end NUMINAMATH_CALUDE_expression_meets_requirements_l1804_180420


namespace NUMINAMATH_CALUDE_sock_pairs_count_l1804_180449

def white_socks : ℕ := 5
def brown_socks : ℕ := 4
def blue_socks : ℕ := 2
def red_socks : ℕ := 1

def total_socks : ℕ := white_socks + brown_socks + blue_socks + red_socks

theorem sock_pairs_count :
  (blue_socks * white_socks) + (blue_socks * brown_socks) + (blue_socks * red_socks) = 20 := by
  sorry

end NUMINAMATH_CALUDE_sock_pairs_count_l1804_180449


namespace NUMINAMATH_CALUDE_alice_bushes_l1804_180435

/-- The number of bushes needed to cover three sides of a yard --/
def bushes_needed (side_length : ℕ) (sides : ℕ) (bush_width : ℕ) : ℕ :=
  (side_length * sides) / bush_width

/-- Theorem: Alice needs 24 bushes for her yard --/
theorem alice_bushes :
  bushes_needed 24 3 3 = 24 := by
  sorry

end NUMINAMATH_CALUDE_alice_bushes_l1804_180435


namespace NUMINAMATH_CALUDE_log_relation_l1804_180404

theorem log_relation (a b : ℝ) (ha : a = Real.log 128 / Real.log 4) (hb : b = Real.log 16 / Real.log 2) :
  a = (7 * b) / 8 := by sorry

end NUMINAMATH_CALUDE_log_relation_l1804_180404


namespace NUMINAMATH_CALUDE_fifteenth_student_age_l1804_180400

theorem fifteenth_student_age
  (total_students : Nat)
  (avg_age_all : ℝ)
  (num_group1 : Nat)
  (avg_age_group1 : ℝ)
  (num_group2 : Nat)
  (avg_age_group2 : ℝ)
  (h1 : total_students = 15)
  (h2 : avg_age_all = 15)
  (h3 : num_group1 = 6)
  (h4 : avg_age_group1 = 14)
  (h5 : num_group2 = 8)
  (h6 : avg_age_group2 = 16)
  (h7 : num_group1 + num_group2 + 1 = total_students) :
  total_students * avg_age_all - (num_group1 * avg_age_group1 + num_group2 * avg_age_group2) = 13 := by
  sorry

end NUMINAMATH_CALUDE_fifteenth_student_age_l1804_180400


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l1804_180418

-- Define the hyperbola
def hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1 ∧ a > 0 ∧ b > 0

-- Define the foci
def foci (F₁ F₂ : ℝ × ℝ) (a b : ℝ) : Prop :=
  hyperbola a b (F₂.1) (F₂.2) ∧ 
  F₁.1 < F₂.1 ∧
  F₁.2 = F₂.2

-- Define the circle with diameter F₁F₂
def circle_diameter (F₁ F₂ P : ℝ × ℝ) : Prop :=
  (P.1 - F₁.1)^2 + (P.2 - F₁.2)^2 = (F₂.1 - F₁.1)^2 / 4 + (F₂.2 - F₁.2)^2 / 4

-- Define the intersection of PF₁ and the hyperbola
def intersection (P Q F₁ : ℝ × ℝ) (a b : ℝ) : Prop :=
  ∃ t : ℝ, 
    Q.1 = F₁.1 + t * (P.1 - F₁.1) ∧
    Q.2 = F₁.2 + t * (P.2 - F₁.2) ∧
    hyperbola a b Q.1 Q.2

-- Define the distance condition
def distance_condition (P Q F₁ : ℝ × ℝ) : Prop :=
  (P.1 - Q.1)^2 + (P.2 - Q.2)^2 = 4 * ((Q.1 - F₁.1)^2 + (Q.2 - F₁.2)^2)

-- Theorem statement
theorem hyperbola_eccentricity 
  (a b : ℝ) (F₁ F₂ P Q : ℝ × ℝ) :
  hyperbola a b P.1 P.2 →
  foci F₁ F₂ a b →
  circle_diameter F₁ F₂ P →
  intersection P Q F₁ a b →
  distance_condition P Q F₁ →
  ∃ (c : ℝ), c^2 = a^2 + b^2 ∧ c / a = Real.sqrt 5 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l1804_180418


namespace NUMINAMATH_CALUDE_annabelle_savings_l1804_180406

def weekly_allowance : ℚ := 30
def junk_food_fraction : ℚ := 1/3
def sweets_cost : ℚ := 8

theorem annabelle_savings : 
  weekly_allowance - (weekly_allowance * junk_food_fraction + sweets_cost) = 12 := by
  sorry

end NUMINAMATH_CALUDE_annabelle_savings_l1804_180406


namespace NUMINAMATH_CALUDE_polynomial_evaluation_l1804_180469

theorem polynomial_evaluation :
  ∀ x : ℝ, x > 0 → x^2 - 3*x - 9 = 0 → x^3 - 3*x^2 - 9*x + 7 = 7 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_evaluation_l1804_180469


namespace NUMINAMATH_CALUDE_field_area_theorem_l1804_180467

/-- Represents a rectangular field with a given length and breadth. -/
structure RectangularField where
  length : ℝ
  breadth : ℝ

/-- Calculates the perimeter of a rectangular field. -/
def perimeter (field : RectangularField) : ℝ :=
  2 * (field.length + field.breadth)

/-- Calculates the area of a rectangular field. -/
def area (field : RectangularField) : ℝ :=
  field.length * field.breadth

/-- Theorem: The area of a rectangular field with breadth 60% of its length
    and perimeter 800 m is 37500 square meters. -/
theorem field_area_theorem :
  ∃ (field : RectangularField),
    field.breadth = 0.6 * field.length ∧
    perimeter field = 800 ∧
    area field = 37500 := by
  sorry

end NUMINAMATH_CALUDE_field_area_theorem_l1804_180467


namespace NUMINAMATH_CALUDE_sales_solution_l1804_180483

def sales_problem (sales1 sales2 sales3 sales4 desired_average : ℕ) : Prop :=
  let total_months : ℕ := 5
  let known_sales_sum : ℕ := sales1 + sales2 + sales3 + sales4
  let total_required : ℕ := desired_average * total_months
  let fifth_month_sales : ℕ := total_required - known_sales_sum
  fifth_month_sales = 7870

theorem sales_solution :
  sales_problem 5420 5660 6200 6350 6300 :=
by
  sorry

end NUMINAMATH_CALUDE_sales_solution_l1804_180483


namespace NUMINAMATH_CALUDE_fraction_denominator_l1804_180472

theorem fraction_denominator (y a : ℝ) (h1 : y > 0) (h2 : (2 * y) / a + (3 * y) / a = 0.5 * y) : a = 10 := by
  sorry

end NUMINAMATH_CALUDE_fraction_denominator_l1804_180472


namespace NUMINAMATH_CALUDE_photo_ratio_theorem_l1804_180444

/-- Represents the number of photos in various scenarios --/
structure PhotoCounts where
  initial : ℕ  -- Initial number of photos in the gallery
  firstDay : ℕ  -- Number of photos taken on the first day
  secondDay : ℕ  -- Number of photos taken on the second day
  final : ℕ  -- Final number of photos in the gallery

/-- Theorem stating the ratio of first day photos to initial gallery photos --/
theorem photo_ratio_theorem (p : PhotoCounts) 
  (h1 : p.initial = 400)
  (h2 : p.secondDay = p.firstDay + 120)
  (h3 : p.final = 920)
  (h4 : p.final = p.initial + p.firstDay + p.secondDay) :
  p.firstDay * 2 = p.initial := by
  sorry

#check photo_ratio_theorem

end NUMINAMATH_CALUDE_photo_ratio_theorem_l1804_180444


namespace NUMINAMATH_CALUDE_ratio_of_percentages_l1804_180410

theorem ratio_of_percentages (A B C D : ℝ) 
  (hA : A = 0.4 * B) 
  (hB : B = 0.25 * C) 
  (hD : D = 0.6 * C) 
  (hC : C ≠ 0) : A / D = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_percentages_l1804_180410


namespace NUMINAMATH_CALUDE_arithmetic_sequence_fourth_term_l1804_180460

/-- Given an arithmetic sequence where the sum of the third and fifth terms is 10,
    prove that the fourth term is 5 -/
theorem arithmetic_sequence_fourth_term
  (b : ℝ) -- third term
  (d : ℝ) -- common difference
  (h : b + (b + 2*d) = 10) -- sum of third and fifth terms is 10
  : b + d = 5 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_fourth_term_l1804_180460


namespace NUMINAMATH_CALUDE_base_a_equations_l1804_180434

/-- Converts a base-10 number to base-a --/
def toBaseA (n : ℕ) (a : ℕ) : ℕ := sorry

/-- Converts a base-a number to base-10 --/
def fromBaseA (n : ℕ) (a : ℕ) : ℕ := sorry

theorem base_a_equations (a : ℕ) :
  (toBaseA 375 a + toBaseA 596 a = toBaseA (9 * a + fromBaseA 12 10) a) ∧
  (fromBaseA 12 10 = 12) ∧
  (toBaseA 697 a + toBaseA 226 a = toBaseA (9 * a + fromBaseA 13 10) a) ∧
  (fromBaseA 13 10 = 13) →
  a = 14 := by sorry

end NUMINAMATH_CALUDE_base_a_equations_l1804_180434


namespace NUMINAMATH_CALUDE_blue_marbles_count_l1804_180493

theorem blue_marbles_count (red blue : ℕ) : 
  red + blue = 6000 →
  (red + blue) - (blue - red) = 4800 →
  blue > red →
  blue = 3600 := by
sorry

end NUMINAMATH_CALUDE_blue_marbles_count_l1804_180493


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_range_l1804_180486

theorem hyperbola_eccentricity_range (a b : ℝ) (M : ℝ × ℝ) (F P Q : ℝ × ℝ) (h1 : a > 0) (h2 : b > 0) :
  let (x, y) := M
  (x^2 / a^2 - y^2 / b^2 = 1) →  -- M is on the hyperbola
  (F.1 = a * (a^2 + b^2).sqrt / (a^2 + b^2).sqrt ∧ F.2 = 0) →  -- F is a focus on x-axis
  (∃ r : ℝ, (M.1 - F.1)^2 + M.2^2 = r^2 ∧ P.1 = 0 ∧ Q.1 = 0 ∧ (P.2 - M.2)^2 + M.1^2 = r^2 ∧ (Q.2 - M.2)^2 + M.1^2 = r^2) →  -- Circle condition
  (0 < Real.arccos ((P.2 - M.2) * (Q.2 - M.2) / (((P.2 - M.2)^2 + M.1^2) * ((Q.2 - M.2)^2 + M.1^2)).sqrt) ∧ 
   Real.arccos ((P.2 - M.2) * (Q.2 - M.2) / (((P.2 - M.2)^2 + M.1^2) * ((Q.2 - M.2)^2 + M.1^2)).sqrt) < π/2) →  -- Acute triangle condition
  let e := ((a^2 + b^2) / a^2).sqrt
  (Real.sqrt 5 + 1) / 2 < e ∧ e < (Real.sqrt 6 + Real.sqrt 2) / 2 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_range_l1804_180486


namespace NUMINAMATH_CALUDE_sequence_decomposition_l1804_180445

theorem sequence_decomposition (x : ℕ → ℝ) :
  ∃! (y z : ℕ → ℝ), 
    (∀ n, x n = y n - z n) ∧
    (∀ n, y n ≥ 0) ∧
    (∀ n > 0, z n ≥ z (n - 1)) ∧
    (∀ n > 0, y n * (z n - z (n - 1)) = 0) ∧
    (z 0 = 0) := by
  sorry

end NUMINAMATH_CALUDE_sequence_decomposition_l1804_180445


namespace NUMINAMATH_CALUDE_domain_of_fourth_root_power_function_l1804_180426

-- Define the power function
def f (x : ℝ) : ℝ := x^(1/4)

-- State the theorem about the domain of f
theorem domain_of_fourth_root_power_function :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x ≥ 0} := by sorry

end NUMINAMATH_CALUDE_domain_of_fourth_root_power_function_l1804_180426


namespace NUMINAMATH_CALUDE_line_plane_perpendicularity_l1804_180485

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)

-- State the theorem
theorem line_plane_perpendicularity 
  (m n : Line) (α β : Plane) 
  (h1 : m ≠ n) (h2 : α ≠ β) 
  (h3 : parallel m n) (h4 : perpendicular m α) : 
  perpendicular n α :=
sorry

end NUMINAMATH_CALUDE_line_plane_perpendicularity_l1804_180485


namespace NUMINAMATH_CALUDE_large_circle_radius_l1804_180474

/-- A configuration of five circles where four circles of radius 2 are externally
    tangent to each other and internally tangent to a larger circle. -/
structure CircleConfiguration where
  /-- The radius of each of the four smaller circles -/
  small_radius : ℝ
  /-- The radius of the larger circle -/
  large_radius : ℝ
  /-- The four smaller circles are externally tangent to each other -/
  externally_tangent : True
  /-- The four smaller circles are internally tangent to the larger circle -/
  internally_tangent : True

/-- The radius of the larger circle in the configuration is 4 when the smaller circles have radius 2 -/
theorem large_circle_radius (config : CircleConfiguration) 
  (h : config.small_radius = 2) : config.large_radius = 4 := by
  sorry

end NUMINAMATH_CALUDE_large_circle_radius_l1804_180474


namespace NUMINAMATH_CALUDE_additive_inverses_imply_x_equals_one_l1804_180413

theorem additive_inverses_imply_x_equals_one :
  ∀ x : ℝ, (4 * x - 1) + (3 * x - 6) = 0 → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_additive_inverses_imply_x_equals_one_l1804_180413


namespace NUMINAMATH_CALUDE_fourth_power_roots_l1804_180421

theorem fourth_power_roots (p q : ℝ) (r₁ r₂ : ℂ) : 
  (r₁^2 + p*r₁ + q = 0) → 
  (r₂^2 + p*r₂ + q = 0) → 
  (r₁^4)^2 + ((p^2 - 2*q)^2 - 2*q^2)*(r₁^4) + q^4 = 0 ∧
  (r₂^4)^2 + ((p^2 - 2*q)^2 - 2*q^2)*(r₂^4) + q^4 = 0 := by
  sorry

end NUMINAMATH_CALUDE_fourth_power_roots_l1804_180421


namespace NUMINAMATH_CALUDE_equation_solutions_l1804_180439

theorem equation_solutions :
  (∃ x₁ x₂ : ℝ, (2 * x₁ + 3)^2 = 16 ∧ (2 * x₂ + 3)^2 = 16 ∧ x₁ = 1/2 ∧ x₂ = -7/2) ∧
  (∃ y₁ y₂ : ℝ, y₁^2 - 4*y₁ - 3 = 0 ∧ y₂^2 - 4*y₂ - 3 = 0 ∧ y₁ = 2 + Real.sqrt 7 ∧ y₂ = 2 - Real.sqrt 7) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l1804_180439


namespace NUMINAMATH_CALUDE_half_recipe_flour_l1804_180478

-- Define the original amount of flour in the recipe
def original_flour : ℚ := 4 + 1/2

-- Define the fraction of the recipe we're making
def recipe_fraction : ℚ := 1/2

-- Theorem to prove
theorem half_recipe_flour :
  recipe_fraction * original_flour = 2 + 1/4 :=
by sorry

end NUMINAMATH_CALUDE_half_recipe_flour_l1804_180478


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l1804_180482

def A : Set ℤ := {-1, 2, 3, 5}
def B : Set ℤ := {2, 4, 5}

theorem union_of_A_and_B : A ∪ B = {-1, 2, 3, 4, 5} := by
  sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l1804_180482


namespace NUMINAMATH_CALUDE_parabola_focus_l1804_180427

/-- A parabola is defined by the equation y = 8x^2 -/
def parabola_equation (x y : ℝ) : Prop := y = 8 * x^2

/-- The focus of a parabola is a point on its axis of symmetry -/
def is_focus (x y : ℝ) (parabola : ℝ → ℝ → Prop) : Prop :=
  ∃ (p : ℝ), p > 0 ∧ parabola x y ∧
  ∀ (x' y' : ℝ), parabola x' y' → (x' - x)^2 + (y' - y)^2 = (y' + y - 4*p)^2 / 4

/-- The focus of the parabola y = 8x^2 has coordinates (0, 1/32) -/
theorem parabola_focus :
  is_focus 0 (1/32) parabola_equation :=
sorry

end NUMINAMATH_CALUDE_parabola_focus_l1804_180427


namespace NUMINAMATH_CALUDE_sqrt_45_minus_sqrt_20_equals_sqrt_5_l1804_180484

theorem sqrt_45_minus_sqrt_20_equals_sqrt_5 : 
  Real.sqrt 45 - Real.sqrt 20 = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_45_minus_sqrt_20_equals_sqrt_5_l1804_180484


namespace NUMINAMATH_CALUDE_product_of_fractions_l1804_180415

theorem product_of_fractions : 
  let f (n : ℕ) := (n^3 - 1) / (n^3 + 1)
  (f 2) * (f 3) * (f 4) * (f 5) * (f 6) = 43 / 63 := by
  sorry

end NUMINAMATH_CALUDE_product_of_fractions_l1804_180415


namespace NUMINAMATH_CALUDE_triangle_square_perimeter_difference_l1804_180442

theorem triangle_square_perimeter_difference (d : ℕ) : 
  (∃ (a b : ℝ), 
    a > 0 ∧ b > 0 ∧ 
    3 * a - 4 * b = 1989 ∧ 
    a - b = d ∧ 
    4 * b > 0) ↔ 
  d > 663 :=
sorry

end NUMINAMATH_CALUDE_triangle_square_perimeter_difference_l1804_180442


namespace NUMINAMATH_CALUDE_money_distribution_l1804_180429

theorem money_distribution (A B C : ℤ) 
  (total : A + B + C = 300)
  (ac_sum : A + C = 200)
  (bc_sum : B + C = 350) :
  C = 250 := by
sorry

end NUMINAMATH_CALUDE_money_distribution_l1804_180429


namespace NUMINAMATH_CALUDE_original_group_size_l1804_180462

theorem original_group_size (initial_days work_days : ℕ) (absent_men : ℕ) : 
  initial_days = 15 →
  absent_men = 8 →
  work_days = 18 →
  ∃ (original_size : ℕ),
    original_size * initial_days = (original_size - absent_men) * work_days ∧
    original_size = 48 :=
by sorry

end NUMINAMATH_CALUDE_original_group_size_l1804_180462


namespace NUMINAMATH_CALUDE_greatest_prime_factor_of_247_l1804_180441

theorem greatest_prime_factor_of_247 : ∃ p : ℕ, 
  Nat.Prime p ∧ p ∣ 247 ∧ ∀ q : ℕ, Nat.Prime q → q ∣ 247 → q ≤ p :=
sorry

end NUMINAMATH_CALUDE_greatest_prime_factor_of_247_l1804_180441


namespace NUMINAMATH_CALUDE_train_boggies_count_l1804_180409

/-- The length of each boggy in meters -/
def boggy_length : ℝ := 15

/-- The time in seconds for the train to cross a telegraph post before detaching a boggy -/
def initial_crossing_time : ℝ := 18

/-- The time in seconds for the train to cross a telegraph post after detaching a boggy -/
def final_crossing_time : ℝ := 16.5

/-- The number of boggies initially on the train -/
def initial_boggies : ℕ := 12

theorem train_boggies_count :
  ∃ (n : ℕ),
    (n : ℝ) * boggy_length / initial_crossing_time =
    ((n : ℝ) - 1) * boggy_length / final_crossing_time ∧
    n = initial_boggies :=
by sorry

end NUMINAMATH_CALUDE_train_boggies_count_l1804_180409


namespace NUMINAMATH_CALUDE_complement_of_S_in_U_l1804_180419

-- Define the universe set U
def U : Set Nat := {1, 2, 3, 4}

-- Define the set S
def S : Set Nat := {1, 3}

-- Theorem statement
theorem complement_of_S_in_U :
  U \ S = {2, 4} := by sorry

end NUMINAMATH_CALUDE_complement_of_S_in_U_l1804_180419


namespace NUMINAMATH_CALUDE_x1_value_l1804_180424

theorem x1_value (x₁ x₂ x₃ : Real) 
  (h1 : 0 ≤ x₃ ∧ x₃ ≤ x₂ ∧ x₂ ≤ x₁ ∧ x₁ ≤ 1)
  (h2 : (1 - x₁^2) + (x₁ - x₂)^2 + (x₂ - x₃)^2 + x₃^2 = 1/4) :
  x₁ = Real.sqrt 15 / 4 := by
  sorry

end NUMINAMATH_CALUDE_x1_value_l1804_180424


namespace NUMINAMATH_CALUDE_triangle_angle_from_area_l1804_180475

theorem triangle_angle_from_area (a b c : ℝ) (h_positive : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) 
  (h_area : (b^2 + c^2 - a^2) / (4 * Real.sqrt 3) = 1/2 * b * c * Real.sin (Real.arccos ((b^2 + c^2 - a^2) / (2 * b * c)))) :
  Real.arccos ((b^2 + c^2 - a^2) / (2 * b * c)) = π / 6 := by
  sorry

#check triangle_angle_from_area

end NUMINAMATH_CALUDE_triangle_angle_from_area_l1804_180475


namespace NUMINAMATH_CALUDE_shaded_area_semicircles_l1804_180456

/-- The area of the shaded region formed by semicircles -/
theorem shaded_area_semicircles (UV VW WX XY YZ : ℝ) 
  (h_UV : UV = 3) 
  (h_VW : VW = 5) 
  (h_WX : WX = 4) 
  (h_XY : XY = 6) 
  (h_YZ : YZ = 7) : 
  let UZ := UV + VW + WX + XY + YZ
  let area_large := (π / 8) * UZ^2
  let area_small := (π / 8) * (UV^2 + VW^2 + WX^2 + XY^2 + YZ^2)
  area_large - area_small = (247 / 4) * π := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_semicircles_l1804_180456


namespace NUMINAMATH_CALUDE_gcd_special_numbers_l1804_180454

theorem gcd_special_numbers : Nat.gcd 3333333 666666666 = 3 := by
  sorry

end NUMINAMATH_CALUDE_gcd_special_numbers_l1804_180454


namespace NUMINAMATH_CALUDE_envelope_width_l1804_180494

/-- Given a rectangular envelope with an area of 36 square inches and a height of 6 inches,
    prove that its width is 6 inches. -/
theorem envelope_width (area : ℝ) (height : ℝ) (width : ℝ) 
    (h1 : area = 36) 
    (h2 : height = 6) 
    (h3 : area = width * height) : 
  width = 6 := by
  sorry

end NUMINAMATH_CALUDE_envelope_width_l1804_180494


namespace NUMINAMATH_CALUDE_average_weight_l1804_180480

/-- Given three weights a, b, and c, prove that their average is 42 kg
    under the specified conditions. -/
theorem average_weight (a b c : ℝ) : 
  (a + b) / 2 = 40 →   -- The average weight of a and b is 40 kg
  (b + c) / 2 = 43 →   -- The average weight of b and c is 43 kg
  b = 40 →             -- The weight of b is 40 kg
  (a + b + c) / 3 = 42 -- The average weight of a, b, and c is 42 kg
  := by sorry

end NUMINAMATH_CALUDE_average_weight_l1804_180480


namespace NUMINAMATH_CALUDE_complex_cube_simplification_l1804_180425

theorem complex_cube_simplification :
  let i : ℂ := Complex.I
  (1 + Real.sqrt 3 * i)^3 = -8 := by
  sorry

end NUMINAMATH_CALUDE_complex_cube_simplification_l1804_180425


namespace NUMINAMATH_CALUDE_tangent_line_theorem_l1804_180436

theorem tangent_line_theorem (a b : ℝ) : 
  (∀ x y : ℝ, y = x^2 + a*x + b) →
  (∀ x y : ℝ, x - y + 1 = 0 ↔ y = b ∧ x = 0) →
  a + b = 2 := by
sorry

end NUMINAMATH_CALUDE_tangent_line_theorem_l1804_180436


namespace NUMINAMATH_CALUDE_certain_number_minus_two_l1804_180432

theorem certain_number_minus_two (x : ℝ) (h : 6 - x = 2) : x - 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_minus_two_l1804_180432


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l1804_180417

/-- A geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_ratio
  (a : ℕ → ℝ)
  (h_geom : GeometricSequence a)
  (h_prod : a 5 * a 13 = 6)
  (h_sum : a 4 + a 14 = 5) :
  a 80 / a 90 = 2/3 ∨ a 80 / a 90 = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l1804_180417


namespace NUMINAMATH_CALUDE_count_integers_with_repeated_digits_is_1192_l1804_180491

/-- A function that counts the number of positive four-digit integers less than 5000 
    with at least two identical digits -/
def count_integers_with_repeated_digits : ℕ :=
  let lower_bound := 1000
  let upper_bound := 4999
  sorry

/-- Theorem stating that the count of positive four-digit integers less than 5000 
    with at least two identical digits is 1192 -/
theorem count_integers_with_repeated_digits_is_1192 : 
  count_integers_with_repeated_digits = 1192 := by sorry

end NUMINAMATH_CALUDE_count_integers_with_repeated_digits_is_1192_l1804_180491


namespace NUMINAMATH_CALUDE_olympic_medal_distribution_count_l1804_180428

/-- Represents the number of sprinters of each nationality -/
structure SprinterCounts where
  total : Nat
  americans : Nat
  kenyans : Nat
  others : Nat

/-- Represents the constraints on medal distribution -/
structure MedalConstraints where
  max_american_medals : Nat
  min_kenyan_medals : Nat

/-- Calculates the number of ways to award medals given the sprinter counts and constraints -/
def count_medal_distributions (counts : SprinterCounts) (constraints : MedalConstraints) : Nat :=
  sorry

/-- Theorem stating the number of ways to award medals in the given scenario -/
theorem olympic_medal_distribution_count :
  let counts : SprinterCounts := ⟨10, 4, 2, 4⟩
  let constraints : MedalConstraints := ⟨1, 1⟩
  count_medal_distributions counts constraints = 360 := by
  sorry

end NUMINAMATH_CALUDE_olympic_medal_distribution_count_l1804_180428


namespace NUMINAMATH_CALUDE_discriminant_always_positive_roots_as_triangle_legs_m_values_l1804_180448

-- Define the quadratic equation
def quadratic_eq (m x : ℝ) : ℝ := x^2 - (2+3*m)*x + 2*m^2 + 5*m - 4

-- Define the discriminant of the quadratic equation
def discriminant (m : ℝ) : ℝ := (2+3*m)^2 - 4*(2*m^2 + 5*m - 4)

-- Theorem 1: The discriminant is always positive for any real m
theorem discriminant_always_positive (m : ℝ) : discriminant m > 0 := by
  sorry

-- Define the condition for the roots being legs of a right-angled triangle
def roots_are_triangle_legs (m : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, 
    quadratic_eq m x₁ = 0 ∧ 
    quadratic_eq m x₂ = 0 ∧ 
    x₁^2 + x₂^2 = (2 * Real.sqrt 7)^2

-- Theorem 2: When the roots are legs of a right-angled triangle with hypotenuse 2√7, m is either -2 or 8/5
theorem roots_as_triangle_legs_m_values : 
  ∀ m : ℝ, roots_are_triangle_legs m → (m = -2 ∨ m = 8/5) := by
  sorry

end NUMINAMATH_CALUDE_discriminant_always_positive_roots_as_triangle_legs_m_values_l1804_180448


namespace NUMINAMATH_CALUDE_unknown_number_in_set_l1804_180495

theorem unknown_number_in_set (x : ℝ) : 
  let set1 : List ℝ := [12, 32, 56, 78, 91]
  let set2 : List ℝ := [7, 47, 67, 105, x]
  (set1.sum / set1.length : ℝ) = (set2.sum / set2.length : ℝ) + 10 →
  x = -7 := by
sorry

end NUMINAMATH_CALUDE_unknown_number_in_set_l1804_180495


namespace NUMINAMATH_CALUDE_triangle_cosine_proof_l1804_180479

theorem triangle_cosine_proof (A B C : ℝ) (a b c : ℝ) : 
  a = 4 → c = 9 → (Real.sin A) * (Real.sin C) = (Real.sin B)^2 → 
  Real.cos B = 61/72 := by sorry

end NUMINAMATH_CALUDE_triangle_cosine_proof_l1804_180479


namespace NUMINAMATH_CALUDE_parabola_focus_coordinates_l1804_180431

/-- Given a parabola with equation 7x + 4y² = 0, its focus has coordinates (-7/16, 0) -/
theorem parabola_focus_coordinates :
  ∀ (x y : ℝ),
  (7 * x + 4 * y^2 = 0) →
  ∃ (f : ℝ × ℝ),
  f = (-7/16, 0) ∧
  f.1 = -1/(4 * (4/7)) ∧
  f.2 = 0 :=
by sorry

end NUMINAMATH_CALUDE_parabola_focus_coordinates_l1804_180431


namespace NUMINAMATH_CALUDE_rhind_papyrus_bread_division_l1804_180459

theorem rhind_papyrus_bread_division :
  ∀ (a d : ℚ),
    d > 0 →
    5 * a = 100 →
    (1 / 7) * (a + (a + d) + (a + 2 * d)) = (a - 2 * d) + (a - d) →
    a - 2 * d = 5 / 3 :=
by sorry

end NUMINAMATH_CALUDE_rhind_papyrus_bread_division_l1804_180459


namespace NUMINAMATH_CALUDE_max_abs_z_l1804_180443

theorem max_abs_z (z : ℂ) (θ : ℝ) (h : z - 1 = Complex.cos θ + Complex.I * Complex.sin θ) :
  Complex.abs z ≤ 2 ∧ ∃ θ₀ : ℝ, Complex.abs (1 + Complex.cos θ₀ + Complex.I * Complex.sin θ₀) = 2 := by
  sorry

end NUMINAMATH_CALUDE_max_abs_z_l1804_180443


namespace NUMINAMATH_CALUDE_gcd_4830_3289_l1804_180438

theorem gcd_4830_3289 : Nat.gcd 4830 3289 = 23 := by
  sorry

end NUMINAMATH_CALUDE_gcd_4830_3289_l1804_180438


namespace NUMINAMATH_CALUDE_set_operations_l1804_180450

theorem set_operations (M N P : Set ℕ) 
  (hM : M = {1})
  (hN : N = {1, 2})
  (hP : P = {1, 2, 3}) :
  (M ∪ N) ∩ P = {1, 2} := by
  sorry

end NUMINAMATH_CALUDE_set_operations_l1804_180450


namespace NUMINAMATH_CALUDE_fuel_change_calculation_l1804_180412

/-- Calculates the change received when fueling a vehicle --/
theorem fuel_change_calculation (tank_capacity : ℝ) (initial_fuel : ℝ) (fuel_cost : ℝ) (payment : ℝ) :
  tank_capacity = 150 →
  initial_fuel = 38 →
  fuel_cost = 3 →
  payment = 350 →
  payment - (tank_capacity - initial_fuel) * fuel_cost = 14 := by
  sorry

#check fuel_change_calculation

end NUMINAMATH_CALUDE_fuel_change_calculation_l1804_180412


namespace NUMINAMATH_CALUDE_derivative_f_at_zero_l1804_180405

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + 2*x

-- State the theorem
theorem derivative_f_at_zero :
  deriv f 0 = 2 := by sorry

end NUMINAMATH_CALUDE_derivative_f_at_zero_l1804_180405


namespace NUMINAMATH_CALUDE_matrix_inverse_scalar_multiple_l1804_180430

/-- Given a 2x2 matrix B with elements [[4, 5], [3, m]], prove that if B^(-1) = j * B, 
    then m = -4 and j = 1/31 -/
theorem matrix_inverse_scalar_multiple 
  (B : Matrix (Fin 2) (Fin 2) ℝ)
  (h_B : B = !![4, 5; 3, m])
  (h_inv : B⁻¹ = j • B) :
  m = -4 ∧ j = 1 / 31 := by
  sorry

end NUMINAMATH_CALUDE_matrix_inverse_scalar_multiple_l1804_180430


namespace NUMINAMATH_CALUDE_square_of_binomial_equivalence_l1804_180492

theorem square_of_binomial_equivalence (x : ℝ) : (-3 - x) * (3 - x) = (x - 3)^2 := by
  sorry

end NUMINAMATH_CALUDE_square_of_binomial_equivalence_l1804_180492


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l1804_180457

-- Problem 1
theorem problem_1 (a : ℝ) : a * a^3 - 5 * a^4 + (2 * a^2)^2 = 0 := by sorry

-- Problem 2
theorem problem_2 (a b : ℝ) : (2 * a + 3 * b) * (a - 2 * b) - 1/8 * a * (4 * a - 3 * b) = 3/2 * a^2 - 5/8 * a * b - 6 * b^2 := by sorry

-- Problem 3
theorem problem_3 : (-0.125)^2023 * 2^2024 * 4^2024 = -8 := by sorry

-- Problem 4
theorem problem_4 : (2 * (1/2 : ℝ) - (-1))^2 + ((1/2 : ℝ) - (-1)) * ((1/2 : ℝ) + (-1)) - 5 * (1/2 : ℝ) * ((1/2 : ℝ) - 2 * (-1)) = -3 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l1804_180457


namespace NUMINAMATH_CALUDE_product_remainder_zero_l1804_180461

theorem product_remainder_zero (a b c : ℕ) (ha : a = 1256) (hb : b = 7921) (hc : c = 70305) :
  (a * b * c) % 10 = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_remainder_zero_l1804_180461
