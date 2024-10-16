import Mathlib

namespace NUMINAMATH_CALUDE_divisibility_condition_l3665_366552

theorem divisibility_condition (n : ℤ) : (n + 1) ∣ (n^2 + 1) ↔ n = -3 ∨ n = -2 ∨ n = 0 ∨ n = 1 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_condition_l3665_366552


namespace NUMINAMATH_CALUDE_arrange_teachers_and_students_eq_24_l3665_366532

/-- The number of ways to arrange 2 teachers and 4 students in a row -/
def arrange_teachers_and_students : ℕ :=
  /- Two teachers must be in the middle -/
  let teacher_arrangements : ℕ := 2

  /- One specific student (A) cannot be at either end -/
  let student_A_positions : ℕ := 2

  /- Remaining three students can be arranged in the remaining positions -/
  let other_student_arrangements : ℕ := 6

  /- Total number of arrangements -/
  teacher_arrangements * student_A_positions * other_student_arrangements

/-- Theorem stating that the number of arrangements is 24 -/
theorem arrange_teachers_and_students_eq_24 :
  arrange_teachers_and_students = 24 := by
  sorry

end NUMINAMATH_CALUDE_arrange_teachers_and_students_eq_24_l3665_366532


namespace NUMINAMATH_CALUDE_ratio_consequent_l3665_366559

theorem ratio_consequent (antecedent : ℚ) (consequent : ℚ) : 
  antecedent = 30 → (4 : ℚ) / 6 = antecedent / consequent → consequent = 45 := by
  sorry

end NUMINAMATH_CALUDE_ratio_consequent_l3665_366559


namespace NUMINAMATH_CALUDE_sector_angle_l3665_366510

/-- Theorem: For a circular sector with perimeter 4 cm and area 1 cm², 
    the radian measure of its central angle is 2 radians. -/
theorem sector_angle (r : ℝ) (α : ℝ) 
  (h_perimeter : 2 * r + r * α = 4)
  (h_area : 1/2 * α * r^2 = 1) : 
  α = 2 := by
  sorry

end NUMINAMATH_CALUDE_sector_angle_l3665_366510


namespace NUMINAMATH_CALUDE_chess_pool_theorem_l3665_366573

theorem chess_pool_theorem (U : Type) 
  (A : Set U) -- Set of people who play chess
  (B : Set U) -- Set of people who are not interested in mathematics
  (C : Set U) -- Set of people who bathe in the pool every day
  (h1 : (A ∩ B).Nonempty) -- Condition 1
  (h2 : (C ∩ B ∩ A) = ∅) -- Condition 2
  : ¬(A ⊆ C) := by
  sorry

end NUMINAMATH_CALUDE_chess_pool_theorem_l3665_366573


namespace NUMINAMATH_CALUDE_power_inequality_l3665_366572

theorem power_inequality (a b x y : ℝ) 
  (ha : 0 ≤ a) (hb : 0 ≤ b) (hx : 0 ≤ x) (hy : 0 ≤ y)
  (hab : a^5 + b^5 ≤ 1) (hxy : x^5 + y^5 ≤ 1) : 
  a^2 * x^3 + b^2 * y^3 ≤ 1 := by sorry

end NUMINAMATH_CALUDE_power_inequality_l3665_366572


namespace NUMINAMATH_CALUDE_sum_with_reverse_has_even_digit_l3665_366562

/-- A function that reverses a five-digit integer -/
def reverse_digits (n : ℕ) : ℕ :=
  let a := n / 10000
  let b := (n / 1000) % 10
  let c := (n / 100) % 10
  let d := (n / 10) % 10
  let e := n % 10
  e * 10000 + d * 1000 + c * 100 + b * 10 + a

/-- Predicate to check if a natural number has at least one even digit -/
def has_even_digit (n : ℕ) : Prop :=
  ∃ d : ℕ, d < 10 ∧ 2 ∣ d ∧ ∃ k : ℕ, n / (10^k) % 10 = d

theorem sum_with_reverse_has_even_digit (n : ℕ) 
  (h : 10000 ≤ n ∧ n < 100000) : 
  has_even_digit (n + reverse_digits n) :=
sorry

end NUMINAMATH_CALUDE_sum_with_reverse_has_even_digit_l3665_366562


namespace NUMINAMATH_CALUDE_max_students_above_average_l3665_366576

theorem max_students_above_average (n : ℕ) (score1 score2 : ℚ) : 
  n = 150 →
  score1 > score2 →
  (n - 1) * score1 + score2 > n * ((n - 1) * score1 + score2) / n →
  ∃ (m : ℕ), m ≤ n ∧ m = 149 ∧ 
    (∀ (k : ℕ), k > m → 
      k * score1 + (n - k) * score2 ≤ n * (k * score1 + (n - k) * score2) / n) :=
by sorry

end NUMINAMATH_CALUDE_max_students_above_average_l3665_366576


namespace NUMINAMATH_CALUDE_alcohol_mixture_percentage_l3665_366546

theorem alcohol_mixture_percentage (initial_volume : ℝ) (initial_percentage : ℝ) (added_pure_alcohol : ℝ) : 
  initial_volume = 6 →
  initial_percentage = 35 / 100 →
  added_pure_alcohol = 1.8 →
  let initial_alcohol := initial_volume * initial_percentage
  let final_alcohol := initial_alcohol + added_pure_alcohol
  let final_volume := initial_volume + added_pure_alcohol
  final_alcohol / final_volume = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_alcohol_mixture_percentage_l3665_366546


namespace NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l3665_366503

/-- The function f(x) = a^(2x-1) + 2 passes through the point (1/2, 3) for any a > 0 and a ≠ 1 -/
theorem fixed_point_of_exponential_function (a : ℝ) (ha : a > 0) (ha_neq : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(2*x - 1) + 2
  f (1/2) = 3 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l3665_366503


namespace NUMINAMATH_CALUDE_hyperbola_equation_l3665_366526

/-- Given a hyperbola and a circle with specific properties, prove the equation of the hyperbola. -/
theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x y : ℝ, y^2 / a^2 - x^2 / b^2 = 1) →
  (∃ x y : ℝ, x^2 + y^2 - 6*y + 5 = 0) →
  (∃ x₀ y₀ : ℝ, x₀^2 + y₀^2 - 6*y₀ + 5 = 0 ∧ 
    (∀ x y : ℝ, (y - y₀)^2 / a^2 - (x - x₀)^2 / b^2 = 1)) →
  a^2 = 5 ∧ b^2 = 4 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l3665_366526


namespace NUMINAMATH_CALUDE_greatest_number_less_than_200_with_odd_factors_l3665_366597

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def has_odd_number_of_factors (n : ℕ) : Prop := is_perfect_square n

theorem greatest_number_less_than_200_with_odd_factors : 
  (∀ n : ℕ, n < 200 → has_odd_number_of_factors n → n ≤ 196) ∧ 
  has_odd_number_of_factors 196 ∧ 
  196 < 200 :=
sorry

end NUMINAMATH_CALUDE_greatest_number_less_than_200_with_odd_factors_l3665_366597


namespace NUMINAMATH_CALUDE_power_of_two_greater_than_square_minus_two_l3665_366507

theorem power_of_two_greater_than_square_minus_two (n : ℕ) (h : n > 0) : 
  2^n > n^2 - 2 :=
by
  -- Assume the proposition holds for n = 1, n = 2, and n = 3
  have base_case_1 : 2^1 > 1^2 - 2 := by sorry
  have base_case_2 : 2^2 > 2^2 - 2 := by sorry
  have base_case_3 : 2^3 > 3^2 - 2 := by sorry

  -- Proof by induction
  induction n with
  | zero => contradiction
  | succ n ih =>
    -- Inductive step
    sorry

end NUMINAMATH_CALUDE_power_of_two_greater_than_square_minus_two_l3665_366507


namespace NUMINAMATH_CALUDE_cubic_with_infinite_equal_pairs_has_integer_root_l3665_366513

/-- A cubic polynomial with integer coefficients -/
structure CubicPolynomial where
  a : ℤ
  b : ℤ
  c : ℤ
  d : ℤ
  a_nonzero : a ≠ 0

/-- Evaluation of a cubic polynomial at a given integer -/
def eval (P : CubicPolynomial) (x : ℤ) : ℤ :=
  P.a * x^3 + P.b * x^2 + P.c * x + P.d

/-- The property that there are infinitely many pairs of distinct integers (x, y) such that xP(x) = yP(y) -/
def has_infinite_equal_pairs (P : CubicPolynomial) : Prop :=
  ∀ n : ℕ, ∃ x y : ℤ, x ≠ y ∧ x.natAbs > n ∧ y.natAbs > n ∧ x * eval P x = y * eval P y

/-- The main theorem: if a cubic polynomial has infinite equal pairs, then it has an integer root -/
theorem cubic_with_infinite_equal_pairs_has_integer_root (P : CubicPolynomial) 
  (h : has_infinite_equal_pairs P) : ∃ k : ℤ, eval P k = 0 := by
  sorry

end NUMINAMATH_CALUDE_cubic_with_infinite_equal_pairs_has_integer_root_l3665_366513


namespace NUMINAMATH_CALUDE_least_integer_greater_than_sqrt_500_l3665_366519

theorem least_integer_greater_than_sqrt_500 : ∃ n : ℕ, (n : ℝ) > Real.sqrt 500 ∧ ∀ m : ℕ, (m : ℝ) > Real.sqrt 500 → m ≥ n :=
sorry

end NUMINAMATH_CALUDE_least_integer_greater_than_sqrt_500_l3665_366519


namespace NUMINAMATH_CALUDE_sweater_markup_l3665_366537

theorem sweater_markup (wholesale_price : ℝ) (h1 : wholesale_price > 0) :
  let discounted_price := 1.4 * wholesale_price
  let retail_price := 2 * discounted_price
  let markup := (retail_price - wholesale_price) / wholesale_price * 100
  markup = 180 := by
  sorry

end NUMINAMATH_CALUDE_sweater_markup_l3665_366537


namespace NUMINAMATH_CALUDE_largest_four_digit_congruent_to_17_mod_26_l3665_366594

theorem largest_four_digit_congruent_to_17_mod_26 : ∃ (n : ℕ), 
  (n ≤ 9999) ∧ 
  (n ≥ 1000) ∧
  (n % 26 = 17) ∧
  (∀ m : ℕ, (m ≤ 9999) → (m ≥ 1000) → (m % 26 = 17) → m ≤ n) ∧
  (n = 9978) := by
sorry

end NUMINAMATH_CALUDE_largest_four_digit_congruent_to_17_mod_26_l3665_366594


namespace NUMINAMATH_CALUDE_abs_negative_six_l3665_366596

theorem abs_negative_six : |(-6 : ℤ)| = 6 := by
  sorry

end NUMINAMATH_CALUDE_abs_negative_six_l3665_366596


namespace NUMINAMATH_CALUDE_car_speed_problem_l3665_366595

/-- Proves that if a car traveling at 94.73684210526315 km/h takes 2 seconds longer to travel 1 kilometer
    compared to a certain faster speed, then that faster speed is 90 km/h. -/
theorem car_speed_problem (current_speed : ℝ) (faster_speed : ℝ) : 
  current_speed = 94.73684210526315 →
  (1 / current_speed) * 3600 = (1 / faster_speed) * 3600 + 2 →
  faster_speed = 90 := by
  sorry

end NUMINAMATH_CALUDE_car_speed_problem_l3665_366595


namespace NUMINAMATH_CALUDE_notebook_cost_l3665_366509

theorem notebook_cost (total_students : Nat) (total_cost : Nat) : ∃ (buying_students : Nat) (notebooks_per_student : Nat) (cost_per_notebook : Nat),
  total_students = 42 ∧
  buying_students > total_students / 2 ∧
  notebooks_per_student > 1 ∧
  cost_per_notebook > notebooks_per_student ∧
  buying_students * notebooks_per_student * cost_per_notebook = total_cost ∧
  total_cost = 3213 ∧
  cost_per_notebook = 17 :=
by sorry

end NUMINAMATH_CALUDE_notebook_cost_l3665_366509


namespace NUMINAMATH_CALUDE_limit_sqrt_minus_one_over_x_l3665_366581

theorem limit_sqrt_minus_one_over_x (f : ℝ → ℝ) (h : ∀ x ≠ 0, f x = (1 - Real.sqrt (x + 1)) / x) :
  Filter.Tendsto f (Filter.atTop.comap (fun x => 1 / x)) (nhds (-1/2)) := by
sorry

end NUMINAMATH_CALUDE_limit_sqrt_minus_one_over_x_l3665_366581


namespace NUMINAMATH_CALUDE_probability_of_even_sum_l3665_366590

def M : Finset ℕ := {1, 2, 3}
def N : Finset ℕ := {4, 5, 6}

def is_sum_even (x : ℕ) (y : ℕ) : Bool := Even (x + y)

def favorable_outcomes : Finset (ℕ × ℕ) :=
  (M.product N).filter (fun (x, y) => is_sum_even x y)

theorem probability_of_even_sum :
  (favorable_outcomes.card : ℚ) / ((M.card * N.card) : ℚ) = 4 / 9 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_even_sum_l3665_366590


namespace NUMINAMATH_CALUDE_girls_to_boys_ratio_l3665_366536

theorem girls_to_boys_ratio (total : ℕ) (difference : ℕ) (girls boys : ℕ) : 
  total = 30 →
  difference = 6 →
  girls = boys + difference →
  total = girls + boys →
  (girls : ℚ) / (boys : ℚ) = 3 / 2 :=
by
  sorry

end NUMINAMATH_CALUDE_girls_to_boys_ratio_l3665_366536


namespace NUMINAMATH_CALUDE_challenge_result_l3665_366540

theorem challenge_result (x : ℕ) : 3 * (3 * (x + 1) + 3) = 63 := by
  sorry

#check challenge_result

end NUMINAMATH_CALUDE_challenge_result_l3665_366540


namespace NUMINAMATH_CALUDE_negation_of_universal_quantifier_negation_of_proposition_l3665_366535

theorem negation_of_universal_quantifier (P : ℝ → Prop) :
  (¬ ∀ x ∈ Set.Ici 1, P x) ↔ (∃ x ∈ Set.Ici 1, ¬ P x) :=
by sorry

theorem negation_of_proposition :
  (¬ ∀ x ∈ Set.Ici 1, x^2 - 2*x + 1 ≥ 0) ↔ (∃ x ∈ Set.Ici 1, x^2 - 2*x + 1 < 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_quantifier_negation_of_proposition_l3665_366535


namespace NUMINAMATH_CALUDE_odd_function_value_l3665_366561

-- Define f as a function from ℝ to ℝ
variable (f : ℝ → ℝ)

-- Define g in terms of f
def g (x : ℝ) : ℝ := f x + 9

-- State the theorem
theorem odd_function_value (hf_odd : ∀ x, f (-x) = -f x) (hg : g (-2) = 3) : f 2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_value_l3665_366561


namespace NUMINAMATH_CALUDE_complex_real_part_twice_imaginary_l3665_366516

theorem complex_real_part_twice_imaginary (a b : ℝ) : 
  let z : ℂ := Complex.mk a b
  (Complex.re z = 2 * Complex.im z) → a = 2 := by
sorry

end NUMINAMATH_CALUDE_complex_real_part_twice_imaginary_l3665_366516


namespace NUMINAMATH_CALUDE_x_value_proof_l3665_366566

theorem x_value_proof (x : ℚ) 
  (eq1 : 8 * x^2 + 8 * x - 2 = 0) 
  (eq2 : 32 * x^2 + 68 * x - 8 = 0) : 
  x = 1/8 := by
sorry

end NUMINAMATH_CALUDE_x_value_proof_l3665_366566


namespace NUMINAMATH_CALUDE_a_fourth_minus_four_a_cubed_minus_four_a_plus_seven_equals_eight_l3665_366550

theorem a_fourth_minus_four_a_cubed_minus_four_a_plus_seven_equals_eight :
  ∀ a : ℝ, a = 1 / (Real.sqrt 5 - 2) → a^4 - 4*a^3 - 4*a + 7 = 8 := by
sorry

end NUMINAMATH_CALUDE_a_fourth_minus_four_a_cubed_minus_four_a_plus_seven_equals_eight_l3665_366550


namespace NUMINAMATH_CALUDE_intersection_M_N_l3665_366518

def M : Set ℝ := {1, 2, 3, 4, 5}
def N : Set ℝ := {x | Real.log x / Real.log 4 ≥ 1}

theorem intersection_M_N : M ∩ N = {4, 5} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l3665_366518


namespace NUMINAMATH_CALUDE_meaningful_fraction_l3665_366556

theorem meaningful_fraction (x : ℝ) : 
  (∃ y : ℝ, y = 1 / (x - 1)) ↔ x ≠ 1 := by sorry

end NUMINAMATH_CALUDE_meaningful_fraction_l3665_366556


namespace NUMINAMATH_CALUDE_emails_left_theorem_l3665_366523

/-- Calculates the number of emails left in the inbox after a series of moves -/
def emailsLeftInInbox (initialEmails : ℕ) : ℕ :=
  let afterTrash := initialEmails / 2
  let afterWork := afterTrash - (afterTrash * 2 / 5)
  let afterPersonal := afterWork - (afterWork / 4)
  afterPersonal - (afterPersonal / 10)

/-- Theorem stating that given 500 initial emails, after a series of moves, 102 emails are left in the inbox -/
theorem emails_left_theorem :
  emailsLeftInInbox 500 = 102 := by
  sorry

end NUMINAMATH_CALUDE_emails_left_theorem_l3665_366523


namespace NUMINAMATH_CALUDE_problem1_l3665_366574

theorem problem1 (a b : ℝ) (ha : a ≠ 0) :
  (a - b^2 / a) / ((a^2 + 2*a*b + b^2) / a) = (a - b) / (a + b) := by
  sorry

end NUMINAMATH_CALUDE_problem1_l3665_366574


namespace NUMINAMATH_CALUDE_loan_duration_C_l3665_366589

/-- Calculates simple interest -/
def simpleInterest (principal : ℚ) (rate : ℚ) (time : ℚ) : ℚ :=
  principal * rate * time / 100

theorem loan_duration_C (principal_B principal_C total_interest : ℚ) 
  (time_B : ℚ) (rate : ℚ) :
  principal_B = 4000 →
  principal_C = 2000 →
  time_B = 2 →
  rate = 13.75 →
  total_interest = 2200 →
  simpleInterest principal_B rate time_B + simpleInterest principal_C rate (4 : ℚ) = total_interest :=
by sorry

end NUMINAMATH_CALUDE_loan_duration_C_l3665_366589


namespace NUMINAMATH_CALUDE_green_shirt_percentage_l3665_366530

-- Define the total number of students
def total_students : ℕ := 800

-- Define the percentage of students wearing blue shirts
def blue_percentage : ℚ := 45 / 100

-- Define the percentage of students wearing red shirts
def red_percentage : ℚ := 23 / 100

-- Define the number of students wearing other colors
def other_colors : ℕ := 136

-- Theorem to prove
theorem green_shirt_percentage :
  (total_students - (blue_percentage * total_students).floor - 
   (red_percentage * total_students).floor - other_colors) / total_students = 15 / 100 := by
sorry

end NUMINAMATH_CALUDE_green_shirt_percentage_l3665_366530


namespace NUMINAMATH_CALUDE_no_solution_exists_l3665_366551

theorem no_solution_exists : ¬∃ (a b : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ 1 / a^2 + 1 / b^2 = 1 / (a^2 + b^2) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_exists_l3665_366551


namespace NUMINAMATH_CALUDE_rectangle_dimensions_l3665_366582

theorem rectangle_dimensions (x : ℝ) : 
  (x + 3) * (3 * x - 2) = 9 * x + 1 → 
  x > 0 → 
  3 * x - 2 > 0 → 
  x = (11 + Real.sqrt 205) / 6 := by
sorry

end NUMINAMATH_CALUDE_rectangle_dimensions_l3665_366582


namespace NUMINAMATH_CALUDE_benny_attended_games_l3665_366554

/-- 
Given:
- The total number of baseball games is 39.
- Benny missed 25 games.

Prove that the number of games Benny attended is 14.
-/
theorem benny_attended_games (total_games : ℕ) (missed_games : ℕ) 
  (h1 : total_games = 39)
  (h2 : missed_games = 25) :
  total_games - missed_games = 14 := by
  sorry

end NUMINAMATH_CALUDE_benny_attended_games_l3665_366554


namespace NUMINAMATH_CALUDE_right_triangle_sets_l3665_366539

theorem right_triangle_sets :
  let set1 : Fin 3 → ℝ := ![3, 4, 5]
  let set2 : Fin 3 → ℝ := ![9, 12, 15]
  let set3 : Fin 3 → ℝ := ![Real.sqrt 3, 2, Real.sqrt 5]
  let set4 : Fin 3 → ℝ := ![0.3, 0.4, 0.5]

  (set1 0)^2 + (set1 1)^2 = (set1 2)^2 ∧
  (set2 0)^2 + (set2 1)^2 = (set2 2)^2 ∧
  (set3 0)^2 + (set3 1)^2 ≠ (set3 2)^2 ∧
  (set4 0)^2 + (set4 1)^2 = (set4 2)^2 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_sets_l3665_366539


namespace NUMINAMATH_CALUDE_value_difference_l3665_366515

theorem value_difference (n : ℝ) (increase_percent : ℝ) (decrease_percent : ℝ) :
  n = 80 ∧ increase_percent = 0.125 ∧ decrease_percent = 0.25 →
  n * (1 + increase_percent) - n * (1 - decrease_percent) = 30 :=
by sorry

end NUMINAMATH_CALUDE_value_difference_l3665_366515


namespace NUMINAMATH_CALUDE_triangle_side_calculation_l3665_366529

/-- Given a triangle ABC with side a = 4, angle B = π/3, and area S = 6√3,
    prove that side b = 2√7 -/
theorem triangle_side_calculation (A B C : Real) (a b c : Real) :
  -- Conditions
  a = 4 →
  B = π / 3 →
  (1 / 2) * a * c * Real.sin B = 6 * Real.sqrt 3 →
  -- Definition of cosine law
  b ^ 2 = a ^ 2 + c ^ 2 - 2 * a * c * Real.cos B →
  -- Conclusion
  b = 2 * Real.sqrt 7 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_calculation_l3665_366529


namespace NUMINAMATH_CALUDE_wardrobe_cost_is_180_l3665_366545

/-- Calculates the total cost of Marcia's wardrobe given the following conditions:
  - 3 skirts at $20.00 each
  - 5 blouses at $15.00 each
  - 2 pairs of pants at $30.00 each, with a sale: buy 1 pair, get 1 pair 1/2 off
-/
def wardrobeCost (skirtPrice blousePrice pantPrice : ℚ) : ℚ :=
  let skirtCost := 3 * skirtPrice
  let blouseCost := 5 * blousePrice
  let pantCost := pantPrice + (pantPrice / 2)
  skirtCost + blouseCost + pantCost

/-- Proves that the total cost of Marcia's wardrobe is $180.00 -/
theorem wardrobe_cost_is_180 :
  wardrobeCost 20 15 30 = 180 := by
  sorry

#eval wardrobeCost 20 15 30

end NUMINAMATH_CALUDE_wardrobe_cost_is_180_l3665_366545


namespace NUMINAMATH_CALUDE_sergey_mistake_l3665_366567

theorem sergey_mistake : ¬∃ a : ℤ, a % 15 = 8 ∧ a % 20 = 17 := by
  sorry

end NUMINAMATH_CALUDE_sergey_mistake_l3665_366567


namespace NUMINAMATH_CALUDE_price_ratio_theorem_l3665_366591

theorem price_ratio_theorem (cost_price : ℝ) (first_price second_price : ℝ) :
  first_price = cost_price * (1 + 1.4) ∧
  second_price = cost_price * (1 - 0.2) →
  second_price / first_price = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_price_ratio_theorem_l3665_366591


namespace NUMINAMATH_CALUDE_positive_root_range_log_function_range_l3665_366558

-- Part 1
theorem positive_root_range (a : ℝ) :
  (∃ x > 0, 4^x + 2^x = a^2 + a) ↔ a ∈ Set.Ioi 1 ∪ Set.Iio (-2) :=
sorry

-- Part 2
theorem log_function_range (a : ℝ) :
  (∀ y : ℝ, ∃ x : ℝ, Real.log (x^2 + a*x + 1) = y) ↔ a ∈ Set.Ici 2 ∪ Set.Iic (-2) :=
sorry

end NUMINAMATH_CALUDE_positive_root_range_log_function_range_l3665_366558


namespace NUMINAMATH_CALUDE_exponent_multiplication_l3665_366531

theorem exponent_multiplication (a : ℝ) : a^2 * a^3 = a^5 := by
  sorry

end NUMINAMATH_CALUDE_exponent_multiplication_l3665_366531


namespace NUMINAMATH_CALUDE_mary_remaining_money_l3665_366553

def remaining_money (p : ℝ) : ℝ :=
  let drink_cost := p
  let medium_pizza_cost := 3 * p
  let large_pizza_cost := 4 * p
  let total_cost := 5 * drink_cost + medium_pizza_cost + large_pizza_cost
  30 - total_cost

theorem mary_remaining_money (p : ℝ) :
  remaining_money p = 30 - 12 * p :=
by sorry

end NUMINAMATH_CALUDE_mary_remaining_money_l3665_366553


namespace NUMINAMATH_CALUDE_perimeter_of_modified_square_l3665_366592

/-- Given a square with perimeter 40 inches, prove that cutting an equilateral triangle
    from one corner and translating it to the right side results in a figure with
    perimeter 60 inches. -/
theorem perimeter_of_modified_square (square_perimeter : ℝ) 
  (h_square_perimeter : square_perimeter = 40) : ℝ :=
by
  -- Define the side length of the square
  let square_side := square_perimeter / 4
  
  -- Define the side length of the equilateral triangle
  let triangle_side := square_side
  
  -- Calculate the perimeter of the new figure
  let new_perimeter := 2 * square_side + 3 * triangle_side + 2 * square_side
  
  -- Prove that the new perimeter equals 60
  sorry

#check perimeter_of_modified_square

end NUMINAMATH_CALUDE_perimeter_of_modified_square_l3665_366592


namespace NUMINAMATH_CALUDE_smallest_value_for_x_between_0_and_1_l3665_366541

theorem smallest_value_for_x_between_0_and_1 (x : ℝ) (h1 : 0 < x) (h2 : x < 1) :
  x^3 ≤ x ∧ x^3 ≤ x^2 ∧ x^3 ≤ x^3 ∧ x^3 ≤ Real.sqrt x ∧ x^3 ≤ 2*x ∧ x^3 ≤ 1/x :=
by sorry

end NUMINAMATH_CALUDE_smallest_value_for_x_between_0_and_1_l3665_366541


namespace NUMINAMATH_CALUDE_unique_number_between_30_and_40_with_units_digit_2_l3665_366569

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

def has_units_digit (n : ℕ) (d : ℕ) : Prop := n % 10 = d

theorem unique_number_between_30_and_40_with_units_digit_2 :
  ∃! n : ℕ, is_two_digit n ∧ 30 < n ∧ n < 40 ∧ has_units_digit n 2 :=
by sorry

end NUMINAMATH_CALUDE_unique_number_between_30_and_40_with_units_digit_2_l3665_366569


namespace NUMINAMATH_CALUDE_ages_sum_l3665_366583

theorem ages_sum (a b c : ℕ+) : 
  a = b ∧ a > c ∧ a * a * c = 162 → a + b + c = 20 := by
  sorry

end NUMINAMATH_CALUDE_ages_sum_l3665_366583


namespace NUMINAMATH_CALUDE_product_remainder_zero_l3665_366502

theorem product_remainder_zero : (2005 * 2006 * 2007 * 2008 * 2009) % 17 = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_remainder_zero_l3665_366502


namespace NUMINAMATH_CALUDE_flower_beds_count_l3665_366524

theorem flower_beds_count (total_seeds : ℕ) (seeds_per_bed : ℕ) (h1 : total_seeds = 54) (h2 : seeds_per_bed = 6) :
  total_seeds / seeds_per_bed = 9 := by
  sorry

end NUMINAMATH_CALUDE_flower_beds_count_l3665_366524


namespace NUMINAMATH_CALUDE_sisters_sandcastle_height_is_half_foot_l3665_366527

/-- The height of Miki's sister's sandcastle given Miki's sandcastle height and the height difference -/
def sisters_sandcastle_height (mikis_height : ℝ) (height_difference : ℝ) : ℝ :=
  mikis_height - height_difference

/-- Theorem stating that Miki's sister's sandcastle height is 0.50 foot -/
theorem sisters_sandcastle_height_is_half_foot :
  sisters_sandcastle_height 0.83 0.33 = 0.50 := by
  sorry

end NUMINAMATH_CALUDE_sisters_sandcastle_height_is_half_foot_l3665_366527


namespace NUMINAMATH_CALUDE_geometric_progression_problem_l3665_366565

def geometric_progression (b₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ := b₁ * q^(n - 1)

theorem geometric_progression_problem (b₁ b₅ : ℝ) (h₁ : b₁ = Real.sqrt 3) (h₅ : b₅ = Real.sqrt 243) :
  ∃ q : ℝ, (q = Real.sqrt 3 ∨ q = -Real.sqrt 3) ∧
    geometric_progression b₁ q 5 = b₅ ∧
    geometric_progression b₁ q 6 = 27 ∨ geometric_progression b₁ q 6 = -27 :=
by sorry

end NUMINAMATH_CALUDE_geometric_progression_problem_l3665_366565


namespace NUMINAMATH_CALUDE_mindy_message_count_l3665_366568

/-- The number of emails and phone messages Mindy has in total -/
def total_messages (phone_messages : ℕ) (emails : ℕ) : ℕ :=
  phone_messages + emails

/-- The relationship between emails and phone messages -/
def email_phone_relation (phone_messages : ℕ) : ℕ :=
  9 * phone_messages - 7

theorem mindy_message_count :
  ∃ (phone_messages : ℕ),
    email_phone_relation phone_messages = 83 ∧
    total_messages phone_messages 83 = 93 := by
  sorry

end NUMINAMATH_CALUDE_mindy_message_count_l3665_366568


namespace NUMINAMATH_CALUDE_angle_sum_BD_l3665_366543

-- Define the triangle and its angles
structure Triangle (A B C : Type) where
  angleA : ℝ
  angleB : ℝ
  angleC : ℝ

-- Define the configuration
structure Configuration where
  angleA : ℝ
  angleAFG : ℝ
  angleAGF : ℝ
  angleB : ℝ
  angleD : ℝ

-- Theorem statement
theorem angle_sum_BD (config : Configuration) 
  (h1 : config.angleA = 30)
  (h2 : config.angleAFG = config.angleAGF) :
  config.angleB + config.angleD = 75 := by
  sorry


end NUMINAMATH_CALUDE_angle_sum_BD_l3665_366543


namespace NUMINAMATH_CALUDE_sum_of_preceding_terms_l3665_366508

-- Define the arithmetic sequence
def arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ d : ℕ, ∀ n : ℕ, a (n + 1) = a n + d

-- Define our specific sequence
def our_sequence (a : ℕ → ℕ) : Prop :=
  arithmetic_sequence a ∧ a 1 = 3 ∧ a 2 = 8 ∧ ∃ k : ℕ, a k = 33 ∧ ∀ m : ℕ, m > k → a m > 33

theorem sum_of_preceding_terms (a : ℕ → ℕ) (h : our_sequence a) :
  ∃ n : ℕ, a n + a (n + 1) = 51 ∧ a (n + 2) = 33 :=
sorry

end NUMINAMATH_CALUDE_sum_of_preceding_terms_l3665_366508


namespace NUMINAMATH_CALUDE_abc_inequality_l3665_366514

theorem abc_inequality (a b c : ℝ) (h : (1/4)*a^2 + (1/4)*b^2 + c^2 = 1) :
  -2 ≤ a*b + 2*b*c + 2*c*a ∧ a*b + 2*b*c + 2*c*a ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_abc_inequality_l3665_366514


namespace NUMINAMATH_CALUDE_range_of_sin_plus_cos_l3665_366571

theorem range_of_sin_plus_cos :
  ∀ y : ℝ, (∃ x : ℝ, y = Real.sin x + Real.cos x) ↔ -Real.sqrt 2 ≤ y ∧ y ≤ Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_sin_plus_cos_l3665_366571


namespace NUMINAMATH_CALUDE_factor_implies_c_value_l3665_366521

/-- The polynomial P(x) -/
def P (c : ℝ) (x : ℝ) : ℝ := x^3 + 3*x^2 + c*x + 15

/-- Theorem: If x - 3 is a factor of P(x), then c = -23 -/
theorem factor_implies_c_value (c : ℝ) : 
  (∀ x, P c x = 0 ↔ x = 3) → c = -23 := by
  sorry

end NUMINAMATH_CALUDE_factor_implies_c_value_l3665_366521


namespace NUMINAMATH_CALUDE_blocks_left_l3665_366585

/-- Given that Randy has 97 blocks initially and uses 25 blocks to build a tower,
    prove that the number of blocks left is 72. -/
theorem blocks_left (initial_blocks : ℕ) (used_blocks : ℕ) (h1 : initial_blocks = 97) (h2 : used_blocks = 25) :
  initial_blocks - used_blocks = 72 := by
  sorry

end NUMINAMATH_CALUDE_blocks_left_l3665_366585


namespace NUMINAMATH_CALUDE_norm_scalar_multiple_l3665_366560

theorem norm_scalar_multiple (v : ℝ × ℝ) (h : ‖v‖ = 5) : ‖(5 : ℝ) • v‖ = 25 := by
  sorry

end NUMINAMATH_CALUDE_norm_scalar_multiple_l3665_366560


namespace NUMINAMATH_CALUDE_distance_sum_constant_l3665_366534

theorem distance_sum_constant (a b x : ℝ) (h : 0 ≤ x ∧ x ≤ 50) :
  |x - a| + |x - b| = 50 :=
by
  sorry

#check distance_sum_constant

end NUMINAMATH_CALUDE_distance_sum_constant_l3665_366534


namespace NUMINAMATH_CALUDE_multiples_properties_l3665_366505

theorem multiples_properties (c d : ℤ) 
  (hc : ∃ k : ℤ, c = 4 * k) 
  (hd : ∃ m : ℤ, d = 8 * m) : 
  (∃ n : ℤ, d = 4 * n) ∧ 
  (∃ p : ℤ, c - d = 4 * p) ∧ 
  (∃ q : ℤ, c - d = 2 * q) :=
by sorry

end NUMINAMATH_CALUDE_multiples_properties_l3665_366505


namespace NUMINAMATH_CALUDE_complex_magnitude_l3665_366504

theorem complex_magnitude (z : ℂ) (h : z * (1 - Complex.I) = 1 + Complex.I) : 
  Complex.abs z = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l3665_366504


namespace NUMINAMATH_CALUDE_root_relationship_l3665_366586

def f (x : ℝ) : ℝ := x^3 - 7*x^2 + 12*x - 10
def g (x : ℝ) : ℝ := x^3 - 10*x^2 - 2*x + 20

theorem root_relationship :
  ∃ (x₀ : ℝ), f x₀ = 0 ∧ g (2*x₀) = 0 →
  f 5 = 0 ∧ g 10 = 0 := by
sorry

end NUMINAMATH_CALUDE_root_relationship_l3665_366586


namespace NUMINAMATH_CALUDE_at_least_one_vertex_inside_or_on_boundary_l3665_366599

structure CentrallySymmetricPolygon where
  vertices : Set (ℝ × ℝ)
  is_centrally_symmetric : ∃ (center : ℝ × ℝ), ∀ v ∈ vertices, 
    ∃ v' ∈ vertices, v' = (2 * center.1 - v.1, 2 * center.2 - v.2)

structure Polygon where
  vertices : Set (ℝ × ℝ)

def contained_in (T : Polygon) (M : CentrallySymmetricPolygon) : Prop :=
  ∀ v ∈ T.vertices, v ∈ M.vertices

def symmetric_image (T : Polygon) (P : ℝ × ℝ) : Polygon :=
  { vertices := {v' | ∃ v ∈ T.vertices, v' = (2 * P.1 - v.1, 2 * P.2 - v.2)} }

def vertex_inside_or_on_boundary (v : ℝ × ℝ) (M : CentrallySymmetricPolygon) : Prop :=
  v ∈ M.vertices

theorem at_least_one_vertex_inside_or_on_boundary 
  (M : CentrallySymmetricPolygon) (T : Polygon) (P : ℝ × ℝ) :
  contained_in T M →
  P ∈ {p | ∃ v ∈ T.vertices, p = v} →
  ∃ v ∈ (symmetric_image T P).vertices, vertex_inside_or_on_boundary v M :=
sorry

end NUMINAMATH_CALUDE_at_least_one_vertex_inside_or_on_boundary_l3665_366599


namespace NUMINAMATH_CALUDE_equivalent_root_equations_l3665_366528

theorem equivalent_root_equations (a : ℝ) :
  ∀ x : ℝ, x = a + Real.sqrt (a + Real.sqrt x) ↔ x = a + Real.sqrt x :=
by sorry

end NUMINAMATH_CALUDE_equivalent_root_equations_l3665_366528


namespace NUMINAMATH_CALUDE_complementary_angles_ratio_l3665_366520

theorem complementary_angles_ratio (a b : ℝ) : 
  a + b = 90 →  -- The angles are complementary (sum to 90°)
  a / b = 5 / 4 →  -- The ratio of the angles is 5:4
  a > b →  -- a is the larger angle
  a = 50 :=  -- The larger angle measures 50°
by sorry

end NUMINAMATH_CALUDE_complementary_angles_ratio_l3665_366520


namespace NUMINAMATH_CALUDE_parabola_shift_l3665_366578

/-- Represents a parabola in the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Shifts a parabola horizontally and vertically -/
def shift_parabola (p : Parabola) (h : ℝ) (v : ℝ) : Parabola :=
  { a := p.a
  , b := p.b - 2 * p.a * h
  , c := p.a * h^2 - p.b * h + p.c + v }

theorem parabola_shift :
  let original := Parabola.mk 1 (-2) 3
  let shifted := shift_parabola original 1 (-3)
  shifted = Parabola.mk 1 0 (-1) := by sorry

end NUMINAMATH_CALUDE_parabola_shift_l3665_366578


namespace NUMINAMATH_CALUDE_cone_volume_from_cylinder_l3665_366577

/-- Given a cylinder with volume 81π cm³, prove that a cone with the same base radius
    and twice the height of the cylinder has a volume of 54π cm³. -/
theorem cone_volume_from_cylinder (r h : ℝ) : 
  (π * r^2 * h = 81 * π) → 
  ((1/3) * π * r^2 * (2*h) = 54 * π) :=
by sorry

end NUMINAMATH_CALUDE_cone_volume_from_cylinder_l3665_366577


namespace NUMINAMATH_CALUDE_uncovered_side_length_l3665_366557

/-- Represents a rectangular field with three sides fenced -/
structure FencedField where
  length : ℝ
  width : ℝ
  area : ℝ
  fencing : ℝ

/-- The uncovered side of a fenced field is 40 feet given the conditions -/
theorem uncovered_side_length (field : FencedField)
  (h_area : field.area = 680)
  (h_fencing : field.fencing = 74)
  (h_area_calc : field.area = field.length * field.width)
  (h_fencing_calc : field.fencing = 2 * field.width + field.length) :
  field.length = 40 := by
  sorry

end NUMINAMATH_CALUDE_uncovered_side_length_l3665_366557


namespace NUMINAMATH_CALUDE_bruce_age_multiple_l3665_366501

/-- The number of years it takes for a person to become a multiple of another person's age -/
def years_to_multiple (initial_age_older : ℕ) (initial_age_younger : ℕ) (multiple : ℕ) : ℕ :=
  let x := (multiple * initial_age_younger - initial_age_older) / (multiple - 1)
  x

/-- Theorem stating that it takes 6 years for a 36-year-old to become 3 times as old as an 8-year-old -/
theorem bruce_age_multiple : years_to_multiple 36 8 3 = 6 := by
  sorry

end NUMINAMATH_CALUDE_bruce_age_multiple_l3665_366501


namespace NUMINAMATH_CALUDE_cupcake_distribution_l3665_366542

theorem cupcake_distribution (total_cupcakes : ℕ) (num_children : ℕ) 
  (h1 : total_cupcakes = 96) (h2 : num_children = 8) :
  total_cupcakes / num_children = 12 := by
  sorry

end NUMINAMATH_CALUDE_cupcake_distribution_l3665_366542


namespace NUMINAMATH_CALUDE_bobs_family_adults_l3665_366584

theorem bobs_family_adults (total_apples : ℕ) (num_children : ℕ) (apples_per_child : ℕ) (apples_per_adult : ℕ) 
  (h1 : total_apples = 1200)
  (h2 : num_children = 45)
  (h3 : apples_per_child = 15)
  (h4 : apples_per_adult = 5) :
  (total_apples - num_children * apples_per_child) / apples_per_adult = 105 :=
by
  sorry

end NUMINAMATH_CALUDE_bobs_family_adults_l3665_366584


namespace NUMINAMATH_CALUDE_no_solutions_in_interval_l3665_366570

theorem no_solutions_in_interval (x : ℝ) :
  x ∈ Set.Icc (π / 4) (π / 2) →
  ¬(Real.sin (x ^ Real.sin x) = Real.cos (x ^ Real.cos x)) :=
by sorry

end NUMINAMATH_CALUDE_no_solutions_in_interval_l3665_366570


namespace NUMINAMATH_CALUDE_candies_per_packet_candies_per_packet_proof_l3665_366512

/-- The number of candies in a packet given Bobby's eating habits and the time it takes to finish the packets. -/
theorem candies_per_packet : ℕ :=
  let packets : ℕ := 2
  let weekdays : ℕ := 5
  let weekend_days : ℕ := 2
  let candies_per_weekday : ℕ := 2
  let candies_per_weekend_day : ℕ := 1
  let weeks_to_finish : ℕ := 3
  18

/-- Proof that the number of candies in a packet is 18. -/
theorem candies_per_packet_proof :
  let packets : ℕ := 2
  let weekdays : ℕ := 5
  let weekend_days : ℕ := 2
  let candies_per_weekday : ℕ := 2
  let candies_per_weekend_day : ℕ := 1
  let weeks_to_finish : ℕ := 3
  candies_per_packet = 18 := by
  sorry

end NUMINAMATH_CALUDE_candies_per_packet_candies_per_packet_proof_l3665_366512


namespace NUMINAMATH_CALUDE_simplify_fourth_root_and_sum_l3665_366538

theorem simplify_fourth_root_and_sum : ∃ (a b : ℕ+), 
  (↑a : ℝ) * ((↑b : ℝ) ^ (1/4 : ℝ)) = ((2 : ℝ)^9 * (3 : ℝ)^5) ^ (1/4 : ℝ) ∧ 
  a.val + b.val = 502 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fourth_root_and_sum_l3665_366538


namespace NUMINAMATH_CALUDE_bananas_to_pears_cost_equivalence_l3665_366525

/-- Given the cost relationships between bananas, apples, and pears at Lucy's Local Market,
    this theorem proves that 25 bananas cost as much as 10 pears. -/
theorem bananas_to_pears_cost_equivalence 
  (banana_apple_ratio : (5 : ℚ) * banana_cost = (3 : ℚ) * apple_cost)
  (apple_pear_ratio : (9 : ℚ) * apple_cost = (6 : ℚ) * pear_cost)
  (banana_cost apple_cost pear_cost : ℚ) :
  (25 : ℚ) * banana_cost = (10 : ℚ) * pear_cost :=
by sorry


end NUMINAMATH_CALUDE_bananas_to_pears_cost_equivalence_l3665_366525


namespace NUMINAMATH_CALUDE_trajectory_of_P_l3665_366506

-- Define the ellipse C
def C (x y : ℝ) : Prop := x^2/2 + y^2 = 1

-- Define the point M on the ellipse C
def M : ℝ × ℝ → Prop := λ p => C p.1 p.2

-- Define the point N on the x-axis
def N (m : ℝ × ℝ) : ℝ × ℝ := (m.1, 0)

-- Define the vector NP
def NP (n p : ℝ × ℝ) : ℝ × ℝ := (p.1 - n.1, p.2 - n.2)

-- Define the vector NM
def NM (n m : ℝ × ℝ) : ℝ × ℝ := (m.1 - n.1, m.2 - n.2)

-- State the theorem
theorem trajectory_of_P (x y : ℝ) :
  (∃ m : ℝ × ℝ, M m ∧ 
   let n := N m
   NP n (x, y) = Real.sqrt 2 • NM n m) →
  x^2 + y^2 = 2 :=
sorry

end NUMINAMATH_CALUDE_trajectory_of_P_l3665_366506


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3665_366547

theorem arithmetic_sequence_sum (n : ℕ) (a : ℕ → ℝ) : 
  (∀ k, a (k + 1) - a k = a (k + 2) - a (k + 1)) →  -- arithmetic sequence condition
  ((n + 1) * a (n + 1) = 4) →  -- sum of odd-numbered terms
  (n * a (n + 1) = 3) →  -- sum of even-numbered terms
  n = 3 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3665_366547


namespace NUMINAMATH_CALUDE_unique_solution_power_equation_l3665_366544

theorem unique_solution_power_equation :
  ∀ x y : ℕ, x ≥ 1 → y ≥ 1 → (2^x : ℕ) + 3 = 11^y → x = 3 ∧ y = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_power_equation_l3665_366544


namespace NUMINAMATH_CALUDE_total_pencils_l3665_366575

theorem total_pencils (num_people : ℕ) (pencils_per_person : ℕ) : 
  num_people = 5 → pencils_per_person = 15 → num_people * pencils_per_person = 75 := by
  sorry

end NUMINAMATH_CALUDE_total_pencils_l3665_366575


namespace NUMINAMATH_CALUDE_find_p_l3665_366579

def fibonacci_like_sequence (a : ℕ → ℕ) : Prop :=
  ∀ n, n ≥ 2 → a n = a (n - 1) + a (n - 2)

theorem find_p (a : ℕ → ℕ) (h : fibonacci_like_sequence a) 
  (h5 : a 4 = 5) (h8 : a 5 = 8) (h13 : a 6 = 13) : a 0 = 1 := by
  sorry

end NUMINAMATH_CALUDE_find_p_l3665_366579


namespace NUMINAMATH_CALUDE_odot_specific_values_odot_power_relation_l3665_366517

/-- Definition of the ⊙ operation for rational numbers -/
def odot (m n : ℚ) : ℚ := m * n * (m - n)

/-- Theorem for part 1 of the problem -/
theorem odot_specific_values :
  let a : ℚ := 1/2
  let b : ℚ := -1
  odot (a + b) (a - b) = 3/2 := by sorry

/-- Theorem for part 2 of the problem -/
theorem odot_power_relation (x y : ℚ) :
  odot (x^2 * y) (odot x y) = x^5 * y^4 - x^4 * y^5 := by sorry

end NUMINAMATH_CALUDE_odot_specific_values_odot_power_relation_l3665_366517


namespace NUMINAMATH_CALUDE_test_total_points_l3665_366598

theorem test_total_points (total_questions : ℕ) (two_point_questions : ℕ) : 
  total_questions = 40 → 
  two_point_questions = 30 → 
  (total_questions - two_point_questions) * 4 + two_point_questions * 2 = 100 := by
sorry

end NUMINAMATH_CALUDE_test_total_points_l3665_366598


namespace NUMINAMATH_CALUDE_min_expected_weight_l3665_366564

theorem min_expected_weight (x y e : ℝ) :
  y = 0.85 * x - 88 + e →
  |e| ≤ 4 →
  x = 160 →
  ∃ y_min : ℝ, y_min = 44 ∧ ∀ y' : ℝ, (∃ e' : ℝ, y' = 0.85 * x - 88 + e' ∧ |e'| ≤ 4) → y' ≥ y_min :=
by sorry

end NUMINAMATH_CALUDE_min_expected_weight_l3665_366564


namespace NUMINAMATH_CALUDE_hexagon_intersection_collinearity_l3665_366588

/-- Represents a point in 2D space -/
structure Point :=
  (x y : ℝ)

/-- Represents a line in 2D space -/
structure Line :=
  (a b c : ℝ)

/-- Represents a hexagon -/
structure Hexagon :=
  (A B C D E F : Point)

/-- Checks if three points are collinear -/
def collinear (p1 p2 p3 : Point) : Prop := sorry

/-- Returns the intersection point of two lines -/
def intersectionPoint (l1 l2 : Line) : Point := sorry

/-- Theorem: Collinearity of intersections in a hexagon with specific conditions -/
theorem hexagon_intersection_collinearity 
  (ABCDEF : Hexagon)
  (diagonalIntersection : Point)
  (hDiagonals : intersectionPoint (Line.mk 0 0 0) (Line.mk 0 0 0) = diagonalIntersection ∧ 
                intersectionPoint (Line.mk 0 0 0) (Line.mk 0 0 0) = diagonalIntersection ∧ 
                intersectionPoint (Line.mk 0 0 0) (Line.mk 0 0 0) = diagonalIntersection)
  (A' : Point) (hA' : A' = intersectionPoint (Line.mk 0 0 0) (Line.mk 0 0 0))
  (B' : Point) (hB' : B' = intersectionPoint (Line.mk 0 0 0) (Line.mk 0 0 0))
  (C' : Point) (hC' : C' = intersectionPoint (Line.mk 0 0 0) (Line.mk 0 0 0))
  (D' E' F' : Point)
  : collinear 
      (intersectionPoint (Line.mk 0 0 0) (Line.mk 0 0 0))
      (intersectionPoint (Line.mk 0 0 0) (Line.mk 0 0 0))
      (intersectionPoint (Line.mk 0 0 0) (Line.mk 0 0 0)) :=
by sorry

end NUMINAMATH_CALUDE_hexagon_intersection_collinearity_l3665_366588


namespace NUMINAMATH_CALUDE_ratio_problem_l3665_366593

theorem ratio_problem (a b c d : ℚ) 
  (h1 : a / b = 5)
  (h2 : b / c = 1 / 4)
  (h3 : c / d = 7) :
  d / a = 4 / 35 := by
sorry

end NUMINAMATH_CALUDE_ratio_problem_l3665_366593


namespace NUMINAMATH_CALUDE_boy_scout_percentage_l3665_366511

/-- Represents the composition of a group of scouts -/
structure ScoutGroup where
  total : ℝ
  boys : ℝ
  girls : ℝ
  total_is_sum : total = boys + girls

/-- Represents the percentage of scouts with signed permission slips -/
structure PermissionSlips where
  total_percent : ℝ
  boys_percent : ℝ
  girls_percent : ℝ
  total_is_70_percent : total_percent = 0.7
  boys_is_75_percent : boys_percent = 0.75
  girls_is_62_5_percent : girls_percent = 0.625

theorem boy_scout_percentage 
  (group : ScoutGroup) 
  (slips : PermissionSlips) : 
  group.boys / group.total = 0.6 := by
  sorry

end NUMINAMATH_CALUDE_boy_scout_percentage_l3665_366511


namespace NUMINAMATH_CALUDE_carla_cooking_time_l3665_366522

def total_time (sharpening_time peeling_time chopping_time first_break fruits_time second_break salad_time : ℝ) : ℝ :=
  sharpening_time + peeling_time + chopping_time + first_break + fruits_time + second_break + salad_time

theorem carla_cooking_time : ∃ (total : ℝ),
  let sharpening_time : ℝ := 15
  let peeling_time : ℝ := 3 * sharpening_time
  let chopping_time : ℝ := (1 / 4) * peeling_time
  let first_break : ℝ := 5
  let fruits_time : ℝ := 2 * chopping_time
  let second_break : ℝ := 10
  let previous_activities_time : ℝ := sharpening_time + peeling_time + chopping_time + first_break + fruits_time + second_break
  let salad_time : ℝ := (3 / 5) * previous_activities_time
  total = total_time sharpening_time peeling_time chopping_time first_break fruits_time second_break salad_time ∧
  total = 174.6 := by
    sorry

end NUMINAMATH_CALUDE_carla_cooking_time_l3665_366522


namespace NUMINAMATH_CALUDE_subsets_of_B_l3665_366587

def B : Set ℕ := {0, 1, 2}

theorem subsets_of_B :
  {A : Set ℕ | A ⊆ B} =
  {∅, {0}, {1}, {2}, {0, 1}, {0, 2}, {1, 2}, B} :=
by sorry

end NUMINAMATH_CALUDE_subsets_of_B_l3665_366587


namespace NUMINAMATH_CALUDE_remaining_cards_l3665_366500

def initial_cards : ℕ := 87
def sam_cards : ℕ := 8
def alex_cards : ℕ := 13

theorem remaining_cards : initial_cards - (sam_cards + alex_cards) = 66 := by
  sorry

end NUMINAMATH_CALUDE_remaining_cards_l3665_366500


namespace NUMINAMATH_CALUDE_log_100_base_10_l3665_366580

theorem log_100_base_10 : Real.log 100 / Real.log 10 = 2 := by sorry

end NUMINAMATH_CALUDE_log_100_base_10_l3665_366580


namespace NUMINAMATH_CALUDE_unique_solution_xy_l3665_366533

theorem unique_solution_xy : ∃! (x y : ℝ), 
  (x + y = (7 - x) + (7 - y)) ∧ 
  (x - y = (x - 3) + (y - 3)) ∧ 
  x = 1 ∧ y = 6 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_xy_l3665_366533


namespace NUMINAMATH_CALUDE_BC_completion_time_l3665_366549

/-- The time it takes for a group of workers to complete a job -/
def completion_time (work_rate : ℚ) : ℚ := 1 / work_rate

/-- The work rate of a single worker A -/
def work_rate_A : ℚ := 1 / 10

/-- The combined work rate of workers A and B -/
def work_rate_AB : ℚ := 1 / 5

/-- The combined work rate of workers A, B, and C -/
def work_rate_ABC : ℚ := 1 / 3

/-- The combined work rate of workers B and C -/
def work_rate_BC : ℚ := work_rate_ABC - work_rate_A

theorem BC_completion_time :
  completion_time work_rate_BC = 30 / 7 := by
  sorry

end NUMINAMATH_CALUDE_BC_completion_time_l3665_366549


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3665_366563

theorem complex_equation_solution (a : ℝ) : 
  Complex.abs ((a + Complex.I) / Complex.I) = 2 → a = Real.sqrt 3 ∨ a = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3665_366563


namespace NUMINAMATH_CALUDE_similar_triangle_perimeter_l3665_366555

theorem similar_triangle_perimeter :
  ∀ (small_side : ℝ) (small_base : ℝ) (large_base : ℝ),
    small_side > 0 → small_base > 0 → large_base > 0 →
    small_side + small_side + small_base = 7 + 7 + 12 →
    large_base = 36 →
    large_base / small_base = 36 / 12 →
    (2 * small_side * (large_base / small_base) + large_base) = 78 :=
by
  sorry

#check similar_triangle_perimeter

end NUMINAMATH_CALUDE_similar_triangle_perimeter_l3665_366555


namespace NUMINAMATH_CALUDE_no_rational_points_on_sqrt3_circle_l3665_366548

theorem no_rational_points_on_sqrt3_circle : 
  ¬∃ (x y : ℚ), x^2 + y^2 = 3 := by
sorry

end NUMINAMATH_CALUDE_no_rational_points_on_sqrt3_circle_l3665_366548
