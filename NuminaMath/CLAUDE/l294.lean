import Mathlib

namespace NUMINAMATH_CALUDE_tax_saving_theorem_l294_29485

theorem tax_saving_theorem (old_rate new_rate : ℝ) (saving : ℝ) (income : ℝ) : 
  old_rate = 0.45 → 
  new_rate = 0.30 → 
  saving = 7200 → 
  (old_rate - new_rate) * income = saving → 
  income = 48000 := by
sorry

end NUMINAMATH_CALUDE_tax_saving_theorem_l294_29485


namespace NUMINAMATH_CALUDE_no_solution_for_x_equals_one_l294_29420

theorem no_solution_for_x_equals_one :
  ¬∃ (y : ℝ), (1 : ℝ) / (1 + 1) + y = (1 : ℝ) / (1 - 1) :=
by sorry

end NUMINAMATH_CALUDE_no_solution_for_x_equals_one_l294_29420


namespace NUMINAMATH_CALUDE_triangle_properties_l294_29459

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.b = Real.sqrt 13 ∧
  t.a + t.c = 4 ∧
  (Real.cos t.B) / (Real.cos t.C) = -t.b / (2 * t.a + t.c)

-- State the theorem
theorem triangle_properties (t : Triangle) (h : triangle_conditions t) :
  t.B = 2 * Real.pi / 3 ∧
  (1 / 2) * t.a * t.c * Real.sin t.B = 3 * Real.sqrt 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l294_29459


namespace NUMINAMATH_CALUDE_inequality_proof_l294_29479

theorem inequality_proof (x y z : ℝ) (h : x^2 + y^2 + z^2 = 3) :
  x^3 - (y^2 + y*z + z^2)*x + y*z*(y + z) ≤ 3 * Real.sqrt 3 ∧
  (x^3 - (y^2 + y*z + z^2)*x + y*z*(y + z) = 3 * Real.sqrt 3 ↔ 
   x = Real.sqrt 3 ∧ y = 0 ∧ z = 0) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l294_29479


namespace NUMINAMATH_CALUDE_positive_real_inequality_l294_29456

theorem positive_real_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : a^2000 + b^2000 = a^1998 + b^1998) : a^2 + b^2 ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_positive_real_inequality_l294_29456


namespace NUMINAMATH_CALUDE_min_value_theorem_l294_29431

theorem min_value_theorem (x y : ℝ) (h : x^2 * y^2 + y^4 = 1) :
  ∃ (m : ℝ), m = 2 * Real.sqrt 2 ∧ ∀ (z w : ℝ), z^2 * w^2 + w^4 = 1 → x^2 + 3 * y^2 ≤ z^2 + 3 * w^2 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l294_29431


namespace NUMINAMATH_CALUDE_A_nonempty_A_subset_B_l294_29473

/-- Definition of set A -/
def A (a : ℝ) : Set ℝ := {x : ℝ | 2*a + 1 ≤ x ∧ x ≤ 3*a - 5}

/-- Definition of set B -/
def B : Set ℝ := {x : ℝ | x < -1 ∨ x > 16}

/-- Theorem for the non-emptiness of A -/
theorem A_nonempty (a : ℝ) : (A a).Nonempty ↔ a ≥ 6 := by sorry

/-- Theorem for A being a subset of B -/
theorem A_subset_B (a : ℝ) : A a ⊆ B ↔ a < 6 ∨ a > 15/2 := by sorry

end NUMINAMATH_CALUDE_A_nonempty_A_subset_B_l294_29473


namespace NUMINAMATH_CALUDE_roots_quadratic_equation_l294_29458

theorem roots_quadratic_equation (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁^2 - a*x₁ + a^2 - a = 0 ∧ x₂^2 - a*x₂ + a^2 - a = 0 ∧ x₁ ≠ x₂) →
  (0 ≤ a ∧ a ≤ 4/3) :=
by sorry

end NUMINAMATH_CALUDE_roots_quadratic_equation_l294_29458


namespace NUMINAMATH_CALUDE_solution_set_theorem_l294_29451

open Set

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the conditions
variable (h1 : f (-2) = 2013)
variable (h2 : ∀ x : ℝ, deriv f x < 2 * x)

-- Define the solution set
def solution_set (f : ℝ → ℝ) : Set ℝ :=
  {x : ℝ | f x > x^2 + 2009}

-- State the theorem
theorem solution_set_theorem (f : ℝ → ℝ) (h1 : f (-2) = 2013) (h2 : ∀ x : ℝ, deriv f x < 2 * x) :
  solution_set f = Iio (-2) :=
sorry

end NUMINAMATH_CALUDE_solution_set_theorem_l294_29451


namespace NUMINAMATH_CALUDE_student_presentation_time_l294_29428

theorem student_presentation_time 
  (num_students : ℕ) 
  (period_length : ℕ) 
  (num_periods : ℕ) 
  (h1 : num_students = 32) 
  (h2 : period_length = 40) 
  (h3 : num_periods = 4) : 
  (num_periods * period_length) / num_students = 5 := by
  sorry

end NUMINAMATH_CALUDE_student_presentation_time_l294_29428


namespace NUMINAMATH_CALUDE_problem_statement_l294_29419

theorem problem_statement (x y z a b c : ℝ) 
  (h1 : x/a + y/b + z/c = 5)
  (h2 : a/x + b/y + c/z = 6) :
  x^2/a^2 + y^2/b^2 + z^2/c^2 = 13 := by sorry

end NUMINAMATH_CALUDE_problem_statement_l294_29419


namespace NUMINAMATH_CALUDE_yellow_balls_count_l294_29449

theorem yellow_balls_count (total : ℕ) (yellow : ℕ) (h1 : total = 15) 
  (h2 : yellow ≤ total) 
  (h3 : (yellow : ℚ) / total * (yellow - 1) / (total - 1) = 1 / 21) : 
  yellow = 5 := by sorry

end NUMINAMATH_CALUDE_yellow_balls_count_l294_29449


namespace NUMINAMATH_CALUDE_money_division_l294_29426

theorem money_division (r s t u : ℝ) (h1 : r / s = 2.5 / 3.5) (h2 : s / t = 3.5 / 7.5) 
  (h3 : t / u = 7.5 / 9.8) (h4 : t - s = 4500) : u - r = 8212.5 := by
  sorry

end NUMINAMATH_CALUDE_money_division_l294_29426


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l294_29416

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  geometric_sequence a →
  a 4 + a 7 = 2 →
  a 5 * a 6 = -8 →
  a 1 + a 10 = -7 :=
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l294_29416


namespace NUMINAMATH_CALUDE_at_op_difference_l294_29445

-- Define the @ operation
def at_op (x y : ℤ) : ℤ := x * y - 3 * x + y

-- Theorem statement
theorem at_op_difference : (at_op 5 6) - (at_op 6 5) = 4 := by
  sorry

end NUMINAMATH_CALUDE_at_op_difference_l294_29445


namespace NUMINAMATH_CALUDE_angle_D_measure_l294_29424

-- Define the hexagon and its angles
structure Hexagon where
  A : ℝ
  B : ℝ
  C : ℝ
  D : ℝ
  E : ℝ
  F : ℝ

-- Define the properties of the hexagon
def is_convex_hexagon_with_properties (h : Hexagon) : Prop :=
  h.A = h.B ∧ h.B = h.C ∧  -- A, B, C are congruent
  h.D = h.E ∧ h.E = h.F ∧  -- D, E, F are congruent
  h.A + 30 = h.D ∧         -- A is 30° less than D
  h.A + h.B + h.C + h.D + h.E + h.F = 720  -- Sum of angles in a hexagon

-- Theorem statement
theorem angle_D_measure (h : Hexagon) 
  (hprop : is_convex_hexagon_with_properties h) : h.D = 135 := by
  sorry

end NUMINAMATH_CALUDE_angle_D_measure_l294_29424


namespace NUMINAMATH_CALUDE_stating_cat_purchase_possible_l294_29412

/-- Represents the available denominations of rubles --/
def denominations : List ℕ := [1, 5, 10, 50, 100, 500, 1000]

/-- Represents the total amount of money available --/
def total_money : ℕ := 1999

/-- 
Theorem stating that for any price of the cat, 
the buyer can make the purchase and receive correct change
--/
theorem cat_purchase_possible :
  ∀ (price : ℕ), price ≤ total_money →
  ∃ (buyer_money seller_money : List ℕ),
    (buyer_money.sum = price) ∧
    (seller_money.sum = total_money - price) ∧
    (∀ x ∈ buyer_money ∪ seller_money, x ∈ denominations) :=
by sorry

end NUMINAMATH_CALUDE_stating_cat_purchase_possible_l294_29412


namespace NUMINAMATH_CALUDE_sin_range_on_interval_l294_29475

theorem sin_range_on_interval :
  let f : ℝ → ℝ := λ x ↦ Real.sin x
  let S : Set ℝ := { x | -π/4 ≤ x ∧ x ≤ 3*π/4 }
  f '' S = { y | -Real.sqrt 2 / 2 ≤ y ∧ y ≤ 1 } := by
  sorry

end NUMINAMATH_CALUDE_sin_range_on_interval_l294_29475


namespace NUMINAMATH_CALUDE_quadratic_function_relationship_l294_29457

/-- A quadratic function passing through specific points -/
structure QuadraticFunction where
  p : ℝ
  q : ℝ
  a : ℝ
  b : ℝ
  y₁ : ℝ
  y₂ : ℝ
  y₃ : ℝ
  eq_a : b = -5 * a^2 + p * a + q
  eq_b : y₁ = q
  eq_c : b = -5 * (4 - a)^2 + p * (4 - a) + q
  eq_d : y₂ = -5 + p + q
  eq_e : y₃ = -80 + 4 * p + q

/-- The relationship between y₁, y₂, and y₃ for the given quadratic function -/
theorem quadratic_function_relationship (f : QuadraticFunction) : f.y₃ = f.y₁ ∧ f.y₁ < f.y₂ := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_relationship_l294_29457


namespace NUMINAMATH_CALUDE_opposite_of_negative_two_l294_29418

def opposite (x : ℝ) : ℝ := -x

theorem opposite_of_negative_two :
  opposite (-2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_two_l294_29418


namespace NUMINAMATH_CALUDE_candy_probability_l294_29463

def total_candies : ℕ := 24
def red_candies : ℕ := 12
def blue_candies : ℕ := 12
def terry_picks : ℕ := 2
def mary_picks : ℕ := 3

def same_color_probability : ℚ := 66 / 1771

theorem candy_probability :
  red_candies = blue_candies ∧
  red_candies + blue_candies = total_candies ∧
  terry_picks + mary_picks < total_candies →
  same_color_probability = (2 * (Nat.choose red_candies terry_picks * Nat.choose (red_candies - terry_picks) mary_picks)) / 
                           (Nat.choose total_candies terry_picks * Nat.choose (total_candies - terry_picks) mary_picks) :=
by sorry

end NUMINAMATH_CALUDE_candy_probability_l294_29463


namespace NUMINAMATH_CALUDE_linear_equation_exponent_l294_29484

theorem linear_equation_exponent (k : ℝ) : 
  (∀ x, ∃ a b : ℝ, x^(2*k - 1) + 2 = a*x + b) → k = 1 := by
  sorry

end NUMINAMATH_CALUDE_linear_equation_exponent_l294_29484


namespace NUMINAMATH_CALUDE_chocolate_profit_l294_29480

theorem chocolate_profit (num_bars : ℕ) (cost_per_bar : ℝ) (total_selling_price : ℝ) (packaging_cost_per_bar : ℝ) :
  num_bars = 5 →
  cost_per_bar = 5 →
  total_selling_price = 90 →
  packaging_cost_per_bar = 2 →
  total_selling_price - (num_bars * cost_per_bar + num_bars * packaging_cost_per_bar) = 55 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_profit_l294_29480


namespace NUMINAMATH_CALUDE_negative_fraction_comparison_l294_29432

theorem negative_fraction_comparison : -3/5 < -1/5 := by
  sorry

end NUMINAMATH_CALUDE_negative_fraction_comparison_l294_29432


namespace NUMINAMATH_CALUDE_symmetric_curve_is_correct_l294_29454

/-- The equation of the original circle -/
def original_circle (x y : ℝ) : Prop := (x - 2)^2 + (y + 1)^2 = 1

/-- The equation of the line of symmetry -/
def symmetry_line (x y : ℝ) : Prop := x - y + 3 = 0

/-- The equation of the symmetric curve -/
def symmetric_curve (x y : ℝ) : Prop := (x + 4)^2 + (y - 5)^2 = 1

/-- Theorem stating that the symmetric_curve is indeed symmetric to the original_circle with respect to the symmetry_line -/
theorem symmetric_curve_is_correct : 
  ∀ (x y : ℝ), symmetric_curve x y ↔ 
  ∃ (x' y' : ℝ), original_circle x' y' ∧ 
  ∃ (x_mid y_mid : ℝ), symmetry_line x_mid y_mid ∧
  x_mid = (x + x') / 2 ∧ y_mid = (y + y') / 2 :=
sorry

end NUMINAMATH_CALUDE_symmetric_curve_is_correct_l294_29454


namespace NUMINAMATH_CALUDE_solve_a_b_l294_29462

def U : Set ℝ := Set.univ

def A (a b : ℝ) : Set ℝ := {x | x^2 + a*x + 12*b = 0}

def B (a b : ℝ) : Set ℝ := {x | x^2 - a*x + b = 0}

theorem solve_a_b :
  ∀ a b : ℝ, 
    (2 ∈ (U \ A a b) ∩ (B a b)) → 
    (4 ∈ (A a b) ∩ (U \ B a b)) → 
    a = 8/7 ∧ b = -12/7 := by
  sorry

end NUMINAMATH_CALUDE_solve_a_b_l294_29462


namespace NUMINAMATH_CALUDE_smallest_power_l294_29407

theorem smallest_power (a b c d : ℕ) : 
  a = 2 → b = 3 → c = 5 → d = 6 → 
  a^55 < b^44 ∧ a^55 < c^33 ∧ a^55 < d^22 := by
  sorry

end NUMINAMATH_CALUDE_smallest_power_l294_29407


namespace NUMINAMATH_CALUDE_condition_for_inequality_l294_29438

theorem condition_for_inequality (a b c : ℝ) :
  (¬ (∀ c, a > b → a * c^2 > b * c^2)) ∧
  ((a * c^2 > b * c^2) → a > b) :=
by sorry

end NUMINAMATH_CALUDE_condition_for_inequality_l294_29438


namespace NUMINAMATH_CALUDE_sum_coefficients_when_binomial_sum_is_8_l294_29466

/-- Given a natural number n, this function represents the sum of the binomial coefficients
    of the expansion of (x^2 - 2/x)^n when x = 1 -/
def sumBinomialCoefficients (n : ℕ) : ℤ := (-1 : ℤ) ^ n

/-- Given a natural number n, this function represents the sum of the coefficients
    of the expansion of (x^2 - 2/x)^n when x = 1 -/
def sumCoefficients (n : ℕ) : ℤ := ((-1 : ℤ) - 2) ^ n

theorem sum_coefficients_when_binomial_sum_is_8 :
  ∃ n : ℕ, sumBinomialCoefficients n = 8 ∧ sumCoefficients n = -1 := by
  sorry

end NUMINAMATH_CALUDE_sum_coefficients_when_binomial_sum_is_8_l294_29466


namespace NUMINAMATH_CALUDE_permutations_formula_l294_29410

def factorial (n : ℕ) : ℕ := Nat.factorial n

def permutations_with_repetition (n : ℕ) (k : List ℕ) : ℚ :=
  (factorial n) / (k.map factorial).prod

theorem permutations_formula (n : ℕ) (k : List ℕ) 
  (h : k.sum = n) : 
  permutations_with_repetition n k = 
    (factorial n) / (k.map factorial).prod := by
  sorry

#eval permutations_with_repetition 5 [5]  -- for "замок"
#eval permutations_with_repetition 5 [1, 2, 2]  -- for "ротор"
#eval permutations_with_repetition 5 [3, 2]  -- for "топор"
#eval permutations_with_repetition 7 [1, 2, 2, 3]  -- for "колокол"

end NUMINAMATH_CALUDE_permutations_formula_l294_29410


namespace NUMINAMATH_CALUDE_james_profit_l294_29461

def total_tickets : ℕ := 200
def ticket_prices : List ℚ := [1, 3, 4]
def ticket_percentages : List ℚ := [50, 30, 20]
def winning_odds : List ℚ := [30, 20, 10]
def winning_percentages : List ℚ := [80, 1, 19]
def winning_amounts : List ℚ := [5, 5000, 15]
def tax_rate : ℚ := 10

def calculate_profit (total_tickets : ℕ) (ticket_prices : List ℚ) (ticket_percentages : List ℚ)
  (winning_odds : List ℚ) (winning_percentages : List ℚ) (winning_amounts : List ℚ) (tax_rate : ℚ) : ℚ :=
  sorry

theorem james_profit :
  calculate_profit total_tickets ticket_prices ticket_percentages winning_odds winning_percentages winning_amounts tax_rate = 4109.50 := by
  sorry

end NUMINAMATH_CALUDE_james_profit_l294_29461


namespace NUMINAMATH_CALUDE_combined_situps_total_l294_29493

/-- Performance profile for Adam -/
def adam_profile (round : ℕ) : ℕ :=
  match round with
  | 1 => 40
  | 2 => 35
  | 3 => 30
  | 4 => 20
  | _ => 0

/-- Performance profile for Barney -/
def barney_profile (round : ℕ) : ℕ :=
  let base := 45
  let decrease := 3 * round
  max (base - decrease) 0

/-- Performance profile for Carrie -/
def carrie_profile (round : ℕ) : ℕ :=
  match round with
  | 1 | 2 => 90
  | 3 | 4 => 80
  | 5 => 70
  | _ => 0

/-- Performance profile for Jerrie -/
def jerrie_profile (round : ℕ) : ℕ :=
  match round with
  | 1 | 2 => 95
  | 3 | 4 => 101
  | 5 => 94
  | 6 => 87
  | 7 => 80
  | _ => 0

/-- Theorem stating the combined total number of sit-ups -/
theorem combined_situps_total :
  (List.sum (List.map adam_profile [1, 2, 3, 4])) +
  (List.sum (List.map barney_profile [1, 2, 3, 4, 5, 6])) +
  (List.sum (List.map carrie_profile [1, 2, 3, 4, 5])) +
  (List.sum (List.map jerrie_profile [1, 2, 3, 4, 5, 6, 7])) = 1353 := by
  sorry

end NUMINAMATH_CALUDE_combined_situps_total_l294_29493


namespace NUMINAMATH_CALUDE_inequality_solution_set_l294_29491

theorem inequality_solution_set (a : ℝ) : 
  (∃ x : ℝ, |x - 3| + |x - 4| < a) → a > 1 := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l294_29491


namespace NUMINAMATH_CALUDE_baseball_average_calculation_l294_29474

/-- Proves the required average for the remaining games to achieve a target season average -/
theorem baseball_average_calculation
  (total_games : ℕ)
  (completed_games : ℕ)
  (remaining_games : ℕ)
  (current_average : ℚ)
  (target_average : ℚ)
  (h_total : total_games = completed_games + remaining_games)
  (h_completed : completed_games = 20)
  (h_remaining : remaining_games = 10)
  (h_current : current_average = 2)
  (h_target : target_average = 3) :
  (target_average * total_games - current_average * completed_games) / remaining_games = 5 := by
  sorry

#check baseball_average_calculation

end NUMINAMATH_CALUDE_baseball_average_calculation_l294_29474


namespace NUMINAMATH_CALUDE_find_k_l294_29441

def f (n : ℤ) : ℤ :=
  if n % 2 = 1 then n + 5 else n / 2

theorem find_k (k : ℤ) (h_odd : k % 2 = 1) (h_eq : f (f (f k)) = 31) : k = 119 := by
  sorry

end NUMINAMATH_CALUDE_find_k_l294_29441


namespace NUMINAMATH_CALUDE_staircase_theorem_l294_29415

def staircase_problem (first_staircase : ℕ) (step_height : ℚ) : ℚ :=
  let second_staircase := 2 * first_staircase
  let third_staircase := second_staircase - 10
  let total_steps := first_staircase + second_staircase + third_staircase
  total_steps * step_height

theorem staircase_theorem :
  staircase_problem 20 (1/2) = 45 := by
  sorry

end NUMINAMATH_CALUDE_staircase_theorem_l294_29415


namespace NUMINAMATH_CALUDE_product_of_large_numbers_l294_29455

theorem product_of_large_numbers : (300000 : ℕ) * 300000 * 3 = 270000000000 := by
  sorry

end NUMINAMATH_CALUDE_product_of_large_numbers_l294_29455


namespace NUMINAMATH_CALUDE_some_mythical_beings_are_mystical_spirits_l294_29487

-- Define our universe
variable (U : Type)

-- Define our predicates
variable (Dragon : U → Prop)
variable (MythicalBeing : U → Prop)
variable (MysticalSpirit : U → Prop)

-- State the theorem
theorem some_mythical_beings_are_mystical_spirits
  (h1 : ∀ x, Dragon x → MythicalBeing x)
  (h2 : ∃ x, MysticalSpirit x ∧ Dragon x) :
  ∃ x, MythicalBeing x ∧ MysticalSpirit x :=
by sorry

end NUMINAMATH_CALUDE_some_mythical_beings_are_mystical_spirits_l294_29487


namespace NUMINAMATH_CALUDE_age_calculation_l294_29470

/-- Given a two-digit birth year satisfying certain conditions, prove the person's age in 1955 --/
theorem age_calculation (x y : ℕ) (h : 10 * x + y + 4 = 43) : 1955 - (1900 + 10 * x + y) = 16 := by
  sorry

end NUMINAMATH_CALUDE_age_calculation_l294_29470


namespace NUMINAMATH_CALUDE_factorization_problem_1_l294_29414

theorem factorization_problem_1 (x : ℝ) :
  x^4 - 8*x^2 + 4 = (x^2 + 2*x - 2) * (x^2 - 2*x - 2) := by
sorry

end NUMINAMATH_CALUDE_factorization_problem_1_l294_29414


namespace NUMINAMATH_CALUDE_hua_luogeng_birthday_factorization_l294_29488

theorem hua_luogeng_birthday_factorization (h : 19101112 = 1163 * 16424) :
  Nat.Prime 1163 ∧ ¬Nat.Prime 16424 := by
  sorry

end NUMINAMATH_CALUDE_hua_luogeng_birthday_factorization_l294_29488


namespace NUMINAMATH_CALUDE_count_is_thirty_l294_29430

/-- 
Counts the number of non-negative integers n less than 120 for which 
there exists an integer m divisible by 4 such that the roots of 
x^2 - nx + m = 0 are consecutive non-negative integers.
-/
def count_valid_n : ℕ := by
  sorry

/-- The main theorem stating that the count is equal to 30 -/
theorem count_is_thirty : count_valid_n = 30 := by
  sorry

end NUMINAMATH_CALUDE_count_is_thirty_l294_29430


namespace NUMINAMATH_CALUDE_mixed_gender_more_likely_l294_29468

def child_gender := Bool

def prob_all_same_gender (n : ℕ) : ℚ :=
  (1 / 2) ^ n

def prob_mixed_gender (n : ℕ) : ℚ :=
  1 - prob_all_same_gender n

theorem mixed_gender_more_likely (n : ℕ) (h : n = 3) :
  prob_mixed_gender n > prob_all_same_gender n :=
sorry

end NUMINAMATH_CALUDE_mixed_gender_more_likely_l294_29468


namespace NUMINAMATH_CALUDE_total_animal_eyes_l294_29406

theorem total_animal_eyes (num_snakes num_alligators : ℕ) 
  (snake_eyes alligator_eyes : ℕ) : ℕ :=
  by
    -- Define the number of snakes and alligators
    have h1 : num_snakes = 18 := by sorry
    have h2 : num_alligators = 10 := by sorry
    
    -- Define the number of eyes for each snake and alligator
    have h3 : snake_eyes = 2 := by sorry
    have h4 : alligator_eyes = 2 := by sorry
    
    -- Calculate total number of eyes
    have h5 : num_snakes * snake_eyes + num_alligators * alligator_eyes = 56 := by sorry
    
    exact 56

#check total_animal_eyes

end NUMINAMATH_CALUDE_total_animal_eyes_l294_29406


namespace NUMINAMATH_CALUDE_product_equals_negative_six_l294_29490

/-- Given eight real numbers satisfying certain conditions, prove that their product equals -6 -/
theorem product_equals_negative_six
  (a₁ a₂ a₃ a₄ b₁ b₂ b₃ b₄ : ℝ)
  (eq1 : a₁ * b₁ + a₂ * b₃ = 1)
  (eq2 : a₁ * b₂ + a₂ * b₄ = 0)
  (eq3 : a₃ * b₁ + a₄ * b₃ = 0)
  (eq4 : a₃ * b₂ + a₄ * b₄ = 1)
  (eq5 : a₂ * b₃ = 7) :
  a₄ * b₄ = -6 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_negative_six_l294_29490


namespace NUMINAMATH_CALUDE_numeralia_license_plate_probability_l294_29436

/-- Represents the set of possible symbols for each position in a Numeralia license plate -/
structure NumeraliaLicensePlate :=
  (vowels : Finset Char)
  (nonVowels : Finset Char)
  (digits : Finset Char)

/-- The probability of a specific valid license plate configuration in Numeralia -/
def licensePlateProbability (plate : NumeraliaLicensePlate) : ℚ :=
  1 / ((plate.vowels.card : ℚ) * (plate.nonVowels.card : ℚ) * ((plate.nonVowels.card - 1) : ℚ) * 
       (plate.digits.card + plate.vowels.card : ℚ))

/-- Theorem stating the probability of a specific valid license plate configuration in Numeralia -/
theorem numeralia_license_plate_probability 
  (plate : NumeraliaLicensePlate)
  (h1 : plate.vowels.card = 5)
  (h2 : plate.nonVowels.card = 21)
  (h3 : plate.digits.card = 10) :
  licensePlateProbability plate = 1 / 31500 := by
  sorry


end NUMINAMATH_CALUDE_numeralia_license_plate_probability_l294_29436


namespace NUMINAMATH_CALUDE_integer_solutions_count_l294_29478

theorem integer_solutions_count : ∃! (n : ℕ), ∃ (S : Finset ℤ),
  (∀ y ∈ S, (-4:ℤ) * y ≥ 2 * y + 10 ∧
            (-3:ℤ) * y ≤ 15 ∧
            (-5:ℤ) * y ≥ 3 * y + 24 ∧
            y ≤ -1) ∧
  (∀ y : ℤ, (-4:ℤ) * y ≥ 2 * y + 10 ∧
            (-3:ℤ) * y ≤ 15 ∧
            (-5:ℤ) * y ≥ 3 * y + 24 ∧
            y ≤ -1 → y ∈ S) ∧
  Finset.card S = n ∧ n = 3 :=
sorry

end NUMINAMATH_CALUDE_integer_solutions_count_l294_29478


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l294_29443

/-- A regular polygon with an exterior angle of 18 degrees has 20 sides. -/
theorem regular_polygon_sides (n : ℕ) : n > 0 → (360 : ℝ) / n = 18 → n = 20 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l294_29443


namespace NUMINAMATH_CALUDE_rs_equals_240_l294_29444

-- Define the triangle DEF
structure Triangle (DE EF : ℝ) where
  de_positive : DE > 0
  ef_positive : EF > 0

-- Define points Q, R, S, N
structure Points (D E F Q R S N : ℝ × ℝ) where
  q_on_de : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ Q = (1 - t) • D + t • E
  r_on_df : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ R = (1 - t) • D + t • F
  s_on_fq : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ S = (1 - t) • F + t • Q
  s_on_er : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ S = (1 - t) • E + t • R
  n_on_fq : ∃ t : ℝ, t > 1 ∧ N = (1 - t) • F + t • Q

-- Define the conditions
def Conditions (D E F Q R S N : ℝ × ℝ) (triangle : Triangle 600 400) (points : Points D E F Q R S N) : Prop :=
  let de := ‖E - D‖
  let dq := ‖Q - D‖
  let qe := ‖E - Q‖
  let dn := ‖N - D‖
  let sn := ‖N - S‖
  let sq := ‖Q - S‖
  de = 600 ∧ dq = qe ∧ dn = 240 ∧ sn = sq

-- Theorem statement
theorem rs_equals_240 (D E F Q R S N : ℝ × ℝ) 
  (triangle : Triangle 600 400) (points : Points D E F Q R S N) 
  (h : Conditions D E F Q R S N triangle points) : 
  ‖R - S‖ = 240 := by sorry

end NUMINAMATH_CALUDE_rs_equals_240_l294_29444


namespace NUMINAMATH_CALUDE_z_value_l294_29482

theorem z_value (x y z : ℝ) 
  (h1 : (x + y) / 2 = 4) 
  (h2 : x + y + z = 0) : 
  z = -8 := by
sorry

end NUMINAMATH_CALUDE_z_value_l294_29482


namespace NUMINAMATH_CALUDE_prob_at_most_one_value_l294_29440

/-- The probability that A hits the target -/
def prob_A : ℝ := 0.6

/-- The probability that B hits the target -/
def prob_B : ℝ := 0.7

/-- The probability that at most one of A and B hits the target -/
def prob_at_most_one : ℝ := 1 - prob_A * prob_B

theorem prob_at_most_one_value : prob_at_most_one = 0.58 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_most_one_value_l294_29440


namespace NUMINAMATH_CALUDE_lucky_years_2010_to_2014_l294_29489

/-- A year is lucky if there exists a date in that year where the product of the month and day
    equals the last two digits of the year. -/
def is_lucky_year (year : ℕ) : Prop :=
  ∃ (month day : ℕ), 1 ≤ month ∧ month ≤ 12 ∧ 1 ≤ day ∧ day ≤ 31 ∧
    month * day = year % 100

/-- 2013 is not a lucky year, while 2010, 2011, 2012, and 2014 are lucky years. -/
theorem lucky_years_2010_to_2014 :
  is_lucky_year 2010 ∧ is_lucky_year 2011 ∧ is_lucky_year 2012 ∧
  ¬is_lucky_year 2013 ∧ is_lucky_year 2014 := by
  sorry

#check lucky_years_2010_to_2014

end NUMINAMATH_CALUDE_lucky_years_2010_to_2014_l294_29489


namespace NUMINAMATH_CALUDE_inequality_system_solution_l294_29427

theorem inequality_system_solution :
  ∀ x : ℝ, (5 * (x - 1) ≤ x + 3 ∧ (x + 1) / 2 < 2 * x) ↔ (1/3 < x ∧ x ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l294_29427


namespace NUMINAMATH_CALUDE_second_element_of_sequence_l294_29409

theorem second_element_of_sequence (n : ℕ) : 
  n > 1 → (n * (n + 1)) / 2 = 78 → 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_second_element_of_sequence_l294_29409


namespace NUMINAMATH_CALUDE_harkamal_purchase_amount_l294_29467

/-- The total amount paid by Harkamal for grapes and mangoes -/
def total_amount_paid (grape_quantity : ℕ) (grape_rate : ℕ) (mango_quantity : ℕ) (mango_rate : ℕ) : ℕ :=
  grape_quantity * grape_rate + mango_quantity * mango_rate

/-- Theorem stating that Harkamal paid 1135 for his purchase -/
theorem harkamal_purchase_amount :
  total_amount_paid 8 80 9 55 = 1135 := by
  sorry


end NUMINAMATH_CALUDE_harkamal_purchase_amount_l294_29467


namespace NUMINAMATH_CALUDE_salary_calculation_l294_29483

/-- Represents the number of turbans given as part of the yearly salary -/
def turbans_per_year : ℕ := sorry

/-- The price of a turban in rupees -/
def turban_price : ℕ := 110

/-- The base salary in rupees for a full year -/
def base_salary : ℕ := 90

/-- The amount in rupees received by the servant after 9 months -/
def received_amount : ℕ := 40

/-- The number of months the servant worked -/
def months_worked : ℕ := 9

/-- The total number of months in a year -/
def months_in_year : ℕ := 12

theorem salary_calculation :
  (months_worked : ℚ) / months_in_year * (base_salary + turbans_per_year * turban_price) =
  received_amount + turban_price ∧ turbans_per_year = 1 := by sorry

end NUMINAMATH_CALUDE_salary_calculation_l294_29483


namespace NUMINAMATH_CALUDE_binomial_nine_choose_five_l294_29404

theorem binomial_nine_choose_five : Nat.choose 9 5 = 126 := by sorry

end NUMINAMATH_CALUDE_binomial_nine_choose_five_l294_29404


namespace NUMINAMATH_CALUDE_fraction_equality_l294_29402

theorem fraction_equality (x y : ℝ) (h : x / y = 2) : (x - y) / y = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l294_29402


namespace NUMINAMATH_CALUDE_function_properties_l294_29434

noncomputable def f (m : ℝ) (x : ℝ) : ℝ :=
  if x < 0 then 3^x else m - x^2

theorem function_properties :
  (∀ m < 0, ¬ ∃ x, f m x = 0) ∧
  f (1/9) (f (1/9) (-1)) = 0 := by sorry

end NUMINAMATH_CALUDE_function_properties_l294_29434


namespace NUMINAMATH_CALUDE_geometric_sequence_general_term_l294_29452

/-- Given a geometric sequence {a_n} where the first three terms are a-2, a+2, and a+8,
    prove that the general term a_n is equal to 8 · (3/2)^(n-1) -/
theorem geometric_sequence_general_term (a : ℝ) (a_n : ℕ → ℝ) :
  (a_n 1 = a - 2) →
  (a_n 2 = a + 2) →
  (a_n 3 = a + 8) →
  (∀ n : ℕ, n ≥ 1 → a_n (n + 1) / a_n n = a_n 2 / a_n 1) →
  (∀ n : ℕ, n ≥ 1 → a_n n = 8 * (3/2)^(n - 1)) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_general_term_l294_29452


namespace NUMINAMATH_CALUDE_wills_calories_burned_per_minute_l294_29447

/-- Calories burned per minute while jogging -/
def calories_burned_per_minute (initial_calories net_calories jogging_duration_minutes : ℕ) : ℚ :=
  (initial_calories - net_calories : ℚ) / jogging_duration_minutes

/-- Theorem stating the calories burned per minute for Will's specific case -/
theorem wills_calories_burned_per_minute :
  calories_burned_per_minute 900 600 30 = 10 := by
  sorry

end NUMINAMATH_CALUDE_wills_calories_burned_per_minute_l294_29447


namespace NUMINAMATH_CALUDE_f_always_above_g_iff_m_less_than_5_l294_29472

/-- The function f(x) = |x-2| -/
def f (x : ℝ) : ℝ := |x - 2|

/-- The function g(x) = -|x+3| + m -/
def g (x m : ℝ) : ℝ := -|x + 3| + m

/-- Theorem stating that f(x) > g(x) for all x if and only if m < 5 -/
theorem f_always_above_g_iff_m_less_than_5 :
  (∀ x : ℝ, f x > g x m) ↔ m < 5 := by sorry

end NUMINAMATH_CALUDE_f_always_above_g_iff_m_less_than_5_l294_29472


namespace NUMINAMATH_CALUDE_parallelogram_area_l294_29476

-- Define the parallelogram and its properties
structure Parallelogram :=
  (area : ℝ)
  (inscribed_circles : ℕ)
  (circle_radius : ℝ)
  (touching_sides : ℕ)
  (vertex_to_tangency : ℝ)

-- Define the conditions of the problem
def problem_conditions (p : Parallelogram) : Prop :=
  p.inscribed_circles = 2 ∧
  p.circle_radius = 1 ∧
  p.touching_sides = 3 ∧
  p.vertex_to_tangency = Real.sqrt 3

-- Theorem statement
theorem parallelogram_area 
  (p : Parallelogram) 
  (h : problem_conditions p) : 
  p.area = 4 * (1 + Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_l294_29476


namespace NUMINAMATH_CALUDE_circle_intersection_and_tangent_line_l294_29439

theorem circle_intersection_and_tangent_line :
  ∃ (A B C : ℝ),
    (∀ x y : ℝ, A * x^2 + A * y^2 + B * x + C = 0) ∧
    (∀ x y : ℝ, (A * x^2 + A * y^2 + B * x + C = 0) →
      ((x^2 + y^2 - 1 = 0) ∧ (x^2 - 4*x + y^2 = 0)) ∨
      ((x^2 + y^2 - 1 ≠ 0) ∧ (x^2 - 4*x + y^2 ≠ 0))) ∧
    (∃ x₀ y₀ : ℝ,
      A * x₀^2 + A * y₀^2 + B * x₀ + C = 0 ∧
      x₀ - Real.sqrt 3 * y₀ - 6 = 0 ∧
      ∀ x y : ℝ, A * x^2 + A * y^2 + B * x + C = 0 →
        (x - Real.sqrt 3 * y - 6)^2 ≥ (x₀ - Real.sqrt 3 * y₀ - 6)^2) :=
by sorry

end NUMINAMATH_CALUDE_circle_intersection_and_tangent_line_l294_29439


namespace NUMINAMATH_CALUDE_max_working_groups_is_18_more_than_18_groups_impossible_l294_29435

/-- Represents a working group formation problem -/
structure WorkingGroupProblem where
  totalTeachers : ℕ
  groupSize : ℕ
  maxGroupsPerTeacher : ℕ

/-- Calculates the maximum number of working groups that can be formed -/
def maxWorkingGroups (problem : WorkingGroupProblem) : ℕ :=
  min
    (problem.totalTeachers * problem.maxGroupsPerTeacher / problem.groupSize)
    ((problem.totalTeachers * problem.maxGroupsPerTeacher) / problem.groupSize)

/-- The specific problem instance -/
def specificProblem : WorkingGroupProblem :=
  { totalTeachers := 36
    groupSize := 4
    maxGroupsPerTeacher := 2 }

/-- Theorem stating that the maximum number of working groups is 18 -/
theorem max_working_groups_is_18 :
  maxWorkingGroups specificProblem = 18 := by
  sorry

/-- Theorem proving that more than 18 groups is impossible -/
theorem more_than_18_groups_impossible (n : ℕ) :
  n > 18 → n * specificProblem.groupSize > specificProblem.totalTeachers * specificProblem.maxGroupsPerTeacher := by
  sorry

end NUMINAMATH_CALUDE_max_working_groups_is_18_more_than_18_groups_impossible_l294_29435


namespace NUMINAMATH_CALUDE_system_solution_l294_29471

theorem system_solution (x y : ℝ) :
  (2 * x + 3 * y = 14) → (x + 4 * y = 11) → (x - y = 3) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l294_29471


namespace NUMINAMATH_CALUDE_absent_percentage_l294_29497

theorem absent_percentage (total_students : ℕ) (boys : ℕ) (girls : ℕ)
  (h_total : total_students = 180)
  (h_boys : boys = 100)
  (h_girls : girls = 80)
  (h_sum : total_students = boys + girls)
  (absent_boys_fraction : ℚ)
  (absent_girls_fraction : ℚ)
  (h_absent_boys : absent_boys_fraction = 1 / 5)
  (h_absent_girls : absent_girls_fraction = 1 / 4) :
  (absent_boys_fraction * boys + absent_girls_fraction * girls) / total_students = 2 / 9 := by
sorry

end NUMINAMATH_CALUDE_absent_percentage_l294_29497


namespace NUMINAMATH_CALUDE_box_height_is_twelve_l294_29498

-- Define the box dimensions and costs
def box_base_length : ℝ := 20
def box_base_width : ℝ := 20
def cost_per_box : ℝ := 0.50
def total_volume_needed : ℝ := 2160000
def min_spending : ℝ := 225

-- Theorem to prove
theorem box_height_is_twelve :
  ∃ (h : ℝ), h > 0 ∧ 
    (total_volume_needed / (box_base_length * box_base_width * h)) * cost_per_box ≥ min_spending ∧
    ∀ (h' : ℝ), h' > h → 
      (total_volume_needed / (box_base_length * box_base_width * h')) * cost_per_box < min_spending ∧
    h = 12 :=
by sorry

end NUMINAMATH_CALUDE_box_height_is_twelve_l294_29498


namespace NUMINAMATH_CALUDE_max_value_of_quadratic_l294_29429

theorem max_value_of_quadratic (x : ℝ) (h1 : 0 < x) (h2 : x < 1/3) :
  x * (1 - 3*x) ≤ 1/12 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_quadratic_l294_29429


namespace NUMINAMATH_CALUDE_part_one_part_two_l294_29446

-- Define the sets P and Q
def P (a : ℝ) : Set ℝ := {x | a + 1 ≤ x ∧ x ≤ 2 * a + 1}
def Q : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}

-- Part 1
theorem part_one : (Set.univ \ P 3) ∩ Q = {x | -2 ≤ x ∧ x < 4} := by sorry

-- Part 2
theorem part_two : {a : ℝ | P a ⊂ Q ∧ P a ≠ ∅} = {a : ℝ | 0 ≤ a ∧ a ≤ 2} := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l294_29446


namespace NUMINAMATH_CALUDE_article_cost_price_l294_29453

theorem article_cost_price (C : ℝ) (S : ℝ) : 
  S = 1.05 * C →                           -- Selling price is 5% more than cost price
  (1.05 * C - 5) = 1.1 * (0.95 * C) →      -- New selling price (5 less) is 10% more than new cost price (5% less)
  C = 1000 := by
sorry

end NUMINAMATH_CALUDE_article_cost_price_l294_29453


namespace NUMINAMATH_CALUDE_root_sum_reciprocals_l294_29494

theorem root_sum_reciprocals (p q r s : ℂ) : 
  (p^4 + 6*p^3 + 11*p^2 + 7*p + 5 = 0) →
  (q^4 + 6*q^3 + 11*q^2 + 7*q + 5 = 0) →
  (r^4 + 6*r^3 + 11*r^2 + 7*r + 5 = 0) →
  (s^4 + 6*s^3 + 11*s^2 + 7*s + 5 = 0) →
  1/(p*q) + 1/(p*r) + 1/(p*s) + 1/(q*r) + 1/(q*s) + 1/(r*s) = 11/5 := by
sorry

end NUMINAMATH_CALUDE_root_sum_reciprocals_l294_29494


namespace NUMINAMATH_CALUDE_north_movement_representation_l294_29469

/-- Represents the direction of movement -/
inductive Direction
  | North
  | South

/-- Represents a movement with distance and direction -/
structure Movement where
  distance : ℝ
  direction : Direction

/-- Converts a movement to its numerical representation -/
def movementToMeters (m : Movement) : ℝ :=
  match m.direction with
  | Direction.North => m.distance
  | Direction.South => -m.distance

theorem north_movement_representation (d : ℝ) (h : d > 0) :
  let southMovement : Movement := ⟨d, Direction.South⟩
  let northMovement : Movement := ⟨d, Direction.North⟩
  movementToMeters southMovement = -d →
  movementToMeters northMovement = d :=
by sorry

end NUMINAMATH_CALUDE_north_movement_representation_l294_29469


namespace NUMINAMATH_CALUDE_place_value_decomposition_l294_29486

theorem place_value_decomposition :
  (286 = 200 + 80 + 6) ∧
  (7560 = 7000 + 500 + 60) ∧
  (2048 = 2000 + 40 + 8) ∧
  (8009 = 8000 + 9) ∧
  (3070 = 3000 + 70) := by
  sorry

end NUMINAMATH_CALUDE_place_value_decomposition_l294_29486


namespace NUMINAMATH_CALUDE_decagon_equilateral_triangles_l294_29442

/-- A regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  vertices : Fin n → ℝ × ℝ

/-- An equilateral triangle -/
structure EquilateralTriangle where
  vertices : Fin 3 → ℝ × ℝ

/-- The number of distinct equilateral triangles with at least two vertices 
    from a given set of points -/
def countDistinctEquilateralTriangles (points : Set (ℝ × ℝ)) : ℕ := sorry

/-- The theorem stating the number of distinct equilateral triangles 
    in a regular decagon -/
theorem decagon_equilateral_triangles (d : RegularPolygon 10) :
  countDistinctEquilateralTriangles (Set.range d.vertices) = 90 := by sorry

end NUMINAMATH_CALUDE_decagon_equilateral_triangles_l294_29442


namespace NUMINAMATH_CALUDE_condition_analysis_l294_29450

theorem condition_analysis (a b c : ℝ) : 
  (∀ a b c : ℝ, a * c^2 < b * c^2 → a < b) ∧ 
  (∃ a b c : ℝ, a < b ∧ a * c^2 ≥ b * c^2) :=
by sorry

end NUMINAMATH_CALUDE_condition_analysis_l294_29450


namespace NUMINAMATH_CALUDE_divisibility_of_sum_l294_29448

theorem divisibility_of_sum : 
  let x : ℕ := 50 + 100 + 140 + 180 + 320 + 400 + 5000
  (x % 5 = 0 ∧ x % 10 = 0) ∧ (x % 20 ≠ 0 ∧ x % 40 ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_of_sum_l294_29448


namespace NUMINAMATH_CALUDE_perpendicular_slope_l294_29417

theorem perpendicular_slope (x y : ℝ) :
  (3 * x - 4 * y = 8) →
  (∃ m : ℝ, m = -4/3 ∧ m * (3/4) = -1) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_slope_l294_29417


namespace NUMINAMATH_CALUDE_backpack_traverse_time_l294_29421

/-- Theorem: Time taken to carry backpack through obstacle course --/
theorem backpack_traverse_time (total_time door_time second_traverse_minutes second_traverse_seconds : ℕ) :
  let second_traverse_time := second_traverse_minutes * 60 + second_traverse_seconds
  let remaining_time := total_time - (door_time + second_traverse_time)
  total_time = 874 ∧ door_time = 73 ∧ second_traverse_minutes = 5 ∧ second_traverse_seconds = 58 →
  remaining_time = 443 := by
  sorry

end NUMINAMATH_CALUDE_backpack_traverse_time_l294_29421


namespace NUMINAMATH_CALUDE_fair_attendance_percentage_l294_29423

/-- The percent of projected attendance that was the actual attendance --/
theorem fair_attendance_percentage (A : ℝ) (V W : ℝ) : 
  let projected_attendance := 1.25 * A
  let actual_attendance := 0.8 * A
  (actual_attendance / projected_attendance) * 100 = 64 := by
  sorry

end NUMINAMATH_CALUDE_fair_attendance_percentage_l294_29423


namespace NUMINAMATH_CALUDE_tetrahedron_sum_l294_29433

/-- A regular tetrahedron is a three-dimensional shape with four congruent equilateral triangular faces. -/
structure RegularTetrahedron where
  -- We don't need to define any fields here, as we're only interested in its properties

/-- The number of edges in a regular tetrahedron -/
def num_edges (t : RegularTetrahedron) : ℕ := 6

/-- The number of vertices in a regular tetrahedron -/
def num_vertices (t : RegularTetrahedron) : ℕ := 4

/-- The number of faces in a regular tetrahedron -/
def num_faces (t : RegularTetrahedron) : ℕ := 4

/-- The theorem stating that the sum of edges, vertices, and faces of a regular tetrahedron is 14 -/
theorem tetrahedron_sum (t : RegularTetrahedron) : 
  num_edges t + num_vertices t + num_faces t = 14 := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_sum_l294_29433


namespace NUMINAMATH_CALUDE_sum_of_roots_is_twelve_l294_29465

/-- A function g: ℝ → ℝ that satisfies g(3+x) = g(3-x) for all real x -/
def SymmetricAboutThree (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (3 + x) = g (3 - x)

/-- The proposition that g has exactly four distinct real roots -/
def HasFourDistinctRoots (g : ℝ → ℝ) : Prop :=
  ∃ (a b c d : ℝ), (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) ∧
    (g a = 0 ∧ g b = 0 ∧ g c = 0 ∧ g d = 0) ∧
    (∀ x : ℝ, g x = 0 → (x = a ∨ x = b ∨ x = c ∨ x = d))

/-- The theorem stating that the sum of roots is 12 -/
theorem sum_of_roots_is_twelve (g : ℝ → ℝ) 
    (h1 : SymmetricAboutThree g) (h2 : HasFourDistinctRoots g) : 
    ∃ (a b c d : ℝ), (g a = 0 ∧ g b = 0 ∧ g c = 0 ∧ g d = 0) ∧ (a + b + c + d = 12) :=
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_is_twelve_l294_29465


namespace NUMINAMATH_CALUDE_sum_of_integers_l294_29405

theorem sum_of_integers (m n : ℕ+) (h1 : m * n = 2 * (m + n)) (h2 : m * n = 6 * (m - n)) :
  m + n = 9 := by sorry

end NUMINAMATH_CALUDE_sum_of_integers_l294_29405


namespace NUMINAMATH_CALUDE_boat_downstream_distance_l294_29437

/-- Proof of downstream distance traveled by a boat given upstream travel time and distance, and stream speed. -/
theorem boat_downstream_distance
  (upstream_distance : ℝ)
  (upstream_time : ℝ)
  (stream_speed : ℝ)
  (downstream_time : ℝ)
  (h1 : upstream_distance = 75)
  (h2 : upstream_time = 15)
  (h3 : stream_speed = 3.75)
  (h4 : downstream_time = 8) :
  let upstream_speed := upstream_distance / upstream_time
  let boat_speed := upstream_speed + stream_speed
  let downstream_speed := boat_speed + stream_speed
  downstream_speed * downstream_time = 100 := by
  sorry


end NUMINAMATH_CALUDE_boat_downstream_distance_l294_29437


namespace NUMINAMATH_CALUDE_parallel_lines_a_equals_one_l294_29411

/-- Two lines in the xy-plane -/
structure ParallelLines where
  /-- The first line equation: x + 2y - 4 = 0 -/
  line1 : ℝ → ℝ → Prop := fun x y => x + 2*y - 4 = 0
  /-- The second line equation: ax + 2y + 6 = 0 -/
  line2 : ℝ → ℝ → ℝ → Prop := fun a x y => a*x + 2*y + 6 = 0
  /-- The lines are parallel -/
  parallel : ∀ (a : ℝ), (∀ x y, line1 x y ↔ ∃ k, line2 a (x + k) (y + k))

/-- If two lines are parallel as defined, then a = 1 -/
theorem parallel_lines_a_equals_one (pl : ParallelLines) : ∃ a, ∀ x y, pl.line2 a x y ↔ pl.line2 1 x y := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_a_equals_one_l294_29411


namespace NUMINAMATH_CALUDE_garden_perimeter_l294_29477

/-- The perimeter of a rectangle given its length and width -/
def rectanglePerimeter (length width : ℝ) : ℝ := 2 * (length + width)

/-- Theorem: The perimeter of a rectangular garden with length 25 meters and width 15 meters is 80 meters -/
theorem garden_perimeter :
  rectanglePerimeter 25 15 = 80 := by
  sorry

end NUMINAMATH_CALUDE_garden_perimeter_l294_29477


namespace NUMINAMATH_CALUDE_rationalize_denominator_l294_29408

theorem rationalize_denominator :
  ∃ (A B C D E : ℤ),
    (5 : ℝ) / (4 * Real.sqrt 7 + 3 * Real.sqrt 13) = A * Real.sqrt B + C * Real.sqrt D ∧
    B < D ∧
    A = -4 ∧
    B = 7 ∧
    C = 3 ∧
    D = 13 ∧
    E = 1 :=
by sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l294_29408


namespace NUMINAMATH_CALUDE_sam_initial_pennies_l294_29401

/-- The number of pennies Sam spent -/
def pennies_spent : ℕ := 93

/-- The number of pennies Sam has left -/
def pennies_left : ℕ := 5

/-- The initial number of pennies in Sam's bank -/
def initial_pennies : ℕ := pennies_spent + pennies_left

theorem sam_initial_pennies : initial_pennies = 98 := by
  sorry

end NUMINAMATH_CALUDE_sam_initial_pennies_l294_29401


namespace NUMINAMATH_CALUDE_race_total_time_l294_29425

theorem race_total_time (total_runners : ℕ) (first_group : ℕ) (first_time : ℕ) (extra_time : ℕ) 
  (h1 : total_runners = 8)
  (h2 : first_group = 5)
  (h3 : first_time = 8)
  (h4 : extra_time = 2) :
  first_group * first_time + (total_runners - first_group) * (first_time + extra_time) = 70 := by
  sorry

end NUMINAMATH_CALUDE_race_total_time_l294_29425


namespace NUMINAMATH_CALUDE_trebled_result_proof_l294_29495

theorem trebled_result_proof (initial_number : ℕ) : 
  initial_number = 18 → 
  3 * (2 * initial_number + 5) = 123 := by
sorry

end NUMINAMATH_CALUDE_trebled_result_proof_l294_29495


namespace NUMINAMATH_CALUDE_y_expression_equivalence_l294_29413

theorem y_expression_equivalence (x : ℝ) : 
  Real.sqrt ((x - 2)^2) + Real.sqrt (x^2 + 4*x + 5) = 
  |x - 2| + Real.sqrt ((x + 2)^2 + 1) := by
  sorry

end NUMINAMATH_CALUDE_y_expression_equivalence_l294_29413


namespace NUMINAMATH_CALUDE_classical_mechanics_not_incorrect_l294_29499

/-- Represents a scientific theory -/
structure ScientificTheory where
  name : String
  hasLimitations : Bool
  isIncorrect : Bool

/-- Classical mechanics as a scientific theory -/
def classicalMechanics : ScientificTheory := {
  name := "Classical Mechanics"
  hasLimitations := true
  isIncorrect := false
}

/-- Truth has relativity -/
axiom truth_relativity : Prop

/-- Scientific exploration is endless -/
axiom endless_exploration : Prop

/-- Theorem stating that classical mechanics is not an incorrect scientific theory -/
theorem classical_mechanics_not_incorrect :
  classicalMechanics.hasLimitations ∧ truth_relativity ∧ endless_exploration →
  ¬classicalMechanics.isIncorrect := by
  sorry


end NUMINAMATH_CALUDE_classical_mechanics_not_incorrect_l294_29499


namespace NUMINAMATH_CALUDE_greatest_difference_under_300_l294_29403

/-- Represents a three-digit positive integer -/
structure ThreeDigitInt where
  value : ℕ
  is_three_digit : 100 ≤ value ∧ value ≤ 999

/-- Given two three-digit positive integers with reversed digits -/
def reversed_digits (q r : ThreeDigitInt) : Prop :=
  ∃ (a b c : ℕ), 
    0 < a ∧ a ≤ 9 ∧
    0 ≤ b ∧ b ≤ 9 ∧
    0 < c ∧ c ≤ 9 ∧
    q.value = 100 * a + 10 * b + c ∧
    r.value = 100 * c + 10 * b + a

theorem greatest_difference_under_300 (q r : ThreeDigitInt) 
  (h_reversed : reversed_digits q r) 
  (h_diff : q.value > r.value ∧ q.value - r.value < 300) :
  q.value - r.value ≤ 297 ∧ 
  ∃ (q' r' : ThreeDigitInt), reversed_digits q' r' ∧ q'.value - r'.value = 297 :=
sorry

end NUMINAMATH_CALUDE_greatest_difference_under_300_l294_29403


namespace NUMINAMATH_CALUDE_f_symmetry_l294_29496

-- Define the function f
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x + 2

-- State the theorem
theorem f_symmetry (a b : ℝ) : 
  f a b (-5) = 3 → f a b 5 = 1 := by
  sorry

end NUMINAMATH_CALUDE_f_symmetry_l294_29496


namespace NUMINAMATH_CALUDE_hcf_problem_l294_29481

theorem hcf_problem (a b : ℕ+) (h1 : a * b = 1991) (h2 : Nat.lcm a b = 181) :
  Nat.gcd a b = 11 := by
  sorry

end NUMINAMATH_CALUDE_hcf_problem_l294_29481


namespace NUMINAMATH_CALUDE_job_completion_time_solution_l294_29460

/-- Represents the time taken by three machines working together to complete a job -/
def job_completion_time (y : ℝ) : Prop :=
  let machine_a_time := y + 4
  let machine_b_time := y + 3
  let machine_c_time := 3 * y
  (1 / machine_a_time) + (1 / machine_b_time) + (1 / machine_c_time) = 1 / y

/-- Proves that the job completion time satisfies the given equation -/
theorem job_completion_time_solution :
  ∃ y : ℝ, job_completion_time y ∧ y = (-14 + Real.sqrt 296) / 10 := by
  sorry

end NUMINAMATH_CALUDE_job_completion_time_solution_l294_29460


namespace NUMINAMATH_CALUDE_complex_coordinate_l294_29422

theorem complex_coordinate (z : ℂ) (h : Complex.I * z = 1 + 2 * Complex.I) : z = 2 - Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_coordinate_l294_29422


namespace NUMINAMATH_CALUDE_birds_on_fence_l294_29400

theorem birds_on_fence (initial_birds : ℕ) (new_birds : ℕ) : 
  initial_birds = 1 → new_birds = 4 → initial_birds + new_birds = 5 := by
  sorry

end NUMINAMATH_CALUDE_birds_on_fence_l294_29400


namespace NUMINAMATH_CALUDE_sum_first_10_even_integers_l294_29464

/-- The sum of the first n positive even integers -/
def sum_first_n_even_integers (n : ℕ) : ℕ :=
  2 * n * (n + 1)

/-- Theorem: The sum of the first 10 positive even integers is 110 -/
theorem sum_first_10_even_integers :
  sum_first_n_even_integers 10 = 110 := by
  sorry

end NUMINAMATH_CALUDE_sum_first_10_even_integers_l294_29464


namespace NUMINAMATH_CALUDE_hex_pattern_theorem_l294_29492

/-- Represents a hexagonal tile pattern -/
structure HexPattern where
  blue_tiles : ℕ
  green_tiles : ℕ
  red_tiles : ℕ

/-- Creates a new pattern by adding green and red tiles -/
def add_border (initial : HexPattern) (green_layers : ℕ) : HexPattern :=
  let new_green := initial.green_tiles + green_layers * 24
  let new_red := 12
  { blue_tiles := initial.blue_tiles,
    green_tiles := new_green,
    red_tiles := new_red }

theorem hex_pattern_theorem (initial : HexPattern) :
  initial.blue_tiles = 20 →
  initial.green_tiles = 9 →
  let new_pattern := add_border initial 2
  new_pattern.red_tiles = 12 ∧
  new_pattern.green_tiles + new_pattern.red_tiles - new_pattern.blue_tiles = 25 := by
  sorry

end NUMINAMATH_CALUDE_hex_pattern_theorem_l294_29492
