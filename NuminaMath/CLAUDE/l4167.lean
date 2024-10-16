import Mathlib

namespace NUMINAMATH_CALUDE_max_factors_upper_bound_max_factors_achievable_max_factors_is_maximum_l4167_416797

def max_factors (b n : ℕ+) : ℕ :=
  sorry

theorem max_factors_upper_bound (b n : ℕ+) (hb : b ≤ 15) (hn : n ≤ 20) :
  max_factors b n ≤ 861 :=
sorry

theorem max_factors_achievable :
  ∃ (b n : ℕ+), b ≤ 15 ∧ n ≤ 20 ∧ max_factors b n = 861 :=
sorry

theorem max_factors_is_maximum :
  ∀ (b n : ℕ+), b ≤ 15 → n ≤ 20 → max_factors b n ≤ 861 :=
sorry

end NUMINAMATH_CALUDE_max_factors_upper_bound_max_factors_achievable_max_factors_is_maximum_l4167_416797


namespace NUMINAMATH_CALUDE_line_circle_intersection_l4167_416787

/-- The line equation y = √3 * x + m -/
def line_equation (x y m : ℝ) : Prop := y = Real.sqrt 3 * x + m

/-- The circle equation x^2 + (y - 3)^2 = 6 -/
def circle_equation (x y : ℝ) : Prop := x^2 + (y - 3)^2 = 6

/-- Two points A and B on both the line and the circle -/
def intersection_points (A B : ℝ × ℝ) (m : ℝ) : Prop :=
  line_equation A.1 A.2 m ∧ circle_equation A.1 A.2 ∧
  line_equation B.1 B.2 m ∧ circle_equation B.1 B.2

/-- The distance between points A and B is 2√2 -/
def distance_condition (A B : ℝ × ℝ) : Prop :=
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = 8

theorem line_circle_intersection (m : ℝ) :
  (∃ A B : ℝ × ℝ, intersection_points A B m ∧ distance_condition A B) →
  m = -1 ∨ m = 7 := by sorry

end NUMINAMATH_CALUDE_line_circle_intersection_l4167_416787


namespace NUMINAMATH_CALUDE_rooster_earnings_l4167_416778

def price_per_kg : ℚ := 1/2

def rooster1_weight : ℚ := 30
def rooster2_weight : ℚ := 40

def total_earnings : ℚ := price_per_kg * (rooster1_weight + rooster2_weight)

theorem rooster_earnings : total_earnings = 35 := by
  sorry

end NUMINAMATH_CALUDE_rooster_earnings_l4167_416778


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l4167_416741

theorem sufficient_not_necessary_condition :
  (∃ a b : ℝ, a < 0 ∧ -1 < b ∧ b < 0 → a + a * b < 0) ∧
  (∃ a b : ℝ, a + a * b < 0 ∧ ¬(a < 0 ∧ -1 < b ∧ b < 0)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l4167_416741


namespace NUMINAMATH_CALUDE_dodecagon_pie_trim_l4167_416724

theorem dodecagon_pie_trim (d : ℝ) (h : d = 8) : ∃ (a b : ℤ),
  (π * (d / 2)^2 - 3 * (d / 2)^2 = a * π - b) ∧ (a + b = 64) := by
  sorry

end NUMINAMATH_CALUDE_dodecagon_pie_trim_l4167_416724


namespace NUMINAMATH_CALUDE_parabola_vertex_l4167_416756

/-- The parabola defined by y = -x^2 + cx + d -/
def parabola (c d : ℝ) (x : ℝ) : ℝ := -x^2 + c*x + d

/-- The solution set of the inequality -x^2 + cx + d ≤ 0 -/
def solution_set (c d : ℝ) : Set ℝ := {x | x ∈ Set.Icc (-6) (-1) ∨ x ∈ Set.Ici 4}

theorem parabola_vertex (c d : ℝ) :
  (solution_set c d = {x | x ∈ Set.Icc (-6) (-1) ∨ x ∈ Set.Ici 4}) →
  (∃ (x y : ℝ), x = 7/2 ∧ y = -171/4 ∧
    ∀ (t : ℝ), parabola c d t ≤ parabola c d x) :=
by sorry

end NUMINAMATH_CALUDE_parabola_vertex_l4167_416756


namespace NUMINAMATH_CALUDE_least_positive_integer_to_multiple_of_five_l4167_416754

theorem least_positive_integer_to_multiple_of_five : 
  ∃ (n : ℕ), n > 0 ∧ (∀ m : ℕ, m > 0 → (624 + m) % 5 = 0 → m ≥ n) ∧ (624 + n) % 5 = 0 :=
by sorry

end NUMINAMATH_CALUDE_least_positive_integer_to_multiple_of_five_l4167_416754


namespace NUMINAMATH_CALUDE_fifteen_factorial_base_eight_zeroes_l4167_416782

/-- The number of trailing zeroes in n! when written in base b --/
def trailingZeroes (n : ℕ) (b : ℕ) : ℕ :=
  sorry

/-- The factorial function --/
def factorial (n : ℕ) : ℕ :=
  sorry

theorem fifteen_factorial_base_eight_zeroes :
  trailingZeroes (factorial 15) 8 = 3 :=
sorry

end NUMINAMATH_CALUDE_fifteen_factorial_base_eight_zeroes_l4167_416782


namespace NUMINAMATH_CALUDE_hans_age_l4167_416731

theorem hans_age (hans_age josiah_age : ℕ) : 
  josiah_age = 3 * hans_age →
  hans_age + 3 + josiah_age + 3 = 66 →
  hans_age = 15 := by
sorry

end NUMINAMATH_CALUDE_hans_age_l4167_416731


namespace NUMINAMATH_CALUDE_book_pages_proof_l4167_416755

/-- Calculates the number of digits used to number pages from 1 to n -/
def digits_used (n : ℕ) : ℕ := sorry

/-- The number of pages in the book -/
def num_pages : ℕ := 155

/-- The total number of digits used to number all pages -/
def total_digits : ℕ := 357

theorem book_pages_proof : digits_used num_pages = total_digits := by sorry

end NUMINAMATH_CALUDE_book_pages_proof_l4167_416755


namespace NUMINAMATH_CALUDE_sin_alpha_abs_value_l4167_416763

/-- Theorem: If point P(3a, 4a) lies on the terminal side of angle α, where a ≠ 0, then |sin α| = 4/5 -/
theorem sin_alpha_abs_value (a : ℝ) (α : ℝ) (ha : a ≠ 0) :
  let P : ℝ × ℝ := (3 * a, 4 * a)
  (P.1 = 3 * a ∧ P.2 = 4 * a) → |Real.sin α| = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_sin_alpha_abs_value_l4167_416763


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l4167_416769

theorem partial_fraction_decomposition (x A B C : ℝ) (h1 : x ≠ 2) (h2 : x ≠ 4) :
  (6 * x) / ((x - 4) * (x - 2)^2) = A / (x - 4) + B / (x - 2) + C / (x - 2)^2 ↔ 
  A = 3 ∧ B = -3 ∧ C = -6 := by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l4167_416769


namespace NUMINAMATH_CALUDE_greatest_whole_number_satisfying_inequalities_l4167_416746

theorem greatest_whole_number_satisfying_inequalities :
  ∃ (n : ℕ), n = 1 ∧
  (∀ (x : ℝ), (x > n → ¬(3 * x - 5 < 1 - x ∧ 2 * x + 4 ≤ 8))) ∧
  (3 * n - 5 < 1 - n ∧ 2 * n + 4 ≤ 8) :=
by sorry

end NUMINAMATH_CALUDE_greatest_whole_number_satisfying_inequalities_l4167_416746


namespace NUMINAMATH_CALUDE_lakers_win_probability_l4167_416729

/-- The probability of the Celtics winning a single game -/
def p_celtics : ℚ := 3/4

/-- The probability of the Lakers winning a single game -/
def p_lakers : ℚ := 1 - p_celtics

/-- The number of games needed to win the series -/
def games_to_win : ℕ := 4

/-- The maximum number of games in the series -/
def max_games : ℕ := 2 * games_to_win - 1

/-- The probability of the Lakers winning the NBA finals in exactly 7 games -/
def lakers_win_in_seven : ℚ := 540/16384

theorem lakers_win_probability :
  lakers_win_in_seven = (Nat.choose 6 3 : ℚ) * p_lakers^3 * p_celtics^3 * p_lakers :=
sorry

end NUMINAMATH_CALUDE_lakers_win_probability_l4167_416729


namespace NUMINAMATH_CALUDE_candy_distribution_l4167_416757

theorem candy_distribution (total_candies : ℕ) 
  (lollipops_per_boy : ℕ) (candy_canes_per_girl : ℕ) : 
  total_candies = 90 →
  lollipops_per_boy = 3 →
  candy_canes_per_girl = 2 →
  (total_candies / 3 : ℕ) % lollipops_per_boy = 0 →
  ((2 * total_candies / 3) : ℕ) % candy_canes_per_girl = 0 →
  (total_candies / 3 / lollipops_per_boy : ℕ) + 
  ((2 * total_candies / 3) / candy_canes_per_girl : ℕ) = 40 :=
by sorry

end NUMINAMATH_CALUDE_candy_distribution_l4167_416757


namespace NUMINAMATH_CALUDE_point_difference_l4167_416725

def wildcats_rate : ℝ := 2.5
def panthers_rate : ℝ := 1.3
def half_duration : ℝ := 24

theorem point_difference : 
  wildcats_rate * half_duration - panthers_rate * half_duration = 28.8 := by
sorry

end NUMINAMATH_CALUDE_point_difference_l4167_416725


namespace NUMINAMATH_CALUDE_unique_solution_system_l4167_416790

theorem unique_solution_system (a b c d e : Real) :
  a ∈ Set.Icc (-2 : Real) 2 ∧
  b ∈ Set.Icc (-2 : Real) 2 ∧
  c ∈ Set.Icc (-2 : Real) 2 ∧
  d ∈ Set.Icc (-2 : Real) 2 ∧
  e ∈ Set.Icc (-2 : Real) 2 ∧
  a + b + c + d + e = 0 ∧
  a^3 + b^3 + c^3 + d^3 + e^3 = 0 ∧
  a^5 + b^5 + c^5 + d^5 + e^5 = 10 →
  (a = 2 ∧
   b = (Real.sqrt 5 - 1) / 2 ∧
   c = (Real.sqrt 5 - 1) / 2 ∧
   d = -(1 + Real.sqrt 5) / 2 ∧
   e = -(1 + Real.sqrt 5) / 2) :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_system_l4167_416790


namespace NUMINAMATH_CALUDE_vinegar_left_is_60_l4167_416702

/-- Represents the pickle-making scenario with given supplies and rules. -/
structure PickleScenario where
  jars : ℕ
  cucumbers : ℕ
  initial_vinegar : ℕ
  pickles_per_cucumber : ℕ
  pickles_per_jar : ℕ
  vinegar_per_jar : ℕ

/-- Calculates the amount of vinegar left after making pickles. -/
def vinegar_left (scenario : PickleScenario) : ℕ :=
  let total_pickles := scenario.cucumbers * scenario.pickles_per_cucumber
  let max_jarred_pickles := scenario.jars * scenario.pickles_per_jar
  let actual_jarred_pickles := min total_pickles max_jarred_pickles
  let jars_used := actual_jarred_pickles / scenario.pickles_per_jar
  let vinegar_used := jars_used * scenario.vinegar_per_jar
  scenario.initial_vinegar - vinegar_used

/-- Theorem stating that given the specific scenario, 60 oz of vinegar will be left. -/
theorem vinegar_left_is_60 :
  let scenario : PickleScenario := {
    jars := 4,
    cucumbers := 10,
    initial_vinegar := 100,
    pickles_per_cucumber := 6,
    pickles_per_jar := 12,
    vinegar_per_jar := 10
  }
  vinegar_left scenario = 60 := by sorry

end NUMINAMATH_CALUDE_vinegar_left_is_60_l4167_416702


namespace NUMINAMATH_CALUDE_binary_subtraction_l4167_416706

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

def a : List Bool := [true, true, true, true, true, true, true, true, true, true, true]
def b : List Bool := [true, true, true, true, true, true, true]

theorem binary_subtraction :
  binary_to_decimal a - binary_to_decimal b = 1920 := by sorry

end NUMINAMATH_CALUDE_binary_subtraction_l4167_416706


namespace NUMINAMATH_CALUDE_consecutive_non_prime_powers_l4167_416743

/-- A number is a prime power if it can be expressed as p^k where p is prime and k ≥ 1 -/
def IsPrimePower (n : ℕ) : Prop :=
  ∃ (p k : ℕ), Prime p ∧ k ≥ 1 ∧ n = p^k

theorem consecutive_non_prime_powers (N : ℕ) (h : N > 0) :
  ∃ (M : ℤ), ∀ (i : ℕ), i < N → ¬IsPrimePower (Int.toNat (M + i)) :=
sorry

end NUMINAMATH_CALUDE_consecutive_non_prime_powers_l4167_416743


namespace NUMINAMATH_CALUDE_floor_sum_inequality_floor_fractional_part_max_n_value_l4167_416726

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ := Int.floor x

-- Proposition B
theorem floor_sum_inequality (x y : ℝ) : floor x + floor y ≤ floor (x + y) := by sorry

-- Proposition C
theorem floor_fractional_part (x : ℝ) : 0 ≤ x - floor x ∧ x - floor x < 1 := by sorry

-- Proposition D
def satisfies_conditions (t : ℝ) (n : ℕ) : Prop :=
  ∀ k ∈ Finset.range (n - 2), floor (t ^ (k + 3)) = k + 1

theorem max_n_value :
  (∃ n : ℕ, n > 2 ∧ ∃ t : ℝ, satisfies_conditions t n) →
  (∀ n : ℕ, n > 5 → ¬∃ t : ℝ, satisfies_conditions t n) := by sorry

end NUMINAMATH_CALUDE_floor_sum_inequality_floor_fractional_part_max_n_value_l4167_416726


namespace NUMINAMATH_CALUDE_multiplicative_inverse_208_mod_307_l4167_416708

theorem multiplicative_inverse_208_mod_307 : ∃ x : ℕ, x < 307 ∧ (208 * x) % 307 = 1 :=
by
  use 240
  sorry

end NUMINAMATH_CALUDE_multiplicative_inverse_208_mod_307_l4167_416708


namespace NUMINAMATH_CALUDE_mystery_number_l4167_416740

theorem mystery_number : ∃ x : ℤ, x + 45 = 92 ∧ x = 47 := by
  sorry

end NUMINAMATH_CALUDE_mystery_number_l4167_416740


namespace NUMINAMATH_CALUDE_max_min_x_plus_y_l4167_416722

theorem max_min_x_plus_y (x y : ℝ) (h : x^2 + y^2 - 4*x + 2*y + 2 = 0) :
  (∃ (a b : ℝ), (∀ (x' y' : ℝ), x'^2 + y'^2 - 4*x' + 2*y' + 2 = 0 → x' + y' ≤ a ∧ b ≤ x' + y') ∧
  a = 1 + Real.sqrt 6 ∧ b = 1 - Real.sqrt 6) :=
by sorry

end NUMINAMATH_CALUDE_max_min_x_plus_y_l4167_416722


namespace NUMINAMATH_CALUDE_inlet_fill_rate_l4167_416767

/-- The rate at which the inlet pipe fills water, given tank capacity and emptying times. -/
theorem inlet_fill_rate (tank_capacity : ℝ) (leak_empty_time : ℝ) (combined_empty_time : ℝ) :
  tank_capacity = 5760 →
  leak_empty_time = 6 →
  combined_empty_time = 8 →
  (tank_capacity / leak_empty_time - tank_capacity / combined_empty_time) / 60 = 12 :=
by sorry

end NUMINAMATH_CALUDE_inlet_fill_rate_l4167_416767


namespace NUMINAMATH_CALUDE_lcm_of_prime_and_nondivisor_l4167_416764

theorem lcm_of_prime_and_nondivisor (p n : ℕ) (hp : Nat.Prime p) (hn : ¬(n ∣ p)) :
  Nat.lcm p n = p * n :=
by sorry

end NUMINAMATH_CALUDE_lcm_of_prime_and_nondivisor_l4167_416764


namespace NUMINAMATH_CALUDE_installment_plan_properties_l4167_416766

/-- Represents the installment plan for a household appliance purchase -/
structure InstallmentPlan where
  initialPrice : ℝ
  initialPayment : ℝ
  monthlyBasePayment : ℝ
  monthlyInterestRate : ℝ

/-- Calculates the payment for a given month in the installment plan -/
def monthlyPayment (plan : InstallmentPlan) (month : ℕ) : ℝ :=
  plan.monthlyBasePayment + (plan.initialPrice - plan.initialPayment - plan.monthlyBasePayment * (month - 1)) * plan.monthlyInterestRate

/-- Calculates the total amount paid over the course of the installment plan -/
def totalPayment (plan : InstallmentPlan) (totalMonths : ℕ) : ℝ :=
  plan.initialPayment + (Finset.range totalMonths).sum (fun i => monthlyPayment plan (i + 1))

/-- Theorem stating the properties of the specific installment plan -/
theorem installment_plan_properties :
  let plan : InstallmentPlan := {
    initialPrice := 1150,
    initialPayment := 150,
    monthlyBasePayment := 50,
    monthlyInterestRate := 0.01
  }
  (monthlyPayment plan 10 = 55.5) ∧
  (totalPayment plan 20 = 1255) := by
  sorry


end NUMINAMATH_CALUDE_installment_plan_properties_l4167_416766


namespace NUMINAMATH_CALUDE_arithmetic_and_geometric_sequence_l4167_416739

theorem arithmetic_and_geometric_sequence (a b c : ℝ) :
  (b - a = c - b) → -- arithmetic sequence condition
  (b / a = c / b) → -- geometric sequence condition
  (a ≠ 0) →         -- non-zero condition for geometric sequence
  (a = b ∧ b = c ∧ a ≠ 0) := by
sorry

end NUMINAMATH_CALUDE_arithmetic_and_geometric_sequence_l4167_416739


namespace NUMINAMATH_CALUDE_conference_hall_tables_l4167_416785

/-- Given a conference hall with tables and chairs, prove the number of tables. -/
theorem conference_hall_tables (total_legs : ℕ) (chairs_per_table : ℕ) (chair_legs : ℕ) (table_legs : ℕ)
  (h1 : chairs_per_table = 8)
  (h2 : chair_legs = 4)
  (h3 : table_legs = 4)
  (h4 : total_legs = 648) :
  ∃ (num_tables : ℕ), num_tables = 18 ∧ 
    total_legs = num_tables * table_legs + num_tables * chairs_per_table * chair_legs :=
by sorry

end NUMINAMATH_CALUDE_conference_hall_tables_l4167_416785


namespace NUMINAMATH_CALUDE_representative_selection_counts_l4167_416709

def num_boys : Nat := 5
def num_girls : Nat := 3
def num_representatives : Nat := 5
def num_subjects : Nat := 5

theorem representative_selection_counts :
  let scenario1 := (num_girls.choose 1) * (num_boys.choose 4) * (num_representatives.factorial) +
                   (num_girls.choose 2) * (num_boys.choose 3) * (num_representatives.factorial)
  let scenario2 := ((num_boys + num_girls - 1).choose (num_representatives - 1)) * ((num_representatives - 1).factorial)
  let scenario3 := ((num_boys + num_girls - 1).choose (num_representatives - 1)) * ((num_representatives - 1).factorial) * (num_subjects - 1)
  let scenario4 := ((num_boys + num_girls - 2).choose (num_representatives - 2)) * ((num_representatives - 2).factorial) * (num_subjects - 1)
  (∃ (count1 count2 count3 count4 : Nat),
    count1 = scenario1 ∧
    count2 = scenario2 ∧
    count3 = scenario3 ∧
    count4 = scenario4) := by sorry

end NUMINAMATH_CALUDE_representative_selection_counts_l4167_416709


namespace NUMINAMATH_CALUDE_max_b_value_l4167_416710

theorem max_b_value (b : ℕ+) (x : ℤ) (h : x^2 + b*x = -21) : b ≤ 22 := by
  sorry

end NUMINAMATH_CALUDE_max_b_value_l4167_416710


namespace NUMINAMATH_CALUDE_sum_of_solutions_is_four_l4167_416792

theorem sum_of_solutions_is_four : ∃ (S : Finset Int), 
  (∀ x : Int, x ∈ S ↔ x^2 = 192 + x) ∧ (S.sum id = 4) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_solutions_is_four_l4167_416792


namespace NUMINAMATH_CALUDE_rectangle_area_l4167_416750

-- Define the shapes in the rectangle
structure Rectangle :=
  (squares : Fin 2 → ℝ)  -- Areas of the two squares
  (triangle : ℝ)         -- Area of the triangle

-- Define the properties of the rectangle
def valid_rectangle (r : Rectangle) : Prop :=
  r.squares 0 = 4 ∧                     -- Area of smaller square is 4
  r.squares 1 = r.squares 0 ∧           -- Both squares have the same area
  r.triangle = r.squares 0 / 2          -- Area of triangle is half of square area

-- Theorem: The area of the rectangle is 10 square inches
theorem rectangle_area (r : Rectangle) (h : valid_rectangle r) : 
  r.squares 0 + r.squares 1 + r.triangle = 10 := by
  sorry

#check rectangle_area

end NUMINAMATH_CALUDE_rectangle_area_l4167_416750


namespace NUMINAMATH_CALUDE_remainder_987654_div_8_l4167_416737

theorem remainder_987654_div_8 : 987654 % 8 = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_987654_div_8_l4167_416737


namespace NUMINAMATH_CALUDE_family_eating_habits_l4167_416768

theorem family_eating_habits (only_veg only_nonveg total_veg : ℕ) 
  (h1 : only_veg = 16)
  (h2 : only_nonveg = 9)
  (h3 : total_veg = 28) :
  total_veg - only_veg = 12 := by
  sorry

end NUMINAMATH_CALUDE_family_eating_habits_l4167_416768


namespace NUMINAMATH_CALUDE_egg_processing_plant_l4167_416748

theorem egg_processing_plant (E : ℕ) : 
  (96 : ℚ) / 100 * E + (4 : ℚ) / 100 * E = E → -- Original ratio
  ((96 : ℚ) / 100 * E + 12) / E = (99 : ℚ) / 100 → -- New ratio with 12 additional accepted eggs
  E = 400 := by
sorry

end NUMINAMATH_CALUDE_egg_processing_plant_l4167_416748


namespace NUMINAMATH_CALUDE_range_of_values_l4167_416771

theorem range_of_values (x y : ℝ) 
  (hx : 30 < x ∧ x < 42) 
  (hy : 16 < y ∧ y < 24) : 
  (46 < x + y ∧ x + y < 66) ∧ 
  (-18 < x - 2*y ∧ x - 2*y < 10) ∧ 
  (5/4 < x/y ∧ x/y < 21/8) := by
sorry

end NUMINAMATH_CALUDE_range_of_values_l4167_416771


namespace NUMINAMATH_CALUDE_not_square_expression_l4167_416704

theorem not_square_expression (n : ℕ) (a : ℕ) (h1 : n > 2) (h2 : Odd a) (h3 : a > 0) : 
  let b := 2^(2^n)
  a ≤ b ∧ b ≤ 2*a → ¬ ∃ (k : ℕ), a^2 + b^2 - a*b = k^2 := by
  sorry

end NUMINAMATH_CALUDE_not_square_expression_l4167_416704


namespace NUMINAMATH_CALUDE_largest_consecutive_composite_l4167_416758

theorem largest_consecutive_composite : ∃ (n : ℕ), 
  (n < 50) ∧ 
  (n ≥ 10) ∧ 
  (∀ i ∈ Finset.range 10, ¬(Nat.Prime (n - i))) ∧
  (∀ m : ℕ, m > n → ¬(∀ i ∈ Finset.range 10, ¬(Nat.Prime (m - i)))) :=
by sorry

end NUMINAMATH_CALUDE_largest_consecutive_composite_l4167_416758


namespace NUMINAMATH_CALUDE_special_gp_common_ratio_l4167_416793

/-- A geometric progression where each term, starting from the third, 
    is equal to the sum of the two preceding terms. -/
def SpecialGeometricProgression (u : ℕ → ℝ) : Prop :=
  ∃ (q : ℝ), ∀ (n : ℕ),
    u (n + 1) = u n * q ∧ 
    u (n + 2) = u (n + 1) + u n

/-- The common ratio of a special geometric progression 
    is either (1 + √5) / 2 or (1 - √5) / 2. -/
theorem special_gp_common_ratio 
  (u : ℕ → ℝ) (h : SpecialGeometricProgression u) : 
  ∃ (q : ℝ), (∀ (n : ℕ), u (n + 1) = u n * q) ∧ 
    (q = (1 + Real.sqrt 5) / 2 ∨ q = (1 - Real.sqrt 5) / 2) := by
  sorry

end NUMINAMATH_CALUDE_special_gp_common_ratio_l4167_416793


namespace NUMINAMATH_CALUDE_M_equals_N_l4167_416775

-- Define the sets M and N
def M : Set ℝ := {x | x^2 = 1}
def N : Set ℝ := {a | ∃ x ∈ M, a * x = 1}

-- Theorem statement
theorem M_equals_N : M = N := by
  sorry

end NUMINAMATH_CALUDE_M_equals_N_l4167_416775


namespace NUMINAMATH_CALUDE_equivalent_discount_l4167_416747

theorem equivalent_discount (original_price : ℝ) (first_discount second_discount : ℝ) :
  original_price = 50 →
  first_discount = 0.3 →
  second_discount = 0.4 →
  let discounted_price := original_price * (1 - first_discount)
  let final_price := discounted_price * (1 - second_discount)
  let equivalent_discount := (original_price - final_price) / original_price
  equivalent_discount = 0.58 := by
sorry

end NUMINAMATH_CALUDE_equivalent_discount_l4167_416747


namespace NUMINAMATH_CALUDE_sufficient_condition_for_quadratic_inequality_l4167_416781

theorem sufficient_condition_for_quadratic_inequality (a : ℝ) :
  (∀ x : ℝ, x > a → x^2 > 2*x) ∧
  (∃ x : ℝ, x^2 > 2*x ∧ x ≤ a) →
  a ≥ 2 :=
sorry

end NUMINAMATH_CALUDE_sufficient_condition_for_quadratic_inequality_l4167_416781


namespace NUMINAMATH_CALUDE_paperboy_delivery_sequences_l4167_416742

/-- Recurrence relation for the number of valid delivery sequences -/
def D : ℕ → ℕ
  | 0 => 1
  | 1 => 2
  | 2 => 4
  | 3 => 7
  | n + 4 => D (n + 3) + D (n + 2) + D (n + 1)

/-- Number of valid delivery sequences ending with a delivery -/
def E (n : ℕ) : ℕ := D (n - 2)

/-- The number of houses on King's Avenue -/
def num_houses : ℕ := 15

theorem paperboy_delivery_sequences :
  E num_houses = 3136 := by sorry

end NUMINAMATH_CALUDE_paperboy_delivery_sequences_l4167_416742


namespace NUMINAMATH_CALUDE_jose_profit_share_l4167_416774

/-- Calculates the share of profit for an investor in a business partnership --/
def calculate_profit_share (total_profit investment_amount investment_duration total_investment_duration : ℕ) : ℕ :=
  (investment_amount * investment_duration * total_profit) / (total_investment_duration)

theorem jose_profit_share :
  let tom_investment := 30000
  let jose_investment := 45000
  let total_profit := 27000
  let tom_duration := 12
  let jose_duration := 10
  let total_investment_duration := tom_investment * tom_duration + jose_investment * jose_duration
  
  calculate_profit_share total_profit jose_investment jose_duration total_investment_duration = 15000 := by
  sorry

end NUMINAMATH_CALUDE_jose_profit_share_l4167_416774


namespace NUMINAMATH_CALUDE_square_difference_given_sum_and_weighted_sum_l4167_416714

theorem square_difference_given_sum_and_weighted_sum (x y : ℝ) 
  (h1 : x + y = 15) (h2 : 3 * x + y = 20) : x^2 - y^2 = -150 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_given_sum_and_weighted_sum_l4167_416714


namespace NUMINAMATH_CALUDE_polynomial_value_theorem_l4167_416799

theorem polynomial_value_theorem (P : ℤ → ℤ) : 
  (∃ a b c d e : ℤ, a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ 
                     b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ 
                     c ≠ d ∧ c ≠ e ∧ 
                     d ≠ e ∧
                     P a = 5 ∧ P b = 5 ∧ P c = 5 ∧ P d = 5 ∧ P e = 5) →
  (∀ x : ℤ, P x ≠ 9) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_value_theorem_l4167_416799


namespace NUMINAMATH_CALUDE_log_four_one_sixtyfourth_l4167_416730

-- Define the logarithm function
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- State the theorem
theorem log_four_one_sixtyfourth : log 4 (1/64) = -3 := by
  sorry

end NUMINAMATH_CALUDE_log_four_one_sixtyfourth_l4167_416730


namespace NUMINAMATH_CALUDE_michelle_crayon_boxes_l4167_416760

theorem michelle_crayon_boxes (total_crayons : ℕ) (crayons_per_box : ℕ) (num_boxes : ℕ) :
  total_crayons = 35 →
  crayons_per_box = 5 →
  num_boxes * crayons_per_box = total_crayons →
  num_boxes = 7 := by
  sorry

end NUMINAMATH_CALUDE_michelle_crayon_boxes_l4167_416760


namespace NUMINAMATH_CALUDE_clinton_belts_l4167_416789

/-- Represents the number of items Clinton has in his wardrobe -/
structure Wardrobe where
  shoes : ℕ
  belts : ℕ
  hats : ℕ

/-- Clinton's wardrobe satisfies the given conditions -/
def clinton_wardrobe (w : Wardrobe) : Prop :=
  w.shoes = 2 * w.belts ∧
  ∃ n : ℕ, w.belts = w.hats + n ∧
  w.hats = 5 ∧
  w.shoes = 14

theorem clinton_belts :
  ∀ w : Wardrobe, clinton_wardrobe w → w.belts = 7 := by
  sorry

end NUMINAMATH_CALUDE_clinton_belts_l4167_416789


namespace NUMINAMATH_CALUDE_roots_of_quadratic_equation_l4167_416751

theorem roots_of_quadratic_equation (θ : Real) (x₁ x₂ : ℂ) :
  θ ∈ Set.Icc 0 π ∧
  x₁^2 - 3 * Real.sin θ * x₁ + Real.sin θ^2 + 1 = 0 ∧
  x₂^2 - 3 * Real.sin θ * x₂ + Real.sin θ^2 + 1 = 0 ∧
  Complex.abs x₁ + Complex.abs x₂ = 2 →
  θ = 0 := by sorry

end NUMINAMATH_CALUDE_roots_of_quadratic_equation_l4167_416751


namespace NUMINAMATH_CALUDE_samson_age_relation_l4167_416752

/-- Samson's current age in years -/
def samsonAge : ℝ := 6.25

/-- Samson's mother's current age in years -/
def motherAge : ℝ := 30.65

/-- The age Samson will be when his mother is exactly 4 times his age -/
def targetAge : ℝ := 8.1333

theorem samson_age_relation :
  ∃ (T : ℝ), 
    (samsonAge + T = targetAge) ∧ 
    (motherAge + T = 4 * (samsonAge + T)) := by
  sorry

end NUMINAMATH_CALUDE_samson_age_relation_l4167_416752


namespace NUMINAMATH_CALUDE_sum_product_bound_l4167_416700

theorem sum_product_bound (a b c d : ℝ) (h : a + b + c + d = 1) :
  ∃ (x : ℝ), x ≤ 0.5 ∧ (ab + ac + ad + bc + bd + cd ≤ x) ∧
  ∀ (y : ℝ), ∃ (a' b' c' d' : ℝ), a' + b' + c' + d' = 1 ∧
  a'*b' + a'*c' + a'*d' + b'*c' + b'*d' + c'*d' < y :=
sorry

end NUMINAMATH_CALUDE_sum_product_bound_l4167_416700


namespace NUMINAMATH_CALUDE_investment_duration_theorem_l4167_416738

def initial_investment : ℝ := 2000
def interest_rate_1 : ℝ := 0.08
def interest_rate_2 : ℝ := 0.12
def final_value : ℝ := 6620
def years_at_rate_1 : ℕ := 2

def investment_equation (t : ℝ) : Prop :=
  initial_investment * (1 + interest_rate_1) ^ years_at_rate_1 * (1 + interest_rate_2) ^ (t - years_at_rate_1) = final_value

theorem investment_duration_theorem :
  ∃ t : ℕ, (∀ s : ℝ, investment_equation s → t ≥ ⌈s⌉) ∧ investment_equation (t : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_investment_duration_theorem_l4167_416738


namespace NUMINAMATH_CALUDE_consecutive_integers_square_sum_l4167_416728

theorem consecutive_integers_square_sum (n : ℕ) : 
  (n > 0) → (n^2 + (n+1)^2 = n*(n+1) + 91) → (n+1 = 10) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_square_sum_l4167_416728


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_is_two_l4167_416735

/-- A hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola (a b : ℝ) where
  a_pos : a > 0
  b_pos : b > 0

/-- The eccentricity of a hyperbola -/
def eccentricity (h : Hyperbola a b) : ℝ := sorry

/-- The right focus of a hyperbola -/
def right_focus (h : Hyperbola a b) : ℝ × ℝ := sorry

/-- An asymptote of a hyperbola -/
def asymptote (h : Hyperbola a b) : ℝ → ℝ := sorry

/-- The symmetric point of a point with respect to a line -/
def symmetric_point (p : ℝ × ℝ) (l : ℝ → ℝ) : ℝ × ℝ := sorry

/-- Theorem: The eccentricity of a hyperbola is 2 given the specified conditions -/
theorem hyperbola_eccentricity_is_two (a b : ℝ) (h : Hyperbola a b) :
  let l₁ := asymptote h
  let l₂ := fun x => -l₁ x
  let f := right_focus h
  let s := symmetric_point f l₁
  s.2 = l₂ s.1 →
  eccentricity h = 2 := by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_is_two_l4167_416735


namespace NUMINAMATH_CALUDE_square_sum_geq_negative_double_product_l4167_416786

theorem square_sum_geq_negative_double_product (a b : ℝ) : a^2 + b^2 ≥ -2*a*b := by
  sorry

end NUMINAMATH_CALUDE_square_sum_geq_negative_double_product_l4167_416786


namespace NUMINAMATH_CALUDE_ellipse_equation_l4167_416770

theorem ellipse_equation (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  let C : Set (ℝ × ℝ) := {(x, y) | x^2 / a^2 + y^2 / b^2 = 1}
  let e : ℝ := 1/3
  let A₁ : ℝ × ℝ := (-a, 0)
  let A₂ : ℝ × ℝ := (a, 0)
  let B : ℝ × ℝ := (0, b)
  (∀ (x y : ℝ), (x, y) ∈ C ↔ x^2 / a^2 + y^2 / b^2 = 1) →
  e = Real.sqrt (1 - b^2 / a^2) →
  ((B.1 - A₁.1) * (B.1 - A₂.1) + (B.2 - A₁.2) * (B.2 - A₂.2) = -1) →
  (∀ (x y : ℝ), (x, y) ∈ C ↔ x^2 / 9 + y^2 / 8 = 1) := by
sorry

end NUMINAMATH_CALUDE_ellipse_equation_l4167_416770


namespace NUMINAMATH_CALUDE_sum_of_x_and_y_l4167_416776

theorem sum_of_x_and_y (x y : ℝ) 
  (h1 : x^2 * y^3 + y^2 * x^3 = 27) 
  (h2 : x * y = 3) : 
  x + y = 3 := by
sorry

end NUMINAMATH_CALUDE_sum_of_x_and_y_l4167_416776


namespace NUMINAMATH_CALUDE_expression_evaluation_l4167_416780

theorem expression_evaluation : 
  let a := 12
  let b := 14
  let c := 18
  let numerator := a^2 * (1/b - 1/c) + b^2 * (1/c - 1/a) + c^2 * (1/a - 1/b)
  let denominator := a * (1/b - 1/c) + b * (1/c - 1/a) + c * (1/a - 1/b)
  numerator / denominator = 44 := by
sorry


end NUMINAMATH_CALUDE_expression_evaluation_l4167_416780


namespace NUMINAMATH_CALUDE_frank_jim_speed_difference_l4167_416705

theorem frank_jim_speed_difference : 
  ∀ (jim_distance frank_distance : ℝ) (time : ℝ),
    jim_distance = 16 →
    frank_distance = 20 →
    time = 2 →
    (frank_distance / time) - (jim_distance / time) = 2 := by
  sorry

end NUMINAMATH_CALUDE_frank_jim_speed_difference_l4167_416705


namespace NUMINAMATH_CALUDE_incorrect_relation_l4167_416762

theorem incorrect_relation (a b : ℝ) (h : a > b) : ∃ c : ℝ, ¬(a * c^2 > b * c^2) := by
  sorry

end NUMINAMATH_CALUDE_incorrect_relation_l4167_416762


namespace NUMINAMATH_CALUDE_lines_intersect_l4167_416798

/-- Two lines in 2D space -/
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Define when two lines are intersecting -/
def are_intersecting (l1 l2 : Line2D) : Prop :=
  l1.a * l2.b ≠ l1.b * l2.a

/-- The problem statement -/
theorem lines_intersect : 
  let line1 : Line2D := ⟨3, -2, 5⟩
  let line2 : Line2D := ⟨1, 3, 10⟩
  are_intersecting line1 line2 := by
  sorry

end NUMINAMATH_CALUDE_lines_intersect_l4167_416798


namespace NUMINAMATH_CALUDE_largest_n_binomial_equality_l4167_416717

theorem largest_n_binomial_equality : 
  ∃ (n : ℕ), (Nat.choose 8 3 + Nat.choose 8 4 = Nat.choose 9 n) ∧ 
  (∀ (m : ℕ), m > n → Nat.choose 8 3 + Nat.choose 8 4 ≠ Nat.choose 9 m) ∧ 
  n = 5 := by
  sorry

end NUMINAMATH_CALUDE_largest_n_binomial_equality_l4167_416717


namespace NUMINAMATH_CALUDE_min_c_value_l4167_416719

/-- Given natural numbers a, b, c where a < b < c, and a system of equations with exactly one solution,
    prove that the minimum possible value of c is 1018. -/
theorem min_c_value (a b c : ℕ) (h1 : a < b) (h2 : b < c)
    (h3 : ∃! (x y : ℝ), 2 * x + y = 2035 ∧ y = |x - a| + |x - b| + |x - c|) :
  c ≥ 1018 ∧ ∃ (a' b' : ℕ), a' < b' ∧ b' < 1018 ∧
    ∃! (x y : ℝ), 2 * x + y = 2035 ∧ y = |x - a'| + |x - b'| + |x - 1018| :=
by sorry

end NUMINAMATH_CALUDE_min_c_value_l4167_416719


namespace NUMINAMATH_CALUDE_fraction_simplification_l4167_416784

theorem fraction_simplification : 
  (1/3 + 1/5) / ((2/7) * (3/4) - 1/7) = 112/15 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l4167_416784


namespace NUMINAMATH_CALUDE_fraction_addition_simplification_l4167_416791

theorem fraction_addition_simplification :
  5 / 462 + 23 / 42 = 43 / 77 := by sorry

end NUMINAMATH_CALUDE_fraction_addition_simplification_l4167_416791


namespace NUMINAMATH_CALUDE_sum_factorials_mod_5_l4167_416779

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

def sum_factorials : ℕ := 
  (factorial 1) + (factorial 2) + (factorial 3) + (factorial 4) + 
  (factorial 5) + (factorial 6) + (factorial 7) + (factorial 8) + 
  (factorial 9) + (factorial 10)

theorem sum_factorials_mod_5 : sum_factorials % 5 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_factorials_mod_5_l4167_416779


namespace NUMINAMATH_CALUDE_final_amount_after_three_late_charges_l4167_416716

/-- The final amount owed after applying three consecutive 5% late charges to an initial bill of $200 -/
theorem final_amount_after_three_late_charges : 
  let initial_bill : ℝ := 200
  let late_charge_rate : ℝ := 0.05
  let num_late_charges : ℕ := 3
  let final_amount := initial_bill * (1 + late_charge_rate)^num_late_charges
  final_amount = 231.525 := by sorry

end NUMINAMATH_CALUDE_final_amount_after_three_late_charges_l4167_416716


namespace NUMINAMATH_CALUDE_new_girls_count_l4167_416712

theorem new_girls_count (initial_girls : ℕ) (initial_boys : ℕ) (total_after : ℕ) : 
  initial_girls = 706 →
  initial_boys = 222 →
  total_after = 1346 →
  total_after - (initial_girls + initial_boys) = 418 :=
by
  sorry

end NUMINAMATH_CALUDE_new_girls_count_l4167_416712


namespace NUMINAMATH_CALUDE_quadratic_roots_property_l4167_416753

theorem quadratic_roots_property (m n : ℝ) : 
  (m^2 - 2*m - 2025 = 0) → 
  (n^2 - 2*n - 2025 = 0) → 
  (m^2 - 3*m - n = 2023) := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_property_l4167_416753


namespace NUMINAMATH_CALUDE_pigeonhole_on_floor_division_l4167_416707

theorem pigeonhole_on_floor_division (n : ℕ) (h_n : n > 3) 
  (nums : Finset ℕ) (h_nums_card : nums.card = n) 
  (h_nums_distinct : nums.card = Finset.card (Finset.image id nums))
  (h_nums_bound : ∀ x ∈ nums, x < Nat.factorial (n - 1)) :
  ∃ (a b c d : ℕ), a ∈ nums ∧ b ∈ nums ∧ c ∈ nums ∧ d ∈ nums ∧ 
    a > b ∧ c > d ∧ (a ≠ c ∨ b ≠ d) ∧ 
    (a / b : ℕ) = (c / d : ℕ) := by
  sorry

end NUMINAMATH_CALUDE_pigeonhole_on_floor_division_l4167_416707


namespace NUMINAMATH_CALUDE_nine_payment_methods_l4167_416795

/-- Represents the number of ways to pay an amount using given denominations -/
def paymentMethods (amount : ℕ) (denominations : List ℕ) : ℕ := sorry

/-- The cost of the book in yuan -/
def bookCost : ℕ := 20

/-- Available note denominations in yuan -/
def availableNotes : List ℕ := [10, 5, 1]

/-- Theorem stating that there are 9 ways to pay for the book -/
theorem nine_payment_methods : paymentMethods bookCost availableNotes = 9 := by sorry

end NUMINAMATH_CALUDE_nine_payment_methods_l4167_416795


namespace NUMINAMATH_CALUDE_points_scored_third_game_l4167_416720

-- Define the average points per game after 2 games
def avg_points_2_games : ℝ := 61.5

-- Define the total points needed to exceed after 3 games
def total_points_threshold : ℕ := 500

-- Define the additional points needed after 3 games
def additional_points_needed : ℕ := 330

-- Theorem to prove
theorem points_scored_third_game :
  let total_points_2_games := 2 * avg_points_2_games
  let points_third_game := total_points_threshold - additional_points_needed - total_points_2_games
  points_third_game = 47 := by
  sorry

end NUMINAMATH_CALUDE_points_scored_third_game_l4167_416720


namespace NUMINAMATH_CALUDE_square_of_difference_l4167_416721

theorem square_of_difference (x : ℝ) :
  (7 - (x^3 - 49)^(1/3))^2 = 49 - 14 * (x^3 - 49)^(1/3) + ((x^3 - 49)^(1/3))^2 := by
  sorry

end NUMINAMATH_CALUDE_square_of_difference_l4167_416721


namespace NUMINAMATH_CALUDE_a_fourth_plus_inverse_a_fourth_l4167_416733

theorem a_fourth_plus_inverse_a_fourth (a : ℝ) (h : (a + 1/a)^2 = 5) :
  a^4 + 1/a^4 = 7 := by
sorry

end NUMINAMATH_CALUDE_a_fourth_plus_inverse_a_fourth_l4167_416733


namespace NUMINAMATH_CALUDE_complex_equality_and_minimum_distance_l4167_416723

open Complex

theorem complex_equality_and_minimum_distance (z : ℂ) :
  (abs z = abs (z + 1 + I)) →
  (∃ (a : ℝ), z = a + a * I) →
  (z = -1 - I) ∧
  (∃ (min_dist : ℝ), min_dist = Real.sqrt 2 ∧
    ∀ (w : ℂ), abs w = abs (w + 1 + I) → abs (w - (2 - I)) ≥ min_dist) :=
by sorry

end NUMINAMATH_CALUDE_complex_equality_and_minimum_distance_l4167_416723


namespace NUMINAMATH_CALUDE_chocolate_box_theorem_l4167_416715

/-- Represents a box of chocolates -/
structure ChocolateBox where
  total : ℕ  -- Total number of chocolates originally
  rows : ℕ   -- Number of rows
  cols : ℕ   -- Number of columns

/-- Míša's actions on the chocolate box -/
def misaActions (box : ChocolateBox) : Prop :=
  ∃ (eaten1 eaten2 : ℕ),
    -- After all actions, 1/3 of chocolates remain
    box.total / 3 = box.total - eaten1 - eaten2 - (box.rows - 1) - (box.cols - 1) ∧
    -- After first rearrangement, 3 rows are filled except for one space
    3 * box.cols - 1 = box.total - eaten1 ∧
    -- After second rearrangement, 5 columns are filled except for one space
    5 * box.rows - 1 = box.total - eaten1 - eaten2 - (box.rows - 1)

theorem chocolate_box_theorem (box : ChocolateBox) :
  misaActions box →
  box.total = 60 ∧ box.rows = 5 ∧ box.cols = 12 ∧
  ∃ (eaten1 : ℕ), eaten1 = 25 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_box_theorem_l4167_416715


namespace NUMINAMATH_CALUDE_cake_division_l4167_416727

theorem cake_division (x y z : ℚ) :
  x + y + z = 1 →
  2 * z = x →
  z = (1/2) * (y + (2/3) * x) →
  (2/3) * x = 4/11 :=
by
  sorry

end NUMINAMATH_CALUDE_cake_division_l4167_416727


namespace NUMINAMATH_CALUDE_cuboid_height_l4167_416711

/-- The height of a rectangular cuboid given its surface area, length, and width -/
theorem cuboid_height (surface_area length width height : ℝ) : 
  surface_area = 2 * length * width + 2 * length * height + 2 * width * height →
  surface_area = 442 →
  length = 7 →
  width = 8 →
  height = 11 := by
  sorry


end NUMINAMATH_CALUDE_cuboid_height_l4167_416711


namespace NUMINAMATH_CALUDE_min_value_of_f_l4167_416796

-- Define the function f(x) = x^3 - 3x
def f (x : ℝ) : ℝ := x^3 - 3*x

-- State the theorem that the minimum value of f(x) is -2
theorem min_value_of_f :
  ∃ (m : ℝ), (∀ (x : ℝ), f x ≥ m) ∧ (∃ (x : ℝ), f x = m) ∧ m = -2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_f_l4167_416796


namespace NUMINAMATH_CALUDE_inequality_proof_l4167_416783

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_condition : a * b + b * c + c * a ≤ 1) :
  a + b + c + Real.sqrt 3 ≥ 8 * a * b * c * (1 / (a^2 + 1) + 1 / (b^2 + 1) + 1 / (c^2 + 1)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l4167_416783


namespace NUMINAMATH_CALUDE_second_chapter_longer_l4167_416745

/-- A book with two chapters -/
structure Book where
  chapter1_pages : ℕ
  chapter2_pages : ℕ

/-- The difference in pages between two chapters -/
def page_difference (b : Book) : ℕ := b.chapter2_pages - b.chapter1_pages

theorem second_chapter_longer (b : Book) 
  (h1 : b.chapter1_pages = 37) 
  (h2 : b.chapter2_pages = 80) : 
  page_difference b = 43 := by
  sorry

end NUMINAMATH_CALUDE_second_chapter_longer_l4167_416745


namespace NUMINAMATH_CALUDE_probability_increasing_maxima_correct_l4167_416744

/-- The probability that the maximum numbers in each row of a triangular array
    are in strictly increasing order. -/
def probability_increasing_maxima (n : ℕ) : ℚ :=
  (2 ^ n : ℚ) / (n + 1).factorial

/-- Theorem stating that the probability of increasing maxima in a triangular array
    with n rows is equal to 2^n / (n+1)! -/
theorem probability_increasing_maxima_correct (n : ℕ) :
  let array_size := n * (n + 1) / 2
  probability_increasing_maxima n =
    (2 ^ n : ℚ) / (n + 1).factorial :=
by sorry

end NUMINAMATH_CALUDE_probability_increasing_maxima_correct_l4167_416744


namespace NUMINAMATH_CALUDE_tim_running_schedule_l4167_416765

/-- The number of days Tim used to run per week -/
def previous_running_days (hours_per_day : ℕ) (total_hours_per_week : ℕ) (extra_days : ℕ) : ℕ :=
  total_hours_per_week / hours_per_day - extra_days

theorem tim_running_schedule :
  previous_running_days 2 10 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_tim_running_schedule_l4167_416765


namespace NUMINAMATH_CALUDE_volunteer_selection_theorem_l4167_416701

/-- The number of ways to select three volunteers from five for three specific roles -/
def select_volunteers (n : ℕ) (k : ℕ) (excluded : ℕ) : ℕ :=
  (n - 1) * (n - 1) * (n - 2)

/-- The theorem stating that selecting three volunteers from five for three specific roles,
    where one volunteer cannot serve in a particular role, results in 48 different ways -/
theorem volunteer_selection_theorem :
  select_volunteers 5 3 1 = 48 := by
  sorry

end NUMINAMATH_CALUDE_volunteer_selection_theorem_l4167_416701


namespace NUMINAMATH_CALUDE_dividend_percentage_calculation_l4167_416749

/-- Calculates the dividend percentage given investment details and dividend amount -/
theorem dividend_percentage_calculation
  (investment : ℝ)
  (share_face_value : ℝ)
  (premium_percentage : ℝ)
  (dividend_received : ℝ)
  (h1 : investment = 14400)
  (h2 : share_face_value = 100)
  (h3 : premium_percentage = 20)
  (h4 : dividend_received = 600) :
  let share_cost := share_face_value * (1 + premium_percentage / 100)
  let num_shares := investment / share_cost
  let dividend_per_share := dividend_received / num_shares
  let dividend_percentage := (dividend_per_share / share_face_value) * 100
  dividend_percentage = 5 := by sorry

end NUMINAMATH_CALUDE_dividend_percentage_calculation_l4167_416749


namespace NUMINAMATH_CALUDE_decagon_triangles_l4167_416732

/-- The number of marked points on the decagon -/
def n : ℕ := 20

/-- The number of sides of the decagon -/
def sides : ℕ := 10

/-- The number of points to choose for each triangle -/
def k : ℕ := 3

/-- The total number of ways to choose 3 points out of 20 -/
def total_combinations : ℕ := Nat.choose n k

/-- The number of non-triangle-forming sets (collinear points on each side) -/
def non_triangle_sets : ℕ := sides

/-- The number of valid triangles -/
def valid_triangles : ℕ := total_combinations - non_triangle_sets

theorem decagon_triangles :
  valid_triangles = 1130 :=
sorry

end NUMINAMATH_CALUDE_decagon_triangles_l4167_416732


namespace NUMINAMATH_CALUDE_odd_cube_plus_linear_plus_constant_l4167_416703

theorem odd_cube_plus_linear_plus_constant (o n m : ℤ) 
  (ho : ∃ k : ℤ, o = 2*k + 1) : 
  Odd (o^3 + n*o + m) ↔ Even m := by
  sorry

end NUMINAMATH_CALUDE_odd_cube_plus_linear_plus_constant_l4167_416703


namespace NUMINAMATH_CALUDE_dunk_height_above_rim_l4167_416773

/-- Represents the height of a basketball player in inches -/
def player_height : ℝ := 6 * 12

/-- Represents the additional reach of the player above their head in inches -/
def player_reach : ℝ := 22

/-- Represents the height of the basketball rim in inches -/
def rim_height : ℝ := 10 * 12

/-- Theorem stating the height above the rim a player must reach to dunk -/
theorem dunk_height_above_rim : 
  rim_height - (player_height + player_reach) = 26 := by sorry

end NUMINAMATH_CALUDE_dunk_height_above_rim_l4167_416773


namespace NUMINAMATH_CALUDE_right_triangle_sets_l4167_416736

def is_right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2

theorem right_triangle_sets :
  ¬(is_right_triangle 5 7 10) ∧
  (is_right_triangle 3 4 5) ∧
  (is_right_triangle 6 8 10) ∧
  (is_right_triangle 1 2 (Real.sqrt 3)) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_sets_l4167_416736


namespace NUMINAMATH_CALUDE_students_walking_home_fraction_l4167_416788

theorem students_walking_home_fraction (total : ℚ) 
  (bus_fraction : ℚ) (auto_fraction : ℚ) (bike_fraction : ℚ) (scooter_fraction : ℚ) :
  bus_fraction = 1/3 →
  auto_fraction = 1/5 →
  bike_fraction = 1/8 →
  scooter_fraction = 1/15 →
  total = 1 →
  total - (bus_fraction + auto_fraction + bike_fraction + scooter_fraction) = 33/120 := by
  sorry

end NUMINAMATH_CALUDE_students_walking_home_fraction_l4167_416788


namespace NUMINAMATH_CALUDE_present_age_of_B_l4167_416772

/-- 
Given two people A and B, where:
1. In 20 years, A will be twice as old as B was 20 years ago.
2. A is now 10 years older than B.
Prove that the present age of B is 70 years.
-/
theorem present_age_of_B (A B : ℕ) 
  (h1 : A + 20 = 2 * (B - 20))
  (h2 : A = B + 10) : 
  B = 70 := by
  sorry


end NUMINAMATH_CALUDE_present_age_of_B_l4167_416772


namespace NUMINAMATH_CALUDE_propositions_truth_l4167_416713

theorem propositions_truth : 
  (∀ x : ℝ, x^2 - x + 1 ≥ 0) ∧ 
  (∃ x : ℝ, x^2 - 4 = 0) := by
  sorry

end NUMINAMATH_CALUDE_propositions_truth_l4167_416713


namespace NUMINAMATH_CALUDE_predict_grain_demand_2012_l4167_416777

/-- Regression equation for grain demand -/
def grain_demand (x : ℝ) : ℝ := 6.5 * (x - 2006) + 261

/-- Theorem: The predicted grain demand for 2012 is 300 ten thousand tons -/
theorem predict_grain_demand_2012 : grain_demand 2012 = 300 := by
  sorry

end NUMINAMATH_CALUDE_predict_grain_demand_2012_l4167_416777


namespace NUMINAMATH_CALUDE_sequence_growth_l4167_416759

theorem sequence_growth (a : ℕ → ℕ) 
  (h1 : ∀ n, a n > 1) 
  (h2 : ∀ m n, m ≠ n → a m ≠ a n) : 
  ∀ N : ℕ, ∃ n : ℕ, n > N ∧ a n > n :=
sorry

end NUMINAMATH_CALUDE_sequence_growth_l4167_416759


namespace NUMINAMATH_CALUDE_rectangles_count_l4167_416761

/-- The number of rectangles formed by p parallel lines and q perpendicular lines -/
def num_rectangles (p q : ℕ) : ℚ :=
  (p * (p - 1) * q * (q - 1)) / 4

/-- Theorem stating that num_rectangles gives the correct number of rectangles -/
theorem rectangles_count (p q : ℕ) :
  num_rectangles p q = (p * (p - 1) * q * (q - 1)) / 4 := by
  sorry

#check rectangles_count

end NUMINAMATH_CALUDE_rectangles_count_l4167_416761


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_length_l4167_416794

/-- An isosceles triangle with two sides of length 8 cm and perimeter 30 cm has a base of length 14 cm. -/
theorem isosceles_triangle_base_length :
  ∀ (base_length : ℝ),
    base_length > 0 →
    2 * 8 + base_length = 30 →
    base_length = 14 :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_length_l4167_416794


namespace NUMINAMATH_CALUDE_angle_inequality_l4167_416734

theorem angle_inequality (α β : Real) (h1 : 0 < α ∧ α < π / 2) (h2 : 0 < β ∧ β < π / 2)
  (h3 : α / (2 * (1 + Real.cos (α / 2))) < Real.tan β ∧ Real.tan β < (1 - Real.cos α) / α) :
  α / 4 < β ∧ β < α / 2 := by sorry

end NUMINAMATH_CALUDE_angle_inequality_l4167_416734


namespace NUMINAMATH_CALUDE_rain_probability_l4167_416718

/-- The probability of rain on Friday -/
def prob_friday : ℝ := 0.7

/-- The probability of rain on Saturday -/
def prob_saturday : ℝ := 0.5

/-- The probability of rain on Sunday -/
def prob_sunday : ℝ := 0.3

/-- The events are independent -/
axiom independence : True

/-- The probability of rain on all three days -/
def prob_all_days : ℝ := prob_friday * prob_saturday * prob_sunday

/-- Theorem: The probability of rain on all three days is 10.5% -/
theorem rain_probability : prob_all_days = 0.105 := by
  sorry

end NUMINAMATH_CALUDE_rain_probability_l4167_416718
