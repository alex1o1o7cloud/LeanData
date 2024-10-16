import Mathlib

namespace NUMINAMATH_CALUDE_unique_integer_square_less_than_double_l3079_307983

theorem unique_integer_square_less_than_double :
  ∃! x : ℤ, x^2 < 2*x :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_integer_square_less_than_double_l3079_307983


namespace NUMINAMATH_CALUDE_min_beta_delta_sum_l3079_307936

open Complex

/-- A complex function g satisfying certain conditions -/
def g (β δ : ℂ) : ℂ → ℂ := λ z ↦ (3 + 2*I)*z^3 + β*z + δ

/-- The theorem stating the minimum value of |β| + |δ| -/
theorem min_beta_delta_sum (β δ : ℂ) :
  (g β δ 1).im = 0 →
  (g β δ (-I)).im = -Real.pi →
  ∃ (min : ℝ), min = Real.sqrt (Real.pi^2 + 2*Real.pi + 2) + 2 ∧
    ∀ (β' δ' : ℂ), (g β' δ' 1).im = 0 → (g β' δ' (-I)).im = -Real.pi →
      Complex.abs β' + Complex.abs δ' ≥ min :=
sorry


end NUMINAMATH_CALUDE_min_beta_delta_sum_l3079_307936


namespace NUMINAMATH_CALUDE_max_gcd_of_sequence_l3079_307988

theorem max_gcd_of_sequence (n : ℕ+) :
  let a : ℕ+ → ℕ := fun k => 120 + k^2
  let d : ℕ+ → ℕ := fun k => Nat.gcd (a k) (a (k + 1))
  ∃ k : ℕ+, d k = 121 ∧ ∀ m : ℕ+, d m ≤ 121 :=
by sorry

end NUMINAMATH_CALUDE_max_gcd_of_sequence_l3079_307988


namespace NUMINAMATH_CALUDE_solve_allowance_problem_l3079_307913

def allowance_problem (initial_amount spent_amount final_amount : ℕ) : Prop :=
  ∃ allowance : ℕ, 
    initial_amount - spent_amount + allowance = final_amount

theorem solve_allowance_problem :
  allowance_problem 5 2 8 → ∃ allowance : ℕ, allowance = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_solve_allowance_problem_l3079_307913


namespace NUMINAMATH_CALUDE_replacement_paint_intensity_l3079_307912

theorem replacement_paint_intensity
  (original_intensity : ℝ)
  (new_mixture_intensity : ℝ)
  (fraction_replaced : ℝ)
  (replacement_intensity : ℝ)
  (h1 : original_intensity = 50)
  (h2 : new_mixture_intensity = 40)
  (h3 : fraction_replaced = 1 / 3)
  (h4 : (1 - fraction_replaced) * original_intensity + fraction_replaced * replacement_intensity = new_mixture_intensity) :
  replacement_intensity = 20 := by
sorry

end NUMINAMATH_CALUDE_replacement_paint_intensity_l3079_307912


namespace NUMINAMATH_CALUDE_sector_central_angle_l3079_307973

theorem sector_central_angle (area : Real) (radius : Real) (centralAngle : Real) :
  area = 3 * Real.pi / 8 →
  radius = 1 →
  centralAngle = area * 2 / (radius ^ 2) →
  centralAngle = 3 * Real.pi / 4 := by
  sorry

end NUMINAMATH_CALUDE_sector_central_angle_l3079_307973


namespace NUMINAMATH_CALUDE_factor_polynomial_l3079_307955

theorem factor_polynomial (x : ℝ) : 54 * x^5 - 135 * x^9 = 27 * x^5 * (2 - 5 * x^4) := by
  sorry

end NUMINAMATH_CALUDE_factor_polynomial_l3079_307955


namespace NUMINAMATH_CALUDE_alpha_value_l3079_307947

theorem alpha_value (α β : Real) 
  (h1 : 0 < α ∧ α < π/2) 
  (h2 : 0 < β ∧ β < π/2)
  (h3 : Real.tan (α - β) = 1/2)
  (h4 : Real.tan β = 1/3) : 
  α = π/4 := by
  sorry

end NUMINAMATH_CALUDE_alpha_value_l3079_307947


namespace NUMINAMATH_CALUDE_smallest_k_for_largest_three_digit_prime_l3079_307937

theorem smallest_k_for_largest_three_digit_prime (p k : ℕ) : 
  p = 997 →  -- p is the largest 3-digit prime
  k > 0 →    -- k is positive
  (∀ m : ℕ, m > 0 ∧ m < k → ¬(10 ∣ (p^2 - m))) →  -- k is the smallest such positive integer
  (10 ∣ (p^2 - k)) →  -- p^2 - k is divisible by 10
  k = 9 :=  -- k equals 9
by sorry

end NUMINAMATH_CALUDE_smallest_k_for_largest_three_digit_prime_l3079_307937


namespace NUMINAMATH_CALUDE_speedster_convertibles_l3079_307959

/-- The number of Speedster convertibles given the total number of vehicles,
    non-Speedsters, and the fraction of Speedsters that are convertibles. -/
theorem speedster_convertibles
  (total_vehicles : ℕ)
  (non_speedsters : ℕ)
  (speedster_convertible_fraction : ℚ)
  (h1 : total_vehicles = 80)
  (h2 : non_speedsters = 50)
  (h3 : speedster_convertible_fraction = 4/5) :
  (total_vehicles - non_speedsters) * speedster_convertible_fraction = 24 := by
  sorry

#eval (80 - 50) * (4/5 : ℚ)

end NUMINAMATH_CALUDE_speedster_convertibles_l3079_307959


namespace NUMINAMATH_CALUDE_last_score_is_70_l3079_307960

def scores : List ℕ := [65, 70, 85, 90]

def is_valid_sequence (seq : List ℕ) : Prop :=
  ∀ i : Fin seq.length, (seq.take (i + 1)).sum % (i + 1) = 0

theorem last_score_is_70 :
  ∃ (perm : List ℕ), perm.reverse = scores ∧ 
  is_valid_sequence perm ∧ 
  perm.head! = 70 :=
sorry

end NUMINAMATH_CALUDE_last_score_is_70_l3079_307960


namespace NUMINAMATH_CALUDE_f_101_form_l3079_307944

def is_valid_f (f : ℕ → ℕ) : Prop :=
  ∀ m n : ℕ, m > 0 ∧ n > 0 → (m * n + 1) ∣ (f m * f n + 1)

theorem f_101_form (f : ℕ → ℕ) (h : is_valid_f f) :
  ∃ k : ℕ, k % 2 = 1 ∧ f 101 = 101^k :=
by sorry

end NUMINAMATH_CALUDE_f_101_form_l3079_307944


namespace NUMINAMATH_CALUDE_clock_second_sale_price_l3079_307962

/-- Represents the clock sale scenario in the shop -/
structure ClockSale where
  originalCost : ℝ
  firstSalePrice : ℝ
  buyBackPrice : ℝ
  secondSalePrice : ℝ

/-- The conditions of the clock sale problem -/
def clockSaleProblem (sale : ClockSale) : Prop :=
  sale.firstSalePrice = 1.2 * sale.originalCost ∧
  sale.buyBackPrice = 0.5 * sale.firstSalePrice ∧
  sale.originalCost - sale.buyBackPrice = 100 ∧
  sale.secondSalePrice = sale.buyBackPrice * 1.8

/-- The theorem stating that under the given conditions, 
    the second sale price is 270 -/
theorem clock_second_sale_price (sale : ClockSale) :
  clockSaleProblem sale → sale.secondSalePrice = 270 := by
  sorry

end NUMINAMATH_CALUDE_clock_second_sale_price_l3079_307962


namespace NUMINAMATH_CALUDE_circle_area_difference_l3079_307957

theorem circle_area_difference (r₁ r₂ r : ℝ) (h₁ : r₁ = 15) (h₂ : r₂ = 25) :
  π * r₂^2 - π * r₁^2 = π * r^2 → r = 20 :=
by sorry

end NUMINAMATH_CALUDE_circle_area_difference_l3079_307957


namespace NUMINAMATH_CALUDE_scientific_notation_2023_l3079_307945

/-- Scientific notation representation with a specified number of significant figures -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  sigFigs : ℕ

/-- Convert a real number to scientific notation with a specified number of significant figures -/
def toScientificNotation (x : ℝ) (sigFigs : ℕ) : ScientificNotation :=
  sorry

/-- Check if a ScientificNotation representation is valid -/
def isValidScientificNotation (sn : ScientificNotation) : Prop :=
  1 ≤ sn.coefficient ∧ sn.coefficient < 10 ∧ sn.sigFigs > 0

theorem scientific_notation_2023 :
  let sn := toScientificNotation 2023 2
  isValidScientificNotation sn ∧ sn.coefficient = 2.0 ∧ sn.exponent = 3 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_2023_l3079_307945


namespace NUMINAMATH_CALUDE_opposite_of_negative_two_l3079_307911

theorem opposite_of_negative_two : 
  ∃ x : ℤ, -x = -2 ∧ x = 2 := by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_two_l3079_307911


namespace NUMINAMATH_CALUDE_team_a_games_played_l3079_307975

theorem team_a_games_played (team_a_win_ratio : ℚ) (team_b_win_ratio : ℚ) 
  (team_b_extra_wins : ℕ) (team_b_extra_losses : ℕ) :
  team_a_win_ratio = 3/4 →
  team_b_win_ratio = 2/3 →
  team_b_extra_wins = 5 →
  team_b_extra_losses = 3 →
  ∃ (a : ℕ), 
    a = 4 ∧
    team_b_win_ratio * (a + team_b_extra_wins + team_b_extra_losses) = 
      team_a_win_ratio * a + team_b_extra_wins :=
by sorry

end NUMINAMATH_CALUDE_team_a_games_played_l3079_307975


namespace NUMINAMATH_CALUDE_range_of_a_l3079_307918

theorem range_of_a (a : ℝ) :
  (∃ x₀ : ℝ, -1 < x₀ ∧ x₀ < 1 ∧ 2 * a * x₀ - a + 3 = 0) →
  (a < -3 ∨ a > 1) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l3079_307918


namespace NUMINAMATH_CALUDE_net_difference_in_expenditure_l3079_307995

/-- Represents the problem of calculating the net difference in expenditure after a price increase --/
theorem net_difference_in_expenditure
  (original_price : ℝ)
  (original_quantity : ℝ)
  (price_increase_percentage : ℝ)
  (budget : ℝ)
  (purchased_percentage : ℝ)
  (h1 : price_increase_percentage = 0.25)
  (h2 : budget = 150)
  (h3 : purchased_percentage = 0.64)
  (h4 : original_price * original_quantity = budget)
  (h5 : original_quantity ≤ 40) :
  original_price * original_quantity - (original_price * (1 + price_increase_percentage)) * (purchased_percentage * original_quantity) = 30 :=
by sorry

end NUMINAMATH_CALUDE_net_difference_in_expenditure_l3079_307995


namespace NUMINAMATH_CALUDE_rose_additional_money_needed_l3079_307991

/-- The amount of additional money Rose needs to buy her art supplies -/
theorem rose_additional_money_needed 
  (paintbrush_cost : ℚ)
  (paints_cost : ℚ)
  (easel_cost : ℚ)
  (rose_current_money : ℚ)
  (h1 : paintbrush_cost = 2.40)
  (h2 : paints_cost = 9.20)
  (h3 : easel_cost = 6.50)
  (h4 : rose_current_money = 7.10) :
  paintbrush_cost + paints_cost + easel_cost - rose_current_money = 11 :=
by sorry

end NUMINAMATH_CALUDE_rose_additional_money_needed_l3079_307991


namespace NUMINAMATH_CALUDE_factor_expression_l3079_307984

theorem factor_expression (x : ℝ) : 35 * x^11 + 49 * x^22 = 7 * x^11 * (5 + 7 * x^11) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l3079_307984


namespace NUMINAMATH_CALUDE_cube_root_sum_equals_sixty_l3079_307989

theorem cube_root_sum_equals_sixty : 
  (30^3 + 40^3 + 50^3 : ℝ)^(1/3) = 60 := by sorry

end NUMINAMATH_CALUDE_cube_root_sum_equals_sixty_l3079_307989


namespace NUMINAMATH_CALUDE_waitress_income_fraction_l3079_307954

theorem waitress_income_fraction (S : ℚ) : 
  let first_week_salary := S
  let first_week_tips := (11 / 4) * S
  let second_week_salary := (5 / 4) * S
  let second_week_tips := (7 / 3) * second_week_salary
  let total_salary := first_week_salary + second_week_salary
  let total_tips := first_week_tips + second_week_tips
  let total_income := total_salary + total_tips
  (total_tips / total_income) = 68 / 95 := by
  sorry

end NUMINAMATH_CALUDE_waitress_income_fraction_l3079_307954


namespace NUMINAMATH_CALUDE_two_correct_implications_l3079_307990

-- Define the types for planes and lines
def Plane : Type := Unit
def Line : Type := Unit

-- Define the relations between lines and planes
def perpendicular_to_plane (l : Line) (p : Plane) : Prop := sorry
def parallel_to_plane (l : Line) (p : Plane) : Prop := sorry
def perpendicular_planes (p1 p2 : Plane) : Prop := sorry
def not_in_plane (l : Line) (p : Plane) : Prop := sorry

theorem two_correct_implications 
  (α β : Plane) 
  (l : Line) 
  (h_diff : α ≠ β)
  (h_not_in_α : not_in_plane l α)
  (h_not_in_β : not_in_plane l β)
  (h1 : perpendicular_to_plane l α)
  (h2 : parallel_to_plane l β)
  (h3 : perpendicular_planes α β) :
  ∃ (P Q R : Prop),
    (P ∧ Q → R) ∧
    (P ∧ R → Q) ∧
    ¬(Q ∧ R → P) ∧
    P ≠ Q ∧ Q ≠ R ∧ R ≠ P ∧
    (P = perpendicular_to_plane l α ∨ 
     P = parallel_to_plane l β ∨ 
     P = perpendicular_planes α β) ∧
    (Q = perpendicular_to_plane l α ∨ 
     Q = parallel_to_plane l β ∨ 
     Q = perpendicular_planes α β) ∧
    (R = perpendicular_to_plane l α ∨ 
     R = parallel_to_plane l β ∨ 
     R = perpendicular_planes α β) :=
by sorry

end NUMINAMATH_CALUDE_two_correct_implications_l3079_307990


namespace NUMINAMATH_CALUDE_nine_sequences_exist_l3079_307948

/-- An arithmetic sequence of natural numbers. -/
structure ArithSeq where
  first : ℕ
  diff : ℕ

/-- The nth term of an arithmetic sequence. -/
def ArithSeq.nthTerm (seq : ArithSeq) (n : ℕ) : ℕ :=
  seq.first + (n - 1) * seq.diff

/-- The sum of the first n terms of an arithmetic sequence. -/
def ArithSeq.sumFirstN (seq : ArithSeq) (n : ℕ) : ℕ :=
  n * (2 * seq.first + (n - 1) * seq.diff) / 2

/-- The property that the ratio of sum of first 2n terms to sum of first n terms is constant. -/
def ArithSeq.hasConstantRatio (seq : ArithSeq) : Prop :=
  ∀ n : ℕ, n > 0 → (seq.sumFirstN (2*n)) / (seq.sumFirstN n) = 4

/-- The property that 1971 is a term in the sequence. -/
def ArithSeq.contains1971 (seq : ArithSeq) : Prop :=
  ∃ k : ℕ, seq.nthTerm k = 1971

/-- The main theorem stating that there are exactly 9 sequences satisfying both properties. -/
theorem nine_sequences_exist : 
  ∃! (s : Finset ArithSeq), 
    s.card = 9 ∧ 
    (∀ seq ∈ s, seq.hasConstantRatio ∧ seq.contains1971) ∧
    (∀ seq : ArithSeq, seq.hasConstantRatio ∧ seq.contains1971 → seq ∈ s) :=
sorry

end NUMINAMATH_CALUDE_nine_sequences_exist_l3079_307948


namespace NUMINAMATH_CALUDE_fraction_value_l3079_307972

theorem fraction_value (a b c : ℤ) 
  (eq1 : a + b = 20) 
  (eq2 : b + c = 22) 
  (eq3 : c + a = 2022) : 
  (a - b) / (c - a) = 1000 := by
sorry

end NUMINAMATH_CALUDE_fraction_value_l3079_307972


namespace NUMINAMATH_CALUDE_evaluate_f_l3079_307996

def f (x : ℝ) : ℝ := 3 * x^2 - 6 * x + 10

theorem evaluate_f : 3 * f 2 + 2 * f (-2) = 98 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_f_l3079_307996


namespace NUMINAMATH_CALUDE_consecutive_product_square_appendage_l3079_307914

theorem consecutive_product_square_appendage (n : ℕ) :
  ∃ (a b : ℕ), a < 10 ∧ b < 10 ∧ ∃ (k : ℕ), 100 * (n * (n + 1)) + 10 * a + b = k ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_product_square_appendage_l3079_307914


namespace NUMINAMATH_CALUDE_matilda_has_420_jellybeans_l3079_307958

/-- The number of jellybeans Steve has -/
def steve_jellybeans : ℕ := 84

/-- The number of jellybeans Matt has -/
def matt_jellybeans : ℕ := 10 * steve_jellybeans

/-- The number of jellybeans Matilda has -/
def matilda_jellybeans : ℕ := matt_jellybeans / 2

/-- Theorem stating that Matilda has 420 jellybeans -/
theorem matilda_has_420_jellybeans : matilda_jellybeans = 420 := by
  sorry

end NUMINAMATH_CALUDE_matilda_has_420_jellybeans_l3079_307958


namespace NUMINAMATH_CALUDE_minimum_fraction_ponies_with_horseshoes_l3079_307934

theorem minimum_fraction_ponies_with_horseshoes :
  ∀ (num_ponies num_horses num_ponies_with_horseshoes num_icelandic_ponies_with_horseshoes : ℕ),
  num_horses = num_ponies + 4 →
  num_horses + num_ponies ≥ 164 →
  8 * num_icelandic_ponies_with_horseshoes = 5 * num_ponies_with_horseshoes →
  num_ponies_with_horseshoes ≤ num_ponies →
  (∃ (min_fraction : ℚ), 
    min_fraction = num_ponies_with_horseshoes / num_ponies ∧
    min_fraction = 1 / 10) :=
by sorry

end NUMINAMATH_CALUDE_minimum_fraction_ponies_with_horseshoes_l3079_307934


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l3079_307900

theorem geometric_sequence_ratio (a : ℕ → ℝ) (h_positive : ∀ n, a n > 0) 
  (h_geometric : ∃ q : ℝ, q > 0 ∧ ∀ n, a (n + 1) = q * a n)
  (h_arithmetic : 6 * a 1 + 4 * a 2 = 2 * a 3) :
  (a 11 + a 13 + a 16 + a 20 + a 21) / (a 8 + a 10 + a 13 + a 17 + a 18) = 27 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l3079_307900


namespace NUMINAMATH_CALUDE_equation_solution_l3079_307907

/-- The floor function -/
noncomputable def floor (x : ℝ) : ℤ := Int.floor x

/-- The fractional part function -/
noncomputable def frac (x : ℝ) : ℝ := x - floor x

/-- The set of solutions to the equation -/
def solution_set : Set ℝ := {29/12, 19/6, 97/24}

/-- The main theorem -/
theorem equation_solution :
  ∀ x : ℝ, (1 / (floor x : ℝ) + 1 / (floor (2*x) : ℝ) = frac x + 1/3) ↔ x ∈ solution_set := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3079_307907


namespace NUMINAMATH_CALUDE_mark_sold_nine_boxes_l3079_307939

/-- Given that Mark and Ann were allocated n boxes of cookies to sell, prove that Mark sold 9 boxes. -/
theorem mark_sold_nine_boxes (n : ℕ) (mark_boxes ann_boxes : ℕ) : 
  n = 10 →
  mark_boxes < n →
  ann_boxes = n - 2 →
  mark_boxes ≥ 1 →
  ann_boxes ≥ 1 →
  mark_boxes + ann_boxes < n →
  mark_boxes = 9 := by
sorry

end NUMINAMATH_CALUDE_mark_sold_nine_boxes_l3079_307939


namespace NUMINAMATH_CALUDE_sodas_drunk_equals_three_l3079_307967

/-- The number of sodas Robin bought -/
def total_sodas : ℕ := 11

/-- The number of sodas left after drinking -/
def extras : ℕ := 8

/-- The number of sodas drunk -/
def sodas_drunk : ℕ := total_sodas - extras

theorem sodas_drunk_equals_three : sodas_drunk = 3 := by
  sorry

end NUMINAMATH_CALUDE_sodas_drunk_equals_three_l3079_307967


namespace NUMINAMATH_CALUDE_student_age_l3079_307999

theorem student_age (student_age man_age : ℕ) : 
  man_age = student_age + 26 →
  man_age + 2 = 2 * (student_age + 2) →
  student_age = 24 := by
sorry

end NUMINAMATH_CALUDE_student_age_l3079_307999


namespace NUMINAMATH_CALUDE_smallest_composite_no_small_factors_l3079_307994

def is_composite (n : ℕ) : Prop := ∃ a b, 1 < a ∧ 1 < b ∧ n = a * b

def has_no_small_prime_factors (n : ℕ) : Prop := ∀ p, p < 13 → ¬(Nat.Prime p ∧ p ∣ n)

theorem smallest_composite_no_small_factors : 
  (is_composite 169 ∧ has_no_small_prime_factors 169) ∧ 
  (∀ m : ℕ, m < 169 → ¬(is_composite m ∧ has_no_small_prime_factors m)) :=
sorry

end NUMINAMATH_CALUDE_smallest_composite_no_small_factors_l3079_307994


namespace NUMINAMATH_CALUDE_unique_polynomial_coefficients_l3079_307927

theorem unique_polynomial_coefficients :
  ∃! (a b c : ℕ+),
  let x : ℝ := Real.sqrt ((Real.sqrt 105) / 2 + 7 / 2)
  x^100 = 3*x^98 + 15*x^96 + 12*x^94 - x^50 + (a:ℝ)*x^46 + (b:ℝ)*x^44 + (c:ℝ)*x^40 ∧
  a + b + c = 5824 := by
sorry

end NUMINAMATH_CALUDE_unique_polynomial_coefficients_l3079_307927


namespace NUMINAMATH_CALUDE_f_monotonicity_f_monotonic_increasing_iff_f_monotonic_decreasing_increasing_iff_l3079_307956

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a * x - 1

theorem f_monotonicity (a : ℝ) :
  (a > 0 → ∀ x y, x > Real.log a → y > Real.log a → x < y → f a x < f a y) ∧
  (a ≤ 0 → ∀ x y, x < y → f a x < f a y) :=
sorry

theorem f_monotonic_increasing_iff (a : ℝ) :
  (∀ x y, x < y → f a x < f a y) ↔ a ≤ 0 :=
sorry

theorem f_monotonic_decreasing_increasing_iff (a : ℝ) :
  (∀ x y, x < y → x ≤ 0 → f a x > f a y) ∧
  (∀ x y, x < y → x ≥ 0 → f a x < f a y) ↔
  a = 1 :=
sorry

end NUMINAMATH_CALUDE_f_monotonicity_f_monotonic_increasing_iff_f_monotonic_decreasing_increasing_iff_l3079_307956


namespace NUMINAMATH_CALUDE_brett_travel_distance_l3079_307997

/-- The distance traveled given a constant speed and time -/
def distance_traveled (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Theorem: Brett's travel distance in 12 hours at 75 miles per hour is 900 miles -/
theorem brett_travel_distance : distance_traveled 75 12 = 900 := by
  sorry

end NUMINAMATH_CALUDE_brett_travel_distance_l3079_307997


namespace NUMINAMATH_CALUDE_billion_to_scientific_notation_l3079_307977

/-- Represents the number 56.9 billion -/
def billion_value : ℝ := 56.9 * 1000000000

/-- Represents the scientific notation of 56.9 billion -/
def scientific_notation : ℝ := 5.69 * 10^9

/-- Theorem stating that 56.9 billion is equal to 5.69 × 10^9 in scientific notation -/
theorem billion_to_scientific_notation : billion_value = scientific_notation := by
  sorry

end NUMINAMATH_CALUDE_billion_to_scientific_notation_l3079_307977


namespace NUMINAMATH_CALUDE_flora_milk_consumption_l3079_307920

/-- Calculates the total amount of milk Flora needs to drink based on the given conditions -/
def total_milk_gallons (weeks : ℕ) (flora_estimate : ℕ) (brother_additional : ℕ) : ℕ :=
  let days := weeks * 7
  let daily_amount := flora_estimate + brother_additional
  days * daily_amount

/-- Theorem stating that the total amount of milk Flora needs to drink is 105 gallons -/
theorem flora_milk_consumption :
  total_milk_gallons 3 3 2 = 105 := by
  sorry

end NUMINAMATH_CALUDE_flora_milk_consumption_l3079_307920


namespace NUMINAMATH_CALUDE_like_terms_exponent_product_l3079_307976

theorem like_terms_exponent_product (x y : ℝ) (m n : ℕ) : 
  (∀ (a b : ℝ), a * x^3 * y^n = b * x^m * y^2 → a ≠ 0 → b ≠ 0 → m = 3 ∧ n = 2) →
  m * n = 6 := by
  sorry

end NUMINAMATH_CALUDE_like_terms_exponent_product_l3079_307976


namespace NUMINAMATH_CALUDE_vector_addition_l3079_307952

theorem vector_addition (a b : ℝ × ℝ) :
  a = (2, 1) → b = (-3, 4) → a + b = (-1, 5) := by
  sorry

end NUMINAMATH_CALUDE_vector_addition_l3079_307952


namespace NUMINAMATH_CALUDE_sticker_packs_total_cost_l3079_307919

/-- Calculates the total cost of sticker packs bought over three days --/
def total_cost (monday_packs : ℕ) (monday_price : ℚ) (monday_discount : ℚ)
                (tuesday_packs : ℕ) (tuesday_price : ℚ) (tuesday_tax : ℚ)
                (wednesday_packs : ℕ) (wednesday_price : ℚ) (wednesday_discount : ℚ) (wednesday_tax : ℚ) : ℚ :=
  let monday_cost := (monday_packs : ℚ) * monday_price * (1 - monday_discount)
  let tuesday_cost := (tuesday_packs : ℚ) * tuesday_price * (1 + tuesday_tax)
  let wednesday_cost := (wednesday_packs : ℚ) * wednesday_price * (1 - wednesday_discount) * (1 + wednesday_tax)
  monday_cost + tuesday_cost + wednesday_cost

/-- Theorem stating the total cost of sticker packs over three days --/
theorem sticker_packs_total_cost :
  total_cost 15 (5/2) (1/10) 25 3 (1/20) 30 (7/2) (3/20) (2/25) = 20889/100 :=
by sorry

end NUMINAMATH_CALUDE_sticker_packs_total_cost_l3079_307919


namespace NUMINAMATH_CALUDE_largest_g_is_correct_l3079_307986

/-- The largest positive integer g for which there exists exactly one pair of positive integers (a, b) satisfying 5a + gb = 70 -/
def largest_g : ℕ := 65

/-- The unique pair of positive integers (a, b) satisfying 5a + (largest_g)b = 70 -/
def unique_pair : ℕ × ℕ := (1, 1)

theorem largest_g_is_correct :
  (∀ g : ℕ, g > largest_g →
    ¬(∃! p : ℕ × ℕ, p.1 > 0 ∧ p.2 > 0 ∧ 5 * p.1 + g * p.2 = 70)) ∧
  (∃! p : ℕ × ℕ, p.1 > 0 ∧ p.2 > 0 ∧ 5 * p.1 + largest_g * p.2 = 70) ∧
  (unique_pair.1 > 0 ∧ unique_pair.2 > 0 ∧ 5 * unique_pair.1 + largest_g * unique_pair.2 = 70) :=
by sorry

#check largest_g_is_correct

end NUMINAMATH_CALUDE_largest_g_is_correct_l3079_307986


namespace NUMINAMATH_CALUDE_number_ratio_and_sum_of_squares_l3079_307998

theorem number_ratio_and_sum_of_squares (x y : ℝ) (h1 : x > 0) (h2 : y > 0) : 
  x / y = 2 / (3/2) → x^2 + y^2 = 400 → x = 16 ∧ y = 12 := by
  sorry

end NUMINAMATH_CALUDE_number_ratio_and_sum_of_squares_l3079_307998


namespace NUMINAMATH_CALUDE_base_prime_rep_360_l3079_307928

def base_prime_representation (n : ℕ) : List ℕ := sorry

theorem base_prime_rep_360 :
  base_prime_representation 360 = [3, 2, 0, 1] :=
by
  sorry

end NUMINAMATH_CALUDE_base_prime_rep_360_l3079_307928


namespace NUMINAMATH_CALUDE_maximize_sqrt_expression_l3079_307992

theorem maximize_sqrt_expression :
  let add := Real.sqrt 8 + Real.sqrt 2
  let mul := Real.sqrt 8 * Real.sqrt 2
  let div := Real.sqrt 8 / Real.sqrt 2
  let sub := Real.sqrt 8 - Real.sqrt 2
  add > mul ∧ add > div ∧ add > sub := by
  sorry

end NUMINAMATH_CALUDE_maximize_sqrt_expression_l3079_307992


namespace NUMINAMATH_CALUDE_double_in_fifty_years_l3079_307942

/-- The interest rate (in percentage) that doubles an initial sum in 50 years under simple interest -/
def double_interest_rate : ℝ := 2

theorem double_in_fifty_years (P : ℝ) (P_pos : P > 0) :
  P * (1 + double_interest_rate * 50 / 100) = 2 * P := by
  sorry

#check double_in_fifty_years

end NUMINAMATH_CALUDE_double_in_fifty_years_l3079_307942


namespace NUMINAMATH_CALUDE_cost_of_chips_l3079_307930

/-- The cost of chips when three friends split the bill equally -/
theorem cost_of_chips (num_friends : ℕ) (num_bags : ℕ) (payment_per_friend : ℚ) : 
  num_friends = 3 → num_bags = 5 → payment_per_friend = 5 →
  (num_friends * payment_per_friend) / num_bags = 3 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_chips_l3079_307930


namespace NUMINAMATH_CALUDE_set_relationship_l3079_307926

theorem set_relationship (A B C : Set α) (hAnonempty : A.Nonempty) (hBnonempty : B.Nonempty) (hCnonempty : C.Nonempty)
  (hUnion : A ∪ B = C) (hNotSubset : ¬(B ⊆ A)) :
  (∀ x, x ∈ A → x ∈ C) ∧ ¬(∀ x, x ∈ C → x ∈ A) := by
sorry

end NUMINAMATH_CALUDE_set_relationship_l3079_307926


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l3079_307963

theorem min_value_reciprocal_sum (a b : ℝ) : 
  a > 0 → b > 0 → 2*a + 2*b = 2 → (1/a + 1/b ≥ 4) ∧ (∃ a b, 1/a + 1/b = 4) :=
by sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l3079_307963


namespace NUMINAMATH_CALUDE_total_spent_after_discount_and_tax_l3079_307941

def bracelet_price : ℝ := 4
def keychain_price : ℝ := 5
def coloring_book_price : ℝ := 3
def sticker_pack_price : ℝ := 1
def toy_car_price : ℝ := 6

def bracelet_discount_rate : ℝ := 0.1
def sales_tax_rate : ℝ := 0.05

def paula_bracelets : ℕ := 3
def paula_keychains : ℕ := 2
def paula_coloring_books : ℕ := 1
def paula_sticker_packs : ℕ := 4

def olive_coloring_books : ℕ := 1
def olive_bracelets : ℕ := 2
def olive_toy_cars : ℕ := 1
def olive_sticker_packs : ℕ := 3

def nathan_toy_cars : ℕ := 4
def nathan_sticker_packs : ℕ := 5
def nathan_keychains : ℕ := 1

theorem total_spent_after_discount_and_tax : 
  let paula_total := paula_bracelets * bracelet_price + paula_keychains * keychain_price + 
                     paula_coloring_books * coloring_book_price + paula_sticker_packs * sticker_pack_price
  let olive_total := olive_coloring_books * coloring_book_price + olive_bracelets * bracelet_price + 
                     olive_toy_cars * toy_car_price + olive_sticker_packs * sticker_pack_price
  let nathan_total := nathan_toy_cars * toy_car_price + nathan_sticker_packs * sticker_pack_price + 
                      nathan_keychains * keychain_price
  let paula_discount := paula_bracelets * bracelet_price * bracelet_discount_rate
  let olive_discount := olive_bracelets * bracelet_price * bracelet_discount_rate
  let total_before_tax := paula_total - paula_discount + olive_total - olive_discount + nathan_total
  let total_after_tax := total_before_tax * (1 + sales_tax_rate)
  total_after_tax = 85.05 := by sorry

end NUMINAMATH_CALUDE_total_spent_after_discount_and_tax_l3079_307941


namespace NUMINAMATH_CALUDE_abscissa_of_point_M_l3079_307910

/-- Given a point M with coordinates (1,1), prove that its abscissa is 1 -/
theorem abscissa_of_point_M (M : ℝ × ℝ) (h : M = (1, 1)) : M.1 = 1 := by
  sorry

end NUMINAMATH_CALUDE_abscissa_of_point_M_l3079_307910


namespace NUMINAMATH_CALUDE_find_divisor_l3079_307949

theorem find_divisor (dividend quotient remainder : ℕ) (h1 : dividend = 17698) (h2 : quotient = 89) (h3 : remainder = 14) :
  ∃ (divisor : ℕ), dividend = divisor * quotient + remainder ∧ divisor = 198 := by
  sorry

end NUMINAMATH_CALUDE_find_divisor_l3079_307949


namespace NUMINAMATH_CALUDE_triangle_side_lengths_l3079_307933

/-- Given a triangle with sides in ratio 3:4:5 and perimeter 60, prove its side lengths are 15, 20, and 25 -/
theorem triangle_side_lengths (a b c : ℝ) (h_ratio : a / b = 3 / 4 ∧ b / c = 4 / 5) 
  (h_perimeter : a + b + c = 60) : a = 15 ∧ b = 20 ∧ c = 25 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_lengths_l3079_307933


namespace NUMINAMATH_CALUDE_tailor_cut_difference_l3079_307932

theorem tailor_cut_difference (dress_outer dress_middle dress_inner pants_outer pants_inner : ℝ) 
  (h1 : dress_outer = 0.75)
  (h2 : dress_middle = 0.60)
  (h3 : dress_inner = 0.55)
  (h4 : pants_outer = 0.50)
  (h5 : pants_inner = 0.45) :
  (dress_outer + dress_middle + dress_inner) - (pants_outer + pants_inner) = 0.95 := by
  sorry

end NUMINAMATH_CALUDE_tailor_cut_difference_l3079_307932


namespace NUMINAMATH_CALUDE_part_one_part_two_l3079_307905

-- Define the equation
def equation (x a : ℝ) : Prop := (x + a) / (x - 2) - 5 / x = 1

-- Part 1: When x = 5 is a root
theorem part_one (a : ℝ) : (5 + a) / 3 - 1 = 1 → a = 1 := by sorry

-- Part 2: When the equation has no solution
theorem part_two (a : ℝ) : (∀ x : ℝ, ¬ equation x a) ↔ a = 3 ∨ a = -2 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l3079_307905


namespace NUMINAMATH_CALUDE_function_composition_sum_l3079_307946

theorem function_composition_sum (a b : ℝ) :
  (∀ x, (5 * (a * x + b) - 7) = 4 * x + 6) →
  a + b = 17 / 5 := by
sorry

end NUMINAMATH_CALUDE_function_composition_sum_l3079_307946


namespace NUMINAMATH_CALUDE_solve_equation_l3079_307915

theorem solve_equation : ∃ x : ℚ, 5 * (x - 9) = 7 * (3 - 3 * x) + 10 ∧ x = 38 / 13 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l3079_307915


namespace NUMINAMATH_CALUDE_triangle_incenter_distance_l3079_307982

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

end NUMINAMATH_CALUDE_triangle_incenter_distance_l3079_307982


namespace NUMINAMATH_CALUDE_shoe_cost_l3079_307903

def budget : ℕ := 200
def other_expenses : ℕ := 143
def remaining : ℕ := 16

theorem shoe_cost (budget : ℕ) (other_expenses : ℕ) (remaining : ℕ) :
  budget = other_expenses + remaining + 41 :=
by sorry

end NUMINAMATH_CALUDE_shoe_cost_l3079_307903


namespace NUMINAMATH_CALUDE_workshop_selection_count_l3079_307968

/-- The number of photography enthusiasts --/
def total_students : ℕ := 4

/-- The number of sessions in the workshop --/
def num_sessions : ℕ := 3

/-- The number of students who cannot participate in the first session --/
def restricted_students : ℕ := 2

/-- The number of different ways to select students for the workshop --/
def selection_methods : ℕ := (total_students - restricted_students) * (total_students - 1) * (total_students - 2)

theorem workshop_selection_count :
  selection_methods = 12 :=
sorry

end NUMINAMATH_CALUDE_workshop_selection_count_l3079_307968


namespace NUMINAMATH_CALUDE_tower_combinations_l3079_307909

/-- Represents the number of cubes of each color --/
structure CubeColors where
  red : Nat
  blue : Nat
  green : Nat
  yellow : Nat

/-- Calculates the number of different towers that can be built --/
def numTowers (colors : CubeColors) (towerHeight : Nat) : Nat :=
  if towerHeight ≠ colors.red + colors.blue + colors.green + colors.yellow - 1 then 0
  else if colors.yellow = 0 then 0
  else
    let n := towerHeight - 1
    Nat.factorial n / (Nat.factorial colors.red * Nat.factorial colors.blue * 
                       Nat.factorial colors.green * Nat.factorial (colors.yellow - 1))

/-- The main theorem to be proven --/
theorem tower_combinations : 
  let colors := CubeColors.mk 3 4 2 2
  numTowers colors 10 = 1260 := by
  sorry

end NUMINAMATH_CALUDE_tower_combinations_l3079_307909


namespace NUMINAMATH_CALUDE_vector_problem_l3079_307929

def a : ℝ × ℝ := (3, 4)
def b : ℝ × ℝ := (-1, 7)

theorem vector_problem :
  (a.1 * b.1 + a.2 * b.2 = 25) ∧
  (Real.arccos ((a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2))) = π / 4) := by
  sorry

end NUMINAMATH_CALUDE_vector_problem_l3079_307929


namespace NUMINAMATH_CALUDE_black_population_percentage_in_south_l3079_307924

/-- Represents the population data for a region --/
structure RegionData where
  white : ℕ
  black : ℕ
  asian : ℕ
  other : ℕ

/-- Represents the demographic data for the entire nation --/
structure NationData where
  ne : RegionData
  mw : RegionData
  central : RegionData
  south : RegionData
  west : RegionData

def total_black_population (data : NationData) : ℕ :=
  data.ne.black + data.mw.black + data.central.black + data.south.black + data.west.black

def black_population_in_south (data : NationData) : ℕ :=
  data.south.black

def percentage_in_south (data : NationData) : ℚ :=
  (black_population_in_south data : ℚ) / (total_black_population data : ℚ) * 100

def round_to_nearest_percent (x : ℚ) : ℕ :=
  (x + 1/2).floor.toNat

theorem black_population_percentage_in_south (data : NationData) :
  data.ne.black = 6 →
  data.mw.black = 7 →
  data.central.black = 3 →
  data.south.black = 23 →
  data.west.black = 5 →
  round_to_nearest_percent (percentage_in_south data) = 52 := by
  sorry

end NUMINAMATH_CALUDE_black_population_percentage_in_south_l3079_307924


namespace NUMINAMATH_CALUDE_remainder_sum_mod_l3079_307980

theorem remainder_sum_mod (x y : ℤ) (hx : x ≠ y) 
  (hx_mod : x % 124 = 13) (hy_mod : y % 186 = 17) : 
  (x + y + 19) % 62 = 49 := by
  sorry

end NUMINAMATH_CALUDE_remainder_sum_mod_l3079_307980


namespace NUMINAMATH_CALUDE_total_wheels_is_64_l3079_307917

/-- The number of wheels on a four-wheeler -/
def wheels_per_four_wheeler : ℕ := 4

/-- The number of four-wheelers parked in the school -/
def num_four_wheelers : ℕ := 16

/-- The total number of wheels for all four-wheelers parked in the school -/
def total_wheels_four_wheelers : ℕ := num_four_wheelers * wheels_per_four_wheeler

/-- Theorem: The total number of wheels for the four-wheelers parked in the school is 64 -/
theorem total_wheels_is_64 : total_wheels_four_wheelers = 64 := by
  sorry

end NUMINAMATH_CALUDE_total_wheels_is_64_l3079_307917


namespace NUMINAMATH_CALUDE_two_books_cost_exceeds_min_preparation_l3079_307943

/-- The cost of one storybook in yuan -/
def storybook_cost : ℚ := 25.5

/-- The minimum amount Wang Hong needs to prepare in yuan -/
def min_preparation : ℚ := 50

/-- Theorem: The cost of two storybooks is greater than the minimum preparation amount -/
theorem two_books_cost_exceeds_min_preparation : 2 * storybook_cost > min_preparation := by
  sorry

end NUMINAMATH_CALUDE_two_books_cost_exceeds_min_preparation_l3079_307943


namespace NUMINAMATH_CALUDE_jake_shooting_improvement_l3079_307923

theorem jake_shooting_improvement (initial_shots : ℕ) (initial_percentage : ℚ) 
  (additional_shots : ℕ) (new_percentage : ℚ) : 
  initial_shots = 30 ∧ 
  initial_percentage = 3/5 ∧ 
  additional_shots = 10 ∧ 
  new_percentage = 31/50 →
  (new_percentage * (initial_shots + additional_shots) - initial_percentage * initial_shots : ℚ) = 7 := by
  sorry

end NUMINAMATH_CALUDE_jake_shooting_improvement_l3079_307923


namespace NUMINAMATH_CALUDE_stratified_sampling_third_grade_l3079_307953

def total_students : ℕ := 270000
def third_grade_students : ℕ := 81000
def sample_size : ℕ := 3000

theorem stratified_sampling_third_grade :
  (third_grade_students * sample_size) / total_students = 900 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_third_grade_l3079_307953


namespace NUMINAMATH_CALUDE_probability_VIP_ticket_specific_l3079_307965

/-- The probability of drawing a VIP ticket from a set of tickets -/
def probability_VIP_ticket (num_VIP : ℕ) (num_regular : ℕ) : ℚ :=
  num_VIP / (num_VIP + num_regular)

/-- Theorem: The probability of drawing a VIP ticket from a set of 1 VIP ticket and 2 regular tickets is 1/3 -/
theorem probability_VIP_ticket_specific : probability_VIP_ticket 1 2 = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_probability_VIP_ticket_specific_l3079_307965


namespace NUMINAMATH_CALUDE_min_resistance_optimal_l3079_307970

noncomputable def min_resistance (a₁ a₂ a₃ a₄ a₅ a₆ : ℝ) : ℝ :=
  let r₁₂ := (a₁ * a₂) / (a₁ + a₂)
  let r₁₂₃ := r₁₂ + a₃
  let r₄₅ := (a₄ * a₅) / (a₄ + a₅)
  let r₄₅₆ := r₄₅ + a₆
  (r₁₂₃ * r₄₅₆) / (r₁₂₃ + r₄₅₆)

theorem min_resistance_optimal
  (a₁ a₂ a₃ a₄ a₅ a₆ : ℝ)
  (h : a₁ > a₂ ∧ a₂ > a₃ ∧ a₃ > a₄ ∧ a₄ > a₅ ∧ a₅ > a₆) :
  ∀ (r : ℝ), r ≥ min_resistance a₁ a₂ a₃ a₄ a₅ a₆ :=
by sorry

end NUMINAMATH_CALUDE_min_resistance_optimal_l3079_307970


namespace NUMINAMATH_CALUDE_largest_n_satisfying_conditions_l3079_307987

theorem largest_n_satisfying_conditions : 
  ∃ (m : ℤ), (313 : ℤ)^2 = (m + 1)^3 - m^3 ∧ 
  ∃ (k : ℤ), (2 * 313 + 103 : ℤ) = k^2 ∧
  ∀ (n : ℤ), n > 313 → 
    (∃ (m : ℤ), n^2 = (m + 1)^3 - m^3 ∧ 
    ∃ (k : ℤ), (2 * n + 103 : ℤ) = k^2) → False :=
sorry

end NUMINAMATH_CALUDE_largest_n_satisfying_conditions_l3079_307987


namespace NUMINAMATH_CALUDE_ticket_distribution_theorem_l3079_307902

/-- The number of ways to distribute 5 consecutive movie tickets among 5 people,
    with A and B receiving consecutive tickets -/
def ticket_distribution_count : ℕ := 48

/-- The number of ways to group 5 consecutive tickets into 4 groups,
    where one group consists of 2 consecutive tickets -/
def consecutive_pair_groupings : ℕ := 4

/-- The number of ways to distribute 2 consecutive tickets to A and B -/
def ab_distribution_ways : ℕ := 2

/-- The number of ways to permute 3 tickets among 3 people -/
def remaining_ticket_permutations : ℕ := 6

theorem ticket_distribution_theorem :
  ticket_distribution_count =
  consecutive_pair_groupings * ab_distribution_ways * remaining_ticket_permutations :=
sorry

end NUMINAMATH_CALUDE_ticket_distribution_theorem_l3079_307902


namespace NUMINAMATH_CALUDE_regular_tetrahedron_face_center_volume_ratio_l3079_307966

/-- The ratio of the volume of a tetrahedron formed by the centers of the faces of a regular tetrahedron to the volume of the original tetrahedron -/
def face_center_tetrahedron_volume_ratio : ℚ :=
  8 / 27

/-- Theorem stating that in a regular tetrahedron, the ratio of the volume of the tetrahedron 
    formed by the centers of the faces to the volume of the original tetrahedron is 8/27 -/
theorem regular_tetrahedron_face_center_volume_ratio :
  face_center_tetrahedron_volume_ratio = 8 / 27 := by
  sorry

#eval Nat.gcd 8 27  -- To verify that 8 and 27 are coprime

#eval 8 + 27  -- To compute the final answer

end NUMINAMATH_CALUDE_regular_tetrahedron_face_center_volume_ratio_l3079_307966


namespace NUMINAMATH_CALUDE_cookie_sales_revenue_l3079_307964

theorem cookie_sales_revenue : 
  let chocolate_cookies : ℕ := 220
  let vanilla_cookies : ℕ := 70
  let chocolate_price : ℚ := 1
  let vanilla_price : ℚ := 2
  let chocolate_discount : ℚ := 0.1
  let sales_tax_rate : ℚ := 0.05
  
  let chocolate_revenue := chocolate_cookies * chocolate_price
  let chocolate_discount_amount := chocolate_revenue * chocolate_discount
  let discounted_chocolate_revenue := chocolate_revenue - chocolate_discount_amount
  let vanilla_revenue := vanilla_cookies * vanilla_price
  let total_revenue_before_tax := discounted_chocolate_revenue + vanilla_revenue
  let sales_tax := total_revenue_before_tax * sales_tax_rate
  let total_revenue_after_tax := total_revenue_before_tax + sales_tax
  
  total_revenue_after_tax = 354.90 := by sorry

end NUMINAMATH_CALUDE_cookie_sales_revenue_l3079_307964


namespace NUMINAMATH_CALUDE_seven_digit_numbers_even_together_even_odd_together_l3079_307961

/-- The number of even digits from 1 to 9 -/
def num_even_digits : ℕ := 4

/-- The number of odd digits from 1 to 9 -/
def num_odd_digits : ℕ := 5

/-- The number of even digits to be selected -/
def num_even_selected : ℕ := 3

/-- The number of odd digits to be selected -/
def num_odd_selected : ℕ := 4

/-- The total number of digits to be selected -/
def total_selected : ℕ := num_even_selected + num_odd_selected

theorem seven_digit_numbers (n : ℕ) :
  (n = Nat.choose num_even_digits num_even_selected * 
       Nat.choose num_odd_digits num_odd_selected * 
       Nat.factorial total_selected) → 
  n = 100800 := by sorry

theorem even_together (n : ℕ) :
  (n = Nat.choose num_even_digits num_even_selected * 
       Nat.choose num_odd_digits num_odd_selected * 
       Nat.factorial (total_selected - num_even_selected + 1) * 
       Nat.factorial num_even_selected) → 
  n = 14400 := by sorry

theorem even_odd_together (n : ℕ) :
  (n = Nat.choose num_even_digits num_even_selected * 
       Nat.choose num_odd_digits num_odd_selected * 
       Nat.factorial num_even_selected * 
       Nat.factorial num_odd_selected * 
       Nat.factorial 2) → 
  n = 5760 := by sorry

end NUMINAMATH_CALUDE_seven_digit_numbers_even_together_even_odd_together_l3079_307961


namespace NUMINAMATH_CALUDE_grandpas_tomatoes_l3079_307978

/-- The number of tomatoes that grew in Grandpa's absence -/
def tomatoesGrown (initialCount : ℕ) (growthFactor : ℕ) : ℕ :=
  initialCount * growthFactor - initialCount

theorem grandpas_tomatoes :
  tomatoesGrown 36 100 = 3564 := by
  sorry

end NUMINAMATH_CALUDE_grandpas_tomatoes_l3079_307978


namespace NUMINAMATH_CALUDE_M_intersect_N_eq_unit_interval_l3079_307985

-- Define the set M
def M : Set ℝ := {y | ∃ x : ℝ, y = Real.cos x ^ 2 - Real.sin x ^ 2}

-- Define the set N
def N : Set ℝ := {x : ℝ | Complex.abs (2 * x / (1 - Complex.I * Real.sqrt 3)) < 1}

-- State the theorem
theorem M_intersect_N_eq_unit_interval : M ∩ N = Set.Icc 0 1 \ {1} := by sorry

end NUMINAMATH_CALUDE_M_intersect_N_eq_unit_interval_l3079_307985


namespace NUMINAMATH_CALUDE_distance_from_origin_to_point_l3079_307938

theorem distance_from_origin_to_point (z : ℂ) : 
  z = 1260 + 1680 * Complex.I → Complex.abs z = 2100 := by
  sorry

end NUMINAMATH_CALUDE_distance_from_origin_to_point_l3079_307938


namespace NUMINAMATH_CALUDE_all_positive_integers_expressible_l3079_307951

theorem all_positive_integers_expressible (n : ℕ+) :
  ∃ (a b c : ℤ), (n : ℤ) = a^2 + b^2 + c^2 + c := by sorry

end NUMINAMATH_CALUDE_all_positive_integers_expressible_l3079_307951


namespace NUMINAMATH_CALUDE_repeating_decimal_23_value_l3079_307921

/-- The value of the infinite repeating decimal 0.overline{23} -/
def repeating_decimal_23 : ℚ := 23 / 99

/-- Theorem stating that the infinite repeating decimal 0.overline{23} is equal to 23/99 -/
theorem repeating_decimal_23_value : 
  repeating_decimal_23 = 23 / 99 := by sorry

end NUMINAMATH_CALUDE_repeating_decimal_23_value_l3079_307921


namespace NUMINAMATH_CALUDE_function_inequality_implies_upper_bound_l3079_307950

theorem function_inequality_implies_upper_bound (a : ℝ) : 
  (∀ x₁ ∈ Set.Icc (1/2 : ℝ) 1, ∃ x₂ ∈ Set.Icc 2 3, 
    x₁ + 4/x₁ ≥ 2^x₂ + a) → a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_implies_upper_bound_l3079_307950


namespace NUMINAMATH_CALUDE_average_weight_of_class_l3079_307906

/-- The average weight of a class with two sections -/
theorem average_weight_of_class 
  (studentsA : ℕ) (studentsB : ℕ) 
  (avgWeightA : ℚ) (avgWeightB : ℚ) :
  studentsA = 40 →
  studentsB = 30 →
  avgWeightA = 50 →
  avgWeightB = 60 →
  (studentsA * avgWeightA + studentsB * avgWeightB) / (studentsA + studentsB : ℚ) = 3800 / 70 := by
  sorry

#eval (3800 : ℚ) / 70

end NUMINAMATH_CALUDE_average_weight_of_class_l3079_307906


namespace NUMINAMATH_CALUDE_correct_observation_value_l3079_307904

theorem correct_observation_value
  (n : ℕ)
  (initial_mean : ℝ)
  (wrong_value : ℝ)
  (corrected_mean : ℝ)
  (h_n : n = 40)
  (h_initial_mean : initial_mean = 36)
  (h_wrong_value : wrong_value = 20)
  (h_corrected_mean : corrected_mean = 36.45) :
  let correct_value := (n : ℝ) * corrected_mean - ((n : ℝ) - 1) * initial_mean + wrong_value
  correct_value = 58 := by
sorry

end NUMINAMATH_CALUDE_correct_observation_value_l3079_307904


namespace NUMINAMATH_CALUDE_complementary_angles_can_be_equal_l3079_307908

-- Define what complementary angles are
def complementary (α β : ℝ) : Prop := α + β = 90

-- State the theorem
theorem complementary_angles_can_be_equal :
  ∃ (α : ℝ), complementary α α :=
sorry

-- The existence of such an angle pair disproves the statement
-- "Two complementary angles are not equal"

end NUMINAMATH_CALUDE_complementary_angles_can_be_equal_l3079_307908


namespace NUMINAMATH_CALUDE_abs_a_b_sum_l3079_307974

theorem abs_a_b_sum (a b : ℝ) (ha : |a| = 7) (hb : |b| = 3) (hab : a * b > 0) :
  a + b = 10 ∨ a + b = -10 := by
  sorry

end NUMINAMATH_CALUDE_abs_a_b_sum_l3079_307974


namespace NUMINAMATH_CALUDE_cylinder_inscribed_sphere_tangent_spheres_l3079_307925

theorem cylinder_inscribed_sphere_tangent_spheres 
  (cylinder_radius : ℝ) 
  (cylinder_height : ℝ) 
  (large_sphere_radius : ℝ) 
  (small_sphere_radius : ℝ) :
  cylinder_radius = 15 →
  cylinder_height = 16 →
  large_sphere_radius = Real.sqrt (cylinder_radius^2 + (cylinder_height/2)^2) →
  large_sphere_radius = small_sphere_radius + Real.sqrt ((cylinder_height/2 + small_sphere_radius)^2 + (2*small_sphere_radius*Real.sqrt 3/3)^2) →
  small_sphere_radius = (15 * Real.sqrt 37 - 75) / 4 := by
  sorry

end NUMINAMATH_CALUDE_cylinder_inscribed_sphere_tangent_spheres_l3079_307925


namespace NUMINAMATH_CALUDE_odd_function_proof_l3079_307931

-- Define f as a function from ℝ to ℝ
variable (f : ℝ → ℝ)

-- Define h in terms of f
def h (x : ℝ) : ℝ := f x - 9

-- State the theorem
theorem odd_function_proof :
  (∀ x, f (-x) = -f x) →  -- f is an odd function
  h 1 = 2 →               -- h(1) = 2
  f (-1) = -11 :=         -- Conclusion: f(-1) = -11
by
  sorry  -- Proof is omitted as per instructions

end NUMINAMATH_CALUDE_odd_function_proof_l3079_307931


namespace NUMINAMATH_CALUDE_point_placement_result_l3079_307916

theorem point_placement_result (x : ℕ) : ∃ x > 0, 9 * x - 8 = 82 := by
  sorry

#check point_placement_result

end NUMINAMATH_CALUDE_point_placement_result_l3079_307916


namespace NUMINAMATH_CALUDE_integral_equals_minus_eight_implies_a_equals_four_l3079_307981

theorem integral_equals_minus_eight_implies_a_equals_four (a : ℝ) :
  (∫ (x : ℝ) in -a..a, (2 * x - 1)) = -8 → a = 4 := by
  sorry

end NUMINAMATH_CALUDE_integral_equals_minus_eight_implies_a_equals_four_l3079_307981


namespace NUMINAMATH_CALUDE_correct_divisor_l3079_307922

theorem correct_divisor (X D : ℕ) (h1 : X % D = 0) (h2 : X % 12 = 0) (h3 : X / 12 = 56) (h4 : X / D = 32) : D = 21 := by
  sorry

end NUMINAMATH_CALUDE_correct_divisor_l3079_307922


namespace NUMINAMATH_CALUDE_range_of_f_l3079_307993

-- Define the function f
def f (x : ℝ) : ℝ := |x + 4| - |x - 5|

-- State the theorem about the range of f
theorem range_of_f : Set.range f = Set.Icc (-9 : ℝ) 9 := by sorry

end NUMINAMATH_CALUDE_range_of_f_l3079_307993


namespace NUMINAMATH_CALUDE_train_b_length_l3079_307979

/-- Calculates the length of Train B given the conditions of the problem -/
theorem train_b_length : 
  let train_a_speed : ℝ := 10  -- Initial speed of Train A in m/s
  let train_b_speed : ℝ := 12.5  -- Initial speed of Train B in m/s
  let train_a_accel : ℝ := 1  -- Acceleration of Train A in m/s²
  let train_b_decel : ℝ := 0.5  -- Deceleration of Train B in m/s²
  let passing_time : ℝ := 10  -- Time to pass each other in seconds
  
  let train_a_final_speed := train_a_speed + train_a_accel * passing_time
  let train_b_final_speed := train_b_speed - train_b_decel * passing_time
  let relative_speed := train_a_final_speed + train_b_final_speed
  
  relative_speed * passing_time = 275 := by
  sorry

#check train_b_length

end NUMINAMATH_CALUDE_train_b_length_l3079_307979


namespace NUMINAMATH_CALUDE_pizza_combinations_l3079_307971

theorem pizza_combinations (n : ℕ) (k : ℕ) : n = 8 ∧ k = 5 → Nat.choose n k = 56 := by
  sorry

end NUMINAMATH_CALUDE_pizza_combinations_l3079_307971


namespace NUMINAMATH_CALUDE_percentage_sum_l3079_307935

theorem percentage_sum : 
  (20 / 100 * 40) + (25 / 100 * 60) = 23 := by
  sorry

end NUMINAMATH_CALUDE_percentage_sum_l3079_307935


namespace NUMINAMATH_CALUDE_count_special_numbers_eq_252_l3079_307969

/-- The count of numbers between 1000 and 9999 with four different digits 
    in either strictly increasing or strictly decreasing order -/
def count_special_numbers : ℕ := sorry

/-- A number is considered special if it has four different digits 
    in either strictly increasing or strictly decreasing order -/
def is_special (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999 ∧
  (∃ a b c d : ℕ, 
    n = a * 1000 + b * 100 + c * 10 + d ∧
    a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ a ≠ c ∧ a ≠ d ∧ b ≠ d ∧
    ((a < b ∧ b < c ∧ c < d) ∨ (a > b ∧ b > c ∧ c > d)))

theorem count_special_numbers_eq_252 : 
  count_special_numbers = 252 :=
sorry

end NUMINAMATH_CALUDE_count_special_numbers_eq_252_l3079_307969


namespace NUMINAMATH_CALUDE_sons_age_l3079_307940

theorem sons_age (son_age father_age : ℕ) : 
  father_age = son_age + 30 →
  father_age + 5 = 3 * (son_age + 5) →
  son_age = 10 := by
sorry

end NUMINAMATH_CALUDE_sons_age_l3079_307940


namespace NUMINAMATH_CALUDE_cubic_expression_value_l3079_307901

theorem cubic_expression_value (x : ℝ) (h : x^2 + 3*x - 1 = 0) : 
  x^3 + 5*x^2 + 5*x + 18 = 20 := by
  sorry

end NUMINAMATH_CALUDE_cubic_expression_value_l3079_307901
