import Mathlib

namespace NUMINAMATH_CALUDE_rectangle_circle_propositions_l108_10876

theorem rectangle_circle_propositions (p q : Prop) 
  (hp : p) 
  (hq : ¬q) : 
  (p ∨ q) ∧ ¬(p ∧ q) ∧ ¬(¬p) := by
  sorry

end NUMINAMATH_CALUDE_rectangle_circle_propositions_l108_10876


namespace NUMINAMATH_CALUDE_polar_to_cartesian_circle_l108_10837

/-- The polar equation ρ = 4cosθ is equivalent to the Cartesian equation (x-2)^2 + y^2 = 4 -/
theorem polar_to_cartesian_circle :
  ∀ (x y ρ θ : ℝ), 
  (ρ = 4 * Real.cos θ) ∧ 
  (x = ρ * Real.cos θ) ∧ 
  (y = ρ * Real.sin θ) → 
  ((x - 2)^2 + y^2 = 4) :=
by sorry

end NUMINAMATH_CALUDE_polar_to_cartesian_circle_l108_10837


namespace NUMINAMATH_CALUDE_prob_two_red_cards_standard_deck_l108_10807

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : Nat)
  (suits : Nat)
  (cards_per_suit : Nat)
  (red_suits : Nat)
  (black_suits : Nat)

/-- Definition of a standard deck -/
def standard_deck : Deck :=
  { total_cards := 52,
    suits := 4,
    cards_per_suit := 13,
    red_suits := 2,
    black_suits := 2 }

/-- Probability of drawing two red cards in succession -/
def prob_two_red_cards (d : Deck) : Rat :=
  let red_cards := d.red_suits * d.cards_per_suit
  let first_draw := red_cards / d.total_cards
  let second_draw := (red_cards - 1) / (d.total_cards - 1)
  first_draw * second_draw

/-- Theorem: The probability of drawing two red cards in succession from a standard deck is 25/102 -/
theorem prob_two_red_cards_standard_deck :
  prob_two_red_cards standard_deck = 25 / 102 := by
  sorry

end NUMINAMATH_CALUDE_prob_two_red_cards_standard_deck_l108_10807


namespace NUMINAMATH_CALUDE_special_function_is_odd_and_even_l108_10812

/-- A function satisfying the given property -/
def special_function (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x + y ≠ 0 → f (x * y) = (f x + f y) / (x + y)

/-- A function is both odd and even -/
def odd_and_even (f : ℝ → ℝ) : Prop :=
  (∀ x : ℝ, f (-x) = -f x) ∧ (∀ x : ℝ, f (-x) = f x)

/-- The main theorem -/
theorem special_function_is_odd_and_even (f : ℝ → ℝ) (h : special_function f) :
  odd_and_even f :=
sorry

end NUMINAMATH_CALUDE_special_function_is_odd_and_even_l108_10812


namespace NUMINAMATH_CALUDE_election_result_l108_10860

theorem election_result (total_votes : ℕ) (majority : ℕ) (winner_percentage : ℝ) :
  total_votes = 440 →
  majority = 176 →
  winner_percentage * (total_votes : ℝ) / 100 - (100 - winner_percentage) * (total_votes : ℝ) / 100 = majority →
  winner_percentage = 70 := by
sorry

end NUMINAMATH_CALUDE_election_result_l108_10860


namespace NUMINAMATH_CALUDE_real_axis_length_l108_10804

/-- Hyperbola C with center at origin and foci on x-axis -/
structure Hyperbola where
  equation : ℝ → ℝ → Prop
  center_origin : equation 0 0
  foci_on_x_axis : ∀ y, ¬(∃ x ≠ 0, equation x y ∧ equation (-x) y)

/-- Parabola with equation y² = 16x -/
def Parabola : ℝ → ℝ → Prop :=
  λ x y => y^2 = 16 * x

/-- Directrix of the parabola y² = 16x -/
def Directrix : ℝ → Prop :=
  λ x => x = -4

/-- Points A and B where hyperbola C intersects the directrix -/
structure IntersectionPoints (C : Hyperbola) where
  A : ℝ × ℝ
  B : ℝ × ℝ
  on_directrix : Directrix A.1 ∧ Directrix B.1
  on_hyperbola : C.equation A.1 A.2 ∧ C.equation B.1 B.2
  distance : Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 4 * Real.sqrt 3

/-- The theorem to be proved -/
theorem real_axis_length (C : Hyperbola) (AB : IntersectionPoints C) :
  ∃ a : ℝ, a = 4 ∧ ∀ x y, C.equation x y ↔ x^2 / a^2 - y^2 / (a^2 - 4) = 1 :=
sorry

end NUMINAMATH_CALUDE_real_axis_length_l108_10804


namespace NUMINAMATH_CALUDE_quadratic_root_sum_l108_10864

theorem quadratic_root_sum (p q : ℚ) : 
  (1 - Real.sqrt 3) / 2 = -p / 2 - Real.sqrt ((p / 2) ^ 2 - q) →
  |p| + 2 * |q| = 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_sum_l108_10864


namespace NUMINAMATH_CALUDE_first_year_after_2000_with_digit_sum_15_l108_10892

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

def is_valid_year (year : ℕ) : Prop :=
  year > 2000 ∧ sum_of_digits year = 15

theorem first_year_after_2000_with_digit_sum_15 :
  ∀ year : ℕ, is_valid_year year → year ≥ 2049 :=
sorry

end NUMINAMATH_CALUDE_first_year_after_2000_with_digit_sum_15_l108_10892


namespace NUMINAMATH_CALUDE_ivan_petrovich_savings_l108_10874

/-- Simple interest calculation --/
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate * time)

/-- Proof of Ivan Petrovich's retirement savings --/
theorem ivan_petrovich_savings : 
  simple_interest 750000 0.08 12 = 1470000 := by
  sorry

end NUMINAMATH_CALUDE_ivan_petrovich_savings_l108_10874


namespace NUMINAMATH_CALUDE_gloria_money_calculation_l108_10861

def combined_quarters_and_dimes (total_quarters : ℕ) (total_dimes : ℕ) : ℕ :=
  let quarters_put_aside := (2 * total_quarters) / 5
  let remaining_quarters := total_quarters - quarters_put_aside
  remaining_quarters + total_dimes

theorem gloria_money_calculation :
  ∀ (total_quarters : ℕ) (total_dimes : ℕ),
    total_dimes = 5 * total_quarters →
    total_dimes = 350 →
    combined_quarters_and_dimes total_quarters total_dimes = 392 :=
by
  sorry

end NUMINAMATH_CALUDE_gloria_money_calculation_l108_10861


namespace NUMINAMATH_CALUDE_inequality_range_l108_10819

theorem inequality_range (a : ℝ) : 
  (∀ x : ℝ, x^2 - 4*x > 2*a*x + a) ↔ a ∈ Set.Ioo (-4 : ℝ) (-1 : ℝ) :=
sorry

end NUMINAMATH_CALUDE_inequality_range_l108_10819


namespace NUMINAMATH_CALUDE_salary_calculation_l108_10895

theorem salary_calculation (S : ℚ) 
  (food_expense : S / 5 = S * (1 / 5))
  (rent_expense : S / 10 = S * (1 / 10))
  (clothes_expense : S * 3 / 5 = S * (3 / 5))
  (remaining : S - (S / 5) - (S / 10) - (S * 3 / 5) = 18000) :
  S = 180000 := by
sorry

end NUMINAMATH_CALUDE_salary_calculation_l108_10895


namespace NUMINAMATH_CALUDE_number_subtraction_problem_l108_10813

theorem number_subtraction_problem (x : ℝ) : 0.60 * x - 40 = 50 ↔ x = 150 := by
  sorry

end NUMINAMATH_CALUDE_number_subtraction_problem_l108_10813


namespace NUMINAMATH_CALUDE_fraction_change_l108_10857

theorem fraction_change (original_fraction : ℚ) 
  (numerator_increase : ℚ) (denominator_decrease : ℚ) (new_fraction : ℚ) : 
  original_fraction = 3/4 →
  numerator_increase = 12/100 →
  new_fraction = 6/7 →
  (1 + numerator_increase) * original_fraction / (1 - denominator_decrease/100) = new_fraction →
  denominator_decrease = 6 := by
  sorry

end NUMINAMATH_CALUDE_fraction_change_l108_10857


namespace NUMINAMATH_CALUDE_integer_root_values_l108_10899

theorem integer_root_values (b : ℤ) : 
  (∃ x : ℤ, x^3 + 2*x^2 + b*x + 18 = 0) ↔ b ∈ ({-21, 19, -17, -4, 3} : Set ℤ) := by
  sorry

end NUMINAMATH_CALUDE_integer_root_values_l108_10899


namespace NUMINAMATH_CALUDE_ellipse_parabola_intersection_bounds_l108_10810

/-- If an ellipse and a parabola have a common point, then the parameter 'a' of the ellipse is bounded. -/
theorem ellipse_parabola_intersection_bounds (a : ℝ) : 
  (∃ x y : ℝ, x^2 + 4*(y - a)^2 = 4 ∧ x^2 = 2*y) → 
  -1 ≤ a ∧ a ≤ 17/8 := by
sorry

end NUMINAMATH_CALUDE_ellipse_parabola_intersection_bounds_l108_10810


namespace NUMINAMATH_CALUDE_complete_square_formula_l108_10873

theorem complete_square_formula (x y : ℝ) : x^2 - 2*x*y + y^2 = (x - y)^2 := by
  sorry

end NUMINAMATH_CALUDE_complete_square_formula_l108_10873


namespace NUMINAMATH_CALUDE_length_of_segment_AB_l108_10806

-- Define the line l
def line_l (x y : ℝ) : Prop := x - 2*y + 3 = 0

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 6*y + 6 = 0

-- Define the intersection points A and B
def intersection_points (A B : ℝ × ℝ) : Prop :=
  line_l A.1 A.2 ∧ circle_C A.1 A.2 ∧
  line_l B.1 B.2 ∧ circle_C B.1 B.2 ∧
  A ≠ B

-- Theorem statement
theorem length_of_segment_AB (A B : ℝ × ℝ) :
  intersection_points A B →
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 8 * Real.sqrt 5 / 5 :=
sorry

end NUMINAMATH_CALUDE_length_of_segment_AB_l108_10806


namespace NUMINAMATH_CALUDE_total_ingredients_for_batches_l108_10886

/-- The amount of flour needed for one batch of cookies, in cups. -/
def flour_per_batch : ℝ := 4

/-- The amount of sugar needed for one batch of cookies, in cups. -/
def sugar_per_batch : ℝ := 1.5

/-- The number of batches we want to make. -/
def num_batches : ℕ := 8

/-- Theorem: The total amount of flour and sugar needed for 8 batches of cookies is 44 cups. -/
theorem total_ingredients_for_batches : 
  (flour_per_batch + sugar_per_batch) * num_batches = 44 := by sorry

end NUMINAMATH_CALUDE_total_ingredients_for_batches_l108_10886


namespace NUMINAMATH_CALUDE_complex_equation_solutions_l108_10803

theorem complex_equation_solutions :
  ∃! (S : Finset ℂ), 
    (∀ z ∈ S, Complex.abs z < 15 ∧ Complex.exp (2 * z) = (z - 2) / (z + 2)) ∧
    Finset.card S = 9 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solutions_l108_10803


namespace NUMINAMATH_CALUDE_relationship_between_3a_3b_4a_l108_10850

theorem relationship_between_3a_3b_4a (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  3 * b < 3 * a ∧ 3 * a < 4 * a := by
  sorry

end NUMINAMATH_CALUDE_relationship_between_3a_3b_4a_l108_10850


namespace NUMINAMATH_CALUDE_rectangle_area_l108_10809

theorem rectangle_area (L B : ℝ) (h1 : L - B = 23) (h2 : 2 * (L + B) = 206) : L * B = 2520 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l108_10809


namespace NUMINAMATH_CALUDE_inverse_variation_problem_l108_10852

-- Define the relationship between x and y
def inverse_relation (x y : ℝ) : Prop :=
  ∃ k : ℝ, k > 0 ∧ y * x^2 = k

-- State the theorem
theorem inverse_variation_problem (x₁ x₂ y₁ y₂ : ℝ) 
  (h₁ : inverse_relation x₁ y₁)
  (h₂ : x₁ = 3)
  (h₃ : y₁ = 2)
  (h₄ : y₂ = 18)
  (h₅ : inverse_relation x₂ y₂) :
  x₂ = 1 := by
sorry

end NUMINAMATH_CALUDE_inverse_variation_problem_l108_10852


namespace NUMINAMATH_CALUDE_expression_evaluation_l108_10814

theorem expression_evaluation (x y z : ℝ) :
  (x + (y - z)) - ((x + z) - y) = 2*y - 2*z := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l108_10814


namespace NUMINAMATH_CALUDE_monkeys_required_for_new_bananas_l108_10880

/-- Represents the number of monkeys eating bananas -/
def num_monkeys : ℕ := 5

/-- Represents the number of bananas eaten in the initial scenario -/
def initial_bananas : ℕ := 5

/-- Represents the time taken to eat the initial number of bananas -/
def initial_time : ℕ := 5

/-- Represents the number of bananas to be eaten in the new scenario -/
def new_bananas : ℕ := 15

/-- Theorem stating that the number of monkeys required to eat the new number of bananas
    is equal to the initial number of monkeys -/
theorem monkeys_required_for_new_bananas :
  (num_monkeys : ℕ) = (num_monkeys : ℕ) := by sorry

end NUMINAMATH_CALUDE_monkeys_required_for_new_bananas_l108_10880


namespace NUMINAMATH_CALUDE_smallest_divisible_by_one_to_ten_l108_10826

theorem smallest_divisible_by_one_to_ten : 
  ∃ n : ℕ, (∀ k : ℕ, 1 ≤ k ∧ k ≤ 10 → k ∣ n) ∧ 
    (∀ m : ℕ, m < n → ∃ j : ℕ, 1 ≤ j ∧ j ≤ 10 ∧ ¬(j ∣ m)) ∧ 
    n = 2520 :=
by sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_one_to_ten_l108_10826


namespace NUMINAMATH_CALUDE_cyclist_heartbeats_l108_10884

/-- Calculates the total number of heartbeats for a cyclist during a race. -/
def total_heartbeats (heart_rate : ℕ) (race_distance : ℕ) (pace : ℕ) : ℕ :=
  heart_rate * race_distance * pace

/-- Proves that a cyclist's heart beats 16800 times during a 35-mile race. -/
theorem cyclist_heartbeats :
  let heart_rate := 120  -- heartbeats per minute
  let race_distance := 35  -- miles
  let pace := 4  -- minutes per mile
  total_heartbeats heart_rate race_distance pace = 16800 :=
by sorry

end NUMINAMATH_CALUDE_cyclist_heartbeats_l108_10884


namespace NUMINAMATH_CALUDE_proposition_not_hold_for_2_l108_10805

theorem proposition_not_hold_for_2 (P : ℕ → Prop)
  (h1 : ¬ P 3)
  (h2 : ∀ k : ℕ, k > 0 → P k → P (k + 1)) :
  ¬ P 2 := by
  sorry

end NUMINAMATH_CALUDE_proposition_not_hold_for_2_l108_10805


namespace NUMINAMATH_CALUDE_sin_cos_value_f_minus_cos_value_l108_10817

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (Real.sin x + Real.tan x) * Real.cos x / (1 + Real.cos (-x))

-- Theorem 1
theorem sin_cos_value (θ : ℝ) (h : f θ * Real.sin (π/6) - Real.cos θ = 0) :
  Real.sin θ * Real.cos θ = 2/5 := by sorry

-- Theorem 2
theorem f_minus_cos_value (θ : ℝ) (h1 : f θ * Real.cos θ = 1/8) (h2 : π/4 < θ ∧ θ < 3*π/4) :
  f (2019*π - θ) - Real.cos (2018*π - θ) = Real.sqrt 3 / 2 := by sorry

end NUMINAMATH_CALUDE_sin_cos_value_f_minus_cos_value_l108_10817


namespace NUMINAMATH_CALUDE_combined_miles_per_gallon_l108_10808

/-- The combined miles per gallon of two cars given their individual efficiencies and distance ratio -/
theorem combined_miles_per_gallon
  (sam_mpg : ℝ)
  (alex_mpg : ℝ)
  (distance_ratio : ℚ)
  (h_sam_mpg : sam_mpg = 50)
  (h_alex_mpg : alex_mpg = 20)
  (h_distance_ratio : distance_ratio = 2 / 3) :
  (2 * distance_ratio + 3) / (2 * distance_ratio / sam_mpg + 3 / alex_mpg) = 500 / 19 := by
  sorry

end NUMINAMATH_CALUDE_combined_miles_per_gallon_l108_10808


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l108_10862

/-- Given a geometric sequence {a_n}, if a_4 and a_8 are the roots of x^2 - 8x + 9 = 0, then a_6 = 3 -/
theorem geometric_sequence_problem (a : ℕ → ℝ) :
  (∀ n m : ℕ, a (n + m) = a n * a m) →  -- geometric sequence property
  (a 4 + a 8 = 8) →                    -- sum of roots
  (a 4 * a 8 = 9) →                    -- product of roots
  a 6 = 3 := by sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l108_10862


namespace NUMINAMATH_CALUDE_acute_angle_range_l108_10856

def a : ℝ × ℝ := (1, 2)
def b (x : ℝ) : ℝ × ℝ := (x, 4)

def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

def is_acute_angle (v w : ℝ × ℝ) : Prop := dot_product v w > 0

def not_same_direction (v w : ℝ × ℝ) : Prop := v.1 / v.2 ≠ w.1 / w.2

theorem acute_angle_range (x : ℝ) :
  is_acute_angle a (b x) ∧ not_same_direction a (b x) ↔ x ∈ Set.Ioo (-8) 2 ∪ Set.Ioi 2 :=
sorry

end NUMINAMATH_CALUDE_acute_angle_range_l108_10856


namespace NUMINAMATH_CALUDE_three_numbers_sum_divisible_by_three_l108_10843

def set_of_numbers : Finset ℕ := Finset.range 20

theorem three_numbers_sum_divisible_by_three (set_of_numbers : Finset ℕ) :
  (Finset.filter (fun s : Finset ℕ => s.card = 3 ∧ 
    (s.sum id) % 3 = 0 ∧ 
    s ⊆ set_of_numbers) (Finset.powerset set_of_numbers)).card = 384 := by
  sorry

end NUMINAMATH_CALUDE_three_numbers_sum_divisible_by_three_l108_10843


namespace NUMINAMATH_CALUDE_abc_equality_l108_10845

theorem abc_equality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : 4 * a * b * c * (a + b + c) = (a + b)^2 * (a + c)^2) :
  a * (a + b + c) = b * c := by
sorry

end NUMINAMATH_CALUDE_abc_equality_l108_10845


namespace NUMINAMATH_CALUDE_base_five_product_131_21_l108_10802

/-- Represents a number in base 5 --/
def BaseFive : Type := List Nat

/-- Converts a base 5 number to its decimal representation --/
def to_decimal (n : BaseFive) : Nat :=
  n.enum.foldr (fun (i, d) acc => acc + d * (5 ^ i)) 0

/-- Converts a decimal number to its base 5 representation --/
def to_base_five (n : Nat) : BaseFive :=
  sorry

/-- Multiplies two base 5 numbers --/
def base_five_mul (a b : BaseFive) : BaseFive :=
  to_base_five (to_decimal a * to_decimal b)

theorem base_five_product_131_21 :
  base_five_mul [1, 3, 1] [1, 2] = [1, 5, 2, 3] :=
sorry

end NUMINAMATH_CALUDE_base_five_product_131_21_l108_10802


namespace NUMINAMATH_CALUDE_inequality_proof_l108_10893

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (1 / a) + (4 / b) ≥ 9 / (a + b) := by sorry

end NUMINAMATH_CALUDE_inequality_proof_l108_10893


namespace NUMINAMATH_CALUDE_karens_class_size_l108_10849

def total_cookies : ℕ := 50
def kept_cookies : ℕ := 10
def grandparents_cookies : ℕ := 8
def cookies_per_classmate : ℕ := 2

theorem karens_class_size :
  (total_cookies - kept_cookies - grandparents_cookies) / cookies_per_classmate = 16 := by
  sorry

end NUMINAMATH_CALUDE_karens_class_size_l108_10849


namespace NUMINAMATH_CALUDE_investment_calculation_correct_l108_10839

/-- Calculates the total investment given share details and dividend income -/
def calculate_investment (face_value : ℚ) (quoted_price : ℚ) (dividend_rate : ℚ) (annual_income : ℚ) : ℚ :=
  let dividend_per_share := (dividend_rate / 100) * face_value
  let num_shares := annual_income / dividend_per_share
  num_shares * quoted_price

/-- Theorem stating that the investment calculation is correct for the given problem -/
theorem investment_calculation_correct :
  calculate_investment 10 8.25 12 648 = 4455 := by
  sorry

end NUMINAMATH_CALUDE_investment_calculation_correct_l108_10839


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_range_of_a_l108_10891

-- Define the function f(x) for part (1)
def f (x : ℝ) : ℝ := |x - 1| + |x + 1|

-- Define the function f(x) for part (2) with parameter a
def f_with_a (a : ℝ) (x : ℝ) : ℝ := |x - 1| + |x - a|

-- Part (1)
theorem solution_set_of_inequality (x : ℝ) :
  f x ≥ 3 ↔ x ≤ -1.5 ∨ x ≥ 1.5 := by sorry

-- Part (2)
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, f_with_a a x ≥ 2) → (a = -1 ∨ a = 3) := by sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_range_of_a_l108_10891


namespace NUMINAMATH_CALUDE_power_of_81_l108_10858

theorem power_of_81 : (81 : ℝ) ^ (5/4 : ℝ) = 243 := by sorry

end NUMINAMATH_CALUDE_power_of_81_l108_10858


namespace NUMINAMATH_CALUDE_hexagonal_field_fencing_cost_l108_10894

/-- Represents the cost of fencing for a single side of the hexagonal field -/
structure SideCost where
  length : ℝ
  costPerMeter : ℝ

/-- Calculates the total cost of fencing for an irregular hexagonal field -/
def totalFencingCost (sides : List SideCost) : ℝ :=
  sides.foldl (fun acc side => acc + side.length * side.costPerMeter) 0

/-- Theorem stating that the total cost of fencing for the given hexagonal field is 289 rs. -/
theorem hexagonal_field_fencing_cost :
  let sides : List SideCost := [
    ⟨10, 3⟩, ⟨20, 2⟩, ⟨15, 4⟩, ⟨18, 3.5⟩, ⟨12, 2.5⟩, ⟨22, 3⟩
  ]
  totalFencingCost sides = 289 := by
  sorry


end NUMINAMATH_CALUDE_hexagonal_field_fencing_cost_l108_10894


namespace NUMINAMATH_CALUDE_smallest_m_is_16_l108_10844

/-- The set T of complex numbers -/
def T : Set ℂ :=
  {z : ℂ | ∃ (u v : ℝ), z = u + v * Complex.I ∧ Real.sqrt 3 / 3 ≤ u ∧ u ≤ Real.sqrt 3 / 2}

/-- The property P(n) that should hold for all n ≥ m -/
def P (n : ℕ) : Prop :=
  ∃ z ∈ T, z ^ (2 * n) = 1

/-- The theorem stating that 16 is the smallest positive integer m satisfying the condition -/
theorem smallest_m_is_16 :
  (∀ n ≥ 16, P n) ∧ ∀ m < 16, ¬(∀ n ≥ m, P n) :=
sorry

end NUMINAMATH_CALUDE_smallest_m_is_16_l108_10844


namespace NUMINAMATH_CALUDE_inequality_and_equality_condition_l108_10867

theorem inequality_and_equality_condition (a b c d : ℝ) :
  (a^2 + b^2) * (c^2 + d^2) ≥ (a*c + b*d)^2 ∧
  ((a^2 + b^2) * (c^2 + d^2) = (a*c + b*d)^2 ↔ a*d = b*c) := by
  sorry

end NUMINAMATH_CALUDE_inequality_and_equality_condition_l108_10867


namespace NUMINAMATH_CALUDE_num_triangles_on_circle_l108_10820

/-- The number of ways to choose k items from n items. -/
def binomial (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of points on the circle. -/
def numPoints : ℕ := 10

/-- The number of points needed to form a triangle. -/
def pointsPerTriangle : ℕ := 3

/-- Theorem: The number of triangles that can be formed from 10 points on a circle is 120. -/
theorem num_triangles_on_circle :
  binomial numPoints pointsPerTriangle = 120 := by
  sorry

end NUMINAMATH_CALUDE_num_triangles_on_circle_l108_10820


namespace NUMINAMATH_CALUDE_smallest_prime_twelve_less_square_l108_10822

theorem smallest_prime_twelve_less_square : 
  ∃ (n : ℕ), 
    (n > 0) ∧ 
    (Nat.Prime n) ∧ 
    (∃ (m : ℕ), n = m^2 - 12) ∧
    (∀ (k : ℕ), k > 0 → Nat.Prime k → (∃ (l : ℕ), k = l^2 - 12) → k ≥ n) ∧
    n = 13 :=
by sorry

end NUMINAMATH_CALUDE_smallest_prime_twelve_less_square_l108_10822


namespace NUMINAMATH_CALUDE_dhoni_leftover_percentage_l108_10870

/-- Represents the percentage of Dhoni's earnings spent on rent -/
def rent_percentage : ℝ := 20

/-- Represents the difference in percentage between rent and dishwasher expenses -/
def dishwasher_difference : ℝ := 5

/-- Calculates the percentage of earnings spent on the dishwasher -/
def dishwasher_percentage : ℝ := rent_percentage - dishwasher_difference

/-- Calculates the total percentage of earnings spent -/
def total_spent_percentage : ℝ := rent_percentage + dishwasher_percentage

/-- Represents the total percentage (100%) -/
def total_percentage : ℝ := 100

/-- Theorem: The percentage of Dhoni's earning left over is 65% -/
theorem dhoni_leftover_percentage : 
  total_percentage - total_spent_percentage = 65 := by
  sorry

end NUMINAMATH_CALUDE_dhoni_leftover_percentage_l108_10870


namespace NUMINAMATH_CALUDE_linear_equation_solution_l108_10883

theorem linear_equation_solution (x y : ℝ) :
  4 * x - 5 * y = 9 → y = (4 * x - 9) / 5 := by
  sorry

end NUMINAMATH_CALUDE_linear_equation_solution_l108_10883


namespace NUMINAMATH_CALUDE_shaded_cubes_count_l108_10887

/-- Represents a cube made up of smaller cubes --/
structure Cube where
  size : Nat
  shaded_corners : Bool
  shaded_center : Bool

/-- Counts the number of smaller cubes with at least one face shaded --/
def count_shaded_cubes (c : Cube) : Nat :=
  sorry

/-- Theorem stating that a 4x4x4 cube with shaded corners and centers has 14 shaded cubes --/
theorem shaded_cubes_count (c : Cube) :
  c.size = 4 ∧ c.shaded_corners ∧ c.shaded_center →
  count_shaded_cubes c = 14 :=
by sorry

end NUMINAMATH_CALUDE_shaded_cubes_count_l108_10887


namespace NUMINAMATH_CALUDE_function_problem_l108_10816

theorem function_problem (f : ℝ → ℝ) (a b c : ℝ) 
  (h_inv : Function.Injective f)
  (h1 : f a = b)
  (h2 : f b = 5)
  (h3 : f c = 3)
  (h4 : c = a + 1) :
  a - b = -2 := by
  sorry

end NUMINAMATH_CALUDE_function_problem_l108_10816


namespace NUMINAMATH_CALUDE_vector_sum_proof_l108_10830

/-- Given two 2D vectors a and b, prove that 3a + 4b equals the specified result -/
theorem vector_sum_proof (a b : ℝ × ℝ) : 
  a = (2, 1) → b = (-3, 4) → (3 • a + 4 • b : ℝ × ℝ) = (-6, 19) := by
  sorry

end NUMINAMATH_CALUDE_vector_sum_proof_l108_10830


namespace NUMINAMATH_CALUDE_math_club_team_selection_l108_10869

theorem math_club_team_selection (total_boys : ℕ) (total_girls : ℕ) 
  (team_boys : ℕ) (team_girls : ℕ) : 
  total_boys = 7 → 
  total_girls = 9 → 
  team_boys = 4 → 
  team_girls = 3 → 
  (team_boys + team_girls : ℕ) = 7 → 
  (Nat.choose total_boys team_boys) * (Nat.choose total_girls team_girls) = 2940 := by
  sorry

end NUMINAMATH_CALUDE_math_club_team_selection_l108_10869


namespace NUMINAMATH_CALUDE_total_pizza_weight_l108_10832

/-- Represents the weight of a pizza with toppings -/
structure Pizza where
  base : Nat
  toppings : List Nat

/-- Calculates the total weight of a pizza -/
def totalWeight (p : Pizza) : Nat :=
  p.base + p.toppings.sum

/-- Rachel's pizza -/
def rachelPizza : Pizza :=
  { base := 400
  , toppings := [100, 50, 60] }

/-- Bella's pizza -/
def bellaPizza : Pizza :=
  { base := 350
  , toppings := [75, 55, 35] }

/-- Theorem: The total weight of Rachel's and Bella's pizzas is 1125 grams -/
theorem total_pizza_weight :
  totalWeight rachelPizza + totalWeight bellaPizza = 1125 := by
  sorry

end NUMINAMATH_CALUDE_total_pizza_weight_l108_10832


namespace NUMINAMATH_CALUDE_largest_number_l108_10821

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def digit_sum (n : ℕ) : ℕ := 
  if n < 10 then n else (n % 10) + digit_sum (n / 10)

def is_square_of_prime (n : ℕ) : Prop := 
  ∃ p : ℕ, is_prime p ∧ n = p * p

theorem largest_number (P Q R S T : ℕ) : 
  (2 ≤ P ∧ P ≤ 19) →
  (2 ≤ Q ∧ Q ≤ 19) →
  (2 ≤ R ∧ R ≤ 19) →
  (2 ≤ S ∧ S ≤ 19) →
  (2 ≤ T ∧ T ≤ 19) →
  P ≠ Q ∧ P ≠ R ∧ P ≠ S ∧ P ≠ T ∧ Q ≠ R ∧ Q ≠ S ∧ Q ≠ T ∧ R ≠ S ∧ R ≠ T ∧ S ≠ T →
  (P ≥ 10 ∧ P < 100 ∧ is_prime P ∧ is_prime (digit_sum P)) →
  (∃ k : ℕ, Q = 5 * k) →
  (R % 2 = 1 ∧ ¬is_prime R) →
  is_square_of_prime S →
  (is_prime T ∧ T = (P + Q) / 2) →
  Q ≥ P ∧ Q ≥ R ∧ Q ≥ S ∧ Q ≥ T :=
by sorry

end NUMINAMATH_CALUDE_largest_number_l108_10821


namespace NUMINAMATH_CALUDE_sqrt_decimal_expansion_unique_l108_10834

theorem sqrt_decimal_expansion_unique 
  (p n : ℚ) 
  (hp : 0 < p) 
  (hn : 0 < n) 
  (hp_not_square : ¬ ∃ (m : ℚ), p = m ^ 2) 
  (hn_not_square : ¬ ∃ (m : ℚ), n = m ^ 2) : 
  ¬ ∃ (k : ℤ), Real.sqrt p - Real.sqrt n = k := by
  sorry

end NUMINAMATH_CALUDE_sqrt_decimal_expansion_unique_l108_10834


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_distance_l108_10866

theorem isosceles_right_triangle_distance (a : ℝ) (h : a = 8) :
  Real.sqrt (a^2 + a^2) = a * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_isosceles_right_triangle_distance_l108_10866


namespace NUMINAMATH_CALUDE_product_equals_fraction_l108_10828

/-- The repeating decimal 0.1357̄ as a rational number -/
def s : ℚ := 1357 / 9999

/-- The product of 0.1357̄ and 7 -/
def product : ℚ := 7 * s

theorem product_equals_fraction : product = 9499 / 9999 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_fraction_l108_10828


namespace NUMINAMATH_CALUDE_ark5_ensures_metabolic_energy_needs_l108_10800

-- Define the Ark5 enzyme
def Ark5 : Type := Unit

-- Define cancer cells
def CancerCell : Type := Unit

-- Define the function that represents the ability to balance energy
def balanceEnergy (a : Ark5) (c : CancerCell) : Prop := sorry

-- Define the function that represents the ability to proliferate without limit
def proliferateWithoutLimit (c : CancerCell) : Prop := sorry

-- Define the function that represents the state of energy scarcity
def energyScarcity : Prop := sorry

-- Define the function that represents cell death due to lack of energy
def dieFromLackOfEnergy (c : CancerCell) : Prop := sorry

-- Define the function that represents ensuring metabolic energy needs
def ensureMetabolicEnergyNeeds (a : Ark5) (c : CancerCell) : Prop := sorry

-- Theorem statement
theorem ark5_ensures_metabolic_energy_needs :
  ∀ (a : Ark5) (c : CancerCell),
    (¬balanceEnergy a c → (energyScarcity → proliferateWithoutLimit c)) ∧
    (¬balanceEnergy a c → (energyScarcity → dieFromLackOfEnergy c)) →
    ensureMetabolicEnergyNeeds a c :=
by
  sorry

end NUMINAMATH_CALUDE_ark5_ensures_metabolic_energy_needs_l108_10800


namespace NUMINAMATH_CALUDE_sanitizer_sales_theorem_l108_10868

/-- Represents the hand sanitizer sales problem -/
structure SanitizerSales where
  cost : ℝ  -- Cost per bottle in yuan
  initial_price : ℝ  -- Initial selling price per bottle in yuan
  initial_volume : ℝ  -- Initial daily sales volume
  price_sensitivity : ℝ  -- Decrease in sales for every 1 yuan increase in price
  x : ℝ  -- Increase in selling price

/-- Calculates the daily sales volume given the price increase -/
def daily_volume (s : SanitizerSales) : ℝ :=
  s.initial_volume - s.price_sensitivity * s.x

/-- Calculates the profit per bottle given the price increase -/
def profit_per_bottle (s : SanitizerSales) : ℝ :=
  (s.initial_price - s.cost) + s.x

/-- Calculates the daily profit given the price increase -/
def daily_profit (s : SanitizerSales) : ℝ :=
  (daily_volume s) * (profit_per_bottle s)

/-- The main theorem about the sanitizer sales problem -/
theorem sanitizer_sales_theorem (s : SanitizerSales) 
  (h1 : s.cost = 16)
  (h2 : s.initial_price = 20)
  (h3 : s.initial_volume = 60)
  (h4 : s.price_sensitivity = 5) :
  (daily_volume s = 60 - 5 * s.x) ∧
  (profit_per_bottle s = 4 + s.x) ∧
  (daily_profit s = 300 → s.x = 2 ∨ s.x = 6) ∧
  (∃ (max_profit : ℝ), max_profit = 320 ∧ 
    ∀ (y : ℝ), y = daily_profit s → y ≤ max_profit ∧
    (y = max_profit ↔ s.x = 4)) := by
  sorry


end NUMINAMATH_CALUDE_sanitizer_sales_theorem_l108_10868


namespace NUMINAMATH_CALUDE_product_of_slopes_l108_10823

theorem product_of_slopes (m n : ℝ) : 
  (∃ θ₁ θ₂ : ℝ, θ₁ = 3 * θ₂ ∧ m = Real.tan θ₁ ∧ n = Real.tan θ₂) →  -- L1 makes three times the angle with horizontal as L2
  m = 3 * n →                                                      -- L1 has 3 times the slope of L2
  m ≠ 0 →                                                          -- L1 is not vertical
  m * n = 0 :=                                                     -- Conclusion: mn = 0
by sorry

end NUMINAMATH_CALUDE_product_of_slopes_l108_10823


namespace NUMINAMATH_CALUDE_sin_squared_alpha_plus_5pi_12_l108_10831

theorem sin_squared_alpha_plus_5pi_12 (α : ℝ) 
  (h : Real.sin (2 * Real.pi / 3 - 2 * α) = 3 / 5) : 
  Real.sin (α + 5 * Real.pi / 12) ^ 2 = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_sin_squared_alpha_plus_5pi_12_l108_10831


namespace NUMINAMATH_CALUDE_hypotenuse_length_squared_l108_10896

/-- Given complex numbers p, q, and r that are zeros of a polynomial Q(z) = z^3 + sz + t,
    if |p|^2 + |q|^2 + |r|^2 = 300, p + q + r = 0, and p, q, and r form a right triangle
    in the complex plane, then the square of the length of the hypotenuse of this triangle is 450. -/
theorem hypotenuse_length_squared (p q r s t : ℂ) : 
  (Q : ℂ → ℂ) = (fun z ↦ z^3 + s*z + t) →
  p^3 + s*p + t = 0 →
  q^3 + s*q + t = 0 →
  r^3 + s*r + t = 0 →
  Complex.abs p^2 + Complex.abs q^2 + Complex.abs r^2 = 300 →
  p + q + r = 0 →
  ∃ (a b : ℝ), Complex.abs (p - q)^2 = a^2 ∧ Complex.abs (q - r)^2 = b^2 ∧ Complex.abs (p - r)^2 = a^2 + b^2 →
  Complex.abs (p - r)^2 = 450 :=
by sorry

end NUMINAMATH_CALUDE_hypotenuse_length_squared_l108_10896


namespace NUMINAMATH_CALUDE_simplify_expression_l108_10829

theorem simplify_expression (x y z : ℝ) :
  (15 * x + 45 * y - 30 * z) + (20 * x - 10 * y + 5 * z) - (5 * x + 35 * y - 15 * z) = 30 * x - 10 * z := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l108_10829


namespace NUMINAMATH_CALUDE_cup_purchase_properties_prize_purchase_properties_l108_10836

/-- Represents the cost and quantity of insulated cups --/
structure CupPurchase where
  cost_a : ℕ  -- Cost of A type cup
  cost_b : ℕ  -- Cost of B type cup
  quantity_a : ℕ  -- Quantity of A type cups
  quantity_b : ℕ  -- Quantity of B type cups

/-- Theorem stating the properties of the cup purchase --/
theorem cup_purchase_properties :
  ∃ (purchase : CupPurchase),
    -- B type cup costs 10 yuan more than A type cup
    purchase.cost_b = purchase.cost_a + 10 ∧
    -- 1200 yuan buys 1.5 times as many A cups as 1000 yuan buys B cups
    1200 / purchase.cost_a = (3/2) * (1000 / purchase.cost_b) ∧
    -- Company buys 9 fewer B cups than A cups
    purchase.quantity_b = purchase.quantity_a - 9 ∧
    -- Number of A cups is not less than 38
    purchase.quantity_a ≥ 38 ∧
    -- Total cost does not exceed 3150 yuan
    purchase.cost_a * purchase.quantity_a + purchase.cost_b * purchase.quantity_b ≤ 3150 ∧
    -- Cost of A type cup is 40 yuan
    purchase.cost_a = 40 ∧
    -- Cost of B type cup is 50 yuan
    purchase.cost_b = 50 ∧
    -- There are exactly three valid purchasing schemes
    (∃ (scheme1 scheme2 scheme3 : CupPurchase),
      scheme1.quantity_a = 38 ∧ scheme1.quantity_b = 29 ∧
      scheme2.quantity_a = 39 ∧ scheme2.quantity_b = 30 ∧
      scheme3.quantity_a = 40 ∧ scheme3.quantity_b = 31 ∧
      ∀ (other : CupPurchase),
        (other.quantity_a ≥ 38 ∧
         other.quantity_b = other.quantity_a - 9 ∧
         other.cost_a * other.quantity_a + other.cost_b * other.quantity_b ≤ 3150) →
        (other = scheme1 ∨ other = scheme2 ∨ other = scheme3)) :=
by
  sorry

/-- Represents the quantity of prizes --/
structure PrizePurchase where
  quantity_a : ℕ  -- Quantity of A type prizes
  quantity_b : ℕ  -- Quantity of B type prizes

/-- Theorem stating the properties of the prize purchase --/
theorem prize_purchase_properties :
  ∃ (prize : PrizePurchase),
    -- A type prize costs 270 yuan
    -- B type prize costs 240 yuan
    -- Total cost of prizes equals minimum cost from part 2 (2970 yuan)
    270 * prize.quantity_a + 240 * prize.quantity_b = 2970 ∧
    -- There are 3 A type prizes and 9 B type prizes
    prize.quantity_a = 3 ∧
    prize.quantity_b = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_cup_purchase_properties_prize_purchase_properties_l108_10836


namespace NUMINAMATH_CALUDE_pistachio_stairs_l108_10846

/-- The number of steps between each floor -/
def steps_per_floor : ℕ := 20

/-- The floor where Pistachio lives -/
def target_floor : ℕ := 11

/-- The starting floor -/
def start_floor : ℕ := 1

/-- The total number of steps to reach the target floor -/
def total_steps : ℕ := (target_floor - start_floor) * steps_per_floor

theorem pistachio_stairs : total_steps = 200 := by
  sorry

end NUMINAMATH_CALUDE_pistachio_stairs_l108_10846


namespace NUMINAMATH_CALUDE_base5_2202_equals_base10_302_l108_10825

/-- Converts a base 5 digit to its base 10 equivalent --/
def base5ToBase10 (digit : Nat) (position : Nat) : Nat :=
  digit * (5 ^ position)

/-- Theorem: The base 5 number 2202₅ is equal to the base 10 number 302 --/
theorem base5_2202_equals_base10_302 :
  base5ToBase10 2 3 + base5ToBase10 2 2 + base5ToBase10 0 1 + base5ToBase10 2 0 = 302 := by
  sorry

end NUMINAMATH_CALUDE_base5_2202_equals_base10_302_l108_10825


namespace NUMINAMATH_CALUDE_map_scale_conversion_l108_10842

/-- Given a map where 15 cm represents 90 km, a 20 cm length represents 120,000 meters -/
theorem map_scale_conversion (map_scale : ℝ) (h : map_scale * 15 = 90) : 
  map_scale * 20 * 1000 = 120000 := by
  sorry

end NUMINAMATH_CALUDE_map_scale_conversion_l108_10842


namespace NUMINAMATH_CALUDE_best_fit_model_l108_10853

/-- Represents a regression model with its coefficient of determination -/
structure RegressionModel where
  r_squared : ℝ
  h_r_squared_nonneg : 0 ≤ r_squared
  h_r_squared_le_one : r_squared ≤ 1

/-- Determines if a model has better fit than another based on R² values -/
def better_fit (model1 model2 : RegressionModel) : Prop :=
  model1.r_squared > model2.r_squared

theorem best_fit_model (model1 model2 model3 model4 : RegressionModel)
  (h1 : model1.r_squared = 0.05)
  (h2 : model2.r_squared = 0.49)
  (h3 : model3.r_squared = 0.89)
  (h4 : model4.r_squared = 0.98) :
  better_fit model4 model1 ∧ better_fit model4 model2 ∧ better_fit model4 model3 :=
sorry

end NUMINAMATH_CALUDE_best_fit_model_l108_10853


namespace NUMINAMATH_CALUDE_roots_have_unit_modulus_l108_10859

theorem roots_have_unit_modulus (z : ℂ) : 
  11 * z^10 + 10 * Complex.I * z^9 + 10 * Complex.I * z - 11 = 0 → Complex.abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_roots_have_unit_modulus_l108_10859


namespace NUMINAMATH_CALUDE_equation_solution_l108_10877

theorem equation_solution : 
  ∃ y : ℚ, (1 : ℚ) / 3 + 1 / y = (4 : ℚ) / 5 ↔ y = 15 / 7 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l108_10877


namespace NUMINAMATH_CALUDE_quadratic_roots_average_l108_10811

theorem quadratic_roots_average (c : ℝ) : 
  ∃ (x₁ x₂ : ℝ), (2 * x₁^2 - 4 * x₁ + c = 0) ∧ 
                 (2 * x₂^2 - 4 * x₂ + c = 0) ∧ 
                 (x₁ ≠ x₂) → 
                 (x₁ + x₂) / 2 = 1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_average_l108_10811


namespace NUMINAMATH_CALUDE_escalator_ride_time_l108_10878

/-- Represents the time it takes Clea to descend the escalator under different conditions -/
structure EscalatorTime where
  stationary : ℝ  -- Time to walk down stationary escalator
  moving : ℝ      -- Time to walk down moving escalator
  riding : ℝ      -- Time to ride down without walking

/-- The theorem states that given the times for walking down stationary and moving escalators,
    the time to ride without walking can be determined -/
theorem escalator_ride_time (et : EscalatorTime) 
  (h1 : et.stationary = 75) 
  (h2 : et.moving = 30) : 
  et.riding = 50 := by
  sorry

end NUMINAMATH_CALUDE_escalator_ride_time_l108_10878


namespace NUMINAMATH_CALUDE_equal_area_division_l108_10824

/-- Represents a shape on a grid -/
structure GridShape where
  area : ℝ
  mk_area_pos : area > 0

/-- Represents a line on a grid -/
structure GridLine where
  distance_from_origin : ℝ

/-- Represents the division of a shape by a line -/
def divides_equally (s : GridShape) (l : GridLine) : Prop :=
  ∃ (area1 area2 : ℝ), 
    area1 > 0 ∧ 
    area2 > 0 ∧ 
    area1 = area2 ∧ 
    area1 + area2 = s.area

/-- The main theorem -/
theorem equal_area_division 
  (gray_shape : GridShape) 
  (h_area : gray_shape.area = 10) 
  (mo : GridLine) 
  (parallel_line : GridLine) 
  (h_distance : parallel_line.distance_from_origin = mo.distance_from_origin + 2.6) :
  divides_equally gray_shape parallel_line := by
  sorry

end NUMINAMATH_CALUDE_equal_area_division_l108_10824


namespace NUMINAMATH_CALUDE_inequality_solution_l108_10863

theorem inequality_solution (x : ℝ) : (x + 3) * (x - 2) < 0 ↔ -3 < x ∧ x < 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l108_10863


namespace NUMINAMATH_CALUDE_min_book_cover_area_l108_10898

/-- Given a book cover with reported dimensions of 5 inches by 7 inches,
    where each dimension can vary by ±0.5 inches, the minimum possible area
    of the book cover is 29.25 square inches. -/
theorem min_book_cover_area (reported_length : ℝ) (reported_width : ℝ)
    (actual_length : ℝ) (actual_width : ℝ) :
  reported_length = 5 →
  reported_width = 7 →
  abs (actual_length - reported_length) ≤ 0.5 →
  abs (actual_width - reported_width) ≤ 0.5 →
  ∀ area : ℝ, area = actual_length * actual_width →
    area ≥ 29.25 :=
by sorry

end NUMINAMATH_CALUDE_min_book_cover_area_l108_10898


namespace NUMINAMATH_CALUDE_andy_final_position_l108_10848

/-- Represents the position of Andy the Ant -/
structure Position :=
  (x : Int) (y : Int)

/-- Represents the direction Andy is facing -/
inductive Direction
  | North
  | East
  | South
  | West

/-- Represents Andy's state at any given moment -/
structure AntState :=
  (pos : Position)
  (dir : Direction)
  (moveCount : Nat)

/-- The movement function for Andy -/
def move (state : AntState) : AntState :=
  sorry

/-- The main theorem stating Andy's final position -/
theorem andy_final_position :
  let initialState : AntState :=
    { pos := { x := 30, y := -30 }
    , dir := Direction.North
    , moveCount := 0
    }
  let finalState := (move^[3030]) initialState
  finalState.pos = { x := 4573, y := -1546 } :=
sorry

end NUMINAMATH_CALUDE_andy_final_position_l108_10848


namespace NUMINAMATH_CALUDE_greatest_n_with_divisibility_conditions_l108_10818

theorem greatest_n_with_divisibility_conditions :
  ∃ (n : ℕ), n < 1000 ∧
  (Int.floor (Real.sqrt n) - 2 : ℤ) ∣ (n - 4 : ℤ) ∧
  (Int.floor (Real.sqrt n) + 2 : ℤ) ∣ (n + 4 : ℤ) ∧
  (∀ (m : ℕ), m < 1000 →
    (Int.floor (Real.sqrt m) - 2 : ℤ) ∣ (m - 4 : ℤ) →
    (Int.floor (Real.sqrt m) + 2 : ℤ) ∣ (m + 4 : ℤ) →
    m ≤ n) ∧
  n = 956 :=
sorry

end NUMINAMATH_CALUDE_greatest_n_with_divisibility_conditions_l108_10818


namespace NUMINAMATH_CALUDE_append_two_to_three_digit_number_l108_10854

/-- Given a three-digit number with digits h, t, and u, appending 2 results in 1000h + 100t + 10u + 2 -/
theorem append_two_to_three_digit_number (h t u : ℕ) :
  let original := 100 * h + 10 * t + u
  let appended := original * 10 + 2
  appended = 1000 * h + 100 * t + 10 * u + 2 := by
sorry

end NUMINAMATH_CALUDE_append_two_to_three_digit_number_l108_10854


namespace NUMINAMATH_CALUDE_solution_replacement_fraction_l108_10835

theorem solution_replacement_fraction (initial_conc : ℚ) (replacement_conc : ℚ) (final_conc : ℚ)
  (h_initial : initial_conc = 60 / 100)
  (h_replacement : replacement_conc = 25 / 100)
  (h_final : final_conc = 35 / 100) :
  let replaced_fraction := (initial_conc - final_conc) / (initial_conc - replacement_conc)
  replaced_fraction = 5 / 7 := by
sorry

end NUMINAMATH_CALUDE_solution_replacement_fraction_l108_10835


namespace NUMINAMATH_CALUDE_simplify_expression_l108_10838

theorem simplify_expression (x : ℝ) (h : x^2 ≠ 1) :
  Real.sqrt (1 + ((x^4 + 1) / (2 * x^2))^2) = (Real.sqrt (x^8 + 6 * x^4 + 1)) / (2 * x^2) :=
by sorry

end NUMINAMATH_CALUDE_simplify_expression_l108_10838


namespace NUMINAMATH_CALUDE_fraction_simplification_l108_10851

theorem fraction_simplification (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (a^2 * b^2) / (b/a)^2 = a^4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l108_10851


namespace NUMINAMATH_CALUDE_divisor_function_ratio_l108_10879

/-- τ(n) denotes the number of positive divisors of n -/
def τ (n : ℕ+) : ℕ := sorry

theorem divisor_function_ratio (n : ℕ+) (h : τ (n^2) / τ n = 3) : 
  τ (n^7) / τ n = 29 := by sorry

end NUMINAMATH_CALUDE_divisor_function_ratio_l108_10879


namespace NUMINAMATH_CALUDE_smallest_qnnn_l108_10875

def is_two_digit_with_equal_digits (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99 ∧ n % 11 = 0

def is_one_digit (n : ℕ) : Prop :=
  1 ≤ n ∧ n ≤ 9

def satisfies_condition (nn : ℕ) (n : ℕ) (qnnn : ℕ) : Prop :=
  is_two_digit_with_equal_digits nn ∧
  is_one_digit n ∧
  nn * n = qnnn ∧
  1000 ≤ qnnn ∧ qnnn ≤ 9999 ∧
  qnnn % 1000 % 100 % 10 = n ∧
  qnnn % 1000 % 100 / 10 = n ∧
  qnnn % 1000 / 100 = n

theorem smallest_qnnn :
  ∀ qnnn : ℕ, (∃ nn n : ℕ, satisfies_condition nn n qnnn) →
  2555 ≤ qnnn :=
sorry

end NUMINAMATH_CALUDE_smallest_qnnn_l108_10875


namespace NUMINAMATH_CALUDE_chores_repayment_l108_10815

/-- Calculates the amount earned for a given hour in the chore cycle -/
def hourly_rate (hour : ℕ) : ℕ :=
  match hour % 3 with
  | 1 => 2
  | 2 => 4
  | 0 => 6
  | _ => 0 -- This case should never occur, but Lean requires it for completeness

/-- Calculates the total amount earned for a given number of hours -/
def total_earned (hours : ℕ) : ℕ :=
  (List.range hours).map hourly_rate |>.sum

/-- The main theorem stating that 45 hours of chores results in $180 earned -/
theorem chores_repayment : total_earned 45 = 180 := by
  sorry

end NUMINAMATH_CALUDE_chores_repayment_l108_10815


namespace NUMINAMATH_CALUDE_smallest_addend_for_divisibility_l108_10833

theorem smallest_addend_for_divisibility (a b : ℕ) (ha : a = 87908235) (hb : b = 12587) :
  let x := (b - (a % b)) % b
  (a + x) % b = 0 ∧ ∀ y : ℕ, y < x → (a + y) % b ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_smallest_addend_for_divisibility_l108_10833


namespace NUMINAMATH_CALUDE_contrapositive_square_inequality_l108_10890

theorem contrapositive_square_inequality (x y : ℝ) :
  (¬(x^2 > y^2) → ¬(x > y)) ↔ (x^2 ≤ y^2 → x ≤ y) := by sorry

end NUMINAMATH_CALUDE_contrapositive_square_inequality_l108_10890


namespace NUMINAMATH_CALUDE_find_T_l108_10841

theorem find_T (S : ℚ) (T : ℚ) (h1 : (1/4) * (1/6) * T = (1/2) * (1/8) * S) (h2 : S = 64) :
  T = 96 := by
  sorry

end NUMINAMATH_CALUDE_find_T_l108_10841


namespace NUMINAMATH_CALUDE_trip_duration_is_101_l108_10847

/-- Calculates the total trip duration for Jill's journey to the library --/
def total_trip_duration (first_bus_wait : ℕ) (first_bus_ride : ℕ) (first_bus_delay : ℕ)
                        (walk_time : ℕ) (train_wait : ℕ) (train_ride : ℕ) (train_delay : ℕ)
                        (second_bus_wait_A : ℕ) (second_bus_ride_A : ℕ)
                        (second_bus_wait_B : ℕ) (second_bus_ride_B : ℕ) : ℕ :=
  let first_bus_total := first_bus_wait + first_bus_ride + first_bus_delay
  let train_total := walk_time + train_wait + train_ride + train_delay
  let second_bus_total := if second_bus_ride_A < second_bus_ride_B
                          then (second_bus_wait_A + second_bus_ride_A) / 2
                          else (second_bus_wait_B + second_bus_ride_B) / 2
  first_bus_total + train_total + second_bus_total

/-- Theorem stating that the total trip duration is 101 minutes --/
theorem trip_duration_is_101 :
  total_trip_duration 12 30 5 10 8 20 3 15 10 20 6 = 101 := by
  sorry

end NUMINAMATH_CALUDE_trip_duration_is_101_l108_10847


namespace NUMINAMATH_CALUDE_initial_machines_count_l108_10865

/-- The number of bottles produced per minute by the initial number of machines -/
def initial_production_rate : ℕ := 270

/-- The number of machines used in the second scenario -/
def second_scenario_machines : ℕ := 20

/-- The number of bottles produced in the second scenario -/
def second_scenario_production : ℕ := 3600

/-- The time in minutes for the second scenario -/
def second_scenario_time : ℕ := 4

/-- The number of machines running initially -/
def initial_machines : ℕ := 6

theorem initial_machines_count :
  initial_machines * initial_production_rate = second_scenario_machines * (second_scenario_production / second_scenario_time) :=
by sorry

end NUMINAMATH_CALUDE_initial_machines_count_l108_10865


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l108_10827

/-- The imaginary part of (1+2i) / (1-i)² is 1/2 -/
theorem imaginary_part_of_z : Complex.im ((1 + 2*Complex.I) / (1 - Complex.I)^2) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l108_10827


namespace NUMINAMATH_CALUDE_consecutive_integers_product_plus_one_is_square_l108_10840

theorem consecutive_integers_product_plus_one_is_square (n : ℤ) :
  n * (n + 1) * (n + 2) * (n + 3) + 1 = (n^2 + 3*n + 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_product_plus_one_is_square_l108_10840


namespace NUMINAMATH_CALUDE_car_travel_time_l108_10872

/-- Proves that a car with given specifications traveling for a certain time uses the specified amount of fuel -/
theorem car_travel_time (speed : ℝ) (fuel_efficiency : ℝ) (tank_capacity : ℝ) (fuel_used_ratio : ℝ) : 
  speed = 40 →
  fuel_efficiency = 40 →
  tank_capacity = 12 →
  fuel_used_ratio = 0.4166666666666667 →
  (fuel_used_ratio * tank_capacity * fuel_efficiency) / speed = 5 := by
  sorry

end NUMINAMATH_CALUDE_car_travel_time_l108_10872


namespace NUMINAMATH_CALUDE_fraction_meaningful_iff_not_equal_two_l108_10897

theorem fraction_meaningful_iff_not_equal_two (x : ℝ) : 
  (∃ y : ℝ, y = 7 / (x - 2)) ↔ x ≠ 2 := by
sorry

end NUMINAMATH_CALUDE_fraction_meaningful_iff_not_equal_two_l108_10897


namespace NUMINAMATH_CALUDE_divisor_sum_relation_l108_10889

theorem divisor_sum_relation (n f g : ℕ) : 
  n > 1 → 
  (∃ d1 d2 : ℕ, d1 ∣ n ∧ d2 ∣ n ∧ d1 ≤ d2 ∧ ∀ d : ℕ, d ∣ n → d = d1 ∨ d ≥ d2 → f = d1 + d2) →
  (∃ d3 d4 : ℕ, d3 ∣ n ∧ d4 ∣ n ∧ d3 ≥ d4 ∧ ∀ d : ℕ, d ∣ n → d = d3 ∨ d ≤ d4 → g = d3 + d4) →
  n = (g * (f - 1)) / f :=
sorry

end NUMINAMATH_CALUDE_divisor_sum_relation_l108_10889


namespace NUMINAMATH_CALUDE_maxwell_brad_meeting_time_l108_10871

/-- Proves that Maxwell walks for 2 hours before meeting Brad -/
theorem maxwell_brad_meeting_time
  (distance : ℝ)
  (maxwell_speed : ℝ)
  (brad_speed : ℝ)
  (head_start : ℝ)
  (h_distance : distance = 14)
  (h_maxwell_speed : maxwell_speed = 4)
  (h_brad_speed : brad_speed = 6)
  (h_head_start : head_start = 1) :
  ∃ (t : ℝ), t + head_start = 2 ∧ maxwell_speed * (t + head_start) + brad_speed * t = distance :=
by sorry

end NUMINAMATH_CALUDE_maxwell_brad_meeting_time_l108_10871


namespace NUMINAMATH_CALUDE_one_in_set_zero_one_l108_10888

theorem one_in_set_zero_one : 1 ∈ ({0, 1} : Set ℕ) := by
  sorry

end NUMINAMATH_CALUDE_one_in_set_zero_one_l108_10888


namespace NUMINAMATH_CALUDE_chess_game_probability_l108_10855

theorem chess_game_probability (draw_prob win_prob lose_prob : ℚ) : 
  draw_prob = 1/2 →
  win_prob = 1/3 →
  draw_prob + win_prob + lose_prob = 1 →
  lose_prob = 1/6 :=
by
  sorry

end NUMINAMATH_CALUDE_chess_game_probability_l108_10855


namespace NUMINAMATH_CALUDE_simplify_fraction_product_l108_10882

theorem simplify_fraction_product : 8 * (15 / 4) * (-28 / 45) = -56 / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_product_l108_10882


namespace NUMINAMATH_CALUDE_solve_a_l108_10885

def star (a b : ℝ) : ℝ := 2 * a - b^2

theorem solve_a : ∃ (a : ℝ), star a 5 = 9 ∧ a = 17 := by sorry

end NUMINAMATH_CALUDE_solve_a_l108_10885


namespace NUMINAMATH_CALUDE_ages_sum_l108_10801

theorem ages_sum (a b c : ℕ) : 
  a = b + c + 18 ∧ 
  a^2 = (b + c)^2 + 2016 → 
  a + b + c = 112 :=
by
  sorry

end NUMINAMATH_CALUDE_ages_sum_l108_10801


namespace NUMINAMATH_CALUDE_deepak_age_l108_10881

/-- Given the ratio of Rahul's age to Deepak's age and Rahul's future age, prove Deepak's present age -/
theorem deepak_age (rahul_age deepak_age : ℕ) : 
  (rahul_age : ℚ) / deepak_age = 4 / 3 →
  rahul_age + 6 = 26 →
  deepak_age = 15 := by
sorry

end NUMINAMATH_CALUDE_deepak_age_l108_10881
