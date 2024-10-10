import Mathlib

namespace binomial_minimum_sum_reciprocals_l2510_251010

/-- A discrete random variable following a binomial distribution -/
structure BinomialRV where
  n : ℕ
  p : ℝ
  h1 : 0 < p
  h2 : p < 1

/-- The expectation of a binomial random variable -/
def expectation (X : BinomialRV) : ℝ := X.n * X.p

/-- The variance of a binomial random variable -/
def variance (X : BinomialRV) : ℝ := X.n * X.p * (1 - X.p)

theorem binomial_minimum_sum_reciprocals (X : BinomialRV) (q : ℝ) 
    (h_expect : expectation X = 4)
    (h_var : variance X = q) :
    (∀ p q, p > 0 → q > 0 → 1/p + 1/q ≥ 9/4) ∧ 
    (∃ p q, p > 0 ∧ q > 0 ∧ 1/p + 1/q = 9/4) := by
  sorry

end binomial_minimum_sum_reciprocals_l2510_251010


namespace T_equals_five_l2510_251060

theorem T_equals_five :
  let T := 1 / (3 - Real.sqrt 8) - 1 / (Real.sqrt 8 - Real.sqrt 7) + 
           1 / (Real.sqrt 7 - Real.sqrt 6) - 1 / (Real.sqrt 6 - Real.sqrt 5) + 
           1 / (Real.sqrt 5 - 2)
  T = 5 := by sorry

end T_equals_five_l2510_251060


namespace polynomial_sequence_gcd_l2510_251013

/-- A sequence defined by polynomials with positive integer coefficients -/
def PolynomialSequence (p : ℕ → ℕ → ℕ) (a₀ : ℕ) : ℕ → ℕ :=
  fun n => p n a₀

/-- The theorem statement -/
theorem polynomial_sequence_gcd
  (p : ℕ → ℕ → ℕ)
  (h_p : ∀ n x, p n x > 0)
  (a₀ : ℕ)
  (a : ℕ → ℕ)
  (h_a : a = PolynomialSequence p a₀)
  (m k : ℕ) :
  Nat.gcd (a m) (a k) = a (Nat.gcd m k) := by
  sorry

end polynomial_sequence_gcd_l2510_251013


namespace polynomial_simplification_l2510_251053

theorem polynomial_simplification (x : ℝ) :
  3 - 5*x - 7*x^2 + 9 + 11*x - 13*x^2 - 15 + 17*x + 19*x^2 = -x^2 + 23*x - 3 := by
  sorry

end polynomial_simplification_l2510_251053


namespace new_person_weight_l2510_251090

/-- Given a group of 8 persons, if replacing one person weighing 65 kg
    with a new person increases the average weight by 2.5 kg,
    then the weight of the new person is 85 kg. -/
theorem new_person_weight (initial_count : ℕ) (weight_increase : ℝ) (replaced_weight : ℝ) :
  initial_count = 8 →
  weight_increase = 2.5 →
  replaced_weight = 65 →
  (initial_count : ℝ) * weight_increase + replaced_weight = 85 :=
by sorry

end new_person_weight_l2510_251090


namespace smallest_integer_in_set_l2510_251074

theorem smallest_integer_in_set (n : ℤ) : 
  (n + 5 < 3 * ((n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5)) / 6)) →
  (0 ≤ n) ∧ (∀ m : ℤ, m < n → m + 5 ≥ 3 * ((m + (m + 1) + (m + 2) + (m + 3) + (m + 4) + (m + 5)) / 6)) :=
by sorry

end smallest_integer_in_set_l2510_251074


namespace binomial_coefficient_20_19_l2510_251039

theorem binomial_coefficient_20_19 : Nat.choose 20 19 = 20 := by
  sorry

end binomial_coefficient_20_19_l2510_251039


namespace average_age_increase_l2510_251080

theorem average_age_increase (num_students : ℕ) (student_avg_age : ℝ) (teacher_age : ℝ) :
  num_students = 15 →
  student_avg_age = 10 →
  teacher_age = 26 →
  (((num_students : ℝ) * student_avg_age + teacher_age) / ((num_students : ℝ) + 1)) - student_avg_age = 1 := by
  sorry

end average_age_increase_l2510_251080


namespace inequality_solution_set_l2510_251056

theorem inequality_solution_set (a : ℝ) (ha : a < 0) :
  {x : ℝ | Real.sqrt (a^2 - 2*x^2) > x + a} = {x : ℝ | (Real.sqrt 2 / 2) * a ≤ x ∧ x ≤ -(Real.sqrt 2 / 2) * a} :=
by sorry

end inequality_solution_set_l2510_251056


namespace sum_leq_fourth_powers_over_product_l2510_251026

theorem sum_leq_fourth_powers_over_product (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  a + b + c ≤ (a^4 + b^4 + c^4) / (a * b * c) := by
  sorry

end sum_leq_fourth_powers_over_product_l2510_251026


namespace equation_solution_l2510_251099

theorem equation_solution : 
  ∃ y : ℝ, (3 / y + (4 / y) / (6 / y) = 1.5) ∧ y = 3.6 :=
by
  sorry

end equation_solution_l2510_251099


namespace average_correction_problem_l2510_251097

theorem average_correction_problem (initial_avg : ℚ) (misread : ℚ) (correct : ℚ) (correct_avg : ℚ) :
  initial_avg = 14 →
  misread = 26 →
  correct = 36 →
  correct_avg = 15 →
  ∃ n : ℕ, n > 0 ∧ 
    (n : ℚ) * initial_avg - misread + correct = (n : ℚ) * correct_avg ∧
    n = 10 := by
  sorry

end average_correction_problem_l2510_251097


namespace cost_difference_is_1267_50_l2510_251031

def initial_order : ℝ := 20000

def scheme1_discount1 : ℝ := 0.25
def scheme1_discount2 : ℝ := 0.15
def scheme1_discount3 : ℝ := 0.05

def scheme2_discount1 : ℝ := 0.20
def scheme2_discount2 : ℝ := 0.10
def scheme2_discount3 : ℝ := 0.05
def scheme2_rebate : ℝ := 300

def apply_discount (amount : ℝ) (discount : ℝ) : ℝ :=
  amount * (1 - discount)

def scheme1_final_cost : ℝ :=
  apply_discount (apply_discount (apply_discount initial_order scheme1_discount1) scheme1_discount2) scheme1_discount3

def scheme2_final_cost : ℝ :=
  apply_discount (apply_discount (apply_discount initial_order scheme2_discount1) scheme2_discount2) scheme2_discount3 - scheme2_rebate

theorem cost_difference_is_1267_50 :
  scheme1_final_cost - scheme2_final_cost = 1267.50 := by
  sorry

end cost_difference_is_1267_50_l2510_251031


namespace pizzas_with_mushrooms_or_olives_l2510_251011

def num_toppings : ℕ := 8

-- Function to calculate combinations
def combinations (n k : ℕ) : ℕ :=
  if k > n then 0
  else (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

-- Total number of pizzas with 1, 2, or 3 toppings
def total_pizzas : ℕ :=
  combinations num_toppings 1 + combinations num_toppings 2 + combinations num_toppings 3

-- Number of pizzas with mushrooms (or olives)
def pizzas_with_one_topping : ℕ :=
  1 + combinations (num_toppings - 1) 1 + combinations (num_toppings - 1) 2

-- Number of pizzas with both mushrooms and olives
def pizzas_with_both : ℕ :=
  1 + combinations (num_toppings - 2) 1 + combinations (num_toppings - 2) 2

-- Main theorem
theorem pizzas_with_mushrooms_or_olives :
  pizzas_with_one_topping * 2 - pizzas_with_both = 86 :=
sorry

end pizzas_with_mushrooms_or_olives_l2510_251011


namespace sqrt_equation_solution_l2510_251015

theorem sqrt_equation_solution :
  ∀ x : ℝ, Real.sqrt (2 * x + 14) = 10 → x = 43 := by
  sorry

end sqrt_equation_solution_l2510_251015


namespace angle_difference_l2510_251040

theorem angle_difference (α β : Real) 
  (h1 : 3 * Real.sin α - Real.cos α = 0)
  (h2 : 7 * Real.sin β + Real.cos β = 0)
  (h3 : 0 < α)
  (h4 : α < Real.pi / 2)
  (h5 : Real.pi / 2 < β)
  (h6 : β < Real.pi) :
  2 * α - β = -3 * Real.pi / 4 := by
sorry

end angle_difference_l2510_251040


namespace logo_enlargement_l2510_251063

/-- Calculates the height of a proportionally enlarged logo --/
def enlarged_logo_height (original_width original_height new_width : ℚ) : ℚ :=
  (new_width / original_width) * original_height

/-- Proves that a 3x2 inch logo enlarged to 12 inches wide will be 8 inches tall --/
theorem logo_enlargement :
  enlarged_logo_height 3 2 12 = 8 := by
  sorry

end logo_enlargement_l2510_251063


namespace initial_group_size_l2510_251075

theorem initial_group_size (initial_avg : ℝ) (new_people : ℕ) (new_avg : ℝ) (final_avg : ℝ) :
  initial_avg = 16 →
  new_people = 20 →
  new_avg = 15 →
  final_avg = 15.5 →
  ∃ x : ℕ, x = 20 ∧
    (x : ℝ) * initial_avg + (new_people : ℝ) * new_avg = (x + new_people : ℝ) * final_avg :=
by
  sorry

end initial_group_size_l2510_251075


namespace quadratic_equation_root_l2510_251093

theorem quadratic_equation_root (m : ℝ) : 
  (∃ x : ℝ, 3 * x^2 - m * x - 3 = 0 ∧ x = 1) → 
  (∃ y : ℝ, y ≠ 1 ∧ 3 * y^2 - m * y - 3 = 0 ∧ y = -1) := by
  sorry

end quadratic_equation_root_l2510_251093


namespace rectangular_box_volume_l2510_251058

theorem rectangular_box_volume (l w h : ℝ) (h1 : l * w = 30) (h2 : w * h = 20) (h3 : l * h = 12) :
  l * w * h = 60 := by
  sorry

end rectangular_box_volume_l2510_251058


namespace nth_power_divisors_l2510_251005

theorem nth_power_divisors (n : ℕ+) : 
  (∃ (d : ℕ), d = (Finset.card (Nat.divisors (n^n.val)))) → 
  d = 861 → 
  n = 20 := by
sorry

end nth_power_divisors_l2510_251005


namespace omelet_distribution_l2510_251084

theorem omelet_distribution (total_eggs : ℕ) (eggs_per_omelet : ℕ) (num_people : ℕ) :
  total_eggs = 36 →
  eggs_per_omelet = 4 →
  num_people = 3 →
  (total_eggs / eggs_per_omelet) / num_people = 3 := by
sorry

end omelet_distribution_l2510_251084


namespace book_length_ratio_l2510_251012

/- Define the variables -/
def starting_age : ℕ := 6
def starting_book_length : ℕ := 8
def current_book_length : ℕ := 480

/- Define the book length at twice the starting age -/
def book_length_twice_starting_age : ℕ := starting_book_length * 5

/- Define the book length 8 years after twice the starting age -/
def book_length_8_years_after : ℕ := book_length_twice_starting_age * 3

/- Theorem: The ratio of current book length to the book length 8 years after twice the starting age is 4:1 -/
theorem book_length_ratio :
  current_book_length / book_length_8_years_after = 4 :=
by sorry

end book_length_ratio_l2510_251012


namespace santa_gifts_l2510_251004

theorem santa_gifts (x : ℕ) (h1 : x < 100) (h2 : x % 2 = 0) (h3 : x % 5 = 0) (h4 : x % 7 = 0) :
  x - (x / 2 + x / 5 + x / 7) = 11 :=
by sorry

end santa_gifts_l2510_251004


namespace tangent_point_x_coordinate_l2510_251078

/-- Given a curve y = x^2 - 3x, if there exists a point where the tangent line
    has a slope of 1, then the x-coordinate of this point is 2. -/
theorem tangent_point_x_coordinate (x : ℝ) : 
  (∃ y : ℝ, y = x^2 - 3*x ∧ (deriv (fun x => x^2 - 3*x)) x = 1) → x = 2 :=
by sorry

end tangent_point_x_coordinate_l2510_251078


namespace inequality_system_solution_l2510_251091

theorem inequality_system_solution (m : ℝ) : 
  (∀ x : ℝ, (x + 3 < 3*x + 1 ∧ x > m + 1) ↔ x > 1) → 
  m ≤ 0 := by sorry

end inequality_system_solution_l2510_251091


namespace alternating_color_probability_value_l2510_251065

/-- Represents the number of balls of each color in the box -/
def num_balls : ℕ := 5

/-- Represents the total number of balls in the box -/
def total_balls : ℕ := 3 * num_balls

/-- Calculates the number of ways to arrange the balls -/
def total_arrangements : ℕ := Nat.choose total_balls num_balls * Nat.choose (2 * num_balls) num_balls

/-- Calculates the number of successful sequences (alternating colors) -/
def successful_sequences : ℕ := 2 * (3 ^ (num_balls - 1))

/-- The probability of drawing balls with alternating colors -/
def alternating_color_probability : ℚ := successful_sequences / total_arrangements

theorem alternating_color_probability_value : alternating_color_probability = 162 / 1001 := by
  sorry

end alternating_color_probability_value_l2510_251065


namespace darius_score_l2510_251054

/-- Represents the scores of Darius, Matt, and Marius in a table football game. -/
structure TableFootballScores where
  darius : ℕ
  matt : ℕ
  marius : ℕ

/-- The conditions of the table football game. -/
def game_conditions (scores : TableFootballScores) : Prop :=
  scores.marius = scores.darius + 3 ∧
  scores.matt = scores.darius + 5 ∧
  scores.darius + scores.matt + scores.marius = 38

/-- Theorem stating that under the given conditions, Darius scored 10 points. -/
theorem darius_score (scores : TableFootballScores) 
  (h : game_conditions scores) : scores.darius = 10 := by
  sorry

end darius_score_l2510_251054


namespace grocer_sales_problem_l2510_251008

theorem grocer_sales_problem (sales1 sales3 sales4 sales5 : ℕ) 
  (h1 : sales1 = 5420)
  (h3 : sales3 = 6200)
  (h4 : sales4 = 6350)
  (h5 : sales5 = 6500)
  (target_average : ℕ) 
  (h_target : target_average = 6000) :
  ∃ sales2 : ℕ, 
    sales2 = 5530 ∧ 
    (sales1 + sales2 + sales3 + sales4 + sales5) / 5 = target_average :=
by
  sorry

end grocer_sales_problem_l2510_251008


namespace smallest_four_digit_divisible_by_9_three_even_one_odd_l2510_251007

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def has_three_even_one_odd (n : ℕ) : Prop :=
  let digits := n.digits 10
  3 = (digits.filter (λ d => d % 2 = 0)).length ∧
  1 = (digits.filter (λ d => d % 2 = 1)).length

theorem smallest_four_digit_divisible_by_9_three_even_one_odd :
  ∀ n : ℕ, is_four_digit n → n % 9 = 0 → has_three_even_one_odd n → 1026 ≤ n := by
sorry

end smallest_four_digit_divisible_by_9_three_even_one_odd_l2510_251007


namespace perfect_square_trinomial_l2510_251073

theorem perfect_square_trinomial (a b t : ℝ) : 
  (∃ k : ℝ, a^2 + (2*t - 1)*a*b + 4*b^2 = (k*a + 2*b)^2) → 
  (t = 5/2 ∨ t = -3/2) :=
by sorry

end perfect_square_trinomial_l2510_251073


namespace cookie_pattern_holds_l2510_251055

/-- Represents the number of cookies on each plate -/
def cookie_sequence : Fin 6 → ℕ
  | 0 => 5
  | 1 => 7
  | 2 => 10
  | 3 => 14
  | 4 => 19
  | 5 => 25

/-- The difference between consecutive cookie counts increases by 1 each time -/
def increasing_difference (seq : Fin 6 → ℕ) : Prop :=
  ∀ i : Fin 4, seq (i + 1) - seq i = seq (i + 2) - seq (i + 1) + 1

theorem cookie_pattern_holds :
  increasing_difference cookie_sequence ∧ cookie_sequence 4 = 19 := by
  sorry

end cookie_pattern_holds_l2510_251055


namespace complex_modulus_problem_l2510_251087

theorem complex_modulus_problem (z : ℂ) (h : z * Complex.I = 1 + 2 * Complex.I) : 
  Complex.abs z = Real.sqrt 5 := by
  sorry

end complex_modulus_problem_l2510_251087


namespace polynomial_properties_l2510_251081

def f (x : ℝ) : ℝ := 8*x^7 + 5*x^6 + 3*x^4 + 2*x + 1

theorem polynomial_properties :
  (f 2 = 1397) ∧
  (f (-1) = -1) ∧
  (∃ c : ℝ, c ∈ Set.Icc (-1) 2 ∧ f c = 0) := by
  sorry

end polynomial_properties_l2510_251081


namespace division_theorem_l2510_251018

/-- The dividend polynomial -/
def f (x : ℝ) : ℝ := 3*x^5 - 2*x^3 + 5*x - 9

/-- The divisor polynomial -/
def g (x : ℝ) : ℝ := x^2 - 3*x + 2

/-- The proposed remainder polynomial -/
def r (x : ℝ) : ℝ := 92*x - 95

/-- Statement: The remainder when f(x) is divided by g(x) is r(x) -/
theorem division_theorem : ∃ q : ℝ → ℝ, ∀ x, f x = g x * q x + r x := by
  sorry

end division_theorem_l2510_251018


namespace eliza_age_l2510_251045

/-- Given the ages of Aunt Ellen, Dina, and Eliza, prove Eliza's age -/
theorem eliza_age (aunt_ellen_age : ℕ) (dina_age : ℕ) (eliza_age : ℕ) : 
  aunt_ellen_age = 48 →
  dina_age = aunt_ellen_age / 2 →
  eliza_age = dina_age - 6 →
  eliza_age = 18 := by
sorry

end eliza_age_l2510_251045


namespace factor_expression_l2510_251061

theorem factor_expression (x : ℝ) : 75 * x^2 + 50 * x = 25 * x * (3 * x + 2) := by
  sorry

end factor_expression_l2510_251061


namespace bingley_final_bracelets_l2510_251096

/-- The number of bracelets Bingley has at the end of the exchange process. -/
def final_bracelets : ℕ :=
  let bingley_initial := 5
  let kelly_initial := 16
  let kelly_gives := kelly_initial / 4
  let kelly_sets := kelly_gives / 3
  let bingley_receives := kelly_sets
  let bingley_after_receiving := bingley_initial + bingley_receives
  let bingley_gives_away := bingley_receives / 2
  let bingley_before_sister := bingley_after_receiving - bingley_gives_away
  let sister_gets := bingley_before_sister / 3
  bingley_before_sister - sister_gets

/-- Theorem stating that Bingley ends up with 4 bracelets. -/
theorem bingley_final_bracelets : final_bracelets = 4 := by
  sorry

end bingley_final_bracelets_l2510_251096


namespace min_arcs_for_circle_l2510_251016

theorem min_arcs_for_circle (arc_measure : ℝ) (n : ℕ) : 
  arc_measure = 120 → 
  (n : ℝ) * arc_measure = 360 → 
  n ≥ 3 ∧ ∀ m : ℕ, m < n → (m : ℝ) * arc_measure ≠ 360 :=
by sorry

end min_arcs_for_circle_l2510_251016


namespace train_crossing_time_l2510_251068

/-- Proves that a train with given length and speed takes the calculated time to cross a post -/
theorem train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) : 
  train_length = 120 → 
  train_speed_kmh = 72 → 
  (train_length / (train_speed_kmh * 1000 / 3600)) = 6 := by
  sorry

end train_crossing_time_l2510_251068


namespace mans_age_twice_sons_l2510_251070

/-- Proves that it takes 2 years for a man's age to be twice his son's age -/
theorem mans_age_twice_sons (
  son_age : ℕ) 
  (man_age : ℕ) 
  (h1 : son_age = 20) 
  (h2 : man_age = son_age + 22) : 
  ∃ y : ℕ, y = 2 ∧ man_age + y = 2 * (son_age + y) :=
sorry

end mans_age_twice_sons_l2510_251070


namespace square_minus_product_plus_square_l2510_251047

theorem square_minus_product_plus_square (a b : ℝ) 
  (h1 : a + b = 10) (h2 : a * b = 11) : 
  a^2 - a*b + b^2 = 67 := by sorry

end square_minus_product_plus_square_l2510_251047


namespace alicia_scored_14_points_per_half_l2510_251001

/-- Alicia's points per half of the game -/
def alicia_points_per_half (total_points : ℕ) (num_players : ℕ) (other_players_average : ℕ) : ℕ :=
  (total_points - (num_players - 1) * other_players_average) / 2

/-- Proof that Alicia scored 14 points in each half of the game -/
theorem alicia_scored_14_points_per_half :
  alicia_points_per_half 63 8 5 = 14 := by
  sorry

end alicia_scored_14_points_per_half_l2510_251001


namespace sin_cos_sum_equals_shifted_sin_l2510_251019

theorem sin_cos_sum_equals_shifted_sin (x : ℝ) : 
  Real.sin (3 * x) + Real.cos (3 * x) = Real.sqrt 2 * Real.sin (3 * (x + π / 12)) :=
by sorry

end sin_cos_sum_equals_shifted_sin_l2510_251019


namespace billy_candy_boxes_l2510_251077

/-- Given that Billy bought boxes of candy with 3 pieces per box and has a total of 21 pieces,
    prove that he bought 7 boxes. -/
theorem billy_candy_boxes : 
  ∀ (boxes : ℕ) (pieces_per_box : ℕ) (total_pieces : ℕ),
    pieces_per_box = 3 →
    total_pieces = 21 →
    boxes * pieces_per_box = total_pieces →
    boxes = 7 := by
  sorry

end billy_candy_boxes_l2510_251077


namespace simplify_expression_l2510_251051

theorem simplify_expression (x : ℝ) (h : x^2 ≥ 16) :
  (4 - Real.sqrt (x^2 - 16))^2 = x^2 - 8 * Real.sqrt (x^2 - 16) := by
  sorry

end simplify_expression_l2510_251051


namespace expression_evaluation_l2510_251067

theorem expression_evaluation (x : ℝ) : 
  x * (x * (x * (3 - x) - 5) + 12) + 2 = -x^4 + 3*x^3 - 5*x^2 + 12*x + 2 := by
  sorry

end expression_evaluation_l2510_251067


namespace fraction_value_l2510_251020

theorem fraction_value (a b c d : ℝ) 
  (h1 : a = 3 * b) 
  (h2 : b = 3 * c) 
  (h3 : c = 5 * d) : 
  (a * c) / (b * d) = 15 := by
  sorry

end fraction_value_l2510_251020


namespace irene_weekly_income_l2510_251057

/-- Calculates the total weekly income after taxes and deductions for an employee with given conditions --/
def total_weekly_income (base_salary : ℕ) (base_hours : ℕ) (overtime_rate1 : ℕ) (overtime_rate2 : ℕ) (overtime_rate3 : ℕ) (tax_rate : ℚ) (insurance_premium : ℕ) (hours_worked : ℕ) : ℕ :=
  sorry

/-- Theorem stating that given the specific conditions, the total weekly income is $645 --/
theorem irene_weekly_income :
  let base_salary := 500
  let base_hours := 40
  let overtime_rate1 := 20
  let overtime_rate2 := 30
  let overtime_rate3 := 40
  let tax_rate := 15 / 100
  let insurance_premium := 50
  let hours_worked := 50
  total_weekly_income base_salary base_hours overtime_rate1 overtime_rate2 overtime_rate3 tax_rate insurance_premium hours_worked = 645 :=
by
  sorry

end irene_weekly_income_l2510_251057


namespace binomial_coefficient_equality_l2510_251025

theorem binomial_coefficient_equality (n : ℕ+) :
  (Nat.choose n 2 = Nat.choose (n - 1) 2 + Nat.choose (n - 1) 3) → n = 5 := by
  sorry

end binomial_coefficient_equality_l2510_251025


namespace problem_statement_l2510_251037

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem problem_statement (f : ℝ → ℝ) 
  (h1 : is_even_function f)
  (h2 : has_period f 2)
  (h3 : ∀ x, 0 ≤ x ∧ x ≤ 1 → f x = x) :
  f (-1) + f (-2017) = 2 := by
  sorry

end problem_statement_l2510_251037


namespace integral_sqrt_minus_one_l2510_251002

theorem integral_sqrt_minus_one (f : ℝ → ℝ) :
  (∀ x, f x = Real.sqrt (1 - x^2) - 1) →
  (∫ x in (-1)..1, f x) = π / 2 - 2 := by
  sorry

end integral_sqrt_minus_one_l2510_251002


namespace exam_maximum_marks_l2510_251038

theorem exam_maximum_marks :
  ∀ (max_marks : ℕ) (passing_percentage : ℚ) (obtained_marks : ℕ) (failed_by : ℕ),
    passing_percentage = 40 / 100 →
    obtained_marks = 40 →
    failed_by = 40 →
    passing_percentage * max_marks = obtained_marks + failed_by →
    max_marks = 200 := by
  sorry

end exam_maximum_marks_l2510_251038


namespace double_earnings_in_ten_days_l2510_251044

/-- Calculates the number of additional days needed to earn twice the current amount --/
def additional_days_to_double_earnings (days_worked : ℕ) (total_earned : ℚ) : ℕ :=
  let daily_rate := total_earned / days_worked
  let target_amount := 2 * total_earned
  let total_days_needed := (target_amount / daily_rate).ceil.toNat
  total_days_needed - days_worked

/-- Theorem stating that for the given conditions, 10 additional days are needed --/
theorem double_earnings_in_ten_days :
  additional_days_to_double_earnings 10 250 = 10 := by
  sorry

#eval additional_days_to_double_earnings 10 250

end double_earnings_in_ten_days_l2510_251044


namespace hat_code_is_312_l2510_251006

def code_to_digit (c : Char) : Fin 6 :=
  match c with
  | 'M' => 0
  | 'A' => 1
  | 'T' => 2
  | 'H' => 3
  | 'I' => 4
  | 'S' => 5
  | _ => 0  -- Default case, should not occur in our problem

theorem hat_code_is_312 : 
  (code_to_digit 'H') * 100 + (code_to_digit 'A') * 10 + (code_to_digit 'T') = 312 := by
  sorry

end hat_code_is_312_l2510_251006


namespace g_monotone_decreasing_l2510_251076

/-- The function g(x) defined in terms of parameter a -/
def g (a : ℝ) (x : ℝ) : ℝ := a * x^3 + 2 * (1 - a) * x^2 - 3 * a * x

/-- The derivative of g(x) with respect to x -/
def g' (a : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + 4 * (1 - a) * x - 3 * a

/-- Theorem stating the condition for g(x) to be monotonically decreasing -/
theorem g_monotone_decreasing (a : ℝ) :
  (∀ x < a / 3, g' a x ≤ 0) ↔ -1 ≤ a ∧ a ≤ 0 :=
sorry

end g_monotone_decreasing_l2510_251076


namespace average_non_defective_cookies_l2510_251066

def cookie_counts : List Nat := [9, 11, 13, 16, 17, 18, 21, 22]

theorem average_non_defective_cookies :
  (cookie_counts.sum : ℚ) / cookie_counts.length = 127 / 8 := by
  sorry

end average_non_defective_cookies_l2510_251066


namespace negation_equivalence_l2510_251048

theorem negation_equivalence :
  (¬ ∃ x₀ : ℝ, x₀^2 + x₀ + 2 < 0) ↔ (∀ x : ℝ, x^2 + x + 2 ≥ 0) := by
  sorry

end negation_equivalence_l2510_251048


namespace soccer_ball_purchase_l2510_251082

theorem soccer_ball_purchase (first_batch_cost second_batch_cost : ℕ) 
  (unit_price_difference : ℕ) :
  first_batch_cost = 800 →
  second_batch_cost = 1560 →
  unit_price_difference = 2 →
  ∃ (first_batch_quantity second_batch_quantity : ℕ) 
    (first_unit_price second_unit_price : ℕ),
    first_batch_quantity * first_unit_price = first_batch_cost ∧
    second_batch_quantity * second_unit_price = second_batch_cost ∧
    second_batch_quantity = 2 * first_batch_quantity ∧
    first_unit_price = second_unit_price + unit_price_difference ∧
    first_batch_quantity + second_batch_quantity = 30 :=
by sorry

end soccer_ball_purchase_l2510_251082


namespace wednesday_sites_count_l2510_251014

theorem wednesday_sites_count (monday_sites tuesday_sites : ℕ)
  (monday_avg tuesday_avg wednesday_avg overall_avg : ℚ)
  (h1 : monday_sites = 5)
  (h2 : tuesday_sites = 5)
  (h3 : monday_avg = 7)
  (h4 : tuesday_avg = 5)
  (h5 : wednesday_avg = 8)
  (h6 : overall_avg = 7) :
  ∃ wednesday_sites : ℕ,
    (monday_sites * monday_avg + tuesday_sites * tuesday_avg + wednesday_sites * wednesday_avg) /
    (monday_sites + tuesday_sites + wednesday_sites : ℚ) = overall_avg ∧
    wednesday_sites = 10 := by
  sorry

end wednesday_sites_count_l2510_251014


namespace min_value_expression_min_value_attained_l2510_251030

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (6 * x) / (2 * y + z) + (3 * y) / (x + 2 * z) + (9 * z) / (x + y) ≥ 83 :=
by sorry

theorem min_value_attained (ε : ℝ) (hε : ε > 0) :
  ∃ x y z : ℝ, x > 0 ∧ y > 0 ∧ z > 0 ∧
    (6 * x) / (2 * y + z) + (3 * y) / (x + 2 * z) + (9 * z) / (x + y) < 83 + ε :=
by sorry

end min_value_expression_min_value_attained_l2510_251030


namespace cube_split_contains_31_l2510_251021

def split_cube (m : ℕ) : List ℕ :=
  let start := 2 * m * m - 2 * m + 1
  List.range m |>.map (fun i => start + 2 * i)

theorem cube_split_contains_31 (m : ℕ) (h1 : m > 1) :
  31 ∈ split_cube m → m = 6 := by
  sorry

end cube_split_contains_31_l2510_251021


namespace ellipse_eccentricity_l2510_251023

/-- The eccentricity of an ellipse with equation 16x²+4y²=1 is √3/2 -/
theorem ellipse_eccentricity : 
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 
  (∀ (x y : ℝ), 16 * x^2 + 4 * y^2 = 1 ↔ x^2 / a^2 + y^2 / b^2 = 1) ∧
  (a^2 - b^2) / a^2 = 3/4 :=
by sorry

end ellipse_eccentricity_l2510_251023


namespace pattern1_unique_violation_l2510_251079

/-- Represents a square in a pattern --/
structure Square where
  color : String

/-- Represents a pattern of squares --/
structure Pattern where
  squares : List Square
  arrangement : String

/-- Checks if a pattern can be folded into a cube --/
def can_fold_to_cube (p : Pattern) : Prop :=
  p.squares.length = 6 ∧ p.arrangement ≠ "linear"

/-- Checks if a pattern violates the adjacent color rule --/
def violates_adjacent_color_rule (p : Pattern) : Prop :=
  ∃ (s1 s2 : Square), s1 ∈ p.squares ∧ s2 ∈ p.squares ∧ s1.color = s2.color

/-- The four patterns described in the problem --/
def pattern1 : Pattern :=
  { squares := [
      { color := "blue" }, { color := "green" }, { color := "red" },
      { color := "blue" }, { color := "yellow" }, { color := "green" }
    ],
    arrangement := "cross"
  }

def pattern2 : Pattern :=
  { squares := [
      { color := "blue" }, { color := "green" }, { color := "red" },
      { color := "blue" }, { color := "yellow" }
    ],
    arrangement := "T"
  }

def pattern3 : Pattern :=
  { squares := [
      { color := "blue" }, { color := "green" }, { color := "red" },
      { color := "blue" }, { color := "yellow" }, { color := "green" },
      { color := "red" }
    ],
    arrangement := "custom"
  }

def pattern4 : Pattern :=
  { squares := [
      { color := "blue" }, { color := "green" }, { color := "red" },
      { color := "blue" }, { color := "yellow" }, { color := "green" }
    ],
    arrangement := "linear"
  }

/-- The main theorem --/
theorem pattern1_unique_violation :
  (can_fold_to_cube pattern1 ∧ violates_adjacent_color_rule pattern1) ∧
  (¬can_fold_to_cube pattern2 ∨ ¬violates_adjacent_color_rule pattern2) ∧
  (¬can_fold_to_cube pattern3 ∨ ¬violates_adjacent_color_rule pattern3) ∧
  (¬can_fold_to_cube pattern4 ∨ ¬violates_adjacent_color_rule pattern4) :=
sorry

end pattern1_unique_violation_l2510_251079


namespace a_range_is_open_2_5_l2510_251069

-- Define the sequence a_n
def a_n (a : ℝ) (n : ℕ) : ℝ :=
  if n ≤ 5 then (5 - a) * n - 11 else a ^ (n - 4)

-- Theorem statement
theorem a_range_is_open_2_5 :
  ∀ a : ℝ, (∀ n : ℕ, a_n a n < a_n a (n + 1)) →
  (2 < a ∧ a < 5) :=
by sorry

end a_range_is_open_2_5_l2510_251069


namespace triangle_area_in_square_l2510_251095

/-- The area of triangle ABC in a 12x12 square with specific point locations -/
theorem triangle_area_in_square : 
  let square_side : ℝ := 12
  let point_A : ℝ × ℝ := (square_side / 2, square_side)
  let point_B : ℝ × ℝ := (0, square_side / 4)
  let point_C : ℝ × ℝ := (square_side, square_side / 4)
  let triangle_area := (1 / 2) * 
    (|((point_C.1 - point_A.1) * (point_B.2 - point_A.2) - 
       (point_B.1 - point_A.1) * (point_C.2 - point_A.2))|)
  triangle_area = (27 * Real.sqrt 10) / 4 := by
  sorry


end triangle_area_in_square_l2510_251095


namespace motor_pool_vehicles_l2510_251083

theorem motor_pool_vehicles (x y : ℕ) : 
  x + y < 18 →
  y < 2 * x →
  x + 4 < y →
  (x = 6 ∧ y = 11) ∨ (∀ a b : ℕ, (a + b < 18 ∧ b < 2 * a ∧ a + 4 < b) → (a ≠ x ∨ b ≠ y)) :=
by sorry

end motor_pool_vehicles_l2510_251083


namespace function_composition_result_l2510_251089

/-- Given a function f(x) = x^2 - 2x, prove that f(f(f(1))) = 3 -/
theorem function_composition_result (f : ℝ → ℝ) (h : ∀ x, f x = x^2 - 2*x) : f (f (f 1)) = 3 := by
  sorry

end function_composition_result_l2510_251089


namespace pencils_bought_l2510_251028

theorem pencils_bought (glue_cost pencil_cost total_paid change : ℕ) 
  (h1 : glue_cost = 270)
  (h2 : pencil_cost = 210)
  (h3 : total_paid = 1000)
  (h4 : change = 100) :
  ∃ (num_pencils : ℕ), 
    glue_cost + num_pencils * pencil_cost = total_paid - change ∧ 
    num_pencils = 3 := by
  sorry

end pencils_bought_l2510_251028


namespace final_result_l2510_251086

def chosen_number : ℕ := 122
def multiplier : ℕ := 2
def subtractor : ℕ := 138

theorem final_result :
  chosen_number * multiplier - subtractor = 106 := by
  sorry

end final_result_l2510_251086


namespace product_sum_difference_equality_l2510_251046

theorem product_sum_difference_equality : 45 * 28 + 45 * 72 - 10 * 45 = 4050 := by
  sorry

end product_sum_difference_equality_l2510_251046


namespace largest_y_coordinate_l2510_251033

theorem largest_y_coordinate (x y : ℝ) : 
  (x^2 / 49) + ((y - 3)^2 / 25) = 0 → y ≤ 3 := by
  sorry

end largest_y_coordinate_l2510_251033


namespace law_of_sines_iff_equilateral_l2510_251034

/-- In a triangle ABC, the law of sines condition is equivalent to the triangle being equilateral -/
theorem law_of_sines_iff_equilateral (A B C : ℝ) (a b c : ℝ) :
  (A + B + C = π) →
  (a > 0 ∧ b > 0 ∧ c > 0) →
  (a / Real.sin B = b / Real.sin C ∧ b / Real.sin C = c / Real.sin A) ↔
  (a = b ∧ b = c) := by
  sorry


end law_of_sines_iff_equilateral_l2510_251034


namespace increase_by_percentage_l2510_251098

theorem increase_by_percentage (initial : ℝ) (percentage : ℝ) (final : ℝ) :
  initial = 80 ∧ percentage = 50 → final = initial * (1 + percentage / 100) → final = 120 := by
  sorry

end increase_by_percentage_l2510_251098


namespace min_value_2x_plus_y_l2510_251052

theorem min_value_2x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2*y - 2*x*y = 0) :
  2*x + y ≥ 9/2 ∧ ∃ x₀ y₀ : ℝ, x₀ > 0 ∧ y₀ > 0 ∧ x₀ + 2*y₀ - 2*x₀*y₀ = 0 ∧ 2*x₀ + y₀ = 9/2 := by
  sorry

end min_value_2x_plus_y_l2510_251052


namespace least_hour_square_remainder_fifteen_satisfies_condition_fifteen_is_least_l2510_251035

theorem least_hour_square_remainder (n : ℕ) : n > 9 ∧ n % 12 = (n^2) % 12 → n ≥ 15 := by
  sorry

theorem fifteen_satisfies_condition : 15 % 12 = (15^2) % 12 := by
  sorry

theorem fifteen_is_least : ∀ m : ℕ, m > 9 ∧ m % 12 = (m^2) % 12 → m ≥ 15 := by
  sorry

end least_hour_square_remainder_fifteen_satisfies_condition_fifteen_is_least_l2510_251035


namespace sum_of_tags_is_1000_l2510_251085

/-- The sum of tagged numbers on four cards W, X, Y, Z -/
def sum_of_tags (w x y z : ℕ) : ℕ := w + x + y + z

/-- Theorem stating the sum of tagged numbers is 1000 -/
theorem sum_of_tags_is_1000 :
  ∀ (w x y z : ℕ),
  w = 200 →
  x = w / 2 →
  y = x + w →
  z = 400 →
  sum_of_tags w x y z = 1000 := by
sorry

end sum_of_tags_is_1000_l2510_251085


namespace roots_of_f_l2510_251024

-- Define the polynomial function f
def f (x : ℝ) : ℝ := -3 * (x + 5)^2 + 45 * (x + 5) - 108

-- State the theorem
theorem roots_of_f :
  (f 7 = 0) ∧ (f (-2) = 0) ∧
  (∀ x : ℝ, f x = 0 → x = 7 ∨ x = -2) := by
  sorry

end roots_of_f_l2510_251024


namespace triangle_problem_l2510_251072

/-- 
Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
this theorem proves that under certain conditions, the angle C is 60° and 
the sides have specific lengths.
-/
theorem triangle_problem (a b c : ℝ) (A B C : ℝ) : 
  a > 0 → b > 0 → c > 0 →  -- Positive side lengths
  A > 0 → B > 0 → C > 0 →  -- Positive angles
  a > b →  -- Given condition
  a * (Real.sqrt 3 * Real.tan B - 1) = 
    (b * Real.cos A / Real.cos B) + (c * Real.cos A / Real.cos C) →  -- Given equation
  a + b + c = 20 →  -- Perimeter condition
  (1/2) * a * b * Real.sin C = 10 * Real.sqrt 3 →  -- Area condition
  C = Real.pi / 3 ∧ a = 8 ∧ b = 5 ∧ c = 7 := by
  sorry

end triangle_problem_l2510_251072


namespace matrix_equation_solution_l2510_251027

def B : Matrix (Fin 3) (Fin 3) ℤ := !![0, -1, 1; -1, 2, -1; 1, -1, 0]

theorem matrix_equation_solution :
  ∃ (s t u : ℤ), 
    B^3 + s • B^2 + t • B + u • (1 : Matrix (Fin 3) (Fin 3) ℤ) = 0 ∧ 
    s = -1 ∧ t = 0 ∧ u = 2 := by
  sorry

end matrix_equation_solution_l2510_251027


namespace rectangle_max_area_l2510_251064

/-- 
Given a rectangle with perimeter 60 units and one side at least half the length of the other,
the maximum possible area is 200 square units.
-/
theorem rectangle_max_area : 
  ∀ (a b : ℝ), 
  a > 0 ∧ b > 0 ∧                 -- sides are positive
  2 * (a + b) = 60 ∧              -- perimeter is 60
  a ≥ (1/2) * b ∧ b ≥ (1/2) * a → -- one side is at least half the other
  a * b ≤ 200 :=                  -- area is at most 200
by sorry

end rectangle_max_area_l2510_251064


namespace problem_1_problem_2_l2510_251049

-- Problem 1
theorem problem_1 (a b : ℝ) (ha : a ≠ 0) : (2 * a^2 * b) * a * b^2 / (4 * a^3) = (1/2) * b^3 := by
  sorry

-- Problem 2
theorem problem_2 (x : ℝ) : (2*x + 5) * (x - 3) = 2*x^2 - x - 15 := by
  sorry

end problem_1_problem_2_l2510_251049


namespace rectangle_dimensions_l2510_251043

/-- Given a rectangle with width x, length 4x, and area 120 square inches,
    prove that the width is √30 inches and the length is 4√30 inches. -/
theorem rectangle_dimensions (x : ℝ) (h1 : x > 0) (h2 : x * (4 * x) = 120) :
  x = Real.sqrt 30 ∧ 4 * x = 4 * Real.sqrt 30 := by
  sorry

#check rectangle_dimensions

end rectangle_dimensions_l2510_251043


namespace probability_specific_draw_l2510_251062

/-- Represents a standard deck of 52 playing cards -/
def StandardDeck : ℕ := 52

/-- Represents the number of cards drawn -/
def CardsDrawn : ℕ := 4

/-- Represents the number of 4s in a standard deck -/
def FoursInDeck : ℕ := 4

/-- Represents the number of clubs in a standard deck -/
def ClubsInDeck : ℕ := 13

/-- Represents the number of 2s in a standard deck -/
def TwosInDeck : ℕ := 4

/-- Represents the number of hearts in a standard deck -/
def HeartsInDeck : ℕ := 13

/-- The probability of drawing a 4, then a club, then a 2, then a heart from a standard 52-card deck -/
theorem probability_specific_draw : 
  (FoursInDeck : ℚ) / StandardDeck *
  ClubsInDeck / (StandardDeck - 1) *
  TwosInDeck / (StandardDeck - 2) *
  HeartsInDeck / (StandardDeck - 3) = 4 / 10829 := by
  sorry

end probability_specific_draw_l2510_251062


namespace line_charts_show_trend_bar_charts_dont_l2510_251042

-- Define the types of charts
inductive Chart
| BarChart
| LineChart

-- Define the capabilities of charts
def can_show_amount (c : Chart) : Prop :=
  match c with
  | Chart.BarChart => true
  | Chart.LineChart => true

def can_reflect_changes (c : Chart) : Prop :=
  match c with
  | Chart.BarChart => false
  | Chart.LineChart => true

-- Define what it means to show a trend
def can_show_trend (c : Chart) : Prop :=
  can_show_amount c ∧ can_reflect_changes c

-- Theorem statement
theorem line_charts_show_trend_bar_charts_dont :
  can_show_trend Chart.LineChart ∧ ¬can_show_trend Chart.BarChart :=
sorry

end line_charts_show_trend_bar_charts_dont_l2510_251042


namespace rational_polynomial_has_rational_coeffs_l2510_251088

/-- A polynomial that maps rationals to rationals has rational coefficients -/
theorem rational_polynomial_has_rational_coeffs (P : Polynomial ℚ) :
  (∀ q : ℚ, ∃ r : ℚ, P.eval q = r) →
  (∀ q : ℚ, ∃ r : ℚ, (P.eval q : ℚ) = r) →
  ∀ i : ℕ, ∃ q : ℚ, P.coeff i = q :=
sorry

end rational_polynomial_has_rational_coeffs_l2510_251088


namespace largest_integer_with_conditions_l2510_251036

def digits_of (n : ℕ) : List ℕ := sorry

def sum_of_squares (l : List ℕ) : ℕ := sorry

def is_strictly_increasing (l : List ℕ) : Prop := sorry

def product_of_list (l : List ℕ) : ℕ := sorry

theorem largest_integer_with_conditions : 
  let n := 2346
  (sum_of_squares (digits_of n) = 65) ∧ 
  (is_strictly_increasing (digits_of n)) ∧
  (∀ m : ℕ, m > n → 
    (sum_of_squares (digits_of m) ≠ 65) ∨ 
    (¬ is_strictly_increasing (digits_of m))) ∧
  (product_of_list (digits_of n) = 144) := by sorry

end largest_integer_with_conditions_l2510_251036


namespace player1_wins_533_player1_wins_1000_l2510_251032

/-- A game where two players alternately write 1 or 2, and the player who makes the sum reach or exceed the target loses. -/
def Game (target : ℕ) := Unit

/-- A strategy for playing the game. -/
def Strategy (target : ℕ) := Unit

/-- Determines if a strategy is winning for Player 1. -/
def is_winning_strategy (target : ℕ) (s : Strategy target) : Prop := sorry

/-- Player 1 has a winning strategy for the game with target 533. -/
theorem player1_wins_533 : ∃ s : Strategy 533, is_winning_strategy 533 s := sorry

/-- Player 1 has a winning strategy for the game with target 1000. -/
theorem player1_wins_1000 : ∃ s : Strategy 1000, is_winning_strategy 1000 s := sorry

end player1_wins_533_player1_wins_1000_l2510_251032


namespace min_value_z_l2510_251022

theorem min_value_z (x y : ℝ) (h1 : y ≥ x + 2) (h2 : x + y ≤ 6) (h3 : x ≥ 1) :
  ∃ (z : ℝ), z = 2 * |x - 2| + |y| ∧ z ≥ 4 ∧ ∀ (w : ℝ), w = 2 * |x - 2| + |y| → w ≥ z :=
by sorry

end min_value_z_l2510_251022


namespace circle_radius_l2510_251009

/-- The radius of the circle described by x^2 + y^2 - 4x + 6y = 0 is √13 -/
theorem circle_radius (x y : ℝ) : 
  (∀ x y, x^2 + y^2 - 4*x + 6*y = 0) → 
  ∃ r : ℝ, r = Real.sqrt 13 ∧ ∀ x y, (x - 2)^2 + (y + 3)^2 = r^2 :=
by sorry

end circle_radius_l2510_251009


namespace pushup_ratio_l2510_251094

theorem pushup_ratio : 
  ∀ (monday tuesday wednesday thursday friday : ℕ),
    monday = 5 →
    tuesday = 7 →
    wednesday = 2 * tuesday →
    friday = monday + tuesday + wednesday + thursday →
    friday = 39 →
    2 * thursday = monday + tuesday + wednesday :=
by
  sorry

end pushup_ratio_l2510_251094


namespace evaluate_expression_l2510_251050

theorem evaluate_expression : 3000^3 - 2998*3000^2 - 2998^2*3000 + 2998^3 = 23992 := by
  sorry

end evaluate_expression_l2510_251050


namespace sum_of_fourth_powers_simplification_l2510_251092

/-- Given distinct real numbers a, b, c, and d, the sum of four rational expressions
    simplifies to a linear polynomial. -/
theorem sum_of_fourth_powers_simplification 
  (a b c d : ℝ) (hab : a ≠ b) (hac : a ≠ c) (had : a ≠ d) 
  (hbc : b ≠ c) (hbd : b ≠ d) (hcd : c ≠ d) :
  let f : ℝ → ℝ := λ x => 
    ((x + a)^4) / ((a - b)*(a - c)*(a - d)) + 
    ((x + b)^4) / ((b - a)*(b - c)*(b - d)) + 
    ((x + c)^4) / ((c - a)*(c - b)*(c - d)) + 
    ((x + d)^4) / ((d - a)*(d - b)*(d - c))
  ∀ x, f x = a + b + c + d + 4*x := by
  sorry

end sum_of_fourth_powers_simplification_l2510_251092


namespace exist_three_numbers_with_equal_sum_l2510_251017

-- Define the sum of digits function
def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

-- Statement of the theorem
theorem exist_three_numbers_with_equal_sum :
  ∃ (m n p : ℕ), m ≠ n ∧ n ≠ p ∧ m ≠ p ∧
    m + sumOfDigits m = n + sumOfDigits n ∧
    n + sumOfDigits n = p + sumOfDigits p :=
sorry

end exist_three_numbers_with_equal_sum_l2510_251017


namespace x_thirteen_percent_greater_than_80_l2510_251071

theorem x_thirteen_percent_greater_than_80 :
  let x := 80 * (1 + 13 / 100)
  x = 90.4 := by sorry

end x_thirteen_percent_greater_than_80_l2510_251071


namespace triangles_in_circle_l2510_251000

/-- Given n points on a circle's circumference (n ≥ 6), with each pair connected by a chord
    and no three chords intersecting at a common point inside the circle,
    this function calculates the number of different triangles formed by the intersecting chords. -/
def num_triangles (n : ℕ) : ℕ :=
  Nat.choose n 3 + 4 * Nat.choose n 4 + 5 * Nat.choose n 5 + Nat.choose n 6

/-- Theorem stating that the number of triangles formed by intersecting chords
    in a circle with n points (n ≥ 6) on its circumference is given by num_triangles n. -/
theorem triangles_in_circle (n : ℕ) (h : n ≥ 6) :
  (num_triangles n) =
    Nat.choose n 3 + 4 * Nat.choose n 4 + 5 * Nat.choose n 5 + Nat.choose n 6 := by
  sorry

end triangles_in_circle_l2510_251000


namespace collinear_dots_probability_l2510_251059

/-- The number of dots in each row and column of the grid -/
def gridSize : ℕ := 5

/-- The number of possible sets of four collinear dots in a 5x5 grid -/
def collinearSets : ℕ := 16

/-- The total number of ways to choose 4 dots from 25 -/
def totalChoices : ℕ := 12650

/-- The probability of selecting four collinear dots in a 5x5 grid -/
theorem collinear_dots_probability :
  (collinearSets : ℚ) / totalChoices = 8 / 6325 := by sorry

end collinear_dots_probability_l2510_251059


namespace equation_roots_and_sum_l2510_251003

theorem equation_roots_and_sum : ∃ (c d : ℝ),
  (∃! (r₁ r₂ : ℝ), r₁ ≠ r₂ ∧ 
    (∀ x : ℝ, (x + 3) * (x + c) * (x - 9) = 0 ↔ (x = r₁ ∨ x = r₂))) ∧
  (∃! (s₁ s₂ s₃ : ℝ), s₁ ≠ s₂ ∧ s₁ ≠ s₃ ∧ s₂ ≠ s₃ ∧ 
    (∀ x : ℝ, (x - c) * (x - 7) * (x + 5) = 0 ↔ (x = s₁ ∨ x = s₂ ∨ x = s₃))) ∧
  80 * c + 10 * d = 650 :=
by sorry

end equation_roots_and_sum_l2510_251003


namespace arithmetic_mean_problem_l2510_251041

theorem arithmetic_mean_problem (p q r : ℝ) : 
  (p + q) / 2 = 10 → 
  (q + r) / 2 = 26 → 
  r - p = 32 → 
  (q + r) / 2 = 26 := by
sorry

end arithmetic_mean_problem_l2510_251041


namespace solution_y_initial_weight_l2510_251029

/-- Proves that the initial weight of solution Y is 8 kg given the problem conditions --/
theorem solution_y_initial_weight :
  ∀ (W : ℝ),
  (W > 0) →
  (0.20 * W = W * 0.20) →
  (0.25 * W = 0.20 * W + 0.4) →
  W = 8 := by
sorry

end solution_y_initial_weight_l2510_251029
