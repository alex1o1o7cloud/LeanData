import Mathlib

namespace NUMINAMATH_CALUDE_expand_and_simplify_l653_65388

theorem expand_and_simplify (x : ℝ) (h : x ≠ 0) :
  (3 / 4) * (8 / x^2 + 12 * x - 5) = 6 / x^2 + 9 * x - 15 / 4 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_l653_65388


namespace NUMINAMATH_CALUDE_probability_sum_greater_than_six_l653_65305

/-- Box A contains ping-pong balls numbered 1 and 2 -/
def box_A : Finset ℕ := {1, 2}

/-- Box B contains ping-pong balls numbered 3, 4, 5, and 6 -/
def box_B : Finset ℕ := {3, 4, 5, 6}

/-- The set of all possible outcomes when drawing one ball from each box -/
def all_outcomes : Finset (ℕ × ℕ) :=
  box_A.product box_B

/-- The set of favorable outcomes (sum greater than 6) -/
def favorable_outcomes : Finset (ℕ × ℕ) :=
  all_outcomes.filter (fun p => p.1 + p.2 > 6)

/-- The probability of drawing balls with sum greater than 6 -/
def probability : ℚ :=
  (favorable_outcomes.card : ℚ) / (all_outcomes.card : ℚ)

theorem probability_sum_greater_than_six : probability = 3/8 := by
  sorry

end NUMINAMATH_CALUDE_probability_sum_greater_than_six_l653_65305


namespace NUMINAMATH_CALUDE_three_thousand_six_hundred_factorization_l653_65330

theorem three_thousand_six_hundred_factorization (a b c d : ℕ+) 
  (h1 : 3600 = 2^(a.val) * 3^(b.val) * 4^(c.val) * 5^(d.val))
  (h2 : a.val + b.val + c.val + d.val = 7) : c.val = 1 := by
  sorry

end NUMINAMATH_CALUDE_three_thousand_six_hundred_factorization_l653_65330


namespace NUMINAMATH_CALUDE_maxRegions_correct_maxRegions_is_maximal_l653_65350

/-- The maximal number of regions a circle can be divided into by segments joining n points on its boundary -/
def maxRegions (n : ℕ) : ℕ :=
  Nat.choose n 4 + Nat.choose n 2 + 1

/-- Theorem stating that maxRegions gives the correct number of regions -/
theorem maxRegions_correct (n : ℕ) : 
  maxRegions n = Nat.choose n 4 + Nat.choose n 2 + 1 := by
  sorry

/-- Theorem stating that maxRegions indeed gives the maximal number of regions -/
theorem maxRegions_is_maximal (n : ℕ) :
  ∀ k : ℕ, k ≤ maxRegions n := by
  sorry

end NUMINAMATH_CALUDE_maxRegions_correct_maxRegions_is_maximal_l653_65350


namespace NUMINAMATH_CALUDE_a_m_prime_factors_l653_65317

def a_m (m : ℕ) : ℕ := (2^(2*m+1))^2 + 1

def has_at_most_two_prime_factors (n : ℕ) : Prop :=
  ∃ (p q : ℕ), Nat.Prime p ∧ Nat.Prime q ∧ ∀ (r : ℕ), Nat.Prime r → r ∣ n → r = p ∨ r = q

theorem a_m_prime_factors (m : ℕ) :
  has_at_most_two_prime_factors (a_m m) ↔ m = 0 ∨ m = 1 ∨ m = 2 := by sorry

end NUMINAMATH_CALUDE_a_m_prime_factors_l653_65317


namespace NUMINAMATH_CALUDE_factor_difference_of_squares_l653_65333

theorem factor_difference_of_squares (y : ℝ) : 81 - 16 * y^2 = (9 - 4*y) * (9 + 4*y) := by
  sorry

end NUMINAMATH_CALUDE_factor_difference_of_squares_l653_65333


namespace NUMINAMATH_CALUDE_product_simplification_l653_65360

theorem product_simplification (y : ℝ) (h : y ≠ 0) :
  (21 * y^3) * (9 * y^2) * (1 / (7*y)^2) = 27/7 * y^3 := by
  sorry

end NUMINAMATH_CALUDE_product_simplification_l653_65360


namespace NUMINAMATH_CALUDE_sum_in_M_l653_65371

/-- Define the set Mα for a positive real number α -/
def M (α : ℝ) (f : ℝ → ℝ) : Prop :=
  ∀ (x1 x2 : ℝ), x2 > x1 → 
    -α * (x2 - x1) < f x2 - f x1 ∧ f x2 - f x1 < α * (x2 - x1)

/-- Theorem: If f ∈ Mα1 and g ∈ Mα2, then f + g ∈ Mα1+α2 -/
theorem sum_in_M (α1 α2 : ℝ) (f g : ℝ → ℝ) 
    (hα1 : α1 > 0) (hα2 : α2 > 0) 
    (hf : M α1 f) (hg : M α2 g) : 
  M (α1 + α2) (f + g) := by
  sorry

end NUMINAMATH_CALUDE_sum_in_M_l653_65371


namespace NUMINAMATH_CALUDE_parabola_through_origin_l653_65340

/-- A parabola passing through the origin can be represented by the equation y = ax^2 + bx,
    where a and b are real numbers and at least one of them is non-zero. -/
theorem parabola_through_origin :
  ∀ (f : ℝ → ℝ), (∃ (a b : ℝ), a ≠ 0 ∨ b ≠ 0) →
  (f 0 = 0) →
  (∀ x, ∃ y, f x = y ∧ y = a * x^2 + b * x) →
  ∃ (a b : ℝ), (a ≠ 0 ∨ b ≠ 0) ∧ (∀ x, f x = a * x^2 + b * x) :=
by sorry

end NUMINAMATH_CALUDE_parabola_through_origin_l653_65340


namespace NUMINAMATH_CALUDE_complex_fraction_modulus_l653_65341

theorem complex_fraction_modulus (a b : ℝ) (i : ℂ) (h : i^2 = -1) :
  (1 + 2*i) / (a + b*i) = 2 - i → Complex.abs (a - b*i) = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_modulus_l653_65341


namespace NUMINAMATH_CALUDE_allison_wins_prob_l653_65313

/-- Represents a 6-sided cube with specified face values -/
structure Cube :=
  (faces : Fin 6 → ℕ)

/-- Allison's cube configuration -/
def allison_cube : Cube :=
  { faces := λ _ => 7 }

/-- Brian's cube configuration -/
def brian_cube : Cube :=
  { faces := λ i => i.val + 1 }

/-- Noah's cube configuration -/
def noah_cube : Cube :=
  { faces := λ i => if i.val < 2 then 3 else 5 }

/-- The probability of rolling a specific value or less on a given cube -/
def prob_roll_le (c : Cube) (n : ℕ) : ℚ :=
  (Finset.filter (λ i => c.faces i ≤ n) (Finset.univ : Finset (Fin 6))).card / 6

/-- The main theorem stating the probability of Allison's roll being greater than both Brian's and Noah's -/
theorem allison_wins_prob : 
  prob_roll_le brian_cube 6 * prob_roll_le noah_cube 6 = 1 := by
  sorry


end NUMINAMATH_CALUDE_allison_wins_prob_l653_65313


namespace NUMINAMATH_CALUDE_volume_ratio_l653_65331

/-- The domain S bounded by two curves -/
structure Domain (a b : ℝ) where
  (a_pos : a > 0)
  (b_pos : b > 0)

/-- The volume formed by revolving the domain around the x-axis -/
noncomputable def volume_x (d : Domain a b) : ℝ := sorry

/-- The volume formed by revolving the domain around the y-axis -/
noncomputable def volume_y (d : Domain a b) : ℝ := sorry

/-- The theorem stating the ratio of volumes -/
theorem volume_ratio (a b : ℝ) (d : Domain a b) :
  (volume_x d) / (volume_y d) = 14 / 5 := by sorry

end NUMINAMATH_CALUDE_volume_ratio_l653_65331


namespace NUMINAMATH_CALUDE_johns_total_pay_johns_total_pay_this_year_l653_65384

/-- Calculates the total pay (salary + bonus) given a salary and bonus percentage -/
def totalPay (salary : ℝ) (bonusPercentage : ℝ) : ℝ :=
  salary * (1 + bonusPercentage)

/-- Theorem: John's total pay is equal to his salary plus his bonus -/
theorem johns_total_pay (salary : ℝ) (bonusPercentage : ℝ) :
  totalPay salary bonusPercentage = salary + (salary * bonusPercentage) :=
by sorry

/-- Theorem: John's total pay this year is $220,000 -/
theorem johns_total_pay_this_year 
  (lastYearSalary lastYearBonus thisYearSalary : ℝ)
  (h1 : lastYearSalary = 100000)
  (h2 : lastYearBonus = 10000)
  (h3 : thisYearSalary = 200000)
  (h4 : lastYearBonus / lastYearSalary = thisYearSalary * bonusPercentage / thisYearSalary) :
  totalPay thisYearSalary (lastYearBonus / lastYearSalary) = 220000 :=
by sorry

end NUMINAMATH_CALUDE_johns_total_pay_johns_total_pay_this_year_l653_65384


namespace NUMINAMATH_CALUDE_reappearance_line_l653_65344

def letter_cycle_length : ℕ := 5
def digit_cycle_length : ℕ := 4

theorem reappearance_line : Nat.lcm letter_cycle_length digit_cycle_length = 20 := by
  sorry

end NUMINAMATH_CALUDE_reappearance_line_l653_65344


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_negation_of_sqrt_proposition_l653_65392

theorem negation_of_universal_proposition (p : ℝ → Prop) :
  (¬ ∀ x ≤ 0, p x) ↔ (∃ x₀ ≤ 0, ¬ p x₀) := by sorry

-- Define the specific proposition
def sqrt_prop (x : ℝ) : Prop := Real.sqrt (x^2) = -x

-- Main theorem
theorem negation_of_sqrt_proposition :
  (¬ ∀ x ≤ 0, sqrt_prop x) ↔ (∃ x₀ ≤ 0, ¬ sqrt_prop x₀) :=
negation_of_universal_proposition sqrt_prop

end NUMINAMATH_CALUDE_negation_of_universal_proposition_negation_of_sqrt_proposition_l653_65392


namespace NUMINAMATH_CALUDE_infinitely_many_circled_l653_65351

/-- A sequence of natural numbers -/
def Sequence := ℕ → ℕ

/-- Predicate that checks if a number in the sequence is circled -/
def IsCircled (a : Sequence) (n : ℕ) : Prop := a n ≥ n

/-- The main theorem stating that infinitely many numbers are circled -/
theorem infinitely_many_circled (a : Sequence) : 
  Set.Infinite {n : ℕ | IsCircled a n} := by
  sorry

end NUMINAMATH_CALUDE_infinitely_many_circled_l653_65351


namespace NUMINAMATH_CALUDE_cards_per_page_l653_65389

theorem cards_per_page
  (num_packs : ℕ)
  (cards_per_pack : ℕ)
  (num_pages : ℕ)
  (h1 : num_packs = 60)
  (h2 : cards_per_pack = 7)
  (h3 : num_pages = 42)
  : (num_packs * cards_per_pack) / num_pages = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_cards_per_page_l653_65389


namespace NUMINAMATH_CALUDE_inequality_proof_l653_65386

theorem inequality_proof (x y z : ℝ) (h1 : x < 0) (h2 : x < y) (h3 : y < z) :
  x + y < y + z := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l653_65386


namespace NUMINAMATH_CALUDE_complex_fraction_sum_l653_65369

theorem complex_fraction_sum (a b c d : ℝ) (ω : ℂ) 
  (ha : a ≠ -1) (hb : b ≠ -1) (hc : c ≠ -1) (hd : d ≠ -1)
  (hω1 : ω^3 = 1) (hω2 : ω ≠ 1)
  (h : (1 / (a + ω) + 1 / (b + ω) + 1 / (c + ω) + 1 / (d + ω)) = 3 / ω) :
  (1 / (a + 1) + 1 / (b + 1) + 1 / (c + 1) + 1 / (d + 1)) = 3 := by
sorry

end NUMINAMATH_CALUDE_complex_fraction_sum_l653_65369


namespace NUMINAMATH_CALUDE_parking_fee_range_l653_65324

/-- Represents the parking fee function --/
def parking_fee (x : ℝ) : ℝ := -5 * x + 12000

/-- Theorem: The parking fee range is [6900, 8100] given the problem conditions --/
theorem parking_fee_range :
  ∀ x : ℝ,
  0 ≤ x ∧ x ≤ 1200 ∧
  1200 * 0.65 ≤ x ∧ x ≤ 1200 * 0.85 →
  6900 ≤ parking_fee x ∧ parking_fee x ≤ 8100 :=
by sorry

end NUMINAMATH_CALUDE_parking_fee_range_l653_65324


namespace NUMINAMATH_CALUDE_quadratic_function_values_l653_65361

/-- A quadratic function f(x) = ax^2 + bx + c -/
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

/-- Theorem: If f(1) = 3 and f(2) = 5, then f(3) = 7 -/
theorem quadratic_function_values (a b c : ℝ) :
  f a b c 1 = 3 → f a b c 2 = 5 → f a b c 3 = 7 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_values_l653_65361


namespace NUMINAMATH_CALUDE_symmetry_of_point_l653_65309

/-- Given a point P and a line L, this function returns the point symmetric to P with respect to L -/
def symmetricPoint (P : ℝ × ℝ) (L : ℝ × ℝ → Prop) : ℝ × ℝ := sorry

/-- The line x + y = 1 -/
def lineXPlusYEq1 (p : ℝ × ℝ) : Prop := p.1 + p.2 = 1

theorem symmetry_of_point :
  symmetricPoint (2, 5) lineXPlusYEq1 = (-4, -1) := by sorry

end NUMINAMATH_CALUDE_symmetry_of_point_l653_65309


namespace NUMINAMATH_CALUDE_best_play_win_probability_best_play_win_probability_multi_l653_65398

/-- The probability that the best play wins with a majority of votes in a two-play competition. -/
theorem best_play_win_probability (n : ℕ) : ℝ :=
  let total_mothers : ℕ := 2 * n
  let confident_mothers : ℕ := n
  let non_confident_mothers : ℕ := n
  let vote_for_best_prob : ℝ := 1 / 2
  let vote_for_child_prob : ℝ := 1 / 2
  1 - (1 / 2) ^ n

/-- The probability that the best play wins with a majority of votes in a multi-play competition. -/
theorem best_play_win_probability_multi (n s : ℕ) : ℝ :=
  let total_mothers : ℕ := s * n
  let confident_mothers : ℕ := n
  let non_confident_mothers : ℕ := (s - 1) * n
  let vote_for_best_prob : ℝ := 1 / 2
  let vote_for_child_prob : ℝ := 1 / 2
  1 - (1 / 2) ^ ((s - 1) * n)

#check best_play_win_probability
#check best_play_win_probability_multi

end NUMINAMATH_CALUDE_best_play_win_probability_best_play_win_probability_multi_l653_65398


namespace NUMINAMATH_CALUDE_estimate_pi_random_simulation_l653_65343

/-- Estimate pi using a random simulation method with a square paper and inscribed circle -/
theorem estimate_pi_random_simulation (total_seeds : ℕ) (seeds_in_circle : ℕ) :
  total_seeds = 1000 →
  seeds_in_circle = 778 →
  ∃ (pi_estimate : ℝ), pi_estimate = 4 * (seeds_in_circle : ℝ) / (total_seeds : ℝ) ∧ 
                        abs (pi_estimate - 3.112) < 0.001 :=
by
  sorry

end NUMINAMATH_CALUDE_estimate_pi_random_simulation_l653_65343


namespace NUMINAMATH_CALUDE_gas_cost_equation_l653_65383

/-- The total cost of gas for a trip satisfies the given equation based on the change in cost per person when additional friends join. -/
theorem gas_cost_equation (x : ℝ) : x > 0 → (x / 5) - (x / 8) = 15.50 := by
  sorry

end NUMINAMATH_CALUDE_gas_cost_equation_l653_65383


namespace NUMINAMATH_CALUDE_exponent_division_l653_65338

theorem exponent_division (a : ℝ) : a^10 / a^9 = a := by
  sorry

end NUMINAMATH_CALUDE_exponent_division_l653_65338


namespace NUMINAMATH_CALUDE_cubic_equation_with_double_root_l653_65378

/-- The cubic equation coefficients -/
def a : ℝ := 3
def b : ℝ := 9
def c : ℝ := -135

/-- The cubic equation has a double root -/
def has_double_root (x y : ℝ) : Prop :=
  x = 2 * y ∨ y = 2 * x

/-- The value of k for which the statement holds -/
def k : ℝ := 525

/-- The main theorem -/
theorem cubic_equation_with_double_root :
  ∃ (x y : ℝ),
    a * x^3 + b * x^2 + c * x + k = 0 ∧
    a * y^3 + b * y^2 + c * y + k = 0 ∧
    has_double_root x y ∧
    k > 0 :=
sorry

end NUMINAMATH_CALUDE_cubic_equation_with_double_root_l653_65378


namespace NUMINAMATH_CALUDE_range_of_a_l653_65335

-- Define the propositions p, q, and r
def p (x : ℝ) : Prop := (x - 3) * (x + 1) < 0
def q (x : ℝ) : Prop := (x - 2) / (x - 4) < 0
def r (x a : ℝ) : Prop := a < x ∧ x < 2 * a

-- Define the theorem
theorem range_of_a (a : ℝ) :
  (a > 0) →
  (∀ x : ℝ, (p x ∧ q x) → r x a) →
  (3 / 2 ≤ a ∧ a ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l653_65335


namespace NUMINAMATH_CALUDE_distinct_roots_rectangle_perimeter_l653_65318

-- Define the quadratic equation
def quadratic (k : ℝ) (x : ℝ) : ℝ := x^2 - (2*k + 1)*x + 4*k - 3

-- Define the discriminant of the quadratic equation
def discriminant (k : ℝ) : ℝ := (2*k + 1)^2 - 4*(4*k - 3)

-- Statement 1: The equation always has two distinct real roots
theorem distinct_roots (k : ℝ) : discriminant k > 0 := by sorry

-- Define the sum and product of roots
def sum_of_roots (k : ℝ) : ℝ := 2*k + 1
def product_of_roots (k : ℝ) : ℝ := 4*k - 3

-- Statement 2: When roots represent rectangle sides with diagonal √31, perimeter is 14
theorem rectangle_perimeter (k : ℝ) 
  (h1 : sum_of_roots k^2 + product_of_roots k = 31) 
  (h2 : k > 0) : 
  2 * sum_of_roots k = 14 := by sorry

end NUMINAMATH_CALUDE_distinct_roots_rectangle_perimeter_l653_65318


namespace NUMINAMATH_CALUDE_gcf_of_lcm_sum_and_difference_l653_65354

theorem gcf_of_lcm_sum_and_difference : Nat.gcd (Nat.lcm 9 15 + 5) (Nat.lcm 10 21 - 7) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_lcm_sum_and_difference_l653_65354


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_sum_l653_65303

theorem geometric_sequence_common_ratio_sum (k a₂ a₃ b₂ b₃ : ℝ) (p r : ℝ) 
  (hk : k ≠ 0)
  (hp : p ≠ 1)
  (hr : r ≠ 1)
  (hp_neq_r : p ≠ r)
  (ha₂ : a₂ = k * p)
  (ha₃ : a₃ = k * p^2)
  (hb₂ : b₂ = k * r)
  (hb₃ : b₃ = k * r^2)
  (h_eq : a₃^2 - b₃^2 = 3 * (a₂^2 - b₂^2)) :
  p^2 + r^2 = 3 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_sum_l653_65303


namespace NUMINAMATH_CALUDE_milk_water_mixture_l653_65319

theorem milk_water_mixture (total_weight : ℝ) (added_water : ℝ) (new_ratio : ℝ) :
  total_weight = 85 →
  added_water = 5 →
  new_ratio = 3 →
  let initial_water := (total_weight - new_ratio * added_water) / (new_ratio + 1)
  let initial_milk := total_weight - initial_water
  (initial_milk / initial_water) = 27 / 7 := by
  sorry

end NUMINAMATH_CALUDE_milk_water_mixture_l653_65319


namespace NUMINAMATH_CALUDE_toms_average_increase_l653_65362

/-- Calculates the increase in average score given four exam scores -/
def increase_in_average (score1 score2 score3 score4 : ℚ) : ℚ :=
  let initial_average := (score1 + score2 + score3) / 3
  let new_average := (score1 + score2 + score3 + score4) / 4
  new_average - initial_average

/-- Theorem: The increase in Tom's average score is 3.25 -/
theorem toms_average_increase :
  increase_in_average 72 78 81 90 = 13/4 := by
  sorry

end NUMINAMATH_CALUDE_toms_average_increase_l653_65362


namespace NUMINAMATH_CALUDE_inverse_of_3_mod_1013_l653_65368

theorem inverse_of_3_mod_1013 : ∃ x : ℕ, 0 ≤ x ∧ x < 1013 ∧ (3 * x) % 1013 = 1 :=
by
  use 338
  sorry

end NUMINAMATH_CALUDE_inverse_of_3_mod_1013_l653_65368


namespace NUMINAMATH_CALUDE_smallest_integer_with_remainders_l653_65314

theorem smallest_integer_with_remainders : ∃ n : ℕ, 
  (n > 0) ∧ 
  (n % 6 = 5) ∧ 
  (n % 8 = 7) ∧ 
  (∀ m : ℕ, m > 0 ∧ m % 6 = 5 ∧ m % 8 = 7 → m ≥ n) ∧
  n = 23 := by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_with_remainders_l653_65314


namespace NUMINAMATH_CALUDE_binomial_n_minus_two_l653_65347

theorem binomial_n_minus_two (n : ℕ) (h : n ≥ 2) : 
  (n.choose (n - 2)) = n * (n - 1) / 2 :=
by sorry

end NUMINAMATH_CALUDE_binomial_n_minus_two_l653_65347


namespace NUMINAMATH_CALUDE_cubic_expression_evaluation_l653_65364

theorem cubic_expression_evaluation : 7^3 - 3 * 7^2 + 3 * 7 - 1 = 216 := by
  sorry

end NUMINAMATH_CALUDE_cubic_expression_evaluation_l653_65364


namespace NUMINAMATH_CALUDE_store_revenue_l653_65375

theorem store_revenue (december : ℝ) (november january : ℝ)
  (h1 : november = (3/5) * december)
  (h2 : january = (1/3) * november) :
  december = (5/2) * ((november + january) / 2) :=
by sorry

end NUMINAMATH_CALUDE_store_revenue_l653_65375


namespace NUMINAMATH_CALUDE_circle_condition_l653_65363

theorem circle_condition (m : ℝ) : 
  (∃ (h k r : ℝ), ∀ (x y : ℝ), x^2 + y^2 + 4*m*x - 2*y + 5*m = 0 ↔ (x - h)^2 + (y - k)^2 = r^2) ↔ 
  (m < 1/4 ∨ m > 1) :=
sorry

end NUMINAMATH_CALUDE_circle_condition_l653_65363


namespace NUMINAMATH_CALUDE_worker_y_defective_rate_l653_65399

/-- Calculates the defective rate of worker y given the conditions of the problem -/
theorem worker_y_defective_rate 
  (x_rate : Real) 
  (y_fraction : Real) 
  (total_rate : Real) 
  (hx : x_rate = 0.005) 
  (hy : y_fraction = 0.8) 
  (ht : total_rate = 0.0074) : 
  Real :=
by
  sorry

#check worker_y_defective_rate

end NUMINAMATH_CALUDE_worker_y_defective_rate_l653_65399


namespace NUMINAMATH_CALUDE_inequality_proof_l653_65373

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (h : a * b * c ≤ a + b + c) : 
  a^2 + b^2 + c^2 ≥ Real.sqrt 3 * (a * b * c) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l653_65373


namespace NUMINAMATH_CALUDE_chord_length_polar_l653_65377

/-- The chord length cut by the line ρcos(θ) = 1/2 from the circle ρ = 2cos(θ) is √3 -/
theorem chord_length_polar (ρ θ : ℝ) : 
  (ρ * Real.cos θ = 1/2) →  -- Line equation
  (ρ = 2 * Real.cos θ) →    -- Circle equation
  ∃ (chord_length : ℝ), chord_length = Real.sqrt 3 ∧ 
    chord_length = 2 * Real.sqrt (1 - (1/2)^2) := by
  sorry

end NUMINAMATH_CALUDE_chord_length_polar_l653_65377


namespace NUMINAMATH_CALUDE_circle_diameter_from_area_l653_65308

theorem circle_diameter_from_area (A : Real) (d : Real) :
  A = 225 * Real.pi → d = 30 :=
by
  sorry

end NUMINAMATH_CALUDE_circle_diameter_from_area_l653_65308


namespace NUMINAMATH_CALUDE_rice_cooking_is_algorithm_l653_65356

/-- Characteristics of an algorithm -/
structure AlgorithmCharacteristics where
  finite : Bool
  definite : Bool
  sequential : Bool
  correct : Bool
  nonUnique : Bool
  universal : Bool

/-- Representation of an algorithm -/
inductive AlgorithmRepresentation
  | NaturalLanguage
  | GraphicalLanguage
  | ProgrammingLanguage

/-- Steps for cooking rice -/
inductive RiceCookingStep
  | WashPot
  | RinseRice
  | AddWater
  | Heat

/-- Definition of an algorithm -/
def isAlgorithm (steps : List RiceCookingStep) (representation : AlgorithmRepresentation) 
  (characteristics : AlgorithmCharacteristics) : Prop :=
  characteristics.finite ∧
  characteristics.definite ∧
  characteristics.sequential ∧
  characteristics.correct ∧
  characteristics.nonUnique ∧
  characteristics.universal

/-- Theorem: The steps for cooking rice form an algorithm -/
theorem rice_cooking_is_algorithm : 
  ∃ (representation : AlgorithmRepresentation) (characteristics : AlgorithmCharacteristics),
    isAlgorithm [RiceCookingStep.WashPot, RiceCookingStep.RinseRice, 
                 RiceCookingStep.AddWater, RiceCookingStep.Heat] 
                representation characteristics :=
  sorry

end NUMINAMATH_CALUDE_rice_cooking_is_algorithm_l653_65356


namespace NUMINAMATH_CALUDE_logarithm_inequality_l653_65385

theorem logarithm_inequality (x : ℝ) (h : 1 < x ∧ x < 10) :
  Real.log (Real.log x) < (Real.log x)^2 ∧ (Real.log x)^2 < Real.log (x^2) := by
  sorry

end NUMINAMATH_CALUDE_logarithm_inequality_l653_65385


namespace NUMINAMATH_CALUDE_sum_of_squares_minus_linear_l653_65329

theorem sum_of_squares_minus_linear : ∀ x y : ℝ, 
  x ≠ y → 
  x^2 - 2000*x = y^2 - 2000*y → 
  x + y = 2000 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_minus_linear_l653_65329


namespace NUMINAMATH_CALUDE_beaded_corset_cost_l653_65336

/-- The number of rows of purple beads -/
def purple_rows : ℕ := 50

/-- The number of beads per row of purple beads -/
def purple_beads_per_row : ℕ := 20

/-- The number of rows of blue beads -/
def blue_rows : ℕ := 40

/-- The number of beads per row of blue beads -/
def blue_beads_per_row : ℕ := 18

/-- The number of gold beads -/
def gold_beads : ℕ := 80

/-- The cost of beads in dollars per 10 beads -/
def cost_per_10_beads : ℚ := 1

/-- The total cost of all beads in dollars -/
def total_cost : ℚ := 180

theorem beaded_corset_cost :
  (purple_rows * purple_beads_per_row + blue_rows * blue_beads_per_row + gold_beads) / 10 * cost_per_10_beads = total_cost := by
  sorry

end NUMINAMATH_CALUDE_beaded_corset_cost_l653_65336


namespace NUMINAMATH_CALUDE_divide_powers_of_nineteen_l653_65310

theorem divide_powers_of_nineteen : 19^12 / 19^8 = 130321 := by sorry

end NUMINAMATH_CALUDE_divide_powers_of_nineteen_l653_65310


namespace NUMINAMATH_CALUDE_chocolate_bar_cost_l653_65358

/-- The number of chocolate bars in the box initially -/
def initial_bars : ℕ := 13

/-- The number of bars Rachel didn't sell -/
def unsold_bars : ℕ := 4

/-- The total amount Rachel made from selling the bars -/
def total_revenue : ℚ := 18

/-- The cost of each chocolate bar -/
def bar_cost : ℚ := 2

theorem chocolate_bar_cost :
  (initial_bars - unsold_bars) * bar_cost = total_revenue :=
sorry

end NUMINAMATH_CALUDE_chocolate_bar_cost_l653_65358


namespace NUMINAMATH_CALUDE_power_equality_implies_exponent_l653_65320

theorem power_equality_implies_exponent (q : ℕ) : 27^8 = 9^q → q = 12 := by
  sorry

end NUMINAMATH_CALUDE_power_equality_implies_exponent_l653_65320


namespace NUMINAMATH_CALUDE_not_perfect_square_infinitely_often_l653_65355

theorem not_perfect_square_infinitely_often (a b : ℕ+) (h : ∃ p : ℕ, Nat.Prime p ∧ b = a + p) :
  ∃ S : Set ℕ, Set.Infinite S ∧ ∀ n ∈ S, ¬∃ k : ℕ, (a^n + a + 1) * (b^n + b + 1) = k^2 := by
  sorry

end NUMINAMATH_CALUDE_not_perfect_square_infinitely_often_l653_65355


namespace NUMINAMATH_CALUDE_complex_equation_sum_l653_65315

theorem complex_equation_sum (a b : ℝ) (i : ℂ) : 
  i * i = -1 → (a + 3 * i) / i = b - 2 * i → a + b = 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_sum_l653_65315


namespace NUMINAMATH_CALUDE_expression_value_l653_65322

theorem expression_value (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (sum_zero : a + b + c = 0) (sum_prod_nonzero : a * b + a * c + b * c ≠ 0) :
  (a^7 + b^7 + c^7) / (a * b * c * (a * b + a * c + b * c)) = -7 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l653_65322


namespace NUMINAMATH_CALUDE_town_population_problem_l653_65304

theorem town_population_problem (original_population : ℕ) : 
  (((original_population + 1200) : ℚ) * (1 - 11/100) : ℚ).floor = original_population - 32 → 
  original_population = 10000 := by
  sorry

end NUMINAMATH_CALUDE_town_population_problem_l653_65304


namespace NUMINAMATH_CALUDE_smallest_linear_combination_l653_65345

theorem smallest_linear_combination : 
  ∃ (k : ℕ), k > 0 ∧ 
  (∃ (m n p : ℤ), k = 2010 * m + 44550 * n + 100 * p) ∧
  (∀ (j : ℕ), j > 0 → (∃ (x y z : ℤ), j = 2010 * x + 44550 * y + 100 * z) → j ≥ k) ∧
  k = 10 := by
  sorry

end NUMINAMATH_CALUDE_smallest_linear_combination_l653_65345


namespace NUMINAMATH_CALUDE_angle_sum_from_tan_roots_l653_65372

theorem angle_sum_from_tan_roots (α β : Real) :
  (∃ x y : Real, x^2 + 6*x + 7 = 0 ∧ y^2 + 6*y + 7 = 0 ∧ x = Real.tan α ∧ y = Real.tan β) →
  α ∈ Set.Ioo (-π/2) (π/2) →
  β ∈ Set.Ioo (-π/2) (π/2) →
  α + β = -3*π/4 := by
sorry

end NUMINAMATH_CALUDE_angle_sum_from_tan_roots_l653_65372


namespace NUMINAMATH_CALUDE_rectangle_increase_l653_65327

/-- Proves that for a rectangle with length increased by 10% and area increased by 37.5%,
    the breadth must be increased by 25% -/
theorem rectangle_increase (L B : ℝ) (h_pos_L : L > 0) (h_pos_B : B > 0) : 
  ∃ p : ℝ, 
    (1.1 * L) * (B * (1 + p / 100)) = 1.375 * (L * B) ∧ 
    p = 25 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_increase_l653_65327


namespace NUMINAMATH_CALUDE_point_satisfies_inequality_l653_65307

/-- A point in the Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- The inequality function -/
def inequality (p : Point) : ℝ :=
  (p.x + 2*p.y - 1) * (p.x - p.y + 3)

/-- Theorem stating that the point (0,2) satisfies the inequality -/
theorem point_satisfies_inequality : 
  let p : Point := ⟨0, 2⟩
  inequality p > 0 := by
  sorry


end NUMINAMATH_CALUDE_point_satisfies_inequality_l653_65307


namespace NUMINAMATH_CALUDE_solution_value_l653_65366

/-- The function F as defined in the problem -/
def F (a b c : ℝ) : ℝ := a * b^3 + c^2

/-- Theorem stating that 5/19 is the value of a that satisfies the equation -/
theorem solution_value : ∃ (a : ℝ), F a 3 2 = F a 2 3 ∧ a = 5/19 := by
  sorry

end NUMINAMATH_CALUDE_solution_value_l653_65366


namespace NUMINAMATH_CALUDE_patanjali_walk_l653_65339

/-- Represents the walking scenario of Patanjali over three days --/
structure WalkingScenario where
  hours_day1 : ℕ
  speed_day1 : ℕ
  total_distance : ℕ

/-- Calculates the distance walked on the first day given a WalkingScenario --/
def distance_day1 (scenario : WalkingScenario) : ℕ :=
  scenario.hours_day1 * scenario.speed_day1

/-- Calculates the total distance walked over three days given a WalkingScenario --/
def total_distance (scenario : WalkingScenario) : ℕ :=
  (distance_day1 scenario) + 
  (scenario.hours_day1 - 1) * (scenario.speed_day1 + 1) + 
  scenario.hours_day1 * (scenario.speed_day1 + 1)

/-- Theorem stating that given the conditions, the distance walked on the first day is 18 miles --/
theorem patanjali_walk (scenario : WalkingScenario) 
  (h1 : scenario.speed_day1 = 3) 
  (h2 : total_distance scenario = 62) : 
  distance_day1 scenario = 18 := by
  sorry

#eval distance_day1 { hours_day1 := 6, speed_day1 := 3, total_distance := 62 }

end NUMINAMATH_CALUDE_patanjali_walk_l653_65339


namespace NUMINAMATH_CALUDE_smallest_prime_with_digit_sum_23_l653_65381

/-- A function that returns the sum of digits of a natural number -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- A function that checks if a natural number is prime -/
def is_prime (n : ℕ) : Prop := sorry

/-- Theorem: 599 is the smallest prime number whose digits sum to 23 -/
theorem smallest_prime_with_digit_sum_23 :
  (is_prime 599) ∧ 
  (digit_sum 599 = 23) ∧ 
  (∀ n : ℕ, n < 599 → ¬(is_prime n ∧ digit_sum n = 23)) := by sorry

end NUMINAMATH_CALUDE_smallest_prime_with_digit_sum_23_l653_65381


namespace NUMINAMATH_CALUDE_y1_gt_y2_l653_65387

/-- A quadratic function with a positive leading coefficient and symmetric axis at x = 1 -/
structure SymmetricQuadratic where
  a : ℝ
  b : ℝ
  c : ℝ
  a_pos : a > 0
  sym_axis : b = -2 * a

/-- The y-coordinate of the quadratic function at a given x -/
def y_coord (q : SymmetricQuadratic) (x : ℝ) : ℝ :=
  q.a * x^2 + q.b * x + q.c

/-- Theorem stating that y₁ > y₂ for the given quadratic function -/
theorem y1_gt_y2 (q : SymmetricQuadratic) (y₁ y₂ : ℝ)
  (h1 : y_coord q (-1) = y₁)
  (h2 : y_coord q 2 = y₂) :
  y₁ > y₂ := by
  sorry

end NUMINAMATH_CALUDE_y1_gt_y2_l653_65387


namespace NUMINAMATH_CALUDE_total_marbles_l653_65394

/-- The total number of marbles for three people given specific conditions -/
theorem total_marbles (my_marbles : ℕ) (brother_marbles : ℕ) (friend_marbles : ℕ) 
  (h1 : my_marbles = 16)
  (h2 : my_marbles - 2 = 2 * (brother_marbles + 2))
  (h3 : friend_marbles = 3 * (my_marbles - 2)) :
  my_marbles + brother_marbles + friend_marbles = 63 := by
  sorry

end NUMINAMATH_CALUDE_total_marbles_l653_65394


namespace NUMINAMATH_CALUDE_percentage_enrolled_in_biology_l653_65357

theorem percentage_enrolled_in_biology (total_students : ℕ) (not_enrolled : ℕ) 
  (h1 : total_students = 880) (h2 : not_enrolled = 638) :
  (((total_students - not_enrolled) : ℚ) / total_students) * 100 = 27.5 := by
  sorry

end NUMINAMATH_CALUDE_percentage_enrolled_in_biology_l653_65357


namespace NUMINAMATH_CALUDE_pyramid_angle_ratio_relationship_l653_65396

/-- A pyramid with all lateral faces forming the same angle with the base -/
structure Pyramid where
  base_area : ℝ
  lateral_angle : ℝ
  total_to_base_ratio : ℝ

/-- The angle formed by the lateral faces with the base of the pyramid -/
def lateral_angle (p : Pyramid) : ℝ := p.lateral_angle

/-- The ratio of the total surface area to the base area of the pyramid -/
def total_to_base_ratio (p : Pyramid) : ℝ := p.total_to_base_ratio

/-- Theorem stating the relationship between the lateral angle and the total-to-base area ratio -/
theorem pyramid_angle_ratio_relationship (p : Pyramid) :
  lateral_angle p = Real.arccos (4 / (total_to_base_ratio p - 1)) ∧
  total_to_base_ratio p > 5 := by sorry

end NUMINAMATH_CALUDE_pyramid_angle_ratio_relationship_l653_65396


namespace NUMINAMATH_CALUDE_min_tenuous_g7_l653_65376

/-- A tenuous function is an integer-valued function g such that
    g(x) + g(y) > x^2 for all positive integers x and y. -/
def Tenuous (g : ℕ+ → ℤ) : Prop :=
  ∀ x y : ℕ+, g x + g y > (x : ℤ)^2

/-- The sum of g(1) to g(10) for a function g. -/
def SumG (g : ℕ+ → ℤ) : ℤ :=
  (Finset.range 10).sum (fun i => g ⟨i + 1, Nat.succ_pos i⟩)

/-- A tenuous function g that minimizes the sum of g(1) to g(10). -/
def MinTenuous (g : ℕ+ → ℤ) : Prop :=
  Tenuous g ∧ ∀ h : ℕ+ → ℤ, Tenuous h → SumG g ≤ SumG h

theorem min_tenuous_g7 (g : ℕ+ → ℤ) (hg : MinTenuous g) : g ⟨7, by norm_num⟩ ≥ 49 := by
  sorry

end NUMINAMATH_CALUDE_min_tenuous_g7_l653_65376


namespace NUMINAMATH_CALUDE_calculation_proof_l653_65379

theorem calculation_proof : (1000 : ℤ) * 7 / 10 * 17 * (5^2) = 297500 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l653_65379


namespace NUMINAMATH_CALUDE_second_week_rainfall_l653_65348

/-- Proves that the rainfall during the second week of January was 15 inches,
    given the total rainfall and the relationship between the two weeks. -/
theorem second_week_rainfall (total_rainfall : ℝ) (first_week : ℝ) (second_week : ℝ) : 
  total_rainfall = 25 →
  second_week = 1.5 * first_week →
  total_rainfall = first_week + second_week →
  second_week = 15 := by
  sorry


end NUMINAMATH_CALUDE_second_week_rainfall_l653_65348


namespace NUMINAMATH_CALUDE_fishing_ratio_l653_65323

/-- Given that Jordan caught 4 fish and after losing one-fourth of their total catch, 
    9 fish remain, prove that the ratio of Perry's catch to Jordan's catch is 2:1 -/
theorem fishing_ratio : 
  let jordan_catch : ℕ := 4
  let remaining_fish : ℕ := 9
  let total_catch : ℕ := remaining_fish * 4 / 3
  let perry_catch : ℕ := total_catch - jordan_catch
  (perry_catch : ℚ) / jordan_catch = 2 / 1 := by
sorry


end NUMINAMATH_CALUDE_fishing_ratio_l653_65323


namespace NUMINAMATH_CALUDE_baseball_cards_pages_l653_65352

def organize_baseball_cards (new_cards : ℕ) (old_cards : ℕ) (cards_per_page : ℕ) : ℕ :=
  (new_cards + old_cards) / cards_per_page

theorem baseball_cards_pages :
  organize_baseball_cards 8 10 3 = 6 := by
  sorry

end NUMINAMATH_CALUDE_baseball_cards_pages_l653_65352


namespace NUMINAMATH_CALUDE_root_ratio_implies_k_value_l653_65311

theorem root_ratio_implies_k_value (k : ℝ) :
  (∃ r s : ℝ, r ≠ 0 ∧ s ≠ 0 ∧ r / s = 3 ∧ 
   r^2 - 4*r + k = 0 ∧ s^2 - 4*s + k = 0) →
  k = 3 := by
sorry

end NUMINAMATH_CALUDE_root_ratio_implies_k_value_l653_65311


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l653_65326

-- Define the polynomial representation
def polynomial (a : ℝ) (a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ : ℝ) (x : ℝ) : ℝ :=
  a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7 + a₈*x^8 + a₉*x^9 + a₁₀*x^10

-- Define the given equation
def equation (a : ℝ) (a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ : ℝ) (x : ℝ) : Prop :=
  (1 - 2*x)^10 = polynomial a a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ x

-- Theorem to prove
theorem sum_of_coefficients 
  (a : ℝ) (a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ : ℝ) :
  (∀ x, equation a a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ x) →
  10*a₁ + 9*a₂ + 8*a₃ + 7*a₄ + 6*a₅ + 5*a₆ + 4*a₇ + 3*a₈ + 2*a₉ + a₁₀ = -20 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l653_65326


namespace NUMINAMATH_CALUDE_shooting_training_results_l653_65395

/-- Represents the shooting scores and their frequencies -/
structure ShootingData :=
  (scores : List Nat)
  (frequencies : List Nat)
  (excellent_threshold : Nat)
  (total_freshmen : Nat)

/-- Calculates the mode of the shooting data -/
def mode (data : ShootingData) : Nat :=
  sorry

/-- Calculates the average score of the shooting data -/
def average_score (data : ShootingData) : Rat :=
  sorry

/-- Estimates the number of excellent shooters -/
def estimate_excellent_shooters (data : ShootingData) : Nat :=
  sorry

/-- The main theorem proving the results of the shooting training -/
theorem shooting_training_results (data : ShootingData) 
  (h1 : data.scores = [6, 7, 8, 9])
  (h2 : data.frequencies = [1, 6, 3, 2])
  (h3 : data.excellent_threshold = 8)
  (h4 : data.total_freshmen = 1500) :
  mode data = 7 ∧ 
  average_score data = 15/2 ∧ 
  estimate_excellent_shooters data = 625 :=
sorry

end NUMINAMATH_CALUDE_shooting_training_results_l653_65395


namespace NUMINAMATH_CALUDE_product_of_real_parts_l653_65332

theorem product_of_real_parts (x : ℂ) : 
  x^2 - 6*x = -8 + 2*I → 
  ∃ (x₁ x₂ : ℂ), x₁ ≠ x₂ ∧ 
    x₁^2 - 6*x₁ = -8 + 2*I ∧ 
    x₂^2 - 6*x₂ = -8 + 2*I ∧
    (x₁.re * x₂.re = 9 - Real.sqrt 5 / 2) :=
by sorry

end NUMINAMATH_CALUDE_product_of_real_parts_l653_65332


namespace NUMINAMATH_CALUDE_non_officers_count_l653_65337

/-- Prove the number of non-officers in an office given salary information -/
theorem non_officers_count (avg_salary : ℝ) (officer_salary : ℝ) (non_officer_salary : ℝ) 
  (officer_count : ℕ) (h1 : avg_salary = 120) (h2 : officer_salary = 430) 
  (h3 : non_officer_salary = 110) (h4 : officer_count = 15) : 
  ∃ (non_officer_count : ℕ), 
    avg_salary * (officer_count + non_officer_count) = 
    officer_salary * officer_count + non_officer_salary * non_officer_count ∧ 
    non_officer_count = 465 := by
  sorry

end NUMINAMATH_CALUDE_non_officers_count_l653_65337


namespace NUMINAMATH_CALUDE_unique_solution_system_l653_65370

theorem unique_solution_system (x y : ℝ) : 
  (2 * x + y = 3 ∧ x - y = 3) ↔ (x = 2 ∧ y = -1) :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_system_l653_65370


namespace NUMINAMATH_CALUDE_watch_cost_l653_65353

theorem watch_cost (watch_cost strap_cost : ℝ) 
  (total_cost : watch_cost + strap_cost = 120)
  (cost_difference : watch_cost = strap_cost + 100) :
  watch_cost = 110 := by
sorry

end NUMINAMATH_CALUDE_watch_cost_l653_65353


namespace NUMINAMATH_CALUDE_largest_expression_l653_65312

theorem largest_expression : 
  let a := (1 : ℚ) / 2
  let b := (1 : ℚ) / 3 + (1 : ℚ) / 4
  let c := (1 : ℚ) / 4 + (1 : ℚ) / 5 + (1 : ℚ) / 6
  let d := (1 : ℚ) / 5 + (1 : ℚ) / 6 + (1 : ℚ) / 7 + (1 : ℚ) / 8
  let e := (1 : ℚ) / 6 + (1 : ℚ) / 7 + (1 : ℚ) / 8 + (1 : ℚ) / 9 + (1 : ℚ) / 10
  e > a ∧ e > b ∧ e > c ∧ e > d := by
  sorry

end NUMINAMATH_CALUDE_largest_expression_l653_65312


namespace NUMINAMATH_CALUDE_triangle_inequality_cube_root_l653_65321

theorem triangle_inequality_cube_root (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ b + c > a ∧ c + a > b) :
  (((a^2 + b*c) * (b^2 + c*a) * (c^2 + a*b))^(1/3) : ℝ) > (a^2 + b^2 + c^2) / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_cube_root_l653_65321


namespace NUMINAMATH_CALUDE_expression_independent_of_alpha_l653_65334

theorem expression_independent_of_alpha :
  ∀ α : ℝ, 
    Real.sin (250 * π / 180 + α) * Real.cos (200 * π / 180 - α) - 
    Real.cos (240 * π / 180) * Real.cos (220 * π / 180 - 2 * α) = 
    1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_independent_of_alpha_l653_65334


namespace NUMINAMATH_CALUDE_inequality_system_solution_l653_65300

theorem inequality_system_solution :
  let S : Set ℝ := {x | (x - 1) / 2 + 2 > x ∧ 2 * (x - 2) ≤ 3 * x - 5}
  S = {x | 1 ≤ x ∧ x < 3} := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l653_65300


namespace NUMINAMATH_CALUDE_minimal_solution_l653_65325

def is_solution (A B C : ℕ) : Prop :=
  A ≠ B ∧ B ≠ C ∧ A ≠ C ∧
  (1 : ℚ) / A + (1 : ℚ) / B + (1 : ℚ) / C = (1 : ℚ) / 6 ∧
  6 ∣ A ∧ 6 ∣ B ∧ 6 ∣ C

theorem minimal_solution :
  ∀ A B C : ℕ, is_solution A B C →
  A + B + C ≥ 12 + 18 + 36 ∧
  is_solution 12 18 36 :=
sorry

end NUMINAMATH_CALUDE_minimal_solution_l653_65325


namespace NUMINAMATH_CALUDE_independence_test_not_always_correct_l653_65301

-- Define the independence test
def independence_test (sample : Type) : Prop := True

-- Define the principle of small probability
def principle_of_small_probability : Prop := True

-- Define the concept of different samples
def different_samples (s1 s2 : Type) : Prop := s1 ≠ s2

-- Define the concept of different conclusions
def different_conclusions (c1 c2 : Prop) : Prop := c1 ≠ c2

-- Define other methods for determining categorical variable relationships
def other_methods_exist : Prop := True

-- Theorem statement
theorem independence_test_not_always_correct :
  (∀ (s : Type), independence_test s → principle_of_small_probability) →
  (∀ (s1 s2 : Type), different_samples s1 s2 → 
    ∃ (c1 c2 : Prop), different_conclusions c1 c2) →
  other_methods_exist →
  ¬(∀ (s : Type), independence_test s → 
    ∀ (conclusion : Prop), conclusion) :=
by sorry

end NUMINAMATH_CALUDE_independence_test_not_always_correct_l653_65301


namespace NUMINAMATH_CALUDE_range_of_x2_plus_y2_l653_65397

theorem range_of_x2_plus_y2 (x y : ℝ) (h : (x + 2)^2 + y^2/4 = 1) :
  ∃ (min max : ℝ), min = 1 ∧ max = 28/3 ∧
  (x^2 + y^2 ≥ min ∧ x^2 + y^2 ≤ max) ∧
  (∀ z, (∃ a b : ℝ, (a + 2)^2 + b^2/4 = 1 ∧ z = a^2 + b^2) → z ≥ min ∧ z ≤ max) :=
sorry

end NUMINAMATH_CALUDE_range_of_x2_plus_y2_l653_65397


namespace NUMINAMATH_CALUDE_complex_magnitude_l653_65316

theorem complex_magnitude (z : ℂ) (h : z * (1 + Complex.I) = Complex.I) :
  Complex.abs z = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l653_65316


namespace NUMINAMATH_CALUDE_password_generation_l653_65328

def polynomial (x y : ℤ) : ℤ := 32 * x^3 - 8 * x * y^2

def factor1 (x : ℤ) : ℤ := 8 * x
def factor2 (x y : ℤ) : ℤ := 2 * x + y
def factor3 (x y : ℤ) : ℤ := 2 * x - y

def concatenate (a b c : ℤ) : ℤ := a * 100000 + b * 1000 + c

theorem password_generation (x y : ℤ) (h1 : x = 10) (h2 : y = 10) :
  concatenate (factor1 x) (factor2 x y) (factor3 x y) = 803010 :=
by sorry

end NUMINAMATH_CALUDE_password_generation_l653_65328


namespace NUMINAMATH_CALUDE_number_of_boys_who_love_marbles_l653_65342

def total_marbles : ℕ := 35
def marbles_per_boy : ℕ := 7

theorem number_of_boys_who_love_marbles : 
  total_marbles / marbles_per_boy = 5 := by
  sorry

end NUMINAMATH_CALUDE_number_of_boys_who_love_marbles_l653_65342


namespace NUMINAMATH_CALUDE_beach_population_evening_l653_65349

/-- The number of people at the beach in the evening -/
def beach_population (initial : ℕ) (joined : ℕ) (left : ℕ) : ℕ :=
  initial + joined - left

/-- Theorem stating the total number of people at the beach in the evening -/
theorem beach_population_evening :
  beach_population 3 100 40 = 63 := by
  sorry

end NUMINAMATH_CALUDE_beach_population_evening_l653_65349


namespace NUMINAMATH_CALUDE_box_weights_sum_l653_65390

theorem box_weights_sum (heavy_box light_box sum : ℚ) : 
  heavy_box = 14/15 → 
  light_box = heavy_box - 1/10 → 
  sum = heavy_box + light_box → 
  sum = 53/30 := by sorry

end NUMINAMATH_CALUDE_box_weights_sum_l653_65390


namespace NUMINAMATH_CALUDE_rectangle_area_l653_65359

-- Define the rectangle
def Rectangle (length width : ℝ) : Prop :=
  width = (2/3) * length ∧ 2 * (length + width) = 148

-- State the theorem
theorem rectangle_area (l w : ℝ) (h : Rectangle l w) : l * w = 1314.24 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l653_65359


namespace NUMINAMATH_CALUDE_least_four_digit_multiple_l653_65391

theorem least_four_digit_multiple : ∃ n : ℕ, 
  (n ≥ 1000 ∧ n < 10000) ∧ 
  3 ∣ n ∧ 5 ∣ n ∧ 7 ∣ n ∧
  (∀ m : ℕ, m ≥ 1000 ∧ m < 10000 ∧ 3 ∣ m ∧ 5 ∣ m ∧ 7 ∣ m → n ≤ m) ∧
  n = 1050 :=
by sorry

end NUMINAMATH_CALUDE_least_four_digit_multiple_l653_65391


namespace NUMINAMATH_CALUDE_inequality_preservation_l653_65365

theorem inequality_preservation (a b c : ℝ) : a > b → a - c > b - c := by
  sorry

end NUMINAMATH_CALUDE_inequality_preservation_l653_65365


namespace NUMINAMATH_CALUDE_min_value_expression_l653_65374

theorem min_value_expression (x : ℝ) :
  Real.sqrt (x^2 - 2 * Real.sqrt 3 * abs x + 4) +
  Real.sqrt (x^2 + 2 * Real.sqrt 3 * abs x + 12) ≥ 2 * Real.sqrt 7 ∧
  (Real.sqrt (x^2 - 2 * Real.sqrt 3 * abs x + 4) +
   Real.sqrt (x^2 + 2 * Real.sqrt 3 * abs x + 12) = 2 * Real.sqrt 7 ↔
   x = Real.sqrt 3 / 2 ∨ x = -Real.sqrt 3 / 2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l653_65374


namespace NUMINAMATH_CALUDE_expression_simplification_l653_65367

theorem expression_simplification (a : ℝ) (h : a = Real.sqrt 2 - 2) :
  a^2 / (a^2 + 2*a) - (a^2 - 2*a + 1) / (a + 2) / ((a^2 - 1) / (a + 1)) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l653_65367


namespace NUMINAMATH_CALUDE_number_exists_l653_65393

theorem number_exists : ∃ x : ℝ, (2/3 * x)^3 - 10 = 14 := by
  sorry

end NUMINAMATH_CALUDE_number_exists_l653_65393


namespace NUMINAMATH_CALUDE_coupon_savings_difference_l653_65302

/-- Represents the savings from Coupon A (20% off total price) -/
def savingsA (price : ℝ) : ℝ := 0.2 * price

/-- Represents the savings from Coupon B (flat $40 discount) -/
def savingsB : ℝ := 40

/-- Represents the savings from Coupon C (30% off amount exceeding $120) -/
def savingsC (price : ℝ) : ℝ := 0.3 * (price - 120)

/-- Theorem stating the difference between max and min prices where Coupon A is optimal -/
theorem coupon_savings_difference (minPrice maxPrice : ℝ) : 
  (minPrice > 120) →
  (maxPrice > 120) →
  (∀ p : ℝ, minPrice ≤ p → p ≤ maxPrice → 
    savingsA p ≥ max (savingsB) (savingsC p)) →
  (∃ p : ℝ, p > maxPrice → 
    savingsA p < max (savingsB) (savingsC p)) →
  (∃ p : ℝ, p < minPrice → p > 120 → 
    savingsA p < max (savingsB) (savingsC p)) →
  maxPrice - minPrice = 160 := by
sorry

end NUMINAMATH_CALUDE_coupon_savings_difference_l653_65302


namespace NUMINAMATH_CALUDE_x_is_twenty_percent_greater_than_80_l653_65346

/-- If x is 20 percent greater than 80, then x equals 96. -/
theorem x_is_twenty_percent_greater_than_80 : ∀ x : ℝ, x = 80 * (1 + 20 / 100) → x = 96 :=
by
  sorry

end NUMINAMATH_CALUDE_x_is_twenty_percent_greater_than_80_l653_65346


namespace NUMINAMATH_CALUDE_set_union_problem_l653_65380

theorem set_union_problem (M N : Set ℕ) (x : ℕ) :
  M = {0, x} →
  N = {1, 2} →
  M ∩ N = {2} →
  M ∪ N = {0, 1, 2} := by
sorry

end NUMINAMATH_CALUDE_set_union_problem_l653_65380


namespace NUMINAMATH_CALUDE_john_biking_distance_l653_65306

/-- Converts a base-7 number to base 10 --/
def base7ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (7 ^ i)) 0

/-- The problem statement --/
theorem john_biking_distance :
  base7ToBase10 [2, 5, 6, 3] = 1360 := by
  sorry

end NUMINAMATH_CALUDE_john_biking_distance_l653_65306


namespace NUMINAMATH_CALUDE_circle_equation_constant_l653_65382

theorem circle_equation_constant (F : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 - 4*x + 8*y + F = 0 → 
    ∃ h k : ℝ, ∀ x' y' : ℝ, (x' - h)^2 + (y' - k)^2 = 4^2 → 
      x'^2 + y'^2 - 4*x' + 8*y' + F = 0) → 
  F = 4 := by
sorry

end NUMINAMATH_CALUDE_circle_equation_constant_l653_65382
