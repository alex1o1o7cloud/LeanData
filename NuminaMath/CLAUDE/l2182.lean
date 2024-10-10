import Mathlib

namespace increasing_quadratic_condition_l2182_218224

/-- If f(x) = -x^2 + 2ax - 3 is increasing on (-∞, 4), then a < 4 -/
theorem increasing_quadratic_condition (a : ℝ) : 
  (∀ x < 4, Monotone (fun x => -x^2 + 2*a*x - 3)) → a < 4 := by
  sorry

end increasing_quadratic_condition_l2182_218224


namespace line_segment_endpoint_l2182_218269

/-- Given a line segment from (2,2) to (x,6) with length 5 and x > 0, prove x = 5 -/
theorem line_segment_endpoint (x : ℝ) 
  (h1 : (x - 2)^2 + (6 - 2)^2 = 5^2) 
  (h2 : x > 0) : 
  x = 5 := by
sorry

end line_segment_endpoint_l2182_218269


namespace no_prime_sum_56_l2182_218223

-- Define what it means for a number to be prime
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the property we want to prove
theorem no_prime_sum_56 : ¬∃ (p q : ℕ), isPrime p ∧ isPrime q ∧ p + q = 56 := by
  sorry

end no_prime_sum_56_l2182_218223


namespace trigonometric_identities_l2182_218242

theorem trigonometric_identities (x : Real) (h : Real.tan x = 2) :
  (2/3 * Real.sin x^2 + 1/4 * Real.cos x^2 = 7/12) ∧
  (2 * Real.sin x^2 - Real.sin x * Real.cos x + Real.cos x^2 = 7/5) := by
  sorry

end trigonometric_identities_l2182_218242


namespace range_of_a_l2182_218281

/-- Given two statements p and q, where p: x^2 - 8x - 20 < 0 and q: x^2 - 2x + 1 - a^2 ≤ 0 with a > 0,
    and ¬p is a necessary but not sufficient condition for ¬q,
    prove that the range of values for the real number a is [9, +∞). -/
theorem range_of_a (p q : ℝ → Prop) (a : ℝ) 
    (hp : ∀ x, p x ↔ x^2 - 8*x - 20 < 0)
    (hq : ∀ x, q x ↔ x^2 - 2*x + 1 - a^2 ≤ 0)
    (ha : a > 0)
    (hnec : ∀ x, ¬(p x) → ¬(q x))
    (hnsuff : ∃ x, ¬(q x) ∧ p x) :
  a ≥ 9 := by
  sorry

#check range_of_a

end range_of_a_l2182_218281


namespace quadratic_inequality_l2182_218258

/-- A quadratic function f(x) = ax^2 + bx + c where the solution set of f(x) ≥ 0 is [-1, 4] -/
def QuadraticFunction (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

theorem quadratic_inequality (a b c : ℝ) (h1 : a ≠ 0) 
  (h2 : Set.Icc (-1 : ℝ) 4 = {x | QuadraticFunction a b c x ≥ 0}) :
  QuadraticFunction a b c 2 > QuadraticFunction a b c 3 ∧ 
  QuadraticFunction a b c 3 > QuadraticFunction a b c (-1/2) := by
  sorry

end quadratic_inequality_l2182_218258


namespace problem_solution_l2182_218241

theorem problem_solution (a b : ℝ) 
  (h1 : 2 + a = 5 - b) 
  (h2 : 5 + b = 8 + a) : 
  2 - a = 2 := by
sorry

end problem_solution_l2182_218241


namespace image_of_one_three_l2182_218243

/-- A set of ordered pairs of real numbers -/
def RealPair : Type := ℝ × ℝ

/-- The mapping f: A → B -/
def f (p : RealPair) : RealPair :=
  (p.1 - p.2, p.1 + p.2)

/-- Theorem: The image of (1, 3) under f is (-2, 4) -/
theorem image_of_one_three :
  f (1, 3) = (-2, 4) := by
  sorry

end image_of_one_three_l2182_218243


namespace expression_evaluation_l2182_218251

theorem expression_evaluation :
  let x : ℝ := 1
  let y : ℝ := Real.sqrt 2
  (x + 2*y)^2 - x*(x + 4*y) + (1 - y)*(1 + y) = 7 := by
sorry

end expression_evaluation_l2182_218251


namespace cubic_minus_linear_factorization_l2182_218252

theorem cubic_minus_linear_factorization (m : ℝ) : m^3 - m = m * (m + 1) * (m - 1) := by
  sorry

end cubic_minus_linear_factorization_l2182_218252


namespace dealers_dishonesty_percentage_l2182_218267

/-- The dealer's percentage of dishonesty in terms of weight -/
theorem dealers_dishonesty_percentage
  (standard_weight : ℝ)
  (dealer_weight : ℝ)
  (h1 : standard_weight = 16)
  (h2 : dealer_weight = 14.8) :
  (standard_weight - dealer_weight) / standard_weight * 100 = 7.5 := by
sorry

end dealers_dishonesty_percentage_l2182_218267


namespace cube_volume_from_space_diagonal_l2182_218228

theorem cube_volume_from_space_diagonal (d : ℝ) (h : d = 6 * Real.sqrt 3) :
  ∃ (s : ℝ), s > 0 ∧ s * Real.sqrt 3 = d ∧ s^3 = 216 := by
  sorry

end cube_volume_from_space_diagonal_l2182_218228


namespace x_power_8000_minus_inverse_l2182_218278

theorem x_power_8000_minus_inverse (x : ℂ) : 
  x - 1/x = 2*Complex.I → x^8000 - 1/x^8000 = 0 := by sorry

end x_power_8000_minus_inverse_l2182_218278


namespace solve_average_height_l2182_218276

def average_height_problem (n : ℕ) (initial_average : ℝ) (incorrect_height : ℝ) (correct_height : ℝ) : Prop :=
  let total_incorrect := n * initial_average
  let height_difference := incorrect_height - correct_height
  let total_correct := total_incorrect - height_difference
  let actual_average := total_correct / n
  actual_average = 174.25

theorem solve_average_height :
  average_height_problem 20 175 151 136 := by sorry

end solve_average_height_l2182_218276


namespace expected_participants_l2182_218215

/-- The expected number of participants in a school clean-up event after three years,
    given an initial number of participants and an annual increase rate. -/
theorem expected_participants (initial : ℕ) (increase_rate : ℚ) :
  initial = 800 →
  increase_rate = 1/2 →
  (initial * (1 + increase_rate)^3 : ℚ) = 2700 := by
  sorry

end expected_participants_l2182_218215


namespace x_minus_y_equals_60_l2182_218261

theorem x_minus_y_equals_60 (x y : ℤ) (h1 : x + y = 14) (h2 : x = 37) : x - y = 60 := by
  sorry

end x_minus_y_equals_60_l2182_218261


namespace wickets_before_last_match_is_85_l2182_218201

/-- Represents a bowler's statistics -/
structure BowlerStats where
  initialAverage : ℝ
  lastMatchWickets : ℕ
  lastMatchRuns : ℕ
  averageDecrease : ℝ

/-- Calculates the number of wickets taken before the last match -/
def wicketsBeforeLastMatch (stats : BowlerStats) : ℕ :=
  sorry

/-- Theorem stating that for the given conditions, the number of wickets before the last match is 85 -/
theorem wickets_before_last_match_is_85 (stats : BowlerStats) 
  (h1 : stats.initialAverage = 12.4)
  (h2 : stats.lastMatchWickets = 5)
  (h3 : stats.lastMatchRuns = 26)
  (h4 : stats.averageDecrease = 0.4) :
  wicketsBeforeLastMatch stats = 85 :=
by sorry

end wickets_before_last_match_is_85_l2182_218201


namespace parentheses_removal_l2182_218234

theorem parentheses_removal (a b : ℝ) : -(-a + b - 1) = a - b + 1 := by
  sorry

end parentheses_removal_l2182_218234


namespace absolute_difference_of_xy_l2182_218235

theorem absolute_difference_of_xy (x y : ℝ) 
  (h1 : x * y = 6) 
  (h2 : x + y = 7) : 
  |x - y| = 5 := by
sorry

end absolute_difference_of_xy_l2182_218235


namespace exponent_equation_l2182_218245

theorem exponent_equation (a : ℝ) (m : ℝ) (h : a ≠ 0) : 
  a^(m + 1) * a^(2*m - 1) = a^9 → m = 3 := by
  sorry

end exponent_equation_l2182_218245


namespace lg_expression_equals_one_l2182_218257

-- Define the logarithm function (base 10)
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem lg_expression_equals_one :
  lg 2 * lg 5 + (lg 5)^2 + lg 2 = 1 := by sorry

end lg_expression_equals_one_l2182_218257


namespace circle_a_properties_l2182_218249

/-- Circle A with center (m, 2/m) passing through origin -/
def CircleA (m : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - m)^2 + (p.2 - 2/m)^2 = m^2 + 4/m^2}

/-- Line l: 2x + y - 4 = 0 -/
def LineL : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 2 * p.1 + p.2 - 4 = 0}

theorem circle_a_properties (m : ℝ) (hm : m > 0) :
  /- When m = 2, the circle equation is (x-2)² + (y-1)² = 5 -/
  (∀ p : ℝ × ℝ, p ∈ CircleA 2 ↔ (p.1 - 2)^2 + (p.2 - 1)^2 = 5) ∧
  /- The area of triangle OBC is constant and equal to 4 -/
  (∃ B C : ℝ × ℝ, B ∈ CircleA m ∧ C ∈ CircleA m ∧ B.2 = 0 ∧ C.1 = 0 ∧
    abs (B.1 * C.2) / 2 = 4) ∧
  /- If line l intersects circle A at P and Q where |OP| = |OQ|, then |PQ| = 4√30/5 -/
  (∃ P Q : ℝ × ℝ, P ∈ CircleA 2 ∧ Q ∈ CircleA 2 ∧ P ∈ LineL ∧ Q ∈ LineL ∧
    P.1^2 + P.2^2 = Q.1^2 + Q.2^2 →
    (P.1 - Q.1)^2 + (P.2 - Q.2)^2 = (4 * Real.sqrt 30 / 5)^2) :=
by sorry


end circle_a_properties_l2182_218249


namespace max_inverse_sum_l2182_218283

theorem max_inverse_sum (x y a b : ℝ) 
  (ha : a > 1) 
  (hb : b > 1) 
  (hax : a^x = 2) 
  (hby : b^y = 2) 
  (hab : 2*a + b = 8) : 
  (∀ w z : ℝ, a^w = 2 → b^z = 2 → 1/w + 1/z ≤ 3) ∧ 
  (∃ w z : ℝ, a^w = 2 ∧ b^z = 2 ∧ 1/w + 1/z = 3) :=
sorry

end max_inverse_sum_l2182_218283


namespace luke_received_21_dollars_l2182_218279

/-- Calculates the amount of money Luke received from his mom -/
def money_from_mom (initial amount_spent final : ℕ) : ℕ :=
  final - (initial - amount_spent)

/-- Proves that Luke received 21 dollars from his mom -/
theorem luke_received_21_dollars :
  money_from_mom 48 11 58 = 21 := by
  sorry

end luke_received_21_dollars_l2182_218279


namespace sum_of_bases_equals_1657_l2182_218296

/-- Converts a number from base 13 to base 10 -/
def base13ToBase10 (a b c : ℕ) : ℕ := a * 13^2 + b * 13 + c

/-- Converts a number from base 14 to base 10 -/
def base14ToBase10 (a b c : ℕ) : ℕ := a * 14^2 + b * 14 + c

/-- The value of digit C in base 14 -/
def C : ℕ := 12

theorem sum_of_bases_equals_1657 :
  base13ToBase10 4 2 0 + base14ToBase10 4 C 3 = 1657 := by sorry

end sum_of_bases_equals_1657_l2182_218296


namespace matrix_product_zero_l2182_218237

variable {R : Type*} [CommRing R]

def A (d e : R) : Matrix (Fin 3) (Fin 3) R :=
  ![![0, d, -e],
    ![-d, 0, d],
    ![e, -d, 0]]

def B (d e : R) : Matrix (Fin 3) (Fin 3) R :=
  ![![d * d, d * e, d * d],
    ![d * e, e * e, e * d],
    ![d * d, e * d, d * d]]

theorem matrix_product_zero (d e : R) (h1 : d = e) :
  A d e * B d e = 0 := by
  sorry

end matrix_product_zero_l2182_218237


namespace prime_square_sum_equation_l2182_218299

theorem prime_square_sum_equation :
  ∃ (p q r : ℕ), Prime p ∧ Prime q ∧ Prime r ∧ p^2 + 1 = q^2 + r^2 := by
  sorry

end prime_square_sum_equation_l2182_218299


namespace point_on_hyperbola_l2182_218209

/-- Given a hyperbola y = k/x where the point (3, -2) lies on it, 
    prove that the point (-2, 3) also lies on the same hyperbola. -/
theorem point_on_hyperbola (k : ℝ) : 
  (∃ k, k = 3 * (-2) ∧ -2 = k / 3) → 
  (∃ k, k = (-2) * 3 ∧ 3 = k / (-2)) := by
  sorry

end point_on_hyperbola_l2182_218209


namespace cubic_equation_roots_l2182_218211

theorem cubic_equation_roots : ∃ (x₁ x₂ x₃ : ℚ),
  (x₁ = 3/2 ∧ x₂ = 1/2 ∧ x₃ = -5/2) ∧
  (8 * x₁^3 + 4 * x₁^2 - 34 * x₁ + 15 = 0) ∧
  (8 * x₂^3 + 4 * x₂^2 - 34 * x₂ + 15 = 0) ∧
  (8 * x₃^3 + 4 * x₃^2 - 34 * x₃ + 15 = 0) ∧
  (2 * x₁ - 4 * x₂ = 1) := by
  sorry

#check cubic_equation_roots

end cubic_equation_roots_l2182_218211


namespace first_super_lucky_year_l2182_218272

def is_valid_date (month day year : ℕ) : Prop :=
  1 ≤ month ∧ month ≤ 12 ∧ 1 ≤ day ∧ day ≤ 31 ∧ year > 2000

def is_super_lucky_date (month day year : ℕ) : Prop :=
  is_valid_date month day year ∧ month * day = year % 100

def has_two_super_lucky_dates (year : ℕ) : Prop :=
  ∃ (m1 d1 m2 d2 : ℕ), 
    is_super_lucky_date m1 d1 year ∧ 
    is_super_lucky_date m2 d2 year ∧ 
    (m1 ≠ m2 ∨ d1 ≠ d2)

theorem first_super_lucky_year : 
  (∀ y, 2000 < y ∧ y < 2004 → ¬ has_two_super_lucky_dates y) ∧ 
  has_two_super_lucky_dates 2004 :=
sorry

end first_super_lucky_year_l2182_218272


namespace xyz_sum_l2182_218200

theorem xyz_sum (x y z : ℕ+) 
  (h1 : x * y + z = 47)
  (h2 : y * z + x = 47)
  (h3 : z * x + y = 47) : 
  x + y + z = 48 := by
sorry

end xyz_sum_l2182_218200


namespace arithmetic_sequence_sum_relation_l2182_218240

/-- Prove that for an arithmetic sequence, R = 2n²d -/
theorem arithmetic_sequence_sum_relation 
  (a d n : ℝ) 
  (S₁ : ℝ := n / 2 * (2 * a + (n - 1) * d))
  (S₂ : ℝ := n * (2 * a + (2 * n - 1) * d))
  (S₃ : ℝ := 3 * n / 2 * (2 * a + (3 * n - 1) * d))
  (R : ℝ := S₃ - S₂ - S₁) :
  R = 2 * n^2 * d := by
sorry

end arithmetic_sequence_sum_relation_l2182_218240


namespace expand_expression_l2182_218256

theorem expand_expression (x : ℝ) : 16 * (2 * x + 5) = 32 * x + 80 := by
  sorry

end expand_expression_l2182_218256


namespace fraction_modification_l2182_218265

theorem fraction_modification (x : ℚ) : 
  x = 437 → (537 - x) / (463 + x) = 1/9 := by
sorry

end fraction_modification_l2182_218265


namespace diophantine_equation_solutions_l2182_218220

theorem diophantine_equation_solutions :
  ∀ (a b : ℕ), (2017 : ℕ) ^ a = b ^ 6 - 32 * b + 1 ↔ (a = 0 ∧ b = 0) ∨ (a = 0 ∧ b = 2) :=
by sorry

end diophantine_equation_solutions_l2182_218220


namespace tangent_identity_l2182_218294

theorem tangent_identity (α β γ : Real) (h : α + β + γ = π/4) :
  (1 + Real.tan α) * (1 + Real.tan β) * (1 + Real.tan γ) / (1 + Real.tan α * Real.tan β * Real.tan γ) = 2 := by
  sorry

end tangent_identity_l2182_218294


namespace max_distance_for_specific_bicycle_l2182_218277

/-- Represents a bicycle with swappable tires -/
structure Bicycle where
  front_tire_life : ℝ
  rear_tire_life : ℝ

/-- Calculates the maximum distance a bicycle can travel with tire swapping -/
def max_distance (b : Bicycle) : ℝ :=
  sorry

/-- Theorem stating the maximum distance for a specific bicycle -/
theorem max_distance_for_specific_bicycle :
  let b : Bicycle := { front_tire_life := 5000, rear_tire_life := 3000 }
  max_distance b = 3750 := by
  sorry

end max_distance_for_specific_bicycle_l2182_218277


namespace tooth_arrangements_l2182_218247

def word_length : Nat := 5
def repeated_letter_count : Nat := 2

theorem tooth_arrangements : 
  (word_length.factorial) / (repeated_letter_count.factorial * repeated_letter_count.factorial) = 30 := by
  sorry

end tooth_arrangements_l2182_218247


namespace trig_expression_value_l2182_218293

theorem trig_expression_value (α : Real) (h : Real.tan α = 2) :
  (Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) + Real.cos α ^ 2 = 16/5 := by
  sorry

end trig_expression_value_l2182_218293


namespace hotdog_competition_ratio_l2182_218260

/-- Hotdog eating competition rates and ratios -/
theorem hotdog_competition_ratio :
  let first_rate := 10 -- hot dogs per minute
  let second_rate := 3 * first_rate
  let third_rate := 300 / 5 -- 300 hot dogs in 5 minutes
  third_rate / second_rate = 2 := by
  sorry

end hotdog_competition_ratio_l2182_218260


namespace cubic_polynomial_inequality_iff_coeff_conditions_l2182_218227

/-- A cubic polynomial -/
structure CubicPolynomial where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Evaluation of a cubic polynomial at a point -/
def eval (p : CubicPolynomial) (x : ℝ) : ℝ :=
  p.a * x^3 + p.b * x^2 + p.c * x + p.d

/-- The inequality condition for the polynomial -/
def satisfiesInequality (p : CubicPolynomial) : Prop :=
  ∀ x y, x ≥ 0 → y ≥ 0 → eval p (x + y) ≥ eval p x + eval p y

/-- The conditions on the coefficients -/
def satisfiesCoeffConditions (p : CubicPolynomial) : Prop :=
  p.a > 0 ∧ p.d ≤ 0 ∧ 8 * p.b^3 ≥ 243 * p.a^2 * p.d

theorem cubic_polynomial_inequality_iff_coeff_conditions (p : CubicPolynomial) :
  satisfiesInequality p ↔ satisfiesCoeffConditions p := by sorry

end cubic_polynomial_inequality_iff_coeff_conditions_l2182_218227


namespace min_value_of_expression_l2182_218202

theorem min_value_of_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2 * a + 3 * b = 1) :
  (1 / a + 3 / b) ≥ 15 := by
  sorry

end min_value_of_expression_l2182_218202


namespace max_abs_u_l2182_218246

/-- Given a complex number z with |z| = 1, prove that the maximum value of 
    |z^4 - z^3 - 3z^2i - z + 1| is 5 and occurs when z = -1 -/
theorem max_abs_u (z : ℂ) (h : Complex.abs z = 1) :
  ∃ (max : ℝ), max = 5 ∧
  Complex.abs (z^4 - z^3 - 3*z^2*Complex.I - z + 1) ≤ max ∧
  Complex.abs ((-1 : ℂ)^4 - (-1 : ℂ)^3 - 3*(-1 : ℂ)^2*Complex.I - (-1 : ℂ) + 1) = max :=
sorry

end max_abs_u_l2182_218246


namespace sequential_discount_equivalence_l2182_218222

/-- The equivalent single discount percentage for two sequential discounts -/
def equivalent_discount (first_discount second_discount : ℝ) : ℝ :=
  1 - (1 - first_discount) * (1 - second_discount)

/-- Theorem stating that a 15% discount followed by a 25% discount 
    is equivalent to a single 36.25% discount -/
theorem sequential_discount_equivalence : 
  equivalent_discount 0.15 0.25 = 0.3625 := by
  sorry

#eval equivalent_discount 0.15 0.25

end sequential_discount_equivalence_l2182_218222


namespace linear_function_characterization_l2182_218213

/-- A function satisfying the Cauchy functional equation -/
def is_additive (f : ℝ → ℝ) : Prop :=
  ∀ x y, f (x + y) = f x + f y

/-- A monotonic function -/
def is_monotonic (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ≤ y → f x ≤ f y

/-- A function bounded between 0 and 1 -/
def is_bounded_01 (f : ℝ → ℝ) : Prop :=
  ∀ x, 0 ≤ f x ∧ f x ≤ 1

/-- Main theorem: If f satisfies the given conditions, then it is linear -/
theorem linear_function_characterization (f : ℝ → ℝ)
  (h_additive : is_additive f)
  (h_monotonic : is_monotonic f)
  (h_bounded : is_bounded_01 f) :
  ∀ x, f x = x * f 1 := by
  sorry

end linear_function_characterization_l2182_218213


namespace tian_ji_win_probability_l2182_218206

/-- Represents the tiers of horses -/
inductive Tier
| Top
| Middle
| Bottom

/-- Represents a horse with its owner and tier -/
structure Horse :=
  (owner : String)
  (tier : Tier)

/-- Determines if one horse is better than another -/
def isBetter (h1 h2 : Horse) : Prop := sorry

/-- The set of all horses in the competition -/
def allHorses : Finset Horse := sorry

/-- The set of Tian Ji's horses -/
def tianJiHorses : Finset Horse := sorry

/-- The set of King Qi's horses -/
def kingQiHorses : Finset Horse := sorry

/-- Axioms representing the given conditions -/
axiom horse_count : (tianJiHorses.card = 3) ∧ (kingQiHorses.card = 3)
axiom tian_ji_top_vs_qi_middle : 
  ∃ (ht hm : Horse), ht ∈ tianJiHorses ∧ hm ∈ kingQiHorses ∧ 
  ht.tier = Tier.Top ∧ hm.tier = Tier.Middle ∧ isBetter ht hm
axiom tian_ji_top_vs_qi_top : 
  ∃ (ht1 ht2 : Horse), ht1 ∈ tianJiHorses ∧ ht2 ∈ kingQiHorses ∧ 
  ht1.tier = Tier.Top ∧ ht2.tier = Tier.Top ∧ isBetter ht2 ht1
axiom tian_ji_middle_vs_qi_bottom : 
  ∃ (hm hb : Horse), hm ∈ tianJiHorses ∧ hb ∈ kingQiHorses ∧ 
  hm.tier = Tier.Middle ∧ hb.tier = Tier.Bottom ∧ isBetter hm hb
axiom tian_ji_middle_vs_qi_middle : 
  ∃ (hm1 hm2 : Horse), hm1 ∈ tianJiHorses ∧ hm2 ∈ kingQiHorses ∧ 
  hm1.tier = Tier.Middle ∧ hm2.tier = Tier.Middle ∧ isBetter hm2 hm1
axiom tian_ji_bottom_vs_qi_bottom : 
  ∃ (hb1 hb2 : Horse), hb1 ∈ tianJiHorses ∧ hb2 ∈ kingQiHorses ∧ 
  hb1.tier = Tier.Bottom ∧ hb2.tier = Tier.Bottom ∧ isBetter hb2 hb1

/-- The probability of Tian Ji's horse winning in a random matchup -/
def tianJiWinProbability : ℚ := sorry

/-- Main theorem: The probability of Tian Ji's horse winning is 1/3 -/
theorem tian_ji_win_probability : tianJiWinProbability = 1/3 := by sorry

end tian_ji_win_probability_l2182_218206


namespace gcd_of_squares_l2182_218254

theorem gcd_of_squares : Nat.gcd (114^2 + 226^2 + 338^2) (113^2 + 225^2 + 339^2) = 1 := by
  sorry

end gcd_of_squares_l2182_218254


namespace inequality_proof_l2182_218207

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : 1/a + 1/b + 1/c ≥ a + b + c) : a + b + c ≥ 3*a*b*c := by
  sorry

end inequality_proof_l2182_218207


namespace laptop_savings_l2182_218275

/-- The in-store price of the laptop in dollars -/
def in_store_price : ℚ := 299.99

/-- The cost of one payment in the radio offer in dollars -/
def radio_payment : ℚ := 55.98

/-- The number of payments in the radio offer -/
def num_payments : ℕ := 5

/-- The shipping and handling charge in dollars -/
def shipping_charge : ℚ := 12.99

/-- The number of cents in a dollar -/
def cents_per_dollar : ℕ := 100

theorem laptop_savings : 
  (in_store_price - (radio_payment * num_payments + shipping_charge)) * cents_per_dollar = 710 := by
  sorry

end laptop_savings_l2182_218275


namespace factorization_equality_l2182_218297

theorem factorization_equality (x : ℝ) : 45 * x^2 + 135 * x = 45 * x * (x + 3) := by
  sorry

end factorization_equality_l2182_218297


namespace carbonate_weight_proof_l2182_218216

/-- The molecular weight of the carbonate part in Al2(CO3)3 -/
def carbonate_weight (total_weight : ℝ) (al_weight : ℝ) : ℝ :=
  total_weight - 2 * al_weight

/-- Proof that the molecular weight of the carbonate part in Al2(CO3)3 is 180.04 g/mol -/
theorem carbonate_weight_proof (total_weight : ℝ) (al_weight : ℝ) 
  (h1 : total_weight = 234)
  (h2 : al_weight = 26.98) :
  carbonate_weight total_weight al_weight = 180.04 := by
  sorry

#eval carbonate_weight 234 26.98

end carbonate_weight_proof_l2182_218216


namespace roots_of_quartic_equation_l2182_218232

theorem roots_of_quartic_equation :
  let f : ℝ → ℝ := λ x => 7 * x^4 - 44 * x^3 + 78 * x^2 - 44 * x + 7
  ∃ (a b c d : ℝ), 
    (∀ x : ℝ, f x = 0 ↔ x = a ∨ x = b ∨ x = c ∨ x = d) ∧
    a = 2 ∧ 
    b = 1/2 ∧ 
    c = (8 + Real.sqrt 15) / 7 ∧ 
    d = (8 - Real.sqrt 15) / 7 :=
by sorry

end roots_of_quartic_equation_l2182_218232


namespace cube_difference_l2182_218298

theorem cube_difference (a b : ℝ) (h1 : a - b = 3) (h2 : a^2 + b^2 = 27) : 
  a^3 - b^3 = 108 := by
sorry

end cube_difference_l2182_218298


namespace zoe_strawberry_count_l2182_218270

/-- The number of strawberries Zoe ate -/
def num_strawberries : ℕ := sorry

/-- The number of ounces of yogurt Zoe ate -/
def yogurt_ounces : ℕ := 6

/-- Calories per strawberry -/
def calories_per_strawberry : ℕ := 4

/-- Calories per ounce of yogurt -/
def calories_per_yogurt_ounce : ℕ := 17

/-- Total calories consumed -/
def total_calories : ℕ := 150

theorem zoe_strawberry_count :
  num_strawberries * calories_per_strawberry +
  yogurt_ounces * calories_per_yogurt_ounce = total_calories ∧
  num_strawberries = 12 := by sorry

end zoe_strawberry_count_l2182_218270


namespace missed_solution_l2182_218286

theorem missed_solution (x : ℝ) : x * (x - 3) = x - 3 → (x = 1 ∨ x = 3) := by
  sorry

end missed_solution_l2182_218286


namespace f_is_odd_l2182_218204

def f (x : ℝ) : ℝ := |x + 1| - |x - 1|

theorem f_is_odd : ∀ x : ℝ, f (-x) = -f x := by
  sorry

end f_is_odd_l2182_218204


namespace fruit_arrangement_theorem_l2182_218225

def number_of_arrangements (total : ℕ) (group1 : ℕ) (group2 : ℕ) (unique : ℕ) : ℕ :=
  Nat.factorial total / (Nat.factorial group1 * Nat.factorial group2 * Nat.factorial unique)

theorem fruit_arrangement_theorem :
  number_of_arrangements 7 4 2 1 = 105 := by
  sorry

end fruit_arrangement_theorem_l2182_218225


namespace greatest_multiple_of_5_and_6_less_than_1000_l2182_218253

theorem greatest_multiple_of_5_and_6_less_than_1000 : ∃ n : ℕ, 
  n = 990 ∧ 
  n % 5 = 0 ∧ 
  n % 6 = 0 ∧ 
  n < 1000 ∧ 
  ∀ m : ℕ, m % 5 = 0 ∧ m % 6 = 0 ∧ m < 1000 → m ≤ n :=
by sorry

end greatest_multiple_of_5_and_6_less_than_1000_l2182_218253


namespace age_difference_l2182_218280

/-- Given the ages of four people with specific relationships, prove that Jack's age is 5 years more than twice Shannen's age. -/
theorem age_difference (beckett_age olaf_age shannen_age jack_age : ℕ) : 
  beckett_age = 12 →
  olaf_age = beckett_age + 3 →
  shannen_age = olaf_age - 2 →
  jack_age > 2 * shannen_age →
  beckett_age + olaf_age + shannen_age + jack_age = 71 →
  jack_age - 2 * shannen_age = 5 := by
sorry

end age_difference_l2182_218280


namespace bananas_removed_l2182_218285

theorem bananas_removed (original : ℕ) (remaining : ℕ) (removed : ℕ)
  (h1 : original = 46)
  (h2 : remaining = 41)
  (h3 : removed = original - remaining) :
  removed = 5 := by
  sorry

end bananas_removed_l2182_218285


namespace shirt_purchase_problem_l2182_218203

theorem shirt_purchase_problem (shirt_price pants_price : ℝ) 
  (num_shirts : ℕ) (num_pants : ℕ) (total_cost refund : ℝ) :
  shirt_price ≠ pants_price →
  shirt_price = 45 →
  num_pants = 3 →
  total_cost = 120 →
  refund = 0.25 * total_cost →
  total_cost = num_shirts * shirt_price + num_pants * pants_price →
  total_cost - refund = num_shirts * shirt_price →
  num_shirts = 2 :=
by
  sorry

#check shirt_purchase_problem

end shirt_purchase_problem_l2182_218203


namespace rationalize_denominator_l2182_218282

theorem rationalize_denominator : 
  7 / Real.sqrt 75 = (7 * Real.sqrt 3) / 15 := by
  sorry

end rationalize_denominator_l2182_218282


namespace zilla_savings_theorem_l2182_218288

/-- Represents Zilla's monthly financial breakdown -/
structure ZillaFinances where
  total_earnings : ℝ
  rent_percentage : ℝ
  rent_amount : ℝ
  other_expenses_percentage : ℝ

/-- Calculates Zilla's savings based on her financial breakdown -/
def calculate_savings (z : ZillaFinances) : ℝ :=
  z.total_earnings - (z.rent_amount + z.total_earnings * z.other_expenses_percentage)

/-- Theorem stating Zilla's savings amount to $817 -/
theorem zilla_savings_theorem (z : ZillaFinances) 
  (h1 : z.rent_percentage = 0.07)
  (h2 : z.other_expenses_percentage = 0.5)
  (h3 : z.rent_amount = 133)
  (h4 : z.rent_amount = z.total_earnings * z.rent_percentage) :
  calculate_savings z = 817 := by
  sorry

#eval calculate_savings { total_earnings := 1900, rent_percentage := 0.07, rent_amount := 133, other_expenses_percentage := 0.5 }

end zilla_savings_theorem_l2182_218288


namespace circumcircle_equation_l2182_218291

-- Define the given circle
def given_circle (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define point P
def point_P : ℝ × ℝ := (4, 2)

-- Define a predicate for points on the circumcircle of triangle ABP
def on_circumcircle (x y : ℝ) : Prop := (x - 2)^2 + (y - 1)^2 = 5

-- Theorem statement
theorem circumcircle_equation :
  ∀ A B : ℝ × ℝ,
  given_circle A.1 A.2 →
  given_circle B.1 B.2 →
  (∃ t : ℝ, A = (4 * t / (t^2 + 1), 2 * t^2 / (t^2 + 1))) →
  (∃ s : ℝ, B = (4 * s / (s^2 + 1), 2 * s^2 / (s^2 + 1))) →
  on_circumcircle A.1 A.2 ∧ on_circumcircle B.1 B.2 ∧ on_circumcircle point_P.1 point_P.2 :=
by sorry

end circumcircle_equation_l2182_218291


namespace S_five_three_l2182_218259

def S (a b : ℝ) : ℝ := 4 * a + 6 * b - 2 * a * b

theorem S_five_three : S 5 3 = 8 := by
  sorry

end S_five_three_l2182_218259


namespace coin_distribution_l2182_218212

theorem coin_distribution (a d : ℤ) : 
  (a - 3*d) + (a - 2*d) = 58 ∧ 
  (a + d) + (a + 2*d) + (a + 3*d) = 60 →
  (a - 2*d = 28 ∧ a = 24) := by
  sorry

end coin_distribution_l2182_218212


namespace subtracted_value_l2182_218289

theorem subtracted_value (N : ℝ) (x : ℝ) : 
  ((N - x) / 7 = 7) ∧ ((N - 4) / 10 = 5) → x = 5 := by
  sorry

end subtracted_value_l2182_218289


namespace minimal_m_value_l2182_218284

theorem minimal_m_value (n k : ℕ) (hn : n > k) (hk : k > 1) :
  let m := (10^n - 1) / (10^k - 1)
  (∀ n' k' : ℕ, n' > k' → k' > 1 → (10^n' - 1) / (10^k' - 1) ≥ m) →
  m = 101 := by
  sorry

end minimal_m_value_l2182_218284


namespace dish_washing_time_l2182_218292

theorem dish_washing_time (dawn_time andy_time : ℕ) : 
  andy_time = 2 * dawn_time + 6 →
  andy_time = 46 →
  dawn_time = 20 := by
sorry

end dish_washing_time_l2182_218292


namespace required_machines_eq_ten_l2182_218239

/-- The number of cell phones produced by 2 machines per minute -/
def phones_per_2machines : ℕ := 10

/-- The number of machines used in the given condition -/
def given_machines : ℕ := 2

/-- The desired number of cell phones to be produced per minute -/
def desired_phones : ℕ := 50

/-- Calculates the number of machines required to produce the desired number of phones per minute -/
def required_machines : ℕ := desired_phones * given_machines / phones_per_2machines

theorem required_machines_eq_ten : required_machines = 10 := by
  sorry

end required_machines_eq_ten_l2182_218239


namespace percentage_problem_l2182_218295

theorem percentage_problem (x : ℝ) (h : 0.40 * x = 160) : 0.20 * x = 80 := by
  sorry

end percentage_problem_l2182_218295


namespace ratio_equality_l2182_218230

theorem ratio_equality : (1722^2 - 1715^2) / (1729^2 - 1708^2) = 1/3 := by
  sorry

end ratio_equality_l2182_218230


namespace smallest_integer_satisfying_inequality_negative_two_satisfies_inequality_smallest_integer_is_negative_two_l2182_218248

theorem smallest_integer_satisfying_inequality :
  ∀ x : ℤ, (3 * x^2 - 4 < 20) → x ≥ -2 :=
by
  sorry

theorem negative_two_satisfies_inequality :
  3 * (-2)^2 - 4 < 20 :=
by
  sorry

theorem smallest_integer_is_negative_two :
  ∃ x : ℤ, (∀ y : ℤ, (3 * y^2 - 4 < 20) → y ≥ x) ∧ (3 * x^2 - 4 < 20) ∧ x = -2 :=
by
  sorry

end smallest_integer_satisfying_inequality_negative_two_satisfies_inequality_smallest_integer_is_negative_two_l2182_218248


namespace rectangle_area_error_l2182_218263

theorem rectangle_area_error (L W : ℝ) (hL : L > 0) (hW : W > 0) :
  let measured_length := 1.15 * L
  let measured_width := 1.20 * W
  let true_area := L * W
  let calculated_area := measured_length * measured_width
  (calculated_area - true_area) / true_area * 100 = 38 := by
sorry

end rectangle_area_error_l2182_218263


namespace population_equality_l2182_218219

/-- The number of years it takes for two villages' populations to be equal -/
def years_until_equal_population (x_initial : ℕ) (x_decrease : ℕ) (y_initial : ℕ) (y_increase : ℕ) : ℕ :=
  18

theorem population_equality (x_initial y_initial x_decrease y_increase : ℕ)
  (h1 : x_initial = 78000)
  (h2 : x_decrease = 1200)
  (h3 : y_initial = 42000)
  (h4 : y_increase = 800) :
  x_initial - x_decrease * (years_until_equal_population x_initial x_decrease y_initial y_increase) =
  y_initial + y_increase * (years_until_equal_population x_initial x_decrease y_initial y_increase) :=
by sorry

end population_equality_l2182_218219


namespace percentage_of_difference_l2182_218214

theorem percentage_of_difference (x y : ℝ) (P : ℝ) :
  P * (x - y) = 0.3 * (x + y) →
  y = (1/3) * x →
  P = 0.6 := by
  sorry

end percentage_of_difference_l2182_218214


namespace not_equivalent_statement_and_converse_l2182_218250

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (subset : Line → Plane → Prop)
variable (perp : Line → Plane → Prop)
variable (perpPlanes : Plane → Plane → Prop)

-- Define the given lines and planes
variable (a b c : Line)
variable (α β : Plane)

-- State the theorem
theorem not_equivalent_statement_and_converse :
  (b ≠ a ∧ c ≠ a ∧ c ≠ b) →  -- three different lines
  (α ≠ β) →  -- two different planes
  (subset b α) →  -- b is a subset of α
  (¬ subset c α) →  -- c is not a subset of α
  ¬ (((perp b β → perpPlanes α β) ↔ (perpPlanes α β → perp b β))) :=
by sorry

end not_equivalent_statement_and_converse_l2182_218250


namespace complement_A_intersect_B_l2182_218217

-- Define the universal set U
def U : Finset Nat := {1, 2, 3, 4, 5}

-- Define set A
def A : Finset Nat := {1, 3, 5}

-- Define set B
def B : Finset Nat := {3, 4}

-- Theorem statement
theorem complement_A_intersect_B :
  (U \ A) ∩ B = {4} := by
  sorry

end complement_A_intersect_B_l2182_218217


namespace negation_of_existence_negation_of_proposition_l2182_218210

theorem negation_of_existence (p : ℝ → Prop) :
  (¬∃ x, p x) ↔ (∀ x, ¬p x) :=
by sorry

theorem negation_of_proposition :
  (¬∃ x : ℝ, x^2 + 2 > 3*x) ↔ (∀ x : ℝ, x^2 + 2 ≤ 3*x) :=
by sorry

end negation_of_existence_negation_of_proposition_l2182_218210


namespace modulus_of_complex_number_l2182_218271

theorem modulus_of_complex_number : 
  let i : ℂ := Complex.I
  let z : ℂ := 2 / (1 + i) + (1 - i)^2
  Complex.abs z = Real.sqrt 10 := by
sorry

end modulus_of_complex_number_l2182_218271


namespace jean_calories_l2182_218274

/-- Calculates the total calories consumed based on pages written and calories per donut -/
def total_calories (pages_written : ℕ) (pages_per_donut : ℕ) (calories_per_donut : ℕ) : ℕ :=
  (pages_written / pages_per_donut) * calories_per_donut

/-- Proves that Jean eats 900 calories given the conditions -/
theorem jean_calories : total_calories 12 2 150 = 900 := by
  sorry

end jean_calories_l2182_218274


namespace negative_fraction_comparison_l2182_218205

theorem negative_fraction_comparison : -2/3 > -3/4 := by
  sorry

end negative_fraction_comparison_l2182_218205


namespace simple_interest_rate_calculation_l2182_218231

theorem simple_interest_rate_calculation (P : ℝ) (P_positive : P > 0) :
  let final_amount := (7 / 6) * P
  let time := 6
  let interest := final_amount - P
  let R := (interest / P / time) * 100
  R = 100 / 36 := by
  sorry

end simple_interest_rate_calculation_l2182_218231


namespace expo_stamps_theorem_l2182_218226

theorem expo_stamps_theorem (total_cost : ℕ) (cost_4 cost_8 : ℕ) (difference : ℕ) :
  total_cost = 660 →
  cost_4 = 4 →
  cost_8 = 8 →
  difference = 30 →
  ∃ (stamps_4 stamps_8 : ℕ),
    stamps_8 = stamps_4 + difference ∧
    total_cost = cost_4 * stamps_4 + cost_8 * stamps_8 →
    stamps_4 + stamps_8 = 100 :=
by sorry

end expo_stamps_theorem_l2182_218226


namespace imaginary_number_properties_l2182_218290

/-- An imaginary number is a complex number with a non-zero imaginary part -/
def IsImaginary (z : ℂ) : Prop := z.im ≠ 0

theorem imaginary_number_properties (x y : ℝ) (z : ℂ) 
  (h1 : z = Complex.mk x y) (h2 : IsImaginary z) : 
  x ∈ Set.univ ∧ y ≠ 0 := by sorry

end imaginary_number_properties_l2182_218290


namespace f_monotonicity_f_two_zeros_l2182_218266

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a * (x + 2)

theorem f_monotonicity :
  let f₁ := f 1
  (∀ x y, x < y → x < 0 → y < 0 → f₁ y < f₁ x) ∧
  (∀ x y, 0 < x → x < y → f₁ x < f₁ y) :=
sorry

theorem f_two_zeros (a : ℝ) :
  (∃ x y, x < y ∧ f a x = 0 ∧ f a y = 0) ↔ (Real.exp (-1) < a) :=
sorry

end f_monotonicity_f_two_zeros_l2182_218266


namespace closest_to_100_l2182_218236

def expression : ℝ := (2.1 * (50.2 + 0.08)) - 5

def options : List ℝ := [95, 100, 101, 105]

theorem closest_to_100 : 
  ∀ x ∈ options, |expression - 100| ≤ |expression - x| :=
sorry

end closest_to_100_l2182_218236


namespace find_m_l2182_218229

theorem find_m : ∃ m : ℝ, ∀ x : ℝ, (x + 2) * (x + 3) = x^2 + m*x + 6 → m = 5 := by
  sorry

end find_m_l2182_218229


namespace ark5_ensures_metabolic_energy_needs_l2182_218287

-- Define the enzyme Ark5
structure Ark5 where
  activity : Bool

-- Define cancer cells
structure CancerCell where
  energy_balanced : Bool
  proliferating : Bool
  alive : Bool

-- Define the effect of Ark5 on cancer cells
def ark5_effect (a : Ark5) (c : CancerCell) : CancerCell :=
  { energy_balanced := a.activity
  , proliferating := true
  , alive := a.activity }

-- Theorem statement
theorem ark5_ensures_metabolic_energy_needs :
  ∀ (a : Ark5) (c : CancerCell),
    (¬a.activity → ¬c.energy_balanced) ∧
    (¬a.activity → c.proliferating) ∧
    (¬a.activity → ¬c.alive) →
    (a.activity → c.energy_balanced) :=
sorry

end ark5_ensures_metabolic_energy_needs_l2182_218287


namespace girls_in_school_l2182_218262

theorem girls_in_school (total_students : ℕ) (sample_size : ℕ) 
  (h1 : total_students = 1600)
  (h2 : sample_size = 200)
  (h3 : ∃ (boys girls : ℕ), boys + girls = sample_size ∧ girls = boys - 10) :
  ∃ (school_girls : ℕ), school_girls = 760 ∧ 
    school_girls * sample_size = total_students * 95 :=
by sorry

end girls_in_school_l2182_218262


namespace egg_plant_theorem_l2182_218264

/-- Represents the egg processing plant scenario --/
structure EggPlant where
  accepted : ℕ
  rejected : ℕ
  total : ℕ
  accepted_to_rejected_ratio : ℚ

/-- The initial state of the egg plant --/
def initial_state : EggPlant := {
  accepted := 0,
  rejected := 0,
  total := 400,
  accepted_to_rejected_ratio := 0
}

/-- The state after additional eggs are accepted --/
def modified_state (initial : EggPlant) : EggPlant := {
  accepted := initial.accepted + 12,
  rejected := initial.rejected - 4,
  total := initial.total,
  accepted_to_rejected_ratio := 99/1
}

/-- The theorem to prove --/
theorem egg_plant_theorem (initial : EggPlant) : 
  initial.accepted = 392 ∧ 
  initial.rejected = 8 ∧
  initial.total = 400 ∧
  (initial.accepted : ℚ) / initial.rejected = (initial.accepted + 12 : ℚ) / (initial.rejected - 4) ∧
  (initial.accepted + 12 : ℚ) / (initial.rejected - 4) = 99/1 := by
  sorry

end egg_plant_theorem_l2182_218264


namespace tan_y_plus_pi_third_l2182_218238

theorem tan_y_plus_pi_third (y : Real) (h : Real.tan y = -3) :
  Real.tan (y + π / 3) = -(5 * Real.sqrt 3 - 6) / 13 := by
  sorry

end tan_y_plus_pi_third_l2182_218238


namespace solutions_to_quadratic_equation_l2182_218221

theorem solutions_to_quadratic_equation :
  ∀ x : ℝ, 3 * x^2 = 27 ↔ x = 3 ∨ x = -3 := by sorry

end solutions_to_quadratic_equation_l2182_218221


namespace f_on_negative_interval_l2182_218268

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def period_two (f : ℝ → ℝ) : Prop := ∀ x, f x = f (x + 2)

theorem f_on_negative_interval 
  (f : ℝ → ℝ) 
  (h_even : is_even f) 
  (h_period : period_two f) 
  (h_interval : ∀ x ∈ Set.Icc 2 3, f x = x) :
  ∀ x ∈ Set.Icc (-1) 0, f x = 2 - x := by
sorry

end f_on_negative_interval_l2182_218268


namespace consecutive_even_sum_l2182_218273

theorem consecutive_even_sum (n : ℤ) : 
  (∃ m : ℤ, m = n + 2 ∧ (m^2 - n^2 = 84)) → n + (n + 2) = 42 := by
  sorry

end consecutive_even_sum_l2182_218273


namespace equal_digging_time_l2182_218244

/-- Given the same number of people, if it takes a certain number of days to dig one volume,
    it will take the same number of days to dig an equal volume. -/
theorem equal_digging_time (people : ℕ) (depth1 length1 breadth1 depth2 length2 breadth2 : ℝ)
  (days : ℝ) (h1 : depth1 * length1 * breadth1 = depth2 * length2 * breadth2)
  (h2 : depth1 = 100) (h3 : length1 = 25) (h4 : breadth1 = 30)
  (h5 : depth2 = 75) (h6 : length2 = 20) (h7 : breadth2 = 50)
  (h8 : days = 12) :
  days = 12 := by
  sorry

#check equal_digging_time

end equal_digging_time_l2182_218244


namespace greatest_number_with_gcd_l2182_218255

theorem greatest_number_with_gcd (X : ℕ) : 
  X ≤ 840 ∧ 
  7 ∣ X ∧ 
  Nat.gcd X 91 = 7 ∧ 
  Nat.gcd X 840 = 7 →
  X = 840 :=
by sorry

end greatest_number_with_gcd_l2182_218255


namespace fundraiser_proof_l2182_218218

/-- The number of students asked to bring brownies -/
def num_brownie_students : ℕ := 30

/-- The number of brownies each student brings -/
def brownies_per_student : ℕ := 12

/-- The number of students asked to bring cookies -/
def num_cookie_students : ℕ := 20

/-- The number of cookies each student brings -/
def cookies_per_student : ℕ := 24

/-- The number of students asked to bring donuts -/
def num_donut_students : ℕ := 15

/-- The number of donuts each student brings -/
def donuts_per_student : ℕ := 12

/-- The price of each item in dollars -/
def price_per_item : ℕ := 2

/-- The total amount raised in dollars -/
def total_amount_raised : ℕ := 2040

theorem fundraiser_proof :
  num_brownie_students * brownies_per_student * price_per_item +
  num_cookie_students * cookies_per_student * price_per_item +
  num_donut_students * donuts_per_student * price_per_item =
  total_amount_raised :=
by sorry

end fundraiser_proof_l2182_218218


namespace round_39_982_to_three_sig_figs_l2182_218208

/-- Rounds a number to a specified number of significant figures -/
def roundToSigFigs (x : ℝ) (n : ℕ) : ℝ := sorry

/-- Checks if a real number has exactly n significant figures -/
def hasSigFigs (x : ℝ) (n : ℕ) : Prop := sorry

theorem round_39_982_to_three_sig_figs :
  let x := 39.982
  let result := roundToSigFigs x 3
  result = 40.0 ∧ hasSigFigs result 3 := by sorry

end round_39_982_to_three_sig_figs_l2182_218208


namespace refrigerator_part_payment_l2182_218233

/-- Given a refrigerator purchase where a part payment of 25% has been made
    and $2625 remains to be paid (representing 75% of the total cost),
    prove that the part payment is equal to $875. -/
theorem refrigerator_part_payment
  (total_cost : ℝ)
  (part_payment_percentage : ℝ)
  (remaining_payment : ℝ)
  (remaining_percentage : ℝ)
  (h1 : part_payment_percentage = 0.25)
  (h2 : remaining_payment = 2625)
  (h3 : remaining_percentage = 0.75)
  (h4 : remaining_payment = remaining_percentage * total_cost) :
  part_payment_percentage * total_cost = 875 := by
  sorry

end refrigerator_part_payment_l2182_218233
