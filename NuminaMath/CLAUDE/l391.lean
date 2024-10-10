import Mathlib

namespace unique_base_system_solution_l391_39183

/-- Represents a base-b numeral system where 1987 is written as xyz --/
structure BaseSystem where
  b : ℕ
  x : ℕ
  y : ℕ
  z : ℕ
  h1 : b > 1
  h2 : x < b ∧ y < b ∧ z < b
  h3 : x + y + z = 25
  h4 : x * b^2 + y * b + z = 1987

/-- The unique solution to the base system problem --/
theorem unique_base_system_solution :
  ∃! (s : BaseSystem), s.b = 19 ∧ s.x = 5 ∧ s.y = 9 ∧ s.z = 11 :=
sorry

end unique_base_system_solution_l391_39183


namespace prob_one_red_ball_eq_one_third_l391_39146

/-- The probability of drawing exactly one red ball from a bag -/
def prob_one_red_ball (red_balls black_balls : ℕ) : ℚ :=
  red_balls / (red_balls + black_balls)

/-- Theorem: The probability of drawing exactly one red ball from a bag
    containing 2 red balls and 4 black balls is 1/3 -/
theorem prob_one_red_ball_eq_one_third :
  prob_one_red_ball 2 4 = 1/3 := by sorry

end prob_one_red_ball_eq_one_third_l391_39146


namespace horse_race_theorem_l391_39101

def horse_primes : List Nat := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

def is_valid_subset (s : List Nat) : Prop :=
  s.length = 5 ∧ s.toFinset ⊆ horse_primes.toFinset

def least_common_time (s : List Nat) : Nat :=
  s.foldl Nat.lcm 1

def sum_of_digits (n : Nat) : Nat :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem horse_race_theorem :
  ∃ (s : List Nat), is_valid_subset s ∧
    (∀ (t : List Nat), is_valid_subset t → least_common_time s ≤ least_common_time t) ∧
    least_common_time s = 2310 ∧
    sum_of_digits (least_common_time s) = 6 :=
  sorry

end horse_race_theorem_l391_39101


namespace product_of_roots_cubic_l391_39170

theorem product_of_roots_cubic (x : ℝ) : 
  (∃ p q r : ℝ, x^3 - 9*x^2 + 27*x - 5 = (x - p) * (x - q) * (x - r)) → 
  (∃ p q r : ℝ, x^3 - 9*x^2 + 27*x - 5 = (x - p) * (x - q) * (x - r) ∧ p * q * r = 5) := by
  sorry

end product_of_roots_cubic_l391_39170


namespace vector_equation_solution_l391_39152

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

theorem vector_equation_solution (a x : V) 
  (h : 2 • x - 3 • (x - 2 • a) = 0) : x = 6 • a := by
  sorry

end vector_equation_solution_l391_39152


namespace quadratic_equation_with_ratio_roots_l391_39185

theorem quadratic_equation_with_ratio_roots (k : ℝ) : 
  (∃ r : ℝ, r ≠ 0 ∧ 
    (∀ x : ℝ, x^2 + 8*x + k = 0 ↔ (x = 3*r ∨ x = r)) ∧
    3*r ≠ r) → 
  k = 12 := by
sorry

end quadratic_equation_with_ratio_roots_l391_39185


namespace no_solutions_exist_l391_39188

theorem no_solutions_exist : ¬∃ (x y : ℕ), 
  x^4 * y^4 - 14 * x^2 * y^2 + 49 = 0 ∧ x + y = 10 := by
  sorry

end no_solutions_exist_l391_39188


namespace weights_not_divisible_by_three_l391_39151

theorem weights_not_divisible_by_three :
  ¬ (∃ k : ℕ, 3 * k = (67 * 68) / 2) := by
  sorry

end weights_not_divisible_by_three_l391_39151


namespace angle_bisector_length_squared_l391_39123

/-- Given a triangle with sides a, b, and c, fa is the length of the angle bisector of angle α,
    and u and v are the lengths of the segments into which fa divides side a. -/
theorem angle_bisector_length_squared (a b c fa u v : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ fa > 0 ∧ u > 0 ∧ v > 0)
  (h_triangle : a < b + c ∧ b < a + c ∧ c < a + b)
  (h_segments : u + v = a)
  (h_ratio : u / v = c / b) :
  fa^2 = b * c - u * v := by
  sorry

end angle_bisector_length_squared_l391_39123


namespace game_specific_outcome_l391_39155

def game_probability (total_rounds : ℕ) 
                     (alex_prob : ℚ) 
                     (mel_chelsea_ratio : ℚ) 
                     (alex_wins : ℕ) 
                     (mel_wins : ℕ) 
                     (chelsea_wins : ℕ) : ℚ :=
  sorry

theorem game_specific_outcome : 
  game_probability 7 (3/5) 2 4 2 1 = 18144/1125 := by sorry

end game_specific_outcome_l391_39155


namespace unique_number_with_properties_l391_39113

theorem unique_number_with_properties : ∃! n : ℕ, 
  50 < n ∧ n < 70 ∧ 
  ∃ k : ℤ, n - 3 = 5 * k ∧
  ∃ l : ℤ, n - 2 = 7 * l :=
by
  -- The proof would go here
  sorry

end unique_number_with_properties_l391_39113


namespace fraction_multiplication_l391_39122

theorem fraction_multiplication (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) : 
  (b * c / a^2) * (a / b^2) = c / (a * b) := by
sorry

end fraction_multiplication_l391_39122


namespace probability_non_littermates_correct_l391_39130

/-- Represents the number of dogs with a specific number of littermates -/
structure DogGroup where
  count : Nat
  littermates : Nat

/-- Represents the total number of dogs and their groupings by littermates -/
structure BreedingKennel where
  totalDogs : Nat
  groups : List DogGroup

/-- Calculates the probability of selecting two non-littermate dogs from a breeding kennel -/
def probabilityNonLittermates (kennel : BreedingKennel) : Rat :=
  sorry

theorem probability_non_littermates_correct (kennel : BreedingKennel) :
  kennel.totalDogs = 20 ∧
  kennel.groups = [
    ⟨8, 1⟩,
    ⟨6, 2⟩,
    ⟨4, 3⟩,
    ⟨2, 4⟩
  ] →
  probabilityNonLittermates kennel = 82 / 95 :=
sorry

end probability_non_littermates_correct_l391_39130


namespace supplement_of_angle_with_30_degree_complement_l391_39126

theorem supplement_of_angle_with_30_degree_complement :
  ∀ (angle : ℝ), 
  (90 - angle = 30) →
  (180 - angle = 120) :=
by
  sorry

end supplement_of_angle_with_30_degree_complement_l391_39126


namespace sqrt_16_divided_by_2_l391_39111

theorem sqrt_16_divided_by_2 : Real.sqrt 16 / 2 = 2 := by
  sorry

end sqrt_16_divided_by_2_l391_39111


namespace vector_at_t_3_l391_39103

/-- A line in 3D space parameterized by t -/
structure ParametricLine where
  -- The vector on the line at any given t
  vector : ℝ → ℝ × ℝ × ℝ

/-- Theorem: Given a parameterized line with specific points, 
    we can determine the vector at t = 3 -/
theorem vector_at_t_3 
  (line : ParametricLine)
  (h1 : line.vector (-1) = (1, 3, 8))
  (h2 : line.vector 2 = (0, -2, -4)) :
  line.vector 3 = (-1/3, -11/3, -8) := by
  sorry

end vector_at_t_3_l391_39103


namespace quadratic_equation_proof_l391_39199

theorem quadratic_equation_proof (m : ℝ) (x₁ x₂ : ℝ) : 
  (∀ x, m * x^2 + 2*(m+1)*x + (m-1) = 0 ↔ x = x₁ ∨ x = x₂) →  -- equation has roots x₁ and x₂
  x₁ ≠ x₂ →  -- roots are distinct
  m > -1/3 →  -- condition from part 1
  m ≠ 0 →  -- condition from part 1
  x₁^2 + x₂^2 = 8 →  -- given condition
  m = 2 :=  -- conclusion
by sorry

end quadratic_equation_proof_l391_39199


namespace least_period_is_30_l391_39195

/-- A function satisfying the given condition -/
def satisfies_condition (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (x + 5) + f (x - 5) = f x

/-- The period of a function -/
def is_period (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x : ℝ, f (x + p) = f x

/-- The least common positive period for all functions satisfying the condition -/
def least_common_positive_period (p : ℝ) : Prop :=
  p > 0 ∧
  (∀ f : ℝ → ℝ, satisfies_condition f → is_period f p) ∧
  ∀ q : ℝ, 0 < q ∧ q < p → ∃ f : ℝ → ℝ, satisfies_condition f ∧ ¬is_period f q

theorem least_period_is_30 :
  least_common_positive_period 30 := by sorry

end least_period_is_30_l391_39195


namespace min_value_product_l391_39194

theorem min_value_product (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : (2/x) + (3/y) + (1/z) = 12) : 
  x^2 * y^3 * z ≥ (1/64) :=
by sorry

end min_value_product_l391_39194


namespace no_m_exists_for_equality_m_range_for_subset_l391_39118

-- Define the sets P and S
def P : Set ℝ := {x | x^2 - 8*x - 20 ≤ 0}
def S (m : ℝ) : Set ℝ := {x | 1 - m ≤ x ∧ x ≤ 1 + m}

-- Theorem 1: No m exists such that P = S(m)
theorem no_m_exists_for_equality : ¬ ∃ m : ℝ, P = S m := by sorry

-- Theorem 2: The set of m such that P ⊆ S(m) is {m | m ≤ 3}
theorem m_range_for_subset : {m : ℝ | P ⊆ S m} = {m : ℝ | m ≤ 3} := by sorry

end no_m_exists_for_equality_m_range_for_subset_l391_39118


namespace proposition_q_must_be_true_l391_39181

theorem proposition_q_must_be_true (p q : Prop) 
  (h1 : ¬p) (h2 : p ∨ q) : q := by
  sorry

end proposition_q_must_be_true_l391_39181


namespace parabola_y1_gt_y2_l391_39139

/-- A parabola with axis of symmetry at x = 1 -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ
  h : a > 0

/-- The y-coordinate of a point on the parabola given its x-coordinate -/
def Parabola.y_at (p : Parabola) (x : ℝ) : ℝ :=
  p.a * x^2 + p.b * x + p.c

theorem parabola_y1_gt_y2 (p : Parabola) :
  p.y_at (-1) > p.y_at 2 := by
  sorry

end parabola_y1_gt_y2_l391_39139


namespace q_satisfies_conditions_l391_39138

/-- A quadratic polynomial that satisfies specific conditions -/
def q (x : ℚ) : ℚ := (4 * x^2 - 6 * x + 5) / 3

/-- Theorem stating that q satisfies the given conditions -/
theorem q_satisfies_conditions :
  q (-1) = 5 ∧ q 2 = 3 ∧ q 4 = 15 := by
  sorry


end q_satisfies_conditions_l391_39138


namespace joans_kittens_l391_39198

theorem joans_kittens (initial : ℕ) (additional : ℕ) (total : ℕ) 
  (h1 : initial = 15)
  (h2 : additional = 5)
  (h3 : total = initial + additional) : 
  total = 20 := by
  sorry

end joans_kittens_l391_39198


namespace victors_friend_bought_two_decks_l391_39161

/-- The number of decks Victor's friend bought given the conditions of the problem -/
def victors_friend_decks (deck_cost : ℕ) (victors_decks : ℕ) (total_spent : ℕ) : ℕ :=
  (total_spent - deck_cost * victors_decks) / deck_cost

/-- Theorem stating that Victor's friend bought 2 decks under the given conditions -/
theorem victors_friend_bought_two_decks :
  victors_friend_decks 8 6 64 = 2 := by
  sorry

end victors_friend_bought_two_decks_l391_39161


namespace problem_solution_l391_39186

theorem problem_solution (x y z : ℝ) 
  (h1 : 5 = 0.25 * x)
  (h2 : 5 = 0.10 * y)
  (h3 : z = 2 * y) :
  x - z = -80 := by
  sorry

end problem_solution_l391_39186


namespace perpendicular_line_unique_l391_39107

-- Define a line by its coefficients (a, b, c) in the equation ax + by + c = 0
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

def Line.throughPoint (l : Line) (p : ℝ × ℝ) : Prop :=
  l.a * p.1 + l.b * p.2 + l.c = 0

def Line.perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

theorem perpendicular_line_unique :
  ∃! l : Line, l.throughPoint (3, 0) ∧
                l.perpendicular { a := 2, b := 1, c := -5 } ∧
                l = { a := 1, b := -2, c := -3 } := by
  sorry

end perpendicular_line_unique_l391_39107


namespace integer_roots_of_polynomial_l391_39140

def f (x : ℤ) : ℤ := x^3 - 4*x^2 - 7*x + 10

theorem integer_roots_of_polynomial :
  {x : ℤ | f x = 0} = {1, -2, 5} := by sorry

end integer_roots_of_polynomial_l391_39140


namespace pure_imaginary_fraction_l391_39142

theorem pure_imaginary_fraction (a : ℝ) : 
  (∀ z : ℂ, z = (Complex.I : ℂ) / (1 + a * Complex.I) → Complex.re z = 0 ∧ Complex.im z ≠ 0) → 
  a = 0 := by
  sorry

end pure_imaginary_fraction_l391_39142


namespace race_problem_l391_39154

/-- The race problem -/
theorem race_problem (jack_first_half jack_second_half jill_total : ℕ) 
  (h1 : jack_first_half = 19)
  (h2 : jack_second_half = 6)
  (h3 : jill_total = 32) :
  jill_total - (jack_first_half + jack_second_half) = 7 := by
  sorry

end race_problem_l391_39154


namespace bridget_apples_bridget_apples_proof_l391_39135

theorem bridget_apples : ℕ → Prop :=
  fun x =>
    let remaining_after_ann := x / 3
    let remaining_after_cassie := remaining_after_ann - 5
    let remaining_after_found := remaining_after_cassie + 3
    remaining_after_found = 6 → x = 24

-- Proof
theorem bridget_apples_proof : ∃ x : ℕ, bridget_apples x := by
  sorry

end bridget_apples_bridget_apples_proof_l391_39135


namespace M_geq_N_l391_39115

theorem M_geq_N (a : ℝ) : 2 * a * (a - 2) + 3 ≥ (a - 1) * (a - 3) := by
  sorry

end M_geq_N_l391_39115


namespace total_rope_length_l391_39128

-- Define the lengths of rope used for each post
def post1_length : ℕ := 24
def post2_length : ℕ := 20
def post3_length : ℕ := 14
def post4_length : ℕ := 12

-- Theorem stating that the total rope length is 70 inches
theorem total_rope_length :
  post1_length + post2_length + post3_length + post4_length = 70 :=
by sorry

end total_rope_length_l391_39128


namespace union_M_complement_N_l391_39133

def U : Set Nat := {1, 2, 3, 4, 5, 6, 7, 8, 9}
def M : Set Nat := {1, 3, 5, 6}
def N : Set Nat := {1, 2, 4, 7, 9}

theorem union_M_complement_N : M ∪ (U \ N) = {1, 3, 5, 6, 8} := by sorry

end union_M_complement_N_l391_39133


namespace classroom_a_fundraising_l391_39163

/-- The fundraising goal for each classroom -/
def goal : ℕ := 200

/-- The amount raised from two families at $20 each -/
def amount_20 : ℕ := 2 * 20

/-- The amount raised from eight families at $10 each -/
def amount_10 : ℕ := 8 * 10

/-- The amount raised from ten families at $5 each -/
def amount_5 : ℕ := 10 * 5

/-- The total amount raised by Classroom A -/
def total_raised : ℕ := amount_20 + amount_10 + amount_5

/-- The additional amount needed to reach the goal -/
def additional_amount_needed : ℕ := goal - total_raised

theorem classroom_a_fundraising :
  additional_amount_needed = 30 :=
by sorry

end classroom_a_fundraising_l391_39163


namespace min_box_value_l391_39160

/-- Given that (ax+b)(bx+a) = 30x^2 + ⬜x + 30, where a, b, and ⬜ are distinct integers,
    prove that the minimum possible value of ⬜ is 61. -/
theorem min_box_value (a b box : ℤ) : 
  (∀ x, (a*x + b)*(b*x + a) = 30*x^2 + box*x + 30) →
  a ≠ b ∧ b ≠ box ∧ a ≠ box →
  a * b = 30 →
  box = a^2 + b^2 →
  (∀ a' b' box' : ℤ, 
    (∀ x, (a'*x + b')*(b'*x + a') = 30*x^2 + box'*x + 30) →
    a' ≠ b' ∧ b' ≠ box' ∧ a' ≠ box' →
    a' * b' = 30 →
    box' = a'^2 + b'^2 →
    box ≤ box') →
  box = 61 := by
sorry

end min_box_value_l391_39160


namespace egg_production_increase_proof_l391_39165

/-- The increase in egg production from last year to this year -/
def egg_production_increase (last_year_production this_year_production : ℕ) : ℕ :=
  this_year_production - last_year_production

/-- Theorem stating the increase in egg production -/
theorem egg_production_increase_proof 
  (last_year_production : ℕ) 
  (this_year_production : ℕ) 
  (h1 : last_year_production = 1416)
  (h2 : this_year_production = 4636) : 
  egg_production_increase last_year_production this_year_production = 3220 := by
  sorry

end egg_production_increase_proof_l391_39165


namespace value_of_y_l391_39108

theorem value_of_y (x y : ℝ) (h1 : x^(2*y) = 16) (h2 : x = 16) : y = 1/4 := by
  sorry

end value_of_y_l391_39108


namespace fifth_term_of_geometric_sequence_l391_39147

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- State the theorem
theorem fifth_term_of_geometric_sequence 
  (a : ℕ → ℝ) 
  (h_geo : geometric_sequence a) 
  (h_a1 : a 1 = 3) 
  (h_a3 : a 3 = 12) : 
  a 5 = 48 := by
sorry

end fifth_term_of_geometric_sequence_l391_39147


namespace range_of_a_l391_39100

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, |x - a| < 1 ↔ (1/2 : ℝ) < x ∧ x < (3/2 : ℝ)) ↔ 
  ((1/2 : ℝ) ≤ a ∧ a ≤ (3/2 : ℝ)) :=
sorry

end range_of_a_l391_39100


namespace parabola_equation_l391_39162

/-- The equation of a parabola with focus at (2, 1) and the y-axis as its directrix -/
theorem parabola_equation (x y : ℝ) :
  let focus : ℝ × ℝ := (2, 1)
  let directrix : Set (ℝ × ℝ) := {p | p.1 = 0}
  let parabola_equation : ℝ × ℝ → Prop := λ p => (p.2 - 1)^2 = 4 * (p.1 - 1)
  (∀ p, p ∈ directrix → dist p focus = dist p (x, y)) ↔ parabola_equation (x, y) :=
by sorry

end parabola_equation_l391_39162


namespace even_function_implies_a_equals_negative_one_l391_39110

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := (x - 1) * (x - a)

-- State the theorem
theorem even_function_implies_a_equals_negative_one :
  (∀ x : ℝ, f a x = f a (-x)) → a = -1 := by
  sorry

end even_function_implies_a_equals_negative_one_l391_39110


namespace base_seven_to_ten_63524_l391_39164

/-- Converts a digit in base 7 to its value in base 10 -/
def baseSevenDigitToBaseTen (d : Nat) : Nat :=
  if d < 7 then d else 0

/-- Converts a list of digits in base 7 to its value in base 10 -/
def baseSevenToBaseTen (digits : List Nat) : Nat :=
  digits.foldr (fun d acc => acc * 7 + baseSevenDigitToBaseTen d) 0

/-- The base 7 number 63524 converted to base 10 equals 15698 -/
theorem base_seven_to_ten_63524 :
  baseSevenToBaseTen [6, 3, 5, 2, 4] = 15698 := by
  sorry

end base_seven_to_ten_63524_l391_39164


namespace regular_pay_is_2_40_l391_39172

/-- Calculates the regular pay per hour given the following conditions:
  - Regular week: 5 working days, 8 hours per day
  - Overtime pay: Rs. 3.20 per hour
  - Total earnings in 4 weeks: Rs. 432
  - Total hours worked in 4 weeks: 175 hours
-/
def regularPayPerHour (
  workingDaysPerWeek : ℕ)
  (workingHoursPerDay : ℕ)
  (overtimePay : ℚ)
  (totalEarnings : ℚ)
  (totalHoursWorked : ℕ) : ℚ :=
  let regularHoursPerWeek := workingDaysPerWeek * workingHoursPerDay
  let totalRegularHours := 4 * regularHoursPerWeek
  let overtimeHours := totalHoursWorked - totalRegularHours
  let overtimeEarnings := overtimeHours * overtimePay
  let regularEarnings := totalEarnings - overtimeEarnings
  regularEarnings / totalRegularHours

/-- Proves that the regular pay per hour is Rs. 2.40 given the specified conditions. -/
theorem regular_pay_is_2_40 :
  regularPayPerHour 5 8 (32/10) 432 175 = 24/10 := by
  sorry

end regular_pay_is_2_40_l391_39172


namespace total_late_time_l391_39191

def charlize_late : ℕ := 20

def ana_late : ℕ := charlize_late + charlize_late / 4
def ben_late : ℕ := charlize_late * 3 / 4
def clara_late : ℕ := charlize_late * 2
def daniel_late : ℕ := 30 * 4 / 5

def ana_missed : ℕ := 5
def ben_missed : ℕ := 2
def clara_missed : ℕ := 15
def daniel_missed : ℕ := 10

theorem total_late_time :
  charlize_late +
  (ana_late + ana_missed) +
  (ben_late + ben_missed) +
  (clara_late + clara_missed) +
  (daniel_late + daniel_missed) = 156 := by
  sorry

end total_late_time_l391_39191


namespace sandwich_combinations_l391_39141

theorem sandwich_combinations (n m k l : ℕ) (hn : n = 7) (hm : m = 3) (hk : k = 2) (hl : l = 1) :
  (n.choose k) * (m.choose l) = 63 := by
  sorry

end sandwich_combinations_l391_39141


namespace cos_20_minus_cos_40_l391_39168

theorem cos_20_minus_cos_40 : Real.cos (20 * π / 180) - Real.cos (40 * π / 180) = 1 / 2 := by
  sorry

end cos_20_minus_cos_40_l391_39168


namespace employee_salaries_l391_39157

/-- Proves that the salaries of employees m, n, p, and q sum up to $3000 given the stated conditions --/
theorem employee_salaries (n m p q : ℝ) : 
  (m = 1.4 * n) →
  (p = 0.85 * (m - n)) →
  (q = 1.1 * p) →
  (n + m + p + q = 3000) :=
by
  sorry

end employee_salaries_l391_39157


namespace lcm_problem_l391_39143

theorem lcm_problem (a b : ℕ+) (h1 : a + b = 55) (h2 : Nat.gcd a b = 5) 
  (h3 : (1 : ℚ) / a + (1 : ℚ) / b = 0.09166666666666666) : 
  Nat.lcm a b = 120 := by
  sorry

end lcm_problem_l391_39143


namespace two_special_birth_years_l391_39105

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

theorem two_special_birth_years :
  ∃ (y1 y2 : ℕ),
    y1 ≠ y2 ∧
    y1 ≥ 1900 ∧ y1 ≤ 2021 ∧
    y2 ≥ 1900 ∧ y2 ≤ 2021 ∧
    2021 - y1 = sum_of_digits y1 ∧
    2021 - y2 = sum_of_digits y2 ∧
    2022 - y1 = 8 ∧
    2022 - y2 = 26 :=
by sorry

end two_special_birth_years_l391_39105


namespace triangle_problem_l391_39177

theorem triangle_problem (A B C : Real) (a b c : Real) :
  (1/2 * b * c * Real.sin A = 10 * Real.sqrt 3) →  -- Area condition
  (a = 7) →  -- Given side length
  (Real.sin A)^2 = (Real.sin B)^2 + (Real.sin C)^2 - Real.sin B * Real.sin C →  -- Given equation
  (A = π/3 ∧ ((b = 5 ∧ c = 8) ∨ (b = 8 ∧ c = 5))) := by
  sorry

end triangle_problem_l391_39177


namespace job_completion_proof_l391_39175

/-- Given workers P, Q, and R who can complete a job in 3, 9, and 6 hours respectively,
    prove that the combined work of P (1 hour), Q (2 hours), and R (3 hours) completes the job. -/
theorem job_completion_proof (p q r : ℝ) (hp : p = 1/3) (hq : q = 1/9) (hr : r = 1/6) :
  p * 1 + q * 2 + r * 3 ≥ 1 := by
  sorry

#check job_completion_proof

end job_completion_proof_l391_39175


namespace college_students_count_l391_39137

theorem college_students_count (boys girls : ℕ) (h1 : boys * 5 = girls * 8) (h2 : girls = 400) :
  boys + girls = 1040 := by
  sorry

end college_students_count_l391_39137


namespace shopping_mall_sales_l391_39180

/-- Shopping mall sales problem -/
theorem shopping_mall_sales
  (initial_cost : ℝ)
  (initial_price : ℝ)
  (january_sales : ℝ)
  (march_sales : ℝ)
  (price_decrease : ℝ)
  (sales_increase : ℝ)
  (desired_profit : ℝ)
  (h1 : initial_cost = 60)
  (h2 : initial_price = 80)
  (h3 : january_sales = 64)
  (h4 : march_sales = 100)
  (h5 : price_decrease = 0.5)
  (h6 : sales_increase = 5)
  (h7 : desired_profit = 2160) :
  ∃ (growth_rate : ℝ) (optimal_price : ℝ),
    growth_rate = 0.25 ∧
    optimal_price = 72 ∧
    (1 + growth_rate)^2 * january_sales = march_sales ∧
    (optimal_price - initial_cost) * (march_sales + (sales_increase / price_decrease) * (initial_price - optimal_price)) = desired_profit :=
by sorry

end shopping_mall_sales_l391_39180


namespace rectangle_perimeter_l391_39187

/-- Properties of a rectangle and an ellipse -/
structure RectangleEllipseSystem where
  /-- Length of the rectangle -/
  x : ℝ
  /-- Width of the rectangle -/
  y : ℝ
  /-- Semi-major axis of the ellipse -/
  a : ℝ
  /-- Semi-minor axis of the ellipse -/
  b : ℝ
  /-- The area of the rectangle is 3260 -/
  area_rectangle : x * y = 3260
  /-- The area of the ellipse is 3260π -/
  area_ellipse : π * a * b = 3260 * π
  /-- The sum of length and width equals twice the semi-major axis -/
  major_axis : x + y = 2 * a
  /-- The rectangle diagonal equals twice the focal distance -/
  focal_distance : x^2 + y^2 = 4 * (a^2 - b^2)

/-- The perimeter of the rectangle is 8√1630 -/
theorem rectangle_perimeter (s : RectangleEllipseSystem) : 
  2 * (s.x + s.y) = 8 * Real.sqrt 1630 := by
  sorry

end rectangle_perimeter_l391_39187


namespace mans_rate_l391_39149

/-- The man's rate in still water given his speeds with and against the stream -/
theorem mans_rate (speed_with_stream speed_against_stream : ℝ) 
  (h1 : speed_with_stream = 24)
  (h2 : speed_against_stream = 10) : 
  (speed_with_stream + speed_against_stream) / 2 = 17 := by
  sorry

end mans_rate_l391_39149


namespace robin_sodas_l391_39134

/-- The number of sodas Robin and her friends drank -/
def sodas_drunk : ℕ := 3

/-- The number of extra sodas Robin had -/
def sodas_extra : ℕ := 8

/-- The total number of sodas Robin bought -/
def total_sodas : ℕ := sodas_drunk + sodas_extra

theorem robin_sodas : total_sodas = 11 := by sorry

end robin_sodas_l391_39134


namespace second_exam_sleep_duration_l391_39174

/-- Represents the relationship between sleep duration and test score -/
structure SleepScoreRelation where
  sleep : ℝ
  score : ℝ
  constant : ℝ
  inv_relation : sleep * score = constant

/-- Proves the required sleep duration for the second exam -/
theorem second_exam_sleep_duration 
  (first_exam : SleepScoreRelation)
  (h_first_exam : first_exam.sleep = 9 ∧ first_exam.score = 75)
  (target_average : ℝ)
  (h_target_average : target_average = 85) :
  ∃ (second_exam : SleepScoreRelation),
    second_exam.constant = first_exam.constant ∧
    (first_exam.score + second_exam.score) / 2 = target_average ∧
    second_exam.sleep = 135 / 19 := by
  sorry

end second_exam_sleep_duration_l391_39174


namespace alpha_gt_beta_not_sufficient_nor_necessary_for_sin_alpha_gt_sin_beta_l391_39104

theorem alpha_gt_beta_not_sufficient_nor_necessary_for_sin_alpha_gt_sin_beta :
  ∃ (α β : Real),
    0 < α ∧ α < π/2 ∧
    0 < β ∧ β < π/2 ∧
    (
      (α > β ∧ ¬(Real.sin α > Real.sin β)) ∧
      (Real.sin α > Real.sin β ∧ ¬(α > β))
    ) :=
by sorry

end alpha_gt_beta_not_sufficient_nor_necessary_for_sin_alpha_gt_sin_beta_l391_39104


namespace initial_books_correct_l391_39176

/-- The number of books initially in the pile to be put away. -/
def initial_books : ℝ := 46.0

/-- The number of books added by the librarian. -/
def added_books : ℝ := 10.0

/-- The number of books that can fit on each shelf. -/
def books_per_shelf : ℝ := 4.0

/-- The number of shelves needed to arrange all books. -/
def shelves_needed : ℕ := 14

/-- Theorem stating that the initial number of books is correct given the conditions. -/
theorem initial_books_correct : 
  initial_books = (books_per_shelf * shelves_needed : ℝ) - added_books :=
by sorry

end initial_books_correct_l391_39176


namespace composite_form_l391_39190

theorem composite_form (n : ℕ) : 
  ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ (111111 + 9 * 10^n = a * b) := by
  sorry

end composite_form_l391_39190


namespace crazy_silly_school_series_problem_l391_39125

/-- The 'Crazy Silly School' series problem -/
theorem crazy_silly_school_series_problem 
  (total_books : ℕ) 
  (total_movies : ℕ) 
  (books_read : ℕ) 
  (movies_watched : ℕ) 
  (h1 : total_books = 25) 
  (h2 : total_movies = 35) 
  (h3 : books_read = 15) 
  (h4 : movies_watched = 29) :
  movies_watched - books_read = 14 := by
  sorry

end crazy_silly_school_series_problem_l391_39125


namespace consecutive_even_sequence_unique_l391_39145

/-- A sequence of four consecutive even integers -/
def ConsecutiveEvenSequence (a b c d : ℤ) : Prop :=
  (b = a + 2) ∧ (c = b + 2) ∧ (d = c + 2) ∧ Even a ∧ Even b ∧ Even c ∧ Even d

theorem consecutive_even_sequence_unique :
  ∀ a b c d : ℤ,
  ConsecutiveEvenSequence a b c d →
  c = 14 →
  a + b + c + d = 52 →
  a = 10 ∧ b = 12 ∧ c = 14 ∧ d = 16 := by
sorry

end consecutive_even_sequence_unique_l391_39145


namespace theta_value_l391_39106

theorem theta_value (θ : Real)
  (h1 : 3 * Real.pi ≤ θ ∧ θ ≤ 4 * Real.pi)
  (h2 : Real.sqrt ((1 + Real.cos θ) / 2) + Real.sqrt ((1 - Real.cos θ) / 2) = Real.sqrt 6 / 2) :
  θ = 19 * Real.pi / 6 ∨ θ = 23 * Real.pi / 6 := by
  sorry

end theta_value_l391_39106


namespace floor_equality_l391_39112

theorem floor_equality (a b : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : a * b > 1) :
  ⌊((a - b)^2 - 1 : ℚ) / (a * b)⌋ = ⌊((a - b)^2 - 1 : ℚ) / (a * b - 1)⌋ := by
  sorry

end floor_equality_l391_39112


namespace imaginary_part_of_z_l391_39158

theorem imaginary_part_of_z (z : ℂ) : z = (Complex.I ^ 2017) / (1 - 2 * Complex.I) → z.im = 1/5 := by
  sorry

end imaginary_part_of_z_l391_39158


namespace equation_solution_l391_39197

theorem equation_solution : 
  {x : ℝ | (5 + x) / (7 + x) = (2 + x^2) / (4 + x)} = {1, -2, -3} := by
  sorry

end equation_solution_l391_39197


namespace max_value_of_2sinx_l391_39120

theorem max_value_of_2sinx :
  ∃ (M : ℝ), M = 2 ∧ ∀ (x : ℝ), 2 * Real.sin x ≤ M :=
by sorry

end max_value_of_2sinx_l391_39120


namespace student_difference_l391_39184

theorem student_difference (lower_grades : ℕ) (middle_upper_grades : ℕ) : 
  lower_grades = 325 →
  middle_upper_grades = 4 * lower_grades →
  middle_upper_grades - lower_grades = 975 := by
  sorry

end student_difference_l391_39184


namespace inequality_proof_l391_39171

theorem inequality_proof (a b c : ℝ) (k : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hk : k ≥ 2) (habc : a * b * c = 1) :
  (a^k / (a + b)) + (b^k / (b + c)) + (c^k / (c + a)) ≥ 3/2 := by
  sorry

end inequality_proof_l391_39171


namespace percentage_passed_both_l391_39109

theorem percentage_passed_both (failed_hindi failed_english failed_both : ℚ) 
  (h1 : failed_hindi = 35 / 100)
  (h2 : failed_english = 45 / 100)
  (h3 : failed_both = 20 / 100) :
  1 - (failed_hindi + failed_english - failed_both) = 40 / 100 := by
  sorry

end percentage_passed_both_l391_39109


namespace complex_magnitude_product_l391_39150

theorem complex_magnitude_product : Complex.abs (5 - 3*I) * Complex.abs (5 + 3*I) = 34 := by
  sorry

end complex_magnitude_product_l391_39150


namespace binomial_11_1_l391_39131

theorem binomial_11_1 : (11 : ℕ).choose 1 = 11 := by
  sorry

end binomial_11_1_l391_39131


namespace point_coordinates_in_fourth_quadrant_l391_39102

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define the fourth quadrant
def in_fourth_quadrant (p : Point2D) : Prop :=
  p.x > 0 ∧ p.y < 0

-- Define distance to x-axis
def distance_to_x_axis (p : Point2D) : ℝ :=
  |p.y|

-- Define distance to y-axis
def distance_to_y_axis (p : Point2D) : ℝ :=
  |p.x|

-- Theorem statement
theorem point_coordinates_in_fourth_quadrant (p : Point2D) 
  (h1 : in_fourth_quadrant p)
  (h2 : distance_to_x_axis p = 3)
  (h3 : distance_to_y_axis p = 8) :
  p = Point2D.mk 8 (-3) := by
  sorry

end point_coordinates_in_fourth_quadrant_l391_39102


namespace cat_mouse_positions_after_258_moves_l391_39148

/-- Represents the positions on the square grid --/
inductive Position
  | TopLeft
  | TopRight
  | BottomRight
  | BottomLeft
  | TopMiddle
  | RightMiddle
  | BottomMiddle
  | LeftMiddle

/-- Represents the movement of the cat --/
def catMove (n : ℕ) : Position :=
  match n % 4 with
  | 0 => Position.TopLeft
  | 1 => Position.TopRight
  | 2 => Position.BottomRight
  | 3 => Position.BottomLeft
  | _ => Position.TopLeft  -- This case is unreachable, but needed for exhaustiveness

/-- Represents the movement of the mouse --/
def mouseMove (n : ℕ) : Position :=
  match n % 8 with
  | 0 => Position.TopMiddle
  | 1 => Position.TopRight
  | 2 => Position.RightMiddle
  | 3 => Position.BottomRight
  | 4 => Position.BottomMiddle
  | 5 => Position.BottomLeft
  | 6 => Position.LeftMiddle
  | 7 => Position.TopLeft
  | _ => Position.TopMiddle  -- This case is unreachable, but needed for exhaustiveness

theorem cat_mouse_positions_after_258_moves :
  catMove 258 = Position.TopRight ∧ mouseMove 258 = Position.TopRight :=
sorry

end cat_mouse_positions_after_258_moves_l391_39148


namespace proportional_equation_inequality_l391_39129

theorem proportional_equation_inequality (a b c d : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) :
  b / d = c / a → ¬(a * d = b * c) :=
by sorry

end proportional_equation_inequality_l391_39129


namespace maurice_prior_rides_eq_eight_l391_39178

/-- The number of times Maurice rode during his visit -/
def maurice_visit_rides : ℕ := 8

/-- The number of times Matt rode without Maurice -/
def matt_solo_rides : ℕ := 16

/-- The total number of times Matt rode -/
def matt_total_rides : ℕ := maurice_visit_rides + matt_solo_rides

/-- The number of times Maurice rode before his visit -/
def maurice_prior_rides : ℕ := matt_total_rides / 3

theorem maurice_prior_rides_eq_eight :
  maurice_prior_rides = 8 := by sorry

end maurice_prior_rides_eq_eight_l391_39178


namespace binomial_expansion_properties_l391_39124

theorem binomial_expansion_properties :
  let n : ℕ := 15
  let last_three_sum := (n.choose (n-2)) + (n.choose (n-1)) + (n.choose n)
  let term (r : ℕ) := (n.choose r) * (3^r)
  ∃ (r₁ r₂ : ℕ),
    (last_three_sum = 121) ∧
    (∀ k, 0 ≤ k ∧ k ≤ n → (n.choose k) ≤ (n.choose r₁) ∧ (n.choose k) ≤ (n.choose r₂)) ∧
    (∀ k, 0 ≤ k ∧ k ≤ n → term k ≤ term r₁ ∧ term k ≤ term r₂) ∧
    r₁ = 11 ∧ r₂ = 12 := by
  sorry

end binomial_expansion_properties_l391_39124


namespace modulus_of_z_l391_39117

theorem modulus_of_z (z : ℂ) (h : (1 - Complex.I) * z = 1 + Complex.I) : Complex.abs z = 1 := by
  sorry

end modulus_of_z_l391_39117


namespace division_problem_l391_39156

theorem division_problem (N : ℕ) : 
  (N / 3 = 4) ∧ (N % 3 = 3) → N = 15 := by
  sorry

end division_problem_l391_39156


namespace website_visitors_ratio_l391_39193

/-- Proves that the ratio of visitors on the last day to the total visitors on the first 6 days is 2:1 -/
theorem website_visitors_ratio (daily_visitors : ℕ) (constant_days : ℕ) (revenue_per_visit : ℚ) (total_revenue : ℚ) 
  (h1 : daily_visitors = 100)
  (h2 : constant_days = 6)
  (h3 : revenue_per_visit = 1 / 100)
  (h4 : total_revenue = 18) :
  (total_revenue / revenue_per_visit - daily_visitors * constant_days) / (daily_visitors * constant_days) = 2 := by
sorry

end website_visitors_ratio_l391_39193


namespace max_angle_A_l391_39153

/-- Represents the side lengths of a triangle sequence -/
structure TriangleSequence where
  a : ℕ → ℝ
  b : ℕ → ℝ
  c : ℕ → ℝ

/-- Conditions for the triangle sequence -/
def ValidTriangleSequence (t : TriangleSequence) : Prop :=
  (t.b 1 > t.c 1) ∧
  (t.b 1 + t.c 1 = 2 * t.a 1) ∧
  (∀ n, t.a (n + 1) = t.a n) ∧
  (∀ n, t.b (n + 1) = (t.c n + t.a n) / 2) ∧
  (∀ n, t.c (n + 1) = (t.b n + t.a n) / 2)

/-- The angle A_n in the triangle sequence -/
noncomputable def angleA (t : TriangleSequence) (n : ℕ) : ℝ :=
  Real.arccos ((t.b n ^ 2 + t.c n ^ 2 - t.a n ^ 2) / (2 * t.b n * t.c n))

/-- The theorem stating the maximum value of angle A_n -/
theorem max_angle_A (t : TriangleSequence) (h : ValidTriangleSequence t) :
    (∀ n, angleA t n ≤ π / 3) ∧ (∃ n, angleA t n = π / 3) := by
  sorry


end max_angle_A_l391_39153


namespace arithmetic_geometric_mean_two_variables_l391_39116

theorem arithmetic_geometric_mean_two_variables (a b : ℝ) : (a^2 + b^2) / 2 ≥ a * b := by
  sorry

end arithmetic_geometric_mean_two_variables_l391_39116


namespace shopkeeper_profit_days_l391_39173

/-- Proves that given the specified mean profits, the total number of days is 30 -/
theorem shopkeeper_profit_days : 
  ∀ (total_days : ℕ) (mean_profit mean_first_15 mean_last_15 : ℚ),
  mean_profit = 350 →
  mean_first_15 = 225 →
  mean_last_15 = 475 →
  mean_profit * total_days = mean_first_15 * 15 + mean_last_15 * 15 →
  total_days = 30 := by
sorry

end shopkeeper_profit_days_l391_39173


namespace tangent_line_y_intercept_l391_39132

-- Define the circles
def circle1_center : ℝ × ℝ := (3, 0)
def circle1_radius : ℝ := 3
def circle2_center : ℝ × ℝ := (8, 0)
def circle2_radius : ℝ := 2

-- Define the tangent line
def tangent_line (y_intercept : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ (m : ℝ), p.2 = m * p.1 + y_intercept}

-- Define the condition for the line to be tangent to a circle
def is_tangent_to_circle (line : Set (ℝ × ℝ)) (center : ℝ × ℝ) (radius : ℝ) : Prop :=
  ∃ (p : ℝ × ℝ), p ∈ line ∧ 
    (p.1 - center.1) ^ 2 + (p.2 - center.2) ^ 2 = radius ^ 2 ∧
    ∀ (q : ℝ × ℝ), q ∈ line → q ≠ p → 
      (q.1 - center.1) ^ 2 + (q.2 - center.2) ^ 2 > radius ^ 2

-- Theorem statement
theorem tangent_line_y_intercept : 
  ∃ (y_intercept : ℝ), 
    y_intercept = 2 * Real.sqrt 104 ∧
    let line := tangent_line y_intercept
    is_tangent_to_circle line circle1_center circle1_radius ∧
    is_tangent_to_circle line circle2_center circle2_radius ∧
    ∀ (p : ℝ × ℝ), p ∈ line → p.1 ≥ 0 ∧ p.2 ≥ 0 := by
  sorry

end tangent_line_y_intercept_l391_39132


namespace max_value_expression_l391_39196

theorem max_value_expression (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h : a^2 + b^2 + c^2 = 2*a*b*c + 1) :
  (a - 2*b*c) * (b - 2*c*a) * (c - 2*a*b) ≤ 1/8 := by sorry

end max_value_expression_l391_39196


namespace gcd_of_180_and_270_l391_39169

theorem gcd_of_180_and_270 : Nat.gcd 180 270 = 90 := by
  sorry

end gcd_of_180_and_270_l391_39169


namespace f_equals_g_l391_39166

-- Define the functions f and g
def f (x : ℝ) : ℝ := |x|
def g (x : ℝ) : ℝ := (x^4)^(1/4)

-- State the theorem
theorem f_equals_g : ∀ x : ℝ, f x = g x := by
  sorry

end f_equals_g_l391_39166


namespace vector_equation_solution_l391_39119

/-- Given vectors a and b, if 3a - 2b + c = 0, then c = (-23, -12) -/
theorem vector_equation_solution (a b c : ℝ × ℝ) :
  a = (5, 2) →
  b = (-4, -3) →
  3 • a - 2 • b + c = (0, 0) →
  c = (-23, -12) := by sorry

end vector_equation_solution_l391_39119


namespace remainder_3_20_mod_11_l391_39189

theorem remainder_3_20_mod_11 (h : Prime 11) : 3^20 ≡ 1 [MOD 11] := by
  sorry

end remainder_3_20_mod_11_l391_39189


namespace intersection_of_A_and_B_l391_39144

def A : Set ℝ := {x | |x| < 2}
def B : Set ℝ := {-1, 0, 1, 2, 3}

theorem intersection_of_A_and_B : A ∩ B = {-1, 0, 1} := by sorry

end intersection_of_A_and_B_l391_39144


namespace two_numbers_with_specific_means_l391_39182

theorem two_numbers_with_specific_means (x y : ℝ) (hx : x > 0) (hy : y > 0) : 
  (x * y = 600^2) → 
  ((x + y) / 2 = (2 * x * y) / (x + y) + 49) →
  ({x, y} : Set ℝ) = {800, 450} := by
sorry

end two_numbers_with_specific_means_l391_39182


namespace valid_drawings_for_ten_balls_l391_39121

/-- The number of ways to draw balls from a box -/
def validDrawings (n k : ℕ) : ℕ :=
  Nat.choose n k - Nat.choose n (k + 1)

/-- Theorem stating the number of valid ways to draw balls -/
theorem valid_drawings_for_ten_balls :
  validDrawings 10 5 = 42 := by
  sorry

end valid_drawings_for_ten_balls_l391_39121


namespace jeff_travel_distance_l391_39114

def speed1 : ℝ := 80
def time1 : ℝ := 6
def speed2 : ℝ := 60
def time2 : ℝ := 4
def speed3 : ℝ := 40
def time3 : ℝ := 2

def total_distance : ℝ := speed1 * time1 + speed2 * time2 + speed3 * time3

theorem jeff_travel_distance : total_distance = 800 := by
  sorry

end jeff_travel_distance_l391_39114


namespace undergrad_play_count_l391_39127

-- Define the total number of students
def total_students : ℕ := 800

-- Define the percentage of undergraduates who play a musical instrument
def undergrad_play_percent : ℚ := 25 / 100

-- Define the percentage of postgraduates who do not play a musical instrument
def postgrad_not_play_percent : ℚ := 20 / 100

-- Define the percentage of all students who do not play a musical instrument
def total_not_play_percent : ℚ := 355 / 1000

-- Theorem stating that the number of undergraduates who play a musical instrument is 57
theorem undergrad_play_count : ℕ := by
  sorry

end undergrad_play_count_l391_39127


namespace three_and_one_fifth_cubed_l391_39167

theorem three_and_one_fifth_cubed : (3 + 1/5) ^ 3 = 32.768 := by sorry

end three_and_one_fifth_cubed_l391_39167


namespace integer_roots_of_polynomial_l391_39192

def polynomial (x : ℤ) : ℤ := x^3 - 4*x^2 - 11*x + 24

def is_root (x : ℤ) : Prop := polynomial x = 0

theorem integer_roots_of_polynomial :
  {x : ℤ | is_root x} = {-1, 3, 4} := by sorry

end integer_roots_of_polynomial_l391_39192


namespace min_value_quadratic_expression_l391_39179

theorem min_value_quadratic_expression (x y : ℝ) :
  x^2 + y^2 - 4*x + 6*y + 20 ≥ 7 := by
sorry

end min_value_quadratic_expression_l391_39179


namespace function_properties_l391_39136

/-- Given a function f(x) = sin(ωx + φ) with the following properties:
    - ω > 0
    - 0 < φ < π
    - The distance between two adjacent zeros of f(x) is π/2
    - g(x) is f(x) shifted left by π/6 units
    - g(x) is an even function
    
    This theorem states that:
    1. f(x) = sin(2x + π/6)
    2. The axis of symmetry is x = kπ/2 + π/6 for k ∈ ℤ
    3. The interval of monotonic increase is [kπ - π/3, kπ + π/6] for k ∈ ℤ -/
theorem function_properties (ω φ : ℝ) (h_ω : ω > 0) (h_φ : 0 < φ ∧ φ < π)
  (f : ℝ → ℝ) (hf : f = fun x ↦ Real.sin (ω * x + φ))
  (h_zeros : ∀ x₁ x₂, f x₁ = 0 → f x₂ = 0 → x₁ ≠ x₂ → |x₁ - x₂| = π / 2)
  (g : ℝ → ℝ) (hg : g = fun x ↦ f (x + π / 6))
  (h_even : ∀ x, g x = g (-x)) :
  (f = fun x ↦ Real.sin (2 * x + π / 6)) ∧
  (∀ k : ℤ, ∃ x, x = k * π / 2 + π / 6 ∧ ∀ y, f (2 * x - y) = f (2 * x + y)) ∧
  (∀ k : ℤ, ∀ x, k * π - π / 3 ≤ x ∧ x ≤ k * π + π / 6 → Monotone (f ∘ (fun y ↦ y + x))) :=
by sorry

end function_properties_l391_39136


namespace min_people_like_both_l391_39159

/-- Represents the number of people who like both Vivaldi and Chopin -/
def both_like (v c b : ℕ) : Prop := b = v + c - 150

/-- The minimum number of people who like both Vivaldi and Chopin -/
def min_both_like (v c : ℕ) : ℕ := max 0 (v + c - 150)

theorem min_people_like_both (total v c : ℕ) 
  (h_total : total = 150) 
  (h_v : v = 120) 
  (h_c : c = 90) : 
  min_both_like v c = 60 := by
  sorry

#eval min_both_like 120 90

end min_people_like_both_l391_39159
