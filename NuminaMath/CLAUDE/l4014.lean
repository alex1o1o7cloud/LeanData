import Mathlib

namespace product_bcd_value_l4014_401404

theorem product_bcd_value
  (a b c d e f : ℝ)
  (h1 : a * b * c = 130)
  (h2 : c * d * e = 500)
  (h3 : d * e * f = 250)
  (h4 : (a * f) / (c * d) = 1) :
  b * c * d = 65 := by
  sorry

end product_bcd_value_l4014_401404


namespace reciprocal_sum_theorem_l4014_401479

theorem reciprocal_sum_theorem (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  x + y = 7 * x * y → 1 / x + 1 / y = 7 := by
sorry

end reciprocal_sum_theorem_l4014_401479


namespace fraction_equality_implies_numerator_equality_l4014_401411

theorem fraction_equality_implies_numerator_equality
  (a b c : ℝ) (h1 : c ≠ 0) (h2 : a / c = b / c) : a = b :=
by sorry

end fraction_equality_implies_numerator_equality_l4014_401411


namespace max_value_expressions_l4014_401475

theorem max_value_expressions (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (Real.sqrt (a / (2 * a + b)) + Real.sqrt (b / (2 * b + a)) ≤ 2 * Real.sqrt 3 / 3) ∧
  (Real.sqrt (a / (a + 2 * b)) + Real.sqrt (b / (b + 2 * a)) ≤ 2 * Real.sqrt 3 / 3) :=
by sorry

end max_value_expressions_l4014_401475


namespace pens_probability_l4014_401494

theorem pens_probability (total_pens : Nat) (defective_pens : Nat) (bought_pens : Nat) :
  total_pens = 12 →
  defective_pens = 4 →
  bought_pens = 2 →
  (total_pens - defective_pens : ℚ) / total_pens * 
  ((total_pens - defective_pens - 1 : ℚ) / (total_pens - 1)) = 14 / 33 := by
  sorry

#eval (14 : ℚ) / 33 -- To verify the approximate decimal value

end pens_probability_l4014_401494


namespace absolute_value_not_positive_l4014_401487

theorem absolute_value_not_positive (x : ℚ) : ¬(|2*x - 7| > 0) ↔ x = 7/2 := by
  sorry

end absolute_value_not_positive_l4014_401487


namespace circle_tangency_l4014_401429

/-- Two circles are externally tangent if the distance between their centers
    is equal to the sum of their radii -/
def externally_tangent (c1 c2 : ℝ × ℝ) (r1 r2 : ℝ) : Prop :=
  (c1.1 - c2.1)^2 + (c1.2 - c2.2)^2 = (r1 + r2)^2

theorem circle_tangency (t : ℝ) : 
  externally_tangent (0, 0) (t, 0) 2 1 → t = 3 ∨ t = -3 := by
  sorry

#check circle_tangency

end circle_tangency_l4014_401429


namespace sum_of_arithmetic_sequence_ending_in_seven_l4014_401413

theorem sum_of_arithmetic_sequence_ending_in_seven : 
  ∀ (a : ℕ) (d : ℕ) (n : ℕ),
    a = 107 → d = 10 → n = 40 →
    (a + (n - 1) * d = 497) →
    (n * (a + (a + (n - 1) * d))) / 2 = 12080 := by
  sorry

end sum_of_arithmetic_sequence_ending_in_seven_l4014_401413


namespace sum_of_digits_of_power_product_l4014_401437

/-- The sum of the digits of the decimal representation of 2^2010 * 5^2012 * 7 is 13 -/
theorem sum_of_digits_of_power_product : ∃ n : ℕ, 
  (n = 2^2010 * 5^2012 * 7) ∧ 
  (List.sum (Nat.digits 10 n) = 13) := by
  sorry

end sum_of_digits_of_power_product_l4014_401437


namespace clarence_oranges_left_l4014_401421

def initial_oranges : ℕ := 5
def received_oranges : ℕ := 3
def total_oranges : ℕ := initial_oranges + received_oranges

theorem clarence_oranges_left : (total_oranges / 2 : ℕ) = 4 := by
  sorry

end clarence_oranges_left_l4014_401421


namespace problem_statement_l4014_401414

theorem problem_statement (m n : ℤ) (h : m * n = m + 3) : 
  3 * m - 3 * (m * n) + 10 = 1 := by
sorry

end problem_statement_l4014_401414


namespace valid_parameterization_l4014_401449

/-- A structure representing a vector parameterization of a line -/
structure VectorParam where
  x₀ : ℝ
  y₀ : ℝ
  dx : ℝ
  dy : ℝ

/-- Predicate to check if a vector parameterization represents the line y = 2x - 7 -/
def IsValidParam (p : VectorParam) : Prop :=
  p.y₀ = 2 * p.x₀ - 7 ∧ ∃ (k : ℝ), p.dx = k * 1 ∧ p.dy = k * 2

/-- Theorem stating the conditions for a valid parameterization of y = 2x - 7 -/
theorem valid_parameterization (p : VectorParam) :
  IsValidParam p ↔ 
  ∀ (t : ℝ), (p.y₀ + t * p.dy) = 2 * (p.x₀ + t * p.dx) - 7 :=
sorry

end valid_parameterization_l4014_401449


namespace solve_equation_l4014_401430

theorem solve_equation (y z : ℝ) (h1 : y = -2.6) (h2 : z = 4.3) :
  ∃ x : ℝ, 5 * x - 2 * y + 3.7 * z = 1.45 ∧ x = -3.932 := by
  sorry

end solve_equation_l4014_401430


namespace gecko_eggs_hatched_l4014_401438

/-- The number of eggs that actually hatch from a gecko's yearly egg-laying, given the total number of eggs, infertility rate, and calcification issue rate. -/
theorem gecko_eggs_hatched (total_eggs : ℕ) (infertility_rate : ℚ) (calcification_rate : ℚ) : 
  total_eggs = 30 →
  infertility_rate = 1/5 →
  calcification_rate = 1/3 →
  (total_eggs : ℚ) * (1 - infertility_rate) * (1 - calcification_rate) = 16 := by
  sorry

end gecko_eggs_hatched_l4014_401438


namespace quadratic_solution_l4014_401435

theorem quadratic_solution : ∃ x : ℝ, x^2 - 2*x + 1 = 0 ∧ x = 1 := by
  sorry

end quadratic_solution_l4014_401435


namespace train_station_distance_l4014_401490

/-- The distance to the train station -/
def distance : ℝ := 4

/-- The speed of the man in the first scenario (km/h) -/
def speed1 : ℝ := 4

/-- The speed of the man in the second scenario (km/h) -/
def speed2 : ℝ := 5

/-- The time difference between the man's arrival and the train's arrival in the first scenario (minutes) -/
def time_diff1 : ℝ := 6

/-- The time difference between the man's arrival and the train's arrival in the second scenario (minutes) -/
def time_diff2 : ℝ := -6

theorem train_station_distance :
  (distance / speed1 - distance / speed2) * 60 = time_diff1 - time_diff2 := by sorry

end train_station_distance_l4014_401490


namespace dot_product_AB_AC_t_value_l4014_401403

-- Define the points
def A : ℝ × ℝ := (-1, -2)
def B : ℝ × ℝ := (2, 3)
def C : ℝ × ℝ := (-2, -1)

-- Define vectors
def AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)
def AC : ℝ × ℝ := (C.1 - A.1, C.2 - A.2)
def OB : ℝ × ℝ := B
def OC : ℝ × ℝ := C

-- Define dot product
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Theorem 1: Dot product of AB and AC
theorem dot_product_AB_AC : dot_product AB AC = 2 := by sorry

-- Theorem 2: Value of t
theorem t_value : ∃ t : ℝ, t = -3 ∧ dot_product (AB.1 - t * OC.1, AB.2 - t * OC.2) OB = 0 := by sorry

end dot_product_AB_AC_t_value_l4014_401403


namespace salary_comparison_l4014_401400

/-- Represents the salary distribution for graduates --/
structure SalaryDistribution where
  total_students : ℕ
  graduating_students : ℕ
  dropout_salary : ℝ
  high_salary : ℝ
  mid_salary : ℝ
  low_salary : ℝ
  default_salary : ℝ
  high_salary_ratio : ℝ
  mid_salary_ratio : ℝ
  low_salary_ratio : ℝ

/-- Represents Fyodor's salary growth --/
structure SalaryGrowth where
  initial_salary : ℝ
  yearly_increase : ℝ
  years : ℕ

/-- Calculates the expected salary based on the given distribution --/
def expected_salary (d : SalaryDistribution) : ℝ :=
  let graduate_prob := d.graduating_students / d.total_students
  let default_salary_ratio := 1 - d.high_salary_ratio - d.mid_salary_ratio - d.low_salary_ratio
  graduate_prob * (d.high_salary_ratio * d.high_salary + 
                   d.mid_salary_ratio * d.mid_salary + 
                   d.low_salary_ratio * d.low_salary + 
                   default_salary_ratio * d.default_salary) +
  (1 - graduate_prob) * d.dropout_salary

/-- Calculates Fyodor's salary after a given number of years --/
def fyodor_salary (g : SalaryGrowth) : ℝ :=
  g.initial_salary + g.yearly_increase * g.years

/-- The main theorem to prove --/
theorem salary_comparison 
  (d : SalaryDistribution)
  (g : SalaryGrowth)
  (h1 : d.total_students = 300)
  (h2 : d.graduating_students = 270)
  (h3 : d.dropout_salary = 25000)
  (h4 : d.high_salary = 60000)
  (h5 : d.mid_salary = 80000)
  (h6 : d.low_salary = 25000)
  (h7 : d.default_salary = 40000)
  (h8 : d.high_salary_ratio = 1/5)
  (h9 : d.mid_salary_ratio = 1/10)
  (h10 : d.low_salary_ratio = 1/20)
  (h11 : g.initial_salary = 25000)
  (h12 : g.yearly_increase = 3000)
  (h13 : g.years = 4)
  : expected_salary d = 39625 ∧ expected_salary d - fyodor_salary g = 2625 := by
  sorry


end salary_comparison_l4014_401400


namespace reciprocal_of_two_l4014_401408

theorem reciprocal_of_two : (2⁻¹ : ℚ) = 1/2 := by
  sorry

end reciprocal_of_two_l4014_401408


namespace no_positive_integer_divisible_by_its_square_plus_one_l4014_401496

theorem no_positive_integer_divisible_by_its_square_plus_one :
  ∀ n : ℕ, n > 0 → ¬(n^2 + 1 ∣ n) := by
  sorry

end no_positive_integer_divisible_by_its_square_plus_one_l4014_401496


namespace multiples_of_eight_range_l4014_401484

theorem multiples_of_eight_range (end_num : ℕ) (num_multiples : ℚ) : 
  end_num = 200 →
  num_multiples = 13.5 →
  ∃ (start_num : ℕ), 
    start_num = 84 ∧
    (end_num - start_num) / 8 + 1 = num_multiples ∧
    start_num ≤ end_num ∧
    ∀ n : ℕ, start_num ≤ n ∧ n ≤ end_num → (n - start_num) % 8 = 0 → n ≤ end_num :=
by sorry


end multiples_of_eight_range_l4014_401484


namespace quadratic_range_l4014_401452

def f (x : ℝ) := x^2 - 4*x + 1

theorem quadratic_range : 
  ∀ y ∈ Set.Icc (-2 : ℝ) 6, ∃ x ∈ Set.Icc 3 5, f x = y ∧
  ∀ x ∈ Set.Icc 3 5, f x ∈ Set.Icc (-2 : ℝ) 6 :=
by sorry

end quadratic_range_l4014_401452


namespace zero_exponent_is_one_l4014_401426

theorem zero_exponent_is_one (x : ℚ) (h : x ≠ 0) : x^0 = 1 := by
  sorry

end zero_exponent_is_one_l4014_401426


namespace complex_number_problem_l4014_401485

theorem complex_number_problem (z₁ z₂ : ℂ) : 
  z₁ = 1 + Complex.I * Real.sqrt 3 →
  Complex.abs z₂ = 2 →
  ∃ (r : ℝ), r > 0 ∧ z₁ * z₂ = r →
  z₂ = 1 - Complex.I * Real.sqrt 3 := by
sorry

end complex_number_problem_l4014_401485


namespace arithmetic_seq_common_diff_is_two_l4014_401425

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- Sum function
  h_arithmetic : ∀ n, a (n + 1) - a n = a 1 - a 0  -- Arithmetic property
  h_sum : ∀ n, S n = n * a 0 + (n * (n - 1) / 2) * (a 1 - a 0)  -- Sum formula

/-- The common difference of an arithmetic sequence is 2 given the condition -/
theorem arithmetic_seq_common_diff_is_two (seq : ArithmeticSequence) 
    (h : seq.S 2020 / 2020 - seq.S 20 / 20 = 2000) : 
    seq.a 1 - seq.a 0 = 2 := by
  sorry

end arithmetic_seq_common_diff_is_two_l4014_401425


namespace system_real_solutions_l4014_401431

/-- The system of equations has real solutions if and only if p ≤ 0, q ≥ 0, and p^2 - 4q ≥ 0 -/
theorem system_real_solutions (p q : ℝ) :
  (∃ (x y z : ℝ), (Real.sqrt x + Real.sqrt y = z) ∧
                   (2 * x + 2 * y + p = 0) ∧
                   (z^4 + p * z^2 + q = 0)) ↔
  (p ≤ 0 ∧ q ≥ 0 ∧ p^2 - 4*q ≥ 0) :=
by sorry

end system_real_solutions_l4014_401431


namespace office_printing_calculation_l4014_401419

/-- Calculate the number of one-page documents printed per day -/
def documents_per_day (packs : ℕ) (sheets_per_pack : ℕ) (days : ℕ) : ℕ :=
  (packs * sheets_per_pack) / days

theorem office_printing_calculation :
  documents_per_day 2 240 6 = 80 := by
  sorry

end office_printing_calculation_l4014_401419


namespace specific_jump_record_l4014_401465

/-- The standard distance for the long jump competition -/
def standard_distance : ℝ := 4.00

/-- Calculate the recorded result for a given jump distance -/
def record_jump (jump_distance : ℝ) : ℝ :=
  jump_distance - standard_distance

/-- The specific jump distance we want to prove about -/
def specific_jump : ℝ := 3.85

/-- Theorem stating that the record for the specific jump should be -0.15 -/
theorem specific_jump_record :
  record_jump specific_jump = -0.15 := by sorry

end specific_jump_record_l4014_401465


namespace smallest_root_between_3_and_4_l4014_401498

noncomputable def g (x : ℝ) : ℝ := 3 * Real.sin x - 4 * Real.cos x + 5 * Real.tan x

def is_smallest_positive_root (s : ℝ) : Prop :=
  s > 0 ∧ g s = 0 ∧ ∀ x, 0 < x ∧ x < s → g x ≠ 0

theorem smallest_root_between_3_and_4 :
  ∃ s, is_smallest_positive_root s ∧ 3 ≤ s ∧ s < 4 := by
  sorry

end smallest_root_between_3_and_4_l4014_401498


namespace largest_base5_three_digit_to_base10_l4014_401424

-- Define a function to convert a base-5 number to base 10
def base5ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (5 ^ i)) 0

-- Define the largest three-digit number in base 5
def largestBase5ThreeDigit : List Nat := [4, 4, 4]

-- Theorem statement
theorem largest_base5_three_digit_to_base10 :
  base5ToBase10 largestBase5ThreeDigit = 124 := by
  sorry

end largest_base5_three_digit_to_base10_l4014_401424


namespace clinton_belts_l4014_401420

/-- Proves that Clinton has 7 belts given the conditions -/
theorem clinton_belts :
  ∀ (shoes belts : ℕ),
  shoes = 14 →
  shoes = 2 * belts →
  belts = 7 := by
  sorry

end clinton_belts_l4014_401420


namespace grover_profit_l4014_401450

def number_of_boxes : ℕ := 3
def masks_per_box : ℕ := 20
def total_cost : ℚ := 15
def selling_price_per_mask : ℚ := 1/2

def total_masks : ℕ := number_of_boxes * masks_per_box
def total_revenue : ℚ := (total_masks : ℚ) * selling_price_per_mask
def profit : ℚ := total_revenue - total_cost

theorem grover_profit : profit = 15 := by
  sorry

end grover_profit_l4014_401450


namespace paint_usage_l4014_401439

theorem paint_usage (total_paint : ℚ) (first_week_fraction : ℚ) (second_week_fraction : ℚ) :
  total_paint = 360 →
  first_week_fraction = 1 / 9 →
  second_week_fraction = 1 / 5 →
  let first_week_usage := first_week_fraction * total_paint
  let remaining_paint := total_paint - first_week_usage
  let second_week_usage := second_week_fraction * remaining_paint
  first_week_usage + second_week_usage = 104 := by
  sorry

end paint_usage_l4014_401439


namespace middle_card_is_four_l4014_401464

/-- Represents the three cards with positive integers -/
structure Cards where
  left : ℕ+
  middle : ℕ+
  right : ℕ+
  different : left ≠ middle ∧ middle ≠ right ∧ left ≠ right
  increasing : left < middle ∧ middle < right
  sum_15 : left + middle + right = 15

/-- Casey cannot determine the other two numbers given the leftmost card -/
def casey_statement (cards : Cards) : Prop :=
  ∀ (other_cards : Cards), other_cards.left = cards.left → 
    (other_cards.middle ≠ cards.middle ∨ other_cards.right ≠ cards.right)

/-- Tracy cannot determine the other two numbers given the rightmost card -/
def tracy_statement (cards : Cards) : Prop :=
  ∀ (other_cards : Cards), other_cards.right = cards.right → 
    (other_cards.left ≠ cards.left ∨ other_cards.middle ≠ cards.middle)

/-- Stacy cannot determine the other two numbers given the middle card -/
def stacy_statement (cards : Cards) : Prop :=
  ∀ (other_cards : Cards), other_cards.middle = cards.middle → 
    (other_cards.left ≠ cards.left ∨ other_cards.right ≠ cards.right)

/-- The main theorem stating that the middle card must be 4 -/
theorem middle_card_is_four (cards : Cards) 
  (h_casey : casey_statement cards)
  (h_tracy : tracy_statement cards)
  (h_stacy : stacy_statement cards) : 
  cards.middle = 4 := by
  sorry

end middle_card_is_four_l4014_401464


namespace sum_of_cubes_of_roots_l4014_401488

/-- Given a cubic polynomial P(X) = X^3 - 3X^2 - 1 with roots r₁, r₂, r₃,
    prove that the sum of the cubes of the roots is 24. -/
theorem sum_of_cubes_of_roots (r₁ r₂ r₃ : ℂ) : 
  (r₁^3 + r₁^2 * 3 + 1 = 0) → 
  (r₂^3 + r₂^2 * 3 + 1 = 0) → 
  (r₃^3 + r₃^2 * 3 + 1 = 0) → 
  r₁^3 + r₂^3 + r₃^3 = 24 := by
sorry

end sum_of_cubes_of_roots_l4014_401488


namespace calculation_proof_l4014_401456

theorem calculation_proof : 2456 + 144 / 12 * 5 - 256 = 2260 := by
  sorry

end calculation_proof_l4014_401456


namespace absolute_value_inequality_l4014_401486

theorem absolute_value_inequality (x : ℝ) :
  |x^2 - 5*x + 3| < 9 ↔ (-1 < x ∧ x < 3) ∨ (4 < x ∧ x < 6) :=
by sorry

end absolute_value_inequality_l4014_401486


namespace egg_collection_ratio_l4014_401407

/-- 
Given:
- Benjamin collects 6 dozen eggs
- Trisha collects 4 dozen less than Benjamin
- The total eggs collected by all three is 26 dozen

Prove that the ratio of Carla's eggs to Benjamin's eggs is 3:1
-/
theorem egg_collection_ratio : 
  let benjamin_eggs : ℕ := 6
  let trisha_eggs : ℕ := benjamin_eggs - 4
  let total_eggs : ℕ := 26
  let carla_eggs : ℕ := total_eggs - benjamin_eggs - trisha_eggs
  (carla_eggs : ℚ) / benjamin_eggs = 3 / 1 := by
  sorry

end egg_collection_ratio_l4014_401407


namespace quadratic_equation_solution_l4014_401481

theorem quadratic_equation_solution :
  ∃ (x₁ x₂ : ℝ), 
    (x₁ * (5 * x₁ - 9) = -4) ∧
    (x₂ * (5 * x₂ - 9) = -4) ∧
    (x₁ = (9 + Real.sqrt 1) / 10) ∧
    (x₂ = (9 - Real.sqrt 1) / 10) ∧
    (9 + 1 + 10 = 20) := by
  sorry

end quadratic_equation_solution_l4014_401481


namespace total_flowers_in_gardens_l4014_401477

/-- Given 10 gardens, each with 544 pots, and 32 flowers per pot,
    prove that the total number of flowers in all gardens is 174,080. -/
theorem total_flowers_in_gardens : 
  let num_gardens : ℕ := 10
  let pots_per_garden : ℕ := 544
  let flowers_per_pot : ℕ := 32
  num_gardens * pots_per_garden * flowers_per_pot = 174080 :=
by sorry

end total_flowers_in_gardens_l4014_401477


namespace palindrome_count_is_420_l4014_401499

/-- Represents the count of each digit available -/
def digit_counts : List (Nat × Nat) := [(2, 2), (3, 3), (5, 4)]

/-- The total number of digits available -/
def total_digits : Nat := (digit_counts.map Prod.snd).sum

/-- A function to calculate the number of 9-digit palindromes -/
def count_palindromes (counts : List (Nat × Nat)) : Nat :=
  sorry

theorem palindrome_count_is_420 :
  total_digits = 9 ∧ count_palindromes digit_counts = 420 :=
sorry

end palindrome_count_is_420_l4014_401499


namespace root_implies_q_value_l4014_401455

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the equation
def is_root (x p q : ℂ) : Prop := x^2 + p*x + q = 0

-- State the theorem
theorem root_implies_q_value (p q : ℝ) :
  is_root (2 + 3*i) p q → q = 13 := by
  sorry

end root_implies_q_value_l4014_401455


namespace square_area_on_parabola_l4014_401440

/-- The area of a square with one side on y = 8 and endpoints on y = x^2 + 4x + 3 is 36 -/
theorem square_area_on_parabola : ∃ (x₁ x₂ : ℝ),
  (x₁^2 + 4*x₁ + 3 = 8) ∧
  (x₂^2 + 4*x₂ + 3 = 8) ∧
  (x₁ ≠ x₂) ∧
  ((x₂ - x₁)^2 = 36) := by
  sorry

end square_area_on_parabola_l4014_401440


namespace transformation_matrix_correct_l4014_401447

/-- The transformation matrix M -/
def M : Matrix (Fin 2) (Fin 2) ℝ := !![0, -2; 2, 0]

/-- Rotation matrix for 90 degrees counterclockwise -/
def R : Matrix (Fin 2) (Fin 2) ℝ := !![0, -1; 1, 0]

/-- Scaling matrix with factor 2 -/
def S : Matrix (Fin 2) (Fin 2) ℝ := !![2, 0; 0, 2]

theorem transformation_matrix_correct :
  M = S * R :=
sorry

end transformation_matrix_correct_l4014_401447


namespace smallest_four_digit_multiple_of_18_l4014_401422

theorem smallest_four_digit_multiple_of_18 :
  ∃ n : ℕ, n = 1008 ∧ 
  (∀ m : ℕ, m ≥ 1000 ∧ m < 10000 ∧ 18 ∣ m → n ≤ m) ∧
  n ≥ 1000 ∧ n < 10000 ∧ 18 ∣ n :=
by sorry

end smallest_four_digit_multiple_of_18_l4014_401422


namespace lucy_popsicles_l4014_401436

/-- The maximum number of popsicles Lucy can buy given her funds and the pricing structure -/
def max_popsicles (total_funds : ℚ) (first_tier_price : ℚ) (second_tier_price : ℚ) (first_tier_limit : ℕ) : ℕ :=
  let first_tier_cost := first_tier_limit * first_tier_price
  let remaining_funds := total_funds - first_tier_cost
  let additional_popsicles := (remaining_funds / second_tier_price).floor
  first_tier_limit + additional_popsicles.toNat

/-- Theorem stating that Lucy can buy 15 popsicles -/
theorem lucy_popsicles :
  max_popsicles 25.5 1.75 1.5 8 = 15 := by
  sorry

#eval max_popsicles 25.5 1.75 1.5 8

end lucy_popsicles_l4014_401436


namespace polynomial_division_remainder_l4014_401472

theorem polynomial_division_remainder : ∃ (q : Polynomial ℝ), 
  x^6 - x^5 - x^4 + x^3 + x^2 - x = (x^2 - 4) * (x + 1) * q + (21*x^2 - 13*x - 32) :=
by
  sorry

end polynomial_division_remainder_l4014_401472


namespace bobs_money_l4014_401406

theorem bobs_money (X : ℝ) :
  X > 0 →
  let day1_remainder := X / 2
  let day2_remainder := day1_remainder * 4 / 5
  let day3_remainder := day2_remainder * 5 / 8
  day3_remainder = 20 →
  X = 80 :=
by sorry

end bobs_money_l4014_401406


namespace f_sum_symmetric_l4014_401482

def f (x : ℝ) : ℝ := x^3 + 2*x

theorem f_sum_symmetric : f 5 + f (-5) = 0 := by
  sorry

end f_sum_symmetric_l4014_401482


namespace wrapping_paper_area_theorem_l4014_401463

/-- The area of wrapping paper required to wrap a rectangular box -/
def wrapping_paper_area (w : ℝ) (h : ℝ) : ℝ :=
  4 * (w + h)^2

/-- Theorem: The area of a square sheet of wrapping paper required to wrap a rectangular box
    with dimensions 2w × w × h, such that the corners of the paper meet at the center of the
    top of the box, is equal to 4(w + h)^2. -/
theorem wrapping_paper_area_theorem (w : ℝ) (h : ℝ) 
    (hw : w > 0) (hh : h > 0) : 
    wrapping_paper_area w h = 4 * (w + h)^2 := by
  sorry

end wrapping_paper_area_theorem_l4014_401463


namespace f_value_at_2_l4014_401401

-- Define the function f
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^5 + b * x^3 + c * x - 8

-- State the theorem
theorem f_value_at_2 (a b c : ℝ) : f a b c (-2) = 10 → f a b c 2 = -26 := by
  sorry

end f_value_at_2_l4014_401401


namespace odd_function_properties_l4014_401432

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := (b - 2^x) / (2^x + a)

theorem odd_function_properties (a b : ℝ) 
  (h_odd : ∀ x, f a b x = -f a b (-x)) :
  (a = 1 ∧ b = 1) ∧
  (∀ t : ℝ, f 1 1 (t^2 - 2*t) + f 1 1 (2*t^2 - 1) < 0 ↔ t > 1 ∨ t < -1/3) :=
sorry

end odd_function_properties_l4014_401432


namespace tomatoes_left_after_yesterday_l4014_401412

theorem tomatoes_left_after_yesterday (initial_tomatoes picked_yesterday : ℕ) : 
  initial_tomatoes = 160 → picked_yesterday = 56 → initial_tomatoes - picked_yesterday = 104 := by
  sorry

end tomatoes_left_after_yesterday_l4014_401412


namespace fuel_station_theorem_l4014_401451

/-- Represents the fuel station problem --/
def fuel_station_problem (service_cost : ℚ) (fuel_cost_per_liter : ℚ) 
  (num_minivans : ℕ) (num_trucks : ℕ) (total_cost : ℚ) (minivan_tank : ℚ) : Prop :=
  let total_service_cost := (num_minivans + num_trucks : ℚ) * service_cost
  let total_fuel_cost := total_cost - total_service_cost
  let minivan_fuel_cost := (num_minivans : ℚ) * minivan_tank * fuel_cost_per_liter
  let truck_fuel_cost := total_fuel_cost - minivan_fuel_cost
  let truck_fuel_liters := truck_fuel_cost / fuel_cost_per_liter
  let truck_tank := truck_fuel_liters / (num_trucks : ℚ)
  let percentage_increase := (truck_tank - minivan_tank) / minivan_tank * 100
  percentage_increase = 120

/-- The main theorem to be proved --/
theorem fuel_station_theorem : 
  fuel_station_problem 2.2 0.7 4 2 395.4 65 := by
  sorry

end fuel_station_theorem_l4014_401451


namespace negation_p_sufficient_not_necessary_for_q_l4014_401476

theorem negation_p_sufficient_not_necessary_for_q :
  (∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → x^2 + x - 6 < 0) ∧
  (∃ x : ℝ, x^2 + x - 6 < 0 ∧ (x < -1 ∨ x > 1)) := by sorry

end negation_p_sufficient_not_necessary_for_q_l4014_401476


namespace function_symmetry_l4014_401427

/-- Given a function f(x) = a*sin(x) - b*cos(x) where f(x) takes an extreme value when x = π/4,
    prove that y = f(3π/4 - x) is an odd function and its graph is symmetric about (π, 0) -/
theorem function_symmetry (a b : ℝ) :
  let f : ℝ → ℝ := λ x ↦ a * Real.sin x - b * Real.cos x
  (∃ (extreme : ℝ), f (π/4) = extreme ∧ ∀ x, f x ≤ extreme) →
  let y : ℝ → ℝ := λ x ↦ f (3*π/4 - x)
  (∀ x, y (-x) = -y x) ∧  -- odd function
  (∀ x, y (2*π - x) = -y x)  -- symmetry about (π, 0)
:= by sorry

end function_symmetry_l4014_401427


namespace action_figures_total_l4014_401493

theorem action_figures_total (initial : ℕ) (added : ℕ) : 
  initial = 8 → added = 2 → initial + added = 10 := by
sorry

end action_figures_total_l4014_401493


namespace night_day_crew_loading_ratio_l4014_401444

theorem night_day_crew_loading_ratio :
  ∀ (D N B : ℚ),
  N = (2/3) * D →                     -- Night crew has 2/3 as many workers as day crew
  (2/3) * B = D * (B / D) →           -- Day crew loaded 2/3 of all boxes
  (1/3) * B = N * (B / N) →           -- Night crew loaded 1/3 of all boxes
  (B / N) / (B / D) = 3/4              -- Ratio of boxes loaded by each night worker to each day worker
:= by sorry

end night_day_crew_loading_ratio_l4014_401444


namespace f_13_equals_223_l4014_401415

def f (n : ℕ) : ℕ := n^2 + n + 41

theorem f_13_equals_223 : f 13 = 223 := by
  sorry

end f_13_equals_223_l4014_401415


namespace quiz_average_change_l4014_401458

theorem quiz_average_change (total_students : ℕ) (dropped_score : ℝ) (new_average : ℝ) :
  total_students = 16 →
  dropped_score = 55 →
  new_average = 63 →
  (((total_students : ℝ) * new_average + dropped_score) / (total_students : ℝ)) = 62.5 :=
by sorry

end quiz_average_change_l4014_401458


namespace gmat_question_percentage_l4014_401454

theorem gmat_question_percentage
  (second_correct : Real)
  (neither_correct : Real)
  (both_correct : Real)
  (h1 : second_correct = 0.8)
  (h2 : neither_correct = 0.05)
  (h3 : both_correct = 0.7) :
  ∃ (first_correct : Real),
    first_correct = 0.85 ∧
    first_correct + second_correct - both_correct = 1 - neither_correct :=
by
  sorry

end gmat_question_percentage_l4014_401454


namespace carly_job_applications_l4014_401478

theorem carly_job_applications : ∃ (x : ℕ), x + 2*x = 600 ∧ x = 200 := by sorry

end carly_job_applications_l4014_401478


namespace photo_album_and_film_prices_l4014_401470

theorem photo_album_and_film_prices :
  ∀ (x y : ℚ),
    5 * x + 4 * y = 139 →
    4 * x + 5 * y = 140 →
    x = 15 ∧ y = 16 := by
  sorry

end photo_album_and_film_prices_l4014_401470


namespace quadratic_curve_point_exclusion_l4014_401480

theorem quadratic_curve_point_exclusion (a c : ℝ) (h : a * c > 0) :
  ¬∃ d : ℝ, 0 = a * 2018^2 + c * 2018 + d := by
  sorry

end quadratic_curve_point_exclusion_l4014_401480


namespace sufficient_not_necessary_condition_l4014_401489

theorem sufficient_not_necessary_condition (x y : ℝ) :
  (∀ x y : ℝ, x > 3 ∧ y > 3 → x + y > 6) ∧
  (∃ x y : ℝ, x + y > 6 ∧ ¬(x > 3 ∧ y > 3)) :=
by sorry

end sufficient_not_necessary_condition_l4014_401489


namespace equation_equivalence_l4014_401462

theorem equation_equivalence (x : ℝ) : 
  (1 - (x + 3) / 6 = x / 2) ↔ (6 - x - 3 = 3 * x) := by
  sorry

end equation_equivalence_l4014_401462


namespace minimum_framing_feet_l4014_401497

-- Define the original picture dimensions
def original_width : ℕ := 5
def original_height : ℕ := 7

-- Define the enlargement factor
def enlargement_factor : ℕ := 2

-- Define the border width
def border_width : ℕ := 3

-- Define the number of inches in a foot
def inches_per_foot : ℕ := 12

-- Theorem statement
theorem minimum_framing_feet :
  let enlarged_width := original_width * enlargement_factor
  let enlarged_height := original_height * enlargement_factor
  let total_width := enlarged_width + 2 * border_width
  let total_height := enlarged_height + 2 * border_width
  let perimeter := 2 * (total_width + total_height)
  (perimeter + inches_per_foot - 1) / inches_per_foot = 6 := by
  sorry

end minimum_framing_feet_l4014_401497


namespace sales_and_profit_l4014_401434

theorem sales_and_profit (x : ℤ) (y : ℝ) : 
  (8 ≤ x ∧ x ≤ 15) →
  (y = -5 * (x : ℝ) + 150) →
  (y = 105 ↔ x = 9) →
  (y = 95 ↔ x = 11) →
  (y = 85 ↔ x = 13) →
  (∃ (x : ℤ), 8 ≤ x ∧ x ≤ 15 ∧ (x - 8) * (-5 * x + 150) = 425 ↔ x = 13) :=
by sorry

end sales_and_profit_l4014_401434


namespace star_equality_implies_power_equality_l4014_401467

/-- The k-th smallest positive integer not in X -/
def f (X : Finset Nat) (k : Nat) : Nat :=
  (Finset.range k.succ \ X).min' sorry

/-- The * operation for finite sets of positive integers -/
def star (X Y : Finset Nat) : Finset Nat :=
  X ∪ (Y.image (f X))

/-- Repeated application of star operation -/
def repeat_star (X : Finset Nat) : Nat → Finset Nat
  | 0 => X
  | n + 1 => star X (repeat_star X n)

theorem star_equality_implies_power_equality
  (A B : Finset Nat) (a b : Nat) (ha : a > 0) (hb : b > 0) :
  star A B = star B A →
  repeat_star A b = repeat_star B a :=
sorry

end star_equality_implies_power_equality_l4014_401467


namespace circle_area_difference_l4014_401461

theorem circle_area_difference : 
  let r1 : ℝ := 30
  let d2 : ℝ := 12
  let r2 : ℝ := d2 / 2
  π * r1^2 - π * r2^2 = 864 * π := by sorry

end circle_area_difference_l4014_401461


namespace remainder_divisibility_l4014_401483

theorem remainder_divisibility (N : ℤ) : 
  (∃ k : ℤ, N = 39 * k + 15) → (∃ m : ℤ, N = 13 * m + 2) :=
by sorry

end remainder_divisibility_l4014_401483


namespace unique_arrangements_zoo_animals_l4014_401448

def num_elephants : ℕ := 4
def num_rabbits : ℕ := 3
def num_parrots : ℕ := 5

def total_animals : ℕ := num_elephants + num_rabbits + num_parrots

theorem unique_arrangements_zoo_animals :
  (Nat.factorial 3) * (Nat.factorial num_elephants) * (Nat.factorial num_rabbits) * (Nat.factorial num_parrots) = 103680 :=
by sorry

end unique_arrangements_zoo_animals_l4014_401448


namespace consecutive_integers_average_l4014_401409

theorem consecutive_integers_average (x y : ℝ) : 
  (∃ (a b : ℝ), a = x + 2 ∧ b = x + 4 ∧ y = (x + a + b) / 3) →
  (x + 3 + (x + 4) + (x + 5) + (x + 6)) / 4 = x + 4.5 := by
  sorry

end consecutive_integers_average_l4014_401409


namespace angle_U_measure_l4014_401428

/-- Represents a hexagon with specific angle properties -/
structure Hexagon where
  F : ℝ  -- Measure of angle F
  I : ℝ  -- Measure of angle I
  U : ℝ  -- Measure of angle U
  G : ℝ  -- Measure of angle G
  E : ℝ  -- Measure of angle E
  R : ℝ  -- Measure of angle R

/-- The theorem stating the property of angle U in the given hexagon -/
theorem angle_U_measure (FIGURE : Hexagon) 
  (h1 : FIGURE.F = FIGURE.I ∧ FIGURE.I = FIGURE.U)  -- ∠F ≅ ∠I ≅ ∠U
  (h2 : FIGURE.G + FIGURE.E = 180)  -- ∠G is supplementary to ∠E
  (h3 : FIGURE.R = 2 * FIGURE.U)  -- ∠R = 2∠U
  : FIGURE.U = 108 := by
  sorry

#check angle_U_measure

end angle_U_measure_l4014_401428


namespace unique_triplet_solution_l4014_401416

theorem unique_triplet_solution :
  ∀ x y z : ℕ,
    (1 + x / (y + z : ℚ))^2 + (1 + y / (z + x : ℚ))^2 + (1 + z / (x + y : ℚ))^2 = 27/4
    ↔ x = 1 ∧ y = 1 ∧ z = 1 := by
  sorry

end unique_triplet_solution_l4014_401416


namespace expression_simplification_l4014_401441

theorem expression_simplification (a b : ℤ) (h1 : a = -2) (h2 : b = 1) :
  2 * (a^2 * b + a * b^2) - 3 * (a^2 * b + 1) - 2 * a * b^2 - 2 = -9 := by
  sorry

end expression_simplification_l4014_401441


namespace soda_price_calculation_l4014_401402

/-- Calculates the price of soda cans with applicable discounts -/
theorem soda_price_calculation (regular_price : ℝ) (case_discount : ℝ) (bulk_discount : ℝ) (cans : ℕ) : 
  regular_price = 0.15 →
  case_discount = 0.12 →
  bulk_discount = 0.05 →
  cans = 75 →
  let discounted_price := regular_price * (1 - case_discount)
  let bulk_discounted_price := discounted_price * (1 - bulk_discount)
  let total_price := bulk_discounted_price * cans
  total_price = 9.405 := by
  sorry

#check soda_price_calculation

end soda_price_calculation_l4014_401402


namespace disjoint_subsets_equal_sum_l4014_401469

theorem disjoint_subsets_equal_sum (n : ℕ) (A : Finset ℕ) : 
  A.card = n → 
  (∀ a ∈ A, a > 0) → 
  A.sum id < 2^n - 1 → 
  ∃ (B C : Finset ℕ), B ⊆ A ∧ C ⊆ A ∧ B ≠ ∅ ∧ C ≠ ∅ ∧ B ∩ C = ∅ ∧ B.sum id = C.sum id :=
sorry

end disjoint_subsets_equal_sum_l4014_401469


namespace power_function_satisfies_no_equation_l4014_401466

theorem power_function_satisfies_no_equation (a : ℝ) :
  ¬(∀ x y : ℝ, (x*y)^a = x^a + y^a) ∧
  ¬(∀ x y : ℝ, (x+y)^a = x^a * y^a) ∧
  ¬(∀ x y : ℝ, (x+y)^a = x^a + y^a) :=
by sorry

end power_function_satisfies_no_equation_l4014_401466


namespace percentage_problem_l4014_401495

theorem percentage_problem (x y : ℝ) (P : ℝ) :
  (P / 100) * (x - y) = 0.3 * (x + y) →
  y = 0.4 * x →
  P = 70 := by
sorry

end percentage_problem_l4014_401495


namespace initial_distance_between_students_l4014_401443

theorem initial_distance_between_students
  (speed1 : ℝ) (speed2 : ℝ) (time : ℝ)
  (h1 : speed1 = 1.6)
  (h2 : speed2 = 1.9)
  (h3 : time = 100)
  (h4 : speed1 > 0)
  (h5 : speed2 > 0)
  (h6 : time > 0) :
  speed1 * time + speed2 * time = 350 := by
sorry

end initial_distance_between_students_l4014_401443


namespace triangle_angle_proof_l4014_401442

theorem triangle_angle_proof (a b c : ℝ) (S : ℝ) (C : ℝ) :
  a > 0 → b > 0 → c > 0 → S > 0 →
  0 < C → C < π →
  S = (1/2) * a * b * Real.sin C →
  a^2 + b^2 - c^2 = 4 * Real.sqrt 3 * S →
  C = π / 6 := by
  sorry

end triangle_angle_proof_l4014_401442


namespace cost_to_selling_price_ratio_l4014_401418

/-- Given a 50% profit percent, prove that the ratio of the cost price to the selling price is 2:3 -/
theorem cost_to_selling_price_ratio (cost_price selling_price : ℝ) 
  (h_positive : cost_price > 0 ∧ selling_price > 0)
  (h_profit : selling_price = cost_price * 1.5) : 
  cost_price / selling_price = 2 / 3 := by
sorry

end cost_to_selling_price_ratio_l4014_401418


namespace trajectory_is_ellipse_l4014_401468

/-- Definition of the set of points M(x, y) satisfying the given equation -/
def TrajectorySet : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | let (x, y) := p; Real.sqrt (x^2 + (y-3)^2) + Real.sqrt (x^2 + (y+3)^2) = 10}

/-- Definition of an ellipse with foci (0, -3) and (0, 3), and major axis length 10 -/
def EllipseSet : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | let (x, y) := p; 
               Real.sqrt (x^2 + (y+3)^2) + Real.sqrt (x^2 + (y-3)^2) = 10}

/-- Theorem stating that the trajectory set is equivalent to the ellipse set -/
theorem trajectory_is_ellipse : TrajectorySet = EllipseSet := by
  sorry


end trajectory_is_ellipse_l4014_401468


namespace goldfish_count_l4014_401445

/-- The number of goldfish in Catriona's aquarium -/
def num_goldfish : ℕ := 8

/-- The number of angelfish in Catriona's aquarium -/
def num_angelfish : ℕ := num_goldfish + 4

/-- The number of guppies in Catriona's aquarium -/
def num_guppies : ℕ := 2 * num_angelfish

/-- The total number of fish in Catriona's aquarium -/
def total_fish : ℕ := 44

/-- Theorem stating that the number of goldfish is 8 -/
theorem goldfish_count : num_goldfish = 8 ∧ 
  num_angelfish = num_goldfish + 4 ∧ 
  num_guppies = 2 * num_angelfish ∧ 
  total_fish = num_goldfish + num_angelfish + num_guppies :=
by sorry

end goldfish_count_l4014_401445


namespace binomial_18_6_l4014_401417

theorem binomial_18_6 : Nat.choose 18 6 = 18564 := by
  sorry

end binomial_18_6_l4014_401417


namespace equation_solution_l4014_401405

theorem equation_solution :
  ∃ x : ℝ, 38 + 2 * x^3 = 1250 ∧ x = (606 : ℝ)^(1/3) := by
  sorry

end equation_solution_l4014_401405


namespace sum_of_fractions_geq_three_l4014_401453

theorem sum_of_fractions_geq_three (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (1 + a^2) / (1 + a*b) + (1 + b^2) / (1 + b*c) + (1 + c^2) / (1 + c*a) ≥ 3 := by
  sorry

end sum_of_fractions_geq_three_l4014_401453


namespace greatest_three_digit_non_divisor_l4014_401491

theorem greatest_three_digit_non_divisor : ∃ n : ℕ, 
  n = 998 ∧ 
  n ≥ 100 ∧ n < 1000 ∧ 
  ∀ m : ℕ, m > n → m < 1000 → 
    (m * (m + 1) / 2 ∣ Nat.factorial (m - 1)) ∧
  ¬(n * (n + 1) / 2 ∣ Nat.factorial (n - 1)) := by
  sorry

#check greatest_three_digit_non_divisor

end greatest_three_digit_non_divisor_l4014_401491


namespace problem_solution_l4014_401460

/-- Proposition p -/
def p (m x : ℝ) : Prop := m * x + 1 > 0

/-- Proposition q -/
def q (x : ℝ) : Prop := (3 * x - 1) * (x + 2) < 0

theorem problem_solution (m : ℝ) (hm : m > 0) :
  (∃ a b : ℝ, a < b ∧ 
    (m = 1 → 
      (∀ x : ℝ, p m x ∧ q x ↔ a < x ∧ x < b) ∧
      a = -1 ∧ b = 1/3)) ∧
  (∀ x : ℝ, q x → p m x) ∧ 
  (∃ x : ℝ, p m x ∧ ¬q x) →
  0 < m ∧ m ≤ 1/2 :=
sorry

end problem_solution_l4014_401460


namespace season_games_count_l4014_401492

/-- The number of baseball games in a month -/
def games_per_month : ℕ := 7

/-- The number of months in a season -/
def months_in_season : ℕ := 2

/-- The total number of baseball games in a season -/
def total_games : ℕ := games_per_month * months_in_season

theorem season_games_count : total_games = 14 := by
  sorry

end season_games_count_l4014_401492


namespace f_composition_half_equals_one_l4014_401459

-- Define the function f
def f (x : ℝ) : ℝ := |x - 1| - |x|

-- State the theorem
theorem f_composition_half_equals_one : f (f (1/2)) = 1 := by
  sorry

end f_composition_half_equals_one_l4014_401459


namespace liz_additional_money_needed_l4014_401474

def original_price : ℝ := 32500
def new_car_price : ℝ := 30000
def sale_percentage : ℝ := 0.8

theorem liz_additional_money_needed :
  new_car_price - sale_percentage * original_price = 4000 := by
  sorry

end liz_additional_money_needed_l4014_401474


namespace basketball_score_difference_l4014_401433

theorem basketball_score_difference 
  (tim joe ken : ℕ) 
  (h1 : tim > joe)
  (h2 : tim = ken / 2)
  (h3 : tim + joe + ken = 100)
  (h4 : tim = 30) :
  tim - joe = 20 := by
  sorry

end basketball_score_difference_l4014_401433


namespace quadratic_solution_difference_l4014_401410

theorem quadratic_solution_difference : ∃ (x₁ x₂ : ℝ),
  (x₁^2 - 5*x₁ + 16 = x₁ + 55) ∧
  (x₂^2 - 5*x₂ + 16 = x₂ + 55) ∧
  x₁ ≠ x₂ ∧
  |x₁ - x₂| = 16 := by
  sorry

end quadratic_solution_difference_l4014_401410


namespace quadratic_equation_negative_root_l4014_401473

theorem quadratic_equation_negative_root (a : ℝ) :
  (∃ x : ℝ, x < 0 ∧ a * x^2 + 2 * x + 1 = 0) ↔ (0 < a ∧ a ≤ 1) ∨ a < 0 :=
sorry

end quadratic_equation_negative_root_l4014_401473


namespace price_reduction_percentage_l4014_401471

theorem price_reduction_percentage (original_price reduction_amount : ℝ) :
  original_price = 500 →
  reduction_amount = 400 →
  (reduction_amount / original_price) * 100 = 80 := by
  sorry

end price_reduction_percentage_l4014_401471


namespace michaels_weight_loss_goal_l4014_401457

/-- The total weight Michael wants to lose by June -/
def total_weight_loss (march_loss april_loss may_loss : ℕ) : ℕ :=
  march_loss + april_loss + may_loss

/-- Proof that Michael's total weight loss goal is 10 pounds -/
theorem michaels_weight_loss_goal :
  ∃ (march_loss april_loss may_loss : ℕ),
    march_loss = 3 ∧
    april_loss = 4 ∧
    may_loss = 3 ∧
    total_weight_loss march_loss april_loss may_loss = 10 :=
by
  sorry

end michaels_weight_loss_goal_l4014_401457


namespace urn_probability_l4014_401423

/-- Represents the contents of the urn -/
structure UrnContents :=
  (red : ℕ)
  (blue : ℕ)

/-- The operation of drawing a ball and adding another of the same color -/
def draw_and_add (contents : UrnContents) : UrnContents → ℕ → ℝ
  | contents, n => sorry

/-- The probability of having a specific urn content after n operations -/
def prob_after_operations (initial : UrnContents) (final : UrnContents) (n : ℕ) : ℝ :=
  sorry

/-- The probability of removing a specific color ball given the urn contents -/
def prob_remove_color (contents : UrnContents) (remove_red : Bool) : ℝ :=
  sorry

/-- The main theorem to prove -/
theorem urn_probability :
  let initial := UrnContents.mk 2 1
  let final := UrnContents.mk 4 4
  let operations := 6
  (prob_after_operations initial (UrnContents.mk 5 4) operations *
   prob_remove_color (UrnContents.mk 5 4) true) = 5/63 :=
by sorry

end urn_probability_l4014_401423


namespace sliced_meat_cost_l4014_401446

/-- Given a 4-pack of sliced meat costing $40.00 with a 30% rush delivery fee,
    the cost per type of sliced meat is $13.00. -/
theorem sliced_meat_cost (pack_cost : ℝ) (num_types : ℕ) (rush_fee_percent : ℝ) :
  pack_cost = 40 →
  num_types = 4 →
  rush_fee_percent = 0.3 →
  (pack_cost + pack_cost * rush_fee_percent) / num_types = 13 := by
  sorry

end sliced_meat_cost_l4014_401446
