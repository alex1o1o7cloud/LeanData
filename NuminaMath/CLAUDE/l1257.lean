import Mathlib

namespace unique_integer_sequence_l1257_125733

theorem unique_integer_sequence : ∃! x : ℤ, x = ((x + 2)/2 + 2)/2 + 1 := by
  sorry

end unique_integer_sequence_l1257_125733


namespace no_positive_reals_satisfy_inequalities_l1257_125715

theorem no_positive_reals_satisfy_inequalities :
  ¬ ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
    (4 * (a * b + b * c + c * a) - 1 ≥ a^2 + b^2 + c^2) ∧
    (a^2 + b^2 + c^2 ≥ 3 * (a^3 + b^3 + c^3)) := by
  sorry

end no_positive_reals_satisfy_inequalities_l1257_125715


namespace statement_two_is_false_l1257_125727

/-- Definition of the heart operation -/
def heart (x y : ℝ) : ℝ := 2 * |x - y| + 1

/-- Theorem stating that Statement 2 is false -/
theorem statement_two_is_false :
  ∃ x y : ℝ, 3 * (heart x y) ≠ heart (3 * x) (3 * y) :=
sorry

end statement_two_is_false_l1257_125727


namespace not_sum_solution_equation_example_sum_solution_equation_condition_l1257_125720

/-- Definition of a sum solution equation -/
def is_sum_solution_equation (a b : ℝ) : Prop :=
  (b / a) = b + a

/-- Theorem 1: 3x = 4.5 is not a sum solution equation -/
theorem not_sum_solution_equation_example : ¬ is_sum_solution_equation 3 4.5 := by
  sorry

/-- Theorem 2: 5x = m + 1 is a sum solution equation iff m = -29/4 -/
theorem sum_solution_equation_condition (m : ℝ) : 
  is_sum_solution_equation 5 (m + 1) ↔ m = -29/4 := by
  sorry

end not_sum_solution_equation_example_sum_solution_equation_condition_l1257_125720


namespace two_over_x_is_proper_convert_improper_to_mixed_integer_values_for_integer_result_l1257_125704

-- Define proper and improper expressions
def is_proper_expression (num denom : Polynomial ℚ) : Prop :=
  num.degree < denom.degree

def is_improper_expression (num denom : Polynomial ℚ) : Prop :=
  num.degree ≥ denom.degree

-- Statement 1
theorem two_over_x_is_proper :
  is_proper_expression (2 : Polynomial ℚ) (X : Polynomial ℚ) :=
sorry

-- Statement 2
theorem convert_improper_to_mixed :
  (X^2 - 1) / (X + 2) = X - 2 + 3 / (X + 2) :=
sorry

-- Statement 3
theorem integer_values_for_integer_result :
  {x : ℤ | ∃ (y : ℤ), (2*x - 1) / (x + 1) = y} = {0, -2, 2, -4} :=
sorry

end two_over_x_is_proper_convert_improper_to_mixed_integer_values_for_integer_result_l1257_125704


namespace line_through_points_2m_minus_b_l1257_125793

/-- Given a line passing through points (1,3) and (4,15), prove that 2m - b = 9 where y = mx + b is the equation of the line. -/
theorem line_through_points_2m_minus_b (m b : ℝ) : 
  (3 : ℝ) = m * 1 + b → 
  (15 : ℝ) = m * 4 + b → 
  2 * m - b = 9 := by
  sorry

end line_through_points_2m_minus_b_l1257_125793


namespace log7_10_approximation_l1257_125770

-- Define the approximations given in the problem
def log10_2_approx : ℝ := 0.301
def log10_3_approx : ℝ := 0.499

-- Define the target approximation
def log7_10_approx : ℝ := 2

-- Theorem statement
theorem log7_10_approximation :
  abs (Real.log 10 / Real.log 7 - log7_10_approx) < 0.1 :=
sorry


end log7_10_approximation_l1257_125770


namespace sufficient_condition_range_l1257_125784

-- Define the conditions
def p (x : ℝ) : Prop := |x - 4| ≤ 6
def q (x m : ℝ) : Prop := x ≤ 1 + m

-- State the theorem
theorem sufficient_condition_range (m : ℝ) :
  (∀ x, p x → q x m) ∧ (∃ x, q x m ∧ ¬p x) ↔ m ≥ 9 :=
sorry

end sufficient_condition_range_l1257_125784


namespace difference_between_fractions_l1257_125790

theorem difference_between_fractions (n : ℝ) : n = 100 → (3/5 * n) - (1/2 * n) = 10 := by
  sorry

end difference_between_fractions_l1257_125790


namespace problem_solution_l1257_125702

theorem problem_solution (a : ℝ) (f g : ℝ → ℝ) 
  (h1 : a > 0) 
  (h2 : f (g a) = 18)
  (h3 : ∀ x, f x = x^2 - 2)
  (h4 : ∀ x, g x = x^2 + 6) : 
  a = Real.sqrt 14 := by
sorry

end problem_solution_l1257_125702


namespace valid_arrangements_l1257_125735

/-- The number of boys. -/
def num_boys : ℕ := 4

/-- The number of girls. -/
def num_girls : ℕ := 3

/-- The number of people to be selected. -/
def num_selected : ℕ := 3

/-- The number of tasks. -/
def num_tasks : ℕ := 3

/-- The function to calculate the number of permutations. -/
def permutations (n k : ℕ) : ℕ := 
  Nat.factorial n / Nat.factorial (n - k)

/-- The theorem stating the number of valid arrangements. -/
theorem valid_arrangements : 
  permutations (num_boys + num_girls) num_selected - 
  permutations num_boys num_selected = 186 := by
  sorry

end valid_arrangements_l1257_125735


namespace square_difference_simplification_l1257_125780

theorem square_difference_simplification (y : ℝ) (h : y^2 ≥ 16) :
  (4 - Real.sqrt (y^2 - 16))^2 = y^2 - 8 * Real.sqrt (y^2 - 16) := by
  sorry

end square_difference_simplification_l1257_125780


namespace stating_probability_of_target_sequence_l1257_125741

/-- The number of balls in the box -/
def total_balls : ℕ := 500

/-- The number of balls selected -/
def selections : ℕ := 5

/-- The probability of selecting an odd-numbered ball -/
def prob_odd : ℚ := 1 / 2

/-- The probability of selecting an even-numbered ball -/
def prob_even : ℚ := 1 / 2

/-- The sequence of selections we're interested in (odd, even, odd, even, odd) -/
def target_sequence : List Bool := [true, false, true, false, true]

/-- 
Theorem stating that the probability of selecting the target sequence 
(odd, even, odd, even, odd) from a box of 500 balls numbered 1 to 500, 
with 5 selections and replacement, is 1/32.
-/
theorem probability_of_target_sequence : 
  (List.prod (target_sequence.map (fun b => if b then prob_odd else prob_even))) = (1 : ℚ) / 32 := by
  sorry

end stating_probability_of_target_sequence_l1257_125741


namespace sum_of_roots_sum_of_roots_is_twenty_l1257_125747

/-- Square with sides parallel to coordinate axes -/
structure Square :=
  (side_length : ℝ)
  (bottom_left : ℝ × ℝ)

/-- Parabola defined by y = (1/5)x^2 + ax + b -/
structure Parabola :=
  (a : ℝ)
  (b : ℝ)

/-- Configuration of square and parabola -/
structure Configuration :=
  (square : Square)
  (parabola : Parabola)
  (passes_through_B : Bool)
  (passes_through_C : Bool)
  (vertex_on_AD : Bool)

/-- Theorem: Sum of roots of quadratic polynomial -/
theorem sum_of_roots (config : Configuration) : ℝ :=
  20

/-- Main theorem: Sum of roots is 20 -/
theorem sum_of_roots_is_twenty (config : Configuration) :
  sum_of_roots config = 20 := by
  sorry

end sum_of_roots_sum_of_roots_is_twenty_l1257_125747


namespace olympiad_numbers_equal_divisors_of_1998_l1257_125728

/-- The year of the first Olympiad -/
def firstOlympiadYear : ℕ := 1999

/-- The year of the n-th Olympiad -/
def olympiadYear (n : ℕ) : ℕ := firstOlympiadYear + n - 1

/-- The set of positive integers n such that n divides the year of the n-th Olympiad -/
def validOlympiadNumbers : Set ℕ :=
  {n : ℕ | n > 0 ∧ n ∣ olympiadYear n}

/-- The set of divisors of 1998 -/
def divisorsOf1998 : Set ℕ :=
  {n : ℕ | n > 0 ∧ n ∣ 1998}

theorem olympiad_numbers_equal_divisors_of_1998 :
  validOlympiadNumbers = divisorsOf1998 :=
by sorry

end olympiad_numbers_equal_divisors_of_1998_l1257_125728


namespace arithmetic_sequence_ninth_term_l1257_125701

/-- 
Given an arithmetic sequence where:
- a is the first term
- d is the common difference
- The first term (a) is 5/6
- The seventeenth term (a + 16d) is 5/8

Prove that the ninth term (a + 8d) is 15/16
-/
theorem arithmetic_sequence_ninth_term 
  (a d : ℚ) 
  (h1 : a = 5/6) 
  (h2 : a + 16*d = 5/8) : 
  a + 8*d = 15/16 := by
  sorry


end arithmetic_sequence_ninth_term_l1257_125701


namespace wax_already_possessed_l1257_125751

/-- Given the total amount of wax needed and the additional amount required,
    calculate the amount of wax already possessed. -/
theorem wax_already_possessed
  (total_wax : ℕ)
  (additional_wax : ℕ)
  (h1 : total_wax = 288)
  (h2 : additional_wax = 260)
  : total_wax - additional_wax = 28 :=
by sorry

end wax_already_possessed_l1257_125751


namespace f_monotonicity_and_inequality_l1257_125758

noncomputable def f (x : ℝ) : ℝ := x^3 - 3 * Real.log x + 11

theorem f_monotonicity_and_inequality :
  (∀ x ∈ Set.Ioo (0 : ℝ) 1, StrictMonoOn f (Set.Ioo 0 1)) ∧
  (∀ x ∈ Set.Ioi 1, StrictMonoOn f (Set.Ioi 1)) ∧
  (∀ x > 0, f x > -x^3 + 3*x^2 + (3 - x) * Real.exp x) := by
  sorry

end f_monotonicity_and_inequality_l1257_125758


namespace olivia_payment_l1257_125796

/-- The number of quarters in a dollar -/
def quarters_per_dollar : ℕ := 4

/-- The number of quarters Olivia pays for chips -/
def chips_quarters : ℕ := 4

/-- The number of quarters Olivia pays for soda -/
def soda_quarters : ℕ := 12

/-- The total amount Olivia pays in dollars -/
def total_dollars : ℚ := (chips_quarters + soda_quarters) / quarters_per_dollar

theorem olivia_payment :
  total_dollars = 4 := by sorry

end olivia_payment_l1257_125796


namespace existence_of_special_number_l1257_125772

/-- A function that returns the decimal representation of a natural number as a list of digits -/
def decimal_representation (n : ℕ) : List ℕ := sorry

/-- A function that counts the occurrences of a digit in a list of digits -/
def count_occurrences (digit : ℕ) (digits : List ℕ) : ℕ := sorry

/-- A function that interchanges two digits at given positions in a list of digits -/
def interchange_digits (digits : List ℕ) (pos1 pos2 : ℕ) : List ℕ := sorry

/-- A function that converts a list of digits back to a natural number -/
def from_digits (digits : List ℕ) : ℕ := sorry

/-- The set of prime divisors of a natural number -/
def prime_divisors (n : ℕ) : Set ℕ := sorry

theorem existence_of_special_number :
  ∃ n : ℕ,
    (∀ d : ℕ, d < 10 → count_occurrences d (decimal_representation n) ≥ 2006) ∧
    (∃ pos1 pos2 : ℕ,
      pos1 ≠ pos2 ∧
      let digits := decimal_representation n
      let m := from_digits (interchange_digits digits pos1 pos2)
      n ≠ m ∧
      prime_divisors n = prime_divisors m) :=
sorry

end existence_of_special_number_l1257_125772


namespace sixth_group_frequency_is_one_tenth_l1257_125782

/-- Represents the distribution of students across six groups in a mathematics competition. -/
structure StudentDistribution where
  total : ℕ
  group1 : ℕ
  group2 : ℕ
  group3 : ℕ
  group4 : ℕ
  freq5 : ℚ

/-- Calculates the frequency of the sixth group given a student distribution. -/
def sixthGroupFrequency (d : StudentDistribution) : ℚ :=
  1 - (d.group1 + d.group2 + d.group3 + d.group4 : ℚ) / d.total - d.freq5

/-- Theorem stating that for the given distribution, the frequency of the sixth group is 0.1. -/
theorem sixth_group_frequency_is_one_tenth 
  (d : StudentDistribution)
  (h1 : d.total = 40)
  (h2 : d.group1 = 10)
  (h3 : d.group2 = 5)
  (h4 : d.group3 = 7)
  (h5 : d.group4 = 6)
  (h6 : d.freq5 = 1/5) :
  sixthGroupFrequency d = 1/10 := by
  sorry


end sixth_group_frequency_is_one_tenth_l1257_125782


namespace inverse_sum_property_l1257_125717

-- Define the function f and its properties
def f_property (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x + f (-x) = 2

-- Define the inverse function property
def has_inverse (f : ℝ → ℝ) : Prop :=
  ∃ g : ℝ → ℝ, (∀ x, g (f x) = x) ∧ (∀ y, f (g y) = y)

-- State the theorem
theorem inverse_sum_property
  (f : ℝ → ℝ)
  (h_inv : has_inverse f)
  (h_prop : f_property f) :
  ∀ x : ℝ, f⁻¹ (2008 - x) + f⁻¹ (x - 2006) = 0 :=
sorry

end inverse_sum_property_l1257_125717


namespace complex_number_in_first_quadrant_l1257_125755

theorem complex_number_in_first_quadrant : ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ (1 : ℂ) / (1 - Complex.I) = ↑a + ↑b * Complex.I :=
sorry

end complex_number_in_first_quadrant_l1257_125755


namespace assembly_time_proof_l1257_125762

/-- Calculates the total time spent assembling furniture -/
def total_assembly_time (chairs tables time_per_piece : ℕ) : ℕ :=
  (chairs + tables) * time_per_piece

/-- Proves that given 20 chairs, 8 tables, and 6 minutes per piece, 
    the total assembly time is 168 minutes -/
theorem assembly_time_proof :
  total_assembly_time 20 8 6 = 168 := by
  sorry

end assembly_time_proof_l1257_125762


namespace xy_values_l1257_125776

theorem xy_values (x y : ℝ) (h1 : x - y = 6) (h2 : x^3 - y^3 = 108) : 
  xy = 0 ∨ xy = 72 := by
sorry

end xy_values_l1257_125776


namespace systematic_sampling_interval_l1257_125775

/-- Calculates the sampling interval for systematic sampling -/
def sampling_interval (N : ℕ) (n : ℕ) : ℕ := N / n

/-- Theorem: The sampling interval for a population of 1000 and sample size of 20 is 50 -/
theorem systematic_sampling_interval :
  sampling_interval 1000 20 = 50 := by
  sorry

end systematic_sampling_interval_l1257_125775


namespace expand_expression_l1257_125769

theorem expand_expression (x : ℝ) : 5 * (4 * x^3 - 3 * x^2 + 2 * x - 7) = 20 * x^3 - 15 * x^2 + 10 * x - 35 := by
  sorry

end expand_expression_l1257_125769


namespace problem_solution_l1257_125729

/-- Permutations of n items taken r at a time -/
def permutations (n : ℕ) (r : ℕ) : ℕ := sorry

/-- Combinations of n items taken r at a time -/
def combinations (n : ℕ) (r : ℕ) : ℕ := sorry

theorem problem_solution (r : ℕ) (k : ℕ) : 
  permutations 32 r = k * combinations 32 r → k = 720 → r = 6 := by
  sorry

end problem_solution_l1257_125729


namespace ellipse_property_l1257_125785

/-- Given an ellipse with foci at (2, 0) and (8, 0) passing through (5, 3),
    prove that the sum of its semi-major axis length and the y-coordinate of its center is 3√2. -/
theorem ellipse_property (a b h k : ℝ) : 
  a > 0 → b > 0 →
  (∀ x y : ℝ, (x - h)^2 / a^2 + (y - k)^2 / b^2 = 1 ↔ 
    (x - 2)^2 + y^2 + (x - 8)^2 + y^2 = ((x - 2)^2 + y^2 + (x - 8)^2 + y^2)) →
  (5 - h)^2 / a^2 + (3 - k)^2 / b^2 = 1 →
  a + k = 3 * Real.sqrt 2 :=
by sorry

end ellipse_property_l1257_125785


namespace sum_of_abs_values_l1257_125724

theorem sum_of_abs_values (a b : ℝ) : 
  (|a| = 3 ∧ |b| = 5 ∧ a > b) → (a + b = -2 ∨ a + b = -8) := by
sorry

end sum_of_abs_values_l1257_125724


namespace standard_poodle_height_difference_l1257_125721

/-- The height difference between the standard poodle and the miniature poodle -/
def height_difference (standard_height miniature_height : ℕ) : ℕ :=
  standard_height - miniature_height

/-- Theorem: The standard poodle is 8 inches taller than the miniature poodle -/
theorem standard_poodle_height_difference :
  let toy_height : ℕ := 14
  let standard_height : ℕ := 28
  let miniature_height : ℕ := toy_height + 6
  height_difference standard_height miniature_height = 8 := by
  sorry

end standard_poodle_height_difference_l1257_125721


namespace enrollment_difference_l1257_125718

def highest_enrollment : ℕ := 2150
def lowest_enrollment : ℕ := 980

theorem enrollment_difference : highest_enrollment - lowest_enrollment = 1170 := by
  sorry

end enrollment_difference_l1257_125718


namespace total_ways_from_A_to_C_l1257_125760

/-- The number of roads from village A to village B -/
def roads_A_to_B : ℕ := 3

/-- The number of roads from village B to village C -/
def roads_B_to_C : ℕ := 2

/-- The total number of different ways to go from village A to village C via village B -/
def total_ways : ℕ := roads_A_to_B * roads_B_to_C

theorem total_ways_from_A_to_C : total_ways = 6 := by
  sorry

end total_ways_from_A_to_C_l1257_125760


namespace gino_bears_count_l1257_125753

/-- The number of brown bears Gino has -/
def brown_bears : ℕ := 15

/-- The number of white bears Gino has -/
def white_bears : ℕ := 24

/-- The number of black bears Gino has -/
def black_bears : ℕ := 27

/-- The total number of bears Gino has -/
def total_bears : ℕ := brown_bears + white_bears + black_bears

theorem gino_bears_count : total_bears = 66 := by
  sorry

end gino_bears_count_l1257_125753


namespace password_probability_l1257_125798

def password_space : ℕ := 10 * 52 * 52 * 10

def even_start_space : ℕ := 5 * 52 * 52 * 10

def diff_letters_space : ℕ := 10 * 52 * 51 * 10

def non_zero_end_space : ℕ := 10 * 52 * 52 * 9

def valid_password_space : ℕ := 5 * 52 * 51 * 9

theorem password_probability :
  (valid_password_space : ℚ) / password_space = 459 / 1040 := by
  sorry

end password_probability_l1257_125798


namespace inequality_proof_l1257_125714

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (b + c) / a + (c + a) / b + (a + b) / c ≥ 
  ((a^2 + b^2 + c^2) * (a*b + b*c + c*a)) / (a*b*c * (a + b + c)) + 3 := by
  sorry

end inequality_proof_l1257_125714


namespace incorrect_statement_is_E_l1257_125763

theorem incorrect_statement_is_E :
  -- Statement A
  (∀ (a b c : ℝ), c > 0 → (a > b ↔ a + c > b + c)) ∧
  (∀ (a b c : ℝ), c > 0 → (a > b ↔ a * c > b * c)) ∧
  (∀ (a b c : ℝ), c > 0 → (a > b ↔ a / c > b / c)) ∧
  -- Statement B
  (∀ (a b : ℝ), a > 0 → b > 0 → a ≠ b → (a + b) / 2 > Real.sqrt (a * b)) ∧
  -- Statement C
  (∀ (s : ℝ), s > 0 → ∃ (x : ℝ), x > 0 ∧ x < s ∧
    ∀ (y : ℝ), y > 0 → y < s → x * (s - x) ≥ y * (s - y)) ∧
  -- Statement D
  (∀ (a b : ℝ), a > 0 → b > 0 → a ≠ b →
    (a^2 + b^2) / 2 > ((a + b) / 2)^2) ∧
  -- Statement E (negation)
  (∃ (p : ℝ), p > 0 ∧ ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x * y = p ∧ x + y > 2 * Real.sqrt p) :=
by sorry

end incorrect_statement_is_E_l1257_125763


namespace simplify_fraction_l1257_125711

theorem simplify_fraction : (48 : ℚ) / 72 = 2 / 3 := by
  sorry

end simplify_fraction_l1257_125711


namespace camryn_practice_schedule_l1257_125726

theorem camryn_practice_schedule :
  let trumpet := 11
  let flute := 3
  let piano := 7
  let violin := 13
  let guitar := 5
  Nat.lcm trumpet (Nat.lcm flute (Nat.lcm piano (Nat.lcm violin guitar))) = 15015 := by
  sorry

end camryn_practice_schedule_l1257_125726


namespace alan_phone_price_l1257_125730

theorem alan_phone_price (john_price : ℝ) (percentage : ℝ) (alan_price : ℝ) :
  john_price = 2040 →
  percentage = 0.02 →
  john_price = alan_price * (1 + percentage) →
  alan_price = 1999.20 := by
sorry

end alan_phone_price_l1257_125730


namespace jordana_age_proof_l1257_125700

/-- Jennifer's age in ten years -/
def jennifer_future_age : ℕ := 30

/-- Number of years in the future we're considering -/
def years_ahead : ℕ := 10

/-- Jordana's age relative to Jennifer's in the future -/
def jordana_relative_age : ℕ := 3

/-- Calculate Jordana's current age -/
def jordana_current_age : ℕ :=
  jennifer_future_age * jordana_relative_age - years_ahead

theorem jordana_age_proof :
  jordana_current_age = 80 := by
  sorry

end jordana_age_proof_l1257_125700


namespace parabola_line_intersection_l1257_125797

/-- A parabola intersects a line at exactly one point if and only if b = 49/12 -/
theorem parabola_line_intersection (b : ℝ) : 
  (∃! x : ℝ, bx^2 + 5*x + 4 = -2*x + 1) ↔ b = 49/12 := by
sorry

end parabola_line_intersection_l1257_125797


namespace arithmetic_sequence_m_l1257_125748

/-- An arithmetic sequence with its first n terms sum -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  d : ℝ      -- Common difference
  S : ℕ → ℝ  -- Sum of first n terms
  is_arithmetic : ∀ n, a (n + 1) = a n + d
  sum_formula : ∀ n, S n = n * (2 * a 1 + (n - 1) * d) / 2

/-- The main theorem -/
theorem arithmetic_sequence_m (seq : ArithmeticSequence) (m : ℕ) :
  m ≥ 2 →
  seq.S (m - 1) = -2 →
  seq.S m = 0 →
  seq.S (m + 1) = 3 →
  m = 5 := by
  sorry

end arithmetic_sequence_m_l1257_125748


namespace max_area_triangle_area_equals_perimeter_l1257_125719

theorem max_area_triangle_area_equals_perimeter : ∃ (a b c : ℕ+),
  (∃ (s : ℝ), s = (a + b + c : ℝ) / 2 ∧ 
   (s * (s - a) * (s - b) * (s - c) : ℝ) = ((a + b + c) ^ 2 : ℝ) / 4) ∧
  (∀ (x y z : ℕ+), 
    (∃ (t : ℝ), t = (x + y + z : ℝ) / 2 ∧ 
     (t * (t - x) * (t - y) * (t - z) : ℝ) = ((x + y + z) ^ 2 : ℝ) / 4) →
    (x + y + z : ℝ) ≤ (a + b + c : ℝ)) :=
by sorry

#check max_area_triangle_area_equals_perimeter

end max_area_triangle_area_equals_perimeter_l1257_125719


namespace point_y_coordinate_l1257_125773

/-- A straight line in the xy-plane -/
structure Line where
  slope : ℝ
  y_intercept : ℝ

/-- A point in the xy-plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Given line has slope 2 and y-intercept 2 -/
def given_line : Line :=
  { slope := 2, y_intercept := 2 }

/-- The x-coordinate of the point in question is 239 -/
def given_x : ℝ := 239

/-- A point is on a line if its coordinates satisfy the line equation -/
def point_on_line (p : Point) (l : Line) : Prop :=
  p.y = l.slope * p.x + l.y_intercept

/-- Theorem: The point on the given line with x-coordinate 239 has y-coordinate 480 -/
theorem point_y_coordinate :
  ∃ p : Point, p.x = given_x ∧ point_on_line p given_line ∧ p.y = 480 :=
sorry

end point_y_coordinate_l1257_125773


namespace stove_and_wall_repair_cost_l1257_125708

/-- The total cost of replacing a stove and repairing wall damage -/
theorem stove_and_wall_repair_cost :
  let stove_cost : ℚ := 1200
  let wall_repair_cost : ℚ := stove_cost / 6
  let total_cost : ℚ := stove_cost + wall_repair_cost
  total_cost = 1400 := by sorry

end stove_and_wall_repair_cost_l1257_125708


namespace min_value_of_complex_expression_l1257_125791

theorem min_value_of_complex_expression :
  ∃ (min_u : ℝ), min_u = (3/2) * Real.sqrt 3 ∧
  ∀ (z : ℂ), Complex.abs z = 2 →
  Complex.abs (z^2 - z + 1) ≥ min_u :=
sorry

end min_value_of_complex_expression_l1257_125791


namespace no_integer_solutions_l1257_125750

theorem no_integer_solutions : ¬ ∃ (a b : ℤ), 3 * a^2 = b^2 + 1 := by
  sorry

end no_integer_solutions_l1257_125750


namespace distinct_colorings_count_l1257_125705

/-- The number of distinct colorings of n points on a circle with k blue points
    and at least p red points between each pair of consecutive blue points. -/
def distinctColorings (n k p : ℕ) : ℚ :=
  if 2 ≤ k ∧ k ≤ n / (p + 1) then
    (1 : ℚ) / k * (Nat.choose (n - k * p - 1) (k - 1) : ℚ)
  else
    0

theorem distinct_colorings_count
  (n k p : ℕ)
  (h1 : 0 < n ∧ 0 < k ∧ 0 < p)
  (h2 : 2 ≤ k)
  (h3 : k ≤ n / (p + 1)) :
  distinctColorings n k p = (1 : ℚ) / k * (Nat.choose (n - k * p - 1) (k - 1) : ℚ) :=
by sorry

end distinct_colorings_count_l1257_125705


namespace additional_machines_needed_l1257_125712

/-- Given that 15 machines can finish a job in 36 days, prove that 5 additional machines
    are needed to finish the job in one-fourth less time. -/
theorem additional_machines_needed (machines : ℕ) (days : ℕ) (job : ℕ) :
  machines = 15 →
  days = 36 →
  job = machines * days →
  (machines + 5) * (days - days / 4) = job :=
by sorry

end additional_machines_needed_l1257_125712


namespace sum_of_cubes_l1257_125787

theorem sum_of_cubes (a b : ℝ) (h1 : a + b = 7) (h2 : a * b = 3) : a^3 + b^3 = 280 := by
  sorry

end sum_of_cubes_l1257_125787


namespace fraction_equality_l1257_125799

theorem fraction_equality : (4^3 : ℝ) / (10^2 - 6^2) = 1 := by sorry

end fraction_equality_l1257_125799


namespace simplify_sqrt_240_l1257_125707

theorem simplify_sqrt_240 : Real.sqrt 240 = 4 * Real.sqrt 15 := by
  sorry

end simplify_sqrt_240_l1257_125707


namespace teaching_ratio_l1257_125765

def total_years : ℕ := 52
def calculus_years : ℕ := 4

def algebra_years (c : ℕ) : ℕ := 2 * c

def statistics_years (t a c : ℕ) : ℕ := t - a - c

theorem teaching_ratio :
  let c := calculus_years
  let a := algebra_years c
  let s := statistics_years total_years a c
  (s : ℚ) / a = 5 / 1 :=
sorry

end teaching_ratio_l1257_125765


namespace students_left_proof_l1257_125756

/-- The number of students who showed up initially -/
def initial_students : ℕ := 16

/-- The number of students who were checked out early -/
def checked_out_students : ℕ := 7

/-- The number of students left at the end of the day -/
def remaining_students : ℕ := initial_students - checked_out_students

theorem students_left_proof : remaining_students = 9 := by
  sorry

end students_left_proof_l1257_125756


namespace exchange_properties_l1257_125764

/-- Represents a box containing red and yellow balls -/
structure Box where
  red : ℕ
  yellow : ℕ

/-- Calculates the expected number of red balls after exchanging i balls -/
noncomputable def expected_red (box_a box_b : Box) (i : ℕ) : ℚ :=
  sorry

/-- Box A initially contains 3 red balls and 1 yellow ball -/
def initial_box_a : Box := ⟨3, 1⟩

/-- Box B initially contains 1 red ball and 3 yellow balls -/
def initial_box_b : Box := ⟨1, 3⟩

theorem exchange_properties :
  let E₁ := expected_red initial_box_a initial_box_b
  let E₂ := expected_red initial_box_b initial_box_a
  (E₁ 1 > E₂ 1) ∧
  (E₁ 2 = E₂ 2) ∧
  (E₁ 2 = 2) ∧
  (E₁ 1 + E₂ 1 = 4) ∧
  (E₁ 3 = 3/2) :=
by sorry

end exchange_properties_l1257_125764


namespace polynomial_division_quotient_l1257_125743

theorem polynomial_division_quotient :
  let dividend := fun (z : ℚ) => 4 * z^5 + 2 * z^4 - 7 * z^3 + 5 * z^2 - 3 * z + 8
  let divisor := fun (z : ℚ) => 3 * z + 1
  let quotient := fun (z : ℚ) => (4/3) * z^4 - (19/3) * z^3 + (34/3) * z^2 - (61/9) * z - 1
  ∀ z : ℚ, dividend z = divisor z * quotient z + (275/27) := by
  sorry

end polynomial_division_quotient_l1257_125743


namespace girls_combined_average_l1257_125742

/-- Represents a high school with exam scores -/
structure School where
  boys_avg : ℝ
  girls_avg : ℝ
  combined_avg : ℝ

/-- The average score calculation problem -/
theorem girls_combined_average 
  (cedar : School)
  (drake : School)
  (boys_combined_avg : ℝ)
  (h_cedar : cedar.combined_avg = 78)
  (h_drake : drake.combined_avg = 88)
  (h_cedar_boys : cedar.boys_avg = 75)
  (h_cedar_girls : cedar.girls_avg = 80)
  (h_drake_boys : drake.boys_avg = 85)
  (h_drake_girls : drake.girls_avg = 92)
  (h_boys_combined : boys_combined_avg = 83) :
  ∃ (girls_combined_avg : ℝ), girls_combined_avg = 88 := by
  sorry


end girls_combined_average_l1257_125742


namespace quadratic_equation_roots_l1257_125771

theorem quadratic_equation_roots (m : ℝ) : 
  (∃ x : ℝ, 3 * x^2 + m * x - 7 = 0 ∧ x = 1) → 
  (∃ y : ℝ, 3 * y^2 + m * y - 7 = 0 ∧ y = -7/3) :=
by sorry

end quadratic_equation_roots_l1257_125771


namespace partnership_profit_l1257_125725

/-- Calculates the profit of a business partnership given the investments and profit sharing rules -/
theorem partnership_profit (mary_investment mike_investment : ℚ) 
  (h1 : mary_investment = 700)
  (h2 : mike_investment = 300)
  (h3 : mary_investment + mike_investment > 0) :
  ∃ (P : ℚ), 
    (P / 6 + 7 * (2 * P / 3) / 10) - (P / 6 + 3 * (2 * P / 3) / 10) = 800 ∧ 
    P = 3000 := by
  sorry

end partnership_profit_l1257_125725


namespace sandwich_shop_period_length_l1257_125706

/-- Represents the Eat "N Go Mobile Sausage Sandwich Shop scenario -/
structure SandwichShop where
  jalapeno_strips_per_sandwich : ℕ
  minutes_per_sandwich : ℕ
  total_jalapeno_strips : ℕ

/-- Calculates the period length in minutes for a given SandwichShop scenario -/
def period_length (shop : SandwichShop) : ℕ :=
  (shop.total_jalapeno_strips / shop.jalapeno_strips_per_sandwich) * shop.minutes_per_sandwich

/-- Theorem stating that under the given conditions, the period length is 60 minutes -/
theorem sandwich_shop_period_length :
  ∀ (shop : SandwichShop),
    shop.jalapeno_strips_per_sandwich = 4 →
    shop.minutes_per_sandwich = 5 →
    shop.total_jalapeno_strips = 48 →
    period_length shop = 60 := by
  sorry


end sandwich_shop_period_length_l1257_125706


namespace cubic_inequality_false_l1257_125794

theorem cubic_inequality_false : 
  ¬(∃ x : ℝ, x^3 - x^2 + 1 ≤ 0) := by
sorry

end cubic_inequality_false_l1257_125794


namespace degrees_minutes_to_decimal_l1257_125757

-- Define the conversion factor from minutes to degrees
def minutes_to_degrees (m : ℚ) : ℚ := m / 60

-- Define the problem
theorem degrees_minutes_to_decimal (d : ℚ) (m : ℚ) :
  d + minutes_to_degrees m = 18.4 → d = 18 ∧ m = 24 :=
by sorry

end degrees_minutes_to_decimal_l1257_125757


namespace garden_breadth_calculation_l1257_125745

/-- Represents a rectangular garden --/
structure RectangularGarden where
  length : ℝ
  breadth : ℝ

/-- Calculates the perimeter of a rectangular garden --/
def perimeter (g : RectangularGarden) : ℝ :=
  2 * (g.length + g.breadth)

theorem garden_breadth_calculation :
  ∀ g : RectangularGarden,
    g.length = 205 →
    perimeter g = 600 →
    g.breadth = 95 := by
  sorry

end garden_breadth_calculation_l1257_125745


namespace round_robin_tournament_l1257_125703

theorem round_robin_tournament (n : ℕ) (h : n * (n - 1) / 2 = 28) : n = 8 := by
  sorry

end round_robin_tournament_l1257_125703


namespace fourth_year_afforestation_l1257_125736

/-- The area afforested in a given year, starting from an initial area and increasing by a fixed percentage annually. -/
def afforestedArea (initialArea : ℝ) (increaseRate : ℝ) (year : ℕ) : ℝ :=
  initialArea * (1 + increaseRate) ^ (year - 1)

/-- Theorem stating that given an initial afforestation of 10,000 acres and an annual increase of 20%, 
    the area afforested in the fourth year is 17,280 acres. -/
theorem fourth_year_afforestation :
  afforestedArea 10000 0.2 4 = 17280 := by
  sorry

end fourth_year_afforestation_l1257_125736


namespace parallel_line_through_point_l1257_125786

/-- A line in 2D space represented by the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line -/
def pointOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if two lines are parallel -/
def parallelLines (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

theorem parallel_line_through_point 
  (given_line : Line) 
  (point : Point) 
  (h_given_line : given_line = ⟨2, 1, -5⟩) 
  (h_point : point = ⟨1, 0⟩) : 
  ∃ (parallel_line : Line), 
    parallelLines given_line parallel_line ∧ 
    pointOnLine point parallel_line ∧ 
    parallel_line = ⟨2, 1, -2⟩ :=
sorry

end parallel_line_through_point_l1257_125786


namespace combined_tax_rate_approx_l1257_125738

/-- Represents the tax system for a group of individuals in a fictional universe. -/
structure TaxSystem where
  mork_tax_rate : ℝ
  mork_deduction : ℝ
  mindy_tax_rate : ℝ
  mindy_income_multiple : ℝ
  mindy_tax_break : ℝ
  bickley_income_multiple : ℝ
  bickley_tax_rate : ℝ
  bickley_deduction : ℝ
  exidor_income_fraction : ℝ
  exidor_tax_rate : ℝ
  exidor_tax_break : ℝ

/-- Calculates the combined tax rate for the group. -/
def combined_tax_rate (ts : TaxSystem) : ℝ :=
  sorry

/-- Theorem stating that the combined tax rate is approximately 23.57% -/
theorem combined_tax_rate_approx (ts : TaxSystem) 
  (h1 : ts.mork_tax_rate = 0.45)
  (h2 : ts.mork_deduction = 0.10)
  (h3 : ts.mindy_tax_rate = 0.20)
  (h4 : ts.mindy_income_multiple = 4)
  (h5 : ts.mindy_tax_break = 0.05)
  (h6 : ts.bickley_income_multiple = 2)
  (h7 : ts.bickley_tax_rate = 0.25)
  (h8 : ts.bickley_deduction = 0.07)
  (h9 : ts.exidor_income_fraction = 0.5)
  (h10 : ts.exidor_tax_rate = 0.30)
  (h11 : ts.exidor_tax_break = 0.08) :
  abs (combined_tax_rate ts - 0.2357) < 0.0001 := by
  sorry

end combined_tax_rate_approx_l1257_125738


namespace multiply_special_polynomials_l1257_125754

theorem multiply_special_polynomials (x : ℝ) : 
  (x^4 + 16*x^2 + 256) * (x^2 - 16) = x^6 - 4096 := by
sorry

end multiply_special_polynomials_l1257_125754


namespace optimal_sampling_methods_for_given_surveys_l1257_125739

/-- Represents different sampling methods -/
inductive SamplingMethod
| Stratified
| Random
| Systematic

/-- Represents a survey with its characteristics -/
structure Survey where
  population_size : ℕ
  sample_size : ℕ
  has_distinct_groups : Bool

/-- Determines the optimal sampling method for a given survey -/
def optimal_sampling_method (s : Survey) : SamplingMethod :=
  if s.has_distinct_groups then SamplingMethod.Stratified
  else if s.population_size ≤ 20 then SamplingMethod.Random
  else SamplingMethod.Systematic

/-- The first survey from the problem -/
def survey1 : Survey :=
  { population_size := 500
  , sample_size := 100
  , has_distinct_groups := true }

/-- The second survey from the problem -/
def survey2 : Survey :=
  { population_size := 12
  , sample_size := 3
  , has_distinct_groups := false }

theorem optimal_sampling_methods_for_given_surveys :
  optimal_sampling_method survey1 = SamplingMethod.Stratified ∧
  optimal_sampling_method survey2 = SamplingMethod.Random :=
sorry


end optimal_sampling_methods_for_given_surveys_l1257_125739


namespace symmetric_point_xoz_l1257_125761

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The xOz plane -/
def xOzPlane : Set Point3D := {p : Point3D | p.y = 0}

/-- Symmetry with respect to the xOz plane -/
def symmetricPointXOZ (p : Point3D) : Point3D :=
  { x := p.x, y := -p.y, z := p.z }

theorem symmetric_point_xoz (p : Point3D) :
  p.x = 2 ∧ p.y = 1 ∧ p.z = 3 →
  symmetricPointXOZ p = Point3D.mk 2 (-1) 3 := by
  sorry

end symmetric_point_xoz_l1257_125761


namespace abc_product_l1257_125752

theorem abc_product (a b c : ℤ) 
  (h1 : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0)
  (h2 : a + b + c = 30)
  (h3 : (1 : ℚ) / a + (1 : ℚ) / b + (1 : ℚ) / c + 630 / (a * b * c) = 1) :
  a * b * c = 483 := by
  sorry

end abc_product_l1257_125752


namespace new_person_weight_l1257_125740

/-- Given a group of 8 people where one person weighing 40 kg is replaced,
    if the average weight increases by 2.5 kg, then the new person weighs 60 kg. -/
theorem new_person_weight (initial_count : ℕ) (replaced_weight : ℝ) (avg_increase : ℝ) :
  initial_count = 8 →
  replaced_weight = 40 →
  avg_increase = 2.5 →
  (initial_count : ℝ) * avg_increase + replaced_weight = 60 :=
by sorry

end new_person_weight_l1257_125740


namespace quadratic_roots_when_k_negative_l1257_125768

theorem quadratic_roots_when_k_negative (k : ℝ) (h : k < 0) :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
  (x₁^2 + x₁ + k - 1 = 0) ∧ 
  (x₂^2 + x₂ + k - 1 = 0) := by
sorry

end quadratic_roots_when_k_negative_l1257_125768


namespace quadratic_equation_unique_solution_l1257_125777

theorem quadratic_equation_unique_solution (b : ℝ) (hb : b ≠ 0) :
  (∃! x, b * x^2 + 15 * x + 4 = 0) →
  (∃ x, b * x^2 + 15 * x + 4 = 0 ∧ x = -8/15) :=
by sorry

end quadratic_equation_unique_solution_l1257_125777


namespace vovochka_max_candies_l1257_125737

/-- Represents the candy distribution problem --/
structure CandyDistribution where
  total_candies : ℕ
  num_classmates : ℕ
  min_group_size : ℕ
  min_group_candies : ℕ

/-- Calculates the maximum number of candies Vovochka can keep --/
def max_candies_kept (cd : CandyDistribution) : ℕ :=
  cd.total_candies - (cd.num_classmates * (cd.min_group_candies / cd.min_group_size))

/-- Theorem stating the maximum number of candies Vovochka can keep --/
theorem vovochka_max_candies :
  let cd := CandyDistribution.mk 200 25 16 100
  max_candies_kept cd = 37 := by
  sorry

end vovochka_max_candies_l1257_125737


namespace largest_value_l1257_125723

theorem largest_value (a b : ℝ) (ha : 0 < a) (ha1 : a < 1) (hb : 0 < b) (hb1 : b < 1) (hab : a ≠ b) :
  (a + b) = max (a + b) (max (2 * Real.sqrt (a * b)) (max (a^2 + b^2) (2 * a * b))) :=
by sorry

end largest_value_l1257_125723


namespace tank_plastering_cost_l1257_125746

/-- Calculates the cost of plastering a rectangular tank's walls and bottom -/
def plasteringCost (length width depth rate : ℝ) : ℝ :=
  let wallArea := 2 * (length * depth + width * depth)
  let bottomArea := length * width
  let totalArea := wallArea + bottomArea
  totalArea * rate

/-- Theorem: The cost of plastering a 35m x 18m x 10m tank at ₹135 per sq m is ₹228,150 -/
theorem tank_plastering_cost :
  plasteringCost 35 18 10 135 = 228150 := by
  sorry

end tank_plastering_cost_l1257_125746


namespace man_speed_calculation_man_speed_approx_7_9916_l1257_125713

/-- Calculates the speed of a man given the parameters of a passing train -/
theorem man_speed_calculation (train_length : ℝ) (train_speed_kmph : ℝ) (passing_time : ℝ) : ℝ :=
  let train_speed_mps := train_speed_kmph * (1000 / 3600)
  let relative_speed := train_length / passing_time
  let man_speed_mps := train_speed_mps - relative_speed
  let man_speed_kmph := man_speed_mps * (3600 / 1000)
  man_speed_kmph

/-- The speed of the man is approximately 7.9916 kmph -/
theorem man_speed_approx_7_9916 : 
  ∃ ε > 0, |man_speed_calculation 350 68 20.99832013438925 - 7.9916| < ε :=
sorry

end man_speed_calculation_man_speed_approx_7_9916_l1257_125713


namespace birch_tree_arrangement_probability_l1257_125778

def num_maple_trees : ℕ := 5
def num_oak_trees : ℕ := 4
def num_birch_trees : ℕ := 6

def total_trees : ℕ := num_maple_trees + num_oak_trees + num_birch_trees

def num_non_birch_trees : ℕ := num_maple_trees + num_oak_trees

def num_slots_for_birch : ℕ := num_non_birch_trees + 1

theorem birch_tree_arrangement_probability :
  (Nat.choose num_slots_for_birch num_birch_trees : ℚ) / (Nat.choose total_trees num_birch_trees) = 2 / 45 := by
  sorry

end birch_tree_arrangement_probability_l1257_125778


namespace train_speed_and_length_l1257_125766

/-- Proves that given a bridge of length 1000m, if a train takes 60s to pass from the beginning
    to the end of the bridge and spends 40s on the bridge, then the speed of the train is 20 m/s
    and its length is 200m. -/
theorem train_speed_and_length
  (bridge_length : ℝ)
  (time_to_pass : ℝ)
  (time_on_bridge : ℝ)
  (h1 : bridge_length = 1000)
  (h2 : time_to_pass = 60)
  (h3 : time_on_bridge = 40)
  : ∃ (speed length : ℝ),
    speed = 20 ∧
    length = 200 ∧
    time_to_pass * speed = bridge_length + length ∧
    time_on_bridge * speed = bridge_length - length :=
by sorry

end train_speed_and_length_l1257_125766


namespace donny_gas_station_payment_l1257_125749

/-- Given the conditions of Donny's gas station visit, prove that he paid $350. -/
theorem donny_gas_station_payment (tank_capacity : ℝ) (initial_fuel : ℝ) (fuel_cost : ℝ) (change : ℝ)
  (h1 : tank_capacity = 150)
  (h2 : initial_fuel = 38)
  (h3 : fuel_cost = 3)
  (h4 : change = 14) :
  (tank_capacity - initial_fuel) * fuel_cost + change = 350 := by
  sorry

#check donny_gas_station_payment

end donny_gas_station_payment_l1257_125749


namespace ring_arrangement_count_l1257_125779

def ring_arrangements (total_rings : ℕ) (chosen_rings : ℕ) (fingers : ℕ) : ℕ :=
  (total_rings.choose chosen_rings) * (chosen_rings.factorial) * ((chosen_rings + fingers - 1).choose (fingers - 1))

theorem ring_arrangement_count :
  ring_arrangements 8 5 4 = 376320 :=
by sorry

end ring_arrangement_count_l1257_125779


namespace find_c_l1257_125759

theorem find_c (m c : ℕ) : 
  m < 10 → c < 10 → m = 2 * c → 
  (10 * m + c : ℚ) / 99 = (c + 4 : ℚ) / (m + 5) → 
  c = 3 :=
sorry

end find_c_l1257_125759


namespace triangle_side_length_l1257_125774

theorem triangle_side_length (A B : Real) (b : Real) (hA : A = 60 * π / 180) (hB : B = 45 * π / 180) (hb : b = Real.sqrt 2) :
  ∃ a : Real, a = Real.sqrt 3 ∧ a * Real.sin B = b * Real.sin A := by
sorry

end triangle_side_length_l1257_125774


namespace solve_eggs_problem_l1257_125716

def eggs_problem (breakfast_eggs lunch_eggs total_eggs : ℕ) : Prop :=
  let dinner_eggs := total_eggs - (breakfast_eggs + lunch_eggs)
  dinner_eggs = 1

theorem solve_eggs_problem :
  eggs_problem 2 3 6 :=
sorry

end solve_eggs_problem_l1257_125716


namespace quadratic_equation_roots_l1257_125731

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

theorem quadratic_equation_roots (p q : ℕ) (hp : is_prime p) (hq : is_prime q) :
  ∃ (x y : ℕ), x ≠ y ∧ x > 0 ∧ y > 0 ∧ x^2 - p*x + 2*q = 0 ∧ y^2 - p*y + 2*q = 0 →
  (∃ r : ℕ, (r = x ∨ r = y) ∧ is_prime r) ∧
  is_prime (p - q) ∧
  ¬(∀ x y : ℕ, x^2 - p*x + 2*q = 0 → y^2 - p*y + 2*q = 0 → x ≠ y → Even (x - y)) ∧
  ¬(is_prime (p^2 + 2*q)) :=
by sorry

end quadratic_equation_roots_l1257_125731


namespace quadratic_roots_imply_composite_l1257_125744

theorem quadratic_roots_imply_composite (a b : ℤ) :
  (∃ x₁ x₂ : ℕ+, x₁^2 + a * x₁ + b + 1 = 0 ∧ x₂^2 + a * x₂ + b + 1 = 0) →
  ∃ m n : ℕ, m > 1 ∧ n > 1 ∧ a^2 + b^2 = m * n :=
by sorry

end quadratic_roots_imply_composite_l1257_125744


namespace square_side_length_l1257_125788

theorem square_side_length 
  (total_wire : ℝ) 
  (triangle_perimeter : ℝ) 
  (h1 : total_wire = 78) 
  (h2 : triangle_perimeter = 46) : 
  (total_wire - triangle_perimeter) / 4 = 8 :=
by sorry

end square_side_length_l1257_125788


namespace compare_magnitudes_l1257_125709

theorem compare_magnitudes (a b : ℝ) (ha : a ≠ 1) :
  a^2 + b^2 > 2*(a - b - 1) := by
sorry

end compare_magnitudes_l1257_125709


namespace cost_of_birdhouses_l1257_125783

-- Define the constants
def planks_per_birdhouse : ℕ := 7
def nails_per_birdhouse : ℕ := 20
def cost_per_nail : ℚ := 0.05
def cost_per_plank : ℕ := 3
def num_birdhouses : ℕ := 4

-- Define the theorem
theorem cost_of_birdhouses :
  (num_birdhouses * (planks_per_birdhouse * cost_per_plank +
   nails_per_birdhouse * cost_per_nail) : ℚ) = 88 := by
  sorry

end cost_of_birdhouses_l1257_125783


namespace two_distinct_decorations_l1257_125795

/-- Represents the two types of decorations --/
inductive Decoration
| A
| B

/-- Represents a triangle decoration --/
structure TriangleDecoration :=
  (v1 v2 v3 : Decoration)

/-- Checks if a triangle decoration is valid according to the rules --/
def isValidDecoration (td : TriangleDecoration) : Prop :=
  (td.v1 = td.v2 ∧ td.v3 ≠ td.v1) ∨
  (td.v1 = td.v3 ∧ td.v2 ≠ td.v1) ∨
  (td.v2 = td.v3 ∧ td.v1 ≠ td.v2)

/-- Checks if two triangle decorations are equivalent under rotation or flipping --/
def areEquivalentDecorations (td1 td2 : TriangleDecoration) : Prop :=
  td1 = td2 ∨
  td1 = {v1 := td2.v2, v2 := td2.v3, v3 := td2.v1} ∨
  td1 = {v1 := td2.v3, v2 := td2.v1, v3 := td2.v2} ∨
  td1 = {v1 := td2.v1, v2 := td2.v3, v3 := td2.v2} ∨
  td1 = {v1 := td2.v3, v2 := td2.v2, v3 := td2.v1} ∨
  td1 = {v1 := td2.v2, v2 := td2.v1, v3 := td2.v3}

/-- The main theorem stating that there are exactly two distinct decorations --/
theorem two_distinct_decorations :
  ∃ (d1 d2 : TriangleDecoration),
    isValidDecoration d1 ∧
    isValidDecoration d2 ∧
    ¬(areEquivalentDecorations d1 d2) ∧
    (∀ d : TriangleDecoration, isValidDecoration d →
      (areEquivalentDecorations d d1 ∨ areEquivalentDecorations d d2)) :=
  sorry

end two_distinct_decorations_l1257_125795


namespace meaningful_expression_range_l1257_125734

-- Define the set of real numbers except 1
def RealExceptOne : Set ℝ := {x : ℝ | x ≠ 1}

-- Define the property of the expression being meaningful
def IsMeaningful (x : ℝ) : Prop := x - 1 ≠ 0

-- Theorem statement
theorem meaningful_expression_range :
  ∀ x : ℝ, IsMeaningful x ↔ x ∈ RealExceptOne :=
by sorry

end meaningful_expression_range_l1257_125734


namespace geometric_sequence_a3_l1257_125767

theorem geometric_sequence_a3 (a : ℕ → ℝ) :
  (∀ n : ℕ, a (n + 1) / a n = a 2 / a 1) →  -- geometric sequence condition
  a 1 = 2 →
  a 5 = 8 →
  a 3 = 4 ∨ a 3 = -4 := by
sorry

end geometric_sequence_a3_l1257_125767


namespace arithmetic_sequence_problem_l1257_125710

theorem arithmetic_sequence_problem (a₁ a₂ a₃ : ℚ) (x : ℚ) 
  (h1 : a₁ = 1/3)
  (h2 : a₂ = 2*x)
  (h3 : a₃ = x + 4)
  (h_arithmetic : a₃ - a₂ = a₂ - a₁) :
  x = 13/3 := by
  sorry

end arithmetic_sequence_problem_l1257_125710


namespace sin_330_degrees_l1257_125732

theorem sin_330_degrees : Real.sin (330 * π / 180) = -1/2 := by
  sorry

end sin_330_degrees_l1257_125732


namespace factorial_sum_div_l1257_125781

theorem factorial_sum_div (n : ℕ) : (8 * n.factorial + 9 * 8 * n.factorial + 10 * 9 * 8 * n.factorial) / n.factorial = 800 := by
  sorry

end factorial_sum_div_l1257_125781


namespace inequality_proof_l1257_125792

theorem inequality_proof (x y : ℝ) (h : x^8 + y^8 ≤ 1) :
  x^12 - y^12 + 2 * x^6 * y^6 ≤ π / 2 := by
  sorry

end inequality_proof_l1257_125792


namespace sufficient_condition_not_necessary_condition_sufficient_but_not_necessary_l1257_125789

/-- The equation represents a hyperbola -/
def is_hyperbola (k : ℝ) : Prop :=
  (k - 3) * (k + 3) > 0

/-- The condition k > 3 is sufficient for the equation to represent a hyperbola -/
theorem sufficient_condition (k : ℝ) (h : k > 3) : is_hyperbola k := by sorry

/-- The condition k > 3 is not necessary for the equation to represent a hyperbola -/
theorem not_necessary_condition : ∃ k, k ≤ 3 ∧ is_hyperbola k := by sorry

/-- k > 3 is a sufficient but not necessary condition for the equation to represent a hyperbola -/
theorem sufficient_but_not_necessary : 
  (∀ k, k > 3 → is_hyperbola k) ∧ (∃ k, k ≤ 3 ∧ is_hyperbola k) := by sorry

end sufficient_condition_not_necessary_condition_sufficient_but_not_necessary_l1257_125789


namespace rectangle_area_l1257_125722

theorem rectangle_area (length width diagonal : ℝ) (h1 : length / width = 5 / 2) (h2 : length^2 + width^2 = diagonal^2) (h3 : diagonal = 13) : 
  length * width = (10 / 29) * diagonal^2 := by
sorry

end rectangle_area_l1257_125722
