import Mathlib

namespace charlies_subtraction_l1306_130661

theorem charlies_subtraction (charlie_add : 41^2 = 40^2 + 81) : 39^2 = 40^2 - 79 := by
  sorry

end charlies_subtraction_l1306_130661


namespace four_digit_number_divisibility_l1306_130691

theorem four_digit_number_divisibility (a b c d : ℕ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) :
  let M := 1000 * a + 100 * b + 10 * c + d
  let N := 1000 * d + 100 * c + 10 * b + a
  (101 ∣ (M + N)) → a + d = b + c :=
by sorry

end four_digit_number_divisibility_l1306_130691


namespace all_methods_applicable_l1306_130627

structure Population where
  total : Nat
  farmers : Nat
  workers : Nat
  sample_size : Nat

inductive SamplingMethod
  | SimpleRandom
  | Systematic
  | Stratified

def is_applicable (pop : Population) (method : SamplingMethod) : Prop :=
  match method with
  | SamplingMethod.SimpleRandom => pop.workers > 0
  | SamplingMethod.Systematic => pop.farmers > 0
  | SamplingMethod.Stratified => pop.farmers ≠ pop.workers

theorem all_methods_applicable (pop : Population) 
  (h1 : pop.total = 2004)
  (h2 : pop.farmers = 1600)
  (h3 : pop.workers = 303)
  (h4 : pop.sample_size = 40) :
  (∀ m : SamplingMethod, is_applicable pop m) :=
by sorry

end all_methods_applicable_l1306_130627


namespace binomial_100_100_l1306_130663

theorem binomial_100_100 : Nat.choose 100 100 = 1 := by
  sorry

end binomial_100_100_l1306_130663


namespace cakes_donated_proof_l1306_130641

/-- The number of slices per cake -/
def slices_per_cake : ℕ := 8

/-- The price of each slice in dollars -/
def price_per_slice : ℚ := 1

/-- The donation from the first business owner per slice in dollars -/
def donation1_per_slice : ℚ := 1/2

/-- The donation from the second business owner per slice in dollars -/
def donation2_per_slice : ℚ := 1/4

/-- The total amount raised in dollars -/
def total_raised : ℚ := 140

/-- The number of cakes donated -/
def num_cakes : ℕ := 10

theorem cakes_donated_proof :
  (num_cakes : ℚ) * slices_per_cake * (price_per_slice + donation1_per_slice + donation2_per_slice) = total_raised :=
by sorry

end cakes_donated_proof_l1306_130641


namespace unique_solution_quadratic_inequality_l1306_130606

theorem unique_solution_quadratic_inequality (a : ℝ) : 
  (∃! x : ℝ, |x^2 + 2*a*x + 3*a| ≤ 2) ↔ (a = 1 ∨ a = 2) :=
sorry

end unique_solution_quadratic_inequality_l1306_130606


namespace prob_all_red_before_both_green_is_one_third_l1306_130665

/-- The number of red chips in the hat -/
def num_red : ℕ := 4

/-- The number of green chips in the hat -/
def num_green : ℕ := 2

/-- The total number of chips in the hat -/
def total_chips : ℕ := num_red + num_green

/-- The probability of drawing all red chips before both green chips -/
def prob_all_red_before_both_green : ℚ :=
  (total_chips - 1).choose num_green / total_chips.choose num_green

theorem prob_all_red_before_both_green_is_one_third :
  prob_all_red_before_both_green = 1 / 3 := by sorry

end prob_all_red_before_both_green_is_one_third_l1306_130665


namespace intersection_of_A_and_B_l1306_130658

def set_A : Set ℝ := { x | (x - 2) / (x + 5) < 0 }

def set_B : Set ℝ := { x | x^2 - 2*x - 3 ≥ 0 }

theorem intersection_of_A_and_B :
  ∀ x : ℝ, x ∈ set_A ∩ set_B ↔ -5 < x ∧ x ≤ -1 :=
sorry

end intersection_of_A_and_B_l1306_130658


namespace right_triangle_geometric_sequence_l1306_130671

theorem right_triangle_geometric_sequence (a b c q : ℝ) : 
  q > 1 →
  a > 0 →
  b > 0 →
  c > 0 →
  a * q = b →
  b * q = c →
  a^2 + b^2 = c^2 →
  q^2 = (Real.sqrt 5 + 1) / 2 := by
sorry

end right_triangle_geometric_sequence_l1306_130671


namespace evaluate_expression_l1306_130666

theorem evaluate_expression (x y : ℚ) (hx : x = 3) (hy : y = -3) :
  (4 + y * x * (y + x) - 4^2) / (y - 4 + y^2) = -6 := by sorry

end evaluate_expression_l1306_130666


namespace project_work_time_l1306_130660

/-- Calculates the time spent working on a project given the project duration and nap information -/
def time_spent_working (project_days : ℕ) (num_naps : ℕ) (nap_duration : ℕ) : ℕ :=
  let total_hours := project_days * 24
  let total_nap_hours := num_naps * nap_duration
  total_hours - total_nap_hours

/-- Proves that given a 4-day project and 6 seven-hour naps, the time spent working is 54 hours -/
theorem project_work_time : time_spent_working 4 6 7 = 54 := by
  sorry

end project_work_time_l1306_130660


namespace modulus_of_one_plus_three_i_l1306_130668

theorem modulus_of_one_plus_three_i : Complex.abs (1 + 3 * Complex.I) = Real.sqrt 10 := by
  sorry

end modulus_of_one_plus_three_i_l1306_130668


namespace imaginary_part_of_complex_fraction_l1306_130645

theorem imaginary_part_of_complex_fraction : Complex.im ((2 * Complex.I) / (1 - Complex.I) * Complex.I) = 1 := by
  sorry

end imaginary_part_of_complex_fraction_l1306_130645


namespace integral_shift_reciprocal_l1306_130629

/-- For a continuous function f: ℝ → ℝ, if the integral of f over the real line exists,
    then the integral of f(x - 1/x) over the real line equals the integral of f. -/
theorem integral_shift_reciprocal (f : ℝ → ℝ) (hf : Continuous f) 
  (L : ℝ) (hL : ∫ (x : ℝ), f x = L) :
  ∫ (x : ℝ), f (x - 1/x) = L := by
  sorry

end integral_shift_reciprocal_l1306_130629


namespace water_consumption_in_five_hours_l1306_130680

/-- The number of glasses of water consumed in a given time period. -/
def glasses_consumed (rate : ℚ) (time : ℚ) : ℚ :=
  time / rate

/-- Theorem stating that drinking a glass of water every 20 minutes for 5 hours results in 15 glasses. -/
theorem water_consumption_in_five_hours :
  glasses_consumed (20 : ℚ) (5 * 60 : ℚ) = 15 := by
  sorry

end water_consumption_in_five_hours_l1306_130680


namespace sum_of_fractions_l1306_130620

theorem sum_of_fractions : (2 : ℚ) / 5 + (3 : ℚ) / 10 = (7 : ℚ) / 10 := by
  sorry

end sum_of_fractions_l1306_130620


namespace minimum_excellent_all_exams_l1306_130699

theorem minimum_excellent_all_exams (total_students : ℕ) 
  (excellent_first : ℕ) (excellent_second : ℕ) (excellent_third : ℕ) 
  (h_total : total_students = 200)
  (h_first : excellent_first = (80 : ℝ) / 100 * total_students)
  (h_second : excellent_second = (70 : ℝ) / 100 * total_students)
  (h_third : excellent_third = (59 : ℝ) / 100 * total_students) :
  ∃ (excellent_all : ℕ), 
    excellent_all ≥ 18 ∧ 
    (∀ (n : ℕ), n < excellent_all → 
      ∃ (m1 m2 m3 m12 m13 m23 : ℕ),
        m1 + m2 + m3 + m12 + m13 + m23 + n > total_students ∨
        m1 + m12 + m13 + n > excellent_first ∨
        m2 + m12 + m23 + n > excellent_second ∨
        m3 + m13 + m23 + n > excellent_third) :=
sorry

end minimum_excellent_all_exams_l1306_130699


namespace initial_crayons_count_l1306_130640

/-- The number of crayons initially in the drawer -/
def initial_crayons : ℕ := sorry

/-- The number of crayons Benny added to the drawer -/
def added_crayons : ℕ := 3

/-- The total number of crayons in the drawer after Benny's addition -/
def total_crayons : ℕ := 12

/-- Theorem stating that the initial number of crayons is 9 -/
theorem initial_crayons_count : initial_crayons = 9 := by sorry

end initial_crayons_count_l1306_130640


namespace polynomial_identity_sum_of_squares_l1306_130636

theorem polynomial_identity_sum_of_squares : 
  ∀ (a b c d e f : ℤ), 
  (∀ x : ℝ, 216 * x^3 + 64 = (a * x^2 + b * x + c) * (d * x^2 + e * x + f)) →
  a^2 + b^2 + c^2 + d^2 + e^2 + f^2 = 2180 := by
sorry

end polynomial_identity_sum_of_squares_l1306_130636


namespace weight_replacement_l1306_130696

theorem weight_replacement (n : ℕ) (avg_increase weight_new : ℝ) :
  n = 9 ∧ 
  avg_increase = 5.5 ∧
  weight_new = 135.5 →
  (n * avg_increase + weight_new - n * avg_increase) = 86 :=
by sorry

end weight_replacement_l1306_130696


namespace combined_salaries_combined_salaries_proof_l1306_130653

/-- The combined salaries of A, B, C, and E, given D's salary and the average salary of all five. -/
theorem combined_salaries (salary_D : ℕ) (average_salary : ℕ) : ℕ :=
  let total_salary := average_salary * 5
  total_salary - salary_D

/-- Proof that the combined salaries of A, B, C, and E is 38000, given the conditions. -/
theorem combined_salaries_proof (salary_D : ℕ) (average_salary : ℕ)
    (h1 : salary_D = 7000)
    (h2 : average_salary = 9000) :
    combined_salaries salary_D average_salary = 38000 := by
  sorry

end combined_salaries_combined_salaries_proof_l1306_130653


namespace exponent_simplification_l1306_130693

theorem exponent_simplification (x : ℝ) : (x^5 * x^2) * x^4 = x^11 := by
  sorry

end exponent_simplification_l1306_130693


namespace largest_attendance_difference_largest_attendance_difference_holds_l1306_130625

/-- The largest possible difference between attendances in Chicago and Detroit --/
theorem largest_attendance_difference : ℝ → Prop :=
  fun max_diff =>
  ∀ (chicago_actual detroit_actual : ℝ),
  (chicago_actual ≥ 80000 * 0.95 ∧ chicago_actual ≤ 80000 * 1.05) →
  (detroit_actual ≥ 95000 / 1.15 ∧ detroit_actual ≤ 95000 / 0.85) →
  max_diff = 36000 ∧
  ∀ (diff : ℝ),
  diff ≤ detroit_actual - chicago_actual →
  ⌊diff / 1000⌋ * 1000 ≤ max_diff

/-- The theorem holds --/
theorem largest_attendance_difference_holds :
  largest_attendance_difference 36000 := by sorry

end largest_attendance_difference_largest_attendance_difference_holds_l1306_130625


namespace radical_conjugate_sum_product_l1306_130649

theorem radical_conjugate_sum_product (c d : ℝ) 
  (h1 : (c + Real.sqrt d) + (c - Real.sqrt d) = -6)
  (h2 : (c + Real.sqrt d) * (c - Real.sqrt d) = 1) :
  4 * c + d = -4 := by
  sorry

end radical_conjugate_sum_product_l1306_130649


namespace inequality_solution_l1306_130630

-- Define the inequality function
def inequality (x : ℝ) : Prop :=
  9.216 * (Real.log x / Real.log 5) + (Real.log x - Real.log 3) / (Real.log x)
  < ((Real.log x / Real.log 5) * (2 - Real.log x / Real.log 3)) / (Real.log x / Real.log 3)

-- State the theorem
theorem inequality_solution :
  ∀ x : ℝ, 
  x > 0 → 
  inequality x ↔ (0 < x ∧ x < 1 / Real.sqrt 5) ∨ (1 < x ∧ x < 3) :=
sorry

end inequality_solution_l1306_130630


namespace increasing_function_integral_inequality_l1306_130603

theorem increasing_function_integral_inequality
  (f : ℝ → ℝ) (hf : Continuous f) :
  (∀ (a b c : ℝ), a < b → b < c →
    (c - b) * ∫ x in a..b, f x ≤ (b - a) * ∫ x in b..c, f x) ↔
  Monotone f :=
by sorry

end increasing_function_integral_inequality_l1306_130603


namespace negative_two_times_inequality_l1306_130667

theorem negative_two_times_inequality (m n : ℝ) (h : m > n) : -2 * m < -2 * n := by
  sorry

end negative_two_times_inequality_l1306_130667


namespace diagonal_sequence_theorem_l1306_130610

/-- A convex polygon with 1994 sides and 997 diagonals -/
structure ConvexPolygon :=
  (sides : ℕ)
  (diagonals : ℕ)
  (is_convex : Bool)
  (sides_eq : sides = 1994)
  (diagonals_eq : diagonals = 997)
  (convex : is_convex = true)

/-- The length of a diagonal is the number of sides in the smaller part of the perimeter it divides -/
def diagonal_length (p : ConvexPolygon) (d : ℕ) : ℕ := sorry

/-- Each vertex has exactly one diagonal emanating from it -/
def one_diagonal_per_vertex (p : ConvexPolygon) : Prop := sorry

/-- The sequence of diagonal lengths in decreasing order -/
def diagonal_sequence (p : ConvexPolygon) : List ℕ := sorry

theorem diagonal_sequence_theorem (p : ConvexPolygon) 
  (h : one_diagonal_per_vertex p) :
  (∃ (seq : List ℕ), diagonal_sequence p = seq ∧ 
    seq.length = 997 ∧
    seq.count 3 = 991 ∧ 
    seq.count 2 = 6) ∧
  ¬(∃ (seq : List ℕ), diagonal_sequence p = seq ∧ 
    seq.length = 997 ∧
    seq.count 8 = 4 ∧ 
    seq.count 6 = 985 ∧ 
    seq.count 3 = 8) :=
sorry

end diagonal_sequence_theorem_l1306_130610


namespace remainder_theorem_l1306_130662

theorem remainder_theorem (n m : ℤ) 
  (hn : n % 37 = 15) 
  (hm : m % 47 = 21) : 
  (3 * n + 2 * m) % 59 = 28 := by
  sorry

end remainder_theorem_l1306_130662


namespace max_value_expression_l1306_130614

theorem max_value_expression (a b c d : Real) 
  (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) (hc : 0 ≤ c ∧ c ≤ 1) (hd : 0 ≤ d ∧ d ≤ 1) : 
  (∃ x y z w, 0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 1 ∧ 0 ≤ z ∧ z ≤ 1 ∧ 0 ≤ w ∧ w ≤ 1 ∧ 
    x + y + z + w - x*y - y*z - z*w - w*x = 2) ∧ 
  (∀ a b c d, 0 ≤ a ∧ a ≤ 1 → 0 ≤ b ∧ b ≤ 1 → 0 ≤ c ∧ c ≤ 1 → 0 ≤ d ∧ d ≤ 1 → 
    a + b + c + d - a*b - b*c - c*d - d*a ≤ 2) :=
by sorry

end max_value_expression_l1306_130614


namespace loan_repayment_proof_l1306_130633

/-- Calculates the total amount to be repaid for a loan with simple interest -/
def total_repayment (initial_loan : ℝ) (additional_loan : ℝ) (initial_period : ℝ) (total_period : ℝ) (rate : ℝ) : ℝ :=
  let initial_with_interest := initial_loan * (1 + rate * initial_period)
  let total_loan := initial_with_interest + additional_loan
  total_loan * (1 + rate * (total_period - initial_period))

/-- Proves that the total repayment for the given loan scenario is 27376 Rs -/
theorem loan_repayment_proof :
  total_repayment 10000 12000 2 5 0.06 = 27376 := by
  sorry

#eval total_repayment 10000 12000 2 5 0.06

end loan_repayment_proof_l1306_130633


namespace sibling_of_five_sevenths_unique_parent_one_over_2008_descendant_of_one_l1306_130692

-- Define the child relation
def is_child (x y : ℝ) : Prop :=
  (y = x + 1) ∨ (y = x / (x + 1))

-- Define the sibling relation
def is_sibling (x y : ℝ) : Prop :=
  ∃ z, is_child z x ∧ y = z + 1

-- Define the descendant relation
def is_descendant (x y : ℝ) : Prop :=
  ∃ n : ℕ, ∃ f : ℕ → ℝ,
    f 0 = x ∧ f n = y ∧
    ∀ i < n, is_child (f i) (f (i + 1))

theorem sibling_of_five_sevenths :
  is_sibling (5/7) (7/2) :=
sorry

theorem unique_parent (x y z : ℝ) (hx : x > 0) (hz : z > 0) :
  is_child x y → is_child z y → x = z :=
sorry

theorem one_over_2008_descendant_of_one :
  is_descendant 1 (1/2008) :=
sorry

end sibling_of_five_sevenths_unique_parent_one_over_2008_descendant_of_one_l1306_130692


namespace football_throw_distance_l1306_130632

theorem football_throw_distance (parker_distance : ℝ) :
  let grant_distance := parker_distance * 1.25
  let kyle_distance := grant_distance * 2
  kyle_distance - parker_distance = 24 →
  parker_distance = 16 := by
sorry

end football_throw_distance_l1306_130632


namespace sum_a_b_equals_14_l1306_130698

theorem sum_a_b_equals_14 (a b c d : ℝ) 
  (h1 : b + c = 9) 
  (h2 : c + d = 3) 
  (h3 : a + d = 8) : 
  a + b = 14 := by
  sorry

end sum_a_b_equals_14_l1306_130698


namespace number_value_l1306_130647

theorem number_value (x : ℚ) (n : ℚ) : 
  x = 12 → n + 7 / x = 6 - 5 / x → n = 5 := by sorry

end number_value_l1306_130647


namespace twelve_tone_equal_temperament_l1306_130676

theorem twelve_tone_equal_temperament (a : ℕ → ℝ) :
  (∀ n, 1 ≤ n → n < 13 → a (n + 1) / a n = a 2 / a 1) →  -- Equal ratio between adjacent terms
  a 13 = 2 * a 1 →                                      -- Last term is twice the first term
  a 8 / a 2 = Real.sqrt 2 := by
sorry

end twelve_tone_equal_temperament_l1306_130676


namespace polygon_interior_angles_l1306_130637

theorem polygon_interior_angles (n : ℕ) (h1 : n > 0) : 
  (n - 2) * 180 = n * 177 → n = 120 := by sorry

end polygon_interior_angles_l1306_130637


namespace arithmetic_mean_greater_than_geometric_mean_l1306_130688

theorem arithmetic_mean_greater_than_geometric_mean
  (a b : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hab : a ≠ b) (ha_pos : a ≠ 0) (hb_pos : b ≠ 0) :
  (a + b) / 2 > Real.sqrt (a * b) := by
  sorry

end arithmetic_mean_greater_than_geometric_mean_l1306_130688


namespace decreasing_f_sufficient_not_necessary_for_increasing_g_l1306_130672

open Real

theorem decreasing_f_sufficient_not_necessary_for_increasing_g
  (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (∀ x y : ℝ, x < y → a^x > a^y) →
  (∀ x y : ℝ, x < y → (2 - a) * x^3 < (2 - a) * y^3) ∧
  ¬(∀ x y : ℝ, x < y → (2 - a) * x^3 < (2 - a) * y^3 →
    a^x > a^y) :=
by sorry

end decreasing_f_sufficient_not_necessary_for_increasing_g_l1306_130672


namespace sum_of_divisors_360_l1306_130631

/-- The sum of the positive whole number divisors of 360 is 1170. -/
theorem sum_of_divisors_360 : (Finset.filter (· ∣ 360) (Finset.range 361)).sum id = 1170 := by
  sorry

end sum_of_divisors_360_l1306_130631


namespace circle_number_placement_l1306_130634

-- Define the type for circle positions
inductive CirclePosition
  | one | two | three | four | five | six | seven | eight

-- Define the neighborhood relation
def isNeighbor : CirclePosition → CirclePosition → Prop
  | CirclePosition.one, CirclePosition.two => True
  | CirclePosition.one, CirclePosition.four => True
  | CirclePosition.two, CirclePosition.three => True
  | CirclePosition.two, CirclePosition.four => True
  | CirclePosition.two, CirclePosition.six => True
  | CirclePosition.three, CirclePosition.four => True
  | CirclePosition.three, CirclePosition.seven => True
  | CirclePosition.four, CirclePosition.five => True
  | CirclePosition.five, CirclePosition.six => True
  | CirclePosition.six, CirclePosition.seven => True
  | CirclePosition.seven, CirclePosition.eight => True
  | _, _ => False

-- Define the valid numbers
def validNumbers : List Nat := [2, 3, 4, 5, 6, 7, 8, 9]

-- Define a function to check if a number is a divisor of another
def isDivisor (a b : Nat) : Prop := b % a = 0 ∧ a ≠ 1 ∧ a ≠ b

-- Define the main theorem
theorem circle_number_placement :
  ∃ (f : CirclePosition → Nat),
    (∀ p, f p ∈ validNumbers) ∧
    (∀ p₁ p₂, p₁ ≠ p₂ → f p₁ ≠ f p₂) ∧
    (∀ p₁ p₂, isNeighbor p₁ p₂ → ¬isDivisor (f p₁) (f p₂)) := by
  sorry

end circle_number_placement_l1306_130634


namespace polynomial_simplification_l1306_130694

theorem polynomial_simplification (x : ℝ) :
  (x^5 + 3*x^4 + x^2 + 13) + (x^5 - 4*x^4 + x^3 - x^2 + 15) = 2*x^5 - x^4 + x^3 + 28 := by
  sorry

end polynomial_simplification_l1306_130694


namespace second_person_speed_l1306_130697

/-- Given two people traveling between points A and B, prove the speed of the second person. -/
theorem second_person_speed 
  (distance : ℝ) 
  (speed_first : ℝ) 
  (travel_time : ℝ) 
  (h1 : distance = 600) 
  (h2 : speed_first = 70) 
  (h3 : travel_time = 4) : 
  ∃ speed_second : ℝ, speed_second = 80 ∧ 
  speed_first * travel_time + speed_second * travel_time = distance :=
by
  sorry

#check second_person_speed

end second_person_speed_l1306_130697


namespace odd_product_minus_one_divisible_by_four_l1306_130644

theorem odd_product_minus_one_divisible_by_four (a b c : ℤ) 
  (ha : Odd a) (hb : Odd b) (hc : Odd c) : 
  (4 ∣ a * b - 1) ∨ (4 ∣ b * c - 1) ∨ (4 ∣ c * a - 1) := by
  sorry

end odd_product_minus_one_divisible_by_four_l1306_130644


namespace work_completion_time_l1306_130612

theorem work_completion_time (ajay_time vijay_time combined_time : ℝ) : 
  ajay_time = 8 →
  combined_time = 6 →
  1 / ajay_time + 1 / vijay_time = 1 / combined_time →
  vijay_time = 24 := by
sorry

end work_completion_time_l1306_130612


namespace R_final_coordinates_l1306_130646

/-- Reflect a point over the x-axis -/
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

/-- Reflect a point over the y-axis -/
def reflect_y (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, p.2)

/-- Reflect a point over the line y=x -/
def reflect_y_eq_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.2, p.1)

/-- The initial point R -/
def R : ℝ × ℝ := (0, -5)

/-- The sequence of reflections applied to R -/
def R_transformed : ℝ × ℝ :=
  reflect_y_eq_x (reflect_y (reflect_x R))

theorem R_final_coordinates :
  R_transformed = (5, 0) := by
  sorry

end R_final_coordinates_l1306_130646


namespace western_olympiad_2004_l1306_130621

theorem western_olympiad_2004 (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  1 < (a / Real.sqrt (a^2 + b^2)) + (b / Real.sqrt (b^2 + c^2)) + (c / Real.sqrt (c^2 + a^2)) ∧
  (a / Real.sqrt (a^2 + b^2)) + (b / Real.sqrt (b^2 + c^2)) + (c / Real.sqrt (c^2 + a^2)) ≤ 3 * Real.sqrt 2 / 2 := by
  sorry

end western_olympiad_2004_l1306_130621


namespace remainder_theorem_l1306_130669

theorem remainder_theorem (n : ℤ) (k : ℤ) (h : n = 30 * k - 1) :
  (n^2 + 2*n + n^3 + 3) % 30 = 1 := by
  sorry

end remainder_theorem_l1306_130669


namespace sqrt_ten_plus_three_squared_times_sqrt_ten_minus_three_l1306_130651

theorem sqrt_ten_plus_three_squared_times_sqrt_ten_minus_three :
  (Real.sqrt 10 + 3)^2 * (Real.sqrt 10 - 3) = Real.sqrt 10 + 3 := by
  sorry

end sqrt_ten_plus_three_squared_times_sqrt_ten_minus_three_l1306_130651


namespace k_greater_than_one_over_e_l1306_130616

/-- Given that k(e^(kx)+1)-(1+1/x)ln(x) > 0 for all x > 0, prove that k > 1/e -/
theorem k_greater_than_one_over_e (k : ℝ) 
  (h : ∀ x : ℝ, x > 0 → k * (Real.exp (k * x) + 1) - (1 + 1 / x) * Real.log x > 0) : 
  k > 1 / Real.exp 1 := by
  sorry

end k_greater_than_one_over_e_l1306_130616


namespace rachel_score_l1306_130657

/-- Rachel's video game scoring system -/
structure GameScore where
  points_per_treasure : ℕ
  treasures_level1 : ℕ
  treasures_level2 : ℕ

/-- Calculate the total score for Rachel's game -/
def total_score (game : GameScore) : ℕ :=
  game.points_per_treasure * (game.treasures_level1 + game.treasures_level2)

/-- Theorem: Rachel's total score is 63 points -/
theorem rachel_score :
  ∀ (game : GameScore),
  game.points_per_treasure = 9 →
  game.treasures_level1 = 5 →
  game.treasures_level2 = 2 →
  total_score game = 63 :=
by
  sorry

end rachel_score_l1306_130657


namespace range_of_a_l1306_130654

/-- Proposition p -/
def p (x : ℝ) : Prop := (2*x - 1) / (x - 1) < 0

/-- Proposition q -/
def q (x a : ℝ) : Prop := x^2 - (2*a + 1)*x + a*(a + 1) ≤ 0

/-- The main theorem -/
theorem range_of_a (a : ℝ) : 
  (∀ x, p x ↔ q x a) → 0 ≤ a ∧ a ≤ 1/2 :=
by sorry

end range_of_a_l1306_130654


namespace equation_solution_range_l1306_130611

theorem equation_solution_range (a : ℝ) : 
  ∃ x : ℝ, 
    ((x - 3) / (x - 2) + 1 = 3 / (2 - x)) ∧ 
    ((2 - a) * x - 3 > 0) → 
    a < -1 := by
  sorry

end equation_solution_range_l1306_130611


namespace flagpole_height_l1306_130605

/-- The height of a flagpole given shadow lengths -/
theorem flagpole_height (flagpole_shadow : ℝ) (tree_height : ℝ) (tree_shadow : ℝ)
  (h1 : flagpole_shadow = 90)
  (h2 : tree_height = 15)
  (h3 : tree_shadow = 30)
  : ∃ (flagpole_height : ℝ), flagpole_height = 45 :=
by sorry

end flagpole_height_l1306_130605


namespace swim_meet_capacity_theorem_l1306_130619

/-- Represents the swimming club's transportation scenario -/
structure SwimMeetTransport where
  num_cars : ℕ
  num_vans : ℕ
  people_per_car : ℕ
  people_per_van : ℕ
  max_car_capacity : ℕ
  max_van_capacity : ℕ

/-- Calculates the number of additional people that could have ridden with the swim team -/
def additional_capacity (t : SwimMeetTransport) : ℕ :=
  (t.num_cars * t.max_car_capacity + t.num_vans * t.max_van_capacity) -
  (t.num_cars * t.people_per_car + t.num_vans * t.people_per_van)

/-- Theorem stating that 17 more people could have ridden with the swim team -/
theorem swim_meet_capacity_theorem (t : SwimMeetTransport)
  (h1 : t.num_cars = 2)
  (h2 : t.num_vans = 3)
  (h3 : t.people_per_car = 5)
  (h4 : t.people_per_van = 3)
  (h5 : t.max_car_capacity = 6)
  (h6 : t.max_van_capacity = 8) :
  additional_capacity t = 17 := by
  sorry

#eval additional_capacity {
  num_cars := 2,
  num_vans := 3,
  people_per_car := 5,
  people_per_van := 3,
  max_car_capacity := 6,
  max_van_capacity := 8
}

end swim_meet_capacity_theorem_l1306_130619


namespace fruit_price_adjustment_l1306_130622

/-- Represents the problem of adjusting fruit quantities to achieve a desired average price --/
theorem fruit_price_adjustment
  (apple_price : ℚ)
  (orange_price : ℚ)
  (total_fruits : ℕ)
  (initial_avg_price : ℚ)
  (desired_avg_price : ℚ)
  (h1 : apple_price = 40/100)
  (h2 : orange_price = 60/100)
  (h3 : total_fruits = 10)
  (h4 : initial_avg_price = 52/100)
  (h5 : desired_avg_price = 44/100)
  : ∃ (oranges_to_remove : ℕ),
    oranges_to_remove = 5 ∧
    ∃ (apples : ℕ) (oranges : ℕ),
      apples + oranges = total_fruits ∧
      (apple_price * apples + orange_price * oranges) / total_fruits = initial_avg_price ∧
      (apple_price * apples + orange_price * (oranges - oranges_to_remove)) / (total_fruits - oranges_to_remove) = desired_avg_price :=
by
  sorry

end fruit_price_adjustment_l1306_130622


namespace number_of_trailing_zeros_l1306_130681

theorem number_of_trailing_zeros : ∃ n : ℕ, (10^100 * 100^10 : ℕ) = n * 10^120 ∧ n % 10 ≠ 0 := by
  sorry

end number_of_trailing_zeros_l1306_130681


namespace three_digit_square_proof_l1306_130626

theorem three_digit_square_proof : 
  ∃! (S : Finset Nat), 
    (∀ n ∈ S, 100 ≤ n ∧ n < 1000) ∧ 
    (∀ n ∈ S, ∃ k : Nat, 1000 * n = n^2 + k ∧ k < 1000) ∧
    S.card = 2 := by
  sorry

end three_digit_square_proof_l1306_130626


namespace hyperbola_foci_l1306_130683

-- Define the hyperbola equation
def hyperbola_equation (x y : ℝ) : Prop :=
  x^2 / 9 - y^2 / 16 = 1

-- Define the foci of the hyperbola
def foci : Set (ℝ × ℝ) :=
  {(5, 0), (-5, 0)}

-- Theorem statement
theorem hyperbola_foci :
  ∀ (x y : ℝ), hyperbola_equation x y → (x, y) ∈ foci :=
by sorry

end hyperbola_foci_l1306_130683


namespace probability_b_speaks_truth_l1306_130675

theorem probability_b_speaks_truth (prob_a_truth : ℝ) (prob_both_truth : ℝ) :
  prob_a_truth = 0.75 →
  prob_both_truth = 0.45 →
  ∃ prob_b_truth : ℝ, prob_b_truth = 0.6 ∧ prob_a_truth * prob_b_truth = prob_both_truth :=
by sorry

end probability_b_speaks_truth_l1306_130675


namespace bert_grocery_fraction_l1306_130679

def bert_spending (initial_amount : ℚ) (hardware_fraction : ℚ) (dry_cleaner_amount : ℚ) (final_amount : ℚ) : Prop :=
  let hardware_spent := initial_amount * hardware_fraction
  let after_hardware := initial_amount - hardware_spent
  let after_dry_cleaner := after_hardware - dry_cleaner_amount
  let grocery_spent := after_dry_cleaner - final_amount
  grocery_spent / after_dry_cleaner = 1/2

theorem bert_grocery_fraction :
  bert_spending 44 (1/4) 9 12 :=
by
  sorry

end bert_grocery_fraction_l1306_130679


namespace sequence_expression_evaluation_l1306_130609

theorem sequence_expression_evaluation :
  ∀ (x : ℝ),
  (∀ (n : ℕ), n > 0 → n = 2^(n-1) * x) →
  x = 1 →
  2*x * 6*x + 5*x / (4*x) - 56*x = 69/8 := by
  sorry

end sequence_expression_evaluation_l1306_130609


namespace sin_alpha_value_l1306_130674

theorem sin_alpha_value (α β : Real) (h_acute : 0 < α ∧ α < π / 2)
  (h1 : 2 * Real.tan (π - α) - 3 * Real.cos (π / 2 + β) + 5 = 0)
  (h2 : Real.tan (π + α) + 6 * Real.sin (π + β) = 1) :
  Real.sin α = 3 * Real.sqrt 10 / 10 := by
  sorry

end sin_alpha_value_l1306_130674


namespace problem_1_l1306_130687

theorem problem_1 : |-2| - 8 / (-2) / (-1/2) = -6 := by sorry

end problem_1_l1306_130687


namespace game_night_sandwiches_l1306_130600

theorem game_night_sandwiches (num_friends : ℕ) (sandwiches_per_friend : ℕ) 
  (h1 : num_friends = 7) (h2 : sandwiches_per_friend = 5) : 
  num_friends * sandwiches_per_friend = 35 := by
  sorry

end game_night_sandwiches_l1306_130600


namespace total_entertainment_cost_l1306_130684

def computer_game_cost : ℕ := 66
def movie_ticket_cost : ℕ := 12
def number_of_tickets : ℕ := 3

theorem total_entertainment_cost : 
  computer_game_cost + number_of_tickets * movie_ticket_cost = 102 := by
  sorry

end total_entertainment_cost_l1306_130684


namespace cubic_sum_theorem_l1306_130648

theorem cubic_sum_theorem (x y : ℝ) (h1 : y^2 - 3 = (x - 3)^3) (h2 : x^2 - 3 = (y - 3)^2) (h3 : x ≠ y) :
  x^3 + y^3 = 27 + 3 * Real.sqrt 3 ∨ x^3 + y^3 = 27 - 3 * Real.sqrt 3 := by
  sorry

end cubic_sum_theorem_l1306_130648


namespace neither_necessary_nor_sufficient_l1306_130615

theorem neither_necessary_nor_sufficient (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  ¬(∀ a b, a > b → 1/a < 1/b) ∧ ¬(∀ a b, 1/a < 1/b → a > b) :=
sorry

end neither_necessary_nor_sufficient_l1306_130615


namespace fib_150_mod_9_l1306_130689

def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

theorem fib_150_mod_9 : fib 150 % 9 = 8 := by
  sorry

end fib_150_mod_9_l1306_130689


namespace cos_beta_minus_alpha_l1306_130650

theorem cos_beta_minus_alpha (α β : ℝ) (h1 : 0 < α) (h2 : α < β) (h3 : β < 2 * π)
  (h4 : 5 * Real.sin (α - π / 6) = 1) (h5 : 5 * Real.sin (β - π / 6) = 1) :
  Real.cos (β - α) = -23 / 25 := by
sorry

end cos_beta_minus_alpha_l1306_130650


namespace round_table_numbers_l1306_130638

theorem round_table_numbers (n : Fin 10 → ℝ) 
  (h1 : (n 9 + n 1) / 2 = 1)
  (h2 : (n 0 + n 2) / 2 = 2)
  (h3 : (n 1 + n 3) / 2 = 3)
  (h4 : (n 2 + n 4) / 2 = 4)
  (h5 : (n 3 + n 5) / 2 = 5)
  (h6 : (n 4 + n 6) / 2 = 6)
  (h7 : (n 5 + n 7) / 2 = 7)
  (h8 : (n 6 + n 8) / 2 = 8)
  (h9 : (n 7 + n 9) / 2 = 9)
  (h10 : (n 8 + n 0) / 2 = 10) :
  n 5 = 7 := by
sorry

end round_table_numbers_l1306_130638


namespace bridge_length_calculation_l1306_130607

/-- Given a train of length 100 meters traveling at 45 km/hr that crosses a bridge in 30 seconds, 
    the length of the bridge is 275 meters. -/
theorem bridge_length_calculation (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 100 →
  train_speed_kmh = 45 →
  crossing_time = 30 →
  (train_speed_kmh * 1000 / 3600 * crossing_time) - train_length = 275 := by
  sorry

end bridge_length_calculation_l1306_130607


namespace parallel_vectors_solution_l1306_130613

open Real

theorem parallel_vectors_solution (x : ℝ) :
  let a : ℝ × ℝ := (sin x, 3/4)
  let b : ℝ × ℝ := (1/3, (1/2) * cos x)
  let c : ℝ × ℝ := (1/6, cos x)
  0 < x ∧ x < (5 * π) / 12 ∧ 
  (∃ (k : ℝ), k * a.1 = (b.1 + c.1) ∧ k * a.2 = (b.2 + c.2)) →
  x = π / 12 :=
by sorry

end parallel_vectors_solution_l1306_130613


namespace phones_left_theorem_l1306_130685

/-- Calculates the number of phones left in the factory after doubling production and selling a quarter --/
def phones_left_in_factory (last_year_production : ℕ) : ℕ :=
  let this_year_production := 2 * last_year_production
  let sold_phones := this_year_production / 4
  this_year_production - sold_phones

/-- Theorem stating that given last year's production of 5000 phones, 
    if this year's production is doubled and a quarter of it is sold, 
    then the number of phones left in the factory is 7500 --/
theorem phones_left_theorem : phones_left_in_factory 5000 = 7500 := by
  sorry

end phones_left_theorem_l1306_130685


namespace min_folders_required_l1306_130628

/-- Represents the types of files --/
inductive FileType
  | PDF
  | Word
  | PPT

/-- Represents the initial file counts --/
structure InitialFiles where
  pdf : Nat
  word : Nat
  ppt : Nat

/-- Represents the deleted file counts --/
structure DeletedFiles where
  pdf : Nat
  ppt : Nat

/-- Calculates the remaining files after deletion --/
def remainingFiles (initial : InitialFiles) (deleted : DeletedFiles) : Nat :=
  initial.pdf + initial.word + initial.ppt - deleted.pdf - deleted.ppt

/-- Represents the folder allocation problem --/
structure FolderAllocationProblem where
  initial : InitialFiles
  deleted : DeletedFiles
  folderCapacity : Nat
  wordImportance : Nat

/-- Theorem: The minimum number of folders required is 6 --/
theorem min_folders_required (problem : FolderAllocationProblem)
  (h1 : problem.initial = ⟨43, 30, 30⟩)
  (h2 : problem.deleted = ⟨33, 30⟩)
  (h3 : problem.folderCapacity = 7)
  (h4 : problem.wordImportance = 2) :
  let remainingWordFiles := problem.initial.word
  let remainingPDFFiles := problem.initial.pdf - problem.deleted.pdf
  let totalRemainingFiles := remainingFiles problem.initial problem.deleted
  let minFolders := 
    (remainingWordFiles / problem.folderCapacity) +
    ((remainingWordFiles % problem.folderCapacity + remainingPDFFiles + problem.folderCapacity - 1) / problem.folderCapacity)
  minFolders = 6 := by
  sorry

end min_folders_required_l1306_130628


namespace eighth_term_is_negative_one_thirty_second_l1306_130670

/-- Sequence definition -/
def a (n : ℕ) : ℚ := (-1)^(n+1) * (n : ℚ) / 2^n

/-- Theorem: The 8th term of the sequence is -1/32 -/
theorem eighth_term_is_negative_one_thirty_second : a 8 = -1/32 := by
  sorry

end eighth_term_is_negative_one_thirty_second_l1306_130670


namespace intersection_of_A_and_B_l1306_130652

-- Define the sets A and B
def A : Set ℕ := {1, 2, 4}
def B : Set ℕ := {2, 4, 6}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = {2, 4} := by sorry

end intersection_of_A_and_B_l1306_130652


namespace v_2010_equals_0_l1306_130608

-- Define the function g
def g : ℕ → ℕ
| 0 => 2
| 1 => 5
| 2 => 3
| 3 => 0
| 4 => 1
| 5 => 4
| _ => 0  -- For completeness, although not used in the problem

-- Define the sequence v
def v : ℕ → ℕ
| 0 => 0
| (n + 1) => g (v n)

-- Theorem to prove
theorem v_2010_equals_0 : v 2010 = 0 := by
  sorry

end v_2010_equals_0_l1306_130608


namespace divisors_of_prime_products_l1306_130655

theorem divisors_of_prime_products (p q : ℕ) (m n : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hpq : p ≠ q) :
  let num_divisors := fun x => (Nat.divisors x).card
  (num_divisors (p * q) = 4) ∧
  (num_divisors (p^2 * q) = 6) ∧
  (num_divisors (p^2 * q^2) = 9) ∧
  (num_divisors (p^m * q^n) = (m + 1) * (n + 1)) :=
by sorry

end divisors_of_prime_products_l1306_130655


namespace spring_spending_is_1_7_l1306_130673

/-- The spending of Rivertown government in millions of dollars -/
structure RivertownSpending where
  /-- Total accumulated spending by the end of February -/
  february_end : ℝ
  /-- Total accumulated spending by the end of May -/
  may_end : ℝ

/-- The spending during March, April, and May -/
def spring_spending (s : RivertownSpending) : ℝ :=
  s.may_end - s.february_end

theorem spring_spending_is_1_7 (s : RivertownSpending) 
  (h_feb : s.february_end = 0.8)
  (h_may : s.may_end = 2.5) : 
  spring_spending s = 1.7 := by
  sorry

end spring_spending_is_1_7_l1306_130673


namespace complex_modulus_l1306_130617

theorem complex_modulus (z : ℂ) (h : z + 2*Complex.I - 3 = 3 - 3*Complex.I) : 
  Complex.abs z = Real.sqrt 61 := by
  sorry

end complex_modulus_l1306_130617


namespace product_of_distinct_solutions_l1306_130690

theorem product_of_distinct_solutions (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hxy : x ≠ y) 
  (h : x + 2 / x = y + 2 / y) : x * y = 2 := by
  sorry

end product_of_distinct_solutions_l1306_130690


namespace right_triangle_leg_sum_l1306_130601

theorem right_triangle_leg_sum (a b c : ℕ) : 
  a^2 + b^2 = c^2 →  -- Pythagorean theorem
  b = a + 2 →        -- legs differ by 2
  c = 53 →           -- hypotenuse is 53
  a + b = 104 :=     -- sum of legs is 104
by
  sorry

end right_triangle_leg_sum_l1306_130601


namespace total_pages_in_paper_l1306_130695

/-- Represents the number of days Stacy has to complete her history paper -/
def days_to_complete : ℕ := 7

/-- Represents the number of pages Stacy needs to write per day -/
def pages_per_day : ℕ := 9

/-- Theorem stating that the total number of pages in Stacy's history paper is 63 -/
theorem total_pages_in_paper : days_to_complete * pages_per_day = 63 := by
  sorry

end total_pages_in_paper_l1306_130695


namespace arcsin_plus_arccos_eq_pi_sixth_l1306_130643

theorem arcsin_plus_arccos_eq_pi_sixth (x : ℝ) :
  Real.arcsin x + Real.arccos (3 * x) = π / 6 → x = Real.sqrt (3 / 124) :=
by sorry

end arcsin_plus_arccos_eq_pi_sixth_l1306_130643


namespace line_equation_problem_l1306_130656

/-- Two lines are the same if their coefficients are proportional -/
def same_line (a b c : ℝ) (d e f : ℝ) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ k * a = d ∧ k * b = e ∧ k * c = f

/-- The problem statement -/
theorem line_equation_problem (p q : ℝ) :
  same_line p 2 7 3 q 5 → p = 21/5 := by
  sorry

end line_equation_problem_l1306_130656


namespace least_integer_with_divisibility_conditions_l1306_130618

def is_divisible (a b : ℕ) : Prop := b ≠ 0 ∧ a % b = 0

def consecutive (a b : ℕ) : Prop := b = a + 1

theorem least_integer_with_divisibility_conditions : 
  let N : ℕ := 2329089562800
  ∀ k : ℕ, k ≤ 30 → k ≠ 28 → k ≠ 29 → is_divisible N k ∧ 
  ¬is_divisible N 28 ∧ 
  ¬is_divisible N 29 ∧
  consecutive 28 29 ∧
  28 > 15 ∧ 29 > 15 ∧
  (∀ m : ℕ, m < N → 
    ¬(∀ j : ℕ, j ≤ 30 → j ≠ 28 → j ≠ 29 → is_divisible m j) ∨ 
    is_divisible m 28 ∨ 
    is_divisible m 29 ∨
    ¬(∃ p q : ℕ, p > 15 ∧ q > 15 ∧ consecutive p q ∧ ¬is_divisible m p ∧ ¬is_divisible m q)
  ) :=
by
  sorry

end least_integer_with_divisibility_conditions_l1306_130618


namespace geometric_sequence_common_ratio_l1306_130635

/-- Given a geometric sequence {a_n} with common ratio q, 
    if a_2 + a_4 = 20 and a_3 + a_5 = 40, then q = 2 -/
theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) 
  (q : ℝ) 
  (h1 : ∀ n, a (n + 1) = a n * q) 
  (h2 : a 2 + a 4 = 20) 
  (h3 : a 3 + a 5 = 40) : 
  q = 2 := by
sorry

end geometric_sequence_common_ratio_l1306_130635


namespace fixed_point_on_line_l1306_130659

theorem fixed_point_on_line (k : ℝ) : k * 2 + 0 - 2 * k = 0 := by
  sorry

end fixed_point_on_line_l1306_130659


namespace parabola_intersection_theorem_l1306_130686

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define the directrix of the parabola
def directrix (x : ℝ) : Prop := x = -1

-- Define a line passing through the focus
def line_through_focus (m : ℝ) (x y : ℝ) : Prop :=
  y - (focus.2) = m * (x - (focus.1))

-- Define the theorem
theorem parabola_intersection_theorem 
  (A B C : ℝ × ℝ) 
  (m : ℝ) 
  (h1 : parabola A.1 A.2)
  (h2 : parabola B.1 B.2)
  (h3 : directrix C.1)
  (h4 : line_through_focus m A.1 A.2)
  (h5 : line_through_focus m B.1 B.2)
  (h6 : line_through_focus m C.1 C.2)
  (h7 : A.2 * C.2 ≥ 0)  -- A and C on the same side of x-axis
  (h8 : Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2) = 
        2 * Real.sqrt ((A.1 - focus.1)^2 + (A.2 - focus.2)^2)) :
  Real.sqrt ((B.1 - focus.1)^2 + (B.2 - focus.2)^2) = 4 :=
sorry

end parabola_intersection_theorem_l1306_130686


namespace marks_candy_bars_l1306_130677

def total_candy_bars (snickers mars butterfingers : ℕ) : ℕ :=
  snickers + mars + butterfingers

theorem marks_candy_bars : total_candy_bars 3 2 7 = 12 := by
  sorry

end marks_candy_bars_l1306_130677


namespace meaningful_range_for_sqrt_fraction_l1306_130682

/-- The range of x for which the expression sqrt(x-1)/(x-3) is meaningful in the real number system. -/
theorem meaningful_range_for_sqrt_fraction (x : ℝ) :
  (∃ y : ℝ, y^2 = x - 1 ∧ x - 3 ≠ 0) ↔ x ≥ 1 ∧ x ≠ 3 :=
by sorry

end meaningful_range_for_sqrt_fraction_l1306_130682


namespace prime_power_sum_square_l1306_130664

/-- A function that checks if a number is prime -/
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

/-- A function that checks if a number is a perfect square -/
def isPerfectSquare (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

/-- The set of valid triples (p, q, r) -/
def validTriples : Set (ℕ × ℕ × ℕ) :=
  {(2, 5, 2), (2, 2, 5), (2, 3, 3), (3, 3, 2)} ∪ 
  {x | ∃ n : ℕ, x = (2, 2*n+1, 2*n+1)}

/-- The main theorem -/
theorem prime_power_sum_square (p q r : ℕ) :
  isPrime p ∧ isPrime q ∧ isPrime r ∧ 
  isPerfectSquare (p^q + p^r) ↔ 
  (p, q, r) ∈ validTriples := by sorry

end prime_power_sum_square_l1306_130664


namespace cal_anthony_transaction_ratio_l1306_130639

theorem cal_anthony_transaction_ratio :
  ∀ (mabel_transactions anthony_transactions cal_transactions jade_transactions : ℕ),
    mabel_transactions = 90 →
    anthony_transactions = mabel_transactions + mabel_transactions / 10 →
    jade_transactions = 84 →
    jade_transactions = cal_transactions + 18 →
    cal_transactions * 3 = anthony_transactions * 2 := by
  sorry

end cal_anthony_transaction_ratio_l1306_130639


namespace triangle_area_side_a_value_l1306_130623

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  cosA : ℝ

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.cosA = 1/2 ∧ t.b * t.c = 3

-- Theorem 1: Area of the triangle
theorem triangle_area (t : Triangle) (h : triangle_conditions t) : 
  (1/2 : ℝ) * t.b * t.c * Real.sqrt (1 - t.cosA^2) = Real.sqrt 3 / 2 := by
  sorry

-- Theorem 2: Value of side a when c = 1
theorem side_a_value (t : Triangle) (h : triangle_conditions t) (h_c : t.c = 1) :
  t.a = 2 * Real.sqrt 5 := by
  sorry

end triangle_area_side_a_value_l1306_130623


namespace square_sum_value_l1306_130624

theorem square_sum_value (x y : ℝ) 
  (h1 : (x + y)^2 = 36) 
  (h2 : x * y = 8) : 
  x^2 + y^2 = 20 := by
sorry

end square_sum_value_l1306_130624


namespace final_painting_height_l1306_130642

/-- Calculates the height of the final painting given the conditions -/
theorem final_painting_height :
  let total_paintings : ℕ := 5
  let total_area : ℝ := 200
  let small_painting_side : ℝ := 5
  let small_painting_count : ℕ := 3
  let large_painting_width : ℝ := 10
  let large_painting_height : ℝ := 8
  let final_painting_width : ℝ := 9
  
  let small_paintings_area : ℝ := small_painting_count * (small_painting_side * small_painting_side)
  let large_painting_area : ℝ := large_painting_width * large_painting_height
  let known_area : ℝ := small_paintings_area + large_painting_area
  let final_painting_area : ℝ := total_area - known_area
  
  final_painting_area / final_painting_width = 5 :=
by sorry

end final_painting_height_l1306_130642


namespace abc_value_l1306_130602

theorem abc_value (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hab : a * b = 30 * Real.rpow 3 (1/3))
  (hac : a * c = 42 * Real.rpow 3 (1/3))
  (hbc : b * c = 18 * Real.rpow 3 (1/3)) :
  a * b * c = 90 * Real.sqrt 3 := by
sorry

end abc_value_l1306_130602


namespace age_ratio_proof_l1306_130604

def henry_age : ℕ := 27
def jill_age : ℕ := 16

theorem age_ratio_proof :
  (henry_age + jill_age = 43) →
  (henry_age - 5) / (jill_age - 5) = 2 := by
  sorry

end age_ratio_proof_l1306_130604


namespace f_is_smallest_not_on_board_l1306_130678

/-- The game function that represents the number left on the board after subtraction -/
def g (k : ℕ) (x : ℕ) : ℕ := x^2 - k

/-- The smallest integer a such that g_{2n}(a) - g_{2n}(a-1) ≥ 3 -/
def x (n : ℕ) : ℕ := 2*n + 2

/-- The function f(2n) representing the smallest positive integer not written on the board -/
def f (n : ℕ) : ℕ := (2*n + 1)^2 - 2*n

/-- Theorem stating that f(2n) is the smallest positive integer not written on the board -/
theorem f_is_smallest_not_on_board (n : ℕ) :
  f n = (2*n + 1)^2 - 2*n ∧
  ∀ m < f n, ∃ i ≤ x n, m = g (2*n) i ∨ m = g (2*n) (i+1) :=
sorry

end f_is_smallest_not_on_board_l1306_130678
