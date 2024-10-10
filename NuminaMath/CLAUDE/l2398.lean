import Mathlib

namespace terriers_groomed_count_l2398_239888

/-- Represents the time in minutes to groom a poodle -/
def poodle_groom_time : ℕ := 30

/-- Represents the time in minutes to groom a terrier -/
def terrier_groom_time : ℕ := poodle_groom_time / 2

/-- Represents the number of poodles groomed -/
def poodles_groomed : ℕ := 3

/-- Represents the total grooming time in minutes -/
def total_groom_time : ℕ := 210

/-- Proves that the number of terriers groomed is 8 -/
theorem terriers_groomed_count : ℕ := by
  sorry

end terriers_groomed_count_l2398_239888


namespace unique_four_digit_square_repeated_digits_l2398_239893

-- Define a four-digit number with repeated digits
def fourDigitRepeated (x y : Nat) : Nat :=
  1100 * x + 11 * y

-- Theorem statement
theorem unique_four_digit_square_repeated_digits :
  ∃! n : Nat, 
    1000 ≤ n ∧ n < 10000 ∧  -- four-digit number
    (∃ x y : Nat, n = fourDigitRepeated x y) ∧  -- repeated digits
    (∃ m : Nat, n = m ^ 2) ∧  -- perfect square
    n = 7744 := by
  sorry


end unique_four_digit_square_repeated_digits_l2398_239893


namespace root_difference_implies_k_value_l2398_239859

theorem root_difference_implies_k_value :
  ∀ (k : ℝ) (r s : ℝ),
  (r^2 + k*r + 12 = 0) ∧ 
  (s^2 + k*s + 12 = 0) ∧
  ((r-3)^2 - k*(r-3) + 12 = 0) ∧ 
  ((s-3)^2 - k*(s-3) + 12 = 0) →
  k = -3 :=
by sorry

end root_difference_implies_k_value_l2398_239859


namespace magnitude_of_z_l2398_239812

theorem magnitude_of_z : ∀ z : ℂ, z = (Complex.abs (2 + Complex.I) + 2 * Complex.I) / Complex.I → Complex.abs z = 3 := by
  sorry

end magnitude_of_z_l2398_239812


namespace arithmetic_progression_includes_1999_l2398_239834

/-- An arithmetic progression is a sequence where the difference between
    successive terms is constant. -/
def IsArithmeticProgression (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

theorem arithmetic_progression_includes_1999
  (a : ℕ → ℝ)
  (d : ℝ)
  (h_positive : d > 0)
  (h_arithmetic : IsArithmeticProgression a d)
  (h_7 : ∃ n, a n = 7)
  (h_15 : ∃ n, a n = 15)
  (h_27 : ∃ n, a n = 27) :
  ∃ n, a n = 1999 :=
sorry

end arithmetic_progression_includes_1999_l2398_239834


namespace inequality_equivalence_l2398_239896

theorem inequality_equivalence (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  x / y + 1 / x + y ≥ y / x + 1 / y + x ↔ (x - y) * (x - 1) * (1 - y) ≥ 0 := by
  sorry

end inequality_equivalence_l2398_239896


namespace vector_dot_product_l2398_239869

/-- Given two vectors a and b in ℝ², where a is parallel to (a + b), prove that their dot product is 4. -/
theorem vector_dot_product (x : ℝ) : 
  let a : Fin 2 → ℝ := ![2, x]
  let b : Fin 2 → ℝ := ![1, -1]
  (∃ (k : ℝ), a = k • (a + b)) → 
  (a • b = 4) := by
sorry

end vector_dot_product_l2398_239869


namespace f_properties_l2398_239838

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := -(Real.sin x)^2 + a * Real.sin x - 1

theorem f_properties :
  (∀ x, f 1 x ≥ -3) ∧
  (∀ x, f 1 x = -3 → ∃ y, f 1 y = -3) ∧
  (∀ a, (∀ x, f a x ≤ 1/2) ∧ (∃ y, f a y = 1/2) ↔ a = -5/2 ∨ a = 5/2) :=
sorry

end f_properties_l2398_239838


namespace proportional_segments_l2398_239875

theorem proportional_segments (a b c d : ℝ) : 
  b = 3 → c = 6 → d = 9 → (a / b = c / d) → a = 2 := by sorry

end proportional_segments_l2398_239875


namespace square_sum_factorization_l2398_239805

theorem square_sum_factorization (x y : ℝ) : x^2 + 2*x*y + y^2 = (x + y)^2 := by
  sorry

end square_sum_factorization_l2398_239805


namespace selection_schemes_eq_240_l2398_239804

-- Define the number of people and cities
def total_people : ℕ := 6
def total_cities : ℕ := 4

-- Define the function to calculate the number of selection schemes
def selection_schemes : ℕ :=
  -- Options for city A (excluding person A and B)
  (total_people - 2) *
  -- Options for city B
  (total_people - 1) *
  -- Options for city C
  (total_people - 2) *
  -- Options for city D
  (total_people - 3)

-- Theorem to prove
theorem selection_schemes_eq_240 : selection_schemes = 240 := by
  sorry

end selection_schemes_eq_240_l2398_239804


namespace red_ball_removal_l2398_239816

theorem red_ball_removal (total : ℕ) (initial_red_percent : ℚ) (final_red_percent : ℚ) 
  (removed : ℕ) (h_total : total = 600) (h_initial_red : initial_red_percent = 70/100) 
  (h_final_red : final_red_percent = 60/100) (h_removed : removed = 150) : 
  (initial_red_percent * total - removed) / (total - removed) = final_red_percent := by
  sorry

end red_ball_removal_l2398_239816


namespace library_books_total_l2398_239889

theorem library_books_total (initial_books : ℕ) (additional_books : ℕ) : 
  initial_books = 54 → additional_books = 23 → initial_books + additional_books = 77 := by
sorry

end library_books_total_l2398_239889


namespace no_distributive_laws_hold_l2398_239891

-- Define the # operation
def hash (a b : ℝ) : ℝ := a + 2*b

-- Theorem stating that none of the distributive laws hold
theorem no_distributive_laws_hold :
  ¬(∀ (x y z : ℝ), hash x (y + z) = hash x y + hash x z) ∧
  ¬(∀ (x y z : ℝ), x + hash y z = hash (x + y) (x + z)) ∧
  ¬(∀ (x y z : ℝ), hash x (hash y z) = hash (hash x y) (hash x z)) :=
by sorry

end no_distributive_laws_hold_l2398_239891


namespace triangle_circle_area_l2398_239846

theorem triangle_circle_area (a : ℝ) (h : a > 0) : 
  let base := a
  let angle1 := Real.pi / 4  -- 45 degrees in radians
  let angle2 := Real.pi / 12 -- 15 degrees in radians
  let height := a / (1 + Real.tan (Real.pi / 12))
  let circle_area := Real.pi * height^2
  let sector_angle := 2 * Real.pi / 3 -- 120 degrees in radians
  sector_angle / (2 * Real.pi) * circle_area = (Real.pi * a^2 * (2 - Real.sqrt 3)) / 18
  := by sorry

end triangle_circle_area_l2398_239846


namespace correct_arrangement_count_l2398_239829

/-- The number of ways to arrange 5 people in a row with two specific people having exactly one person between them -/
def arrangement_count : ℕ := 36

/-- The number of people in the row -/
def total_people : ℕ := 5

/-- The number of people that can be placed between the two specific people -/
def middle_choices : ℕ := 3

/-- The number of ways to arrange the two specific people with one person between them -/
def specific_arrangement : ℕ := 2

/-- The number of ways to arrange the group of three (two specific people and the one between them) with the other two people -/
def group_arrangement : ℕ := 6

theorem correct_arrangement_count :
  arrangement_count = middle_choices * specific_arrangement * group_arrangement :=
sorry

end correct_arrangement_count_l2398_239829


namespace probability_nine_heads_in_twelve_flips_l2398_239849

def n : ℕ := 12
def k : ℕ := 9

theorem probability_nine_heads_in_twelve_flips :
  (n.choose k : ℚ) / 2^n = 220 / 4096 := by sorry

end probability_nine_heads_in_twelve_flips_l2398_239849


namespace fraction_relations_l2398_239845

theorem fraction_relations (x y : ℚ) (h : x / y = 2 / 5) :
  (x + y) / y = 7 / 5 ∧ 
  y / (y - x) = 5 / 3 ∧ 
  x / (3 * y) = 2 / 15 ∧ 
  (x + 3 * y) / x ≠ 17 / 2 ∧ 
  (x - y) / y ≠ 3 / 5 := by
  sorry

end fraction_relations_l2398_239845


namespace wendys_cookies_l2398_239857

/-- Represents the number of pastries in various categories -/
structure Pastries where
  cupcakes : ℕ
  cookies : ℕ
  taken_home : ℕ
  sold : ℕ

/-- The theorem statement for Wendy's bake sale problem -/
theorem wendys_cookies (w : Pastries) 
  (h1 : w.cupcakes = 4)
  (h2 : w.taken_home = 24)
  (h3 : w.sold = 9)
  (h4 : w.cupcakes + w.cookies = w.taken_home + w.sold) :
  w.cookies = 29 := by
  sorry

end wendys_cookies_l2398_239857


namespace exponent_division_equality_l2398_239801

theorem exponent_division_equality (a b : ℝ) :
  (a^2 * b)^3 / ((-a * b)^2) = a^4 * b :=
by sorry

end exponent_division_equality_l2398_239801


namespace triangle_arctans_sum_l2398_239884

theorem triangle_arctans_sum (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : b^2 + c^2 = a^2) (h5 : Real.arcsin (1/2) + Real.arcsin (1/2) = Real.pi/2) :
  Real.arctan (b/(c+a)) + Real.arctan (c/(b+a)) = Real.pi/4 := by
sorry

end triangle_arctans_sum_l2398_239884


namespace triangle_side_length_l2398_239803

theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) :
  A = π/4 →
  2*b*(Real.sin B) - c*(Real.sin C) = 2*a*(Real.sin A) →
  (1/2)*b*c*(Real.sin A) = 3 →
  c = 2*(Real.sqrt 2) :=
by sorry

end triangle_side_length_l2398_239803


namespace sum_congruence_l2398_239819

theorem sum_congruence : (1 + 23 + 456 + 7890) % 7 = 0 := by
  sorry

end sum_congruence_l2398_239819


namespace largest_five_digit_with_product_120_l2398_239867

def is_five_digit (n : ℕ) : Prop := n ≥ 10000 ∧ n < 100000

def digit_product (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.prod

theorem largest_five_digit_with_product_120 :
  ∀ n : ℕ, is_five_digit n → digit_product n = 120 → n ≤ 85311 :=
by sorry

end largest_five_digit_with_product_120_l2398_239867


namespace marble_jar_ratio_l2398_239820

/-- Given three jars of marbles with specific conditions, prove the ratio of marbles in Jar C to Jar B -/
theorem marble_jar_ratio :
  let jar_a : ℕ := 28
  let jar_b : ℕ := jar_a + 12
  let total : ℕ := 148
  let jar_c : ℕ := total - (jar_a + jar_b)
  (jar_c : ℚ) / jar_b = 2 / 1 :=
by
  sorry

end marble_jar_ratio_l2398_239820


namespace opposite_of_negative_2023_l2398_239824

-- Define the concept of opposite
def opposite (a : ℤ) : ℤ := -a

-- Theorem statement
theorem opposite_of_negative_2023 : opposite (-2023) = 2023 := by
  sorry

end opposite_of_negative_2023_l2398_239824


namespace arithmetic_calculations_l2398_239890

theorem arithmetic_calculations : 
  ((1 : Int) * (-11) + 8 + (-14) = -17) ∧ 
  (13 - (-12) + (-21) = 4) := by
  sorry

end arithmetic_calculations_l2398_239890


namespace binomial_600_0_l2398_239873

theorem binomial_600_0 : (600 : ℕ).choose 0 = 1 := by sorry

end binomial_600_0_l2398_239873


namespace samantha_overall_percentage_l2398_239887

/-- Represents an exam with its number of questions, weight per question, and percentage correct --/
structure Exam where
  questions : ℕ
  weight : ℕ
  percentCorrect : ℚ

/-- Calculates the total weighted questions for an exam --/
def totalWeightedQuestions (e : Exam) : ℚ :=
  (e.questions * e.weight : ℚ)

/-- Calculates the number of weighted questions answered correctly for an exam --/
def weightedQuestionsCorrect (e : Exam) : ℚ :=
  e.percentCorrect * totalWeightedQuestions e

/-- Calculates the overall percentage of weighted questions answered correctly across multiple exams --/
def overallPercentageCorrect (exams : List Exam) : ℚ :=
  let totalCorrect := (exams.map weightedQuestionsCorrect).sum
  let totalQuestions := (exams.map totalWeightedQuestions).sum
  totalCorrect / totalQuestions

/-- The three exams Samantha took --/
def samanthasExams : List Exam :=
  [{ questions := 30, weight := 1, percentCorrect := 75/100 },
   { questions := 50, weight := 1, percentCorrect := 80/100 },
   { questions := 20, weight := 2, percentCorrect := 65/100 }]

theorem samantha_overall_percentage :
  ∃ (ε : ℚ), ε > 0 ∧ ε < 1/100 ∧ 
  |overallPercentageCorrect samanthasExams - 74/100| < ε :=
sorry

end samantha_overall_percentage_l2398_239887


namespace eighth_of_two_power_44_l2398_239821

theorem eighth_of_two_power_44 (x : ℤ) :
  (2^44 : ℚ) / 8 = 2^x → x = 41 := by
  sorry

end eighth_of_two_power_44_l2398_239821


namespace diameter_scientific_notation_l2398_239813

-- Define the original diameter value
def original_diameter : ℝ := 0.000000103

-- Define the scientific notation components
def coefficient : ℝ := 1.03
def exponent : ℤ := -7

-- Theorem to prove the equality
theorem diameter_scientific_notation :
  original_diameter = coefficient * (10 : ℝ) ^ exponent :=
by
  sorry

end diameter_scientific_notation_l2398_239813


namespace parabola_point_x_coordinate_l2398_239892

/-- The x-coordinate of a point on a parabola at a given distance from the directrix -/
theorem parabola_point_x_coordinate 
  (x y : ℝ) -- x and y coordinates of point M
  (h1 : y^2 = 4*x) -- point M is on the parabola y² = 4x
  (h2 : |x + 1| = 3) -- distance from M to the directrix x = -1 is 3
  : x = 2 := by sorry

end parabola_point_x_coordinate_l2398_239892


namespace fraction_power_multiplication_compute_fraction_power_l2398_239871

theorem fraction_power_multiplication (a b c : ℚ) (n : ℕ) :
  a * (b / c)^n = (a * b^n) / c^n :=
by sorry

theorem compute_fraction_power : 7 * (1 / 5)^3 = 7 / 125 :=
by sorry

end fraction_power_multiplication_compute_fraction_power_l2398_239871


namespace unique_natural_number_l2398_239848

theorem unique_natural_number : ∃! n : ℕ, 
  (∃ a : ℕ, n - 45 = a^2) ∧ 
  (∃ b : ℕ, n + 44 = b^2) ∧ 
  n = 1981 := by
  sorry

end unique_natural_number_l2398_239848


namespace impossible_three_similar_parts_l2398_239874

theorem impossible_three_similar_parts : 
  ∀ (x : ℝ), x > 0 → ¬∃ (a b c : ℝ), 
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    a + b + c = x ∧
    a ≤ b ∧ b ≤ c ∧
    c ≤ Real.sqrt 2 * b ∧
    b ≤ Real.sqrt 2 * a :=
by sorry

end impossible_three_similar_parts_l2398_239874


namespace matrix_equation_solution_l2398_239837

theorem matrix_equation_solution (x : ℝ) : 
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![2, 4; 1, x]
  let B : Matrix (Fin 2) (Fin 2) ℝ := !![2, -2; -1, 1]
  B * A = !![2, 4; -1, -2] → x = 2 := by
  sorry

end matrix_equation_solution_l2398_239837


namespace inequality_proof_l2398_239807

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  Real.rpow (a^2 / (b + c)^2) (1/3) + 
  Real.rpow (b^2 / (c + a)^2) (1/3) + 
  Real.rpow (c^2 / (a + b)^2) (1/3) ≥ 
  3 / Real.rpow 4 (1/3) := by
  sorry

end inequality_proof_l2398_239807


namespace fermat_prime_l2398_239858

theorem fermat_prime (n : ℕ) (p : ℕ) (h1 : p = 2^n + 1) 
  (h2 : (3^((p-1)/2) + 1) % p = 0) : Nat.Prime p := by
  sorry

end fermat_prime_l2398_239858


namespace tile_square_side_length_l2398_239899

/-- Given tiles with width 16 and length 24, proves that the side length of a square
    formed by a minimum of 6 tiles is 48. -/
theorem tile_square_side_length
  (tile_width : ℕ) (tile_length : ℕ) (min_tiles : ℕ)
  (hw : tile_width = 16)
  (hl : tile_length = 24)
  (hm : min_tiles = 6) :
  2 * tile_length = 3 * tile_width ∧ 2 * tile_length = 48 := by
  sorry

#check tile_square_side_length

end tile_square_side_length_l2398_239899


namespace poes_speed_l2398_239839

theorem poes_speed (teena_speed : ℝ) (initial_distance : ℝ) (final_distance : ℝ) (time : ℝ) :
  teena_speed = 55 →
  initial_distance = 7.5 →
  final_distance = 15 →
  time = 1.5 →
  ∃ (poe_speed : ℝ), 
    poe_speed = 40 ∧
    teena_speed * time - poe_speed * time = initial_distance + final_distance :=
by
  sorry

end poes_speed_l2398_239839


namespace parallel_vectors_x_value_l2398_239897

/-- Two vectors in R² are parallel if and only if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_x_value :
  ∀ x : ℝ, parallel (x, 2) (1, 6) → x = 1/3 := by
  sorry

end parallel_vectors_x_value_l2398_239897


namespace fourth_term_coefficient_l2398_239827

theorem fourth_term_coefficient : 
  let a := (1/2 : ℚ)
  let b := (2/3 : ℚ)
  let n := 6
  let k := 4
  (n.choose (k-1)) * a^(n-(k-1)) * b^(k-1) = 20 := by sorry

end fourth_term_coefficient_l2398_239827


namespace complex_equation_solution_l2398_239860

theorem complex_equation_solution (z : ℂ) : z * (1 + Complex.I) = 2 * Complex.I → z = 1 + Complex.I := by
  sorry

end complex_equation_solution_l2398_239860


namespace eighteenth_prime_l2398_239856

-- Define a function that returns the nth prime number
def nthPrime (n : ℕ) : ℕ :=
  sorry

-- State the theorem
theorem eighteenth_prime :
  (nthPrime 7 = 17) → (nthPrime 18 = 67) :=
by sorry

end eighteenth_prime_l2398_239856


namespace geometric_mean_minimum_l2398_239876

theorem geometric_mean_minimum (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h : Real.sqrt 2 = Real.sqrt (4^a * 2^b)) :
  2/a + 1/b ≥ 9 := by
  sorry

end geometric_mean_minimum_l2398_239876


namespace modulo_equivalence_56234_l2398_239808

theorem modulo_equivalence_56234 :
  ∃! n : ℤ, 0 ≤ n ∧ n < 23 ∧ 56234 ≡ n [ZMOD 23] ∧ n = 1 := by
  sorry

end modulo_equivalence_56234_l2398_239808


namespace truck_distance_l2398_239894

/-- Proves that a truck traveling at a rate of 2 miles per 4 minutes will cover 90 miles in 3 hours -/
theorem truck_distance (rate : ℚ) (time : ℚ) : 
  rate = 2 / 4 → time = 3 * 60 → rate * time = 90 :=
by sorry

end truck_distance_l2398_239894


namespace salary_increase_percentage_l2398_239878

theorem salary_increase_percentage (initial_salary final_salary : ℝ) 
  (h1 : initial_salary = 1000.0000000000001)
  (h2 : final_salary = 1045) :
  ∃ P : ℝ, 
    (P = 10) ∧ 
    (final_salary = initial_salary * (1 + P / 100) * (1 - 5 / 100)) := by
  sorry

end salary_increase_percentage_l2398_239878


namespace merry_go_round_revolutions_l2398_239852

theorem merry_go_round_revolutions (outer_radius inner_radius : ℝ) 
  (outer_revolutions : ℕ) (h1 : outer_radius = 30) (h2 : inner_radius = 5) 
  (h3 : outer_revolutions = 15) : 
  ∃ inner_revolutions : ℕ, 
    (2 * Real.pi * outer_radius * outer_revolutions) = 
    (2 * Real.pi * inner_radius * inner_revolutions) ∧ 
    inner_revolutions = 90 := by
  sorry

end merry_go_round_revolutions_l2398_239852


namespace blue_ball_probability_l2398_239850

theorem blue_ball_probability (initial_total : ℕ) (initial_blue : ℕ) (removed_blue : ℕ) :
  initial_total = 18 →
  initial_blue = 6 →
  removed_blue = 3 →
  (initial_blue - removed_blue : ℚ) / (initial_total - removed_blue : ℚ) = 1 / 5 := by
sorry

end blue_ball_probability_l2398_239850


namespace even_odd_sum_difference_l2398_239810

def first_n_even_sum (n : ℕ) : ℕ := n * (n + 1)

def first_n_odd_sum (n : ℕ) : ℕ := n^2

theorem even_odd_sum_difference :
  first_n_even_sum 1500 - first_n_odd_sum 1500 = 1500 := by
  sorry

end even_odd_sum_difference_l2398_239810


namespace least_subtraction_for_divisibility_least_subtraction_62575_99_l2398_239832

theorem least_subtraction_for_divisibility (n : ℕ) (d : ℕ) (h : d > 0) :
  ∃ (k : ℕ), k < d ∧ (n - k) % d = 0 ∧ ∀ (m : ℕ), m < k → (n - m) % d ≠ 0 :=
by
  sorry

theorem least_subtraction_62575_99 :
  ∃ (k : ℕ), k < 99 ∧ (62575 - k) % 99 = 0 ∧ ∀ (m : ℕ), m < k → (62575 - m) % 99 ≠ 0 ∧ k = 43 :=
by
  sorry

end least_subtraction_for_divisibility_least_subtraction_62575_99_l2398_239832


namespace cyclic_inequality_l2398_239831

theorem cyclic_inequality (x₁ x₂ x₃ x₄ x₅ x₆ x₇ x₈ x₉ : ℝ) 
  (h₁ : x₁ > 0) (h₂ : x₂ > 0) (h₃ : x₃ > 0) (h₄ : x₄ > 0) (h₅ : x₅ > 0)
  (h₆ : x₆ > 0) (h₇ : x₇ > 0) (h₈ : x₈ > 0) (h₉ : x₉ > 0) :
  (x₁ - x₃) / (x₁ * x₃ + 2 * x₂ * x₃ + x₂^2) +
  (x₂ - x₄) / (x₂ * x₄ + 2 * x₃ * x₄ + x₃^2) +
  (x₃ - x₅) / (x₃ * x₅ + 2 * x₄ * x₅ + x₄^2) +
  (x₄ - x₆) / (x₄ * x₆ + 2 * x₅ * x₆ + x₅^2) +
  (x₅ - x₇) / (x₅ * x₇ + 2 * x₆ * x₇ + x₆^2) +
  (x₆ - x₈) / (x₆ * x₈ + 2 * x₇ * x₈ + x₇^2) +
  (x₇ - x₉) / (x₇ * x₉ + 2 * x₈ * x₉ + x₈^2) +
  (x₈ - x₁) / (x₈ * x₁ + 2 * x₉ * x₁ + x₉^2) +
  (x₉ - x₂) / (x₉ * x₂ + 2 * x₁ * x₂ + x₁^2) ≥ 0 := by
  sorry

end cyclic_inequality_l2398_239831


namespace bob_cleaning_time_l2398_239840

/-- Given that Alice takes 30 minutes to clean her room and Bob takes 1/3 of Alice's time,
    prove that Bob takes 10 minutes to clean his room. -/
theorem bob_cleaning_time (alice_time bob_time : ℚ) : 
  alice_time = 30 → bob_time = (1/3) * alice_time → bob_time = 10 := by
  sorry

end bob_cleaning_time_l2398_239840


namespace yahs_to_bahs_l2398_239881

-- Define the units
variable (bah rah yah : ℕ → ℚ)

-- Define the conversion rates
axiom bah_to_rah : ∀ x, bah x = rah (2 * x)
axiom rah_to_yah : ∀ x, rah x = yah (2 * x)

-- State the theorem
theorem yahs_to_bahs : yah 1200 = bah 300 := by
  sorry

end yahs_to_bahs_l2398_239881


namespace double_area_right_triangle_l2398_239800

/-- The area of a triangle with double the area of a right-angled triangle -/
theorem double_area_right_triangle (a b : ℝ) : 
  let triangle_I_base : ℝ := a + b
  let triangle_I_height : ℝ := a + b
  let triangle_I_area : ℝ := (1 / 2) * triangle_I_base * triangle_I_height
  let triangle_II_area : ℝ := 2 * triangle_I_area
  triangle_II_area = (a + b)^2 := by
  sorry

end double_area_right_triangle_l2398_239800


namespace anniversary_18_months_ago_proof_l2398_239885

/-- The anniversary Bella and Bob celebrated 18 months ago -/
def anniversary_18_months_ago : ℕ := 2

/-- The number of months until their 4th anniversary -/
def months_until_4th_anniversary : ℕ := 6

/-- The current duration of their relationship in months -/
def current_relationship_duration : ℕ := 4 * 12 - months_until_4th_anniversary

/-- The duration of their relationship 18 months ago in months -/
def relationship_duration_18_months_ago : ℕ := current_relationship_duration - 18

theorem anniversary_18_months_ago_proof :
  anniversary_18_months_ago = relationship_duration_18_months_ago / 12 :=
by sorry

end anniversary_18_months_ago_proof_l2398_239885


namespace equal_area_division_l2398_239868

/-- A point on a grid --/
structure GridPoint where
  x : ℤ
  y : ℤ

/-- A figure on a grid --/
structure GridFigure where
  area : ℚ
  points : Set GridPoint

/-- A ray on a grid --/
structure GridRay where
  start : GridPoint
  direction : GridPoint

/-- Theorem: There exists a ray that divides a figure of area 9 into two equal parts --/
theorem equal_area_division (fig : GridFigure) (A : GridPoint) :
  fig.area = 9 →
  ∃ (B : GridPoint) (ray : GridRay),
    B ≠ A ∧
    ray.start = A ∧
    (∃ (t : ℚ), ray.start.x + t * ray.direction.x = B.x ∧ ray.start.y + t * ray.direction.y = B.y) ∧
    ∃ (left_area right_area : ℚ),
      left_area = right_area ∧
      left_area + right_area = fig.area := by
  sorry

end equal_area_division_l2398_239868


namespace temperature_80_degrees_l2398_239818

-- Define the temperature function
def temperature (t : ℝ) : ℝ := -t^2 + 10*t + 60

-- State the theorem
theorem temperature_80_degrees :
  ∃ t₁ t₂ : ℝ, 
    t₁ = 5 + 3 * Real.sqrt 5 ∧ 
    t₂ = 5 - 3 * Real.sqrt 5 ∧ 
    temperature t₁ = 80 ∧ 
    temperature t₂ = 80 ∧ 
    (∀ t : ℝ, temperature t = 80 → t = t₁ ∨ t = t₂) := by
  sorry

end temperature_80_degrees_l2398_239818


namespace philips_weekly_mileage_l2398_239886

/-- Calculate Philip's car's mileage for a typical week -/
theorem philips_weekly_mileage (school_distance : ℝ) (market_distance : ℝ)
  (school_trips_per_day : ℕ) (school_days_per_week : ℕ) (market_trips_per_week : ℕ)
  (h1 : school_distance = 2.5)
  (h2 : market_distance = 2)
  (h3 : school_trips_per_day = 2)
  (h4 : school_days_per_week = 4)
  (h5 : market_trips_per_week = 1) :
  school_distance * 2 * ↑school_trips_per_day * ↑school_days_per_week +
  market_distance * 2 * ↑market_trips_per_week = 44 := by
  sorry

#check philips_weekly_mileage

end philips_weekly_mileage_l2398_239886


namespace complex_equation_solution_l2398_239815

theorem complex_equation_solution (z : ℂ) :
  (Complex.I * 3 + Real.sqrt 3) * z = Complex.I * 3 →
  z = Complex.mk (3 / 4) (Real.sqrt 3 / 4) := by
  sorry

end complex_equation_solution_l2398_239815


namespace parabola_equation_l2398_239855

/-- A parabola with specified properties -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ
  axis_of_symmetry : a ≠ 0 → b = -4 * a
  tangent_line : ∃ x : ℝ, a * x^2 + b * x + c = 2 * x + 1 ∧
                 ∀ y : ℝ, y ≠ x → a * y^2 + b * y + c > 2 * y + 1
  y_intercepts : ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
                 a * x₁^2 + b * x₁ + c = 0 ∧
                 a * x₂^2 + b * x₂ + c = 0 ∧
                 (x₁ - x₂)^2 = 8

/-- The parabola equation is one of the two specified forms -/
theorem parabola_equation (p : Parabola) : 
  (p.a = 1 ∧ p.b = 4 ∧ p.c = 2) ∨ (p.a = 1/2 ∧ p.b = 2 ∧ p.c = 1) :=
sorry

end parabola_equation_l2398_239855


namespace derivative_of_2_sqrt_x_cubed_l2398_239822

theorem derivative_of_2_sqrt_x_cubed (x : ℝ) (h : x > 0) :
  deriv (λ x => 2 * Real.sqrt (x^3)) x = 3 * Real.sqrt x :=
by sorry

end derivative_of_2_sqrt_x_cubed_l2398_239822


namespace factorization_200_perfect_square_factors_200_l2398_239825

/-- A function that returns the number of positive factors of n that are perfect squares -/
def perfect_square_factors (n : ℕ) : ℕ := sorry

/-- The prime factorization of 200 is 2^3 * 5^2 -/
theorem factorization_200 : 200 = 2^3 * 5^2 := sorry

/-- Theorem stating that the number of positive factors of 200 that are perfect squares is 4 -/
theorem perfect_square_factors_200 : perfect_square_factors 200 = 4 := by sorry

end factorization_200_perfect_square_factors_200_l2398_239825


namespace f_equals_g_l2398_239847

-- Define the functions f and g
def f (x : ℝ) : ℝ := x
def g (x : ℝ) : ℝ := 5 * x^5

-- Theorem statement
theorem f_equals_g : f = g := by sorry

end f_equals_g_l2398_239847


namespace exist_k_m_with_prime_divisor_diff_l2398_239861

/-- The number of prime divisors of a positive integer -/
def num_prime_divisors (n : ℕ+) : ℕ := sorry

/-- For any positive integer n, there exist positive integers k and m such that 
    k - m = n and the number of prime divisors of k is exactly one more than 
    the number of prime divisors of m -/
theorem exist_k_m_with_prime_divisor_diff (n : ℕ+) : 
  ∃ (k m : ℕ+), k - m = n ∧ num_prime_divisors k = num_prime_divisors m + 1 := by sorry

end exist_k_m_with_prime_divisor_diff_l2398_239861


namespace cricket_bat_price_l2398_239854

theorem cricket_bat_price (profit_A_to_B : ℝ) (profit_B_to_C : ℝ) (price_C : ℝ) : 
  profit_A_to_B = 0.20 →
  profit_B_to_C = 0.25 →
  price_C = 222 →
  ∃ (cost_price_A : ℝ), cost_price_A = 148 ∧ 
    price_C = cost_price_A * (1 + profit_A_to_B) * (1 + profit_B_to_C) :=
by
  sorry

end cricket_bat_price_l2398_239854


namespace player_positions_satisfy_distances_l2398_239883

/-- Represents the positions of four players on a number line -/
def PlayerPositions : Fin 4 → ℝ
| 0 => 0
| 1 => 1
| 2 => 4
| 3 => 6

/-- Calculates the distance between two players -/
def distance (i j : Fin 4) : ℝ :=
  |PlayerPositions i - PlayerPositions j|

/-- The set of required pairwise distances -/
def RequiredDistances : Set ℝ := {1, 2, 3, 4, 5, 6}

/-- Theorem stating that the player positions satisfy the required distances -/
theorem player_positions_satisfy_distances :
  ∀ i j : Fin 4, i ≠ j → distance i j ∈ RequiredDistances := by
  sorry

#check player_positions_satisfy_distances

end player_positions_satisfy_distances_l2398_239883


namespace fraction_comparison_l2398_239880

theorem fraction_comparison (x y : ℝ) (h1 : x > y) (h2 : y > 2) :
  y / (y^2 - y + 1) > x / (x^2 - x + 1) := by
  sorry

end fraction_comparison_l2398_239880


namespace trigonometric_identities_l2398_239870

theorem trigonometric_identities (α : ℝ) (h : 2 * Real.sin α + Real.cos α = 0) :
  (((2 * Real.cos α - Real.sin α) / (Real.sin α + Real.cos α)) = 5) ∧
  ((Real.sin α / (Real.sin α ^ 3 - Real.cos α ^ 3)) = 5/3) := by
  sorry

end trigonometric_identities_l2398_239870


namespace ten_coin_flips_sequences_l2398_239842

/-- The number of distinct sequences possible when flipping a coin n times -/
def coinFlipSequences (n : ℕ) : ℕ := 2^n

/-- Theorem: The number of distinct sequences possible when flipping a coin 10 times is 1024 -/
theorem ten_coin_flips_sequences : coinFlipSequences 10 = 1024 := by
  sorry

end ten_coin_flips_sequences_l2398_239842


namespace inequality_solution_l2398_239863

theorem inequality_solution (x : ℝ) : 
  (3 - 1 / (3 * x + 2) < 5) ↔ (x < -5/3 ∨ x > -2/3) := by sorry

end inequality_solution_l2398_239863


namespace group_size_l2398_239843

/-- The number of people in the group -/
def n : ℕ := sorry

/-- The original weight of each person in kg -/
def original_weight : ℝ := 50

/-- The weight of the new person in kg -/
def new_person_weight : ℝ := 70

/-- The average weight increase in kg -/
def average_increase : ℝ := 2.5

theorem group_size :
  (n : ℝ) * (original_weight + average_increase) = n * original_weight + (new_person_weight - original_weight) →
  n = 8 :=
by sorry

end group_size_l2398_239843


namespace trig_problem_l2398_239866

theorem trig_problem (θ : ℝ) 
  (h : (2 * Real.cos ((3/2) * Real.pi + θ) + Real.cos (Real.pi + θ)) / 
       (3 * Real.sin (Real.pi - θ) + 2 * Real.sin ((5/2) * Real.pi + θ)) = 1/5) : 
  Real.tan θ = 1 ∧ Real.sin θ^2 + 3 * Real.sin θ * Real.cos θ = 2 := by
  sorry

end trig_problem_l2398_239866


namespace min_abs_z_l2398_239802

theorem min_abs_z (z : ℂ) (h : Complex.abs (z - 8) + Complex.abs (z - Complex.I * 7) = 15) :
  ∃ (w : ℂ), Complex.abs w ≤ Complex.abs z ∧ Complex.abs (w - 8) + Complex.abs (w - Complex.I * 7) = 15 ∧ Complex.abs w = 56 / 15 :=
by
  sorry

end min_abs_z_l2398_239802


namespace log_ratio_equality_l2398_239895

-- Define the logarithm base 10 function
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Theorem statement
theorem log_ratio_equality (m n : ℝ) 
  (h1 : log10 2 = m) 
  (h2 : log10 3 = n) : 
  (log10 12) / (log10 15) = (2*m + n) / (1 - m + n) := by
  sorry

end log_ratio_equality_l2398_239895


namespace valid_permutations_64420_l2398_239898

def digits : List Nat := [6, 4, 4, 2, 0]

/-- The number of permutations of the digits that form a 5-digit number not starting with 0 -/
def valid_permutations (ds : List Nat) : Nat :=
  let non_zero_digits := ds.filter (· ≠ 0)
  let zero_digits := ds.filter (· = 0)
  non_zero_digits.length * (ds.length - 1).factorial / (non_zero_digits.map (λ d => (ds.filter (· = d)).length)).prod

theorem valid_permutations_64420 :
  valid_permutations digits = 48 := by
  sorry

end valid_permutations_64420_l2398_239898


namespace geometric_sequence_ratio_l2398_239817

/-- A geometric sequence is a sequence where each term after the first is found by 
    multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def IsGeometricSequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

theorem geometric_sequence_ratio 
  (a : ℕ → ℝ) (q : ℝ) (h : IsGeometricSequence a q) 
  (h_eq : 16 * a 6 = a 2) :
  q = 1/2 ∨ q = -1/2 := by
  sorry

end geometric_sequence_ratio_l2398_239817


namespace trig_identity_l2398_239836

theorem trig_identity (α : Real) (h : Real.tan (π + α) = 2) :
  4 * Real.sin α * Real.cos α + 3 * (Real.cos α)^2 = 11/5 := by
  sorry

end trig_identity_l2398_239836


namespace probability_at_least_one_tenth_grade_l2398_239862

/-- The number of volunteers from the 10th grade -/
def tenth_grade_volunteers : ℕ := 2

/-- The number of volunteers from the 11th grade -/
def eleventh_grade_volunteers : ℕ := 4

/-- The total number of volunteers -/
def total_volunteers : ℕ := tenth_grade_volunteers + eleventh_grade_volunteers

/-- The number of volunteers to be selected -/
def selected_volunteers : ℕ := 2

/-- The probability of selecting at least one volunteer from the 10th grade -/
theorem probability_at_least_one_tenth_grade :
  (1 : ℚ) - (Nat.choose eleventh_grade_volunteers selected_volunteers : ℚ) / 
  (Nat.choose total_volunteers selected_volunteers : ℚ) = 3/5 := by sorry

end probability_at_least_one_tenth_grade_l2398_239862


namespace fixed_distance_point_l2398_239814

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

/-- Given vectors a and b, if p satisfies ‖p - b‖ = 3 ‖p - a‖, 
    then p is at a fixed distance from (9/8)a - (1/8)b -/
theorem fixed_distance_point (a b p : V) 
  (h : ‖p - b‖ = 3 * ‖p - a‖) :
  ∃ (c : ℝ), ∀ (q : V), 
    (‖q - b‖ = 3 * ‖q - a‖) → 
    ‖q - ((9/8 : ℝ) • a - (1/8 : ℝ) • b)‖ = c :=
sorry

end fixed_distance_point_l2398_239814


namespace remainder_sum_l2398_239851

theorem remainder_sum (p q : ℤ) (hp : p % 80 = 75) (hq : q % 120 = 115) : (p + q) % 40 = 30 := by
  sorry

end remainder_sum_l2398_239851


namespace a_eq_one_sufficient_not_necessary_for_a_sq_eq_one_l2398_239872

theorem a_eq_one_sufficient_not_necessary_for_a_sq_eq_one :
  ∃ (a : ℝ), (a = 1 → a^2 = 1) ∧ ¬(a^2 = 1 → a = 1) :=
by sorry

end a_eq_one_sufficient_not_necessary_for_a_sq_eq_one_l2398_239872


namespace gcd_2146_1813_l2398_239835

theorem gcd_2146_1813 : Nat.gcd 2146 1813 = 37 := by
  sorry

end gcd_2146_1813_l2398_239835


namespace smallest_solution_5x2_eq_3y5_l2398_239865

theorem smallest_solution_5x2_eq_3y5 :
  ∃! (x y : ℕ), 
    (5 * x^2 = 3 * y^5) ∧ 
    (∀ (a b : ℕ), (5 * a^2 = 3 * b^5) → (x ≤ a ∧ y ≤ b)) ∧
    x = 675 ∧ y = 15 := by sorry

end smallest_solution_5x2_eq_3y5_l2398_239865


namespace smallest_number_with_same_factors_l2398_239879

def alice_number : Nat := 30

-- Bob's number must have all prime factors of Alice's number
def has_all_prime_factors (m n : Nat) : Prop :=
  ∀ p : Nat, Nat.Prime p → (p ∣ n → p ∣ m)

-- The theorem to prove
theorem smallest_number_with_same_factors (n : Nat) (h : n = alice_number) :
  ∃ m : Nat, has_all_prime_factors m n ∧ 
  (∀ k : Nat, has_all_prime_factors k n → m ≤ k) ∧
  m = n :=
sorry

end smallest_number_with_same_factors_l2398_239879


namespace log_inequality_l2398_239809

theorem log_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  Real.log ((Real.sqrt a + Real.sqrt b) / 2) > Real.log (Real.sqrt (a + b) / 2) := by
  sorry

end log_inequality_l2398_239809


namespace scout_troop_profit_l2398_239828

/-- The profit calculation for a scout troop selling candy bars -/
theorem scout_troop_profit :
  -- Number of candy bars
  let n : ℕ := 1500
  -- Buy price (in cents) for 3 bars
  let buy_price : ℕ := 150
  -- Sell price (in cents) for 3 bars
  let sell_price : ℕ := 200
  -- All candy bars are sold (implied in the problem)
  -- Profit calculation (in cents)
  let profit : ℚ := n * sell_price / 3 - n * buy_price / 3
  -- The theorem: profit equals 25050 cents (250.50 dollars)
  profit = 25050 := by
  sorry

end scout_troop_profit_l2398_239828


namespace salary_savings_percentage_l2398_239844

theorem salary_savings_percentage (prev_salary : ℝ) (prev_savings_rate : ℝ) 
  (h1 : prev_savings_rate > 0) 
  (h2 : prev_savings_rate < 1) : 
  let new_salary : ℝ := prev_salary * 1.1
  let new_savings_rate : ℝ := 0.1
  let new_savings : ℝ := new_salary * new_savings_rate
  let prev_savings : ℝ := prev_salary * prev_savings_rate
  new_savings = prev_savings * 1.8333333333333331 → prev_savings_rate = 0.06 := by
sorry

end salary_savings_percentage_l2398_239844


namespace libor_lucky_numbers_l2398_239811

theorem libor_lucky_numbers :
  {n : ℕ | n < 1000 ∧ 7 ∣ n^2 ∧ 8 ∣ n^2 ∧ 9 ∣ n^2 ∧ 10 ∣ n^2} = {420, 840} :=
by sorry

end libor_lucky_numbers_l2398_239811


namespace sqrt_equation_solution_l2398_239806

theorem sqrt_equation_solution (x : ℝ) : Real.sqrt (100 - x) = 9 → x = 19 := by
  sorry

end sqrt_equation_solution_l2398_239806


namespace carnation_bouquets_l2398_239833

theorem carnation_bouquets (b1 b2 b3 : ℝ) (total_bouquets : ℕ) (avg : ℝ) :
  b1 = 9.5 →
  b2 = 14.25 →
  b3 = 18.75 →
  total_bouquets = 6 →
  avg = 16 →
  ∃ b4 b5 b6 : ℝ, b4 + b5 + b6 = total_bouquets * avg - (b1 + b2 + b3) ∧
                  b4 + b5 + b6 = 53.5 :=
by sorry

end carnation_bouquets_l2398_239833


namespace candy_distribution_l2398_239864

/-- Given that Frank has a total of 16 pieces of candy and divides them equally into 2 bags,
    prove that each bag contains 8 pieces of candy. -/
theorem candy_distribution (total_candy : ℕ) (num_bags : ℕ) (candy_per_bag : ℕ) : 
  total_candy = 16 → num_bags = 2 → total_candy = num_bags * candy_per_bag → candy_per_bag = 8 := by
  sorry

end candy_distribution_l2398_239864


namespace total_books_l2398_239853

theorem total_books (keith_books jason_books megan_books : ℕ) 
  (h1 : keith_books = 20) 
  (h2 : jason_books = 21) 
  (h3 : megan_books = 15) : 
  keith_books + jason_books + megan_books = 56 := by
  sorry

end total_books_l2398_239853


namespace patty_avoids_chores_for_ten_weeks_l2398_239882

/-- Represents the cookie exchange system set up by Patty --/
structure CookieExchange where
  cookie_per_chore : ℕ
  chores_per_week : ℕ
  money_available : ℕ
  cookies_per_pack : ℕ
  cost_per_pack : ℕ

/-- Calculates the number of weeks Patty can avoid chores --/
def weeks_without_chores (ce : CookieExchange) : ℕ :=
  let packs_bought := ce.money_available / ce.cost_per_pack
  let total_cookies := packs_bought * ce.cookies_per_pack
  let cookies_per_week := ce.cookie_per_chore * ce.chores_per_week
  total_cookies / cookies_per_week

/-- Theorem stating that Patty can avoid chores for 10 weeks --/
theorem patty_avoids_chores_for_ten_weeks :
  let ce : CookieExchange := {
    cookie_per_chore := 3,
    chores_per_week := 4,
    money_available := 15,
    cookies_per_pack := 24,
    cost_per_pack := 3
  }
  weeks_without_chores ce = 10 := by
  sorry

end patty_avoids_chores_for_ten_weeks_l2398_239882


namespace line_above_function_l2398_239826

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (x + 1/a) - a*x

theorem line_above_function (a : ℝ) (h : a ≠ 0) :
  (∀ x, a*x > f a x) ↔ a > Real.exp 1 / 2 := by sorry

end line_above_function_l2398_239826


namespace prob_white_after_transfer_l2398_239877

/-- Represents a bag of balls -/
structure Bag where
  white : ℕ
  black : ℕ

/-- The probability of drawing a white ball from a bag -/
def prob_white (bag : Bag) : ℚ :=
  bag.white / (bag.white + bag.black)

theorem prob_white_after_transfer : 
  let bag_a := Bag.mk 4 6
  let bag_b := Bag.mk 4 5
  let new_bag_b := Bag.mk (bag_b.white + 1) bag_b.black
  prob_white new_bag_b = 1/2 := by
  sorry

end prob_white_after_transfer_l2398_239877


namespace domain_of_h_l2398_239823

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the domain of f
def domain_f : Set ℝ := Set.Icc (-10) 3

-- Define the function h
def h (x : ℝ) : ℝ := f (-3 * x)

-- Define the domain of h
def domain_h : Set ℝ := Set.Ici (10/3)

-- Theorem statement
theorem domain_of_h :
  ∀ x : ℝ, x ∈ domain_h ↔ -3 * x ∈ domain_f :=
sorry

end domain_of_h_l2398_239823


namespace max_b_value_l2398_239830

/-- Given a box with volume 360 cubic units and integer dimensions a, b, and c
    satisfying 1 < c < b < a, the maximum possible value of b is 12. -/
theorem max_b_value (a b c : ℕ) : 
  a * b * c = 360 → 
  1 < c → c < b → b < a → 
  b ≤ 12 ∧ ∃ (a' b' c' : ℕ), a' * b' * c' = 360 ∧ 1 < c' ∧ c' < b' ∧ b' < a' ∧ b' = 12 :=
by sorry

end max_b_value_l2398_239830


namespace shaded_area_fraction_l2398_239841

theorem shaded_area_fraction (a r : ℝ) (h1 : a = 1/4) (h2 : r = 1/16) :
  let S := a / (1 - r)
  S = 4/15 := by sorry

end shaded_area_fraction_l2398_239841
