import Mathlib

namespace shaded_area_in_square_with_semicircle_l3020_302028

theorem shaded_area_in_square_with_semicircle (d : ℝ) (h : d > 0) :
  let s := d / Real.sqrt 2
  let square_area := s^2
  let semicircle_area := π * (d/2)^2 / 2
  square_area - semicircle_area = s^2 - (π/8) * d^2 := by sorry

end shaded_area_in_square_with_semicircle_l3020_302028


namespace intersection_equality_implies_complement_union_equality_l3020_302019

universe u

theorem intersection_equality_implies_complement_union_equality
  (U : Type u) [Nonempty U]
  (A B C : Set U)
  (h_nonempty_A : A.Nonempty)
  (h_nonempty_B : B.Nonempty)
  (h_nonempty_C : C.Nonempty)
  (h_intersection : A ∩ B = A ∩ C) :
  (Aᶜ ∪ B) = (Aᶜ ∪ C) :=
by sorry

end intersection_equality_implies_complement_union_equality_l3020_302019


namespace apple_difference_is_two_l3020_302058

/-- The number of apples Jackie has -/
def jackies_apples : ℕ := 10

/-- The number of apples Adam has -/
def adams_apples : ℕ := 8

/-- The difference in apples between Jackie and Adam -/
def apple_difference : ℕ := jackies_apples - adams_apples

theorem apple_difference_is_two : apple_difference = 2 := by
  sorry

end apple_difference_is_two_l3020_302058


namespace price_decrease_percentage_l3020_302089

theorem price_decrease_percentage (original_price new_price : ℝ) 
  (h1 : original_price = 1300)
  (h2 : new_price = 988) :
  (original_price - new_price) / original_price * 100 = 24 := by
  sorry

end price_decrease_percentage_l3020_302089


namespace log_identities_l3020_302060

-- Define the logarithm function
noncomputable def log (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

-- Theorem statement
theorem log_identities (a P : ℝ) (ha : a > 0) (ha1 : a ≠ 1) :
  (log (a^2) P = (log a P) / 2) ∧
  (log (Real.sqrt a) P = 2 * log a P) ∧
  (log (1/a) P = -(log a P)) := by
  sorry

end log_identities_l3020_302060


namespace tangent_line_values_l3020_302044

/-- A line y = kx + b is tangent to two circles -/
def is_tangent_to_circles (k b : ℝ) : Prop :=
  k > 0 ∧
  ∃ (x₁ y₁ : ℝ), x₁^2 + y₁^2 = 1 ∧ y₁ = k * x₁ + b ∧
  ∃ (x₂ y₂ : ℝ), (x₂ - 4)^2 + y₂^2 = 1 ∧ y₂ = k * x₂ + b

/-- The unique values of k and b for a line tangent to both circles -/
theorem tangent_line_values :
  ∀ k b : ℝ, is_tangent_to_circles k b →
  k = Real.sqrt 3 / 3 ∧ b = -(2 * Real.sqrt 3 / 3) :=
by sorry

end tangent_line_values_l3020_302044


namespace tax_increase_proof_l3020_302084

theorem tax_increase_proof (item_cost : ℝ) (old_rate new_rate : ℝ) 
  (h1 : item_cost = 1000)
  (h2 : old_rate = 0.07)
  (h3 : new_rate = 0.075) :
  new_rate * item_cost - old_rate * item_cost = 5 :=
by sorry

end tax_increase_proof_l3020_302084


namespace alien_artifact_age_conversion_l3020_302036

/-- Converts a number from base 8 to base 10 -/
def octal_to_decimal (n : ℕ) : ℕ := sorry

/-- Converts a number from base 10 to base 16 -/
def decimal_to_hex (n : ℕ) : String := sorry

/-- Represents the age in octal -/
def age_octal : ℕ := 7231

/-- The expected result in hexadecimal -/
def expected_hex : String := "E99"

theorem alien_artifact_age_conversion :
  decimal_to_hex (octal_to_decimal age_octal) = expected_hex := by sorry

end alien_artifact_age_conversion_l3020_302036


namespace student_weights_l3020_302040

theorem student_weights (A B C D : ℕ) : 
  A < B ∧ B < C ∧ C < D →
  A + B = 45 →
  A + C = 49 →
  B + C = 54 →
  B + D = 60 →
  C + D = 64 →
  D = 35 := by
sorry

end student_weights_l3020_302040


namespace borrowed_sheets_average_l3020_302025

/-- Represents a document with pages printed on both sides of sheets. -/
structure Document where
  totalPages : Nat
  totalSheets : Nat
  pagesPerSheet : Nat
  borrowedSheets : Nat

/-- Calculates the average page number of remaining sheets after borrowing. -/
def averagePageNumber (doc : Document) : Rat :=
  let remainingSheets := doc.totalSheets - doc.borrowedSheets
  let totalPageSum := doc.totalPages * (doc.totalPages + 1) / 2
  let borrowedPagesStart := doc.borrowedSheets * doc.pagesPerSheet - (doc.pagesPerSheet - 1)
  let borrowedPagesEnd := doc.borrowedSheets * doc.pagesPerSheet
  let borrowedPageSum := (borrowedPagesStart + borrowedPagesEnd) * doc.borrowedSheets / 2
  (totalPageSum - borrowedPageSum) / remainingSheets

theorem borrowed_sheets_average (doc : Document) :
  doc.totalPages = 50 ∧
  doc.totalSheets = 25 ∧
  doc.pagesPerSheet = 2 ∧
  doc.borrowedSheets = 13 →
  averagePageNumber doc = 19 := by
  sorry

end borrowed_sheets_average_l3020_302025


namespace tangent_sum_l3020_302079

theorem tangent_sum (x y a b : Real) 
  (h1 : Real.sin (2 * x) + Real.sin (2 * y) = a)
  (h2 : Real.cos (2 * x) + Real.cos (2 * y) = b)
  (h3 : a^2 + b^2 ≤ 4)
  (h4 : a^2 + b^2 + 2*b ≠ 0) :
  Real.tan x + Real.tan y = 4 * a / (a^2 + b^2 + 2*b) :=
sorry

end tangent_sum_l3020_302079


namespace fraction_product_equals_one_l3020_302071

theorem fraction_product_equals_one : 
  (4 + 6 + 8) / (3 + 5 + 7) * (3 + 5 + 7) / (4 + 6 + 8) = 1 := by
  sorry

end fraction_product_equals_one_l3020_302071


namespace count_five_digit_even_divisible_by_five_l3020_302047

def is_even_digit (d : Nat) : Bool :=
  d % 2 = 0

def has_only_even_digits (n : Nat) : Bool :=
  ∀ d, d ∈ n.digits 10 → is_even_digit d

theorem count_five_digit_even_divisible_by_five : 
  (Finset.filter (λ n : Nat => 
    10000 ≤ n ∧ n ≤ 99999 ∧ 
    has_only_even_digits n ∧ 
    n % 5 = 0
  ) (Finset.range 100000)).card = 500 := by
  sorry

end count_five_digit_even_divisible_by_five_l3020_302047


namespace cube_volume_from_space_diagonal_l3020_302094

/-- Given a cube with space diagonal 5√3, prove its volume is 125 -/
theorem cube_volume_from_space_diagonal :
  ∀ s : ℝ,
  s > 0 →
  (s * s * s = 5 * 5 * 5) →
  (s * s + s * s + s * s = (5 * Real.sqrt 3) * (5 * Real.sqrt 3)) →
  s * s * s = 125 := by
  sorry

end cube_volume_from_space_diagonal_l3020_302094


namespace interview_pass_probability_l3020_302030

/-- Represents a job interview with three questions and three chances to answer. -/
structure JobInterview where
  num_questions : ℕ
  num_chances : ℕ
  prob_correct : ℝ

/-- The probability of passing the given job interview. -/
def pass_probability (interview : JobInterview) : ℝ :=
  interview.prob_correct +
  (1 - interview.prob_correct) * interview.prob_correct +
  (1 - interview.prob_correct) * (1 - interview.prob_correct) * interview.prob_correct

/-- Theorem stating that for the specific interview conditions, 
    the probability of passing is 0.973. -/
theorem interview_pass_probability :
  let interview : JobInterview := {
    num_questions := 3,
    num_chances := 3,
    prob_correct := 0.7
  }
  pass_probability interview = 0.973 := by
  sorry


end interview_pass_probability_l3020_302030


namespace third_term_of_geometric_sequence_l3020_302013

/-- A geometric sequence with a given product of its first five terms -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (r : ℝ), ∀ n, a (n + 1) = a n * r

theorem third_term_of_geometric_sequence (a : ℕ → ℝ) 
  (h_geometric : geometric_sequence a) 
  (h_product : a 1 * a 2 * a 3 * a 4 * a 5 = 32) : 
  a 3 = 2 := by
  sorry

end third_term_of_geometric_sequence_l3020_302013


namespace cube_equation_solution_l3020_302037

theorem cube_equation_solution (a n : ℕ) (h1 : a = 105) (h2 : a^3 = 21 * n * 45 * 49) : n = 25 := by
  sorry

end cube_equation_solution_l3020_302037


namespace pizza_cut_area_theorem_l3020_302017

/-- Represents a circular pizza -/
structure Pizza where
  area : ℝ
  radius : ℝ

/-- Represents a cut on the pizza -/
structure Cut where
  distance_from_center : ℝ

/-- Theorem: Given a circular pizza with area 4 μ² cut into 4 parts by two perpendicular
    straight cuts each at a distance of 50 cm from the center, the sum of the areas of
    two opposite pieces is equal to 1.5 μ² -/
theorem pizza_cut_area_theorem (p : Pizza) (c : Cut) :
  p.area = 4 →
  c.distance_from_center = 0.5 →
  ∃ (piece1 piece2 : ℝ), piece1 + piece2 = 1.5 ∧ 
    piece1 + piece2 = (p.area - 1) / 2 :=
by sorry

end pizza_cut_area_theorem_l3020_302017


namespace age_of_other_man_l3020_302000

theorem age_of_other_man (n : ℕ) (initial_avg : ℝ) (new_avg : ℝ) (man_age : ℕ) (women_avg : ℝ) :
  n = 8 ∧ 
  new_avg = initial_avg + 2 ∧ 
  man_age = 20 ∧ 
  women_avg = 30 → 
  ∃ x : ℕ, x = 24 ∧ 
    n * initial_avg - (man_age + x) + 2 * women_avg = n * new_avg :=
by sorry

end age_of_other_man_l3020_302000


namespace inheritance_calculation_l3020_302073

/-- The inheritance amount in dollars -/
def inheritance : ℝ := 33000

/-- The federal tax rate as a decimal -/
def federal_tax_rate : ℝ := 0.25

/-- The state tax rate as a decimal -/
def state_tax_rate : ℝ := 0.15

/-- The additional fee in dollars -/
def additional_fee : ℝ := 50

/-- The total amount paid for taxes and fee in dollars -/
def total_paid : ℝ := 12000

theorem inheritance_calculation :
  federal_tax_rate * inheritance + additional_fee +
  state_tax_rate * ((1 - federal_tax_rate) * inheritance - additional_fee) = total_paid :=
by sorry

end inheritance_calculation_l3020_302073


namespace min_buildings_correct_min_buildings_20x20_min_buildings_50x90_l3020_302045

/-- Represents a rectangular city grid -/
structure City where
  rows : ℕ
  cols : ℕ

/-- Calculates the minimum number of buildings after renovation -/
def min_buildings (city : City) : ℕ :=
  ((city.rows * city.cols + 15) / 16 : ℕ)

/-- Theorem: The minimum number of buildings after renovation is correct -/
theorem min_buildings_correct (city : City) :
  min_buildings city = ⌈(city.rows * city.cols : ℚ) / 16⌉ :=
sorry

/-- Corollary: For a 20x20 grid, the minimum number of buildings is 25 -/
theorem min_buildings_20x20 :
  min_buildings { rows := 20, cols := 20 } = 25 :=
sorry

/-- Corollary: For a 50x90 grid, the minimum number of buildings is 282 -/
theorem min_buildings_50x90 :
  min_buildings { rows := 50, cols := 90 } = 282 :=
sorry

end min_buildings_correct_min_buildings_20x20_min_buildings_50x90_l3020_302045


namespace at_least_one_even_digit_in_sum_l3020_302015

def is_17_digit (n : ℕ) : Prop := 10^16 ≤ n ∧ n < 10^17

def reverse_number (n : ℕ) : ℕ := 
  let digits := List.reverse (Nat.digits 10 n)
  digits.foldl (λ acc d => acc * 10 + d) 0

theorem at_least_one_even_digit_in_sum (M : ℕ) (hM : is_17_digit M) :
  ∃ d : ℕ, d < 10 ∧ d % 2 = 0 ∧ d ∈ Nat.digits 10 (M + reverse_number M) :=
sorry

end at_least_one_even_digit_in_sum_l3020_302015


namespace minibuses_needed_l3020_302050

theorem minibuses_needed (students : ℕ) (teacher : ℕ) (capacity : ℕ) : 
  students = 48 → teacher = 1 → capacity = 8 → 
  (students + teacher + capacity - 1) / capacity = 7 := by
  sorry

end minibuses_needed_l3020_302050


namespace angle_ratio_MBQ_ABQ_l3020_302020

-- Define the points
variable (A B C P Q M : Point)

-- Define the angles
def angle (X Y Z : Point) : ℝ := sorry

-- State the conditions
axiom BP_bisects_ABC : angle A B P = angle P B C
axiom BQ_bisects_ABC : angle A B Q = angle Q B C
axiom BM_bisects_PBQ : angle P B M = angle M B Q

-- State the theorem
theorem angle_ratio_MBQ_ABQ : 
  (angle M B Q) / (angle A B Q) = 1 / 4 := by sorry

end angle_ratio_MBQ_ABQ_l3020_302020


namespace inverse_proposition_false_l3020_302022

theorem inverse_proposition_false : ¬(∀ a b : ℝ, a + b > 0 → a > 0 ∧ b > 0) := by
  sorry

end inverse_proposition_false_l3020_302022


namespace parallel_plane_intersection_theorem_l3020_302055

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the parallel relation for planes
variable (parallel_planes : Plane → Plane → Prop)

-- Define the parallel relation for lines
variable (parallel_lines : Line → Line → Prop)

-- Define the intersection operation for planes
variable (intersect : Plane → Plane → Line)

-- Theorem statement
theorem parallel_plane_intersection_theorem
  (α β γ : Plane) (a b : Line)
  (h1 : parallel_planes α β)
  (h2 : intersect α γ = a)
  (h3 : intersect β γ = b) :
  parallel_lines a b :=
sorry

end parallel_plane_intersection_theorem_l3020_302055


namespace math_homework_percentage_l3020_302042

/-- Proves that the percentage of time spent on math homework is 30%, given the total homework time,
    time spent on science, and time spent on other subjects. -/
theorem math_homework_percentage
  (total_time : ℝ)
  (science_percentage : ℝ)
  (other_subjects_time : ℝ)
  (h1 : total_time = 150)
  (h2 : science_percentage = 0.4)
  (h3 : other_subjects_time = 45)
  : (total_time - science_percentage * total_time - other_subjects_time) / total_time = 0.3 := by
  sorry

#check math_homework_percentage

end math_homework_percentage_l3020_302042


namespace greatest_multiple_of_four_less_than_hundred_l3020_302098

theorem greatest_multiple_of_four_less_than_hundred : 
  ∀ n : ℕ, n % 4 = 0 ∧ n < 100 → n ≤ 96 :=
by sorry

end greatest_multiple_of_four_less_than_hundred_l3020_302098


namespace quadratic_root_implies_u_l3020_302083

theorem quadratic_root_implies_u (u : ℝ) : 
  (4 * (((-15 + Real.sqrt 205) / 8) ^ 2) + 15 * ((-15 + Real.sqrt 205) / 8) + u = 0) → 
  u = 5/4 := by
sorry

end quadratic_root_implies_u_l3020_302083


namespace inverse_f_at_3_l3020_302078

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + 2

-- Define the domain of f
def f_domain (x : ℝ) : Prop := -2 ≤ x ∧ x < 0

-- State the theorem
theorem inverse_f_at_3 :
  ∃ (f_inv : ℝ → ℝ), 
    (∀ x, f_domain x → f_inv (f x) = x) ∧
    (∀ y, (∃ x, f_domain x ∧ f x = y) → f (f_inv y) = y) ∧
    f_inv 3 = -1 :=
sorry

end inverse_f_at_3_l3020_302078


namespace quadratic_function_property_l3020_302032

theorem quadratic_function_property (a b : ℝ) : 
  let f := λ x : ℝ => x^2 + a*x + b
  (f 1 = 0) → (f 2 = 0) → (f (-1) = 6) := by
sorry

end quadratic_function_property_l3020_302032


namespace b_performance_conditions_l3020_302086

/-- Represents a person's shooting performance -/
structure ShootingPerformance where
  average : ℝ
  variance : ℝ

/-- The shooting performances of A, C, and D -/
def performances : List ShootingPerformance := [
  ⟨9.7, 0.25⟩,  -- A
  ⟨9.3, 0.28⟩,  -- C
  ⟨9.6, 0.27⟩   -- D
]

/-- B's performance is the best and most stable -/
def b_is_best (m n : ℝ) : Prop :=
  ∀ p ∈ performances, m > p.average ∧ n < p.variance

/-- Theorem stating the conditions for B's performance -/
theorem b_performance_conditions (m n : ℝ) 
  (h : b_is_best m n) : m > 9.7 ∧ n < 0.25 := by
  sorry

#check b_performance_conditions

end b_performance_conditions_l3020_302086


namespace socks_worn_l3020_302009

/-- Given 3 pairs of socks, if the number of pairs that can be formed from worn socks
    (where no worn socks are from the same original pair) is 6,
    then the number of socks worn is 3. -/
theorem socks_worn (total_pairs : ℕ) (formed_pairs : ℕ) (worn_socks : ℕ) :
  total_pairs = 3 →
  formed_pairs = 6 →
  worn_socks ≤ total_pairs * 2 →
  (∀ (i j : ℕ), i < worn_socks → j < worn_socks → i ≠ j →
    ∃ (p q : ℕ), p < total_pairs → q < total_pairs → p ≠ q) →
  (formed_pairs = worn_socks.choose 2) →
  worn_socks = 3 := by
sorry

end socks_worn_l3020_302009


namespace triangle_side_length_l3020_302007

/-- Given a triangle ABC with side lengths a, b, c, prove that if a = 2, b + c = 7, and cos B = -1/4, then b = 4 -/
theorem triangle_side_length (a b c : ℝ) (B : ℝ) (h1 : a = 2) (h2 : b + c = 7) (h3 : Real.cos B = -1/4) : b = 4 := by
  sorry

end triangle_side_length_l3020_302007


namespace janines_earnings_l3020_302035

/-- Represents the day of the week --/
inductive Day
| Monday
| Tuesday
| Thursday
| Saturday

/-- Calculates the pay rate for a given day --/
def payRate (d : Day) : ℚ :=
  match d with
  | Day.Monday => 4
  | Day.Tuesday => 4
  | Day.Thursday => 4
  | Day.Saturday => 5

/-- Calculates the bonus rate for a given day and hours worked --/
def bonusRate (hours : ℚ) : ℚ :=
  if hours > 2 then 1 else 0

/-- Calculates the earnings for a single day --/
def dailyEarnings (d : Day) (hours : ℚ) : ℚ :=
  hours * (payRate d + bonusRate hours)

/-- Janine's work schedule --/
def schedule : List (Day × ℚ) :=
  [(Day.Monday, 2), (Day.Tuesday, 3/2), (Day.Thursday, 7/2), (Day.Saturday, 5/2)]

/-- Theorem: Janine's total earnings for the week equal $46.5 --/
theorem janines_earnings :
  (schedule.map (fun (d, h) => dailyEarnings d h)).sum = 93/2 := by
  sorry

end janines_earnings_l3020_302035


namespace cube_root_equation_solution_l3020_302054

theorem cube_root_equation_solution :
  ∃ x : ℝ, (2 * x * (x^3)^(1/2))^(1/3) = 6 ∧ x = 108^(2/5) := by
  sorry

end cube_root_equation_solution_l3020_302054


namespace complex_modulus_l3020_302008

theorem complex_modulus (z : ℂ) : (z - 1) * I = I - 1 → Complex.abs z = Real.sqrt 5 := by
  sorry

end complex_modulus_l3020_302008


namespace cost_price_calculation_l3020_302061

theorem cost_price_calculation (selling_price : ℚ) (profit_percentage : ℚ) : 
  selling_price = 120 → profit_percentage = 25 → 
  ∃ (cost_price : ℚ), cost_price = 96 ∧ selling_price = cost_price * (1 + profit_percentage / 100) :=
by
  sorry

end cost_price_calculation_l3020_302061


namespace university_population_l3020_302012

/-- Represents the total number of students at the university -/
def total_students : ℕ := 5000

/-- Represents the sample size -/
def sample_size : ℕ := 500

/-- Represents the number of freshmen in the sample -/
def freshmen_sample : ℕ := 200

/-- Represents the number of sophomores in the sample -/
def sophomore_sample : ℕ := 100

/-- Represents the number of students in other grades -/
def other_grades : ℕ := 2000

/-- Theorem stating that given the sample size, freshmen sample, sophomore sample, 
    and number of students in other grades, the total number of students at the 
    university is 5000 -/
theorem university_population : 
  sample_size = freshmen_sample + sophomore_sample + (other_grades / 10) ∧
  total_students = freshmen_sample * 10 + sophomore_sample * 10 + other_grades :=
sorry

end university_population_l3020_302012


namespace no_solution_l3020_302097

/-- P(n) denotes the greatest prime factor of n -/
def P (n : ℕ) : ℕ := sorry

/-- The theorem states that there are no positive integers n > 1 satisfying both conditions -/
theorem no_solution :
  ¬ ∃ (n : ℕ), n > 1 ∧ (P n : ℝ) = Real.sqrt n ∧ (P (n + 60) : ℝ) = Real.sqrt (n + 60) := by
  sorry

end no_solution_l3020_302097


namespace book_pages_count_l3020_302096

/-- The number of pages read each night -/
def pages_per_night : ℕ := 12

/-- The number of nights needed to finish the book -/
def nights_to_finish : ℕ := 10

/-- The total number of pages in the book -/
def total_pages : ℕ := pages_per_night * nights_to_finish

theorem book_pages_count : total_pages = 120 := by
  sorry

end book_pages_count_l3020_302096


namespace unique_solution_absolute_value_equation_l3020_302003

theorem unique_solution_absolute_value_equation :
  ∃! x : ℝ, |x - 15| + |x - 25| = |3*x - 75| :=
by
  -- The proof goes here
  sorry

end unique_solution_absolute_value_equation_l3020_302003


namespace book_pages_equation_l3020_302046

theorem book_pages_equation (x : ℝ) : 
  x > 0 → 
  20 + (1/2) * (x - 20) + 15 = x := by
  sorry

end book_pages_equation_l3020_302046


namespace correct_log_values_l3020_302070

-- Define the logarithm function (base 10)
noncomputable def log (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define variables a, b, c
variable (a b c : ℝ)

-- Define the given correct logarithmic values
axiom log_3 : log 3 = 2*a - b
axiom log_5 : log 5 = a + c
axiom log_9 : log 9 = 4*a - 2*b
axiom log_0_27 : log 0.27 = 6*a - 3*b - 2
axiom log_8 : log 8 = 3 - 3*a - 3*c
axiom log_6 : log 6 = 1 + a - b - c

-- State the theorem to be proved
theorem correct_log_values :
  log 1.5 = 3*a - b + c - 1 ∧ log 7 = 2*b + c :=
sorry

end correct_log_values_l3020_302070


namespace parabola_line_intersection_l3020_302029

/-- The parabola P with equation y = x^2 -/
def P : Set (ℝ × ℝ) := {p | p.2 = p.1^2}

/-- The point Q -/
def Q : ℝ × ℝ := (10, 6)

/-- The line through Q with slope n -/
def line (n : ℝ) : Set (ℝ × ℝ) := {p | p.2 - Q.2 = n * (p.1 - Q.1)}

/-- The condition for non-intersection -/
def non_intersect (n : ℝ) : Prop := line n ∩ P = ∅

/-- The theorem to be proved -/
theorem parabola_line_intersection :
  ∃ (a b : ℝ), (∀ n, non_intersect n ↔ a < n ∧ n < b) → a + b = 40 := by sorry

end parabola_line_intersection_l3020_302029


namespace dress_ratio_proof_l3020_302023

/-- Proves that the ratio of Melissa's dresses to Emily's dresses is 1:2 -/
theorem dress_ratio_proof (melissa debora emily : ℕ) : 
  debora = melissa + 12 →
  emily = 16 →
  melissa + debora + emily = 44 →
  melissa = emily / 2 := by
sorry

end dress_ratio_proof_l3020_302023


namespace simplify_expression_l3020_302087

theorem simplify_expression : (2^8 + 4^5) * (2^3 - (-2)^3)^7 = 1280 * 16^7 := by
  sorry

end simplify_expression_l3020_302087


namespace tangent_line_power_l3020_302093

/-- Given a curve y = x^2 + ax + b with a tangent line at (0, b) of equation x - y + 1 = 0, prove a^b = 1 -/
theorem tangent_line_power (a b : ℝ) : 
  (∀ x, x^2 + a*x + b = 0 → x - (x^2 + a*x + b) + 1 = 0) → 
  a^b = 1 := by
  sorry

end tangent_line_power_l3020_302093


namespace absent_present_probability_l3020_302049

theorem absent_present_probability (p : ℝ) (h1 : p = 2/30) :
  let q := 1 - p
  2 * (p * q) = 28/225 := by sorry

end absent_present_probability_l3020_302049


namespace rectangle_perimeter_l3020_302065

theorem rectangle_perimeter (x y : ℝ) (h1 : x^2 + y^2 = 900) (h2 : x * y = 200) :
  2 * (x + y) = 20 * Real.sqrt 13 := by
  sorry

end rectangle_perimeter_l3020_302065


namespace fraction_zero_l3020_302005

theorem fraction_zero (x : ℝ) (h : x ≠ 3) : 
  x = 0 ↔ (2 * x^2 - 6 * x) / (x - 3) = 0 := by sorry

end fraction_zero_l3020_302005


namespace seating_arrangement_with_constraint_l3020_302033

def total_arrangements (n : ℕ) : ℕ := Nat.factorial n

def arrangements_with_pair_together (n : ℕ) : ℕ :=
  Nat.factorial (n - 1) * Nat.factorial 2

theorem seating_arrangement_with_constraint :
  total_arrangements 8 - arrangements_with_pair_together 8 = 30240 := by
  sorry

end seating_arrangement_with_constraint_l3020_302033


namespace complex_sum_as_polar_l3020_302076

open Complex

theorem complex_sum_as_polar : ∃ (r θ : ℝ),
  7 * exp (3 * π * I / 14) - 7 * exp (10 * π * I / 21) = r * exp (θ * I) ∧
  r = Real.sqrt (2 - Real.sqrt 3 / 2) ∧
  θ = 29 * π / 84 + Real.arctan (-2 / (Real.sqrt 3 - 1)) :=
by sorry

end complex_sum_as_polar_l3020_302076


namespace greatest_base8_digit_sum_l3020_302066

/-- Represents a positive integer in base 8 --/
structure Base8Int where
  digits : List Nat
  positive : digits ≠ []
  valid : ∀ d ∈ digits, d < 8

/-- Converts a Base8Int to its decimal representation --/
def toDecimal (n : Base8Int) : Nat :=
  sorry

/-- Computes the sum of digits of a Base8Int --/
def digitSum (n : Base8Int) : Nat :=
  sorry

/-- The theorem to be proved --/
theorem greatest_base8_digit_sum :
  (∃ (n : Base8Int), toDecimal n < 1728 ∧
    ∀ (m : Base8Int), toDecimal m < 1728 → digitSum m ≤ digitSum n) ∧
  (∀ (n : Base8Int), toDecimal n < 1728 → digitSum n ≤ 23) :=
sorry

end greatest_base8_digit_sum_l3020_302066


namespace smallest_a_for_quadratic_roots_l3020_302039

theorem smallest_a_for_quadratic_roots (a : ℕ) (b c : ℝ) : 
  (∃ x y : ℝ, 
    x ≠ y ∧ 
    0 < x ∧ x ≤ 1/1000 ∧ 
    0 < y ∧ y ≤ 1/1000 ∧ 
    a * x^2 + b * x + c = 0 ∧ 
    a * y^2 + b * y + c = 0) →
  a ≥ 1001000 :=
sorry

end smallest_a_for_quadratic_roots_l3020_302039


namespace sam_remaining_yellow_marbles_l3020_302014

/-- The number of yellow marbles Sam has after Joan took some -/
def remaining_yellow_marbles (initial : ℕ) (taken : ℕ) : ℕ :=
  initial - taken

/-- Proof that Sam has 61 yellow marbles after Joan took 25 -/
theorem sam_remaining_yellow_marbles :
  remaining_yellow_marbles 86 25 = 61 := by
  sorry

end sam_remaining_yellow_marbles_l3020_302014


namespace ship_supplies_l3020_302041

theorem ship_supplies (x : ℝ) : 
  x > 0 →
  (x - 2/5 * x) * (1 - 3/5) = 96 →
  x = 400 :=
by sorry

end ship_supplies_l3020_302041


namespace total_combinations_l3020_302080

/-- Represents the number of friends in Victoria's group. -/
def num_friends : ℕ := 35

/-- Represents the minimum shoe size. -/
def min_size : ℕ := 5

/-- Represents the maximum shoe size. -/
def max_size : ℕ := 15

/-- Represents the number of unique designs in the store. -/
def num_designs : ℕ := 20

/-- Represents the number of colors for each design. -/
def colors_per_design : ℕ := 4

/-- Represents the number of colors each friend needs to select. -/
def colors_to_select : ℕ := 3

/-- Calculates the number of ways to choose k items from n items. -/
def combination (n k : ℕ) : ℕ :=
  Nat.choose n k

/-- Theorem stating the total number of combinations to explore. -/
theorem total_combinations : 
  num_friends * num_designs * combination colors_per_design colors_to_select = 2800 := by
  sorry

end total_combinations_l3020_302080


namespace range_of_f_l3020_302004

def f (x : ℝ) : ℝ := x^4 + 6*x^2 + 9

theorem range_of_f :
  ∀ y : ℝ, (∃ x : ℝ, x ≥ 0 ∧ f x = y) ↔ y ≥ 9 :=
by sorry

end range_of_f_l3020_302004


namespace sum_of_solutions_l3020_302016

theorem sum_of_solutions (a b : ℝ) (ha : a > 1) (hb : b > 0) :
  let f : ℝ → ℝ := λ x => Real.sqrt (a - Real.sqrt (a + b^x))
  ∃ x : ℝ, f x = x ∧
  (∀ y : ℝ, f y = y → y ≤ x) ∧
  x = (Real.sqrt (4 * a - 3 * b) - 1) / 2 := by
sorry

end sum_of_solutions_l3020_302016


namespace margin_selling_price_relation_l3020_302031

/-- Proof of the relationship between margin, cost, and selling price -/
theorem margin_selling_price_relation (n : ℝ) (C S M : ℝ) 
  (h1 : n > 2) 
  (h2 : M = (2/n) * C) 
  (h3 : S = C + M) : 
  M = (2/(n+2)) * S := by
  sorry

end margin_selling_price_relation_l3020_302031


namespace equality_of_gcd_lcm_sets_l3020_302088

theorem equality_of_gcd_lcm_sets (a b c : ℕ) :
  ({Nat.gcd a b, Nat.gcd b c, Nat.gcd a c} : Set ℕ) =
  ({Nat.lcm a b, Nat.lcm b c, Nat.lcm a c} : Set ℕ) →
  a = b ∧ b = c := by
sorry

end equality_of_gcd_lcm_sets_l3020_302088


namespace set_operations_l3020_302099

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | -2 < x ∧ x < 3}
def B : Set ℝ := {x : ℝ | -3 < x ∧ x ≤ 3}

-- State the theorem
theorem set_operations :
  (Aᶜ : Set ℝ) = {x : ℝ | x ≥ 3 ∨ x ≤ -2} ∧
  (A ∩ B : Set ℝ) = {x : ℝ | -2 < x ∧ x < 3} ∧
  ((A ∩ B)ᶜ : Set ℝ) = {x : ℝ | x ≥ 3 ∨ x ≤ -2} ∧
  (Aᶜ ∩ B : Set ℝ) = {x : ℝ | (-3 < x ∧ x ≤ -2) ∨ x = 3} :=
by sorry

end set_operations_l3020_302099


namespace beach_house_rent_total_l3020_302011

theorem beach_house_rent_total (num_people : ℕ) (rent_per_person : ℚ) : 
  num_people = 7 → rent_per_person = 70 → num_people * rent_per_person = 490 := by
  sorry

end beach_house_rent_total_l3020_302011


namespace height_for_weight_35_l3020_302053

/-- Linear regression equation relating height to weight -/
def linear_regression (x : ℝ) : ℝ := 0.1 * x + 20

/-- Theorem stating that a person weighing 35 kg has a height of 150 cm
    according to the given linear regression equation -/
theorem height_for_weight_35 :
  ∃ x : ℝ, linear_regression x = 35 ∧ x = 150 := by
  sorry

end height_for_weight_35_l3020_302053


namespace heartsuit_five_three_l3020_302026

def heartsuit (x y : ℝ) : ℝ := 4 * x + 6 * y

theorem heartsuit_five_three : heartsuit 5 3 = 38 := by
  sorry

end heartsuit_five_three_l3020_302026


namespace right_triangle_area_l3020_302082

/-- A right triangle with hypotenuse 13 and one leg 5 has an area of 30 -/
theorem right_triangle_area (a b c : ℝ) (h1 : a = 13) (h2 : b = 5) 
  (h3 : c^2 = a^2 - b^2) (h4 : a > 0 ∧ b > 0 ∧ c > 0) : (1/2) * b * c = 30 := by
  sorry

#check right_triangle_area

end right_triangle_area_l3020_302082


namespace circle_and_line_intersection_l3020_302085

-- Define the curve
def curve (x y : ℝ) : Prop := y = x^2 - 6*x + 1

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 3)^2 + (y - 1)^2 = 9

-- Define the line
def line (x y : ℝ) (a : ℝ) : Prop := x - y + a = 0

-- Define the perpendicularity condition
def perpendicular (x1 y1 x2 y2 : ℝ) : Prop := x1 * x2 + y1 * y2 = 0

-- Main theorem
theorem circle_and_line_intersection :
  ∃ (x1 y1 x2 y2 : ℝ),
    -- The curve intersects the coordinate axes at points on circle C
    (curve 0 y1 ∧ circle_C 0 y1) ∧
    (curve x1 0 ∧ circle_C x1 0) ∧
    (curve x2 0 ∧ circle_C x2 0) ∧
    -- Circle C intersects the line at A(x1, y1) and B(x2, y2)
    (circle_C x1 y1 ∧ line x1 y1 (-1)) ∧
    (circle_C x2 y2 ∧ line x2 y2 (-1)) ∧
    -- OA ⊥ OB
    perpendicular x1 y1 x2 y2 :=
  sorry

end circle_and_line_intersection_l3020_302085


namespace total_marbles_l3020_302006

theorem total_marbles (jars : ℕ) (clay_pots : ℕ) (marbles_per_jar : ℕ) :
  jars = 16 →
  jars = 2 * clay_pots →
  marbles_per_jar = 5 →
  jars * marbles_per_jar + clay_pots * (3 * marbles_per_jar) = 200 :=
by
  sorry

end total_marbles_l3020_302006


namespace teachers_not_adjacent_arrangements_l3020_302038

/-- The number of arrangements of n distinct objects taken r at a time -/
def A (n : ℕ) (r : ℕ) : ℕ := sorry

/-- The number of students -/
def num_students : ℕ := 8

/-- The number of teachers -/
def num_teachers : ℕ := 2

/-- The number of gaps between and around students -/
def num_gaps : ℕ := num_students + 1

theorem teachers_not_adjacent_arrangements :
  (A num_students num_students) * (A num_gaps num_teachers) =
  (A num_students num_students) * (A 9 2) := by sorry

end teachers_not_adjacent_arrangements_l3020_302038


namespace line_circle_properties_l3020_302072

-- Define the line l and circle C
def line_l (m x y : ℝ) : Prop := (m + 1) * x + 2 * y - m - 3 = 0
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 4*y + 4 = 0

-- Define the center of the circle
def circle_center : ℝ × ℝ := (2, 2)

-- Theorem statement
theorem line_circle_properties (m : ℝ) :
  (∀ x y, line_l m x y → (x = 1 ∧ y = 1)) ∧
  (∃ x y, line_l m x y ∧ circle_C x y) ∧
  (∃ x y, line_l m x y ∧ 
    Real.sqrt ((x - circle_center.1)^2 + (y - circle_center.2)^2) = Real.sqrt 2) :=
by sorry

end line_circle_properties_l3020_302072


namespace sphere_volume_from_parallel_planes_l3020_302068

theorem sphere_volume_from_parallel_planes (R : ℝ) :
  R > 0 →
  ∃ (h : ℝ),
    h > 0 ∧
    h < R ∧
    (h^2 + 9^2 = R^2) ∧
    ((h + 3)^2 + 12^2 = R^2) →
    (4 / 3 * Real.pi * R^3 = 4050 * Real.pi) :=
by sorry

end sphere_volume_from_parallel_planes_l3020_302068


namespace inequality_proof_l3020_302024

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (1 / a + 1 / b ≥ 4) ∧ (a^2 + b^2 ≥ 1 / 2) := by
  sorry

end inequality_proof_l3020_302024


namespace johns_actual_marks_l3020_302092

theorem johns_actual_marks (total : ℝ) (n : ℕ) (wrong_mark : ℝ) (increase : ℝ) :
  n = 80 →
  wrong_mark = 82 →
  increase = 1/2 →
  (total + wrong_mark) / n = (total + johns_mark) / n + increase →
  johns_mark = 42 :=
by sorry

end johns_actual_marks_l3020_302092


namespace contrapositive_equivalence_l3020_302043

theorem contrapositive_equivalence :
  (∀ x : ℝ, x ≤ -3 → x < 0) ↔ (∀ x : ℝ, x ≥ 0 → x > -3) :=
by sorry

end contrapositive_equivalence_l3020_302043


namespace sales_tax_difference_l3020_302001

-- Define the original price, discount rate, and tax rates
def original_price : ℝ := 50
def discount_rate : ℝ := 0.10
def tax_rate_1 : ℝ := 0.08
def tax_rate_2 : ℝ := 0.075

-- Define the discounted price
def discounted_price : ℝ := original_price * (1 - discount_rate)

-- Define the tax difference function
def tax_difference (price : ℝ) (rate1 : ℝ) (rate2 : ℝ) : ℝ :=
  price * rate1 - price * rate2

-- Theorem statement
theorem sales_tax_difference :
  tax_difference discounted_price tax_rate_1 tax_rate_2 = 0.225 := by
  sorry

end sales_tax_difference_l3020_302001


namespace quadratic_equation_solution_l3020_302018

theorem quadratic_equation_solution :
  let a : ℝ := 1
  let b : ℝ := 5
  let c : ℝ := -1
  let x₁ : ℝ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let x₂ : ℝ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  x₁ = (-5 + Real.sqrt 29) / 2 ∧
  x₂ = (-5 - Real.sqrt 29) / 2 ∧
  a * x₁^2 + b * x₁ + c = 0 ∧
  a * x₂^2 + b * x₂ + c = 0 :=
by sorry

end quadratic_equation_solution_l3020_302018


namespace chad_pet_food_difference_l3020_302027

theorem chad_pet_food_difference :
  let cat_packages : ℕ := 6
  let dog_packages : ℕ := 2
  let cat_cans_per_package : ℕ := 9
  let dog_cans_per_package : ℕ := 3
  let total_cat_cans := cat_packages * cat_cans_per_package
  let total_dog_cans := dog_packages * dog_cans_per_package
  total_cat_cans - total_dog_cans = 48 :=
by sorry

end chad_pet_food_difference_l3020_302027


namespace expression_simplification_l3020_302069

theorem expression_simplification (x y : ℚ) (hx : x = -2) (hy : y = -1) :
  (2 * (x - 2*y) * (2*x + y) - (x + 2*y)^2 + x * (8*y - 3*x)) / (6*y) = 2 := by
  sorry

end expression_simplification_l3020_302069


namespace solution_set_inequality_l3020_302059

theorem solution_set_inequality (x : ℝ) :
  (x - 1) * |x + 2| ≥ 0 ↔ x ≥ 1 ∨ x = -2 := by sorry

end solution_set_inequality_l3020_302059


namespace candy_cost_proof_l3020_302010

/-- The cost of candy A per pound -/
def cost_candy_A : ℝ := 3.20

/-- The cost of candy B per pound -/
def cost_candy_B : ℝ := 1.70

/-- The total weight of the mixture in pounds -/
def total_weight : ℝ := 5

/-- The cost per pound of the mixture -/
def mixture_cost_per_pound : ℝ := 2

/-- The weight of candy A in the mixture -/
def weight_candy_A : ℝ := 1

/-- The weight of candy B in the mixture -/
def weight_candy_B : ℝ := total_weight - weight_candy_A

theorem candy_cost_proof :
  cost_candy_A * weight_candy_A + cost_candy_B * weight_candy_B = mixture_cost_per_pound * total_weight :=
by sorry

end candy_cost_proof_l3020_302010


namespace vinay_position_from_right_l3020_302051

/-- Represents the position of a boy in a row. -/
structure Position where
  fromLeft : Nat
  fromRight : Nat
  total : Nat
  valid : fromLeft + fromRight = total + 1

/-- Given the conditions of the problem, calculate Vinay's position. -/
def vinayPosition (totalBoys : Nat) (rajanFromLeft : Nat) (betweenRajanAndVinay : Nat) : Position :=
  { fromLeft := rajanFromLeft + betweenRajanAndVinay + 1,
    fromRight := totalBoys - (rajanFromLeft + betweenRajanAndVinay),
    total := totalBoys,
    valid := by sorry }

/-- The main theorem to be proved. -/
theorem vinay_position_from_right :
  let p := vinayPosition 24 6 8
  p.fromRight = 9 := by sorry

end vinay_position_from_right_l3020_302051


namespace sum_of_number_and_five_is_nine_l3020_302021

theorem sum_of_number_and_five_is_nine (x : ℤ) : x + 5 = 9 → x = 4 := by
  sorry

end sum_of_number_and_five_is_nine_l3020_302021


namespace total_payment_after_discounts_l3020_302048

def shirt_price : ℝ := 80
def pants_price : ℝ := 100
def shirt_discount : ℝ := 0.15
def pants_discount : ℝ := 0.10
def coupon_discount : ℝ := 0.05

theorem total_payment_after_discounts :
  let discounted_shirt := shirt_price * (1 - shirt_discount)
  let discounted_pants := pants_price * (1 - pants_discount)
  let total_before_coupon := discounted_shirt + discounted_pants
  let final_amount := total_before_coupon * (1 - coupon_discount)
  final_amount = 150.10 := by
  sorry

end total_payment_after_discounts_l3020_302048


namespace function_value_problem_l3020_302063

theorem function_value_problem (f : ℝ → ℝ) (m : ℝ) 
  (h1 : ∀ x, f x = 2^x - 5)
  (h2 : f m = 3) : 
  m = 3 := by sorry

end function_value_problem_l3020_302063


namespace right_triangle_cosine_l3020_302081

theorem right_triangle_cosine (a b c : ℝ) (h1 : a^2 + b^2 = c^2) (h2 : a = 5) (h3 : c = 13) :
  let cos_C := a / c
  cos_C = 5 / 13 := by
sorry

end right_triangle_cosine_l3020_302081


namespace min_value_expression_l3020_302067

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  y / x + 16 * x / (2 * x + y) ≥ 6 ∧
  ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ y₀ / x₀ + 16 * x₀ / (2 * x₀ + y₀) = 6 :=
by sorry

end min_value_expression_l3020_302067


namespace fish_in_pond_l3020_302052

theorem fish_in_pond (tagged_fish : ℕ) (second_catch : ℕ) (tagged_in_second : ℕ) :
  tagged_fish = 60 →
  second_catch = 50 →
  tagged_in_second = 2 →
  (tagged_in_second : ℚ) / second_catch = tagged_fish / (1500 : ℚ) :=
by
  sorry

end fish_in_pond_l3020_302052


namespace exists_nth_root_product_in_disc_l3020_302056

/-- A closed disc in the complex plane -/
structure ClosedDisc where
  center : ℂ
  radius : ℝ
  radius_nonneg : 0 ≤ radius

/-- A point is in a closed disc if its distance from the center is at most the radius -/
def in_closed_disc (z : ℂ) (D : ClosedDisc) : Prop :=
  Complex.abs (z - D.center) ≤ D.radius

/-- The main theorem -/
theorem exists_nth_root_product_in_disc (D : ClosedDisc) (n : ℕ) (h_n : 0 < n) 
    (z_list : List ℂ) (h_z_list : ∀ z ∈ z_list, in_closed_disc z D) :
    ∃ z : ℂ, in_closed_disc z D ∧ z^n = z_list.prod := by
  sorry

end exists_nth_root_product_in_disc_l3020_302056


namespace max_min_values_l3020_302095

theorem max_min_values (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + 5 * y = 20) :
  (∃ (u : ℝ), u = Real.log x / Real.log 10 + Real.log y / Real.log 10 ∧
    u ≤ 1 ∧
    ∀ (v : ℝ), v = Real.log x / Real.log 10 + Real.log y / Real.log 10 → v ≤ u) ∧
  (∃ (w : ℝ), w = 1 / x + 1 / y ∧
    w ≥ (7 + 2 * Real.sqrt 10) / 20 ∧
    ∀ (z : ℝ), z = 1 / x + 1 / y → z ≥ w) := by
  sorry

end max_min_values_l3020_302095


namespace two_zeros_implies_m_values_l3020_302074

/-- A cubic function with a parameter m -/
def f (m : ℝ) (x : ℝ) : ℝ := -x^3 + 3*x + m

/-- The statement that f has exactly two zeros -/
def has_two_zeros (m : ℝ) : Prop :=
  ∃ x y : ℝ, x ≠ y ∧ f m x = 0 ∧ f m y = 0 ∧ ∀ z : ℝ, f m z = 0 → z = x ∨ z = y

/-- The theorem stating that if f has exactly two zeros, then m = -2 or m = 2 -/
theorem two_zeros_implies_m_values (m : ℝ) :
  has_two_zeros m → m = -2 ∨ m = 2 := by
  sorry

end two_zeros_implies_m_values_l3020_302074


namespace first_player_winning_strategy_l3020_302090

/-- Represents the state of the game -/
structure GameState :=
  (dominoes : Finset Nat)
  (current_score : Nat)
  (last_move : Nat)

/-- Defines a valid move in the game -/
def valid_move (state : GameState) (move : Nat) : Prop :=
  move ∈ state.dominoes ∧ move ≠ state.last_move

/-- Defines the winning condition -/
def is_winning_state (state : GameState) : Prop :=
  state.current_score = 37 ∨ state.current_score > 37

/-- Represents a strategy for playing the game -/
def Strategy := GameState → Nat

/-- Defines a winning strategy for the first player -/
def winning_strategy (s : Strategy) : Prop :=
  ∀ (initial_state : GameState),
    initial_state.dominoes = {1, 2, 3, 4, 5} →
    initial_state.current_score = 0 →
    ∃ (final_state : GameState),
      is_winning_state final_state ∧
      (∀ (opponent_move : Nat),
        valid_move initial_state opponent_move →
        valid_move (GameState.mk 
          initial_state.dominoes
          (initial_state.current_score + opponent_move)
          opponent_move) 
        (s (GameState.mk 
          initial_state.dominoes
          (initial_state.current_score + opponent_move)
          opponent_move)))

/-- Theorem stating that there exists a winning strategy for the first player -/
theorem first_player_winning_strategy :
  ∃ (s : Strategy), winning_strategy s :=
sorry

end first_player_winning_strategy_l3020_302090


namespace train_speed_l3020_302002

/-- The speed of a train given its length, the platform length, and the time to cross the platform -/
theorem train_speed (train_length platform_length : Real) (crossing_time : Real) :
  train_length = 110 ∧ 
  platform_length = 323.36799999999994 ∧ 
  crossing_time = 30 →
  (train_length + platform_length) / crossing_time * 3.6 = 52.00416 := by
  sorry

end train_speed_l3020_302002


namespace bathroom_floor_space_l3020_302062

/-- Calculates the available floor space in an L-shaped bathroom with a pillar -/
theorem bathroom_floor_space
  (main_width : ℕ) (main_length : ℕ)
  (alcove_width : ℕ) (alcove_depth : ℕ)
  (pillar_width : ℕ) (pillar_length : ℕ)
  (tile_size : ℚ) :
  main_width = 15 →
  main_length = 25 →
  alcove_width = 10 →
  alcove_depth = 8 →
  pillar_width = 3 →
  pillar_length = 5 →
  tile_size = 1/2 →
  (main_width * main_length * tile_size^2 +
   alcove_width * alcove_depth * tile_size^2 -
   pillar_width * pillar_length * tile_size^2) = 110 :=
by sorry

end bathroom_floor_space_l3020_302062


namespace sum_of_solutions_l3020_302077

-- Define the equation
def equation (x : ℝ) : Prop := |x - 1| = 3 * |x + 3|

-- Define the set of solutions
def solution_set : Set ℝ := {x : ℝ | equation x}

-- State the theorem
theorem sum_of_solutions :
  ∃ (s : Finset ℝ), s.toSet = solution_set ∧ s.sum id = -7 :=
sorry

end sum_of_solutions_l3020_302077


namespace dans_age_problem_l3020_302057

theorem dans_age_problem (x : ℝ) : (8 + 20 : ℝ) = 7 * (8 - x) → x = 4 := by
  sorry

end dans_age_problem_l3020_302057


namespace complex_product_pure_imaginary_l3020_302034

def is_pure_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

theorem complex_product_pure_imaginary (b : ℝ) :
  is_pure_imaginary ((1 + b * Complex.I) * (2 + Complex.I)) → b = 2 := by
  sorry

end complex_product_pure_imaginary_l3020_302034


namespace triangle_proof_l3020_302075

-- Define a triangle structure
structure Triangle where
  angle1 : ℝ
  angle2 : ℝ
  angle3 : ℝ

-- Define the property of being acute
def isAcute (t : Triangle) : Prop :=
  t.angle1 < 90 ∧ t.angle2 < 90 ∧ t.angle3 < 90

-- Theorem statement
theorem triangle_proof (t : Triangle) 
  (h1 : t.angle1 = 48)
  (h2 : t.angle2 = 52)
  (h3 : t.angle1 + t.angle2 + t.angle3 = 180) : 
  t.angle3 = 80 ∧ isAcute t := by
  sorry

end triangle_proof_l3020_302075


namespace solve_exponential_equation_l3020_302091

theorem solve_exponential_equation :
  ∃ x : ℝ, (9 : ℝ)^x * (9 : ℝ)^x * (9 : ℝ)^x = (81 : ℝ)^3 ∧ x = 2 := by
  sorry

end solve_exponential_equation_l3020_302091


namespace inequality_solution_and_function_property_l3020_302064

def f (x : ℝ) : ℝ := |x + 1|

theorem inequality_solution_and_function_property :
  (∃ (S : Set ℝ), S = {x : ℝ | x ≤ -10 ∨ x ≥ 0} ∧
    ∀ x, x ∈ S ↔ f (x + 8) ≥ 10 - f x) ∧
  (∀ x y, |x| > 1 → |y| < 1 → f y < |x| * f (y / x^2)) := by
  sorry

end inequality_solution_and_function_property_l3020_302064
