import Mathlib

namespace wrapping_paper_area_theorem_l3869_386902

/-- The area of a square sheet of wrapping paper for a box with base side length s -/
def wrapping_paper_area (s : ℝ) : ℝ := 4 * s^2

/-- Theorem: The area of the square sheet of wrapping paper is 4s² -/
theorem wrapping_paper_area_theorem (s : ℝ) (h : s > 0) :
  wrapping_paper_area s = 4 * s^2 := by
  sorry

#check wrapping_paper_area_theorem

end wrapping_paper_area_theorem_l3869_386902


namespace emily_salary_adjustment_l3869_386972

/-- Calculates Emily's new salary after adjusting employee salaries -/
def emilysNewSalary (initialSalary: ℕ) (numEmployees: ℕ) (initialEmployeeSalary targetEmployeeSalary: ℕ) : ℕ :=
  initialSalary - numEmployees * (targetEmployeeSalary - initialEmployeeSalary)

/-- Proves that Emily's new salary is $850,000 given the initial conditions -/
theorem emily_salary_adjustment :
  emilysNewSalary 1000000 10 20000 35000 = 850000 := by
  sorry

end emily_salary_adjustment_l3869_386972


namespace mod_eight_equivalence_l3869_386948

theorem mod_eight_equivalence :
  ∃ n : ℤ, 0 ≤ n ∧ n ≤ 7 ∧ n ≡ -4850 [ZMOD 8] ∧ n = 6 := by
  sorry

end mod_eight_equivalence_l3869_386948


namespace eds_cats_l3869_386920

/-- Proves that Ed has 3 cats given the conditions of the problem -/
theorem eds_cats (dogs : ℕ) (cats : ℕ) (fish : ℕ) : 
  dogs = 2 → 
  fish = 2 * (cats + dogs) → 
  dogs + cats + fish = 15 → 
  cats = 3 := by
sorry

end eds_cats_l3869_386920


namespace inverse_function_property_l3869_386973

def invertible_function (f : ℝ → ℝ) : Prop :=
  ∃ g : ℝ → ℝ, (∀ x, g (f x) = x) ∧ (∀ y, f (g y) = y)

theorem inverse_function_property
  (f : ℝ → ℝ)
  (h_inv : invertible_function f)
  (h_point : 1 - f 1 = 2) :
  ∃ g : ℝ → ℝ, invertible_function g ∧ g = f⁻¹ ∧ g (-1) - (-1) = 2 :=
sorry

end inverse_function_property_l3869_386973


namespace cylinder_height_l3869_386999

theorem cylinder_height (perimeter : Real) (diagonal : Real) (height : Real) : 
  perimeter = 6 → diagonal = 10 → height = 8 → 
  perimeter = 2 * Real.pi * (perimeter / (2 * Real.pi)) ∧ 
  diagonal^2 = perimeter^2 + height^2 :=
by
  sorry

end cylinder_height_l3869_386999


namespace english_only_enrollment_l3869_386912

/-- The number of students enrolled only in English -/
def students_only_english (total : ℕ) (both_eng_ger : ℕ) (german : ℕ) (spanish : ℕ) : ℕ :=
  total - (german + spanish - both_eng_ger)

theorem english_only_enrollment :
  let total := 75
  let both_eng_ger := 18
  let german := 32
  let spanish := 25
  students_only_english total both_eng_ger german spanish = 18 := by
  sorry

#eval students_only_english 75 18 32 25

end english_only_enrollment_l3869_386912


namespace sequence_general_term_l3869_386939

theorem sequence_general_term (n : ℕ) :
  let S : ℕ → ℤ := λ k => 3 * k^2 - 2 * k
  let a : ℕ → ℤ := λ k => S k - S (k - 1)
  a n = 6 * n - 5 :=
by sorry

end sequence_general_term_l3869_386939


namespace friday_temperature_l3869_386963

/-- Temperatures for each day of the week -/
structure WeekTemperatures where
  monday : ℝ
  tuesday : ℝ
  wednesday : ℝ
  thursday : ℝ
  friday : ℝ

/-- The theorem stating the temperature on Friday given the conditions -/
theorem friday_temperature (temps : WeekTemperatures)
  (h1 : (temps.monday + temps.tuesday + temps.wednesday + temps.thursday) / 4 = 48)
  (h2 : (temps.tuesday + temps.wednesday + temps.thursday + temps.friday) / 4 = 46)
  (h3 : temps.monday = 41) :
  temps.friday = 33 := by
  sorry

#check friday_temperature

end friday_temperature_l3869_386963


namespace divided_triangle_area_l3869_386971

/-- Represents a triangle with parallel lines dividing its sides -/
structure DividedTriangle where
  /-- The area of the original triangle -/
  area : ℝ
  /-- The number of equal segments the sides are divided into -/
  num_segments : ℕ
  /-- The area of the largest part after division -/
  largest_part_area : ℝ

/-- Theorem stating the relationship between the area of the largest part
    and the total area of the triangle -/
theorem divided_triangle_area (t : DividedTriangle)
    (h1 : t.num_segments = 10)
    (h2 : t.largest_part_area = 38) :
    t.area = 200 := by
  sorry

end divided_triangle_area_l3869_386971


namespace article_cost_l3869_386980

theorem article_cost (decreased_price : ℝ) (decrease_percentage : ℝ) (h1 : decreased_price = 760) (h2 : decrease_percentage = 24) : 
  ∃ (original_price : ℝ), original_price * (1 - decrease_percentage / 100) = decreased_price ∧ original_price = 1000 :=
by
  sorry

end article_cost_l3869_386980


namespace problem_solution_l3869_386919

theorem problem_solution : 
  ((-5) * (-7) + 20 / (-4) = 30) ∧ 
  ((1 / 9 + 1 / 6 - 1 / 4) * (-36) = -1) := by
sorry

end problem_solution_l3869_386919


namespace geometric_sequence_sine_values_l3869_386976

theorem geometric_sequence_sine_values (α β γ : Real) :
  (β = 2 * α ∧ γ = 4 * α) →  -- geometric sequence condition
  (0 ≤ α ∧ α ≤ 2 * Real.pi) →  -- α ∈ [0, 2π]
  ((Real.sin β) / (Real.sin α) = (Real.sin γ) / (Real.sin β)) →  -- sine values form geometric sequence
  ((α = 2 * Real.pi / 3 ∧ β = 4 * Real.pi / 3 ∧ γ = 8 * Real.pi / 3) ∨
   (α = 4 * Real.pi / 3 ∧ β = 8 * Real.pi / 3 ∧ γ = 16 * Real.pi / 3)) :=
by sorry

end geometric_sequence_sine_values_l3869_386976


namespace smallest_n_congruence_l3869_386959

theorem smallest_n_congruence : ∃! n : ℕ+, n.val = 20 ∧ 
  (∀ m : ℕ+, m.val < n.val → ¬(5 * m.val ≡ 1826 [ZMOD 26])) ∧
  (5 * n.val ≡ 1826 [ZMOD 26]) :=
sorry

end smallest_n_congruence_l3869_386959


namespace probability_of_sum_25_l3869_386966

/-- Represents a die with numbered and blank faces -/
structure Die where
  faces : ℕ
  numbered_faces : ℕ
  min_number : ℕ
  max_number : ℕ

/-- The first die with 18 numbered faces (1-18) and 2 blank faces -/
def die1 : Die :=
  { faces := 20
  , numbered_faces := 18
  , min_number := 1
  , max_number := 18 }

/-- The second die with 19 numbered faces (2-20) and 1 blank face -/
def die2 : Die :=
  { faces := 20
  , numbered_faces := 19
  , min_number := 2
  , max_number := 20 }

/-- Calculates the number of ways to roll a specific sum with two dice -/
def waysToRollSum (d1 d2 : Die) (sum : ℕ) : ℕ :=
  sorry

/-- Calculates the total number of possible outcomes when rolling two dice -/
def totalOutcomes (d1 d2 : Die) : ℕ :=
  d1.faces * d2.faces

/-- The main theorem stating the probability of rolling a sum of 25 -/
theorem probability_of_sum_25 :
  (waysToRollSum die1 die2 25 : ℚ) / (totalOutcomes die1 die2 : ℚ) = 7 / 200 :=
sorry

end probability_of_sum_25_l3869_386966


namespace cindy_earnings_l3869_386992

/-- Calculates the earnings for teaching one math course in a month --/
def earnings_per_course (total_hours_per_week : ℕ) (num_courses : ℕ) (hourly_rate : ℕ) (weeks_per_month : ℕ) : ℕ :=
  (total_hours_per_week / num_courses) * weeks_per_month * hourly_rate

/-- Theorem: Cindy's earnings for one math course in a month --/
theorem cindy_earnings :
  let total_hours_per_week : ℕ := 48
  let num_courses : ℕ := 4
  let hourly_rate : ℕ := 25
  let weeks_per_month : ℕ := 4
  earnings_per_course total_hours_per_week num_courses hourly_rate weeks_per_month = 1200 := by
  sorry

#eval earnings_per_course 48 4 25 4

end cindy_earnings_l3869_386992


namespace max_value_fraction_l3869_386951

theorem max_value_fraction (x y : ℝ) (hx : -6 ≤ x ∧ x ≤ -2) (hy : 0 ≤ y ∧ y ≤ 4) :
  (x + y) / x ≤ 1/3 :=
sorry

end max_value_fraction_l3869_386951


namespace min_x_value_and_factor_sum_l3869_386954

theorem min_x_value_and_factor_sum (x y : ℕ+) (h : 3 * x^7 = 17 * y^11) :
  ∃ (a b c d : ℕ),
    x = a^c * b^d ∧
    a = 17 ∧ b = 3 ∧ c = 6 ∧ d = 4 ∧
    a + b + c + d = 30 ∧
    (∀ (x' : ℕ+), 3 * x'^7 = 17 * y^11 → x ≤ x') := by
sorry

end min_x_value_and_factor_sum_l3869_386954


namespace expression_lower_bound_l3869_386970

theorem expression_lower_bound :
  ∃ (L : ℤ), L = 3 ∧
  (∃ (S : Finset ℤ), S.card = 20 ∧
    ∀ n ∈ S, L < 4 * n + 7 ∧ 4 * n + 7 < 80) ∧
  ∀ (n : ℤ), 4 * n + 7 ≥ L :=
by sorry

end expression_lower_bound_l3869_386970


namespace min_value_of_line_through_point_l3869_386978

/-- Given a line ax + by - 1 = 0 passing through the point (1, 2),
    where a and b are positive real numbers,
    the minimum value of 1/a + 2/b is 9. -/
theorem min_value_of_line_through_point (a b : ℝ) : 
  a > 0 → b > 0 → a + 2*b = 1 → (1/a + 2/b) ≥ 9 := by sorry

end min_value_of_line_through_point_l3869_386978


namespace average_apple_weight_l3869_386964

def apple_weights : List ℝ := [120, 150, 180, 200, 220]

theorem average_apple_weight :
  (apple_weights.sum / apple_weights.length : ℝ) = 174 := by
  sorry

end average_apple_weight_l3869_386964


namespace factorial_ratio_simplification_l3869_386930

theorem factorial_ratio_simplification (N : ℕ) :
  (Nat.factorial N * (N + 2)) / Nat.factorial (N + 3) = 1 / ((N + 3) * (N + 1)) :=
by sorry

end factorial_ratio_simplification_l3869_386930


namespace product_sum_in_base_l3869_386975

/-- Represents a number in base b --/
structure BaseNumber (b : ℕ) where
  value : ℕ

/-- Converts a base b number to its decimal representation --/
def to_decimal (b : ℕ) (n : BaseNumber b) : ℕ := sorry

/-- Converts a decimal number to its representation in base b --/
def from_decimal (b : ℕ) (n : ℕ) : BaseNumber b := sorry

/-- Multiplies two numbers in base b --/
def mul_base (b : ℕ) (x y : BaseNumber b) : BaseNumber b := sorry

/-- Adds two numbers in base b --/
def add_base (b : ℕ) (x y : BaseNumber b) : BaseNumber b := sorry

theorem product_sum_in_base (b : ℕ) 
  (h : mul_base b (mul_base b (from_decimal b 14) (from_decimal b 17)) (from_decimal b 18) = from_decimal b 6180) :
  add_base b (add_base b (from_decimal b 14) (from_decimal b 17)) (from_decimal b 18) = from_decimal b 53 := by
  sorry

end product_sum_in_base_l3869_386975


namespace parallel_implies_n_eq_two_transform_implies_m_n_eq_neg_one_l3869_386917

-- Define points A and B in the Cartesian coordinate system
def A (m : ℝ) : ℝ × ℝ := (3, 2*m - 1)
def B (n : ℝ) : ℝ × ℝ := (n + 1, -1)

-- Define the condition that A and B are not coincident
def not_coincident (m n : ℝ) : Prop := A m ≠ B n

-- Define what it means for AB to be parallel to y-axis
def parallel_to_y_axis (m n : ℝ) : Prop := (A m).1 = (B n).1

-- Define the transformation of A to B
def transform_A_to_B (m n : ℝ) : Prop :=
  (A m).1 - 3 = (B n).1 ∧ (A m).2 + 2 = (B n).2

-- Theorem 1
theorem parallel_implies_n_eq_two (m n : ℝ) 
  (h1 : not_coincident m n) (h2 : parallel_to_y_axis m n) : n = 2 := by sorry

-- Theorem 2
theorem transform_implies_m_n_eq_neg_one (m n : ℝ) 
  (h1 : not_coincident m n) (h2 : transform_A_to_B m n) : m = -1 ∧ n = -1 := by sorry

end parallel_implies_n_eq_two_transform_implies_m_n_eq_neg_one_l3869_386917


namespace all_odd_rolls_probability_l3869_386984

def standard_die_odd_prob : ℚ := 1/2

def roll_count : ℕ := 8

theorem all_odd_rolls_probability :
  (standard_die_odd_prob ^ roll_count : ℚ) = 1/256 := by
  sorry

end all_odd_rolls_probability_l3869_386984


namespace congruence_problem_l3869_386940

theorem congruence_problem (x : ℤ) :
  (5 * x + 9) % 19 = 3 → (3 * x + 18) % 19 = 3 := by
  sorry

end congruence_problem_l3869_386940


namespace solution_existence_unique_solution_l3869_386916

noncomputable def has_solution (a : ℝ) : Prop :=
  ∃ x : ℝ, x > 0 ∧ x ≠ 1 ∧ 2*a - x > 0 ∧
    (Real.log x / Real.log a) / (Real.log 2 / Real.log a) +
    (Real.log (2*a - x) / Real.log x) / (Real.log 2 / Real.log x) =
    1 / (Real.log 2 / Real.log (a^2 - 1))

noncomputable def has_unique_solution (a : ℝ) : Prop :=
  ∃! x : ℝ, x > 0 ∧ x ≠ 1 ∧ 2*a - x > 0 ∧
    (Real.log x / Real.log a) / (Real.log 2 / Real.log a) +
    (Real.log (2*a - x) / Real.log x) / (Real.log 2 / Real.log x) =
    1 / (Real.log 2 / Real.log (a^2 - 1))

theorem solution_existence (a : ℝ) :
  has_solution a ↔ (a > 1 ∧ a ≠ Real.sqrt 2) :=
sorry

theorem unique_solution (a : ℝ) :
  has_unique_solution a ↔ a = 2 :=
sorry

end solution_existence_unique_solution_l3869_386916


namespace sum_two_smallest_prime_factors_of_120_l3869_386974

theorem sum_two_smallest_prime_factors_of_120 :
  ∃ (p q : Nat), Nat.Prime p ∧ Nat.Prime q ∧ p < q ∧
  p ∣ 120 ∧ q ∣ 120 ∧
  (∀ r : Nat, Nat.Prime r → r ∣ 120 → r = p ∨ r ≥ q) ∧
  p + q = 5 := by
  sorry

end sum_two_smallest_prime_factors_of_120_l3869_386974


namespace abc_sum_theorem_l3869_386983

def is_valid_digit (d : ℕ) : Prop := d > 0 ∧ d < 6

def to_base_6 (n : ℕ) : ℕ := n

theorem abc_sum_theorem (A B C : ℕ) 
  (h_distinct : A ≠ B ∧ B ≠ C ∧ A ≠ C)
  (h_valid : is_valid_digit A ∧ is_valid_digit B ∧ is_valid_digit C)
  (h_equation : to_base_6 (A * 36 + B * 6 + C) + to_base_6 (B * 6 + C) = to_base_6 (A * 36 + C * 6 + A)) :
  to_base_6 (A + B + C) = 11 :=
sorry

end abc_sum_theorem_l3869_386983


namespace rectangle_formation_ways_l3869_386932

def horizontal_lines : ℕ := 5
def vertical_lines : ℕ := 5

theorem rectangle_formation_ways : 
  (Nat.choose horizontal_lines 2) * (Nat.choose vertical_lines 2) = 100 := by
  sorry

end rectangle_formation_ways_l3869_386932


namespace cricketer_average_score_l3869_386934

theorem cricketer_average_score 
  (initial_innings : ℕ) 
  (last_inning_score : ℕ) 
  (average_increase : ℕ) 
  (h1 : initial_innings = 18) 
  (h2 : last_inning_score = 95) 
  (h3 : average_increase = 4) :
  (initial_innings * (average_increase + (last_inning_score / (initial_innings + 1))) + last_inning_score) / (initial_innings + 1) = 23 :=
by sorry

end cricketer_average_score_l3869_386934


namespace expand_and_simplify_l3869_386950

theorem expand_and_simplify (x : ℝ) : 2*x*(x-4) - (2*x-3)*(x+2) = -9*x + 6 := by
  sorry

end expand_and_simplify_l3869_386950


namespace pet_store_birds_pet_store_birds_after_changes_l3869_386900

/-- The number of birds in a pet store after sales and additions --/
theorem pet_store_birds (num_cages : ℕ) (initial_parrots : ℕ) (initial_parakeets : ℕ) (initial_canaries : ℕ)
  (sold_parrots : ℕ) (sold_canaries : ℕ) (added_parakeets : ℕ) : ℕ :=
  let total_initial_parrots := num_cages * initial_parrots
  let total_initial_parakeets := num_cages * initial_parakeets
  let total_initial_canaries := num_cages * initial_canaries
  let final_parrots := total_initial_parrots - sold_parrots
  let final_parakeets := total_initial_parakeets + added_parakeets
  let final_canaries := total_initial_canaries - sold_canaries
  final_parrots + final_parakeets + final_canaries

/-- The number of birds in the pet store after changes is 235 --/
theorem pet_store_birds_after_changes : pet_store_birds 15 3 8 5 5 2 2 = 235 := by
  sorry

end pet_store_birds_pet_store_birds_after_changes_l3869_386900


namespace base7_digit_sum_theorem_l3869_386949

/-- Converts a base-7 number to base-10 --/
def base7ToBase10 (n : ℕ) : ℕ := sorry

/-- Converts a base-10 number to base-7 --/
def base10ToBase7 (n : ℕ) : ℕ := sorry

/-- Multiplies two base-7 numbers --/
def multiplyBase7 (a b : ℕ) : ℕ := 
  base10ToBase7 (base7ToBase10 a * base7ToBase10 b)

/-- Adds two base-7 numbers --/
def addBase7 (a b : ℕ) : ℕ := 
  base10ToBase7 (base7ToBase10 a + base7ToBase10 b)

/-- Sums the digits of a base-7 number --/
def sumDigitsBase7 (n : ℕ) : ℕ := sorry

/-- The main theorem --/
theorem base7_digit_sum_theorem : 
  sumDigitsBase7 (addBase7 (multiplyBase7 36 52) 20) = 23 := by sorry

end base7_digit_sum_theorem_l3869_386949


namespace min_value_of_expression_l3869_386928

theorem min_value_of_expression (x y : ℝ) : 
  (x * y - 2)^2 + (x + y + 1)^2 ≥ 5 ∧ 
  ∃ (a b : ℝ), (a * b - 2)^2 + (a + b + 1)^2 = 5 := by
sorry

end min_value_of_expression_l3869_386928


namespace cubic_equation_solutions_l3869_386910

theorem cubic_equation_solutions :
  {x : ℝ | x^3 + (2 - x)^3 = 8} = {0, 2} := by
sorry

end cubic_equation_solutions_l3869_386910


namespace negation_of_forall_x_squared_gt_one_l3869_386931

theorem negation_of_forall_x_squared_gt_one :
  (¬ ∀ x : ℝ, x^2 > 1) ↔ (∃ x : ℝ, x^2 ≤ 1) := by sorry

end negation_of_forall_x_squared_gt_one_l3869_386931


namespace inequality_proof_l3869_386952

theorem inequality_proof (x y z : ℝ) 
  (hx : x ≠ 1) (hy : y ≠ 1) (hz : z ≠ 1) (hxyz : x * y * z = 1) : 
  (x^2 / (x - 1)^2) + (y^2 / (y - 1)^2) + (z^2 / (z - 1)^2) ≥ 1 := by
  sorry

end inequality_proof_l3869_386952


namespace hyperbola_vertex_distance_l3869_386990

/-- The equation of the hyperbola -/
def hyperbola_equation (x y : ℝ) : Prop :=
  4 * x^2 + 24 * x - 4 * y^2 + 8 * y + 44 = 0

/-- The distance between the vertices of the hyperbola -/
def vertex_distance : ℝ := 2

theorem hyperbola_vertex_distance :
  ∀ x y : ℝ, hyperbola_equation x y → vertex_distance = 2 := by sorry

end hyperbola_vertex_distance_l3869_386990


namespace probability_adjacent_knights_l3869_386945

/-- The number of knights at the round table -/
def total_knights : ℕ := 30

/-- The number of knights chosen for the quest -/
def chosen_knights : ℕ := 4

/-- The probability that at least two of the four chosen knights were sitting next to each other -/
def Q : ℚ := 389 / 437

/-- Theorem stating that Q is the correct probability -/
theorem probability_adjacent_knights : 
  Q = 1 - (total_knights - chosen_knights) * (total_knights - chosen_knights - 2) * 
        (total_knights - chosen_knights - 4) * (total_knights - chosen_knights - 6) / 
        ((total_knights - 1) * total_knights * (total_knights + 1) * (total_knights - chosen_knights + 3)) :=
by sorry

end probability_adjacent_knights_l3869_386945


namespace smallest_candy_count_l3869_386943

theorem smallest_candy_count : ∃ n : ℕ,
  (100 ≤ n ∧ n < 1000) ∧
  (n + 7) % 9 = 0 ∧
  (n - 9) % 6 = 0 ∧
  (∀ m : ℕ, 100 ≤ m ∧ m < n → (m + 7) % 9 ≠ 0 ∨ (m - 9) % 6 ≠ 0) ∧
  n = 137 := by
sorry

end smallest_candy_count_l3869_386943


namespace train_speed_problem_l3869_386938

theorem train_speed_problem (distance : ℝ) (speed_ab : ℝ) (time_difference : ℝ) :
  distance = 480 →
  speed_ab = 160 →
  time_difference = 1 →
  let time_ab := distance / speed_ab
  let time_ba := time_ab + time_difference
  let speed_ba := distance / time_ba
  speed_ba = 120 := by sorry

end train_speed_problem_l3869_386938


namespace square_side_length_l3869_386962

theorem square_side_length (d : ℝ) (h : d = 4) : 
  ∃ (s : ℝ), s * s + s * s = d * d ∧ s = 2 * Real.sqrt 2 := by
  sorry

end square_side_length_l3869_386962


namespace digit_five_minus_nine_in_book_pages_l3869_386915

/-- Counts the occurrences of a digit in a number -/
def countDigit (d : Nat) (n : Nat) : Nat :=
  sorry

/-- Counts the occurrences of a digit in a range of numbers -/
def countDigitInRange (d : Nat) (start finish : Nat) : Nat :=
  sorry

theorem digit_five_minus_nine_in_book_pages : 
  ∀ (n : Nat), n = 599 →
  (countDigitInRange 5 1 n) - (countDigitInRange 9 1 n) = 100 := by
  sorry

end digit_five_minus_nine_in_book_pages_l3869_386915


namespace power_inequality_specific_power_inequality_l3869_386960

theorem power_inequality (a : ℕ) (h : a ≥ 3) : a^(a+1) > (a+1)^a := by sorry

theorem specific_power_inequality : (2023 : ℕ)^2024 > 2024^2023 := by sorry

end power_inequality_specific_power_inequality_l3869_386960


namespace binomial_12_choose_3_l3869_386906

theorem binomial_12_choose_3 : Nat.choose 12 3 = 220 := by sorry

end binomial_12_choose_3_l3869_386906


namespace percentage_problem_l3869_386921

theorem percentage_problem (p : ℝ) : 
  (0.65 * 40 = p / 100 * 60 + 23) → p = 5 := by
  sorry

end percentage_problem_l3869_386921


namespace ott_fraction_of_total_l3869_386965

/-- Represents the amount of money each person has -/
structure Money where
  loki : ℚ
  moe : ℚ
  nick : ℚ
  ott : ℚ

/-- The initial state of money distribution -/
def initial_money : Money := {
  loki := 5,
  moe := 5,
  nick := 3,
  ott := 0
}

/-- The amount of money given to Ott -/
def money_given : ℚ := 1

/-- The state of money after giving to Ott -/
def final_money : Money := {
  loki := initial_money.loki - money_given,
  moe := initial_money.moe - money_given,
  nick := initial_money.nick - money_given,
  ott := initial_money.ott + 3 * money_given
}

/-- The theorem to prove -/
theorem ott_fraction_of_total (m : Money := final_money) :
  m.ott / (m.loki + m.moe + m.nick + m.ott) = 3 / 13 := by
  sorry

end ott_fraction_of_total_l3869_386965


namespace box_height_is_eight_inches_l3869_386961

/-- Represents the dimensions of a rectangular object -/
structure Dimensions where
  height : ℝ
  width : ℝ
  length : ℝ

/-- Calculates the volume of a rectangular object given its dimensions -/
def volume (d : Dimensions) : ℝ := d.height * d.width * d.length

theorem box_height_is_eight_inches
  (box : Dimensions)
  (block : Dimensions)
  (h1 : box.width = 10)
  (h2 : box.length = 12)
  (h3 : block.height = 3)
  (h4 : block.width = 2)
  (h5 : block.length = 4)
  (h6 : volume box = 40 * volume block) :
  box.height = 8 := by
  sorry

end box_height_is_eight_inches_l3869_386961


namespace prime_sum_problem_l3869_386935

theorem prime_sum_problem (p q r : ℕ) : 
  Nat.Prime p → Nat.Prime q → Nat.Prime r →
  p + q = r + 2 →
  1 < p →
  p < q →
  p = 2 := by
sorry

end prime_sum_problem_l3869_386935


namespace milk_powder_sampling_l3869_386987

/-- Represents a system sampling method. -/
structure SystemSampling where
  totalItems : ℕ
  sampledItems : ℕ
  firstSampledNumber : ℕ

/-- Calculates the number of the nth sampled item in a system sampling method. -/
def nthSampledNumber (s : SystemSampling) (n : ℕ) : ℕ :=
  s.firstSampledNumber + (n - 1) * (s.totalItems / s.sampledItems)

/-- Theorem stating that for the given system sampling parameters, 
    the 41st sampled item will be numbered 607. -/
theorem milk_powder_sampling :
  let s : SystemSampling := {
    totalItems := 3000,
    sampledItems := 200,
    firstSampledNumber := 7
  }
  nthSampledNumber s 41 = 607 := by
  sorry

end milk_powder_sampling_l3869_386987


namespace wilsons_theorem_l3869_386953

theorem wilsons_theorem (N : ℕ) (h : N > 1) :
  (Nat.factorial (N - 1) % N = N - 1) ↔ Nat.Prime N := by
  sorry

end wilsons_theorem_l3869_386953


namespace translation_proof_l3869_386924

/-- Represents a line in the form y = mx + b -/
structure Line where
  slope : ℝ
  yIntercept : ℝ

/-- Translates a line vertically by a given distance -/
def translateVertically (l : Line) (distance : ℝ) : Line :=
  { slope := l.slope, yIntercept := l.yIntercept + distance }

theorem translation_proof (l₁ l₂ : Line) :
  l₁.slope = 2 ∧ l₁.yIntercept = -2 ∧ l₂.slope = 2 ∧ l₂.yIntercept = 0 →
  translateVertically l₁ 2 = l₂ := by
  sorry

end translation_proof_l3869_386924


namespace equilateral_triangle_area_l3869_386986

/-- The area of an equilateral triangle with perimeter 3p is (√3/4) * p^2 -/
theorem equilateral_triangle_area (p : ℝ) (p_pos : p > 0) :
  let perimeter := 3 * p
  ∃ (area : ℝ), area = (Real.sqrt 3 / 4) * p^2 ∧
  ∀ (side : ℝ), side > 0 → 3 * side = perimeter →
  area = (Real.sqrt 3 / 4) * side^2 :=
sorry

end equilateral_triangle_area_l3869_386986


namespace max_y_value_l3869_386947

theorem max_y_value (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x * y = (x - y) / (x + 3 * y)) : 
  y ≤ 1/3 ∧ ∃ (x₀ : ℝ), x₀ > 0 ∧ x₀ * (1/3) = (x₀ - 1/3) / (x₀ + 1) := by
sorry

end max_y_value_l3869_386947


namespace equal_roots_k_value_l3869_386936

/-- The cubic equation with parameter k -/
def cubic_equation (x k : ℝ) : ℝ :=
  3 * x^3 + 9 * x^2 - 162 * x + k

/-- Theorem stating that if the cubic equation has two equal roots and k is positive, then k = 7983/125 -/
theorem equal_roots_k_value (k : ℝ) :
  (∃ a b : ℝ, a ≠ b ∧
    cubic_equation a k = 0 ∧
    cubic_equation b k = 0 ∧
    (∃ x : ℝ, x ≠ a ∧ x ≠ b ∧ cubic_equation x k = 0)) →
  k > 0 →
  k = 7983 / 125 := by
  sorry

end equal_roots_k_value_l3869_386936


namespace fraction_sum_equals_half_l3869_386914

theorem fraction_sum_equals_half : (2 / 12 : ℚ) + (4 / 24 : ℚ) + (6 / 36 : ℚ) = (1 / 2 : ℚ) := by
  sorry

end fraction_sum_equals_half_l3869_386914


namespace dennis_initial_money_l3869_386926

-- Define the sale discount
def sale_discount : ℚ := 25 / 100

-- Define the original price of the shirts
def original_price : ℚ := 125

-- Define the amount Dennis paid
def amount_paid : ℚ := 100 + 50 + 4 * 5

-- Define the change Dennis received
def change_received : ℚ := 3 * 20 + 10 + 2 * 5 + 4

-- Theorem statement
theorem dennis_initial_money :
  let discounted_price := original_price * (1 - sale_discount)
  let initial_money := discounted_price + change_received
  initial_money = 177.75 := by
  sorry

end dennis_initial_money_l3869_386926


namespace max_min_on_interval_l3869_386958

-- Define the function
def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 5

-- Define the interval
def interval : Set ℝ := Set.Icc 1 3

-- State the theorem
theorem max_min_on_interval :
  (∃ (x : ℝ), x ∈ interval ∧ ∀ (y : ℝ), y ∈ interval → f y ≤ f x) ∧
  (∃ (x : ℝ), x ∈ interval ∧ ∀ (y : ℝ), y ∈ interval → f x ≤ f y) ∧
  (∀ (x : ℝ), x ∈ interval → 1 ≤ f x ∧ f x ≤ 5) :=
sorry

end max_min_on_interval_l3869_386958


namespace product_of_imaginary_parts_l3869_386908

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the equation
def equation (z : ℂ) : Prop := z^2 + 3*z = 3 - 4*i

-- Define a function to get the imaginary part of a complex number
def imag (z : ℂ) : ℝ := z.im

-- Theorem statement
theorem product_of_imaginary_parts : 
  ∃ (z₁ z₂ : ℂ), equation z₁ ∧ equation z₂ ∧ z₁ ≠ z₂ ∧ (imag z₁ * imag z₂ = 16/25) :=
sorry

end product_of_imaginary_parts_l3869_386908


namespace prime_extension_l3869_386997

theorem prime_extension (n : ℕ) (h1 : n ≥ 2) :
  (∀ k : ℕ, 0 ≤ k ∧ k ≤ Real.sqrt (n / 3) → Nat.Prime (k^2 + k + n)) →
  (∀ k : ℕ, 0 ≤ k ∧ k ≤ n - 2 → Nat.Prime (k^2 + k + n)) := by
  sorry

end prime_extension_l3869_386997


namespace binary_sum_equality_l3869_386913

def binary_to_nat (b : List Bool) : Nat :=
  b.foldl (fun acc x => 2 * acc + if x then 1 else 0) 0

theorem binary_sum_equality : 
  let b1 := [true, true, false, true]  -- 1101₂
  let b2 := [true, false, true]        -- 101₂
  let b3 := [true, true, true, false]  -- 1110₂
  let b4 := [true, false, true, true, true]  -- 10111₂
  let b5 := [true, true, false, false, false]  -- 11000₂
  let sum := [true, true, true, false, false, false, true, false]  -- 11100010₂
  binary_to_nat b1 + binary_to_nat b2 + binary_to_nat b3 + 
  binary_to_nat b4 + binary_to_nat b5 = binary_to_nat sum := by
  sorry

#eval binary_to_nat [true, true, true, false, false, false, true, false]  -- Should output 226

end binary_sum_equality_l3869_386913


namespace complex_expression_simplification_l3869_386988

theorem complex_expression_simplification :
  Real.sqrt 2 * (Real.sqrt 6 - Real.sqrt 12) + (Real.sqrt 3 + 1)^2 + 12 / Real.sqrt 6 = 4 + 4 * Real.sqrt 3 := by
  sorry

end complex_expression_simplification_l3869_386988


namespace complement_intersection_theorem_l3869_386995

def U : Set Nat := {1,2,3,4,5,6,7,8}
def M : Set Nat := {1,3,5,7}
def N : Set Nat := {2,5,8}

theorem complement_intersection_theorem : 
  (U \ M) ∩ N = {2,8} := by sorry

end complement_intersection_theorem_l3869_386995


namespace monomial_sum_implies_expression_l3869_386955

/-- If the sum of two monomials is still a monomial, then a specific expression evaluates to -1 --/
theorem monomial_sum_implies_expression (m n : ℝ) : 
  (∃ (a : ℝ), ∃ (k : ℕ), ∃ (l : ℕ), 3 * (X : ℝ → ℝ → ℝ) k l + (-2) * (X : ℝ → ℝ → ℝ) (2*m+3) 3 = a * (X : ℝ → ℝ → ℝ) k l) →
  (4*m - n)^n = -1 := by
  sorry

/-- Helper function to represent monomials --/
def X (i j : ℕ) : ℝ → ℝ → ℝ := fun x y ↦ x^i * y^j

end monomial_sum_implies_expression_l3869_386955


namespace total_profit_is_89_10_l3869_386907

def base_price : ℚ := 12
def day1_sales : ℕ := 3
def day2_sales : ℕ := 4
def day3_sales : ℕ := 5
def day1_cost : ℚ := 4
def day2_cost : ℚ := 5
def day3_cost : ℚ := 2
def extra_money : ℚ := 7
def day3_discount : ℚ := 2
def sales_tax_rate : ℚ := 1/10

def day1_profit : ℚ := (day1_sales * base_price + extra_money - day1_sales * day1_cost) * (1 - sales_tax_rate)
def day2_profit : ℚ := (day2_sales * base_price - day2_sales * day2_cost) * (1 - sales_tax_rate)
def day3_profit : ℚ := (day3_sales * (base_price - day3_discount) - day3_sales * day3_cost) * (1 - sales_tax_rate)

theorem total_profit_is_89_10 : 
  day1_profit + day2_profit + day3_profit = 89.1 := by
  sorry

end total_profit_is_89_10_l3869_386907


namespace tetrahedron_volume_formula_l3869_386922

/-- A tetrahedron with its properties -/
structure Tetrahedron where
  S : ℝ  -- Surface area
  R : ℝ  -- Radius of inscribed sphere
  V : ℝ  -- Volume

/-- Theorem: The volume of a tetrahedron is one-third the product of its surface area and the radius of its inscribed sphere -/
theorem tetrahedron_volume_formula (t : Tetrahedron) : t.V = (1/3) * t.S * t.R := by
  sorry

end tetrahedron_volume_formula_l3869_386922


namespace total_amount_calculation_total_amount_is_3693_2_l3869_386923

/-- Calculate the total amount received after selling three items with given prices, losses, and VAT -/
theorem total_amount_calculation (price_A price_B price_C : ℝ)
                                 (loss_A loss_B loss_C : ℝ)
                                 (vat : ℝ) : ℝ :=
  let selling_price_A := price_A * (1 - loss_A)
  let selling_price_B := price_B * (1 - loss_B)
  let selling_price_C := price_C * (1 - loss_C)
  let total_selling_price := selling_price_A + selling_price_B + selling_price_C
  let total_with_vat := total_selling_price * (1 + vat)
  total_with_vat

/-- The total amount received after selling all three items, including VAT, is Rs. 3693.2 -/
theorem total_amount_is_3693_2 :
  total_amount_calculation 1300 750 1800 0.20 0.15 0.10 0.12 = 3693.2 := by
  sorry

end total_amount_calculation_total_amount_is_3693_2_l3869_386923


namespace train_crossing_time_train_crossing_time_specific_l3869_386918

/-- The time taken for a train to cross a post, given its speed and length -/
theorem train_crossing_time (speed_kmh : ℝ) (length_m : ℝ) : ℝ :=
  let speed_ms : ℝ := speed_kmh * 1000 / 3600
  length_m / speed_ms

/-- Proof that a train with speed 40 km/h and length 220.0176 m takes approximately 19.80176 seconds to cross a post -/
theorem train_crossing_time_specific :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.00001 ∧ 
  |train_crossing_time 40 220.0176 - 19.80176| < ε :=
sorry

end train_crossing_time_train_crossing_time_specific_l3869_386918


namespace expression_value_l3869_386909

theorem expression_value (x y : ℝ) (h : x^2 - 2*y = -1) : 
  3*x^2 - 6*y + 2023 = 2020 := by
sorry

end expression_value_l3869_386909


namespace value_of_M_l3869_386967

theorem value_of_M : ∀ M : ℝ, (0.25 * M = 0.55 * 1500) → M = 3300 := by
  sorry

end value_of_M_l3869_386967


namespace sqrt_77_plus_28sqrt3_l3869_386942

theorem sqrt_77_plus_28sqrt3 :
  ∃ (x y z : ℤ), 
    (∀ (k : ℕ), k > 1 → ¬ (∃ (m : ℕ), z = k^2 * m)) →
    (x + y * Real.sqrt z : ℝ) = Real.sqrt (77 + 28 * Real.sqrt 3) ∧
    x = 7 ∧ y = 2 ∧ z = 7 := by
  sorry

end sqrt_77_plus_28sqrt3_l3869_386942


namespace cookie_cost_difference_l3869_386929

theorem cookie_cost_difference (cookie_cost diane_money : ℕ) 
  (h1 : cookie_cost = 65)
  (h2 : diane_money = 27) :
  cookie_cost - diane_money = 38 := by
  sorry

end cookie_cost_difference_l3869_386929


namespace sqrt_representation_condition_l3869_386903

theorem sqrt_representation_condition (A B : ℚ) :
  (∃ x y : ℚ, ∀ (sign : Bool), 
    Real.sqrt (A + (-1)^(sign.toNat : ℕ) * Real.sqrt B) = 
    Real.sqrt x + (-1)^(sign.toNat : ℕ) * Real.sqrt y) 
  ↔ 
  ∃ k : ℚ, A^2 - B = k^2 :=
by sorry

end sqrt_representation_condition_l3869_386903


namespace arithmetic_sequence_ratio_l3869_386911

/-- An arithmetic sequence -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  seq_def : ∀ n, a (n + 1) = a n + d

/-- Sum of the first n terms of an arithmetic sequence -/
def sum_n_terms (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  n * (seq.a 1 + seq.a n) / 2

theorem arithmetic_sequence_ratio (a b : ArithmeticSequence) :
  (∀ n : ℕ, sum_n_terms a n / sum_n_terms b n = (2 * n + 2 : ℚ) / (n + 3)) →
  a.a 10 / b.a 10 = 20 / 11 := by
  sorry

end arithmetic_sequence_ratio_l3869_386911


namespace multiply_by_seven_l3869_386944

theorem multiply_by_seven (x : ℝ) (h : 8 * x = 64) : 7 * x = 56 := by
  sorry

end multiply_by_seven_l3869_386944


namespace ice_chests_filled_example_l3869_386985

/-- Given an ice machine with a total number of ice cubes and a fixed number of ice cubes per chest,
    calculate the number of ice chests that can be filled. -/
def ice_chests_filled (total_ice_cubes : ℕ) (ice_cubes_per_chest : ℕ) : ℕ :=
  total_ice_cubes / ice_cubes_per_chest

/-- Prove that with 294 ice cubes in total and 42 ice cubes per chest, 7 ice chests can be filled. -/
theorem ice_chests_filled_example : ice_chests_filled 294 42 = 7 := by
  sorry

end ice_chests_filled_example_l3869_386985


namespace binomial_probability_l3869_386991

/-- A random variable following a binomial distribution -/
structure BinomialVariable where
  n : ℕ
  p : ℝ
  h_p : 0 ≤ p ∧ p ≤ 1

/-- The expected value of a binomial distribution -/
def expectedValue (ξ : BinomialVariable) : ℝ := ξ.n * ξ.p

/-- The variance of a binomial distribution -/
def variance (ξ : BinomialVariable) : ℝ := ξ.n * ξ.p * (1 - ξ.p)

theorem binomial_probability (ξ : BinomialVariable) 
  (h_exp : expectedValue ξ = 7)
  (h_var : variance ξ = 6) : 
  ξ.p = 1/7 := by
  sorry

end binomial_probability_l3869_386991


namespace room_tiles_l3869_386904

/-- Calculates the number of tiles needed for a rectangular room with a border --/
def total_tiles (length width : ℕ) (border_width : ℕ) : ℕ :=
  let border_tiles := 2 * (length + width - 2 * border_width) * border_width
  let inner_length := length - 2 * border_width
  let inner_width := width - 2 * border_width
  let inner_tiles := (inner_length * inner_width) / 4
  border_tiles + inner_tiles

/-- Theorem stating that a 15x20 room with a 2-foot border requires 168 tiles --/
theorem room_tiles : total_tiles 20 15 2 = 168 := by
  sorry

end room_tiles_l3869_386904


namespace min_value_circle_l3869_386927

theorem min_value_circle (x y : ℝ) (h : x^2 + y^2 - 4*x + 1 = 0) :
  ∃ (min : ℝ), (∀ (a b : ℝ), a^2 + b^2 - 4*a + 1 = 0 → x^2 + y^2 ≤ a^2 + b^2) ∧ min = 7 - 4*Real.sqrt 3 := by
  sorry

end min_value_circle_l3869_386927


namespace arcsin_arctan_equation_solution_l3869_386994

theorem arcsin_arctan_equation_solution :
  ∃ x : ℝ, x ∈ Set.Icc (-1/2 : ℝ) (1/2 : ℝ) ∧ Real.arcsin x + Real.arcsin (2*x) = Real.arctan x :=
by
  sorry

end arcsin_arctan_equation_solution_l3869_386994


namespace gcd_105_88_l3869_386946

theorem gcd_105_88 : Nat.gcd 105 88 = 1 := by
  sorry

end gcd_105_88_l3869_386946


namespace parabola_equation_from_hyperbola_vertex_l3869_386979

/-- Given a hyperbola with equation x²/16 - y²/9 = 1, 
    prove that the standard equation of a parabola 
    with its focus at the right vertex of this hyperbola is y² = 16x -/
theorem parabola_equation_from_hyperbola_vertex (x y : ℝ) : 
  (x^2 / 16 - y^2 / 9 = 1) → 
  ∃ (x₀ y₀ : ℝ), 
    (x₀ > 0 ∧ y₀ = 0 ∧ x₀^2 / 16 - y₀^2 / 9 = 1) ∧ 
    (∀ (x' y' : ℝ), (y' - y₀)^2 = 16 * (x' - x₀)) :=
sorry

end parabola_equation_from_hyperbola_vertex_l3869_386979


namespace parabola_directrix_l3869_386996

/-- The equation of a parabola -/
def parabola (x y : ℝ) : Prop := y = 4 * x^2 - 3

/-- The equation of the directrix -/
def directrix (y : ℝ) : Prop := y = -49/16

/-- Theorem stating that the directrix of the given parabola is y = -49/16 -/
theorem parabola_directrix : 
  ∀ x y : ℝ, parabola x y → ∃ d : ℝ, directrix d ∧ 
  (∀ p : ℝ × ℝ, p.1 = x ∧ p.2 = y → 
    (p.2 - d) = (x^2 + (y + 3 - 1/(16:ℝ))^2) / (4 * 1/(16:ℝ))) :=
sorry

end parabola_directrix_l3869_386996


namespace arithmetic_sequence_ratio_l3869_386905

def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

def sum_of_arithmetic_sequence (a : ℕ → ℚ) (n : ℕ) : ℚ :=
  (n : ℚ) * (a 1 + a n) / 2

theorem arithmetic_sequence_ratio
  (a b : ℕ → ℚ)
  (ha : arithmetic_sequence a)
  (hb : arithmetic_sequence b)
  (h : ∀ n : ℕ+, sum_of_arithmetic_sequence a n / sum_of_arithmetic_sequence b n = (n + 1) / (2 * n - 1)) :
  (a 3 + a 7) / (b 1 + b 9) = 10 / 17 := by
  sorry

end arithmetic_sequence_ratio_l3869_386905


namespace cakes_served_total_l3869_386956

/-- The number of cakes served in a restaurant over two days. -/
def total_cakes (lunch_today dinner_today yesterday : ℕ) : ℕ :=
  lunch_today + dinner_today + yesterday

/-- Theorem stating that the total number of cakes served is 14 -/
theorem cakes_served_total :
  total_cakes 5 6 3 = 14 := by
  sorry

end cakes_served_total_l3869_386956


namespace calculate_expression_l3869_386998

/-- Proves that 8 * 9(2/5) - 3 = 72(1/5) -/
theorem calculate_expression : 8 * (9 + 2/5) - 3 = 72 + 1/5 := by
  sorry

end calculate_expression_l3869_386998


namespace cos_36_degrees_l3869_386933

theorem cos_36_degrees (x y : ℝ) : 
  x = Real.cos (36 * π / 180) →
  y = Real.cos (72 * π / 180) →
  y = 2 * x^2 - 1 →
  x = 2 * y^2 - 1 →
  x = (1 + Real.sqrt 5) / 4 := by
sorry

end cos_36_degrees_l3869_386933


namespace geometric_sequence_product_l3869_386989

/-- A geometric sequence of 5 terms -/
def GeometricSequence (a : Fin 5 → ℝ) : Prop :=
  ∀ i j k, i < j → j < k → a i * a k = a j ^ 2

theorem geometric_sequence_product (a : Fin 5 → ℝ) 
  (h_geom : GeometricSequence a)
  (h_first : a 0 = 1/2)
  (h_last : a 4 = 8) :
  a 1 * a 2 * a 3 = 8 := by
sorry

end geometric_sequence_product_l3869_386989


namespace baker_pastries_sold_l3869_386977

/-- The number of cakes sold by the baker -/
def cakes_sold : ℕ := 78

/-- The difference between pastries and cakes sold -/
def pastry_cake_difference : ℕ := 76

/-- The number of pastries sold by the baker -/
def pastries_sold : ℕ := cakes_sold + pastry_cake_difference

theorem baker_pastries_sold : pastries_sold = 154 := by
  sorry

end baker_pastries_sold_l3869_386977


namespace vector_sum_zero_parallel_necessary_not_sufficient_l3869_386982

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

def parallel (a b : V) : Prop := ∃ (k : ℝ), b = k • a

theorem vector_sum_zero_parallel_necessary_not_sufficient :
  (∀ (a b : V), a ≠ 0 ∧ b ≠ 0 → (a + b = 0 → parallel a b)) ∧
  (∃ (a b : V), a ≠ 0 ∧ b ≠ 0 ∧ parallel a b ∧ a + b ≠ 0) :=
by sorry

end vector_sum_zero_parallel_necessary_not_sufficient_l3869_386982


namespace min_first_prize_l3869_386993

/-- Represents the prize structure and constraints for a competition --/
structure PrizeStructure where
  total_fund : ℕ
  first_prize : ℕ
  second_prize : ℕ
  third_prize : ℕ
  first_winners : ℕ
  second_winners : ℕ
  third_winners : ℕ

/-- Defines the conditions for a valid prize structure --/
def is_valid_structure (p : PrizeStructure) : Prop :=
  p.total_fund = 10800 ∧
  p.first_prize = 3 * p.second_prize ∧
  p.second_prize = 3 * p.third_prize ∧
  p.third_prize * p.third_winners > p.second_prize * p.second_winners ∧
  p.second_prize * p.second_winners > p.first_prize * p.first_winners ∧
  p.first_winners + p.second_winners + p.third_winners ≤ 20 ∧
  p.first_prize * p.first_winners + p.second_prize * p.second_winners + p.third_prize * p.third_winners = p.total_fund

/-- Theorem stating the minimum first prize amount --/
theorem min_first_prize (p : PrizeStructure) (h : is_valid_structure p) : 
  p.first_prize ≥ 2700 := by
  sorry

#check min_first_prize

end min_first_prize_l3869_386993


namespace age_ratio_proof_l3869_386937

/-- Given three people a, b, and c, prove that the ratio of b's age to c's age is 2:1 -/
theorem age_ratio_proof (a b c : ℕ) : 
  a = b + 2 →  -- a is two years older than b
  ∃ k : ℕ, b = k * c →  -- b is some multiple of c's age
  a + b + c = 32 →  -- The total of the ages of a, b, and c is 32
  b = 12 →  -- b is 12 years old
  b = 2 * c  -- The ratio of b's age to c's age is 2:1
:= by sorry

end age_ratio_proof_l3869_386937


namespace sample_size_proof_l3869_386941

theorem sample_size_proof (n : ℕ) (f₁ f₂ f₃ f₄ f₅ f₆ : ℕ) : 
  f₁ + f₂ + f₃ + f₄ + f₅ + f₆ = n →
  f₁ + f₂ + f₃ = 27 →
  ∃ (k : ℕ), f₁ = 2*k ∧ f₂ = 3*k ∧ f₃ = 4*k ∧ f₄ = 6*k ∧ f₅ = 4*k ∧ f₆ = k →
  n = 60 := by
sorry

end sample_size_proof_l3869_386941


namespace quartic_inequality_l3869_386901

theorem quartic_inequality (a b : ℝ) : 
  (∃ x : ℝ, x^4 - a*x^3 + 2*x^2 - b*x + 1 = 0) → a^2 + b^2 ≥ 8 := by
  sorry

end quartic_inequality_l3869_386901


namespace exist_nonzero_superintegers_with_zero_product_l3869_386925

-- Define a super-integer as a function from ℕ to ℕ
def SuperInteger := ℕ → ℕ

-- Define a zero super-integer
def isZeroSuperInteger (x : SuperInteger) : Prop :=
  ∀ n, x n = 0

-- Define non-zero super-integer
def isNonZeroSuperInteger (x : SuperInteger) : Prop :=
  ∃ n, x n ≠ 0

-- Define the product of two super-integers
def superIntegerProduct (x y : SuperInteger) : SuperInteger :=
  fun n => (x n * y n) % (10^n)

-- Theorem statement
theorem exist_nonzero_superintegers_with_zero_product :
  ∃ (x y : SuperInteger),
    isNonZeroSuperInteger x ∧
    isNonZeroSuperInteger y ∧
    isZeroSuperInteger (superIntegerProduct x y) := by
  sorry


end exist_nonzero_superintegers_with_zero_product_l3869_386925


namespace triangle_properties_l3869_386969

-- Define the triangle ABC
structure Triangle :=
  (A B C : Real)  -- angles
  (a b c : Real)  -- sides

-- Define the main theorem
theorem triangle_properties (abc : Triangle) 
  (h1 : Real.tan abc.B + Real.tan abc.C = (2 * Real.sin abc.A) / Real.cos abc.C)
  (h2 : abc.a = abc.c + 2)
  (h3 : ∃ θ : Real, θ > π / 2 ∧ (θ = abc.A ∨ θ = abc.B ∨ θ = abc.C)) :
  abc.B = π / 3 ∧ (0 < abc.c ∧ abc.c < 2) :=
sorry

end triangle_properties_l3869_386969


namespace sqrt_200_simplification_l3869_386968

theorem sqrt_200_simplification : Real.sqrt 200 = 10 * Real.sqrt 2 := by
  sorry

end sqrt_200_simplification_l3869_386968


namespace peach_ratio_l3869_386981

/-- Proves the ratio of peaches in knapsack to one cloth bag is 1:2 --/
theorem peach_ratio (total_peaches : ℕ) (knapsack_peaches : ℕ) (num_cloth_bags : ℕ) :
  total_peaches = 5 * 12 →
  knapsack_peaches = 12 →
  num_cloth_bags = 2 →
  (total_peaches - knapsack_peaches) % num_cloth_bags = 0 →
  (knapsack_peaches : ℚ) / ((total_peaches - knapsack_peaches) / num_cloth_bags) = 1 / 2 := by
  sorry

end peach_ratio_l3869_386981


namespace corresponding_angles_not_always_equal_l3869_386957

-- Define the concept of corresponding angles
def corresponding_angles (l1 l2 t : Line) (a1 a2 : Angle) : Prop :=
  -- We don't provide a specific definition, as it's not given in the problem
  sorry

-- Define the theorem
theorem corresponding_angles_not_always_equal :
  ¬ ∀ (l1 l2 t : Line) (a1 a2 : Angle),
    corresponding_angles l1 l2 t a1 a2 → a1 = a2 :=
by
  sorry

end corresponding_angles_not_always_equal_l3869_386957
