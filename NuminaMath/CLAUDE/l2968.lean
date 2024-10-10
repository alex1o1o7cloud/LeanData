import Mathlib

namespace geometric_sequence_formula_l2968_296853

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_formula 
  (a : ℕ → ℝ) 
  (h_pos : ∀ n, a n > 0)
  (h_geometric : is_geometric_sequence a)
  (h_a4_a2 : a 4 = (a 2)^2)
  (h_sum : a 2 + a 4 = 5/16) :
  ∀ n : ℕ, a n = (1/2)^n :=
sorry

end geometric_sequence_formula_l2968_296853


namespace inequality_system_solution_l2968_296893

theorem inequality_system_solution (x : ℝ) : 
  (2*x - 1)/3 - (5*x + 1)/2 ≤ 1 → 
  5*x - 1 < 3*(x + 1) → 
  -1 ≤ x ∧ x < 2 := by
sorry

end inequality_system_solution_l2968_296893


namespace product_equals_zero_l2968_296840

theorem product_equals_zero (a : ℤ) (h : a = 9) : 
  (a - 10) * (a - 9) * (a - 8) * (a - 7) * (a - 6) * (a - 5) * (a - 4) * (a - 3) * (a - 2) * (a - 1) * a * (a + 1) = 0 := by
  sorry

end product_equals_zero_l2968_296840


namespace largest_two_digit_prime_factor_of_binom_300_150_l2968_296823

theorem largest_two_digit_prime_factor_of_binom_300_150 :
  ∃ (p : ℕ), p = 97 ∧ 
  Prime p ∧ 
  10 ≤ p ∧ p < 100 ∧
  p ∣ Nat.choose 300 150 ∧
  ∀ (q : ℕ), Prime q → 10 ≤ q → q < 100 → q ∣ Nat.choose 300 150 → q ≤ p :=
by sorry

end largest_two_digit_prime_factor_of_binom_300_150_l2968_296823


namespace imaginary_part_of_one_plus_i_squared_l2968_296865

theorem imaginary_part_of_one_plus_i_squared (i : ℂ) (h : i^2 = -1) :
  (Complex.im ((1 : ℂ) + i)^2) = 2 := by sorry

end imaginary_part_of_one_plus_i_squared_l2968_296865


namespace tangent_line_equation_l2968_296833

/-- The parabola C: y^2 = 4x -/
def parabola (x y : ℝ) : Prop := y^2 = 4*x

/-- The directrix of the parabola -/
def directrix : ℝ → Prop := λ x => x = -1

/-- The axis of symmetry of the parabola -/
def axis_of_symmetry : ℝ → Prop := λ y => y = 0

/-- Point P is the intersection of the directrix and the axis of symmetry -/
def point_P : ℝ × ℝ := (-1, 0)

/-- A tangent line to the parabola C -/
def tangent_line (m b : ℝ) (x y : ℝ) : Prop := y = m * x + b

theorem tangent_line_equation :
  ∃ (s : ℝ), s = 1 ∨ s = -1 ∧
  ∃ (m b : ℝ), m = s ∧ b = 1 ∧
  ∀ (x y : ℝ),
    parabola x y →
    tangent_line m b x y →
    x = point_P.1 ∧ y = point_P.2 →
    x + s * y + 1 = 0 :=
sorry

end tangent_line_equation_l2968_296833


namespace limit_rational_power_to_one_l2968_296818

theorem limit_rational_power_to_one (a : ℝ) (h : a > 0) :
  ∀ (x : ℚ → ℝ), (∀ ε > 0, ∃ N, ∀ n ≥ N, |x n| < ε) →
    ∀ ε > 0, ∃ N, ∀ n ≥ N, |a^(x n) - 1| < ε :=
by sorry

end limit_rational_power_to_one_l2968_296818


namespace unique_solution_l2968_296810

/-- Floor function: greatest integer less than or equal to x -/
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

/-- The equation to be solved -/
def equation (x : ℝ) : Prop :=
  3 * x + 5 * (floor x) - 2017 = 0

/-- The theorem stating the unique solution -/
theorem unique_solution :
  ∃! x : ℝ, equation x ∧ x = 252 + 1/3 :=
by sorry

end unique_solution_l2968_296810


namespace no_four_consecutive_integers_product_perfect_square_l2968_296804

theorem no_four_consecutive_integers_product_perfect_square :
  ∀ x : ℕ+, ∃ y : ℕ+, x * (x + 1) * (x + 2) * (x + 3) = y^2 → False :=
by sorry

end no_four_consecutive_integers_product_perfect_square_l2968_296804


namespace f_difference_equals_690_l2968_296855

/-- Given a function f(x) = x^5 + 3x^3 + 7x, prove that f(3) - f(-3) = 690 -/
theorem f_difference_equals_690 : 
  let f : ℝ → ℝ := λ x ↦ x^5 + 3*x^3 + 7*x
  f 3 - f (-3) = 690 := by sorry

end f_difference_equals_690_l2968_296855


namespace race_finish_positions_l2968_296800

/-- Represents a runner in the race -/
structure Runner where
  speed : ℝ
  position : ℝ

/-- Represents the state of the race -/
structure RaceState where
  a : Runner
  b : Runner
  c : Runner

/-- The race is 100 meters long -/
def race_length : ℝ := 100

theorem race_finish_positions (initial : RaceState) 
  (h1 : initial.a.position = race_length) 
  (h2 : initial.b.position = race_length - 5)
  (h3 : initial.c.position = race_length - 10)
  (h4 : ∀ r : Runner, r.speed > 0) :
  ∃ (final : RaceState), 
    final.b.position = race_length ∧ 
    final.c.position = race_length - (5 * 5 / 19) := by
  sorry

end race_finish_positions_l2968_296800


namespace orthogonal_centers_eq_radical_axis_l2968_296817

-- Define the circle structure
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the orthogonality condition for circles
def is_orthogonal (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x2 - x1)^2 + (y2 - y1)^2 = c1.radius^2 + c2.radius^2

-- Define the radical axis of two circles
def radical_axis (c1 c2 : Circle) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | let (x, y) := p
               let (x1, y1) := c1.center
               let (x2, y2) := c2.center
               (x - x1)^2 + (y - y1)^2 - c1.radius^2 = 
               (x - x2)^2 + (y - y2)^2 - c2.radius^2}

-- Define the set of centers of circles orthogonal to both given circles
def orthogonal_centers (c1 c2 : Circle) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ (r : ℝ), is_orthogonal (Circle.mk p r) c1 ∧
                           is_orthogonal (Circle.mk p r) c2}

-- Define the common chord of two intersecting circles
def common_chord (c1 c2 : Circle) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | let (x, y) := p
               let (x1, y1) := c1.center
               let (x2, y2) := c2.center
               (x - x1)^2 + (y - y1)^2 = c1.radius^2 ∧
               (x - x2)^2 + (y - y2)^2 = c2.radius^2}

-- Theorem statement
theorem orthogonal_centers_eq_radical_axis (c1 c2 : Circle) 
  (h : c1.center ≠ c2.center) : 
  orthogonal_centers c1 c2 = radical_axis c1 c2 \ common_chord c1 c2 :=
by sorry

end orthogonal_centers_eq_radical_axis_l2968_296817


namespace equation_solution_l2968_296816

theorem equation_solution : ∃ x : ℝ, (2 * x + 6) / (x - 3) = 4 ∧ x = 9 := by
  sorry

end equation_solution_l2968_296816


namespace father_age_three_times_l2968_296826

/-- Marika's birth year -/
def marika_birth_year : ℕ := 1996

/-- The year when Marika's father's age was five times her age -/
def reference_year : ℕ := 2006

/-- Marika's father's age is five times her age in the reference year -/
axiom father_age_five_times (y : ℕ) : y = reference_year → 
  5 * (y - marika_birth_year) = y - (marika_birth_year - 50)

/-- The year we're looking for -/
def target_year : ℕ := 2016

/-- Theorem: In the target year, Marika's father's age will be three times her age -/
theorem father_age_three_times : 
  3 * (target_year - marika_birth_year) = target_year - (marika_birth_year - 50) :=
sorry

end father_age_three_times_l2968_296826


namespace total_pet_food_is_624_ounces_l2968_296880

/-- Calculates the total weight of pet food in ounces based on given conditions --/
def total_pet_food_ounces : ℕ :=
  let cat_food_bags : ℕ := 3
  let cat_food_weight : ℕ := 3
  let dog_food_bags : ℕ := 4
  let dog_food_weight : ℕ := cat_food_weight + 2
  let bird_food_bags : ℕ := 5
  let bird_food_weight : ℕ := cat_food_weight - 1
  let ounces_per_pound : ℕ := 16
  
  let total_weight_pounds : ℕ := 
    cat_food_bags * cat_food_weight +
    dog_food_bags * dog_food_weight +
    bird_food_bags * bird_food_weight
  
  total_weight_pounds * ounces_per_pound

/-- Theorem stating that the total weight of pet food is 624 ounces --/
theorem total_pet_food_is_624_ounces : 
  total_pet_food_ounces = 624 := by
  sorry

end total_pet_food_is_624_ounces_l2968_296880


namespace equation_one_solution_l2968_296813

theorem equation_one_solution (x : ℝ) : x^2 = -4*x → x = 0 ∨ x = -4 := by
  sorry

end equation_one_solution_l2968_296813


namespace smallest_divisible_m_l2968_296850

theorem smallest_divisible_m : ∃ (m : ℕ),
  (∀ k < m, ¬(k + 9 ∣ k^3 - 90)) ∧ (m + 9 ∣ m^3 - 90) ∧ m = 12 := by
  sorry

end smallest_divisible_m_l2968_296850


namespace inequality_solution_set_not_sufficient_l2968_296839

theorem inequality_solution_set (a : ℝ) : 
  (∀ x : ℝ, x^2 - 2*a*x + a > 0) → 0 ≤ a ∧ a < 1 :=
by sorry

theorem not_sufficient (a : ℝ) :
  ∃ a : ℝ, 0 ≤ a ∧ a < 1 ∧ ∃ x : ℝ, x^2 - 2*a*x + a ≤ 0 :=
by sorry

end inequality_solution_set_not_sufficient_l2968_296839


namespace equation_solution_l2968_296829

theorem equation_solution : ∃ x : ℚ, 64 * (2 * x - 1)^3 = 27 ∧ x = 7/4 := by
  sorry

end equation_solution_l2968_296829


namespace average_equation_solution_l2968_296882

theorem average_equation_solution (a : ℝ) : 
  ((2 * a + 16) + (3 * a - 8)) / 2 = 79 → a = 30 := by
sorry

end average_equation_solution_l2968_296882


namespace number_of_valid_choices_is_84_l2968_296801

/-- Represents a digit from 1 to 9 -/
def Digit := Fin 9

/-- The number of ways to choose three different digits a, b, c from 1 to 9 such that a < b < c -/
def NumberOfValidChoices : ℕ := sorry

/-- The theorem stating that the number of valid choices is 84 -/
theorem number_of_valid_choices_is_84 : NumberOfValidChoices = 84 := by sorry

end number_of_valid_choices_is_84_l2968_296801


namespace smallest_lcm_with_gcd_five_l2968_296860

theorem smallest_lcm_with_gcd_five (k l : ℕ) : 
  1000 ≤ k ∧ k < 10000 →
  1000 ≤ l ∧ l < 10000 →
  Nat.gcd k l = 5 →
  ∀ m n : ℕ, 1000 ≤ m ∧ m < 10000 → 1000 ≤ n ∧ n < 10000 → Nat.gcd m n = 5 →
  Nat.lcm k l ≤ Nat.lcm m n →
  Nat.lcm k l = 203010 :=
by sorry

end smallest_lcm_with_gcd_five_l2968_296860


namespace students_opted_for_math_and_science_l2968_296809

/-- Given a class with the following properties:
  * There are 40 students in total.
  * 10 students did not opt for math.
  * 15 students did not opt for science.
  * 20 students did not opt for history.
  * 5 students did not opt for geography.
  * 2 students did not opt for either math or science.
  * 3 students did not opt for either math or history.
  * 4 students did not opt for either math or geography.
  * 7 students did not opt for either science or history.
  * 8 students did not opt for either science or geography.
  * 10 students did not opt for either history or geography.

  Prove that the number of students who opted for both math and science is 17. -/
theorem students_opted_for_math_and_science
  (total : ℕ) (not_math : ℕ) (not_science : ℕ) (not_history : ℕ) (not_geography : ℕ)
  (not_math_or_science : ℕ) (not_math_or_history : ℕ) (not_math_or_geography : ℕ)
  (not_science_or_history : ℕ) (not_science_or_geography : ℕ) (not_history_or_geography : ℕ)
  (h_total : total = 40)
  (h_not_math : not_math = 10)
  (h_not_science : not_science = 15)
  (h_not_history : not_history = 20)
  (h_not_geography : not_geography = 5)
  (h_not_math_or_science : not_math_or_science = 2)
  (h_not_math_or_history : not_math_or_history = 3)
  (h_not_math_or_geography : not_math_or_geography = 4)
  (h_not_science_or_history : not_science_or_history = 7)
  (h_not_science_or_geography : not_science_or_geography = 8)
  (h_not_history_or_geography : not_history_or_geography = 10) :
  (total - not_math) + (total - not_science) - (total - not_math_or_science) = 17 := by
  sorry

end students_opted_for_math_and_science_l2968_296809


namespace product_of_square_roots_of_nine_l2968_296859

theorem product_of_square_roots_of_nine (a b : ℝ) : 
  a ^ 2 = 9 ∧ b ^ 2 = 9 ∧ a ≠ b → a * b = -9 := by
  sorry

end product_of_square_roots_of_nine_l2968_296859


namespace sum_due_example_l2968_296883

/-- Given a Banker's Discount and a True Discount, calculate the sum due -/
def sum_due (BD TD : ℕ) : ℕ := TD + (BD - TD)

/-- Theorem: For a Banker's Discount of 288 and a True Discount of 240, the sum due is 288 -/
theorem sum_due_example : sum_due 288 240 = 288 := by
  sorry

end sum_due_example_l2968_296883


namespace ratio_of_a_to_c_l2968_296897

theorem ratio_of_a_to_c (a b c : ℚ) 
  (h1 : a / b = 5 / 3) 
  (h2 : b / c = 1 / 5) : 
  a / c = 1 / 3 := by
  sorry

end ratio_of_a_to_c_l2968_296897


namespace inequality_proof_l2968_296802

theorem inequality_proof (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  a + b + c + d + 8 / (a * b + b * c + c * d + d * a) ≥ 6 := by
  sorry

end inequality_proof_l2968_296802


namespace point_config_theorem_l2968_296830

/-- Given three points A, B, C on a straight line in the Cartesian coordinate system -/
structure PointConfig where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  m : ℝ
  n : ℝ
  on_line : A.1 < B.1 ∧ B.1 < C.1 ∨ C.1 < B.1 ∧ B.1 < A.1

/-- The given conditions -/
def satisfies_conditions (config : PointConfig) : Prop :=
  config.A = (-3, config.m + 1) ∧
  config.B = (config.n, 3) ∧
  config.C = (7, 4) ∧
  config.A.1 * config.B.1 + config.A.2 * config.B.2 = 0 ∧  -- OA ⟂ OB
  ∃ (G : ℝ × ℝ), (G.1 = 2/3 * config.B.1 ∧ G.2 = 2/3 * config.B.2)  -- OG = (2/3) * OB

/-- The theorem to prove -/
theorem point_config_theorem (config : PointConfig) 
  (h : satisfies_conditions config) :
  (config.m = 1 ∧ config.n = 2) ∨ (config.m = 8 ∧ config.n = 9) ∧
  (config.A.1 * config.C.1 + config.A.2 * config.C.2) / 
  (Real.sqrt (config.A.1^2 + config.A.2^2) * Real.sqrt (config.C.1^2 + config.C.2^2)) = -Real.sqrt 5 / 5 :=
by sorry

end point_config_theorem_l2968_296830


namespace final_roll_probability_l2968_296824

/-- Probability of rolling a specific number on a standard die -/
def standardProbability : ℚ := 1 / 6

/-- Probability of not rolling the same number as the previous roll -/
def differentRollProbability : ℚ := 5 / 6

/-- Probability of rolling a 6 on the 15th roll if the 14th roll was 6 -/
def specialSixProbability : ℚ := 1 / 2

/-- Number of rolls before the final roll -/
def numPreviousRolls : ℕ := 13

/-- Probability that the 14th roll is a 6 given it's different from the 13th -/
def fourteenthRollSixProbability : ℚ := 1 / 5

/-- Combined probability for the 15th roll being the last -/
def fifteenthRollProbability : ℚ := 7 / 30

theorem final_roll_probability :
  (differentRollProbability ^ numPreviousRolls) * fifteenthRollProbability =
  (5 / 6 : ℚ) ^ 13 * (7 / 30 : ℚ) := by sorry

end final_roll_probability_l2968_296824


namespace f_is_quadratic_l2968_296846

-- Define a quadratic function
def is_quadratic (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

-- Define the specific function
def f (x : ℝ) : ℝ := 2 * x^2 - 7

-- Theorem statement
theorem f_is_quadratic : is_quadratic f := by
  sorry

end f_is_quadratic_l2968_296846


namespace product_312_57_base7_units_digit_l2968_296856

theorem product_312_57_base7_units_digit : 
  (312 * 57) % 7 = 4 := by sorry

end product_312_57_base7_units_digit_l2968_296856


namespace quadratic_inequality_theorem_l2968_296815

-- Define the quadratic function
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 3 * x + 2

-- Define the solution set
def solution_set (b : ℝ) : Set ℝ := {x | x < 1 ∨ x > b}

-- Main theorem
theorem quadratic_inequality_theorem (a b : ℝ) (h : ∀ x, f a x > 0 ↔ x ∈ solution_set b) :
  a = 1 ∧ b = 2 ∧
  (∀ c : ℝ, 
    (c > 2 → {x | x^2 - (c+2)*x + 2*c < 0} = {x | 2 < x ∧ x < c}) ∧
    (c < 2 → {x | x^2 - (c+2)*x + 2*c < 0} = {x | c < x ∧ x < 2}) ∧
    (c = 2 → {x | x^2 - (c+2)*x + 2*c < 0} = ∅)) :=
by sorry

end quadratic_inequality_theorem_l2968_296815


namespace counterexample_exists_l2968_296864

theorem counterexample_exists : ∃ (a b : ℝ), a^2 > b^2 ∧ a ≤ b := by
  sorry

end counterexample_exists_l2968_296864


namespace period_of_trigonometric_function_l2968_296814

/-- The period of the function y = 3sin(x) + 4cos(x - π/6) is 2π. -/
theorem period_of_trigonometric_function :
  let f : ℝ → ℝ := λ x ↦ 3 * Real.sin x + 4 * Real.cos (x - π/6)
  ∃ T : ℝ, T > 0 ∧ ∀ x : ℝ, f (x + T) = f x ∧ ∀ S : ℝ, (0 < S ∧ S < T) → ∃ x : ℝ, f (x + S) ≠ f x ∧ T = 2 * π :=
by sorry

end period_of_trigonometric_function_l2968_296814


namespace route_down_is_twelve_miles_l2968_296842

/-- Represents the hiking trip up and down a mountain -/
structure MountainHike where
  rate_up : ℝ
  time_up : ℝ
  rate_down_factor : ℝ

/-- The length of the route down the mountain -/
def route_down_length (hike : MountainHike) : ℝ :=
  hike.rate_up * hike.rate_down_factor * hike.time_up

/-- Theorem stating that the length of the route down is 12 miles -/
theorem route_down_is_twelve_miles (hike : MountainHike) 
  (h1 : hike.rate_up = 4)
  (h2 : hike.time_up = 2)
  (h3 : hike.rate_down_factor = 1.5) : 
  route_down_length hike = 12 := by
  sorry

#eval route_down_length ⟨4, 2, 1.5⟩

end route_down_is_twelve_miles_l2968_296842


namespace geometric_progression_fourth_term_l2968_296812

theorem geometric_progression_fourth_term 
  (a : ℝ) (r : ℝ) 
  (h1 : a = 4^(1/2 : ℝ)) 
  (h2 : a * r = 4^(1/3 : ℝ)) 
  (h3 : a * r^2 = 4^(1/6 : ℝ)) : 
  a * r^3 = 1 := by
sorry

end geometric_progression_fourth_term_l2968_296812


namespace distinct_roots_sum_squares_l2968_296851

theorem distinct_roots_sum_squares (k : ℝ) (x₁ x₂ : ℝ) : 
  x₁ ≠ x₂ → 
  x₁^2 + 2*x₁ - k = 0 → 
  x₂^2 + 2*x₂ - k = 0 → 
  x₁^2 + x₂^2 - 2 > 0 :=
by
  sorry

end distinct_roots_sum_squares_l2968_296851


namespace sarahs_book_pages_l2968_296841

/-- Calculates the number of pages in each book given Sarah's reading parameters --/
theorem sarahs_book_pages
  (reading_speed : ℕ)  -- words per minute
  (reading_time : ℕ)   -- hours
  (num_books : ℕ)      -- number of books
  (words_per_page : ℕ) -- words per page
  (h1 : reading_speed = 40)
  (h2 : reading_time = 20)
  (h3 : num_books = 6)
  (h4 : words_per_page = 100)
  : (reading_speed * reading_time * 60) / (num_books * words_per_page) = 80 :=
by sorry

end sarahs_book_pages_l2968_296841


namespace maggie_spent_170_l2968_296805

/-- The total amount Maggie spent on books and magazines -/
def total_spent (num_books num_magazines book_price magazine_price : ℕ) : ℕ :=
  num_books * book_price + num_magazines * magazine_price

/-- Theorem stating that Maggie spent $170 in total -/
theorem maggie_spent_170 :
  total_spent 10 10 15 2 = 170 := by
  sorry

end maggie_spent_170_l2968_296805


namespace map_length_l2968_296891

/-- The length of a rectangular map given its area and width -/
theorem map_length (area : ℝ) (width : ℝ) (h1 : area = 10) (h2 : width = 2) :
  area / width = 5 := by
  sorry

end map_length_l2968_296891


namespace unique_a_value_l2968_296878

theorem unique_a_value (a b c : ℤ) :
  (∀ x : ℤ, (x - a) * (x - 5) + 4 = (x + b) * (x + c)) →
  a = 5 :=
by sorry

end unique_a_value_l2968_296878


namespace smallest_winning_number_l2968_296868

def game_sequence (n : ℕ) : ℕ := 16 * n + 700

theorem smallest_winning_number :
  ∃ (N : ℕ),
    N ≤ 999 ∧
    950 ≤ game_sequence N ∧
    game_sequence N ≤ 999 ∧
    ∀ (m : ℕ), m < N →
      (m ≤ 999 →
       (game_sequence m < 950 ∨ game_sequence m > 999)) ∧
    N = 16 :=
  sorry

end smallest_winning_number_l2968_296868


namespace therapy_hours_is_five_l2968_296834

/-- Represents the pricing structure and charges for therapy sessions -/
structure TherapyPricing where
  firstHourPrice : ℕ
  additionalHourPrice : ℕ
  firstPatientTotalCharge : ℕ
  threeHourCharge : ℕ

/-- Calculates the number of therapy hours for the first patient -/
def calculateTherapyHours (pricing : TherapyPricing) : ℕ :=
  sorry

/-- Theorem stating that the calculated number of therapy hours is 5 -/
theorem therapy_hours_is_five (pricing : TherapyPricing) 
  (h1 : pricing.firstHourPrice = pricing.additionalHourPrice + 30)
  (h2 : pricing.threeHourCharge = 252)
  (h3 : pricing.firstPatientTotalCharge = 400) : 
  calculateTherapyHours pricing = 5 := by
  sorry

end therapy_hours_is_five_l2968_296834


namespace finish_books_in_two_weeks_l2968_296871

/-- The number of weeks needed to finish two books given their page counts and daily reading rate -/
def weeks_to_finish (book1_pages book2_pages daily_pages : ℕ) : ℚ :=
  (book1_pages + book2_pages : ℚ) / (daily_pages * 7 : ℚ)

/-- Theorem: It takes 2 weeks to finish two books with 180 and 100 pages when reading 20 pages per day -/
theorem finish_books_in_two_weeks :
  weeks_to_finish 180 100 20 = 2 := by
  sorry

end finish_books_in_two_weeks_l2968_296871


namespace quadratic_integer_root_l2968_296885

theorem quadratic_integer_root (a : ℤ) : 
  (∃ x : ℤ, x^2 + a*x + a = 0) ↔ (a = 0 ∨ a = 4) := by
  sorry

end quadratic_integer_root_l2968_296885


namespace repetend_of_four_seventeenths_l2968_296869

def repetend (n d : ℕ) : List ℕ :=
  sorry

theorem repetend_of_four_seventeenths :
  repetend 4 17 = [2, 3, 5, 2, 9, 4] :=
sorry

end repetend_of_four_seventeenths_l2968_296869


namespace savings_ratio_l2968_296881

/-- Proves that the ratio of Megan's daily savings to Leah's daily savings is 2:1 -/
theorem savings_ratio :
  -- Josiah's savings
  let josiah_daily : ℚ := 1/4
  let josiah_days : ℕ := 24
  -- Leah's savings
  let leah_daily : ℚ := 1/2
  let leah_days : ℕ := 20
  -- Megan's savings
  let megan_days : ℕ := 12
  -- Total savings
  let total_savings : ℚ := 28
  -- Calculations
  let josiah_total : ℚ := josiah_daily * josiah_days
  let leah_total : ℚ := leah_daily * leah_days
  let megan_total : ℚ := total_savings - josiah_total - leah_total
  let megan_daily : ℚ := megan_total / megan_days
  -- Theorem
  megan_daily / leah_daily = 2 := by
  sorry

end savings_ratio_l2968_296881


namespace arithmetic_geometric_sequence_l2968_296835

/-- An arithmetic sequence with common difference 2 -/
def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∀ n, a (n + 1) = a n + 2

/-- Three terms form a geometric sequence -/
def geometric_sequence (x y z : ℚ) : Prop :=
  y * y = x * z

theorem arithmetic_geometric_sequence (a : ℕ → ℚ) 
  (h_arith : arithmetic_sequence a)
  (h_geom : geometric_sequence (a 1) (a 3) (a 4)) :
  a 6 = 2 := by
  sorry

end arithmetic_geometric_sequence_l2968_296835


namespace alcohol_concentration_proof_l2968_296890

/-- Proves that adding 3.6 liters of pure alcohol to a 6-liter solution
    containing 20% alcohol results in a solution with 50% alcohol concentration. -/
theorem alcohol_concentration_proof (initial_volume : Real) (initial_concentration : Real)
  (added_alcohol : Real) (final_concentration : Real)
  (h1 : initial_volume = 6)
  (h2 : initial_concentration = 0.2)
  (h3 : added_alcohol = 3.6)
  (h4 : final_concentration = 0.5) :
  (initial_volume * initial_concentration + added_alcohol) / (initial_volume + added_alcohol) = final_concentration :=
by sorry

end alcohol_concentration_proof_l2968_296890


namespace tan_half_product_l2968_296806

theorem tan_half_product (a b : ℝ) :
  7 * (Real.cos a + Real.cos b) + 2 * (Real.cos a * Real.cos b + 1) + 3 * Real.sin a * Real.sin b = 0 →
  Real.tan (a / 2) * Real.tan (b / 2) = -2 ∨ Real.tan (a / 2) * Real.tan (b / 2) = 2 := by
  sorry

end tan_half_product_l2968_296806


namespace club_average_age_l2968_296811

/-- Represents the average age of a group of people -/
def average_age (total_age : ℕ) (num_people : ℕ) : ℚ :=
  (total_age : ℚ) / (num_people : ℚ)

/-- Represents the total age of a group of people -/
def total_age (avg_age : ℕ) (num_people : ℕ) : ℕ :=
  avg_age * num_people

theorem club_average_age 
  (num_women : ℕ) (women_avg_age : ℕ) 
  (num_men : ℕ) (men_avg_age : ℕ) 
  (num_children : ℕ) (children_avg_age : ℕ) :
  num_women = 12 → 
  women_avg_age = 32 → 
  num_men = 18 → 
  men_avg_age = 36 → 
  num_children = 20 → 
  children_avg_age = 10 → 
  average_age 
    (total_age women_avg_age num_women + 
     total_age men_avg_age num_men + 
     total_age children_avg_age num_children)
    (num_women + num_men + num_children) = 24 := by
  sorry

end club_average_age_l2968_296811


namespace village_population_l2968_296845

theorem village_population (P : ℕ) (h : (90 : ℕ) * P = 8100 * 100) : P = 9000 := by
  sorry

end village_population_l2968_296845


namespace cubic_equation_roots_l2968_296898

theorem cubic_equation_roots (a b : ℝ) : 
  (∀ x : ℝ, x^3 + a*x^2 + b*x + 6 = 0 ↔ x = 2 ∨ x = 3 ∨ x = -1) →
  a = -4 ∧ b = 1 := by sorry

end cubic_equation_roots_l2968_296898


namespace triangle_side_inequality_l2968_296873

theorem triangle_side_inequality (a b c : ℝ) (h_positive : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) : 
  (a / (1 + a)) + (b / (1 + b)) ≥ c / (1 + c) := by
  sorry

end triangle_side_inequality_l2968_296873


namespace equal_selection_probability_l2968_296844

/-- Represents the probability of a student being selected in the survey -/
def probability_of_selection (total_students : ℕ) (students_to_select : ℕ) (students_to_eliminate : ℕ) : ℚ :=
  (students_to_select : ℚ) / (total_students : ℚ)

/-- Theorem stating that the probability of selection is equal for all students and is 50/2007 -/
theorem equal_selection_probability :
  let total_students : ℕ := 2007
  let students_to_select : ℕ := 50
  let students_to_eliminate : ℕ := 7
  probability_of_selection total_students students_to_select students_to_eliminate = 50 / 2007 := by
  sorry

#check equal_selection_probability

end equal_selection_probability_l2968_296844


namespace right_triangle_third_side_l2968_296872

theorem right_triangle_third_side (a b : ℝ) (ha : a = 6) (hb : b = 10) :
  ∃ c : ℝ, (c = 2 * Real.sqrt 34 ∨ c = 8) ∧
    (c^2 = a^2 + b^2 ∨ b^2 = a^2 + c^2 ∨ a^2 = b^2 + c^2) := by
  sorry

end right_triangle_third_side_l2968_296872


namespace initial_girls_count_l2968_296836

theorem initial_girls_count (b g : ℕ) : 
  (3 * (g - 12) = b) →
  (4 * (b - 36) = g - 12) →
  g = 25 := by
  sorry

end initial_girls_count_l2968_296836


namespace group_b_forms_triangle_group_a_not_triangle_group_c_not_triangle_group_d_not_triangle_only_group_b_forms_triangle_l2968_296899

/-- A function that checks if three numbers can form a triangle based on the triangle inequality theorem -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- Theorem stating that the group (3, 4, 6) can form a triangle -/
theorem group_b_forms_triangle :
  can_form_triangle 3 4 6 := by sorry

/-- Theorem stating that the group (3, 4, 7) cannot form a triangle -/
theorem group_a_not_triangle :
  ¬ can_form_triangle 3 4 7 := by sorry

/-- Theorem stating that the group (5, 7, 12) cannot form a triangle -/
theorem group_c_not_triangle :
  ¬ can_form_triangle 5 7 12 := by sorry

/-- Theorem stating that the group (2, 3, 6) cannot form a triangle -/
theorem group_d_not_triangle :
  ¬ can_form_triangle 2 3 6 := by sorry

/-- Main theorem stating that only group B (3, 4, 6) can form a triangle among the given groups -/
theorem only_group_b_forms_triangle :
  can_form_triangle 3 4 6 ∧
  ¬ can_form_triangle 3 4 7 ∧
  ¬ can_form_triangle 5 7 12 ∧
  ¬ can_form_triangle 2 3 6 := by sorry

end group_b_forms_triangle_group_a_not_triangle_group_c_not_triangle_group_d_not_triangle_only_group_b_forms_triangle_l2968_296899


namespace wire_length_ratio_l2968_296838

/-- The length of each wire piece used by Bonnie to construct her cube frame -/
def bonnie_wire_length : ℚ := 8

/-- The number of wire pieces used by Bonnie to construct her cube frame -/
def bonnie_wire_count : ℕ := 12

/-- The length of each wire piece used by Roark to construct unit cube frames -/
def roark_wire_length : ℚ := 2

/-- The volume of a unit cube constructed by Roark -/
def unit_cube_volume : ℚ := 1

/-- The number of edges in a cube -/
def cube_edge_count : ℕ := 12

theorem wire_length_ratio :
  let bonnie_total_length := bonnie_wire_length * bonnie_wire_count
  let bonnie_cube_volume := (bonnie_wire_length / 4) ^ 3
  let roark_unit_cube_wire_length := roark_wire_length * cube_edge_count
  let roark_cube_count := bonnie_cube_volume / unit_cube_volume
  let roark_total_length := roark_unit_cube_wire_length * roark_cube_count
  bonnie_total_length / roark_total_length = 1 / 128 := by
sorry

end wire_length_ratio_l2968_296838


namespace distribute_5_balls_4_boxes_l2968_296858

/-- The number of ways to distribute indistinguishable balls into indistinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ := sorry

/-- Theorem: There are 6 ways to distribute 5 indistinguishable balls into 4 indistinguishable boxes -/
theorem distribute_5_balls_4_boxes : distribute_balls 5 4 = 6 := by sorry

end distribute_5_balls_4_boxes_l2968_296858


namespace unique_four_digit_difference_l2968_296803

/-- Represents a four-digit number -/
structure FourDigitNumber where
  a : Nat
  b : Nat
  c : Nat
  d : Nat
  h1 : a < 10
  h2 : b < 10
  h3 : c < 10
  h4 : d < 10
  h5 : a > b
  h6 : b > c
  h7 : c > d

/-- Converts a FourDigitNumber to its decimal representation -/
def toDecimal (n : FourDigitNumber) : Nat :=
  1000 * n.a + 100 * n.b + 10 * n.c + n.d

/-- The difference between a number and its reverse -/
def difference (n : FourDigitNumber) : Int :=
  (toDecimal n) - (1000 * n.d + 100 * n.c + 10 * n.b + n.a)

theorem unique_four_digit_difference (n : FourDigitNumber) :
  0 < difference n ∧ difference n < 10000 → n.a = 7 ∧ n.b = 6 ∧ n.c = 4 ∧ n.d = 1 :=
by
  sorry

#eval toDecimal { a := 7, b := 6, c := 4, d := 1, h1 := by norm_num, h2 := by norm_num, h3 := by norm_num, h4 := by norm_num, h5 := by norm_num, h6 := by norm_num, h7 := by norm_num }

end unique_four_digit_difference_l2968_296803


namespace F_of_4_f_of_5_eq_77_l2968_296879

-- Define the function f
def f (a : ℝ) : ℝ := 2 * a - 3

-- Define the function F
def F (a b : ℝ) : ℝ := b * (a + b)

-- Theorem statement
theorem F_of_4_f_of_5_eq_77 : F 4 (f 5) = 77 := by
  sorry

end F_of_4_f_of_5_eq_77_l2968_296879


namespace pizza_slices_eaten_l2968_296849

theorem pizza_slices_eaten 
  (small_pizza_slices : ℕ) 
  (large_pizza_slices : ℕ) 
  (slices_left_per_person : ℕ) 
  (num_people : ℕ) : 
  small_pizza_slices + large_pizza_slices - (slices_left_per_person * num_people) = 
  (small_pizza_slices + large_pizza_slices) - (slices_left_per_person * num_people) :=
by
  sorry

end pizza_slices_eaten_l2968_296849


namespace semicircles_to_circle_area_ratio_l2968_296825

theorem semicircles_to_circle_area_ratio :
  ∀ r : ℝ,
  r > 0 →
  let circle_area := π * (2*r)^2
  let semicircle_area := π * r^2
  (semicircle_area / circle_area) = 1/4 :=
by
  sorry

end semicircles_to_circle_area_ratio_l2968_296825


namespace smallest_distance_between_points_on_circles_l2968_296888

theorem smallest_distance_between_points_on_circles (z w : ℂ) 
  (hz : Complex.abs (z - (2 - 4*Complex.I)) = 2)
  (hw : Complex.abs (w - (5 + 6*Complex.I)) = 4) :
  ∃ (m : ℝ), m = Real.sqrt 109 - 6 ∧ ∀ (z' w' : ℂ), 
    Complex.abs (z' - (2 - 4*Complex.I)) = 2 → 
    Complex.abs (w' - (5 + 6*Complex.I)) = 4 → 
    m ≤ Complex.abs (z' - w') :=
by sorry

end smallest_distance_between_points_on_circles_l2968_296888


namespace lengthXY_is_six_l2968_296877

/-- An isosceles triangle with given properties -/
structure IsoscelesTriangle where
  -- The area of the triangle
  area : ℝ
  -- The length of the altitude from P
  altitude : ℝ
  -- The area of the trapezoid formed by dividing line XY
  trapezoidArea : ℝ

/-- The length of XY in the given isosceles triangle -/
def lengthXY (t : IsoscelesTriangle) : ℝ :=
  sorry

/-- Theorem stating the length of XY is 6 inches for the given conditions -/
theorem lengthXY_is_six (t : IsoscelesTriangle) 
    (h1 : t.area = 180)
    (h2 : t.altitude = 30)
    (h3 : t.trapezoidArea = 135) : 
  lengthXY t = 6 := by
  sorry

end lengthXY_is_six_l2968_296877


namespace quadratic_range_on_interval_l2968_296867

/-- The range of a quadratic function on a closed interval -/
theorem quadratic_range_on_interval (a b c : ℝ) (h : a < 0) :
  let f : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c
  let vertex_x : ℝ := -b / (2 * a)
  let range : Set ℝ := Set.range (fun x ↦ f x)
  (Set.Icc 0 2).image f =
    if 0 ≤ vertex_x ∧ vertex_x ≤ 2 then
      Set.Icc (4 * a + 2 * b + c) (-b^2 / (4 * a) + c)
    else
      Set.Icc (4 * a + 2 * b + c) c := by
  sorry

end quadratic_range_on_interval_l2968_296867


namespace binary_digit_difference_l2968_296896

/-- Returns the number of digits in the base-2 representation of a natural number -/
def numDigitsBinary (n : ℕ) : ℕ :=
  if n = 0 then 1 else (Nat.log2 n).succ

/-- The difference between the number of digits in the base-2 representation of 1500
    and the number of digits in the base-2 representation of 300 is 2 -/
theorem binary_digit_difference :
  numDigitsBinary 1500 - numDigitsBinary 300 = 2 := by
  sorry

#eval numDigitsBinary 1500 - numDigitsBinary 300

end binary_digit_difference_l2968_296896


namespace largest_sides_is_eight_l2968_296870

/-- A convex polygon with exactly five obtuse interior angles -/
structure ConvexPolygon where
  n : ℕ  -- number of sides
  is_convex : Bool
  obtuse_count : ℕ
  h_convex : is_convex = true
  h_obtuse : obtuse_count = 5

/-- The largest possible number of sides for a convex polygon with exactly five obtuse interior angles -/
def largest_sides : ℕ := 8

/-- Theorem stating that the largest possible number of sides for a convex polygon 
    with exactly five obtuse interior angles is 8 -/
theorem largest_sides_is_eight (p : ConvexPolygon) : 
  p.n ≤ largest_sides ∧ 
  ∃ (q : ConvexPolygon), q.n = largest_sides :=
sorry

end largest_sides_is_eight_l2968_296870


namespace black_stones_count_l2968_296886

theorem black_stones_count (total : ℕ) (difference : ℕ) (black : ℕ) : 
  total = 950 → 
  difference = 150 → 
  total = black + (black + difference) → 
  black = 400 := by
sorry

end black_stones_count_l2968_296886


namespace probability_one_tail_given_at_least_one_head_l2968_296857

def fair_coin_toss (n : ℕ) := 1 / 2 ^ n

def probability_at_least_one_head := 1 - fair_coin_toss 3

def probability_exactly_one_tail := 3 * (1 / 2) * (1 / 2)^2

theorem probability_one_tail_given_at_least_one_head :
  probability_exactly_one_tail / probability_at_least_one_head = 3 / 7 := by
  sorry

end probability_one_tail_given_at_least_one_head_l2968_296857


namespace complex_simplification_l2968_296828

/-- The imaginary unit -/
def i : ℂ := Complex.I

/-- Simplification of a complex expression -/
theorem complex_simplification : 7 * (4 - 2*i) + 4*i * (7 - 2*i) = 36 + 14*i := by sorry

end complex_simplification_l2968_296828


namespace percent_relation_l2968_296894

theorem percent_relation (a b : ℝ) (h : a = 1.8 * b) : 
  (4 * b) / a * 100 = 222.22 := by
  sorry

end percent_relation_l2968_296894


namespace miranda_monthly_savings_l2968_296854

/-- Calculates the monthly savings given total cost, sister's contribution, and number of months saved. -/
def monthlySavings (totalCost : ℚ) (sisterContribution : ℚ) (monthsSaved : ℕ) : ℚ :=
  (totalCost - sisterContribution) / monthsSaved

/-- Proves that Miranda's monthly savings for the heels is $70. -/
theorem miranda_monthly_savings :
  let totalCost : ℚ := 260
  let sisterContribution : ℚ := 50
  let monthsSaved : ℕ := 3
  monthlySavings totalCost sisterContribution monthsSaved = 70 := by
sorry

end miranda_monthly_savings_l2968_296854


namespace bug_probability_after_10_moves_l2968_296827

/-- Probability of the bug being at the starting vertex after n moves -/
def P : ℕ → ℚ
  | 0 => 1
  | n + 1 => 1/2 * (1 - P n)

/-- The probability of the bug being at the starting vertex after 10 moves is 171/512 -/
theorem bug_probability_after_10_moves : P 10 = 171 / 512 := by
  sorry

end bug_probability_after_10_moves_l2968_296827


namespace correct_number_placement_l2968_296822

-- Define the grid
inductive Square
| A | B | C | D | E | F | G | One | Nine

-- Define the arrow directions
inductive Direction
| Right | Down | Left | Up

-- Function to get the number in a square
def number_in_square (s : Square) : ℕ :=
  match s with
  | Square.A => 6
  | Square.B => 2
  | Square.C => 4
  | Square.D => 5
  | Square.E => 3
  | Square.F => 8
  | Square.G => 7
  | Square.One => 1
  | Square.Nine => 9

-- Function to get the directions of arrows in a square
def arrows_in_square (s : Square) : List Direction :=
  match s with
  | Square.One => [Direction.Right, Direction.Down]
  | Square.B => [Direction.Right, Direction.Down]
  | Square.C => [Direction.Right, Direction.Down]
  | Square.D => [Direction.Up]
  | Square.E => [Direction.Left]
  | Square.F => [Direction.Left]
  | Square.G => [Direction.Up, Direction.Right]
  | _ => []

-- Function to get the next square in a given direction
def next_square (s : Square) (d : Direction) : Option Square :=
  match s, d with
  | Square.One, Direction.Right => some Square.B
  | Square.One, Direction.Down => some Square.D
  | Square.B, Direction.Right => some Square.C
  | Square.B, Direction.Down => some Square.E
  | Square.C, Direction.Right => some Square.Nine
  | Square.C, Direction.Down => some Square.F
  | Square.D, Direction.Up => some Square.A
  | Square.E, Direction.Left => some Square.D
  | Square.F, Direction.Left => some Square.E
  | Square.G, Direction.Up => some Square.D
  | Square.G, Direction.Right => some Square.F
  | _, _ => none

-- Theorem statement
theorem correct_number_placement :
  (∀ s : Square, number_in_square s ∈ Set.range (fun i => i + 1) ∩ Set.Icc 1 9) ∧
  (∀ s : Square, s ≠ Square.Nine → 
    ∃ d ∈ arrows_in_square s, 
      ∃ next : Square, 
        next_square s d = some next ∧ 
        number_in_square next = number_in_square s + 1) :=
sorry

end correct_number_placement_l2968_296822


namespace geometric_sequence_middle_term_l2968_296892

theorem geometric_sequence_middle_term (a b c : ℝ) : 
  (∃ r : ℝ, b = a * r ∧ c = b * r) →  -- geometric sequence condition
  a = 5 + 2 * Real.sqrt 6 →           -- given value of a
  c = 5 - 2 * Real.sqrt 6 →           -- given value of c
  b = 1 ∨ b = -1 :=                   -- conclusion
by sorry

end geometric_sequence_middle_term_l2968_296892


namespace conference_married_men_fraction_l2968_296863

theorem conference_married_men_fraction 
  (total_women : ℕ) 
  (single_women : ℕ) 
  (married_women : ℕ) 
  (married_men : ℕ) 
  (h1 : single_women + married_women = total_women)
  (h2 : married_women = married_men)
  (h3 : (single_women : ℚ) / total_women = 3 / 7) :
  (married_men : ℚ) / (total_women + married_men) = 4 / 11 := by
sorry

end conference_married_men_fraction_l2968_296863


namespace range_of_x_l2968_296884

theorem range_of_x (x : ℝ) : 
  (¬ (x ∈ Set.Icc 2 5 ∨ x < 1 ∨ x > 4)) → 
  x ∈ Set.Ico 1 2 := by
sorry

end range_of_x_l2968_296884


namespace tan_sum_greater_than_three_l2968_296808

theorem tan_sum_greater_than_three :
  Real.tan (40 * π / 180) + Real.tan (45 * π / 180) + Real.tan (50 * π / 180) > 3 := by
  sorry

end tan_sum_greater_than_three_l2968_296808


namespace twentyFifthBaseSum4_l2968_296876

/-- Converts a natural number to its base 4 representation --/
def toBase4 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec aux (m : ℕ) : List ℕ :=
    if m = 0 then [] else (m % 4) :: aux (m / 4)
  aux n |>.reverse

/-- Calculates the sum of digits in a list --/
def sumDigits (l : List ℕ) : ℕ :=
  l.sum

theorem twentyFifthBaseSum4 :
  let base4Rep := toBase4 25
  base4Rep = [1, 2, 1] ∧ sumDigits base4Rep = 4 := by sorry

end twentyFifthBaseSum4_l2968_296876


namespace increase_by_percentage_increase_700_by_75_percent_l2968_296862

theorem increase_by_percentage (initial : ℝ) (percentage : ℝ) :
  initial * (1 + percentage / 100) = initial + initial * (percentage / 100) := by sorry

theorem increase_700_by_75_percent :
  700 * (1 + 75 / 100) = 1225 := by sorry

end increase_by_percentage_increase_700_by_75_percent_l2968_296862


namespace chord_length_and_circle_M_equation_l2968_296832

-- Define the circle
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 = 8

-- Define point P0
def P0 : ℝ × ℝ := (-1, 2)

-- Define point C
def C : ℝ × ℝ := (3, 0)

-- Define the angle of inclination
def alpha : ℝ := 135

-- Define the chord AB
def chord_AB (x y : ℝ) : Prop :=
  y = -x + 1 ∧ circle_equation x y

-- Define circle M
def circle_M (x y : ℝ) : Prop :=
  (x - 1/4)^2 + (y + 1/2)^2 = 125/16

theorem chord_length_and_circle_M_equation :
  (∃ A B : ℝ × ℝ, 
    chord_AB A.1 A.2 ∧ 
    chord_AB B.1 B.2 ∧ 
    P0 = ((A.1 + B.1)/2, (A.2 + B.2)/2) ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = Real.sqrt 30) ∧
  (∀ x y : ℝ, circle_M x y ↔ 
    (∃ A B : ℝ × ℝ, 
      chord_AB A.1 A.2 ∧ 
      chord_AB B.1 B.2 ∧ 
      P0 = ((A.1 + B.1)/2, (A.2 + B.2)/2) ∧
      circle_M C.1 C.2 ∧
      (∀ t : ℝ, circle_M (A.1 + t*(B.1 - A.1)) (A.2 + t*(B.2 - A.2)) → t = 1/2))) := by
  sorry

end chord_length_and_circle_M_equation_l2968_296832


namespace decimal_51_to_binary_l2968_296875

def decimal_to_binary (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 2) ((m % 2) :: acc)
    aux n []

theorem decimal_51_to_binary :
  decimal_to_binary 51 = [1, 1, 0, 0, 1, 1] :=
by sorry

end decimal_51_to_binary_l2968_296875


namespace karen_tagalong_boxes_l2968_296848

/-- The number of Tagalong boxes Karen sold -/
def total_boxes (cases : ℕ) (boxes_per_case : ℕ) : ℕ :=
  cases * boxes_per_case

/-- Theorem stating that Karen sold 36 boxes of Tagalongs -/
theorem karen_tagalong_boxes : total_boxes 3 12 = 36 := by
  sorry

end karen_tagalong_boxes_l2968_296848


namespace complex_power_four_l2968_296821

theorem complex_power_four (i : ℂ) : i^2 = -1 → (1 - i)^4 = -4 := by
  sorry

end complex_power_four_l2968_296821


namespace volume_of_specific_box_l2968_296889

/-- The volume of a box formed by cutting squares from corners of a rectangle --/
def box_volume (length width y : ℝ) : ℝ :=
  (length - 2*y) * (width - 2*y) * y

/-- Theorem: The volume of the box formed from a 12 by 15 inch sheet --/
theorem volume_of_specific_box (y : ℝ) :
  box_volume 15 12 y = 180*y - 54*y^2 + 4*y^3 :=
by sorry

end volume_of_specific_box_l2968_296889


namespace mary_green_crayons_mary_green_crayons_correct_l2968_296807

theorem mary_green_crayons 
  (initial_blue : ℕ) 
  (green_given : ℕ) 
  (blue_given : ℕ) 
  (remaining : ℕ) : ℕ :=
  
  let initial_total := remaining + green_given + blue_given
  let initial_green := initial_total - initial_blue

  initial_green

theorem mary_green_crayons_correct 
  (initial_blue : ℕ) 
  (green_given : ℕ) 
  (blue_given : ℕ) 
  (remaining : ℕ) : 
  mary_green_crayons initial_blue green_given blue_given remaining = 5 :=
by
  -- Given conditions
  have h1 : initial_blue = 8 := by sorry
  have h2 : green_given = 3 := by sorry
  have h3 : blue_given = 1 := by sorry
  have h4 : remaining = 9 := by sorry

  -- Proof
  sorry

end mary_green_crayons_mary_green_crayons_correct_l2968_296807


namespace function_and_cosine_value_l2968_296852

noncomputable def f (ω : ℝ) (m : ℝ) (x : ℝ) : ℝ := 
  2 * (Real.cos (ω * x))^2 + 2 * Real.sqrt 3 * Real.sin (ω * x) * Real.cos (ω * x) + m

theorem function_and_cosine_value 
  (ω : ℝ) (m : ℝ) (x₀ : ℝ) 
  (h_ω : ω > 0)
  (h_highest : f ω m (π / 6) = f ω m x → x ≤ π / 6)
  (h_passes : f ω m 0 = 2)
  (h_x₀_value : f ω m x₀ = 11 / 5)
  (h_x₀_range : π / 4 ≤ x₀ ∧ x₀ ≤ π / 2) :
  (∀ x, f ω m x = 2 * Real.sin (2 * x + π / 6) + 1) ∧
  Real.cos (2 * x₀) = (3 - 4 * Real.sqrt 3) / 10 := by
sorry

end function_and_cosine_value_l2968_296852


namespace distance_to_xoy_plane_l2968_296895

-- Define a 3D point
def Point3D := ℝ × ℝ × ℝ

-- Define the distance from a point to the xOy plane
def distToXOYPlane (p : Point3D) : ℝ := |p.2.2|

-- Theorem statement
theorem distance_to_xoy_plane :
  let P : Point3D := (1, -3, 2)
  distToXOYPlane P = 2 := by
  sorry

end distance_to_xoy_plane_l2968_296895


namespace base6_greater_than_base8_l2968_296847

/-- Convert a base-6 number to base-10 --/
def base6_to_decimal (n : ℕ) : ℕ :=
  (n % 10) + 6 * ((n / 10) % 10) + 36 * (n / 100)

/-- Convert a base-8 number to base-10 --/
def base8_to_decimal (n : ℕ) : ℕ :=
  (n % 10) + 8 * ((n / 10) % 10) + 64 * (n / 100)

theorem base6_greater_than_base8 : base6_to_decimal 403 > base8_to_decimal 217 := by
  sorry

end base6_greater_than_base8_l2968_296847


namespace possible_values_of_a_l2968_296843

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x^2 ≠ 1}
def B (a : ℝ) : Set ℝ := {x : ℝ | a * x = 1}

-- State the theorem
theorem possible_values_of_a (a : ℝ) : 
  (B a ⊆ A) ↔ (a = 0 ∨ a = 1 ∨ a = -1) := by sorry

end possible_values_of_a_l2968_296843


namespace radical_simplification_l2968_296861

theorem radical_simplification (x : ℝ) (h : 4 < x ∧ x < 7) : 
  (((x - 4) ^ 4) ^ (1/4)) + (((x - 7) ^ 4) ^ (1/4)) = 3 := by
  sorry

end radical_simplification_l2968_296861


namespace three_solutions_iff_specific_a_l2968_296819

-- Define the system of equations
def system (x y a : ℝ) : Prop :=
  ((abs (y + 2) + abs (x - 11) - 3) * (x^2 + y^2 - 13) = 0) ∧
  ((x - 5)^2 + (y + 2)^2 = a)

-- Define the condition for exactly three solutions
def has_exactly_three_solutions (a : ℝ) : Prop :=
  ∃! (s₁ s₂ s₃ : ℝ × ℝ), 
    system s₁.1 s₁.2 a ∧ 
    system s₂.1 s₂.2 a ∧ 
    system s₃.1 s₃.2 a ∧
    s₁ ≠ s₂ ∧ s₁ ≠ s₃ ∧ s₂ ≠ s₃

-- Theorem statement
theorem three_solutions_iff_specific_a :
  ∀ a : ℝ, has_exactly_three_solutions a ↔ (a = 9 ∨ a = 42 + 2 * Real.sqrt 377) :=
sorry

end three_solutions_iff_specific_a_l2968_296819


namespace john_taller_than_lena_l2968_296866

/-- Proves that John is 15 cm taller than Lena given the problem conditions -/
theorem john_taller_than_lena (john_height rebeca_height lena_height : ℕ) :
  john_height = 152 →
  john_height = rebeca_height - 6 →
  lena_height + rebeca_height = 295 →
  john_height - lena_height = 15 :=
by
  sorry

end john_taller_than_lena_l2968_296866


namespace inverse_proportion_problem_l2968_296831

/-- A function representing inverse proportionality --/
def inversely_proportional (f : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ x : ℝ, x ≠ 0 → f x * x = k

/-- The main theorem --/
theorem inverse_proportion_problem (f : ℝ → ℝ) 
  (h1 : inversely_proportional f) 
  (h2 : f (-10) = 5) : 
  f (-4) = 25/2 := by
  sorry

end inverse_proportion_problem_l2968_296831


namespace age_difference_l2968_296887

theorem age_difference (A B : ℕ) : B = 39 → A + 10 = 2 * (B - 10) → A - B = 9 := by
  sorry

end age_difference_l2968_296887


namespace complex_fraction_simplification_l2968_296820

theorem complex_fraction_simplification :
  (5 + 6 * Complex.I) / (3 + Complex.I) = 21/10 + 13/10 * Complex.I := by
  sorry

end complex_fraction_simplification_l2968_296820


namespace statue_weight_calculation_l2968_296837

/-- The weight of a marble statue after three successive cuts -/
def final_statue_weight (initial_weight : ℝ) : ℝ :=
  initial_weight * (1 - 0.3) * (1 - 0.3) * (1 - 0.15)

/-- Theorem stating the final weight of the statue -/
theorem statue_weight_calculation :
  final_statue_weight 300 = 124.95 := by
  sorry

#eval final_statue_weight 300

end statue_weight_calculation_l2968_296837


namespace election_majority_proof_l2968_296874

theorem election_majority_proof (total_votes : ℕ) (winning_percentage : ℚ) : 
  total_votes = 450 → 
  winning_percentage = 70 / 100 → 
  (winning_percentage * total_votes : ℚ).num - ((1 - winning_percentage) * total_votes : ℚ).num = 180 := by
sorry

end election_majority_proof_l2968_296874
