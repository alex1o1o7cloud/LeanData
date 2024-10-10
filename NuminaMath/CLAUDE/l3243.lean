import Mathlib

namespace custom_distance_additive_on_line_segment_l3243_324375

/-- Custom distance function between two points in 2D space -/
def custom_distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  |x₂ - x₁| + |y₂ - y₁|

/-- Theorem: For any three points A, B, and C, where C is on the line segment AB,
    the sum of the custom distances AC and CB equals the custom distance AB -/
theorem custom_distance_additive_on_line_segment 
  (x₁ y₁ x₂ y₂ x y : ℝ) 
  (h_between_x : (x₁ - x) * (x₂ - x) ≤ 0)
  (h_between_y : (y₁ - y) * (y₂ - y) ≤ 0) :
  custom_distance x₁ y₁ x y + custom_distance x y x₂ y₂ = custom_distance x₁ y₁ x₂ y₂ :=
by sorry

#check custom_distance_additive_on_line_segment

end custom_distance_additive_on_line_segment_l3243_324375


namespace sugar_purchase_efficiency_l3243_324392

/-- Proves that Xiao Li's method of buying sugar is more cost-effective than Xiao Wang's --/
theorem sugar_purchase_efficiency
  (n : ℕ) (a : ℕ → ℝ)
  (h_n : n > 1)
  (h_a : ∀ i, i ∈ Finset.range n → a i > 0) :
  (Finset.sum (Finset.range n) a) / n ≥ n / (Finset.sum (Finset.range n) (λ i => 1 / a i)) :=
by sorry

end sugar_purchase_efficiency_l3243_324392


namespace baker_a_remaining_pastries_l3243_324362

/-- Baker A's initial number of cakes -/
def baker_a_initial_cakes : ℕ := 7

/-- Baker A's initial number of pastries -/
def baker_a_initial_pastries : ℕ := 148

/-- Baker B's initial number of cakes -/
def baker_b_initial_cakes : ℕ := 10

/-- Baker B's initial number of pastries -/
def baker_b_initial_pastries : ℕ := 200

/-- Number of pastries Baker A sold -/
def baker_a_sold_pastries : ℕ := 103

/-- Theorem: Baker A will have 71 pastries left after redistribution and selling -/
theorem baker_a_remaining_pastries :
  (baker_a_initial_pastries + baker_b_initial_pastries) / 2 - baker_a_sold_pastries = 71 := by
  sorry

end baker_a_remaining_pastries_l3243_324362


namespace triangle_angle_measure_l3243_324374

theorem triangle_angle_measure (D E F : ℝ) : 
  D = 80 →
  E = 2 * F + 24 →
  D + E + F = 180 →
  F = 76 / 3 := by
sorry

end triangle_angle_measure_l3243_324374


namespace selling_price_ratio_l3243_324340

/-- Proves that the ratio of selling prices is 2:1 given different profit percentages -/
theorem selling_price_ratio (cost : ℝ) (profit1 profit2 : ℝ) 
  (h1 : profit1 = 0.20)
  (h2 : profit2 = 1.40) :
  (cost + profit2 * cost) / (cost + profit1 * cost) = 2 := by
  sorry

end selling_price_ratio_l3243_324340


namespace real_part_of_complex_product_l3243_324317

theorem real_part_of_complex_product : 
  let i : ℂ := Complex.I
  let z : ℂ := (1 + 3*i) * i
  Complex.re z = -3 := by sorry

end real_part_of_complex_product_l3243_324317


namespace geometric_sequence_fourth_term_l3243_324367

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n

theorem geometric_sequence_fourth_term
  (a : ℕ → ℝ)
  (h_geom : geometric_sequence a)
  (h_ratio : 3 * a 5 = a 6)
  (h_second : a 2 = 1) :
  a 4 = 9 := by
sorry

end geometric_sequence_fourth_term_l3243_324367


namespace condition_p_sufficient_not_necessary_for_q_l3243_324319

theorem condition_p_sufficient_not_necessary_for_q :
  (∀ x : ℝ, |x + 1| ≤ 2 → -3 ≤ x ∧ x ≤ 2) ∧
  (∃ x : ℝ, -3 ≤ x ∧ x ≤ 2 ∧ ¬(|x + 1| ≤ 2)) := by
  sorry

end condition_p_sufficient_not_necessary_for_q_l3243_324319


namespace unique_three_digit_number_square_equals_sum_of_digits_power_five_l3243_324314

/-- A function that returns the sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Theorem stating that 243 is the only three-digit number whose square 
    is equal to the sum of its digits raised to the power of 5 -/
theorem unique_three_digit_number_square_equals_sum_of_digits_power_five : 
  ∃! n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n^2 = (sum_of_digits n)^5 := by sorry

end unique_three_digit_number_square_equals_sum_of_digits_power_five_l3243_324314


namespace exists_strategy_to_find_genuine_coin_l3243_324358

/-- Represents a coin, which can be either genuine or counterfeit. -/
inductive Coin
| genuine
| counterfeit

/-- Represents the result of weighing two coins. -/
inductive WeighResult
| equal
| left_heavier
| right_heavier

/-- A function that simulates weighing two coins. -/
def weigh : Coin → Coin → WeighResult := sorry

/-- The total number of coins. -/
def total_coins : Nat := 100

/-- A function that represents the distribution of coins. -/
def coin_distribution : Fin total_coins → Coin := sorry

/-- The number of counterfeit coins. -/
def counterfeit_count : Nat := sorry

/-- Assumption that there are more than 0 but less than 99 counterfeit coins. -/
axiom counterfeit_range : 0 < counterfeit_count ∧ counterfeit_count < 99

/-- A strategy is a function that takes the current state and returns the next pair of coins to weigh. -/
def Strategy := List WeighResult → Fin total_coins × Fin total_coins

/-- The theorem stating that there exists a strategy to find a genuine coin. -/
theorem exists_strategy_to_find_genuine_coin :
  ∃ (s : Strategy), ∀ (coin_dist : Fin total_coins → Coin),
    ∃ (n : Nat), n ≤ 99 ∧
      (∃ (i : Fin total_coins), coin_dist i = Coin.genuine ∧
        (∀ (j : Fin total_coins), j ≠ i → coin_dist j = Coin.counterfeit)) :=
sorry

end exists_strategy_to_find_genuine_coin_l3243_324358


namespace max_abs_z_l3243_324303

theorem max_abs_z (z : ℂ) : 
  Complex.abs (z + 3 + 4*I) ≤ 2 → 
  ∃ (w : ℂ), Complex.abs (w + 3 + 4*I) ≤ 2 ∧ 
             ∀ (u : ℂ), Complex.abs (u + 3 + 4*I) ≤ 2 → Complex.abs u ≤ Complex.abs w ∧
             Complex.abs w = 7 :=
by sorry

end max_abs_z_l3243_324303


namespace max_value_of_ab_l3243_324364

theorem max_value_of_ab (a b : ℝ) (h1 : b > 0) (h2 : 3 * a + 4 * b = 2) :
  a * b ≤ 1 / 12 ∧ ∃ (a₀ b₀ : ℝ), b₀ > 0 ∧ 3 * a₀ + 4 * b₀ = 2 ∧ a₀ * b₀ = 1 / 12 := by
  sorry

end max_value_of_ab_l3243_324364


namespace sequence_inequality_l3243_324354

/-- The sum of the first n terms of the sequence -/
def S (n : ℕ+) : ℤ := -2 * n.val ^ 2 + 3 * n.val

/-- The n-th term of the sequence -/
def a (n : ℕ+) : ℤ := -4 * n.val + 5

/-- Theorem stating the relationship between na_n, S_n, and na_1 for n ≥ 2 -/
theorem sequence_inequality (n : ℕ+) (h : 2 ≤ n.val) :
  (n.val : ℤ) * a n < S n ∧ S n < (n.val : ℤ) * a 1 :=
sorry

end sequence_inequality_l3243_324354


namespace koala_fiber_consumption_l3243_324313

/-- Given that koalas absorb 30% of the fiber they eat, 
    prove that if a koala absorbed 12 ounces of fiber, 
    it ate 40 ounces of fiber. -/
theorem koala_fiber_consumption 
  (absorption_rate : ℝ) 
  (absorbed_amount : ℝ) 
  (h1 : absorption_rate = 0.3)
  (h2 : absorbed_amount = 12) :
  absorbed_amount / absorption_rate = 40 := by
  sorry

end koala_fiber_consumption_l3243_324313


namespace continuous_with_property_F_is_nondecreasing_l3243_324377

-- Define property F
def has_property_F (f : ℝ → ℝ) : Prop :=
  ∀ a : ℝ, ∃ b : ℝ, b < a ∧ ∀ x ∈ Set.Ioo b a, f x ≤ f a

-- Define nondecreasing
def nondecreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x ≤ y → f x ≤ f y

-- Theorem statement
theorem continuous_with_property_F_is_nondecreasing (f : ℝ → ℝ) 
  (hf : Continuous f) (hF : has_property_F f) : nondecreasing f := by
  sorry


end continuous_with_property_F_is_nondecreasing_l3243_324377


namespace admission_probability_l3243_324330

theorem admission_probability (P_A P_B : ℝ) (h_A : P_A = 0.6) (h_B : P_B = 0.7) 
  (h_indep : P_A + P_B - P_A * P_B = P_A + P_B - (P_A * P_B)) : 
  P_A + P_B - P_A * P_B = 0.88 := by
sorry

end admission_probability_l3243_324330


namespace infinite_squares_l3243_324365

theorem infinite_squares (k : ℕ) (hk : k ≥ 2) :
  ∃ (S : Set ℕ), Set.Infinite S ∧
    ∀ n ∈ S, ∃ (u v : ℕ), k * n + 1 = u^2 ∧ (k + 1) * n + 1 = v^2 := by
  sorry

end infinite_squares_l3243_324365


namespace m_arithmetic_pascal_triangle_structure_l3243_324336

/-- m-arithmetic Pascal triangle with s-th row zeros except extremes -/
structure MArithmeticPascalTriangle where
  m : ℕ
  s : ℕ
  Δ : ℕ → ℕ → ℕ
  P : ℕ → ℕ → ℕ

/-- Properties of the m-arithmetic Pascal triangle -/
class MArithmeticPascalTriangleProps (T : MArithmeticPascalTriangle) where
  zero_middle : ∀ (i k : ℕ), i % T.s = 0 → 0 < k → k < i → T.Δ i k = 0
  relation_a : ∀ (n k : ℕ), T.Δ n (k-1) + T.Δ n k = T.Δ (n+1) k
  relation_b : ∀ (n k : ℕ), T.Δ n k = T.P n k * T.Δ 0 0

/-- Theorem: The structure of the m-arithmetic Pascal triangle is correct -/
theorem m_arithmetic_pascal_triangle_structure 
  (T : MArithmeticPascalTriangle) 
  [MArithmeticPascalTriangleProps T] : 
  ∀ (n k : ℕ), T.Δ n k = T.P n k * T.Δ 0 0 := by
  sorry

end m_arithmetic_pascal_triangle_structure_l3243_324336


namespace longer_train_length_l3243_324361

/-- Calculates the length of the longer train given the speeds of two trains,
    the time they take to cross each other, and the length of the shorter train. -/
theorem longer_train_length
  (speed1 speed2 : ℝ)
  (crossing_time : ℝ)
  (shorter_train_length : ℝ)
  (h1 : speed1 = 68)
  (h2 : speed2 = 40)
  (h3 : crossing_time = 11.999040076793857)
  (h4 : shorter_train_length = 160)
  (h5 : speed1 > 0)
  (h6 : speed2 > 0)
  (h7 : crossing_time > 0)
  (h8 : shorter_train_length > 0) :
  let relative_speed := (speed1 + speed2) * (1000 / 3600)
  let total_distance := relative_speed * crossing_time
  total_distance - shorter_train_length = 200 := by
  sorry

end longer_train_length_l3243_324361


namespace other_diagonal_length_l3243_324376

-- Define the rhombus
structure Rhombus where
  diagonal1 : ℝ
  diagonal2 : ℝ
  triangle_area : ℝ

-- Define the properties of the rhombus
def rhombus_properties (r : Rhombus) : Prop :=
  r.diagonal1 = 20 ∧ r.triangle_area = 75

-- Theorem statement
theorem other_diagonal_length (r : Rhombus) 
  (h : rhombus_properties r) : r.diagonal2 = 15 := by
  sorry

end other_diagonal_length_l3243_324376


namespace negation_of_existential_proposition_l3243_324341

theorem negation_of_existential_proposition :
  ¬(∃ x : ℝ, x ≥ 0 ∧ x^2 > 3) ↔ ∀ x : ℝ, x ≥ 0 → x^2 ≤ 3 := by
  sorry

end negation_of_existential_proposition_l3243_324341


namespace trigonometric_expression_simplification_l3243_324300

theorem trigonometric_expression_simplification :
  let expr := (Real.sin (10 * π / 180) + Real.sin (20 * π / 180) + 
               Real.sin (30 * π / 180) + Real.sin (40 * π / 180) + 
               Real.sin (50 * π / 180) + Real.sin (60 * π / 180) + 
               Real.sin (70 * π / 180) + Real.sin (80 * π / 180)) / 
              (Real.cos (5 * π / 180) * Real.cos (10 * π / 180) * Real.cos (20 * π / 180))
  expr = 4 * Real.sqrt 2 := by
sorry

end trigonometric_expression_simplification_l3243_324300


namespace common_course_probability_l3243_324306

/-- Represents the set of all possible course selections for a student -/
def CourseSelection : Type := Fin 10

/-- The total number of possible course selections for three students -/
def totalCombinations : ℕ := 1000

/-- The number of favorable combinations where at least two students share two courses -/
def favorableCombinations : ℕ := 280

/-- The probability that any one student will have at least two elective courses in common with the other two students -/
def commonCourseProbability : ℚ := 79 / 250

theorem common_course_probability :
  (favorableCombinations : ℚ) / totalCombinations = commonCourseProbability := by
  sorry

end common_course_probability_l3243_324306


namespace a_less_than_b_l3243_324332

-- Define the function f
def f (x m : ℝ) : ℝ := -4 * x^2 + 8 * x + m

-- State the theorem
theorem a_less_than_b (m : ℝ) (a b : ℝ) 
  (h1 : f (-2) m = a) 
  (h2 : f 3 m = b) : 
  a < b := by
  sorry

end a_less_than_b_l3243_324332


namespace inequality_generalization_l3243_324305

theorem inequality_generalization (x : ℝ) (n : ℕ) (h : x > 0) :
  x^n + n / x > n + 1 := by
  sorry

end inequality_generalization_l3243_324305


namespace jogging_track_circumference_jogging_track_circumference_value_l3243_324355

/-- The circumference of a circular jogging track where two people walking in opposite directions meet. -/
theorem jogging_track_circumference (deepak_speed wife_speed : ℝ) (meeting_time : ℝ) : ℝ :=
  let deepak_speed := 4.5
  let wife_speed := 3.75
  let meeting_time := 3.84 / 60
  2 * (deepak_speed + wife_speed) * meeting_time

/-- The circumference of the jogging track is 1.056 km. -/
theorem jogging_track_circumference_value :
  jogging_track_circumference 4.5 3.75 (3.84 / 60) = 1.056 := by
  sorry

end jogging_track_circumference_jogging_track_circumference_value_l3243_324355


namespace sams_watermelons_l3243_324388

theorem sams_watermelons (grown : ℕ) (eaten : ℕ) (h1 : grown = 4) (h2 : eaten = 3) :
  grown - eaten = 1 := by
  sorry

end sams_watermelons_l3243_324388


namespace daniel_noodles_l3243_324382

/-- The number of noodles Daniel had initially -/
def initial_noodles : ℝ := 54.0

/-- The number of noodles Daniel gave away -/
def given_away : ℝ := 12.0

/-- The number of noodles Daniel had left -/
def remaining_noodles : ℝ := initial_noodles - given_away

theorem daniel_noodles : remaining_noodles = 42.0 := by sorry

end daniel_noodles_l3243_324382


namespace power_two_2005_mod_7_l3243_324331

theorem power_two_2005_mod_7 : 2^2005 % 7 = 4 := by
  sorry

end power_two_2005_mod_7_l3243_324331


namespace find_number_l3243_324345

theorem find_number : ∃ x : ℝ, 4 * x - 23 = 33 ∧ x = 14 := by
  sorry

end find_number_l3243_324345


namespace cube_root_negative_27_l3243_324334

theorem cube_root_negative_27 :
  (∃ x : ℝ, x^3 = -27 ∧ x = -3) ∧
  (¬ (∀ x : ℝ, x^2 = 64 → x = 8 ∨ x = -8)) ∧
  (¬ ((-Real.sqrt 2)^2 = 4)) ∧
  (¬ (Real.sqrt ((-5)^2) = -5)) :=
by sorry

end cube_root_negative_27_l3243_324334


namespace certain_number_proof_l3243_324370

theorem certain_number_proof (y : ℝ) : 
  (0.25 * 680 = 0.20 * y - 30) → y = 1000 := by
  sorry

end certain_number_proof_l3243_324370


namespace cookie_ratio_proof_l3243_324357

theorem cookie_ratio_proof (raisin_cookies oatmeal_cookies : ℕ) : 
  raisin_cookies = 42 → 
  raisin_cookies + oatmeal_cookies = 49 → 
  raisin_cookies / oatmeal_cookies = 6 := by
sorry

end cookie_ratio_proof_l3243_324357


namespace f_nonnegative_range_l3243_324371

def f (a x : ℝ) : ℝ := a * x^2 - (a + 1) * x + 2

theorem f_nonnegative_range (a : ℝ) :
  (∀ x ∈ Set.Icc (-1) 3, f a x ≥ 0) →
  1/6 ≤ a ∧ a ≤ 3 + 2 * Real.sqrt 2 :=
by sorry

end f_nonnegative_range_l3243_324371


namespace geometry_test_passing_l3243_324390

theorem geometry_test_passing (total_problems : ℕ) (passing_percentage : ℚ) 
  (hp : passing_percentage = 85 / 100) (ht : total_problems = 50) :
  ∃ (max_missable : ℕ), 
    (max_missable : ℚ) / total_problems ≤ 1 - passing_percentage ∧
    ∀ (n : ℕ), (n : ℚ) / total_problems ≤ 1 - passing_percentage → n ≤ max_missable :=
by sorry

end geometry_test_passing_l3243_324390


namespace selection_methods_count_l3243_324380

/-- The number of different ways to select one teacher and one student -/
def selection_methods (num_teachers : ℕ) (num_male_students : ℕ) (num_female_students : ℕ) : ℕ :=
  num_teachers * (num_male_students + num_female_students)

/-- Theorem stating that the number of selection methods for the given problem is 39 -/
theorem selection_methods_count :
  selection_methods 3 8 5 = 39 := by
  sorry

end selection_methods_count_l3243_324380


namespace salem_population_decrease_l3243_324325

def salem_leesburg_ratio : ℕ := 15
def leesburg_population : ℕ := 58940
def salem_women_population : ℕ := 377050

def salem_original_population : ℕ := salem_leesburg_ratio * leesburg_population
def salem_current_population : ℕ := 2 * salem_women_population

theorem salem_population_decrease :
  salem_original_population - salem_current_population = 130000 :=
by sorry

end salem_population_decrease_l3243_324325


namespace final_answer_is_67_l3243_324344

def ben_calculation (x : ℕ) : ℕ :=
  ((x + 2) * 3) + 5

def sue_calculation (x : ℕ) : ℕ :=
  ((x - 3) * 3) + 7

theorem final_answer_is_67 :
  sue_calculation (ben_calculation 4) = 67 := by
  sorry

end final_answer_is_67_l3243_324344


namespace cow_count_l3243_324363

/-- Given a group of ducks and cows, where the total number of legs is 36 more than
    twice the number of heads, prove that the number of cows is 18. -/
theorem cow_count (ducks cows : ℕ) : 
  (2 * ducks + 4 * cows = 2 * (ducks + cows) + 36) → cows = 18 := by
  sorry

end cow_count_l3243_324363


namespace no_solution_l3243_324389

theorem no_solution : ¬∃ (A B : ℤ), 
  A = 5 + 3 ∧ 
  B = A - 2 ∧ 
  0 ≤ A ∧ A ≤ 9 ∧ 
  0 ≤ B ∧ B ≤ 9 ∧ 
  0 ≤ A + B ∧ A + B ≤ 9 := by
  sorry

end no_solution_l3243_324389


namespace problem_solution_l3243_324398

theorem problem_solution (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h1 : a^2 + a*b + b^2 = 9)
  (h2 : b^2 + b*c + c^2 = 52)
  (h3 : c^2 + c*a + a^2 = 49) :
  (49*b^2 - 33*b*c + 9*c^2) / a^2 = 52 := by
sorry

end problem_solution_l3243_324398


namespace special_polynomial_derivative_theorem_l3243_324385

/-- A second-degree polynomial with roots in [-1, 1] and |f(x₀)| = 1 for some x₀ ∈ [-1, 1] -/
structure SpecialPolynomial where
  f : ℝ → ℝ
  degree_two : ∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c
  roots_in_interval : ∀ r, f r = 0 → r ∈ Set.Icc (-1 : ℝ) 1
  exists_unit_value : ∃ x₀ ∈ Set.Icc (-1 : ℝ) 1, |f x₀| = 1

/-- The main theorem about special polynomials -/
theorem special_polynomial_derivative_theorem (p : SpecialPolynomial) :
  (∀ α ∈ Set.Icc (0 : ℝ) 1, ∃ ζ ∈ Set.Icc (-1 : ℝ) 1, |deriv p.f ζ| = α) ∧
  (¬∃ ζ ∈ Set.Icc (-1 : ℝ) 1, |deriv p.f ζ| > 1) :=
by sorry

end special_polynomial_derivative_theorem_l3243_324385


namespace cauchy_schwarz_inequality_3d_l3243_324379

theorem cauchy_schwarz_inequality_3d (x₁ y₁ z₁ x₂ y₂ z₂ : ℝ) :
  (x₁ * x₂ + y₁ * y₂ + z₁ * z₂)^2 ≤ (x₁^2 + y₁^2 + z₁^2) * (x₂^2 + y₂^2 + z₂^2) := by
  sorry

end cauchy_schwarz_inequality_3d_l3243_324379


namespace error_clock_correct_time_fraction_l3243_324352

/-- Represents a 24-hour digital clock with specific display errors -/
structure ErrorClock where
  /-- The clock displays 9 instead of 1 -/
  error_one : Nat
  /-- The clock displays 5 instead of 2 -/
  error_two : Nat

/-- Calculates the fraction of the day an ErrorClock shows the correct time -/
def correct_time_fraction (clock : ErrorClock) : Rat :=
  sorry

/-- Theorem stating that the ErrorClock shows the correct time for 7/36 of the day -/
theorem error_clock_correct_time_fraction :
  ∀ (clock : ErrorClock),
  clock.error_one = 9 ∧ clock.error_two = 5 →
  correct_time_fraction clock = 7 / 36 := by
  sorry

end error_clock_correct_time_fraction_l3243_324352


namespace correct_stratified_sampling_l3243_324395

/-- Represents the number of students in each year --/
structure StudentCounts where
  firstYear : ℕ
  secondYear : ℕ
  thirdYear : ℕ

/-- Calculates the stratified sample size for a given year --/
def stratifiedSampleSize (yearCount : ℕ) (totalCount : ℕ) (sampleSize : ℕ) : ℕ :=
  (yearCount * sampleSize + totalCount - 1) / totalCount

/-- Theorem stating the correct stratified sampling for the given problem --/
theorem correct_stratified_sampling (totalStudents : StudentCounts) 
    (h1 : totalStudents.firstYear = 540)
    (h2 : totalStudents.secondYear = 440)
    (h3 : totalStudents.thirdYear = 420)
    (totalSampleSize : ℕ) 
    (h4 : totalSampleSize = 70) :
  let totalCount := totalStudents.firstYear + totalStudents.secondYear + totalStudents.thirdYear
  (stratifiedSampleSize totalStudents.firstYear totalCount totalSampleSize,
   stratifiedSampleSize totalStudents.secondYear totalCount totalSampleSize,
   stratifiedSampleSize totalStudents.thirdYear totalCount totalSampleSize) = (27, 22, 21) := by
  sorry

end correct_stratified_sampling_l3243_324395


namespace calculate_tip_percentage_l3243_324399

/-- Calculates the percentage tip given the prices of four ice cream sundaes and the final bill -/
theorem calculate_tip_percentage (price1 price2 price3 price4 final_bill : ℚ) : 
  price1 = 9 ∧ price2 = 7.5 ∧ price3 = 10 ∧ price4 = 8.5 ∧ final_bill = 42 →
  (final_bill - (price1 + price2 + price3 + price4)) / (price1 + price2 + price3 + price4) * 100 = 20 := by
  sorry

end calculate_tip_percentage_l3243_324399


namespace triangle_stack_sum_impossible_l3243_324315

theorem triangle_stack_sum_impossible : ¬ ∃ k : ℕ+, 6 * k = 165 := by
  sorry

end triangle_stack_sum_impossible_l3243_324315


namespace contest_scores_mode_and_median_l3243_324329

def scores : List ℕ := [91, 95, 89, 93, 88, 94, 95]

def mode (l : List ℕ) : ℕ := sorry

def median (l : List ℕ) : ℕ := sorry

theorem contest_scores_mode_and_median :
  mode scores = 95 ∧ median scores = 93 := by sorry

end contest_scores_mode_and_median_l3243_324329


namespace choose_leaders_count_l3243_324312

/-- A club with members divided by gender and class -/
structure Club where
  total_members : ℕ
  boys : ℕ
  girls : ℕ
  classes : ℕ
  boys_per_class : ℕ
  girls_per_class : ℕ

/-- The number of ways to choose a president and vice-president -/
def choose_leaders (c : Club) : ℕ := sorry

/-- The specific club configuration -/
def my_club : Club := {
  total_members := 24,
  boys := 12,
  girls := 12,
  classes := 2,
  boys_per_class := 6,
  girls_per_class := 6
}

/-- Theorem stating the number of ways to choose leaders for the given club -/
theorem choose_leaders_count : choose_leaders my_club = 144 := by sorry

end choose_leaders_count_l3243_324312


namespace amusement_park_probabilities_l3243_324366

/-- Amusement park problem -/
theorem amusement_park_probabilities
  (p_A1 : ℝ)
  (p_B1 : ℝ)
  (p_A2_given_A1 : ℝ)
  (p_A2_given_B1 : ℝ)
  (h1 : p_A1 = 0.3)
  (h2 : p_B1 = 0.7)
  (h3 : p_A2_given_A1 = 0.7)
  (h4 : p_A2_given_B1 = 0.6)
  (h5 : p_A1 + p_B1 = 1) :
  let p_A2 := p_A1 * p_A2_given_A1 + p_B1 * p_A2_given_B1
  let p_B1_given_A2 := (p_B1 * p_A2_given_B1) / p_A2
  ∃ (ε : ℝ), abs (p_A2 - 0.63) < ε ∧ abs (p_B1_given_A2 - (2/3)) < ε :=
by
  sorry


end amusement_park_probabilities_l3243_324366


namespace nested_fraction_equals_five_thirds_l3243_324301

theorem nested_fraction_equals_five_thirds :
  1 + (1 / (1 + (1 / (1 + 1)))) = 5 / 3 := by
  sorry

end nested_fraction_equals_five_thirds_l3243_324301


namespace hundred_with_fewer_threes_l3243_324333

/-- An arithmetic expression using only the number 3 and basic operations -/
inductive Expr
  | three : Expr
  | add : Expr → Expr → Expr
  | sub : Expr → Expr → Expr
  | mul : Expr → Expr → Expr
  | div : Expr → Expr → Expr

/-- Evaluate an expression to a rational number -/
def eval : Expr → ℚ
  | Expr.three => 3
  | Expr.add e1 e2 => eval e1 + eval e2
  | Expr.sub e1 e2 => eval e1 - eval e2
  | Expr.mul e1 e2 => eval e1 * eval e2
  | Expr.div e1 e2 => eval e1 / eval e2

/-- Count the number of 3's in an expression -/
def count_threes : Expr → Nat
  | Expr.three => 1
  | Expr.add e1 e2 => count_threes e1 + count_threes e2
  | Expr.sub e1 e2 => count_threes e1 + count_threes e2
  | Expr.mul e1 e2 => count_threes e1 + count_threes e2
  | Expr.div e1 e2 => count_threes e1 + count_threes e2

/-- Theorem: There exists an expression that evaluates to 100 using fewer than 10 threes -/
theorem hundred_with_fewer_threes : ∃ e : Expr, eval e = 100 ∧ count_threes e < 10 := by
  sorry

end hundred_with_fewer_threes_l3243_324333


namespace ball_probabilities_l3243_324310

/-- Represents the color of a ball -/
inductive BallColor
  | Yellow
  | Green
  | Red

/-- Represents the box of balls -/
structure BallBox where
  total : Nat
  yellow : Nat
  green : Nat
  red : Nat

/-- Calculates the probability of drawing a ball of a specific color -/
def probability (box : BallBox) (color : BallColor) : Rat :=
  match color with
  | BallColor.Yellow => box.yellow / box.total
  | BallColor.Green => box.green / box.total
  | BallColor.Red => box.red / box.total

/-- The main theorem to prove -/
theorem ball_probabilities (box : BallBox) : 
  box.total = 10 ∧ 
  box.yellow = 1 ∧ 
  box.green = 3 ∧ 
  box.red = box.total - box.yellow - box.green →
  probability box BallColor.Green > probability box BallColor.Yellow ∧
  probability box BallColor.Red = 3/5 := by
  sorry


end ball_probabilities_l3243_324310


namespace inverse_inequality_l3243_324349

theorem inverse_inequality (x y : ℝ) (h1 : x > y) (h2 : y > 0) : 1/x < 1/y := by
  sorry

end inverse_inequality_l3243_324349


namespace clothes_shop_discount_l3243_324368

theorem clothes_shop_discount (num_friends : ℕ) (original_price : ℝ) (discount_percent : ℝ) : 
  num_friends = 4 → 
  original_price = 20 → 
  discount_percent = 50 → 
  (num_friends : ℝ) * (original_price * (1 - discount_percent / 100)) = 40 := by
sorry

end clothes_shop_discount_l3243_324368


namespace real_roots_necessary_condition_l3243_324324

theorem real_roots_necessary_condition (a : ℝ) :
  (∃ x : ℝ, x^2 - a*x + 1 = 0) →
  (a ≥ 1 ∨ a ≤ -2) :=
by sorry

end real_roots_necessary_condition_l3243_324324


namespace function_inequality_l3243_324347

/-- Given functions f and g, prove that a ≤ 1 -/
theorem function_inequality (f g : ℝ → ℝ) (a : ℝ) : 
  (∀ x, f x = x + 4 / x) →
  (∀ x, g x = 2^x + a) →
  (∀ x₁ ∈ Set.Icc (1/2) 1, ∃ x₂ ∈ Set.Icc 2 3, f x₁ ≥ g x₂) →
  a ≤ 1 := by sorry

end function_inequality_l3243_324347


namespace unique_solution_l3243_324337

/-- The function f(x) = x^3 + 3x^2 + 1 -/
def f (x : ℝ) : ℝ := x^3 + 3*x^2 + 1

/-- Theorem stating the unique solution for a and b -/
theorem unique_solution (a b : ℝ) : 
  a ≠ 0 ∧ 
  (∀ x : ℝ, f x - f a = (x - b) * (x - a)^2) → 
  a = -2 ∧ b = 1 :=
sorry

end unique_solution_l3243_324337


namespace y1_value_l3243_324338

theorem y1_value (y1 y2 y3 : Real) 
  (h1 : 0 ≤ y3 ∧ y3 ≤ y2 ∧ y2 ≤ y1 ∧ y1 ≤ 1)
  (h2 : (1-y1)^2 + 2*(y1-y2)^2 + 2*(y2-y3)^2 + y3^2 = 1/2) :
  y1 = 3/4 := by
sorry

end y1_value_l3243_324338


namespace linear_equation_rewrite_l3243_324360

theorem linear_equation_rewrite (k m : ℚ) : 
  (∀ x y : ℚ, 2 * x + 3 * y - 4 = 0 ↔ y = k * x + m) → 
  k + m = 2/3 := by
sorry

end linear_equation_rewrite_l3243_324360


namespace root_exists_in_interval_l3243_324328

def f (x : ℝ) := x^3 - 9

theorem root_exists_in_interval :
  ∃ (x : ℝ), x ∈ Set.Ioo 2 3 ∧ f x = 0 :=
by
  sorry

end root_exists_in_interval_l3243_324328


namespace hyperbola_focal_length_and_eccentricity_l3243_324393

/-- The focal length of a hyperbola with equation x^2 - y^2/3 = 1 is 4 and its eccentricity is 2 -/
theorem hyperbola_focal_length_and_eccentricity :
  let a : ℝ := 1
  let b : ℝ := Real.sqrt 3
  let c : ℝ := Real.sqrt (a^2 + b^2)
  let focal_length : ℝ := 2 * c
  let eccentricity : ℝ := c / a
  focal_length = 4 ∧ eccentricity = 2 := by
sorry


end hyperbola_focal_length_and_eccentricity_l3243_324393


namespace lines_cannot_form_triangle_iff_m_in_set_l3243_324373

/-- A line in the plane, represented by its equation ax + by = c -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if three lines form a triangle -/
def form_triangle (l₁ l₂ l₃ : Line) : Prop :=
  sorry

/-- The set of m values for which the lines cannot form a triangle -/
def invalid_m_values : Set ℝ :=
  {4, -1/6, -1, 2/3}

theorem lines_cannot_form_triangle_iff_m_in_set (m : ℝ) :
  let l₁ : Line := ⟨4, 1, 4⟩
  let l₂ : Line := ⟨m, 1, 0⟩
  let l₃ : Line := ⟨2, -3*m, 4⟩
  ¬(form_triangle l₁ l₂ l₃) ↔ m ∈ invalid_m_values :=
by sorry

end lines_cannot_form_triangle_iff_m_in_set_l3243_324373


namespace lifeguard_swimming_test_l3243_324326

/-- Lifeguard swimming test problem -/
theorem lifeguard_swimming_test
  (total_distance : ℝ)
  (front_crawl_speed : ℝ)
  (total_time : ℝ)
  (front_crawl_time : ℝ)
  (h1 : total_distance = 500)
  (h2 : front_crawl_speed = 45)
  (h3 : total_time = 12)
  (h4 : front_crawl_time = 8) :
  let front_crawl_distance := front_crawl_speed * front_crawl_time
  let breaststroke_distance := total_distance - front_crawl_distance
  let breaststroke_time := total_time - front_crawl_time
  let breaststroke_speed := breaststroke_distance / breaststroke_time
  breaststroke_speed = 35 := by
sorry


end lifeguard_swimming_test_l3243_324326


namespace calculation_proof_l3243_324353

theorem calculation_proof : (3.25 - 1.57) * 2 = 3.36 := by
  sorry

end calculation_proof_l3243_324353


namespace probability_ace_king_queen_standard_deck_l3243_324350

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : ℕ)
  (num_aces : ℕ)
  (num_kings : ℕ)
  (num_queens : ℕ)

/-- Calculates the probability of drawing an Ace, then a King, then a Queen from a standard deck without replacement -/
def probability_ace_king_queen (d : Deck) : ℚ :=
  (d.num_aces : ℚ) / d.total_cards *
  (d.num_kings : ℚ) / (d.total_cards - 1) *
  (d.num_queens : ℚ) / (d.total_cards - 2)

/-- Theorem stating the probability of drawing an Ace, then a King, then a Queen from a standard 52-card deck -/
theorem probability_ace_king_queen_standard_deck :
  probability_ace_king_queen ⟨52, 4, 4, 4⟩ = 8 / 16575 := by
  sorry

end probability_ace_king_queen_standard_deck_l3243_324350


namespace cantaloupes_left_l3243_324369

/-- The number of cantaloupes left after growing and losing some due to bad weather -/
theorem cantaloupes_left (fred tim maria lost : ℕ) (h1 : fred = 38) (h2 : tim = 44) (h3 : maria = 57) (h4 : lost = 12) :
  fred + tim + maria - lost = 127 := by
  sorry

end cantaloupes_left_l3243_324369


namespace cherry_pits_correct_l3243_324346

/-- The number of cherry pits Kim planted -/
def cherry_pits : ℕ := 80

/-- The fraction of cherry pits that sprout -/
def sprout_rate : ℚ := 1/4

/-- The number of saplings Kim sold -/
def saplings_sold : ℕ := 6

/-- The number of saplings left after selling -/
def saplings_left : ℕ := 14

/-- Theorem stating that the number of cherry pits is correct given the conditions -/
theorem cherry_pits_correct : 
  (↑cherry_pits * sprout_rate : ℚ) - saplings_sold = saplings_left := by sorry

end cherry_pits_correct_l3243_324346


namespace cube_octahedron_surface_area_ratio_l3243_324335

/-- The ratio of the surface area of a cube to the surface area of an inscribed regular octahedron --/
theorem cube_octahedron_surface_area_ratio :
  ∀ (cube_side_length : ℝ) (octahedron_side_length : ℝ),
    cube_side_length = 2 →
    octahedron_side_length = Real.sqrt 2 →
    (6 * cube_side_length^2) / (2 * Real.sqrt 3 * octahedron_side_length^2) = 2 * Real.sqrt 3 := by
  sorry

end cube_octahedron_surface_area_ratio_l3243_324335


namespace root_bounds_l3243_324318

theorem root_bounds (x : ℝ) : 
  x^2014 - 100*x + 1 = 0 → 1/100 ≤ x ∧ x ≤ 100^(1/2013) := by
  sorry

end root_bounds_l3243_324318


namespace arithmetic_sequence_length_l3243_324343

theorem arithmetic_sequence_length :
  ∀ (a₁ d l : ℝ) (n : ℕ),
    a₁ = 3.5 →
    d = 4 →
    l = 55.5 →
    l = a₁ + (n - 1) * d →
    n = 14 :=
by sorry

end arithmetic_sequence_length_l3243_324343


namespace kates_average_speed_l3243_324320

theorem kates_average_speed (bike_speed : ℝ) (bike_time : ℝ) (walk_speed : ℝ) (walk_time : ℝ) 
  (h1 : bike_speed = 20) 
  (h2 : bike_time = 45 / 60) 
  (h3 : walk_speed = 3) 
  (h4 : walk_time = 60 / 60) : 
  (bike_speed * bike_time + walk_speed * walk_time) / (bike_time + walk_time) = 10 := by
  sorry

#check kates_average_speed

end kates_average_speed_l3243_324320


namespace plate_selection_probability_l3243_324321

def red_plates : ℕ := 6
def light_blue_plates : ℕ := 3
def dark_blue_plates : ℕ := 3

def total_plates : ℕ := red_plates + light_blue_plates + dark_blue_plates

def favorable_outcomes : ℕ := 
  (red_plates.choose 2) + 
  (light_blue_plates.choose 2) + 
  (dark_blue_plates.choose 2) + 
  (light_blue_plates * dark_blue_plates)

def total_outcomes : ℕ := total_plates.choose 2

theorem plate_selection_probability : 
  (favorable_outcomes : ℚ) / total_outcomes = 5 / 11 := by sorry

end plate_selection_probability_l3243_324321


namespace expression_evaluation_l3243_324378

theorem expression_evaluation : 18 * (150 / 3 + 36 / 6 + 16 / 32 + 2) = 1053 := by
  sorry

end expression_evaluation_l3243_324378


namespace negation_of_zero_product_property_l3243_324316

theorem negation_of_zero_product_property :
  (¬ ∀ (x y : ℝ), x * y = 0 → x = 0 ∨ y = 0) ↔
  (∃ (x y : ℝ), x * y = 0 ∧ x ≠ 0 ∧ y ≠ 0) :=
by sorry

end negation_of_zero_product_property_l3243_324316


namespace abs_geq_ax_implies_a_in_range_l3243_324391

theorem abs_geq_ax_implies_a_in_range (a : ℝ) :
  (∀ x : ℝ, |x| ≥ a * x) → -1 ≤ a ∧ a ≤ 1 := by
  sorry

end abs_geq_ax_implies_a_in_range_l3243_324391


namespace february_first_is_friday_l3243_324322

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a date in February -/
structure FebruaryDate where
  day : Nat
  dayOfWeek : DayOfWeek

/-- Given that February 13th is a Wednesday, prove that February 1st is a Friday -/
theorem february_first_is_friday 
  (feb13 : FebruaryDate)
  (h13 : feb13.day = 13)
  (hWed : feb13.dayOfWeek = DayOfWeek.Wednesday) :
  ∃ (feb1 : FebruaryDate), feb1.day = 1 ∧ feb1.dayOfWeek = DayOfWeek.Friday :=
sorry

end february_first_is_friday_l3243_324322


namespace area_of_triangle_ABC_l3243_324348

-- Define the curve
def f (x : ℝ) : ℝ := x^2 - 7*x + 12

-- Define the points A, B, and C
def A : ℝ × ℝ := (3, 0)
def B : ℝ × ℝ := (4, 0)
def C : ℝ × ℝ := (0, 12)

-- Theorem statement
theorem area_of_triangle_ABC : 
  let triangle_area := (1/2) * |A.1 - B.1| * C.2
  (f A.1 = 0) ∧ (f B.1 = 0) ∧ (f 0 = C.2) → triangle_area = 6 := by
  sorry

end area_of_triangle_ABC_l3243_324348


namespace carson_saw_five_octopuses_l3243_324394

/-- The number of legs an octopus has -/
def legs_per_octopus : ℕ := 8

/-- The total number of octopus legs Carson saw -/
def total_legs : ℕ := 40

/-- The number of octopuses Carson saw -/
def num_octopuses : ℕ := total_legs / legs_per_octopus

theorem carson_saw_five_octopuses : num_octopuses = 5 := by
  sorry

end carson_saw_five_octopuses_l3243_324394


namespace expand_and_simplify_l3243_324383

theorem expand_and_simplify (x : ℝ) : (1 - x^2) * (1 + x^4 + x^6) = 1 - x^2 + x^4 - x^8 := by
  sorry

end expand_and_simplify_l3243_324383


namespace floor_equation_solution_l3243_324323

theorem floor_equation_solution (n : ℤ) : (Int.floor (n^2 / 4 : ℚ) - Int.floor (n / 2 : ℚ)^2 = 5) ↔ n = 11 := by
  sorry

end floor_equation_solution_l3243_324323


namespace subway_ride_time_l3243_324327

theorem subway_ride_time (total_time subway_time train_time bike_time : ℝ) : 
  total_time = 38 →
  train_time = 2 * subway_time →
  bike_time = 8 →
  total_time = subway_time + train_time + bike_time →
  subway_time = 10 :=
by
  sorry

end subway_ride_time_l3243_324327


namespace product_of_sum_and_difference_l3243_324359

theorem product_of_sum_and_difference (a b : ℝ) 
  (sum_eq : a + b = 3) 
  (diff_eq : a - b = 7) : 
  a * b = -10 := by sorry

end product_of_sum_and_difference_l3243_324359


namespace discount_difference_l3243_324311

def original_price : ℝ := 12000

def single_discount_rate : ℝ := 0.45
def successive_discount_rate1 : ℝ := 0.35
def successive_discount_rate2 : ℝ := 0.10

def price_after_single_discount : ℝ := original_price * (1 - single_discount_rate)
def price_after_successive_discounts : ℝ := original_price * (1 - successive_discount_rate1) * (1 - successive_discount_rate2)

theorem discount_difference :
  price_after_successive_discounts - price_after_single_discount = 420 := by
  sorry

end discount_difference_l3243_324311


namespace isosceles_triangle_perimeter_l3243_324372

/-- An isosceles triangle with sides of 4 cm and 7 cm has a perimeter of either 15 cm or 18 cm. -/
theorem isosceles_triangle_perimeter (a b c : ℝ) : 
  a = 4 ∧ b = 7 ∧ 
  ((a = b ∧ c = 7) ∨ (a = c ∧ b = 7) ∨ (b = c ∧ a = 4)) → 
  (a + b + c = 15 ∨ a + b + c = 18) :=
by sorry

end isosceles_triangle_perimeter_l3243_324372


namespace nested_sqrt_bounds_l3243_324384

theorem nested_sqrt_bounds : 
  ∃ x : ℝ, x = Real.sqrt (1 + x) ∧ 1 < x ∧ x < 2 := by
  sorry

end nested_sqrt_bounds_l3243_324384


namespace coat_drive_total_l3243_324304

theorem coat_drive_total (high_school_coats elementary_school_coats : ℕ) 
  (h1 : high_school_coats = 6922)
  (h2 : elementary_school_coats = 2515) :
  high_school_coats + elementary_school_coats = 9437 :=
by sorry

end coat_drive_total_l3243_324304


namespace planted_fraction_is_404_841_l3243_324302

/-- Represents a rectangular field with a planted right triangular area and an unplanted square --/
structure PlantedField where
  length : ℝ
  width : ℝ
  square_side : ℝ
  hypotenuse_distance : ℝ

/-- The fraction of the field that is planted --/
def planted_fraction (field : PlantedField) : ℚ :=
  sorry

/-- Theorem stating the planted fraction for the given field configuration --/
theorem planted_fraction_is_404_841 :
  let field : PlantedField := {
    length := 5,
    width := 6,
    square_side := 33 / 41,
    hypotenuse_distance := 3
  }
  planted_fraction field = 404 / 841 := by sorry

end planted_fraction_is_404_841_l3243_324302


namespace fourth_term_is_one_l3243_324397

/-- A geometric sequence with the given properties -/
structure GeometricSequence where
  a : ℕ → ℚ
  is_geometric : ∀ n : ℕ, n > 0 → ∃ q : ℚ, a (n + 1) = a n * q
  first_fifth_diff : a 1 - a 5 = -15/2
  sum_first_four : (a 1) + (a 2) + (a 3) + (a 4) = -5

/-- The fourth term of the geometric sequence is 1 -/
theorem fourth_term_is_one (seq : GeometricSequence) : seq.a 4 = 1 := by
  sorry


end fourth_term_is_one_l3243_324397


namespace negation_equivalence_l3243_324386

-- Define the set S
variable (S : Set ℝ)

-- Define the original statement
def original_statement : Prop :=
  ∀ x ∈ S, |x| ≥ 2

-- Define the negation of the original statement
def negation_statement : Prop :=
  ∃ x ∈ S, |x| < 2

-- Theorem stating the equivalence
theorem negation_equivalence :
  ¬(original_statement S) ↔ negation_statement S :=
by sorry

end negation_equivalence_l3243_324386


namespace slope_of_intersection_line_l3243_324381

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 6*x + 4*y - 15 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 16*x + 8*y + 48 = 0

-- Define the intersection points
def intersection (C D : ℝ × ℝ) : Prop :=
  circle1 C.1 C.2 ∧ circle1 D.1 D.2 ∧ circle2 C.1 C.2 ∧ circle2 D.1 D.2 ∧ C ≠ D

-- Theorem statement
theorem slope_of_intersection_line (C D : ℝ × ℝ) (h : intersection C D) : 
  (D.2 - C.2) / (D.1 - C.1) = 11/6 := by sorry

end slope_of_intersection_line_l3243_324381


namespace square_area_relation_l3243_324309

theorem square_area_relation (a b : ℝ) : 
  let diagonal_I := 2*a + 2*b
  let area_I := (diagonal_I / Real.sqrt 2)^2
  let area_II := 3 * area_I
  area_II = 6 * (a + b)^2 := by
sorry

end square_area_relation_l3243_324309


namespace simplify_polynomial_l3243_324351

theorem simplify_polynomial (x : ℝ) : (3 * x^2 + 6 * x - 5) - (2 * x^2 + 4 * x - 8) = x^2 + 2 * x + 3 := by
  sorry

end simplify_polynomial_l3243_324351


namespace function_and_monotonicity_l3243_324356

-- Define the function f(x)
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x + 1

-- Define the derivative of f(x)
def f' (a b : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + b

theorem function_and_monotonicity 
  (a b : ℝ) 
  (h1 : f a b 1 = -3)  -- f(1) = -3
  (h2 : f' a b 1 = 0)  -- f'(1) = 0
  : 
  (∀ x, f a b x = 2 * x^3 - 6 * x + 1) ∧  -- Explicit formula
  (∀ x, x < -1 → (f' a b x > 0)) ∧       -- Monotonically increasing for x < -1
  (∀ x, x > 1 → (f' a b x > 0))          -- Monotonically increasing for x > 1
  := by sorry

end function_and_monotonicity_l3243_324356


namespace board_game_investment_l3243_324307

/-- Proves that the investment in equipment for a board game business is $10,410 -/
theorem board_game_investment
  (manufacture_cost : ℝ)
  (selling_price : ℝ)
  (break_even_quantity : ℕ)
  (h1 : manufacture_cost = 2.65)
  (h2 : selling_price = 20)
  (h3 : break_even_quantity = 600)
  (h4 : selling_price * break_even_quantity = 
        manufacture_cost * break_even_quantity + investment) :
  investment = 10410 := by
  sorry

end board_game_investment_l3243_324307


namespace circle_symmetry_l3243_324342

-- Define the original circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define the line l
def line_l (x y : ℝ) : Prop := x + y = 1

-- Define the symmetric circle
def symmetric_circle (x y : ℝ) : Prop := (x - 1)^2 + (y - 1)^2 = 1

-- Theorem statement
theorem circle_symmetry :
  ∀ (x y : ℝ), circle_C x y ∧ line_l x y → symmetric_circle x y :=
by sorry

end circle_symmetry_l3243_324342


namespace smallest_m_with_divisibility_l3243_324308

def is_divisible (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

theorem smallest_m_with_divisibility : 
  (∀ M : ℕ, M > 0 ∧ M < 250 → 
    ¬(is_divisible M (5^3) ∧ is_divisible (M+1) (2^3) ∧ is_divisible (M+2) (3^2) ∨
      is_divisible M (5^3) ∧ is_divisible (M+1) (3^2) ∧ is_divisible (M+2) (2^3) ∨
      is_divisible M (2^3) ∧ is_divisible (M+1) (5^3) ∧ is_divisible (M+2) (3^2) ∨
      is_divisible M (2^3) ∧ is_divisible (M+1) (3^2) ∧ is_divisible (M+2) (5^3) ∨
      is_divisible M (3^2) ∧ is_divisible (M+1) (5^3) ∧ is_divisible (M+2) (2^3) ∨
      is_divisible M (3^2) ∧ is_divisible (M+1) (2^3) ∧ is_divisible (M+2) (5^3))) ∧
  (is_divisible 250 (5^3) ∧ is_divisible 252 (2^3) ∧ is_divisible 252 (3^2)) :=
by sorry

end smallest_m_with_divisibility_l3243_324308


namespace p_or_q_is_true_l3243_324387

-- Define proposition p
def p (a b : ℝ) : Prop := a^2 + b^2 < 0

-- Define proposition q
def q (a b : ℝ) : Prop := (a - 2)^2 + |b - 3| ≥ 0

-- Theorem statement
theorem p_or_q_is_true :
  (∀ a b : ℝ, ¬(p a b)) ∧ (∀ a b : ℝ, q a b) → ∀ a b : ℝ, p a b ∨ q a b :=
by sorry

end p_or_q_is_true_l3243_324387


namespace photo_reactions_l3243_324396

/-- 
Proves that given a photo with a starting score of 0, where "thumbs up" increases 
the score by 1 and "thumbs down" decreases it by 1, if the current score is 50 
and 75% of reactions are "thumbs up", then the total number of reactions is 100.
-/
theorem photo_reactions 
  (score : ℤ) 
  (total_reactions : ℕ) 
  (thumbs_up_ratio : ℚ) :
  score = 0 + total_reactions * thumbs_up_ratio - total_reactions * (1 - thumbs_up_ratio) →
  score = 50 →
  thumbs_up_ratio = 3/4 →
  total_reactions = 100 := by
  sorry

#check photo_reactions

end photo_reactions_l3243_324396


namespace students_left_fourth_grade_students_left_l3243_324339

theorem students_left (initial_students : ℝ) (final_students : ℝ) (transferred_students : ℝ) :
  initial_students ≥ final_students + transferred_students →
  initial_students - (final_students + transferred_students) =
  initial_students - final_students - transferred_students :=
by sorry

def calculate_students_left (initial_students : ℝ) (final_students : ℝ) (transferred_students : ℝ) : ℝ :=
  initial_students - final_students - transferred_students

theorem fourth_grade_students_left :
  let initial_students : ℝ := 42.0
  let final_students : ℝ := 28.0
  let transferred_students : ℝ := 10.0
  calculate_students_left initial_students final_students transferred_students = 4 :=
by sorry

end students_left_fourth_grade_students_left_l3243_324339
