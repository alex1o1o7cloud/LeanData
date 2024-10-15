import Mathlib

namespace NUMINAMATH_CALUDE_inequality_and_optimization_l2668_266898

theorem inequality_and_optimization (m : ℝ) :
  (∀ x : ℝ, |x + 3| + |x + m| ≥ 2*m) →
  m ≤ 1 ∧
  (∀ a b c : ℝ, a > 0 → b > 0 → c > 0 → a + b + c = 1 →
    a^2 + 2*b^2 + 3*c^2 ≥ 6/11 ∧
    (a^2 + 2*b^2 + 3*c^2 = 6/11 ↔ a = 6/11 ∧ b = 3/11 ∧ c = 2/11)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_optimization_l2668_266898


namespace NUMINAMATH_CALUDE_system_of_equations_l2668_266885

theorem system_of_equations (x y a b : ℝ) (h1 : 4 * x - 3 * y = a) (h2 : 6 * y - 8 * x = b) (h3 : b ≠ 0) :
  a / b = -1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_system_of_equations_l2668_266885


namespace NUMINAMATH_CALUDE_pet_store_animals_l2668_266874

/-- Calculates the total number of animals in a pet store given the number of dogs and ratios for other animals. -/
def total_animals (num_dogs : ℕ) : ℕ :=
  let num_cats := num_dogs / 2
  let num_birds := num_dogs * 2
  let num_fish := num_dogs * 3
  num_dogs + num_cats + num_birds + num_fish

/-- Theorem stating that a pet store with 6 dogs and specified ratios of other animals has 39 animals in total. -/
theorem pet_store_animals : total_animals 6 = 39 := by
  sorry

end NUMINAMATH_CALUDE_pet_store_animals_l2668_266874


namespace NUMINAMATH_CALUDE_min_value_fraction_l2668_266863

theorem min_value_fraction (p q r : ℝ) (hp : p > 0) (hq : q > 0) (hr : r > 0) 
  (hsum : p + q + r = 2) : 
  (p + q) / (p * q * r) ≥ 9 ∧ ∃ p q r, p > 0 ∧ q > 0 ∧ r > 0 ∧ p + q + r = 2 ∧ (p + q) / (p * q * r) = 9 :=
sorry

end NUMINAMATH_CALUDE_min_value_fraction_l2668_266863


namespace NUMINAMATH_CALUDE_line_through_intersection_and_parallel_l2668_266855

-- Define the lines l1 and l2
def l1 (x y : ℝ) : Prop := x + 3 * y - 3 = 0
def l2 (x y : ℝ) : Prop := x - y + 1 = 0

-- Define the parallel line
def parallel_line (x y : ℝ) : Prop := 2 * x + y - 3 = 0

-- Theorem statement
theorem line_through_intersection_and_parallel :
  ∃ (a b c : ℝ), 
    (∀ x y : ℝ, a * x + b * y + c = 0 ↔ 
      (∃ x0 y0 : ℝ, l1 x0 y0 ∧ l2 x0 y0 ∧ 
        (y - y0 = -(a/b) * (x - x0))) ∧
      (∃ k : ℝ, a/b = -2)) :=
sorry

end NUMINAMATH_CALUDE_line_through_intersection_and_parallel_l2668_266855


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l2668_266867

theorem quadratic_equation_roots (c : ℝ) : 
  (∀ x : ℝ, 2 * x^2 + 16 * x + c = 0 ↔ x = (-16 + Real.sqrt 24) / 4 ∨ x = (-16 - Real.sqrt 24) / 4) →
  c = 29 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l2668_266867


namespace NUMINAMATH_CALUDE_calculation_proof_l2668_266896

theorem calculation_proof : 4 * Real.sqrt 24 * (Real.sqrt 6 / 8) / Real.sqrt 3 - 3 * Real.sqrt 3 = - Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l2668_266896


namespace NUMINAMATH_CALUDE_prime_factorization_2020_2021_l2668_266804

theorem prime_factorization_2020_2021 : 
  (2020 = 2^2 * 5 * 101) ∧ (2021 = 43 * 47) := by
  sorry

end NUMINAMATH_CALUDE_prime_factorization_2020_2021_l2668_266804


namespace NUMINAMATH_CALUDE_cosine_identity_problem_l2668_266850

theorem cosine_identity_problem (α : Real) 
  (h : Real.cos (π / 4 + α) = -1 / 3) : 
  (Real.sin (2 * α) - 2 * Real.sin α ^ 2) / Real.sqrt (1 - Real.cos (2 * α)) = 2 / 3 ∨ 
  (Real.sin (2 * α) - 2 * Real.sin α ^ 2) / Real.sqrt (1 - Real.cos (2 * α)) = -2 / 3 :=
sorry

end NUMINAMATH_CALUDE_cosine_identity_problem_l2668_266850


namespace NUMINAMATH_CALUDE_det_E_l2668_266882

/-- A 2x2 matrix representing a dilation centered at the origin with scale factor 5 -/
def E : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![5, 0],
    ![0, 5]]

/-- Theorem stating that the determinant of E is 25 -/
theorem det_E : Matrix.det E = 25 := by
  sorry

end NUMINAMATH_CALUDE_det_E_l2668_266882


namespace NUMINAMATH_CALUDE_banana_arrangement_count_l2668_266837

/-- The number of unique arrangements of the letters in "BANANA" -/
def banana_arrangements : ℕ := 60

/-- The total number of letters in "BANANA" -/
def total_letters : ℕ := 6

/-- The number of A's in "BANANA" -/
def num_a : ℕ := 3

/-- The number of N's in "BANANA" -/
def num_n : ℕ := 2

/-- The number of B's in "BANANA" -/
def num_b : ℕ := 1

/-- Theorem stating that the number of unique arrangements of the letters in "BANANA" is 60 -/
theorem banana_arrangement_count :
  banana_arrangements = (Nat.factorial total_letters) / ((Nat.factorial num_a) * (Nat.factorial num_n)) :=
sorry

end NUMINAMATH_CALUDE_banana_arrangement_count_l2668_266837


namespace NUMINAMATH_CALUDE_seven_people_seven_rooms_l2668_266880

/-- The number of ways to assign n people to m rooms with at most k people per room -/
def assignmentCount (n m k : ℕ) : ℕ := sorry

/-- Theorem stating that there are 131460 ways to assign 7 people to 7 rooms with at most 2 people per room -/
theorem seven_people_seven_rooms : assignmentCount 7 7 2 = 131460 := by sorry

end NUMINAMATH_CALUDE_seven_people_seven_rooms_l2668_266880


namespace NUMINAMATH_CALUDE_money_spending_l2668_266821

theorem money_spending (M : ℚ) : 
  (2 / 7 : ℚ) * M = 500 →
  M = 1750 := by
  sorry

end NUMINAMATH_CALUDE_money_spending_l2668_266821


namespace NUMINAMATH_CALUDE_smaller_number_proof_l2668_266877

theorem smaller_number_proof (x y : ℝ) : 
  x - y = 9 → x + y = 46 → min x y = 18.5 := by
sorry

end NUMINAMATH_CALUDE_smaller_number_proof_l2668_266877


namespace NUMINAMATH_CALUDE_polynomial_division_l2668_266816

-- Define the polynomial that can be divided by x^2 + 3x - 4
def is_divisible (a b c : ℝ) : Prop :=
  ∃ (q : ℝ → ℝ), ∀ x, x^3 + a*x^2 + b*x + c = (x^2 + 3*x - 4) * q x

-- Main theorem
theorem polynomial_division (a b c : ℝ) 
  (h : is_divisible a b c) : 
  (4*a + c = 12) ∧ 
  (2*a - 2*b - c = 14) ∧ 
  (∀ (a' b' c' : ℤ), (is_divisible (a' : ℝ) (b' : ℝ) (c' : ℝ)) → 
    c' ≥ a' ∧ a' > 1 → a' = 2 ∧ b' = -7 ∧ c' = 4) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_division_l2668_266816


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l2668_266895

/-- Given an ellipse and a hyperbola with related equations, prove that the hyperbola's eccentricity is √6/2 -/
theorem hyperbola_eccentricity
  (m n : ℝ)
  (h_pos : 0 < m ∧ m < n)
  (h_ellipse : ∀ x y : ℝ, m * x^2 + n * y^2 = 1)
  (h_ellipse_ecc : Real.sqrt 2 / 2 = Real.sqrt (1 - (1/n) / (1/m)))
  (h_hyperbola : ∀ x y : ℝ, m * x^2 - n * y^2 = 1) :
  Real.sqrt 6 / 2 = Real.sqrt (1 + (1/n) / (1/m)) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l2668_266895


namespace NUMINAMATH_CALUDE_cookies_and_game_cost_l2668_266870

-- Define the quantities of each item
def bracelets : ℕ := 12
def necklaces : ℕ := 8
def rings : ℕ := 20

-- Define the costs to make each item
def bracelet_cost : ℚ := 1
def necklace_cost : ℚ := 2
def ring_cost : ℚ := 1/2

-- Define the selling prices of each item
def bracelet_price : ℚ := 3/2
def necklace_price : ℚ := 3
def ring_price : ℚ := 1

-- Define the target profit margin
def target_margin : ℚ := 1/2

-- Define the remaining money after purchases
def remaining_money : ℚ := 5

-- Theorem to prove
theorem cookies_and_game_cost :
  let total_cost := bracelets * bracelet_cost + necklaces * necklace_cost + rings * ring_cost
  let total_revenue := bracelets * bracelet_price + necklaces * necklace_price + rings * ring_price
  let profit := total_revenue - total_cost
  let target_profit := total_cost * target_margin
  let cost_of_purchases := profit - remaining_money
  cost_of_purchases = 43 := by sorry

end NUMINAMATH_CALUDE_cookies_and_game_cost_l2668_266870


namespace NUMINAMATH_CALUDE_semicircle_limit_l2668_266803

/-- The limit of the sum of areas of n semicircles constructed on equal parts 
    of a circle's diameter approaches 0 as n approaches infinity. -/
theorem semicircle_limit (D : ℝ) (h : D > 0) : 
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, (π * D^2) / (8 * n) < ε := by
sorry

end NUMINAMATH_CALUDE_semicircle_limit_l2668_266803


namespace NUMINAMATH_CALUDE_distance_XY_is_16_l2668_266842

-- Define the travel parameters
def travel_time_A : ℕ → Prop := λ t => t * t = 16

def travel_time_B : ℕ → Prop := λ t => 
  ∃ (rest : ℕ), t = 11 ∧ 2 * (t - rest) = 16 ∧ 4 * rest < 16 ∧ 4 * rest + 4 ≥ 16

-- Theorem statement
theorem distance_XY_is_16 : 
  (∃ t : ℕ, travel_time_A t ∧ travel_time_B t) → 
  (∃ d : ℕ, d = 16 ∧ ∀ t : ℕ, travel_time_A t → t * t = d) :=
by
  sorry

end NUMINAMATH_CALUDE_distance_XY_is_16_l2668_266842


namespace NUMINAMATH_CALUDE_greatest_sum_consecutive_integers_l2668_266868

theorem greatest_sum_consecutive_integers (n : ℕ) : 
  (n * (n + 1) < 500 ∧ (n + 1) * (n + 2) ≥ 500) → n + (n + 1) = 43 := by
  sorry

end NUMINAMATH_CALUDE_greatest_sum_consecutive_integers_l2668_266868


namespace NUMINAMATH_CALUDE_sum_due_calculation_l2668_266856

/-- Represents the relationship between banker's discount, true discount, and face value -/
def banker_discount_relation (banker_discount true_discount face_value : ℚ) : Prop :=
  banker_discount = true_discount + (true_discount * banker_discount) / face_value

/-- Proves that given a banker's discount of 576 and a true discount of 480, the sum due (face value) is 2880 -/
theorem sum_due_calculation (banker_discount true_discount : ℚ) 
  (h1 : banker_discount = 576)
  (h2 : true_discount = 480) :
  ∃ face_value : ℚ, face_value = 2880 ∧ banker_discount_relation banker_discount true_discount face_value :=
by
  sorry

end NUMINAMATH_CALUDE_sum_due_calculation_l2668_266856


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l2668_266857

theorem geometric_sequence_problem (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = q * a n) →  -- geometric sequence condition
  a 1 + a 2 + a 3 + a 4 + a 5 = 3 →
  a 1^2 + a 2^2 + a 3^2 + a 4^2 + a 5^2 = 15 →
  a 1 - a 2 + a 3 - a 4 + a 5 = 5 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l2668_266857


namespace NUMINAMATH_CALUDE_perpendicular_vector_t_value_l2668_266846

/-- Given vectors a and b, if a is perpendicular to (t*a + b), then t = -5 -/
theorem perpendicular_vector_t_value (a b : ℝ × ℝ) (t : ℝ) 
  (h1 : a = (1, -1))
  (h2 : b = (6, -4))
  (h3 : a.1 * (t * a.1 + b.1) + a.2 * (t * a.2 + b.2) = 0) :
  t = -5 := by sorry

end NUMINAMATH_CALUDE_perpendicular_vector_t_value_l2668_266846


namespace NUMINAMATH_CALUDE_cindys_calculation_l2668_266845

theorem cindys_calculation (x : ℝ) : (x - 8) / 4 = 24 → (x - 4) / 8 = 12.5 := by
  sorry

end NUMINAMATH_CALUDE_cindys_calculation_l2668_266845


namespace NUMINAMATH_CALUDE_piggy_bank_coins_l2668_266824

theorem piggy_bank_coins (nickels : ℕ) (dimes : ℕ) (quarters : ℕ) : 
  dimes = 2 * nickels →
  quarters = dimes / 2 →
  5 * nickels + 10 * dimes + 25 * quarters = 1950 →
  nickels = 39 := by
sorry

end NUMINAMATH_CALUDE_piggy_bank_coins_l2668_266824


namespace NUMINAMATH_CALUDE_set_operation_result_l2668_266872

open Set

def U : Set Int := univ
def A : Set Int := {-2, -1, 0, 1, 2}
def B : Set Int := {-1, 0, 1, 2, 3}

theorem set_operation_result : A ∩ (U \ B) = {-2} := by
  sorry

end NUMINAMATH_CALUDE_set_operation_result_l2668_266872


namespace NUMINAMATH_CALUDE_exists_unique_decomposition_l2668_266841

def sequence_decomposition (x : ℕ → ℝ) : Prop :=
  ∃! (y z : ℕ → ℝ), 
    (∀ n : ℕ, x n = y n - z n) ∧
    (∀ n : ℕ, y n ≥ 0) ∧
    (∀ n : ℕ, n > 0 → z n ≥ z (n-1)) ∧
    (∀ n : ℕ, n > 0 → y n * (z n - z (n-1)) = 0) ∧
    (z 0 = 0)

theorem exists_unique_decomposition (x : ℕ → ℝ) : sequence_decomposition x := by
  sorry

end NUMINAMATH_CALUDE_exists_unique_decomposition_l2668_266841


namespace NUMINAMATH_CALUDE_two_number_difference_l2668_266802

theorem two_number_difference (a b : ℕ) (h1 : a = 10 * b) (h2 : a + b = 17402) : 
  a - b = 14238 := by
sorry

end NUMINAMATH_CALUDE_two_number_difference_l2668_266802


namespace NUMINAMATH_CALUDE_gcd_of_numbers_l2668_266827

theorem gcd_of_numbers : Nat.gcd 128 (Nat.gcd 144 (Nat.gcd 480 450)) = 6 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_numbers_l2668_266827


namespace NUMINAMATH_CALUDE_smallest_c_for_g_range_five_l2668_266823

/-- The function g(x) defined in the problem -/
def g (c : ℝ) (x : ℝ) : ℝ := 2 * x^2 - 4 * x + c

/-- Theorem stating that 7 is the smallest value of c such that 5 is in the range of g(x) -/
theorem smallest_c_for_g_range_five :
  ∀ c : ℝ, (∃ x : ℝ, g c x = 5) ↔ c ≥ 7 :=
by sorry

end NUMINAMATH_CALUDE_smallest_c_for_g_range_five_l2668_266823


namespace NUMINAMATH_CALUDE_gcd_48576_34650_l2668_266825

theorem gcd_48576_34650 : Nat.gcd 48576 34650 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_48576_34650_l2668_266825


namespace NUMINAMATH_CALUDE_sin_negative_thirty_degrees_l2668_266840

theorem sin_negative_thirty_degrees : 
  Real.sin (-(30 * π / 180)) = -(1 / 2) := by
  sorry

end NUMINAMATH_CALUDE_sin_negative_thirty_degrees_l2668_266840


namespace NUMINAMATH_CALUDE_factor_expression_l2668_266879

theorem factor_expression (y : ℝ) : 16 * y^2 + 8 * y = 8 * y * (2 * y + 1) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l2668_266879


namespace NUMINAMATH_CALUDE_tonys_fever_l2668_266814

theorem tonys_fever (normal_temp : ℝ) (temp_increase : ℝ) (fever_threshold : ℝ)
  (h1 : normal_temp = 95)
  (h2 : temp_increase = 10)
  (h3 : fever_threshold = 100) :
  normal_temp + temp_increase - fever_threshold = 5 :=
by sorry

end NUMINAMATH_CALUDE_tonys_fever_l2668_266814


namespace NUMINAMATH_CALUDE_special_die_probability_sum_l2668_266851

/-- Represents a die with special probability distribution -/
structure SpecialDie where
  /-- Probability of rolling an odd number -/
  odd_prob : ℚ
  /-- Probability of rolling an even number -/
  even_prob : ℚ
  /-- Ensure even probability is twice odd probability -/
  even_twice_odd : even_prob = 2 * odd_prob
  /-- Ensure total probability is 1 -/
  total_prob_one : 3 * odd_prob + 3 * even_prob = 1

/-- Calculates the probability of rolling 1, 2, or 3 on the special die -/
def prob_not_exceeding_three (d : SpecialDie) : ℚ :=
  2 * d.odd_prob + d.even_prob

/-- The main theorem stating the sum of numerator and denominator is 13 -/
theorem special_die_probability_sum : 
  ∀ (d : SpecialDie), 
  let p := prob_not_exceeding_three d
  let n := p.den
  let m := p.num
  m + n = 13 := by sorry

end NUMINAMATH_CALUDE_special_die_probability_sum_l2668_266851


namespace NUMINAMATH_CALUDE_park_outer_diameter_l2668_266887

/-- Given a circular park with a central fountain, surrounded by a garden ring and a walking path,
    this theorem proves the diameter of the outer boundary of the walking path. -/
theorem park_outer_diameter
  (fountain_diameter : ℝ)
  (garden_width : ℝ)
  (path_width : ℝ)
  (h1 : fountain_diameter = 20)
  (h2 : garden_width = 10)
  (h3 : path_width = 6) :
  2 * (fountain_diameter / 2 + garden_width + path_width) = 52 :=
by sorry

end NUMINAMATH_CALUDE_park_outer_diameter_l2668_266887


namespace NUMINAMATH_CALUDE_least_common_denominator_of_fractions_l2668_266871

theorem least_common_denominator_of_fractions : Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 7)))) = 420 := by
  sorry

end NUMINAMATH_CALUDE_least_common_denominator_of_fractions_l2668_266871


namespace NUMINAMATH_CALUDE_min_value_theorem_l2668_266853

theorem min_value_theorem (n : ℕ+) : 
  (n : ℝ) / 3 + 27 / (n : ℝ) ≥ 6 ∧ 
  ((n : ℝ) / 3 + 27 / (n : ℝ) = 6 ↔ n = 9) := by
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2668_266853


namespace NUMINAMATH_CALUDE_probability_A_timeliness_at_least_75_l2668_266881

/-- Represents the survey data for a company -/
structure SurveyData where
  total_questionnaires : ℕ
  excellent_timeliness : ℕ
  good_timeliness : ℕ
  fair_timeliness : ℕ

/-- Calculates the probability of timeliness rating at least 75 points -/
def probabilityAtLeast75 (data : SurveyData) : ℚ :=
  (data.excellent_timeliness + data.good_timeliness : ℚ) / data.total_questionnaires

/-- The survey data for company A -/
def companyA : SurveyData := {
  total_questionnaires := 120,
  excellent_timeliness := 29,
  good_timeliness := 47,
  fair_timeliness := 44
}

/-- Theorem stating the probability of company A's delivery timeliness being at least 75 points -/
theorem probability_A_timeliness_at_least_75 :
  probabilityAtLeast75 companyA = 19 / 30 := by sorry

end NUMINAMATH_CALUDE_probability_A_timeliness_at_least_75_l2668_266881


namespace NUMINAMATH_CALUDE_junk_mail_distribution_l2668_266808

theorem junk_mail_distribution (blocks : ℕ) (pieces_per_block : ℕ) (h1 : blocks = 4) (h2 : pieces_per_block = 48) :
  blocks * pieces_per_block = 192 := by
  sorry

end NUMINAMATH_CALUDE_junk_mail_distribution_l2668_266808


namespace NUMINAMATH_CALUDE_algebra_test_average_l2668_266891

theorem algebra_test_average (total_average : ℝ) (male_average : ℝ) (male_count : ℕ) (female_count : ℕ) 
  (h1 : total_average = 90)
  (h2 : male_average = 87)
  (h3 : male_count = 8)
  (h4 : female_count = 12) :
  let total_count := male_count + female_count
  let total_score := total_average * total_count
  let male_score := male_average * male_count
  let female_score := total_score - male_score
  female_score / female_count = 92 := by
sorry

end NUMINAMATH_CALUDE_algebra_test_average_l2668_266891


namespace NUMINAMATH_CALUDE_new_pages_read_per_week_jim_new_pages_read_l2668_266894

/-- Calculates the new number of pages read per week after changes in reading speed and time --/
theorem new_pages_read_per_week
  (initial_rate : ℝ)
  (initial_pages : ℝ)
  (speed_increase : ℝ)
  (time_decrease : ℝ)
  (h1 : initial_rate = 40)
  (h2 : initial_pages = 600)
  (h3 : speed_increase = 1.5)
  (h4 : time_decrease = 4)
  : ℝ :=
  by
  -- Proof goes here
  sorry

/-- The main theorem stating that Jim now reads 660 pages per week --/
theorem jim_new_pages_read :
  new_pages_read_per_week 40 600 1.5 4 rfl rfl rfl rfl = 660 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_new_pages_read_per_week_jim_new_pages_read_l2668_266894


namespace NUMINAMATH_CALUDE_max_y_coordinate_l2668_266875

theorem max_y_coordinate (x y : ℝ) : 
  (x^2 / 49) + ((y - 3)^2 / 25) = 0 → y ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_max_y_coordinate_l2668_266875


namespace NUMINAMATH_CALUDE_average_speed_calculation_l2668_266839

theorem average_speed_calculation (speed1 speed2 : ℝ) (h1 : speed1 = 70) (h2 : speed2 = 90) :
  (speed1 + speed2) / 2 = 80 := by
  sorry

end NUMINAMATH_CALUDE_average_speed_calculation_l2668_266839


namespace NUMINAMATH_CALUDE_bobby_candy_consumption_l2668_266810

/-- The number of candies Bobby eats per day during weekdays -/
def weekday_candies : ℕ := 2

/-- The number of candies Bobby eats per day during weekends -/
def weekend_candies : ℕ := 1

/-- The number of weekdays in a week -/
def weekdays : ℕ := 5

/-- The number of weekend days in a week -/
def weekend_days : ℕ := 2

/-- The number of weeks it takes Bobby to finish the packets -/
def weeks : ℕ := 3

/-- The number of packets Bobby buys -/
def packets : ℕ := 2

/-- The number of candies in a packet -/
def candies_per_packet : ℕ := 18

theorem bobby_candy_consumption :
  weekday_candies * weekdays * weeks +
  weekend_candies * weekend_days * weeks =
  candies_per_packet * packets := by
  sorry

end NUMINAMATH_CALUDE_bobby_candy_consumption_l2668_266810


namespace NUMINAMATH_CALUDE_salary_increase_percentage_l2668_266805

theorem salary_increase_percentage (original_salary : ℝ) (h : original_salary > 0) : 
  let decreased_salary := 0.5 * original_salary
  let final_salary := 0.75 * original_salary
  ∃ P : ℝ, decreased_salary * (1 + P) = final_salary ∧ P = 0.5 :=
by sorry

end NUMINAMATH_CALUDE_salary_increase_percentage_l2668_266805


namespace NUMINAMATH_CALUDE_average_weight_BCDE_l2668_266828

/-- Given the weights of individuals A, B, C, D, and E, prove that the average weight of B, C, D, and E is 51 kg. -/
theorem average_weight_BCDE (w_A w_B w_C w_D w_E : ℝ) : 
  (w_A + w_B + w_C) / 3 = 50 →
  (w_A + w_B + w_C + w_D) / 4 = 53 →
  w_E = w_D + 3 →
  w_A = 73 →
  (w_B + w_C + w_D + w_E) / 4 = 51 := by
sorry

end NUMINAMATH_CALUDE_average_weight_BCDE_l2668_266828


namespace NUMINAMATH_CALUDE_common_factor_proof_l2668_266886

theorem common_factor_proof (n : ℤ) : ∃ (k₁ k₂ : ℤ), 
  n^2 - 1 = (n + 1) * k₁ ∧ n^2 + n = (n + 1) * k₂ := by
  sorry

end NUMINAMATH_CALUDE_common_factor_proof_l2668_266886


namespace NUMINAMATH_CALUDE_abs_negative_two_l2668_266861

theorem abs_negative_two : |(-2 : ℤ)| = 2 := by
  sorry

end NUMINAMATH_CALUDE_abs_negative_two_l2668_266861


namespace NUMINAMATH_CALUDE_no_natural_n_for_sum_of_squares_l2668_266807

theorem no_natural_n_for_sum_of_squares : 
  ¬ ∃ (n : ℕ), ∃ (x y : ℕ+), 
    2 * n * (n + 1) * (n + 2) * (n + 3) + 12 = x^2 + y^2 := by
  sorry

end NUMINAMATH_CALUDE_no_natural_n_for_sum_of_squares_l2668_266807


namespace NUMINAMATH_CALUDE_credit_limit_problem_l2668_266892

/-- The credit limit problem -/
theorem credit_limit_problem (payments_made : ℕ) (remaining_payment : ℕ) 
  (h1 : payments_made = 38)
  (h2 : remaining_payment = 62) :
  payments_made + remaining_payment = 100 := by
  sorry

end NUMINAMATH_CALUDE_credit_limit_problem_l2668_266892


namespace NUMINAMATH_CALUDE_hash_difference_l2668_266813

-- Define the # operation
def hash (x y : ℤ) : ℤ := x * y - x - 2 * y

-- State the theorem
theorem hash_difference : (hash 6 4) - (hash 4 6) = 2 := by sorry

end NUMINAMATH_CALUDE_hash_difference_l2668_266813


namespace NUMINAMATH_CALUDE_base_10_to_7_395_l2668_266815

/-- Converts a natural number from base 10 to base 7 -/
def toBase7 (n : ℕ) : List ℕ :=
  sorry

theorem base_10_to_7_395 :
  toBase7 395 = [1, 1, 0, 3] :=
sorry

end NUMINAMATH_CALUDE_base_10_to_7_395_l2668_266815


namespace NUMINAMATH_CALUDE_triangle_max_area_l2668_266829

theorem triangle_max_area (A B C : ℝ) (a b c : ℝ) : 
  A = (2 * π) / 3 →
  b + 2 * c = 8 →
  0 < a ∧ 0 < b ∧ 0 < c →
  (∀ b' c' : ℝ, b' + 2 * c' = 8 → 
    b' * c' * Real.sin A ≤ b * c * Real.sin A) →
  a^2 = b^2 + c^2 - 2 * b * c * Real.cos A →
  a = 2 * Real.sqrt 7 := by
sorry

end NUMINAMATH_CALUDE_triangle_max_area_l2668_266829


namespace NUMINAMATH_CALUDE_optimal_system_is_best_l2668_266897

/-- Represents a monetary system with three coin denominations -/
structure MonetarySystem where
  d1 : ℕ
  d2 : ℕ
  d3 : ℕ
  h1 : 0 < d1 ∧ d1 < d2 ∧ d2 < d3
  h2 : d3 ≤ 100

/-- Calculates the minimum number of coins required for a given monetary system -/
def minCoinsRequired (system : MonetarySystem) : ℕ := sorry

/-- The optimal monetary system -/
def optimalSystem : MonetarySystem :=
  { d1 := 1, d2 := 7, d3 := 14,
    h1 := by simp,
    h2 := by simp }

theorem optimal_system_is_best :
  (∀ system : MonetarySystem, minCoinsRequired system ≥ minCoinsRequired optimalSystem) ∧
  minCoinsRequired optimalSystem = 14 := by sorry

end NUMINAMATH_CALUDE_optimal_system_is_best_l2668_266897


namespace NUMINAMATH_CALUDE_integer_list_mean_mode_relation_l2668_266822

theorem integer_list_mean_mode_relation : 
  ∀ x : ℕ, 
  x ≤ 100 → 
  x > 0 →
  let list := [20, x, x, x, x]
  let mean := (20 + 4 * x) / 5
  let mode := x
  mean = 2 * mode → 
  x = 10 := by
sorry

end NUMINAMATH_CALUDE_integer_list_mean_mode_relation_l2668_266822


namespace NUMINAMATH_CALUDE_lipstick_ratio_l2668_266878

def lipstick_problem (total_students : ℕ) (blue_lipstick : ℕ) : Prop :=
  let red_lipstick := blue_lipstick * 5
  let colored_lipstick := red_lipstick * 4
  colored_lipstick * 2 = total_students

theorem lipstick_ratio :
  lipstick_problem 200 5 :=
sorry

end NUMINAMATH_CALUDE_lipstick_ratio_l2668_266878


namespace NUMINAMATH_CALUDE_apple_weight_probability_l2668_266836

theorem apple_weight_probability (p_less_200 p_more_300 : ℝ) 
  (h1 : p_less_200 = 0.10)
  (h2 : p_more_300 = 0.12) :
  1 - p_less_200 - p_more_300 = 0.78 := by
  sorry

end NUMINAMATH_CALUDE_apple_weight_probability_l2668_266836


namespace NUMINAMATH_CALUDE_circle_radius_l2668_266862

/-- The radius of a circle given by the equation x^2 + 10x + y^2 - 8y + 25 = 0 is 4 -/
theorem circle_radius (x y : ℝ) : x^2 + 10*x + y^2 - 8*y + 25 = 0 → ∃ (h k : ℝ), (x - h)^2 + (y - k)^2 = 4^2 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_l2668_266862


namespace NUMINAMATH_CALUDE_number_line_properties_l2668_266811

-- Definition of distance between points on a number line
def distance (a b : ℚ) : ℚ := |a - b|

-- Statements to prove
theorem number_line_properties :
  -- 1. The distance between 2 and 5 is 3
  distance 2 5 = 3 ∧
  -- 2. The distance between x and -6 is |x + 6|
  ∀ x : ℚ, distance x (-6) = |x + 6| ∧
  -- 3. For -2 < x < 2, |x-2|+|x+2| = 4
  ∀ x : ℚ, -2 < x → x < 2 → |x-2|+|x+2| = 4 ∧
  -- 4. For |x-1|+|x+3| > 4, x > 1 or x < -3
  ∀ x : ℚ, |x-1|+|x+3| > 4 → x > 1 ∨ x < -3 ∧
  -- 5. The minimum value of |x-3|+|x+2|+|x+1| is 5, occurring at x = -1
  (∀ x : ℚ, |x-3|+|x+2|+|x+1| ≥ 5) ∧ (|-1-3|+|-1+2|+|-1+1| = 5) ∧
  -- 6. The maximum value of y when |x-1|+|x+2|=10-|y-3|-|y+4| is 3
  ∀ x y : ℚ, |x-1|+|x+2| = 10-|y-3|-|y+4| → y ≤ 3 :=
by sorry

end NUMINAMATH_CALUDE_number_line_properties_l2668_266811


namespace NUMINAMATH_CALUDE_min_sum_of_product_3920_l2668_266865

theorem min_sum_of_product_3920 (x y z : ℕ+) (h : x * y * z = 3920) :
  ∃ (a b c : ℕ+), a * b * c = 3920 ∧ (∀ x' y' z' : ℕ+, x' * y' * z' = 3920 → a + b + c ≤ x' + y' + z') ∧ a + b + c = 70 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_of_product_3920_l2668_266865


namespace NUMINAMATH_CALUDE_inequality_proof_l2668_266817

theorem inequality_proof (a b c : ℝ) (h1 : c > b) (h2 : b > a) :
  a^2*b + b^2*c + c^2*a < a*b^2 + b*c^2 + c*a^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2668_266817


namespace NUMINAMATH_CALUDE_pet_store_birds_l2668_266848

theorem pet_store_birds (num_cages : ℕ) (parrots_per_cage : ℕ) (parakeets_per_cage : ℕ)
  (h1 : num_cages = 9)
  (h2 : parrots_per_cage = 2)
  (h3 : parakeets_per_cage = 2) :
  num_cages * (parrots_per_cage + parakeets_per_cage) = 36 := by
  sorry

end NUMINAMATH_CALUDE_pet_store_birds_l2668_266848


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l2668_266869

/-- Given a hyperbola with equation y²/2 - x²/8 = 1, its eccentricity is √5 -/
theorem hyperbola_eccentricity :
  ∀ (x y : ℝ), y^2/2 - x^2/8 = 1 → 
  ∃ (e : ℝ), e = Real.sqrt 5 ∧ e = Real.sqrt ((2 + 8) / 2) := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l2668_266869


namespace NUMINAMATH_CALUDE_candy_bar_sales_theorem_l2668_266854

/-- Calculates the total money earned from candy bar sales given the number of members,
    average number of candy bars sold per member, and the cost per candy bar. -/
def total_money_earned (num_members : ℕ) (avg_bars_per_member : ℕ) (cost_per_bar : ℚ) : ℚ :=
  (num_members * avg_bars_per_member : ℚ) * cost_per_bar

/-- Proves that a group of 20 members selling an average of 8 candy bars at $0.50 each
    earns a total of $80 from their sales. -/
theorem candy_bar_sales_theorem :
  total_money_earned 20 8 (1/2) = 80 := by
  sorry

end NUMINAMATH_CALUDE_candy_bar_sales_theorem_l2668_266854


namespace NUMINAMATH_CALUDE_M_intersect_N_l2668_266820

/-- The set M defined by the condition √x < 4 -/
def M : Set ℝ := {x | Real.sqrt x < 4}

/-- The set N defined by the condition 3x ≥ 1 -/
def N : Set ℝ := {x | 3 * x ≥ 1}

/-- The intersection of sets M and N -/
theorem M_intersect_N : M ∩ N = {x : ℝ | 1/3 ≤ x ∧ x < 16} := by sorry

end NUMINAMATH_CALUDE_M_intersect_N_l2668_266820


namespace NUMINAMATH_CALUDE_new_person_weight_l2668_266883

/-- Given a group of 8 people, if replacing one person weighing 65 kg
    with a new person increases the average weight by 3.5 kg,
    then the weight of the new person is 93 kg. -/
theorem new_person_weight
  (initial_count : Nat)
  (weight_increase : ℝ)
  (replaced_weight : ℝ)
  (h1 : initial_count = 8)
  (h2 : weight_increase = 3.5)
  (h3 : replaced_weight = 65)
  : ℝ :=
by
  sorry

end NUMINAMATH_CALUDE_new_person_weight_l2668_266883


namespace NUMINAMATH_CALUDE_hcf_problem_l2668_266838

theorem hcf_problem (a b : ℕ+) (h1 : a * b = 2560) (h2 : Nat.lcm a b = 128) :
  Nat.gcd a b = 20 := by
  sorry

end NUMINAMATH_CALUDE_hcf_problem_l2668_266838


namespace NUMINAMATH_CALUDE_circle_line_intersection_l2668_266801

-- Define the circle equation
def circle_eq (x y a : ℝ) : Prop := x^2 + y^2 + 2*x - 2*y + 2*a = 0

-- Define the line equation
def line_eq (x y : ℝ) : Prop := x + y + 2 = 0

-- Define the intersection of the circle and the line
def intersection (a : ℝ) : Prop := ∃ x y : ℝ, circle_eq x y a ∧ line_eq x y

-- Define the chord length
def chord_length (a : ℝ) : ℝ := 4

-- Theorem statement
theorem circle_line_intersection (a : ℝ) : 
  intersection a ∧ chord_length a = 4 → a = -2 := by
  sorry

end NUMINAMATH_CALUDE_circle_line_intersection_l2668_266801


namespace NUMINAMATH_CALUDE_parabola_directrix_l2668_266833

/-- Given a parabola with equation y = 4x^2, its directrix has equation y = -1/16 -/
theorem parabola_directrix (x y : ℝ) : 
  (y = 4 * x^2) → (∃ p : ℝ, p > 0 ∧ y = (1 / (4 * p)) * x^2 ∧ -1 / (4 * p) = -1/16) :=
by sorry

end NUMINAMATH_CALUDE_parabola_directrix_l2668_266833


namespace NUMINAMATH_CALUDE_estimate_excellent_scores_result_l2668_266831

/-- Estimates the number of excellent scores in a population based on a sample. -/
def estimate_excellent_scores (total_population : ℕ) (sample_size : ℕ) (excellent_in_sample : ℕ) : ℕ :=
  (total_population * excellent_in_sample) / sample_size

/-- Theorem stating that the estimated number of excellent scores is 152 given the problem conditions. -/
theorem estimate_excellent_scores_result :
  estimate_excellent_scores 380 50 20 = 152 := by
  sorry

end NUMINAMATH_CALUDE_estimate_excellent_scores_result_l2668_266831


namespace NUMINAMATH_CALUDE_fishing_tournament_l2668_266888

theorem fishing_tournament (jacob_initial : ℕ) : 
  (7 * jacob_initial - 23 = jacob_initial + 26 - 1) → jacob_initial = 8 := by sorry

end NUMINAMATH_CALUDE_fishing_tournament_l2668_266888


namespace NUMINAMATH_CALUDE_forty_five_candies_cost_candies_for_fifty_l2668_266890

-- Define the cost of one candy in rubles
def cost_per_candy : ℝ := 1

-- Define the relationship between 45 candies and their cost
theorem forty_five_candies_cost (c : ℝ) : c * 45 = 45 := by sorry

-- Define the number of candies that can be bought for 20 rubles
def candies_for_twenty : ℝ := 20

-- Theorem to prove
theorem candies_for_fifty : ℝ := by
  -- The number of candies that can be bought for 50 rubles is 50
  exact 50

/- Proof
sorry
-/

end NUMINAMATH_CALUDE_forty_five_candies_cost_candies_for_fifty_l2668_266890


namespace NUMINAMATH_CALUDE_decision_box_has_two_exits_l2668_266843

/-- Represents a decision box in a program flowchart -/
structure DecisionBox where
  entrance : Nat
  exits : Nat

/-- Represents a flowchart -/
structure Flowchart where
  endpoints : Nat

/-- Theorem: A decision box in a program flowchart has exactly 2 exits -/
theorem decision_box_has_two_exits (d : DecisionBox) (f : Flowchart) : 
  d.entrance = 1 ∧ f.endpoints ≥ 1 → d.exits = 2 := by
  sorry

end NUMINAMATH_CALUDE_decision_box_has_two_exits_l2668_266843


namespace NUMINAMATH_CALUDE_monotone_increasing_range_l2668_266864

/-- A function g(x) = ax³ + ax² + x is monotonically increasing on ℝ -/
def is_monotone_increasing (a : ℝ) : Prop :=
  ∀ x y : ℝ, x < y → (a * x^3 + a * x^2 + x) < (a * y^3 + a * y^2 + y)

/-- The range of a for which g(x) = ax³ + ax² + x is monotonically increasing on ℝ -/
theorem monotone_increasing_range :
  ∀ a : ℝ, is_monotone_increasing a ↔ (0 ≤ a ∧ a ≤ 3) :=
by sorry

end NUMINAMATH_CALUDE_monotone_increasing_range_l2668_266864


namespace NUMINAMATH_CALUDE_bijection_iteration_fixed_point_l2668_266866

theorem bijection_iteration_fixed_point {n : ℕ} (f : Fin n → Fin n) (h : Function.Bijective f) :
  ∃ M : ℕ+, ∀ i : Fin n, (f^[M.val] i) = f i := by
  sorry

end NUMINAMATH_CALUDE_bijection_iteration_fixed_point_l2668_266866


namespace NUMINAMATH_CALUDE_stars_per_jar_l2668_266800

theorem stars_per_jar (stars_made : ℕ) (bottles_to_fill : ℕ) (stars_to_make : ℕ) : 
  stars_made = 33 →
  bottles_to_fill = 4 →
  stars_to_make = 307 →
  (stars_made + stars_to_make) / bottles_to_fill = 85 :=
by
  sorry

end NUMINAMATH_CALUDE_stars_per_jar_l2668_266800


namespace NUMINAMATH_CALUDE_chips_per_console_is_five_l2668_266830

/-- The number of computer chips created per day -/
def chips_per_day : ℕ := 467

/-- The number of video game consoles created per day -/
def consoles_per_day : ℕ := 93

/-- The number of computer chips needed per console -/
def chips_per_console : ℕ := chips_per_day / consoles_per_day

theorem chips_per_console_is_five : chips_per_console = 5 := by
  sorry

end NUMINAMATH_CALUDE_chips_per_console_is_five_l2668_266830


namespace NUMINAMATH_CALUDE_sum_and_difference_of_numbers_l2668_266860

theorem sum_and_difference_of_numbers : ∃ (a b : ℕ), 
  b = 100 * a ∧ 
  a + b = 36400 ∧ 
  b - a = 35640 := by
sorry

end NUMINAMATH_CALUDE_sum_and_difference_of_numbers_l2668_266860


namespace NUMINAMATH_CALUDE_perpendicular_planes_parallel_perpendicular_planes_line_l2668_266818

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Line → Prop)
variable (subset : Line → Plane → Prop)
variable (plane_parallel : Plane → Plane → Prop)
variable (plane_perpendicular : Plane → Plane → Prop)
variable (line_perpendicular_plane : Line → Plane → Prop)
variable (intersection : Plane → Plane → Line)

-- Theorem 1
theorem perpendicular_planes_parallel 
  (m n l : Line) (α β : Plane) :
  line_perpendicular_plane l α →
  line_perpendicular_plane m β →
  parallel l m →
  plane_parallel α β := by sorry

-- Theorem 2
theorem perpendicular_planes_line 
  (m n : Line) (α β : Plane) :
  plane_perpendicular α β →
  intersection α β = m →
  subset n β →
  perpendicular n m →
  line_perpendicular_plane n α := by sorry

end NUMINAMATH_CALUDE_perpendicular_planes_parallel_perpendicular_planes_line_l2668_266818


namespace NUMINAMATH_CALUDE_probability_at_least_one_defective_l2668_266809

/-- The probability of selecting at least one defective item from a set of products -/
theorem probability_at_least_one_defective 
  (total : ℕ) 
  (defective : ℕ) 
  (selected : ℕ) 
  (h1 : total = 10) 
  (h2 : defective = 3) 
  (h3 : selected = 3) :
  (1 : ℚ) - (Nat.choose (total - defective) selected : ℚ) / (Nat.choose total selected : ℚ) = 17/24 := by
  sorry

#check probability_at_least_one_defective

end NUMINAMATH_CALUDE_probability_at_least_one_defective_l2668_266809


namespace NUMINAMATH_CALUDE_square_cut_corners_l2668_266858

theorem square_cut_corners (s : ℝ) (h : (2 / 9) * s^2 = 288) :
  s - 2 * (1 / 3 * s) = 24 := by
  sorry

end NUMINAMATH_CALUDE_square_cut_corners_l2668_266858


namespace NUMINAMATH_CALUDE_decimal_shift_problem_l2668_266893

theorem decimal_shift_problem (x : ℝ) (h1 : x > 0) (h2 : 1000 * x = 3 * (1 / x)) : 
  x = Real.sqrt 30 / 100 := by
  sorry

end NUMINAMATH_CALUDE_decimal_shift_problem_l2668_266893


namespace NUMINAMATH_CALUDE_sum_of_x_coordinates_l2668_266847

-- Define the points
def O : ℝ × ℝ := (0, 0)
def P : ℝ → ℝ × ℝ := λ t ↦ (5*t, 12*t)
def Q : ℝ → ℝ × ℝ := λ t ↦ (8*t, 6*t)

-- State the theorem
theorem sum_of_x_coordinates (t : ℝ) : 
  (P t).1 + (Q t).1 = 13*t := by
  sorry

end NUMINAMATH_CALUDE_sum_of_x_coordinates_l2668_266847


namespace NUMINAMATH_CALUDE_fifteenth_replacement_in_april_l2668_266884

def months : List String := ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]

def monthsAfterFebruary (n : Nat) : Nat :=
  (months.indexOf "February" + n) % months.length

theorem fifteenth_replacement_in_april :
  months[monthsAfterFebruary 98] = "April" := by
  sorry

end NUMINAMATH_CALUDE_fifteenth_replacement_in_april_l2668_266884


namespace NUMINAMATH_CALUDE_not_divisible_1998_pow_minus_1_l2668_266819

theorem not_divisible_1998_pow_minus_1 (m : ℕ) : ¬(1000^m - 1 ∣ 1998^m - 1) := by
  sorry

end NUMINAMATH_CALUDE_not_divisible_1998_pow_minus_1_l2668_266819


namespace NUMINAMATH_CALUDE_degrees_to_minutes_03_negative_comparison_l2668_266876

-- Define the conversion factor from degrees to minutes
def degrees_to_minutes (d : ℝ) : ℝ := d * 60

-- Theorem 1: 0.3 degrees is equal to 18 minutes
theorem degrees_to_minutes_03 : degrees_to_minutes 0.3 = 18 := by sorry

-- Theorem 2: -2 is greater than -3
theorem negative_comparison : -2 > -3 := by sorry

end NUMINAMATH_CALUDE_degrees_to_minutes_03_negative_comparison_l2668_266876


namespace NUMINAMATH_CALUDE_distribute_5_3_l2668_266812

/-- The number of ways to distribute n volunteers to k schools, 
    with each school receiving at least one volunteer -/
def distribute (n k : ℕ) : ℕ := sorry

/-- The number of ways to distribute 5 volunteers to 3 schools, 
    with each school receiving at least one volunteer, is 150 -/
theorem distribute_5_3 : distribute 5 3 = 150 := by sorry

end NUMINAMATH_CALUDE_distribute_5_3_l2668_266812


namespace NUMINAMATH_CALUDE_scientific_notation_of_0_0008_l2668_266826

theorem scientific_notation_of_0_0008 : ∃ (a : ℝ) (n : ℤ), 
  0.0008 = a * (10 : ℝ) ^ n ∧ 1 ≤ a ∧ a < 10 ∧ a = 8 ∧ n = -4 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_0_0008_l2668_266826


namespace NUMINAMATH_CALUDE_shopping_cart_fruit_ratio_l2668_266806

theorem shopping_cart_fruit_ratio :
  ∀ (apples oranges pears : ℕ),
    oranges = 3 * apples →
    apples = (pears : ℚ) * (83333333333333333 : ℚ) / (1000000000000000000 : ℚ) →
    (pears : ℚ) / (oranges : ℚ) = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_shopping_cart_fruit_ratio_l2668_266806


namespace NUMINAMATH_CALUDE_tablet_savings_l2668_266834

/-- The amount saved when buying a tablet from the cheaper store --/
theorem tablet_savings (list_price : ℝ) (tech_discount_percent : ℝ) (electro_discount : ℝ) :
  list_price = 120 →
  tech_discount_percent = 15 →
  electro_discount = 20 →
  list_price * (1 - tech_discount_percent / 100) - (list_price - electro_discount) = 2 :=
by sorry

end NUMINAMATH_CALUDE_tablet_savings_l2668_266834


namespace NUMINAMATH_CALUDE_shirt_price_theorem_l2668_266832

/-- Represents the problem of determining shirt prices and profits --/
structure ShirtProblem where
  first_batch_cost : ℝ
  second_batch_cost : ℝ
  quantity_ratio : ℝ
  price_difference : ℝ
  discount_quantity : ℕ
  discount_rate : ℝ
  min_profit : ℝ

/-- Calculates the unit price of the first batch --/
def first_batch_unit_price (p : ShirtProblem) : ℝ := 80

/-- Calculates the minimum selling price per shirt --/
def min_selling_price (p : ShirtProblem) : ℝ := 120

/-- Theorem stating the correctness of the calculated prices --/
theorem shirt_price_theorem (p : ShirtProblem) 
  (h1 : p.first_batch_cost = 3200)
  (h2 : p.second_batch_cost = 7200)
  (h3 : p.quantity_ratio = 2)
  (h4 : p.price_difference = 10)
  (h5 : p.discount_quantity = 20)
  (h6 : p.discount_rate = 0.2)
  (h7 : p.min_profit = 3520) :
  first_batch_unit_price p = 80 ∧ 
  min_selling_price p = 120 ∧
  min_selling_price p ≥ (p.min_profit + p.first_batch_cost + p.second_batch_cost) / 
    (p.first_batch_cost / first_batch_unit_price p + 
     p.second_batch_cost / (first_batch_unit_price p + p.price_difference) + 
     p.discount_quantity * (1 - p.discount_rate)) := by
  sorry


end NUMINAMATH_CALUDE_shirt_price_theorem_l2668_266832


namespace NUMINAMATH_CALUDE_total_points_sum_l2668_266849

def g (n : ℕ) : ℕ :=
  if n % 3 = 0 then 8
  else if n % 2 = 0 then 3
  else 0

def gina_rolls : List ℕ := [6, 5, 2, 3, 4]
def helen_rolls : List ℕ := [1, 2, 4, 6, 3]

theorem total_points_sum : (gina_rolls.map g).sum + (helen_rolls.map g).sum = 44 := by
  sorry

end NUMINAMATH_CALUDE_total_points_sum_l2668_266849


namespace NUMINAMATH_CALUDE_negation_of_rectangle_diagonals_equal_l2668_266835

theorem negation_of_rectangle_diagonals_equal :
  let p := "The diagonals of a rectangle are equal"
  ¬p = "The diagonals of a rectangle are not equal" := by
  sorry

end NUMINAMATH_CALUDE_negation_of_rectangle_diagonals_equal_l2668_266835


namespace NUMINAMATH_CALUDE_triangle_ratio_equality_l2668_266889

/-- Given a triangle ABC with sides a, b, c, height ha corresponding to side a,
    and inscribed circle radius r, prove that (a + b + c) / a = ha / r -/
theorem triangle_ratio_equality (a b c ha r : ℝ) (ha_pos : ha > 0) (r_pos : r > 0) 
  (a_pos : a > 0) (b_pos : b > 0) (c_pos : c > 0) : (a + b + c) / a = ha / r :=
sorry

end NUMINAMATH_CALUDE_triangle_ratio_equality_l2668_266889


namespace NUMINAMATH_CALUDE_michaels_fish_count_l2668_266844

theorem michaels_fish_count (original_count added_count total_count : ℕ) : 
  added_count = 18 →
  total_count = 49 →
  original_count + added_count = total_count :=
by
  sorry

end NUMINAMATH_CALUDE_michaels_fish_count_l2668_266844


namespace NUMINAMATH_CALUDE_sqrt_2_times_sqrt_8_l2668_266852

theorem sqrt_2_times_sqrt_8 : Real.sqrt 2 * Real.sqrt 8 = 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_2_times_sqrt_8_l2668_266852


namespace NUMINAMATH_CALUDE_container_volume_ratio_l2668_266899

theorem container_volume_ratio :
  ∀ (V₁ V₂ : ℝ), V₁ > 0 → V₂ > 0 →
  (3/4 : ℝ) * V₁ = (2/3 : ℝ) * V₂ →
  V₁ / V₂ = 8/9 := by
sorry

end NUMINAMATH_CALUDE_container_volume_ratio_l2668_266899


namespace NUMINAMATH_CALUDE_total_viewing_time_l2668_266859

-- Define the viewing segments
def segment1 : ℕ := 35
def segment2 : ℕ := 45
def segment3 : ℕ := 20

-- Define the rewind times
def rewind1 : ℕ := 5
def rewind2 : ℕ := 15

-- Theorem to prove
theorem total_viewing_time :
  segment1 + segment2 + segment3 + rewind1 + rewind2 = 120 := by
  sorry

end NUMINAMATH_CALUDE_total_viewing_time_l2668_266859


namespace NUMINAMATH_CALUDE_left_handed_sci_fi_fans_count_l2668_266873

/-- Represents the book club with its member properties -/
structure BookClub where
  total_members : ℕ
  left_handed : ℕ
  sci_fi_fans : ℕ
  right_handed_non_sci_fi : ℕ

/-- The number of left-handed members who like sci-fi books in the book club -/
def left_handed_sci_fi_fans (club : BookClub) : ℕ :=
  club.total_members - (club.left_handed + club.sci_fi_fans + club.right_handed_non_sci_fi) + club.left_handed + club.sci_fi_fans - club.total_members

/-- Theorem stating that the number of left-handed sci-fi fans is 4 for the given book club -/
theorem left_handed_sci_fi_fans_count (club : BookClub) 
  (h1 : club.total_members = 30)
  (h2 : club.left_handed = 12)
  (h3 : club.sci_fi_fans = 18)
  (h4 : club.right_handed_non_sci_fi = 4) :
  left_handed_sci_fi_fans club = 4 := by
  sorry

end NUMINAMATH_CALUDE_left_handed_sci_fi_fans_count_l2668_266873
