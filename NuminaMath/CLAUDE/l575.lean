import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_rational_root_even_coeff_l575_57519

theorem quadratic_rational_root_even_coeff
  (a b c : ℤ) (x : ℚ)
  (h_a_nonzero : a ≠ 0)
  (h_root : a * x^2 + b * x + c = 0) :
  Even a ∨ Even b ∨ Even c :=
sorry

end NUMINAMATH_CALUDE_quadratic_rational_root_even_coeff_l575_57519


namespace NUMINAMATH_CALUDE_hannah_grapes_count_l575_57583

def sophie_oranges_daily : ℕ := 20
def observation_days : ℕ := 30
def total_fruits : ℕ := 1800

def hannah_grapes_daily : ℕ := (total_fruits - sophie_oranges_daily * observation_days) / observation_days

theorem hannah_grapes_count : hannah_grapes_daily = 40 := by
  sorry

end NUMINAMATH_CALUDE_hannah_grapes_count_l575_57583


namespace NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l575_57527

/-- The repeating decimal 0.464646... expressed as a real number -/
def repeating_decimal : ℚ := 46 / 99

/-- The theorem stating that the repeating decimal 0.464646... is equal to 46/99 -/
theorem repeating_decimal_equals_fraction : repeating_decimal = 46 / 99 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l575_57527


namespace NUMINAMATH_CALUDE_largest_multiple_of_15_under_500_l575_57561

theorem largest_multiple_of_15_under_500 : 
  ∀ n : ℕ, n * 15 < 500 → n * 15 ≤ 495 :=
sorry

end NUMINAMATH_CALUDE_largest_multiple_of_15_under_500_l575_57561


namespace NUMINAMATH_CALUDE_abs_sum_inequality_l575_57558

theorem abs_sum_inequality (x : ℝ) : 
  |x - 1| + |x - 2| > 3 ↔ x < 0 ∨ x > 3 := by sorry

end NUMINAMATH_CALUDE_abs_sum_inequality_l575_57558


namespace NUMINAMATH_CALUDE_quadratic_inequality_all_reals_l575_57597

/-- The quadratic inequality ax^2 + bx + c > 0 has all real numbers as its solution set
    if and only if a > 0 and the discriminant is negative. -/
theorem quadratic_inequality_all_reals 
  (a b c : ℝ) (Δ : ℝ) (hΔ : Δ = b^2 - 4*a*c) :
  (∀ x : ℝ, a * x^2 + b * x + c > 0) ↔ (a > 0 ∧ Δ < 0) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_all_reals_l575_57597


namespace NUMINAMATH_CALUDE_james_milk_consumption_l575_57544

/-- The amount of milk James drank, given his initial amount, the conversion rate from gallons to ounces, and the remaining amount. -/
def milk_drank (initial_gallons : ℕ) (ounces_per_gallon : ℕ) (remaining_ounces : ℕ) : ℕ :=
  initial_gallons * ounces_per_gallon - remaining_ounces

/-- Theorem stating that James drank 13 ounces of milk. -/
theorem james_milk_consumption :
  milk_drank 3 128 371 = 13 := by
  sorry

end NUMINAMATH_CALUDE_james_milk_consumption_l575_57544


namespace NUMINAMATH_CALUDE_square_sum_geq_product_sum_l575_57524

theorem square_sum_geq_product_sum (a b c : ℝ) : a^2 + b^2 + c^2 ≥ a*b + b*c + c*a := by
  sorry

end NUMINAMATH_CALUDE_square_sum_geq_product_sum_l575_57524


namespace NUMINAMATH_CALUDE_school_visit_arrangements_l575_57555

/-- Represents the number of days available for scheduling -/
def num_days : ℕ := 5

/-- Represents the number of schools to be scheduled -/
def num_schools : ℕ := 3

/-- Calculates the number of permutations of r items chosen from n items -/
def permutations (n r : ℕ) : ℕ :=
  if r > n then 0
  else Nat.factorial n / Nat.factorial (n - r)

/-- Calculates the number of valid arrangements for the school visits -/
def count_arrangements : ℕ :=
  permutations 4 2 + permutations 3 2 + permutations 2 2

/-- Theorem stating that the number of valid arrangements is 20 -/
theorem school_visit_arrangements :
  count_arrangements = 20 :=
sorry

end NUMINAMATH_CALUDE_school_visit_arrangements_l575_57555


namespace NUMINAMATH_CALUDE_manufacturer_measures_l575_57590

def samples_A : List ℝ := [3, 4, 5, 6, 8, 8, 8, 10]
def samples_B : List ℝ := [4, 6, 6, 6, 8, 9, 12, 13]
def samples_C : List ℝ := [3, 3, 4, 7, 9, 10, 11, 12]

def claimed_lifespan : ℝ := 8

def mode (l : List ℝ) : ℝ := sorry
def mean (l : List ℝ) : ℝ := sorry
def median (l : List ℝ) : ℝ := sorry

theorem manufacturer_measures :
  mode samples_A = claimed_lifespan ∧
  mean samples_B = claimed_lifespan ∧
  median samples_C = claimed_lifespan :=
sorry

end NUMINAMATH_CALUDE_manufacturer_measures_l575_57590


namespace NUMINAMATH_CALUDE_max_diff_squares_consecutive_integers_l575_57569

theorem max_diff_squares_consecutive_integers (n : ℤ) : 
  n + (n + 1) < 150 → (n + 1)^2 - n^2 ≤ 149 := by
  sorry

end NUMINAMATH_CALUDE_max_diff_squares_consecutive_integers_l575_57569


namespace NUMINAMATH_CALUDE_real_roots_range_roots_condition_m_value_l575_57510

/-- Quadratic equation parameters -/
def a : ℝ := 1
def b (m : ℝ) : ℝ := 2 * (m - 1)
def c (m : ℝ) : ℝ := m^2 + 2

/-- Discriminant of the quadratic equation -/
def discriminant (m : ℝ) : ℝ := (b m)^2 - 4 * a * (c m)

/-- Theorem stating the range of m for real roots -/
theorem real_roots_range (m : ℝ) :
  (∃ x : ℝ, a * x^2 + (b m) * x + (c m) = 0) ↔ m ≤ -1/2 := by sorry

/-- Theorem stating the value of m when the roots satisfy the given condition -/
theorem roots_condition_m_value (m : ℝ) (x₁ x₂ : ℝ) 
  (hroots : a * x₁^2 + (b m) * x₁ + (c m) = 0 ∧ a * x₂^2 + (b m) * x₂ + (c m) = 0)
  (hcond : (x₁ - x₂)^2 = 18 - x₁ * x₂) :
  m = -2 := by sorry

end NUMINAMATH_CALUDE_real_roots_range_roots_condition_m_value_l575_57510


namespace NUMINAMATH_CALUDE_factorization_of_difference_of_squares_l575_57581

theorem factorization_of_difference_of_squares (x y : ℝ) :
  4 * x^2 - y^4 = (2*x + y^2) * (2*x - y^2) := by sorry

end NUMINAMATH_CALUDE_factorization_of_difference_of_squares_l575_57581


namespace NUMINAMATH_CALUDE_new_bus_distance_l575_57528

theorem new_bus_distance (old_distance : ℝ) (percentage_increase : ℝ) (new_distance : ℝ) : 
  old_distance = 300 →
  percentage_increase = 0.30 →
  new_distance = old_distance * (1 + percentage_increase) →
  new_distance = 390 := by
sorry

end NUMINAMATH_CALUDE_new_bus_distance_l575_57528


namespace NUMINAMATH_CALUDE_lillian_candy_count_l575_57587

/-- The total number of candies Lillian has after receiving candies from her father and best friend. -/
def lillian_total_candies (initial : ℕ) (father_gave : ℕ) (friend_multiplier : ℕ) : ℕ :=
  initial + father_gave + friend_multiplier * father_gave

/-- Theorem stating that Lillian will have 113 candies given the initial conditions. -/
theorem lillian_candy_count :
  lillian_total_candies 88 5 4 = 113 := by
  sorry

#eval lillian_total_candies 88 5 4

end NUMINAMATH_CALUDE_lillian_candy_count_l575_57587


namespace NUMINAMATH_CALUDE_no_real_numbers_with_integer_roots_l575_57532

theorem no_real_numbers_with_integer_roots : 
  ¬ ∃ (a b c : ℝ), 
    (∃ (x₁ x₂ : ℤ), x₁ ≠ x₂ ∧ a * x₁^2 + b * x₁ + c = 0 ∧ a * x₂^2 + b * x₂ + c = 0) ∧
    (∃ (y₁ y₂ : ℤ), y₁ ≠ y₂ ∧ (a+1) * y₁^2 + (b+1) * y₁ + (c+1) = 0 ∧ (a+1) * y₂^2 + (b+1) * y₂ + (c+1) = 0) :=
by
  sorry


end NUMINAMATH_CALUDE_no_real_numbers_with_integer_roots_l575_57532


namespace NUMINAMATH_CALUDE_fraction_equality_l575_57541

theorem fraction_equality (m n r t : ℚ) 
  (h1 : m / n = 5 / 2) 
  (h2 : r / t = 8 / 5) : 
  (2 * m * r - 3 * n * t) / (5 * n * t - 4 * m * r) = -5 / 11 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l575_57541


namespace NUMINAMATH_CALUDE_board_game_cost_l575_57501

def number_of_games : ℕ := 6
def total_paid : ℕ := 100
def change_bill_value : ℕ := 5
def number_of_change_bills : ℕ := 2

theorem board_game_cost :
  (total_paid - (change_bill_value * number_of_change_bills)) / number_of_games = 15 := by
  sorry

end NUMINAMATH_CALUDE_board_game_cost_l575_57501


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l575_57574

theorem sufficient_not_necessary_condition (m : ℝ) :
  (∀ m, m < -2 → ∃ x : ℝ, x^2 + m*x + 1 = 0) ∧
  ¬(∀ x : ℝ, x^2 + m*x + 1 = 0 → m < -2) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l575_57574


namespace NUMINAMATH_CALUDE_cylinder_cone_lateral_area_ratio_l575_57568

/-- The ratio of lateral surface areas of a cylinder and a cone with equal slant heights and base radii -/
theorem cylinder_cone_lateral_area_ratio 
  (r : ℝ) -- base radius
  (l : ℝ) -- slant height
  (h_pos_r : r > 0)
  (h_pos_l : l > 0) :
  (2 * π * r * l) / (π * r * l) = 2 := by
  sorry

#check cylinder_cone_lateral_area_ratio

end NUMINAMATH_CALUDE_cylinder_cone_lateral_area_ratio_l575_57568


namespace NUMINAMATH_CALUDE_distance_between_parallel_lines_l575_57556

/-- The distance between two parallel lines -/
theorem distance_between_parallel_lines :
  let l₁ : ℝ → ℝ → Prop := fun x y ↦ x - 2 * y + 1 = 0
  let l₂ : ℝ → ℝ → Prop := fun x y ↦ x - 2 * y - 4 = 0
  ∃ d : ℝ, d = Real.sqrt 5 ∧
    ∀ (x₁ y₁ x₂ y₂ : ℝ), l₁ x₁ y₁ → l₂ x₂ y₂ →
      ((x₂ - x₁)^2 + (y₂ - y₁)^2 : ℝ) ≥ d^2 :=
by sorry


end NUMINAMATH_CALUDE_distance_between_parallel_lines_l575_57556


namespace NUMINAMATH_CALUDE_steps_calculation_l575_57584

/-- The number of steps Benjamin took from the hotel to Times Square. -/
def total_steps : ℕ := 582

/-- The number of steps Benjamin took from the hotel to Rockefeller Center. -/
def steps_to_rockefeller : ℕ := 354

/-- The number of steps Benjamin took from Rockefeller Center to Times Square. -/
def steps_rockefeller_to_times_square : ℕ := total_steps - steps_to_rockefeller

theorem steps_calculation :
  steps_rockefeller_to_times_square = 228 :=
by sorry

end NUMINAMATH_CALUDE_steps_calculation_l575_57584


namespace NUMINAMATH_CALUDE_quadratic_solution_sum_l575_57570

theorem quadratic_solution_sum (a b : ℝ) : 
  (5 * (a + b * Complex.I)^2 + 4 * (a + b * Complex.I) + 1 = 0 ∧
   5 * (a - b * Complex.I)^2 + 4 * (a - b * Complex.I) + 1 = 0) →
  a + b^2 = -9/25 := by
sorry

end NUMINAMATH_CALUDE_quadratic_solution_sum_l575_57570


namespace NUMINAMATH_CALUDE_mushroom_collection_problem_l575_57520

theorem mushroom_collection_problem :
  ∃ (x₁ x₂ x₃ x₄ : ℕ),
    x₁ + x₂ = 7 ∧
    x₁ + x₃ = 9 ∧
    x₁ + x₄ = 10 ∧
    x₂ + x₃ = 10 ∧
    x₂ + x₄ = 11 ∧
    x₃ + x₄ = 13 ∧
    x₁ ≤ x₂ ∧ x₂ ≤ x₃ ∧ x₃ ≤ x₄ :=
by sorry

end NUMINAMATH_CALUDE_mushroom_collection_problem_l575_57520


namespace NUMINAMATH_CALUDE_stratified_sample_size_l575_57565

/-- Proves that for a population of 600 with 250 young employees,
    a stratified sample with 5 young employees has a total size of 12 -/
theorem stratified_sample_size
  (total_population : ℕ)
  (young_population : ℕ)
  (young_sample : ℕ)
  (h1 : total_population = 600)
  (h2 : young_population = 250)
  (h3 : young_sample = 5)
  (h4 : young_population ≤ total_population)
  (h5 : young_sample > 0) :
  ∃ (sample_size : ℕ),
    sample_size * young_population = young_sample * total_population ∧
    sample_size = 12 := by
  sorry


end NUMINAMATH_CALUDE_stratified_sample_size_l575_57565


namespace NUMINAMATH_CALUDE_probability_same_color_problem_l575_57516

/-- Probability of drawing two balls of the same color with replacement -/
def probability_same_color (green red blue : ℕ) : ℚ :=
  let total := green + red + blue
  (green^2 + red^2 + blue^2) / total^2

/-- Theorem: The probability of drawing two balls of the same color is 29/81 -/
theorem probability_same_color_problem :
  probability_same_color 8 6 4 = 29 / 81 := by
  sorry

end NUMINAMATH_CALUDE_probability_same_color_problem_l575_57516


namespace NUMINAMATH_CALUDE_x_value_l575_57513

theorem x_value (x : ℝ) (h : x ∈ ({1, x^2} : Set ℝ)) : x = 0 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l575_57513


namespace NUMINAMATH_CALUDE_primitive_root_existence_l575_57557

theorem primitive_root_existence (p : Nat) (hp : Nat.Prime p) (hp_odd : p % 2 = 1) :
  ∃ g : Nat, 1 < g ∧ g < p ∧ ∀ n : Nat, n > 0 → IsPrimitiveRoot g (p^n) :=
by sorry

/- Definitions used:
Nat.Prime: Prime number predicate
IsPrimitiveRoot: Predicate for primitive root
-/

end NUMINAMATH_CALUDE_primitive_root_existence_l575_57557


namespace NUMINAMATH_CALUDE_green_pill_cost_l575_57585

theorem green_pill_cost (daily_green : ℕ) (daily_pink : ℕ) (days : ℕ) 
  (green_pink_diff : ℚ) (total_cost : ℚ) :
  daily_green = 2 →
  daily_pink = 1 →
  days = 21 →
  green_pink_diff = 1 →
  total_cost = 819 →
  ∃ (green_cost : ℚ), 
    green_cost = 40 / 3 ∧ 
    (daily_green * green_cost + daily_pink * (green_cost - green_pink_diff)) * days = total_cost :=
by sorry

end NUMINAMATH_CALUDE_green_pill_cost_l575_57585


namespace NUMINAMATH_CALUDE_sum_product_inequalities_l575_57567

theorem sum_product_inequalities (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ((a + b) * (1/a + 1/b) ≥ 4) ∧ ((a + b + c) * (1/a + 1/b + 1/c) ≥ 9) := by sorry

end NUMINAMATH_CALUDE_sum_product_inequalities_l575_57567


namespace NUMINAMATH_CALUDE_max_b_squared_l575_57514

theorem max_b_squared (a b : ℤ) : 
  (a + b)^2 + a*(a + b) + b = 0 → b^2 ≤ 81 :=
by sorry

end NUMINAMATH_CALUDE_max_b_squared_l575_57514


namespace NUMINAMATH_CALUDE_prob_at_least_two_different_fruits_l575_57589

def num_fruit_types : ℕ := 4
def num_meals : ℕ := 4

def prob_same_fruit : ℚ := (1 / num_fruit_types) ^ num_meals * num_fruit_types

theorem prob_at_least_two_different_fruits :
  1 - prob_same_fruit = 63 / 64 := by sorry

end NUMINAMATH_CALUDE_prob_at_least_two_different_fruits_l575_57589


namespace NUMINAMATH_CALUDE_deepak_age_l575_57517

/-- Given that the ratio of Rahul's age to Deepak's age is 4:3, 
    and Rahul's age after 4 years will be 32, 
    prove that Deepak's current age is 21 years. -/
theorem deepak_age (rahul_age deepak_age : ℕ) : 
  (rahul_age : ℚ) / deepak_age = 4 / 3 →
  rahul_age + 4 = 32 →
  deepak_age = 21 := by
  sorry

end NUMINAMATH_CALUDE_deepak_age_l575_57517


namespace NUMINAMATH_CALUDE_sports_conference_games_l575_57533

/-- Calculates the total number of games in a sports conference season --/
def total_games (total_teams : ℕ) (teams_per_division : ℕ) (intra_division_games : ℕ) (inter_division_games : ℕ) : ℕ :=
  let games_per_team := (teams_per_division - 1) * intra_division_games + teams_per_division * inter_division_games
  (total_teams * games_per_team) / 2

theorem sports_conference_games : 
  total_games 12 6 3 2 = 162 := by sorry

end NUMINAMATH_CALUDE_sports_conference_games_l575_57533


namespace NUMINAMATH_CALUDE_arithmetic_progression_implies_equal_numbers_l575_57540

theorem arithmetic_progression_implies_equal_numbers
  (a b : ℝ) 
  (ha : 0 < a) 
  (hb : 0 < b) 
  (h_arith_prog : (a + b) / 2 = (Real.sqrt (a * b) + Real.sqrt ((a^2 + b^2) / 2)) / 2) :
  a = b :=
sorry

end NUMINAMATH_CALUDE_arithmetic_progression_implies_equal_numbers_l575_57540


namespace NUMINAMATH_CALUDE_b_value_when_square_zero_l575_57529

theorem b_value_when_square_zero (b : ℝ) : (b + 5)^2 = 0 → b = -5 := by
  sorry

end NUMINAMATH_CALUDE_b_value_when_square_zero_l575_57529


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l575_57595

theorem sufficient_not_necessary (a b : ℝ) : 
  ((a > b ∧ b > 1) → (a - b < a^2 - b^2)) ∧ 
  ¬((a - b < a^2 - b^2) → (a > b ∧ b > 1)) := by
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l575_57595


namespace NUMINAMATH_CALUDE_append_12_to_three_digit_number_l575_57531

theorem append_12_to_three_digit_number (h t u : ℕ) :
  let original := 100 * h + 10 * t + u
  let new_number := original * 100 + 12
  new_number = 10000 * h + 1000 * t + 100 * u + 12 :=
by sorry

end NUMINAMATH_CALUDE_append_12_to_three_digit_number_l575_57531


namespace NUMINAMATH_CALUDE_like_terms_exponent_sum_l575_57506

/-- Given that 3x^(2m)y^3 and -2x^2y^n are like terms, prove that m + n = 4 -/
theorem like_terms_exponent_sum (m n : ℕ) : 
  (∀ x y : ℝ, 3 * x^(2*m) * y^3 = -2 * x^2 * y^n) → m + n = 4 := by
  sorry

end NUMINAMATH_CALUDE_like_terms_exponent_sum_l575_57506


namespace NUMINAMATH_CALUDE_poultry_farm_loss_l575_57582

theorem poultry_farm_loss (initial_chickens initial_turkeys initial_guinea_fowls : ℕ)
  (daily_turkey_loss daily_guinea_fowl_loss : ℕ)
  (total_birds_after_week : ℕ)
  (h1 : initial_chickens = 300)
  (h2 : initial_turkeys = 200)
  (h3 : initial_guinea_fowls = 80)
  (h4 : daily_turkey_loss = 8)
  (h5 : daily_guinea_fowl_loss = 5)
  (h6 : total_birds_after_week = 349)
  (h7 : initial_chickens + initial_turkeys + initial_guinea_fowls
      - (7 * daily_turkey_loss + 7 * daily_guinea_fowl_loss)
      - total_birds_after_week = 7 * daily_chicken_loss) :
  daily_chicken_loss = 20 := by sorry

end NUMINAMATH_CALUDE_poultry_farm_loss_l575_57582


namespace NUMINAMATH_CALUDE_remainder_problem_l575_57560

theorem remainder_problem (x R : ℤ) : 
  (∃ k : ℤ, x = 82 * k + R) → 
  (∃ m : ℤ, x + 17 = 41 * m + 22) → 
  R = 5 := by
sorry

end NUMINAMATH_CALUDE_remainder_problem_l575_57560


namespace NUMINAMATH_CALUDE_opposite_of_2023_l575_57522

theorem opposite_of_2023 :
  ∃ y : ℤ, (2023 : ℤ) + y = 0 ∧ y = -2023 :=
by sorry

end NUMINAMATH_CALUDE_opposite_of_2023_l575_57522


namespace NUMINAMATH_CALUDE_relationship_abc_l575_57598

theorem relationship_abc : 3^(1/5) > 0.3^2 ∧ 0.3^2 > Real.log 0.3 / Real.log 2 := by sorry

end NUMINAMATH_CALUDE_relationship_abc_l575_57598


namespace NUMINAMATH_CALUDE_existence_of_integers_l575_57542

theorem existence_of_integers : ∃ (list : List Int), 
  (list.length = 2016) ∧ 
  (list.prod = 9) ∧ 
  (list.sum = 0) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_integers_l575_57542


namespace NUMINAMATH_CALUDE_isosceles_triangle_from_cosine_condition_l575_57503

/-- Given a triangle ABC where a*cos(B) = b*cos(A), prove that the triangle is isosceles -/
theorem isosceles_triangle_from_cosine_condition (A B C : ℝ) (a b c : ℝ) 
  (h_triangle : A + B + C = π) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_cosine : a * Real.cos B = b * Real.cos A) : 
  a = b ∨ b = c ∨ a = c :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_from_cosine_condition_l575_57503


namespace NUMINAMATH_CALUDE_gwen_games_remaining_l575_57576

/-- The number of games remaining after giving some away -/
def remaining_games (initial : ℕ) (given_away : ℕ) : ℕ :=
  initial - given_away

/-- Theorem: Given 98 initial games and 7 games given away, 91 games remain -/
theorem gwen_games_remaining :
  remaining_games 98 7 = 91 := by
  sorry

end NUMINAMATH_CALUDE_gwen_games_remaining_l575_57576


namespace NUMINAMATH_CALUDE_two_numbers_problem_l575_57518

theorem two_numbers_problem (a b : ℝ) (h1 : a > b) (h2 : a > 0) (h3 : b > 0) 
  (h4 : a + b = 6) (h5 : a / b = 6) : a * b - (a - b) = 6 / 49 := by
  sorry

end NUMINAMATH_CALUDE_two_numbers_problem_l575_57518


namespace NUMINAMATH_CALUDE_difference_three_fifths_l575_57535

theorem difference_three_fifths (x : ℝ) : x - (3/5) * x = 145 → x = 362.5 := by
  sorry

end NUMINAMATH_CALUDE_difference_three_fifths_l575_57535


namespace NUMINAMATH_CALUDE_shopping_mall_probabilities_l575_57592

/-- Probability of a customer buying product A -/
def prob_A : ℝ := 0.5

/-- Probability of a customer buying product B -/
def prob_B : ℝ := 0.6

/-- Probability of a customer buying neither product A nor B -/
def prob_neither : ℝ := (1 - prob_A) * (1 - prob_B)

/-- Probability of a customer buying at least one product -/
def prob_at_least_one : ℝ := 1 - prob_neither

theorem shopping_mall_probabilities :
  (1 - (prob_A * prob_B) - prob_neither = 0.5) ∧
  (1 - (prob_at_least_one^3 + 3 * prob_at_least_one^2 * prob_neither) = 0.104) :=
sorry

end NUMINAMATH_CALUDE_shopping_mall_probabilities_l575_57592


namespace NUMINAMATH_CALUDE_ranch_feed_corn_cost_l575_57512

/-- Represents the ranch with its animals and pasture. -/
structure Ranch where
  sheep : ℕ
  cattle : ℕ
  pasture_acres : ℕ

/-- Represents the feed requirements and costs. -/
structure FeedInfo where
  cow_grass_per_month : ℕ
  sheep_grass_per_month : ℕ
  corn_bag_cost : ℕ
  cow_corn_months_per_bag : ℕ
  sheep_corn_months_per_bag : ℕ

/-- Calculates the annual cost of feed corn for the ranch. -/
def annual_feed_corn_cost (ranch : Ranch) (feed : FeedInfo) : ℕ :=
  sorry

/-- Theorem stating the annual feed corn cost for the given ranch and feed information. -/
theorem ranch_feed_corn_cost :
  let ranch := Ranch.mk 8 5 144
  let feed := FeedInfo.mk 2 1 10 1 2
  annual_feed_corn_cost ranch feed = 360 :=
sorry

end NUMINAMATH_CALUDE_ranch_feed_corn_cost_l575_57512


namespace NUMINAMATH_CALUDE_smallest_three_digit_palindrome_not_six_digit_palindrome_product_l575_57588

/-- Checks if a number is a three-digit palindrome -/
def isThreeDigitPalindrome (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999 ∧ (n / 100 = n % 10)

/-- Checks if a number is a six-digit palindrome -/
def isSixDigitPalindrome (n : ℕ) : Prop :=
  100000 ≤ n ∧ n ≤ 999999 ∧ (n / 100000 = n % 10) ∧ ((n / 10000) % 10 = (n / 10) % 10) ∧ ((n / 1000) % 10 = (n / 100) % 10)

/-- The main theorem -/
theorem smallest_three_digit_palindrome_not_six_digit_palindrome_product :
  isThreeDigitPalindrome 404 ∧
  ¬(isSixDigitPalindrome (404 * 102)) ∧
  ∀ n : ℕ, isThreeDigitPalindrome n → n < 404 → isSixDigitPalindrome (n * 102) :=
by sorry

end NUMINAMATH_CALUDE_smallest_three_digit_palindrome_not_six_digit_palindrome_product_l575_57588


namespace NUMINAMATH_CALUDE_montoya_family_budget_l575_57580

theorem montoya_family_budget (budget : ℝ) 
  (grocery_fraction : ℝ) (total_food_fraction : ℝ) :
  grocery_fraction = 0.6 →
  total_food_fraction = 0.8 →
  total_food_fraction = grocery_fraction + (budget - grocery_fraction * budget) / budget →
  (budget - grocery_fraction * budget) / budget = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_montoya_family_budget_l575_57580


namespace NUMINAMATH_CALUDE_total_pay_for_two_employees_l575_57566

/-- Proves that the total amount paid to two employees X and Y is 770 units of currency,
    given that X is paid 120% of Y's pay and Y is paid 350 units per week. -/
theorem total_pay_for_two_employees (y_pay : ℝ) (x_pay : ℝ) : 
  y_pay = 350 → x_pay = 1.2 * y_pay → x_pay + y_pay = 770 := by
  sorry

end NUMINAMATH_CALUDE_total_pay_for_two_employees_l575_57566


namespace NUMINAMATH_CALUDE_tangent_line_equation_l575_57552

def S (x : ℝ) : ℝ := 3*x - x^3

theorem tangent_line_equation (x₀ y₀ : ℝ) (h : y₀ = S x₀) (h₀ : x₀ = 2) (h₁ : y₀ = -2) :
  ∃ (m b : ℝ), (∀ x y, y = m*x + b → (x = x₀ ∧ y = y₀) ∨ (y - y₀ = m*(x - x₀))) ∧
  ((m = -9 ∧ b = 16) ∨ (m = 0 ∧ b = -2)) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l575_57552


namespace NUMINAMATH_CALUDE_magazines_to_boxes_l575_57526

theorem magazines_to_boxes (total_magazines : ℕ) (magazines_per_box : ℕ) (h1 : total_magazines = 63) (h2 : magazines_per_box = 9) :
  total_magazines / magazines_per_box = 7 := by
  sorry

end NUMINAMATH_CALUDE_magazines_to_boxes_l575_57526


namespace NUMINAMATH_CALUDE_variance_scaling_l575_57571

/-- Given a set of data points, this function returns the variance of the data set. -/
noncomputable def variance (data : Finset ℝ) : ℝ := sorry

/-- Given a set of data points, this function multiplies each point by a scalar. -/
def scaleData (data : Finset ℝ) (scalar : ℝ) : Finset ℝ := sorry

theorem variance_scaling (data : Finset ℝ) (s : ℝ) :
  variance data = s^2 → variance (scaleData data 2) = 4 * s^2 := by sorry

end NUMINAMATH_CALUDE_variance_scaling_l575_57571


namespace NUMINAMATH_CALUDE_afternoon_eggs_l575_57578

theorem afternoon_eggs (total_eggs day_eggs morning_eggs : ℕ) 
  (h1 : total_eggs = 1339)
  (h2 : morning_eggs = 816) :
  total_eggs - morning_eggs = 523 := by
  sorry

end NUMINAMATH_CALUDE_afternoon_eggs_l575_57578


namespace NUMINAMATH_CALUDE_system_of_equations_l575_57530

theorem system_of_equations (x y k : ℝ) : 
  (2 * x + y = 1) → 
  (x + 2 * y = k - 2) → 
  (x - y = 2) → 
  (k = 1) := by
sorry

end NUMINAMATH_CALUDE_system_of_equations_l575_57530


namespace NUMINAMATH_CALUDE_hours_worked_per_day_l575_57572

/-- Given a person who worked for 5 days and a total of 15 hours, 
    prove that the number of hours worked each day is equal to 3. -/
theorem hours_worked_per_day 
  (days_worked : ℕ) 
  (total_hours : ℕ) 
  (h1 : days_worked = 5) 
  (h2 : total_hours = 15) : 
  total_hours / days_worked = 3 := by
  sorry

end NUMINAMATH_CALUDE_hours_worked_per_day_l575_57572


namespace NUMINAMATH_CALUDE_apricot_trees_count_apricot_trees_proof_l575_57573

theorem apricot_trees_count : ℕ → ℕ → Prop :=
  fun apricot_count peach_count =>
    peach_count = 3 * apricot_count →
    apricot_count + peach_count = 232 →
    apricot_count = 58

-- The proof is omitted as per instructions
theorem apricot_trees_proof : apricot_trees_count 58 174 := by
  sorry

end NUMINAMATH_CALUDE_apricot_trees_count_apricot_trees_proof_l575_57573


namespace NUMINAMATH_CALUDE_evaluate_expression_l575_57586

theorem evaluate_expression : ((3^5 / 3^2) * 2^10) + (1/2) = 27648.5 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l575_57586


namespace NUMINAMATH_CALUDE_expression_evaluation_l575_57546

theorem expression_evaluation (x : ℝ) (h : x = -2) :
  (1 + 1 / (x - 1)) / (x / (x^2 - 1)) = -1 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l575_57546


namespace NUMINAMATH_CALUDE_laborer_income_l575_57551

/-- Represents the financial situation of a laborer over a 10-month period. -/
structure LaborerFinances where
  monthly_income : ℝ
  initial_expenditure : ℝ
  reduced_expenditure : ℝ
  initial_period : ℕ
  reduced_period : ℕ
  savings : ℝ

/-- The laborer's finances satisfy the given conditions. -/
def satisfies_conditions (f : LaborerFinances) : Prop :=
  f.initial_expenditure = 75 ∧
  f.reduced_expenditure = 60 ∧
  f.initial_period = 6 ∧
  f.reduced_period = 4 ∧
  f.savings = 30 ∧
  f.initial_period * f.monthly_income < f.initial_period * f.initial_expenditure ∧
  f.reduced_period * f.monthly_income = f.reduced_period * f.reduced_expenditure + 
    (f.initial_period * f.initial_expenditure - f.initial_period * f.monthly_income) + f.savings

/-- Theorem stating that if the laborer's finances satisfy the given conditions, 
    then their monthly income is 72. -/
theorem laborer_income (f : LaborerFinances) 
  (h : satisfies_conditions f) : f.monthly_income = 72 := by
  sorry

end NUMINAMATH_CALUDE_laborer_income_l575_57551


namespace NUMINAMATH_CALUDE_order_of_6_l575_57537

def f (x : ℕ) : ℕ := x^2 % 13

def is_periodic (f : ℕ → ℕ) (x : ℕ) (period : ℕ) : Prop :=
  ∀ n, f^[n + period] x = f^[n] x

theorem order_of_6 (h : is_periodic f 6 72) :
  ∀ k, 0 < k → k < 72 → ¬ is_periodic f 6 k :=
sorry

end NUMINAMATH_CALUDE_order_of_6_l575_57537


namespace NUMINAMATH_CALUDE_bernoulli_inequality_l575_57534

theorem bernoulli_inequality (p : ℝ) (k : ℚ) (hp : p > 0) (hk : k > 1) :
  (1 + p)^(k : ℝ) > 1 + p * k := by
  sorry

end NUMINAMATH_CALUDE_bernoulli_inequality_l575_57534


namespace NUMINAMATH_CALUDE_inscribed_circles_chord_length_l575_57599

/-- Given two circles, one inscribed in an angle α with radius r and another of radius R 
    touching one side of the angle at the same point as the first circle and intersecting 
    the other side at points A and B, the length of AB can be calculated. -/
theorem inscribed_circles_chord_length (α r R : ℝ) (h_pos_r : r > 0) (h_pos_R : R > 0) :
  ∃ (AB : ℝ), AB = 4 * Real.cos (α / 2) * Real.sqrt ((R - r) * (R * Real.sin (α / 2)^2 + r * Real.cos (α / 2)^2)) := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circles_chord_length_l575_57599


namespace NUMINAMATH_CALUDE_horizontal_asymptote_of_f_l575_57548

noncomputable def f (x : ℝ) : ℝ := (8 * x^2 - 4) / (4 * x^2 + 8 * x + 3)

theorem horizontal_asymptote_of_f :
  ∀ ε > 0, ∃ N, ∀ x > N, |f x - 2| < ε :=
sorry

end NUMINAMATH_CALUDE_horizontal_asymptote_of_f_l575_57548


namespace NUMINAMATH_CALUDE_find_b_value_l575_57502

theorem find_b_value (a b : ℚ) (h1 : 3 * a - 2 = 1) (h2 : 2 * b - 3 * a = 2) : b = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_find_b_value_l575_57502


namespace NUMINAMATH_CALUDE_total_winter_clothing_l575_57543

def number_of_boxes : ℕ := 8
def scarves_per_box : ℕ := 4
def mittens_per_box : ℕ := 6

theorem total_winter_clothing : 
  number_of_boxes * (scarves_per_box + mittens_per_box) = 80 := by
  sorry

end NUMINAMATH_CALUDE_total_winter_clothing_l575_57543


namespace NUMINAMATH_CALUDE_jane_vases_last_day_l575_57539

/-- The number of vases Jane arranges on the last day given her daily rate, total vases, and total days --/
def vases_on_last_day (daily_rate : ℕ) (total_vases : ℕ) (total_days : ℕ) : ℕ :=
  if total_vases ≤ daily_rate * (total_days - 1)
  then 0
  else total_vases - daily_rate * (total_days - 1)

theorem jane_vases_last_day :
  vases_on_last_day 25 378 17 = 0 := by
  sorry

end NUMINAMATH_CALUDE_jane_vases_last_day_l575_57539


namespace NUMINAMATH_CALUDE_tim_earnings_l575_57564

/-- Calculates the total money earned by Tim given the number of coins received from various sources. -/
def total_money_earned (shine_pennies shine_nickels shine_dimes shine_quarters : ℕ)
                       (tip_pennies tip_nickels tip_dimes tip_half_dollars : ℕ)
                       (stranger_pennies stranger_quarters : ℕ) : ℚ :=
  let penny_value : ℚ := 1 / 100
  let nickel_value : ℚ := 5 / 100
  let dime_value : ℚ := 10 / 100
  let quarter_value : ℚ := 25 / 100
  let half_dollar_value : ℚ := 50 / 100

  let shine_total : ℚ := shine_pennies * penny_value + shine_nickels * nickel_value +
                         shine_dimes * dime_value + shine_quarters * quarter_value
  let tip_total : ℚ := tip_pennies * penny_value + tip_nickels * nickel_value +
                       tip_dimes * dime_value + tip_half_dollars * half_dollar_value
  let stranger_total : ℚ := stranger_pennies * penny_value + stranger_quarters * quarter_value

  shine_total + tip_total + stranger_total

/-- Theorem stating that Tim's total earnings equal $9.79 given the specified coin counts. -/
theorem tim_earnings :
  total_money_earned 4 3 13 6 15 12 7 9 10 3 = 979 / 100 := by
  sorry

end NUMINAMATH_CALUDE_tim_earnings_l575_57564


namespace NUMINAMATH_CALUDE_min_value_problem_l575_57593

theorem min_value_problem (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h : 1 / (x + 3) + 1 / (y + 4) = 1 / 2) :
  ∀ a b : ℝ, a > 0 ∧ b > 0 ∧ 1 / (a + 3) + 1 / (b + 4) = 1 / 2 →
  2 * x + y ≤ 2 * a + b ∧ 2 * x + y = 1 + 4 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_problem_l575_57593


namespace NUMINAMATH_CALUDE_total_money_is_correct_l575_57554

/-- Calculates the total amount of money in Euros given the specified coins and bills and the conversion rate. -/
def total_money_in_euros : ℝ :=
  let pennies : ℕ := 9
  let nickels : ℕ := 4
  let dimes : ℕ := 3
  let quarters : ℕ := 7
  let half_dollars : ℕ := 5
  let one_dollar_coins : ℕ := 2
  let two_dollar_bills : ℕ := 1
  
  let penny_value : ℝ := 0.01
  let nickel_value : ℝ := 0.05
  let dime_value : ℝ := 0.10
  let quarter_value : ℝ := 0.25
  let half_dollar_value : ℝ := 0.50
  let one_dollar_value : ℝ := 1.00
  let two_dollar_value : ℝ := 2.00
  
  let usd_to_euro_rate : ℝ := 0.85
  
  let total_usd : ℝ := 
    pennies * penny_value +
    nickels * nickel_value +
    dimes * dime_value +
    quarters * quarter_value +
    half_dollars * half_dollar_value +
    one_dollar_coins * one_dollar_value +
    two_dollar_bills * two_dollar_value
  
  total_usd * usd_to_euro_rate

/-- Theorem stating that the total amount of money in Euros is equal to 7.514. -/
theorem total_money_is_correct : total_money_in_euros = 7.514 := by
  sorry

end NUMINAMATH_CALUDE_total_money_is_correct_l575_57554


namespace NUMINAMATH_CALUDE_jungkook_paper_count_l575_57596

/-- Calculates the total number of pieces of colored paper given the number of bundles,
    pieces per bundle, and individual pieces. -/
def total_pieces (bundles : ℕ) (pieces_per_bundle : ℕ) (individual_pieces : ℕ) : ℕ :=
  bundles * pieces_per_bundle + individual_pieces

/-- Proves that given 3 bundles of 10 pieces each and 8 individual pieces,
    the total number of pieces is 38. -/
theorem jungkook_paper_count :
  total_pieces 3 10 8 = 38 := by
  sorry

end NUMINAMATH_CALUDE_jungkook_paper_count_l575_57596


namespace NUMINAMATH_CALUDE_homework_submission_negation_l575_57562

variable (Student : Type)
variable (inClass : Student → Prop)
variable (submittedHomework : Student → Prop)

theorem homework_submission_negation :
  (¬ ∀ s : Student, inClass s → submittedHomework s) ↔
  (∃ s : Student, inClass s ∧ ¬ submittedHomework s) :=
by sorry

end NUMINAMATH_CALUDE_homework_submission_negation_l575_57562


namespace NUMINAMATH_CALUDE_day_after_2_pow_20_l575_57545

/-- Represents days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Calculates the day of the week after a given number of days from Monday -/
def dayAfter (days : Nat) : DayOfWeek :=
  match (days % 7) with
  | 0 => DayOfWeek.Monday
  | 1 => DayOfWeek.Tuesday
  | 2 => DayOfWeek.Wednesday
  | 3 => DayOfWeek.Thursday
  | 4 => DayOfWeek.Friday
  | 5 => DayOfWeek.Saturday
  | _ => DayOfWeek.Sunday

/-- Theorem: After 2^20 days from Monday, it will be Friday -/
theorem day_after_2_pow_20 : dayAfter (2^20) = DayOfWeek.Friday := by
  sorry


end NUMINAMATH_CALUDE_day_after_2_pow_20_l575_57545


namespace NUMINAMATH_CALUDE_part1_part2_l575_57538

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x + 3| + |x - a|

-- Part 1
theorem part1 (x : ℝ) :
  f 4 x = 7 → -3 ≤ x ∧ x ≤ 4 :=
by sorry

-- Part 2
theorem part2 (a : ℝ) (h : a > 0) :
  ({x : ℝ | f a x ≥ 6} = {x : ℝ | x ≤ -4 ∨ x ≥ 2}) → a = 1 :=
by sorry

end NUMINAMATH_CALUDE_part1_part2_l575_57538


namespace NUMINAMATH_CALUDE_reading_difference_l575_57511

theorem reading_difference (min_assigned : ℕ) (harrison_extra : ℕ) (sam_pages : ℕ) :
  min_assigned = 25 →
  harrison_extra = 10 →
  sam_pages = 100 →
  ∃ (pam_pages : ℕ) (harrison_pages : ℕ),
    pam_pages = sam_pages / 2 ∧
    harrison_pages = min_assigned + harrison_extra ∧
    pam_pages > harrison_pages ∧
    pam_pages - harrison_pages = 15 :=
by sorry

end NUMINAMATH_CALUDE_reading_difference_l575_57511


namespace NUMINAMATH_CALUDE_number_operations_l575_57591

theorem number_operations (x : ℝ) : (3 * ((x - 50) / 4) + 28 = 73) ↔ (x = 110) := by
  sorry

end NUMINAMATH_CALUDE_number_operations_l575_57591


namespace NUMINAMATH_CALUDE_triangle_angle_and_side_length_l575_57575

theorem triangle_angle_and_side_length 
  (A B C : ℝ) 
  (a b c : ℝ) 
  (m n : ℝ × ℝ) :
  A > 0 ∧ A < π ∧
  B > 0 ∧ B < π ∧
  C > 0 ∧ C < π ∧
  A + B + C = π ∧
  m = (Real.sqrt 3, Real.cos A + 1) ∧
  n = (Real.sin A, -1) ∧
  m.1 * n.1 + m.2 * n.2 = 0 ∧
  a = 2 ∧
  Real.cos B = Real.sqrt 3 / 3 →
  A = π / 3 ∧ b = 4 * Real.sqrt 2 / 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_and_side_length_l575_57575


namespace NUMINAMATH_CALUDE_symmetry_line_l575_57553

-- Define the circles
def circle1 (x y : ℝ) : Prop := (x - 7)^2 + (y + 4)^2 = 16
def circle2 (x y : ℝ) : Prop := (x + 5)^2 + (y - 6)^2 = 16

-- Define the line of symmetry
def line_of_symmetry (x y : ℝ) : Prop := 6*x - 5*y - 1 = 0

-- Theorem statement
theorem symmetry_line :
  ∀ (x₁ y₁ x₂ y₂ : ℝ),
  circle1 x₁ y₁ → circle2 x₂ y₂ →
  ∃ (x y : ℝ),
  line_of_symmetry x y ∧
  (x = (x₁ + x₂) / 2) ∧
  (y = (y₁ + y₂) / 2) ∧
  ((x - x₁)^2 + (y - y₁)^2 = (x - x₂)^2 + (y - y₂)^2) :=
by sorry

end NUMINAMATH_CALUDE_symmetry_line_l575_57553


namespace NUMINAMATH_CALUDE_english_only_students_l575_57594

theorem english_only_students (total : ℕ) (all_three : ℕ) (english_only : ℕ) (french_only : ℕ) :
  total = 35 →
  all_three = 2 →
  english_only = 3 * french_only →
  english_only + french_only + all_three = total →
  english_only - all_three = 23 := by
  sorry

end NUMINAMATH_CALUDE_english_only_students_l575_57594


namespace NUMINAMATH_CALUDE_number_comparison_l575_57521

theorem number_comparison : ∃ (a b c : ℝ), 
  a = 7^(0.3 : ℝ) ∧ 
  b = (0.3 : ℝ)^7 ∧ 
  c = Real.log 0.3 ∧ 
  a > b ∧ b > c := by
  sorry

end NUMINAMATH_CALUDE_number_comparison_l575_57521


namespace NUMINAMATH_CALUDE_circles_externally_tangent_l575_57579

def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 2*y + 1 = 0

def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 4*x - 4*y - 1 = 0

def center1 : ℝ × ℝ := (2, -1)

def center2 : ℝ × ℝ := (-2, 2)

def radius1 : ℝ := 2

def radius2 : ℝ := 3

theorem circles_externally_tangent :
  let d := Real.sqrt ((center1.1 - center2.1)^2 + (center1.2 - center2.2)^2)
  d = radius1 + radius2 := by sorry

end NUMINAMATH_CALUDE_circles_externally_tangent_l575_57579


namespace NUMINAMATH_CALUDE_orange_bags_l575_57508

def total_weight : ℝ := 45.0
def bag_capacity : ℝ := 23.0

theorem orange_bags : ⌊total_weight / bag_capacity⌋ = 1 := by sorry

end NUMINAMATH_CALUDE_orange_bags_l575_57508


namespace NUMINAMATH_CALUDE_sin_15_cos_15_eq_quarter_l575_57509

theorem sin_15_cos_15_eq_quarter : Real.sin (15 * π / 180) * Real.cos (15 * π / 180) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_sin_15_cos_15_eq_quarter_l575_57509


namespace NUMINAMATH_CALUDE_heathers_oranges_l575_57536

/-- The total number of oranges Heather has after receiving oranges from Russell -/
def total_oranges (initial : ℝ) (received : ℝ) : ℝ :=
  initial + received

/-- Theorem stating that Heather's total oranges is 96.3 given the initial and received amounts -/
theorem heathers_oranges :
  total_oranges 60.5 35.8 = 96.3 := by
  sorry

end NUMINAMATH_CALUDE_heathers_oranges_l575_57536


namespace NUMINAMATH_CALUDE_largest_integer_negative_quadratic_l575_57523

theorem largest_integer_negative_quadratic : 
  (∀ m : ℤ, m > 7 → m^2 - 11*m + 24 ≥ 0) ∧ 
  (7^2 - 11*7 + 24 < 0) := by
  sorry

end NUMINAMATH_CALUDE_largest_integer_negative_quadratic_l575_57523


namespace NUMINAMATH_CALUDE_simplify_expression_l575_57563

theorem simplify_expression (a b m : ℝ) (h1 : a + b = m) (h2 : a * b = -4) :
  (a - 2) * (b - 2) = -2 * m := by sorry

end NUMINAMATH_CALUDE_simplify_expression_l575_57563


namespace NUMINAMATH_CALUDE_intersection_complement_theorem_l575_57577

-- Define the sets A and B
def A : Set ℝ := {x | |x + 3| - |x - 3| > 3}
def B : Set ℝ := {x | ∃ t > 0, x = (t^2 - 4*t + 1) / t}

-- State the theorem
theorem intersection_complement_theorem : B ∩ (Set.univ \ A) = Set.Icc (-2) (3/2) := by sorry

end NUMINAMATH_CALUDE_intersection_complement_theorem_l575_57577


namespace NUMINAMATH_CALUDE_function_ordering_l575_57525

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def monotone_decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f y < f x

theorem function_ordering (f : ℝ → ℝ) 
  (h1 : is_even f)
  (h2 : ∀ x₁ x₂, x₁ ≤ -1 ∧ x₂ ≤ -1 → (x₂ - x₁) * (f x₂ - f x₁) < 0) :
  f (-1) < f (-3/2) ∧ f (-3/2) < f 2 :=
sorry

end NUMINAMATH_CALUDE_function_ordering_l575_57525


namespace NUMINAMATH_CALUDE_range_of_n_minus_m_l575_57504

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then Real.exp x - 1 else (3/2) * x + 1

theorem range_of_n_minus_m (m n : ℝ) (h1 : m < n) (h2 : f m = f n) :
  2/3 < n - m ∧ n - m ≤ Real.log (3/2) + 1/3 :=
sorry

end NUMINAMATH_CALUDE_range_of_n_minus_m_l575_57504


namespace NUMINAMATH_CALUDE_sqrt_2023_bounds_l575_57549

theorem sqrt_2023_bounds : 40 < Real.sqrt 2023 ∧ Real.sqrt 2023 < 45 := by
  have h1 : 1600 < 2023 := by sorry
  have h2 : 2023 < 2025 := by sorry
  sorry

end NUMINAMATH_CALUDE_sqrt_2023_bounds_l575_57549


namespace NUMINAMATH_CALUDE_trebled_result_is_72_l575_57500

theorem trebled_result_is_72 (x : ℕ) (h : x = 9) : 3 * (2 * x + 6) = 72 := by
  sorry

end NUMINAMATH_CALUDE_trebled_result_is_72_l575_57500


namespace NUMINAMATH_CALUDE_shopkeeper_gain_percentage_l575_57559

/-- The gain percentage of a shopkeeper using false weights -/
theorem shopkeeper_gain_percentage 
  (claimed_weight : ℝ) 
  (actual_weight : ℝ) 
  (claimed_weight_is_kg : claimed_weight = 1000) 
  (actual_weight_used : actual_weight = 980) : 
  (claimed_weight - actual_weight) / actual_weight * 100 = 
  (1000 - 980) / 980 * 100 := by
sorry

end NUMINAMATH_CALUDE_shopkeeper_gain_percentage_l575_57559


namespace NUMINAMATH_CALUDE_rectangles_with_one_gray_cell_l575_57507

/-- The number of rectangles containing exactly one gray cell in a 2x20 grid --/
def num_rectangles_with_one_gray_cell (total_gray_cells : ℕ) 
  (blue_cells : ℕ) (red_cells : ℕ) : ℕ :=
  blue_cells * 4 + red_cells * 8

/-- Theorem stating the number of rectangles with one gray cell in the given grid --/
theorem rectangles_with_one_gray_cell :
  num_rectangles_with_one_gray_cell 40 36 4 = 176 := by
  sorry

#eval num_rectangles_with_one_gray_cell 40 36 4

end NUMINAMATH_CALUDE_rectangles_with_one_gray_cell_l575_57507


namespace NUMINAMATH_CALUDE_problem_solution_l575_57505

theorem problem_solution : 
  (12345679^2 * 81 - 1) / 11111111 / 10 * 9 - 8 = 10000000000 := by sorry

end NUMINAMATH_CALUDE_problem_solution_l575_57505


namespace NUMINAMATH_CALUDE_average_marks_combined_classes_l575_57515

theorem average_marks_combined_classes (n1 n2 : ℕ) (avg1 avg2 : ℚ) 
  (h1 : n1 = 30) (h2 : n2 = 50) (h3 : avg1 = 40) (h4 : avg2 = 90) :
  (n1 * avg1 + n2 * avg2) / (n1 + n2 : ℚ) = 71.25 := by
  sorry

end NUMINAMATH_CALUDE_average_marks_combined_classes_l575_57515


namespace NUMINAMATH_CALUDE_greatest_integer_x_prime_l575_57550

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

def f (x : ℤ) : ℤ := |6 * x^2 - 47 * x + 15|

theorem greatest_integer_x_prime :
  ∀ x : ℤ, (is_prime (f x).toNat → x ≤ 8) ∧
  (is_prime (f 8).toNat) :=
sorry

end NUMINAMATH_CALUDE_greatest_integer_x_prime_l575_57550


namespace NUMINAMATH_CALUDE_geometric_series_sum_l575_57547

theorem geometric_series_sum : 
  let a : ℝ := (Real.sqrt 2 + 1) / (Real.sqrt 2 - 1)
  let q : ℝ := (Real.sqrt 2 - 1) / Real.sqrt 2
  let S : ℝ := a / (1 - q)
  S = 6 + 4 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l575_57547
