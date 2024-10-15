import Mathlib

namespace NUMINAMATH_CALUDE_det_special_matrix_l2246_224664

/-- The determinant of the matrix [[x+2, x, x], [x, x+2, x], [x, x, x+2]] is equal to 8x + 8 for any real number x. -/
theorem det_special_matrix (x : ℝ) : 
  Matrix.det !![x + 2, x, x; x, x + 2, x; x, x, x + 2] = 8 * x + 8 := by
  sorry

end NUMINAMATH_CALUDE_det_special_matrix_l2246_224664


namespace NUMINAMATH_CALUDE_incorrect_induction_proof_l2246_224690

theorem incorrect_induction_proof (n : ℕ+) : 
  ¬(∀ k : ℕ+, (∀ m : ℕ+, m < k → Real.sqrt (m^2 + m) < m + 1) → 
    Real.sqrt ((k+1)^2 + (k+1)) < (k+1) + 1) := by
  sorry

#check incorrect_induction_proof

end NUMINAMATH_CALUDE_incorrect_induction_proof_l2246_224690


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_by_11_l2246_224611

theorem least_subtraction_for_divisibility_by_11 : 
  ∃ (x : ℕ), x = 7 ∧ 
  (∀ (y : ℕ), y < x → ¬(11 ∣ (427398 - y))) ∧
  (11 ∣ (427398 - x)) := by
sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_by_11_l2246_224611


namespace NUMINAMATH_CALUDE_multiple_of_nine_implies_multiple_of_three_l2246_224610

theorem multiple_of_nine_implies_multiple_of_three (n : ℤ) :
  (∀ m : ℤ, 9 ∣ m → 3 ∣ m) →
  (∃ k : ℤ, n = 9 * k ∧ n % 2 = 1) →
  3 ∣ n :=
by sorry

end NUMINAMATH_CALUDE_multiple_of_nine_implies_multiple_of_three_l2246_224610


namespace NUMINAMATH_CALUDE_class_size_is_fifteen_l2246_224657

-- Define the number of students
def N : ℕ := sorry

-- Define the average age function
def averageAge (numStudents : ℕ) (totalAge : ℕ) : ℚ :=
  totalAge / numStudents

-- Theorem statement
theorem class_size_is_fifteen :
  -- Conditions
  (averageAge (N - 1) (15 * (N - 1)) = 15) →
  (averageAge 4 (14 * 4) = 14) →
  (averageAge 9 (16 * 9) = 16) →
  -- Conclusion
  N = 15 := by
  sorry

end NUMINAMATH_CALUDE_class_size_is_fifteen_l2246_224657


namespace NUMINAMATH_CALUDE_hockey_league_season_games_l2246_224699

/-- The number of games played in a hockey league season -/
def hockey_league_games (n : ℕ) (m : ℕ) : ℕ :=
  n * (n - 1) * m / 2

/-- Theorem: In a hockey league with 15 teams, where each team plays every other team 10 times,
    the total number of games played in the season is 1050. -/
theorem hockey_league_season_games :
  hockey_league_games 15 10 = 1050 := by
  sorry

end NUMINAMATH_CALUDE_hockey_league_season_games_l2246_224699


namespace NUMINAMATH_CALUDE_locus_of_parabola_vertices_l2246_224647

/-- The locus of vertices of parabolas -/
theorem locus_of_parabola_vertices
  (a c : ℝ) (hz : a > 0) (hc : c > 0) :
  ∀ (z : ℝ), ∃ (x_z y_z : ℝ),
    (x_z = -z / (2 * a)) ∧
    (y_z = a * x_z^2 + z * x_z + c) ∧
    (y_z = -a * x_z^2 + c) :=
by sorry

end NUMINAMATH_CALUDE_locus_of_parabola_vertices_l2246_224647


namespace NUMINAMATH_CALUDE_sum_of_coordinates_X_l2246_224672

/-- Given points Y and Z, and the condition that XZ/XY = ZY/XY = 1/3,
    prove that the sum of the coordinates of point X is 10. -/
theorem sum_of_coordinates_X (Y Z X : ℝ × ℝ) : 
  Y = (2, 8) →
  Z = (0, -4) →
  (X.1 - Z.1) / (X.1 - Y.1) = 1/3 →
  (X.2 - Z.2) / (X.2 - Y.2) = 1/3 →
  (Z.1 - Y.1) / (X.1 - Y.1) = 1/3 →
  (Z.2 - Y.2) / (X.2 - Y.2) = 1/3 →
  X.1 + X.2 = 10 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coordinates_X_l2246_224672


namespace NUMINAMATH_CALUDE_hyperbola_and_angle_bisector_l2246_224670

/-- A hyperbola with given properties -/
structure Hyperbola where
  -- Point A lies on the hyperbola
  point_A : ℝ × ℝ
  point_A_on_hyperbola : point_A.1^2 / 4 - point_A.2^2 / 12 = 1
  -- Eccentricity is 2
  eccentricity : ℝ
  eccentricity_eq : eccentricity = 2

/-- The angle bisector of ∠F₁AF₂ -/
def angle_bisector (h : Hyperbola) : ℝ → ℝ := 
  fun x ↦ 2 * x - 2

theorem hyperbola_and_angle_bisector (h : Hyperbola) 
  (h_point_A : h.point_A = (4, 6)) :
  (∀ x y : ℝ, x^2 / 4 - y^2 / 12 = 1 ↔ (x, y) ∈ {p : ℝ × ℝ | p.1^2 / 4 - p.2^2 / 12 = 1}) ∧
  (∀ x : ℝ, angle_bisector h x = 2 * x - 2) := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_and_angle_bisector_l2246_224670


namespace NUMINAMATH_CALUDE_divisibility_implication_l2246_224608

theorem divisibility_implication (a b : ℤ) :
  (∃ k : ℤ, a^2 + 9*a*b + b^2 = 11*k) → (∃ m : ℤ, a^2 - b^2 = 11*m) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_implication_l2246_224608


namespace NUMINAMATH_CALUDE_function_extrema_implies_a_range_l2246_224696

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + 3*a*x^2 + 3*(a+2)*x + 1

-- State the theorem
theorem function_extrema_implies_a_range (a : ℝ) :
  (∃ (max min : ℝ), ∀ x, f a x ≤ max ∧ f a x ≥ min) →
  (a < -1 ∨ a > 2) :=
by sorry

end NUMINAMATH_CALUDE_function_extrema_implies_a_range_l2246_224696


namespace NUMINAMATH_CALUDE_least_subtrahend_l2246_224618

def problem (n : ℕ) : Prop :=
  (2590 - n) % 9 = 6 ∧ 
  (2590 - n) % 11 = 6 ∧ 
  (2590 - n) % 13 = 6

theorem least_subtrahend : 
  problem 10 ∧ ∀ m : ℕ, m < 10 → ¬(problem m) :=
by sorry

end NUMINAMATH_CALUDE_least_subtrahend_l2246_224618


namespace NUMINAMATH_CALUDE_min_omega_for_even_shifted_sine_l2246_224661

/-- Given a function g and a real number ω, this theorem states that
    if g is defined as g(x) = sin(ω(x - π/3) + π/6),
    ω is positive, and g is an even function,
    then the minimum value of ω is 2. -/
theorem min_omega_for_even_shifted_sine (g : ℝ → ℝ) (ω : ℝ) :
  (∀ x, g x = Real.sin (ω * (x - Real.pi / 3) + Real.pi / 6)) →
  ω > 0 →
  (∀ x, g x = g (-x)) →
  ω ≥ 2 ∧ ∃ ω₀, ω₀ = 2 ∧ 
    (∀ x, Real.sin (ω₀ * (x - Real.pi / 3) + Real.pi / 6) = 
          Real.sin (ω₀ * ((-x) - Real.pi / 3) + Real.pi / 6)) :=
by sorry

end NUMINAMATH_CALUDE_min_omega_for_even_shifted_sine_l2246_224661


namespace NUMINAMATH_CALUDE_max_a_for_monotonic_f_l2246_224653

/-- Given a function f(x) = x^3 - ax that is monotonically increasing on [1, +∞),
    the maximum value of a is 3. -/
theorem max_a_for_monotonic_f (a : ℝ) : 
  (∀ x ≥ 1, ∀ y ≥ x, (x^3 - a*x) ≤ (y^3 - a*y)) → a ≤ 3 :=
sorry

end NUMINAMATH_CALUDE_max_a_for_monotonic_f_l2246_224653


namespace NUMINAMATH_CALUDE_min_value_log_sum_equality_condition_l2246_224637

theorem min_value_log_sum (x : ℝ) (h : x > 1) :
  (Real.log 9 / Real.log x) + (Real.log x / Real.log 27) ≥ 2 * Real.sqrt 6 / 3 :=
by sorry

theorem equality_condition (x : ℝ) (h : x > 1) :
  (Real.log 9 / Real.log x) + (Real.log x / Real.log 27) = 2 * Real.sqrt 6 / 3 ↔ x = 3 * Real.sqrt 6 :=
by sorry

end NUMINAMATH_CALUDE_min_value_log_sum_equality_condition_l2246_224637


namespace NUMINAMATH_CALUDE_quadratic_factorization_l2246_224633

theorem quadratic_factorization (x : ℂ) : 
  2 * x^2 + 8 * x + 26 = 2 * (x + 2 - 3 * I) * (x + 2 + 3 * I) := by
sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l2246_224633


namespace NUMINAMATH_CALUDE_bridget_apples_proof_l2246_224674

/-- The number of apples Bridget bought -/
def total_apples : ℕ := 21

/-- The number of apples Bridget gave to Cassie and Dan -/
def apples_to_cassie_and_dan : ℕ := 7

/-- The number of apples Bridget kept for herself -/
def apples_kept : ℕ := 7

theorem bridget_apples_proof :
  total_apples = 21 ∧
  (2 * total_apples) / 3 = apples_to_cassie_and_dan + apples_kept :=
by sorry

end NUMINAMATH_CALUDE_bridget_apples_proof_l2246_224674


namespace NUMINAMATH_CALUDE_geometric_series_product_l2246_224651

theorem geometric_series_product (y : ℝ) : 
  (∑' n, (1/3)^n) * (∑' n, (-1/3)^n) = ∑' n, (1/y)^n → y = 9 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_product_l2246_224651


namespace NUMINAMATH_CALUDE_linda_remaining_candies_l2246_224634

/-- The number of candies Linda has left after giving some away -/
def candies_left (initial : ℝ) (given_away : ℝ) : ℝ := initial - given_away

/-- Theorem stating that Linda's remaining candies is the difference between initial and given away -/
theorem linda_remaining_candies (initial : ℝ) (given_away : ℝ) :
  candies_left initial given_away = initial - given_away :=
by sorry

end NUMINAMATH_CALUDE_linda_remaining_candies_l2246_224634


namespace NUMINAMATH_CALUDE_initial_cats_is_28_l2246_224648

/-- Represents the animal shelter scenario --/
structure AnimalShelter where
  initialDogs : ℕ
  initialLizards : ℕ
  dogAdoptionRate : ℚ
  catAdoptionRate : ℚ
  lizardAdoptionRate : ℚ
  newPetsPerMonth : ℕ
  totalPetsAfterOneMonth : ℕ

/-- Calculates the initial number of cats in the shelter --/
def calculateInitialCats (shelter : AnimalShelter) : ℚ :=
  let remainingDogs : ℚ := shelter.initialDogs * (1 - shelter.dogAdoptionRate)
  let remainingLizards : ℚ := shelter.initialLizards * (1 - shelter.lizardAdoptionRate)
  let nonCatPets : ℚ := remainingDogs + remainingLizards + shelter.newPetsPerMonth
  let remainingCats : ℚ := shelter.totalPetsAfterOneMonth - nonCatPets
  remainingCats / (1 - shelter.catAdoptionRate)

/-- Theorem stating that the initial number of cats is 28 --/
theorem initial_cats_is_28 (shelter : AnimalShelter) 
  (h1 : shelter.initialDogs = 30)
  (h2 : shelter.initialLizards = 20)
  (h3 : shelter.dogAdoptionRate = 1/2)
  (h4 : shelter.catAdoptionRate = 1/4)
  (h5 : shelter.lizardAdoptionRate = 1/5)
  (h6 : shelter.newPetsPerMonth = 13)
  (h7 : shelter.totalPetsAfterOneMonth = 65) :
  calculateInitialCats shelter = 28 := by
  sorry

#eval calculateInitialCats {
  initialDogs := 30,
  initialLizards := 20,
  dogAdoptionRate := 1/2,
  catAdoptionRate := 1/4,
  lizardAdoptionRate := 1/5,
  newPetsPerMonth := 13,
  totalPetsAfterOneMonth := 65
}

end NUMINAMATH_CALUDE_initial_cats_is_28_l2246_224648


namespace NUMINAMATH_CALUDE_root_sum_pq_l2246_224659

theorem root_sum_pq (p q : ℝ) : 
  (2 * Complex.I ^ 2 + p * Complex.I + q = 0) →
  (2 * (-3 + 2 * Complex.I) ^ 2 + p * (-3 + 2 * Complex.I) + q = 0) →
  p + q = 38 := by
sorry

end NUMINAMATH_CALUDE_root_sum_pq_l2246_224659


namespace NUMINAMATH_CALUDE_election_winner_votes_l2246_224621

theorem election_winner_votes (total_votes : ℕ) (winner_percentage : ℚ) (vote_difference : ℕ) : 
  (winner_percentage = 62 / 100) →
  (winner_percentage * total_votes - (1 - winner_percentage) * total_votes = vote_difference) →
  (vote_difference = 300) →
  (winner_percentage * total_votes = 775) := by
  sorry

end NUMINAMATH_CALUDE_election_winner_votes_l2246_224621


namespace NUMINAMATH_CALUDE_interchanged_digits_theorem_l2246_224627

theorem interchanged_digits_theorem (n m a b : ℕ) : 
  n = 10 * a + b → 
  n = m * (a + b + a) → 
  10 * b + a = (9 - m) * (a + b) := by
  sorry

end NUMINAMATH_CALUDE_interchanged_digits_theorem_l2246_224627


namespace NUMINAMATH_CALUDE_selling_cheat_theorem_l2246_224620

/-- Represents a shop owner's pricing strategy -/
structure ShopOwner where
  buying_cheat_percent : ℝ
  profit_percent : ℝ

/-- Calculates the percentage by which the shop owner cheats while selling -/
def selling_cheat_percent (owner : ShopOwner) : ℝ :=
  -- The actual implementation is not provided here
  sorry

/-- Theorem stating the selling cheat percentage for a specific shop owner -/
theorem selling_cheat_theorem (owner : ShopOwner) 
  (h1 : owner.buying_cheat_percent = 12)
  (h2 : owner.profit_percent = 60) :
  selling_cheat_percent owner = 60 := by
  sorry

end NUMINAMATH_CALUDE_selling_cheat_theorem_l2246_224620


namespace NUMINAMATH_CALUDE_quadratic_inequality_properties_l2246_224636

-- Define the quadratic function
def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the solution set
def solution_set (a b c : ℝ) : Set ℝ := {x | f a b c x > 0}

-- State the theorem
theorem quadratic_inequality_properties (a b c : ℝ) :
  solution_set a b c = Set.Ioo (-1/2 : ℝ) 2 →
  a < 0 ∧ b > 0 ∧ c > 0 ∧ a + b + c > 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_properties_l2246_224636


namespace NUMINAMATH_CALUDE_digit_concatenation_divisibility_l2246_224635

theorem digit_concatenation_divisibility (n : ℕ) (a : ℕ) (h1 : n > 1) (h2 : 10^(n-1) ≤ a) (h3 : a < 10^n) :
  let b := a * (10^n + 1)
  ∃! k : ℕ, k > 1 ∧ k ≤ 10 ∧ b = k * a^2 ∧ k = 7 :=
sorry

end NUMINAMATH_CALUDE_digit_concatenation_divisibility_l2246_224635


namespace NUMINAMATH_CALUDE_quadratic_minimum_l2246_224650

theorem quadratic_minimum (x : ℝ) : x^2 + 6*x + 5 ≥ -4 ∧ ∃ y : ℝ, y^2 + 6*y + 5 = -4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l2246_224650


namespace NUMINAMATH_CALUDE_problem_solution_l2246_224626

theorem problem_solution (a b c d : ℝ) :
  a^2 + b^2 + c^2 + 2 = d + Real.sqrt (a + b + c - 2*d) →
  d = -1/8 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l2246_224626


namespace NUMINAMATH_CALUDE_burger_expenditure_l2246_224656

theorem burger_expenditure (total : ℝ) (movie_frac ice_cream_frac music_frac : ℚ) :
  total = 50 ∧
  movie_frac = 1/4 ∧
  ice_cream_frac = 1/6 ∧
  music_frac = 1/3 →
  total - (movie_frac + ice_cream_frac + music_frac) * total = 12.5 := by
  sorry

end NUMINAMATH_CALUDE_burger_expenditure_l2246_224656


namespace NUMINAMATH_CALUDE_sqrt_sum_simplification_l2246_224630

theorem sqrt_sum_simplification : 
  Real.sqrt (4 + 2 * Real.sqrt 3) + Real.sqrt (4 - 2 * Real.sqrt 3) = 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_simplification_l2246_224630


namespace NUMINAMATH_CALUDE_calculate_markup_l2246_224619

/-- Calculate the markup for an article given its purchase price, overhead percentage, and desired net profit. -/
theorem calculate_markup (purchase_price overhead_percent net_profit : ℚ) 
  (h1 : purchase_price = 48)
  (h2 : overhead_percent = 10 / 100)
  (h3 : net_profit = 12) :
  purchase_price * overhead_percent + purchase_price + net_profit - purchase_price = 168 / 10 := by
  sorry

end NUMINAMATH_CALUDE_calculate_markup_l2246_224619


namespace NUMINAMATH_CALUDE_vacation_cost_division_l2246_224602

theorem vacation_cost_division (total_cost : ℕ) (n : ℕ) : 
  total_cost = 1000 → 
  (total_cost / 5 + 50 = total_cost / n) → 
  n = 4 := by
sorry

end NUMINAMATH_CALUDE_vacation_cost_division_l2246_224602


namespace NUMINAMATH_CALUDE_lines_are_skew_l2246_224617

-- Define the concept of lines in 3D space
variable (Line : Type)

-- Define the properties of lines
variable (parallel : Line → Line → Prop)
variable (intersect : Line → Line → Prop)
variable (skew : Line → Line → Prop)

-- Theorem statement
theorem lines_are_skew (a b : Line) : 
  (¬ parallel a b) → (¬ intersect a b) → skew a b :=
by sorry

end NUMINAMATH_CALUDE_lines_are_skew_l2246_224617


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l2246_224693

theorem solution_set_of_inequality (x : ℝ) : 
  (x + 1) / (x - 3) < 0 ↔ -1 < x ∧ x < 3 :=
sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l2246_224693


namespace NUMINAMATH_CALUDE_some_base_value_l2246_224677

theorem some_base_value (k : ℕ) (some_base : ℝ) 
  (h1 : (1/2)^16 * (1/some_base)^k = 1/(18^16))
  (h2 : k = 8) : 
  some_base = 81 := by
sorry

end NUMINAMATH_CALUDE_some_base_value_l2246_224677


namespace NUMINAMATH_CALUDE_line_through_points_m_plus_b_l2246_224655

/-- Given a line passing through points (1, 3) and (3, 7) that follows the equation y = mx + b,
    prove that m + b = 3 -/
theorem line_through_points_m_plus_b (m b : ℝ) : 
  (3 : ℝ) = m * (1 : ℝ) + b ∧ 
  (7 : ℝ) = m * (3 : ℝ) + b → 
  m + b = 3 := by
  sorry

end NUMINAMATH_CALUDE_line_through_points_m_plus_b_l2246_224655


namespace NUMINAMATH_CALUDE_largest_sum_l2246_224673

theorem largest_sum : 
  let sum1 := (1/4 : ℚ) + (1/5 : ℚ)
  let sum2 := (1/4 : ℚ) + (1/6 : ℚ)
  let sum3 := (1/4 : ℚ) + (1/3 : ℚ)
  let sum4 := (1/4 : ℚ) + (1/8 : ℚ)
  let sum5 := (1/4 : ℚ) + (1/7 : ℚ)
  sum3 > sum1 ∧ sum3 > sum2 ∧ sum3 > sum4 ∧ sum3 > sum5 := by
  sorry

end NUMINAMATH_CALUDE_largest_sum_l2246_224673


namespace NUMINAMATH_CALUDE_sqrt_equals_self_implies_zero_or_one_l2246_224676

theorem sqrt_equals_self_implies_zero_or_one (x : ℝ) : Real.sqrt x = x → x = 0 ∨ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equals_self_implies_zero_or_one_l2246_224676


namespace NUMINAMATH_CALUDE_fourteenth_root_of_unity_l2246_224658

theorem fourteenth_root_of_unity : 
  (Complex.tan (π / 7) + Complex.I) / (Complex.tan (π / 7) - Complex.I) = 
  Complex.exp (Complex.I * (4 * π / 14)) := by
  sorry

end NUMINAMATH_CALUDE_fourteenth_root_of_unity_l2246_224658


namespace NUMINAMATH_CALUDE_range_of_a_l2246_224616

-- Define the inequality as a function of x and a
def inequality (x a : ℝ) : Prop := 2 * x^2 + a * x - a^2 > 0

-- Define the theorem
theorem range_of_a : 
  (∃ a : ℝ, inequality 2 a) → 
  (∀ a : ℝ, inequality 2 a ↔ -2 < a ∧ a < 4) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l2246_224616


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_sum_of_divisors_450_l2246_224666

-- Define the sum of divisors function
def sum_of_divisors (n : ℕ) : ℕ := sorry

-- Define M as the sum of divisors of 450
def M : ℕ := sum_of_divisors 450

-- Define a function to get the largest prime factor
def largest_prime_factor (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem largest_prime_factor_of_sum_of_divisors_450 :
  largest_prime_factor M = 13 := by sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_sum_of_divisors_450_l2246_224666


namespace NUMINAMATH_CALUDE_stonewall_band_max_members_l2246_224663

theorem stonewall_band_max_members :
  ∃ (max : ℕ),
    (∀ n : ℕ, (30 * n) % 34 = 2 → 30 * n < 1500 → 30 * n ≤ max) ∧
    (∃ n : ℕ, (30 * n) % 34 = 2 ∧ 30 * n < 1500 ∧ 30 * n = max) ∧
    max = 1260 := by
  sorry

end NUMINAMATH_CALUDE_stonewall_band_max_members_l2246_224663


namespace NUMINAMATH_CALUDE_monotone_increasing_range_a_inequality_for_m_n_l2246_224685

noncomputable section

def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - a * (x - 1) / (x + 1)

theorem monotone_increasing_range_a :
  ∀ a : ℝ, (∀ x : ℝ, x > 0 → Monotone (f a)) → a ≤ 2 := by sorry

theorem inequality_for_m_n :
  ∀ m n : ℝ, m ≠ n → (m - n) / (Real.log m - Real.log n) < (m + n) / 2 := by sorry

end NUMINAMATH_CALUDE_monotone_increasing_range_a_inequality_for_m_n_l2246_224685


namespace NUMINAMATH_CALUDE_mothers_biscuits_l2246_224675

/-- Represents the number of biscuits in Randy's scenario -/
structure BiscuitCount where
  initial : Nat
  fromFather : Nat
  fromMother : Nat
  eatenByBrother : Nat
  final : Nat

/-- Calculates the total number of biscuits Randy had before his brother ate some -/
def totalBeforeEating (b : BiscuitCount) : Nat :=
  b.initial + b.fromFather + b.fromMother

/-- Theorem: Randy's mother gave him 15 biscuits -/
theorem mothers_biscuits (b : BiscuitCount) 
  (h1 : b.initial = 32)
  (h2 : b.fromFather = 13)
  (h3 : b.eatenByBrother = 20)
  (h4 : b.final = 40)
  (h5 : totalBeforeEating b = b.final + b.eatenByBrother) : 
  b.fromMother = 15 := by
  sorry

#check mothers_biscuits

end NUMINAMATH_CALUDE_mothers_biscuits_l2246_224675


namespace NUMINAMATH_CALUDE_unique_number_meeting_conditions_l2246_224665

theorem unique_number_meeting_conditions : ∃! n : ℕ+, 
  (((n < 12) ∨ (¬ 7 ∣ n) ∨ (5 * n < 70)) ∧ 
   ¬((n < 12) ∧ (¬ 7 ∣ n) ∧ (5 * n < 70))) ∧
  (((12 * n > 1000) ∨ (10 ∣ n) ∨ (n > 100)) ∧ 
   ¬((12 * n > 1000) ∧ (10 ∣ n) ∧ (n > 100))) ∧
  (((4 ∣ n) ∨ (11 * n < 1000) ∨ (9 ∣ n)) ∧ 
   ¬((4 ∣ n) ∧ (11 * n < 1000) ∧ (9 ∣ n))) ∧
  (((n < 20) ∨ Nat.Prime n ∨ (7 ∣ n)) ∧ 
   ¬((n < 20) ∧ Nat.Prime n ∧ (7 ∣ n))) ∧
  n = 89 := by
  sorry

end NUMINAMATH_CALUDE_unique_number_meeting_conditions_l2246_224665


namespace NUMINAMATH_CALUDE_unique_code_l2246_224606

/-- Represents a three-digit code --/
structure Code where
  A : Nat
  B : Nat
  C : Nat
  h1 : A < 10
  h2 : B < 10
  h3 : C < 10
  h4 : A ≠ B
  h5 : A ≠ C
  h6 : B ≠ C

/-- The conditions for the code --/
def satisfiesConditions (code : Code) : Prop :=
  code.B > code.A ∧
  code.A < code.C ∧
  code.B * 10 + code.B + code.A * 10 + code.A = code.C * 10 + code.C ∧
  code.B * 10 + code.B + code.A * 10 + code.A = 242

theorem unique_code :
  ∃! code : Code, satisfiesConditions code ∧ code.A = 2 ∧ code.B = 3 ∧ code.C = 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_code_l2246_224606


namespace NUMINAMATH_CALUDE_coefficients_of_equation_l2246_224671

-- Define the coefficients of a quadratic equation
def QuadraticCoefficients := ℝ × ℝ × ℝ

-- Function to get coefficients from a quadratic equation
def getCoefficients (a b c : ℝ) : QuadraticCoefficients := (a, b, c)

-- Theorem stating that the coefficients of 2x^2 - 6x = 9 are (2, -6, -9)
theorem coefficients_of_equation : 
  let eq := fun x : ℝ => 2 * x^2 - 6 * x - 9
  getCoefficients 2 (-6) (-9) = (2, -6, -9) := by sorry

end NUMINAMATH_CALUDE_coefficients_of_equation_l2246_224671


namespace NUMINAMATH_CALUDE_pencils_remaining_l2246_224684

theorem pencils_remaining (initial_pencils : ℕ) (removed_pencils : ℕ) 
  (h1 : initial_pencils = 87) 
  (h2 : removed_pencils = 4) : 
  initial_pencils - removed_pencils = 83 := by
  sorry

end NUMINAMATH_CALUDE_pencils_remaining_l2246_224684


namespace NUMINAMATH_CALUDE_vertex_of_quadratic_l2246_224643

def f (x : ℝ) : ℝ := -3 * x^2 + 2

theorem vertex_of_quadratic (x : ℝ) :
  (∀ x, f x ≤ f 0) ∧ f 0 = 2 := by sorry

end NUMINAMATH_CALUDE_vertex_of_quadratic_l2246_224643


namespace NUMINAMATH_CALUDE_sum_plus_ten_is_three_times_square_l2246_224640

theorem sum_plus_ten_is_three_times_square (n : ℤ) (h : n ≠ 0) : 
  ∃ (m : ℤ), (n - 1)^4 + n^4 + (n + 1)^4 + 10 = 3 * m^2 := by
  sorry

end NUMINAMATH_CALUDE_sum_plus_ten_is_three_times_square_l2246_224640


namespace NUMINAMATH_CALUDE_probability_is_one_third_l2246_224607

/-- A game board consisting of an equilateral triangle divided into six smaller triangles -/
structure GameBoard where
  /-- The total number of smaller triangles in the game board -/
  total_triangles : ℕ
  /-- The number of shaded triangles in the game board -/
  shaded_triangles : ℕ
  /-- The shaded triangles are non-adjacent -/
  non_adjacent : Bool
  /-- The total number of triangles is 6 -/
  h_total : total_triangles = 6
  /-- The number of shaded triangles is 2 -/
  h_shaded : shaded_triangles = 2
  /-- The shaded triangles are indeed non-adjacent -/
  h_non_adjacent : non_adjacent = true

/-- The probability of a spinner landing in a shaded region -/
def probability (board : GameBoard) : ℚ :=
  board.shaded_triangles / board.total_triangles

/-- Theorem stating that the probability of landing in a shaded region is 1/3 -/
theorem probability_is_one_third (board : GameBoard) : probability board = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_probability_is_one_third_l2246_224607


namespace NUMINAMATH_CALUDE_sin_inequality_solution_set_l2246_224681

theorem sin_inequality_solution_set (a : ℝ) (θ : ℝ) (h1 : -1 < a) (h2 : a < 0) (h3 : θ = Real.arcsin a) :
  {x : ℝ | ∃ n : ℤ, (2*n - 1)*Real.pi - θ < x ∧ x < 2*n*Real.pi + θ} = {x : ℝ | Real.sin x < a} := by
  sorry

end NUMINAMATH_CALUDE_sin_inequality_solution_set_l2246_224681


namespace NUMINAMATH_CALUDE_max_xy_value_l2246_224646

theorem max_xy_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 9*y = 12) :
  ∀ z : ℝ, z = x * y → z ≤ 4 :=
by
  sorry

end NUMINAMATH_CALUDE_max_xy_value_l2246_224646


namespace NUMINAMATH_CALUDE_lcm_problem_l2246_224652

theorem lcm_problem (m : ℕ+) (h1 : Nat.lcm 30 m = 90) (h2 : Nat.lcm m 45 = 180) : m = 36 := by
  sorry

end NUMINAMATH_CALUDE_lcm_problem_l2246_224652


namespace NUMINAMATH_CALUDE_prob_B_wins_third_round_correct_A_has_lower_expected_additional_time_l2246_224612

-- Define the probabilities of answering correctly for each participant
def prob_correct_A : ℚ := 3/5
def prob_correct_B : ℚ := 2/3

-- Define the number of rounds and questions per round
def num_rounds : ℕ := 3
def questions_per_round : ℕ := 3

-- Define the time penalty for an incorrect answer
def time_penalty : ℕ := 20

-- Define the time difference in recitation per round
def recitation_time_diff : ℕ := 10

-- Define the function to calculate the probability of B winning in the third round
def prob_B_wins_third_round : ℚ := 448/3375

-- Define the expected number of incorrect answers for each participant
def expected_incorrect_A : ℚ := (1 - prob_correct_A) * (num_rounds * questions_per_round : ℚ)
def expected_incorrect_B : ℚ := (1 - prob_correct_B) * (num_rounds * questions_per_round : ℚ)

-- Theorem: The probability of B winning in the third round is 448/3375
theorem prob_B_wins_third_round_correct :
  prob_B_wins_third_round = 448/3375 := by sorry

-- Theorem: A has a lower expected additional time due to incorrect answers
theorem A_has_lower_expected_additional_time :
  expected_incorrect_A * time_penalty < expected_incorrect_B * time_penalty + (num_rounds * recitation_time_diff : ℚ) := by sorry

end NUMINAMATH_CALUDE_prob_B_wins_third_round_correct_A_has_lower_expected_additional_time_l2246_224612


namespace NUMINAMATH_CALUDE_smallest_n_for_book_pricing_l2246_224691

theorem smallest_n_for_book_pricing : 
  ∀ n : ℕ+, (∃ x : ℕ+, (105 * x : ℕ) = 100 * n) → n ≥ 21 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_for_book_pricing_l2246_224691


namespace NUMINAMATH_CALUDE_work_completion_time_l2246_224680

/-- The time taken to complete a work when three workers with given efficiencies work together -/
theorem work_completion_time 
  (total_work : ℝ) 
  (efficiency_x efficiency_y efficiency_z : ℝ) 
  (hx : efficiency_x = 1 / 20)
  (hy : efficiency_y = 3 / 80)
  (hz : efficiency_z = 3 / 40)
  (h_total : total_work = 1) :
  (total_work / (efficiency_x + efficiency_y + efficiency_z)) = 80 / 13 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l2246_224680


namespace NUMINAMATH_CALUDE_problem_solution_l2246_224600

-- Define the conditions p and q
def p (x : ℝ) : Prop := 4 * x^2 + 12 * x - 7 ≤ 0
def q (a x : ℝ) : Prop := a - 3 ≤ x ∧ x ≤ a + 3

theorem problem_solution :
  (∃ x : ℝ, p x ∧ ¬(q 0 x) ∧ -7/2 ≤ x ∧ x < -3) ∧
  (∀ a : ℝ, (∀ x : ℝ, p x → q a x) ↔ -5/2 ≤ a ∧ a ≤ 1/2) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2246_224600


namespace NUMINAMATH_CALUDE_units_digit_sum_base7_l2246_224639

/-- Represents a number in base 7 --/
def Base7 : Type := ℕ

/-- Converts a natural number to its representation in base 7 --/
def toBase7 (n : ℕ) : Base7 := sorry

/-- Adds two numbers in base 7 --/
def addBase7 (a b : Base7) : Base7 := sorry

/-- Gets the units digit of a number in base 7 --/
def unitsDigitBase7 (n : Base7) : ℕ := sorry

theorem units_digit_sum_base7 :
  unitsDigitBase7 (addBase7 (toBase7 65) (toBase7 34)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_sum_base7_l2246_224639


namespace NUMINAMATH_CALUDE_workshop_workers_count_l2246_224628

theorem workshop_workers_count :
  let average_salary : ℕ := 9000
  let technician_count : ℕ := 7
  let technician_salary : ℕ := 12000
  let non_technician_salary : ℕ := 6000
  ∃ (total_workers : ℕ),
    total_workers * average_salary = 
      technician_count * technician_salary + 
      (total_workers - technician_count) * non_technician_salary ∧
    total_workers = 14 :=
by
  sorry

end NUMINAMATH_CALUDE_workshop_workers_count_l2246_224628


namespace NUMINAMATH_CALUDE_equation_solutions_l2246_224624

def equation (x : ℝ) : Prop :=
  1 / ((x - 1) * (x - 2)) + 1 / ((x - 2) * (x - 3)) + 
  1 / ((x - 3) * (x - 4)) + 1 / ((x - 4) * (x - 5)) = 1 / 10

theorem equation_solutions :
  ∀ x : ℝ, equation x ↔ x = 10 ∨ x = -3.5 := by sorry

end NUMINAMATH_CALUDE_equation_solutions_l2246_224624


namespace NUMINAMATH_CALUDE_square_equality_l2246_224614

theorem square_equality : (2023 + (-1011.5))^2 = (-1011.5)^2 := by
  sorry

end NUMINAMATH_CALUDE_square_equality_l2246_224614


namespace NUMINAMATH_CALUDE_ny_striploin_cost_l2246_224609

theorem ny_striploin_cost (total_bill : ℝ) (tax_rate : ℝ) (wine_cost : ℝ) (gratuities : ℝ) :
  total_bill = 140 →
  tax_rate = 0.1 →
  wine_cost = 10 →
  gratuities = 41 →
  ∃ (striploin_cost : ℝ), abs (striploin_cost - 71.82) < 0.01 :=
by
  sorry

end NUMINAMATH_CALUDE_ny_striploin_cost_l2246_224609


namespace NUMINAMATH_CALUDE_prob_not_losing_l2246_224641

/-- Given a chess game between players A and B, this theorem proves
    the probability of A not losing, given the probabilities of a draw
    and A winning. -/
theorem prob_not_losing (p_draw p_win : ℝ) : 
  p_draw = 1/2 → p_win = 1/3 → p_draw + p_win = 5/6 := by
  sorry

end NUMINAMATH_CALUDE_prob_not_losing_l2246_224641


namespace NUMINAMATH_CALUDE_sqrt_sum_squares_geq_sqrt2_sum_l2246_224615

theorem sqrt_sum_squares_geq_sqrt2_sum (a b : ℝ) : 
  Real.sqrt (a^2 + b^2) ≥ (Real.sqrt 2 / 2) * (a + b) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_squares_geq_sqrt2_sum_l2246_224615


namespace NUMINAMATH_CALUDE_correlation_difference_l2246_224644

/-- Represents a relationship between two variables -/
structure Relationship where
  var1 : String
  var2 : String
  description : String

/-- Determines if a relationship represents a positive correlation -/
def is_positive_correlation (r : Relationship) : Bool :=
  sorry  -- The actual implementation would depend on how we define positive correlation

/-- Given set of relationships -/
def relationships : List Relationship := [
  { var1 := "teacher quality", var2 := "student performance", description := "A great teacher produces outstanding students" },
  { var1 := "tide level", var2 := "boat height", description := "A rising tide lifts all boats" },
  { var1 := "moon brightness", var2 := "visible stars", description := "The brighter the moon, the fewer the stars" },
  { var1 := "climbing height", var2 := "viewing distance", description := "Climbing high to see far" }
]

theorem correlation_difference :
  ∃ (i : Fin 4), ¬(is_positive_correlation (relationships.get i)) ∧
    (∀ (j : Fin 4), j ≠ i → is_positive_correlation (relationships.get j)) :=
  sorry

end NUMINAMATH_CALUDE_correlation_difference_l2246_224644


namespace NUMINAMATH_CALUDE_muffin_distribution_l2246_224695

theorem muffin_distribution (total_students : ℕ) (absent_students : ℕ) (extra_muffins : ℕ) : 
  total_students = 400 →
  absent_students = 180 →
  extra_muffins = 36 →
  (total_students * ((total_students - absent_students) * extra_muffins + total_students * (total_students - absent_students))) / 
  ((total_students - absent_students) * total_students) = 80 := by
sorry

end NUMINAMATH_CALUDE_muffin_distribution_l2246_224695


namespace NUMINAMATH_CALUDE_unit_vector_of_a_l2246_224688

/-- Given a vector a = (2, √5), prove that its unit vector is (2/3, √5/3) -/
theorem unit_vector_of_a (a : ℝ × ℝ) (h : a = (2, Real.sqrt 5)) :
  let norm_a := Real.sqrt (a.1^2 + a.2^2)
  (a.1 / norm_a, a.2 / norm_a) = (2/3, Real.sqrt 5 / 3) := by
  sorry

end NUMINAMATH_CALUDE_unit_vector_of_a_l2246_224688


namespace NUMINAMATH_CALUDE_no_such_function_exists_l2246_224698

theorem no_such_function_exists : ¬∃ (f : ℝ → ℝ), 
  (∀ x, f x ≠ 0) ∧ 
  (∀ x, 2 * f (f x) = f x) ∧ 
  (∀ x, f x ≥ 0) ∧
  Differentiable ℝ f :=
by sorry

end NUMINAMATH_CALUDE_no_such_function_exists_l2246_224698


namespace NUMINAMATH_CALUDE_max_removable_edges_50x600_l2246_224687

/-- Represents a rectangular grid -/
structure RectangularGrid where
  rows : ℕ
  cols : ℕ

/-- Calculates the number of vertices in a rectangular grid -/
def vertexCount (grid : RectangularGrid) : ℕ :=
  (grid.rows + 1) * (grid.cols + 1)

/-- Calculates the number of edges in a rectangular grid -/
def edgeCount (grid : RectangularGrid) : ℕ :=
  grid.rows * (grid.cols + 1) + grid.cols * (grid.rows + 1)

/-- Calculates the maximum number of removable edges while keeping the graph connected -/
def maxRemovableEdges (grid : RectangularGrid) : ℕ :=
  edgeCount grid - (vertexCount grid - 1)

/-- Theorem: The maximum number of removable edges in a 50 × 600 grid is 30000 -/
theorem max_removable_edges_50x600 :
  maxRemovableEdges ⟨50, 600⟩ = 30000 := by
  sorry

end NUMINAMATH_CALUDE_max_removable_edges_50x600_l2246_224687


namespace NUMINAMATH_CALUDE_base_conversion_subtraction_l2246_224694

/-- Converts a number from base b to base 10 -/
def to_base_10 (digits : List Nat) (b : Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * b^i) 0

theorem base_conversion_subtraction :
  let base_9_number := [4, 2, 3]  -- 324 in base 9 (least significant digit first)
  let base_6_number := [6, 5, 1]  -- 156 in base 6 (least significant digit first)
  (to_base_10 base_9_number 9) - (to_base_10 base_6_number 6) = 193 := by
sorry

end NUMINAMATH_CALUDE_base_conversion_subtraction_l2246_224694


namespace NUMINAMATH_CALUDE_max_odd_numbers_in_pyramid_l2246_224649

/-- Represents a number pyramid where each number above the bottom row
    is the sum of the two numbers immediately below it. -/
structure NumberPyramid where
  rows : Nat
  cells : Nat

/-- Represents the maximum number of odd numbers that can be placed in a number pyramid. -/
def maxOddNumbers (pyramid : NumberPyramid) : Nat :=
  14

/-- Theorem stating that the maximum number of odd numbers in a number pyramid is 14. -/
theorem max_odd_numbers_in_pyramid (pyramid : NumberPyramid) :
  maxOddNumbers pyramid = 14 := by
  sorry

end NUMINAMATH_CALUDE_max_odd_numbers_in_pyramid_l2246_224649


namespace NUMINAMATH_CALUDE_perpendicular_lines_m_value_l2246_224631

/-- Given two lines l₁ and l₂, prove that if they are perpendicular, then m = 1/2 -/
theorem perpendicular_lines_m_value (m : ℝ) :
  let l₁ := {(x, y) : ℝ × ℝ | m * x + y - 1 = 0}
  let l₂ := {(x, y) : ℝ × ℝ | x + (m - 1) * y + 2 = 0}
  (∀ (p₁ p₂ q₁ q₂ : ℝ × ℝ), p₁ ∈ l₁ → p₂ ∈ l₁ → q₁ ∈ l₂ → q₂ ∈ l₂ → 
    p₁ ≠ p₂ → q₁ ≠ q₂ → (p₁.1 - p₂.1) * (q₁.1 - q₂.1) + (p₁.2 - p₂.2) * (q₁.2 - q₂.2) = 0) →
  m = 1 / 2 :=
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_m_value_l2246_224631


namespace NUMINAMATH_CALUDE_cades_marbles_l2246_224622

/-- The number of marbles Cade has after receiving marbles from Dylan and Ellie -/
def total_marbles (initial : ℕ) (from_dylan : ℕ) (from_ellie : ℕ) : ℕ :=
  initial + from_dylan + from_ellie

/-- Theorem stating that Cade's total marbles after receiving from Dylan and Ellie is 108 -/
theorem cades_marbles :
  total_marbles 87 8 13 = 108 := by
  sorry

end NUMINAMATH_CALUDE_cades_marbles_l2246_224622


namespace NUMINAMATH_CALUDE_imaginary_part_sum_l2246_224662

theorem imaginary_part_sum (z₁ z₂ : ℂ) : z₁ = (1 : ℂ) / (-2 + Complex.I) ∧ z₂ = (1 : ℂ) / (1 - 2*Complex.I) →
  Complex.im (z₁ + z₂) = (1 : ℝ) / 5 := by
sorry

end NUMINAMATH_CALUDE_imaginary_part_sum_l2246_224662


namespace NUMINAMATH_CALUDE_sequence_properties_l2246_224654

/-- Definition of an arithmetic sequence -/
def is_arithmetic_sequence (a : ℕ+ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ+, a (n + 1) = a n + d

/-- Definition of a geometric sequence -/
def is_geometric_sequence (a : ℕ+ → ℝ) : Prop :=
  ∃ q : ℝ, q ≠ 0 ∧ ∀ n : ℕ+, a (n + 1) = a n * q

/-- Definition of the sum of the first n terms -/
def S (a : ℕ+ → ℝ) (n : ℕ+) : ℝ :=
  (Finset.range n).sum (λ i => a ⟨i + 1, Nat.succ_pos i⟩)

/-- Main theorem -/
theorem sequence_properties (a : ℕ+ → ℝ) :
  (∀ n : ℕ+, (is_arithmetic_sequence a ∧ is_geometric_sequence a) → a n = a (n + 1)) ∧
  (∃ α β : ℝ, ∀ n : ℕ+, S a n = α * n^2 + β * n) → is_arithmetic_sequence a ∧
  (∀ n : ℕ+, S a n = 1 - (-1)^(n : ℕ)) → is_geometric_sequence a :=
sorry

end NUMINAMATH_CALUDE_sequence_properties_l2246_224654


namespace NUMINAMATH_CALUDE_cost_per_box_l2246_224638

/-- The cost per box for packaging a fine arts collection -/
theorem cost_per_box (box_volume : ℝ) (total_volume : ℝ) (total_cost : ℝ) : 
  box_volume = 20 * 20 * 15 →
  total_volume = 3060000 →
  total_cost = 663 →
  total_cost / (total_volume / box_volume) = 1.30 := by
  sorry

#eval (663 : ℚ) / ((3060000 : ℚ) / (20 * 20 * 15 : ℚ))

end NUMINAMATH_CALUDE_cost_per_box_l2246_224638


namespace NUMINAMATH_CALUDE_proper_subset_implies_a_geq_two_l2246_224678

def A : Set ℝ := {x | 1 < x ∧ x < 2}
def B (a : ℝ) : Set ℝ := {x | x - a < 0}

theorem proper_subset_implies_a_geq_two (a : ℝ) :
  A ⊂ B a → a ≥ 2 := by sorry

end NUMINAMATH_CALUDE_proper_subset_implies_a_geq_two_l2246_224678


namespace NUMINAMATH_CALUDE_stratified_sampling_sizes_l2246_224601

/-- Represents the income groups in the community -/
inductive IncomeGroup
  | High
  | Middle
  | Low

/-- Calculates the sample size for a given income group -/
def sampleSize (totalPopulation : ℕ) (groupPopulation : ℕ) (totalSample : ℕ) : ℕ :=
  (groupPopulation * totalSample) / totalPopulation

/-- Theorem stating the correct sample sizes for each income group -/
theorem stratified_sampling_sizes :
  let totalPopulation := 600
  let highIncome := 230
  let middleIncome := 290
  let lowIncome := 80
  let totalSample := 60
  (sampleSize totalPopulation highIncome totalSample = 23) ∧
  (sampleSize totalPopulation middleIncome totalSample = 29) ∧
  (sampleSize totalPopulation lowIncome totalSample = 8) :=
by
  sorry

#check stratified_sampling_sizes

end NUMINAMATH_CALUDE_stratified_sampling_sizes_l2246_224601


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_algebraic_expression_value_l2246_224679

-- Part 1
theorem quadratic_equation_roots (x : ℝ) :
  x^2 - 4*x - 3 = 0 ↔ x = 2 + Real.sqrt 7 ∨ x = 2 - Real.sqrt 7 := by
  sorry

-- Part 2
theorem algebraic_expression_value (a : ℝ) :
  a^2 = 3*a + 10 → (a + 4) * (a - 4) - 3 * (a - 1) = -3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_algebraic_expression_value_l2246_224679


namespace NUMINAMATH_CALUDE_consecutive_numbers_sum_l2246_224625

theorem consecutive_numbers_sum (n : ℕ) : 
  (n + (n + 1) + (n + 2) + (n + 3) + (n + 4) = 60) → n = 10 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_numbers_sum_l2246_224625


namespace NUMINAMATH_CALUDE_stream_speed_l2246_224604

/-- Given a canoe's upstream and downstream speeds, prove the speed of the stream -/
theorem stream_speed (upstream_speed downstream_speed : ℝ) 
  (h_upstream : upstream_speed = 3)
  (h_downstream : downstream_speed = 12) :
  ∃ (canoe_speed stream_speed : ℝ),
    canoe_speed - stream_speed = upstream_speed ∧
    canoe_speed + stream_speed = downstream_speed ∧
    stream_speed = 4.5 := by
  sorry

end NUMINAMATH_CALUDE_stream_speed_l2246_224604


namespace NUMINAMATH_CALUDE_large_cube_volume_l2246_224603

theorem large_cube_volume (small_cube_surface_area : ℝ) (num_small_cubes : ℕ) :
  small_cube_surface_area = 96 →
  num_small_cubes = 8 →
  let small_cube_side := Real.sqrt (small_cube_surface_area / 6)
  let large_cube_side := small_cube_side * 2
  large_cube_side ^ 3 = 512 :=
by
  sorry

end NUMINAMATH_CALUDE_large_cube_volume_l2246_224603


namespace NUMINAMATH_CALUDE_wire_cut_square_octagon_ratio_l2246_224683

/-- The ratio of lengths when a wire is cut to form a square and an octagon with equal areas -/
theorem wire_cut_square_octagon_ratio (a b : ℝ) (h : a > 0) (k : b > 0) : 
  (a^2 / 16 = b^2 * (1 + Real.sqrt 2) / 32) → 
  (a / b = Real.sqrt ((1 + Real.sqrt 2) / 2)) := by
  sorry


end NUMINAMATH_CALUDE_wire_cut_square_octagon_ratio_l2246_224683


namespace NUMINAMATH_CALUDE_tan_identity_l2246_224689

theorem tan_identity : 
  (1 + Real.tan (28 * π / 180)) * (1 + Real.tan (17 * π / 180)) = 2 := by sorry

end NUMINAMATH_CALUDE_tan_identity_l2246_224689


namespace NUMINAMATH_CALUDE_x_range_when_ln_x_negative_l2246_224686

theorem x_range_when_ln_x_negative (x : ℝ) (h : Real.log x < 0) : 0 < x ∧ x < 1 := by
  sorry

end NUMINAMATH_CALUDE_x_range_when_ln_x_negative_l2246_224686


namespace NUMINAMATH_CALUDE_total_seats_is_28_l2246_224692

/-- The number of students per bus -/
def students_per_bus : ℝ := 14.0

/-- The number of buses -/
def number_of_buses : ℝ := 2.0

/-- The total number of seats taken up by students -/
def total_seats : ℝ := students_per_bus * number_of_buses

/-- Theorem stating that the total number of seats taken up by students is 28 -/
theorem total_seats_is_28 : total_seats = 28 := by
  sorry

end NUMINAMATH_CALUDE_total_seats_is_28_l2246_224692


namespace NUMINAMATH_CALUDE_rain_probability_l2246_224629

theorem rain_probability (weihai_rain : Real) (zibo_rain : Real) (both_rain : Real) :
  weihai_rain = 0.2 →
  zibo_rain = 0.15 →
  both_rain = 0.06 →
  both_rain / weihai_rain = 0.3 :=
by sorry

end NUMINAMATH_CALUDE_rain_probability_l2246_224629


namespace NUMINAMATH_CALUDE_range_of_a_l2246_224623

def p (a : ℝ) : Prop := ∀ x ∈ Set.Icc 1 2, x^2 ≥ a

def q (a : ℝ) : Prop := ∃ x₀ : ℝ, x₀^2 + 2*a*x₀ + 2 - a = 0

theorem range_of_a (a : ℝ) (h : p a ∧ q a) : a ≤ -2 ∨ a = 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2246_224623


namespace NUMINAMATH_CALUDE_point_in_fourth_quadrant_l2246_224645

theorem point_in_fourth_quadrant (A B C : Real) (h_acute : A > 0 ∧ B > 0 ∧ C > 0 ∧ A < π/2 ∧ B < π/2 ∧ C < π/2) (h_triangle : A + B + C = π) :
  let P : Real × Real := (Real.sin A - Real.cos B, Real.cos A - Real.sin C)
  P.1 > 0 ∧ P.2 < 0 :=
by sorry

end NUMINAMATH_CALUDE_point_in_fourth_quadrant_l2246_224645


namespace NUMINAMATH_CALUDE_definite_integral_2x_l2246_224660

theorem definite_integral_2x : ∫ x in (1:ℝ)..2, 2*x = 3 := by sorry

end NUMINAMATH_CALUDE_definite_integral_2x_l2246_224660


namespace NUMINAMATH_CALUDE_remainder_theorem_l2246_224697

/-- The polynomial f(x) = x^4 - 6x^3 + 12x^2 + 20x - 8 -/
def f (x : ℝ) : ℝ := x^4 - 6*x^3 + 12*x^2 + 20*x - 8

/-- The theorem stating that the remainder when f(x) is divided by (x-4) is 136 -/
theorem remainder_theorem : 
  ∃ q : ℝ → ℝ, ∀ x : ℝ, f x = (x - 4) * q x + 136 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l2246_224697


namespace NUMINAMATH_CALUDE_paint_room_time_l2246_224682

/-- Andy's painting rate in rooms per hour -/
def andy_rate : ℚ := 1 / 4

/-- Bob's painting rate in rooms per hour -/
def bob_rate : ℚ := 1 / 6

/-- The combined painting rate of Andy and Bob in rooms per hour -/
def combined_rate : ℚ := andy_rate + bob_rate

/-- The time taken to paint the room, including the lunch break -/
def t : ℚ := 22 / 5

theorem paint_room_time :
  (combined_rate * (t - 2) = 1) ∧ (combined_rate = 5 / 12) := by
  sorry

end NUMINAMATH_CALUDE_paint_room_time_l2246_224682


namespace NUMINAMATH_CALUDE_h_equals_three_l2246_224605

-- Define the quadratic coefficients
variable (a b c : ℝ)

-- Define the condition that ax^2 + bx + c = 3(x - 3)^2 + 9
def quadratic_condition (a b c : ℝ) : Prop :=
  ∀ x, a * x^2 + b * x + c = 3 * (x - 3)^2 + 9

-- Define the transformed quadratic
def transformed_quadratic (a b c : ℝ) (x : ℝ) : ℝ :=
  5 * a * x^2 + 5 * b * x + 5 * c

-- Theorem stating that h = 3 in the transformed quadratic
theorem h_equals_three (a b c : ℝ) 
  (h : quadratic_condition a b c) :
  ∃ (m k : ℝ), ∀ x, transformed_quadratic a b c x = m * (x - 3)^2 + k :=
sorry

end NUMINAMATH_CALUDE_h_equals_three_l2246_224605


namespace NUMINAMATH_CALUDE_longest_non_decreasing_subsequence_12022_l2246_224613

/-- Represents a natural number as a list of its digits. -/
def digits_of (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec aux (m : ℕ) : List ℕ :=
    if m = 0 then [] else (m % 10) :: aux (m / 10)
  (aux n).reverse

/-- Computes the length of the longest non-decreasing subsequence in a list. -/
def longest_non_decreasing_subsequence_length (l : List ℕ) : ℕ :=
  let rec aux (prev : ℕ) (current : List ℕ) (acc : ℕ) : ℕ :=
    match current with
    | [] => acc
    | x :: xs => if x ≥ prev then aux x xs (acc + 1) else aux prev xs acc
  aux 0 l 0

/-- The theorem stating that the longest non-decreasing subsequence of digits in 12022 has length 3. -/
theorem longest_non_decreasing_subsequence_12022 :
  longest_non_decreasing_subsequence_length (digits_of 12022) = 3 := by
  sorry

end NUMINAMATH_CALUDE_longest_non_decreasing_subsequence_12022_l2246_224613


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l2246_224668

-- Problem 1
theorem problem_1 : |Real.sqrt 3 - 2| + (3 - Real.pi)^0 - Real.sqrt 12 + 6 * Real.cos (30 * π / 180) = 3 := by sorry

-- Problem 2
theorem problem_2 : (1 / ((-5)^2 - 3*(-5))) / (2 / ((-5)^2 - 9)) = 1/5 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l2246_224668


namespace NUMINAMATH_CALUDE_buratino_apples_theorem_l2246_224642

theorem buratino_apples_theorem (a₁ a₂ a₃ a₄ a₅ a₆ : ℝ) 
  (h_distinct : a₁ > a₂ ∧ a₂ > a₃ ∧ a₃ > a₄ ∧ a₄ > a₅ ∧ a₅ > a₆) :
  ∃ x₁ x₂ x₃ x₄ : ℝ, 
    x₁ + x₂ = a₁ ∧ 
    x₁ + x₃ = a₂ ∧ 
    x₂ + x₃ = a₃ ∧ 
    x₃ + x₄ = a₄ ∧ 
    x₁ + x₄ ≥ a₅ ∧ 
    x₂ + x₄ ≥ a₆ ∧
    ∀ y₁ y₂ y₃ y₄ : ℝ, 
      (y₁ + y₂ = a₁ → y₁ + y₃ = a₂ → y₂ + y₃ = a₃ → y₃ + y₄ = a₄ → 
       y₁ + y₄ = a₅ → y₂ + y₄ = a₆) → False :=
by sorry

end NUMINAMATH_CALUDE_buratino_apples_theorem_l2246_224642


namespace NUMINAMATH_CALUDE_infinitely_many_solutions_l2246_224669

theorem infinitely_many_solutions (c : ℝ) : 
  (∀ x : ℝ, 3 * (5 + c * x) = 18 * x + 15) ↔ c = 6 :=
by sorry

end NUMINAMATH_CALUDE_infinitely_many_solutions_l2246_224669


namespace NUMINAMATH_CALUDE_happy_properties_l2246_224667

/-- A positive integer is happy if it can be expressed as the sum of two squares. -/
def IsHappy (n : ℕ+) : Prop :=
  ∃ a b : ℤ, n.val = a^2 + b^2

theorem happy_properties (t : ℕ+) (ht : IsHappy t) :
  (IsHappy (2 * t)) ∧ (¬IsHappy (3 * t)) := by
  sorry

end NUMINAMATH_CALUDE_happy_properties_l2246_224667


namespace NUMINAMATH_CALUDE_prism_pyramid_volume_ratio_l2246_224632

/-- Given a triangular prism with height m, we extend a side edge by x to form a pyramid.
    The volume ratio k of the remaining part of the prism (outside the pyramid) to the original prism
    must be less than or equal to 3/4. -/
theorem prism_pyramid_volume_ratio (m : ℝ) (x : ℝ) (k : ℝ) 
  (h1 : m > 0) (h2 : x > 0) (h3 : k > 0) : k ≤ 3/4 := by
  sorry

end NUMINAMATH_CALUDE_prism_pyramid_volume_ratio_l2246_224632
