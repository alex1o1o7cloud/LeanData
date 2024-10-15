import Mathlib

namespace NUMINAMATH_CALUDE_race_heartbeats_l1494_149440

/-- Calculates the total number of heartbeats during a race -/
def total_heartbeats (heart_rate : ℕ) (pace : ℕ) (distance : ℕ) : ℕ :=
  heart_rate * pace * distance

/-- Proves that the total number of heartbeats during the race is 27000 -/
theorem race_heartbeats :
  let heart_rate : ℕ := 180  -- heartbeats per minute
  let pace : ℕ := 3          -- minutes per kilometer
  let distance : ℕ := 50     -- kilometers
  total_heartbeats heart_rate pace distance = 27000 := by
  sorry


end NUMINAMATH_CALUDE_race_heartbeats_l1494_149440


namespace NUMINAMATH_CALUDE_unique_prime_with_next_square_is_three_l1494_149416

theorem unique_prime_with_next_square_is_three :
  ∀ p : ℕ, Prime p → (∃ n : ℕ, p + 1 = n^2) → p = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_unique_prime_with_next_square_is_three_l1494_149416


namespace NUMINAMATH_CALUDE_lcm_of_ratio_and_hcf_l1494_149468

theorem lcm_of_ratio_and_hcf (a b : ℕ+) : 
  (a : ℚ) / b = 4 / 5 → Nat.gcd a b = 4 → Nat.lcm a b = 80 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_ratio_and_hcf_l1494_149468


namespace NUMINAMATH_CALUDE_video_game_lives_l1494_149454

theorem video_game_lives (initial_players : ℕ) (players_quit : ℕ) (total_lives : ℕ) :
  initial_players = 10 →
  players_quit = 7 →
  total_lives = 24 →
  (total_lives / (initial_players - players_quit) : ℚ) = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_video_game_lives_l1494_149454


namespace NUMINAMATH_CALUDE_chicken_theorem_l1494_149462

/-- The number of chickens Colten has -/
def colten_chickens : ℕ := 37

/-- The number of chickens Skylar has -/
def skylar_chickens : ℕ := 3 * colten_chickens - 4

/-- The number of chickens Quentin has -/
def quentin_chickens : ℕ := 2 * skylar_chickens + 25

theorem chicken_theorem : 
  colten_chickens + skylar_chickens + quentin_chickens = 383 :=
by sorry

end NUMINAMATH_CALUDE_chicken_theorem_l1494_149462


namespace NUMINAMATH_CALUDE_coles_return_speed_coles_return_speed_is_120_l1494_149425

/-- Calculates the average speed of the return trip given the conditions of Cole's journey --/
theorem coles_return_speed (speed_to_work : ℝ) (total_time : ℝ) (time_to_work : ℝ) : ℝ :=
  let distance_to_work := speed_to_work * time_to_work
  let time_to_return := total_time - time_to_work
  distance_to_work / time_to_return

/-- Proves that Cole's average speed driving back home is 120 km/h --/
theorem coles_return_speed_is_120 :
  coles_return_speed 80 2 (72 / 60) = 120 := by
  sorry

end NUMINAMATH_CALUDE_coles_return_speed_coles_return_speed_is_120_l1494_149425


namespace NUMINAMATH_CALUDE_fraction_addition_simplification_l1494_149484

theorem fraction_addition_simplification :
  (1 : ℚ) / 462 + 23 / 42 = 127 / 231 := by sorry

end NUMINAMATH_CALUDE_fraction_addition_simplification_l1494_149484


namespace NUMINAMATH_CALUDE_factorization_proof_l1494_149452

theorem factorization_proof :
  ∀ x : ℝ,
  (x^2 - x - 6 = (x + 2) * (x - 3)) ∧
  ¬(x^2 - 1 = x * (x - 1/x)) ∧
  ¬(7 * x^2 * y^5 = x * y * 7 * x * y^4) ∧
  ¬(x^2 + 4*x + 4 = x * (x + 4) + 4) :=
by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l1494_149452


namespace NUMINAMATH_CALUDE_unique_relative_minimum_l1494_149404

/-- The function f(x) = x^4 - x^3 - x^2 + ax + 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^4 - x^3 - x^2 + a*x + 1

/-- The derivative of f with respect to x -/
def f_derivative (a : ℝ) (x : ℝ) : ℝ := 4*x^3 - 3*x^2 - 2*x + a

theorem unique_relative_minimum (a : ℝ) :
  (∃ (x : ℝ), f a x = x ∧ 
    ∀ (y : ℝ), y ≠ x → f a y > f a x) ↔ a = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_relative_minimum_l1494_149404


namespace NUMINAMATH_CALUDE_inverse_proportion_relationship_l1494_149491

theorem inverse_proportion_relationship (x₁ x₂ y₁ y₂ : ℝ) :
  x₁ < x₂ → x₂ < 0 → y₁ = 2 / x₁ → y₂ = 2 / x₂ → y₂ < y₁ ∧ y₁ < 0 := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_relationship_l1494_149491


namespace NUMINAMATH_CALUDE_range_of_m_l1494_149465

/-- The range of m satisfying the given conditions -/
def M : Set ℝ := { m | ∀ x ∈ Set.Icc 0 1, 2 * m - 1 < x * (m^2 - 1) }

/-- Theorem stating that M is equal to the open interval (-∞, 0) -/
theorem range_of_m : M = Set.Ioi 0 := by sorry

end NUMINAMATH_CALUDE_range_of_m_l1494_149465


namespace NUMINAMATH_CALUDE_circle_center_sum_l1494_149492

/-- Given a circle with equation x^2 + y^2 = 6x + 18y - 63, 
    prove that the sum of the coordinates of its center is 12. -/
theorem circle_center_sum (h k : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 = 6*x + 18*y - 63 ↔ (x - h)^2 + (y - k)^2 = (h^2 + k^2 - 6*h - 18*k + 63)) →
  h + k = 12 := by
sorry

end NUMINAMATH_CALUDE_circle_center_sum_l1494_149492


namespace NUMINAMATH_CALUDE_f_inequality_range_l1494_149410

noncomputable def f (x : ℝ) : ℝ := Real.exp (abs x) - 1 / (x^2 + 2)

theorem f_inequality_range (x : ℝ) : 
  f x > f (2 * x - 1) ↔ 1/3 < x ∧ x < 1 :=
by sorry

end NUMINAMATH_CALUDE_f_inequality_range_l1494_149410


namespace NUMINAMATH_CALUDE_empty_solution_set_implies_a_range_l1494_149498

theorem empty_solution_set_implies_a_range 
  (h : ∀ x : ℝ, ¬(|x + 3| + |x - 1| < a^2 - 3*a)) : 
  a ∈ Set.Icc (-1 : ℝ) 4 := by
  sorry

end NUMINAMATH_CALUDE_empty_solution_set_implies_a_range_l1494_149498


namespace NUMINAMATH_CALUDE_fermat_divisibility_l1494_149441

/-- Fermat number -/
def F (n : ℕ) : ℕ := 2^(2^n) + 1

/-- Theorem: For all natural numbers n, F_n divides 2^F_n - 2 -/
theorem fermat_divisibility (n : ℕ) : (F n) ∣ (2^(F n) - 2) := by
  sorry

end NUMINAMATH_CALUDE_fermat_divisibility_l1494_149441


namespace NUMINAMATH_CALUDE_complement_equal_l1494_149433

/-- The complement of an angle is the angle that, when added to the original angle, results in a right angle (90 degrees). -/
def complement (α : ℝ) : ℝ := 90 - α

/-- For any angle, its complement is equal to itself. -/
theorem complement_equal (α : ℝ) : complement α = complement α := by sorry

end NUMINAMATH_CALUDE_complement_equal_l1494_149433


namespace NUMINAMATH_CALUDE_cid_car_wash_count_l1494_149494

theorem cid_car_wash_count :
  let oil_change_price : ℕ := 20
  let repair_price : ℕ := 30
  let car_wash_price : ℕ := 5
  let oil_change_count : ℕ := 5
  let repair_count : ℕ := 10
  let total_earnings : ℕ := 475
  let car_wash_count : ℕ := (total_earnings - (oil_change_price * oil_change_count + repair_price * repair_count)) / car_wash_price
  car_wash_count = 15 :=
by sorry

end NUMINAMATH_CALUDE_cid_car_wash_count_l1494_149494


namespace NUMINAMATH_CALUDE_tangent_line_to_logarithmic_curve_l1494_149402

theorem tangent_line_to_logarithmic_curve : ∃ (n : ℕ+) (a : ℝ), 
  (n : ℝ) < a ∧ a < (n : ℝ) + 1 ∧
  (∃ (x : ℝ), x > 0 ∧ x + 1 = a * Real.log x ∧ 1 = a / x) ∧
  n = 3 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_to_logarithmic_curve_l1494_149402


namespace NUMINAMATH_CALUDE_total_grains_in_gray_parts_l1494_149475

/-- Represents a circle with grains -/
structure GrainCircle where
  total : ℕ
  intersection : ℕ

/-- Calculates the number of grains in the non-intersecting part of a circle -/
def nonIntersectingGrains (circle : GrainCircle) : ℕ :=
  circle.total - circle.intersection

/-- The main theorem -/
theorem total_grains_in_gray_parts 
  (circle1 circle2 : GrainCircle)
  (h1 : circle1.total = 87)
  (h2 : circle2.total = 110)
  (h3 : circle1.intersection = 68)
  (h4 : circle2.intersection = 68) :
  nonIntersectingGrains circle1 + nonIntersectingGrains circle2 = 61 := by
  sorry

#eval nonIntersectingGrains { total := 87, intersection := 68 } +
      nonIntersectingGrains { total := 110, intersection := 68 }

end NUMINAMATH_CALUDE_total_grains_in_gray_parts_l1494_149475


namespace NUMINAMATH_CALUDE_hearts_on_card_l1494_149420

/-- The number of hearts on each card in a hypothetical deck -/
def hearts_per_card : ℕ := sorry

/-- The number of cows in Devonshire -/
def num_cows : ℕ := 2 * hearts_per_card

/-- The cost of each cow in dollars -/
def cost_per_cow : ℕ := 200

/-- The total cost of all cows in dollars -/
def total_cost : ℕ := 83200

theorem hearts_on_card :
  hearts_per_card = 208 :=
sorry

end NUMINAMATH_CALUDE_hearts_on_card_l1494_149420


namespace NUMINAMATH_CALUDE_number_of_girls_in_college_l1494_149401

theorem number_of_girls_in_college (total_students : ℕ) (boys_to_girls_ratio : ℚ) 
  (h1 : total_students = 416) 
  (h2 : boys_to_girls_ratio = 8 / 5) : 
  ∃ (girls : ℕ), girls = 160 ∧ 
    (girls : ℚ) * (1 + boys_to_girls_ratio) = total_students := by
  sorry

end NUMINAMATH_CALUDE_number_of_girls_in_college_l1494_149401


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l1494_149483

theorem absolute_value_equation_solution :
  ∃! y : ℝ, (|y - 4| + 3 * y = 14) :=
by
  -- The unique solution is y = 4.5
  use 4.5
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l1494_149483


namespace NUMINAMATH_CALUDE_min_value_of_sum_squares_l1494_149479

theorem min_value_of_sum_squares (x y z : ℝ) (h : x + y + z = 2) :
  x^2 + 2*y^2 + z^2 ≥ 4/3 ∧ 
  ∃ (a b c : ℝ), a + b + c = 2 ∧ a^2 + 2*b^2 + c^2 = 4/3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_sum_squares_l1494_149479


namespace NUMINAMATH_CALUDE_buffer_solution_calculation_l1494_149455

theorem buffer_solution_calculation (initial_volume_A initial_volume_B total_volume_needed : ℚ) :
  initial_volume_A = 0.05 →
  initial_volume_B = 0.03 →
  initial_volume_A + initial_volume_B = 0.08 →
  total_volume_needed = 0.64 →
  (total_volume_needed * (initial_volume_B / (initial_volume_A + initial_volume_B))) = 0.24 := by
sorry

end NUMINAMATH_CALUDE_buffer_solution_calculation_l1494_149455


namespace NUMINAMATH_CALUDE_fourth_game_score_l1494_149413

def game_scores (game1 game2 game3 game4 total : ℕ) : Prop :=
  game1 = 10 ∧ game2 = 14 ∧ game3 = 6 ∧ game1 + game2 + game3 + game4 = total

theorem fourth_game_score (game1 game2 game3 game4 total : ℕ) :
  game_scores game1 game2 game3 game4 total → total = 40 → game4 = 10 := by
  sorry

end NUMINAMATH_CALUDE_fourth_game_score_l1494_149413


namespace NUMINAMATH_CALUDE_like_terms_imply_zero_power_l1494_149497

theorem like_terms_imply_zero_power (n : ℕ) : 
  (∃ x y, -x^(2*n-1) * y = 3 * x^8 * y) → (2*n - 9)^2013 = 0 := by
  sorry

end NUMINAMATH_CALUDE_like_terms_imply_zero_power_l1494_149497


namespace NUMINAMATH_CALUDE_intersection_probability_l1494_149442

-- Define the probability measure q
variable (q : Set ℝ → ℝ)

-- Define events g and h
variable (g h : Set ℝ)

-- Define the conditions
variable (hg : q g = 0.30)
variable (hh : q h = 0.9)
variable (hgh : q (g ∩ h) / q h = 1 / 3)
variable (hhg : q (g ∩ h) / q g = 1 / 3)

-- The theorem to prove
theorem intersection_probability : q (g ∩ h) = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_intersection_probability_l1494_149442


namespace NUMINAMATH_CALUDE_sum_of_four_with_common_divisors_l1494_149426

theorem sum_of_four_with_common_divisors (n : ℕ) (h : n > 31) :
  ∃ (a b c d : ℕ), 
    n = a + b + c + d ∧
    a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧
    (∃ (k₁ : ℕ), k₁ > 1 ∧ k₁ ∣ a ∧ k₁ ∣ b) ∧
    (∃ (k₂ : ℕ), k₂ > 1 ∧ k₂ ∣ a ∧ k₂ ∣ c) ∧
    (∃ (k₃ : ℕ), k₃ > 1 ∧ k₃ ∣ a ∧ k₃ ∣ d) ∧
    (∃ (k₄ : ℕ), k₄ > 1 ∧ k₄ ∣ b ∧ k₄ ∣ c) ∧
    (∃ (k₅ : ℕ), k₅ > 1 ∧ k₅ ∣ b ∧ k₅ ∣ d) ∧
    (∃ (k₆ : ℕ), k₆ > 1 ∧ k₆ ∣ c ∧ k₆ ∣ d) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_four_with_common_divisors_l1494_149426


namespace NUMINAMATH_CALUDE_counterexample_exists_l1494_149445

theorem counterexample_exists : ∃ n : ℕ, ¬(Nat.Prime n) ∧ Nat.Prime (n + 3) := by
  sorry

end NUMINAMATH_CALUDE_counterexample_exists_l1494_149445


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l1494_149446

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- The main theorem -/
theorem geometric_sequence_ratio
  (a : ℕ → ℝ)
  (h_geo : geometric_sequence a)
  (h_prod : a 7 * a 11 = 6)
  (h_sum : a 4 + a 14 = 5) :
  a 20 / a 10 = 2/3 ∨ a 20 / a 10 = 3/2 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l1494_149446


namespace NUMINAMATH_CALUDE_solution_set_equivalence_k_range_l1494_149467

noncomputable section

-- Define the function f
def f (k x : ℝ) : ℝ := (k * x) / (x^2 + 3 * k)

-- Define the conditions
variable (k : ℝ)
variable (h_k_pos : k > 0)

-- Part 1
theorem solution_set_equivalence :
  (∃ m : ℝ, ∀ x : ℝ, f k x > m ↔ x < -3 ∨ x > -2) →
  ∃ m : ℝ, ∀ x : ℝ, 5 * m * x^2 + (k / 2) * x + 3 > 0 ↔ -1 < x ∧ x < 3/2 :=
sorry

-- Part 2
theorem k_range :
  (∃ x : ℝ, x > 3 ∧ f k x > 1) →
  k > 12 :=
sorry

end

end NUMINAMATH_CALUDE_solution_set_equivalence_k_range_l1494_149467


namespace NUMINAMATH_CALUDE_honey_percentage_l1494_149435

theorem honey_percentage (initial_honey : ℝ) (final_honey : ℝ) (repetitions : ℕ) 
  (h_initial : initial_honey = 1250)
  (h_final : final_honey = 512)
  (h_repetitions : repetitions = 4) :
  ∃ (percentage : ℝ), 
    percentage = 0.2 ∧ 
    final_honey = initial_honey * (1 - percentage) ^ repetitions :=
by sorry

end NUMINAMATH_CALUDE_honey_percentage_l1494_149435


namespace NUMINAMATH_CALUDE_angle_equality_l1494_149496

theorem angle_equality (α β : Real) : 
  0 < α ∧ α < π/2 →  -- α is acute
  0 < β ∧ β < π/2 →  -- β is acute
  Real.cos α + Real.cos (2*β) - Real.cos (α + β) = 3/2 →
  α = π/3 ∧ β = π/3 := by
  sorry

end NUMINAMATH_CALUDE_angle_equality_l1494_149496


namespace NUMINAMATH_CALUDE_lcm_of_ratio_two_three_l1494_149489

/-- Given two numbers a and b in the ratio 2:3, where a = 40 and b = 60, prove that their LCM is 60. -/
theorem lcm_of_ratio_two_three (a b : ℕ) (h1 : a = 40) (h2 : b = 60) (h3 : 3 * a = 2 * b) :
  Nat.lcm a b = 60 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_ratio_two_three_l1494_149489


namespace NUMINAMATH_CALUDE_problem_solution_roots_product_l1494_149493

noncomputable section

-- Define the functions f and g
def f (x : ℝ) : ℝ := Real.log x
def g (m : ℝ) (x : ℝ) : ℝ := x + m

-- Define the function F
def F (m : ℝ) (x : ℝ) : ℝ := f x - g m x

theorem problem_solution (m : ℝ) :
  (∀ x > 0, f x ≤ g m x) ↔ m ≥ -1 :=
sorry

theorem roots_product (m : ℝ) (x₁ x₂ : ℝ) :
  x₁ < x₂ →
  F m x₁ = 0 →
  F m x₂ = 0 →
  x₁ * x₂ < 1 :=
sorry

end NUMINAMATH_CALUDE_problem_solution_roots_product_l1494_149493


namespace NUMINAMATH_CALUDE_normal_distribution_probability_l1494_149459

/-- A random variable following a normal distribution -/
structure NormalRandomVariable where
  μ : ℝ
  σ : ℝ
  hσ_pos : σ > 0

/-- The probability that a normal random variable is less than a given value -/
def prob_less_than (ξ : NormalRandomVariable) (x : ℝ) : ℝ := sorry

/-- The probability that a normal random variable is between two given values -/
def prob_between (ξ : NormalRandomVariable) (a b : ℝ) : ℝ := sorry

/-- Theorem: For a normal random variable ξ with mean 40, 
    if P(ξ < 30) = 0.2, then P(30 < ξ < 50) = 0.6 -/
theorem normal_distribution_probability 
  (ξ : NormalRandomVariable) 
  (h_mean : ξ.μ = 40) 
  (h_prob : prob_less_than ξ 30 = 0.2) : 
  prob_between ξ 30 50 = 0.6 := by sorry

end NUMINAMATH_CALUDE_normal_distribution_probability_l1494_149459


namespace NUMINAMATH_CALUDE_xiaoming_age_l1494_149400

def is_valid_age (birth_year : ℕ) (current_year : ℕ) : Prop :=
  current_year - birth_year = (birth_year / 1000) + ((birth_year / 100) % 10) + ((birth_year / 10) % 10) + (birth_year % 10)

theorem xiaoming_age :
  ∃ (age : ℕ), (age = 22 ∨ age = 4) ∧
  ∃ (birth_year : ℕ),
    birth_year ≥ 1900 ∧
    birth_year < 2015 ∧
    is_valid_age birth_year 2015 ∧
    age = 2015 - birth_year :=
by sorry

end NUMINAMATH_CALUDE_xiaoming_age_l1494_149400


namespace NUMINAMATH_CALUDE_scientific_notation_of_216000_l1494_149447

theorem scientific_notation_of_216000 :
  (216000 : ℝ) = 2.16 * (10 ^ 5) := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_216000_l1494_149447


namespace NUMINAMATH_CALUDE_inequality_solution_range_l1494_149431

theorem inequality_solution_range (k : ℝ) : 
  (1 : ℝ) ∈ {x : ℝ | k^2 * x^2 - 6*k*x + 8 ≥ 0} ↔ k ≥ 4 ∨ k ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l1494_149431


namespace NUMINAMATH_CALUDE_root_difference_range_l1494_149421

/-- Given a quadratic function f(x) = ax² + (b-a)x + (c-b) where a > b > c and a + b + c = 0,
    the absolute difference between its roots |x₁ - x₂| lies in the open interval (3/2, 2√3). -/
theorem root_difference_range (a b c : ℝ) (ha : a > b) (hb : b > c) (hsum : a + b + c = 0) :
  let f : ℝ → ℝ := λ x => a * x^2 + (b - a) * x + (c - b)
  let x₁ := (-(b - a) + Real.sqrt ((b - a)^2 - 4 * a * (c - b))) / (2 * a)
  let x₂ := (-(b - a) - Real.sqrt ((b - a)^2 - 4 * a * (c - b))) / (2 * a)
  3/2 < |x₁ - x₂| ∧ |x₁ - x₂| < 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_root_difference_range_l1494_149421


namespace NUMINAMATH_CALUDE_quadratic_discriminant_zero_implies_perfect_square_l1494_149419

/-- 
If the discriminant of a quadratic equation ax^2 + bx + c = 0 is zero,
then the left-hand side is a perfect square.
-/
theorem quadratic_discriminant_zero_implies_perfect_square
  (a b c : ℝ) (h : b^2 - 4*a*c = 0) :
  ∃ k : ℝ, ∀ x : ℝ, a*x^2 + b*x + c = k*(2*a*x + b)^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_discriminant_zero_implies_perfect_square_l1494_149419


namespace NUMINAMATH_CALUDE_same_number_on_cards_l1494_149463

theorem same_number_on_cards (n : ℕ) (cards : Fin n → ℕ) : 
  (∀ i, cards i ∈ Finset.range n) →
  (∀ s : Finset (Fin n), (s.sum cards) % (n + 1) ≠ 0) →
  ∀ i j, cards i = cards j :=
by sorry

end NUMINAMATH_CALUDE_same_number_on_cards_l1494_149463


namespace NUMINAMATH_CALUDE_grandma_olga_grandchildren_l1494_149405

/-- Represents the number of grandchildren Grandma Olga has. -/
def total_grandchildren : ℕ :=
  let daughters := 5
  let sons := 4
  let children_per_daughter := 8 + 7
  let children_per_son := 6 + 3
  daughters * children_per_daughter + sons * children_per_son

/-- Proves that Grandma Olga has 111 grandchildren. -/
theorem grandma_olga_grandchildren : total_grandchildren = 111 := by
  sorry

end NUMINAMATH_CALUDE_grandma_olga_grandchildren_l1494_149405


namespace NUMINAMATH_CALUDE_airplane_cost_l1494_149480

def initial_amount : ℚ := 5.00
def change_received : ℚ := 0.72

theorem airplane_cost : initial_amount - change_received = 4.28 := by
  sorry

end NUMINAMATH_CALUDE_airplane_cost_l1494_149480


namespace NUMINAMATH_CALUDE_matrix_inverse_scalar_multiple_l1494_149406

/-- Given a 2x2 matrix A with elements [[1, 3], [4, d]] where A⁻¹ = k * A,
    prove that d = 6 and k = 1/6 -/
theorem matrix_inverse_scalar_multiple (d k : ℝ) : 
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![1, 3; 4, d]
  A⁻¹ = k • A → d = 6 ∧ k = 1/6 := by
  sorry

end NUMINAMATH_CALUDE_matrix_inverse_scalar_multiple_l1494_149406


namespace NUMINAMATH_CALUDE_fraction_sum_equal_decimal_l1494_149487

theorem fraction_sum_equal_decimal : 
  (2 / 20 : ℝ) + (8 / 200 : ℝ) + (3 / 300 : ℝ) + 2 * (5 / 40000 : ℝ) = 0.15025 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equal_decimal_l1494_149487


namespace NUMINAMATH_CALUDE_puzzle_completion_percentage_l1494_149423

theorem puzzle_completion_percentage (total_pieces : ℕ) 
  (day1_percentage day2_percentage : ℚ) (pieces_left : ℕ) : 
  total_pieces = 1000 →
  day1_percentage = 1/10 →
  day2_percentage = 1/5 →
  pieces_left = 504 →
  let pieces_after_day1 := total_pieces - (total_pieces * day1_percentage).num
  let pieces_after_day2 := pieces_after_day1 - (pieces_after_day1 * day2_percentage).num
  let pieces_completed_day3 := pieces_after_day2 - pieces_left
  (pieces_completed_day3 : ℚ) / pieces_after_day2 = 3/10 := by sorry

end NUMINAMATH_CALUDE_puzzle_completion_percentage_l1494_149423


namespace NUMINAMATH_CALUDE_constant_term_expansion_l1494_149427

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℕ := sorry

-- Define the function to calculate the constant term
def constantTerm (a b : ℕ) : ℕ :=
  binomial 8 3 * (5 ^ 5) * (2 ^ 3)

-- Theorem statement
theorem constant_term_expansion :
  constantTerm 5 2 = 1400000 := by sorry

end NUMINAMATH_CALUDE_constant_term_expansion_l1494_149427


namespace NUMINAMATH_CALUDE_correct_quotient_after_error_l1494_149411

theorem correct_quotient_after_error (dividend : ℕ) (incorrect_divisor correct_divisor incorrect_quotient : ℕ) :
  incorrect_divisor = 48 →
  correct_divisor = 36 →
  incorrect_quotient = 24 →
  dividend = incorrect_divisor * incorrect_quotient →
  dividend / correct_divisor = 32 :=
by sorry

end NUMINAMATH_CALUDE_correct_quotient_after_error_l1494_149411


namespace NUMINAMATH_CALUDE_systems_solutions_l1494_149474

-- Define the systems of equations
def system1 (x y : ℝ) : Prop :=
  y = 2 * x - 3 ∧ 3 * x + 2 * y = 8

def system2 (x y : ℝ) : Prop :=
  x + 2 * y = 3 ∧ 2 * x - 4 * y = -10

-- State the theorem
theorem systems_solutions :
  (∃ x y : ℝ, system1 x y ∧ x = 2 ∧ y = 1) ∧
  (∃ x y : ℝ, system2 x y ∧ x = -1 ∧ y = 2) :=
sorry

end NUMINAMATH_CALUDE_systems_solutions_l1494_149474


namespace NUMINAMATH_CALUDE_modular_inverse_17_mod_800_l1494_149466

theorem modular_inverse_17_mod_800 : ∃ x : ℕ, x < 800 ∧ (17 * x) % 800 = 1 :=
by
  use 753
  sorry

end NUMINAMATH_CALUDE_modular_inverse_17_mod_800_l1494_149466


namespace NUMINAMATH_CALUDE_shells_equation_initial_shells_value_l1494_149457

/-- The number of shells Lucy initially put in her bucket -/
def initial_shells : ℕ := sorry

/-- The number of additional shells Lucy found -/
def additional_shells : ℕ := 21

/-- The total number of shells Lucy has now -/
def total_shells : ℕ := 89

/-- Theorem stating that the initial number of shells plus the additional shells equals the total shells -/
theorem shells_equation : initial_shells + additional_shells = total_shells := by sorry

/-- Theorem proving that the initial number of shells is 68 -/
theorem initial_shells_value : initial_shells = 68 := by sorry

end NUMINAMATH_CALUDE_shells_equation_initial_shells_value_l1494_149457


namespace NUMINAMATH_CALUDE_eggs_per_friend_l1494_149417

/-- Proves that sharing 16 eggs equally among 8 friends results in 2 eggs per friend -/
theorem eggs_per_friend (total_eggs : ℕ) (num_friends : ℕ) (eggs_per_friend : ℕ) 
  (h1 : total_eggs = 16) 
  (h2 : num_friends = 8) 
  (h3 : eggs_per_friend * num_friends = total_eggs) : 
  eggs_per_friend = 2 := by
  sorry

end NUMINAMATH_CALUDE_eggs_per_friend_l1494_149417


namespace NUMINAMATH_CALUDE_triangle_tiling_exists_quadrilateral_tiling_exists_hexagon_tiling_exists_l1494_149424

/-- A polygon in the plane -/
structure Polygon :=
  (vertices : List (ℝ × ℝ))

/-- A tiling of the plane using a given polygon -/
def Tiling (p : Polygon) := 
  List (ℝ × ℝ) → Prop

/-- Predicate for a centrally symmetric hexagon -/
def IsCentrallySymmetricHexagon (p : Polygon) : Prop :=
  p.vertices.length = 6 ∧ 
  ∃ center : ℝ × ℝ, ∀ v ∈ p.vertices, 
    ∃ v' ∈ p.vertices, v' = (2 * center.1 - v.1, 2 * center.2 - v.2)

/-- Theorem stating that any triangle can tile the plane -/
theorem triangle_tiling_exists (t : Polygon) (h : t.vertices.length = 3) :
  ∃ tiling : Tiling t, True :=
sorry

/-- Theorem stating that any quadrilateral can tile the plane -/
theorem quadrilateral_tiling_exists (q : Polygon) (h : q.vertices.length = 4) :
  ∃ tiling : Tiling q, True :=
sorry

/-- Theorem stating that any centrally symmetric hexagon can tile the plane -/
theorem hexagon_tiling_exists (h : Polygon) (symmetric : IsCentrallySymmetricHexagon h) :
  ∃ tiling : Tiling h, True :=
sorry

end NUMINAMATH_CALUDE_triangle_tiling_exists_quadrilateral_tiling_exists_hexagon_tiling_exists_l1494_149424


namespace NUMINAMATH_CALUDE_cistern_width_is_six_l1494_149437

/-- Represents a rectangular cistern with water --/
structure Cistern where
  length : ℝ
  width : ℝ
  depth : ℝ

/-- Calculates the total wet surface area of the cistern --/
def wetSurfaceArea (c : Cistern) : ℝ :=
  c.length * c.width + 2 * c.length * c.depth + 2 * c.width * c.depth

/-- Theorem: Given the dimensions and wet surface area, the width of the cistern is 6 meters --/
theorem cistern_width_is_six (c : Cistern) 
    (h_length : c.length = 8)
    (h_depth : c.depth = 1.25)
    (h_area : wetSurfaceArea c = 83) : 
  c.width = 6 := by
  sorry

end NUMINAMATH_CALUDE_cistern_width_is_six_l1494_149437


namespace NUMINAMATH_CALUDE_sweets_distribution_l1494_149460

theorem sweets_distribution (num_children : ℕ) (sweets_per_child : ℕ) (remaining_fraction : ℚ) :
  num_children = 48 →
  sweets_per_child = 4 →
  remaining_fraction = 1/3 →
  (num_children * sweets_per_child) / (1 - remaining_fraction) = 288 := by
  sorry

end NUMINAMATH_CALUDE_sweets_distribution_l1494_149460


namespace NUMINAMATH_CALUDE_tammys_climbing_speed_l1494_149415

/-- Tammy's mountain climbing problem -/
theorem tammys_climbing_speed 
  (total_time : ℝ) 
  (total_distance : ℝ) 
  (speed_difference : ℝ) 
  (time_difference : ℝ) 
  (h1 : total_time = 14) 
  (h2 : total_distance = 52) 
  (h3 : speed_difference = 0.5) 
  (h4 : time_difference = 2) :
  ∃ (speed_day1 speed_day2 time_day1 time_day2 : ℝ),
    speed_day2 = speed_day1 + speed_difference ∧
    time_day2 = time_day1 - time_difference ∧
    time_day1 + time_day2 = total_time ∧
    speed_day1 * time_day1 + speed_day2 * time_day2 = total_distance ∧
    speed_day2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_tammys_climbing_speed_l1494_149415


namespace NUMINAMATH_CALUDE_ratio_first_term_to_common_difference_l1494_149443

/-- An arithmetic progression where the sum of the first twenty terms
    is five times the sum of the first ten terms -/
structure ArithmeticProgression where
  a : ℚ  -- First term
  d : ℚ  -- Common difference
  sum_condition : (20 * a + 190 * d) = 5 * (10 * a + 45 * d)

/-- The ratio of the first term to the common difference is -7/6 -/
theorem ratio_first_term_to_common_difference
  (ap : ArithmeticProgression) : ap.a / ap.d = -7 / 6 := by
  sorry

end NUMINAMATH_CALUDE_ratio_first_term_to_common_difference_l1494_149443


namespace NUMINAMATH_CALUDE_profit_equals_cost_of_three_toys_l1494_149478

/-- Proves that the number of toys whose cost price equals the profit is 3 -/
theorem profit_equals_cost_of_three_toys 
  (total_toys : ℕ) 
  (selling_price : ℕ) 
  (cost_per_toy : ℕ) 
  (h1 : total_toys = 18)
  (h2 : selling_price = 25200)
  (h3 : cost_per_toy = 1200) :
  (selling_price - total_toys * cost_per_toy) / cost_per_toy = 3 := by
  sorry

end NUMINAMATH_CALUDE_profit_equals_cost_of_three_toys_l1494_149478


namespace NUMINAMATH_CALUDE_units_digit_factorial_sum_plus_7_l1494_149472

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def units_digit (n : ℕ) : ℕ := n % 10

def factorial_sum (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

theorem units_digit_factorial_sum_plus_7 :
  units_digit (factorial_sum 25 + 7) = 0 := by sorry

end NUMINAMATH_CALUDE_units_digit_factorial_sum_plus_7_l1494_149472


namespace NUMINAMATH_CALUDE_not_p_or_not_q_true_l1494_149482

theorem not_p_or_not_q_true (p q : Prop) (h : ¬(p ∧ q)) : (¬p) ∨ (¬q) := by
  sorry

end NUMINAMATH_CALUDE_not_p_or_not_q_true_l1494_149482


namespace NUMINAMATH_CALUDE_triangle_problem_l1494_149408

noncomputable def f (x θ : Real) : Real :=
  2 * Real.sin x * (Real.cos (θ / 2))^2 + Real.cos x * Real.sin θ - Real.sin x

theorem triangle_problem (θ : Real) (h1 : 0 < θ) (h2 : θ < π) 
  (h3 : ∀ x, f x θ ≥ f π θ) :
  ∃ (A B C : Real),
    θ = π / 2 ∧
    0 < A ∧ A < π ∧
    0 < B ∧ B < π ∧
    0 < C ∧ C < π ∧
    A + B + C = π ∧
    Real.sin B / Real.sin A = Real.sqrt 2 ∧
    f A (π / 2) = Real.sqrt 3 / 2 ∧
    (C = 7 * π / 12 ∨ C = π / 12) := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l1494_149408


namespace NUMINAMATH_CALUDE_cubic_sum_theorem_l1494_149485

theorem cubic_sum_theorem (a b c : ℝ) 
  (sum_condition : a + b + c = 1)
  (product_sum_condition : a * b + a * c + b * c = -4)
  (product_condition : a * b * c = -4) : 
  a^3 + b^3 + c^3 = 1 := by sorry

end NUMINAMATH_CALUDE_cubic_sum_theorem_l1494_149485


namespace NUMINAMATH_CALUDE_chocolate_topping_proof_l1494_149434

/-- Proves that adding 220 ounces of pure chocolate to the initial mixture 
    results in a 75% chocolate topping -/
theorem chocolate_topping_proof 
  (initial_total : ℝ) 
  (initial_chocolate : ℝ) 
  (initial_other : ℝ) 
  (target_percentage : ℝ) 
  (h1 : initial_total = 220)
  (h2 : initial_chocolate = 110)
  (h3 : initial_other = 110)
  (h4 : initial_total = initial_chocolate + initial_other)
  (h5 : target_percentage = 0.75) : 
  let added_chocolate : ℝ := 220
  let final_chocolate : ℝ := initial_chocolate + added_chocolate
  let final_total : ℝ := initial_total + added_chocolate
  final_chocolate / final_total = target_percentage :=
by sorry

end NUMINAMATH_CALUDE_chocolate_topping_proof_l1494_149434


namespace NUMINAMATH_CALUDE_subset_count_with_nonempty_intersection_l1494_149481

theorem subset_count_with_nonempty_intersection :
  let A : Finset ℕ := Finset.range 10
  let B : Finset ℕ := {1, 2, 3, 4}
  (Finset.filter (fun C => (C ∩ B).Nonempty) (Finset.powerset A)).card = 960 := by
  sorry

end NUMINAMATH_CALUDE_subset_count_with_nonempty_intersection_l1494_149481


namespace NUMINAMATH_CALUDE_lost_to_initial_ratio_l1494_149439

/-- Represents the number of black socks Andy initially had -/
def initial_black_socks : ℕ := 6

/-- Represents the number of white socks Andy initially had -/
def initial_white_socks : ℕ := 4 * initial_black_socks

/-- Represents the number of white socks Andy has after losing some -/
def remaining_white_socks : ℕ := initial_black_socks + 6

/-- Represents the number of white socks Andy lost -/
def lost_white_socks : ℕ := initial_white_socks - remaining_white_socks

/-- Theorem stating that the ratio of lost white socks to initial white socks is 1/2 -/
theorem lost_to_initial_ratio :
  (lost_white_socks : ℚ) / initial_white_socks = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_lost_to_initial_ratio_l1494_149439


namespace NUMINAMATH_CALUDE_parallel_line_plane_not_imply_parallel_lines_l1494_149430

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel_line_plane : Line → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (contains : Plane → Line → Prop)

-- State the theorem
theorem parallel_line_plane_not_imply_parallel_lines 
  (l m : Line) (α : Plane) : 
  ¬(parallel_line_plane l α ∧ contains α m → parallel_lines l m) := by
  sorry

end NUMINAMATH_CALUDE_parallel_line_plane_not_imply_parallel_lines_l1494_149430


namespace NUMINAMATH_CALUDE_smallest_possible_a_l1494_149403

theorem smallest_possible_a (a b c : ℕ) : 
  a > 0 → b > 0 → c > 0 →
  (a + b + c) / 3 = 20 →
  a ≤ b →
  b ≤ c →
  c ≥ 25 →
  ∃ (a' b' c' : ℕ), a' > 0 ∧ b' > 0 ∧ c' > 0 ∧
    (a' + b' + c') / 3 = 20 ∧
    a' ≤ b' ∧
    b' ≤ c' ∧
    c' ≥ 25 ∧
    a' = 1 ∧
    ∀ (a'' : ℕ), a'' > 0 → 
      (∃ (b'' c'' : ℕ), b'' > 0 ∧ c'' > 0 ∧
        (a'' + b'' + c'') / 3 = 20 ∧
        a'' ≤ b'' ∧
        b'' ≤ c'' ∧
        c'' ≥ 25) →
      a'' ≥ a' := by
  sorry


end NUMINAMATH_CALUDE_smallest_possible_a_l1494_149403


namespace NUMINAMATH_CALUDE_square_division_impossibility_l1494_149470

/-- A point in a 2D plane --/
structure Point where
  x : ℝ
  y : ℝ

/-- A square in a 2D plane --/
structure Square where
  side : ℝ
  center : Point

/-- Represents a division of a square --/
structure SquareDivision where
  square : Square
  point1 : Point
  point2 : Point

/-- Checks if a point is inside a square --/
def is_inside (p : Point) (s : Square) : Prop :=
  abs (p.x - s.center.x) < s.side / 2 ∧ abs (p.y - s.center.y) < s.side / 2

/-- The theorem stating the impossibility of the division --/
theorem square_division_impossibility (s : Square) : 
  ¬ ∃ (d : SquareDivision), 
    (is_inside d.point1 s) ∧ 
    (is_inside d.point2 s) ∧ 
    (∃ (areas : List ℝ), areas.length = 9 ∧ (∀ a ∈ areas, a > 0) ∧ areas.sum = s.side ^ 2) :=
sorry

end NUMINAMATH_CALUDE_square_division_impossibility_l1494_149470


namespace NUMINAMATH_CALUDE_salt_solution_concentration_l1494_149477

/-- Proves that the concentration of the salt solution is 50% given the specified conditions. -/
theorem salt_solution_concentration
  (water_volume : Real)
  (salt_solution_volume : Real)
  (total_volume : Real)
  (mixture_concentration : Real)
  (h1 : water_volume = 1)
  (h2 : salt_solution_volume = 0.25)
  (h3 : total_volume = water_volume + salt_solution_volume)
  (h4 : mixture_concentration = 0.1)
  (h5 : salt_solution_volume * (concentration / 100) = total_volume * mixture_concentration) :
  concentration = 50 := by
  sorry

end NUMINAMATH_CALUDE_salt_solution_concentration_l1494_149477


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l1494_149486

theorem negation_of_universal_proposition :
  (¬ (∀ x : ℝ, x^2 + 1 > 0)) ↔ (∃ x : ℝ, x^2 + 1 ≤ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l1494_149486


namespace NUMINAMATH_CALUDE_cubic_roots_sum_cubes_l1494_149409

theorem cubic_roots_sum_cubes (r s t : ℝ) : 
  (8 * r^3 + 1001 * r + 2008 = 0) →
  (8 * s^3 + 1001 * s + 2008 = 0) →
  (8 * t^3 + 1001 * t + 2008 = 0) →
  (r + s)^3 + (s + t)^3 + (t + r)^3 = 753 := by
  sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_cubes_l1494_149409


namespace NUMINAMATH_CALUDE_smallest_digit_divisible_by_nine_l1494_149448

theorem smallest_digit_divisible_by_nine :
  ∃ (d : ℕ), d < 10 ∧ 
    (∀ (x : ℕ), x < d → ¬(528000 + x * 100 + 46) % 9 = 0) ∧
    (528000 + d * 100 + 46) % 9 = 0 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_digit_divisible_by_nine_l1494_149448


namespace NUMINAMATH_CALUDE_min_selling_price_A_l1494_149464

/-- Represents the water purifier problem with given conditions -/
structure WaterPurifierProblem where
  total_units : ℕ
  price_A : ℕ
  price_B : ℕ
  total_cost : ℕ
  units_A : ℕ
  units_B : ℕ
  min_total_profit : ℕ

/-- The specific instance of the water purifier problem -/
def problem : WaterPurifierProblem := {
  total_units := 160,
  price_A := 150,
  price_B := 350,
  total_cost := 36000,
  units_A := 100,
  units_B := 60,
  min_total_profit := 11000
}

/-- Theorem stating the minimum selling price for model A -/
theorem min_selling_price_A (p : WaterPurifierProblem) : 
  p.total_units = p.units_A + p.units_B →
  p.total_cost = p.price_A * p.units_A + p.price_B * p.units_B →
  ∀ selling_price_A : ℕ, 
    (selling_price_A - p.price_A) * p.units_A + 
    (2 * (selling_price_A - p.price_A)) * p.units_B ≥ p.min_total_profit →
    selling_price_A ≥ 200 := by
  sorry

#check min_selling_price_A problem

end NUMINAMATH_CALUDE_min_selling_price_A_l1494_149464


namespace NUMINAMATH_CALUDE_floor_neg_seven_fourths_l1494_149449

theorem floor_neg_seven_fourths : ⌊(-7 : ℚ) / 4⌋ = -2 := by sorry

end NUMINAMATH_CALUDE_floor_neg_seven_fourths_l1494_149449


namespace NUMINAMATH_CALUDE_stating_valid_orderings_count_l1494_149499

/-- 
Given a positive integer n, this function returns the number of ways to order 
integers from 1 to n, where except for the first integer, every integer differs 
by 1 from some integer to its left.
-/
def validOrderings (n : ℕ) : ℕ :=
  2^(n-1)

/-- 
Theorem stating that the number of valid orderings of integers from 1 to n 
is equal to 2^(n-1), where a valid ordering is one in which, except for the 
first integer, every integer differs by 1 from some integer to its left.
-/
theorem valid_orderings_count (n : ℕ) (h : n > 0) : 
  validOrderings n = 2^(n-1) := by
  sorry

end NUMINAMATH_CALUDE_stating_valid_orderings_count_l1494_149499


namespace NUMINAMATH_CALUDE_train_crossing_time_l1494_149444

theorem train_crossing_time (train_length : ℝ) (train_speed : ℝ) (man_speed : ℝ) :
  train_length = 330 →
  train_speed = 25 →
  man_speed = 2 →
  (train_length / ((train_speed + man_speed) * (1000 / 3600))) = 44 := by
  sorry

end NUMINAMATH_CALUDE_train_crossing_time_l1494_149444


namespace NUMINAMATH_CALUDE_car_travel_problem_l1494_149461

/-- Represents a car's travel information -/
structure CarTravel where
  speed : ℝ
  time : ℝ
  distance : ℝ

/-- The problem statement -/
theorem car_travel_problem 
  (p : CarTravel) 
  (q : CarTravel) 
  (h1 : p.time = 3) 
  (h2 : p.speed = 60) 
  (h3 : q.speed = 3 * p.speed) 
  (h4 : q.distance = p.distance / 2) 
  (h5 : p.distance = p.speed * p.time) 
  (h6 : q.distance = q.speed * q.time) : 
  q.time = 0.5 := by
sorry

end NUMINAMATH_CALUDE_car_travel_problem_l1494_149461


namespace NUMINAMATH_CALUDE_point_distance_theorem_l1494_149458

theorem point_distance_theorem (x y : ℝ) (h1 : x > 1) :
  y = 12 ∧ (x - 1)^2 + (y - 6)^2 = 10^2 →
  x^2 + y^2 = 15^2 := by
sorry

end NUMINAMATH_CALUDE_point_distance_theorem_l1494_149458


namespace NUMINAMATH_CALUDE_fewer_vip_tickets_l1494_149450

/-- Represents the number of tickets sold in a snooker tournament -/
structure TicketSales where
  vip : ℕ
  general : ℕ

/-- The ticket prices and sales data for the snooker tournament -/
def snookerTournament : TicketSales → Prop := fun ts =>
  ts.vip + ts.general = 320 ∧
  40 * ts.vip + 10 * ts.general = 7500

theorem fewer_vip_tickets (ts : TicketSales) 
  (h : snookerTournament ts) : ts.general - ts.vip = 34 := by
  sorry

end NUMINAMATH_CALUDE_fewer_vip_tickets_l1494_149450


namespace NUMINAMATH_CALUDE_coal_transport_trucks_l1494_149469

/-- The number of trucks needed to transport a given amount of coal -/
def trucks_needed (total_coal : ℕ) (truck_capacity : ℕ) : ℕ :=
  (total_coal + truck_capacity - 1) / truck_capacity

/-- Proof that 19 trucks are needed to transport 47,500 kg of coal when each truck can carry 2,500 kg -/
theorem coal_transport_trucks : trucks_needed 47500 2500 = 19 := by
  sorry

end NUMINAMATH_CALUDE_coal_transport_trucks_l1494_149469


namespace NUMINAMATH_CALUDE_probability_for_2x3x4_prism_l1494_149488

/-- Represents a rectangular prism with dimensions a, b, and c. -/
structure RectangularPrism where
  a : ℝ
  b : ℝ
  c : ℝ
  a_pos : 0 < a
  b_pos : 0 < b
  c_pos : 0 < c

/-- The probability that a plane determined by three randomly selected distinct vertices
    of a rectangular prism contains points inside the prism. -/
def probability_plane_intersects_interior (prism : RectangularPrism) : ℚ :=
  4/7

/-- Theorem stating that for a 2x3x4 rectangular prism, the probability of a plane
    determined by three randomly selected distinct vertices containing points inside
    the prism is 4/7. -/
theorem probability_for_2x3x4_prism :
  let prism : RectangularPrism := ⟨2, 3, 4, by norm_num, by norm_num, by norm_num⟩
  probability_plane_intersects_interior prism = 4/7 := by
  sorry


end NUMINAMATH_CALUDE_probability_for_2x3x4_prism_l1494_149488


namespace NUMINAMATH_CALUDE_diet_soda_bottles_l1494_149412

theorem diet_soda_bottles (regular_soda : ℕ) (lite_soda : ℕ) (total_bottles : ℕ) 
  (h1 : regular_soda = 57)
  (h2 : lite_soda = 27)
  (h3 : total_bottles = 110) :
  total_bottles - (regular_soda + lite_soda) = 26 := by
  sorry

end NUMINAMATH_CALUDE_diet_soda_bottles_l1494_149412


namespace NUMINAMATH_CALUDE_range_of_m_l1494_149432

theorem range_of_m (a b m : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b - a * b = 0)
  (h_log : ∀ a b, 0 < a → 0 < b → a + b - a * b = 0 → Real.log (m ^ 2 / (a + b)) ≤ 0) :
  -2 ≤ m ∧ m ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l1494_149432


namespace NUMINAMATH_CALUDE_algebraic_simplification_l1494_149490

theorem algebraic_simplification (x y : ℝ) :
  (18 * x^3 * y) * (8 * x * y^2) * (1 / (6 * x * y)^2) = 4 * x * y := by
  sorry

end NUMINAMATH_CALUDE_algebraic_simplification_l1494_149490


namespace NUMINAMATH_CALUDE_decimal_expansion_415th_digit_l1494_149414

/-- The decimal expansion of 17/29 -/
def decimal_expansion : ℚ := 17 / 29

/-- The length of the repeating cycle in the decimal expansion of 17/29 -/
def cycle_length : ℕ := 87

/-- The position of the 415th digit within the repeating cycle -/
def position_in_cycle : ℕ := 415 % cycle_length

/-- The 415th digit in the decimal expansion of 17/29 -/
def digit_415 : ℕ := 8

/-- Theorem stating that the 415th digit to the right of the decimal point
    in the decimal expansion of 17/29 is 8 -/
theorem decimal_expansion_415th_digit :
  digit_415 = 8 :=
sorry

end NUMINAMATH_CALUDE_decimal_expansion_415th_digit_l1494_149414


namespace NUMINAMATH_CALUDE_max_value_x_cubed_over_y_fourth_l1494_149451

theorem max_value_x_cubed_over_y_fourth (x y : ℝ) 
  (h1 : 3 ≤ x * y^2 ∧ x * y^2 ≤ 8) 
  (h2 : 4 ≤ x^2 / y ∧ x^2 / y ≤ 9) : 
  x^3 / y^4 ≤ 27 := by
sorry

end NUMINAMATH_CALUDE_max_value_x_cubed_over_y_fourth_l1494_149451


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l1494_149495

theorem complex_fraction_simplification :
  ∀ (z : ℂ), z = (3 : ℂ) + 8 * I →
  (1 / ((1 : ℂ) - 4 * I)) * z = 2 + (3 / 17) * I :=
by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l1494_149495


namespace NUMINAMATH_CALUDE_choose_three_from_nine_l1494_149476

theorem choose_three_from_nine : Nat.choose 9 3 = 84 := by
  sorry

end NUMINAMATH_CALUDE_choose_three_from_nine_l1494_149476


namespace NUMINAMATH_CALUDE_weight_of_new_person_l1494_149436

/-- The weight of the new person when the average weight of a group increases -/
def new_person_weight (num_people : ℕ) (avg_increase : ℝ) (replaced_weight : ℝ) : ℝ :=
  replaced_weight + num_people * avg_increase

/-- Theorem stating the weight of the new person under given conditions -/
theorem weight_of_new_person :
  new_person_weight 12 4 65 = 113 := by
  sorry

end NUMINAMATH_CALUDE_weight_of_new_person_l1494_149436


namespace NUMINAMATH_CALUDE_correct_delivery_probability_l1494_149471

def num_houses : ℕ := 5

def num_correct_deliveries : ℕ := 3

def probability_correct_deliveries : ℚ :=
  (num_houses.choose num_correct_deliveries : ℚ) / (num_houses.factorial : ℚ)

theorem correct_delivery_probability :
  probability_correct_deliveries = 1 / 12 := by
  sorry

end NUMINAMATH_CALUDE_correct_delivery_probability_l1494_149471


namespace NUMINAMATH_CALUDE_exist_unequal_triangles_with_equal_angles_and_two_sides_l1494_149429

-- Define two triangles
structure Triangle :=
  (a b c : ℝ)
  (α β γ : ℝ)

-- Define the conditions for our triangles
def triangles_satisfy_conditions (t1 t2 : Triangle) : Prop :=
  t1.α = t2.α ∧ t1.β = t2.β ∧ t1.γ = t2.γ ∧
  ((t1.a = t2.a ∧ t1.b = t2.b) ∨ (t1.a = t2.a ∧ t1.c = t2.c) ∨ (t1.b = t2.b ∧ t1.c = t2.c))

-- Define triangle inequality
def triangles_not_congruent (t1 t2 : Triangle) : Prop :=
  t1.a ≠ t2.a ∨ t1.b ≠ t2.b ∨ t1.c ≠ t2.c

-- Theorem statement
theorem exist_unequal_triangles_with_equal_angles_and_two_sides :
  ∃ (t1 t2 : Triangle), triangles_satisfy_conditions t1 t2 ∧ triangles_not_congruent t1 t2 :=
sorry

end NUMINAMATH_CALUDE_exist_unequal_triangles_with_equal_angles_and_two_sides_l1494_149429


namespace NUMINAMATH_CALUDE_stratified_sampling_population_size_l1494_149418

theorem stratified_sampling_population_size
  (x : ℕ) -- number of individuals in stratum A
  (y : ℕ) -- number of individuals in stratum B
  (h1 : (20 : ℚ) * y / (x + y) = (1 : ℚ) / 12 * y) -- equation from stratified sampling
  : x + y = 240 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_population_size_l1494_149418


namespace NUMINAMATH_CALUDE_polynomial_remainder_l1494_149473

theorem polynomial_remainder (x : ℝ) : 
  (5*x^8 - x^7 + 3*x^6 - 5*x^4 + 6*x^3 - 7) % (3*x - 6) = 1305 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l1494_149473


namespace NUMINAMATH_CALUDE_equation_solution_l1494_149428

theorem equation_solution : 
  ∃ n : ℚ, (3 - n) / (n + 2) + (3 * n - 9) / (3 - n) = 2 ∧ n = -7/6 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1494_149428


namespace NUMINAMATH_CALUDE_product_sum_theorem_l1494_149422

theorem product_sum_theorem (a b c d : ℝ) 
  (eq1 : a + b + c = 1)
  (eq2 : a + b + d = 6)
  (eq3 : a + c + d = 15)
  (eq4 : b + c + d = 10) :
  a * b + c * d = 408 / 9 := by
sorry

end NUMINAMATH_CALUDE_product_sum_theorem_l1494_149422


namespace NUMINAMATH_CALUDE_largest_non_sum_36_composite_l1494_149453

/-- A function that checks if a number is composite -/
def is_composite (n : ℕ) : Prop :=
  ∃ a b, 1 < a ∧ 1 < b ∧ n = a * b

/-- A function that checks if a number can be expressed as the sum of a positive integral multiple of 36 and a positive composite integer -/
def is_sum_of_multiple_36_and_composite (n : ℕ) : Prop :=
  ∃ k m, k > 0 ∧ is_composite m ∧ n = 36 * k + m

/-- The theorem stating that 145 is the largest positive integer that cannot be expressed as the sum of a positive integral multiple of 36 and a positive composite integer -/
theorem largest_non_sum_36_composite : 
  (∀ n : ℕ, n > 145 → is_sum_of_multiple_36_and_composite n) ∧
  ¬is_sum_of_multiple_36_and_composite 145 :=
sorry

end NUMINAMATH_CALUDE_largest_non_sum_36_composite_l1494_149453


namespace NUMINAMATH_CALUDE_emilys_spending_l1494_149438

/-- Emily's spending problem -/
theorem emilys_spending (X : ℝ) : 
  X + 2 * X + 3 * X = 120 → X = 20 := by
  sorry

end NUMINAMATH_CALUDE_emilys_spending_l1494_149438


namespace NUMINAMATH_CALUDE_percentage_decrease_after_increase_l1494_149456

theorem percentage_decrease_after_increase (x : ℝ) (hx : x > 0) :
  let y := x * 1.6
  y * (1 - 0.375) = x :=
by sorry

end NUMINAMATH_CALUDE_percentage_decrease_after_increase_l1494_149456


namespace NUMINAMATH_CALUDE_ivan_bought_ten_cards_l1494_149407

/-- The number of Uno Giant Family Cards Ivan bought -/
def num_cards : ℕ := 10

/-- The original price of each card in dollars -/
def original_price : ℚ := 12

/-- The discount per card in dollars -/
def discount : ℚ := 2

/-- The total amount Ivan paid in dollars -/
def total_paid : ℚ := 100

/-- Theorem stating that Ivan bought 10 Uno Giant Family Cards -/
theorem ivan_bought_ten_cards :
  (original_price - discount) * num_cards = total_paid :=
by sorry

end NUMINAMATH_CALUDE_ivan_bought_ten_cards_l1494_149407
