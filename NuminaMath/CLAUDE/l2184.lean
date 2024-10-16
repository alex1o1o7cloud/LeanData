import Mathlib

namespace NUMINAMATH_CALUDE_envelope_weight_proof_l2184_218406

/-- The weight of the envelope in Jessica's letter mailing scenario -/
def envelope_weight : ℚ := 2/5

/-- The number of pieces of paper used -/
def paper_count : ℕ := 8

/-- The weight of each piece of paper in ounces -/
def paper_weight : ℚ := 1/5

/-- The number of stamps needed -/
def stamps_needed : ℕ := 2

/-- The maximum weight in ounces that can be mailed with the given number of stamps -/
def max_weight (stamps : ℕ) : ℚ := stamps

theorem envelope_weight_proof :
  (paper_count : ℚ) * paper_weight + envelope_weight > (stamps_needed - 1 : ℚ) ∧
  (paper_count : ℚ) * paper_weight + envelope_weight ≤ stamps_needed ∧
  envelope_weight > 0 :=
sorry

end NUMINAMATH_CALUDE_envelope_weight_proof_l2184_218406


namespace NUMINAMATH_CALUDE_system_solution_l2184_218447

theorem system_solution : 
  let x : ℚ := -135/41
  let y : ℚ := 192/41
  (7 * x = -9 - 3 * y) ∧ (2 * x = 5 * y - 30) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l2184_218447


namespace NUMINAMATH_CALUDE_parabola_intersection_l2184_218450

def parabola1 (x : ℝ) : ℝ := 3 * x^2 - 9 * x - 5
def parabola2 (x : ℝ) : ℝ := x^2 - 6 * x + 10

theorem parabola_intersection :
  ∃ (x1 x2 : ℝ),
    x1 = (3 + Real.sqrt 129) / 4 ∧
    x2 = (3 - Real.sqrt 129) / 4 ∧
    parabola1 x1 = parabola2 x1 ∧
    parabola1 x2 = parabola2 x2 ∧
    ∀ (x : ℝ), parabola1 x = parabola2 x → x = x1 ∨ x = x2 :=
by sorry

end NUMINAMATH_CALUDE_parabola_intersection_l2184_218450


namespace NUMINAMATH_CALUDE_apple_distribution_l2184_218480

theorem apple_distribution (total_apples : ℕ) (red_percentage : ℚ) (classmates : ℕ) (extra_red : ℕ) : 
  total_apples = 80 →
  red_percentage = 3/5 →
  classmates = 6 →
  extra_red = 3 →
  (↑(total_apples) * red_percentage - extra_red) / classmates = 7.5 →
  ∃ (apples_per_classmate : ℕ), apples_per_classmate = 7 ∧ 
    apples_per_classmate * classmates ≤ ↑(total_apples) * red_percentage - extra_red ∧
    (apples_per_classmate + 1) * classmates > ↑(total_apples) * red_percentage - extra_red :=
by sorry

end NUMINAMATH_CALUDE_apple_distribution_l2184_218480


namespace NUMINAMATH_CALUDE_secret_spreading_l2184_218417

/-- 
Theorem: Secret Spreading
Given:
- On day 0 (Monday), one person knows a secret.
- Each day, every person who knows the secret tells two new people.
- The number of people who know the secret on day n is 2^(n+1) - 1.

Prove: It takes 9 days for 1023 people to know the secret.
-/
theorem secret_spreading (n : ℕ) : 
  (2^(n+1) - 1 = 1023) → n = 9 := by
  sorry

#check secret_spreading

end NUMINAMATH_CALUDE_secret_spreading_l2184_218417


namespace NUMINAMATH_CALUDE_salt_bag_weight_l2184_218489

/-- Given a bag of sugar weighing 16 kg and the fact that removing 4 kg from the combined
    weight of sugar and salt bags results in 42 kg, prove that the salt bag weighs 30 kg. -/
theorem salt_bag_weight (sugar_weight : ℕ) (combined_minus_four : ℕ) :
  sugar_weight = 16 ∧ combined_minus_four = 42 →
  ∃ (salt_weight : ℕ), salt_weight = 30 ∧ sugar_weight + salt_weight = combined_minus_four + 4 :=
by sorry

end NUMINAMATH_CALUDE_salt_bag_weight_l2184_218489


namespace NUMINAMATH_CALUDE_sieve_of_eratosthenes_complexity_l2184_218497

/-- The Sieve of Eratosthenes algorithm for finding prime numbers up to n. -/
def sieve_of_eratosthenes (n : ℕ) : List ℕ := sorry

/-- The time complexity function for the Sieve of Eratosthenes algorithm. -/
def time_complexity (n : ℕ) : ℝ := sorry

/-- Big O notation for comparing functions. -/
def big_o (f g : ℕ → ℝ) : Prop := 
  ∃ c k : ℝ, c > 0 ∧ ∀ n : ℕ, n ≥ k → f n ≤ c * g n

/-- Theorem stating that the time complexity of the Sieve of Eratosthenes is O(n log(n)^2). -/
theorem sieve_of_eratosthenes_complexity :
  big_o time_complexity (λ n => n * (Real.log n)^2) :=
sorry

end NUMINAMATH_CALUDE_sieve_of_eratosthenes_complexity_l2184_218497


namespace NUMINAMATH_CALUDE_baseball_games_per_month_l2184_218400

theorem baseball_games_per_month 
  (total_games : ℕ) 
  (season_months : ℕ) 
  (h1 : total_games = 14) 
  (h2 : season_months = 2) : 
  total_games / season_months = 7 := by
sorry

end NUMINAMATH_CALUDE_baseball_games_per_month_l2184_218400


namespace NUMINAMATH_CALUDE_empty_solution_set_range_l2184_218470

theorem empty_solution_set_range (m : ℝ) : 
  (∀ x : ℝ, m * x^2 + 2 * m * x + 1 ≥ 0) ↔ m ∈ Set.Icc 0 1 := by
  sorry

end NUMINAMATH_CALUDE_empty_solution_set_range_l2184_218470


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l2184_218471

-- Problem 1
theorem problem_1 : (1/2)⁻¹ - 2 * Real.tan (45 * π / 180) + |1 - Real.sqrt 2| = Real.sqrt 2 - 1 := by sorry

-- Problem 2
theorem problem_2 (a : ℝ) (h : a = Real.sqrt 3 + 2) : 
  (a / (a^2 - 4) + 1 / (2 - a)) / ((2*a + 4) / (a^2 + 4*a + 4)) = -Real.sqrt 3 / 3 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l2184_218471


namespace NUMINAMATH_CALUDE_greatest_divisor_three_consecutive_integers_l2184_218412

theorem greatest_divisor_three_consecutive_integers :
  ∃ (d : ℕ), d > 0 ∧ 
  (∀ (n : ℕ), n > 0 → d ∣ (n * (n + 1) * (n + 2))) ∧
  (∀ (k : ℕ), k > d → ∃ (m : ℕ), m > 0 ∧ ¬(k ∣ (m * (m + 1) * (m + 2)))) ∧
  d = 6 :=
sorry

end NUMINAMATH_CALUDE_greatest_divisor_three_consecutive_integers_l2184_218412


namespace NUMINAMATH_CALUDE_zoo_rhinos_count_zoo_rhinos_count_is_three_l2184_218428

/-- Calculates the number of endangered rhinos taken in by a zoo --/
theorem zoo_rhinos_count (initial_animals : ℕ) (gorilla_family : ℕ) (hippo : ℕ) 
  (lion_cubs : ℕ) (final_animals : ℕ) : ℕ :=
  let animals_after_gorillas := initial_animals - gorilla_family
  let animals_after_hippo := animals_after_gorillas + hippo
  let animals_after_cubs := animals_after_hippo + lion_cubs
  let meerkats := 2 * lion_cubs
  let animals_before_rhinos := animals_after_cubs + meerkats
  final_animals - animals_before_rhinos

/-- Proves that the number of endangered rhinos taken in is 3 --/
theorem zoo_rhinos_count_is_three : 
  zoo_rhinos_count 68 6 1 8 90 = 3 := by
  sorry

end NUMINAMATH_CALUDE_zoo_rhinos_count_zoo_rhinos_count_is_three_l2184_218428


namespace NUMINAMATH_CALUDE_f_is_generalized_distance_l2184_218437

def generalizedDistance (f : ℝ → ℝ → ℝ) : Prop :=
  (∀ x y, f x y ≥ 0 ∧ (f x y = 0 ↔ x = 0 ∧ y = 0)) ∧
  (∀ x y, f x y = f y x) ∧
  (∀ x y z, f x y ≤ f x z + f z y)

def f (x y : ℝ) : ℝ := x^2 + y^2

theorem f_is_generalized_distance : generalizedDistance f := by sorry

end NUMINAMATH_CALUDE_f_is_generalized_distance_l2184_218437


namespace NUMINAMATH_CALUDE_juice_transfer_difference_l2184_218411

/-- Represents a barrel with a certain volume of juice -/
structure Barrel where
  volume : ℝ

/-- Represents the state of two barrels -/
structure TwoBarrels where
  barrel1 : Barrel
  barrel2 : Barrel

/-- Transfers a given volume from one barrel to another -/
def transfer (barrels : TwoBarrels) (amount : ℝ) : TwoBarrels :=
  { barrel1 := { volume := barrels.barrel1.volume + amount },
    barrel2 := { volume := barrels.barrel2.volume - amount } }

/-- Calculates the difference in volume between two barrels -/
def volumeDifference (barrels : TwoBarrels) : ℝ :=
  barrels.barrel1.volume - barrels.barrel2.volume

/-- Theorem stating that after transferring 3 L from the 8 L barrel to the 10 L barrel,
    the difference in volume between the two barrels is 8 L -/
theorem juice_transfer_difference :
  let initialBarrels : TwoBarrels := { barrel1 := { volume := 10 }, barrel2 := { volume := 8 } }
  let finalBarrels := transfer initialBarrels 3
  volumeDifference finalBarrels = 8 := by
  sorry


end NUMINAMATH_CALUDE_juice_transfer_difference_l2184_218411


namespace NUMINAMATH_CALUDE_gcd_of_powers_of_47_l2184_218436

theorem gcd_of_powers_of_47 :
  Nat.Prime 47 →
  Nat.gcd (47^5 + 1) (47^5 + 47^3 + 47 + 1) = 1 := by
sorry

end NUMINAMATH_CALUDE_gcd_of_powers_of_47_l2184_218436


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l2184_218490

-- Define the sets M and N
def M : Set ℝ := {x : ℝ | x^2 < 4}
def N : Set ℝ := {x : ℝ | x < 1}

-- State the theorem
theorem intersection_of_M_and_N :
  M ∩ N = {x : ℝ | -2 < x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l2184_218490


namespace NUMINAMATH_CALUDE_root_difference_is_one_l2184_218454

theorem root_difference_is_one (p : ℝ) : 
  let α := (p + 1) / 2
  let β := (p - 1) / 2
  α - β = 1 ∧ 
  α^2 - p*α + (p^2 - 1)/4 = 0 ∧ 
  β^2 - p*β + (p^2 - 1)/4 = 0 ∧
  α ≥ β := by
sorry

end NUMINAMATH_CALUDE_root_difference_is_one_l2184_218454


namespace NUMINAMATH_CALUDE_book_price_change_l2184_218484

theorem book_price_change (P : ℝ) (h : P > 0) :
  let price_after_decrease : ℝ := P * 0.5
  let final_price : ℝ := P * 1.2
  ∃ x : ℝ, price_after_decrease * (1 + x / 100) = final_price ∧ x = 140 :=
by sorry

end NUMINAMATH_CALUDE_book_price_change_l2184_218484


namespace NUMINAMATH_CALUDE_perfect_square_factors_count_l2184_218433

/-- The number of positive factors of 450 that are perfect squares -/
def perfect_square_factors_of_450 : ℕ :=
  let prime_factorization : ℕ × ℕ × ℕ := (1, 2, 2)  -- Exponents of 2, 3, and 5 in 450's factorization
  2 * 2 * 2  -- Number of ways to choose even exponents for each prime factor

theorem perfect_square_factors_count :
  perfect_square_factors_of_450 = 8 :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_factors_count_l2184_218433


namespace NUMINAMATH_CALUDE_triangle_ABC_properties_l2184_218410

open Real

theorem triangle_ABC_properties (A B C a b c : Real) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧ 
  A + B + C = π ∧
  2 * sin A * sin C * (1 / (tan A * tan C) - 1) = -1 ∧
  a + c = 3 * sqrt 3 / 2 ∧
  b = sqrt 3 →
  B = π / 3 ∧ 
  (1 / 2) * a * c * sin B = 5 * sqrt 3 / 16 :=
by sorry

end NUMINAMATH_CALUDE_triangle_ABC_properties_l2184_218410


namespace NUMINAMATH_CALUDE_clothing_distribution_l2184_218486

def total_clothing : ℕ := 135
def first_load : ℕ := 29
def num_small_loads : ℕ := 7

theorem clothing_distribution :
  (total_clothing - first_load) / num_small_loads = 15 :=
by sorry

end NUMINAMATH_CALUDE_clothing_distribution_l2184_218486


namespace NUMINAMATH_CALUDE_angle_in_second_quadrant_l2184_218485

theorem angle_in_second_quadrant (θ : Real) (h : θ = 27 * Real.pi / 4) :
  0 < θ % (2 * Real.pi) ∧ θ % (2 * Real.pi) < Real.pi :=
by sorry

end NUMINAMATH_CALUDE_angle_in_second_quadrant_l2184_218485


namespace NUMINAMATH_CALUDE_plot_width_l2184_218461

/-- Given a rectangular plot with length 90 meters and perimeter that can be enclosed
    by 52 poles placed 5 meters apart, the width of the plot is 40 meters. -/
theorem plot_width (length : ℝ) (num_poles : ℕ) (pole_distance : ℝ) :
  length = 90 ∧ num_poles = 52 ∧ pole_distance = 5 →
  2 * (length + (num_poles * pole_distance / 2 - length) / 2) = num_poles * pole_distance →
  (num_poles * pole_distance / 2 - length) / 2 = 40 :=
by
  sorry

end NUMINAMATH_CALUDE_plot_width_l2184_218461


namespace NUMINAMATH_CALUDE_sum_1000th_to_1010th_term_l2184_218449

def arithmeticSequence (a₁ d n : ℕ) : ℕ := a₁ + (n - 1) * d

def sumArithmeticSequence (a₁ d m n : ℕ) : ℕ :=
  ((n - m + 1) * (arithmeticSequence a₁ d m + arithmeticSequence a₁ d n)) / 2

theorem sum_1000th_to_1010th_term :
  sumArithmeticSequence 3 7 1000 1010 = 77341 := by
  sorry

end NUMINAMATH_CALUDE_sum_1000th_to_1010th_term_l2184_218449


namespace NUMINAMATH_CALUDE_restaurant_menu_fraction_l2184_218435

theorem restaurant_menu_fraction (total_dishes : ℕ) 
  (h1 : 6 = (1 / 3 : ℚ) * total_dishes)
  (h2 : 4 ≤ 6) : 
  (2 : ℚ) / total_dishes = 1 / 9 := by sorry

end NUMINAMATH_CALUDE_restaurant_menu_fraction_l2184_218435


namespace NUMINAMATH_CALUDE_cubic_roots_sum_l2184_218499

theorem cubic_roots_sum (p q r : ℝ) : 
  (6 * p^3 + 500 * p + 1234 = 0) → 
  (6 * q^3 + 500 * q + 1234 = 0) → 
  (6 * r^3 + 500 * r + 1234 = 0) → 
  (p + q)^3 + (q + r)^3 + (r + p)^3 + 100 = 717 := by
sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_l2184_218499


namespace NUMINAMATH_CALUDE_class_size_proof_l2184_218419

/-- Given a student's rank from top and bottom in a class, 
    calculate the total number of students -/
def total_students (rank_from_top rank_from_bottom : ℕ) : ℕ :=
  rank_from_top + rank_from_bottom - 1

/-- Theorem stating that a class with a student ranking 24th from top 
    and 34th from bottom has 57 students in total -/
theorem class_size_proof :
  total_students 24 34 = 57 := by
  sorry

end NUMINAMATH_CALUDE_class_size_proof_l2184_218419


namespace NUMINAMATH_CALUDE_distance_sum_on_unit_circle_l2184_218415

theorem distance_sum_on_unit_circle (a b : ℝ) (h : a^2 + b^2 = 1) :
  a^4 + b^4 + ((a - b)^4 / 4) + ((a + b)^4 / 4) = 3/2 := by sorry

end NUMINAMATH_CALUDE_distance_sum_on_unit_circle_l2184_218415


namespace NUMINAMATH_CALUDE_trailingZeros_50_factorial_l2184_218404

/-- The number of trailing zeros in n! -/
def trailingZeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25)

/-- Theorem: The number of trailing zeros in 50! is 12 -/
theorem trailingZeros_50_factorial : trailingZeros 50 = 12 := by
  sorry

end NUMINAMATH_CALUDE_trailingZeros_50_factorial_l2184_218404


namespace NUMINAMATH_CALUDE_min_sum_of_parallel_vectors_l2184_218467

-- Define the vectors
def a (x : ℝ) : Fin 2 → ℝ := ![1, x - 1]
def b (y : ℝ) : Fin 2 → ℝ := ![y, 2]

-- Theorem statement
theorem min_sum_of_parallel_vectors (x y : ℝ) 
  (h1 : x - 1 ≥ 0) 
  (h2 : y ≥ 0) 
  (h3 : ∃ (k : ℝ), k ≠ 0 ∧ a x = k • b y) : 
  x + y ≥ 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_of_parallel_vectors_l2184_218467


namespace NUMINAMATH_CALUDE_consecutive_even_integers_cube_sum_l2184_218403

theorem consecutive_even_integers_cube_sum : 
  ∀ a : ℕ, 
    a > 0 → 
    (2*a - 2) * (2*a) * (2*a + 2) = 12 * (6*a) → 
    (2*a - 2)^3 + (2*a)^3 + (2*a + 2)^3 = 8568 :=
by
  sorry

end NUMINAMATH_CALUDE_consecutive_even_integers_cube_sum_l2184_218403


namespace NUMINAMATH_CALUDE_solve_system_l2184_218408

theorem solve_system (x y : ℝ) : 
  (5 * x - 3 = 2 * x + 9) → 
  (x + y = 10) → 
  (x = 4 ∧ y = 6) := by
sorry


end NUMINAMATH_CALUDE_solve_system_l2184_218408


namespace NUMINAMATH_CALUDE_final_price_approximation_l2184_218426

/-- Represents the price reduction scenario for oil --/
structure OilPriceReduction where
  initialPrice : ℝ
  week1Reduction : ℝ := 0.10
  week2Reduction : ℝ := 0.15
  week3Reduction : ℝ := 0.20
  additionalQuantity : ℝ := 5
  fixedCost : ℝ := 800

/-- Calculates the final price after three weeks of reductions --/
def finalPrice (opr : OilPriceReduction) : ℝ :=
  opr.initialPrice * (1 - opr.week1Reduction) * (1 - opr.week2Reduction) * (1 - opr.week3Reduction)

/-- Theorem stating the final reduced price is approximately 62.06 --/
theorem final_price_approximation (opr : OilPriceReduction) : 
  ∃ (initialQuantity : ℝ), 
    opr.fixedCost = initialQuantity * opr.initialPrice ∧
    opr.fixedCost = (initialQuantity + opr.additionalQuantity) * (finalPrice opr) ∧
    abs ((finalPrice opr) - 62.06) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_final_price_approximation_l2184_218426


namespace NUMINAMATH_CALUDE_probability_different_colors_eq_137_162_l2184_218492

/-- Represents the number of chips of each color in the bag -/
structure ChipCounts where
  blue : ℕ
  red : ℕ
  yellow : ℕ
  green : ℕ

/-- Calculates the probability of drawing two chips of different colors -/
def probabilityDifferentColors (counts : ChipCounts) : ℚ :=
  let total := counts.blue + counts.red + counts.yellow + counts.green
  let pBlue := counts.blue / total
  let pRed := counts.red / total
  let pYellow := counts.yellow / total
  let pGreen := counts.green / total
  pBlue * (1 - pBlue) + pRed * (1 - pRed) + pYellow * (1 - pYellow) + pGreen * (1 - pGreen)

/-- Theorem stating the probability of drawing two chips of different colors -/
theorem probability_different_colors_eq_137_162 :
  probabilityDifferentColors ⟨6, 5, 4, 3⟩ = 137 / 162 := by
  sorry

end NUMINAMATH_CALUDE_probability_different_colors_eq_137_162_l2184_218492


namespace NUMINAMATH_CALUDE_max_product_bound_l2184_218465

theorem max_product_bound (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a * b = a + b + 3) :
  a * b ≤ 9 := by
sorry

end NUMINAMATH_CALUDE_max_product_bound_l2184_218465


namespace NUMINAMATH_CALUDE_prob_black_second_draw_with_replacement_prob_black_second_draw_without_replacement_expected_black_balls_three_draws_l2184_218478

/-- A bag contains white and black balls -/
structure Bag where
  white : ℕ
  black : ℕ

/-- Probability of drawing a black ball from the bag -/
def prob_black (b : Bag) : ℚ :=
  b.black / (b.white + b.black)

/-- Probability of drawing a white ball from the bag -/
def prob_white (b : Bag) : ℚ :=
  b.white / (b.white + b.black)

/-- The initial bag configuration -/
def initial_bag : Bag :=
  { white := 4, black := 2 }

/-- Theorem for part (1) -/
theorem prob_black_second_draw_with_replacement (b : Bag) :
  prob_black b = 1 / 3 :=
sorry

/-- Theorem for part (2) -/
theorem prob_black_second_draw_without_replacement (b : Bag) :
  prob_black { white := b.white - 1, black := b.black } = 2 / 5 :=
sorry

/-- Expected value of binomial distribution -/
def binomial_expected (n : ℕ) (p : ℚ) : ℚ :=
  n * p

/-- Theorem for part (3) -/
theorem expected_black_balls_three_draws (b : Bag) :
  binomial_expected 3 (prob_black b) = 1 :=
sorry

end NUMINAMATH_CALUDE_prob_black_second_draw_with_replacement_prob_black_second_draw_without_replacement_expected_black_balls_three_draws_l2184_218478


namespace NUMINAMATH_CALUDE_sqrt_four_ninths_l2184_218493

theorem sqrt_four_ninths :
  Real.sqrt (4/9) = 2/3 ∨ Real.sqrt (4/9) = -2/3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_four_ninths_l2184_218493


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l2184_218422

theorem complex_number_quadrant : ∃ (z : ℂ), 
  z = (1 - Complex.I * Real.sqrt 3) / (2 * Complex.I) ∧ 
  z.re > 0 ∧ z.im < 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l2184_218422


namespace NUMINAMATH_CALUDE_parallelogram_side_sum_l2184_218424

/-- A parallelogram with given side lengths -/
structure Parallelogram where
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ → ℝ
  side4 : ℝ → ℝ

/-- The specific parallelogram from the problem -/
def problem_parallelogram : Parallelogram where
  side1 := 12
  side2 := 15
  side3 := fun y => 10 * y - 3
  side4 := fun x => 3 * x + 6

/-- The theorem stating the solution to the problem -/
theorem parallelogram_side_sum (p : Parallelogram) 
  (h1 : p.side1 = p.side4 1)
  (h2 : p.side2 = p.side3 2)
  (h3 : p = problem_parallelogram) :
  ∃ (x y : ℝ), x + y = 3.8 ∧ p.side3 y = 15 ∧ p.side4 x = 12 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_side_sum_l2184_218424


namespace NUMINAMATH_CALUDE_chocolate_eggs_weight_l2184_218473

/-- Calculates the total weight of remaining chocolate eggs after discarding one box -/
theorem chocolate_eggs_weight (total_eggs : ℕ) (weight_per_egg : ℕ) (num_boxes : ℕ) :
  total_eggs = 12 →
  weight_per_egg = 10 →
  num_boxes = 4 →
  (total_eggs - (total_eggs / num_boxes)) * weight_per_egg = 90 := by
  sorry

#check chocolate_eggs_weight

end NUMINAMATH_CALUDE_chocolate_eggs_weight_l2184_218473


namespace NUMINAMATH_CALUDE_alcohol_solution_volume_l2184_218476

theorem alcohol_solution_volume 
  (V : ℝ) 
  (h1 : 0.30 * V + 2.4 = 0.50 * (V + 2.4)) : 
  V = 6 := by
sorry

end NUMINAMATH_CALUDE_alcohol_solution_volume_l2184_218476


namespace NUMINAMATH_CALUDE_range_of_a_l2184_218401

open Set Real

theorem range_of_a (a : ℝ) : 
  (∀ x ∈ Icc 1 2, 3 * x^2 - a ≥ 0) ∧ 
  (∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0) →
  a ≤ -2 ∨ (1 ≤ a ∧ a ≤ 3) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l2184_218401


namespace NUMINAMATH_CALUDE_bee_colony_loss_rate_l2184_218414

/-- Proves that given a colony of bees with an initial population of 80,000 individuals,
    if after 50 days the population reduces to one-fourth of its initial size,
    then the daily loss rate is 1,200 bees per day. -/
theorem bee_colony_loss_rate (initial_population : ℕ) (days : ℕ) (final_population : ℕ) :
  initial_population = 80000 →
  days = 50 →
  final_population = initial_population / 4 →
  (initial_population - final_population) / days = 1200 := by
  sorry

end NUMINAMATH_CALUDE_bee_colony_loss_rate_l2184_218414


namespace NUMINAMATH_CALUDE_triangle_isosceles_condition_l2184_218463

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if a = 2b cos C, then the triangle is isosceles with B = C -/
theorem triangle_isosceles_condition (A B C a b c : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧  -- Angles are positive
  A + B + C = π ∧          -- Sum of angles in a triangle
  a > 0 ∧ b > 0 ∧ c > 0 ∧  -- Sides are positive
  a = 2 * b * Real.cos C   -- Given condition
  → B = C := by sorry

end NUMINAMATH_CALUDE_triangle_isosceles_condition_l2184_218463


namespace NUMINAMATH_CALUDE_opposite_numbers_sum_power_l2184_218468

/-- If a and b are opposite numbers, then (a+b)^2023 = 0 -/
theorem opposite_numbers_sum_power (a b : ℝ) : a = -b → (a + b)^2023 = 0 := by
  sorry

end NUMINAMATH_CALUDE_opposite_numbers_sum_power_l2184_218468


namespace NUMINAMATH_CALUDE_max_non_managers_l2184_218407

theorem max_non_managers (managers : ℕ) (non_managers : ℕ) : 
  managers = 8 →
  (managers : ℚ) / non_managers > 7 / 32 →
  non_managers ≤ 36 :=
by sorry

end NUMINAMATH_CALUDE_max_non_managers_l2184_218407


namespace NUMINAMATH_CALUDE_equilateral_triangle_side_length_l2184_218418

/-- An equilateral triangle with a point inside -/
structure EquilateralTriangleWithPoint where
  -- The side length of the equilateral triangle
  side_length : ℝ
  -- The perpendicular distances from the point to the sides
  dist_to_AB : ℝ
  dist_to_BC : ℝ
  dist_to_CA : ℝ
  -- Ensure the triangle is equilateral and the point is inside
  side_length_pos : 0 < side_length
  dist_pos : 0 < dist_to_AB ∧ 0 < dist_to_BC ∧ 0 < dist_to_CA
  point_inside : dist_to_AB + dist_to_BC + dist_to_CA < side_length * Real.sqrt 3

/-- The theorem statement -/
theorem equilateral_triangle_side_length 
  (triangle : EquilateralTriangleWithPoint) 
  (h1 : triangle.dist_to_AB = 2)
  (h2 : triangle.dist_to_BC = 3)
  (h3 : triangle.dist_to_CA = 4) : 
  triangle.side_length = 6 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_side_length_l2184_218418


namespace NUMINAMATH_CALUDE_dads_dimes_l2184_218423

theorem dads_dimes (initial : ℕ) (from_mother : ℕ) (total : ℕ) : 
  initial = 7 → from_mother = 4 → total = 19 → 
  total - (initial + from_mother) = 8 := by
sorry

end NUMINAMATH_CALUDE_dads_dimes_l2184_218423


namespace NUMINAMATH_CALUDE_average_marks_chemistry_mathematics_l2184_218496

/-- Given that the total marks in physics, chemistry, and mathematics is 140 more than 
    the marks in physics, prove that the average mark in chemistry and mathematics is 70. -/
theorem average_marks_chemistry_mathematics (P C M : ℕ) 
  (h : P + C + M = P + 140) : (C + M) / 2 = 70 := by
  sorry

end NUMINAMATH_CALUDE_average_marks_chemistry_mathematics_l2184_218496


namespace NUMINAMATH_CALUDE_rectangle_length_l2184_218483

/-- The length of a rectangle with width 4 cm and area equal to a square with side length 8 cm -/
theorem rectangle_length (square_side : ℝ) (rect_width : ℝ) (rect_length : ℝ) : 
  square_side = 8 →
  rect_width = 4 →
  square_side * square_side = rect_width * rect_length →
  rect_length = 16 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_length_l2184_218483


namespace NUMINAMATH_CALUDE_total_puppies_adopted_l2184_218443

def puppies_week1 : ℕ := 20

def puppies_week2 : ℕ := (2 * puppies_week1) / 5

def puppies_week3 : ℕ := (3 * puppies_week2) / 8

def puppies_week4 : ℕ := 2 * puppies_week2

def puppies_week5 : ℕ := puppies_week1 + 10

def puppies_week6 : ℕ := 2 * puppies_week3 - 5

def puppies_week7 : ℕ := 2 * puppies_week6

def puppies_week8 : ℕ := (7 * puppies_week6) / 4

def puppies_week9 : ℕ := (3 * puppies_week8) / 2

def puppies_week10 : ℕ := (9 * puppies_week1) / 4

def puppies_week11 : ℕ := (5 * puppies_week10) / 6

theorem total_puppies_adopted : 
  puppies_week1 + puppies_week2 + puppies_week3 + puppies_week4 + 
  puppies_week5 + puppies_week6 + puppies_week7 + puppies_week8 + 
  puppies_week9 + puppies_week10 + puppies_week11 = 164 := by
  sorry

end NUMINAMATH_CALUDE_total_puppies_adopted_l2184_218443


namespace NUMINAMATH_CALUDE_f_positive_iff_triangle_l2184_218456

/-- A polynomial function representing the triangle inequality condition -/
def f (x y z : ℝ) : ℝ := (x + y + z) * (-x + y + z) * (x - y + z) * (x + y - z)

/-- Predicate to check if three real numbers can form the sides of a triangle -/
def is_triangle (x y z : ℝ) : Prop :=
  x > 0 ∧ y > 0 ∧ z > 0 ∧ x + y > z ∧ x + z > y ∧ y + z > x

/-- Theorem stating that f is positive iff x, y, z can form a triangle -/
theorem f_positive_iff_triangle (x y z : ℝ) :
  f x y z > 0 ↔ is_triangle (|x|) (|y|) (|z|) := by sorry

end NUMINAMATH_CALUDE_f_positive_iff_triangle_l2184_218456


namespace NUMINAMATH_CALUDE_probability_not_jim_pictures_l2184_218466

/-- Given a set of pictures, calculate the probability of picking two pictures
    that are not among those bought by Jim. -/
theorem probability_not_jim_pictures
  (total_pictures : ℕ)
  (jim_bought : ℕ)
  (pick_count : ℕ)
  (h_total : total_pictures = 10)
  (h_jim : jim_bought = 3)
  (h_pick : pick_count = 2) :
  (pick_count.choose (total_pictures - jim_bought)) / (pick_count.choose total_pictures) = 7 / 15 := by
  sorry

end NUMINAMATH_CALUDE_probability_not_jim_pictures_l2184_218466


namespace NUMINAMATH_CALUDE_function_properties_and_inequality_l2184_218469

/-- Given a function f(x) = ax / (x^2 + b) with specific properties, 
    prove its exact form and a related inequality. -/
theorem function_properties_and_inequality 
  (f : ℝ → ℝ) 
  (a b : ℝ) 
  (h_a : a > 0) 
  (h_b : b > 1) 
  (h_def : ∀ x, f x = a * x / (x^2 + b)) 
  (h_f1 : f 1 = 1) 
  (h_max : ∀ x, f x ≤ 3 * Real.sqrt 2 / 4) 
  (h_attains_max : ∃ x, f x = 3 * Real.sqrt 2 / 4) :
  (∀ x, f x = 3 * x / (x^2 + 2)) ∧ 
  (∀ m, (2 < m ∧ m ≤ 4) ↔ 
    (∀ x ∈ Set.Icc 1 2, f x ≤ 3 * m / ((x^2 + 2) * |x - m|))) := by
  sorry

end NUMINAMATH_CALUDE_function_properties_and_inequality_l2184_218469


namespace NUMINAMATH_CALUDE_item_list_price_l2184_218498

theorem item_list_price (list_price : ℝ) : 
  (0.15 * (list_price - 15) = 0.25 * (list_price - 25)) → list_price = 40 := by
  sorry

end NUMINAMATH_CALUDE_item_list_price_l2184_218498


namespace NUMINAMATH_CALUDE_final_dog_count_l2184_218444

def initial_dogs : ℕ := 80
def adoption_rate : ℚ := 40 / 100
def returned_dogs : ℕ := 5

theorem final_dog_count : 
  initial_dogs - (initial_dogs * adoption_rate).floor + returned_dogs = 53 := by
  sorry

end NUMINAMATH_CALUDE_final_dog_count_l2184_218444


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2184_218431

theorem complex_equation_solution (x y : ℝ) : 
  (Complex.mk (2 * x - 1) 1 = Complex.mk y (y - 2)) → x = 2 ∧ y = 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2184_218431


namespace NUMINAMATH_CALUDE_service_cost_equations_global_connect_more_cost_effective_l2184_218494

/-- Represents the cost of a mobile communication service based on monthly fee and per-minute rate -/
def service_cost (monthly_fee : ℝ) (per_minute_rate : ℝ) (minutes : ℝ) : ℝ :=
  monthly_fee + per_minute_rate * minutes

/-- Theorem stating the cost equations for Global Connect and Quick Connect services -/
theorem service_cost_equations 
  (x : ℝ) 
  (y1 : ℝ) 
  (y2 : ℝ) : 
  y1 = service_cost 50 0.4 x ∧ 
  y2 = service_cost 0 0.6 x :=
sorry

/-- Theorem stating that Global Connect is more cost-effective for 300 minutes of calls -/
theorem global_connect_more_cost_effective : 
  service_cost 50 0.4 300 < service_cost 0 0.6 300 :=
sorry

end NUMINAMATH_CALUDE_service_cost_equations_global_connect_more_cost_effective_l2184_218494


namespace NUMINAMATH_CALUDE_matching_shoes_probability_l2184_218420

theorem matching_shoes_probability (n : ℕ) (h : n = 12) :
  let total_shoes := 2 * n
  let total_combinations := (total_shoes * (total_shoes - 1)) / 2
  let matching_pairs := n
  (matching_pairs : ℚ) / total_combinations = 1 / 23 := by
  sorry

end NUMINAMATH_CALUDE_matching_shoes_probability_l2184_218420


namespace NUMINAMATH_CALUDE_range_of_a_l2184_218438

open Set
open Real

theorem range_of_a (a : ℝ) : 
  (¬ ∃ x₀ : ℝ, x₀^2 - a*x₀ + a = 0) →
  (∀ x : ℝ, x > 1 → x + 1/(x-1) ≥ a) →
  a ∈ Ioo 0 3 := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l2184_218438


namespace NUMINAMATH_CALUDE_quadratic_equations_solutions_l2184_218458

theorem quadratic_equations_solutions :
  (∀ x : ℝ, 2 * (x - 1)^2 = 18 ↔ x = 4 ∨ x = -2) ∧
  (∀ x : ℝ, x^2 - 4*x - 3 = 0 ↔ x = 2 + Real.sqrt 7 ∨ x = 2 - Real.sqrt 7) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equations_solutions_l2184_218458


namespace NUMINAMATH_CALUDE_square_difference_fourth_power_l2184_218442

theorem square_difference_fourth_power : (7^2 - 5^2)^4 = 331776 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_fourth_power_l2184_218442


namespace NUMINAMATH_CALUDE_perpendicular_vectors_m_value_l2184_218488

/-- Given two vectors a and b in ℝ², prove that if a = (1,2) and b = (-1,m) are perpendicular, then m = 1/2 -/
theorem perpendicular_vectors_m_value (a b : ℝ × ℝ) (m : ℝ) : 
  a = (1, 2) → b = (-1, m) → a.1 * b.1 + a.2 * b.2 = 0 → m = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_m_value_l2184_218488


namespace NUMINAMATH_CALUDE_least_number_with_divisibility_conditions_l2184_218446

theorem least_number_with_divisibility_conditions : ∃ n : ℕ, 
  (∀ k : ℕ, 2 ≤ k → k ≤ 7 → n % k = 1) ∧ 
  (n % 8 = 0) ∧
  (∀ m : ℕ, m < n → ¬((∀ k : ℕ, 2 ≤ k → k ≤ 7 → m % k = 1) ∧ (m % 8 = 0))) ∧
  n = 1681 := by
sorry

end NUMINAMATH_CALUDE_least_number_with_divisibility_conditions_l2184_218446


namespace NUMINAMATH_CALUDE_solution_set_and_range_l2184_218487

def f (x : ℝ) : ℝ := |2*x + 1| + |2*x - 3|

theorem solution_set_and_range :
  (∀ x : ℝ, f x ≤ 6 ↔ x ∈ Set.Icc (-1) 2) ∧
  (∀ a : ℝ, a > 0 → (∃ x : ℝ, f x < |a - 2|) ↔ a > 6) := by sorry

end NUMINAMATH_CALUDE_solution_set_and_range_l2184_218487


namespace NUMINAMATH_CALUDE_binomial_expansion_coefficient_l2184_218491

theorem binomial_expansion_coefficient (a : ℝ) : 
  (∃ k : ℝ, k = (Nat.choose 6 3) * a^3 ∧ k = -160) → a = -2 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_coefficient_l2184_218491


namespace NUMINAMATH_CALUDE_unique_integer_solution_l2184_218416

theorem unique_integer_solution (x y : ℤ) : 
  ({2 * x, x + y} : Set ℤ) = {7, 4} → x = 2 ∧ y = 5 := by
  sorry

end NUMINAMATH_CALUDE_unique_integer_solution_l2184_218416


namespace NUMINAMATH_CALUDE_shaded_area_equals_sixteen_twentyseventh_l2184_218474

/-- Represents the fraction of shaded area in each iteration -/
def shaded_fraction : ℕ → ℚ
  | 0 => 4/9
  | n + 1 => shaded_fraction n + (4/9) * (1/4)^(n+1)

/-- The limit of the shaded fraction as the number of iterations approaches infinity -/
def shaded_limit : ℚ := 16/27

theorem shaded_area_equals_sixteen_twentyseventh :
  ∀ ε > 0, ∃ N, ∀ n ≥ N, |shaded_fraction n - shaded_limit| < ε :=
sorry

end NUMINAMATH_CALUDE_shaded_area_equals_sixteen_twentyseventh_l2184_218474


namespace NUMINAMATH_CALUDE_adults_at_ball_game_l2184_218413

theorem adults_at_ball_game :
  let num_children : ℕ := 11
  let adult_ticket_price : ℕ := 8
  let child_ticket_price : ℕ := 4
  let total_bill : ℕ := 124
  let num_adults : ℕ := (total_bill - num_children * child_ticket_price) / adult_ticket_price
  num_adults = 10 := by
sorry

end NUMINAMATH_CALUDE_adults_at_ball_game_l2184_218413


namespace NUMINAMATH_CALUDE_students_behind_yoongi_l2184_218441

/-- Given a line of students, calculates the number of students behind a specific student -/
def studentsInBack (totalStudents : ℕ) (studentsBetween : ℕ) : ℕ :=
  totalStudents - (studentsBetween + 2)

theorem students_behind_yoongi :
  let totalStudents : ℕ := 20
  let studentsBetween : ℕ := 5
  studentsInBack totalStudents studentsBetween = 13 := by
  sorry

end NUMINAMATH_CALUDE_students_behind_yoongi_l2184_218441


namespace NUMINAMATH_CALUDE_shells_not_red_or_green_l2184_218457

theorem shells_not_red_or_green (total : ℕ) (red : ℕ) (green : ℕ) 
  (h1 : total = 291) (h2 : red = 76) (h3 : green = 49) :
  total - (red + green) = 166 := by
  sorry

end NUMINAMATH_CALUDE_shells_not_red_or_green_l2184_218457


namespace NUMINAMATH_CALUDE_quadratic_root_difference_l2184_218429

theorem quadratic_root_difference (a b c : ℝ) (h : a > 0) :
  let equation := fun x => (5 + 2 * Real.sqrt 5) * x^2 - (3 + Real.sqrt 5) * x + 1
  let larger_root := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let smaller_root := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  equation larger_root = 0 ∧ equation smaller_root = 0 →
  larger_root - smaller_root = Real.sqrt (-3 + (2 * Real.sqrt 5) / 5) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_difference_l2184_218429


namespace NUMINAMATH_CALUDE_sum_of_irrationals_can_be_rational_l2184_218475

theorem sum_of_irrationals_can_be_rational :
  ∃ (x y : ℝ), Irrational x ∧ Irrational y ∧ ∃ (q : ℚ), x + y = q := by
  sorry

end NUMINAMATH_CALUDE_sum_of_irrationals_can_be_rational_l2184_218475


namespace NUMINAMATH_CALUDE_units_digit_34_plus_47_base_8_l2184_218453

def base_8_addition (a b : Nat) : Nat :=
  (a + b) % 8

theorem units_digit_34_plus_47_base_8 :
  base_8_addition (34 % 8) (47 % 8) = 3 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_34_plus_47_base_8_l2184_218453


namespace NUMINAMATH_CALUDE_quadratic_polynomial_solutions_l2184_218464

-- Define a quadratic polynomial
def QuadraticPolynomial (α : Type*) [Field α] := α → α

-- Define the property of having exactly three solutions for (f(x))^3 - 4f(x) = 0
def HasThreeSolutionsCubicMinusFour (f : QuadraticPolynomial ℝ) : Prop :=
  ∃ (x₁ x₂ x₃ : ℝ), (∀ x : ℝ, f x ^ 3 - 4 * f x = 0 ↔ x = x₁ ∨ x = x₂ ∨ x = x₃) ∧
  x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃

-- Define the property of having exactly two solutions for (f(x))^2 = 1
def HasTwoSolutionsSquaredEqualsOne (f : QuadraticPolynomial ℝ) : Prop :=
  ∃ (y₁ y₂ : ℝ), (∀ x : ℝ, f x ^ 2 = 1 ↔ x = y₁ ∨ x = y₂) ∧ y₁ ≠ y₂

-- The theorem statement
theorem quadratic_polynomial_solutions (f : QuadraticPolynomial ℝ) :
  HasThreeSolutionsCubicMinusFour f → HasTwoSolutionsSquaredEqualsOne f := by
  sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_solutions_l2184_218464


namespace NUMINAMATH_CALUDE_distinct_arrangements_count_is_eight_l2184_218495

/-- Represents a key on the keychain -/
inductive Key
| House
| Car
| Work
| Garage
| Other

/-- Represents a pair of adjacent keys -/
structure KeyPair :=
  (first : Key)
  (second : Key)

/-- Represents an arrangement of keys on the keychain -/
structure KeyArrangement :=
  (pair1 : KeyPair)
  (pair2 : KeyPair)
  (single : Key)

/-- Checks if two KeyArrangements are considered identical (up to rotation and reflection) -/
def are_identical (a b : KeyArrangement) : Prop := sorry

/-- The set of all valid key arrangements -/
def valid_arrangements : Set KeyArrangement :=
  {arr | (arr.pair1.first = Key.House ∧ arr.pair1.second = Key.Car) ∨
         (arr.pair1.first = Key.Car ∧ arr.pair1.second = Key.House) ∧
         (arr.pair2.first = Key.Work ∧ arr.pair2.second = Key.Garage) ∨
         (arr.pair2.first = Key.Garage ∧ arr.pair2.second = Key.Work) ∧
         arr.single = Key.Other}

/-- The number of distinct arrangements -/
def distinct_arrangement_count : ℕ := sorry

theorem distinct_arrangements_count_is_eight :
  distinct_arrangement_count = 8 := by sorry

end NUMINAMATH_CALUDE_distinct_arrangements_count_is_eight_l2184_218495


namespace NUMINAMATH_CALUDE_michaels_lap_time_l2184_218402

/-- Race on a circular track -/
structure RaceTrack where
  length : ℝ
  donovan_lap_time : ℝ
  michael_laps_to_pass : ℕ

/-- Given race conditions, prove Michael's lap time -/
theorem michaels_lap_time (race : RaceTrack)
  (h1 : race.length = 300)
  (h2 : race.donovan_lap_time = 45)
  (h3 : race.michael_laps_to_pass = 9) :
  ∃ t : ℝ, t = 50 ∧ t * race.michael_laps_to_pass = (race.michael_laps_to_pass + 1) * race.donovan_lap_time :=
by sorry

end NUMINAMATH_CALUDE_michaels_lap_time_l2184_218402


namespace NUMINAMATH_CALUDE_sum_of_roots_equation_l2184_218477

theorem sum_of_roots_equation (x : ℝ) : (x - 1) * (x + 4) = 18 → ∃ y : ℝ, (y - 1) * (y + 4) = 18 ∧ x + y = -3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_equation_l2184_218477


namespace NUMINAMATH_CALUDE_martha_tech_support_ratio_l2184_218472

/-- Proves that the ratio of yelling time to hold time is 1:2 given the conditions of Martha's tech support experience. -/
theorem martha_tech_support_ratio :
  ∀ (router_time hold_time yelling_time : ℕ),
    router_time = 10 →
    hold_time = 6 * router_time →
    router_time + hold_time + yelling_time = 100 →
    (yelling_time : ℚ) / hold_time = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_martha_tech_support_ratio_l2184_218472


namespace NUMINAMATH_CALUDE_quadratic_composition_no_roots_l2184_218445

/-- A quadratic function f(x) = ax^2 + bx + c -/
def QuadraticFunction (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

/-- The statement that f(x) = x has no real roots -/
def NoRealRoots (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f x ≠ x

theorem quadratic_composition_no_roots (a b c : ℝ) (ha : a ≠ 0) :
  let f := QuadraticFunction a b c
  NoRealRoots f → NoRealRoots (f ∘ f) := by
  sorry

#check quadratic_composition_no_roots

end NUMINAMATH_CALUDE_quadratic_composition_no_roots_l2184_218445


namespace NUMINAMATH_CALUDE_repeating_decimal_as_fraction_l2184_218405

/-- The repeating decimal 0.4747... expressed as a real number -/
def repeating_decimal : ℚ :=
  (0.47 : ℚ) + (0.0047 : ℚ) / (1 - (0.01 : ℚ))

theorem repeating_decimal_as_fraction :
  repeating_decimal = 47 / 99 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_as_fraction_l2184_218405


namespace NUMINAMATH_CALUDE_max_wednesday_pizzas_exists_five_pizzas_wednesday_l2184_218455

/-- Represents the number of pizzas baked on each day -/
structure PizzaSchedule where
  saturday : ℕ
  sunday : ℕ
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ

/-- Checks if the pizza schedule satisfies the given conditions -/
def isValidSchedule (schedule : PizzaSchedule) : Prop :=
  let total := 50
  schedule.saturday = (3 * total) / 5 ∧
  schedule.sunday = (3 * (total - schedule.saturday)) / 5 ∧
  schedule.monday < schedule.sunday ∧
  schedule.tuesday < schedule.monday ∧
  schedule.wednesday < schedule.tuesday ∧
  schedule.saturday + schedule.sunday + schedule.monday + schedule.tuesday + schedule.wednesday = total

/-- Theorem stating the maximum number of pizzas that could be baked on Wednesday -/
theorem max_wednesday_pizzas :
  ∀ (schedule : PizzaSchedule), isValidSchedule schedule → schedule.wednesday ≤ 5 := by
  sorry

/-- Theorem stating that there exists a valid schedule with 5 pizzas on Wednesday -/
theorem exists_five_pizzas_wednesday :
  ∃ (schedule : PizzaSchedule), isValidSchedule schedule ∧ schedule.wednesday = 5 := by
  sorry

end NUMINAMATH_CALUDE_max_wednesday_pizzas_exists_five_pizzas_wednesday_l2184_218455


namespace NUMINAMATH_CALUDE_pirate_loot_sum_is_correct_l2184_218430

/-- Converts a base 5 number to base 10 -/
def base5ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (5^i)) 0

/-- The sum of the pirate's loot in base 10 -/
def pirateLootSum : Nat :=
  base5ToBase10 [2, 3, 1, 4] + 
  base5ToBase10 [2, 3, 4, 1] + 
  base5ToBase10 [4, 2, 0, 2] + 
  base5ToBase10 [4, 2, 2]

theorem pirate_loot_sum_is_correct : pirateLootSum = 1112 := by
  sorry

end NUMINAMATH_CALUDE_pirate_loot_sum_is_correct_l2184_218430


namespace NUMINAMATH_CALUDE_car_stop_time_l2184_218427

/-- The distance traveled by a car after braking -/
def S (t : ℝ) : ℝ := -3 * t^2 + 18 * t

/-- The time required for the car to stop after braking -/
theorem car_stop_time : ∃ t : ℝ, S t = 0 ∧ t = 6 := by
  sorry

end NUMINAMATH_CALUDE_car_stop_time_l2184_218427


namespace NUMINAMATH_CALUDE_largest_decimal_l2184_218459

theorem largest_decimal : 
  let a := 0.9877
  let b := 0.9789
  let c := 0.9700
  let d := 0.9790
  let e := 0.9709
  (a > b) ∧ (a > c) ∧ (a > d) ∧ (a > e) :=
by sorry

end NUMINAMATH_CALUDE_largest_decimal_l2184_218459


namespace NUMINAMATH_CALUDE_brads_running_speed_l2184_218432

/-- Prove that Brad's running speed is 6 km/h given the conditions of the problem -/
theorem brads_running_speed (maxwell_speed : ℝ) (total_distance : ℝ) (brad_delay : ℝ) (total_time : ℝ)
  (h1 : maxwell_speed = 4)
  (h2 : total_distance = 74)
  (h3 : brad_delay = 1)
  (h4 : total_time = 8) :
  (total_distance - maxwell_speed * total_time) / (total_time - brad_delay) = 6 := by
  sorry

end NUMINAMATH_CALUDE_brads_running_speed_l2184_218432


namespace NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l2184_218452

-- Problem 1
theorem simplify_expression_1 (x : ℝ) : 
  x^2 + (3*x - 5) - (4*x - 1) = x^2 - x - 4 := by sorry

-- Problem 2
theorem simplify_expression_2 (a b : ℝ) : 
  7*a + 3*(a - 3*b) - 2*(b - a) = 12*a - 11*b := by sorry

end NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l2184_218452


namespace NUMINAMATH_CALUDE_final_x_value_l2184_218462

/-- Represents the state of the program at each iteration -/
structure ProgramState where
  x : ℕ
  y : ℕ

/-- Updates the program state according to the given rules -/
def updateState (state : ProgramState) : ProgramState :=
  { x := state.x + 2,
    y := state.y + state.x + 2 }

/-- Checks if the program should continue running -/
def shouldContinue (state : ProgramState) : Bool :=
  state.y < 10000

/-- Computes the final state of the program -/
def finalState : ProgramState :=
  sorry

/-- Proves that the final value of x is 201 -/
theorem final_x_value :
  finalState.x = 201 :=
sorry

end NUMINAMATH_CALUDE_final_x_value_l2184_218462


namespace NUMINAMATH_CALUDE_uf_games_before_championship_l2184_218425

/-- The number of games UF played before the championship game -/
def n : ℕ := sorry

/-- The total points UF scored in previous games -/
def total_points : ℕ := 720

/-- UF's score in the championship game -/
def championship_score : ℕ := total_points / (2 * n) - 2

/-- UF's opponent's score in the championship game -/
def opponent_score : ℕ := 11

theorem uf_games_before_championship : 
  (total_points / n = championship_score + 2) ∧ 
  (championship_score = opponent_score + 2) ∧
  (n = 24) := by sorry

end NUMINAMATH_CALUDE_uf_games_before_championship_l2184_218425


namespace NUMINAMATH_CALUDE_point_coordinates_l2184_218451

-- Define a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Define the second quadrant
def secondQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

-- Define distance to x-axis
def distToXAxis (p : Point) : ℝ :=
  |p.y|

-- Define distance to y-axis
def distToYAxis (p : Point) : ℝ :=
  |p.x|

theorem point_coordinates (M : Point) :
  secondQuadrant M ∧ distToXAxis M = 1 ∧ distToYAxis M = 2 →
  M.x = -2 ∧ M.y = 1 := by sorry

end NUMINAMATH_CALUDE_point_coordinates_l2184_218451


namespace NUMINAMATH_CALUDE_total_pencils_l2184_218439

/-- Given 4.0 pencil boxes, each filled with 648.0 pencils, prove that the total number of pencils is 2592.0 -/
theorem total_pencils (num_boxes : Float) (pencils_per_box : Float) 
  (h1 : num_boxes = 4.0) 
  (h2 : pencils_per_box = 648.0) : 
  num_boxes * pencils_per_box = 2592.0 := by
  sorry

end NUMINAMATH_CALUDE_total_pencils_l2184_218439


namespace NUMINAMATH_CALUDE_circle_area_when_six_times_reciprocal_circumference_equals_diameter_l2184_218448

theorem circle_area_when_six_times_reciprocal_circumference_equals_diameter :
  ∀ (r : ℝ), r > 0 → (6 * (1 / (2 * π * r)) = 2 * r) → π * r^2 = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_when_six_times_reciprocal_circumference_equals_diameter_l2184_218448


namespace NUMINAMATH_CALUDE_marilyn_bottle_caps_l2184_218481

/-- The number of bottle caps Marilyn starts with -/
def initial_caps : ℕ := 51

/-- The number of bottle caps Marilyn shares with Nancy -/
def shared_caps : ℕ := 36

/-- The number of bottle caps Marilyn ends up with -/
def remaining_caps : ℕ := initial_caps - shared_caps

theorem marilyn_bottle_caps : remaining_caps = 15 := by
  sorry

end NUMINAMATH_CALUDE_marilyn_bottle_caps_l2184_218481


namespace NUMINAMATH_CALUDE_reflection_line_equation_l2184_218479

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- The line of reflection for a triangle and its image -/
def LineOfReflection (A B C A' B' C' : Point2D) : Set (ℝ × ℝ) :=
  {(x, y) | ∀ (P P' : Point2D), (P = A ∧ P' = A') ∨ (P = B ∧ P' = B') ∨ (P = C ∧ P' = C') →
    y = (P.y + P'.y) / 2}

/-- The theorem stating that the line of reflection for the given triangle is y = -1 -/
theorem reflection_line_equation :
  let D : Point2D := ⟨1, 4⟩
  let E : Point2D := ⟨6, 9⟩
  let F : Point2D := ⟨-5, 7⟩
  let D' : Point2D := ⟨1, -6⟩
  let E' : Point2D := ⟨6, -11⟩
  let F' : Point2D := ⟨-5, -9⟩
  LineOfReflection D E F D' E' F' = {(x, y) | y = -1} :=
by sorry

end NUMINAMATH_CALUDE_reflection_line_equation_l2184_218479


namespace NUMINAMATH_CALUDE_mask_selection_probability_l2184_218460

theorem mask_selection_probability :
  let total_colors : ℕ := 5
  let selected_masks : ℕ := 3
  let favorable_outcomes : ℕ := (total_colors - 2).choose 1
  let total_outcomes : ℕ := total_colors.choose selected_masks
  (favorable_outcomes : ℚ) / total_outcomes = 3 / 10 := by
sorry

end NUMINAMATH_CALUDE_mask_selection_probability_l2184_218460


namespace NUMINAMATH_CALUDE_inequality_solution_implies_a_value_l2184_218440

/-- Given that the solution set of the inequality (ax)/(x-1) > 1 is (1, 2), prove that a = 1/2 --/
theorem inequality_solution_implies_a_value (a : ℝ) :
  (∀ x : ℝ, (1 < x ∧ x < 2) ↔ (a * x) / (x - 1) > 1) →
  a = 1/2 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_implies_a_value_l2184_218440


namespace NUMINAMATH_CALUDE_solve_for_r_l2184_218482

theorem solve_for_r (k r : ℝ) (h1 : 5 = k * 3^r) (h2 : 45 = k * 9^r) : r = 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_r_l2184_218482


namespace NUMINAMATH_CALUDE_train_speed_calculation_l2184_218409

/-- Proves that given a train journey with an original time of 50 minutes and a reduced time of 40 minutes
    at a speed of 60 km/h, the original average speed of the train is 48 km/h. -/
theorem train_speed_calculation (distance : ℝ) (original_time : ℝ) (reduced_time : ℝ) (new_speed : ℝ) :
  original_time = 50 / 60 →
  reduced_time = 40 / 60 →
  new_speed = 60 →
  distance = new_speed * reduced_time →
  distance / original_time = 48 :=
by sorry

end NUMINAMATH_CALUDE_train_speed_calculation_l2184_218409


namespace NUMINAMATH_CALUDE_square_area_ratio_l2184_218421

theorem square_area_ratio : 
  let side_C : ℝ := 24
  let side_D : ℝ := 30
  let area_C := side_C ^ 2
  let area_D := side_D ^ 2
  area_C / area_D = 16 / 25 := by
  sorry

end NUMINAMATH_CALUDE_square_area_ratio_l2184_218421


namespace NUMINAMATH_CALUDE_petya_ice_cream_l2184_218434

theorem petya_ice_cream (ice_cream_cost : ℕ) (petya_money : ℕ) : 
  ice_cream_cost = 2000 →
  petya_money = 400^5 - 399^2 * (400^3 + 2 * 400^2 + 3 * 400 + 4) →
  petya_money < ice_cream_cost :=
by
  sorry

end NUMINAMATH_CALUDE_petya_ice_cream_l2184_218434
