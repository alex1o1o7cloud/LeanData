import Mathlib

namespace NUMINAMATH_CALUDE_hyperbola_vertices_distance_l136_13647

theorem hyperbola_vertices_distance (x y : ℝ) :
  (((x - 1)^2 / 16) - (y^2 / 25) = 1) →
  (∃ v₁ v₂ : ℝ, v₁ ≠ v₂ ∧ 
    (((v₁ - 1)^2 / 16) - (0^2 / 25) = 1) ∧
    (((v₂ - 1)^2 / 16) - (0^2 / 25) = 1) ∧
    |v₁ - v₂| = 8) :=
by
  sorry

end NUMINAMATH_CALUDE_hyperbola_vertices_distance_l136_13647


namespace NUMINAMATH_CALUDE_xy_inequality_l136_13667

theorem xy_inequality (x y : ℝ) (h : x^2 + y^2 ≤ 2) : x * y + 3 ≥ 2 * x + 2 * y := by
  sorry

end NUMINAMATH_CALUDE_xy_inequality_l136_13667


namespace NUMINAMATH_CALUDE_trig_identity_l136_13626

theorem trig_identity : Real.sin (47 * π / 180) * Real.cos (17 * π / 180) + 
                        Real.cos (47 * π / 180) * Real.cos (107 * π / 180) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l136_13626


namespace NUMINAMATH_CALUDE_intersection_P_Q_l136_13659

def P : Set ℝ := {1, 2, 3, 4}
def Q : Set ℝ := {x : ℝ | |x| ≤ 3}

theorem intersection_P_Q : P ∩ Q = {1, 2, 3} := by sorry

end NUMINAMATH_CALUDE_intersection_P_Q_l136_13659


namespace NUMINAMATH_CALUDE_no_divisible_by_19_l136_13677

def a (n : ℕ) : ℤ := 9 * 10^n + 11

theorem no_divisible_by_19 : ∀ k : ℕ, k < 3050 → ¬(19 ∣ a k) := by
  sorry

end NUMINAMATH_CALUDE_no_divisible_by_19_l136_13677


namespace NUMINAMATH_CALUDE_difference_of_squares_division_l136_13600

theorem difference_of_squares_division : (324^2 - 300^2) / 24 = 624 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_division_l136_13600


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l136_13631

-- Problem 1
theorem problem_1 (a b : ℝ) : 4 * a^4 * b^3 / ((-2 * a * b)^2) = a^2 * b :=
by sorry

-- Problem 2
theorem problem_2 (x y : ℝ) : (3*x - y)^2 - (3*x + 2*y) * (3*x - 2*y) = 5*y^2 - 6*x*y :=
by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l136_13631


namespace NUMINAMATH_CALUDE_square_diagonal_l136_13606

theorem square_diagonal (perimeter : ℝ) (h : perimeter = 28) :
  let side := perimeter / 4
  let diagonal := Real.sqrt (2 * side ^ 2)
  diagonal = 7 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_square_diagonal_l136_13606


namespace NUMINAMATH_CALUDE_solve_for_b_l136_13633

theorem solve_for_b (m a b c k : ℝ) (h : m = (c^2 * a * b) / (a - k * b)) : 
  b = m * a / (c^2 * a + m * k) := by
  sorry

end NUMINAMATH_CALUDE_solve_for_b_l136_13633


namespace NUMINAMATH_CALUDE_product_inequality_l136_13669

theorem product_inequality (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (h_prod : a * b * c * d = 1) :
  (a^4 + b^4) / (a^2 + b^2) + (b^4 + c^4) / (b^2 + c^2) + 
  (c^4 + d^4) / (c^2 + d^2) + (d^4 + a^4) / (d^2 + a^2) ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_product_inequality_l136_13669


namespace NUMINAMATH_CALUDE_cube_side_length_l136_13698

theorem cube_side_length (volume_submerged_min : ℝ) (volume_submerged_max : ℝ)
  (density_ratio : ℝ) (volume_above_min : ℝ) (volume_above_max : ℝ) :
  volume_submerged_min = 0.58 →
  volume_submerged_max = 0.87 →
  density_ratio = 0.95 →
  volume_above_min = 10 →
  volume_above_max = 29 →
  ∃ (s : ℕ), s = 4 ∧
    (volume_submerged_min * s^3 ≤ density_ratio * s^3) ∧
    (density_ratio * s^3 ≤ volume_submerged_max * s^3) ∧
    (volume_above_min ≤ s^3 - volume_submerged_max * s^3) ∧
    (s^3 - volume_submerged_min * s^3 ≤ volume_above_max) :=
by sorry

end NUMINAMATH_CALUDE_cube_side_length_l136_13698


namespace NUMINAMATH_CALUDE_sum_reciprocals_equals_two_l136_13664

theorem sum_reciprocals_equals_two
  (a b c d : ℝ)
  (ω : ℂ)
  (ha : a ≠ -1)
  (hb : b ≠ -1)
  (hc : c ≠ -1)
  (hd : d ≠ -1)
  (hω1 : ω^4 = 1)
  (hω2 : ω ≠ 1)
  (h : 1 / (a + ω) + 1 / (b + ω) + 1 / (c + ω) + 1 / (d + ω) = 2 / ω^2) :
  1 / (a + 1) + 1 / (b + 1) + 1 / (c + 1) + 1 / (d + 1) = 2 :=
by sorry

end NUMINAMATH_CALUDE_sum_reciprocals_equals_two_l136_13664


namespace NUMINAMATH_CALUDE_factorial_500_properties_l136_13690

/-- The number of trailing zeroes in n! -/
def trailingZeroes (n : ℕ) : ℕ := sorry

/-- The highest power of 3 that divides n! -/
def highestPowerOfThree (n : ℕ) : ℕ := sorry

/-- Theorem about 500! -/
theorem factorial_500_properties :
  (trailingZeroes 500 = 124) ∧ (highestPowerOfThree 500 = 247) := by sorry

end NUMINAMATH_CALUDE_factorial_500_properties_l136_13690


namespace NUMINAMATH_CALUDE_fish_brought_home_l136_13683

/-- The number of fish Kendra caught -/
def kendras_catch : ℕ := 30

/-- The number of fish Ken released -/
def ken_released : ℕ := 3

/-- The number of fish Ken caught -/
def kens_catch : ℕ := 2 * kendras_catch

/-- The number of fish Ken brought home -/
def ken_brought_home : ℕ := kens_catch - ken_released

/-- The total number of fish brought home by Ken and Kendra -/
def total_brought_home : ℕ := ken_brought_home + kendras_catch

theorem fish_brought_home :
  total_brought_home = 87 :=
by sorry

end NUMINAMATH_CALUDE_fish_brought_home_l136_13683


namespace NUMINAMATH_CALUDE_fence_bricks_l136_13675

/-- Calculates the number of bricks needed for a rectangular fence --/
def bricks_needed (length width height depth : ℕ) : ℕ :=
  4 * length * width * depth

theorem fence_bricks :
  bricks_needed 20 5 2 = 800 := by
  sorry

end NUMINAMATH_CALUDE_fence_bricks_l136_13675


namespace NUMINAMATH_CALUDE_vinnie_saturday_words_l136_13680

/-- The number of words Vinnie wrote on Saturday -/
def saturday_words : ℕ := sorry

/-- The word limit -/
def word_limit : ℕ := 1000

/-- The number of words Vinnie wrote on Sunday -/
def sunday_words : ℕ := 650

/-- The number of words Vinnie exceeded the limit by -/
def excess_words : ℕ := 100

/-- Theorem stating that Vinnie wrote 450 words on Saturday -/
theorem vinnie_saturday_words :
  saturday_words = 450 ∧
  saturday_words + sunday_words = word_limit + excess_words :=
sorry

end NUMINAMATH_CALUDE_vinnie_saturday_words_l136_13680


namespace NUMINAMATH_CALUDE_rancher_unique_solution_l136_13611

/-- Represents the solution to the rancher's problem -/
structure RancherSolution where
  steers : ℕ
  cows : ℕ

/-- Checks if a given solution satisfies all conditions of the rancher's problem -/
def is_valid_solution (s : RancherSolution) : Prop :=
  s.steers > 0 ∧ 
  s.cows > 0 ∧ 
  30 * s.steers + 25 * s.cows = 1200

/-- Theorem stating that (5, 42) is the only valid solution to the rancher's problem -/
theorem rancher_unique_solution : 
  ∀ s : RancherSolution, is_valid_solution s ↔ s.steers = 5 ∧ s.cows = 42 := by
  sorry

#check rancher_unique_solution

end NUMINAMATH_CALUDE_rancher_unique_solution_l136_13611


namespace NUMINAMATH_CALUDE_infinite_logarithm_equation_l136_13668

theorem infinite_logarithm_equation : ∃! x : ℝ, x > 0 ∧ 2^x = x + 64 := by
  sorry

end NUMINAMATH_CALUDE_infinite_logarithm_equation_l136_13668


namespace NUMINAMATH_CALUDE_meena_baked_five_dozens_l136_13672

/-- The number of cookies in a dozen -/
def cookies_per_dozen : ℕ := 12

/-- The number of dozens of cookies sold to Mr. Stone -/
def dozens_sold_to_stone : ℕ := 2

/-- The number of cookies bought by Brock -/
def cookies_bought_by_brock : ℕ := 7

/-- The number of cookies Meena has left -/
def cookies_left : ℕ := 15

/-- Theorem: Meena baked 5 dozens of cookies initially -/
theorem meena_baked_five_dozens :
  let cookies_sold_to_stone := dozens_sold_to_stone * cookies_per_dozen
  let cookies_bought_by_katy := 2 * cookies_bought_by_brock
  let total_cookies_sold := cookies_sold_to_stone + cookies_bought_by_brock + cookies_bought_by_katy
  let total_cookies := total_cookies_sold + cookies_left
  total_cookies / cookies_per_dozen = 5 := by
  sorry

end NUMINAMATH_CALUDE_meena_baked_five_dozens_l136_13672


namespace NUMINAMATH_CALUDE_phone_number_combinations_l136_13637

def first_four_digits : ℕ := 12

def fifth_digit_options : ℕ := 2

def sixth_digit_options : ℕ := 10

theorem phone_number_combinations : 
  first_four_digits * fifth_digit_options * sixth_digit_options = 240 := by
  sorry

end NUMINAMATH_CALUDE_phone_number_combinations_l136_13637


namespace NUMINAMATH_CALUDE_sticker_distribution_l136_13684

/-- The number of ways to distribute n identical objects into k distinct containers -/
def distribute (n : ℕ) (k : ℕ) : ℕ := sorry

/-- The number of stickers -/
def num_stickers : ℕ := 10

/-- The number of sheets of paper -/
def num_sheets : ℕ := 5

theorem sticker_distribution :
  distribute num_stickers num_sheets = 29 := by sorry

end NUMINAMATH_CALUDE_sticker_distribution_l136_13684


namespace NUMINAMATH_CALUDE_equal_probability_implies_g_equals_5_l136_13608

-- Define the number of marbles in each bag
def redMarbles1 : ℕ := 2
def blueMarbles1 : ℕ := 2
def redMarbles2 : ℕ := 2
def blueMarbles2 : ℕ := 2

-- Define the probability function for bag 1
def prob1 : ℚ := (redMarbles1 * (redMarbles1 - 1) + blueMarbles1 * (blueMarbles1 - 1)) / 
              ((redMarbles1 + blueMarbles1) * (redMarbles1 + blueMarbles1 - 1))

-- Define the probability function for bag 2
def prob2 (g : ℕ) : ℚ := (redMarbles2 * (redMarbles2 - 1) + blueMarbles2 * (blueMarbles2 - 1) + g * (g - 1)) / 
                       ((redMarbles2 + blueMarbles2 + g) * (redMarbles2 + blueMarbles2 + g - 1))

-- Theorem statement
theorem equal_probability_implies_g_equals_5 :
  ∃ (g : ℕ), g > 0 ∧ prob1 = prob2 g → g = 5 :=
sorry

end NUMINAMATH_CALUDE_equal_probability_implies_g_equals_5_l136_13608


namespace NUMINAMATH_CALUDE_perfect_square_arrangement_l136_13678

/-- A function that checks if a number is a perfect square -/
def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

/-- A function that represents a permutation of numbers from 1 to n -/
def permutation (n : ℕ) := Fin n → Fin n

/-- A property that checks if a permutation satisfies the perfect square sum condition -/
def valid_permutation (n : ℕ) (p : permutation n) : Prop :=
  ∀ i : Fin n, is_perfect_square (i.val + 1 + (p i).val + 1)

theorem perfect_square_arrangement :
  (∃ p : permutation 9, valid_permutation 9 p) ∧
  (¬ ∃ p : permutation 11, valid_permutation 11 p) ∧
  (∃ p : permutation 1996, valid_permutation 1996 p) :=
sorry

end NUMINAMATH_CALUDE_perfect_square_arrangement_l136_13678


namespace NUMINAMATH_CALUDE_group_size_proof_l136_13627

/-- The number of men in a group where:
    1) The average age increases by 1 year
    2) Two men aged 21 and 23 are replaced by two men with an average age of 32 -/
def number_of_men : ℕ := 20

theorem group_size_proof :
  let original_average : ℝ := number_of_men
  let new_average : ℝ := original_average + 1
  let replaced_sum : ℝ := 21 + 23
  let new_sum : ℝ := 2 * 32
  number_of_men * original_average + new_sum - replaced_sum = number_of_men * new_average :=
by sorry

end NUMINAMATH_CALUDE_group_size_proof_l136_13627


namespace NUMINAMATH_CALUDE_plant_supplier_remaining_money_l136_13635

/-- Calculates the remaining money for a plant supplier after sales and expenses. -/
theorem plant_supplier_remaining_money
  (orchid_price : ℕ)
  (orchid_quantity : ℕ)
  (money_plant_price : ℕ)
  (money_plant_quantity : ℕ)
  (worker_pay : ℕ)
  (worker_count : ℕ)
  (pot_cost : ℕ)
  (h1 : orchid_price = 50)
  (h2 : orchid_quantity = 20)
  (h3 : money_plant_price = 25)
  (h4 : money_plant_quantity = 15)
  (h5 : worker_pay = 40)
  (h6 : worker_count = 2)
  (h7 : pot_cost = 150) :
  (orchid_price * orchid_quantity + money_plant_price * money_plant_quantity) -
  (worker_pay * worker_count + pot_cost) = 1145 := by
  sorry

end NUMINAMATH_CALUDE_plant_supplier_remaining_money_l136_13635


namespace NUMINAMATH_CALUDE_jason_remaining_seashells_l136_13671

def initial_seashells : ℕ := 49
def seashells_given_away : ℕ := 13

theorem jason_remaining_seashells :
  initial_seashells - seashells_given_away = 36 :=
by sorry

end NUMINAMATH_CALUDE_jason_remaining_seashells_l136_13671


namespace NUMINAMATH_CALUDE_cos_six_arccos_one_fourth_l136_13641

theorem cos_six_arccos_one_fourth : 
  Real.cos (6 * Real.arccos (1/4)) = -7/128 := by
  sorry

end NUMINAMATH_CALUDE_cos_six_arccos_one_fourth_l136_13641


namespace NUMINAMATH_CALUDE_bullet_problem_l136_13643

theorem bullet_problem :
  ∀ (initial_bullets : ℕ),
    (5 * (initial_bullets - 4) = initial_bullets) →
    initial_bullets = 5 := by
  sorry

end NUMINAMATH_CALUDE_bullet_problem_l136_13643


namespace NUMINAMATH_CALUDE_fraction_sum_integer_l136_13653

theorem fraction_sum_integer (a b c : ℤ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) 
  (h_sum : ∃ n : ℤ, (a * b) / c + (b * c) / a + (c * a) / b = n) : 
  (∃ n1 : ℤ, (a * b) / c = n1) ∧ (∃ n2 : ℤ, (b * c) / a = n2) ∧ (∃ n3 : ℤ, (c * a) / b = n3) :=
sorry

end NUMINAMATH_CALUDE_fraction_sum_integer_l136_13653


namespace NUMINAMATH_CALUDE_gcd_360_504_l136_13673

theorem gcd_360_504 : Nat.gcd 360 504 = 72 := by sorry

end NUMINAMATH_CALUDE_gcd_360_504_l136_13673


namespace NUMINAMATH_CALUDE_system_solution_existence_l136_13674

theorem system_solution_existence (a b : ℤ) :
  (∃ x y : ℝ, ⌊x⌋ + 2 * y = a ∧ ⌊y⌋ + 2 * x = b) ↔
  (a + b) % 3 = 0 ∨ (a + b) % 3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_existence_l136_13674


namespace NUMINAMATH_CALUDE_distance_from_origin_l136_13636

theorem distance_from_origin (a : ℝ) : |a| = 4 → a = 4 ∨ a = -4 := by sorry

end NUMINAMATH_CALUDE_distance_from_origin_l136_13636


namespace NUMINAMATH_CALUDE_simplified_expression_equals_one_l136_13622

theorem simplified_expression_equals_one (a : ℚ) (h : a = 1/2) :
  (1 / (a + 2) + 1 / (a - 2)) / (1 / (a^2 - 4)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_simplified_expression_equals_one_l136_13622


namespace NUMINAMATH_CALUDE_john_remaining_money_l136_13623

def trip_finances (initial_amount spent_amount remaining_amount : ℕ) : Prop :=
  (initial_amount = 1600) ∧
  (remaining_amount = spent_amount - 600) ∧
  (remaining_amount = initial_amount - spent_amount)

theorem john_remaining_money :
  ∃ (spent_amount remaining_amount : ℕ),
    trip_finances 1600 spent_amount remaining_amount ∧
    remaining_amount = 500 :=
by
  sorry

end NUMINAMATH_CALUDE_john_remaining_money_l136_13623


namespace NUMINAMATH_CALUDE_total_rainfall_2004_l136_13682

/-- The average monthly rainfall in Mathborough in 2003 -/
def mathborough_2003 : ℝ := 41.5

/-- The increase in average monthly rainfall in Mathborough from 2003 to 2004 -/
def mathborough_increase : ℝ := 5

/-- The average monthly rainfall in Hightown in 2003 -/
def hightown_2003 : ℝ := 38

/-- The increase in average monthly rainfall in Hightown from 2003 to 2004 -/
def hightown_increase : ℝ := 3

/-- The number of months in a year -/
def months_in_year : ℕ := 12

theorem total_rainfall_2004 : 
  (mathborough_2003 + mathborough_increase) * months_in_year = 558 ∧
  (hightown_2003 + hightown_increase) * months_in_year = 492 := by
sorry

end NUMINAMATH_CALUDE_total_rainfall_2004_l136_13682


namespace NUMINAMATH_CALUDE_product_remainder_l136_13658

theorem product_remainder (a b : ℕ) (ha : a % 3 = 2) (hb : b % 3 = 2) : (a * b) % 3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_product_remainder_l136_13658


namespace NUMINAMATH_CALUDE_distance_from_origin_to_point_l136_13614

theorem distance_from_origin_to_point : Real.sqrt (12^2 + (-16)^2) = 20 := by
  sorry

end NUMINAMATH_CALUDE_distance_from_origin_to_point_l136_13614


namespace NUMINAMATH_CALUDE_max_servings_is_twelve_l136_13681

/-- Represents the number of servings that can be made from a given ingredient --/
def ServingsFromIngredient (available : ℕ) (required : ℕ) : ℕ :=
  (available * 4) / required

/-- Represents the recipe and available ingredients --/
structure SmoothieRecipe where
  bananas_required : ℕ
  yogurt_required : ℕ
  strawberries_required : ℕ
  bananas_available : ℕ
  yogurt_available : ℕ
  strawberries_available : ℕ

/-- Calculates the maximum number of servings that can be made --/
def MaxServings (recipe : SmoothieRecipe) : ℕ :=
  min (ServingsFromIngredient recipe.bananas_available recipe.bananas_required)
    (min (ServingsFromIngredient recipe.yogurt_available recipe.yogurt_required)
      (ServingsFromIngredient recipe.strawberries_available recipe.strawberries_required))

theorem max_servings_is_twelve :
  ∀ (recipe : SmoothieRecipe),
    recipe.bananas_required = 3 →
    recipe.yogurt_required = 2 →
    recipe.strawberries_required = 1 →
    recipe.bananas_available = 9 →
    recipe.yogurt_available = 10 →
    recipe.strawberries_available = 3 →
    MaxServings recipe = 12 := by
  sorry

end NUMINAMATH_CALUDE_max_servings_is_twelve_l136_13681


namespace NUMINAMATH_CALUDE_spinner_probability_l136_13649

theorem spinner_probability (pA pB pC pD pE : ℚ) : 
  pA = 3/8 →
  pB = 1/8 →
  pC = pD →
  pC = pE →
  pA + pB + pC + pD + pE = 1 →
  pC = 1/6 := by
sorry

end NUMINAMATH_CALUDE_spinner_probability_l136_13649


namespace NUMINAMATH_CALUDE_polynomial_perfect_square_l136_13604

/-- The polynomial function P(x) = x^4 + 2x^3 - 2x^2 - 4x - 5 -/
def P (x : ℝ) : ℝ := x^4 + 2*x^3 - 2*x^2 - 4*x - 5

/-- A function is a perfect square if there exists a real function g such that f(x) = g(x)^2 for all x -/
def is_perfect_square (f : ℝ → ℝ) : Prop :=
  ∃ g : ℝ → ℝ, ∀ x, f x = (g x)^2

theorem polynomial_perfect_square :
  ∀ x : ℝ, is_perfect_square P ↔ (x = 3 ∨ x = -3) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_perfect_square_l136_13604


namespace NUMINAMATH_CALUDE_simplify_expression_l136_13616

theorem simplify_expression :
  ∀ x : ℝ, x > 0 →
  (3 * Real.sqrt 10) / (Real.sqrt 3 + Real.sqrt 4 + Real.sqrt 7) =
  Real.sqrt 3 + Real.sqrt 4 - Real.sqrt 7 :=
by sorry

end NUMINAMATH_CALUDE_simplify_expression_l136_13616


namespace NUMINAMATH_CALUDE_pairing_count_l136_13603

/-- The number of bowls -/
def num_bowls : ℕ := 4

/-- The number of glasses -/
def num_glasses : ℕ := 5

/-- The total number of possible pairings -/
def total_pairings : ℕ := num_bowls * num_glasses

theorem pairing_count : total_pairings = 20 := by sorry

end NUMINAMATH_CALUDE_pairing_count_l136_13603


namespace NUMINAMATH_CALUDE_simplest_common_denominator_l136_13632

variable (x : ℝ)

theorem simplest_common_denominator :
  ∃ (d : ℝ), d = x * (x + 1) * (x - 1) ∧
  (∃ (a b : ℝ), a / (x^2 - 1) + b / (x^2 + x) = (a * (x^2 + x) + b * (x^2 - 1)) / d) ∧
  (∀ (d' : ℝ), (∃ (a' b' : ℝ), a' / (x^2 - 1) + b' / (x^2 + x) = (a' * (x^2 + x) + b' * (x^2 - 1)) / d') →
    d ∣ d') :=
sorry

end NUMINAMATH_CALUDE_simplest_common_denominator_l136_13632


namespace NUMINAMATH_CALUDE_right_handed_players_count_l136_13624

theorem right_handed_players_count (total_players throwers : ℕ) 
  (h1 : total_players = 70)
  (h2 : throwers = 37)
  (h3 : throwers ≤ total_players)
  (h4 : (total_players - throwers) % 3 = 0) :
  throwers + 2 * ((total_players - throwers) / 3) = 59 := by
sorry

end NUMINAMATH_CALUDE_right_handed_players_count_l136_13624


namespace NUMINAMATH_CALUDE_flock_size_l136_13634

/-- Represents the number of sheep in a flock -/
structure Flock :=
  (rams : ℕ)
  (ewes : ℕ)

/-- The ratio of rams to ewes after one ram runs away -/
def ratio_after_ram_leaves (f : Flock) : ℚ :=
  (f.rams - 1 : ℚ) / f.ewes

/-- The ratio of rams to ewes after the ram returns and one ewe runs away -/
def ratio_after_ewe_leaves (f : Flock) : ℚ :=
  (f.rams : ℚ) / (f.ewes - 1)

/-- The theorem stating the total number of sheep in the flock -/
theorem flock_size (f : Flock) :
  (ratio_after_ram_leaves f = 7/5) →
  (ratio_after_ewe_leaves f = 5/3) →
  f.rams + f.ewes = 25 := by
  sorry

end NUMINAMATH_CALUDE_flock_size_l136_13634


namespace NUMINAMATH_CALUDE_train_length_calculation_l136_13689

theorem train_length_calculation (train_speed : ℝ) (bridge_length : ℝ) (passing_time : ℝ) : 
  train_speed = 90 ∧ bridge_length = 140 ∧ passing_time = 20 → 
  (train_speed * 1000 / 3600) * passing_time - bridge_length = 360 := by
  sorry

#check train_length_calculation

end NUMINAMATH_CALUDE_train_length_calculation_l136_13689


namespace NUMINAMATH_CALUDE_secant_minimum_value_l136_13615

/-- The secant function -/
noncomputable def sec (x : ℝ) : ℝ := 1 / Real.cos x

/-- The function y = a * sec(bx + c) -/
noncomputable def f (a b c x : ℝ) : ℝ := a * sec (b * x + c)

theorem secant_minimum_value (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (∀ x : ℝ, f a b c x > 0 → f a b c x ≥ 3) →
  (∃ x : ℝ, f a b c x = 3) →
  a = 3 :=
sorry

end NUMINAMATH_CALUDE_secant_minimum_value_l136_13615


namespace NUMINAMATH_CALUDE_N_eq_P_l136_13660

def N : Set ℚ := {x | ∃ n : ℤ, x = n / 2 - 1 / 3}
def P : Set ℚ := {x | ∃ p : ℤ, x = p / 2 + 1 / 6}

theorem N_eq_P : N = P := by sorry

end NUMINAMATH_CALUDE_N_eq_P_l136_13660


namespace NUMINAMATH_CALUDE_profit_discount_rate_l136_13620

/-- Proves that a 20% profit on a product with a purchase price of 200 yuan and a marked price of 300 yuan is achieved by selling at 80% of the marked price. -/
theorem profit_discount_rate (purchase_price marked_price : ℝ) 
  (h_purchase : purchase_price = 200)
  (h_marked : marked_price = 300)
  (profit_rate : ℝ) (h_profit : profit_rate = 0.2)
  (discount_rate : ℝ) :
  discount_rate * marked_price = purchase_price * (1 + profit_rate) →
  discount_rate = 0.8 := by
sorry

end NUMINAMATH_CALUDE_profit_discount_rate_l136_13620


namespace NUMINAMATH_CALUDE_fraction_sum_difference_l136_13695

theorem fraction_sum_difference : (7 : ℚ) / 12 + 8 / 15 - 2 / 5 = 43 / 60 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_difference_l136_13695


namespace NUMINAMATH_CALUDE_domain_of_f_l136_13625

def f (x : ℝ) : ℝ := (x - 3) ^ (1/3) + (5 - x) ^ (1/3) + (x + 1) ^ (1/2)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x ≥ -1} :=
sorry

end NUMINAMATH_CALUDE_domain_of_f_l136_13625


namespace NUMINAMATH_CALUDE_no_solution_iff_m_equals_five_l136_13618

theorem no_solution_iff_m_equals_five :
  ∀ m : ℝ, (∀ x : ℝ, x ≠ 5 ∧ x ≠ 8 → (x - 2) / (x - 5) ≠ (x - m) / (x - 8)) ↔ m = 5 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_iff_m_equals_five_l136_13618


namespace NUMINAMATH_CALUDE_similar_triangle_perimeter_l136_13697

/-- Represents a triangle with side lengths a, b, and c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if a triangle is isosceles -/
def Triangle.isIsosceles (t : Triangle) : Prop :=
  t.a = t.b ∨ t.b = t.c ∨ t.c = t.a

/-- Calculates the perimeter of a triangle -/
def Triangle.perimeter (t : Triangle) : ℝ :=
  t.a + t.b + t.c

/-- Checks if two triangles are similar -/
def Triangle.isSimilar (t1 t2 : Triangle) : Prop :=
  ∃ (k : ℝ), k > 0 ∧ t2.a = k * t1.a ∧ t2.b = k * t1.b ∧ t2.c = k * t1.c

theorem similar_triangle_perimeter (t1 t2 : Triangle) :
  t1.isIsosceles ∧
  t1.a = 18 ∧ t1.b = 18 ∧ t1.c = 12 ∧
  t2.isSimilar t1 ∧
  min t2.a (min t2.b t2.c) = 30 →
  t2.perimeter = 120 := by
sorry

end NUMINAMATH_CALUDE_similar_triangle_perimeter_l136_13697


namespace NUMINAMATH_CALUDE_correct_second_number_l136_13662

/-- Proves that the correct value of the second wrongly copied number is 27 --/
theorem correct_second_number (n : ℕ) (original_avg correct_avg : ℚ) 
  (first_error second_error : ℚ) (h1 : n = 10) (h2 : original_avg = 40.2) 
  (h3 : correct_avg = 40) (h4 : first_error = 16) (h5 : second_error = 13) : 
  ∃ (x : ℚ), n * correct_avg = n * original_avg - first_error - second_error + x ∧ x = 27 := by
  sorry

end NUMINAMATH_CALUDE_correct_second_number_l136_13662


namespace NUMINAMATH_CALUDE_not_p_sufficient_not_necessary_for_q_l136_13646

-- Define p and q
def p (x : ℝ) : Prop := x^2 > 4
def q (x : ℝ) : Prop := x ≤ 2

-- Define the negation of p
def not_p (x : ℝ) : Prop := ¬(p x)

-- Theorem stating that ¬p is a sufficient but not necessary condition for q
theorem not_p_sufficient_not_necessary_for_q :
  (∀ x : ℝ, not_p x → q x) ∧ 
  (∃ x : ℝ, q x ∧ ¬(not_p x)) :=
by sorry

end NUMINAMATH_CALUDE_not_p_sufficient_not_necessary_for_q_l136_13646


namespace NUMINAMATH_CALUDE_price_difference_in_cents_l136_13691

-- Define the list price and discounts
def list_price : ℚ := 5999 / 100  -- $59.99 represented as a rational number
def tech_bargains_discount : ℚ := 15  -- $15 off
def budget_bytes_discount_rate : ℚ := 30 / 100  -- 30% off

-- Calculate the sale prices
def tech_bargains_price : ℚ := list_price - tech_bargains_discount
def budget_bytes_price : ℚ := list_price * (1 - budget_bytes_discount_rate)

-- Find the cheaper price
def cheaper_price : ℚ := min tech_bargains_price budget_bytes_price
def more_expensive_price : ℚ := max tech_bargains_price budget_bytes_price

-- Define the theorem
theorem price_difference_in_cents : 
  (more_expensive_price - cheaper_price) * 100 = 300 := by
  sorry

end NUMINAMATH_CALUDE_price_difference_in_cents_l136_13691


namespace NUMINAMATH_CALUDE_certain_number_multiplication_l136_13679

theorem certain_number_multiplication (x : ℝ) : 37 - x = 24 → x * 24 = 312 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_multiplication_l136_13679


namespace NUMINAMATH_CALUDE_multiply_add_distribute_l136_13652

theorem multiply_add_distribute : 3.5 * 2.5 + 6.5 * 2.5 = 25 := by
  sorry

end NUMINAMATH_CALUDE_multiply_add_distribute_l136_13652


namespace NUMINAMATH_CALUDE_train_speed_l136_13619

/-- The speed of a train passing a jogger --/
theorem train_speed (jogger_speed : ℝ) (initial_lead : ℝ) (train_length : ℝ) (passing_time : ℝ) : 
  jogger_speed = 9 →
  initial_lead = 240 →
  train_length = 110 →
  passing_time = 35 →
  ∃ (train_speed : ℝ), train_speed = 45 := by
  sorry

#check train_speed

end NUMINAMATH_CALUDE_train_speed_l136_13619


namespace NUMINAMATH_CALUDE_final_walnut_count_l136_13665

/-- The number of walnuts left in the burrow after the squirrels' actions --/
def walnuts_left (initial_walnuts boy_walnuts boy_dropped girl_walnuts girl_eaten : ℕ) : ℕ :=
  initial_walnuts + (boy_walnuts - boy_dropped) + girl_walnuts - girl_eaten

/-- Theorem stating the final number of walnuts in the burrow --/
theorem final_walnut_count :
  walnuts_left 12 6 1 5 2 = 20 := by
  sorry

end NUMINAMATH_CALUDE_final_walnut_count_l136_13665


namespace NUMINAMATH_CALUDE_quadratic_roots_l136_13617

theorem quadratic_roots : 
  let f : ℝ → ℝ := λ x => x^2 - 2*x
  ∃ x₁ x₂ : ℝ, x₁ = 0 ∧ x₂ = 2 ∧ f x₁ = 0 ∧ f x₂ = 0 ∧ 
  ∀ x : ℝ, f x = 0 → x = x₁ ∨ x = x₂ :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_l136_13617


namespace NUMINAMATH_CALUDE_product_evaluation_l136_13609

theorem product_evaluation (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (a + b + c + d)⁻¹ * (a⁻¹ + b⁻¹ + c⁻¹ + d⁻¹) * (a*b + b*c + c*d + d*a + a*c + b*d)⁻¹ *
  ((a*b)⁻¹ + (b*c)⁻¹ + (c*d)⁻¹ + (d*a)⁻¹ + (a*c)⁻¹ + (b*d)⁻¹) = (a*a*b*b*c*c*d*d)⁻¹ :=
by sorry

end NUMINAMATH_CALUDE_product_evaluation_l136_13609


namespace NUMINAMATH_CALUDE_axisymmetric_shapes_l136_13645

-- Define the basic shapes
inductive Shape
  | Triangle
  | Parallelogram
  | Rectangle
  | Circle

-- Define the property of being axisymmetric
def is_axisymmetric (s : Shape) : Prop :=
  match s with
  | Shape.Rectangle => true
  | Shape.Circle => true
  | _ => false

-- Theorem statement
theorem axisymmetric_shapes :
  ∀ s : Shape, is_axisymmetric s ↔ (s = Shape.Rectangle ∨ s = Shape.Circle) :=
by sorry

end NUMINAMATH_CALUDE_axisymmetric_shapes_l136_13645


namespace NUMINAMATH_CALUDE_circle_equation_radius_l136_13654

/-- Given a circle with equation x^2 - 8x + y^2 + 10y + d = 0 and radius 5, prove that d = 16 -/
theorem circle_equation_radius (d : ℝ) : 
  (∀ x y : ℝ, x^2 - 8*x + y^2 + 10*y + d = 0 → (x - 4)^2 + (y + 5)^2 = 5^2) → 
  d = 16 := by
sorry

end NUMINAMATH_CALUDE_circle_equation_radius_l136_13654


namespace NUMINAMATH_CALUDE_max_value_cos_sin_l136_13630

theorem max_value_cos_sin (a b : ℝ) : 
  (∀ θ : ℝ, a * Real.cos θ + b * Real.sin θ ≤ Real.sqrt (a^2 + b^2)) ∧ 
  (∃ θ : ℝ, a * Real.cos θ + b * Real.sin θ = Real.sqrt (a^2 + b^2)) := by
  sorry

end NUMINAMATH_CALUDE_max_value_cos_sin_l136_13630


namespace NUMINAMATH_CALUDE_intersection_M_N_l136_13693

def M : Set ℕ := {1, 2, 4, 8}

def N : Set ℕ := {x : ℕ | x > 0 ∧ 4 % x = 0}

theorem intersection_M_N : M ∩ N = {1, 2, 4} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l136_13693


namespace NUMINAMATH_CALUDE_three_digit_number_operation_l136_13642

theorem three_digit_number_operation : ∀ a b c : ℕ,
  a ≥ 1 → a ≤ 9 →
  b ≥ 0 → b ≤ 9 →
  c ≥ 0 → c ≤ 9 →
  a = c + 3 →
  (100 * a + 10 * b + c) - ((100 * c + 10 * b + a) + 50) ≡ 7 [MOD 10] := by
  sorry

end NUMINAMATH_CALUDE_three_digit_number_operation_l136_13642


namespace NUMINAMATH_CALUDE_rachel_songs_theorem_l136_13696

/-- The number of songs in each of Rachel's albums -/
def album_songs : List Nat := [5, 6, 8, 10, 12, 14, 16, 7, 9, 11, 13, 15, 17, 4, 6, 8, 10, 12, 14, 3]

/-- The total number of songs Rachel bought -/
def total_songs : Nat := album_songs.sum

theorem rachel_songs_theorem : total_songs = 200 := by
  sorry

end NUMINAMATH_CALUDE_rachel_songs_theorem_l136_13696


namespace NUMINAMATH_CALUDE_cubic_equation_has_real_root_l136_13601

theorem cubic_equation_has_real_root (a b : ℝ) : ∃ x : ℝ, x^3 + a*x - b = 0 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_has_real_root_l136_13601


namespace NUMINAMATH_CALUDE_regular_pentagon_diagonal_angle_l136_13621

/-- A regular pentagon is a polygon with 5 equal sides and 5 equal angles -/
structure RegularPentagon where
  vertices : Fin 5 → ℝ × ℝ
  is_regular : sorry

/-- The measure of an angle in degrees -/
def angle_measure (a b c : ℝ × ℝ) : ℝ := sorry

theorem regular_pentagon_diagonal_angle 
  (ABCDE : RegularPentagon) 
  (h_interior : ∀ (i : Fin 5), angle_measure (ABCDE.vertices i) (ABCDE.vertices (i + 1)) (ABCDE.vertices (i + 2)) = 108) :
  angle_measure (ABCDE.vertices 0) (ABCDE.vertices 2) (ABCDE.vertices 1) = 36 := by
  sorry

end NUMINAMATH_CALUDE_regular_pentagon_diagonal_angle_l136_13621


namespace NUMINAMATH_CALUDE_initial_value_proof_l136_13629

theorem initial_value_proof (final_number : ℕ) (divisor : ℕ) (h1 : final_number = 859560) (h2 : divisor = 456) :
  ∃ (initial_value : ℕ) (added_number : ℕ),
    initial_value + added_number = final_number ∧
    final_number % divisor = 0 ∧
    initial_value = 859376 := by
  sorry

end NUMINAMATH_CALUDE_initial_value_proof_l136_13629


namespace NUMINAMATH_CALUDE_perfect_square_values_l136_13656

theorem perfect_square_values (p : ℤ) (n : ℚ) : 
  n = 16 * (10 : ℚ)^(-p) →
  -4 < p →
  p < 2 →
  (∃ (a b c : ℤ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    (16 * (10 : ℚ)^(-a) = (m : ℚ)^2 ∧
     16 * (10 : ℚ)^(-b) = (k : ℚ)^2 ∧
     16 * (10 : ℚ)^(-c) = (l : ℚ)^2) ∧
    (∀ (x : ℤ), x ≠ a ∧ x ≠ b ∧ x ≠ c →
      ¬∃ (y : ℚ), 16 * (10 : ℚ)^(-x) = y^2)) :=
by
  sorry

end NUMINAMATH_CALUDE_perfect_square_values_l136_13656


namespace NUMINAMATH_CALUDE_remainder_theorem_l136_13666

theorem remainder_theorem (x y u v : ℕ) (h1 : 0 < x) (h2 : 0 < y) 
  (h3 : x = u * y + v) (h4 : v < y) : 
  (x + 4 * u * y) % y = v := by
sorry

end NUMINAMATH_CALUDE_remainder_theorem_l136_13666


namespace NUMINAMATH_CALUDE_base_five_digits_of_1250_l136_13657

theorem base_five_digits_of_1250 : ∃ n : ℕ, n > 0 ∧ 5^(n-1) ≤ 1250 ∧ 1250 < 5^n ∧ n = 5 := by
  sorry

end NUMINAMATH_CALUDE_base_five_digits_of_1250_l136_13657


namespace NUMINAMATH_CALUDE_polygon_intersection_points_l136_13651

/-- The number of intersection points between two regular polygons inscribed in a circle -/
def intersectionPoints (n m : ℕ) : ℕ := 2 * min n m

/-- The total number of intersection points for four regular polygons -/
def totalIntersectionPoints (a b c d : ℕ) : ℕ :=
  intersectionPoints a b + intersectionPoints a c + intersectionPoints a d +
  intersectionPoints b c + intersectionPoints b d + intersectionPoints c d

theorem polygon_intersection_points :
  totalIntersectionPoints 6 7 8 9 = 80 := by
  sorry

#eval totalIntersectionPoints 6 7 8 9

end NUMINAMATH_CALUDE_polygon_intersection_points_l136_13651


namespace NUMINAMATH_CALUDE_skittles_left_l136_13699

def initial_skittles : ℕ := 250
def reduction_percentage : ℚ := 175 / 1000

theorem skittles_left :
  ⌊(initial_skittles : ℚ) - (initial_skittles : ℚ) * reduction_percentage⌋ = 206 :=
by
  sorry

end NUMINAMATH_CALUDE_skittles_left_l136_13699


namespace NUMINAMATH_CALUDE_square_side_length_l136_13650

theorem square_side_length (area : ℚ) (h : area = 9/16) : 
  ∃ (side : ℚ), side * side = area ∧ side = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l136_13650


namespace NUMINAMATH_CALUDE_no_primes_in_factorial_range_l136_13602

theorem no_primes_in_factorial_range (n : ℕ) (h : n > 1) :
  ∀ k ∈ Set.Ioo (n! + 1) (n! + n), ¬ Nat.Prime k := by
  sorry

end NUMINAMATH_CALUDE_no_primes_in_factorial_range_l136_13602


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l136_13638

theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = a n * q) →  -- a_n is a geometric sequence with common ratio q
  a 1 + a 2 = 3 →               -- a_1 + a_2 = 3
  a 3 + a 4 = 6 →               -- a_3 + a_4 = 6
  a 7 + a 8 = 24 :=             -- a_7 + a_8 = 24
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l136_13638


namespace NUMINAMATH_CALUDE_g_negative_six_l136_13613

def g (x : ℝ) : ℝ := 2 * x^7 - 3 * x^3 + 4 * x - 8

theorem g_negative_six (h : g 6 = 12) : g (-6) = -28 := by
  sorry

end NUMINAMATH_CALUDE_g_negative_six_l136_13613


namespace NUMINAMATH_CALUDE_rationalization_sum_l136_13644

theorem rationalization_sum (A B C D : ℤ) : 
  (7 / (3 + Real.sqrt 8) = (A * Real.sqrt B + C) / D) →
  (Nat.gcd A.natAbs C.natAbs = 1) →
  (Nat.gcd A.natAbs D.natAbs = 1) →
  (Nat.gcd C.natAbs D.natAbs = 1) →
  A + B + C + D = 23 := by
sorry

end NUMINAMATH_CALUDE_rationalization_sum_l136_13644


namespace NUMINAMATH_CALUDE_greatest_integer_radius_l136_13686

theorem greatest_integer_radius (r : ℕ) (A : ℝ) : 
  A < 75 * Real.pi → A = Real.pi * (r : ℝ)^2 → r ≤ 8 ∧ ∃ (s : ℕ), s = 8 ∧ Real.pi * (s : ℝ)^2 < 75 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_greatest_integer_radius_l136_13686


namespace NUMINAMATH_CALUDE_binomial_expansion_coeff_l136_13692

/-- The binomial coefficient -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The coefficient of x^3 in the expansion of (x^2 - m/x)^6 -/
def coeff_x3 (m : ℝ) : ℝ := (-1)^3 * binomial 6 3 * m^3

theorem binomial_expansion_coeff (m : ℝ) :
  coeff_x3 m = -160 → m = 2 := by sorry

end NUMINAMATH_CALUDE_binomial_expansion_coeff_l136_13692


namespace NUMINAMATH_CALUDE_total_price_of_hats_l136_13694

/-- Calculates the total price of hats given the conditions --/
theorem total_price_of_hats :
  let total_hats : ℕ := 85
  let green_hats : ℕ := 30
  let blue_hats : ℕ := total_hats - green_hats
  let price_green : ℕ := 7
  let price_blue : ℕ := 6
  (green_hats * price_green + blue_hats * price_blue) = 540 := by
  sorry

end NUMINAMATH_CALUDE_total_price_of_hats_l136_13694


namespace NUMINAMATH_CALUDE_geometric_progression_equality_l136_13610

theorem geometric_progression_equality (a r : ℝ) (n : ℕ) (hr : r ≠ 1) :
  let S : ℕ → ℝ := λ m ↦ a * (r^m - 1) / (r - 1)
  (S n) / (S (2*n) - S n) = (S (2*n) - S n) / (S (3*n) - S (2*n)) := by
  sorry

end NUMINAMATH_CALUDE_geometric_progression_equality_l136_13610


namespace NUMINAMATH_CALUDE_clock_hand_overlaps_l136_13605

/-- Represents a clock with hour and minute hands -/
structure Clock :=
  (hour_hand : ℝ)
  (minute_hand : ℝ)

/-- The number of hours in a day -/
def hours_per_day : ℕ := 24

/-- The number of overlaps in a 12-hour period -/
def overlaps_per_half_day : ℕ := 11

/-- Calculates the number of times the hour and minute hands overlap in a day -/
def overlaps_per_day (c : Clock) : ℕ :=
  2 * overlaps_per_half_day

/-- Theorem: The number of times the hour and minute hands of a clock overlap in a 24-hour day is 22 -/
theorem clock_hand_overlaps :
  ∀ c : Clock, overlaps_per_day c = 22 :=
by
  sorry

end NUMINAMATH_CALUDE_clock_hand_overlaps_l136_13605


namespace NUMINAMATH_CALUDE_tan_product_equals_neg_one_fifth_l136_13685

theorem tan_product_equals_neg_one_fifth 
  (α β : ℝ) (h : 2 * Real.cos (2 * α + β) - 3 * Real.cos β = 0) :
  Real.tan α * Real.tan (α + β) = -1/5 := by
  sorry

end NUMINAMATH_CALUDE_tan_product_equals_neg_one_fifth_l136_13685


namespace NUMINAMATH_CALUDE_max_value_of_fraction_l136_13661

theorem max_value_of_fraction (x : ℝ) : (4*x^2 + 8*x + 19) / (4*x^2 + 8*x + 5) ≤ 15 := by
  sorry

#check max_value_of_fraction

end NUMINAMATH_CALUDE_max_value_of_fraction_l136_13661


namespace NUMINAMATH_CALUDE_sum_of_powers_divisibility_l136_13639

theorem sum_of_powers_divisibility (n : ℕ+) :
  (((1:ℤ)^n.val + 2^n.val + 3^n.val + 4^n.val) % 5 = 0) ↔ (n.val % 4 ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_powers_divisibility_l136_13639


namespace NUMINAMATH_CALUDE_normal_distribution_problem_l136_13687

theorem normal_distribution_problem (σ μ : ℝ) (h1 : σ = 2) (h2 : μ = 55) :
  ∃ k : ℕ, k = 3 ∧ μ - k * σ > 48 ∧ ∀ m : ℕ, m > k → μ - m * σ ≤ 48 :=
sorry

end NUMINAMATH_CALUDE_normal_distribution_problem_l136_13687


namespace NUMINAMATH_CALUDE_subset_implies_m_values_l136_13663

def A (m : ℝ) : Set ℝ := {-1, 3, 2*m-1}
def B (m : ℝ) : Set ℝ := {3, m}

theorem subset_implies_m_values (m : ℝ) :
  B m ⊆ A m → m = 1 ∨ m = -1 := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_m_values_l136_13663


namespace NUMINAMATH_CALUDE_tim_weekly_fluid_intake_l136_13628

/-- Calculates Tim's weekly fluid intake in ounces -/
def weekly_fluid_intake : ℝ :=
  let water_bottles_per_day : ℝ := 2
  let water_quarts_per_bottle : ℝ := 1.5
  let orange_juice_oz_per_day : ℝ := 20
  let soda_liters_per_other_day : ℝ := 1.5
  let coffee_cups_per_week : ℝ := 4
  let quart_to_oz : ℝ := 32
  let liter_to_oz : ℝ := 33.814
  let cup_to_oz : ℝ := 8
  let days_per_week : ℝ := 7
  let soda_days_per_week : ℝ := 4

  let water_oz : ℝ := water_bottles_per_day * water_quarts_per_bottle * quart_to_oz * days_per_week
  let orange_juice_oz : ℝ := orange_juice_oz_per_day * days_per_week
  let soda_oz : ℝ := soda_liters_per_other_day * liter_to_oz * soda_days_per_week
  let coffee_oz : ℝ := coffee_cups_per_week * cup_to_oz

  water_oz + orange_juice_oz + soda_oz + coffee_oz

/-- Theorem stating Tim's weekly fluid intake -/
theorem tim_weekly_fluid_intake : weekly_fluid_intake = 1046.884 := by
  sorry

end NUMINAMATH_CALUDE_tim_weekly_fluid_intake_l136_13628


namespace NUMINAMATH_CALUDE_coprime_divisibility_theorem_l136_13648

theorem coprime_divisibility_theorem (a b : ℕ+) :
  (Nat.gcd (2 * a.val - 1) (2 * b.val + 1) = 1) →
  (a.val + b.val ∣ 4 * a.val * b.val + 1) →
  ∃ n : ℕ+, a.val = n.val ∧ b.val = n.val + 1 :=
by sorry

end NUMINAMATH_CALUDE_coprime_divisibility_theorem_l136_13648


namespace NUMINAMATH_CALUDE_isosceles_triangle_on_cube_l136_13688

-- Define a cube
def Cube : Type := Unit

-- Define a function to count the number of ways to choose 3 vertices from 8
def choose_3_from_8 : ℕ := 56

-- Define the number of isosceles triangles that can be formed on the cube
def isosceles_triangles_count : ℕ := 32

-- Define the probability of forming an isosceles triangle
def isosceles_triangle_probability : ℚ := 4/7

-- Theorem statement
theorem isosceles_triangle_on_cube :
  (isosceles_triangles_count : ℚ) / choose_3_from_8 = isosceles_triangle_probability :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_on_cube_l136_13688


namespace NUMINAMATH_CALUDE_equation_solution_l136_13640

theorem equation_solution : ∃ x : ℝ, 45 - (28 - (37 - (15 - x))) = 54 ∧ x = 15 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l136_13640


namespace NUMINAMATH_CALUDE_solid_yellow_marbles_percentage_l136_13612

theorem solid_yellow_marbles_percentage
  (total_marbles : ℝ)
  (solid_color_percentage : ℝ)
  (solid_color_not_yellow_percentage : ℝ)
  (h1 : solid_color_percentage = 90)
  (h2 : solid_color_not_yellow_percentage = 85)
  : (solid_color_percentage - solid_color_not_yellow_percentage) * total_marbles / 100 = 5 * total_marbles / 100 :=
by sorry

end NUMINAMATH_CALUDE_solid_yellow_marbles_percentage_l136_13612


namespace NUMINAMATH_CALUDE_credit_card_balance_l136_13670

/-- Represents the initial balance on a credit card -/
def initial_balance : ℝ := 170

/-- Represents the payment made on the credit card -/
def payment : ℝ := 50

/-- Represents the new balance after the payment -/
def new_balance : ℝ := 120

/-- Theorem stating that the initial balance minus the payment equals the new balance -/
theorem credit_card_balance :
  initial_balance - payment = new_balance := by sorry

end NUMINAMATH_CALUDE_credit_card_balance_l136_13670


namespace NUMINAMATH_CALUDE_unique_solution_characterization_l136_13607

-- Define the function representing the equation
def f (a : ℝ) (x : ℝ) : Prop :=
  2 * Real.log (x + 3) = Real.log (a * x)

-- Define the set of a values for which the equation has a unique solution
def uniqueSolutionSet : Set ℝ :=
  {a : ℝ | a < 0 ∨ a = 12}

-- Theorem statement
theorem unique_solution_characterization (a : ℝ) :
  (∃! x : ℝ, f a x) ↔ a ∈ uniqueSolutionSet :=
sorry

end NUMINAMATH_CALUDE_unique_solution_characterization_l136_13607


namespace NUMINAMATH_CALUDE_solve_for_y_l136_13655

theorem solve_for_y (x y : ℝ) (h1 : x + 2 * y = 10) (h2 : x = 4) : y = 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l136_13655


namespace NUMINAMATH_CALUDE_section_B_students_l136_13676

def section_A_students : ℕ := 50
def section_A_avg_weight : ℝ := 50
def section_B_avg_weight : ℝ := 70
def total_avg_weight : ℝ := 58.89

theorem section_B_students :
  ∃ x : ℕ, 
    (section_A_students * section_A_avg_weight + x * section_B_avg_weight) / (section_A_students + x) = total_avg_weight ∧
    x = 40 :=
by sorry

end NUMINAMATH_CALUDE_section_B_students_l136_13676
