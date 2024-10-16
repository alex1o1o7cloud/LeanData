import Mathlib

namespace NUMINAMATH_CALUDE_abc_inequality_l2767_276725

theorem abc_inequality (a b c : ℝ) (h1 : a ≥ b) (h2 : b ≥ c) (h3 : c > 0) (h4 : a + b + c = 3) :
  a * b^2 + b * c^2 + c * a^2 ≤ 27/8 := by
  sorry

end NUMINAMATH_CALUDE_abc_inequality_l2767_276725


namespace NUMINAMATH_CALUDE_min_odd_integers_l2767_276741

theorem min_odd_integers (a b c d e f : ℤ) 
  (sum_3 : a + b + c = 36)
  (sum_5 : a + b + c + d + e = 59)
  (sum_6 : a + b + c + d + e + f = 78) :
  ∃ (odds : Finset ℤ), odds ⊆ {a, b, c, d, e, f} ∧ 
    odds.card = 2 ∧
    (∀ x ∈ odds, Odd x) ∧
    (∀ (odds' : Finset ℤ), odds' ⊆ {a, b, c, d, e, f} ∧ 
      (∀ x ∈ odds', Odd x) → odds'.card ≥ 2) :=
by sorry

end NUMINAMATH_CALUDE_min_odd_integers_l2767_276741


namespace NUMINAMATH_CALUDE_parabola_single_intersection_l2767_276729

/-- A parabola y = x^2 + 4x + 5 - m intersects the x-axis at only one point if and only if m = 1 -/
theorem parabola_single_intersection (m : ℝ) : 
  (∃! x, x^2 + 4*x + 5 - m = 0) ↔ m = 1 := by
sorry

end NUMINAMATH_CALUDE_parabola_single_intersection_l2767_276729


namespace NUMINAMATH_CALUDE_collatz_3_reaches_421_cycle_l2767_276779

-- Define the operation for a single step
def collatzStep (n : ℕ) : ℕ :=
  if n % 2 = 0 then n / 2 else 3 * n + 1

-- Define the sequence of Collatz numbers starting from n
def collatzSequence (n : ℕ) : ℕ → ℕ
  | 0 => n
  | k + 1 => collatzStep (collatzSequence n k)

-- Theorem stating that the Collatz sequence starting from 3 eventually reaches the cycle 4, 2, 1
theorem collatz_3_reaches_421_cycle :
  ∃ k : ℕ, ∃ m : ℕ, m ≥ k ∧
    (collatzSequence 3 m = 4 ∧
     collatzSequence 3 (m + 1) = 2 ∧
     collatzSequence 3 (m + 2) = 1 ∧
     collatzSequence 3 (m + 3) = 4) :=
sorry

end NUMINAMATH_CALUDE_collatz_3_reaches_421_cycle_l2767_276779


namespace NUMINAMATH_CALUDE_households_with_bike_only_l2767_276793

theorem households_with_bike_only 
  (total : ℕ) 
  (without_car_or_bike : ℕ) 
  (with_both : ℕ) 
  (with_car : ℕ) 
  (h1 : total = 90)
  (h2 : without_car_or_bike = 11)
  (h3 : with_both = 18)
  (h4 : with_car = 44) :
  total - without_car_or_bike - (with_car - with_both) - with_both = 35 := by
  sorry

end NUMINAMATH_CALUDE_households_with_bike_only_l2767_276793


namespace NUMINAMATH_CALUDE_ellipse_hyperbola_relation_l2767_276770

/-- Given an ellipse and a hyperbola with coincident foci, prove that the semi-major axis of the ellipse is greater than that of the hyperbola, and the product of their eccentricities is greater than 1. -/
theorem ellipse_hyperbola_relation (m n : ℝ) (e₁ e₂ : ℝ) : 
  m > 1 →
  n > 0 →
  (∀ x y : ℝ, x^2 / m^2 + y^2 = 1 ↔ x^2 / n^2 - y^2 = 1) →
  e₁^2 = (m^2 - 1) / m^2 →
  e₂^2 = (n^2 + 1) / n^2 →
  m > n ∧ e₁ * e₂ > 1 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_hyperbola_relation_l2767_276770


namespace NUMINAMATH_CALUDE_consumption_wage_ratio_l2767_276773

/-- Linear regression equation parameters -/
def a : ℝ := 0.6
def b : ℝ := 1.5

/-- Average consumption per capita -/
def y : ℝ := 7.5

/-- Theorem stating the ratio of average consumption to average wage -/
theorem consumption_wage_ratio :
  ∃ x : ℝ, y = a * x + b ∧ y / x = 0.75 := by
  sorry

end NUMINAMATH_CALUDE_consumption_wage_ratio_l2767_276773


namespace NUMINAMATH_CALUDE_max_alpha_value_l2767_276786

-- Define the set of functions F
def F : Set (ℝ → ℝ) :=
  {f | ∀ x : ℝ, x ≥ 0 → f x ≥ 0 ∧ f (3 * x) ≥ f (f (2 * x)) + x}

-- State the theorem
theorem max_alpha_value :
  ∃ α : ℝ, α = 1/2 ∧
  (∀ β : ℝ, (∀ f ∈ F, ∀ x : ℝ, x ≥ 0 → f x ≥ β * x) → β ≤ α) ∧
  (∀ f ∈ F, ∀ x : ℝ, x ≥ 0 → f x ≥ α * x) :=
sorry

end NUMINAMATH_CALUDE_max_alpha_value_l2767_276786


namespace NUMINAMATH_CALUDE_game_not_fair_first_player_win_probability_limit_l2767_276777

/-- Represents a card game with n players and n cards. -/
structure CardGame where
  n : ℕ
  n_pos : 0 < n

/-- The probability of the first player winning the game. -/
noncomputable def firstPlayerWinProbability (game : CardGame) : ℝ :=
  Real.exp 1 / (game.n * (Real.exp 1 - 1))

/-- Theorem stating that the game is not fair for all players. -/
theorem game_not_fair (game : CardGame) : 
  ∃ (i j : ℕ), i ≠ j ∧ i ≤ game.n ∧ j ≤ game.n ∧ 
  (1 : ℝ) / game.n * (1 - (1 - 1 / game.n) ^ (game.n * (i - 1))) ≠ 
  (1 : ℝ) / game.n * (1 - (1 - 1 / game.n) ^ (game.n * (j - 1))) :=
sorry

/-- Theorem stating that the probability of the first player winning
    approaches e / (n * (e - 1)) as n becomes large. -/
theorem first_player_win_probability_limit (game : CardGame) :
  ∀ ε > 0, ∃ N, ∀ n ≥ N, game.n = n → 
  |firstPlayerWinProbability game - Real.exp 1 / (n * (Real.exp 1 - 1))| < ε :=
sorry

end NUMINAMATH_CALUDE_game_not_fair_first_player_win_probability_limit_l2767_276777


namespace NUMINAMATH_CALUDE_circular_seating_sum_l2767_276724

theorem circular_seating_sum (n : ℕ) (h : n ≥ 3) :
  (∀ (girl_sum : ℕ → ℕ) (boy_cards : ℕ → ℕ) (girl_cards : ℕ → ℕ),
    (∀ i : ℕ, i ∈ Finset.range n → 1 ≤ boy_cards i ∧ boy_cards i ≤ n) →
    (∀ i : ℕ, i ∈ Finset.range n → n + 1 ≤ girl_cards i ∧ girl_cards i ≤ 2*n) →
    (∀ i : ℕ, i ∈ Finset.range n → 
      girl_sum i = girl_cards i + boy_cards i + boy_cards ((i + 1) % n)) →
    (∀ i j : ℕ, i ∈ Finset.range n → j ∈ Finset.range n → girl_sum i = girl_sum j)) ↔
  Odd n :=
by sorry

end NUMINAMATH_CALUDE_circular_seating_sum_l2767_276724


namespace NUMINAMATH_CALUDE_smallest_next_divisor_l2767_276784

def is_odd (n : ℕ) : Prop := ∃ k, n = 2 * k + 1

theorem smallest_next_divisor (m : ℕ) (h1 : 1000 ≤ m ∧ m ≤ 9999) 
  (h2 : is_odd m) (h3 : m % 437 = 0) :
  ∃ d : ℕ, d > 437 ∧ m % d = 0 ∧ is_odd d ∧
  ∀ d' : ℕ, d' > 437 → m % d' = 0 → is_odd d' → d ≤ d' :=
sorry

end NUMINAMATH_CALUDE_smallest_next_divisor_l2767_276784


namespace NUMINAMATH_CALUDE_factoring_expression_l2767_276722

theorem factoring_expression (x y : ℝ) : 
  72 * x^4 * y^2 - 180 * x^8 * y^5 = 36 * x^4 * y^2 * (2 - 5 * x^4 * y^3) := by
  sorry

end NUMINAMATH_CALUDE_factoring_expression_l2767_276722


namespace NUMINAMATH_CALUDE_claudia_earnings_l2767_276740

def class_price : ℕ := 10
def saturday_attendance : ℕ := 20
def sunday_attendance : ℕ := saturday_attendance / 2

def total_earnings : ℕ := class_price * (saturday_attendance + sunday_attendance)

theorem claudia_earnings : total_earnings = 300 := by
  sorry

end NUMINAMATH_CALUDE_claudia_earnings_l2767_276740


namespace NUMINAMATH_CALUDE_martha_apples_theorem_l2767_276712

def apples_to_give_away (initial_apples : ℕ) (jane_apples : ℕ) (james_extra_apples : ℕ) (final_apples : ℕ) : ℕ :=
  initial_apples - jane_apples - (jane_apples + james_extra_apples) - final_apples

theorem martha_apples_theorem :
  apples_to_give_away 20 5 2 4 = 4 := by
sorry

end NUMINAMATH_CALUDE_martha_apples_theorem_l2767_276712


namespace NUMINAMATH_CALUDE_triangle_heights_existence_l2767_276769

/-- Check if a triangle with given heights exists -/
def triangle_exists (h₁ h₂ h₃ : ℝ) : Prop :=
  ∃ a b c : ℝ, 
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    a + b > c ∧ b + c > a ∧ c + a > b ∧
    h₁ * a = h₂ * b ∧ h₂ * b = h₃ * c

theorem triangle_heights_existence :
  (¬ triangle_exists 2 3 6) ∧ (triangle_exists 2 3 5) := by
  sorry


end NUMINAMATH_CALUDE_triangle_heights_existence_l2767_276769


namespace NUMINAMATH_CALUDE_quadratic_roots_when_positive_discriminant_l2767_276720

theorem quadratic_roots_when_positive_discriminant
  (a b c : ℝ) (h_a : a ≠ 0) (h_disc : b^2 - 4*a*c > 0) :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ a*x₁^2 + b*x₁ + c = 0 ∧ a*x₂^2 + b*x₂ + c = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_when_positive_discriminant_l2767_276720


namespace NUMINAMATH_CALUDE_tan_alpha_equals_sqrt_three_l2767_276704

theorem tan_alpha_equals_sqrt_three (α : Real) : 
  (∃ (x y : Real), x = -1 ∧ y = Real.sqrt 3 ∧ 
    Real.tan (2 * α) = y / x ∧ 
    2 * α ∈ Set.Icc 0 (2 * Real.pi)) →
  Real.tan α = Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_tan_alpha_equals_sqrt_three_l2767_276704


namespace NUMINAMATH_CALUDE_find_number_l2767_276744

theorem find_number (x : ℤ) : 
  (∃ q r : ℤ, 5 * (x + 3) = 8 * q + r ∧ q = 156 ∧ r = 2) → x = 247 := by
sorry

end NUMINAMATH_CALUDE_find_number_l2767_276744


namespace NUMINAMATH_CALUDE_equation_solution_l2767_276733

theorem equation_solution : ∃ x : ℝ, (5 + 3.6 * x = 2.1 * x - 25) ∧ (x = -20) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2767_276733


namespace NUMINAMATH_CALUDE_waiter_tips_fraction_l2767_276790

/-- Given a waiter's salary and tips, where the tips are 7/4 of the salary,
    prove that the fraction of total income from tips is 7/11. -/
theorem waiter_tips_fraction (salary : ℚ) (tips : ℚ) (total_income : ℚ) : 
  tips = (7 : ℚ) / 4 * salary →
  total_income = salary + tips →
  tips / total_income = (7 : ℚ) / 11 := by
  sorry

end NUMINAMATH_CALUDE_waiter_tips_fraction_l2767_276790


namespace NUMINAMATH_CALUDE_tom_peeled_24_potatoes_l2767_276796

/-- Represents the potato peeling scenario -/
structure PotatoPeeling where
  initialPile : ℕ
  maryRate : ℕ
  tomRate : ℕ
  maryAloneTime : ℕ

/-- Calculates the number of potatoes Tom peeled -/
def potatoesPeeledByTom (scenario : PotatoPeeling) : ℕ :=
  let potatoesPeeledByMaryAlone := scenario.maryRate * scenario.maryAloneTime
  let remainingPotatoes := scenario.initialPile - potatoesPeeledByMaryAlone
  let combinedRate := scenario.maryRate + scenario.tomRate
  let timeToFinish := remainingPotatoes / combinedRate
  scenario.tomRate * timeToFinish

/-- Theorem stating that Tom peeled 24 potatoes -/
theorem tom_peeled_24_potatoes :
  let scenario : PotatoPeeling := {
    initialPile := 60,
    maryRate := 4,
    tomRate := 6,
    maryAloneTime := 5
  }
  potatoesPeeledByTom scenario = 24 := by sorry

end NUMINAMATH_CALUDE_tom_peeled_24_potatoes_l2767_276796


namespace NUMINAMATH_CALUDE_remainder_divisibility_l2767_276764

theorem remainder_divisibility (N : ℤ) : 
  ∃ k : ℤ, N = 142 * k + 110 → ∃ m : ℤ, N = 14 * m + 8 := by
  sorry

end NUMINAMATH_CALUDE_remainder_divisibility_l2767_276764


namespace NUMINAMATH_CALUDE_volume_object_A_l2767_276756

/-- Represents the volume of an object in a cube-shaped fishbowl --/
def volume_object (side_length : ℝ) (height_with_object : ℝ) (height_without_object : ℝ) : ℝ :=
  side_length^2 * (height_with_object - height_without_object)

/-- The volume of object (A) in the fishbowl is 2800 cubic centimeters --/
theorem volume_object_A :
  let side_length : ℝ := 20
  let height_with_object : ℝ := 16
  let height_without_object : ℝ := 9
  volume_object side_length height_with_object height_without_object = 2800 := by
  sorry

end NUMINAMATH_CALUDE_volume_object_A_l2767_276756


namespace NUMINAMATH_CALUDE_wire_cutting_problem_l2767_276717

/-- The length of wire pieces that satisfies the given conditions -/
def wire_piece_length : ℕ := 83

theorem wire_cutting_problem (initial_length second_length : ℕ) 
  (h1 : initial_length = 1000)
  (h2 : second_length = 1070)
  (h3 : 12 * wire_piece_length ≤ initial_length)
  (h4 : 12 * wire_piece_length ≤ second_length)
  (h5 : ∀ x : ℕ, x > wire_piece_length → 12 * x > second_length) :
  wire_piece_length = 83 := by
  sorry

end NUMINAMATH_CALUDE_wire_cutting_problem_l2767_276717


namespace NUMINAMATH_CALUDE_prove_ball_size_ratio_l2767_276707

def ball_size_ratio (first_ball : ℝ) (second_ball : ℝ) (third_ball : ℝ) : Prop :=
  first_ball = second_ball / 2 ∧ 
  second_ball = 18 ∧ 
  third_ball = 27 ∧ 
  third_ball / first_ball = 3

theorem prove_ball_size_ratio : 
  ∃ (first_ball second_ball third_ball : ℝ), 
    ball_size_ratio first_ball second_ball third_ball :=
sorry

end NUMINAMATH_CALUDE_prove_ball_size_ratio_l2767_276707


namespace NUMINAMATH_CALUDE_replaced_girl_weight_l2767_276711

/-- Given a group of girls where replacing one with a heavier girl increases the average weight, 
    this theorem proves the weight of the replaced girl. -/
theorem replaced_girl_weight 
  (n : ℕ) 
  (initial_average : ℝ) 
  (new_girl_weight : ℝ) 
  (average_increase : ℝ) 
  (h1 : n = 10)
  (h2 : new_girl_weight = 100)
  (h3 : average_increase = 5) :
  initial_average * n + new_girl_weight - (initial_average * n + n * average_increase) = 50 :=
by
  sorry

#check replaced_girl_weight

end NUMINAMATH_CALUDE_replaced_girl_weight_l2767_276711


namespace NUMINAMATH_CALUDE_pirate_count_l2767_276798

/-- Represents the total number of pirates on the schooner -/
def total_pirates : ℕ := 30

/-- Represents the number of pirates who did not participate in the battle -/
def non_participants : ℕ := 10

/-- Represents the percentage of battle participants who lost an arm -/
def arm_loss_percentage : ℚ := 54 / 100

/-- Represents the percentage of battle participants who lost both an arm and a leg -/
def both_loss_percentage : ℚ := 34 / 100

/-- Represents the fraction of all pirates who lost a leg -/
def leg_loss_fraction : ℚ := 2 / 3

theorem pirate_count : 
  total_pirates = 30 ∧
  non_participants = 10 ∧
  arm_loss_percentage = 54 / 100 ∧
  both_loss_percentage = 34 / 100 ∧
  leg_loss_fraction = 2 / 3 →
  total_pirates = 30 :=
by sorry

end NUMINAMATH_CALUDE_pirate_count_l2767_276798


namespace NUMINAMATH_CALUDE_jeanine_pencils_proof_l2767_276761

/-- The number of pencils Jeanine bought initially -/
def jeanine_pencils : ℕ := 18

/-- The number of pencils Clare bought -/
def clare_pencils : ℕ := jeanine_pencils / 2

/-- The number of pencils Jeanine has after giving some to Abby -/
def jeanine_remaining_pencils : ℕ := (2 * jeanine_pencils) / 3

theorem jeanine_pencils_proof :
  (clare_pencils = jeanine_pencils / 2) ∧
  (jeanine_remaining_pencils = (2 * jeanine_pencils) / 3) ∧
  (jeanine_remaining_pencils = clare_pencils + 3) →
  jeanine_pencils = 18 := by
sorry

end NUMINAMATH_CALUDE_jeanine_pencils_proof_l2767_276761


namespace NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l2767_276751

theorem absolute_value_inequality_solution_set :
  {x : ℝ | |2*x - 5| ≤ 7} = {x : ℝ | -1 ≤ x ∧ x ≤ 6} := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l2767_276751


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l2767_276789

theorem complex_fraction_simplification : 
  (((10^4+324)*(22^4+324)*(34^4+324)*(46^4+324)*(58^4+324)) / 
   ((4^4+324)*(16^4+324)*(28^4+324)*(40^4+324)*(52^4+324))) = 373 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l2767_276789


namespace NUMINAMATH_CALUDE_multiply_six_and_mixed_number_l2767_276710

theorem multiply_six_and_mixed_number : 6 * (8 + 1/3) = 50 := by
  sorry

end NUMINAMATH_CALUDE_multiply_six_and_mixed_number_l2767_276710


namespace NUMINAMATH_CALUDE_range_of_a_l2767_276746

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, 2 * x^2 + (a - 1) * x + 1/2 > 0) → 
  -1 < a ∧ a < 3 := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l2767_276746


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2767_276774

theorem quadratic_inequality_solution_set (m : ℝ) (h : m > 1) :
  {x : ℝ | x^2 + (m - 1) * x - m ≥ 0} = {x : ℝ | x ≤ -m ∨ x ≥ 1} := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2767_276774


namespace NUMINAMATH_CALUDE_total_food_amount_l2767_276702

-- Define the number of boxes
def num_boxes : ℕ := 388

-- Define the amount of food per box in kilograms
def food_per_box : ℕ := 2

-- Theorem to prove the total amount of food
theorem total_food_amount : num_boxes * food_per_box = 776 := by
  sorry

end NUMINAMATH_CALUDE_total_food_amount_l2767_276702


namespace NUMINAMATH_CALUDE_difference_of_squares_l2767_276795

theorem difference_of_squares (x y : ℝ) (h1 : x + y = 10) (h2 : x - y = 19) : x^2 - y^2 = 190 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l2767_276795


namespace NUMINAMATH_CALUDE_total_temp_remaining_days_l2767_276754

/-- Calculates the total temperature of the remaining days in a week given specific conditions. -/
theorem total_temp_remaining_days 
  (avg_temp : ℝ) 
  (days_in_week : ℕ) 
  (temp_first_three : ℝ) 
  (days_first_three : ℕ) 
  (temp_thur_fri : ℝ) 
  (days_thur_fri : ℕ) :
  avg_temp = 60 ∧ 
  days_in_week = 7 ∧ 
  temp_first_three = 40 ∧ 
  days_first_three = 3 ∧ 
  temp_thur_fri = 80 ∧ 
  days_thur_fri = 2 →
  avg_temp * days_in_week - (temp_first_three * days_first_three + temp_thur_fri * days_thur_fri) = 140 :=
by sorry

end NUMINAMATH_CALUDE_total_temp_remaining_days_l2767_276754


namespace NUMINAMATH_CALUDE_rosy_fish_count_l2767_276703

/-- The number of fish Lilly has -/
def lillys_fish : ℕ := 10

/-- The total number of fish Lilly and Rosy have together -/
def total_fish : ℕ := 19

/-- The number of fish Rosy has -/
def rosys_fish : ℕ := total_fish - lillys_fish

theorem rosy_fish_count : rosys_fish = 9 := by
  sorry

end NUMINAMATH_CALUDE_rosy_fish_count_l2767_276703


namespace NUMINAMATH_CALUDE_coplanar_vectors_lambda_l2767_276706

/-- Given three vectors a, b, and c in R³, if they are coplanar and have specific coordinates,
    then the third coordinate of c is 65/7. -/
theorem coplanar_vectors_lambda (a b c : ℝ × ℝ × ℝ) :
  a = (2, -1, 3) →
  b = (-1, 4, -2) →
  c.1 = 7 ∧ c.2.1 = 5 →
  (∃ (p q : ℝ), c = p • a + q • b) →
  c.2.2 = 65 / 7 := by
  sorry

end NUMINAMATH_CALUDE_coplanar_vectors_lambda_l2767_276706


namespace NUMINAMATH_CALUDE_existence_of_special_permutation_sum_l2767_276780

/-- A function that checks if two numbers are permutations of each other's digits -/
def arePermutations (a b : ℕ) : Prop := sorry

/-- A function that checks if a number is five-digit -/
def isFiveDigit (n : ℕ) : Prop := 10000 ≤ n ∧ n < 100000

theorem existence_of_special_permutation_sum :
  ∃ (a b c : ℕ),
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    isFiveDigit a ∧ isFiveDigit b ∧ isFiveDigit c ∧
    arePermutations a b ∧ arePermutations b c ∧ arePermutations a c ∧
    b + c = 2 * a :=
sorry

end NUMINAMATH_CALUDE_existence_of_special_permutation_sum_l2767_276780


namespace NUMINAMATH_CALUDE_smallest_denominator_fraction_l2767_276743

-- Define the fraction type
structure Fraction where
  numerator : ℕ
  denominator : ℕ
  denom_pos : denominator > 0

-- Define the property of being in the open interval
def inOpenInterval (f : Fraction) : Prop :=
  47 / 245 < f.numerator / f.denominator ∧ f.numerator / f.denominator < 34 / 177

-- Define the property of having the smallest denominator
def hasSmallestDenominator (f : Fraction) : Prop :=
  ∀ g : Fraction, inOpenInterval g → f.denominator ≤ g.denominator

-- The main theorem
theorem smallest_denominator_fraction :
  ∃ f : Fraction, f.numerator = 19 ∧ f.denominator = 99 ∧
  inOpenInterval f ∧ hasSmallestDenominator f :=
sorry

end NUMINAMATH_CALUDE_smallest_denominator_fraction_l2767_276743


namespace NUMINAMATH_CALUDE_prove_c_minus_d_equals_negative_three_l2767_276748

-- Define the function g
noncomputable def g : ℝ → ℝ := sorry

-- Define c and d
noncomputable def c : ℝ := sorry
noncomputable def d : ℝ := sorry

-- State the theorem
theorem prove_c_minus_d_equals_negative_three :
  Function.Injective g ∧ g c = d ∧ g d = 5 → c - d = -3 := by sorry

end NUMINAMATH_CALUDE_prove_c_minus_d_equals_negative_three_l2767_276748


namespace NUMINAMATH_CALUDE_smallest_non_prime_digit_divisible_by_all_single_digit_primes_l2767_276700

def is_prime (n : ℕ) : Prop := sorry

def single_digit_primes : List ℕ := [2, 3, 5, 7]

def digits (n : ℕ) : List ℕ := sorry

theorem smallest_non_prime_digit_divisible_by_all_single_digit_primes :
  ∃ (N : ℕ),
    (∀ d ∈ digits N, ¬ is_prime d) ∧
    (∀ p ∈ single_digit_primes, N % p = 0) ∧
    (∀ m < N, ¬(∀ d ∈ digits m, ¬ is_prime d) ∨ ¬(∀ p ∈ single_digit_primes, m % p = 0)) ∧
    N = 840 :=
sorry

end NUMINAMATH_CALUDE_smallest_non_prime_digit_divisible_by_all_single_digit_primes_l2767_276700


namespace NUMINAMATH_CALUDE_time_to_burn_fifteen_coals_l2767_276781

/-- Represents the number of bags of coal used -/
def bags : ℕ := 3

/-- Represents the number of coals in each bag -/
def coals_per_bag : ℕ := 60

/-- Represents the total running time of the grill in minutes -/
def total_time : ℕ := 240

/-- Represents the number of coals that burn in one cycle -/
def coals_per_cycle : ℕ := 15

/-- Theorem stating that the time to burn 15 coals is 20 minutes -/
theorem time_to_burn_fifteen_coals :
  (total_time * coals_per_cycle) / (bags * coals_per_bag) = 20 := by
  sorry

end NUMINAMATH_CALUDE_time_to_burn_fifteen_coals_l2767_276781


namespace NUMINAMATH_CALUDE_cone_volume_l2767_276745

/-- Given a cone with base radius 1 and lateral surface area √5π, its volume is 2π/3 -/
theorem cone_volume (r h : ℝ) (lateral_area : ℝ) : 
  r = 1 → 
  lateral_area = Real.sqrt 5 * Real.pi → 
  2 * Real.pi * r * h = lateral_area → 
  (1/3) * Real.pi * r^2 * h = (2/3) * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_cone_volume_l2767_276745


namespace NUMINAMATH_CALUDE_additive_function_properties_l2767_276752

/-- A function satisfying f(x + y) = f(x) + f(y) for all real x and y -/
def AdditiveFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) = f x + f y

theorem additive_function_properties (f : ℝ → ℝ) (h : AdditiveFunction f) :
  (∀ x : ℝ, f (-x) = -f x) ∧ f 24 = -8 * f (-3) := by
  sorry

end NUMINAMATH_CALUDE_additive_function_properties_l2767_276752


namespace NUMINAMATH_CALUDE_cashier_money_value_l2767_276735

theorem cashier_money_value (total_bills : ℕ) (five_dollar_bills : ℕ) : 
  total_bills = 126 →
  five_dollar_bills = 84 →
  (total_bills - five_dollar_bills) * 10 + five_dollar_bills * 5 = 840 :=
by sorry

end NUMINAMATH_CALUDE_cashier_money_value_l2767_276735


namespace NUMINAMATH_CALUDE_expression_value_l2767_276792

theorem expression_value : 
  ∃ (m : ℕ), (3^1005 + 7^1006)^2 - (3^1005 - 7^1006)^2 = m * 10^1006 ∧ m = 280 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2767_276792


namespace NUMINAMATH_CALUDE_toms_deck_cost_l2767_276726

/-- Represents the cost of a deck of cards -/
def deck_cost (rare_count : ℕ) (uncommon_count : ℕ) (common_count : ℕ) 
              (rare_price : ℚ) (uncommon_price : ℚ) (common_price : ℚ) : ℚ :=
  rare_count * rare_price + uncommon_count * uncommon_price + common_count * common_price

/-- Theorem stating that the cost of Tom's deck is $32 -/
theorem toms_deck_cost : 
  deck_cost 19 11 30 1 (1/2) (1/4) = 32 := by
  sorry

end NUMINAMATH_CALUDE_toms_deck_cost_l2767_276726


namespace NUMINAMATH_CALUDE_target_line_is_correct_l2767_276715

/-- The line we want to prove is correct -/
def target_line (x y : ℝ) : Prop := y = -3 * x - 2

/-- The line perpendicular to our target line -/
def perpendicular_line (x y : ℝ) : Prop := 2 * x - 6 * y + 1 = 0

/-- The curve to which our target line is tangent -/
def curve (x : ℝ) : ℝ := x^3 + 3 * x^2 - 1

/-- Theorem stating that our target line is perpendicular to the given line
    and tangent to the given curve -/
theorem target_line_is_correct :
  (∀ x y : ℝ, perpendicular_line x y → 
    ∃ k : ℝ, k ≠ 0 ∧ (∀ x' y' : ℝ, target_line x' y' → 
      y' - y = k * (x' - x))) ∧ 
  (∃ x y : ℝ, target_line x y ∧ y = curve x ∧ 
    ∀ h : ℝ, h ≠ 0 → (curve (x + h) - curve x) / h ≠ -3) :=
sorry

end NUMINAMATH_CALUDE_target_line_is_correct_l2767_276715


namespace NUMINAMATH_CALUDE_dj_oldies_ratio_l2767_276783

/-- Represents the number of song requests for each genre and the total requests --/
structure SongRequests where
  total : Nat
  electropop : Nat
  dance : Nat
  rock : Nat
  oldies : Nat
  rap : Nat

/-- Calculates the number of DJ's choice songs --/
def djChoice (s : SongRequests) : Nat :=
  s.total - (s.electropop + s.rock + s.oldies + s.rap)

/-- Theorem stating the ratio of DJ's choice to oldies songs --/
theorem dj_oldies_ratio (s : SongRequests) : 
  s.total = 30 ∧ 
  s.electropop = s.total / 2 ∧ 
  s.dance = s.electropop / 3 ∧ 
  s.rock = 5 ∧ 
  s.oldies = s.rock - 3 ∧ 
  s.rap = 2 → 
  (djChoice s : Int) / s.oldies = 3 := by
  sorry

end NUMINAMATH_CALUDE_dj_oldies_ratio_l2767_276783


namespace NUMINAMATH_CALUDE_value_of_x_l2767_276755

theorem value_of_x (x y z : ℚ) : x = y / 3 → y = z / 4 → z = 48 → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_value_of_x_l2767_276755


namespace NUMINAMATH_CALUDE_company_growth_rate_l2767_276785

/-- Represents the yearly capital growth rate as a real number between 0 and 1 -/
def yearly_growth_rate : ℝ := sorry

/-- The initial loan amount in ten thousands of yuan -/
def initial_loan : ℝ := 200

/-- The loan duration in years -/
def loan_duration : ℕ := 2

/-- The annual interest rate as a real number between 0 and 1 -/
def interest_rate : ℝ := 0.08

/-- The surplus after repayment in ten thousands of yuan -/
def surplus : ℝ := 72

theorem company_growth_rate :
  (initial_loan * (1 + yearly_growth_rate) ^ loan_duration) =
  (initial_loan * (1 + interest_rate) ^ loan_duration + surplus) ∧
  yearly_growth_rate = 0.2 := by sorry

end NUMINAMATH_CALUDE_company_growth_rate_l2767_276785


namespace NUMINAMATH_CALUDE_quadratic_roots_property_l2767_276731

theorem quadratic_roots_property (p q : ℝ) : 
  (3 * p^2 - 5 * p - 14 = 0) → 
  (3 * q^2 - 5 * q - 14 = 0) → 
  p ≠ q →
  (3 * p^2 - 3 * q^2) * (p - q)⁻¹ = 5 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_property_l2767_276731


namespace NUMINAMATH_CALUDE_arithmetic_sequence_solution_l2767_276791

theorem arithmetic_sequence_solution (x : ℝ) (h1 : x ≠ 0) (h2 : x < 4) : 
  (⌊x⌋ + 1 - x + ⌊x⌋ = x + 1 - (⌊x⌋ + 1)) → 
  (x = 0.5 ∨ x = 1.5 ∨ x = 2.5 ∨ x = 3.5) := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_solution_l2767_276791


namespace NUMINAMATH_CALUDE_cubic_root_sum_squares_l2767_276742

theorem cubic_root_sum_squares (p q r : ℝ) (x : ℝ → ℝ) :
  (∀ t, x t = 0 ↔ t^3 - p*t^2 + q*t - r = 0) →
  ∃ r s t, (x r = 0 ∧ x s = 0 ∧ x t = 0) ∧ 
           (r^2 + s^2 + t^2 = p^2 - 2*q) :=
by sorry

end NUMINAMATH_CALUDE_cubic_root_sum_squares_l2767_276742


namespace NUMINAMATH_CALUDE_shirts_not_washed_l2767_276737

theorem shirts_not_washed 
  (short_sleeve : ℕ) 
  (long_sleeve : ℕ) 
  (washed : ℕ) 
  (h1 : short_sleeve = 9)
  (h2 : long_sleeve = 21)
  (h3 : washed = 29) : 
  short_sleeve + long_sleeve - washed = 1 := by
sorry

end NUMINAMATH_CALUDE_shirts_not_washed_l2767_276737


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l2767_276762

theorem right_triangle_hypotenuse (a b : ℝ) (h1 : a > 0) (h2 : b > 0) : 
  (1 / 3 * π * b * a^2 = 1280 * π) → (b / a = 3 / 4) → 
  Real.sqrt (a^2 + b^2) = 20 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l2767_276762


namespace NUMINAMATH_CALUDE_comic_book_stacking_l2767_276772

theorem comic_book_stacking (batman : ℕ) (xmen : ℕ) (calvin_hobbes : ℕ) :
  batman = 5 →
  xmen = 4 →
  calvin_hobbes = 3 →
  (Nat.factorial batman * Nat.factorial xmen * Nat.factorial calvin_hobbes) *
  Nat.factorial 3 = 103680 :=
by sorry

end NUMINAMATH_CALUDE_comic_book_stacking_l2767_276772


namespace NUMINAMATH_CALUDE_geometric_arithmetic_inequality_l2767_276799

/-- A positive geometric sequence -/
def is_positive_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (r : ℝ), r > 0 ∧ ∀ n, a (n + 1) = r * a n ∧ a n > 0

/-- An arithmetic sequence -/
def is_arithmetic_sequence (b : ℕ → ℝ) : Prop :=
  ∃ (d : ℝ), ∀ n, b (n + 1) = b n + d

theorem geometric_arithmetic_inequality 
  (a b : ℕ → ℝ) 
  (ha : is_positive_geometric_sequence a) 
  (hb : is_arithmetic_sequence b) 
  (h_eq : a 6 = b 7) : 
  a 3 + a 9 ≥ b 4 + b 10 := by
sorry

end NUMINAMATH_CALUDE_geometric_arithmetic_inequality_l2767_276799


namespace NUMINAMATH_CALUDE_regular_polygon_with_60_degree_exterior_angle_has_6_sides_l2767_276788

/-- The number of sides of a regular polygon with an exterior angle of 60 degrees is 6. -/
theorem regular_polygon_with_60_degree_exterior_angle_has_6_sides :
  ∀ (n : ℕ), n > 0 →
  (360 / n = 60) →
  n = 6 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_with_60_degree_exterior_angle_has_6_sides_l2767_276788


namespace NUMINAMATH_CALUDE_evaluate_expression_l2767_276771

theorem evaluate_expression : (1 / ((-7^3)^3)) * ((-7)^10) = -7 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2767_276771


namespace NUMINAMATH_CALUDE_two_digit_numbers_with_special_property_l2767_276739

def is_valid_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99 ∧
  n % 7 = 1 ∧
  (10 * (n % 10) + n / 10) % 7 = 1

theorem two_digit_numbers_with_special_property :
  {n : ℕ | is_valid_number n} = {22, 29, 92, 99} :=
by sorry

end NUMINAMATH_CALUDE_two_digit_numbers_with_special_property_l2767_276739


namespace NUMINAMATH_CALUDE_smaller_number_proof_l2767_276723

theorem smaller_number_proof (x y : ℝ) 
  (h1 : y = 71.99999999999999)
  (h2 : y - x = (1/3) * y) : 
  x = 48 := by
  sorry

end NUMINAMATH_CALUDE_smaller_number_proof_l2767_276723


namespace NUMINAMATH_CALUDE_functional_equation_solution_l2767_276760

theorem functional_equation_solution (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (x^2 - y^2) = x * f x - y * f y) :
  ∃ c : ℝ, ∀ x : ℝ, f x = c * x :=
by sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l2767_276760


namespace NUMINAMATH_CALUDE_percentage_of_b_grades_l2767_276794

def grading_scale : List (String × Nat × Nat) :=
  [("A", 93, 100), ("B", 87, 92), ("C", 78, 86), ("D", 70, 77), ("F", 0, 69)]

def grades : List Nat :=
  [88, 66, 92, 83, 90, 99, 74, 78, 85, 72, 95, 86, 79, 68, 81, 64, 87, 91, 76, 89]

def is_grade_b (grade : Nat) : Bool :=
  87 ≤ grade ∧ grade ≤ 92

def count_grade_b (grades : List Nat) : Nat :=
  grades.filter is_grade_b |>.length

theorem percentage_of_b_grades :
  (count_grade_b grades : Rat) / grades.length * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_b_grades_l2767_276794


namespace NUMINAMATH_CALUDE_largest_mersenne_prime_under_500_l2767_276732

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def mersenne_prime (p : ℕ) : Prop := is_prime p ∧ ∃ n : ℕ, is_prime n ∧ p = 2^n - 1

theorem largest_mersenne_prime_under_500 :
  ∃ p : ℕ, mersenne_prime p ∧ p < 500 ∧ ∀ q : ℕ, mersenne_prime q → q < 500 → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_largest_mersenne_prime_under_500_l2767_276732


namespace NUMINAMATH_CALUDE_multiple_problem_l2767_276753

theorem multiple_problem (m : ℤ) : 17 = m * (2625 / 1000) - 4 ↔ m = 8 := by
  sorry

end NUMINAMATH_CALUDE_multiple_problem_l2767_276753


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l2767_276736

def P : Set ℝ := {1, 2, 3, 4}
def Q : Set ℝ := {x : ℝ | 0 < x ∧ x < 5}

theorem sufficient_not_necessary : 
  (∀ x, x ∈ P → x ∈ Q) ∧ (∃ x, x ∈ Q ∧ x ∉ P) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l2767_276736


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l2767_276797

theorem regular_polygon_sides (n : ℕ) (exterior_angle : ℝ) : 
  exterior_angle = 24 → n * exterior_angle = 360 → n = 15 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l2767_276797


namespace NUMINAMATH_CALUDE_curve_is_circle_l2767_276705

/-- A point in polar coordinates -/
structure PolarPoint where
  r : ℝ
  θ : ℝ

/-- The set of points satisfying r = 3 in polar coordinates -/
def Curve : Set PolarPoint :=
  {p : PolarPoint | p.r = 3}

/-- Definition of a circle in polar coordinates -/
def IsCircle (s : Set PolarPoint) : Prop :=
  ∃ (center : PolarPoint) (radius : ℝ),
    ∀ p ∈ s, p.r = radius

theorem curve_is_circle : IsCircle Curve := by
  sorry

end NUMINAMATH_CALUDE_curve_is_circle_l2767_276705


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l2767_276778

/-- A geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- The theorem stating the property of the geometric sequence -/
theorem geometric_sequence_property (a : ℕ → ℝ) 
    (h_geo : GeometricSequence a) 
    (h_prod : a 1 * a 7 * a 13 = 8) : 
    a 3 * a 11 = 4 := by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_property_l2767_276778


namespace NUMINAMATH_CALUDE_unique_valid_stamp_set_l2767_276714

/-- Given stamps of denominations 7, n, and n+1 cents, 
    110 cents is the greatest postage that cannot be formed -/
def is_valid_stamp_set (n : ℕ) : Prop :=
  ∀ m : ℕ, m > 110 → ∃ (a b c : ℕ), m = 7 * a + n * b + (n + 1) * c ∧
  ¬∃ (a b c : ℕ), 110 = 7 * a + n * b + (n + 1) * c

theorem unique_valid_stamp_set :
  ∃! n : ℕ, n > 0 ∧ is_valid_stamp_set n :=
by sorry

end NUMINAMATH_CALUDE_unique_valid_stamp_set_l2767_276714


namespace NUMINAMATH_CALUDE_riverdale_high_quiz_l2767_276775

theorem riverdale_high_quiz (total_contestants : ℕ) (total_students : ℕ) 
  (h1 : total_contestants = 234) 
  (h2 : total_students = 420) : 
  ∃ (freshmen juniors : ℕ), 
    freshmen + juniors = total_students ∧
    (3 * freshmen) / 7 + (3 * juniors) / 4 = total_contestants ∧
    freshmen = 64 ∧ 
    juniors = 356 := by
  sorry

end NUMINAMATH_CALUDE_riverdale_high_quiz_l2767_276775


namespace NUMINAMATH_CALUDE_hens_count_l2767_276716

/-- Represents the number of hens and cows in a farm -/
structure Farm where
  hens : ℕ
  cows : ℕ

/-- The total number of heads in the farm -/
def totalHeads (f : Farm) : ℕ := f.hens + f.cows

/-- The total number of feet in the farm -/
def totalFeet (f : Farm) : ℕ := 2 * f.hens + 4 * f.cows

/-- A farm satisfying the given conditions -/
def satisfiesConditions (f : Farm) : Prop :=
  totalHeads f = 50 ∧ totalFeet f = 144

theorem hens_count (f : Farm) (h : satisfiesConditions f) : f.hens = 28 := by
  sorry

end NUMINAMATH_CALUDE_hens_count_l2767_276716


namespace NUMINAMATH_CALUDE_domain_of_function_l2767_276750

/-- The domain of the function f(x) = √(2x-1) / (x^2 + x - 2) -/
theorem domain_of_function (x : ℝ) : 
  x ∈ {y : ℝ | y ≥ (1/2 : ℝ) ∧ y ≠ 1} ↔ 
    (2*x - 1 ≥ 0 ∧ x^2 + x - 2 ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_domain_of_function_l2767_276750


namespace NUMINAMATH_CALUDE_jean_burglary_charges_l2767_276738

/-- Represents the charges and sentences for Jean's case -/
structure CriminalCase where
  arson_counts : ℕ
  burglary_charges : ℕ
  petty_larceny_charges : ℕ
  arson_sentence : ℕ
  burglary_sentence : ℕ
  petty_larceny_sentence : ℕ
  total_sentence : ℕ

/-- Calculates the total sentence for a given criminal case -/
def total_sentence (case : CriminalCase) : ℕ :=
  case.arson_counts * case.arson_sentence +
  case.burglary_charges * case.burglary_sentence +
  case.petty_larceny_charges * case.petty_larceny_sentence

/-- Theorem stating that Jean's case has 2 burglary charges -/
theorem jean_burglary_charges :
  ∃ (case : CriminalCase),
    case.arson_counts = 3 ∧
    case.petty_larceny_charges = 6 * case.burglary_charges ∧
    case.arson_sentence = 36 ∧
    case.burglary_sentence = 18 ∧
    case.petty_larceny_sentence = case.burglary_sentence / 3 ∧
    total_sentence case = 216 ∧
    case.burglary_charges = 2 :=
sorry

end NUMINAMATH_CALUDE_jean_burglary_charges_l2767_276738


namespace NUMINAMATH_CALUDE_green_square_coincidence_l2767_276757

/-- Represents a half of the figure -/
structure HalfFigure where
  greenSquares : ℕ
  redTriangles : ℕ
  blueTriangles : ℕ

/-- Represents the folded figure -/
structure FoldedFigure where
  coincidingGreenSquares : ℕ
  coincidingRedTrianglePairs : ℕ
  coincidingBlueTrianglePairs : ℕ
  redBluePairs : ℕ

/-- The theorem to be proved -/
theorem green_square_coincidence 
  (half : HalfFigure) 
  (folded : FoldedFigure) : 
  half.greenSquares = 4 ∧ 
  half.redTriangles = 3 ∧ 
  half.blueTriangles = 6 ∧
  folded.coincidingRedTrianglePairs = 2 ∧
  folded.coincidingBlueTrianglePairs = 2 ∧
  folded.redBluePairs = 3 →
  folded.coincidingGreenSquares = half.greenSquares :=
by sorry

end NUMINAMATH_CALUDE_green_square_coincidence_l2767_276757


namespace NUMINAMATH_CALUDE_nonzero_real_solution_l2767_276766

theorem nonzero_real_solution (u v : ℝ) (hu : u ≠ 0) (hv : v ≠ 0)
  (eq1 : u + 1 / v = 8) (eq2 : v + 1 / u = 16 / 3) :
  u = 4 + Real.sqrt 232 / 4 ∨ u = 4 - Real.sqrt 232 / 4 := by
sorry

end NUMINAMATH_CALUDE_nonzero_real_solution_l2767_276766


namespace NUMINAMATH_CALUDE_east_high_sports_percentage_l2767_276727

/-- The percentage of students who play sports at East High School -/
def percentage_sports (total_students : ℕ) (soccer_players : ℕ) (soccer_percentage : ℚ) : ℚ :=
  (soccer_players : ℚ) / (soccer_percentage * (total_students : ℚ))

theorem east_high_sports_percentage :
  percentage_sports 400 26 (25 / 200) = 13 / 25 :=
by sorry

end NUMINAMATH_CALUDE_east_high_sports_percentage_l2767_276727


namespace NUMINAMATH_CALUDE_carl_teaches_six_periods_l2767_276713

-- Define the given conditions
def cards_per_student : ℕ := 10
def students_per_class : ℕ := 30
def cards_per_pack : ℕ := 50
def cost_per_pack : ℚ := 3
def total_spent : ℚ := 108

-- Define the number of periods
def periods : ℕ := 6

-- Theorem statement
theorem carl_teaches_six_periods :
  (total_spent / cost_per_pack) * cards_per_pack =
  periods * students_per_class * cards_per_student :=
by sorry

end NUMINAMATH_CALUDE_carl_teaches_six_periods_l2767_276713


namespace NUMINAMATH_CALUDE_equation_solution_l2767_276728

theorem equation_solution (x : ℝ) (hx : x ≠ 0) :
  (5 * x)^4 = (10 * x)^3 → x = 8/5 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2767_276728


namespace NUMINAMATH_CALUDE_unique_number_l2767_276749

def is_valid_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧ n % 10 = 2 ∧
  200 + (n / 10) = n + 18

theorem unique_number : ∃! n : ℕ, is_valid_number n ∧ n = 202 :=
sorry

end NUMINAMATH_CALUDE_unique_number_l2767_276749


namespace NUMINAMATH_CALUDE_sum_of_integers_l2767_276718

theorem sum_of_integers (x y : ℕ+) 
  (h1 : x.val^2 + y.val^2 = 145) 
  (h2 : x.val * y.val = 72) : 
  x.val + y.val = 17 := by
sorry

end NUMINAMATH_CALUDE_sum_of_integers_l2767_276718


namespace NUMINAMATH_CALUDE_base_8_2453_equals_1323_l2767_276709

def base_8_to_10 (digits : List Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * (8 ^ i)) 0

theorem base_8_2453_equals_1323 :
  base_8_to_10 [3, 5, 4, 2] = 1323 := by
  sorry

end NUMINAMATH_CALUDE_base_8_2453_equals_1323_l2767_276709


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_length_l2767_276730

/-- An isosceles triangle with specific height measurements -/
structure IsoscelesTriangle where
  -- The length of the base
  base : ℝ
  -- The length of the equal sides
  side : ℝ
  -- The height to the base
  height_to_base : ℝ
  -- The height to a lateral side
  height_to_side : ℝ
  -- Conditions for an isosceles triangle
  isosceles : side > 0
  base_positive : base > 0
  height_to_base_positive : height_to_base > 0
  height_to_side_positive : height_to_side > 0
  -- Pythagorean theorem for the height to the base
  pythagorean_base : side^2 = height_to_base^2 + (base/2)^2
  -- Pythagorean theorem for the height to the side
  pythagorean_side : side^2 = height_to_side^2 + (base/2)^2

/-- Theorem: If the height to the base is 10 and the height to the side is 12,
    then the base of the isosceles triangle is 15 -/
theorem isosceles_triangle_base_length
  (triangle : IsoscelesTriangle)
  (h1 : triangle.height_to_base = 10)
  (h2 : triangle.height_to_side = 12) :
  triangle.base = 15 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_length_l2767_276730


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l2767_276701

-- Problem 1
theorem problem_1 : 23 + (-16) - (-7) = 14 := by sorry

-- Problem 2
theorem problem_2 : (3/4 - 7/8 - 5/12) * (-24) = 13 := by sorry

-- Problem 3
theorem problem_3 : (7/4 - 7/8 - 7/12) / (-7/8) + (-7/8) / (7/4 - 7/8 - 7/12) = -10/3 := by sorry

-- Problem 4
theorem problem_4 : -(1^4) - (1 - 0.5) * (1/3) * (2 - (-3)^2) = 1/6 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l2767_276701


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l2767_276708

theorem quadratic_inequality_solution (a b : ℝ) : 
  (∀ x, x^2 - a*x + b < 0 ↔ -1 < x ∧ x < 3) → a + b = -1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l2767_276708


namespace NUMINAMATH_CALUDE_two_numbers_difference_l2767_276758

theorem two_numbers_difference (x y : ℝ) : 
  x + y = 40 ∧ 
  3 * max x y - 4 * min x y = 44 → 
  |x - y| = 18 := by
sorry

end NUMINAMATH_CALUDE_two_numbers_difference_l2767_276758


namespace NUMINAMATH_CALUDE_batsman_average_l2767_276721

/-- Represents a batsman's performance over multiple innings -/
structure Batsman where
  innings : Nat
  totalRuns : Nat
  latestScore : Nat
  averageIncrease : Nat

/-- Calculates the average score of a batsman after their latest innings -/
def calculateAverage (b : Batsman) : Nat :=
  (b.totalRuns + b.latestScore) / b.innings

/-- Theorem: Given the conditions, prove that the batsman's average after the 12th innings is 58 runs -/
theorem batsman_average (b : Batsman) 
  (h1 : b.innings = 12)
  (h2 : b.latestScore = 80)
  (h3 : calculateAverage b = calculateAverage { b with innings := b.innings - 1 } + 2) :
  calculateAverage b = 58 := by
  sorry

end NUMINAMATH_CALUDE_batsman_average_l2767_276721


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l2767_276763

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_ratio
  (a : ℕ → ℝ)
  (h_pos : ∀ n, a n > 0)
  (h_decr : ∀ n, a (n + 1) < a n)
  (h_geom : geometric_sequence a)
  (h_prod : a 2 * a 8 = 6)
  (h_sum : a 4 + a 6 = 5) :
  a 5 / a 7 = 3/2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l2767_276763


namespace NUMINAMATH_CALUDE_brendas_age_l2767_276782

/-- Given the ages of Addison, Brenda, and Janet, prove that Brenda's age is 8/3 years. -/
theorem brendas_age (A B J : ℚ) 
  (h1 : A = 4 * B)  -- Addison's age is four times Brenda's age
  (h2 : J = B + 8)  -- Janet is eight years older than Brenda
  (h3 : A = J)      -- Addison and Janet are twins (same age)
  : B = 8 / 3 := by
  sorry

end NUMINAMATH_CALUDE_brendas_age_l2767_276782


namespace NUMINAMATH_CALUDE_remaining_boys_average_weight_l2767_276719

/-- Proves that given a class of 30 boys where 22 boys have an average weight of 50.25 kg
    and the average weight of all boys is 48.89 kg, the average weight of the remaining boys is 45.15 kg. -/
theorem remaining_boys_average_weight :
  let total_boys : ℕ := 30
  let known_boys : ℕ := 22
  let known_boys_avg_weight : ℝ := 50.25
  let all_boys_avg_weight : ℝ := 48.89
  let remaining_boys : ℕ := total_boys - known_boys
  let remaining_boys_avg_weight : ℝ := (total_boys * all_boys_avg_weight - known_boys * known_boys_avg_weight) / remaining_boys
  remaining_boys_avg_weight = 45.15 := by
sorry

end NUMINAMATH_CALUDE_remaining_boys_average_weight_l2767_276719


namespace NUMINAMATH_CALUDE_units_digit_of_sum_of_squares_2007_odd_integers_l2767_276768

def first_n_odd_integers (n : ℕ) : List ℕ :=
  List.range n |> List.map (fun i => 2 * i + 1)

def square (n : ℕ) : ℕ := n * n

def units_digit (n : ℕ) : ℕ := n % 10

def sum_of_squares (list : List ℕ) : ℕ :=
  list.map square |> List.sum

theorem units_digit_of_sum_of_squares_2007_odd_integers :
  units_digit (sum_of_squares (first_n_odd_integers 2007)) = 5 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_sum_of_squares_2007_odd_integers_l2767_276768


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2767_276776

theorem sufficient_not_necessary_condition (x : ℝ) :
  (∀ x, x > 1 → x ≥ 1) ∧ (∃ x, x ≥ 1 ∧ ¬(x > 1)) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2767_276776


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l2767_276759

theorem simplify_and_evaluate (x y : ℚ) 
  (hx : x = 1/2) (hy : y = 1/3) : 
  (3*x - y)^2 - (3*x + 2*y)*(3*x - 2*y) = -4/9 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l2767_276759


namespace NUMINAMATH_CALUDE_laura_charge_account_balance_l2767_276767

/-- Calculates the total amount owed after applying simple interest --/
def total_amount_owed (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate * time)

/-- Theorem stating that the total amount owed is $37.10 given the problem conditions --/
theorem laura_charge_account_balance : 
  total_amount_owed 35 0.06 1 = 37.10 := by
  sorry

end NUMINAMATH_CALUDE_laura_charge_account_balance_l2767_276767


namespace NUMINAMATH_CALUDE_teachers_on_field_trip_l2767_276747

theorem teachers_on_field_trip 
  (num_students : ℕ) 
  (student_ticket_cost : ℕ) 
  (teacher_ticket_cost : ℕ) 
  (total_cost : ℕ) :
  num_students = 12 →
  student_ticket_cost = 1 →
  teacher_ticket_cost = 3 →
  total_cost = 24 →
  ∃ (num_teachers : ℕ), 
    num_students * student_ticket_cost + num_teachers * teacher_ticket_cost = total_cost ∧
    num_teachers = 4 := by
sorry

end NUMINAMATH_CALUDE_teachers_on_field_trip_l2767_276747


namespace NUMINAMATH_CALUDE_fabric_cutting_l2767_276765

theorem fabric_cutting (initial_length : ℚ) (desired_length : ℚ) : 
  initial_length = 2/3 → desired_length = 1/2 → 
  initial_length - (initial_length / 4) = desired_length :=
by sorry

end NUMINAMATH_CALUDE_fabric_cutting_l2767_276765


namespace NUMINAMATH_CALUDE_triangle_max_area_l2767_276787

theorem triangle_max_area (a b c : ℝ) (A B C : ℝ) :
  C = π / 6 →
  a + b = 12 →
  0 < a ∧ 0 < b ∧ 0 < c →
  (∀ a' b' c' : ℝ, 
    a' + b' = 12 → 
    0 < a' ∧ 0 < b' ∧ 0 < c' → 
    (1/2) * a' * b' * Real.sin C ≤ 9) ∧
  (∃ a' b' c' : ℝ, 
    a' + b' = 12 ∧ 
    0 < a' ∧ 0 < b' ∧ 0 < c' ∧ 
    (1/2) * a' * b' * Real.sin C = 9) :=
by sorry

end NUMINAMATH_CALUDE_triangle_max_area_l2767_276787


namespace NUMINAMATH_CALUDE_pregnant_fish_count_l2767_276734

theorem pregnant_fish_count (num_tanks : ℕ) (young_per_fish : ℕ) (total_young : ℕ) :
  num_tanks = 3 →
  young_per_fish = 20 →
  total_young = 240 →
  ∃ (fish_per_tank : ℕ), fish_per_tank * num_tanks * young_per_fish = total_young ∧ fish_per_tank = 4 :=
by sorry

end NUMINAMATH_CALUDE_pregnant_fish_count_l2767_276734
