import Mathlib

namespace NUMINAMATH_CALUDE_dorothy_taxes_l124_12467

/-- Calculates the amount left after taxes given an annual income and tax rate. -/
def amountLeftAfterTaxes (annualIncome : ℝ) (taxRate : ℝ) : ℝ :=
  annualIncome * (1 - taxRate)

/-- Proves that given an annual income of $60,000 and a tax rate of 18%, 
    the amount left after taxes is $49,200. -/
theorem dorothy_taxes : 
  amountLeftAfterTaxes 60000 0.18 = 49200 := by
  sorry

end NUMINAMATH_CALUDE_dorothy_taxes_l124_12467


namespace NUMINAMATH_CALUDE_graph_translation_l124_12472

-- Define the original function
def original_function (x : ℝ) : ℝ := 3 * x - 4

-- Define the transformation (moving up by 2 units)
def transform (f : ℝ → ℝ) : ℝ → ℝ := λ x => f x + 2

-- State the theorem
theorem graph_translation :
  ∀ x : ℝ, transform original_function x = 3 * x - 2 := by
sorry

end NUMINAMATH_CALUDE_graph_translation_l124_12472


namespace NUMINAMATH_CALUDE_solution_set_correct_l124_12437

/-- The set of solutions to the equation 1/(x^2 + 13x - 12) + 1/(x^2 + 4x - 12) + 1/(x^2 - 11x - 12) = 0 -/
def solution_set : Set ℝ := {1, -12, 4, -3}

/-- The equation to be solved -/
def equation (x : ℝ) : Prop :=
  1 / (x^2 + 13*x - 12) + 1 / (x^2 + 4*x - 12) + 1 / (x^2 - 11*x - 12) = 0

theorem solution_set_correct :
  ∀ x : ℝ, equation x ↔ x ∈ solution_set := by sorry

end NUMINAMATH_CALUDE_solution_set_correct_l124_12437


namespace NUMINAMATH_CALUDE_integral_of_power_function_l124_12453

theorem integral_of_power_function : 
  ∫ x in (0:ℝ)..2, (1 + 3*x)^4 = 1120.4 := by sorry

end NUMINAMATH_CALUDE_integral_of_power_function_l124_12453


namespace NUMINAMATH_CALUDE_yeri_change_correct_l124_12447

def calculate_change (num_candies : ℕ) (candy_cost : ℕ) (num_chocolates : ℕ) (chocolate_cost : ℕ) (amount_paid : ℕ) : ℕ :=
  amount_paid - (num_candies * candy_cost + num_chocolates * chocolate_cost)

theorem yeri_change_correct : 
  calculate_change 5 120 3 350 2500 = 850 := by
  sorry

end NUMINAMATH_CALUDE_yeri_change_correct_l124_12447


namespace NUMINAMATH_CALUDE_derivative_x_exp_cos_l124_12413

/-- The derivative of xe^(cos x) is -x sin x * e^(cos x) + e^(cos x) -/
theorem derivative_x_exp_cos (x : ℝ) :
  deriv (fun x => x * Real.exp (Real.cos x)) x =
  -x * Real.sin x * Real.exp (Real.cos x) + Real.exp (Real.cos x) := by
  sorry

end NUMINAMATH_CALUDE_derivative_x_exp_cos_l124_12413


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l124_12474

/-- The eccentricity of a hyperbola with specific intersection properties -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let Γ := {p : ℝ × ℝ | p.1^2 / a^2 - p.2^2 / b^2 = 1}
  let c := Real.sqrt (a^2 + b^2)
  let F₁ : ℝ × ℝ := (-c, 0)
  let F₂ : ℝ × ℝ := (c, 0)
  ∀ A B : ℝ × ℝ,
    A ∈ Γ → B ∈ Γ →
    (∃ t : ℝ, A = F₂ + t • (B - F₂)) →
    ‖A - F₁‖ = ‖F₁ - F₂‖ →
    ‖B - F₂‖ = 2 * ‖A - F₂‖ →
    c / a = 5 / 3 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l124_12474


namespace NUMINAMATH_CALUDE_water_usage_fraction_l124_12419

theorem water_usage_fraction (initial_water : ℚ) (car_water : ℚ) (num_cars : ℕ) 
  (plant_water_diff : ℚ) (plate_clothes_water : ℚ) : 
  initial_water = 65 → 
  car_water = 7 → 
  num_cars = 2 → 
  plant_water_diff = 11 → 
  plate_clothes_water = 24 → 
  let total_car_water := car_water * num_cars
  let plant_water := total_car_water - plant_water_diff
  let total_used_water := total_car_water + plant_water
  let remaining_water := initial_water - total_used_water
  plate_clothes_water / remaining_water = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_water_usage_fraction_l124_12419


namespace NUMINAMATH_CALUDE_larger_number_proof_l124_12459

theorem larger_number_proof (x y : ℝ) (h1 : y > x) (h2 : 5 * y = 6 * x) (h3 : y - x = 12) : y = 72 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_proof_l124_12459


namespace NUMINAMATH_CALUDE_power_zero_equivalence_l124_12475

theorem power_zero_equivalence (x : ℝ) (h : x ≠ 0) : x^0 = 1/(x^0) := by
  sorry

end NUMINAMATH_CALUDE_power_zero_equivalence_l124_12475


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_product_l124_12440

theorem quadratic_roots_sum_product (p q : ℝ) : 
  (∃ x y : ℝ, 3 * x^2 - p * x + q = 0 ∧ 3 * y^2 - p * y + q = 0 ∧ x + y = 9 ∧ x * y = 14) → 
  p + q = 69 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_product_l124_12440


namespace NUMINAMATH_CALUDE_min_value_function_l124_12477

theorem min_value_function (x y : ℝ) (hx : x > 1) (hy : y > 1) :
  (x^2 + y) / (y^2 - 1) + (y^2 + x) / (x^2 - 1) ≥ 8/3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_function_l124_12477


namespace NUMINAMATH_CALUDE_consecutive_numbers_problem_l124_12409

theorem consecutive_numbers_problem (x y z : ℤ) : 
  (y = z + 1) →  -- x, y, and z are consecutive
  (x = y + 1) →  -- x, y, and z are consecutive
  (x > y) →      -- x > y > z
  (y > z) →      -- x > y > z
  (2*x + 3*y + 3*z = 5*y + 11) →  -- given equation
  (z = 3) →      -- given value of z
  y = 4 :=
by sorry

end NUMINAMATH_CALUDE_consecutive_numbers_problem_l124_12409


namespace NUMINAMATH_CALUDE_max_product_distances_l124_12484

/-- Two perpendicular lines passing through points A and B, intersecting at P -/
structure PerpendicularLines where
  A : ℝ × ℝ
  B : ℝ × ℝ
  P : ℝ × ℝ
  perpendicular : (P.1 - A.1) * (P.1 - B.1) + (P.2 - A.2) * (P.2 - B.2) = 0

/-- The maximum value of |PA| * |PB| for perpendicular lines through A(0, 0) and B(1, 3) -/
theorem max_product_distances (l : PerpendicularLines) 
  (h_A : l.A = (0, 0)) (h_B : l.B = (1, 3)) : 
  ∃ (max : ℝ), ∀ (P : ℝ × ℝ), 
    ((P.1 - l.A.1)^2 + (P.2 - l.A.2)^2) * ((P.1 - l.B.1)^2 + (P.2 - l.B.2)^2) ≤ max^2 ∧ 
    max = 5 :=
sorry

end NUMINAMATH_CALUDE_max_product_distances_l124_12484


namespace NUMINAMATH_CALUDE_unique_prime_p_l124_12428

theorem unique_prime_p (p : ℕ) : 
  Prime p ∧ 
  Prime (8 * p^4 - 3003) ∧ 
  (8 * p^4 - 3003 > 0) ↔ 
  p = 5 := by sorry

end NUMINAMATH_CALUDE_unique_prime_p_l124_12428


namespace NUMINAMATH_CALUDE_sticker_distribution_l124_12451

/-- The number of ways to distribute n identical objects into k distinct containers --/
def distribute (n : ℕ) (k : ℕ) : ℕ := sorry

theorem sticker_distribution : distribute 10 4 = 251 := by sorry

end NUMINAMATH_CALUDE_sticker_distribution_l124_12451


namespace NUMINAMATH_CALUDE_train_speed_problem_l124_12434

theorem train_speed_problem (train_length : ℝ) (crossing_time : ℝ) (speed_ratio : ℝ) :
  train_length = 150 →
  crossing_time = 12 →
  speed_ratio = 3 →
  let slower_speed := (2 * train_length) / (crossing_time * (speed_ratio + 1))
  let faster_speed := speed_ratio * slower_speed
  faster_speed = 18.75 :=
by
  sorry

end NUMINAMATH_CALUDE_train_speed_problem_l124_12434


namespace NUMINAMATH_CALUDE_eunji_lives_higher_l124_12422

def yoojung_floor : ℕ := 17
def eunji_floor : ℕ := 25

theorem eunji_lives_higher : eunji_floor > yoojung_floor := by
  sorry

end NUMINAMATH_CALUDE_eunji_lives_higher_l124_12422


namespace NUMINAMATH_CALUDE_t_shape_perimeter_l124_12415

/-- The perimeter of a T shape formed by two rectangles with given dimensions and overlap -/
theorem t_shape_perimeter (horizontal_width horizontal_height vertical_width vertical_height overlap : ℝ) :
  horizontal_width = 3 →
  horizontal_height = 5 →
  vertical_width = 2 →
  vertical_height = 4 →
  overlap = 1 →
  2 * (horizontal_width + horizontal_height) + 2 * (vertical_width + vertical_height) - 2 * overlap = 26 := by
  sorry

#check t_shape_perimeter

end NUMINAMATH_CALUDE_t_shape_perimeter_l124_12415


namespace NUMINAMATH_CALUDE_area_BCD_l124_12471

-- Define the points A, B, C, D
variable (A B C D : ℝ × ℝ)

-- Define the conditions
variable (area_ABC : Real)
variable (length_AC : Real)
variable (length_CD : Real)

-- Axioms
axiom area_ABC_value : area_ABC = 45
axiom AC_length : length_AC = 10
axiom CD_length : length_CD = 30
axiom B_perpendicular_AD : (B.2 - A.2) * (D.1 - A.1) = (B.1 - A.1) * (D.2 - A.2)

-- Theorem to prove
theorem area_BCD (h : ℝ) : 
  area_ABC = 1/2 * length_AC * h → 
  1/2 * length_CD * h = 135 :=
sorry

end NUMINAMATH_CALUDE_area_BCD_l124_12471


namespace NUMINAMATH_CALUDE_sum_cube_minus_twice_sum_square_is_zero_l124_12455

theorem sum_cube_minus_twice_sum_square_is_zero
  (p q r s : ℝ)
  (sum_condition : p + q + r + s = 8)
  (sum_square_condition : p^2 + q^2 + r^2 + s^2 = 16) :
  p^3 + q^3 + r^3 + s^3 - 2*(p^2 + q^2 + r^2 + s^2) = 0 :=
by sorry

end NUMINAMATH_CALUDE_sum_cube_minus_twice_sum_square_is_zero_l124_12455


namespace NUMINAMATH_CALUDE_coats_collected_from_high_schools_l124_12424

theorem coats_collected_from_high_schools 
  (total_coats : ℕ) 
  (elementary_coats : ℕ) 
  (h1 : total_coats = 9437)
  (h2 : elementary_coats = 2515) :
  total_coats - elementary_coats = 6922 := by
sorry

end NUMINAMATH_CALUDE_coats_collected_from_high_schools_l124_12424


namespace NUMINAMATH_CALUDE_eighth_power_fraction_l124_12425

theorem eighth_power_fraction (x : ℝ) (h : x > 0) :
  (x^(1/2)) / (x^(1/4)) = x^(1/4) :=
by sorry

end NUMINAMATH_CALUDE_eighth_power_fraction_l124_12425


namespace NUMINAMATH_CALUDE_no_equation_fits_l124_12469

def points : List (ℝ × ℝ) := [(0, 200), (1, 140), (2, 80), (3, 20), (4, 0)]

def equation1 (x : ℝ) : ℝ := 200 - 15 * x
def equation2 (x : ℝ) : ℝ := 200 - 20 * x + 5 * x^2
def equation3 (x : ℝ) : ℝ := 200 - 30 * x + 10 * x^2
def equation4 (x : ℝ) : ℝ := 150 - 50 * x

theorem no_equation_fits : 
  ∀ (x y : ℝ), (x, y) ∈ points → 
    (y ≠ equation1 x) ∨ 
    (y ≠ equation2 x) ∨ 
    (y ≠ equation3 x) ∨ 
    (y ≠ equation4 x) := by
  sorry

end NUMINAMATH_CALUDE_no_equation_fits_l124_12469


namespace NUMINAMATH_CALUDE_divisibility_of_n_squared_plus_n_plus_two_l124_12450

theorem divisibility_of_n_squared_plus_n_plus_two :
  (∀ n : ℕ, 2 ∣ (n^2 + n + 2)) ∧
  (∃ n : ℕ, ¬(5 ∣ (n^2 + n + 2))) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_of_n_squared_plus_n_plus_two_l124_12450


namespace NUMINAMATH_CALUDE_x_with_three_prime_divisors_including_2_l124_12497

theorem x_with_three_prime_divisors_including_2 (x n : ℕ) :
  x = 2^n - 32 ∧ 
  (∃ p q : ℕ, Prime p ∧ Prime q ∧ p ≠ q ∧ p ≠ 2 ∧ q ≠ 2 ∧
    (∀ r : ℕ, Prime r → r ∣ x → r = 2 ∨ r = p ∨ r = q)) →
  x = 2016 ∨ x = 16352 := by
sorry

end NUMINAMATH_CALUDE_x_with_three_prime_divisors_including_2_l124_12497


namespace NUMINAMATH_CALUDE_greatest_lower_bound_l124_12438

theorem greatest_lower_bound (x y : ℝ) (h1 : x ≠ y) (h2 : x * y = 2) :
  ((x + y)^2 - 6) * ((x - y)^2 + 8) / (x - y)^2 ≥ 18 ∧
  ∀ C > 18, ∃ x y : ℝ, x ≠ y ∧ x * y = 2 ∧
    ((x + y)^2 - 6) * ((x - y)^2 + 8) / (x - y)^2 < C :=
by sorry

end NUMINAMATH_CALUDE_greatest_lower_bound_l124_12438


namespace NUMINAMATH_CALUDE_square_sum_from_means_l124_12462

theorem square_sum_from_means (x y : ℝ) 
  (h_arithmetic : (x + y) / 2 = 18) 
  (h_geometric : Real.sqrt (x * y) = Real.sqrt 92) : 
  x^2 + y^2 = 1112 := by sorry

end NUMINAMATH_CALUDE_square_sum_from_means_l124_12462


namespace NUMINAMATH_CALUDE_double_beavers_half_time_beavers_build_dam_l124_12478

/-- Represents the time (in hours) it takes a given number of beavers to build a dam -/
def build_time (num_beavers : ℕ) : ℝ := 
  if num_beavers = 18 then 8 else 
  if num_beavers = 36 then 4 else 0

/-- The proposition that doubling the number of beavers halves the build time -/
theorem double_beavers_half_time : 
  build_time 36 = (build_time 18) / 2 := by
sorry

/-- The main theorem stating that 36 beavers can build the dam in 4 hours -/
theorem beavers_build_dam : 
  build_time 36 = 4 := by
sorry

end NUMINAMATH_CALUDE_double_beavers_half_time_beavers_build_dam_l124_12478


namespace NUMINAMATH_CALUDE_prime_value_of_polynomial_l124_12490

theorem prime_value_of_polynomial (a : ℕ) :
  Nat.Prime (a^4 - 4*a^3 + 15*a^2 - 30*a + 27) →
  a^4 - 4*a^3 + 15*a^2 - 30*a + 27 = 11 :=
by sorry

end NUMINAMATH_CALUDE_prime_value_of_polynomial_l124_12490


namespace NUMINAMATH_CALUDE_knowledge_competition_theorem_l124_12418

/-- Represents a player in the knowledge competition --/
structure Player where
  correct_prob : ℚ
  deriving Repr

/-- Represents the game setup --/
structure Game where
  player_a : Player
  player_b : Player
  num_questions : ℕ
  deriving Repr

/-- Calculates the probability of a specific score for a player --/
def prob_score (game : Game) (player : Player) (score : ℕ) : ℚ :=
  sorry

/-- Calculates the mathematical expectation of a player's score --/
def expected_score (game : Game) (player : Player) : ℚ :=
  sorry

/-- The main theorem to prove --/
theorem knowledge_competition_theorem (game : Game) :
  game.player_a = Player.mk (2/3)
  → game.player_b = Player.mk (4/5)
  → game.num_questions = 2
  → prob_score game game.player_b 10 = 337/900
  ∧ expected_score game game.player_a = 23/3 :=
  sorry

end NUMINAMATH_CALUDE_knowledge_competition_theorem_l124_12418


namespace NUMINAMATH_CALUDE_marble_difference_l124_12407

theorem marble_difference (connie_marbles juan_marbles : ℕ) 
  (h1 : connie_marbles = 39)
  (h2 : juan_marbles = 64)
  : juan_marbles - connie_marbles = 25 := by
  sorry

end NUMINAMATH_CALUDE_marble_difference_l124_12407


namespace NUMINAMATH_CALUDE_no_prime_fraction_equality_l124_12431

theorem no_prime_fraction_equality : ¬∃ (a b c d : ℕ), 
  Prime a ∧ Prime b ∧ Prime c ∧ Prime d ∧
  a < b ∧ b < c ∧ c < d ∧
  (1 : ℚ) / a + (1 : ℚ) / d = (1 : ℚ) / b + (1 : ℚ) / c := by
  sorry

end NUMINAMATH_CALUDE_no_prime_fraction_equality_l124_12431


namespace NUMINAMATH_CALUDE_inequality_proof_l124_12449

theorem inequality_proof (x : ℝ) :
  (x > 0 → x + 1/x ≥ 2) ∧
  (x > 0 → (x + 1/x = 2 ↔ x = 1)) ∧
  (x < 0 → x + 1/x ≤ -2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l124_12449


namespace NUMINAMATH_CALUDE_tv_watching_time_equivalence_l124_12495

/-- The number of minutes in an hour -/
def minutes_per_hour : ℕ := 60

/-- The number of hours Ava watched television -/
def hours_watched : ℕ := 4

/-- The theorem stating that watching TV for 4 hours is equivalent to 240 minutes -/
theorem tv_watching_time_equivalence : 
  hours_watched * minutes_per_hour = 240 := by
  sorry

end NUMINAMATH_CALUDE_tv_watching_time_equivalence_l124_12495


namespace NUMINAMATH_CALUDE_tree_break_height_l124_12417

theorem tree_break_height (tree_height road_width break_height : ℝ) 
  (h_tree : tree_height = 36)
  (h_road : road_width = 12)
  (h_pythagoras : (tree_height - break_height)^2 = break_height^2 + road_width^2) :
  break_height = 16 := by
sorry

end NUMINAMATH_CALUDE_tree_break_height_l124_12417


namespace NUMINAMATH_CALUDE_sqrt_750_minus_29_cube_l124_12414

theorem sqrt_750_minus_29_cube (a b : ℕ+) :
  (Real.sqrt 750 - 29 : ℝ) = (Real.sqrt a.val - b.val : ℝ)^3 →
  a.val + b.val = 28 := by sorry

end NUMINAMATH_CALUDE_sqrt_750_minus_29_cube_l124_12414


namespace NUMINAMATH_CALUDE_jordan_fourth_period_shots_l124_12482

/-- The number of shots blocked by Jordan in each period of a hockey game --/
structure ShotsBlocked where
  first : ℕ
  second : ℕ
  third : ℕ
  fourth : ℕ

/-- Conditions for Jordan's shot-blocking performance --/
def jordan_performance (shots : ShotsBlocked) : Prop :=
  shots.first = 4 ∧
  shots.second = 2 * shots.first ∧
  shots.third = shots.second - 3 ∧
  shots.first + shots.second + shots.third + shots.fourth = 21

/-- Theorem stating that Jordan blocked 4 shots in the fourth period --/
theorem jordan_fourth_period_shots (shots : ShotsBlocked) 
  (h : jordan_performance shots) : shots.fourth = 4 := by
  sorry

#check jordan_fourth_period_shots

end NUMINAMATH_CALUDE_jordan_fourth_period_shots_l124_12482


namespace NUMINAMATH_CALUDE_matchsticks_count_l124_12402

/-- The number of matchsticks in a box -/
def initial_matchsticks : ℕ := sorry

/-- The number of matchsticks Elvis uses per square -/
def elvis_matchsticks_per_square : ℕ := 4

/-- The number of squares Elvis makes -/
def elvis_squares : ℕ := 5

/-- The number of matchsticks Ralph uses per square -/
def ralph_matchsticks_per_square : ℕ := 8

/-- The number of squares Ralph makes -/
def ralph_squares : ℕ := 3

/-- The number of matchsticks left in the box -/
def remaining_matchsticks : ℕ := 6

theorem matchsticks_count : initial_matchsticks = 50 := by sorry

end NUMINAMATH_CALUDE_matchsticks_count_l124_12402


namespace NUMINAMATH_CALUDE_gcd_factorial_eight_ten_l124_12430

theorem gcd_factorial_eight_ten : Nat.gcd (Nat.factorial 8) (Nat.factorial 10) = Nat.factorial 8 := by
  sorry

end NUMINAMATH_CALUDE_gcd_factorial_eight_ten_l124_12430


namespace NUMINAMATH_CALUDE_final_hair_length_l124_12410

def hair_length (initial_length cut_length growth_length : ℕ) : ℕ :=
  initial_length - cut_length + growth_length

theorem final_hair_length :
  hair_length 16 11 12 = 17 := by
  sorry

end NUMINAMATH_CALUDE_final_hair_length_l124_12410


namespace NUMINAMATH_CALUDE_right_triangle_rational_sides_equiv_arithmetic_progression_l124_12439

theorem right_triangle_rational_sides_equiv_arithmetic_progression (d : ℤ) :
  (∃ (a b c : ℚ), a^2 + b^2 = c^2 ∧ (1/2) * a * b = d) ↔
  (∃ (x y z : ℚ), 2 * y^2 = x^2 + z^2) :=
sorry

end NUMINAMATH_CALUDE_right_triangle_rational_sides_equiv_arithmetic_progression_l124_12439


namespace NUMINAMATH_CALUDE_percentage_of_muslim_boys_l124_12420

theorem percentage_of_muslim_boys (total_boys : ℕ) (hindu_percentage : ℚ) (sikh_percentage : ℚ) (other_boys : ℕ) :
  total_boys = 300 →
  hindu_percentage = 28 / 100 →
  sikh_percentage = 10 / 100 →
  other_boys = 54 →
  (total_boys - (hindu_percentage * total_boys + sikh_percentage * total_boys + other_boys)) / total_boys = 44 / 100 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_muslim_boys_l124_12420


namespace NUMINAMATH_CALUDE_janice_purchase_l124_12423

theorem janice_purchase (a b c : ℕ) : 
  a + b + c = 50 →
  50 * a + 400 * b + 500 * c = 10000 →
  a = 23 :=
by sorry

end NUMINAMATH_CALUDE_janice_purchase_l124_12423


namespace NUMINAMATH_CALUDE_divisibility_condition_l124_12426

theorem divisibility_condition (a b : ℕ+) : 
  (a * b^2 + b + 7) ∣ (a^2 * b + a + b) →
  ((a = 11 ∧ b = 1) ∨ (a = 49 ∧ b = 1) ∨ (∃ k : ℕ+, a = 7 * k^2 ∧ b = 7 * k)) :=
sorry

end NUMINAMATH_CALUDE_divisibility_condition_l124_12426


namespace NUMINAMATH_CALUDE_arrangements_with_adjacent_pair_l124_12427

-- Define the number of students
def total_students : ℕ := 5

-- Define the function to calculate permutations
def permutations (n : ℕ) (k : ℕ) : ℕ :=
  Nat.factorial n / Nat.factorial (n - k)

-- Define the theorem
theorem arrangements_with_adjacent_pair :
  permutations 4 4 * permutations 2 2 = 48 := by
  sorry

end NUMINAMATH_CALUDE_arrangements_with_adjacent_pair_l124_12427


namespace NUMINAMATH_CALUDE_workers_read_all_three_l124_12443

/-- Represents the number of workers who have read books by different authors -/
structure BookReaders where
  total : ℕ
  saramago : ℕ
  kureishi : ℕ
  atwood : ℕ
  saramagoKureishi : ℕ
  allThree : ℕ

/-- The theorem to prove -/
theorem workers_read_all_three (r : BookReaders) : r.allThree = 6 :=
  by
  have h1 : r.total = 75 := by sorry
  have h2 : r.saramago = r.total / 2 := by sorry
  have h3 : r.kureishi = r.total / 4 := by sorry
  have h4 : r.atwood = r.total / 5 := by sorry
  have h5 : r.total - (r.saramago + r.kureishi + r.atwood - (r.saramagoKureishi + r.allThree)) = 
            r.saramago - (r.saramagoKureishi + r.allThree) - 1 := by sorry
  have h6 : r.saramagoKureishi = 2 * r.allThree := by sorry
  sorry

#check workers_read_all_three

end NUMINAMATH_CALUDE_workers_read_all_three_l124_12443


namespace NUMINAMATH_CALUDE_quarter_circle_roll_path_length_l124_12486

/-- The length of the path traveled by the center of a quarter-circle when rolled along a straight line -/
theorem quarter_circle_roll_path_length (r : ℝ) (h : r = 1 / Real.pi) : 
  let path_length := 2 * (r * Real.pi / 4) + r * Real.pi / 2
  path_length = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_quarter_circle_roll_path_length_l124_12486


namespace NUMINAMATH_CALUDE_smallest_divisible_by_1_to_12_and_15_l124_12411

theorem smallest_divisible_by_1_to_12_and_15 : ∃ n : ℕ, 
  (∀ k : ℕ, k ≤ 12 → k ∣ n) ∧ 
  (15 ∣ n) ∧ 
  (∀ m : ℕ, m < n → ¬(∀ k : ℕ, k ≤ 12 → k ∣ m) ∨ ¬(15 ∣ m)) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_1_to_12_and_15_l124_12411


namespace NUMINAMATH_CALUDE_mans_rowing_speed_l124_12468

/-- The man's rowing speed in still water -/
def rowing_speed : ℝ := 3.9

/-- The speed of the current -/
def current_speed : ℝ := 1.3

/-- The ratio of time taken to row upstream compared to downstream -/
def time_ratio : ℝ := 2

theorem mans_rowing_speed :
  (rowing_speed + current_speed) * time_ratio = (rowing_speed - current_speed) * (time_ratio * 2) ∧
  rowing_speed = 3 * current_speed := by
  sorry

#check mans_rowing_speed

end NUMINAMATH_CALUDE_mans_rowing_speed_l124_12468


namespace NUMINAMATH_CALUDE_chord_length_parabola_l124_12421

/-- Parabola structure -/
structure Parabola where
  equation : ℝ → ℝ → Prop

/-- Line structure -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Point structure -/
structure Point where
  x : ℝ
  y : ℝ

/-- Theorem statement -/
theorem chord_length_parabola (C : Parabola) (l : Line) (A B : Point) :
  C.equation = (fun x y => x^2 = 4*y) →
  l.intercept = 1 →
  C.equation A.x A.y →
  C.equation B.x B.y →
  (A.y + B.y) / 2 = 5 →
  ∃ k, l.slope = k ∧ k^2 = 2 →
  ∃ AB : ℝ, AB = 6 ∧ AB = Real.sqrt ((A.x - B.x)^2 + (A.y - B.y)^2) := by
  sorry

end NUMINAMATH_CALUDE_chord_length_parabola_l124_12421


namespace NUMINAMATH_CALUDE_sports_league_games_l124_12429

/-- Represents a sports league with the given conditions -/
structure SportsLeague where
  total_teams : Nat
  teams_per_division : Nat
  intra_division_games : Nat
  inter_division_games : Nat

/-- Calculate the total number of games in the sports league -/
def total_games (league : SportsLeague) : Nat :=
  let games_per_team := (league.teams_per_division - 1) * league.intra_division_games +
                        league.teams_per_division * league.inter_division_games
  (games_per_team * league.total_teams) / 2

/-- Theorem stating the total number of games in the given sports league configuration -/
theorem sports_league_games :
  let league := SportsLeague.mk 16 8 3 2
  total_games league = 296 := by
  sorry

end NUMINAMATH_CALUDE_sports_league_games_l124_12429


namespace NUMINAMATH_CALUDE_zuzka_structure_bounds_l124_12463

/-- A structure made of cubes -/
structure CubeStructure where
  base : Nat
  layers : Nat
  third_layer : Nat
  total : Nat

/-- The conditions of Zuzka's cube structure -/
def zuzka_structure (s : CubeStructure) : Prop :=
  s.base = 16 ∧ 
  s.layers ≥ 3 ∧ 
  s.third_layer = 2 ∧
  s.total = s.base + (s.layers - 1) * s.third_layer + (s.total - s.base - s.third_layer)

/-- The theorem stating the range of possible total cubes -/
theorem zuzka_structure_bounds (s : CubeStructure) :
  zuzka_structure s → 22 ≤ s.total ∧ s.total ≤ 27 :=
by
  sorry


end NUMINAMATH_CALUDE_zuzka_structure_bounds_l124_12463


namespace NUMINAMATH_CALUDE_number_puzzle_l124_12416

theorem number_puzzle : ∃ x : ℝ, x^2 + 50 = (x - 10)^2 ∧ x = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_number_puzzle_l124_12416


namespace NUMINAMATH_CALUDE_partners_shares_correct_l124_12487

/-- Represents the investment ratio of partners A, B, and C -/
def investment_ratio : Fin 3 → ℕ
| 0 => 2  -- Partner A
| 1 => 3  -- Partner B
| 2 => 5  -- Partner C

/-- The total profit in rupees -/
def total_profit : ℕ := 22400

/-- Calculates a partner's share of the profit based on their investment ratio -/
def partner_share (i : Fin 3) : ℕ :=
  (investment_ratio i * total_profit) / (investment_ratio 0 + investment_ratio 1 + investment_ratio 2)

/-- Theorem stating that the partners' shares are correct -/
theorem partners_shares_correct :
  partner_share 0 = 4480 ∧
  partner_share 1 = 6720 ∧
  partner_share 2 = 11200 := by
  sorry


end NUMINAMATH_CALUDE_partners_shares_correct_l124_12487


namespace NUMINAMATH_CALUDE_cupcake_distribution_l124_12436

/-- Given the initial number of cupcakes, the number of eaten cupcakes, and the number of packages,
    calculate the number of cupcakes in each package. -/
def cupcakes_per_package (initial : ℕ) (eaten : ℕ) (packages : ℕ) : ℕ :=
  (initial - eaten) / packages

/-- Theorem stating that given 71 initial cupcakes, 43 eaten cupcakes, and 4 packages,
    the number of cupcakes in each package is 7. -/
theorem cupcake_distribution :
  cupcakes_per_package 71 43 4 = 7 := by
  sorry


end NUMINAMATH_CALUDE_cupcake_distribution_l124_12436


namespace NUMINAMATH_CALUDE_pie_rows_l124_12492

/-- Given the number of pecan and apple pies, and the number of pies per row,
    calculate the number of complete rows. -/
def calculate_rows (pecan_pies apple_pies pies_per_row : ℕ) : ℕ :=
  (pecan_pies + apple_pies) / pies_per_row

/-- Theorem stating that with 16 pecan pies and 14 apple pies,
    arranged in rows of 5 pies each, there will be 6 complete rows. -/
theorem pie_rows : calculate_rows 16 14 5 = 6 := by
  sorry

end NUMINAMATH_CALUDE_pie_rows_l124_12492


namespace NUMINAMATH_CALUDE_larry_lunch_cost_l124_12432

/-- Calculates the amount spent on lunch given initial amount, final amount, and amount given to brother -/
def lunch_cost (initial : ℕ) (final : ℕ) (given_to_brother : ℕ) : ℕ :=
  initial - final - given_to_brother

/-- Proves that Larry's lunch cost is $5 given the problem conditions -/
theorem larry_lunch_cost :
  lunch_cost 22 15 2 = 5 := by sorry

end NUMINAMATH_CALUDE_larry_lunch_cost_l124_12432


namespace NUMINAMATH_CALUDE_gift_combinations_count_l124_12435

/-- The number of different gift packaging combinations -/
def gift_combinations (wrapping_paper : ℕ) (ribbon : ℕ) (gift_card : ℕ) (gift_box : ℕ) : ℕ :=
  wrapping_paper * ribbon * gift_card * gift_box

/-- Theorem stating the number of gift packaging combinations -/
theorem gift_combinations_count :
  gift_combinations 10 3 4 5 = 600 := by
  sorry

end NUMINAMATH_CALUDE_gift_combinations_count_l124_12435


namespace NUMINAMATH_CALUDE_three_digit_multiples_of_three_l124_12458

theorem three_digit_multiples_of_three : 
  (Finset.filter (fun c => (100 + 10 * c + 7) % 3 = 0) (Finset.range 10)).card = 3 := by
  sorry

end NUMINAMATH_CALUDE_three_digit_multiples_of_three_l124_12458


namespace NUMINAMATH_CALUDE_little_twelve_games_l124_12498

/-- Represents a basketball conference with two divisions -/
structure BasketballConference :=
  (teams_per_division : ℕ)
  (intra_division_games : ℕ)
  (inter_division_games : ℕ)

/-- The Little Twelve Basketball Conference setup -/
def little_twelve : BasketballConference :=
  { teams_per_division := 6,
    intra_division_games := 2,
    inter_division_games := 2 }

/-- Calculate the total number of conference games -/
def total_conference_games (conf : BasketballConference) : ℕ :=
  let total_teams := 2 * conf.teams_per_division
  let games_per_team := (conf.teams_per_division - 1) * conf.intra_division_games +
                        conf.teams_per_division * conf.inter_division_games
  (total_teams * games_per_team) / 2

theorem little_twelve_games :
  total_conference_games little_twelve = 132 := by
  sorry

end NUMINAMATH_CALUDE_little_twelve_games_l124_12498


namespace NUMINAMATH_CALUDE_A_contains_B_l124_12404

def A (x : ℝ) : Prop := -2 ≤ x ∧ x ≤ 5

def B (x m : ℝ) : Prop := (x - m + 1) * (x - 2 * m - 1) < 0

theorem A_contains_B (m : ℝ) : 
  (∀ x, B x m → A x) ↔ (m = -2 ∨ (-1 ≤ m ∧ m ≤ 2)) := by sorry

end NUMINAMATH_CALUDE_A_contains_B_l124_12404


namespace NUMINAMATH_CALUDE_unique_sum_property_l124_12483

theorem unique_sum_property (A : ℕ) : 
  (0 ≤ A * (A - 1999) / 2 ∧ A * (A - 1999) / 2 ≤ 999) ↔ A = 1999 :=
sorry

end NUMINAMATH_CALUDE_unique_sum_property_l124_12483


namespace NUMINAMATH_CALUDE_sufficient_condition_range_l124_12476

theorem sufficient_condition_range (m : ℝ) : 
  (∀ x : ℝ, |x - 4| ≤ 6 → x ≤ 1 + m) ∧ 
  (∃ x : ℝ, x ≤ 1 + m ∧ |x - 4| > 6) → 
  m ≥ 9 := by sorry

end NUMINAMATH_CALUDE_sufficient_condition_range_l124_12476


namespace NUMINAMATH_CALUDE_total_height_is_24cm_l124_12493

/-- The number of washers in the stack -/
def num_washers : ℕ := 11

/-- The thickness of each washer in cm -/
def washer_thickness : ℝ := 2

/-- The outer diameter of the top washer in cm -/
def top_diameter : ℝ := 24

/-- The outer diameter of the bottom washer in cm -/
def bottom_diameter : ℝ := 4

/-- The decrease in diameter between consecutive washers in cm -/
def diameter_decrease : ℝ := 2

/-- The extra height for hooks at top and bottom in cm -/
def hook_height : ℝ := 2

theorem total_height_is_24cm : 
  (num_washers : ℝ) * washer_thickness + hook_height = 24 := by
  sorry

end NUMINAMATH_CALUDE_total_height_is_24cm_l124_12493


namespace NUMINAMATH_CALUDE_train_crossing_time_l124_12444

/-- Proves that a train 360 m long traveling at 43.2 km/h takes 30 seconds to cross an electric pole. -/
theorem train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 360 →
  train_speed_kmh = 43.2 →
  crossing_time = train_length / (train_speed_kmh * 1000 / 3600) →
  crossing_time = 30 := by
  sorry

#check train_crossing_time

end NUMINAMATH_CALUDE_train_crossing_time_l124_12444


namespace NUMINAMATH_CALUDE_first_valid_year_is_2030_l124_12406

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

def is_valid_year (year : ℕ) : Prop :=
  year > 2022 ∧ sum_of_digits year = 5

theorem first_valid_year_is_2030 :
  is_valid_year 2030 ∧ ∀ y, is_valid_year y → y ≥ 2030 :=
sorry

end NUMINAMATH_CALUDE_first_valid_year_is_2030_l124_12406


namespace NUMINAMATH_CALUDE_profit_per_meter_of_cloth_l124_12448

/-- Profit per meter of cloth calculation -/
theorem profit_per_meter_of_cloth
  (total_meters : ℕ)
  (selling_price : ℕ)
  (cost_price_per_meter : ℕ)
  (h1 : total_meters = 45)
  (h2 : selling_price = 4500)
  (h3 : cost_price_per_meter = 86) :
  (selling_price - total_meters * cost_price_per_meter) / total_meters = 14 := by
sorry

end NUMINAMATH_CALUDE_profit_per_meter_of_cloth_l124_12448


namespace NUMINAMATH_CALUDE_apple_profit_percentage_l124_12496

theorem apple_profit_percentage 
  (total_apples : ℝ)
  (first_portion : ℝ)
  (second_portion : ℝ)
  (second_profit : ℝ)
  (overall_profit : ℝ)
  (h1 : total_apples = 280)
  (h2 : first_portion = 0.4)
  (h3 : second_portion = 0.6)
  (h4 : first_portion + second_portion = 1)
  (h5 : second_profit = 0.3)
  (h6 : overall_profit = 0.26)
  : ∃ (first_profit : ℝ),
    first_profit * first_portion * total_apples + 
    second_profit * second_portion * total_apples = 
    overall_profit * total_apples ∧
    first_profit = 0.2 := by
sorry

end NUMINAMATH_CALUDE_apple_profit_percentage_l124_12496


namespace NUMINAMATH_CALUDE_hyperbola_intersection_theorem_l124_12489

/-- The hyperbola C with real semi-axis length √3 and the same foci as the ellipse x²/8 + y²/4 = 1 -/
structure Hyperbola where
  /-- The real semi-axis length of the hyperbola -/
  a : ℝ
  /-- The imaginary semi-axis length of the hyperbola -/
  b : ℝ
  /-- The focal distance of the hyperbola -/
  c : ℝ
  /-- The real semi-axis length is √3 -/
  ha : a = Real.sqrt 3
  /-- The focal distance is the same as the ellipse x²/8 + y²/4 = 1 -/
  hc : c = 2
  /-- Relation between a, b, and c in a hyperbola -/
  hab : c^2 = a^2 + b^2

/-- The equation of the hyperbola -/
def hyperbola_equation (h : Hyperbola) (x y : ℝ) : Prop :=
  x^2 / h.a^2 - y^2 / h.b^2 = 1

/-- The line intersecting the hyperbola -/
def line_equation (k : ℝ) (x y : ℝ) : Prop :=
  y = k * x + Real.sqrt 2

/-- The dot product of two points with the origin -/
def dot_product (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  x₁ * x₂ + y₁ * y₂

/-- The main theorem -/
theorem hyperbola_intersection_theorem (h : Hyperbola) :
  (∀ x y, hyperbola_equation h x y ↔ x^2 / 3 - y^2 = 1) ∧
  (∀ k, (∃ x₁ y₁ x₂ y₂, 
    x₁ ≠ x₂ ∧
    hyperbola_equation h x₁ y₁ ∧
    hyperbola_equation h x₂ y₂ ∧
    line_equation k x₁ y₁ ∧
    line_equation k x₂ y₂ ∧
    dot_product x₁ y₁ x₂ y₂ > 2) ↔
   (k > -1 ∧ k < -Real.sqrt 3 / 3) ∨ (k > Real.sqrt 3 / 3 ∧ k < 1)) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_intersection_theorem_l124_12489


namespace NUMINAMATH_CALUDE_num_arrangements_eq_192_l124_12412

/-- The number of different arrangements for 7 students in a row with specific conditions -/
def num_arrangements : ℕ :=
  let total_students : ℕ := 7
  let middle_student : ℕ := 1
  let together_students : ℕ := 2
  let remaining_students : ℕ := total_students - middle_student - together_students
  let middle_positions : ℕ := 1
  let together_positions : ℕ := 2 * 4
  let remaining_positions : ℕ := remaining_students.factorial
  middle_positions * together_positions * remaining_positions

/-- Theorem stating that the number of arrangements is 192 -/
theorem num_arrangements_eq_192 : num_arrangements = 192 := by
  sorry

end NUMINAMATH_CALUDE_num_arrangements_eq_192_l124_12412


namespace NUMINAMATH_CALUDE_max_value_of_f_l124_12452

theorem max_value_of_f (x : ℝ) : 
  let f := fun (x : ℝ) => 1 / (1 - x * (1 - x))
  f x ≤ 4/3 ∧ ∃ y, f y = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_f_l124_12452


namespace NUMINAMATH_CALUDE_intersecting_circles_common_chord_l124_12464

/-- Two intersecting circles with given radii and distance between centers have a common chord of length 10 -/
theorem intersecting_circles_common_chord 
  (R : ℝ) 
  (r : ℝ) 
  (d : ℝ) 
  (h1 : R = 13) 
  (h2 : r = 5) 
  (h3 : d = 12) :
  ∃ (chord_length : ℝ), 
    chord_length = 10 ∧ 
    chord_length = 2 * R * Real.sqrt (1 - ((R^2 + d^2 - r^2) / (2 * R * d))^2) := by
  sorry

end NUMINAMATH_CALUDE_intersecting_circles_common_chord_l124_12464


namespace NUMINAMATH_CALUDE_scientific_notation_of_8500_billion_l124_12466

/-- Scientific notation representation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  h_coefficient : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- Convert a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

/-- The value in yuan -/
def value : ℝ := 8500000000000

/-- The scientific notation representation of the value -/
def scientificForm : ScientificNotation := toScientificNotation value

/-- Theorem stating that the scientific notation of 8500 billion yuan is 8.5 × 10^11 -/
theorem scientific_notation_of_8500_billion :
  scientificForm.coefficient = 8.5 ∧ scientificForm.exponent = 11 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_8500_billion_l124_12466


namespace NUMINAMATH_CALUDE_andrew_mango_purchase_l124_12480

/-- Calculates the quantity of mangoes purchased given the total amount paid,
    grape quantity, grape price, and mango price. -/
def mango_quantity (total_paid : ℕ) (grape_quantity : ℕ) (grape_price : ℕ) (mango_price : ℕ) : ℕ :=
  (total_paid - grape_quantity * grape_price) / mango_price

theorem andrew_mango_purchase :
  mango_quantity 908 7 68 48 = 9 := by
  sorry

end NUMINAMATH_CALUDE_andrew_mango_purchase_l124_12480


namespace NUMINAMATH_CALUDE_weight_relationship_and_sum_l124_12499

/-- Given the weights of Haley, Verna, and Sherry, prove their relationship and combined weight -/
theorem weight_relationship_and_sum (haley_weight verna_weight sherry_weight : ℕ) : 
  haley_weight = 103 →
  verna_weight = haley_weight + 17 →
  verna_weight * 2 = sherry_weight →
  verna_weight + sherry_weight = 360 := by
  sorry

end NUMINAMATH_CALUDE_weight_relationship_and_sum_l124_12499


namespace NUMINAMATH_CALUDE_consecutive_negative_integers_sum_l124_12433

theorem consecutive_negative_integers_sum (n : ℤ) : 
  n < 0 ∧ n * (n + 1) = 812 → n + (n + 1) = -57 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_negative_integers_sum_l124_12433


namespace NUMINAMATH_CALUDE_function_minimum_value_l124_12405

/-- The function f(x) = x + a / (x - 2) where x > 2 and f(3) = 7 has a minimum value of 6 -/
theorem function_minimum_value (a : ℝ) : 
  (∀ x > 2, ∃ y, y = x + a / (x - 2)) → 
  (3 + a / (3 - 2) = 7) → 
  (∃ m : ℝ, ∀ x > 2, x + a / (x - 2) ≥ m ∧ ∃ x₀ > 2, x₀ + a / (x₀ - 2) = m) →
  (∀ x > 2, x + a / (x - 2) ≥ 6) ∧ ∃ x₀ > 2, x₀ + a / (x₀ - 2) = 6 :=
by sorry

end NUMINAMATH_CALUDE_function_minimum_value_l124_12405


namespace NUMINAMATH_CALUDE_condition_sufficient_not_necessary_l124_12481

theorem condition_sufficient_not_necessary :
  (∀ x : ℝ, x > 4 → x ≥ 4) ∧
  (∃ x : ℝ, x ≥ 4 ∧ ¬(x > 4)) := by
  sorry

end NUMINAMATH_CALUDE_condition_sufficient_not_necessary_l124_12481


namespace NUMINAMATH_CALUDE_sum_of_roots_is_zero_l124_12445

theorem sum_of_roots_is_zero (x : ℝ) :
  (x^2 - 7*|x| + 6 = 0) →
  (∃ x₁ x₂ x₃ x₄ : ℝ,
    ((x₁ > 0 ∧ x₂ > 0 ∧ x₁^2 - 7*x₁ + 6 = 0 ∧ x₂^2 - 7*x₂ + 6 = 0) ∨
     (x₃ < 0 ∧ x₄ < 0 ∧ x₃^2 + 7*x₃ + 6 = 0 ∧ x₄^2 + 7*x₄ + 6 = 0)) ∧
    x₁ + x₂ + x₃ + x₄ = 0) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_is_zero_l124_12445


namespace NUMINAMATH_CALUDE_divisibility_of_f_minus_p_l124_12460

/-- 
For a number n in base 10, let f(n) be the sum of all numbers possible by removing some digits of n 
(including none and all). This theorem proves that for any 2011-digit integer p, f(p) - p is divisible by 9.
-/
theorem divisibility_of_f_minus_p (p : ℕ) (h : 10^2010 ≤ p ∧ p < 10^2011) : 
  ∃ k : ℤ, (2^2010 - 1) * p = 9 * k := by
  sorry

end NUMINAMATH_CALUDE_divisibility_of_f_minus_p_l124_12460


namespace NUMINAMATH_CALUDE_expression_simplification_l124_12461

theorem expression_simplification (x : ℝ) (hx : x ≠ 0 ∧ x ≠ 1 ∧ x ≠ -1) :
  (1 + 1/x) * (1 - 2/(x+1)) * (1 + 2/(x-1)) = (x+1)/x :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_l124_12461


namespace NUMINAMATH_CALUDE_movie_ticket_cost_l124_12485

theorem movie_ticket_cost 
  (num_adults : ℕ) 
  (num_children : ℕ) 
  (concession_cost : ℚ) 
  (total_cost : ℚ) 
  (child_ticket_cost : ℚ) : 
  num_adults = 5 → 
  num_children = 2 → 
  concession_cost = 12 → 
  total_cost = 76 → 
  child_ticket_cost = 7 → 
  (total_cost - concession_cost - num_children * child_ticket_cost) / num_adults = 10 :=
by sorry

end NUMINAMATH_CALUDE_movie_ticket_cost_l124_12485


namespace NUMINAMATH_CALUDE_cake_eaten_percentage_l124_12400

theorem cake_eaten_percentage (total_pieces : ℕ) (sisters : ℕ) (pieces_per_sister : ℕ) 
  (h1 : total_pieces = 240)
  (h2 : sisters = 3)
  (h3 : pieces_per_sister = 32) :
  (total_pieces - sisters * pieces_per_sister) / total_pieces * 100 = 60 := by
  sorry

#check cake_eaten_percentage

end NUMINAMATH_CALUDE_cake_eaten_percentage_l124_12400


namespace NUMINAMATH_CALUDE_trapezoid_not_axisymmetric_l124_12465

-- Define the shapes
inductive Shape
  | Angle
  | Rectangle
  | Trapezoid
  | Rhombus

-- Define the property of being axisymmetric
def is_axisymmetric (s : Shape) : Prop :=
  match s with
  | Shape.Angle => True
  | Shape.Rectangle => True
  | Shape.Rhombus => True
  | Shape.Trapezoid => false

-- Theorem stating that trapezoid is the only shape not necessarily axisymmetric
theorem trapezoid_not_axisymmetric :
  ∀ (s : Shape), ¬is_axisymmetric s ↔ s = Shape.Trapezoid :=
by sorry

end NUMINAMATH_CALUDE_trapezoid_not_axisymmetric_l124_12465


namespace NUMINAMATH_CALUDE_complex_magnitude_l124_12479

theorem complex_magnitude (z : ℂ) : z = (2 - I) / (1 + I) → Complex.abs z = Real.sqrt 10 / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l124_12479


namespace NUMINAMATH_CALUDE_wall_bricks_proof_l124_12457

/-- The time (in hours) it takes Alice to build the wall alone -/
def alice_time : ℝ := 8

/-- The time (in hours) it takes Bob to build the wall alone -/
def bob_time : ℝ := 12

/-- The decrease in productivity (in bricks per hour) when Alice and Bob work together -/
def productivity_decrease : ℝ := 15

/-- The time (in hours) it takes Alice and Bob to build the wall together -/
def combined_time : ℝ := 6

/-- The number of bricks in the wall -/
def wall_bricks : ℝ := 360

theorem wall_bricks_proof :
  let alice_rate := wall_bricks / alice_time
  let bob_rate := wall_bricks / bob_time
  let combined_rate := alice_rate + bob_rate - productivity_decrease
  combined_rate * combined_time = wall_bricks := by
  sorry

#check wall_bricks_proof

end NUMINAMATH_CALUDE_wall_bricks_proof_l124_12457


namespace NUMINAMATH_CALUDE_polynomial_simplification_l124_12473

theorem polynomial_simplification (x : ℝ) :
  (2 * x^5 - 3 * x^4 + x^3 + 5 * x^2 - 8 * x + 15) + (-5 * x^4 - 2 * x^3 + 3 * x^2 + 8 * x + 9) =
  2 * x^5 - 8 * x^4 - x^3 + 8 * x^2 + 24 := by
    sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l124_12473


namespace NUMINAMATH_CALUDE_cos_300_degrees_l124_12441

theorem cos_300_degrees : Real.cos (300 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_300_degrees_l124_12441


namespace NUMINAMATH_CALUDE_min_phi_for_odd_function_l124_12442

open Real

theorem min_phi_for_odd_function (φ : ℝ) : 
  (φ > 0 ∧ 
   (∀ x, cos (π * x - π * φ - π / 3) = -cos (π * (-x) - π * φ - π / 3))) 
  ↔ 
  φ = 1 / 6 := by
sorry

end NUMINAMATH_CALUDE_min_phi_for_odd_function_l124_12442


namespace NUMINAMATH_CALUDE_fair_distribution_exists_l124_12470

/-- Represents a piece of ham with its value according to the store scale -/
structure HamPiece where
  value : ℕ

/-- Represents a woman and her belief about the ham's value -/
inductive Woman
  | TrustsHomeScales
  | TrustsStoreScales
  | BelievesEqual

/-- Represents the distribution of ham pieces to women -/
def Distribution := Woman → HamPiece

/-- Checks if a distribution is fair according to each woman's belief -/
def is_fair_distribution (d : Distribution) : Prop :=
  (d Woman.TrustsHomeScales).value ≥ 15 ∧
  (d Woman.TrustsStoreScales).value ≥ 15 ∧
  (d Woman.BelievesEqual).value > 0

/-- The main theorem stating that a fair distribution exists -/
theorem fair_distribution_exists : ∃ (d : Distribution), is_fair_distribution d := by
  sorry

end NUMINAMATH_CALUDE_fair_distribution_exists_l124_12470


namespace NUMINAMATH_CALUDE_greatest_integer_less_than_negative_seventeen_thirds_l124_12446

theorem greatest_integer_less_than_negative_seventeen_thirds :
  ⌊-17/3⌋ = -6 :=
sorry

end NUMINAMATH_CALUDE_greatest_integer_less_than_negative_seventeen_thirds_l124_12446


namespace NUMINAMATH_CALUDE_function_value_theorem_l124_12456

/-- A function f(x) = a(x+2)^2 + 3 passing through points (-2, 3) and (0, 7) -/
def f (a : ℝ) (x : ℝ) : ℝ := a * (x + 2)^2 + 3

/-- The theorem stating that given the conditions, a+3a+2 equals 6 -/
theorem function_value_theorem (a : ℝ) :
  f a (-2) = 3 ∧ f a 0 = 7 → a + 3*a + 2 = 6 := by
  sorry


end NUMINAMATH_CALUDE_function_value_theorem_l124_12456


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l124_12454

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_sum
  (a : ℕ → ℝ)
  (h_geom : geometric_sequence a)
  (h_pos : a 1 > 0)
  (h_sum : a 4 + a 7 = 2)
  (h_prod : a 5 * a 6 = -8) :
  a 1 + a 4 + a 7 + a 10 = -5 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l124_12454


namespace NUMINAMATH_CALUDE_rugby_team_lineup_count_l124_12401

/-- The number of ways to form a team lineup -/
def team_lineup_ways (total_members : ℕ) (specialized_kickers : ℕ) (lineup_size : ℕ) : ℕ :=
  specialized_kickers * (Nat.choose (total_members - 1) (lineup_size - 1))

/-- Theorem: The number of ways to form the team lineup is 151164 -/
theorem rugby_team_lineup_count :
  team_lineup_ways 20 2 9 = 151164 := by
  sorry

end NUMINAMATH_CALUDE_rugby_team_lineup_count_l124_12401


namespace NUMINAMATH_CALUDE_simple_interest_theorem_l124_12408

def simple_interest_problem (principal rate time : ℝ) : Prop :=
  let simple_interest := principal * rate * time / 100
  principal - simple_interest = 2080

theorem simple_interest_theorem :
  simple_interest_problem 2600 4 5 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_theorem_l124_12408


namespace NUMINAMATH_CALUDE_sin_300_degrees_l124_12494

theorem sin_300_degrees : Real.sin (300 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_300_degrees_l124_12494


namespace NUMINAMATH_CALUDE_pyramid_frustum_volume_l124_12488

/-- Given a square pyramid with base edge s and altitude h, 
    if a smaller similar pyramid with altitude h/3 is removed from the apex, 
    the volume of the remaining frustum is 26/27 of the original pyramid's volume. -/
theorem pyramid_frustum_volume 
  (s h : ℝ) 
  (h_pos : 0 < h) 
  (s_pos : 0 < s) : 
  let v_original := (1 / 3) * s^2 * h
  let v_smaller := (1 / 3) * (s / 3)^2 * (h / 3)
  let v_frustum := v_original - v_smaller
  v_frustum = (26 / 27) * v_original := by
  sorry

end NUMINAMATH_CALUDE_pyramid_frustum_volume_l124_12488


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l124_12403

/-- The eccentricity of a hyperbola with the given condition -/
theorem hyperbola_eccentricity (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) : 
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) →
  (b = (a + c) / 2) →
  (c^2 = a^2 + b^2) →
  (c / a : ℝ) = 5/3 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l124_12403


namespace NUMINAMATH_CALUDE_two_color_rectangle_exists_l124_12491

/-- A color type with two possible values -/
inductive Color
| Red
| Blue

/-- A point in a 2D grid -/
structure Point where
  x : Nat
  y : Nat

/-- A coloring function that assigns a color to each point in a grid -/
def Coloring := Point → Color

/-- A rectangle defined by its four vertices -/
structure Rectangle where
  topLeft : Point
  topRight : Point
  bottomLeft : Point
  bottomRight : Point

/-- Predicate to check if all vertices of a rectangle have the same color -/
def sameColorRectangle (c : Coloring) (r : Rectangle) : Prop :=
  c r.topLeft = c r.topRight ∧
  c r.topLeft = c r.bottomLeft ∧
  c r.topLeft = c r.bottomRight

/-- Theorem stating that in any 7x3 grid colored with two colors,
    there exists a rectangle with vertices of the same color -/
theorem two_color_rectangle_exists :
  ∀ (c : Coloring),
  (∀ (p : Point), p.x < 7 ∧ p.y < 3 → (c p = Color.Red ∨ c p = Color.Blue)) →
  ∃ (r : Rectangle),
    r.topLeft.x < 7 ∧ r.topLeft.y < 3 ∧
    r.topRight.x < 7 ∧ r.topRight.y < 3 ∧
    r.bottomLeft.x < 7 ∧ r.bottomLeft.y < 3 ∧
    r.bottomRight.x < 7 ∧ r.bottomRight.y < 3 ∧
    sameColorRectangle c r :=
by
  sorry


end NUMINAMATH_CALUDE_two_color_rectangle_exists_l124_12491
