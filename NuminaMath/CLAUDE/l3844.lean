import Mathlib

namespace NUMINAMATH_CALUDE_female_democrats_count_l3844_384462

theorem female_democrats_count (total : ℕ) (male female : ℕ) (male_democrats female_democrats : ℕ) :
  total = 870 →
  male + female = total →
  female_democrats = female / 2 →
  male_democrats = male / 4 →
  female_democrats + male_democrats = total / 3 →
  female_democrats = 145 := by
  sorry

end NUMINAMATH_CALUDE_female_democrats_count_l3844_384462


namespace NUMINAMATH_CALUDE_max_difference_averages_l3844_384483

theorem max_difference_averages (x y : ℝ) (hx : 4 ≤ x ∧ x ≤ 100) (hy : 4 ≤ y ∧ y ≤ 100) :
  ∃ (z : ℝ), z = |((x + y) / 2) - ((x + 2 * y) / 3)| ∧
  z ≤ 16 ∧
  ∃ (a b : ℝ), (4 ≤ a ∧ a ≤ 100) ∧ (4 ≤ b ∧ b ≤ 100) ∧
    |((a + b) / 2) - ((a + 2 * b) / 3)| = 16 :=
by sorry

end NUMINAMATH_CALUDE_max_difference_averages_l3844_384483


namespace NUMINAMATH_CALUDE_londolozi_lions_growth_l3844_384404

/-- The number of lion cubs born per month in Londolozi -/
def cubs_per_month : ℕ := sorry

/-- The initial number of lions in Londolozi -/
def initial_lions : ℕ := 100

/-- The number of months in a year -/
def months_in_year : ℕ := 12

/-- The number of lions that die per month -/
def lions_die_per_month : ℕ := 1

/-- The number of lions after one year -/
def lions_after_year : ℕ := 148

theorem londolozi_lions_growth :
  cubs_per_month * months_in_year - lions_die_per_month * months_in_year + initial_lions = lions_after_year ∧
  cubs_per_month = 5 := by sorry

end NUMINAMATH_CALUDE_londolozi_lions_growth_l3844_384404


namespace NUMINAMATH_CALUDE_system_solution_set_l3844_384493

theorem system_solution_set (x : ℝ) : 
  (x - 1 < 1 ∧ x + 3 > 0) ↔ (-3 < x ∧ x < 2) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_set_l3844_384493


namespace NUMINAMATH_CALUDE_modified_rectangle_remaining_length_l3844_384496

/-- The total length of remaining segments after modifying a rectangle --/
def remaining_length (height width top_right_removed middle_left_removed bottom_removed top_left_removed : ℕ) : ℕ :=
  (height - middle_left_removed) + 
  (width - bottom_removed) + 
  (width - top_right_removed) + 
  (height - top_left_removed)

/-- Theorem stating the total length of remaining segments in the modified rectangle --/
theorem modified_rectangle_remaining_length :
  remaining_length 10 7 2 2 3 1 = 26 := by
  sorry

end NUMINAMATH_CALUDE_modified_rectangle_remaining_length_l3844_384496


namespace NUMINAMATH_CALUDE_max_value_product_l3844_384416

theorem max_value_product (x y z : ℝ) 
  (nonneg_x : 0 ≤ x) (nonneg_y : 0 ≤ y) (nonneg_z : 0 ≤ z) 
  (sum_eq_three : x + y + z = 3) : 
  (x^3 - x*y^2 + y^3) * (x^3 - x*z^2 + z^3) * (y^3 - y*z^2 + z^3) ≤ 2916/2187 := by
  sorry

end NUMINAMATH_CALUDE_max_value_product_l3844_384416


namespace NUMINAMATH_CALUDE_simplified_expression_l3844_384425

theorem simplified_expression :
  (3 * Real.sqrt 8) / (Real.sqrt 3 + Real.sqrt 5 + Real.sqrt 7) =
  (108 * (Real.sqrt 10 + Real.sqrt 14 - Real.sqrt 6 - Real.sqrt 490)) / (-59) :=
by sorry

end NUMINAMATH_CALUDE_simplified_expression_l3844_384425


namespace NUMINAMATH_CALUDE_fox_coins_l3844_384413

def bridge_crossings (initial_coins : ℕ) : ℕ → ℕ
  | 0 => initial_coins + 10
  | n + 1 => (2 * bridge_crossings initial_coins n) - 50

theorem fox_coins (x : ℕ) : x = 37 → bridge_crossings x 4 = 0 := by
  sorry

end NUMINAMATH_CALUDE_fox_coins_l3844_384413


namespace NUMINAMATH_CALUDE_polynomial_division_quotient_l3844_384417

theorem polynomial_division_quotient : 
  let dividend : Polynomial ℚ := 8 * X^4 - 4 * X^3 + 3 * X^2 - 5 * X - 10
  let divisor : Polynomial ℚ := X^2 + 3 * X + 2
  let quotient : Polynomial ℚ := 8 * X^2 - 28 * X + 89
  dividend = divisor * quotient + (dividend.mod divisor) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_quotient_l3844_384417


namespace NUMINAMATH_CALUDE_floor_product_equality_l3844_384484

theorem floor_product_equality (Y : ℝ) : ⌊(0.3242 * Y)⌋ = 0.3242 * Y := by
  sorry

end NUMINAMATH_CALUDE_floor_product_equality_l3844_384484


namespace NUMINAMATH_CALUDE_smallest_composite_with_large_factors_l3844_384450

def is_composite (n : ℕ) : Prop := ∃ a b, 1 < a ∧ 1 < b ∧ n = a * b

def has_no_small_prime_factors (n : ℕ) : Prop := ∀ p, p < 20 → ¬(Nat.Prime p ∧ p ∣ n)

theorem smallest_composite_with_large_factors :
  ∃ n : ℕ, is_composite n ∧
           has_no_small_prime_factors n ∧
           (∀ m, m < n → ¬(is_composite m ∧ has_no_small_prime_factors m)) ∧
           n = 529 ∧
           520 < n ∧ n ≤ 530 :=
sorry

end NUMINAMATH_CALUDE_smallest_composite_with_large_factors_l3844_384450


namespace NUMINAMATH_CALUDE_dogsled_race_distance_l3844_384446

/-- The distance of the dogsled race course -/
def distance : ℝ := sorry

/-- The time taken by Team W to complete the course -/
def time_W : ℝ := sorry

/-- The time taken by Team A to complete the course -/
def time_A : ℝ := sorry

/-- The average speed of Team W -/
def speed_W : ℝ := 20

/-- The average speed of Team A -/
def speed_A : ℝ := speed_W + 5

theorem dogsled_race_distance :
  (time_A = time_W - 3) →
  (distance = speed_W * time_W) →
  (distance = speed_A * time_A) →
  distance = 300 := by sorry

end NUMINAMATH_CALUDE_dogsled_race_distance_l3844_384446


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l3844_384422

/-- Given a geometric sequence {a_n} with a_1 = 3 and a_4 = 24, prove that a_3 + a_4 + a_5 = 84 -/
theorem geometric_sequence_sum (a : ℕ → ℝ) (h_geom : ∀ n, a (n + 1) / a n = a 2 / a 1) 
  (h_a1 : a 1 = 3) (h_a4 : a 4 = 24) : a 3 + a 4 + a 5 = 84 := by
  sorry

#check geometric_sequence_sum

end NUMINAMATH_CALUDE_geometric_sequence_sum_l3844_384422


namespace NUMINAMATH_CALUDE_problem_solution_l3844_384420

theorem problem_solution (a b : ℝ) (h : |a - 1| + Real.sqrt (b + 2) = 0) : 
  (a + b) ^ 2022 = 1 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3844_384420


namespace NUMINAMATH_CALUDE_consecutive_integers_sum_l3844_384427

theorem consecutive_integers_sum (n : ℤ) : 
  (n - 1) * n * (n + 1) = -336 ∧ n < 0 → (n - 1) + n + (n + 1) = -21 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_sum_l3844_384427


namespace NUMINAMATH_CALUDE_nested_square_root_value_l3844_384456

theorem nested_square_root_value :
  ∃ x : ℝ, x = Real.sqrt (3 - x) ∧ x = (-1 + Real.sqrt 13) / 2 := by
  sorry

end NUMINAMATH_CALUDE_nested_square_root_value_l3844_384456


namespace NUMINAMATH_CALUDE_sequence_periodicity_l3844_384479

def is_periodic (a : ℕ → ℝ) (p : ℕ) : Prop :=
  ∃ k : ℕ, ∀ n ≥ k, a n = a (n + p)

def smallest_period (a : ℕ → ℝ) (p : ℕ) : Prop :=
  is_periodic a p ∧ ∀ q < p, ¬ is_periodic a q

theorem sequence_periodicity (a : ℕ → ℝ) 
  (h1 : ∃ n, a n ≠ 0)
  (h2 : ∀ n : ℕ, a (n + 2) = |a (n + 1)| - a n) :
  smallest_period a 9 :=
sorry

end NUMINAMATH_CALUDE_sequence_periodicity_l3844_384479


namespace NUMINAMATH_CALUDE_score_for_91_correct_out_of_100_l3844_384497

/-- Calculates the score for a test based on the number of correct responses and total questions. -/
def calculateScore (totalQuestions : ℕ) (correctResponses : ℕ) : ℤ :=
  correctResponses - 2 * (totalQuestions - correctResponses)

/-- Proves that for a 100-question test with 91 correct responses, the calculated score is 73. -/
theorem score_for_91_correct_out_of_100 :
  calculateScore 100 91 = 73 := by
  sorry

end NUMINAMATH_CALUDE_score_for_91_correct_out_of_100_l3844_384497


namespace NUMINAMATH_CALUDE_dove_population_growth_l3844_384476

theorem dove_population_growth (initial_doves : ℕ) (eggs_per_dove : ℕ) (hatch_rate : ℚ) : 
  initial_doves = 20 →
  eggs_per_dove = 3 →
  hatch_rate = 3/4 →
  initial_doves + (initial_doves * eggs_per_dove * hatch_rate).floor = 65 :=
by sorry

end NUMINAMATH_CALUDE_dove_population_growth_l3844_384476


namespace NUMINAMATH_CALUDE_cape_may_less_than_double_daytona_main_result_l3844_384486

/-- The number of shark sightings in Cape May -/
def cape_may_sightings : ℕ := 24

/-- The number of shark sightings in Daytona Beach -/
def daytona_beach_sightings : ℕ := 40 - cape_may_sightings

/-- The total number of shark sightings in both locations -/
def total_sightings : ℕ := 40

/-- Cape May has some less than double the number of shark sightings of Daytona Beach -/
theorem cape_may_less_than_double_daytona : cape_may_sightings < 2 * daytona_beach_sightings :=
sorry

/-- The difference between double the number of shark sightings in Daytona Beach and Cape May -/
def sightings_difference : ℕ := 2 * daytona_beach_sightings - cape_may_sightings

theorem main_result : sightings_difference = 8 := by
  sorry

end NUMINAMATH_CALUDE_cape_may_less_than_double_daytona_main_result_l3844_384486


namespace NUMINAMATH_CALUDE_fraction_product_squared_main_theorem_l3844_384461

theorem fraction_product_squared (a b c d : ℚ) : 
  (a / b) ^ 2 * (c / d) ^ 2 = (a * c / (b * d)) ^ 2 :=
by sorry

theorem main_theorem : (6 / 7) ^ 2 * (1 / 2) ^ 2 = 9 / 49 :=
by sorry

end NUMINAMATH_CALUDE_fraction_product_squared_main_theorem_l3844_384461


namespace NUMINAMATH_CALUDE_jane_crayons_l3844_384495

/-- The number of crayons Jane ends up with after a hippopotamus eats some. -/
def crayons_left (initial : ℕ) (eaten : ℕ) : ℕ :=
  initial - eaten

/-- Theorem: If Jane starts with 87 crayons and 7 are eaten by a hippopotamus,
    she will end up with 80 crayons. -/
theorem jane_crayons : crayons_left 87 7 = 80 := by
  sorry

end NUMINAMATH_CALUDE_jane_crayons_l3844_384495


namespace NUMINAMATH_CALUDE_lisa_pizza_meat_distribution_l3844_384455

/-- The number of pieces of meat on each slice of Lisa's pizza --/
def pieces_per_slice : ℕ :=
  let pepperoni : ℕ := 30
  let ham : ℕ := 2 * pepperoni
  let sausage : ℕ := pepperoni + 12
  let total_meat : ℕ := pepperoni + ham + sausage
  let num_slices : ℕ := 6
  total_meat / num_slices

theorem lisa_pizza_meat_distribution :
  pieces_per_slice = 22 := by
  sorry

end NUMINAMATH_CALUDE_lisa_pizza_meat_distribution_l3844_384455


namespace NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l3844_384452

theorem solution_set_quadratic_inequality :
  let f : ℝ → ℝ := λ x => -3 * x^2 + 2 * x + 8
  {x : ℝ | f x > 0} = Set.Ioo (-4/3 : ℝ) 2 := by sorry

end NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l3844_384452


namespace NUMINAMATH_CALUDE_roots_sum_and_product_inequality_l3844_384426

noncomputable def f (x : ℝ) : ℝ := x * Real.exp (x + 1)

theorem roots_sum_and_product_inequality 
  (x₁ x₂ : ℝ) 
  (h_pos₁ : x₁ > 0) 
  (h_pos₂ : x₂ > 0) 
  (h_distinct : x₁ ≠ x₂) 
  (h_root₁ : f x₁ = 3 * Real.exp 1 * x₁ + 3 * Real.exp 1 * Real.log x₁)
  (h_root₂ : f x₂ = 3 * Real.exp 1 * x₂ + 3 * Real.exp 1 * Real.log x₂) :
  x₁ + x₂ + Real.log (x₁ * x₂) > 2 := by
  sorry

end NUMINAMATH_CALUDE_roots_sum_and_product_inequality_l3844_384426


namespace NUMINAMATH_CALUDE_house_transaction_loss_l3844_384430

def initial_value : ℝ := 12000
def loss_percentage : ℝ := 0.15
def gain_percentage : ℝ := 0.20

def first_transaction (value : ℝ) (loss : ℝ) : ℝ :=
  value * (1 - loss)

def second_transaction (value : ℝ) (gain : ℝ) : ℝ :=
  value * (1 + gain)

theorem house_transaction_loss :
  second_transaction (first_transaction initial_value loss_percentage) gain_percentage - initial_value = 240 := by
  sorry

end NUMINAMATH_CALUDE_house_transaction_loss_l3844_384430


namespace NUMINAMATH_CALUDE_solve_dimes_problem_l3844_384478

def dimes_problem (initial_dimes : ℕ) (given_to_mother : ℕ) (final_dimes : ℕ) : Prop :=
  ∃ (dimes_from_dad : ℕ),
    initial_dimes - given_to_mother + dimes_from_dad = final_dimes

theorem solve_dimes_problem :
  dimes_problem 7 4 11 → ∃ (dimes_from_dad : ℕ), dimes_from_dad = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_solve_dimes_problem_l3844_384478


namespace NUMINAMATH_CALUDE_sum_of_even_factors_420_l3844_384414

def sumOfEvenFactors (n : ℕ) : ℕ := sorry

theorem sum_of_even_factors_420 :
  sumOfEvenFactors 420 = 1152 := by sorry

end NUMINAMATH_CALUDE_sum_of_even_factors_420_l3844_384414


namespace NUMINAMATH_CALUDE_bridge_length_calculation_l3844_384443

/-- Calculates the length of a bridge given train parameters --/
theorem bridge_length_calculation (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 135 →
  train_speed_kmh = 45 →
  crossing_time = 30 →
  (train_speed_kmh * 1000 / 3600) * crossing_time - train_length = 240 := by
  sorry

end NUMINAMATH_CALUDE_bridge_length_calculation_l3844_384443


namespace NUMINAMATH_CALUDE_sqrt_square_equals_abs_l3844_384480

theorem sqrt_square_equals_abs (a : ℝ) : Real.sqrt (a^2) = |a| := by
  sorry

end NUMINAMATH_CALUDE_sqrt_square_equals_abs_l3844_384480


namespace NUMINAMATH_CALUDE_brent_candy_count_l3844_384468

/-- Calculates the total number of candy pieces Brent has left after trick-or-treating and giving some to his sister. -/
def total_candy_left (kitkat : ℕ) (nerds : ℕ) (initial_lollipops : ℕ) (baby_ruth : ℕ) (given_lollipops : ℕ) : ℕ :=
  let hershey := 3 * kitkat
  let reese := baby_ruth / 2
  let remaining_lollipops := initial_lollipops - given_lollipops
  kitkat + hershey + nerds + baby_ruth + reese + remaining_lollipops

theorem brent_candy_count : 
  total_candy_left 5 8 11 10 5 = 49 := by
  sorry

end NUMINAMATH_CALUDE_brent_candy_count_l3844_384468


namespace NUMINAMATH_CALUDE_mixture_temperature_swap_l3844_384441

theorem mixture_temperature_swap (a b c : ℝ) :
  let x := a + b - c
  ∃ (m_a m_b : ℝ), m_a > 0 ∧ m_b > 0 ∧
    (m_a * (a - c) + m_b * (b - c) = 0) ∧
    (m_b * (a - x) + m_a * (b - x) = 0) :=
by sorry

end NUMINAMATH_CALUDE_mixture_temperature_swap_l3844_384441


namespace NUMINAMATH_CALUDE_square_perimeter_from_rectangle_perimeter_l3844_384482

/-- Given a square divided into four congruent rectangles, if the perimeter of each rectangle is 32 inches, then the perimeter of the square is 51.2 inches. -/
theorem square_perimeter_from_rectangle_perimeter (s : ℝ) 
  (h1 : s > 0) 
  (h2 : 2 * s + 2 * (s / 4) = 32) : 
  4 * s = 51.2 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_from_rectangle_perimeter_l3844_384482


namespace NUMINAMATH_CALUDE_simplify_expression_l3844_384428

theorem simplify_expression (a b : ℝ) : (2 * a^2)^3 * (-a * b) = -8 * a^7 * b := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3844_384428


namespace NUMINAMATH_CALUDE_adams_father_deposit_l3844_384402

/-- Calculates the total amount after a given number of years, given an initial deposit,
    annual interest rate, and immediate withdrawal of interest. -/
def totalAmount (initialDeposit : ℝ) (interestRate : ℝ) (years : ℝ) : ℝ :=
  initialDeposit + (initialDeposit * interestRate * years)

/-- Proves that given an initial deposit of $2000 with an 8% annual interest rate,
    where interest is withdrawn immediately upon receipt, the total amount after 2.5 years
    will be $2400. -/
theorem adams_father_deposit : totalAmount 2000 0.08 2.5 = 2400 := by
  sorry

end NUMINAMATH_CALUDE_adams_father_deposit_l3844_384402


namespace NUMINAMATH_CALUDE_probability_three_same_tunes_l3844_384470

/-- A defective toy train that produces only two different tunes at random -/
structure DefectiveToyTrain where
  tunes : Fin 2

/-- The probability of a specific sequence of tunes occurring -/
def probability_of_sequence (n : ℕ) : ℚ :=
  (1 / 2) ^ n

/-- The probability of producing n music tunes of the same type in a row -/
def probability_same_tune (n : ℕ) : ℚ :=
  2 * probability_of_sequence n

theorem probability_three_same_tunes :
  probability_same_tune 3 = 1 / 4 := by sorry

end NUMINAMATH_CALUDE_probability_three_same_tunes_l3844_384470


namespace NUMINAMATH_CALUDE_target_number_scientific_notation_l3844_384458

/-- The number we want to express in scientific notation -/
def target_number : ℕ := 1200000000

/-- Definition of scientific notation for positive integers -/
def scientific_notation (n : ℕ) (a : ℚ) (b : ℤ) : Prop :=
  (1 ≤ a) ∧ (a < 10) ∧ (n = (a * 10^b).floor)

/-- Theorem stating that 1,200,000,000 is equal to 1.2 × 10^9 in scientific notation -/
theorem target_number_scientific_notation :
  scientific_notation target_number (12/10) 9 := by
  sorry

end NUMINAMATH_CALUDE_target_number_scientific_notation_l3844_384458


namespace NUMINAMATH_CALUDE_honey_servings_per_ounce_l3844_384440

/-- Represents the number of servings of honey per ounce -/
def servings_per_ounce : ℕ := sorry

/-- Represents the number of servings used per cup of tea -/
def servings_per_cup : ℕ := 1

/-- Represents the number of cups of tea consumed per night -/
def cups_per_night : ℕ := 2

/-- Represents the size of the honey container in ounces -/
def container_size : ℕ := 16

/-- Represents the number of nights the honey lasts -/
def nights_lasted : ℕ := 48

theorem honey_servings_per_ounce :
  servings_per_ounce = 6 :=
sorry

end NUMINAMATH_CALUDE_honey_servings_per_ounce_l3844_384440


namespace NUMINAMATH_CALUDE_probability_of_B_is_one_fourth_l3844_384451

/-- The probability of choosing a specific letter from a bag of letters -/
def probability_of_letter (total_letters : ℕ) (target_letters : ℕ) : ℚ :=
  target_letters / total_letters

/-- The bag contains 8 letters in total -/
def total_letters : ℕ := 8

/-- The bag contains 2 B's -/
def number_of_Bs : ℕ := 2

/-- The probability of choosing a B is 1/4 -/
theorem probability_of_B_is_one_fourth :
  probability_of_letter total_letters number_of_Bs = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_B_is_one_fourth_l3844_384451


namespace NUMINAMATH_CALUDE_ice_cream_arrangements_l3844_384472

theorem ice_cream_arrangements (n : ℕ) (h : n = 5) : Nat.factorial n = 120 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_arrangements_l3844_384472


namespace NUMINAMATH_CALUDE_total_squares_5x5_with_2_removed_l3844_384405

/-- Represents a square grid --/
structure Grid :=
  (size : ℕ)
  (removed : ℕ)

/-- Calculates the total number of squares in a grid --/
def total_squares (g : Grid) : ℕ :=
  sorry

/-- The theorem to prove --/
theorem total_squares_5x5_with_2_removed :
  ∃ (g : Grid), g.size = 5 ∧ g.removed = 2 ∧ total_squares g = 55 :=
sorry

end NUMINAMATH_CALUDE_total_squares_5x5_with_2_removed_l3844_384405


namespace NUMINAMATH_CALUDE_range_of_a_for_false_quadratic_inequality_l3844_384409

theorem range_of_a_for_false_quadratic_inequality :
  (∃ a : ℝ, ∀ x : ℝ, x^2 - a*x + 1 > 0) ↔ 
  (∃ a : ℝ, -2 < a ∧ a < 2) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_for_false_quadratic_inequality_l3844_384409


namespace NUMINAMATH_CALUDE_midpoint_trajectory_l3844_384433

/-- The trajectory of the midpoint of a line segment between a point on a circle and a fixed point -/
theorem midpoint_trajectory (x₀ y₀ x y : ℝ) : 
  x₀^2 + y₀^2 = 4 →  -- P is on the circle x^2 + y^2 = 4
  x = (x₀ + 8) / 2 →  -- x-coordinate of midpoint M
  y = y₀ / 2 →  -- y-coordinate of midpoint M
  (x - 4)^2 + y^2 = 1 :=  -- Trajectory equation
by sorry

end NUMINAMATH_CALUDE_midpoint_trajectory_l3844_384433


namespace NUMINAMATH_CALUDE_min_value_expression_l3844_384463

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x + 1/y) * (x + 1/y - 100) + (y + 1/x) * (y + 1/x - 100) ≥ -2500 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l3844_384463


namespace NUMINAMATH_CALUDE_quadratic_transformation_l3844_384447

theorem quadratic_transformation (x : ℝ) : 
  (2 * x^2 - 3 * x + 1 = 0) ↔ ((x - 3/4)^2 = 1/16) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_transformation_l3844_384447


namespace NUMINAMATH_CALUDE_liams_numbers_l3844_384439

theorem liams_numbers (x y : ℤ) : 
  (3 * x + 2 * y = 75) →  -- Sum of five numbers is 75
  (x = 15) →              -- The number written three times is 15
  (x * y % 5 = 0) →       -- Product of the two numbers is a multiple of 5
  (y = 15) :=             -- The other number (written twice) is 15
by sorry

end NUMINAMATH_CALUDE_liams_numbers_l3844_384439


namespace NUMINAMATH_CALUDE_max_area_rectangular_enclosure_l3844_384418

/-- The maximum area of a rectangular enclosure with given constraints -/
theorem max_area_rectangular_enclosure 
  (perimeter : ℝ) 
  (min_length : ℝ) 
  (min_width : ℝ) 
  (h_perimeter : perimeter = 400) 
  (h_min_length : min_length = 100) 
  (h_min_width : min_width = 50) : 
  ∃ (length width : ℝ), 
    length ≥ min_length ∧ 
    width ≥ min_width ∧ 
    2 * (length + width) = perimeter ∧ 
    ∀ (l w : ℝ), 
      l ≥ min_length → 
      w ≥ min_width → 
      2 * (l + w) = perimeter → 
      length * width ≥ l * w ∧ 
      length * width = 10000 :=
sorry

end NUMINAMATH_CALUDE_max_area_rectangular_enclosure_l3844_384418


namespace NUMINAMATH_CALUDE_first_player_wins_petya_wins_1000x2020_l3844_384466

/-- Represents the state of the rectangular grid game -/
structure GameState where
  m : ℕ+
  n : ℕ+

/-- Determines if a player has a winning strategy based on the game state -/
def has_winning_strategy (state : GameState) : Prop :=
  ∃ (a b : ℕ), state.m = 2^a * (2 * state.m.val + 1) ∧ state.n = 2^b * (2 * state.n.val + 1) ∧ a ≠ b

/-- Theorem stating the winning condition for the first player -/
theorem first_player_wins (initial_state : GameState) :
  has_winning_strategy initial_state ↔ 
  ∀ (strategy : GameState → GameState), 
    ∃ (counter_strategy : GameState → GameState),
      ∀ (game_length : ℕ),
        (game_length > 0 ∧ has_winning_strategy (counter_strategy (strategy initial_state))) ∨
        (game_length = 0 ∧ ¬ has_winning_strategy initial_state) :=
by sorry

/-- The specific case for the 1000 × 2020 grid -/
theorem petya_wins_1000x2020 :
  has_winning_strategy { m := 1000, n := 2020 } :=
by sorry

end NUMINAMATH_CALUDE_first_player_wins_petya_wins_1000x2020_l3844_384466


namespace NUMINAMATH_CALUDE_typist_salary_problem_l3844_384498

theorem typist_salary_problem (original_salary : ℝ) : 
  let increased_salary := original_salary * 1.1
  let final_salary := increased_salary * 0.95
  final_salary = 5225 →
  original_salary = 5000 := by
sorry

end NUMINAMATH_CALUDE_typist_salary_problem_l3844_384498


namespace NUMINAMATH_CALUDE_f_properties_l3844_384442

noncomputable section

def f (x : ℝ) : ℝ := (2*x - x^2) * Real.exp x

theorem f_properties :
  (∀ x, f x > 0 ↔ 0 < x ∧ x < 2) ∧
  (∃ max_x, ∀ x, f x ≤ f max_x ∧ max_x = Real.sqrt 2) ∧
  (¬ ∃ min_x, ∀ x, f min_x ≤ f x) ∧
  (∃ max_x, ∀ x, f x ≤ f max_x) :=
sorry

end

end NUMINAMATH_CALUDE_f_properties_l3844_384442


namespace NUMINAMATH_CALUDE_closest_point_and_area_l3844_384475

def parabola_C (x y : ℝ) : Prop := x^2 = 4*y

def line_l (x y : ℝ) : Prop := y = -x - 2

def point_P : ℝ × ℝ := (-2, 1)

def focus_C : ℝ × ℝ := (0, 1)

def is_centroid (G A B C : ℝ × ℝ) : Prop :=
  G.1 = (A.1 + B.1 + C.1) / 3 ∧ G.2 = (A.2 + B.2 + C.2) / 3

theorem closest_point_and_area :
  ∀ (A B : ℝ × ℝ),
    parabola_C point_P.1 point_P.2 →
    (∀ (Q : ℝ × ℝ), parabola_C Q.1 Q.2 →
      ∃ (d_P d_Q : ℝ),
        d_P = abs (point_P.2 + point_P.1 + 2) / Real.sqrt 2 ∧
        d_Q = abs (Q.2 + Q.1 + 2) / Real.sqrt 2 ∧
        d_P ≤ d_Q) →
    parabola_C A.1 A.2 →
    parabola_C B.1 B.2 →
    is_centroid focus_C point_P A B →
    ∃ (area : ℝ), area = (3 * Real.sqrt 3) / 2 := by sorry

end NUMINAMATH_CALUDE_closest_point_and_area_l3844_384475


namespace NUMINAMATH_CALUDE_min_p_plus_q_l3844_384494

theorem min_p_plus_q (p q : ℕ+) (h : 98 * p = q^3) : 
  ∀ (p' q' : ℕ+), 98 * p' = q'^3 → p + q ≤ p' + q' :=
by
  sorry

#check min_p_plus_q

end NUMINAMATH_CALUDE_min_p_plus_q_l3844_384494


namespace NUMINAMATH_CALUDE_angle_conversion_l3844_384477

theorem angle_conversion :
  ∃ (k : ℤ) (α : ℝ), -1485 = k * 360 + α ∧ 0 ≤ α ∧ α < 360 :=
by
  use -5
  use 315
  sorry

end NUMINAMATH_CALUDE_angle_conversion_l3844_384477


namespace NUMINAMATH_CALUDE_jan_extra_distance_l3844_384457

/-- Represents the driving scenario of Ian, Han, and Jan -/
structure DrivingScenario where
  ian_time : ℝ
  ian_speed : ℝ
  han_time : ℝ
  han_speed : ℝ
  jan_time : ℝ
  jan_speed : ℝ
  han_extra_distance : ℝ

/-- The conditions of the driving scenario -/
def scenario_conditions (s : DrivingScenario) : Prop :=
  s.han_time = s.ian_time + 2 ∧
  s.han_speed = s.ian_speed + 10 ∧
  s.jan_time = s.ian_time + 3 ∧
  s.jan_speed = s.ian_speed + 15 ∧
  s.han_extra_distance = 120

/-- The theorem stating that Jan drove 195 miles more than Ian -/
theorem jan_extra_distance (s : DrivingScenario) 
  (h : scenario_conditions s) : 
  s.jan_speed * s.jan_time - s.ian_speed * s.ian_time = 195 :=
sorry


end NUMINAMATH_CALUDE_jan_extra_distance_l3844_384457


namespace NUMINAMATH_CALUDE_smallest_divisible_by_10_13_14_l3844_384460

theorem smallest_divisible_by_10_13_14 : ∃ (n : ℕ), n > 0 ∧ 
  10 ∣ n ∧ 13 ∣ n ∧ 14 ∣ n ∧ 
  ∀ (m : ℕ), m > 0 → 10 ∣ m → 13 ∣ m → 14 ∣ m → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_10_13_14_l3844_384460


namespace NUMINAMATH_CALUDE_video_streaming_cost_l3844_384444

/-- Represents the monthly cost of a video streaming subscription -/
def monthly_cost : ℝ := 14

/-- Represents the number of months in a year -/
def months_in_year : ℕ := 12

/-- Represents the total cost paid by one person for a year -/
def total_cost_per_person : ℝ := 84

theorem video_streaming_cost : 
  monthly_cost * months_in_year = 2 * total_cost_per_person :=
by sorry

end NUMINAMATH_CALUDE_video_streaming_cost_l3844_384444


namespace NUMINAMATH_CALUDE_floor_times_self_72_l3844_384412

theorem floor_times_self_72 :
  ∃ (x : ℝ), x > 0 ∧ (Int.floor x : ℝ) * x = 72 ∧ x = 9 := by
  sorry

end NUMINAMATH_CALUDE_floor_times_self_72_l3844_384412


namespace NUMINAMATH_CALUDE_parallelepiped_dimensions_l3844_384437

theorem parallelepiped_dimensions (n : ℕ) (h1 : n > 6) : 
  (n - 2 : ℕ) > 0 ∧ (n - 4 : ℕ) > 0 ∧
  (n - 2 : ℕ) * (n - 4 : ℕ) * (n - 6 : ℕ) = 2 * n * (n - 2 : ℕ) * (n - 4 : ℕ) / 3 →
  n = 18 := by sorry

end NUMINAMATH_CALUDE_parallelepiped_dimensions_l3844_384437


namespace NUMINAMATH_CALUDE_egg_acceptance_ratio_l3844_384401

/-- Represents the egg processing plant scenario -/
structure EggPlant where
  total_eggs : ℕ  -- Total number of eggs processed per day
  normal_accepted : ℕ  -- Number of eggs normally accepted in a batch
  normal_rejected : ℕ  -- Number of eggs normally rejected in a batch
  additional_accepted : ℕ  -- Additional eggs accepted on the particular day

/-- Defines the conditions of the egg processing plant -/
def egg_plant_conditions (plant : EggPlant) : Prop :=
  plant.total_eggs = 400 ∧
  plant.normal_accepted = 96 ∧
  plant.normal_rejected = 4 ∧
  plant.additional_accepted = 12

/-- Calculates the ratio of accepted to rejected eggs on the particular day -/
def acceptance_ratio (plant : EggPlant) : ℚ :=
  let normal_batches := plant.total_eggs / (plant.normal_accepted + plant.normal_rejected)
  let accepted := normal_batches * plant.normal_accepted + plant.additional_accepted
  let rejected := plant.total_eggs - accepted
  accepted / rejected

/-- Theorem stating that under the given conditions, the acceptance ratio is 99:1 -/
theorem egg_acceptance_ratio (plant : EggPlant) 
  (h : egg_plant_conditions plant) : acceptance_ratio plant = 99 / 1 := by
  sorry


end NUMINAMATH_CALUDE_egg_acceptance_ratio_l3844_384401


namespace NUMINAMATH_CALUDE_equation_solutions_l3844_384465

/-- The equation from the original problem -/
def original_equation (x : ℝ) : Prop :=
  (15 * x - x^2) / (x + 2) * (x + (15 - x) / (x + 2)) = 48

/-- The theorem stating the solutions to the equation -/
theorem equation_solutions :
  ∀ x : ℝ, original_equation x ↔ (x = 6 ∨ x = 8) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l3844_384465


namespace NUMINAMATH_CALUDE_unique_solution_iff_a_in_open_interval_l3844_384421

/-- The system of equations has exactly one solution if and only if 0 < a < 4 -/
theorem unique_solution_iff_a_in_open_interval (a : ℝ) :
  (∃! x y z : ℝ, x + y + z = 0 ∧ x*y + y*z + a*z*x = 0) ↔ 0 < a ∧ a < 4 :=
sorry

end NUMINAMATH_CALUDE_unique_solution_iff_a_in_open_interval_l3844_384421


namespace NUMINAMATH_CALUDE_count_valid_primes_l3844_384487

def isSubnumber (n m : ℕ) : Prop :=
  ∃ (k l : ℕ), n = (m / 10^k) % (10^l)

def hasNonPrimeSubnumber (n : ℕ) : Prop :=
  ∃ (m : ℕ), isSubnumber m n ∧ m > 1 ∧ ¬ Nat.Prime m

def validPrime (n : ℕ) : Prop :=
  Nat.Prime n ∧ n < 1000000000 ∧ ¬ hasNonPrimeSubnumber n

theorem count_valid_primes :
  ∃! (s : Finset ℕ), (∀ n ∈ s, validPrime n) ∧ s.card = 9 :=
sorry

end NUMINAMATH_CALUDE_count_valid_primes_l3844_384487


namespace NUMINAMATH_CALUDE_approximate_profit_percent_l3844_384431

-- Define the selling price and cost price
def selling_price : Float := 2552.36
def cost_price : Float := 2400.0

-- Define the profit amount
def profit_amount : Float := selling_price - cost_price

-- Define the profit percent
def profit_percent : Float := (profit_amount / cost_price) * 100

-- Theorem to prove the approximate profit percent
theorem approximate_profit_percent :
  (Float.round (profit_percent * 100) / 100) = 6.35 := by
  sorry

end NUMINAMATH_CALUDE_approximate_profit_percent_l3844_384431


namespace NUMINAMATH_CALUDE_conference_handshakes_l3844_384434

/-- The number of unique handshakes in a circular seating arrangement --/
def unique_handshakes (n : ℕ) : ℕ := n

/-- Theorem: In a circular seating arrangement with 30 people, 
    where each person shakes hands only with their immediate neighbors, 
    the number of unique handshakes is equal to 30. --/
theorem conference_handshakes : 
  unique_handshakes 30 = 30 := by
  sorry

end NUMINAMATH_CALUDE_conference_handshakes_l3844_384434


namespace NUMINAMATH_CALUDE_proposition_3_proposition_4_l3844_384429

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel and perpendicular relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Line → Prop)
variable (planeParallel : Plane → Plane → Prop)
variable (planePerpendicular : Plane → Plane → Prop)
variable (lineParallelToPlane : Line → Plane → Prop)
variable (linePerpendicularToPlane : Line → Plane → Prop)

-- Notation
local infix:50 " ∥ " => parallel
local infix:50 " ⊥ " => perpendicular
local infix:50 " ∥ᵖ " => planeParallel
local infix:50 " ⊥ᵖ " => planePerpendicular
local infix:50 " ∥ᵖˡ " => lineParallelToPlane
local infix:50 " ⊥ᵖˡ " => linePerpendicularToPlane

-- Theorem statements
theorem proposition_3 (m n : Line) (α β : Plane) :
  m ⊥ᵖˡ α → n ∥ᵖˡ β → α ∥ᵖ β → m ⊥ n := by sorry

theorem proposition_4 (m n : Line) (α β : Plane) :
  m ⊥ᵖˡ α → n ⊥ᵖˡ β → α ⊥ᵖ β → m ⊥ n := by sorry

end NUMINAMATH_CALUDE_proposition_3_proposition_4_l3844_384429


namespace NUMINAMATH_CALUDE_line_contains_point_l3844_384464

/-- Proves that k = 11 for the line 3 - 3kx = 4y containing the point (1/3, -2) -/
theorem line_contains_point (k : ℝ) : 
  (3 - 3 * k * (1/3) = 4 * (-2)) → k = 11 := by
  sorry

end NUMINAMATH_CALUDE_line_contains_point_l3844_384464


namespace NUMINAMATH_CALUDE_shopping_money_l3844_384436

theorem shopping_money (initial_amount : ℝ) : 
  0.7 * initial_amount = 2800 → initial_amount = 4000 := by
  sorry

end NUMINAMATH_CALUDE_shopping_money_l3844_384436


namespace NUMINAMATH_CALUDE_common_factor_proof_l3844_384445

def expression (x y : ℝ) : ℝ := 9 * x^3 * y^2 + 12 * x^2 * y^3

def common_factor (x y : ℝ) : ℝ := 3 * x^2 * y^2

theorem common_factor_proof :
  ∀ x y : ℝ, ∃ k : ℝ, expression x y = common_factor x y * k :=
by sorry

end NUMINAMATH_CALUDE_common_factor_proof_l3844_384445


namespace NUMINAMATH_CALUDE_mistaken_quotient_l3844_384453

theorem mistaken_quotient (D : ℕ) : 
  D % 21 = 0 ∧ D / 21 = 20 → D / 12 = 35 := by
  sorry

end NUMINAMATH_CALUDE_mistaken_quotient_l3844_384453


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3844_384438

def A : Set ℝ := {x | -2 < x ∧ x < 4}
def B : Set ℝ := {2, 3, 4, 5}

theorem intersection_of_A_and_B : A ∩ B = {2, 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3844_384438


namespace NUMINAMATH_CALUDE_isosceles_triangle_count_l3844_384492

-- Define the geoboard as a square grid
structure Geoboard :=
  (size : ℕ)

-- Define a point on the geoboard
structure Point :=
  (x : ℕ)
  (y : ℕ)

-- Define a triangle on the geoboard
structure Triangle :=
  (A : Point)
  (B : Point)
  (C : Point)

-- Function to check if a triangle is isosceles
def isIsosceles (t : Triangle) : Prop :=
  (t.A.x - t.C.x)^2 + (t.A.y - t.C.y)^2 = (t.B.x - t.C.x)^2 + (t.B.y - t.C.y)^2 ∨
  (t.A.x - t.C.x)^2 + (t.A.y - t.C.y)^2 = (t.A.x - t.B.x)^2 + (t.A.y - t.B.y)^2 ∨
  (t.B.x - t.C.x)^2 + (t.B.y - t.C.y)^2 = (t.A.x - t.B.x)^2 + (t.A.y - t.B.y)^2

-- Theorem statement
theorem isosceles_triangle_count (g : Geoboard) (A B : Point) 
  (h1 : A.y = B.y) -- A and B are on the same horizontal line
  (h2 : B.x - A.x = 3) -- Distance between A and B is 3 units
  (h3 : A.x > 0 ∧ A.y > 0 ∧ B.x < g.size ∧ B.y < g.size) -- A and B are within the grid
  : ∃ (S : Finset Point), 
    (∀ C ∈ S, C ≠ A ∧ C ≠ B ∧ C.x > 0 ∧ C.y > 0 ∧ C.x ≤ g.size ∧ C.y ≤ g.size) ∧ 
    (∀ C ∈ S, isIsosceles ⟨A, B, C⟩) ∧
    S.card = 3 :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_count_l3844_384492


namespace NUMINAMATH_CALUDE_intersection_sum_l3844_384415

/-- Given two lines that intersect at (2,1), prove that a + b = 5/3 -/
theorem intersection_sum (a b : ℝ) : 
  (2 = (1/3) * 1 + a) →  -- First line equation at (2,1)
  (1 = (1/2) * 2 + b) →  -- Second line equation at (2,1)
  a + b = 5/3 := by
  sorry

end NUMINAMATH_CALUDE_intersection_sum_l3844_384415


namespace NUMINAMATH_CALUDE_intersection_A_B_solution_set_quadratic_l3844_384408

-- Define sets A and B
def A : Set ℝ := {x | x^2 < 4}
def B : Set ℝ := {x | 1 < x ∧ x < 3}

-- Theorem for the intersection of A and B
theorem intersection_A_B : A ∩ B = {x | 1 < x ∧ x < 2} := by sorry

-- Define the quadratic inequality
def quadratic_inequality (x : ℝ) : Prop := 2*x^2 + 4*x - 6 < 0

-- Theorem for the solution set of the quadratic inequality
theorem solution_set_quadratic : {x | quadratic_inequality x} = B := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_solution_set_quadratic_l3844_384408


namespace NUMINAMATH_CALUDE_valid_sequences_12_l3844_384406

def fibonacci : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fibonacci (n + 1) + fibonacci n

def valid_sequences (n : ℕ) : ℕ :=
  fibonacci (n + 2)

theorem valid_sequences_12 :
  valid_sequences 12 = 377 :=
by sorry

#eval valid_sequences 12

end NUMINAMATH_CALUDE_valid_sequences_12_l3844_384406


namespace NUMINAMATH_CALUDE_committee_count_is_738_l3844_384488

/-- Represents a department in the university's science division -/
inductive Department
| Physics
| Chemistry
| Biology

/-- Represents the gender of a professor -/
inductive Gender
| Male
| Female

/-- Represents a professor with their department and gender -/
structure Professor :=
  (dept : Department)
  (gender : Gender)

/-- The total number of professors in each department for each gender -/
def professors_per_dept_gender : Nat := 3

/-- The total number of professors in the committee -/
def committee_size : Nat := 7

/-- The number of male professors required in the committee -/
def required_males : Nat := 4

/-- The number of female professors required in the committee -/
def required_females : Nat := 3

/-- The number of professors required from the physics department -/
def required_physics : Nat := 3

/-- The number of professors required from each of chemistry and biology departments -/
def required_chem_bio : Nat := 2

/-- Calculates the number of possible committees given the conditions -/
def count_committees (professors : List Professor) : Nat :=
  sorry

/-- Theorem stating that the number of possible committees is 738 -/
theorem committee_count_is_738 (professors : List Professor) : 
  count_committees professors = 738 := by
  sorry

end NUMINAMATH_CALUDE_committee_count_is_738_l3844_384488


namespace NUMINAMATH_CALUDE_power_mod_seven_l3844_384473

theorem power_mod_seven : 3^87 + 5 ≡ 4 [ZMOD 7] := by sorry

end NUMINAMATH_CALUDE_power_mod_seven_l3844_384473


namespace NUMINAMATH_CALUDE_first_month_sale_l3844_384410

def sales_data : List ℕ := [6927, 6855, 7230, 6562]
def required_sixth_month_sale : ℕ := 5591
def target_average : ℕ := 6600
def num_months : ℕ := 6

theorem first_month_sale (sales : List ℕ) (sixth_sale target_avg n_months : ℕ)
  (h1 : sales = sales_data)
  (h2 : sixth_sale = required_sixth_month_sale)
  (h3 : target_avg = target_average)
  (h4 : n_months = num_months) :
  ∃ (first_sale : ℕ), 
    (first_sale + sales.sum + sixth_sale) / n_months = target_avg ∧ 
    first_sale = 6435 := by
  sorry

end NUMINAMATH_CALUDE_first_month_sale_l3844_384410


namespace NUMINAMATH_CALUDE_last_four_digits_of_5_pow_2015_l3844_384491

theorem last_four_digits_of_5_pow_2015 : ∃ n : ℕ, 5^2015 ≡ 8125 [MOD 10000] := by
  sorry

end NUMINAMATH_CALUDE_last_four_digits_of_5_pow_2015_l3844_384491


namespace NUMINAMATH_CALUDE_max_profit_theorem_l3844_384448

/-- Represents the profit function for a product given its price increase -/
def profit_function (x : ℕ) : ℝ := -10 * x^2 + 170 * x + 2100

/-- Represents the constraint on the price increase -/
def price_increase_constraint (x : ℕ) : Prop := 0 < x ∧ x ≤ 15

theorem max_profit_theorem :
  ∃ (x : ℕ), price_increase_constraint x ∧
    (∀ (y : ℕ), price_increase_constraint y → profit_function x ≥ profit_function y) ∧
    profit_function x = 2400 ∧
    (x = 5 ∨ x = 6) := by sorry

end NUMINAMATH_CALUDE_max_profit_theorem_l3844_384448


namespace NUMINAMATH_CALUDE_cross_country_winning_scores_l3844_384489

/-- Represents a cross country meet between two teams -/
structure CrossCountryMeet where
  runners_per_team : ℕ
  total_runners : ℕ
  min_score : ℕ
  max_winning_score : ℕ

/-- The number of different possible winning scores in a cross country meet -/
def winning_scores (meet : CrossCountryMeet) : ℕ :=
  meet.max_winning_score - meet.min_score + 1

/-- Theorem stating the number of different possible winning scores in the specific meet conditions -/
theorem cross_country_winning_scores :
  ∃ (meet : CrossCountryMeet),
    meet.runners_per_team = 6 ∧
    meet.total_runners = 12 ∧
    meet.min_score = 21 ∧
    meet.max_winning_score = 38 ∧
    winning_scores meet = 18 :=
  sorry

end NUMINAMATH_CALUDE_cross_country_winning_scores_l3844_384489


namespace NUMINAMATH_CALUDE_cos_equality_solutions_l3844_384471

theorem cos_equality_solutions (n : ℤ) : 
  0 ≤ n ∧ n ≤ 360 ∧ Real.cos (n * π / 180) = Real.cos (340 * π / 180) → n = 20 ∨ n = 340 := by
  sorry

end NUMINAMATH_CALUDE_cos_equality_solutions_l3844_384471


namespace NUMINAMATH_CALUDE_unique_valid_ticket_l3844_384469

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def is_multiple_of_5 (n : ℕ) : Prop := n % 5 = 0

def is_even (n : ℕ) : Prop := n % 2 = 0

def ticket_valid (n : ℕ) : Prop :=
  n ≥ 1000 ∧ n < 10000 ∧
  is_prime (n % 10) ∧
  is_multiple_of_5 ((n / 10) % 10) ∧
  is_even ((n / 100) % 10) ∧
  n / 1000 = 3 * (n % 10)

theorem unique_valid_ticket : ∀ n : ℕ, ticket_valid n ↔ n = 9853 :=
sorry

end NUMINAMATH_CALUDE_unique_valid_ticket_l3844_384469


namespace NUMINAMATH_CALUDE_sandwich_combinations_l3844_384424

def num_toppings : ℕ := 10
def num_patty_types : ℕ := 3

theorem sandwich_combinations :
  (2^num_toppings) * num_patty_types = 3072 := by sorry

end NUMINAMATH_CALUDE_sandwich_combinations_l3844_384424


namespace NUMINAMATH_CALUDE_quadratic_expression_value_l3844_384474

theorem quadratic_expression_value (x : ℝ) (h : x^2 - 3*x + 1 = 4) : 2*x^2 - 6*x + 5 = 11 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_expression_value_l3844_384474


namespace NUMINAMATH_CALUDE_band_sections_fraction_l3844_384423

theorem band_sections_fraction (trumpet_fraction trombone_fraction : ℚ) 
  (h1 : trumpet_fraction = 1/2)
  (h2 : trombone_fraction = 1/8) :
  trumpet_fraction + trombone_fraction = 5/8 := by
  sorry

end NUMINAMATH_CALUDE_band_sections_fraction_l3844_384423


namespace NUMINAMATH_CALUDE_triangle_calculation_l3844_384490

-- Define the triangle operation
def triangle (a b : ℚ) : ℚ := a * b + 2 * a

-- State the theorem
theorem triangle_calculation : triangle (-3) (triangle (-4) (1/2)) = 24 := by
  sorry

end NUMINAMATH_CALUDE_triangle_calculation_l3844_384490


namespace NUMINAMATH_CALUDE_two_digit_number_formation_l3844_384454

theorem two_digit_number_formation (k : ℕ) 
  (h1 : k > 0)
  (h2 : k ≤ 9)
  (h3 : ∀ (S T : ℕ), S = 11 * T * (k - 1) → S / T = 22) :
  k = 3 := by
sorry

end NUMINAMATH_CALUDE_two_digit_number_formation_l3844_384454


namespace NUMINAMATH_CALUDE_twenty_five_percent_problem_l3844_384485

theorem twenty_five_percent_problem (x : ℚ) : x + (1/4) * x = 80 - (1/4) * 80 ↔ x = 48 := by
  sorry

end NUMINAMATH_CALUDE_twenty_five_percent_problem_l3844_384485


namespace NUMINAMATH_CALUDE_series_calculation_l3844_384467

def series_sum (n : ℕ) : ℤ :=
  (n + 1) * 3

theorem series_calculation : series_sum 32 = 1584 := by
  sorry

#eval series_sum 32

end NUMINAMATH_CALUDE_series_calculation_l3844_384467


namespace NUMINAMATH_CALUDE_mean_temperature_l3844_384459

def temperatures : List ℚ := [79, 78, 82, 86, 88, 90, 88, 90, 89]

theorem mean_temperature : 
  (temperatures.sum / temperatures.length : ℚ) = 770 / 9 := by sorry

end NUMINAMATH_CALUDE_mean_temperature_l3844_384459


namespace NUMINAMATH_CALUDE_triangle_problem_l3844_384449

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) :
  0 < A ∧ A < π →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  a > 0 ∧ b > 0 ∧ c > 0 →
  Real.sqrt 3 * c * Real.cos A + a * Real.sin C = Real.sqrt 3 * c →
  b + c = 5 →
  (1/2) * b * c * Real.sin A = Real.sqrt 3 →
  A = π/3 ∧ a = Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l3844_384449


namespace NUMINAMATH_CALUDE_fraction_equation_solution_l3844_384481

theorem fraction_equation_solution :
  ∃ x : ℚ, (x + 7) / (x - 4) = (x - 5) / (x + 2) ∧ x = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equation_solution_l3844_384481


namespace NUMINAMATH_CALUDE_root_equivalence_l3844_384407

theorem root_equivalence (r : ℝ) : r^2 - 2*r - 1 = 0 → r^5 - 29*r - 12 = 0 := by
  sorry

end NUMINAMATH_CALUDE_root_equivalence_l3844_384407


namespace NUMINAMATH_CALUDE_problem1_problem2_l3844_384411

-- Problem 1
theorem problem1 : Real.sqrt 3 ^ 2 - (2023 + π / 2) ^ 0 - (-1) ^ (-1 : ℤ) = 3 := by sorry

-- Problem 2
theorem problem2 : ¬∃ x : ℝ, 5 * x - 4 > 3 * x ∧ (2 * x - 1) / 3 < x / 2 := by sorry

end NUMINAMATH_CALUDE_problem1_problem2_l3844_384411


namespace NUMINAMATH_CALUDE_tulip_fraction_l3844_384432

theorem tulip_fraction (total : ℕ) (yellow_ratio red_ratio pink_ratio : ℚ) : 
  total = 60 ∧
  yellow_ratio = 1/2 ∧
  red_ratio = 1/3 ∧
  pink_ratio = 1/4 →
  (total - (yellow_ratio * total) - 
   (red_ratio * (total - yellow_ratio * total)) - 
   (pink_ratio * (total - yellow_ratio * total - red_ratio * (total - yellow_ratio * total)))) / total = 1/4 :=
by sorry

end NUMINAMATH_CALUDE_tulip_fraction_l3844_384432


namespace NUMINAMATH_CALUDE_quadratic_roots_properties_l3844_384403

theorem quadratic_roots_properties (a b : ℝ) : 
  (a^2 + 3*a - 2 = 0) → (b^2 + 3*b - 2 = 0) → 
  (a + b = -3) ∧ (a^3 + 3*a^2 + 2*b = -6) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_properties_l3844_384403


namespace NUMINAMATH_CALUDE_geometric_sequence_proof_l3844_384499

def geometric_sequence (a₁ a₂ a₃ : ℝ) : Prop :=
  ∃ r : ℝ, a₂ = a₁ * r ∧ a₃ = a₂ * r

def arithmetic_sequence (a₁ a₂ a₃ : ℝ) : Prop :=
  ∃ d : ℝ, a₂ = a₁ + d ∧ a₃ = a₂ + d

theorem geometric_sequence_proof (b : ℝ) 
  (h₁ : geometric_sequence 150 b (60/36)) 
  (h₂ : b > 0) : 
  b = 5 * Real.sqrt 10 ∧ ¬ arithmetic_sequence 150 b (60/36) := by
  sorry

#check geometric_sequence_proof

end NUMINAMATH_CALUDE_geometric_sequence_proof_l3844_384499


namespace NUMINAMATH_CALUDE_expr3_greatest_l3844_384400

def expr1 (x y z : ℝ) := 4 * x^2 - 3 * y + 2 * z
def expr2 (x y z : ℝ) := 6 * x - 2 * y^3 + 3 * z^2
def expr3 (x y z : ℝ) := 2 * x^3 - y^2 * z
def expr4 (x y z : ℝ) := x * y^3 - z^2

theorem expr3_greatest :
  let x : ℝ := 3
  let y : ℝ := 2
  let z : ℝ := 1
  expr3 x y z > expr1 x y z ∧
  expr3 x y z > expr2 x y z ∧
  expr3 x y z > expr4 x y z := by
sorry

end NUMINAMATH_CALUDE_expr3_greatest_l3844_384400


namespace NUMINAMATH_CALUDE_folded_perimeter_not_greater_l3844_384419

/-- Represents a polygon in 2D space -/
structure Polygon where
  vertices : List (ℝ × ℝ)
  is_closed : vertices.length ≥ 3

/-- Represents a line in 2D space -/
structure Line where
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ

/-- Calculates the perimeter of a polygon -/
def perimeter (p : Polygon) : ℝ := sorry

/-- Folds a polygon along a line and glues the halves together -/
def fold_and_glue (p : Polygon) (l : Line) : Polygon := sorry

/-- Theorem: The perimeter of a folded and glued polygon is not greater than the original -/
theorem folded_perimeter_not_greater (p : Polygon) (l : Line) :
  perimeter (fold_and_glue p l) ≤ perimeter p := by sorry

end NUMINAMATH_CALUDE_folded_perimeter_not_greater_l3844_384419


namespace NUMINAMATH_CALUDE_power_subtraction_l3844_384435

theorem power_subtraction : (2 : ℕ) ^ 4 - (2 : ℕ) ^ 3 = (2 : ℕ) ^ 3 := by
  sorry

end NUMINAMATH_CALUDE_power_subtraction_l3844_384435
