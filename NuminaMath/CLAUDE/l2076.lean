import Mathlib

namespace NUMINAMATH_CALUDE_yoongi_has_fewest_apples_l2076_207686

def yoongi_apples : ℕ := 4
def yuna_apples : ℕ := 5
def jungkook_apples : ℕ := 6 * 3

theorem yoongi_has_fewest_apples :
  yoongi_apples < yuna_apples ∧ yoongi_apples < jungkook_apples := by
  sorry

end NUMINAMATH_CALUDE_yoongi_has_fewest_apples_l2076_207686


namespace NUMINAMATH_CALUDE_expression_evaluation_l2076_207690

theorem expression_evaluation :
  let x : ℚ := -1/3
  (2*x - 1)^2 - (3*x + 1)*(3*x - 1) + 5*x*(x - 1) = 5 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2076_207690


namespace NUMINAMATH_CALUDE_rebecca_eggs_l2076_207614

/-- The number of eggs Rebecca has -/
def num_eggs : ℕ := 3 * 6

/-- The number of groups Rebecca will create -/
def num_groups : ℕ := 3

/-- The number of eggs in each group -/
def eggs_per_group : ℕ := 6

/-- Theorem stating that Rebecca has 18 eggs -/
theorem rebecca_eggs : num_eggs = 18 := by
  sorry

end NUMINAMATH_CALUDE_rebecca_eggs_l2076_207614


namespace NUMINAMATH_CALUDE_f_has_one_zero_l2076_207616

/-- The function f(x) defined in terms of the parameter m -/
def f (m : ℝ) (x : ℝ) : ℝ := (m - 1) * x^2 + 2 * (m + 1) * x - 1

/-- The set of real numbers m for which f(x) has exactly one zero -/
def one_zero_set : Set ℝ := {m : ℝ | m = -3 ∨ m = 0}

/-- Theorem stating that f(x) has exactly one zero if and only if m is in the one_zero_set -/
theorem f_has_one_zero (m : ℝ) : 
  (∃! x : ℝ, f m x = 0) ↔ m ∈ one_zero_set :=
sorry

end NUMINAMATH_CALUDE_f_has_one_zero_l2076_207616


namespace NUMINAMATH_CALUDE_six_double_prime_value_l2076_207656

-- Define the prime operation
def prime (q : ℝ) : ℝ := 3 * q - 3

-- State the theorem
theorem six_double_prime_value : prime (prime 6) = 42 := by
  sorry

end NUMINAMATH_CALUDE_six_double_prime_value_l2076_207656


namespace NUMINAMATH_CALUDE_radio_selling_price_l2076_207621

/-- Calculates the selling price of a radio given its purchase price, overhead expenses, and profit percentage. -/
def calculate_selling_price (purchase_price : ℚ) (overhead_expenses : ℚ) (profit_percentage : ℚ) : ℚ :=
  let total_cost := purchase_price + overhead_expenses
  let profit_amount := (profit_percentage / 100) * total_cost
  total_cost + profit_amount

/-- Theorem stating that the selling price of a radio with given parameters is 350 Rs. -/
theorem radio_selling_price :
  let purchase_price : ℚ := 225
  let overhead_expenses : ℚ := 15
  let profit_percentage : ℚ := 45833333333333314 / 1000000000000000
  calculate_selling_price purchase_price overhead_expenses profit_percentage = 350 := by
  sorry


end NUMINAMATH_CALUDE_radio_selling_price_l2076_207621


namespace NUMINAMATH_CALUDE_sin_450_degrees_l2076_207605

theorem sin_450_degrees : Real.sin (450 * π / 180) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sin_450_degrees_l2076_207605


namespace NUMINAMATH_CALUDE_stratified_sample_grad_count_l2076_207680

/-- Represents the number of students to be sampled from each stratum in a stratified sampling -/
structure StratifiedSample where
  total : ℕ
  junior : ℕ
  undergrad : ℕ
  grad : ℕ

/-- Calculates the stratified sample given the total population and sample size -/
def calculateStratifiedSample (totalPopulation : ℕ) (juniorCount : ℕ) (undergradCount : ℕ) (sampleSize : ℕ) : StratifiedSample :=
  let gradCount := totalPopulation - juniorCount - undergradCount
  let sampleRatio := sampleSize / totalPopulation
  { total := sampleSize,
    junior := juniorCount * sampleRatio,
    undergrad := undergradCount * sampleRatio,
    grad := sampleSize - (juniorCount * sampleRatio) - (undergradCount * sampleRatio) }

theorem stratified_sample_grad_count 
  (totalPopulation : ℕ) (juniorCount : ℕ) (undergradCount : ℕ) (sampleSize : ℕ)
  (h1 : totalPopulation = 5600)
  (h2 : juniorCount = 1300)
  (h3 : undergradCount = 3000)
  (h4 : sampleSize = 280) :
  (calculateStratifiedSample totalPopulation juniorCount undergradCount sampleSize).grad = 65 := by
  sorry

#eval (calculateStratifiedSample 5600 1300 3000 280).grad

end NUMINAMATH_CALUDE_stratified_sample_grad_count_l2076_207680


namespace NUMINAMATH_CALUDE_equal_area_division_l2076_207660

-- Define a type for points in a plane
variable (Point : Type)

-- Define a type for lines in a plane
variable (Line : Type)

-- Define a type for figures in a plane
variable (Figure : Type)

-- Function to check if two lines are parallel
variable (parallel : Line → Line → Prop)

-- Function to measure the area of a figure
variable (area : Figure → ℝ)

-- Function to get the part of a figure on one side of a line
variable (figurePart : Figure → Line → Figure)

-- Theorem statement
theorem equal_area_division 
  (Φ : Figure) (l₀ : Line) : 
  ∃ (l : Line), 
    parallel l l₀ ∧ 
    area (figurePart Φ l) = (area Φ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_equal_area_division_l2076_207660


namespace NUMINAMATH_CALUDE_roots_are_correct_all_roots_found_l2076_207643

/-- The roots of the equation 5x^4 - 28x^3 + 49x^2 - 28x + 5 = 0 -/
def roots : Set ℝ :=
  {2, 1/2, (5 + Real.sqrt 21)/5, (5 - Real.sqrt 21)/5}

/-- The polynomial function corresponding to the equation -/
def f (x : ℝ) : ℝ := 5*x^4 - 28*x^3 + 49*x^2 - 28*x + 5

theorem roots_are_correct : ∀ x ∈ roots, f x = 0 := by
  sorry

theorem all_roots_found : ∀ x, f x = 0 → x ∈ roots := by
  sorry

end NUMINAMATH_CALUDE_roots_are_correct_all_roots_found_l2076_207643


namespace NUMINAMATH_CALUDE_natalia_clip_sales_l2076_207671

/-- The number of clips Natalia sold in April and May combined -/
def total_clips (april_sales : ℕ) (may_sales : ℕ) : ℕ :=
  april_sales + may_sales

/-- Theorem stating that given the conditions, Natalia sold 72 clips in total -/
theorem natalia_clip_sales : 
  ∀ (april_sales : ℕ) (may_sales : ℕ),
    april_sales = 48 →
    may_sales = april_sales / 2 →
    total_clips april_sales may_sales = 72 :=
by
  sorry

end NUMINAMATH_CALUDE_natalia_clip_sales_l2076_207671


namespace NUMINAMATH_CALUDE_square_sum_of_differences_l2076_207619

theorem square_sum_of_differences (x y z : ℤ) : 
  ∃ (σ₂ : ℤ), (1/2 : ℚ) * ((x - y)^4 + (y - z)^4 + (z - x)^4) = (σ₂^2 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_square_sum_of_differences_l2076_207619


namespace NUMINAMATH_CALUDE_largest_quantity_l2076_207628

def A : ℚ := 3004 / 3003 + 3004 / 3005
def B : ℚ := 3006 / 3005 + 3006 / 3007
def C : ℚ := 3005 / 3004 + 3005 / 3006

theorem largest_quantity : A > B ∧ A > C := by
  sorry

end NUMINAMATH_CALUDE_largest_quantity_l2076_207628


namespace NUMINAMATH_CALUDE_basic_computer_price_l2076_207687

theorem basic_computer_price 
  (total_price : ℝ) 
  (enhanced_computer_diff : ℝ) 
  (printer_ratio : ℝ) :
  total_price = 2500 →
  enhanced_computer_diff = 500 →
  printer_ratio = 1/4 →
  ∃ (basic_computer : ℝ) (printer : ℝ),
    basic_computer + printer = total_price ∧
    printer = printer_ratio * (basic_computer + enhanced_computer_diff + printer) ∧
    basic_computer = 1750 :=
by sorry

end NUMINAMATH_CALUDE_basic_computer_price_l2076_207687


namespace NUMINAMATH_CALUDE_no_rational_roots_l2076_207677

theorem no_rational_roots (p q : ℤ) 
  (hp : p % 3 = 2) 
  (hq : q % 3 = 2) : 
  ¬ ∃ (r : ℚ), r^2 + p * r + q = 0 := by
sorry

end NUMINAMATH_CALUDE_no_rational_roots_l2076_207677


namespace NUMINAMATH_CALUDE_quadratic_roots_relation_l2076_207648

theorem quadratic_roots_relation (m n p : ℝ) (hm : m ≠ 0) (hn : n ≠ 0) (hp : p ≠ 0) :
  (∃ r₁ r₂ : ℝ, (r₁ + r₂ = -p ∧ r₁ * r₂ = m) ∧
               (3 * r₁ + 3 * r₂ = -m ∧ 9 * r₁ * r₂ = n)) →
  n / p = 27 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_relation_l2076_207648


namespace NUMINAMATH_CALUDE_area_of_special_triangle_l2076_207604

/-- Given points A and B on the graph of y = 1/x in the first quadrant,
    if ∠OAB = 90° and OA = AB, then the area of triangle OAB is √5/2 -/
theorem area_of_special_triangle (A B : ℝ × ℝ) : 
  (A.2 = 1 / A.1) →  -- A is on y = 1/x
  (B.2 = 1 / B.1) →  -- B is on y = 1/x
  (A.1 > 0 ∧ A.2 > 0) →  -- A is in first quadrant
  (B.1 > 0 ∧ B.2 > 0) →  -- B is in first quadrant
  (A.1 * (B.1 - A.1) + A.2 * (B.2 - A.2) = 0) →  -- ∠OAB = 90°
  (A.1^2 + A.2^2 = (B.1 - A.1)^2 + (B.2 - A.2)^2) →  -- OA = AB
  (1/2 * Real.sqrt (A.1^2 + A.2^2) = Real.sqrt 5 / 2) := by
sorry

end NUMINAMATH_CALUDE_area_of_special_triangle_l2076_207604


namespace NUMINAMATH_CALUDE_interest_difference_approximately_128_l2076_207657

-- Define the initial deposit
def initial_deposit : ℝ := 14000

-- Define the interest rates
def compound_rate : ℝ := 0.06
def simple_rate : ℝ := 0.08

-- Define the time period
def years : ℕ := 10

-- Define the compound interest function
def compound_interest (p r : ℝ) (n : ℕ) : ℝ := p * (1 + r) ^ n

-- Define the simple interest function
def simple_interest (p r : ℝ) (t : ℕ) : ℝ := p * (1 + r * t)

-- State the theorem
theorem interest_difference_approximately_128 :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 1 ∧
  abs (simple_interest initial_deposit simple_rate years - 
       compound_interest initial_deposit compound_rate years - 128) < ε :=
sorry

end NUMINAMATH_CALUDE_interest_difference_approximately_128_l2076_207657


namespace NUMINAMATH_CALUDE_result_line_properties_l2076_207672

/-- The circle equation -/
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 4*y = 0

/-- The given line equation -/
def given_line_eq (x y : ℝ) : Prop := 2*x + 3*y = 0

/-- The resulting line equation -/
def result_line_eq (x y : ℝ) : Prop := 3*x - 2*y + 7 = 0

/-- The center of the circle -/
def circle_center : ℝ × ℝ := (-1, 2)

/-- Theorem stating that the resulting line passes through the center of the circle
    and is perpendicular to the given line -/
theorem result_line_properties :
  result_line_eq (circle_center.1) (circle_center.2) ∧
  (∀ (x₁ y₁ x₂ y₂ : ℝ), given_line_eq x₁ y₁ → given_line_eq x₂ y₂ →
    result_line_eq x₁ y₁ → result_line_eq x₂ y₂ →
    (x₂ - x₁) * (x₂ - x₁) + (y₂ - y₁) * (y₂ - y₁) ≠ 0 →
    ((x₂ - x₁) * 2 + (y₂ - y₁) * 3) * ((x₂ - x₁) * 3 + (y₂ - y₁) * (-2)) = 0) :=
sorry

end NUMINAMATH_CALUDE_result_line_properties_l2076_207672


namespace NUMINAMATH_CALUDE_allison_win_probability_l2076_207668

-- Define the faces of each cube
def allison_cube : Finset Nat := {4, 4, 4, 4, 4, 4}
def charlie_cube : Finset Nat := {1, 1, 2, 3, 4, 5}
def dani_cube : Finset Nat := {3, 3, 3, 3, 5, 5}

-- Define the probability of rolling each number for each person
def prob_roll (cube : Finset Nat) (n : Nat) : Rat :=
  (cube.filter (· = n)).card / cube.card

-- Define the event of Allison winning
def allison_wins (c : Nat) (d : Nat) : Prop :=
  4 > c ∧ 4 > d

-- Theorem statement
theorem allison_win_probability :
  (prob_roll charlie_cube 1 + prob_roll charlie_cube 2) *
  (prob_roll dani_cube 3) = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_allison_win_probability_l2076_207668


namespace NUMINAMATH_CALUDE_find_t_l2076_207653

theorem find_t : ∃ t : ℝ, 
  (∀ x : ℝ, |2*x + t| - t ≤ 8 ↔ -5 ≤ x ∧ x ≤ 4) → 
  t = 1 := by sorry

end NUMINAMATH_CALUDE_find_t_l2076_207653


namespace NUMINAMATH_CALUDE_product_sum_theorem_l2076_207675

theorem product_sum_theorem (a b c d e : ℤ) :
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧
  c ≠ d ∧ c ≠ e ∧
  d ≠ e →
  (7 - a) * (7 - b) * (7 - c) * (7 - d) * (7 - e) = 120 →
  a + b + c + d + e = 33 := by
sorry

end NUMINAMATH_CALUDE_product_sum_theorem_l2076_207675


namespace NUMINAMATH_CALUDE_cookie_sale_loss_l2076_207632

/-- Represents the cookie sale scenario --/
structure CookieSale where
  total_cookies : ℕ
  purchase_rate : ℚ  -- cookies per dollar
  selling_rate : ℚ   -- cookies per dollar

/-- Calculates the loss from a cookie sale --/
def calculate_loss (sale : CookieSale) : ℚ :=
  let cost := sale.total_cookies / sale.purchase_rate
  let revenue := sale.total_cookies / sale.selling_rate
  cost - revenue

/-- The main theorem stating the loss for the given scenario --/
theorem cookie_sale_loss : 
  let sale : CookieSale := {
    total_cookies := 800,
    purchase_rate := 4/3,  -- 4 cookies for $3
    selling_rate := 3/2    -- 3 cookies for $2
  }
  calculate_loss sale = 64 := by
  sorry


end NUMINAMATH_CALUDE_cookie_sale_loss_l2076_207632


namespace NUMINAMATH_CALUDE_initial_meals_correct_l2076_207642

/-- The number of meals Colt and Curt initially prepared -/
def initial_meals : ℕ := 113

/-- The number of meals Sole Mart provided -/
def sole_mart_meals : ℕ := 50

/-- The number of meals given away -/
def meals_given_away : ℕ := 85

/-- The number of meals left to be distributed -/
def meals_left : ℕ := 78

/-- Theorem stating that the initial number of meals is correct -/
theorem initial_meals_correct : 
  initial_meals + sole_mart_meals = meals_given_away + meals_left := by
  sorry

end NUMINAMATH_CALUDE_initial_meals_correct_l2076_207642


namespace NUMINAMATH_CALUDE_mans_speed_with_current_l2076_207630

/-- Given a man's speed against the current and the speed of the current,
    calculate the man's speed with the current. -/
def speed_with_current (speed_against_current : ℝ) (current_speed : ℝ) : ℝ :=
  speed_against_current + 2 * current_speed

/-- Theorem stating that given the specific conditions, 
    the man's speed with the current is 21 km/hr. -/
theorem mans_speed_with_current :
  let speed_against_current : ℝ := 16
  let current_speed : ℝ := 2.5
  speed_with_current speed_against_current current_speed = 21 := by
  sorry

#eval speed_with_current 16 2.5

end NUMINAMATH_CALUDE_mans_speed_with_current_l2076_207630


namespace NUMINAMATH_CALUDE_triangle_area_l2076_207608

def a : ℝ × ℝ := (4, -3)
def b : ℝ × ℝ := (-6, 5)
def c : ℝ × ℝ := (-12, 10)

theorem triangle_area : 
  let det := a.1 * c.2 - a.2 * c.1
  (1/2 : ℝ) * |det| = 2 := by sorry

end NUMINAMATH_CALUDE_triangle_area_l2076_207608


namespace NUMINAMATH_CALUDE_ben_walking_time_l2076_207682

/-- Given that Ben walks at a constant speed and covers 3 km in 2 hours,
    prove that the time required to walk 12 km is 480 minutes. -/
theorem ben_walking_time (speed : ℝ) (h1 : speed > 0) : 
  (3 : ℝ) / speed = 2 → (12 : ℝ) / speed * 60 = 480 := by
sorry

end NUMINAMATH_CALUDE_ben_walking_time_l2076_207682


namespace NUMINAMATH_CALUDE_peggy_needs_825_stamps_l2076_207617

/-- The number of stamps Peggy needs to add to have as many as Bert -/
def stamps_to_add (peggy_stamps : ℕ) : ℕ :=
  4 * (3 * peggy_stamps) - peggy_stamps

/-- Theorem stating that Peggy needs to add 825 stamps to have as many as Bert -/
theorem peggy_needs_825_stamps : stamps_to_add 75 = 825 := by
  sorry

end NUMINAMATH_CALUDE_peggy_needs_825_stamps_l2076_207617


namespace NUMINAMATH_CALUDE_cubic_sum_theorem_l2076_207673

theorem cubic_sum_theorem (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) 
  (h_eq : (a^3 - 12) / a = (b^3 - 12) / b ∧ (b^3 - 12) / b = (c^3 - 12) / c) : 
  a^3 + b^3 + c^3 = 36 := by
sorry

end NUMINAMATH_CALUDE_cubic_sum_theorem_l2076_207673


namespace NUMINAMATH_CALUDE_village_population_l2076_207669

theorem village_population (P : ℕ) : 
  (P : ℝ) * 0.9 * 0.85 = 2907 → P = 3801 := by
sorry

end NUMINAMATH_CALUDE_village_population_l2076_207669


namespace NUMINAMATH_CALUDE_fraction_equality_sum_l2076_207645

theorem fraction_equality_sum (C D : ℚ) :
  (∀ x : ℚ, x ≠ 4 ∧ x ≠ 5 → (D * x - 13) / (x^2 - 9*x + 20) = C / (x - 4) + 5 / (x - 5)) →
  C + D = 1/5 := by
sorry

end NUMINAMATH_CALUDE_fraction_equality_sum_l2076_207645


namespace NUMINAMATH_CALUDE_largest_number_in_ratio_l2076_207696

theorem largest_number_in_ratio (a b c : ℕ) : 
  a ≠ 0 → b ≠ 0 → c ≠ 0 →
  (b = (5 * a) / 3) →
  (c = (7 * a) / 3) →
  (c - a = 32) →
  c = 56 := by
sorry

end NUMINAMATH_CALUDE_largest_number_in_ratio_l2076_207696


namespace NUMINAMATH_CALUDE_ryan_final_tokens_l2076_207678

def initial_tokens : ℕ := 36
def pacman_fraction : ℚ := 1/3
def candy_crush_fraction : ℚ := 1/4
def skiball_tokens : ℕ := 7
def parent_multiplier : ℕ := 7

theorem ryan_final_tokens :
  let pacman_tokens := (pacman_fraction * initial_tokens).floor
  let candy_crush_tokens := (candy_crush_fraction * initial_tokens).floor
  let total_spent := pacman_tokens + candy_crush_tokens + skiball_tokens
  let tokens_left := initial_tokens - total_spent
  let parent_bought := parent_multiplier * skiball_tokens
  tokens_left + parent_bought = 57 := by
  sorry

end NUMINAMATH_CALUDE_ryan_final_tokens_l2076_207678


namespace NUMINAMATH_CALUDE_ones_digit_largest_power_of_two_32_factorial_l2076_207641

/-- The largest power of 2 that divides n! -/
def largest_power_of_two (n : ℕ) : ℕ := sorry

/-- The ones digit of a natural number -/
def ones_digit (n : ℕ) : ℕ := n % 10

theorem ones_digit_largest_power_of_two_32_factorial :
  ones_digit (2^(largest_power_of_two 32)) = 4 := by sorry

end NUMINAMATH_CALUDE_ones_digit_largest_power_of_two_32_factorial_l2076_207641


namespace NUMINAMATH_CALUDE_dress_shop_inventory_l2076_207610

def total_dresses (red : ℕ) (blue : ℕ) : ℕ := red + blue

theorem dress_shop_inventory : 
  let red : ℕ := 83
  let blue : ℕ := red + 34
  total_dresses red blue = 200 := by
sorry

end NUMINAMATH_CALUDE_dress_shop_inventory_l2076_207610


namespace NUMINAMATH_CALUDE_correct_stratified_sample_l2076_207693

/-- Represents the population sizes for each age group -/
structure PopulationSizes where
  young : ℕ
  middleAged : ℕ
  elderly : ℕ

/-- Represents the sample sizes for each age group -/
structure SampleSizes where
  young : ℕ
  middleAged : ℕ
  elderly : ℕ

/-- Calculates the stratified sample sizes given population sizes and total sample size -/
def stratifiedSampleSizes (pop : PopulationSizes) (totalSample : ℕ) : SampleSizes :=
  { young := (pop.young * totalSample) / (pop.young + pop.middleAged + pop.elderly),
    middleAged := (pop.middleAged * totalSample) / (pop.young + pop.middleAged + pop.elderly),
    elderly := (pop.elderly * totalSample) / (pop.young + pop.middleAged + pop.elderly) }

theorem correct_stratified_sample (pop : PopulationSizes) (totalSample : ℕ) :
  pop.young = 45 ∧ pop.middleAged = 25 ∧ pop.elderly = 30 ∧ totalSample = 20 →
  let sample := stratifiedSampleSizes pop totalSample
  sample.young = 9 ∧ sample.middleAged = 5 ∧ sample.elderly = 6 := by
  sorry

end NUMINAMATH_CALUDE_correct_stratified_sample_l2076_207693


namespace NUMINAMATH_CALUDE_soccer_team_selection_l2076_207665

theorem soccer_team_selection (total_players : ℕ) (quadruplets : ℕ) (starters : ℕ) (quad_starters : ℕ) :
  total_players = 16 →
  quadruplets = 4 →
  starters = 7 →
  quad_starters = 2 →
  (Nat.choose quadruplets quad_starters) * (Nat.choose (total_players - quadruplets) (starters - quad_starters)) = 4752 := by
  sorry

end NUMINAMATH_CALUDE_soccer_team_selection_l2076_207665


namespace NUMINAMATH_CALUDE_morning_afternoon_emails_l2076_207601

theorem morning_afternoon_emails (morning_emails afternoon_emails : ℕ) 
  (h1 : morning_emails = 5)
  (h2 : afternoon_emails = 8) :
  morning_emails + afternoon_emails = 13 := by
sorry

end NUMINAMATH_CALUDE_morning_afternoon_emails_l2076_207601


namespace NUMINAMATH_CALUDE_total_volume_of_stacked_boxes_l2076_207688

/-- The volume of a single box in cubic centimeters -/
def single_box_volume : ℝ := 30

/-- The number of horizontal rows of boxes -/
def horizontal_rows : ℕ := 7

/-- The number of vertical rows of boxes -/
def vertical_rows : ℕ := 5

/-- The number of floors of boxes -/
def floors : ℕ := 3

/-- The total number of boxes -/
def total_boxes : ℕ := horizontal_rows * vertical_rows * floors

/-- The theorem stating the total volume of stacked boxes -/
theorem total_volume_of_stacked_boxes :
  (single_box_volume * total_boxes : ℝ) = 3150 := by sorry

end NUMINAMATH_CALUDE_total_volume_of_stacked_boxes_l2076_207688


namespace NUMINAMATH_CALUDE_win_sector_area_l2076_207685

theorem win_sector_area (r : ℝ) (p : ℝ) (h1 : r = 8) (h2 : p = 1/4) :
  p * π * r^2 = 16 * π := by
  sorry

end NUMINAMATH_CALUDE_win_sector_area_l2076_207685


namespace NUMINAMATH_CALUDE_books_per_day_calculation_l2076_207625

/-- Calculates the number of books read per day given the total books read and the number of reading days. -/
def books_per_day (total_books : ℕ) (reading_days : ℕ) : ℚ :=
  (total_books : ℚ) / (reading_days : ℚ)

/-- Represents the reading habits of a person over a period of weeks. -/
structure ReadingHabit where
  days_per_week : ℕ
  weeks : ℕ
  total_books : ℕ

theorem books_per_day_calculation (habit : ReadingHabit) 
    (h1 : habit.days_per_week = 2)
    (h2 : habit.weeks = 6)
    (h3 : habit.total_books = 48) :
  books_per_day habit.total_books (habit.days_per_week * habit.weeks) = 4 := by
  sorry

end NUMINAMATH_CALUDE_books_per_day_calculation_l2076_207625


namespace NUMINAMATH_CALUDE_us_apples_sold_fresh_l2076_207689

/-- Calculates the amount of apples sold fresh given total production and mixing percentage -/
def apples_sold_fresh (total_production : ℝ) (mixing_percentage : ℝ) : ℝ :=
  let remaining := total_production * (1 - mixing_percentage)
  remaining * 0.4

/-- Theorem stating that given the U.S. apple production conditions, 
    the amount of apples sold fresh is 2.24 million tons -/
theorem us_apples_sold_fresh :
  apples_sold_fresh 8 0.3 = 2.24 := by
  sorry

#eval apples_sold_fresh 8 0.3

end NUMINAMATH_CALUDE_us_apples_sold_fresh_l2076_207689


namespace NUMINAMATH_CALUDE_sequence_sum_property_l2076_207655

/-- A sequence of positive terms satisfying a specific equation. -/
def sequence_a (n : ℕ) : ℝ :=
  sorry

/-- The sum of the first n terms of the sequence. -/
def S (n : ℕ) : ℝ :=
  sorry

/-- The main theorem stating the property of the sequence sum. -/
theorem sequence_sum_property :
  ∀ (n : ℕ), n ≥ 1 →
  (n * (n + 1) * (sequence_a n)^2 + (n^2 + n - 1) * sequence_a n - 1 = 0) →
  sequence_a n > 0 →
  2019 * S 2018 = 2018 :=
sorry

end NUMINAMATH_CALUDE_sequence_sum_property_l2076_207655


namespace NUMINAMATH_CALUDE_systematic_sampling_first_two_numbers_l2076_207699

/-- Systematic sampling function that returns the nth sample number -/
def systematicSample (populationSize sampleSize n : ℕ) : ℕ :=
  (n - 1) * (populationSize / sampleSize)

/-- Theorem stating the first two sample numbers in the given systematic sampling scenario -/
theorem systematic_sampling_first_two_numbers
  (populationSize : ℕ)
  (sampleSize : ℕ)
  (lastSampleNumber : ℕ)
  (h1 : populationSize = 8000)
  (h2 : sampleSize = 50)
  (h3 : lastSampleNumber = 7900) :
  systematicSample populationSize sampleSize 1 = 159 ∧
  systematicSample populationSize sampleSize 2 = 319 := by
  sorry

#eval systematicSample 8000 50 1  -- Expected: 159
#eval systematicSample 8000 50 2  -- Expected: 319

end NUMINAMATH_CALUDE_systematic_sampling_first_two_numbers_l2076_207699


namespace NUMINAMATH_CALUDE_special_quadrilateral_angles_l2076_207624

/-- A quadrilateral with three equal sides and two specific angles -/
structure SpecialQuadrilateral where
  -- Three equal sides
  side : ℝ
  side_positive : side > 0
  -- Two angles formed by the equal sides
  angle1 : ℝ
  angle2 : ℝ
  -- Angle conditions
  angle1_is_90 : angle1 = 90
  angle2_is_150 : angle2 = 150

/-- The other two angles of the special quadrilateral -/
def other_angles (q : SpecialQuadrilateral) : ℝ × ℝ :=
  (45, 75)

/-- Theorem stating that the other two angles are 45° and 75° -/
theorem special_quadrilateral_angles (q : SpecialQuadrilateral) :
  other_angles q = (45, 75) := by
  sorry

end NUMINAMATH_CALUDE_special_quadrilateral_angles_l2076_207624


namespace NUMINAMATH_CALUDE_candy_sales_theorem_l2076_207633

-- Define the candy sales for each week
structure CandySales :=
  (week1_initial : ℕ)
  (week1_monday : ℕ)
  (week1_tuesday : ℕ)
  (week1_wednesday_left : ℕ)
  (week2_initial : ℕ)
  (week2_monday : ℕ)
  (week2_tuesday : ℕ)
  (week2_wednesday : ℕ)
  (week2_thursday : ℕ)
  (week2_friday : ℕ)
  (week3_initial : ℕ)
  (week3_highest : ℕ)

-- Define the theorem
theorem candy_sales_theorem (sales : CandySales) 
  (h1 : sales.week1_initial = 80)
  (h2 : sales.week1_monday = 15)
  (h3 : sales.week1_tuesday = 2 * sales.week1_monday)
  (h4 : sales.week1_wednesday_left = 7)
  (h5 : sales.week2_initial = 100)
  (h6 : sales.week2_monday = 12)
  (h7 : sales.week2_tuesday = 18)
  (h8 : sales.week2_wednesday = 20)
  (h9 : sales.week2_thursday = 11)
  (h10 : sales.week2_friday = 25)
  (h11 : sales.week3_initial = 120)
  (h12 : sales.week3_highest = 40) :
  (sales.week1_initial - sales.week1_wednesday_left = 73) ∧
  (sales.week2_monday + sales.week2_tuesday + sales.week2_wednesday + sales.week2_thursday + sales.week2_friday = 86) ∧
  (sales.week3_highest = 40) := by
  sorry

end NUMINAMATH_CALUDE_candy_sales_theorem_l2076_207633


namespace NUMINAMATH_CALUDE_unique_magnitude_for_complex_roots_l2076_207666

theorem unique_magnitude_for_complex_roots (x : ℂ) : 
  x^2 - 4*x + 29 = 0 → ∃! m : ℝ, ∃ y : ℂ, y^2 - 4*y + 29 = 0 ∧ Complex.abs y = m :=
by sorry

end NUMINAMATH_CALUDE_unique_magnitude_for_complex_roots_l2076_207666


namespace NUMINAMATH_CALUDE_min_draw_same_number_and_suit_min_draw_consecutive_numbers_l2076_207636

/-- Represents a card in the deck -/
structure Card where
  suit : Fin 4
  number : Fin 13

/-- The deck of cards -/
def Deck : Finset Card := sorry

/-- The number of cards in the deck -/
def deck_size : Nat := 52

/-- The number of suits in the deck -/
def num_suits : Nat := 4

/-- The number of cards per suit -/
def cards_per_suit : Nat := 13

theorem min_draw_same_number_and_suit :
  ∀ (S : Finset Card), S ⊆ Deck → S.card = 27 →
    ∃ (c1 c2 : Card), c1 ∈ S ∧ c2 ∈ S ∧ c1 ≠ c2 ∧ c1.number = c2.number ∧ c1.suit = c2.suit :=
sorry

theorem min_draw_consecutive_numbers :
  ∀ (S : Finset Card), S ⊆ Deck → S.card = 37 →
    ∃ (c1 c2 c3 : Card), c1 ∈ S ∧ c2 ∈ S ∧ c3 ∈ S ∧
      c1 ≠ c2 ∧ c2 ≠ c3 ∧ c1 ≠ c3 ∧
      (c1.number + 1 = c2.number ∧ c2.number + 1 = c3.number ∨
       c2.number + 1 = c1.number ∧ c1.number + 1 = c3.number ∨
       c1.number + 1 = c3.number ∧ c3.number + 1 = c2.number) :=
sorry

end NUMINAMATH_CALUDE_min_draw_same_number_and_suit_min_draw_consecutive_numbers_l2076_207636


namespace NUMINAMATH_CALUDE_tobias_driveways_shoveled_l2076_207684

/-- Calculates the number of driveways Tobias shoveled given his earnings and expenses. -/
theorem tobias_driveways_shoveled (
  shoe_cost : ℕ)
  (saving_months : ℕ)
  (monthly_allowance : ℕ)
  (lawn_mowing_fee : ℕ)
  (driveway_shoveling_fee : ℕ)
  (change_after_purchase : ℕ)
  (lawns_mowed : ℕ)
  (h1 : shoe_cost = 95)
  (h2 : saving_months = 3)
  (h3 : monthly_allowance = 5)
  (h4 : lawn_mowing_fee = 15)
  (h5 : driveway_shoveling_fee = 7)
  (h6 : change_after_purchase = 15)
  (h7 : lawns_mowed = 4) :
  (shoe_cost + change_after_purchase
    - saving_months * monthly_allowance
    - lawns_mowed * lawn_mowing_fee) / driveway_shoveling_fee = 5 :=
by sorry


end NUMINAMATH_CALUDE_tobias_driveways_shoveled_l2076_207684


namespace NUMINAMATH_CALUDE_max_projection_sum_l2076_207638

-- Define the plane as ℝ²
def Plane := ℝ × ℝ

-- Define the dot product for vectors in the plane
def dot_product (v w : Plane) : ℝ := v.1 * w.1 + v.2 * w.2

-- Define what it means for a vector to be a unit vector
def is_unit_vector (v : Plane) : Prop := dot_product v v = 1

-- State the theorem
theorem max_projection_sum 
  (a b c : Plane) 
  (ha : is_unit_vector a) 
  (hb : is_unit_vector b) 
  (hc : is_unit_vector c) 
  (hab : a ≠ b) 
  (hbc : b ≠ c) 
  (hac : a ≠ c)
  (hab_dot : dot_product a b = 1/2) 
  (hbc_dot : dot_product b c = 1/2) :
  ∃ (max : ℝ), max = 5 ∧ 
    ∀ (e : Plane), is_unit_vector e → 
      |dot_product a e| + |2 * dot_product b e| + 3 * |dot_product c e| ≤ max :=
sorry

end NUMINAMATH_CALUDE_max_projection_sum_l2076_207638


namespace NUMINAMATH_CALUDE_pentagon_diagonals_through_vertex_l2076_207692

/-- The number of diagonals passing through a vertex in a pentagon -/
def diagonals_through_vertex_in_pentagon : ℕ :=
  (5 : ℕ) - 3

theorem pentagon_diagonals_through_vertex :
  diagonals_through_vertex_in_pentagon = 2 :=
by sorry

end NUMINAMATH_CALUDE_pentagon_diagonals_through_vertex_l2076_207692


namespace NUMINAMATH_CALUDE_village_assistants_selection_l2076_207676

theorem village_assistants_selection (n : ℕ) (k : ℕ) (a b : ℕ) :
  n = 10 → k = 3 → a ≠ b → a ≤ n → b ≤ n →
  (Nat.choose n k - Nat.choose (n - 2) k) = 64 := by
  sorry

end NUMINAMATH_CALUDE_village_assistants_selection_l2076_207676


namespace NUMINAMATH_CALUDE_cistern_wet_surface_area_l2076_207681

/-- Calculates the total wet surface area of a rectangular cistern. -/
def totalWetSurfaceArea (length width depth : ℝ) : ℝ :=
  length * width + 2 * length * depth + 2 * width * depth

/-- Theorem: The total wet surface area of a cistern with given dimensions is 62 square meters. -/
theorem cistern_wet_surface_area :
  totalWetSurfaceArea 8 4 1.25 = 62 := by
  sorry

end NUMINAMATH_CALUDE_cistern_wet_surface_area_l2076_207681


namespace NUMINAMATH_CALUDE_number_of_binders_l2076_207646

theorem number_of_binders (total_sheets : ℕ) (sheets_per_binder : ℕ) 
  (h1 : total_sheets = 2450)
  (h2 : sheets_per_binder = 490)
  (h3 : total_sheets % sheets_per_binder = 0) :
  total_sheets / sheets_per_binder = 5 := by
  sorry

end NUMINAMATH_CALUDE_number_of_binders_l2076_207646


namespace NUMINAMATH_CALUDE_vegetable_planting_methods_l2076_207607

theorem vegetable_planting_methods (n m : ℕ) (hn : n = 4) (hm : m = 3) :
  (n.choose m) * (m.factorial) = 24 := by
  sorry

end NUMINAMATH_CALUDE_vegetable_planting_methods_l2076_207607


namespace NUMINAMATH_CALUDE_copper_in_mixture_l2076_207603

/-- Given a mixture of zinc and copper in the ratio 9:11 with a total weight of 60 kg,
    the amount of copper in the mixture is 33 kg. -/
theorem copper_in_mixture (zinc_ratio : ℕ) (copper_ratio : ℕ) (total_weight : ℝ) :
  zinc_ratio = 9 →
  copper_ratio = 11 →
  total_weight = 60 →
  (copper_ratio : ℝ) / ((zinc_ratio : ℝ) + (copper_ratio : ℝ)) * total_weight = 33 :=
by sorry

end NUMINAMATH_CALUDE_copper_in_mixture_l2076_207603


namespace NUMINAMATH_CALUDE_basketball_cost_l2076_207629

theorem basketball_cost (initial_amount : ℕ) (jersey_cost : ℕ) (jersey_count : ℕ) (shorts_cost : ℕ) (remaining_amount : ℕ) : 
  initial_amount = 50 →
  jersey_cost = 2 →
  jersey_count = 5 →
  shorts_cost = 8 →
  remaining_amount = 14 →
  initial_amount - (jersey_cost * jersey_count + shorts_cost + remaining_amount) = 18 :=
by sorry

end NUMINAMATH_CALUDE_basketball_cost_l2076_207629


namespace NUMINAMATH_CALUDE_employee_payments_l2076_207670

theorem employee_payments (total_payment : ℕ) (base_c : ℕ) (commission_c : ℕ) :
  total_payment = 2000 ∧
  base_c = 400 ∧
  commission_c = 100 →
  ∃ (payment_a payment_b payment_c : ℕ),
    payment_a = (3 * payment_b) / 2 ∧
    payment_c = base_c + commission_c ∧
    payment_a + payment_b + payment_c = total_payment ∧
    payment_a = 900 ∧
    payment_b = 600 ∧
    payment_c = 500 :=
by
  sorry


end NUMINAMATH_CALUDE_employee_payments_l2076_207670


namespace NUMINAMATH_CALUDE_approximate_cost_of_bicycle_and_fan_l2076_207639

/-- The cost of a bicycle in yuan -/
def bicycle_cost : ℕ := 389

/-- The cost of an electric fan in yuan -/
def fan_cost : ℕ := 189

/-- The approximate total cost of buying a bicycle and an electric fan -/
def approximate_total_cost : ℕ := 600

/-- Theorem stating that the approximate total cost is 600 yuan -/
theorem approximate_cost_of_bicycle_and_fan :
  ∃ (error : ℕ), bicycle_cost + fan_cost = approximate_total_cost + error ∧ error < 100 := by
  sorry

end NUMINAMATH_CALUDE_approximate_cost_of_bicycle_and_fan_l2076_207639


namespace NUMINAMATH_CALUDE_total_flowering_bulbs_l2076_207622

/-- Calculates the total number of small flowering bulbs that can be purchased given the costs and constraints. -/
theorem total_flowering_bulbs 
  (crocus_cost : ℚ)
  (daffodil_cost : ℚ)
  (total_budget : ℚ)
  (crocus_count : ℕ)
  (h1 : crocus_cost = 35/100)
  (h2 : daffodil_cost = 65/100)
  (h3 : total_budget = 2915/100)
  (h4 : crocus_count = 22) :
  ∃ (daffodil_count : ℕ), 
    (crocus_count : ℚ) * crocus_cost + (daffodil_count : ℚ) * daffodil_cost ≤ total_budget ∧
    crocus_count + daffodil_count = 55 :=
by sorry

end NUMINAMATH_CALUDE_total_flowering_bulbs_l2076_207622


namespace NUMINAMATH_CALUDE_game_prep_time_calculation_l2076_207609

/-- Calculates the total time before playing the main game --/
def totalGamePrepTime (downloadTime installTime updateTime accountTime issuesTime tutorialTime : ℕ) : ℕ :=
  downloadTime + installTime + updateTime + accountTime + issuesTime + tutorialTime

theorem game_prep_time_calculation :
  let downloadTime : ℕ := 10
  let installTime : ℕ := downloadTime / 2
  let updateTime : ℕ := downloadTime * 2
  let accountTime : ℕ := 5
  let issuesTime : ℕ := 15
  let preGameTime : ℕ := downloadTime + installTime + updateTime + accountTime + issuesTime
  let tutorialTime : ℕ := preGameTime * 3
  totalGamePrepTime downloadTime installTime updateTime accountTime issuesTime tutorialTime = 220 := by
  sorry

#eval totalGamePrepTime 10 5 20 5 15 165

end NUMINAMATH_CALUDE_game_prep_time_calculation_l2076_207609


namespace NUMINAMATH_CALUDE_pages_remaining_l2076_207620

/-- Given a book with 93 pages, if Jerry reads 30 pages on Saturday and 20 pages on Sunday,
    then the number of pages remaining to finish the book is 43. -/
theorem pages_remaining (total_pages : Nat) (pages_read_saturday : Nat) (pages_read_sunday : Nat)
    (h1 : total_pages = 93)
    (h2 : pages_read_saturday = 30)
    (h3 : pages_read_sunday = 20) :
    total_pages - pages_read_saturday - pages_read_sunday = 43 := by
  sorry

end NUMINAMATH_CALUDE_pages_remaining_l2076_207620


namespace NUMINAMATH_CALUDE_unique_positive_solution_l2076_207635

theorem unique_positive_solution :
  ∃! (x : ℝ), x > 0 ∧ (x - 6) / 12 = 6 / (x - 12) ∧ x = 18 := by
  sorry

end NUMINAMATH_CALUDE_unique_positive_solution_l2076_207635


namespace NUMINAMATH_CALUDE_h_has_two_roots_l2076_207602

/-- The function f(x) = 2x -/
def f (x : ℝ) : ℝ := 2 * x

/-- The function g(x) = 3 - x^2 -/
def g (x : ℝ) : ℝ := 3 - x^2

/-- The function h(x) = f(x) - g(x) -/
def h (x : ℝ) : ℝ := f x - g x

/-- The theorem stating that h(x) has exactly two distinct real roots -/
theorem h_has_two_roots : ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ h x₁ = 0 ∧ h x₂ = 0 ∧ ∀ x, h x = 0 → x = x₁ ∨ x = x₂ :=
sorry

end NUMINAMATH_CALUDE_h_has_two_roots_l2076_207602


namespace NUMINAMATH_CALUDE_prove_c_value_l2076_207606

-- Define the variables
variable (c k x y z : ℝ)

-- Define the conditions
axiom model : y = c * Real.exp (k * x)
axiom log_transform : z = Real.log y
axiom regression : z = 0.4 * x + 2

-- Theorem to prove
theorem prove_c_value : c = Real.exp 2 := by
  sorry

end NUMINAMATH_CALUDE_prove_c_value_l2076_207606


namespace NUMINAMATH_CALUDE_find_b_l2076_207644

-- Define the real number √3
noncomputable def sqrt3 : ℝ := Real.sqrt 3

-- Define the equation (1 + √3)^5 = a + b√3
def equation (a b : ℚ) : Prop := (1 + sqrt3) ^ 5 = a + b * sqrt3

-- Theorem statement
theorem find_b : ∃ (a b : ℚ), equation a b ∧ b = 44 := by
  sorry

end NUMINAMATH_CALUDE_find_b_l2076_207644


namespace NUMINAMATH_CALUDE_extreme_values_and_monotonicity_l2076_207683

-- Define the function f
def f (a b c x : ℝ) : ℝ := x^3 + a*x^2 + b*x + c

-- State the theorem
theorem extreme_values_and_monotonicity 
  (a b c : ℝ) 
  (h1 : ∃ (y : ℝ), y = f a b c (-2) ∧ (∀ x, f a b c x ≤ y))
  (h2 : ∃ (y : ℝ), y = f a b c 1 ∧ (∀ x, f a b c x ≤ y))
  (h3 : ∀ x ∈ Set.Icc (-1) 2, f a b c x < c^2) :
  (a = 3/2 ∧ b = -6) ∧ 
  (∀ x < -2, ∀ y ∈ Set.Ioo x (-2), f a b c x < f a b c y) ∧
  (∀ x ∈ Set.Ioo (-2) 1, ∀ y ∈ Set.Ioo x 1, f a b c x > f a b c y) ∧
  (∀ x > 1, ∀ y ∈ Set.Ioo 1 x, f a b c x > f a b c y) ∧
  (c > 2 ∨ c < -1) := by
  sorry


end NUMINAMATH_CALUDE_extreme_values_and_monotonicity_l2076_207683


namespace NUMINAMATH_CALUDE_remainder_2_1000_mod_17_l2076_207631

theorem remainder_2_1000_mod_17 (h : Prime 17) : 2^1000 % 17 = 0 := by
  sorry

end NUMINAMATH_CALUDE_remainder_2_1000_mod_17_l2076_207631


namespace NUMINAMATH_CALUDE_milk_fraction_is_two_thirds_l2076_207650

/-- Represents the content of a cup --/
structure CupContent where
  milk : ℚ
  honey : ℚ

/-- Performs the transfers between cups as described in the problem --/
def performTransfers (initial1 initial2 : CupContent) : CupContent × CupContent :=
  let afterFirstTransfer1 := CupContent.mk (initial1.milk / 2) 0
  let afterFirstTransfer2 := CupContent.mk (initial1.milk / 2) initial2.honey
  
  let totalSecond := afterFirstTransfer2.milk + afterFirstTransfer2.honey
  let secondToFirst := totalSecond / 2
  let milkRatio := afterFirstTransfer2.milk / totalSecond
  
  let afterSecondTransfer1 := CupContent.mk 
    (afterFirstTransfer1.milk + secondToFirst * milkRatio)
    (secondToFirst * (1 - milkRatio))
  let afterSecondTransfer2 := CupContent.mk 
    (afterFirstTransfer2.milk - secondToFirst * milkRatio)
    (afterFirstTransfer2.honey - secondToFirst * (1 - milkRatio))
  
  let thirdTransferAmount := (afterSecondTransfer1.milk + afterSecondTransfer1.honey) / 3
  let finalFirst := CupContent.mk 
    (afterSecondTransfer1.milk - thirdTransferAmount)
    afterSecondTransfer1.honey
  let finalSecond := CupContent.mk 
    (afterSecondTransfer2.milk + thirdTransferAmount)
    afterSecondTransfer2.honey
  
  (finalFirst, finalSecond)

/-- The main theorem stating that the fraction of milk in the second cup is 2/3 after transfers --/
theorem milk_fraction_is_two_thirds :
  let initial1 := CupContent.mk 8 0
  let initial2 := CupContent.mk 0 6
  let (_, finalSecond) := performTransfers initial1 initial2
  finalSecond.milk / (finalSecond.milk + finalSecond.honey) = 2/3 := by
  sorry


end NUMINAMATH_CALUDE_milk_fraction_is_two_thirds_l2076_207650


namespace NUMINAMATH_CALUDE_complex_power_sum_l2076_207647

theorem complex_power_sum (a b c : ℂ) 
  (h1 : Complex.abs a = 1) 
  (h2 : Complex.abs b = 1) 
  (h3 : Complex.abs c = 1) 
  (h4 : a^2 + b^2 + c^2 = 1) : 
  Complex.abs (a^2020 + b^2020 + c^2020) = 3 := by
sorry

end NUMINAMATH_CALUDE_complex_power_sum_l2076_207647


namespace NUMINAMATH_CALUDE_total_books_is_24_l2076_207654

/-- The number of boxes Victor bought -/
def num_boxes : ℕ := 8

/-- The number of books in each box -/
def books_per_box : ℕ := 3

/-- Theorem: The total number of books Victor bought is 24 -/
theorem total_books_is_24 : num_boxes * books_per_box = 24 := by
  sorry

end NUMINAMATH_CALUDE_total_books_is_24_l2076_207654


namespace NUMINAMATH_CALUDE_modulus_of_complex_square_l2076_207651

theorem modulus_of_complex_square : ∃ (z : ℂ), z = (3 - Complex.I)^2 ∧ Complex.abs z = 10 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_complex_square_l2076_207651


namespace NUMINAMATH_CALUDE_photo_arrangements_count_l2076_207652

/-- The number of ways to rearrange a group photo with the given conditions -/
def photo_arrangements : ℕ :=
  Nat.choose 8 2 * (5 * 4)

/-- Theorem stating that the number of photo arrangements is 560 -/
theorem photo_arrangements_count : photo_arrangements = 560 := by
  sorry

end NUMINAMATH_CALUDE_photo_arrangements_count_l2076_207652


namespace NUMINAMATH_CALUDE_music_stand_cost_l2076_207659

/-- The cost of Jason's music stand purchase --/
theorem music_stand_cost (flute_cost song_book_cost total_spent : ℝ) 
  (h1 : flute_cost = 142.46)
  (h2 : song_book_cost = 7)
  (h3 : total_spent = 158.35) :
  total_spent - (flute_cost + song_book_cost) = 8.89 := by
  sorry

end NUMINAMATH_CALUDE_music_stand_cost_l2076_207659


namespace NUMINAMATH_CALUDE_length_of_chord_AB_equation_of_line_PQ_l2076_207658

-- Define the circles and line
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 4
def line_l (x y : ℝ) : Prop := x - y + 2 = 0
def circle_C (x y : ℝ) : Prop := x^2 + y^2 = 2*x + 2*Real.sqrt 3*y

-- Theorem for the length of chord AB
theorem length_of_chord_AB :
  ∃ (A B : ℝ × ℝ),
    circle_O A.1 A.2 ∧ circle_O B.1 B.2 ∧
    line_l A.1 A.2 ∧ line_l B.1 B.2 ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2 * Real.sqrt 2 :=
sorry

-- Theorem for the equation of line PQ
theorem equation_of_line_PQ :
  ∃ (P Q : ℝ × ℝ),
    circle_O P.1 P.2 ∧ circle_O Q.1 Q.2 ∧
    circle_C P.1 P.2 ∧ circle_C Q.1 Q.2 ∧
    ∀ (x y : ℝ), (x - P.1) * (Q.2 - P.2) = (y - P.2) * (Q.1 - P.1) ↔
      x + Real.sqrt 3 * y - 2 = 0 :=
sorry

end NUMINAMATH_CALUDE_length_of_chord_AB_equation_of_line_PQ_l2076_207658


namespace NUMINAMATH_CALUDE_downstream_distance_l2076_207612

/-- Calculates the distance traveled downstream given boat speed, stream speed, and time -/
theorem downstream_distance
  (boat_speed : ℝ)
  (stream_speed : ℝ)
  (time : ℝ)
  (h1 : boat_speed = 14)
  (h2 : stream_speed = 6)
  (h3 : time = 3.6) :
  boat_speed + stream_speed * time = 72 :=
by sorry

end NUMINAMATH_CALUDE_downstream_distance_l2076_207612


namespace NUMINAMATH_CALUDE_find_e_l2076_207613

theorem find_e : ∃ e : ℕ, (1/5 : ℝ)^e * (1/4 : ℝ)^18 = 1 / (2 * 10^35) ∧ e = 35 := by
  sorry

end NUMINAMATH_CALUDE_find_e_l2076_207613


namespace NUMINAMATH_CALUDE_food_preferences_l2076_207698

theorem food_preferences (total students : ℕ)
  (french_fries burgers pizza tacos : ℕ)
  (fries_burgers fries_pizza fries_tacos : ℕ)
  (burgers_pizza burgers_tacos pizza_tacos : ℕ)
  (all_four : ℕ)
  (h_total : total = 30)
  (h_fries : french_fries = 18)
  (h_burgers : burgers = 12)
  (h_pizza : pizza = 14)
  (h_tacos : tacos = 10)
  (h_fries_burgers : fries_burgers = 8)
  (h_fries_pizza : fries_pizza = 6)
  (h_fries_tacos : fries_tacos = 4)
  (h_burgers_pizza : burgers_pizza = 5)
  (h_burgers_tacos : burgers_tacos = 3)
  (h_pizza_tacos : pizza_tacos = 7)
  (h_all_four : all_four = 2) :
  total - (french_fries + burgers + pizza + tacos
           - fries_burgers - fries_pizza - fries_tacos
           - burgers_pizza - burgers_tacos - pizza_tacos
           + all_four) = 11 := by
  sorry

end NUMINAMATH_CALUDE_food_preferences_l2076_207698


namespace NUMINAMATH_CALUDE_sarah_candy_theorem_l2076_207634

/-- The number of candy pieces Sarah received from her neighbors -/
def candy_from_neighbors : ℕ := sorry

/-- The number of candy pieces Sarah received from her older sister -/
def candy_from_sister : ℕ := 15

/-- The number of candy pieces Sarah ate per day -/
def candy_per_day : ℕ := 9

/-- The number of days the candy lasted -/
def days_lasted : ℕ := 9

/-- The total number of candy pieces Sarah had -/
def total_candy : ℕ := candy_per_day * days_lasted

theorem sarah_candy_theorem : candy_from_neighbors = 66 := by
  sorry

end NUMINAMATH_CALUDE_sarah_candy_theorem_l2076_207634


namespace NUMINAMATH_CALUDE_one_root_in_first_quadrant_l2076_207615

def complex_equation (z : ℂ) : Prop := z^7 = -1 + Complex.I * Real.sqrt 3

def is_in_first_quadrant (z : ℂ) : Prop :=
  z.re > 0 ∧ z.im > 0

theorem one_root_in_first_quadrant :
  ∃! z, complex_equation z ∧ is_in_first_quadrant z :=
sorry

end NUMINAMATH_CALUDE_one_root_in_first_quadrant_l2076_207615


namespace NUMINAMATH_CALUDE_circle_passes_through_point_l2076_207637

theorem circle_passes_through_point :
  ∀ (a b r : ℝ),
  b^2 = 8*a →                          -- Center (a, b) is on the parabola y² = 8x
  (a + 2)^2 + b^2 = r^2 →              -- Circle is tangent to the line x + 2 = 0
  (2 - a)^2 + b^2 = r^2 :=             -- Circle passes through (2, 0)
by
  sorry

end NUMINAMATH_CALUDE_circle_passes_through_point_l2076_207637


namespace NUMINAMATH_CALUDE_orange_harvest_duration_l2076_207674

/-- The number of sacks of oranges harvested per day -/
def sacks_per_day : ℕ := 14

/-- The total number of sacks of oranges harvested -/
def total_sacks : ℕ := 56

/-- The number of days the harvest lasts -/
def harvest_days : ℕ := total_sacks / sacks_per_day

theorem orange_harvest_duration :
  harvest_days = 4 := by
  sorry

end NUMINAMATH_CALUDE_orange_harvest_duration_l2076_207674


namespace NUMINAMATH_CALUDE_mutual_greetings_l2076_207627

theorem mutual_greetings (n : ℕ) (min_sent : ℕ) (h1 : n = 30) (h2 : min_sent = 16) :
  let total_sent := n * min_sent
  let total_pairs := n * (n - 1) / 2
  let mutual_greetings := {x : ℕ // x ≤ total_pairs ∧ 2 * x + (total_sent - 2 * x) ≤ total_sent}
  ∃ (x : mutual_greetings), x.val ≥ 45 :=
by sorry

end NUMINAMATH_CALUDE_mutual_greetings_l2076_207627


namespace NUMINAMATH_CALUDE_rogers_coins_l2076_207600

theorem rogers_coins (quarter_piles dime_piles coins_per_pile : ℕ) 
  (h1 : quarter_piles = 3)
  (h2 : dime_piles = 3)
  (h3 : coins_per_pile = 7) :
  quarter_piles * coins_per_pile + dime_piles * coins_per_pile = 42 :=
by sorry

end NUMINAMATH_CALUDE_rogers_coins_l2076_207600


namespace NUMINAMATH_CALUDE_sum_of_exponents_15_factorial_l2076_207667

def largest_perfect_cube_divisor (n : ℕ) : ℕ := sorry

def cube_root (n : ℕ) : ℕ := sorry

def sum_of_prime_exponents (n : ℕ) : ℕ := sorry

theorem sum_of_exponents_15_factorial : 
  sum_of_prime_exponents (cube_root (largest_perfect_cube_divisor (Nat.factorial 15))) = 6 := by sorry

end NUMINAMATH_CALUDE_sum_of_exponents_15_factorial_l2076_207667


namespace NUMINAMATH_CALUDE_part_one_part_two_l2076_207694

-- Define the quadratic function
def f (a b x : ℝ) := x^2 - a*x + b

-- Define the solution set condition
def solution_set (a b : ℝ) : Prop :=
  ∀ x, f a b x < 0 ↔ 2 < x ∧ x < 3

-- Part I
theorem part_one (a b : ℝ) (h : solution_set a b) : a + b = 11 := by
  sorry

-- Part II
def g (b c x : ℝ) := -x^2 + b*x + c

theorem part_two (b c : ℝ) (h1 : b = 6) 
  (h2 : ∀ x, g b c x ≤ 0) : c ≤ -9 := by
  sorry

end NUMINAMATH_CALUDE_part_one_part_two_l2076_207694


namespace NUMINAMATH_CALUDE_hair_sufficient_for_skin_l2076_207640

/-- Represents the state of having skin -/
def HasSkin : Prop := sorry

/-- Represents the state of having hair -/
def HasHair : Prop := sorry

/-- If there is no skin, there cannot be hair -/
axiom no_skin_no_hair : ¬HasSkin → ¬HasHair

/-- Prove that having hair is a sufficient condition for having skin -/
theorem hair_sufficient_for_skin : HasHair → HasSkin := by
  sorry

end NUMINAMATH_CALUDE_hair_sufficient_for_skin_l2076_207640


namespace NUMINAMATH_CALUDE_bacteria_growth_l2076_207618

/-- The number of times a bacteria culture doubles in 4 minutes -/
def doublings : ℕ := 240 / 30

/-- The final number of bacteria after 4 minutes -/
def final_count : ℕ := 524288

theorem bacteria_growth (n : ℕ) : n * 2^doublings = final_count ↔ n = 2048 := by
  sorry

end NUMINAMATH_CALUDE_bacteria_growth_l2076_207618


namespace NUMINAMATH_CALUDE_M_remainder_mod_32_l2076_207611

def M : ℕ := (List.filter (fun p => Nat.Prime p ∧ p % 2 = 1) (List.range 32)).prod

theorem M_remainder_mod_32 : M % 32 = 17 := by sorry

end NUMINAMATH_CALUDE_M_remainder_mod_32_l2076_207611


namespace NUMINAMATH_CALUDE_square_difference_equals_sixteen_l2076_207663

theorem square_difference_equals_sixteen
  (x y : ℝ)
  (sum_eq : x + y = 6)
  (product_eq : x * y = 5) :
  (x - y)^2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_equals_sixteen_l2076_207663


namespace NUMINAMATH_CALUDE_count_minimally_intersecting_mod_1000_l2076_207679

def Universe : Finset Nat := {1,2,3,4,5,6,7,8}

def MinimallyIntersecting (D E F : Finset Nat) : Prop :=
  (D ∩ E).card = 1 ∧ (E ∩ F).card = 1 ∧ (F ∩ D).card = 1 ∧ (D ∩ E ∩ F).card = 0

def CountMinimallyIntersecting : Nat :=
  (Finset.powerset Universe).card.choose 3

theorem count_minimally_intersecting_mod_1000 :
  CountMinimallyIntersecting % 1000 = 64 := by sorry

end NUMINAMATH_CALUDE_count_minimally_intersecting_mod_1000_l2076_207679


namespace NUMINAMATH_CALUDE_regular_polygon_angle_sum_l2076_207695

/-- For a regular polygon with n sides, if the sum of its interior angles
    is 4 times the sum of its exterior angles, then n = 10 -/
theorem regular_polygon_angle_sum (n : ℕ) : n ≥ 3 →
  (n - 2) * 180 = 4 * 360 → n = 10 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_angle_sum_l2076_207695


namespace NUMINAMATH_CALUDE_buddy_program_fraction_l2076_207626

theorem buddy_program_fraction (s n : ℕ) (hs : s > 0) (hn : n > 0) : 
  (n / 4 : ℚ) = (s / 3 : ℚ) → 
  ((n / 4 + s / 3) / (n + s) : ℚ) = 2 / 7 := by
sorry

end NUMINAMATH_CALUDE_buddy_program_fraction_l2076_207626


namespace NUMINAMATH_CALUDE_sticker_count_after_loss_l2076_207623

/-- Given a number of stickers per page, an initial number of pages, and a number of lost pages,
    calculate the total number of remaining stickers. -/
def remaining_stickers (stickers_per_page : ℕ) (initial_pages : ℕ) (lost_pages : ℕ) : ℕ :=
  (initial_pages - lost_pages) * stickers_per_page

theorem sticker_count_after_loss :
  remaining_stickers 20 12 1 = 220 := by
  sorry

end NUMINAMATH_CALUDE_sticker_count_after_loss_l2076_207623


namespace NUMINAMATH_CALUDE_total_metal_needed_l2076_207661

/-- Given that Charlie has 276 lbs of metal in storage and needs to buy an additional 359 lbs,
    prove that the total amount of metal he needs for the wings is 635 lbs. -/
theorem total_metal_needed (storage : ℕ) (additional : ℕ) (total : ℕ) 
    (h1 : storage = 276)
    (h2 : additional = 359)
    (h3 : total = storage + additional) : 
  total = 635 := by
  sorry

end NUMINAMATH_CALUDE_total_metal_needed_l2076_207661


namespace NUMINAMATH_CALUDE_cloth_trimming_l2076_207697

theorem cloth_trimming (x : ℝ) : 
  x > 0 → 
  (x - 4) * (x - 3) = 120 → 
  x = 12 :=
by sorry

end NUMINAMATH_CALUDE_cloth_trimming_l2076_207697


namespace NUMINAMATH_CALUDE_arccos_sqrt3_over_2_l2076_207662

theorem arccos_sqrt3_over_2 : Real.arccos (Real.sqrt 3 / 2) = π / 6 := by
  sorry

end NUMINAMATH_CALUDE_arccos_sqrt3_over_2_l2076_207662


namespace NUMINAMATH_CALUDE_sugar_used_in_two_minutes_l2076_207691

/-- Calculates the total sugar used in chocolate production over a given time period. -/
def sugarUsed (sugarPerBar : ℝ) (barsPerMinute : ℕ) (minutes : ℕ) : ℝ :=
  sugarPerBar * (barsPerMinute : ℝ) * (minutes : ℝ)

/-- Theorem stating that given the specified production parameters, 
    the total sugar used in two minutes is 108 grams. -/
theorem sugar_used_in_two_minutes :
  sugarUsed 1.5 36 2 = 108 := by
  sorry

end NUMINAMATH_CALUDE_sugar_used_in_two_minutes_l2076_207691


namespace NUMINAMATH_CALUDE_randy_pictures_l2076_207649

theorem randy_pictures (peter_pictures quincy_pictures randy_pictures total_pictures : ℕ) :
  peter_pictures = 8 →
  quincy_pictures = peter_pictures + 20 →
  total_pictures = 41 →
  total_pictures = peter_pictures + quincy_pictures + randy_pictures →
  randy_pictures = 5 := by
sorry

end NUMINAMATH_CALUDE_randy_pictures_l2076_207649


namespace NUMINAMATH_CALUDE_cliff_rock_ratio_l2076_207664

/-- Represents Cliff's rock collection -/
structure RockCollection where
  igneous : ℕ
  sedimentary : ℕ
  shinyIgneous : ℕ
  shinySedimentary : ℕ

/-- The properties of Cliff's rock collection -/
def cliffCollection : RockCollection where
  igneous := 90
  sedimentary := 180
  shinyIgneous := 30
  shinySedimentary := 36

theorem cliff_rock_ratio :
  let c := cliffCollection
  c.igneous + c.sedimentary = 270 ∧
  c.shinyIgneous = 30 ∧
  c.shinyIgneous = c.igneous / 3 ∧
  c.shinySedimentary = c.sedimentary / 5 →
  c.igneous / c.sedimentary = 1 / 2 := by
  sorry

#check cliff_rock_ratio

end NUMINAMATH_CALUDE_cliff_rock_ratio_l2076_207664
