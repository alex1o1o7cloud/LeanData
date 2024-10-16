import Mathlib

namespace NUMINAMATH_CALUDE_first_group_size_first_group_size_is_16_l2835_283544

/-- The number of days taken by the first group to complete the work -/
def days_first_group : ℝ := 25

/-- The number of men in the second group -/
def men_second_group : ℝ := 15

/-- The number of days taken by the second group to complete the work -/
def days_second_group : ℝ := 26.666666666666668

/-- The work done is inversely proportional to the number of days taken -/
axiom work_time_inverse_proportion {m1 m2 d1 d2 : ℝ} :
  m1 * d1 = m2 * d2

theorem first_group_size : ℝ := by
  sorry

theorem first_group_size_is_16 : first_group_size = 16 := by
  sorry

end NUMINAMATH_CALUDE_first_group_size_first_group_size_is_16_l2835_283544


namespace NUMINAMATH_CALUDE_books_ratio_l2835_283569

/-- Given the number of books Elmo, Laura, and Stu have, prove the ratio of Laura's books to Stu's books -/
theorem books_ratio (elmo_books laura_books stu_books : ℕ) : 
  elmo_books = 24 →
  stu_books = 4 →
  elmo_books = 3 * laura_books →
  laura_books / stu_books = 2 := by
sorry

end NUMINAMATH_CALUDE_books_ratio_l2835_283569


namespace NUMINAMATH_CALUDE_correct_units_l2835_283536

-- Define the volume units
inductive VolumeUnit
| Milliliter
| Liter

-- Define the containers
structure Container where
  name : String
  volume : ℕ
  unit : VolumeUnit

-- Define the given containers
def orangeJuiceCup : Container :=
  { name := "Cup of orange juice", volume := 500, unit := VolumeUnit.Milliliter }

def waterBottle : Container :=
  { name := "Water bottle", volume := 3, unit := VolumeUnit.Liter }

-- Theorem to prove
theorem correct_units :
  (orangeJuiceCup.unit = VolumeUnit.Milliliter) ∧
  (waterBottle.unit = VolumeUnit.Liter) :=
by sorry

end NUMINAMATH_CALUDE_correct_units_l2835_283536


namespace NUMINAMATH_CALUDE_cubic_root_sum_cubes_l2835_283510

theorem cubic_root_sum_cubes (a b c : ℂ) : 
  (a^3 - 2*a^2 + 3*a - 4 = 0) → 
  (b^3 - 2*b^2 + 3*b - 4 = 0) → 
  (c^3 - 2*c^2 + 3*c - 4 = 0) → 
  a^3 + b^3 + c^3 = 2 := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_sum_cubes_l2835_283510


namespace NUMINAMATH_CALUDE_problem_solution_l2835_283516

theorem problem_solution (a b c : ℝ) : 
  (∀ x : ℝ, (x - a) * (x - b) / (x - c) ≥ 0 ↔ x ≤ -2 ∨ |x - 30| < 2) →
  a < b →
  a + 2*b + 3*c = 86 :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l2835_283516


namespace NUMINAMATH_CALUDE_different_color_probability_l2835_283560

def totalChips : ℕ := 18

def blueChips : ℕ := 4
def greenChips : ℕ := 5
def redChips : ℕ := 6
def yellowChips : ℕ := 3

def probBlue : ℚ := blueChips / totalChips
def probGreen : ℚ := greenChips / totalChips
def probRed : ℚ := redChips / totalChips
def probYellow : ℚ := yellowChips / totalChips

theorem different_color_probability : 
  (probBlue * probGreen * probRed + 
   probBlue * probGreen * probYellow + 
   probBlue * probRed * probYellow + 
   probGreen * probRed * probYellow) * 6 = 141 / 162 := by sorry

end NUMINAMATH_CALUDE_different_color_probability_l2835_283560


namespace NUMINAMATH_CALUDE_equation_solution_l2835_283522

theorem equation_solution : ∃ x : ℝ, 61 + 5 * x / (180 / 3) = 62 ∧ x = 12 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2835_283522


namespace NUMINAMATH_CALUDE_sin_sum_difference_zero_l2835_283581

theorem sin_sum_difference_zero : 
  Real.sin (75 * π / 180) * Real.sin (165 * π / 180) - 
  Real.sin (15 * π / 180) * Real.sin (105 * π / 180) = 0 := by
  sorry

end NUMINAMATH_CALUDE_sin_sum_difference_zero_l2835_283581


namespace NUMINAMATH_CALUDE_mica_shopping_cost_l2835_283527

/-- The total cost of Mica's grocery shopping --/
def total_cost (pasta_price : ℝ) (pasta_quantity : ℝ) 
               (beef_price : ℝ) (beef_quantity : ℝ)
               (sauce_price : ℝ) (sauce_quantity : ℕ)
               (quesadilla_price : ℝ) : ℝ :=
  pasta_price * pasta_quantity + 
  beef_price * beef_quantity + 
  sauce_price * (sauce_quantity : ℝ) + 
  quesadilla_price

/-- Theorem stating that the total cost of Mica's shopping is $15 --/
theorem mica_shopping_cost : 
  total_cost 1.5 2 8 (1/4) 2 2 6 = 15 := by
  sorry

end NUMINAMATH_CALUDE_mica_shopping_cost_l2835_283527


namespace NUMINAMATH_CALUDE_sphere_radius_ratio_l2835_283529

theorem sphere_radius_ratio (V_large V_small : ℝ) (h1 : V_large = 450 * Real.pi) 
  (h2 : V_small = 0.25 * V_large) : 
  (V_small / V_large) ^ (1/3 : ℝ) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_sphere_radius_ratio_l2835_283529


namespace NUMINAMATH_CALUDE_cycle_original_price_l2835_283580

/-- Given a cycle sold at a 10% loss for Rs. 1080, prove its original price was Rs. 1200 -/
theorem cycle_original_price (selling_price : ℝ) (loss_percentage : ℝ) :
  selling_price = 1080 →
  loss_percentage = 10 →
  selling_price = (1 - loss_percentage / 100) * 1200 :=
by sorry

end NUMINAMATH_CALUDE_cycle_original_price_l2835_283580


namespace NUMINAMATH_CALUDE_ratio_calculation_l2835_283518

theorem ratio_calculation (A B C : ℚ) (h : A / B = 3 / 2 ∧ B / C = 2 / 5) :
  (2 * A + 3 * B) / (5 * C - 2 * A) = 12 / 19 := by
  sorry

end NUMINAMATH_CALUDE_ratio_calculation_l2835_283518


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_special_set_l2835_283578

theorem arithmetic_mean_of_special_set (n : ℕ) (h : n > 1) : 
  let set := [1 - 1 / n, 1 + 1 / n] ++ List.replicate (n - 1) 1
  (List.sum set) / (n + 1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_special_set_l2835_283578


namespace NUMINAMATH_CALUDE_scientific_notation_of_twelve_million_l2835_283513

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a positive real number to scientific notation -/
def to_scientific_notation (n : ℝ) : ScientificNotation :=
  sorry

/-- The number we want to convert to scientific notation -/
def target_number : ℝ := 12000000

/-- Theorem stating that the scientific notation of 12,000,000 is 1.2 × 10^7 -/
theorem scientific_notation_of_twelve_million :
  (to_scientific_notation target_number).coefficient = 1.2 ∧
  (to_scientific_notation target_number).exponent = 7 :=
sorry

end NUMINAMATH_CALUDE_scientific_notation_of_twelve_million_l2835_283513


namespace NUMINAMATH_CALUDE_sheetrock_area_is_30_l2835_283558

/-- Represents the area of a rectangular sheetrock given its length and width. -/
def sheetrockArea (length width : ℝ) : ℝ := length * width

/-- Theorem stating that the area of a rectangular sheetrock with length 6 feet and width 5 feet is 30 square feet. -/
theorem sheetrock_area_is_30 : sheetrockArea 6 5 = 30 := by
  sorry

end NUMINAMATH_CALUDE_sheetrock_area_is_30_l2835_283558


namespace NUMINAMATH_CALUDE_smallest_divisible_by_prime_main_result_l2835_283585

def consecutive_even_product (n : ℕ) : ℕ :=
  Finset.prod (Finset.range (n/2 + 1)) (λ i => 2 * i)

theorem smallest_divisible_by_prime (p : ℕ) (hp : Nat.Prime p) :
  (∀ m : ℕ, m < 2 * p → ¬(p ∣ consecutive_even_product m)) ∧
  (p ∣ consecutive_even_product (2 * p)) :=
sorry

theorem main_result : 
  ∀ n : ℕ, n < 63994 → ¬(31997 ∣ consecutive_even_product n) ∧
  31997 ∣ consecutive_even_product 63994 :=
sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_prime_main_result_l2835_283585


namespace NUMINAMATH_CALUDE_equation_sum_squares_l2835_283579

theorem equation_sum_squares (a b c x y z : ℝ) 
  (h1 : x / a + y / b + z / c = 4)
  (h2 : a / x + b / y + c / z = 3) :
  x^2 / a^2 + y^2 / b^2 + z^2 / c^2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_equation_sum_squares_l2835_283579


namespace NUMINAMATH_CALUDE_inequalities_proof_l2835_283548

theorem inequalities_proof (a b c : ℝ) (ha : a > 0) (hbc : a < b ∧ b < c) : 
  (a * b < b * c) ∧ 
  (a * c < b * c) ∧ 
  (a + b < b + c) ∧ 
  (c / a > 1) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_proof_l2835_283548


namespace NUMINAMATH_CALUDE_courtyard_length_l2835_283564

/-- Proves that a courtyard with given dimensions and number of bricks has a specific length -/
theorem courtyard_length 
  (width : ℝ) 
  (brick_length : ℝ) 
  (brick_width : ℝ) 
  (num_bricks : ℕ) : 
  width = 18 → 
  brick_length = 0.2 → 
  brick_width = 0.1 → 
  num_bricks = 22500 → 
  (width * (num_bricks * brick_length * brick_width / width)) = 25 := by
  sorry

end NUMINAMATH_CALUDE_courtyard_length_l2835_283564


namespace NUMINAMATH_CALUDE_paul_buys_two_toys_l2835_283538

/-- The number of toys Paul can buy given his savings, allowance, and toy price -/
def toys_paul_can_buy (savings : ℕ) (allowance : ℕ) (toy_price : ℕ) : ℕ :=
  (savings + allowance) / toy_price

/-- Theorem: Paul can buy 2 toys with his savings and allowance -/
theorem paul_buys_two_toys :
  toys_paul_can_buy 3 7 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_paul_buys_two_toys_l2835_283538


namespace NUMINAMATH_CALUDE_initial_daily_production_l2835_283532

/-- The number of days the company worked after the initial 3 days -/
def additional_days : ℕ := 20

/-- The total number of parts produced -/
def total_parts : ℕ := 675

/-- The number of extra parts produced beyond the plan -/
def extra_parts : ℕ := 100

/-- The daily increase in parts production after the initial 3 days -/
def daily_increase : ℕ := 5

theorem initial_daily_production :
  ∃ (x : ℕ),
    x > 0 ∧
    3 * x + additional_days * (x + daily_increase) = total_parts + extra_parts ∧
    x = 29 := by
  sorry

end NUMINAMATH_CALUDE_initial_daily_production_l2835_283532


namespace NUMINAMATH_CALUDE_banana_arrangement_count_l2835_283509

/-- The number of unique arrangements of the letters in "BANANA" -/
def banana_arrangements : ℕ := 60

/-- The total number of letters in "BANANA" -/
def total_letters : ℕ := 6

/-- The number of times 'A' appears in "BANANA" -/
def a_count : ℕ := 3

/-- The number of times 'N' appears in "BANANA" -/
def n_count : ℕ := 2

/-- The number of times 'B' appears in "BANANA" -/
def b_count : ℕ := 1

theorem banana_arrangement_count : 
  banana_arrangements = Nat.factorial total_letters / (Nat.factorial a_count * Nat.factorial n_count * Nat.factorial b_count) :=
by sorry

end NUMINAMATH_CALUDE_banana_arrangement_count_l2835_283509


namespace NUMINAMATH_CALUDE_sin_three_pi_halves_l2835_283570

theorem sin_three_pi_halves : Real.sin (3 * π / 2) = -1 := by
  sorry

end NUMINAMATH_CALUDE_sin_three_pi_halves_l2835_283570


namespace NUMINAMATH_CALUDE_weighted_coin_probability_l2835_283503

/-- Represents the weighting of the coin -/
inductive CoinWeight
| Heads
| Tails

/-- The probability of getting heads given the coin's weight -/
def prob_heads (w : CoinWeight) : ℚ :=
  match w with
  | CoinWeight.Heads => 2/3
  | CoinWeight.Tails => 1/3

/-- The probability of getting the observed result (two heads) given the coin's weight -/
def prob_observed (w : CoinWeight) : ℚ :=
  (prob_heads w) * (prob_heads w)

/-- The prior probability of each weighting -/
def prior_prob : CoinWeight → ℚ
| _ => 1/2

theorem weighted_coin_probability :
  let posterior_prob_heads := (prob_observed CoinWeight.Heads * prior_prob CoinWeight.Heads) /
    (prob_observed CoinWeight.Heads * prior_prob CoinWeight.Heads + 
     prob_observed CoinWeight.Tails * prior_prob CoinWeight.Tails)
  let prob_next_heads := posterior_prob_heads * prob_heads CoinWeight.Heads +
    (1 - posterior_prob_heads) * prob_heads CoinWeight.Tails
  prob_next_heads = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_weighted_coin_probability_l2835_283503


namespace NUMINAMATH_CALUDE_rational_coefficient_terms_count_l2835_283515

theorem rational_coefficient_terms_count (x y : ℝ) : 
  (Finset.filter (fun k => (k % 4 = 0) ∧ ((1200 - k) % 5 = 0)) (Finset.range 1201)).card = 61 := by
  sorry

end NUMINAMATH_CALUDE_rational_coefficient_terms_count_l2835_283515


namespace NUMINAMATH_CALUDE_order_of_mnpq_l2835_283576

theorem order_of_mnpq (m n p q : ℝ) 
  (h1 : m < n) 
  (h2 : p < q) 
  (h3 : (p - m) * (p - n) < 0) 
  (h4 : (q - m) * (q - n) < 0) : 
  m < p ∧ p < q ∧ q < n := by
  sorry

end NUMINAMATH_CALUDE_order_of_mnpq_l2835_283576


namespace NUMINAMATH_CALUDE_line_representation_l2835_283547

/-- A line in the xy-plane is represented by the equation y = k(x+1) -/
structure Line where
  k : ℝ

/-- The point (-1,0) in the xy-plane -/
def point : ℝ × ℝ := (-1, 0)

/-- A line passes through a point if the point satisfies the line's equation -/
def passes_through (l : Line) (p : ℝ × ℝ) : Prop :=
  p.2 = l.k * (p.1 + 1)

/-- A line is perpendicular to the x-axis if its slope is undefined (i.e., infinite) -/
def perpendicular_to_x_axis (l : Line) : Prop :=
  l.k = 0

/-- Main theorem: The equation y = k(x+1) represents all lines passing through
    the point (-1,0) and not perpendicular to the x-axis -/
theorem line_representation (l : Line) :
  (passes_through l point ∧ ¬perpendicular_to_x_axis l) ↔ 
  ∃ (k : ℝ), l = Line.mk k :=
sorry

end NUMINAMATH_CALUDE_line_representation_l2835_283547


namespace NUMINAMATH_CALUDE_largest_cube_forming_integer_l2835_283598

theorem largest_cube_forming_integer : 
  ∀ n : ℕ, n > 19 → ¬∃ k : ℤ, n^3 + 4*n^2 - 15*n - 18 = k^3 :=
by sorry

end NUMINAMATH_CALUDE_largest_cube_forming_integer_l2835_283598


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_l2835_283506

/-- A line in 2D space represented by its equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a line passes through a point -/
def Line.passesThrough (l : Line) (p : Point) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if two lines are perpendicular -/
def Line.isPerpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

theorem perpendicular_line_equation :
  ∃ (l : Line),
    l.passesThrough ⟨1, -2⟩ ∧
    l.isPerpendicular ⟨2, 3, -1⟩ ∧
    l = ⟨3, -2, -7⟩ := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_line_equation_l2835_283506


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l2835_283507

-- Problem 1
theorem problem_1 (a b : ℝ) (h1 : |a| = 5) (h2 : |b| = 3) (h3 : a > 0) (h4 : b > 0) :
  a + b = 8 := by sorry

-- Problem 2
theorem problem_2 (a b c : ℝ) (h : |a - 2| + |b - 3| + |c - 4| = 0) :
  a + b + c = 9 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l2835_283507


namespace NUMINAMATH_CALUDE_oak_willow_difference_l2835_283583

theorem oak_willow_difference (total_trees : ℕ) (willows : ℕ) 
  (h1 : total_trees = 83) (h2 : willows = 36) : total_trees - willows - willows = 11 := by
  sorry

end NUMINAMATH_CALUDE_oak_willow_difference_l2835_283583


namespace NUMINAMATH_CALUDE_veggie_patty_percentage_l2835_283543

/-- Proves that the percentage of a veggie patty that is not made up of spices and additives is 70% -/
theorem veggie_patty_percentage (total_weight spice_weight : ℝ) 
  (h1 : total_weight = 150)
  (h2 : spice_weight = 45) :
  (total_weight - spice_weight) / total_weight * 100 = 70 := by
  sorry

end NUMINAMATH_CALUDE_veggie_patty_percentage_l2835_283543


namespace NUMINAMATH_CALUDE_missy_capacity_l2835_283587

/-- The number of claims each agent can handle -/
structure AgentCapacity where
  jan : ℕ
  john : ℕ
  missy : ℕ

/-- Calculate the capacity of insurance agents based on given conditions -/
def calculate_capacity : AgentCapacity :=
  let jan_capacity := 20
  let john_capacity := jan_capacity + (jan_capacity * 30 / 100)
  let missy_capacity := john_capacity + 15
  { jan := jan_capacity,
    john := john_capacity,
    missy := missy_capacity }

/-- Theorem stating that Missy can handle 41 claims -/
theorem missy_capacity : (calculate_capacity).missy = 41 := by
  sorry

end NUMINAMATH_CALUDE_missy_capacity_l2835_283587


namespace NUMINAMATH_CALUDE_perpendicular_lines_a_value_l2835_283591

theorem perpendicular_lines_a_value (a : ℝ) :
  (∀ x y : ℝ, (a * x + y - 1 = 0 ∧ x - y + 3 = 0) → 
   (a * 1 + (-1) * 1 = -1)) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_lines_a_value_l2835_283591


namespace NUMINAMATH_CALUDE_factor_sum_l2835_283546

/-- If x^2 + 2√2x + 5 is a factor of x^4 + Px^2 + Q, then P + Q = 27 -/
theorem factor_sum (P Q : ℝ) : 
  (∃ b c : ℝ, ∀ x : ℝ, (x^2 + 2 * Real.sqrt 2 * x + 5) * (x^2 + b * x + c) = x^4 + P * x^2 + Q) →
  P + Q = 27 := by
  sorry

end NUMINAMATH_CALUDE_factor_sum_l2835_283546


namespace NUMINAMATH_CALUDE_fifth_term_of_sequence_l2835_283540

def geometric_sequence (a₁ : ℝ) (r : ℝ) (n : ℕ) : ℝ := a₁ * r ^ (n - 1)

theorem fifth_term_of_sequence (y : ℝ) :
  let a₁ := 3
  let r := 3 * y
  geometric_sequence a₁ r 5 = 243 * y^4 := by
  sorry

end NUMINAMATH_CALUDE_fifth_term_of_sequence_l2835_283540


namespace NUMINAMATH_CALUDE_chips_yield_more_ounces_l2835_283575

def total_ounces (budget : ℚ) (price_per_bag : ℚ) (ounces_per_bag : ℚ) : ℚ :=
  (budget / price_per_bag).floor * ounces_per_bag

theorem chips_yield_more_ounces : 
  let budget : ℚ := 7
  let candy_price : ℚ := 1
  let candy_ounces : ℚ := 12
  let chips_price : ℚ := 1.4
  let chips_ounces : ℚ := 17
  total_ounces budget chips_price chips_ounces > total_ounces budget candy_price candy_ounces := by
  sorry

end NUMINAMATH_CALUDE_chips_yield_more_ounces_l2835_283575


namespace NUMINAMATH_CALUDE_loan_principal_calculation_l2835_283539

/-- Calculates the principal amount given the interest rate, time, and total interest --/
def calculate_principal (rate : ℚ) (time : ℕ) (interest : ℚ) : ℚ :=
  (interest * 100) / (rate * time)

/-- Theorem: Given a loan with 12% per annum simple interest rate, 
    if the interest after 3 years is 4320, then the principal amount borrowed was 12000 --/
theorem loan_principal_calculation :
  let rate : ℚ := 12
  let time : ℕ := 3
  let interest : ℚ := 4320
  calculate_principal rate time interest = 12000 := by
  sorry

end NUMINAMATH_CALUDE_loan_principal_calculation_l2835_283539


namespace NUMINAMATH_CALUDE_decision_box_is_diamond_l2835_283526

/-- A type representing different shapes that can be used in a flowchart --/
inductive FlowchartShape
  | Rectangle
  | Diamond
  | Oval
  | Parallelogram

/-- A function that returns the shape used for decision boxes in a flowchart --/
def decisionBoxShape : FlowchartShape := FlowchartShape.Diamond

/-- Theorem stating that the decision box in a flowchart is represented by a diamond shape --/
theorem decision_box_is_diamond : decisionBoxShape = FlowchartShape.Diamond := by
  sorry

end NUMINAMATH_CALUDE_decision_box_is_diamond_l2835_283526


namespace NUMINAMATH_CALUDE_hostel_accommodation_l2835_283557

/-- Proves that 20 additional students were accommodated in the hostel --/
theorem hostel_accommodation :
  ∀ (initial_students : ℕ) 
    (initial_avg_expenditure : ℚ)
    (avg_decrease : ℚ)
    (total_increase : ℕ)
    (new_total_expenditure : ℕ),
  initial_students = 100 →
  avg_decrease = 5 →
  total_increase = 400 →
  new_total_expenditure = 5400 →
  ∃ (additional_students : ℕ),
    additional_students = 20 ∧
    (initial_avg_expenditure - avg_decrease) * (initial_students + additional_students) = new_total_expenditure :=
by sorry

end NUMINAMATH_CALUDE_hostel_accommodation_l2835_283557


namespace NUMINAMATH_CALUDE_range_of_omega_l2835_283541

/-- Given a function f and its shifted version g, prove the range of ω -/
theorem range_of_omega (f g : ℝ → ℝ) (ω : ℝ) : 
  (ω > 0) →
  (∀ x, f x = Real.sin (π / 3 - ω * x)) →
  (∀ x, g x = Real.sin (ω * x - π / 3)) →
  (∀ x ∈ Set.Icc 0 π, -Real.sqrt 3 / 2 ≤ g x ∧ g x ≤ 1) →
  (5 / 6 : ℝ) ≤ ω ∧ ω ≤ (5 / 3 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_range_of_omega_l2835_283541


namespace NUMINAMATH_CALUDE_train_passing_time_l2835_283520

/-- Time for a train to pass a moving platform -/
theorem train_passing_time (train_length platform_length : ℝ) 
  (train_speed platform_speed : ℝ) : 
  train_length = 157 →
  platform_length = 283 →
  train_speed = 72 →
  platform_speed = 18 →
  (train_length + platform_length) / ((train_speed - platform_speed) * (1000 / 3600)) = 
    440 / (54 * (1000 / 3600)) := by
  sorry

end NUMINAMATH_CALUDE_train_passing_time_l2835_283520


namespace NUMINAMATH_CALUDE_product_at_one_zeros_of_h_monotonicity_of_h_l2835_283505

-- Define the functions f and g
def f (x : ℝ) : ℝ := 2 * x - 4
def g (x : ℝ) : ℝ := -x + 4

-- Define the product function h
def h (x : ℝ) : ℝ := f x * g x

-- Theorem 1: f(1) * g(1) = -6
theorem product_at_one : h 1 = -6 := by sorry

-- Theorem 2: The zeros of h are x = 2 and x = 4
theorem zeros_of_h : ∀ x : ℝ, h x = 0 ↔ x = 2 ∨ x = 4 := by sorry

-- Theorem 3: h is increasing on (-∞, 3] and decreasing on [3, ∞)
theorem monotonicity_of_h :
  (∀ x y : ℝ, x ≤ y ∧ y ≤ 3 → h x ≤ h y) ∧
  (∀ x y : ℝ, 3 ≤ x ∧ x ≤ y → h x ≥ h y) := by sorry

end NUMINAMATH_CALUDE_product_at_one_zeros_of_h_monotonicity_of_h_l2835_283505


namespace NUMINAMATH_CALUDE_lucas_sum_is_19_89_l2835_283528

/-- Lucas numbers sequence -/
def lucas : ℕ → ℕ
  | 0 => 2
  | 1 => 1
  | (n + 2) => lucas (n + 1) + lucas n

/-- Function to get the nth digit of a Lucas number (with overlapping) -/
def lucasDigit (n : ℕ) : ℕ :=
  lucas n % 10

/-- The infinite sum of Lucas digits divided by increasing powers of 10 -/
noncomputable def r : ℚ :=
  ∑' n, (lucasDigit n : ℚ) / 10^(n + 1)

/-- Main theorem: The sum of Lucas digits is equal to 19/89 -/
theorem lucas_sum_is_19_89 : r = 19 / 89 := by
  sorry

end NUMINAMATH_CALUDE_lucas_sum_is_19_89_l2835_283528


namespace NUMINAMATH_CALUDE_expression_value_l2835_283521

theorem expression_value : 
  let a : ℚ := 1/3
  let b : ℚ := 3
  (2 * a⁻¹ + a⁻¹ / b) / a = 21 := by sorry

end NUMINAMATH_CALUDE_expression_value_l2835_283521


namespace NUMINAMATH_CALUDE_binomial_coefficient_20_19_l2835_283504

theorem binomial_coefficient_20_19 : (Nat.choose 20 19) = 20 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_20_19_l2835_283504


namespace NUMINAMATH_CALUDE_min_coach_handshakes_l2835_283574

/-- The total number of handshakes at the gymnastics meet -/
def total_handshakes : ℕ := 903

/-- The number of gymnasts at the meet -/
def n : ℕ := 43

/-- The number of coaches at the meet -/
def num_coaches : ℕ := 3

/-- Function to calculate the number of handshakes between gymnasts -/
def gymnast_handshakes (m : ℕ) : ℕ := m * (m - 1) / 2

/-- Theorem stating the minimum number of handshakes involving coaches -/
theorem min_coach_handshakes : 
  ∃ (k₁ k₂ k₃ : ℕ), 
    gymnast_handshakes n + k₁ + k₂ + k₃ = total_handshakes ∧ 
    k₁ + k₂ + k₃ = 0 := by
  sorry

end NUMINAMATH_CALUDE_min_coach_handshakes_l2835_283574


namespace NUMINAMATH_CALUDE_line_circle_properties_l2835_283553

-- Define the line l
def line_l (k : ℝ) (x y : ℝ) : Prop := k * x - y + 3 * k + 1 = 0

-- Define the circle O
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 16

theorem line_circle_properties :
  -- 1. Line l always passes through (-3, 1)
  (∀ k : ℝ, line_l k (-3) 1) ∧
  -- 2. Maximum distance from (0, 0) to line l is √10
  (∃ d : ℝ, d = Real.sqrt 10 ∧ 
    ∀ k x y : ℝ, line_l k x y → (x^2 + y^2 : ℝ) ≤ d^2) ∧
  -- 3. Line l intersects circle O for all k
  (∀ k : ℝ, ∃ x y : ℝ, line_l k x y ∧ circle_O x y) :=
sorry

end NUMINAMATH_CALUDE_line_circle_properties_l2835_283553


namespace NUMINAMATH_CALUDE_total_area_is_1800_l2835_283596

/-- Calculates the total area of rooms given initial dimensions and modifications -/
def total_area (length width increase_amount : ℕ) : ℕ :=
  let new_length := length + increase_amount
  let new_width := width + increase_amount
  let single_room_area := new_length * new_width
  let four_rooms_area := 4 * single_room_area
  let double_room_area := 2 * single_room_area
  four_rooms_area + double_room_area

/-- Theorem stating that the total area of rooms is 1800 square feet -/
theorem total_area_is_1800 :
  total_area 13 18 2 = 1800 := by sorry

end NUMINAMATH_CALUDE_total_area_is_1800_l2835_283596


namespace NUMINAMATH_CALUDE_product_of_polynomials_l2835_283533

theorem product_of_polynomials (p q : ℚ) : 
  (∀ d : ℚ, (8 * d^2 - 4 * d + p) * (4 * d^2 + q * d - 9) = 32 * d^4 - 68 * d^3 + 5 * d^2 + 23 * d - 36) →
  p + q = 3/4 := by
sorry

end NUMINAMATH_CALUDE_product_of_polynomials_l2835_283533


namespace NUMINAMATH_CALUDE_f_inequality_solution_set_l2835_283577

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := (m + 1) * x^2 - (m - 1) * x + m - 1

-- State the theorem
theorem f_inequality_solution_set (m : ℝ) :
  (∀ x : ℝ, f m x < 1) ↔ m < (1 - 2 * Real.sqrt 7) / 3 :=
sorry

end NUMINAMATH_CALUDE_f_inequality_solution_set_l2835_283577


namespace NUMINAMATH_CALUDE_parallel_lines_distance_l2835_283511

/-- A circle intersected by three equally spaced parallel lines -/
structure CircleWithParallelLines where
  /-- The radius of the circle -/
  r : ℝ
  /-- The distance between adjacent parallel lines -/
  d : ℝ
  /-- The length of the first chord -/
  chord1 : ℝ
  /-- The length of the second chord -/
  chord2 : ℝ
  /-- The length of the third chord -/
  chord3 : ℝ
  /-- The first chord has length 42 -/
  chord1_eq : chord1 = 42
  /-- The second chord has length 42 -/
  chord2_eq : chord2 = 42
  /-- The third chord has length 40 -/
  chord3_eq : chord3 = 40

/-- The theorem stating that the distance between adjacent parallel lines is 3 3/8 -/
theorem parallel_lines_distance (c : CircleWithParallelLines) : c.d = 3 + 3 / 8 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_distance_l2835_283511


namespace NUMINAMATH_CALUDE_equation_has_three_solutions_l2835_283501

-- Define the equation
def f (x : ℝ) : Prop := Real.sqrt (9 - x) = x^2 * Real.sqrt (9 - x)

-- Theorem statement
theorem equation_has_three_solutions :
  ∃ (a b c : ℝ), (a ≠ b ∧ b ≠ c ∧ a ≠ c) ∧ 
  (f a ∧ f b ∧ f c) ∧
  (∀ x : ℝ, f x → (x = a ∨ x = b ∨ x = c)) :=
sorry

end NUMINAMATH_CALUDE_equation_has_three_solutions_l2835_283501


namespace NUMINAMATH_CALUDE_difference_of_y_coordinates_is_two_l2835_283563

noncomputable def e : ℝ := Real.exp 1

theorem difference_of_y_coordinates_is_two :
  ∀ a b : ℝ,
  (a^2 + e^4 = 2 * e^2 * a + 1) →
  (b^2 + e^4 = 2 * e^2 * b + 1) →
  a ≠ b →
  |a - b| = 2 := by
sorry

end NUMINAMATH_CALUDE_difference_of_y_coordinates_is_two_l2835_283563


namespace NUMINAMATH_CALUDE_weight_range_proof_l2835_283593

/-- Given the weights of Tracy, John, and Jake, prove that the range of their weights is 14 kg -/
theorem weight_range_proof (tracy_weight john_weight jake_weight : ℕ) 
  (h1 : tracy_weight + john_weight + jake_weight = 158)
  (h2 : tracy_weight = 52)
  (h3 : jake_weight = tracy_weight + 8) :
  (max tracy_weight (max john_weight jake_weight)) - 
  (min tracy_weight (min john_weight jake_weight)) = 14 := by
  sorry

#check weight_range_proof

end NUMINAMATH_CALUDE_weight_range_proof_l2835_283593


namespace NUMINAMATH_CALUDE_rectangle_to_squares_l2835_283562

/-- Represents a rectangle with given length and width -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Represents a square with a given side length -/
structure Square where
  side : ℝ

/-- Function to cut a rectangle in half across its length -/
def cutRectangleInHalf (r : Rectangle) : Square :=
  { side := r.width }

theorem rectangle_to_squares (r : Rectangle) 
  (h1 : r.length = 10)
  (h2 : r.width = 5) :
  (cutRectangleInHalf r).side = 5 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_to_squares_l2835_283562


namespace NUMINAMATH_CALUDE_nathan_tokens_used_l2835_283551

/-- The number of tokens Nathan used at the arcade -/
def tokens_used (air_hockey_games basketball_games tokens_per_game : ℕ) : ℕ :=
  (air_hockey_games + basketball_games) * tokens_per_game

/-- Theorem: Nathan used 18 tokens at the arcade -/
theorem nathan_tokens_used :
  tokens_used 2 4 3 = 18 := by
  sorry

end NUMINAMATH_CALUDE_nathan_tokens_used_l2835_283551


namespace NUMINAMATH_CALUDE_inequality_proof_l2835_283552

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a * b * c ≥ (a + b + c) / (1 / a^2 + 1 / b^2 + 1 / c^2) ∧
  (a + b + c) / (1 / a^2 + 1 / b^2 + 1 / c^2) ≥ (a + b - c) * (b + c - a) * (c + a - b) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l2835_283552


namespace NUMINAMATH_CALUDE_jim_grove_other_row_l2835_283565

/-- The number of lemons produced by a normal lemon tree per year -/
def normal_lemon_production : ℕ := 60

/-- The percentage increase in lemon production for Jim's engineered trees -/
def engineered_production_increase : ℚ := 50 / 100

/-- The number of trees in one row of Jim's grove -/
def trees_in_one_row : ℕ := 50

/-- The total number of lemons produced by Jim's grove in 5 years -/
def total_lemons_produced : ℕ := 675000

/-- The number of years of lemon production -/
def years_of_production : ℕ := 5

/-- The number of trees in the other row of Jim's grove -/
def trees_in_other_row : ℕ := 1450

theorem jim_grove_other_row :
  trees_in_other_row = 
    (total_lemons_produced / (normal_lemon_production * (1 + engineered_production_increase) * years_of_production)).floor - trees_in_one_row :=
by sorry

end NUMINAMATH_CALUDE_jim_grove_other_row_l2835_283565


namespace NUMINAMATH_CALUDE_arcsin_neg_sqrt2_over_2_l2835_283597

theorem arcsin_neg_sqrt2_over_2 : Real.arcsin (-Real.sqrt 2 / 2) = -π / 4 := by sorry

end NUMINAMATH_CALUDE_arcsin_neg_sqrt2_over_2_l2835_283597


namespace NUMINAMATH_CALUDE_emmy_has_200_l2835_283550

/-- The amount of money Emmy has -/
def emmys_money : ℕ := sorry

/-- The amount of money Gerry has -/
def gerrys_money : ℕ := 100

/-- The cost of one apple -/
def apple_cost : ℕ := 2

/-- The total number of apples Emmy and Gerry can buy -/
def total_apples : ℕ := 150

/-- Theorem: Emmy has $200 -/
theorem emmy_has_200 : emmys_money = 200 := by
  have total_cost : ℕ := apple_cost * total_apples
  have sum_of_money : emmys_money + gerrys_money = total_cost := sorry
  sorry

end NUMINAMATH_CALUDE_emmy_has_200_l2835_283550


namespace NUMINAMATH_CALUDE_saltwater_aquariums_count_l2835_283592

/-- The number of saltwater aquariums Tyler has -/
def saltwater_aquariums : ℕ := 2184 / 39

/-- The number of freshwater aquariums Tyler has -/
def freshwater_aquariums : ℕ := 10

/-- The number of animals in each aquarium -/
def animals_per_aquarium : ℕ := 39

/-- The total number of saltwater animals Tyler has -/
def total_saltwater_animals : ℕ := 2184

theorem saltwater_aquariums_count : saltwater_aquariums = 56 := by
  sorry

end NUMINAMATH_CALUDE_saltwater_aquariums_count_l2835_283592


namespace NUMINAMATH_CALUDE_function_inequality_l2835_283537

/-- Given functions f and g, prove that if 2f(x) ≥ g(x) for all x > 0, then a ≤ 4 -/
theorem function_inequality (a : ℝ) : 
  (∀ x > 0, 2 * (x * Real.log x) ≥ -x^2 + a*x - 3) → a ≤ 4 := by
  sorry


end NUMINAMATH_CALUDE_function_inequality_l2835_283537


namespace NUMINAMATH_CALUDE_goldfish_equality_l2835_283512

/-- The number of months after which Alice and Bob have the same number of goldfish -/
def same_goldfish_month : ℕ := 7

/-- Alice's initial number of goldfish -/
def alice_initial : ℕ := 3

/-- Bob's initial number of goldfish -/
def bob_initial : ℕ := 256

/-- Alice's goldfish growth rate per month -/
def alice_growth_rate : ℕ := 3

/-- Bob's goldfish growth rate per month -/
def bob_growth_rate : ℕ := 4

/-- Alice's number of goldfish after n months -/
def alice_goldfish (n : ℕ) : ℕ := alice_initial * (alice_growth_rate ^ n)

/-- Bob's number of goldfish after n months -/
def bob_goldfish (n : ℕ) : ℕ := bob_initial * (bob_growth_rate ^ n)

theorem goldfish_equality :
  alice_goldfish same_goldfish_month = bob_goldfish same_goldfish_month ∧
  ∀ m : ℕ, m < same_goldfish_month → alice_goldfish m ≠ bob_goldfish m :=
by sorry

end NUMINAMATH_CALUDE_goldfish_equality_l2835_283512


namespace NUMINAMATH_CALUDE_remainder_theorem_l2835_283517

def dividend (k : ℤ) (x : ℝ) : ℝ := 3 * x^3 + k * x^2 + 8 * x - 24

def divisor (x : ℝ) : ℝ := 3 * x + 4

theorem remainder_theorem (k : ℤ) :
  (∃ q : ℝ → ℝ, ∀ x, dividend k x = (divisor x) * (q x) + 5) ↔ k = 29 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l2835_283517


namespace NUMINAMATH_CALUDE_leap_day_2024_is_thursday_l2835_283530

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a date -/
structure Date where
  year : ℕ
  month : ℕ
  day : ℕ

/-- Function to determine if a year is a leap year -/
def isLeapYear (year : ℕ) : Bool :=
  (year % 4 == 0 && year % 100 ≠ 0) || (year % 400 == 0)

/-- Function to calculate the number of days between two dates -/
def daysBetween (date1 date2 : Date) : ℕ :=
  sorry

/-- Function to determine the day of the week given a starting day and number of days passed -/
def getDayOfWeek (startDay : DayOfWeek) (daysPassed : ℕ) : DayOfWeek :=
  sorry

theorem leap_day_2024_is_thursday :
  let leap_day_1996 : Date := ⟨1996, 2, 29⟩
  let leap_day_2024 : Date := ⟨2024, 2, 29⟩
  let days_between := daysBetween leap_day_1996 leap_day_2024
  getDayOfWeek DayOfWeek.Thursday days_between = DayOfWeek.Thursday :=
sorry

end NUMINAMATH_CALUDE_leap_day_2024_is_thursday_l2835_283530


namespace NUMINAMATH_CALUDE_equation_solution_l2835_283568

theorem equation_solution (x : ℝ) : 
  Real.sqrt (1 + Real.sqrt (4 + Real.sqrt (2 * x + 3))) = (1 + Real.sqrt (2 * x + 3)) ^ (1/4) → 
  x = -23/32 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l2835_283568


namespace NUMINAMATH_CALUDE_hit_at_least_once_complement_of_miss_all_l2835_283535

-- Define the sample space
def Ω : Type := Fin 3 → Bool

-- Define the event of hitting the target at least once
def hit_at_least_once (ω : Ω) : Prop :=
  ∃ i, ω i = true

-- Define the event of not hitting the target at all
def miss_all (ω : Ω) : Prop :=
  ∀ i, ω i = false

-- Theorem statement
theorem hit_at_least_once_complement_of_miss_all :
  ∀ ω : Ω, hit_at_least_once ω ↔ ¬(miss_all ω) :=
sorry

end NUMINAMATH_CALUDE_hit_at_least_once_complement_of_miss_all_l2835_283535


namespace NUMINAMATH_CALUDE_inequality_preservation_l2835_283571

theorem inequality_preservation (a b : ℝ) (h : a > b) : a - 5 > b - 5 := by
  sorry

end NUMINAMATH_CALUDE_inequality_preservation_l2835_283571


namespace NUMINAMATH_CALUDE_full_price_revenue_is_2128_l2835_283555

/-- Represents the ticket sale scenario -/
structure TicketSale where
  total_tickets : ℕ
  total_revenue : ℕ
  full_price_tickets : ℕ
  discounted_tickets : ℕ
  full_price : ℕ

/-- The conditions of the ticket sale -/
def valid_ticket_sale (sale : TicketSale) : Prop :=
  sale.total_tickets = 200 ∧
  sale.total_revenue = 2688 ∧
  sale.full_price_tickets + sale.discounted_tickets = sale.total_tickets ∧
  sale.full_price_tickets * sale.full_price + sale.discounted_tickets * (sale.full_price / 3) = sale.total_revenue

/-- The theorem to be proved -/
theorem full_price_revenue_is_2128 (sale : TicketSale) :
  valid_ticket_sale sale →
  sale.full_price_tickets * sale.full_price = 2128 :=
by sorry

end NUMINAMATH_CALUDE_full_price_revenue_is_2128_l2835_283555


namespace NUMINAMATH_CALUDE_line_vector_at_negative_two_l2835_283582

/-- A line in a plane parameterized by s -/
def Line := ℝ → ℝ × ℝ

/-- The vector on the line at a given s -/
def vectorAtS (line : Line) (s : ℝ) : ℝ × ℝ := line s

theorem line_vector_at_negative_two 
  (line : Line) 
  (h1 : vectorAtS line 1 = (2, 5)) 
  (h4 : vectorAtS line 4 = (8, -7)) : 
  vectorAtS line (-2) = (-4, 17) := by
sorry

end NUMINAMATH_CALUDE_line_vector_at_negative_two_l2835_283582


namespace NUMINAMATH_CALUDE_thirtieth_term_is_59_l2835_283572

def arithmetic_sequence (n : ℕ) : ℕ := 2 * n - 1

theorem thirtieth_term_is_59 : arithmetic_sequence 30 = 59 := by
  sorry

end NUMINAMATH_CALUDE_thirtieth_term_is_59_l2835_283572


namespace NUMINAMATH_CALUDE_intersected_cubes_count_l2835_283566

-- Define a cube structure
structure Cube where
  size : ℕ
  unit_cubes : ℕ

-- Define a plane that bisects the diagonal
structure BisectingPlane where
  perpendicular_to_diagonal : Bool
  bisects_diagonal : Bool

-- Define the function to count intersected cubes
def count_intersected_cubes (c : Cube) (p : BisectingPlane) : ℕ :=
  sorry

-- Theorem statement
theorem intersected_cubes_count 
  (c : Cube) 
  (p : BisectingPlane) 
  (h1 : c.size = 4) 
  (h2 : c.unit_cubes = 64) 
  (h3 : p.perpendicular_to_diagonal = true) 
  (h4 : p.bisects_diagonal = true) : 
  count_intersected_cubes c p = 24 :=
sorry

end NUMINAMATH_CALUDE_intersected_cubes_count_l2835_283566


namespace NUMINAMATH_CALUDE_cricketer_matches_l2835_283523

theorem cricketer_matches (score1 score2 overall_avg : ℚ) (matches1 matches2 : ℕ) :
  score1 = 40 →
  score2 = 10 →
  matches1 = 2 →
  matches2 = 3 →
  overall_avg = 22 →
  (score1 * matches1 + score2 * matches2) / (matches1 + matches2) = overall_avg →
  matches1 + matches2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_cricketer_matches_l2835_283523


namespace NUMINAMATH_CALUDE_smallest_number_divisible_by_28_remainder_4_mod_15_l2835_283508

theorem smallest_number_divisible_by_28_remainder_4_mod_15 :
  ∃ n : ℕ, (n % 28 = 0) ∧ (n % 15 = 4) ∧ 
  (∀ m : ℕ, m < n → (m % 28 ≠ 0 ∨ m % 15 ≠ 4)) ∧ 
  n = 364 := by
sorry

end NUMINAMATH_CALUDE_smallest_number_divisible_by_28_remainder_4_mod_15_l2835_283508


namespace NUMINAMATH_CALUDE_oranges_taken_l2835_283595

theorem oranges_taken (initial : ℕ) (remaining : ℕ) (taken : ℕ) 
  (h1 : initial = 60)
  (h2 : remaining = 25)
  (h3 : initial = remaining + taken) : 
  taken = 35 := by
  sorry

end NUMINAMATH_CALUDE_oranges_taken_l2835_283595


namespace NUMINAMATH_CALUDE_triangle_angle_calculation_l2835_283573

theorem triangle_angle_calculation (a b : ℝ) (A B : ℝ) :
  a = Real.sqrt 3 * b →
  A = 2 * π / 3 →
  B = π / 6 :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_calculation_l2835_283573


namespace NUMINAMATH_CALUDE_square_areas_side_lengths_sum_l2835_283500

theorem square_areas_side_lengths_sum (r1 r2 r3 : ℚ) 
  (h_ratio : r1 = 345/45 ∧ r2 = 345/30 ∧ r3 = 345/15) :
  ∃ (a1 b1 c1 a2 b2 c2 a3 b3 c3 : ℕ),
    (r1.sqrt = (a1 : ℚ) * (b1 : ℚ).sqrt / c1) ∧
    (r2.sqrt = (a2 : ℚ) * (b2 : ℚ).sqrt / c2) ∧
    (r3.sqrt = (a3 : ℚ) * (b3 : ℚ).sqrt / c3) ∧
    (a1 + b1 + c1 = 73) ∧
    (a2 + b2 + c2 = 49) ∧
    (a3 + b3 + c3 = 531) ∧
    (max (a1 + b1 + c1) (max (a2 + b2 + c2) (a3 + b3 + c3)) = 531) :=
by sorry

end NUMINAMATH_CALUDE_square_areas_side_lengths_sum_l2835_283500


namespace NUMINAMATH_CALUDE_building_height_proof_l2835_283549

/-- Proves the height of the first 10 stories in a 20-story building -/
theorem building_height_proof (total_stories : Nat) (first_section : Nat) (height_difference : Nat) (total_height : Nat) :
  total_stories = 20 →
  first_section = 10 →
  height_difference = 3 →
  total_height = 270 →
  ∃ (first_story_height : Nat),
    first_story_height * first_section + (first_story_height + height_difference) * (total_stories - first_section) = total_height ∧
    first_story_height = 12 := by
  sorry

end NUMINAMATH_CALUDE_building_height_proof_l2835_283549


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_range_of_a_for_not_p_sufficient_not_necessary_for_q_l2835_283586

-- Define the sets A and B
def A : Set ℝ := {x | 6 + 5*x - x^2 > 0}
def B (a : ℝ) : Set ℝ := {x | (x - (1-a)) * (x - (1+a)) > 0}

-- Define propositions p and q
def p (x : ℝ) : Prop := x ∈ A
def q (a : ℝ) (x : ℝ) : Prop := x ∈ B a

-- Statement 1: A ∩ (ℝ\B) when a = 2
theorem intersection_A_complement_B :
  A ∩ (Set.univ \ B 2) = {x : ℝ | -1 < x ∧ x ≤ 3} :=
sorry

-- Statement 2: Range of a where ¬p is sufficient but not necessary for q
theorem range_of_a_for_not_p_sufficient_not_necessary_for_q :
  {a : ℝ | 0 < a ∧ a < 2} =
  {a : ℝ | ∀ x, ¬(p x) → q a x ∧ ∃ y, q a y ∧ p y} :=
sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_range_of_a_for_not_p_sufficient_not_necessary_for_q_l2835_283586


namespace NUMINAMATH_CALUDE_sofa_bench_arrangement_l2835_283561

/-- The number of ways to arrange n indistinguishable objects of one type
    and k indistinguishable objects of another type in a row -/
def arrangements (n k : ℕ) : ℕ := Nat.choose (n + k) n

/-- Theorem: There are 210 distinct ways to arrange 6 indistinguishable objects
    of one type and 4 indistinguishable objects of another type in a row -/
theorem sofa_bench_arrangement : arrangements 6 4 = 210 := by
  sorry

end NUMINAMATH_CALUDE_sofa_bench_arrangement_l2835_283561


namespace NUMINAMATH_CALUDE_g_value_at_pi_over_4_l2835_283545

open Real

noncomputable def f (x : ℝ) : ℝ := sin x * cos x - Real.sqrt 3 * (sin x)^2

noncomputable def g (x : ℝ) : ℝ := f (x - π / 12) + Real.sqrt 3 / 2

theorem g_value_at_pi_over_4 : g (π / 4) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_g_value_at_pi_over_4_l2835_283545


namespace NUMINAMATH_CALUDE_wire_length_around_square_field_l2835_283542

theorem wire_length_around_square_field (area : ℝ) (rounds : ℕ) 
  (h1 : area = 69696) 
  (h2 : rounds = 15) : 
  Real.sqrt area * 4 * rounds = 15840 := by
  sorry

end NUMINAMATH_CALUDE_wire_length_around_square_field_l2835_283542


namespace NUMINAMATH_CALUDE_inscribed_circles_distance_l2835_283567

-- Define the triangle
def Triangle (X Y Z : ℝ × ℝ) : Prop :=
  (X.1 - Y.1)^2 + (X.2 - Y.2)^2 = 100^2 ∧
  (X.1 - Z.1)^2 + (X.2 - Z.2)^2 = 160^2 ∧
  (Y.1 - Z.1)^2 + (Y.2 - Z.2)^2 = 200^2

-- Define a right angle
def RightAngle (A B C : ℝ × ℝ) : Prop :=
  (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0

-- Define a line perpendicular to another line
def Perpendicular (A B C D : ℝ × ℝ) : Prop :=
  (B.1 - A.1) * (D.1 - C.1) + (B.2 - A.2) * (D.2 - C.2) = 0

-- Define the inscribed circle
def InscribedCircle (C : ℝ × ℝ) (r : ℝ) (A B D : ℝ × ℝ) : Prop :=
  (C.1 - A.1)^2 + (C.2 - A.2)^2 = r^2 ∧
  (C.1 - B.1)^2 + (C.2 - B.2)^2 = r^2 ∧
  (C.1 - D.1)^2 + (C.2 - D.2)^2 = r^2

theorem inscribed_circles_distance 
  (X Y Z M N O P : ℝ × ℝ) 
  (C₁ C₂ C₃ : ℝ × ℝ) 
  (r₁ r₂ r₃ : ℝ) :
  Triangle X Y Z →
  RightAngle X Y Z →
  Perpendicular X Z M N →
  Perpendicular X Y O P →
  InscribedCircle C₁ r₁ X Y Z →
  InscribedCircle C₂ r₂ Z M N →
  InscribedCircle C₃ r₃ Y O P →
  (C₂.1 - C₃.1)^2 + (C₂.2 - C₃.2)^2 = 26000 :=
sorry

end NUMINAMATH_CALUDE_inscribed_circles_distance_l2835_283567


namespace NUMINAMATH_CALUDE_hyperbola_intersection_range_l2835_283525

-- Define the hyperbola C
def hyperbola_C (x y : ℝ) : Prop := x^2 / 3 - y^2 = 1

-- Define the line l
def line_l (k : ℝ) (x y : ℝ) : Prop := y = k * x + Real.sqrt 2

-- Define the condition for intersection points
def intersects_at_two_points (k : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ), x₁ ≠ x₂ ∧ 
    hyperbola_C x₁ y₁ ∧ hyperbola_C x₂ y₂ ∧
    line_l k x₁ y₁ ∧ line_l k x₂ y₂

-- Define the dot product condition
def dot_product_condition (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₁ * x₂ + y₁ * y₂ > 2

-- Main theorem
theorem hyperbola_intersection_range :
  ∀ k : ℝ, 
    (intersects_at_two_points k ∧ 
     ∀ x₁ y₁ x₂ y₂ : ℝ, hyperbola_C x₁ y₁ ∧ hyperbola_C x₂ y₂ ∧ 
                        line_l k x₁ y₁ ∧ line_l k x₂ y₂ → 
                        dot_product_condition x₁ y₁ x₂ y₂) →
    (k > -1 ∧ k < -Real.sqrt 3 / 3) ∨ (k > Real.sqrt 3 / 3 ∧ k < 1) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_intersection_range_l2835_283525


namespace NUMINAMATH_CALUDE_base_b_subtraction_divisibility_other_bases_divisible_l2835_283531

theorem base_b_subtraction_divisibility (b : ℤ) : 
  b = 6 ↔ (b^3 - 3*b^2 + 3*b - 2) % 5 ≠ 0 := by sorry

theorem other_bases_divisible : 
  ∀ b ∈ ({5, 7, 9, 10} : Set ℤ), (b^3 - 3*b^2 + 3*b - 2) % 5 = 0 := by sorry

end NUMINAMATH_CALUDE_base_b_subtraction_divisibility_other_bases_divisible_l2835_283531


namespace NUMINAMATH_CALUDE_smallest_base_perfect_square_l2835_283590

theorem smallest_base_perfect_square : 
  ∃ (b : ℕ), b > 4 ∧ b = 6 ∧ ∃ (n : ℕ), 2 * b + 4 = n^2 ∧
  ∀ (c : ℕ), c > 4 ∧ c < b → ¬∃ (m : ℕ), 2 * c + 4 = m^2 :=
sorry

end NUMINAMATH_CALUDE_smallest_base_perfect_square_l2835_283590


namespace NUMINAMATH_CALUDE_number_2008_row_l2835_283534

theorem number_2008_row : ∃ (n : ℕ), n = 45 ∧ 
  (n - 1)^2 < 2008 ∧ 2008 ≤ n^2 ∧ 
  (∀ (k : ℕ), k < n → k^2 < 2008) :=
by sorry

end NUMINAMATH_CALUDE_number_2008_row_l2835_283534


namespace NUMINAMATH_CALUDE_house_height_calculation_l2835_283519

/-- Given two trees with consistent shadow ratios and a house with a known shadow length,
    prove that the height of the house can be determined. -/
theorem house_height_calculation (tree1_height tree1_shadow tree2_height tree2_shadow house_shadow : ℝ)
    (h1 : tree1_height > 0)
    (h2 : tree2_height > 0)
    (h3 : tree1_shadow > 0)
    (h4 : tree2_shadow > 0)
    (h5 : house_shadow > 0)
    (h6 : tree1_shadow / tree1_height = tree2_shadow / tree2_height) :
    ∃ (house_height : ℝ), house_height = tree1_height * (house_shadow / tree1_shadow) :=
  sorry

end NUMINAMATH_CALUDE_house_height_calculation_l2835_283519


namespace NUMINAMATH_CALUDE_opposite_of_sqrt_three_l2835_283599

theorem opposite_of_sqrt_three : -(Real.sqrt 3) = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_sqrt_three_l2835_283599


namespace NUMINAMATH_CALUDE_sarka_age_l2835_283554

/-- Represents the ages of three sisters and their mother -/
structure FamilyAges where
  sarka : ℕ
  liba : ℕ
  eliska : ℕ
  mother : ℕ

/-- The conditions of the problem -/
def problem_conditions (ages : FamilyAges) : Prop :=
  ages.liba = ages.sarka + 3 ∧
  ages.eliska = ages.sarka + 8 ∧
  ages.mother = ages.sarka + 29 ∧
  (ages.sarka + ages.liba + ages.eliska + ages.mother) / 4 = 21

/-- The theorem stating Šárka's age -/
theorem sarka_age :
  ∃ (ages : FamilyAges), problem_conditions ages ∧ ages.sarka = 11 := by
  sorry

end NUMINAMATH_CALUDE_sarka_age_l2835_283554


namespace NUMINAMATH_CALUDE_arithmetic_mean_relation_l2835_283584

theorem arithmetic_mean_relation (n : ℕ) (d : ℕ) (h1 : d > 0) :
  let seq := List.range d
  let arithmetic_mean := (n * d + (d * (d - 1)) / 2) / d
  let largest := n + d - 1
  arithmetic_mean = 5 * n →
  largest / arithmetic_mean = 9 / 5 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_mean_relation_l2835_283584


namespace NUMINAMATH_CALUDE_same_total_price_implies_sams_price_l2835_283502

/-- The price per sheet charged by Sam's Picture Emporium -/
def sams_price_per_sheet : ℝ := sorry

/-- The price per sheet charged by John's Photo World -/
def johns_price_per_sheet : ℝ := 2.75

/-- The sitting fee charged by John's Photo World -/
def johns_sitting_fee : ℝ := 125

/-- The sitting fee charged by Sam's Picture Emporium -/
def sams_sitting_fee : ℝ := 140

/-- The number of sheets in the package -/
def num_sheets : ℕ := 12

theorem same_total_price_implies_sams_price (h : johns_price_per_sheet * num_sheets + johns_sitting_fee = sams_price_per_sheet * num_sheets + sams_sitting_fee) : 
  sams_price_per_sheet = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_same_total_price_implies_sams_price_l2835_283502


namespace NUMINAMATH_CALUDE_city_connections_l2835_283556

/-- The number of cities in the problem -/
def num_cities : ℕ := 6

/-- The function to calculate the number of unique pairwise connections -/
def unique_connections (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem stating that for 6 cities, the number of unique pairwise connections is 15 -/
theorem city_connections : unique_connections num_cities = 15 := by
  sorry

end NUMINAMATH_CALUDE_city_connections_l2835_283556


namespace NUMINAMATH_CALUDE_percentage_failed_both_l2835_283524

/-- Percentage of students who failed in Hindi -/
def failed_hindi : ℝ := 35

/-- Percentage of students who failed in English -/
def failed_english : ℝ := 45

/-- Percentage of students who passed in both subjects -/
def passed_both : ℝ := 40

/-- Percentage of students who failed in both subjects -/
def failed_both : ℝ := 20

theorem percentage_failed_both :
  failed_both = failed_hindi + failed_english - (100 - passed_both) := by
  sorry

end NUMINAMATH_CALUDE_percentage_failed_both_l2835_283524


namespace NUMINAMATH_CALUDE_r_and_s_earnings_l2835_283559

/-- The daily earnings of individuals p, q, r, and s --/
structure DailyEarnings where
  p : ℚ
  q : ℚ
  r : ℚ
  s : ℚ

/-- The conditions given in the problem --/
def problem_conditions (e : DailyEarnings) : Prop :=
  e.p + e.q + e.r + e.s = 2380 / 9 ∧
  e.p + e.r = 600 / 5 ∧
  e.q + e.s = 800 / 6 ∧
  e.q + e.r = 910 / 7 ∧
  e.p = 150 / 3

/-- The theorem stating that r and s together earn 430/3 Rs per day --/
theorem r_and_s_earnings (e : DailyEarnings) :
  problem_conditions e → e.r + e.s = 430 / 3 := by
  sorry

#check r_and_s_earnings

end NUMINAMATH_CALUDE_r_and_s_earnings_l2835_283559


namespace NUMINAMATH_CALUDE_max_profit_at_60_l2835_283594

/-- The profit function for a travel agency chartering a plane -/
def profit (x : ℕ) : ℝ :=
  if x ≤ 30 then
    900 * x - 15000
  else if x ≤ 75 then
    (-10 * x + 1200) * x - 15000
  else
    0

/-- The maximum number of people allowed in the tour group -/
def max_people : ℕ := 75

/-- The charter fee for the travel agency -/
def charter_fee : ℝ := 15000

theorem max_profit_at_60 :
  ∀ x : ℕ, x ≤ max_people → profit x ≤ profit 60 ∧ profit 60 = 21000 :=
sorry

end NUMINAMATH_CALUDE_max_profit_at_60_l2835_283594


namespace NUMINAMATH_CALUDE_product_remainder_mod_five_l2835_283588

theorem product_remainder_mod_five : (1234 * 5678 * 9012) % 5 = 4 := by
  sorry

end NUMINAMATH_CALUDE_product_remainder_mod_five_l2835_283588


namespace NUMINAMATH_CALUDE_parallelepiped_analogy_l2835_283514

/-- A type representing plane figures -/
inductive PlaneFigure
  | triangle
  | trapezoid
  | parallelogram
  | rectangle

/-- A type representing 3D figures -/
inductive SpaceFigure
  | parallelepiped

/-- A function that determines the most analogous plane figure to a given space figure -/
def mostAnalogousPlaneFigure (sf : SpaceFigure) : PlaneFigure :=
  match sf with
  | SpaceFigure.parallelepiped => PlaneFigure.parallelogram

/-- Theorem stating that a parallelogram is the most analogous plane figure to a parallelepiped -/
theorem parallelepiped_analogy :
  mostAnalogousPlaneFigure SpaceFigure.parallelepiped = PlaneFigure.parallelogram :=
by sorry

end NUMINAMATH_CALUDE_parallelepiped_analogy_l2835_283514


namespace NUMINAMATH_CALUDE_depression_comparison_l2835_283589

-- Define the prevalence of depression for women and men
def depression_prevalence_women : ℝ := 2
def depression_prevalence_men : ℝ := 1

-- Define the correct comparative phrase
def correct_phrase : String := "twice as...as"

-- Theorem to prove
theorem depression_comparison (w m : ℝ) (phrase : String) :
  w = 2 * m → phrase = correct_phrase → 
  (w = depression_prevalence_women ∧ m = depression_prevalence_men) :=
by sorry

end NUMINAMATH_CALUDE_depression_comparison_l2835_283589
