import Mathlib

namespace NUMINAMATH_CALUDE_expression_evaluation_l202_20251

theorem expression_evaluation (x : ℝ) (h : x^2 - 3*x - 2 = 0) :
  (x + 1) * (x - 1) - (x + 3)^2 + 2*x^2 = -6 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l202_20251


namespace NUMINAMATH_CALUDE_total_jellybeans_proof_l202_20294

/-- The number of jellybeans needed to fill a large drinking glass -/
def large_glass_beans : ℕ := 50

/-- The number of large drinking glasses -/
def num_large_glasses : ℕ := 5

/-- The number of small drinking glasses -/
def num_small_glasses : ℕ := 3

/-- The number of jellybeans needed to fill a small drinking glass -/
def small_glass_beans : ℕ := large_glass_beans / 2

/-- The total number of jellybeans needed to fill all glasses -/
def total_beans : ℕ := large_glass_beans * num_large_glasses + small_glass_beans * num_small_glasses

theorem total_jellybeans_proof : total_beans = 325 := by
  sorry

end NUMINAMATH_CALUDE_total_jellybeans_proof_l202_20294


namespace NUMINAMATH_CALUDE_later_arrival_l202_20232

/-- A man's journey to his office -/
structure JourneyToOffice where
  usual_rate : ℝ
  usual_time : ℝ
  slower_rate : ℝ
  slower_time : ℝ

/-- The conditions of the problem -/
def journey_conditions (j : JourneyToOffice) : Prop :=
  j.usual_time = 1 ∧ j.slower_rate = 3/4 * j.usual_rate

/-- The theorem to be proved -/
theorem later_arrival (j : JourneyToOffice) 
  (h : journey_conditions j) : 
  j.slower_time - j.usual_time = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_later_arrival_l202_20232


namespace NUMINAMATH_CALUDE_anns_shopping_cost_l202_20276

theorem anns_shopping_cost (shorts_count : ℕ) (shorts_price : ℚ)
                            (shoes_count : ℕ) (shoes_price : ℚ)
                            (tops_count : ℕ) (total_cost : ℚ) :
  shorts_count = 5 →
  shorts_price = 7 →
  shoes_count = 2 →
  shoes_price = 10 →
  tops_count = 4 →
  total_cost = 75 →
  ∃ (top_price : ℚ), top_price = 5 ∧
    total_cost = shorts_count * shorts_price + shoes_count * shoes_price + tops_count * top_price :=
by
  sorry


end NUMINAMATH_CALUDE_anns_shopping_cost_l202_20276


namespace NUMINAMATH_CALUDE_equation_solution_l202_20214

theorem equation_solution (x : ℝ) : x^2 + x = 5 + Real.sqrt 5 ↔ x = Real.sqrt 5 ∨ x = -Real.sqrt 5 - 1 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l202_20214


namespace NUMINAMATH_CALUDE_proportion_equality_l202_20216

theorem proportion_equality (a b c x : ℝ) (h : a / x = 4 * a * b / (17.5 * c)) :
  x = 17.5 * c / (4 * b) := by
sorry

end NUMINAMATH_CALUDE_proportion_equality_l202_20216


namespace NUMINAMATH_CALUDE_arithmetic_equation_l202_20252

theorem arithmetic_equation : 50 + 5 * 12 / (180 / 3) = 51 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_equation_l202_20252


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l202_20275

theorem sum_of_coefficients (x : ℝ) : 
  let p (x : ℝ) := 3*(x^8 - 2*x^5 + x^3 - 6) - 5*(2*x^4 + 3*x^2) + 2*(x^6 - 5)
  p 1 = -51 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l202_20275


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l202_20263

theorem complex_fraction_equality : (Complex.I : ℂ) ^ 2 = -1 → (2 + 2 * Complex.I) / (1 - Complex.I) = 2 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l202_20263


namespace NUMINAMATH_CALUDE_school_population_l202_20229

/-- Given a school population where:
  * The number of boys is 4 times the number of girls
  * The number of girls is 8 times the number of teachers
This theorem proves that the total number of boys, girls, and teachers
is equal to 41/32 times the number of boys. -/
theorem school_population (b g t : ℕ) (h1 : b = 4 * g) (h2 : g = 8 * t) :
  b + g + t = (41 * b) / 32 := by
  sorry

end NUMINAMATH_CALUDE_school_population_l202_20229


namespace NUMINAMATH_CALUDE_reciprocal_sum_one_l202_20227

theorem reciprocal_sum_one (x y z : ℕ+) (h_sum : (1 : ℚ) / x + (1 : ℚ) / y + (1 : ℚ) / z = 1) 
  (h_order : x ≤ y ∧ y ≤ z) : 
  (x = 2 ∧ y = 4 ∧ z = 4) ∨ (x = 2 ∧ y = 3 ∧ z = 6) ∨ (x = 3 ∧ y = 3 ∧ z = 3) := by
  sorry

#check reciprocal_sum_one

end NUMINAMATH_CALUDE_reciprocal_sum_one_l202_20227


namespace NUMINAMATH_CALUDE_smaller_root_of_equation_l202_20257

theorem smaller_root_of_equation (x : ℝ) : 
  (x - 5/8) * (x - 5/8) + (x - 5/8) * (x - 2/3) = 0 → 
  (∃ y : ℝ, (y - 5/8) * (y - 5/8) + (y - 5/8) * (y - 2/3) = 0 ∧ y ≤ x) → 
  x = 29/48 := by
sorry

end NUMINAMATH_CALUDE_smaller_root_of_equation_l202_20257


namespace NUMINAMATH_CALUDE_ones_digit_of_nine_to_46_l202_20202

theorem ones_digit_of_nine_to_46 : (9^46 : ℕ) % 10 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ones_digit_of_nine_to_46_l202_20202


namespace NUMINAMATH_CALUDE_max_value_of_a_l202_20235

-- Define the operation
def determinant (a b c d : ℝ) : ℝ := a * d - b * c

-- State the theorem
theorem max_value_of_a :
  (∀ x : ℝ, determinant (x - 1) (a - 2) (a + 1) x ≥ 1) →
  (∃ a_max : ℝ, a ≤ a_max ∧ a_max = 3/2) :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_a_l202_20235


namespace NUMINAMATH_CALUDE_max_value_theorem_l202_20206

theorem max_value_theorem (k : ℝ) (hk : k > 0) :
  (3 * k^3 + 3 * k) / ((3/2 * k^2 + 14) * (14 * k^2 + 3/2)) ≤ Real.sqrt 21 / 175 := by
  sorry

end NUMINAMATH_CALUDE_max_value_theorem_l202_20206


namespace NUMINAMATH_CALUDE_simplify_expression_l202_20243

theorem simplify_expression (x : ℝ) (h : x ≠ -1) :
  (x + 1)⁻¹ - 2 = (-2*x - 1) / (x + 1) := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l202_20243


namespace NUMINAMATH_CALUDE_worker_earnings_l202_20210

theorem worker_earnings
  (regular_rate : ℝ)
  (total_surveys : ℕ)
  (cellphone_rate_increase : ℝ)
  (cellphone_surveys : ℕ)
  (h1 : regular_rate = 30)
  (h2 : total_surveys = 100)
  (h3 : cellphone_rate_increase = 0.2)
  (h4 : cellphone_surveys = 50) :
  let cellphone_rate := regular_rate * (1 + cellphone_rate_increase)
  let regular_surveys := total_surveys - cellphone_surveys
  let total_earnings := regular_rate * regular_surveys + cellphone_rate * cellphone_surveys
  total_earnings = 3300 :=
by sorry

end NUMINAMATH_CALUDE_worker_earnings_l202_20210


namespace NUMINAMATH_CALUDE_parabola_sum_l202_20291

theorem parabola_sum (a b c : ℝ) : 
  (∀ y : ℝ, 10 = a * (-6)^2 + b * (-6) + c) →
  (∀ y : ℝ, 8 = a * (-4)^2 + b * (-4) + c) →
  a + b + c = -39 := by
sorry

end NUMINAMATH_CALUDE_parabola_sum_l202_20291


namespace NUMINAMATH_CALUDE_ab_inequality_and_minimum_l202_20241

theorem ab_inequality_and_minimum (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) (hab : a * b = a + b + 8) : 
  (a * b ≥ 16) ∧ 
  (a + 4 * b ≥ 17) ∧ 
  (a + 4 * b = 17 → a = 7) := by
sorry

end NUMINAMATH_CALUDE_ab_inequality_and_minimum_l202_20241


namespace NUMINAMATH_CALUDE_sandal_price_proof_l202_20283

/-- Proves that the price of each pair of sandals is $3 given the conditions of Yanna's purchase. -/
theorem sandal_price_proof (num_shirts : ℕ) (shirt_price : ℕ) (num_sandals : ℕ) (bill_paid : ℕ) (change_received : ℕ) :
  num_shirts = 10 →
  shirt_price = 5 →
  num_sandals = 3 →
  bill_paid = 100 →
  change_received = 41 →
  (bill_paid - change_received - num_shirts * shirt_price) / num_sandals = 3 :=
by sorry

end NUMINAMATH_CALUDE_sandal_price_proof_l202_20283


namespace NUMINAMATH_CALUDE_calculate_marked_price_jobber_marked_price_l202_20253

/-- Calculate the marked price of an article given the original price, purchase discount,
    desired profit margin, and selling discount. -/
theorem calculate_marked_price (original_price : ℝ) (purchase_discount : ℝ) 
    (profit_margin : ℝ) (selling_discount : ℝ) : ℝ :=
  let purchase_price := original_price * (1 - purchase_discount)
  let desired_selling_price := purchase_price * (1 + profit_margin)
  desired_selling_price / (1 - selling_discount)

/-- The marked price of the article should be $50.00 -/
theorem jobber_marked_price : 
  calculate_marked_price 40 0.25 0.5 0.1 = 50 := by
  sorry

end NUMINAMATH_CALUDE_calculate_marked_price_jobber_marked_price_l202_20253


namespace NUMINAMATH_CALUDE_average_first_30_multiples_of_29_l202_20223

theorem average_first_30_multiples_of_29 : 
  let n : ℕ := 30
  let base : ℕ := 29
  let sum : ℕ := n * (base + n * base) / 2
  (sum : ℚ) / n = 449.5 := by sorry

end NUMINAMATH_CALUDE_average_first_30_multiples_of_29_l202_20223


namespace NUMINAMATH_CALUDE_profit_maximized_at_95_l202_20248

/-- Represents the profit function for a product sale scenario -/
def profit_function (x : ℝ) : ℝ := -20 * x^2 + 200 * x + 4000

/-- Theorem stating that the profit is maximized at a selling price of 95 yuan -/
theorem profit_maximized_at_95 :
  let initial_purchase_price : ℝ := 80
  let initial_selling_price : ℝ := 90
  let initial_sales_volume : ℝ := 400
  let price_increase_rate : ℝ := 1
  let sales_decrease_rate : ℝ := 20
  ∃ (max_profit : ℝ), 
    (∀ x, profit_function x ≤ max_profit) ∧ 
    (profit_function 5 = max_profit) ∧
    (initial_selling_price + 5 = 95) := by
  sorry

#check profit_maximized_at_95

end NUMINAMATH_CALUDE_profit_maximized_at_95_l202_20248


namespace NUMINAMATH_CALUDE_average_age_decrease_l202_20296

def original_strength : ℕ := 8
def original_average_age : ℕ := 40
def new_students : ℕ := 8
def new_students_average_age : ℕ := 32

theorem average_age_decrease :
  let original_total_age := original_strength * original_average_age
  let new_total_age := original_total_age + new_students * new_students_average_age
  let new_total_strength := original_strength + new_students
  let new_average_age := new_total_age / new_total_strength
  original_average_age - new_average_age = 4 := by
sorry

end NUMINAMATH_CALUDE_average_age_decrease_l202_20296


namespace NUMINAMATH_CALUDE_smallest_s_for_F_l202_20289

def F (a b c d : ℕ) : ℕ := a * b^(c^d)

theorem smallest_s_for_F : 
  (∀ s : ℕ, s > 0 ∧ s < 9 → F s s 2 2 < 65536) ∧ 
  F 9 9 2 2 = 65536 := by
  sorry

end NUMINAMATH_CALUDE_smallest_s_for_F_l202_20289


namespace NUMINAMATH_CALUDE_number_equation_solution_l202_20284

theorem number_equation_solution : 
  ∃ x : ℝ, (5020 - (x / 20.08) = 4970) ∧ (x = 1004) := by sorry

end NUMINAMATH_CALUDE_number_equation_solution_l202_20284


namespace NUMINAMATH_CALUDE_min_value_theorem_l202_20268

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_geom_mean : 2 = Real.sqrt (4^a * 2^b)) :
  ∀ x y : ℝ, x > 0 → y > 0 → 2 = Real.sqrt (4^x * 2^y) → 
    (2/a + 1/b) ≤ (2/x + 1/y) ∧ 
    (∃ a₀ b₀ : ℝ, a₀ > 0 ∧ b₀ > 0 ∧ 2 = Real.sqrt (4^a₀ * 2^b₀) ∧ 2/a₀ + 1/b₀ = 9/2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l202_20268


namespace NUMINAMATH_CALUDE_tourist_distribution_l202_20221

theorem tourist_distribution (n m : ℕ) (hn : n = 8) (hm : m = 3) :
  (m ^ n : ℕ) - m * ((m - 1) ^ n) + (m.choose 2) * (1 ^ n) = 5796 :=
sorry

end NUMINAMATH_CALUDE_tourist_distribution_l202_20221


namespace NUMINAMATH_CALUDE_system_solution_l202_20287

theorem system_solution :
  ∀ (x y z : ℝ),
    ((x = 38 ∧ y = 4 ∧ z = 9) ∨ (x = 110 ∧ y = 2 ∧ z = 33)) →
    (x * y - 2 * y = x + 106) ∧
    (y * z + 3 * y = z + 39) ∧
    (z * x + 3 * x = 2 * z + 438) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l202_20287


namespace NUMINAMATH_CALUDE_smallest_four_digit_solution_l202_20201

def is_valid (x : ℕ) : Prop :=
  (3 * x) % 12 = 6 ∧
  (5 * x + 20) % 15 = 25 ∧
  (3 * x - 2) % 35 = (2 * x) % 35

def is_four_digit (x : ℕ) : Prop :=
  1000 ≤ x ∧ x ≤ 9999

theorem smallest_four_digit_solution :
  is_valid 1274 ∧ is_four_digit 1274 ∧
  ∀ y : ℕ, (is_valid y ∧ is_four_digit y) → 1274 ≤ y :=
by sorry

end NUMINAMATH_CALUDE_smallest_four_digit_solution_l202_20201


namespace NUMINAMATH_CALUDE_probability_specific_marble_draw_l202_20228

/-- Represents the number of marbles of each color in the jar -/
structure MarbleCount where
  red : ℕ
  green : ℕ
  white : ℕ

/-- Calculates the probability of drawing two red marbles followed by one green marble -/
def probability_two_red_one_green (mc : MarbleCount) : ℚ :=
  let total := mc.red + mc.green + mc.white
  (mc.red : ℚ) / total *
  ((mc.red - 1) : ℚ) / (total - 1) *
  (mc.green : ℚ) / (total - 2)

/-- The main theorem stating the probability for the given marble counts -/
theorem probability_specific_marble_draw :
  probability_two_red_one_green ⟨3, 4, 12⟩ = 12 / 2907 := by
  sorry

#eval probability_two_red_one_green ⟨3, 4, 12⟩

end NUMINAMATH_CALUDE_probability_specific_marble_draw_l202_20228


namespace NUMINAMATH_CALUDE_jonas_bookshelves_l202_20266

/-- Calculates the maximum number of bookshelves that can fit in a room -/
def max_bookshelves (total_space desk_space shelf_space : ℕ) : ℕ :=
  (total_space - desk_space) / shelf_space

/-- Proves that the maximum number of bookshelves in Jonas' room is 3 -/
theorem jonas_bookshelves :
  max_bookshelves 400 160 80 = 3 := by
  sorry

#eval max_bookshelves 400 160 80

end NUMINAMATH_CALUDE_jonas_bookshelves_l202_20266


namespace NUMINAMATH_CALUDE_sum_of_complex_equality_l202_20290

theorem sum_of_complex_equality (x y : ℝ) :
  (x - 2 : ℂ) + y * Complex.I = -1 + Complex.I →
  x + y = 2 := by sorry

end NUMINAMATH_CALUDE_sum_of_complex_equality_l202_20290


namespace NUMINAMATH_CALUDE_fifth_square_area_l202_20271

theorem fifth_square_area (s : ℝ) (h : s + 5 = 11) : s^2 = 36 := by
  sorry

end NUMINAMATH_CALUDE_fifth_square_area_l202_20271


namespace NUMINAMATH_CALUDE_monotone_increasing_condition_l202_20255

open Real

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x * log x - x^2

-- State the theorem
theorem monotone_increasing_condition (a : ℝ) :
  (∀ x ∈ Set.Icc 1 (exp 1), MonotoneOn (f a) (Set.Icc 1 (exp 1))) ↔ a ≥ exp 1 :=
by sorry

end NUMINAMATH_CALUDE_monotone_increasing_condition_l202_20255


namespace NUMINAMATH_CALUDE_imaginary_part_of_product_l202_20217

theorem imaginary_part_of_product (i : ℂ) : 
  i * i = -1 →
  Complex.im ((1 + 2*i) * (2 - i)) = 3 := by
sorry

end NUMINAMATH_CALUDE_imaginary_part_of_product_l202_20217


namespace NUMINAMATH_CALUDE_cheetah_speed_calculation_l202_20260

/-- The top speed of a cheetah in miles per hour -/
def cheetah_speed : ℝ := 60

/-- The top speed of a gazelle in miles per hour -/
def gazelle_speed : ℝ := 40

/-- Conversion factor from miles per hour to feet per second -/
def mph_to_fps : ℝ := 1.5

/-- Time taken for the cheetah to catch up to the gazelle in seconds -/
def catch_up_time : ℝ := 7

/-- Initial distance between the cheetah and the gazelle in feet -/
def initial_distance : ℝ := 210

theorem cheetah_speed_calculation :
  cheetah_speed * mph_to_fps - gazelle_speed * mph_to_fps = initial_distance / catch_up_time :=
by sorry

end NUMINAMATH_CALUDE_cheetah_speed_calculation_l202_20260


namespace NUMINAMATH_CALUDE_inequalities_hold_l202_20299

theorem inequalities_hold (a b c : ℝ) (h1 : c < b) (h2 : b < a) (h3 : a * c < 0) :
  (b / a > c / a) ∧ ((b - a) / c > 0) ∧ ((a - c) / (a * c) < 0) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_hold_l202_20299


namespace NUMINAMATH_CALUDE_pizza_toppings_l202_20282

theorem pizza_toppings (total_slices : ℕ) (pepperoni_slices : ℕ) (mushroom_slices : ℕ) 
  (h1 : total_slices = 20)
  (h2 : pepperoni_slices = 12)
  (h3 : mushroom_slices = 14)
  (h4 : ∀ slice, slice ≤ total_slices → (slice ≤ pepperoni_slices ∨ slice ≤ mushroom_slices)) :
  ∃ both_toppings : ℕ, 
    both_toppings = pepperoni_slices + mushroom_slices - total_slices ∧
    both_toppings = 6 :=
by sorry

end NUMINAMATH_CALUDE_pizza_toppings_l202_20282


namespace NUMINAMATH_CALUDE_arithmetic_square_root_of_nine_l202_20245

theorem arithmetic_square_root_of_nine (x : ℝ) :
  (x ≥ 0 ∧ x ^ 2 = 9) → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_square_root_of_nine_l202_20245


namespace NUMINAMATH_CALUDE_set_union_complement_and_subset_necessary_not_sufficient_condition_l202_20254

def A : Set ℝ := {x | x^2 + 2*x - 3 < 0}

def B (a : ℝ) : Set ℝ := {x | -a - 1 < x ∧ x < -a + 1}

theorem set_union_complement_and_subset (a : ℝ) :
  a = 3 → (Set.univ \ A) ∪ B a = {x | x < -2 ∨ x ≥ 1} :=
sorry

theorem necessary_not_sufficient_condition (a : ℝ) :
  (∀ x, x ∈ B a → x ∈ A) ∧ (∃ x, x ∈ A ∧ x ∉ B a) ↔ 0 ≤ a ∧ a ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_set_union_complement_and_subset_necessary_not_sufficient_condition_l202_20254


namespace NUMINAMATH_CALUDE_equation_solution_l202_20220

theorem equation_solution (a : ℝ) : 
  (∀ x, 2*(x+1) = 3*(x-1) ↔ x = a+2) →
  (∃! x, 2*(2*(x+3) - 3*(x-a)) = 3*a ∧ x = 10) := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l202_20220


namespace NUMINAMATH_CALUDE_root_location_l202_20288

/-- Given a function f(x) = a^x + x - b with a root x₀ ∈ (n, n+1), where n is an integer,
    and constants a and b satisfying 2^a = 3 and 3^b = 2, prove that n = -1. -/
theorem root_location (a b : ℝ) (n : ℤ) (x₀ : ℝ) :
  (2 : ℝ) ^ a = 3 →
  (3 : ℝ) ^ b = 2 →
  (∃ x₀, x₀ ∈ Set.Ioo n (n + 1) ∧ a ^ x₀ + x₀ - b = 0) →
  n = -1 := by
  sorry

end NUMINAMATH_CALUDE_root_location_l202_20288


namespace NUMINAMATH_CALUDE_simplify_expression_l202_20265

theorem simplify_expression (x : ℝ) : 3*x + 6*x + 9*x + 12*x + 15*x + 18 = 45*x + 18 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l202_20265


namespace NUMINAMATH_CALUDE_school_election_votes_l202_20267

theorem school_election_votes (total_votes : ℕ) (brenda_votes : ℕ) : 
  brenda_votes = 50 → 
  4 * brenda_votes = total_votes →
  total_votes = 200 := by
sorry

end NUMINAMATH_CALUDE_school_election_votes_l202_20267


namespace NUMINAMATH_CALUDE_geometric_series_common_ratio_l202_20213

theorem geometric_series_common_ratio : 
  let a₁ : ℚ := 5/6
  let a₂ : ℚ := -4/9
  let a₃ : ℚ := 32/135
  let r : ℚ := a₂ / a₁
  (∀ n : ℕ, n ≥ 2 → a₂ = a₁ * r ∧ a₃ = a₂ * r) →
  r = -8/15 := by
sorry

end NUMINAMATH_CALUDE_geometric_series_common_ratio_l202_20213


namespace NUMINAMATH_CALUDE_round_trip_percentage_l202_20239

/-- Represents the distribution of passenger types and classes on a transatlantic ship crossing -/
structure PassengerDistribution where
  /-- Percentage of Type A passengers (round-trip with car) -/
  type_a_percent : ℝ
  /-- Percentage of round-trip passengers not taking cars -/
  no_car_percent : ℝ
  /-- Percentage of round-trip passengers in luxury class -/
  luxury_percent : ℝ
  /-- Percentage of round-trip passengers in economy class -/
  economy_percent : ℝ
  /-- Percentage of Type C passengers in economy class -/
  type_c_economy_percent : ℝ

/-- Theorem stating that given the passenger distribution, the percentage of round-trip passengers is 40% -/
theorem round_trip_percentage (pd : PassengerDistribution)
  (h1 : pd.type_a_percent = 0.2)
  (h2 : pd.no_car_percent = 0.5)
  (h3 : pd.luxury_percent = 0.3)
  (h4 : pd.economy_percent = 0.7)
  (h5 : pd.type_c_economy_percent = 0.4)
  : ℝ :=
  by sorry

end NUMINAMATH_CALUDE_round_trip_percentage_l202_20239


namespace NUMINAMATH_CALUDE_david_money_left_l202_20205

/-- Calculates the amount of money David has left after his trip -/
def money_left (initial_amount : ℕ) (difference : ℕ) : ℕ :=
  initial_amount - (initial_amount - difference) / 2

theorem david_money_left :
  money_left 1800 800 = 500 := by
  sorry

end NUMINAMATH_CALUDE_david_money_left_l202_20205


namespace NUMINAMATH_CALUDE_shaded_area_square_configuration_l202_20258

/-- The area of the shaded region in a geometric configuration where a 4-inch square adjoins a 12-inch square -/
theorem shaded_area_square_configuration : 
  -- Large square side length
  ∀ (large_side : ℝ) 
  -- Small square side length
  (small_side : ℝ),
  -- Conditions
  large_side = 12 →
  small_side = 4 →
  -- The shaded area is the difference between the small square's area and the area of a triangle
  let shaded_area := small_side^2 - (1/2 * (3/4 * small_side) * small_side)
  -- Theorem statement
  shaded_area = 10 := by
sorry

end NUMINAMATH_CALUDE_shaded_area_square_configuration_l202_20258


namespace NUMINAMATH_CALUDE_expansion_properties_l202_20292

theorem expansion_properties (x : ℝ) (x_ne_zero : x ≠ 0) : 
  let expansion := (1/x - x)^6
  ∃ (coeffs : List ℤ), 
    -- The expansion can be represented as a list of integer coefficients
    (∀ i, 0 ≤ i ∧ i < 7 → coeffs.get! i = (Nat.choose 6 i) * (-1)^i) ∧
    -- The binomial coefficient of the 4th term is the largest
    (∀ i, 0 ≤ i ∧ i < 7 → coeffs.get! 3 ≥ coeffs.get! i) ∧
    -- The sum of all coefficients is 0
    (coeffs.sum = 0) := by
  sorry

end NUMINAMATH_CALUDE_expansion_properties_l202_20292


namespace NUMINAMATH_CALUDE_ball_probabilities_l202_20295

/-- Represents the number of balls in the bag -/
def total_balls : ℕ := 6

/-- Represents the number of red balls in the bag -/
def red_balls : ℕ := 2

/-- Represents the number of white balls in the bag -/
def white_balls : ℕ := 4

/-- Represents the number of balls drawn -/
def drawn_balls : ℕ := 2

/-- Calculates the probability of drawing two red balls -/
def prob_two_red : ℚ := (red_balls.choose drawn_balls : ℚ) / (total_balls.choose drawn_balls : ℚ)

/-- Calculates the probability of drawing at least one red ball -/
def prob_at_least_one_red : ℚ := 1 - (white_balls.choose drawn_balls : ℚ) / (total_balls.choose drawn_balls : ℚ)

theorem ball_probabilities :
  prob_two_red = 1/15 ∧ prob_at_least_one_red = 3/5 := by sorry

end NUMINAMATH_CALUDE_ball_probabilities_l202_20295


namespace NUMINAMATH_CALUDE_sphere_volume_from_surface_area_l202_20269

theorem sphere_volume_from_surface_area :
  ∀ (R : ℝ), 
  R > 0 →
  4 * π * R^2 = 24 * π →
  (4 / 3) * π * R^3 = 8 * Real.sqrt 6 * π :=
by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_from_surface_area_l202_20269


namespace NUMINAMATH_CALUDE_fountain_distance_is_30_l202_20209

/-- The distance from Mrs. Hilt's desk to the water fountain -/
def fountain_distance (total_distance : ℕ) (num_trips : ℕ) : ℕ :=
  total_distance / num_trips

/-- Theorem stating that the distance to the water fountain is 30 feet -/
theorem fountain_distance_is_30 :
  fountain_distance 120 4 = 30 := by
  sorry

end NUMINAMATH_CALUDE_fountain_distance_is_30_l202_20209


namespace NUMINAMATH_CALUDE_batsman_average_after_12_innings_l202_20281

/-- Represents a batsman's performance over multiple innings -/
structure Batsman where
  innings : Nat
  totalRuns : Nat
  averageIncrease : Nat
  lastInningsScore : Nat

/-- Calculates the average score of a batsman after a given number of innings -/
def averageScore (b : Batsman) : Rat :=
  b.totalRuns / b.innings

/-- Theorem: Given the conditions, prove that the batsman's average after 12 innings is 47 -/
theorem batsman_average_after_12_innings (b : Batsman) 
  (h1 : b.innings = 12)
  (h2 : b.lastInningsScore = 80)
  (h3 : b.averageIncrease = 3)
  : averageScore b = 47 := by
  sorry

end NUMINAMATH_CALUDE_batsman_average_after_12_innings_l202_20281


namespace NUMINAMATH_CALUDE_patio_layout_change_l202_20204

theorem patio_layout_change (initial_tiles initial_rows initial_columns : ℕ) 
  (new_rows : ℕ) (h1 : initial_tiles = 160) (h2 : initial_rows = 10) 
  (h3 : initial_columns * initial_rows = initial_tiles)
  (h4 : new_rows = initial_rows + 4) :
  ∃ (new_columns : ℕ), 
    new_columns * new_rows = initial_tiles ∧ 
    initial_columns - new_columns = 5 := by
  sorry

end NUMINAMATH_CALUDE_patio_layout_change_l202_20204


namespace NUMINAMATH_CALUDE_product_of_squares_and_fourth_powers_l202_20208

theorem product_of_squares_and_fourth_powers (r s : ℝ) 
  (h_positive_r : r > 0) (h_positive_s : s > 0)
  (h_sum_squares : r^2 + s^2 = 1) 
  (h_sum_fourth_powers : r^4 + s^4 = 7/8) : r * s = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_product_of_squares_and_fourth_powers_l202_20208


namespace NUMINAMATH_CALUDE_triangle_triplets_characterization_l202_20212

def is_valid_triplet (a b c : ℕ) : Prop :=
  a ≤ b ∧ b ≤ c ∧
  (∃ r : ℚ, r > 0 ∧ b = a * r ∧ c = b * r) ∧
  (a = 100 ∨ c = 100)

def valid_triplets : Set (ℕ × ℕ × ℕ) :=
  {(49,70,100), (64,80,100), (81,90,100), (100,100,100), (100,110,121),
   (100,120,144), (100,130,169), (100,140,196), (100,150,225), (100,160,256)}

theorem triangle_triplets_characterization :
  {(a, b, c) | is_valid_triplet a b c} = valid_triplets :=
by sorry

end NUMINAMATH_CALUDE_triangle_triplets_characterization_l202_20212


namespace NUMINAMATH_CALUDE_equation_solution_l202_20237

theorem equation_solution : ∃ (a b c d : ℕ), 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  0 < a ∧ a < 10 ∧ 0 < b ∧ b < 10 ∧ 0 < c ∧ c < 10 ∧ 0 < d ∧ d < 10 ∧
  a + b = c * d ∧ c * d = 12 ∧ 12 = (10 * c + d) / b :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l202_20237


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_arithmetic_sequence_l202_20200

/-- 
Given an ellipse with major axis length 2a, minor axis length 2b, and focal length 2c,
where these lengths form an arithmetic sequence, prove that the eccentricity is 3/5.
-/
theorem ellipse_eccentricity_arithmetic_sequence 
  (a b c : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_arithmetic : 2 * b = a + c)
  (h_ellipse : b^2 = a^2 - c^2)
  (e : ℝ) 
  (h_eccentricity : e = c / a) :
  e = 3/5 := by
sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_arithmetic_sequence_l202_20200


namespace NUMINAMATH_CALUDE_product_of_fractions_l202_20226

theorem product_of_fractions : 
  (4 : ℚ) / 5 * 9 / 6 * 12 / 4 * 20 / 15 * 14 / 21 * 35 / 28 * 48 / 32 * 24 / 16 = 54 := by
  sorry

end NUMINAMATH_CALUDE_product_of_fractions_l202_20226


namespace NUMINAMATH_CALUDE_math_course_scheduling_l202_20279

theorem math_course_scheduling (n : ℕ) (k : ℕ) (courses : ℕ) : 
  n = 6 → k = 3 → courses = 3 →
  (Nat.choose (n - k + 1) k) * (Nat.factorial courses) = 24 := by
sorry

end NUMINAMATH_CALUDE_math_course_scheduling_l202_20279


namespace NUMINAMATH_CALUDE_star_property_l202_20298

/-- Operation ⋆ for positive real numbers -/
noncomputable def star (k : ℝ) (x y : ℝ) : ℝ := x^y * k

/-- Theorem stating the properties of the ⋆ operation and the result to be proved -/
theorem star_property (k : ℝ) :
  (k > 0) →
  (∀ x y, x > 0 → y > 0 → (star k (x^y) y = x * star k y y)) →
  (∀ x, x > 0 → star k (star k x 1) x = star k x 1) →
  (star k 1 1 = k) →
  (star k 2 3 = 8 * k) := by
  sorry

end NUMINAMATH_CALUDE_star_property_l202_20298


namespace NUMINAMATH_CALUDE_hockey_league_games_l202_20270

/-- The number of games played in a hockey league season -/
def number_of_games (n : ℕ) (m : ℕ) : ℕ :=
  n * (n - 1) / 2 * m

theorem hockey_league_games :
  number_of_games 17 10 = 1360 := by
  sorry

end NUMINAMATH_CALUDE_hockey_league_games_l202_20270


namespace NUMINAMATH_CALUDE_cubic_expression_value_l202_20207

theorem cubic_expression_value (p q : ℝ) : 
  3 * p^2 - 7 * p - 6 = 0 →
  3 * q^2 - 7 * q - 6 = 0 →
  p ≠ q →
  (5 * p^3 - 5 * q^3) * (p - q)⁻¹ = 335 / 9 := by
  sorry

end NUMINAMATH_CALUDE_cubic_expression_value_l202_20207


namespace NUMINAMATH_CALUDE_bus_stop_distance_unit_l202_20286

/-- Represents units of length measurement -/
inductive LengthUnit
  | Millimeter
  | Decimeter
  | Meter
  | Kilometer

/-- The distance between two bus stops in an arbitrary unit -/
def bus_stop_distance : ℕ := 3000

/-- Predicate to determine if a unit is appropriate for measuring bus stop distances -/
def is_appropriate_unit (unit : LengthUnit) : Prop :=
  match unit with
  | LengthUnit.Meter => True
  | _ => False

theorem bus_stop_distance_unit :
  is_appropriate_unit LengthUnit.Meter :=
sorry

end NUMINAMATH_CALUDE_bus_stop_distance_unit_l202_20286


namespace NUMINAMATH_CALUDE_male_puppies_count_l202_20224

/-- Proves that the number of male puppies is 10 given the specified conditions -/
theorem male_puppies_count (total_puppies : ℕ) (female_puppies : ℕ) (ratio : ℚ) :
  total_puppies = 12 →
  female_puppies = 2 →
  ratio = 1/5 →
  total_puppies = female_puppies + (female_puppies / ratio) :=
by
  sorry

end NUMINAMATH_CALUDE_male_puppies_count_l202_20224


namespace NUMINAMATH_CALUDE_paving_rate_calculation_l202_20230

/-- Given a rectangular room with specified dimensions and total paving cost,
    calculate the rate per square meter for paving the floor. -/
theorem paving_rate_calculation (length width total_cost : ℝ) 
    (h1 : length = 9)
    (h2 : width = 4.75)
    (h3 : total_cost = 38475) : 
  total_cost / (length * width) = 900 := by
  sorry

end NUMINAMATH_CALUDE_paving_rate_calculation_l202_20230


namespace NUMINAMATH_CALUDE_fraction_irreducible_l202_20297

theorem fraction_irreducible (n : ℤ) : Int.gcd (39 * n + 4) (26 * n + 3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_irreducible_l202_20297


namespace NUMINAMATH_CALUDE_equation_solution_l202_20218

theorem equation_solution (x : ℝ) : 3 / (x + 10) = 1 / (2 * x) → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l202_20218


namespace NUMINAMATH_CALUDE_solution_set_f_solution_set_g_l202_20238

-- Define the quadratic functions
def f (x : ℝ) : ℝ := x^2 - 3*x - 4
def g (x : ℝ) : ℝ := x^2 - x - 6

-- Define the solution sets
def S₁ : Set ℝ := {x | -1 < x ∧ x < 4}
def S₂ : Set ℝ := {x | x < -2 ∨ x > 3}

-- Theorem for the first inequality
theorem solution_set_f : {x : ℝ | f x < 0} = S₁ := by sorry

-- Theorem for the second inequality
theorem solution_set_g : {x : ℝ | g x > 0} = S₂ := by sorry

end NUMINAMATH_CALUDE_solution_set_f_solution_set_g_l202_20238


namespace NUMINAMATH_CALUDE_fraction_of_juniors_studying_japanese_l202_20234

/-- Proves that the fraction of juniors studying Japanese is 3/4 given the specified conditions. -/
theorem fraction_of_juniors_studying_japanese :
  ∀ (j s : ℕ), -- j: number of juniors, s: number of seniors
  s = 2 * j → -- senior class is twice the size of junior class
  ∃ (x : ℚ), -- x: fraction of juniors studying Japanese
  (1 / 8 : ℚ) * s + x * j = (1 / 3 : ℚ) * (j + s) ∧ -- equation based on given conditions
  x = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_juniors_studying_japanese_l202_20234


namespace NUMINAMATH_CALUDE_caravan_keepers_count_l202_20272

/-- Represents the number of feet for different animals and humans -/
def feet_count : Nat → Nat
| 0 => 2  -- humans and hens
| 1 => 4  -- goats and camels
| _ => 0

/-- The caravan problem -/
theorem caravan_keepers_count 
  (hens goats camels : Nat) 
  (hens_count : hens = 60)
  (goats_count : goats = 35)
  (camels_count : camels = 6)
  (feet_head_diff : 
    ∃ keepers : Nat, 
      hens * feet_count 0 + 
      goats * feet_count 1 + 
      camels * feet_count 1 + 
      keepers * feet_count 0 = 
      hens + goats + camels + keepers + 193) :
  ∃ keepers : Nat, keepers = 10 := by
sorry

end NUMINAMATH_CALUDE_caravan_keepers_count_l202_20272


namespace NUMINAMATH_CALUDE_articles_produced_is_y_l202_20249

/-- Given that x men working x hours a day for x days produce x articles,
    this function calculates the number of articles produced by x men
    working x hours a day for y days. -/
def articles_produced (x y : ℝ) : ℝ :=
  y

/-- Theorem stating that the number of articles produced is y -/
theorem articles_produced_is_y (x y : ℝ) (h : x > 0) :
  articles_produced x y = y :=
by sorry

end NUMINAMATH_CALUDE_articles_produced_is_y_l202_20249


namespace NUMINAMATH_CALUDE_translated_line_point_l202_20262

/-- A line in the xy-plane -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Translates a line vertically by a given amount -/
def translateLine (l : Line) (amount : ℝ) : Line :=
  { slope := l.slope, intercept := l.intercept + amount }

/-- Checks if a point lies on a line -/
def pointOnLine (l : Line) (x y : ℝ) : Prop :=
  y = l.slope * x + l.intercept

theorem translated_line_point (m : ℝ) : 
  let original_line : Line := { slope := 1, intercept := 0 }
  let translated_line := translateLine original_line 3
  pointOnLine translated_line 2 m → m = 5 := by
  sorry

end NUMINAMATH_CALUDE_translated_line_point_l202_20262


namespace NUMINAMATH_CALUDE_games_left_l202_20280

def initial_games : ℕ := 50
def games_given_away : ℕ := 15

theorem games_left : initial_games - games_given_away = 35 := by
  sorry

end NUMINAMATH_CALUDE_games_left_l202_20280


namespace NUMINAMATH_CALUDE_little_john_remaining_money_l202_20231

/-- Calculates the remaining money after Little John's expenditures -/
def remaining_money (initial_amount spent_on_sweets toy_cost friend_gift number_of_friends : ℚ) : ℚ :=
  initial_amount - (spent_on_sweets + toy_cost + friend_gift * number_of_friends)

/-- Theorem: Given Little John's initial amount and expenditures, the remaining money is $11.55 -/
theorem little_john_remaining_money :
  remaining_money 20.10 1.05 2.50 1.00 5 = 11.55 := by
  sorry

#eval remaining_money 20.10 1.05 2.50 1.00 5

end NUMINAMATH_CALUDE_little_john_remaining_money_l202_20231


namespace NUMINAMATH_CALUDE_train_length_l202_20242

/-- Calculates the length of a train given its speed, time to pass a station, and the station's length. -/
theorem train_length (train_speed : ℝ) (time_to_pass : ℝ) (station_length : ℝ) :
  train_speed = 36 * (1000 / 3600) →
  time_to_pass = 45 →
  station_length = 200 →
  train_speed * time_to_pass - station_length = 250 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l202_20242


namespace NUMINAMATH_CALUDE_x_minus_q_in_terms_of_q_l202_20259

theorem x_minus_q_in_terms_of_q (x q : ℝ) (h1 : |x - 3| = q) (h2 : x < 3) : x - q = 3 - 2*q := by
  sorry

end NUMINAMATH_CALUDE_x_minus_q_in_terms_of_q_l202_20259


namespace NUMINAMATH_CALUDE_circumscribed_odd_equal_sides_is_regular_l202_20244

/-- A polygon with an odd number of sides -/
structure OddPolygon where
  n : ℕ
  vertices : Fin (2 * n + 1) → ℝ × ℝ

/-- A circle -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A polygon is circumscribed around a circle if all its sides are tangent to the circle -/
def isCircumscribed (p : OddPolygon) (c : Circle) : Prop := sorry

/-- All sides of a polygon have equal length -/
def hasEqualSides (p : OddPolygon) : Prop := sorry

/-- A polygon is regular if all its sides have equal length and all its angles are equal -/
def isRegular (p : OddPolygon) : Prop := sorry

/-- Main theorem: A circumscribed polygon with an odd number of sides and all sides of equal length is regular -/
theorem circumscribed_odd_equal_sides_is_regular 
  (p : OddPolygon) (c : Circle) 
  (h1 : isCircumscribed p c) 
  (h2 : hasEqualSides p) : 
  isRegular p := by sorry

end NUMINAMATH_CALUDE_circumscribed_odd_equal_sides_is_regular_l202_20244


namespace NUMINAMATH_CALUDE_shaded_area_is_600_l202_20247

-- Define the vertices of the rectangle
def rectangle_vertices : List (ℝ × ℝ) := [(0, 0), (40, 0), (40, 20), (0, 20)]

-- Define the vertices of the shaded polygon
def polygon_vertices : List (ℝ × ℝ) := [(0, 0), (20, 0), (40, 10), (40, 20), (10, 20)]

-- Function to calculate the area of a polygon given its vertices
def polygon_area (vertices : List (ℝ × ℝ)) : ℝ := sorry

-- Theorem statement
theorem shaded_area_is_600 :
  polygon_area polygon_vertices = 600 := by sorry

end NUMINAMATH_CALUDE_shaded_area_is_600_l202_20247


namespace NUMINAMATH_CALUDE_range_of_a_l202_20211

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x^2 - 2*x > a^2 - a - 3) → a ∈ Set.Ioo (-1 : ℝ) 2 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l202_20211


namespace NUMINAMATH_CALUDE_intersection_forms_line_l202_20273

-- Define the equations
def hyperbola (x y : ℝ) : Prop := x * y = 12
def ellipse (x y : ℝ) : Prop := (x^2 / 16) + (y^2 / 36) = 1

-- Define the intersection points
def intersection_points : Set (ℝ × ℝ) :=
  {p | ∃ x y, p = (x, y) ∧ hyperbola x y ∧ ellipse x y}

-- Theorem statement
theorem intersection_forms_line :
  ∃ (a b : ℝ), ∀ (p : ℝ × ℝ), p ∈ intersection_points → 
  (p.1 = a * p.2 + b ∨ p.2 = a * p.1 + b) :=
sorry

end NUMINAMATH_CALUDE_intersection_forms_line_l202_20273


namespace NUMINAMATH_CALUDE_expression_evaluation_l202_20277

theorem expression_evaluation (a b : ℤ) (h1 : a = 2) (h2 : b = -1) :
  (2 * a^2 * b - 4 * a * b^2) - 2 * (a * b^2 + a^2 * b) = -12 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l202_20277


namespace NUMINAMATH_CALUDE_triangle_perimeters_l202_20236

def triangle_side_equation (x : ℝ) : Prop := x^2 - 6*x + 8 = 0

def is_valid_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

def triangle_perimeter (a b c : ℝ) : ℝ := a + b + c

theorem triangle_perimeters : 
  ∃ (a b c : ℝ), 
    triangle_side_equation a ∧ 
    triangle_side_equation b ∧ 
    triangle_side_equation c ∧
    is_valid_triangle a b c ∧
    (triangle_perimeter a b c = 6 ∨ 
     triangle_perimeter a b c = 10 ∨ 
     triangle_perimeter a b c = 12) :=
by sorry

end NUMINAMATH_CALUDE_triangle_perimeters_l202_20236


namespace NUMINAMATH_CALUDE_sum_lowest_two_scores_l202_20215

/-- Represents a set of math test scores -/
structure MathTests where
  scores : Finset ℕ
  count : Nat
  average : ℕ
  median : ℕ
  mode : ℕ

/-- The sum of the lowest two scores in a set of math tests -/
def sumLowestTwo (tests : MathTests) : ℕ :=
  sorry

/-- Theorem: Given 5 math test scores with an average of 90, a median of 91, 
    and a mode of 93, the sum of the lowest two scores is 173 -/
theorem sum_lowest_two_scores (tests : MathTests) 
  (h_count : tests.count = 5)
  (h_avg : tests.average = 90)
  (h_median : tests.median = 91)
  (h_mode : tests.mode = 93) :
  sumLowestTwo tests = 173 := by
  sorry

end NUMINAMATH_CALUDE_sum_lowest_two_scores_l202_20215


namespace NUMINAMATH_CALUDE_male_average_is_100_l202_20256

/-- Represents the average number of tickets sold by a group of members -/
structure GroupAverage where
  count : ℕ  -- Number of members in the group
  average : ℝ  -- Average number of tickets sold by the group

/-- Represents the charitable association -/
structure Association where
  male : GroupAverage
  female : GroupAverage
  nonBinary : GroupAverage

/-- The ratio of male to female to non-binary members is 2:3:5 -/
def memberRatio (a : Association) : Prop :=
  a.male.count = 2 * a.female.count / 3 ∧
  a.nonBinary.count = 5 * a.female.count / 3

/-- The average number of tickets sold by all members is 66 -/
def totalAverage (a : Association) : Prop :=
  (a.male.count * a.male.average + a.female.count * a.female.average + a.nonBinary.count * a.nonBinary.average) /
  (a.male.count + a.female.count + a.nonBinary.count) = 66

/-- Main theorem: Given the conditions, prove that the average number of tickets sold by male members is 100 -/
theorem male_average_is_100 (a : Association)
  (h_ratio : memberRatio a)
  (h_total_avg : totalAverage a)
  (h_female_avg : a.female.average = 70)
  (h_nonbinary_avg : a.nonBinary.average = 50) :
  a.male.average = 100 := by
  sorry

end NUMINAMATH_CALUDE_male_average_is_100_l202_20256


namespace NUMINAMATH_CALUDE_anatoliy_handshakes_l202_20233

theorem anatoliy_handshakes (n : ℕ) (total_handshakes : ℕ) : 
  total_handshakes = 197 →
  (n * (n - 1)) / 2 + 7 = total_handshakes →
  ∃ (k : ℕ), k = 7 ∧ k ≤ n ∧ (n * (n - 1)) / 2 + k = total_handshakes :=
by sorry

end NUMINAMATH_CALUDE_anatoliy_handshakes_l202_20233


namespace NUMINAMATH_CALUDE_alpha_plus_beta_equals_negative_55_l202_20274

theorem alpha_plus_beta_equals_negative_55 :
  ∀ α β : ℝ, 
  (∀ x : ℝ, (x - α) / (x + β) = (x^2 - 50*x + 621) / (x^2 + 75*x - 2016)) →
  α + β = -55 :=
by sorry

end NUMINAMATH_CALUDE_alpha_plus_beta_equals_negative_55_l202_20274


namespace NUMINAMATH_CALUDE_negation_of_p_l202_20261

variable (I : Set ℝ)

def p : Prop := ∀ x ∈ I, x^3 - x^2 + 1 ≤ 0

theorem negation_of_p : 
  ¬p I ↔ ∃ x ∈ I, x^3 - x^2 + 1 > 0 := by sorry

end NUMINAMATH_CALUDE_negation_of_p_l202_20261


namespace NUMINAMATH_CALUDE_parabola_focus_coordinates_focus_coordinates_y_eq_2x_squared_l202_20225

/-- The focus of a parabola y = ax^2 has coordinates (0, 1/(4a)) -/
theorem parabola_focus_coordinates (a : ℝ) (h : a ≠ 0) :
  let f : ℝ × ℝ → ℝ := fun (x, y) ↦ y - a * x^2
  ∃ p : ℝ × ℝ, p.1 = 0 ∧ p.2 = 1 / (4 * a) ∧ 
    ∀ (x y : ℝ), f (x, y) = 0 → (x - p.1)^2 + (y - p.2)^2 = (y - p.2 + 1 / (4 * a))^2 :=
sorry

/-- The focus of the parabola y = 2x^2 has coordinates (0, 1/8) -/
theorem focus_coordinates_y_eq_2x_squared :
  let f : ℝ × ℝ → ℝ := fun (x, y) ↦ y - 2 * x^2
  ∃ p : ℝ × ℝ, p.1 = 0 ∧ p.2 = 1/8 ∧ 
    ∀ (x y : ℝ), f (x, y) = 0 → (x - p.1)^2 + (y - p.2)^2 = (y - p.2 + 1/8)^2 :=
sorry

end NUMINAMATH_CALUDE_parabola_focus_coordinates_focus_coordinates_y_eq_2x_squared_l202_20225


namespace NUMINAMATH_CALUDE_sin_330_degrees_l202_20240

theorem sin_330_degrees : Real.sin (330 * π / 180) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_330_degrees_l202_20240


namespace NUMINAMATH_CALUDE_sqrt_AB_value_l202_20250

def A : ℕ := 10^9 - 987654321
def B : ℚ := (123456789 + 1) / 10

theorem sqrt_AB_value : Real.sqrt (A * B) = 12345679 := by sorry

end NUMINAMATH_CALUDE_sqrt_AB_value_l202_20250


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l202_20203

theorem sufficient_not_necessary_condition (x y : ℝ) :
  (x > y ∧ y > 0 → x / y > 1) ∧
  ∃ x y : ℝ, x / y > 1 ∧ ¬(x > y ∧ y > 0) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l202_20203


namespace NUMINAMATH_CALUDE_problem_solution_l202_20222

theorem problem_solution (m n : ℝ) (h : |3*m - 15| + ((n/3 + 1)^2) = 0) : 2*m - n = 13 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l202_20222


namespace NUMINAMATH_CALUDE_middle_group_frequency_is_32_l202_20285

/-- Represents a histogram with a specified number of rectangles and sample size -/
structure Histogram where
  num_rectangles : ℕ
  sample_size : ℕ

/-- Calculates the frequency of the middle group in a histogram -/
def middle_group_frequency (h : Histogram) (middle_area_ratio : ℚ) : ℕ :=
  sorry

/-- Theorem: The frequency of the middle group is 32 for the given conditions -/
theorem middle_group_frequency_is_32 (h : Histogram) 
    (h_rectangles : h.num_rectangles = 11)
    (h_sample_size : h.sample_size = 160)
    (h_middle_area : middle_group_frequency h (1/4) = 32) : 
  middle_group_frequency h (1/4) = 32 := by
  sorry

end NUMINAMATH_CALUDE_middle_group_frequency_is_32_l202_20285


namespace NUMINAMATH_CALUDE_range_of_a_l202_20219

-- Define the propositions p and q
def p (x a : ℝ) : Prop := -4 < x - a ∧ x - a < 4
def q (x : ℝ) : Prop := (x - 2) * (3 - x) > 0

-- State the theorem
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, (¬p x a → ¬q x)) →
  -1 ≤ a ∧ a ≤ 6 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l202_20219


namespace NUMINAMATH_CALUDE_stratified_sampling_possible_after_adjustment_l202_20293

/-- Represents the population sizes of different age groups -/
structure Population where
  elderly : Nat
  middleAged : Nat
  young : Nat

/-- Represents the sampling parameters -/
structure SamplingParams where
  population : Population
  sampleSize : Nat

/-- Checks if stratified sampling is possible with equal sampling fractions -/
def canStratifySample (p : SamplingParams) : Prop :=
  ∃ (k : Nat), k > 0 ∧
    k ∣ p.sampleSize ∧
    k ∣ p.population.elderly ∧
    k ∣ p.population.middleAged ∧
    k ∣ p.population.young

/-- The given population and sample size -/
def givenParams : SamplingParams :=
  { population := { elderly := 28, middleAged := 54, young := 81 },
    sampleSize := 36 }

/-- The adjusted parameters after removing one elderly person -/
def adjustedParams : SamplingParams :=
  { population := { elderly := 27, middleAged := 54, young := 81 },
    sampleSize := 36 }

/-- Theorem stating that stratified sampling becomes possible after adjustment -/
theorem stratified_sampling_possible_after_adjustment :
  ¬canStratifySample givenParams ∧ canStratifySample adjustedParams :=
sorry


end NUMINAMATH_CALUDE_stratified_sampling_possible_after_adjustment_l202_20293


namespace NUMINAMATH_CALUDE_equation_solution_l202_20246

theorem equation_solution : ∃ (z₁ z₂ : ℂ), 
  z₁ = (-1 + Complex.I * Real.sqrt 21) / 2 ∧
  z₂ = (-1 - Complex.I * Real.sqrt 21) / 2 ∧
  ∀ x : ℂ, (4 * x^2 + 3 * x + 1) / (x - 2) = 2 * x + 5 ↔ x = z₁ ∨ x = z₂ := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l202_20246


namespace NUMINAMATH_CALUDE_sphere_surface_area_l202_20264

theorem sphere_surface_area (V : ℝ) (r : ℝ) (A : ℝ) : 
  V = 72 * Real.pi →
  V = (4/3) * Real.pi * r^3 →
  A = 4 * Real.pi * r^2 →
  A = 36 * Real.pi * 2^(2/3) :=
by sorry

end NUMINAMATH_CALUDE_sphere_surface_area_l202_20264


namespace NUMINAMATH_CALUDE_round_trip_completion_percentage_l202_20278

/-- Calculates the completion percentage of a round-trip given delays on the outbound journey -/
theorem round_trip_completion_percentage 
  (T : ℝ) -- Normal one-way travel time
  (h1 : T > 0) -- Assumption that travel time is positive
  (traffic_delay : ℝ := 0.15) -- 15% increase due to traffic
  (construction_delay : ℝ := 0.10) -- 10% increase due to construction
  (return_completion : ℝ := 0.20) -- 20% of return journey completed
  : (T * (1 + traffic_delay + construction_delay) + return_completion * T) / (2 * T) = 0.725 := by
sorry

end NUMINAMATH_CALUDE_round_trip_completion_percentage_l202_20278
