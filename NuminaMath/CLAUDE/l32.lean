import Mathlib

namespace NUMINAMATH_CALUDE_plate_arrangement_theorem_l32_3276

def blue_plates : ℕ := 6
def red_plates : ℕ := 3
def green_plates : ℕ := 2
def yellow_plates : ℕ := 2

def total_plates : ℕ := blue_plates + red_plates + green_plates + yellow_plates

def circular_arrangements (n : ℕ) (k : List ℕ) : ℕ :=
  Nat.factorial (n - 1) / (k.map Nat.factorial).prod

theorem plate_arrangement_theorem :
  let total_arrangements := circular_arrangements total_plates [blue_plates, red_plates, green_plates, yellow_plates]
  let green_adjacent := circular_arrangements (total_plates - 1) [blue_plates, red_plates, 1, yellow_plates]
  let yellow_adjacent := circular_arrangements (total_plates - 1) [blue_plates, red_plates, green_plates, 1]
  let both_adjacent := circular_arrangements (total_plates - 2) [blue_plates, red_plates, 1, 1]
  total_arrangements - green_adjacent - yellow_adjacent + both_adjacent = 50400 := by
  sorry

end NUMINAMATH_CALUDE_plate_arrangement_theorem_l32_3276


namespace NUMINAMATH_CALUDE_general_admission_price_is_21_85_l32_3217

/-- Represents the ticket sales data for a snooker tournament --/
structure TicketSales where
  totalTickets : ℕ
  totalRevenue : ℚ
  vipPrice : ℚ
  vipGenDifference : ℕ

/-- Calculates the price of a general admission ticket --/
def generalAdmissionPrice (sales : TicketSales) : ℚ :=
  let genTickets := (sales.totalTickets + sales.vipGenDifference) / 2
  let vipTickets := sales.totalTickets - genTickets
  (sales.totalRevenue - sales.vipPrice * vipTickets) / genTickets

/-- Theorem stating that the general admission price is $21.85 --/
theorem general_admission_price_is_21_85 (sales : TicketSales) 
  (h1 : sales.totalTickets = 320)
  (h2 : sales.totalRevenue = 7500)
  (h3 : sales.vipPrice = 45)
  (h4 : sales.vipGenDifference = 276) :
  generalAdmissionPrice sales = 21.85 := by
  sorry

end NUMINAMATH_CALUDE_general_admission_price_is_21_85_l32_3217


namespace NUMINAMATH_CALUDE_line_length_problem_l32_3216

theorem line_length_problem (L : ℝ) (h : 0.75 * L - 0.4 * L = 28) : L = 80 := by
  sorry

end NUMINAMATH_CALUDE_line_length_problem_l32_3216


namespace NUMINAMATH_CALUDE_quadratic_function_transformation_l32_3229

/-- Represents a quadratic function of the form y = (x + a)² + b -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ

/-- Shifts a quadratic function horizontally -/
def shift_horizontal (f : QuadraticFunction) (h : ℝ) : QuadraticFunction :=
  ⟨f.a - h, f.b⟩

/-- Shifts a quadratic function vertically -/
def shift_vertical (f : QuadraticFunction) (v : ℝ) : QuadraticFunction :=
  ⟨f.a, f.b + v⟩

/-- The original quadratic function y = (x + 1)² + 3 -/
def original_function : QuadraticFunction :=
  ⟨1, 3⟩

theorem quadratic_function_transformation :
  let f1 := shift_horizontal original_function 2
  let f2 := shift_vertical f1 (-1)
  f2 = ⟨-1, 2⟩ := by sorry

end NUMINAMATH_CALUDE_quadratic_function_transformation_l32_3229


namespace NUMINAMATH_CALUDE_quadratic_ratio_l32_3263

theorem quadratic_ratio (b c : ℝ) : 
  (∀ x, x^2 - 2100*x - 8400 = (x + b)^2 + c) → c / b = 1058 := by
sorry

end NUMINAMATH_CALUDE_quadratic_ratio_l32_3263


namespace NUMINAMATH_CALUDE_starting_lineup_count_l32_3253

/-- The number of ways to choose a starting lineup from a basketball team -/
def choose_lineup (team_size : ℕ) (lineup_size : ℕ) (point_guard : ℕ) : ℕ :=
  team_size * (Nat.choose (team_size - 1) (lineup_size - 1))

/-- Theorem stating the number of ways to choose the starting lineup -/
theorem starting_lineup_count :
  choose_lineup 12 5 1 = 3960 := by
  sorry

end NUMINAMATH_CALUDE_starting_lineup_count_l32_3253


namespace NUMINAMATH_CALUDE_logarithm_equality_l32_3228

theorem logarithm_equality (a b c x : ℝ) (p q r y : ℝ) :
  a > 0 → b > 0 → c > 0 → x > 0 → x ≠ 1 →
  (∀ (base : ℝ), base > 1 →
    (Real.log a / p = Real.log b / q) ∧
    (Real.log b / q = Real.log c / r) ∧
    (Real.log c / r = Real.log x)) →
  b^3 / (a^2 * c) = x^y →
  y = 3*q - 2*p - r :=
by sorry

end NUMINAMATH_CALUDE_logarithm_equality_l32_3228


namespace NUMINAMATH_CALUDE_parallelogram_points_l32_3296

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if four points form a parallelogram -/
def is_parallelogram (a b c d : Point) : Prop :=
  (b.x - a.x = d.x - c.x ∧ b.y - a.y = d.y - c.y) ∨
  (c.x - a.x = d.x - b.x ∧ c.y - a.y = d.y - b.y) ∨
  (b.x - a.x = c.x - d.x ∧ b.y - a.y = c.y - d.y)

/-- The main theorem -/
theorem parallelogram_points :
  let a : Point := ⟨3, 7⟩
  let b : Point := ⟨4, 6⟩
  let c : Point := ⟨1, -2⟩
  ∀ d : Point, is_parallelogram a b c d ↔ d = ⟨0, -1⟩ ∨ d = ⟨2, -3⟩ ∨ d = ⟨6, 15⟩ :=
by sorry

end NUMINAMATH_CALUDE_parallelogram_points_l32_3296


namespace NUMINAMATH_CALUDE_tenth_term_geometric_sequence_l32_3283

def geometric_sequence (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * r ^ (n - 1)

theorem tenth_term_geometric_sequence :
  geometric_sequence 4 (5/3) 10 = 7812500/19683 := by
  sorry

end NUMINAMATH_CALUDE_tenth_term_geometric_sequence_l32_3283


namespace NUMINAMATH_CALUDE_min_cost_for_89_coins_l32_3218

/-- Represents the cost structure for the coin problem -/
structure CoinProblem where
  total_coins : Nat
  coin_cost : Nat
  yes_fee : Nat
  no_fee : Nat

/-- Calculates the minimum cost to guarantee obtaining the lucky coin -/
def min_cost_to_get_lucky_coin (problem : CoinProblem) : Nat :=
  sorry

/-- Theorem stating the minimum cost for the specific problem instance -/
theorem min_cost_for_89_coins :
  let problem : CoinProblem := {
    total_coins := 89,
    coin_cost := 30,
    yes_fee := 20,
    no_fee := 10
  }
  min_cost_to_get_lucky_coin problem = 130 := by
  sorry

end NUMINAMATH_CALUDE_min_cost_for_89_coins_l32_3218


namespace NUMINAMATH_CALUDE_cos_30_plus_2alpha_l32_3281

theorem cos_30_plus_2alpha (α : ℝ) : 
  (Real.cos 75 * Real.cos α + Real.sin 75 * Real.sin α = 1/3) → 
  Real.cos (30 + 2*α) = 7/9 := by
  sorry

end NUMINAMATH_CALUDE_cos_30_plus_2alpha_l32_3281


namespace NUMINAMATH_CALUDE_inequality_proof_l32_3235

theorem inequality_proof (x y z : ℝ) 
  (sum_zero : x + y + z = 0) 
  (abs_sum_leq_one : |x| + |y| + |z| ≤ 1) : 
  x + y/3 + z/5 ≤ 2/5 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l32_3235


namespace NUMINAMATH_CALUDE_intersection_points_sum_l32_3261

theorem intersection_points_sum (a b c d : ℝ) : 
  (∀ x y : ℝ, (y = -|x - a|^2 + b ∧ y = |x - c|^2 + d) → (x = 1 ∧ y = 8) ∨ (x = 9 ∧ y = 4)) →
  a + c = 10 := by
  sorry

end NUMINAMATH_CALUDE_intersection_points_sum_l32_3261


namespace NUMINAMATH_CALUDE_exists_consecutive_numbers_with_properties_l32_3204

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Theorem stating the existence of two consecutive numbers with given properties -/
theorem exists_consecutive_numbers_with_properties :
  ∃ n : ℕ, sum_of_digits n = 8 ∧ (n + 1) % 8 = 0 := by
  sorry

end NUMINAMATH_CALUDE_exists_consecutive_numbers_with_properties_l32_3204


namespace NUMINAMATH_CALUDE_intersection_implies_a_value_l32_3255

theorem intersection_implies_a_value (A B : Set ℝ) (a : ℝ) : 
  A = {-1, 1, 3} →
  B = {a + 2, a^2 + 4} →
  A ∩ B = {1} →
  a = 1 :=
by sorry

end NUMINAMATH_CALUDE_intersection_implies_a_value_l32_3255


namespace NUMINAMATH_CALUDE_parrot_days_theorem_l32_3290

/-- The number of days in a week -/
def days_per_week : ℕ := 7

/-- The number of phrases the parrot currently knows -/
def current_phrases : ℕ := 17

/-- The number of phrases Georgina teaches the parrot per week -/
def phrases_per_week : ℕ := 2

/-- The number of phrases the parrot knew when Georgina bought it -/
def initial_phrases : ℕ := 3

/-- The number of days Georgina has had the parrot -/
def days_with_parrot : ℕ := 49

theorem parrot_days_theorem :
  (current_phrases - initial_phrases) / phrases_per_week * days_per_week = days_with_parrot := by
  sorry

end NUMINAMATH_CALUDE_parrot_days_theorem_l32_3290


namespace NUMINAMATH_CALUDE_oliver_card_arrangement_l32_3274

/-- Calculates the minimum number of pages required to arrange Oliver's baseball cards --/
def min_pages_for_cards : ℕ :=
  let cards_per_page : ℕ := 3
  let new_cards : ℕ := 2
  let old_cards : ℕ := 10
  let rare_cards : ℕ := 3
  let pages_for_new_cards : ℕ := 1
  let pages_for_rare_cards : ℕ := 1
  let remaining_old_cards : ℕ := old_cards - rare_cards
  let pages_for_remaining_old_cards : ℕ := (remaining_old_cards + cards_per_page - 1) / cards_per_page

  pages_for_new_cards + pages_for_rare_cards + pages_for_remaining_old_cards

theorem oliver_card_arrangement :
  min_pages_for_cards = 5 := by
  sorry

end NUMINAMATH_CALUDE_oliver_card_arrangement_l32_3274


namespace NUMINAMATH_CALUDE_average_speed_first_part_l32_3278

def total_distance : ℝ := 250
def total_time : ℝ := 5.4
def distance_at_v : ℝ := 148
def speed_known : ℝ := 60

theorem average_speed_first_part (v : ℝ) : 
  (distance_at_v / v) + ((total_distance - distance_at_v) / speed_known) = total_time →
  v = 40 := by
sorry

end NUMINAMATH_CALUDE_average_speed_first_part_l32_3278


namespace NUMINAMATH_CALUDE_find_c_and_d_l32_3238

/-- Definition of the polynomial g(x) -/
def g (c d x : ℝ) : ℝ := c * x^3 - 8 * x^2 + d * x - 7

/-- Theorem stating the conditions and the result to be proved -/
theorem find_c_and_d :
  ∀ c d : ℝ,
  g c d 2 = -7 →
  g c d (-1) = -25 →
  c = 2 ∧ d = 8 := by
sorry

end NUMINAMATH_CALUDE_find_c_and_d_l32_3238


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_equation_l32_3282

theorem negation_of_existence (P : ℝ → Prop) :
  (¬ ∃ x, P x) ↔ (∀ x, ¬ P x) :=
by
  sorry

theorem negation_of_quadratic_equation :
  (¬ ∃ x : ℝ, x^2 + 2*x + 5 = 0) ↔ (∀ x : ℝ, x^2 + 2*x + 5 ≠ 0) :=
by
  sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_equation_l32_3282


namespace NUMINAMATH_CALUDE_mutually_exclusive_not_contradictory_l32_3275

/-- Represents the total number of products -/
def total_products : ℕ := 5

/-- Represents the number of genuine products -/
def genuine_products : ℕ := 3

/-- Represents the number of defective products -/
def defective_products : ℕ := 2

/-- Represents the number of products selected -/
def selected_products : ℕ := 2

/-- Represents the event of selecting exactly one defective product -/
def event_one_defective (selected : ℕ) : Prop :=
  selected = 1

/-- Represents the event of selecting exactly two genuine products -/
def event_two_genuine (selected : ℕ) : Prop :=
  selected = 2

/-- Theorem stating that the events are mutually exclusive and not contradictory -/
theorem mutually_exclusive_not_contradictory :
  (¬ (event_one_defective selected_products ∧ event_two_genuine selected_products)) ∧
  (∃ (x : ℕ), ¬ (event_one_defective x ∨ event_two_genuine x)) :=
sorry

end NUMINAMATH_CALUDE_mutually_exclusive_not_contradictory_l32_3275


namespace NUMINAMATH_CALUDE_system_properties_l32_3244

-- Define the system of equations
def system (x y a : ℝ) : Prop :=
  x + 3 * y = 4 - a ∧ x - y = 3 * a

-- Define the statements to be proven
theorem system_properties :
  -- Statement 1
  (∃ x y : ℝ, system x y 2 ∧ x = 5 ∧ y = -1) ∧
  -- Statement 2
  (∃ x y : ℝ, system x y (-2) ∧ x = -y) ∧
  -- Statement 3
  (∀ x y a : ℝ, system x y a → x + 2 * y = 3) ∧
  -- Statement 4
  (∃ x y : ℝ, system x y (-1) ∧ x + y ≠ 4 - (-1)) :=
by sorry

end NUMINAMATH_CALUDE_system_properties_l32_3244


namespace NUMINAMATH_CALUDE_polynomial_solution_l32_3262

theorem polynomial_solution (x : ℝ) : (2*x - 1)^2 = 9 → x = 2 ∨ x = -1 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_solution_l32_3262


namespace NUMINAMATH_CALUDE_perimeter_of_figure_c_l32_3297

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Calculates the perimeter of a rectangle -/
def Rectangle.perimeter (r : Rectangle) : ℝ :=
  2 * (r.width + r.height)

/-- Represents the large rectangle composed of small rectangles -/
structure LargeRectangle where
  small_rectangle : Rectangle
  total_count : ℕ

/-- Theorem: Given the conditions, the perimeter of figure C is 40 cm -/
theorem perimeter_of_figure_c (large_rect : LargeRectangle)
    (h1 : large_rect.total_count = 20)
    (h2 : Rectangle.perimeter { width := 6 * large_rect.small_rectangle.width,
                                height := large_rect.small_rectangle.height } = 56)
    (h3 : Rectangle.perimeter { width := 2 * large_rect.small_rectangle.width,
                                height := 3 * large_rect.small_rectangle.height } = 56) :
  Rectangle.perimeter { width := large_rect.small_rectangle.width,
                        height := 3 * large_rect.small_rectangle.height } = 40 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_of_figure_c_l32_3297


namespace NUMINAMATH_CALUDE_triangle_sine_sum_maximized_equilateral_maximizes_sine_sum_l32_3208

open Real

theorem triangle_sine_sum_maximized (α β γ : ℝ) : 
  0 < α ∧ 0 < β ∧ 0 < γ →
  α + β + γ = π →
  sin α + sin β + sin γ ≤ 3 * sin (π / 3) :=
sorry

theorem equilateral_maximizes_sine_sum (α β γ : ℝ) :
  0 < α ∧ 0 < β ∧ 0 < γ →
  α + β + γ = π →
  sin α + sin β + sin γ = 3 * sin (π / 3) ↔ α = β ∧ β = γ :=
sorry

end NUMINAMATH_CALUDE_triangle_sine_sum_maximized_equilateral_maximizes_sine_sum_l32_3208


namespace NUMINAMATH_CALUDE_investment_return_l32_3209

/-- Given an investment scenario, calculate the percentage return -/
theorem investment_return (total_investment annual_income stock_price : ℝ)
  (h1 : total_investment = 6800)
  (h2 : stock_price = 136)
  (h3 : annual_income = 500) :
  (annual_income / total_investment) * 100 = (500 / 6800) * 100 := by
sorry

#eval (500 / 6800) * 100 -- To display the actual percentage

end NUMINAMATH_CALUDE_investment_return_l32_3209


namespace NUMINAMATH_CALUDE_function_properties_l32_3271

-- Define the function f
def f (a b x : ℝ) : ℝ := x^3 - 3*a*x^2 + 2*b*x

-- State the theorem
theorem function_properties :
  ∃ (a b : ℝ),
  (∀ x, f a b x ≥ f a b 1) ∧ 
  (f a b 1 = -1) ∧
  (a = 1/3) ∧ 
  (b = -1/2) ∧
  (∀ x, x ≤ -1/3 ∨ x ≥ 1 → (deriv (f a b)) x ≥ 0) ∧
  (∀ x, -1/3 ≤ x ∧ x ≤ 1 → (deriv (f a b)) x ≤ 0) ∧
  (∀ α, -1 < α ∧ α < 5/27 → ∃ x y z, x < y ∧ y < z ∧ f a b x = α ∧ f a b y = α ∧ f a b z = α) :=
sorry

end NUMINAMATH_CALUDE_function_properties_l32_3271


namespace NUMINAMATH_CALUDE_second_smallest_five_digit_pascal_correct_l32_3213

/-- Binomial coefficient (n choose k) -/
def binomial (n k : ℕ) : ℕ := Nat.choose n k

/-- Predicate to check if a number is five digits -/
def is_five_digit (n : ℕ) : Prop := 10000 ≤ n ∧ n < 100000

/-- Predicate to check if a number appears in Pascal's triangle -/
def in_pascal_triangle (n : ℕ) : Prop := ∃ (row col : ℕ), binomial row col = n

/-- The second smallest five-digit number in Pascal's triangle -/
def second_smallest_five_digit_pascal : ℕ := 31465

theorem second_smallest_five_digit_pascal_correct :
  is_five_digit second_smallest_five_digit_pascal ∧
  in_pascal_triangle second_smallest_five_digit_pascal ∧
  ∃ (m : ℕ), is_five_digit m ∧ 
             in_pascal_triangle m ∧ 
             m < second_smallest_five_digit_pascal ∧
             ∀ (k : ℕ), is_five_digit k ∧ 
                        in_pascal_triangle k ∧ 
                        k ≠ m → 
                        second_smallest_five_digit_pascal ≤ k :=
by sorry

end NUMINAMATH_CALUDE_second_smallest_five_digit_pascal_correct_l32_3213


namespace NUMINAMATH_CALUDE_polynomial_identity_l32_3295

theorem polynomial_identity : 
  ∀ (a₀ a₁ a₂ a₃ a₄ : ℝ),
  (∀ x : ℝ, (2*x + Real.sqrt 3)^4 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4) →
  (a₀ + a₂ + a₄)^2 - (a₁ + a₃)^2 = 1 := by
sorry

end NUMINAMATH_CALUDE_polynomial_identity_l32_3295


namespace NUMINAMATH_CALUDE_sin_alpha_plus_pi_third_l32_3293

theorem sin_alpha_plus_pi_third (α : Real) (h1 : 0 < α ∧ α < π / 2) 
  (h2 : 7 * Real.sin α = 2 * Real.cos (2 * α)) : 
  Real.sin (α + π / 3) = (1 + 3 * Real.sqrt 5) / 8 := by
  sorry

end NUMINAMATH_CALUDE_sin_alpha_plus_pi_third_l32_3293


namespace NUMINAMATH_CALUDE_like_terms_imply_sum_six_l32_3236

/-- Two monomials are like terms if they have the same variables with the same exponents -/
def like_terms (term1 term2 : ℕ → ℕ → ℚ) : Prop :=
  ∀ (x y : ℕ), term1 x y ≠ 0 ∧ term2 x y ≠ 0 → x = 2 ∧ y = 3

/-- The first monomial 5a^m * b^3 -/
def term1 (m : ℕ) (x y : ℕ) : ℚ :=
  if x = m ∧ y = 3 then 5 else 0

/-- The second monomial -4a^2 * b^(n-1) -/
def term2 (n : ℕ) (x y : ℕ) : ℚ :=
  if x = 2 ∧ y = n - 1 then -4 else 0

theorem like_terms_imply_sum_six (m n : ℕ) :
  like_terms (term1 m) (term2 n) → m + n = 6 := by
  sorry

end NUMINAMATH_CALUDE_like_terms_imply_sum_six_l32_3236


namespace NUMINAMATH_CALUDE_line_perp_to_parallel_planes_l32_3214

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perp : Line → Plane → Prop)
variable (para : Plane → Plane → Prop)

-- State the theorem
theorem line_perp_to_parallel_planes 
  (l : Line) (α β : Plane) :
  perp l β → para α β → perp l α :=
sorry

end NUMINAMATH_CALUDE_line_perp_to_parallel_planes_l32_3214


namespace NUMINAMATH_CALUDE_olivia_spent_l32_3249

/-- Calculates the amount spent given initial amount, amount collected, and amount left after shopping. -/
def amount_spent (initial : ℕ) (collected : ℕ) (left : ℕ) : ℕ :=
  initial + collected - left

/-- Proves that Olivia spent 89 dollars given the problem conditions. -/
theorem olivia_spent (initial : ℕ) (collected : ℕ) (left : ℕ)
  (h1 : initial = 100)
  (h2 : collected = 148)
  (h3 : left = 159) :
  amount_spent initial collected left = 89 := by
  sorry

#eval amount_spent 100 148 159

end NUMINAMATH_CALUDE_olivia_spent_l32_3249


namespace NUMINAMATH_CALUDE_gcd_smallest_prime_factor_subtraction_l32_3234

theorem gcd_smallest_prime_factor_subtraction : 
  10 - (Nat.minFac (Nat.gcd 105 90)) = 7 := by
  sorry

end NUMINAMATH_CALUDE_gcd_smallest_prime_factor_subtraction_l32_3234


namespace NUMINAMATH_CALUDE_inverse_function_implies_a_value_l32_3286

def f (a : ℝ) (x : ℝ) : ℝ := a - 2 * x

theorem inverse_function_implies_a_value (a : ℝ) :
  (∃ g : ℝ → ℝ, Function.LeftInverse g (f a) ∧ Function.RightInverse g (f a) ∧ g (-3) = 3) →
  a = 3 := by
  sorry

end NUMINAMATH_CALUDE_inverse_function_implies_a_value_l32_3286


namespace NUMINAMATH_CALUDE_roses_count_l32_3294

/-- The number of pots of roses in the People's Park -/
def roses : ℕ := 65

/-- The number of pots of lilac flowers in the People's Park -/
def lilacs : ℕ := 180

/-- Theorem stating that the number of pots of roses is correct given the conditions -/
theorem roses_count :
  roses = 65 ∧ lilacs = 180 ∧ lilacs = 3 * roses - 15 :=
by sorry

end NUMINAMATH_CALUDE_roses_count_l32_3294


namespace NUMINAMATH_CALUDE_max_area_rectangle_l32_3273

/-- Represents a rectangle with integer dimensions --/
structure Rectangle where
  length : ℕ
  width : ℕ

/-- Calculates the perimeter of a rectangle --/
def perimeter (r : Rectangle) : ℕ := 2 * (r.length + r.width)

/-- Calculates the area of a rectangle --/
def area (r : Rectangle) : ℕ := r.length * r.width

/-- Checks if a number is even --/
def isEven (n : ℕ) : Prop := ∃ k, n = 2 * k

/-- Theorem: The maximum area of a rectangle with perimeter 40 and even length is 100 --/
theorem max_area_rectangle :
  ∀ r : Rectangle,
    perimeter r = 40 →
    isEven r.length →
    area r ≤ 100 ∧
    (area r = 100 ↔ r.length = 10 ∧ r.width = 10) :=
by sorry


end NUMINAMATH_CALUDE_max_area_rectangle_l32_3273


namespace NUMINAMATH_CALUDE_square_area_l32_3223

theorem square_area (side_length : ℝ) (h : side_length = 19) :
  side_length * side_length = 361 := by
  sorry

end NUMINAMATH_CALUDE_square_area_l32_3223


namespace NUMINAMATH_CALUDE_binomial_sum_equals_higher_binomial_l32_3277

-- Define the binomial coefficient function
def binomial (n k : ℕ) : ℕ := 
  if k > n then 0
  else (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

-- State the theorem
theorem binomial_sum_equals_higher_binomial :
  binomial 6 3 + binomial 6 2 = binomial 7 3 := by
  sorry

end NUMINAMATH_CALUDE_binomial_sum_equals_higher_binomial_l32_3277


namespace NUMINAMATH_CALUDE_james_fish_catch_l32_3285

/-- The total weight of fish James caught -/
def total_fish_weight (trout salmon tuna bass catfish : ℝ) : ℝ :=
  trout + salmon + tuna + bass + catfish

/-- Theorem stating the total weight of fish James caught -/
theorem james_fish_catch :
  ∃ (trout salmon tuna bass catfish : ℝ),
    trout = 200 ∧
    salmon = trout * 1.6 ∧
    tuna = trout * 2 ∧
    bass = salmon * 3 ∧
    catfish = tuna / 3 ∧
    total_fish_weight trout salmon tuna bass catfish = 2013.33 :=
by
  sorry

end NUMINAMATH_CALUDE_james_fish_catch_l32_3285


namespace NUMINAMATH_CALUDE_point_on_line_segment_l32_3215

structure Point where
  x : ℝ
  y : ℝ

def Triangle (A B C : Point) : Prop :=
  A ≠ B ∧ B ≠ C ∧ C ≠ A

def OnSegment (D A B : Point) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ D.x = A.x + t * (B.x - A.x) ∧ D.y = A.y + t * (B.y - A.y)

theorem point_on_line_segment (A B C D : Point) :
  Triangle A B C →
  A = Point.mk 1 2 →
  B = Point.mk 4 6 →
  C = Point.mk 6 3 →
  OnSegment D A B →
  D.y = (4/3) * D.x - (2/3) →
  ∃ t : ℝ, 1 ≤ t ∧ t ≤ 4 ∧ D = Point.mk t ((4/3) * t - (2/3)) :=
by sorry

end NUMINAMATH_CALUDE_point_on_line_segment_l32_3215


namespace NUMINAMATH_CALUDE_solve_for_a_l32_3221

theorem solve_for_a (x a : ℚ) (h1 : x - 2 * a + 5 = 0) (h2 : x = -2) : a = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_a_l32_3221


namespace NUMINAMATH_CALUDE_red_part_length_l32_3224

/-- The length of the red part of a pencil given specific color proportions -/
theorem red_part_length (total_length : ℝ) (green_ratio : ℝ) (gold_ratio : ℝ) (red_ratio : ℝ)
  (h_total : total_length = 15)
  (h_green : green_ratio = 7/10)
  (h_gold : gold_ratio = 3/7)
  (h_red : red_ratio = 2/3) :
  red_ratio * (total_length - green_ratio * total_length - gold_ratio * (total_length - green_ratio * total_length)) =
  2/3 * (15 - 15 * 7/10 - (15 - 15 * 7/10) * 3/7) :=
by sorry

end NUMINAMATH_CALUDE_red_part_length_l32_3224


namespace NUMINAMATH_CALUDE_sterling_candy_proof_l32_3279

/-- The number of candy pieces earned for a correct answer -/
def correct_reward : ℕ := 3

/-- The total number of questions answered -/
def total_questions : ℕ := 7

/-- The number of questions answered correctly -/
def correct_answers : ℕ := 7

/-- The number of additional correct answers -/
def additional_correct : ℕ := 2

/-- The total number of candy pieces earned if Sterling answered 2 more questions correctly -/
def total_candy : ℕ := correct_reward * (correct_answers + additional_correct)

theorem sterling_candy_proof :
  total_candy = 27 :=
sorry

end NUMINAMATH_CALUDE_sterling_candy_proof_l32_3279


namespace NUMINAMATH_CALUDE_age_difference_l32_3242

theorem age_difference (man_age son_age : ℕ) : 
  man_age > son_age →
  son_age = 25 →
  man_age + 2 = 2 * (son_age + 2) →
  man_age - son_age = 27 := by
sorry

end NUMINAMATH_CALUDE_age_difference_l32_3242


namespace NUMINAMATH_CALUDE_salem_poem_lines_per_stanza_l32_3220

/-- Represents a poem with stanzas, lines, and words. -/
structure Poem where
  num_stanzas : ℕ
  words_per_line : ℕ
  total_words : ℕ

/-- Calculates the number of lines per stanza in a poem. -/
def lines_per_stanza (p : Poem) : ℕ :=
  (p.total_words / p.words_per_line) / p.num_stanzas

/-- Theorem stating that for a poem with 20 stanzas, 8 words per line, 
    and 1600 total words, each stanza has 10 lines. -/
theorem salem_poem_lines_per_stanza :
  let p : Poem := { num_stanzas := 20, words_per_line := 8, total_words := 1600 }
  lines_per_stanza p = 10 := by
  sorry

end NUMINAMATH_CALUDE_salem_poem_lines_per_stanza_l32_3220


namespace NUMINAMATH_CALUDE_bernold_can_win_l32_3299

/-- Represents the game board -/
structure GameBoard :=
  (size : Nat)
  (arnold_moves : Nat → Nat → Bool)
  (bernold_moves : Nat → Nat → Bool)

/-- Defines the game rules -/
def game_rules (board : GameBoard) : Prop :=
  board.size = 2007 ∧
  (∀ x y, board.arnold_moves x y ↔ 
    x + 1 < board.size ∧ y + 1 < board.size ∧ 
    ¬board.bernold_moves x y ∧ ¬board.bernold_moves (x+1) y ∧ 
    ¬board.bernold_moves x (y+1) ∧ ¬board.bernold_moves (x+1) (y+1)) ∧
  (∀ x y, board.bernold_moves x y → x < board.size ∧ y < board.size)

/-- Theorem: Bernold can always win -/
theorem bernold_can_win (board : GameBoard) (h : game_rules board) :
  ∃ (strategy : Nat → Nat → Bool), 
    (∀ x y, strategy x y → board.bernold_moves x y) ∧
    (∀ (arnold_strategy : Nat → Nat → Bool), 
      (∀ x y, arnold_strategy x y → board.arnold_moves x y) →
      (Finset.sum (Finset.product (Finset.range board.size) (Finset.range board.size))
        (fun (x, y) => if arnold_strategy x y then 4 else 0) ≤ 
          (1003 * 1004) / 2)) :=
sorry

end NUMINAMATH_CALUDE_bernold_can_win_l32_3299


namespace NUMINAMATH_CALUDE_restricted_arrangements_five_students_l32_3258

/-- The number of ways to arrange n students in a row. -/
def arrangements (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to arrange n students in a row with one specific student not in the front. -/
def arrangementsWithRestriction (n : ℕ) : ℕ := (n - 1) * arrangements (n - 1)

/-- Theorem stating that for 5 students, there are 96 ways to arrange them with one specific student not in the front. -/
theorem restricted_arrangements_five_students :
  arrangementsWithRestriction 5 = 96 := by
  sorry

#eval arrangementsWithRestriction 5  -- This should output 96

end NUMINAMATH_CALUDE_restricted_arrangements_five_students_l32_3258


namespace NUMINAMATH_CALUDE_sequence_sum_l32_3243

/-- Given a sequence {a_n} where a_1 = 1 and S_n = n^2 * a_n for all positive integers n,
    prove that the sum of the first n terms (S_n) is equal to 2n / (n+1). -/
theorem sequence_sum (a : ℕ+ → ℝ) (S : ℕ+ → ℝ) :
  a 1 = 1 →
  (∀ n : ℕ+, S n = n^2 * a n) →
  ∀ n : ℕ+, S n = (2 * n : ℝ) / (n + 1) :=
sorry

end NUMINAMATH_CALUDE_sequence_sum_l32_3243


namespace NUMINAMATH_CALUDE_perfect_square_addition_l32_3206

theorem perfect_square_addition (n : Nat) : ∃ (m : Nat), (n + 49)^2 = 4440 + 49 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_addition_l32_3206


namespace NUMINAMATH_CALUDE_triangle_area_implies_x_value_l32_3298

theorem triangle_area_implies_x_value (x : ℝ) (h1 : x > 0) :
  (1/2 : ℝ) * x * (3*x) = 54 → x = 6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_implies_x_value_l32_3298


namespace NUMINAMATH_CALUDE_parabola_c_value_l32_3237

/-- Given a parabola y = x^2 + bx + c passing through (2,3) and (4,3), prove c = 11 -/
theorem parabola_c_value (b c : ℝ) 
  (eq1 : 3 = 2^2 + 2*b + c) 
  (eq2 : 3 = 4^2 + 4*b + c) : 
  c = 11 := by sorry

end NUMINAMATH_CALUDE_parabola_c_value_l32_3237


namespace NUMINAMATH_CALUDE_parabola_equation_correct_l32_3288

/-- A parabola with vertex (h, k) and passing through point (x₀, y₀) -/
structure Parabola where
  h : ℝ  -- x-coordinate of vertex
  k : ℝ  -- y-coordinate of vertex
  x₀ : ℝ  -- x-coordinate of point on parabola
  y₀ : ℝ  -- y-coordinate of point on parabola

/-- The equation of a parabola in the form ax^2 + bx + c -/
structure ParabolaEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Theorem: The given parabola equation represents the specified parabola -/
theorem parabola_equation_correct (p : Parabola) (eq : ParabolaEquation) : 
  p.h = 3 ∧ p.k = 5 ∧ p.x₀ = 2 ∧ p.y₀ = 2 ∧
  eq.a = -3 ∧ eq.b = 18 ∧ eq.c = -22 →
  ∀ x y : ℝ, y = eq.a * x^2 + eq.b * x + eq.c ↔ 
    (x = p.h ∧ y = p.k) ∨ 
    (y = eq.a * (x - p.h)^2 + p.k) ∨
    (x = p.x₀ ∧ y = p.y₀) := by
  sorry

end NUMINAMATH_CALUDE_parabola_equation_correct_l32_3288


namespace NUMINAMATH_CALUDE_metro_line_stations_l32_3211

theorem metro_line_stations (x : ℕ) (h : x * (x - 1) = 1482) :
  x * (x - 1) = 1482 ∧ x > 1 := by
  sorry

end NUMINAMATH_CALUDE_metro_line_stations_l32_3211


namespace NUMINAMATH_CALUDE_total_pens_l32_3233

theorem total_pens (black_pens blue_pens : ℕ) :
  black_pens = 4 → blue_pens = 4 → black_pens + blue_pens = 8 :=
by sorry

end NUMINAMATH_CALUDE_total_pens_l32_3233


namespace NUMINAMATH_CALUDE_problem_statement_l32_3256

theorem problem_statement (n : ℕ) (x m : ℝ) :
  let p := x^2 - 2*x - 8 ≤ 0
  let q := |x - 2| ≤ m
  (∀ k : ℕ, k ≤ n → ((-1:ℝ)^k * (n.choose k) = (-1)^n * (n.choose (n-k)))) →
  (
    (m = 3 ∧ p ∧ q) → -1 ≤ x ∧ x ≤ 4
  ) ∧
  (
    (∀ y : ℝ, (y^2 - 2*y - 8 ≤ 0) → |y - 2| ≤ m) → m ≥ 4
  ) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l32_3256


namespace NUMINAMATH_CALUDE_even_increasing_function_inequality_l32_3212

-- Define an even function on ℝ
def even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

-- Define an increasing function on [0,+∞)
def increasing_on_nonneg (f : ℝ → ℝ) : Prop := ∀ x y, 0 ≤ x → 0 ≤ y → x < y → f x < f y

theorem even_increasing_function_inequality (f : ℝ → ℝ) (k : ℝ) 
  (h_even : even_function f) 
  (h_increasing : increasing_on_nonneg f) 
  (h_inequality : f k > f 2) : 
  k > 2 ∨ k < -2 :=
sorry

end NUMINAMATH_CALUDE_even_increasing_function_inequality_l32_3212


namespace NUMINAMATH_CALUDE_video_game_lives_l32_3254

theorem video_game_lives (initial_lives lost_lives gained_lives : ℕ) 
  (h1 : initial_lives = 10)
  (h2 : lost_lives = 6)
  (h3 : gained_lives = 37) :
  initial_lives - lost_lives + gained_lives = 41 :=
by sorry

end NUMINAMATH_CALUDE_video_game_lives_l32_3254


namespace NUMINAMATH_CALUDE_equilateral_triangle_third_vertex_y_coord_l32_3239

/-- Given an equilateral triangle with two vertices at (0,3) and (10,3),
    prove that the y-coordinate of the third vertex in the first quadrant is 3 + 5√3. -/
theorem equilateral_triangle_third_vertex_y_coord :
  let v1 : ℝ × ℝ := (0, 3)
  let v2 : ℝ × ℝ := (10, 3)
  let side_length : ℝ := 10
  ∃ (x y : ℝ), 
    x > 0 ∧ y > 3 ∧
    (x - 0)^2 + (y - 3)^2 = side_length^2 ∧
    (x - 10)^2 + (y - 3)^2 = side_length^2 ∧
    y = 3 + 5 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_equilateral_triangle_third_vertex_y_coord_l32_3239


namespace NUMINAMATH_CALUDE_initial_distance_adrian_colton_initial_distance_l32_3232

/-- The initial distance between Adrian and Colton given their relative motion -/
theorem initial_distance (speed : ℝ) (time : ℝ) (final_distance : ℝ) : ℝ :=
  let distance_run := speed * time
  distance_run + final_distance

/-- Proof of the initial distance between Adrian and Colton -/
theorem adrian_colton_initial_distance : 
  initial_distance 17 13 68 = 289 := by
  sorry

end NUMINAMATH_CALUDE_initial_distance_adrian_colton_initial_distance_l32_3232


namespace NUMINAMATH_CALUDE_fourth_person_height_l32_3287

theorem fourth_person_height (h₁ h₂ h₃ h₄ : ℕ) : 
  h₁ < h₂ ∧ h₂ < h₃ ∧ h₃ < h₄ →  -- Heights are in increasing order
  h₂ = h₁ + 2 →                 -- Difference between 1st and 2nd is 2 inches
  h₃ = h₂ + 2 →                 -- Difference between 2nd and 3rd is 2 inches
  h₄ = h₃ + 6 →                 -- Difference between 3rd and 4th is 6 inches
  (h₁ + h₂ + h₃ + h₄) / 4 = 78  -- Average height is 78 inches
  → h₄ = 84 :=                  -- Fourth person's height is 84 inches
by sorry

end NUMINAMATH_CALUDE_fourth_person_height_l32_3287


namespace NUMINAMATH_CALUDE_upstream_downstream_time_ratio_l32_3205

/-- Proves that the ratio of time taken to row upstream to downstream is 2:1 -/
theorem upstream_downstream_time_ratio 
  (boat_speed : ℝ) 
  (stream_speed : ℝ) 
  (h1 : boat_speed = 36) 
  (h2 : stream_speed = 12) : 
  (boat_speed - stream_speed) / (boat_speed + stream_speed) = 1 / 2 := by
  sorry

#check upstream_downstream_time_ratio

end NUMINAMATH_CALUDE_upstream_downstream_time_ratio_l32_3205


namespace NUMINAMATH_CALUDE_root_implies_p_minus_q_l32_3284

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the equation
def equation (p q : ℝ) (x : ℂ) : Prop :=
  2 * x^2 + p * x + q = 0

-- State the theorem
theorem root_implies_p_minus_q (p q : ℝ) :
  equation p q (-2 * i - 3) → p - q = -14 := by
  sorry

end NUMINAMATH_CALUDE_root_implies_p_minus_q_l32_3284


namespace NUMINAMATH_CALUDE_partnership_investment_ratio_l32_3222

/-- Given a partnership business where:
  * A's investment is k times B's investment
  * A's investment period is twice B's investment period
  * B's profit is 7000
  * Total profit is 49000
  Prove that the ratio of A's investment to B's investment is 3:1 -/
theorem partnership_investment_ratio 
  (k : ℚ) 
  (b_profit : ℚ) 
  (total_profit : ℚ) 
  (h1 : b_profit = 7000)
  (h2 : total_profit = 49000)
  (h3 : k * b_profit * 2 + b_profit = total_profit) : 
  k = 3 := by
sorry

end NUMINAMATH_CALUDE_partnership_investment_ratio_l32_3222


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l32_3230

theorem sum_of_squares_of_roots (p q r : ℝ) : 
  (3 * p^3 - 2 * p^2 + 5 * p + 15 = 0) →
  (3 * q^3 - 2 * q^2 + 5 * q + 15 = 0) →
  (3 * r^3 - 2 * r^2 + 5 * r + 15 = 0) →
  p^2 + q^2 + r^2 = -26/9 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l32_3230


namespace NUMINAMATH_CALUDE_characterize_valid_functions_l32_3245

def is_valid_function (f : ℕ → ℕ) : Prop :=
  (∀ x, f x ≤ x^2) ∧
  (∀ x y, x > y → (x - y) ∣ (f x - f y))

theorem characterize_valid_functions :
  ∀ f : ℕ → ℕ, is_valid_function f ↔
    (∀ x, f x = 0) ∨
    (∀ x, f x = x) ∨
    (∀ x, f x = x^2 - x) ∨
    (∀ x, f x = x^2) :=
sorry


end NUMINAMATH_CALUDE_characterize_valid_functions_l32_3245


namespace NUMINAMATH_CALUDE_cosine_function_theorem_l32_3251

def is_periodic (f : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x, f (x + T) = f x

theorem cosine_function_theorem (f : ℝ → ℝ) (T : ℝ) :
  is_periodic f T →
  (∀ x, Real.cos x = f x - 2 * f (x - π)) →
  (∀ x, Real.cos x = f (x - T) - 2 * f (x - T - π)) →
  (∀ x, Real.cos x = Real.cos (x - T)) →
  (∀ x, f x = (1/3) * Real.cos x) :=
by sorry

end NUMINAMATH_CALUDE_cosine_function_theorem_l32_3251


namespace NUMINAMATH_CALUDE_factorial_sum_equality_l32_3240

theorem factorial_sum_equality : 7 * Nat.factorial 7 + 6 * Nat.factorial 6 + 2 * Nat.factorial 5 + Nat.factorial 5 = 39960 := by
  sorry

end NUMINAMATH_CALUDE_factorial_sum_equality_l32_3240


namespace NUMINAMATH_CALUDE_l_shaped_floor_paving_cost_l32_3280

/-- Calculates the total cost of paving an L-shaped floor with two types of slabs -/
def total_paving_cost (large_length large_width small_length small_width type_a_cost type_b_cost : ℝ) : ℝ :=
  let large_area := large_length * large_width
  let small_area := small_length * small_width
  let large_cost := large_area * type_a_cost
  let small_cost := small_area * type_b_cost
  large_cost + small_cost

/-- Theorem stating that the total cost of paving the L-shaped floor is Rs. 13,781.25 -/
theorem l_shaped_floor_paving_cost :
  total_paving_cost 5.5 3.75 2.5 1.25 600 450 = 13781.25 := by
  sorry

end NUMINAMATH_CALUDE_l_shaped_floor_paving_cost_l32_3280


namespace NUMINAMATH_CALUDE_petes_flag_shapes_l32_3250

def us_stars : ℕ := 50
def us_stripes : ℕ := 13

def circles : ℕ := us_stars / 2 - 3
def squares : ℕ := us_stripes * 2 + 6
def triangles : ℕ := (us_stars - us_stripes) * 2
def diamonds : ℕ := (us_stars + us_stripes) / 4

theorem petes_flag_shapes :
  circles + squares + triangles + diamonds = 143 := by
  sorry

end NUMINAMATH_CALUDE_petes_flag_shapes_l32_3250


namespace NUMINAMATH_CALUDE_simplify_expression_l32_3257

theorem simplify_expression (x : ℝ) (hx : x ≠ 0) :
  x⁻¹ - x + 2 = (1 - (x - 1)^2) / x := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l32_3257


namespace NUMINAMATH_CALUDE_limit_of_ratio_l32_3247

def arithmetic_sequence (n : ℕ) : ℝ := 2 * n - 1

def sum_of_terms (n : ℕ) : ℝ := n^2

theorem limit_of_ratio :
  ∀ ε > 0, ∃ N, ∀ n ≥ N, |sum_of_terms n / (arithmetic_sequence n)^2 - 1/4| < ε :=
by sorry

end NUMINAMATH_CALUDE_limit_of_ratio_l32_3247


namespace NUMINAMATH_CALUDE_distance_to_larger_section_specific_case_l32_3292

/-- Represents a right triangular pyramid with two parallel cross sections -/
structure RightTriangularPyramid where
  /-- Area of the smaller cross section -/
  area_small : ℝ
  /-- Area of the larger cross section -/
  area_large : ℝ
  /-- Distance between the two cross sections -/
  cross_section_distance : ℝ

/-- Calculates the distance from the apex to the larger cross section -/
def distance_to_larger_section (p : RightTriangularPyramid) : ℝ :=
  sorry

/-- Theorem stating the distance to the larger cross section for specific conditions -/
theorem distance_to_larger_section_specific_case :
  let p : RightTriangularPyramid := {
    area_small := 150 * Real.sqrt 3,
    area_large := 300 * Real.sqrt 3,
    cross_section_distance := 10
  }
  distance_to_larger_section p = 10 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_larger_section_specific_case_l32_3292


namespace NUMINAMATH_CALUDE_existence_of_odd_fifth_powers_sum_l32_3246

theorem existence_of_odd_fifth_powers_sum (m : ℤ) :
  ∃ (a b : ℤ) (k : ℕ+), 
    Odd a ∧ Odd b ∧ (2 * m = a^5 + b^5 + k * 2^100) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_odd_fifth_powers_sum_l32_3246


namespace NUMINAMATH_CALUDE_line_tangent_to_parabola_l32_3264

/-- A line is tangent to a parabola if and only if the value of k satisfies the tangency condition -/
theorem line_tangent_to_parabola (x y k : ℝ) : 
  (∃ (x₀ y₀ : ℝ), (4 * x₀ + 3 * y₀ + k = 0) ∧ (y₀^2 = 12 * x₀) ∧
    (∀ (x' y' : ℝ), (4 * x' + 3 * y' + k = 0) ∧ (y'^2 = 12 * x') → (x' = x₀ ∧ y' = y₀))) ↔
  (k = 27 / 4) := by
sorry

end NUMINAMATH_CALUDE_line_tangent_to_parabola_l32_3264


namespace NUMINAMATH_CALUDE_total_items_sold_is_727_l32_3252

/-- Represents the data for a single day of James' sales --/
structure DayData where
  houses : ℕ
  successRate : ℚ
  itemsPerHouse : ℕ

/-- Calculates the number of items sold in a day --/
def itemsSoldInDay (data : DayData) : ℚ :=
  data.houses * data.successRate * data.itemsPerHouse

/-- The week's data --/
def weekData : List DayData := [
  { houses := 20, successRate := 1, itemsPerHouse := 2 },
  { houses := 40, successRate := 4/5, itemsPerHouse := 3 },
  { houses := 50, successRate := 9/10, itemsPerHouse := 1 },
  { houses := 60, successRate := 3/4, itemsPerHouse := 4 },
  { houses := 80, successRate := 1/2, itemsPerHouse := 2 },
  { houses := 100, successRate := 7/10, itemsPerHouse := 1 },
  { houses := 120, successRate := 3/5, itemsPerHouse := 3 }
]

/-- Theorem: The total number of items sold during the week is 727 --/
theorem total_items_sold_is_727 : 
  (weekData.map itemsSoldInDay).sum = 727 := by
  sorry

end NUMINAMATH_CALUDE_total_items_sold_is_727_l32_3252


namespace NUMINAMATH_CALUDE_gcd_properties_l32_3203

theorem gcd_properties (a b n : ℕ) (c : ℤ) (h1 : a ≠ 0) (h2 : c > 0) :
  let d := Nat.gcd a b
  (n ∣ a ∧ n ∣ b ↔ n ∣ d) ∧
  (Nat.gcd (a * c.natAbs) (b * c.natAbs) = c.natAbs * Nat.gcd a b) :=
by sorry

end NUMINAMATH_CALUDE_gcd_properties_l32_3203


namespace NUMINAMATH_CALUDE_prob_diff_suits_one_heart_correct_l32_3260

/-- The number of cards in a standard deck --/
def standard_deck_size : ℕ := 52

/-- The number of suits in a standard deck --/
def num_suits : ℕ := 4

/-- The number of cards of each suit in a standard deck --/
def cards_per_suit : ℕ := standard_deck_size / num_suits

/-- The total number of cards in two combined standard decks --/
def total_cards : ℕ := 2 * standard_deck_size

/-- The probability of selecting two cards from two combined standard 52-card decks,
    where the cards are of different suits and at least one is a heart --/
def prob_diff_suits_one_heart : ℚ := 91467 / 276044

theorem prob_diff_suits_one_heart_correct :
  let total_combinations := total_cards.choose 2
  let diff_suit_prob := (total_cards - cards_per_suit) / (total_cards - 1)
  let at_least_one_heart := total_combinations - (total_cards - 2 * cards_per_suit).choose 2
  diff_suit_prob * (at_least_one_heart / total_combinations) = prob_diff_suits_one_heart := by
  sorry

end NUMINAMATH_CALUDE_prob_diff_suits_one_heart_correct_l32_3260


namespace NUMINAMATH_CALUDE_largest_integer_for_negative_quadratic_six_satisfies_inequality_seven_does_not_satisfy_inequality_l32_3248

theorem largest_integer_for_negative_quadratic : 
  ∀ x : ℤ, x^2 - 11*x + 28 < 0 → x ≤ 6 :=
by sorry

theorem six_satisfies_inequality : 
  (6 : ℤ)^2 - 11*6 + 28 < 0 :=
by sorry

theorem seven_does_not_satisfy_inequality : 
  (7 : ℤ)^2 - 11*7 + 28 ≥ 0 :=
by sorry

end NUMINAMATH_CALUDE_largest_integer_for_negative_quadratic_six_satisfies_inequality_seven_does_not_satisfy_inequality_l32_3248


namespace NUMINAMATH_CALUDE_jack_savings_after_eight_weeks_l32_3272

/-- Calculates the amount in Jack's savings account after a given number of weeks -/
def savings_after_weeks (initial_amount : ℕ) (weekly_allowance : ℕ) (weekly_spending : ℕ) (weeks : ℕ) : ℕ :=
  initial_amount + (weekly_allowance - weekly_spending) * weeks

/-- Theorem: Jack's savings after 8 weeks -/
theorem jack_savings_after_eight_weeks :
  savings_after_weeks 43 10 3 8 = 99 := by
  sorry

end NUMINAMATH_CALUDE_jack_savings_after_eight_weeks_l32_3272


namespace NUMINAMATH_CALUDE_top_field_is_nine_l32_3265

/-- Represents a labelling of the figure -/
def Labelling := Fin 9 → Fin 9

/-- Check if a labelling is valid -/
def is_valid (l : Labelling) : Prop :=
  let s := l 0 + l 1 + l 2 -- sum of top row
  (l 0 + l 1 + l 2 = s) ∧
  (l 3 + l 4 + l 5 = s) ∧
  (l 6 + l 7 + l 8 = s) ∧
  (l 0 + l 3 + l 6 = s) ∧
  (l 1 + l 4 + l 7 = s) ∧
  (l 2 + l 5 + l 8 = s) ∧
  (l 0 + l 4 + l 8 = s) ∧
  (l 2 + l 4 + l 6 = s) ∧
  Function.Injective l

theorem top_field_is_nine (l : Labelling) (h : is_valid l) : l 0 = 9 := by
  sorry

#check top_field_is_nine

end NUMINAMATH_CALUDE_top_field_is_nine_l32_3265


namespace NUMINAMATH_CALUDE_dandelion_ratio_l32_3219

theorem dandelion_ratio : 
  ∀ (billy_initial george_initial billy_additional george_additional : ℕ) 
    (average : ℚ),
  billy_initial = 36 →
  billy_additional = 10 →
  george_additional = 10 →
  average = 34 →
  (billy_initial + george_initial + billy_additional + george_additional : ℚ) / 2 = average →
  george_initial * 3 = billy_initial :=
by sorry

end NUMINAMATH_CALUDE_dandelion_ratio_l32_3219


namespace NUMINAMATH_CALUDE_tangent_lines_count_l32_3268

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 - 3 * x^2 + 1

-- Define the arithmetic sequence condition
def arithmetic_sequence (a : ℝ) : Prop :=
  ∃ d : ℝ, d ≠ 0 ∧ f a (-a) + f a (3*a) = 2 * f a a

-- Define a tangent line from the origin
def tangent_from_origin (a : ℝ) (x₀ : ℝ) : Prop :=
  ∃ y₀ : ℝ, f a x₀ = y₀ ∧ y₀ = (3 * a * x₀^2 - 6 * x₀) * x₀

-- Main theorem
theorem tangent_lines_count (a : ℝ) (ha : a ≠ 0) :
  arithmetic_sequence a →
  ∃! (count : ℕ), count = 2 ∧ 
    ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
      tangent_from_origin a x₁ ∧ 
      tangent_from_origin a x₂ ∧
      ∀ (x : ℝ), tangent_from_origin a x → (x = x₁ ∨ x = x₂) :=
sorry

end NUMINAMATH_CALUDE_tangent_lines_count_l32_3268


namespace NUMINAMATH_CALUDE_quadratic_function_range_l32_3259

def quadratic_function (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 4 * a * x - 6

theorem quadratic_function_range (a : ℝ) :
  (∃ y₁ y₂ y₃ y₄ : ℝ,
    quadratic_function a (-4) = y₁ ∧
    quadratic_function a (-3) = y₂ ∧
    quadratic_function a 0 = y₃ ∧
    quadratic_function a 2 = y₄ ∧
    (y₁ ≤ 0 ∧ y₂ ≤ 0 ∧ y₃ ≤ 0 ∧ y₄ > 0) ∨
    (y₁ ≤ 0 ∧ y₂ > 0 ∧ y₃ ≤ 0 ∧ y₄ ≤ 0) ∨
    (y₁ ≤ 0 ∧ y₂ ≤ 0 ∧ y₃ > 0 ∧ y₄ ≤ 0) ∨
    (y₁ > 0 ∧ y₂ ≤ 0 ∧ y₃ ≤ 0 ∧ y₄ ≤ 0)) →
  a < -2 ∨ a > 1/2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_function_range_l32_3259


namespace NUMINAMATH_CALUDE_gum_to_candy_ratio_l32_3241

/-- The cost of a candy bar in dollars -/
def candy_cost : ℚ := 3/2

/-- The total cost of the purchase in dollars -/
def total_cost : ℚ := 6

/-- The number of packs of gum purchased -/
def gum_packs : ℕ := 2

/-- The number of candy bars purchased -/
def candy_bars : ℕ := 3

theorem gum_to_candy_ratio :
  ∃ (gum_cost : ℚ), 
    gum_cost * gum_packs + candy_cost * candy_bars = total_cost ∧
    gum_cost / candy_cost = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_gum_to_candy_ratio_l32_3241


namespace NUMINAMATH_CALUDE_ten_thousand_one_hundred_one_l32_3269

theorem ten_thousand_one_hundred_one (n : ℕ) : n = 10101 → n = 10000 + 100 + 1 := by
  sorry

end NUMINAMATH_CALUDE_ten_thousand_one_hundred_one_l32_3269


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l32_3201

-- Define the necessary structures and functions
structure Point where
  x : ℝ
  y : ℝ

def distance (p q : Point) : ℝ := sorry

def is_hyperbola (trajectory : Set Point) (F₁ F₂ : Point) : Prop := sorry

def is_constant (f : Point → ℝ) : Prop := sorry

-- State the theorem
theorem necessary_but_not_sufficient 
  (M : Point) (F₁ F₂ : Point) (trajectory : Set Point) :
  (∀ M ∈ trajectory, is_hyperbola trajectory F₁ F₂) →
    is_constant (λ M => |distance M F₁ - distance M F₂|) ∧
  ∃ trajectory' : Set Point, 
    is_constant (λ M => |distance M F₁ - distance M F₂|) ∧
    ¬(is_hyperbola trajectory' F₁ F₂) :=
by
  sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l32_3201


namespace NUMINAMATH_CALUDE_company_total_individuals_l32_3225

/-- Represents the hierarchical structure of a company -/
structure CompanyHierarchy where
  workers_per_team_lead : Nat
  team_leads_per_manager : Nat
  managers_per_supervisor : Nat

/-- Calculates the total number of individuals in the company given the hierarchy and number of supervisors -/
def total_individuals (h : CompanyHierarchy) (supervisors : Nat) : Nat :=
  let managers := supervisors * h.managers_per_supervisor
  let team_leads := managers * h.team_leads_per_manager
  let workers := team_leads * h.workers_per_team_lead
  workers + team_leads + managers + supervisors

/-- Theorem stating that given the specific hierarchy and 10 supervisors, the total number of individuals is 3260 -/
theorem company_total_individuals :
  let h : CompanyHierarchy := {
    workers_per_team_lead := 15,
    team_leads_per_manager := 4,
    managers_per_supervisor := 5
  }
  total_individuals h 10 = 3260 := by
  sorry

end NUMINAMATH_CALUDE_company_total_individuals_l32_3225


namespace NUMINAMATH_CALUDE_truck_driver_speed_l32_3227

/-- A truck driver's problem -/
theorem truck_driver_speed 
  (gas_cost : ℝ)
  (fuel_efficiency : ℝ)
  (pay_rate : ℝ)
  (total_pay : ℝ)
  (total_hours : ℝ)
  (h1 : gas_cost = 2)
  (h2 : fuel_efficiency = 10)
  (h3 : pay_rate = 0.5)
  (h4 : total_pay = 90)
  (h5 : total_hours = 10)
  : (total_pay / pay_rate) / total_hours = 18 := by
  sorry


end NUMINAMATH_CALUDE_truck_driver_speed_l32_3227


namespace NUMINAMATH_CALUDE_factorization_equality_l32_3207

theorem factorization_equality (x : ℝ) : x * (x + 2) - x - 2 = (x + 2) * (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l32_3207


namespace NUMINAMATH_CALUDE_sin_300_degrees_l32_3210

theorem sin_300_degrees : Real.sin (300 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_300_degrees_l32_3210


namespace NUMINAMATH_CALUDE_cats_and_fish_l32_3270

theorem cats_and_fish (c d : ℕ) : 
  (6 : ℕ) * (1 : ℕ) * (6 : ℕ) = (6 : ℕ) * (6 : ℕ) →  -- 6 cats eat 6 fish in 1 day
  c * d * (1 : ℕ) = (91 : ℕ) →                      -- c cats eat 91 fish in d days
  1 < c →                                           -- c is more than 1
  c < (10 : ℕ) →                                    -- c is less than 10
  c + d = (20 : ℕ) :=                               -- prove that c + d = 20
by sorry

end NUMINAMATH_CALUDE_cats_and_fish_l32_3270


namespace NUMINAMATH_CALUDE_min_questionnaires_correct_l32_3267

/-- The minimum number of questionnaires needed to achieve the desired responses -/
def min_questionnaires : ℕ := 513

/-- The number of desired responses -/
def desired_responses : ℕ := 750

/-- The initial response rate -/
def initial_rate : ℚ := 60 / 100

/-- The decline rate for follow-ups -/
def decline_rate : ℚ := 20 / 100

/-- Calculate the total responses given the number of questionnaires sent -/
def total_responses (n : ℕ) : ℚ :=
  n * initial_rate * (1 + (1 - decline_rate) + (1 - decline_rate)^2)

/-- Theorem stating that min_questionnaires is the minimum number needed -/
theorem min_questionnaires_correct :
  (total_responses min_questionnaires ≥ desired_responses) ∧
  (∀ m : ℕ, m < min_questionnaires → total_responses m < desired_responses) :=
by sorry


end NUMINAMATH_CALUDE_min_questionnaires_correct_l32_3267


namespace NUMINAMATH_CALUDE_linear_function_intersection_k_range_l32_3231

-- Define the linear function
def linear_function (k : ℝ) (x : ℝ) : ℝ := k * x + (2 - 2 * k)

-- Define the intersection function
def intersection_function (x : ℝ) : ℝ := -x + 3

-- Define the domain
def in_domain (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ 3

-- Theorem statement
theorem linear_function_intersection_k_range :
  ∀ k : ℝ, 
  (∃ x : ℝ, in_domain x ∧ linear_function k x = intersection_function x) →
  ((k ≤ -2 ∨ k ≥ -1/2) ∧ k ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_linear_function_intersection_k_range_l32_3231


namespace NUMINAMATH_CALUDE_shopkeeper_profit_margin_l32_3200

theorem shopkeeper_profit_margin
  (C : ℝ) -- Current cost
  (S : ℝ) -- Selling price
  (y : ℝ) -- Original profit margin percentage
  (h1 : S = C * (1 + 0.01 * y)) -- Current profit margin equation
  (h2 : S = 0.9 * C * (1 + 0.01 * (y + 15))) -- New profit margin equation
  : y = 35 := by
  sorry

end NUMINAMATH_CALUDE_shopkeeper_profit_margin_l32_3200


namespace NUMINAMATH_CALUDE_counterexample_exists_l32_3266

theorem counterexample_exists : ∃ n : ℕ, 
  (∃ p : ℕ, Prime p ∧ ∃ k : ℕ, n = p^k) ∧ 
  Prime (n - 2) ∧ 
  n = 25 := by
  sorry

end NUMINAMATH_CALUDE_counterexample_exists_l32_3266


namespace NUMINAMATH_CALUDE_mans_speed_in_still_water_l32_3289

/-- The speed of a man rowing in still water, given his downstream performance and the current speed. -/
theorem mans_speed_in_still_water (current_speed : ℝ) (distance : ℝ) (time : ℝ) : 
  current_speed = 3 →
  distance = 15 / 1000 →
  time = 2.9997600191984644 / 3600 →
  (distance / time) - current_speed = 15 := by
  sorry

end NUMINAMATH_CALUDE_mans_speed_in_still_water_l32_3289


namespace NUMINAMATH_CALUDE_total_chicken_pieces_l32_3226

-- Define the number of chicken pieces per order type
def chicken_pasta_pieces : ℕ := 2
def barbecue_chicken_pieces : ℕ := 3
def fried_chicken_dinner_pieces : ℕ := 8

-- Define the number of orders for each type
def fried_chicken_dinner_orders : ℕ := 2
def chicken_pasta_orders : ℕ := 6
def barbecue_chicken_orders : ℕ := 3

-- Theorem stating the total number of chicken pieces needed
theorem total_chicken_pieces :
  fried_chicken_dinner_orders * fried_chicken_dinner_pieces +
  chicken_pasta_orders * chicken_pasta_pieces +
  barbecue_chicken_orders * barbecue_chicken_pieces = 37 :=
by sorry

end NUMINAMATH_CALUDE_total_chicken_pieces_l32_3226


namespace NUMINAMATH_CALUDE_point_on_line_ratio_l32_3202

/-- Given six points O, A, B, C, D, E on a straight line in that order, with P between C and D,
    prove that OP = (ce - ad) / (a - c + e - d) when AP:PE = CP:PD -/
theorem point_on_line_ratio (a b c d e x : ℝ) 
  (h_order : 0 < a ∧ a < b ∧ b < c ∧ c < x ∧ x < d ∧ d < e) 
  (h_ratio : (a - x) / (x - e) = (c - x) / (x - d)) : 
  x = (c * e - a * d) / (a - c + e - d) := by
  sorry

end NUMINAMATH_CALUDE_point_on_line_ratio_l32_3202


namespace NUMINAMATH_CALUDE_line_slope_l32_3291

/-- Given a line l with equation y = (1/2)x + 1, its slope is 1/2 -/
theorem line_slope (l : Set (ℝ × ℝ)) (h : l = {(x, y) | y = (1/2) * x + 1}) :
  (∃ m : ℝ, ∀ (x₁ y₁ x₂ y₂ : ℝ), (x₁, y₁) ∈ l → (x₂, y₂) ∈ l → x₁ ≠ x₂ → 
    m = (y₂ - y₁) / (x₂ - x₁)) ∧ 
  (∀ m : ℝ, (∀ (x₁ y₁ x₂ y₂ : ℝ), (x₁, y₁) ∈ l → (x₂, y₂) ∈ l → x₁ ≠ x₂ → 
    m = (y₂ - y₁) / (x₂ - x₁)) → m = 1/2) :=
by sorry

end NUMINAMATH_CALUDE_line_slope_l32_3291
