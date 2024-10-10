import Mathlib

namespace geometric_arithmetic_progression_sum_l1337_133759

theorem geometric_arithmetic_progression_sum (b q : ℝ) (h1 : b > 0) (h2 : q > 0) :
  let a := b
  let d := (b * q^3 - b) / 3
  (∃ (n : ℕ), q^n = 2) →
  (3 * a + 10 * d = 148 / 9) →
  (b * (q^4 - 1) / (q - 1) = 700 / 27) :=
by sorry

end geometric_arithmetic_progression_sum_l1337_133759


namespace vasya_incorrect_l1337_133720

theorem vasya_incorrect : ¬∃ (x y : ℤ), (x + y = 2021) ∧ ((10 * x + y = 2221) ∨ (x + 10 * y = 2221)) := by
  sorry

end vasya_incorrect_l1337_133720


namespace line_equation_solution_l1337_133762

theorem line_equation_solution (a b : ℝ) (h_a : a ≠ 0) :
  (∀ x y : ℝ, y = a * x + b) →
  (4 = a * 0 + b) →
  (0 = a * (-3) + b) →
  (∀ x : ℝ, a * x + b = 0 ↔ x = -3) :=
by sorry

end line_equation_solution_l1337_133762


namespace sqrt_3_irrational_l1337_133780

theorem sqrt_3_irrational (numbers : Set ℝ) (h1 : numbers = {-1, 0, (1/2 : ℝ), Real.sqrt 3}) :
  ∃ x ∈ numbers, Irrational x ∧ ∀ y ∈ numbers, y ≠ x → ¬ Irrational y :=
by
  sorry

end sqrt_3_irrational_l1337_133780


namespace x_values_l1337_133756

theorem x_values (x : ℝ) : 
  ({1, 2} ∪ {x + 1, x^2 - 4*x + 6} : Set ℝ) = {1, 2, 3} → x = 2 ∨ x = 1 := by
  sorry

end x_values_l1337_133756


namespace decreasing_quadratic_implies_a_bound_l1337_133727

-- Define the function f(x) = -x^2 + 2ax + 3
def f (a : ℝ) (x : ℝ) : ℝ := -x^2 + 2*a*x + 3

-- Define what it means for a function to be decreasing on an interval
def decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f y < f x

-- State the theorem
theorem decreasing_quadratic_implies_a_bound :
  ∀ a : ℝ, decreasing_on (f a) 2 6 → a ≤ 2 := by
  sorry

end decreasing_quadratic_implies_a_bound_l1337_133727


namespace large_pizza_slices_correct_l1337_133729

/-- The number of slices a small pizza gives -/
def small_pizza_slices : ℕ := 4

/-- The number of small pizzas purchased -/
def small_pizzas_bought : ℕ := 3

/-- The number of large pizzas purchased -/
def large_pizzas_bought : ℕ := 2

/-- The number of slices George eats -/
def george_slices : ℕ := 3

/-- The number of slices Bob eats -/
def bob_slices : ℕ := george_slices + 1

/-- The number of slices Susie eats -/
def susie_slices : ℕ := bob_slices / 2

/-- The number of slices Bill, Fred, and Mark each eat -/
def others_slices : ℕ := 3

/-- The number of slices left over -/
def leftover_slices : ℕ := 10

/-- The number of slices a large pizza gives -/
def large_pizza_slices : ℕ := 8

theorem large_pizza_slices_correct : 
  small_pizza_slices * small_pizzas_bought + large_pizza_slices * large_pizzas_bought = 
  george_slices + bob_slices + susie_slices + 3 * others_slices + leftover_slices :=
by sorry

end large_pizza_slices_correct_l1337_133729


namespace max_equalization_value_l1337_133769

/-- Represents a 3x3 board with numbers --/
def Board := Matrix (Fin 3) (Fin 3) ℕ

/-- Checks if all elements in the board are equal --/
def all_equal (b : Board) : Prop :=
  ∀ i j k l, b i j = b k l

/-- Represents a valid operation on the board --/
inductive Operation
| row (i : Fin 3) (x : ℝ)
| col (j : Fin 3) (x : ℝ)

/-- Applies an operation to the board --/
def apply_operation (b : Board) (op : Operation) : Board :=
  sorry

/-- Checks if a board can be transformed to have all elements equal to m --/
def can_equalize (b : Board) (m : ℕ) : Prop :=
  ∃ (ops : List Operation), all_equal (ops.foldl apply_operation b) ∧
    ∀ i j, (ops.foldl apply_operation b) i j = m

/-- Initial board configuration --/
def initial_board : Board :=
  λ i j => i.val * 3 + j.val + 1

/-- Main theorem: The maximum value of m for which the board can be equalized is 4 --/
theorem max_equalization_value :
  (∀ m : ℕ, m > 4 → ¬ can_equalize initial_board m) ∧
  can_equalize initial_board 4 :=
sorry

end max_equalization_value_l1337_133769


namespace polynomial_simplification_l1337_133799

theorem polynomial_simplification (x : ℝ) :
  (3 * x^2 + 4 * x + 8) * (2 * x + 1) - (2 * x + 1) * (x^2 + 5 * x - 72) + (4 * x - 15) * (2 * x + 1) * (x + 6) =
  12 * x^3 + 22 * x^2 - 12 * x - 10 := by
  sorry

end polynomial_simplification_l1337_133799


namespace chandra_reading_pages_l1337_133778

/-- Represents the number of pages in the book -/
def total_pages : ℕ := 900

/-- Represents Chandra's reading speed in seconds per page -/
def chandra_speed : ℕ := 30

/-- Represents Daniel's reading speed in seconds per page -/
def daniel_speed : ℕ := 60

/-- Calculates the number of pages Chandra should read -/
def chandra_pages : ℕ := total_pages * daniel_speed / (chandra_speed + daniel_speed)

theorem chandra_reading_pages :
  chandra_pages = 600 ∧
  chandra_pages * chandra_speed = (total_pages - chandra_pages) * daniel_speed :=
by sorry

end chandra_reading_pages_l1337_133778


namespace simplify_and_evaluate_l1337_133784

theorem simplify_and_evaluate : 
  let f (x : ℝ) := (2*x + 4) / (x^2 - 6*x + 9) / ((2*x - 1) / (x - 3) - 1)
  f 0 = -2/3 := by sorry

end simplify_and_evaluate_l1337_133784


namespace total_paintable_area_is_1624_l1337_133777

/-- The number of bedrooms in Isabella's house -/
def num_bedrooms : ℕ := 4

/-- The length of each bedroom in feet -/
def bedroom_length : ℕ := 15

/-- The width of each bedroom in feet -/
def bedroom_width : ℕ := 12

/-- The height of each bedroom in feet -/
def bedroom_height : ℕ := 9

/-- The area occupied by doorways and windows in each bedroom in square feet -/
def unpaintable_area : ℕ := 80

/-- The total area of walls to be painted in square feet -/
def total_paintable_area : ℕ := 
  num_bedrooms * (
    2 * (bedroom_length * bedroom_height + bedroom_width * bedroom_height) - unpaintable_area
  )

theorem total_paintable_area_is_1624 : total_paintable_area = 1624 := by
  sorry

end total_paintable_area_is_1624_l1337_133777


namespace cos_thirteen_pi_thirds_l1337_133707

theorem cos_thirteen_pi_thirds : Real.cos (13 * Real.pi / 3) = 1 / 2 := by
  sorry

end cos_thirteen_pi_thirds_l1337_133707


namespace min_value_quadratic_form_l1337_133798

theorem min_value_quadratic_form :
  ∀ x y : ℝ, x^2 - x*y + y^2 ≥ 0 ∧ (x^2 - x*y + y^2 = 0 ↔ x = 0 ∧ y = 0) := by
  sorry

end min_value_quadratic_form_l1337_133798


namespace tangent_slope_x_squared_at_one_l1337_133766

theorem tangent_slope_x_squared_at_one : 
  let f : ℝ → ℝ := fun x ↦ x^2
  (deriv f) 1 = 2 := by sorry

end tangent_slope_x_squared_at_one_l1337_133766


namespace area_of_EFGH_l1337_133794

/-- Represents a parallelogram with a base and height -/
structure Parallelogram where
  base : ℝ
  height : ℝ

/-- Calculates the area of a parallelogram -/
def area (p : Parallelogram) : ℝ := p.base * p.height

/-- The specific parallelogram EFGH from the problem -/
def EFGH : Parallelogram := { base := 5, height := 3 }

/-- Theorem stating that the area of parallelogram EFGH is 15 square units -/
theorem area_of_EFGH : area EFGH = 15 := by sorry

end area_of_EFGH_l1337_133794


namespace f_min_value_l1337_133725

/-- The function to be minimized -/
def f (x y : ℝ) : ℝ := 2 * x^2 + 3 * x * y + 4 * y^2 - 8 * x + y

/-- The minimum value of the function f -/
def min_value : ℝ := 3.7391

theorem f_min_value :
  ∀ x y : ℝ, f x y ≥ min_value :=
sorry

end f_min_value_l1337_133725


namespace log_equation_solution_l1337_133746

-- Define the logarithm function
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- State the theorem
theorem log_equation_solution :
  ∀ x : ℝ, x > 0 → log 4 (x^3) + log (1/4) x = 12 → x = 4096 := by
  sorry

end log_equation_solution_l1337_133746


namespace complete_square_sum_l1337_133792

theorem complete_square_sum (x : ℝ) : ∃ (a b c : ℤ), 
  (64 * x^2 + 96 * x - 81 = 0) ∧ 
  (a > 0) ∧
  ((a : ℝ) * x + b)^2 = c ∧
  a + b + c = 131 := by
  sorry

end complete_square_sum_l1337_133792


namespace transformations_correctness_l1337_133753

-- Define the transformations
def transformation_A (a b c : ℝ) : Prop := (c ≠ 0) → (a * c) / (b * c) = a / b

def transformation_B (a b : ℝ) : Prop := (a + b ≠ 0) → (-a - b) / (a + b) = -1

def transformation_C (m n : ℝ) : Prop := 
  (0.2 * m - 0.3 * n ≠ 0) → (0.5 * m + n) / (0.2 * m - 0.3 * n) = (5 * m + 10 * n) / (2 * m - 3 * n)

def transformation_D (x : ℝ) : Prop := (x + 1 ≠ 0) → (2 - x) / (x + 1) = (x - 2) / (1 + x)

-- Theorem stating which transformations are correct and which is incorrect
theorem transformations_correctness :
  (∀ a b c, transformation_A a b c) ∧
  (∀ a b, transformation_B a b) ∧
  (∀ m n, transformation_C m n) ∧
  ¬(∀ x, transformation_D x) := by
  sorry

end transformations_correctness_l1337_133753


namespace sachins_age_l1337_133757

theorem sachins_age (sachin_age rahul_age : ℝ) 
  (age_difference : rahul_age = sachin_age + 7)
  (age_ratio : sachin_age / rahul_age = 7 / 9) :
  sachin_age = 24.5 := by
  sorry

end sachins_age_l1337_133757


namespace credit_card_balance_proof_l1337_133761

/-- Calculates the new credit card balance after transactions -/
def new_balance (initial_balance groceries_charge towels_return : ℚ) : ℚ :=
  initial_balance + groceries_charge + (groceries_charge / 2) - towels_return

/-- Proves that the new balance is correct given the transactions -/
theorem credit_card_balance_proof :
  new_balance 126 60 45 = 171 := by
  sorry

end credit_card_balance_proof_l1337_133761


namespace keyboard_cost_l1337_133710

/-- Given the total cost of keyboards and printers, and the cost of a single printer,
    calculate the cost of a single keyboard. -/
theorem keyboard_cost (total_cost printer_cost : ℕ) : 
  total_cost = 2050 →
  printer_cost = 70 →
  ∃ (keyboard_cost : ℕ), 
    keyboard_cost * 15 + printer_cost * 25 = total_cost ∧ 
    keyboard_cost = 20 := by
  sorry

end keyboard_cost_l1337_133710


namespace sector_area_l1337_133724

theorem sector_area (r : ℝ) (θ : ℝ) (h1 : r = 10) (h2 : θ = 42) :
  (θ / 360) * π * r^2 = 35 * π / 3 := by
  sorry

end sector_area_l1337_133724


namespace union_A_B_intersection_A_complement_B_l1337_133776

-- Define the universal set U
def U : Set ℝ := {x | x ≤ 4}

-- Define set A
def A : Set ℝ := {x | -2 < x ∧ x < 3}

-- Define set B
def B : Set ℝ := {x | -3 ≤ x ∧ x ≤ 2}

-- Theorem for A ∪ B
theorem union_A_B : A ∪ B = {x | -3 ≤ x ∧ x < 3} := by sorry

-- Theorem for A ∩ (∁U B)
theorem intersection_A_complement_B : A ∩ (U \ B) = {x | 2 < x ∧ x < 3} := by sorry

end union_A_B_intersection_A_complement_B_l1337_133776


namespace card_sum_perfect_square_l1337_133793

theorem card_sum_perfect_square (n : ℕ) (h : n ≥ 100) :
  ∃ a b c : ℕ, n ≤ a ∧ a < b ∧ b < c ∧ c ≤ 2*n ∧
  ∃ x y z : ℕ, a + b = x^2 ∧ b + c = y^2 ∧ c + a = z^2 :=
by sorry

end card_sum_perfect_square_l1337_133793


namespace saheed_kayla_earnings_ratio_l1337_133731

/-- Proves that the ratio of Saheed's earnings to Kayla's earnings is 4:1 -/
theorem saheed_kayla_earnings_ratio :
  let vika_earnings : ℕ := 84
  let kayla_earnings : ℕ := vika_earnings - 30
  let saheed_earnings : ℕ := 216
  (saheed_earnings : ℚ) / kayla_earnings = 4 := by
  sorry

end saheed_kayla_earnings_ratio_l1337_133731


namespace circle_equation_to_standard_form_1_circle_equation_to_standard_form_2_l1337_133719

/-- Proves that the given circle equation is equivalent to its standard form -/
theorem circle_equation_to_standard_form_1 (x y : ℝ) :
  x^2 + y^2 - 4*x + 6*y - 3 = 0 ↔ (x - 2)^2 + (y + 3)^2 = 16 := by
  sorry

/-- Proves that the given circle equation is equivalent to its standard form -/
theorem circle_equation_to_standard_form_2 (x y : ℝ) :
  4*x^2 + 4*y^2 - 8*x + 4*y - 11 = 0 ↔ (x - 1)^2 + (y + 1/2)^2 = 4 := by
  sorry

end circle_equation_to_standard_form_1_circle_equation_to_standard_form_2_l1337_133719


namespace inverse_proportion_ratio_l1337_133701

theorem inverse_proportion_ratio (x₁ x₂ y₁ y₂ : ℝ) (hx₁ : x₁ ≠ 0) (hx₂ : x₂ ≠ 0) 
  (hy₁ : y₁ ≠ 0) (hy₂ : y₂ ≠ 0) (h_inv : ∃ k, k ≠ 0 ∧ ∀ x y, x * y = k) 
  (h_ratio : x₁ / x₂ = 3 / 5) : 
  y₁ / y₂ = 5 / 3 := by
sorry

end inverse_proportion_ratio_l1337_133701


namespace shopping_money_calculation_l1337_133733

theorem shopping_money_calculation (initial_amount : ℝ) : 
  (0.7 * initial_amount = 350) → initial_amount = 500 := by
  sorry

end shopping_money_calculation_l1337_133733


namespace tunnel_length_is_1200_l1337_133786

/-- Calculates the length of a tunnel given train specifications and crossing times. -/
def tunnel_length (train_length platform_length : ℝ) 
                  (tunnel_time platform_time : ℝ) : ℝ :=
  3 * (train_length + platform_length) - train_length

/-- Proves that the tunnel length is 1200 meters given the specified conditions. -/
theorem tunnel_length_is_1200 :
  tunnel_length 330 180 45 15 = 1200 := by
  sorry

#eval tunnel_length 330 180 45 15

end tunnel_length_is_1200_l1337_133786


namespace prime_representation_l1337_133722

theorem prime_representation (k : ℕ) (h : k ∈ Finset.range 7 \ {0}) :
  (∀ p : ℕ, Prime p → 
    (∃ a b : ℕ, a > 0 ∧ b > 0 ∧ p^2 = a^2 + k*b^2) → 
    (∃ x y : ℕ, x > 0 ∧ y > 0 ∧ p = x^2 + k*y^2)) ↔ 
  k ∈ ({1, 2, 4} : Finset ℕ) :=
sorry

end prime_representation_l1337_133722


namespace distance_product_theorem_l1337_133795

theorem distance_product_theorem (a b : ℝ) (θ : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) :
  let point1 := (Real.sqrt (a^2 - b^2), 0)
  let point2 := (-Real.sqrt (a^2 - b^2), 0)
  let line := fun (x y : ℝ) ↦ x * Real.cos θ / a + y * Real.sin θ / b = 1
  let distance (p : ℝ × ℝ) := 
    abs (b * Real.cos θ * p.1 + a * Real.sin θ * p.2 - a * b) / 
    Real.sqrt ((b * Real.cos θ)^2 + (a * Real.sin θ)^2)
  (distance point1) * (distance point2) = b^2 := by
sorry

end distance_product_theorem_l1337_133795


namespace popped_kernel_probability_l1337_133758

theorem popped_kernel_probability (total : ℝ) (white : ℝ) (yellow : ℝ) 
  (white_pop_rate : ℝ) (yellow_pop_rate : ℝ) :
  white / total = 3 / 4 →
  yellow / total = 1 / 4 →
  white_pop_rate = 2 / 5 →
  yellow_pop_rate = 3 / 4 →
  (white * white_pop_rate) / ((white * white_pop_rate) + (yellow * yellow_pop_rate)) = 24 / 39 := by
  sorry

end popped_kernel_probability_l1337_133758


namespace angle_AOB_measure_l1337_133716

/-- A configuration of rectangles with specific properties -/
structure RectangleConfiguration where
  /-- The number of equal rectangles -/
  num_rectangles : ℕ
  /-- Assertion that one side of each rectangle is twice the other -/
  side_ratio : Prop
  /-- Assertion that points C, O, and B are collinear -/
  collinear_COB : Prop
  /-- Assertion that triangle ACO is right-angled and isosceles -/
  triangle_ACO_properties : Prop

/-- Theorem stating that given the specific configuration, angle AOB measures 135° -/
theorem angle_AOB_measure (config : RectangleConfiguration) 
  (h1 : config.num_rectangles = 5)
  (h2 : config.side_ratio)
  (h3 : config.collinear_COB)
  (h4 : config.triangle_ACO_properties) :
  ∃ (angle_AOB : ℝ), angle_AOB = 135 := by
  sorry

end angle_AOB_measure_l1337_133716


namespace min_holiday_days_l1337_133739

/-- Represents a day during the holiday -/
structure Day where
  morning_sunny : Bool
  afternoon_sunny : Bool

/-- Conditions for the holiday weather -/
def valid_holiday (days : List Day) : Prop :=
  let total_days := days.length
  let rainy_days := days.filter (fun d => ¬d.morning_sunny ∨ ¬d.afternoon_sunny)
  let sunny_afternoons := days.filter (fun d => d.afternoon_sunny)
  let sunny_mornings := days.filter (fun d => d.morning_sunny)
  rainy_days.length = 7 ∧
  days.all (fun d => ¬d.afternoon_sunny → d.morning_sunny) ∧
  sunny_afternoons.length = 5 ∧
  sunny_mornings.length = 6

/-- The theorem to be proved -/
theorem min_holiday_days :
  ∃ (days : List Day), valid_holiday days ∧
    ∀ (other_days : List Day), valid_holiday other_days → days.length ≤ other_days.length :=
by
  sorry

end min_holiday_days_l1337_133739


namespace stratified_sampling_theorem_l1337_133726

theorem stratified_sampling_theorem (total_population : ℕ) (sample_size : ℕ) (stratum_size : ℕ) 
  (h1 : total_population = 500) 
  (h2 : sample_size = 100) 
  (h3 : stratum_size = 95) :
  (stratum_size : ℚ) / total_population * sample_size = 19 := by
  sorry

end stratified_sampling_theorem_l1337_133726


namespace final_position_l1337_133700

/-- A point in 2D Cartesian coordinates -/
structure Point where
  x : ℝ
  y : ℝ

/-- Translate a point horizontally -/
def translateRight (p : Point) (units : ℝ) : Point :=
  { x := p.x + units, y := p.y }

/-- Reflect a point about the origin -/
def reflectOrigin (p : Point) : Point :=
  { x := -p.x, y := -p.y }

/-- The theorem stating the final position of the point after translation and reflection -/
theorem final_position :
  let initial := Point.mk 3 2
  let translated := translateRight initial 2
  let final := reflectOrigin translated
  final = Point.mk (-5) (-2) := by sorry

end final_position_l1337_133700


namespace complement_of_M_in_U_l1337_133735

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def M : Set ℕ := {2, 4, 6}

theorem complement_of_M_in_U : 
  (U \ M) = {1, 3, 5} := by sorry

end complement_of_M_in_U_l1337_133735


namespace largest_prime_divisor_exists_l1337_133736

def base_5_number : ℕ := 2031357

theorem largest_prime_divisor_exists :
  ∃ p : ℕ, Prime p ∧ p ∣ base_5_number ∧ ∀ q : ℕ, Prime q → q ∣ base_5_number → q ≤ p :=
sorry

end largest_prime_divisor_exists_l1337_133736


namespace hundredth_term_of_sequence_l1337_133723

def arithmeticSequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  a₁ + (n - 1) * d

theorem hundredth_term_of_sequence (a₁ d : ℕ) (h₁ : a₁ = 5) (h₂ : d = 4) :
  arithmeticSequence a₁ d 100 = 401 := by
  sorry

end hundredth_term_of_sequence_l1337_133723


namespace divisibility_by_24_l1337_133708

theorem divisibility_by_24 (p : ℕ) (h_prime : Nat.Prime p) (h_ge_5 : p ≥ 5) :
  24 ∣ (p^2 - 1) := by
  sorry

end divisibility_by_24_l1337_133708


namespace sin_210_degrees_l1337_133749

theorem sin_210_degrees : Real.sin (210 * π / 180) = -1/2 := by
  sorry

end sin_210_degrees_l1337_133749


namespace denarii_puzzle_l1337_133797

theorem denarii_puzzle (x y : ℚ) : 
  (x + 7 = 5 * (y - 7)) →
  (y + 5 = 7 * (x - 5)) →
  (x = 11 + 9 / 17 ∧ y = 9 + 14 / 17) :=
by sorry

end denarii_puzzle_l1337_133797


namespace fifth_term_value_l1337_133751

def geometric_sequence (a : ℕ → ℚ) : Prop :=
  a 1 = 1 ∧ ∀ n : ℕ, 2 * a (n + 1) = a n

theorem fifth_term_value (a : ℕ → ℚ) (h : geometric_sequence a) : a 5 = 1 / 16 := by
  sorry

end fifth_term_value_l1337_133751


namespace string_folding_l1337_133750

theorem string_folding (initial_length : ℝ) (folded_twice : ℕ) : 
  initial_length = 12 ∧ folded_twice = 2 → initial_length / (2^folded_twice) = 3 := by
  sorry

end string_folding_l1337_133750


namespace cubic_polynomial_existence_l1337_133718

theorem cubic_polynomial_existence (a b c : ℕ) (hab : a ≠ b) (hbc : b ≠ c) (hac : a ≠ c) :
  ∃ f : ℤ → ℤ, 
    (∃ p q r : ℤ, ∀ x, f x = p * x^3 + q * x^2 + r * x + (a * b * c)) ∧ 
    (p > 0) ∧
    (f a = a^3) ∧ (f b = b^3) ∧ (f c = c^3) :=
sorry

end cubic_polynomial_existence_l1337_133718


namespace complex_division_equivalence_l1337_133704

theorem complex_division_equivalence : Complex.I * (4 - 3 * Complex.I) = 3 - 4 * Complex.I := by
  sorry

end complex_division_equivalence_l1337_133704


namespace fencing_cost_9m_square_l1337_133789

/-- Cost of fencing for each side of a square -/
structure FencingCost where
  first : ℕ
  second : ℕ
  third : ℕ
  fourth : ℕ

/-- Calculate the total cost of fencing a square -/
def totalCost (cost : FencingCost) (sideLength : ℕ) : ℕ :=
  (cost.first + cost.second + cost.third + cost.fourth) * sideLength

/-- The fencing costs for the problem -/
def givenCost : FencingCost :=
  { first := 79
    second := 92
    third := 85
    fourth := 96 }

/-- Theorem: The total cost of fencing the square with side length 9 meters is $3168 -/
theorem fencing_cost_9m_square (cost : FencingCost := givenCost) :
  totalCost cost 9 = 3168 := by
  sorry

#eval totalCost givenCost 9

end fencing_cost_9m_square_l1337_133789


namespace sum_of_numbers_l1337_133772

theorem sum_of_numbers : 2 * 2143 + 4321 + 3214 + 1432 = 13523 := by
  sorry

end sum_of_numbers_l1337_133772


namespace greatest_integer_less_than_negative_21_over_5_l1337_133782

theorem greatest_integer_less_than_negative_21_over_5 :
  Int.floor (-21 / 5 : ℚ) = -5 := by
  sorry

end greatest_integer_less_than_negative_21_over_5_l1337_133782


namespace no_rational_solutions_for_quadratic_l1337_133711

theorem no_rational_solutions_for_quadratic (k : ℕ+) : 
  ¬∃ (x : ℚ), k * x^2 + 18 * x + 3 * k = 0 := by
sorry

end no_rational_solutions_for_quadratic_l1337_133711


namespace fourth_root_equation_solution_l1337_133703

theorem fourth_root_equation_solution :
  ∀ x : ℝ, (x > 0 ∧ x^(1/4) = 16 / (8 - x^(1/4))) ↔ x = 256 := by
  sorry

end fourth_root_equation_solution_l1337_133703


namespace triangle_sine_sum_inequality_l1337_133752

theorem triangle_sine_sum_inequality (A B C : Real) : 
  A + B + C = Real.pi → 0 < A → 0 < B → 0 < C →
  Real.sin A + Real.sin B + Real.sin C ≤ (3 / 2) * Real.sqrt 3 := by
  sorry

end triangle_sine_sum_inequality_l1337_133752


namespace chromium_percentage_calculation_l1337_133705

/-- Percentage of chromium in the first alloy -/
def chromium_percentage_1 : ℝ := 12

/-- Mass of the first alloy in kg -/
def mass_1 : ℝ := 15

/-- Mass of the second alloy in kg -/
def mass_2 : ℝ := 35

/-- Percentage of chromium in the new alloy -/
def chromium_percentage_new : ℝ := 10.6

/-- Percentage of chromium in the second alloy -/
def chromium_percentage_2 : ℝ := 10

theorem chromium_percentage_calculation :
  (chromium_percentage_1 / 100 * mass_1 + chromium_percentage_2 / 100 * mass_2) / (mass_1 + mass_2) * 100 = chromium_percentage_new :=
sorry

end chromium_percentage_calculation_l1337_133705


namespace total_apples_is_36_l1337_133787

/-- The number of apples picked by Mike -/
def mike_apples : ℕ := 7

/-- The number of apples picked by Nancy -/
def nancy_apples : ℕ := 3

/-- The number of apples picked by Keith -/
def keith_apples : ℕ := 6

/-- The number of apples picked by Olivia -/
def olivia_apples : ℕ := 12

/-- The number of apples picked by Thomas -/
def thomas_apples : ℕ := 8

/-- The total number of apples picked -/
def total_apples : ℕ := mike_apples + nancy_apples + keith_apples + olivia_apples + thomas_apples

theorem total_apples_is_36 : total_apples = 36 := by
  sorry

end total_apples_is_36_l1337_133787


namespace simplify_fraction_l1337_133747

theorem simplify_fraction : (123 : ℚ) / 999 * 27 = 123 / 37 := by
  sorry

end simplify_fraction_l1337_133747


namespace circumscribed_sphere_surface_area_is_77pi_l1337_133738

/-- Represents a triangular pyramid with vertices P, A, B, C -/
structure TriangularPyramid where
  PA : ℝ
  BC : ℝ
  AC : ℝ
  BP : ℝ
  CP : ℝ
  AB : ℝ

/-- The surface area of the circumscribed sphere of a triangular pyramid -/
def circumscribedSphereSurfaceArea (t : TriangularPyramid) : ℝ :=
  sorry

/-- Theorem: The surface area of the circumscribed sphere of the given triangular pyramid is 77π -/
theorem circumscribed_sphere_surface_area_is_77pi :
  let t : TriangularPyramid := {
    PA := 2 * Real.sqrt 13,
    BC := 2 * Real.sqrt 13,
    AC := Real.sqrt 41,
    BP := Real.sqrt 41,
    CP := Real.sqrt 61,
    AB := Real.sqrt 61
  }
  circumscribedSphereSurfaceArea t = 77 * Real.pi := by
  sorry

end circumscribed_sphere_surface_area_is_77pi_l1337_133738


namespace b_investment_is_7200_l1337_133730

/-- Represents the investment and profit distribution in a partnership business. -/
structure PartnershipBusiness where
  a_investment : ℕ
  b_investment : ℕ
  c_investment : ℕ
  total_profit : ℕ
  a_profit_share : ℕ

/-- The investment of partner B in the business. -/
def b_investment (pb : PartnershipBusiness) : ℕ :=
  7200

/-- Theorem stating that B's investment is 7200, given the conditions of the problem. -/
theorem b_investment_is_7200 (pb : PartnershipBusiness) 
    (h1 : pb.a_investment = 2400)
    (h2 : pb.c_investment = 9600)
    (h3 : pb.total_profit = 9000)
    (h4 : pb.a_profit_share = 1125) :
  b_investment pb = 7200 := by
  sorry

end b_investment_is_7200_l1337_133730


namespace maxwell_brad_meeting_l1337_133734

/-- The distance between Maxwell's and Brad's homes in kilometers -/
def total_distance : ℝ := 36

/-- Maxwell's walking speed in km/h -/
def maxwell_speed : ℝ := 3

/-- Brad's running speed in km/h -/
def brad_speed : ℝ := 6

/-- The distance traveled by Maxwell when they meet -/
def maxwell_distance : ℝ := 12

theorem maxwell_brad_meeting :
  maxwell_distance * brad_speed = (total_distance - maxwell_distance) * maxwell_speed :=
by sorry

end maxwell_brad_meeting_l1337_133734


namespace ratio_problem_l1337_133773

/-- Given two ratios a:b:c and c:d:e, prove that a:e is 3:10 -/
theorem ratio_problem (a b c d e : ℚ) 
  (h1 : a / b = 2 / 3 ∧ b / c = 3 / 4)
  (h2 : c / d = 3 / 4 ∧ d / e = 4 / 5) :
  a / e = 3 / 10 := by
  sorry

end ratio_problem_l1337_133773


namespace knocks_to_knicks_conversion_l1337_133755

/-- Conversion rate between knicks and knacks -/
def knicks_to_knacks : ℚ := 3 / 8

/-- Conversion rate between knacks and knocks -/
def knacks_to_knocks : ℚ := 6 / 5

/-- The number of knocks we want to convert -/
def target_knocks : ℚ := 30

theorem knocks_to_knicks_conversion :
  target_knocks * knacks_to_knocks⁻¹ * knicks_to_knacks⁻¹ = 200 / 3 :=
sorry

end knocks_to_knicks_conversion_l1337_133755


namespace complex_number_in_fourth_quadrant_l1337_133783

theorem complex_number_in_fourth_quadrant (m : ℝ) (h : 1 < m ∧ m < 2) :
  let z : ℂ := Complex.mk (m - 1) (m - 2)
  0 < z.re ∧ z.im < 0 :=
by sorry

end complex_number_in_fourth_quadrant_l1337_133783


namespace negation_of_implication_l1337_133728

theorem negation_of_implication (a b : ℝ) :
  ¬(a > b → 2*a > 2*b) ↔ (a ≤ b → 2*a ≤ 2*b) :=
by sorry

end negation_of_implication_l1337_133728


namespace nuts_problem_l1337_133717

/-- The number of nuts after one day's operation -/
def nuts_after_day (n : ℕ) : ℕ := 
  if 2 * n > 8 then 2 * n - 8 else 0

/-- The number of nuts after d days, starting with n nuts -/
def nuts_after_days (n : ℕ) (d : ℕ) : ℕ :=
  match d with
  | 0 => n
  | d + 1 => nuts_after_day (nuts_after_days n d)

theorem nuts_problem :
  nuts_after_days 7 4 = 0 := by
  sorry

end nuts_problem_l1337_133717


namespace shaded_area_square_minus_circles_l1337_133721

/-- The shaded area of a square with side length 10 and four circles of radius 3√2 at its vertices -/
theorem shaded_area_square_minus_circles :
  let square_side : ℝ := 10
  let circle_radius : ℝ := 3 * Real.sqrt 2
  let square_area : ℝ := square_side ^ 2
  let circle_area : ℝ := π * circle_radius ^ 2
  let total_circles_area : ℝ := 4 * circle_area
  let shaded_area : ℝ := square_area - total_circles_area
  shaded_area = 100 - 72 * π := by
  sorry

#check shaded_area_square_minus_circles

end shaded_area_square_minus_circles_l1337_133721


namespace min_value_expression_l1337_133788

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hsum : a + b + c = 9) :
  (a^2 + b^2 + c^2)/(a + b + c) + (b^2 + c^2)/(b + c) + (c^2 + a^2)/(c + a) + (a^2 + b^2)/(a + b) ≥ 12 := by
  sorry

end min_value_expression_l1337_133788


namespace chapter_page_difference_l1337_133764

theorem chapter_page_difference (first_chapter_pages second_chapter_pages : ℕ) 
  (h1 : first_chapter_pages = 48) 
  (h2 : second_chapter_pages = 11) : 
  first_chapter_pages - second_chapter_pages = 37 := by
  sorry

end chapter_page_difference_l1337_133764


namespace unique_base_for_625_l1337_133771

def is_four_digit (n : ℕ) (b : ℕ) : Prop :=
  b ^ 3 ≤ n ∧ n < b ^ 4

def last_two_digits_odd (n : ℕ) (b : ℕ) : Prop :=
  ∃ d₁ d₂ d₃ d₄ : ℕ, 
    n = d₁ * b^3 + d₂ * b^2 + d₃ * b^1 + d₄ * b^0 ∧
    d₃ % 2 = 1 ∧ d₄ % 2 = 1

theorem unique_base_for_625 :
  ∃! b : ℕ, b > 1 ∧ is_four_digit 625 b ∧ last_two_digits_odd 625 b :=
sorry

end unique_base_for_625_l1337_133771


namespace happy_equation_properties_l1337_133741

def happy_number (a b c : ℤ) : ℚ :=
  (4 * a * c - b^2) / (4 * a)

def happy_numbers_to_each_other (a b c p q r : ℤ) : Prop :=
  |r * happy_number a b c - c * happy_number p q r| = 0

theorem happy_equation_properties :
  ∀ (a b c m n p q r : ℤ),
  (a ≠ 0 ∧ p ≠ 0) →
  (∃ (x y : ℤ), a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0 ∧ x ≠ y) →
  (∃ (x y : ℤ), p * x^2 + q * x + r = 0 ∧ p * y^2 + q * y + r = 0 ∧ x ≠ y) →
  (happy_number 1 (-2) (-3) = -4) ∧
  (1 < m ∧ m < 6 ∧ 
   ∃ (x y : ℤ), x^2 - (2*m-1)*x + (m^2-2*m-3) = 0 ∧ 
                y^2 - (2*m-1)*y + (m^2-2*m-3) = 0 ∧ 
                x ≠ y →
   m = 3 ∧ happy_number 1 (-5) 0 = -25/4) ∧
  (∃ (x1 y1 x2 y2 : ℤ),
    x1^2 - m*x1 + (m+1) = 0 ∧ y1^2 - m*y1 + (m+1) = 0 ∧ x1 ≠ y1 ∧
    x2^2 - (n+2)*x2 + 2*n = 0 ∧ y2^2 - (n+2)*y2 + 2*n = 0 ∧ x2 ≠ y2 ∧
    happy_numbers_to_each_other 1 (-m) (m+1) 1 (-(n+2)) (2*n) →
    n = 0 ∨ n = 3) := by
  sorry

end happy_equation_properties_l1337_133741


namespace sum_of_c_values_l1337_133763

theorem sum_of_c_values : ∃ (S : Finset ℤ),
  (∀ c ∈ S, c ≤ 30 ∧ 
    ∃ x y : ℚ, y = x^2 - 8*x - c ∧ 
    ∃ k : ℤ, (64 + 4*c = k^2)) ∧
  (∀ c : ℤ, c ≤ 30 → 
    (∃ x y : ℚ, y = x^2 - 8*x - c ∧ 
    ∃ k : ℤ, (64 + 4*c = k^2)) → 
    c ∈ S) ∧
  S.sum id = -11 :=
sorry

end sum_of_c_values_l1337_133763


namespace total_spent_is_correct_l1337_133702

-- Define the value of a penny in dollars
def penny_value : ℚ := 1 / 100

-- Define the value of a dime in dollars
def dime_value : ℚ := 1 / 10

-- Define the number of pennies spent on ice cream
def ice_cream_pennies : ℕ := 2

-- Define the number of dimes spent on baseball cards
def baseball_cards_dimes : ℕ := 12

-- Theorem statement
theorem total_spent_is_correct :
  (ice_cream_pennies : ℚ) * penny_value + (baseball_cards_dimes : ℚ) * dime_value = 122 / 100 := by
  sorry

end total_spent_is_correct_l1337_133702


namespace range_of_a_l1337_133785

theorem range_of_a (a b c : ℝ) 
  (eq1 : a^2 - b*c - 8*a + 7 = 0)
  (eq2 : b^2 + c^2 + b*c - 6*a + 6 = 0) :
  1 ≤ a ∧ a ≤ 9 := by sorry

end range_of_a_l1337_133785


namespace james_sticker_cost_l1337_133781

/-- Calculates James's share of the cost for stickers --/
theorem james_sticker_cost (packs : ℕ) (stickers_per_pack : ℕ) (cost_per_sticker : ℚ) : 
  packs = 4 → 
  stickers_per_pack = 30 → 
  cost_per_sticker = 1/10 →
  (packs * stickers_per_pack * cost_per_sticker) / 2 = 6 := by
  sorry

#check james_sticker_cost

end james_sticker_cost_l1337_133781


namespace decoration_price_increase_l1337_133744

def price_1990 : ℝ := 11500
def increase_1990_to_1996 : ℝ := 0.13
def increase_1996_to_2001 : ℝ := 0.20

def price_2001 : ℝ :=
  price_1990 * (1 + increase_1990_to_1996) * (1 + increase_1996_to_2001)

theorem decoration_price_increase : price_2001 = 15594 := by
  sorry

end decoration_price_increase_l1337_133744


namespace hyperbola_equation_l1337_133790

/-- A hyperbola is defined by its equation and properties --/
structure Hyperbola where
  a : ℝ
  b : ℝ
  eqn : (x y : ℝ) → x^2 / a^2 - y^2 / b^2 = 1
  a_pos : a > 0
  b_pos : b > 0
  imaginary_axis : b = 1
  asymptote : (x : ℝ) → x / 2 = a / b

/-- The theorem states that a hyperbola with given properties has a specific equation --/
theorem hyperbola_equation (h : Hyperbola) : 
  ∀ x y : ℝ, x^2 / 4 - y^2 = 1 := by sorry

end hyperbola_equation_l1337_133790


namespace solution_set_implies_a_greater_than_negative_one_l1337_133754

theorem solution_set_implies_a_greater_than_negative_one (a : ℝ) :
  (∀ x : ℝ, x * (x - a + 1) > a ↔ (x < -1 ∨ x > a)) →
  a > -1 := by
sorry

end solution_set_implies_a_greater_than_negative_one_l1337_133754


namespace arithmetic_sequence_ratio_l1337_133713

/-- An arithmetic sequence with sum S_n of first n terms -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  d : ℝ
  sum : ℕ → ℝ
  sum_def : ∀ n, sum n = n * (2 * a 1 + (n - 1) * d) / 2
  term_def : ∀ n, a n = a 1 + (n - 1) * d

/-- Theorem: For an arithmetic sequence where S_5 = 3(a_2 + a_8), a_5 / a_3 = 5/6 -/
theorem arithmetic_sequence_ratio 
  (seq : ArithmeticSequence) 
  (h : seq.sum 5 = 3 * (seq.a 2 + seq.a 8)) :
  seq.a 5 / seq.a 3 = 5 / 6 := by
  sorry

end arithmetic_sequence_ratio_l1337_133713


namespace max_triangle_area_l1337_133714

/-- The ellipse E defined by x²/13 + y²/4 = 1 -/
def Ellipse : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 / 13 + p.2^2 / 4 = 1}

/-- The left focus F₁ of the ellipse -/
def F₁ : ℝ × ℝ := sorry

/-- The right focus F₂ of the ellipse -/
def F₂ : ℝ × ℝ := sorry

/-- A point P on the ellipse, not coinciding with left and right vertices -/
def P : ℝ × ℝ := sorry

/-- The area of triangle F₂PF₁ -/
def triangleArea (p : ℝ × ℝ) : ℝ := sorry

/-- The theorem stating that the maximum area of triangle F₂PF₁ is 6 -/
theorem max_triangle_area :
  ∀ p ∈ Ellipse, p ≠ F₁ ∧ p ≠ F₂ → triangleArea p ≤ 6 ∧ ∃ q ∈ Ellipse, triangleArea q = 6 :=
sorry

end max_triangle_area_l1337_133714


namespace max_profit_is_21000_l1337_133791

/-- Represents the production capabilities and constraints of a furniture factory -/
structure FurnitureFactory where
  carpenterHoursChair : ℕ
  carpenterHoursDesk : ℕ
  maxCarpenterHours : ℕ
  painterHoursChair : ℕ
  painterHoursDesk : ℕ
  maxPainterHours : ℕ
  profitChair : ℕ
  profitDesk : ℕ

/-- Calculates the profit for a given production plan -/
def calculateProfit (factory : FurnitureFactory) (chairs : ℕ) (desks : ℕ) : ℕ :=
  chairs * factory.profitChair + desks * factory.profitDesk

/-- Checks if a production plan is feasible given the factory's constraints -/
def isFeasible (factory : FurnitureFactory) (chairs : ℕ) (desks : ℕ) : Prop :=
  chairs * factory.carpenterHoursChair + desks * factory.carpenterHoursDesk ≤ factory.maxCarpenterHours ∧
  chairs * factory.painterHoursChair + desks * factory.painterHoursDesk ≤ factory.maxPainterHours

/-- Theorem stating that the maximum profit is 21000 yuan -/
theorem max_profit_is_21000 (factory : FurnitureFactory) 
  (h1 : factory.carpenterHoursChair = 4)
  (h2 : factory.carpenterHoursDesk = 8)
  (h3 : factory.maxCarpenterHours = 8000)
  (h4 : factory.painterHoursChair = 2)
  (h5 : factory.painterHoursDesk = 1)
  (h6 : factory.maxPainterHours = 1300)
  (h7 : factory.profitChair = 15)
  (h8 : factory.profitDesk = 20) :
  (∀ chairs desks, isFeasible factory chairs desks → calculateProfit factory chairs desks ≤ 21000) ∧
  (∃ chairs desks, isFeasible factory chairs desks ∧ calculateProfit factory chairs desks = 21000) :=
sorry

end max_profit_is_21000_l1337_133791


namespace trailing_zeros_mod_500_l1337_133748

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def trailingZeros (n : ℕ) : ℕ :=
  (List.range 51).map factorial
    |> List.foldl (·*·) 1
    |> Nat.digits 10
    |> List.reverse
    |> List.takeWhile (·==0)
    |> List.length

theorem trailing_zeros_mod_500 :
  trailingZeros 50 % 500 = 12 := by sorry

end trailing_zeros_mod_500_l1337_133748


namespace cubic_sum_inequality_l1337_133715

theorem cubic_sum_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x^3 / (x + y)) + (y^3 / (y + z)) + (z^3 / (z + x)) ≥ (x*y + y*z + z*x) / 2 := by
  sorry

end cubic_sum_inequality_l1337_133715


namespace frank_final_balance_l1337_133745

def frank_money_problem (initial_amount : ℤ) 
                        (game_cost : ℤ) 
                        (keychain_cost : ℤ) 
                        (friend_gift : ℤ) 
                        (allowance : ℤ) 
                        (bus_ticket_cost : ℤ) : Prop :=
  initial_amount = 11 ∧
  game_cost = 3 ∧
  keychain_cost = 2 ∧
  friend_gift = 4 ∧
  allowance = 14 ∧
  bus_ticket_cost = 5 ∧
  initial_amount - game_cost - keychain_cost + friend_gift + allowance - bus_ticket_cost = 19

theorem frank_final_balance :
  ∀ (initial_amount game_cost keychain_cost friend_gift allowance bus_ticket_cost : ℤ),
  frank_money_problem initial_amount game_cost keychain_cost friend_gift allowance bus_ticket_cost :=
by
  sorry

end frank_final_balance_l1337_133745


namespace inequality_solution_l1337_133743

theorem inequality_solution (m : ℝ) (hm : 0 < m ∧ m < 1) :
  {x : ℝ | m * x / (x - 3) > 1} = {x : ℝ | 3 < x ∧ x < 3 / (1 - m)} := by
  sorry

end inequality_solution_l1337_133743


namespace always_quadratic_radical_l1337_133709

theorem always_quadratic_radical (a : ℝ) : ∃ (x : ℝ), x ^ 2 = a ^ 2 + 1 := by
  sorry

end always_quadratic_radical_l1337_133709


namespace random_event_identification_l1337_133712

-- Define the three events
def event1 : Prop := ∃ x y : ℝ, x * y < 0 ∧ x + y < 0
def event2 : Prop := ∀ x y : ℝ, x * y < 0 → x * y > 0
def event3 : Prop := ∀ x y : ℝ, x * y < 0 → x / y < 0

-- Define what it means for an event to be certain
def is_certain (e : Prop) : Prop := e ∨ ¬e

-- Theorem stating that event1 is not certain, while event2 and event3 are certain
theorem random_event_identification :
  ¬(is_certain event1) ∧ (is_certain event2) ∧ (is_certain event3) :=
sorry

end random_event_identification_l1337_133712


namespace rectangular_solid_surface_area_l1337_133770

/-- A rectangular solid with prime edge lengths and volume 429 has surface area 430. -/
theorem rectangular_solid_surface_area :
  ∀ l w h : ℕ,
  Prime l → Prime w → Prime h →
  l * w * h = 429 →
  2 * (l * w + w * h + h * l) = 430 := by
sorry

end rectangular_solid_surface_area_l1337_133770


namespace wheel_distance_l1337_133767

/-- The distance covered by a wheel given its circumference and number of revolutions -/
theorem wheel_distance (circumference : ℝ) (revolutions : ℝ) :
  circumference = 56 →
  revolutions = 3.002729754322111 →
  circumference * revolutions = 168.1528670416402 := by
  sorry

end wheel_distance_l1337_133767


namespace remainder_of_2457634_div_8_l1337_133737

theorem remainder_of_2457634_div_8 : 2457634 % 8 = 2 := by
  sorry

end remainder_of_2457634_div_8_l1337_133737


namespace alyona_floor_l1337_133768

/-- Represents a multi-story building with multiple entrances -/
structure Building where
  stories : ℕ
  apartments_per_floor : ℕ
  entrances : ℕ

/-- Calculates the floor number given an apartment number and building structure -/
def floor_number (b : Building) (apartment : ℕ) : ℕ :=
  let apartments_per_entrance := b.stories * b.apartments_per_floor
  let apartments_before_entrance := ((apartment - 1) / apartments_per_entrance) * apartments_per_entrance
  let remaining_apartments := apartment - apartments_before_entrance
  ((remaining_apartments - 1) / b.apartments_per_floor) + 1

/-- Theorem stating that Alyona lives on the 3rd floor -/
theorem alyona_floor :
  ∀ (b : Building),
    b.stories = 9 →
    b.entrances ≥ 10 →
    floor_number b 333 = 3 :=
by sorry

end alyona_floor_l1337_133768


namespace segment_length_segment_length_is_ten_l1337_133706

theorem segment_length : ℝ → Prop :=
  fun length => ∃ x₁ x₂ : ℝ,
    (|x₁ - Real.sqrt 25| = 5) ∧
    (|x₂ - Real.sqrt 25| = 5) ∧
    (x₁ ≠ x₂) ∧
    (length = |x₁ - x₂|) ∧
    (length = 10)

-- The proof goes here
theorem segment_length_is_ten : segment_length 10 := by
  sorry

end segment_length_segment_length_is_ten_l1337_133706


namespace expansion_terms_count_expansion_terms_count_equals_66_l1337_133774

theorem expansion_terms_count : Nat :=
  let n : Nat := 10  -- power in (a + b + c)^10
  let k : Nat := 3   -- number of variables (a, b, c)
  Nat.choose (n + k - 1) (k - 1)

theorem expansion_terms_count_equals_66 : expansion_terms_count = 66 := by
  sorry

end expansion_terms_count_expansion_terms_count_equals_66_l1337_133774


namespace prime_has_property_P_infinitely_many_composite_with_property_P_l1337_133796

-- Define property P
def has_property_P (n : ℕ) : Prop :=
  ∀ a : ℕ, a > 0 → (n ∣ a^n - 1) → (n^2 ∣ a^n - 1)

-- Theorem 1: Every prime number has property P
theorem prime_has_property_P :
  ∀ p : ℕ, Prime p → has_property_P p :=
sorry

-- Define a set of composite numbers with property P
def composite_with_property_P : Set ℕ :=
  {n : ℕ | ¬Prime n ∧ has_property_P n}

-- Theorem 2: There are infinitely many composite numbers with property P
theorem infinitely_many_composite_with_property_P :
  Set.Infinite composite_with_property_P :=
sorry

end prime_has_property_P_infinitely_many_composite_with_property_P_l1337_133796


namespace original_number_proof_l1337_133760

theorem original_number_proof (N : ℕ) : 
  (∃ k : ℕ, N - 33 = 87 * k) ∧ 
  (∀ m : ℕ, m < 33 → ¬∃ j : ℕ, N - m = 87 * j) → 
  N = 120 := by
sorry

end original_number_proof_l1337_133760


namespace sum_of_ages_l1337_133740

/-- The sum of Mario and Maria's ages is 7 years -/
theorem sum_of_ages : 
  ∀ (mario_age maria_age : ℕ),
  mario_age = 4 →
  mario_age = maria_age + 1 →
  mario_age + maria_age = 7 := by
sorry

end sum_of_ages_l1337_133740


namespace joe_weight_loss_l1337_133779

/-- Represents Joe's weight loss problem --/
theorem joe_weight_loss 
  (initial_weight : ℝ) 
  (months_on_diet : ℝ) 
  (future_weight : ℝ) 
  (months_until_future_weight : ℝ) 
  (h1 : initial_weight = 222)
  (h2 : months_on_diet = 3)
  (h3 : future_weight = 170)
  (h4 : months_until_future_weight = 3.5)
  : ∃ (current_weight : ℝ), 
    current_weight = initial_weight - (initial_weight - future_weight) * (months_on_diet / (months_on_diet + months_until_future_weight))
    ∧ current_weight = 198 :=
by sorry

end joe_weight_loss_l1337_133779


namespace two_times_binomial_twelve_choose_three_l1337_133732

theorem two_times_binomial_twelve_choose_three : 2 * (Nat.choose 12 3) = 440 := by
  sorry

end two_times_binomial_twelve_choose_three_l1337_133732


namespace polynomial_remainder_theorem_l1337_133765

theorem polynomial_remainder_theorem (x : ℝ) : 
  ∃ (Q : ℝ → ℝ) (R : ℝ → ℝ), 
    (∀ x, x^50 = (x^2 - 5*x + 6) * Q x + R x) ∧ 
    (∃ a b : ℝ, ∀ x, R x = a*x + b) ∧
    R x = (3^50 - 2^50)*x + (2^50 - 2*3^50 + 2*2^50) := by
  sorry

end polynomial_remainder_theorem_l1337_133765


namespace is_circle_center_l1337_133742

/-- The equation of a circle in the xy-plane -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x + 12*y + 1 = 0

/-- The center of the circle -/
def circle_center : ℝ × ℝ := (1, -6)

/-- Theorem stating that the given point is the center of the circle -/
theorem is_circle_center : 
  ∀ (x y : ℝ), circle_equation x y ↔ (x - circle_center.1)^2 + (y - circle_center.2)^2 = 36 :=
by sorry

end is_circle_center_l1337_133742


namespace a_mod_4_is_2_or_3_a_not_perfect_square_l1337_133775

def a : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => a n * a (n + 1) + 1

theorem a_mod_4_is_2_or_3 (n : ℕ) (h : n ≥ 2) : 
  (a n) % 4 = 2 ∨ (a n) % 4 = 3 :=
by sorry

theorem a_not_perfect_square (n : ℕ) (h : n ≥ 2) : 
  ¬ ∃ (k : ℕ), a n = k * k :=
by sorry

end a_mod_4_is_2_or_3_a_not_perfect_square_l1337_133775
