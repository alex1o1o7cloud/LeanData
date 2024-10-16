import Mathlib

namespace NUMINAMATH_CALUDE_smallest_dual_base_palindrome_l3182_318203

/-- A function that checks if a number is a palindrome in a given base -/
def isPalindrome (n : ℕ) (base : ℕ) : Prop := sorry

/-- A function that converts a number from base 10 to another base -/
def toBase (n : ℕ) (base : ℕ) : List ℕ := sorry

theorem smallest_dual_base_palindrome :
  ∀ n : ℕ,
    n > 10 →
    (isPalindrome n 2 ∧ isPalindrome n 4) →
    n ≥ 15 :=
by sorry

end NUMINAMATH_CALUDE_smallest_dual_base_palindrome_l3182_318203


namespace NUMINAMATH_CALUDE_function_properties_l3182_318283

open Real

theorem function_properties (f : ℝ → ℝ) (k : ℝ) (hf : Differentiable ℝ f) 
  (h0 : f 0 = -1) (hk : k > 1) (hf' : ∀ x, deriv f x > k) :
  (f (1/k) > 1/k - 1) ∧ 
  (f (1/(k-1)) > 1/(k-1)) ∧ 
  (f (1/k) < f (1/(k-1))) := by
sorry

end NUMINAMATH_CALUDE_function_properties_l3182_318283


namespace NUMINAMATH_CALUDE_jelly_bean_probability_l3182_318224

/-- The probability of selecting either a blue or purple jelly bean from a bag -/
theorem jelly_bean_probability : 
  let red : ℕ := 4
  let green : ℕ := 5
  let yellow : ℕ := 9
  let blue : ℕ := 7
  let purple : ℕ := 10
  let total : ℕ := red + green + yellow + blue + purple
  (blue + purple : ℚ) / total = 17 / 35 := by sorry

end NUMINAMATH_CALUDE_jelly_bean_probability_l3182_318224


namespace NUMINAMATH_CALUDE_cost_price_calculation_l3182_318284

theorem cost_price_calculation (selling_price : ℚ) (profit_percentage : ℚ) : 
  selling_price = 120 → profit_percentage = 25 → 
  ∃ (cost_price : ℚ), cost_price = 96 ∧ selling_price = cost_price * (1 + profit_percentage / 100) :=
by
  sorry

end NUMINAMATH_CALUDE_cost_price_calculation_l3182_318284


namespace NUMINAMATH_CALUDE_closer_to_b_probability_is_half_l3182_318254

/-- A triangle with sides of length 6, 8, and 10 -/
structure SpecialTriangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  ab_length : dist A B = 6
  bc_length : dist B C = 8
  ca_length : dist C A = 10

/-- The probability that a random point in the triangle is closer to B than to A or C -/
def closerToBProbability (t : SpecialTriangle) : ℝ :=
  sorry

/-- The main theorem stating that the probability is 1/2 -/
theorem closer_to_b_probability_is_half (t : SpecialTriangle) :
  closerToBProbability t = 1/2 :=
sorry

end NUMINAMATH_CALUDE_closer_to_b_probability_is_half_l3182_318254


namespace NUMINAMATH_CALUDE_abs_two_i_over_one_minus_i_l3182_318231

/-- The absolute value of the complex number 2i / (1-i) is √2 -/
theorem abs_two_i_over_one_minus_i :
  let i : ℂ := Complex.I
  let z : ℂ := 2 * i / (1 - i)
  Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_abs_two_i_over_one_minus_i_l3182_318231


namespace NUMINAMATH_CALUDE_rectangle_diagonal_l3182_318297

/-- The diagonal of a rectangle with width 16 and length 12 is 20. -/
theorem rectangle_diagonal : ∃ (d : ℝ), d = 20 ∧ d^2 = 16^2 + 12^2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_diagonal_l3182_318297


namespace NUMINAMATH_CALUDE_binomial_multiplication_l3182_318278

theorem binomial_multiplication (x : ℝ) : (4*x + 3) * (2*x - 7) = 8*x^2 - 22*x - 21 := by
  sorry

end NUMINAMATH_CALUDE_binomial_multiplication_l3182_318278


namespace NUMINAMATH_CALUDE_perpendicular_parallel_implies_plane_parallel_plane_parallel_line_in_plane_implies_line_parallel_plane_l3182_318200

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (contains : Plane → Line → Prop)
variable (plane_parallel : Plane → Plane → Prop)
variable (line_parallel_plane : Line → Plane → Prop)

-- Axioms
axiom distinct_lines (m n : Line) : m ≠ n
axiom non_coincident_planes (α β : Plane) : α ≠ β

-- Theorem 1
theorem perpendicular_parallel_implies_plane_parallel 
  (m n : Line) (α β : Plane) :
  perpendicular m α → perpendicular n β → parallel m n → 
  plane_parallel α β :=
sorry

-- Theorem 2
theorem plane_parallel_line_in_plane_implies_line_parallel_plane 
  (m : Line) (α β : Plane) :
  plane_parallel α β → contains α m → line_parallel_plane m β :=
sorry

end NUMINAMATH_CALUDE_perpendicular_parallel_implies_plane_parallel_plane_parallel_line_in_plane_implies_line_parallel_plane_l3182_318200


namespace NUMINAMATH_CALUDE_red_light_runners_estimate_l3182_318228

/-- Represents the result of a survey on traffic law compliance -/
structure SurveyResult where
  total_students : ℕ
  yes_answers : ℕ
  id_range : Finset ℕ
  odd_ids : Finset ℕ

/-- Calculates the estimated number of students who have run a red light -/
def estimate_red_light_runners (result : SurveyResult) : ℕ :=
  2 * (result.yes_answers - result.odd_ids.card / 2)

/-- Theorem stating the estimated number of red light runners based on the survey -/
theorem red_light_runners_estimate 
  (result : SurveyResult)
  (h1 : result.total_students = 300)
  (h2 : result.yes_answers = 90)
  (h3 : result.id_range = Finset.range 300)
  (h4 : result.odd_ids = result.id_range.filter (fun n => n % 2 = 1)) :
  estimate_red_light_runners result = 30 := by
  sorry

end NUMINAMATH_CALUDE_red_light_runners_estimate_l3182_318228


namespace NUMINAMATH_CALUDE_afternoon_sales_l3182_318260

/-- Represents the amount of pears sold by a salesman in a day -/
structure PearSales where
  morning : ℕ
  afternoon : ℕ
  total : ℕ

/-- Theorem stating the afternoon sales given the conditions -/
theorem afternoon_sales (sales : PearSales) 
  (h1 : sales.afternoon = 2 * sales.morning) 
  (h2 : sales.total = sales.morning + sales.afternoon)
  (h3 : sales.total = 510) : 
  sales.afternoon = 340 := by
  sorry

#check afternoon_sales

end NUMINAMATH_CALUDE_afternoon_sales_l3182_318260


namespace NUMINAMATH_CALUDE_tank_plastering_cost_l3182_318237

/-- Calculates the cost of plastering a rectangular tank's walls and bottom -/
def plasteringCost (length width depth rate : ℝ) : ℝ :=
  let wallArea := 2 * (length * depth + width * depth)
  let bottomArea := length * width
  let totalArea := wallArea + bottomArea
  totalArea * rate

/-- Theorem: The cost of plastering a 35m x 18m x 10m tank at ₹135 per sq m is ₹228,150 -/
theorem tank_plastering_cost :
  plasteringCost 35 18 10 135 = 228150 := by
  sorry

end NUMINAMATH_CALUDE_tank_plastering_cost_l3182_318237


namespace NUMINAMATH_CALUDE_tangent_line_to_logarithmic_curve_l3182_318247

theorem tangent_line_to_logarithmic_curve : ∃ (n : ℕ+) (a : ℝ), 
  (n : ℝ) < a ∧ a < (n : ℝ) + 1 ∧
  (∃ (x : ℝ), x > 0 ∧ x + 1 = a * Real.log x ∧ 1 = a / x) ∧
  n = 3 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_to_logarithmic_curve_l3182_318247


namespace NUMINAMATH_CALUDE_rectangle_length_breadth_difference_l3182_318204

/-- Given a rectangular plot with breadth 11 metres and area 21 times its breadth,
    the difference between its length and breadth is 10 metres. -/
theorem rectangle_length_breadth_difference : ℝ → Prop :=
  fun difference =>
    ∀ (length breadth area : ℝ),
      breadth = 11 →
      area = 21 * breadth →
      area = length * breadth →
      difference = length - breadth →
      difference = 10

/-- Proof of the theorem -/
lemma prove_rectangle_length_breadth_difference :
  rectangle_length_breadth_difference 10 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_length_breadth_difference_l3182_318204


namespace NUMINAMATH_CALUDE_three_intersection_points_l3182_318225

-- Define the three lines
def line1 (x y : ℝ) : Prop := 4 * y - 3 * x = 2
def line2 (x y : ℝ) : Prop := 2 * x + 3 * y = 9
def line3 (x y : ℝ) : Prop := x - y = 1

-- Define an intersection point
def is_intersection (x y : ℝ) : Prop :=
  (line1 x y ∧ line2 x y) ∨ (line2 x y ∧ line3 x y) ∨ (line1 x y ∧ line3 x y)

-- Theorem statement
theorem three_intersection_points :
  ∃ (p1 p2 p3 : ℝ × ℝ),
    is_intersection p1.1 p1.2 ∧
    is_intersection p2.1 p2.2 ∧
    is_intersection p3.1 p3.2 ∧
    p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 ∧
    ∀ (x y : ℝ), is_intersection x y → (x, y) = p1 ∨ (x, y) = p2 ∨ (x, y) = p3 :=
sorry

end NUMINAMATH_CALUDE_three_intersection_points_l3182_318225


namespace NUMINAMATH_CALUDE_paul_initial_pens_l3182_318236

/-- The number of pens Paul sold in the garage sale. -/
def pens_sold : ℕ := 92

/-- The number of pens Paul had left after the garage sale. -/
def pens_left : ℕ := 14

/-- The initial number of pens Paul had. -/
def initial_pens : ℕ := pens_sold + pens_left

theorem paul_initial_pens : initial_pens = 106 := by
  sorry

end NUMINAMATH_CALUDE_paul_initial_pens_l3182_318236


namespace NUMINAMATH_CALUDE_password_probability_l3182_318281

def password_space : ℕ := 10 * 52 * 52 * 10

def even_start_space : ℕ := 5 * 52 * 52 * 10

def diff_letters_space : ℕ := 10 * 52 * 51 * 10

def non_zero_end_space : ℕ := 10 * 52 * 52 * 9

def valid_password_space : ℕ := 5 * 52 * 51 * 9

theorem password_probability :
  (valid_password_space : ℚ) / password_space = 459 / 1040 := by
  sorry

end NUMINAMATH_CALUDE_password_probability_l3182_318281


namespace NUMINAMATH_CALUDE_day_301_is_sunday_l3182_318214

/-- Days of the week -/
inductive DayOfWeek
| Sunday
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday

/-- Function to determine the day of the week given a day number -/
def dayOfWeek (dayNumber : Nat) : DayOfWeek :=
  match dayNumber % 7 with
  | 0 => DayOfWeek.Sunday
  | 1 => DayOfWeek.Monday
  | 2 => DayOfWeek.Tuesday
  | 3 => DayOfWeek.Wednesday
  | 4 => DayOfWeek.Thursday
  | 5 => DayOfWeek.Friday
  | _ => DayOfWeek.Saturday

/-- Theorem: If the 35th day is a Sunday, then the 301st day is also a Sunday -/
theorem day_301_is_sunday (h : dayOfWeek 35 = DayOfWeek.Sunday) :
  dayOfWeek 301 = DayOfWeek.Sunday :=
by
  sorry


end NUMINAMATH_CALUDE_day_301_is_sunday_l3182_318214


namespace NUMINAMATH_CALUDE_problem_building_has_20_stories_l3182_318248

/-- A building with specific height properties -/
structure Building where
  first_stories : ℕ
  first_story_height : ℕ
  remaining_story_height : ℕ
  total_height : ℕ

/-- The number of stories in the building -/
def Building.total_stories (b : Building) : ℕ :=
  b.first_stories + (b.total_height - b.first_stories * b.first_story_height) / b.remaining_story_height

/-- The specific building described in the problem -/
def problem_building : Building := {
  first_stories := 10
  first_story_height := 12
  remaining_story_height := 15
  total_height := 270
}

/-- Theorem stating that the problem building has 20 stories -/
theorem problem_building_has_20_stories :
  problem_building.total_stories = 20 := by
  sorry

end NUMINAMATH_CALUDE_problem_building_has_20_stories_l3182_318248


namespace NUMINAMATH_CALUDE_impossibleToGather_l3182_318222

/-- Represents the number of islands and ships -/
def n : ℕ := 1002

/-- Represents the position of a ship on the circular archipelago -/
def Position := Fin n

/-- Represents the fleet of ships -/
def Fleet := Multiset Position

/-- Represents a single day's movement of two ships -/
def Move := Position × Position × Position × Position

/-- Checks if all ships are gathered on a single island -/
def allGathered (fleet : Fleet) : Prop :=
  ∃ p : Position, fleet = Multiset.replicate n p

/-- Applies a move to the fleet -/
def applyMove (fleet : Fleet) (move : Move) : Fleet :=
  sorry

/-- The main theorem stating that it's impossible to gather all ships -/
theorem impossibleToGather (initialFleet : Fleet) :
  ¬∃ (moves : List Move), allGathered (moves.foldl applyMove initialFleet) :=
sorry

end NUMINAMATH_CALUDE_impossibleToGather_l3182_318222


namespace NUMINAMATH_CALUDE_min_value_of_f_l3182_318226

/-- Given positive real numbers a, b, c, x, y, z satisfying certain conditions,
    the function f(x, y, z) has a minimum value of 1/2. -/
theorem min_value_of_f (a b c x y z : ℝ) 
    (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c)
    (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z)
    (eq1 : c * y + b * z = a)
    (eq2 : a * z + c * x = b)
    (eq3 : b * x + a * y = c) :
    let f := fun (x y z : ℝ) => x^2 / (1 + x) + y^2 / (1 + y) + z^2 / (1 + z)
    ∀ x' y' z' : ℝ, 0 < x' → 0 < y' → 0 < z' → f x' y' z' ≥ 1/2 ∧ 
    ∃ x₀ y₀ z₀ : ℝ, 0 < x₀ ∧ 0 < y₀ ∧ 0 < z₀ ∧ f x₀ y₀ z₀ = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_f_l3182_318226


namespace NUMINAMATH_CALUDE_fonzie_payment_l3182_318207

/-- Proves that Fonzie's payment for the treasure map is $7000 -/
theorem fonzie_payment (fonzie_payment : ℝ) : 
  (∀ total_payment : ℝ, 
    total_payment = fonzie_payment + 8000 + 9000 ∧ 
    9000 / total_payment = 337500 / 900000) →
  fonzie_payment = 7000 := by
sorry

end NUMINAMATH_CALUDE_fonzie_payment_l3182_318207


namespace NUMINAMATH_CALUDE_gcd_of_powers_minus_one_l3182_318288

theorem gcd_of_powers_minus_one : Nat.gcd (4^8 - 1) (8^12 - 1) = 15 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_powers_minus_one_l3182_318288


namespace NUMINAMATH_CALUDE_tax_increase_proof_l3182_318219

theorem tax_increase_proof (item_cost : ℝ) (old_rate new_rate : ℝ) 
  (h1 : item_cost = 1000)
  (h2 : old_rate = 0.07)
  (h3 : new_rate = 0.075) :
  new_rate * item_cost - old_rate * item_cost = 5 :=
by sorry

end NUMINAMATH_CALUDE_tax_increase_proof_l3182_318219


namespace NUMINAMATH_CALUDE_square_difference_l3182_318291

theorem square_difference (x y : ℝ) (h1 : x + y = 20) (h2 : x - y = 4) : x^2 - y^2 = 80 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_l3182_318291


namespace NUMINAMATH_CALUDE_f_pos_theorem_l3182_318287

-- Define an even function
def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

-- Define the function for x < 0
def f_neg (x : ℝ) : ℝ := x * (x + 1)

-- State the theorem
theorem f_pos_theorem (f : ℝ → ℝ) 
  (h_even : even_function f) 
  (h_neg : ∀ x < 0, f x = f_neg x) : 
  ∀ x > 0, f x = x^2 - x := by sorry

end NUMINAMATH_CALUDE_f_pos_theorem_l3182_318287


namespace NUMINAMATH_CALUDE_quadratic_form_sum_l3182_318244

theorem quadratic_form_sum (x : ℝ) : ∃ (a b c : ℝ),
  (-3 * x^2 + 24 * x + 81 = a * (x + b)^2 + c) ∧ (a + b + c = 122) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_form_sum_l3182_318244


namespace NUMINAMATH_CALUDE_trigonometric_identity_l3182_318263

theorem trigonometric_identity (α β : Real) 
  (h1 : Real.tan (α + β) = 2/5)
  (h2 : Real.tan (β - π/4) = 1/4) : 
  (Real.cos α + Real.sin α) / (Real.cos α - Real.sin α) = 3/22 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l3182_318263


namespace NUMINAMATH_CALUDE_binary_110011_equals_51_l3182_318217

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_110011_equals_51 :
  binary_to_decimal [true, true, false, false, true, true] = 51 := by
  sorry

end NUMINAMATH_CALUDE_binary_110011_equals_51_l3182_318217


namespace NUMINAMATH_CALUDE_expression_evaluation_l3182_318220

theorem expression_evaluation (x : ℝ) (h : x = 6) :
  (1 + 2 / (x + 1)) * ((x^2 + x) / (x^2 - 9)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3182_318220


namespace NUMINAMATH_CALUDE_game_cost_is_two_l3182_318271

/-- Calculates the cost of a new game based on initial money, allowance, and final amount. -/
def game_cost (initial_money : ℝ) (allowance : ℝ) (final_amount : ℝ) : ℝ :=
  initial_money + allowance - final_amount

/-- Proves that the cost of the new game is $2 given the specific amounts in the problem. -/
theorem game_cost_is_two :
  game_cost 5 5 8 = 2 := by
  sorry

end NUMINAMATH_CALUDE_game_cost_is_two_l3182_318271


namespace NUMINAMATH_CALUDE_chess_match_draw_probability_l3182_318299

theorem chess_match_draw_probability (john_win_prob mike_win_prob : ℚ) 
  (h1 : john_win_prob = 4/9)
  (h2 : mike_win_prob = 5/18) : 
  1 - (john_win_prob + mike_win_prob) = 5/18 := by
  sorry

end NUMINAMATH_CALUDE_chess_match_draw_probability_l3182_318299


namespace NUMINAMATH_CALUDE_problem_statement_l3182_318295

theorem problem_statement : 
  (Real.sin (15 * π / 180)) / (Real.cos (75 * π / 180)) + 
  1 / (Real.sin (75 * π / 180))^2 - 
  (Real.tan (15 * π / 180))^2 = 2 := by sorry

end NUMINAMATH_CALUDE_problem_statement_l3182_318295


namespace NUMINAMATH_CALUDE_quadratic_factoring_l3182_318274

theorem quadratic_factoring (x : ℝ) : x^2 - 2*x - 2 = 0 ↔ (x - 1)^2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factoring_l3182_318274


namespace NUMINAMATH_CALUDE_product_divisible_by_5_probability_l3182_318234

/-- The number of sides on each die -/
def num_sides : ℕ := 8

/-- The number of dice rolled -/
def num_dice : ℕ := 6

/-- The probability that the product of the numbers rolled is divisible by 5 -/
def prob_divisible_by_5 : ℚ := 144495 / 262144

/-- Theorem stating the probability of the product being divisible by 5 -/
theorem product_divisible_by_5_probability :
  (1 : ℚ) - (1 - 1 / num_sides) ^ num_dice = prob_divisible_by_5 := by
  sorry

end NUMINAMATH_CALUDE_product_divisible_by_5_probability_l3182_318234


namespace NUMINAMATH_CALUDE_calculate_expression_l3182_318221

theorem calculate_expression : 8 * (2 / 16) * 32 - 10 = 22 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l3182_318221


namespace NUMINAMATH_CALUDE_train_speed_l3182_318269

/-- The speed of a train given its length, the platform length, and the time to cross the platform -/
theorem train_speed (train_length platform_length : Real) (crossing_time : Real) :
  train_length = 110 ∧ 
  platform_length = 323.36799999999994 ∧ 
  crossing_time = 30 →
  (train_length + platform_length) / crossing_time * 3.6 = 52.00416 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l3182_318269


namespace NUMINAMATH_CALUDE_sales_tax_difference_l3182_318268

-- Define the original price, discount rate, and tax rates
def original_price : ℝ := 50
def discount_rate : ℝ := 0.10
def tax_rate_1 : ℝ := 0.08
def tax_rate_2 : ℝ := 0.075

-- Define the discounted price
def discounted_price : ℝ := original_price * (1 - discount_rate)

-- Define the tax difference function
def tax_difference (price : ℝ) (rate1 : ℝ) (rate2 : ℝ) : ℝ :=
  price * rate1 - price * rate2

-- Theorem statement
theorem sales_tax_difference :
  tax_difference discounted_price tax_rate_1 tax_rate_2 = 0.225 := by
  sorry

end NUMINAMATH_CALUDE_sales_tax_difference_l3182_318268


namespace NUMINAMATH_CALUDE_matrix_inverse_scalar_multiple_l3182_318232

/-- Given a 2x2 matrix A with elements [[1, 3], [4, d]] where A⁻¹ = k * A,
    prove that d = 6 and k = 1/6 -/
theorem matrix_inverse_scalar_multiple (d k : ℝ) : 
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![1, 3; 4, d]
  A⁻¹ = k • A → d = 6 ∧ k = 1/6 := by
  sorry

end NUMINAMATH_CALUDE_matrix_inverse_scalar_multiple_l3182_318232


namespace NUMINAMATH_CALUDE_correct_quotient_after_error_l3182_318210

theorem correct_quotient_after_error (dividend : ℕ) (incorrect_divisor correct_divisor incorrect_quotient : ℕ) :
  incorrect_divisor = 48 →
  correct_divisor = 36 →
  incorrect_quotient = 24 →
  dividend = incorrect_divisor * incorrect_quotient →
  dividend / correct_divisor = 32 :=
by sorry

end NUMINAMATH_CALUDE_correct_quotient_after_error_l3182_318210


namespace NUMINAMATH_CALUDE_tournament_boxes_needed_l3182_318242

/-- A single-elimination tennis tournament -/
structure TennisTournament where
  participants : ℕ
  boxes_per_match : ℕ

/-- The number of boxes needed for a single-elimination tournament -/
def boxes_needed (t : TennisTournament) : ℕ :=
  t.participants - 1

/-- Theorem: A single-elimination tournament with 199 participants needs 198 boxes -/
theorem tournament_boxes_needed :
  ∀ t : TennisTournament, t.participants = 199 ∧ t.boxes_per_match = 1 →
  boxes_needed t = 198 :=
by sorry

end NUMINAMATH_CALUDE_tournament_boxes_needed_l3182_318242


namespace NUMINAMATH_CALUDE_quadratic_root_implies_m_value_l3182_318233

theorem quadratic_root_implies_m_value (m : ℝ) : 
  (∃ x : ℝ, 2 * x^2 - m * x + 3 = 0 ∧ x = 3) → m = 7 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_m_value_l3182_318233


namespace NUMINAMATH_CALUDE_triangle_proof_l3182_318251

-- Define a triangle structure
structure Triangle where
  angle1 : ℝ
  angle2 : ℝ
  angle3 : ℝ

-- Define the property of being acute
def isAcute (t : Triangle) : Prop :=
  t.angle1 < 90 ∧ t.angle2 < 90 ∧ t.angle3 < 90

-- Theorem statement
theorem triangle_proof (t : Triangle) 
  (h1 : t.angle1 = 48)
  (h2 : t.angle2 = 52)
  (h3 : t.angle1 + t.angle2 + t.angle3 = 180) : 
  t.angle3 = 80 ∧ isAcute t := by
  sorry

end NUMINAMATH_CALUDE_triangle_proof_l3182_318251


namespace NUMINAMATH_CALUDE_min_buildings_correct_min_buildings_20x20_min_buildings_50x90_l3182_318250

/-- Represents a rectangular city grid -/
structure City where
  rows : ℕ
  cols : ℕ

/-- Calculates the minimum number of buildings after renovation -/
def min_buildings (city : City) : ℕ :=
  ((city.rows * city.cols + 15) / 16 : ℕ)

/-- Theorem: The minimum number of buildings after renovation is correct -/
theorem min_buildings_correct (city : City) :
  min_buildings city = ⌈(city.rows * city.cols : ℚ) / 16⌉ :=
sorry

/-- Corollary: For a 20x20 grid, the minimum number of buildings is 25 -/
theorem min_buildings_20x20 :
  min_buildings { rows := 20, cols := 20 } = 25 :=
sorry

/-- Corollary: For a 50x90 grid, the minimum number of buildings is 282 -/
theorem min_buildings_50x90 :
  min_buildings { rows := 50, cols := 90 } = 282 :=
sorry

end NUMINAMATH_CALUDE_min_buildings_correct_min_buildings_20x20_min_buildings_50x90_l3182_318250


namespace NUMINAMATH_CALUDE_vertex_ordinate_zero_l3182_318201

/-- A quadratic polynomial with real coefficients -/
def QuadraticPolynomial (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

/-- The number of solutions to the equation (f x)^3 - f x = 0 -/
def numSolutions (f : ℝ → ℝ) : ℕ := sorry

/-- The ordinate (y-coordinate) of the vertex of a quadratic polynomial -/
def vertexOrdinate (f : ℝ → ℝ) : ℝ := sorry

/-- 
If f is a quadratic polynomial and (f x)^3 - f x = 0 has exactly three solutions,
then the ordinate of the vertex of f is 0
-/
theorem vertex_ordinate_zero 
  (a b c : ℝ) 
  (f : ℝ → ℝ) 
  (hf : f = QuadraticPolynomial a b c) 
  (h_solutions : numSolutions f = 3) : 
  vertexOrdinate f = 0 := by sorry

end NUMINAMATH_CALUDE_vertex_ordinate_zero_l3182_318201


namespace NUMINAMATH_CALUDE_age_of_other_man_l3182_318267

theorem age_of_other_man (n : ℕ) (initial_avg : ℝ) (new_avg : ℝ) (man_age : ℕ) (women_avg : ℝ) :
  n = 8 ∧ 
  new_avg = initial_avg + 2 ∧ 
  man_age = 20 ∧ 
  women_avg = 30 → 
  ∃ x : ℕ, x = 24 ∧ 
    n * initial_avg - (man_age + x) + 2 * women_avg = n * new_avg :=
by sorry

end NUMINAMATH_CALUDE_age_of_other_man_l3182_318267


namespace NUMINAMATH_CALUDE_inequality_proof_l3182_318218

theorem inequality_proof (a b c : ℝ) (h1 : a > b) (h2 : b > c) : 
  1 / (a - b) + 1 / (b - c) + 4 / (c - a) ≥ 0 :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l3182_318218


namespace NUMINAMATH_CALUDE_min_value_quadratic_l3182_318275

theorem min_value_quadratic (x : ℝ) (y : ℝ) :
  x ∈ Set.Icc (-1 : ℝ) (1 : ℝ) →
  y = 2 * x^2 - 6 * x + 3 →
  ∃ (m : ℝ), m = -1 ∧ ∀ z ∈ Set.Icc (-1 : ℝ) (1 : ℝ), 2 * z^2 - 6 * z + 3 ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l3182_318275


namespace NUMINAMATH_CALUDE_worksheets_graded_l3182_318245

theorem worksheets_graded (total_worksheets : ℕ) (problems_per_worksheet : ℕ) (problems_left : ℕ) : 
  total_worksheets = 16 →
  problems_per_worksheet = 4 →
  problems_left = 32 →
  total_worksheets * problems_per_worksheet - problems_left = 8 * problems_per_worksheet :=
by sorry

end NUMINAMATH_CALUDE_worksheets_graded_l3182_318245


namespace NUMINAMATH_CALUDE_rectangle_area_difference_l3182_318279

/-- Given a large rectangle of dimensions A × B containing a smaller rectangle of dimensions a × b,
    the difference between the total area of yellow regions and green regions is A × b - a × B. -/
theorem rectangle_area_difference (A B a b : ℝ) (hA : A > 0) (hB : B > 0) (ha : a > 0) (hb : b > 0)
  (ha_le_A : a ≤ A) (hb_le_B : b ≤ B) :
  A * b - a * B = A * b - a * B := by sorry

end NUMINAMATH_CALUDE_rectangle_area_difference_l3182_318279


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l3182_318259

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

theorem quadratic_equation_roots (p q : ℕ) (hp : is_prime p) (hq : is_prime q) :
  ∃ (x y : ℕ), x ≠ y ∧ x > 0 ∧ y > 0 ∧ x^2 - p*x + 2*q = 0 ∧ y^2 - p*y + 2*q = 0 →
  (∃ r : ℕ, (r = x ∨ r = y) ∧ is_prime r) ∧
  is_prime (p - q) ∧
  ¬(∀ x y : ℕ, x^2 - p*x + 2*q = 0 → y^2 - p*y + 2*q = 0 → x ≠ y → Even (x - y)) ∧
  ¬(is_prime (p^2 + 2*q)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l3182_318259


namespace NUMINAMATH_CALUDE_triangle_abc_area_l3182_318229

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a circle with a center and radius -/
structure Circle where
  center : Point
  radius : ℝ

/-- The line to which the circles are tangent -/
def line_m : Set Point := sorry

theorem triangle_abc_area :
  let circle_a : Circle := { center := { x := -5, y := 2 }, radius := 2 }
  let circle_b : Circle := { center := { x := 0, y := 3 }, radius := 3 }
  let circle_c : Circle := { center := { x := 7, y := 4 }, radius := 4 }
  let point_a' : Point := sorry
  let point_b' : Point := sorry
  let point_c' : Point := sorry

  -- Circles are tangent to line m
  (point_a' ∈ line_m) ∧
  (point_b' ∈ line_m) ∧
  (point_c' ∈ line_m) →

  -- Circle B is externally tangent to circles A and C
  (circle_b.center.x - circle_a.center.x)^2 + (circle_b.center.y - circle_a.center.y)^2 = (circle_b.radius + circle_a.radius)^2 ∧
  (circle_b.center.x - circle_c.center.x)^2 + (circle_b.center.y - circle_c.center.y)^2 = (circle_b.radius + circle_c.radius)^2 →

  -- B' is between A' and C' on line m
  (point_b'.x > point_a'.x ∧ point_b'.x < point_c'.x) →

  -- Centers A and C are aligned horizontally
  circle_a.center.y = circle_c.center.y →

  -- The area of triangle ABC is 6
  abs ((circle_a.center.x * (circle_b.center.y - circle_c.center.y) +
        circle_b.center.x * (circle_c.center.y - circle_a.center.y) +
        circle_c.center.x * (circle_a.center.y - circle_b.center.y)) / 2) = 6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_abc_area_l3182_318229


namespace NUMINAMATH_CALUDE_multiples_of_four_between_100_and_350_l3182_318240

theorem multiples_of_four_between_100_and_350 : 
  (Finset.filter (fun n => n % 4 = 0) (Finset.range 350 \ Finset.range 100)).card = 62 := by
  sorry

end NUMINAMATH_CALUDE_multiples_of_four_between_100_and_350_l3182_318240


namespace NUMINAMATH_CALUDE_solution_equation_1_solution_equation_2_l3182_318252

-- Equation 1
theorem solution_equation_1 (x : ℝ) : 2*x - 3*(2*x - 3) = x + 4 ↔ x = 1 := by sorry

-- Equation 2
theorem solution_equation_2 (x : ℝ) : (3*x - 1)/4 - 1 = (5*x - 7)/6 ↔ x = -1 := by sorry

end NUMINAMATH_CALUDE_solution_equation_1_solution_equation_2_l3182_318252


namespace NUMINAMATH_CALUDE_simplify_tan_product_l3182_318292

theorem simplify_tan_product : (1 + Real.tan (30 * π / 180)) * (1 + Real.tan (15 * π / 180)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_tan_product_l3182_318292


namespace NUMINAMATH_CALUDE_geometric_sequence_middle_term_l3182_318270

theorem geometric_sequence_middle_term 
  (a b c : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c)
  (geom_seq : b^2 = a * c)
  (value_a : a = 5 + 2 * Real.sqrt 3)
  (value_c : c = 5 - 2 * Real.sqrt 3) : 
  b = Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_middle_term_l3182_318270


namespace NUMINAMATH_CALUDE_yellow_crayon_count_l3182_318209

/-- Given the number of red, blue, and yellow crayons with specific relationships,
    prove that the number of yellow crayons is 32. -/
theorem yellow_crayon_count :
  ∀ (red blue yellow : ℕ),
  red = 14 →
  blue = red + 5 →
  yellow = 2 * blue - 6 →
  yellow = 32 := by
sorry

end NUMINAMATH_CALUDE_yellow_crayon_count_l3182_318209


namespace NUMINAMATH_CALUDE_exam_score_calculation_l3182_318261

theorem exam_score_calculation (total_questions : ℕ) (correct_answers : ℕ) 
  (correct_score : ℤ) (wrong_score : ℤ) : 
  total_questions = 120 →
  correct_answers = 75 →
  correct_score = 3 →
  wrong_score = -1 →
  (correct_answers * correct_score + (total_questions - correct_answers) * wrong_score : ℤ) = 180 := by
  sorry

end NUMINAMATH_CALUDE_exam_score_calculation_l3182_318261


namespace NUMINAMATH_CALUDE_inequality_and_equality_condition_l3182_318298

theorem inequality_and_equality_condition (α β a b : ℝ) (h_pos_α : 0 < α) (h_pos_β : 0 < β)
  (h_a_range : α ≤ a ∧ a ≤ β) (h_b_range : α ≤ b ∧ b ≤ β) :
  b / a + a / b ≤ β / α + α / β ∧
  (b / a + a / b = β / α + α / β ↔ (a = α ∧ b = β) ∨ (a = β ∧ b = α)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_equality_condition_l3182_318298


namespace NUMINAMATH_CALUDE_continued_fraction_value_l3182_318215

theorem continued_fraction_value : 
  ∃ y : ℝ, y = 3 + 5 / (4 + 5 / y) ∧ y = 5 := by
  sorry

end NUMINAMATH_CALUDE_continued_fraction_value_l3182_318215


namespace NUMINAMATH_CALUDE_diet_soda_bottles_l3182_318211

theorem diet_soda_bottles (regular_soda : ℕ) (lite_soda : ℕ) (total_bottles : ℕ) 
  (h1 : regular_soda = 57)
  (h2 : lite_soda = 27)
  (h3 : total_bottles = 110) :
  total_bottles - (regular_soda + lite_soda) = 26 := by
  sorry

end NUMINAMATH_CALUDE_diet_soda_bottles_l3182_318211


namespace NUMINAMATH_CALUDE_x0_value_l3182_318216

open Real

noncomputable def f (x : ℝ) : ℝ := x * log x

theorem x0_value (x₀ : ℝ) (h : x₀ > 0) :
  (deriv f x₀ = 2) → x₀ = exp 1 := by
  sorry

end NUMINAMATH_CALUDE_x0_value_l3182_318216


namespace NUMINAMATH_CALUDE_tangent_line_values_l3182_318249

/-- A line y = kx + b is tangent to two circles -/
def is_tangent_to_circles (k b : ℝ) : Prop :=
  k > 0 ∧
  ∃ (x₁ y₁ : ℝ), x₁^2 + y₁^2 = 1 ∧ y₁ = k * x₁ + b ∧
  ∃ (x₂ y₂ : ℝ), (x₂ - 4)^2 + y₂^2 = 1 ∧ y₂ = k * x₂ + b

/-- The unique values of k and b for a line tangent to both circles -/
theorem tangent_line_values :
  ∀ k b : ℝ, is_tangent_to_circles k b →
  k = Real.sqrt 3 / 3 ∧ b = -(2 * Real.sqrt 3 / 3) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_values_l3182_318249


namespace NUMINAMATH_CALUDE_work_completion_time_l3182_318272

/-- If P people can complete a job in 20 days, then 2P people can complete half of the job in 5 days -/
theorem work_completion_time 
  (P : ℕ) -- number of people
  (full_work_time : ℕ := 20) -- time to complete full work with P people
  (h : P > 0) -- ensure P is positive
  : (2 * P) * 5 = P * full_work_time / 2 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l3182_318272


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l3182_318277

theorem rectangle_perimeter (x y : ℝ) (h1 : x^2 + y^2 = 900) (h2 : x * y = 200) :
  2 * (x + y) = 20 * Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l3182_318277


namespace NUMINAMATH_CALUDE_cubic_roots_sum_cubes_l3182_318241

theorem cubic_roots_sum_cubes (r s t : ℝ) : 
  (8 * r^3 + 1001 * r + 2008 = 0) →
  (8 * s^3 + 1001 * s + 2008 = 0) →
  (8 * t^3 + 1001 * t + 2008 = 0) →
  (r + s)^3 + (s + t)^3 + (t + r)^3 = 753 := by
  sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_cubes_l3182_318241


namespace NUMINAMATH_CALUDE_paper_strip_dimensions_l3182_318206

theorem paper_strip_dimensions 
  (a b c : ℕ) 
  (h1 : a > 0 ∧ b > 0 ∧ c > 0)  -- Positive dimensions
  (h2 : a * b + a * c + a * (b - a) + a^2 + a * (c - a) = 43)  -- Sum of areas is 43
  : a = 1 ∧ b + c = 22 := by
  sorry

end NUMINAMATH_CALUDE_paper_strip_dimensions_l3182_318206


namespace NUMINAMATH_CALUDE_red_balls_count_l3182_318290

theorem red_balls_count (total : ℕ) (prob : ℚ) : 
  total = 15 → 
  prob = 1 / 35 →
  (∃ r : ℕ, r ≤ total ∧ 
    prob = (r * (r - 1) * (r - 2)) / (total * (total - 1) * (total - 2))) →
  (∃ r : ℕ, r = 7 ∧ r ≤ total ∧ 
    prob = (r * (r - 1) * (r - 2)) / (total * (total - 1) * (total - 2))) :=
by sorry

end NUMINAMATH_CALUDE_red_balls_count_l3182_318290


namespace NUMINAMATH_CALUDE_library_visitors_month_length_l3182_318286

theorem library_visitors_month_length :
  ∀ (s : ℕ) (d : ℕ),
    s > 0 →  -- At least one Sunday
    s + d > 0 →  -- Total days in month is positive
    150 * s + 120 * d = 125 * (s + d) →  -- Equation balancing total visitors
    s = 5 ∧ d = 25 :=
by
  sorry

end NUMINAMATH_CALUDE_library_visitors_month_length_l3182_318286


namespace NUMINAMATH_CALUDE_number_of_girls_in_college_l3182_318246

theorem number_of_girls_in_college (total_students : ℕ) (boys_to_girls_ratio : ℚ) 
  (h1 : total_students = 416) 
  (h2 : boys_to_girls_ratio = 8 / 5) : 
  ∃ (girls : ℕ), girls = 160 ∧ 
    (girls : ℚ) * (1 + boys_to_girls_ratio) = total_students := by
  sorry

end NUMINAMATH_CALUDE_number_of_girls_in_college_l3182_318246


namespace NUMINAMATH_CALUDE_absolute_value_integral_l3182_318230

theorem absolute_value_integral : ∫ x in (0:ℝ)..2, |1 - x| = 1 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_integral_l3182_318230


namespace NUMINAMATH_CALUDE_max_partition_product_l3182_318205

def partition_product (p : List Nat) : Nat :=
  p.prod

def is_valid_partition (p : List Nat) : Prop :=
  p.sum = 25 ∧ p.all (· > 0) ∧ p.length ≤ 25

theorem max_partition_product :
  ∃ (max_p : List Nat), 
    is_valid_partition max_p ∧ 
    partition_product max_p = 8748 ∧
    ∀ (p : List Nat), is_valid_partition p → partition_product p ≤ 8748 := by
  sorry

end NUMINAMATH_CALUDE_max_partition_product_l3182_318205


namespace NUMINAMATH_CALUDE_largest_prime_factors_difference_l3182_318227

-- Define the number we're working with
def n : ℕ := 175616

-- State the theorem
theorem largest_prime_factors_difference (p q : ℕ) : 
  (Nat.Prime p ∧ Nat.Prime q ∧ p ∣ n ∧ q ∣ n ∧ 
   ∀ r, Nat.Prime r → r ∣ n → r ≤ p ∧ r ≤ q) → 
  p - q = 5 ∨ q - p = 5 :=
sorry

end NUMINAMATH_CALUDE_largest_prime_factors_difference_l3182_318227


namespace NUMINAMATH_CALUDE_star_composition_l3182_318293

/-- Define the binary operation ★ -/
def star (x y : ℝ) : ℝ := x^2 - 2*y + 1

/-- Theorem: For any real number k, k ★ (k ★ k) = -k^2 + 4k - 1 -/
theorem star_composition (k : ℝ) : star k (star k k) = -k^2 + 4*k - 1 := by
  sorry

end NUMINAMATH_CALUDE_star_composition_l3182_318293


namespace NUMINAMATH_CALUDE_remaining_pages_l3182_318202

/-- Calculate the remaining pages of books after some are lost -/
theorem remaining_pages (initial_books : ℕ) (pages_per_book : ℕ) (lost_books : ℕ) 
  (h1 : initial_books ≥ lost_books) :
  (initial_books - lost_books) * pages_per_book = 
  initial_books * pages_per_book - lost_books * pages_per_book :=
by sorry

#check remaining_pages

end NUMINAMATH_CALUDE_remaining_pages_l3182_318202


namespace NUMINAMATH_CALUDE_fourth_power_difference_l3182_318273

theorem fourth_power_difference (a b : ℝ) : 
  (a - b)^4 = a^4 - 4*a^3*b + 6*a^2*b^2 - 4*a*b^3 + b^4 :=
by
  sorry

end NUMINAMATH_CALUDE_fourth_power_difference_l3182_318273


namespace NUMINAMATH_CALUDE_class_trip_collection_l3182_318256

/-- Calculates the total amount collected by a class for a field trip. -/
def total_collection (num_students : ℕ) (contribution_per_student : ℕ) (num_weeks : ℕ) : ℕ :=
  num_students * contribution_per_student * num_weeks

/-- Proves that a class of 30 students, each contributing $2 per week for 8 weeks, will collect $480. -/
theorem class_trip_collection :
  total_collection 30 2 8 = 480 := by
  sorry

#eval total_collection 30 2 8

end NUMINAMATH_CALUDE_class_trip_collection_l3182_318256


namespace NUMINAMATH_CALUDE_susan_added_35_oranges_l3182_318264

/-- The number of oranges Susan put into the box -/
def susans_oranges (initial_oranges final_oranges : ℝ) : ℝ :=
  final_oranges - initial_oranges

/-- Proof that Susan put 35.0 oranges into the box -/
theorem susan_added_35_oranges :
  susans_oranges 55.0 90.0 = 35.0 := by
  sorry

end NUMINAMATH_CALUDE_susan_added_35_oranges_l3182_318264


namespace NUMINAMATH_CALUDE_ariella_meetings_percentage_l3182_318235

theorem ariella_meetings_percentage : 
  let work_day_hours : ℝ := 8
  let first_meeting_minutes : ℝ := 60
  let second_meeting_factor : ℝ := 1.5
  let work_day_minutes : ℝ := work_day_hours * 60
  let second_meeting_minutes : ℝ := second_meeting_factor * first_meeting_minutes
  let total_meeting_minutes : ℝ := first_meeting_minutes + second_meeting_minutes
  let meeting_percentage : ℝ := (total_meeting_minutes / work_day_minutes) * 100
  meeting_percentage = 31.25 := by sorry

end NUMINAMATH_CALUDE_ariella_meetings_percentage_l3182_318235


namespace NUMINAMATH_CALUDE_function_decreasing_implies_a_range_a_in_range_l3182_318289

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 0 then a^x else (a - 3) * x + 4 * a

-- State the theorem
theorem function_decreasing_implies_a_range (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (f a x₁ - f a x₂) / (x₁ - x₂) < 0) →
  0 < a ∧ a ≤ 1/4 := by
  sorry

-- Define the set of possible values for a
def a_range : Set ℝ := { a | 0 < a ∧ a ≤ 1/4 }

-- State the final theorem
theorem a_in_range :
  ∀ a : ℝ, (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (f a x₁ - f a x₂) / (x₁ - x₂) < 0) →
  a ∈ a_range := by
  sorry

end NUMINAMATH_CALUDE_function_decreasing_implies_a_range_a_in_range_l3182_318289


namespace NUMINAMATH_CALUDE_exponential_function_fixed_point_l3182_318262

/-- The function f(x) = a^(x-2) + 1 passes through the point (2, 2) for any a > 0 and a ≠ 1 -/
theorem exponential_function_fixed_point (a : ℝ) (ha : a > 0) (ha1 : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(x - 2) + 1
  f 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_exponential_function_fixed_point_l3182_318262


namespace NUMINAMATH_CALUDE_problem_solution_l3182_318257

/-- Permutations of n items taken r at a time -/
def permutations (n : ℕ) (r : ℕ) : ℕ := sorry

/-- Combinations of n items taken r at a time -/
def combinations (n : ℕ) (r : ℕ) : ℕ := sorry

theorem problem_solution (r : ℕ) (k : ℕ) : 
  permutations 32 r = k * combinations 32 r → k = 720 → r = 6 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3182_318257


namespace NUMINAMATH_CALUDE_increase_in_average_weight_l3182_318223

/-- Theorem: Increase in average weight when replacing a person in a group -/
theorem increase_in_average_weight 
  (group_size : ℕ) 
  (old_weight new_weight : ℝ) 
  (h1 : group_size = 8)
  (h2 : old_weight = 55)
  (h3 : new_weight = 87) :
  (new_weight - old_weight) / group_size = 4 := by
  sorry

end NUMINAMATH_CALUDE_increase_in_average_weight_l3182_318223


namespace NUMINAMATH_CALUDE_total_days_1996_to_2000_l3182_318213

def isLeapYear (year : Nat) : Bool :=
  (year % 4 == 0 && year % 100 ≠ 0) || (year % 400 == 0)

def daysInYear (year : Nat) : Nat :=
  if isLeapYear year then 366 else 365

def totalDays (startYear endYear : Nat) : Nat :=
  (List.range (endYear - startYear + 1)).map (fun i => daysInYear (startYear + i))
    |> List.sum

theorem total_days_1996_to_2000 :
  totalDays 1996 2000 = 1827 := by sorry

end NUMINAMATH_CALUDE_total_days_1996_to_2000_l3182_318213


namespace NUMINAMATH_CALUDE_fraction_equality_l3182_318282

theorem fraction_equality : (4^3 : ℝ) / (10^2 - 6^2) = 1 := by sorry

end NUMINAMATH_CALUDE_fraction_equality_l3182_318282


namespace NUMINAMATH_CALUDE_system_solutions_l3182_318243

/-- The system of equations has only three real solutions -/
theorem system_solutions (a b c : ℝ) : 
  (2 * a - b = a^2 * b) ∧ 
  (2 * b - c = b^2 * c) ∧ 
  (2 * c - a = c^2 * a) → 
  ((a = -1 ∧ b = -1 ∧ c = -1) ∨ 
   (a = 0 ∧ b = 0 ∧ c = 0) ∨ 
   (a = 1 ∧ b = 1 ∧ c = 1)) :=
by sorry

end NUMINAMATH_CALUDE_system_solutions_l3182_318243


namespace NUMINAMATH_CALUDE_average_of_numbers_l3182_318255

def numbers : List ℕ := [54, 55, 57, 58, 59, 62, 62, 63, 65, 65]

theorem average_of_numbers : (numbers.sum : ℚ) / numbers.length = 60 := by
  sorry

end NUMINAMATH_CALUDE_average_of_numbers_l3182_318255


namespace NUMINAMATH_CALUDE_box_third_side_length_l3182_318208

/-- Proves that the third side of a rectangular box is 6.75 cm given specific conditions -/
theorem box_third_side_length (num_cubes : ℕ) (cube_volume : ℝ) (side1 side2 : ℝ) :
  num_cubes = 24 →
  cube_volume = 27 →
  side1 = 8 →
  side2 = 12 →
  (num_cubes : ℝ) * cube_volume = side1 * side2 * 6.75 :=
by sorry

end NUMINAMATH_CALUDE_box_third_side_length_l3182_318208


namespace NUMINAMATH_CALUDE_reflection_over_x_axis_l3182_318253

/-- Reflects a point over the x-axis -/
def reflect_over_x_axis (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

/-- The reflection of (-4, 3) over the x-axis is (-4, -3) -/
theorem reflection_over_x_axis :
  reflect_over_x_axis (-4, 3) = (-4, -3) := by
  sorry

end NUMINAMATH_CALUDE_reflection_over_x_axis_l3182_318253


namespace NUMINAMATH_CALUDE_bathroom_floor_space_l3182_318285

/-- Calculates the available floor space in an L-shaped bathroom with a pillar -/
theorem bathroom_floor_space
  (main_width : ℕ) (main_length : ℕ)
  (alcove_width : ℕ) (alcove_depth : ℕ)
  (pillar_width : ℕ) (pillar_length : ℕ)
  (tile_size : ℚ) :
  main_width = 15 →
  main_length = 25 →
  alcove_width = 10 →
  alcove_depth = 8 →
  pillar_width = 3 →
  pillar_length = 5 →
  tile_size = 1/2 →
  (main_width * main_length * tile_size^2 +
   alcove_width * alcove_depth * tile_size^2 -
   pillar_width * pillar_length * tile_size^2) = 110 :=
by sorry

end NUMINAMATH_CALUDE_bathroom_floor_space_l3182_318285


namespace NUMINAMATH_CALUDE_determine_contents_l3182_318280

-- Define the colors of balls
inductive Color
| White
| Black

-- Define the types of boxes
inductive BoxType
| TwoWhite
| TwoBlack
| OneWhiteOneBlack

-- Define a box with a label and contents
structure Box where
  label : BoxType
  contents : BoxType

-- Define the problem setup
def problem_setup : Prop :=
  ∃ (box1 box2 box3 : Box),
    -- Three boxes with different labels
    box1.label ≠ box2.label ∧ box2.label ≠ box3.label ∧ box1.label ≠ box3.label ∧
    -- Contents don't match labels
    box1.contents ≠ box1.label ∧ box2.contents ≠ box2.label ∧ box3.contents ≠ box3.label ∧
    -- One box has two white balls, one has two black balls, and one has one of each
    (box1.contents = BoxType.TwoWhite ∧ box2.contents = BoxType.TwoBlack ∧ box3.contents = BoxType.OneWhiteOneBlack) ∨
    (box1.contents = BoxType.TwoWhite ∧ box2.contents = BoxType.OneWhiteOneBlack ∧ box3.contents = BoxType.TwoBlack) ∨
    (box1.contents = BoxType.TwoBlack ∧ box2.contents = BoxType.TwoWhite ∧ box3.contents = BoxType.OneWhiteOneBlack) ∨
    (box1.contents = BoxType.TwoBlack ∧ box2.contents = BoxType.OneWhiteOneBlack ∧ box3.contents = BoxType.TwoWhite) ∨
    (box1.contents = BoxType.OneWhiteOneBlack ∧ box2.contents = BoxType.TwoWhite ∧ box3.contents = BoxType.TwoBlack) ∨
    (box1.contents = BoxType.OneWhiteOneBlack ∧ box2.contents = BoxType.TwoBlack ∧ box3.contents = BoxType.TwoWhite)

-- Define the theorem
theorem determine_contents (setup : problem_setup) :
  ∃ (box : Box) (c : Color),
    box.label = BoxType.OneWhiteOneBlack →
    (c = Color.White → box.contents = BoxType.TwoWhite) ∧
    (c = Color.Black → box.contents = BoxType.TwoBlack) :=
sorry

end NUMINAMATH_CALUDE_determine_contents_l3182_318280


namespace NUMINAMATH_CALUDE_perpendicular_to_two_planes_implies_parallel_l3182_318296

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular relation between a line and a plane
variable (perpendicular : Line → Plane → Prop)

-- Define the parallel relation between two planes
variable (parallel : Plane → Plane → Prop)

-- State the theorem
theorem perpendicular_to_two_planes_implies_parallel 
  (α β : Plane) (a : Line) 
  (h_diff : α ≠ β) 
  (h_perp_α : perpendicular a α) 
  (h_perp_β : perpendicular a β) : 
  parallel α β :=
sorry

end NUMINAMATH_CALUDE_perpendicular_to_two_planes_implies_parallel_l3182_318296


namespace NUMINAMATH_CALUDE_expression_simplification_l3182_318238

theorem expression_simplification (x : ℝ) (h : x = Real.sqrt 3 - 2) :
  (1 + 1 / (x - 2)) * ((x^2 - 4) / (x - 1)) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3182_318238


namespace NUMINAMATH_CALUDE_jordana_is_80_l3182_318239

/-- Jennifer's age in 10 years -/
def jennifer_future_age : ℕ := 30

/-- The number of years in the future we're considering -/
def years_ahead : ℕ := 10

/-- Jordana's age relative to Jennifer's in the future -/
def jordana_age_multiplier : ℕ := 3

/-- Calculate Jordana's current age based on the given conditions -/
def jordana_current_age : ℕ :=
  jennifer_future_age * jordana_age_multiplier - years_ahead

/-- Theorem stating that Jordana's current age is 80 years old -/
theorem jordana_is_80 : jordana_current_age = 80 := by
  sorry

end NUMINAMATH_CALUDE_jordana_is_80_l3182_318239


namespace NUMINAMATH_CALUDE_midpoint_x_coordinate_sum_l3182_318276

theorem midpoint_x_coordinate_sum (a b c : ℝ) (h : a + b + c = 12) :
  (a + b) / 2 + (a + c) / 2 + (b + c) / 2 = 12 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_x_coordinate_sum_l3182_318276


namespace NUMINAMATH_CALUDE_infinitely_many_integer_triangles_60_l3182_318265

/-- A triangle with integer sides and a 60° angle -/
structure IntegerTriangle60 where
  x : ℤ
  y : ℤ
  z : ℤ
  angle_60 : x^2 + y^2 - x*y = z^2  -- Law of cosines for 60° angle

/-- Parameters for generating integer triangles with 60° angle -/
structure TriangleParams where
  m : ℤ
  n : ℤ
  Δ : ℤ
  Δ_pos : Δ > 0

/-- Function to generate an IntegerTriangle60 from TriangleParams -/
def generateTriangle (p : TriangleParams) : IntegerTriangle60 :=
  { x := (p.m^2 - p.n^2) / p.Δ
    y := p.m * (p.m - 2*p.n) / p.Δ
    z := (p.m^2 - p.m*p.n + p.n^2) / p.Δ
    angle_60 := by sorry }

/-- Theorem stating that there are infinitely many integer triangles with a 60° angle -/
theorem infinitely_many_integer_triangles_60 :
  ∀ k : ℕ, ∃ (triangles : Fin k → IntegerTriangle60),
    ∀ i j : Fin k, i ≠ j → triangles i ≠ triangles j :=
by sorry

end NUMINAMATH_CALUDE_infinitely_many_integer_triangles_60_l3182_318265


namespace NUMINAMATH_CALUDE_sum_of_fractions_equals_25_12_l3182_318266

theorem sum_of_fractions_equals_25_12 : 
  (3 + 6 + 9) / (4 + 8 + 12) + (4 + 8 + 12) / (3 + 6 + 9) = 25 / 12 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_equals_25_12_l3182_318266


namespace NUMINAMATH_CALUDE_non_shaded_perimeter_l3182_318212

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Calculates the area of a rectangle -/
def area (r : Rectangle) : ℝ := r.width * r.height

/-- Calculates the perimeter of a rectangle -/
def perimeter (r : Rectangle) : ℝ := 2 * (r.width + r.height)

theorem non_shaded_perimeter (large : Rectangle) (shaded : Rectangle) :
  large.width = 12 ∧
  large.height = 10 ∧
  shaded.width = 5 ∧
  shaded.height = 11 ∧
  area shaded = 55 →
  perimeter large + perimeter shaded - 2 * (shaded.width + shaded.height) = 48 := by
  sorry

end NUMINAMATH_CALUDE_non_shaded_perimeter_l3182_318212


namespace NUMINAMATH_CALUDE_sum_reciprocals_l3182_318294

theorem sum_reciprocals (a b c d : ℝ) (ω : ℂ) :
  a ≠ -1 → b ≠ -1 → c ≠ -1 → d ≠ -1 →
  ω^3 = 1 →
  ω ≠ 1 →
  (1 / (a + ω)) + (1 / (b + ω)) + (1 / (c + ω)) + (1 / (d + ω)) = 3 / ω →
  (1 / (a + 1)) + (1 / (b + 1)) + (1 / (c + 1)) + (1 / (d + 1)) = 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_reciprocals_l3182_318294


namespace NUMINAMATH_CALUDE_alan_phone_price_l3182_318258

theorem alan_phone_price (john_price : ℝ) (percentage : ℝ) (alan_price : ℝ) :
  john_price = 2040 →
  percentage = 0.02 →
  john_price = alan_price * (1 + percentage) →
  alan_price = 1999.20 := by
sorry

end NUMINAMATH_CALUDE_alan_phone_price_l3182_318258
