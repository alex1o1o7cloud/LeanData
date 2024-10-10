import Mathlib

namespace trigonometric_properties_l3142_314236

theorem trigonometric_properties :
  (∀ x, 2 * Real.sin (2 * x - π / 3) = 2 * Real.sin (2 * (5 * π / 6 - x) - π / 3)) ∧
  (∀ x, Real.tan x = -Real.tan (π - x)) ∧
  (∃ x₁ x₂, x₁ > 0 ∧ x₂ > 0 ∧ x₁ < π / 2 ∧ x₂ < π / 2 ∧ x₁ > x₂ ∧ Real.sin x₁ < Real.sin x₂) ∧
  (∀ x₁ x₂, Real.sin (2 * x₁ - π / 4) = Real.sin (2 * x₂ - π / 4) →
    (∃ k : ℤ, x₁ - x₂ = k * π ∨ x₁ + x₂ = k * π + 3 * π / 4)) :=
by sorry

end trigonometric_properties_l3142_314236


namespace board_length_proof_l3142_314297

/-- Given a board cut into two pieces, where one piece is twice the length of the other
    and the shorter piece is 23 inches long, the total length of the board is 69 inches. -/
theorem board_length_proof (shorter_piece longer_piece total_length : ℕ) :
  shorter_piece = 23 →
  longer_piece = 2 * shorter_piece →
  total_length = shorter_piece + longer_piece →
  total_length = 69 := by
  sorry

#check board_length_proof

end board_length_proof_l3142_314297


namespace kindergarten_class_size_l3142_314224

theorem kindergarten_class_size 
  (num_groups : ℕ) 
  (time_per_student : ℕ) 
  (time_per_group : ℕ) 
  (h1 : num_groups = 3)
  (h2 : time_per_student = 4)
  (h3 : time_per_group = 24) :
  num_groups * (time_per_group / time_per_student) = 18 :=
by
  sorry

end kindergarten_class_size_l3142_314224


namespace distribute_five_students_three_classes_l3142_314214

/-- The number of ways to distribute students into classes -/
def distribute_students (total_students : ℕ) (num_classes : ℕ) (pre_assigned : ℕ) : ℕ :=
  sorry

/-- Theorem stating the correct number of distributions for the given problem -/
theorem distribute_five_students_three_classes : 
  distribute_students 5 3 1 = 56 := by sorry

end distribute_five_students_three_classes_l3142_314214


namespace range_of_a_l3142_314203

-- Define the set of real numbers where the expression is meaningful
def MeaningfulSet : Set ℝ :=
  {a : ℝ | a - 2 ≥ 0 ∧ a ≠ 4}

-- Theorem stating the range of values for a
theorem range_of_a : MeaningfulSet = Set.Icc 2 4 ∪ Set.Ioi 4 := by
  sorry

end range_of_a_l3142_314203


namespace function_ranges_l3142_314272

-- Define the functions f and g
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - a*x - 2
def g (a : ℝ) (x : ℝ) : ℝ := -x^2 + x + a

-- State the theorem
theorem function_ranges :
  ∀ a : ℝ,
  (f a (-1) = 0) →
  (∀ x₁ ∈ Set.Icc (1/4 : ℝ) 1, ∃ x₂ ∈ Set.Icc 1 2, g a x₁ > f a x₂ + 3) →
  (Set.range (f a) = Set.Ici (-9/4 : ℝ)) ∧
  (a ∈ Set.Ioi 1) :=
by sorry

end function_ranges_l3142_314272


namespace liars_on_black_chairs_l3142_314235

def room_scenario (total_people : ℕ) (initial_black_claims : ℕ) (final_white_claims : ℕ) : Prop :=
  -- Total number of people is positive
  total_people > 0 ∧
  -- Initially, all people claim to be on black chairs
  initial_black_claims = total_people ∧
  -- After rearrangement, some people claim to be on white chairs
  final_white_claims > 0 ∧ final_white_claims < total_people

theorem liars_on_black_chairs 
  (total_people : ℕ) 
  (initial_black_claims : ℕ) 
  (final_white_claims : ℕ) 
  (h : room_scenario total_people initial_black_claims final_white_claims) :
  -- The number of liars on black chairs after rearrangement
  (final_white_claims / 2) = 8 :=
sorry

end liars_on_black_chairs_l3142_314235


namespace chris_candy_distribution_l3142_314256

/-- The number of friends Chris has -/
def num_friends : ℕ := 35

/-- The number of candy pieces each friend receives -/
def candy_per_friend : ℕ := 12

/-- The total number of candy pieces Chris gave to his friends -/
def total_candy : ℕ := num_friends * candy_per_friend

theorem chris_candy_distribution :
  total_candy = 420 :=
by sorry

end chris_candy_distribution_l3142_314256


namespace root_existence_l3142_314210

theorem root_existence : ∃ x : ℝ, 3 < x ∧ x < 4 ∧ Real.log x = 8 - 2 * x := by
  sorry

end root_existence_l3142_314210


namespace five_digit_square_number_l3142_314220

theorem five_digit_square_number : ∃! n : ℕ, 
  (n * n ≥ 10000) ∧ 
  (n * n < 100000) ∧ 
  (n * n / 10000 = 2) ∧ 
  ((n * n / 10) % 10 = 5) ∧ 
  (∃ m : ℕ, n * n = m * m) :=
by sorry

end five_digit_square_number_l3142_314220


namespace octal_135_to_binary_l3142_314221

/-- Converts an octal digit to its binary representation --/
def octal_to_binary_digit (d : Nat) : Nat :=
  match d with
  | 0 => 0
  | 1 => 1
  | 2 => 10
  | 3 => 11
  | 4 => 100
  | 5 => 101
  | 6 => 110
  | 7 => 111
  | _ => 0  -- Default case, should not occur for valid octal digits

/-- Converts an octal number to its binary representation --/
def octal_to_binary (n : Nat) : Nat :=
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let ones := n % 10
  octal_to_binary_digit hundreds * 1000000 +
  octal_to_binary_digit tens * 1000 +
  octal_to_binary_digit ones

theorem octal_135_to_binary : octal_to_binary 135 = 1011101 := by
  sorry

end octal_135_to_binary_l3142_314221


namespace cubic_sum_prime_power_l3142_314211

theorem cubic_sum_prime_power (a b p n : ℕ) : 
  0 < a ∧ 0 < b ∧ 0 < p ∧ 0 < n ∧ 
  Nat.Prime p ∧ 
  a^3 + b^3 = p^n →
  (∃ k : ℕ, (a = 2^k ∧ b = 2^k ∧ p = 2 ∧ n = 3*k + 1) ∨
             (a = 2*(3^k) ∧ b = 3^k ∧ p = 3 ∧ n = 3*k + 2) ∨
             (a = 3^k ∧ b = 2*(3^k) ∧ p = 3 ∧ n = 3*k + 2)) :=
by sorry

end cubic_sum_prime_power_l3142_314211


namespace cone_height_l3142_314296

/-- Given a cone with base radius 1 and central angle of the unfolded side view 2/3π,
    the height of the cone is 2√2. -/
theorem cone_height (r : ℝ) (θ : ℝ) (h : ℝ) : 
  r = 1 → θ = (2/3) * Real.pi → h = 2 * Real.sqrt 2 := by
  sorry

end cone_height_l3142_314296


namespace staff_discount_price_l3142_314255

/-- Given a dress with original price d, after a 35% discount and an additional 30% staff discount,
    the final price is 0.455 times the original price. -/
theorem staff_discount_price (d : ℝ) : d * (1 - 0.35) * (1 - 0.30) = d * 0.455 := by
  sorry

#check staff_discount_price

end staff_discount_price_l3142_314255


namespace baseball_football_fans_l3142_314273

theorem baseball_football_fans (total : ℕ) (baseball_only : ℕ) (football_only : ℕ) (neither : ℕ) 
  (h1 : total = 16)
  (h2 : baseball_only = 2)
  (h3 : football_only = 3)
  (h4 : neither = 6) :
  total - baseball_only - football_only - neither = 5 := by
sorry

end baseball_football_fans_l3142_314273


namespace quadratic_increasing_l3142_314284

-- Define the quadratic function
def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem quadratic_increasing (a b c : ℝ) 
  (h1 : f a b c 0 = f a b c 6) 
  (h2 : f a b c 6 < f a b c 7) :
  ∀ x y, 3 < x → x < y → f a b c x < f a b c y := by
  sorry

end quadratic_increasing_l3142_314284


namespace ellipse_minimum_area_l3142_314230

/-- An ellipse containing two specific circles has a minimum area of 16π -/
theorem ellipse_minimum_area :
  ∀ a b : ℝ,
  (∀ x y : ℝ, x^2 / (4*a^2) + y^2 / (4*b^2) = 1 →
    ((x - 2)^2 + y^2 ≤ 4 ∧ (x + 2)^2 + y^2 ≤ 4)) →
  4 * π * a * b ≥ 16 * π :=
by sorry

end ellipse_minimum_area_l3142_314230


namespace cubic_equation_value_l3142_314258

theorem cubic_equation_value (m : ℝ) (h : m^2 + m - 1 = 0) :
  m^3 + 2*m^2 + 2006 = 2007 := by
  sorry

end cubic_equation_value_l3142_314258


namespace tank_plastering_cost_l3142_314205

/-- The cost of plastering a rectangular tank's walls and bottom -/
theorem tank_plastering_cost
  (length width depth : ℝ)
  (cost_per_sq_m_paise : ℝ)
  (h_length : length = 40)
  (h_width : width = 18)
  (h_depth : depth = 10)
  (h_cost : cost_per_sq_m_paise = 125) :
  let bottom_area := length * width
  let perimeter := 2 * (length + width)
  let wall_area := perimeter * depth
  let total_area := bottom_area + wall_area
  let cost_per_sq_m_rupees := cost_per_sq_m_paise / 100
  total_area * cost_per_sq_m_rupees = 2350 :=
by
  sorry


end tank_plastering_cost_l3142_314205


namespace milk_packets_problem_l3142_314227

theorem milk_packets_problem (n : ℕ) 
  (h1 : n > 2)
  (h2 : n * 20 = (n - 2) * 12 + 2 * 32) : 
  n = 5 := by
sorry

end milk_packets_problem_l3142_314227


namespace equation_solution_l3142_314233

theorem equation_solution (y : ℝ) : 
  ∃ x : ℝ, 19 * (x + y) + 17 = 19 * (-x + y) - 21 ∧ x = -21/38 := by
sorry

end equation_solution_l3142_314233


namespace red_paint_amount_l3142_314232

/-- Given a paint mixture with a ratio of red to white as 5:7, 
    if 21 quarts of white paint are used, then 15 quarts of red paint should be used. -/
theorem red_paint_amount (red white : ℚ) : 
  (red / white = 5 / 7) → (white = 21) → (red = 15) := by
  sorry

end red_paint_amount_l3142_314232


namespace license_plate_increase_l3142_314219

/-- The number of possible characters for letters in the new scheme -/
def new_letter_options : ℕ := 30

/-- The number of letters in the English alphabet -/
def alphabet_size : ℕ := 26

/-- The number of digits (0-9) -/
def digit_options : ℕ := 10

/-- The number of letters in the new license plate scheme -/
def new_letter_count : ℕ := 2

/-- The number of digits in the new license plate scheme -/
def new_digit_count : ℕ := 5

/-- The number of letters in the previous license plate scheme -/
def old_letter_count : ℕ := 3

/-- The number of digits in the previous license plate scheme -/
def old_digit_count : ℕ := 3

theorem license_plate_increase :
  (new_letter_options ^ new_letter_count * digit_options ^ new_digit_count) /
  (alphabet_size ^ old_letter_count * digit_options ^ old_digit_count) =
  (900 : ℚ) / 17576 * 100 := by
  sorry

end license_plate_increase_l3142_314219


namespace cos_a_minus_b_l3142_314215

theorem cos_a_minus_b (A B : ℝ) 
  (h1 : Real.sin A + Real.sin B = 3/2) 
  (h2 : Real.cos A + Real.cos B = 1) : 
  Real.cos (A - B) = 5/8 := by
sorry

end cos_a_minus_b_l3142_314215


namespace simplify_expression_l3142_314208

theorem simplify_expression : 
  2 - (2 / (2 + Real.sqrt 5)) + (2 / (2 - Real.sqrt 5)) = 2 - 4 * Real.sqrt 5 := by
  sorry

end simplify_expression_l3142_314208


namespace solution_set_a_eq_one_solution_set_is_real_l3142_314207

-- Define the quadratic function
def f (a x : ℝ) : ℝ := a * x^2 + a * x - 2

-- Part 1: Solution set for a = 1
theorem solution_set_a_eq_one :
  {x : ℝ | f 1 x ≤ 0} = {x : ℝ | -2 ≤ x ∧ x ≤ 1} := by sorry

-- Part 2: Conditions for solution set to be ℝ
theorem solution_set_is_real :
  ∀ a : ℝ, (∀ x : ℝ, f a x ≤ 0) ↔ -8 ≤ a ∧ a ≤ 0 := by sorry

end solution_set_a_eq_one_solution_set_is_real_l3142_314207


namespace f_one_root_m_range_l3142_314216

/-- A cubic function with parameter m -/
def f (m : ℝ) (x : ℝ) : ℝ := x^3 - 3*x + m

/-- The theorem stating the range of m for which f has exactly one real root -/
theorem f_one_root_m_range (m : ℝ) :
  (∃! x, f m x = 0) ↔ m < -2 ∨ m > 2 := by
  sorry

end f_one_root_m_range_l3142_314216


namespace square_perimeter_relation_l3142_314251

/-- Given a square A with perimeter 24 cm and a square B with area equal to one-fourth the area of square A, prove that the perimeter of square B is 12 cm. -/
theorem square_perimeter_relation (A B : ℝ → ℝ → Prop) : 
  (∃ a, ∀ x y, A x y ↔ (x = 0 ∨ x = a) ∧ (y = 0 ∨ y = a) ∧ 4 * a = 24) →
  (∃ b, ∀ x y, B x y ↔ (x = 0 ∨ x = b) ∧ (y = 0 ∨ y = b) ∧ b^2 = (a^2 / 4)) →
  (∃ p, p = 4 * b ∧ p = 12) :=
sorry

end square_perimeter_relation_l3142_314251


namespace library_books_checkout_l3142_314260

theorem library_books_checkout (fiction_books : ℕ) (nonfiction_ratio fiction_ratio : ℕ) : 
  fiction_books = 24 → 
  nonfiction_ratio = 7 →
  fiction_ratio = 6 →
  ∃ (total_books : ℕ), total_books = fiction_books + (fiction_books * nonfiction_ratio) / fiction_ratio ∧ total_books = 52 :=
by
  sorry

end library_books_checkout_l3142_314260


namespace smallest_angle_in_right_triangle_l3142_314252

theorem smallest_angle_in_right_triangle (α β γ : ℝ) : 
  α = 90 → β = 55 → α + β + γ = 180 → min α (min β γ) = 35 := by
  sorry

end smallest_angle_in_right_triangle_l3142_314252


namespace prob_score_over_14_is_0_3_expected_value_is_13_6_l3142_314265

-- Define the success rates and point values
def three_week_success_rate : ℝ := 0.7
def four_week_success_rate : ℝ := 0.3
def three_week_success_points : ℝ := 8
def three_week_failure_points : ℝ := 4
def four_week_success_points : ℝ := 15
def four_week_failure_points : ℝ := 6

-- Define the probability of scoring more than 14 points
-- in a sequence of a three-week jump followed by a four-week jump
def prob_score_over_14 : ℝ :=
  three_week_success_rate * four_week_success_rate +
  (1 - three_week_success_rate) * four_week_success_rate

-- Define the expected value of the total score for two consecutive three-week jumps
def expected_value_two_three_week_jumps : ℝ :=
  (1 - three_week_success_rate)^2 * (2 * three_week_failure_points) +
  2 * three_week_success_rate * (1 - three_week_success_rate) * (three_week_success_points + three_week_failure_points) +
  three_week_success_rate^2 * (2 * three_week_success_points)

-- Theorem statements
theorem prob_score_over_14_is_0_3 : prob_score_over_14 = 0.3 := by sorry

theorem expected_value_is_13_6 : expected_value_two_three_week_jumps = 13.6 := by sorry

end prob_score_over_14_is_0_3_expected_value_is_13_6_l3142_314265


namespace cos_sum_fifteenths_l3142_314240

theorem cos_sum_fifteenths : Real.cos (4 * π / 15) + Real.cos (10 * π / 15) + Real.cos (14 * π / 15) = 1 := by
  sorry

end cos_sum_fifteenths_l3142_314240


namespace total_apples_is_33_l3142_314271

/-- The number of apples picked by each person -/
structure ApplePickers where
  mike : ℕ
  nancy : ℕ
  keith : ℕ
  jennifer : ℕ
  tom : ℕ
  stacy : ℕ

/-- The total number of apples picked -/
def total_apples (pickers : ApplePickers) : ℕ :=
  pickers.mike + pickers.nancy + pickers.keith + pickers.jennifer + pickers.tom + pickers.stacy

/-- Theorem stating that the total number of apples picked is 33 -/
theorem total_apples_is_33 (pickers : ApplePickers) 
    (h_mike : pickers.mike = 7)
    (h_nancy : pickers.nancy = 3)
    (h_keith : pickers.keith = 6)
    (h_jennifer : pickers.jennifer = 5)
    (h_tom : pickers.tom = 8)
    (h_stacy : pickers.stacy = 4) : 
  total_apples pickers = 33 := by
  sorry

end total_apples_is_33_l3142_314271


namespace negation_of_existence_l3142_314295

theorem negation_of_existence (p : Prop) :
  (¬ ∃ x : ℝ, x^2 - x + 1 < 0) ↔ (∀ x : ℝ, x^2 - x + 1 ≥ 0) := by
  sorry

end negation_of_existence_l3142_314295


namespace points_three_units_from_negative_three_l3142_314248

def distance (x y : ℝ) : ℝ := |x - y|

theorem points_three_units_from_negative_three :
  ∀ x : ℝ, distance x (-3) = 3 ↔ x = 0 ∨ x = -6 := by
  sorry

end points_three_units_from_negative_three_l3142_314248


namespace square_partition_exists_l3142_314244

/-- A square is a four-sided polygon with all sides equal and all angles equal to 90 degrees. -/
structure Square where
  sides : Fin 4 → ℝ
  angles : Fin 4 → ℝ
  sides_equal : ∀ i j, sides i = sides j
  angles_right : ∀ i, angles i = 90

/-- A convex pentagon is a five-sided polygon with all interior angles less than 180 degrees. -/
structure ConvexPentagon where
  sides : Fin 5 → ℝ
  angles : Fin 5 → ℝ
  angles_convex : ∀ i, angles i < 180

/-- A partition of a square into convex pentagons -/
structure SquarePartition where
  square : Square
  pentagons : List ConvexPentagon
  is_partition : Square → List ConvexPentagon → Prop

/-- Theorem: There exists a partition of a square into a finite number of convex pentagons -/
theorem square_partition_exists : ∃ p : SquarePartition, p.pentagons.length > 0 := by
  sorry

end square_partition_exists_l3142_314244


namespace initial_outlay_is_10000_l3142_314246

/-- Calculates the profit for a horseshoe manufacturing company --/
def horseshoe_profit (initial_outlay : ℝ) (sets_produced : ℕ) : ℝ :=
  let manufacturing_cost := initial_outlay + 20 * sets_produced
  let revenue := 50 * sets_produced
  revenue - manufacturing_cost

/-- Proves that the initial outlay is $10,000 given the conditions --/
theorem initial_outlay_is_10000 :
  ∃ (initial_outlay : ℝ),
    horseshoe_profit initial_outlay 500 = 5000 ∧
    initial_outlay = 10000 :=
by
  sorry

end initial_outlay_is_10000_l3142_314246


namespace algebraic_expression_equality_l3142_314218

theorem algebraic_expression_equality (x : ℝ) (h : x^2 + 2*x - 3 = 7) : 
  2*x^2 + 4*x + 1 = 21 := by
  sorry

end algebraic_expression_equality_l3142_314218


namespace sculpture_exposed_area_l3142_314225

/-- Represents the sculpture with its properties --/
structure Sculpture where
  cubeEdge : Real
  bottomLayerCubes : Nat
  middleLayerCubes : Nat
  topLayerCubes : Nat
  submersionRatio : Real

/-- Calculates the exposed surface area of the sculpture --/
def exposedSurfaceArea (s : Sculpture) : Real :=
  sorry

/-- Theorem stating that the exposed surface area of the given sculpture is 12.75 square meters --/
theorem sculpture_exposed_area :
  let s : Sculpture := {
    cubeEdge := 0.5,
    bottomLayerCubes := 16,
    middleLayerCubes := 9,
    topLayerCubes := 1,
    submersionRatio := 0.5
  }
  exposedSurfaceArea s = 12.75 := by
  sorry

end sculpture_exposed_area_l3142_314225


namespace third_purchase_total_l3142_314243

/-- Represents the clothing purchase scenario -/
structure ClothingPurchase where
  initialCost : ℕ
  typeAIncrease : ℕ
  typeBIncrease : ℕ
  secondCostIncrease : ℕ
  averageIncrease : ℕ
  profitMargin : ℚ
  thirdTypeBCost : ℕ

/-- Theorem stating the total number of pieces in the third purchase -/
theorem third_purchase_total (cp : ClothingPurchase)
  (h1 : cp.initialCost = 3600)
  (h2 : cp.typeAIncrease = 20)
  (h3 : cp.typeBIncrease = 5)
  (h4 : cp.secondCostIncrease = 400)
  (h5 : cp.averageIncrease = 8)
  (h6 : cp.profitMargin = 35 / 100)
  (h7 : cp.thirdTypeBCost = 3000) :
  ∃ (x y : ℕ),
    x + y = 50 ∧
    20 * x + 5 * y = 400 ∧
    8 * (x + y) = 400 ∧
    (3600 + 400) * (1 + cp.profitMargin) = 5400 ∧
    x * 60 + y * 75 = 3600 ∧
    3000 / 75 = (5400 - 3000) / 60 ∧
    (3000 / 75 + 3000 / 75) = 80 :=
  sorry


end third_purchase_total_l3142_314243


namespace beneficial_average_recording_l3142_314287

/-- Proves that recording the average of two new test grades is beneficial
    if the average of previous grades is higher than the average of the new grades -/
theorem beneficial_average_recording (n : ℕ) (x y : ℝ) (h : x / n > y / 2) :
  (x + y) / (n + 2) > (x + y / 2) / (n + 1) := by
  sorry

#check beneficial_average_recording

end beneficial_average_recording_l3142_314287


namespace min_sum_quadratic_roots_l3142_314228

theorem min_sum_quadratic_roots (a b : ℕ+) (h1 : ∃ x y : ℝ, 
  x ≠ y ∧ -1 < x ∧ x < 0 ∧ -1 < y ∧ y < 0 ∧ 
  a * x^2 + b * x + 1 = 0 ∧ a * y^2 + b * y + 1 = 0) : 
  (∀ a' b' : ℕ+, (∃ x y : ℝ, 
    x ≠ y ∧ -1 < x ∧ x < 0 ∧ -1 < y ∧ y < 0 ∧ 
    a' * x^2 + b' * x + 1 = 0 ∧ a' * y^2 + b' * y + 1 = 0) → 
  (a'.val + b'.val : ℕ) ≥ (a.val + b.val)) ∧ 
  (a.val + b.val : ℕ) = 10 := by sorry

end min_sum_quadratic_roots_l3142_314228


namespace inequality_solution_set_l3142_314270

-- Define the inequality
def inequality (x : ℝ) : Prop := -x^2 - 5*x + 6 ≥ 0

-- Define the solution set
def solution_set : Set ℝ := {x : ℝ | -6 ≤ x ∧ x ≤ 1}

-- Theorem statement
theorem inequality_solution_set : 
  {x : ℝ | inequality x} = solution_set := by sorry

end inequality_solution_set_l3142_314270


namespace at_least_one_red_probability_l3142_314242

theorem at_least_one_red_probability
  (prob_red_A prob_red_B : ℚ)
  (h_prob_A : prob_red_A = 1/3)
  (h_prob_B : prob_red_B = 1/2) :
  1 - (1 - prob_red_A) * (1 - prob_red_B) = 2/3 := by
  sorry

end at_least_one_red_probability_l3142_314242


namespace choir_members_count_l3142_314253

theorem choir_members_count : ∃! n : ℕ, 
  150 < n ∧ n < 300 ∧ 
  n % 6 = 1 ∧ 
  n % 8 = 3 ∧ 
  n % 9 = 5 ∧ 
  n = 193 := by sorry

end choir_members_count_l3142_314253


namespace triangle_area_l3142_314266

theorem triangle_area (A B C : Real) (a b c : Real) :
  -- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively
  C = π / 4 →
  c = 2 →
  -- Area formula
  (1 / 2) * a * c * Real.sin B = (3 + Real.sqrt 3) / 2 :=
by sorry

end triangle_area_l3142_314266


namespace no_natural_square_diff_2014_l3142_314231

theorem no_natural_square_diff_2014 : ¬∃ (m n : ℕ), m^2 = n^2 + 2014 := by
  sorry

end no_natural_square_diff_2014_l3142_314231


namespace contrapositive_proof_l3142_314209

theorem contrapositive_proof : 
  (∀ x : ℝ, x^2 < 1 → -1 < x ∧ x < 1) ↔ 
  (∀ x : ℝ, x ≥ 1 ∨ x ≤ -1 → x^2 ≥ 1) := by
sorry

end contrapositive_proof_l3142_314209


namespace apartments_with_one_resident_l3142_314261

theorem apartments_with_one_resident (total : ℕ) (at_least_one_percent : ℚ) (at_least_two_percent : ℚ) :
  total = 120 →
  at_least_one_percent = 85 / 100 →
  at_least_two_percent = 60 / 100 →
  (total * at_least_one_percent - total * at_least_two_percent : ℚ) = 30 := by
sorry

end apartments_with_one_resident_l3142_314261


namespace domino_tiling_theorem_l3142_314277

/-- Represents a rectangle tiled with dominoes -/
structure DominoRectangle where
  width : ℕ
  height : ℕ
  dominoes : ℕ

/-- Condition that any grid line intersects a multiple of four dominoes -/
def grid_line_condition (r : DominoRectangle) : Prop :=
  ∀ (line : ℕ), line ≤ r.width ∨ line ≤ r.height → 
    (if line ≤ r.width then r.height else r.width) % 4 = 0

/-- Main theorem: If the grid line condition holds, then one side is divisible by 4 -/
theorem domino_tiling_theorem (r : DominoRectangle) 
  (h : grid_line_condition r) : 
  r.width % 4 = 0 ∨ r.height % 4 = 0 := by
  sorry

end domino_tiling_theorem_l3142_314277


namespace arithmetic_calculation_l3142_314289

theorem arithmetic_calculation : 2 + 3 * 4 - 5 + 6 - (1 + 2) = 12 := by
  sorry

end arithmetic_calculation_l3142_314289


namespace square_grid_perimeter_l3142_314241

/-- The perimeter of a 3x3 grid of congruent squares with a total area of 576 square centimeters is 192 centimeters. -/
theorem square_grid_perimeter (total_area : ℝ) (side_length : ℝ) (perimeter : ℝ) : 
  total_area = 576 →
  side_length * side_length * 9 = total_area →
  perimeter = 4 * 3 * side_length →
  perimeter = 192 := by
sorry

end square_grid_perimeter_l3142_314241


namespace b_zero_iff_f_even_l3142_314249

-- Define the function f
def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define what it means for f to be even
def is_even (a b c : ℝ) : Prop :=
  ∀ x, f a b c x = f a b c (-x)

-- State the theorem
theorem b_zero_iff_f_even (a b c : ℝ) :
  b = 0 ↔ is_even a b c :=
sorry

end b_zero_iff_f_even_l3142_314249


namespace election_winner_percentage_l3142_314206

theorem election_winner_percentage (total_votes : ℕ) (winner_votes : ℕ) (margin : ℕ) : 
  (total_votes = winner_votes + (winner_votes - margin)) →
  (winner_votes = 650) →
  (margin = 300) →
  (winner_votes : ℚ) / (total_votes : ℚ) = 13/20 :=
by sorry

end election_winner_percentage_l3142_314206


namespace total_feed_amount_l3142_314222

/-- The price per pound of the cheaper feed -/
def cheap_price : ℚ := 18 / 100

/-- The price per pound of the expensive feed -/
def expensive_price : ℚ := 53 / 100

/-- The desired price per pound of the mixed feed -/
def mixed_price : ℚ := 36 / 100

/-- The amount of cheaper feed used (in pounds) -/
def cheap_amount : ℚ := 17

/-- The theorem stating that the total amount of feed mixed is 35 pounds -/
theorem total_feed_amount : 
  ∃ (expensive_amount : ℚ),
    cheap_amount + expensive_amount = 35 ∧
    (cheap_amount * cheap_price + expensive_amount * expensive_price) / (cheap_amount + expensive_amount) = mixed_price :=
by sorry

end total_feed_amount_l3142_314222


namespace skyler_song_difference_l3142_314281

def composer_songs (total_songs hit_songs top_100_extra : ℕ) : Prop :=
  let top_100_songs := hit_songs + top_100_extra
  let unreleased_songs := total_songs - (hit_songs + top_100_songs)
  hit_songs - unreleased_songs = 5

theorem skyler_song_difference :
  composer_songs 80 25 10 :=
by
  sorry

end skyler_song_difference_l3142_314281


namespace marble_selection_ways_l3142_314288

def total_marbles : ℕ := 15
def red_marbles : ℕ := 2
def green_marbles : ℕ := 2
def blue_marbles : ℕ := 2
def marbles_to_choose : ℕ := 5
def special_marbles_to_choose : ℕ := 2

theorem marble_selection_ways :
  (Nat.choose 3 2 * (Nat.choose red_marbles 1 * Nat.choose green_marbles 1 +
   Nat.choose red_marbles 1 * Nat.choose blue_marbles 1 +
   Nat.choose green_marbles 1 * Nat.choose blue_marbles 1) +
   Nat.choose 3 1 * Nat.choose red_marbles 2) *
  Nat.choose (total_marbles - (red_marbles + green_marbles + blue_marbles)) (marbles_to_choose - special_marbles_to_choose) = 3300 := by
  sorry

end marble_selection_ways_l3142_314288


namespace systematic_sampling_theorem_l3142_314294

/-- Represents the sampling methods --/
inductive SamplingMethod
  | StratifiedSampling
  | LotteryMethod
  | SystematicSampling
  | RandomNumberTableMethod

/-- Represents a grade with classes and students --/
structure Grade where
  num_classes : Nat
  students_per_class : Nat
  selected_number : Nat

/-- Determines the sampling method based on the grade structure --/
def determineSamplingMethod (g : Grade) : SamplingMethod :=
  if g.num_classes > 0 ∧ 
     g.students_per_class > 0 ∧ 
     g.selected_number > 0 ∧ 
     g.selected_number ≤ g.students_per_class
  then SamplingMethod.SystematicSampling
  else SamplingMethod.StratifiedSampling  -- Default case, not actually used in this problem

theorem systematic_sampling_theorem (g : Grade) :
  g.num_classes = 12 ∧ 
  g.students_per_class = 50 ∧ 
  g.selected_number = 14 →
  determineSamplingMethod g = SamplingMethod.SystematicSampling :=
by
  sorry

end systematic_sampling_theorem_l3142_314294


namespace weight_loss_difference_l3142_314269

/-- Given the weight loss of three people, prove how much more Veronica lost compared to Seth. -/
theorem weight_loss_difference (seth_loss jerome_loss veronica_loss total_loss : ℝ) : 
  seth_loss = 17.5 →
  jerome_loss = 3 * seth_loss →
  total_loss = 89 →
  total_loss = seth_loss + jerome_loss + veronica_loss →
  veronica_loss > seth_loss →
  veronica_loss - seth_loss = 1.5 := by
  sorry

#check weight_loss_difference

end weight_loss_difference_l3142_314269


namespace quadratic_greater_than_zero_l3142_314212

theorem quadratic_greater_than_zero (x : ℝ) :
  (x + 2) * (x - 3) - 4 > 0 ↔ x < (1 - Real.sqrt 41) / 2 ∨ x > (1 + Real.sqrt 41) / 2 := by
  sorry

end quadratic_greater_than_zero_l3142_314212


namespace solve_coloring_books_problem_l3142_314291

def coloring_books_problem (initial_stock : ℝ) (coupons_per_book : ℝ) (total_coupons_used : ℕ) : Prop :=
  initial_stock = 40.0 ∧
  coupons_per_book = 4.0 ∧
  total_coupons_used = 80 →
  initial_stock - (total_coupons_used : ℝ) / coupons_per_book = 20

theorem solve_coloring_books_problem :
  ∃ (initial_stock coupons_per_book : ℝ) (total_coupons_used : ℕ),
    coloring_books_problem initial_stock coupons_per_book total_coupons_used :=
by
  sorry

end solve_coloring_books_problem_l3142_314291


namespace matrix_equation_solution_l3142_314213

theorem matrix_equation_solution :
  let A : Matrix (Fin 2) (Fin 2) ℚ := !![2, -5; 4, -3]
  let B : Matrix (Fin 2) (Fin 2) ℚ := !![-20, -8; 9, 3]
  let N : Matrix (Fin 2) (Fin 2) ℚ := !![46/7, -58/7; -39/14, 51/14]
  N * A = B := by sorry

end matrix_equation_solution_l3142_314213


namespace sum_of_reciprocals_theorem_l3142_314238

/-- Given two positive integers m and n with specific HCF, LCM, and sum,
    prove that the sum of their reciprocals equals 2/31.5 -/
theorem sum_of_reciprocals_theorem (m n : ℕ+) : 
  Nat.gcd m.val n.val = 6 →
  Nat.lcm m.val n.val = 210 →
  m + n = 80 →
  (1 : ℚ) / m + (1 : ℚ) / n = 2 / 31.5 := by
  sorry

end sum_of_reciprocals_theorem_l3142_314238


namespace outfits_count_l3142_314274

/-- The number of shirts available -/
def num_shirts : ℕ := 8

/-- The number of ties available -/
def num_ties : ℕ := 5

/-- The number of pants available -/
def num_pants : ℕ := 4

/-- The number of jackets available -/
def num_jackets : ℕ := 2

/-- The number of tie options (including not wearing a tie) -/
def tie_options : ℕ := num_ties + 1

/-- The number of jacket options (including not wearing a jacket) -/
def jacket_options : ℕ := num_jackets + 1

/-- The total number of possible outfits -/
def total_outfits : ℕ := num_shirts * num_pants * tie_options * jacket_options

theorem outfits_count : total_outfits = 576 := by
  sorry

end outfits_count_l3142_314274


namespace triangle_inequality_l3142_314204

theorem triangle_inequality (a b c : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_sum : a + b + c ≤ 2) : 
  -3 < (a^3/b + b^3/c + c^3/a - a^3/c - b^3/a - c^3/b) ∧
  (a^3/b + b^3/c + c^3/a - a^3/c - b^3/a - c^3/b) < 3 := by
  sorry

end triangle_inequality_l3142_314204


namespace restaurant_production_june_l3142_314298

theorem restaurant_production_june :
  let weekday_cheese_pizzas := 60 + 40
  let weekday_pepperoni_pizzas := 2 * weekday_cheese_pizzas
  let weekday_beef_hotdogs := 30
  let weekday_chicken_hotdogs := 30
  let weekend_cheese_pizzas := 50 + 30
  let weekend_pepperoni_pizzas := 2 * weekend_cheese_pizzas
  let weekend_beef_hotdogs := 20
  let weekend_chicken_hotdogs := 30
  let weekend_bbq_chicken_pizzas := 25
  let weekend_veggie_pizzas := 15
  let weekdays_in_june := 20
  let weekends_in_june := 10
  
  (weekday_cheese_pizzas * weekdays_in_june + weekend_cheese_pizzas * weekends_in_june = 2800) ∧
  (weekday_pepperoni_pizzas * weekdays_in_june + weekend_pepperoni_pizzas * weekends_in_june = 5600) ∧
  (weekday_beef_hotdogs * weekdays_in_june + weekend_beef_hotdogs * weekends_in_june = 800) ∧
  (weekday_chicken_hotdogs * weekdays_in_june + weekend_chicken_hotdogs * weekends_in_june = 900) ∧
  (weekend_bbq_chicken_pizzas * weekends_in_june = 250) ∧
  (weekend_veggie_pizzas * weekends_in_june = 150) := by
  sorry

end restaurant_production_june_l3142_314298


namespace triangle_area_l3142_314267

theorem triangle_area (a b c : ℝ) (A B C : ℝ) :
  c^2 = (a - b)^2 + 6 →
  C = π / 3 →
  (1/2) * a * b * Real.sin C = (3 * Real.sqrt 3) / 2 :=
by sorry

end triangle_area_l3142_314267


namespace smallest_whole_number_above_sum_l3142_314283

theorem smallest_whole_number_above_sum : ∃ n : ℕ, 
  (n : ℝ) > 3 + 1/3 + 4 + 1/4 + 5 + 1/6 + 6 + 1/8 - 2 ∧ 
  ∀ m : ℕ, (m : ℝ) > 3 + 1/3 + 4 + 1/4 + 5 + 1/6 + 6 + 1/8 - 2 → m ≥ n :=
by
  -- Proof goes here
  sorry

end smallest_whole_number_above_sum_l3142_314283


namespace derivative_at_zero_l3142_314279

theorem derivative_at_zero (f : ℝ → ℝ) (h : ∀ x, f x = x^2 + 2*x*(deriv f 1)) :
  deriv f 0 = -4 := by
  sorry

end derivative_at_zero_l3142_314279


namespace fewer_spoons_purchased_l3142_314263

/-- The number of types of silverware --/
def numTypes : ℕ := 4

/-- The initially planned number of pieces per type --/
def initialPerType : ℕ := 15

/-- The total number of pieces actually purchased --/
def actualTotal : ℕ := 44

/-- Theorem stating that the number of fewer spoons purchased is 4 --/
theorem fewer_spoons_purchased :
  (numTypes * initialPerType - actualTotal) / numTypes = 4 := by
  sorry

end fewer_spoons_purchased_l3142_314263


namespace intersection_of_S_and_T_l3142_314262

open Set

def S : Set ℝ := {x | x > -2}
def T : Set ℝ := {x | -4 ≤ x ∧ x ≤ 1}

theorem intersection_of_S_and_T : S ∩ T = Ioc (-2) 1 := by sorry

end intersection_of_S_and_T_l3142_314262


namespace cristinas_leftover_croissants_l3142_314278

/-- Represents the types of croissants --/
inductive CroissantType
  | Chocolate
  | Plain

/-- Represents a guest's dietary restriction --/
inductive DietaryRestriction
  | Vegan
  | ChocolateAllergy
  | NoRestriction

/-- Represents the croissant distribution problem --/
structure CroissantDistribution where
  total_croissants : ℕ
  chocolate_croissants : ℕ
  plain_croissants : ℕ
  guests : List DietaryRestriction
  more_chocolate : chocolate_croissants > plain_croissants

/-- The specific instance of the problem --/
def cristinas_distribution : CroissantDistribution := {
  total_croissants := 17,
  chocolate_croissants := 12,
  plain_croissants := 5,
  guests := [DietaryRestriction.Vegan, DietaryRestriction.Vegan, DietaryRestriction.Vegan,
             DietaryRestriction.ChocolateAllergy, DietaryRestriction.ChocolateAllergy,
             DietaryRestriction.NoRestriction, DietaryRestriction.NoRestriction],
  more_chocolate := by sorry
}

/-- Function to calculate the number of leftover croissants --/
def leftover_croissants (d : CroissantDistribution) : ℕ := 
  d.total_croissants - d.guests.length

/-- Theorem stating that the number of leftover croissants in Cristina's distribution is 3 --/
theorem cristinas_leftover_croissants :
  leftover_croissants cristinas_distribution = 3 := by sorry

end cristinas_leftover_croissants_l3142_314278


namespace apartment_groceries_cost_l3142_314286

/-- Proves the cost of groceries for three roommates given their expenses -/
theorem apartment_groceries_cost 
  (rent : ℕ) 
  (utilities : ℕ) 
  (internet : ℕ) 
  (cleaning_supplies : ℕ) 
  (one_roommate_total : ℕ) 
  (h1 : rent = 1100)
  (h2 : utilities = 114)
  (h3 : internet = 60)
  (h4 : cleaning_supplies = 40)
  (h5 : one_roommate_total = 924) :
  (one_roommate_total - (rent + utilities + internet + cleaning_supplies) / 3) * 3 = 1458 :=
by sorry

end apartment_groceries_cost_l3142_314286


namespace comic_cost_l3142_314229

theorem comic_cost (initial_money : ℕ) (comics_bought : ℕ) (money_left : ℕ) : 
  initial_money = 87 → comics_bought = 8 → money_left = 55 → 
  (initial_money - money_left) / comics_bought = 4 := by
sorry

end comic_cost_l3142_314229


namespace maddie_friday_episodes_l3142_314200

/-- Represents the TV watching schedule for a week -/
structure TVSchedule where
  total_episodes : ℕ
  episode_duration : ℕ
  monday_minutes : ℕ
  thursday_minutes : ℕ
  weekend_minutes : ℕ

/-- Calculates the number of episodes watched on Friday -/
def episodes_on_friday (schedule : TVSchedule) : ℕ :=
  let total_minutes := schedule.total_episodes * schedule.episode_duration
  let other_days_minutes := schedule.monday_minutes + schedule.thursday_minutes + schedule.weekend_minutes
  let friday_minutes := total_minutes - other_days_minutes
  friday_minutes / schedule.episode_duration

/-- Theorem stating that Maddie watched 2 episodes on Friday -/
theorem maddie_friday_episodes :
  let schedule := TVSchedule.mk 8 44 138 21 105
  episodes_on_friday schedule = 2 := by
  sorry

end maddie_friday_episodes_l3142_314200


namespace infinitely_many_squares_l3142_314250

/-- An arithmetic sequence of positive integers -/
def ArithmeticSequence (a d : ℕ) : ℕ → ℕ := fun n => a + n * d

/-- A number is a perfect square -/
def IsPerfectSquare (n : ℕ) : Prop := ∃ k : ℕ, n = k * k

theorem infinitely_many_squares
  (a d : ℕ) -- First term and common difference
  (h_pos : ∀ n, 0 < ArithmeticSequence a d n) -- Sequence is positive
  (h_square : ∃ n, IsPerfectSquare (ArithmeticSequence a d n)) -- At least one square exists
  : ∀ m : ℕ, ∃ n > m, IsPerfectSquare (ArithmeticSequence a d n) :=
sorry

end infinitely_many_squares_l3142_314250


namespace geometric_sequence_inequality_l3142_314280

theorem geometric_sequence_inequality (a : Fin 8 → ℝ) (q : ℝ) :
  (∀ i : Fin 8, a i > 0) →
  (∀ i : Fin 7, a (i + 1) = a i * q) →
  q ≠ 1 →
  a 1 + a 8 > a 4 + a 5 := by
  sorry

end geometric_sequence_inequality_l3142_314280


namespace special_equation_result_l3142_314257

theorem special_equation_result (x : ℝ) (h : x + 1/x = Real.sqrt 7) :
  x^12 - 5*x^8 + 2*x^6 = 1944 * Real.sqrt 7 * x - 2494 := by
  sorry

end special_equation_result_l3142_314257


namespace statement_equivalence_l3142_314239

theorem statement_equivalence (triangle_red circle_large : Prop) :
  (triangle_red → ¬circle_large) ↔ 
  (circle_large → ¬triangle_red) ∧ 
  (¬triangle_red ∨ ¬circle_large) := by sorry

end statement_equivalence_l3142_314239


namespace janous_problem_l3142_314282

def is_valid_triple (x y z : ℕ+) : Prop :=
  x ∣ (y + 1) ∧ y ∣ (z + 1) ∧ z ∣ (x + 1)

def solution_set : Set (ℕ+ × ℕ+ × ℕ+) :=
  {(1, 1, 1), (1, 2, 1), (1, 1, 2), (1, 3, 2), (2, 1, 1), (2, 1, 3), (3, 1, 2), 
   (2, 2, 2), (2, 2, 3), (2, 3, 2), (3, 2, 2)}

theorem janous_problem :
  ∀ x y z : ℕ+, is_valid_triple x y z ↔ (x, y, z) ∈ solution_set := by
  sorry

end janous_problem_l3142_314282


namespace hyperbola_asymptote_a_value_l3142_314264

/-- Proves that for a hyperbola x²/a² - y² = 1 with a > 0, 
    if one of its asymptotes is y + 2x = 0, then a = 2 -/
theorem hyperbola_asymptote_a_value (a : ℝ) (h1 : a > 0) : 
  (∃ x y : ℝ, x^2 / a^2 - y^2 = 1 ∧ y + 2*x = 0) → a = 2 := by
  sorry

end hyperbola_asymptote_a_value_l3142_314264


namespace positive_number_has_square_root_l3142_314247

theorem positive_number_has_square_root :
  ∀ x : ℝ, x > 0 → ∃ y : ℝ, y * y = x :=
by sorry

end positive_number_has_square_root_l3142_314247


namespace sandwich_combinations_l3142_314254

/-- Represents the number of different types of bread available. -/
def num_breads : ℕ := 5

/-- Represents the number of different types of meat available. -/
def num_meats : ℕ := 7

/-- Represents the number of different types of cheese available. -/
def num_cheeses : ℕ := 6

/-- Represents the number of bread options that can go with a specific meat/cheese combination. -/
def num_bread_options : ℕ := 5

/-- Represents the number of restricted combinations (ham/cheddar and turkey/swiss). -/
def num_restricted_combinations : ℕ := 2

theorem sandwich_combinations :
  (num_breads * num_meats * num_cheeses) - (num_bread_options * num_restricted_combinations) = 200 := by
  sorry

end sandwich_combinations_l3142_314254


namespace A_2022_coordinates_l3142_314234

/-- The companion point transformation --/
def companion_point (p : ℝ × ℝ) : ℝ × ℝ :=
  (-(p.2) + 1, p.1 + 1)

/-- The sequence of points starting from A1 --/
def A : ℕ → ℝ × ℝ
  | 0 => (2, 4)
  | n + 1 => companion_point (A n)

/-- The main theorem --/
theorem A_2022_coordinates :
  A 2021 = (-3, 3) := by
  sorry

end A_2022_coordinates_l3142_314234


namespace triangle_theorem_l3142_314237

noncomputable def triangle_problem (a b c : ℝ) (A B C : ℝ) : Prop :=
  let condition1 := 2 * a * Real.cos B = b + 2 * c
  let condition2 := 2 * Real.sin C + Real.tan A * Real.cos B + Real.sin B = 0
  let condition3 := (a - c) / Real.sin B = (b + c) / (Real.sin A + Real.sin C)
  b = 2 ∧ c = 4 ∧ 
  (condition1 ∨ condition2 ∨ condition3) →
  A = 2 * Real.pi / 3 ∧
  ∃ (D : ℝ × ℝ), 
    let BC := Real.sqrt ((b - c * Real.cos A)^2 + (c * Real.sin A)^2)
    let BD := BC / 4
    let AD := Real.sqrt (((3/4) * b)^2 + ((1/4) * c)^2 + 
               (3/4) * b * (1/4) * c * Real.cos A)
    AD = Real.sqrt 31 / 2

theorem triangle_theorem : 
  ∀ (a b c A B C : ℝ), triangle_problem a b c A B C :=
sorry

end triangle_theorem_l3142_314237


namespace exchange_three_cows_to_chickens_l3142_314245

/-- Exchange rates between animals -/
structure ExchangeRates where
  cows_to_sheep : ℚ      -- Rate of cows to sheep
  sheep_to_rabbits : ℚ   -- Rate of sheep to rabbits
  rabbits_to_chickens : ℚ -- Rate of rabbits to chickens

/-- Given the exchange rates, calculate how many chickens can be exchanged for a given number of cows -/
def cows_to_chickens (rates : ExchangeRates) (num_cows : ℚ) : ℚ :=
  num_cows * rates.cows_to_sheep * rates.sheep_to_rabbits * rates.rabbits_to_chickens

/-- Theorem stating that 3 cows can be exchanged for 819 chickens given the specified exchange rates -/
theorem exchange_three_cows_to_chickens :
  let rates : ExchangeRates := {
    cows_to_sheep := 42 / 2,
    sheep_to_rabbits := 26 / 3,
    rabbits_to_chickens := 3 / 2
  }
  cows_to_chickens rates 3 = 819 := by
  sorry


end exchange_three_cows_to_chickens_l3142_314245


namespace outfits_count_l3142_314217

/-- The number of unique outfits that can be made from a given number of shirts, ties, and belts. -/
def uniqueOutfits (shirts ties belts : ℕ) : ℕ := shirts * ties * belts

/-- Theorem stating that with 8 shirts, 6 ties, and 4 belts, the number of unique outfits is 192. -/
theorem outfits_count : uniqueOutfits 8 6 4 = 192 := by
  sorry

end outfits_count_l3142_314217


namespace roots_sum_zero_l3142_314226

theorem roots_sum_zero (m : ℝ) (x₁ x₂ : ℝ) 
  (h₁ : Real.log (abs x₁) = m) 
  (h₂ : Real.log (abs x₂) = m) 
  (h₃ : x₁ ≠ x₂) : 
  x₁ + x₂ = 0 := by
sorry

end roots_sum_zero_l3142_314226


namespace difference_30th_28th_triangular_l3142_314276

def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

theorem difference_30th_28th_triangular : 
  triangular_number 30 - triangular_number 28 = 59 := by
  sorry

end difference_30th_28th_triangular_l3142_314276


namespace cube_vertex_configurations_l3142_314223

/-- Represents a vertex of a cube -/
inductive CubeVertex
  | A | B | C | D | A1 | B1 | C1 | D1

/-- Represents a set of 4 vertices from a cube -/
def VertexSet := Finset CubeVertex

/-- Checks if a set of vertices forms a rectangle -/
def is_rectangle (vs : VertexSet) : Prop := sorry

/-- Checks if a set of vertices forms a tetrahedron with all equilateral triangle faces -/
def is_equilateral_tetrahedron (vs : VertexSet) : Prop := sorry

/-- Checks if a set of vertices forms a tetrahedron with all right triangle faces -/
def is_right_tetrahedron (vs : VertexSet) : Prop := sorry

/-- Checks if a set of vertices forms a tetrahedron with three isosceles right triangle faces and one equilateral triangle face -/
def is_mixed_tetrahedron (vs : VertexSet) : Prop := sorry

theorem cube_vertex_configurations :
  ∃ (vs1 vs2 vs3 vs4 : VertexSet),
    is_rectangle vs1 ∧
    is_equilateral_tetrahedron vs2 ∧
    is_right_tetrahedron vs3 ∧
    is_mixed_tetrahedron vs4 :=
  sorry

end cube_vertex_configurations_l3142_314223


namespace fish_tank_balls_l3142_314290

/-- The number of goldfish in the tank -/
def num_goldfish : ℕ := 3

/-- The number of platyfish in the tank -/
def num_platyfish : ℕ := 10

/-- The number of red balls each goldfish plays with -/
def red_balls_per_goldfish : ℕ := 10

/-- The number of white balls each platyfish plays with -/
def white_balls_per_platyfish : ℕ := 5

/-- The total number of balls in the fish tank -/
def total_balls : ℕ := num_goldfish * red_balls_per_goldfish + num_platyfish * white_balls_per_platyfish

theorem fish_tank_balls : total_balls = 80 := by
  sorry

end fish_tank_balls_l3142_314290


namespace jamies_mean_score_l3142_314202

def scores : List ℕ := [80, 85, 90, 95, 100, 105]

theorem jamies_mean_score 
  (h1 : scores.length = 6)
  (h2 : ∃ alex_scores jamie_scores : List ℕ, 
        alex_scores.length = 3 ∧ 
        jamie_scores.length = 3 ∧ 
        scores = alex_scores ++ jamie_scores)
  (h3 : ∃ alex_scores : List ℕ, 
        alex_scores.length = 3 ∧ 
        alex_scores.sum / alex_scores.length = 85)
  : ∃ jamie_scores : List ℕ,
    jamie_scores.length = 3 ∧
    jamie_scores.sum / jamie_scores.length = 100 := by
  sorry

end jamies_mean_score_l3142_314202


namespace paint_fraction_in_15_minutes_l3142_314299

/-- The fraction of a wall that can be painted by two people working together,
    given their individual rates and a specific time. -/
def fractionPainted (rate1 rate2 time : ℚ) : ℚ :=
  (rate1 + rate2) * time

theorem paint_fraction_in_15_minutes :
  let heidi_rate : ℚ := 1 / 60
  let zoe_rate : ℚ := 1 / 90
  let time : ℚ := 15
  fractionPainted heidi_rate zoe_rate time = 5 / 12 := by
  sorry

end paint_fraction_in_15_minutes_l3142_314299


namespace distance_sasha_kolya_l3142_314275

/-- Represents a runner in the race -/
structure Runner where
  speed : ℝ
  position : ℝ

/-- The race setup -/
structure Race where
  sasha : Runner
  lyosha : Runner
  kolya : Runner
  length : ℝ

/-- Conditions of the race -/
def race_conditions (r : Race) : Prop :=
  r.length = 100 ∧
  r.sasha.speed > 0 ∧
  r.lyosha.speed > 0 ∧
  r.kolya.speed > 0 ∧
  r.sasha.position = r.length ∧
  r.lyosha.position = r.length - 10 ∧
  r.kolya.position = r.lyosha.position * (r.kolya.speed / r.lyosha.speed)

/-- The theorem to be proved -/
theorem distance_sasha_kolya (r : Race) (h : race_conditions r) :
  r.sasha.position - r.kolya.position = 19 := by
  sorry

end distance_sasha_kolya_l3142_314275


namespace custom_mul_theorem_l3142_314293

/-- Custom multiplication operation -/
def custom_mul (a b : ℝ) : ℝ := 3 * a - 2 * b^2

/-- Theorem stating that if a * 6 = -3 using the custom multiplication, then a = 23 -/
theorem custom_mul_theorem (a : ℝ) (h : custom_mul a 6 = -3) : a = 23 := by
  sorry

end custom_mul_theorem_l3142_314293


namespace inscribed_quadrilateral_exists_l3142_314285

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a quadrilateral
structure Quadrilateral where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ

-- Define the property of a quadrilateral being circumscribed around a circle
def isCircumscribed (q : Quadrilateral) (c : Circle) : Prop :=
  sorry

-- Define the property of a point being on a circle
def isOnCircle (p : ℝ × ℝ) (c : Circle) : Prop :=
  sorry

-- Define homothety between two quadrilaterals
def isHomothetic (q1 q2 : Quadrilateral) : Prop :=
  sorry

-- Theorem statement
theorem inscribed_quadrilateral_exists (ABCD : Quadrilateral) (c : Circle) :
  isCircumscribed ABCD c →
  isOnCircle ABCD.A c →
  isOnCircle ABCD.B c →
  isOnCircle ABCD.C c →
  ∃ (EFGH : Quadrilateral), isCircumscribed EFGH c ∧ isHomothetic ABCD EFGH :=
sorry

end inscribed_quadrilateral_exists_l3142_314285


namespace equation_solution_l3142_314268

theorem equation_solution : ∃ y : ℚ, (8 + 3.2 * y = 0.8 * y + 40) ∧ (y = 40 / 3) := by
  sorry

end equation_solution_l3142_314268


namespace negative_difference_l3142_314292

theorem negative_difference (m n : ℝ) : -(m - n) = -m + n := by
  sorry

end negative_difference_l3142_314292


namespace power_equality_l3142_314259

theorem power_equality (p : ℕ) (h : (81 : ℕ)^6 = 3^p) : p = 24 := by
  sorry

end power_equality_l3142_314259


namespace sum_of_roots_l3142_314201

theorem sum_of_roots (k d y₁ y₂ : ℝ) 
  (h₁ : y₁ ≠ y₂) 
  (h₂ : 4 * y₁^2 - k * y₁ = d) 
  (h₃ : 4 * y₂^2 - k * y₂ = d) : 
  y₁ + y₂ = k / 4 := by
sorry

end sum_of_roots_l3142_314201
