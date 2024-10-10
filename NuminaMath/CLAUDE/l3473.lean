import Mathlib

namespace distance_between_lines_l3473_347333

/-- A circle intersected by three equally spaced parallel lines -/
structure CircleWithParallelLines where
  /-- The radius of the circle -/
  r : ℝ
  /-- The distance between adjacent parallel lines -/
  d : ℝ
  /-- The lengths of the three chords created by the parallel lines -/
  chord1 : ℝ
  chord2 : ℝ
  chord3 : ℝ
  /-- The chords are positive -/
  chord1_pos : chord1 > 0
  chord2_pos : chord2 > 0
  chord3_pos : chord3 > 0
  /-- The radius is positive -/
  r_pos : r > 0
  /-- The distance between lines is positive -/
  d_pos : d > 0
  /-- The chords satisfy Stewart's theorem -/
  stewart_theorem1 : (chord1 / 2) ^ 2 * chord1 + (d / 2) ^ 2 * chord1 = (chord1 / 2) * r ^ 2 + (chord1 / 2) * r ^ 2
  stewart_theorem2 : (chord3 / 2) ^ 2 * chord3 + ((3 * d) / 2) ^ 2 * chord3 = (chord3 / 2) * r ^ 2 + (chord3 / 2) * r ^ 2

/-- The main theorem stating that for the given chord lengths, the distance between lines is 6 -/
theorem distance_between_lines (c : CircleWithParallelLines) 
    (h1 : c.chord1 = 40) (h2 : c.chord2 = 40) (h3 : c.chord3 = 36) : c.d = 6 := by
  sorry

end distance_between_lines_l3473_347333


namespace volume_of_extended_box_l3473_347332

/-- Represents a rectangular parallelepiped (box) -/
structure Box where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of the set of points inside or within one unit of a box -/
def volumeWithinOneUnit (b : Box) : ℝ := sorry

/-- Checks if two integers are relatively prime -/
def isRelativelyPrime (a b : ℕ) : Prop := sorry

theorem volume_of_extended_box (m n p : ℕ) :
  (∃ b : Box, b.length = 2 ∧ b.width = 3 ∧ b.height = 6) →
  (∃ v : ℝ, v = volumeWithinOneUnit b) →
  v = (m + n * Real.pi) / p →
  m > 0 ∧ n > 0 ∧ p > 0 →
  isRelativelyPrime n p →
  m + n + p = 364 := by
  sorry

end volume_of_extended_box_l3473_347332


namespace unit_price_ratio_l3473_347345

theorem unit_price_ratio (quantity_B price_B : ℝ) (quantity_B_pos : quantity_B > 0) (price_B_pos : price_B > 0) :
  let quantity_A := 1.3 * quantity_B
  let price_A := 0.85 * price_B
  (price_A / quantity_A) / (price_B / quantity_B) = 17 / 26 := by
sorry

end unit_price_ratio_l3473_347345


namespace chocolate_bars_count_l3473_347340

def total_candies : ℕ := 50
def chewing_gums : ℕ := 15
def assorted_candies : ℕ := 15

theorem chocolate_bars_count :
  total_candies - chewing_gums - assorted_candies = 20 :=
by sorry

end chocolate_bars_count_l3473_347340


namespace planes_parallel_if_perpendicular_lines_l3473_347304

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations
variable (parallel : Plane → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (line_parallel : Line → Line → Prop)
variable (non_overlapping_planes : Plane → Plane → Prop)
variable (non_overlapping_lines : Line → Line → Prop)

-- State the theorem
theorem planes_parallel_if_perpendicular_lines
  (α β : Plane) (l m : Line)
  (h1 : non_overlapping_planes α β)
  (h2 : non_overlapping_lines l m)
  (h3 : perpendicular l α)
  (h4 : perpendicular m β)
  (h5 : line_parallel l m) :
  parallel α β :=
sorry

end planes_parallel_if_perpendicular_lines_l3473_347304


namespace baseball_card_count_l3473_347363

def final_card_count (initial_count : ℕ) : ℕ :=
  let after_maria := initial_count - (initial_count + 1) / 2
  let after_peter := after_maria - 1
  let final_count := after_peter * 3
  final_count

theorem baseball_card_count : final_card_count 15 = 18 := by
  sorry

end baseball_card_count_l3473_347363


namespace lemonade_glasses_per_gallon_l3473_347362

theorem lemonade_glasses_per_gallon 
  (total_gallons : ℕ) 
  (cost_per_gallon : ℚ) 
  (price_per_glass : ℚ) 
  (glasses_drunk : ℕ) 
  (glasses_unsold : ℕ) 
  (net_profit : ℚ) :
  total_gallons = 2 ∧ 
  cost_per_gallon = 7/2 ∧ 
  price_per_glass = 1 ∧ 
  glasses_drunk = 5 ∧ 
  glasses_unsold = 6 ∧ 
  net_profit = 14 →
  ∃ (glasses_per_gallon : ℕ),
    glasses_per_gallon = 16 ∧
    (total_gallons * glasses_per_gallon - glasses_drunk - glasses_unsold) * price_per_glass
    = total_gallons * cost_per_gallon + net_profit :=
by sorry

end lemonade_glasses_per_gallon_l3473_347362


namespace division_problem_l3473_347310

theorem division_problem : 
  (1 / 24) / ((1 / 12) - (5 / 16) + (7 / 24) - (2 / 3)) = -(2 / 29) := by
  sorry

end division_problem_l3473_347310


namespace composite_numbers_with_special_divisors_l3473_347361

theorem composite_numbers_with_special_divisors :
  ∀ n : ℕ, n > 1 →
    (∀ d : ℕ, d ∣ n → d ≠ 1 → d ≠ n → n - 20 ≤ d ∧ d ≤ n - 12) →
    n = 21 ∨ n = 25 := by
  sorry

end composite_numbers_with_special_divisors_l3473_347361


namespace parallelogram_area_l3473_347305

def v : Fin 2 → ℝ := ![6, -4]
def w : Fin 2 → ℝ := ![8, -1]

theorem parallelogram_area : 
  abs (Matrix.det !![v 0, v 1; 2 * w 0, 2 * w 1]) = 52 := by sorry

end parallelogram_area_l3473_347305


namespace rice_price_reduction_l3473_347398

theorem rice_price_reduction (P : ℝ) (h : P > 0) :
  49 * P = 50 * (P * (1 - 2/100)) :=
by sorry

end rice_price_reduction_l3473_347398


namespace smallest_square_area_l3473_347346

/-- The smallest square area containing two non-overlapping rectangles -/
theorem smallest_square_area (r1_width r1_height r2_width r2_height : ℕ) 
  (h1 : r1_width = 3 ∧ r1_height = 5)
  (h2 : r2_width = 4 ∧ r2_height = 6) :
  (max (r1_width + r2_height) (r1_height + r2_width))^2 = 81 := by
  sorry

end smallest_square_area_l3473_347346


namespace kids_staying_home_l3473_347382

def total_kids : ℕ := 898051
def kids_at_camp : ℕ := 629424

theorem kids_staying_home : total_kids - kids_at_camp = 268627 := by
  sorry

end kids_staying_home_l3473_347382


namespace penguin_fish_theorem_l3473_347342

theorem penguin_fish_theorem (fish_counts : List ℕ) : 
  fish_counts.length = 10 ∧ 
  fish_counts.sum = 50 ∧ 
  (∀ x ∈ fish_counts, x > 0) →
  ∃ i j, i ≠ j ∧ i < fish_counts.length ∧ j < fish_counts.length ∧ fish_counts[i]! = fish_counts[j]! := by
  sorry

end penguin_fish_theorem_l3473_347342


namespace hotel_beds_count_l3473_347303

theorem hotel_beds_count (total_rooms : ℕ) (two_bed_rooms : ℕ) (beds_in_two_bed_room : ℕ) (beds_in_three_bed_room : ℕ) 
  (h1 : total_rooms = 13)
  (h2 : two_bed_rooms = 8)
  (h3 : beds_in_two_bed_room = 2)
  (h4 : beds_in_three_bed_room = 3) :
  two_bed_rooms * beds_in_two_bed_room + (total_rooms - two_bed_rooms) * beds_in_three_bed_room = 31 :=
by sorry

end hotel_beds_count_l3473_347303


namespace calculate_expression_l3473_347391

theorem calculate_expression (y : ℝ) (h : y = 3) : y + y * (y^y)^2 = 2190 := by
  sorry

end calculate_expression_l3473_347391


namespace not_sum_product_equal_neg_two_four_sum_product_equal_sqrt_two_plus_two_sqrt_two_sum_product_equal_relation_l3473_347357

/-- Definition of sum-product equal number pair -/
def is_sum_product_equal (a b : ℝ) : Prop := a + b = a * b

/-- Theorem 1: (-2, 4) is not a sum-product equal number pair -/
theorem not_sum_product_equal_neg_two_four : ¬ is_sum_product_equal (-2) 4 := by sorry

/-- Theorem 2: (√2+2, √2) is a sum-product equal number pair -/
theorem sum_product_equal_sqrt_two_plus_two_sqrt_two : is_sum_product_equal (Real.sqrt 2 + 2) (Real.sqrt 2) := by sorry

/-- Theorem 3: For (m,n) where m,n ≠ 1, if it's a sum-product equal number pair, then m = n / (n-1) -/
theorem sum_product_equal_relation (m n : ℝ) (hm : m ≠ 1) (hn : n ≠ 1) :
  is_sum_product_equal m n → m = n / (n - 1) := by sorry

end not_sum_product_equal_neg_two_four_sum_product_equal_sqrt_two_plus_two_sqrt_two_sum_product_equal_relation_l3473_347357


namespace complex_magnitude_product_l3473_347369

theorem complex_magnitude_product : Complex.abs (4 - 3*I) * Complex.abs (4 + 3*I) = 25 := by
  sorry

end complex_magnitude_product_l3473_347369


namespace triangle_abc_is_acute_l3473_347311

/-- A triangle is acute if all its angles are less than 90 degrees --/
def IsAcuteTriangle (a b c : ℝ) : Prop :=
  let cosA := (b^2 + c^2 - a^2) / (2*b*c)
  let cosB := (a^2 + c^2 - b^2) / (2*a*c)
  let cosC := (a^2 + b^2 - c^2) / (2*a*b)
  0 < cosA ∧ cosA < 1 ∧
  0 < cosB ∧ cosB < 1 ∧
  0 < cosC ∧ cosC < 1

theorem triangle_abc_is_acute :
  let a : ℝ := 9
  let b : ℝ := 10
  let c : ℝ := 12
  IsAcuteTriangle a b c := by
  sorry

end triangle_abc_is_acute_l3473_347311


namespace pen_pencil_distribution_l3473_347368

theorem pen_pencil_distribution (P : ℕ) : 
  (∃ (k : ℕ), P = 20 * k) ↔ 
  (∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ 1340 / x = y ∧ P / x = y ∧ x ≤ 20 ∧ 
   ∀ (z : ℕ), z > x → (1340 / z ≠ P / z ∨ 1340 % z ≠ 0 ∨ P % z ≠ 0)) :=
by sorry

end pen_pencil_distribution_l3473_347368


namespace modulus_of_complex_product_l3473_347372

/-- The modulus of the complex number z = (1-2i)(3+i) is equal to 5√2 -/
theorem modulus_of_complex_product : 
  let z : ℂ := (1 - 2*I) * (3 + I)
  ‖z‖ = 5 * Real.sqrt 2 := by sorry

end modulus_of_complex_product_l3473_347372


namespace differential_equation_solution_l3473_347367

/-- Given a differential equation y = x * y' + a / (2 * y'), where a is a constant,
    prove that the solutions are:
    1. y = C * x + a / (2 * C), where C is a constant
    2. y^2 = 2 * a * x
-/
theorem differential_equation_solution (a : ℝ) (x y : ℝ → ℝ) (y' : ℝ → ℝ) :
  (∀ t, y t = t * y' t + a / (2 * y' t)) →
  (∃ C : ℝ, ∀ t, y t = C * t + a / (2 * C)) ∨
  (∀ t, (y t)^2 = 2 * a * t) := by
  sorry

end differential_equation_solution_l3473_347367


namespace meeting_point_coordinates_l3473_347350

/-- The point that divides a line segment in a given ratio -/
def dividing_point (x₁ y₁ x₂ y₂ : ℚ) (m n : ℚ) : ℚ × ℚ :=
  ((m * x₂ + n * x₁) / (m + n), (m * y₂ + n * y₁) / (m + n))

/-- Proof that the point dividing the line segment from (2, 5) to (10, 1) 
    in the ratio 1:3 starting from (2, 5) has coordinates (4, 4) -/
theorem meeting_point_coordinates : 
  dividing_point 2 5 10 1 1 3 = (4, 4) := by
  sorry

end meeting_point_coordinates_l3473_347350


namespace arithmetic_sequence_sum_l3473_347352

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Theorem: In an arithmetic sequence where a_7 = 12, the sum of a_3 and a_11 is 24 -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) 
    (h_arith : arithmetic_sequence a) 
    (h_a7 : a 7 = 12) : 
  a 3 + a 11 = 24 := by
sorry

end arithmetic_sequence_sum_l3473_347352


namespace previous_weekend_earnings_l3473_347317

-- Define the given amounts
def saturday_earnings : ℕ := 18
def sunday_earnings : ℕ := saturday_earnings / 2
def pogo_stick_cost : ℕ := 60
def additional_needed : ℕ := 13

-- Define the total earnings for this weekend
def this_weekend_earnings : ℕ := saturday_earnings + sunday_earnings

-- Define the theorem
theorem previous_weekend_earnings :
  pogo_stick_cost - additional_needed - this_weekend_earnings = 20 := by
  sorry

end previous_weekend_earnings_l3473_347317


namespace min_reciprocal_sum_l3473_347377

theorem min_reciprocal_sum (x y : ℝ) (h : Real.log (x + y) = 0) :
  (1 / x + 1 / y) ≥ 4 ∧ ∃ a b : ℝ, Real.log (a + b) = 0 ∧ 1 / a + 1 / b = 4 :=
by sorry

end min_reciprocal_sum_l3473_347377


namespace unique_common_solution_coefficient_l3473_347374

theorem unique_common_solution_coefficient : 
  ∃! a : ℝ, ∃ x : ℝ, (x^2 + a*x + 1 = 0) ∧ (x^2 - x - a = 0) ∧ (a = 2) := by
  sorry

end unique_common_solution_coefficient_l3473_347374


namespace decimal_digits_divisibility_l3473_347387

def repeatedDigits (a b c : ℕ) : ℕ :=
  a * (10^4006 - 10^2004) / 99 + b * 10^2002 + c * (10^2002 - 1) / 99

theorem decimal_digits_divisibility (a b c : ℕ) 
  (ha : a ≤ 9) (hb : b ≤ 9) (hc : c ≤ 9) 
  (h_div : 37 ∣ repeatedDigits a b c) : 
  b = a + c := by sorry

end decimal_digits_divisibility_l3473_347387


namespace perspective_triangle_area_l3473_347326

/-- An equilateral triangle with side length 1 -/
structure EquilateralTriangle where
  side_length : ℝ
  is_equilateral : side_length = 1

/-- The perspective plane triangle of an equilateral triangle -/
structure PerspectiveTriangle (et : EquilateralTriangle) where

/-- The area of a triangle -/
def area (t : Type) : ℝ := sorry

/-- The theorem stating the area of the perspective plane triangle -/
theorem perspective_triangle_area (et : EquilateralTriangle) 
  (pt : PerspectiveTriangle et) : 
  area (PerspectiveTriangle et) = Real.sqrt 6 / 16 := by sorry

end perspective_triangle_area_l3473_347326


namespace adult_ticket_cost_l3473_347397

theorem adult_ticket_cost (student_price : ℕ) (num_students : ℕ) (num_adults : ℕ) (total_amount : ℕ) : 
  student_price = 6 →
  num_students = 20 →
  num_adults = 12 →
  total_amount = 216 →
  ∃ (adult_price : ℕ), 
    student_price * num_students + adult_price * num_adults = total_amount ∧ 
    adult_price = 8 := by
  sorry

end adult_ticket_cost_l3473_347397


namespace minimum_phrases_to_study_l3473_347341

/-- 
Given a total of 800 French phrases and a required quiz score of 90%,
prove that the minimum number of phrases to study is 720.
-/
theorem minimum_phrases_to_study (total_phrases : ℕ) (required_score : ℚ) : 
  total_phrases = 800 → required_score = 90 / 100 → 
  ⌈(required_score * total_phrases : ℚ)⌉ = 720 := by
sorry

end minimum_phrases_to_study_l3473_347341


namespace value_of_a_l3473_347300

/-- Given that 4 * ((a * 0.48 * 2.50) / (0.12 * 0.09 * 0.5)) = 3200.0000000000005,
    prove that a is approximately equal to 3.6 -/
theorem value_of_a (a : ℝ) : 
  4 * ((a * 0.48 * 2.50) / (0.12 * 0.09 * 0.5)) = 3200.0000000000005 → 
  ∃ ε > 0, |a - 3.6| < ε := by
sorry

end value_of_a_l3473_347300


namespace interest_rate_calculation_l3473_347334

/-- Given:
A lends B Rs. 3500
B lends C Rs. 3500 at 11.5% per annum
B's gain over 3 years is Rs. 157.5
Prove: The interest rate at which A lent to B is 10% per annum
-/
theorem interest_rate_calculation (principal : ℝ) (rate_b_to_c : ℝ) (time : ℝ) (gain : ℝ)
  (h1 : principal = 3500)
  (h2 : rate_b_to_c = 11.5)
  (h3 : time = 3)
  (h4 : gain = 157.5)
  (h5 : gain = principal * rate_b_to_c / 100 * time - principal * rate_a_to_b / 100 * time) :
  rate_a_to_b = 10 := by
  sorry

end interest_rate_calculation_l3473_347334


namespace parallelogram_count_l3473_347384

/-- The number of ways to choose 2 items from 4 -/
def choose_2_from_4 : ℕ := 6

/-- The number of horizontal lines -/
def horizontal_lines : ℕ := 4

/-- The number of vertical lines -/
def vertical_lines : ℕ := 4

/-- The number of parallelograms formed -/
def num_parallelograms : ℕ := choose_2_from_4 * choose_2_from_4

theorem parallelogram_count : num_parallelograms = 36 := by
  sorry

end parallelogram_count_l3473_347384


namespace digit_57_of_21_over_22_l3473_347335

def decimal_representation (n d : ℕ) : ℕ → ℕ
  | 0 => (n * 10 / d) % 10
  | i + 1 => decimal_representation n d i

theorem digit_57_of_21_over_22 :
  decimal_representation 21 22 56 = 4 := by
  sorry

end digit_57_of_21_over_22_l3473_347335


namespace fuel_distribution_l3473_347394

def total_fuel : ℝ := 60

theorem fuel_distribution (second_third : ℝ) (final_third : ℝ) 
  (h1 : second_third = total_fuel / 3)
  (h2 : final_third = second_third / 2)
  (h3 : second_third + final_third + (total_fuel - second_third - final_third) = total_fuel) :
  total_fuel - second_third - final_third = 30 := by
sorry

end fuel_distribution_l3473_347394


namespace expansion_coefficient_implies_a_value_l3473_347339

/-- The coefficient of x^n in the expansion of (x + 1/x)^m -/
def binomialCoeff (m n : ℕ) : ℚ := sorry

/-- The coefficient of x^n in the expansion of (x^2 - a)(x + 1/x)^m -/
def expandedCoeff (m n : ℕ) (a : ℚ) : ℚ := 
  binomialCoeff m (m - n + 2) - a * binomialCoeff m (m - n)

theorem expansion_coefficient_implies_a_value : 
  expandedCoeff 10 6 a = 30 → a = 2 := by sorry

end expansion_coefficient_implies_a_value_l3473_347339


namespace number_1991_in_32nd_group_l3473_347325

/-- The function that gives the number of elements in the nth group of odd numbers -/
def group_size (n : ℕ) : ℕ := 2 * n - 1

/-- The function that gives the sum of elements in the first n groups -/
def sum_of_first_n_groups (n : ℕ) : ℕ := n^2

/-- The theorem stating that 1991 appears in the 32nd group -/
theorem number_1991_in_32nd_group :
  (∀ k < 32, sum_of_first_n_groups k < 1991) ∧
  sum_of_first_n_groups 32 ≥ 1991 := by
  sorry

end number_1991_in_32nd_group_l3473_347325


namespace function_properties_l3473_347347

noncomputable section

def f (k a x : ℝ) : ℝ := 
  if x ≥ 0 then k * x + k * (1 - a^2) else x^2 + (a^2 - 4*a) * x + (3 - a)^2

theorem function_properties (a : ℝ) 
  (h1 : ∀ (x₁ : ℝ), x₁ ≠ 0 → ∃! (x₂ : ℝ), x₂ ≠ 0 ∧ x₂ ≠ x₁ ∧ f k a x₂ = f k a x₁) :
  ∃ k : ℝ, k = (3 - a)^2 / (1 - a^2) ∧ 0 ≤ a ∧ a < 1 :=
sorry

end function_properties_l3473_347347


namespace slope_angle_sqrt3_l3473_347395

/-- The slope angle of the line y = √3x + 1 is 60° -/
theorem slope_angle_sqrt3 : 
  let l : ℝ → ℝ := λ x => Real.sqrt 3 * x + 1
  ∃ θ : ℝ, θ = 60 * π / 180 ∧ Real.tan θ = Real.sqrt 3 :=
by sorry

end slope_angle_sqrt3_l3473_347395


namespace simplify_fraction_product_l3473_347319

theorem simplify_fraction_product (x : ℝ) (h1 : x ≠ 3) (h2 : x ≠ 5) :
  (x^2 - 4*x + 3) / (x^2 - 6*x + 9) * (x^2 - 6*x + 8) / (x^2 - 8*x + 15) =
  ((x - 1) * (x - 2) * (x - 4)) / ((x - 3) * (x - 5)) := by
  sorry

end simplify_fraction_product_l3473_347319


namespace lcm_gcf_ratio_l3473_347337

theorem lcm_gcf_ratio : 
  (Nat.lcm 252 630) / (Nat.gcd 252 630) = 10 := by sorry

end lcm_gcf_ratio_l3473_347337


namespace cube_sum_product_l3473_347302

theorem cube_sum_product : ∃ (a b : ℤ), a^3 + b^3 = 189 ∧ a * b = 20 := by
  sorry

end cube_sum_product_l3473_347302


namespace population_changes_l3473_347389

/-- Enumeration of possible population number changes --/
inductive PopulationChange
  | Increase
  | Decrease
  | Fluctuation
  | Extinction

/-- Theorem stating that population changes can be increase, decrease, fluctuation, or extinction --/
theorem population_changes : 
  ∀ (change : PopulationChange), 
    change = PopulationChange.Increase ∨
    change = PopulationChange.Decrease ∨
    change = PopulationChange.Fluctuation ∨
    change = PopulationChange.Extinction :=
by
  sorry

#check population_changes

end population_changes_l3473_347389


namespace simplify_expression_l3473_347388

theorem simplify_expression (x : ℝ) : 
  3 * x^3 + 4 * x^2 + 2 - (7 - 3 * x^3 - 4 * x^2) = 6 * x^3 + 8 * x^2 - 5 := by
  sorry

end simplify_expression_l3473_347388


namespace lunch_cost_proof_l3473_347393

theorem lunch_cost_proof (adam_cost rick_cost jose_cost : ℝ) :
  adam_cost = (2/3) * rick_cost →
  rick_cost = jose_cost →
  jose_cost = 45 →
  adam_cost + rick_cost + jose_cost = 120 := by
sorry

end lunch_cost_proof_l3473_347393


namespace point_on_x_axis_l3473_347343

theorem point_on_x_axis (m : ℝ) : 
  (∃ x : ℝ, (x = m - 1 ∧ 0 = 2 * m + 3)) → m = -3/2 := by
  sorry

end point_on_x_axis_l3473_347343


namespace negation_of_forall_abs_plus_square_nonnegative_l3473_347313

theorem negation_of_forall_abs_plus_square_nonnegative :
  (¬ ∀ x : ℝ, |x| + x^2 ≥ 0) ↔ (∃ x : ℝ, |x| + x^2 < 0) := by sorry

end negation_of_forall_abs_plus_square_nonnegative_l3473_347313


namespace smallest_consecutive_number_l3473_347351

theorem smallest_consecutive_number (a b c d : ℕ) : 
  a + 1 = b ∧ b + 1 = c ∧ c + 1 = d ∧ a * b * c * d = 4574880 → a = 43 :=
by sorry

end smallest_consecutive_number_l3473_347351


namespace egg_marble_distribution_unique_l3473_347378

/-- Represents the distribution of eggs and marbles among three groups. -/
structure EggMarbleDistribution where
  eggs_a : ℕ
  eggs_b : ℕ
  eggs_c : ℕ
  marbles_a : ℕ
  marbles_b : ℕ
  marbles_c : ℕ

/-- Checks if the given distribution satisfies all conditions. -/
def is_valid_distribution (d : EggMarbleDistribution) : Prop :=
  d.eggs_a + d.eggs_b + d.eggs_c = 15 ∧
  d.marbles_a + d.marbles_b + d.marbles_c = 4 ∧
  d.eggs_a ≠ d.eggs_b ∧ d.eggs_b ≠ d.eggs_c ∧ d.eggs_a ≠ d.eggs_c ∧
  d.eggs_b = d.marbles_b - d.marbles_a ∧
  d.eggs_c = d.marbles_c - d.marbles_b

theorem egg_marble_distribution_unique :
  ∃! d : EggMarbleDistribution, is_valid_distribution d ∧
    d.eggs_a = 12 ∧ d.eggs_b = 1 ∧ d.eggs_c = 2 :=
sorry

end egg_marble_distribution_unique_l3473_347378


namespace parabola_vertex_on_line_l3473_347320

/-- The parabola function -/
def f (c : ℝ) (x : ℝ) : ℝ := x^2 - 10*x + c

/-- The x-coordinate of the vertex of the parabola -/
def vertex_x : ℝ := 5

/-- The y-coordinate of the vertex of the parabola -/
def vertex_y (c : ℝ) : ℝ := f c vertex_x

/-- The theorem stating that the value of c for which the vertex of the parabola
    y = x^2 - 10x + c lies on the line y = 3 is 28 -/
theorem parabola_vertex_on_line : ∃ c : ℝ, vertex_y c = 3 ∧ c = 28 := by sorry

end parabola_vertex_on_line_l3473_347320


namespace line_equation_problem_1_line_equation_problem_2_l3473_347354

-- Define a line by its equation ax + by + c = 0
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Define the problem statements
theorem line_equation_problem_1 (l : Line) (A : Point) :
  A.x = 0 ∧ A.y = 2 ∧ 
  (l.a^2 / (l.a^2 + l.b^2) = 1/4) →
  ∃ (k : ℝ), k > 0 ∧ l.a = k * Real.sqrt 3 ∧ l.b = -3 * k ∧ l.c = 6 * k :=
sorry

theorem line_equation_problem_2 (l l₁ : Line) (A : Point) :
  A.x = 2 ∧ A.y = 1 ∧
  l₁.a = 3 ∧ l₁.b = 4 ∧ l₁.c = 5 ∧
  (l.a / l.b = (l₁.a / l₁.b) / 2) →
  ∃ (k : ℝ), k > 0 ∧ l.a = 3 * k ∧ l.b = -k ∧ l.c = -5 * k :=
sorry

end line_equation_problem_1_line_equation_problem_2_l3473_347354


namespace root_sum_equality_l3473_347385

-- Define the polynomial f(x)
def f (a b c : ℝ) (x : ℝ) : ℝ := x^4 + a*x^3 + b*x^2 + c*x

-- Define the theorem
theorem root_sum_equality 
  (a b c : ℝ) 
  (x₁ x₂ x₃ x₄ y₁ y₂ y₃ y₄ : ℝ) : 
  (f a b c x₁ = 1) ∧ (f a b c x₂ = 1) ∧ (f a b c x₃ = 1) ∧ (f a b c x₄ = 1) →
  (f a b c y₁ = 2) ∧ (f a b c y₂ = 2) ∧ (f a b c y₃ = 2) ∧ (f a b c y₄ = 2) →
  (x₁ + x₂ = x₃ + x₄) →
  (y₁ + y₂ = y₃ + y₄) :=
by sorry

end root_sum_equality_l3473_347385


namespace solution_sum_l3473_347370

/-- The solutions of the quadratic equation 2x(5x-11) = -10 -/
def solutions (x : ℝ) : Prop :=
  2 * x * (5 * x - 11) = -10

/-- The rational form of the solutions -/
def rational_form (m n p : ℤ) (x : ℝ) : Prop :=
  (x = (m + Real.sqrt n) / p) ∨ (x = (m - Real.sqrt n) / p)

/-- The theorem statement -/
theorem solution_sum (m n p : ℤ) :
  (∀ x, solutions x → rational_form m n p x) →
  Int.gcd m (Int.gcd n p) = 1 →
  m + n + p = 242 := by
  sorry

end solution_sum_l3473_347370


namespace license_plate_difference_l3473_347323

def california_plates := 26^3 * 10^4
def texas_plates := 26^3 * 10^3

theorem license_plate_difference :
  california_plates - texas_plates = 4553200000 := by
  sorry

end license_plate_difference_l3473_347323


namespace count_closest_to_two_sevenths_l3473_347371

def is_closest_to_two_sevenths (r : ℚ) : Prop :=
  ∀ n d : ℕ, n ≤ 2 → d > 0 → |r - 2/7| ≤ |r - (n : ℚ)/d|

def is_four_place_decimal (r : ℚ) : Prop :=
  ∃ a b c d : ℕ, a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧
    r = (a * 1000 + b * 100 + c * 10 + d) / 10000

theorem count_closest_to_two_sevenths :
  ∃! (s : Finset ℚ), 
    (∀ r ∈ s, is_four_place_decimal r ∧ is_closest_to_two_sevenths r) ∧
    s.card = 3 :=
sorry

end count_closest_to_two_sevenths_l3473_347371


namespace quadratic_always_positive_l3473_347366

theorem quadratic_always_positive (k : ℝ) :
  (∀ x : ℝ, x^2 - (k - 3) * x - 2 * k + 12 > 0) ↔ -7 < k ∧ k < 5 := by
  sorry

end quadratic_always_positive_l3473_347366


namespace restaurant_glasses_count_l3473_347349

theorem restaurant_glasses_count :
  ∀ (x y : ℕ),
  -- x is the number of 12-glass boxes, y is the number of 16-glass boxes
  y = x + 16 →
  -- The average number of glasses per box is 15
  (12 * x + 16 * y) / (x + y) = 15 →
  -- The total number of glasses is 480
  12 * x + 16 * y = 480 :=
by
  sorry

end restaurant_glasses_count_l3473_347349


namespace inequality_proof_l3473_347309

theorem inequality_proof (a b c d : ℝ) 
  (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : 0 ≤ d) 
  (h5 : c + d ≤ a) (h6 : c + d ≤ b) : 
  a * d + b * c ≤ a * b := by
  sorry

end inequality_proof_l3473_347309


namespace smallest_two_digit_with_digit_product_12_l3473_347365

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99

def digit_product (n : ℕ) : ℕ :=
  (n / 10) * (n % 10)

theorem smallest_two_digit_with_digit_product_12 :
  ∃ (n : ℕ), is_two_digit n ∧ digit_product n = 12 ∧
  ∀ (m : ℕ), is_two_digit m → digit_product m = 12 → n ≤ m :=
by sorry

end smallest_two_digit_with_digit_product_12_l3473_347365


namespace square_side_length_l3473_347301

theorem square_side_length (circle_area : ℝ) (h1 : circle_area = 100) :
  ∃ (square_side : ℝ), square_side * 4 = circle_area ∧ square_side = 25 := by
  sorry

end square_side_length_l3473_347301


namespace rsa_congruence_l3473_347348

theorem rsa_congruence (p q e d M : ℕ) : 
  Nat.Prime p → 
  Nat.Prime q → 
  p ≠ q → 
  (e * d) % ((p - 1) * (q - 1)) = 1 → 
  ((M ^ e) ^ d) % (p * q) = M % (p * q) := by
sorry

end rsa_congruence_l3473_347348


namespace triangle_side_sum_l3473_347328

theorem triangle_side_sum (a b c : ℝ) (A B C : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < c →
  0 < A ∧ 0 < B ∧ 0 < C →
  A + B + C = π →
  b * Real.cos C + c * Real.cos B = 3 * a * Real.cos B →
  b = 2 →
  (1 / 2) * a * b * Real.sin C = (3 * Real.sqrt 2) / 2 →
  a + c = 4 := by
sorry


end triangle_side_sum_l3473_347328


namespace two_thirds_of_45_plus_10_l3473_347376

theorem two_thirds_of_45_plus_10 : ((2 : ℚ) / 3) * 45 + 10 = 40 := by
  sorry

end two_thirds_of_45_plus_10_l3473_347376


namespace base5_divisibility_by_29_l3473_347321

def base5ToDecimal (a b c d : ℕ) : ℕ := a * 5^3 + b * 5^2 + c * 5^1 + d * 5^0

def isDivisibleBy29 (n : ℕ) : Prop := ∃ k : ℕ, n = 29 * k

theorem base5_divisibility_by_29 (y : ℕ) :
  isDivisibleBy29 (base5ToDecimal 4 2 y 3) ↔ y = 4 := by sorry

end base5_divisibility_by_29_l3473_347321


namespace only_negative_one_point_one_less_than_negative_one_l3473_347331

theorem only_negative_one_point_one_less_than_negative_one :
  let numbers : List ℝ := [0, 1, -0.9, -1.1]
  ∀ x ∈ numbers, x < -1 ↔ x = -1.1 :=
by sorry

end only_negative_one_point_one_less_than_negative_one_l3473_347331


namespace sin_shift_left_l3473_347386

/-- Shifting a sinusoidal function to the left -/
theorem sin_shift_left (x : ℝ) :
  let f (t : ℝ) := Real.sin (2 * t)
  let g (t : ℝ) := f (t + π / 6)
  g x = Real.sin (2 * x + π / 3) :=
by sorry

end sin_shift_left_l3473_347386


namespace mikes_candies_l3473_347383

theorem mikes_candies (initial_candies : ℕ) : 
  (initial_candies > 0) →
  (initial_candies % 4 = 0) →
  (∃ (sister_took : ℕ), 1 ≤ sister_took ∧ sister_took ≤ 4 ∧
    5 + sister_took = initial_candies * 3 / 4 * 2 / 3 - 24) →
  initial_candies = 64 := by
sorry

end mikes_candies_l3473_347383


namespace bee_multiple_l3473_347364

theorem bee_multiple (bees_day1 bees_day2 : ℕ) (h1 : bees_day1 = 144) (h2 : bees_day2 = 432) :
  bees_day2 / bees_day1 = 3 := by
  sorry

end bee_multiple_l3473_347364


namespace real_part_zero_necessary_not_sufficient_l3473_347360

/-- A complex number is purely imaginary if and only if its real part is zero and its imaginary part is non-zero. -/
def is_purely_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

/-- The condition "real part is zero" is necessary but not sufficient for a complex number to be purely imaginary. -/
theorem real_part_zero_necessary_not_sufficient :
  ∀ (a b : ℝ), 
    (∀ (z : ℂ), is_purely_imaginary z → z.re = 0) ∧
    ¬(∀ (z : ℂ), z.re = 0 → is_purely_imaginary z) :=
by sorry

end real_part_zero_necessary_not_sufficient_l3473_347360


namespace grasshopper_frog_jump_contest_l3473_347327

theorem grasshopper_frog_jump_contest (grasshopper_jump frog_jump : ℕ) 
  (h1 : grasshopper_jump = 31) 
  (h2 : frog_jump = 35) : 
  grasshopper_jump + frog_jump = 66 := by
  sorry

end grasshopper_frog_jump_contest_l3473_347327


namespace gas_pressure_final_l3473_347318

/-- Given a gas with pressure inversely proportional to volume, prove the final pressure -/
theorem gas_pressure_final (p₀ v₀ v₁ v₂ : ℝ) (h₀ : p₀ > 0) (h₁ : v₀ > 0) (h₂ : v₁ > 0) (h₃ : v₂ > 0)
  (h_initial : p₀ * v₀ = 6 * 3.6)
  (h_v₁ : v₁ = 7.2)
  (h_v₂ : v₂ = 3.6)
  (h_half : v₂ = v₀) :
  ∃ (p₂ : ℝ), p₂ * v₂ = p₀ * v₀ ∧ p₂ = 6 := by
  sorry

#check gas_pressure_final

end gas_pressure_final_l3473_347318


namespace purely_imaginary_complex_fraction_l3473_347380

theorem purely_imaginary_complex_fraction (a b : ℝ) (h : b ≠ 0) :
  (∃ (k : ℝ), (Complex.I : ℂ) * k = (a + Complex.I * b) / (4 + Complex.I * 3)) →
  a / b = -3/4 := by
sorry

end purely_imaginary_complex_fraction_l3473_347380


namespace bmw_sales_count_l3473_347379

def total_cars : ℕ := 300
def ford_percentage : ℚ := 20 / 100
def nissan_percentage : ℚ := 25 / 100
def volkswagen_percentage : ℚ := 10 / 100

theorem bmw_sales_count :
  (total_cars : ℚ) * (1 - (ford_percentage + nissan_percentage + volkswagen_percentage)) = 135 := by
  sorry

end bmw_sales_count_l3473_347379


namespace smallest_four_digit_number_with_second_digit_6_l3473_347396

/-- A function that returns true if all digits in a number are different --/
def allDigitsDifferent (n : ℕ) : Prop := sorry

/-- A function that returns the digit at a specific position in a number --/
def digitAt (n : ℕ) (pos : ℕ) : ℕ := sorry

theorem smallest_four_digit_number_with_second_digit_6 :
  ∀ n : ℕ,
  (1000 ≤ n ∧ n < 10000) →  -- four-digit number
  (digitAt n 2 = 6) →       -- second digit is 6
  allDigitsDifferent n →    -- all digits are different
  1602 ≤ n :=
by sorry

end smallest_four_digit_number_with_second_digit_6_l3473_347396


namespace cardboard_pins_l3473_347312

/-- Calculates the total number of pins used on a rectangular cardboard -/
def total_pins (length width pins_per_side : ℕ) : ℕ :=
  2 * pins_per_side * (length + width)

/-- Theorem: For a 34 * 14 cardboard with 35 pins per side, the total pins used is 140 -/
theorem cardboard_pins :
  total_pins 34 14 35 = 140 := by
  sorry

end cardboard_pins_l3473_347312


namespace negation_of_universal_proposition_l3473_347336

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 > 0) ↔ (∃ x : ℝ, x^2 ≤ 0) := by sorry

end negation_of_universal_proposition_l3473_347336


namespace circle_center_l3473_347356

/-- The center of a circle given by the equation x^2 - 8x + y^2 + 4y = -3 -/
theorem circle_center (x y : ℝ) : 
  (x^2 - 8*x + y^2 + 4*y = -3) → (∃ r : ℝ, (x - 4)^2 + (y + 2)^2 = r^2) :=
by sorry

end circle_center_l3473_347356


namespace sin_sum_to_product_l3473_347322

theorem sin_sum_to_product (x : ℝ) : 
  Real.sin (3 * x) + Real.sin (7 * x) = 2 * Real.sin (5 * x) * Real.cos (2 * x) := by
  sorry

end sin_sum_to_product_l3473_347322


namespace ariane_victory_condition_l3473_347338

/-- The game between Ariane and Bérénice -/
def game (n : ℕ) : Prop :=
  1 ≤ n ∧ n ≤ 30 ∧
  ∃ (S : Finset ℕ),
    S.card = n ∧
    (∀ x ∈ S, x ≥ 1 ∧ x ≤ 30) ∧
    (∀ d : ℕ, d ≥ 2 →
      (∃ a b : ℕ, a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ d ∣ a ∧ d ∣ b) ∨
      (∀ a b : ℕ, a ∈ S → b ∈ S → a ≠ b → ¬(d ∣ a ∧ d ∣ b)))

/-- Ariane's winning condition -/
def ariane_wins (n : ℕ) : Prop :=
  game n ∧
  ∃ (S : Finset ℕ),
    S.card = n ∧
    (∀ x ∈ S, x ≥ 1 ∧ x ≤ 30) ∧
    ∀ d : ℕ, d ≥ 2 →
      ∀ a b : ℕ, a ∈ S → b ∈ S → a ≠ b → ¬(d ∣ a ∧ d ∣ b)

/-- The main theorem: Ariane can ensure victory if and only if 1 ≤ n ≤ 11 -/
theorem ariane_victory_condition :
  ∀ n : ℕ, ariane_wins n ↔ (1 ≤ n ∧ n ≤ 11) :=
sorry

end ariane_victory_condition_l3473_347338


namespace given_number_eq_scientific_repr_l3473_347353

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  property : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- The given number -0.000032 -/
def given_number : ℝ := -0.000032

/-- The scientific notation representation of the given number -/
def scientific_repr : ScientificNotation :=
  { coefficient := -3.2
    exponent := -5
    property := by sorry }

/-- Theorem stating that the given number is equal to its scientific notation representation -/
theorem given_number_eq_scientific_repr :
  given_number = scientific_repr.coefficient * (10 : ℝ) ^ scientific_repr.exponent := by
  sorry

end given_number_eq_scientific_repr_l3473_347353


namespace triangle_side_length_l3473_347381

theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) : 
  a = 10 → B = π/3 → C = π/4 → 
  c = 10 * (Real.sqrt 3 - 1) :=
by
  sorry

end triangle_side_length_l3473_347381


namespace discount_percentage_l3473_347314

theorem discount_percentage (tshirt_cost pants_cost shoes_cost : ℝ)
  (tshirt_qty pants_qty shoes_qty : ℕ)
  (total_paid : ℝ)
  (h1 : tshirt_cost = 20)
  (h2 : pants_cost = 80)
  (h3 : shoes_cost = 150)
  (h4 : tshirt_qty = 4)
  (h5 : pants_qty = 3)
  (h6 : shoes_qty = 2)
  (h7 : total_paid = 558) :
  (1 - total_paid / (tshirt_cost * tshirt_qty + pants_cost * pants_qty + shoes_cost * shoes_qty)) * 100 = 10 := by
sorry

end discount_percentage_l3473_347314


namespace quadratic_rewrite_l3473_347392

theorem quadratic_rewrite (d e f : ℤ) :
  (∀ x : ℝ, 16 * x^2 - 40 * x - 56 = (d * x + e)^2 + f) →
  d * e = -20 := by
sorry

end quadratic_rewrite_l3473_347392


namespace a_range_l3473_347324

noncomputable def f (x : ℝ) : ℝ := 1 / Real.exp x - Real.exp x + 2 * x - (1 / 3) * x^3

theorem a_range (a : ℝ) : f (3 * a^2) + f (2 * a - 1) ≥ 0 → a ∈ Set.Icc (-1) (1/3) := by
  sorry

end a_range_l3473_347324


namespace average_monthly_income_l3473_347390

/-- Given a person's expenses and savings over a year, calculate their average monthly income. -/
theorem average_monthly_income
  (expense_first_3_months : ℕ)
  (expense_next_4_months : ℕ)
  (expense_last_5_months : ℕ)
  (yearly_savings : ℕ)
  (h1 : expense_first_3_months = 1700)
  (h2 : expense_next_4_months = 1550)
  (h3 : expense_last_5_months = 1800)
  (h4 : yearly_savings = 5200) :
  (3 * expense_first_3_months + 4 * expense_next_4_months + 5 * expense_last_5_months + yearly_savings) / 12 = 2125 := by
  sorry

end average_monthly_income_l3473_347390


namespace trent_onions_per_pot_l3473_347307

/-- The number of pots of soup Trent is making -/
def num_pots : ℕ := 6

/-- The total number of tears Trent cries -/
def total_tears : ℕ := 16

/-- The ratio of tears to onions -/
def tear_to_onion_ratio : ℚ := 2 / 3

/-- The number of onions Trent needs to chop per pot of soup -/
def onions_per_pot : ℕ := 4

theorem trent_onions_per_pot :
  onions_per_pot * num_pots * tear_to_onion_ratio = total_tears :=
sorry

end trent_onions_per_pot_l3473_347307


namespace initial_water_is_11_l3473_347315

/-- Represents the hiking scenario with given conditions -/
structure HikeScenario where
  hikeLength : ℝ
  hikeDuration : ℝ
  leakRate : ℝ
  lastMileConsumption : ℝ
  regularConsumption : ℝ
  remainingWater : ℝ

/-- Calculates the initial amount of water in the canteen -/
def initialWater (scenario : HikeScenario) : ℝ :=
  scenario.regularConsumption * (scenario.hikeLength - 1) +
  scenario.lastMileConsumption +
  scenario.leakRate * scenario.hikeDuration +
  scenario.remainingWater

/-- Theorem stating that the initial amount of water is 11 cups -/
theorem initial_water_is_11 (scenario : HikeScenario) 
  (hLength : scenario.hikeLength = 7)
  (hDuration : scenario.hikeDuration = 3)
  (hLeak : scenario.leakRate = 1)
  (hLastMile : scenario.lastMileConsumption = 3)
  (hRegular : scenario.regularConsumption = 0.5)
  (hRemaining : scenario.remainingWater = 2) :
  initialWater scenario = 11 := by
  sorry

end initial_water_is_11_l3473_347315


namespace collinear_vectors_x_value_l3473_347329

/-- Given vectors a, b, and c in ℝ², prove that if 3a + b is collinear with c, then x = 4 -/
theorem collinear_vectors_x_value (a b c : ℝ × ℝ) (x : ℝ) 
  (ha : a = (-2, 0)) 
  (hb : b = (2, 1)) 
  (hc : c = (x, -1)) 
  (hcollinear : ∃ (k : ℝ), k ≠ 0 ∧ 3 • a + b = k • c) : 
  x = 4 := by
  sorry

end collinear_vectors_x_value_l3473_347329


namespace symmetric_line_x_axis_l3473_347316

/-- The equation of a line symmetric to another line with respect to the x-axis -/
theorem symmetric_line_x_axis (a b c : ℝ) :
  (∀ x y, a * x + b * y + c = 0 ↔ a * x - b * y - c = 0) →
  (∀ x y, 3 * x + 4 * y - 5 = 0 ↔ 3 * x - 4 * y + 5 = 0) :=
by sorry

end symmetric_line_x_axis_l3473_347316


namespace cost_of_paving_floor_l3473_347375

/-- The cost of paving a rectangular floor -/
theorem cost_of_paving_floor (length width rate : ℝ) : 
  length = 6 → width = 4.75 → rate = 900 → length * width * rate = 25650 := by sorry

end cost_of_paving_floor_l3473_347375


namespace ab_greater_than_b_squared_l3473_347358

theorem ab_greater_than_b_squared {a b : ℝ} (h1 : a < b) (h2 : b < 0) : a * b > b ^ 2 := by
  sorry

end ab_greater_than_b_squared_l3473_347358


namespace average_of_first_five_multiples_of_five_l3473_347330

theorem average_of_first_five_multiples_of_five :
  let multiples : List ℕ := [5, 10, 15, 20, 25]
  (multiples.sum / multiples.length : ℚ) = 15 := by
  sorry

end average_of_first_five_multiples_of_five_l3473_347330


namespace power_zero_plus_power_division_l3473_347373

theorem power_zero_plus_power_division (x y : ℕ) : 3^0 + 9^5 / 9^3 = 82 := by
  sorry

end power_zero_plus_power_division_l3473_347373


namespace largest_integer_with_remainder_l3473_347344

theorem largest_integer_with_remainder : ∃ n : ℕ, n = 95 ∧ 
  n < 100 ∧ 
  n % 7 = 4 ∧ 
  ∀ m : ℕ, m < 100 → m % 7 = 4 → m ≤ n :=
by sorry

end largest_integer_with_remainder_l3473_347344


namespace parabola_midpoint_to_directrix_l3473_347355

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola y² = 2px -/
structure Parabola where
  p : ℝ
  h : p > 0

/-- Theorem statement -/
theorem parabola_midpoint_to_directrix 
  (para : Parabola) 
  (A B M : Point) 
  (h_line : (B.y - A.y) / (B.x - A.x) = 1) -- Slope of line AB is 1
  (h_on_parabola : A.y^2 = 2 * para.p * A.x ∧ B.y^2 = 2 * para.p * B.x) -- A and B are on the parabola
  (h_midpoint : M.x = (A.x + B.x) / 2 ∧ M.y = (A.y + B.y) / 2) -- M is midpoint of AB
  (h_m_y : M.y = 2) -- y-coordinate of M is 2
  : M.x - (-para.p) = 4 := by sorry

end parabola_midpoint_to_directrix_l3473_347355


namespace inequality_range_l3473_347306

theorem inequality_range (a x : ℝ) : 
  (∀ a, |a| ≤ 1 → x^2 + (a - 6) * x + (9 - 3 * a) > 0) ↔ 
  (x < 2 ∨ x > 4) := by sorry

end inequality_range_l3473_347306


namespace arithmetic_sequence_sum_l3473_347308

-- Define an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- State the theorem
theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  is_arithmetic_sequence a → a 4 + a 8 = 16 → a 2 + a 10 = 16 := by
  sorry

end arithmetic_sequence_sum_l3473_347308


namespace distance_on_parametric_line_l3473_347399

/-- The distance between two points on a parametric line --/
theorem distance_on_parametric_line :
  let line : ℝ → ℝ × ℝ := λ t ↦ (1 + 3 * t, 1 + t)
  let point1 := line 0
  let point2 := line 1
  (point1.1 - point2.1)^2 + (point1.2 - point2.2)^2 = 10 := by
  sorry

end distance_on_parametric_line_l3473_347399


namespace max_value_quadratic_l3473_347359

theorem max_value_quadratic (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x^2 - 2*x*y + 3*y^2 = 9) : 
  x^2 + 2*x*y + 3*y^2 ≤ (117 + 36*Real.sqrt 3) / 11 := by
  sorry

end max_value_quadratic_l3473_347359
