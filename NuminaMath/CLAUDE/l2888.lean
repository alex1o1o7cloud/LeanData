import Mathlib

namespace distance_to_place_l2888_288841

/-- Proves that the distance to a place is 72 km given the specified conditions -/
theorem distance_to_place (still_water_speed : ℝ) (current_speed : ℝ) (total_time : ℝ) :
  still_water_speed = 10 →
  current_speed = 2 →
  total_time = 15 →
  ∃ (distance : ℝ), distance = 72 ∧
    distance / (still_water_speed - current_speed) +
    distance / (still_water_speed + current_speed) = total_time :=
by sorry

end distance_to_place_l2888_288841


namespace honor_distribution_proof_l2888_288822

/-- The number of ways to distribute honors among people -/
def distribute_honors (num_honors num_people : ℕ) (incompatible_pair : Bool) : ℕ :=
  sorry

/-- The number of ways to distribute honors in the specific problem -/
def problem_distribution : ℕ :=
  distribute_honors 5 3 true

theorem honor_distribution_proof :
  problem_distribution = 114 := by sorry

end honor_distribution_proof_l2888_288822


namespace total_money_l2888_288850

/-- Given that A and C together have 400, B and C together have 750, and C has 250,
    prove that the total amount of money A, B, and C have between them is 900. -/
theorem total_money (a b c : ℕ) 
  (h1 : a + c = 400)
  (h2 : b + c = 750)
  (h3 : c = 250) :
  a + b + c = 900 := by
  sorry

end total_money_l2888_288850


namespace A_B_symmetric_x_l2888_288848

-- Define the points A and B
def A : ℝ × ℝ := (-2, 3)
def B : ℝ × ℝ := (-2, -3)

-- Define symmetry with respect to x-axis
def symmetric_x (p q : ℝ × ℝ) : Prop :=
  p.1 = q.1 ∧ p.2 = -q.2

-- Theorem statement
theorem A_B_symmetric_x : symmetric_x A B := by
  sorry

end A_B_symmetric_x_l2888_288848


namespace marble_selection_ways_l2888_288865

theorem marble_selection_ways (n m : ℕ) (h1 : n = 9) (h2 : m = 4) :
  Nat.choose n m = 126 := by
  sorry

end marble_selection_ways_l2888_288865


namespace square_area_on_parabola_l2888_288830

/-- Given a square with one side on the line y = 7 and endpoints on the parabola y = x^2 + 4x + 3,
    prove that its area is 32. -/
theorem square_area_on_parabola : 
  ∀ (x₁ x₂ : ℝ),
  (x₁^2 + 4*x₁ + 3 = 7) →
  (x₂^2 + 4*x₂ + 3 = 7) →
  x₁ ≠ x₂ →
  (x₂ - x₁)^2 = 32 := by
sorry

end square_area_on_parabola_l2888_288830


namespace platform_length_l2888_288898

/-- The length of a platform given train crossing times -/
theorem platform_length (train_length : ℝ) (post_time : ℝ) (platform_time : ℝ) :
  train_length = 150 →
  post_time = 15 →
  platform_time = 25 →
  ∃ (platform_length : ℝ),
    platform_length = 100 ∧
    train_length / post_time = (train_length + platform_length) / platform_time :=
by sorry

end platform_length_l2888_288898


namespace complex_number_conditions_l2888_288867

theorem complex_number_conditions (α : ℂ) :
  α ≠ 1 →
  Complex.abs (α^2 - 1) = 3 * Complex.abs (α - 1) →
  Complex.abs (α^3 - 1) = 6 * Complex.abs (α - 1) →
  α = -1 := by
sorry

end complex_number_conditions_l2888_288867


namespace value_of_equation_l2888_288892

theorem value_of_equation (x y V : ℝ) 
  (eq1 : x + |x| + y = V)
  (eq2 : x + |y| - y = 6)
  (eq3 : x + y = 12) :
  V = 18 := by
  sorry

end value_of_equation_l2888_288892


namespace circle_tangency_count_l2888_288851

theorem circle_tangency_count : ∃ (S : Finset ℕ), 
  (∀ r ∈ S, r < 120 ∧ 120 % r = 0) ∧ 
  (∀ r < 120, 120 % r = 0 → r ∈ S) ∧ 
  Finset.card S = 15 := by
sorry

end circle_tangency_count_l2888_288851


namespace system_solution_ratio_l2888_288853

theorem system_solution_ratio (x y c d : ℝ) : 
  x ≠ 0 → y ≠ 0 → d ≠ 0 → 
  (4 * x + 5 * y = c) → (8 * y - 10 * x = d) → 
  c / d = 1 / 2 := by
sorry

end system_solution_ratio_l2888_288853


namespace geometric_sequence_product_l2888_288886

/-- A geometric sequence with a_2 = 5 and a_6 = 33 -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (q : ℝ), ∀ (n : ℕ), a (n + 1) = a n * q ∧ a 2 = 5 ∧ a 6 = 33

/-- The product of a_3 and a_5 in the geometric sequence is 165 -/
theorem geometric_sequence_product (a : ℕ → ℝ) (h : geometric_sequence a) :
  a 3 * a 5 = 165 := by
  sorry

end geometric_sequence_product_l2888_288886


namespace line_through_point_with_given_segment_length_l2888_288825

-- Define the angle BAC
def Angle (A B C : ℝ × ℝ) : Prop := sorry

-- Define a point on the angle bisector
def OnAngleBisector (D A B C : ℝ × ℝ) : Prop := sorry

-- Define a line passing through two points
def Line (P Q : ℝ × ℝ) : Set (ℝ × ℝ) := sorry

-- Define the length of a segment
def SegmentLength (P Q : ℝ × ℝ) : ℝ := sorry

-- Define a point being on a line
def OnLine (P : ℝ × ℝ) (L : Set (ℝ × ℝ)) : Prop := sorry

theorem line_through_point_with_given_segment_length 
  (A B C D : ℝ × ℝ) (l : ℝ) 
  (h1 : Angle A B C) 
  (h2 : OnAngleBisector D A B C) 
  (h3 : l > 0) : 
  ∃ (E F : ℝ × ℝ), 
    OnLine E (Line A B) ∧ 
    OnLine F (Line A C) ∧ 
    OnLine D (Line E F) ∧ 
    SegmentLength E F = l := 
sorry

end line_through_point_with_given_segment_length_l2888_288825


namespace absolute_value_equation_solution_l2888_288821

theorem absolute_value_equation_solution :
  ∃! y : ℚ, |5 * y - 6| = 0 ∧ y = 6/5 := by
  sorry

end absolute_value_equation_solution_l2888_288821


namespace prom_services_cost_l2888_288855

/-- Calculate the total cost of prom services for Keesha --/
theorem prom_services_cost : 
  let hair_cost : ℚ := 50
  let hair_discount : ℚ := 0.1
  let manicure_cost : ℚ := 30
  let pedicure_cost : ℚ := 35
  let pedicure_discount : ℚ := 0.5
  let makeup_cost : ℚ := 40
  let makeup_tax : ℚ := 0.05
  let tip_percentage : ℚ := 0.2

  let hair_total := (hair_cost * (1 - hair_discount)) * (1 + tip_percentage)
  let nails_total := (manicure_cost + pedicure_cost * pedicure_discount) * (1 + tip_percentage)
  let makeup_total := (makeup_cost * (1 + makeup_tax)) * (1 + tip_percentage)

  hair_total + nails_total + makeup_total = 161.4 := by
    sorry

end prom_services_cost_l2888_288855


namespace sqrt_equation_solution_l2888_288849

theorem sqrt_equation_solution :
  ∃! z : ℚ, Real.sqrt (3 - 4*z) = 7 ∧ z = -23/2 := by
  sorry

end sqrt_equation_solution_l2888_288849


namespace investment_solution_l2888_288807

/-- Represents the investment scenario described in the problem -/
structure Investment where
  total : ℝ
  rate1 : ℝ
  rate2 : ℝ
  years : ℕ
  finalAmount : ℝ

/-- Calculates the final amount after compound interest -/
def compoundInterest (principal : ℝ) (rate : ℝ) (years : ℕ) : ℝ :=
  principal * (1 + rate) ^ years

/-- Theorem stating the solution to the investment problem -/
theorem investment_solution (inv : Investment) 
  (h1 : inv.total = 1500)
  (h2 : inv.rate1 = 0.04)
  (h3 : inv.rate2 = 0.06)
  (h4 : inv.years = 3)
  (h5 : inv.finalAmount = 1824.89) :
  ∃ (x : ℝ), x = 580 ∧ 
    compoundInterest x inv.rate1 inv.years + 
    compoundInterest (inv.total - x) inv.rate2 inv.years = 
    inv.finalAmount := by
  sorry


end investment_solution_l2888_288807


namespace line_circle_intersection_point_on_line_point_inside_circle_l2888_288883

/-- The line y = kx + 1 intersects the circle x^2 + y^2 = 2 but doesn't pass through its center -/
theorem line_circle_intersection (k : ℝ) : 
  ∃ (x y : ℝ), y = k * x + 1 ∧ x^2 + y^2 = 2 ∧ (x ≠ 0 ∨ y ≠ 0) := by
  sorry

/-- The point (0, 1) is always on the line y = kx + 1 -/
theorem point_on_line (k : ℝ) : k * 0 + 1 = 1 := by
  sorry

/-- The point (0, 1) is inside the circle x^2 + y^2 = 2 -/
theorem point_inside_circle : 0^2 + 1^2 < 2 := by
  sorry

end line_circle_intersection_point_on_line_point_inside_circle_l2888_288883


namespace integral_equals_two_plus_half_pi_l2888_288808

open Set
open MeasureTheory
open Interval

theorem integral_equals_two_plus_half_pi :
  ∫ x in (Icc (-1) 1), (1 + x + Real.sqrt (1 - x^2)) = 2 + π / 2 := by
  sorry

end integral_equals_two_plus_half_pi_l2888_288808


namespace tree_growth_fraction_l2888_288888

/-- Represents the growth of a tree over time -/
def TreeGrowth (initial_height : ℝ) (growth_rate : ℝ) (years : ℕ) : ℝ :=
  initial_height + growth_rate * years

theorem tree_growth_fraction :
  let initial_height : ℝ := 4
  let growth_rate : ℝ := 0.5
  let height_at_4_years := TreeGrowth initial_height growth_rate 4
  let height_at_6_years := TreeGrowth initial_height growth_rate 6
  (height_at_6_years - height_at_4_years) / height_at_4_years = 1 / 6 := by
  sorry

end tree_growth_fraction_l2888_288888


namespace fraction_is_positive_integer_l2888_288845

theorem fraction_is_positive_integer (p : ℕ+) :
  (∃ k : ℕ+, (5 * p + 15 : ℚ) / (3 * p - 9 : ℚ) = k) ↔ 4 ≤ p ∧ p ≤ 19 := by
  sorry

end fraction_is_positive_integer_l2888_288845


namespace binomial_odd_even_difference_squares_l2888_288810

variable (x a : ℝ) (n : ℕ)

def A (x a : ℝ) (n : ℕ) : ℝ := sorry
def B (x a : ℝ) (n : ℕ) : ℝ := sorry

/-- For the binomial expansion (x+a)^n, where A is the sum of odd-position terms
    and B is the sum of even-position terms, A^2 - B^2 = (x^2 - a^2)^n -/
theorem binomial_odd_even_difference_squares :
  (A x a n)^2 - (B x a n)^2 = (x^2 - a^2)^n := by sorry

end binomial_odd_even_difference_squares_l2888_288810


namespace square_sum_reciprocal_l2888_288854

theorem square_sum_reciprocal (m : ℝ) (hm : m > 0) (h : m - 1/m = 3) : 
  m^2 + 1/m^2 = 11 := by
sorry

end square_sum_reciprocal_l2888_288854


namespace janet_initial_clips_l2888_288836

/-- The number of paper clips Janet had in the morning -/
def initial_clips : ℕ := sorry

/-- The number of paper clips Janet used during the day -/
def used_clips : ℕ := 59

/-- The number of paper clips Janet had left at the end of the day -/
def remaining_clips : ℕ := 26

/-- Theorem: Janet had 85 paper clips in the morning -/
theorem janet_initial_clips : initial_clips = 85 := by sorry

end janet_initial_clips_l2888_288836


namespace egyptian_fraction_sum_l2888_288877

theorem egyptian_fraction_sum : ∃ (b₂ b₃ b₄ b₅ b₆ b₇ b₈ : ℕ),
  (5 : ℚ) / 8 = b₂ / 2 + b₃ / 6 + b₄ / 24 + b₅ / 120 + b₆ / 720 + b₇ / 5040 + b₈ / 40320 ∧
  b₂ < 2 ∧ b₃ < 3 ∧ b₄ < 4 ∧ b₅ < 5 ∧ b₆ < 6 ∧ b₇ < 7 ∧ b₈ < 8 ∧
  b₂ + b₃ + b₄ + b₅ + b₆ + b₇ + b₈ = 4 := by
  sorry

end egyptian_fraction_sum_l2888_288877


namespace profit_achieved_l2888_288885

/-- The minimum number of disks Maria needs to sell to make a profit of $120 -/
def disks_to_sell : ℕ := 219

/-- The cost of buying 5 disks -/
def buy_price : ℚ := 6

/-- The selling price of 4 disks -/
def sell_price : ℚ := 7

/-- The desired profit -/
def target_profit : ℚ := 120

theorem profit_achieved :
  let cost_per_disk : ℚ := buy_price / 5
  let revenue_per_disk : ℚ := sell_price / 4
  let profit_per_disk : ℚ := revenue_per_disk - cost_per_disk
  (disks_to_sell : ℚ) * profit_per_disk ≥ target_profit ∧
  ∀ n : ℕ, (n : ℚ) * profit_per_disk < target_profit → n < disks_to_sell :=
by sorry

end profit_achieved_l2888_288885


namespace expression_evaluation_l2888_288815

theorem expression_evaluation :
  let a : ℤ := -2
  let b : ℤ := 4
  (-(-3*a)^2 + 6*a*b) - (a^2 + 3*(a - 2*a*b)) = 14 :=
by sorry

end expression_evaluation_l2888_288815


namespace lamp_savings_l2888_288866

theorem lamp_savings (num_lamps : ℕ) (original_price : ℚ) (discount_rate : ℚ) (additional_discount : ℚ) :
  num_lamps = 3 →
  original_price = 15 →
  discount_rate = 0.25 →
  additional_discount = 5 →
  num_lamps * original_price - (num_lamps * (original_price * (1 - discount_rate)) - additional_discount) = 16.25 :=
by sorry

end lamp_savings_l2888_288866


namespace money_problem_l2888_288801

theorem money_problem (a b : ℝ) 
  (h1 : 4 * a + 2 * b > 110)
  (h2 : 2 * a + 3 * b = 105) :
  a > 15 ∧ b < 25 := by
  sorry

end money_problem_l2888_288801


namespace expression_factorization_l2888_288878

theorem expression_factorization (a b c d : ℝ) : 
  a^3 * (b^2 - d^2) + b^3 * (c^2 - a^2) + c^3 * (d^2 - b^2) + d^3 * (a^2 - c^2) = 
  (a - b) * (b - c) * (c - d) * (d - a) * 
  (a^2 + a*b + a*c + a*d + b^2 + b*c + b*d + c^2 + c*d + d^2) := by
  sorry

end expression_factorization_l2888_288878


namespace sequence_sum_theorem_l2888_288833

def sequence_a (n : ℕ) : ℕ :=
  2 * n

def sum_S (n : ℕ) : ℕ :=
  n * n

def sequence_b (n : ℕ) : ℚ :=
  1 / (n * (n + 1))

def sum_T (n : ℕ) : ℚ :=
  n / (n + 1)

theorem sequence_sum_theorem (n : ℕ) :
  sequence_a 2 = 4 ∧
  (∀ k : ℕ, sequence_a (k + 1) = sequence_a k + 2) ∧
  (∀ k : ℕ, sum_S k = k * k) ∧
  (∀ k : ℕ, sequence_b k = 1 / sum_S k) →
  sum_T n = n / (n + 1) := by
  sorry

end sequence_sum_theorem_l2888_288833


namespace unit_digit_of_fraction_l2888_288875

theorem unit_digit_of_fraction : 
  (998 * 999 * 1000 * 1001 * 1002 * 1003) / 10000 % 10 = 6 := by sorry

end unit_digit_of_fraction_l2888_288875


namespace geometric_sequence_problem_l2888_288870

theorem geometric_sequence_problem (a : ℕ → ℝ) (q : ℝ) :
  (∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r) →  -- geometric sequence condition
  (abs q > 1) →
  (a 2 + a 7 = 2) →
  (a 4 * a 5 = -15) →
  (a 12 = -25/3) :=
by sorry

end geometric_sequence_problem_l2888_288870


namespace certain_number_proof_l2888_288894

theorem certain_number_proof : ∃ x : ℝ, (20 + x + 60) / 3 = (20 + 60 + 25) / 3 + 5 :=
by
  sorry

end certain_number_proof_l2888_288894


namespace isabel_toy_cost_l2888_288818

theorem isabel_toy_cost (total_money : ℕ) (num_toys : ℕ) (cost_per_toy : ℕ) 
  (h1 : total_money = 14) 
  (h2 : num_toys = 7) 
  (h3 : total_money = num_toys * cost_per_toy) : 
  cost_per_toy = 2 := by
  sorry

end isabel_toy_cost_l2888_288818


namespace point_in_second_quadrant_l2888_288840

/-- A point in the xy-plane is represented by its coordinates -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of being in the second quadrant -/
def isInSecondQuadrant (p : Point) : Prop := p.x < 0 ∧ p.y > 0

/-- The point (-1,4) -/
def point : Point := ⟨-1, 4⟩

/-- Theorem: The point (-1,4) is in the second quadrant -/
theorem point_in_second_quadrant : isInSecondQuadrant point := by
  sorry

end point_in_second_quadrant_l2888_288840


namespace probability_two_red_balls_l2888_288820

def bag_total_balls : ℕ := 4
def red_balls : ℕ := 2
def white_balls : ℕ := 2
def drawn_balls : ℕ := 2

theorem probability_two_red_balls : 
  (Nat.choose red_balls drawn_balls : ℚ) / (Nat.choose bag_total_balls drawn_balls) = 1 / 6 :=
sorry

end probability_two_red_balls_l2888_288820


namespace position_after_four_steps_l2888_288881

/-- Given a number line where the distance from 0 to 30 is divided into 6 equal steps,
    the position reached after 4 steps is 20. -/
theorem position_after_four_steps :
  ∀ (total_distance : ℝ) (total_steps : ℕ) (steps_taken : ℕ),
    total_distance = 30 →
    total_steps = 6 →
    steps_taken = 4 →
    (total_distance / total_steps) * steps_taken = 20 :=
by
  sorry

#check position_after_four_steps

end position_after_four_steps_l2888_288881


namespace most_likely_gender_combination_l2888_288869

theorem most_likely_gender_combination (n : ℕ) (p : ℝ) : 
  n = 5 → p = 1/2 → 2 * (n.choose 3) * p^n = 5/8 := by sorry

end most_likely_gender_combination_l2888_288869


namespace percent_of_y_l2888_288811

theorem percent_of_y (y : ℝ) (h : y > 0) : ((4 * y) / 20 + (3 * y) / 10) / y = 1 / 2 := by
  sorry

end percent_of_y_l2888_288811


namespace paiges_flowers_l2888_288839

theorem paiges_flowers (flowers_per_bouquet : ℕ) (wilted_flowers : ℕ) (remaining_bouquets : ℕ) :
  flowers_per_bouquet = 7 →
  wilted_flowers = 18 →
  remaining_bouquets = 5 →
  flowers_per_bouquet * remaining_bouquets + wilted_flowers = 53 :=
by sorry

end paiges_flowers_l2888_288839


namespace choir_size_proof_l2888_288835

theorem choir_size_proof (n : ℕ) : 
  (∃ (p : ℕ), p > 10 ∧ Prime p ∧ p ∣ n) ∧ 
  9 ∣ n ∧ 10 ∣ n ∧ 12 ∣ n →
  n ≥ 1980 :=
by sorry

end choir_size_proof_l2888_288835


namespace set_equality_l2888_288829

-- Define the sets M, N, and P
def M : Set ℝ := {x | ∃ n : ℤ, x = n}
def N : Set ℝ := {x | ∃ n : ℤ, x = n / 2}
def P : Set ℝ := {x | ∃ n : ℤ, x = n + 1 / 2}

-- Theorem statement
theorem set_equality : N = M ∪ P := by
  sorry

end set_equality_l2888_288829


namespace box_with_balls_l2888_288826

theorem box_with_balls (total : ℕ) (white blue red : ℕ) : 
  total = 100 →
  blue = white + 12 →
  red = 2 * blue →
  total = white + blue + red →
  white = 16 := by
sorry

end box_with_balls_l2888_288826


namespace gcd_5factorial_8factorial_div_3factorial_l2888_288857

def factorial (n : ℕ) : ℕ := Nat.factorial n

theorem gcd_5factorial_8factorial_div_3factorial : 
  Nat.gcd (factorial 5) ((factorial 8) / (factorial 3)) = 120 := by sorry

end gcd_5factorial_8factorial_div_3factorial_l2888_288857


namespace waiter_customers_l2888_288809

/-- Represents the number of customers a waiter had at lunch -/
def lunch_customers (non_tipping : ℕ) (tip_amount : ℕ) (total_tips : ℕ) : ℕ :=
  non_tipping + (total_tips / tip_amount)

/-- Theorem stating the number of customers the waiter had at lunch -/
theorem waiter_customers :
  lunch_customers 4 9 27 = 7 := by
  sorry

end waiter_customers_l2888_288809


namespace quadratic_root_implies_a_l2888_288852

theorem quadratic_root_implies_a (a : ℝ) : (2^2 - 2 + a = 0) → a = -2 := by
  sorry

end quadratic_root_implies_a_l2888_288852


namespace probability_all_girls_l2888_288800

def total_members : ℕ := 12
def num_boys : ℕ := 7
def num_girls : ℕ := 5
def chosen_members : ℕ := 3

theorem probability_all_girls :
  (Nat.choose num_girls chosen_members : ℚ) / (Nat.choose total_members chosen_members) = 1 / 22 := by
  sorry

end probability_all_girls_l2888_288800


namespace sum_of_squares_l2888_288879

theorem sum_of_squares (x y : ℝ) (h1 : x + y = 18) (h2 : x * y = 52) : x^2 + y^2 = 220 := by
  sorry

end sum_of_squares_l2888_288879


namespace last_number_proof_l2888_288882

theorem last_number_proof (a b c d : ℝ) : 
  (a + b + c) / 3 = 6 →
  (b + c + d) / 3 = 3 →
  a + d = 13 →
  d = 2 := by
sorry

end last_number_proof_l2888_288882


namespace min_attempts_for_two_unknown_digits_l2888_288842

/-- Represents a phone number with known and unknown digits -/
structure PhoneNumber :=
  (known_digits : Nat)
  (unknown_digits : Nat)
  (total_digits : Nat)
  (h_total : total_digits = known_digits + unknown_digits)

/-- The number of possible combinations for the unknown digits -/
def possible_combinations (pn : PhoneNumber) : Nat :=
  10 ^ pn.unknown_digits

/-- The minimum number of attempts required to guarantee dialing the correct number -/
def min_attempts (pn : PhoneNumber) : Nat :=
  possible_combinations pn

theorem min_attempts_for_two_unknown_digits 
  (pn : PhoneNumber) 
  (h_seven_digits : pn.total_digits = 7) 
  (h_five_known : pn.known_digits = 5) 
  (h_two_unknown : pn.unknown_digits = 2) : 
  min_attempts pn = 100 := by
  sorry

#check min_attempts_for_two_unknown_digits

end min_attempts_for_two_unknown_digits_l2888_288842


namespace exactly_one_inscribed_rhombus_l2888_288895

/-- Represents a hyperbola in the xy-plane -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  eq : (x y : ℝ) → Prop

/-- The first hyperbola C₁: x²/a² - y²/b² = 1 -/
def C₁ (a b : ℝ) : Hyperbola :=
  { a := a
    b := b
    eq := fun x y ↦ x^2 / a^2 - y^2 / b^2 = 1 }

/-- The second hyperbola C₂: y²/b² - x²/a² = 1 -/
def C₂ (a b : ℝ) : Hyperbola :=
  { a := a
    b := b
    eq := fun x y ↦ y^2 / b^2 - x^2 / a^2 = 1 }

/-- A predicate indicating whether a hyperbola has an inscribed rhombus -/
def has_inscribed_rhombus (h : Hyperbola) : Prop := sorry

/-- The main theorem stating that exactly one of C₁ or C₂ has an inscribed rhombus -/
theorem exactly_one_inscribed_rhombus (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b) :
  (has_inscribed_rhombus (C₁ a b) ∧ ¬has_inscribed_rhombus (C₂ a b)) ∨
  (has_inscribed_rhombus (C₂ a b) ∧ ¬has_inscribed_rhombus (C₁ a b)) :=
sorry

end exactly_one_inscribed_rhombus_l2888_288895


namespace nested_cube_roots_l2888_288887

theorem nested_cube_roots (N M : ℝ) (hN : N > 1) (hM : M > 1) :
  (N * (M * (N * M^(1/3))^(1/3))^(1/3))^(1/3) = N^(2/3) * M^(2/3) := by
  sorry

end nested_cube_roots_l2888_288887


namespace fourth_root_equation_solutions_l2888_288891

theorem fourth_root_equation_solutions :
  ∀ x : ℝ, (x ^ (1/4) = 16 / (9 - x ^ (1/4))) ↔ (x = 4096 ∨ x = 1) := by
  sorry

end fourth_root_equation_solutions_l2888_288891


namespace point_in_first_quadrant_l2888_288831

theorem point_in_first_quadrant (a : ℕ+) :
  (4 > 0 ∧ 2 - a.val > 0) → a = 1 := by
  sorry

end point_in_first_quadrant_l2888_288831


namespace polynomial_simplification_l2888_288863

theorem polynomial_simplification (x : ℝ) : 
  x * (x * (x * (3 - x) - 6) + 12) + 2 = -x^4 + 3*x^3 - 6*x^2 + 12*x + 2 := by
  sorry

end polynomial_simplification_l2888_288863


namespace value_of_a_l2888_288880

/-- Converts paise to rupees -/
def paise_to_rupees (paise : ℚ) : ℚ := paise / 100

/-- The problem statement -/
theorem value_of_a (a : ℚ) (h : (0.5 / 100) * a = 75) : 
  paise_to_rupees a = 150 := by sorry

end value_of_a_l2888_288880


namespace firm_employs_50_looms_l2888_288817

/-- Represents the number of looms employed by a textile manufacturing firm. -/
def number_of_looms : ℕ := sorry

/-- The aggregate sales value of the output of the looms in rupees. -/
def aggregate_sales : ℕ := 500000

/-- The monthly manufacturing expenses in rupees. -/
def manufacturing_expenses : ℕ := 150000

/-- The monthly establishment charges in rupees. -/
def establishment_charges : ℕ := 75000

/-- The decrease in profit when one loom breaks down for a month, in rupees. -/
def profit_decrease : ℕ := 7000

/-- Theorem stating that the number of looms employed by the firm is 50. -/
theorem firm_employs_50_looms :
  number_of_looms = 50 ∧
  aggregate_sales / number_of_looms - manufacturing_expenses / number_of_looms = profit_decrease :=
sorry

end firm_employs_50_looms_l2888_288817


namespace hexagon_five_layers_dots_l2888_288874

/-- Calculates the number of dots in a hexagonal layer -/
def dots_in_layer (n : ℕ) : ℕ := 6 * n

/-- Calculates the total number of dots up to and including a given layer -/
def total_dots (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | m + 1 => total_dots m + dots_in_layer (m + 1)

theorem hexagon_five_layers_dots :
  total_dots 5 = 61 := by
  sorry

end hexagon_five_layers_dots_l2888_288874


namespace park_tree_density_l2888_288890

/-- Given a rectangular park with length, width, and number of trees, 
    calculate the area occupied by each tree. -/
def area_per_tree (length width num_trees : ℕ) : ℚ :=
  (length * width : ℚ) / num_trees

/-- Theorem stating that in a park of 1000 feet long and 2000 feet wide, 
    with 100,000 trees, each tree occupies 20 square feet. -/
theorem park_tree_density :
  area_per_tree 1000 2000 100000 = 20 := by
  sorry

#eval area_per_tree 1000 2000 100000

end park_tree_density_l2888_288890


namespace min_value_trig_expression_l2888_288889

theorem min_value_trig_expression (α β : ℝ) :
  9 * (Real.cos α)^2 - 10 * Real.cos α * Real.sin β - 8 * Real.cos β * Real.sin α + 17 ≥ 1 := by
  sorry

end min_value_trig_expression_l2888_288889


namespace not_prime_4k4_plus_1_and_k4_plus_4_l2888_288864

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

theorem not_prime_4k4_plus_1_and_k4_plus_4 (k : ℕ) : 
  ¬(is_prime (4 * k^4 + 1)) ∧ ¬(is_prime (k^4 + 4)) := by
  sorry


end not_prime_4k4_plus_1_and_k4_plus_4_l2888_288864


namespace equation_solution_in_interval_l2888_288823

theorem equation_solution_in_interval :
  ∃! x : ℝ, x ∈ (Set.Ioo 0 1) ∧ 3^x + x = 3 := by
  sorry

end equation_solution_in_interval_l2888_288823


namespace equidistant_point_x_coord_l2888_288884

/-- The point (x, y) that is equidistant from the x-axis, y-axis, and the line 2x + 3y = 6 -/
def equidistant_point (x y : ℝ) : Prop :=
  let d_x_axis := |y|
  let d_y_axis := |x|
  let d_line := |2*x + 3*y - 6| / Real.sqrt 13
  d_x_axis = d_y_axis ∧ d_x_axis = d_line

/-- The x-coordinate of the equidistant point is 6/5 -/
theorem equidistant_point_x_coord :
  ∃ y : ℝ, equidistant_point (6/5) y :=
sorry

end equidistant_point_x_coord_l2888_288884


namespace cubic_root_conditions_l2888_288805

theorem cubic_root_conditions (a b c d : ℝ) (ha : a ≠ 0) 
  (h_roots : ∀ z : ℂ, a * z^3 + b * z^2 + c * z + d = 0 → z.re < 0) :
  ab > 0 ∧ bc - ad > 0 ∧ ad > 0 := by
  sorry

end cubic_root_conditions_l2888_288805


namespace kidney_apples_amount_l2888_288803

/-- The amount of golden apples in kg -/
def golden_apples : ℕ := 37

/-- The amount of Canada apples in kg -/
def canada_apples : ℕ := 14

/-- The amount of apples sold in kg -/
def apples_sold : ℕ := 36

/-- The amount of apples left in kg -/
def apples_left : ℕ := 38

/-- The amount of kidney apples in kg -/
def kidney_apples : ℕ := 23

theorem kidney_apples_amount :
  kidney_apples = apples_left + apples_sold - golden_apples - canada_apples :=
by sorry

end kidney_apples_amount_l2888_288803


namespace second_walking_speed_l2888_288856

/-- Proves that the second walking speed is 6 km/h given the problem conditions -/
theorem second_walking_speed (distance : ℝ) (speed1 : ℝ) (miss_time : ℝ) (early_time : ℝ) (v : ℝ) : 
  distance = 13.5 ∧ 
  speed1 = 5 ∧ 
  miss_time = 12 / 60 ∧ 
  early_time = 15 / 60 ∧ 
  distance / speed1 - miss_time = distance / v + early_time → 
  v = 6 := by
  sorry

end second_walking_speed_l2888_288856


namespace rectangle_area_change_l2888_288868

theorem rectangle_area_change (L B : ℝ) (h1 : L > 0) (h2 : B > 0) :
  let new_length := 1.1 * L
  let new_breadth := 0.9 * B
  let original_area := L * B
  let new_area := new_length * new_breadth
  new_area = 0.99 * original_area := by
sorry

end rectangle_area_change_l2888_288868


namespace tangerines_remain_odd_last_fruit_is_tangerine_l2888_288846

/-- Represents the types of fruits in the vase -/
inductive Fruit
| Tangerine
| Apple

/-- Represents the state of the vase -/
structure VaseState where
  tangerines : Nat
  apples : Nat

/-- Represents the action of taking fruits -/
inductive TakeAction
| TwoTangerines
| TangerineAndApple
| TwoApples

/-- Function to update the vase state based on the take action -/
def updateVase (state : VaseState) (action : TakeAction) : VaseState :=
  match action with
  | TakeAction.TwoTangerines => 
      { tangerines := state.tangerines - 2, apples := state.apples + 1 }
  | TakeAction.TangerineAndApple => state
  | TakeAction.TwoApples => 
      { tangerines := state.tangerines, apples := state.apples - 1 }

/-- Theorem stating that the number of tangerines remains odd throughout the process -/
theorem tangerines_remain_odd (initial_tangerines : Nat) 
    (h_initial_odd : Odd initial_tangerines) 
    (actions : List TakeAction) :
    let final_state := actions.foldl updateVase { tangerines := initial_tangerines, apples := 0 }
    Odd final_state.tangerines ∧ final_state.tangerines > 0 := by
  sorry

/-- Theorem stating that the last fruit in the vase is a tangerine -/
theorem last_fruit_is_tangerine (initial_tangerines : Nat) 
    (h_initial_odd : Odd initial_tangerines) 
    (actions : List TakeAction) 
    (h_one_left : (actions.foldl updateVase { tangerines := initial_tangerines, apples := 0 }).tangerines + 
                  (actions.foldl updateVase { tangerines := initial_tangerines, apples := 0 }).apples = 1) :
    (actions.foldl updateVase { tangerines := initial_tangerines, apples := 0 }).tangerines = 1 := by
  sorry

end tangerines_remain_odd_last_fruit_is_tangerine_l2888_288846


namespace selling_price_calculation_l2888_288838

def cost_price : ℕ := 50
def profit_rate : ℕ := 100

theorem selling_price_calculation (cost_price : ℕ) (profit_rate : ℕ) :
  cost_price = 50 → profit_rate = 100 → cost_price + (profit_rate * cost_price) / 100 = 100 := by
  sorry

end selling_price_calculation_l2888_288838


namespace fraction_equality_l2888_288858

theorem fraction_equality (a b : ℝ) (h : 1/a - 1/b = 1) :
  (a + a*b - b) / (a - 2*a*b - b) = 0 := by
sorry

end fraction_equality_l2888_288858


namespace kelly_games_theorem_l2888_288819

/-- The number of Nintendo games Kelly needs to give away to have 12 left -/
def games_to_give_away (initial_games : ℕ) (desired_games : ℕ) : ℕ :=
  initial_games - desired_games

theorem kelly_games_theorem :
  let initial_nintendo_games : ℕ := 20
  let desired_nintendo_games : ℕ := 12
  games_to_give_away initial_nintendo_games desired_nintendo_games = 8 := by
  sorry

end kelly_games_theorem_l2888_288819


namespace man_speed_man_speed_result_l2888_288816

/-- Calculates the speed of a man running opposite to a train --/
theorem man_speed (train_speed : Real) (train_length : Real) (passing_time : Real) : Real :=
  let train_speed_mps := train_speed * 1000 / 3600
  let relative_speed := train_length / passing_time
  let man_speed_mps := relative_speed - train_speed_mps
  let man_speed_kmph := man_speed_mps * 3600 / 1000
  man_speed_kmph

/-- The speed of the man is approximately 6.0024 km/h --/
theorem man_speed_result : 
  ∃ ε > 0, |man_speed 60 110 5.999520038396929 - 6.0024| < ε :=
by
  sorry

end man_speed_man_speed_result_l2888_288816


namespace sqrt_five_squared_l2888_288813

theorem sqrt_five_squared : (Real.sqrt 5)^2 = 5 := by
  sorry

end sqrt_five_squared_l2888_288813


namespace triangle_area_triangle_area_is_54_l2888_288873

/-- A triangle with side lengths 9, 12, and 15 units has an area of 54 square units. -/
theorem triangle_area : ℝ :=
  let a := 9
  let b := 12
  let c := 15
  let s := (a + b + c) / 2  -- semi-perimeter
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))  -- Heron's formula
  54

/-- The theorem statement -/
theorem triangle_area_is_54 : triangle_area = 54 := by sorry

end triangle_area_triangle_area_is_54_l2888_288873


namespace system_solution_l2888_288861

theorem system_solution (x y b : ℝ) : 
  (5 * x + y = b) → 
  (3 * x + 4 * y = 3 * b) → 
  (x = 3) → 
  (b = 60) := by
sorry

end system_solution_l2888_288861


namespace projected_revenue_increase_l2888_288896

theorem projected_revenue_increase 
  (last_year_revenue : ℝ) 
  (h1 : actual_revenue = 0.75 * last_year_revenue) 
  (h2 : actual_revenue = 0.60 * projected_revenue) 
  (projected_revenue := last_year_revenue * (1 + projected_increase / 100)) :
  projected_increase = 25 := by
sorry

end projected_revenue_increase_l2888_288896


namespace functional_equation_solution_l2888_288812

/-- A function satisfying the given functional equation -/
def FunctionalEquation (g : ℝ → ℝ) : Prop :=
  (g 1 = 1) ∧ 
  (∀ x y : ℝ, g (x + y) = 4^y * g x + 3^x * g y)

/-- The main theorem stating that the function g(x) = 4^x - 3^x satisfies the functional equation -/
theorem functional_equation_solution :
  ∃ g : ℝ → ℝ, FunctionalEquation g ∧ (∀ x : ℝ, g x = 4^x - 3^x) :=
sorry

end functional_equation_solution_l2888_288812


namespace quadratic_equation_value_l2888_288844

theorem quadratic_equation_value (x : ℝ) (h : x^2 - 3*x = 4) : 2*x^2 - 6*x - 3 = 5 := by
  sorry

end quadratic_equation_value_l2888_288844


namespace complex_expression_value_l2888_288876

theorem complex_expression_value : 
  let x : ℂ := Complex.exp (Complex.I * (2 * Real.pi / 9))
  (2 * x + x^3) * (2 * x^3 + x^9) * (2 * x^6 + x^18) * 
  (2 * x^2 + x^6) * (2 * x^5 + x^15) * (2 * x^7 + x^21) = 557 := by
  sorry

end complex_expression_value_l2888_288876


namespace richard_patrick_diff_l2888_288843

/-- Bowling game results -/
def bowling_game (patrick_round1 richard_round1_diff : ℕ) : ℕ × ℕ :=
  let richard_round1 := patrick_round1 + richard_round1_diff
  let patrick_round2 := 2 * richard_round1
  let richard_round2 := patrick_round2 - 3
  let patrick_total := patrick_round1 + patrick_round2
  let richard_total := richard_round1 + richard_round2
  (patrick_total, richard_total)

/-- Theorem stating the difference in total pins knocked down -/
theorem richard_patrick_diff (patrick_round1 : ℕ) : 
  (bowling_game patrick_round1 15).2 - (bowling_game patrick_round1 15).1 = 12 :=
by sorry

end richard_patrick_diff_l2888_288843


namespace two_digit_decimal_bounds_l2888_288837

-- Define a two-digit decimal number accurate to the tenth place
def TwoDigitDecimal (x : ℝ) : Prop :=
  10 ≤ x ∧ x < 100 ∧ ∃ (n : ℤ), x = n / 10

-- Define the approximation to the tenth place
def ApproximateToTenth (x y : ℝ) : Prop :=
  ∃ (n : ℤ), y = n / 10 ∧ |x - y| < 0.05

-- Theorem statement
theorem two_digit_decimal_bounds :
  ∀ x : ℝ,
  TwoDigitDecimal x →
  ApproximateToTenth x 15.6 →
  x ≤ 15.64 ∧ x ≥ 15.55 :=
by sorry

end two_digit_decimal_bounds_l2888_288837


namespace sin_alpha_plus_pi_third_l2888_288802

theorem sin_alpha_plus_pi_third (α : Real) (h1 : 0 < α ∧ α < π / 2) 
  (h2 : 7 * Real.sin α = 2 * Real.cos (2 * α)) : 
  Real.sin (α + π / 3) = (1 + 3 * Real.sqrt 5) / 8 := by
  sorry

end sin_alpha_plus_pi_third_l2888_288802


namespace polynomial_remainder_l2888_288847

theorem polynomial_remainder (a b : ℤ) : 
  (∀ x : ℤ, ∃ q : ℤ, x^3 - 2*x^2 + a*x + b = (x-1)*(x-2)*q + (2*x + 1)) → 
  a = 1 ∧ b = 3 := by
sorry

end polynomial_remainder_l2888_288847


namespace perfect_square_equation_l2888_288827

theorem perfect_square_equation (a b c : ℤ) 
  (h : (a - 5)^2 + (b - 12)^2 - (c - 13)^2 = a^2 + b^2 - c^2) :
  ∃ k : ℤ, a^2 + b^2 - c^2 = k^2 := by
  sorry

end perfect_square_equation_l2888_288827


namespace islander_group_composition_l2888_288824

/-- Represents the type of islander: knight or liar -/
inductive IslanderType
| Knight
| Liar

/-- Represents an islander's statement about the group composition -/
inductive Statement
| MoreLiars
| MoreKnights
| Equal

/-- A function that returns the true statement about group composition -/
def trueStatement (knights liars : Nat) : Statement :=
  if knights > liars then Statement.MoreKnights
  else if liars > knights then Statement.MoreLiars
  else Statement.Equal

/-- A function that determines what an islander would say based on their type and the true group composition -/
def islanderStatement (type : IslanderType) (knights liars : Nat) : Statement :=
  match type with
  | IslanderType.Knight => trueStatement knights liars
  | IslanderType.Liar => 
    match trueStatement knights liars with
    | Statement.MoreLiars => Statement.MoreKnights
    | Statement.MoreKnights => Statement.MoreLiars
    | Statement.Equal => Statement.MoreLiars  -- Arbitrarily chosen, could be MoreKnights as well

theorem islander_group_composition 
  (total : Nat) 
  (h_total : total = 10) 
  (knights liars : Nat) 
  (h_sum : knights + liars = total) 
  (h_five_more_liars : ∃ (group : Finset IslanderType), 
    group.card = 5 ∧ 
    ∀ i ∈ group, islanderStatement i knights liars = Statement.MoreLiars) :
  knights = liars ∧ 
  ∃ (other_group : Finset IslanderType), 
    other_group.card = 5 ∧ 
    ∀ i ∈ other_group, islanderStatement i knights liars = Statement.Equal :=
sorry


end islander_group_composition_l2888_288824


namespace smallest_cube_divisible_by_primes_l2888_288806

theorem smallest_cube_divisible_by_primes (p q r : ℕ) : 
  Prime p → Prime q → Prime r → p ≠ q → p ≠ r → q ≠ r → p ≠ 1 → q ≠ 1 → r ≠ 1 →
  (∀ m : ℕ, m > 0 → (p^2 * q^3 * r^4) ∣ m → m = m^3 → m ≥ (p^2 * q^2 * r^2)^3) ∧
  (p^2 * q^3 * r^4) ∣ (p^2 * q^2 * r^2)^3 ∧
  ((p^2 * q^2 * r^2)^3)^(1/3) = p^2 * q^2 * r^2 :=
by sorry

end smallest_cube_divisible_by_primes_l2888_288806


namespace loop_contains_conditional_l2888_288872

/-- Represents a flowchart structure -/
inductive FlowchartStructure
  | Sequential
  | Conditional
  | Loop

/-- Represents the containment relationship between flowchart structures -/
def contains : FlowchartStructure → FlowchartStructure → Prop := sorry

/-- A loop structure must contain a conditional structure -/
theorem loop_contains_conditional :
  ∀ (loop : FlowchartStructure), loop = FlowchartStructure.Loop →
    ∃ (cond : FlowchartStructure), cond = FlowchartStructure.Conditional ∧ contains loop cond :=
  sorry

end loop_contains_conditional_l2888_288872


namespace thirteen_in_binary_l2888_288871

theorem thirteen_in_binary : 
  (13 : ℕ).digits 2 = [1, 0, 1, 1] :=
sorry

end thirteen_in_binary_l2888_288871


namespace magnitude_of_complex_number_l2888_288828

theorem magnitude_of_complex_number (i : ℂ) (h : i^2 = -1) :
  let z : ℂ := (3 + 2*i) * i
  Complex.abs z = Real.sqrt 13 := by sorry

end magnitude_of_complex_number_l2888_288828


namespace least_sum_of_bases_l2888_288897

theorem least_sum_of_bases (c d : ℕ) (h1 : c > 0) (h2 : d > 0) 
  (h3 : 3 * c + 6 = 6 * d + 3) : 
  (∀ x y : ℕ, x > 0 → y > 0 → 3 * x + 6 = 6 * y + 3 → c + d ≤ x + y) ∧ c + d = 5 :=
sorry

end least_sum_of_bases_l2888_288897


namespace evas_numbers_l2888_288814

theorem evas_numbers (a b : ℕ) (h1 : a > b) 
  (h2 : 10 ≤ a + b) (h3 : a + b < 100)
  (h4 : 10 ≤ a - b) (h5 : a - b < 100)
  (h6 : (a + b) * (a - b) = 645) : 
  a = 29 ∧ b = 14 := by
sorry

end evas_numbers_l2888_288814


namespace quadrilateral_impossibility_l2888_288832

theorem quadrilateral_impossibility : ¬ ∃ (a b c d : ℝ),
  (2 * a^2 - 18 * a + 36 = 0 ∨ a^2 - 20 * a + 75 = 0) ∧
  (2 * b^2 - 18 * b + 36 = 0 ∨ b^2 - 20 * b + 75 = 0) ∧
  (2 * c^2 - 18 * c + 36 = 0 ∨ c^2 - 20 * c + 75 = 0) ∧
  (2 * d^2 - 18 * d + 36 = 0 ∨ d^2 - 20 * d + 75 = 0) ∧
  (a + b + c > d) ∧ (a + b + d > c) ∧ (a + c + d > b) ∧ (b + c + d > a) ∧
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 :=
sorry

end quadrilateral_impossibility_l2888_288832


namespace average_of_combined_results_l2888_288804

theorem average_of_combined_results :
  let n₁ : ℕ := 40
  let avg₁ : ℚ := 30
  let n₂ : ℕ := 30
  let avg₂ : ℚ := 40
  let total_sum : ℚ := n₁ * avg₁ + n₂ * avg₂
  let total_count : ℕ := n₁ + n₂
  (total_sum / total_count : ℚ) = 2400 / 70 :=
by sorry

end average_of_combined_results_l2888_288804


namespace right_triangular_pyramid_area_relation_l2888_288899

/-- Represents a triangular pyramid with right angles at each vertex -/
structure RightTriangularPyramid where
  /-- The length of the first base edge -/
  a : ℝ
  /-- The length of the second base edge -/
  b : ℝ
  /-- The length of the third base edge -/
  c : ℝ
  /-- The length of the first edge from apex to base -/
  e₁ : ℝ
  /-- The length of the second edge from apex to base -/
  e₂ : ℝ
  /-- The length of the third edge from apex to base -/
  e₃ : ℝ
  /-- The area of the first lateral face -/
  t₁ : ℝ
  /-- The area of the second lateral face -/
  t₂ : ℝ
  /-- The area of the third lateral face -/
  t₃ : ℝ
  /-- The area of the base -/
  T : ℝ
  /-- Condition: right angles at vertices -/
  right_angles : a^2 = e₁^2 + e₂^2 ∧ b^2 = e₂^2 + e₃^2 ∧ c^2 = e₃^2 + e₁^2
  /-- Condition: lateral face areas -/
  lateral_areas : t₁ = (1/2) * e₁ * e₂ ∧ t₂ = (1/2) * e₂ * e₃ ∧ t₃ = (1/2) * e₃ * e₁
  /-- Condition: base area -/
  base_area : T = (1/4) * Real.sqrt ((a+b+c)*(a+b-c)*(a-b+c)*(b+c-a))

/-- The square of the base area is equal to the sum of the squares of the lateral face areas -/
theorem right_triangular_pyramid_area_relation (p : RightTriangularPyramid) :
  p.T^2 = p.t₁^2 + p.t₂^2 + p.t₃^2 := by
  sorry

end right_triangular_pyramid_area_relation_l2888_288899


namespace function_inequality_existence_l2888_288862

theorem function_inequality_existence (f : ℝ → ℝ) 
  (hf : ∀ x, 0 < x → 0 < f x) : 
  ¬(∀ x y, 0 < x ∧ 0 < y → f (x + y) ≥ f x + y * f (f x)) := by
  sorry

end function_inequality_existence_l2888_288862


namespace least_positive_integer_satisfying_congruences_l2888_288834

theorem least_positive_integer_satisfying_congruences : ∃ n : ℕ, 
  n > 0 ∧
  n % 3 = 1 ∧
  n % 5 = 3 ∧
  n % 6 = 5 ∧
  n % 7 = 2 ∧
  (∀ m : ℕ, m > 0 ∧ m % 3 = 1 ∧ m % 5 = 3 ∧ m % 6 = 5 ∧ m % 7 = 2 → m ≥ n) ∧
  n = 83 :=
by sorry

end least_positive_integer_satisfying_congruences_l2888_288834


namespace train_crossing_time_l2888_288893

/-- Proves that a train with given length and speed takes a specific time to cross an electric pole. -/
theorem train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) : 
  train_length = 320 →
  train_speed_kmh = 144 →
  crossing_time = train_length / (train_speed_kmh * 1000 / 3600) →
  crossing_time = 8 := by
  sorry

#check train_crossing_time

end train_crossing_time_l2888_288893


namespace exists_valid_pairs_l2888_288860

def digits_ge_6 (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d ≥ 6

def a (k : ℕ) : ℕ :=
  (10^k - 3) * 10^2 + 97

theorem exists_valid_pairs :
  ∃ n : ℕ, ∀ k ≥ n, 
    digits_ge_6 (a k) ∧ 
    digits_ge_6 7 ∧ 
    digits_ge_6 (a k * 7) :=
sorry

end exists_valid_pairs_l2888_288860


namespace men_to_women_ratio_l2888_288859

/-- Proves that the ratio of men to women workers is 1:3 given the problem conditions -/
theorem men_to_women_ratio (woman_wage : ℝ) (num_women : ℕ) : 
  let man_wage := 2 * woman_wage
  let women_earnings := num_women * woman_wage * 30
  let men_earnings := (num_women / 3) * man_wage * 20
  women_earnings = 21600 → men_earnings = 14400 := by
  sorry

end men_to_women_ratio_l2888_288859
