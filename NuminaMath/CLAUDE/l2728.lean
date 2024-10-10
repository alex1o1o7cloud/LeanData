import Mathlib

namespace cherry_revenue_is_180_l2728_272801

/-- Calculates the revenue from cherry pies given the total number of pies,
    the ratio of pie types, and the price of a cherry pie. -/
def cherry_pie_revenue (total_pies : ℕ) (apple_ratio blueberry_ratio cherry_ratio : ℕ) (cherry_price : ℕ) : ℕ :=
  let total_ratio := apple_ratio + blueberry_ratio + cherry_ratio
  let cherry_pies := (total_pies * cherry_ratio) / total_ratio
  cherry_pies * cherry_price

/-- Theorem stating that given 36 total pies with a ratio of 3:2:5 for apple:blueberry:cherry pies,
    and a price of $10 per cherry pie, the total revenue from cherry pies is $180. -/
theorem cherry_revenue_is_180 :
  cherry_pie_revenue 36 3 2 5 10 = 180 := by
  sorry

end cherry_revenue_is_180_l2728_272801


namespace smallest_n_value_l2728_272861

/-- The number of ordered quadruplets (a, b, c, d) satisfying the conditions -/
def num_quadruplets : ℕ := 60000

/-- The greatest common divisor of the quadruplets -/
def gcd_value : ℕ := 60

/-- The function that counts the number of ordered quadruplets (a, b, c, d) 
    such that gcd(a, b, c, d) = gcd_value and lcm(a, b, c, d) = n -/
def count_quadruplets (n : ℕ) : ℕ :=
  sorry

/-- The theorem stating that 6480 is the smallest value of n 
    satisfying the given conditions -/
theorem smallest_n_value : 
  (∃ n : ℕ, count_quadruplets n = num_quadruplets) →
  (∀ m : ℕ, count_quadruplets m = num_quadruplets → m ≥ 6480) ∧
  (count_quadruplets 6480 = num_quadruplets) :=
sorry

end smallest_n_value_l2728_272861


namespace rectangular_prism_volume_l2728_272897

theorem rectangular_prism_volume 
  (side_area front_area bottom_area : ℝ)
  (h_side : side_area = 15)
  (h_front : front_area = 10)
  (h_bottom : bottom_area = 6) :
  ∃ (a b c : ℝ), 
    a * b = side_area ∧ 
    b * c = front_area ∧ 
    c * a = bottom_area ∧ 
    a * b * c = 30 := by
  sorry

end rectangular_prism_volume_l2728_272897


namespace final_a_is_three_l2728_272851

/-- Given initial values of a and b, compute the final value of a after the operation a = a + b -/
def compute_final_a (initial_a : ℕ) (initial_b : ℕ) : ℕ :=
  initial_a + initial_b

/-- Theorem stating that given the initial conditions a = 1 and b = 2, 
    after the operation a = a + b, the final value of a is 3 -/
theorem final_a_is_three : compute_final_a 1 2 = 3 := by
  sorry

end final_a_is_three_l2728_272851


namespace complete_square_form_l2728_272830

theorem complete_square_form (x : ℝ) : 
  (∃ a b : ℝ, (-x + 1) * (1 - x) = (a + b)^2) ∧ 
  (¬∃ a b : ℝ, (1 + x) * (1 - x) = (a + b)^2) ∧ 
  (¬∃ a b : ℝ, (-x - 1) * (-1 + x) = (a + b)^2) ∧ 
  (¬∃ a b : ℝ, (x - 1) * (1 + x) = (a + b)^2) :=
by sorry

end complete_square_form_l2728_272830


namespace clara_stickers_l2728_272819

def stickers_left (initial : ℕ) (given_to_boy : ℕ) : ℕ := 
  (initial - given_to_boy) / 2

theorem clara_stickers : stickers_left 100 10 = 45 := by
  sorry

end clara_stickers_l2728_272819


namespace number_problem_l2728_272872

theorem number_problem (x : ℝ) : (1 / 5 * x - 5 = 5) → x = 50 := by
  sorry

end number_problem_l2728_272872


namespace no_valid_flippy_numbers_l2728_272839

/-- A five-digit flippy number is a number of the form ababa or babab where a and b are distinct digits -/
def is_flippy_number (n : ℕ) : Prop :=
  ∃ a b : ℕ, a ≠ b ∧ a < 10 ∧ b < 10 ∧
  ((n = a * 10000 + b * 1000 + a * 100 + b * 10 + a) ∨
   (n = b * 10000 + a * 1000 + b * 100 + a * 10 + b))

/-- The sum of digits of a five-digit flippy number -/
def sum_of_digits (n : ℕ) : ℕ :=
  (n / 10000) + ((n / 1000) % 10) + ((n / 100) % 10) + ((n / 10) % 10) + (n % 10)

/-- There are no five-digit flippy numbers that are divisible by 11 and have a sum of digits divisible by 6 -/
theorem no_valid_flippy_numbers :
  ¬ ∃ n : ℕ, is_flippy_number n ∧ n % 11 = 0 ∧ (sum_of_digits n) % 6 = 0 := by
  sorry

#check no_valid_flippy_numbers

end no_valid_flippy_numbers_l2728_272839


namespace mean_temperature_is_88_point_2_l2728_272841

def temperatures : List ℝ := [78, 80, 82, 85, 88, 90, 92, 95, 97, 95]

theorem mean_temperature_is_88_point_2 :
  (temperatures.sum / temperatures.length : ℝ) = 88.2 := by
  sorry

end mean_temperature_is_88_point_2_l2728_272841


namespace quadratic_polynomial_proof_l2728_272813

theorem quadratic_polynomial_proof :
  ∃ (q : ℝ → ℝ),
    (∀ x, q x = (1/3) * (2*x^2 - 4*x + 9)) ∧
    q (-2) = 8 ∧
    q 1 = 2 ∧
    q 3 = 10 :=
by
  sorry

end quadratic_polynomial_proof_l2728_272813


namespace hundred_chicken_equations_l2728_272898

def hundred_chicken_problem (x y : ℝ) : Prop :=
  (x + y + 81 = 100) ∧ (5*x + 3*y + (1/3) * 81 = 100)

theorem hundred_chicken_equations :
  ∀ x y : ℝ,
  (x ≥ 0) → (y ≥ 0) →
  (x + y + 81 = 100) →
  (5*x + 3*y + 27 = 100) →
  hundred_chicken_problem x y :=
by
  sorry

end hundred_chicken_equations_l2728_272898


namespace subtraction_result_l2728_272837

theorem subtraction_result : 2014 - 4102 = -2088 := by sorry

end subtraction_result_l2728_272837


namespace williams_tickets_l2728_272860

/-- William's ticket problem -/
theorem williams_tickets : 
  ∀ (initial_tickets additional_tickets : ℕ),
  initial_tickets = 15 → 
  additional_tickets = 3 → 
  initial_tickets + additional_tickets = 18 := by
sorry

end williams_tickets_l2728_272860


namespace race_participants_l2728_272858

/-- Represents a bicycle race with participants. -/
structure BicycleRace where
  participants : ℕ
  petya_position : ℕ
  vasya_position : ℕ
  vasya_position_from_end : ℕ

/-- The bicycle race satisfies the given conditions. -/
def valid_race (race : BicycleRace) : Prop :=
  race.petya_position = 10 ∧
  race.vasya_position = race.petya_position - 1 ∧
  race.vasya_position_from_end = 15

theorem race_participants (race : BicycleRace) :
  valid_race race → race.participants = 23 := by
  sorry

#check race_participants

end race_participants_l2728_272858


namespace M_intersect_N_eq_M_l2728_272859

-- Define the sets M and N
def M : Set ℝ := {y | ∃ x, y = 2^x}
def N : Set ℝ := {y | ∃ x, y = x^2}

-- State the theorem
theorem M_intersect_N_eq_M : M ∩ N = M := by
  sorry

end M_intersect_N_eq_M_l2728_272859


namespace least_common_denominator_l2728_272883

theorem least_common_denominator : Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 9)))))) = 2520 := by
  sorry

end least_common_denominator_l2728_272883


namespace fixed_point_on_parabola_l2728_272853

theorem fixed_point_on_parabola (a b c : ℝ) 
  (h1 : |a| ≥ |b - c|) 
  (h2 : |b| ≥ |a + c|) 
  (h3 : |c| ≥ |a - b|) : 
  a * (-1)^2 + b * (-1) + c = 0 := by
sorry

end fixed_point_on_parabola_l2728_272853


namespace x_equals_five_l2728_272886

/-- A composite rectangular figure with specific segment lengths -/
structure CompositeRectangle where
  top_left : ℝ
  top_middle : ℝ
  top_right : ℝ
  bottom_left : ℝ
  bottom_middle : ℝ
  bottom_right : ℝ

/-- The theorem stating that X equals 5 in the given composite rectangle -/
theorem x_equals_five (r : CompositeRectangle) 
  (h1 : r.top_left = 3)
  (h2 : r.top_right = 4)
  (h3 : r.bottom_left = 5)
  (h4 : r.bottom_middle = 7)
  (h5 : r.top_middle = r.bottom_right)
  (h6 : r.top_left + r.top_middle + r.top_right = r.bottom_left + r.bottom_middle + r.bottom_right) :
  r.top_middle = 5 := by
  sorry

end x_equals_five_l2728_272886


namespace sausages_theorem_l2728_272826

def sausages_left (initial : ℕ) : ℕ :=
  let after_monday := initial - (2 * initial / 5)
  let after_tuesday := after_monday - (after_monday / 2)
  let after_wednesday := after_tuesday - (after_tuesday / 4)
  let after_thursday := after_wednesday - (after_wednesday / 3)
  let after_sharing := after_thursday - (after_thursday / 5)
  after_sharing - ((3 * after_sharing) / 5)

theorem sausages_theorem :
  sausages_left 1200 = 58 := by
  sorry

end sausages_theorem_l2728_272826


namespace youngest_child_age_problem_l2728_272815

/-- The age of the youngest child given the conditions of the problem -/
def youngest_child_age (n : ℕ) (interval : ℕ) (total_age : ℕ) : ℕ :=
  (total_age - (n - 1) * n * interval / 2) / n

/-- Theorem stating the age of the youngest child under the given conditions -/
theorem youngest_child_age_problem :
  youngest_child_age 5 2 50 = 6 := by sorry

end youngest_child_age_problem_l2728_272815


namespace quadratic_inequality_solution_rational_inequality_solution_l2728_272809

-- Part 1
theorem quadratic_inequality_solution (b c : ℝ) :
  (∀ x, 5 * x^2 - b * x + c < 0 ↔ -1 < x ∧ x < 3) →
  b + c = -5 :=
sorry

-- Part 2
theorem rational_inequality_solution :
  {x : ℝ | (2 * x - 5) / (x + 4) ≥ 0} = {x : ℝ | x ≥ 5/2 ∨ x < -4} :=
sorry

end quadratic_inequality_solution_rational_inequality_solution_l2728_272809


namespace expression_simplification_l2728_272878

theorem expression_simplification (x : ℝ) (h1 : x ≠ -1) (h2 : x ≠ 2) (h3 : x ≠ -2) :
  (x - 1 - 3 / (x + 1)) / ((x^2 - 4) / (x^2 + 2*x + 1)) = x + 1 :=
by sorry

end expression_simplification_l2728_272878


namespace f_composition_equals_14_l2728_272827

-- Define the functions f and g
def f (x : ℝ) : ℝ := 3 * x - 4
def g (x : ℝ) : ℝ := x + 2

-- State the theorem
theorem f_composition_equals_14 : f (1 + g 3) = 14 := by
  sorry

end f_composition_equals_14_l2728_272827


namespace sally_carl_owe_amount_l2728_272816

def total_promised : ℝ := 400
def amount_received : ℝ := 285
def amy_owes : ℝ := 30

theorem sally_carl_owe_amount :
  ∃ (s : ℝ), 
    s > 0 ∧
    2 * s + amy_owes + amy_owes / 2 = total_promised - amount_received ∧
    s = 35 := by sorry

end sally_carl_owe_amount_l2728_272816


namespace five_balls_three_boxes_l2728_272892

/-- The number of ways to put n distinguishable balls into k distinguishable boxes -/
def balls_in_boxes (n : ℕ) (k : ℕ) : ℕ := k^n

/-- Theorem: There are 243 ways to put 5 distinguishable balls into 3 distinguishable boxes -/
theorem five_balls_three_boxes : balls_in_boxes 5 3 = 243 := by
  sorry

end five_balls_three_boxes_l2728_272892


namespace cos_two_sum_l2728_272881

theorem cos_two_sum (α β : Real) 
  (h1 : Real.sin α + Real.sin β = 1) 
  (h2 : Real.cos α + Real.cos β = 0) : 
  Real.cos (2 * α) + Real.cos (2 * β) = 1 := by
  sorry

end cos_two_sum_l2728_272881


namespace norma_cards_l2728_272829

theorem norma_cards (initial_cards lost_cards remaining_cards : ℕ) :
  lost_cards = 70 →
  remaining_cards = 18 →
  initial_cards = lost_cards + remaining_cards →
  initial_cards = 88 := by
sorry

end norma_cards_l2728_272829


namespace distance_to_tangent_point_l2728_272856

/-- Two externally tangent circles with a common external tangent -/
structure TangentCircles where
  /-- Radius of the larger circle -/
  r₁ : ℝ
  /-- Radius of the smaller circle -/
  r₂ : ℝ
  /-- The circles are externally tangent -/
  tangent : r₁ > 0 ∧ r₂ > 0
  /-- The common external tangent exists -/
  common_tangent_exists : True

/-- The distance from the center of the larger circle to the point where 
    the common external tangent touches the smaller circle -/
theorem distance_to_tangent_point (c : TangentCircles) (h₁ : c.r₁ = 10) (h₂ : c.r₂ = 5) :
  ∃ d : ℝ, d = 10 * Real.sqrt 3 := by
  sorry

end distance_to_tangent_point_l2728_272856


namespace probability_distance_sqrt2_over_2_l2728_272852

/-- A point on a unit square, either a vertex or the center -/
inductive SquarePoint
  | vertex : Fin 4 → SquarePoint
  | center : SquarePoint

/-- The distance between two points on a unit square -/
def distance (p q : SquarePoint) : ℝ :=
  sorry

/-- The set of all possible pairs of points -/
def allPairs : Finset (SquarePoint × SquarePoint) :=
  sorry

/-- The set of pairs of points with distance √2/2 -/
def pairsWithDistance : Finset (SquarePoint × SquarePoint) :=
  sorry

theorem probability_distance_sqrt2_over_2 :
  (Finset.card pairsWithDistance : ℚ) / (Finset.card allPairs : ℚ) = 2 / 5 :=
sorry

end probability_distance_sqrt2_over_2_l2728_272852


namespace no_perfect_square_in_range_l2728_272823

theorem no_perfect_square_in_range : 
  ¬ ∃ (n : ℤ), 5 ≤ n ∧ n ≤ 15 ∧ ∃ (m : ℤ), 2 * n^2 + n + 2 = m^2 := by
  sorry

end no_perfect_square_in_range_l2728_272823


namespace factorization_equality_l2728_272884

theorem factorization_equality (a b : ℝ) :
  a * b^2 - 2 * a^2 * b + a^2 = a * (b - a)^2 := by
  sorry

end factorization_equality_l2728_272884


namespace tangent_sum_simplification_l2728_272843

theorem tangent_sum_simplification :
  (Real.tan (10 * π / 180) + Real.tan (30 * π / 180) + 
   Real.tan (50 * π / 180) + Real.tan (70 * π / 180)) / 
  Real.cos (30 * π / 180) = 
  4 * Real.sin (40 * π / 180) + 1 := by
  sorry

end tangent_sum_simplification_l2728_272843


namespace tangent_line_at_zero_l2728_272850

/-- The curve function -/
def f (a b x : ℝ) : ℝ := x^2 + a*x + b

/-- The derivative of the curve function -/
def f' (a : ℝ) (x : ℝ) : ℝ := 2*x + a

/-- The tangent line function -/
def tangent_line (x y : ℝ) : ℝ := x - y + 1

theorem tangent_line_at_zero (a b : ℝ) :
  (∀ x y, y = f a b x → tangent_line x y = 0 → x = 0 ∧ y = b) →
  (f' a 0 = -1) →
  a = -1 ∧ b = 1 := by sorry

end tangent_line_at_zero_l2728_272850


namespace compaction_percentage_is_twenty_l2728_272804

/-- Represents the compaction problem with cans -/
structure CanCompaction where
  num_cans : ℕ
  space_before : ℕ
  total_space_after : ℕ

/-- Calculates the percentage of original space each can takes up after compaction -/
def compaction_percentage (c : CanCompaction) : ℚ :=
  (c.total_space_after : ℚ) / ((c.num_cans * c.space_before) : ℚ) * 100

/-- Theorem stating that for the given conditions, the compaction percentage is 20% -/
theorem compaction_percentage_is_twenty (c : CanCompaction) 
  (h1 : c.num_cans = 60)
  (h2 : c.space_before = 30)
  (h3 : c.total_space_after = 360) : 
  compaction_percentage c = 20 := by
  sorry

end compaction_percentage_is_twenty_l2728_272804


namespace inequality_solution_set_l2728_272808

theorem inequality_solution_set (x : ℝ) :
  (3 - 2*x - x^2 ≤ 0) ↔ (x ≤ -3 ∨ x ≥ 1) :=
by sorry

end inequality_solution_set_l2728_272808


namespace cubic_root_interval_l2728_272874

theorem cubic_root_interval (a b : ℤ) : 
  (∃ x : ℝ, x^3 - x + 1 = 0 ∧ a < x ∧ x < b) →
  b - a = 1 →
  a + b = -3 := by
sorry

end cubic_root_interval_l2728_272874


namespace inequality_solution_set_l2728_272862

/-- The solution set of the inequality (a^2-1)x^2-(a-1)x-1 < 0 is ℝ if and only if -3/5 < a ≤ 1 -/
theorem inequality_solution_set (a : ℝ) : 
  (∀ x : ℝ, (a^2 - 1) * x^2 - (a - 1) * x - 1 < 0) ↔ -3/5 < a ∧ a ≤ 1 :=
sorry

end inequality_solution_set_l2728_272862


namespace trip_distance_l2728_272840

theorem trip_distance (total : ℝ) 
  (h1 : total > 0)
  (h2 : total / 4 + 30 + total / 10 + (total - (total / 4 + 30 + total / 10)) = total) : 
  total = 60 := by
sorry

end trip_distance_l2728_272840


namespace sin_cos_range_l2728_272842

theorem sin_cos_range (x : ℝ) : 
  -1 ≤ Real.sin x + Real.cos x + Real.sin x * Real.cos x ∧ 
  Real.sin x + Real.cos x + Real.sin x * Real.cos x ≤ 1/2 + Real.sqrt 2 := by
  sorry

end sin_cos_range_l2728_272842


namespace root_condition_implies_m_range_l2728_272846

theorem root_condition_implies_m_range :
  ∀ m : ℝ,
  (∀ x : ℝ, x^2 - (3*m + 2)*x + 2*(m + 6) = 0 → x > 3) →
  m ≥ 2 ∧ m < 15/7 := by
sorry

end root_condition_implies_m_range_l2728_272846


namespace one_basket_total_peaches_l2728_272877

/-- Given a basket of peaches with red and green peaches, calculate the total number of peaches -/
def total_peaches (red_peaches green_peaches : ℕ) : ℕ :=
  red_peaches + green_peaches

/-- Theorem: The total number of peaches in 1 basket is 7 -/
theorem one_basket_total_peaches :
  total_peaches 4 3 = 7 := by
  sorry

end one_basket_total_peaches_l2728_272877


namespace equation_solution_l2728_272885

theorem equation_solution (n : ℝ) (h : n = 3) :
  n^4 - 20*n + 1 = 22 :=
by sorry

end equation_solution_l2728_272885


namespace lcm_gcd_sum_reciprocal_sum_l2728_272863

theorem lcm_gcd_sum_reciprocal_sum (m n : ℕ+) 
  (h_lcm : Nat.lcm m n = 210)
  (h_gcd : Nat.gcd m n = 6)
  (h_sum : m + n = 72) :
  1 / (m : ℚ) + 1 / (n : ℚ) = 2 / 35 := by
  sorry

end lcm_gcd_sum_reciprocal_sum_l2728_272863


namespace min_product_xyz_l2728_272802

theorem min_product_xyz (x y z : ℝ) : 
  x > 0 → y > 0 → z > 0 →
  x + y + z = 1 →
  x ≤ 3*y ∧ x ≤ 3*z ∧ y ≤ 3*x ∧ y ≤ 3*z ∧ z ≤ 3*x ∧ z ≤ 3*y →
  x * y * z ≥ 3/125 := by
  sorry

end min_product_xyz_l2728_272802


namespace total_cost_is_51_l2728_272810

/-- The cost of a single shirt in dollars -/
def shirt_cost : ℕ := 5

/-- The cost of a single hat in dollars -/
def hat_cost : ℕ := 4

/-- The cost of a single pair of jeans in dollars -/
def jeans_cost : ℕ := 10

/-- The number of shirts to be purchased -/
def num_shirts : ℕ := 3

/-- The number of hats to be purchased -/
def num_hats : ℕ := 4

/-- The number of pairs of jeans to be purchased -/
def num_jeans : ℕ := 2

/-- Theorem stating that the total cost of the purchase is $51 -/
theorem total_cost_is_51 : 
  num_shirts * shirt_cost + num_hats * hat_cost + num_jeans * jeans_cost = 51 := by
  sorry

end total_cost_is_51_l2728_272810


namespace complex_magnitude_problem_l2728_272873

theorem complex_magnitude_problem (x y : ℝ) :
  (Complex.I + 1) * x = Complex.I * y + 1 →
  Complex.abs (x + Complex.I * y) = Real.sqrt 2 := by
sorry

end complex_magnitude_problem_l2728_272873


namespace rectangle_circle_chord_length_l2728_272870

theorem rectangle_circle_chord_length :
  ∀ (rectangle : Set (ℝ × ℝ)) (circle : Set (ℝ × ℝ)) (P Q : ℝ × ℝ),
    -- Rectangle properties
    (∀ (x y : ℝ), (x, y) ∈ rectangle ↔ 0 ≤ x ∧ x ≤ 4 ∧ 0 ≤ y ∧ y ≤ 2) →
    -- Circle properties
    (∃ (cx cy : ℝ), ∀ (x y : ℝ), (x, y) ∈ circle ↔ (x - cx)^2 + (y - cy)^2 = 1) →
    -- Circle touches three sides of the rectangle
    (∃ (x : ℝ), (x, 0) ∈ circle ∧ 0 < x ∧ x < 4) →
    (∃ (y : ℝ), (0, y) ∈ circle ∧ 0 < y ∧ y < 2) →
    (∃ (x : ℝ), (x, 2) ∈ circle ∧ 0 < x ∧ x < 4) →
    -- P and Q are on the circle and the diagonal
    P ∈ circle → Q ∈ circle →
    (∃ (t : ℝ), 0 < t ∧ t < 1 ∧ P = (4*t, 2*t) ∧ Q = (4*(1-t), 2*(1-t))) →
    -- Conclusion: length of PQ is 4/√5
    Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) = 4 / Real.sqrt 5 :=
by sorry

end rectangle_circle_chord_length_l2728_272870


namespace solution_to_linear_equation_l2728_272806

theorem solution_to_linear_equation :
  ∃ (x y : ℝ), x + 2 * y = 6 ∧ x = 2 ∧ y = 2 := by
sorry

end solution_to_linear_equation_l2728_272806


namespace perpendicular_lines_m_line_l_equation_l2728_272896

-- Define the lines l₁ and l₂
def l₁ (m : ℝ) (x y : ℝ) : Prop := (m + 2) * x + m * y - 6 = 0
def l₂ (m : ℝ) (x y : ℝ) : Prop := m * x + y - 3 = 0

-- Define perpendicularity of lines
def perpendicular (m : ℝ) : Prop := 
  (m = 0) ∨ (m ≠ 0 ∧ (m + 2) / m * m = -1)

-- Define the point P
def P (m : ℝ) : ℝ × ℝ := (1, 2 * m)

-- Define the line l
def l (k : ℝ) (x y : ℝ) : Prop := y - 2 = k * (x - 1)

-- Define the intercept condition
def intercept_condition (k : ℝ) : Prop :=
  (k - 2) / k = 2 * (2 - k)

theorem perpendicular_lines_m (m : ℝ) : 
  perpendicular m → m = -3 ∨ m = 0 :=
sorry

theorem line_l_equation (m : ℝ) :
  l₂ m 1 (2 * m) →
  (∃ k, l k 1 (2 * m) ∧ intercept_condition k) →
  (∀ x y, l 2 x y ∨ l (-1/2) x y) :=
sorry

end perpendicular_lines_m_line_l_equation_l2728_272896


namespace checkerboard_square_count_l2728_272879

/-- Represents a square on the checkerboard -/
structure Square where
  size : Nat
  topLeftRow : Nat
  topLeftCol : Nat

/-- The size of the checkerboard -/
def boardSize : Nat := 10

/-- Checks if a square contains at least 5 black squares -/
def containsAtLeast5Black (s : Square) : Bool :=
  sorry

/-- Counts the number of valid squares of a given size -/
def countValidSquares (size : Nat) : Nat :=
  sorry

/-- Counts the total number of squares containing at least 5 black squares -/
def totalValidSquares : Nat :=
  sorry

/-- Main theorem: The number of distinct squares containing at least 5 black squares is 172 -/
theorem checkerboard_square_count : totalValidSquares = 172 := by
  sorry

end checkerboard_square_count_l2728_272879


namespace test_total_points_l2728_272893

theorem test_total_points : 
  ∀ (total_problems : ℕ) 
    (three_point_problems : ℕ) 
    (four_point_problems : ℕ),
  total_problems = 30 →
  four_point_problems = 10 →
  three_point_problems + four_point_problems = total_problems →
  3 * three_point_problems + 4 * four_point_problems = 100 :=
by
  sorry

end test_total_points_l2728_272893


namespace shopping_money_theorem_l2728_272865

theorem shopping_money_theorem (initial_money : ℚ) : 
  (initial_money - 3/7 * initial_money - 2/5 * initial_money - 1/4 * initial_money = 24) →
  (initial_money - 1/2 * initial_money - 1/3 * initial_money = 36) →
  (initial_money + initial_money) / 2 = 458.18 := by
sorry

end shopping_money_theorem_l2728_272865


namespace qinJiushao_V₁_for_f_10_l2728_272812

/-- Qin Jiushao's algorithm for polynomial evaluation -/
def qinJiushao (f : ℝ → ℝ) (x : ℝ) : ℝ := sorry

/-- The polynomial f(x) = 3x⁴ + 2x² + x + 4 -/
def f (x : ℝ) : ℝ := 3 * x^4 + 2 * x^2 + x + 4

/-- V₁ in Qin Jiushao's algorithm for f(10) -/
def V₁ : ℝ := 3 * 10 + 2

theorem qinJiushao_V₁_for_f_10 : 
  V₁ = 32 := by sorry

end qinJiushao_V₁_for_f_10_l2728_272812


namespace complex_equation_solution_l2728_272831

theorem complex_equation_solution (z : ℂ) (h : (1 + 3*Complex.I)*z = Complex.I - 3) : z = Complex.I := by
  sorry

end complex_equation_solution_l2728_272831


namespace marbles_problem_l2728_272835

theorem marbles_problem (total_marbles : ℕ) (marbles_per_boy : ℕ) (h1 : total_marbles = 99) (h2 : marbles_per_boy = 9) :
  total_marbles / marbles_per_boy = 11 :=
by sorry

end marbles_problem_l2728_272835


namespace eggs_left_over_l2728_272845

def total_eggs : ℕ := 114
def carton_size : ℕ := 15

theorem eggs_left_over : total_eggs % carton_size = 9 := by
  sorry

end eggs_left_over_l2728_272845


namespace ab_value_l2728_272833

theorem ab_value (a b : ℝ) (h : b = Real.sqrt (1 - 2*a) + Real.sqrt (2*a - 1) + 3) : 
  a^b = (1/8 : ℝ) := by
sorry

end ab_value_l2728_272833


namespace square_sum_identity_l2728_272847

theorem square_sum_identity (x : ℝ) : (x + 2)^2 + 2*(x + 2)*(5 - x) + (5 - x)^2 = 49 := by
  sorry

end square_sum_identity_l2728_272847


namespace area_triangle_EYH_l2728_272848

/-- Represents a trapezoid with bases and diagonals -/
structure Trapezoid where
  base1 : ℝ
  base2 : ℝ
  area : ℝ

/-- Theorem: Area of triangle EYH in trapezoid EFGH -/
theorem area_triangle_EYH (EFGH : Trapezoid) (h1 : EFGH.base1 = 15) (h2 : EFGH.base2 = 35) (h3 : EFGH.area = 400) :
  ∃ (area_EYH : ℝ), area_EYH = 84 := by
  sorry

end area_triangle_EYH_l2728_272848


namespace correct_bullseyes_needed_l2728_272875

/-- Represents the archery contest scenario -/
structure ArcheryContest where
  totalShots : Nat
  shotsCompleted : Nat
  pointsAhead : Nat
  minPointsPerShot : Nat

/-- Calculates the minimum number of bullseyes needed to secure victory -/
def minBullseyesNeeded (contest : ArcheryContest) : Nat :=
  sorry

/-- Theorem stating the correct number of bullseyes needed -/
theorem correct_bullseyes_needed (contest : ArcheryContest) 
  (h1 : contest.totalShots = 150)
  (h2 : contest.shotsCompleted = 75)
  (h3 : contest.pointsAhead = 70)
  (h4 : contest.minPointsPerShot = 2) :
  minBullseyesNeeded contest = 67 := by
  sorry

end correct_bullseyes_needed_l2728_272875


namespace alan_market_cost_l2728_272888

/-- Calculates the total cost of Alan's market purchase including discount and tax --/
def market_cost (egg_price : ℝ) (egg_quantity : ℕ) 
                (chicken_price : ℝ) (chicken_quantity : ℕ) 
                (milk_price : ℝ) (milk_quantity : ℕ) 
                (bread_price : ℝ) (bread_quantity : ℕ) 
                (chicken_discount : ℕ → ℕ) (tax_rate : ℝ) : ℝ :=
  let egg_cost := egg_price * egg_quantity
  let chicken_cost := chicken_price * (chicken_quantity - chicken_discount chicken_quantity)
  let milk_cost := milk_price * milk_quantity
  let bread_cost := bread_price * bread_quantity
  let subtotal := egg_cost + chicken_cost + milk_cost + bread_cost
  let tax := subtotal * tax_rate
  subtotal + tax

/-- Theorem stating that Alan's market cost is $103.95 --/
theorem alan_market_cost : 
  market_cost 2 20 8 6 4 3 3.5 2 (fun n => n / 4) 0.05 = 103.95 := by
  sorry

end alan_market_cost_l2728_272888


namespace det_equals_nine_l2728_272880

-- Define the determinant for a 2x2 matrix
def det2x2 (a b c d : ℝ) : ℝ := a * d - b * c

-- State the theorem
theorem det_equals_nine (x : ℝ) (h : x^2 - 2*x - 5 = 0) : 
  det2x2 (x + 1) x (4 - x) (x - 1) = 9 := by
  sorry

end det_equals_nine_l2728_272880


namespace candy_left_l2728_272834

theorem candy_left (initial : ℝ) (morning : ℝ) (afternoon : ℝ) :
  initial = 38 →
  morning = 7.5 →
  afternoon = 15.25 →
  initial - morning - afternoon = 15.25 :=
by
  sorry

end candy_left_l2728_272834


namespace x_value_l2728_272825

theorem x_value (w y z x : ℕ) 
  (hw : w = 95)
  (hz : z = w + 25)
  (hy : y = z + 15)
  (hx : x = y + 10) : 
  x = 145 := by
  sorry

end x_value_l2728_272825


namespace longest_side_is_80_l2728_272803

/-- A rectangle with specific properties -/
structure SpecialRectangle where
  length : ℝ
  width : ℝ
  perimeter_eq : length + width = 120
  area_eq : length * width = 2400

/-- The longest side of a SpecialRectangle is 80 -/
theorem longest_side_is_80 (rect : SpecialRectangle) : 
  max rect.length rect.width = 80 := by
  sorry

#check longest_side_is_80

end longest_side_is_80_l2728_272803


namespace quadratic_equation_in_y_l2728_272867

theorem quadratic_equation_in_y (x y : ℝ) 
  (eq1 : 3 * x^2 + 5 * x + 4 * y + 2 = 0)
  (eq2 : 3 * x + y + 4 = 0) : 
  y^2 + 15 * y + 2 = 0 := by
sorry

end quadratic_equation_in_y_l2728_272867


namespace f_extrema_l2728_272871

/-- A cubic function f(x) = x³ - px² - qx that is tangent to the x-axis at (1,0) -/
def f (p q : ℝ) (x : ℝ) : ℝ := x^3 - p*x^2 - q*x

/-- The condition that f(x) is tangent to the x-axis at (1,0) -/
def is_tangent (p q : ℝ) : Prop :=
  f p q 1 = 0 ∧ (p + q = 1) ∧ (p^2 + 4*q = 0)

theorem f_extrema (p q : ℝ) (h : is_tangent p q) :
  (∃ x, f p q x = 4/27) ∧ (∀ x, f p q x ≥ 0) ∧ (∃ x, f p q x = 0) :=
sorry

end f_extrema_l2728_272871


namespace phil_change_is_seven_l2728_272895

/-- The change Phil received after buying apples -/
def change_received : ℚ :=
  let number_of_apples : ℕ := 4
  let cost_per_apple : ℚ := 75 / 100
  let amount_paid : ℚ := 10
  amount_paid - (number_of_apples * cost_per_apple)

/-- Proof that Phil received $7.00 in change -/
theorem phil_change_is_seven : change_received = 7 := by
  sorry

end phil_change_is_seven_l2728_272895


namespace quadratic_distinct_roots_condition_l2728_272857

/-- For a quadratic equation x^2 + 2x - k = 0 to have two distinct real roots, k must be greater than -1. -/
theorem quadratic_distinct_roots_condition (k : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 + 2*x - k = 0 ∧ y^2 + 2*y - k = 0) ↔ k > -1 :=
by sorry

end quadratic_distinct_roots_condition_l2728_272857


namespace ratio_percentage_difference_l2728_272854

theorem ratio_percentage_difference (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h_ratio : x / 8 = y / 7) : (y - x) / x = 1 / 8 := by
  sorry

end ratio_percentage_difference_l2728_272854


namespace student_average_age_l2728_272807

theorem student_average_age (num_students : ℕ) (teacher_age : ℕ) (avg_increase : ℕ) :
  num_students = 15 →
  teacher_age = 26 →
  avg_increase = 1 →
  (num_students * 10 + teacher_age) / (num_students + 1) = 10 + avg_increase :=
by sorry

end student_average_age_l2728_272807


namespace negation_of_universal_proposition_l2728_272817

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 + x - 6 < 0) ↔ (∃ x : ℝ, x^2 + x - 6 ≥ 0) := by
  sorry

end negation_of_universal_proposition_l2728_272817


namespace min_value_theorem_l2728_272899

theorem min_value_theorem (p q r s t u v w : ℝ) 
  (h1 : p * q * r * s = 16) (h2 : t * u * v * w = 25) :
  (p * t)^2 + (q * u)^2 + (r * v)^2 + (s * w)^2 ≥ 400 := by
  sorry

end min_value_theorem_l2728_272899


namespace solve_for_n_l2728_272822

theorem solve_for_n (s m k r P : ℝ) (h : P = (s + m) / ((1 + k)^n + r)) :
  n = Real.log ((s + m - P * r) / P) / Real.log (1 + k) := by
  sorry

end solve_for_n_l2728_272822


namespace cube_root_of_square_l2728_272894

theorem cube_root_of_square (x : ℝ) : x > 0 → (x^2)^(1/3) = x^(2/3) := by sorry

end cube_root_of_square_l2728_272894


namespace inequality_solution_range_l2728_272838

theorem inequality_solution_range (m : ℝ) : 
  (∃ (a b : ℤ), ∀ (x : ℤ), (x : ℝ)^2 + (m + 1) * (x : ℝ) + m < 0 ↔ x = a ∨ x = b) →
  (-2 ≤ m ∧ m < -1) ∨ (3 < m ∧ m ≤ 4) :=
sorry

end inequality_solution_range_l2728_272838


namespace f_triangle_condition_l2728_272866

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := x^4 - 4*x + m

-- Define the interval [0, 2]
def I : Set ℝ := Set.Icc 0 2

-- Define the triangle existence condition
def triangle_exists (m : ℝ) : Prop :=
  ∃ (a b c : ℝ), a ∈ I ∧ b ∈ I ∧ c ∈ I ∧
    f m a + f m b > f m c ∧
    f m b + f m c > f m a ∧
    f m c + f m a > f m b

-- State the theorem
theorem f_triangle_condition (m : ℝ) :
  triangle_exists m → m > 14 := by sorry

end f_triangle_condition_l2728_272866


namespace sum_specific_terms_l2728_272891

/-- Given a sequence {a_n} where S_n = n^2 - 1 for n ∈ ℕ+, prove a_1 + a_3 + a_5 + a_7 + a_9 = 44 -/
theorem sum_specific_terms (a : ℕ → ℝ) (S : ℕ → ℝ) : 
  (∀ n : ℕ+, S n = n^2 - 1) → 
  (∀ n : ℕ+, S n - S (n-1) = a n) → 
  a 1 + a 3 + a 5 + a 7 + a 9 = 44 := by
sorry

end sum_specific_terms_l2728_272891


namespace select_students_equality_l2728_272821

/-- The number of ways to select 5 students from a class of 50, including one president and one 
    vice-president, with at least one of the president or vice-president attending. -/
def select_students (n : ℕ) (k : ℕ) (total : ℕ) (leaders : ℕ) : ℕ :=
  Nat.choose leaders 1 * Nat.choose (total - leaders) (k - 1) +
  Nat.choose leaders 2 * Nat.choose (total - leaders) (k - 2)

theorem select_students_equality :
  select_students 5 5 50 2 = Nat.choose 50 5 - Nat.choose 48 5 :=
sorry

end select_students_equality_l2728_272821


namespace constant_term_expansion_l2728_272882

/-- The binomial coefficient -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The constant term in the expansion of (1/x + 2x)^6 -/
def constantTerm : ℕ :=
  binomial 6 3 * (2^3)

theorem constant_term_expansion :
  constantTerm = 160 := by sorry

end constant_term_expansion_l2728_272882


namespace distance_to_focus_is_4_l2728_272869

/-- The distance from a point on the parabola y^2 = 4x with x-coordinate 3 to its focus -/
def distance_to_focus (y : ℝ) : ℝ :=
  4

/-- A point P lies on the parabola y^2 = 4x -/
def on_parabola (x y : ℝ) : Prop :=
  y^2 = 4*x

theorem distance_to_focus_is_4 :
  ∀ y : ℝ, on_parabola 3 y → distance_to_focus y = 4 := by
  sorry

end distance_to_focus_is_4_l2728_272869


namespace tan_range_proof_l2728_272820

theorem tan_range_proof (x : ℝ) (hx : x ∈ Set.Icc (-π/4) (π/4) ∧ x ≠ 0) :
  ∃ y, y = Real.tan (π/2 - x) ↔ y ∈ Set.Iic (-1) ∪ Set.Ici 1 :=
sorry

end tan_range_proof_l2728_272820


namespace first_battery_was_voltaic_pile_l2728_272868

/-- Represents a battery -/
structure Battery where
  year : Nat
  creator : String
  components : List String

/-- The first recognized battery in the world -/
def first_battery : Battery :=
  { year := 1800,
    creator := "Alessandro Volta",
    components := ["different metals", "electrolyte"] }

/-- Theorem stating that the first recognized battery was the Voltaic pile -/
theorem first_battery_was_voltaic_pile :
  first_battery.year = 1800 ∧
  first_battery.creator = "Alessandro Volta" ∧
  first_battery.components = ["different metals", "electrolyte"] :=
by sorry

#check first_battery_was_voltaic_pile

end first_battery_was_voltaic_pile_l2728_272868


namespace min_cubes_in_block_l2728_272849

theorem min_cubes_in_block (l m n : ℕ) : 
  (l - 1) * (m - 1) * (n - 1) = 252 → 
  l * m * n ≥ 392 ∧ 
  (∃ (l' m' n' : ℕ), (l' - 1) * (m' - 1) * (n' - 1) = 252 ∧ l' * m' * n' = 392) :=
by sorry

end min_cubes_in_block_l2728_272849


namespace no_valid_sequence_exists_l2728_272887

theorem no_valid_sequence_exists : ¬ ∃ (seq : Fin 100 → ℤ),
  (∀ i, Odd (seq i)) ∧ 
  (∀ i, i + 4 < 100 → ∃ k, (seq i + seq (i+1) + seq (i+2) + seq (i+3) + seq (i+4)) = k^2) ∧
  (∀ i, i + 8 < 100 → ∃ k, (seq i + seq (i+1) + seq (i+2) + seq (i+3) + seq (i+4) + 
                             seq (i+5) + seq (i+6) + seq (i+7) + seq (i+8)) = k^2) :=
by sorry

end no_valid_sequence_exists_l2728_272887


namespace gcd_inequality_l2728_272824

theorem gcd_inequality (n : ℕ) :
  (∀ k ∈ Finset.range 34, Nat.gcd n (n + k) < Nat.gcd n (n + k + 1)) →
  Nat.gcd n (n + 35) < Nat.gcd n (n + 36) := by
sorry

end gcd_inequality_l2728_272824


namespace max_sum_of_digits_watch_l2728_272832

-- Define the type for hours and minutes
def Hour := Fin 12
def Minute := Fin 60

-- Function to calculate the sum of digits
def sumOfDigits (n : Nat) : Nat :=
  let digits := n.repr.toList.map (λc => c.toNat - '0'.toNat)
  digits.sum

-- Define the theorem
theorem max_sum_of_digits_watch :
  ∃ (h : Hour) (m : Minute),
    ∀ (h' : Hour) (m' : Minute),
      sumOfDigits (h.val + 1) + sumOfDigits m.val ≥ 
      sumOfDigits (h'.val + 1) + sumOfDigits m'.val ∧
      sumOfDigits (h.val + 1) + sumOfDigits m.val = 23 :=
by sorry

end max_sum_of_digits_watch_l2728_272832


namespace ac_squared_gt_bc_squared_l2728_272814

theorem ac_squared_gt_bc_squared (a b c : ℝ) : a > b → a * c^2 > b * c^2 := by
  sorry

end ac_squared_gt_bc_squared_l2728_272814


namespace partial_fraction_decomposition_l2728_272890

theorem partial_fraction_decomposition :
  ∃ (P Q R : ℝ),
    (∀ x : ℝ, x ≠ 4 ∧ x ≠ 2 →
      5 * x^2 / ((x - 4) * (x - 2)^3) = P / (x - 4) + Q / (x - 2) + R / (x - 2)^3) ∧
    P = 10 ∧ Q = -10 ∧ R = -10 := by
  sorry

end partial_fraction_decomposition_l2728_272890


namespace economic_loss_scientific_notation_l2728_272818

-- Define the original number in millions
def original_number : ℝ := 16823

-- Define the scientific notation components
def coefficient : ℝ := 1.6823
def exponent : ℤ := 4

-- Theorem statement
theorem economic_loss_scientific_notation :
  original_number = coefficient * (10 : ℝ) ^ exponent :=
sorry

end economic_loss_scientific_notation_l2728_272818


namespace stratified_sampling_equal_probability_l2728_272889

/-- Represents the number of individuals in each stratum -/
structure StrataSize where
  general : ℕ
  deputy : ℕ
  logistics : ℕ

/-- Represents the sample size for each stratum -/
structure StrataSample where
  general : ℕ
  deputy : ℕ
  logistics : ℕ

/-- The total population size -/
def totalPopulation (s : StrataSize) : ℕ :=
  s.general + s.deputy + s.logistics

/-- The total sample size -/
def totalSample (s : StrataSample) : ℕ :=
  s.general + s.deputy + s.logistics

/-- The probability of selection for an individual in a given stratum -/
def selectionProbability (popSize : ℕ) (sampleSize : ℕ) : ℚ :=
  sampleSize / popSize

theorem stratified_sampling_equal_probability 
  (strata : StrataSize) 
  (sample : StrataSample) 
  (h1 : totalPopulation strata = 160)
  (h2 : strata.general = 112)
  (h3 : strata.deputy = 16)
  (h4 : strata.logistics = 32)
  (h5 : totalSample sample = 20) :
  ∃ (p : ℚ), 
    selectionProbability strata.general sample.general = p ∧
    selectionProbability strata.deputy sample.deputy = p ∧
    selectionProbability strata.logistics sample.logistics = p :=
sorry

end stratified_sampling_equal_probability_l2728_272889


namespace isosceles_triangle_two_two_one_l2728_272836

/-- Checks if three numbers can form a triangle based on the triangle inequality theorem -/
def is_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- Checks if three numbers can form an isosceles triangle -/
def is_isosceles_triangle (a b c : ℝ) : Prop :=
  is_triangle a b c ∧ (a = b ∨ b = c ∨ c = a)

/-- Theorem: The set of side lengths (2, 2, 1) forms an isosceles triangle -/
theorem isosceles_triangle_two_two_one :
  is_isosceles_triangle 2 2 1 := by
  sorry

end isosceles_triangle_two_two_one_l2728_272836


namespace bread_cost_l2728_272800

/-- Proves that the cost of a loaf of bread is $2 given the specified conditions --/
theorem bread_cost (total_budget : ℝ) (candy_cost : ℝ) (turkey_proportion : ℝ) (money_left : ℝ)
  (h1 : total_budget = 32)
  (h2 : candy_cost = 2)
  (h3 : turkey_proportion = 1/3)
  (h4 : money_left = 18)
  : ∃ (bread_cost : ℝ),
    bread_cost = 2 ∧
    money_left = total_budget - candy_cost - turkey_proportion * (total_budget - candy_cost) - bread_cost :=
by
  sorry

end bread_cost_l2728_272800


namespace franks_reading_time_l2728_272864

/-- Represents the problem of calculating Frank's effective reading time --/
theorem franks_reading_time (total_pages : ℕ) (reading_speed : ℕ) (total_days : ℕ) :
  total_pages = 2345 →
  reading_speed = 50 →
  total_days = 34 →
  ∃ (effective_time : ℚ),
    effective_time > 2.03 ∧
    effective_time < 2.05 ∧
    effective_time = (total_pages : ℚ) / reading_speed / ((2 * total_days : ℚ) / 3) :=
by sorry

end franks_reading_time_l2728_272864


namespace barbie_gave_four_pairs_l2728_272811

/-- The number of pairs of earrings Barbie bought -/
def total_earrings : ℕ := 12

/-- The number of pairs of earrings Barbie gave to Alissa -/
def earrings_given : ℕ := 4

/-- Alissa's total collection after receiving earrings from Barbie -/
def alissa_total (x : ℕ) : ℕ := 3 * x

theorem barbie_gave_four_pairs :
  earrings_given = 4 ∧
  alissa_total earrings_given + earrings_given = total_earrings :=
by sorry

end barbie_gave_four_pairs_l2728_272811


namespace van_rental_cost_l2728_272876

/-- Calculates the total cost of van rental given the specified conditions -/
theorem van_rental_cost 
  (daily_rate : ℝ) 
  (mileage_rate : ℝ) 
  (num_days : ℕ) 
  (num_miles : ℕ) 
  (booking_fee : ℝ) 
  (h1 : daily_rate = 30)
  (h2 : mileage_rate = 0.25)
  (h3 : num_days = 3)
  (h4 : num_miles = 450)
  (h5 : booking_fee = 15) :
  daily_rate * num_days + mileage_rate * num_miles + booking_fee = 217.5 := by
  sorry


end van_rental_cost_l2728_272876


namespace keegan_class_count_l2728_272855

/-- Calculates the number of classes Keegan is taking given his school schedule --/
theorem keegan_class_count :
  ∀ (total_school_time : ℝ) 
    (history_chem_time : ℝ) 
    (avg_other_class_time : ℝ),
  total_school_time = 7.5 →
  history_chem_time = 1.5 →
  avg_other_class_time = 72 / 60 →
  (total_school_time - history_chem_time) / avg_other_class_time + 2 = 7 :=
by
  sorry

end keegan_class_count_l2728_272855


namespace point_in_region_l2728_272844

-- Define the plane region
def in_region (x y : ℝ) : Prop := 2 * x + y - 6 < 0

-- Theorem to prove
theorem point_in_region : in_region 0 1 := by
  sorry

end point_in_region_l2728_272844


namespace ratio_equivalence_l2728_272828

theorem ratio_equivalence : ∃ (x y : ℚ) (z : ℕ),
  (4 : ℚ) / 5 = 20 / x ∧
  (4 : ℚ) / 5 = y / 20 ∧
  (4 : ℚ) / 5 = (z : ℚ) / 100 := by
  sorry

end ratio_equivalence_l2728_272828


namespace problem_1_problem_2_problem_3_problem_4_l2728_272805

-- Problem 1
theorem problem_1 : 12 - (-18) - |(-7)| + 15 = 38 := by sorry

-- Problem 2
theorem problem_2 : -24 / (-3/2) + 6 * (-1/3) = 14 := by sorry

-- Problem 3
theorem problem_3 : (-7/9 + 5/6 - 1/4) * (-36) = 7 := by sorry

-- Problem 4
theorem problem_4 : -1^2 + 1/4 * (-2)^3 + (-3)^2 = 6 := by sorry

end problem_1_problem_2_problem_3_problem_4_l2728_272805
