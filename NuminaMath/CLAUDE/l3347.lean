import Mathlib

namespace pet_walking_problem_l3347_334710

def smallest_common_multiple (a b : ℕ) : ℕ := Nat.lcm a b

theorem pet_walking_problem (gabe_group_size steven_group_size : ℕ) 
  (h1 : gabe_group_size = 2) 
  (h2 : steven_group_size = 10) : 
  smallest_common_multiple gabe_group_size steven_group_size = 20 := by
  sorry

#check pet_walking_problem

end pet_walking_problem_l3347_334710


namespace mango_distribution_l3347_334777

/-- Given 560 mangoes, if half are sold and the remainder is distributed evenly among 8 neighbors,
    each neighbor receives 35 mangoes. -/
theorem mango_distribution (total_mangoes : ℕ) (neighbors : ℕ) 
    (h1 : total_mangoes = 560) 
    (h2 : neighbors = 8) : 
  (total_mangoes / 2) / neighbors = 35 := by
  sorry

end mango_distribution_l3347_334777


namespace ellipse_points_equiv_target_set_l3347_334726

/-- An ellipse passing through (2,1) with the given conditions -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h1 : a > b
  h2 : b > 0
  h3 : 4 / a^2 + 1 / b^2 = 1

/-- The set of points on the ellipse satisfying |y| > 1 -/
def ellipse_points (e : Ellipse) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 / e.a^2 + p.2^2 / e.b^2 = 1 ∧ |p.2| > 1}

/-- The target set -/
def target_set : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 < 5 ∧ |p.2| > 1}

/-- The main theorem -/
theorem ellipse_points_equiv_target_set (e : Ellipse) :
  ellipse_points e = target_set := by sorry

end ellipse_points_equiv_target_set_l3347_334726


namespace initial_men_count_l3347_334725

/-- Represents the work scenario with given parameters -/
structure WorkScenario where
  men : ℕ
  hoursPerDay : ℕ
  depth : ℕ

/-- Calculates the work done in a given scenario -/
def workDone (scenario : WorkScenario) : ℕ :=
  scenario.men * scenario.hoursPerDay * scenario.depth

theorem initial_men_count : ∃ (initialMen : ℕ),
  let scenario1 := WorkScenario.mk initialMen 8 30
  let scenario2 := WorkScenario.mk (initialMen + 55) 6 50
  workDone scenario1 = workDone scenario2 ∧ initialMen = 275 := by
  sorry

#check initial_men_count

end initial_men_count_l3347_334725


namespace right_triangle_probability_l3347_334714

/-- A 3x3 grid of nine unit squares -/
structure Grid :=
  (vertices : Fin 16 → ℝ × ℝ)

/-- Three vertices selected from the grid -/
structure SelectedVertices :=
  (v1 v2 v3 : Fin 16)

/-- Predicate to check if three vertices form a right triangle -/
def is_right_triangle (g : Grid) (sv : SelectedVertices) : Prop :=
  sorry

/-- The total number of ways to select three vertices from 16 -/
def total_selections : ℕ := Nat.choose 16 3

/-- The number of right triangles that can be formed -/
def right_triangle_count (g : Grid) : ℕ :=
  sorry

/-- The main theorem stating the probability -/
theorem right_triangle_probability (g : Grid) :
  (right_triangle_count g : ℚ) / total_selections = 5 / 14 :=
sorry

end right_triangle_probability_l3347_334714


namespace quadratic_square_of_binomial_l3347_334700

/-- Given a quadratic expression of the form bx^2 + 16x + 16,
    if it is the square of a binomial, then b = 4. -/
theorem quadratic_square_of_binomial (b : ℝ) :
  (∃ t u : ℝ, ∀ x : ℝ, bx^2 + 16*x + 16 = (t*x + u)^2) →
  b = 4 := by
  sorry

end quadratic_square_of_binomial_l3347_334700


namespace division_problem_l3347_334747

theorem division_problem (dividend : ℕ) (divisor : ℕ) (remainder : ℕ) (quotient : ℕ) : 
  dividend = 507 → divisor = 8 → remainder = 19 → 
  dividend = divisor * quotient + remainder →
  quotient = 61 := by
sorry

end division_problem_l3347_334747


namespace profit_ratio_theorem_l3347_334752

/-- Given two partners p and q with investment ratio 7:5, where p invests for 10 months
    and q invests for 20 months, prove that the ratio of their profits is 7:10 -/
theorem profit_ratio_theorem (p q : ℕ) (investment_p investment_q : ℝ) 
  (time_p time_q : ℕ) (profit_p profit_q : ℝ) :
  investment_p / investment_q = 7 / 5 →
  time_p = 10 →
  time_q = 20 →
  profit_p = investment_p * time_p →
  profit_q = investment_q * time_q →
  profit_p / profit_q = 7 / 10 := by
  sorry

end profit_ratio_theorem_l3347_334752


namespace smallest_yummy_number_l3347_334758

/-- Definition of a yummy number -/
def is_yummy (A : ℕ) : Prop :=
  ∃ n : ℕ+, n * (2 * A + n - 1) = 2 * 2023

/-- Theorem stating that 1011 is the smallest yummy number -/
theorem smallest_yummy_number :
  is_yummy 1011 ∧ ∀ A : ℕ, A < 1011 → ¬is_yummy A :=
sorry

end smallest_yummy_number_l3347_334758


namespace larger_number_proof_l3347_334770

theorem larger_number_proof (a b : ℕ+) (h1 : Nat.gcd a b = 59) 
  (h2 : Nat.lcm a b = 12272) (h3 : 13 ∣ Nat.lcm a b) (h4 : 16 ∣ Nat.lcm a b) :
  max a b = 944 := by
  sorry

end larger_number_proof_l3347_334770


namespace adjacent_sum_negative_total_sum_positive_l3347_334784

theorem adjacent_sum_negative_total_sum_positive :
  ∃ (a₁ a₂ a₃ a₄ a₅ : ℝ),
    (a₁ + a₂ < 0) ∧
    (a₂ + a₃ < 0) ∧
    (a₃ + a₄ < 0) ∧
    (a₄ + a₅ < 0) ∧
    (a₅ + a₁ < 0) ∧
    (a₁ + a₂ + a₃ + a₄ + a₅ > 0) :=
  sorry

end adjacent_sum_negative_total_sum_positive_l3347_334784


namespace pen_probabilities_l3347_334703

/-- The number of pens in the box -/
def total_pens : ℕ := 6

/-- The number of first-class pens -/
def first_class_pens : ℕ := 4

/-- The number of second-class pens -/
def second_class_pens : ℕ := 2

/-- The number of pens drawn -/
def drawn_pens : ℕ := 2

/-- The probability of drawing exactly one first-class pen -/
def prob_one_first_class : ℚ := 8 / 15

/-- The probability of drawing at least one second-class pen -/
def prob_second_class : ℚ := 3 / 5

theorem pen_probabilities :
  (total_pens = first_class_pens + second_class_pens) →
  (prob_one_first_class = (Nat.choose first_class_pens 1 * Nat.choose second_class_pens 1 : ℚ) / Nat.choose total_pens drawn_pens) ∧
  (prob_second_class = 1 - (Nat.choose first_class_pens drawn_pens : ℚ) / Nat.choose total_pens drawn_pens) :=
by sorry

end pen_probabilities_l3347_334703


namespace small_animal_weight_l3347_334746

def bear_weight_gain (total_weight : ℝ) (berry_fraction : ℝ) (acorn_multiplier : ℝ) (salmon_fraction : ℝ) : ℝ :=
  let berry_weight := total_weight * berry_fraction
  let acorn_weight := berry_weight * acorn_multiplier
  let remaining_weight := total_weight - (berry_weight + acorn_weight)
  let salmon_weight := remaining_weight * salmon_fraction
  total_weight - (berry_weight + acorn_weight + salmon_weight)

theorem small_animal_weight :
  bear_weight_gain 1000 (1/5) 2 (1/2) = 200 := by
  sorry

end small_animal_weight_l3347_334746


namespace train_departure_sequences_l3347_334769

theorem train_departure_sequences :
  let total_trains : ℕ := 6
  let trains_per_group : ℕ := 3
  let num_special_trains : ℕ := 2  -- G1 and G2
  let num_regular_trains : ℕ := total_trains - num_special_trains

  -- Number of ways to choose trains for G1's group (excluding G1 itself)
  let group_formations : ℕ := Nat.choose num_regular_trains (trains_per_group - 1)

  -- Number of permutations for each group
  let group_permutations : ℕ := Nat.factorial trains_per_group

  -- Total number of departure sequences
  group_formations * group_permutations * group_permutations = 216 :=
by
  sorry

end train_departure_sequences_l3347_334769


namespace root_sum_bound_implies_m_range_l3347_334757

theorem root_sum_bound_implies_m_range (m : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁^2 - 2*x₁ + m + 2 = 0 ∧
               x₂^2 - 2*x₂ + m + 2 = 0 ∧
               x₁ ≠ x₂ ∧
               |x₁| + |x₂| ≤ 3) →
  -13/4 ≤ m ∧ m ≤ -1 :=
by sorry

end root_sum_bound_implies_m_range_l3347_334757


namespace sqrt_4_not_plus_minus_2_l3347_334759

theorem sqrt_4_not_plus_minus_2 : ¬(Real.sqrt 4 = 2 ∨ Real.sqrt 4 = -2) := by
  sorry

end sqrt_4_not_plus_minus_2_l3347_334759


namespace curve_transformation_l3347_334768

/-- Given a scaling transformation and the equation of the transformed curve,
    prove the equation of the original curve. -/
theorem curve_transformation (x y x' y' : ℝ) :
  (x' = 5 * x) →
  (y' = 3 * y) →
  (x'^2 + 4 * y'^2 = 1) →
  (25 * x^2 + 36 * y^2 = 1) := by
  sorry

end curve_transformation_l3347_334768


namespace line_intersects_circle_right_angle_l3347_334737

theorem line_intersects_circle_right_angle (k : ℝ) :
  (∃ (P Q : ℝ × ℝ), 
    P.1^2 + P.2^2 = 1 ∧ 
    Q.1^2 + Q.2^2 = 1 ∧ 
    P.2 = k * P.1 + 1 ∧ 
    Q.2 = k * Q.1 + 1 ∧ 
    (P.1 * Q.1 + P.2 * Q.2 = 0)) →
  k = 1 ∨ k = -1 :=
by sorry

end line_intersects_circle_right_angle_l3347_334737


namespace symmetry_sum_l3347_334779

/-- Two points are symmetric with respect to the origin if their coordinates sum to zero -/
def symmetric_wrt_origin (p q : ℝ × ℝ) : Prop :=
  p.1 + q.1 = 0 ∧ p.2 + q.2 = 0

theorem symmetry_sum (m n : ℝ) : 
  symmetric_wrt_origin (m, 1) (-2, n) → m + n = 1 := by
  sorry

end symmetry_sum_l3347_334779


namespace complex_real_condition_l3347_334783

theorem complex_real_condition (m : ℝ) : 
  let z : ℂ := (m + 2*I) / (3 - 4*I)
  (∃ (x : ℝ), z = x) → m = -3/2 := by
  sorry

end complex_real_condition_l3347_334783


namespace unattainable_value_of_function_l3347_334767

theorem unattainable_value_of_function (x : ℝ) (y : ℝ) : 
  x ≠ -4/3 → 
  y = (2-x) / (3*x+4) → 
  y ≠ -1/3 := by
sorry

end unattainable_value_of_function_l3347_334767


namespace peaches_theorem_l3347_334705

def peaches_problem (peaches_per_basket : ℕ) (num_baskets : ℕ) (peaches_eaten : ℕ) (peaches_per_box : ℕ) : Prop :=
  let total_peaches := peaches_per_basket * num_baskets
  let remaining_peaches := total_peaches - peaches_eaten
  remaining_peaches / peaches_per_box = 8

theorem peaches_theorem : 
  peaches_problem 25 5 5 15 := by sorry

end peaches_theorem_l3347_334705


namespace max_thursday_money_l3347_334773

def tuesday_amount : ℕ := 8

def wednesday_amount : ℕ := 5 * tuesday_amount

def thursday_amount : ℕ := tuesday_amount + 41

theorem max_thursday_money : thursday_amount = 49 := by
  sorry

end max_thursday_money_l3347_334773


namespace max_magic_triangle_sum_l3347_334733

def MagicTriangle : Type := Fin 6 → Nat

def isValidTriangle (t : MagicTriangle) : Prop :=
  (∀ i : Fin 6, t i ≥ 4 ∧ t i ≤ 9) ∧
  (∀ i j : Fin 6, i ≠ j → t i ≠ t j)

def sumS (t : MagicTriangle) : Nat :=
  3 * t 0 + 2 * t 1 + 2 * t 2 + t 3 + t 4

def isBalanced (t : MagicTriangle) : Prop :=
  sumS t = 2 * t 2 + t 3 + 2 * t 4 ∧
  sumS t = 2 * t 4 + t 5 + 2 * t 1

theorem max_magic_triangle_sum :
  ∀ t : MagicTriangle, isValidTriangle t → isBalanced t →
  sumS t ≤ 40 :=
sorry

end max_magic_triangle_sum_l3347_334733


namespace quadratic_coefficient_sum_l3347_334748

theorem quadratic_coefficient_sum (m n : ℤ) : 
  (∀ x : ℤ, (x + 2) * (x - 1) = x^2 + m*x + n) → m + n = -1 := by
  sorry

end quadratic_coefficient_sum_l3347_334748


namespace smallest_square_multiplier_l3347_334719

def y : ℕ := 2^4 * 3^3 * 5^4 * 7^2 * 6^7 * 8^3 * 9^10

theorem smallest_square_multiplier (n : ℕ) : 
  (∃ m : ℕ, n * y = m^2) ∧ (∀ k : ℕ, 0 < k ∧ k < n → ¬∃ m : ℕ, k * y = m^2) → n = 1 :=
by sorry

end smallest_square_multiplier_l3347_334719


namespace smallest_sum_of_squares_l3347_334775

theorem smallest_sum_of_squares (x y : ℕ) : 
  x^2 - y^2 = 187 → ∀ a b : ℕ, a^2 - b^2 = 187 → x^2 + y^2 ≤ a^2 + b^2 → x^2 + y^2 = 205 :=
by sorry

end smallest_sum_of_squares_l3347_334775


namespace S_2n_plus_one_not_div_by_three_l3347_334708

/-- 
For a non-negative integer n, S_n is defined as the sum of squares 
of the coefficients of the polynomial (1+x)^n
-/
def S (n : ℕ) : ℕ := (Finset.range (n + 1)).sum (fun k => (Nat.choose n k) ^ 2)

/-- 
For any non-negative integer n, S(2n) + 1 is not divisible by 3
-/
theorem S_2n_plus_one_not_div_by_three (n : ℕ) : ¬ (3 ∣ (S (2 * n) + 1)) := by
  sorry

end S_2n_plus_one_not_div_by_three_l3347_334708


namespace third_generation_tail_length_l3347_334790

/-- The tail length of a generation of kittens -/
def tail_length (n : ℕ) : ℝ :=
  if n = 0 then 16
  else tail_length (n - 1) * 1.25

/-- The theorem stating that the third generation's tail length is 25 cm -/
theorem third_generation_tail_length :
  tail_length 2 = 25 := by sorry

end third_generation_tail_length_l3347_334790


namespace test_questions_count_l3347_334789

theorem test_questions_count : ∃ (n : ℕ), 
  (n > 0) ∧ 
  (5 * n = 45) ∧ 
  (32 > 0.70 * 45) ∧ 
  (32 < 0.77 * 45) := by
  sorry

end test_questions_count_l3347_334789


namespace cake_measuring_l3347_334749

theorem cake_measuring (flour_needed : ℚ) (milk_needed : ℚ) (cup_capacity : ℚ) : 
  flour_needed = 10/3 ∧ milk_needed = 3/2 ∧ cup_capacity = 1/3 → 
  Int.ceil (flour_needed / cup_capacity) + Int.ceil (milk_needed / cup_capacity) = 15 := by
sorry

end cake_measuring_l3347_334749


namespace highway_vehicle_ratio_l3347_334723

theorem highway_vehicle_ratio (total_vehicles : ℕ) (num_trucks : ℕ) : 
  total_vehicles = 300 → 
  num_trucks = 100 → 
  ∃ (k : ℕ), k * num_trucks = total_vehicles - num_trucks → 
  (total_vehicles - num_trucks) / num_trucks = 2 := by
  sorry

end highway_vehicle_ratio_l3347_334723


namespace fundraising_shortfall_l3347_334760

def goal : ℚ := 500
def pizza_price : ℚ := 12
def fries_price : ℚ := 0.3
def soda_price : ℚ := 2
def pizza_sold : ℕ := 15
def fries_sold : ℕ := 40
def soda_sold : ℕ := 25

theorem fundraising_shortfall :
  goal - (pizza_price * pizza_sold + fries_price * fries_sold + soda_price * soda_sold) = 258 := by
  sorry

end fundraising_shortfall_l3347_334760


namespace sum_of_possible_x_coordinates_of_A_sum_of_possible_x_coordinates_of_A_is_400_l3347_334745

/-- Given two triangles ABC and ADE with specified areas and coordinates for points B, C, D, and E,
    prove that the sum of all possible x-coordinates of point A is 400. -/
theorem sum_of_possible_x_coordinates_of_A : ℝ → Prop :=
  fun sum_x =>
    ∀ (A B C D E : ℝ × ℝ)
      (area_ABC area_ADE : ℝ),
    B = (0, 0) →
    C = (200, 0) →
    D = (600, 400) →
    E = (610, 410) →
    area_ABC = 3000 →
    area_ADE = 6000 →
    (∃ (x₁ x₂ : ℝ), 
      (A.1 = x₁ ∨ A.1 = x₂) ∧ 
      sum_x = x₁ + x₂) →
    sum_x = 400

/-- Proof of the theorem -/
theorem sum_of_possible_x_coordinates_of_A_is_400 :
  sum_of_possible_x_coordinates_of_A 400 := by
  sorry

end sum_of_possible_x_coordinates_of_A_sum_of_possible_x_coordinates_of_A_is_400_l3347_334745


namespace simultaneous_inequalities_l3347_334741

theorem simultaneous_inequalities (x : ℝ) :
  x^3 - 11*x^2 + 10*x < 0 ∧ x^3 - 12*x^2 + 32*x > 0 → (1 < x ∧ x < 4) ∨ (8 < x ∧ x < 10) :=
by sorry

end simultaneous_inequalities_l3347_334741


namespace prob_at_least_two_dice_less_than_10_l3347_334715

/-- The probability of a single 20-sided die showing a number less than 10 -/
def p_less_than_10 : ℚ := 9 / 20

/-- The probability of a single 20-sided die showing a number 10 or above -/
def p_10_or_above : ℚ := 11 / 20

/-- The number of dice rolled -/
def n : ℕ := 5

/-- The probability of exactly k dice showing a number less than 10 -/
def prob_k (k : ℕ) : ℚ :=
  (n.choose k) * (p_less_than_10 ^ k) * (p_10_or_above ^ (n - k))

/-- The probability of at least two dice showing a number less than 10 -/
def prob_at_least_two : ℚ :=
  prob_k 2 + prob_k 3 + prob_k 4 + prob_k 5

theorem prob_at_least_two_dice_less_than_10 :
  prob_at_least_two = 157439 / 20000 := by
  sorry

end prob_at_least_two_dice_less_than_10_l3347_334715


namespace smallest_block_volume_l3347_334704

theorem smallest_block_volume (N : ℕ) : 
  (∃ x y z : ℕ, 
    N = x * y * z ∧ 
    (x - 1) * (y - 1) * (z - 1) = 231 ∧
    ∀ a b c : ℕ, a * b * c = N → (a - 1) * (b - 1) * (c - 1) = 231 → 
      x * y * z ≤ a * b * c) → 
  N = 384 := by
sorry

end smallest_block_volume_l3347_334704


namespace waiter_income_fraction_l3347_334734

theorem waiter_income_fraction (salary tips income : ℚ) : 
  income = salary + tips → 
  tips = (5 : ℚ) / 4 * salary → 
  tips / income = (5 : ℚ) / 9 := by
  sorry

end waiter_income_fraction_l3347_334734


namespace solution_set_part1_integer_a_part2_l3347_334711

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - 1| - |2*x - a|

-- Part 1
theorem solution_set_part1 :
  {x : ℝ | f 5 x ≥ 0} = {x : ℝ | 2 ≤ x ∧ x ≤ 4} :=
sorry

-- Part 2
theorem integer_a_part2 (a : ℤ) :
  (f a 5 ≥ 3 ∧ f a 6 < 3) → a = 9 :=
sorry

end solution_set_part1_integer_a_part2_l3347_334711


namespace complex_second_quadrant_second_quadrant_implies_neg_real_pos_imag_l3347_334782

/-- A complex number is in the second quadrant if its real part is negative and its imaginary part is positive -/
theorem complex_second_quadrant (z : ℂ) :
  (z.re < 0 ∧ z.im > 0) ↔ (z.arg > Real.pi / 2 ∧ z.arg < Real.pi) :=
by sorry

/-- If a complex number is in the second quadrant, then its real part is negative and its imaginary part is positive -/
theorem second_quadrant_implies_neg_real_pos_imag (z : ℂ) 
  (h : z.arg > Real.pi / 2 ∧ z.arg < Real.pi) : 
  z.re < 0 ∧ z.im > 0 :=
by sorry

end complex_second_quadrant_second_quadrant_implies_neg_real_pos_imag_l3347_334782


namespace relatively_prime_power_sums_l3347_334763

theorem relatively_prime_power_sums (a n m : ℕ) (h_odd : Odd a) (h_pos_n : n > 0) (h_pos_m : m > 0) (h_neq : n ≠ m) :
  Nat.gcd (a^(2^m) + 2^(2^m)) (a^(2^n) + 2^(2^n)) = 1 := by
sorry

end relatively_prime_power_sums_l3347_334763


namespace triangle_vector_relation_l3347_334799

theorem triangle_vector_relation (A B C : ℝ × ℝ) (a b : ℝ × ℝ) :
  (B.1 - C.1, B.2 - C.2) = a →
  (C.1 - A.1, C.2 - A.2) = b →
  (A.1 - B.1, A.2 - B.2) = (b.1 - a.1, b.2 - a.2) := by
  sorry

end triangle_vector_relation_l3347_334799


namespace current_wax_amount_l3347_334755

theorem current_wax_amount (total_required : ℕ) (additional_needed : ℕ) 
  (h1 : total_required = 492)
  (h2 : additional_needed = 481) :
  total_required - additional_needed = 11 := by
  sorry

end current_wax_amount_l3347_334755


namespace soda_cost_l3347_334794

theorem soda_cost (burger_cost soda_cost : ℕ) : 
  (4 * burger_cost + 3 * soda_cost = 440) →
  (3 * burger_cost + 2 * soda_cost = 310) →
  soda_cost = 80 := by
sorry

end soda_cost_l3347_334794


namespace rectangular_plot_breadth_l3347_334709

/-- Proves that for a rectangular plot where the length is thrice the breadth
    and the area is 432 sq m, the breadth is 12 m. -/
theorem rectangular_plot_breadth : 
  ∀ (breadth length area : ℝ),
    length = 3 * breadth →
    area = length * breadth →
    area = 432 →
    breadth = 12 := by
  sorry

end rectangular_plot_breadth_l3347_334709


namespace joan_gemstones_l3347_334785

/-- Represents Joan's rock collection --/
structure RockCollection where
  minerals_yesterday : ℕ
  gemstones : ℕ
  minerals_today : ℕ

/-- Theorem about Joan's rock collection --/
theorem joan_gemstones (collection : RockCollection) 
  (h1 : collection.gemstones = collection.minerals_yesterday / 2)
  (h2 : collection.minerals_today = collection.minerals_yesterday + 6)
  (h3 : collection.minerals_today = 48) : 
  collection.gemstones = 21 := by
sorry

end joan_gemstones_l3347_334785


namespace boat_speed_in_still_water_l3347_334732

/-- 
Given a boat traveling downstream in a stream, this theorem proves that 
the speed of the boat in still water is 5 km/hr, based on the given conditions.
-/
theorem boat_speed_in_still_water 
  (stream_speed : ℝ) 
  (downstream_distance : ℝ) 
  (downstream_time : ℝ) 
  (h1 : stream_speed = 5)
  (h2 : downstream_distance = 100)
  (h3 : downstream_time = 10)
  (h4 : downstream_distance = (boat_speed + stream_speed) * downstream_time) :
  boat_speed = 5 := by
  sorry


end boat_speed_in_still_water_l3347_334732


namespace second_player_winning_strategy_l3347_334792

/-- Represents the possible states of a cell in the game grid -/
inductive Cell
| Empty : Cell
| S : Cell
| O : Cell

/-- Represents the game state -/
structure GameState where
  grid : Vector Cell 2000
  currentPlayer : Nat

/-- Checks if a player has won by forming SOS pattern -/
def hasWon (state : GameState) : Bool :=
  sorry

/-- Checks if the game is a draw -/
def isDraw (state : GameState) : Bool :=
  sorry

/-- Represents a player's strategy -/
def Strategy := GameState → Nat

/-- Checks if a strategy is winning for the given player -/
def isWinningStrategy (player : Nat) (strategy : Strategy) : Prop :=
  sorry

/-- The main theorem stating that the second player has a winning strategy -/
theorem second_player_winning_strategy :
  ∃ (strategy : Strategy), isWinningStrategy 2 strategy :=
sorry

end second_player_winning_strategy_l3347_334792


namespace stationery_box_sheets_l3347_334740

/-- Represents a stationery box with sheets of paper and envelopes -/
structure StationeryBox where
  sheets : ℕ
  envelopes : ℕ

/-- Joe's usage of the stationery box -/
def joe_usage (box : StationeryBox) : Prop :=
  box.sheets - box.envelopes = 70

/-- Lily's usage of the stationery box -/
def lily_usage (box : StationeryBox) : Prop :=
  4 * (box.envelopes - 20) = box.sheets

theorem stationery_box_sheets : 
  ∃ (box : StationeryBox), joe_usage box ∧ lily_usage box ∧ box.sheets = 120 := by
  sorry


end stationery_box_sheets_l3347_334740


namespace inequality_proof_l3347_334761

theorem inequality_proof (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h : ∀ x, f x > deriv f x) (a b : ℝ) (hab : a > b) : 
  Real.exp a * f b > Real.exp b * f a := by
  sorry

end inequality_proof_l3347_334761


namespace randi_has_more_nickels_l3347_334742

/-- The value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- Ray's initial amount in cents -/
def ray_initial_amount : ℕ := 175

/-- Amount given to Peter in cents -/
def amount_to_peter : ℕ := 30

/-- Amount given to Randi in cents -/
def amount_to_randi : ℕ := 2 * amount_to_peter

/-- Number of nickels Randi receives -/
def randi_nickels : ℕ := amount_to_randi / nickel_value

/-- Number of nickels Peter receives -/
def peter_nickels : ℕ := amount_to_peter / nickel_value

theorem randi_has_more_nickels : randi_nickels - peter_nickels = 6 := by
  sorry

end randi_has_more_nickels_l3347_334742


namespace min_distance_curve_to_line_l3347_334778

noncomputable def f (x : ℝ) := x^2 - Real.log x

def line (x : ℝ) := x - 2

theorem min_distance_curve_to_line :
  ∀ x > 0, ∃ d : ℝ,
    d = Real.sqrt 2 ∧
    ∀ y > 0, 
      let p₁ := (x, f x)
      let p₂ := (y, line y)
      d ≤ Real.sqrt ((x - y)^2 + (f x - line y)^2) :=
by sorry

end min_distance_curve_to_line_l3347_334778


namespace inscribed_square_area_l3347_334720

/-- The parabola function -/
def f (x : ℝ) : ℝ := x^2 - 6*x + 8

/-- The side length of the inscribed square -/
noncomputable def s : ℝ := -2 + 2 * Real.sqrt 2

/-- Theorem: The area of the inscribed square is 12 - 8√2 -/
theorem inscribed_square_area :
  let square_area := s^2
  square_area = 12 - 8 * Real.sqrt 2 := by sorry

end inscribed_square_area_l3347_334720


namespace base3_to_base10_conversion_l3347_334718

/-- Converts a list of digits in base 3 to a natural number in base 10 -/
def base3ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (3 ^ i)) 0

/-- The base-3 representation of the number -/
def base3Number : List Nat := [2, 0, 1, 0, 2, 1]

theorem base3_to_base10_conversion :
  base3ToBase10 base3Number = 416 := by
  sorry

end base3_to_base10_conversion_l3347_334718


namespace lines_intersection_l3347_334788

theorem lines_intersection (k : ℝ) : 
  ∃ (x y : ℝ), ∀ (k : ℝ), k * x + y + 3 * k + 1 = 0 ∧ x = -3 ∧ y = -1 := by
  sorry

end lines_intersection_l3347_334788


namespace ordering_of_powers_l3347_334764

theorem ordering_of_powers : 5^15 < 3^20 ∧ 3^20 < 2^30 := by
  sorry

end ordering_of_powers_l3347_334764


namespace purely_imaginary_complex_number_l3347_334774

theorem purely_imaginary_complex_number (m : ℝ) : 
  let z : ℂ := (m^2 - m) + m * Complex.I
  (∃ (y : ℝ), z = y * Complex.I) → m = 1 := by
  sorry

end purely_imaginary_complex_number_l3347_334774


namespace sum_of_solutions_quadratic_l3347_334765

theorem sum_of_solutions_quadratic (x : ℝ) : 
  (∃ a b c : ℝ, (4*x + 7)*(3*x - 5) = 15 ∧ a*x^2 + b*x + c = 0) → 
  (∃ x₁ x₂ : ℝ, (4*x₁ + 7)*(3*x₁ - 5) = 15 ∧ 
                (4*x₂ + 7)*(3*x₂ - 5) = 15 ∧ 
                x₁ + x₂ = -1/12) :=
by sorry

end sum_of_solutions_quadratic_l3347_334765


namespace exists_valid_arrangement_l3347_334728

/-- Represents the positions on the square --/
inductive Position
  | TopLeft
  | TopRight
  | BottomLeft
  | BottomRight
  | Center

/-- Defines whether two positions are connected --/
def connected (p1 p2 : Position) : Prop :=
  (p1 = Position.Center ∨ p2 = Position.Center) ∧ p1 ≠ p2

/-- Defines an arrangement of numbers on the square --/
def Arrangement := Position → ℕ

/-- Checks if the arrangement satisfies the required conditions --/
def valid_arrangement (arr : Arrangement) : Prop :=
  (∀ p1 p2, connected p1 p2 → ∃ d > 1, d ∣ arr p1 ∧ d ∣ arr p2) ∧
  (∀ p1 p2, ¬connected p1 p2 → Nat.gcd (arr p1) (arr p2) = 1)

/-- The main theorem stating the existence of a valid arrangement --/
theorem exists_valid_arrangement : ∃ arr : Arrangement, valid_arrangement arr := by
  sorry

end exists_valid_arrangement_l3347_334728


namespace logical_equivalence_l3347_334795

theorem logical_equivalence (S X Y : Prop) :
  (S → ¬X ∧ ¬Y) ↔ (X ∨ Y → ¬S) := by
  sorry

end logical_equivalence_l3347_334795


namespace product_mod_600_l3347_334727

theorem product_mod_600 : (2537 * 1985) % 600 = 145 := by
  sorry

end product_mod_600_l3347_334727


namespace diamond_three_four_l3347_334786

-- Define the ⊕ operation
def oplus (a b : ℝ) : ℝ := a^2 + 2*a*b

-- Define the ◇ operation
def diamond (a b : ℝ) : ℝ := 4*a + 6*b - (oplus a b)

-- Theorem statement
theorem diamond_three_four : diamond 3 4 = 3 := by sorry

end diamond_three_four_l3347_334786


namespace deshaun_summer_reading_l3347_334736

theorem deshaun_summer_reading 
  (summer_break_days : ℕ) 
  (avg_pages_per_book : ℕ) 
  (closest_person_percentage : ℚ) 
  (second_person_pages_per_day : ℕ) 
  (h1 : summer_break_days = 80)
  (h2 : avg_pages_per_book = 320)
  (h3 : closest_person_percentage = 3/4)
  (h4 : second_person_pages_per_day = 180) :
  ∃ (books_read : ℕ), books_read = 60 ∧ 
    (books_read * avg_pages_per_book : ℚ) = 
      (second_person_pages_per_day * summer_break_days : ℚ) / closest_person_percentage :=
by sorry

end deshaun_summer_reading_l3347_334736


namespace total_onions_grown_l3347_334729

theorem total_onions_grown (nancy_onions dan_onions mike_onions : ℕ) 
  (h1 : nancy_onions = 2)
  (h2 : dan_onions = 9)
  (h3 : mike_onions = 4) :
  nancy_onions + dan_onions + mike_onions = 15 := by
  sorry

end total_onions_grown_l3347_334729


namespace arithmetic_sequence_ratio_l3347_334707

/-- Two arithmetic sequences and their sums -/
structure ArithmeticSequencePair where
  a : ℕ → ℚ  -- First arithmetic sequence
  b : ℕ → ℚ  -- Second arithmetic sequence
  S : ℕ → ℚ  -- Sum of first n terms of a
  T : ℕ → ℚ  -- Sum of first n terms of b

/-- The main theorem -/
theorem arithmetic_sequence_ratio 
  (seq : ArithmeticSequencePair)
  (h : ∀ n : ℕ, seq.S n / seq.T n = (n + 3) / (2 * n + 1)) :
  seq.a 6 / seq.b 6 = 14 / 23 := by
  sorry

end arithmetic_sequence_ratio_l3347_334707


namespace quadratic_root_relation_l3347_334721

theorem quadratic_root_relation (p q : ℝ) : 
  (∀ x : ℝ, x^2 - p^2*x + p*q = 0 ↔ (∃ y : ℝ, y^2 + p*y + q = 0 ∧ x = y + 1)) →
  (p = 1 ∨ (p = -2 ∧ q = -1)) :=
by sorry

end quadratic_root_relation_l3347_334721


namespace g_at_5_l3347_334791

/-- A function g satisfying the given equation for all real x -/
def g : ℝ → ℝ := sorry

/-- The main property of function g -/
axiom g_property : ∀ x : ℝ, g x + 3 * g (2 - x) = 4 * x^2 - 1

/-- The theorem to be proved -/
theorem g_at_5 : g 5 = 3/4 := by sorry

end g_at_5_l3347_334791


namespace arcsin_three_fifths_cos_tan_l3347_334739

theorem arcsin_three_fifths_cos_tan :
  (Real.cos (Real.arcsin (3/5)) = 4/5) ∧ 
  (Real.tan (Real.arcsin (3/5)) = 3/4) := by
sorry

end arcsin_three_fifths_cos_tan_l3347_334739


namespace product_of_roots_l3347_334797

theorem product_of_roots (x₁ x₂ : ℝ) 
  (h1 : x₁^2 - 2*x₁ = 2) 
  (h2 : x₂^2 - 2*x₂ = 2) 
  (h3 : x₁ ≠ x₂) : 
  x₁ * x₂ = -2 := by
sorry

end product_of_roots_l3347_334797


namespace horse_fertilizer_production_l3347_334716

-- Define the given constants
def num_horses : ℕ := 80
def total_acres : ℕ := 20
def gallons_per_acre : ℕ := 400
def acres_per_day : ℕ := 4
def total_days : ℕ := 25

-- Define the function to calculate daily fertilizer production per horse
def daily_fertilizer_per_horse : ℚ :=
  (total_acres * gallons_per_acre : ℚ) / (num_horses * total_days)

-- Theorem statement
theorem horse_fertilizer_production :
  daily_fertilizer_per_horse = 20 := by
  sorry

end horse_fertilizer_production_l3347_334716


namespace tigers_season_games_l3347_334717

def total_games (games_won : ℕ) (games_lost : ℕ) : ℕ :=
  games_won + games_lost

theorem tigers_season_games :
  let games_won : ℕ := 18
  let games_lost : ℕ := games_won + 21
  total_games games_won games_lost = 57 := by
  sorry

end tigers_season_games_l3347_334717


namespace intersection_of_M_and_N_l3347_334776

-- Define the sets M and N
def M : Set ℝ := {x | x^2 - 2*x < 0}
def N : Set ℝ := {x | -1 ≤ x ∧ x ≤ 1}

-- State the theorem
theorem intersection_of_M_and_N :
  M ∩ N = {x | 0 < x ∧ x ≤ 1} := by sorry

end intersection_of_M_and_N_l3347_334776


namespace sum_of_reciprocals_of_roots_l3347_334787

theorem sum_of_reciprocals_of_roots (a b c d : ℝ) (z₁ z₂ z₃ z₄ : ℂ) : 
  z₁^4 + a*z₁^3 + b*z₁^2 + c*z₁ + d = 0 ∧
  z₂^4 + a*z₂^3 + b*z₂^2 + c*z₂ + d = 0 ∧
  z₃^4 + a*z₃^3 + b*z₃^2 + c*z₃ + d = 0 ∧
  z₄^4 + a*z₄^3 + b*z₄^2 + c*z₄ + d = 0 ∧
  Complex.abs z₁ = 1 ∧ Complex.abs z₂ = 1 ∧ Complex.abs z₃ = 1 ∧ Complex.abs z₄ = 1 →
  1/z₁ + 1/z₂ + 1/z₃ + 1/z₄ = -a := by
sorry

end sum_of_reciprocals_of_roots_l3347_334787


namespace rationalize_denominator_l3347_334750

theorem rationalize_denominator :
  (1 : ℝ) / (Real.rpow 3 (1/3) + Real.rpow 27 (1/3)) = 
  (Real.rpow 9 (1/3)) / (3 * (Real.rpow 9 (1/3) + 1)) := by sorry

end rationalize_denominator_l3347_334750


namespace raffle_tickets_sold_l3347_334762

/-- Given that a school sold $620 worth of raffle tickets at $4 per ticket,
    prove that the number of tickets sold is 155. -/
theorem raffle_tickets_sold (total_money : ℕ) (ticket_cost : ℕ) (num_tickets : ℕ)
  (h1 : total_money = 620)
  (h2 : ticket_cost = 4)
  (h3 : total_money = ticket_cost * num_tickets) :
  num_tickets = 155 := by
  sorry

end raffle_tickets_sold_l3347_334762


namespace vacant_seats_l3347_334754

theorem vacant_seats (total_seats : ℕ) (filled_percentage : ℚ) (h1 : total_seats = 600) (h2 : filled_percentage = 1/2) :
  (total_seats : ℚ) * (1 - filled_percentage) = 300 := by
  sorry

end vacant_seats_l3347_334754


namespace student_average_age_l3347_334713

/-- Given a class of students and a staff member, if including the staff's age
    increases the average age by 1 year, then we can determine the average age of the students. -/
theorem student_average_age
  (num_students : ℕ)
  (staff_age : ℕ)
  (avg_increase : ℝ)
  (h1 : num_students = 32)
  (h2 : staff_age = 49)
  (h3 : avg_increase = 1) :
  (num_students * (staff_age - num_students - 1 : ℝ)) / num_students = 16 := by
  sorry

end student_average_age_l3347_334713


namespace max_contribution_l3347_334781

theorem max_contribution (n : ℕ) (total : ℝ) (min_contrib : ℝ) (h1 : n = 15) (h2 : total = 30) (h3 : min_contrib = 1) :
  let max_single := total - (n - 1) * min_contrib
  max_single = 16 := by
sorry

end max_contribution_l3347_334781


namespace focal_radius_circle_tangent_y_axis_l3347_334701

/-- Represents a parabola y^2 = 2px where p > 0 -/
structure Parabola where
  p : ℝ
  h_p_pos : p > 0

/-- Represents a circle with diameter equal to the focal radius of a parabola -/
structure FocalRadiusCircle (para : Parabola) where
  center : ℝ × ℝ
  radius : ℝ

/-- The circle with diameter equal to the focal radius of the parabola y^2 = 2px (p > 0) is tangent to the y-axis -/
theorem focal_radius_circle_tangent_y_axis (para : Parabola) :
  ∃ (c : FocalRadiusCircle para), c.center.1 = c.radius := by
  sorry

end focal_radius_circle_tangent_y_axis_l3347_334701


namespace jungkook_has_biggest_number_l3347_334724

def jungkook_number : ℕ := 6 * 3
def yoongi_number : ℕ := 4
def yuna_number : ℕ := 5

theorem jungkook_has_biggest_number :
  jungkook_number > yoongi_number ∧ jungkook_number > yuna_number := by
  sorry

end jungkook_has_biggest_number_l3347_334724


namespace product_identity_l3347_334738

theorem product_identity (x y : ℝ) : (x + y^2) * (x - y^2) * (x^2 + y^4) = x^4 - y^8 := by
  sorry

end product_identity_l3347_334738


namespace pencil_count_l3347_334702

theorem pencil_count (initial_pencils additional_pencils : ℕ) 
  (h1 : initial_pencils = 27)
  (h2 : additional_pencils = 45) :
  initial_pencils + additional_pencils = 72 :=
by sorry

end pencil_count_l3347_334702


namespace expected_sides_theorem_expected_sides_rectangle_limit_l3347_334743

/-- The expected number of sides of a randomly selected polygon after cuts -/
def expected_sides (n k : ℕ) : ℚ :=
  (n + 4 * k) / (k + 1)

/-- Theorem: The expected number of sides of a randomly selected polygon
    after k cuts, starting with an n-sided polygon, is (n + 4k) / (k + 1) -/
theorem expected_sides_theorem (n k : ℕ) :
  expected_sides n k = (n + 4 * k) / (k + 1) := by
  sorry

/-- Corollary: For a rectangle (n = 4) and large k, the expectation approaches 4 -/
theorem expected_sides_rectangle_limit :
  ∀ ε > 0, ∃ K : ℕ, ∀ k ≥ K, |expected_sides 4 k - 4| < ε := by
  sorry

end expected_sides_theorem_expected_sides_rectangle_limit_l3347_334743


namespace ceiling_negative_fraction_squared_l3347_334798

theorem ceiling_negative_fraction_squared : ⌈(-7/4)^2⌉ = 4 := by
  sorry

end ceiling_negative_fraction_squared_l3347_334798


namespace inverse_proportion_l3347_334706

/-- Given that y is inversely proportional to x and when x = 2, y = -3,
    this theorem proves the relationship between y and x, and the value of x when y = 2. -/
theorem inverse_proportion (x y : ℝ) : 
  (∃ k : ℝ, ∀ x ≠ 0, y = k / x) →  -- y is inversely proportional to x
  (2 : ℝ) * (-3 : ℝ) = y * x →     -- when x = 2, y = -3
  y = -6 / x ∧                     -- the function relationship
  (y = 2 → x = -3)                 -- when y = 2, x = -3
  := by sorry

end inverse_proportion_l3347_334706


namespace odd_selections_from_eleven_l3347_334731

theorem odd_selections_from_eleven (n : ℕ) (h : n = 11) :
  (Finset.range n).sum (fun k => if k % 2 = 1 then Nat.choose n k else 0) = 2^(n-1) := by
  sorry

end odd_selections_from_eleven_l3347_334731


namespace forty_percent_value_l3347_334756

theorem forty_percent_value (x : ℝ) : (0.6 * x = 240) → (0.4 * x = 160) := by
  sorry

end forty_percent_value_l3347_334756


namespace intersection_equality_range_l3347_334753

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | a - 1 ≤ x ∧ x ≤ 2 * a + 3}
def B : Set ℝ := {x | -2 ≤ x ∧ x ≤ 4}

-- Define the universal set U
def U : Set ℝ := Set.univ

-- Theorem statement
theorem intersection_equality_range (a : ℝ) :
  A a ∩ B = A a ↔ a ∈ Set.Iic (-4) ∪ Set.Icc (-1) (1/2) := by
  sorry

end intersection_equality_range_l3347_334753


namespace age_difference_l3347_334751

/-- Given Billy's current age and the ratio of my age to Billy's, 
    prove the difference between our ages. -/
theorem age_difference (billy_age : ℕ) (age_ratio : ℕ) : 
  billy_age = 4 → age_ratio = 4 → age_ratio * billy_age - billy_age = 12 := by
  sorry

end age_difference_l3347_334751


namespace log_inequality_l3347_334722

theorem log_inequality (a b c : ℝ) : 
  a = Real.log (2/3) → b = Real.log (2/5) → c = Real.log (3/2) → c > a ∧ a > b :=
by sorry

end log_inequality_l3347_334722


namespace infinite_geometric_series_ratio_l3347_334796

/-- For an infinite geometric series with first term a and sum S, 
    the common ratio r can be calculated. -/
theorem infinite_geometric_series_ratio 
  (a : ℝ) (S : ℝ) (h1 : a = 400) (h2 : S = 2500) :
  let r := 1 - a / S
  r = 21 / 25 := by
sorry

end infinite_geometric_series_ratio_l3347_334796


namespace school_female_students_l3347_334780

theorem school_female_students 
  (total_students : ℕ) 
  (sample_size : ℕ) 
  (sample_difference : ℕ) : 
  total_students = 1600 → 
  sample_size = 200 → 
  sample_difference = 10 →
  (∃ (female_students : ℕ), 
    female_students = 760 ∧ 
    female_students + (total_students - female_students) = total_students ∧
    (female_students : ℚ) / (total_students - female_students) = 
      ((sample_size / 2 - sample_difference / 2) : ℚ) / (sample_size / 2 + sample_difference / 2)) :=
by sorry

end school_female_students_l3347_334780


namespace masud_siblings_count_l3347_334766

theorem masud_siblings_count :
  ∀ (janet_siblings masud_siblings carlos_siblings : ℕ),
    janet_siblings = 4 * masud_siblings - 60 →
    carlos_siblings = 3 * masud_siblings / 4 →
    janet_siblings = carlos_siblings + 135 →
    masud_siblings = 60 := by
  sorry

end masud_siblings_count_l3347_334766


namespace sequence_inequality_l3347_334793

theorem sequence_inequality (a : ℕ → ℝ) 
  (h1 : a 1 = π / 3)
  (h2 : ∀ n, 0 < a n ∧ a n < π / 3)
  (h3 : ∀ n ≥ 2, Real.sin (a (n + 1)) ≤ (1 / 3) * Real.sin (3 * a n)) :
  ∀ n, Real.sin (a n) < 1 / Real.sqrt n := by
sorry

end sequence_inequality_l3347_334793


namespace min_reciprocal_sum_min_reciprocal_sum_attained_l3347_334744

theorem min_reciprocal_sum (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0)
  (h4 : x + y + z = 3) (h5 : y = 2 * x) :
  (1 / x + 1 / y + 1 / z) ≥ 10 / 3 := by
  sorry

theorem min_reciprocal_sum_attained (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0)
  (h4 : x + y + z = 3) (h5 : y = 2 * x) :
  ∃ (x₀ y₀ z₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ z₀ > 0 ∧ x₀ + y₀ + z₀ = 3 ∧ y₀ = 2 * x₀ ∧
  (1 / x₀ + 1 / y₀ + 1 / z₀) = 10 / 3 := by
  sorry

end min_reciprocal_sum_min_reciprocal_sum_attained_l3347_334744


namespace bus_capacity_is_180_l3347_334771

/-- Represents the seating capacity of a double-decker bus -/
def double_decker_bus_capacity : ℕ :=
  let lower_left := 15 * 3
  let lower_right := 12 * 3
  let lower_back := 9
  let upper_left := 20 * 2
  let upper_right := 20 * 2
  let jump_seats := 4 * 1
  let emergency := 6
  lower_left + lower_right + lower_back + upper_left + upper_right + jump_seats + emergency

/-- Theorem stating the total seating capacity of the double-decker bus -/
theorem bus_capacity_is_180 : double_decker_bus_capacity = 180 := by
  sorry

#eval double_decker_bus_capacity

end bus_capacity_is_180_l3347_334771


namespace water_bottles_taken_out_l3347_334735

theorem water_bottles_taken_out (red : ℕ) (black : ℕ) (blue : ℕ) (remaining : ℕ) :
  red = 2 → black = 3 → blue = 4 → remaining = 4 →
  red + black + blue - remaining = 5 :=
by sorry

end water_bottles_taken_out_l3347_334735


namespace smallest_sum_B_plus_c_l3347_334772

theorem smallest_sum_B_plus_c : ∃ (B c : ℕ),
  (B < 5) ∧                        -- B is a digit in base 5
  (c > 7) ∧                        -- c is a base greater than 7
  (31 * B = 4 * c + 4) ∧           -- BBB_5 = 44_c
  (∀ (B' c' : ℕ),                  -- For all other valid B' and c'
    (B' < 5) →
    (c' > 7) →
    (31 * B' = 4 * c' + 4) →
    (B + c ≤ B' + c')) ∧
  (B + c = 25)                     -- The smallest sum is 25
  := by sorry

end smallest_sum_B_plus_c_l3347_334772


namespace odd_function_has_zero_l3347_334730

-- Define an odd function
def OddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

-- Theorem statement
theorem odd_function_has_zero (f : ℝ → ℝ) (h : OddFunction f) : 
  ∃ x : ℝ, f x = 0 := by
  sorry

end odd_function_has_zero_l3347_334730


namespace skating_time_calculation_l3347_334712

theorem skating_time_calculation (distance : ℝ) (speed : ℝ) (time : ℝ) :
  distance = 150 →
  speed = 12 →
  time = distance / speed →
  time = 12.5 :=
by sorry

end skating_time_calculation_l3347_334712
