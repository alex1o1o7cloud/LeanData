import Mathlib

namespace NUMINAMATH_GPT_roof_length_width_difference_l1076_107669

theorem roof_length_width_difference
  {w l : ℝ} 
  (h_area : l * w = 576) 
  (h_length : l = 4 * w) 
  (hw_pos : w > 0) :
  l - w = 36 :=
by 
  sorry

end NUMINAMATH_GPT_roof_length_width_difference_l1076_107669


namespace NUMINAMATH_GPT_smallest_term_index_l1076_107683

theorem smallest_term_index (a_n : ℕ → ℤ) (h : ∀ n, a_n n = 3 * n^2 - 38 * n + 12) : ∃ n, a_n n = a_n 6 ∧ ∀ m, a_n m ≥ a_n 6 :=
by
  sorry

end NUMINAMATH_GPT_smallest_term_index_l1076_107683


namespace NUMINAMATH_GPT_lowest_score_on_one_of_last_two_tests_l1076_107632

-- define conditions
variables (score1 score2 : ℕ) (total_score average desired_score : ℕ)

-- Shauna's scores on the first two tests are 82 and 75
def shauna_score1 := 82
def shauna_score2 := 75

-- Shauna wants to average 85 over 4 tests
def desired_average := 85
def number_of_tests := 4

-- total points needed for desired average
def total_points_needed := desired_average * number_of_tests

-- total points from first two tests
def total_first_two_tests := shauna_score1 + shauna_score2

-- total points needed on last two tests
def points_needed_last_two_tests := total_points_needed - total_first_two_tests

-- Prove the lowest score on one of the last two tests
theorem lowest_score_on_one_of_last_two_tests : 
  (∃ (score3 score4 : ℕ), score3 + score4 = points_needed_last_two_tests ∧ score3 ≤ 100 ∧ score4 ≤ 100 ∧ (score3 ≥ 83 ∨ score4 ≥ 83)) :=
sorry

end NUMINAMATH_GPT_lowest_score_on_one_of_last_two_tests_l1076_107632


namespace NUMINAMATH_GPT_find_prices_max_basketballs_l1076_107682

-- Definition of given conditions
def conditions1 (x y : ℝ) : Prop := 
  (x - y = 50) ∧ (6 * x + 8 * y = 1700)

-- Definitions of questions:
-- Question 1: Find the price of one basketball and one soccer ball
theorem find_prices (x y : ℝ) (h: conditions1 x y) : x = 150 ∧ y = 100 := sorry

-- Definition of given conditions for Question 2
def conditions2 (x y : ℝ) (a : ℕ) : Prop :=
  (x = 150) ∧ (y = 100) ∧ 
  (0.9 * x * a + 0.85 * y * (10 - a) ≤ 1150)

-- Question 2: The school plans to purchase 10 items with given discounts
theorem max_basketballs (x y : ℝ) (a : ℕ) (h1: x = 150) (h2: y = 100) (h3: a ≤ 10) (h4: conditions2 x y a) : a ≤ 6 := sorry

end NUMINAMATH_GPT_find_prices_max_basketballs_l1076_107682


namespace NUMINAMATH_GPT_intersection_empty_implies_m_leq_neg1_l1076_107634

theorem intersection_empty_implies_m_leq_neg1 (m : ℝ) :
  (∀ (x y: ℝ), (x < m) → (y = x^2 + 2*x) → y < -1) →
  m ≤ -1 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_intersection_empty_implies_m_leq_neg1_l1076_107634


namespace NUMINAMATH_GPT_m_lt_n_l1076_107689

theorem m_lt_n (a t : ℝ) (h : 0 < t ∧ t < 1) : 
  abs (Real.log (1 + t) / Real.log a) < abs (Real.log (1 - t) / Real.log a) :=
sorry

end NUMINAMATH_GPT_m_lt_n_l1076_107689


namespace NUMINAMATH_GPT_discount_calculation_l1076_107649

-- Definitions based on the given conditions
def cost_magazine : Float := 0.85
def cost_pencil : Float := 0.50
def amount_spent : Float := 1.00

-- Define the total cost before discount
def total_cost_before_discount : Float := cost_magazine + cost_pencil

-- Goal: Prove that the discount is $0.35
theorem discount_calculation : total_cost_before_discount - amount_spent = 0.35 := by
  -- Proof (to be filled in later)
  sorry

end NUMINAMATH_GPT_discount_calculation_l1076_107649


namespace NUMINAMATH_GPT_calc_expression_l1076_107696

theorem calc_expression :
  (2014 * 2014 + 2012) - 2013 * 2013 = 6039 :=
by
  -- Let 2014 = 2013 + 1 and 2012 = 2013 - 1
  have h2014 : 2014 = 2013 + 1 := by sorry
  have h2012 : 2012 = 2013 - 1 := by sorry
  -- Start the main proof
  sorry

end NUMINAMATH_GPT_calc_expression_l1076_107696


namespace NUMINAMATH_GPT_a2_a4_a6_a8_a10_a12_sum_l1076_107692

theorem a2_a4_a6_a8_a10_a12_sum :
  ∀ (x : ℝ), 
    (1 + x + x^2)^6 = a + a1 * x + a2 * x^2 + a3 * x^3 + a4 * x^4 + a5 * x^5 + a6 * x^6 + a7 * x^7 + a8 * x^8 + a9 * x^9 + a10 * x^10 + a11 * x^11 + a12 * x^12 →
    a2 + a4 + a6 + a8 + a10 + a12 = 364 :=
sorry

end NUMINAMATH_GPT_a2_a4_a6_a8_a10_a12_sum_l1076_107692


namespace NUMINAMATH_GPT_total_amount_spent_correct_l1076_107670

-- Definitions based on conditions
def price_of_food_before_tax_and_tip : ℝ := 140
def sales_tax_rate : ℝ := 0.10
def tip_rate : ℝ := 0.20

-- Definitions of intermediate steps
def sales_tax : ℝ := sales_tax_rate * price_of_food_before_tax_and_tip
def total_before_tip : ℝ := price_of_food_before_tax_and_tip + sales_tax
def tip : ℝ := tip_rate * total_before_tip
def total_amount_spent : ℝ := total_before_tip + tip

-- Theorem statement to be proved
theorem total_amount_spent_correct : total_amount_spent = 184.80 :=
by
  sorry -- Proof is skipped

end NUMINAMATH_GPT_total_amount_spent_correct_l1076_107670


namespace NUMINAMATH_GPT_solve_for_A_l1076_107663

variable (x y : ℝ)

theorem solve_for_A (A : ℝ) : (2 * x - y) ^ 2 + A = (2 * x + y) ^ 2 → A = 8 * x * y :=
by
  intro h
  sorry

end NUMINAMATH_GPT_solve_for_A_l1076_107663


namespace NUMINAMATH_GPT_work_completion_l1076_107627

theorem work_completion (A B C D : ℝ) :
  (A = 1 / 5) →
  (A + C = 2 / 5) →
  (B + C = 1 / 4) →
  (A + D = 1 / 3.6) →
  (B + C + D = 1 / 2) →
  B = 1 / 20 :=
by
  sorry

end NUMINAMATH_GPT_work_completion_l1076_107627


namespace NUMINAMATH_GPT_max_discount_l1076_107693

-- Definitions:
def cost_price : ℝ := 400
def sale_price : ℝ := 600
def desired_profit_margin : ℝ := 0.05

-- Statement:
theorem max_discount 
  (x : ℝ) 
  (hx : sale_price * (1 - x / 100) ≥ cost_price * (1 + desired_profit_margin)) :
  x ≤ 90 := 
sorry

end NUMINAMATH_GPT_max_discount_l1076_107693


namespace NUMINAMATH_GPT_cathy_remaining_money_l1076_107618

noncomputable def remaining_money (initial : ℝ) (dad : ℝ) (book : ℝ) (cab_percentage : ℝ) (food_percentage : ℝ) : ℝ :=
  let money_mom := 2 * dad
  let total_money := initial + dad + money_mom
  let remaining_after_book := total_money - book
  let cab_cost := cab_percentage * remaining_after_book
  let food_budget := food_percentage * total_money
  let dinner_cost := 0.5 * food_budget
  remaining_after_book - cab_cost - dinner_cost

theorem cathy_remaining_money :
  remaining_money 12 25 15 0.03 0.4 = 52.44 :=
by
  sorry

end NUMINAMATH_GPT_cathy_remaining_money_l1076_107618


namespace NUMINAMATH_GPT_alex_distribution_ways_l1076_107687

theorem alex_distribution_ways : (15^5 = 759375) := by {
  sorry
}

end NUMINAMATH_GPT_alex_distribution_ways_l1076_107687


namespace NUMINAMATH_GPT_two_integer_solutions_iff_l1076_107684

theorem two_integer_solutions_iff (a : ℝ) :
  (∃ (n m : ℤ), n ≠ m ∧ |n - 1| < a * n ∧ |m - 1| < a * m ∧
    ∀ (k : ℤ), |k - 1| < a * k → k = n ∨ k = m) ↔
  (1/2 : ℝ) < a ∧ a ≤ (2/3 : ℝ) :=
by
  sorry

end NUMINAMATH_GPT_two_integer_solutions_iff_l1076_107684


namespace NUMINAMATH_GPT_rhombus_perimeter_is_80_l1076_107612

-- Definitions of the conditions
def rhombus_diagonals_ratio : Prop := ∃ (d1 d2 : ℝ), d1 / d2 = 3 / 4 ∧ d1 + d2 = 56

-- The goal is to prove that given the conditions, the perimeter of the rhombus is 80
theorem rhombus_perimeter_is_80 (h : rhombus_diagonals_ratio) : ∃ (p : ℝ), p = 80 :=
by
  sorry  -- The actual proof steps would go here

end NUMINAMATH_GPT_rhombus_perimeter_is_80_l1076_107612


namespace NUMINAMATH_GPT_jennifer_money_left_l1076_107625

def money_left (initial_amount sandwich_fraction museum_fraction book_fraction : ℚ) : ℚ :=
  initial_amount - (initial_amount * sandwich_fraction + initial_amount * museum_fraction + initial_amount * book_fraction)

theorem jennifer_money_left :
  money_left 150 (1/5) (1/6) (1/2) = 20 := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_jennifer_money_left_l1076_107625


namespace NUMINAMATH_GPT_problem1_l1076_107695

def f (x : ℝ) := (1 - 3 * x) * (1 + x) ^ 5

theorem problem1 :
  let a : ℝ := f (1 / 3)
  a = 0 :=
by
  let a := f (1 / 3)
  sorry

end NUMINAMATH_GPT_problem1_l1076_107695


namespace NUMINAMATH_GPT_Ursula_hot_dogs_l1076_107679

theorem Ursula_hot_dogs 
  (H : ℕ) 
  (cost_hot_dog : ℚ := 1.50) 
  (cost_salad : ℚ := 2.50) 
  (num_salads : ℕ := 3) 
  (total_money : ℚ := 20) 
  (change : ℚ := 5) :
  (cost_hot_dog * H + cost_salad * num_salads = total_money - change) → H = 5 :=
by
  sorry

end NUMINAMATH_GPT_Ursula_hot_dogs_l1076_107679


namespace NUMINAMATH_GPT_bound_on_f_l1076_107615

theorem bound_on_f 
  (f : ℝ → ℝ) 
  (h_domain : ∀ x, 0 ≤ x ∧ x ≤ 1) 
  (h_zeros : f 0 = 0 ∧ f 1 = 0)
  (h_condition : ∀ x1 x2, 0 ≤ x1 ∧ x1 ≤ 1 ∧ 0 ≤ x2 ∧ x2 ≤ 1 ∧ x1 ≠ x2 → |f x2 - f x1| < |x2 - x1|) 
  : ∀ x1 x2, 0 ≤ x1 ∧ x1 ≤ 1 ∧ 0 ≤ x2 ∧ x2 ≤ 1 → |f x2 - f x1| < 1/2 :=
by
  sorry

end NUMINAMATH_GPT_bound_on_f_l1076_107615


namespace NUMINAMATH_GPT_book_club_boys_count_l1076_107633

theorem book_club_boys_count (B G : ℕ) 
  (h1 : B + G = 30) 
  (h2 : B + (1 / 3 : ℝ) * G = 18) :
  B = 12 :=
by
  have h3 : 3 • B + G = 54 := sorry
  have h4 : 3 • B + G - (B + G) = 54 - 30 := sorry
  have h5 : 2 • B = 24 := sorry
  have h6 : B = 12 := sorry
  exact h6

end NUMINAMATH_GPT_book_club_boys_count_l1076_107633


namespace NUMINAMATH_GPT_option_d_correct_l1076_107605

def f (x : ℝ) : ℝ := (x - 1) * (x - 2) * (x - 3)
def M : Set ℝ := {x | f x = 0}

theorem option_d_correct : ({1, 3} ∪ {2, 3} : Set ℝ) = M := by
  sorry

end NUMINAMATH_GPT_option_d_correct_l1076_107605


namespace NUMINAMATH_GPT_triangles_sticks_not_proportional_l1076_107691

theorem triangles_sticks_not_proportional :
  ∀ (n_triangles n_sticks : ℕ), 
  (∃ k : ℕ, n_triangles = k * n_sticks) 
  ∨ 
  (∃ k : ℕ, n_triangles * n_sticks = k) 
  → False :=
by
  sorry

end NUMINAMATH_GPT_triangles_sticks_not_proportional_l1076_107691


namespace NUMINAMATH_GPT_cos_585_eq_neg_sqrt2_div_2_l1076_107674

theorem cos_585_eq_neg_sqrt2_div_2 :
  Real.cos (585 * Real.pi / 180) = -Real.sqrt 2 / 2 :=
by
  sorry

end NUMINAMATH_GPT_cos_585_eq_neg_sqrt2_div_2_l1076_107674


namespace NUMINAMATH_GPT_sum_of_inscribed_angles_l1076_107613

-- Define the circle and its division into arcs.
def circle_division (O : Type) (total_arcs : ℕ) := total_arcs = 16

-- Define the inscribed angles x and y.
def inscribed_angle (O : Type) (arc_subtended : ℕ) := arc_subtended

-- Define the conditions for angles x and y subtending 3 and 5 arcs respectively.
def angle_x := inscribed_angle ℝ 3
def angle_y := inscribed_angle ℝ 5

-- Theorem stating the sum of the inscribed angles x and y.
theorem sum_of_inscribed_angles 
  (O : Type)
  (total_arcs : ℕ)
  (h1 : circle_division O total_arcs)
  (h2 : inscribed_angle O angle_x = 3)
  (h3 : inscribed_angle O angle_y = 5) :
  33.75 + 56.25 = 90 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_inscribed_angles_l1076_107613


namespace NUMINAMATH_GPT_square_side_length_false_l1076_107643

theorem square_side_length_false (perimeter : ℝ) (side_length : ℝ) (h1 : perimeter = 8) (h2 : side_length = 4) :
  ¬(4 * side_length = perimeter) :=
by
  sorry

end NUMINAMATH_GPT_square_side_length_false_l1076_107643


namespace NUMINAMATH_GPT_pieces_eaten_first_l1076_107659

variable (initial_candy : ℕ) (remaining_candy : ℕ) (candy_eaten_second : ℕ)

theorem pieces_eaten_first 
    (initial_candy := 21) 
    (remaining_candy := 7)
    (candy_eaten_second := 9) :
    (initial_candy - remaining_candy - candy_eaten_second = 5) :=
sorry

end NUMINAMATH_GPT_pieces_eaten_first_l1076_107659


namespace NUMINAMATH_GPT_scheduling_arrangements_correct_l1076_107603

-- Define the set of employees
inductive Employee
| A | B | C | D | E | F deriving DecidableEq

open Employee

-- Define the days of the festival
inductive Day
| May31 | June1 | June2 deriving DecidableEq

open Day

def canWork (e : Employee) (d : Day) : Prop :=
match e, d with
| A, May31 => False
| B, June2 => False
| _, _ => True

def schedulingArrangements : ℕ :=
  -- Calculations go here, placeholder for now
  sorry

theorem scheduling_arrangements_correct : schedulingArrangements = 42 := 
  sorry

end NUMINAMATH_GPT_scheduling_arrangements_correct_l1076_107603


namespace NUMINAMATH_GPT_triangle_is_isosceles_l1076_107637

variable (A B C a b c : ℝ)
variable (sin : ℝ → ℝ)

theorem triangle_is_isosceles (h1 : a * sin A - b * sin B = 0) :
  a = b :=
by
  sorry

end NUMINAMATH_GPT_triangle_is_isosceles_l1076_107637


namespace NUMINAMATH_GPT_fraction_simplification_l1076_107688

theorem fraction_simplification (x : ℚ) : 
  (3 / 4) * 60 - x * 60 + 63 = 12 → 
  x = (8 / 5) :=
by
  sorry

end NUMINAMATH_GPT_fraction_simplification_l1076_107688


namespace NUMINAMATH_GPT_maximum_matches_l1076_107600

theorem maximum_matches (A B C : ℕ) (h1 : A > B) (h2 : B > C) 
    (h3 : A ≥ B + 10) (h4 : B ≥ C + 10) (h5 : B + C > A) : 
    A + B + C - 1 ≤ 62 :=
sorry

end NUMINAMATH_GPT_maximum_matches_l1076_107600


namespace NUMINAMATH_GPT_largest_angle_bounds_triangle_angles_l1076_107639

theorem largest_angle_bounds (A B C : ℝ) (angle_A angle_B angle_C : ℝ)
  (h_triangle : angle_A + angle_B + angle_C = 180)
  (h_tangent : angle_B + 2 * angle_C = 90) :
  90 ≤ angle_A ∧ angle_A < 135 :=
sorry

theorem triangle_angles (A B C : ℝ) (angle_A angle_B angle_C : ℝ)
  (h_triangle : angle_A + angle_B + angle_C = 180)
  (h_tangent_B : angle_B + 2 * angle_C = 90)
  (h_tangent_C : angle_C + 2 * angle_B = 90) :
  angle_A = 120 ∧ angle_B = 30 ∧ angle_C = 30 :=
sorry

end NUMINAMATH_GPT_largest_angle_bounds_triangle_angles_l1076_107639


namespace NUMINAMATH_GPT_contrapositive_example_l1076_107624

theorem contrapositive_example (x : ℝ) :
  (x < -1 ∨ x ≥ 1) → (x^2 ≥ 1) :=
sorry

end NUMINAMATH_GPT_contrapositive_example_l1076_107624


namespace NUMINAMATH_GPT_petya_friends_l1076_107694

variable (friends stickers : Nat)

-- Condition where giving 5 stickers to each friend leaves Petya with 8 stickers.
def condition1 := stickers - friends * 5 = 8

-- Condition where giving 6 stickers to each friend makes Petya short of 11 stickers.
def condition2 := stickers = friends * 6 - 11

-- The theorem that states Petya has 19 friends given the above conditions
theorem petya_friends : ∀ {friends stickers : Nat}, 
  (stickers - friends * 5 = 8) →
  (stickers = friends * 6 - 11) →
  friends = 19 := 
by
  intros friends stickers cond1 cond2
  have proof : friends = 19 := sorry
  exact proof

end NUMINAMATH_GPT_petya_friends_l1076_107694


namespace NUMINAMATH_GPT_incorrect_calculation_l1076_107658

theorem incorrect_calculation : ¬ (3 + 2 * Real.sqrt 2 = 5 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_GPT_incorrect_calculation_l1076_107658


namespace NUMINAMATH_GPT_chocolate_distribution_l1076_107636

theorem chocolate_distribution :
  let total_chocolate := 60 / 7
  let piles := 5
  let eaten_piles := 1
  let friends := 2
  let one_pile := total_chocolate / piles
  let remaining_chocolate := total_chocolate - eaten_piles * one_pile
  let chocolate_per_friend := remaining_chocolate / friends
  chocolate_per_friend = 24 / 7 :=
by
  sorry

end NUMINAMATH_GPT_chocolate_distribution_l1076_107636


namespace NUMINAMATH_GPT_find_x4_l1076_107653

open Real

theorem find_x4 (x : ℝ) (h₁ : 0 < x) (h₂ : sqrt (1 - x^2) + sqrt (1 + x^2) = 2) : x^4 = 0 :=
by
  sorry

end NUMINAMATH_GPT_find_x4_l1076_107653


namespace NUMINAMATH_GPT_square_areas_l1076_107642

theorem square_areas (z : ℂ) 
  (h1 : ¬ (2 : ℂ) * z^2 = z)
  (h2 : ¬ (3 : ℂ) * z^3 = z)
  (sz : (3 * z^3 - z) = (I * (2 * z^2 - z)) ∨ (3 * z^3 - z) = (-I * (2 * z^2 - z))) :
  ∃ (areas : Finset ℝ), areas = {85, 4500} :=
by {
  sorry
}

end NUMINAMATH_GPT_square_areas_l1076_107642


namespace NUMINAMATH_GPT_tenth_equation_sum_of_cubes_l1076_107630

theorem tenth_equation_sum_of_cubes :
  (1^3 + 2^3 + 3^3 + 4^3 + 5^3 + 6^3 + 7^3 + 8^3 + 9^3 + 10^3) = 55^2 := 
by sorry

end NUMINAMATH_GPT_tenth_equation_sum_of_cubes_l1076_107630


namespace NUMINAMATH_GPT_find_a_l1076_107635

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if h : x < 0 then a ^ x - 1 else 2 * x ^ 2

theorem find_a (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : ∀ m n : ℝ, f a m ≤ f a n ↔ m ≤ n)
  (h4 : f a a = 5 * a - 2) : a = 2 :=
sorry

end NUMINAMATH_GPT_find_a_l1076_107635


namespace NUMINAMATH_GPT_no_solutions_sinx_eq_sin_sinx_l1076_107668

open Real

theorem no_solutions_sinx_eq_sin_sinx (x : ℝ) (h : 0 ≤ x ∧ x ≤ arcsin 0.9) : ¬ (sin x = sin (sin x)) :=
by
  sorry

end NUMINAMATH_GPT_no_solutions_sinx_eq_sin_sinx_l1076_107668


namespace NUMINAMATH_GPT_total_stickers_at_end_of_week_l1076_107672

-- Defining the initial and earned stickers as constants
def initial_stickers : ℕ := 39
def earned_stickers : ℕ := 22

-- Defining the goal as a proof statement
theorem total_stickers_at_end_of_week : initial_stickers + earned_stickers = 61 := 
by {
  sorry
}

end NUMINAMATH_GPT_total_stickers_at_end_of_week_l1076_107672


namespace NUMINAMATH_GPT_relation_between_m_and_n_l1076_107685

variable {A x y z a b c d e n m : ℝ}
variable {p r : ℝ}
variable (s : finset ℝ) (hset : s = {x, y, z, a, b, c, d, e})
variable (hsorted : x < y ∧ y < z ∧ z < a ∧ a < b ∧ b < c ∧ c < d ∧ d < e)
variable (hne : n ∉ s)
variable (hme : m ∉ s)

theorem relation_between_m_and_n 
  (h_avg_n : (s.sum + n) / 9 = (s.sum / 8) * (1 + p / 100)) 
  (h_avg_m : (s.sum + m) / 9 = (s.sum / 8) * (1 + r / 100)) 
  : m = n + 9 * (s.sum / 8) * (r / 100 - p / 100) :=
sorry

end NUMINAMATH_GPT_relation_between_m_and_n_l1076_107685


namespace NUMINAMATH_GPT_inequality_am_gm_l1076_107686

theorem inequality_am_gm 
  (a b c : ℝ) : a^2 + b^2 + c^2 ≥ a*b + a*c + b*c := 
by 
  sorry

end NUMINAMATH_GPT_inequality_am_gm_l1076_107686


namespace NUMINAMATH_GPT_video_game_cost_l1076_107675

theorem video_game_cost
  (weekly_allowance1 : ℕ)
  (weeks1 : ℕ)
  (weekly_allowance2 : ℕ)
  (weeks2 : ℕ)
  (money_spent_on_clothes_fraction : ℚ)
  (remaining_money : ℕ)
  (allowance1 : weekly_allowance1 = 5)
  (duration1 : weeks1 = 8)
  (allowance2 : weekly_allowance2 = 6)
  (duration2 : weeks2 = 6)
  (money_spent_fraction : money_spent_on_clothes_fraction = 1/2)
  (remaining_money_condition : remaining_money = 3) :
  (weekly_allowance1 * weeks1 + weekly_allowance2 * weeks2) * (1 - money_spent_on_clothes_fraction) - remaining_money = 35 :=
by
  rw [allowance1, duration1, allowance2, duration2, money_spent_fraction, remaining_money_condition]
  -- Calculation steps are omitted; they can be filled in here.
  exact sorry

end NUMINAMATH_GPT_video_game_cost_l1076_107675


namespace NUMINAMATH_GPT_solve_inequality_l1076_107697

theorem solve_inequality (x : ℝ) : |x - 1| + |x - 2| > 5 ↔ (x < -1 ∨ x > 4) :=
by
  sorry

end NUMINAMATH_GPT_solve_inequality_l1076_107697


namespace NUMINAMATH_GPT_inequality_1_plus_a_1_plus_b_1_plus_c_geq_1_minus_d_squared_l1076_107629

theorem inequality_1_plus_a_1_plus_b_1_plus_c_geq_1_minus_d_squared 
  (a b c : ℝ)
  (h_sum : a + b + c = 0)
  (d : ℝ) 
  (h_d : d = max (abs a) (max (abs b) (abs c))) : 
  abs ((1 + a) * (1 + b) * (1 + c)) ≥ 1 - d^2 :=
by 
  sorry

end NUMINAMATH_GPT_inequality_1_plus_a_1_plus_b_1_plus_c_geq_1_minus_d_squared_l1076_107629


namespace NUMINAMATH_GPT_gcd_of_ratios_l1076_107645

noncomputable def gcd_of_two_ratios (A B : ℕ) : ℕ :=
  if h : A % B = 0 then B else gcd B (A % B)

theorem gcd_of_ratios (A B : ℕ) (k : ℕ) (h1 : Nat.lcm A B = 180) (h2 : A = 2 * k) (h3 : B = 3 * k) : gcd_of_two_ratios A B = 30 :=
  by
    sorry

end NUMINAMATH_GPT_gcd_of_ratios_l1076_107645


namespace NUMINAMATH_GPT_total_distance_traveled_l1076_107620

noncomputable def travel_distance : ℝ :=
  1280 * Real.sqrt 2 + 640 * Real.sqrt (2 + Real.sqrt 2) + 640

theorem total_distance_traveled :
  let n := 8
  let r := 40
  let theta := 2 * Real.pi / n
  let d_2arcs := 2 * r * Real.sin (theta)
  let d_3arcs := r * (2 + Real.sqrt (2))
  let d_4arcs := 2 * r
  (8 * (4 * d_2arcs + 2 * d_3arcs + d_4arcs)) = travel_distance := by
  sorry

end NUMINAMATH_GPT_total_distance_traveled_l1076_107620


namespace NUMINAMATH_GPT_Harriett_total_money_l1076_107662

open Real

theorem Harriett_total_money :
    let quarters := 14 * 0.25
    let dimes := 7 * 0.10
    let nickels := 9 * 0.05
    let pennies := 13 * 0.01
    let half_dollars := 4 * 0.50
    quarters + dimes + nickels + pennies + half_dollars = 6.78 :=
by
    sorry

end NUMINAMATH_GPT_Harriett_total_money_l1076_107662


namespace NUMINAMATH_GPT_lines_perpendicular_iff_l1076_107661

/-- Given two lines y = k₁ x + l₁ and y = k₂ x + l₂, 
    which are not parallel to the coordinate axes,
    these lines are perpendicular if and only if k₁ * k₂ = -1. -/
theorem lines_perpendicular_iff 
  (k₁ k₂ l₁ l₂ : ℝ) (h1 : k₁ ≠ 0) (h2 : k₂ ≠ 0) :
  (∀ x, k₁ * x + l₁ = k₂ * x + l₂) <-> k₁ * k₂ = -1 :=
sorry

end NUMINAMATH_GPT_lines_perpendicular_iff_l1076_107661


namespace NUMINAMATH_GPT_max_eccentricity_of_ellipse_l1076_107660

theorem max_eccentricity_of_ellipse 
  (R_large : ℝ)
  (r_cylinder : ℝ)
  (R_small : ℝ)
  (D_centers : ℝ)
  (a : ℝ)
  (b : ℝ)
  (e : ℝ) :
  R_large = 1 → 
  r_cylinder = 1 → 
  R_small = 1/4 → 
  D_centers = 10/3 → 
  a = 5/3 → 
  b = 1 → 
  e = Real.sqrt (1 - (b / a) ^ 2) → 
  e = 4/5 := by 
  sorry

end NUMINAMATH_GPT_max_eccentricity_of_ellipse_l1076_107660


namespace NUMINAMATH_GPT_diameter_other_endpoint_l1076_107646

def center : ℝ × ℝ := (1, -2)
def endpoint1 : ℝ × ℝ := (4, 3)
def expected_endpoint2 : ℝ × ℝ := (7, -7)

theorem diameter_other_endpoint (c : ℝ × ℝ) (e1 e2 : ℝ × ℝ) (h₁ : c = center) (h₂ : e1 = endpoint1) : e2 = expected_endpoint2 :=
by
  sorry

end NUMINAMATH_GPT_diameter_other_endpoint_l1076_107646


namespace NUMINAMATH_GPT_checkered_fabric_cost_l1076_107607

variable (P : ℝ) (cost_per_yard : ℝ) (total_yards : ℕ)
variable (x : ℝ) (C : ℝ)

theorem checkered_fabric_cost :
  P = 45 ∧ cost_per_yard = 7.50 ∧ total_yards = 16 →
  C = cost_per_yard * (total_yards - x) →
  7.50 * (16 - x) = 45 →
  C = 75 :=
by
  intro h1 h2 h3
  sorry

end NUMINAMATH_GPT_checkered_fabric_cost_l1076_107607


namespace NUMINAMATH_GPT_exponent_identity_l1076_107647

theorem exponent_identity (m : ℕ) : 5 ^ m = 5 * (25 ^ 4) * (625 ^ 3) ↔ m = 21 := by
  sorry

end NUMINAMATH_GPT_exponent_identity_l1076_107647


namespace NUMINAMATH_GPT_interest_rate_simple_and_compound_l1076_107640

theorem interest_rate_simple_and_compound (P T: ℝ) (SI CI R: ℝ) 
  (simple_interest_eq: SI = (P * R * T) / 100)
  (compound_interest_eq: CI = P * ((1 + R / 100) ^ T - 1)) 
  (hP : P = 3000) (hT : T = 2) (hSI : SI = 300) (hCI : CI = 307.50) :
  R = 5 :=
by
  sorry

end NUMINAMATH_GPT_interest_rate_simple_and_compound_l1076_107640


namespace NUMINAMATH_GPT_Ramsey_number_bound_l1076_107619

noncomputable def Ramsey_number (k : ℕ) : ℕ := sorry

theorem Ramsey_number_bound (k : ℕ) (h : k ≥ 3) : Ramsey_number k > 2^(k / 2) := sorry

end NUMINAMATH_GPT_Ramsey_number_bound_l1076_107619


namespace NUMINAMATH_GPT_solve_eq1_solve_eq2_l1076_107671

-- Prove the solution of the first equation
theorem solve_eq1 (x : ℝ) : 3 * x - (x - 1) = 7 ↔ x = 3 :=
by
  sorry

-- Prove the solution of the second equation
theorem solve_eq2 (x : ℝ) : (2 * x - 1) / 3 - (x - 3) / 6 = 1 ↔ x = (5 : ℝ) / 3 :=
by
  sorry

end NUMINAMATH_GPT_solve_eq1_solve_eq2_l1076_107671


namespace NUMINAMATH_GPT_cos_equiv_l1076_107690

theorem cos_equiv (n : ℤ) (hn : 0 ≤ n ∧ n ≤ 180) (hcos : Real.cos (n * Real.pi / 180) = Real.cos (1018 * Real.pi / 180)) : n = 62 := 
sorry

end NUMINAMATH_GPT_cos_equiv_l1076_107690


namespace NUMINAMATH_GPT_circle_tangent_to_x_axis_l1076_107657

theorem circle_tangent_to_x_axis (b : ℝ) :
  (∃ c : ℝ, ∀ x y : ℝ,
    (x^2 + y^2 + 4 * x + 2 * b * y + c = 0) ∧ (∃ r : ℝ, r > 0 ∧ ∀ y : ℝ, y = -b ↔ y = 2)) ↔ (b = 2 ∨ b = -2) :=
sorry

end NUMINAMATH_GPT_circle_tangent_to_x_axis_l1076_107657


namespace NUMINAMATH_GPT_lcm_1540_2310_l1076_107622

theorem lcm_1540_2310 : Nat.lcm 1540 2310 = 4620 :=
by sorry

end NUMINAMATH_GPT_lcm_1540_2310_l1076_107622


namespace NUMINAMATH_GPT_find_divisor_l1076_107616

theorem find_divisor (d : ℕ) (h : 127 = d * 5 + 2) : d = 25 :=
by 
  -- Given conditions
  -- 127 = d * 5 + 2
  -- We need to prove d = 25
  sorry

end NUMINAMATH_GPT_find_divisor_l1076_107616


namespace NUMINAMATH_GPT_flyDistanceCeiling_l1076_107648

variable (P : ℝ × ℝ × ℝ)
variable (x : ℝ)
variable (y : ℝ)
variable (z : ℝ)

-- Defining the conditions
def isAtRightAngles (P : ℝ × ℝ × ℝ) : Prop :=
  P = (0, 0, 0)

def distanceFromWall1 (x : ℝ) : Prop :=
  x = 2

def distanceFromWall2 (y : ℝ) : Prop :=
  y = 5

def distanceFromPointP (x y z : ℝ) : Prop :=
  7 = Real.sqrt (x^2 + y^2 + z^2)

-- Proving the distance from the ceiling
theorem flyDistanceCeiling (P : ℝ × ℝ × ℝ) (x y z : ℝ) :
  isAtRightAngles P →
  distanceFromWall1 x →
  distanceFromWall2 y →
  distanceFromPointP x y z →
  z = 2 * Real.sqrt 5 := 
sorry

end NUMINAMATH_GPT_flyDistanceCeiling_l1076_107648


namespace NUMINAMATH_GPT_machines_remain_closed_l1076_107698

open Real

/-- A techno company has 14 machines of equal efficiency in its factory.
The annual manufacturing costs are Rs 42000 and establishment charges are Rs 12000.
The annual output of the company is Rs 70000. The annual output and manufacturing
costs are directly proportional to the number of machines. The shareholders get
12.5% profit, which is directly proportional to the annual output of the company.
If some machines remain closed throughout the year, then the percentage decrease
in the amount of profit of the shareholders is 12.5%. Prove that 2 machines remain
closed throughout the year. -/
theorem machines_remain_closed (machines total_cost est_charges output : ℝ)
    (shareholders_profit : ℝ)
    (machines_closed percentage_decrease : ℝ) :
  machines = 14 →
  total_cost = 42000 →
  est_charges = 12000 →
  output = 70000 →
  shareholders_profit = 0.125 →
  percentage_decrease = 0.125 →
  machines_closed = 2 :=
by
  sorry

end NUMINAMATH_GPT_machines_remain_closed_l1076_107698


namespace NUMINAMATH_GPT_solve_abs_eq_l1076_107626

theorem solve_abs_eq (x : ℝ) (h : |x + 2| = |x - 3|) : x = 1 / 2 :=
sorry

end NUMINAMATH_GPT_solve_abs_eq_l1076_107626


namespace NUMINAMATH_GPT_find_second_divisor_l1076_107699

theorem find_second_divisor
  (N D : ℕ)
  (h1 : ∃ k : ℕ, N = 35 * k + 25)
  (h2 : ∃ m : ℕ, N = D * m + 4) :
  D = 21 :=
sorry

end NUMINAMATH_GPT_find_second_divisor_l1076_107699


namespace NUMINAMATH_GPT_fewest_students_possible_l1076_107638

theorem fewest_students_possible (N : ℕ) :
  (N % 5 = 2) ∧ (N % 6 = 3) ∧ (N % 8 = 4) ↔ N = 59 :=
by
  sorry

end NUMINAMATH_GPT_fewest_students_possible_l1076_107638


namespace NUMINAMATH_GPT_n_squared_plus_n_plus_1_is_odd_l1076_107652

theorem n_squared_plus_n_plus_1_is_odd (n : ℤ) : Odd (n^2 + n + 1) :=
sorry

end NUMINAMATH_GPT_n_squared_plus_n_plus_1_is_odd_l1076_107652


namespace NUMINAMATH_GPT_possible_k_value_l1076_107606

theorem possible_k_value (a n k : ℕ) (h1 : n > 1) (h2 : 10^(n-1) ≤ a ∧ a < 10^n)
    (h3 : b = a * (10^n + 1)) (h4 : k = b / a^2) (h5 : b = a * 10 ^n + a) :
  k = 7 := 
sorry

end NUMINAMATH_GPT_possible_k_value_l1076_107606


namespace NUMINAMATH_GPT_total_payment_360_l1076_107654

noncomputable def q : ℝ := 12
noncomputable def p_wage : ℝ := 1.5 * q
noncomputable def p_hourly_rate : ℝ := q + 6
noncomputable def h : ℝ := 20
noncomputable def total_payment_p : ℝ := p_wage * h -- The total payment when candidate p is hired
noncomputable def total_payment_q : ℝ := q * (h + 10) -- The total payment when candidate q is hired

theorem total_payment_360 : 
  p_wage = p_hourly_rate ∧ 
  total_payment_p = total_payment_q ∧ 
  total_payment_p = 360 := by
  sorry

end NUMINAMATH_GPT_total_payment_360_l1076_107654


namespace NUMINAMATH_GPT_sum_of_squares_consecutive_nat_l1076_107631

theorem sum_of_squares_consecutive_nat (n : ℕ) (h : n = 26) : (n - 1) ^ 2 + n ^ 2 + (n + 1) ^ 2 = 2030 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_squares_consecutive_nat_l1076_107631


namespace NUMINAMATH_GPT_leap_day_2040_is_friday_l1076_107617

def leap_day_day_of_week (start_year : ℕ) (start_day : ℕ) (end_year : ℕ) : ℕ :=
  let num_years := end_year - start_year
  let num_leap_years := (num_years + 4) / 4 -- number of leap years including start and end year
  let total_days := 365 * (num_years - num_leap_years) + 366 * num_leap_years
  let day_of_week := (total_days % 7 + start_day) % 7
  day_of_week

theorem leap_day_2040_is_friday :
  leap_day_day_of_week 2008 5 2040 = 5 := 
  sorry

end NUMINAMATH_GPT_leap_day_2040_is_friday_l1076_107617


namespace NUMINAMATH_GPT_ratio_of_speeds_l1076_107601

/-- Define the conditions -/
def distance_AB : ℝ := 540 -- Distance between city A and city B is 540 km
def time_Eddy : ℝ := 3     -- Eddy takes 3 hours to travel to city B
def distance_AC : ℝ := 300 -- Distance between city A and city C is 300 km
def time_Freddy : ℝ := 4   -- Freddy takes 4 hours to travel to city C

/-- Define the average speeds -/
noncomputable def avg_speed_Eddy : ℝ := distance_AB / time_Eddy
noncomputable def avg_speed_Freddy : ℝ := distance_AC / time_Freddy

/-- The statement to prove -/
theorem ratio_of_speeds : avg_speed_Eddy / avg_speed_Freddy = 12 / 5 :=
by sorry

end NUMINAMATH_GPT_ratio_of_speeds_l1076_107601


namespace NUMINAMATH_GPT_max_int_value_of_a_real_roots_l1076_107651

-- Definitions and theorem statement based on the above conditions
theorem max_int_value_of_a_real_roots (a : ℤ) :
  (∃ x : ℝ, (a-1) * x^2 - 2 * x + 3 = 0) ↔ a ≠ 1 ∧ a ≤ 0 := by
  sorry

end NUMINAMATH_GPT_max_int_value_of_a_real_roots_l1076_107651


namespace NUMINAMATH_GPT_total_fast_food_order_cost_l1076_107664

def burger_cost : ℕ := 5
def sandwich_cost : ℕ := 4
def smoothie_cost : ℕ := 4
def smoothies_quantity : ℕ := 2

theorem total_fast_food_order_cost : burger_cost + sandwich_cost + smoothies_quantity * smoothie_cost = 17 := 
by
  sorry

end NUMINAMATH_GPT_total_fast_food_order_cost_l1076_107664


namespace NUMINAMATH_GPT_income_expenses_opposite_l1076_107610

def income_denotation (income : Int) : Int := income

theorem income_expenses_opposite :
  income_denotation 5 = 5 →
  income_denotation (-5) = -5 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_income_expenses_opposite_l1076_107610


namespace NUMINAMATH_GPT_find_smaller_part_l1076_107655

noncomputable def smaller_part (x y : ℕ) : ℕ :=
  if x ≤ y then x else y

theorem find_smaller_part (x y : ℕ) (h1 : x + y = 24) (h2 : 7 * x + 5 * y = 146) : smaller_part x y = 11 :=
  sorry

end NUMINAMATH_GPT_find_smaller_part_l1076_107655


namespace NUMINAMATH_GPT_cost_each_side_is_56_l1076_107614

-- Define the total cost and number of sides
def total_cost : ℕ := 224
def number_of_sides : ℕ := 4

-- Define the cost per side as the division of total cost by number of sides
def cost_per_side : ℕ := total_cost / number_of_sides

-- The theorem stating the cost per side is 56
theorem cost_each_side_is_56 : cost_per_side = 56 :=
by
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_cost_each_side_is_56_l1076_107614


namespace NUMINAMATH_GPT_first_place_team_wins_l1076_107608

-- Define the conditions in Lean 4
variable (joe_won : ℕ := 1) (joe_draw : ℕ := 3) (fp_draw : ℕ := 2) (joe_points : ℕ := 3 * joe_won + joe_draw)
variable (fp_points : ℕ := joe_points + 2)

 -- Define the proof problem
theorem first_place_team_wins : 3 * (fp_points - fp_draw) / 3 = 2 := by
  sorry

end NUMINAMATH_GPT_first_place_team_wins_l1076_107608


namespace NUMINAMATH_GPT_total_books_correct_l1076_107604

-- Define the number of books each person has
def booksKeith : Nat := 20
def booksJason : Nat := 21
def booksMegan : Nat := 15

-- Define the total number of books they have together
def totalBooks : Nat := booksKeith + booksJason + booksMegan

-- Prove that the total number of books is 56
theorem total_books_correct : totalBooks = 56 := by
  sorry

end NUMINAMATH_GPT_total_books_correct_l1076_107604


namespace NUMINAMATH_GPT_quadratic_no_real_roots_l1076_107680

theorem quadratic_no_real_roots 
  (a b c m : ℝ) 
  (h1 : c > 0) 
  (h2 : c = a * m^2) 
  (h3 : c = b * m)
  : ∀ x : ℝ, ¬ (a * x^2 + b * x + c = 0) :=
by 
  sorry

end NUMINAMATH_GPT_quadratic_no_real_roots_l1076_107680


namespace NUMINAMATH_GPT_no_eleven_points_achieve_any_score_l1076_107677

theorem no_eleven_points (x y : ℕ) : 3 * x + 7 * y ≠ 11 := 
sorry

theorem achieve_any_score (S : ℕ) (h : S ≥ 12) : ∃ (x y : ℕ), 3 * x + 7 * y = S :=
sorry

end NUMINAMATH_GPT_no_eleven_points_achieve_any_score_l1076_107677


namespace NUMINAMATH_GPT_parabola_focus_coordinates_l1076_107621

theorem parabola_focus_coordinates :
  (∃ f : ℝ × ℝ, f = (0, 2) ∧ ∀ x y : ℝ, y = (1/8) * x^2 ↔ f = (0, 2)) :=
sorry

end NUMINAMATH_GPT_parabola_focus_coordinates_l1076_107621


namespace NUMINAMATH_GPT_max_crosses_4x10_proof_l1076_107676

def max_crosses_4x10 (table : Matrix ℕ ℕ Bool) : ℕ :=
  sorry -- Placeholder for actual function implementation

theorem max_crosses_4x10_proof (table : Matrix ℕ ℕ Bool) (h : ∀ i < 4, ∃ j < 10, table i j = tt) :
  max_crosses_4x10 table = 30 :=
sorry

end NUMINAMATH_GPT_max_crosses_4x10_proof_l1076_107676


namespace NUMINAMATH_GPT_ratio_is_one_half_l1076_107667

namespace CupRice

-- Define the grains of rice in one cup
def grains_in_one_cup : ℕ := 480

-- Define the grains of rice in the portion of the cup
def grains_in_portion : ℕ := 8 * 3 * 10

-- Define the ratio of the portion of the cup to the whole cup
def portion_to_cup_ratio := grains_in_portion / grains_in_one_cup

-- Prove that the ratio of the portion of the cup to the whole cup is 1:2
theorem ratio_is_one_half : portion_to_cup_ratio = 1 / 2 := by
  -- Proof goes here, but we skip it as required
  sorry
end CupRice

end NUMINAMATH_GPT_ratio_is_one_half_l1076_107667


namespace NUMINAMATH_GPT_power_equality_l1076_107628

theorem power_equality (p : ℕ) : 16^10 = 4^p → p = 20 :=
by
  intro h
  -- proof goes here
  sorry

end NUMINAMATH_GPT_power_equality_l1076_107628


namespace NUMINAMATH_GPT_greatest_integer_a_l1076_107623

theorem greatest_integer_a (a : ℤ) : a * a < 44 → a ≤ 6 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_greatest_integer_a_l1076_107623


namespace NUMINAMATH_GPT_real_and_equal_roots_of_quadratic_l1076_107656

theorem real_and_equal_roots_of_quadratic (k: ℝ) :
  (-(k+2))^2 - 4 * 3 * 12 = 0 ↔ k = 10 ∨ k = -14 :=
by
  sorry

end NUMINAMATH_GPT_real_and_equal_roots_of_quadratic_l1076_107656


namespace NUMINAMATH_GPT_initial_weight_of_alloy_is_16_l1076_107611

variable (Z C : ℝ)
variable (h1 : Z / C = 5 / 3)
variable (h2 : (Z + 8) / C = 3)
variable (A : ℝ := Z + C)

theorem initial_weight_of_alloy_is_16 (h1 : Z / C = 5 / 3) (h2 : (Z + 8) / C = 3) : A = 16 := by
  sorry

end NUMINAMATH_GPT_initial_weight_of_alloy_is_16_l1076_107611


namespace NUMINAMATH_GPT_curve_is_circle_l1076_107650

theorem curve_is_circle (ρ θ : ℝ) (h : ρ = 5 * Real.sin θ) : 
  ∃ (C : ℝ × ℝ) (r : ℝ), ∀ (x y : ℝ),
  (x, y) = (ρ * Real.cos θ, ρ * Real.sin θ) → 
  (x - C.1) ^ 2 + (y - C.2) ^ 2 = r ^ 2 :=
by
  existsi (0, 5 / 2), 5 / 2
  sorry

end NUMINAMATH_GPT_curve_is_circle_l1076_107650


namespace NUMINAMATH_GPT_parallel_lines_iff_a_eq_3_l1076_107665

theorem parallel_lines_iff_a_eq_3 (a : ℝ) :
  (∀ x y : ℝ, (6 * x - 4 * y + 1 = 0) ↔ (a * x - 2 * y - 1 = 0)) ↔ (a = 3) := 
sorry

end NUMINAMATH_GPT_parallel_lines_iff_a_eq_3_l1076_107665


namespace NUMINAMATH_GPT_problem_statement_l1076_107666

variable {x y z : ℝ}

theorem problem_statement (h : x^3 + y^3 + z^3 - 3 * x * y * z - 3 * (x^2 + y^2 + z^2 - x * y - y * z - z * x) = 0)
  (hne : ¬(x = y ∧ y = z)) (hpos : x > 0 ∧ y > 0 ∧ z > 0) :
  (x + y + z = 3) ∧ (x^2 * (1 + y) + y^2 * (1 + z) + z^2 * (1 + x) > 6) :=
sorry

end NUMINAMATH_GPT_problem_statement_l1076_107666


namespace NUMINAMATH_GPT_count_two_digit_numbers_with_unit_7_lt_50_l1076_107678

def is_two_digit_nat (n : ℕ) : Prop := n ≥ 10 ∧ n < 100
def has_unit_digit_7 (n : ℕ) : Prop := n % 10 = 7
def less_than_50 (n : ℕ) : Prop := n < 50

theorem count_two_digit_numbers_with_unit_7_lt_50 : 
  ∃ (s : Finset ℕ), 
    (∀ n ∈ s, is_two_digit_nat n ∧ has_unit_digit_7 n ∧ less_than_50 n) ∧ s.card = 4 := 
by
  sorry

end NUMINAMATH_GPT_count_two_digit_numbers_with_unit_7_lt_50_l1076_107678


namespace NUMINAMATH_GPT_relationship_between_a_and_b_l1076_107609

-- Define the function f
def f (x : ℝ) : ℝ := 3 * x + 1

-- Given conditions
variables (b a : ℝ)
variables (hx : 0 < b) (ha : 0 < a)
variables (x : ℝ) (hb : |x - 1| < b) (hf : |f x - 4| < a)

-- The theorem statement
theorem relationship_between_a_and_b
  (hf_x : ∀ x : ℝ, |x - 1| < b -> |f x - 4| < a) :
  a - 3 * b ≥ 0 :=
sorry

end NUMINAMATH_GPT_relationship_between_a_and_b_l1076_107609


namespace NUMINAMATH_GPT_inequality_division_by_positive_l1076_107681

theorem inequality_division_by_positive (x y : ℝ) (h : x > y) : (x / 5 > y / 5) :=
by
  sorry

end NUMINAMATH_GPT_inequality_division_by_positive_l1076_107681


namespace NUMINAMATH_GPT_total_bathing_suits_l1076_107641

theorem total_bathing_suits 
  (a : ℕ) (b : ℕ) (c : ℕ) (d : ℕ) (e : ℕ)
  (ha : a = 8500) (hb : b = 12750) (hc : c = 5900) (hd : d = 7250) (he : e = 1100) :
  a + b + c + d + e = 35500 :=
by
  sorry

end NUMINAMATH_GPT_total_bathing_suits_l1076_107641


namespace NUMINAMATH_GPT_complex_expression_evaluation_l1076_107602

theorem complex_expression_evaluation (z : ℂ) (h : z^2 + z + 1 = 0) :
  z^101 + z^102 + z^103 + z^104 + z^105 = -1 := 
sorry

end NUMINAMATH_GPT_complex_expression_evaluation_l1076_107602


namespace NUMINAMATH_GPT_question_eq_answer_l1076_107673

theorem question_eq_answer (n : ℝ) (h : 0.25 * 0.1 * n = 15) :
  0.1 * 0.25 * n = 15 :=
by
  sorry

end NUMINAMATH_GPT_question_eq_answer_l1076_107673


namespace NUMINAMATH_GPT_evaluate_expression_l1076_107644

theorem evaluate_expression : 
  1 + 2 / (3 + 4 / (5 + 6 / 7)) = 233 / 151 := 
by 
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1076_107644
