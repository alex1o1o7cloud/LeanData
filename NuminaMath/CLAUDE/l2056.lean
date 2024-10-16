import Mathlib

namespace NUMINAMATH_CALUDE_square_cube_root_product_l2056_205649

theorem square_cube_root_product (a b : ℝ) 
  (ha : a^2 = 16/25) (hb : b^3 = 125/8) : 
  Real.sqrt (a * b) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_square_cube_root_product_l2056_205649


namespace NUMINAMATH_CALUDE_cube_coloring_count_l2056_205699

/-- The number of distinct colorings of a cube with 6 faces using m colors -/
def g (m : ℕ) : ℚ :=
  (1 / 24) * (m^6 + 3*m^4 + 12*m^3 + 8*m^2)

/-- Theorem: The number of distinct colorings of a cube with 6 faces,
    using m colors, where each face is painted one color,
    is equal to (1/24)(m^6 + 3m^4 + 12m^3 + 8m^2) -/
theorem cube_coloring_count (m : ℕ) :
  g m = (1 / 24) * (m^6 + 3*m^4 + 12*m^3 + 8*m^2) :=
by sorry

end NUMINAMATH_CALUDE_cube_coloring_count_l2056_205699


namespace NUMINAMATH_CALUDE_log11_not_expressible_l2056_205622

-- Define the given logarithmic values
def log5 : ℝ := 0.6990
def log6 : ℝ := 0.7782

-- Define a type for basic logarithmic expressions
inductive LogExpr
| Const : ℝ → LogExpr
| Log5 : LogExpr
| Log6 : LogExpr
| Add : LogExpr → LogExpr → LogExpr
| Sub : LogExpr → LogExpr → LogExpr
| Mul : ℝ → LogExpr → LogExpr

-- Function to evaluate a LogExpr
def eval : LogExpr → ℝ
| LogExpr.Const r => r
| LogExpr.Log5 => log5
| LogExpr.Log6 => log6
| LogExpr.Add e1 e2 => eval e1 + eval e2
| LogExpr.Sub e1 e2 => eval e1 - eval e2
| LogExpr.Mul r e => r * eval e

-- Theorem stating that log 11 cannot be expressed using log 5 and log 6
theorem log11_not_expressible : ∀ e : LogExpr, eval e ≠ Real.log 11 := by
  sorry

end NUMINAMATH_CALUDE_log11_not_expressible_l2056_205622


namespace NUMINAMATH_CALUDE_maximum_marks_proof_l2056_205605

-- Define the maximum marks
def M : ℝ := 500

-- Define Pradeep's marks
def pradeep_marks : ℝ := 150

-- Define the passing threshold
def passing_threshold : ℝ := 0.35

-- Theorem statement
theorem maximum_marks_proof :
  (passing_threshold * M = pradeep_marks + 25) ∧ (M = 500) := by
  sorry

end NUMINAMATH_CALUDE_maximum_marks_proof_l2056_205605


namespace NUMINAMATH_CALUDE_factor_implies_m_value_l2056_205612

theorem factor_implies_m_value (m : ℝ) : 
  (∃ k : ℝ, ∀ x : ℝ, x^2 - m*x - 40 = (x + 5) * k) → m = 3 := by
  sorry

end NUMINAMATH_CALUDE_factor_implies_m_value_l2056_205612


namespace NUMINAMATH_CALUDE_equation_solution_l2056_205642

theorem equation_solution : ∃! x : ℝ, (2 / x = 1 / (x + 1)) ∧ (x = -2) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2056_205642


namespace NUMINAMATH_CALUDE_least_positive_angle_theorem_l2056_205655

theorem least_positive_angle_theorem : 
  ∃ θ : Real, θ > 0 ∧ θ = 15 * π / 180 ∧
  (∀ φ : Real, φ > 0 ∧ Real.cos (15 * π / 180) = Real.sin (45 * π / 180) + Real.sin φ → θ ≤ φ) :=
sorry

end NUMINAMATH_CALUDE_least_positive_angle_theorem_l2056_205655


namespace NUMINAMATH_CALUDE_vegan_nut_free_menu_fraction_l2056_205602

theorem vegan_nut_free_menu_fraction :
  let total_vegan_dishes : ℕ := 8
  let vegan_menu_fraction : ℚ := 1/4
  let nut_containing_vegan_dishes : ℕ := 5
  let nut_free_vegan_dishes : ℕ := total_vegan_dishes - nut_containing_vegan_dishes
  let nut_free_vegan_fraction : ℚ := nut_free_vegan_dishes / total_vegan_dishes
  nut_free_vegan_fraction * vegan_menu_fraction = 3/32 :=
by sorry

end NUMINAMATH_CALUDE_vegan_nut_free_menu_fraction_l2056_205602


namespace NUMINAMATH_CALUDE_min_friend_pairs_2000_users_min_friend_pairs_within_bounds_l2056_205681

/-- Represents a social network with a fixed number of users and invitations per user. -/
structure SocialNetwork where
  numUsers : ℕ
  invitationsPerUser : ℕ

/-- Calculates the minimum number of friend pairs in a social network where friendships
    are formed only when invitations are mutual. -/
def minFriendPairs (network : SocialNetwork) : ℕ :=
  (network.numUsers * network.invitationsPerUser) / 2

/-- Theorem stating that in a social network with 2000 users, each inviting 1000 others,
    the minimum number of friend pairs is 1000. -/
theorem min_friend_pairs_2000_users :
  let network : SocialNetwork := { numUsers := 2000, invitationsPerUser := 1000 }
  minFriendPairs network = 1000 := by
  sorry

/-- Verifies that the calculated minimum number of friend pairs does not exceed
    the maximum possible number of pairs given the number of users. -/
theorem min_friend_pairs_within_bounds (network : SocialNetwork) :
  minFriendPairs network ≤ (network.numUsers.choose 2) := by
  sorry

end NUMINAMATH_CALUDE_min_friend_pairs_2000_users_min_friend_pairs_within_bounds_l2056_205681


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_l2056_205638

theorem necessary_but_not_sufficient_condition (x : ℝ) : 
  (∀ y : ℝ, 2*y > 2 → y > -1) ∧ 
  (∃ z : ℝ, z > -1 ∧ ¬(2*z > 2)) := by
  sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_l2056_205638


namespace NUMINAMATH_CALUDE_z_in_third_quadrant_l2056_205610

def z : ℂ := (-2 + Complex.I) * Complex.I

theorem z_in_third_quadrant : 
  Real.sign (z.re) = -1 ∧ Real.sign (z.im) = -1 :=
by sorry

end NUMINAMATH_CALUDE_z_in_third_quadrant_l2056_205610


namespace NUMINAMATH_CALUDE_no_integer_solution_l2056_205600

theorem no_integer_solution : ¬ ∃ (x y z : ℤ), x^4 + y^4 + z^4 - 2*x^2*y^2 - 2*y^2*z^2 - 2*z^2*x^2 = 2000 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solution_l2056_205600


namespace NUMINAMATH_CALUDE_perfect_squares_as_sum_of_powers_of_two_l2056_205623

theorem perfect_squares_as_sum_of_powers_of_two (n a b : ℕ) (h1 : a ≥ b) (h2 : n^2 = 2^a + 2^b) :
  ∃ k : ℕ, n^2 = 4^(k+1) ∨ n^2 = 9 * 4^k :=
sorry

end NUMINAMATH_CALUDE_perfect_squares_as_sum_of_powers_of_two_l2056_205623


namespace NUMINAMATH_CALUDE_olivia_made_45_dollars_l2056_205643

/-- The amount of money Olivia made selling chocolate bars -/
def olivia_earnings (bar_price : ℕ) (total_bars : ℕ) (unsold_bars : ℕ) : ℕ :=
  (total_bars - unsold_bars) * bar_price

/-- Proof that Olivia made $45 -/
theorem olivia_made_45_dollars :
  olivia_earnings 5 15 6 = 45 := by
  sorry

end NUMINAMATH_CALUDE_olivia_made_45_dollars_l2056_205643


namespace NUMINAMATH_CALUDE_optimal_transport_minimum_cost_l2056_205626

structure TransportProblem where
  total_parcels : ℕ
  tent_excess : ℕ
  type_a_tent_capacity : ℕ
  type_a_food_capacity : ℕ
  type_b_tent_capacity : ℕ
  type_b_food_capacity : ℕ
  type_a_cost : ℕ
  type_b_cost : ℕ
  total_trucks : ℕ

def optimal_transport (p : TransportProblem) : ℕ × ℕ × ℕ :=
  sorry

theorem optimal_transport_minimum_cost (p : TransportProblem) :
  p.total_parcels = 360 ∧
  p.tent_excess = 110 ∧
  p.type_a_tent_capacity = 40 ∧
  p.type_a_food_capacity = 10 ∧
  p.type_b_tent_capacity = 20 ∧
  p.type_b_food_capacity = 20 ∧
  p.type_a_cost = 4000 ∧
  p.type_b_cost = 3600 ∧
  p.total_trucks = 9 →
  optimal_transport p = (3, 6, 33600) :=
by sorry

end NUMINAMATH_CALUDE_optimal_transport_minimum_cost_l2056_205626


namespace NUMINAMATH_CALUDE_simplify_cubic_root_sum_exponents_l2056_205680

-- Define the expression
def radicand : ℕ → ℕ → ℕ → ℕ → ℕ
  | a, b, c, d => 60 * a^5 * b^7 * c^8 * d^2

-- Define the function to calculate the sum of exponents outside the radical
def sum_exponents_outside_radical : ℕ → ℕ → ℕ → ℕ → ℕ
  | a, b, c, d => 5

-- Theorem statement
theorem simplify_cubic_root_sum_exponents
  (a b c d : ℕ) :
  sum_exponents_outside_radical a b c d = 5 :=
by sorry

end NUMINAMATH_CALUDE_simplify_cubic_root_sum_exponents_l2056_205680


namespace NUMINAMATH_CALUDE_brownie_pieces_theorem_l2056_205613

/-- The number of big square pieces the brownies were cut into -/
def num_pieces : ℕ := sorry

/-- The total amount of money Tamara made from selling brownies -/
def total_amount : ℕ := 32

/-- The cost of each brownie -/
def cost_per_brownie : ℕ := 2

/-- The number of pans of brownies made -/
def num_pans : ℕ := 2

theorem brownie_pieces_theorem :
  num_pieces = total_amount / cost_per_brownie :=
sorry

end NUMINAMATH_CALUDE_brownie_pieces_theorem_l2056_205613


namespace NUMINAMATH_CALUDE_geometric_progression_problem_l2056_205604

theorem geometric_progression_problem (b₁ q : ℚ) : 
  (b₁ * q^3 - b₁ * q = -45/32) → 
  (b₁ * q^5 - b₁ * q^3 = -45/512) → 
  ((b₁ = 6 ∧ q = 1/4) ∨ (b₁ = -6 ∧ q = -1/4)) :=
by sorry

end NUMINAMATH_CALUDE_geometric_progression_problem_l2056_205604


namespace NUMINAMATH_CALUDE_units_digit_problem_l2056_205684

theorem units_digit_problem : ∃ n : ℕ, (15 + Real.sqrt 225)^17 + (15 - Real.sqrt 225)^17 = n * 10 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_problem_l2056_205684


namespace NUMINAMATH_CALUDE_picnic_watermelon_slices_l2056_205686

/-- The number of watermelon slices at a family picnic -/
def total_watermelon_slices : ℕ :=
  let danny_watermelons : ℕ := 3
  let danny_slices_per_watermelon : ℕ := 10
  let sister_watermelons : ℕ := 1
  let sister_slices_per_watermelon : ℕ := 15
  (danny_watermelons * danny_slices_per_watermelon) + (sister_watermelons * sister_slices_per_watermelon)

theorem picnic_watermelon_slices : total_watermelon_slices = 45 := by
  sorry

end NUMINAMATH_CALUDE_picnic_watermelon_slices_l2056_205686


namespace NUMINAMATH_CALUDE_work_completion_theorem_l2056_205650

/-- The number of men originally employed to finish the work in 11 days -/
def original_men : ℕ := 27

/-- The number of additional men who joined -/
def additional_men : ℕ := 10

/-- The original number of days to finish the work -/
def original_days : ℕ := 11

/-- The number of days saved after additional men joined -/
def days_saved : ℕ := 3

theorem work_completion_theorem :
  original_men + additional_men = 37 ∧
  original_men * original_days = (original_men + additional_men) * (original_days - days_saved) :=
sorry

end NUMINAMATH_CALUDE_work_completion_theorem_l2056_205650


namespace NUMINAMATH_CALUDE_binary_1101_equals_13_l2056_205629

def binary_to_decimal (b : List Bool) : Nat :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_1101_equals_13 :
  binary_to_decimal [true, false, true, true] = 13 := by
  sorry

end NUMINAMATH_CALUDE_binary_1101_equals_13_l2056_205629


namespace NUMINAMATH_CALUDE_sum_of_digits_of_1962_digit_number_div_by_9_l2056_205628

/-- A function that returns the sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- A function that checks if a natural number has exactly 1962 digits -/
def has1962Digits (n : ℕ) : Prop := sorry

theorem sum_of_digits_of_1962_digit_number_div_by_9 (n : ℕ) 
  (h1 : has1962Digits n) 
  (h2 : n % 9 = 0) : 
  let a := sumOfDigits n
  let b := sumOfDigits a
  let c := sumOfDigits b
  c = 9 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_1962_digit_number_div_by_9_l2056_205628


namespace NUMINAMATH_CALUDE_infinitely_many_super_abundant_l2056_205682

/-- Sum of divisors function -/
def sigma (n : ℕ) : ℕ := sorry

/-- Super-abundant number -/
def is_super_abundant (m : ℕ) : Prop :=
  m ≥ 1 ∧ ∀ k ∈ Finset.range m, (sigma m : ℚ) / m > (sigma k : ℚ) / k

/-- There are infinitely many super-abundant numbers -/
theorem infinitely_many_super_abundant :
  ∀ N, ∃ m > N, is_super_abundant m :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_super_abundant_l2056_205682


namespace NUMINAMATH_CALUDE_lollipops_left_after_sharing_l2056_205606

def raspberry_lollipops : ℕ := 51
def mint_lollipops : ℕ := 121
def chocolate_lollipops : ℕ := 9
def blueberry_lollipops : ℕ := 232
def num_friends : ℕ := 13

theorem lollipops_left_after_sharing :
  (raspberry_lollipops + mint_lollipops + chocolate_lollipops + blueberry_lollipops) % num_friends = 10 := by
  sorry

end NUMINAMATH_CALUDE_lollipops_left_after_sharing_l2056_205606


namespace NUMINAMATH_CALUDE_chocolate_division_l2056_205679

theorem chocolate_division (total_chocolate : ℚ) (num_piles : ℕ) (piles_given : ℕ) :
  total_chocolate = 72 / 7 →
  num_piles = 6 →
  piles_given = 2 →
  piles_given * (total_chocolate / num_piles) = 24 / 7 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_division_l2056_205679


namespace NUMINAMATH_CALUDE_alok_chapati_order_l2056_205653

/-- The number of chapatis ordered by Alok -/
def num_chapatis : ℕ := sorry

/-- The cost of a single chapati in Rupees -/
def cost_chapati : ℕ := 6

/-- The number of plates of rice ordered -/
def num_rice : ℕ := 5

/-- The cost of a single plate of rice in Rupees -/
def cost_rice : ℕ := 45

/-- The number of plates of mixed vegetable ordered -/
def num_vegetable : ℕ := 7

/-- The cost of a single plate of mixed vegetable in Rupees -/
def cost_vegetable : ℕ := 70

/-- The number of ice-cream cups ordered -/
def num_icecream : ℕ := 6

/-- The cost of a single ice-cream cup in Rupees -/
def cost_icecream : ℕ := 40

/-- The total amount paid by Alok in Rupees -/
def total_paid : ℕ := 1051

theorem alok_chapati_order :
  num_chapatis * cost_chapati +
  num_rice * cost_rice +
  num_vegetable * cost_vegetable +
  num_icecream * cost_icecream = total_paid ∧
  num_chapatis = 16 := by sorry

end NUMINAMATH_CALUDE_alok_chapati_order_l2056_205653


namespace NUMINAMATH_CALUDE_pages_torn_off_l2056_205670

def total_pages : ℕ := 100
def sum_remaining_pages : ℕ := 4949

theorem pages_torn_off : 
  ∃ (torn_pages : Finset ℕ), 
    torn_pages.card = 3 ∧ 
    (Finset.range total_pages.succ).sum id - torn_pages.sum id = sum_remaining_pages :=
sorry

end NUMINAMATH_CALUDE_pages_torn_off_l2056_205670


namespace NUMINAMATH_CALUDE_keith_picked_six_apples_l2056_205671

/-- Given the number of apples picked by Mike, eaten by Nancy, and left in total,
    calculate the number of apples picked by Keith. -/
def keith_apples (mike_picked : ℝ) (nancy_ate : ℝ) (total_left : ℝ) : ℝ :=
  total_left - (mike_picked - nancy_ate)

/-- Theorem stating that Keith picked 6.0 apples given the problem conditions. -/
theorem keith_picked_six_apples :
  keith_apples 7.0 3.0 10 = 6.0 := by
  sorry

end NUMINAMATH_CALUDE_keith_picked_six_apples_l2056_205671


namespace NUMINAMATH_CALUDE_equation_D_is_quadratic_l2056_205631

/-- A quadratic equation in x is of the form ax² + bx + c = 0, where a ≠ 0 -/
def is_quadratic_in_x (a b c : ℝ) : Prop := a ≠ 0

/-- The equation (k²+1)x²-2x+1=0 -/
def equation_D (k : ℝ) (x : ℝ) : Prop :=
  (k^2 + 1) * x^2 - 2*x + 1 = 0

theorem equation_D_is_quadratic (k : ℝ) :
  is_quadratic_in_x (k^2 + 1) (-2) 1 := by sorry

end NUMINAMATH_CALUDE_equation_D_is_quadratic_l2056_205631


namespace NUMINAMATH_CALUDE_lg_17_not_uniquely_calculable_l2056_205692

-- Define the logarithm function (base 10)
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Given conditions
axiom lg_8 : lg 8 = 0.9031
axiom lg_9 : lg 9 = 0.9542

-- Define a proposition that lg 17 cannot be uniquely calculated
def lg_17_not_calculable : Prop :=
  ∀ f : ℝ → ℝ → ℝ, 
    (∀ x y : ℝ, f (lg x) (lg y) = lg (x + y)) → 
    ¬∃! z : ℝ, f (lg 8) (lg 9) = z ∧ z = lg 17

-- Theorem statement
theorem lg_17_not_uniquely_calculable : lg_17_not_calculable :=
sorry

end NUMINAMATH_CALUDE_lg_17_not_uniquely_calculable_l2056_205692


namespace NUMINAMATH_CALUDE_sin_alpha_plus_7pi_over_6_l2056_205614

theorem sin_alpha_plus_7pi_over_6 (α : ℝ) 
  (h : Real.cos (α - π/6) + Real.sin α = (4/5) * Real.sqrt 3) : 
  Real.sin (α + 7*π/6) = -4/5 := by sorry

end NUMINAMATH_CALUDE_sin_alpha_plus_7pi_over_6_l2056_205614


namespace NUMINAMATH_CALUDE_becky_new_necklaces_l2056_205687

/-- The number of new necklaces Becky bought -/
def new_necklaces (initial : ℕ) (broken : ℕ) (given_away : ℕ) (final : ℕ) : ℕ :=
  final - (initial - broken - given_away)

/-- Theorem stating that Becky bought 5 new necklaces -/
theorem becky_new_necklaces :
  new_necklaces 50 3 15 37 = 5 := by
  sorry

end NUMINAMATH_CALUDE_becky_new_necklaces_l2056_205687


namespace NUMINAMATH_CALUDE_vacation_cost_l2056_205633

theorem vacation_cost (cost : ℝ) : 
  (cost / 3 - cost / 4 = 60) → cost = 720 := by
  sorry

end NUMINAMATH_CALUDE_vacation_cost_l2056_205633


namespace NUMINAMATH_CALUDE_number_divisibility_l2056_205661

theorem number_divisibility (x : ℝ) : (x / 6) * 12 = 18 → x = 9 := by
  sorry

end NUMINAMATH_CALUDE_number_divisibility_l2056_205661


namespace NUMINAMATH_CALUDE_raja_medicine_percentage_l2056_205639

/-- Raja's monthly expenses and savings --/
def monthly_expenses (income medicine_percentage : ℝ) : Prop :=
  let household_percentage : ℝ := 0.35
  let clothes_percentage : ℝ := 0.20
  let savings : ℝ := 15000
  household_percentage * income + 
  clothes_percentage * income + 
  medicine_percentage * income + 
  savings = income

theorem raja_medicine_percentage : 
  ∃ (medicine_percentage : ℝ), 
    monthly_expenses 37500 medicine_percentage ∧ 
    medicine_percentage = 0.05 := by
  sorry

end NUMINAMATH_CALUDE_raja_medicine_percentage_l2056_205639


namespace NUMINAMATH_CALUDE_vector_ratio_bounds_l2056_205654

theorem vector_ratio_bounds (a b : ℝ × ℝ) 
  (h1 : ‖a + b‖ = 3)
  (h2 : ‖a - b‖ = 2) :
  (2 / 5 : ℝ) ≤ ‖a‖ / (a.1 * b.1 + a.2 * b.2) ∧ 
  ‖a‖ / (a.1 * b.1 + a.2 * b.2) ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_vector_ratio_bounds_l2056_205654


namespace NUMINAMATH_CALUDE_parabola_fixed_point_l2056_205660

/-- The parabola equation as a function of x and p -/
def parabola (x p : ℝ) : ℝ := 2 * x^2 - p * x + 4 * p + 1

/-- The fixed point through which the parabola passes -/
def fixed_point : ℝ × ℝ := (4, 33)

theorem parabola_fixed_point :
  ∀ p : ℝ, parabola (fixed_point.1) p = fixed_point.2 := by
  sorry

#check parabola_fixed_point

end NUMINAMATH_CALUDE_parabola_fixed_point_l2056_205660


namespace NUMINAMATH_CALUDE_correct_quadratic_equation_l2056_205609

theorem correct_quadratic_equation :
  ∀ (b c : ℝ),
  (∃ (b' : ℝ), b' ≠ b ∧ (8 : ℝ) * (2 : ℝ) = 9) →
  (∃ (c' : ℝ), c' ≠ c ∧ (-9 : ℝ) + (-1 : ℝ) = -b') →
  (b = -10 ∧ c = 9) :=
by sorry

end NUMINAMATH_CALUDE_correct_quadratic_equation_l2056_205609


namespace NUMINAMATH_CALUDE_adjacent_chair_subsets_l2056_205615

/-- Given 12 chairs arranged in a circle, this function calculates the number of subsets
    containing at least three adjacent chairs. -/
def subsets_with_adjacent_chairs (num_chairs : ℕ) : ℕ :=
  if num_chairs = 12 then
    2010
  else
    0

/-- Theorem stating that for 12 chairs in a circle, there are 2010 subsets
    with at least three adjacent chairs. -/
theorem adjacent_chair_subsets :
  subsets_with_adjacent_chairs 12 = 2010 := by
  sorry

#eval subsets_with_adjacent_chairs 12

end NUMINAMATH_CALUDE_adjacent_chair_subsets_l2056_205615


namespace NUMINAMATH_CALUDE_sphere_to_hemisphere_volume_ratio_l2056_205665

/-- The ratio of the volume of a sphere to the volume of a hemisphere -/
theorem sphere_to_hemisphere_volume_ratio (r : ℝ) (r_pos : r > 0) :
  (4 / 3 * Real.pi * r^3) / (1 / 2 * 4 / 3 * Real.pi * (3 * r)^3) = 1 / 13.5 := by
  sorry

#check sphere_to_hemisphere_volume_ratio

end NUMINAMATH_CALUDE_sphere_to_hemisphere_volume_ratio_l2056_205665


namespace NUMINAMATH_CALUDE_digit_difference_in_base_d_l2056_205611

/-- Represents a digit in base d -/
def Digit (d : ℕ) := {n : ℕ // n < d}

/-- Converts a two-digit number AB in base d to its decimal representation -/
def toDecimal (d : ℕ) (A B : Digit d) : ℕ := A.val * d + B.val

theorem digit_difference_in_base_d 
  (d : ℕ) 
  (h_d : d > 7) 
  (A B : Digit d) 
  (h_sum : toDecimal d A B + toDecimal d A A = 1 * d * d + 7 * d + 2) :
  (A.val - B.val : ℤ) = 4 :=
sorry

end NUMINAMATH_CALUDE_digit_difference_in_base_d_l2056_205611


namespace NUMINAMATH_CALUDE_linear_equation_integer_solutions_l2056_205632

theorem linear_equation_integer_solutions :
  ∃! (S : Finset ℕ), 
    (∀ m ∈ S, m > 0) ∧ 
    (∀ m ∈ S, ∃ x : ℕ, x > 0 ∧ m * x + 2 * x - 12 = 0) ∧
    (∀ m : ℕ, m > 0 → (∃ x : ℕ, x > 0 ∧ m * x + 2 * x - 12 = 0) → m ∈ S) ∧
    S.card = 4 :=
by sorry

end NUMINAMATH_CALUDE_linear_equation_integer_solutions_l2056_205632


namespace NUMINAMATH_CALUDE_centroid_is_unique_interior_point_l2056_205641

/-- A point in the integer lattice -/
structure LatticePoint where
  x : ℤ
  y : ℤ

/-- A triangle defined by three lattice points -/
structure LatticeTriangle where
  A : LatticePoint
  B : LatticePoint
  C : LatticePoint

/-- Predicate to check if a point is inside a triangle -/
def IsInside (p : LatticePoint) (t : LatticeTriangle) : Prop := sorry

/-- Predicate to check if a point is on the boundary of a triangle -/
def IsOnBoundary (p : LatticePoint) (t : LatticeTriangle) : Prop := sorry

/-- The centroid of a triangle -/
def Centroid (t : LatticeTriangle) : LatticePoint := sorry

/-- Main theorem -/
theorem centroid_is_unique_interior_point (t : LatticeTriangle) 
  (h1 : ∀ p : LatticePoint, IsOnBoundary p t → p = t.A ∨ p = t.B ∨ p = t.C)
  (h2 : ∃! p : LatticePoint, IsInside p t) :
  ∃ p : LatticePoint, IsInside p t ∧ p = Centroid t := by
  sorry

end NUMINAMATH_CALUDE_centroid_is_unique_interior_point_l2056_205641


namespace NUMINAMATH_CALUDE_total_volume_carl_and_kate_l2056_205627

/-- The volume of a cube with side length s -/
def cube_volume (s : ℕ) : ℕ := s^3

/-- The total volume of n cubes, each with side length s -/
def total_volume (n : ℕ) (s : ℕ) : ℕ := n * cube_volume s

theorem total_volume_carl_and_kate : 
  total_volume 4 3 + total_volume 3 4 = 300 := by
  sorry

end NUMINAMATH_CALUDE_total_volume_carl_and_kate_l2056_205627


namespace NUMINAMATH_CALUDE_washing_loads_proof_l2056_205667

def washing_machine_capacity : ℕ := 8
def num_shirts : ℕ := 39
def num_sweaters : ℕ := 33

theorem washing_loads_proof :
  let total_clothes := num_shirts + num_sweaters
  let num_loads := (total_clothes + washing_machine_capacity - 1) / washing_machine_capacity
  num_loads = 9 := by sorry

end NUMINAMATH_CALUDE_washing_loads_proof_l2056_205667


namespace NUMINAMATH_CALUDE_subset_property_j_bound_l2056_205678

variable (m n : ℕ+)
variable (A : Finset ℕ)
variable (B : Finset ℕ)
variable (S : Finset (ℕ × ℕ))

def setA : Finset ℕ := Finset.range n
def setB : Finset ℕ := Finset.range m

def property_j (S : Finset (ℕ × ℕ)) : Prop :=
  ∀ (a b x y : ℕ), (a, b) ∈ S → (x, y) ∈ S → (a - x) * (b - y) ≤ 0

theorem subset_property_j_bound :
  A = setA m → B = setB n → S ⊆ A ×ˢ B → property_j S → S.card ≤ m + n - 1 := by
  sorry

end NUMINAMATH_CALUDE_subset_property_j_bound_l2056_205678


namespace NUMINAMATH_CALUDE_apple_sharing_ways_l2056_205685

/-- The number of ways to distribute n indistinguishable objects into k distinguishable boxes --/
def stars_and_bars (n k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- The number of ways to distribute apples among people with a minimum requirement --/
def apple_distribution (total min_per_person people : ℕ) : ℕ :=
  stars_and_bars (total - min_per_person * people) people

theorem apple_sharing_ways :
  apple_distribution 24 2 3 = 190 := by
  sorry

end NUMINAMATH_CALUDE_apple_sharing_ways_l2056_205685


namespace NUMINAMATH_CALUDE_symmetric_points_sum_l2056_205683

/-- Two points are symmetric with respect to the origin if their coordinates are negatives of each other -/
def symmetric_wrt_origin (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₁ = -x₂ ∧ y₁ = -y₂

/-- Given points A(1, a) and B(b, -2) symmetric with respect to the origin, prove a + b = 1 -/
theorem symmetric_points_sum (a b : ℝ) 
  (h : symmetric_wrt_origin 1 a b (-2)) : a + b = 1 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_sum_l2056_205683


namespace NUMINAMATH_CALUDE_volume_of_removed_tetrahedra_l2056_205666

-- Define the cube
def cube_side_length : ℝ := 2

-- Define the number of segments per edge
def segments_per_edge : ℕ := 3

-- Define the number of corners (tetrahedra)
def num_corners : ℕ := 8

-- Theorem statement
theorem volume_of_removed_tetrahedra :
  let segment_length : ℝ := cube_side_length / segments_per_edge
  let base_area : ℝ := (1 / 2) * segment_length^2
  let tetrahedron_height : ℝ := segment_length
  let tetrahedron_volume : ℝ := (1 / 3) * base_area * tetrahedron_height
  let total_volume : ℝ := num_corners * tetrahedron_volume
  total_volume = 32 / 81 := by sorry

end NUMINAMATH_CALUDE_volume_of_removed_tetrahedra_l2056_205666


namespace NUMINAMATH_CALUDE_absolute_value_expression_l2056_205677

theorem absolute_value_expression : |-2| * (|-25| - |5|) = 40 := by sorry

end NUMINAMATH_CALUDE_absolute_value_expression_l2056_205677


namespace NUMINAMATH_CALUDE_certain_person_age_l2056_205646

def sandy_age : ℕ := 34
def person_age : ℕ := 10

theorem certain_person_age :
  (sandy_age * 10 = 340) →
  ((sandy_age + 2) = 3 * (person_age + 2)) →
  person_age = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_certain_person_age_l2056_205646


namespace NUMINAMATH_CALUDE_correct_average_after_mark_correction_l2056_205690

theorem correct_average_after_mark_correction 
  (n : ℕ) 
  (initial_average : ℚ) 
  (wrong_mark correct_mark : ℚ) :
  n = 25 →
  initial_average = 100 →
  wrong_mark = 60 →
  correct_mark = 10 →
  (n * initial_average - (wrong_mark - correct_mark)) / n = 98 := by
sorry

end NUMINAMATH_CALUDE_correct_average_after_mark_correction_l2056_205690


namespace NUMINAMATH_CALUDE_min_max_values_l2056_205651

theorem min_max_values (a b : ℝ) (h1 : a > b) (h2 : b > 1) (h3 : a + 3*b = 5) :
  (((1 : ℝ) / (a - b)) + (4 / (b - 1)) ≥ 25) ∧ (a*b - b^2 - a + b ≤ (1 : ℝ) / 16) := by
  sorry

end NUMINAMATH_CALUDE_min_max_values_l2056_205651


namespace NUMINAMATH_CALUDE_remainder_444_pow_444_mod_13_l2056_205658

theorem remainder_444_pow_444_mod_13 : 444^444 ≡ 1 [ZMOD 13] := by
  sorry

end NUMINAMATH_CALUDE_remainder_444_pow_444_mod_13_l2056_205658


namespace NUMINAMATH_CALUDE_unit_vector_AB_l2056_205675

/-- Given points A(1,3) and B(4,-1), the unit vector in the same direction as vector AB is (3/5, -4/5) -/
theorem unit_vector_AB (A B : ℝ × ℝ) (h : A = (1, 3) ∧ B = (4, -1)) :
  let AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)
  let magnitude : ℝ := Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)
  (AB.1 / magnitude, AB.2 / magnitude) = (3/5, -4/5) := by
  sorry


end NUMINAMATH_CALUDE_unit_vector_AB_l2056_205675


namespace NUMINAMATH_CALUDE_crabapple_sequences_count_l2056_205636

/-- The number of ways to select 5 students from a group of 13 students,
    where the order matters and no student is selected more than once. -/
def crabapple_sequences : ℕ :=
  13 * 12 * 11 * 10 * 9

/-- Theorem stating that the number of crabapple recipient sequences is 154,440. -/
theorem crabapple_sequences_count : crabapple_sequences = 154440 := by
  sorry

end NUMINAMATH_CALUDE_crabapple_sequences_count_l2056_205636


namespace NUMINAMATH_CALUDE_cos_2alpha_minus_2beta_l2056_205669

theorem cos_2alpha_minus_2beta (α β : ℝ) 
  (h1 : Real.sin (α + β) = 2/3)
  (h2 : Real.sin α * Real.cos β = 1/2) : 
  Real.cos (2*α - 2*β) = 7/9 := by
  sorry

end NUMINAMATH_CALUDE_cos_2alpha_minus_2beta_l2056_205669


namespace NUMINAMATH_CALUDE_minimum_value_of_function_l2056_205624

theorem minimum_value_of_function (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_sum : a + b + c = 1) : 
  (a^4 / (a^3 + b^2 + c^2)) + (b^4 / (b^3 + a^2 + c^2)) + (c^4 / (c^3 + b^2 + a^2)) ≥ 1/7 := by
  sorry

end NUMINAMATH_CALUDE_minimum_value_of_function_l2056_205624


namespace NUMINAMATH_CALUDE_pastries_made_pastries_made_correct_l2056_205693

/-- Given information about Baker's cakes and pastries -/
structure BakerInfo where
  cakes_made : ℕ
  cakes_sold : ℕ
  pastries_sold : ℕ
  cakes_pastries_diff : ℕ
  h1 : cakes_made = 157
  h2 : cakes_sold = 158
  h3 : pastries_sold = 147
  h4 : cakes_sold - pastries_sold = cakes_pastries_diff
  h5 : cakes_pastries_diff = 11

/-- Theorem stating the number of pastries Baker made -/
theorem pastries_made (info : BakerInfo) : ℕ := by
  sorry

#check @pastries_made

/-- The actual number of pastries Baker made -/
def actual_pastries_made : ℕ := 146

/-- Theorem proving that the calculated number of pastries matches the actual number -/
theorem pastries_made_correct (info : BakerInfo) : pastries_made info = actual_pastries_made := by
  sorry

end NUMINAMATH_CALUDE_pastries_made_pastries_made_correct_l2056_205693


namespace NUMINAMATH_CALUDE_max_constant_term_l2056_205617

def p₁ (a : ℝ) (x : ℝ) : ℝ := x - a

def p₂ (r s t : ℕ) (x : ℝ) : ℝ := (x - 1) ^ r * (x - 2) ^ s * (x + 3) ^ t

def constant_term (a : ℝ) (r s t : ℕ) : ℝ :=
  (-1) ^ (r + s) * 2 ^ s * 3 ^ t - a

theorem max_constant_term :
  ∀ a : ℝ, ∀ r s t : ℕ,
    r ≥ 1 → s ≥ 1 → t ≥ 1 → r + s + t = 4 →
    constant_term a r s t ≤ 21 ∧
    (constant_term (-3) 1 1 2 = 21) :=
sorry

end NUMINAMATH_CALUDE_max_constant_term_l2056_205617


namespace NUMINAMATH_CALUDE_degree_of_composed_product_l2056_205695

/-- Given polynomials p and q with degrees 3 and 4 respectively,
    prove that the degree of p(x^4) * q(x^5) is 32 -/
theorem degree_of_composed_product (p q : Polynomial ℝ) 
  (hp : Polynomial.degree p = 3) (hq : Polynomial.degree q = 4) :
  Polynomial.degree (p.comp (Polynomial.X ^ 4) * q.comp (Polynomial.X ^ 5)) = 32 := by
  sorry

end NUMINAMATH_CALUDE_degree_of_composed_product_l2056_205695


namespace NUMINAMATH_CALUDE_roots_equation_q_value_l2056_205663

theorem roots_equation_q_value (a b m p q : ℝ) : 
  (a^2 - m*a + 3 = 0) →
  (b^2 - m*b + 3 = 0) →
  ((a + 1/b)^2 - p*(a + 1/b) + q = 0) →
  ((b + 1/a)^2 - p*(b + 1/a) + q = 0) →
  q = 16/3 := by
sorry

end NUMINAMATH_CALUDE_roots_equation_q_value_l2056_205663


namespace NUMINAMATH_CALUDE_project_hours_difference_l2056_205694

/-- 
Given a project where:
- The total hours charged is 180
- Pat charged twice as much time as Kate
- Pat charged 1/3 as much time as Mark

Prove that Mark charged 100 more hours than Kate.
-/
theorem project_hours_difference (kate : ℝ) (pat : ℝ) (mark : ℝ) 
  (h1 : kate + pat + mark = 180)
  (h2 : pat = 2 * kate)
  (h3 : pat = (1/3) * mark) : 
  mark - kate = 100 := by
  sorry

end NUMINAMATH_CALUDE_project_hours_difference_l2056_205694


namespace NUMINAMATH_CALUDE_fifth_term_value_l2056_205640

/-- Given a sequence {aₙ} where Sₙ denotes the sum of its first n terms and Sₙ = n² + 1,
    prove that a₅ = 9. -/
theorem fifth_term_value (a : ℕ → ℝ) (S : ℕ → ℝ) 
    (h : ∀ n, S n = n^2 + 1) : a 5 = 9 := by
  sorry

end NUMINAMATH_CALUDE_fifth_term_value_l2056_205640


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_three_l2056_205691

theorem reciprocal_of_negative_three :
  (1 : ℚ) / (-3 : ℚ) = -1/3 := by sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_three_l2056_205691


namespace NUMINAMATH_CALUDE_system_solution_value_l2056_205647

theorem system_solution_value (x y m : ℝ) : 
  (2 * x + 6 * y = 25) →
  (6 * x + 2 * y = -11) →
  (x - y = m - 1) →
  m = -8 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_value_l2056_205647


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l2056_205607

theorem imaginary_part_of_z (z : ℂ) (h : (1 + Complex.I) * z = 2 - 3 * Complex.I) :
  z.im = -5/2 := by
sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l2056_205607


namespace NUMINAMATH_CALUDE_pascal_interior_sum_l2056_205645

def interior_sum (n : ℕ) : ℕ := 2^(n-1) - 2

theorem pascal_interior_sum : 
  interior_sum 6 = 30 → interior_sum 8 + interior_sum 9 = 380 := by
sorry

end NUMINAMATH_CALUDE_pascal_interior_sum_l2056_205645


namespace NUMINAMATH_CALUDE_point_in_second_quadrant_l2056_205698

theorem point_in_second_quadrant (A B C : Real) (h1 : 0 < A ∧ A < π/2) 
  (h2 : 0 < B ∧ B < π/2) (h3 : 0 < C ∧ C < π/2) (h4 : A + B + C = π) :
  let P : ℝ × ℝ := (Real.cos B - Real.sin A, Real.sin B - Real.cos A)
  (P.1 < 0 ∧ P.2 > 0) := by sorry

end NUMINAMATH_CALUDE_point_in_second_quadrant_l2056_205698


namespace NUMINAMATH_CALUDE_prob_two_females_is_three_tenths_l2056_205603

-- Define the total number of contestants
def total_contestants : ℕ := 5

-- Define the number of female contestants
def female_contestants : ℕ := 3

-- Define the number of contestants to be chosen
def chosen_contestants : ℕ := 2

-- Define the probability of choosing 2 female contestants
def prob_two_females : ℚ := (female_contestants.choose chosen_contestants) / (total_contestants.choose chosen_contestants)

-- Theorem statement
theorem prob_two_females_is_three_tenths : prob_two_females = 3 / 10 := by
  sorry

end NUMINAMATH_CALUDE_prob_two_females_is_three_tenths_l2056_205603


namespace NUMINAMATH_CALUDE_exterior_angle_hexagon_octagon_exterior_angle_hexagon_octagon_is_105_l2056_205673

/-- The measure of an exterior angle formed by a regular hexagon and a regular octagon sharing a common side -/
theorem exterior_angle_hexagon_octagon : ℝ :=
  let hexagon_interior_angle := (180 * (6 - 2) / 6 : ℝ)
  let octagon_interior_angle := (180 * (8 - 2) / 8 : ℝ)
  360 - (hexagon_interior_angle + octagon_interior_angle)

/-- The exterior angle formed by a regular hexagon and a regular octagon sharing a common side is 105 degrees -/
theorem exterior_angle_hexagon_octagon_is_105 :
  exterior_angle_hexagon_octagon = 105 := by
  sorry

end NUMINAMATH_CALUDE_exterior_angle_hexagon_octagon_exterior_angle_hexagon_octagon_is_105_l2056_205673


namespace NUMINAMATH_CALUDE_band_tryouts_l2056_205621

theorem band_tryouts (total_flutes total_clarinets total_trumpets total_pianists : ℕ)
  (flute_ratio clarinet_ratio trumpet_ratio : ℚ)
  (total_in_band : ℕ)
  (h1 : total_flutes = 20)
  (h2 : total_clarinets = 30)
  (h3 : total_trumpets = 60)
  (h4 : total_pianists = 20)
  (h5 : flute_ratio = 4/5)
  (h6 : clarinet_ratio = 1/2)
  (h7 : trumpet_ratio = 1/3)
  (h8 : total_in_band = 53)
  : (total_in_band - (total_flutes * flute_ratio + total_clarinets * clarinet_ratio + total_trumpets * trumpet_ratio).floor) / total_pianists = 1/10 := by
  sorry

end NUMINAMATH_CALUDE_band_tryouts_l2056_205621


namespace NUMINAMATH_CALUDE_largest_prime_and_composite_under_20_l2056_205620

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def is_composite (n : ℕ) : Prop := n > 1 ∧ ∃ d : ℕ, d > 1 ∧ d < n ∧ n % d = 0

def is_two_digit (n : ℕ) : Prop := n ≥ 10 ∧ n < 100

theorem largest_prime_and_composite_under_20 :
  (∀ n : ℕ, is_two_digit n → n < 20 → is_prime n → n ≤ 19) ∧
  (is_prime 19) ∧
  (∀ n : ℕ, is_two_digit n → n < 20 → is_composite n → n ≤ 18) ∧
  (is_composite 18) :=
sorry

end NUMINAMATH_CALUDE_largest_prime_and_composite_under_20_l2056_205620


namespace NUMINAMATH_CALUDE_zeta_sum_seventh_power_l2056_205648

theorem zeta_sum_seventh_power (ζ₁ ζ₂ ζ₃ : ℂ) 
  (h1 : ζ₁ + ζ₂ + ζ₃ = 2)
  (h2 : ζ₁^2 + ζ₂^2 + ζ₃^2 = 6)
  (h3 : ζ₁^3 + ζ₂^3 + ζ₃^3 = 8) :
  ζ₁^7 + ζ₂^7 + ζ₃^7 = 58 := by
  sorry

end NUMINAMATH_CALUDE_zeta_sum_seventh_power_l2056_205648


namespace NUMINAMATH_CALUDE_rectangle_tiling_existence_l2056_205656

/-- A rectangle is represented by its width and height -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Represents a tiling of a rectangle using smaller rectangles -/
def CanTile (r : Rectangle) (tiles : List Rectangle) : Prop :=
  sorry

/-- The main theorem: there exists an N such that all rectangles with sides > N can be tiled -/
theorem rectangle_tiling_existence : 
  ∃ N : ℕ, ∀ m n : ℕ, m > N → n > N → 
    CanTile ⟨m, n⟩ [⟨4, 6⟩, ⟨5, 7⟩] :=
  sorry

end NUMINAMATH_CALUDE_rectangle_tiling_existence_l2056_205656


namespace NUMINAMATH_CALUDE_interest_difference_theorem_l2056_205616

/-- The difference between compound and simple interest over 2 years at 10% per annum -/
def interestDifference (P : ℝ) : ℝ :=
  P * ((1 + 0.1)^2 - 1) - P * 0.1 * 2

/-- The problem statement -/
theorem interest_difference_theorem (P : ℝ) :
  interestDifference P = 18 → P = 1800 := by
  sorry

end NUMINAMATH_CALUDE_interest_difference_theorem_l2056_205616


namespace NUMINAMATH_CALUDE_abigail_savings_l2056_205688

def monthly_savings : ℕ := 4000
def months_in_year : ℕ := 12

theorem abigail_savings : monthly_savings * months_in_year = 48000 := by
  sorry

end NUMINAMATH_CALUDE_abigail_savings_l2056_205688


namespace NUMINAMATH_CALUDE_prime_square_divisibility_l2056_205601

theorem prime_square_divisibility (p : Nat) (h_prime : Prime p) (h_gt_3 : p > 3) :
  96 ∣ (4 * p^2 - 100) ↔ p^2 % 24 = 25 := by
  sorry

end NUMINAMATH_CALUDE_prime_square_divisibility_l2056_205601


namespace NUMINAMATH_CALUDE_function_symmetry_l2056_205625

def symmetric_about (f : ℝ → ℝ) (p : ℝ × ℝ) : Prop :=
  ∀ x y, f x = y ↔ f (2 * p.1 - x) = 2 * p.2 - y

theorem function_symmetry (f : ℝ → ℝ) 
    (h1 : symmetric_about f (-1, 0))
    (h2 : ∀ x > 0, f x = 1 / x) :
    ∀ x < -2, f x = 1 / (x + 2) := by
  sorry

end NUMINAMATH_CALUDE_function_symmetry_l2056_205625


namespace NUMINAMATH_CALUDE_problem_solution_l2056_205676

theorem problem_solution (m n : ℕ) (x : ℝ) 
  (h1 : 2^m = 8)
  (h2 : 2^n = 32)
  (h3 : x = 2^m - 1) :
  (2^(2*m + n - 4) = 128) ∧ 
  (1 + 4^(m+1) = 4*x^2 + 8*x + 5) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2056_205676


namespace NUMINAMATH_CALUDE_L_shape_sum_implies_all_ones_l2056_205696

/-- A type representing a 2015x2015 matrix of real numbers -/
def Matrix2015 := Fin 2015 → Fin 2015 → ℝ

/-- The L-shape property: sum of any three numbers in an L-shape is 3 -/
def has_L_shape_property (M : Matrix2015) : Prop :=
  ∀ i j k l : Fin 2015, 
    ((i = k ∧ j ≠ l) ∨ (i ≠ k ∧ j = l)) → 
    M i j + M k j + M k l = 3

/-- All elements in the matrix are 1 -/
def all_ones (M : Matrix2015) : Prop :=
  ∀ i j : Fin 2015, M i j = 1

/-- Main theorem: If a 2015x2015 matrix has the L-shape property, then all its elements are 1 -/
theorem L_shape_sum_implies_all_ones (M : Matrix2015) :
  has_L_shape_property M → all_ones M := by
  sorry


end NUMINAMATH_CALUDE_L_shape_sum_implies_all_ones_l2056_205696


namespace NUMINAMATH_CALUDE_arithmetic_mean_with_additional_number_l2056_205608

theorem arithmetic_mean_with_additional_number : 
  let numbers : List ℕ := [16, 24, 45, 63]
  let additional_number := 2 * numbers.head!
  let total_sum := numbers.sum + additional_number
  let count := numbers.length + 1
  (total_sum : ℚ) / count = 36 := by sorry

end NUMINAMATH_CALUDE_arithmetic_mean_with_additional_number_l2056_205608


namespace NUMINAMATH_CALUDE_square_difference_l2056_205619

theorem square_difference (a b : ℝ) (h1 : a + b = 6) (h2 : a - b = 2) : a^2 - b^2 = 12 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_l2056_205619


namespace NUMINAMATH_CALUDE_lowest_cost_per_pack_10_plus_cartons_cost_10_plus_lower_than_5_to_9_l2056_205618

/-- Represents the number of boxes per carton -/
def boxes_per_carton : ℕ := 15

/-- Represents the number of packs per box -/
def packs_per_box : ℕ := 12

/-- Represents the total cost for 12 cartons before discounts -/
def total_cost_12_cartons : ℝ := 3000

/-- Represents the quantity discount for 5 or more cartons -/
def quantity_discount_5_plus : ℝ := 0.10

/-- Represents the quantity discount for 10 or more cartons -/
def quantity_discount_10_plus : ℝ := 0.15

/-- Represents the gold tier membership discount -/
def gold_tier_discount : ℝ := 0.10

/-- Represents the seasonal promotion discount -/
def seasonal_discount : ℝ := 0.03

/-- Theorem stating that purchasing 10 or more cartons results in the lowest cost per pack -/
theorem lowest_cost_per_pack_10_plus_cartons :
  let cost_per_carton := total_cost_12_cartons / 12
  let packs_per_carton := boxes_per_carton * packs_per_box
  let total_discount := quantity_discount_10_plus + gold_tier_discount + seasonal_discount
  let cost_per_carton_after_discount := cost_per_carton * (1 - total_discount)
  cost_per_carton_after_discount / packs_per_carton = 1 :=
sorry

/-- Theorem stating that the cost per pack for 10 or more cartons is lower than for 5-9 cartons -/
theorem cost_10_plus_lower_than_5_to_9 :
  let cost_per_carton := total_cost_12_cartons / 12
  let packs_per_carton := boxes_per_carton * packs_per_box
  let total_discount_10_plus := quantity_discount_10_plus + gold_tier_discount + seasonal_discount
  let total_discount_5_to_9 := quantity_discount_5_plus + gold_tier_discount + seasonal_discount
  let cost_per_pack_10_plus := (cost_per_carton * (1 - total_discount_10_plus)) / packs_per_carton
  let cost_per_pack_5_to_9 := (cost_per_carton * (1 - total_discount_5_to_9)) / packs_per_carton
  cost_per_pack_10_plus < cost_per_pack_5_to_9 :=
sorry

end NUMINAMATH_CALUDE_lowest_cost_per_pack_10_plus_cartons_cost_10_plus_lower_than_5_to_9_l2056_205618


namespace NUMINAMATH_CALUDE_fifteenth_digit_of_sum_l2056_205659

def decimal_rep_1_9 : ℚ := 1 / 9
def decimal_rep_1_11 : ℚ := 1 / 11

def sum_of_reps : ℚ := decimal_rep_1_9 + decimal_rep_1_11

def nth_digit_after_decimal (q : ℚ) (n : ℕ) : ℕ := sorry

theorem fifteenth_digit_of_sum :
  nth_digit_after_decimal sum_of_reps 15 = 1 := by sorry

end NUMINAMATH_CALUDE_fifteenth_digit_of_sum_l2056_205659


namespace NUMINAMATH_CALUDE_min_coefficient_value_l2056_205668

theorem min_coefficient_value (a b : ℤ) (box : ℤ) : 
  (∀ x, (a * x + b) * (b * x + a) = 15 * x^2 + box * x + 15) →
  a ≠ b ∧ a ≠ box ∧ b ≠ box →
  ∃ (min_box : ℤ), (min_box = 34 ∧ box ≥ min_box) := by
sorry

end NUMINAMATH_CALUDE_min_coefficient_value_l2056_205668


namespace NUMINAMATH_CALUDE_not_closed_under_addition_l2056_205664

-- Define a "good set" S
def GoodSet (S : Set ℤ) : Prop :=
  ∀ a b : ℤ, (a^2 - b^2) ∈ S

-- Theorem statement
theorem not_closed_under_addition
  (S : Set ℤ) (hS : S.Nonempty) (hGood : GoodSet S) :
  ¬ (∀ x y : ℤ, x ∈ S → y ∈ S → (x + y) ∈ S) :=
sorry

end NUMINAMATH_CALUDE_not_closed_under_addition_l2056_205664


namespace NUMINAMATH_CALUDE_middle_card_is_five_l2056_205637

/-- Represents a triple of distinct positive integers in ascending order --/
structure CardTriple where
  left : Nat
  middle : Nat
  right : Nat
  distinct : left < middle ∧ middle < right
  sum_20 : left + middle + right = 20

/-- Predicate for Bella's statement --/
def bella_cant_determine (t : CardTriple) : Prop :=
  ∃ t' : CardTriple, t'.left = t.left ∧ t' ≠ t

/-- Predicate for Della's statement --/
def della_cant_determine (t : CardTriple) : Prop :=
  ∃ t' : CardTriple, t'.middle = t.middle ∧ t' ≠ t

/-- Predicate for Nella's statement --/
def nella_cant_determine (t : CardTriple) : Prop :=
  ∃ t' : CardTriple, t'.right = t.right ∧ t' ≠ t

/-- The main theorem --/
theorem middle_card_is_five :
  ∀ t : CardTriple,
    bella_cant_determine t →
    della_cant_determine t →
    nella_cant_determine t →
    t.middle = 5 := by
  sorry

end NUMINAMATH_CALUDE_middle_card_is_five_l2056_205637


namespace NUMINAMATH_CALUDE_functional_equation_solution_l2056_205689

/-- A function satisfying the given properties -/
def FunctionalEquation (g : ℝ → ℝ) : Prop :=
  (∀ x y : ℝ, g (x - y) = g x * g y) ∧
  (∀ x : ℝ, g x ≠ 0) ∧
  (g 0 = 1)

/-- Theorem stating that g(5) = e^5 for functions satisfying the given properties -/
theorem functional_equation_solution (g : ℝ → ℝ) (h : FunctionalEquation g) :
  g 5 = Real.exp 5 := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l2056_205689


namespace NUMINAMATH_CALUDE_roots_arithmetic_prog_not_all_real_implies_a_eq_neg_26_l2056_205652

/-- The polynomial whose roots we're investigating -/
def f (a : ℝ) (x : ℂ) : ℂ := x^3 - 6*x^2 + 21*x + a

/-- Condition that the roots form an arithmetic progression -/
def roots_in_arithmetic_progression (a : ℝ) : Prop :=
  ∃ (r d : ℂ), (f a (r - d) = 0) ∧ (f a r = 0) ∧ (f a (r + d) = 0)

/-- Condition that not all roots are real -/
def not_all_roots_real (a : ℝ) : Prop :=
  ∃ (z : ℂ), (f a z = 0) ∧ (z.im ≠ 0)

/-- Main theorem -/
theorem roots_arithmetic_prog_not_all_real_implies_a_eq_neg_26 :
  ∀ a : ℝ, roots_in_arithmetic_progression a ∧ not_all_roots_real a → a = -26 :=
sorry

end NUMINAMATH_CALUDE_roots_arithmetic_prog_not_all_real_implies_a_eq_neg_26_l2056_205652


namespace NUMINAMATH_CALUDE_college_strength_l2056_205635

theorem college_strength (cricket_players basketball_players both : ℕ) 
  (h1 : cricket_players = 500)
  (h2 : basketball_players = 600)
  (h3 : both = 220) :
  cricket_players + basketball_players - both = 880 :=
by sorry

end NUMINAMATH_CALUDE_college_strength_l2056_205635


namespace NUMINAMATH_CALUDE_angela_insects_l2056_205634

theorem angela_insects (dean_insects : ℕ) (jacob_insects : ℕ) (angela_insects : ℕ)
  (h1 : dean_insects = 30)
  (h2 : jacob_insects = 5 * dean_insects)
  (h3 : angela_insects = jacob_insects / 2) :
  angela_insects = 75 := by
  sorry

end NUMINAMATH_CALUDE_angela_insects_l2056_205634


namespace NUMINAMATH_CALUDE_complex_subtraction_problem_l2056_205697

theorem complex_subtraction_problem :
  (4 : ℂ) - 3*I - ((5 : ℂ) - 12*I) = -1 + 9*I := by
  sorry

end NUMINAMATH_CALUDE_complex_subtraction_problem_l2056_205697


namespace NUMINAMATH_CALUDE_laptop_sticker_price_is_750_l2056_205657

/-- The sticker price of the laptop -/
def sticker_price : ℝ := 750

/-- Store A's pricing strategy -/
def store_A_price (x : ℝ) : ℝ := 0.80 * x - 100

/-- Store B's pricing strategy -/
def store_B_price (x : ℝ) : ℝ := 0.70 * x

/-- The theorem stating that the sticker price is correct -/
theorem laptop_sticker_price_is_750 :
  store_B_price sticker_price - store_A_price sticker_price = 25 := by
  sorry

end NUMINAMATH_CALUDE_laptop_sticker_price_is_750_l2056_205657


namespace NUMINAMATH_CALUDE_rikkis_earnings_l2056_205644

/-- Represents Rikki's poetry writing and selling scenario -/
structure PoetryScenario where
  price_per_word : ℚ
  words_per_unit : ℕ
  minutes_per_unit : ℕ
  total_hours : ℕ

/-- Calculates the expected earnings for a given poetry scenario -/
def expected_earnings (scenario : PoetryScenario) : ℚ :=
  let total_minutes : ℕ := scenario.total_hours * 60
  let total_units : ℕ := total_minutes / scenario.minutes_per_unit
  let total_words : ℕ := total_units * scenario.words_per_unit
  (total_words : ℚ) * scenario.price_per_word

/-- Rikki's specific poetry scenario -/
def rikkis_scenario : PoetryScenario :=
  { price_per_word := 1 / 100
  , words_per_unit := 25
  , minutes_per_unit := 5
  , total_hours := 2 }

theorem rikkis_earnings :
  expected_earnings rikkis_scenario = 6 := by
  sorry

end NUMINAMATH_CALUDE_rikkis_earnings_l2056_205644


namespace NUMINAMATH_CALUDE_point_B_coordinates_l2056_205672

/-- Given point A and vector AB, find the coordinates of point B -/
theorem point_B_coordinates (A B : ℝ × ℝ × ℝ) (AB : ℝ × ℝ × ℝ) :
  A = (3, -1, 0) →
  AB = (2, 5, -3) →
  B = (5, 4, -3) :=
by sorry

end NUMINAMATH_CALUDE_point_B_coordinates_l2056_205672


namespace NUMINAMATH_CALUDE_tangent_parallel_to_xy_l2056_205630

-- Define the function f(x) = x^2 - x
def f (x : ℝ) : ℝ := x^2 - x

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 2 * x - 1

-- Theorem statement
theorem tangent_parallel_to_xy (P : ℝ × ℝ) :
  (P.1 = 1 ∧ P.2 = 0) ↔
  (f' P.1 = 1 ∧ P.2 = f P.1) := by
  sorry

#check tangent_parallel_to_xy

end NUMINAMATH_CALUDE_tangent_parallel_to_xy_l2056_205630


namespace NUMINAMATH_CALUDE_cosine_angle_special_vectors_l2056_205674

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

theorem cosine_angle_special_vectors (a b : E) (ha : a ≠ 0) (hb : b ≠ 0)
  (norm_a : ‖a‖ = 1) (norm_b : ‖b‖ = 1) (norm_sum : ‖a + 2 • b‖ = 1) :
  inner a b / (‖a‖ * ‖b‖) = -1 :=
sorry

end NUMINAMATH_CALUDE_cosine_angle_special_vectors_l2056_205674


namespace NUMINAMATH_CALUDE_min_y_squared_l2056_205662

/-- An isosceles trapezoid with specific properties -/
structure IsoscelesTrapezoid where
  EF : ℝ
  GH : ℝ
  y : ℝ
  is_isosceles : EF > GH
  circle_tangent : True  -- Represents the condition about the tangent circle

/-- The theorem stating the minimum value of y^2 -/
theorem min_y_squared (t : IsoscelesTrapezoid) 
  (h1 : t.EF = 72) 
  (h2 : t.GH = 45) : 
  ∃ (n : ℝ), n^2 = 486 ∧ ∀ (y : ℝ), 
    (∃ (t' : IsoscelesTrapezoid), t'.y = y ∧ t'.EF = t.EF ∧ t'.GH = t.GH) → 
    y^2 ≥ n^2 :=
sorry

end NUMINAMATH_CALUDE_min_y_squared_l2056_205662
