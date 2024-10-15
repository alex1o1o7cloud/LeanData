import Mathlib

namespace NUMINAMATH_CALUDE_cubic_equation_unique_solution_l3364_336462

theorem cubic_equation_unique_solution :
  ∀ (a : ℝ), ∃! (x : ℝ), x^3 - 2*a*x^2 + 3*a*x + a^2 - 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_unique_solution_l3364_336462


namespace NUMINAMATH_CALUDE_min_value_of_f_l3364_336411

def f (x m : ℝ) : ℝ := 2 * x^3 - 6 * x^2 + m

theorem min_value_of_f (m : ℝ) :
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, f x m = 3) ∧
  (∀ x ∈ Set.Icc (-2 : ℝ) 2, f x m ≤ 3) →
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, f x m = -37) ∧
  (∀ x ∈ Set.Icc (-2 : ℝ) 2, f x m ≥ -37) :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_f_l3364_336411


namespace NUMINAMATH_CALUDE_number_of_boys_l3364_336430

/-- Proves that the number of boys is 5 given the problem conditions -/
theorem number_of_boys (men : ℕ) (women : ℕ) (boys : ℕ) (total_earnings : ℕ) (men_wage : ℕ) :
  men = 5 →
  men = women →
  women = boys →
  total_earnings = 90 →
  men_wage = 6 →
  boys = 5 := by
sorry

end NUMINAMATH_CALUDE_number_of_boys_l3364_336430


namespace NUMINAMATH_CALUDE_chicken_flock_ratio_l3364_336474

/-- Chicken flock problem -/
theorem chicken_flock_ratio : 
  ∀ (susie_rir susie_gc britney_gc britney_total : ℕ),
  susie_rir = 11 →
  susie_gc = 6 →
  britney_gc = susie_gc / 2 →
  britney_total = (susie_rir + susie_gc) + 8 →
  ∃ (britney_rir : ℕ),
    britney_rir + britney_gc = britney_total ∧
    britney_rir = 2 * susie_rir :=
by sorry

end NUMINAMATH_CALUDE_chicken_flock_ratio_l3364_336474


namespace NUMINAMATH_CALUDE_total_towels_weight_lb_l3364_336467

-- Define the given conditions
def mary_towels : ℕ := 24
def frances_towels : ℕ := mary_towels / 4
def frances_towels_weight_oz : ℚ := 128

-- Define the weight of one towel in ounces
def towel_weight_oz : ℚ := frances_towels_weight_oz / frances_towels

-- Define the total number of towels
def total_towels : ℕ := mary_towels + frances_towels

-- Define the conversion factor from ounces to pounds
def oz_to_lb : ℚ := 1 / 16

-- Theorem to prove
theorem total_towels_weight_lb :
  (total_towels : ℚ) * towel_weight_oz * oz_to_lb = 40 :=
sorry

end NUMINAMATH_CALUDE_total_towels_weight_lb_l3364_336467


namespace NUMINAMATH_CALUDE_max_log_sin_l3364_336421

open Real

theorem max_log_sin (x : ℝ) (h : 0 < x ∧ x < π) : 
  ∃ c : ℝ, c = 0 ∧ ∀ y : ℝ, 0 < y ∧ y < π → log (sin y) ≤ c :=
by sorry

end NUMINAMATH_CALUDE_max_log_sin_l3364_336421


namespace NUMINAMATH_CALUDE_council_composition_l3364_336441

/-- Represents a member of the council -/
inductive Member
| Knight
| Liar

/-- The total number of council members -/
def total_members : Nat := 101

/-- Proposition that if any member is removed, the majority of remaining members would be liars -/
def majority_liars_if_removed (knights : Nat) (liars : Nat) : Prop :=
  ∀ (m : Member), 
    (m = Member.Knight → liars > (knights + liars - 1) / 2) ∧
    (m = Member.Liar → knights ≤ (knights + liars - 1) / 2)

theorem council_composition :
  ∃ (knights liars : Nat),
    knights + liars = total_members ∧
    majority_liars_if_removed knights liars ∧
    knights = 50 ∧ liars = 51 := by
  sorry

end NUMINAMATH_CALUDE_council_composition_l3364_336441


namespace NUMINAMATH_CALUDE_calculate_expression_solve_equation_l3364_336480

-- Problem 1
theorem calculate_expression : 2 * (-3)^2 - 4 * (-3) - 15 = 15 := by sorry

-- Problem 2
theorem solve_equation :
  ∀ x : ℚ, (4 - 2*x) / 3 - x = 1 → x = 1/5 := by sorry

end NUMINAMATH_CALUDE_calculate_expression_solve_equation_l3364_336480


namespace NUMINAMATH_CALUDE_cubic_foot_to_cubic_inches_l3364_336479

/-- Proves that 1 cubic foot equals 1728 cubic inches, given that 1 foot equals 12 inches. -/
theorem cubic_foot_to_cubic_inches :
  (1 : ℝ) * (1 : ℝ) * (1 : ℝ) = 1728 * (1 / 12 : ℝ) * (1 / 12 : ℝ) * (1 / 12 : ℝ) :=
by
  sorry

#check cubic_foot_to_cubic_inches

end NUMINAMATH_CALUDE_cubic_foot_to_cubic_inches_l3364_336479


namespace NUMINAMATH_CALUDE_cos_two_alpha_value_l3364_336408

theorem cos_two_alpha_value (α : ℝ) (h : Real.sin (π / 2 - α) = 1 / 4) : 
  Real.cos (2 * α) = -7 / 8 := by
  sorry

end NUMINAMATH_CALUDE_cos_two_alpha_value_l3364_336408


namespace NUMINAMATH_CALUDE_factor_expression_l3364_336493

theorem factor_expression (x : ℝ) : 16 * x^2 + 8 * x = 8 * x * (2 * x + 1) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l3364_336493


namespace NUMINAMATH_CALUDE_student_count_l3364_336423

theorem student_count (rank_from_right rank_from_left : ℕ) 
  (h1 : rank_from_right = 13) 
  (h2 : rank_from_left = 8) : 
  rank_from_right + rank_from_left - 1 = 20 := by
  sorry

end NUMINAMATH_CALUDE_student_count_l3364_336423


namespace NUMINAMATH_CALUDE_wallet_value_theorem_l3364_336443

/-- Represents the total value of bills in a wallet -/
def wallet_value (five_dollar_bills : ℕ) (ten_dollar_bills : ℕ) : ℕ :=
  5 * five_dollar_bills + 10 * ten_dollar_bills

/-- Theorem: The total value of 4 $5 bills and 8 $10 bills is $100 -/
theorem wallet_value_theorem : wallet_value 4 8 = 100 := by
  sorry

#eval wallet_value 4 8

end NUMINAMATH_CALUDE_wallet_value_theorem_l3364_336443


namespace NUMINAMATH_CALUDE_a_eq_zero_necessary_not_sufficient_l3364_336472

/-- A complex number is pure imaginary if its real part is zero -/
def isPureImaginary (z : ℂ) : Prop := z.re = 0

theorem a_eq_zero_necessary_not_sufficient :
  ∃ (a b : ℝ), (∀ (a' b' : ℝ), isPureImaginary (Complex.mk a' b') → a' = 0) ∧
               (∃ (a'' b'' : ℝ), a'' = 0 ∧ ¬isPureImaginary (Complex.mk a'' b'')) :=
by sorry

end NUMINAMATH_CALUDE_a_eq_zero_necessary_not_sufficient_l3364_336472


namespace NUMINAMATH_CALUDE_g_limit_pos_infinity_g_limit_neg_infinity_g_behavior_l3364_336412

/-- The polynomial function g(x) -/
def g (x : ℝ) : ℝ := 3*x^4 - 2*x^3 + x - 9

/-- Theorem stating that g(x) approaches infinity as x approaches positive infinity -/
theorem g_limit_pos_infinity : 
  Filter.Tendsto g Filter.atTop Filter.atTop :=
sorry

/-- Theorem stating that g(x) approaches infinity as x approaches negative infinity -/
theorem g_limit_neg_infinity : 
  Filter.Tendsto g Filter.atBot Filter.atTop :=
sorry

/-- Main theorem combining both limits to show the behavior of g(x) -/
theorem g_behavior : 
  (Filter.Tendsto g Filter.atTop Filter.atTop) ∧ 
  (Filter.Tendsto g Filter.atBot Filter.atTop) :=
sorry

end NUMINAMATH_CALUDE_g_limit_pos_infinity_g_limit_neg_infinity_g_behavior_l3364_336412


namespace NUMINAMATH_CALUDE_tyler_cds_l3364_336468

theorem tyler_cds (initial : ℕ) : 
  (2 / 3 : ℚ) * initial + 8 = 22 → initial = 21 := by
  sorry

end NUMINAMATH_CALUDE_tyler_cds_l3364_336468


namespace NUMINAMATH_CALUDE_total_packs_eq_51_l3364_336449

/-- The number of cookie packs sold in the first village -/
def village1_packs : ℕ := 23

/-- The number of cookie packs sold in the second village -/
def village2_packs : ℕ := 28

/-- The total number of cookie packs sold in both villages -/
def total_packs : ℕ := village1_packs + village2_packs

theorem total_packs_eq_51 : total_packs = 51 := by
  sorry

end NUMINAMATH_CALUDE_total_packs_eq_51_l3364_336449


namespace NUMINAMATH_CALUDE_smallest_two_digit_prime_with_reverse_property_l3364_336410

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def reverse_digits (n : ℕ) : ℕ :=
  let ones := n % 10
  let tens := n / 10
  ones * 10 + tens

def is_composite (n : ℕ) : Prop := n > 1 ∧ ¬(is_prime n)

theorem smallest_two_digit_prime_with_reverse_property : 
  ∀ n : ℕ, 
    n ≥ 20 ∧ n < 30 ∧ 
    is_prime n ∧
    is_composite (reverse_digits n) ∧
    (reverse_digits n % 3 = 0 ∨ reverse_digits n % 7 = 0) →
    n ≥ 21 :=
sorry

end NUMINAMATH_CALUDE_smallest_two_digit_prime_with_reverse_property_l3364_336410


namespace NUMINAMATH_CALUDE_no_real_solutions_l3364_336475

theorem no_real_solutions :
  ∀ x : ℝ, (3 * x) / (x^2 + 2*x + 4) + (4 * x) / (x^2 - 4*x + 5) ≠ 1 := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_l3364_336475


namespace NUMINAMATH_CALUDE_ratio_problem_l3364_336469

theorem ratio_problem (a b c d : ℝ) 
  (h1 : a / b = 5)
  (h2 : b / c = 1 / 2)
  (h3 : c / d = 7) :
  d / a = 2 / 35 := by
sorry

end NUMINAMATH_CALUDE_ratio_problem_l3364_336469


namespace NUMINAMATH_CALUDE_M_subset_N_l3364_336494

def M : Set ℝ := {-1, 1}

def N : Set ℝ := {x | (1 / x) < 3}

theorem M_subset_N : M ⊆ N := by
  sorry

end NUMINAMATH_CALUDE_M_subset_N_l3364_336494


namespace NUMINAMATH_CALUDE_double_sized_cube_weight_l3364_336483

/-- Given a cubical block of metal, this function calculates the weight of another cube of the same metal with sides twice as long. -/
def weight_of_double_sized_cube (original_weight : ℝ) : ℝ :=
  8 * original_weight

/-- Theorem stating that if a cubical block of metal weighs 3 pounds, then another cube of the same metal with sides twice as long will weigh 24 pounds. -/
theorem double_sized_cube_weight :
  weight_of_double_sized_cube 3 = 24 := by
  sorry

#eval weight_of_double_sized_cube 3

end NUMINAMATH_CALUDE_double_sized_cube_weight_l3364_336483


namespace NUMINAMATH_CALUDE_problem_2023_l3364_336409

theorem problem_2023 : 
  (2023^3 - 2 * 2023^2 * 2024 + 3 * 2023 * 2024^2 - 2024^3 + 1) / (2023 * 2024) = 2023 := by
  sorry

end NUMINAMATH_CALUDE_problem_2023_l3364_336409


namespace NUMINAMATH_CALUDE_quadratic_real_roots_condition_l3364_336444

theorem quadratic_real_roots_condition (k : ℝ) :
  (∃ x : ℝ, k * x^2 + 2 * x - 1 = 0) ↔ (k ≥ -1 ∧ k ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_condition_l3364_336444


namespace NUMINAMATH_CALUDE_pascals_triangle_56th_row_second_to_last_l3364_336416

theorem pascals_triangle_56th_row_second_to_last : Nat.choose 56 55 = 56 := by
  sorry

end NUMINAMATH_CALUDE_pascals_triangle_56th_row_second_to_last_l3364_336416


namespace NUMINAMATH_CALUDE_boys_employed_is_50_l3364_336407

/-- Represents the roadway construction scenario --/
structure RoadwayConstruction where
  totalLength : ℝ
  totalTime : ℝ
  initialLength : ℝ
  initialTime : ℝ
  initialMen : ℕ
  initialHours : ℝ
  overtimeHours : ℝ
  boyEfficiency : ℝ

/-- Calculates the number of boys employed in the roadway construction --/
def calculateBoysEmployed (rc : RoadwayConstruction) : ℕ :=
  sorry

/-- Theorem stating that the number of boys employed is 50 --/
theorem boys_employed_is_50 (rc : RoadwayConstruction) : 
  rc.totalLength = 15 ∧ 
  rc.totalTime = 40 ∧ 
  rc.initialLength = 3 ∧ 
  rc.initialTime = 10 ∧ 
  rc.initialMen = 180 ∧ 
  rc.initialHours = 8 ∧ 
  rc.overtimeHours = 1 ∧ 
  rc.boyEfficiency = 2/3 → 
  calculateBoysEmployed rc = 50 := by
  sorry

end NUMINAMATH_CALUDE_boys_employed_is_50_l3364_336407


namespace NUMINAMATH_CALUDE_min_distance_B_to_M_l3364_336400

-- Define the rectilinear distance function
def rectilinearDistance (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  |x₁ - x₂| + |y₁ - y₂|

-- Define the point B
def B : ℝ × ℝ := (1, 1)

-- Define the line on which M moves
def lineM (x y : ℝ) : Prop :=
  x - y + 4 = 0

-- Theorem statement
theorem min_distance_B_to_M :
  ∃ (min : ℝ), min = 4 ∧
  ∀ (x y : ℝ), lineM x y →
    rectilinearDistance B.1 B.2 x y ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_distance_B_to_M_l3364_336400


namespace NUMINAMATH_CALUDE_max_value_abc_l3364_336450

theorem max_value_abc (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : a + b + c = 3) :
  a + b^2 + c^4 ≤ 3 := by
sorry

end NUMINAMATH_CALUDE_max_value_abc_l3364_336450


namespace NUMINAMATH_CALUDE_cost_of_balls_and_shuttlecocks_l3364_336459

/-- The cost of ping-pong balls and badminton shuttlecocks -/
theorem cost_of_balls_and_shuttlecocks 
  (ping_pong : ℝ) 
  (shuttlecock : ℝ) 
  (h1 : 3 * ping_pong + 2 * shuttlecock = 15.5)
  (h2 : 2 * ping_pong + 3 * shuttlecock = 17) :
  4 * ping_pong + 4 * shuttlecock = 26 :=
by sorry

end NUMINAMATH_CALUDE_cost_of_balls_and_shuttlecocks_l3364_336459


namespace NUMINAMATH_CALUDE_prob_diff_color_is_29_50_l3364_336476

-- Define the contents of the boxes
def boxA : Finset (Fin 3) := {0, 0, 1, 1, 2}
def boxB : Finset (Fin 3) := {0, 0, 0, 0, 1, 1, 1, 2, 2}

-- Define the probability of drawing a ball of a different color
def prob_diff_color : ℚ :=
  let total_A := boxA.card
  let total_B := boxB.card + 1
  let prob_white := (boxA.filter (· = 0)).card / total_A *
                    (boxB.filter (· ≠ 0)).card / total_B
  let prob_red := (boxA.filter (· = 1)).card / total_A *
                  (boxB.filter (· ≠ 1)).card / total_B
  let prob_black := (boxA.filter (· = 2)).card / total_A *
                    (boxB.filter (· ≠ 2)).card / total_B
  prob_white + prob_red + prob_black

-- Theorem statement
theorem prob_diff_color_is_29_50 : prob_diff_color = 29 / 50 := by
  sorry

end NUMINAMATH_CALUDE_prob_diff_color_is_29_50_l3364_336476


namespace NUMINAMATH_CALUDE_randy_piggy_bank_l3364_336435

/-- Calculates the remaining money in Randy's piggy bank after a year -/
theorem randy_piggy_bank (initial_amount : ℕ) (spend_per_trip : ℕ) (trips_per_month : ℕ) (months : ℕ) :
  initial_amount = 200 →
  spend_per_trip = 2 →
  trips_per_month = 4 →
  months = 12 →
  initial_amount - (spend_per_trip * trips_per_month * months) = 104 := by
  sorry

#check randy_piggy_bank

end NUMINAMATH_CALUDE_randy_piggy_bank_l3364_336435


namespace NUMINAMATH_CALUDE_largest_two_digit_prime_factor_of_binomial_l3364_336428

theorem largest_two_digit_prime_factor_of_binomial :
  ∃ (p : ℕ), p = 73 ∧ 
    Prime p ∧ 
    10 ≤ p ∧ p < 100 ∧
    p ∣ (Nat.choose 150 75) ∧
    ∀ q : ℕ, Prime q → 10 ≤ q → q < 100 → q ∣ (Nat.choose 150 75) → q ≤ p :=
by sorry


end NUMINAMATH_CALUDE_largest_two_digit_prime_factor_of_binomial_l3364_336428


namespace NUMINAMATH_CALUDE_peytons_score_l3364_336405

theorem peytons_score (n : ℕ) (avg_14 : ℚ) (avg_15 : ℚ) (peyton_score : ℚ) : 
  n = 15 → 
  avg_14 = 80 → 
  avg_15 = 81 → 
  (n - 1) * avg_14 + peyton_score = n * avg_15 →
  peyton_score = 95 := by
  sorry

end NUMINAMATH_CALUDE_peytons_score_l3364_336405


namespace NUMINAMATH_CALUDE_monotonicity_and_slope_conditions_l3364_336413

-- Define the function f
def f (a b x : ℝ) : ℝ := -x^3 + x^2 + a*x + b

-- Define the derivative of f
def f' (a x : ℝ) : ℝ := -3*x^2 + 2*x + a

theorem monotonicity_and_slope_conditions (a b : ℝ) :
  -- Part 1: Monotonicity when a = 3
  (∀ x ∈ Set.Ioo (-1 : ℝ) 3, (f' 3 x > 0)) ∧
  (∀ x ∈ Set.Iic (-1 : ℝ), (f' 3 x < 0)) ∧
  (∀ x ∈ Set.Ici 3, (f' 3 x < 0)) ∧
  -- Part 2: Condition on a based on slope
  ((∀ x : ℝ, f' a x < 2*a^2) → (a > 1 ∨ a < -1/2)) :=
sorry

end NUMINAMATH_CALUDE_monotonicity_and_slope_conditions_l3364_336413


namespace NUMINAMATH_CALUDE_num_pencils_is_75_l3364_336401

/-- The number of pencils purchased given the conditions of the problem -/
def num_pencils : ℕ :=
  let num_pens : ℕ := 30
  let total_cost : ℕ := 570
  let pencil_price : ℕ := 2
  let pen_price : ℕ := 14
  let pen_cost : ℕ := num_pens * pen_price
  let pencil_cost : ℕ := total_cost - pen_cost
  pencil_cost / pencil_price

theorem num_pencils_is_75 : num_pencils = 75 := by
  sorry

end NUMINAMATH_CALUDE_num_pencils_is_75_l3364_336401


namespace NUMINAMATH_CALUDE_correct_product_l3364_336486

theorem correct_product (a b : ℕ) : 
  10 ≤ a ∧ a < 100 →  -- a is a two-digit number
  (∃ x y : ℕ, x * 10 + y = a ∧ y * 10 + x = (189 / b)) →  -- reversing digits of a and multiplying by b gives 189
  a * b = 108 := by
sorry

end NUMINAMATH_CALUDE_correct_product_l3364_336486


namespace NUMINAMATH_CALUDE_quadratic_unique_solution_l3364_336466

-- Define the quadratic equation
def quadratic (a x : ℝ) : ℝ := x^2 + (2-a)*x + 1

-- Define the solution range
def in_range (x : ℝ) : Prop := -1 < x ∧ x ≤ 3 ∧ x ≠ 0 ∧ x ≠ 1 ∧ x ≠ 2

-- Define the uniqueness of the solution
def unique_solution (a : ℝ) : Prop :=
  ∃! x, quadratic a x = 0 ∧ in_range x

-- State the theorem
theorem quadratic_unique_solution :
  ∀ a : ℝ, unique_solution a ↔ 
    (a = 4.5) ∨ 
    (a < 0) ∨ 
    (a > 16/3) :=
sorry

end NUMINAMATH_CALUDE_quadratic_unique_solution_l3364_336466


namespace NUMINAMATH_CALUDE_sum_of_four_real_numbers_l3364_336419

theorem sum_of_four_real_numbers (a b c d : ℝ) 
  (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) 
  (h_eq : (a^2 + b^2 - 1)*(a + b) = (b^2 + c^2 - 1)*(b + c) ∧ 
          (b^2 + c^2 - 1)*(b + c) = (c^2 + d^2 - 1)*(c + d)) : 
  a + b + c + d = 0 := by
sorry

end NUMINAMATH_CALUDE_sum_of_four_real_numbers_l3364_336419


namespace NUMINAMATH_CALUDE_largest_n_binomial_sum_l3364_336471

theorem largest_n_binomial_sum (n : ℕ) : 
  (Nat.choose 12 5 + Nat.choose 12 6 = Nat.choose 13 n) → n ≤ 7 :=
by sorry

end NUMINAMATH_CALUDE_largest_n_binomial_sum_l3364_336471


namespace NUMINAMATH_CALUDE_factorization_of_cubic_minus_linear_l3364_336424

theorem factorization_of_cubic_minus_linear (x : ℝ) :
  3 * x^3 - 12 * x = 3 * x * (x - 2) * (x + 2) := by sorry

end NUMINAMATH_CALUDE_factorization_of_cubic_minus_linear_l3364_336424


namespace NUMINAMATH_CALUDE_tom_tickets_l3364_336420

/-- The number of tickets Tom has left after playing games and spending some tickets -/
def tickets_left (whack_a_mole skee_ball ring_toss hat plush_toy : ℕ) : ℕ :=
  (whack_a_mole + skee_ball + ring_toss) - (hat + plush_toy)

/-- Theorem stating that Tom is left with 100 tickets -/
theorem tom_tickets : 
  tickets_left 45 38 52 12 23 = 100 := by
  sorry

end NUMINAMATH_CALUDE_tom_tickets_l3364_336420


namespace NUMINAMATH_CALUDE_solutions_equation1_solutions_equation2_l3364_336426

-- Define the equations
def equation1 (x : ℝ) : Prop := x^2 - 5*x + 6 = 0
def equation2 (x : ℝ) : Prop := (x + 2)*(x - 1) = x + 2

-- Theorem for equation1
theorem solutions_equation1 :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ equation1 x₁ ∧ equation1 x₂ ∧ x₁ = 3 ∧ x₂ = 2 :=
sorry

-- Theorem for equation2
theorem solutions_equation2 :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ equation2 x₁ ∧ equation2 x₂ ∧ x₁ = -2 ∧ x₂ = 2 :=
sorry

end NUMINAMATH_CALUDE_solutions_equation1_solutions_equation2_l3364_336426


namespace NUMINAMATH_CALUDE_f_derivative_at_2_l3364_336440

-- Define the function f
def f (x : ℝ) (k : ℝ) : ℝ := x^3 - k*x^2 + 3*x - 5

-- State the theorem
theorem f_derivative_at_2 : 
  ∃ k : ℝ, (∀ x, deriv (f · k) x = 3*x^2 - 2*k*x + 3) ∧ deriv (f · k) 2 = k ∧ k = 3 := by
  sorry

end NUMINAMATH_CALUDE_f_derivative_at_2_l3364_336440


namespace NUMINAMATH_CALUDE_geometric_sequence_a1_value_l3364_336454

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (q : ℝ), q > 0 ∧ ∀ n, a (n + 1) = a n * q

theorem geometric_sequence_a1_value
  (a : ℕ → ℝ)
  (h_geom : geometric_sequence a)
  (h_a3 : a 3 = 1)
  (h_mean : (a 5 + (3/2) * a 4) / 2 = 1/2) :
  a 1 = 4 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_a1_value_l3364_336454


namespace NUMINAMATH_CALUDE_tunnel_crossing_possible_l3364_336404

/-- Represents a friend with their crossing time -/
structure Friend where
  name : String
  time : Nat

/-- Represents a crossing of the tunnel -/
inductive Crossing
  | Forward : List Friend → Crossing
  | Backward : Friend → Crossing

/-- Calculates the time taken for a crossing -/
def crossingTime (c : Crossing) : Nat :=
  match c with
  | Crossing.Forward friends => friends.map Friend.time |>.maximum?.getD 0
  | Crossing.Backward friend => friend.time

/-- The tunnel crossing problem -/
def tunnelCrossing (friends : List Friend) : Prop :=
  ∃ (crossings : List Crossing),
    -- All friends have crossed
    (crossings.filter (λ c => match c with
      | Crossing.Forward _ => true
      | Crossing.Backward _ => false
    )).bind (λ c => match c with
      | Crossing.Forward fs => fs
      | Crossing.Backward _ => []
    ) = friends
    ∧
    -- The total time is exactly 17 minutes
    (crossings.map crossingTime).sum = 17
    ∧
    -- Each crossing involves at most two friends
    ∀ c ∈ crossings, match c with
      | Crossing.Forward fs => fs.length ≤ 2
      | Crossing.Backward _ => true

theorem tunnel_crossing_possible : 
  let friends := [
    { name := "One", time := 1 },
    { name := "Two", time := 2 },
    { name := "Five", time := 5 },
    { name := "Ten", time := 10 }
  ]
  tunnelCrossing friends :=
by
  sorry


end NUMINAMATH_CALUDE_tunnel_crossing_possible_l3364_336404


namespace NUMINAMATH_CALUDE_jessicas_allowance_l3364_336447

theorem jessicas_allowance (allowance : ℝ) : 
  (allowance / 2 + 6 = 11) → allowance = 10 := by
  sorry

end NUMINAMATH_CALUDE_jessicas_allowance_l3364_336447


namespace NUMINAMATH_CALUDE_g_comp_four_roots_l3364_336453

/-- The function g(x) defined as x^2 + 8x + d -/
def g (d : ℝ) (x : ℝ) : ℝ := x^2 + 8*x + d

/-- The composition of g with itself -/
def g_comp (d : ℝ) (x : ℝ) : ℝ := g d (g d x)

/-- Theorem stating that g(g(x)) has exactly 4 distinct real roots iff d < 4 -/
theorem g_comp_four_roots (d : ℝ) :
  (∃ (x₁ x₂ x₃ x₄ : ℝ), x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄ ∧
    g_comp d x₁ = 0 ∧ g_comp d x₂ = 0 ∧ g_comp d x₃ = 0 ∧ g_comp d x₄ = 0 ∧
    ∀ (y : ℝ), g_comp d y = 0 → y = x₁ ∨ y = x₂ ∨ y = x₃ ∨ y = x₄) ↔
  d < 4 :=
sorry

end NUMINAMATH_CALUDE_g_comp_four_roots_l3364_336453


namespace NUMINAMATH_CALUDE_rhombus_acute_angle_l3364_336489

/-- Given a rhombus, prove that its acute angle is arccos(1/9) when the ratio of volumes of rotation is 1:2√5 -/
theorem rhombus_acute_angle (a : ℝ) (h : a > 0) : 
  let α := Real.arccos (1/9)
  let V₁ := (1/3) * π * (a * Real.sin (α/2))^2 * (2 * a * Real.cos (α/2))
  let V₂ := π * (a * Real.sin α)^2 * a
  V₁ / V₂ = 1 / (2 * Real.sqrt 5) → 
  α = Real.arccos (1/9) := by
sorry

end NUMINAMATH_CALUDE_rhombus_acute_angle_l3364_336489


namespace NUMINAMATH_CALUDE_matrix_power_4_l3364_336463

def A : Matrix (Fin 2) (Fin 2) ℤ := !![2, -1; 1, 0]

theorem matrix_power_4 :
  A ^ 4 = !![5, -4; 4, -3] := by sorry

end NUMINAMATH_CALUDE_matrix_power_4_l3364_336463


namespace NUMINAMATH_CALUDE_amount_over_limit_l3364_336490

/-- Calculates the amount spent over a given limit when purchasing a necklace and a book,
    where the book costs $5 more than the necklace. -/
theorem amount_over_limit (necklace_cost book_cost limit : ℕ) : 
  necklace_cost = 34 →
  book_cost = necklace_cost + 5 →
  limit = 70 →
  (necklace_cost + book_cost) - limit = 3 := by
sorry


end NUMINAMATH_CALUDE_amount_over_limit_l3364_336490


namespace NUMINAMATH_CALUDE_absolute_value_plus_exponent_l3364_336482

theorem absolute_value_plus_exponent : |(-2 : ℝ)| + (π - 3)^(0 : ℝ) = 3 := by sorry

end NUMINAMATH_CALUDE_absolute_value_plus_exponent_l3364_336482


namespace NUMINAMATH_CALUDE_geometric_arithmetic_sequence_problem_l3364_336437

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

def is_arithmetic_sequence (b : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, b (n + 1) = b n + d

theorem geometric_arithmetic_sequence_problem
  (a b : ℕ → ℝ)
  (h_geometric : is_geometric_sequence a)
  (h_arithmetic : is_arithmetic_sequence b)
  (h_a_prod : a 1 * a 6 * a 11 = -3 * Real.sqrt 3)
  (h_b_sum : b 1 + b 6 + b 11 = 7 * Real.pi) :
  Real.tan ((b 3 + b 9) / (1 - a 4 * a 8)) = -Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_geometric_arithmetic_sequence_problem_l3364_336437


namespace NUMINAMATH_CALUDE_hyperbola_point_k_l3364_336434

/-- Given a point P(-3, 1) on the hyperbola y = k/x where k ≠ 0, prove that k = -3 -/
theorem hyperbola_point_k (k : ℝ) (h1 : k ≠ 0) (h2 : (1 : ℝ) = k / (-3)) : k = -3 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_point_k_l3364_336434


namespace NUMINAMATH_CALUDE_only_set_D_forms_triangle_l3364_336481

/-- Checks if three lengths can form a triangle according to the triangle inequality theorem -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- The sets of line segments given in the problem -/
def segment_sets : List (ℝ × ℝ × ℝ) :=
  [(3, 4, 8), (5, 6, 11), (3, 1, 1), (3, 4, 6)]

/-- Theorem stating that only the set (3, 4, 6) can form a triangle -/
theorem only_set_D_forms_triangle :
  ∃! (a b c : ℝ), (a, b, c) ∈ segment_sets ∧ can_form_triangle a b c :=
by sorry

end NUMINAMATH_CALUDE_only_set_D_forms_triangle_l3364_336481


namespace NUMINAMATH_CALUDE_smallest_n_with_seven_in_squares_l3364_336495

def contains_seven (n : ℕ) : Prop :=
  ∃ d k, n = 10 * k + 7 * d ∧ d ≤ 9

theorem smallest_n_with_seven_in_squares : 
  ∀ n : ℕ, n < 26 → ¬(contains_seven (n^2) ∧ contains_seven ((n+1)^2)) ∧
  (contains_seven (26^2) ∧ contains_seven (27^2)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_with_seven_in_squares_l3364_336495


namespace NUMINAMATH_CALUDE_closest_point_parabola_to_line_l3364_336403

/-- The point (1, 1) on the parabola y^2 = x is the closest point to the line x - 2y + 4 = 0 -/
theorem closest_point_parabola_to_line :
  let parabola := {p : ℝ × ℝ | p.2^2 = p.1}
  let line := {p : ℝ × ℝ | p.1 - 2*p.2 + 4 = 0}
  let distance (p : ℝ × ℝ) := |p.1 - 2*p.2 + 4| / Real.sqrt 5
  ∀ p ∈ parabola, distance (1, 1) ≤ distance p :=
by sorry

end NUMINAMATH_CALUDE_closest_point_parabola_to_line_l3364_336403


namespace NUMINAMATH_CALUDE_trigonometric_identity_l3364_336473

theorem trigonometric_identity (a b x : ℝ) (ha : a ≠ 0) (hb : b ≠ 0)
  (h : Real.sin x ^ 4 / a ^ 2 + Real.cos x ^ 4 / b ^ 2 = 1 / (a ^ 2 + b ^ 2)) :
  Real.sin x ^ 2008 / a ^ 2006 + Real.cos x ^ 2008 / b ^ 2006 = 1 / (a ^ 2 + b ^ 2) ^ 1003 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l3364_336473


namespace NUMINAMATH_CALUDE_sum_lower_bound_l3364_336487

theorem sum_lower_bound (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a * b = a + b + 3) :
  a + b ≥ 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_lower_bound_l3364_336487


namespace NUMINAMATH_CALUDE_intersection_point_l3364_336485

-- Define the system of equations
def line1 (x y : ℚ) : Prop := 6 * x - 3 * y = 18
def line2 (x y : ℚ) : Prop := 8 * x + 2 * y = 20

-- State the theorem
theorem intersection_point :
  ∃! p : ℚ × ℚ, line1 p.1 p.2 ∧ line2 p.1 p.2 ∧ p = (8/3, -2/3) := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_l3364_336485


namespace NUMINAMATH_CALUDE_coloring_scheme_exists_l3364_336414

-- Define the color type
inductive Color
| White
| Red
| Black

-- Define the coloring function type
def ColoringFunction := ℤ × ℤ → Color

-- Statement of the theorem
theorem coloring_scheme_exists : ∃ (f : ColoringFunction),
  (∀ c : Color, ∃ (S : Set ℤ), Set.Infinite S ∧ 
    ∀ y ∈ S, Set.Infinite {x : ℤ | f (x, y) = c}) ∧
  (∀ (x₁ y₁ x₂ y₂ x₃ y₃ : ℤ),
    f (x₁, y₁) = Color.White →
    f (x₂, y₂) = Color.Black →
    f (x₃, y₃) = Color.Red →
    f (x₁ + x₂ - x₃, y₁ + y₂ - y₃) = Color.Red) :=
by sorry


end NUMINAMATH_CALUDE_coloring_scheme_exists_l3364_336414


namespace NUMINAMATH_CALUDE_monotonic_decreasing_interval_l3364_336402

-- Define the function f(x) = x^3 - 3x + 1
def f (x : ℝ) : ℝ := x^3 - 3*x + 1

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3*x^2 - 3

-- Theorem statement
theorem monotonic_decreasing_interval :
  ∀ x : ℝ, (x > -1 ∧ x < 1) ↔ (∀ y : ℝ, y > x → f y < f x) :=
by sorry

end NUMINAMATH_CALUDE_monotonic_decreasing_interval_l3364_336402


namespace NUMINAMATH_CALUDE_triangle_third_side_length_l3364_336455

theorem triangle_third_side_length (a b : ℝ) (cos_theta : ℝ) : 
  a = 5 → b = 3 → 
  (5 * cos_theta^2 - 7 * cos_theta - 6 = 0) →
  ∃ c : ℝ, c^2 = a^2 + b^2 - 2 * a * b * cos_theta ∧ c = 2 * Real.sqrt 13 :=
by sorry

end NUMINAMATH_CALUDE_triangle_third_side_length_l3364_336455


namespace NUMINAMATH_CALUDE_traffic_light_theorem_l3364_336497

structure TrafficLightSystem where
  p1 : ℝ
  p2 : ℝ
  p3 : ℝ
  h1 : 0 ≤ p1 ∧ p1 ≤ 1
  h2 : 0 ≤ p2 ∧ p2 ≤ 1
  h3 : 0 ≤ p3 ∧ p3 ≤ 1
  h4 : p1 < p2
  h5 : p2 < p3
  h6 : p1 = 1/2
  h7 : (1 - p1) * (1 - p2) * (1 - p3) = 1/24
  h8 : p1 * p2 * p3 = 1/4

def prob_first_red_at_third (s : TrafficLightSystem) : ℝ :=
  (1 - s.p1) * (1 - s.p2) * s.p3

def expected_red_lights (s : TrafficLightSystem) : ℝ :=
  s.p1 + s.p2 + s.p3

theorem traffic_light_theorem (s : TrafficLightSystem) :
  prob_first_red_at_third s = 1/8 ∧ expected_red_lights s = 23/12 := by
  sorry

end NUMINAMATH_CALUDE_traffic_light_theorem_l3364_336497


namespace NUMINAMATH_CALUDE_tysons_age_l3364_336492

/-- Given the ages and relationships between Kyle, Julian, Frederick, and Tyson, prove Tyson's age --/
theorem tysons_age (kyle_age julian_age frederick_age tyson_age : ℕ) : 
  kyle_age = 25 →
  kyle_age = julian_age + 5 →
  frederick_age = julian_age + 20 →
  frederick_age = 2 * tyson_age →
  tyson_age = 20 := by
  sorry

end NUMINAMATH_CALUDE_tysons_age_l3364_336492


namespace NUMINAMATH_CALUDE_quadratic_no_real_roots_l3364_336460

theorem quadratic_no_real_roots (a : ℝ) : 
  (∀ x : ℝ, a * x^2 + 6 * x + 1 ≠ 0) → a > 9 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_no_real_roots_l3364_336460


namespace NUMINAMATH_CALUDE_symmetric_sine_function_value_l3364_336496

theorem symmetric_sine_function_value (a φ : ℝ) :
  let f : ℝ → ℝ := λ x => Real.sin (3 * x + φ)
  (∀ x, f (a + x) = f (a - x)) →
  f (a + π / 6) = 0 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_sine_function_value_l3364_336496


namespace NUMINAMATH_CALUDE_largest_divisor_of_product_l3364_336451

theorem largest_divisor_of_product (n : ℕ) (h : Odd n) :
  (∃ (k : ℕ), (n + 2) * (n + 4) * (n + 6) * (n + 8) * (n + 10) = 480 * k) ∧
  (∀ (m : ℕ), m > 480 → ∃ (n : ℕ), Odd n ∧ ¬(∃ (k : ℕ), (n + 2) * (n + 4) * (n + 6) * (n + 8) * (n + 10) = m * k)) :=
sorry

end NUMINAMATH_CALUDE_largest_divisor_of_product_l3364_336451


namespace NUMINAMATH_CALUDE_intersection_point_l3364_336484

def line1 (x y : ℚ) : Prop := 12 * x - 5 * y = 8
def line2 (x y : ℚ) : Prop := 10 * x + 2 * y = 20

theorem intersection_point :
  ∃! p : ℚ × ℚ, line1 p.1 p.2 ∧ line2 p.1 p.2 ∧ p = (58/37, 667/370) := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_l3364_336484


namespace NUMINAMATH_CALUDE_fraction_division_equality_l3364_336439

theorem fraction_division_equality : (8/9 - 5/6 + 2/3) / (-5/18) = -13/5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_division_equality_l3364_336439


namespace NUMINAMATH_CALUDE_min_valid_positions_l3364_336448

/-- Represents a disk with a certain number of sectors and red sectors. -/
structure Disk :=
  (total_sectors : ℕ)
  (red_sectors : ℕ)
  (h_red_le_total : red_sectors ≤ total_sectors)

/-- Represents the configuration of two overlapping disks. -/
structure DiskOverlay :=
  (disk1 : Disk)
  (disk2 : Disk)
  (h_same_sectors : disk1.total_sectors = disk2.total_sectors)

/-- Calculates the number of positions with at most 20 overlapping red sectors. -/
def count_valid_positions (overlay : DiskOverlay) : ℕ :=
  overlay.disk1.total_sectors - (overlay.disk1.red_sectors * overlay.disk2.red_sectors) / 21 + 1

theorem min_valid_positions (overlay : DiskOverlay) 
  (h_total : overlay.disk1.total_sectors = 1965)
  (h_red1 : overlay.disk1.red_sectors = 200)
  (h_red2 : overlay.disk2.red_sectors = 200) :
  count_valid_positions overlay = 61 :=
sorry

end NUMINAMATH_CALUDE_min_valid_positions_l3364_336448


namespace NUMINAMATH_CALUDE_unique_solution_for_P_equals_2C_l3364_336499

def P (r n : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - r)

def C (n r : ℕ) : ℕ := Nat.factorial n / (Nat.factorial r * Nat.factorial (n - r))

theorem unique_solution_for_P_equals_2C (n : ℕ+) : 
  P 8 n = 2 * C 8 2 ↔ n = 2 := by sorry

end NUMINAMATH_CALUDE_unique_solution_for_P_equals_2C_l3364_336499


namespace NUMINAMATH_CALUDE_folded_square_distance_l3364_336491

/-- Given a square sheet of paper with area 18 cm², prove that when folded so that
    a corner touches the line from midpoint of adjacent side to opposite corner,
    creating equal visible areas, the distance from corner to original position is 3 cm. -/
theorem folded_square_distance (s : ℝ) (h1 : s^2 = 18) : 
  let d := s * Real.sqrt 2 / 2
  d = 3 := by sorry

end NUMINAMATH_CALUDE_folded_square_distance_l3364_336491


namespace NUMINAMATH_CALUDE_work_completion_time_l3364_336431

/-- The number of days it takes for A to complete the work alone -/
def days_A : ℝ := 6

/-- The number of days it takes for B to complete the work alone -/
def days_B : ℝ := 12

/-- The number of days it takes for A and B to complete the work together -/
def days_AB : ℝ := 4

/-- Theorem stating that given the time for B and the time for A and B together, 
    we can determine the time for A alone -/
theorem work_completion_time : 
  (1 / days_A + 1 / days_B = 1 / days_AB) → days_A = 6 := by sorry

end NUMINAMATH_CALUDE_work_completion_time_l3364_336431


namespace NUMINAMATH_CALUDE_game_draw_probability_l3364_336488

/-- In a game between two players, given the probabilities of not losing and losing for each player, 
    we can calculate the probability of a draw. -/
theorem game_draw_probability (p_not_losing p_losing : ℚ) : 
  p_not_losing = 3/4 → p_losing = 1/2 → p_not_losing - p_losing = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_game_draw_probability_l3364_336488


namespace NUMINAMATH_CALUDE_expenditure_recording_l3364_336464

/-- Represents the way a transaction is recorded in accounting -/
inductive AccountingRecord
  | Positive (amount : ℤ)
  | Negative (amount : ℤ)

/-- Records an income transaction -/
def recordIncome (amount : ℤ) : AccountingRecord :=
  AccountingRecord.Positive amount

/-- Records an expenditure transaction -/
def recordExpenditure (amount : ℤ) : AccountingRecord :=
  AccountingRecord.Negative amount

/-- The accounting principle for recording transactions -/
axiom accounting_principle (amount : ℤ) :
  (recordIncome amount = AccountingRecord.Positive amount) ∧
  (recordExpenditure amount = AccountingRecord.Negative amount)

/-- Theorem: An expenditure of 100 should be recorded as -100 -/
theorem expenditure_recording :
  recordExpenditure 100 = AccountingRecord.Negative 100 := by
  sorry

end NUMINAMATH_CALUDE_expenditure_recording_l3364_336464


namespace NUMINAMATH_CALUDE_product_real_imag_parts_l3364_336470

theorem product_real_imag_parts : ∃ (z : ℂ), 
  z = (2 + 3*Complex.I) / (1 + Complex.I) ∧ 
  (z.re * z.im = 5/4) := by sorry

end NUMINAMATH_CALUDE_product_real_imag_parts_l3364_336470


namespace NUMINAMATH_CALUDE_largest_N_for_dispersive_connective_perm_l3364_336406

/-- The set of residues modulo 17 -/
def X : Set ℕ := {x | x < 17}

/-- Two numbers in X are adjacent if they differ by 1 or are 0 and 16 -/
def adjacent (a b : ℕ) : Prop :=
  (a ∈ X ∧ b ∈ X) ∧ ((a + 1 ≡ b [ZMOD 17]) ∨ (b + 1 ≡ a [ZMOD 17]))

/-- A permutation on X -/
def permutation_on_X (p : ℕ → ℕ) : Prop :=
  Function.Bijective p ∧ ∀ x, x ∈ X → p x ∈ X

/-- A permutation is dispersive if it never maps adjacent values to adjacent values -/
def dispersive (p : ℕ → ℕ) : Prop :=
  permutation_on_X p ∧ ∀ a b, adjacent a b → ¬adjacent (p a) (p b)

/-- A permutation is connective if it always maps adjacent values to adjacent values -/
def connective (p : ℕ → ℕ) : Prop :=
  permutation_on_X p ∧ ∀ a b, adjacent a b → adjacent (p a) (p b)

/-- The composition of a permutation with itself n times -/
def iterate_perm (p : ℕ → ℕ) : ℕ → (ℕ → ℕ)
  | 0 => id
  | n + 1 => p ∘ (iterate_perm p n)

/-- The theorem stating the largest N for which the described permutation exists -/
theorem largest_N_for_dispersive_connective_perm :
  ∃ (p : ℕ → ℕ), permutation_on_X p ∧
    (∀ k < 8, dispersive (iterate_perm p k)) ∧
    connective (iterate_perm p 8) ∧
    ∀ (q : ℕ → ℕ) (m : ℕ),
      (permutation_on_X q ∧
       (∀ k < m, dispersive (iterate_perm q k)) ∧
       connective (iterate_perm q m)) →
      m ≤ 8 :=
sorry

end NUMINAMATH_CALUDE_largest_N_for_dispersive_connective_perm_l3364_336406


namespace NUMINAMATH_CALUDE_age_ratio_l3364_336433

def tom_age : ℝ := 40.5
def total_age : ℝ := 54

theorem age_ratio : 
  let antonette_age := total_age - tom_age
  tom_age / antonette_age = 3 := by sorry

end NUMINAMATH_CALUDE_age_ratio_l3364_336433


namespace NUMINAMATH_CALUDE_sin_squared_minus_cos_squared_l3364_336436

theorem sin_squared_minus_cos_squared (α : Real) (h : Real.sin α = Real.sqrt 5 / 5) :
  Real.sin α ^ 2 - Real.cos α ^ 2 = -3/5 := by
  sorry

end NUMINAMATH_CALUDE_sin_squared_minus_cos_squared_l3364_336436


namespace NUMINAMATH_CALUDE_symbol_equations_l3364_336415

theorem symbol_equations :
  ∀ (triangle circle square star : ℤ),
  triangle = circle + 2 →
  square = triangle + triangle →
  star = triangle + square + 5 →
  star = circle + 31 →
  triangle = 12 ∧ circle = 10 ∧ square = 24 ∧ star = 41 := by
sorry

end NUMINAMATH_CALUDE_symbol_equations_l3364_336415


namespace NUMINAMATH_CALUDE_work_completion_time_l3364_336425

/-- Given that A can do a work in 15 days and when A and B work together for 4 days
    they complete 0.4666666666666667 of the work, prove that B can do the work alone in 20 days. -/
theorem work_completion_time (a b : ℝ) (ha : a = 1 / 15) 
    (h_together : 4 * (a + 1 / b) = 0.4666666666666667) : b = 20 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l3364_336425


namespace NUMINAMATH_CALUDE_unique_solution_sequence_l3364_336438

theorem unique_solution_sequence (n : ℕ) (hn : n ≥ 4) :
  ∃! (a : ℕ → ℝ),
    (∀ i, i ∈ Finset.range (2 * n) → a i > 0) ∧
    (∀ k, k ∈ Finset.range n →
      a (2 * k) = a (2 * k - 1) + a ((2 * k + 1) % (2 * n))) ∧
    (∀ k, k ∈ Finset.range n →
      a (2 * k - 1) = 1 / a ((2 * k - 2) % (2 * n)) + 1 / a (2 * k)) ∧
    (∀ i, i ∈ Finset.range (2 * n) → a i = if i % 2 = 0 then 2 else 1) :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_sequence_l3364_336438


namespace NUMINAMATH_CALUDE_hyperbola_vertex_distance_l3364_336417

/-- The distance between the vertices of a hyperbola with equation x^2/36 - y^2/25 = 1 is 12 -/
theorem hyperbola_vertex_distance : 
  let a : ℝ := Real.sqrt 36
  let b : ℝ := Real.sqrt 25
  let hyperbola := fun (x y : ℝ) ↦ x^2 / 36 - y^2 / 25 = 1
  2 * a = 12 := by sorry

end NUMINAMATH_CALUDE_hyperbola_vertex_distance_l3364_336417


namespace NUMINAMATH_CALUDE_tangent_line_y_intercept_l3364_336445

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a line in 2D space -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Checks if a line is tangent to a circle -/
def isTangent (l : Line) (c : Circle) : Prop :=
  sorry

/-- Checks if a point is in the first quadrant -/
def isInFirstQuadrant (p : ℝ × ℝ) : Prop :=
  p.1 > 0 ∧ p.2 > 0

theorem tangent_line_y_intercept :
  let c1 : Circle := { center := (3, 0), radius := 3 }
  let c2 : Circle := { center := (8, 0), radius := 2 }
  ∀ l : Line,
    (∃ p1 p2 : ℝ × ℝ,
      isTangent l c1 ∧
      isTangent l c2 ∧
      isInFirstQuadrant p1 ∧
      isInFirstQuadrant p2 ∧
      (p1.1 - 3)^2 + p1.2^2 = 9 ∧
      (p2.1 - 8)^2 + p2.2^2 = 4) →
    l.intercept = 13/4 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_y_intercept_l3364_336445


namespace NUMINAMATH_CALUDE_prob_same_flavor_is_one_fourth_l3364_336477

/-- The number of flavors available -/
def num_flavors : ℕ := 4

/-- The probability of selecting two bags of biscuits with the same flavor -/
def prob_same_flavor : ℚ := 1 / 4

/-- Theorem: The probability of selecting two bags of biscuits with the same flavor
    out of four possible flavors is 1/4 -/
theorem prob_same_flavor_is_one_fourth :
  prob_same_flavor = 1 / 4 := by sorry

end NUMINAMATH_CALUDE_prob_same_flavor_is_one_fourth_l3364_336477


namespace NUMINAMATH_CALUDE_inverse_B_cubed_l3364_336442

theorem inverse_B_cubed (B : Matrix (Fin 2) (Fin 2) ℝ) 
  (h : B⁻¹ = !![3, 7; -2, -5]) : 
  (B^3)⁻¹ = !![13, 0; -42, -95] := by
  sorry

end NUMINAMATH_CALUDE_inverse_B_cubed_l3364_336442


namespace NUMINAMATH_CALUDE_f_increasing_on_neg_reals_l3364_336457

/-- The function f(x) = -x^2 + 2x is monotonically increasing on (-∞, 0) -/
theorem f_increasing_on_neg_reals (x y : ℝ) :
  x < y → x < 0 → y < 0 → (-x^2 + 2*x) < (-y^2 + 2*y) := by
  sorry

end NUMINAMATH_CALUDE_f_increasing_on_neg_reals_l3364_336457


namespace NUMINAMATH_CALUDE_prob_product_not_zero_l3364_336432

/-- The number of sides on each die -/
def numSides : ℕ := 8

/-- The probability of getting a number other than 1 on a single die -/
def probNotOne : ℚ := (numSides - 1) / numSides

/-- The number of dice tossed -/
def numDice : ℕ := 3

/-- The probability that (a-1)(b-1)(c-1) ≠ 0 when tossing three eight-sided dice -/
theorem prob_product_not_zero : 
  (probNotOne ^ numDice : ℚ) = 343 / 512 := by sorry

end NUMINAMATH_CALUDE_prob_product_not_zero_l3364_336432


namespace NUMINAMATH_CALUDE_aquarium_fish_problem_l3364_336418

theorem aquarium_fish_problem (initial_fish : ℕ) : 
  initial_fish > 0 → initial_fish + 3 = 13 :=
by
  sorry

end NUMINAMATH_CALUDE_aquarium_fish_problem_l3364_336418


namespace NUMINAMATH_CALUDE_line_intercept_sum_l3364_336452

/-- Given a line with equation 3x + 5y + k = 0, where the sum of its x-intercept and y-intercept is 16, prove that k = -30. -/
theorem line_intercept_sum (k : ℝ) : 
  (∃ (x y : ℝ), 3 * x + 5 * y + k = 0 ∧ 
   (3 * 0 + 5 * y + k = 0 → 3 * x + 5 * 0 + k = 0 → x + y = 16)) → 
  k = -30 := by
sorry

end NUMINAMATH_CALUDE_line_intercept_sum_l3364_336452


namespace NUMINAMATH_CALUDE_circle_ratio_l3364_336465

theorem circle_ratio (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  π * b^2 - π * a^2 = 5 * (π * a^2) → a / b = Real.sqrt 6 / 6 :=
by sorry

end NUMINAMATH_CALUDE_circle_ratio_l3364_336465


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l3364_336478

theorem sufficient_not_necessary :
  (∀ a b : ℝ, a > 2 ∧ b > 1 → a + b > 3 ∧ a * b > 2) ∧
  (∃ a b : ℝ, a + b > 3 ∧ a * b > 2 ∧ ¬(a > 2 ∧ b > 1)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l3364_336478


namespace NUMINAMATH_CALUDE_asian_games_competition_l3364_336498

/-- Represents a player in the competition -/
structure Player where
  prelim_prob : ℚ  -- Probability of passing preliminary round
  final_prob : ℚ   -- Probability of passing final round

/-- The three players in the competition -/
def players : List Player := [
  ⟨1/2, 1/3⟩,  -- Player A
  ⟨1/3, 1/3⟩,  -- Player B
  ⟨1/2, 1/3⟩   -- Player C
]

/-- Probability of a player participating in the city competition -/
def city_comp_prob (p : Player) : ℚ := p.prelim_prob * p.final_prob

/-- Probability of at least one player participating in the city competition -/
def at_least_one_prob : ℚ :=
  1 - (players.map (λ p => 1 - city_comp_prob p)).prod

/-- Expected value of Option 1 (Lottery) -/
def option1_expected : ℚ := 3 * (1/3) * 600

/-- Expected value of Option 2 (Fixed Rewards) -/
def option2_expected : ℚ := 700

/-- Main theorem to prove -/
theorem asian_games_competition :
  at_least_one_prob = 31/81 ∧ option2_expected > option1_expected := by
  sorry


end NUMINAMATH_CALUDE_asian_games_competition_l3364_336498


namespace NUMINAMATH_CALUDE_prob_even_sum_coins_and_dice_l3364_336461

/-- Represents the outcome of tossing a fair coin -/
inductive CoinOutcome
| Heads
| Tails

/-- Represents the outcome of rolling a fair die -/
def DieOutcome := Fin 6

/-- The probability of getting heads on a fair coin toss -/
def probHeads : ℚ := 1/2

/-- The probability of getting an even number on a fair die roll -/
def probEvenDie : ℚ := 1/2

/-- The number of coins tossed -/
def numCoins : ℕ := 3

/-- Calculates the probability of getting k heads in n coin tosses -/
def probKHeads (n k : ℕ) : ℚ := sorry

/-- Calculates the probability of getting an even sum when rolling k fair dice -/
def probEvenSumKDice (k : ℕ) : ℚ := sorry

theorem prob_even_sum_coins_and_dice :
  (probKHeads numCoins 0 * 1 +
   probKHeads numCoins 1 * probEvenDie +
   probKHeads numCoins 2 * probEvenSumKDice 2 +
   probKHeads numCoins 3 * probEvenSumKDice 3) = 15/16 := by sorry

end NUMINAMATH_CALUDE_prob_even_sum_coins_and_dice_l3364_336461


namespace NUMINAMATH_CALUDE_cloth_selling_price_l3364_336427

/-- Calculates the total selling price of cloth given the quantity, profit per metre, and cost price per metre. -/
def total_selling_price (quantity : ℕ) (profit_per_metre : ℕ) (cost_price_per_metre : ℕ) : ℕ :=
  quantity * (cost_price_per_metre + profit_per_metre)

/-- Proves that the total selling price of 30 meters of cloth with a profit of Rs. 10 per metre
    and a cost price of Rs. 140 per metre is Rs. 4500. -/
theorem cloth_selling_price :
  total_selling_price 30 10 140 = 4500 := by
  sorry

end NUMINAMATH_CALUDE_cloth_selling_price_l3364_336427


namespace NUMINAMATH_CALUDE_ball_arrangement_theorem_l3364_336429

-- Define the number of balls and boxes
def n : ℕ := 4

-- Define the function for the number of arrangements with each box containing one ball
def arrangements_full (n : ℕ) : ℕ := n.factorial

-- Define the function for the number of arrangements with exactly one box empty
def arrangements_one_empty (n : ℕ) : ℕ := n.choose 2 * (n - 1).factorial

-- State the theorem
theorem ball_arrangement_theorem :
  arrangements_full n = 24 ∧ arrangements_one_empty n = 144 := by
  sorry


end NUMINAMATH_CALUDE_ball_arrangement_theorem_l3364_336429


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3364_336456

theorem complex_equation_solution (z : ℂ) :
  z + Complex.abs z = 2 + 8 * Complex.I → z = -15 + 8 * Complex.I :=
by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3364_336456


namespace NUMINAMATH_CALUDE_carl_winning_configurations_l3364_336458

def board_size : ℕ := 4

def winning_configurations : ℕ := 10

def remaining_cells_after_win : ℕ := 13

def ways_to_choose_three_from_twelve : ℕ := 220

theorem carl_winning_configurations :
  (winning_configurations * board_size * remaining_cells_after_win * ways_to_choose_three_from_twelve) = 114400 :=
by sorry

end NUMINAMATH_CALUDE_carl_winning_configurations_l3364_336458


namespace NUMINAMATH_CALUDE_dorothy_profit_l3364_336422

/-- Given the cost of ingredients, number of doughnuts made, and selling price per doughnut,
    calculate the profit. -/
def calculate_profit (ingredient_cost : ℕ) (num_doughnuts : ℕ) (price_per_doughnut : ℕ) : ℕ :=
  num_doughnuts * price_per_doughnut - ingredient_cost

/-- Theorem stating that Dorothy's profit is $22 given the problem conditions. -/
theorem dorothy_profit :
  calculate_profit 53 25 3 = 22 := by
  sorry

end NUMINAMATH_CALUDE_dorothy_profit_l3364_336422


namespace NUMINAMATH_CALUDE_triangle_existence_l3364_336446

/-- A triangle with semiperimeter s and two excircle radii r_a and r_b exists if and only if s^2 > r_a * r_b -/
theorem triangle_existence (s r_a r_b : ℝ) (h_s : s > 0) (h_ra : r_a > 0) (h_rb : r_b > 0) :
  (∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b + c = 2 * s ∧
    ∃ (r_ea r_eb : ℝ), r_ea = r_a ∧ r_eb = r_b ∧
    r_ea = s * (b + c - a) / (b + c) ∧
    r_eb = s * (a + c - b) / (a + c)) ↔
  s^2 > r_a * r_b :=
sorry

end NUMINAMATH_CALUDE_triangle_existence_l3364_336446
