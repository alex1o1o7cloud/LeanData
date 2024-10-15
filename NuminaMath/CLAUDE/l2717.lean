import Mathlib

namespace NUMINAMATH_CALUDE_graph_not_in_second_quadrant_l2717_271763

/-- A linear function y = 3x + k - 2 -/
def f (k : ℝ) (x : ℝ) : ℝ := 3 * x + k - 2

/-- The second quadrant -/
def second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

/-- The graph does not pass through the second quadrant -/
def not_in_second_quadrant (k : ℝ) : Prop :=
  ∀ x, ¬(second_quadrant x (f k x))

/-- Theorem: The graph of y = 3x + k - 2 does not pass through the second quadrant
    if and only if k ≤ 2 -/
theorem graph_not_in_second_quadrant (k : ℝ) :
  not_in_second_quadrant k ↔ k ≤ 2 := by
  sorry


end NUMINAMATH_CALUDE_graph_not_in_second_quadrant_l2717_271763


namespace NUMINAMATH_CALUDE_log_10_7_in_terms_of_r_s_l2717_271774

-- Define the logarithm function
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- State the theorem
theorem log_10_7_in_terms_of_r_s (r s : ℝ) 
  (h1 : log 4 2 = r) 
  (h2 : log 2 7 = s) : 
  log 10 7 = s / (1 + s) := by
  sorry

end NUMINAMATH_CALUDE_log_10_7_in_terms_of_r_s_l2717_271774


namespace NUMINAMATH_CALUDE_emma_bank_money_l2717_271733

theorem emma_bank_money (X : ℝ) : 
  X > 0 →
  (1/4 : ℝ) * (X - 400) = 400 →
  X = 2000 := by
sorry

end NUMINAMATH_CALUDE_emma_bank_money_l2717_271733


namespace NUMINAMATH_CALUDE_impossible_grid_arrangement_l2717_271744

/-- A type representing the digits 0, 1, and 2 -/
inductive Digit
  | zero
  | one
  | two

/-- A type representing a 100 x 100 grid filled with Digits -/
def Grid := Fin 100 → Fin 100 → Digit

/-- A function to count the number of a specific digit in a 3 x 4 rectangle -/
def countDigitIn3x4Rectangle (g : Grid) (i j : Fin 100) (d : Digit) : ℕ :=
  sorry

/-- A predicate to check if a 3 x 4 rectangle satisfies the condition -/
def isValid3x4Rectangle (g : Grid) (i j : Fin 100) : Prop :=
  countDigitIn3x4Rectangle g i j Digit.zero = 3 ∧
  countDigitIn3x4Rectangle g i j Digit.one = 4 ∧
  countDigitIn3x4Rectangle g i j Digit.two = 5

/-- The main theorem stating that it's impossible to fill the grid satisfying the conditions -/
theorem impossible_grid_arrangement : ¬ ∃ (g : Grid), ∀ (i j : Fin 100), isValid3x4Rectangle g i j := by
  sorry

end NUMINAMATH_CALUDE_impossible_grid_arrangement_l2717_271744


namespace NUMINAMATH_CALUDE_coordinates_wrt_origin_specific_point_coordinates_l2717_271779

/-- A point in a 2D Cartesian coordinate system -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- The origin of the Cartesian coordinate system -/
def origin : Point2D := ⟨0, 0⟩

/-- The coordinates of a point with respect to the origin are the same as its coordinates -/
theorem coordinates_wrt_origin (P : Point2D) : 
  P.x = P.x - origin.x ∧ P.y = P.y - origin.y :=
by sorry

/-- For the specific point P(-2, -4), its coordinates with respect to the origin are (-2, -4) -/
theorem specific_point_coordinates : 
  let P : Point2D := ⟨-2, -4⟩
  P.x - origin.x = -2 ∧ P.y - origin.y = -4 :=
by sorry

end NUMINAMATH_CALUDE_coordinates_wrt_origin_specific_point_coordinates_l2717_271779


namespace NUMINAMATH_CALUDE_soup_problem_solution_l2717_271703

/-- Represents the number of people a can of soup can feed -/
structure SoupCan where
  adults : ℕ
  children : ℕ

/-- Represents the problem setup -/
structure SoupProblem where
  can : SoupCan
  totalCans : ℕ
  childrenFed : ℕ

/-- Calculates the number of adults that can be fed with the remaining soup -/
def remainingAdults (problem : SoupProblem) : ℕ :=
  let cansUsedForChildren := problem.childrenFed / problem.can.children
  let remainingCans := problem.totalCans - cansUsedForChildren
  remainingCans * problem.can.adults

/-- Theorem stating that given the problem conditions, 12 adults can be fed with the remaining soup -/
theorem soup_problem_solution (problem : SoupProblem) 
  (h1 : problem.can.adults = 4)
  (h2 : problem.can.children = 6)
  (h3 : problem.totalCans = 6)
  (h4 : problem.childrenFed = 18) :
  remainingAdults problem = 12 := by
  sorry

#eval remainingAdults { can := { adults := 4, children := 6 }, totalCans := 6, childrenFed := 18 }

end NUMINAMATH_CALUDE_soup_problem_solution_l2717_271703


namespace NUMINAMATH_CALUDE_probability_five_blue_marbles_l2717_271753

def total_marbles : ℕ := 15
def blue_marbles : ℕ := 9
def red_marbles : ℕ := 6
def total_draws : ℕ := 8
def blue_draws : ℕ := 5

def prob_blue : ℚ := blue_marbles / total_marbles
def prob_red : ℚ := red_marbles / total_marbles

theorem probability_five_blue_marbles :
  (Nat.choose total_draws blue_draws : ℚ) *
  (prob_blue ^ blue_draws) *
  (prob_red ^ (total_draws - blue_draws)) =
  108864 / 390625 := by sorry

end NUMINAMATH_CALUDE_probability_five_blue_marbles_l2717_271753


namespace NUMINAMATH_CALUDE_item_cost_before_tax_reduction_cost_is_1000_l2717_271710

theorem item_cost_before_tax_reduction (tax_difference : ℝ) (cost_difference : ℝ) : ℝ :=
  let original_tax_rate := 0.05
  let new_tax_rate := 0.04
  let item_cost := cost_difference / (original_tax_rate - new_tax_rate)
  item_cost

theorem cost_is_1000 :
  item_cost_before_tax_reduction 0.01 10 = 1000 := by
  sorry

end NUMINAMATH_CALUDE_item_cost_before_tax_reduction_cost_is_1000_l2717_271710


namespace NUMINAMATH_CALUDE_min_socks_for_15_pairs_l2717_271765

/-- Represents a box of socks with four different colors. -/
structure SockBox where
  purple : ℕ
  orange : ℕ
  yellow : ℕ
  green : ℕ

/-- The minimum number of socks needed to guarantee at least n pairs. -/
def min_socks_for_pairs (n : ℕ) : ℕ := 2 * n + 3

/-- Theorem stating the minimum number of socks needed for 15 pairs. -/
theorem min_socks_for_15_pairs (box : SockBox) :
  min_socks_for_pairs 15 = 33 :=
sorry

end NUMINAMATH_CALUDE_min_socks_for_15_pairs_l2717_271765


namespace NUMINAMATH_CALUDE_simultaneous_completion_time_specific_completion_time_l2717_271724

/-- The time taken for two machines to complete an order when working simultaneously, 
    given their individual completion times. -/
theorem simultaneous_completion_time (t1 t2 : ℝ) (h1 : t1 > 0) (h2 : t2 > 0) : 
  (t1 * t2) / (t1 + t2) = (t1 * t2) / ((t1 * t2) * (1 / t1 + 1 / t2)) := by
  sorry

/-- Proof that two machines with completion times of 9 hours and 8 hours respectively
    will take 72/17 hours to complete the order when working simultaneously. -/
theorem specific_completion_time : 
  (9 : ℝ) * 8 / (9 + 8) = 72 / 17 := by
  sorry

end NUMINAMATH_CALUDE_simultaneous_completion_time_specific_completion_time_l2717_271724


namespace NUMINAMATH_CALUDE_new_person_age_l2717_271714

theorem new_person_age (n : ℕ) (initial_avg : ℝ) (new_avg : ℝ) (new_person_age : ℝ) : 
  n = 9 → 
  initial_avg = 14 → 
  new_avg = 16 → 
  (n * initial_avg + new_person_age) / (n + 1) = new_avg → 
  new_person_age = 34 := by
sorry

end NUMINAMATH_CALUDE_new_person_age_l2717_271714


namespace NUMINAMATH_CALUDE_min_abs_z_plus_i_l2717_271727

theorem min_abs_z_plus_i (z : ℂ) (h : Complex.abs (z^2 + 9) = Complex.abs (z * (z + 3*I))) :
  ∃ (w : ℂ), Complex.abs (w + I) = 2 ∧ ∀ (z : ℂ), Complex.abs (z^2 + 9) = Complex.abs (z * (z + 3*I)) → Complex.abs (z + I) ≥ 2 :=
sorry

end NUMINAMATH_CALUDE_min_abs_z_plus_i_l2717_271727


namespace NUMINAMATH_CALUDE_problem_solution_l2717_271775

theorem problem_solution (x y : ℝ) : x / y = 12 / 3 → y = 27 → x = 108 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2717_271775


namespace NUMINAMATH_CALUDE_max_value_of_shui_l2717_271760

def ChineseDigit := Fin 8

structure Phrase :=
  (jin xin li : ChineseDigit)
  (ke ba shan : ChineseDigit)
  (qiong shui : ChineseDigit)

def is_valid_phrase (p : Phrase) : Prop :=
  p.jin.val + p.xin.val + p.jin.val + p.li.val = 19 ∧
  p.li.val + p.ke.val + p.ba.val + p.shan.val = 19 ∧
  p.shan.val + p.qiong.val + p.shui.val + p.jin.val = 19

def all_different (p : Phrase) : Prop :=
  p.jin ≠ p.xin ∧ p.jin ≠ p.li ∧ p.jin ≠ p.ke ∧ p.jin ≠ p.ba ∧ p.jin ≠ p.shan ∧ p.jin ≠ p.qiong ∧ p.jin ≠ p.shui ∧
  p.xin ≠ p.li ∧ p.xin ≠ p.ke ∧ p.xin ≠ p.ba ∧ p.xin ≠ p.shan ∧ p.xin ≠ p.qiong ∧ p.xin ≠ p.shui ∧
  p.li ≠ p.ke ∧ p.li ≠ p.ba ∧ p.li ≠ p.shan ∧ p.li ≠ p.qiong ∧ p.li ≠ p.shui ∧
  p.ke ≠ p.ba ∧ p.ke ≠ p.shan ∧ p.ke ≠ p.qiong ∧ p.ke ≠ p.shui ∧
  p.ba ≠ p.shan ∧ p.ba ≠ p.qiong ∧ p.ba ≠ p.shui ∧
  p.shan ≠ p.qiong ∧ p.shan ≠ p.shui ∧
  p.qiong ≠ p.shui

theorem max_value_of_shui (p : Phrase) 
  (h1 : is_valid_phrase p)
  (h2 : all_different p)
  (h3 : p.jin.val > p.shan.val ∧ p.shan.val > p.li.val) :
  p.shui.val ≤ 7 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_shui_l2717_271760


namespace NUMINAMATH_CALUDE_keiko_speed_l2717_271711

theorem keiko_speed (track_width : ℝ) (time_difference : ℝ) : 
  track_width = 8 → time_difference = 48 → 
  (track_width * π) / time_difference = π / 3 := by
sorry

end NUMINAMATH_CALUDE_keiko_speed_l2717_271711


namespace NUMINAMATH_CALUDE_negation_of_some_primes_even_l2717_271799

theorem negation_of_some_primes_even :
  (¬ ∃ p, Nat.Prime p ∧ Even p) ↔ (∀ p, Nat.Prime p → Odd p) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_some_primes_even_l2717_271799


namespace NUMINAMATH_CALUDE_f_neg_one_equals_one_l2717_271720

-- Define the function f
variable (f : ℝ → ℝ)

-- State the theorem
theorem f_neg_one_equals_one (h : ∀ x, f (x - 1) = x^2 + 1) : f (-1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_f_neg_one_equals_one_l2717_271720


namespace NUMINAMATH_CALUDE_youtube_views_multiple_l2717_271797

/-- The multiple by which views increased on the fourth day -/
def viewMultiple (initialViews : ℕ) (totalViews : ℕ) (additionalViews : ℕ) : ℚ :=
  (totalViews - additionalViews - initialViews) / initialViews

theorem youtube_views_multiple :
  let initialViews : ℕ := 4000
  let totalViews : ℕ := 94000
  let additionalViews : ℕ := 50000
  viewMultiple initialViews totalViews additionalViews = 11 := by
sorry

end NUMINAMATH_CALUDE_youtube_views_multiple_l2717_271797


namespace NUMINAMATH_CALUDE_square_sum_from_conditions_l2717_271795

theorem square_sum_from_conditions (x y : ℝ) 
  (h1 : (x + y)^2 = 36) 
  (h2 : x * y = 8) : 
  x^2 + y^2 = 20 := by
sorry

end NUMINAMATH_CALUDE_square_sum_from_conditions_l2717_271795


namespace NUMINAMATH_CALUDE_complex_exponential_sum_l2717_271731

theorem complex_exponential_sum (α β : ℝ) :
  Complex.exp (Complex.I * α) + Complex.exp (Complex.I * β) = (3/5 : ℂ) + (2/5 : ℂ) * Complex.I →
  Complex.exp (-Complex.I * α) + Complex.exp (-Complex.I * β) = (3/5 : ℂ) - (2/5 : ℂ) * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_exponential_sum_l2717_271731


namespace NUMINAMATH_CALUDE_stock_sale_loss_l2717_271750

/-- Calculates the overall loss amount for a stock sale scenario -/
theorem stock_sale_loss (stock_worth : ℝ) (profit_percent : ℝ) (loss_percent : ℝ) 
  (profit_stock_percent : ℝ) (loss_stock_percent : ℝ) :
  stock_worth = 10000 →
  profit_percent = 10 →
  loss_percent = 5 →
  profit_stock_percent = 20 →
  loss_stock_percent = 80 →
  let profit_amount := (profit_stock_percent / 100) * stock_worth * (1 + profit_percent / 100)
  let loss_amount := (loss_stock_percent / 100) * stock_worth * (1 - loss_percent / 100)
  let total_sale := profit_amount + loss_amount
  stock_worth - total_sale = 200 := by sorry

end NUMINAMATH_CALUDE_stock_sale_loss_l2717_271750


namespace NUMINAMATH_CALUDE_floor_times_x_equals_48_l2717_271793

theorem floor_times_x_equals_48 :
  ∃! (x : ℝ), x > 0 ∧ (⌊x⌋ : ℝ) * x = 48 ∧ x = 8 := by
  sorry

end NUMINAMATH_CALUDE_floor_times_x_equals_48_l2717_271793


namespace NUMINAMATH_CALUDE_train_speed_l2717_271725

theorem train_speed (train_length : Real) (crossing_time : Real) (h1 : train_length = 1600) (h2 : crossing_time = 40) :
  (train_length / 1000) / (crossing_time / 3600) = 144 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l2717_271725


namespace NUMINAMATH_CALUDE_eighteen_power_mn_l2717_271708

theorem eighteen_power_mn (m n : ℤ) (R S : ℝ) (hR : R = 2^m) (hS : S = 3^n) :
  18^(m+n) = R^n * S^(2*m) := by
  sorry

end NUMINAMATH_CALUDE_eighteen_power_mn_l2717_271708


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l2717_271782

/-- Represents a hyperbola with parameters a and b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos_a : a > 0
  h_pos_b : b > 0
  h_rel : a = Real.sqrt 2 * b

/-- The equation of the asymptotes of a hyperbola -/
def asymptote_equation (h : Hyperbola) : Set (ℝ × ℝ) :=
  {(x, y) | y = Real.sqrt 2 * x ∨ y = -Real.sqrt 2 * x}

/-- Theorem: The asymptotes of the given hyperbola are y = ±√2x -/
theorem hyperbola_asymptotes (h : Hyperbola) : 
  asymptote_equation h = {(x, y) | y^2 / h.a^2 - x^2 / h.b^2 = 1} := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l2717_271782


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l2717_271771

/-- Proves that the speed of a boat in still water is 12 mph given certain conditions -/
theorem boat_speed_in_still_water 
  (distance : ℝ) 
  (downstream_time : ℝ) 
  (downstream_speed : ℝ) 
  (h1 : distance = 45) 
  (h2 : downstream_time = 3)
  (h3 : downstream_speed = distance / downstream_time)
  (h4 : ∃ (current_speed : ℝ), downstream_speed = 12 + current_speed) :
  12 = (12 : ℝ) := by sorry

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l2717_271771


namespace NUMINAMATH_CALUDE_ellipse_equation_l2717_271713

/-- The standard equation of an ellipse with given properties -/
theorem ellipse_equation (b c : ℝ) (h1 : b = 3) (h2 : c = 2) :
  ∃ a : ℝ, a^2 = b^2 + c^2 ∧ 
  (∀ x y : ℝ, (x^2 / b^2) + (y^2 / a^2) = 1 ↔ 
    x^2 / 9 + y^2 / 13 = 1) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_equation_l2717_271713


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2717_271796

theorem complex_equation_solution (x y : ℝ) (h : (x : ℂ) / (1 + Complex.I) = 1 - y * Complex.I) :
  (x : ℂ) + y * Complex.I = 2 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2717_271796


namespace NUMINAMATH_CALUDE_class_size_from_incorrect_mark_l2717_271764

theorem class_size_from_incorrect_mark (original_mark correct_mark : ℚ)
  (h1 : original_mark = 33)
  (h2 : correct_mark = 85)
  (h3 : ∀ (n : ℕ) (A : ℚ), n * (A + 1/2) = n * A + (correct_mark - original_mark)) :
  ∃ (n : ℕ), n = 104 := by
  sorry

end NUMINAMATH_CALUDE_class_size_from_incorrect_mark_l2717_271764


namespace NUMINAMATH_CALUDE_triangle_not_right_angle_l2717_271705

theorem triangle_not_right_angle (a b c : ℝ) (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_ratio : b = (4/3) * a ∧ c = (5/3) * a) (h_sum : a + b + c = 180) :
  ¬ (a = 90 ∨ b = 90 ∨ c = 90) :=
sorry

end NUMINAMATH_CALUDE_triangle_not_right_angle_l2717_271705


namespace NUMINAMATH_CALUDE_largest_prime_sum_l2717_271743

/-- A function that checks if a number is prime -/
def isPrime (n : ℕ) : Prop := sorry

/-- A function that returns the digits of a natural number -/
def digits (n : ℕ) : List ℕ := sorry

/-- A function that checks if a list contains exactly the digits 1 to 9 -/
def usesAllDigits (l : List ℕ) : Prop := sorry

theorem largest_prime_sum :
  ∀ (p₁ p₂ p₃ p₄ : ℕ),
    isPrime p₁ ∧ isPrime p₂ ∧ isPrime p₃ ∧ isPrime p₄ →
    usesAllDigits (digits p₁ ++ digits p₂ ++ digits p₃ ++ digits p₄) →
    p₁ + p₂ + p₃ + p₄ ≤ 1798 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_prime_sum_l2717_271743


namespace NUMINAMATH_CALUDE_quadratic_root_implies_n_l2717_271746

theorem quadratic_root_implies_n (n : ℝ) : 
  (∃ x : ℝ, x^2 - 2*x + n = 0) ∧ (3^2 - 2*3 + n = 0) → n = -3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_n_l2717_271746


namespace NUMINAMATH_CALUDE_erasers_per_group_l2717_271752

theorem erasers_per_group (total_erasers : ℕ) (num_groups : ℕ) (h1 : total_erasers = 270) (h2 : num_groups = 3) :
  total_erasers / num_groups = 90 := by
  sorry

end NUMINAMATH_CALUDE_erasers_per_group_l2717_271752


namespace NUMINAMATH_CALUDE_ten_sparklers_to_crackers_five_ornaments_one_cracker_more_valuable_l2717_271788

-- Define the exchange rates
def ornament_to_cracker : ℚ := 2
def sparkler_to_garland : ℚ := 2/5
def ornament_to_garland : ℚ := 1/4

-- Define the conversion function
def convert (item : String) (quantity : ℚ) : ℚ :=
  match item with
  | "sparkler" => quantity * sparkler_to_garland * (1 / ornament_to_garland) * ornament_to_cracker
  | "ornament" => quantity * ornament_to_cracker
  | _ => 0

-- Theorem for part (a)
theorem ten_sparklers_to_crackers :
  convert "sparkler" 10 = 32 := by sorry

-- Theorem for part (b)
theorem five_ornaments_one_cracker_more_valuable :
  convert "ornament" 5 + 1 > convert "sparkler" 2 := by sorry

end NUMINAMATH_CALUDE_ten_sparklers_to_crackers_five_ornaments_one_cracker_more_valuable_l2717_271788


namespace NUMINAMATH_CALUDE_waiter_customers_problem_l2717_271791

theorem waiter_customers_problem :
  ∃ x : ℝ, x > 0 ∧ ((x - 19.0) - 14.0 = 3) → x = 36.0 := by
  sorry

end NUMINAMATH_CALUDE_waiter_customers_problem_l2717_271791


namespace NUMINAMATH_CALUDE_jacket_cost_calculation_l2717_271741

def shorts_cost : ℚ := 13.99
def shirt_cost : ℚ := 12.14
def total_cost : ℚ := 33.56

theorem jacket_cost_calculation :
  ∃ (jacket_cost : ℚ), 
    jacket_cost = total_cost - (shorts_cost + shirt_cost) ∧
    jacket_cost = 7.43 := by
  sorry

end NUMINAMATH_CALUDE_jacket_cost_calculation_l2717_271741


namespace NUMINAMATH_CALUDE_taxi_truck_speed_ratio_l2717_271777

/-- Given a truck that travels 2.1 km in 1 minute and a taxi that travels 10.5 km in 4 minutes,
    prove that the taxi is 1.25 times faster than the truck. -/
theorem taxi_truck_speed_ratio :
  let truck_speed := 2.1 -- km per minute
  let taxi_speed := 10.5 / 4 -- km per minute
  taxi_speed / truck_speed = 1.25 := by sorry

end NUMINAMATH_CALUDE_taxi_truck_speed_ratio_l2717_271777


namespace NUMINAMATH_CALUDE_largest_prime_divisor_to_test_l2717_271707

theorem largest_prime_divisor_to_test (n : ℕ) : 
  1000 ≤ n ∧ n ≤ 1100 → 
  (∀ p : ℕ, p.Prime → p ≤ 31 → n % p ≠ 0) → 
  n.Prime ∨ n = 1 :=
sorry

end NUMINAMATH_CALUDE_largest_prime_divisor_to_test_l2717_271707


namespace NUMINAMATH_CALUDE_permutation_5_2_combination_6_3_plus_6_4_l2717_271728

-- Define permutation function
def A (n k : ℕ) : ℕ := sorry

-- Define combination function
def C (n k : ℕ) : ℕ := sorry

-- Theorem for A_5^2
theorem permutation_5_2 : A 5 2 = 20 := by sorry

-- Theorem for C_6^3 + C_6^4
theorem combination_6_3_plus_6_4 : C 6 3 + C 6 4 = 35 := by sorry

end NUMINAMATH_CALUDE_permutation_5_2_combination_6_3_plus_6_4_l2717_271728


namespace NUMINAMATH_CALUDE_consecutive_sum_product_l2717_271769

theorem consecutive_sum_product (n : ℕ) (h : n > 100) :
  ∃ (a b c : ℕ), a > 1 ∧ b > 1 ∧ c > 1 ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  (3 * (n + 1) = a * b * c ∨ 3 * (n + 2) = a * b * c) :=
by sorry

end NUMINAMATH_CALUDE_consecutive_sum_product_l2717_271769


namespace NUMINAMATH_CALUDE_zeros_between_seven_and_three_l2717_271712

theorem zeros_between_seven_and_three : ∀ n : ℕ, 
  (7 * 10^(n + 1) + 3 = 70003) ↔ (n = 4) :=
by sorry

end NUMINAMATH_CALUDE_zeros_between_seven_and_three_l2717_271712


namespace NUMINAMATH_CALUDE_triangle_theorem_l2717_271792

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The given condition in the problem -/
def condition (t : Triangle) : Prop :=
  Real.sqrt 2 * t.b * Real.sin t.C + t.a * Real.sin t.A = t.b * Real.sin t.B + t.c * Real.sin t.C

/-- The theorem to be proved -/
theorem triangle_theorem (t : Triangle) (h1 : condition t) (h2 : t.a = Real.sqrt 2) :
  t.A = π / 4 ∧ 
  ∀ (AD : ℝ), AD ≤ 1 + Real.sqrt 2 / 2 :=
sorry

end NUMINAMATH_CALUDE_triangle_theorem_l2717_271792


namespace NUMINAMATH_CALUDE_square_fence_perimeter_l2717_271734

-- Define the number of posts
def num_posts : ℕ := 24

-- Define the width of each post in feet
def post_width : ℚ := 1 / 3

-- Define the distance between adjacent posts in feet
def post_spacing : ℕ := 5

-- Define the number of posts per side (excluding corners)
def posts_per_side : ℕ := (num_posts - 4) / 4

-- Define the total number of posts per side (including corners)
def total_posts_per_side : ℕ := posts_per_side + 2

-- Define the number of gaps between posts on one side
def gaps_per_side : ℕ := total_posts_per_side - 1

-- Define the length of one side of the square
def side_length : ℚ := gaps_per_side * post_spacing + total_posts_per_side * post_width

-- Theorem statement
theorem square_fence_perimeter :
  4 * side_length = 129 + 1 / 3 :=
sorry

end NUMINAMATH_CALUDE_square_fence_perimeter_l2717_271734


namespace NUMINAMATH_CALUDE_problem_statement_l2717_271790

theorem problem_statement (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x * y * z = 1)
  (h2 : x + 1 / z = 5)
  (h3 : y + 1 / x = 29) :
  z + 1 / y = 1 / 4 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l2717_271790


namespace NUMINAMATH_CALUDE_lcm_gcd_problem_l2717_271762

theorem lcm_gcd_problem (a b : ℕ+) : 
  Nat.lcm a b = 4620 → 
  Nat.gcd a b = 21 → 
  a = 210 → 
  b = 462 := by
sorry

end NUMINAMATH_CALUDE_lcm_gcd_problem_l2717_271762


namespace NUMINAMATH_CALUDE_smallest_even_number_sum_1194_l2717_271732

/-- Given three consecutive even numbers whose sum is 1194, 
    the smallest of these numbers is 396. -/
theorem smallest_even_number_sum_1194 (x : ℕ) 
  (h1 : x % 2 = 0)  -- x is even
  (h2 : x + (x + 2) + (x + 4) = 1194) : x = 396 := by
  sorry

end NUMINAMATH_CALUDE_smallest_even_number_sum_1194_l2717_271732


namespace NUMINAMATH_CALUDE_cube_root_27_times_fourth_root_16_l2717_271770

theorem cube_root_27_times_fourth_root_16 : (27 : ℝ) ^ (1/3) * (16 : ℝ) ^ (1/4) = 6 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_27_times_fourth_root_16_l2717_271770


namespace NUMINAMATH_CALUDE_blue_square_area_ratio_l2717_271759

/-- Represents a square flag with a symmetric cross -/
structure CrossFlag where
  side : ℝ
  cross_area_ratio : ℝ
  symmetric : Bool

/-- The area of the flag -/
def flag_area (flag : CrossFlag) : ℝ := flag.side ^ 2

/-- The area of the cross -/
def cross_area (flag : CrossFlag) : ℝ := flag.cross_area_ratio * flag_area flag

/-- The theorem stating the relationship between the blue square area and the flag area -/
theorem blue_square_area_ratio (flag : CrossFlag) 
  (h1 : flag.cross_area_ratio = 0.36)
  (h2 : flag.symmetric = true) : 
  (flag.side * 0.2) ^ 2 / flag_area flag = 0.04 := by
  sorry

#check blue_square_area_ratio

end NUMINAMATH_CALUDE_blue_square_area_ratio_l2717_271759


namespace NUMINAMATH_CALUDE_g_symmetric_to_f_max_value_of_a_l2717_271756

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + 4*x + 3

-- Define the function g (to be proved)
def g (x : ℝ) : ℝ := x^2 - 8*x + 15

-- Theorem 1: Prove that g is symmetric to f about x=1
theorem g_symmetric_to_f : ∀ x : ℝ, g x = f (2 - x) := by sorry

-- Theorem 2: Prove the maximum value of a
theorem max_value_of_a : 
  (∀ x : ℝ, g x ≥ g 6 - 4) ∧ 
  (∀ a : ℝ, (∀ x : ℝ, g x ≥ g a - 4) → a ≤ 6) := by sorry

end NUMINAMATH_CALUDE_g_symmetric_to_f_max_value_of_a_l2717_271756


namespace NUMINAMATH_CALUDE_cos_A_value_projection_BA_on_BC_l2717_271736

noncomputable section

variables (A B C : ℝ) (a b c : ℝ)

-- Define the triangle ABC
def triangle_ABC : Prop :=
  2 * (Real.cos ((A - B) / 2))^2 * Real.cos B - Real.sin (A - B) * Real.sin B + Real.cos (A + C) = -3/5

-- Define the side lengths
def side_lengths : Prop :=
  a = 4 * Real.sqrt 2 ∧ b = 5

-- Theorem for part 1
theorem cos_A_value (h : triangle_ABC A B C) : Real.cos A = -3/5 := by sorry

-- Theorem for part 2
theorem projection_BA_on_BC (h1 : triangle_ABC A B C) (h2 : side_lengths a b) :
  ∃ (proj : ℝ), proj = Real.sqrt 2 / 2 ∧ proj = c * Real.cos B := by sorry

end

end NUMINAMATH_CALUDE_cos_A_value_projection_BA_on_BC_l2717_271736


namespace NUMINAMATH_CALUDE_planted_fraction_is_correct_l2717_271729

/-- Represents a right triangle with an unplanted square in the corner -/
structure FieldTriangle where
  leg1 : ℝ
  leg2 : ℝ
  square_distance : ℝ

/-- The fraction of the field that is planted -/
def planted_fraction (f : FieldTriangle) : ℚ :=
  367 / 375

theorem planted_fraction_is_correct (f : FieldTriangle) 
  (h1 : f.leg1 = 5)
  (h2 : f.leg2 = 12)
  (h3 : f.square_distance = 4) :
  planted_fraction f = 367 / 375 := by
  sorry

end NUMINAMATH_CALUDE_planted_fraction_is_correct_l2717_271729


namespace NUMINAMATH_CALUDE_ratio_odd_even_divisors_l2717_271726

def N : ℕ := 34 * 34 * 63 * 270

def sum_odd_divisors (n : ℕ) : ℕ := sorry

def sum_even_divisors (n : ℕ) : ℕ := sorry

theorem ratio_odd_even_divisors :
  (sum_odd_divisors N : ℚ) / (sum_even_divisors N : ℚ) = 1 / 14 := by sorry

end NUMINAMATH_CALUDE_ratio_odd_even_divisors_l2717_271726


namespace NUMINAMATH_CALUDE_students_just_passed_l2717_271722

theorem students_just_passed (total : ℕ) (first_div_percent : ℚ) (second_div_percent : ℚ) 
  (h_total : total = 300)
  (h_first : first_div_percent = 29/100)
  (h_second : second_div_percent = 54/100)
  (h_no_fail : first_div_percent + second_div_percent < 1) :
  total - (total * first_div_percent).floor - (total * second_div_percent).floor = 51 := by
  sorry

end NUMINAMATH_CALUDE_students_just_passed_l2717_271722


namespace NUMINAMATH_CALUDE_truck_weight_problem_l2717_271700

theorem truck_weight_problem (truck_weight trailer_weight : ℝ) : 
  truck_weight + trailer_weight = 7000 →
  trailer_weight = 0.5 * truck_weight - 200 →
  truck_weight = 4800 := by
sorry

end NUMINAMATH_CALUDE_truck_weight_problem_l2717_271700


namespace NUMINAMATH_CALUDE_prob_three_heads_eight_tosses_l2717_271755

/-- The probability of getting exactly k heads in n tosses of a fair coin -/
def prob_k_heads (n k : ℕ) : ℚ :=
  (Nat.choose n k : ℚ) / (2 ^ n : ℚ)

/-- Theorem: The probability of getting exactly 3 heads in 8 tosses of a fair coin is 7/32 -/
theorem prob_three_heads_eight_tosses :
  prob_k_heads 8 3 = 7 / 32 := by
  sorry

end NUMINAMATH_CALUDE_prob_three_heads_eight_tosses_l2717_271755


namespace NUMINAMATH_CALUDE_temperature_at_4km_l2717_271785

def temperature_at_altitude (ground_temp : ℝ) (altitude : ℝ) : ℝ :=
  ground_temp - 5 * altitude

theorem temperature_at_4km (ground_temp : ℝ) (h1 : ground_temp = 15) : 
  temperature_at_altitude ground_temp 4 = -5 := by
  sorry

end NUMINAMATH_CALUDE_temperature_at_4km_l2717_271785


namespace NUMINAMATH_CALUDE_a1_value_l2717_271758

theorem a1_value (x : ℝ) (a : Fin 8 → ℝ) :
  (x - 1)^7 = a 0 + a 1 * (x + 1) + a 2 * (x + 1)^2 + a 3 * (x + 1)^3 + 
              a 4 * (x + 1)^4 + a 5 * (x + 1)^5 + a 6 * (x + 1)^6 + a 7 * (x + 1)^7 →
  a 1 = 448 := by
sorry

end NUMINAMATH_CALUDE_a1_value_l2717_271758


namespace NUMINAMATH_CALUDE_stating_total_dark_triangles_formula_l2717_271778

/-- 
Given a sequence of figures formed by an increasing number of dark equilateral triangles,
this function represents the total number of dark triangles used in the first n figures.
-/
def total_dark_triangles (n : ℕ) : ℕ := n * (n + 1) * (n + 2) / 6

/-- 
Theorem stating that the total number of dark triangles used in the first n figures
of the sequence is (n(n+1)(n+2))/6.
-/
theorem total_dark_triangles_formula (n : ℕ) :
  total_dark_triangles n = n * (n + 1) * (n + 2) / 6 := by
  sorry

end NUMINAMATH_CALUDE_stating_total_dark_triangles_formula_l2717_271778


namespace NUMINAMATH_CALUDE_correct_num_browsers_l2717_271783

/-- The number of browsers James had on his computer. -/
def num_browsers : ℕ := 2

/-- The number of windows per browser. -/
def windows_per_browser : ℕ := 3

/-- The number of tabs per window. -/
def tabs_per_window : ℕ := 10

/-- The total number of tabs in all browsers. -/
def total_tabs : ℕ := 60

/-- Theorem stating that the number of browsers is correct given the conditions. -/
theorem correct_num_browsers :
  num_browsers * windows_per_browser * tabs_per_window = total_tabs :=
by sorry

end NUMINAMATH_CALUDE_correct_num_browsers_l2717_271783


namespace NUMINAMATH_CALUDE_division_of_squares_l2717_271757

theorem division_of_squares (a : ℝ) : 2 * a^2 / a^2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_division_of_squares_l2717_271757


namespace NUMINAMATH_CALUDE_simple_interest_rate_l2717_271772

/-- Simple interest rate calculation -/
theorem simple_interest_rate 
  (principal : ℝ) 
  (final_amount : ℝ) 
  (time : ℝ) 
  (h1 : principal = 750) 
  (h2 : final_amount = 900) 
  (h3 : time = 8) :
  (final_amount - principal) * 100 / (principal * time) = 2.5 := by
sorry

end NUMINAMATH_CALUDE_simple_interest_rate_l2717_271772


namespace NUMINAMATH_CALUDE_solve_class_selection_problem_l2717_271784

/-- The number of students who selected only "Selected Lectures on Geometric Proofs" -/
def students_only_geometric_proofs (total : ℕ) (both : ℕ) (difference : ℕ) : ℕ :=
  let geometric := (total + both - difference) / 2
  geometric - both

/-- The main theorem -/
theorem solve_class_selection_problem :
  let total := 54  -- Total number of students
  let both := 6    -- Number of students who selected both topics
  let difference := 8  -- Difference between selections
  students_only_geometric_proofs total both difference = 20 := by
  sorry

#eval students_only_geometric_proofs 54 6 8

end NUMINAMATH_CALUDE_solve_class_selection_problem_l2717_271784


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2717_271754

theorem arithmetic_sequence_sum (a₁ aₙ : ℤ) (n : ℕ) (h : n > 0) :
  (a₁ = -4) → (aₙ = 37) → (n = 10) →
  (∃ d : ℚ, ∀ k : ℕ, k < n → a₁ + k * d = aₙ - (n - 1 - k) * d) →
  (n : ℚ) * (a₁ + aₙ) / 2 = 165 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2717_271754


namespace NUMINAMATH_CALUDE_problem_solution_l2717_271721

theorem problem_solution (n m q q' r r' : ℕ) : 
  n > m ∧ m > 1 ∧
  n = q * m + r ∧ r < m ∧
  n - 1 = q' * m + r' ∧ r' < m ∧
  q + q' = 99 ∧ r + r' = 99 →
  n = 5000 ∧ ∃ k : ℕ, 2 * n = k * k :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l2717_271721


namespace NUMINAMATH_CALUDE_kim_average_increase_l2717_271761

def kim_scores : List ℝ := [92, 85, 90, 95]

theorem kim_average_increase :
  let initial_avg := (kim_scores.take 3).sum / 3
  let new_avg := kim_scores.sum / 4
  new_avg - initial_avg = 1.5 := by sorry

end NUMINAMATH_CALUDE_kim_average_increase_l2717_271761


namespace NUMINAMATH_CALUDE_nine_sided_polygon_diagonals_l2717_271780

-- Define a convex polygon
structure ConvexPolygon where
  sides : ℕ
  is_convex : Bool

-- Define the number of right angles in the polygon
def right_angles (p : ConvexPolygon) : ℕ := 2

-- Define the function to calculate the number of diagonals
def num_diagonals (p : ConvexPolygon) : ℕ :=
  p.sides * (p.sides - 3) / 2

-- Theorem statement
theorem nine_sided_polygon_diagonals (p : ConvexPolygon) :
  p.sides = 9 → p.is_convex = true → right_angles p = 2 → num_diagonals p = 27 := by
  sorry

end NUMINAMATH_CALUDE_nine_sided_polygon_diagonals_l2717_271780


namespace NUMINAMATH_CALUDE_percent_within_one_sd_is_68_l2717_271749

/-- A symmetric distribution with a given percentage below one standard deviation above the mean -/
structure SymmetricDistribution where
  /-- The percentage of the distribution below one standard deviation above the mean -/
  percent_below_one_sd : ℝ
  /-- Assumption that the percentage is 84% -/
  percent_is_84 : percent_below_one_sd = 84

/-- The percentage of a symmetric distribution that lies within one standard deviation of the mean -/
def percent_within_one_sd (d : SymmetricDistribution) : ℝ :=
  2 * d.percent_below_one_sd - 100

theorem percent_within_one_sd_is_68 (d : SymmetricDistribution) :
  percent_within_one_sd d = 68 := by
  sorry

end NUMINAMATH_CALUDE_percent_within_one_sd_is_68_l2717_271749


namespace NUMINAMATH_CALUDE_quadrilateral_ratio_l2717_271701

theorem quadrilateral_ratio (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (h1 : a^2 + d^2 - a*d = b^2 + c^2 + b*c) (h2 : a^2 + b^2 = c^2 + d^2) :
  (a*b + c*d) / (a*d + b*c) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_ratio_l2717_271701


namespace NUMINAMATH_CALUDE_brick_width_l2717_271745

/-- The surface area of a rectangular prism. -/
def surface_area (l w h : ℝ) : ℝ := 2 * (l * w + l * h + w * h)

/-- Theorem: The width of a brick with given dimensions and surface area. -/
theorem brick_width (l h : ℝ) (sa : ℝ) (hl : l = 10) (hh : h = 3) (hsa : sa = 164) :
  ∃ w : ℝ, w = 4 ∧ surface_area l w h = sa :=
sorry

end NUMINAMATH_CALUDE_brick_width_l2717_271745


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l2717_271738

theorem arithmetic_sequence_ratio (a d : ℚ) : 
  let S : ℕ → ℚ := λ n => n / 2 * (2 * a + (n - 1) * d)
  S 15 = 3 * S 8 → a / d = 7 / 3 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l2717_271738


namespace NUMINAMATH_CALUDE_walking_speed_problem_l2717_271766

/-- The walking speed problem -/
theorem walking_speed_problem 
  (distance_between_homes : ℝ)
  (bob_speed : ℝ)
  (alice_distance : ℝ)
  (time_difference : ℝ)
  (h1 : distance_between_homes = 41)
  (h2 : bob_speed = 4)
  (h3 : alice_distance = 25)
  (h4 : time_difference = 1)
  : ∃ (alice_speed : ℝ), 
    alice_speed = 5 ∧ 
    alice_distance / alice_speed = (distance_between_homes - alice_distance) / bob_speed + time_difference :=
by sorry

end NUMINAMATH_CALUDE_walking_speed_problem_l2717_271766


namespace NUMINAMATH_CALUDE_competition_scores_l2717_271735

theorem competition_scores (score24 score46 score12 : ℕ) : 
  score24 + score46 + score12 = 285 →
  ∃ (x : ℕ), score24 - 8 = x ∧ score46 - 12 = x ∧ score12 - 7 = x →
  score24 + score12 = 187 := by
sorry

end NUMINAMATH_CALUDE_competition_scores_l2717_271735


namespace NUMINAMATH_CALUDE_circle_area_radius_increase_l2717_271781

theorem circle_area_radius_increase : 
  ∀ (r : ℝ) (r' : ℝ), r > 0 → r' > 0 → 
  (π * r' ^ 2 = 4 * π * r ^ 2) → 
  (r' = 2 * r) := by
sorry

end NUMINAMATH_CALUDE_circle_area_radius_increase_l2717_271781


namespace NUMINAMATH_CALUDE_cindy_used_stickers_l2717_271704

theorem cindy_used_stickers (initial_stickers : ℕ) (cindy_remaining : ℕ) : 
  initial_stickers + 18 = cindy_remaining + 33 → 
  initial_stickers - cindy_remaining = 15 :=
by
  sorry

end NUMINAMATH_CALUDE_cindy_used_stickers_l2717_271704


namespace NUMINAMATH_CALUDE_bigger_part_is_thirteen_l2717_271767

theorem bigger_part_is_thirteen (x y : ℝ) (h1 : x + y = 24) (h2 : 7 * x + 5 * y = 146) 
  (h3 : x > 0) (h4 : y > 0) : max x y = 13 := by
  sorry

end NUMINAMATH_CALUDE_bigger_part_is_thirteen_l2717_271767


namespace NUMINAMATH_CALUDE_no_negative_sum_of_squares_l2717_271715

theorem no_negative_sum_of_squares : ¬∃ (x y : ℝ), x^2 + y^2 < 0 := by
  sorry

end NUMINAMATH_CALUDE_no_negative_sum_of_squares_l2717_271715


namespace NUMINAMATH_CALUDE_transform_minus_four_plus_two_i_l2717_271737

/-- Applies a 270° counter-clockwise rotation followed by a scaling of 2 to a complex number -/
def transform (z : ℂ) : ℂ := 2 * (z * Complex.I)

/-- The result of applying the transformation to -4 + 2i -/
theorem transform_minus_four_plus_two_i :
  transform (Complex.ofReal (-4) + Complex.I * Complex.ofReal 2) = Complex.ofReal 4 + Complex.I * Complex.ofReal 8 := by
  sorry

#check transform_minus_four_plus_two_i

end NUMINAMATH_CALUDE_transform_minus_four_plus_two_i_l2717_271737


namespace NUMINAMATH_CALUDE_second_number_in_first_set_l2717_271748

theorem second_number_in_first_set (x : ℝ) : 
  (20 + x + 60) / 3 = (10 + 70 + 16) / 3 + 8 ↔ x = 40 := by
  sorry

end NUMINAMATH_CALUDE_second_number_in_first_set_l2717_271748


namespace NUMINAMATH_CALUDE_hamburger_combinations_l2717_271742

/-- The number of condiments available for hamburgers -/
def num_condiments : ℕ := 9

/-- The number of choices for meat patties -/
def num_patty_choices : ℕ := 4

/-- The number of bread type choices -/
def num_bread_choices : ℕ := 2

/-- The total number of different hamburger combinations -/
def total_combinations : ℕ := 2^num_condiments * num_patty_choices * num_bread_choices

theorem hamburger_combinations :
  total_combinations = 4096 :=
sorry

end NUMINAMATH_CALUDE_hamburger_combinations_l2717_271742


namespace NUMINAMATH_CALUDE_evaluate_expression_l2717_271723

theorem evaluate_expression : -(16 / 2 * 12 - 75 + 4 * (2 * 5) + 25) = -86 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2717_271723


namespace NUMINAMATH_CALUDE_train_length_l2717_271789

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 60 → time = 18 → ∃ (length : ℝ), 
  (length ≥ 299.5 ∧ length ≤ 300.5) ∧ 
  length = speed * (1000 / 3600) * time := by
  sorry


end NUMINAMATH_CALUDE_train_length_l2717_271789


namespace NUMINAMATH_CALUDE_diamond_equation_solution_l2717_271776

-- Define the diamond operation
def diamond (a b : ℝ) : ℝ := 3 * a + 3 * b^2

-- Theorem statement
theorem diamond_equation_solution :
  ∀ a : ℝ, diamond a 4 = 75 → a = 9 := by
  sorry

end NUMINAMATH_CALUDE_diamond_equation_solution_l2717_271776


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2717_271786

-- Define the sets A and B
def A : Set ℝ := {x | ∃ y, y = Real.sqrt x}
def B : Set ℝ := {x | x^2 - x - 2 < 0}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | 0 ≤ x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2717_271786


namespace NUMINAMATH_CALUDE_y_value_theorem_l2717_271740

theorem y_value_theorem (y₁ y₂ y₃ y₄ y₅ y₆ y₇ y₈ : ℝ) 
  (eq1 : y₁ + 4*y₂ + 9*y₃ + 16*y₄ + 25*y₅ + 36*y₆ + 49*y₇ + 64*y₈ = 3)
  (eq2 : 4*y₁ + 9*y₂ + 16*y₃ + 25*y₄ + 36*y₅ + 49*y₆ + 64*y₇ + 81*y₈ = 15)
  (eq3 : 9*y₁ + 16*y₂ + 25*y₃ + 36*y₄ + 49*y₅ + 64*y₆ + 81*y₇ + 100*y₈ = 140) :
  16*y₁ + 25*y₂ + 36*y₃ + 49*y₄ + 64*y₅ + 81*y₆ + 100*y₇ + 121*y₈ = 472 :=
by sorry

end NUMINAMATH_CALUDE_y_value_theorem_l2717_271740


namespace NUMINAMATH_CALUDE_low_card_value_is_one_l2717_271717

/-- A card type in the high-low game -/
inductive CardType
| High
| Low

/-- The high-low card game -/
structure HighLowGame where
  total_cards : Nat
  high_cards : Nat
  low_cards : Nat
  high_card_value : Nat
  low_card_value : Nat
  target_points : Nat
  target_low_cards : Nat
  ways_to_reach_target : Nat

/-- Conditions for the high-low game -/
def game_conditions (g : HighLowGame) : Prop :=
  g.total_cards = 52 ∧
  g.high_cards = g.low_cards ∧
  g.high_cards + g.low_cards = g.total_cards ∧
  g.high_card_value = 2 ∧
  g.target_points = 5 ∧
  g.target_low_cards = 3 ∧
  g.ways_to_reach_target = 4

/-- Theorem stating that under the given conditions, the low card value must be 1 -/
theorem low_card_value_is_one (g : HighLowGame) :
  game_conditions g → g.low_card_value = 1 := by
  sorry

end NUMINAMATH_CALUDE_low_card_value_is_one_l2717_271717


namespace NUMINAMATH_CALUDE_beef_weight_loss_percentage_l2717_271751

/-- Given a side of beef weighing 800 pounds before processing and 640 pounds after processing,
    the percentage of weight lost during processing is 20%. -/
theorem beef_weight_loss_percentage (weight_before : ℝ) (weight_after : ℝ) :
  weight_before = 800 ∧ weight_after = 640 →
  (weight_before - weight_after) / weight_before * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_beef_weight_loss_percentage_l2717_271751


namespace NUMINAMATH_CALUDE_wx_length_is_25_l2717_271773

/-- A quadrilateral with two right angles and specific side lengths -/
structure RightQuadrilateral where
  W : ℝ × ℝ
  X : ℝ × ℝ
  Y : ℝ × ℝ
  Z : ℝ × ℝ
  right_angle_X : (X.1 - W.1) * (Y.1 - X.1) + (X.2 - W.2) * (Y.2 - X.2) = 0
  right_angle_Y : (Y.1 - X.1) * (Z.1 - Y.1) + (Y.2 - X.2) * (Z.2 - Y.2) = 0
  wz_length : Real.sqrt ((W.1 - Z.1)^2 + (W.2 - Z.2)^2) = 7
  xy_length : Real.sqrt ((X.1 - Y.1)^2 + (X.2 - Y.2)^2) = 14
  yz_length : Real.sqrt ((Y.1 - Z.1)^2 + (Y.2 - Z.2)^2) = 24

/-- The length of WX in the given quadrilateral is 25 -/
theorem wx_length_is_25 (q : RightQuadrilateral) :
  Real.sqrt ((q.W.1 - q.X.1)^2 + (q.W.2 - q.X.2)^2) = 25 := by
  sorry

end NUMINAMATH_CALUDE_wx_length_is_25_l2717_271773


namespace NUMINAMATH_CALUDE_customer_satisfaction_probability_l2717_271702

/-- The probability that a dissatisfied customer leaves an angry review -/
def prob_dissatisfied_angry : ℝ := 0.80

/-- The probability that a satisfied customer leaves a positive review -/
def prob_satisfied_positive : ℝ := 0.15

/-- The number of angry reviews received -/
def num_angry_reviews : ℕ := 60

/-- The number of positive reviews received -/
def num_positive_reviews : ℕ := 20

/-- The probability that a customer is satisfied -/
def prob_satisfied : ℝ := 0.64

theorem customer_satisfaction_probability :
  prob_satisfied = 0.64 :=
sorry

end NUMINAMATH_CALUDE_customer_satisfaction_probability_l2717_271702


namespace NUMINAMATH_CALUDE_range_of_g_l2717_271706

noncomputable def g (x : ℝ) : ℝ := Real.arctan (2 * x) + Real.arctan ((2 - 3 * x) / (2 + 3 * x))

theorem range_of_g :
  Set.range g = {-3 * Real.pi / 4, Real.pi / 4} := by sorry

end NUMINAMATH_CALUDE_range_of_g_l2717_271706


namespace NUMINAMATH_CALUDE_estimate_wild_rabbits_l2717_271709

theorem estimate_wild_rabbits (initial_marked : ℕ) (recaptured : ℕ) (marked_in_recapture : ℕ) :
  initial_marked = 100 →
  recaptured = 40 →
  marked_in_recapture = 5 →
  (recaptured * initial_marked) / marked_in_recapture = 800 :=
by sorry

end NUMINAMATH_CALUDE_estimate_wild_rabbits_l2717_271709


namespace NUMINAMATH_CALUDE_complex_magnitude_squared_l2717_271798

theorem complex_magnitude_squared (a b : ℝ) : 
  let z : ℂ := Complex.mk a b
  (z - Complex.abs z = 4 - 6*I) → Complex.normSq z = 42.25 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_squared_l2717_271798


namespace NUMINAMATH_CALUDE_lcm_gcf_product_20_90_l2717_271794

theorem lcm_gcf_product_20_90 : Nat.lcm 20 90 * Nat.gcd 20 90 = 1800 := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcf_product_20_90_l2717_271794


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_l2717_271718

theorem inscribed_circle_radius (A p r s : ℝ) : 
  A = 2 * p →  -- Area is twice the perimeter
  A = r * s →  -- Area formula using inradius and semiperimeter
  p = 2 * s →  -- Perimeter is twice the semiperimeter
  r = 4 := by sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_l2717_271718


namespace NUMINAMATH_CALUDE_marcia_pants_count_l2717_271716

/-- Represents the number of items in Marcia's wardrobe -/
structure Wardrobe where
  skirts : Nat
  blouses : Nat
  pants : Nat

/-- Represents the prices of items and the total budget -/
structure Prices where
  skirt_price : ℕ
  blouse_price : ℕ
  pant_price : ℕ
  total_budget : ℕ

/-- Calculates the cost of pants with the sale applied -/
def pants_cost (n : ℕ) (price : ℕ) : ℕ :=
  if n % 2 = 0 then
    n / 2 * price + n / 2 * (price / 2)
  else
    (n / 2 + 1) * price + (n / 2) * (price / 2)

/-- Theorem stating that Marcia needs to add 2 pairs of pants -/
theorem marcia_pants_count (w : Wardrobe) (p : Prices) : w.pants = 2 :=
  by
    have h1 : w.skirts = 3 := by sorry
    have h2 : w.blouses = 5 := by sorry
    have h3 : p.skirt_price = 20 := by sorry
    have h4 : p.blouse_price = 15 := by sorry
    have h5 : p.pant_price = 30 := by sorry
    have h6 : p.total_budget = 180 := by sorry
    
    have skirt_cost : ℕ := w.skirts * p.skirt_price
    have blouse_cost : ℕ := w.blouses * p.blouse_price
    have remaining_budget : ℕ := p.total_budget - (skirt_cost + blouse_cost)
    
    have pants_fit_budget : pants_cost w.pants p.pant_price = remaining_budget := by sorry
    
    sorry -- Complete the proof here

end NUMINAMATH_CALUDE_marcia_pants_count_l2717_271716


namespace NUMINAMATH_CALUDE_august_math_problems_l2717_271768

def problem (x y z : ℝ) : Prop :=
  let first_answer := x
  let second_answer := 2 * x - y
  let third_answer := 3 * x - z
  let fourth_answer := (x + (2 * x - y) + (3 * x - z)) / 3
  x = 600 ∧
  y > 0 ∧
  z = (x + (2 * x - y)) - 400 ∧
  first_answer + second_answer + third_answer + fourth_answer = 2933.33

theorem august_math_problems :
  ∃ (y z : ℝ), problem 600 y z :=
sorry

end NUMINAMATH_CALUDE_august_math_problems_l2717_271768


namespace NUMINAMATH_CALUDE_train_bridge_crossing_time_l2717_271747

/-- Proves that a train with given length and speed takes the specified time to cross a bridge of given length -/
theorem train_bridge_crossing_time
  (train_length : ℝ)
  (train_speed_kmh : ℝ)
  (bridge_length : ℝ)
  (h1 : train_length = 150)
  (h2 : train_speed_kmh = 45)
  (h3 : bridge_length = 225) :
  (train_length + bridge_length) / (train_speed_kmh * 1000 / 3600) = 30 := by
  sorry

#check train_bridge_crossing_time

end NUMINAMATH_CALUDE_train_bridge_crossing_time_l2717_271747


namespace NUMINAMATH_CALUDE_polynomial_factorization_l2717_271739

theorem polynomial_factorization (x y : ℝ) : 3 * x^2 - 3 * y^2 = 3 * (x + y) * (x - y) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l2717_271739


namespace NUMINAMATH_CALUDE_license_plate_count_l2717_271787

/-- The number of vowels in the alphabet -/
def num_vowels : ℕ := 5

/-- The number of consonants in the alphabet (including Y) -/
def num_consonants : ℕ := 21

/-- The number of digits (0-9) -/
def num_digits : ℕ := 10

/-- The total number of possible license plates -/
def total_license_plates : ℕ := num_consonants * num_consonants * num_vowels * num_vowels * num_digits

theorem license_plate_count :
  total_license_plates = 110250 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_count_l2717_271787


namespace NUMINAMATH_CALUDE_tank_capacity_l2717_271719

/-- The capacity of a tank given specific filling and draining rates and a cyclic operation pattern. -/
theorem tank_capacity 
  (fill_rate_A : ℕ) 
  (fill_rate_B : ℕ) 
  (drain_rate_C : ℕ) 
  (total_time : ℕ) 
  (h1 : fill_rate_A = 40)
  (h2 : fill_rate_B = 30)
  (h3 : drain_rate_C = 20)
  (h4 : total_time = 57) :
  fill_rate_A + fill_rate_B - drain_rate_C = 50 →
  (total_time / 3) * (fill_rate_A + fill_rate_B - drain_rate_C) + fill_rate_A = 990 := by
  sorry

#check tank_capacity

end NUMINAMATH_CALUDE_tank_capacity_l2717_271719


namespace NUMINAMATH_CALUDE_similar_triangles_shortest_side_l2717_271730

theorem similar_triangles_shortest_side
  (a b c : ℝ)  -- sides of the first triangle
  (k : ℝ)      -- scaling factor
  (h1 : a^2 + b^2 = c^2)  -- Pythagorean theorem for the first triangle
  (h2 : a = 15)           -- given side length of the first triangle
  (h3 : c = 17)           -- hypotenuse of the first triangle
  (h4 : k * c = 68)       -- hypotenuse of the second triangle
  : k * min a b = 32 :=
by sorry

end NUMINAMATH_CALUDE_similar_triangles_shortest_side_l2717_271730
