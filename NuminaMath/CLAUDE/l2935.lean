import Mathlib

namespace NUMINAMATH_CALUDE_largest_four_digit_divisible_by_50_l2935_293546

theorem largest_four_digit_divisible_by_50 :
  ∀ n : ℕ, 1000 ≤ n ∧ n ≤ 9999 ∧ n % 50 = 0 → n ≤ 9950 :=
by sorry

end NUMINAMATH_CALUDE_largest_four_digit_divisible_by_50_l2935_293546


namespace NUMINAMATH_CALUDE_cupcake_distribution_l2935_293507

def dozen : ℕ := 12

theorem cupcake_distribution (total_dozens : ℕ) (cupcakes_per_cousin : ℕ) : 
  total_dozens = 4 → cupcakes_per_cousin = 3 → (dozen * total_dozens) / cupcakes_per_cousin = 16 := by
  sorry

end NUMINAMATH_CALUDE_cupcake_distribution_l2935_293507


namespace NUMINAMATH_CALUDE_total_groceries_l2935_293528

def cookies : ℕ := 12
def noodles : ℕ := 16

theorem total_groceries : cookies + noodles = 28 := by
  sorry

end NUMINAMATH_CALUDE_total_groceries_l2935_293528


namespace NUMINAMATH_CALUDE_complement_intersection_S_T_l2935_293564

def S : Finset Int := {-2, -1, 0, 1, 2}
def T : Finset Int := {-1, 0, 1}

theorem complement_intersection_S_T :
  (S \ (S ∩ T)) = {-2, 2} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_S_T_l2935_293564


namespace NUMINAMATH_CALUDE_twenty_seven_thousand_six_hundred_scientific_notation_l2935_293558

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  significand : ℝ
  exponent : ℤ
  is_valid : 1 ≤ significand ∧ significand < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem twenty_seven_thousand_six_hundred_scientific_notation :
  toScientificNotation 27600 = ScientificNotation.mk 2.76 4 (by norm_num) :=
sorry

end NUMINAMATH_CALUDE_twenty_seven_thousand_six_hundred_scientific_notation_l2935_293558


namespace NUMINAMATH_CALUDE_smallest_number_l2935_293572

theorem smallest_number (S : Set ℤ) (h1 : S = {1, 0, -2, -3}) :
  ∃ x ∈ S, ∀ y ∈ S, x ≤ y ∧ x = -3 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_l2935_293572


namespace NUMINAMATH_CALUDE_cost_difference_l2935_293597

def vacation_cost (tom dorothy sammy : ℝ) : Prop :=
  tom + dorothy + sammy = 400 ∧ tom = 95 ∧ dorothy = 140 ∧ sammy = 165

theorem cost_difference (tom dorothy sammy t d : ℝ) 
  (h : vacation_cost tom dorothy sammy) :
  t - d = 45 :=
sorry

end NUMINAMATH_CALUDE_cost_difference_l2935_293597


namespace NUMINAMATH_CALUDE_pool_capacity_after_addition_l2935_293592

/-- Proves that adding 300 gallons to a pool with given conditions results in 40.38% capacity filled -/
theorem pool_capacity_after_addition
  (total_capacity : ℝ)
  (additional_water : ℝ)
  (increase_percentage : ℝ)
  (h1 : total_capacity = 1529.4117647058824)
  (h2 : additional_water = 300)
  (h3 : increase_percentage = 30)
  (h4 : (additional_water / total_capacity) * 100 = increase_percentage) :
  let final_percentage := (((increase_percentage / 100) * total_capacity) / total_capacity) * 100
  ∃ ε > 0, |final_percentage - 40.38| < ε :=
by sorry

end NUMINAMATH_CALUDE_pool_capacity_after_addition_l2935_293592


namespace NUMINAMATH_CALUDE_optimal_price_l2935_293557

def sales_volume (x : ℝ) : ℝ := -10 * x + 800

theorem optimal_price (production_cost : ℝ) (max_price : ℝ) (target_profit : ℝ) :
  production_cost = 20 →
  max_price = 45 →
  target_profit = 8000 →
  sales_volume 30 = 500 →
  sales_volume 40 = 400 →
  ∃ (price : ℝ), price ≤ max_price ∧
                 (price - production_cost) * sales_volume price = target_profit ∧
                 price = 40 := by
  sorry

end NUMINAMATH_CALUDE_optimal_price_l2935_293557


namespace NUMINAMATH_CALUDE_expression_simplification_l2935_293521

theorem expression_simplification (a b c d : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) (h : a + b + c = d) :
  1 / (b^2 + c^2 - a^2) + 1 / (a^2 + c^2 - b^2) + 1 / (a^2 + b^2 - c^2) = 0 := by
sorry


end NUMINAMATH_CALUDE_expression_simplification_l2935_293521


namespace NUMINAMATH_CALUDE_sequence_expression_evaluation_l2935_293568

theorem sequence_expression_evaluation :
  ∀ (x : ℝ),
  (∀ (n : ℕ), n > 0 → n = 2^(n-1) * x) →
  x = 1 →
  2*x * 6*x + 5*x / (4*x) - 56*x = 69/8 := by
  sorry

end NUMINAMATH_CALUDE_sequence_expression_evaluation_l2935_293568


namespace NUMINAMATH_CALUDE_circled_plus_two_three_four_l2935_293520

/-- The operation ⊕ is defined for real numbers a, b, and c. -/
def CircledPlus (a b c : ℝ) : ℝ := b^2 - 3*a*c

/-- Theorem: The value of ⊕(2, 3, 4) is -15. -/
theorem circled_plus_two_three_four :
  CircledPlus 2 3 4 = -15 := by
  sorry

end NUMINAMATH_CALUDE_circled_plus_two_three_four_l2935_293520


namespace NUMINAMATH_CALUDE_sum_s_r_x_is_negative_fifteen_l2935_293525

def r (x : ℝ) : ℝ := |x| - 3
def s (x : ℝ) : ℝ := -|x|

def x_values : List ℝ := [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]

theorem sum_s_r_x_is_negative_fifteen :
  (x_values.map (λ x => s (r x))).sum = -15 := by sorry

end NUMINAMATH_CALUDE_sum_s_r_x_is_negative_fifteen_l2935_293525


namespace NUMINAMATH_CALUDE_bill_calculation_l2935_293581

/-- Given an initial bill amount, calculate the final amount after applying two successive late charges -/
def final_bill_amount (initial_amount : ℝ) (first_charge_rate : ℝ) (second_charge_rate : ℝ) : ℝ :=
  initial_amount * (1 + first_charge_rate) * (1 + second_charge_rate)

/-- Theorem: The final bill amount after applying late charges is $525.30 -/
theorem bill_calculation : 
  final_bill_amount 500 0.02 0.03 = 525.30 := by
  sorry

#eval final_bill_amount 500 0.02 0.03

end NUMINAMATH_CALUDE_bill_calculation_l2935_293581


namespace NUMINAMATH_CALUDE_min_max_abs_quadratic_l2935_293594

theorem min_max_abs_quadratic (p q : ℝ) :
  (∃ (M : ℝ), M ≥ 1/2 ∧ ∀ (x : ℝ), -1 ≤ x ∧ x ≤ 1 → |x^2 + p*x + q| ≤ M) ∧
  (∀ (M : ℝ), (∀ (x : ℝ), -1 ≤ x ∧ x ≤ 1 → |x^2 + p*x + q| ≤ M) → M ≥ 1/2) :=
sorry

end NUMINAMATH_CALUDE_min_max_abs_quadratic_l2935_293594


namespace NUMINAMATH_CALUDE_rectangle_area_with_perimeter_and_breadth_l2935_293555

/-- Theorem: Area of a rectangle with given perimeter and breadth -/
theorem rectangle_area_with_perimeter_and_breadth
  (perimeter : ℝ) (breadth : ℝ) (h_perimeter : perimeter = 900)
  (h_breadth : breadth = 190) :
  let length : ℝ := perimeter / 2 - breadth
  let area : ℝ := length * breadth
  area = 49400 := by sorry

end NUMINAMATH_CALUDE_rectangle_area_with_perimeter_and_breadth_l2935_293555


namespace NUMINAMATH_CALUDE_at_least_one_geq_two_l2935_293537

theorem at_least_one_geq_two (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x + 1/y ≥ 2) ∨ (y + 1/z ≥ 2) ∨ (z + 1/x ≥ 2) :=
by sorry

end NUMINAMATH_CALUDE_at_least_one_geq_two_l2935_293537


namespace NUMINAMATH_CALUDE_geometric_sequence_sixth_term_l2935_293549

theorem geometric_sequence_sixth_term 
  (a : ℕ → ℝ) 
  (n : ℕ) 
  (h1 : n = 9) 
  (h2 : a 1 = 9) 
  (h3 : a n = 26244) 
  (h4 : ∀ i j : ℕ, 1 ≤ i ∧ i < j ∧ j ≤ n → a j / a i = a (i + 1) / a i) : 
  a 6 = 2187 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sixth_term_l2935_293549


namespace NUMINAMATH_CALUDE_reading_order_l2935_293503

variable (a b c d : ℝ)

theorem reading_order (h1 : a + c = b + d) (h2 : a + b > c + d) (h3 : d > b + c) :
  a > d ∧ d > b ∧ b > c := by
  sorry

end NUMINAMATH_CALUDE_reading_order_l2935_293503


namespace NUMINAMATH_CALUDE_larger_cube_volume_l2935_293542

-- Define the number of smaller cubes
def num_small_cubes : ℕ := 343

-- Define the volume of each smaller cube
def small_cube_volume : ℝ := 1

-- Define the surface area difference
def surface_area_difference : ℝ := 1764

-- Theorem statement
theorem larger_cube_volume :
  let large_cube_side : ℝ := (num_small_cubes : ℝ) ^ (1/3)
  let small_cube_side : ℝ := small_cube_volume ^ (1/3)
  let large_cube_volume : ℝ := large_cube_side ^ 3
  (num_small_cubes : ℝ) * (6 * small_cube_side ^ 2) - (6 * large_cube_side ^ 2) = surface_area_difference →
  large_cube_volume = num_small_cubes * small_cube_volume :=
by
  sorry

end NUMINAMATH_CALUDE_larger_cube_volume_l2935_293542


namespace NUMINAMATH_CALUDE_parrots_per_cage_l2935_293529

/-- Given a pet store with birds, calculate the number of parrots per cage. -/
theorem parrots_per_cage
  (num_cages : ℕ)
  (parakeets_per_cage : ℕ)
  (total_birds : ℕ)
  (h1 : num_cages = 6)
  (h2 : parakeets_per_cage = 7)
  (h3 : total_birds = 54) :
  (total_birds - num_cages * parakeets_per_cage) / num_cages = 2 :=
by sorry

end NUMINAMATH_CALUDE_parrots_per_cage_l2935_293529


namespace NUMINAMATH_CALUDE_quadratic_roots_inequality_l2935_293582

theorem quadratic_roots_inequality (a b c : ℝ) (ha : a ≠ 0) (habc : a + b + c = 0)
  (hx : ∃ x₁ x₂ : ℝ, 3 * a * x₁^2 + 2 * b * x₁ + c = 0 ∧ 3 * a * x₂^2 + 2 * b * x₂ + c = 0) :
  ∃ x₁ x₂ : ℝ, (3 * a * x₁^2 + 2 * b * x₁ + c = 0 ∧ 3 * a * x₂^2 + 2 * b * x₂ + c = 0) →
    1 / |2 * x₁ - 1| + 1 / |2 * x₂ - 1| ≥ 2 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_inequality_l2935_293582


namespace NUMINAMATH_CALUDE_equation_solution_l2935_293562

theorem equation_solution (x y : ℝ) : 
  x / 3 - y / 2 = 1 → y = 2 * x / 3 - 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2935_293562


namespace NUMINAMATH_CALUDE_ranch_minimum_animals_l2935_293515

theorem ranch_minimum_animals : ∀ (ponies horses : ℕ),
  ponies > 0 →
  horses = ponies + 4 →
  (5 * ponies) % 6 = 0 →
  (10 * ponies) % 18 = 0 →
  ponies + horses ≥ 40 ∧
  ∀ (p h : ℕ), p > 0 → h = p + 4 → (5 * p) % 6 = 0 → (10 * p) % 18 = 0 → p + h ≥ ponies + horses :=
by sorry

end NUMINAMATH_CALUDE_ranch_minimum_animals_l2935_293515


namespace NUMINAMATH_CALUDE_isosceles_minimizes_perimeter_l2935_293548

/-- Given a base length and area, the isosceles triangle minimizes the sum of the other two sides -/
theorem isosceles_minimizes_perimeter (a S : ℝ) (ha : a > 0) (hS : S > 0) :
  ∃ (h : ℝ), h > 0 ∧
  ∀ (b c : ℝ), b > 0 → c > 0 →
  (a * h / 2 = S) →
  (a * (b^2 - h^2).sqrt / 2 = S) →
  (a * (c^2 - h^2).sqrt / 2 = S) →
  b + c ≥ 2 * (4 * S^2 / a^2 + a^2 / 4).sqrt :=
sorry

end NUMINAMATH_CALUDE_isosceles_minimizes_perimeter_l2935_293548


namespace NUMINAMATH_CALUDE_cube_sum_problem_l2935_293595

theorem cube_sum_problem (x y z : ℝ) 
  (sum_eq : x + y + z = 8)
  (sum_prod_eq : x*y + x*z + y*z = 17)
  (prod_eq : x*y*z = -14) :
  x^3 + y^3 + z^3 = 62 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_problem_l2935_293595


namespace NUMINAMATH_CALUDE_sqrt_inequality_triangle_inequality_l2935_293579

-- Problem 1
theorem sqrt_inequality (a : ℝ) (ha : a > 0) :
  Real.sqrt (a + 5) - Real.sqrt (a + 3) > Real.sqrt (a + 6) - Real.sqrt (a + 4) := by
  sorry

-- Problem 2
theorem triangle_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hab : a + b > c) (hbc : b + c > a) (hca : c + a > b) :
  (a + b) / (1 + a + b) > c / (1 + c) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_inequality_triangle_inequality_l2935_293579


namespace NUMINAMATH_CALUDE_f_min_value_l2935_293580

/-- The quadratic function f(x) = 5x^2 - 15x - 2 -/
def f (x : ℝ) : ℝ := 5 * x^2 - 15 * x - 2

/-- The minimum value of f(x) is -13.25 -/
theorem f_min_value : ∃ (x : ℝ), f x = -13.25 ∧ ∀ (y : ℝ), f y ≥ -13.25 :=
sorry

end NUMINAMATH_CALUDE_f_min_value_l2935_293580


namespace NUMINAMATH_CALUDE_range_of_a_l2935_293505

theorem range_of_a (a : ℝ) : 
  (∀ x, -2 < x ∧ x < 3 → -2 < x ∧ x < a) ∧ 
  (∃ x, -2 < x ∧ x < a ∧ ¬(-2 < x ∧ x < 3)) 
  ↔ a > 3 := by sorry

end NUMINAMATH_CALUDE_range_of_a_l2935_293505


namespace NUMINAMATH_CALUDE_sum_always_positive_l2935_293589

/-- An increasing function on ℝ -/
def IncreasingFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

/-- An odd function -/
def OddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

theorem sum_always_positive
  (f : ℝ → ℝ)
  (a : ℕ → ℝ)
  (h_incr : IncreasingFunction f)
  (h_odd : OddFunction f)
  (h_arith : ArithmeticSequence a)
  (h_a3_pos : a 3 > 0) :
  f (a 1) + f (a 3) + f (a 5) > 0 :=
sorry

end NUMINAMATH_CALUDE_sum_always_positive_l2935_293589


namespace NUMINAMATH_CALUDE_ratio_x_to_y_l2935_293533

theorem ratio_x_to_y (x y : ℝ) (h : 3 * x = (12 / 100) * 250 * y) : x / y = 10 / 1 := by
  sorry

end NUMINAMATH_CALUDE_ratio_x_to_y_l2935_293533


namespace NUMINAMATH_CALUDE_arrangement_count_l2935_293543

/-- The number of ways to choose 2 items from a set of 4 items -/
def choose_2_from_4 : ℕ := 6

/-- The number of ways to arrange 3 items -/
def arrange_3 : ℕ := 6

/-- The total number of arrangements -/
def total_arrangements : ℕ := choose_2_from_4 * arrange_3

/-- Theorem: The number of ways to arrange 4 letters from the set {a, b, c, d, e, f},
    where a and b must be selected and adjacent (with a in front of b), is equal to 36 -/
theorem arrangement_count : total_arrangements = 36 := by
  sorry

end NUMINAMATH_CALUDE_arrangement_count_l2935_293543


namespace NUMINAMATH_CALUDE_acute_triangle_inequality_l2935_293538

/-- Triangle with acute angle opposite to side c -/
structure AcuteTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  ha : a > 0
  hb : b > 0
  hc : c > 0
  triangle_ineq : a + b > c ∧ b + c > a ∧ c + a > b
  acute_angle : c^2 < a^2 + b^2

theorem acute_triangle_inequality (t : AcuteTriangle) :
  (t.a^2 + t.b^2 + t.c^2) / (t.a^2 + t.b^2) > 1 :=
sorry

end NUMINAMATH_CALUDE_acute_triangle_inequality_l2935_293538


namespace NUMINAMATH_CALUDE_monic_quartic_polynomial_value_at_zero_l2935_293502

/-- A monic quartic polynomial is a polynomial of degree 4 with leading coefficient 1 -/
def MonicQuarticPolynomial (f : ℝ → ℝ) : Prop :=
  ∃ a b c d : ℝ, ∀ x, f x = x^4 + a*x^3 + b*x^2 + c*x + d

theorem monic_quartic_polynomial_value_at_zero 
  (f : ℝ → ℝ) 
  (hf : MonicQuarticPolynomial f) 
  (h1 : f (-2) = -4)
  (h2 : f 1 = -1)
  (h3 : f 3 = -9)
  (h4 : f 5 = -25) :
  f 0 = -30 := by
sorry

end NUMINAMATH_CALUDE_monic_quartic_polynomial_value_at_zero_l2935_293502


namespace NUMINAMATH_CALUDE_farmer_cows_problem_l2935_293508

theorem farmer_cows_problem (initial_food : ℝ) (initial_cows : ℕ) :
  initial_food > 0 →
  initial_cows > 0 →
  (initial_food / 50 = initial_food / (5 * 10)) →
  (initial_cows = 200) :=
by
  sorry

end NUMINAMATH_CALUDE_farmer_cows_problem_l2935_293508


namespace NUMINAMATH_CALUDE_razorback_shop_revenue_l2935_293526

theorem razorback_shop_revenue :
  let tshirt_price : ℕ := 98
  let hat_price : ℕ := 45
  let scarf_price : ℕ := 60
  let tshirts_sold : ℕ := 42
  let hats_sold : ℕ := 32
  let scarves_sold : ℕ := 15
  tshirt_price * tshirts_sold + hat_price * hats_sold + scarf_price * scarves_sold = 6456 :=
by sorry

end NUMINAMATH_CALUDE_razorback_shop_revenue_l2935_293526


namespace NUMINAMATH_CALUDE_geometric_sum_is_60_l2935_293518

/-- The sum of a geometric sequence with 4 terms, first term 4, and common ratio 2 -/
def geometric_sum : ℕ := 
  let a := 4  -- first term
  let r := 2  -- common ratio
  let n := 4  -- number of terms
  a * (r^n - 1) / (r - 1)

/-- Theorem stating that the geometric sum is equal to 60 -/
theorem geometric_sum_is_60 : geometric_sum = 60 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sum_is_60_l2935_293518


namespace NUMINAMATH_CALUDE_num_ways_to_sum_equals_two_pow_n_minus_one_l2935_293532

/-- The number of ways to express a positive integer as a sum of one or more positive integers. -/
def num_ways_to_sum (n : ℕ+) : ℕ :=
  2^(n.val - 1)

/-- Theorem: For any positive integer n, the number of ways to express n as a sum of one or more
    positive integers is equal to 2^(n-1). -/
theorem num_ways_to_sum_equals_two_pow_n_minus_one (n : ℕ+) :
  (num_ways_to_sum n) = 2^(n.val - 1) := by
  sorry

end NUMINAMATH_CALUDE_num_ways_to_sum_equals_two_pow_n_minus_one_l2935_293532


namespace NUMINAMATH_CALUDE_vote_ways_l2935_293506

/-- The number of ways an open vote can occur in a society of n members -/
def openVoteWays (n : ℕ) : ℕ := n^n

/-- The number of ways a secret vote can occur in a society of n members -/
def secretVoteWays (n : ℕ) : ℕ := Nat.choose (2*n - 1) (n - 1)

/-- Theorem stating the number of ways for open and secret votes in a society of n members -/
theorem vote_ways (n : ℕ) :
  (openVoteWays n = n^n) ∧ (secretVoteWays n = Nat.choose (2*n - 1) (n - 1)) := by
  sorry

end NUMINAMATH_CALUDE_vote_ways_l2935_293506


namespace NUMINAMATH_CALUDE_watermelon_slices_l2935_293566

theorem watermelon_slices (danny_watermelons : ℕ) (sister_watermelon : ℕ) 
  (sister_slices : ℕ) (total_slices : ℕ) (danny_slices : ℕ) : 
  danny_watermelons = 3 → 
  sister_watermelon = 1 → 
  sister_slices = 15 → 
  total_slices = 45 → 
  danny_watermelons * danny_slices + sister_watermelon * sister_slices = total_slices → 
  danny_slices = 10 := by
sorry

end NUMINAMATH_CALUDE_watermelon_slices_l2935_293566


namespace NUMINAMATH_CALUDE_current_speed_current_speed_is_3_l2935_293569

/-- The speed of the current in a river, given the man's rowing speed in still water,
    the distance covered downstream, and the time taken to cover that distance. -/
theorem current_speed (mans_speed : ℝ) (distance : ℝ) (time : ℝ) : ℝ :=
  let downstream_speed := distance / (time / 3600)
  downstream_speed - mans_speed

/-- Proof that the speed of the current is 3 kmph -/
theorem current_speed_is_3 :
  current_speed 15 0.06 11.999040076793857 = 3 := by
  sorry

end NUMINAMATH_CALUDE_current_speed_current_speed_is_3_l2935_293569


namespace NUMINAMATH_CALUDE_max_correct_answers_l2935_293560

theorem max_correct_answers (total_questions : ℕ) (correct_score : ℤ) (incorrect_score : ℤ) (total_score : ℤ) :
  total_questions = 25 →
  correct_score = 4 →
  incorrect_score = -3 →
  total_score = 57 →
  ∃ (correct incorrect unanswered : ℕ),
    correct + incorrect + unanswered = total_questions ∧
    correct_score * correct + incorrect_score * incorrect = total_score ∧
    correct ≤ 18 ∧
    ∀ c i u : ℕ,
      c + i + u = total_questions →
      correct_score * c + incorrect_score * i = total_score →
      c ≤ correct :=
by sorry

end NUMINAMATH_CALUDE_max_correct_answers_l2935_293560


namespace NUMINAMATH_CALUDE_smallest_angle_WYZ_l2935_293540

-- Define the angles
def angle_XYZ : ℝ := 36
def angle_XYW : ℝ := 15

-- Theorem to prove
theorem smallest_angle_WYZ : 
  let angle_WYZ := angle_XYZ - angle_XYW
  angle_WYZ = 21 := by sorry

end NUMINAMATH_CALUDE_smallest_angle_WYZ_l2935_293540


namespace NUMINAMATH_CALUDE_soccer_club_girls_l2935_293593

theorem soccer_club_girls (total_members : ℕ) (meeting_attendance : ℕ) 
  (h1 : total_members = 30)
  (h2 : meeting_attendance = 18)
  (h3 : ∃ (boys girls : ℕ), 
    boys + girls = total_members ∧ 
    boys + girls / 3 = meeting_attendance) :
  ∃ (girls : ℕ), girls = 18 ∧ 
    ∃ (boys : ℕ), boys + girls = total_members ∧ 
                   boys + girls / 3 = meeting_attendance :=
by sorry

end NUMINAMATH_CALUDE_soccer_club_girls_l2935_293593


namespace NUMINAMATH_CALUDE_fish_problem_l2935_293504

theorem fish_problem (total : ℕ) (carla_fish : ℕ) (kyle_fish : ℕ) (tasha_fish : ℕ) :
  total = 36 →
  carla_fish = 8 →
  kyle_fish = tasha_fish →
  total = carla_fish + kyle_fish + tasha_fish →
  kyle_fish = 14 := by
  sorry

end NUMINAMATH_CALUDE_fish_problem_l2935_293504


namespace NUMINAMATH_CALUDE_not_suff_not_nec_condition_l2935_293559

theorem not_suff_not_nec_condition (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  ¬(∀ a b, a > b → 1/a < 1/b) ∧ ¬(∀ a b, 1/a < 1/b → a > b) := by
  sorry

end NUMINAMATH_CALUDE_not_suff_not_nec_condition_l2935_293559


namespace NUMINAMATH_CALUDE_problem_hall_tilings_l2935_293516

/-- Represents a tiling configuration for a rectangular hall. -/
structure HallTiling where
  width : ℕ
  length : ℕ
  black_tiles : ℕ
  white_tiles : ℕ

/-- Counts the number of valid tiling configurations. -/
def countValidTilings (h : HallTiling) : ℕ :=
  sorry

/-- The specific hall configuration from the problem. -/
def problemHall : HallTiling :=
  { width := 2
  , length := 13
  , black_tiles := 11
  , white_tiles := 15 }

/-- Theorem stating that the number of valid tilings for the problem hall is 486. -/
theorem problem_hall_tilings :
  countValidTilings problemHall = 486 :=
sorry

end NUMINAMATH_CALUDE_problem_hall_tilings_l2935_293516


namespace NUMINAMATH_CALUDE_perpendicular_vector_t_value_l2935_293588

/-- Given vectors a and b, if a is perpendicular to (t*a + b), then t = -5 -/
theorem perpendicular_vector_t_value (a b : ℝ × ℝ) (t : ℝ) 
  (h1 : a = (1, -1))
  (h2 : b = (6, -4))
  (h3 : a.1 * (t * a.1 + b.1) + a.2 * (t * a.2 + b.2) = 0) :
  t = -5 := by sorry

end NUMINAMATH_CALUDE_perpendicular_vector_t_value_l2935_293588


namespace NUMINAMATH_CALUDE_circle_and_point_position_l2935_293571

-- Define the circle
def circle_equation (x y : ℝ) : Prop :=
  (x - 18/5)^2 + y^2 = 569/25

-- Define the points
def point_A : ℝ × ℝ := (1, 4)
def point_B : ℝ × ℝ := (3, 2)
def point_P : ℝ × ℝ := (2, 4)

-- Define what it means for a point to be on the circle
def on_circle (p : ℝ × ℝ) : Prop :=
  circle_equation p.1 p.2

-- Define what it means for a point to be inside the circle
def inside_circle (p : ℝ × ℝ) : Prop :=
  (p.1 - 18/5)^2 + p.2^2 < 569/25

-- Theorem statement
theorem circle_and_point_position :
  (on_circle point_A) ∧ 
  (on_circle point_B) ∧ 
  (inside_circle point_P) := by
  sorry

end NUMINAMATH_CALUDE_circle_and_point_position_l2935_293571


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l2935_293585

theorem min_value_reciprocal_sum (m n : ℝ) (h1 : m > 0) (h2 : n > 0) (h3 : m + n = 1) :
  (1 / m + 1 / n) ≥ 4 :=
by sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l2935_293585


namespace NUMINAMATH_CALUDE_infinitely_many_palindromes_l2935_293596

/-- A function that checks if a natural number is a palindrome in decimal representation -/
def isPalindrome (n : ℕ) : Prop := sorry

/-- The sequence defined in the problem -/
def x (n : ℕ) : ℕ := 2013 + 317 * n

/-- The main theorem to prove -/
theorem infinitely_many_palindromes :
  ∀ k : ℕ, ∃ n : ℕ, n ≥ k ∧ isPalindrome (x n) := by sorry

end NUMINAMATH_CALUDE_infinitely_many_palindromes_l2935_293596


namespace NUMINAMATH_CALUDE_candy_container_volume_l2935_293577

theorem candy_container_volume (a b c : ℕ) (h : a * b * c = 216) :
  (3 * a) * (2 * b) * (4 * c) = 5184 := by
  sorry

end NUMINAMATH_CALUDE_candy_container_volume_l2935_293577


namespace NUMINAMATH_CALUDE_perimeter_equality_y_approximation_l2935_293567

-- Define the side length of the square
def y : ℝ := sorry

-- Define the radius of the circle
def r : ℝ := 4

-- Theorem stating the equality of square perimeter and circle circumference
theorem perimeter_equality : 4 * y = 2 * Real.pi * r := sorry

-- Theorem stating the approximate value of y
theorem y_approximation : ∃ (ε : ℝ), ε < 0.005 ∧ |y - 6.28| < ε := sorry

end NUMINAMATH_CALUDE_perimeter_equality_y_approximation_l2935_293567


namespace NUMINAMATH_CALUDE_biased_coin_prob_l2935_293575

/-- The probability of getting heads for a biased coin -/
def h : ℚ := 2/5

/-- The number of flips -/
def n : ℕ := 4

/-- The probability of getting exactly k heads in n flips -/
def prob_k_heads (k : ℕ) : ℚ := 
  (n.choose k) * h^k * (1-h)^(n-k)

theorem biased_coin_prob : 
  prob_k_heads 1 = prob_k_heads 2 → 
  prob_k_heads 2 = 216/625 :=
by sorry

end NUMINAMATH_CALUDE_biased_coin_prob_l2935_293575


namespace NUMINAMATH_CALUDE_p_sufficient_for_q_l2935_293522

theorem p_sufficient_for_q : ∀ (x y : ℝ),
  (x - 1)^2 + (y - 1)^2 ≤ 2 →
  y ≥ x - 1 ∧ y ≥ 1 - x ∧ y ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_p_sufficient_for_q_l2935_293522


namespace NUMINAMATH_CALUDE_concentric_circles_area_ratio_l2935_293556

theorem concentric_circles_area_ratio : 
  let d₁ : ℝ := 2  -- diameter of smaller circle
  let d₂ : ℝ := 6  -- diameter of larger circle
  let r₁ : ℝ := d₁ / 2  -- radius of smaller circle
  let r₂ : ℝ := d₂ / 2  -- radius of larger circle
  let A₁ : ℝ := π * r₁^2  -- area of smaller circle
  let A₂ : ℝ := π * r₂^2  -- area of larger circle
  (A₂ - A₁) / A₁ = 8
  := by sorry

end NUMINAMATH_CALUDE_concentric_circles_area_ratio_l2935_293556


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l2935_293535

theorem arithmetic_mean_problem (y : ℝ) : 
  (8 + 17 + 23 + 7 + y) / 5 = 15 → y = 20 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l2935_293535


namespace NUMINAMATH_CALUDE_modulo_equivalence_unique_solution_l2935_293563

theorem modulo_equivalence_unique_solution : ∃! n : ℤ, 0 ≤ n ∧ n ≤ 11 ∧ n ≡ 15827 [ZMOD 12] := by
  sorry

end NUMINAMATH_CALUDE_modulo_equivalence_unique_solution_l2935_293563


namespace NUMINAMATH_CALUDE_cindys_calculation_l2935_293587

theorem cindys_calculation (x : ℝ) : (x - 8) / 4 = 24 → (x - 4) / 8 = 12.5 := by
  sorry

end NUMINAMATH_CALUDE_cindys_calculation_l2935_293587


namespace NUMINAMATH_CALUDE_triangle_side_length_l2935_293550

theorem triangle_side_length (x y z : ℝ) (Y Z : ℝ) :
  y = 7 →
  z = 3 →
  Real.cos (Y - Z) = 11 / 15 →
  x^2 = y^2 + z^2 - 2 * y * z * Real.cos Y →
  x = Real.sqrt 38.4 :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2935_293550


namespace NUMINAMATH_CALUDE_lilys_balance_proof_l2935_293584

/-- Calculates Lily's final account balance after a series of transactions --/
def lilys_final_balance (initial_amount shirt_cost book_price book_discount 
  savings_rate gift_percentage : ℚ) (num_books : ℕ) : ℚ :=
  let shoes_cost := 3 * shirt_cost
  let discounted_book_price := book_price * (1 - book_discount)
  let total_book_cost := (num_books : ℚ) * discounted_book_price
  let remaining_after_purchases := initial_amount - shirt_cost - shoes_cost - total_book_cost
  let savings := remaining_after_purchases / 2
  let savings_with_interest := savings * (1 + savings_rate)
  let gift_cost := savings_with_interest * gift_percentage
  savings_with_interest - gift_cost

theorem lilys_balance_proof :
  lilys_final_balance 55 7 8 0.2 0.2 0.25 4 = 0.63 := by
  sorry

end NUMINAMATH_CALUDE_lilys_balance_proof_l2935_293584


namespace NUMINAMATH_CALUDE_worst_quality_component_l2935_293599

theorem worst_quality_component (a b c d : ℝ) 
  (ha : a = 0.16) (hb : b = -0.12) (hc : c = -0.15) (hd : d = 0.11) : 
  abs a > abs b ∧ abs a > abs c ∧ abs a > abs d :=
sorry

end NUMINAMATH_CALUDE_worst_quality_component_l2935_293599


namespace NUMINAMATH_CALUDE_students_taking_one_subject_l2935_293531

theorem students_taking_one_subject (total_geometry : ℕ) (both_subjects : ℕ) (science_only : ℕ)
  (h1 : both_subjects = 15)
  (h2 : total_geometry = 30)
  (h3 : science_only = 18) :
  total_geometry - both_subjects + science_only = 33 := by
sorry

end NUMINAMATH_CALUDE_students_taking_one_subject_l2935_293531


namespace NUMINAMATH_CALUDE_dave_initial_boxes_l2935_293576

def boxes_given : ℕ := 5
def pieces_per_box : ℕ := 3
def pieces_left : ℕ := 21

theorem dave_initial_boxes : 
  ∃ (initial_boxes : ℕ), 
    initial_boxes * pieces_per_box = 
      boxes_given * pieces_per_box + pieces_left ∧
    initial_boxes = 12 :=
by sorry

end NUMINAMATH_CALUDE_dave_initial_boxes_l2935_293576


namespace NUMINAMATH_CALUDE_modular_inverse_3_mod_197_l2935_293586

theorem modular_inverse_3_mod_197 :
  ∃ x : ℕ, x < 197 ∧ (3 * x) % 197 = 1 :=
by
  use 66
  sorry

end NUMINAMATH_CALUDE_modular_inverse_3_mod_197_l2935_293586


namespace NUMINAMATH_CALUDE_mixture_replacement_theorem_l2935_293509

/-- The amount of mixture replaced to change the ratio from 7:5 to 7:9 -/
def mixture_replaced (initial_total : ℝ) (replaced : ℝ) : Prop :=
  let initial_a := 21
  let initial_b := initial_total - initial_a
  let new_b := initial_b + replaced
  (initial_a / initial_total = 7 / 12) ∧
  (initial_a / new_b = 7 / 9) ∧
  replaced = 12

/-- Theorem stating that 12 liters of mixture were replaced -/
theorem mixture_replacement_theorem :
  ∃ (initial_total : ℝ), mixture_replaced initial_total 12 := by
  sorry

end NUMINAMATH_CALUDE_mixture_replacement_theorem_l2935_293509


namespace NUMINAMATH_CALUDE_mean_temperature_l2935_293551

def temperatures : List ℝ := [-6.5, -3, -2, 4, 2.5]

theorem mean_temperature : (temperatures.sum / temperatures.length) = -1 := by
  sorry

end NUMINAMATH_CALUDE_mean_temperature_l2935_293551


namespace NUMINAMATH_CALUDE_intersection_product_l2935_293539

/-- Curve C in polar coordinates -/
def curve_C (ρ θ : ℝ) : Prop :=
  ρ^2 - 2*ρ*Real.cos θ - 3 = 0

/-- Line l in polar coordinates -/
def line_l (k : ℝ) (θ : ℝ) : Prop :=
  k > 0 ∧ θ ∈ Set.Ioo 0 (Real.pi / 2)

/-- Intersection points of curve C and line l -/
def intersection_points (ρ₁ ρ₂ θ : ℝ) : Prop :=
  curve_C ρ₁ θ ∧ curve_C ρ₂ θ ∧ ∃ k, line_l k θ

theorem intersection_product (ρ₁ ρ₂ θ : ℝ) :
  intersection_points ρ₁ ρ₂ θ → |ρ₁ * ρ₂| = 3 := by
  sorry

end NUMINAMATH_CALUDE_intersection_product_l2935_293539


namespace NUMINAMATH_CALUDE_probability_one_common_course_is_two_thirds_l2935_293541

def total_courses : ℕ := 4
def courses_per_person : ℕ := 2

def probability_one_common_course : ℚ :=
  let total_selections := Nat.choose total_courses courses_per_person * Nat.choose total_courses courses_per_person
  let no_common_courses := Nat.choose total_courses courses_per_person
  let all_common_courses := Nat.choose total_courses courses_per_person
  let one_common_course := total_selections - no_common_courses - all_common_courses
  ↑one_common_course / ↑total_selections

theorem probability_one_common_course_is_two_thirds :
  probability_one_common_course = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_probability_one_common_course_is_two_thirds_l2935_293541


namespace NUMINAMATH_CALUDE_dans_tshirt_production_rate_l2935_293517

/-- The time it takes Dan to make one t-shirt in the first hour -/
def time_per_shirt_first_hour (total_shirts : ℕ) (second_hour_rate : ℕ) (minutes_per_hour : ℕ) : ℕ :=
  let second_hour_shirts := minutes_per_hour / second_hour_rate
  let first_hour_shirts := total_shirts - second_hour_shirts
  minutes_per_hour / first_hour_shirts

/-- Theorem stating that it takes Dan 12 minutes to make one t-shirt in the first hour -/
theorem dans_tshirt_production_rate :
  time_per_shirt_first_hour 15 6 60 = 12 :=
by
  sorry

#eval time_per_shirt_first_hour 15 6 60

end NUMINAMATH_CALUDE_dans_tshirt_production_rate_l2935_293517


namespace NUMINAMATH_CALUDE_problem_solution_l2935_293583

theorem problem_solution (x y z : ℝ) (hx : x = 550) (hy : y = 104) (hz : z = Real.sqrt 20.8) :
  x - (y / z^2)^3 = 425 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2935_293583


namespace NUMINAMATH_CALUDE_prime_squared_plus_17_mod_12_l2935_293523

theorem prime_squared_plus_17_mod_12 (p : ℕ) (h_prime : Nat.Prime p) (h_gt_3 : p > 3) :
  (p^2 + 17) % 12 = 6 := by
  sorry

end NUMINAMATH_CALUDE_prime_squared_plus_17_mod_12_l2935_293523


namespace NUMINAMATH_CALUDE_tan_value_from_equation_l2935_293591

theorem tan_value_from_equation (x : ℝ) :
  (1 - Real.cos x + Real.sin x) / (1 + Real.cos x + Real.sin x) = -2 →
  Real.tan x = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_value_from_equation_l2935_293591


namespace NUMINAMATH_CALUDE_blake_change_l2935_293534

def oranges_cost : ℝ := 40
def apples_cost : ℝ := 50
def mangoes_cost : ℝ := 60
def strawberries_cost : ℝ := 30
def bananas_cost : ℝ := 20
def strawberries_discount : ℝ := 0.10
def bananas_discount : ℝ := 0.05
def blake_money : ℝ := 300

theorem blake_change :
  let discounted_strawberries := strawberries_cost * (1 - strawberries_discount)
  let discounted_bananas := bananas_cost * (1 - bananas_discount)
  let total_cost := oranges_cost + apples_cost + mangoes_cost + discounted_strawberries + discounted_bananas
  blake_money - total_cost = 104 := by sorry

end NUMINAMATH_CALUDE_blake_change_l2935_293534


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_equation_l2935_293510

theorem negation_of_existence (p : ℝ → Prop) : (¬ ∃ x, p x) ↔ (∀ x, ¬ p x) := by sorry

theorem negation_of_quadratic_equation :
  (¬ ∃ x : ℝ, x^2 + x + 1 = 0) ↔ (∀ x : ℝ, x^2 + x + 1 ≠ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_equation_l2935_293510


namespace NUMINAMATH_CALUDE_age_difference_is_zero_l2935_293545

/-- Given that Carlos and David were born on the same day in different years,
    prove that the age difference between them is 0 years. -/
theorem age_difference_is_zero (C D m : ℕ) : 
  C = D + m →
  C - 1 = 6 * (D - 1) →
  C = D^3 →
  m = 0 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_is_zero_l2935_293545


namespace NUMINAMATH_CALUDE_fraction_equality_l2935_293501

theorem fraction_equality (x : ℝ) : (5 + x) / (7 + x) = (3 + x) / (4 + x) ↔ x = -1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2935_293501


namespace NUMINAMATH_CALUDE_geometric_sequence_constant_l2935_293554

/-- A geometric sequence with positive terms. -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q ∧ a n > 0

/-- The theorem stating that if a geometric sequence satisfies the given condition,
    then it is a constant sequence. -/
theorem geometric_sequence_constant
  (a : ℕ → ℝ)
  (h_geometric : GeometricSequence a)
  (h_condition : (a 3 + a 11) / a 7 ≤ 2) :
  ∃ c : ℝ, ∀ n : ℕ, a n = c :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_constant_l2935_293554


namespace NUMINAMATH_CALUDE_arithmetic_sequence_difference_l2935_293519

/-- Given an arithmetic sequence with first term -3 and second term 5,
    the positive difference between the 1010th and 1000th terms is 80. -/
theorem arithmetic_sequence_difference : ∀ (a : ℕ → ℤ),
  (a 1 = -3) →
  (a 2 = 5) →
  (∀ n : ℕ, a (n + 1) - a n = a 2 - a 1) →
  |a 1010 - a 1000| = 80 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_difference_l2935_293519


namespace NUMINAMATH_CALUDE_golden_ratio_between_zero_and_one_l2935_293511

theorem golden_ratio_between_zero_and_one :
  let φ := (Real.sqrt 5 - 1) / 2
  0 < φ ∧ φ < 1 := by sorry

end NUMINAMATH_CALUDE_golden_ratio_between_zero_and_one_l2935_293511


namespace NUMINAMATH_CALUDE_right_triangle_area_l2935_293530

/-- Given a right triangle with perimeter 4 + √26 and median length 2 on the hypotenuse, its area is 5/2 -/
theorem right_triangle_area (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →  -- Positive side lengths
  a^2 + b^2 = c^2 →  -- Right triangle (Pythagorean theorem)
  a + b + c = 4 + Real.sqrt 26 →  -- Perimeter condition
  c / 2 = 2 →  -- Median length condition
  (1/2) * a * b = 5/2 := by  -- Area of the triangle
sorry

end NUMINAMATH_CALUDE_right_triangle_area_l2935_293530


namespace NUMINAMATH_CALUDE_binary_1011001_to_base5_l2935_293570

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (λ acc (i, bit) => acc + if bit then 2^i else 0) 0

def decimal_to_base5 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc else aux (m / 5) ((m % 5) :: acc)
    aux n []

theorem binary_1011001_to_base5 :
  decimal_to_base5 (binary_to_decimal [true, false, false, true, true, false, true]) = [3, 2, 4] :=
sorry

end NUMINAMATH_CALUDE_binary_1011001_to_base5_l2935_293570


namespace NUMINAMATH_CALUDE_tom_balloons_l2935_293590

theorem tom_balloons (initial_balloons : ℕ) (given_balloons : ℕ) : 
  initial_balloons = 30 → given_balloons = 16 → initial_balloons - given_balloons = 14 := by
  sorry

end NUMINAMATH_CALUDE_tom_balloons_l2935_293590


namespace NUMINAMATH_CALUDE_right_triangle_similarity_x_values_l2935_293524

theorem right_triangle_similarity_x_values :
  let segments : Finset ℝ := {1, 9, 5, x}
  ∃ (AB CD : ℝ) (a b c d : ℝ),
    AB ∈ segments ∧ CD ∈ segments ∧
    a ∈ segments ∧ b ∈ segments ∧ c ∈ segments ∧ d ∈ segments ∧
    a^2 + b^2 = AB^2 ∧ c^2 + d^2 = CD^2 ∧
    a / c = b / d ∧
    x > 0 →
    (∃! (s : Finset ℝ), s.card = 3 ∧ ∀ y ∈ s, ∃ (AB CD a b c d : ℝ),
      AB ∈ segments ∧ CD ∈ segments ∧
      a ∈ segments ∧ b ∈ segments ∧ c ∈ segments ∧ d ∈ segments ∧
      a^2 + b^2 = AB^2 ∧ c^2 + d^2 = CD^2 ∧
      a / c = b / d ∧
      y = x) :=
by
  sorry

end NUMINAMATH_CALUDE_right_triangle_similarity_x_values_l2935_293524


namespace NUMINAMATH_CALUDE_speed_ratio_is_five_sixths_l2935_293565

/-- Two objects A and B moving uniformly along perpendicular paths -/
structure MotionProblem where
  /-- Speed of object A in yards per minute -/
  vA : ℝ
  /-- Speed of object B in yards per minute -/
  vB : ℝ
  /-- Initial distance of B from point O in yards -/
  initialDistB : ℝ
  /-- Time when A and B are first equidistant from O in minutes -/
  t1 : ℝ
  /-- Time when A and B are again equidistant from O in minutes -/
  t2 : ℝ

/-- The theorem stating the ratio of speeds given the problem conditions -/
theorem speed_ratio_is_five_sixths (p : MotionProblem)
  (h1 : p.initialDistB = 500)
  (h2 : p.t1 = 2)
  (h3 : p.t2 = 10)
  (h4 : p.t1 * p.vA = abs (p.initialDistB - p.t1 * p.vB))
  (h5 : p.t2 * p.vA = abs (p.initialDistB - p.t2 * p.vB)) :
  p.vA / p.vB = 5 / 6 := by
  sorry


end NUMINAMATH_CALUDE_speed_ratio_is_five_sixths_l2935_293565


namespace NUMINAMATH_CALUDE_star_operation_result_l2935_293544

-- Define the set of elements
inductive Element : Type
  | one : Element
  | two : Element
  | three : Element
  | four : Element

-- Define the operation *
def star : Element → Element → Element
  | Element.one, Element.one => Element.four
  | Element.one, Element.two => Element.three
  | Element.one, Element.three => Element.two
  | Element.one, Element.four => Element.one
  | Element.two, Element.one => Element.three
  | Element.two, Element.two => Element.one
  | Element.two, Element.three => Element.four
  | Element.two, Element.four => Element.two
  | Element.three, Element.one => Element.two
  | Element.three, Element.two => Element.four
  | Element.three, Element.three => Element.one
  | Element.three, Element.four => Element.three
  | Element.four, Element.one => Element.one
  | Element.four, Element.two => Element.two
  | Element.four, Element.three => Element.three
  | Element.four, Element.four => Element.four

theorem star_operation_result :
  star (star Element.three Element.one) (star Element.four Element.two) = Element.one := by
  sorry

end NUMINAMATH_CALUDE_star_operation_result_l2935_293544


namespace NUMINAMATH_CALUDE_cost_of_dozen_pens_l2935_293598

theorem cost_of_dozen_pens (pen_cost pencil_cost : ℕ) : 
  (3 * pen_cost + 5 * pencil_cost = 240) →
  (pen_cost = 5 * pencil_cost) →
  (12 * pen_cost = 720) := by
  sorry

end NUMINAMATH_CALUDE_cost_of_dozen_pens_l2935_293598


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l2935_293500

theorem polynomial_divisibility (a b c d m : ℤ) 
  (h1 : (a * m^3 + b * m^2 + c * m + d) % 5 = 0)
  (h2 : d % 5 ≠ 0) :
  ∃ n : ℤ, (d * n^3 + c * n^2 + b * n + a) % 5 = 0 := by
sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l2935_293500


namespace NUMINAMATH_CALUDE_cost_price_percentage_l2935_293574

theorem cost_price_percentage (selling_price cost_price : ℝ) :
  selling_price > 0 →
  cost_price > 0 →
  (selling_price - cost_price) / cost_price = 11.11111111111111 / 100 →
  cost_price / selling_price = 9 / 10 := by
sorry

end NUMINAMATH_CALUDE_cost_price_percentage_l2935_293574


namespace NUMINAMATH_CALUDE_set_intersection_equality_l2935_293578

def S : Set Int := {s | ∃ n : Int, s = 2*n + 1}
def T : Set Int := {t | ∃ n : Int, t = 4*n + 1}

theorem set_intersection_equality : S ∩ T = T := by
  sorry

end NUMINAMATH_CALUDE_set_intersection_equality_l2935_293578


namespace NUMINAMATH_CALUDE_sports_club_membership_l2935_293512

theorem sports_club_membership (total : ℕ) (badminton : ℕ) (tennis : ℕ) (both : ℕ) :
  total = 30 →
  badminton = 17 →
  tennis = 21 →
  both = 10 →
  total - (badminton + tennis - both) = 2 :=
by sorry

end NUMINAMATH_CALUDE_sports_club_membership_l2935_293512


namespace NUMINAMATH_CALUDE_expand_product_l2935_293513

theorem expand_product (x : ℝ) : 2 * (x + 3) * (x + 6) = 2 * x^2 + 18 * x + 36 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l2935_293513


namespace NUMINAMATH_CALUDE_remainder_invariance_l2935_293514

theorem remainder_invariance (S A : ℤ) (K : ℤ) : 
  S % A = (S + A * K) % A := by sorry

end NUMINAMATH_CALUDE_remainder_invariance_l2935_293514


namespace NUMINAMATH_CALUDE_salary_calculation_l2935_293536

def initial_salary : ℝ := 5000

def final_salary (s : ℝ) : ℝ :=
  let s1 := s * 1.3
  let s2 := s1 * 0.93
  let s3 := s2 * 0.8
  let s4 := s3 - 100
  let s5 := s4 * 1.1
  let s6 := s5 * 0.9
  s6 * 0.75

theorem salary_calculation :
  final_salary initial_salary = 3516.48 := by sorry

end NUMINAMATH_CALUDE_salary_calculation_l2935_293536


namespace NUMINAMATH_CALUDE_quadratic_equation_m_value_l2935_293553

/-- 
Given that (m+2)x^(m^2-2) + 2x + 1 = 0 is a quadratic equation in x and m+2 ≠ 0, 
prove that m = 2.
-/
theorem quadratic_equation_m_value (m : ℝ) : 
  (∀ x, ∃ a b c : ℝ, (m + 2) * x^(m^2 - 2) + 2*x + 1 = a*x^2 + b*x + c) →
  (m + 2 ≠ 0) →
  m = 2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_m_value_l2935_293553


namespace NUMINAMATH_CALUDE_gcd_inequality_l2935_293561

theorem gcd_inequality (n d₁ d₂ : ℕ+) : 
  (Nat.gcd n (d₁ + d₂) : ℚ) / (Nat.gcd n d₁ * Nat.gcd n d₂) ≥ 1 / n.val :=
by sorry

end NUMINAMATH_CALUDE_gcd_inequality_l2935_293561


namespace NUMINAMATH_CALUDE_power_4_2023_mod_17_l2935_293547

theorem power_4_2023_mod_17 : 4^2023 % 17 = 13 := by
  sorry

end NUMINAMATH_CALUDE_power_4_2023_mod_17_l2935_293547


namespace NUMINAMATH_CALUDE_locus_of_equal_power_l2935_293573

/-- Given two non-concentric circles in a plane, the locus of points with equal power
    relative to both circles is a straight line. -/
theorem locus_of_equal_power (R₁ R₂ a : ℝ) (ha : a ≠ 0) :
  ∃ k : ℝ, ∀ x y : ℝ, ((x + a)^2 + y^2 - R₁^2 = (x - a)^2 + y^2 - R₂^2) ↔ (x = k) :=
by sorry

end NUMINAMATH_CALUDE_locus_of_equal_power_l2935_293573


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l2935_293527

/-- Given a geometric sequence {a_n} with positive terms where a₁, (1/2)a₃, and 2a₂ form an arithmetic sequence,
    prove that (a₉ + a₁₀) / (a₇ + a₈) = 3 + 2√2 -/
theorem geometric_sequence_ratio (a : ℕ → ℝ) (h_pos : ∀ n, a n > 0)
  (h_geom : ∃ q : ℝ, ∀ n, a (n + 1) = q * a n)
  (h_arith : a 1 + 2 * a 2 = a 3) :
  (a 9 + a 10) / (a 7 + a 8) = 3 + 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l2935_293527


namespace NUMINAMATH_CALUDE_max_value_x_minus_2y_l2935_293552

theorem max_value_x_minus_2y (x y : ℝ) (h : x^2 + y^2 - 2*x + 4*y = 0) :
  x - 2*y ≤ 10 :=
by sorry

end NUMINAMATH_CALUDE_max_value_x_minus_2y_l2935_293552
