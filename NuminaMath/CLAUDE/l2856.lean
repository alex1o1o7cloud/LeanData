import Mathlib

namespace NUMINAMATH_CALUDE_prob_higher_2012_l2856_285640

-- Define the probability of guessing correctly
def p : ℝ := 0.25

-- Define the complementary probability
def q : ℝ := 1 - p

-- Define the binomial probability function
def binomProb (n : ℕ) (k : ℕ) : ℝ :=
  (n.choose k) * (p ^ k) * (q ^ (n - k))

-- Define the probability of passing in 2011
def prob2011 : ℝ :=
  1 - (binomProb 20 0 + binomProb 20 1 + binomProb 20 2)

-- Define the probability of passing in 2012
def prob2012 : ℝ :=
  1 - (binomProb 40 0 + binomProb 40 1 + binomProb 40 2 + binomProb 40 3 + binomProb 40 4 + binomProb 40 5)

-- Theorem statement
theorem prob_higher_2012 : prob2012 > prob2011 := by
  sorry

end NUMINAMATH_CALUDE_prob_higher_2012_l2856_285640


namespace NUMINAMATH_CALUDE_roots_of_equation_l2856_285616

theorem roots_of_equation (x : ℝ) :
  x * (x - 3)^2 * (5 + x) = 0 ↔ x = 0 ∨ x = 3 ∨ x = -5 := by
  sorry

end NUMINAMATH_CALUDE_roots_of_equation_l2856_285616


namespace NUMINAMATH_CALUDE_mean_of_remaining_numbers_l2856_285625

def numbers : List ℕ := [1867, 1993, 2019, 2025, 2109, 2121]

theorem mean_of_remaining_numbers :
  ∀ (four_nums : List ℕ),
    four_nums.length = 4 →
    four_nums.all (· ∈ numbers) →
    (four_nums.sum : ℚ) / 4 = 2008 →
    let remaining_nums := numbers.filter (· ∉ four_nums)
    (remaining_nums.sum : ℚ) / 2 = 2051 := by
  sorry

end NUMINAMATH_CALUDE_mean_of_remaining_numbers_l2856_285625


namespace NUMINAMATH_CALUDE_factorial_equation_solutions_l2856_285681

theorem factorial_equation_solutions :
  ∀ a b c : ℕ+,
    a^2 + b^2 + 1 = c! →
    ((a = 2 ∧ b = 1 ∧ c = 3) ∨ (a = 1 ∧ b = 2 ∧ c = 3)) :=
by sorry

end NUMINAMATH_CALUDE_factorial_equation_solutions_l2856_285681


namespace NUMINAMATH_CALUDE_solution_problem_l2856_285678

theorem solution_problem (x y : ℕ) 
  (h1 : 0 < x ∧ x < 30) 
  (h2 : 0 < y ∧ y < 30) 
  (h3 : x + y + x * y = 104) : 
  x + y = 20 := by
sorry

end NUMINAMATH_CALUDE_solution_problem_l2856_285678


namespace NUMINAMATH_CALUDE_inequality_proof_l2856_285621

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_sum : a + b + c = 1) : 
  (a^2 + b^2 + c^2) * (a / (b + c) + b / (c + a) + c / (a + b)) ≥ 1/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2856_285621


namespace NUMINAMATH_CALUDE_triangle_angle_cosine_inequality_l2856_285628

theorem triangle_angle_cosine_inequality (α β γ : Real) 
  (h : α + β + γ = Real.pi) : 
  Real.cos (α + Real.pi / 3) + Real.cos (β + Real.pi / 3) + Real.cos (γ + Real.pi / 3) + 3 / 2 ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_cosine_inequality_l2856_285628


namespace NUMINAMATH_CALUDE_part_one_part_two_l2856_285608

-- Define the logarithm function (base 10)
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Part I
theorem part_one : lg 24 - lg 3 - lg 4 + lg 5 = 1 := by sorry

-- Part II
theorem part_two : (((3 : ℝ) ^ (1/3) * (2 : ℝ) ^ (1/2)) ^ 6) + 
                   (((3 : ℝ) * (3 : ℝ) ^ (1/2)) ^ (1/2)) ^ (4/3) - 
                   ((2 : ℝ) ^ (1/4)) * (8 : ℝ) ^ (1/4) - 
                   (2015 : ℝ) ^ 0 = 72 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l2856_285608


namespace NUMINAMATH_CALUDE_six_star_three_equals_three_l2856_285644

-- Define the * operation
def star (a b : ℤ) : ℤ := 4*a + 5*b - 2*a*b

-- Theorem statement
theorem six_star_three_equals_three : star 6 3 = 3 := by sorry

end NUMINAMATH_CALUDE_six_star_three_equals_three_l2856_285644


namespace NUMINAMATH_CALUDE_unique_y_for_diamond_eq_21_l2856_285655

def diamond (x y : ℝ) : ℝ := 5 * x - 4 * y + 2 * x * y + 1

theorem unique_y_for_diamond_eq_21 : ∃! y : ℝ, diamond 4 y = 21 := by
  sorry

end NUMINAMATH_CALUDE_unique_y_for_diamond_eq_21_l2856_285655


namespace NUMINAMATH_CALUDE_candy_problem_l2856_285654

theorem candy_problem (total_candies : ℕ) : 
  (∃ (n : ℕ), n > 10 ∧ total_candies = 3 * (n - 1) + 2) ∧ 
  (∃ (m : ℕ), m < 10 ∧ total_candies = 4 * (m - 1) + 3) →
  total_candies = 35 := by
sorry

end NUMINAMATH_CALUDE_candy_problem_l2856_285654


namespace NUMINAMATH_CALUDE_business_value_l2856_285667

/-- Given a man who owns 2/3 of a business and sells 3/4 of his shares for 45,000 Rs,
    prove that the value of the entire business is 90,000 Rs. -/
theorem business_value (man_share : ℚ) (sold_portion : ℚ) (sold_value : ℕ) :
  man_share = 2/3 →
  sold_portion = 3/4 →
  sold_value = 45000 →
  ∃ (total_value : ℕ), total_value = 90000 ∧
    (total_value : ℚ) = sold_value / (man_share * sold_portion) :=
by sorry

end NUMINAMATH_CALUDE_business_value_l2856_285667


namespace NUMINAMATH_CALUDE_inequality_system_solution_l2856_285642

theorem inequality_system_solution (a b : ℝ) : 
  (∀ x : ℝ, (x - 1 ≥ a ∧ 2*x - b < 3) ↔ (3 ≤ x ∧ x < 5)) → 
  a + b = 9 := by
sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l2856_285642


namespace NUMINAMATH_CALUDE_smallest_prime_dividing_sum_l2856_285698

theorem smallest_prime_dividing_sum : 
  ∀ p : Nat, Prime p → p ∣ (2^14 + 7^9) → p ≥ 7 :=
by sorry

end NUMINAMATH_CALUDE_smallest_prime_dividing_sum_l2856_285698


namespace NUMINAMATH_CALUDE_double_volume_double_capacity_l2856_285632

/-- Represents the capacity of a container in number of marbles -/
def ContainerCapacity (volume : ℝ) : ℝ := sorry

theorem double_volume_double_capacity :
  let v₁ : ℝ := 36
  let v₂ : ℝ := 72
  let c₁ : ℝ := 120
  ContainerCapacity v₁ = c₁ →
  ContainerCapacity v₂ = 2 * c₁ :=
by sorry

end NUMINAMATH_CALUDE_double_volume_double_capacity_l2856_285632


namespace NUMINAMATH_CALUDE_isosceles_triangle_angles_l2856_285683

-- Define an isosceles triangle with an exterior angle of 140°
structure IsoscelesTriangle where
  angles : Fin 3 → ℝ
  isIsosceles : (angles 0 = angles 1) ∨ (angles 1 = angles 2) ∨ (angles 0 = angles 2)
  sumOfAngles : angles 0 + angles 1 + angles 2 = 180
  exteriorAngle : ℝ
  exteriorAngleValue : exteriorAngle = 140

-- Theorem statement
theorem isosceles_triangle_angles (t : IsoscelesTriangle) :
  (t.angles 0 = 40 ∧ t.angles 1 = 40 ∧ t.angles 2 = 100) ∨
  (t.angles 0 = 70 ∧ t.angles 1 = 70 ∧ t.angles 2 = 40) :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_angles_l2856_285683


namespace NUMINAMATH_CALUDE_prob_sum_six_is_five_thirty_sixths_l2856_285615

/-- The number of possible outcomes when throwing two dice -/
def total_outcomes : ℕ := 36

/-- The number of favorable outcomes (sum equals 6) when throwing two dice -/
def favorable_outcomes : ℕ := 5

/-- The probability of the sum of two fair dice equaling 6 -/
def prob_sum_six : ℚ := favorable_outcomes / total_outcomes

theorem prob_sum_six_is_five_thirty_sixths : 
  prob_sum_six = 5 / 36 := by sorry

end NUMINAMATH_CALUDE_prob_sum_six_is_five_thirty_sixths_l2856_285615


namespace NUMINAMATH_CALUDE_find_a_l2856_285670

-- Define the universal set U
def U (a : ℝ) : Set ℝ := {2, 4, 3 - a^2}

-- Define set P
def P (a : ℝ) : Set ℝ := {2, a^2 - a + 2}

-- Define the complement of P with respect to U
def complementP (a : ℝ) : Set ℝ := {-1}

-- Theorem statement
theorem find_a : ∃ a : ℝ, 
  (U a = P a ∪ complementP a) ∧ 
  (U a = {2, 4, 3 - a^2}) ∧ 
  (P a = {2, a^2 - a + 2}) ∧ 
  (complementP a = {-1}) →
  a = 2 := by sorry

end NUMINAMATH_CALUDE_find_a_l2856_285670


namespace NUMINAMATH_CALUDE_sunflower_cost_l2856_285622

theorem sunflower_cost
  (num_roses : ℕ)
  (num_sunflowers : ℕ)
  (cost_per_rose : ℚ)
  (total_cost : ℚ)
  (h1 : num_roses = 24)
  (h2 : num_sunflowers = 3)
  (h3 : cost_per_rose = 3/2)
  (h4 : total_cost = 45) :
  (total_cost - num_roses * cost_per_rose) / num_sunflowers = 3 := by
sorry

end NUMINAMATH_CALUDE_sunflower_cost_l2856_285622


namespace NUMINAMATH_CALUDE_point_on_coordinate_axes_l2856_285618

/-- A point in a 2D Cartesian coordinate system -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- The coordinate axes in a 2D Cartesian coordinate system -/
def CoordinateAxes : Set Point2D :=
  {p : Point2D | p.x = 0 ∨ p.y = 0}

/-- 
Given a point M(a,b) in a Cartesian coordinate system where ab = 0, 
prove that M is located on the coordinate axes.
-/
theorem point_on_coordinate_axes (M : Point2D) (h : M.x * M.y = 0) : 
  M ∈ CoordinateAxes := by
  sorry


end NUMINAMATH_CALUDE_point_on_coordinate_axes_l2856_285618


namespace NUMINAMATH_CALUDE_soda_discount_percentage_l2856_285602

-- Define the given constants
def full_price : ℚ := 30
def group_size : ℕ := 10
def num_children : ℕ := 4
def soda_price : ℚ := 5
def total_paid : ℚ := 197

-- Define the calculation functions
def adult_price := full_price
def child_price := full_price / 2

def total_price_without_discount : ℚ :=
  (group_size - num_children) * adult_price + num_children * child_price

def price_paid_for_tickets : ℚ := total_paid - soda_price

def discount_amount : ℚ := total_price_without_discount - price_paid_for_tickets

def discount_percentage : ℚ := (discount_amount / total_price_without_discount) * 100

-- State the theorem
theorem soda_discount_percentage : discount_percentage = 20 := by sorry

end NUMINAMATH_CALUDE_soda_discount_percentage_l2856_285602


namespace NUMINAMATH_CALUDE_midpoint_coordinates_sum_l2856_285606

/-- Given that M(-1,6) is the midpoint of CD and C(5,4) is one endpoint, 
    the sum of the coordinates of point D is 1. -/
theorem midpoint_coordinates_sum (C D M : ℝ × ℝ) : 
  C = (5, 4) → 
  M = (-1, 6) → 
  M = ((C.1 + D.1) / 2, (C.2 + D.2) / 2) → 
  D.1 + D.2 = 1 := by
sorry

end NUMINAMATH_CALUDE_midpoint_coordinates_sum_l2856_285606


namespace NUMINAMATH_CALUDE_solve_system_l2856_285661

theorem solve_system (x y : ℝ) (hx : x > 1) (hy : y > 1)
  (h1 : 1/x + 1/y = 3/2) (h2 : x*y = 9) : y = 6 := by
  sorry

end NUMINAMATH_CALUDE_solve_system_l2856_285661


namespace NUMINAMATH_CALUDE_baker_remaining_cakes_l2856_285676

theorem baker_remaining_cakes (total_cakes sold_cakes : ℕ) 
  (h1 : total_cakes = 155)
  (h2 : sold_cakes = 140) :
  total_cakes - sold_cakes = 15 := by
  sorry

end NUMINAMATH_CALUDE_baker_remaining_cakes_l2856_285676


namespace NUMINAMATH_CALUDE_jellybeans_per_child_l2856_285637

theorem jellybeans_per_child 
  (initial_jellybeans : ℕ) 
  (normal_class_size : ℕ) 
  (absent_children : ℕ) 
  (remaining_jellybeans : ℕ) 
  (h1 : initial_jellybeans = 100)
  (h2 : normal_class_size = 24)
  (h3 : absent_children = 2)
  (h4 : remaining_jellybeans = 34)
  : (initial_jellybeans - remaining_jellybeans) / (normal_class_size - absent_children) = 3 := by
  sorry

end NUMINAMATH_CALUDE_jellybeans_per_child_l2856_285637


namespace NUMINAMATH_CALUDE_one_female_selection_l2856_285694

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of male students in group A -/
def maleA : ℕ := 5

/-- The number of female students in group A -/
def femaleA : ℕ := 3

/-- The number of male students in group B -/
def maleB : ℕ := 6

/-- The number of female students in group B -/
def femaleB : ℕ := 2

/-- The number of students to be selected from each group -/
def selectPerGroup : ℕ := 2

/-- The total number of ways to select exactly one female student among 4 chosen students -/
theorem one_female_selection : 
  (choose femaleA 1 * choose maleA 1 * choose maleB selectPerGroup) + 
  (choose femaleB 1 * choose maleB 1 * choose maleA selectPerGroup) = 345 := by
  sorry

end NUMINAMATH_CALUDE_one_female_selection_l2856_285694


namespace NUMINAMATH_CALUDE_sum_of_roots_l2856_285658

theorem sum_of_roots (x₁ x₂ : ℝ) : 
  (x₁^2 - 3*x₁ + 2 = 0) → (x₂^2 - 3*x₂ + 2 = 0) → x₁ + x₂ = 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_l2856_285658


namespace NUMINAMATH_CALUDE_count_integers_satisfying_inequality_l2856_285690

theorem count_integers_satisfying_inequality :
  ∃! (S : Finset ℤ), (∀ n ∈ S, (3 * (n - 1) * (n + 5) : ℤ) < 0) ∧ S.card = 5 := by
  sorry

end NUMINAMATH_CALUDE_count_integers_satisfying_inequality_l2856_285690


namespace NUMINAMATH_CALUDE_fred_limes_picked_l2856_285638

theorem fred_limes_picked (total_limes : ℕ) (alyssa_limes : ℕ) (nancy_limes : ℕ) 
  (h1 : total_limes = 103)
  (h2 : alyssa_limes = 32)
  (h3 : nancy_limes = 35) :
  total_limes - (alyssa_limes + nancy_limes) = 36 := by
sorry

end NUMINAMATH_CALUDE_fred_limes_picked_l2856_285638


namespace NUMINAMATH_CALUDE_regular_polygon_140_degree_interior_l2856_285609

/-- A regular polygon with interior angles measuring 140° has 9 sides. -/
theorem regular_polygon_140_degree_interior : ∀ n : ℕ, 
  n > 2 → -- ensure it's a valid polygon
  (180 * (n - 2) : ℝ) = (140 * n : ℝ) → -- sum of interior angles formula
  n = 9 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_140_degree_interior_l2856_285609


namespace NUMINAMATH_CALUDE_leak_emptying_time_l2856_285688

theorem leak_emptying_time (fill_time_no_leak fill_time_with_leak : ℝ) 
  (h1 : fill_time_no_leak = 8)
  (h2 : fill_time_with_leak = 12) :
  let fill_rate := 1 / fill_time_no_leak
  let combined_rate := 1 / fill_time_with_leak
  let leak_rate := fill_rate - combined_rate
  24 = 1 / leak_rate := by
sorry

end NUMINAMATH_CALUDE_leak_emptying_time_l2856_285688


namespace NUMINAMATH_CALUDE_rebecca_tips_calculation_l2856_285693

/-- Rebecca's hair salon earnings calculation -/
def rebeccaEarnings (haircut_price perm_price dye_price dye_cost : ℕ) 
  (num_haircuts num_perms num_dyes : ℕ) (total_end_day : ℕ) : ℕ :=
  let service_earnings := haircut_price * num_haircuts + perm_price * num_perms + dye_price * num_dyes
  let dye_costs := dye_cost * num_dyes
  let tips := total_end_day - (service_earnings - dye_costs)
  tips

/-- Theorem stating that Rebecca's tips are $50 given the problem conditions -/
theorem rebecca_tips_calculation :
  rebeccaEarnings 30 40 60 10 4 1 2 310 = 50 := by
  sorry

end NUMINAMATH_CALUDE_rebecca_tips_calculation_l2856_285693


namespace NUMINAMATH_CALUDE_custom_mul_theorem_l2856_285605

/-- Custom multiplication operation -/
def custom_mul (m : ℚ) (x y : ℚ) : ℚ := (x * y) / (m * x + 2 * y)

theorem custom_mul_theorem (m : ℚ) :
  custom_mul m 1 2 = 2/5 →
  m = 1 ∧ custom_mul m 2 6 = 6/7 := by
  sorry

end NUMINAMATH_CALUDE_custom_mul_theorem_l2856_285605


namespace NUMINAMATH_CALUDE_product_expansion_l2856_285613

theorem product_expansion (x : ℝ) (hx : x ≠ 0) :
  (3 / 4) * ((8 / x^2) + 5*x - 6) = 6 / x^2 + (15*x) / 4 - 4.5 := by
  sorry

end NUMINAMATH_CALUDE_product_expansion_l2856_285613


namespace NUMINAMATH_CALUDE_ray_AB_bisects_angle_PAQ_l2856_285611

-- Define the circle C
def circle_C (x y : ℝ) : Prop :=
  (x - 2)^2 + (y - 5/2)^2 = 25/4

-- Define points T, A, and B
def point_T : ℝ × ℝ := (2, 0)
def point_A : ℝ × ℝ := (0, 4)
def point_B : ℝ × ℝ := (0, 1)

-- Define the ellipse
def ellipse (x y : ℝ) : Prop :=
  x^2/8 + y^2/4 = 1

-- Define line l passing through B
def line_l (x y : ℝ) : Prop :=
  ∃ k, y = k * x + 1

-- Define points P and Q as intersections of line l and the ellipse
def point_P : ℝ × ℝ := sorry
def point_Q : ℝ × ℝ := sorry

-- State the theorem
theorem ray_AB_bisects_angle_PAQ :
  circle_C point_T.1 point_T.2 ∧
  circle_C point_A.1 point_A.2 ∧
  circle_C point_B.1 point_B.2 ∧
  point_A.2 > point_B.2 ∧
  point_A.2 - point_B.2 = 3 ∧
  line_l point_P.1 point_P.2 ∧
  line_l point_Q.1 point_Q.2 ∧
  ellipse point_P.1 point_P.2 ∧
  ellipse point_Q.1 point_Q.2 →
  -- The conclusion that ray AB bisects angle PAQ
  -- This would typically involve showing that the angles are equal
  -- or that the dot product of vectors is zero, but we'll leave it as 'sorry'
  sorry :=
sorry

end NUMINAMATH_CALUDE_ray_AB_bisects_angle_PAQ_l2856_285611


namespace NUMINAMATH_CALUDE_special_arithmetic_sequence_101st_term_l2856_285669

/-- An arithmetic sequence where the square of each term equals the sum of the first 2n-1 terms. -/
def SpecialArithmeticSequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a n ≠ 0) ∧
  (∃ d, ∀ n, a (n + 1) = a n + d) ∧
  (∀ n, (a n)^2 = (2 * n - 1) * (a 1 + a n) / 2)

theorem special_arithmetic_sequence_101st_term
  (a : ℕ → ℝ) (h : SpecialArithmeticSequence a) : a 101 = 201 := by
  sorry

end NUMINAMATH_CALUDE_special_arithmetic_sequence_101st_term_l2856_285669


namespace NUMINAMATH_CALUDE_red_peaches_count_l2856_285646

/-- The number of baskets of peaches -/
def num_baskets : ℕ := 6

/-- The number of red peaches in each basket -/
def red_peaches_per_basket : ℕ := 16

/-- The number of green peaches in each basket -/
def green_peaches_per_basket : ℕ := 18

/-- The total number of red peaches in all baskets -/
def total_red_peaches : ℕ := num_baskets * red_peaches_per_basket

theorem red_peaches_count : total_red_peaches = 96 := by
  sorry

end NUMINAMATH_CALUDE_red_peaches_count_l2856_285646


namespace NUMINAMATH_CALUDE_range_of_c_l2856_285684

theorem range_of_c (a c : ℝ) : 
  (∀ x > 0, 2*x + a/x ≥ c) → 
  (a ≥ 1/8 → ∀ x > 0, 2*x + a/x ≥ c) → 
  (∃ a < 1/8, ∀ x > 0, 2*x + a/x ≥ c) → 
  c ≤ 1 := by sorry

end NUMINAMATH_CALUDE_range_of_c_l2856_285684


namespace NUMINAMATH_CALUDE_least_positive_integer_with_remainder_one_l2856_285660

theorem least_positive_integer_with_remainder_one (n : ℕ) : n = 421 ↔ 
  (n > 1) ∧ 
  (∀ d ∈ ({3, 4, 5, 6, 7, 10, 12} : Set ℕ), n % d = 1) ∧
  (∀ m : ℕ, m > 1 → (∀ d ∈ ({3, 4, 5, 6, 7, 10, 12} : Set ℕ), m % d = 1) → m ≥ n) :=
by sorry

end NUMINAMATH_CALUDE_least_positive_integer_with_remainder_one_l2856_285660


namespace NUMINAMATH_CALUDE_power_eleven_mod_hundred_l2856_285677

theorem power_eleven_mod_hundred : 11^2023 % 100 = 31 := by
  sorry

end NUMINAMATH_CALUDE_power_eleven_mod_hundred_l2856_285677


namespace NUMINAMATH_CALUDE_exponential_strictly_increasing_l2856_285601

theorem exponential_strictly_increasing (a b : ℝ) : a < b → (2 : ℝ) ^ a < (2 : ℝ) ^ b := by
  sorry

end NUMINAMATH_CALUDE_exponential_strictly_increasing_l2856_285601


namespace NUMINAMATH_CALUDE_power_division_result_l2856_285635

theorem power_division_result : (3 : ℕ)^12 / 27^2 = 729 := by
  sorry

end NUMINAMATH_CALUDE_power_division_result_l2856_285635


namespace NUMINAMATH_CALUDE_max_profit_at_800_l2856_285674

/-- Price function for desk orders -/
def P (x : ℕ) : ℚ :=
  if x ≤ 100 then 80
  else 82 - 0.02 * x

/-- Profit function for desk orders -/
def f (x : ℕ) : ℚ :=
  if x ≤ 100 then 30 * x
  else (32 * x - 0.02 * x^2)

/-- Theorem stating the maximum profit and corresponding order quantity -/
theorem max_profit_at_800 :
  (∀ x : ℕ, 0 < x ∧ x ≤ 1000 → f x ≤ f 800) ∧
  f 800 = 12800 :=
sorry

end NUMINAMATH_CALUDE_max_profit_at_800_l2856_285674


namespace NUMINAMATH_CALUDE_polynomial_value_relation_l2856_285671

theorem polynomial_value_relation (y : ℝ) : 
  4 * y^2 - 2 * y + 5 = 7 → 2 * y^2 - y + 1 = 2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_value_relation_l2856_285671


namespace NUMINAMATH_CALUDE_other_endpoint_of_diameter_l2856_285627

/-- A circle in a 2D coordinate plane --/
structure Circle where
  center : ℝ × ℝ

/-- A diameter of a circle --/
structure Diameter where
  circle : Circle
  endpoint1 : ℝ × ℝ
  endpoint2 : ℝ × ℝ

/-- The given circle P --/
def circleP : Circle :=
  { center := (3, 4) }

/-- The diameter of circle P --/
def diameterP : Diameter :=
  { circle := circleP
    endpoint1 := (0, 0)
    endpoint2 := (-3, -4) }

/-- Theorem: The other endpoint of the diameter is at (-3, -4) --/
theorem other_endpoint_of_diameter :
  diameterP.endpoint2 = (-3, -4) := by
  sorry

#check other_endpoint_of_diameter

end NUMINAMATH_CALUDE_other_endpoint_of_diameter_l2856_285627


namespace NUMINAMATH_CALUDE_rectangular_field_perimeter_l2856_285636

def rectangle_perimeter (length width : ℝ) : ℝ :=
  2 * (length + width)

theorem rectangular_field_perimeter :
  let length : ℝ := 15
  let width : ℝ := 20
  rectangle_perimeter length width = 70 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_field_perimeter_l2856_285636


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_implies_b_equals_one_l2856_285650

/-- 
Given a hyperbola with equation x²/4 - y²/b² = 1 where b > 0,
if the asymptotes are y = ±(1/2)x, then b = 1.
-/
theorem hyperbola_asymptote_implies_b_equals_one (b : ℝ) : 
  b > 0 → 
  (∀ x y : ℝ, x^2 / 4 - y^2 / b^2 = 1 → 
    (y = (1/2) * x ∨ y = -(1/2) * x)) → 
  b = 1 := by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_implies_b_equals_one_l2856_285650


namespace NUMINAMATH_CALUDE_diamond_six_three_l2856_285633

-- Define the diamond operation
noncomputable def diamond (x y : ℝ) : ℝ := 
  sorry

-- Axioms for the diamond operation
axiom diamond_zero (x : ℝ) : diamond x 0 = 2 * x
axiom diamond_comm (x y : ℝ) : diamond x y = diamond y x
axiom diamond_succ (x y : ℝ) : diamond (x + 1) y = diamond x y * (y + 2)

-- Theorem to prove
theorem diamond_six_three : diamond 6 3 = 93750 := by
  sorry

end NUMINAMATH_CALUDE_diamond_six_three_l2856_285633


namespace NUMINAMATH_CALUDE_exists_multiple_irreducible_representations_l2856_285662

/-- The set V_n for a given n > 2 -/
def V_n (n : ℕ) : Set ℕ :=
  {m : ℕ | ∃ k : ℕ, m = 1 + k * n}

/-- A number is irreducible in V_n if it cannot be expressed as a product of two numbers in V_n -/
def irreducible_in_V_n (n : ℕ) (m : ℕ) : Prop :=
  m ∈ V_n n ∧ ∀ p q : ℕ, p ∈ V_n n → q ∈ V_n n → p * q ≠ m

/-- The main theorem -/
theorem exists_multiple_irreducible_representations (n : ℕ) (h : n > 2) :
  ∃ r : ℕ, r ∈ V_n n ∧
    ∃ (irreducibles1 irreducibles2 : List ℕ),
      irreducibles1 ≠ irreducibles2 ∧
      (∀ x ∈ irreducibles1, irreducible_in_V_n n x) ∧
      (∀ x ∈ irreducibles2, irreducible_in_V_n n x) ∧
      (irreducibles1.prod = r) ∧
      (irreducibles2.prod = r) :=
sorry

end NUMINAMATH_CALUDE_exists_multiple_irreducible_representations_l2856_285662


namespace NUMINAMATH_CALUDE_parabola_tangent_intersection_l2856_285687

noncomputable def parabola (x : ℝ) : ℝ := x^2

def point_A : ℝ × ℝ := (1, 1)

noncomputable def point_B (x2 : ℝ) : ℝ × ℝ := (x2, x2^2)

noncomputable def tangent_slope (x : ℝ) : ℝ := 2 * x

noncomputable def tangent_line_A (x : ℝ) : ℝ := 2 * (x - 1) + 1

noncomputable def tangent_line_B (x2 x : ℝ) : ℝ := 2 * x2 * (x - x2) + x2^2

noncomputable def intersection_point (x2 : ℝ) : ℝ × ℝ :=
  let x_c := (x2^2 - 1) / (2 - 2*x2)
  let y_c := 2 * x_c - 1
  (x_c, y_c)

noncomputable def vector_AC (x2 : ℝ) : ℝ × ℝ :=
  let C := intersection_point x2
  (C.1 - point_A.1, C.2 - point_A.2)

noncomputable def vector_BC (x2 : ℝ) : ℝ × ℝ :=
  let C := intersection_point x2
  let B := point_B x2
  (C.1 - B.1, C.2 - B.2)

noncomputable def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

theorem parabola_tangent_intersection (x2 : ℝ) :
  dot_product (vector_AC x2) (vector_BC x2) = 0 → x2 = -1/4 :=
by sorry

end NUMINAMATH_CALUDE_parabola_tangent_intersection_l2856_285687


namespace NUMINAMATH_CALUDE_room_breadth_calculation_l2856_285617

/-- Given a room with specified dimensions and carpeting costs, calculate its breadth. -/
theorem room_breadth_calculation (room_length : ℝ) (carpet_width : ℝ) (carpet_cost_per_meter : ℝ) (total_cost : ℝ) :
  room_length = 15 →
  carpet_width = 0.75 →
  carpet_cost_per_meter = 0.3 →
  total_cost = 36 →
  (total_cost / carpet_cost_per_meter) * carpet_width / room_length = 6 := by
  sorry

end NUMINAMATH_CALUDE_room_breadth_calculation_l2856_285617


namespace NUMINAMATH_CALUDE_angle_sum_proof_l2856_285672

theorem angle_sum_proof (x y : Real) (h1 : 0 < x ∧ x < π/2) (h2 : 0 < y ∧ y < π/2)
  (h3 : 4 * (Real.cos x)^2 + 3 * (Real.cos y)^2 = 1)
  (h4 : 4 * Real.sin (2*x) + 3 * Real.sin (2*y) = 0) :
  x + 2*y = π/6*5 := by
sorry

end NUMINAMATH_CALUDE_angle_sum_proof_l2856_285672


namespace NUMINAMATH_CALUDE_salt_percentage_in_water_l2856_285652

def salt_mass : ℝ := 10
def water_mass : ℝ := 40

theorem salt_percentage_in_water :
  (salt_mass / water_mass) * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_salt_percentage_in_water_l2856_285652


namespace NUMINAMATH_CALUDE_segments_form_triangle_l2856_285649

/-- Triangle inequality theorem: the sum of the lengths of any two sides of a triangle 
    must be greater than the length of the remaining side -/
def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- A set of line segments can form a triangle if they satisfy the triangle inequality -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ triangle_inequality a b c

/-- The set of line segments 5cm, 8cm, and 12cm can form a triangle -/
theorem segments_form_triangle : can_form_triangle 5 8 12 := by
  sorry

end NUMINAMATH_CALUDE_segments_form_triangle_l2856_285649


namespace NUMINAMATH_CALUDE_weekly_rainfall_sum_l2856_285680

def monday_rainfall : ℝ := 0.12962962962962962
def tuesday_rainfall : ℝ := 0.35185185185185186
def wednesday_rainfall : ℝ := 0.09259259259259259
def thursday_rainfall : ℝ := 0.25925925925925924
def friday_rainfall : ℝ := 0.48148148148148145
def saturday_rainfall : ℝ := 0.2222222222222222
def sunday_rainfall : ℝ := 0.4444444444444444

theorem weekly_rainfall_sum :
  monday_rainfall + tuesday_rainfall + wednesday_rainfall + thursday_rainfall +
  friday_rainfall + saturday_rainfall + sunday_rainfall = 1.9814814814814815 := by
  sorry

end NUMINAMATH_CALUDE_weekly_rainfall_sum_l2856_285680


namespace NUMINAMATH_CALUDE_inequality_proof_l2856_285666

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (a + 1 / (2 * b^2)) * (b + 1 / (2 * a^2)) ≥ 25 / 4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2856_285666


namespace NUMINAMATH_CALUDE_charlie_banana_consumption_l2856_285614

/-- Represents the daily banana consumption of Charlie the chimp over 7 days -/
def BananaSequence : Type := Fin 7 → ℚ

/-- The sum of bananas eaten over 7 days is 150 -/
def SumIs150 (seq : BananaSequence) : Prop :=
  (Finset.sum Finset.univ seq) = 150

/-- Each day's consumption is 4 more than the previous day -/
def ArithmeticProgression (seq : BananaSequence) : Prop :=
  ∀ i : Fin 6, seq (i.succ) = seq i + 4

/-- The theorem to be proved -/
theorem charlie_banana_consumption
  (seq : BananaSequence)
  (sum_cond : SumIs150 seq)
  (prog_cond : ArithmeticProgression seq) :
  seq 6 = 33 + 4/7 := by sorry

end NUMINAMATH_CALUDE_charlie_banana_consumption_l2856_285614


namespace NUMINAMATH_CALUDE_sequence_terms_l2856_285657

def a (n : ℕ) : ℤ := (-1)^(n+1) * (3*n - 2)

theorem sequence_terms : 
  (a 1 = 1) ∧ (a 2 = -4) ∧ (a 3 = 7) ∧ (a 4 = -10) ∧ (a 5 = 13) := by
  sorry

end NUMINAMATH_CALUDE_sequence_terms_l2856_285657


namespace NUMINAMATH_CALUDE_remainder_problem_l2856_285619

theorem remainder_problem (divisor remainder_1657 : ℕ) 
  (h1 : divisor = 127)
  (h2 : remainder_1657 = 6)
  (h3 : ∃ k : ℕ, 1657 = k * divisor + remainder_1657)
  (h4 : ∃ m r : ℕ, 2037 = m * divisor + r ∧ r < divisor)
  (h5 : ∀ d : ℕ, d > divisor → ¬(∃ k1 k2 r1 r2 : ℕ, 1657 = k1 * d + r1 ∧ 2037 = k2 * d + r2 ∧ r1 < d ∧ r2 < d)) :
  ∃ m : ℕ, 2037 = m * divisor + 5 :=
sorry

end NUMINAMATH_CALUDE_remainder_problem_l2856_285619


namespace NUMINAMATH_CALUDE_expected_interval_is_three_l2856_285675

/-- Represents the train system with given conditions --/
structure TrainSystem where
  northern_route_time : ℝ
  southern_route_time : ℝ
  arrival_time_difference : ℝ
  commute_time_difference : ℝ

/-- The expected interval between trains in one direction --/
def expected_interval (ts : TrainSystem) : ℝ := 3

/-- Theorem stating that the expected interval is 3 minutes given the conditions --/
theorem expected_interval_is_three (ts : TrainSystem) 
  (h1 : ts.northern_route_time = 17)
  (h2 : ts.southern_route_time = 11)
  (h3 : ts.arrival_time_difference = 1.25)
  (h4 : ts.commute_time_difference = 1) :
  expected_interval ts = 3 := by
  sorry

#check expected_interval_is_three

end NUMINAMATH_CALUDE_expected_interval_is_three_l2856_285675


namespace NUMINAMATH_CALUDE_min_selling_price_A_l2856_285665

/-- Represents the number of units of model A purchased -/
def units_A : ℕ := 100

/-- Represents the number of units of model B purchased -/
def units_B : ℕ := 160 - units_A

/-- Represents the cost price of model A in yuan -/
def cost_A : ℕ := 150

/-- Represents the cost price of model B in yuan -/
def cost_B : ℕ := 350

/-- Represents the total cost of purchasing both models in yuan -/
def total_cost : ℕ := 36000

/-- Represents the minimum required gross profit in yuan -/
def min_gross_profit : ℕ := 11000

/-- Theorem stating that the minimum selling price of model A is 200 yuan -/
theorem min_selling_price_A : 
  ∃ (selling_price_A : ℕ), 
    selling_price_A = 200 ∧ 
    units_A * cost_A + units_B * cost_B = total_cost ∧
    units_A * (selling_price_A - cost_A) + units_B * (2 * (selling_price_A - cost_A)) ≥ min_gross_profit ∧
    ∀ (price : ℕ), price < selling_price_A → 
      units_A * (price - cost_A) + units_B * (2 * (price - cost_A)) < min_gross_profit :=
by
  sorry


end NUMINAMATH_CALUDE_min_selling_price_A_l2856_285665


namespace NUMINAMATH_CALUDE_twenty_percent_less_than_sixty_l2856_285679

theorem twenty_percent_less_than_sixty (x : ℝ) : x + (1/3) * x = 48 → x = 36 := by
  sorry

end NUMINAMATH_CALUDE_twenty_percent_less_than_sixty_l2856_285679


namespace NUMINAMATH_CALUDE_salary_reduction_percentage_l2856_285610

theorem salary_reduction_percentage (S : ℝ) (R : ℝ) (h : S > 0) :
  (S - R / 100 * S) * (1 + 25 / 100) = S → R = 20 := by
  sorry

end NUMINAMATH_CALUDE_salary_reduction_percentage_l2856_285610


namespace NUMINAMATH_CALUDE_child_ticket_price_is_correct_l2856_285639

/-- Calculates the price of a child's ticket given the group composition, adult ticket price, senior discount, and total bill. -/
def childTicketPrice (totalPeople adultCount seniorCount childCount : ℕ) 
                     (adultPrice : ℚ) (seniorDiscount : ℚ) (totalBill : ℚ) : ℚ :=
  let seniorPrice := adultPrice * (1 - seniorDiscount)
  let adultTotal := adultPrice * adultCount
  let seniorTotal := seniorPrice * seniorCount
  let childTotal := totalBill - adultTotal - seniorTotal
  childTotal / childCount

/-- Theorem stating that the child ticket price is $5.63 given the problem conditions. -/
theorem child_ticket_price_is_correct :
  childTicketPrice 50 25 15 10 15 0.25 600 = 5.63 := by
  sorry

end NUMINAMATH_CALUDE_child_ticket_price_is_correct_l2856_285639


namespace NUMINAMATH_CALUDE_alcohol_quantity_in_mixture_l2856_285682

/-- Given a mixture of alcohol and water, this theorem proves that if the initial ratio
    of alcohol to water is 4:3, and adding 4 liters of water changes the ratio to 4:5,
    then the initial quantity of alcohol in the mixture is 8 liters. -/
theorem alcohol_quantity_in_mixture
  (initial_alcohol : ℝ) (initial_water : ℝ)
  (h1 : initial_alcohol / initial_water = 4 / 3)
  (h2 : initial_alcohol / (initial_water + 4) = 4 / 5) :
  initial_alcohol = 8 := by
  sorry

end NUMINAMATH_CALUDE_alcohol_quantity_in_mixture_l2856_285682


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2856_285645

def A : Set ℝ := {0, 1, 2}
def B : Set ℝ := {x | 1 < x ∧ x < 4}

theorem intersection_of_A_and_B : A ∩ B = {2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2856_285645


namespace NUMINAMATH_CALUDE_bottle_caps_found_l2856_285663

theorem bottle_caps_found (earlier_total current_total : ℕ) 
  (h1 : earlier_total = 25) 
  (h2 : current_total = 32) : 
  current_total - earlier_total = 7 := by
  sorry

end NUMINAMATH_CALUDE_bottle_caps_found_l2856_285663


namespace NUMINAMATH_CALUDE_hacky_sack_jumping_rope_problem_l2856_285643

theorem hacky_sack_jumping_rope_problem : 
  ∀ (hacky_sack_players jump_rope_players : ℕ),
    hacky_sack_players = 6 →
    jump_rope_players = 6 * hacky_sack_players →
    jump_rope_players ≠ 12 :=
by
  sorry

end NUMINAMATH_CALUDE_hacky_sack_jumping_rope_problem_l2856_285643


namespace NUMINAMATH_CALUDE_fred_change_theorem_l2856_285692

/-- Calculates the change received after a purchase -/
def calculate_change (ticket_price : ℚ) (num_tickets : ℕ) (borrowed_movie_price : ℚ) (paid_amount : ℚ) : ℚ :=
  paid_amount - (ticket_price * num_tickets + borrowed_movie_price)

theorem fred_change_theorem :
  let ticket_price : ℚ := 8.25
  let num_tickets : ℕ := 4
  let borrowed_movie_price : ℚ := 9.50
  let paid_amount : ℚ := 50
  calculate_change ticket_price num_tickets borrowed_movie_price paid_amount = 7.50 := by
  sorry

#eval calculate_change 8.25 4 9.50 50

end NUMINAMATH_CALUDE_fred_change_theorem_l2856_285692


namespace NUMINAMATH_CALUDE_range_of_x_l2856_285600

theorem range_of_x (x : ℝ) : (x + 1)^0 = 1 → x ≠ -1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_x_l2856_285600


namespace NUMINAMATH_CALUDE_jasmine_weight_l2856_285648

/-- The weight of a bag of chips in ounces -/
def bag_weight : ℕ := 20

/-- The weight of a tin of cookies in ounces -/
def tin_weight : ℕ := 9

/-- The number of bags of chips Jasmine buys -/
def num_bags : ℕ := 6

/-- The number of tins of cookies Jasmine buys -/
def num_tins : ℕ := 4 * num_bags

/-- The number of ounces in a pound -/
def ounces_per_pound : ℕ := 16

/-- The total weight Jasmine has to carry in pounds -/
def total_weight : ℕ := (bag_weight * num_bags + tin_weight * num_tins) / ounces_per_pound

theorem jasmine_weight : total_weight = 21 := by
  sorry

end NUMINAMATH_CALUDE_jasmine_weight_l2856_285648


namespace NUMINAMATH_CALUDE_hammer_weight_exceeds_ton_on_10th_day_l2856_285651

def hammer_weight (day : ℕ) : ℝ :=
  7 * (2 ^ (day - 1))

theorem hammer_weight_exceeds_ton_on_10th_day :
  (∀ d : ℕ, d < 10 → hammer_weight d ≤ 2000) ∧
  hammer_weight 10 > 2000 :=
by sorry

end NUMINAMATH_CALUDE_hammer_weight_exceeds_ton_on_10th_day_l2856_285651


namespace NUMINAMATH_CALUDE_cone_surface_area_l2856_285673

/-- The surface area of a cone, given its lateral surface properties -/
theorem cone_surface_area (r l θ : Real) (h1 : l = 1) (h2 : θ = π / 2) :
  let lateral_area := π * r * l
  let base_area := π * r^2
  lateral_area = l^2 * θ / 2 →
  lateral_area + base_area = 5 * π / 16 := by
  sorry

end NUMINAMATH_CALUDE_cone_surface_area_l2856_285673


namespace NUMINAMATH_CALUDE_algebraic_simplification_l2856_285630

theorem algebraic_simplification (a b : ℝ) : 
  14 * a^8 * b^4 / (7 * a^4 * b^4) - a^3 * a - (2 * a^2)^2 = -3 * a^4 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_simplification_l2856_285630


namespace NUMINAMATH_CALUDE_expression_evaluation_l2856_285624

theorem expression_evaluation : 2 + 3 * 4 - 5 * 6 + 7 = -9 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2856_285624


namespace NUMINAMATH_CALUDE_equation_solutions_l2856_285629

noncomputable def fourthRoot (x : ℝ) : ℝ := Real.rpow x (1/4)

theorem equation_solutions :
  let f : ℝ → ℝ := λ x => fourthRoot (53 - 3*x) + fourthRoot (29 + x)
  ∀ x : ℝ, f x = 4 ↔ x = 2 ∨ x = 16 :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l2856_285629


namespace NUMINAMATH_CALUDE_quadratic_solution_l2856_285656

theorem quadratic_solution (x : ℝ) : x^2 - 6*x + 8 = 0 → x = 2 ∨ x = 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_l2856_285656


namespace NUMINAMATH_CALUDE_leg_head_difference_l2856_285620

/-- Represents a group of ducks and cows -/
structure AnimalGroup where
  ducks : ℕ
  cows : ℕ

/-- Calculates the total number of legs in the group -/
def totalLegs (g : AnimalGroup) : ℕ := 2 * g.ducks + 4 * g.cows

/-- Calculates the total number of heads in the group -/
def totalHeads (g : AnimalGroup) : ℕ := g.ducks + g.cows

/-- The main theorem -/
theorem leg_head_difference (g : AnimalGroup) 
  (h1 : g.cows = 20)
  (h2 : ∃ k : ℕ, totalLegs g = 2 * totalHeads g + k) :
  ∃ k : ℕ, k = 40 ∧ totalLegs g = 2 * totalHeads g + k := by
  sorry


end NUMINAMATH_CALUDE_leg_head_difference_l2856_285620


namespace NUMINAMATH_CALUDE_power_function_is_odd_l2856_285607

/-- A function f is a power function if it has the form f(x) = ax^n, where a ≠ 0 and n is a real number. -/
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ (a n : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x ^ n

/-- A function f is odd if f(-x) = -f(x) for all x in the domain of f. -/
def isOddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- Given that f(x) = (m - 1)x^(m^2 - 4m + 3) is a power function, prove that f is an odd function. -/
theorem power_function_is_odd (m : ℝ) :
  let f := fun x => (m - 1) * x ^ (m^2 - 4*m + 3)
  isPowerFunction f → isOddFunction f := by
  sorry


end NUMINAMATH_CALUDE_power_function_is_odd_l2856_285607


namespace NUMINAMATH_CALUDE_pony_jeans_discount_rate_l2856_285664

theorem pony_jeans_discount_rate 
  (fox_price : ℝ) 
  (pony_price : ℝ) 
  (total_savings : ℝ) 
  (fox_quantity : ℕ) 
  (pony_quantity : ℕ) 
  (total_discount_rate : ℝ) :
  fox_price = 15 →
  pony_price = 18 →
  total_savings = 8.55 →
  fox_quantity = 3 →
  pony_quantity = 2 →
  total_discount_rate = 22 →
  ∃ (fox_discount_rate : ℝ) (pony_discount_rate : ℝ),
    fox_discount_rate + pony_discount_rate = total_discount_rate ∧
    fox_quantity * (fox_price * fox_discount_rate / 100) + 
    pony_quantity * (pony_price * pony_discount_rate / 100) = total_savings ∧
    pony_discount_rate = 15 :=
by sorry

end NUMINAMATH_CALUDE_pony_jeans_discount_rate_l2856_285664


namespace NUMINAMATH_CALUDE_projection_problem_l2856_285689

/-- Given a projection that takes (3, 6) to (9/5, 18/5), prove that it takes (1, -1) to (-1/5, -2/5) -/
theorem projection_problem (proj : ℝ × ℝ → ℝ × ℝ) 
  (h : proj (3, 6) = (9/5, 18/5)) : 
  proj (1, -1) = (-1/5, -2/5) := by
  sorry

end NUMINAMATH_CALUDE_projection_problem_l2856_285689


namespace NUMINAMATH_CALUDE_quadratic_real_solutions_range_l2856_285653

-- Define the quadratic equation
def quadratic_equation (m : ℝ) (x : ℝ) : ℝ := (m - 3) * x^2 + 4 * x + 1

-- Define the condition for real solutions
def has_real_solutions (m : ℝ) : Prop :=
  ∃ x : ℝ, quadratic_equation m x = 0

-- Theorem statement
theorem quadratic_real_solutions_range :
  ∀ m : ℝ, has_real_solutions m ↔ m ≤ 7 ∧ m ≠ 3 :=
sorry

end NUMINAMATH_CALUDE_quadratic_real_solutions_range_l2856_285653


namespace NUMINAMATH_CALUDE_max_value_of_expression_l2856_285699

theorem max_value_of_expression (m n t : ℝ) (hm : m > 0) (hn : n > 0) (ht : t > 0)
  (heq : m^2 - 3*m*n + 4*n^2 - t = 0) :
  ∃ (m₀ n₀ t₀ : ℝ), m₀ > 0 ∧ n₀ > 0 ∧ t₀ > 0 ∧
    m₀^2 - 3*m₀*n₀ + 4*n₀^2 - t₀ = 0 ∧
    (∀ m' n' t' : ℝ, m' > 0 → n' > 0 → t' > 0 → m'^2 - 3*m'*n' + 4*n'^2 - t' = 0 →
      t₀/(m₀*n₀) ≤ t'/(m'*n')) ∧
    (∀ m' n' t' : ℝ, m' > 0 → n' > 0 → t' > 0 → m'^2 - 3*m'*n' + 4*n'^2 - t' = 0 →
      m' + 2*n' - t' ≤ 2) ∧
    m₀ + 2*n₀ - t₀ = 2 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l2856_285699


namespace NUMINAMATH_CALUDE_expression_factorization_l2856_285604

theorem expression_factorization (x : ℝ) : 
  (16 * x^7 + 36 * x^4 - 9) - (4 * x^7 - 6 * x^4 - 9) = 6 * x^4 * (2 * x^3 + 7) := by
  sorry

end NUMINAMATH_CALUDE_expression_factorization_l2856_285604


namespace NUMINAMATH_CALUDE_min_value_ab_min_value_is_two_l2856_285647

theorem min_value_ab (a b : ℝ) (h : (a⁻¹ + b⁻¹ : ℝ) = Real.sqrt (a * b)) :
  ∀ x y : ℝ, x > 0 → y > 0 → (x⁻¹ + y⁻¹ : ℝ) = Real.sqrt (x * y) → a * b ≤ x * y :=
by sorry

theorem min_value_is_two (a b : ℝ) (h : (a⁻¹ + b⁻¹ : ℝ) = Real.sqrt (a * b)) :
  a * b = 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_ab_min_value_is_two_l2856_285647


namespace NUMINAMATH_CALUDE_supplement_of_complement_of_36_l2856_285612

-- Define the original angle
def original_angle : ℝ := 36

-- Define the complement of an angle
def complement (angle : ℝ) : ℝ := 90 - angle

-- Define the supplement of an angle
def supplement (angle : ℝ) : ℝ := 180 - angle

-- Theorem statement
theorem supplement_of_complement_of_36 : 
  supplement (complement original_angle) = 126 := by
  sorry

end NUMINAMATH_CALUDE_supplement_of_complement_of_36_l2856_285612


namespace NUMINAMATH_CALUDE_range_of_a_l2856_285634

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, |x + a| + |x - 1| + a < 2011) ↔ a < 1005 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l2856_285634


namespace NUMINAMATH_CALUDE_percent_greater_l2856_285659

theorem percent_greater (w x y z : ℝ) (hw : w > 0) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (hxy : x = 1.2 * y) (hyz : y = 1.2 * z) (hwx : w = 0.8 * x) :
  w = 1.152 * z := by
  sorry

end NUMINAMATH_CALUDE_percent_greater_l2856_285659


namespace NUMINAMATH_CALUDE_total_octopus_legs_l2856_285696

/-- The number of octopuses Sawyer saw -/
def num_octopuses : ℕ := 5

/-- The number of legs each octopus has -/
def legs_per_octopus : ℕ := 8

/-- The total number of octopus legs -/
def total_legs : ℕ := num_octopuses * legs_per_octopus

theorem total_octopus_legs : total_legs = 40 := by
  sorry

end NUMINAMATH_CALUDE_total_octopus_legs_l2856_285696


namespace NUMINAMATH_CALUDE_rectangular_box_surface_area_l2856_285626

/-- The surface area of a rectangular box -/
def surface_area (l w h : ℝ) : ℝ := 2 * (l * h + l * w + w * h)

/-- Theorem: The surface area of a rectangular box with length l, width w, and height h
    is equal to 2(lh + lw + wh) -/
theorem rectangular_box_surface_area (l w h : ℝ) :
  surface_area l w h = 2 * (l * h + l * w + w * h) := by
  sorry

end NUMINAMATH_CALUDE_rectangular_box_surface_area_l2856_285626


namespace NUMINAMATH_CALUDE_cube_root_and_square_root_problem_l2856_285641

theorem cube_root_and_square_root_problem :
  ∀ (a b : ℝ),
  (5 * a + 2) ^ (1/3 : ℝ) = 3 →
  (3 * a + b - 1) ^ (1/2 : ℝ) = 4 →
  a = 5 ∧ b = 2 ∧ (3 * a - b + 3) ^ (1/2 : ℝ) = 4 :=
by sorry

end NUMINAMATH_CALUDE_cube_root_and_square_root_problem_l2856_285641


namespace NUMINAMATH_CALUDE_profit_percentage_calculation_l2856_285623

theorem profit_percentage_calculation (selling_price cost_price : ℝ) : 
  selling_price = 600 → 
  cost_price = 480 → 
  (selling_price - cost_price) / cost_price * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_profit_percentage_calculation_l2856_285623


namespace NUMINAMATH_CALUDE_lou_fine_shoes_pricing_l2856_285603

/-- Calculates the price of shoes after Lou's Fine Shoes pricing strategy --/
theorem lou_fine_shoes_pricing (initial_price : ℝ) : 
  initial_price = 50 →
  (initial_price * (1 + 0.2)) * (1 - 0.2) = 48 := by
sorry

end NUMINAMATH_CALUDE_lou_fine_shoes_pricing_l2856_285603


namespace NUMINAMATH_CALUDE_greatest_four_digit_number_with_conditions_l2856_285695

/-- The greatest four-digit number that is two more than a multiple of 8 and four more than a multiple of 7 -/
def greatest_number : ℕ := 9990

/-- A number is four-digit if it's between 1000 and 9999 inclusive -/
def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

/-- A number is two more than a multiple of 8 -/
def is_two_more_than_multiple_of_eight (n : ℕ) : Prop := ∃ k : ℕ, n = 8 * k + 2

/-- A number is four more than a multiple of 7 -/
def is_four_more_than_multiple_of_seven (n : ℕ) : Prop := ∃ k : ℕ, n = 7 * k + 4

theorem greatest_four_digit_number_with_conditions :
  is_four_digit greatest_number ∧
  is_two_more_than_multiple_of_eight greatest_number ∧
  is_four_more_than_multiple_of_seven greatest_number ∧
  ∀ n : ℕ, is_four_digit n →
    is_two_more_than_multiple_of_eight n →
    is_four_more_than_multiple_of_seven n →
    n ≤ greatest_number :=
by sorry

end NUMINAMATH_CALUDE_greatest_four_digit_number_with_conditions_l2856_285695


namespace NUMINAMATH_CALUDE_mel_katherine_age_difference_l2856_285697

/-- Given that Mel is younger than Katherine, and when Katherine is 24, Mel is 21,
    prove that Mel is 3 years younger than Katherine. -/
theorem mel_katherine_age_difference :
  ∀ (katherine_age mel_age : ℕ),
  katherine_age > mel_age →
  (katherine_age = 24 → mel_age = 21) →
  katherine_age - mel_age = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_mel_katherine_age_difference_l2856_285697


namespace NUMINAMATH_CALUDE_empty_proper_subset_implies_nonempty_l2856_285668

theorem empty_proper_subset_implies_nonempty (A : Set α) :
  ∅ ⊂ A → A ≠ ∅ := by
  sorry

end NUMINAMATH_CALUDE_empty_proper_subset_implies_nonempty_l2856_285668


namespace NUMINAMATH_CALUDE_candle_length_correct_l2856_285631

/-- Represents the remaining length of a burning candle after t hours. -/
def candle_length (t : ℝ) : ℝ := 20 - 5 * t

theorem candle_length_correct (t : ℝ) (h : 0 ≤ t ∧ t ≤ 4) : 
  candle_length t = 20 - 5 * t ∧ candle_length t ≥ 0 := by
  sorry

#check candle_length_correct

end NUMINAMATH_CALUDE_candle_length_correct_l2856_285631


namespace NUMINAMATH_CALUDE_intersection_when_m_is_3_range_of_m_when_union_equals_B_l2856_285686

-- Define the sets A and B
def A : Set ℝ := {x | x < 0 ∨ x > 3}
def B (m : ℝ) : Set ℝ := {x | x < m - 1 ∨ x > 2 * m}

-- Part 1: Prove that when m = 3, A ∩ B = {x | x < 0 ∨ x > 6}
theorem intersection_when_m_is_3 : A ∩ B 3 = {x | x < 0 ∨ x > 6} := by sorry

-- Part 2: Prove that when B ∪ A = B, the range of m is [1, 3/2]
theorem range_of_m_when_union_equals_B :
  (∀ m : ℝ, B m ∪ A = B m) ↔ (∀ m : ℝ, 1 ≤ m ∧ m ≤ 3/2) := by sorry

end NUMINAMATH_CALUDE_intersection_when_m_is_3_range_of_m_when_union_equals_B_l2856_285686


namespace NUMINAMATH_CALUDE_sum_product_inequality_l2856_285691

theorem sum_product_inequality (a b c : ℝ) (h : a + b + c = 0) : a * b + a * c + b * c ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_product_inequality_l2856_285691


namespace NUMINAMATH_CALUDE_symmetrical_point_l2856_285685

/-- Given a point (m, m+1) and a line of symmetry x=3, 
    the symmetrical point is (6-m, m+1) --/
theorem symmetrical_point (m : ℝ) : 
  let original_point := (m, m+1)
  let line_of_symmetry := 3
  let symmetrical_point := (6-m, m+1)
  symmetrical_point = 
    (2 * line_of_symmetry - original_point.1, original_point.2) := by
  sorry

end NUMINAMATH_CALUDE_symmetrical_point_l2856_285685
