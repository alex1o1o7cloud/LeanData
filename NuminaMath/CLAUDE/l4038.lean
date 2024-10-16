import Mathlib

namespace NUMINAMATH_CALUDE_original_triangle_area_l4038_403828

/-- 
Given a triangle whose dimensions are doubled to form a new triangle,
if the area of the new triangle is 32 square feet,
then the area of the original triangle is 8 square feet.
-/
theorem original_triangle_area (original : Real) (new : Real) : 
  (new = 32) → (new = 4 * original) → (original = 8) :=
by
  sorry

end NUMINAMATH_CALUDE_original_triangle_area_l4038_403828


namespace NUMINAMATH_CALUDE_relationship_between_x_and_y_l4038_403875

-- Define variables x and y
variable (x y : ℝ)

-- Define the conditions
def condition1 (x y : ℝ) : Prop := 2 * x - 3 * y < x - 1
def condition2 (x y : ℝ) : Prop := 3 * x + 4 * y > 2 * y + 5

-- State the theorem
theorem relationship_between_x_and_y 
  (h1 : condition1 x y) (h2 : condition2 x y) : 
  x < 3 * y - 1 ∧ y > (5 - 3 * x) / 2 := by
  sorry

end NUMINAMATH_CALUDE_relationship_between_x_and_y_l4038_403875


namespace NUMINAMATH_CALUDE_three_similar_points_l4038_403894

/-- Right trapezoid ABCD with given side lengths -/
structure RightTrapezoid where
  AB : ℝ
  AD : ℝ
  BC : ℝ
  ab_positive : AB > 0
  ad_positive : AD > 0
  bc_positive : BC > 0

/-- Point P on side AB of the trapezoid -/
def PointP (t : RightTrapezoid) := { x : ℝ // 0 ≤ x ∧ x ≤ t.AB }

/-- Condition for triangle PAD to be similar to triangle PBC -/
def IsSimilar (t : RightTrapezoid) (p : PointP t) : Prop :=
  p.val / (t.AB - p.val) = t.AD / t.BC ∨ p.val / t.BC = t.AD / (t.AB - p.val)

/-- The main theorem stating that there are exactly 3 points P satisfying the similarity condition -/
theorem three_similar_points (t : RightTrapezoid) 
  (h1 : t.AB = 7) (h2 : t.AD = 2) (h3 : t.BC = 3) : 
  ∃! (s : Finset (PointP t)), s.card = 3 ∧ ∀ p ∈ s, IsSimilar t p := by
  sorry

end NUMINAMATH_CALUDE_three_similar_points_l4038_403894


namespace NUMINAMATH_CALUDE_power_equality_l4038_403887

theorem power_equality (y : ℝ) (h : (10 : ℝ) ^ (4 * y) = 100) : (10 : ℝ) ^ (y / 2) = (10 : ℝ) ^ (1 / 4) := by
  sorry

end NUMINAMATH_CALUDE_power_equality_l4038_403887


namespace NUMINAMATH_CALUDE_diamond_with_zero_not_always_double_l4038_403842

def diamond (x y : ℝ) : ℝ := x + y - |x - y|

theorem diamond_with_zero_not_always_double :
  ¬ (∀ x : ℝ, diamond x 0 = 2 * x) := by
  sorry

end NUMINAMATH_CALUDE_diamond_with_zero_not_always_double_l4038_403842


namespace NUMINAMATH_CALUDE_son_age_proof_l4038_403816

theorem son_age_proof (son_age father_age : ℕ) : 
  father_age = son_age + 20 →
  father_age + 2 = 2 * (son_age + 2) →
  son_age = 18 := by
sorry

end NUMINAMATH_CALUDE_son_age_proof_l4038_403816


namespace NUMINAMATH_CALUDE_new_shoes_cost_approx_l4038_403854

/-- Cost of repairing used shoes in dollars -/
def repair_cost : ℝ := 13.50

/-- Duration of repaired shoes in years -/
def repaired_duration : ℝ := 1

/-- Duration of new shoes in years -/
def new_duration : ℝ := 2

/-- Percentage increase in average cost per year of new shoes compared to repaired shoes -/
def percentage_increase : ℝ := 0.1852

/-- Cost of purchasing new shoes -/
def new_shoes_cost : ℝ := 2 * (repair_cost + percentage_increase * repair_cost)

theorem new_shoes_cost_approx :
  ∃ ε > 0, |new_shoes_cost - 32| < ε :=
sorry

end NUMINAMATH_CALUDE_new_shoes_cost_approx_l4038_403854


namespace NUMINAMATH_CALUDE_greater_number_of_sum_and_difference_l4038_403831

theorem greater_number_of_sum_and_difference (x y : ℝ) 
  (sum_eq : x + y = 40) 
  (diff_eq : x - y = 10) : 
  max x y = 25 := by
sorry

end NUMINAMATH_CALUDE_greater_number_of_sum_and_difference_l4038_403831


namespace NUMINAMATH_CALUDE_line_circle_intersection_range_l4038_403896

/-- Given a line intersecting a circle, prove the range of the parameter a -/
theorem line_circle_intersection_range (a : ℝ) : 
  (∃ A B : ℝ × ℝ, A ≠ B ∧ 
   (A.1 + A.2 + a = 0) ∧ (B.1 + B.2 + a = 0) ∧
   (A.1^2 + A.2^2 = 1) ∧ (B.1^2 + B.2^2 = 1) ∧
   (‖(A.1, A.2)‖ + ‖(B.1, B.2)‖)^2 ≥ ((A.1 - B.1)^2 + (A.2 - B.2)^2)) →
  a ∈ Set.Icc (-Real.sqrt 2) (-1) ∪ Set.Icc 1 (Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_line_circle_intersection_range_l4038_403896


namespace NUMINAMATH_CALUDE_chicken_adventure_feathers_l4038_403801

/-- Calculates the number of feathers remaining after a chicken's thrill-seeking adventure. -/
def remaining_feathers (initial_feathers : ℕ) (cars_dodged : ℕ) : ℕ :=
  initial_feathers - 2 * cars_dodged

/-- Theorem stating the number of feathers remaining after the chicken's adventure. -/
theorem chicken_adventure_feathers :
  remaining_feathers 5263 23 = 5217 := by
  sorry

#eval remaining_feathers 5263 23

end NUMINAMATH_CALUDE_chicken_adventure_feathers_l4038_403801


namespace NUMINAMATH_CALUDE_arithmetic_expression_equality_l4038_403858

theorem arithmetic_expression_equality : 61 + 5 * 12 / (180 / 3) = 62 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_equality_l4038_403858


namespace NUMINAMATH_CALUDE_range_of_m_l4038_403877

def p (m : ℝ) : Prop := ∃ x y : ℝ, x < 0 ∧ y < 0 ∧ x ≠ y ∧ x^2 + m*x + 1 = 0 ∧ y^2 + m*y + 1 = 0

def q (m : ℝ) : Prop := ∀ x : ℝ, 4*x^2 + 4*(m-2)*x + 1 ≠ 0

theorem range_of_m (m : ℝ) :
  ((p m ∨ q m) ∧ ¬(p m ∧ q m)) → (1 < m ∧ m ≤ 2) ∨ (m ≥ 3) :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l4038_403877


namespace NUMINAMATH_CALUDE_gcd_840_1764_l4038_403826

theorem gcd_840_1764 : Nat.gcd 840 1764 = 84 := by
  sorry

end NUMINAMATH_CALUDE_gcd_840_1764_l4038_403826


namespace NUMINAMATH_CALUDE_picture_frame_perimeter_l4038_403844

theorem picture_frame_perimeter (width height : ℕ) (h1 : width = 6) (h2 : height = 9) :
  2 * width + 2 * height = 30 :=
by sorry

end NUMINAMATH_CALUDE_picture_frame_perimeter_l4038_403844


namespace NUMINAMATH_CALUDE_calculate_expression_l4038_403849

theorem calculate_expression : 5 * 401 + 4 * 401 + 3 * 401 + 400 = 5212 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l4038_403849


namespace NUMINAMATH_CALUDE_certain_number_l4038_403867

theorem certain_number : ∃ x : ℕ, x - 2 - 2 = 5 ∧ x = 9 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_l4038_403867


namespace NUMINAMATH_CALUDE_paul_spending_l4038_403807

/-- Calculates the weekly spending given total earnings and number of weeks --/
def weekly_spending (total_earnings : ℕ) (num_weeks : ℕ) : ℕ :=
  total_earnings / num_weeks

/-- Represents Paul's earnings and spending scenario --/
theorem paul_spending (lawn_money weed_money : ℕ) (weeks : ℕ) :
  lawn_money = 3 →
  weed_money = 3 →
  weeks = 2 →
  weekly_spending (lawn_money + weed_money) weeks = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_paul_spending_l4038_403807


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l4038_403861

theorem absolute_value_inequality (a b c : ℝ) (h : |a - c| < |b|) : |a| < |b| + |c| := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l4038_403861


namespace NUMINAMATH_CALUDE_three_number_average_l4038_403862

theorem three_number_average : 
  ∀ (x y z : ℝ),
  y = 2 * x →
  z = 4 * y →
  x = 45 →
  (x + y + z) / 3 = 165 := by
sorry

end NUMINAMATH_CALUDE_three_number_average_l4038_403862


namespace NUMINAMATH_CALUDE_person_speed_l4038_403827

/-- The speed of a person crossing a street -/
theorem person_speed (distance : ℝ) (time : ℝ) (h1 : distance = 600) (h2 : time = 5) :
  (distance / 1000) / (time / 60) = 7.2 := by
  sorry

end NUMINAMATH_CALUDE_person_speed_l4038_403827


namespace NUMINAMATH_CALUDE_combined_average_score_l4038_403836

/-- Given two math modeling clubs with their respective member counts and average scores,
    calculate the combined average score of both clubs. -/
theorem combined_average_score
  (club_a_members : ℕ)
  (club_b_members : ℕ)
  (club_a_average : ℝ)
  (club_b_average : ℝ)
  (h1 : club_a_members = 40)
  (h2 : club_b_members = 50)
  (h3 : club_a_average = 90)
  (h4 : club_b_average = 81) :
  (club_a_members * club_a_average + club_b_members * club_b_average) /
  (club_a_members + club_b_members : ℝ) = 85 := by
  sorry

end NUMINAMATH_CALUDE_combined_average_score_l4038_403836


namespace NUMINAMATH_CALUDE_sqrt_five_minus_two_power_2023_times_sqrt_five_plus_two_power_2023_equals_one_l4038_403891

theorem sqrt_five_minus_two_power_2023_times_sqrt_five_plus_two_power_2023_equals_one :
  (Real.sqrt 5 - 2) ^ 2023 * (Real.sqrt 5 + 2) ^ 2023 = 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_five_minus_two_power_2023_times_sqrt_five_plus_two_power_2023_equals_one_l4038_403891


namespace NUMINAMATH_CALUDE_abs_T_equals_1024_l4038_403898

-- Define the complex number i
def i : ℂ := Complex.I

-- Define T as in the problem
def T : ℂ := (1 + i)^18 - (1 - i)^18

-- Theorem statement
theorem abs_T_equals_1024 : Complex.abs T = 1024 := by
  sorry

end NUMINAMATH_CALUDE_abs_T_equals_1024_l4038_403898


namespace NUMINAMATH_CALUDE_arithmetic_equalities_l4038_403833

theorem arithmetic_equalities : 
  (Real.sqrt 27 + 3 * Real.sqrt (1/3) - Real.sqrt 24 * Real.sqrt 2 = 0) ∧
  ((Real.sqrt 5 - 2) * (2 + Real.sqrt 5) - (Real.sqrt 3 - 1)^2 = -3 + 2 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_equalities_l4038_403833


namespace NUMINAMATH_CALUDE_brothers_age_sum_l4038_403878

theorem brothers_age_sum : ∃ x : ℤ, 5 * x - 6 = 89 :=
  sorry

end NUMINAMATH_CALUDE_brothers_age_sum_l4038_403878


namespace NUMINAMATH_CALUDE_manufacturer_buyers_count_l4038_403874

theorem manufacturer_buyers_count :
  ∀ (N : ℕ) 
    (cake_buyers muffin_buyers both_buyers : ℕ)
    (prob_neither : ℚ),
  cake_buyers = 50 →
  muffin_buyers = 40 →
  both_buyers = 15 →
  prob_neither = 1/4 →
  (N : ℚ) * prob_neither = N - (cake_buyers + muffin_buyers - both_buyers) →
  N = 100 := by
sorry

end NUMINAMATH_CALUDE_manufacturer_buyers_count_l4038_403874


namespace NUMINAMATH_CALUDE_log_inequality_equivalence_l4038_403829

/-- A function that is even and monotonically increasing on [0,+∞) -/
def EvenMonoIncreasing (f : ℝ → ℝ) : Prop :=
  (∀ x, f x = f (-x)) ∧ 
  (∀ x y, 0 ≤ x → x ≤ y → f x ≤ f y)

theorem log_inequality_equivalence (f : ℝ → ℝ) (h : EvenMonoIncreasing f) :
  (∀ x : ℝ, f 1 < f (Real.log x) ↔ (x > 10 ∨ 0 < x ∧ x < (1/10))) :=
sorry

end NUMINAMATH_CALUDE_log_inequality_equivalence_l4038_403829


namespace NUMINAMATH_CALUDE_line_vector_to_slope_intercept_l4038_403855

/-- Given a line in vector form, proves that its slope-intercept form has specific m and b values -/
theorem line_vector_to_slope_intercept :
  let vector_form := fun (x y : ℝ) => -3 * (x - 5) + 2 * (y + 1) = 0
  let slope_intercept_form := fun (x y : ℝ) => y = (3/2) * x - 17/2
  (∀ x y, vector_form x y ↔ slope_intercept_form x y) := by
  sorry

end NUMINAMATH_CALUDE_line_vector_to_slope_intercept_l4038_403855


namespace NUMINAMATH_CALUDE_f_equals_g_l4038_403805

def N : Set ℕ := {n : ℕ | n > 0}

theorem f_equals_g
  (f g : ℕ → ℕ)
  (f_onto : ∀ y : ℕ, ∃ x : ℕ, f x = y)
  (g_one_one : ∀ x y : ℕ, g x = g y → x = y)
  (f_ge_g : ∀ n : ℕ, f n ≥ g n)
  : ∀ n : ℕ, f n = g n :=
sorry

end NUMINAMATH_CALUDE_f_equals_g_l4038_403805


namespace NUMINAMATH_CALUDE_constant_t_equality_l4038_403803

theorem constant_t_equality (x : ℝ) : 
  (5*x^2 - 6*x + 7) * (4*x^2 + (-6)*x + 10) = 20*x^4 - 54*x^3 + 114*x^2 - 102*x + 70 := by
  sorry


end NUMINAMATH_CALUDE_constant_t_equality_l4038_403803


namespace NUMINAMATH_CALUDE_largest_integer_m_l4038_403813

theorem largest_integer_m (m : ℤ) : (∀ k : ℤ, k > 6 → k^2 - 11*k + 28 ≥ 0) ∧ 6^2 - 11*6 + 28 < 0 := by
  sorry

end NUMINAMATH_CALUDE_largest_integer_m_l4038_403813


namespace NUMINAMATH_CALUDE_count_special_numbers_eq_266_l4038_403883

/-- The number of natural numbers from 1 to 1992 that are multiples of 3, but not multiples of 2 or 5 -/
def count_special_numbers : ℕ := 
  (Nat.floor (1992 / 3) : ℕ) - 
  (Nat.floor (1992 / 6) : ℕ) - 
  (Nat.floor (1992 / 15) : ℕ) + 
  (Nat.floor (1992 / 30) : ℕ)

theorem count_special_numbers_eq_266 : count_special_numbers = 266 := by
  sorry

end NUMINAMATH_CALUDE_count_special_numbers_eq_266_l4038_403883


namespace NUMINAMATH_CALUDE_regular_polygon_with_20_diagonals_has_8_sides_l4038_403869

/-- The number of diagonals in a regular polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: A regular polygon with 20 diagonals has 8 sides -/
theorem regular_polygon_with_20_diagonals_has_8_sides :
  ∃ (n : ℕ), n > 2 ∧ num_diagonals n = 20 → n = 8 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_with_20_diagonals_has_8_sides_l4038_403869


namespace NUMINAMATH_CALUDE_smallest_solution_is_smaller_root_l4038_403865

-- Define the quadratic equation
def quadratic_eq (x : ℝ) : Prop := 2 * x^2 + 9 * x - 92 = 0

-- Define the original equation
def original_eq (x : ℝ) : Prop := 3 * x^2 + 24 * x - 92 = x * (x + 15)

-- Theorem statement
theorem smallest_solution_is_smaller_root :
  ∃ (x : ℝ), quadratic_eq x ∧ 
  (∀ (y : ℝ), quadratic_eq y → x ≤ y) ∧
  (∀ (z : ℝ), original_eq z → x ≤ z) := by
  sorry

end NUMINAMATH_CALUDE_smallest_solution_is_smaller_root_l4038_403865


namespace NUMINAMATH_CALUDE_andy_solves_56_problems_l4038_403872

/-- The number of problems Andy solves -/
def problems_solved (first last : ℕ) : ℕ := last - first + 1

/-- Theorem stating that Andy solves 56 problems -/
theorem andy_solves_56_problems : 
  problems_solved 70 125 = 56 := by sorry

end NUMINAMATH_CALUDE_andy_solves_56_problems_l4038_403872


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l4038_403857

theorem necessary_not_sufficient_condition (x : ℝ) :
  (x^2 - x < 0) → (-1 < x ∧ x < 1) ∧
  ∃ y, -1 < y ∧ y < 1 ∧ ¬(y^2 - y < 0) := by
  sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l4038_403857


namespace NUMINAMATH_CALUDE_sara_bought_movie_cost_l4038_403885

/-- The cost of Sara's bought movie -/
def cost_of_bought_movie (ticket_price : ℚ) (num_tickets : ℕ) (rental_price : ℚ) (total_spent : ℚ) : ℚ :=
  total_spent - (ticket_price * num_tickets + rental_price)

/-- Theorem stating the cost of Sara's bought movie -/
theorem sara_bought_movie_cost :
  let ticket_price : ℚ := 10.62
  let num_tickets : ℕ := 2
  let rental_price : ℚ := 1.59
  let total_spent : ℚ := 36.78
  cost_of_bought_movie ticket_price num_tickets rental_price total_spent = 13.95 := by
  sorry

end NUMINAMATH_CALUDE_sara_bought_movie_cost_l4038_403885


namespace NUMINAMATH_CALUDE_circle_area_theorem_l4038_403837

theorem circle_area_theorem (z₁ z₂ : ℂ) 
  (h₁ : z₁^2 - 4*z₁*z₂ + 4*z₂^2 = 0) 
  (h₂ : Complex.abs z₂ = 2) : 
  Real.pi * (Complex.abs z₁ / 2)^2 = 4 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_circle_area_theorem_l4038_403837


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l4038_403890

-- Define the quadratic function
def f (x : ℝ) : ℝ := x^2 - 5*x - 14

-- Define the solution set
def solution_set : Set ℝ := {x | -2 < x ∧ x < 7}

-- Theorem statement
theorem quadratic_inequality_solution :
  {x : ℝ | f x < 0} = solution_set := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l4038_403890


namespace NUMINAMATH_CALUDE_speed_difference_l4038_403880

/-- The difference in average speeds between no traffic and heavy traffic conditions -/
theorem speed_difference (distance : ℝ) (time_heavy : ℝ) (time_no : ℝ)
  (h_distance : distance = 200)
  (h_time_heavy : time_heavy = 5)
  (h_time_no : time_no = 4) :
  distance / time_no - distance / time_heavy = 10 := by
  sorry

end NUMINAMATH_CALUDE_speed_difference_l4038_403880


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l4038_403871

/-- Given a geometric sequence with common ratio 2, prove that (2a₁ + a₂) / (2a₃ + a₄) = 1/4 -/
theorem geometric_sequence_ratio (a : ℕ → ℝ) :
  (∀ n, a (n + 1) = 2 * a n) →
  (2 * a 1 + a 2) / (2 * a 3 + a 4) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l4038_403871


namespace NUMINAMATH_CALUDE_profit_difference_exists_l4038_403893

/-- Represents the strategy of selling or renting a movie -/
inductive SaleStrategy
  | Forever
  | Rental

/-- Represents the economic factors affecting movie sales -/
structure EconomicFactors where
  price : ℝ
  customerBase : ℕ
  sharingRate : ℝ
  rentalFrequency : ℕ
  adminCosts : ℝ
  piracyRisk : ℝ

/-- Calculates the total profit for a given sale strategy and economic factors -/
def totalProfit (strategy : SaleStrategy) (factors : EconomicFactors) : ℝ :=
  sorry

/-- Theorem stating that the total profit from selling a movie "forever" 
    may be different from the total profit from temporary rentals -/
theorem profit_difference_exists :
  ∃ (f₁ f₂ : EconomicFactors), 
    totalProfit SaleStrategy.Forever f₁ ≠ totalProfit SaleStrategy.Rental f₂ :=
  sorry

end NUMINAMATH_CALUDE_profit_difference_exists_l4038_403893


namespace NUMINAMATH_CALUDE_dale_max_nuts_l4038_403808

/-- The maximum number of nuts Dale can guarantee to get -/
def max_nuts_dale : ℕ := 71

/-- The total number of nuts -/
def total_nuts : ℕ := 1001

/-- The number of initial piles -/
def initial_piles : ℕ := 3

/-- The number of possible pile configurations -/
def pile_configs : ℕ := 8

theorem dale_max_nuts :
  ∀ (a b c : ℕ) (N : ℕ),
  a + b + c = total_nuts →
  1 ≤ N ∧ N ≤ total_nuts →
  (∃ (moved : ℕ), moved ≤ max_nuts_dale ∧
    (N = 0 ∨ N = a ∨ N = b ∨ N = c ∨ N = a + b ∨ N = b + c ∨ N = c + a ∨ N = total_nuts ∨
     (N < total_nuts ∧ moved = N - min N (min a (min b (min c (min (a + b) (min (b + c) (c + a))))))) ∨
     (N > 0 ∧ moved = min (a - N) (min (b - N) (min (c - N) (min (a + b - N) (min (b + c - N) (c + a - N)))))))) :=
by sorry

end NUMINAMATH_CALUDE_dale_max_nuts_l4038_403808


namespace NUMINAMATH_CALUDE_basic_astrophysics_is_108_degrees_l4038_403851

/-- Represents the research and development budget allocation --/
structure BudgetAllocation where
  microphotonics : ℝ
  home_electronics : ℝ
  food_additives : ℝ
  genetically_modified_microorganisms : ℝ
  industrial_lubricants : ℝ

/-- The total degrees in a circle --/
def total_degrees : ℝ := 360

/-- Calculate the degrees for basic astrophysics research --/
def basic_astrophysics_degrees (ba : BudgetAllocation) : ℝ :=
  total_degrees * (1 - (ba.microphotonics + ba.home_electronics + ba.food_additives + 
                        ba.genetically_modified_microorganisms + ba.industrial_lubricants))

/-- Theorem stating that the degrees for basic astrophysics research is 108 --/
theorem basic_astrophysics_is_108_degrees (ba : BudgetAllocation) 
    (h1 : ba.microphotonics = 0.09)
    (h2 : ba.home_electronics = 0.14)
    (h3 : ba.food_additives = 0.10)
    (h4 : ba.genetically_modified_microorganisms = 0.29)
    (h5 : ba.industrial_lubricants = 0.08) :
    basic_astrophysics_degrees ba = 108 := by
  sorry

#check basic_astrophysics_is_108_degrees

end NUMINAMATH_CALUDE_basic_astrophysics_is_108_degrees_l4038_403851


namespace NUMINAMATH_CALUDE_complement_of_A_l4038_403879

def U : Set ℝ := Set.univ
def A : Set ℝ := {x | x < 1}

theorem complement_of_A : Set.compl A = {x : ℝ | x ≥ 1} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_l4038_403879


namespace NUMINAMATH_CALUDE_area_ratio_circle_ellipse_l4038_403868

/-- The ratio of the area between a circle and an ellipse to the area of the circle -/
theorem area_ratio_circle_ellipse :
  let circle_diameter : ℝ := 4
  let ellipse_major_axis : ℝ := 8
  let ellipse_minor_axis : ℝ := 6
  let circle_area := π * (circle_diameter / 2)^2
  let ellipse_area := π * (ellipse_major_axis / 2) * (ellipse_minor_axis / 2)
  (ellipse_area - circle_area) / circle_area = 2 := by
  sorry

end NUMINAMATH_CALUDE_area_ratio_circle_ellipse_l4038_403868


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l4038_403845

theorem quadratic_inequality_range (a : ℝ) : 
  (∃ x : ℝ, x^2 + 2*a*x + 1 < 0) ↔ a ∈ Set.Iio (-1) ∪ Set.Ioi 1 :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l4038_403845


namespace NUMINAMATH_CALUDE_cubic_equation_property_l4038_403881

/-- A cubic equation with coefficients a, b, c, and three non-zero real roots forming a geometric progression -/
structure CubicEquation where
  a : ℝ
  b : ℝ
  c : ℝ
  root1 : ℝ
  root2 : ℝ
  root3 : ℝ
  nonzero_roots : root1 ≠ 0 ∧ root2 ≠ 0 ∧ root3 ≠ 0
  is_root1 : root1^3 + a*root1^2 + b*root1 + c = 0
  is_root2 : root2^3 + a*root2^2 + b*root2 + c = 0
  is_root3 : root3^3 + a*root3^2 + b*root3 + c = 0
  geometric_progression : ∃ (q : ℝ), q ≠ 0 ∧ q ≠ 1 ∧ (root2 = q * root1) ∧ (root3 = q * root2)

/-- The theorem stating that a^3c - b^3 = 0 for a cubic equation with three non-zero real roots in geometric progression -/
theorem cubic_equation_property (eq : CubicEquation) : eq.a^3 * eq.c - eq.b^3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_property_l4038_403881


namespace NUMINAMATH_CALUDE_smallest_possible_a_l4038_403889

theorem smallest_possible_a (a b c d x : ℤ) 
  (h1 : (a - 2*b) * x = 1)
  (h2 : (b - 3*c) * x = 1)
  (h3 : (c - 4*d) * x = 1)
  (h4 : x + 100 = d)
  (h5 : x > 0) :
  a ≥ 2433 ∧ ∃ (a₀ b₀ c₀ d₀ x₀ : ℤ), 
    a₀ = 2433 ∧
    (a₀ - 2*b₀) * x₀ = 1 ∧
    (b₀ - 3*c₀) * x₀ = 1 ∧
    (c₀ - 4*d₀) * x₀ = 1 ∧
    x₀ + 100 = d₀ ∧
    x₀ > 0 :=
by sorry

end NUMINAMATH_CALUDE_smallest_possible_a_l4038_403889


namespace NUMINAMATH_CALUDE_prob_pass_exactly_once_l4038_403822

/-- The probability of passing a single computer test -/
def p : ℚ := 1 / 3

/-- The number of times the test is taken -/
def n : ℕ := 3

/-- The number of times we want the event to occur -/
def k : ℕ := 1

/-- The probability of passing exactly k times in n independent trials -/
def prob_exactly_k (p : ℚ) (n k : ℕ) : ℚ :=
  (n.choose k) * p^k * (1 - p)^(n - k)

theorem prob_pass_exactly_once :
  prob_exactly_k p n k = 4 / 9 := by
  sorry

end NUMINAMATH_CALUDE_prob_pass_exactly_once_l4038_403822


namespace NUMINAMATH_CALUDE_lemonade_pitchers_sum_l4038_403892

theorem lemonade_pitchers_sum : 
  let first_intermission : ℚ := 0.25
  let second_intermission : ℚ := 0.42
  let third_intermission : ℚ := 0.25
  first_intermission + second_intermission + third_intermission = 0.92 := by
sorry

end NUMINAMATH_CALUDE_lemonade_pitchers_sum_l4038_403892


namespace NUMINAMATH_CALUDE_waiter_new_customers_l4038_403819

theorem waiter_new_customers 
  (initial_customers : ℕ) 
  (left_customers : ℕ) 
  (final_customers : ℕ) 
  (h1 : initial_customers = 14) 
  (h2 : left_customers = 3) 
  (h3 : final_customers = 50) : 
  final_customers - (initial_customers - left_customers) = 39 := by
  sorry

end NUMINAMATH_CALUDE_waiter_new_customers_l4038_403819


namespace NUMINAMATH_CALUDE_ratio_section_area_l4038_403812

/-- Regular quadrilateral prism -/
structure RegularQuadPrism where
  base : Real
  height : Real

/-- Cross-section passing through midpoints -/
def midpoint_section (p : RegularQuadPrism) : Real :=
  12

/-- Cross-section dividing axis in ratio 1:3 -/
def ratio_section (p : RegularQuadPrism) : Real :=
  9

/-- Theorem statement -/
theorem ratio_section_area (p : RegularQuadPrism) :
  midpoint_section p = 12 → ratio_section p = 9 := by
  sorry

end NUMINAMATH_CALUDE_ratio_section_area_l4038_403812


namespace NUMINAMATH_CALUDE_sandy_shopping_money_l4038_403846

theorem sandy_shopping_money (initial_amount : ℝ) : 
  (initial_amount * 0.7 = 210) → initial_amount = 300 := by
  sorry

end NUMINAMATH_CALUDE_sandy_shopping_money_l4038_403846


namespace NUMINAMATH_CALUDE_molly_current_age_l4038_403897

/-- Represents Molly's age and candle information --/
structure MollysBirthday where
  last_year_candles : ℕ
  additional_candles : ℕ
  friend_gift_candles : ℕ

/-- Calculates Molly's current age based on her birthday information --/
def current_age (mb : MollysBirthday) : ℕ :=
  mb.last_year_candles + 1

/-- Theorem stating Molly's current age --/
theorem molly_current_age (mb : MollysBirthday)
  (h1 : mb.last_year_candles = 14)
  (h2 : mb.additional_candles = 6)
  (h3 : mb.friend_gift_candles = 3) :
  current_age mb = 15 := by
  sorry

end NUMINAMATH_CALUDE_molly_current_age_l4038_403897


namespace NUMINAMATH_CALUDE_functional_equation_solution_l4038_403818

noncomputable def f (x : ℝ) : ℝ := (4 * x^2 - x + 1) / (5 * (x - 1))

theorem functional_equation_solution (x : ℝ) (h1 : x ≠ 0) (h2 : x ≠ 1) (h3 : x ≠ -1) :
  x * f x + 2 * f ((x - 1) / (x + 1)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l4038_403818


namespace NUMINAMATH_CALUDE_water_container_percentage_l4038_403888

theorem water_container_percentage (initial_water : ℝ) (capacity : ℝ) (added_water : ℝ) :
  capacity = 40 →
  added_water = 14 →
  (initial_water + added_water) / capacity = 3/4 →
  initial_water / capacity = 2/5 :=
by sorry

end NUMINAMATH_CALUDE_water_container_percentage_l4038_403888


namespace NUMINAMATH_CALUDE_largest_c_for_range_l4038_403882

theorem largest_c_for_range (f : ℝ → ℝ) (c : ℝ) : 
  (∀ x, f x = x^2 - 7*x + c) →
  (∃ x, f x = 3) →
  c ≤ 61/4 ∧ ∀ d > 61/4, ¬∃ x, x^2 - 7*x + d = 3 :=
by sorry

end NUMINAMATH_CALUDE_largest_c_for_range_l4038_403882


namespace NUMINAMATH_CALUDE_whistle_search_bound_l4038_403839

/-- Represents a football field -/
structure FootballField where
  length : ℝ
  width : ℝ

/-- Represents the position of an object on the field -/
structure Position where
  x : ℝ
  y : ℝ

/-- Represents the referee's search process -/
def search (field : FootballField) (start : Position) (whistle : Position) : ℕ :=
  sorry

/-- Theorem stating the upper bound on the number of steps needed to find the whistle -/
theorem whistle_search_bound 
  (field : FootballField)
  (start : Position)
  (whistle : Position)
  (h_field_size : field.length = 100 ∧ field.width = 70)
  (h_start_corner : start.x = 0 ∧ start.y = 0)
  (h_whistle_on_field : whistle.x ≥ 0 ∧ whistle.x ≤ field.length ∧ whistle.y ≥ 0 ∧ whistle.y ≤ field.width)
  (d : ℝ)
  (h_initial_distance : d = Real.sqrt ((whistle.x - start.x)^2 + (whistle.y - start.y)^2)) :
  (search field start whistle) ≤ ⌊Real.sqrt 2 * (d + 1)⌋ + 4 :=
sorry

end NUMINAMATH_CALUDE_whistle_search_bound_l4038_403839


namespace NUMINAMATH_CALUDE_five_balls_three_boxes_l4038_403825

/-- The number of ways to distribute distinguishable balls into distinguishable boxes -/
def distribute_balls (num_balls : ℕ) (num_boxes : ℕ) : ℕ :=
  num_boxes ^ num_balls

/-- Theorem: There are 243 ways to distribute 5 distinguishable balls into 3 distinguishable boxes -/
theorem five_balls_three_boxes : distribute_balls 5 3 = 243 := by
  sorry

end NUMINAMATH_CALUDE_five_balls_three_boxes_l4038_403825


namespace NUMINAMATH_CALUDE_product_of_difference_and_sum_of_squares_l4038_403835

theorem product_of_difference_and_sum_of_squares (a b : ℝ) 
  (h1 : a - b = 5) 
  (h2 : a^2 + b^2 = 13) : 
  a * b = -6 := by sorry

end NUMINAMATH_CALUDE_product_of_difference_and_sum_of_squares_l4038_403835


namespace NUMINAMATH_CALUDE_book_pages_introduction_l4038_403824

theorem book_pages_introduction (total_pages : ℕ) (text_pages : ℕ) : 
  total_pages = 98 →
  text_pages = 19 →
  (total_pages - total_pages / 2 - text_pages * 2 = 11) :=
by
  sorry

end NUMINAMATH_CALUDE_book_pages_introduction_l4038_403824


namespace NUMINAMATH_CALUDE_pie_slices_remaining_l4038_403860

theorem pie_slices_remaining (total_slices : ℕ) 
  (joe_fraction darcy_fraction carl_fraction emily_fraction : ℚ) : 
  total_slices = 24 →
  joe_fraction = 1/3 →
  darcy_fraction = 1/4 →
  carl_fraction = 1/6 →
  emily_fraction = 1/8 →
  total_slices - (total_slices * joe_fraction + total_slices * darcy_fraction + 
    total_slices * carl_fraction + total_slices * emily_fraction) = 3 := by
  sorry

end NUMINAMATH_CALUDE_pie_slices_remaining_l4038_403860


namespace NUMINAMATH_CALUDE_dogwood_tree_count_l4038_403830

/-- The number of dogwood trees in the park after planting -/
def total_trees (current : ℕ) (planted_today : ℕ) (planted_tomorrow : ℕ) : ℕ :=
  current + planted_today + planted_tomorrow

/-- Theorem stating that the total number of trees after planting is 100 -/
theorem dogwood_tree_count : total_trees 39 41 20 = 100 := by
  sorry

end NUMINAMATH_CALUDE_dogwood_tree_count_l4038_403830


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_12_l4038_403886

def arithmeticSequence (a : ℕ → ℚ) : Prop :=
  ∀ n, a (n + 1) - a n = 1

def geometricMean (x y z : ℚ) : Prop :=
  z * z = x * y

def sumOfTerms (a : ℕ → ℚ) (n : ℕ) : ℚ :=
  (n : ℚ) * (a 1 + a n) / 2

theorem arithmetic_sequence_sum_12 (a : ℕ → ℚ) :
  arithmeticSequence a →
  geometricMean (a 3) (a 11) (a 6) →
  sumOfTerms a 12 = 96 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_12_l4038_403886


namespace NUMINAMATH_CALUDE_f_monotone_increasing_max_a_value_l4038_403806

open Real

noncomputable def f (x : ℝ) : ℝ := (x^2 - x + 1) / (Real.exp x)

theorem f_monotone_increasing :
  ∀ x y, 1 < x ∧ x < y ∧ y < 2 → f x < f y :=
sorry

theorem max_a_value :
  (∀ x, x > 0 → Real.exp x * f x ≥ a + Real.log x) → a ≤ 1 :=
sorry

end NUMINAMATH_CALUDE_f_monotone_increasing_max_a_value_l4038_403806


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l4038_403843

/-- Given an arithmetic sequence {a_n} with sum of first n terms S_n,
    prove that if S_6 = 3S_2 + 24, then the common difference d = 2 -/
theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℝ) -- a_n is the nth term of the arithmetic sequence
  (S : ℕ → ℝ) -- S_n is the sum of the first n terms
  (h_arithmetic : ∀ n, a (n + 1) - a n = a 2 - a 1) -- condition for arithmetic sequence
  (h_sum : ∀ n, S n = n * (a 1 + a n) / 2) -- formula for sum of arithmetic sequence
  (h_given : S 6 = 3 * S 2 + 24) -- given condition
  : a 2 - a 1 = 2 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l4038_403843


namespace NUMINAMATH_CALUDE_cube_surface_area_l4038_403850

/-- The surface area of a cube with edge length 8 cm is 384 cm² -/
theorem cube_surface_area : 
  let edge_length : ℝ := 8
  let face_area : ℝ := edge_length * edge_length
  let surface_area : ℝ := 6 * face_area
  surface_area = 384 :=
by sorry

end NUMINAMATH_CALUDE_cube_surface_area_l4038_403850


namespace NUMINAMATH_CALUDE_units_digit_factorial_sum_100_l4038_403856

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def units_digit (n : ℕ) : ℕ := n % 10

def factorial_sum (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

theorem units_digit_factorial_sum_100 :
  units_digit (factorial_sum 100) = 3 := by
sorry

end NUMINAMATH_CALUDE_units_digit_factorial_sum_100_l4038_403856


namespace NUMINAMATH_CALUDE_circle_area_with_radius_3_l4038_403841

theorem circle_area_with_radius_3 :
  ∀ (π : ℝ), π > 0 →
  let r : ℝ := 3
  let area := π * r^2
  area = 9 * π :=
by sorry

end NUMINAMATH_CALUDE_circle_area_with_radius_3_l4038_403841


namespace NUMINAMATH_CALUDE_music_tool_cost_l4038_403870

/-- The cost of Joan's music tool purchase -/
theorem music_tool_cost (trumpet_cost song_book_cost total_spent : ℚ)
  (h1 : trumpet_cost = 149.16)
  (h2 : song_book_cost = 4.14)
  (h3 : total_spent = 163.28) :
  total_spent - (trumpet_cost + song_book_cost) = 9.98 := by
  sorry

end NUMINAMATH_CALUDE_music_tool_cost_l4038_403870


namespace NUMINAMATH_CALUDE_v_4_value_l4038_403866

/-- A sequence of real numbers satisfying the given recurrence relation -/
def RecursiveSequence (v : ℕ → ℝ) : Prop :=
  ∀ n, v (n + 2) = 2 * v (n + 1) + v n

theorem v_4_value (v : ℕ → ℝ) (h_rec : RecursiveSequence v) 
    (h_v2 : v 2 = 7) (h_v5 : v 5 = 53) : v 4 = 22.6 := by
  sorry

end NUMINAMATH_CALUDE_v_4_value_l4038_403866


namespace NUMINAMATH_CALUDE_first_hole_depth_l4038_403804

/-- Represents the depth of a hole dug by workers -/
structure HoleDigging where
  workers : ℕ
  hours : ℕ
  depth : ℝ

theorem first_hole_depth 
  (hole1 : HoleDigging)
  (hole2 : HoleDigging)
  (h1 : hole1.workers = 45)
  (h2 : hole1.hours = 8)
  (h3 : hole2.workers = 110)
  (h4 : hole2.hours = 6)
  (h5 : hole2.depth = 55)
  (h6 : hole1.workers * hole1.hours * hole2.depth = hole2.workers * hole2.hours * hole1.depth) :
  hole1.depth = 30 := by
sorry


end NUMINAMATH_CALUDE_first_hole_depth_l4038_403804


namespace NUMINAMATH_CALUDE_first_term_is_24_l4038_403832

/-- The first term of an infinite geometric series with common ratio -1/3 and sum 18 -/
def first_term_geometric_series (r : ℚ) (S : ℚ) : ℚ :=
  S * (1 - r)

/-- Theorem: The first term of an infinite geometric series with common ratio -1/3 and sum 18 is 24 -/
theorem first_term_is_24 :
  first_term_geometric_series (-1/3) 18 = 24 := by
  sorry

end NUMINAMATH_CALUDE_first_term_is_24_l4038_403832


namespace NUMINAMATH_CALUDE_nurses_who_quit_l4038_403817

theorem nurses_who_quit (initial_doctors initial_nurses doctors_quit total_remaining : ℕ) :
  initial_doctors = 11 →
  initial_nurses = 18 →
  doctors_quit = 5 →
  total_remaining = 22 →
  initial_doctors + initial_nurses - doctors_quit - total_remaining = 2 := by
  sorry

end NUMINAMATH_CALUDE_nurses_who_quit_l4038_403817


namespace NUMINAMATH_CALUDE_ellipse_focal_length_implies_m_8_l4038_403820

-- Define the ellipse equation
def ellipse_equation (x y m : ℝ) : Prop :=
  x^2 / (10 - m) + y^2 / (m - 2) = 1

-- Define the condition for major axis along y-axis
def major_axis_y (m : ℝ) : Prop :=
  m - 2 > 10 - m

-- Define the focal length
def focal_length (m : ℝ) : ℝ :=
  4

-- Theorem statement
theorem ellipse_focal_length_implies_m_8 :
  ∀ m : ℝ,
  (∀ x y : ℝ, ellipse_equation x y m) →
  major_axis_y m →
  focal_length m = 4 →
  m = 8 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_focal_length_implies_m_8_l4038_403820


namespace NUMINAMATH_CALUDE_games_not_working_l4038_403859

theorem games_not_working (friend_games garage_games good_games : ℕ) : 
  friend_games = 2 → garage_games = 2 → good_games = 2 →
  friend_games + garage_games - good_games = 2 := by
sorry

end NUMINAMATH_CALUDE_games_not_working_l4038_403859


namespace NUMINAMATH_CALUDE_vector_collinearity_l4038_403848

def a : ℝ × ℝ := (1, 3)
def b : ℝ × ℝ := (-2, 1)
def c : ℝ × ℝ := (3, 2)

def collinear (u v : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, u.1 * v.2 = t * u.2 * v.1

theorem vector_collinearity (k : ℝ) :
  collinear c ((k * a.1 + b.1, k * a.2 + b.2)) → k = -1 := by
  sorry

end NUMINAMATH_CALUDE_vector_collinearity_l4038_403848


namespace NUMINAMATH_CALUDE_rectangle_squares_l4038_403834

theorem rectangle_squares (N : ℕ) : 
  (∃ x y : ℕ, N = x * (x + 9) ∧ N = y * (y + 6)) → N = 112 := by
sorry

end NUMINAMATH_CALUDE_rectangle_squares_l4038_403834


namespace NUMINAMATH_CALUDE_cone_volume_ratio_l4038_403810

theorem cone_volume_ratio : 
  let r_C : ℝ := 20
  let h_C : ℝ := 50
  let r_D : ℝ := 25
  let h_D : ℝ := 40
  let V_C := (1/3) * π * r_C^2 * h_C
  let V_D := (1/3) * π * r_D^2 * h_D
  V_C / V_D = 4/5 := by sorry

end NUMINAMATH_CALUDE_cone_volume_ratio_l4038_403810


namespace NUMINAMATH_CALUDE_chocolate_kisses_bags_l4038_403873

theorem chocolate_kisses_bags (total_candy : ℕ) (total_bags : ℕ) (heart_bags : ℕ) (non_chocolate_pieces : ℕ) :
  total_candy = 63 →
  total_bags = 9 →
  heart_bags = 2 →
  non_chocolate_pieces = 28 →
  total_candy % total_bags = 0 →
  ∃ (kisses_bags : ℕ),
    kisses_bags = total_bags - heart_bags - (non_chocolate_pieces / (total_candy / total_bags)) ∧
    kisses_bags = 3 :=
by sorry

end NUMINAMATH_CALUDE_chocolate_kisses_bags_l4038_403873


namespace NUMINAMATH_CALUDE_cos_max_value_l4038_403895

theorem cos_max_value (a b : ℝ) (h : Real.cos (a + b) = Real.cos a + Real.cos b) :
  ∃ (max : ℝ), max = Real.sqrt 3 - 1 ∧ Real.cos a ≤ max ∧ 
  ∃ (a₀ b₀ : ℝ), Real.cos (a₀ + b₀) = Real.cos a₀ + Real.cos b₀ ∧ Real.cos a₀ = max :=
by sorry

end NUMINAMATH_CALUDE_cos_max_value_l4038_403895


namespace NUMINAMATH_CALUDE_library_capacity_is_400_l4038_403853

/-- The capacity of Karson's home library -/
def library_capacity : ℕ := sorry

/-- The number of books Karson currently has -/
def current_books : ℕ := 120

/-- The number of additional books Karson needs to buy -/
def additional_books : ℕ := 240

/-- The percentage of the library that will be full after buying additional books -/
def full_percentage : ℚ := 9/10

theorem library_capacity_is_400 : 
  library_capacity = 400 :=
by
  have h1 : current_books + additional_books = (library_capacity : ℚ) * full_percentage :=
    sorry
  sorry

end NUMINAMATH_CALUDE_library_capacity_is_400_l4038_403853


namespace NUMINAMATH_CALUDE_solutions_equation1_solutions_equation2_l4038_403821

-- Define the equations
def equation1 (x : ℝ) : Prop := x^2 - 4*x + 2 = 0
def equation2 (x : ℝ) : Prop := (x-3)^2 = 2*x - 6

-- Theorem for the first equation
theorem solutions_equation1 : 
  {x : ℝ | equation1 x} = {2 + Real.sqrt 2, 2 - Real.sqrt 2} :=
sorry

-- Theorem for the second equation
theorem solutions_equation2 :
  {x : ℝ | equation2 x} = {3, 5} :=
sorry

end NUMINAMATH_CALUDE_solutions_equation1_solutions_equation2_l4038_403821


namespace NUMINAMATH_CALUDE_range_of_a_l4038_403863

-- Define proposition p
def p (a : ℝ) : Prop := ∀ x : ℝ, x^2 + 2*a*x + 4 > 0

-- Define proposition q
def q (a : ℝ) : Prop := ∀ x y : ℝ, x < y → (3 - 2*a)^x < (3 - 2*a)^y

-- Define the range of a
def range_a (a : ℝ) : Prop := (1 ≤ a ∧ a < 2) ∨ a ≤ -2

-- State the theorem
theorem range_of_a : ∀ a : ℝ, (p a ∨ q a) ∧ ¬(p a ∧ q a) → range_a a := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l4038_403863


namespace NUMINAMATH_CALUDE_cat_weight_ratio_l4038_403815

theorem cat_weight_ratio (female_weight male_weight : ℝ) : 
  female_weight = 2 →
  male_weight > female_weight →
  female_weight + male_weight = 6 →
  male_weight / female_weight = 2 := by
sorry

end NUMINAMATH_CALUDE_cat_weight_ratio_l4038_403815


namespace NUMINAMATH_CALUDE_distance_traveled_is_9_miles_l4038_403852

/-- The total distance traveled when biking and jogging for a given time and rate -/
def total_distance (bike_time : ℚ) (bike_rate : ℚ) (jog_time : ℚ) (jog_rate : ℚ) : ℚ :=
  (bike_time * bike_rate) + (jog_time * jog_rate)

/-- Theorem stating that the total distance traveled is 9 miles -/
theorem distance_traveled_is_9_miles :
  let bike_time : ℚ := 1/2  -- 30 minutes in hours
  let bike_rate : ℚ := 6
  let jog_time : ℚ := 3/4   -- 45 minutes in hours
  let jog_rate : ℚ := 8
  total_distance bike_time bike_rate jog_time jog_rate = 9 := by
  sorry

#eval total_distance (1/2) 6 (3/4) 8

end NUMINAMATH_CALUDE_distance_traveled_is_9_miles_l4038_403852


namespace NUMINAMATH_CALUDE_k_range_for_three_elements_l4038_403864

def P (k : ℝ) : Set ℕ := {x : ℕ | 2 < x ∧ x < k}

theorem k_range_for_three_elements (k : ℝ) :
  (∃ (a b c : ℕ), P k = {a, b, c} ∧ a ≠ b ∧ a ≠ c ∧ b ≠ c) →
  5 < k ∧ k ≤ 6 :=
by sorry

end NUMINAMATH_CALUDE_k_range_for_three_elements_l4038_403864


namespace NUMINAMATH_CALUDE_cuboidal_block_dimension_l4038_403811

/-- Given a cuboidal block with dimensions x cm × 9 cm × 12 cm that can be cut into at least 24 equal cubes,
    prove that the length of the first dimension (x) must be 6 cm. -/
theorem cuboidal_block_dimension (x : ℕ) : 
  (∃ (n : ℕ), n ≥ 24 ∧ x * 9 * 12 = n * (gcd x (gcd 9 12))^3) → x = 6 := by
  sorry

end NUMINAMATH_CALUDE_cuboidal_block_dimension_l4038_403811


namespace NUMINAMATH_CALUDE_min_value_parallel_vectors_l4038_403840

theorem min_value_parallel_vectors (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  let a : Fin 2 → ℝ := ![3, 2]
  let b : Fin 2 → ℝ := ![x, 1 - y]
  (∃ (k : ℝ), ∀ i, a i = k * b i) →
  (3 / x + 2 / y) ≥ 8 ∧ ∃ x₀ y₀, x₀ > 0 ∧ y₀ > 0 ∧ 3 / x₀ + 2 / y₀ = 8 :=
by sorry

end NUMINAMATH_CALUDE_min_value_parallel_vectors_l4038_403840


namespace NUMINAMATH_CALUDE_largest_consecutive_sum_l4038_403838

theorem largest_consecutive_sum (n : ℕ) : n = 14141 ↔ 
  (∀ k : ℕ, k ≤ n → (k * (k + 1)) / 2 ≤ 100000000) ∧
  (∀ m : ℕ, m > n → (m * (m + 1)) / 2 > 100000000) := by
sorry

end NUMINAMATH_CALUDE_largest_consecutive_sum_l4038_403838


namespace NUMINAMATH_CALUDE_smallest_four_digit_prime_divisible_proof_l4038_403802

def smallest_four_digit_prime_divisible : ℕ := 2310

theorem smallest_four_digit_prime_divisible_proof :
  (smallest_four_digit_prime_divisible ≥ 1000) ∧
  (smallest_four_digit_prime_divisible < 10000) ∧
  (smallest_four_digit_prime_divisible % 2 = 0) ∧
  (smallest_four_digit_prime_divisible % 3 = 0) ∧
  (smallest_four_digit_prime_divisible % 5 = 0) ∧
  (smallest_four_digit_prime_divisible % 7 = 0) ∧
  (smallest_four_digit_prime_divisible % 11 = 0) ∧
  (∀ n : ℕ, n ≥ 1000 ∧ n < 10000 ∧ n % 2 = 0 ∧ n % 3 = 0 ∧ n % 5 = 0 ∧ n % 7 = 0 ∧ n % 11 = 0 → n ≥ smallest_four_digit_prime_divisible) :=
by sorry

end NUMINAMATH_CALUDE_smallest_four_digit_prime_divisible_proof_l4038_403802


namespace NUMINAMATH_CALUDE_fair_coin_head_is_random_event_l4038_403814

/-- Represents the possible outcomes of a coin toss -/
inductive CoinOutcome
  | Head
  | Tail

/-- Represents different types of events -/
inductive EventType
  | Impossible
  | Certain
  | Random
  | Definite

/-- A fair coin toss -/
structure FairCoinToss where
  /-- The coin has two possible outcomes -/
  outcome : CoinOutcome
  /-- The probability of getting heads is 0.5 -/
  prob_head : ℝ := 0.5
  /-- The probability of getting tails is 0.5 -/
  prob_tail : ℝ := 0.5
  /-- The probabilities sum to 1 -/
  prob_sum : prob_head + prob_tail = 1

/-- The theorem stating that tossing a fair coin with the head facing up is a random event -/
theorem fair_coin_head_is_random_event (toss : FairCoinToss) : 
  EventType.Random = 
    match toss.outcome with
    | CoinOutcome.Head => EventType.Random
    | CoinOutcome.Tail => EventType.Random :=
by
  sorry


end NUMINAMATH_CALUDE_fair_coin_head_is_random_event_l4038_403814


namespace NUMINAMATH_CALUDE_exactly_one_subset_exactly_one_element_l4038_403899

-- Define the set A
def A (a : ℝ) : Set ℝ := {x | a * x^2 + 2 * x + 1 = 0}

-- Theorem for part 1
theorem exactly_one_subset (a : ℝ) : (∃! (S : Set ℝ), S ⊆ A a) ↔ a > 1 := by sorry

-- Theorem for part 2
theorem exactly_one_element (a : ℝ) : (∃! x, x ∈ A a) ↔ a = 0 ∨ a = 1 := by sorry

end NUMINAMATH_CALUDE_exactly_one_subset_exactly_one_element_l4038_403899


namespace NUMINAMATH_CALUDE_digital_earth_definition_l4038_403800

/-- Definition of Digital Earth -/
def DigitalEarth : Type := Unit

/-- Property of Digital Earth being a digitized, informational virtual Earth -/
def is_digitized_informational_virtual_earth (de : DigitalEarth) : Prop :=
  -- This is left abstract as the problem doesn't provide specific criteria
  True

/-- Theorem stating that Digital Earth refers to a digitized, informational virtual Earth -/
theorem digital_earth_definition :
  ∀ (de : DigitalEarth), is_digitized_informational_virtual_earth de :=
sorry

end NUMINAMATH_CALUDE_digital_earth_definition_l4038_403800


namespace NUMINAMATH_CALUDE_sine_cosine_inequality_l4038_403823

theorem sine_cosine_inequality (b a : ℝ) (hb : 0 < b ∧ b < 1) (ha : 0 < a ∧ a < Real.pi / 2) :
  Real.rpow b (Real.sin a) < Real.rpow b (Real.sin a) ∧ Real.rpow b (Real.sin a) < Real.rpow b (Real.cos a) := by
  sorry

end NUMINAMATH_CALUDE_sine_cosine_inequality_l4038_403823


namespace NUMINAMATH_CALUDE_largest_common_divisor_l4038_403884

def is_odd (n : ℕ) : Prop := ∃ k, n = 2*k + 1

def product_function (n : ℕ) : ℕ := (n+2)*(n+4)*(n+6)*(n+8)*(n+10)

theorem largest_common_divisor (n : ℕ) (h : is_odd n) :
  (∀ m : ℕ, m > 8 → ¬(m ∣ product_function n)) ∧
  (8 ∣ product_function n) :=
sorry

end NUMINAMATH_CALUDE_largest_common_divisor_l4038_403884


namespace NUMINAMATH_CALUDE_last_digit_322_369_l4038_403809

theorem last_digit_322_369 : (322^369) % 10 = 2 := by sorry

end NUMINAMATH_CALUDE_last_digit_322_369_l4038_403809


namespace NUMINAMATH_CALUDE_triangle_property_triangle_area_l4038_403847

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c
  angle_sum : A + B + C = π
  cosine_law : c^2 = a^2 + b^2 - 2*a*b*Real.cos C

theorem triangle_property (t : Triangle) 
  (h : t.c - t.a * Real.cos t.B = (Real.sqrt 2 / 2) * t.b) : 
  t.A = π / 4 := by sorry

theorem triangle_area (t : Triangle) 
  (h1 : t.c - t.a * Real.cos t.B = (Real.sqrt 2 / 2) * t.b)
  (h2 : t.c = 4 * Real.sqrt 2)
  (h3 : Real.cos t.B = 7 * Real.sqrt 2 / 10) : 
  (1 / 2) * t.b * t.c * Real.sin t.A = 2 := by sorry

end NUMINAMATH_CALUDE_triangle_property_triangle_area_l4038_403847


namespace NUMINAMATH_CALUDE_population_growth_percentage_l4038_403876

theorem population_growth_percentage (initial_population final_population : ℕ) 
  (h1 : initial_population = 684)
  (h2 : final_population = 513) :
  ∃ (P : ℝ), 
    (P > 0) ∧ 
    (initial_population : ℝ) * (1 + P / 100) * (1 - 40 / 100) = final_population ∧
    P = 25 := by
  sorry

end NUMINAMATH_CALUDE_population_growth_percentage_l4038_403876
