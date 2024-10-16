import Mathlib

namespace NUMINAMATH_CALUDE_skee_ball_tickets_value_l1098_109863

/-- The number of tickets Luke won playing 'whack a mole' -/
def whack_a_mole_tickets : ℕ := 2

/-- The cost of one candy in tickets -/
def candy_cost : ℕ := 3

/-- The number of candies Luke could buy -/
def candies_bought : ℕ := 5

/-- The number of tickets Luke won playing 'skee ball' -/
def skee_ball_tickets : ℕ := candies_bought * candy_cost - whack_a_mole_tickets

theorem skee_ball_tickets_value : skee_ball_tickets = 13 := by
  sorry

end NUMINAMATH_CALUDE_skee_ball_tickets_value_l1098_109863


namespace NUMINAMATH_CALUDE_optimal_tax_and_revenue_l1098_109884

-- Define the market supply function
def supply_function (P : ℝ) : ℝ := 6 * P - 312

-- Define the market demand function
def demand_function (P : ℝ) (a : ℝ) : ℝ := a - 4 * P

-- Define the tax revenue function
def tax_revenue (t : ℝ) : ℝ := 288 * t - 2.4 * t^2

-- State the theorem
theorem optimal_tax_and_revenue 
  (elasticity_ratio : ℝ) 
  (consumer_price_after_tax : ℝ) 
  (initial_tax_rate : ℝ) :
  elasticity_ratio = 1.5 →
  consumer_price_after_tax = 118 →
  initial_tax_rate = 30 →
  ∃ (optimal_tax : ℝ) (max_revenue : ℝ),
    optimal_tax = 60 ∧
    max_revenue = 8640 ∧
    ∀ (t : ℝ), tax_revenue t ≤ max_revenue :=
by sorry

end NUMINAMATH_CALUDE_optimal_tax_and_revenue_l1098_109884


namespace NUMINAMATH_CALUDE_grade_difference_l1098_109823

theorem grade_difference (a b c : ℕ) : 
  a + b + c = 25 → 
  3 * a + 4 * b + 5 * c = 106 → 
  c - a = 6 := by
  sorry

end NUMINAMATH_CALUDE_grade_difference_l1098_109823


namespace NUMINAMATH_CALUDE_cubic_sum_nonnegative_l1098_109812

theorem cubic_sum_nonnegative (c : ℝ) (X Y : ℝ) 
  (hX : X^2 - c*X - c = 0) 
  (hY : Y^2 - c*Y - c = 0) : 
  X^3 + Y^3 + (X*Y)^3 ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_cubic_sum_nonnegative_l1098_109812


namespace NUMINAMATH_CALUDE_wallet_cost_proof_l1098_109892

/-- The cost of a pair of sneakers -/
def sneaker_cost : ℕ := 100

/-- The cost of a backpack -/
def backpack_cost : ℕ := 100

/-- The cost of a pair of jeans -/
def jeans_cost : ℕ := 50

/-- The total amount spent by Leonard and Michael -/
def total_spent : ℕ := 450

/-- The cost of the wallet -/
def wallet_cost : ℕ := 50

theorem wallet_cost_proof :
  wallet_cost + 2 * sneaker_cost + backpack_cost + 2 * jeans_cost = total_spent :=
by sorry

end NUMINAMATH_CALUDE_wallet_cost_proof_l1098_109892


namespace NUMINAMATH_CALUDE_f_monotone_and_roots_sum_l1098_109841

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2*x - 2*(a+1)*Real.exp x + a*Real.exp (2*x)

theorem f_monotone_and_roots_sum (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x < f a y) ↔ a = 1 ∧
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → f a x₁ = a * Real.exp (2*x₁) → f a x₂ = a * Real.exp (2*x₂) → x₁ + x₂ > 2 :=
sorry

end NUMINAMATH_CALUDE_f_monotone_and_roots_sum_l1098_109841


namespace NUMINAMATH_CALUDE_waiting_time_problem_l1098_109845

/-- Proves that the waiting time for the man to catch up is 25 minutes -/
theorem waiting_time_problem (man_speed woman_speed : ℚ) (stop_time : ℚ) :
  man_speed = 5 →
  woman_speed = 25 →
  stop_time = 5 / 60 →
  let distance_traveled := woman_speed * stop_time
  let catch_up_time := distance_traveled / man_speed
  catch_up_time = 25 / 60 := by sorry

end NUMINAMATH_CALUDE_waiting_time_problem_l1098_109845


namespace NUMINAMATH_CALUDE_arrangement_count_l1098_109880

theorem arrangement_count (n_boys n_girls : ℕ) (h_boys : n_boys = 5) (h_girls : n_girls = 6) :
  (Nat.factorial n_girls) * (Nat.choose (n_girls + 1) n_boys) * (Nat.factorial n_boys) =
  (Nat.factorial n_girls) * 2520 :=
sorry

end NUMINAMATH_CALUDE_arrangement_count_l1098_109880


namespace NUMINAMATH_CALUDE_parabola_properties_l1098_109802

/-- Parabola equation -/
def parabola (x : ℝ) : ℝ := -2 * x^2 + 8 * x - 6

/-- Vertex of the parabola -/
def vertex : ℝ × ℝ := (2, 2)

/-- Axis of symmetry -/
def axis_of_symmetry : ℝ := 2

theorem parabola_properties :
  (∀ x : ℝ, parabola x = -2 * (x - 2)^2 + 2) ∧
  (vertex = (2, 2)) ∧
  (axis_of_symmetry = 2) ∧
  (∀ x : ℝ, x ≥ 2 → ∀ y : ℝ, y > x → parabola y < parabola x) :=
by sorry

end NUMINAMATH_CALUDE_parabola_properties_l1098_109802


namespace NUMINAMATH_CALUDE_scarves_per_yarn_l1098_109873

/-- Given the total number of scarves and yarns, calculate the number of scarves per yarn -/
theorem scarves_per_yarn (total_scarves total_yarns : ℕ) 
  (h1 : total_scarves = 36)
  (h2 : total_yarns = 12) :
  total_scarves / total_yarns = 3 := by
  sorry

#eval 36 / 12  -- This should output 3

end NUMINAMATH_CALUDE_scarves_per_yarn_l1098_109873


namespace NUMINAMATH_CALUDE_solution_equation1_solution_equation2_l1098_109808

-- Define the equations
def equation1 (x : ℝ) : Prop := 2 * (2 * x + 1) - (3 * x - 4) = 2
def equation2 (y : ℝ) : Prop := (3 * y - 1) / 4 - 1 = (5 * y - 7) / 6

-- Theorem statements
theorem solution_equation1 : ∃ x : ℝ, equation1 x ∧ x = -4 := by sorry

theorem solution_equation2 : ∃ y : ℝ, equation2 y ∧ y = -1 := by sorry

end NUMINAMATH_CALUDE_solution_equation1_solution_equation2_l1098_109808


namespace NUMINAMATH_CALUDE_expression_simplification_l1098_109862

theorem expression_simplification (x : ℝ) : 
  2*x - 3*(2 - x) + 4*(2 + x) - 5*(1 - 3*x) = 24*x - 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1098_109862


namespace NUMINAMATH_CALUDE_decimal_equivalent_one_fourth_power_one_l1098_109810

theorem decimal_equivalent_one_fourth_power_one :
  (1 / 4 : ℚ) ^ (1 : ℕ) = 0.25 := by sorry

end NUMINAMATH_CALUDE_decimal_equivalent_one_fourth_power_one_l1098_109810


namespace NUMINAMATH_CALUDE_exponential_inequality_l1098_109888

theorem exponential_inequality (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  (3 : ℝ)^b < (3 : ℝ)^a ∧ (3 : ℝ)^a < (4 : ℝ)^a :=
by sorry

end NUMINAMATH_CALUDE_exponential_inequality_l1098_109888


namespace NUMINAMATH_CALUDE_albaszu_machine_productivity_l1098_109858

-- Define the number of trees cut daily before improvement
def trees_before : ℕ := 16

-- Define the productivity increase factor
def productivity_increase : ℚ := 3/2

-- Define the number of trees cut daily after improvement
def trees_after : ℕ := 25

-- Theorem statement
theorem albaszu_machine_productivity : 
  ↑trees_after = ↑trees_before * productivity_increase :=
by sorry

end NUMINAMATH_CALUDE_albaszu_machine_productivity_l1098_109858


namespace NUMINAMATH_CALUDE_min_value_geometric_sequence_l1098_109889

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem min_value_geometric_sequence (a : ℕ → ℝ) 
    (h_geo : is_geometric_sequence a)
    (h_pos : ∀ n, a n > 0)
    (h_2018 : a 2018 = Real.sqrt 2 / 2) :
    (1 / a 2017 + 2 / a 2019) ≥ 4 ∧ 
    ∃ a, is_geometric_sequence a ∧ (∀ n, a n > 0) ∧ 
         a 2018 = Real.sqrt 2 / 2 ∧ 1 / a 2017 + 2 / a 2019 = 4 :=
by sorry

end NUMINAMATH_CALUDE_min_value_geometric_sequence_l1098_109889


namespace NUMINAMATH_CALUDE_x_squared_plus_x_minus_one_zero_l1098_109866

theorem x_squared_plus_x_minus_one_zero (x : ℝ) :
  x^2 + x - 1 = 0 → x^4 + 2*x^3 - 3*x^2 - 4*x + 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_x_squared_plus_x_minus_one_zero_l1098_109866


namespace NUMINAMATH_CALUDE_number_line_real_bijection_l1098_109872

-- Define the number line as a type
def NumberLine : Type := ℝ

-- Define a point on the number line
def Point : Type := NumberLine

-- State the theorem
theorem number_line_real_bijection : 
  ∃ f : Point → ℝ, Function.Bijective f :=
sorry

end NUMINAMATH_CALUDE_number_line_real_bijection_l1098_109872


namespace NUMINAMATH_CALUDE_flea_jump_angle_rational_l1098_109864

/-- A flea jumping between two intersecting lines --/
structure FleaJump where
  α : ℝ  -- Angle between the lines in radians
  jumpLength : ℝ  -- Length of each jump
  returnsToStart : Prop  -- Flea eventually returns to starting point
  noPreviousPosition : Prop  -- Flea never returns to previous position

/-- Theorem stating that if a flea jumps as described, the angle is rational --/
theorem flea_jump_angle_rational (jump : FleaJump) 
  (h1 : jump.jumpLength = 1)
  (h2 : jump.returnsToStart)
  (h3 : jump.noPreviousPosition) :
  ∃ (p q : ℤ), jump.α = (p / q) * (π / 180) :=
sorry

end NUMINAMATH_CALUDE_flea_jump_angle_rational_l1098_109864


namespace NUMINAMATH_CALUDE_initial_distance_proof_l1098_109875

/-- The distance between two people after a series of movements -/
def final_distance (initial_distance : ℝ) (towards : ℝ) (away : ℝ) : ℝ :=
  initial_distance - towards + away

/-- Theorem: If the final distance is 2000m after moving 200m towards and 1000m away, 
    then the initial distance was 1200m -/
theorem initial_distance_proof (initial_distance : ℝ) :
  final_distance initial_distance 200 1000 = 2000 →
  initial_distance = 1200 := by
  sorry

#check initial_distance_proof

end NUMINAMATH_CALUDE_initial_distance_proof_l1098_109875


namespace NUMINAMATH_CALUDE_complex_square_roots_l1098_109896

theorem complex_square_roots : ∃ (z₁ z₂ : ℂ),
  z₁^2 = -100 - 49*I ∧ 
  z₂^2 = -100 - 49*I ∧ 
  z₁ = (7*Real.sqrt 2)/2 - (7*Real.sqrt 2)/2*I ∧
  z₂ = -(7*Real.sqrt 2)/2 + (7*Real.sqrt 2)/2*I ∧
  ∀ (z : ℂ), z^2 = -100 - 49*I → (z = z₁ ∨ z = z₂) := by
sorry

end NUMINAMATH_CALUDE_complex_square_roots_l1098_109896


namespace NUMINAMATH_CALUDE_quadratic_perfect_square_condition_l1098_109840

theorem quadratic_perfect_square_condition (a b : ℤ) :
  (∃ S : Set ℤ, (Set.Infinite S) ∧ (∀ x ∈ S, ∃ y : ℤ, x^2 + a*x + b = y^2)) ↔ a^2 = 4*b :=
sorry

end NUMINAMATH_CALUDE_quadratic_perfect_square_condition_l1098_109840


namespace NUMINAMATH_CALUDE_difference_of_squares_divisible_by_eight_l1098_109895

theorem difference_of_squares_divisible_by_eight (a b : ℤ) (h : a > b) :
  ∃ k : ℤ, 4 * (a - b) * (a + b + 1) = 8 * k := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_divisible_by_eight_l1098_109895


namespace NUMINAMATH_CALUDE_square_properties_l1098_109839

/-- Given a square with perimeter 48 feet, prove its side length and area. -/
theorem square_properties (perimeter : ℝ) (h : perimeter = 48) :
  ∃ (side_length area : ℝ),
    side_length = 12 ∧
    area = 144 ∧
    perimeter = 4 * side_length ∧
    area = side_length * side_length := by
  sorry

end NUMINAMATH_CALUDE_square_properties_l1098_109839


namespace NUMINAMATH_CALUDE_sum_of_m_and_n_l1098_109807

theorem sum_of_m_and_n (m n : ℕ) (hm : m > 1) (hn : n > 1) 
  (h : 2005^2 + m^2 = 2004^2 + n^2) : m + n = 211 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_m_and_n_l1098_109807


namespace NUMINAMATH_CALUDE_oblique_triangular_prism_volume_l1098_109803

/-- The volume of an oblique triangular prism with specific properties -/
theorem oblique_triangular_prism_volume (a : ℝ) (h : a > 0) :
  let base_area := (a^2 * Real.sqrt 3) / 4
  let height := a * Real.sqrt 3 / 2
  base_area * height = (3 * a^3) / 8 := by
  sorry

#check oblique_triangular_prism_volume

end NUMINAMATH_CALUDE_oblique_triangular_prism_volume_l1098_109803


namespace NUMINAMATH_CALUDE_perpendicular_transitivity_l1098_109818

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between lines and planes
variable (perp : Line → Plane → Prop)

-- Define the parallel relation between lines
variable (parallel : Line → Line → Prop)

-- Theorem statement
theorem perpendicular_transitivity 
  (m n : Line) (α β : Plane) 
  (h1 : perp m β) 
  (h2 : perp n β) 
  (h3 : perp n α) : 
  perp m α :=
sorry

end NUMINAMATH_CALUDE_perpendicular_transitivity_l1098_109818


namespace NUMINAMATH_CALUDE_three_leaf_clover_count_l1098_109801

/-- The number of leaves on a three-leaf clover -/
def three_leaf_count : ℕ := 3

/-- The number of leaves on a four-leaf clover -/
def four_leaf_count : ℕ := 4

/-- The total number of leaves collected -/
def total_leaves : ℕ := 100

/-- The number of four-leaf clovers found -/
def four_leaf_clovers : ℕ := 1

theorem three_leaf_clover_count :
  (total_leaves - four_leaf_count * four_leaf_clovers) / three_leaf_count = 32 := by
  sorry

end NUMINAMATH_CALUDE_three_leaf_clover_count_l1098_109801


namespace NUMINAMATH_CALUDE_min_decimal_digits_l1098_109832

theorem min_decimal_digits (n : ℕ) (d : ℕ) : 
  n = 987654321 ∧ d = 2^30 * 5^6 → 
  (∃ (k : ℕ), k = 30 ∧ 
    ∀ (m : ℕ), (∃ (q r : ℚ), q * 10^m = n / d ∧ r = 0) → m ≥ k) ∧
  (∀ (l : ℕ), l < 30 → 
    ∃ (q r : ℚ), q * 10^l = n / d ∧ r ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_min_decimal_digits_l1098_109832


namespace NUMINAMATH_CALUDE_min_value_theorem_l1098_109855

theorem min_value_theorem (a b c : ℝ) 
  (h : ∀ x y : ℝ, x + 2*y - 3 ≤ a*x + b*y + c ∧ a*x + b*y + c ≤ x + 2*y + 3) :
  ∃ m : ℝ, m = -4 ∧ ∀ k : ℝ, (∃ a' b' c' : ℝ, 
    (∀ x y : ℝ, x + 2*y - 3 ≤ a'*x + b'*y + c' ∧ a'*x + b'*y + c' ≤ x + 2*y + 3) ∧
    k = a' + 2*b' - 3*c') → m ≤ k :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1098_109855


namespace NUMINAMATH_CALUDE_ratio_problem_l1098_109830

theorem ratio_problem (A B C : ℚ) 
  (sum_eq : A + B + C = 98)
  (ratio_bc : B / C = 5 / 8)
  (b_eq : B = 30) :
  A / B = 2 / 3 := by
sorry

end NUMINAMATH_CALUDE_ratio_problem_l1098_109830


namespace NUMINAMATH_CALUDE_age_of_other_man_l1098_109838

/-- Proves that the age of the other replaced man is 20 years old given the problem conditions -/
theorem age_of_other_man (n : ℕ) (avg_increase : ℝ) (age_one_man : ℕ) (avg_age_women : ℝ) : 
  n = 8 ∧ 
  avg_increase = 2 ∧ 
  age_one_man = 22 ∧ 
  avg_age_women = 29 → 
  ∃ (age_other_man : ℕ), 
    age_other_man = 20 ∧ 
    2 * avg_age_women - (age_one_man + age_other_man) = n * avg_increase :=
by sorry

end NUMINAMATH_CALUDE_age_of_other_man_l1098_109838


namespace NUMINAMATH_CALUDE_nested_square_root_value_l1098_109817

theorem nested_square_root_value :
  ∃ x : ℝ, x = Real.sqrt (3 - x) ∧ x = (-1 + Real.sqrt 13) / 2 := by
  sorry

end NUMINAMATH_CALUDE_nested_square_root_value_l1098_109817


namespace NUMINAMATH_CALUDE_no_reciprocal_sum_equals_sum_reciprocals_l1098_109850

theorem no_reciprocal_sum_equals_sum_reciprocals :
  ¬∃ (x y : ℝ), x ≠ -y ∧ x ≠ 0 ∧ y ≠ 0 ∧ (1 / (x + y) = 1 / x + 1 / y) := by
  sorry

end NUMINAMATH_CALUDE_no_reciprocal_sum_equals_sum_reciprocals_l1098_109850


namespace NUMINAMATH_CALUDE_min_distinct_values_l1098_109847

/-- Represents a list of positive integers -/
def IntegerList := List Nat

/-- Checks if a given value is the unique mode of the list occurring exactly n times -/
def isUniqueMode (list : IntegerList) (mode : Nat) (n : Nat) : Prop :=
  (list.count mode = n) ∧ 
  ∀ x, x ≠ mode → list.count x < n

/-- Theorem: The minimum number of distinct values in a list of 2018 positive integers
    with a unique mode occurring exactly 10 times is 225 -/
theorem min_distinct_values (list : IntegerList) (mode : Nat) :
  list.length = 2018 →
  isUniqueMode list mode 10 →
  list.toFinset.card ≥ 225 :=
sorry

end NUMINAMATH_CALUDE_min_distinct_values_l1098_109847


namespace NUMINAMATH_CALUDE_work_completion_time_l1098_109820

/-- The number of days it takes for the original number of people to complete the work -/
def days_to_complete_work (original_people : ℕ) (total_work : ℝ) : ℕ :=
  16

theorem work_completion_time 
  (original_people : ℕ) 
  (total_work : ℝ) 
  (h : (2 * original_people : ℝ) * 4 = total_work / 2) : 
  days_to_complete_work original_people total_work = 16 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l1098_109820


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l1098_109859

theorem sum_of_coefficients (A B : ℝ) : 
  (∀ x : ℝ, x ≠ 2 → A / (x - 2) + B * (x + 3) = (-5 * x^2 + 20 * x + 34) / (x - 2)) →
  A + B = 9 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l1098_109859


namespace NUMINAMATH_CALUDE_square_area_ratio_l1098_109805

theorem square_area_ratio (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_perimeter : 4 * a = 4 * (4 * b)) : a^2 = 16 * b^2 := by
  sorry

end NUMINAMATH_CALUDE_square_area_ratio_l1098_109805


namespace NUMINAMATH_CALUDE_fraction_equality_l1098_109869

theorem fraction_equality : (2 - (1/2) * (1 - 1/4)) / (2 - (1 - 1/3)) = 39/32 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1098_109869


namespace NUMINAMATH_CALUDE_not_sophomores_percentage_l1098_109874

/-- Represents the composition of a high school student body -/
structure StudentBody where
  total : ℕ
  juniors : ℕ
  seniors : ℕ
  freshmen : ℕ
  sophomores : ℕ

/-- Calculates the percentage of students who are not sophomores -/
def percentageNotSophomores (sb : StudentBody) : ℚ :=
  (sb.total - sb.sophomores : ℚ) / sb.total * 100

/-- Theorem stating the percentage of students who are not sophomores -/
theorem not_sophomores_percentage
  (sb : StudentBody)
  (h_total : sb.total = 800)
  (h_juniors : sb.juniors = (23 : ℚ) / 100 * 800)
  (h_seniors : sb.seniors = 160)
  (h_freshmen_sophomores : sb.freshmen = sb.sophomores + 56)
  (h_sum : sb.freshmen + sb.sophomores + sb.juniors + sb.seniors = sb.total) :
  percentageNotSophomores sb = 75 := by
  sorry


end NUMINAMATH_CALUDE_not_sophomores_percentage_l1098_109874


namespace NUMINAMATH_CALUDE_original_number_proof_l1098_109886

theorem original_number_proof (x y : ℝ) : 
  x = 13.0 →
  7 * x + 5 * y = 146 →
  x + y = 24.0 := by
sorry

end NUMINAMATH_CALUDE_original_number_proof_l1098_109886


namespace NUMINAMATH_CALUDE_equation_solution_l1098_109822

theorem equation_solution : ∃ x : ℚ, (2 / 7) * (1 / 4) * x = 8 ∧ x = 112 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1098_109822


namespace NUMINAMATH_CALUDE_ones_digit_of_large_power_l1098_109843

theorem ones_digit_of_large_power : ∃ n : ℕ, n < 10 ∧ 34^(34 * 17^17) ≡ n [ZMOD 10] ∧ n = 6 := by
  sorry

end NUMINAMATH_CALUDE_ones_digit_of_large_power_l1098_109843


namespace NUMINAMATH_CALUDE_x_range_l1098_109856

theorem x_range (x : ℝ) : (x^2 - 4 < 0 ∨ |x| = 2) → x ∈ Set.Icc (-2) 2 := by
  sorry

end NUMINAMATH_CALUDE_x_range_l1098_109856


namespace NUMINAMATH_CALUDE_max_product_under_constraint_l1098_109819

theorem max_product_under_constraint (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 3*a + 2*b = 1) :
  a * b ≤ 1/24 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ 3*a₀ + 2*b₀ = 1 ∧ a₀ * b₀ = 1/24 :=
by sorry

end NUMINAMATH_CALUDE_max_product_under_constraint_l1098_109819


namespace NUMINAMATH_CALUDE_atop_distributive_laws_l1098_109814

-- Define the @ operation
def atop (a b : ℝ) : ℝ := a + 2 * b

-- State the theorem
theorem atop_distributive_laws :
  (∀ x y z : ℝ, x * (atop y z) = atop (x * y) (x * z)) ∧
  (∃ x y z : ℝ, atop x (y * z) ≠ (atop x y) * (atop x z)) ∧
  (∃ x y z : ℝ, atop (atop x y) (atop x z) ≠ atop x (y * z)) := by
  sorry

end NUMINAMATH_CALUDE_atop_distributive_laws_l1098_109814


namespace NUMINAMATH_CALUDE_exactly_one_valid_sequence_of_length_15_l1098_109898

/-- Represents a sequence of A's and B's -/
inductive ABSequence
  | empty : ABSequence
  | cons_a : ABSequence → ABSequence
  | cons_b : ABSequence → ABSequence

/-- Returns true if the given sequence satisfies the run length conditions -/
def valid_sequence (s : ABSequence) : Bool :=
  sorry

/-- Returns the length of the sequence -/
def sequence_length (s : ABSequence) : Nat :=
  sorry

/-- The main theorem to be proved -/
theorem exactly_one_valid_sequence_of_length_15 :
  ∃! (s : ABSequence), valid_sequence s ∧ sequence_length s = 15 :=
  sorry

end NUMINAMATH_CALUDE_exactly_one_valid_sequence_of_length_15_l1098_109898


namespace NUMINAMATH_CALUDE_monotonic_increase_interval_l1098_109829

noncomputable def f (x : ℝ) : ℝ := 3 * x^2 - 2 * Real.log x

theorem monotonic_increase_interval :
  ∀ x : ℝ, x > 0 → (∀ y : ℝ, y > x → f y > f x) ↔ x > Real.sqrt 3 / 3 :=
by sorry

end NUMINAMATH_CALUDE_monotonic_increase_interval_l1098_109829


namespace NUMINAMATH_CALUDE_room_width_calculation_l1098_109860

/-- Given a rectangular room with length 5.5 m, if the cost of paving its floor
    at a rate of 1000 per sq. meter is 20625, then the width of the room is 3.75 m. -/
theorem room_width_calculation (length cost rate : ℝ) (h1 : length = 5.5)
    (h2 : cost = 20625) (h3 : rate = 1000) : 
    cost / rate / length = 3.75 := by
  sorry

end NUMINAMATH_CALUDE_room_width_calculation_l1098_109860


namespace NUMINAMATH_CALUDE_angle_count_in_plane_l1098_109806

/-- Given n points in a plane, this theorem proves the number of 0° and 180° angles formed. -/
theorem angle_count_in_plane (n : ℕ) : 
  let zero_angles := n * (n - 1) * (n - 2) / 3
  let straight_angles := n * (n - 1) * (n - 2) / 6
  let total_angles := n * (n - 1) * (n - 2) / 2
  (zero_angles : ℚ) + (straight_angles : ℚ) = (total_angles : ℚ) :=
by sorry

/-- The total number of angles formed by n points in a plane. -/
def N (n : ℕ) : ℕ := n * (n - 1) * (n - 2) / 2

/-- The number of 0° angles formed by n points in a plane. -/
def zero_angles (n : ℕ) : ℕ := n * (n - 1) * (n - 2) / 3

/-- The number of 180° angles formed by n points in a plane. -/
def straight_angles (n : ℕ) : ℕ := n * (n - 1) * (n - 2) / 6

end NUMINAMATH_CALUDE_angle_count_in_plane_l1098_109806


namespace NUMINAMATH_CALUDE_classroom_average_l1098_109883

theorem classroom_average (class_size : ℕ) (class_avg : ℚ) (two_thirds_avg : ℚ) :
  class_size > 0 →
  class_avg = 55 →
  two_thirds_avg = 60 →
  ∃ (one_third_avg : ℚ),
    (1 : ℚ) / 3 * one_third_avg + (2 : ℚ) / 3 * two_thirds_avg = class_avg ∧
    one_third_avg = 45 :=
by sorry

end NUMINAMATH_CALUDE_classroom_average_l1098_109883


namespace NUMINAMATH_CALUDE_max_volume_after_dilutions_l1098_109851

/-- The maximum volume of a bucket that satisfies the given dilution conditions -/
theorem max_volume_after_dilutions : 
  ∃ (V : ℝ), V > 0 ∧ 
  (V - 10 - 8 * (V - 10) / V) / V ≤ 0.6 ∧
  ∀ (W : ℝ), W > 0 → (W - 10 - 8 * (W - 10) / W) / W ≤ 0.6 → W ≤ V ∧
  V = 40 :=
sorry

end NUMINAMATH_CALUDE_max_volume_after_dilutions_l1098_109851


namespace NUMINAMATH_CALUDE_max_value_of_f_l1098_109816

def f (x : ℝ) : ℝ := x - 5

theorem max_value_of_f :
  ∃ (max : ℝ), max = 8 ∧
  ∀ x : ℝ, -5 ≤ x ∧ x ≤ 13 → f x ≤ max :=
by
  sorry

end NUMINAMATH_CALUDE_max_value_of_f_l1098_109816


namespace NUMINAMATH_CALUDE_partner_calculation_l1098_109813

theorem partner_calculation (x : ℝ) : 4 * (3 * (x + 2) - 2) = 4 * (3 * x + 4) := by
  sorry

#check partner_calculation

end NUMINAMATH_CALUDE_partner_calculation_l1098_109813


namespace NUMINAMATH_CALUDE_power_of_64_l1098_109878

theorem power_of_64 : (64 : ℝ) ^ (3/2) = 512 := by sorry

end NUMINAMATH_CALUDE_power_of_64_l1098_109878


namespace NUMINAMATH_CALUDE_yellow_shirt_pairs_l1098_109876

theorem yellow_shirt_pairs (blue_students : ℕ) (yellow_students : ℕ) (total_students : ℕ) 
  (total_pairs : ℕ) (blue_blue_pairs : ℕ) :
  blue_students = 60 →
  yellow_students = 72 →
  total_students = 132 →
  total_pairs = 66 →
  blue_blue_pairs = 25 →
  ∃ (yellow_yellow_pairs : ℕ), yellow_yellow_pairs = 31 ∧ 
    yellow_yellow_pairs = total_pairs - blue_blue_pairs - (blue_students - 2 * blue_blue_pairs) :=
by
  sorry

end NUMINAMATH_CALUDE_yellow_shirt_pairs_l1098_109876


namespace NUMINAMATH_CALUDE_abc_product_l1098_109861

theorem abc_product (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hab : a * b = 30 * Real.rpow 3 (1/3))
  (hac : a * c = 42 * Real.rpow 3 (1/3))
  (hbc : b * c = 21 * Real.rpow 3 (1/3)) :
  a * b * c = 210 := by
  sorry

end NUMINAMATH_CALUDE_abc_product_l1098_109861


namespace NUMINAMATH_CALUDE_parallel_vectors_y_value_l1098_109800

theorem parallel_vectors_y_value :
  ∀ (y : ℝ),
  let a : Fin 2 → ℝ := ![(-1), 3]
  let b : Fin 2 → ℝ := ![2, y]
  (∃ (k : ℝ), k ≠ 0 ∧ (∀ i, a i = k * b i)) →
  y = -6 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_y_value_l1098_109800


namespace NUMINAMATH_CALUDE_multiples_of_13_in_three_digits_l1098_109882

theorem multiples_of_13_in_three_digits : 
  (Finset.filter (fun n => 100 ≤ n ∧ n ≤ 999 ∧ n % 13 = 0) (Finset.range 1000)).card = 69 := by
  sorry

end NUMINAMATH_CALUDE_multiples_of_13_in_three_digits_l1098_109882


namespace NUMINAMATH_CALUDE_determinant_zero_implies_y_eq_neg_b_l1098_109842

variable (b y : ℝ)

def matrix (b y : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![y + b, y, y],
    ![y, y + b, y],
    ![y, y, y + b]]

theorem determinant_zero_implies_y_eq_neg_b (h1 : b ≠ 0) 
  (h2 : Matrix.det (matrix b y) = 0) : y = -b := by
  sorry

end NUMINAMATH_CALUDE_determinant_zero_implies_y_eq_neg_b_l1098_109842


namespace NUMINAMATH_CALUDE_abs_sum_min_value_l1098_109893

theorem abs_sum_min_value (x : ℚ) : 
  ∃ (min : ℚ), min = 5 ∧ (∀ y : ℚ, |y - 2| + |y + 3| ≥ min) ∧ (|x - 2| + |x + 3| = min) :=
sorry

end NUMINAMATH_CALUDE_abs_sum_min_value_l1098_109893


namespace NUMINAMATH_CALUDE_class_size_calculation_l1098_109881

theorem class_size_calculation (boys_avg : ℝ) (girls_avg : ℝ) (class_avg : ℝ) (boys_girls_diff : ℕ) :
  boys_avg = 73 →
  girls_avg = 77 →
  class_avg = 74 →
  boys_girls_diff = 22 →
  ∃ (total_students : ℕ), total_students = 44 :=
by
  sorry

end NUMINAMATH_CALUDE_class_size_calculation_l1098_109881


namespace NUMINAMATH_CALUDE_fractional_equation_root_l1098_109857

theorem fractional_equation_root (m : ℝ) :
  (∃ x : ℝ, x > 0 ∧ x ≠ 2 ∧ m / (x - 2) + 2 * x / (x - 2) = 1) → m = -4 := by
  sorry

end NUMINAMATH_CALUDE_fractional_equation_root_l1098_109857


namespace NUMINAMATH_CALUDE_profit_maximizing_price_profit_function_increase_current_state_verification_cost_price_verification_l1098_109833

/-- Represents the profit function for a product with given pricing and demand characteristics. -/
def profit_function (x : ℝ) : ℝ :=
  (60 + x - 40) * (300 - 10 * x)

/-- Theorem stating that the profit-maximizing price is 65 yuan. -/
theorem profit_maximizing_price :
  ∃ (max_profit : ℝ), 
    (∀ (x : ℝ), profit_function x ≤ profit_function 5) ∧ 
    (profit_function 5 = max_profit) ∧
    (60 + 5 = 65) := by
  sorry

/-- Verifies that the profit function behaves as expected for price increases. -/
theorem profit_function_increase (x : ℝ) :
  profit_function x = -10 * x^2 + 100 * x + 6000 := by
  sorry

/-- Verifies that the current price and sales volume are consistent with the problem statement. -/
theorem current_state_verification :
  profit_function 0 = (60 - 40) * 300 := by
  sorry

/-- Ensures that the cost price is correctly represented in the profit function. -/
theorem cost_price_verification (x : ℝ) :
  (60 + x - 40) = (profit_function x) / (300 - 10 * x) := by
  sorry

end NUMINAMATH_CALUDE_profit_maximizing_price_profit_function_increase_current_state_verification_cost_price_verification_l1098_109833


namespace NUMINAMATH_CALUDE_range_of_k_l1098_109887

def is_sufficient_condition (k : ℝ) : Prop :=
  ∀ x, x > k → 3 / (x + 1) < 1

def is_not_necessary_condition (k : ℝ) : Prop :=
  ∃ x, 3 / (x + 1) < 1 ∧ x ≤ k

theorem range_of_k : 
  ∀ k, (is_sufficient_condition k ∧ is_not_necessary_condition k) ↔ k ∈ Set.Ici 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_k_l1098_109887


namespace NUMINAMATH_CALUDE_f_derivative_sum_l1098_109890

def f (x : ℝ) := x^4 + x - 1

theorem f_derivative_sum : (deriv f 1) + (deriv f (-1)) = 2 := by sorry

end NUMINAMATH_CALUDE_f_derivative_sum_l1098_109890


namespace NUMINAMATH_CALUDE_num_factors_of_M_l1098_109821

/-- The number of natural-number factors of M, where M = 2^5 · 3^2 · 7^3 · 11^1 -/
def num_factors (M : ℕ) : ℕ :=
  (5 + 1) * (2 + 1) * (3 + 1) * (1 + 1)

/-- M is defined as 2^5 · 3^2 · 7^3 · 11^1 -/
def M : ℕ := 2^5 * 3^2 * 7^3 * 11

theorem num_factors_of_M :
  num_factors M = 144 :=
sorry

end NUMINAMATH_CALUDE_num_factors_of_M_l1098_109821


namespace NUMINAMATH_CALUDE_triangle_side_b_is_4_triangle_area_is_4_sqrt_3_l1098_109849

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  acute : 0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2
  side_angle_relation : a = b * Real.cos C + c * Real.cos B

-- Theorem 1
theorem triangle_side_b_is_4 (abc : Triangle) (h : abc.a - 4 * Real.cos abc.C = abc.c * Real.cos abc.B) :
  abc.b = 4 := by sorry

-- Theorem 2
theorem triangle_area_is_4_sqrt_3 (abc : Triangle)
  (h1 : abc.a - 4 * Real.cos abc.C = abc.c * Real.cos abc.B)
  (h2 : abc.a^2 + abc.b^2 + abc.c^2 = 2 * Real.sqrt 3 * abc.a * abc.b * Real.sin abc.C) :
  abc.a * abc.b * Real.sin abc.C / 2 = 4 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_triangle_side_b_is_4_triangle_area_is_4_sqrt_3_l1098_109849


namespace NUMINAMATH_CALUDE_cost_per_foot_metal_roofing_l1098_109827

/-- Calculates the cost per foot of metal roofing --/
theorem cost_per_foot_metal_roofing (total_required : ℕ) (free_provided : ℕ) (cost_remaining : ℕ) :
  total_required = 300 →
  free_provided = 250 →
  cost_remaining = 400 →
  (cost_remaining : ℚ) / (total_required - free_provided : ℚ) = 8 := by
  sorry

#check cost_per_foot_metal_roofing

end NUMINAMATH_CALUDE_cost_per_foot_metal_roofing_l1098_109827


namespace NUMINAMATH_CALUDE_revenue_fall_percentage_l1098_109894

theorem revenue_fall_percentage (R R' P P' : ℝ) 
  (h1 : P = 0.1 * R)
  (h2 : P' = 0.14 * R')
  (h3 : P' = 0.98 * P) :
  R' = 0.7 * R := by
sorry

end NUMINAMATH_CALUDE_revenue_fall_percentage_l1098_109894


namespace NUMINAMATH_CALUDE_johns_piggy_bank_l1098_109877

theorem johns_piggy_bank (quarters dimes nickels : ℕ) : 
  quarters = 22 →
  dimes = quarters + 3 →
  nickels = quarters - 6 →
  quarters + dimes + nickels = 63 := by
sorry

end NUMINAMATH_CALUDE_johns_piggy_bank_l1098_109877


namespace NUMINAMATH_CALUDE_boat_payment_l1098_109815

theorem boat_payment (total : ℚ) (a b c d e : ℚ) : 
  total = 120 →
  a = (1/3) * (b + c + d + e) →
  b = (1/4) * (a + c + d + e) →
  c = (1/5) * (a + b + d + e) →
  d = 2 * e →
  a + b + c + d + e = total →
  e = 40/3 := by
sorry

end NUMINAMATH_CALUDE_boat_payment_l1098_109815


namespace NUMINAMATH_CALUDE_pie_eating_contest_l1098_109854

theorem pie_eating_contest (first_student second_student : ℚ) 
  (h1 : first_student = 7/8)
  (h2 : second_student = 5/6) :
  first_student - second_student = 1/24 := by
sorry

end NUMINAMATH_CALUDE_pie_eating_contest_l1098_109854


namespace NUMINAMATH_CALUDE_constant_kill_time_l1098_109809

/-- Represents the time taken for lions to kill deers -/
def killTime (numLions : ℕ) : ℕ := 13

/-- The assumption that 13 lions can kill 13 deers in 13 minutes -/
axiom base_case : killTime 13 = 13

/-- Theorem stating that for any number of lions (equal to deers), 
    the time taken to kill all deers is always 13 minutes -/
theorem constant_kill_time (n : ℕ) : killTime n = 13 := by
  sorry

end NUMINAMATH_CALUDE_constant_kill_time_l1098_109809


namespace NUMINAMATH_CALUDE_oreos_and_cookies_problem_l1098_109811

theorem oreos_and_cookies_problem :
  ∀ (oreos cookies : ℕ) (oreo_price cookie_price : ℚ),
    oreos * 9 = cookies * 4 →
    oreo_price = 2 →
    cookie_price = 3 →
    cookies * cookie_price - oreos * oreo_price = 95 →
    oreos + cookies = 65 := by
  sorry

end NUMINAMATH_CALUDE_oreos_and_cookies_problem_l1098_109811


namespace NUMINAMATH_CALUDE_remaining_area_after_cutting_triangles_l1098_109848

/-- The area of a square with side length n -/
def square_area (n : ℕ) : ℕ := n * n

/-- The area of a rectangle with width w and height h -/
def rectangle_area (w h : ℕ) : ℕ := w * h

theorem remaining_area_after_cutting_triangles :
  let total_area := square_area 6
  let dark_gray_area := rectangle_area 1 3
  let light_gray_area := rectangle_area 2 3
  total_area - (dark_gray_area + light_gray_area) = 27 := by
  sorry

end NUMINAMATH_CALUDE_remaining_area_after_cutting_triangles_l1098_109848


namespace NUMINAMATH_CALUDE_equilateral_triangle_condition_l1098_109824

theorem equilateral_triangle_condition (a b c : ℂ) :
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  Complex.abs a = 1 →
  Complex.abs b = 1 →
  Complex.abs c = 1 →
  Complex.abs (a + b - c) ^ 2 + Complex.abs (b + c - a) ^ 2 + Complex.abs (c + a - b) ^ 2 = 12 →
  a + b + c = 0 := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_condition_l1098_109824


namespace NUMINAMATH_CALUDE_quadratic_function_range_l1098_109853

/-- A quadratic function f(x) = a + bx - x^2 -/
def f (a b : ℝ) (x : ℝ) : ℝ := a + b * x - x^2

theorem quadratic_function_range (a b m : ℝ) :
  (∀ x, f a b (1 + x) = f a b (1 - x)) →
  (∀ x ≤ 4, Monotone (fun x ↦ f a b (x + m))) →
  m ≤ -3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_range_l1098_109853


namespace NUMINAMATH_CALUDE_geometric_arithmetic_ratio_l1098_109852

/-- Given a geometric sequence {a_n} with common ratio q ≠ 1,
    if a_4, a_3, a_5 form an arithmetic sequence,
    then (a_3 + a_4) / (a_2 + a_3) = -2 -/
theorem geometric_arithmetic_ratio (a : ℕ → ℝ) (q : ℝ) :
  q ≠ 1 →
  (∀ n : ℕ, a (n + 1) = q * a n) →
  2 * a 3 = a 4 + a 5 →
  (a 3 + a 4) / (a 2 + a 3) = -2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_arithmetic_ratio_l1098_109852


namespace NUMINAMATH_CALUDE_ordering_abc_l1098_109844

theorem ordering_abc (a b c : ℝ) : 
  a = 31/32 → b = Real.cos (1/4) → c = 4 * Real.sin (1/4) → c > b ∧ b > a := by sorry

end NUMINAMATH_CALUDE_ordering_abc_l1098_109844


namespace NUMINAMATH_CALUDE_number_to_add_for_divisibility_l1098_109837

theorem number_to_add_for_divisibility (a b : ℕ) (h : b > 0) : 
  ∃ n : ℕ, (a + n) % b = 0 ∧ n = if a % b = 0 then 0 else b - a % b :=
sorry

end NUMINAMATH_CALUDE_number_to_add_for_divisibility_l1098_109837


namespace NUMINAMATH_CALUDE_sum_interior_angles_formula_l1098_109831

/-- The sum of interior angles of a polygon with n sides -/
def sum_interior_angles (n : ℕ) : ℝ :=
  (n - 2) * 180

/-- Theorem: For a polygon with n sides (where n ≥ 3), 
    the sum of interior angles is (n-2) * 180° -/
theorem sum_interior_angles_formula (n : ℕ) (h : n ≥ 3) : 
  sum_interior_angles n = (n - 2) * 180 := by
  sorry

#check sum_interior_angles_formula

end NUMINAMATH_CALUDE_sum_interior_angles_formula_l1098_109831


namespace NUMINAMATH_CALUDE_curve_T_and_fixed_point_l1098_109891

-- Define the points A, B, C, and O
def A : ℝ × ℝ := (-1, 0)
def B : ℝ × ℝ := (1, 0)
def C : ℝ × ℝ := (0, -1)
def O : ℝ × ℝ := (0, 0)

-- Define the condition for point M
def condition_M (M : ℝ × ℝ) : Prop :=
  let (x, y) := M
  (x + 1) * (x - 1) + y * y = y * (y + 1)

-- Define the curve T
def curve_T (x y : ℝ) : Prop := y = x^2 - 1

-- Define the tangent line at point P
def tangent_line (P : ℝ × ℝ) (x y : ℝ) : Prop :=
  let (x₀, y₀) := P
  y - y₀ = 2 * x₀ * (x - x₀)

-- Define the line y = -5/4
def line_y_eq_neg_5_4 (x y : ℝ) : Prop := y = -5/4

-- Define the circle with diameter PQ
def circle_PQ (P Q H : ℝ × ℝ) : Prop :=
  let (xp, yp) := P
  let (xq, yq) := Q
  let (xh, yh) := H
  (xh - xp) * (xh - xq) + (yh - yp) * (yh - yq) = 0

-- State the theorem
theorem curve_T_and_fixed_point :
  -- Part 1: The trajectory of point M is curve T
  (∀ M : ℝ × ℝ, condition_M M ↔ curve_T M.1 M.2) ∧
  -- Part 2: The circle with diameter PQ passes through a fixed point
  (∀ P : ℝ × ℝ, P.1 ≠ 0 → curve_T P.1 P.2 →
    ∃ Q : ℝ × ℝ,
      tangent_line P Q.1 Q.2 ∧
      line_y_eq_neg_5_4 Q.1 Q.2 ∧
      circle_PQ P Q (0, -3/4)) := by
  sorry

end NUMINAMATH_CALUDE_curve_T_and_fixed_point_l1098_109891


namespace NUMINAMATH_CALUDE_cos_2alpha_problem_l1098_109899

theorem cos_2alpha_problem (α : Real) : 
  (π/2 < α ∧ α < π) →  -- α is in the second quadrant
  (Real.sin α + Real.cos α = Real.sqrt 3 / 3) →
  Real.cos (2 * α) = -(Real.sqrt 5 / 3) := by
sorry

end NUMINAMATH_CALUDE_cos_2alpha_problem_l1098_109899


namespace NUMINAMATH_CALUDE_tan_two_alpha_l1098_109871

theorem tan_two_alpha (α : Real) 
  (h : (Real.sin (Real.pi - α) + Real.sin (Real.pi / 2 - α)) / (Real.sin α - Real.cos α) = 1 / 2) : 
  Real.tan (2 * α) = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_tan_two_alpha_l1098_109871


namespace NUMINAMATH_CALUDE_ten_faucets_fifty_gallons_l1098_109897

/-- The time (in seconds) it takes for a given number of faucets to fill a pool of a given volume. -/
def fill_time (num_faucets : ℕ) (volume : ℝ) : ℝ :=
  sorry

theorem ten_faucets_fifty_gallons
  (h1 : fill_time 5 200 = 15 * 60) -- Five faucets fill 200 gallons in 15 minutes
  (h2 : ∀ (n : ℕ) (v : ℝ), fill_time n v > 0) -- All fill times are positive
  (h3 : ∀ (n m : ℕ) (v : ℝ), n ≠ 0 → m ≠ 0 → fill_time n v * m = fill_time m v * n) -- Faucets dispense water at the same rate
  : fill_time 10 50 = 112.5 := by
  sorry

end NUMINAMATH_CALUDE_ten_faucets_fifty_gallons_l1098_109897


namespace NUMINAMATH_CALUDE_committee_selection_problem_l1098_109834

/-- The number of ways to select a committee under specific constraints -/
def committeeSelections (n : ℕ) (k : ℕ) (pairTogether : Fin n → Fin n → Prop) (pairApart : Fin n → Fin n → Prop) : ℕ :=
  sorry

/-- The specific problem setup -/
theorem committee_selection_problem :
  let n : ℕ := 9
  let k : ℕ := 5
  let a : Fin n := 0
  let b : Fin n := 1
  let c : Fin n := 2
  let d : Fin n := 3
  let pairTogether (i j : Fin n) := (i = a ∧ j = b) ∨ (i = b ∧ j = a)
  let pairApart (i j : Fin n) := (i = c ∧ j = d) ∨ (i = d ∧ j = c)
  committeeSelections n k pairTogether pairApart = 41 :=
sorry

end NUMINAMATH_CALUDE_committee_selection_problem_l1098_109834


namespace NUMINAMATH_CALUDE_y_intercept_of_line_l1098_109828

/-- The y-intercept of the line 3x - 4y = 12 is -3 -/
theorem y_intercept_of_line (x y : ℝ) : 3 * x - 4 * y = 12 → x = 0 → y = -3 := by
  sorry

end NUMINAMATH_CALUDE_y_intercept_of_line_l1098_109828


namespace NUMINAMATH_CALUDE_stratified_sampling_school_a_l1098_109836

theorem stratified_sampling_school_a (school_a : ℕ) (school_b : ℕ) (school_c : ℕ) (sample_size : ℕ) :
  school_a = 3600 →
  school_b = 5400 →
  school_c = 1800 →
  sample_size = 90 →
  (school_a * sample_size) / (school_a + school_b + school_c) = 30 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_school_a_l1098_109836


namespace NUMINAMATH_CALUDE_kenny_trumpet_practice_time_l1098_109825

/-- Given Kenny's activities last week, prove the time he spent practicing trumpet. -/
theorem kenny_trumpet_practice_time :
  ∀ (basketball_time running_time trumpet_time : ℕ),
  basketball_time = 10 →
  running_time = 2 * basketball_time →
  trumpet_time = 2 * running_time →
  trumpet_time = 40 :=
by
  sorry

end NUMINAMATH_CALUDE_kenny_trumpet_practice_time_l1098_109825


namespace NUMINAMATH_CALUDE_floor_sum_equals_126_l1098_109885

theorem floor_sum_equals_126 
  (x y z w : ℝ) 
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z) (pos_w : 0 < w)
  (eq1 : x^2 + y^2 = 2010)
  (eq2 : z^2 + w^2 = 2010)
  (eq3 : x * z = 1008)
  (eq4 : y * w = 1008) :
  ⌊x + y + z + w⌋ = 126 := by
sorry

end NUMINAMATH_CALUDE_floor_sum_equals_126_l1098_109885


namespace NUMINAMATH_CALUDE_shopkeeper_milk_packets_l1098_109867

/-- Proves that the shopkeeper bought 150 packets of milk given the conditions -/
theorem shopkeeper_milk_packets 
  (packet_volume : ℕ) 
  (ounce_to_ml : ℕ) 
  (total_ounces : ℕ) 
  (h1 : packet_volume = 250)
  (h2 : ounce_to_ml = 30)
  (h3 : total_ounces = 1250) :
  (total_ounces * ounce_to_ml) / packet_volume = 150 := by
  sorry

end NUMINAMATH_CALUDE_shopkeeper_milk_packets_l1098_109867


namespace NUMINAMATH_CALUDE_constant_pace_run_time_l1098_109865

/-- Represents the time taken to run a certain distance at a constant pace -/
structure RunTime where
  distance : ℝ
  time : ℝ

/-- Calculates the time taken to run a given distance at a constant pace -/
def calculateTime (pace : ℝ) (distance : ℝ) : ℝ :=
  pace * distance

theorem constant_pace_run_time 
  (store_run : RunTime) 
  (friend_house_distance : ℝ) 
  (h1 : store_run.distance = 3) 
  (h2 : store_run.time = 24) 
  (h3 : friend_house_distance = 1.5) :
  calculateTime (store_run.time / store_run.distance) friend_house_distance = 12 := by
  sorry

#check constant_pace_run_time

end NUMINAMATH_CALUDE_constant_pace_run_time_l1098_109865


namespace NUMINAMATH_CALUDE_blocks_used_proof_l1098_109870

/-- The number of blocks Randy used to build a tower -/
def tower_blocks : ℕ := 27

/-- The number of blocks Randy used to build a house -/
def house_blocks : ℕ := 53

/-- The total number of blocks Randy used for both the tower and the house -/
def total_blocks : ℕ := tower_blocks + house_blocks

theorem blocks_used_proof : total_blocks = 80 := by
  sorry

end NUMINAMATH_CALUDE_blocks_used_proof_l1098_109870


namespace NUMINAMATH_CALUDE_grunters_lineup_count_l1098_109879

/-- Represents the number of players in each position --/
structure TeamComposition where
  guards : ℕ
  forwards : ℕ
  centers : ℕ

/-- Represents the starting lineup requirements --/
structure LineupRequirements where
  total_starters : ℕ
  guards : ℕ
  forwards : ℕ
  centers : ℕ

/-- Calculates the number of possible lineups --/
def calculate_lineups (team : TeamComposition) (req : LineupRequirements) : ℕ :=
  (team.guards.choose req.guards) * (team.forwards.choose req.forwards) * (team.centers.choose req.centers)

theorem grunters_lineup_count :
  let team : TeamComposition := ⟨5, 6, 3⟩  -- 4+1 guards, 5+1 forwards, 3 centers
  let req : LineupRequirements := ⟨5, 2, 2, 1⟩  -- 5 total, 2 guards, 2 forwards, 1 center
  calculate_lineups team req = 60 := by
  sorry

#check grunters_lineup_count

end NUMINAMATH_CALUDE_grunters_lineup_count_l1098_109879


namespace NUMINAMATH_CALUDE_bottle_production_l1098_109868

/-- Given that 6 identical machines produce 420 bottles per minute at a constant rate,
    prove that 10 such machines will produce 2800 bottles in 4 minutes. -/
theorem bottle_production
  (machines : ℕ → ℕ) -- Function mapping number of machines to bottles produced per minute
  (h1 : machines 6 = 420) -- 6 machines produce 420 bottles per minute
  (h2 : ∀ n : ℕ, machines n = n * (machines 1)) -- Constant rate production
  : machines 10 * 4 = 2800 := by
  sorry


end NUMINAMATH_CALUDE_bottle_production_l1098_109868


namespace NUMINAMATH_CALUDE_fibonacci_like_invariant_l1098_109835

def fibonacci_like_sequence (u : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, u (n + 2) = u n + u (n + 1)

theorem fibonacci_like_invariant (u : ℕ → ℤ) (h : fibonacci_like_sequence u) :
  ∃ c : ℕ, ∀ n : ℕ, n ≥ 1 → |u (n - 1) * u (n + 2) - u n * u (n + 1)| = c :=
sorry

end NUMINAMATH_CALUDE_fibonacci_like_invariant_l1098_109835


namespace NUMINAMATH_CALUDE_square_area_is_121_l1098_109826

/-- A square in a 2D coordinate system --/
structure Square where
  x : ℝ
  y : ℝ

/-- The area of a square --/
def square_area (s : Square) : ℝ :=
  (20 - 9) ^ 2

/-- Theorem: The area of the given square is 121 square units --/
theorem square_area_is_121 (s : Square) : square_area s = 121 := by
  sorry

end NUMINAMATH_CALUDE_square_area_is_121_l1098_109826


namespace NUMINAMATH_CALUDE_carly_to_lisa_jeans_ratio_l1098_109804

/-- Represents the spending of a person on different items --/
structure Spending :=
  (tshirts : ℚ)
  (jeans : ℚ)
  (coats : ℚ)

/-- Calculate the total spending of a person --/
def totalSpending (s : Spending) : ℚ :=
  s.tshirts + s.jeans + s.coats

/-- Lisa's spending based on the given conditions --/
def lisa : Spending :=
  { tshirts := 40
  , jeans := 40 / 2
  , coats := 40 * 2 }

/-- Carly's spending based on the given conditions --/
def carly : Spending :=
  { tshirts := lisa.tshirts / 4
  , jeans := lisa.jeans * (230 - totalSpending lisa - (lisa.tshirts / 4) - (lisa.coats / 4)) / lisa.jeans
  , coats := lisa.coats / 4 }

/-- The main theorem to prove --/
theorem carly_to_lisa_jeans_ratio :
  carly.jeans / lisa.jeans = 3 := by sorry

end NUMINAMATH_CALUDE_carly_to_lisa_jeans_ratio_l1098_109804


namespace NUMINAMATH_CALUDE_arithmetic_mean_relation_l1098_109846

theorem arithmetic_mean_relation (a b x : ℝ) : 
  (2 * x = a + b) → 
  (2 * x^2 = a^2 - b^2) → 
  (a = -b ∨ a = 3*b) := by
sorry

end NUMINAMATH_CALUDE_arithmetic_mean_relation_l1098_109846
