import Mathlib

namespace NUMINAMATH_CALUDE_business_income_calculation_l2975_297505

theorem business_income_calculation 
  (spending income : ℕ) 
  (spending_income_ratio : spending * 9 = income * 5) 
  (profit : ℕ) 
  (profit_equation : profit = income - spending) 
  (profit_value : profit = 48000) : income = 108000 := by
sorry

end NUMINAMATH_CALUDE_business_income_calculation_l2975_297505


namespace NUMINAMATH_CALUDE_negation_equivalence_l2975_297597

theorem negation_equivalence :
  (¬ ∃ x₀ > 2, x₀^3 - 2*x₀^2 < 0) ↔ (∀ x > 2, x^3 - 2*x^2 ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2975_297597


namespace NUMINAMATH_CALUDE_unique_solution_for_equation_l2975_297535

/-- Represents a three-digit number ABC --/
def three_digit_number (a b c : ℕ) : ℕ := 100 * a + 10 * b + c

/-- Represents a two-digit number AB --/
def two_digit_number (a b : ℕ) : ℕ := 10 * a + b

/-- Predicate to check if a number is a single digit --/
def is_single_digit (n : ℕ) : Prop := n < 10

/-- Predicate to check if four numbers are distinct --/
def are_distinct (a b c d : ℕ) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

theorem unique_solution_for_equation :
  ∃! (a b c d : ℕ),
    is_single_digit a ∧
    is_single_digit b ∧
    is_single_digit c ∧
    is_single_digit d ∧
    are_distinct a b c d ∧
    three_digit_number a b c * two_digit_number a b + c * d = 2017 ∧
    two_digit_number a b = 14 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_for_equation_l2975_297535


namespace NUMINAMATH_CALUDE_quadratic_inequality_implies_not_roots_l2975_297500

theorem quadratic_inequality_implies_not_roots (a b x : ℝ) :
  x^2 - (a + b)*x + a*b ≠ 0 → ¬(x = a ∨ x = b) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_implies_not_roots_l2975_297500


namespace NUMINAMATH_CALUDE_omega_set_classification_l2975_297578

-- Define the concept of an Ω set
def is_omega_set (M : Set (ℝ × ℝ)) : Prop :=
  ∀ (p₁ : ℝ × ℝ), p₁ ∈ M → ∃ (p₂ : ℝ × ℝ), p₂ ∈ M ∧ p₁.1 * p₂.1 + p₁.2 * p₂.2 = 0

-- Define the sets
def set1 : Set (ℝ × ℝ) := {p | p.2 = 1 / p.1}
def set2 : Set (ℝ × ℝ) := {p | p.2 = (p.1 - 1) / Real.exp p.1}
def set3 : Set (ℝ × ℝ) := {p | p.2 = Real.sqrt (1 - p.1^2)}
def set4 : Set (ℝ × ℝ) := {p | p.2 = p.1^2 - 2*p.1 + 2}
def set5 : Set (ℝ × ℝ) := {p | p.2 = Real.cos p.1 + Real.sin p.1}

-- State the theorem
theorem omega_set_classification :
  (¬ is_omega_set set1) ∧
  (is_omega_set set2) ∧
  (is_omega_set set3) ∧
  (¬ is_omega_set set4) ∧
  (is_omega_set set5) := by
  sorry

end NUMINAMATH_CALUDE_omega_set_classification_l2975_297578


namespace NUMINAMATH_CALUDE_weighted_average_is_correct_l2975_297591

/-- Represents the number of pens sold for each type -/
def pens_sold : Fin 2 → ℕ
  | 0 => 100  -- Type A
  | 1 => 200  -- Type B

/-- Represents the number of pens gained for each type -/
def pens_gained : Fin 2 → ℕ
  | 0 => 30   -- Type A
  | 1 => 40   -- Type B

/-- Calculates the gain percentage for each pen type -/
def gain_percentage (i : Fin 2) : ℚ :=
  (pens_gained i : ℚ) / (pens_sold i : ℚ) * 100

/-- Calculates the weighted average of gain percentages -/
def weighted_average : ℚ :=
  (gain_percentage 0 * pens_sold 0 + gain_percentage 1 * pens_sold 1) / (pens_sold 0 + pens_sold 1)

theorem weighted_average_is_correct :
  weighted_average = 7000 / 300 :=
sorry

end NUMINAMATH_CALUDE_weighted_average_is_correct_l2975_297591


namespace NUMINAMATH_CALUDE_spherical_coord_transformation_l2975_297569

/-- Given a point with rectangular coordinates (a, b, c) and 
    spherical coordinates (3, 3π/4, π/6), prove that the point 
    with rectangular coordinates (a, -b, c) has spherical 
    coordinates (3, 7π/4, π/6) -/
theorem spherical_coord_transformation 
  (a b c : ℝ) 
  (h1 : a = 3 * Real.sin (π/6) * Real.cos (3*π/4))
  (h2 : b = 3 * Real.sin (π/6) * Real.sin (3*π/4))
  (h3 : c = 3 * Real.cos (π/6)) :
  ∃ (ρ θ φ : ℝ), 
    ρ = 3 ∧ 
    θ = 7*π/4 ∧ 
    φ = π/6 ∧
    a = ρ * Real.sin φ * Real.cos θ ∧
    -b = ρ * Real.sin φ * Real.sin θ ∧
    c = ρ * Real.cos φ :=
by sorry

end NUMINAMATH_CALUDE_spherical_coord_transformation_l2975_297569


namespace NUMINAMATH_CALUDE_inequality_system_subset_circle_l2975_297575

theorem inequality_system_subset_circle (m : ℝ) :
  m > 0 →
  (∀ x y : ℝ, x - 2*y + 5 ≥ 0 ∧ 3 - x ≥ 0 ∧ x + y ≥ 0 → x^2 + y^2 ≤ m^2) →
  m ≥ 3 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_inequality_system_subset_circle_l2975_297575


namespace NUMINAMATH_CALUDE_fruit_purchase_cost_l2975_297513

/-- Given information about oranges and apples, calculates the total cost of a specific purchase. -/
theorem fruit_purchase_cost 
  (orange_bags : ℕ) (orange_weight : ℝ) (apple_bags : ℕ) (apple_weight : ℝ)
  (orange_price : ℝ) (apple_price : ℝ)
  (h_orange : orange_bags * (orange_weight / orange_bags) = 24)
  (h_apple : apple_bags * (apple_weight / apple_bags) = 30)
  (h_orange_price : orange_price = 1.5)
  (h_apple_price : apple_price = 2) :
  5 * (orange_weight / orange_bags) * orange_price + 
  4 * (apple_weight / apple_bags) * apple_price = 45 := by
  sorry


end NUMINAMATH_CALUDE_fruit_purchase_cost_l2975_297513


namespace NUMINAMATH_CALUDE_somu_age_problem_l2975_297593

/-- Somu's age problem -/
theorem somu_age_problem (somu_age : ℕ) (father_age : ℕ) (years_back : ℕ) :
  somu_age = 18 →
  somu_age = father_age / 3 →
  somu_age - years_back = (father_age - years_back) / 5 →
  years_back = 9 := by
  sorry

end NUMINAMATH_CALUDE_somu_age_problem_l2975_297593


namespace NUMINAMATH_CALUDE_complement_of_union_equals_singleton_five_l2975_297551

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4, 5}

-- Define set M
def M : Set Nat := {1, 2}

-- Define set N
def N : Set Nat := {3, 4}

-- Theorem statement
theorem complement_of_union_equals_singleton_five :
  (U \ (M ∪ N)) = {5} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_union_equals_singleton_five_l2975_297551


namespace NUMINAMATH_CALUDE_parabola_vertex_condition_l2975_297556

/-- A parabola with equation y = x^2 + 2x + a -/
structure Parabola where
  a : ℝ

/-- The vertex of a parabola y = x^2 + 2x + a is below the x-axis -/
def vertex_below_x_axis (p : Parabola) : Prop :=
  let x := -1  -- x-coordinate of the vertex
  let y := x^2 + 2*x + p.a  -- y-coordinate of the vertex
  y < 0

/-- If the vertex of the parabola y = x^2 + 2x + a is below the x-axis, then a < 1 -/
theorem parabola_vertex_condition (p : Parabola) : vertex_below_x_axis p → p.a < 1 := by
  sorry

end NUMINAMATH_CALUDE_parabola_vertex_condition_l2975_297556


namespace NUMINAMATH_CALUDE_mountain_paths_theorem_l2975_297516

/-- Number of paths to the mountain top -/
def num_paths : ℕ := 5

/-- Number of people ascending and descending -/
def num_people : ℕ := 2

/-- Calculates the number of ways to ascend and descend the mountain for scenario a -/
def scenario_a (n p : ℕ) : ℕ := 
  Nat.choose n p * Nat.choose (n - p) p

/-- Calculates the number of ways to ascend and descend the mountain for scenario b -/
def scenario_b (n p : ℕ) : ℕ := 
  Nat.choose n p * Nat.choose n p

/-- Calculates the number of ways to ascend and descend the mountain for scenario c -/
def scenario_c (n p : ℕ) : ℕ := 
  (n ^ p) * (n ^ p)

/-- Calculates the number of ways to ascend and descend the mountain for scenario d -/
def scenario_d (n p : ℕ) : ℕ := 
  (Nat.factorial n / Nat.factorial (n - p)) * (Nat.factorial (n - p) / Nat.factorial (n - 2*p))

/-- Calculates the number of ways to ascend and descend the mountain for scenario e -/
def scenario_e (n p : ℕ) : ℕ := 
  (Nat.factorial n / Nat.factorial (n - p)) * (Nat.factorial n / Nat.factorial (n - p))

/-- Calculates the number of ways to ascend and descend the mountain for scenario f -/
def scenario_f (n p : ℕ) : ℕ := 
  (n ^ p) * (n ^ p)

theorem mountain_paths_theorem :
  scenario_a num_paths num_people = 30 ∧
  scenario_b num_paths num_people = 100 ∧
  scenario_c num_paths num_people = 625 ∧
  scenario_d num_paths num_people = 120 ∧
  scenario_e num_paths num_people = 400 ∧
  scenario_f num_paths num_people = 625 :=
by sorry

end NUMINAMATH_CALUDE_mountain_paths_theorem_l2975_297516


namespace NUMINAMATH_CALUDE_speed_equivalence_l2975_297533

/-- Conversion factor from m/s to km/h -/
def mps_to_kmph : ℝ := 3.6

/-- The given speed in meters per second -/
def speed_mps : ℝ := 115.00919999999999

/-- The calculated speed in kilometers per hour -/
def speed_kmph : ℝ := 414.03312

/-- Theorem stating that the given speed in m/s is equivalent to the calculated speed in km/h -/
theorem speed_equivalence : speed_mps * mps_to_kmph = speed_kmph := by
  sorry

end NUMINAMATH_CALUDE_speed_equivalence_l2975_297533


namespace NUMINAMATH_CALUDE_range_of_a_l2975_297594

/-- The piecewise function f(x) as defined in the problem -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ a then |Real.log x| else -(x - 3*a + 1)^2 + (2*a - 1)^2 + a

/-- The function g(x) defined as f(x) - b -/
noncomputable def g (a b : ℝ) (x : ℝ) : ℝ := f a x - b

/-- The theorem stating the range of a given the conditions -/
theorem range_of_a (a : ℝ) :
  (∃ b : ℝ, b > 0 ∧ (∃ x₁ x₂ x₃ x₄ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄ ∧
    g a b x₁ = 0 ∧ g a b x₂ = 0 ∧ g a b x₃ = 0 ∧ g a b x₄ = 0)) →
  0 < a ∧ a < 1/2 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l2975_297594


namespace NUMINAMATH_CALUDE_sin_negative_31pi_over_6_l2975_297553

theorem sin_negative_31pi_over_6 : Real.sin (-31 * Real.pi / 6) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_negative_31pi_over_6_l2975_297553


namespace NUMINAMATH_CALUDE_triangle_inequality_l2975_297530

theorem triangle_inequality (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ b + c > a ∧ c + a > b) :
  a^3 + b^3 + c^3 ≤ (a + b + c) * (a*b + b*c + c*a) := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l2975_297530


namespace NUMINAMATH_CALUDE_cos_product_pi_ninths_l2975_297538

theorem cos_product_pi_ninths : 
  Real.cos (π / 9) * Real.cos (2 * π / 9) * Real.cos (4 * π / 9) = 1 / 8 := by
  sorry

end NUMINAMATH_CALUDE_cos_product_pi_ninths_l2975_297538


namespace NUMINAMATH_CALUDE_matrix_equality_l2975_297557

theorem matrix_equality (C D : Matrix (Fin 2) (Fin 2) ℝ) 
  (h1 : C + D = C * D) 
  (h2 : C * D = ![![5, 2], ![-3, 6]]) : 
  D * C = ![![5, 2], ![-3, 6]] := by
sorry

end NUMINAMATH_CALUDE_matrix_equality_l2975_297557


namespace NUMINAMATH_CALUDE_chess_club_election_l2975_297526

theorem chess_club_election (total_candidates : ℕ) (officer_positions : ℕ) (past_officers : ℕ) :
  total_candidates = 20 →
  officer_positions = 6 →
  past_officers = 8 →
  (Nat.choose total_candidates officer_positions - 
   Nat.choose (total_candidates - past_officers) officer_positions) = 37836 :=
by sorry

end NUMINAMATH_CALUDE_chess_club_election_l2975_297526


namespace NUMINAMATH_CALUDE_new_average_income_l2975_297581

/-- Given a family with 3 earning members and an average monthly income,
    calculate the new average income after one member passes away. -/
theorem new_average_income
  (initial_members : ℕ)
  (initial_average : ℚ)
  (deceased_income : ℚ)
  (h1 : initial_members = 3)
  (h2 : initial_average = 735)
  (h3 : deceased_income = 905) :
  let total_income := initial_members * initial_average
  let remaining_income := total_income - deceased_income
  let remaining_members := initial_members - 1
  remaining_income / remaining_members = 650 := by
sorry

end NUMINAMATH_CALUDE_new_average_income_l2975_297581


namespace NUMINAMATH_CALUDE_sarah_ate_jawbreakers_l2975_297522

def package_size : ℕ := 8
def jawbreakers_left : ℕ := 4

theorem sarah_ate_jawbreakers : 
  package_size - jawbreakers_left = 4 := by
  sorry

end NUMINAMATH_CALUDE_sarah_ate_jawbreakers_l2975_297522


namespace NUMINAMATH_CALUDE_least_prime_factor_of_9_4_minus_9_3_l2975_297547

theorem least_prime_factor_of_9_4_minus_9_3 :
  Nat.minFac (9^4 - 9^3) = 2 := by sorry

end NUMINAMATH_CALUDE_least_prime_factor_of_9_4_minus_9_3_l2975_297547


namespace NUMINAMATH_CALUDE_area_right_triangle_45_deg_l2975_297543

theorem area_right_triangle_45_deg (a : ℝ) (h1 : a = 8) (h2 : a > 0) : 
  (1 / 2 : ℝ) * a * a = 32 := by
sorry

end NUMINAMATH_CALUDE_area_right_triangle_45_deg_l2975_297543


namespace NUMINAMATH_CALUDE_phillips_money_l2975_297574

/-- The amount of money Phillip's mother gave him -/
def total_money : ℕ := sorry

/-- The amount Phillip spent on oranges -/
def oranges_cost : ℕ := 14

/-- The amount Phillip spent on apples -/
def apples_cost : ℕ := 25

/-- The amount Phillip spent on candy -/
def candy_cost : ℕ := 6

/-- The amount Phillip has left -/
def money_left : ℕ := 50

/-- Theorem stating that the total money given by Phillip's mother
    is equal to the sum of his expenses plus the amount left -/
theorem phillips_money :
  total_money = oranges_cost + apples_cost + candy_cost + money_left :=
sorry

end NUMINAMATH_CALUDE_phillips_money_l2975_297574


namespace NUMINAMATH_CALUDE_problem_solution_l2975_297596

theorem problem_solution (x : ℝ) 
  (h : x * Real.sqrt (x^2 - 1) + 1 / (x + Real.sqrt (x^2 - 1)) = 21) :
  x^2 * Real.sqrt (x^4 - 1) + 1 / (x^2 + Real.sqrt (x^4 - 1)) = 2 * x^2 :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l2975_297596


namespace NUMINAMATH_CALUDE_probability_one_of_each_color_is_9_28_l2975_297572

def total_balls : ℕ := 9
def balls_per_color : ℕ := 3
def selected_balls : ℕ := 3

def probability_one_of_each_color : ℚ :=
  (balls_per_color ^ 3 : ℚ) / (total_balls.choose selected_balls)

theorem probability_one_of_each_color_is_9_28 :
  probability_one_of_each_color = 9 / 28 := by
  sorry

end NUMINAMATH_CALUDE_probability_one_of_each_color_is_9_28_l2975_297572


namespace NUMINAMATH_CALUDE_invalid_vote_percentage_l2975_297565

theorem invalid_vote_percentage
  (total_votes : ℕ)
  (candidate_a_percentage : ℝ)
  (candidate_a_votes : ℕ)
  (h1 : total_votes = 560000)
  (h2 : candidate_a_percentage = 0.85)
  (h3 : candidate_a_votes = 404600) :
  (1 - (candidate_a_votes : ℝ) / (candidate_a_percentage * total_votes)) * 100 = 15 := by
  sorry

end NUMINAMATH_CALUDE_invalid_vote_percentage_l2975_297565


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l2975_297521

/-- Given a data set (2, 4, 6, 8) with median m and variance n, 
    and the equation ma + nb = 1 where a > 0 and b > 0,
    prove that the minimum value of 1/a + 1/b is 20. -/
theorem min_value_reciprocal_sum (m n a b : ℝ) : 
  m = 5 → 
  n = 5 → 
  m * a + n * b = 1 → 
  a > 0 → 
  b > 0 → 
  (1 / a + 1 / b) ≥ 20 := by
  sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l2975_297521


namespace NUMINAMATH_CALUDE_line_slope_l2975_297529

theorem line_slope (x y : ℝ) : 3 * y + 2 = -4 * x - 9 → (y - (-11/3)) / (x - 0) = -4/3 := by
  sorry

end NUMINAMATH_CALUDE_line_slope_l2975_297529


namespace NUMINAMATH_CALUDE_f_properties_l2975_297534

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then 2 * x + a else -x - 2 * a

theorem f_properties (a : ℝ) (h : a ≠ 0) :
  (f (-3) 10 = -4 ∧ f (-3) (f (-3) 10) = -11) ∧
  (∀ b : ℝ, b ≠ 0 → (f b (1 - b) = f b (1 + b) ↔ b = -3/4)) := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l2975_297534


namespace NUMINAMATH_CALUDE_sum_first_ten_primes_with_units_digit_3_l2975_297583

def is_prime (n : ℕ) : Prop := sorry

def has_units_digit_3 (n : ℕ) : Prop := n % 10 = 3

def first_ten_primes_with_units_digit_3 : List ℕ := sorry

theorem sum_first_ten_primes_with_units_digit_3 :
  (first_ten_primes_with_units_digit_3.foldl (· + ·) 0) = 793 := by sorry

end NUMINAMATH_CALUDE_sum_first_ten_primes_with_units_digit_3_l2975_297583


namespace NUMINAMATH_CALUDE_subcommittee_formation_count_l2975_297570

theorem subcommittee_formation_count : 
  let total_republicans : ℕ := 10
  let total_democrats : ℕ := 7
  let subcommittee_republicans : ℕ := 4
  let subcommittee_democrats : ℕ := 3
  (Nat.choose total_republicans subcommittee_republicans) * 
  (Nat.choose total_democrats subcommittee_democrats) = 7350 := by
  sorry

end NUMINAMATH_CALUDE_subcommittee_formation_count_l2975_297570


namespace NUMINAMATH_CALUDE_rain_probability_l2975_297587

theorem rain_probability (p : ℚ) (n : ℕ) (hp : p = 3/5) (hn : n = 5) :
  1 - (1 - p)^n = 3093/3125 := by
  sorry

end NUMINAMATH_CALUDE_rain_probability_l2975_297587


namespace NUMINAMATH_CALUDE_cos_two_pi_thirds_l2975_297517

theorem cos_two_pi_thirds : Real.cos (2 * Real.pi / 3) = -(1 / 2) := by
  sorry

end NUMINAMATH_CALUDE_cos_two_pi_thirds_l2975_297517


namespace NUMINAMATH_CALUDE_prob_angle_AQB_obtuse_l2975_297571

/-- Pentagon ABCDE with vertices A(0,3), B(5,0), C(2π,0), D(2π,5), E(0,5) -/
def pentagon : Set (ℝ × ℝ) :=
  {p | p.1 ≥ 0 ∧ p.1 ≤ 2*Real.pi ∧ p.2 ≥ 0 ∧ p.2 ≤ 5 ∧
       (p.2 ≥ 3 - 3/5 * p.1 ∨ p.1 ≥ 2*Real.pi)}

/-- Point A -/
def A : ℝ × ℝ := (0, 3)

/-- Point B -/
def B : ℝ × ℝ := (5, 0)

/-- Random point Q in the pentagon -/
def Q : ℝ × ℝ := sorry

/-- Angle AQB -/
def angle_AQB : ℝ := sorry

/-- Probability measure on the pentagon -/
def prob : MeasureTheory.Measure (ℝ × ℝ) := sorry

/-- The probability that angle AQB is obtuse -/
theorem prob_angle_AQB_obtuse :
  prob {q ∈ pentagon | angle_AQB > Real.pi/2} / prob pentagon = 17/128 := by sorry

end NUMINAMATH_CALUDE_prob_angle_AQB_obtuse_l2975_297571


namespace NUMINAMATH_CALUDE_martha_lasagna_cost_l2975_297542

/-- The cost of ingredients for Martha's lasagna --/
theorem martha_lasagna_cost : 
  let cheese_weight : ℝ := 1.5
  let meat_weight : ℝ := 0.5
  let cheese_price : ℝ := 6
  let meat_price : ℝ := 8
  cheese_weight * cheese_price + meat_weight * meat_price = 13 := by
  sorry

end NUMINAMATH_CALUDE_martha_lasagna_cost_l2975_297542


namespace NUMINAMATH_CALUDE_parabola_coefficient_b_l2975_297599

/-- Given a parabola y = ax^2 + bx + c with vertex (p, p) and y-intercept (0, 2p), where p ≠ 0,
    the coefficient b is equal to -2. -/
theorem parabola_coefficient_b (a b c p : ℝ) : p ≠ 0 →
  (∀ x, a * x^2 + b * x + c = (x - p)^2 / p + p) →
  a * 0^2 + b * 0 + c = 2 * p →
  b = -2 := by sorry

end NUMINAMATH_CALUDE_parabola_coefficient_b_l2975_297599


namespace NUMINAMATH_CALUDE_friend_meeting_distance_l2975_297559

theorem friend_meeting_distance (trail_length : ℝ) (rate_difference : ℝ) : 
  trail_length = 36 → rate_difference = 0.25 → 
  let faster_friend_distance : ℝ := trail_length * (1 + rate_difference) / (2 + rate_difference)
  faster_friend_distance = 20 := by
sorry

end NUMINAMATH_CALUDE_friend_meeting_distance_l2975_297559


namespace NUMINAMATH_CALUDE_pie_chart_most_appropriate_for_percentages_l2975_297520

/-- Represents different types of charts --/
inductive ChartType
| PieChart
| LineChart
| BarChart

/-- Represents characteristics of data --/
structure DataCharacteristics where
  is_percentage : Bool
  total_is_100_percent : Bool
  part_whole_relationship_important : Bool

/-- Determines the most appropriate chart type based on data characteristics --/
def most_appropriate_chart (data : DataCharacteristics) : ChartType :=
  if data.is_percentage ∧ data.total_is_100_percent ∧ data.part_whole_relationship_important then
    ChartType.PieChart
  else
    ChartType.BarChart

/-- Theorem stating that a pie chart is most appropriate for percentage data summing to 100% 
    where the part-whole relationship is important --/
theorem pie_chart_most_appropriate_for_percentages 
  (data : DataCharacteristics) 
  (h1 : data.is_percentage = true) 
  (h2 : data.total_is_100_percent = true)
  (h3 : data.part_whole_relationship_important = true) : 
  most_appropriate_chart data = ChartType.PieChart :=
by
  sorry

#check pie_chart_most_appropriate_for_percentages

end NUMINAMATH_CALUDE_pie_chart_most_appropriate_for_percentages_l2975_297520


namespace NUMINAMATH_CALUDE_other_communities_count_l2975_297527

theorem other_communities_count (total : ℕ) (muslim_percent hindu_percent sikh_percent : ℚ) : 
  total = 850 →
  muslim_percent = 44 / 100 →
  hindu_percent = 28 / 100 →
  sikh_percent = 10 / 100 →
  ↑total * (1 - (muslim_percent + hindu_percent + sikh_percent)) = 153 := by
  sorry

end NUMINAMATH_CALUDE_other_communities_count_l2975_297527


namespace NUMINAMATH_CALUDE_fraction_sum_positive_l2975_297539

theorem fraction_sum_positive (a b c : ℝ) (h1 : a > b) (h2 : b > c) :
  (1 / (a - b) + 1 / (b - c) + 1 / (c - a)) > 0 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_positive_l2975_297539


namespace NUMINAMATH_CALUDE_factorization_proof_l2975_297519

theorem factorization_proof (a : ℝ) : 2*a - 2*a^3 = 2*a*(1+a)*(1-a) := by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l2975_297519


namespace NUMINAMATH_CALUDE_book_page_digits_l2975_297562

/-- The total number of digits used to number pages in a book -/
def totalDigits (n : ℕ) : ℕ :=
  (min n 9) +
  2 * (min n 99 - min n 9) +
  3 * (n - min n 99)

/-- Theorem: The total number of digits used in numbering the pages of a book with 366 pages is 990 -/
theorem book_page_digits : totalDigits 366 = 990 := by
  sorry

end NUMINAMATH_CALUDE_book_page_digits_l2975_297562


namespace NUMINAMATH_CALUDE_probability_of_winning_l2975_297568

/-- The probability of player A winning a match in a game with 2n rounds -/
def P (n : ℕ) : ℚ :=
  1/2 * (1 - (Nat.choose (2*n) n : ℚ) / (2^(2*n)))

/-- Theorem stating the probability of player A winning the match -/
theorem probability_of_winning (n : ℕ) (h : n > 0) : 
  P n = 1/2 * (1 - (Nat.choose (2*n) n : ℚ) / (2^(2*n))) :=
by sorry

end NUMINAMATH_CALUDE_probability_of_winning_l2975_297568


namespace NUMINAMATH_CALUDE_square_equation_solution_l2975_297580

theorem square_equation_solution :
  ∃! x : ℚ, (2015 + x)^2 = x^2 ∧ x = -2015/2 := by sorry

end NUMINAMATH_CALUDE_square_equation_solution_l2975_297580


namespace NUMINAMATH_CALUDE_addition_problem_l2975_297598

theorem addition_problem : ∃ x : ℝ, 37 + x = 52 ∧ x = 15 := by
  sorry

end NUMINAMATH_CALUDE_addition_problem_l2975_297598


namespace NUMINAMATH_CALUDE_rice_box_theorem_l2975_297545

/-- Represents the number of grains in each box -/
def grains_in_box (first_grain_count : ℕ) (common_difference : ℕ) (box_number : ℕ) : ℕ :=
  first_grain_count + (box_number - 1) * common_difference

/-- The total number of grains in all boxes -/
def total_grains (first_grain_count : ℕ) (common_difference : ℕ) (num_boxes : ℕ) : ℕ :=
  (num_boxes * (2 * first_grain_count + (num_boxes - 1) * common_difference)) / 2

theorem rice_box_theorem :
  (∃ (d : ℕ), total_grains 11 d 9 = 351 ∧ d = 7) ∧
  (∃ (d : ℕ), grains_in_box (23 - 2 * d) d 3 = 23 ∧ total_grains (23 - 2 * d) d 9 = 351 ∧ d = 8) :=
by sorry

end NUMINAMATH_CALUDE_rice_box_theorem_l2975_297545


namespace NUMINAMATH_CALUDE_friday_thirteenth_most_common_l2975_297511

/-- Represents a day of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a month in the Gregorian calendar -/
structure Month where
  startDay : DayOfWeek
  length : Nat

/-- Represents a year in the Gregorian calendar -/
structure Year where
  isLeap : Bool
  months : List Month

/-- Calculates the day of week for the 13th of a given month -/
def thirteenthDayOfMonth (m : Month) : DayOfWeek :=
  sorry

/-- Counts the occurrences of each day as the 13th in a 400-year cycle -/
def countThirteenths (years : List Year) : DayOfWeek → Nat :=
  sorry

/-- The Gregorian calendar repeats every 400 years -/
def gregorianCycle : List Year :=
  sorry

/-- Main theorem: Friday is the most common day for the 13th of a month -/
theorem friday_thirteenth_most_common :
  ∀ d : DayOfWeek, d ≠ DayOfWeek.Friday →
    countThirteenths gregorianCycle DayOfWeek.Friday > countThirteenths gregorianCycle d :=
  sorry

end NUMINAMATH_CALUDE_friday_thirteenth_most_common_l2975_297511


namespace NUMINAMATH_CALUDE_divisor_of_p_l2975_297563

theorem divisor_of_p (p q r s : ℕ+) 
  (h1 : Nat.gcd p.val q.val = 40)
  (h2 : Nat.gcd q.val r.val = 50)
  (h3 : Nat.gcd r.val s.val = 60)
  (h4 : 80 < Nat.gcd s.val p.val ∧ Nat.gcd s.val p.val < 120) :
  13 ∣ p.val :=
by sorry

end NUMINAMATH_CALUDE_divisor_of_p_l2975_297563


namespace NUMINAMATH_CALUDE_product_pure_imaginary_l2975_297546

theorem product_pure_imaginary (x : ℝ) :
  (∃ y : ℝ, (x + Complex.I) * ((x + 1) + Complex.I) * ((x + 2) + 2 * Complex.I) = Complex.I * y) ↔
  x^3 + 3*x^2 - 9*x - 7 = 0 :=
by sorry

end NUMINAMATH_CALUDE_product_pure_imaginary_l2975_297546


namespace NUMINAMATH_CALUDE_camera_rental_theorem_l2975_297554

def camera_rental_problem (camera_value : ℝ) (rental_weeks : ℕ) 
  (base_fee_rate : ℝ) (high_demand_rate : ℝ) (low_demand_rate : ℝ)
  (insurance_rate : ℝ) (sales_tax_rate : ℝ)
  (mike_contribution_rate : ℝ) (sarah_contribution_rate : ℝ) (sarah_contribution_cap : ℝ)
  (alex_contribution_rate : ℝ) (alex_contribution_cap : ℝ) : Prop :=
  let base_fee := camera_value * base_fee_rate
  let high_demand_fee := base_fee + (camera_value * high_demand_rate)
  let low_demand_fee := base_fee - (camera_value * low_demand_rate)
  let total_rental_fee := 2 * high_demand_fee + 2 * low_demand_fee
  let insurance_fee := camera_value * insurance_rate
  let subtotal := total_rental_fee + insurance_fee
  let total_cost := subtotal + (subtotal * sales_tax_rate)
  let mike_contribution := total_cost * mike_contribution_rate
  let sarah_contribution := min (total_cost * sarah_contribution_rate) sarah_contribution_cap
  let alex_contribution := min (total_cost * alex_contribution_rate) alex_contribution_cap
  let total_contribution := mike_contribution + sarah_contribution + alex_contribution
  let john_payment := total_cost - total_contribution
  john_payment = 1015.20

theorem camera_rental_theorem : 
  camera_rental_problem 5000 4 0.10 0.03 0.02 0.05 0.08 0.20 0.30 1000 0.10 700 := by
  sorry

#check camera_rental_theorem

end NUMINAMATH_CALUDE_camera_rental_theorem_l2975_297554


namespace NUMINAMATH_CALUDE_f_properties_l2975_297532

-- Define the function f(x) = sin(1/x)
noncomputable def f (x : ℝ) : ℝ := Real.sin (1 / x)

-- State the theorem
theorem f_properties :
  -- The range of f(x) is [-1, 1]
  (∀ y ∈ Set.range f, -1 ≤ y ∧ y ≤ 1) ∧
  (∀ y : ℝ, -1 ≤ y ∧ y ≤ 1 → ∃ x ≠ 0, f x = y) ∧
  -- f(x) is monotonically decreasing on [2/π, +∞)
  (∀ x₁ x₂ : ℝ, x₁ ≥ 2/Real.pi ∧ x₂ ≥ 2/Real.pi ∧ x₁ < x₂ → f x₁ > f x₂) ∧
  -- For any m ∈ [-1, 1], f(x) = m has infinitely many solutions in (0, 1)
  (∀ m : ℝ, -1 ≤ m ∧ m ≤ 1 → ∃ (S : Set ℝ), S.Infinite ∧ S ⊆ Set.Ioo 0 1 ∧ ∀ x ∈ S, f x = m) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l2975_297532


namespace NUMINAMATH_CALUDE_five_fourths_of_twelve_fifths_l2975_297503

theorem five_fourths_of_twelve_fifths (x : ℚ) : x = 12 / 5 → (5 / 4) * x = 3 := by
  sorry

end NUMINAMATH_CALUDE_five_fourths_of_twelve_fifths_l2975_297503


namespace NUMINAMATH_CALUDE_andrei_club_visits_l2975_297512

theorem andrei_club_visits :
  ∀ (d c : ℕ),
  15 * d + 11 * c = 115 →
  d + c = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_andrei_club_visits_l2975_297512


namespace NUMINAMATH_CALUDE_judgment_not_basic_structure_l2975_297525

/-- Represents the basic structures of flowcharts in algorithms -/
inductive FlowchartStructure
  | Sequential
  | Selection
  | Loop
  | Judgment

/-- The set of basic flowchart structures -/
def BasicStructures : Set FlowchartStructure :=
  {FlowchartStructure.Sequential, FlowchartStructure.Selection, FlowchartStructure.Loop}

/-- Theorem: The judgment structure is not one of the three basic structures of flowcharts -/
theorem judgment_not_basic_structure :
  FlowchartStructure.Judgment ∉ BasicStructures :=
by sorry

end NUMINAMATH_CALUDE_judgment_not_basic_structure_l2975_297525


namespace NUMINAMATH_CALUDE_consecutive_numbers_probability_l2975_297523

def choose (n k : ℕ) : ℕ := Nat.choose n k

def p : ℚ :=
  1 - (choose 40 6 + choose 5 1 * choose 39 5 + choose 4 2 * choose 38 4 + choose 37 3) / choose 45 6

theorem consecutive_numbers_probability : 
  ⌊1000 * p⌋ = 56 := by sorry

end NUMINAMATH_CALUDE_consecutive_numbers_probability_l2975_297523


namespace NUMINAMATH_CALUDE_mollys_age_l2975_297582

theorem mollys_age (sandy_current : ℕ) (molly_current : ℕ) : 
  (sandy_current : ℚ) / molly_current = 4 / 3 →
  sandy_current + 6 = 38 →
  molly_current = 24 := by
sorry

end NUMINAMATH_CALUDE_mollys_age_l2975_297582


namespace NUMINAMATH_CALUDE_greatest_valid_n_l2975_297586

def is_valid (n : ℕ) : Prop :=
  n ≥ 100 ∧ n ≤ 999 ∧ ¬((Nat.factorial (n / 2)) % (n * (n + 1)) = 0)

theorem greatest_valid_n : 
  (∀ m : ℕ, m > 996 → m ≤ 999 → ¬(is_valid m)) ∧
  is_valid 996 := by sorry

end NUMINAMATH_CALUDE_greatest_valid_n_l2975_297586


namespace NUMINAMATH_CALUDE_remaining_packs_eq_26_l2975_297544

/-- The number of cookie packs Tory needs to sell -/
def total_goal : ℕ := 50

/-- The number of cookie packs Tory sold to his grandmother -/
def sold_to_grandmother : ℕ := 12

/-- The number of cookie packs Tory sold to his uncle -/
def sold_to_uncle : ℕ := 7

/-- The number of cookie packs Tory sold to a neighbor -/
def sold_to_neighbor : ℕ := 5

/-- The number of remaining cookie packs Tory needs to sell -/
def remaining_packs : ℕ := total_goal - (sold_to_grandmother + sold_to_uncle + sold_to_neighbor)

theorem remaining_packs_eq_26 : remaining_packs = 26 := by
  sorry

end NUMINAMATH_CALUDE_remaining_packs_eq_26_l2975_297544


namespace NUMINAMATH_CALUDE_evaluate_32_to_5_over_2_l2975_297518

theorem evaluate_32_to_5_over_2 : 32^(5/2) = 4096 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_32_to_5_over_2_l2975_297518


namespace NUMINAMATH_CALUDE_difference_constant_sum_not_always_minimal_when_equal_l2975_297549

theorem difference_constant_sum_not_always_minimal_when_equal :
  ¬ (∀ (a b : ℝ) (d : ℝ), 
    a > 0 → b > 0 → a - b = d → 
    (∀ (x y : ℝ), x > 0 → y > 0 → x - y = d → a + b ≤ x + y)) :=
sorry

end NUMINAMATH_CALUDE_difference_constant_sum_not_always_minimal_when_equal_l2975_297549


namespace NUMINAMATH_CALUDE_folded_rectangle_BC_l2975_297566

/-- A rectangle ABCD with the following properties:
  - AB = 10
  - AD is folded onto AB, creating crease AE
  - Triangle AED is folded along DE
  - AE intersects BC at point F
  - Area of triangle ABF is 2 -/
structure FoldedRectangle where
  AB : ℝ
  BC : ℝ
  area_ABF : ℝ
  AB_eq_10 : AB = 10
  area_ABF_eq_2 : area_ABF = 2

/-- Theorem: In a FoldedRectangle, BC = 5.2 -/
theorem folded_rectangle_BC (r : FoldedRectangle) : r.BC = 5.2 := by
  sorry

end NUMINAMATH_CALUDE_folded_rectangle_BC_l2975_297566


namespace NUMINAMATH_CALUDE_bowling_ball_weight_l2975_297506

theorem bowling_ball_weight (canoe_weight : ℕ) (num_canoes num_balls : ℕ) :
  canoe_weight = 35 →
  num_canoes = 4 →
  num_balls = 10 →
  num_canoes * canoe_weight = num_balls * (num_canoes * canoe_weight / num_balls) →
  (num_canoes * canoe_weight / num_balls : ℕ) = 14 :=
by
  sorry

end NUMINAMATH_CALUDE_bowling_ball_weight_l2975_297506


namespace NUMINAMATH_CALUDE_hexagon_vertex_traces_line_hexagon_lines_intersect_common_point_l2975_297552

/-- A point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- A line in 2D space -/
structure Line :=
  (a : ℝ)
  (b : ℝ)
  (c : ℝ)

/-- A regular hexagon -/
structure RegularHexagon :=
  (center : Point)
  (vertices : Fin 6 → Point)

/-- The center of the hexagon moves along this line -/
def centerLine : Line := sorry

/-- The fixed vertex A of the hexagon -/
def fixedVertexA : Point := sorry

/-- Function to check if a point lies on a line -/
def pointOnLine (p : Point) (l : Line) : Prop := sorry

/-- Function to check if two lines intersect -/
def linesIntersect (l1 l2 : Line) : Prop := sorry

/-- Theorem: Each vertex of a regular hexagon traces a straight line when the center moves along a line -/
theorem hexagon_vertex_traces_line (h : RegularHexagon) (i : Fin 6) :
  ∃ l : Line, ∀ t : ℝ, pointOnLine (h.vertices i) l :=
sorry

/-- Theorem: The lines traced by the five non-fixed vertices intersect at a common point -/
theorem hexagon_lines_intersect_common_point (h : RegularHexagon) :
  ∃ p : Point, ∀ i j : Fin 6, i ≠ j → i ≠ 0 → j ≠ 0 →
    ∃ l1 l2 : Line,
      (∀ t : ℝ, pointOnLine (h.vertices i) l1) ∧
      (∀ t : ℝ, pointOnLine (h.vertices j) l2) ∧
      linesIntersect l1 l2 ∧
      pointOnLine p l1 ∧ pointOnLine p l2 :=
sorry

end NUMINAMATH_CALUDE_hexagon_vertex_traces_line_hexagon_lines_intersect_common_point_l2975_297552


namespace NUMINAMATH_CALUDE_projectile_max_height_l2975_297558

/-- The height function of the projectile -/
def h (t : ℝ) : ℝ := -20 * t^2 + 80 * t + 36

/-- The maximum height reached by the projectile -/
def max_height : ℝ := 116

/-- Theorem stating that the maximum height reached by the projectile is 116 meters -/
theorem projectile_max_height :
  ∃ t : ℝ, h t = max_height ∧ ∀ s : ℝ, h s ≤ max_height :=
sorry

end NUMINAMATH_CALUDE_projectile_max_height_l2975_297558


namespace NUMINAMATH_CALUDE_infinitely_many_squares_2012_2013_divisibility_condition_l2975_297555

-- Part (a)
theorem infinitely_many_squares_2012_2013 :
  ∀ k : ℕ, ∃ t > k, ∃ a b : ℕ,
    2012 * t + 1 = a^2 ∧ 2013 * t + 1 = b^2 :=
sorry

-- Part (b)
theorem divisibility_condition (m n : ℕ) (hm : m > 0) (hn : n > 0) :
  (∃ x y : ℕ, m * n + 1 = x^2 ∧ m * n + n + 1 = y^2) →
  8 * (2 * m + 1) ∣ n :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_squares_2012_2013_divisibility_condition_l2975_297555


namespace NUMINAMATH_CALUDE_greatest_three_digit_number_l2975_297560

theorem greatest_three_digit_number : ∃ (n : ℕ), 
  n = 970 ∧ 
  n < 1000 ∧ 
  n ≥ 100 ∧ 
  ∃ (k : ℕ), n = 8 * k + 2 ∧ 
  ∃ (m : ℕ), n = 7 * m + 4 ∧ 
  ∀ (x : ℕ), x < 1000 ∧ x ≥ 100 ∧ (∃ (a : ℕ), x = 8 * a + 2) ∧ (∃ (b : ℕ), x = 7 * b + 4) → x ≤ n :=
by sorry

end NUMINAMATH_CALUDE_greatest_three_digit_number_l2975_297560


namespace NUMINAMATH_CALUDE_cube_root_equal_self_l2975_297537

theorem cube_root_equal_self : {x : ℝ | x = x^(1/3)} = {-1, 0, 1} := by sorry

end NUMINAMATH_CALUDE_cube_root_equal_self_l2975_297537


namespace NUMINAMATH_CALUDE_range_of_power_function_l2975_297524

theorem range_of_power_function (k : ℝ) (h : k > 0) :
  Set.range (fun x : ℝ => x ^ k) ∩ Set.Ici 1 = Set.Ici 1 := by sorry

end NUMINAMATH_CALUDE_range_of_power_function_l2975_297524


namespace NUMINAMATH_CALUDE_employed_females_percentage_l2975_297515

theorem employed_females_percentage
  (total_population : ℕ)
  (employed_percentage : ℚ)
  (employed_males_percentage : ℚ)
  (h1 : employed_percentage = 60 / 100)
  (h2 : employed_males_percentage = 48 / 100)
  : (employed_percentage - employed_males_percentage) / employed_percentage = 20 / 100 := by
  sorry

end NUMINAMATH_CALUDE_employed_females_percentage_l2975_297515


namespace NUMINAMATH_CALUDE_square_side_length_l2975_297561

theorem square_side_length (rectangle_length : ℝ) (rectangle_width : ℝ) (square_side : ℝ) : 
  rectangle_length = 9 →
  rectangle_width = 16 →
  square_side * square_side = rectangle_length * rectangle_width →
  square_side = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l2975_297561


namespace NUMINAMATH_CALUDE_dodecagon_enclosure_l2975_297507

/-- The number of sides of the central polygon -/
def m : ℕ := 12

/-- The number of smaller polygons enclosing the central polygon -/
def num_enclosing : ℕ := 12

/-- The number of smaller polygons meeting at each vertex of the central polygon -/
def num_meeting : ℕ := 3

/-- The number of sides of each smaller polygon -/
def n : ℕ := 12

/-- The interior angle of a regular polygon with m sides -/
def interior_angle (m : ℕ) : ℚ := (m - 2) * 180 / m

/-- The exterior angle of a regular polygon with m sides -/
def exterior_angle (m : ℕ) : ℚ := 180 - interior_angle m

/-- Theorem stating that n must be 12 for the given configuration -/
theorem dodecagon_enclosure :
  exterior_angle m = num_meeting * (exterior_angle n / num_meeting) :=
sorry

end NUMINAMATH_CALUDE_dodecagon_enclosure_l2975_297507


namespace NUMINAMATH_CALUDE_constant_term_binomial_expansion_l2975_297589

theorem constant_term_binomial_expansion (x : ℝ) : 
  let binomial := (x - 1 / (2 * Real.sqrt x)) ^ 9
  ∃ c : ℝ, c = 21/16 ∧ 
    (∀ ε > 0, ∃ δ > 0, ∀ y, |y - x| < δ → |binomial - c| < ε) :=
by
  sorry

end NUMINAMATH_CALUDE_constant_term_binomial_expansion_l2975_297589


namespace NUMINAMATH_CALUDE_man_son_age_ratio_l2975_297504

theorem man_son_age_ratio :
  ∀ (man_age son_age : ℕ),
    man_age = 36 →
    son_age = 12 →
    man_age + 12 = 2 * (son_age + 12) →
    man_age / son_age = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_man_son_age_ratio_l2975_297504


namespace NUMINAMATH_CALUDE_coin_flip_probability_l2975_297508

theorem coin_flip_probability (p : ℝ) : 
  (p ≥ 0 ∧ p ≤ 1) →  -- probability is between 0 and 1
  (∀ (n : ℕ), n > 0 → p = 1 - p) →  -- equal probability of heads and tails
  (3 : ℝ) * p^2 * (1 - p) = (3 / 8 : ℝ) →  -- probability of 2 heads in 3 flips is 0.375
  p = (1 / 2 : ℝ) := by
sorry

end NUMINAMATH_CALUDE_coin_flip_probability_l2975_297508


namespace NUMINAMATH_CALUDE_matching_shoes_probability_l2975_297550

/-- The probability of selecting a matching pair of shoes from a box with 7 pairs -/
theorem matching_shoes_probability (n : ℕ) (total : ℕ) (pairs : ℕ) : 
  n = 7 → total = 2 * n → pairs = n → 
  (pairs : ℚ) / (total.choose 2 : ℚ) = 1 / 13 := by
  sorry

#check matching_shoes_probability

end NUMINAMATH_CALUDE_matching_shoes_probability_l2975_297550


namespace NUMINAMATH_CALUDE_vika_made_84_dollars_l2975_297540

/-- The amount of money Vika made -/
def vika_money : ℕ := sorry

/-- The amount of money Kayla made -/
def kayla_money : ℕ := sorry

/-- The amount of money Saheed made -/
def saheed_money : ℕ := 216

theorem vika_made_84_dollars :
  (saheed_money = 4 * kayla_money) ∧ 
  (kayla_money = vika_money - 30) → 
  vika_money = 84 := by
  sorry

end NUMINAMATH_CALUDE_vika_made_84_dollars_l2975_297540


namespace NUMINAMATH_CALUDE_triangle_translation_l2975_297567

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a translation in 2D space -/
structure Translation where
  dx : ℝ
  dy : ℝ

/-- Applies a translation to a point -/
def applyTranslation (p : Point) (t : Translation) : Point :=
  { x := p.x + t.dx, y := p.y + t.dy }

theorem triangle_translation (a b : ℝ) :
  let A : Point := { x := -1, y := 2 }
  let B : Point := { x := 1, y := -1 }
  let C : Point := { x := 2, y := 1 }
  let A' : Point := { x := -3, y := a }
  let B' : Point := { x := b, y := 3 }
  let t : Translation := { dx := A'.x - A.x, dy := B'.y - B.y }
  applyTranslation C t = { x := 0, y := 5 } := by
  sorry

end NUMINAMATH_CALUDE_triangle_translation_l2975_297567


namespace NUMINAMATH_CALUDE_root_difference_equation_l2975_297564

theorem root_difference_equation (r s : ℝ) : 
  ((r - 5) * (r + 5) = 25 * r - 125) →
  ((s - 5) * (s + 5) = 25 * s - 125) →
  r ≠ s →
  r > s →
  r - s = 15 := by
sorry

end NUMINAMATH_CALUDE_root_difference_equation_l2975_297564


namespace NUMINAMATH_CALUDE_no_x_squared_term_l2975_297579

theorem no_x_squared_term (a : ℝ) : 
  (∀ x : ℝ, (x^2 + a*x + 2) * (2*x - 4) = 2*x^3 + (2*a - 4)*x^2 + (4 - 4*a)*x - 8) →
  (2*a - 4 = 0 ↔ a = 2) :=
by sorry

end NUMINAMATH_CALUDE_no_x_squared_term_l2975_297579


namespace NUMINAMATH_CALUDE_distance_between_points_on_parabola_l2975_297573

/-- The distance between two points on a parabola y = mx^2 + k -/
theorem distance_between_points_on_parabola
  (m k a b c d : ℝ) 
  (h1 : b = m * a^2 + k)
  (h2 : d = m * c^2 + k) :
  Real.sqrt ((c - a)^2 + (d - b)^2) = |c - a| * Real.sqrt (1 + m^2 * (c + a)^2) :=
by sorry

end NUMINAMATH_CALUDE_distance_between_points_on_parabola_l2975_297573


namespace NUMINAMATH_CALUDE_product_of_numbers_l2975_297588

theorem product_of_numbers (x y : ℝ) : 
  x - y = 12 → x^2 + y^2 = 250 → x * y = 52.7364 := by
  sorry

end NUMINAMATH_CALUDE_product_of_numbers_l2975_297588


namespace NUMINAMATH_CALUDE_eight_fourth_equals_sixteen_n_l2975_297528

theorem eight_fourth_equals_sixteen_n (n : ℕ) : 8^4 = 16^n → n = 3 := by
  sorry

end NUMINAMATH_CALUDE_eight_fourth_equals_sixteen_n_l2975_297528


namespace NUMINAMATH_CALUDE_deepak_age_l2975_297576

theorem deepak_age (arun_age deepak_age : ℕ) : 
  (arun_age : ℚ) / deepak_age = 5 / 7 →
  arun_age + 6 = 36 →
  deepak_age = 42 := by
sorry

end NUMINAMATH_CALUDE_deepak_age_l2975_297576


namespace NUMINAMATH_CALUDE_ellipse_foci_distance_l2975_297584

/-- The distance between the foci of the ellipse 9x^2 + 16y^2 = 144 is 2√7 -/
theorem ellipse_foci_distance :
  let a : ℝ := 4
  let b : ℝ := 3
  let c : ℝ := Real.sqrt (a^2 - b^2)
  (∀ x y : ℝ, 9 * x^2 + 16 * y^2 = 144) →
  2 * c = 2 * Real.sqrt 7 := by
sorry

end NUMINAMATH_CALUDE_ellipse_foci_distance_l2975_297584


namespace NUMINAMATH_CALUDE_vector_addition_path_l2975_297595

-- Define a 2D vector
def Vector2D := ℝ × ℝ

-- Define vector addition
def vec_add (v w : Vector2D) : Vector2D :=
  (v.1 + w.1, v.2 + w.2)

-- Define vector from point to point
def vec_from_to (A B : Vector2D) : Vector2D :=
  (B.1 - A.1, B.2 - A.2)

-- Theorem statement
theorem vector_addition_path (A B C D : Vector2D) :
  vec_add (vec_add (vec_from_to A B) (vec_from_to B C)) (vec_from_to C D) =
  vec_from_to A D :=
by sorry

end NUMINAMATH_CALUDE_vector_addition_path_l2975_297595


namespace NUMINAMATH_CALUDE_new_shipment_bears_l2975_297577

theorem new_shipment_bears (initial_stock : ℕ) (bears_per_shelf : ℕ) (num_shelves : ℕ) : 
  initial_stock = 6 → bears_per_shelf = 6 → num_shelves = 4 → 
  num_shelves * bears_per_shelf - initial_stock = 18 :=
by
  sorry

end NUMINAMATH_CALUDE_new_shipment_bears_l2975_297577


namespace NUMINAMATH_CALUDE_prob_different_colors_is_148_225_l2975_297502

def blue_chips : ℕ := 6
def red_chips : ℕ := 5
def yellow_chips : ℕ := 4

def total_chips : ℕ := blue_chips + red_chips + yellow_chips

def prob_different_colors : ℚ :=
  (blue_chips / total_chips) * ((red_chips + yellow_chips) / total_chips) +
  (red_chips / total_chips) * ((blue_chips + yellow_chips) / total_chips) +
  (yellow_chips / total_chips) * ((blue_chips + red_chips) / total_chips)

theorem prob_different_colors_is_148_225 :
  prob_different_colors = 148 / 225 :=
sorry

end NUMINAMATH_CALUDE_prob_different_colors_is_148_225_l2975_297502


namespace NUMINAMATH_CALUDE_cube_sum_theorem_l2975_297509

theorem cube_sum_theorem (a b : ℝ) (h1 : a + b = 7) (h2 : a * b = 5) :
  a^3 + b^3 = 238 := by
sorry

end NUMINAMATH_CALUDE_cube_sum_theorem_l2975_297509


namespace NUMINAMATH_CALUDE_zeros_not_adjacent_probability_l2975_297501

/-- The number of ones in the arrangement -/
def num_ones : ℕ := 4

/-- The number of zeros in the arrangement -/
def num_zeros : ℕ := 2

/-- The total number of elements in the arrangement -/
def total_elements : ℕ := num_ones + num_zeros

/-- The probability that two zeros are not adjacent when randomly arranged with four ones in a row -/
theorem zeros_not_adjacent_probability :
  (Nat.choose (total_elements - 1) num_zeros) / (Nat.choose total_elements num_zeros) = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_zeros_not_adjacent_probability_l2975_297501


namespace NUMINAMATH_CALUDE_total_cost_is_correct_l2975_297536

def tshirt_price : ℝ := 10
def sweater_price : ℝ := 25
def jacket_price : ℝ := 100
def jeans_price : ℝ := 40
def shoes_price : ℝ := 70

def tshirt_discount : ℝ := 0.20
def sweater_discount : ℝ := 0.10
def jacket_discount : ℝ := 0.15
def jeans_discount : ℝ := 0.05
def shoes_discount : ℝ := 0.25

def clothes_tax : ℝ := 0.06
def shoes_tax : ℝ := 0.09

def tshirt_quantity : ℕ := 8
def sweater_quantity : ℕ := 5
def jacket_quantity : ℕ := 3
def jeans_quantity : ℕ := 6
def shoes_quantity : ℕ := 4

def total_cost : ℝ :=
  (tshirt_price * tshirt_quantity * (1 - tshirt_discount) * (1 + clothes_tax)) +
  (sweater_price * sweater_quantity * (1 - sweater_discount) * (1 + clothes_tax)) +
  (jacket_price * jacket_quantity * (1 - jacket_discount) * (1 + clothes_tax)) +
  (jeans_price * jeans_quantity * (1 - jeans_discount) * (1 + clothes_tax)) +
  (shoes_price * shoes_quantity * (1 - shoes_discount) * (1 + shoes_tax))

theorem total_cost_is_correct : total_cost = 927.97 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_is_correct_l2975_297536


namespace NUMINAMATH_CALUDE_smallest_number_with_remainder_one_l2975_297531

theorem smallest_number_with_remainder_one : ∃ n : ℕ, 
  n > 1 ∧ 
  n % 7 = 1 ∧ 
  n % 11 = 1 ∧ 
  (∀ m : ℕ, m > 1 ∧ m % 7 = 1 ∧ m % 11 = 1 → m ≥ n) ∧ 
  n = 78 := by
sorry

end NUMINAMATH_CALUDE_smallest_number_with_remainder_one_l2975_297531


namespace NUMINAMATH_CALUDE_quadratic_two_roots_condition_l2975_297592

theorem quadratic_two_roots_condition (m : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ x^2 + 2*x - 1 + m = 0 ∧ y^2 + 2*y - 1 + m = 0) ↔ m ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_two_roots_condition_l2975_297592


namespace NUMINAMATH_CALUDE_number_ordering_l2975_297541

theorem number_ordering : (1 : ℚ) / 5 < (25 : ℚ) / 100 ∧ (25 : ℚ) / 100 < (42 : ℚ) / 100 ∧ (42 : ℚ) / 100 < (1 : ℚ) / 2 ∧ (1 : ℚ) / 2 < (3 : ℚ) / 4 := by
  sorry

end NUMINAMATH_CALUDE_number_ordering_l2975_297541


namespace NUMINAMATH_CALUDE_dog_play_area_l2975_297514

/-- The area outside of a square doghouse that a dog can reach with a leash -/
theorem dog_play_area (leash_length : ℝ) (doghouse_side : ℝ) : 
  leash_length = 4 →
  doghouse_side = 2 →
  (3 / 4 * π * leash_length^2 + 2 * (1 / 4 * π * doghouse_side^2)) = 14 * π :=
by sorry

end NUMINAMATH_CALUDE_dog_play_area_l2975_297514


namespace NUMINAMATH_CALUDE_probability_nine_heads_in_twelve_flips_l2975_297548

def n : ℕ := 12
def k : ℕ := 9

theorem probability_nine_heads_in_twelve_flips :
  (n.choose k) / (2^n : ℚ) = 220 / 4096 := by sorry

end NUMINAMATH_CALUDE_probability_nine_heads_in_twelve_flips_l2975_297548


namespace NUMINAMATH_CALUDE_consecutive_composites_l2975_297590

theorem consecutive_composites (a n : ℕ) (ha : a ≥ 2) (hn : n > 0) :
  ∃ k : ℕ, k > 0 ∧ ∀ i : ℕ, 1 ≤ i ∧ i ≤ n → ∃ d : ℕ, 1 < d ∧ d < a^k + i ∧ (a^k + i) % d = 0 :=
by sorry

end NUMINAMATH_CALUDE_consecutive_composites_l2975_297590


namespace NUMINAMATH_CALUDE_factorization_of_a_squared_minus_two_a_l2975_297510

theorem factorization_of_a_squared_minus_two_a (a : ℝ) : a^2 - 2*a = a*(a - 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_a_squared_minus_two_a_l2975_297510


namespace NUMINAMATH_CALUDE_ryan_english_hours_l2975_297585

/-- Ryan's daily study schedule -/
structure StudySchedule where
  english_hours : ℕ
  chinese_hours : ℕ
  chinese_more_than_english : chinese_hours = english_hours + 1

/-- Ryan's actual study schedule -/
def ryans_schedule : StudySchedule where
  english_hours := 6
  chinese_hours := 7
  chinese_more_than_english := by rfl

theorem ryan_english_hours :
  ryans_schedule.english_hours = 6 :=
by sorry

end NUMINAMATH_CALUDE_ryan_english_hours_l2975_297585
