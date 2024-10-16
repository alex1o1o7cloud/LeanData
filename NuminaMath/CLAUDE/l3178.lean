import Mathlib

namespace NUMINAMATH_CALUDE_discounted_soda_price_70_cans_l3178_317805

/-- Calculate the price of discounted soda cans -/
def discounted_soda_price (regular_price : ℚ) (discount_percent : ℚ) (num_cans : ℕ) : ℚ :=
  let discounted_price := regular_price * (1 - discount_percent)
  let full_cases := num_cans / 24
  let remaining_cans := num_cans % 24
  discounted_price * (↑full_cases * 24 + ↑remaining_cans)

/-- The price of 70 cans of soda with a regular price of $0.55 and 25% discount in 24-can cases is $28.875 -/
theorem discounted_soda_price_70_cans :
  discounted_soda_price (55/100) (25/100) 70 = 28875/1000 :=
sorry

end NUMINAMATH_CALUDE_discounted_soda_price_70_cans_l3178_317805


namespace NUMINAMATH_CALUDE_cement_mixture_percentage_l3178_317895

/-- Proves that in a concrete mixture, given specific conditions, the remaining mixture must be 20% cement. -/
theorem cement_mixture_percentage
  (total_concrete : ℝ)
  (final_cement_percentage : ℝ)
  (high_cement_mixture_amount : ℝ)
  (high_cement_percentage : ℝ)
  (h1 : total_concrete = 10)
  (h2 : final_cement_percentage = 0.62)
  (h3 : high_cement_mixture_amount = 7)
  (h4 : high_cement_percentage = 0.8)
  : (total_concrete * final_cement_percentage - high_cement_mixture_amount * high_cement_percentage) / (total_concrete - high_cement_mixture_amount) = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_cement_mixture_percentage_l3178_317895


namespace NUMINAMATH_CALUDE_escalator_walking_rate_l3178_317809

/-- Proves that given an escalator moving upward at 10 ft/sec with a length of 112 feet,
    if a person takes 8 seconds to cover the entire length,
    then the person's walking rate on the escalator is 4 ft/sec. -/
theorem escalator_walking_rate
  (escalator_speed : ℝ)
  (escalator_length : ℝ)
  (time_taken : ℝ)
  (h1 : escalator_speed = 10)
  (h2 : escalator_length = 112)
  (h3 : time_taken = 8)
  : ∃ (walking_rate : ℝ),
    walking_rate = 4 ∧
    escalator_length = (walking_rate + escalator_speed) * time_taken :=
by sorry

end NUMINAMATH_CALUDE_escalator_walking_rate_l3178_317809


namespace NUMINAMATH_CALUDE_jan_skips_proof_l3178_317860

/-- Calculates the total number of skips in a given time period after doubling the initial speed -/
def total_skips (initial_speed : ℕ) (time : ℕ) : ℕ :=
  2 * initial_speed * time

/-- Proves that given an initial speed of 70 skips per minute, which doubles after training,
    the total number of skips in 5 minutes is equal to 700 -/
theorem jan_skips_proof :
  total_skips 70 5 = 700 := by
  sorry

#eval total_skips 70 5

end NUMINAMATH_CALUDE_jan_skips_proof_l3178_317860


namespace NUMINAMATH_CALUDE_f_19_equals_zero_l3178_317839

-- Define f as a function from ℝ to ℝ
variable (f : ℝ → ℝ)

-- Define the properties of f
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)
def has_period_two_negation (f : ℝ → ℝ) : Prop := ∀ x, f (x + 2) = -f x

-- Theorem statement
theorem f_19_equals_zero 
  (h1 : is_even f) 
  (h2 : has_period_two_negation f) : 
  f 19 = 0 := by sorry

end NUMINAMATH_CALUDE_f_19_equals_zero_l3178_317839


namespace NUMINAMATH_CALUDE_range_of_m_l3178_317877

theorem range_of_m (x m : ℝ) : 
  (m > 0) →
  (∀ x, (|1 - (x - 1) / 3| ≤ 2) → (x^2 - 2*x + 1 - m^2 ≤ 0)) →
  (∃ x, (|1 - (x - 1) / 3| > 2) ∧ (x^2 - 2*x + 1 - m^2 ≤ 0)) →
  (0 < m ∧ m ≤ 3) :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l3178_317877


namespace NUMINAMATH_CALUDE_rational_representation_condition_l3178_317885

theorem rational_representation_condition (x y : ℚ) (hx : x ≠ 0) (hy : y ≠ 0) :
  (∀ (q : ℚ), q > 0 → ∃ (r : ℚ), r > 0 ∧ q = (r * x) / (r * y)) ↔ x * y < 0 :=
sorry

end NUMINAMATH_CALUDE_rational_representation_condition_l3178_317885


namespace NUMINAMATH_CALUDE_eight_percent_of_fifty_l3178_317848

theorem eight_percent_of_fifty : ∃ x : ℝ, x = 50 * 0.08 ∧ x = 4 := by
  sorry

end NUMINAMATH_CALUDE_eight_percent_of_fifty_l3178_317848


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l3178_317821

/-- The eccentricity of the hyperbola x² - 4y² = 1 is √5/2 -/
theorem hyperbola_eccentricity : 
  let h : ℝ → ℝ → Prop := λ x y ↦ x^2 - 4*y^2 = 1
  ∃ e : ℝ, e = Real.sqrt 5 / 2 ∧ 
    ∀ x y : ℝ, h x y → 
      ∃ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧
        x^2 / a^2 - y^2 / b^2 = 1 ∧
        c^2 = a^2 + b^2 ∧
        e = c / a :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l3178_317821


namespace NUMINAMATH_CALUDE_threes_squared_threes_2009_squared_l3178_317879

/-- Represents a number consisting of n repeated digits -/
def repeated_digit (d : Nat) (n : Nat) : Nat :=
  match n with
  | 0 => 0
  | k + 1 => d + 10 * (repeated_digit d k)

/-- The theorem to be proved -/
theorem threes_squared (n : Nat) (h : n > 0) :
  (repeated_digit 3 n) ^ 2 = 
    repeated_digit 1 (n-1) * 10^n + 
    repeated_digit 8 (n-1) * 10 + 9 := by
  sorry

/-- The specific case for 2009 threes -/
theorem threes_2009_squared :
  (repeated_digit 3 2009) ^ 2 = 
    repeated_digit 1 2008 * 10^2009 + 
    repeated_digit 8 2008 * 10 + 9 := by
  sorry

end NUMINAMATH_CALUDE_threes_squared_threes_2009_squared_l3178_317879


namespace NUMINAMATH_CALUDE_full_price_revenue_is_3000_total_tickets_correct_total_revenue_correct_l3178_317837

/-- Represents the revenue from full-price tickets at a concert venue. -/
def revenue_full_price (total_tickets : ℕ) (total_revenue : ℕ) : ℕ :=
  let p : ℕ := 20  -- Price of a full-price ticket
  let f : ℕ := 150 -- Number of full-price tickets
  f * p

/-- Theorem stating that the revenue from full-price tickets is $3000. -/
theorem full_price_revenue_is_3000 :
  revenue_full_price 250 3500 = 3000 := by
  sorry

/-- Verifies that the total number of tickets is correct. -/
theorem total_tickets_correct (f h q : ℕ) :
  f + h + q = 250 := by
  sorry

/-- Verifies that the total revenue is correct. -/
theorem total_revenue_correct (f h q p : ℕ) :
  f * p + h * (p / 2) + q * (p / 4) = 3500 := by
  sorry

end NUMINAMATH_CALUDE_full_price_revenue_is_3000_total_tickets_correct_total_revenue_correct_l3178_317837


namespace NUMINAMATH_CALUDE_geometric_sum_first_five_terms_l3178_317853

/-- Sum of first n terms of a geometric sequence -/
def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

/-- The first term of the geometric sequence -/
def a : ℚ := 1/4

/-- The common ratio of the geometric sequence -/
def r : ℚ := 1/4

/-- The number of terms to sum -/
def n : ℕ := 5

theorem geometric_sum_first_five_terms :
  geometric_sum a r n = 341/1024 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sum_first_five_terms_l3178_317853


namespace NUMINAMATH_CALUDE_game_result_depends_only_on_blue_parity_l3178_317896

/-- Represents the color of a sprite -/
inductive Color
  | Red
  | Blue

/-- Represents the state of the game -/
structure GameState :=
  (red : Nat)   -- Number of red sprites
  (blue : Nat)  -- Number of blue sprites

/-- Represents the result of the game -/
def gameResult (initialState : GameState) : Color :=
  if initialState.blue % 2 = 1 then Color.Blue else Color.Red

/-- The main theorem stating that the game result only depends on the initial number of blue sprites -/
theorem game_result_depends_only_on_blue_parity (m n : Nat) :
  gameResult { red := m, blue := n } = 
  if n % 2 = 1 then Color.Blue else Color.Red :=
by sorry

end NUMINAMATH_CALUDE_game_result_depends_only_on_blue_parity_l3178_317896


namespace NUMINAMATH_CALUDE_peter_soda_purchase_l3178_317857

/-- The cost of soda per ounce in dollars -/
def soda_cost_per_ounce : ℚ := 25 / 100

/-- The amount Peter brought in dollars -/
def initial_amount : ℚ := 2

/-- The amount Peter left with in dollars -/
def remaining_amount : ℚ := 1 / 2

/-- The number of ounces of soda Peter bought -/
def soda_ounces : ℚ := (initial_amount - remaining_amount) / soda_cost_per_ounce

theorem peter_soda_purchase : soda_ounces = 6 := by
  sorry

end NUMINAMATH_CALUDE_peter_soda_purchase_l3178_317857


namespace NUMINAMATH_CALUDE_function_composition_problem_l3178_317843

/-- Given a function f(x) = ax - b where a > 0 and f(f(x)) = 4x - 3, prove that f(2) = 3 -/
theorem function_composition_problem (a b : ℝ) (h1 : a > 0) :
  (∀ x, ∃ f : ℝ → ℝ, f x = a * x - b) →
  (∀ x, ∃ f : ℝ → ℝ, f (f x) = 4 * x - 3) →
  ∃ f : ℝ → ℝ, f 2 = 3 :=
by sorry

end NUMINAMATH_CALUDE_function_composition_problem_l3178_317843


namespace NUMINAMATH_CALUDE_perfect_square_function_characterization_l3178_317893

theorem perfect_square_function_characterization 
  (g : ℕ → ℕ) 
  (h : ∀ m n : ℕ, ∃ k : ℕ, (g m + n) * (g n + m) = k * k) :
  ∃ c : ℕ, ∀ m : ℕ, g m = m + c :=
sorry

end NUMINAMATH_CALUDE_perfect_square_function_characterization_l3178_317893


namespace NUMINAMATH_CALUDE_max_k_value_l3178_317880

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 8*x + 15 = 0

-- Define the line
def line (k x : ℝ) (y : ℝ) : Prop := y = k*x - 2

-- Define the condition for a point on the line to be the center of a circle with radius 1 that intersects C
def intersects_C (k x y : ℝ) : Prop :=
  line k x y ∧ ∃ (x' y' : ℝ), circle_C x' y' ∧ (x - x')^2 + (y - y')^2 = 1

-- Theorem statement
theorem max_k_value :
  ∃ (k_max : ℝ), k_max = 4/3 ∧
  (∀ k : ℝ, (∃ x y : ℝ, intersects_C k x y) → k ≤ k_max) ∧
  (∃ x y : ℝ, intersects_C k_max x y) :=
sorry

end NUMINAMATH_CALUDE_max_k_value_l3178_317880


namespace NUMINAMATH_CALUDE_no_prime_root_sum_29_l3178_317806

/-- A quadratic equation x^2 - 29x + k = 0 with prime roots -/
def has_prime_roots (k : ℤ) : Prop :=
  ∃ p q : ℕ, 
    Nat.Prime p ∧ 
    Nat.Prime q ∧ 
    (p : ℤ) + (q : ℤ) = 29 ∧ 
    (p : ℤ) * (q : ℤ) = k

/-- There are no integer values of k such that x^2 - 29x + k = 0 has two prime roots -/
theorem no_prime_root_sum_29 : ¬∃ k : ℤ, has_prime_roots k := by
  sorry

end NUMINAMATH_CALUDE_no_prime_root_sum_29_l3178_317806


namespace NUMINAMATH_CALUDE_initial_machines_count_l3178_317812

/-- The number of machines initially operating to fill a production order -/
def initial_machines : ℕ := sorry

/-- The total number of machines available -/
def total_machines : ℕ := 7

/-- The time taken by the initial number of machines to fill the order (in hours) -/
def initial_time : ℕ := 42

/-- The time taken by all machines to fill the order (in hours) -/
def all_machines_time : ℕ := 36

/-- The rate at which each machine works (assumed to be constant and positive) -/
def machine_rate : ℝ := sorry

theorem initial_machines_count :
  initial_machines = 6 :=
by sorry

end NUMINAMATH_CALUDE_initial_machines_count_l3178_317812


namespace NUMINAMATH_CALUDE_ship_length_l3178_317829

/-- The length of a ship given its speed and time to cross a lighthouse -/
theorem ship_length (speed : ℝ) (time : ℝ) : 
  speed = 18 → time = 20 → speed * time * (1000 / 3600) = 100 := by
  sorry

#check ship_length

end NUMINAMATH_CALUDE_ship_length_l3178_317829


namespace NUMINAMATH_CALUDE_covering_circles_highest_point_covered_l3178_317815

/-- A circle in 2D space -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The highest point of a circle -/
def highestPoint (c : Circle) : ℝ × ℝ :=
  (c.center.1, c.center.2 + c.radius)

/-- Check if a point is inside or on a circle -/
def isInside (p : ℝ × ℝ) (c : Circle) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 ≤ c.radius^2

/-- A set of 101 unit circles where the first 100 cover the 101st -/
structure CoveringCircles where
  circles : Fin 101 → Circle
  all_unit : ∀ i, (circles i).radius = 1
  last_covered : ∀ p, isInside p (circles 100) → ∃ i < 100, isInside p (circles i)
  all_distinct : ∀ i j, i ≠ j → circles i ≠ circles j

theorem covering_circles_highest_point_covered (cc : CoveringCircles) :
  ∃ i j, i < 100 ∧ j < 100 ∧ i ≠ j ∧
    isInside (highestPoint (cc.circles j)) (cc.circles i) :=
  sorry

end NUMINAMATH_CALUDE_covering_circles_highest_point_covered_l3178_317815


namespace NUMINAMATH_CALUDE_salary_proof_l3178_317840

/-- Represents the monthly salary in Rupees -/
def monthly_salary : ℝ := 6500

/-- Represents the savings rate as a decimal -/
def savings_rate : ℝ := 0.2

/-- Represents the expense increase rate as a decimal -/
def expense_increase_rate : ℝ := 0.2

/-- Represents the new savings amount in Rupees after expense increase -/
def new_savings : ℝ := 260

theorem salary_proof :
  let original_expenses := monthly_salary * (1 - savings_rate)
  let new_expenses := original_expenses * (1 + expense_increase_rate)
  monthly_salary - new_expenses = new_savings := by
  sorry

#check salary_proof

end NUMINAMATH_CALUDE_salary_proof_l3178_317840


namespace NUMINAMATH_CALUDE_equal_coefficients_implies_n_seven_l3178_317816

theorem equal_coefficients_implies_n_seven (n : ℕ) (h1 : n ≥ 6) :
  (Nat.choose n 5 * 3^5 = Nat.choose n 6 * 3^6) → n = 7 := by
  sorry

end NUMINAMATH_CALUDE_equal_coefficients_implies_n_seven_l3178_317816


namespace NUMINAMATH_CALUDE_shooting_competition_probabilities_l3178_317884

theorem shooting_competition_probabilities 
  (p_A_not_losing : ℝ) 
  (p_B_losing : ℝ) 
  (h1 : p_A_not_losing = 0.59) 
  (h2 : p_B_losing = 0.44) : 
  ∃ (p_A_not_winning p_A_B_drawing : ℝ),
    p_A_not_winning = 0.56 ∧ 
    p_A_B_drawing = 0.15 := by
  sorry

end NUMINAMATH_CALUDE_shooting_competition_probabilities_l3178_317884


namespace NUMINAMATH_CALUDE_distribution_problem_l3178_317890

/-- Represents the number of ways to distribute n distinct objects into k non-empty groups -/
def distribute (n k : ℕ) : ℕ := sorry

/-- Represents the number of ways to distribute n distinct objects into k non-empty groups,
    where two specific objects cannot be in the same group -/
def distributeWithRestriction (n k : ℕ) : ℕ := sorry

/-- The main theorem stating that there are 114 ways to distribute 5 distinct objects
    into 3 non-empty groups, where two specific objects cannot be in the same group -/
theorem distribution_problem : distributeWithRestriction 5 3 = 114 := by sorry

end NUMINAMATH_CALUDE_distribution_problem_l3178_317890


namespace NUMINAMATH_CALUDE_largest_remainder_l3178_317847

theorem largest_remainder (A B : ℕ) : 
  (A / 13 = 33) → (A % 13 = B) → (∀ C : ℕ, (C / 13 = 33) → (C % 13 ≤ B)) → A = 441 :=
by sorry

end NUMINAMATH_CALUDE_largest_remainder_l3178_317847


namespace NUMINAMATH_CALUDE_distance_AB_is_abs_x_minus_one_l3178_317836

/-- Represents a point on a number line with a rational coordinate -/
structure Point where
  coord : ℚ

/-- Calculates the distance between two points on a number line -/
def distance (p q : Point) : ℚ :=
  |p.coord - q.coord|

theorem distance_AB_is_abs_x_minus_one (x : ℚ) :
  let A : Point := ⟨x⟩
  let B : Point := ⟨1⟩
  let C : Point := ⟨-1⟩
  distance A B = |x - 1| := by
  sorry

end NUMINAMATH_CALUDE_distance_AB_is_abs_x_minus_one_l3178_317836


namespace NUMINAMATH_CALUDE_oak_grove_library_books_l3178_317887

/-- The number of books in Oak Grove's school libraries -/
def school_books : ℕ := 5106

/-- The total number of books in all Oak Grove libraries -/
def total_books : ℕ := 7092

/-- The number of books in Oak Grove's public library -/
def public_books : ℕ := total_books - school_books

theorem oak_grove_library_books : public_books = 1986 := by
  sorry

end NUMINAMATH_CALUDE_oak_grove_library_books_l3178_317887


namespace NUMINAMATH_CALUDE_factorization_problem1_l3178_317855

theorem factorization_problem1 (m a : ℝ) : m * (a - 3) + 2 * (3 - a) = (a - 3) * (m - 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_problem1_l3178_317855


namespace NUMINAMATH_CALUDE_part_one_part_two_l3178_317841

-- Define the function f
def f (a x : ℝ) : ℝ := |x + a| + |x - 4|

-- Part I
theorem part_one : 
  ∀ x : ℝ, (f 1 x ≤ 2 * |x - 4|) ↔ (x < 1.5) := by sorry

-- Part II
theorem part_two : 
  (∀ x : ℝ, f a x ≥ 3) ↔ (a ≤ -7 ∨ a ≥ -1) := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l3178_317841


namespace NUMINAMATH_CALUDE_class_size_l3178_317823

/-- Represents the number of students in Ms. Perez's class -/
def S : ℕ := 30

/-- Represents the total number of cans collected -/
def total_cans : ℕ := 232

/-- Represents the number of students who collected 4 cans each -/
def students_4_cans : ℕ := 13

/-- Represents the number of students who didn't collect any cans -/
def students_0_cans : ℕ := 2

theorem class_size :
  S = 30 ∧
  S / 2 * 12 + students_4_cans * 4 = total_cans ∧
  S / 2 + students_4_cans + students_0_cans = S :=
by sorry

end NUMINAMATH_CALUDE_class_size_l3178_317823


namespace NUMINAMATH_CALUDE_dist_to_left_focus_is_ten_l3178_317886

/-- The hyperbola type -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  (pos_a : a > 0)
  (pos_b : b > 0)

/-- A point on the hyperbola -/
structure PointOnHyperbola (h : Hyperbola) where
  x : ℝ
  y : ℝ
  on_hyperbola : x^2 / h.a^2 - y^2 / h.b^2 = 1

/-- The distance from a point to the right focus of the hyperbola -/
def distToRightFocus (h : Hyperbola) (p : PointOnHyperbola h) : ℝ :=
  |p.x - h.a|

/-- The distance from a point to the left focus of the hyperbola -/
def distToLeftFocus (h : Hyperbola) (p : PointOnHyperbola h) : ℝ :=
  |p.x + h.a|

/-- The main theorem -/
theorem dist_to_left_focus_is_ten
  (h : Hyperbola)
  (p : PointOnHyperbola h)
  (right_focus_dist : distToRightFocus h p = 4)
  (h_eq : h.a = 3 ∧ h.b = 4) :
  distToLeftFocus h p = 10 := by
  sorry

end NUMINAMATH_CALUDE_dist_to_left_focus_is_ten_l3178_317886


namespace NUMINAMATH_CALUDE_james_total_points_l3178_317878

/-- Represents the quiz bowl game rules and James' performance --/
structure QuizBowl where
  points_per_correct : ℕ := 2
  points_per_incorrect : ℕ := 1
  bonus_points : ℕ := 4
  total_rounds : ℕ := 5
  questions_per_round : ℕ := 5
  james_correct_answers : ℕ := 24
  james_unanswered : ℕ := 1

/-- Calculates the total points James earned in the quiz bowl --/
def calculate_points (game : QuizBowl) : ℕ :=
  let total_questions := game.total_rounds * game.questions_per_round
  let points_from_correct := game.james_correct_answers * game.points_per_correct
  let full_rounds := (total_questions - game.james_unanswered - game.james_correct_answers) / game.questions_per_round
  let bonus_points := full_rounds * game.bonus_points
  points_from_correct + bonus_points

/-- Theorem stating that James' total points in the quiz bowl are 64 --/
theorem james_total_points (game : QuizBowl) : calculate_points game = 64 := by
  sorry

end NUMINAMATH_CALUDE_james_total_points_l3178_317878


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3178_317852

theorem sufficient_not_necessary_condition (a : ℝ) :
  (a > 1 → 1/a < 1) ∧ ¬(1/a < 1 → a > 1) := by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3178_317852


namespace NUMINAMATH_CALUDE_distribute_five_students_three_dorms_l3178_317866

/-- The number of ways to distribute students into dormitories -/
def distribute_students (n : ℕ) (m : ℕ) (min : ℕ) (max : ℕ) (restricted : ℕ) : ℕ := sorry

/-- The theorem stating the number of ways to distribute 5 students into 3 dormitories -/
theorem distribute_five_students_three_dorms :
  distribute_students 5 3 1 2 1 = 60 := by sorry

end NUMINAMATH_CALUDE_distribute_five_students_three_dorms_l3178_317866


namespace NUMINAMATH_CALUDE_mason_father_age_l3178_317808

/-- Mason's age -/
def mason_age : ℕ := 20

/-- Sydney's age -/
def sydney_age : ℕ := mason_age + 6

/-- Mason's father's age -/
def father_age : ℕ := sydney_age + 6

theorem mason_father_age : father_age = 26 := by
  sorry

end NUMINAMATH_CALUDE_mason_father_age_l3178_317808


namespace NUMINAMATH_CALUDE_prob_at_least_one_box_match_l3178_317803

/-- Represents the probability of a single block matching the previous one -/
def match_probability : ℚ := 1/2

/-- Represents the number of people -/
def num_people : ℕ := 3

/-- Represents the number of boxes -/
def num_boxes : ℕ := 3

/-- Represents the number of colors -/
def num_colors : ℕ := 3

/-- Calculates the probability of all three blocks in a single box being the same color -/
def prob_single_box_match : ℚ := match_probability * match_probability

/-- Calculates the probability of at least one box having all three blocks of the same color -/
theorem prob_at_least_one_box_match : 
  (1 : ℚ) - (1 - prob_single_box_match) ^ num_boxes = 37/64 :=
sorry

end NUMINAMATH_CALUDE_prob_at_least_one_box_match_l3178_317803


namespace NUMINAMATH_CALUDE_regular_polygon_with_150_degree_angles_l3178_317845

theorem regular_polygon_with_150_degree_angles (n : ℕ) : 
  n > 2 →                                 -- n is the number of sides, must be greater than 2
  (180 * (n - 2) : ℝ) = (150 * n : ℝ) →   -- sum of interior angles formula
  n = 12 :=                               -- conclusion: the polygon has 12 sides
by sorry

end NUMINAMATH_CALUDE_regular_polygon_with_150_degree_angles_l3178_317845


namespace NUMINAMATH_CALUDE_sum_20_terms_eq_2870_l3178_317834

-- Define the sequence a_n
def a (n : ℕ) : ℕ := n^2

-- Define the sum of the first n terms of the sequence
def sum_a (n : ℕ) : ℕ := 
  (n * (n + 1) * (2 * n + 1)) / 6

-- Theorem statement
theorem sum_20_terms_eq_2870 :
  sum_a 20 = 2870 := by sorry

end NUMINAMATH_CALUDE_sum_20_terms_eq_2870_l3178_317834


namespace NUMINAMATH_CALUDE_nobel_laureates_count_l3178_317867

/-- The number of Nobel Prize laureates at a workshop given specific conditions -/
theorem nobel_laureates_count (total : ℕ) (wolf : ℕ) (wolf_and_nobel : ℕ) :
  total = 50 →
  wolf = 31 →
  wolf_and_nobel = 14 →
  (total - wolf) = 2 * (total - wolf - 3) / 2 →
  ∃ (nobel : ℕ), nobel = 25 ∧ nobel = wolf_and_nobel + (total - wolf - 3) / 2 + 3 := by
  sorry

#check nobel_laureates_count

end NUMINAMATH_CALUDE_nobel_laureates_count_l3178_317867


namespace NUMINAMATH_CALUDE_line_proof_l3178_317818

-- Define the three given lines
def line1 (x y : ℝ) : Prop := 3 * x + y = 0
def line2 (x y : ℝ) : Prop := x + y - 2 = 0
def line3 (x y : ℝ) : Prop := 2 * x + y + 3 = 0

-- Define the result line
def result_line (x y : ℝ) : Prop := x - 2 * y + 7 = 0

-- Theorem statement
theorem line_proof :
  ∃ (x₀ y₀ : ℝ),
    (line1 x₀ y₀ ∧ line2 x₀ y₀) ∧  -- Intersection point
    (∀ (x y : ℝ), result_line x y → 
      ((y - y₀) = -(1 / (2 : ℝ)) * (x - x₀)) ∧  -- Slope is perpendicular
      (result_line x₀ y₀))  -- Result line passes through intersection
:= by sorry

end NUMINAMATH_CALUDE_line_proof_l3178_317818


namespace NUMINAMATH_CALUDE_min_value_bn_Sn_l3178_317825

def S (n : ℕ) : ℚ := n / (n + 1)

def b (n : ℕ) : ℤ := n - 8

theorem min_value_bn_Sn :
  ∃ (min : ℚ), min = -4 ∧
  ∀ (n : ℕ), n ≥ 1 → (b n : ℚ) * S n ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_value_bn_Sn_l3178_317825


namespace NUMINAMATH_CALUDE_mothers_day_discount_l3178_317817

theorem mothers_day_discount (original_price : ℝ) (final_price : ℝ) 
  (additional_discount : ℝ) (h1 : original_price = 125) 
  (h2 : final_price = 108) (h3 : additional_discount = 0.04) : 
  ∃ (initial_discount : ℝ), 
    final_price = (1 - additional_discount) * (original_price * (1 - initial_discount)) ∧ 
    initial_discount = 0.1 := by
  sorry

end NUMINAMATH_CALUDE_mothers_day_discount_l3178_317817


namespace NUMINAMATH_CALUDE_system_solution_l3178_317832

theorem system_solution (x y b : ℝ) : 
  (4 * x + 2 * y = b) → 
  (3 * x + 6 * y = 3 * b) → 
  (x = 3) → 
  (b = 27) := by
sorry

end NUMINAMATH_CALUDE_system_solution_l3178_317832


namespace NUMINAMATH_CALUDE_shared_course_count_is_24_l3178_317800

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of courses available -/
def total_courses : ℕ := 4

/-- The number of courses each person chooses -/
def courses_per_person : ℕ := 2

/-- The number of ways in which exactly one course is chosen by both people -/
def shared_course_count : ℕ := 
  choose total_courses courses_per_person * choose total_courses courses_per_person -
  choose total_courses courses_per_person -
  choose total_courses courses_per_person

theorem shared_course_count_is_24 : shared_course_count = 24 := by sorry

end NUMINAMATH_CALUDE_shared_course_count_is_24_l3178_317800


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l3178_317889

/-- Given a geometric sequence with common ratio 1/2, prove that S_4 / a_4 = 15 -/
theorem geometric_sequence_ratio (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n, a (n + 1) = (1 / 2) * a n) →  -- Common ratio q = 1/2
  (∀ n, S n = (a 1) * (1 - (1 / 2)^n) / (1 - (1 / 2))) →  -- Sum formula
  S 4 / a 4 = 15 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l3178_317889


namespace NUMINAMATH_CALUDE_two_color_rectangle_exists_l3178_317898

/-- A color type with two possible values -/
inductive Color
| Red
| Blue

/-- A point in a 2D grid -/
structure Point where
  x : Nat
  y : Nat

/-- A coloring function that assigns a color to each point in a grid -/
def Coloring := Point → Color

/-- A rectangle defined by its four vertices -/
structure Rectangle where
  topLeft : Point
  topRight : Point
  bottomLeft : Point
  bottomRight : Point

/-- Predicate to check if all vertices of a rectangle have the same color -/
def sameColorRectangle (c : Coloring) (r : Rectangle) : Prop :=
  c r.topLeft = c r.topRight ∧
  c r.topLeft = c r.bottomLeft ∧
  c r.topLeft = c r.bottomRight

/-- Theorem stating that in any 7x3 grid colored with two colors,
    there exists a rectangle with vertices of the same color -/
theorem two_color_rectangle_exists :
  ∀ (c : Coloring),
  (∀ (p : Point), p.x < 7 ∧ p.y < 3 → (c p = Color.Red ∨ c p = Color.Blue)) →
  ∃ (r : Rectangle),
    r.topLeft.x < 7 ∧ r.topLeft.y < 3 ∧
    r.topRight.x < 7 ∧ r.topRight.y < 3 ∧
    r.bottomLeft.x < 7 ∧ r.bottomLeft.y < 3 ∧
    r.bottomRight.x < 7 ∧ r.bottomRight.y < 3 ∧
    sameColorRectangle c r :=
by
  sorry


end NUMINAMATH_CALUDE_two_color_rectangle_exists_l3178_317898


namespace NUMINAMATH_CALUDE_halloween_candy_problem_l3178_317892

/-- Given the initial candy counts for Katie and her sister, and the number of pieces eaten,
    calculate the remaining candy count. -/
def remaining_candy (katie_candy : ℕ) (sister_candy : ℕ) (eaten : ℕ) : ℕ :=
  katie_candy + sister_candy - eaten

/-- Theorem stating that for the given problem, the remaining candy count is 7. -/
theorem halloween_candy_problem :
  remaining_candy 10 6 9 = 7 := by
  sorry

end NUMINAMATH_CALUDE_halloween_candy_problem_l3178_317892


namespace NUMINAMATH_CALUDE_division_of_eleven_by_five_l3178_317859

theorem division_of_eleven_by_five :
  ∃ (A B : ℕ), 11 = 5 * A + B ∧ B < 5 ∧ A = 2 := by
  sorry

end NUMINAMATH_CALUDE_division_of_eleven_by_five_l3178_317859


namespace NUMINAMATH_CALUDE_hundredth_digit_is_two_l3178_317802

/-- The decimal representation of 7/26 has a repeating cycle of 9 digits -/
def decimal_cycle : Fin 9 → Nat
| 0 => 2
| 1 => 6
| 2 => 9
| 3 => 2
| 4 => 3
| 5 => 0
| 6 => 7
| 7 => 6
| 8 => 9

/-- The 100th digit after the decimal point in the decimal representation of 7/26 -/
def hundredth_digit : Nat :=
  decimal_cycle (100 % 9)

theorem hundredth_digit_is_two : hundredth_digit = 2 := by
  sorry

end NUMINAMATH_CALUDE_hundredth_digit_is_two_l3178_317802


namespace NUMINAMATH_CALUDE_probability_A_B_different_groups_l3178_317811

def number_of_people : ℕ := 6
def number_of_groups : ℕ := 3

theorem probability_A_B_different_groups :
  let total_ways := (number_of_people.choose 2) * ((number_of_people - 2).choose 2) / (number_of_groups.factorial)
  let ways_same_group := ((number_of_people - 2).choose 2) / ((number_of_groups - 1).factorial)
  (total_ways - ways_same_group) / total_ways = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_probability_A_B_different_groups_l3178_317811


namespace NUMINAMATH_CALUDE_laundry_time_proof_l3178_317820

/-- Calculates the total time for laundry given the number of loads, time per load for washing, and time for drying. -/
def totalLaundryTime (numLoads : ℕ) (washTimePerLoad : ℕ) (dryTime : ℕ) : ℕ :=
  numLoads * washTimePerLoad + dryTime

/-- Proves that given the specified conditions, the total laundry time is 165 minutes. -/
theorem laundry_time_proof :
  totalLaundryTime 2 45 75 = 165 := by
  sorry

end NUMINAMATH_CALUDE_laundry_time_proof_l3178_317820


namespace NUMINAMATH_CALUDE_prime_value_of_polynomial_l3178_317897

theorem prime_value_of_polynomial (a : ℕ) :
  Nat.Prime (a^4 - 4*a^3 + 15*a^2 - 30*a + 27) →
  a^4 - 4*a^3 + 15*a^2 - 30*a + 27 = 11 :=
by sorry

end NUMINAMATH_CALUDE_prime_value_of_polynomial_l3178_317897


namespace NUMINAMATH_CALUDE_unique_solution_system_l3178_317827

theorem unique_solution_system :
  ∃! (x y z : ℝ),
    2 * x - y + z = 14 ∧
    y = 2 ∧
    x + z = 3 * y + 5 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_unique_solution_system_l3178_317827


namespace NUMINAMATH_CALUDE_matts_working_ratio_l3178_317862

/-- Matt's working schedule problem -/
theorem matts_working_ratio :
  let monday_minutes : ℕ := 450
  let wednesday_minutes : ℕ := 300
  let tuesday_minutes : ℕ := wednesday_minutes - 75
  tuesday_minutes * 2 = monday_minutes := by sorry

end NUMINAMATH_CALUDE_matts_working_ratio_l3178_317862


namespace NUMINAMATH_CALUDE_percentage_problem_l3178_317833

theorem percentage_problem (x : ℝ) (h : 0.2 * x = 100) : 
  (600 / x) * 100 = 120 := by
sorry

end NUMINAMATH_CALUDE_percentage_problem_l3178_317833


namespace NUMINAMATH_CALUDE_largest_integer_satisfying_inequality_l3178_317894

theorem largest_integer_satisfying_inequality :
  ∃ (x : ℤ), (3 * |2 * x + 1| - 5 < 22) ∧
  (∀ (y : ℤ), y > x → ¬(3 * |2 * y + 1| - 5 < 22)) ∧
  x = 3 := by
  sorry

end NUMINAMATH_CALUDE_largest_integer_satisfying_inequality_l3178_317894


namespace NUMINAMATH_CALUDE_evaluate_complex_expression_l3178_317868

theorem evaluate_complex_expression :
  let N := (Real.sqrt (Real.sqrt 10 + 3) - Real.sqrt (Real.sqrt 10 - 3)) / 
           Real.sqrt (Real.sqrt 10 + 2) - 
           Real.sqrt (6 - 4 * Real.sqrt 2)
  N = 1 + Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_complex_expression_l3178_317868


namespace NUMINAMATH_CALUDE_count_solutions_2x_3y_763_l3178_317870

theorem count_solutions_2x_3y_763 : 
  (Finset.filter (fun p : ℕ × ℕ => 2 * p.1 + 3 * p.2 = 763 ∧ p.1 > 0 ∧ p.2 > 0) (Finset.product (Finset.range 764) (Finset.range 764))).card = 127 := by
  sorry

end NUMINAMATH_CALUDE_count_solutions_2x_3y_763_l3178_317870


namespace NUMINAMATH_CALUDE_diamond_value_l3178_317851

def diamond (a b : ℤ) : ℚ := (a : ℚ)⁻¹ + (b : ℚ)⁻¹

theorem diamond_value (a b : ℤ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h1 : a + b = 10) (h2 : a * b = 24) : 
  diamond a b = 5 / 12 := by
  sorry

end NUMINAMATH_CALUDE_diamond_value_l3178_317851


namespace NUMINAMATH_CALUDE_f_is_quadratic_l3178_317871

/-- Definition of a quadratic equation -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The given equation -/
def f (x : ℝ) : ℝ := x^2 + 2*x - 1

/-- Theorem: f is a quadratic equation -/
theorem f_is_quadratic : is_quadratic_equation f := by
  sorry

end NUMINAMATH_CALUDE_f_is_quadratic_l3178_317871


namespace NUMINAMATH_CALUDE_main_theorem_l3178_317822

/-- A function satisfying the given conditions -/
def SatisfiesConditions (f : ℝ → ℝ) : Prop :=
  Continuous f ∧ f 0 = 1 ∧ ∀ x y, f (x + y) ≥ f x * f y

/-- The main theorem -/
theorem main_theorem (f : ℝ → ℝ) (h : SatisfiesConditions f) :
  ∃ a : ℝ, a > 0 ∧ ∀ x, f x = Real.exp (a * x) := by
  sorry

end NUMINAMATH_CALUDE_main_theorem_l3178_317822


namespace NUMINAMATH_CALUDE_inequality_solution_range_l3178_317828

theorem inequality_solution_range (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  (∃ m : ℝ, (4 / (x + 1) + 1 / y < m^2 + 3/2 * m)) →
  (∃ m : ℝ, m < -3 ∨ m > 3/2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l3178_317828


namespace NUMINAMATH_CALUDE_matches_played_l3178_317831

/-- Represents the number of matches played -/
def n : ℕ := sorry

/-- The current batting average -/
def current_average : ℕ := 50

/-- The runs scored in the next match -/
def next_match_runs : ℕ := 78

/-- The new batting average after the next match -/
def new_average : ℕ := 54

/-- The total runs scored before the next match -/
def total_runs : ℕ := n * current_average

/-- The total runs after the next match -/
def new_total_runs : ℕ := total_runs + next_match_runs

/-- The theorem stating the number of matches played -/
theorem matches_played : n = 6 := by sorry

end NUMINAMATH_CALUDE_matches_played_l3178_317831


namespace NUMINAMATH_CALUDE_inequality_and_equality_conditions_l3178_317801

theorem inequality_and_equality_conditions (x y : ℝ) (h : (x + 1) * (y + 2) = 8) :
  (x * y - 10)^2 ≥ 64 ∧ 
  (∃ (a b : ℝ), (a * b - 10)^2 = 64 ∧ (a + 1) * (b + 2) = 8) ∧
  (∀ (a b : ℝ), (a * b - 10)^2 = 64 ∧ (a + 1) * (b + 2) = 8 → (a = 1 ∧ b = 2) ∨ (a = -3 ∧ b = -6)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_equality_conditions_l3178_317801


namespace NUMINAMATH_CALUDE_money_left_after_spending_l3178_317810

theorem money_left_after_spending (initial_amount spent_on_sweets given_to_friend number_of_friends : ℚ) :
  initial_amount = 8.5 ∧ 
  spent_on_sweets = 1.25 ∧ 
  given_to_friend = 1.2 ∧ 
  number_of_friends = 2 →
  initial_amount - (spent_on_sweets + given_to_friend * number_of_friends) = 4.85 := by
sorry

end NUMINAMATH_CALUDE_money_left_after_spending_l3178_317810


namespace NUMINAMATH_CALUDE_water_pollution_scientific_notation_l3178_317899

/-- The amount of water polluted by a button-sized waste battery in liters -/
def water_pollution : ℕ := 600000

/-- Scientific notation representation of water_pollution -/
def scientific_notation : ℝ := 6 * (10 ^ 5)

theorem water_pollution_scientific_notation :
  (water_pollution : ℝ) = scientific_notation := by
  sorry

end NUMINAMATH_CALUDE_water_pollution_scientific_notation_l3178_317899


namespace NUMINAMATH_CALUDE_circle_radius_equality_l3178_317826

/-- The radius of a circle whose area is equal to the sum of the areas of four circles with radius 2 cm is 4 cm. -/
theorem circle_radius_equality (r : ℝ) : r > 0 → π * r^2 = 4 * (π * 2^2) → r = 4 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_equality_l3178_317826


namespace NUMINAMATH_CALUDE_sum_of_digits_879_times_492_l3178_317842

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

/-- The theorem stating that the sum of digits in the product of 879 and 492 is 27 -/
theorem sum_of_digits_879_times_492 :
  sum_of_digits (879 * 492) = 27 := by
  sorry

#eval sum_of_digits (879 * 492)  -- This line is optional, for verification

end NUMINAMATH_CALUDE_sum_of_digits_879_times_492_l3178_317842


namespace NUMINAMATH_CALUDE_sum_of_xyz_l3178_317881

theorem sum_of_xyz (x y z : ℕ+) (h : x + 2*x*y + 3*x*y*z = 115) : x + y + z = 10 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_xyz_l3178_317881


namespace NUMINAMATH_CALUDE_product_divisible_by_twelve_l3178_317807

theorem product_divisible_by_twelve (n : ℤ) : 12 ∣ (n^2 * (n^2 - 1)) := by
  sorry

end NUMINAMATH_CALUDE_product_divisible_by_twelve_l3178_317807


namespace NUMINAMATH_CALUDE_madeline_rent_correct_l3178_317882

/-- Calculate the amount Madeline needs for rent given her expenses, savings, hourly wage, and hours worked -/
def rent_amount (hourly_wage : ℝ) (hours_worked : ℝ) (groceries : ℝ) (medical : ℝ) (utilities : ℝ) (savings : ℝ) : ℝ :=
  hourly_wage * hours_worked - (groceries + medical + utilities + savings)

/-- Theorem stating that Madeline's rent amount is correct -/
theorem madeline_rent_correct (hourly_wage : ℝ) (hours_worked : ℝ) (groceries : ℝ) (medical : ℝ) (utilities : ℝ) (savings : ℝ) :
  rent_amount hourly_wage hours_worked groceries medical utilities savings = 1210 :=
by
  sorry

#eval rent_amount 15 138 400 200 60 200

end NUMINAMATH_CALUDE_madeline_rent_correct_l3178_317882


namespace NUMINAMATH_CALUDE_general_admission_tickets_l3178_317883

theorem general_admission_tickets (student_price general_price : ℕ) 
  (total_tickets total_revenue : ℕ) : 
  student_price = 4 →
  general_price = 6 →
  total_tickets = 525 →
  total_revenue = 2876 →
  ∃ (student_tickets general_tickets : ℕ),
    student_tickets + general_tickets = total_tickets ∧
    student_tickets * student_price + general_tickets * general_price = total_revenue ∧
    general_tickets = 388 :=
by sorry

end NUMINAMATH_CALUDE_general_admission_tickets_l3178_317883


namespace NUMINAMATH_CALUDE_range_of_g_l3178_317804

theorem range_of_g (x : ℝ) (h : x ∈ Set.Icc (-1) 1) :
  π/4 ≤ Real.arcsin x + Real.arccos x - Real.arctan x ∧ 
  Real.arcsin x + Real.arccos x - Real.arctan x ≤ 3*π/4 := by
  sorry

end NUMINAMATH_CALUDE_range_of_g_l3178_317804


namespace NUMINAMATH_CALUDE_units_digit_47_pow_25_l3178_317813

theorem units_digit_47_pow_25 : ∃ n : ℕ, 47^25 ≡ 7 [ZMOD 10] :=
sorry

end NUMINAMATH_CALUDE_units_digit_47_pow_25_l3178_317813


namespace NUMINAMATH_CALUDE_total_diagonals_total_internal_angles_l3178_317861

/-- Number of sides in a pentagon -/
def pentagon_sides : ℕ := 5

/-- Number of sides in an octagon -/
def octagon_sides : ℕ := 8

/-- Calculate the number of diagonals in a polygon -/
def diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Calculate the sum of internal angles in a polygon -/
def internal_angles_sum (n : ℕ) : ℕ := (n - 2) * 180

/-- The total number of diagonals in a pentagon and an octagon is 25 -/
theorem total_diagonals : 
  diagonals pentagon_sides + diagonals octagon_sides = 25 := by sorry

/-- The sum of internal angles of a pentagon and an octagon is 1620° -/
theorem total_internal_angles : 
  internal_angles_sum pentagon_sides + internal_angles_sum octagon_sides = 1620 := by sorry

end NUMINAMATH_CALUDE_total_diagonals_total_internal_angles_l3178_317861


namespace NUMINAMATH_CALUDE_larger_integer_proof_l3178_317875

theorem larger_integer_proof (x y : ℕ+) 
  (h1 : y - x = 8)
  (h2 : x * y = 272) : 
  y = 20 := by
  sorry

end NUMINAMATH_CALUDE_larger_integer_proof_l3178_317875


namespace NUMINAMATH_CALUDE_ellipse_intersection_theorem_l3178_317814

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse with equation x²/4 + y²/3 = 1 -/
def Ellipse (p : Point) : Prop :=
  p.x^2 / 4 + p.y^2 / 3 = 1

/-- Checks if two points are symmetric about the x-axis -/
def SymmetricAboutXAxis (p1 p2 : Point) : Prop :=
  p1.x = p2.x ∧ p1.y = -p2.y

/-- Represents a line passing through two points -/
def Line (p1 p2 : Point) : Point → Prop :=
  λ p => (p.y - p1.y) * (p2.x - p1.x) = (p.x - p1.x) * (p2.y - p1.y)

/-- Checks if a point is on the x-axis -/
def OnXAxis (p : Point) : Prop :=
  p.y = 0

/-- Checks if a line intersects the ellipse -/
def IntersectsEllipse (l : Point → Prop) : Prop :=
  ∃ p, l p ∧ Ellipse p

/-- Main theorem -/
theorem ellipse_intersection_theorem (D E A B : Point) 
    (hDE : SymmetricAboutXAxis D E) 
    (hD : Ellipse D) (hE : Ellipse E)
    (hA : OnXAxis A) (hB : OnXAxis B)
    (hDA : ¬IntersectsEllipse (Line D A))
    (hInt : IntersectsEllipse (Line D A) ∧ IntersectsEllipse (Line B E) ∧ 
            ∃ p, Line D A p ∧ Line B E p ∧ Ellipse p) :
    A.x * B.x = 4 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_intersection_theorem_l3178_317814


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l3178_317888

theorem quadratic_inequality_range (m : ℝ) : 
  (¬ ∃ x : ℝ, (m + 1) * x^2 - (m + 1) * x + 1 ≤ 0) ↔ 
  (m ≥ -1 ∧ m < 3) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l3178_317888


namespace NUMINAMATH_CALUDE_pie_chart_probability_l3178_317830

theorem pie_chart_probability (pE pF pG pH : ℚ) : 
  pE = 1/3 →
  pF = 1/6 →
  pG = pH →
  pE + pF + pG + pH = 1 →
  pG = 1/4 := by
sorry

end NUMINAMATH_CALUDE_pie_chart_probability_l3178_317830


namespace NUMINAMATH_CALUDE_linear_system_solution_l3178_317872

theorem linear_system_solution (x y m : ℝ) : 
  x + 2*y = m → 
  2*x - 3*y = 4 → 
  x + y = 7 → 
  m = 9 := by
sorry

end NUMINAMATH_CALUDE_linear_system_solution_l3178_317872


namespace NUMINAMATH_CALUDE_unique_solution_l3178_317838

/-- Floor function: greatest integer less than or equal to x -/
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

/-- The equation to be solved -/
def equation (x : ℝ) : Prop :=
  3 * x + 5 * (floor x) - 2017 = 0

/-- The theorem stating the unique solution -/
theorem unique_solution :
  ∃! x : ℝ, equation x ∧ x = 252 + 1/3 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_l3178_317838


namespace NUMINAMATH_CALUDE_perpendicular_lines_parallel_l3178_317846

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)

-- State the theorem
theorem perpendicular_lines_parallel (a b : Line) (α : Plane) :
  perpendicular a α → perpendicular b α → parallel a b :=
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_parallel_l3178_317846


namespace NUMINAMATH_CALUDE_quadratic_form_sum_l3178_317863

theorem quadratic_form_sum (a h k : ℝ) : 
  (∀ x, 2 * x^2 - 8 * x + 2 = a * (x - h)^2 + k) → a + h + k = -2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_form_sum_l3178_317863


namespace NUMINAMATH_CALUDE_find_m_l3178_317864

theorem find_m : ∃ m : ℝ, 10^m = 10^2 * Real.sqrt (10^90 / 0.0001) ∧ m = 49 := by
  sorry

end NUMINAMATH_CALUDE_find_m_l3178_317864


namespace NUMINAMATH_CALUDE_square_of_difference_formula_l3178_317874

theorem square_of_difference_formula (m n : ℝ) : 
  ¬ ∃ (a b : ℝ), (m - n) * (-m + n) = (a - b) * (a + b) := by
  sorry

end NUMINAMATH_CALUDE_square_of_difference_formula_l3178_317874


namespace NUMINAMATH_CALUDE_perpendicular_to_same_line_relationships_l3178_317850

-- Define a line in 3D space
structure Line3D where
  point : ℝ × ℝ × ℝ
  direction : ℝ × ℝ × ℝ

-- Define perpendicularity between a line and a plane
def perpendicular (l : Line3D) (p : Line3D) : Prop :=
  -- We assume this definition exists in the library
  sorry

-- Define the relationships between two lines
def parallel (l1 l2 : Line3D) : Prop :=
  -- We assume this definition exists in the library
  sorry

def intersect (l1 l2 : Line3D) : Prop :=
  -- We assume this definition exists in the library
  sorry

def skew (l1 l2 : Line3D) : Prop :=
  -- We assume this definition exists in the library
  sorry

-- Theorem statement
theorem perpendicular_to_same_line_relationships 
  (l1 l2 p : Line3D) (h1 : perpendicular l1 p) (h2 : perpendicular l2 p) :
  (parallel l1 l2 ∨ intersect l1 l2 ∨ skew l1 l2) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_to_same_line_relationships_l3178_317850


namespace NUMINAMATH_CALUDE_chipped_marbles_count_l3178_317873

/-- Represents the number of marbles in each bag -/
def bags : List Nat := [18, 19, 21, 23, 25, 34]

/-- The total number of marbles -/
def total_marbles : Nat := bags.sum

/-- Predicate to check if a number is divisible by 3 -/
def divisible_by_three (n : Nat) : Prop := n % 3 = 0

/-- The number of bags Jane takes -/
def jane_bags : Nat := 3

/-- The number of bags George takes -/
def george_bags : Nat := 2

/-- The number of bags that remain -/
def remaining_bags : Nat := bags.length - jane_bags - george_bags

/-- Theorem stating the number of chipped marbles -/
theorem chipped_marbles_count : 
  ∃ (chipped : Nat) (jane george : List Nat),
    chipped ∈ bags ∧
    jane.length = jane_bags ∧
    george.length = george_bags ∧
    (jane.sum = 2 * george.sum) ∧
    (∀ m ∈ jane ++ george, m ≠ chipped) ∧
    divisible_by_three (total_marbles - chipped) ∧
    chipped = 23 := by
  sorry

end NUMINAMATH_CALUDE_chipped_marbles_count_l3178_317873


namespace NUMINAMATH_CALUDE_cos_sum_square_75_15_l3178_317865

theorem cos_sum_square_75_15 :
  Real.cos (75 * π / 180) ^ 2 + Real.cos (15 * π / 180) ^ 2 + 
  Real.cos (75 * π / 180) * Real.cos (15 * π / 180) = 5 / 4 := by
  sorry

end NUMINAMATH_CALUDE_cos_sum_square_75_15_l3178_317865


namespace NUMINAMATH_CALUDE_prime_pair_divisibility_l3178_317891

theorem prime_pair_divisibility (p q : ℕ) : 
  Prime p ∧ Prime q → (p * q ∣ p^p + q^q + 1) ↔ ((p = 2 ∧ q = 5) ∨ (p = 5 ∧ q = 2)) := by
  sorry

end NUMINAMATH_CALUDE_prime_pair_divisibility_l3178_317891


namespace NUMINAMATH_CALUDE_school_fee_calculation_l3178_317869

def mother_money : ℚ :=
  2 * 100 + 1 * 50 + 5 * 20 + 3 * 10 + 4 * 5 + 6 * 0.25 + 10 * 0.1 + 5 * 0.05

def father_money : ℚ :=
  3 * 100 + 4 * 50 + 2 * 20 + 1 * 10 + 6 * 5 + 8 * 0.25 + 7 * 0.1 + 3 * 0.05

def school_fee : ℚ := mother_money + father_money

theorem school_fee_calculation : school_fee = 985.60 := by
  sorry

end NUMINAMATH_CALUDE_school_fee_calculation_l3178_317869


namespace NUMINAMATH_CALUDE_prob_same_color_is_31_105_l3178_317876

def blue_marbles : ℕ := 4
def yellow_marbles : ℕ := 5
def black_marbles : ℕ := 6
def total_marbles : ℕ := blue_marbles + yellow_marbles + black_marbles

def prob_same_color : ℚ :=
  (blue_marbles * (blue_marbles - 1) + yellow_marbles * (yellow_marbles - 1) + black_marbles * (black_marbles - 1)) /
  (total_marbles * (total_marbles - 1))

theorem prob_same_color_is_31_105 : prob_same_color = 31 / 105 := by
  sorry

end NUMINAMATH_CALUDE_prob_same_color_is_31_105_l3178_317876


namespace NUMINAMATH_CALUDE_work_completion_time_l3178_317824

theorem work_completion_time (a b : ℝ) (h1 : a > 0) (h2 : b > 0) : 
  (1 / a + 1 / b = 1 / 10) → (1 / a = 1 / 20) → (1 / b = 1 / 20) := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l3178_317824


namespace NUMINAMATH_CALUDE_no_4digit_square_abba_palindromes_l3178_317856

/-- A function that checks if a number is a 4-digit square --/
def is_4digit_square (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999 ∧ ∃ m : ℕ, m * m = n

/-- A function that checks if a number is a palindrome with two different middle digits (abba form) --/
def is_abba_palindrome (n : ℕ) : Prop :=
  ∃ a b : ℕ, 
    1 ≤ a ∧ a ≤ 9 ∧ 
    0 ≤ b ∧ b ≤ 9 ∧ 
    a ≠ b ∧
    n = a * 1000 + b * 100 + b * 10 + a

/-- The main theorem stating that there are no 4-digit squares that are abba palindromes --/
theorem no_4digit_square_abba_palindromes :
  ¬ ∃ n : ℕ, is_4digit_square n ∧ is_abba_palindrome n :=
sorry

end NUMINAMATH_CALUDE_no_4digit_square_abba_palindromes_l3178_317856


namespace NUMINAMATH_CALUDE_quadratic_equation_exponent_l3178_317819

/-- Given that 2x^m + (2-m)x - 5 = 0 is a quadratic equation in terms of x, prove that m = 2 -/
theorem quadratic_equation_exponent (m : ℝ) : 
  (∀ x, ∃ a b c : ℝ, a ≠ 0 ∧ 2*x^m + (2-m)*x - 5 = a*x^2 + b*x + c) → m = 2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_exponent_l3178_317819


namespace NUMINAMATH_CALUDE_derivative_of_exp_plus_x_l3178_317858

open Real

theorem derivative_of_exp_plus_x (x : ℝ) :
  deriv (fun x => exp x + x) x = exp x + 1 := by sorry

end NUMINAMATH_CALUDE_derivative_of_exp_plus_x_l3178_317858


namespace NUMINAMATH_CALUDE_base_conversion_four_digits_l3178_317835

theorem base_conversion_four_digits (b : ℕ) : b > 1 → (
  (256 < b^4) ∧ (b^3 ≤ 256) ↔ b = 5
) := by sorry

end NUMINAMATH_CALUDE_base_conversion_four_digits_l3178_317835


namespace NUMINAMATH_CALUDE_combined_years_is_75_l3178_317849

/-- The combined total of years taught by Virginia, Adrienne, and Dennis -/
def combinedYears (adrienneYears virginiaYears dennisYears : ℕ) : ℕ :=
  adrienneYears + virginiaYears + dennisYears

/-- Theorem stating the combined total of years taught is 75 -/
theorem combined_years_is_75 :
  ∀ (adrienneYears virginiaYears dennisYears : ℕ),
    virginiaYears = adrienneYears + 9 →
    virginiaYears = dennisYears - 9 →
    dennisYears = 34 →
    combinedYears adrienneYears virginiaYears dennisYears = 75 := by
  sorry


end NUMINAMATH_CALUDE_combined_years_is_75_l3178_317849


namespace NUMINAMATH_CALUDE_avocados_bought_by_sister_georgie_guacamole_problem_l3178_317844

theorem avocados_bought_by_sister (avocados_per_serving : ℕ) (initial_avocados : ℕ) (servings_made : ℕ) : ℕ :=
  let total_avocados_needed := avocados_per_serving * servings_made
  let additional_avocados := total_avocados_needed - initial_avocados
  additional_avocados

theorem georgie_guacamole_problem :
  avocados_bought_by_sister 3 5 3 = 4 := by
  sorry

end NUMINAMATH_CALUDE_avocados_bought_by_sister_georgie_guacamole_problem_l3178_317844


namespace NUMINAMATH_CALUDE_train_journey_time_l3178_317854

/-- Proves that if a train moving at 6/7 of its usual speed arrives 30 minutes late, then its usual journey time is 3 hours. -/
theorem train_journey_time (usual_speed : ℝ) (usual_time : ℝ) 
  (h1 : usual_speed > 0) 
  (h2 : usual_time > 0) 
  (h3 : (6 / 7 * usual_speed) * (usual_time + 1/2) = usual_speed * usual_time) : 
  usual_time = 3 := by
  sorry

end NUMINAMATH_CALUDE_train_journey_time_l3178_317854
