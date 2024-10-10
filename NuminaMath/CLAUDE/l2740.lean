import Mathlib

namespace no_solution_squared_equals_negative_one_l2740_274001

theorem no_solution_squared_equals_negative_one :
  ¬ ∃ x : ℝ, (3*x - 2)^2 = -1 := by
  sorry

end no_solution_squared_equals_negative_one_l2740_274001


namespace xyz_equals_five_l2740_274092

-- Define the variables
variable (x y z : ℝ)

-- Define the theorem
theorem xyz_equals_five
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 24)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 9) :
  x * y * z = 5 := by
  sorry

end xyz_equals_five_l2740_274092


namespace cos_minus_sin_for_point_l2740_274042

theorem cos_minus_sin_for_point (α : Real) :
  (∃ (x y : Real), x = 3/5 ∧ y = -4/5 ∧ x = Real.cos α ∧ y = Real.sin α) →
  Real.cos α - Real.sin α = 7/5 := by
  sorry

end cos_minus_sin_for_point_l2740_274042


namespace green_bay_high_relay_race_length_l2740_274002

/-- Calculates the total length of a relay race given the number of team members and the distance each member runs. -/
def relay_race_length (team_members : ℕ) (distance_per_member : ℕ) : ℕ :=
  team_members * distance_per_member

/-- Theorem stating that a relay race with 5 team members, each running 30 meters, has a total length of 150 meters. -/
theorem green_bay_high_relay_race_length :
  relay_race_length 5 30 = 150 := by
  sorry

#eval relay_race_length 5 30

end green_bay_high_relay_race_length_l2740_274002


namespace min_value_expression_l2740_274028

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : 1/x + 1/y + 1/z = 9) :
  ∃ (m : ℝ), m = 2/675 ∧ ∀ (a b c : ℝ), a > 0 → b > 0 → c > 0 → 1/a + 1/b + 1/c = 9 → 2 * a^3 * b^2 * c ≥ m :=
by sorry

end min_value_expression_l2740_274028


namespace inequality_proof_l2740_274069

theorem inequality_proof (a : ℝ) (ha : a > 0) : 
  Real.sqrt (a^2 + 1/a^2) + 2 ≥ a + 1/a + Real.sqrt 2 := by
  sorry

end inequality_proof_l2740_274069


namespace f_has_root_in_interval_l2740_274054

-- Define the function f(x) = x^3 - 3x - 3
def f (x : ℝ) : ℝ := x^3 - 3*x - 3

-- Theorem: f(x) has a root in the interval (2, 3)
theorem f_has_root_in_interval :
  ∃ x : ℝ, 2 < x ∧ x < 3 ∧ f x = 0 :=
by
  sorry


end f_has_root_in_interval_l2740_274054


namespace total_shaded_area_l2740_274052

/-- The total shaded area of two squares with inscribed circles -/
theorem total_shaded_area (small_side large_side small_radius large_radius : ℝ)
  (h1 : small_side = 6)
  (h2 : large_side = 12)
  (h3 : small_radius = 3)
  (h4 : large_radius = 6) :
  (small_side ^ 2 - π * small_radius ^ 2) + (large_side ^ 2 - π * large_radius ^ 2) = 180 - 45 * π := by
  sorry

end total_shaded_area_l2740_274052


namespace largest_integer_less_than_100_remainder_5_mod_8_l2740_274008

theorem largest_integer_less_than_100_remainder_5_mod_8 :
  ∃ n : ℕ, n < 100 ∧ n % 8 = 5 ∧ ∀ m : ℕ, m < 100 ∧ m % 8 = 5 → m ≤ n :=
by
  sorry

end largest_integer_less_than_100_remainder_5_mod_8_l2740_274008


namespace kids_at_camp_difference_l2740_274093

theorem kids_at_camp_difference (camp_kids home_kids : ℕ) 
  (h1 : camp_kids = 819058) 
  (h2 : home_kids = 668278) : 
  camp_kids - home_kids = 150780 := by
  sorry

end kids_at_camp_difference_l2740_274093


namespace annie_bus_ride_l2740_274043

/-- The number of blocks Annie walked from her house to the bus stop -/
def blocks_to_bus_stop : ℕ := 5

/-- The total number of blocks Annie traveled -/
def total_blocks : ℕ := 24

/-- The number of blocks Annie rode the bus to the coffee shop -/
def blocks_by_bus : ℕ := (total_blocks - 2 * blocks_to_bus_stop) / 2

theorem annie_bus_ride : blocks_by_bus = 7 := by
  sorry

end annie_bus_ride_l2740_274043


namespace number_count_l2740_274036

theorem number_count (avg_all : ℝ) (avg_pair1 : ℝ) (avg_pair2 : ℝ) (avg_pair3 : ℝ) 
  (h1 : avg_all = 4.60)
  (h2 : avg_pair1 = 3.4)
  (h3 : avg_pair2 = 3.8)
  (h4 : avg_pair3 = 6.6) :
  ∃ n : ℕ, n = 6 ∧ n * avg_all = 2 * (avg_pair1 + avg_pair2 + avg_pair3) := by
  sorry

end number_count_l2740_274036


namespace average_book_cost_l2740_274045

theorem average_book_cost (initial_amount : ℕ) (books_bought : ℕ) (amount_left : ℕ) : 
  initial_amount = 236 → 
  books_bought = 6 → 
  amount_left = 14 → 
  (initial_amount - amount_left) / books_bought = 37 :=
by sorry

end average_book_cost_l2740_274045


namespace exists_greatest_n_leq_2008_l2740_274075

/-- Checks if a number is a perfect square -/
def isPerfectSquare (m : ℕ) : Prop := ∃ k : ℕ, m = k * k

/-- The sum of squares formula for natural numbers -/
def sumOfSquares (n : ℕ) : ℕ := n * (n + 1) * (2 * n + 1) / 6

/-- The formula for the sum of squares from n+1 to 3n -/
def sumOfSquaresNTo3N (n : ℕ) : ℕ := (26 * n^3 + 12 * n^2 + n) / 3

/-- The main theorem statement -/
theorem exists_greatest_n_leq_2008 :
  ∃ n : ℕ, n ≤ 2008 ∧ 
    isPerfectSquare (sumOfSquares n * sumOfSquaresNTo3N n) ∧
    ∀ m : ℕ, m > n → m ≤ 2008 → 
      ¬ isPerfectSquare (sumOfSquares m * sumOfSquaresNTo3N m) := by
  sorry

end exists_greatest_n_leq_2008_l2740_274075


namespace gcd_126_105_l2740_274061

theorem gcd_126_105 : Nat.gcd 126 105 = 21 := by
  sorry

end gcd_126_105_l2740_274061


namespace product_of_ratios_l2740_274098

theorem product_of_ratios (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) 
  (h₁ : x₁^3 - 3*x₁*y₁^2 = 2007 ∧ y₁^3 - 3*x₁^2*y₁ = 2006)
  (h₂ : x₂^3 - 3*x₂*y₂^2 = 2007 ∧ y₂^3 - 3*x₂^2*y₂ = 2006)
  (h₃ : x₃^3 - 3*x₃*y₃^2 = 2007 ∧ y₃^3 - 3*x₃^2*y₃ = 2006)
  (h₄ : y₁ ≠ 0 ∧ y₂ ≠ 0 ∧ y₃ ≠ 0) :
  (1 - x₁ / y₁) * (1 - x₂ / y₂) * (1 - x₃ / y₃) = 1 / 1003 := by
sorry

end product_of_ratios_l2740_274098


namespace stratified_sampling_athletes_l2740_274024

theorem stratified_sampling_athletes (total_male : ℕ) (total_female : ℕ) 
  (drawn_male : ℕ) (drawn_female : ℕ) : 
  total_male = 64 → total_female = 56 → drawn_male = 8 →
  (drawn_male : ℚ) / total_male = (drawn_female : ℚ) / total_female →
  drawn_female = 7 := by
  sorry

end stratified_sampling_athletes_l2740_274024


namespace complex_number_simplification_l2740_274032

theorem complex_number_simplification :
  let i : ℂ := Complex.I
  (5 : ℂ) / (2 - i) - i = 2 := by
  sorry

end complex_number_simplification_l2740_274032


namespace quadrilateral_interior_point_angles_l2740_274018

theorem quadrilateral_interior_point_angles 
  (a b c d x y z w : ℝ) 
  (h1 : a = x + y / 2)
  (h2 : b = y + z / 2)
  (h3 : c = z + w / 2)
  (h4 : d = w + x / 2)
  (h5 : x + y + z + w = 360) :
  x = (16 / 15) * (a - b / 2 + c / 4 - d / 8) := by sorry

end quadrilateral_interior_point_angles_l2740_274018


namespace additional_money_needed_l2740_274026

def lee_money : ℚ := 10
def friend_money : ℚ := 8
def chicken_wings_cost : ℚ := 6
def chicken_salad_cost : ℚ := 4
def cheeseburger_cost : ℚ := 3.5
def fries_cost : ℚ := 2
def soda_cost : ℚ := 1
def coupon_discount : ℚ := 0.15
def tax_rate : ℚ := 0.08

def total_order_cost : ℚ := chicken_wings_cost + chicken_salad_cost + 2 * cheeseburger_cost + fries_cost + 2 * soda_cost

def discounted_cost : ℚ := total_order_cost * (1 - coupon_discount)

def final_cost : ℚ := discounted_cost * (1 + tax_rate)

def total_money : ℚ := lee_money + friend_money

theorem additional_money_needed :
  final_cost - total_money = 1.28 := by sorry

end additional_money_needed_l2740_274026


namespace symmetric_quadratic_comparison_l2740_274023

/-- A quadratic function that opens upward and is symmetric about x = 2013 -/
class SymmetricQuadratic (f : ℝ → ℝ) :=
  (opens_upward : ∃ (a b c : ℝ), a > 0 ∧ ∀ x, f x = a * x^2 + b * x + c)
  (symmetric : ∀ x, f (2013 + x) = f (2013 - x))

/-- Theorem: For a symmetric quadratic function f that opens upward,
    f(2011) is greater than f(2014) -/
theorem symmetric_quadratic_comparison
  (f : ℝ → ℝ) [SymmetricQuadratic f] :
  f 2011 > f 2014 :=
sorry

end symmetric_quadratic_comparison_l2740_274023


namespace unique_nested_sqrt_integer_l2740_274086

theorem unique_nested_sqrt_integer : ∃! (n : ℕ+), ∃ (x : ℤ), x^2 = n + Real.sqrt (n + Real.sqrt (n + Real.sqrt n)) := by
  sorry

end unique_nested_sqrt_integer_l2740_274086


namespace xyz_equality_l2740_274053

theorem xyz_equality (x y z : ℕ+) (a b c d : ℝ) 
  (h1 : x ≤ y) (h2 : y ≤ z)
  (h3 : (x : ℝ) ^ a = (y : ℝ) ^ b)
  (h4 : (y : ℝ) ^ b = (z : ℝ) ^ c)
  (h5 : (z : ℝ) ^ c = 70 ^ d)
  (h6 : 1 / a + 1 / b + 1 / c = 1 / d) :
  x + y = z := by sorry

end xyz_equality_l2740_274053


namespace apothem_lateral_face_angle_l2740_274025

/-- Given a regular triangular pyramid where the lateral edge forms an angle of 60° with the base plane,
    the sine of the angle between the apothem and the plane of the adjacent lateral face
    is equal to (3√3) / 13. -/
theorem apothem_lateral_face_angle (a : ℝ) (h : a > 0) :
  let β : ℝ := 60 * π / 180  -- Convert 60° to radians
  let lateral_edge_angle : ℝ := β
  let apothem : ℝ := a * Real.sqrt 13 / (2 * Real.sqrt 3)
  let perpendicular_distance : ℝ := a * Real.sqrt 3 / 8
  let sin_φ : ℝ := perpendicular_distance / apothem
  sin_φ = 3 * Real.sqrt 3 / 13 := by
  sorry


end apothem_lateral_face_angle_l2740_274025


namespace bake_sale_cookies_l2740_274021

/-- The number of chocolate chip cookies Jenny brought to the bake sale -/
def jenny_chocolate_chip : ℕ := 50

/-- The total number of peanut butter cookies at the bake sale -/
def total_peanut_butter : ℕ := 70

/-- The number of lemon cookies Marcus brought to the bake sale -/
def marcus_lemon : ℕ := 20

/-- The probability of picking a peanut butter cookie -/
def prob_peanut_butter : ℚ := 1/2

theorem bake_sale_cookies :
  jenny_chocolate_chip = 50 ∧
  total_peanut_butter = 70 ∧
  marcus_lemon = 20 ∧
  prob_peanut_butter = 1/2 →
  jenny_chocolate_chip + marcus_lemon = total_peanut_butter :=
by sorry

end bake_sale_cookies_l2740_274021


namespace four_last_in_hundreds_of_fib_l2740_274091

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

/-- Function to get the hundreds digit of a natural number -/
def hundredsDigit (n : ℕ) : ℕ :=
  (n / 100) % 10

/-- Predicate to check if a digit has appeared in the hundreds position of any Fibonacci number up to the nth term -/
def digitAppearedInHundreds (d : ℕ) (n : ℕ) : Prop :=
  ∃ k, k ≤ n ∧ hundredsDigit (fib k) = d

/-- The main theorem: 4 is the last digit to appear in the hundreds position of a Fibonacci number -/
theorem four_last_in_hundreds_of_fib :
  ∃ N, digitAppearedInHundreds 4 N ∧
    ∀ d, d ≠ 4 → ∃ n, n < N ∧ digitAppearedInHundreds d n :=
  sorry

end four_last_in_hundreds_of_fib_l2740_274091


namespace train_length_l2740_274070

/-- The length of a train given its speed and time to cross a pole. -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 36 → time = 9 → speed * time * (5 / 18) = 90 :=
by sorry

end train_length_l2740_274070


namespace union_subset_implies_m_leq_three_l2740_274003

def A : Set ℝ := {x | x ≤ -1 ∨ x ≥ 3}
def B : Set ℝ := {x | 1 ≤ x ∧ x ≤ 6}
def C (m : ℝ) : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 2 * m}

theorem union_subset_implies_m_leq_three (m : ℝ) :
  B ∪ C m = B → m ≤ 3 := by sorry

end union_subset_implies_m_leq_three_l2740_274003


namespace daily_houses_count_l2740_274058

/-- Represents Kyle's newspaper delivery route --/
structure NewspaperRoute where
  /-- Number of houses receiving daily paper Monday through Saturday --/
  daily_houses : ℕ
  /-- Total number of papers delivered in a week --/
  total_weekly_papers : ℕ
  /-- Number of regular customers not receiving Sunday paper --/
  sunday_skip : ℕ
  /-- Number of houses receiving paper only on Sunday --/
  sunday_only : ℕ
  /-- Ensures the total weekly papers match the given conditions --/
  papers_match : total_weekly_papers = 
    (6 * daily_houses) + (daily_houses - sunday_skip + sunday_only)

/-- Theorem stating the number of houses receiving daily paper --/
theorem daily_houses_count (route : NewspaperRoute) 
  (h1 : route.total_weekly_papers = 720)
  (h2 : route.sunday_skip = 10)
  (h3 : route.sunday_only = 30) : 
  route.daily_houses = 100 := by
  sorry

#check daily_houses_count

end daily_houses_count_l2740_274058


namespace aubree_animal_count_l2740_274015

def total_animals (initial_beavers : ℕ) (initial_chipmunks : ℕ) : ℕ :=
  let final_beavers := 2 * initial_beavers
  let final_chipmunks := initial_chipmunks - 10
  initial_beavers + initial_chipmunks + final_beavers + final_chipmunks

theorem aubree_animal_count :
  total_animals 20 40 = 130 := by
  sorry

end aubree_animal_count_l2740_274015


namespace smallest_number_with_2020_divisors_l2740_274014

/-- The number of divisors of n -/
def num_divisors (n : ℕ) : ℕ := sorry

/-- n is the smallest natural number with exactly k distinct divisors -/
def is_smallest_with_divisors (n k : ℕ) : Prop :=
  num_divisors n = k ∧ ∀ m < n, num_divisors m ≠ k

theorem smallest_number_with_2020_divisors :
  is_smallest_with_divisors (2^100 * 3^4 * 5 * 7) 2020 := by
  sorry

end smallest_number_with_2020_divisors_l2740_274014


namespace original_number_before_increase_l2740_274078

theorem original_number_before_increase (x : ℝ) : x * 1.5 = 525 → x = 350 := by
  sorry

end original_number_before_increase_l2740_274078


namespace train_length_problem_l2740_274007

/-- Proves that under given conditions, the length of each train is 60 meters -/
theorem train_length_problem (v_fast v_slow : ℝ) (t : ℝ) (h1 : v_fast = 48) (h2 : v_slow = 36) (h3 : t = 36) :
  let v_rel := v_fast - v_slow
  let d := v_rel * t * (5 / 18)
  d / 2 = 60 := by sorry

end train_length_problem_l2740_274007


namespace train_length_calculation_l2740_274056

/-- The length of two trains passing each other on parallel tracks -/
theorem train_length_calculation (faster_speed slower_speed : ℝ) 
  (passing_time : ℝ) (h1 : faster_speed = 46) (h2 : slower_speed = 36) 
  (h3 : passing_time = 54) : 
  let relative_speed := (faster_speed - slower_speed) * (1000 / 3600)
  let train_length := (relative_speed * passing_time) / 2
  train_length = 75 := by
sorry

end train_length_calculation_l2740_274056


namespace solution_correctness_l2740_274005

theorem solution_correctness : 
  (∃ x y : ℚ, 2 * x - y = 3 ∧ 3 * x + 2 * y = 8 ∧ x = 2 ∧ y = 1) ∧
  (∃ x y : ℚ, 2 * x + y = 2 ∧ 8 * x + 3 * y = 9 ∧ x = 3/2 ∧ y = -1) := by
  sorry

#check solution_correctness

end solution_correctness_l2740_274005


namespace sufficient_not_necessary_condition_l2740_274000

theorem sufficient_not_necessary_condition (x : ℝ) : 
  (x ≥ 1 → |x + 1| + |x - 1| = 2 * |x|) ∧ 
  (∃ y : ℝ, y < 1 ∧ |y + 1| + |y - 1| = 2 * |y|) := by
  sorry

end sufficient_not_necessary_condition_l2740_274000


namespace P_when_a_is_3_range_of_a_for_Q_subset_P_l2740_274057

-- Define the sets P and Q
def P (a : ℝ) : Set ℝ := {x : ℝ | (x - a) * (x + 1) ≤ 0}
def Q : Set ℝ := {x : ℝ | |x - 1| ≤ 1}

-- Theorem 1: When a = 3, P = {x | -1 ≤ x ≤ 3}
theorem P_when_a_is_3 : P 3 = {x : ℝ | -1 ≤ x ∧ x ≤ 3} := by sorry

-- Theorem 2: The range of positive a such that Q ⊆ P is [2, +∞)
theorem range_of_a_for_Q_subset_P : 
  {a : ℝ | a > 0 ∧ Q ⊆ P a} = {a : ℝ | a ≥ 2} := by sorry

end P_when_a_is_3_range_of_a_for_Q_subset_P_l2740_274057


namespace rectangle_diagonal_perimeter_ratio_l2740_274022

theorem rectangle_diagonal_perimeter_ratio :
  ∀ (long_side : ℝ),
  long_side > 0 →
  let short_side := (1/3) * long_side
  let diagonal := Real.sqrt (short_side^2 + long_side^2)
  let perimeter := 2 * (short_side + long_side)
  let saved_distance := (1/3) * long_side
  diagonal + saved_distance = long_side →
  diagonal / perimeter = Real.sqrt 10 / 8 := by
  sorry

end rectangle_diagonal_perimeter_ratio_l2740_274022


namespace solution_to_linear_equation_l2740_274011

theorem solution_to_linear_equation :
  ∃ (x y : ℝ), x + 2 * y = 6 ∧ x = 2 ∧ y = 2 := by
  sorry

end solution_to_linear_equation_l2740_274011


namespace ellipse_eccentricity_l2740_274088

/-- The eccentricity of an ellipse with specific properties -/
theorem ellipse_eccentricity (a b : ℝ) (h1 : a > b) (h2 : b > 0) : 
  (∀ (x y : ℝ), x^2 / a^2 + y^2 / b^2 = 1 → 
    ∃ (k₁ k₂ : ℝ), k₁ * k₂ ≠ 0 ∧ 
      (∀ (P : ℝ × ℝ), P.1^2 / a^2 + P.2^2 / b^2 = 1 → 
        (|k₁| + |k₂| ≥ |((P.2) / (P.1 - a))| + |((P.2) / (P.1 + a))|))) →
  (∃ (k₁ k₂ : ℝ), k₁ * k₂ ≠ 0 ∧ |k₁| + |k₂| = 1) →
  Real.sqrt (1 - b^2 / a^2) = Real.sqrt 3 / 2 :=
by sorry

end ellipse_eccentricity_l2740_274088


namespace line_plane_parallelism_l2740_274041

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallelism relations
variable (parallel_lines : Line → Line → Prop)
variable (parallel_line_plane : Line → Plane → Prop)

-- Define the "outside of plane" relation
variable (outside_of_plane : Line → Plane → Prop)

theorem line_plane_parallelism 
  (m n : Line) (α : Plane)
  (h1 : outside_of_plane m α)
  (h2 : outside_of_plane n α)
  (h3 : parallel_lines m n)
  (h4 : parallel_line_plane m α) :
  parallel_line_plane n α :=
sorry

end line_plane_parallelism_l2740_274041


namespace unique_solution_quadratic_inequality_l2740_274044

theorem unique_solution_quadratic_inequality (a : ℝ) :
  (∃! x : ℝ, |x^2 + 2*a*x + a + 5| ≤ 3) ↔ (a = 4 ∨ a = -2) :=
by sorry

end unique_solution_quadratic_inequality_l2740_274044


namespace definite_integral_x_cubed_l2740_274066

theorem definite_integral_x_cubed : ∫ (x : ℝ) in (-1)..(1), x^3 = 0 := by
  sorry

end definite_integral_x_cubed_l2740_274066


namespace interior_lattice_points_collinear_l2740_274072

/-- A lattice point in the plane -/
structure LatticePoint where
  x : ℤ
  y : ℤ

/-- A triangle in the plane -/
structure Triangle where
  A : LatticePoint
  B : LatticePoint
  C : LatticePoint

/-- Check if a point is inside a triangle -/
def isInside (p : LatticePoint) (t : Triangle) : Prop := sorry

/-- Check if a point is on the boundary of a triangle -/
def isOnBoundary (p : LatticePoint) (t : Triangle) : Prop := sorry

/-- Check if points are collinear -/
def areCollinear (points : List LatticePoint) : Prop := sorry

/-- The main theorem -/
theorem interior_lattice_points_collinear (T : Triangle) :
  (∀ p : LatticePoint, isOnBoundary p T → (p = T.A ∨ p = T.B ∨ p = T.C)) →
  (∃! (points : List LatticePoint), points.length = 4 ∧ 
    (∀ p ∈ points, isInside p T) ∧
    (∀ p : LatticePoint, isInside p T → p ∈ points)) →
  ∃ (points : List LatticePoint), points.length = 4 ∧
    (∀ p ∈ points, isInside p T) ∧ areCollinear points :=
by sorry


end interior_lattice_points_collinear_l2740_274072


namespace tangent_line_sum_l2740_274073

-- Define the function f
variable (f : ℝ → ℝ)

-- State the theorem
theorem tangent_line_sum (h : ∀ y, y = f 1 → y = (1 : ℝ) + 2) : 
  f 1 + deriv f 1 = 4 := by sorry

end tangent_line_sum_l2740_274073


namespace circular_garden_area_increase_l2740_274059

theorem circular_garden_area_increase : 
  let original_diameter : ℝ := 20
  let new_diameter : ℝ := 30
  let original_area := π * (original_diameter / 2)^2
  let new_area := π * (new_diameter / 2)^2
  let area_increase := new_area - original_area
  let percent_increase := (area_increase / original_area) * 100
  percent_increase = 125 := by sorry

end circular_garden_area_increase_l2740_274059


namespace f_property_l2740_274074

/-- A function f(x) of the form ax^7 + bx - 2 -/
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^7 + b * x - 2

/-- The theorem stating that if f(2009) = 10, then f(-2009) = -14 -/
theorem f_property (a b : ℝ) (h : f a b 2009 = 10) : f a b (-2009) = -14 := by
  sorry

end f_property_l2740_274074


namespace only_145_satisfies_condition_l2740_274077

/-- Factorial function -/
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

/-- Check if a number is a three-digit number -/
def isThreeDigit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

/-- Get the hundreds digit of a number -/
def hundredsDigit (n : ℕ) : ℕ :=
  n / 100

/-- Get the tens digit of a number -/
def tensDigit (n : ℕ) : ℕ :=
  (n / 10) % 10

/-- Get the ones digit of a number -/
def onesDigit (n : ℕ) : ℕ :=
  n % 10

/-- Check if a number is equal to the sum of the factorials of its digits -/
def isEqualToSumOfDigitFactorials (n : ℕ) : Prop :=
  n = factorial (hundredsDigit n) + factorial (tensDigit n) + factorial (onesDigit n)

theorem only_145_satisfies_condition :
  ∀ n : ℕ, isThreeDigit n ∧ isEqualToSumOfDigitFactorials n ↔ n = 145 := by
  sorry

#check only_145_satisfies_condition

end only_145_satisfies_condition_l2740_274077


namespace circle_distance_characterization_l2740_274087

/-- Given two concentric circles C and S centered at P with radii r and s respectively,
    where s < r, and B is a point within S, this theorem characterizes the set of
    points A such that the distance from A to B is less than the distance from A
    to any point on circle C. -/
theorem circle_distance_characterization
  (P B : EuclideanSpace ℝ (Fin 2))  -- Points in 2D real Euclidean space
  (r s : ℝ)  -- Radii of circles C and S
  (h_s_lt_r : s < r)  -- Condition that s < r
  (h_B_in_S : ‖B - P‖ ≤ s)  -- B is within circle S
  (A : EuclideanSpace ℝ (Fin 2))  -- Arbitrary point A
  : (∀ (C : EuclideanSpace ℝ (Fin 2)), ‖C - P‖ = r → ‖A - B‖ < ‖A - C‖) ↔
    ‖A - B‖ < r - s :=
by sorry

end circle_distance_characterization_l2740_274087


namespace tshirt_sale_revenue_per_minute_l2740_274033

/-- Calculates the money made per minute in a t-shirt sale. -/
theorem tshirt_sale_revenue_per_minute 
  (total_shirts : ℕ) 
  (sale_duration : ℕ) 
  (black_shirt_price : ℕ) 
  (white_shirt_price : ℕ) : 
  total_shirts = 200 →
  sale_duration = 25 →
  black_shirt_price = 30 →
  white_shirt_price = 25 →
  (total_shirts / 2 * black_shirt_price + total_shirts / 2 * white_shirt_price) / sale_duration = 220 :=
by
  sorry

#check tshirt_sale_revenue_per_minute

end tshirt_sale_revenue_per_minute_l2740_274033


namespace closed_set_properties_l2740_274012

-- Define a closed set
def is_closed_set (M : Set Int) : Prop :=
  ∀ a b : Int, a ∈ M → b ∈ M → (a + b) ∈ M ∧ (a - b) ∈ M

-- Define the set M = {-4, -2, 0, 2, 4}
def M : Set Int := {-4, -2, 0, 2, 4}

-- Define the set of positive integers
def positive_integers : Set Int := {n : Int | n > 0}

-- Define a general closed set
def closed_set (A : Set Int) : Prop := is_closed_set A

theorem closed_set_properties :
  (¬ is_closed_set M) ∧
  (¬ is_closed_set positive_integers) ∧
  (∃ A₁ A₂ : Set Int, closed_set A₁ ∧ closed_set A₂ ∧ ¬ is_closed_set (A₁ ∪ A₂)) :=
sorry

end closed_set_properties_l2740_274012


namespace player_B_more_consistent_l2740_274097

def player_A_scores : List ℕ := [9, 7, 8, 7, 8, 10, 7, 9, 8, 7]
def player_B_scores : List ℕ := [7, 8, 9, 8, 7, 8, 9, 8, 9, 7]

def mean (scores : List ℕ) : ℚ :=
  (scores.sum : ℚ) / scores.length

def variance (scores : List ℕ) : ℚ :=
  let m := mean scores
  (scores.map (λ x => ((x : ℚ) - m) ^ 2)).sum / scores.length

theorem player_B_more_consistent :
  variance player_B_scores < variance player_A_scores :=
by sorry

end player_B_more_consistent_l2740_274097


namespace sandy_shopping_money_l2740_274016

theorem sandy_shopping_money (remaining_money : ℝ) (spent_percentage : ℝ) 
  (h1 : remaining_money = 224)
  (h2 : spent_percentage = 0.3)
  : (remaining_money / (1 - spent_percentage)) = 320 := by
  sorry

end sandy_shopping_money_l2740_274016


namespace fraction_problem_l2740_274084

theorem fraction_problem (n : ℕ) : 
  (n : ℚ) / (4 * n - 4) = 1 / 2 → n = 2 := by
sorry

end fraction_problem_l2740_274084


namespace factorization_2y_squared_minus_8_l2740_274081

theorem factorization_2y_squared_minus_8 (y : ℝ) : 2 * y^2 - 8 = 2 * (y + 2) * (y - 2) := by
  sorry

end factorization_2y_squared_minus_8_l2740_274081


namespace rearrangement_time_theorem_l2740_274034

/-- The number of letters in the name -/
def name_length : ℕ := 9

/-- The number of times the repeated letter appears -/
def repeated_letter_count : ℕ := 2

/-- The number of rearrangements that can be written per minute -/
def rearrangements_per_minute : ℕ := 15

/-- Calculate the number of unique rearrangements -/
def unique_rearrangements : ℕ := name_length.factorial / repeated_letter_count.factorial

/-- Calculate the total time in hours to write all rearrangements -/
def total_time_hours : ℚ :=
  (unique_rearrangements / rearrangements_per_minute : ℚ) / 60

/-- Theorem stating the time required to write all rearrangements -/
theorem rearrangement_time_theorem :
  total_time_hours = 201.6 := by sorry

end rearrangement_time_theorem_l2740_274034


namespace duck_race_charity_l2740_274079

/-- The amount of money raised in a charity duck race -/
def charity_amount (regular_price : ℝ) (large_price : ℝ) (regular_sold : ℕ) (large_sold : ℕ) : ℝ :=
  regular_price * (regular_sold : ℝ) + large_price * (large_sold : ℝ)

/-- Theorem stating the amount raised in the specific duck race -/
theorem duck_race_charity : 
  charity_amount 3 5 221 185 = 1588 := by
  sorry

end duck_race_charity_l2740_274079


namespace max_slices_formula_l2740_274031

/-- Represents a triangular cake with candles -/
structure TriangularCake where
  numCandles : ℕ
  candlesNotCollinear : True  -- Placeholder for the condition that no three candles are collinear

/-- The maximum number of triangular slices for a given cake -/
def maxSlices (cake : TriangularCake) : ℕ :=
  2 * cake.numCandles - 5

/-- Theorem stating the maximum number of slices for a cake with k candles -/
theorem max_slices_formula (k : ℕ) (h : k ≥ 3) :
  ∀ (cake : TriangularCake), cake.numCandles = k →
    maxSlices cake = 2 * k - 5 := by
  sorry

end max_slices_formula_l2740_274031


namespace quadratic_root_zero_l2740_274060

/-- Given a quadratic equation (a-1)x² + x + a² - 1 = 0 where one root is 0, prove that a = -1 -/
theorem quadratic_root_zero (a : ℝ) : 
  (∀ x, (a - 1) * x^2 + x + a^2 - 1 = 0 ↔ x = 0 ∨ x ≠ 0) →
  (∃ x, (a - 1) * x^2 + x + a^2 - 1 = 0 ∧ x = 0) →
  a = -1 :=
by sorry

end quadratic_root_zero_l2740_274060


namespace food_consumption_reduction_l2740_274049

/-- Calculates the required reduction in food consumption per student to maintain
    the same total cost given a decrease in student population and an increase in food price. -/
theorem food_consumption_reduction
  (initial_students : ℕ)
  (initial_price : ℝ)
  (student_decrease_rate : ℝ)
  (price_increase_rate : ℝ)
  (h1 : student_decrease_rate = 0.1)
  (h2 : price_increase_rate = 0.2)
  (h3 : initial_students > 0)
  (h4 : initial_price > 0) :
  let new_students : ℝ := initial_students * (1 - student_decrease_rate)
  let new_price : ℝ := initial_price * (1 + price_increase_rate)
  let consumption_ratio : ℝ := (initial_students * initial_price) / (new_students * new_price)
  abs (1 - consumption_ratio - 0.0741) < 0.0001 := by
  sorry


end food_consumption_reduction_l2740_274049


namespace annie_purchase_problem_l2740_274055

/-- Represents the number of items at each price point -/
structure ItemCounts where
  cents20 : ℕ
  dollars4 : ℕ
  dollars5 : ℕ

/-- The problem statement -/
theorem annie_purchase_problem (counts : ItemCounts) : 
  counts.cents20 + counts.dollars4 + counts.dollars5 = 50 →
  20 * counts.cents20 + 400 * counts.dollars4 + 500 * counts.dollars5 = 5000 →
  counts.cents20 = 40 := by
  sorry

end annie_purchase_problem_l2740_274055


namespace equation_solution_l2740_274017

theorem equation_solution : 
  ∃! x : ℚ, (4 * x - 2) / (5 * x - 5) = 3 / 4 ∧ x = -7 := by
  sorry

end equation_solution_l2740_274017


namespace line_up_five_people_l2740_274080

theorem line_up_five_people (people : Finset Char) : 
  people.card = 5 → Nat.factorial 5 = 120 := by
  sorry

end line_up_five_people_l2740_274080


namespace sqrt_less_implies_less_l2740_274038

theorem sqrt_less_implies_less (a b : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) :
  Real.sqrt a < Real.sqrt b → a < b :=
by sorry

end sqrt_less_implies_less_l2740_274038


namespace cos_180_eq_neg_one_l2740_274063

-- Define the rotation function
def rotate_180 (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, -p.2)

-- Define the cosine of 180 degrees
def cos_180 : ℝ := (rotate_180 (1, 0)).1

-- Theorem statement
theorem cos_180_eq_neg_one : cos_180 = -1 := by
  sorry

end cos_180_eq_neg_one_l2740_274063


namespace domino_arrangements_equal_combinations_l2740_274067

/-- Represents a grid with width and height -/
structure Grid :=
  (width : ℕ)
  (height : ℕ)

/-- Represents a domino with length 2 and width 1 -/
structure Domino :=
  (length : ℕ)
  (width : ℕ)

/-- The number of distinct domino arrangements on a grid -/
def num_arrangements (g : Grid) (d : Domino) (num_dominoes : ℕ) : ℕ := sorry

/-- The number of ways to choose k items from n items -/
def choose (n : ℕ) (k : ℕ) : ℕ := sorry

theorem domino_arrangements_equal_combinations (g : Grid) (d : Domino) :
  g.width = 6 →
  g.height = 5 →
  d.length = 2 →
  d.width = 1 →
  num_arrangements g d 5 = choose 9 5 := by sorry

end domino_arrangements_equal_combinations_l2740_274067


namespace angle_E_measure_l2740_274004

-- Define the hexagon and its angles
def Hexagon (A B C D E F : ℝ) : Prop :=
  -- Convexity is implied by the sum of angles being 720°
  A + B + C + D + E + F = 720 ∧
  -- Angles A, C, and D are congruent
  A = C ∧ A = D ∧
  -- Angle B is 20 degrees more than angle A
  B = A + 20 ∧
  -- Angles E and F are congruent
  E = F ∧
  -- Angle A is 30 degrees less than angle E
  A + 30 = E

-- Theorem statement
theorem angle_E_measure (A B C D E F : ℝ) :
  Hexagon A B C D E F → E = 158 := by
  sorry

end angle_E_measure_l2740_274004


namespace school_boys_count_l2740_274095

theorem school_boys_count :
  ∀ (x : ℕ),
  (x + x = 100) →
  (x = 50) :=
by
  sorry

#check school_boys_count

end school_boys_count_l2740_274095


namespace plane_line_relationship_l2740_274082

-- Define the types for planes and lines
variable (α β : Set (ℝ × ℝ × ℝ))
variable (a : Set (ℝ × ℝ × ℝ))

-- Define the perpendicular relation
def perpendicular (S T : Set (ℝ × ℝ × ℝ)) : Prop := sorry

-- Define the parallel relation
def parallel (S T : Set (ℝ × ℝ × ℝ)) : Prop := sorry

-- Define the contained relation
def contained (L P : Set (ℝ × ℝ × ℝ)) : Prop := sorry

-- Theorem statement
theorem plane_line_relationship 
  (h1 : perpendicular α β) 
  (h2 : perpendicular a β) : 
  contained a α ∨ parallel a α := by sorry

end plane_line_relationship_l2740_274082


namespace max_value_expression_l2740_274051

theorem max_value_expression (c d : ℝ) (hc : c > 0) (hd : d > 0) :
  (∀ y : ℝ, 3 * (c - y) * (y + Real.sqrt (y^2 + d^2)) ≤ 3/2 * (c^2 + d^2)) ∧
  (∃ y : ℝ, 3 * (c - y) * (y + Real.sqrt (y^2 + d^2)) = 3/2 * (c^2 + d^2)) :=
sorry

end max_value_expression_l2740_274051


namespace expand_expression_l2740_274085

theorem expand_expression (x y : ℝ) : (-x + 2*y) * (-x - 2*y) = x^2 - 4*y^2 := by
  sorry

end expand_expression_l2740_274085


namespace negation_of_universal_proposition_l2740_274006

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x ≥ 2) ↔ (∃ x : ℝ, x < 2) := by
  sorry

end negation_of_universal_proposition_l2740_274006


namespace min_value_of_expression_l2740_274062

theorem min_value_of_expression (a : ℝ) (ha : a > 0) :
  (a + 1)^2 / a ≥ 4 ∧ ((a + 1)^2 / a = 4 ↔ a = 1) := by
  sorry

end min_value_of_expression_l2740_274062


namespace greatest_common_factor_of_three_digit_palindromes_l2740_274035

-- Define a three-digit palindrome
def is_three_digit_palindrome (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999 ∧ ∃ a b : ℕ, a ≠ 0 ∧ a < 10 ∧ b < 10 ∧ n = 100 * a + 10 * b + a

-- Define the set of all three-digit palindromes
def three_digit_palindromes : Set ℕ :=
  {n : ℕ | is_three_digit_palindrome n}

-- Statement to prove
theorem greatest_common_factor_of_three_digit_palindromes :
  ∃ g : ℕ, g > 0 ∧ 
    (∀ n ∈ three_digit_palindromes, g ∣ n) ∧
    (∀ d : ℕ, d > 0 → (∀ n ∈ three_digit_palindromes, d ∣ n) → d ≤ g) ∧
    g = 101 :=
  sorry

end greatest_common_factor_of_three_digit_palindromes_l2740_274035


namespace candy_chocolate_choices_l2740_274050

theorem candy_chocolate_choices (num_candies num_chocolates : ℕ) : 
  num_candies = 2 → num_chocolates = 3 → num_candies + num_chocolates = 5 := by
  sorry

end candy_chocolate_choices_l2740_274050


namespace max_subsets_l2740_274019

/-- A set with 10 elements -/
def T : Finset (Fin 10) := Finset.univ

/-- The type of 5-element subsets of T -/
def Subset5 : Type := {S : Finset (Fin 10) // S.card = 5}

/-- The property that any two elements appear together in at most two subsets -/
def AtMostTwice (subsets : List Subset5) : Prop :=
  ∀ x y : Fin 10, x ≠ y → (subsets.filter (λ S => x ∈ S.1 ∧ y ∈ S.1)).length ≤ 2

/-- The main theorem -/
theorem max_subsets :
  (∃ subsets : List Subset5, AtMostTwice subsets ∧ subsets.length = 8) ∧
  (∀ subsets : List Subset5, AtMostTwice subsets → subsets.length ≤ 8) := by
  sorry

end max_subsets_l2740_274019


namespace total_investment_l2740_274076

/-- Given two investments with different interest rates, proves the total amount invested. -/
theorem total_investment (amount_at_8_percent : ℝ) (amount_at_9_percent : ℝ) 
  (h1 : amount_at_8_percent = 6000)
  (h2 : 0.08 * amount_at_8_percent + 0.09 * amount_at_9_percent = 840) :
  amount_at_8_percent + amount_at_9_percent = 10000 := by
  sorry

end total_investment_l2740_274076


namespace f_at_4_l2740_274046

/-- The polynomial function f(x) = x^5 + 3x^4 - 5x^3 + 7x^2 - 9x + 11 -/
def f (x : ℝ) : ℝ := x^5 + 3*x^4 - 5*x^3 + 7*x^2 - 9*x + 11

/-- Theorem: The value of f(4) is 1559 -/
theorem f_at_4 : f 4 = 1559 := by
  sorry

end f_at_4_l2740_274046


namespace function_values_l2740_274071

def f (x a b c : ℝ) : ℝ := x^3 + a*x^2 + b*x + c

theorem function_values (a b c : ℝ) :
  (∀ x, f x a b c ≤ f (-1) a b c) ∧
  (f (-1) a b c = 7) ∧
  (∀ x, f x a b c ≥ f 3 a b c) →
  a = -3 ∧ b = -9 ∧ c = 2 ∧ f 3 a b c = -25 :=
by sorry

end function_values_l2740_274071


namespace probability_not_all_same_dice_five_dice_not_all_same_l2740_274068

theorem probability_not_all_same_dice (n : ℕ) (s : ℕ) : 
  n > 0 → s > 1 → (1 - s / s^n : ℚ) = (s^n - s) / s^n := by sorry

-- The probability that five fair 6-sided dice don't all show the same number
theorem five_dice_not_all_same : 
  (1 - (6 : ℚ) / 6^5) = 1295 / 1296 := by
  have h : (1 - 6 / 6^5 : ℚ) = (6^5 - 6) / 6^5 := 
    probability_not_all_same_dice 5 6 (by norm_num) (by norm_num)
  rw [h]
  norm_num


end probability_not_all_same_dice_five_dice_not_all_same_l2740_274068


namespace ratio_fraction_value_l2740_274013

theorem ratio_fraction_value (a b : ℝ) (h : a / b = 4) :
  (a - 3 * b) / (2 * a - b) = 1 / 7 := by
sorry

end ratio_fraction_value_l2740_274013


namespace more_male_students_l2740_274048

theorem more_male_students (total : ℕ) (female : ℕ) (h1 : total = 280) (h2 : female = 127) :
  total - female - female = 26 := by
  sorry

end more_male_students_l2740_274048


namespace johann_oranges_l2740_274010

def oranges_problem (initial_oranges eaten_oranges stolen_fraction returned_oranges : ℕ) : Prop :=
  let remaining_after_eating := initial_oranges - eaten_oranges
  let stolen := (remaining_after_eating / 2 : ℕ)
  let final_oranges := remaining_after_eating - stolen + returned_oranges
  final_oranges = 30

theorem johann_oranges :
  oranges_problem 60 10 2 5 := by sorry

end johann_oranges_l2740_274010


namespace remainder_67_power_67_plus_67_mod_68_l2740_274029

theorem remainder_67_power_67_plus_67_mod_68 : (67^67 + 67) % 68 = 66 := by
  sorry

end remainder_67_power_67_plus_67_mod_68_l2740_274029


namespace solution_correctness_l2740_274009

noncomputable def solution_set : Set ℂ :=
  {0, 15, (1 + Complex.I * Real.sqrt 7) / 2, (1 - Complex.I * Real.sqrt 7) / 2}

def original_equation (x : ℂ) : Prop :=
  (15 * x - x^2) / (x + 1) * (x + (15 - x) / (x + 1)) = 30

theorem solution_correctness :
  ∀ x : ℂ, x ∈ solution_set ↔ original_equation x :=
sorry

end solution_correctness_l2740_274009


namespace smallest_sum_four_consecutive_primes_l2740_274099

/-- A function that returns true if a number is prime, false otherwise -/
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

/-- A function that checks if four consecutive natural numbers are all prime -/
def fourConsecutivePrimes (n : ℕ) : Prop :=
  isPrime n ∧ isPrime (n + 1) ∧ isPrime (n + 2) ∧ isPrime (n + 3)

/-- The sum of four consecutive natural numbers starting from n -/
def sumFourConsecutive (n : ℕ) : ℕ := n + (n + 1) + (n + 2) + (n + 3)

/-- The theorem stating that the smallest sum of four consecutive positive prime numbers
    that is divisible by 3 is 36 -/
theorem smallest_sum_four_consecutive_primes :
  ∃ n : ℕ, fourConsecutivePrimes n ∧ sumFourConsecutive n % 3 = 0 ∧
  sumFourConsecutive n = 36 ∧
  ∀ m : ℕ, m < n → ¬(fourConsecutivePrimes m ∧ sumFourConsecutive m % 3 = 0) :=
sorry

end smallest_sum_four_consecutive_primes_l2740_274099


namespace employee_salary_proof_l2740_274096

-- Define the total weekly salary
def total_salary : ℝ := 594

-- Define the ratio of m's salary to n's salary
def salary_ratio : ℝ := 1.2

-- Define n's salary
def n_salary : ℝ := 270

-- Theorem statement
theorem employee_salary_proof :
  n_salary * (1 + salary_ratio) = total_salary :=
by sorry

end employee_salary_proof_l2740_274096


namespace sqrt_720_equals_12_sqrt_5_l2740_274039

theorem sqrt_720_equals_12_sqrt_5 : Real.sqrt 720 = 12 * Real.sqrt 5 := by
  sorry

end sqrt_720_equals_12_sqrt_5_l2740_274039


namespace license_plate_count_l2740_274090

/-- The number of different license plate combinations with three unique letters 
    followed by a dash and three digits, where exactly one digit is repeated exactly once. -/
def license_plate_combinations : ℕ :=
  let letter_combinations := 26 * 25 * 24
  let digit_combinations := 10 * 3 * 9
  letter_combinations * digit_combinations

/-- Theorem stating that the number of license plate combinations is 4,212,000 -/
theorem license_plate_count :
  license_plate_combinations = 4212000 := by
  sorry

end license_plate_count_l2740_274090


namespace remainder_of_nested_division_l2740_274065

theorem remainder_of_nested_division (P D K Q R R'q R'r : ℕ) :
  D > 0 →
  K > 0 →
  K < D →
  P = Q * D + R →
  R = R'q * K + R'r →
  R'r < K →
  P % (D * K) = R'r :=
sorry

end remainder_of_nested_division_l2740_274065


namespace ellipse_intersection_product_l2740_274047

-- Define the ellipse
def ellipse (a b : ℝ) (x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define a point on the ellipse
def point_on_ellipse (a b : ℝ) (p : ℝ × ℝ) : Prop :=
  ellipse a b p.1 p.2

-- Define a diameter of the ellipse
def is_diameter (a b : ℝ) (c d : ℝ × ℝ) : Prop :=
  point_on_ellipse a b c ∧ point_on_ellipse a b d ∧ 
  c.1 = -d.1 ∧ c.2 = -d.2

-- Define a line parallel to CD passing through A
def parallel_line (a b : ℝ) (c d n m : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, n.1 = -a + t * (d.1 - c.1) ∧ 
            n.2 = t * (d.2 - c.2) ∧
            m.1 = -a + (a / (d.1 - c.1)) * (d.1 - c.1) ∧
            m.2 = (a / (d.1 - c.1)) * (d.2 - c.2)

-- Theorem statement
theorem ellipse_intersection_product (a b : ℝ) (c d n m : ℝ × ℝ) 
  (ha : a > 0) (hb : b > 0) 
  (hcd : is_diameter a b c d)
  (hnm : parallel_line a b c d n m)
  (hn : point_on_ellipse a b n) :
  let a := (-a, 0)
  let o := (0, 0)
  (dist a m) * (dist a n) = (dist c o) * (dist c d) := by sorry


end ellipse_intersection_product_l2740_274047


namespace cubic_three_roots_m_range_l2740_274037

/-- The cubic polynomial f(x) = x³ - 6x² + 9x -/
def f (x : ℝ) : ℝ := x^3 - 6*x^2 + 9*x

/-- Theorem stating the range of m for which x³ - 6x² + 9x + m = 0 has exactly three distinct real roots -/
theorem cubic_three_roots_m_range :
  ∀ m : ℝ, (∃! (r₁ r₂ r₃ : ℝ), r₁ ≠ r₂ ∧ r₂ ≠ r₃ ∧ r₁ ≠ r₃ ∧
    f r₁ + m = 0 ∧ f r₂ + m = 0 ∧ f r₃ + m = 0) ↔ 
  -4 < m ∧ m < 0 :=
sorry

end cubic_three_roots_m_range_l2740_274037


namespace anna_final_mark_l2740_274040

/-- Calculates the final mark given term mark, exam mark, and their respective weights -/
def calculate_final_mark (term_mark : ℝ) (exam_mark : ℝ) (term_weight : ℝ) (exam_weight : ℝ) : ℝ :=
  term_mark * term_weight + exam_mark * exam_weight

/-- Anna's final mark calculation -/
theorem anna_final_mark :
  calculate_final_mark 80 90 0.7 0.3 = 83 := by
  sorry

#eval calculate_final_mark 80 90 0.7 0.3

end anna_final_mark_l2740_274040


namespace fraction_inequality_triangle_sine_inequality_l2740_274020

-- Part 1
theorem fraction_inequality (m n p : ℝ) (hm : m > 0) (hn : n > 0) (hp : p > 0) (hmn : m > n) :
  n / m < (n + p) / (m + p) := by sorry

-- Part 2
theorem triangle_sine_inequality (A B C : ℝ) (h_triangle : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π) :
  (Real.sin C) / (Real.sin A + Real.sin B) + 
  (Real.sin A) / (Real.sin B + Real.sin C) + 
  (Real.sin B) / (Real.sin C + Real.sin A) < 2 := by sorry

end fraction_inequality_triangle_sine_inequality_l2740_274020


namespace intersection_of_A_and_B_l2740_274064

def A : Set ℝ := {2, 4, 6, 8}
def B : Set ℝ := {x : ℝ | 3 ≤ x ∧ x ≤ 6}

theorem intersection_of_A_and_B : A ∩ B = {4, 6} := by
  sorry

end intersection_of_A_and_B_l2740_274064


namespace chocolate_bar_revenue_increase_l2740_274094

theorem chocolate_bar_revenue_increase 
  (original_weight : ℝ) (original_price : ℝ) 
  (new_weight : ℝ) (new_price : ℝ) :
  original_weight = 400 →
  original_price = 150 →
  new_weight = 300 →
  new_price = 180 →
  let original_revenue := original_price / original_weight
  let new_revenue := new_price / new_weight
  let revenue_increase := (new_revenue - original_revenue) / original_revenue
  revenue_increase = 0.6 := by
  sorry

end chocolate_bar_revenue_increase_l2740_274094


namespace crazy_silly_school_series_diff_l2740_274030

theorem crazy_silly_school_series_diff (total_books total_movies : ℕ) 
  (h1 : total_books = 20) 
  (h2 : total_movies = 12) : 
  total_books - total_movies = 8 := by
  sorry

end crazy_silly_school_series_diff_l2740_274030


namespace example_quadratic_equation_l2740_274027

/-- Definition of a quadratic equation -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ (∀ x, f x = a * x^2 + b * x + c)

/-- The equation x² + 2x - 1 = 0 is a quadratic equation -/
theorem example_quadratic_equation :
  is_quadratic_equation (λ x => x^2 + 2*x - 1) :=
by
  sorry


end example_quadratic_equation_l2740_274027


namespace sqrt_expression_equals_six_l2740_274089

theorem sqrt_expression_equals_six :
  Real.sqrt ((16^10 / 16^9)^2 * 6^2) / 2^4 = 6 := by
  sorry

end sqrt_expression_equals_six_l2740_274089


namespace car_speed_l2740_274083

-- Define the problem parameters
def gallons_per_40_miles : ℝ := 1
def tank_capacity : ℝ := 12
def travel_time : ℝ := 5
def fuel_used_fraction : ℝ := 0.4166666666666667

-- Define the theorem
theorem car_speed (speed : ℝ) : 
  (gallons_per_40_miles * speed * travel_time / 40 = fuel_used_fraction * tank_capacity) →
  speed = 40 := by
  sorry


end car_speed_l2740_274083
