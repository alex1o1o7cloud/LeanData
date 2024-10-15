import Mathlib

namespace NUMINAMATH_CALUDE_probability_even_sum_l2187_218751

def card_set : Finset ℕ := {1, 2, 3, 4, 5}

def is_even_sum (pair : ℕ × ℕ) : Bool :=
  (pair.1 + pair.2) % 2 = 0

def favorable_outcomes : Finset (ℕ × ℕ) :=
  (card_set.product card_set).filter (λ pair => pair.1 < pair.2 ∧ is_even_sum pair)

def total_outcomes : Finset (ℕ × ℕ) :=
  (card_set.product card_set).filter (λ pair => pair.1 < pair.2)

theorem probability_even_sum :
  (favorable_outcomes.card : ℚ) / total_outcomes.card = 2 / 5 :=
by sorry

end NUMINAMATH_CALUDE_probability_even_sum_l2187_218751


namespace NUMINAMATH_CALUDE_f_difference_l2187_218715

/-- The function f(x) = x^4 + x^2 + 5x^3 -/
def f (x : ℝ) : ℝ := x^4 + x^2 + 5*x^3

/-- Theorem: f(5) - f(-5) = 1250 -/
theorem f_difference : f 5 - f (-5) = 1250 := by
  sorry

end NUMINAMATH_CALUDE_f_difference_l2187_218715


namespace NUMINAMATH_CALUDE_smallest_third_term_geometric_progression_l2187_218724

theorem smallest_third_term_geometric_progression 
  (a : ℝ) -- Common difference of the arithmetic progression
  (h1 : (5 : ℝ) < 9 + a) -- Ensure the second term of GP is positive
  (h2 : 9 + a < 37 + 2*a) -- Ensure the third term of GP is greater than the second
  (h3 : (9 + a)^2 = 5*(37 + 2*a)) -- Condition for geometric progression
  : 
  ∃ (x : ℝ), x = 29 - 20*Real.sqrt 6 ∧ 
  x ≤ 37 + 2*a ∧
  ∀ (y : ℝ), y = 37 + 2*a → x ≤ y :=
by sorry

end NUMINAMATH_CALUDE_smallest_third_term_geometric_progression_l2187_218724


namespace NUMINAMATH_CALUDE_largest_eight_digit_even_digits_proof_l2187_218728

def even_digits : List Nat := [0, 2, 4, 6, 8]

def is_eight_digit (n : Nat) : Prop := 10000000 ≤ n ∧ n ≤ 99999999

def contains_all_even_digits (n : Nat) : Prop :=
  ∀ d ∈ even_digits, ∃ k : Nat, n / (10^k) % 10 = d

def largest_eight_digit_with_even_digits : Nat := 99986420

theorem largest_eight_digit_even_digits_proof :
  is_eight_digit largest_eight_digit_with_even_digits ∧
  contains_all_even_digits largest_eight_digit_with_even_digits ∧
  ∀ n : Nat, is_eight_digit n → contains_all_even_digits n →
    n ≤ largest_eight_digit_with_even_digits :=
by sorry

end NUMINAMATH_CALUDE_largest_eight_digit_even_digits_proof_l2187_218728


namespace NUMINAMATH_CALUDE_police_coverage_l2187_218710

-- Define the set of intersections
inductive Intersection : Type
  | A | B | C | D | E | F | G | H | I | J | K

-- Define the streets as sets of intersections
def horizontal_streets : List (List Intersection) :=
  [[Intersection.A, Intersection.B, Intersection.C, Intersection.D],
   [Intersection.E, Intersection.F, Intersection.G],
   [Intersection.H, Intersection.I, Intersection.J, Intersection.K]]

def vertical_streets : List (List Intersection) :=
  [[Intersection.A, Intersection.E, Intersection.H],
   [Intersection.B, Intersection.F, Intersection.I],
   [Intersection.D, Intersection.G, Intersection.J]]

def diagonal_streets : List (List Intersection) :=
  [[Intersection.H, Intersection.F, Intersection.C],
   [Intersection.C, Intersection.G, Intersection.K]]

def all_streets : List (List Intersection) :=
  horizontal_streets ++ vertical_streets ++ diagonal_streets

-- Define the function to check if a street is covered by the given intersections
def street_covered (street : List Intersection) (officers : List Intersection) : Prop :=
  ∃ i ∈ street, i ∈ officers

-- Theorem statement
theorem police_coverage :
  ∀ (street : List Intersection),
    street ∈ all_streets →
    street_covered street [Intersection.B, Intersection.G, Intersection.H] :=
by
  sorry


end NUMINAMATH_CALUDE_police_coverage_l2187_218710


namespace NUMINAMATH_CALUDE_condition_p_sufficient_not_necessary_l2187_218716

theorem condition_p_sufficient_not_necessary :
  (∀ x y : ℝ, x > 1 ∧ y > 1 → x + y > 2 ∧ x * y > 1) ∧
  (∃ x y : ℝ, x + y > 2 ∧ x * y > 1 ∧ ¬(x > 1 ∧ y > 1)) := by
  sorry

end NUMINAMATH_CALUDE_condition_p_sufficient_not_necessary_l2187_218716


namespace NUMINAMATH_CALUDE_ellipse_outside_circle_l2187_218747

theorem ellipse_outside_circle (b : ℝ) (m : ℝ) (x y : ℝ) 
  (h_b : b > 0) (h_m : -1 < m ∧ m < 1) 
  (h_ellipse : x^2 / (b^2 + 1) + y^2 / b^2 = 1) :
  (x - m)^2 + y^2 ≥ 1 - m^2 := by sorry

end NUMINAMATH_CALUDE_ellipse_outside_circle_l2187_218747


namespace NUMINAMATH_CALUDE_world_grain_ratio_l2187_218794

def world_grain_supply : ℕ := 1800000
def world_grain_demand : ℕ := 2400000

theorem world_grain_ratio : 
  (world_grain_supply : ℚ) / world_grain_demand = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_world_grain_ratio_l2187_218794


namespace NUMINAMATH_CALUDE_line_point_distance_constraint_l2187_218796

/-- Given a line l: x + y + a = 0 and a point A(2,0), if there exists a point M on line l
    such that |MA| = 2|MO|, then a is in the interval [($2-4\sqrt{2})/3$, ($2+4\sqrt{2})/3$] -/
theorem line_point_distance_constraint (a : ℝ) :
  (∃ x y : ℝ, x + y + a = 0 ∧
    (x - 2)^2 + y^2 = 4 * (x^2 + y^2)) →
  a ∈ Set.Icc ((2 - 4 * Real.sqrt 2) / 3) ((2 + 4 * Real.sqrt 2) / 3) :=
by sorry


end NUMINAMATH_CALUDE_line_point_distance_constraint_l2187_218796


namespace NUMINAMATH_CALUDE_quadratic_roots_shift_l2187_218797

/-- Given a quadratic equation a(x+h)^2+k=0 with roots -3 and 2,
    prove that the roots of a(x-1+h)^2+k=0 are -2 and 3 -/
theorem quadratic_roots_shift (a h k : ℝ) (a_ne_zero : a ≠ 0) :
  (∀ x, a * (x + h)^2 + k = 0 ↔ x = -3 ∨ x = 2) →
  (∀ x, a * (x - 1 + h)^2 + k = 0 ↔ x = -2 ∨ x = 3) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_shift_l2187_218797


namespace NUMINAMATH_CALUDE_swimming_pool_volume_l2187_218709

theorem swimming_pool_volume :
  let shallow_width : ℝ := 9
  let shallow_length : ℝ := 12
  let shallow_depth : ℝ := 1
  let deep_width : ℝ := 15
  let deep_length : ℝ := 18
  let deep_depth : ℝ := 4
  let island_width : ℝ := 3
  let island_length : ℝ := 6
  let island_height : ℝ := 1
  let shallow_volume := shallow_width * shallow_length * shallow_depth
  let deep_volume := deep_width * deep_length * deep_depth
  let island_volume := island_width * island_length * island_height
  let total_volume := shallow_volume + deep_volume - island_volume
  total_volume = 1170 := by
sorry

end NUMINAMATH_CALUDE_swimming_pool_volume_l2187_218709


namespace NUMINAMATH_CALUDE_point_on_coordinate_axes_l2187_218760

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


end NUMINAMATH_CALUDE_point_on_coordinate_axes_l2187_218760


namespace NUMINAMATH_CALUDE_decreasing_quadratic_condition_l2187_218756

/-- A function f is decreasing on an interval [a, +∞) if for all x₁, x₂ in [a, +∞) with x₁ < x₂, f(x₁) > f(x₂) -/
def DecreasingOnInterval (f : ℝ → ℝ) (a : ℝ) :=
  ∀ x₁ x₂, a ≤ x₁ → x₁ < x₂ → f x₁ > f x₂

theorem decreasing_quadratic_condition (a : ℝ) :
  DecreasingOnInterval (fun x => a * x^2 + 4 * (a + 1) * x - 3) 2 ↔ a ≤ -1/2 := by
  sorry

end NUMINAMATH_CALUDE_decreasing_quadratic_condition_l2187_218756


namespace NUMINAMATH_CALUDE_fourth_root_81_times_cube_root_27_times_sqrt_9_equals_27_l2187_218788

theorem fourth_root_81_times_cube_root_27_times_sqrt_9_equals_27 :
  (81 : ℝ) ^ (1/4) * (27 : ℝ) ^ (1/3) * (9 : ℝ) ^ (1/2) = 27 := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_81_times_cube_root_27_times_sqrt_9_equals_27_l2187_218788


namespace NUMINAMATH_CALUDE_final_amount_after_bets_l2187_218754

theorem final_amount_after_bets (initial_amount : ℝ) (num_bets num_wins num_losses : ℕ) 
  (h1 : initial_amount = 64)
  (h2 : num_bets = 6)
  (h3 : num_wins = 3)
  (h4 : num_losses = 3)
  (h5 : num_wins + num_losses = num_bets) :
  let win_factor := (3/2 : ℝ)
  let loss_factor := (1/2 : ℝ)
  let final_factor := win_factor ^ num_wins * loss_factor ^ num_losses
  initial_amount * final_factor = 27 := by
  sorry

end NUMINAMATH_CALUDE_final_amount_after_bets_l2187_218754


namespace NUMINAMATH_CALUDE_midpoint_coordinates_sum_l2187_218764

/-- Given that M(-1,6) is the midpoint of CD and C(5,4) is one endpoint, 
    the sum of the coordinates of point D is 1. -/
theorem midpoint_coordinates_sum (C D M : ℝ × ℝ) : 
  C = (5, 4) → 
  M = (-1, 6) → 
  M = ((C.1 + D.1) / 2, (C.2 + D.2) / 2) → 
  D.1 + D.2 = 1 := by
sorry

end NUMINAMATH_CALUDE_midpoint_coordinates_sum_l2187_218764


namespace NUMINAMATH_CALUDE_range_of_m_l2187_218763

-- Define the propositions p and q
def p (x : ℝ) : Prop := -2 ≤ 1 - (x - 1) / 3 ∧ 1 - (x - 1) / 3 ≤ 2

def q (x m : ℝ) : Prop := x^2 - 2*x + 1 - m^2 ≤ 0

-- Define the sets A and B
def A (m : ℝ) : Set ℝ := {x | ¬(q x m)}
def B : Set ℝ := {x | ¬(p x)}

-- State the theorem
theorem range_of_m (m : ℝ) :
  (m > 0) →
  (∀ x, ¬(p x) → ¬(q x m)) →
  (∃ x, ¬(p x) ∧ (q x m)) →
  m ≥ 6 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l2187_218763


namespace NUMINAMATH_CALUDE_sum_extrema_l2187_218727

theorem sum_extrema (x y z w : ℝ) (h_nonneg : x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧ w ≥ 0) 
  (h_eq : x^2 + y^2 + z^2 + w^2 + x + 2*y + 3*z + 4*w = 17/2) :
  (∀ a b c d : ℝ, a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 → 
    a^2 + b^2 + c^2 + d^2 + a + 2*b + 3*c + 4*d = 17/2 → 
    a + b + c + d ≤ 3) ∧
  (x + y + z + w ≥ -2 + 5/2 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_sum_extrema_l2187_218727


namespace NUMINAMATH_CALUDE_opposite_of_negative_2023_l2187_218743

theorem opposite_of_negative_2023 : 
  -((-2023 : ℤ)) = (2023 : ℤ) := by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_2023_l2187_218743


namespace NUMINAMATH_CALUDE_k_at_one_eq_neg_155_l2187_218776

/-- Polynomial h(x) -/
def h (p : ℝ) (x : ℝ) : ℝ := x^3 + p*x^2 + 2*x + 20

/-- Polynomial k(x) -/
def k (q r : ℝ) (x : ℝ) : ℝ := x^4 + 2*x^3 + q*x^2 + 50*x + r

/-- The theorem stating that k(1) = -155 given the conditions -/
theorem k_at_one_eq_neg_155 (p q r : ℝ) :
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ 
    h p x = 0 ∧ h p y = 0 ∧ h p z = 0 ∧
    k q r x = 0 ∧ k q r y = 0 ∧ k q r z = 0) →
  k q r 1 = -155 := by
  sorry

end NUMINAMATH_CALUDE_k_at_one_eq_neg_155_l2187_218776


namespace NUMINAMATH_CALUDE_pet_store_birds_l2187_218701

theorem pet_store_birds (total_birds talking_birds : ℕ) 
  (h1 : total_birds = 77)
  (h2 : talking_birds = 64) :
  total_birds - talking_birds = 13 := by
  sorry

end NUMINAMATH_CALUDE_pet_store_birds_l2187_218701


namespace NUMINAMATH_CALUDE_point_not_in_region_l2187_218702

/-- The plane region is represented by the inequality 3x + 2y < 6 -/
def in_plane_region (x y : ℝ) : Prop := 3 * x + 2 * y < 6

/-- The point (2, 0) is not in the plane region -/
theorem point_not_in_region : ¬ in_plane_region 2 0 := by
  sorry

end NUMINAMATH_CALUDE_point_not_in_region_l2187_218702


namespace NUMINAMATH_CALUDE_unique_rational_root_l2187_218752

def polynomial (x : ℚ) : ℚ := 3 * x^4 - 5 * x^3 - 8 * x^2 + 5 * x + 1

theorem unique_rational_root :
  ∀ x : ℚ, polynomial x = 0 ↔ x = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_unique_rational_root_l2187_218752


namespace NUMINAMATH_CALUDE_emma_age_when_sister_is_56_l2187_218742

theorem emma_age_when_sister_is_56 (emma_current_age : ℕ) (age_difference : ℕ) (sister_future_age : ℕ) :
  emma_current_age = 7 →
  age_difference = 9 →
  sister_future_age = 56 →
  sister_future_age - age_difference = 47 :=
by sorry

end NUMINAMATH_CALUDE_emma_age_when_sister_is_56_l2187_218742


namespace NUMINAMATH_CALUDE_graph_vertical_shift_l2187_218785

-- Define a real-valued function
variable (f : ℝ → ℝ)

-- Define the vertical translation
def verticalShift (f : ℝ → ℝ) (c : ℝ) : ℝ → ℝ := fun x ↦ f x + c

-- Theorem statement
theorem graph_vertical_shift :
  ∀ (x y : ℝ), y = f x ↔ y + 1 = verticalShift f 1 x := by
  sorry

end NUMINAMATH_CALUDE_graph_vertical_shift_l2187_218785


namespace NUMINAMATH_CALUDE_alpha_minus_beta_equals_pi_over_four_l2187_218791

open Real

theorem alpha_minus_beta_equals_pi_over_four
  (α β : ℝ)
  (h1 : 0 < α ∧ α < π/2)
  (h2 : 0 < β ∧ β < π/2)
  (h3 : tan α = 4/3)
  (h4 : tan β = 1/7) :
  α - β = π/4 := by
sorry

end NUMINAMATH_CALUDE_alpha_minus_beta_equals_pi_over_four_l2187_218791


namespace NUMINAMATH_CALUDE_smallest_integer_satisfying_inequality_l2187_218758

theorem smallest_integer_satisfying_inequality : 
  ∃ x : ℤ, (2 * x : ℚ) / 5 + 3 / 4 > 7 / 5 ∧ 
  ∀ y : ℤ, y < x → (2 * y : ℚ) / 5 + 3 / 4 ≤ 7 / 5 :=
by
  use 2
  sorry

end NUMINAMATH_CALUDE_smallest_integer_satisfying_inequality_l2187_218758


namespace NUMINAMATH_CALUDE_min_value_exponential_quadratic_min_value_achieved_at_zero_l2187_218746

theorem min_value_exponential_quadratic (x : ℝ) : 16^x - 2^x + x^2 + 1 ≥ 1 :=
by
  sorry

theorem min_value_achieved_at_zero : 16^0 - 2^0 + 0^2 + 1 = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_min_value_exponential_quadratic_min_value_achieved_at_zero_l2187_218746


namespace NUMINAMATH_CALUDE_paco_ate_five_sweet_cookies_l2187_218723

/-- Represents the number of cookies Paco had and ate -/
structure CookieCount where
  initial_sweet : Nat
  initial_salty : Nat
  eaten_salty : Nat
  sweet_salty_difference : Nat

/-- Calculates the number of sweet cookies Paco ate -/
def sweet_cookies_eaten (c : CookieCount) : Nat :=
  c.eaten_salty + c.sweet_salty_difference

/-- Theorem: Paco ate 5 sweet cookies -/
theorem paco_ate_five_sweet_cookies (c : CookieCount)
  (h1 : c.initial_sweet = 37)
  (h2 : c.initial_salty = 11)
  (h3 : c.eaten_salty = 2)
  (h4 : c.sweet_salty_difference = 3) :
  sweet_cookies_eaten c = 5 := by
  sorry

end NUMINAMATH_CALUDE_paco_ate_five_sweet_cookies_l2187_218723


namespace NUMINAMATH_CALUDE_price_restoration_l2187_218737

theorem price_restoration (original_price : ℝ) (h : original_price > 0) :
  let reduced_price := 0.8 * original_price
  (reduced_price * 1.25 = original_price) := by sorry

end NUMINAMATH_CALUDE_price_restoration_l2187_218737


namespace NUMINAMATH_CALUDE_part_one_part_two_l2187_218739

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - 1| + |x - a|

-- Part 1
theorem part_one :
  {x : ℝ | f (-1) x ≥ 3} = {x : ℝ | x ≤ -1.5 ∨ x ≥ 1.5} := by sorry

-- Part 2
theorem part_two :
  (∀ x : ℝ, f a x ≥ 2) ↔ (a = 3 ∨ a = -1) := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l2187_218739


namespace NUMINAMATH_CALUDE_cookie_process_time_l2187_218731

/-- Represents the cookie-making process with given times for each step -/
structure CookieProcess where
  total_time : ℕ
  baking_time : ℕ
  white_icing_time : ℕ
  chocolate_icing_time : ℕ

/-- Calculates the time to make dough and cool cookies -/
def dough_and_cooling_time (process : CookieProcess) : ℕ :=
  process.total_time - (process.baking_time + process.white_icing_time + process.chocolate_icing_time)

/-- Theorem stating that the time to make dough and cool cookies is 45 minutes -/
theorem cookie_process_time (process : CookieProcess) 
  (h1 : process.total_time = 120)
  (h2 : process.baking_time = 15)
  (h3 : process.white_icing_time = 30)
  (h4 : process.chocolate_icing_time = 30) : 
  dough_and_cooling_time process = 45 := by
  sorry

end NUMINAMATH_CALUDE_cookie_process_time_l2187_218731


namespace NUMINAMATH_CALUDE_base7_to_base10_65432_l2187_218777

/-- Converts a base-7 number represented as a list of digits to its base-10 equivalent -/
def base7ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (7 ^ i)) 0

/-- The base-7 representation of the number -/
def base7Number : List Nat := [2, 3, 4, 5, 6]

/-- Theorem stating that the base-10 equivalent of 65432 in base-7 is 16340 -/
theorem base7_to_base10_65432 :
  base7ToBase10 base7Number = 16340 := by
  sorry

end NUMINAMATH_CALUDE_base7_to_base10_65432_l2187_218777


namespace NUMINAMATH_CALUDE_sewer_capacity_l2187_218782

/-- The amount of run-off produced per hour of rain in gallons -/
def runoff_per_hour : ℕ := 1000

/-- The number of days the sewers can handle before overflow -/
def days_before_overflow : ℕ := 10

/-- The number of hours in a day -/
def hours_per_day : ℕ := 24

/-- The total gallons of run-off the sewers can handle -/
def total_runoff_capacity : ℕ := runoff_per_hour * days_before_overflow * hours_per_day

theorem sewer_capacity :
  total_runoff_capacity = 240000 := by
  sorry

end NUMINAMATH_CALUDE_sewer_capacity_l2187_218782


namespace NUMINAMATH_CALUDE_robyn_cookie_sales_l2187_218726

/-- Given that Robyn and Lucy sold a total of 98 packs of cookies,
    and Lucy sold 43 packs, prove that Robyn sold 55 packs. -/
theorem robyn_cookie_sales (total : ℕ) (lucy : ℕ) (robyn : ℕ)
    (h1 : total = 98)
    (h2 : lucy = 43)
    (h3 : total = lucy + robyn) :
  robyn = 55 := by
  sorry

end NUMINAMATH_CALUDE_robyn_cookie_sales_l2187_218726


namespace NUMINAMATH_CALUDE_unit_segment_construction_l2187_218773

theorem unit_segment_construction (a : ℝ) (h : a > 1) : (a / a^2) * a = 1 := by
  sorry

end NUMINAMATH_CALUDE_unit_segment_construction_l2187_218773


namespace NUMINAMATH_CALUDE_expression_value_l2187_218718

theorem expression_value : 
  let x : ℝ := 26
  let y : ℝ := 3 * x / 2
  let z : ℝ := 11
  (x - (y - z)) - ((x - y) - z) = 22 := by
sorry

end NUMINAMATH_CALUDE_expression_value_l2187_218718


namespace NUMINAMATH_CALUDE_lcm_gcf_product_l2187_218761

theorem lcm_gcf_product (a b : ℕ) (ha : a = 36) (hb : b = 48) :
  Nat.lcm a b * Nat.gcd a b = 1728 := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcf_product_l2187_218761


namespace NUMINAMATH_CALUDE_quadratic_surds_problem_l2187_218706

-- Define the variables and equations
theorem quadratic_surds_problem (x y : ℝ) 
  (hA : 5 * Real.sqrt (2 * x + 1) = 5 * Real.sqrt 5)
  (hB : 3 * Real.sqrt (x + 3) = 3 * Real.sqrt 5)
  (hC : Real.sqrt (10 * x + 3 * y) = Real.sqrt 320)
  (hAB : 5 * Real.sqrt (2 * x + 1) + 3 * Real.sqrt (x + 3) = Real.sqrt (10 * x + 3 * y)) :
  Real.sqrt (2 * y - x^2) = 14 := by
  sorry


end NUMINAMATH_CALUDE_quadratic_surds_problem_l2187_218706


namespace NUMINAMATH_CALUDE_sum_of_coordinates_X_l2187_218722

def Y : ℝ × ℝ := (2, 9)
def Z : ℝ × ℝ := (1, 5)

theorem sum_of_coordinates_X (X : ℝ × ℝ) 
  (h1 : (dist X Z) / (dist X Y) = 3 / 4)
  (h2 : (dist Z Y) / (dist X Y) = 1 / 4) : 
  X.1 + X.2 = -9 := by
  sorry

#check sum_of_coordinates_X

end NUMINAMATH_CALUDE_sum_of_coordinates_X_l2187_218722


namespace NUMINAMATH_CALUDE_sum_equals_three_halves_l2187_218741

theorem sum_equals_three_halves : 
  let original_sum := (1:ℚ)/3 + 1/5 + 1/7 + 1/9 + 1/11 + 1/13
  let reduced_sum := (1:ℚ)/3 + 1/7 + 1/9 + 1/11
  reduced_sum = 3/2 := by sorry

end NUMINAMATH_CALUDE_sum_equals_three_halves_l2187_218741


namespace NUMINAMATH_CALUDE_log_equality_l2187_218793

theorem log_equality (c d : ℝ) (hc : c = Real.log 625 / Real.log 4) (hd : d = Real.log 25 / Real.log 5) :
  c = d :=
by sorry

end NUMINAMATH_CALUDE_log_equality_l2187_218793


namespace NUMINAMATH_CALUDE_partner_a_income_increase_l2187_218719

/-- Represents the increase in partner a's income when the profit rate changes --/
def income_increase (capital : ℝ) (initial_rate final_rate : ℝ) (share : ℝ) : ℝ :=
  share * (final_rate - initial_rate) * capital

/-- Theorem stating the increase in partner a's income given the problem conditions --/
theorem partner_a_income_increase :
  let capital : ℝ := 10000
  let initial_rate : ℝ := 0.05
  let final_rate : ℝ := 0.07
  let share : ℝ := 2/3
  income_increase capital initial_rate final_rate share = 400/3 := by sorry

end NUMINAMATH_CALUDE_partner_a_income_increase_l2187_218719


namespace NUMINAMATH_CALUDE_cat_mouse_problem_l2187_218792

/-- Given that 5 cats can catch 5 mice in 5 minutes, prove that 5 cats can catch 100 mice in 500 minutes -/
theorem cat_mouse_problem (cats mice minutes : ℕ) 
  (h1 : cats = 5)
  (h2 : mice = 5)
  (h3 : minutes = 5)
  (h4 : cats * mice = cats * minutes) : 
  cats * 100 = cats * 500 := by
  sorry

end NUMINAMATH_CALUDE_cat_mouse_problem_l2187_218792


namespace NUMINAMATH_CALUDE_power_function_is_odd_l2187_218765

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


end NUMINAMATH_CALUDE_power_function_is_odd_l2187_218765


namespace NUMINAMATH_CALUDE_letters_written_in_ten_hours_l2187_218787

/-- The number of letters Nathan can write in one hour -/
def nathanRate : ℕ := 25

/-- The number of letters Jacob can write in one hour -/
def jacobRate : ℕ := 2 * nathanRate

/-- The number of hours they write together -/
def totalHours : ℕ := 10

/-- The total number of letters Jacob and Nathan can write together in the given time -/
def totalLetters : ℕ := (nathanRate + jacobRate) * totalHours

theorem letters_written_in_ten_hours : totalLetters = 750 := by
  sorry

end NUMINAMATH_CALUDE_letters_written_in_ten_hours_l2187_218787


namespace NUMINAMATH_CALUDE_set_equality_implies_a_equals_three_l2187_218786

theorem set_equality_implies_a_equals_three (a : ℝ) : 
  ({0, 1, a^2} : Set ℝ) = ({1, 0, 2*a + 3} : Set ℝ) → a = 3 := by
  sorry

end NUMINAMATH_CALUDE_set_equality_implies_a_equals_three_l2187_218786


namespace NUMINAMATH_CALUDE_line_mb_value_l2187_218759

/-- Proves that for a line y = mx + b passing through points (0, -2) and (1, 1), mb = -10 -/
theorem line_mb_value (m b : ℝ) : 
  (∀ x y : ℝ, y = m * x + b) → -- Line equation
  (-2 : ℝ) = b →               -- y-intercept
  (1 : ℝ) = m * 1 + b →        -- Point (1, 1) satisfies the equation
  m * b = -10 := by
  sorry

end NUMINAMATH_CALUDE_line_mb_value_l2187_218759


namespace NUMINAMATH_CALUDE_disneyland_arrangements_l2187_218789

theorem disneyland_arrangements (n : ℕ) (k : ℕ) : n = 7 → k = 2 → n.factorial * k^n = 645120 := by
  sorry

end NUMINAMATH_CALUDE_disneyland_arrangements_l2187_218789


namespace NUMINAMATH_CALUDE_intersection_line_equation_l2187_218700

/-- Given two lines l₁ and l₂, if a line l intersects both l₁ and l₂ such that the midpoint
    of the segment cut off by l₁ and l₂ is at the origin, then l has the equation x + 6y = 0 -/
theorem intersection_line_equation (x y : ℝ) : 
  let l₁ := {(x, y) : ℝ × ℝ | 4*x + y + 6 = 0}
  let l₂ := {(x, y) : ℝ × ℝ | 3*x - 5*y - 6 = 0}
  let midpoint := (0, 0)
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    (x₁, y₁) ∈ l₁ ∧ (x₂, y₂) ∈ l₂ ∧
    (x₁ + x₂) / 2 = midpoint.1 ∧ (y₁ + y₂) / 2 = midpoint.2 →
  x + 6*y = 0 := by
sorry


end NUMINAMATH_CALUDE_intersection_line_equation_l2187_218700


namespace NUMINAMATH_CALUDE_work_completion_l2187_218774

theorem work_completion (days1 days2 men2 : ℕ) 
  (h1 : days1 > 0)
  (h2 : days2 > 0)
  (h3 : men2 > 0)
  (h4 : days1 = 80)
  (h5 : days2 = 48)
  (h6 : men2 = 20)
  (h7 : ∃ men1 : ℕ, men1 * days1 = men2 * days2) :
  ∃ men1 : ℕ, men1 = 12 ∧ men1 * days1 = men2 * days2 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_l2187_218774


namespace NUMINAMATH_CALUDE_max_grain_mass_on_platform_l2187_218795

/-- Represents a rectangular platform with grain piled on it. -/
structure GrainPlatform where
  length : ℝ
  width : ℝ
  grainDensity : ℝ
  maxAngle : ℝ

/-- Calculates the maximum mass of grain on the platform. -/
def maxGrainMass (platform : GrainPlatform) : ℝ :=
  sorry

/-- Theorem stating the maximum mass of grain on the given platform. -/
theorem max_grain_mass_on_platform :
  let platform : GrainPlatform := {
    length := 8,
    width := 5,
    grainDensity := 1200,
    maxAngle := π/4
  }
  maxGrainMass platform = 47500 := by sorry

end NUMINAMATH_CALUDE_max_grain_mass_on_platform_l2187_218795


namespace NUMINAMATH_CALUDE_income_calculation_l2187_218772

/-- Represents a person's financial situation -/
structure FinancialSituation where
  income : ℕ
  expenditure : ℕ
  savings : ℕ

/-- Proves that given the conditions, the person's income is 10000 -/
theorem income_calculation (f : FinancialSituation) 
  (h1 : f.income * 8 = f.expenditure * 10)  -- income : expenditure = 10 : 8
  (h2 : f.savings = 2000)                   -- savings are 2000
  (h3 : f.income = f.expenditure + f.savings) -- income = expenditure + savings
  : f.income = 10000 := by
  sorry


end NUMINAMATH_CALUDE_income_calculation_l2187_218772


namespace NUMINAMATH_CALUDE_max_visible_time_l2187_218778

/-- The maximum time two people can see each other on a circular track with an obstacle -/
theorem max_visible_time (track_radius : ℝ) (obstacle_radius : ℝ) (speed1 : ℝ) (speed2 : ℝ) 
  (h1 : track_radius = 60)
  (h2 : obstacle_radius = 30)
  (h3 : speed1 = 0.4)
  (h4 : speed2 = 0.2) :
  (track_radius * (2 * Real.pi / 3)) / (speed1 - speed2) = 200 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_max_visible_time_l2187_218778


namespace NUMINAMATH_CALUDE_inscribed_quadrilateral_theorem_l2187_218734

/-- A quadrilateral inscribed in a circle -/
structure InscribedQuadrilateral :=
  (radius : ℝ)
  (side1 : ℝ)
  (side2 : ℝ)
  (side3 : ℝ)
  (side4 : ℝ)

/-- The theorem about the inscribed quadrilateral -/
theorem inscribed_quadrilateral_theorem (q : InscribedQuadrilateral) :
  q.radius = 300 ∧ q.side1 = 300 ∧ q.side2 = 300 ∧ q.side3 = 200 →
  q.side4 = 300 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_quadrilateral_theorem_l2187_218734


namespace NUMINAMATH_CALUDE_sum_of_products_l2187_218768

theorem sum_of_products (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hab : a * b = 36) (hac : a * c = 72) (hbc : b * c = 108) :
  a + b + c = 11 * Real.sqrt 6 := by
sorry

end NUMINAMATH_CALUDE_sum_of_products_l2187_218768


namespace NUMINAMATH_CALUDE_grain_depot_analysis_l2187_218744

def grain_movements : List Int := [25, -31, -16, 33, -36, -20]
def fee_per_ton : ℕ := 5

theorem grain_depot_analysis :
  (List.sum grain_movements = -45) ∧
  (List.sum (List.map (λ x => fee_per_ton * x.natAbs) grain_movements) = 805) := by
  sorry

end NUMINAMATH_CALUDE_grain_depot_analysis_l2187_218744


namespace NUMINAMATH_CALUDE_line_equation_with_intercept_condition_l2187_218770

/-- The equation of a line passing through the intersection of two given lines,
    with its y-intercept being twice its x-intercept. -/
theorem line_equation_with_intercept_condition :
  ∃ (m b : ℝ),
    (∀ x y : ℝ, (2*x + y = 8 ∧ x - 2*y = -1) → (m*x + b = y)) ∧
    (2 * (b/m) = b) ∧
    ((m = 2 ∧ b = 0) ∨ (m = 2 ∧ b = -8)) := by
  sorry

end NUMINAMATH_CALUDE_line_equation_with_intercept_condition_l2187_218770


namespace NUMINAMATH_CALUDE_divisibility_by_13_l2187_218779

theorem divisibility_by_13 (x y : ℤ) 
  (h1 : (x^2 - 3*x*y + 2*y^2 + x - y) % 13 = 0)
  (h2 : (x^2 - 2*x*y + y^2 - 5*x + 7) % 13 = 0) :
  (x*y - 12*x + 15*y) % 13 = 0 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_13_l2187_218779


namespace NUMINAMATH_CALUDE_volunteer_comprehensive_score_l2187_218750

/-- Calculates the comprehensive score of a volunteer guide based on test scores and weights -/
def comprehensive_score (written_score trial_score interview_score : ℝ)
  (written_weight trial_weight interview_weight : ℝ) : ℝ :=
  written_score * written_weight + trial_score * trial_weight + interview_score * interview_weight

/-- Theorem stating that the comprehensive score of the volunteer guide is 92.4 points -/
theorem volunteer_comprehensive_score :
  comprehensive_score 90 94 92 0.3 0.5 0.2 = 92.4 := by
  sorry

end NUMINAMATH_CALUDE_volunteer_comprehensive_score_l2187_218750


namespace NUMINAMATH_CALUDE_certain_number_equation_l2187_218755

theorem certain_number_equation : ∃ x : ℚ, (55 / 100) * 40 = (4 / 5) * x + 2 :=
by
  -- Proof goes here
  sorry

#check certain_number_equation

end NUMINAMATH_CALUDE_certain_number_equation_l2187_218755


namespace NUMINAMATH_CALUDE_coin_draw_probability_l2187_218748

def penny_count : ℕ := 3
def nickel_count : ℕ := 3
def quarter_count : ℕ := 6
def dime_count : ℕ := 3
def total_coins : ℕ := penny_count + nickel_count + quarter_count + dime_count
def drawn_coins : ℕ := 8
def min_value : ℚ := 175/100

def successful_outcomes : ℕ := 9
def total_outcomes : ℕ := Nat.choose total_coins drawn_coins

theorem coin_draw_probability :
  (successful_outcomes : ℚ) / total_outcomes = 9 / 6435 :=
sorry

end NUMINAMATH_CALUDE_coin_draw_probability_l2187_218748


namespace NUMINAMATH_CALUDE_sequence_not_convergent_l2187_218721

theorem sequence_not_convergent (x : ℕ → ℝ) (a : ℝ) :
  (∀ ε > 0, ∀ k, ∃ n > k, |x n - a| ≥ ε) →
  ¬ (∀ ε > 0, ∃ N, ∀ n ≥ N, |x n - a| < ε) :=
by sorry

end NUMINAMATH_CALUDE_sequence_not_convergent_l2187_218721


namespace NUMINAMATH_CALUDE_expression_factorization_l2187_218729

theorem expression_factorization (x y z : ℝ) :
  x^3 * (y^2 - z^2) + y^3 * (z^2 - x^2) + z^3 * (x^2 - y^2) =
  (x - y) * (y - z) * (z - x) * (-(x*y + x*z + y*z)) := by sorry

end NUMINAMATH_CALUDE_expression_factorization_l2187_218729


namespace NUMINAMATH_CALUDE_parabola_focus_l2187_218753

/-- Given a parabola y = ax² passing through (1, 4), its focus is at (0, 1/16) -/
theorem parabola_focus (a : ℝ) : 
  (4 = a * 1^2) → -- Parabola passes through (1, 4)
  let f : ℝ × ℝ := (0, 1/16) -- Define focus coordinates
  (∀ x y : ℝ, y = a * x^2 → -- For all points (x, y) on the parabola
    (x - f.1)^2 = 4 * (1/(4*a)) * (y - f.2)) -- Satisfy the focus-directrix property
  := by sorry

end NUMINAMATH_CALUDE_parabola_focus_l2187_218753


namespace NUMINAMATH_CALUDE_students_passed_both_tests_l2187_218749

theorem students_passed_both_tests 
  (total : ℕ) 
  (passed_long_jump : ℕ) 
  (passed_shot_put : ℕ) 
  (failed_both : ℕ) 
  (h1 : total = 50)
  (h2 : passed_long_jump = 40)
  (h3 : passed_shot_put = 31)
  (h4 : failed_both = 4) :
  ∃ (passed_both : ℕ), 
    passed_both = 25 ∧ 
    total = passed_both + (passed_long_jump - passed_both) + (passed_shot_put - passed_both) + failed_both :=
by sorry

end NUMINAMATH_CALUDE_students_passed_both_tests_l2187_218749


namespace NUMINAMATH_CALUDE_subtraction_equality_l2187_218714

theorem subtraction_equality : 3.65 - 2.27 - 0.48 = 0.90 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_equality_l2187_218714


namespace NUMINAMATH_CALUDE_opposite_terminal_sides_sin_equality_l2187_218735

theorem opposite_terminal_sides_sin_equality (α β : Real) : 
  (∃ k : Int, β = α + (2 * k + 1) * Real.pi) → |Real.sin α| = |Real.sin β| := by
  sorry

end NUMINAMATH_CALUDE_opposite_terminal_sides_sin_equality_l2187_218735


namespace NUMINAMATH_CALUDE_mod_inverse_sum_five_l2187_218705

theorem mod_inverse_sum_five : ∃ (a b : ℤ), 
  (5 * a) % 17 = 1 ∧ 
  (5^2 * b) % 17 = 1 ∧ 
  (a + b) % 17 = 14 := by
sorry

end NUMINAMATH_CALUDE_mod_inverse_sum_five_l2187_218705


namespace NUMINAMATH_CALUDE_compton_basketball_league_members_l2187_218780

theorem compton_basketball_league_members : 
  let sock_cost : ℚ := 4
  let tshirt_cost : ℚ := sock_cost + 6
  let cap_cost : ℚ := tshirt_cost - 3
  let member_cost : ℚ := 2 * (sock_cost + tshirt_cost + cap_cost)
  let total_expenditure : ℚ := 3144
  (total_expenditure / member_cost : ℚ) = 75 := by
  sorry

end NUMINAMATH_CALUDE_compton_basketball_league_members_l2187_218780


namespace NUMINAMATH_CALUDE_hat_cost_l2187_218767

theorem hat_cost (initial_amount : ℕ) (num_sock_pairs : ℕ) (cost_per_sock_pair : ℕ) (amount_left : ℕ) : 
  initial_amount = 20 ∧ 
  num_sock_pairs = 4 ∧ 
  cost_per_sock_pair = 2 ∧ 
  amount_left = 5 → 
  initial_amount - (num_sock_pairs * cost_per_sock_pair) - amount_left = 7 :=
by sorry

end NUMINAMATH_CALUDE_hat_cost_l2187_218767


namespace NUMINAMATH_CALUDE_max_silver_tokens_l2187_218703

/-- Represents the state of tokens -/
structure TokenState where
  red : ℕ
  blue : ℕ
  silver : ℕ

/-- Represents an exchange at a booth -/
inductive Exchange
  | First  : Exchange
  | Second : Exchange

/-- Applies an exchange to a token state -/
def applyExchange (s : TokenState) (e : Exchange) : Option TokenState :=
  match e with
  | Exchange.First =>
      if s.red ≥ 3 then
        some { red := s.red - 3, blue := s.blue + 2, silver := s.silver + 1 }
      else
        none
  | Exchange.Second =>
      if s.blue ≥ 4 then
        some { red := s.red + 2, blue := s.blue - 4, silver := s.silver + 1 }
      else
        none

/-- Theorem: The maximum number of silver tokens Alex can obtain is 39 -/
theorem max_silver_tokens (initialState : TokenState)
    (h_initial_red : initialState.red = 100)
    (h_initial_blue : initialState.blue = 90)
    (h_initial_silver : initialState.silver = 0) :
    (∃ (finalState : TokenState),
      (∀ e : Exchange, applyExchange finalState e = none) ∧
      finalState.silver = 39 ∧
      (∀ otherState : TokenState,
        (∀ e : Exchange, applyExchange otherState e = none) →
        otherState.silver ≤ finalState.silver)) :=
  sorry


end NUMINAMATH_CALUDE_max_silver_tokens_l2187_218703


namespace NUMINAMATH_CALUDE_fraction_simplification_l2187_218783

theorem fraction_simplification : (200 + 10) / (20 + 10) = 7 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2187_218783


namespace NUMINAMATH_CALUDE_lily_trip_distance_l2187_218712

/-- Represents Lily's car and trip details -/
structure CarTrip where
  /-- Miles per gallon of the car -/
  mpg : ℝ
  /-- Capacity of the gas tank in gallons -/
  tank_capacity : ℝ
  /-- Initial distance driven in miles -/
  initial_distance : ℝ
  /-- First gas purchase in gallons -/
  first_gas_purchase : ℝ
  /-- Second gas purchase in gallons -/
  second_gas_purchase : ℝ
  /-- Fraction of tank full at arrival -/
  final_tank_fraction : ℝ

/-- Calculates the total distance driven given the car trip details -/
def total_distance (trip : CarTrip) : ℝ :=
  trip.initial_distance +
  trip.first_gas_purchase * trip.mpg +
  (trip.second_gas_purchase + trip.final_tank_fraction * trip.tank_capacity - trip.tank_capacity) * trip.mpg

/-- Theorem stating that Lily's total distance driven is 880 miles -/
theorem lily_trip_distance :
  let trip : CarTrip := {
    mpg := 40,
    tank_capacity := 12,
    initial_distance := 480,
    first_gas_purchase := 6,
    second_gas_purchase := 4,
    final_tank_fraction := 3/4
  }
  total_distance trip = 880 := by sorry

end NUMINAMATH_CALUDE_lily_trip_distance_l2187_218712


namespace NUMINAMATH_CALUDE_least_integer_with_twelve_factors_l2187_218732

theorem least_integer_with_twelve_factors : 
  ∀ n : ℕ, n > 0 → (∃ f : Finset ℕ, f = {d : ℕ | d ∣ n ∧ d > 0} ∧ f.card = 12) → n ≥ 60 :=
sorry

end NUMINAMATH_CALUDE_least_integer_with_twelve_factors_l2187_218732


namespace NUMINAMATH_CALUDE_choir_composition_theorem_l2187_218730

/-- Represents the choir composition and ratio changes -/
structure ChoirComposition where
  b : ℝ  -- Initial number of blonde girls
  x : ℝ  -- Number of blonde girls added

/-- Theorem about the choir composition changes -/
theorem choir_composition_theorem (choir : ChoirComposition) :
  -- Initial ratio of blonde to black-haired girls is 3:5
  (choir.b) / ((5/3) * choir.b) = 3/5 →
  -- After adding x blonde girls, the ratio becomes 3:2
  (choir.b + choir.x) / ((5/3) * choir.b) = 3/2 →
  -- The final number of black-haired girls is (5/3)b
  (5/3) * choir.b = (5/3) * choir.b ∧
  -- The relationship between x and b is x = (3/2)b
  choir.x = (3/2) * choir.b :=
by sorry

end NUMINAMATH_CALUDE_choir_composition_theorem_l2187_218730


namespace NUMINAMATH_CALUDE_white_squares_20th_row_l2187_218798

/-- Represents the number of squares in a row of the stair-step figure -/
def squares_in_row (n : ℕ) : ℕ := 2 * n + 1

/-- Represents the number of white squares in a row of the stair-step figure -/
def white_squares_in_row (n : ℕ) : ℕ := (squares_in_row n - 1) / 2

theorem white_squares_20th_row :
  white_squares_in_row 20 = 20 := by
  sorry

#eval white_squares_in_row 20

end NUMINAMATH_CALUDE_white_squares_20th_row_l2187_218798


namespace NUMINAMATH_CALUDE_expression_bounds_l2187_218704

theorem expression_bounds (x y z w : Real) 
  (hx : 0 ≤ x ∧ x ≤ 1) (hy : 0 ≤ y ∧ y ≤ 1) 
  (hz : 0 ≤ z ∧ z ≤ 1) (hw : 0 ≤ w ∧ w ≤ 1) : 
  let expr := Real.sqrt (x^2 + (1-y)^2) + Real.sqrt (y^2 + (1-z)^2) + 
              Real.sqrt (z^2 + (1-w)^2) + Real.sqrt (w^2 + (1-x)^2)
  2 * Real.sqrt 2 ≤ expr ∧ expr ≤ 4 ∧ 
  (∃ x y z w, (0 ≤ x ∧ x ≤ 1) ∧ (0 ≤ y ∧ y ≤ 1) ∧ (0 ≤ z ∧ z ≤ 1) ∧ (0 ≤ w ∧ w ≤ 1) ∧ 
              expr = 2 * Real.sqrt 2) ∧
  (∃ x y z w, (0 ≤ x ∧ x ≤ 1) ∧ (0 ≤ y ∧ y ≤ 1) ∧ (0 ≤ z ∧ z ≤ 1) ∧ (0 ≤ w ∧ w ≤ 1) ∧ 
              expr = 4) := by
  sorry

end NUMINAMATH_CALUDE_expression_bounds_l2187_218704


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2187_218784

/-- Arithmetic sequence with first term a₁ and common difference d -/
def arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + d * (n - 1)

theorem arithmetic_sequence_problem (a₁ d : ℝ) :
  arithmetic_sequence a₁ d 11 = -26 →
  arithmetic_sequence a₁ d 51 = 54 →
  (arithmetic_sequence a₁ d 14 = -20) ∧
  (∀ n : ℕ, n < 25 → arithmetic_sequence a₁ d n ≤ 0) ∧
  (arithmetic_sequence a₁ d 25 > 0) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2187_218784


namespace NUMINAMATH_CALUDE_factor_x10_minus_1024_l2187_218738

theorem factor_x10_minus_1024 (x : ℝ) : x^10 - 1024 = (x^5 + 32) * (x^5 - 32) := by
  sorry

end NUMINAMATH_CALUDE_factor_x10_minus_1024_l2187_218738


namespace NUMINAMATH_CALUDE_least_negative_b_for_integer_solutions_l2187_218740

theorem least_negative_b_for_integer_solutions (x b : ℤ) : 
  (∃ x : ℤ, x^2 + b*x = 22) → 
  b < 0 → 
  (∀ b' : ℤ, b' < b → ¬∃ x : ℤ, x^2 + b'*x = 22) →
  b = -21 :=
by sorry

end NUMINAMATH_CALUDE_least_negative_b_for_integer_solutions_l2187_218740


namespace NUMINAMATH_CALUDE_sum_of_max_and_min_is_two_l2187_218707

-- Define the function f
def f (x : ℝ) : ℝ := |x - 2| + |x - 4| - |2*x - 6|

-- Define the domain
def domain : Set ℝ := { x | 2 ≤ x ∧ x ≤ 8 }

-- State the theorem
theorem sum_of_max_and_min_is_two :
  ∃ (max min : ℝ), 
    (∀ x ∈ domain, f x ≤ max) ∧
    (∃ x ∈ domain, f x = max) ∧
    (∀ x ∈ domain, min ≤ f x) ∧
    (∃ x ∈ domain, f x = min) ∧
    max + min = 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_max_and_min_is_two_l2187_218707


namespace NUMINAMATH_CALUDE_total_earrings_l2187_218781

theorem total_earrings (bella_earrings : ℕ) (monica_earrings : ℕ) (rachel_earrings : ℕ) 
  (h1 : bella_earrings = 10)
  (h2 : bella_earrings = monica_earrings / 4)
  (h3 : monica_earrings = 2 * rachel_earrings) : 
  bella_earrings + monica_earrings + rachel_earrings = 70 := by
  sorry

end NUMINAMATH_CALUDE_total_earrings_l2187_218781


namespace NUMINAMATH_CALUDE_binary_sum_equality_l2187_218799

/-- Converts a list of bits to a natural number -/
def bitsToNat (bits : List Bool) : ℕ :=
  bits.foldr (fun b n => 2 * n + if b then 1 else 0) 0

/-- The sum of the given binary numbers is equal to 11110011₂ -/
theorem binary_sum_equality : 
  let a := bitsToNat [true, false, true, false, true]
  let b := bitsToNat [true, false, true, true]
  let c := bitsToNat [true, true, true, false, false]
  let d := bitsToNat [true, false, true, false, true, false, true]
  let sum := bitsToNat [true, true, true, true, false, false, true, true]
  a + b + c + d = sum := by
  sorry

end NUMINAMATH_CALUDE_binary_sum_equality_l2187_218799


namespace NUMINAMATH_CALUDE_negation_of_p_l2187_218733

def p : Prop := ∀ x : ℝ, Real.sqrt (2 - x) < 0

theorem negation_of_p : ¬p ↔ ∃ x₀ : ℝ, Real.sqrt (2 - x₀) ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_negation_of_p_l2187_218733


namespace NUMINAMATH_CALUDE_trigonometric_identity_l2187_218769

theorem trigonometric_identity :
  1 / Real.cos (70 * π / 180) - 2 / Real.sin (70 * π / 180) = 
  4 * Real.sin (10 * π / 180) / Real.sin (40 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l2187_218769


namespace NUMINAMATH_CALUDE_sixth_root_unity_product_l2187_218757

theorem sixth_root_unity_product (s : ℂ) (h1 : s^6 = 1) (h2 : s ≠ 1) :
  (s - 1) * (s^2 - 1) * (s^3 - 1) * (s^4 - 1) * (s^5 - 1) = 0 := by
  sorry

end NUMINAMATH_CALUDE_sixth_root_unity_product_l2187_218757


namespace NUMINAMATH_CALUDE_symmetric_points_sum_power_l2187_218771

theorem symmetric_points_sum_power (a b : ℝ) : 
  (∃ (P1 P2 : ℝ × ℝ), 
    P1 = (a - 1, 5) ∧ 
    P2 = (2, b - 1) ∧ 
    P1.1 = P2.1 ∧ 
    P1.2 = -P2.2) →
  (a + b)^2016 = 1 := by
sorry

end NUMINAMATH_CALUDE_symmetric_points_sum_power_l2187_218771


namespace NUMINAMATH_CALUDE_gcd_n_cubed_plus_16_and_n_plus_4_l2187_218708

theorem gcd_n_cubed_plus_16_and_n_plus_4 (n : ℕ) (h : n > 2^4) :
  Nat.gcd (n^3 + 4^2) (n + 4) = Nat.gcd 48 (n + 4) := by
  sorry

end NUMINAMATH_CALUDE_gcd_n_cubed_plus_16_and_n_plus_4_l2187_218708


namespace NUMINAMATH_CALUDE_defective_units_shipped_percentage_l2187_218713

theorem defective_units_shipped_percentage
  (total_units : ℝ)
  (defective_percentage : ℝ)
  (defective_shipped_percentage : ℝ)
  (h1 : defective_percentage = 8)
  (h2 : defective_shipped_percentage = 0.4) :
  (defective_shipped_percentage / defective_percentage) * 100 = 5 := by
sorry

end NUMINAMATH_CALUDE_defective_units_shipped_percentage_l2187_218713


namespace NUMINAMATH_CALUDE_hyperbola_k_range_l2187_218720

/-- A hyperbola is represented by the equation (x^2 / (k-2)) + (y^2 / (5-k)) = 1 -/
def is_hyperbola (k : ℝ) : Prop :=
  ∃ x y : ℝ, (x^2 / (k-2)) + (y^2 / (5-k)) = 1 ∧ (k-2) * (5-k) < 0

/-- The range of k for which the equation represents a hyperbola -/
theorem hyperbola_k_range :
  ∀ k : ℝ, is_hyperbola k ↔ k < 2 ∨ k > 5 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_k_range_l2187_218720


namespace NUMINAMATH_CALUDE_ball_arrangements_count_l2187_218736

variable (n : ℕ)

/-- The number of ways to arrange 3n people into circles with ABC pattern -/
def ball_arrangements (n : ℕ) : ℕ := (3 * n).factorial

/-- Theorem: The number of ball arrangements is (3n)! -/
theorem ball_arrangements_count :
  ball_arrangements n = (3 * n).factorial := by
  sorry

end NUMINAMATH_CALUDE_ball_arrangements_count_l2187_218736


namespace NUMINAMATH_CALUDE_rectangle_square_problem_l2187_218711

/-- Given a rectangle with length-to-width ratio 2:1 and area 50 cm², 
    and a square with the same area as the rectangle, prove:
    1. The rectangle's length is 10 cm and width is 5 cm
    2. The difference between the square's side length and the rectangle's width is 5(√2 - 1) cm -/
theorem rectangle_square_problem (length width : ℝ) (square_side : ℝ) : 
  length = 2 * width → 
  length * width = 50 → 
  square_side^2 = 50 → 
  (length = 10 ∧ width = 5) ∧ 
  square_side - width = 5 * (Real.sqrt 2 - 1) := by
  sorry

end NUMINAMATH_CALUDE_rectangle_square_problem_l2187_218711


namespace NUMINAMATH_CALUDE_total_chapters_read_l2187_218775

def number_of_books : ℕ := 12
def chapters_per_book : ℕ := 32

theorem total_chapters_read : number_of_books * chapters_per_book = 384 := by
  sorry

end NUMINAMATH_CALUDE_total_chapters_read_l2187_218775


namespace NUMINAMATH_CALUDE_loan_repayment_theorem_l2187_218790

/-- Calculates the lump sum payment for a loan with given parameters -/
def lump_sum_payment (
  principal : ℝ)  -- Initial loan amount
  (rate : ℝ)      -- Annual interest rate as a decimal
  (num_payments : ℕ) -- Total number of annuity payments
  (delay : ℕ)     -- Years before first payment
  (payments_made : ℕ) -- Number of payments made before death
  (years_after_death : ℕ) -- Years after death until lump sum payment
  : ℝ :=
  sorry

theorem loan_repayment_theorem :
  let principal := 20000
  let rate := 0.04
  let num_payments := 10
  let delay := 3
  let payments_made := 5
  let years_after_death := 2
  abs (lump_sum_payment principal rate num_payments delay payments_made years_after_death - 119804.6) < 1 :=
sorry

end NUMINAMATH_CALUDE_loan_repayment_theorem_l2187_218790


namespace NUMINAMATH_CALUDE_apples_left_l2187_218762

/-- Proves that given the conditions in the problem, the number of boxes of apples left is 3 -/
theorem apples_left (saturday_boxes : Nat) (sunday_boxes : Nat) (apples_per_box : Nat) (apples_sold : Nat) : Nat :=
  by
  -- Define the given conditions
  have h1 : saturday_boxes = 50 := by sorry
  have h2 : sunday_boxes = 25 := by sorry
  have h3 : apples_per_box = 10 := by sorry
  have h4 : apples_sold = 720 := by sorry

  -- Calculate the total number of boxes
  let total_boxes := saturday_boxes + sunday_boxes

  -- Calculate the total number of apples initially
  let total_apples := total_boxes * apples_per_box

  -- Calculate the number of apples left
  let apples_left := total_apples - apples_sold

  -- Calculate the number of boxes left
  let boxes_left := apples_left / apples_per_box

  -- Prove that the number of boxes left is 3
  have h5 : boxes_left = 3 := by sorry

  exact boxes_left

end NUMINAMATH_CALUDE_apples_left_l2187_218762


namespace NUMINAMATH_CALUDE_gcd_factorial_eight_and_factorial_six_squared_l2187_218766

theorem gcd_factorial_eight_and_factorial_six_squared : 
  Nat.gcd (Nat.factorial 8) ((Nat.factorial 6)^2) = 11520 := by
  sorry

end NUMINAMATH_CALUDE_gcd_factorial_eight_and_factorial_six_squared_l2187_218766


namespace NUMINAMATH_CALUDE_calculation_proof_l2187_218725

theorem calculation_proof : (1.2 : ℝ)^3 - (0.9 : ℝ)^3 / (1.2 : ℝ)^2 + 1.08 + (0.9 : ℝ)^2 = 3.11175 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l2187_218725


namespace NUMINAMATH_CALUDE_infinite_sequence_exists_l2187_218745

-- Define the Ω function
def Omega (n : ℕ+) : ℕ := sorry

-- Define the f function
def f (n : ℕ+) : Int := (-1) ^ (Omega n)

-- State the theorem
theorem infinite_sequence_exists : 
  ∃ (seq : ℕ → ℕ+), (∀ i : ℕ, 
    f (seq i - 1) = 1 ∧ 
    f (seq i) = 1 ∧ 
    f (seq i + 1) = 1) ∧ 
  (∀ i j : ℕ, i ≠ j → seq i ≠ seq j) :=
sorry

end NUMINAMATH_CALUDE_infinite_sequence_exists_l2187_218745


namespace NUMINAMATH_CALUDE_expand_expression_l2187_218717

theorem expand_expression (x : ℝ) : (3 * x + 5) * (4 * x - 2) = 12 * x^2 + 14 * x - 10 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l2187_218717
