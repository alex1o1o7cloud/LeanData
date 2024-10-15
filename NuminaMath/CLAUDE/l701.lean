import Mathlib

namespace NUMINAMATH_CALUDE_sally_total_spent_l701_70188

/-- The total amount Sally spent on peaches and cherries -/
def total_spent (peach_price_after_coupon : ℚ) (cherry_price : ℚ) : ℚ :=
  peach_price_after_coupon + cherry_price

/-- Theorem stating that Sally spent $23.86 in total -/
theorem sally_total_spent : 
  total_spent 12.32 11.54 = 23.86 := by
  sorry

end NUMINAMATH_CALUDE_sally_total_spent_l701_70188


namespace NUMINAMATH_CALUDE_checkers_tie_fraction_l701_70103

theorem checkers_tie_fraction (ben_win_rate sara_win_rate : ℚ) 
  (h1 : ben_win_rate = 2/5)
  (h2 : sara_win_rate = 1/4) : 
  1 - (ben_win_rate + sara_win_rate) = 7/20 := by
sorry

end NUMINAMATH_CALUDE_checkers_tie_fraction_l701_70103


namespace NUMINAMATH_CALUDE_min_colors_for_subdivided_rectangle_l701_70112

/-- Represents an infinitely subdivided rectangle according to the given pattern. -/
structure InfinitelySubdividedRectangle where
  -- Add necessary fields here
  -- This is left abstract as the exact representation is not crucial for the theorem

/-- The minimum number of colors needed so that no two rectangles sharing an edge have the same color. -/
def minEdgeColors (r : InfinitelySubdividedRectangle) : ℕ := 3

/-- The minimum number of colors needed so that no two rectangles sharing a corner have the same color. -/
def minCornerColors (r : InfinitelySubdividedRectangle) : ℕ := 4

/-- Theorem stating the minimum number of colors needed for edge and corner coloring. -/
theorem min_colors_for_subdivided_rectangle (r : InfinitelySubdividedRectangle) :
  (minEdgeColors r, minCornerColors r) = (3, 4) := by sorry

end NUMINAMATH_CALUDE_min_colors_for_subdivided_rectangle_l701_70112


namespace NUMINAMATH_CALUDE_x_n_prime_iff_n_eq_two_l701_70122

/-- Definition of x_n as a number of the form 10101...1 with n ones -/
def x_n (n : ℕ) : ℕ := (10^(2*n) - 1) / 99

/-- Theorem stating that x_n is prime only when n = 2 -/
theorem x_n_prime_iff_n_eq_two :
  ∀ n : ℕ, Nat.Prime (x_n n) ↔ n = 2 :=
sorry

end NUMINAMATH_CALUDE_x_n_prime_iff_n_eq_two_l701_70122


namespace NUMINAMATH_CALUDE_min_value_x_plus_y_l701_70148

theorem min_value_x_plus_y (x y : ℝ) (hx : x > 1) (hy : y > 2) (h : (x - 1) * (y - 2) = 4) :
  ∀ a b : ℝ, a > 1 → b > 2 → (a - 1) * (b - 2) = 4 → x + y ≤ a + b ∧ ∃ x y : ℝ, x > 1 ∧ y > 2 ∧ (x - 1) * (y - 2) = 4 ∧ x + y = 7 := by
  sorry

end NUMINAMATH_CALUDE_min_value_x_plus_y_l701_70148


namespace NUMINAMATH_CALUDE_sqrt_x_minus_one_real_l701_70126

theorem sqrt_x_minus_one_real (x : ℝ) : (∃ y : ℝ, y ^ 2 = x - 1) ↔ x ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_x_minus_one_real_l701_70126


namespace NUMINAMATH_CALUDE_eventually_divisible_by_large_power_of_two_l701_70159

/-- Represents the state of the board at any given minute -/
structure BoardState where
  numbers : Finset ℕ
  odd_count : ℕ
  minute : ℕ

/-- The initial state of the board -/
def initial_board : BoardState :=
  { numbers := Finset.empty,  -- We don't know the specific numbers, so we use an empty set
    odd_count := 33,
    minute := 0 }

/-- The next state of the board after one minute -/
def next_board_state (state : BoardState) : BoardState :=
  { numbers := state.numbers,  -- We don't update the specific numbers
    odd_count := state.odd_count,
    minute := state.minute + 1 }

/-- Predicate to check if a number is divisible by 2^10000000 -/
def is_divisible_by_large_power_of_two (n : ℕ) : Prop :=
  ∃ k, n = k * (2^10000000)

/-- The main theorem to prove -/
theorem eventually_divisible_by_large_power_of_two :
  ∃ (n : ℕ) (state : BoardState), 
    state.minute = n ∧ 
    ∃ (m : ℕ), m ∈ state.numbers ∧ is_divisible_by_large_power_of_two m :=
  sorry

end NUMINAMATH_CALUDE_eventually_divisible_by_large_power_of_two_l701_70159


namespace NUMINAMATH_CALUDE_abs_inequality_solution_set_l701_70104

theorem abs_inequality_solution_set (x : ℝ) :
  |x - 2| < |2*x + 1| ↔ x < -3 ∨ x > 1/3 := by sorry

end NUMINAMATH_CALUDE_abs_inequality_solution_set_l701_70104


namespace NUMINAMATH_CALUDE_largest_after_removal_l701_70129

/-- Represents the original sequence of digits --/
def original_sequence : List Nat := sorry

/-- Represents the sequence after removing 100 digits --/
def removed_sequence : List Nat := sorry

/-- The number of digits to be removed --/
def digits_to_remove : Nat := 100

/-- Function to convert a list of digits to a natural number --/
def list_to_number (l : List Nat) : Nat := sorry

/-- Function to check if a list of digits is a valid removal from the original sequence --/
def is_valid_removal (l : List Nat) : Prop := sorry

/-- Theorem stating that the removed_sequence is the largest possible after removing 100 digits --/
theorem largest_after_removal :
  (list_to_number removed_sequence = list_to_number original_sequence - digits_to_remove) ∧
  is_valid_removal removed_sequence ∧
  ∀ (other_sequence : List Nat),
    is_valid_removal other_sequence →
    list_to_number other_sequence ≤ list_to_number removed_sequence :=
sorry

end NUMINAMATH_CALUDE_largest_after_removal_l701_70129


namespace NUMINAMATH_CALUDE_complex_modulus_sqrt_5_l701_70105

theorem complex_modulus_sqrt_5 (a b : ℝ) (i : ℂ) (h : i * i = -1) 
  (eq : a + 2 * i = 1 - b * i) : 
  Complex.abs (a + b * i) = Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_complex_modulus_sqrt_5_l701_70105


namespace NUMINAMATH_CALUDE_max_value_of_f_on_S_l701_70123

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^3 - 3*x

-- Define the constraint set
def S : Set ℝ := {x : ℝ | x^4 + 36 ≤ 13*x^2}

-- Theorem statement
theorem max_value_of_f_on_S :
  ∃ (M : ℝ), M = 18 ∧ ∀ x ∈ S, f x ≤ M :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_on_S_l701_70123


namespace NUMINAMATH_CALUDE_tony_total_payment_l701_70157

def lego_price : ℕ := 250
def sword_price : ℕ := 120
def dough_price : ℕ := 35

def lego_quantity : ℕ := 3
def sword_quantity : ℕ := 7
def dough_quantity : ℕ := 10

def total_cost : ℕ := lego_price * lego_quantity + sword_price * sword_quantity + dough_price * dough_quantity

theorem tony_total_payment : total_cost = 1940 := by
  sorry

end NUMINAMATH_CALUDE_tony_total_payment_l701_70157


namespace NUMINAMATH_CALUDE_readers_of_both_l701_70119

def total_readers : ℕ := 400
def science_fiction_readers : ℕ := 250
def literary_works_readers : ℕ := 230

theorem readers_of_both : ℕ := by
  sorry

end NUMINAMATH_CALUDE_readers_of_both_l701_70119


namespace NUMINAMATH_CALUDE_find_k_value_l701_70139

/-- Given functions f and g, prove the value of k when f(3) - g(3) = 6 -/
theorem find_k_value (f g : ℝ → ℝ) (k : ℝ) : 
  (∀ x, f x = 7 * x^2 - 2 / x + 5) →
  (∀ x, g x = x^2 - k) →
  f 3 - g 3 = 6 →
  k = -157 / 3 := by
sorry

end NUMINAMATH_CALUDE_find_k_value_l701_70139


namespace NUMINAMATH_CALUDE_sum_of_three_odd_implies_one_odd_l701_70154

theorem sum_of_three_odd_implies_one_odd (a b c : ℤ) : 
  Odd (a + b + c) → Odd a ∨ Odd b ∨ Odd c := by
  sorry

end NUMINAMATH_CALUDE_sum_of_three_odd_implies_one_odd_l701_70154


namespace NUMINAMATH_CALUDE_hyperbola_point_coordinate_l701_70135

theorem hyperbola_point_coordinate :
  ∀ x : ℝ,
  (Real.sqrt ((x - 5)^2 + 4^2) - Real.sqrt ((x + 5)^2 + 4^2) = 6) →
  x = -3 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_point_coordinate_l701_70135


namespace NUMINAMATH_CALUDE_ricardo_coin_difference_l701_70133

/-- The total number of coins Ricardo has -/
def total_coins : ℕ := 2020

/-- The value of a penny in cents -/
def penny_value : ℕ := 1

/-- The value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- The number of pennies Ricardo has -/
def num_pennies : ℕ → ℕ := λ p => p

/-- The number of nickels Ricardo has -/
def num_nickels : ℕ → ℕ := λ p => total_coins - p

/-- The total value of Ricardo's coins in cents -/
def total_value : ℕ → ℕ := λ p => 
  penny_value * num_pennies p + nickel_value * num_nickels p

/-- The constraint that Ricardo has at least one penny and one nickel -/
def valid_distribution : ℕ → Prop := λ p => 
  1 ≤ num_pennies p ∧ 1 ≤ num_nickels p

theorem ricardo_coin_difference : 
  ∃ (max_p min_p : ℕ), 
    valid_distribution max_p ∧ 
    valid_distribution min_p ∧ 
    (∀ p, valid_distribution p → total_value p ≤ total_value max_p) ∧
    (∀ p, valid_distribution p → total_value min_p ≤ total_value p) ∧
    total_value max_p - total_value min_p = 8072 := by
  sorry

end NUMINAMATH_CALUDE_ricardo_coin_difference_l701_70133


namespace NUMINAMATH_CALUDE_ice_cream_scoops_l701_70168

theorem ice_cream_scoops (oli_scoops : ℕ) (victoria_scoops : ℕ) : 
  oli_scoops = 4 → 
  victoria_scoops = 2 * oli_scoops → 
  victoria_scoops - oli_scoops = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_ice_cream_scoops_l701_70168


namespace NUMINAMATH_CALUDE_boat_stream_speed_ratio_l701_70163

theorem boat_stream_speed_ratio 
  (boat_speed : ℝ) 
  (stream_speed : ℝ) 
  (distance : ℝ) 
  (h1 : boat_speed > stream_speed) 
  (h2 : stream_speed > 0) 
  (h3 : distance > 0) 
  (h4 : distance / (boat_speed - stream_speed) = 2 * (distance / (boat_speed + stream_speed))) :
  boat_speed / stream_speed = 3 := by
sorry

end NUMINAMATH_CALUDE_boat_stream_speed_ratio_l701_70163


namespace NUMINAMATH_CALUDE_intersection_complement_theorem_l701_70147

def M : Set ℝ := {x | ∃ y, y = Real.log (1 - 2/x)}
def N : Set ℝ := {x | ∃ y, y = Real.sqrt (x - 1)}

theorem intersection_complement_theorem : N ∩ (Set.univ \ M) = Set.Icc 1 2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_complement_theorem_l701_70147


namespace NUMINAMATH_CALUDE_total_shaded_area_is_107_l701_70138

/-- Represents a rectangle in a plane -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Represents a triangle in a plane -/
structure Triangle where
  base : ℝ
  height : ℝ

/-- Calculates the area of a rectangle -/
def rectangleArea (r : Rectangle) : ℝ :=
  r.width * r.height

/-- Calculates the area of a triangle -/
def triangleArea (t : Triangle) : ℝ :=
  0.5 * t.base * t.height

/-- Represents the configuration of shapes in the plane -/
structure ShapeConfiguration where
  rect1 : Rectangle
  rect2 : Rectangle
  triangle : Triangle
  rect1TriangleOverlap : ℝ
  rect2TriangleOverlap : ℝ
  rectOverlap : ℝ

/-- Calculates the total shaded area given a ShapeConfiguration -/
def totalShadedArea (config : ShapeConfiguration) : ℝ :=
  rectangleArea config.rect1 + rectangleArea config.rect2 + triangleArea config.triangle -
  config.rectOverlap - config.rect1TriangleOverlap - config.rect2TriangleOverlap

/-- The theorem stating the total shaded area for the given configuration -/
theorem total_shaded_area_is_107 (config : ShapeConfiguration) :
  config.rect1 = ⟨5, 12⟩ →
  config.rect2 = ⟨4, 15⟩ →
  config.triangle = ⟨3, 4⟩ →
  config.rect1TriangleOverlap = 2 →
  config.rect2TriangleOverlap = 1 →
  config.rectOverlap = 16 →
  totalShadedArea config = 107 := by
  sorry

end NUMINAMATH_CALUDE_total_shaded_area_is_107_l701_70138


namespace NUMINAMATH_CALUDE_rent_is_5000_l701_70146

/-- Calculates the monthly rent for John's computer business -/
def calculate_rent (component_cost : ℝ) (markup : ℝ) (computers_sold : ℕ) 
                   (extra_expenses : ℝ) (profit : ℝ) : ℝ :=
  let selling_price := component_cost * markup
  let total_revenue := selling_price * computers_sold
  let total_component_cost := component_cost * computers_sold
  total_revenue - total_component_cost - extra_expenses - profit

/-- Proves that the monthly rent is $5000 given the specified conditions -/
theorem rent_is_5000 : 
  calculate_rent 800 1.4 60 3000 11200 = 5000 := by
  sorry

end NUMINAMATH_CALUDE_rent_is_5000_l701_70146


namespace NUMINAMATH_CALUDE_excess_calories_james_james_excess_calories_l701_70101

/-- Calculates the excess calories James ate after eating Cheezits and going for a run -/
theorem excess_calories_james (bags : ℕ) (ounces_per_bag : ℕ) (calories_per_ounce : ℕ) 
  (run_duration : ℕ) (calories_burned_per_minute : ℕ) : ℕ :=
  let total_ounces := bags * ounces_per_bag
  let total_calories_consumed := total_ounces * calories_per_ounce
  let total_calories_burned := run_duration * calories_burned_per_minute
  total_calories_consumed - total_calories_burned

/-- Proves that James ate 420 excess calories -/
theorem james_excess_calories : 
  excess_calories_james 3 2 150 40 12 = 420 := by
  sorry

end NUMINAMATH_CALUDE_excess_calories_james_james_excess_calories_l701_70101


namespace NUMINAMATH_CALUDE_sammy_total_problems_l701_70109

/-- The total number of math problems Sammy had to do -/
def total_problems (finished : ℕ) (remaining : ℕ) : ℕ :=
  finished + remaining

/-- Theorem stating that Sammy's total math problems equal 9 -/
theorem sammy_total_problems :
  total_problems 2 7 = 9 := by
  sorry

end NUMINAMATH_CALUDE_sammy_total_problems_l701_70109


namespace NUMINAMATH_CALUDE_rounding_approximation_less_than_exact_l701_70114

theorem rounding_approximation_less_than_exact (x y z : ℕ+) :
  (↑(Int.floor (x : ℝ)) / ↑(Int.ceil (y : ℝ)) : ℝ) - ↑(Int.ceil (z : ℝ)) < (x : ℝ) / (y : ℝ) - (z : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_rounding_approximation_less_than_exact_l701_70114


namespace NUMINAMATH_CALUDE_x_in_terms_of_y_l701_70166

theorem x_in_terms_of_y (x y : ℝ) :
  (x + 1) / (x - 2) = (y^2 + 3*y - 2) / (y^2 + 3*y - 5) →
  x = (y^2 + 3*y - 1) / 7 :=
by sorry

end NUMINAMATH_CALUDE_x_in_terms_of_y_l701_70166


namespace NUMINAMATH_CALUDE_mystical_village_population_l701_70152

theorem mystical_village_population : ∃ (x y z : ℕ), 
  (x > 0) ∧ (y > 0) ∧ (z > 0) ∧
  (y^2 = x^2 + 200) ∧
  (z^2 = y^2 + 180) ∧
  (∃ (k : ℕ), x^2 = 5 * k) :=
by sorry

end NUMINAMATH_CALUDE_mystical_village_population_l701_70152


namespace NUMINAMATH_CALUDE_grandmother_pill_duration_l701_70128

/-- Calculates the duration in months for a given pill supply -/
def pillDuration (pillSupply : ℕ) (pillFraction : ℚ) (daysPerDose : ℕ) (daysPerMonth : ℕ) : ℚ :=
  (pillSupply : ℚ) * daysPerDose / pillFraction / daysPerMonth

theorem grandmother_pill_duration :
  pillDuration 60 (1/3) 3 30 = 18 := by
  sorry

end NUMINAMATH_CALUDE_grandmother_pill_duration_l701_70128


namespace NUMINAMATH_CALUDE_triangle_side_length_l701_70180

theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) :
  0 < a ∧ 0 < b ∧ 0 < c ∧
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π ∧
  A + B + C = π ∧
  a = 3 ∧ A = π/6 ∧ B = π/12 →
  c = 3 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l701_70180


namespace NUMINAMATH_CALUDE_percentage_of_x_l701_70121

theorem percentage_of_x (x : ℝ) (p : ℝ) : 
  p * x = 0.3 * (0.7 * x) + 10 ↔ p = 0.21 + 10 / x :=
by sorry

end NUMINAMATH_CALUDE_percentage_of_x_l701_70121


namespace NUMINAMATH_CALUDE_units_digit_of_7_to_50_l701_70174

theorem units_digit_of_7_to_50 : 7^50 ≡ 9 [ZMOD 10] := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_7_to_50_l701_70174


namespace NUMINAMATH_CALUDE_arithmetic_progression_zero_term_l701_70106

/-- An arithmetic progression with a term equal to zero -/
theorem arithmetic_progression_zero_term
  (a : ℕ → ℝ)  -- The arithmetic progression
  (n m : ℕ)    -- Indices of the given terms
  (h : a (2 * n) / a (2 * m) = -1)  -- Given condition
  : ∃ k, a k = 0 ∧ k = n + m := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_progression_zero_term_l701_70106


namespace NUMINAMATH_CALUDE_visibility_theorem_l701_70189

/-- Represents the position of a person at a given time -/
structure Position where
  x : ℝ
  y : ℝ

/-- Represents the movement of a person -/
def move (initial : Position) (speed : ℝ) (time : ℝ) : Position :=
  { x := initial.x + speed * time, y := initial.y }

/-- Represents a circular building -/
structure Building where
  center : Position
  radius : ℝ

/-- The time when two people can see each other after being blocked by a building -/
def visibility_time (jenny_initial : Position) (kenny_initial : Position) 
                    (jenny_speed : ℝ) (kenny_speed : ℝ) (building : Building) : ℚ :=
  240 / 5

theorem visibility_theorem (jenny_initial : Position) (kenny_initial : Position) 
                           (jenny_speed : ℝ) (kenny_speed : ℝ) (building : Building) :
  jenny_initial.y - kenny_initial.y = 300 ∧
  jenny_speed = 1 ∧ 
  kenny_speed = 4 ∧
  building.center.y = (jenny_initial.y + kenny_initial.y) / 2 ∧
  building.radius = 100 →
  visibility_time jenny_initial kenny_initial jenny_speed kenny_speed building = 240 / 5 := by
  sorry

end NUMINAMATH_CALUDE_visibility_theorem_l701_70189


namespace NUMINAMATH_CALUDE_derivative_at_two_l701_70124

/-- Given a function f(x) = ax³ + bx² + 3 where b = f'(2), prove that if f'(1) = -5, then f'(2) = -5 -/
theorem derivative_at_two (a b : ℝ) (f : ℝ → ℝ) (h1 : ∀ x, f x = a * x^3 + b * x^2 + 3)
  (h2 : b = (deriv f) 2) (h3 : (deriv f) 1 = -5) : (deriv f) 2 = -5 := by
  sorry

end NUMINAMATH_CALUDE_derivative_at_two_l701_70124


namespace NUMINAMATH_CALUDE_arithmetic_sequence_max_a5_l701_70196

theorem arithmetic_sequence_max_a5 (a : ℕ → ℝ) (s : ℕ → ℝ) :
  (∀ n, s n = (n * (2 * a 1 + (n - 1) * (a 2 - a 1))) / 2) →
  (∀ n, a n = a 1 + (n - 1) * (a 2 - a 1)) →
  s 2 ≥ 4 →
  s 4 ≤ 16 →
  a 5 ≤ 9 ∧ ∃ (a' : ℕ → ℝ), (∀ n, s n = (n * (2 * a' 1 + (n - 1) * (a' 2 - a' 1))) / 2) ∧
                             (∀ n, a' n = a' 1 + (n - 1) * (a' 2 - a' 1)) ∧
                             s 2 ≥ 4 ∧
                             s 4 ≤ 16 ∧
                             a' 5 = 9 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_max_a5_l701_70196


namespace NUMINAMATH_CALUDE_circle_area_greater_than_rectangle_l701_70127

theorem circle_area_greater_than_rectangle : ∀ (r : ℝ), r = 1 →
  π * r^2 ≥ 1 * 2.4 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_greater_than_rectangle_l701_70127


namespace NUMINAMATH_CALUDE_gcd_930_868_l701_70198

theorem gcd_930_868 : Nat.gcd 930 868 = 62 := by
  sorry

end NUMINAMATH_CALUDE_gcd_930_868_l701_70198


namespace NUMINAMATH_CALUDE_ratio_equality_l701_70132

theorem ratio_equality (a b c x y z : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h1 : a^2 + b^2 + c^2 = 49)
  (h2 : x^2 + y^2 + z^2 = 64)
  (h3 : a*x + b*y + c*z = 56) :
  (a + b + c) / (x + y + z) = 7/8 := by
sorry

end NUMINAMATH_CALUDE_ratio_equality_l701_70132


namespace NUMINAMATH_CALUDE_marble_distribution_l701_70113

theorem marble_distribution (n : ℕ) : n = 720 → 
  (Finset.filter (fun x => x > 1 ∧ x < n) (Finset.range (n + 1))).card = 28 := by
  sorry

end NUMINAMATH_CALUDE_marble_distribution_l701_70113


namespace NUMINAMATH_CALUDE_exponent_operations_l701_70116

theorem exponent_operations (x : ℝ) (x_nonzero : x ≠ 0) :
  (x^2 * x^3 = x^5) ∧
  (x^2 + x^3 ≠ x^5) ∧
  ((x^3)^2 ≠ x^5) ∧
  (x^15 / x^3 ≠ x^5) :=
by sorry

end NUMINAMATH_CALUDE_exponent_operations_l701_70116


namespace NUMINAMATH_CALUDE_roots_sum_zero_l701_70199

theorem roots_sum_zero (a b c : ℂ) : 
  a^3 - 2*a^2 + 3*a - 4 = 0 →
  b^3 - 2*b^2 + 3*b - 4 = 0 →
  c^3 - 2*c^2 + 3*c - 4 = 0 →
  a ≠ b → b ≠ c → a ≠ c →
  1 / (a * (b^2 + c^2 - a^2)) + 1 / (b * (c^2 + a^2 - b^2)) + 1 / (c * (a^2 + b^2 - c^2)) = 0 :=
by sorry

end NUMINAMATH_CALUDE_roots_sum_zero_l701_70199


namespace NUMINAMATH_CALUDE_divisor_count_l701_70187

theorem divisor_count (m n : ℕ+) (h_coprime : Nat.Coprime m n) 
  (h_divisors : (Nat.divisors (m^3 * n^5)).card = 209) : 
  (Nat.divisors (m^5 * n^3)).card = 217 := by
  sorry

end NUMINAMATH_CALUDE_divisor_count_l701_70187


namespace NUMINAMATH_CALUDE_weight_distribution_l701_70141

theorem weight_distribution :
  ∀ x y z : ℕ,
  x + y + z = 100 →
  x + 10 * y + 50 * z = 500 →
  x = 60 ∧ y = 39 ∧ z = 1 :=
by sorry

end NUMINAMATH_CALUDE_weight_distribution_l701_70141


namespace NUMINAMATH_CALUDE_oldies_requests_l701_70142

/-- Represents the number of song requests for each genre --/
structure SongRequests where
  total : ℕ
  electropop : ℕ
  dance : ℕ
  rock : ℕ
  oldies : ℕ
  dj_choice : ℕ
  rap : ℕ

/-- Theorem stating the number of oldies requests given the conditions --/
theorem oldies_requests (sr : SongRequests) : sr.oldies = 5 :=
  by
  have h1 : sr.total = 30 := by sorry
  have h2 : sr.electropop = sr.total / 2 := by sorry
  have h3 : sr.dance = sr.electropop / 3 := by sorry
  have h4 : sr.rock = 5 := by sorry
  have h5 : sr.dj_choice = sr.oldies / 2 := by sorry
  have h6 : sr.rap = 2 := by sorry
  have h7 : sr.total = sr.electropop + sr.rock + sr.oldies + sr.dj_choice + sr.rap := by sorry
  sorry

end NUMINAMATH_CALUDE_oldies_requests_l701_70142


namespace NUMINAMATH_CALUDE_bankers_interest_rate_l701_70158

/-- Proves that given a time period of 3 years, a banker's gain of 270,
    and a banker's discount of 1020, the rate of interest per annum is 12%. -/
theorem bankers_interest_rate 
  (time : ℕ) (bankers_gain : ℚ) (bankers_discount : ℚ) :
  time = 3 → 
  bankers_gain = 270 → 
  bankers_discount = 1020 → 
  ∃ (rate : ℚ), rate = 12 ∧ 
    bankers_gain = bankers_discount - (bankers_discount / (1 + rate / 100 * time)) :=
by sorry

end NUMINAMATH_CALUDE_bankers_interest_rate_l701_70158


namespace NUMINAMATH_CALUDE_zero_intersection_area_l701_70176

-- Define the square pyramid
structure SquarePyramid where
  base_side : ℝ
  slant_edge : ℝ

-- Define the plane passing through midpoints
structure IntersectingPlane where
  pyramid : SquarePyramid

-- Define the intersection area
def intersection_area (plane : IntersectingPlane) : ℝ := sorry

-- Theorem statement
theorem zero_intersection_area 
  (pyramid : SquarePyramid) 
  (h1 : pyramid.base_side = 6) 
  (h2 : pyramid.slant_edge = 5) :
  intersection_area { pyramid := pyramid } = 0 := by sorry

end NUMINAMATH_CALUDE_zero_intersection_area_l701_70176


namespace NUMINAMATH_CALUDE_solution_is_negative_eight_l701_70175

/-- An arithmetic sequence is defined by its first three terms -/
structure ArithmeticSequence :=
  (a₁ : ℚ)
  (a₂ : ℚ)
  (a₃ : ℚ)

/-- The common difference of an arithmetic sequence -/
def ArithmeticSequence.commonDifference (seq : ArithmeticSequence) : ℚ :=
  seq.a₂ - seq.a₁

/-- A sequence is arithmetic if the difference between the second and third terms
    is equal to the difference between the first and second terms -/
def ArithmeticSequence.isArithmetic (seq : ArithmeticSequence) : Prop :=
  seq.a₃ - seq.a₂ = seq.a₂ - seq.a₁

/-- The given sequence -/
def givenSequence (x : ℚ) : ArithmeticSequence :=
  { a₁ := 2
    a₂ := (2*x + 1) / 3
    a₃ := 2*x + 4 }

theorem solution_is_negative_eight :
  ∃ x : ℚ, (givenSequence x).isArithmetic ∧ x = -8 := by sorry

end NUMINAMATH_CALUDE_solution_is_negative_eight_l701_70175


namespace NUMINAMATH_CALUDE_place_values_of_fours_l701_70178

def number : ℕ := 40649003

theorem place_values_of_fours (n : ℕ) (h : n = number) :
  (n / 10000000 % 10 = 4 ∧ n / 10000000 * 10000000 = 40000000) ∧
  (n / 10000 % 10 = 4 ∧ n / 10000 % 10000 * 10000 = 40000) :=
sorry

end NUMINAMATH_CALUDE_place_values_of_fours_l701_70178


namespace NUMINAMATH_CALUDE_y_elimination_condition_l701_70161

/-- Given a system of linear equations 6x + my = 3 and 2x - ny = -6,
    y is directly eliminated by subtracting the second equation from the first
    if and only if m + n = 0 -/
theorem y_elimination_condition (m n : ℝ) : 
  (∀ x y : ℝ, 6 * x + m * y = 3 ∧ 2 * x - n * y = -6) →
  (∃! x : ℝ, ∀ y : ℝ, 6 * x + m * y = 3 ∧ 2 * x - n * y = -6) ↔
  m + n = 0 := by
sorry

end NUMINAMATH_CALUDE_y_elimination_condition_l701_70161


namespace NUMINAMATH_CALUDE_min_value_t_l701_70115

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    satisfying certain conditions, the minimum value of t is 4√2/3. -/
theorem min_value_t (a b c : ℝ) (A B C : ℝ) (S : ℝ) (t : ℝ) :
  a - b = c / 3 →
  3 * Real.sin B = 2 * Real.sin A →
  2 ≤ a * c + c^2 →
  a * c + c^2 ≤ 32 →
  S = (1 / 2) * a * b * Real.sin C →
  t = (S + 2 * Real.sqrt 2) / a →
  t ≥ 4 * Real.sqrt 2 / 3 ∧ 
  ∃ (a₀ b₀ c₀ : ℝ) (A₀ B₀ C₀ : ℝ) (S₀ : ℝ),
    a₀ - b₀ = c₀ / 3 ∧
    3 * Real.sin B₀ = 2 * Real.sin A₀ ∧
    2 ≤ a₀ * c₀ + c₀^2 ∧
    a₀ * c₀ + c₀^2 ≤ 32 ∧
    S₀ = (1 / 2) * a₀ * b₀ * Real.sin C₀ ∧
    (S₀ + 2 * Real.sqrt 2) / a₀ = 4 * Real.sqrt 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_t_l701_70115


namespace NUMINAMATH_CALUDE_night_flying_hours_l701_70167

theorem night_flying_hours (total_required : ℕ) (day_flying : ℕ) (cross_country : ℕ) (monthly_hours : ℕ) (months : ℕ) : 
  total_required = 1500 →
  day_flying = 50 →
  cross_country = 121 →
  monthly_hours = 220 →
  months = 6 →
  total_required - (day_flying + cross_country) - (monthly_hours * months) = 9 := by
  sorry

end NUMINAMATH_CALUDE_night_flying_hours_l701_70167


namespace NUMINAMATH_CALUDE_least_number_of_cans_l701_70179

def maaza_volume : ℕ := 60
def pepsi_volume : ℕ := 144
def sprite_volume : ℕ := 368

def can_volume : ℕ := Nat.gcd maaza_volume (Nat.gcd pepsi_volume sprite_volume)

def maaza_cans : ℕ := maaza_volume / can_volume
def pepsi_cans : ℕ := pepsi_volume / can_volume
def sprite_cans : ℕ := sprite_volume / can_volume

def total_cans : ℕ := maaza_cans + pepsi_cans + sprite_cans

theorem least_number_of_cans : total_cans = 143 := by
  sorry

end NUMINAMATH_CALUDE_least_number_of_cans_l701_70179


namespace NUMINAMATH_CALUDE_quadratic_from_means_l701_70143

theorem quadratic_from_means (a b : ℝ) (h_am : (a + b) / 2 = 7.5) (h_gm : Real.sqrt (a * b) = 12) :
  ∀ x, x ^ 2 - 15 * x + 144 = 0 ↔ (x = a ∨ x = b) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_from_means_l701_70143


namespace NUMINAMATH_CALUDE_decimal_addition_l701_70149

theorem decimal_addition : (0.8 : ℝ) + 0.02 = 0.82 := by
  sorry

end NUMINAMATH_CALUDE_decimal_addition_l701_70149


namespace NUMINAMATH_CALUDE_stockholm_uppsala_distance_l701_70197

/-- The distance between two cities on a map, in centimeters. -/
def map_distance : ℝ := 45

/-- The scale of the map, representing how many kilometers in reality correspond to one centimeter on the map. -/
def map_scale : ℝ := 20

/-- The actual distance between the two cities, in kilometers. -/
def actual_distance : ℝ := map_distance * map_scale

theorem stockholm_uppsala_distance : actual_distance = 900 := by
  sorry

end NUMINAMATH_CALUDE_stockholm_uppsala_distance_l701_70197


namespace NUMINAMATH_CALUDE_sum_fifth_sixth_l701_70130

/-- A geometric sequence with the given properties -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q
  sum_first_two : a 1 + a 2 = 3
  sum_third_fourth : a 3 + a 4 = 12

/-- The sum of the fifth and sixth terms equals 48 -/
theorem sum_fifth_sixth (seq : GeometricSequence) : seq.a 5 + seq.a 6 = 48 := by
  sorry

end NUMINAMATH_CALUDE_sum_fifth_sixth_l701_70130


namespace NUMINAMATH_CALUDE_train_length_calculation_l701_70108

/-- The length of a train in meters. -/
def train_length : ℝ := 1500

/-- The time in seconds it takes for the train to cross a tree. -/
def time_tree : ℝ := 120

/-- The time in seconds it takes for the train to pass a platform. -/
def time_platform : ℝ := 160

/-- The length of the platform in meters. -/
def platform_length : ℝ := 500

theorem train_length_calculation :
  train_length = 1500 ∧
  (train_length / time_tree = (train_length + platform_length) / time_platform) :=
by sorry

end NUMINAMATH_CALUDE_train_length_calculation_l701_70108


namespace NUMINAMATH_CALUDE_geometric_series_sum_l701_70102

theorem geometric_series_sum : 
  let a₁ : ℚ := 1 / 4
  let r : ℚ := -1 / 4
  let n : ℕ := 6
  let series_sum := a₁ * (1 - r^n) / (1 - r)
  series_sum = 81 / 405 := by
sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l701_70102


namespace NUMINAMATH_CALUDE_celsius_to_fahrenheit_l701_70171

/-- Given the relationship between Celsius (C) and Fahrenheit (F) temperatures,
    prove that when C is 25, F is 75. -/
theorem celsius_to_fahrenheit (C F : ℚ) : 
  C = 25 → C = (5 / 9) * (F - 30) → F = 75 := by sorry

end NUMINAMATH_CALUDE_celsius_to_fahrenheit_l701_70171


namespace NUMINAMATH_CALUDE_village_population_l701_70165

theorem village_population (population_percentage : Real) (partial_population : Nat) :
  population_percentage = 80 →
  partial_population = 23040 →
  (partial_population : Real) / (population_percentage / 100) = 28800 :=
by sorry

end NUMINAMATH_CALUDE_village_population_l701_70165


namespace NUMINAMATH_CALUDE_product_of_fractions_l701_70145

theorem product_of_fractions : (3 : ℚ) / 8 * 2 / 5 * 1 / 4 = 3 / 80 := by
  sorry

end NUMINAMATH_CALUDE_product_of_fractions_l701_70145


namespace NUMINAMATH_CALUDE_equality_check_l701_70183

theorem equality_check : 
  (-3^2 ≠ -2^3) ∧ 
  (-6^3 = (-6)^3) ∧ 
  (-6^2 ≠ (-6)^2) ∧ 
  ((-3 * 2)^2 ≠ (-3) * 2^2) :=
by
  sorry

end NUMINAMATH_CALUDE_equality_check_l701_70183


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l701_70182

theorem imaginary_part_of_z (z : ℂ) (h : (1 + Complex.I) * z = 2) : 
  Complex.im z = -1 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l701_70182


namespace NUMINAMATH_CALUDE_square_area_equal_perimeter_triangle_l701_70169

/-- The area of a square with perimeter equal to that of a triangle with sides 7.3 cm, 8.6 cm, and 10.1 cm is 42.25 square centimeters. -/
theorem square_area_equal_perimeter_triangle (a b c : ℝ) (s : ℝ) :
  a = 7.3 ∧ b = 8.6 ∧ c = 10.1 →
  4 * s = a + b + c →
  s^2 = 42.25 := by
  sorry

end NUMINAMATH_CALUDE_square_area_equal_perimeter_triangle_l701_70169


namespace NUMINAMATH_CALUDE_reciprocal_minus_opposite_l701_70195

theorem reciprocal_minus_opposite : 
  let x : ℚ := -4
  (-1 / x) - (-x) = -17 / 4 := by sorry

end NUMINAMATH_CALUDE_reciprocal_minus_opposite_l701_70195


namespace NUMINAMATH_CALUDE_cheese_division_theorem_l701_70155

/-- Represents the weight of cheese pieces -/
structure CheesePair :=
  (larger : ℕ)
  (smaller : ℕ)

/-- Divides cheese by taking a piece equal to the smaller portion from the larger -/
def divide_cheese (pair : CheesePair) : CheesePair :=
  CheesePair.mk pair.smaller (pair.larger - pair.smaller)

/-- The initial weight of the cheese -/
def initial_weight : ℕ := 850

/-- The final weight of each piece of cheese -/
def final_piece_weight : ℕ := 25

theorem cheese_division_theorem :
  let final_state := CheesePair.mk final_piece_weight final_piece_weight
  let third_division := divide_cheese (divide_cheese (divide_cheese (CheesePair.mk initial_weight 0)))
  third_division = final_state :=
sorry

end NUMINAMATH_CALUDE_cheese_division_theorem_l701_70155


namespace NUMINAMATH_CALUDE_square_sum_equals_43_l701_70125

theorem square_sum_equals_43 (x y : ℝ) 
  (h1 : y + 6 = (x - 3)^2) 
  (h2 : x + 6 = (y - 3)^2) 
  (h3 : x ≠ y) : 
  x^2 + y^2 = 43 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_equals_43_l701_70125


namespace NUMINAMATH_CALUDE_triangle_properties_l701_70111

theorem triangle_properties (A B C : ℝ) (a b c : ℝ) :
  (1/2 : ℝ) * Real.cos (2 * A) = Real.cos A ^ 2 - Real.cos A →
  a = 3 →
  Real.sin B = 2 * Real.sin C →
  A = π / 3 ∧ 
  (1/2 : ℝ) * b * c * Real.sin A = (3 * Real.sqrt 3) / 2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l701_70111


namespace NUMINAMATH_CALUDE_man_son_age_ratio_l701_70170

/-- Proves that the ratio of a man's age to his son's age in two years is 2:1,
    given the man is 24 years older than his son and the son's present age is 22 years. -/
theorem man_son_age_ratio :
  ∀ (son_age man_age : ℕ),
    son_age = 22 →
    man_age = son_age + 24 →
    (man_age + 2) / (son_age + 2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_man_son_age_ratio_l701_70170


namespace NUMINAMATH_CALUDE_train_stop_time_l701_70184

/-- Proves that a train with given speeds stops for 20 minutes per hour -/
theorem train_stop_time (speed_without_stops speed_with_stops : ℝ) 
  (h1 : speed_without_stops = 45)
  (h2 : speed_with_stops = 30) : 
  (speed_without_stops - speed_with_stops) / speed_without_stops * 60 = 20 := by
  sorry

#check train_stop_time

end NUMINAMATH_CALUDE_train_stop_time_l701_70184


namespace NUMINAMATH_CALUDE_fraction_of_odd_products_in_table_l701_70117

-- Define the size of the multiplication table
def table_size : Nat := 16

-- Define a function to check if a number is odd
def is_odd (n : Nat) : Bool := n % 2 = 1

-- Define a function to count odd numbers in a range
def count_odd (n : Nat) : Nat :=
  (List.range n).filter is_odd |>.length

-- Statement of the theorem
theorem fraction_of_odd_products_in_table :
  (count_odd table_size ^ 2 : Rat) / (table_size ^ 2) = 1 / 4 := by
  sorry


end NUMINAMATH_CALUDE_fraction_of_odd_products_in_table_l701_70117


namespace NUMINAMATH_CALUDE_slope_of_line_l701_70140

theorem slope_of_line (x y : ℝ) :
  4 * x - 7 * y = 28 → (y - (-4)) / (x - 0) = 4 / 7 :=
by sorry

end NUMINAMATH_CALUDE_slope_of_line_l701_70140


namespace NUMINAMATH_CALUDE_eight_stairs_climb_ways_l701_70131

/-- The number of ways to climb n stairs, taking 1, 2, 3, or 4 steps at a time. -/
def climbWays (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | 1 => 1
  | 2 => 2
  | 3 => 4
  | m + 4 => climbWays m + climbWays (m + 1) + climbWays (m + 2) + climbWays (m + 3)

theorem eight_stairs_climb_ways :
  climbWays 8 = 108 := by
  sorry

#eval climbWays 8

end NUMINAMATH_CALUDE_eight_stairs_climb_ways_l701_70131


namespace NUMINAMATH_CALUDE_melies_money_left_l701_70153

/-- The amount of money Méliès has left after buying meat -/
def money_left (initial_money meat_quantity meat_price : ℝ) : ℝ :=
  initial_money - meat_quantity * meat_price

/-- Theorem: Méliès has $16 left after buying meat -/
theorem melies_money_left :
  let initial_money : ℝ := 180
  let meat_quantity : ℝ := 2
  let meat_price : ℝ := 82
  money_left initial_money meat_quantity meat_price = 16 := by
  sorry

end NUMINAMATH_CALUDE_melies_money_left_l701_70153


namespace NUMINAMATH_CALUDE_unique_g_50_l701_70150

/-- A function from ℕ to ℕ satisfying the given property -/
def special_function (g : ℕ → ℕ) : Prop :=
  ∀ a b : ℕ, 2 * g (a^2 + 2*b^2) = (g a)^2 + 3*(g b)^2

theorem unique_g_50 (g : ℕ → ℕ) (h : special_function g) : g 50 = 0 := by
  sorry

#check unique_g_50

end NUMINAMATH_CALUDE_unique_g_50_l701_70150


namespace NUMINAMATH_CALUDE_complex_fraction_power_four_l701_70191

theorem complex_fraction_power_four (i : ℂ) (h : i * i = -1) : 
  ((1 + i) / (1 - i)) ^ 4 = 1 := by sorry

end NUMINAMATH_CALUDE_complex_fraction_power_four_l701_70191


namespace NUMINAMATH_CALUDE_abc_remainder_mod_7_l701_70193

theorem abc_remainder_mod_7 (a b c : ℕ) 
  (h_a : a < 7) (h_b : b < 7) (h_c : c < 7)
  (h1 : (a + 3*b + 2*c) % 7 = 3)
  (h2 : (2*a + b + 3*c) % 7 = 2)
  (h3 : (3*a + 2*b + c) % 7 = 1) :
  (a * b * c) % 7 = 4 := by
sorry

end NUMINAMATH_CALUDE_abc_remainder_mod_7_l701_70193


namespace NUMINAMATH_CALUDE_expression_simplification_l701_70177

theorem expression_simplification : 
  let x : ℝ := Real.sqrt 6 - Real.sqrt 2
  (x * (Real.sqrt 6 - x) + (x + Real.sqrt 5) * (x - Real.sqrt 5)) = 1 - 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l701_70177


namespace NUMINAMATH_CALUDE_base_r_problem_l701_70137

/-- Represents a number in base r -/
def BaseRNumber (r : ℕ) (digits : List ℕ) : ℕ :=
  digits.foldr (fun d acc => d + r * acc) 0

/-- The problem statement -/
theorem base_r_problem (r : ℕ) : 
  (r > 1) →
  (BaseRNumber r [0, 0, 0, 1] = 1000) →
  (BaseRNumber r [0, 4, 4] = 440) →
  (BaseRNumber r [0, 4, 3] = 340) →
  (1000 - 440 = 340) →
  r = 8 := by
sorry

end NUMINAMATH_CALUDE_base_r_problem_l701_70137


namespace NUMINAMATH_CALUDE_power_division_rule_l701_70144

theorem power_division_rule (m : ℝ) (h : m ≠ 0) : m^7 / m = m^6 := by
  sorry

end NUMINAMATH_CALUDE_power_division_rule_l701_70144


namespace NUMINAMATH_CALUDE_first_part_speed_l701_70136

/-- Proves that given a 50 km trip with two equal parts, where the second part is traveled at 33 km/h, 
    and the average speed of the entire trip is 44.00000000000001 km/h, 
    the speed of the first part of the trip is 66 km/h. -/
theorem first_part_speed (total_distance : ℝ) (first_part_distance : ℝ) (second_part_speed : ℝ) (average_speed : ℝ) :
  total_distance = 50 →
  first_part_distance = 25 →
  second_part_speed = 33 →
  average_speed = 44.00000000000001 →
  (total_distance / (first_part_distance / (total_distance - first_part_distance) * second_part_speed + first_part_distance / second_part_speed)) = average_speed →
  (total_distance - first_part_distance) / second_part_speed + first_part_distance / ((total_distance - first_part_distance) * second_part_speed / first_part_distance) = total_distance / average_speed →
  (total_distance - first_part_distance) * second_part_speed / first_part_distance = 66 :=
by sorry

end NUMINAMATH_CALUDE_first_part_speed_l701_70136


namespace NUMINAMATH_CALUDE_car_distance_proof_l701_70160

/-- Proves the initial distance between two cars driving towards each other --/
theorem car_distance_proof (speed1 : ℝ) (speed2 : ℝ) (time : ℝ) 
  (h1 : speed1 = 100)
  (h2 : speed1 = 1.25 * speed2)
  (h3 : time = 4) :
  speed1 * time + speed2 * time = 720 := by
  sorry

end NUMINAMATH_CALUDE_car_distance_proof_l701_70160


namespace NUMINAMATH_CALUDE_expedition_ratio_l701_70120

/-- Proves the ratio of weeks spent on the last expedition to the second expedition -/
theorem expedition_ratio : 
  ∀ (first second last : ℕ) (total : ℕ),
  first = 3 →
  second = first + 2 →
  total = 7 * (first + second + last) →
  total = 126 →
  last = 2 * second :=
by
  sorry

end NUMINAMATH_CALUDE_expedition_ratio_l701_70120


namespace NUMINAMATH_CALUDE_librarian_books_taken_l701_70151

theorem librarian_books_taken (total_books : ℕ) (books_per_shelf : ℕ) (shelves_needed : ℕ) : 
  total_books - (books_per_shelf * shelves_needed) = 10 :=
by
  sorry

#check librarian_books_taken 46 4 9

end NUMINAMATH_CALUDE_librarian_books_taken_l701_70151


namespace NUMINAMATH_CALUDE_number_of_trucks_l701_70194

theorem number_of_trucks (total_packages : ℕ) (packages_per_truck : ℕ) (h1 : total_packages = 490) (h2 : packages_per_truck = 70) :
  total_packages / packages_per_truck = 7 := by
  sorry

end NUMINAMATH_CALUDE_number_of_trucks_l701_70194


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l701_70134

theorem arithmetic_calculations :
  (15 + (-6) + 3 - (-4) = 16) ∧
  (8 - 2^3 / (4/9) * (-2/3)^2 = 0) := by
sorry

end NUMINAMATH_CALUDE_arithmetic_calculations_l701_70134


namespace NUMINAMATH_CALUDE_min_filtration_cycles_l701_70181

theorem min_filtration_cycles (initial_conc : ℝ) (reduction_rate : ℝ) (target_conc : ℝ) : 
  initial_conc = 225 →
  reduction_rate = 1/3 →
  target_conc = 7.5 →
  (∃ n : ℕ, (initial_conc * (1 - reduction_rate)^n ≤ target_conc ∧ 
             ∀ m : ℕ, m < n → initial_conc * (1 - reduction_rate)^m > target_conc)) →
  (∃ n : ℕ, n = 9 ∧ initial_conc * (1 - reduction_rate)^n ≤ target_conc ∧ 
             ∀ m : ℕ, m < n → initial_conc * (1 - reduction_rate)^m > target_conc) :=
by sorry

end NUMINAMATH_CALUDE_min_filtration_cycles_l701_70181


namespace NUMINAMATH_CALUDE_quadratic_root_difference_l701_70156

theorem quadratic_root_difference (p q : ℝ) (hp : p > 0) (hq : q > 0) :
  (∀ x y : ℝ, x^2 + p*x + q = 0 ∧ y^2 + p*y + q = 0 ∧ x ≠ y → |x - y| = 2) →
  p = 2 * Real.sqrt (q + 1) := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_difference_l701_70156


namespace NUMINAMATH_CALUDE_triangle_angle_proof_l701_70185

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    and the condition c * sin(A) = √3 * a * cos(C), prove that C = π/3. -/
theorem triangle_angle_proof (a b c A B C : ℝ) 
  (h_triangle : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π)
  (h_condition : c * Real.sin A = Real.sqrt 3 * a * Real.cos C) : 
  C = π / 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_proof_l701_70185


namespace NUMINAMATH_CALUDE_diagonal_length_l701_70110

-- Define the quadrilateral EFGH
structure Quadrilateral :=
  (E F G H : ℝ × ℝ)

-- Define the properties of the quadrilateral
def is_valid_quadrilateral (q : Quadrilateral) : Prop :=
  let (ex, ey) := q.E
  let (fx, fy) := q.F
  let (gx, gy) := q.G
  let (hx, hy) := q.H
  -- EF = FG = 12
  (ex - fx)^2 + (ey - fy)^2 = 12^2 ∧
  (fx - gx)^2 + (fy - gy)^2 = 12^2 ∧
  -- GH = HE = 20
  (gx - hx)^2 + (gy - hy)^2 = 20^2 ∧
  (hx - ex)^2 + (hy - ey)^2 = 20^2 ∧
  -- Angle GHE = 90°
  (gx - hx) * (ex - hx) + (gy - hy) * (ey - hy) = 0

-- Theorem statement
theorem diagonal_length (q : Quadrilateral) (h : is_valid_quadrilateral q) :
  let (ex, ey) := q.E
  let (gx, gy) := q.G
  (ex - gx)^2 + (ey - gy)^2 = 2 * 20^2 := by
  sorry

end NUMINAMATH_CALUDE_diagonal_length_l701_70110


namespace NUMINAMATH_CALUDE_complex_polynomial_equality_l701_70186

theorem complex_polynomial_equality (z : ℂ) (h : z = 2 - I) :
  z^6 - 3*z^5 + z^4 + 5*z^3 + 2 = (z^2 - 4*z + 5)*(z^4 + z^3) + 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_polynomial_equality_l701_70186


namespace NUMINAMATH_CALUDE_clouddale_rainfall_2005_l701_70118

/-- Represents the rainfall data for Clouddale -/
structure ClouddaleRainfall where
  initialYear : Nat
  initialAvgMonthlyRainfall : Real
  yearlyIncrease : Real

/-- Calculates the average monthly rainfall for a given year -/
def avgMonthlyRainfall (data : ClouddaleRainfall) (year : Nat) : Real :=
  data.initialAvgMonthlyRainfall + (year - data.initialYear : Real) * data.yearlyIncrease

/-- Calculates the total yearly rainfall for a given year -/
def totalYearlyRainfall (data : ClouddaleRainfall) (year : Nat) : Real :=
  (avgMonthlyRainfall data year) * 12

/-- Theorem: The total rainfall in Clouddale in 2005 was 522 mm -/
theorem clouddale_rainfall_2005 (data : ClouddaleRainfall) 
    (h1 : data.initialYear = 2003)
    (h2 : data.initialAvgMonthlyRainfall = 37.5)
    (h3 : data.yearlyIncrease = 3) : 
    totalYearlyRainfall data 2005 = 522 := by
  sorry


end NUMINAMATH_CALUDE_clouddale_rainfall_2005_l701_70118


namespace NUMINAMATH_CALUDE_x_convergence_bound_l701_70164

def x : ℕ → ℚ
  | 0 => 7
  | n + 1 => (x n ^ 2 + 7 * x n + 12) / (x n + 8)

theorem x_convergence_bound : 
  ∃ m : ℕ, 243 ≤ m ∧ m ≤ 728 ∧ 
    x m ≤ 6 + 1 / 2^18 ∧ 
    ∀ k < m, x k > 6 + 1 / 2^18 :=
sorry

end NUMINAMATH_CALUDE_x_convergence_bound_l701_70164


namespace NUMINAMATH_CALUDE_max_value_sum_fractions_l701_70162

theorem max_value_sum_fractions (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x / (2 * x + y)) + (y / (x + 2 * y)) ≤ 2 / 3 ∧
  ((x / (2 * x + y)) + (y / (x + 2 * y)) = 2 / 3 ↔ x = y) :=
by sorry

end NUMINAMATH_CALUDE_max_value_sum_fractions_l701_70162


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l701_70100

theorem sqrt_equation_solution (x : ℝ) :
  x > 16 →
  (Real.sqrt (x - 8 * Real.sqrt (x - 16)) + 4 = Real.sqrt (x + 8 * Real.sqrt (x - 16)) - 4) ↔
  x ≥ 32 := by
sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l701_70100


namespace NUMINAMATH_CALUDE_target_number_position_l701_70107

/-- Represents a position in the spiral matrix -/
structure Position where
  row : Nat
  col : Nat

/-- Fills a square matrix in a clockwise spiral order -/
def spiralFill (n : Nat) : Nat → Position
  | k => sorry  -- Implementation details omitted

/-- The size of our spiral matrix -/
def matrixSize : Nat := 100

/-- The number we're looking for in the spiral matrix -/
def targetNumber : Nat := 2018

/-- The expected position of the target number -/
def expectedPosition : Position := ⟨34, 95⟩

theorem target_number_position :
  spiralFill matrixSize targetNumber = expectedPosition := by sorry

end NUMINAMATH_CALUDE_target_number_position_l701_70107


namespace NUMINAMATH_CALUDE_polygonal_number_formula_l701_70192

def N (n k : ℕ) : ℚ :=
  match k with
  | 3 => (n^2 + n) / 2
  | 4 => n^2
  | 5 => (3*n^2 - n) / 2
  | 6 => 2*n^2 - n
  | _ => 0

theorem polygonal_number_formula (n k : ℕ) (h : k ≥ 3) :
  N n k = ((k - 2 : ℚ) / 2) * n^2 + ((4 - k : ℚ) / 2) * n :=
by sorry

end NUMINAMATH_CALUDE_polygonal_number_formula_l701_70192


namespace NUMINAMATH_CALUDE_car_distance_theorem_l701_70172

/-- Given a car traveling at a specific speed for a certain time, 
    calculate the distance covered. -/
theorem car_distance_theorem (speed : ℝ) (time : ℝ) (distance : ℝ) : 
  speed = 160 → time = 5 → distance = speed * time → distance = 800 :=
by sorry

end NUMINAMATH_CALUDE_car_distance_theorem_l701_70172


namespace NUMINAMATH_CALUDE_inequality_proof_l701_70190

theorem inequality_proof (x : ℝ) (n : ℕ) (h1 : x > 0) (h2 : n > 0) :
  x + n^n / x^n ≥ n + 1 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l701_70190


namespace NUMINAMATH_CALUDE_percentage_of_percentage_l701_70173

theorem percentage_of_percentage (y : ℝ) (h : y ≠ 0) :
  (0.3 * 0.6 * y) / y * 100 = 18 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_percentage_l701_70173
