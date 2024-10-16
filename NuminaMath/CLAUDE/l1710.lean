import Mathlib

namespace NUMINAMATH_CALUDE_f_not_increasing_l1710_171091

-- Define the function
def f (x : ℝ) : ℝ := |3 - x|

-- State the theorem
theorem f_not_increasing :
  ¬(∀ x y : ℝ, 0 ≤ x ∧ x < y → f x ≤ f y) :=
sorry

end NUMINAMATH_CALUDE_f_not_increasing_l1710_171091


namespace NUMINAMATH_CALUDE_matching_pair_probability_l1710_171022

def black_socks : ℕ := 12
def blue_socks : ℕ := 10

def total_socks : ℕ := black_socks + blue_socks

def choose (n k : ℕ) : ℕ := (n.factorial) / ((k.factorial) * ((n - k).factorial))

def total_ways : ℕ := choose total_socks 2
def black_matching_ways : ℕ := choose black_socks 2
def blue_matching_ways : ℕ := choose blue_socks 2
def matching_ways : ℕ := black_matching_ways + blue_matching_ways

theorem matching_pair_probability :
  (matching_ways : ℚ) / total_ways = 111 / 231 :=
sorry

end NUMINAMATH_CALUDE_matching_pair_probability_l1710_171022


namespace NUMINAMATH_CALUDE_vase_original_price_l1710_171049

/-- Proves that given a vase with an original price P, which is discounted by 25% 
    and then has a 10% sales tax applied, if the total price paid is $165, 
    then the original price P must be $200. -/
theorem vase_original_price (P : ℝ) : 
  (P * (1 - 0.25) * (1 + 0.1) = 165) → P = 200 := by
  sorry

end NUMINAMATH_CALUDE_vase_original_price_l1710_171049


namespace NUMINAMATH_CALUDE_river_current_calculation_l1710_171024

/-- Represents the speed of a boat in still water -/
def boat_speed : ℝ := 20

/-- Represents the distance traveled up the river -/
def distance : ℝ := 91

/-- Represents the total time for the round trip -/
def total_time : ℝ := 10

/-- Calculates the speed of the river's current -/
def river_current_speed : ℝ := 6

theorem river_current_calculation :
  ∃ (c : ℝ), c = river_current_speed ∧
  distance / (boat_speed - c) + distance / (boat_speed + c) = total_time :=
by sorry

end NUMINAMATH_CALUDE_river_current_calculation_l1710_171024


namespace NUMINAMATH_CALUDE_fib_150_mod_9_l1710_171002

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fib n + fib (n + 1)

-- Define the property that Fibonacci sequence mod 9 repeats every 24 terms
axiom fib_mod_9_period : ∀ n : ℕ, fib n % 9 = fib (n % 24) % 9

-- Theorem statement
theorem fib_150_mod_9 : fib 149 % 9 = 8 := by
  sorry

end NUMINAMATH_CALUDE_fib_150_mod_9_l1710_171002


namespace NUMINAMATH_CALUDE_light_ray_reflection_l1710_171063

/-- Represents a direction vector in 3D space -/
structure DirectionVector where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a trirectangular corner -/
structure TrirectangularCorner where

/-- Reflects a direction vector off a plane perpendicular to the x-axis -/
def reflectX (v : DirectionVector) : DirectionVector :=
  { x := -v.x, y := v.y, z := v.z }

/-- Reflects a direction vector off a plane perpendicular to the y-axis -/
def reflectY (v : DirectionVector) : DirectionVector :=
  { x := v.x, y := -v.y, z := v.z }

/-- Reflects a direction vector off a plane perpendicular to the z-axis -/
def reflectZ (v : DirectionVector) : DirectionVector :=
  { x := v.x, y := v.y, z := -v.z }

/-- 
  Theorem: A light ray reflecting off all three faces of a trirectangular corner
  will change its direction to the opposite of its initial direction.
-/
theorem light_ray_reflection 
  (corner : TrirectangularCorner) 
  (initial_direction : DirectionVector) :
  reflectX (reflectY (reflectZ initial_direction)) = 
  { x := -initial_direction.x, 
    y := -initial_direction.y, 
    z := -initial_direction.z } := by
  sorry


end NUMINAMATH_CALUDE_light_ray_reflection_l1710_171063


namespace NUMINAMATH_CALUDE_sets_and_range_theorem_l1710_171000

-- Define the sets A and B
def A : Set ℝ := {x | -x^2 + 3*x + 10 ≥ 0}
def B (m : ℝ) : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 2*m - 1}

-- State the theorem
theorem sets_and_range_theorem (m : ℝ) (h : B m ⊆ A) : 
  A = {x | -2 ≤ x ∧ x ≤ 5} ∧ m ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_sets_and_range_theorem_l1710_171000


namespace NUMINAMATH_CALUDE_rhino_state_reachable_l1710_171052

/-- Represents the state of a Rhinoceros with folds on its skin -/
structure RhinoState :=
  (left_vertical : Nat)
  (left_horizontal : Nat)
  (right_vertical : Nat)
  (right_horizontal : Nat)

/-- Represents the direction of scratching -/
inductive ScratchDirection
  | Vertical
  | Horizontal

/-- Represents the side of the Rhinoceros being scratched -/
inductive Side
  | Left
  | Right

/-- Defines a single transition step for a Rhinoceros state -/
def transition (s : RhinoState) (dir : ScratchDirection) (side : Side) : RhinoState :=
  sorry

/-- Defines if a target state is reachable from an initial state -/
def is_reachable (initial : RhinoState) (target : RhinoState) : Prop :=
  sorry

/-- The main theorem to prove -/
theorem rhino_state_reachable :
  is_reachable
    (RhinoState.mk 0 2 2 1)
    (RhinoState.mk 2 0 2 1) :=
  sorry

end NUMINAMATH_CALUDE_rhino_state_reachable_l1710_171052


namespace NUMINAMATH_CALUDE_ibrahim_palace_count_l1710_171061

/-- Represents a square grid of rooms -/
structure RoomGrid where
  size : Nat
  has_door_between_rooms : Bool
  has_window_on_outer_wall : Bool

/-- Calculates the number of windows in the grid -/
def count_windows (grid : RoomGrid) : Nat :=
  if grid.has_window_on_outer_wall then
    4 * grid.size
  else
    0

/-- Calculates the number of doors in the grid -/
def count_doors (grid : RoomGrid) : Nat :=
  if grid.has_door_between_rooms then
    2 * grid.size * (grid.size - 1)
  else
    0

/-- Theorem stating the number of windows and doors in the specific 10x10 grid -/
theorem ibrahim_palace_count (grid : RoomGrid)
  (h_size : grid.size = 10)
  (h_door : grid.has_door_between_rooms = true)
  (h_window : grid.has_window_on_outer_wall = true) :
  count_windows grid = 40 ∧ count_doors grid = 180 := by
  sorry


end NUMINAMATH_CALUDE_ibrahim_palace_count_l1710_171061


namespace NUMINAMATH_CALUDE_mildreds_father_oranges_l1710_171079

/-- The number of oranges Mildred's father ate -/
def oranges_eaten (initial : ℝ) (remaining : ℝ) : ℝ :=
  initial - remaining

/-- Proof that Mildred's father ate 2.0 oranges -/
theorem mildreds_father_oranges : oranges_eaten 77.0 75 = 2.0 := by
  sorry

end NUMINAMATH_CALUDE_mildreds_father_oranges_l1710_171079


namespace NUMINAMATH_CALUDE_multiple_with_binary_digits_l1710_171009

theorem multiple_with_binary_digits (n : ℤ) : ∃ m : ℤ, 
  (n ∣ m) ∧ 
  (∃ k : ℕ, k ≤ n ∧ m < 10^k) ∧
  (∀ d : ℕ, d < 10 → (m / 10^d % 10 = 0 ∨ m / 10^d % 10 = 1)) :=
by sorry

end NUMINAMATH_CALUDE_multiple_with_binary_digits_l1710_171009


namespace NUMINAMATH_CALUDE_expression_value_l1710_171076

theorem expression_value (x : ℝ) (h : x^2 - 3*x = 12) : 3*x^2 - 9*x + 5 = 41 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l1710_171076


namespace NUMINAMATH_CALUDE_equation_b_not_symmetric_l1710_171045

def is_symmetric_to_x_axis (f : ℝ → ℝ → ℝ) : Prop :=
  ∀ x y, f x y = f x (-y)

theorem equation_b_not_symmetric :
  ¬(is_symmetric_to_x_axis (fun x y => x^2*y + x*y^2 - 1)) ∧
  (is_symmetric_to_x_axis (fun x y => x^2 - x + y^2 - 1)) ∧
  (is_symmetric_to_x_axis (fun x y => 2*x^2 - y^2 - 1)) ∧
  (is_symmetric_to_x_axis (fun x y => x + y^2 + 1)) :=
by sorry

end NUMINAMATH_CALUDE_equation_b_not_symmetric_l1710_171045


namespace NUMINAMATH_CALUDE_fraction_of_pet_owners_l1710_171003

/-- Proves that the fraction of freshmen and sophomores who own a pet is 1/5 -/
theorem fraction_of_pet_owners (total_students : ℕ) (freshmen_sophomores : ℕ) (no_pet : ℕ) :
  total_students = 400 →
  freshmen_sophomores = total_students / 2 →
  no_pet = 160 →
  (freshmen_sophomores - no_pet) / freshmen_sophomores = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_pet_owners_l1710_171003


namespace NUMINAMATH_CALUDE_same_answers_l1710_171080

-- Define a type for questions
variable (Question : Type)

-- Define predicates for each witness's "yes" answers
variable (A B C : Question → Prop)

-- State the conditions
variable (h1 : ∀ q, B q ∧ C q → A q)
variable (h2 : ∀ q, A q → B q)
variable (h3 : ∀ q, B q → A q ∨ C q)

-- Theorem statement
theorem same_answers : ∀ q, A q ↔ B q := by sorry

end NUMINAMATH_CALUDE_same_answers_l1710_171080


namespace NUMINAMATH_CALUDE_skew_parallel_relationship_l1710_171006

-- Define a type for lines in 3D space
structure Line3D where
  -- You might represent a line using a point and a direction vector
  -- But for simplicity, we'll just use an opaque type
  mk :: (dummy : Unit)

-- Define what it means for two lines to be skew
def are_skew (l1 l2 : Line3D) : Prop :=
  -- Two lines are skew if they are not parallel and do not intersect
  sorry

-- Define what it means for two lines to be parallel
def are_parallel (l1 l2 : Line3D) : Prop :=
  -- Two lines are parallel if they have the same direction vector
  sorry

-- Define what it means for two lines to intersect
def do_intersect (l1 l2 : Line3D) : Prop :=
  -- Two lines intersect if they share a point
  sorry

-- The main theorem
theorem skew_parallel_relationship (a b c : Line3D) :
  are_skew a b → are_parallel a c → (are_skew c b ∨ do_intersect c b) :=
by
  sorry

end NUMINAMATH_CALUDE_skew_parallel_relationship_l1710_171006


namespace NUMINAMATH_CALUDE_salary_expenditure_l1710_171085

theorem salary_expenditure (salary : ℝ) (rent_fraction : ℝ) (clothes_fraction : ℝ) (remaining : ℝ) 
  (h1 : salary = 170000)
  (h2 : rent_fraction = 1/10)
  (h3 : clothes_fraction = 3/5)
  (h4 : remaining = 17000)
  (h5 : remaining / salary + rent_fraction + clothes_fraction < 1) :
  let food_fraction := 1 - (remaining / salary + rent_fraction + clothes_fraction)
  food_fraction = 1/5 := by
sorry

end NUMINAMATH_CALUDE_salary_expenditure_l1710_171085


namespace NUMINAMATH_CALUDE_max_a_for_four_near_zero_points_l1710_171014

/-- A quadratic function f(x) = ax^2 + bx + c -/
def quadratic_function (a b c : ℝ) : ℝ → ℝ := λ x => a * x^2 + b * x + c

/-- Definition of a "near-zero point" -/
def is_near_zero_point (f : ℝ → ℝ) (x : ℤ) : Prop :=
  |f x| ≤ 1/4

theorem max_a_for_four_near_zero_points (a b c : ℝ) (ha : a > 0) :
  (∃ x₁ x₂ x₃ x₄ : ℤ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄ ∧
    is_near_zero_point (quadratic_function a b c) x₁ ∧
    is_near_zero_point (quadratic_function a b c) x₂ ∧
    is_near_zero_point (quadratic_function a b c) x₃ ∧
    is_near_zero_point (quadratic_function a b c) x₄) →
  a ≤ 1/4 :=
sorry

end NUMINAMATH_CALUDE_max_a_for_four_near_zero_points_l1710_171014


namespace NUMINAMATH_CALUDE_least_positive_even_congruence_l1710_171032

theorem least_positive_even_congruence : ∃ x : ℕ, 
  (x + 3721 : ℤ) ≡ 1547 [ZMOD 12] ∧ 
  Even x ∧
  x = 2 ∧
  ∀ y : ℕ, y < x → ¬((y + 3721 : ℤ) ≡ 1547 [ZMOD 12] ∧ Even y) := by
  sorry

end NUMINAMATH_CALUDE_least_positive_even_congruence_l1710_171032


namespace NUMINAMATH_CALUDE_roots_of_equation_l1710_171054

theorem roots_of_equation (x : ℝ) : 
  (3 * Real.sqrt x + 3 / Real.sqrt x = 7) ↔ 
  (x = ((7 + Real.sqrt 13) / 6)^2 ∨ x = ((7 - Real.sqrt 13) / 6)^2) :=
by sorry

end NUMINAMATH_CALUDE_roots_of_equation_l1710_171054


namespace NUMINAMATH_CALUDE_coffee_maker_discount_l1710_171069

/-- Calculates the discount amount given the original price and discounted price. -/
def discount_amount (original_price discounted_price : ℝ) : ℝ :=
  original_price - discounted_price

/-- Proves that the discount amount is 20 dollars for a coffee maker with an original price
    of 90 dollars and a discounted price of 70 dollars. -/
theorem coffee_maker_discount : discount_amount 90 70 = 20 := by
  sorry

end NUMINAMATH_CALUDE_coffee_maker_discount_l1710_171069


namespace NUMINAMATH_CALUDE_custom_mult_square_identity_l1710_171046

-- Define the custom multiplication operation
def custom_mult (a b : ℝ) : ℝ := (a - b)^2

-- Theorem statement
theorem custom_mult_square_identity (x y : ℝ) :
  custom_mult (x^2) (y^2) = (x + y)^2 * (x - y)^2 := by
  sorry

end NUMINAMATH_CALUDE_custom_mult_square_identity_l1710_171046


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1710_171075

theorem complex_equation_solution (z : ℂ) : (1 + z * Complex.I = z + Complex.I) → z = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1710_171075


namespace NUMINAMATH_CALUDE_delta_k_zero_iff_ge_four_l1710_171087

def u (n : ℕ) : ℕ := n^3 + n

def Δ : (ℕ → ℕ) → (ℕ → ℕ)
  | f => λ n => f (n + 1) - f n

def Δk : ℕ → (ℕ → ℕ) → (ℕ → ℕ)
  | 0 => id
  | k + 1 => Δ ∘ Δk k

theorem delta_k_zero_iff_ge_four (k : ℕ) :
  (∀ n, Δk k u n = 0) ↔ k ≥ 4 := by sorry

end NUMINAMATH_CALUDE_delta_k_zero_iff_ge_four_l1710_171087


namespace NUMINAMATH_CALUDE_x_percent_plus_six_equals_ten_l1710_171037

theorem x_percent_plus_six_equals_ten (x : ℝ) (h1 : x > 0) 
  (h2 : x * (x / 100) + 6 = 10) : x = 20 := by
  sorry

end NUMINAMATH_CALUDE_x_percent_plus_six_equals_ten_l1710_171037


namespace NUMINAMATH_CALUDE_min_value_expression_l1710_171044

theorem min_value_expression (x y z k : ℝ) 
  (hx : -2 < x ∧ x < 2) 
  (hy : -2 < y ∧ y < 2) 
  (hz : -2 < z ∧ z < 2) 
  (hk : k > 0) :
  (k / ((2 - x) * (2 - y) * (2 - z))) + (k / ((2 + x) * (2 + y) * (2 + z))) ≥ 2 * k :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l1710_171044


namespace NUMINAMATH_CALUDE_associates_hired_to_change_ratio_l1710_171019

/-- The number of additional associates hired to change the ratio -/
def additional_associates (initial_ratio_partners initial_ratio_associates new_ratio_partners new_ratio_associates current_partners : ℕ) : ℕ :=
  let initial_associates := (initial_ratio_associates * current_partners) / initial_ratio_partners
  let total_new_associates := (new_ratio_associates * current_partners) / new_ratio_partners
  total_new_associates - initial_associates

/-- Theorem stating that 50 additional associates were hired to change the ratio -/
theorem associates_hired_to_change_ratio :
  additional_associates 2 63 1 34 20 = 50 := by
  sorry

end NUMINAMATH_CALUDE_associates_hired_to_change_ratio_l1710_171019


namespace NUMINAMATH_CALUDE_no_playful_numbers_l1710_171062

/-- A two-digit positive integer is playful if it equals the sum of the cube of its tens digit and the square of its units digit. -/
def IsPlayful (n : ℕ) : Prop :=
  n ≥ 10 ∧ n ≤ 99 ∧ ∃ (a b : ℕ), a ≥ 1 ∧ a ≤ 9 ∧ b ≤ 9 ∧ n = 10 * a + b ∧ n = a^3 + b^2

/-- The number of playful two-digit positive integers is zero. -/
theorem no_playful_numbers : ∀ n : ℕ, ¬(IsPlayful n) := by sorry

end NUMINAMATH_CALUDE_no_playful_numbers_l1710_171062


namespace NUMINAMATH_CALUDE_probability_four_twos_l1710_171097

def num_dice : ℕ := 12
def num_sides : ℕ := 8
def target_number : ℕ := 2
def num_success : ℕ := 4

theorem probability_four_twos : 
  (Nat.choose num_dice num_success : ℚ) * (1 / num_sides : ℚ)^num_success * ((num_sides - 1) / num_sides : ℚ)^(num_dice - num_success) = 
  495 * (1 / 4096 : ℚ) * (5764801 / 16777216 : ℚ) := by
sorry

end NUMINAMATH_CALUDE_probability_four_twos_l1710_171097


namespace NUMINAMATH_CALUDE_difference_largest_smallest_valid_numbers_l1710_171077

def is_valid_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999 ∧
  (n / 100 ≠ (n / 10) % 10) ∧
  (n / 100 ≠ n % 10) ∧
  ((n / 10) % 10 ≠ n % 10) ∧
  (n / 100 - (n / 10) % 10 = (n / 10) % 10 - n % 10)

def largest_valid_number : ℕ := 951

def smallest_valid_number : ℕ := 159

theorem difference_largest_smallest_valid_numbers :
  largest_valid_number - smallest_valid_number = 792 ∧
  is_valid_number largest_valid_number ∧
  is_valid_number smallest_valid_number ∧
  ∀ n : ℕ, is_valid_number n → 
    smallest_valid_number ≤ n ∧ n ≤ largest_valid_number := by
  sorry

end NUMINAMATH_CALUDE_difference_largest_smallest_valid_numbers_l1710_171077


namespace NUMINAMATH_CALUDE_smallest_k_for_cosine_equation_l1710_171043

theorem smallest_k_for_cosine_equation :
  let f : ℕ → Prop := λ k => Real.cos (k^2 + 8^2 : ℝ)^2 = 1
  ∃ (k₁ k₂ : ℕ), k₁ < k₂ ∧ f k₁ ∧ f k₂ ∧ k₁ = 10 ∧ k₂ = 12 ∧
    ∀ (k : ℕ), 0 < k ∧ k < k₁ → ¬f k :=
by sorry

end NUMINAMATH_CALUDE_smallest_k_for_cosine_equation_l1710_171043


namespace NUMINAMATH_CALUDE_altitude_equation_tangent_lines_equal_intercepts_l1710_171029

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 2*y = 0

-- Define the center of circle C
def center_C : ℝ × ℝ := (-1, 1)

-- Define points A and B
def point_A : ℝ × ℝ := (4, 0)
def point_B : ℝ × ℝ := (0, -2)

-- Theorem for the altitude equation
theorem altitude_equation :
  ∃ (x y : ℝ), 2*x + y + 1 = 0 ∧
  (∀ (p : ℝ × ℝ), p.1 = x ∧ p.2 = y →
    (p.2 - center_C.2) = -2 * (p.1 - center_C.1) ∧
    (p.2 - point_A.2) * (point_B.1 - point_A.1) = -(p.1 - point_A.1) * (point_B.2 - point_A.2)) :=
sorry

-- Theorem for tangent lines with equal intercepts
theorem tangent_lines_equal_intercepts :
  (∀ (x y : ℝ), (x - y = 0 ∨ x + y - 2 = 0 ∨ x + y + 2 = 0) →
    (∃ (t : ℝ), x = t ∧ y = t) ∨
    (∃ (t : ℝ), x = t ∧ y = 2 - t) ∨
    (∃ (t : ℝ), x = t ∧ y = -2 - t)) ∧
  (∀ (x y : ℝ), ((∃ (t : ℝ), x = t ∧ y = t) ∨
                 (∃ (t : ℝ), x = t ∧ y = 2 - t) ∨
                 (∃ (t : ℝ), x = t ∧ y = -2 - t)) →
    (x - center_C.1)^2 + (y - center_C.2)^2 = 2) :=
sorry

end NUMINAMATH_CALUDE_altitude_equation_tangent_lines_equal_intercepts_l1710_171029


namespace NUMINAMATH_CALUDE_base_conversion_theorem_l1710_171070

/-- Converts a base 6 number to decimal --/
def base6ToDecimal (n : ℕ) : ℕ :=
  (n / 100) * 36 + ((n / 10) % 10) * 6 + (n % 10)

/-- Converts a number in base b to decimal --/
def baseBToDecimal (n : ℕ) (b : ℕ) : ℕ :=
  (n / 100) * b^2 + ((n / 10) % 10) * b + (n % 10)

/-- The theorem stating that 5 is the unique positive integer b satisfying 142₆ = 215ᵦ --/
theorem base_conversion_theorem :
  ∃! (b : ℕ), b > 0 ∧ base6ToDecimal 142 = baseBToDecimal 215 b :=
by
  sorry

end NUMINAMATH_CALUDE_base_conversion_theorem_l1710_171070


namespace NUMINAMATH_CALUDE_paper_stack_height_l1710_171001

/-- Given a package of paper with known thickness and number of sheets,
    calculate the height of a stack with a different number of sheets. -/
theorem paper_stack_height
  (package_sheets : ℕ)
  (package_thickness : ℝ)
  (stack_sheets : ℕ)
  (h_package_sheets : package_sheets = 400)
  (h_package_thickness : package_thickness = 4)
  (h_stack_sheets : stack_sheets = 1000) :
  (stack_sheets : ℝ) * package_thickness / package_sheets = 10 :=
sorry

end NUMINAMATH_CALUDE_paper_stack_height_l1710_171001


namespace NUMINAMATH_CALUDE_quadratic_expression_l1710_171018

/-- A quadratic function passing through the point (3, 10) -/
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

/-- The condition that f passes through (3, 10) -/
def passes_through (a b c : ℝ) : Prop := f a b c 3 = 10

theorem quadratic_expression (a b c : ℝ) (h : passes_through a b c) :
  5 * a - 3 * b + c = -4 * a - 6 * b + 10 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_expression_l1710_171018


namespace NUMINAMATH_CALUDE_mishas_current_dollars_l1710_171089

theorem mishas_current_dollars (current_dollars target_dollars needed_dollars : ℕ) 
  (h1 : target_dollars = 47)
  (h2 : needed_dollars = 13)
  (h3 : current_dollars + needed_dollars = target_dollars) :
  current_dollars = 34 := by
  sorry

end NUMINAMATH_CALUDE_mishas_current_dollars_l1710_171089


namespace NUMINAMATH_CALUDE_interest_rate_calculation_l1710_171082

/-- Proves that the interest rate at which A lends to B is 8% per annum -/
theorem interest_rate_calculation (principal : ℝ) (rate_C : ℝ) (time : ℝ) (gain_B : ℝ)
  (h1 : principal = 3150)
  (h2 : rate_C = 12.5)
  (h3 : time = 2)
  (h4 : gain_B = 283.5) :
  let interest_C := principal * rate_C / 100 * time
  let rate_A := (interest_C - gain_B) / (principal * time) * 100
  rate_A = 8 := by sorry

end NUMINAMATH_CALUDE_interest_rate_calculation_l1710_171082


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1710_171098

theorem inequality_solution_set (a : ℝ) : 
  (∀ x : ℝ, (a - 2) * x^2 + 4 * (a - 2) * x - 4 < 0) ↔ (1 < a ∧ a ≤ 2) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1710_171098


namespace NUMINAMATH_CALUDE_perpendicular_planes_parallel_l1710_171078

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between a line and a plane
variable (perpendicular : Line → Plane → Prop)

-- Define the parallel relation between two planes
variable (parallel : Plane → Plane → Prop)

-- State the theorem
theorem perpendicular_planes_parallel 
  (m : Line) (α β : Plane) 
  (h1 : perpendicular m α) 
  (h2 : perpendicular m β) : 
  parallel α β := by sorry

end NUMINAMATH_CALUDE_perpendicular_planes_parallel_l1710_171078


namespace NUMINAMATH_CALUDE_stating_convex_polygon_decomposition_iff_centrally_symmetric_l1710_171090

/-- A convex polygon. -/
structure ConvexPolygon where
  -- Add necessary fields and conditions for a convex polygon
  is_convex : Bool

/-- A parallelogram. -/
structure Parallelogram where
  -- Add necessary fields for a parallelogram

/-- Represents a decomposition of a polygon into parallelograms. -/
def Decomposition (p : ConvexPolygon) := List Parallelogram

/-- Checks if a decomposition is valid for a given polygon. -/
def is_valid_decomposition (p : ConvexPolygon) (d : Decomposition p) : Prop :=
  sorry

/-- Checks if a polygon is centrally symmetric. -/
def is_centrally_symmetric (p : ConvexPolygon) : Prop :=
  sorry

/-- 
Theorem stating that a convex polygon can be decomposed into a finite number of parallelograms 
if and only if it is centrally symmetric.
-/
theorem convex_polygon_decomposition_iff_centrally_symmetric (p : ConvexPolygon) :
  (∃ d : Decomposition p, is_valid_decomposition p d) ↔ is_centrally_symmetric p :=
sorry

end NUMINAMATH_CALUDE_stating_convex_polygon_decomposition_iff_centrally_symmetric_l1710_171090


namespace NUMINAMATH_CALUDE_parabola_circle_tangent_l1710_171065

/-- Represents a parabola in the form y^2 = -8x --/
structure Parabola where
  equation : ℝ → ℝ → Prop
  focus : ℝ × ℝ
  directrix : ℝ → Prop

/-- Represents a circle --/
structure Circle where
  center : ℝ × ℝ
  passes_through : ℝ × ℝ

/-- Theorem: The line x = 2 is a common tangent to all circles whose centers 
    lie on the parabola y^2 = -8x and pass through the point (-2, 0) --/
theorem parabola_circle_tangent (p : Parabola) (c : Circle) : 
  p.equation = (fun x y ↦ y^2 = -8*x) →
  p.focus = (-2, 0) →
  p.directrix = (fun x ↦ x = 2) →
  c.passes_through = (-2, 0) →
  (∃ (y : ℝ), p.equation c.center.1 y) →
  (fun x ↦ x = 2) = (fun x ↦ ∃ (y : ℝ), c.center = (x, y) ∧ 
    (c.center.1 - (-2))^2 + (c.center.2 - 0)^2 = (c.center.1 - 2)^2 + c.center.2^2) :=
by sorry

end NUMINAMATH_CALUDE_parabola_circle_tangent_l1710_171065


namespace NUMINAMATH_CALUDE_inequality_solution_l1710_171086

theorem inequality_solution (x : Real) : 
  0 ≤ x ∧ x ≤ 2 * Real.pi →
  (2 * Real.cos x ≤ |Real.sqrt (1 + Real.sin (2 * x)) - Real.sqrt (1 - Real.sin (2 * x))| ∧
   |Real.sqrt (1 + Real.sin (2 * x)) - Real.sqrt (1 - Real.sin (2 * x))| ≤ Real.sqrt 2) ↔
  ((Real.pi / 4 ≤ x ∧ x ≤ Real.pi / 2) ∨ (3 * Real.pi / 2 ≤ x ∧ x ≤ 7 * Real.pi / 4)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l1710_171086


namespace NUMINAMATH_CALUDE_solution_set_abs_x_squared_minus_two_lt_two_l1710_171040

theorem solution_set_abs_x_squared_minus_two_lt_two :
  {x : ℝ | |x^2 - 2| < 2} = {x : ℝ | -2 < x ∧ x < 0 ∨ 0 < x ∧ x < 2} := by
  sorry

end NUMINAMATH_CALUDE_solution_set_abs_x_squared_minus_two_lt_two_l1710_171040


namespace NUMINAMATH_CALUDE_same_gender_probability_theorem_l1710_171096

/-- Represents a school with a certain number of male and female teachers -/
structure School where
  male_teachers : ℕ
  female_teachers : ℕ

/-- The probability of selecting two teachers of the same gender from two schools -/
def same_gender_probability (school_a school_b : School) : ℚ :=
  let total_combinations := (school_a.male_teachers + school_a.female_teachers) * (school_b.male_teachers + school_b.female_teachers)
  let same_gender_combinations := school_a.male_teachers * school_b.male_teachers + school_a.female_teachers * school_b.female_teachers
  same_gender_combinations / total_combinations

/-- Theorem stating that the probability of selecting two teachers of the same gender
    from the given schools is 4/9 -/
theorem same_gender_probability_theorem :
  let school_a := School.mk 2 1
  let school_b := School.mk 1 2
  same_gender_probability school_a school_b = 4 / 9 := by
  sorry

end NUMINAMATH_CALUDE_same_gender_probability_theorem_l1710_171096


namespace NUMINAMATH_CALUDE_triangle_properties_l1710_171007

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  0 < A → A < π →
  0 < B → B < π →
  0 < C → C < π →
  A + B + C = π →
  b + a * Real.cos C = 0 →
  Real.sin A = 2 * Real.sin (A + C) →
  C = 2 * π / 3 ∧ c / a = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l1710_171007


namespace NUMINAMATH_CALUDE_curve_tangent_sum_l1710_171093

/-- The curve C defined by y = x^3 - x^2 - ax + b -/
def C (x y a b : ℝ) : Prop := y = x^3 - x^2 - a*x + b

/-- The derivative of C with respect to x -/
def C_derivative (x a : ℝ) : ℝ := 3*x^2 - 2*x - a

theorem curve_tangent_sum (a b : ℝ) : 
  C 0 1 a b ∧ C_derivative 0 a = 2 → a + b = -1 := by sorry

end NUMINAMATH_CALUDE_curve_tangent_sum_l1710_171093


namespace NUMINAMATH_CALUDE_infinite_decimal_digits_l1710_171030

/-- The decimal representation of 1 / (2^3 * 5^4 * 3^2) has infinitely many digits after the decimal point. -/
theorem infinite_decimal_digits (n : ℕ) : ∃ (k : ℕ), k > n ∧ 
  (10^k * (1 : ℚ) / (2^3 * 5^4 * 3^2)).num ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_infinite_decimal_digits_l1710_171030


namespace NUMINAMATH_CALUDE_deck_size_l1710_171011

theorem deck_size (toothpicks_per_card : ℕ) (unused_cards : ℕ) (boxes : ℕ) (toothpicks_per_box : ℕ) :
  toothpicks_per_card = 75 →
  unused_cards = 16 →
  boxes = 6 →
  toothpicks_per_box = 450 →
  boxes * toothpicks_per_box / toothpicks_per_card + unused_cards = 52 :=
by
  sorry

end NUMINAMATH_CALUDE_deck_size_l1710_171011


namespace NUMINAMATH_CALUDE_hotel_beds_count_l1710_171021

theorem hotel_beds_count (total_rooms : ℕ) (two_bed_rooms : ℕ) (beds_in_two_bed_room : ℕ) (beds_in_three_bed_room : ℕ) 
    (h1 : total_rooms = 13)
    (h2 : two_bed_rooms = 8)
    (h3 : beds_in_two_bed_room = 2)
    (h4 : beds_in_three_bed_room = 3) :
  two_bed_rooms * beds_in_two_bed_room + (total_rooms - two_bed_rooms) * beds_in_three_bed_room = 31 := by
  sorry

#eval 8 * 2 + (13 - 8) * 3  -- This should output 31

end NUMINAMATH_CALUDE_hotel_beds_count_l1710_171021


namespace NUMINAMATH_CALUDE_river_length_estimate_l1710_171026

/-- Represents a measurement with an associated error probability -/
structure Measurement where
  value : ℝ
  error : ℝ
  errorProb : ℝ

/-- Calculates the best estimate and error probability given two measurements -/
def calculateEstimate (m1 m2 : Measurement) : ℝ × ℝ :=
  sorry

theorem river_length_estimate 
  (gsa awra : Measurement)
  (h1 : gsa.value = 402)
  (h2 : gsa.error = 0.5)
  (h3 : gsa.errorProb = 0.04)
  (h4 : awra.value = 403)
  (h5 : awra.error = 0.5)
  (h6 : awra.errorProb = 0.04) :
  calculateEstimate gsa awra = (402.5, 0.04) :=
sorry

end NUMINAMATH_CALUDE_river_length_estimate_l1710_171026


namespace NUMINAMATH_CALUDE_mary_fruits_left_l1710_171055

def fruits_left (initial_apples initial_oranges initial_blueberries eaten : ℕ) : ℕ :=
  (initial_apples - eaten) + (initial_oranges - eaten) + (initial_blueberries - eaten)

theorem mary_fruits_left : fruits_left 14 9 6 1 = 26 := by
  sorry

end NUMINAMATH_CALUDE_mary_fruits_left_l1710_171055


namespace NUMINAMATH_CALUDE_compute_expression_l1710_171053

theorem compute_expression : 3 * 3^4 - 27^60 / 27^58 = -486 := by
  sorry

end NUMINAMATH_CALUDE_compute_expression_l1710_171053


namespace NUMINAMATH_CALUDE_chocolate_discount_l1710_171057

/-- Calculates the discount amount given the original price and final price -/
def discount (original_price final_price : ℚ) : ℚ :=
  original_price - final_price

/-- Proves that the discount on a chocolate with original price $2.00 and final price $1.43 is $0.57 -/
theorem chocolate_discount :
  let original_price : ℚ := 2
  let final_price : ℚ := 143/100
  discount original_price final_price = 57/100 := by
sorry

end NUMINAMATH_CALUDE_chocolate_discount_l1710_171057


namespace NUMINAMATH_CALUDE_exam_question_distribution_l1710_171042

theorem exam_question_distribution (total_questions : ℕ) 
  (group_a_marks : ℕ → ℕ) (group_b_marks : ℕ → ℕ) (group_c_marks : ℕ → ℕ)
  (group_b_count : ℕ) :
  total_questions = 100 →
  group_b_count = 23 →
  (∀ n, group_a_marks n = n) →
  (∀ n, group_b_marks n = 2 * n) →
  (∀ n, group_c_marks n = 3 * n) →
  (∀ a b c, a + b + c = total_questions → a ≥ 1 ∧ b ≥ 1 ∧ c ≥ 1) →
  (∀ a b c, a + b + c = total_questions → 
    group_a_marks a ≥ (3 * (group_a_marks a + group_b_marks b + group_c_marks c)) / 5) →
  (∀ a b c, a + b + c = total_questions → 
    group_b_marks b ≤ (group_a_marks a + group_b_marks b + group_c_marks c) / 4) →
  ∃ a c, a + group_b_count + c = total_questions ∧ c = 1 :=
by sorry

end NUMINAMATH_CALUDE_exam_question_distribution_l1710_171042


namespace NUMINAMATH_CALUDE_lines_relationship_l1710_171068

/-- A line in 3D space -/
structure Line3D where
  point : ℝ × ℝ × ℝ
  direction : ℝ × ℝ × ℝ

/-- Predicate for non-coplanar lines -/
def nonCoplanar (l₁ l₂ : Line3D) : Prop := sorry

/-- Predicate for intersecting lines -/
def intersects (l₁ l₂ : Line3D) : Prop := sorry

/-- Theorem: Given non-coplanar lines l₁ and l₂, and lines m₁ and m₂ that both intersect with l₁ and l₂,
    the positional relationship between m₁ and m₂ is either intersecting or non-coplanar -/
theorem lines_relationship (l₁ l₂ m₁ m₂ : Line3D)
  (h₁ : nonCoplanar l₁ l₂)
  (h₂ : intersects m₁ l₁)
  (h₃ : intersects m₁ l₂)
  (h₄ : intersects m₂ l₁)
  (h₅ : intersects m₂ l₂) :
  intersects m₁ m₂ ∨ nonCoplanar m₁ m₂ := by
  sorry

end NUMINAMATH_CALUDE_lines_relationship_l1710_171068


namespace NUMINAMATH_CALUDE_milk_mixture_problem_l1710_171081

/-- Proves that the volume removed and replaced with water is 50 litres -/
theorem milk_mixture_problem (total_volume : ℝ) (initial_milk : ℝ) (final_concentration : ℝ) :
  total_volume = 100 →
  initial_milk = 36 →
  final_concentration = 0.09 →
  ∃ (V : ℝ), V = 50 ∧
    (initial_milk / total_volume) * (1 - V / total_volume)^2 = final_concentration :=
by sorry

end NUMINAMATH_CALUDE_milk_mixture_problem_l1710_171081


namespace NUMINAMATH_CALUDE_total_distance_rowed_l1710_171028

/-- The total distance traveled by a man rowing upstream and downstream -/
theorem total_distance_rowed (man_speed : ℝ) (river_speed : ℝ) (total_time : ℝ) : 
  man_speed = 6 →
  river_speed = 1.2 →
  total_time = 1 →
  2 * (total_time * man_speed * river_speed) / (man_speed + river_speed) = 5.76 :=
by
  sorry

end NUMINAMATH_CALUDE_total_distance_rowed_l1710_171028


namespace NUMINAMATH_CALUDE_cone_surface_area_l1710_171050

/-- 
Given a cone with slant height 2 and lateral surface that unfolds into a semicircle,
prove that its surface area is 3π.
-/
theorem cone_surface_area (h : ℝ) (r : ℝ) : 
  h = 2 ∧ 2 * π * r = 2 * π → π * r * h + π * r^2 = 3 * π :=
by sorry

end NUMINAMATH_CALUDE_cone_surface_area_l1710_171050


namespace NUMINAMATH_CALUDE_curve_and_lines_distance_properties_l1710_171031

/-- Given a curve C and lines l and l1 in a 2D plane, prove properties about distances -/
theorem curve_and_lines_distance_properties
  (B : ℝ × ℝ)
  (C : ℝ → ℝ × ℝ)
  (A : ℝ × ℝ)
  (l l1 : ℝ × ℝ → Prop)
  (h_B : B = (1, 1))
  (h_C : ∀ θ, C θ = (2 * Real.cos θ, Real.sqrt 3 * Real.sin θ))
  (h_A : A = (4 * Real.sqrt 2 * Real.cos (π/4), 4 * Real.sqrt 2 * Real.sin (π/4)))
  (h_l : ∃ a, ∀ ρ θ, l (ρ * Real.cos θ, ρ * Real.sin θ) ↔ ρ * Real.cos (θ - π/4) = a)
  (h_l_A : l A)
  (h_l1_parallel : ∃ k, ∀ p, l1 p ↔ l (p.1 - k, p.2 - k))
  (h_l1_B : l1 B)
  (h_l1_intersect : ∃ M N, M ≠ N ∧ l1 (C M) ∧ l1 (C N)) :
  (∃ d, d = (8 * Real.sqrt 2 - Real.sqrt 14) / 2 ∧
    ∀ p, (∃ θ, C θ = p) → 
      ∀ q, l q → Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) ≥ d) ∧
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 12 * Real.sqrt 2 / 7 :=
sorry

end NUMINAMATH_CALUDE_curve_and_lines_distance_properties_l1710_171031


namespace NUMINAMATH_CALUDE_range_of_b_l1710_171038

theorem range_of_b (a b c : ℝ) (h1 : a * c = b^2) (h2 : a + b + c = 3) :
  -3 ≤ b ∧ b ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_b_l1710_171038


namespace NUMINAMATH_CALUDE_max_checkers_theorem_l1710_171047

/-- Represents a chessboard configuration -/
structure ChessboardConfig where
  size : Nat
  white_checkers : Nat
  black_checkers : Nat

/-- Checks if a chessboard configuration is valid -/
def is_valid_config (c : ChessboardConfig) : Prop :=
  c.size = 8 ∧
  c.white_checkers = 2 * c.black_checkers ∧
  c.white_checkers + c.black_checkers ≤ c.size * c.size

/-- The maximum number of checkers that can be placed -/
def max_checkers : Nat := 48

/-- Theorem: The maximum number of checkers that can be placed on an 8x8 chessboard,
    such that each row and column contains twice as many white checkers as black ones, is 48 -/
theorem max_checkers_theorem (c : ChessboardConfig) :
  is_valid_config c → c.white_checkers + c.black_checkers ≤ max_checkers :=
by
  sorry

#check max_checkers_theorem

end NUMINAMATH_CALUDE_max_checkers_theorem_l1710_171047


namespace NUMINAMATH_CALUDE_variable_order_l1710_171073

theorem variable_order (a b c d : ℝ) (h : a - 1 = b + 2 ∧ b + 2 = c - 3 ∧ c - 3 = d + 4) : 
  c > a ∧ a > b ∧ b > d := by
  sorry

end NUMINAMATH_CALUDE_variable_order_l1710_171073


namespace NUMINAMATH_CALUDE_sum_of_tens_and_units_digits_of_8_pow_2003_l1710_171013

/-- The sum of the tens digit and the units digit in the decimal representation of 8^2003 is 2 -/
theorem sum_of_tens_and_units_digits_of_8_pow_2003 : ∃ (a b : ℕ), a < 10 ∧ b < 10 ∧ a + b = 2 ∧ 
  (∃ (k : ℕ), 8^2003 = k * 100 + a * 10 + b) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_tens_and_units_digits_of_8_pow_2003_l1710_171013


namespace NUMINAMATH_CALUDE_average_b_c_l1710_171020

theorem average_b_c (a b c : ℝ) 
  (h1 : (a + b) / 2 = 45)
  (h2 : c - a = 50) :
  (b + c) / 2 = 70 := by
  sorry

end NUMINAMATH_CALUDE_average_b_c_l1710_171020


namespace NUMINAMATH_CALUDE_ellipse_slope_product_l1710_171083

theorem ellipse_slope_product (x₁ y₁ x₂ y₂ : ℝ) :
  x₁ ≠ 0 →
  y₁ ≠ 0 →
  x₁^2 + 4*y₁^2/9 = 1 →
  x₂^2 + 4*y₂^2/9 = 1 →
  3*y₁/(4*x₁) = (y₁ + y₂)/(x₁ + x₂) →
  (y₁/x₁) * ((y₁ - y₂)/(x₁ - x₂)) = -1 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_slope_product_l1710_171083


namespace NUMINAMATH_CALUDE_tangent_product_simplification_l1710_171048

theorem tangent_product_simplification :
  (∀ x y, Real.tan (x + y) = (Real.tan x + Real.tan y) / (1 - Real.tan x * Real.tan y)) →
  Real.tan (45 * π / 180) = 1 →
  (1 + Real.tan (15 * π / 180)) * (1 + Real.tan (30 * π / 180)) = 2 := by
sorry

end NUMINAMATH_CALUDE_tangent_product_simplification_l1710_171048


namespace NUMINAMATH_CALUDE_bread_price_is_two_l1710_171072

/-- The price of a can of spam in dollars -/
def spam_price : ℚ := 3

/-- The price of a jar of peanut butter in dollars -/
def peanut_butter_price : ℚ := 5

/-- The number of cans of spam bought -/
def spam_quantity : ℕ := 12

/-- The number of jars of peanut butter bought -/
def peanut_butter_quantity : ℕ := 3

/-- The number of loaves of bread bought -/
def bread_quantity : ℕ := 4

/-- The total amount paid in dollars -/
def total_paid : ℚ := 59

/-- The price of a loaf of bread in dollars -/
def bread_price : ℚ := 2

theorem bread_price_is_two :
  spam_price * spam_quantity +
  peanut_butter_price * peanut_butter_quantity +
  bread_price * bread_quantity = total_paid := by
  sorry

end NUMINAMATH_CALUDE_bread_price_is_two_l1710_171072


namespace NUMINAMATH_CALUDE_arithmetic_sequence_length_l1710_171084

theorem arithmetic_sequence_length (first last step : ℤ) (h : first ≥ last) : 
  (first - last) / step + 1 = (first - 44) / 4 + 1 → (first - 44) / 4 + 1 = 28 :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_length_l1710_171084


namespace NUMINAMATH_CALUDE_remainder_product_l1710_171092

theorem remainder_product (a b : ℕ) (ha : a > 0) (hb : b > 0) 
  (rem_a : a % 8 = 3) (rem_b : b % 6 = 5) : (a * b) % 48 = 15 := by
  sorry

end NUMINAMATH_CALUDE_remainder_product_l1710_171092


namespace NUMINAMATH_CALUDE_negative_root_range_l1710_171095

theorem negative_root_range (x a : ℝ) : 
  x < 0 → 
  (2/3)^x = (1+a)/(1-a) → 
  0 < a ∧ a < 1 :=
by sorry

end NUMINAMATH_CALUDE_negative_root_range_l1710_171095


namespace NUMINAMATH_CALUDE_probability_of_quarter_l1710_171041

def quarter_value : ℚ := 25 / 100
def nickel_value : ℚ := 5 / 100
def penny_value : ℚ := 1 / 100

def total_quarter_value : ℚ := 10
def total_nickel_value : ℚ := 5
def total_penny_value : ℚ := 15

def num_quarters : ℕ := (total_quarter_value / quarter_value).num.toNat
def num_nickels : ℕ := (total_nickel_value / nickel_value).num.toNat
def num_pennies : ℕ := (total_penny_value / penny_value).num.toNat

def total_coins : ℕ := num_quarters + num_nickels + num_pennies

theorem probability_of_quarter : 
  (num_quarters : ℚ) / total_coins = 1 / 41 := by sorry

end NUMINAMATH_CALUDE_probability_of_quarter_l1710_171041


namespace NUMINAMATH_CALUDE_smallest_enclosing_circle_theorem_l1710_171016

/-- The radius of the smallest circle from which a triangle with sides 2, 3, and 4 can be cut out --/
def smallest_enclosing_circle_radius : ℝ := 2

/-- The three sides of the triangle --/
def triangle_sides : Fin 3 → ℝ
| 0 => 2
| 1 => 3
| 2 => 4
| _ => 0  -- This line is necessary due to Lean's totality requirement

theorem smallest_enclosing_circle_theorem :
  ∀ r : ℝ, (∀ i : Fin 3, triangle_sides i ≤ 2 * r) → r ≥ smallest_enclosing_circle_radius :=
by sorry

end NUMINAMATH_CALUDE_smallest_enclosing_circle_theorem_l1710_171016


namespace NUMINAMATH_CALUDE_quadratic_inequality_l1710_171064

theorem quadratic_inequality (x : ℝ) : 9 - x^2 < 0 → x < -3 ∨ x > 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l1710_171064


namespace NUMINAMATH_CALUDE_sofia_card_theorem_l1710_171088

theorem sofia_card_theorem (y : Real) (h1 : 0 < y) (h2 : y < Real.pi / 2) 
  (h3 : Real.tan y > Real.sin y) (h4 : Real.tan y > Real.cos y) : y = Real.pi / 4 :=
by sorry

end NUMINAMATH_CALUDE_sofia_card_theorem_l1710_171088


namespace NUMINAMATH_CALUDE_solution_to_congruence_l1710_171071

theorem solution_to_congruence (n : ℤ) : 
  0 ≤ n ∧ n < 103 ∧ (100 * n) % 103 = 34 % 103 → n = 52 := by
  sorry

end NUMINAMATH_CALUDE_solution_to_congruence_l1710_171071


namespace NUMINAMATH_CALUDE_snack_eaters_problem_l1710_171060

/-- The number of new outsiders who joined for snacks after the first group left -/
def new_outsiders : ℕ := sorry

theorem snack_eaters_problem (initial_people : ℕ) (initial_snackers : ℕ) (first_outsiders : ℕ) 
  (more_left : ℕ) (final_snackers : ℕ) :
  initial_people = 200 →
  initial_snackers = 100 →
  first_outsiders = 20 →
  more_left = 30 →
  final_snackers = 20 →
  new_outsiders = 40 := by sorry

end NUMINAMATH_CALUDE_snack_eaters_problem_l1710_171060


namespace NUMINAMATH_CALUDE_g_of_5_equals_15_l1710_171035

-- Define the function g
def g (x : ℝ) : ℝ := x^2 - 2*x

-- State the theorem
theorem g_of_5_equals_15 : g 5 = 15 := by sorry

end NUMINAMATH_CALUDE_g_of_5_equals_15_l1710_171035


namespace NUMINAMATH_CALUDE_cleaning_payment_l1710_171025

theorem cleaning_payment (payment_rate rooms_cleaned : ℚ) : 
  payment_rate = 13 / 3 → rooms_cleaned = 8 / 5 → payment_rate * rooms_cleaned = 104 / 15 := by
  sorry

end NUMINAMATH_CALUDE_cleaning_payment_l1710_171025


namespace NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l1710_171059

-- Problem 1
theorem simplify_expression_1 (x y : ℝ) :
  y * (x + y) + (x + y) * (x - y) = x^2 := by sorry

-- Problem 2
theorem simplify_expression_2 (m : ℝ) (h1 : m ≠ -1) (h2 : m^2 + 2*m + 1 ≠ 0) :
  ((2*m + 1) / (m + 1) + m - 1) / ((m + 2) / (m^2 + 2*m + 1)) = m^2 + m := by sorry

end NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l1710_171059


namespace NUMINAMATH_CALUDE_rogers_second_bag_pieces_l1710_171015

/-- Represents the number of candy bags each person has -/
def num_bags : ℕ := 2

/-- Represents the number of candy pieces in each of Sandra's bags -/
def sandra_bag_pieces : ℕ := 6

/-- Represents the number of candy pieces in Roger's first bag -/
def roger_first_bag_pieces : ℕ := 11

/-- Represents the difference in total candy pieces between Roger and Sandra -/
def difference : ℕ := 2

/-- Theorem stating the number of candy pieces in Roger's second bag -/
theorem rogers_second_bag_pieces :
  ∃ (x : ℕ), x = num_bags * sandra_bag_pieces + difference - roger_first_bag_pieces :=
sorry

end NUMINAMATH_CALUDE_rogers_second_bag_pieces_l1710_171015


namespace NUMINAMATH_CALUDE_prob_no_dessert_is_35_percent_l1710_171010

/-- Represents the probability of different order combinations -/
structure OrderProbabilities where
  dessert_coffee : ℝ
  dessert_only : ℝ
  coffee_only : ℝ
  appetizer_dessert : ℝ
  appetizer_coffee : ℝ
  appetizer_dessert_coffee : ℝ

/-- Calculate the probability of not ordering dessert -/
def prob_no_dessert (p : OrderProbabilities) : ℝ :=
  1 - (p.dessert_coffee + p.dessert_only + p.appetizer_dessert + p.appetizer_dessert_coffee)

/-- Theorem: The probability of not ordering dessert is 35% -/
theorem prob_no_dessert_is_35_percent (p : OrderProbabilities)
  (h1 : p.dessert_coffee = 0.60)
  (h2 : p.dessert_only = 0.15)
  (h3 : p.coffee_only = 0.10)
  (h4 : p.appetizer_dessert = 0.05)
  (h5 : p.appetizer_coffee = 0.08)
  (h6 : p.appetizer_dessert_coffee = 0.03) :
  prob_no_dessert p = 0.35 := by
  sorry

end NUMINAMATH_CALUDE_prob_no_dessert_is_35_percent_l1710_171010


namespace NUMINAMATH_CALUDE_pencil_count_theorem_pencils_in_drawer_l1710_171027

/-- The total number of pencils after adding more to the drawer -/
def total_pencils (initial : ℕ) (added : ℕ) : ℕ :=
  initial + added

/-- Theorem: The total number of pencils is the sum of initial and added pencils -/
theorem pencil_count_theorem (initial : ℕ) (added : ℕ) :
  total_pencils initial added = initial + added := by
  sorry

/-- Given information about the pencils in the drawer -/
def initial_pencils : ℕ := 41
def pencils_added : ℕ := 30

/-- The result we want to prove -/
theorem pencils_in_drawer :
  total_pencils initial_pencils pencils_added = 71 := by
  sorry

end NUMINAMATH_CALUDE_pencil_count_theorem_pencils_in_drawer_l1710_171027


namespace NUMINAMATH_CALUDE_original_ratio_first_term_l1710_171074

theorem original_ratio_first_term 
  (original_first : ℚ) 
  (original_second : ℚ) 
  (added_number : ℚ) 
  (new_ratio_first : ℚ) 
  (new_ratio_second : ℚ) :
  original_first / original_second = 4 / 15 →
  added_number = 29 →
  (original_first + added_number) / (original_second + added_number) = new_ratio_first / new_ratio_second →
  new_ratio_first / new_ratio_second = 3 / 4 →
  original_first = 4 :=
by sorry

end NUMINAMATH_CALUDE_original_ratio_first_term_l1710_171074


namespace NUMINAMATH_CALUDE_parabola_tangents_perpendicular_iff_P_on_line_l1710_171067

/-- Parabola C: x^2 = 4y -/
def parabola (x y : ℝ) : Prop := x^2 = 4*y

/-- Line l: y = -1 -/
def line (x y : ℝ) : Prop := y = -1

/-- Point P is on line l -/
def P_on_line (P : ℝ × ℝ) : Prop := line P.1 P.2

/-- PA and PB are perpendicular -/
def tangents_perpendicular (P A B : ℝ × ℝ) : Prop :=
  let slope_PA := (A.2 - P.2) / (A.1 - P.1)
  let slope_PB := (B.2 - P.2) / (B.1 - P.1)
  slope_PA * slope_PB = -1

/-- A and B are points on the parabola -/
def points_on_parabola (A B : ℝ × ℝ) : Prop :=
  parabola A.1 A.2 ∧ parabola B.1 B.2

/-- PA and PB are tangent to the parabola at A and B respectively -/
def tangent_lines (P A B : ℝ × ℝ) : Prop :=
  points_on_parabola A B ∧
  (A.2 - P.2) / (A.1 - P.1) = A.1 / 2 ∧
  (B.2 - P.2) / (B.1 - P.1) = B.1 / 2

theorem parabola_tangents_perpendicular_iff_P_on_line
  (P A B : ℝ × ℝ) :
  tangent_lines P A B →
  (P_on_line P ↔ tangents_perpendicular P A B) :=
sorry

end NUMINAMATH_CALUDE_parabola_tangents_perpendicular_iff_P_on_line_l1710_171067


namespace NUMINAMATH_CALUDE_equation_equals_twentyfour_l1710_171033

theorem equation_equals_twentyfour : 6 / (1 - 3 / 10) = 24 := by sorry

end NUMINAMATH_CALUDE_equation_equals_twentyfour_l1710_171033


namespace NUMINAMATH_CALUDE_abs_sum_eq_six_iff_in_interval_l1710_171012

theorem abs_sum_eq_six_iff_in_interval (x : ℝ) : 
  |x + 1| + |x - 5| = 6 ↔ x ∈ Set.Icc (-1) 5 := by
  sorry

end NUMINAMATH_CALUDE_abs_sum_eq_six_iff_in_interval_l1710_171012


namespace NUMINAMATH_CALUDE_factorial_300_trailing_zeros_l1710_171099

/-- The number of trailing zeros in n! -/
def trailingZeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125)

/-- Theorem: The number of trailing zeros in 300! is 74 -/
theorem factorial_300_trailing_zeros :
  trailingZeros 300 = 74 := by
  sorry

end NUMINAMATH_CALUDE_factorial_300_trailing_zeros_l1710_171099


namespace NUMINAMATH_CALUDE_expression_value_l1710_171056

theorem expression_value (x : ℝ) (h : 3 * x^2 - x - 1 = 0) :
  (2 * x + 3) * (2 * x - 3) - 2 * x * (1 - x) = -7 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l1710_171056


namespace NUMINAMATH_CALUDE_power_difference_l1710_171036

theorem power_difference (a m n : ℝ) (h1 : a^m = 8) (h2 : a^n = 2) : a^(m-3*n) = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_difference_l1710_171036


namespace NUMINAMATH_CALUDE_symmetric_points_sum_l1710_171004

/-- Two points are symmetric with respect to the origin if their coordinates are negatives of each other -/
def symmetric_wrt_origin (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₁ = -x₂ ∧ y₁ = -y₂

/-- The theorem states that if point A(a, 1) and point B(-2, b) are symmetric with respect to the origin,
    then a + b = 1 -/
theorem symmetric_points_sum (a b : ℝ) :
  symmetric_wrt_origin a 1 (-2) b → a + b = 1 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_sum_l1710_171004


namespace NUMINAMATH_CALUDE_parallel_line_through_point_l1710_171005

/-- Given a line L1 with equation x - 2y - 2 = 0, prove that the line L2 with equation x - 2y - 1 = 0
    passes through the point (1,0) and is parallel to L1. -/
theorem parallel_line_through_point (x y : ℝ) : 
  (x - 2*y - 1 = 0) ∧ 
  (1 - 2*0 - 1 = 0) ∧ 
  (∃ k : ℝ, k ≠ 0 ∧ (1 : ℝ) = k * 1 ∧ (-2 : ℝ) = k * (-2)) := by
  sorry

end NUMINAMATH_CALUDE_parallel_line_through_point_l1710_171005


namespace NUMINAMATH_CALUDE_two_digit_product_digits_l1710_171008

theorem two_digit_product_digits (a b : ℕ) (ha : 40 < a ∧ a < 100) (hb : 40 < b ∧ b < 100) :
  (1000 ≤ a * b ∧ a * b < 10000) ∨ (100 ≤ a * b ∧ a * b < 1000) :=
sorry

end NUMINAMATH_CALUDE_two_digit_product_digits_l1710_171008


namespace NUMINAMATH_CALUDE_correct_weighted_mean_l1710_171034

theorem correct_weighted_mean (n : ℕ) (incorrect_mean : ℚ) 
  (error1 error2 error3 : ℚ) (w1 w2 w3 : ℕ) (n1 n2 n3 : ℕ) :
  n = 40 →
  incorrect_mean = 150 →
  error1 = 165 - 135 →
  error2 = 200 - 170 →
  error3 = 185 - 155 →
  w1 = 2 →
  w2 = 3 →
  w3 = 4 →
  n1 = 10 →
  n2 = 20 →
  n3 = 10 →
  n = n1 + n2 + n3 →
  let total_error := error1 + error2 + error3
  let correct_sum := n * incorrect_mean + total_error
  let total_weight := n1 * w1 + n2 * w2 + n3 * w3
  let weighted_mean := correct_sum / total_weight
  weighted_mean = 50.75 := by
sorry

end NUMINAMATH_CALUDE_correct_weighted_mean_l1710_171034


namespace NUMINAMATH_CALUDE_terrell_weight_lifting_l1710_171058

/-- The number of times Terrell lifts the 25-pound weights -/
def original_lifts : ℕ := 10

/-- The weight of each 25-pound weight in pounds -/
def original_weight : ℕ := 25

/-- The number of weights Terrell lifts each time -/
def num_weights : ℕ := 3

/-- The weight of each 20-pound weight in pounds -/
def new_weight : ℕ := 20

/-- The total weight lifted with the original weights -/
def total_weight : ℕ := num_weights * original_weight * original_lifts

/-- The number of times Terrell must lift the new weights to lift the same total weight -/
def new_lifts : ℚ := total_weight / (num_weights * new_weight)

theorem terrell_weight_lifting :
  new_lifts = 12.5 := by sorry

end NUMINAMATH_CALUDE_terrell_weight_lifting_l1710_171058


namespace NUMINAMATH_CALUDE_president_secretary_choice_count_l1710_171066

/-- Represents the number of ways to choose a president and secretary of the same gender -/
def choose_president_and_secretary (total_members : ℕ) (boys : ℕ) (girls : ℕ) : ℕ :=
  (boys * (boys - 1)) + (girls * (girls - 1))

/-- Theorem: Given a society of 25 members (15 boys and 10 girls), 
    the number of ways to choose a president and a secretary of the same gender, 
    where no one can hold both positions, is equal to 300. -/
theorem president_secretary_choice_count :
  choose_president_and_secretary 25 15 10 = 300 := by
  sorry

end NUMINAMATH_CALUDE_president_secretary_choice_count_l1710_171066


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1710_171039

theorem inequality_solution_set (a : ℝ) (h : a > 1) :
  {x : ℝ | (1 - a) * (x - a) * (x - 1/a) < 0} = 
  {x : ℝ | x < 1/a ∨ x > a} :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1710_171039


namespace NUMINAMATH_CALUDE_special_functions_identity_l1710_171017

/-- Non-constant, differentiable functions satisfying certain conditions -/
class SpecialFunctions (f g : ℝ → ℝ) where
  non_constant_f : ∃ x y, f x ≠ f y
  non_constant_g : ∃ x y, g x ≠ g y
  differentiable_f : Differentiable ℝ f
  differentiable_g : Differentiable ℝ g
  condition1 : ∀ x y, f (x + y) = f x * f y - g x * g y
  condition2 : ∀ x y, g (x + y) = f x * g y + g x * f y
  condition3 : deriv f 0 = 0

/-- Theorem stating that f(x)^2 + g(x)^2 = 1 for all x ∈ ℝ -/
theorem special_functions_identity {f g : ℝ → ℝ} [SpecialFunctions f g] :
  ∀ x, f x ^ 2 + g x ^ 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_special_functions_identity_l1710_171017


namespace NUMINAMATH_CALUDE_remainder_theorem_l1710_171023

theorem remainder_theorem : ∃ q : ℤ, 2^160 + 160 = q * (2^80 + 2^40 + 1) + 159 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l1710_171023


namespace NUMINAMATH_CALUDE_triangle_area_l1710_171094

theorem triangle_area (base height : ℝ) (h1 : base = 4.5) (h2 : height = 6) :
  (base * height) / 2 = 13.5 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l1710_171094


namespace NUMINAMATH_CALUDE_fourth_vertex_coordinates_l1710_171051

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if four points form a parallelogram -/
def is_parallelogram (A B C D : Point) : Prop :=
  (B.x - A.x = D.x - C.x) ∧ (B.y - A.y = D.y - C.y)

/-- The theorem stating the possible coordinates of the fourth vertex of the parallelogram -/
theorem fourth_vertex_coordinates :
  let A : Point := ⟨0, -9⟩
  let B : Point := ⟨2, 6⟩
  let C : Point := ⟨4, 5⟩
  ∃ D : Point, (D = ⟨2, -10⟩ ∨ D = ⟨-2, -8⟩ ∨ D = ⟨6, 20⟩) ∧ 
    is_parallelogram A B C D :=
by
  sorry


end NUMINAMATH_CALUDE_fourth_vertex_coordinates_l1710_171051
