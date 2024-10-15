import Mathlib

namespace NUMINAMATH_CALUDE_gym_cost_is_twelve_l2738_273872

/-- Calculates the monthly cost of a gym membership given the total cost for 3 years and the down payment. -/
def monthly_gym_cost (total_cost : ℚ) (down_payment : ℚ) : ℚ :=
  (total_cost - down_payment) / (3 * 12)

/-- Theorem stating that the monthly cost of the gym is $12 under given conditions. -/
theorem gym_cost_is_twelve :
  let total_cost : ℚ := 482
  let down_payment : ℚ := 50
  monthly_gym_cost total_cost down_payment = 12 := by
  sorry

end NUMINAMATH_CALUDE_gym_cost_is_twelve_l2738_273872


namespace NUMINAMATH_CALUDE_total_money_l2738_273824

theorem total_money (A B C : ℕ) 
  (h1 : A + C = 200)
  (h2 : B + C = 350)
  (h3 : C = 50) :
  A + B + C = 500 := by
  sorry

end NUMINAMATH_CALUDE_total_money_l2738_273824


namespace NUMINAMATH_CALUDE_count_four_digit_numbers_l2738_273810

def four_digit_numbers_with_1_and_2 : ℕ :=
  let one_one := 4  -- 1 occurrence of 1, 3 occurrences of 2
  let two_ones := 6 -- 2 occurrences of 1, 2 occurrences of 2
  let three_ones := 4 -- 3 occurrences of 1, 1 occurrence of 2
  one_one + two_ones + three_ones

theorem count_four_digit_numbers : four_digit_numbers_with_1_and_2 = 14 := by
  sorry

end NUMINAMATH_CALUDE_count_four_digit_numbers_l2738_273810


namespace NUMINAMATH_CALUDE_total_photos_l2738_273840

def friends_photos : ℕ := 63
def family_photos : ℕ := 23

theorem total_photos : friends_photos + family_photos = 86 := by
  sorry

end NUMINAMATH_CALUDE_total_photos_l2738_273840


namespace NUMINAMATH_CALUDE_triangle_inequality_l2738_273805

theorem triangle_inequality (a b c : ℝ) (n : ℕ) (S : ℝ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_S : 2 * S = a + b + c) : 
  a^n / (b + c) + b^n / (c + a) + c^n / (a + b) ≥ (2/3)^(n-2) * S^(n-1) := by
sorry

end NUMINAMATH_CALUDE_triangle_inequality_l2738_273805


namespace NUMINAMATH_CALUDE_power_equation_solution_l2738_273883

theorem power_equation_solution (x : ℝ) : (2^4 * 3^6 : ℝ) = 9 * 6^x → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_power_equation_solution_l2738_273883


namespace NUMINAMATH_CALUDE_solution_set_is_open_unit_interval_l2738_273895

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- Define the properties of f
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)
def is_increasing_on_nonneg (f : ℝ → ℝ) : Prop := ∀ x y, 0 ≤ x → x < y → f x < f y

-- Define the set of x values satisfying f(2x-1) < f(1)
def solution_set (f : ℝ → ℝ) : Set ℝ := {x | f (2*x - 1) < f 1}

-- State the theorem
theorem solution_set_is_open_unit_interval (f : ℝ → ℝ) 
  (h_even : is_even f) (h_incr : is_increasing_on_nonneg f) : 
  solution_set f = Set.Ioo 0 1 := by sorry

end NUMINAMATH_CALUDE_solution_set_is_open_unit_interval_l2738_273895


namespace NUMINAMATH_CALUDE_total_cookies_l2738_273897

theorem total_cookies (num_people : ℕ) (cookies_per_person : ℕ) 
  (h1 : num_people = 6)
  (h2 : cookies_per_person = 4) :
  num_people * cookies_per_person = 24 := by
  sorry

end NUMINAMATH_CALUDE_total_cookies_l2738_273897


namespace NUMINAMATH_CALUDE_regular_polygon_properties_l2738_273822

/-- A regular polygon with exterior angles measuring 18 degrees -/
structure RegularPolygon where
  -- The number of sides
  sides : ℕ
  -- The measure of each exterior angle in degrees
  exterior_angle : ℝ
  -- The measure of each interior angle in degrees
  interior_angle : ℝ
  -- Condition: The polygon is regular and each exterior angle measures 18 degrees
  h_exterior : exterior_angle = 18
  -- Relationship between number of sides and exterior angle
  h_sides : sides = (360 : ℝ) / exterior_angle
  -- Relationship between interior and exterior angles
  h_interior : interior_angle = 180 - exterior_angle

/-- Theorem about the properties of the specific regular polygon -/
theorem regular_polygon_properties (p : RegularPolygon) : 
  p.sides = 20 ∧ p.interior_angle = 162 := by
  sorry


end NUMINAMATH_CALUDE_regular_polygon_properties_l2738_273822


namespace NUMINAMATH_CALUDE_sqrt_expression_equals_sqrt_three_l2738_273869

theorem sqrt_expression_equals_sqrt_three :
  Real.sqrt 48 - Real.sqrt 2 * Real.sqrt 6 - Real.sqrt 15 / Real.sqrt 5 = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_expression_equals_sqrt_three_l2738_273869


namespace NUMINAMATH_CALUDE_calories_left_for_dinner_l2738_273853

def daily_allowance : ℕ := 2200
def breakfast_calories : ℕ := 353
def lunch_calories : ℕ := 885
def snack_calories : ℕ := 130

theorem calories_left_for_dinner :
  daily_allowance - (breakfast_calories + lunch_calories + snack_calories) = 832 := by
  sorry

end NUMINAMATH_CALUDE_calories_left_for_dinner_l2738_273853


namespace NUMINAMATH_CALUDE_prob_not_all_even_l2738_273814

/-- The number of sides on a fair die -/
def sides : ℕ := 6

/-- The number of dice rolled -/
def num_dice : ℕ := 5

/-- The number of even outcomes on a single die -/
def even_outcomes : ℕ := 3

/-- The probability that not all dice show an even number when rolling five fair 6-sided dice -/
theorem prob_not_all_even : 
  1 - (even_outcomes : ℚ) ^ num_dice / sides ^ num_dice = 7533 / 7776 := by
sorry

end NUMINAMATH_CALUDE_prob_not_all_even_l2738_273814


namespace NUMINAMATH_CALUDE_merged_class_size_and_rank_l2738_273860

/-- Represents a group of students with known positions from left and right -/
structure StudentGroup where
  leftPos : Nat
  rightPos : Nat

/-- Calculates the total number of students in a group -/
def groupSize (g : StudentGroup) : Nat :=
  g.leftPos + g.rightPos - 1

theorem merged_class_size_and_rank (groupA groupB groupC : StudentGroup)
  (hA : groupA = ⟨8, 13⟩)
  (hB : groupB = ⟨12, 10⟩)
  (hC : groupC = ⟨7, 6⟩) :
  let totalStudents := groupSize groupA + groupSize groupB + groupSize groupC
  let rankFromLeft := groupSize groupA + groupB.leftPos
  totalStudents = 53 ∧ rankFromLeft = 32 := by
  sorry

end NUMINAMATH_CALUDE_merged_class_size_and_rank_l2738_273860


namespace NUMINAMATH_CALUDE_share_difference_l2738_273898

/-- Represents the share of money for each person -/
structure Share :=
  (amount : ℕ)

/-- Represents the distribution of money among three people -/
structure Distribution :=
  (faruk : Share)
  (vasim : Share)
  (ranjith : Share)

/-- Defines the ratio of distribution -/
def distribution_ratio : Distribution → (ℕ × ℕ × ℕ)
  | ⟨f, v, r⟩ => (3, 5, 8)

theorem share_difference (d : Distribution) :
  distribution_ratio d = (3, 5, 8) →
  d.vasim.amount = 1500 →
  d.ranjith.amount - d.faruk.amount = 1500 :=
by sorry

end NUMINAMATH_CALUDE_share_difference_l2738_273898


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l2738_273875

theorem quadratic_equation_solution : ∃ x₁ x₂ : ℝ, 
  (x₁ = 6 ∧ x₂ = -2) ∧ 
  (x₁^2 - 4*x₁ = 12) ∧ 
  (x₂^2 - 4*x₂ = 12) ∧
  (∀ x : ℝ, x^2 - 4*x = 12 → x = x₁ ∨ x = x₂) := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l2738_273875


namespace NUMINAMATH_CALUDE_spinner_probability_l2738_273819

def spinner1 : Finset ℕ := {2, 4, 5, 7, 9}
def spinner2 : Finset ℕ := {3, 4, 6, 8, 10, 12}

def isEven (n : ℕ) : Bool := n % 2 = 0

def productIsEven (x : ℕ) (y : ℕ) : Bool := isEven (x * y)

def favorableOutcomes : ℕ := (spinner1.card * spinner2.card) - 
  (spinner1.filter (λ x => ¬isEven x)).card * (spinner2.filter (λ x => ¬isEven x)).card

theorem spinner_probability : 
  (favorableOutcomes : ℚ) / (spinner1.card * spinner2.card) = 9 / 10 := by
  sorry

end NUMINAMATH_CALUDE_spinner_probability_l2738_273819


namespace NUMINAMATH_CALUDE_cartesian_angle_theorem_l2738_273878

/-- An angle in the Cartesian plane -/
structure CartesianAngle where
  -- The x-coordinate of the point on the terminal side
  x : ℝ
  -- The y-coordinate of the point on the terminal side
  y : ℝ
  -- The initial side is the non-negative half of the x-axis
  initial_side_positive_x : x > 0

/-- The theorem statement for the given problem -/
theorem cartesian_angle_theorem (α : CartesianAngle) 
  (h1 : α.x = 2) (h2 : α.y = 4) : 
  Real.tan (Real.arctan (α.y / α.x)) = 2 ∧ 
  (2 * Real.sin (Real.pi - Real.arctan (α.y / α.x)) + 
   2 * Real.cos (Real.arctan (α.y / α.x) / 2) ^ 2 - 1) / 
  (Real.sqrt 2 * Real.sin (Real.arctan (α.y / α.x) + Real.pi / 4)) = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_cartesian_angle_theorem_l2738_273878


namespace NUMINAMATH_CALUDE_parallel_lines_k_value_l2738_273865

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_lines_equal_slopes {m1 m2 b1 b2 : ℝ} : 
  (∀ x y : ℝ, y = m1 * x + b1 ↔ y = m2 * x + b2) ↔ m1 = m2

/-- The problem statement -/
theorem parallel_lines_k_value :
  (∀ x y : ℝ, y = 15 * x + 5 ↔ y = (5 * k) * x - 7) → k = 3 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_k_value_l2738_273865


namespace NUMINAMATH_CALUDE_debby_bottles_remaining_l2738_273829

/-- Calculates the number of water bottles remaining after a period of consumption. -/
def bottles_remaining (initial : ℕ) (daily_consumption : ℕ) (days : ℕ) : ℕ :=
  initial - daily_consumption * days

/-- Proves that Debby has 99 bottles left after her consumption period. -/
theorem debby_bottles_remaining :
  bottles_remaining 264 15 11 = 99 := by
  sorry

end NUMINAMATH_CALUDE_debby_bottles_remaining_l2738_273829


namespace NUMINAMATH_CALUDE_hamburger_combinations_l2738_273857

/-- Represents the number of available condiments -/
def num_condiments : Nat := 8

/-- Represents the number of choices for meat patties -/
def meat_patty_choices : Nat := 3

/-- Calculates the total number of hamburger combinations -/
def total_combinations : Nat := 2^num_condiments * meat_patty_choices

/-- Theorem: The total number of different hamburger combinations is 768 -/
theorem hamburger_combinations : total_combinations = 768 := by
  sorry

end NUMINAMATH_CALUDE_hamburger_combinations_l2738_273857


namespace NUMINAMATH_CALUDE_complex_quadratic_roots_l2738_273823

theorem complex_quadratic_roots (z : ℂ) :
  z ^ 2 = -91 + 104 * I ∧ (7 + 10 * I) ^ 2 = -91 + 104 * I →
  z = 7 + 10 * I ∨ z = -7 - 10 * I :=
by sorry

end NUMINAMATH_CALUDE_complex_quadratic_roots_l2738_273823


namespace NUMINAMATH_CALUDE_range_of_m_l2738_273808

theorem range_of_m (m : ℝ) : 
  (¬ ∃ x ∈ Set.Icc (1/2 : ℝ) 2, x^2 - 2*x - m ≤ 0) → m < -1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l2738_273808


namespace NUMINAMATH_CALUDE_factory_output_decrease_l2738_273850

theorem factory_output_decrease (initial_output : ℝ) : 
  let increased_output := initial_output * 1.1
  let holiday_output := increased_output * 1.4
  let required_decrease := (holiday_output - initial_output) / holiday_output * 100
  abs (required_decrease - 35.06) < 0.01 := by
sorry

end NUMINAMATH_CALUDE_factory_output_decrease_l2738_273850


namespace NUMINAMATH_CALUDE_m_intersect_n_equals_n_l2738_273873

-- Define the universal set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define set M as the domain of ln(1-x)
def M : Set ℝ := {x | x < 1}

-- Define set N
def N : Set ℝ := {x | -6 < x ∧ x < 1}

-- Theorem statement
theorem m_intersect_n_equals_n : M ∩ N = N := by
  sorry

end NUMINAMATH_CALUDE_m_intersect_n_equals_n_l2738_273873


namespace NUMINAMATH_CALUDE_perimeter_quarter_circle_square_l2738_273816

/-- The perimeter of a region bounded by quarter circular arcs constructed on each side of a square with side length 4/π is equal to 8. -/
theorem perimeter_quarter_circle_square : 
  let side_length : ℝ := 4 / Real.pi
  let quarter_circle_arc_length : ℝ := (1/4) * (2 * Real.pi * side_length)
  let num_arcs : ℕ := 4
  let perimeter : ℝ := num_arcs * quarter_circle_arc_length
  perimeter = 8 := by sorry

end NUMINAMATH_CALUDE_perimeter_quarter_circle_square_l2738_273816


namespace NUMINAMATH_CALUDE_quadratic_completion_square_l2738_273887

theorem quadratic_completion_square (x : ℝ) : 
  (∃ (d e : ℤ), (x + d : ℝ)^2 = e ∧ x^2 - 6*x - 15 = 0) → 
  (∃ (d e : ℤ), (x + d : ℝ)^2 = e ∧ x^2 - 6*x - 15 = 0 ∧ d + e = 21) := by
sorry

end NUMINAMATH_CALUDE_quadratic_completion_square_l2738_273887


namespace NUMINAMATH_CALUDE_circles_externally_tangent_l2738_273877

/-- Two circles with radii R and r, where R and r are the roots of x^2 - 3x + 2 = 0,
    and whose centers are at a distance d = 3 apart, are externally tangent. -/
theorem circles_externally_tangent (R r : ℝ) (d : ℝ) : 
  (R^2 - 3*R + 2 = 0) → 
  (r^2 - 3*r + 2 = 0) → 
  (d = 3) → 
  (R + r = d) := by sorry

end NUMINAMATH_CALUDE_circles_externally_tangent_l2738_273877


namespace NUMINAMATH_CALUDE_monotonicity_and_minimum_l2738_273886

/-- The function f(x) = kx^3 - 3x^2 + 1 -/
noncomputable def f (k : ℝ) (x : ℝ) : ℝ := k * x^3 - 3 * x^2 + 1

/-- The derivative of f(x) -/
noncomputable def f_deriv (k : ℝ) (x : ℝ) : ℝ := 3 * k * x^2 - 6 * x

theorem monotonicity_and_minimum (k : ℝ) (h : k ≥ 0) :
  (∀ x y, x ≤ 0 → y ∈ Set.Ioo 0 (2/k) → f k x ≤ f k y) ∧ 
  (∀ x y, x ∈ Set.Icc 0 (2/k) → y ≥ 2/k → f k x ≤ f k y) ∧
  (k > 2 ↔ f k (2/k) > 0) :=
sorry

end NUMINAMATH_CALUDE_monotonicity_and_minimum_l2738_273886


namespace NUMINAMATH_CALUDE_circle_center_l2738_273803

def is_circle (a : ℝ) : Prop :=
  ∃ (h : a^2 = a + 2 ∧ a^2 ≠ 0),
  ∀ (x y : ℝ), a^2*x^2 + (a+2)*y^2 + 4*x + 8*y + 5*a = 0 →
  ∃ (r : ℝ), (x + 2)^2 + (y + 4)^2 = r^2

theorem circle_center (a : ℝ) (h : is_circle a) :
  ∃ (x y : ℝ), a^2*x^2 + (a+2)*y^2 + 4*x + 8*y + 5*a = 0 ∧ x = -2 ∧ y = -4 :=
sorry

end NUMINAMATH_CALUDE_circle_center_l2738_273803


namespace NUMINAMATH_CALUDE_largest_angle_in_pentagon_l2738_273859

theorem largest_angle_in_pentagon (F G H I J : ℝ) : 
  F = 80 ∧ 
  G = 100 ∧ 
  H = I ∧ 
  J = 2 * H + 20 ∧ 
  F + G + H + I + J = 540 →
  max F (max G (max H (max I J))) = 190 :=
sorry

end NUMINAMATH_CALUDE_largest_angle_in_pentagon_l2738_273859


namespace NUMINAMATH_CALUDE_function_composition_l2738_273815

/-- Given a function f where f(3x) = 3 / (3 + x) for all x > 0, prove that 3f(x) = 27 / (9 + x) -/
theorem function_composition (f : ℝ → ℝ) (h : ∀ x > 0, f (3 * x) = 3 / (3 + x)) :
  ∀ x > 0, 3 * f x = 27 / (9 + x) := by
  sorry

end NUMINAMATH_CALUDE_function_composition_l2738_273815


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l2738_273800

def vector_a (x : ℝ) : ℝ × ℝ := (1, x)
def vector_b : ℝ × ℝ := (-3, 4)

def parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

theorem parallel_vectors_x_value :
  ∀ x : ℝ, parallel (vector_a x) vector_b → x = -4/3 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l2738_273800


namespace NUMINAMATH_CALUDE_unique_divisor_with_remainders_l2738_273837

theorem unique_divisor_with_remainders :
  ∃! N : ℕ,
    10 ≤ N ∧ N < 100 ∧
    5655 % N = 11 ∧
    5879 % N = 14 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_unique_divisor_with_remainders_l2738_273837


namespace NUMINAMATH_CALUDE_m_value_after_subtraction_l2738_273844

theorem m_value_after_subtraction (M : ℝ) : 
  (25 / 100 : ℝ) * M = (55 / 100 : ℝ) * 2500 → 
  M - (10 / 100 : ℝ) * M = 4950 := by
  sorry

end NUMINAMATH_CALUDE_m_value_after_subtraction_l2738_273844


namespace NUMINAMATH_CALUDE_sum_of_roots_equation_l2738_273867

theorem sum_of_roots_equation (x : ℝ) : 
  let f : ℝ → ℝ := λ x => x^3 - 3*x^2 - 10*x - 7*(x + 2)
  (∃ a b : ℝ, (∀ x, f x = (x - a) * (x - b) * (x + 2))) →
  a + b = 5 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_equation_l2738_273867


namespace NUMINAMATH_CALUDE_fraction_sum_theorem_l2738_273881

theorem fraction_sum_theorem (a b c : ℝ) (h : ((a - b) * (b - c) * (c - a)) / ((a + b) * (b + c) * (c + a)) = 2004 / 2005) :
  a / (a + b) + b / (b + c) + c / (c + a) = 4011 / 4010 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_theorem_l2738_273881


namespace NUMINAMATH_CALUDE_line_plane_intersection_l2738_273825

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the subset relation for lines and planes
variable (subset : Line → Plane → Prop)

-- Define the intersection relation for lines and planes
variable (intersect : Line → Line → Prop)
variable (intersect_planes : Plane → Plane → Prop)

-- Theorem statement
theorem line_plane_intersection 
  (a b : Line) (α β : Plane) 
  (h1 : a ≠ b) 
  (h2 : α ≠ β) 
  (h3 : subset a α) 
  (h4 : subset b β) 
  (h5 : intersect a b) : 
  intersect_planes α β :=
sorry

end NUMINAMATH_CALUDE_line_plane_intersection_l2738_273825


namespace NUMINAMATH_CALUDE_complex_fraction_power_l2738_273807

theorem complex_fraction_power (i : ℂ) (a b : ℝ) :
  i * i = -1 →
  (1 : ℂ) / (1 + i) = a + b * i →
  a ^ b = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_power_l2738_273807


namespace NUMINAMATH_CALUDE_probability_penny_dime_same_different_dollar_l2738_273856

-- Define the coin types
inductive Coin
| Penny
| Nickel
| Dime
| Quarter
| Dollar

-- Define the possible outcomes for a coin flip
inductive FlipResult
| Heads
| Tails

-- Define a function to represent the result of flipping all coins
def CoinFlips := Coin → FlipResult

-- Define the condition for a successful outcome
def SuccessfulOutcome (flips : CoinFlips) : Prop :=
  (flips Coin.Penny = flips Coin.Dime) ∧ (flips Coin.Penny ≠ flips Coin.Dollar)

-- Define the total number of possible outcomes
def TotalOutcomes : ℕ := 2^5

-- Define the number of successful outcomes
def SuccessfulOutcomes : ℕ := 8

-- Theorem statement
theorem probability_penny_dime_same_different_dollar :
  (SuccessfulOutcomes : ℚ) / TotalOutcomes = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_probability_penny_dime_same_different_dollar_l2738_273856


namespace NUMINAMATH_CALUDE_specific_coin_flip_probability_l2738_273812

/-- The probability of getting a specific sequence of heads and tails
    when flipping a fair coin multiple times. -/
def coin_flip_probability (n : ℕ) (k : ℕ) : ℚ :=
  (1 / 2) ^ n

theorem specific_coin_flip_probability :
  coin_flip_probability 5 2 = 1 / 32 := by
  sorry

end NUMINAMATH_CALUDE_specific_coin_flip_probability_l2738_273812


namespace NUMINAMATH_CALUDE_no_such_function_l2738_273838

theorem no_such_function : ¬∃ f : ℤ → ℤ, ∀ m n : ℤ, f (m + f n) = f m - n := by
  sorry

end NUMINAMATH_CALUDE_no_such_function_l2738_273838


namespace NUMINAMATH_CALUDE_composite_3p_squared_plus_15_l2738_273855

theorem composite_3p_squared_plus_15 (p : ℕ) (h : Nat.Prime p) :
  ¬ Nat.Prime (3 * p^2 + 15) := by
  sorry

end NUMINAMATH_CALUDE_composite_3p_squared_plus_15_l2738_273855


namespace NUMINAMATH_CALUDE_choose_four_from_ten_l2738_273896

theorem choose_four_from_ten : Nat.choose 10 4 = 210 := by
  sorry

end NUMINAMATH_CALUDE_choose_four_from_ten_l2738_273896


namespace NUMINAMATH_CALUDE_function_properties_l2738_273836

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.sin x - Real.sqrt 3 * Real.cos x

theorem function_properties (a : ℝ) (h : f a (π / 3) = 0) :
  (∃ T > 0, ∀ x, f a (x + T) = f a x ∧ ∀ S, 0 < S → S < T → ∃ y, f a (y + S) ≠ f a y) ∧
  (∀ y ∈ Set.Icc (π / 2) (3 * π / 2), -1 ≤ f a y ∧ f a y ≤ 2) ∧
  (∃ y₁ ∈ Set.Icc (π / 2) (3 * π / 2), f a y₁ = -1) ∧
  (∃ y₂ ∈ Set.Icc (π / 2) (3 * π / 2), f a y₂ = 2) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l2738_273836


namespace NUMINAMATH_CALUDE_ocean_depth_for_specific_mountain_l2738_273841

/-- Represents a cone-shaped mountain partially submerged in water -/
structure SubmergedMountain where
  totalHeight : ℝ
  aboveWaterVolumeFraction : ℝ

/-- Calculates the depth of the ocean at the base of a submerged mountain -/
def oceanDepth (mountain : SubmergedMountain) : ℝ :=
  mountain.totalHeight * (1 - (1 - mountain.aboveWaterVolumeFraction) ^ (1/3))

/-- Theorem stating that for a specific mountain configuration, the ocean depth is 648 feet -/
theorem ocean_depth_for_specific_mountain :
  let mountain : SubmergedMountain := {
    totalHeight := 12000,
    aboveWaterVolumeFraction := 1/6
  }
  oceanDepth mountain = 648 := by sorry

end NUMINAMATH_CALUDE_ocean_depth_for_specific_mountain_l2738_273841


namespace NUMINAMATH_CALUDE_proposition_p_or_q_exclusive_l2738_273821

theorem proposition_p_or_q_exclusive (a : ℝ) : 
  (∃ x y : ℝ, x > 0 ∧ y < 0 ∧ x^2 - x + a^2 - 6*a = 0 ∧ y^2 - y + a^2 - 6*a = 0) ∨
  (∃ x : ℝ, x^2 + (a - 3)*x + 1 = 0) ∧
  ¬((∃ x y : ℝ, x > 0 ∧ y < 0 ∧ x^2 - x + a^2 - 6*a = 0 ∧ y^2 - y + a^2 - 6*a = 0) ∧
    (∃ x : ℝ, x^2 + (a - 3)*x + 1 = 0)) ↔
  a ≥ 6 ∨ a ≤ 0 ∨ (1 < a ∧ a < 5) :=
by sorry

end NUMINAMATH_CALUDE_proposition_p_or_q_exclusive_l2738_273821


namespace NUMINAMATH_CALUDE_same_speed_problem_l2738_273893

theorem same_speed_problem (x : ℝ) :
  let jack_speed := x^2 - 11*x - 22
  let jill_distance := x^2 - 5*x - 36
  let jill_time := x + 4
  jack_speed > 0 ∧ 
  jill_distance > 0 ∧ 
  jill_time > 0 ∧
  jack_speed = jill_distance / jill_time →
  jack_speed = 4 := by
sorry

end NUMINAMATH_CALUDE_same_speed_problem_l2738_273893


namespace NUMINAMATH_CALUDE_book_reading_fraction_l2738_273832

theorem book_reading_fraction (total_pages remaining_pages : ℕ) 
  (h1 : total_pages = 468)
  (h2 : remaining_pages = 96)
  (h3 : (7 : ℚ) / 13 * total_pages + remaining_pages < total_pages) :
  let pages_read_first_week := (7 : ℚ) / 13 * total_pages
  let pages_remaining_after_first_week := total_pages - pages_read_first_week
  let pages_read_second_week := pages_remaining_after_first_week - remaining_pages
  pages_read_second_week / pages_remaining_after_first_week = 5 / 9 := by
sorry

end NUMINAMATH_CALUDE_book_reading_fraction_l2738_273832


namespace NUMINAMATH_CALUDE_min_sphere_surface_area_l2738_273811

/-- Represents a cuboid with vertices on a sphere -/
structure CuboidOnSphere where
  -- The length of edge AB
  ab : ℝ
  -- The length of edge AD
  ad : ℝ
  -- The length of edge AA'
  aa' : ℝ
  -- The radius of the sphere
  r : ℝ
  -- All vertices are on the sphere
  vertices_on_sphere : ab ^ 2 + ad ^ 2 + aa' ^ 2 = (2 * r) ^ 2
  -- AB = 2
  ab_equals_two : ab = 2
  -- Volume of pyramid O-A'B'C'D' is 2
  pyramid_volume : (1 / 3) * ad * aa' = 2

/-- The minimum surface area of the sphere is 16π -/
theorem min_sphere_surface_area (c : CuboidOnSphere) : 
  ∃ (min_area : ℝ), min_area = 16 * π ∧ 
  ∀ (area : ℝ), area = 4 * π * c.r ^ 2 → area ≥ min_area := by
  sorry

end NUMINAMATH_CALUDE_min_sphere_surface_area_l2738_273811


namespace NUMINAMATH_CALUDE_chess_tournament_games_l2738_273888

/-- The number of games in a chess tournament --/
def tournament_games (n : ℕ) (k : ℕ) : ℕ :=
  (n.choose 2) * k

/-- Theorem: In a chess tournament with 50 players, where each player plays
    four times with each opponent, the total number of games is 4900 --/
theorem chess_tournament_games :
  tournament_games 50 4 = 4900 := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_games_l2738_273888


namespace NUMINAMATH_CALUDE_bus_speed_excluding_stoppages_l2738_273818

/-- Given a bus that travels at 45 km/hr including stoppages and stops for 10 minutes per hour,
    prove that its speed excluding stoppages is 54 km/hr. -/
theorem bus_speed_excluding_stoppages :
  let speed_with_stoppages : ℝ := 45
  let stop_time_per_hour : ℝ := 10 / 60
  let travel_time_per_hour : ℝ := 1 - stop_time_per_hour
  let speed_without_stoppages : ℝ := speed_with_stoppages / travel_time_per_hour
  speed_without_stoppages = 54 := by
  sorry

end NUMINAMATH_CALUDE_bus_speed_excluding_stoppages_l2738_273818


namespace NUMINAMATH_CALUDE_incorrect_multiplication_result_l2738_273884

theorem incorrect_multiplication_result 
  (x : ℝ) 
  (h1 : ∃ a b : ℕ, 987 * x = 500000 + 10000 * a + 700 + b / 100 + 0.0989999999)
  (h2 : 987 * x ≠ 555707.2899999999)
  (h3 : 555707.2899999999 = 987 * x) : 
  987 * x = 598707.2989999999 := by
sorry

end NUMINAMATH_CALUDE_incorrect_multiplication_result_l2738_273884


namespace NUMINAMATH_CALUDE_mistaken_division_correction_l2738_273820

theorem mistaken_division_correction (n : ℕ) : 
  (n / 7 = 12 ∧ n % 7 = 4) → n / 3 = 29 := by
  sorry

end NUMINAMATH_CALUDE_mistaken_division_correction_l2738_273820


namespace NUMINAMATH_CALUDE_profit_decrease_for_one_loom_l2738_273848

/-- Represents the profit decrease when one loom breaks down for a month -/
def profit_decrease (num_looms : ℕ) (total_sales : ℕ) (manufacturing_expenses : ℕ) (establishment_charges : ℕ) : ℕ :=
  let sales_per_loom := total_sales / num_looms
  let manufacturing_per_loom := manufacturing_expenses / num_looms
  let establishment_per_loom := establishment_charges / num_looms
  sales_per_loom - manufacturing_per_loom - establishment_per_loom

/-- Theorem stating the profit decrease when one loom breaks down for a month -/
theorem profit_decrease_for_one_loom :
  profit_decrease 125 500000 150000 75000 = 2200 := by
  sorry

#eval profit_decrease 125 500000 150000 75000

end NUMINAMATH_CALUDE_profit_decrease_for_one_loom_l2738_273848


namespace NUMINAMATH_CALUDE_sqrt_equality_problem_l2738_273879

theorem sqrt_equality_problem : ∃ (a x : ℝ), 
  x > 0 ∧ 
  Real.sqrt x = 2 * a - 3 ∧ 
  Real.sqrt x = 5 - a ∧ 
  a = -2 ∧ 
  x = 49 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equality_problem_l2738_273879


namespace NUMINAMATH_CALUDE_tangent_three_implications_l2738_273870

theorem tangent_three_implications (α : Real) (h : Real.tan α = 3) :
  (4 * Real.sin α - 2 * Real.cos α) / (5 * Real.cos α + 3 * Real.sin α) = 5/7 ∧
  1 - 4 * Real.sin α * Real.cos α + 2 * (Real.cos α)^2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_tangent_three_implications_l2738_273870


namespace NUMINAMATH_CALUDE_f_properties_l2738_273831

def f (x : ℝ) := x^3 - 3*x

theorem f_properties :
  (∀ x, f (-x) = -f x) ∧
  (f 1 = -2) ∧
  (∀ x, x = -1 ∨ x = 1 → deriv f x = 0) ∧
  (∀ x, f x ≤ f (-1)) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l2738_273831


namespace NUMINAMATH_CALUDE_zoo_ticket_cost_l2738_273882

theorem zoo_ticket_cost (adult_price : ℝ) : 
  (adult_price > 0) →
  (6 * adult_price + 5 * (adult_price / 2) + 3 * (adult_price - 1.5) = 40.5) →
  (10 * adult_price + 8 * (adult_price / 2) + 4 * (adult_price - 1.5) = 64.38) :=
by
  sorry

end NUMINAMATH_CALUDE_zoo_ticket_cost_l2738_273882


namespace NUMINAMATH_CALUDE_orange_ribbons_l2738_273851

theorem orange_ribbons (total : ℕ) (yellow purple orange silver : ℕ) : 
  yellow + purple + orange + silver = total →
  4 * yellow = total →
  3 * purple = total →
  6 * orange = total →
  silver = 40 →
  orange = 27 := by
sorry

end NUMINAMATH_CALUDE_orange_ribbons_l2738_273851


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l2738_273858

theorem right_triangle_hypotenuse (m1 m2 : ℝ) (h1 : m1 = Real.sqrt 52) (h2 : m2 = Real.sqrt 73) :
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ 
  a^2 + b^2 = c^2 ∧
  m1^2 = (2 * b^2 + 2 * c^2 - a^2) / 4 ∧
  m2^2 = (2 * a^2 + 2 * c^2 - b^2) / 4 ∧
  c = 10 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l2738_273858


namespace NUMINAMATH_CALUDE_quadratic_function_problem_l2738_273871

theorem quadratic_function_problem (a b : ℝ) : 
  (1^2 + a*1 + b = 2) → 
  ((-2)^2 + a*(-2) + b = -1) → 
  ((-3)^2 + a*(-3) + b = 2) :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_problem_l2738_273871


namespace NUMINAMATH_CALUDE_min_value_theorem_l2738_273834

theorem min_value_theorem (a b : ℝ) (h : 2 * a - 3 * b + 6 = 0) :
  ∃ (min_val : ℝ), min_val = (1 / 4 : ℝ) ∧ ∀ (x : ℝ), 4^a + (1 / 8^b) ≥ x → x ≥ min_val :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2738_273834


namespace NUMINAMATH_CALUDE_right_triangle_median_to_hypotenuse_l2738_273833

theorem right_triangle_median_to_hypotenuse (DE DF EF : ℝ) :
  DE = 15 →
  DF = 9 →
  EF = 12 →
  DE^2 = DF^2 + EF^2 →
  (DE / 2 : ℝ) = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_median_to_hypotenuse_l2738_273833


namespace NUMINAMATH_CALUDE_fraction_of_girls_at_event_l2738_273874

theorem fraction_of_girls_at_event (maplewood_total : ℕ) (brookside_total : ℕ)
  (maplewood_boy_ratio maplewood_girl_ratio : ℕ)
  (brookside_boy_ratio brookside_girl_ratio : ℕ)
  (h1 : maplewood_total = 300)
  (h2 : brookside_total = 240)
  (h3 : maplewood_boy_ratio = 3)
  (h4 : maplewood_girl_ratio = 2)
  (h5 : brookside_boy_ratio = 2)
  (h6 : brookside_girl_ratio = 3) :
  (maplewood_total * maplewood_girl_ratio / (maplewood_boy_ratio + maplewood_girl_ratio) +
   brookside_total * brookside_girl_ratio / (brookside_boy_ratio + brookside_girl_ratio)) /
  (maplewood_total + brookside_total) = 22 / 45 := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_girls_at_event_l2738_273874


namespace NUMINAMATH_CALUDE_cos_25_minus_alpha_equals_one_third_l2738_273864

theorem cos_25_minus_alpha_equals_one_third 
  (h : Real.sin (65 * π / 180 + α) = 1 / 3) : 
  Real.cos (25 * π / 180 - α) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_cos_25_minus_alpha_equals_one_third_l2738_273864


namespace NUMINAMATH_CALUDE_chastity_initial_money_l2738_273862

def lollipop_cost : ℚ := 1.5
def gummies_cost : ℚ := 2
def lollipops_bought : ℕ := 4
def gummies_packs_bought : ℕ := 2
def money_left : ℚ := 5

def initial_money : ℚ := 15

theorem chastity_initial_money :
  initial_money = 
    (lollipop_cost * lollipops_bought + gummies_cost * gummies_packs_bought + money_left) :=
by sorry

end NUMINAMATH_CALUDE_chastity_initial_money_l2738_273862


namespace NUMINAMATH_CALUDE_investment_time_q_is_thirteen_l2738_273813

/-- Represents the investment and profit data for two partners -/
structure PartnershipData where
  investment_ratio_p : ℚ
  investment_ratio_q : ℚ
  profit_ratio_p : ℚ
  profit_ratio_q : ℚ
  investment_time_p : ℚ

/-- Calculates the investment time for partner Q given the partnership data -/
def calculate_investment_time_q (data : PartnershipData) : ℚ :=
  (data.profit_ratio_q * data.investment_ratio_p * data.investment_time_p) / 
  (data.profit_ratio_p * data.investment_ratio_q)

/-- Theorem stating that given the specified partnership data, Q's investment time is 13 months -/
theorem investment_time_q_is_thirteen : 
  let data : PartnershipData := {
    investment_ratio_p := 7,
    investment_ratio_q := 5,
    profit_ratio_p := 7,
    profit_ratio_q := 13,
    investment_time_p := 5
  }
  calculate_investment_time_q data = 13 := by sorry

end NUMINAMATH_CALUDE_investment_time_q_is_thirteen_l2738_273813


namespace NUMINAMATH_CALUDE_complex_magnitude_l2738_273876

theorem complex_magnitude (z : ℂ) : (z + Complex.I) * (2 - Complex.I) = 11 + 7 * Complex.I → Complex.abs z = 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l2738_273876


namespace NUMINAMATH_CALUDE_line_perpendicular_to_plane_and_line_in_plane_l2738_273889

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relationships between lines and planes
variable (perpendicular : Line → Plane → Prop)
variable (contains : Plane → Line → Prop)
variable (perpendicularLines : Line → Line → Prop)

-- Theorem statement
theorem line_perpendicular_to_plane_and_line_in_plane
  (m n : Line) (α : Plane)
  (h1 : perpendicular m α)
  (h2 : contains α n) :
  perpendicularLines m n :=
sorry

end NUMINAMATH_CALUDE_line_perpendicular_to_plane_and_line_in_plane_l2738_273889


namespace NUMINAMATH_CALUDE_expressions_equality_l2738_273885

theorem expressions_equality :
  -- Expression 1
  (1 + Real.sqrt 3) * (2 - Real.sqrt 3) = -1 + Real.sqrt 3 ∧
  -- Expression 2
  2 * (Real.sqrt (9/2) - Real.sqrt 8 / 3) * (2 * Real.sqrt 2) = 10/3 ∧
  -- Expression 3
  Real.sqrt 18 - Real.sqrt 8 + Real.sqrt (1/8) = 5 * Real.sqrt 2 / 4 ∧
  -- Expression 4
  (Real.sqrt 6 - 2 * Real.sqrt 15) * Real.sqrt 3 - 6 * Real.sqrt (1/2) = -6 * Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_expressions_equality_l2738_273885


namespace NUMINAMATH_CALUDE_words_per_page_l2738_273826

theorem words_per_page (total_pages : Nat) (max_words_per_page : Nat) (total_words_mod : Nat) :
  total_pages = 154 →
  max_words_per_page = 120 →
  total_words_mod = 221 →
  ∃ words_per_page : Nat,
    words_per_page ≤ max_words_per_page ∧
    (total_pages * words_per_page) % 217 = total_words_mod ∧
    words_per_page = 106 := by
  sorry

end NUMINAMATH_CALUDE_words_per_page_l2738_273826


namespace NUMINAMATH_CALUDE_monotone_increasing_ln_plus_ax_l2738_273846

open Real

theorem monotone_increasing_ln_plus_ax (a : ℝ) :
  (∀ x ∈ Set.Ioo 1 2, Monotone (λ x => Real.log x + a * x)) →
  a ≥ -1/2 := by
sorry

end NUMINAMATH_CALUDE_monotone_increasing_ln_plus_ax_l2738_273846


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2738_273817

theorem complex_equation_solution (z : ℂ) (h : z * Complex.I = 1 - Complex.I) : 
  z = -1 - Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2738_273817


namespace NUMINAMATH_CALUDE_min_sum_odd_days_l2738_273806

/-- A sequence of 5 non-negative integers representing fish caught each day --/
def FishSequence := (ℕ × ℕ × ℕ × ℕ × ℕ)

/-- Check if a sequence is non-increasing --/
def is_non_increasing (seq : FishSequence) : Prop :=
  let (a, b, c, d, e) := seq
  a ≥ b ∧ b ≥ c ∧ c ≥ d ∧ d ≥ e

/-- Calculate the sum of all elements in the sequence --/
def sum_sequence (seq : FishSequence) : ℕ :=
  let (a, b, c, d, e) := seq
  a + b + c + d + e

/-- Calculate the sum of 1st, 3rd, and 5th elements --/
def sum_odd_days (seq : FishSequence) : ℕ :=
  let (a, _, c, _, e) := seq
  a + c + e

/-- The main theorem --/
theorem min_sum_odd_days (seq : FishSequence) :
  is_non_increasing seq →
  sum_sequence seq = 100 →
  sum_odd_days seq ≥ 50 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_odd_days_l2738_273806


namespace NUMINAMATH_CALUDE_intersection_complement_theorem_l2738_273804

def A : Set ℝ := {-2, -1, 0, 1, 2}
def B : Set ℝ := {x : ℝ | x^2 + 2*x < 0}

theorem intersection_complement_theorem :
  A ∩ (Set.univ \ B) = {-2, 0, 1, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_theorem_l2738_273804


namespace NUMINAMATH_CALUDE_complex_number_location_l2738_273828

theorem complex_number_location :
  let z : ℂ := ((-1 : ℂ) + Complex.I) / ((1 : ℂ) + Complex.I) - 1
  z = -1 + Complex.I ∧ z.re < 0 ∧ z.im > 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_location_l2738_273828


namespace NUMINAMATH_CALUDE_rhombus_all_sides_equal_rectangle_not_necessarily_l2738_273802

/-- A rhombus is a quadrilateral with four equal sides. -/
structure Rhombus where
  sides : Fin 4 → ℝ
  all_sides_equal : ∀ (i j : Fin 4), sides i = sides j

/-- A rectangle is a quadrilateral with four right angles and opposite sides equal. -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Theorem stating that all sides of a rhombus are equal, but not necessarily for a rectangle -/
theorem rhombus_all_sides_equal_rectangle_not_necessarily (r : Rhombus) (rect : Rectangle) :
  (∀ (i j : Fin 4), r.sides i = r.sides j) ∧
  ¬(∀ (rect : Rectangle), rect.width = rect.height) :=
sorry

end NUMINAMATH_CALUDE_rhombus_all_sides_equal_rectangle_not_necessarily_l2738_273802


namespace NUMINAMATH_CALUDE_min_value_of_sum_l2738_273842

theorem min_value_of_sum (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (sum_eq : a + b + c = 6) : 
  (∀ x y z : ℝ, x > 0 → y > 0 → z > 0 → x + y + z = 6 → 
    9/a + 4/b + 1/c ≤ 9/x + 4/y + 1/z) ∧ 
  (9/a + 4/b + 1/c = 6) := by
sorry

end NUMINAMATH_CALUDE_min_value_of_sum_l2738_273842


namespace NUMINAMATH_CALUDE_factorial_1000_trailing_zeros_l2738_273839

/-- The number of trailing zeros in n! -/
def trailingZeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125) + (n / 625)

/-- Theorem: 1000! ends with 249 zeros -/
theorem factorial_1000_trailing_zeros :
  trailingZeros 1000 = 249 := by
  sorry

end NUMINAMATH_CALUDE_factorial_1000_trailing_zeros_l2738_273839


namespace NUMINAMATH_CALUDE_tan_half_product_l2738_273894

theorem tan_half_product (a b : Real) :
  3 * (Real.cos a + Real.cos b) + 6 * (Real.cos a * Real.cos b + 1) = 0 →
  (Real.tan (a / 2) * Real.tan (b / 2) = Real.sqrt 2) ∨
  (Real.tan (a / 2) * Real.tan (b / 2) = -Real.sqrt 2) := by
sorry

end NUMINAMATH_CALUDE_tan_half_product_l2738_273894


namespace NUMINAMATH_CALUDE_sum_of_odd_divisors_90_l2738_273891

/-- The sum of the positive odd divisors of 90 -/
def sumOfOddDivisors90 : ℕ := sorry

/-- Theorem stating that the sum of the positive odd divisors of 90 is 78 -/
theorem sum_of_odd_divisors_90 : sumOfOddDivisors90 = 78 := by sorry

end NUMINAMATH_CALUDE_sum_of_odd_divisors_90_l2738_273891


namespace NUMINAMATH_CALUDE_peach_difference_l2738_273801

/-- Given information about peaches owned by Jake, Steven, and Jill -/
theorem peach_difference (jill steven jake : ℕ) 
  (h1 : jake = steven - 5)  -- Jake has 5 fewer peaches than Steven
  (h2 : steven = jill + 18) -- Steven has 18 more peaches than Jill
  (h3 : jill = 87)          -- Jill has 87 peaches
  : jake - jill = 13 :=     -- Prove that Jake has 13 more peaches than Jill
by sorry

end NUMINAMATH_CALUDE_peach_difference_l2738_273801


namespace NUMINAMATH_CALUDE_smallest_sum_of_sequence_l2738_273890

theorem smallest_sum_of_sequence (A B C D : ℕ) : 
  A > 0 → B > 0 → C > 0 → 
  (C - B = B - A) →  -- arithmetic sequence condition
  (C * C = B * D) →  -- geometric sequence condition
  (C : ℚ) / B = 7 / 4 →
  A + B + C + D ≥ 97 :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_of_sequence_l2738_273890


namespace NUMINAMATH_CALUDE_last_digit_77_base_4_l2738_273892

def last_digit_base_4 (n : ℕ) : ℕ :=
  n % 4

theorem last_digit_77_base_4 :
  last_digit_base_4 77 = 1 := by
  sorry

end NUMINAMATH_CALUDE_last_digit_77_base_4_l2738_273892


namespace NUMINAMATH_CALUDE_min_value_of_expression_equality_condition_l2738_273866

theorem min_value_of_expression (x : ℝ) : 
  Real.sqrt (x^2 + (2 - x)^2) + Real.sqrt ((2 - x)^2 + (2 + x)^2) ≥ 2 * Real.sqrt 5 :=
by sorry

theorem equality_condition :
  Real.sqrt ((4/3)^2 + (2 - 4/3)^2) + Real.sqrt ((2 - 4/3)^2 + (2 + 4/3)^2) = 2 * Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_equality_condition_l2738_273866


namespace NUMINAMATH_CALUDE_calculation_proof_l2738_273835

theorem calculation_proof :
  (- (2^3 / 8) - (1/4 * (-2)^2) = -2) ∧
  ((-1/12 - 1/16 + 3/4 - 1/6) * (-48) = -21) := by
sorry

end NUMINAMATH_CALUDE_calculation_proof_l2738_273835


namespace NUMINAMATH_CALUDE_max_value_sqrt_sum_l2738_273899

theorem max_value_sqrt_sum (x : ℝ) (h1 : 0 ≤ x) (h2 : x ≤ 9) :
  Real.sqrt (x + 15) + Real.sqrt (9 - x) + Real.sqrt (2 * x) ≤ Real.sqrt 143 := by
  sorry

end NUMINAMATH_CALUDE_max_value_sqrt_sum_l2738_273899


namespace NUMINAMATH_CALUDE_min_trig_expression_min_trig_expression_equality_l2738_273843

theorem min_trig_expression (θ : Real) (h : 0 < θ ∧ θ < Real.pi / 2) :
  3 * Real.cos θ + 2 / Real.sin θ + 2 * Real.sqrt 2 * Real.tan θ ≥ 7 * Real.sqrt 2 / 2 :=
by sorry

theorem min_trig_expression_equality :
  3 * Real.cos (Real.pi / 4) + 2 / Real.sin (Real.pi / 4) + 2 * Real.sqrt 2 * Real.tan (Real.pi / 4) = 7 * Real.sqrt 2 / 2 :=
by sorry

end NUMINAMATH_CALUDE_min_trig_expression_min_trig_expression_equality_l2738_273843


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l2738_273863

theorem quadratic_equation_solution (c d : ℝ) (hc : c ≠ 0) (hd : d ≠ 0) 
  (h1 : c^2 + c*c + 2*d = 0) (h2 : d^2 + c*d + 2*d = 0) : 
  c = 2 ∧ d = -4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l2738_273863


namespace NUMINAMATH_CALUDE_pebble_color_difference_l2738_273830

theorem pebble_color_difference (total_pebbles red_pebbles blue_pebbles : ℕ) 
  (h1 : total_pebbles = 40)
  (h2 : red_pebbles = 9)
  (h3 : blue_pebbles = 13)
  (h4 : (total_pebbles - red_pebbles - blue_pebbles) % 3 = 0) :
  blue_pebbles - (total_pebbles - red_pebbles - blue_pebbles) / 3 = 7 := by
  sorry

end NUMINAMATH_CALUDE_pebble_color_difference_l2738_273830


namespace NUMINAMATH_CALUDE_a_range_l2738_273845

-- Define propositions p and q as functions of a
def p (a : ℝ) : Prop := 0 < a ∧ a < 6

def q (a : ℝ) : Prop := a ≥ 5 ∨ a ≤ 1

-- Define the range of a
def range_a (a : ℝ) : Prop := a ≥ 6 ∨ a ≤ 0 ∨ (1 < a ∧ a < 5)

theorem a_range :
  (∀ a : ℝ, (p a ∨ q a)) ∧ (∀ a : ℝ, ¬(p a ∧ q a)) →
  ∀ a : ℝ, range_a a ↔ (p a ∨ q a) :=
by sorry

end NUMINAMATH_CALUDE_a_range_l2738_273845


namespace NUMINAMATH_CALUDE_marble_count_l2738_273868

theorem marble_count : ∀ (r b : ℕ),
  (r - 2) * 10 = r + b - 2 →
  r * 6 = r + b - 3 →
  (r - 2) * 8 = r + b - 4 →
  r + b = 42 := by
sorry

end NUMINAMATH_CALUDE_marble_count_l2738_273868


namespace NUMINAMATH_CALUDE_largest_common_divisor_fifteen_always_divides_l2738_273827

def is_odd (n : ℕ) : Prop := ∃ k, n = 2*k + 1

def product (n : ℕ) : ℕ := n * (n+2) * (n+4) * (n+6) * (n+8)

theorem largest_common_divisor :
  ∀ (d : ℕ), d > 15 →
    ∃ (n : ℕ), is_odd n ∧ ¬(d ∣ product n) :=
sorry

theorem fifteen_always_divides :
  ∀ (n : ℕ), is_odd n → (15 ∣ product n) :=
sorry

end NUMINAMATH_CALUDE_largest_common_divisor_fifteen_always_divides_l2738_273827


namespace NUMINAMATH_CALUDE_log_inequality_solution_l2738_273880

-- Define the logarithm function with base 1/2
noncomputable def log_half (x : ℝ) : ℝ := Real.log x / Real.log (1/2)

-- Define the solution set
def solution_set : Set ℝ := {x | log_half (2*x - 1) < log_half (-x + 5)}

-- Theorem statement
theorem log_inequality_solution :
  solution_set = Set.Ioo 2 5 :=
by sorry

end NUMINAMATH_CALUDE_log_inequality_solution_l2738_273880


namespace NUMINAMATH_CALUDE_plastic_bottles_count_l2738_273861

/-- The weight of a glass bottle in grams -/
def glass_bottle_weight : ℕ := 200

/-- The weight of a plastic bottle in grams -/
def plastic_bottle_weight : ℕ := 50

/-- The total weight of the second scenario in grams -/
def total_weight : ℕ := 1050

/-- The number of glass bottles in the second scenario -/
def num_glass_bottles : ℕ := 4

theorem plastic_bottles_count :
  ∃ (x : ℕ), 
    3 * glass_bottle_weight = 600 ∧
    glass_bottle_weight = plastic_bottle_weight + 150 ∧
    4 * glass_bottle_weight + x * plastic_bottle_weight = total_weight ∧
    x = 5 := by
  sorry

end NUMINAMATH_CALUDE_plastic_bottles_count_l2738_273861


namespace NUMINAMATH_CALUDE_ellipse_line_intersection_l2738_273809

-- Define the ellipse
def E (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define the lines
def l (k x y : ℝ) : Prop := y = k * x + 1
def l₁ (k₁ x y : ℝ) : Prop := y = k₁ * x + 1

-- Define the symmetry line
def sym_line (x y : ℝ) : Prop := y = x + 1

-- Define the theorem
theorem ellipse_line_intersection
  (k k₁ : ℝ) 
  (hk : k > 0) 
  (hk_neq : k ≠ 1) 
  (h_sym : ∀ x y, l k x y ↔ l₁ k₁ (y - 1) (x - 1)) :
  ∃ P : ℝ × ℝ, 
    k * k₁ = 1 ∧ 
    ∀ M N : ℝ × ℝ, 
      (E M.1 M.2 ∧ l k M.1 M.2) → 
      (E N.1 N.2 ∧ l₁ k₁ N.1 N.2) → 
      ∃ t : ℝ, P = (1 - t) • M + t • N :=
by sorry

end NUMINAMATH_CALUDE_ellipse_line_intersection_l2738_273809


namespace NUMINAMATH_CALUDE_intersection_of_A_and_complement_of_B_l2738_273854

def U : Set ℝ := Set.univ
def A : Set ℝ := {x | x > 1}
def B : Set ℝ := {x | x > 2}

theorem intersection_of_A_and_complement_of_B :
  A ∩ (U \ B) = {x : ℝ | 1 < x ∧ x ≤ 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_complement_of_B_l2738_273854


namespace NUMINAMATH_CALUDE_line_inclination_angle_l2738_273847

theorem line_inclination_angle (a : ℝ) (h : a < 0) :
  let line := {(x, y) : ℝ × ℝ | x - a * y + 2 = 0}
  let slope := (1 : ℝ) / a
  let inclination_angle := Real.pi + Real.arctan slope
  ∀ (x y : ℝ), (x, y) ∈ line → inclination_angle ∈ Set.Icc 0 Real.pi ∧
    Real.tan inclination_angle = slope :=
by sorry

end NUMINAMATH_CALUDE_line_inclination_angle_l2738_273847


namespace NUMINAMATH_CALUDE_palm_meadows_beds_l2738_273849

theorem palm_meadows_beds (total_rooms : ℕ) (rooms_with_fewer_beds : ℕ) (beds_in_other_rooms : ℕ) (total_beds : ℕ) :
  total_rooms = 13 →
  rooms_with_fewer_beds = 8 →
  total_rooms - rooms_with_fewer_beds = 5 →
  beds_in_other_rooms = 3 →
  total_beds = 31 →
  (rooms_with_fewer_beds * 2) + ((total_rooms - rooms_with_fewer_beds) * beds_in_other_rooms) = total_beds :=
by
  sorry

end NUMINAMATH_CALUDE_palm_meadows_beds_l2738_273849


namespace NUMINAMATH_CALUDE_distance_P_to_y_axis_l2738_273852

/-- The distance from a point to the y-axis in a Cartesian coordinate system --/
def distance_to_y_axis (x y : ℝ) : ℝ := |x|

/-- Point P in the Cartesian coordinate system --/
def P : ℝ × ℝ := (-3, 4)

/-- Theorem: The distance from P(-3,4) to the y-axis is 3 --/
theorem distance_P_to_y_axis :
  distance_to_y_axis P.1 P.2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_distance_P_to_y_axis_l2738_273852
