import Mathlib

namespace expression_value_l3248_324824

theorem expression_value (p q r s : ℝ) 
  (hp : p ≠ 6) (hq : q ≠ 7) (hr : r ≠ 8) (hs : s ≠ 9) : 
  (p - 6) / (8 - r) * (q - 7) / (6 - p) * (r - 8) / (7 - q) * (s - 9) / (9 - s) = 1 := by
  sorry

end expression_value_l3248_324824


namespace complement_A_intersect_B_l3248_324807

def A : Set ℝ := {x | x + 1 > 0}
def B : Set ℝ := {-2, -1, 0, 1}

theorem complement_A_intersect_B :
  (Set.compl A) ∩ B = {-2, -1} := by sorry

end complement_A_intersect_B_l3248_324807


namespace correct_distribution_probability_l3248_324837

/-- Represents the number of guests -/
def num_guests : ℕ := 4

/-- Represents the total number of rolls -/
def total_rolls : ℕ := 8

/-- Represents the number of cheese rolls -/
def cheese_rolls : ℕ := 4

/-- Represents the number of fruit rolls -/
def fruit_rolls : ℕ := 4

/-- Represents the number of rolls per guest -/
def rolls_per_guest : ℕ := 2

/-- The probability of each guest getting one cheese roll and one fruit roll -/
def probability_correct_distribution : ℚ := 1 / 35

theorem correct_distribution_probability :
  probability_correct_distribution = 
    (cheese_rolls.choose 1 * fruit_rolls.choose 1 / (total_rolls.choose 2)) *
    ((cheese_rolls - 1).choose 1 * (fruit_rolls - 1).choose 1 / ((total_rolls - 2).choose 2)) *
    ((cheese_rolls - 2).choose 1 * (fruit_rolls - 2).choose 1 / ((total_rolls - 4).choose 2)) *
    1 := by sorry

#check correct_distribution_probability

end correct_distribution_probability_l3248_324837


namespace x_intercept_after_rotation_l3248_324815

/-- Given a line m with equation 2x - 3y + 30 = 0 in the coordinate plane,
    rotated 30° counterclockwise about the point (10, 10) to form line n,
    the x-coordinate of the x-intercept of line n is (20√3 + 20) / (2√3 + 3). -/
theorem x_intercept_after_rotation :
  let m : Set (ℝ × ℝ) := {(x, y) | 2 * x - 3 * y + 30 = 0}
  let center : ℝ × ℝ := (10, 10)
  let angle : ℝ := π / 6  -- 30° in radians
  let n : Set (ℝ × ℝ) := {(x, y) | ∃ (x₀ y₀ : ℝ), (x₀, y₀) ∈ m ∧
    x - 10 = (x₀ - 10) * Real.cos angle - (y₀ - 10) * Real.sin angle ∧
    y - 10 = (x₀ - 10) * Real.sin angle + (y₀ - 10) * Real.cos angle}
  let x_intercept : ℝ := (20 * Real.sqrt 3 + 20) / (2 * Real.sqrt 3 + 3)
  (0, x_intercept) ∈ n := by sorry

end x_intercept_after_rotation_l3248_324815


namespace port_vessels_l3248_324801

theorem port_vessels (cruise_ships cargo_ships sailboats fishing_boats : ℕ) :
  cruise_ships = 4 →
  cargo_ships = 2 * cruise_ships →
  ∃ (x : ℕ), sailboats = cargo_ships + x →
  sailboats = 7 * fishing_boats →
  cruise_ships + cargo_ships + sailboats + fishing_boats = 28 →
  sailboats - cargo_ships = 6 := by
  sorry

end port_vessels_l3248_324801


namespace pedro_extra_squares_l3248_324826

theorem pedro_extra_squares (jesus_squares linden_squares pedro_squares : ℕ) 
  (h1 : jesus_squares = 60)
  (h2 : linden_squares = 75)
  (h3 : pedro_squares = 200) :
  pedro_squares - (jesus_squares + linden_squares) = 65 := by
  sorry

end pedro_extra_squares_l3248_324826


namespace function_equality_l3248_324840

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > 0 then Real.log x else a^x

theorem function_equality (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  f a (Real.exp 2) = f a (-2) → a = Real.sqrt 2 / 2 := by
  sorry

end function_equality_l3248_324840


namespace m_range_l3248_324844

-- Define the quadratic equations
def eq1 (m : ℝ) (x : ℝ) : Prop := x^2 + m*x + 1 = 0
def eq2 (m : ℝ) (x : ℝ) : Prop := 4*x^2 + 4*(m+2)*x + 1 = 0

-- Define the conditions
def has_two_distinct_roots (m : ℝ) : Prop :=
  ∃ x y, x ≠ y ∧ eq1 m x ∧ eq1 m y

def has_no_real_roots (m : ℝ) : Prop :=
  ∀ x, ¬(eq2 m x)

-- State the theorem
theorem m_range (m : ℝ) :
  has_two_distinct_roots m ∧ has_no_real_roots m ↔ -3 < m ∧ m < -2 :=
sorry

end m_range_l3248_324844


namespace lending_period_is_one_year_l3248_324869

/-- 
Given a person who:
- Borrows an amount at a certain interest rate
- Lends the same amount at a higher interest rate
- Makes a fixed gain per year

This theorem proves that the lending period is 1 year under specific conditions.
-/
theorem lending_period_is_one_year 
  (borrowed_amount : ℝ) 
  (borrowing_rate : ℝ) 
  (lending_rate : ℝ) 
  (gain_per_year : ℝ) 
  (h1 : borrowed_amount = 5000)
  (h2 : borrowing_rate = 0.04)
  (h3 : lending_rate = 0.06)
  (h4 : gain_per_year = 100)
  : ∃ t : ℝ, t = 1 ∧ borrowed_amount * lending_rate * t - borrowed_amount * borrowing_rate * t = gain_per_year :=
sorry

end lending_period_is_one_year_l3248_324869


namespace triangle_formation_l3248_324870

theorem triangle_formation (a b c : ℝ) : 
  a = 4 ∧ b = 9 ∧ c = 9 → 
  a + b > c ∧ b + c > a ∧ c + a > b :=
by sorry

end triangle_formation_l3248_324870


namespace max_volume_pyramid_l3248_324855

noncomputable def pyramid_volume (a b : ℝ) : ℝ :=
  (a * b * Real.sqrt (3 * a^2 - b^2)) / 6

theorem max_volume_pyramid (a : ℝ) (h : a > 0) :
  ∃ b : ℝ, b > 0 ∧ ∀ x : ℝ, x > 0 → pyramid_volume a b ≥ pyramid_volume a x :=
by
  -- The proof goes here
  sorry

end max_volume_pyramid_l3248_324855


namespace probability_at_least_two_same_8sided_dice_l3248_324811

theorem probability_at_least_two_same_8sided_dice (n : ℕ) (s : ℕ) (p : ℚ) :
  n = 5 →
  s = 8 →
  p = 1628 / 2048 →
  p = 1 - (s * (s - 1) * (s - 2) * (s - 3) * (s - 4) : ℚ) / s^n :=
by sorry

end probability_at_least_two_same_8sided_dice_l3248_324811


namespace prob_two_white_is_three_tenths_l3248_324843

-- Define the total number of balls
def total_balls : ℕ := 5

-- Define the number of white balls
def white_balls : ℕ := 3

-- Define the number of black balls
def black_balls : ℕ := 2

-- Define the number of ways to choose 2 balls from the total
def total_choices : ℕ := Nat.choose total_balls 2

-- Define the number of ways to choose 2 white balls
def white_choices : ℕ := Nat.choose white_balls 2

-- Define the probability of drawing two white balls given that they are the same color
def prob_two_white : ℚ := white_choices / total_choices

-- Theorem to prove
theorem prob_two_white_is_three_tenths : 
  prob_two_white = 3 / 10 := by sorry

end prob_two_white_is_three_tenths_l3248_324843


namespace pizza_slices_remaining_l3248_324899

/-- Given a pizza with 8 slices, if two people each eat 3/2 slices, then 5 slices remain. -/
theorem pizza_slices_remaining (total_slices : ℕ) (slices_per_person : ℚ) (people : ℕ) : 
  total_slices = 8 → slices_per_person = 3/2 → people = 2 → 
  total_slices - (↑people * slices_per_person).num = 5 := by
  sorry

end pizza_slices_remaining_l3248_324899


namespace floor_length_approx_l3248_324854

/-- Represents a rectangular floor with length and breadth -/
structure RectangularFloor where
  breadth : ℝ
  length : ℝ

/-- The properties of our specific rectangular floor -/
def floor_properties (floor : RectangularFloor) : Prop :=
  floor.length = 3 * floor.breadth ∧
  floor.length * floor.breadth = 60

/-- The theorem stating the length of the floor -/
theorem floor_length_approx (floor : RectangularFloor) 
  (h : floor_properties floor) : 
  ∃ ε > 0, abs (floor.length - 13.416) < ε :=
sorry

end floor_length_approx_l3248_324854


namespace shopper_receive_amount_l3248_324892

/-- The amount of money each person has and donates --/
def problem (isabella sam giselle valentina ethan : ℚ) : Prop :=
  isabella = giselle + 15 ∧
  isabella = sam + 45 ∧
  giselle = 120 ∧
  valentina = 2 * sam ∧
  ethan = isabella - 75

/-- The total donation amount --/
def total_donation (isabella sam giselle valentina ethan : ℚ) : ℚ :=
  0.2 * isabella + 0.15 * sam + 0.1 * giselle + 0.25 * valentina + 0.3 * ethan

/-- The amount each shopper receives after equal distribution --/
def shopper_receive (isabella sam giselle valentina ethan : ℚ) : ℚ :=
  (total_donation isabella sam giselle valentina ethan) / 4

/-- Theorem stating the amount each shopper receives --/
theorem shopper_receive_amount :
  ∀ isabella sam giselle valentina ethan,
  problem isabella sam giselle valentina ethan →
  shopper_receive isabella sam giselle valentina ethan = 28.875 :=
by sorry

end shopper_receive_amount_l3248_324892


namespace cube_coloring_count_l3248_324814

/-- The number of distinct colorings of a cube's vertices -/
def distinctCubeColorings (m : ℕ) : ℚ :=
  (1 / 24) * m^2 * (m^6 + 17 * m^2 + 6)

/-- Theorem: The number of distinct ways to color the 8 vertices of a cube
    with m different colors, considering the symmetries of the cube,
    is equal to (1/24) * m^2 * (m^6 + 17m^2 + 6) -/
theorem cube_coloring_count (m : ℕ) :
  (distinctCubeColorings m) = (1 / 24) * m^2 * (m^6 + 17 * m^2 + 6) := by
  sorry

end cube_coloring_count_l3248_324814


namespace annes_cats_weight_l3248_324887

/-- The total weight of Anne's four cats -/
def total_weight (first_female_weight : ℝ) : ℝ :=
  let second_female_weight := 1.5 * first_female_weight
  let first_male_weight := 2 * first_female_weight
  let second_male_weight := first_female_weight + second_female_weight
  first_female_weight + second_female_weight + first_male_weight + second_male_weight

/-- Theorem stating that the total weight of Anne's four cats is 14 kilograms -/
theorem annes_cats_weight : total_weight 2 = 14 := by
  sorry

end annes_cats_weight_l3248_324887


namespace combined_original_price_l3248_324861

/-- Proves that the combined original price of a candy box, a can of soda, and a bag of chips
    was 34 pounds, given their new prices after specific percentage increases. -/
theorem combined_original_price (candy_new : ℝ) (soda_new : ℝ) (chips_new : ℝ)
    (h_candy : candy_new = 20)
    (h_soda : soda_new = 6)
    (h_chips : chips_new = 8)
    (h_candy_increase : candy_new = (5/4) * (candy_new - (1/4) * candy_new))
    (h_soda_increase : soda_new = (3/2) * (soda_new - (1/2) * soda_new))
    (h_chips_increase : chips_new = (11/10) * (chips_new - (1/10) * chips_new)) :
  (candy_new - (1/4) * candy_new) + (soda_new - (1/2) * soda_new) + (chips_new - (1/10) * chips_new) = 34 := by
  sorry

end combined_original_price_l3248_324861


namespace function_symmetry_and_translation_l3248_324880

-- Define a function that represents a horizontal translation
def translate (f : ℝ → ℝ) (h : ℝ) : ℝ → ℝ := λ x ↦ f (x - h)

-- Define symmetry with respect to y-axis
def symmetric_to_y_axis (f g : ℝ → ℝ) : Prop :=
  ∀ x, f x = g (-x)

-- State the theorem
theorem function_symmetry_and_translation :
  ∀ f : ℝ → ℝ,
  symmetric_to_y_axis (translate f 1) (λ x ↦ Real.exp x) →
  f = λ x ↦ Real.exp (-x - 1) :=
sorry

end function_symmetry_and_translation_l3248_324880


namespace smallest_base_perfect_square_base_ten_is_perfect_square_ten_is_smallest_base_l3248_324846

theorem smallest_base_perfect_square : 
  ∀ b : ℕ, b > 4 → (2 * b + 5 = n ^ 2 → n ≥ 5) → b ≥ 10 :=
by
  sorry

theorem base_ten_is_perfect_square : 
  ∃ n : ℕ, 2 * 10 + 5 = n ^ 2 :=
by
  sorry

theorem ten_is_smallest_base :
  (∀ b : ℕ, b > 4 ∧ b < 10 → ¬∃ n : ℕ, 2 * b + 5 = n ^ 2) ∧
  (∃ n : ℕ, 2 * 10 + 5 = n ^ 2) :=
by
  sorry

end smallest_base_perfect_square_base_ten_is_perfect_square_ten_is_smallest_base_l3248_324846


namespace sine_shift_left_specific_sine_shift_shift_result_l3248_324851

/-- Shifting a sine function to the left -/
theorem sine_shift_left (A : ℝ) (ω : ℝ) (φ : ℝ) (h : ℝ) :
  (fun x => A * Real.sin (ω * (x + h) + φ)) =
  (fun x => A * Real.sin (ω * x + (ω * h + φ))) :=
by sorry

/-- The specific case of shifting y = 3sin(2x + π/6) left by π/6 -/
theorem specific_sine_shift :
  (fun x => 3 * Real.sin (2 * x + π/6)) =
  (fun x => 3 * Real.sin (2 * (x - π/6) + π/6)) :=
by sorry

/-- The result of the shift is y = 3sin(2x - π/6) -/
theorem shift_result :
  (fun x => 3 * Real.sin (2 * (x - π/6) + π/6)) =
  (fun x => 3 * Real.sin (2 * x - π/6)) :=
by sorry

end sine_shift_left_specific_sine_shift_shift_result_l3248_324851


namespace four_valid_a_values_l3248_324808

theorem four_valid_a_values : 
  let equation_solution (a : ℝ) := (a - 2 : ℝ)
  let inequality_system (a y : ℝ) := y + 9 ≤ 2 * (y + 2) ∧ (2 * y - a) / 3 ≥ 1
  let valid_a (a : ℤ) := 
    equation_solution a > 0 ∧ 
    equation_solution a ≠ 3 ∧ 
    (∀ y : ℝ, inequality_system a y ↔ y ≥ 5)
  ∃! (s : Finset ℤ), (∀ a ∈ s, valid_a a) ∧ s.card = 4 :=
by sorry

end four_valid_a_values_l3248_324808


namespace female_officers_count_l3248_324820

theorem female_officers_count (total_on_duty : ℕ) (female_on_duty_ratio : ℚ) 
  (female_ratio : ℚ) (female_on_duty_percent : ℚ) :
  total_on_duty = 160 →
  female_ratio = 1/2 →
  female_on_duty_percent = 16/100 →
  (female_on_duty_ratio * ↑total_on_duty : ℚ) = (female_ratio * ↑total_on_duty : ℚ) →
  (female_on_duty_percent * (female_on_duty_ratio * ↑total_on_duty / female_on_duty_percent : ℚ) : ℚ) = 500 := by
  sorry

end female_officers_count_l3248_324820


namespace not_invertible_sum_of_squares_l3248_324828

open Matrix

variable {n : Type*} [Fintype n] [DecidableEq n]

theorem not_invertible_sum_of_squares (M N : Matrix n n ℝ) 
  (h_neq : M ≠ N) 
  (h_cube : M ^ 3 = N ^ 3) 
  (h_comm : M ^ 2 * N = N ^ 2 * M) : 
  ¬(IsUnit (M ^ 2 + N ^ 2)) := by
sorry

end not_invertible_sum_of_squares_l3248_324828


namespace sand_art_proof_l3248_324810

/-- The amount of sand needed to fill a rectangular patch and a square patch -/
def total_sand_needed (rect_length rect_width square_side sand_per_inch : ℕ) : ℕ :=
  ((rect_length * rect_width) + (square_side * square_side)) * sand_per_inch

/-- Proof that the total amount of sand needed is 201 grams -/
theorem sand_art_proof :
  total_sand_needed 6 7 5 3 = 201 := by
  sorry

end sand_art_proof_l3248_324810


namespace two_numbers_difference_l3248_324866

theorem two_numbers_difference (a b : ℕ) : 
  a + b = 20500 →
  b % 5 = 0 →
  b = 10 * a + 5 →
  b - a = 16777 := by
sorry

end two_numbers_difference_l3248_324866


namespace stating_exists_k_no_carries_l3248_324805

/-- 
Given two positive integers a and b, returns true if adding a to b
results in no carries during the whole calculation in base 10.
-/
def no_carries (a b : ℕ) : Prop :=
  ∀ d : ℕ, d < 10 → (a / 10^d % 10 + b / 10^d % 10 < 10)

/-- 
Theorem stating that there exists a positive integer k such that
adding 1996k to 1997k results in no carries during the whole calculation.
-/
theorem exists_k_no_carries : ∃ k : ℕ, k > 0 ∧ no_carries (1996 * k) (1997 * k) := by
  sorry

end stating_exists_k_no_carries_l3248_324805


namespace rabbit_log_sawing_l3248_324817

theorem rabbit_log_sawing (cuts pieces : ℕ) (h1 : cuts = 10) (h2 : pieces = 16) :
  pieces - cuts = 6 := by
  sorry

end rabbit_log_sawing_l3248_324817


namespace square_side_length_equals_rectangle_root_area_l3248_324845

theorem square_side_length_equals_rectangle_root_area 
  (rectangle_length : ℝ) 
  (rectangle_breadth : ℝ) 
  (square_side : ℝ) 
  (h1 : rectangle_length = 250) 
  (h2 : rectangle_breadth = 160) 
  (h3 : square_side * square_side = rectangle_length * rectangle_breadth) : 
  square_side = 200 := by
sorry

end square_side_length_equals_rectangle_root_area_l3248_324845


namespace bd_length_l3248_324812

-- Define the triangles and their properties
def right_triangle (a b c : ℝ) : Prop := a^2 + b^2 = c^2

theorem bd_length (c : ℝ) :
  ∀ (AB BC AC AD BD : ℝ),
  right_triangle BC AC AB →
  right_triangle AD BD AB →
  BC = 3 →
  AC = c →
  AD = c - 1 →
  BD = Real.sqrt (2 * c + 8) := by
  sorry

end bd_length_l3248_324812


namespace ratio_of_numbers_l3248_324821

theorem ratio_of_numbers (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x > y) 
  (hsum : x + y = 7 * (x - y)) : x / y = 4 / 3 := by
  sorry

end ratio_of_numbers_l3248_324821


namespace triangle_side_length_l3248_324897

theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) :
  0 < a ∧ 0 < b ∧ 0 < c →
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π →
  A + B + C = π →
  a = 4 →
  B = π / 3 →
  A = π / 4 →
  b = 2 * Real.sqrt 6 :=
by sorry

end triangle_side_length_l3248_324897


namespace simplify_polynomial_no_x_squared_l3248_324876

-- Define the polynomial
def polynomial (x m : ℝ) : ℝ := 4*x^2 - 3*x + 5 - 2*m*x^2 - x + 1

-- Define the coefficient of x^2
def coeff_x_squared (m : ℝ) : ℝ := 4 - 2*m

-- Theorem statement
theorem simplify_polynomial_no_x_squared :
  ∃ (m : ℝ), coeff_x_squared m = 0 ∧ m = 2 :=
sorry

end simplify_polynomial_no_x_squared_l3248_324876


namespace interest_rate_is_six_percent_l3248_324859

-- Define the loan parameters
def initial_loan : ℝ := 10000
def initial_period : ℝ := 2
def additional_loan : ℝ := 12000
def additional_period : ℝ := 3
def total_repayment : ℝ := 27160

-- Define the function to calculate the total amount to be repaid
def total_amount (rate : ℝ) : ℝ :=
  initial_loan * (1 + rate * (initial_period + additional_period)) +
  additional_loan * (1 + rate * additional_period)

-- Theorem statement
theorem interest_rate_is_six_percent :
  ∃ (rate : ℝ), rate > 0 ∧ rate < 1 ∧ total_amount rate = total_repayment ∧ rate = 0.06 := by
  sorry

end interest_rate_is_six_percent_l3248_324859


namespace classroom_seating_l3248_324839

/-- Given a classroom with 53 students seated in rows of either 6 or 7 students,
    with all seats occupied, prove that the number of rows seating exactly 7 students is 5. -/
theorem classroom_seating (total_students : ℕ) (rows_with_seven : ℕ) : 
  total_students = 53 →
  (∃ (rows_with_six : ℕ), total_students = 7 * rows_with_seven + 6 * rows_with_six) →
  rows_with_seven = 5 := by
  sorry

end classroom_seating_l3248_324839


namespace successive_discounts_equivalent_to_single_discount_l3248_324849

-- Define the successive discounts
def discount1 : ℝ := 0.10
def discount2 : ℝ := 0.15
def discount3 : ℝ := 0.25

-- Define the equivalent single discount
def equivalent_discount : ℝ := 0.426

-- Theorem statement
theorem successive_discounts_equivalent_to_single_discount :
  (1 - discount1) * (1 - discount2) * (1 - discount3) = 1 - equivalent_discount :=
by sorry

end successive_discounts_equivalent_to_single_discount_l3248_324849


namespace fraction_multiplication_l3248_324874

theorem fraction_multiplication :
  (2 : ℚ) / 3 * (5 : ℚ) / 7 * (11 : ℚ) / 13 = (110 : ℚ) / 273 := by
  sorry

end fraction_multiplication_l3248_324874


namespace x_plus_y_equals_1003_l3248_324895

theorem x_plus_y_equals_1003 
  (x y : ℝ) 
  (h1 : x + Real.cos y = 1004)
  (h2 : x + 1004 * Real.sin y = 1003)
  (h3 : 0 ≤ y ∧ y ≤ Real.pi / 2) :
  x + y = 1003 := by sorry

end x_plus_y_equals_1003_l3248_324895


namespace page_lines_increase_l3248_324834

theorem page_lines_increase (original : ℕ) (new : ℕ) (increase_percent : ℚ) : 
  new = 240 ∧ 
  increase_percent = 50 ∧ 
  new = original + (increase_percent / 100 : ℚ) * original →
  new - original = 80 := by
  sorry

end page_lines_increase_l3248_324834


namespace subset_implies_m_values_l3248_324804

def A (m : ℝ) : Set ℝ := {1, 3, 2*m+3}
def B (m : ℝ) : Set ℝ := {3, m^2}

theorem subset_implies_m_values (m : ℝ) : B m ⊆ A m → m = 1 ∨ m = 3 := by
  sorry

end subset_implies_m_values_l3248_324804


namespace math_class_students_count_l3248_324868

theorem math_class_students_count :
  ∃! n : ℕ, n < 50 ∧ n % 8 = 5 ∧ n % 6 = 3 ∧ n = 45 :=
by sorry

end math_class_students_count_l3248_324868


namespace real_part_of_complex_fraction_l3248_324873

theorem real_part_of_complex_fraction :
  let i : ℂ := Complex.I
  (2 * i / (1 + i)).re = 1 := by sorry

end real_part_of_complex_fraction_l3248_324873


namespace alex_needs_three_packs_l3248_324877

/-- The number of burgers Alex plans to cook for each guest -/
def burgers_per_guest : ℕ := 3

/-- The number of friends Alex invited -/
def total_friends : ℕ := 10

/-- The number of friends who don't eat meat -/
def non_meat_eaters : ℕ := 1

/-- The number of friends who don't eat bread -/
def non_bread_eaters : ℕ := 1

/-- The number of buns in each pack -/
def buns_per_pack : ℕ := 8

/-- The function to calculate the number of packs of buns Alex needs to buy -/
def packs_of_buns_needed : ℕ :=
  let total_guests := total_friends - non_meat_eaters
  let total_burgers := burgers_per_guest * total_guests
  let burgers_needing_buns := total_burgers - (burgers_per_guest * non_bread_eaters)
  (burgers_needing_buns + buns_per_pack - 1) / buns_per_pack

/-- Theorem stating that Alex needs to buy 3 packs of buns -/
theorem alex_needs_three_packs : packs_of_buns_needed = 3 := by
  sorry

end alex_needs_three_packs_l3248_324877


namespace linear_function_point_l3248_324896

/-- Given a linear function y = x - 1 that passes through the point (m, 2), prove that m = 3 -/
theorem linear_function_point (m : ℝ) : (2 : ℝ) = m - 1 → m = 3 := by
  sorry

end linear_function_point_l3248_324896


namespace min_distance_exp_curve_to_line_l3248_324883

/-- The minimum distance from a point on the curve y = e^x to the line y = x is √2/2 -/
theorem min_distance_exp_curve_to_line : 
  ∃ (d : ℝ), d = Real.sqrt 2 / 2 ∧ 
  ∀ (x y : ℝ), y = Real.exp x → 
  d ≤ Real.sqrt ((x - y)^2 + (y - x)^2) / 2 :=
sorry

end min_distance_exp_curve_to_line_l3248_324883


namespace rectangle_perimeter_l3248_324847

/-- Given a rectangle where the sum of its length and width is 24 centimeters,
    prove that its perimeter is 48 centimeters. -/
theorem rectangle_perimeter (length width : ℝ) (h : length + width = 24) :
  2 * (length + width) = 48 :=
by sorry

end rectangle_perimeter_l3248_324847


namespace parabola_translation_l3248_324850

/-- Represents a parabola in 2D space -/
structure Parabola where
  f : ℝ → ℝ

/-- Applies a vertical translation to a parabola -/
def verticalTranslate (p : Parabola) (v : ℝ) : Parabola where
  f := fun x => p.f x + v

/-- Applies a horizontal translation to a parabola -/
def horizontalTranslate (p : Parabola) (h : ℝ) : Parabola where
  f := fun x => p.f (x + h)

/-- The original parabola y = -x^2 -/
def originalParabola : Parabola where
  f := fun x => -x^2

/-- Theorem stating that translating the parabola y = -x^2 upward by 2 units
    and to the left by 3 units results in the equation y = -(x + 3)^2 + 2 -/
theorem parabola_translation :
  (horizontalTranslate (verticalTranslate originalParabola 2) 3).f =
  fun x => -(x + 3)^2 + 2 := by
  sorry

end parabola_translation_l3248_324850


namespace intersection_of_sets_l3248_324806

theorem intersection_of_sets (a : ℝ) : 
  let A : Set ℝ := {-1, 0, 1}
  let B : Set ℝ := {a - 1, a + 1/a}
  (A ∩ B = {0}) → a = 1 := by
  sorry

end intersection_of_sets_l3248_324806


namespace paving_cost_theorem_l3248_324841

/-- Represents the dimensions and cost of a rectangular room -/
structure RectangularRoom where
  length : ℝ
  width : ℝ
  cost_per_sqm : ℝ

/-- Represents the dimensions and cost of a triangular room -/
structure TriangularRoom where
  base : ℝ
  height : ℝ
  cost_per_sqm : ℝ

/-- Represents the dimensions and cost of a trapezoidal room -/
structure TrapezoidalRoom where
  parallel_side1 : ℝ
  parallel_side2 : ℝ
  height : ℝ
  cost_per_sqm : ℝ

/-- Calculates the total cost of paving three rooms -/
def total_paving_cost (room1 : RectangularRoom) (room2 : TriangularRoom) (room3 : TrapezoidalRoom) : ℝ :=
  (room1.length * room1.width * room1.cost_per_sqm) +
  (0.5 * room2.base * room2.height * room2.cost_per_sqm) +
  (0.5 * (room3.parallel_side1 + room3.parallel_side2) * room3.height * room3.cost_per_sqm)

/-- Theorem stating the total cost of paving the three rooms -/
theorem paving_cost_theorem (room1 : RectangularRoom) (room2 : TriangularRoom) (room3 : TrapezoidalRoom)
  (h1 : room1 = { length := 5.5, width := 3.75, cost_per_sqm := 1400 })
  (h2 : room2 = { base := 4, height := 3, cost_per_sqm := 1500 })
  (h3 : room3 = { parallel_side1 := 6, parallel_side2 := 3.5, height := 2.5, cost_per_sqm := 1600 }) :
  total_paving_cost room1 room2 room3 = 56875 := by
  sorry

#eval total_paving_cost
  { length := 5.5, width := 3.75, cost_per_sqm := 1400 }
  { base := 4, height := 3, cost_per_sqm := 1500 }
  { parallel_side1 := 6, parallel_side2 := 3.5, height := 2.5, cost_per_sqm := 1600 }

end paving_cost_theorem_l3248_324841


namespace mark_kate_difference_l3248_324803

/-- The number of hours Kate charged to the project -/
def kate_hours : ℕ := 28

/-- The total number of hours charged to the project -/
def total_hours : ℕ := 180

/-- Pat's hours are twice Kate's -/
def pat_hours : ℕ := 2 * kate_hours

/-- Mark's hours are three times Kate's -/
def mark_hours : ℕ := 3 * kate_hours

/-- Linda's hours are half of Kate's -/
def linda_hours : ℕ := kate_hours / 2

theorem mark_kate_difference :
  mark_hours - kate_hours = 56 ∧
  pat_hours + kate_hours + mark_hours + linda_hours = total_hours :=
by sorry

end mark_kate_difference_l3248_324803


namespace kevin_repaired_phones_l3248_324867

/-- The number of phones Kevin repaired by the afternoon -/
def phones_repaired : ℕ := 3

/-- The initial number of phones Kevin had to repair -/
def initial_phones : ℕ := 15

/-- The number of phones dropped off by a client -/
def new_phones : ℕ := 6

/-- The number of phones each person (Kevin and his coworker) needs to repair -/
def phones_per_person : ℕ := 9

theorem kevin_repaired_phones :
  phones_repaired = 3 ∧
  initial_phones - phones_repaired + new_phones = 2 * phones_per_person :=
sorry

end kevin_repaired_phones_l3248_324867


namespace half_angle_quadrant_l3248_324852

def second_quadrant (α : Real) : Prop :=
  ∃ k : ℤ, α ∈ Set.Ioo (2 * k * Real.pi + Real.pi / 2) (2 * k * Real.pi + Real.pi)

theorem half_angle_quadrant (α : Real) (h : second_quadrant α) :
  ∃ k : ℤ, α / 2 ∈ Set.Ioo (k * Real.pi) (k * Real.pi + Real.pi / 2) :=
by sorry

end half_angle_quadrant_l3248_324852


namespace expression_evaluation_l3248_324857

theorem expression_evaluation : 
  3 + Real.sqrt 3 + 1 / (3 + Real.sqrt 3) + 1 / (Real.sqrt 3 - 3) = 3 + (2 * Real.sqrt 3) / 3 := by
  sorry

end expression_evaluation_l3248_324857


namespace jeans_average_speed_l3248_324832

/-- Proves that Jean's average speed is 18/11 mph given the problem conditions --/
theorem jeans_average_speed :
  let trail_length : ℝ := 12
  let uphill_length : ℝ := 4
  let chantal_flat_speed : ℝ := 3
  let chantal_uphill_speed : ℝ := 1.5
  let chantal_downhill_speed : ℝ := 2.25
  let jean_delay : ℝ := 2

  let chantal_flat_time : ℝ := (trail_length - uphill_length) / chantal_flat_speed
  let chantal_uphill_time : ℝ := uphill_length / chantal_uphill_speed
  let chantal_downhill_time : ℝ := uphill_length / chantal_downhill_speed
  let chantal_total_time : ℝ := chantal_flat_time + chantal_uphill_time + chantal_downhill_time

  let jean_travel_time : ℝ := chantal_total_time - jean_delay
  let jean_travel_distance : ℝ := uphill_length

  jean_travel_distance / jean_travel_time = 18 / 11 := by sorry

end jeans_average_speed_l3248_324832


namespace point_line_distance_l3248_324838

/-- Given a point (4, 3) and a line 3x - 4y + a = 0, if the distance from the point to the line is 1, then a = ±5 -/
theorem point_line_distance (a : ℝ) : 
  let point : ℝ × ℝ := (4, 3)
  let line_equation (x y : ℝ) := 3 * x - 4 * y + a
  let distance := |line_equation point.1 point.2| / Real.sqrt (3^2 + (-4)^2)
  distance = 1 → a = 5 ∨ a = -5 := by
sorry

end point_line_distance_l3248_324838


namespace darcys_shorts_l3248_324865

theorem darcys_shorts (total_shirts : ℕ) (folded_shirts : ℕ) (folded_shorts : ℕ) (remaining_to_fold : ℕ) : 
  total_shirts = 20 →
  folded_shirts = 12 →
  folded_shorts = 5 →
  remaining_to_fold = 11 →
  total_shirts + (folded_shorts + (remaining_to_fold - (total_shirts - folded_shirts))) = 28 :=
by sorry

end darcys_shorts_l3248_324865


namespace infinitely_many_divisible_numbers_l3248_324886

theorem infinitely_many_divisible_numbers :
  ∃ (a : ℕ → ℕ), (∀ n : ℕ, a n ∣ 2^(a n) + 3^(a n)) ∧
                 (∀ n : ℕ, a n < a (n + 1)) :=
sorry

end infinitely_many_divisible_numbers_l3248_324886


namespace common_point_theorem_l3248_324884

/-- Represents a line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if a point lies on a line -/
def Line.contains (l : Line) (x y : ℝ) : Prop :=
  l.a * x + l.b * y = l.c

/-- Represents a geometric progression -/
def IsGeometricProgression (a c b : ℝ) : Prop :=
  ∃ r : ℝ, c = a * r ∧ b = a * r^2

theorem common_point_theorem :
  ∀ (l : Line), 
    IsGeometricProgression l.a l.c l.b →
    l.contains 0 0 :=
by sorry

end common_point_theorem_l3248_324884


namespace decimal_operation_order_l3248_324835

theorem decimal_operation_order : ¬ ∀ (a b c : ℚ), a + b - c = a + (b - c) := by
  sorry

end decimal_operation_order_l3248_324835


namespace line_and_circle_properties_l3248_324816

-- Define the line l
def line_l (x y : ℝ) : Prop := 2*x - 7*y + 8 = 0

-- Define the perpendicular line
def perp_line (x y : ℝ) : Prop := y = -7/2*x + 1

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x-3)^2 + (y-2)^2 = 13

-- Define points A and B
def point_A : ℝ × ℝ := (6, 0)
def point_B : ℝ × ℝ := (1, 5)

-- Theorem statement
theorem line_and_circle_properties :
  -- Line l passes through (3,2)
  line_l 3 2 ∧
  -- Line l is perpendicular to y = -7/2x + 1
  (∀ x y : ℝ, line_l x y → perp_line x y → x = y) ∧
  -- The center of circle C lies on line l
  (∃ x y : ℝ, line_l x y ∧ circle_C x y) ∧
  -- Circle C passes through points A and B
  circle_C point_A.1 point_A.2 ∧ circle_C point_B.1 point_B.2 →
  -- Conclusion 1: The equation of line l is 2x - 7y + 8 = 0
  (∀ x y : ℝ, line_l x y ↔ 2*x - 7*y + 8 = 0) ∧
  -- Conclusion 2: The standard equation of circle C is (x-3)^2 + (y-2)^2 = 13
  (∀ x y : ℝ, circle_C x y ↔ (x-3)^2 + (y-2)^2 = 13) :=
by
  sorry

end line_and_circle_properties_l3248_324816


namespace cards_found_l3248_324848

theorem cards_found (initial_cards final_cards : ℕ) : 
  initial_cards = 7 → final_cards = 54 → final_cards - initial_cards = 47 := by
  sorry

end cards_found_l3248_324848


namespace algebraic_expressions_l3248_324891

variable (a x : ℝ)

theorem algebraic_expressions :
  ((-3 * a^2)^3 - 4 * a^2 * a^4 + 5 * a^9 / a^3 = -26 * a^6) ∧
  (((x + 1) * (x + 2) + 2 * (x - 1)) / x = x + 5) :=
by sorry

end algebraic_expressions_l3248_324891


namespace cubic_tangent_line_problem_l3248_324823

/-- Given a cubic function f(x) = ax³ + x + 1, prove that if its tangent line
    at x = 1 passes through the point (2, 7), then a = 1. -/
theorem cubic_tangent_line_problem (a : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ a * x^3 + x + 1
  let f' : ℝ → ℝ := λ x ↦ 3 * a * x^2 + 1
  let tangent_line : ℝ → ℝ := λ x ↦ f 1 + f' 1 * (x - 1)
  tangent_line 2 = 7 → a = 1 := by
  sorry

end cubic_tangent_line_problem_l3248_324823


namespace vector_calculation_l3248_324864

def vector_subtraction (v w : Fin 2 → ℝ) : Fin 2 → ℝ := fun i => v i - w i

def scalar_mult (a : ℝ) (v : Fin 2 → ℝ) : Fin 2 → ℝ := fun i => a * v i

theorem vector_calculation :
  let v : Fin 2 → ℝ := ![5, -3]
  let w : Fin 2 → ℝ := ![3, -4]
  vector_subtraction v (scalar_mult (-2) w) = ![11, -11] := by sorry

end vector_calculation_l3248_324864


namespace intersection_with_complement_l3248_324818

-- Define the universal set U
def U : Finset Nat := {1,2,3,4,5,6}

-- Define set P
def P : Finset Nat := {1,2,3,4}

-- Define set Q
def Q : Finset Nat := {3,4,5}

-- Theorem statement
theorem intersection_with_complement :
  P ∩ (U \ Q) = {1,2} := by sorry

end intersection_with_complement_l3248_324818


namespace factorial_bounds_l3248_324885

theorem factorial_bounds (n : ℕ) (h : n ≥ 1) : 2^(n-1) ≤ n! ∧ n! ≤ n^n := by
  sorry

end factorial_bounds_l3248_324885


namespace tan_three_expression_value_l3248_324802

theorem tan_three_expression_value (θ : Real) (h : Real.tan θ = 3) :
  2 * (Real.sin θ)^2 - 3 * (Real.sin θ) * (Real.cos θ) - 4 * (Real.cos θ)^2 = -4/10 := by
  sorry

end tan_three_expression_value_l3248_324802


namespace square_floor_tiles_l3248_324819

theorem square_floor_tiles (s : ℕ) (h1 : s > 0) : 
  (2 * s - 1 : ℝ) / (s^2 : ℝ) = 0.41 → s^2 = 16 := by
  sorry

end square_floor_tiles_l3248_324819


namespace intersection_k_value_l3248_324809

/-- Given two lines that intersect at a point, find the value of k -/
theorem intersection_k_value (k : ℝ) : 
  (∀ x y, y = 2 * x + 3 → (x = 1 ∧ y = 5)) →  -- Line m passes through (1, 5)
  (∀ x y, y = k * x + 2 → (x = 1 ∧ y = 5)) →  -- Line n passes through (1, 5)
  k = 3 := by
sorry

end intersection_k_value_l3248_324809


namespace train_journey_encryption_train_journey_l3248_324882

/-- Represents a city name as a list of alphabet positions --/
def CityCode := List Nat

/-- Defines the alphabet positions for letters --/
def alphabetPosition (c : Char) : Nat :=
  match c with
  | 'A' => 1
  | 'B' => 2
  | 'U' => 21
  | 'K' => 11
  | _ => 0

/-- Encodes a city name to a list of alphabet positions --/
def encodeCity (name : String) : CityCode :=
  name.toList.map alphabetPosition

/-- Theorem: The encrypted city names represent Ufa and Baku --/
theorem train_journey_encryption (departure : CityCode) (arrival : CityCode) : 
  (departure = [21, 2, 1, 21] ∧ arrival = [2, 1, 11, 21]) →
  (encodeCity "UFA" = departure ∧ encodeCity "BAKU" = arrival) :=
by
  sorry

/-- Main theorem: The train traveled from Ufa to Baku --/
theorem train_journey : 
  ∃ (departure arrival : CityCode),
    departure = [21, 2, 1, 21] ∧
    arrival = [2, 1, 11, 21] ∧
    encodeCity "UFA" = departure ∧
    encodeCity "BAKU" = arrival :=
by
  sorry

end train_journey_encryption_train_journey_l3248_324882


namespace sum_of_x_sixth_powers_l3248_324890

theorem sum_of_x_sixth_powers (x : ℕ) (b : ℕ) :
  (x : ℝ) * (x : ℝ)^6 = (x : ℝ)^b → b = 7 := by
  sorry

end sum_of_x_sixth_powers_l3248_324890


namespace animal_sightings_proof_l3248_324853

/-- The number of times families see animals in January -/
def january_sightings : ℕ := 26

/-- The number of times families see animals in February -/
def february_sightings : ℕ := 3 * january_sightings

/-- The number of times families see animals in March -/
def march_sightings : ℕ := february_sightings / 2

/-- The total number of times families see animals in the first three months -/
def total_sightings : ℕ := january_sightings + february_sightings + march_sightings

theorem animal_sightings_proof : total_sightings = 143 := by
  sorry

end animal_sightings_proof_l3248_324853


namespace shortest_chord_length_for_given_circle_and_point_l3248_324830

/-- Circle represented by its equation -/
structure Circle where
  equation : ℝ → ℝ → ℝ

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the shortest chord length for a given circle and point -/
def shortestChordLength (c : Circle) (p : Point) : ℝ := sorry

/-- The main theorem -/
theorem shortest_chord_length_for_given_circle_and_point :
  let c : Circle := { equation := λ x y => x^2 + y^2 - 2*x - 3 }
  let p : Point := { x := 2, y := 1 }
  shortestChordLength c p = 2 * Real.sqrt 2 := by sorry

end shortest_chord_length_for_given_circle_and_point_l3248_324830


namespace cubic_roots_product_l3248_324813

theorem cubic_roots_product (a b c : ℂ) : 
  (3 * a^3 - 7 * a^2 + 4 * a - 9 = 0) ∧
  (3 * b^3 - 7 * b^2 + 4 * b - 9 = 0) ∧
  (3 * c^3 - 7 * c^2 + 4 * c - 9 = 0) →
  a * b * c = 3 := by
sorry

end cubic_roots_product_l3248_324813


namespace shawna_situps_wednesday_l3248_324888

/-- Calculates the number of situps Shawna needs to do on Wednesday -/
def situps_needed_wednesday (daily_goal : ℕ) (monday_situps : ℕ) (tuesday_situps : ℕ) : ℕ :=
  daily_goal + (daily_goal - monday_situps) + (daily_goal - tuesday_situps)

/-- Theorem: Given Shawna's daily goal and her performance on Monday and Tuesday,
    she needs to do 59 situps on Wednesday to meet her goal and make up for missed situps -/
theorem shawna_situps_wednesday :
  situps_needed_wednesday 30 12 19 = 59 := by
  sorry

end shawna_situps_wednesday_l3248_324888


namespace quadratic_roots_l3248_324898

theorem quadratic_roots (p q : ℤ) (h1 : p + q = 198) :
  ∃ x₁ x₂ : ℤ, (x₁^2 + p*x₁ + q = 0 ∧ x₂^2 + p*x₂ + q = 0) →
  ((x₁ = 2 ∧ x₂ = 200) ∨ (x₁ = 0 ∧ x₂ = -198)) := by
  sorry

end quadratic_roots_l3248_324898


namespace age_difference_l3248_324833

theorem age_difference (A B C : ℕ) (h : A + B = B + C + 12) : A - C = 12 := by
  sorry

end age_difference_l3248_324833


namespace polynomial_factorization_l3248_324858

theorem polynomial_factorization (x : ℝ) : 
  45 * x^6 - 270 * x^12 + 90 * x^7 = 45 * x^6 * (1 + 2*x - 6*x^6) := by
  sorry

end polynomial_factorization_l3248_324858


namespace probability_same_color_plates_l3248_324829

def red_plates : ℕ := 6
def blue_plates : ℕ := 5
def green_plates : ℕ := 3

def total_plates : ℕ := red_plates + blue_plates + green_plates

def same_color_combinations : ℕ := (
  Nat.choose red_plates 3 +
  Nat.choose blue_plates 3 +
  Nat.choose green_plates 3
)

def total_combinations : ℕ := Nat.choose total_plates 3

theorem probability_same_color_plates :
  (same_color_combinations : ℚ) / total_combinations = 31 / 364 := by
  sorry

end probability_same_color_plates_l3248_324829


namespace complex_equation_solution_l3248_324881

theorem complex_equation_solution :
  ∀ z : ℂ, z = Complex.I * (2 - z) → z = 1 + Complex.I :=
by
  sorry

end complex_equation_solution_l3248_324881


namespace orange_fraction_l3248_324822

theorem orange_fraction (total_fruit : ℕ) (oranges peaches apples : ℕ) :
  total_fruit = 56 →
  peaches = oranges / 2 →
  apples = 5 * peaches →
  apples = 35 →
  oranges = total_fruit / 4 := by
  sorry

end orange_fraction_l3248_324822


namespace children_clothing_production_l3248_324878

-- Define the constants
def total_sets : ℕ := 50
def type_a_fabric : ℝ := 38
def type_b_fabric : ℝ := 26

-- Define the fabric requirements and profits for each size
def size_l_type_a : ℝ := 0.5
def size_l_type_b : ℝ := 1
def size_l_profit : ℝ := 45

def size_m_type_a : ℝ := 0.9
def size_m_type_b : ℝ := 0.2
def size_m_profit : ℝ := 30

-- Define the profit function
def profit_function (x : ℝ) : ℝ := 15 * x + 1500

-- Theorem statement
theorem children_clothing_production (x : ℝ) :
  (17.5 ≤ x ∧ x ≤ 20) →
  (∀ y : ℝ, y = profit_function x) ∧
  (x * size_l_type_a + (total_sets - x) * size_m_type_a ≤ type_a_fabric) ∧
  (x * size_l_type_b + (total_sets - x) * size_m_type_b ≤ type_b_fabric) :=
by sorry

end children_clothing_production_l3248_324878


namespace arithmetic_sequence_common_difference_l3248_324862

def arithmetic_sequence (a : ℕ → ℝ) := ∀ n, a (n + 1) - a n = a 2 - a 1

theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℝ) 
  (h1 : arithmetic_sequence a) 
  (h2 : a 1 = 2) 
  (h3 : a 3 = 8) : 
  a 2 - a 1 = 3 := by
sorry

end arithmetic_sequence_common_difference_l3248_324862


namespace partition_6_4_l3248_324827

/-- The number of ways to partition n indistinguishable objects into at most k indistinguishable parts -/
def partition_count (n k : ℕ) : ℕ := sorry

/-- Theorem: There are 9 ways to partition 6 indistinguishable objects into at most 4 indistinguishable parts -/
theorem partition_6_4 : partition_count 6 4 = 9 := by sorry

end partition_6_4_l3248_324827


namespace carnival_tickets_l3248_324875

def ticket_distribution (n : Nat) : Nat :=
  let ratio := [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9]
  let total_parts := ratio.sum
  let tickets_per_part := n / total_parts
  tickets_per_part * total_parts

theorem carnival_tickets :
  let friends : Nat := 17
  let initial_tickets : Nat := 865
  let ratio := [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9]
  let total_parts := ratio.sum
  let next_multiple := (initial_tickets / total_parts + 1) * total_parts
  next_multiple - initial_tickets = 26 := by
  sorry

end carnival_tickets_l3248_324875


namespace boss_salary_percentage_larger_l3248_324856

-- Define Werner's salary as a percentage of his boss's salary
def werner_salary_percentage : ℝ := 20

-- Theorem statement
theorem boss_salary_percentage_larger (werner_salary boss_salary : ℝ) 
  (h : werner_salary = (werner_salary_percentage / 100) * boss_salary) : 
  (boss_salary / werner_salary - 1) * 100 = 400 := by
  sorry

end boss_salary_percentage_larger_l3248_324856


namespace route_comparison_l3248_324863

/-- Represents the time difference between two routes when all lights are red on the first route -/
def route_time_difference (first_route_base_time : ℕ) (red_light_delay : ℕ) (num_lights : ℕ) (second_route_time : ℕ) : ℕ :=
  (first_route_base_time + red_light_delay * num_lights) - second_route_time

theorem route_comparison :
  route_time_difference 10 3 3 14 = 5 := by
  sorry

end route_comparison_l3248_324863


namespace fraction_subtraction_l3248_324893

theorem fraction_subtraction : (18 : ℚ) / 42 - 3 / 8 = 3 / 56 := by
  sorry

end fraction_subtraction_l3248_324893


namespace money_loses_exchange_value_on_deserted_island_l3248_324825

-- Define the basic concepts
def Person : Type := String
def Money : Type := ℕ
def Item : Type := String

-- Define the properties of money
structure MoneyProperties :=
  (medium_of_exchange : Bool)
  (store_of_value : Bool)
  (unit_of_account : Bool)
  (standard_of_deferred_payment : Bool)

-- Define the island environment
structure Island :=
  (inhabitants : List Person)
  (items : List Item)
  (currency : Money)

-- Define the value of money in a given context
def money_value (island : Island) (props : MoneyProperties) : ℝ := 
  sorry

-- Theorem: Money loses its value as a medium of exchange on a deserted island
theorem money_loses_exchange_value_on_deserted_island 
  (island : Island) 
  (props : MoneyProperties) :
  island.inhabitants.length = 1 →
  money_value island props = 0 :=
sorry

end money_loses_exchange_value_on_deserted_island_l3248_324825


namespace max_b_value_l3248_324842

def is_valid_number (n : ℕ) : Prop :=
  ∃ (a b c : ℕ),
    a < 10 ∧ b < 10 ∧ c < 10 ∧
    n = a * 1000000 + 2 * 100000 + b * 10000 + 3 * 1000 + 4 * 100 + c

def is_divisible_by_55 (n : ℕ) : Prop :=
  n % 55 = 0

theorem max_b_value (n : ℕ) 
  (h1 : is_valid_number n) 
  (h2 : is_divisible_by_55 n) : 
  ∃ (a c : ℕ), ∃ (b : ℕ), b ≤ 7 ∧ 
    n = a * 1000000 + 2 * 100000 + b * 10000 + 3 * 1000 + 4 * 100 + c :=
sorry

end max_b_value_l3248_324842


namespace power_difference_equals_seven_l3248_324879

theorem power_difference_equals_seven : 2^5 - 5^2 = 7 := by
  sorry

end power_difference_equals_seven_l3248_324879


namespace log_27_3_l3248_324860

theorem log_27_3 : Real.log 3 / Real.log 27 = 1 / 3 := by
  sorry

end log_27_3_l3248_324860


namespace original_price_from_profit_and_selling_price_l3248_324894

/-- Given an article sold at a 10% profit with a selling price of 550, 
    the original price of the article is 500. -/
theorem original_price_from_profit_and_selling_price :
  ∀ (original_price selling_price : ℝ),
    selling_price = 550 →
    selling_price = original_price * 1.1 →
    original_price = 500 := by
  sorry

end original_price_from_profit_and_selling_price_l3248_324894


namespace real_solutions_range_l3248_324871

theorem real_solutions_range (m : ℝ) : 
  (∃ x : ℝ, (m - 2) * x^2 - 2 * x + 1 = 0) → m ≤ 3 :=
by sorry

end real_solutions_range_l3248_324871


namespace least_addition_for_divisibility_by_nine_l3248_324889

theorem least_addition_for_divisibility_by_nine :
  ∃ (n : ℕ), n = 5 ∧ 
  (∀ (m : ℕ), (228712 + m) % 9 = 0 → m ≥ n) ∧
  (228712 + n) % 9 = 0 := by
  sorry

end least_addition_for_divisibility_by_nine_l3248_324889


namespace buddy_card_count_l3248_324836

def card_count (initial : ℕ) : ℕ := 
  let tuesday := initial - (initial * 30 / 100)
  let wednesday := tuesday + (tuesday * 20 / 100)
  let thursday := wednesday - (wednesday * 25 / 100)
  let friday := thursday + (thursday / 3)
  let saturday := friday + (friday * 2)
  let sunday := saturday + (saturday * 40 / 100) - 15
  let next_monday := sunday + ((saturday * 40 / 100) * 3)
  next_monday

theorem buddy_card_count : card_count 200 = 1297 := by
  sorry

end buddy_card_count_l3248_324836


namespace telephone_fee_properties_l3248_324872

-- Define the telephone fee function
def telephone_fee (x : ℝ) : ℝ := 0.4 * x + 18

-- Theorem statement
theorem telephone_fee_properties :
  (∀ x : ℝ, telephone_fee x = 0.4 * x + 18) ∧
  (telephone_fee 10 = 22) ∧
  (telephone_fee 20 = 26) := by
  sorry


end telephone_fee_properties_l3248_324872


namespace total_students_on_trip_l3248_324831

/-- The number of students who went on a trip to the zoo -/
def students_on_trip (num_buses : ℕ) (students_per_bus : ℕ) (students_in_cars : ℕ) : ℕ :=
  num_buses * students_per_bus + students_in_cars

/-- Theorem stating the total number of students on the trip -/
theorem total_students_on_trip :
  students_on_trip 7 56 4 = 396 := by
  sorry

end total_students_on_trip_l3248_324831


namespace fence_painting_fraction_l3248_324800

theorem fence_painting_fraction (total_time : ℝ) (part_time : ℝ) 
  (h1 : total_time = 60) 
  (h2 : part_time = 12) : 
  (part_time / total_time) = (1 : ℝ) / 5 := by
  sorry

end fence_painting_fraction_l3248_324800
