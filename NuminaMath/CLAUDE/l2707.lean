import Mathlib

namespace min_value_parallel_vectors_l2707_270735

theorem min_value_parallel_vectors (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  let a : Fin 2 → ℝ := ![3, 2]
  let b : Fin 2 → ℝ := ![x, 1 - y]
  (∃ (k : ℝ), ∀ i, a i = k * b i) →
  (3 / x + 2 / y) ≥ 8 ∧ ∃ x₀ y₀, x₀ > 0 ∧ y₀ > 0 ∧ 3 / x₀ + 2 / y₀ = 8 :=
by sorry

end min_value_parallel_vectors_l2707_270735


namespace german_students_count_l2707_270726

theorem german_students_count (total_students : ℕ) 
                               (french_students : ℕ) 
                               (both_students : ℕ) 
                               (neither_students : ℕ) 
                               (h1 : total_students = 60)
                               (h2 : french_students = 41)
                               (h3 : both_students = 9)
                               (h4 : neither_students = 6) :
  ∃ german_students : ℕ, german_students = 22 ∧ 
    german_students + french_students - both_students + neither_students = total_students :=
by
  sorry

end german_students_count_l2707_270726


namespace ordered_pair_solution_l2707_270785

theorem ordered_pair_solution : ∃ (x y : ℤ), 
  Real.sqrt (16 - 12 * Real.cos (40 * π / 180)) = ↑x + ↑y * (1 / Real.cos (40 * π / 180)) ∧ 
  (x, y) = (2, 0) := by
  sorry

end ordered_pair_solution_l2707_270785


namespace sum_of_odds_is_even_product_zero_implies_factor_zero_exists_even_prime_l2707_270709

-- Definition of odd integer
def IsOdd (n : ℤ) : Prop := ∃ k : ℤ, n = 2*k + 1

-- Definition of even integer
def IsEven (n : ℤ) : Prop := ∃ k : ℤ, n = 2*k

-- Definition of prime number
def IsPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

theorem sum_of_odds_is_even (x y : ℤ) (hx : IsOdd x) (hy : IsOdd y) : 
  IsEven (x + y) := by sorry

theorem product_zero_implies_factor_zero (x y : ℝ) :
  x * y = 0 → x = 0 ∨ y = 0 := by sorry

theorem exists_even_prime : 
  ∃ n : ℕ, IsPrime n ∧ ¬IsOdd n := by sorry

end sum_of_odds_is_even_product_zero_implies_factor_zero_exists_even_prime_l2707_270709


namespace inequality_solution_set_l2707_270711

-- Define the inequality
def inequality (x : ℝ) : Prop := (x - 2) / (x + 1) ≤ 0

-- Define the solution set
def solution_set : Set ℝ := {x | x > -1 ∧ x ≤ 2}

-- Theorem statement
theorem inequality_solution_set :
  {x : ℝ | inequality x ∧ x + 1 ≠ 0} = solution_set := by sorry

end inequality_solution_set_l2707_270711


namespace M_subset_N_l2707_270730

-- Define set M
def M : Set ℝ := {x | ∃ k : ℤ, x = k / 2 + 1 / 4}

-- Define set N
def N : Set ℝ := {x | ∃ k : ℤ, x = k / 4 + 1 / 2}

-- Theorem statement
theorem M_subset_N : M ⊆ N := by
  sorry

end M_subset_N_l2707_270730


namespace scientific_notation_102200_l2707_270741

theorem scientific_notation_102200 :
  ∃ (a : ℝ) (n : ℤ), 1 ≤ a ∧ a < 10 ∧ 102200 = a * (10 : ℝ) ^ n ∧ a = 1.022 ∧ n = 5 := by
  sorry

end scientific_notation_102200_l2707_270741


namespace factorization_equality_l2707_270783

theorem factorization_equality (x a : ℝ) : 4*x - x*a^2 = x*(2-a)*(2+a) := by
  sorry

end factorization_equality_l2707_270783


namespace equilateral_triangle_area_increase_l2707_270777

/-- The area increase of an equilateral triangle -/
theorem equilateral_triangle_area_increase :
  ∀ (s : ℝ),
  s > 0 →
  s^2 * Real.sqrt 3 / 4 = 100 * Real.sqrt 3 →
  let new_s := s + 3
  let new_area := new_s^2 * Real.sqrt 3 / 4
  let initial_area := 100 * Real.sqrt 3
  new_area - initial_area = 32.25 * Real.sqrt 3 := by
  sorry

end equilateral_triangle_area_increase_l2707_270777


namespace radical_product_simplification_l2707_270788

theorem radical_product_simplification (x : ℝ) (h : x ≥ 0) :
  Real.sqrt (48 * x) * Real.sqrt (3 * x) * Real.sqrt (50 * x) = 60 * x * Real.sqrt x :=
by sorry

end radical_product_simplification_l2707_270788


namespace closest_fraction_is_one_fourth_l2707_270739

def total_medals : ℕ := 150
def won_medals : ℕ := 38

theorem closest_fraction_is_one_fourth :
  let fraction := won_medals / total_medals
  ∀ x ∈ ({1/3, 1/5, 1/6, 1/7} : Set ℚ),
    |fraction - (1/4 : ℚ)| ≤ |fraction - x| :=
by
  sorry

end closest_fraction_is_one_fourth_l2707_270739


namespace equation_solution_l2707_270725

theorem equation_solution : ∃ x : ℤ, 45 - (x - (37 - (15 - 17))) = 56 ∧ x = 28 := by
  sorry

end equation_solution_l2707_270725


namespace max_coins_distribution_l2707_270792

theorem max_coins_distribution (n : ℕ) (h1 : n < 150) 
  (h2 : ∃ k : ℕ, n = 8 * k + 4) : n ≤ 148 := by
  sorry

end max_coins_distribution_l2707_270792


namespace sector_central_angle_l2707_270776

/-- Given a circle sector with radius 10 cm and perimeter 45 cm, 
    the central angle of the sector is 2.5 radians. -/
theorem sector_central_angle (r : ℝ) (p : ℝ) (h1 : r = 10) (h2 : p = 45) :
  (p - 2 * r) / r = 2.5 := by
  sorry

end sector_central_angle_l2707_270776


namespace sine_of_inverse_sum_l2707_270702

theorem sine_of_inverse_sum : 
  Real.sin (Real.arcsin (4/5) + Real.arctan (1/2) + Real.arccos (3/5)) = 41 * Real.sqrt 5 / 125 := by
  sorry

end sine_of_inverse_sum_l2707_270702


namespace hyperbola_equation_l2707_270764

-- Define the hyperbola
def Hyperbola (a b : ℝ) : (ℝ × ℝ) → Prop :=
  λ (x, y) ↦ y^2 / a^2 - x^2 / b^2 = 1

-- Theorem statement
theorem hyperbola_equation (a : ℝ) (h1 : a = 2 * Real.sqrt 5) :
  ∃ b : ℝ, Hyperbola (a^2) (b^2) (2, -5) ∧ b^2 = 16 :=
by sorry

end hyperbola_equation_l2707_270764


namespace speed_conversion_l2707_270767

-- Define the conversion factor
def meters_per_second_to_kmph : ℝ := 3.6

-- Define the given speed in meters per second
def speed_in_mps : ℝ := 16.668

-- State the theorem
theorem speed_conversion :
  speed_in_mps * meters_per_second_to_kmph = 60.0048 := by
  sorry

end speed_conversion_l2707_270767


namespace table_pattern_l2707_270738

/-- Represents the number at position (row, column) in the table -/
def tableEntry (row : ℕ) (column : ℕ) : ℕ := sorry

/-- The first number of each row is equal to the row number -/
axiom first_number (n : ℕ) : tableEntry (n + 1) 1 = n + 1

/-- Each row forms an arithmetic sequence with common difference 1 -/
axiom arithmetic_sequence (n m : ℕ) : 
  tableEntry (n + 1) (m + 1) = tableEntry (n + 1) m + 1

/-- The number at the intersection of the (n+1)th row and the mth column is m + n -/
theorem table_pattern (n m : ℕ) : tableEntry (n + 1) m = m + n := by
  sorry

end table_pattern_l2707_270738


namespace prime_sequence_l2707_270740

theorem prime_sequence (n : ℕ) (h1 : n ≥ 2) :
  (∀ k : ℕ, 0 ≤ k ∧ k ≤ Real.sqrt (n / 3) → Nat.Prime (k^2 + k + n)) →
  (∀ k : ℕ, 0 ≤ k ∧ k ≤ n - 2 → Nat.Prime (k^2 + k + n)) :=
by sorry

end prime_sequence_l2707_270740


namespace coefficient_a5_equals_6_l2707_270759

theorem coefficient_a5_equals_6 
  (a a₁ a₂ a₃ a₄ a₅ a₆ : ℝ) :
  (∀ x : ℝ, x^6 = a + a₁*(x-1) + a₂*(x-1)^2 + a₃*(x-1)^3 + a₄*(x-1)^4 + a₅*(x-1)^5 + a₆*(x-1)^6) →
  a₅ = 6 := by
sorry

end coefficient_a5_equals_6_l2707_270759


namespace sophie_gave_one_box_to_mom_l2707_270713

/-- Represents the number of donuts in a box --/
def donuts_per_box : ℕ := 12

/-- Represents the number of boxes Sophie bought --/
def boxes_bought : ℕ := 4

/-- Represents the number of donuts Sophie gave to her sister --/
def donuts_to_sister : ℕ := 6

/-- Represents the number of donuts Sophie had left for herself --/
def donuts_left_for_sophie : ℕ := 30

/-- Calculates the number of boxes Sophie gave to her mom --/
def boxes_to_mom : ℕ :=
  (boxes_bought * donuts_per_box - donuts_to_sister - donuts_left_for_sophie) / donuts_per_box

theorem sophie_gave_one_box_to_mom :
  boxes_to_mom = 1 :=
sorry

end sophie_gave_one_box_to_mom_l2707_270713


namespace dot_product_example_l2707_270766

theorem dot_product_example : 
  let v1 : Fin 2 → ℝ := ![3, -2]
  let v2 : Fin 2 → ℝ := ![-5, 7]
  Finset.sum (Finset.range 2) (λ i => v1 i * v2 i) = -29 := by
  sorry

end dot_product_example_l2707_270766


namespace hyperbola_branch_from_condition_l2707_270772

/-- The set of points forming one branch of a hyperbola -/
def HyperbolaBranch : Set (ℝ × ℝ) :=
  {P | ∃ (x y : ℝ), P = (x, y) ∧ 
    Real.sqrt ((x + 3)^2 + y^2) - Real.sqrt ((x - 3)^2 + y^2) = 4}

/-- Theorem stating that the given condition forms one branch of a hyperbola -/
theorem hyperbola_branch_from_condition :
  ∃ (F₁ F₂ : ℝ × ℝ), F₁ = (-3, 0) ∧ F₂ = (3, 0) ∧
  HyperbolaBranch = {P | |P.1 - F₁.1| - |P.1 - F₂.1| = 4} :=
by
  sorry


end hyperbola_branch_from_condition_l2707_270772


namespace tenth_term_of_specific_geometric_sequence_l2707_270723

def geometric_sequence (a₁ : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a₁ * r ^ (n - 1)

theorem tenth_term_of_specific_geometric_sequence :
  geometric_sequence 9 (1/3) 10 = 1/2187 := by
  sorry

end tenth_term_of_specific_geometric_sequence_l2707_270723


namespace unknown_number_proof_l2707_270786

theorem unknown_number_proof (x : ℝ) :
  (x + 48 / 69) * 69 = 1980 → x = 28 := by
  sorry

end unknown_number_proof_l2707_270786


namespace max_both_writers_and_editors_l2707_270712

/-- Conference attendees --/
structure Conference where
  total : ℕ
  writers : ℕ
  editors : ℕ
  both : ℕ
  neither : ℕ

/-- Conference conditions --/
def ConferenceConditions (c : Conference) : Prop :=
  c.total = 100 ∧
  c.writers = 40 ∧
  c.editors > 38 ∧
  c.neither = 2 * c.both ∧
  c.writers + c.editors - c.both + c.neither = c.total

/-- Theorem: The maximum number of people who are both writers and editors is 21 --/
theorem max_both_writers_and_editors (c : Conference) 
  (h : ConferenceConditions c) : c.both ≤ 21 := by
  sorry

end max_both_writers_and_editors_l2707_270712


namespace stating_regular_polygon_triangle_counts_l2707_270798

variable (n : ℕ)

/-- A regular polygon with 2n sides -/
structure RegularPolygon (n : ℕ) where
  vertices : Fin (2*n) → ℝ × ℝ

/-- The number of right-angled triangles in a regular polygon with 2n sides -/
def num_right_triangles (n : ℕ) : ℕ := 2*n*(n-1)

/-- The number of acute-angled triangles in a regular polygon with 2n sides -/
def num_acute_triangles (n : ℕ) : ℕ := n*(n-1)*(n-2)/3

/-- 
Theorem stating the number of right-angled and acute-angled triangles 
in a regular polygon with 2n sides
-/
theorem regular_polygon_triangle_counts (n : ℕ) (p : RegularPolygon n) :
  (num_right_triangles n = 2*n*(n-1)) ∧ 
  (num_acute_triangles n = n*(n-1)*(n-2)/3) := by
  sorry


end stating_regular_polygon_triangle_counts_l2707_270798


namespace recipe_change_l2707_270704

/-- Represents the recipe for the apple-grape drink -/
structure Recipe where
  apple_proportion : ℚ  -- Proportion of an apple juice container used per can
  grape_proportion : ℚ  -- Proportion of a grape juice container used per can

/-- The total volume of juice per can -/
def total_volume (r : Recipe) : ℚ :=
  r.apple_proportion + r.grape_proportion

theorem recipe_change (old_recipe new_recipe : Recipe) :
  old_recipe.apple_proportion = 1/6 →
  old_recipe.grape_proportion = 1/10 →
  new_recipe.apple_proportion = 1/5 →
  total_volume old_recipe = total_volume new_recipe →
  new_recipe.grape_proportion = 1/15 := by
  sorry

end recipe_change_l2707_270704


namespace circle_area_with_radius_3_l2707_270736

theorem circle_area_with_radius_3 :
  ∀ (π : ℝ), π > 0 →
  let r : ℝ := 3
  let area := π * r^2
  area = 9 * π :=
by sorry

end circle_area_with_radius_3_l2707_270736


namespace unique_solution_modulo_l2707_270710

theorem unique_solution_modulo : ∃! n : ℤ, 0 ≤ n ∧ n ≤ 14 ∧ n ≡ 16427 [ZMOD 15] := by
  sorry

end unique_solution_modulo_l2707_270710


namespace sum_of_algebra_values_l2707_270771

-- Define the function that assigns numeric values to letters based on their position
def letterValue (position : ℕ) : ℤ :=
  match position % 8 with
  | 1 => 1
  | 2 => 2
  | 3 => 3
  | 4 => 1
  | 5 => 0
  | 6 => -1
  | 7 => -2
  | 0 => -3
  | _ => 0  -- This case should never occur due to the modulo operation

-- Define the positions of letters in "ALGEBRA"
def algebraPositions : List ℕ := [1, 12, 7, 5, 2, 18, 1]

-- Theorem statement
theorem sum_of_algebra_values :
  (algebraPositions.map letterValue).sum = 5 := by
sorry

end sum_of_algebra_values_l2707_270771


namespace arithmetic_sequence_length_l2707_270747

def arithmetic_sequence (a₁ d n : ℕ) := 
  (fun i => a₁ + (i - 1) * d)

theorem arithmetic_sequence_length : 
  ∃ n : ℕ, n > 0 ∧ arithmetic_sequence 15 4 n (n) = 95 ∧ n = 21 := by
sorry

end arithmetic_sequence_length_l2707_270747


namespace rectangle_squares_l2707_270757

theorem rectangle_squares (N : ℕ) : 
  (∃ x y : ℕ, N = x * (x + 9) ∧ N = y * (y + 6)) → N = 112 := by
sorry

end rectangle_squares_l2707_270757


namespace circle_center_line_segment_length_l2707_270745

/-- Circle C with equation x^2 + y^2 - 2x - 2y + 1 = 0 -/
def CircleC (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x - 2*y + 1 = 0

/-- Line l with equation x - y = 0 -/
def LineL (x y : ℝ) : Prop :=
  x - y = 0

/-- The center of circle C is at (1, 1) -/
theorem circle_center : ∃ (x y : ℝ), CircleC x y ∧ x = 1 ∧ y = 1 :=
sorry

/-- The length of line segment AB, where A and B are intersection points of circle C and line l, is 2√2 -/
theorem line_segment_length :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    CircleC x₁ y₁ ∧ CircleC x₂ y₂ ∧
    LineL x₁ y₁ ∧ LineL x₂ y₂ ∧
    x₁ ≠ x₂ ∧
    ((x₁ - x₂)^2 + (y₁ - y₂)^2) = 8 :=
sorry

end circle_center_line_segment_length_l2707_270745


namespace arlo_books_count_l2707_270774

theorem arlo_books_count (total_stationery : ℕ) (book_ratio pen_ratio : ℕ) (h1 : total_stationery = 400) (h2 : book_ratio = 7) (h3 : pen_ratio = 3) : 
  (book_ratio * total_stationery) / (book_ratio + pen_ratio) = 280 := by
sorry

end arlo_books_count_l2707_270774


namespace arithmetic_equalities_l2707_270756

theorem arithmetic_equalities : 
  (Real.sqrt 27 + 3 * Real.sqrt (1/3) - Real.sqrt 24 * Real.sqrt 2 = 0) ∧
  ((Real.sqrt 5 - 2) * (2 + Real.sqrt 5) - (Real.sqrt 3 - 1)^2 = -3 + 2 * Real.sqrt 3) := by
  sorry

end arithmetic_equalities_l2707_270756


namespace lucas_easter_eggs_problem_l2707_270769

theorem lucas_easter_eggs_problem (blue_eggs green_eggs min_eggs : ℕ) 
  (h1 : blue_eggs = 30)
  (h2 : green_eggs = 42)
  (h3 : min_eggs = 5) :
  ∃ (basket_eggs : ℕ), 
    basket_eggs ≥ min_eggs ∧ 
    basket_eggs ∣ blue_eggs ∧ 
    basket_eggs ∣ green_eggs ∧
    ∀ (n : ℕ), n > basket_eggs → ¬(n ∣ blue_eggs ∧ n ∣ green_eggs) :=
by sorry

end lucas_easter_eggs_problem_l2707_270769


namespace diamond_with_zero_not_always_double_l2707_270734

def diamond (x y : ℝ) : ℝ := x + y - |x - y|

theorem diamond_with_zero_not_always_double :
  ¬ (∀ x : ℝ, diamond x 0 = 2 * x) := by
  sorry

end diamond_with_zero_not_always_double_l2707_270734


namespace last_digit_sum_l2707_270773

theorem last_digit_sum (n : ℕ) : 
  (2^2 + 20^20 + 200^200 + 2006^2006) % 10 = 0 := by
  sorry

end last_digit_sum_l2707_270773


namespace binary_110011_equals_51_l2707_270794

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_110011_equals_51 :
  binary_to_decimal [true, true, false, false, true, true] = 51 := by
  sorry

end binary_110011_equals_51_l2707_270794


namespace abigail_expenses_l2707_270763

def initial_amount : ℝ := 200

def food_expense_percentage : ℝ := 0.60

def phone_bill_percentage : ℝ := 0.25

def entertainment_expense : ℝ := 20

def remaining_amount (initial : ℝ) (food_percent : ℝ) (phone_percent : ℝ) (entertainment : ℝ) : ℝ :=
  let after_food := initial * (1 - food_percent)
  let after_phone := after_food * (1 - phone_percent)
  after_phone - entertainment

theorem abigail_expenses :
  remaining_amount initial_amount food_expense_percentage phone_bill_percentage entertainment_expense = 40 := by
  sorry

end abigail_expenses_l2707_270763


namespace sum_of_five_integers_l2707_270717

theorem sum_of_five_integers (C y M A : ℕ) : 
  C > 0 → y > 0 → M > 0 → A > 0 →
  C ≠ y → C ≠ M → C ≠ A → y ≠ M → y ≠ A → M ≠ A →
  C + y + M + M + A = 11 →
  M = 1 := by
sorry

end sum_of_five_integers_l2707_270717


namespace triangle_exists_from_altitudes_l2707_270746

theorem triangle_exists_from_altitudes (h₁ h₂ h₃ : ℝ) 
  (h₁_pos : h₁ > 0) (h₂_pos : h₂ > 0) (h₃_pos : h₃ > 0) :
  ∃ (a b c : ℝ), 
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    (a + b > c) ∧ (b + c > a) ∧ (c + a > b) ∧
    h₁ = (2 * (a * b * c) / (a * (a + b + c))) ∧
    h₂ = (2 * (a * b * c) / (b * (a + b + c))) ∧
    h₃ = (2 * (a * b * c) / (c * (a + b + c))) :=
by sorry

end triangle_exists_from_altitudes_l2707_270746


namespace matrix_power_four_l2707_270761

def A : Matrix (Fin 2) (Fin 2) ℤ := !![1, -2; 2, 1]

theorem matrix_power_four :
  A ^ 4 = !![(-7 : ℤ), 24; -24, 7] := by sorry

end matrix_power_four_l2707_270761


namespace ways_to_top_center_l2707_270791

/-- Number of ways to reach the center square of the topmost row in a grid -/
def numWaysToTopCenter (n : ℕ) : ℕ :=
  2^(n-1)

/-- Theorem: The number of ways to reach the center square of the topmost row
    in a rectangular grid with n rows and 3 columns, starting from the bottom
    left corner and moving either one square right or simultaneously one square
    left and one square up at each step, is equal to 2^(n-1). -/
theorem ways_to_top_center (n : ℕ) (h : n > 0) :
  numWaysToTopCenter n = 2^(n-1) := by
  sorry

end ways_to_top_center_l2707_270791


namespace ceiling_minus_x_bounds_l2707_270714

theorem ceiling_minus_x_bounds (x : ℝ) : 
  ⌈x⌉ - ⌊x⌋ = 1 → 0 < ⌈x⌉ - x ∧ ⌈x⌉ - x ≤ 1 := by sorry

end ceiling_minus_x_bounds_l2707_270714


namespace clara_total_earnings_l2707_270758

/-- Represents a staff member at the cake shop -/
structure Staff :=
  (name : String)
  (hourlyRate : ℝ)
  (holidayBonus : ℝ)

/-- Calculates the total earnings for a staff member -/
def totalEarnings (s : Staff) (hoursWorked : ℝ) : ℝ :=
  s.hourlyRate * hoursWorked + s.holidayBonus

/-- Theorem: Clara's total earnings for the 2-month period -/
theorem clara_total_earnings :
  let clara : Staff := { name := "Clara", hourlyRate := 13, holidayBonus := 60 }
  let standardHours : ℝ := 20 * 8  -- 20 hours per week for 8 weeks
  let vacationHours : ℝ := 20 * 1.5  -- 10 days vacation (1.5 weeks)
  let claraHours : ℝ := standardHours - vacationHours
  totalEarnings clara claraHours = 1750 := by
  sorry

end clara_total_earnings_l2707_270758


namespace rainfall_difference_l2707_270707

/-- Calculates the difference between the average rainfall and the actual rainfall for the first three days of May. -/
theorem rainfall_difference (day1 day2 day3 avg : ℝ) 
  (h1 : day1 = 26)
  (h2 : day2 = 34)
  (h3 : day3 = day2 - 12)
  (h4 : avg = 140) :
  avg - (day1 + day2 + day3) = 58 := by
sorry

end rainfall_difference_l2707_270707


namespace area_ratio_circle_ellipse_l2707_270727

/-- The ratio of the area between a circle and an ellipse to the area of the circle -/
theorem area_ratio_circle_ellipse :
  let circle_diameter : ℝ := 4
  let ellipse_major_axis : ℝ := 8
  let ellipse_minor_axis : ℝ := 6
  let circle_area := π * (circle_diameter / 2)^2
  let ellipse_area := π * (ellipse_major_axis / 2) * (ellipse_minor_axis / 2)
  (ellipse_area - circle_area) / circle_area = 2 := by
  sorry

end area_ratio_circle_ellipse_l2707_270727


namespace game_result_l2707_270705

def f (n : ℕ) : ℕ :=
  if n % 2 = 0 ∧ n % 3 = 0 then 7
  else if n % 2 = 0 then 3
  else if Nat.Prime n then 5
  else 0

def allie_rolls : List ℕ := [2, 3, 4, 5, 6]
def betty_rolls : List ℕ := [6, 3, 4, 2, 1]

theorem game_result :
  (List.sum (List.map f allie_rolls)) * (List.sum (List.map f betty_rolls)) = 500 := by
  sorry

end game_result_l2707_270705


namespace arithmetic_sequence_sum_l2707_270703

-- Define an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- State the theorem
theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  is_arithmetic_sequence a → a 2 + a 3 + a 10 + a 11 = 48 → a 6 + a 7 = 24 :=
by
  sorry

end arithmetic_sequence_sum_l2707_270703


namespace inequality_proof_l2707_270789

theorem inequality_proof (x y z : ℝ) (hx : 0 < x ∧ x < π/2) (hy : 0 < y ∧ y < π/2) (hz : 0 < z ∧ z < π/2) :
  (x * Real.cos x + y * Real.cos y + z * Real.cos z) / (x + y + z) ≤ (Real.cos x + Real.cos y + Real.cos z) / 3 := by
  sorry

end inequality_proof_l2707_270789


namespace sara_bought_movie_cost_l2707_270728

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

end sara_bought_movie_cost_l2707_270728


namespace apple_probabilities_l2707_270799

structure ApplePlot where
  name : String
  first_grade_ratio : ℚ
  production_ratio : ℕ

def plot_a : ApplePlot := ⟨"A", 3/4, 2⟩
def plot_b : ApplePlot := ⟨"B", 3/5, 5⟩
def plot_c : ApplePlot := ⟨"C", 4/5, 3⟩

def total_production : ℕ := plot_a.production_ratio + plot_b.production_ratio + plot_c.production_ratio

theorem apple_probabilities :
  (plot_a.production_ratio : ℚ) / total_production = 1/5 ∧
  (plot_a.production_ratio * plot_a.first_grade_ratio +
   plot_b.production_ratio * plot_b.first_grade_ratio +
   plot_c.production_ratio * plot_c.first_grade_ratio) / total_production = 69/100 ∧
  (plot_a.production_ratio * plot_a.first_grade_ratio) /
  (plot_a.production_ratio * plot_a.first_grade_ratio +
   plot_b.production_ratio * plot_b.first_grade_ratio +
   plot_c.production_ratio * plot_c.first_grade_ratio) = 5/23 := by
  sorry


end apple_probabilities_l2707_270799


namespace remainder_zero_l2707_270722

theorem remainder_zero (k α : ℕ+) (h : 10 * k.val - α.val > 0) :
  (8^(10 * k.val + α.val) + 6^(10 * k.val - α.val) - 
   7^(10 * k.val - α.val) - 2^(10 * k.val + α.val)) % 11 = 0 := by
  sorry

end remainder_zero_l2707_270722


namespace second_number_is_40_l2707_270720

theorem second_number_is_40 (a b c : ℕ+) 
  (sum_eq : a + b + c = 120)
  (ratio_ab : (a : ℚ) / (b : ℚ) = 3 / 4)
  (ratio_bc : (b : ℚ) / (c : ℚ) = 7 / 9) :
  b = 40 := by
  sorry

end second_number_is_40_l2707_270720


namespace son_age_proof_l2707_270731

theorem son_age_proof (son_age father_age : ℕ) : 
  father_age = son_age + 20 →
  father_age + 2 = 2 * (son_age + 2) →
  son_age = 18 := by
sorry

end son_age_proof_l2707_270731


namespace sqrt_equation_solution_l2707_270780

theorem sqrt_equation_solution :
  ∀ a b : ℕ+,
    a < b →
    (Real.sqrt (1 + Real.sqrt (25 + 20 * Real.sqrt 3)) = Real.sqrt a + Real.sqrt b) ↔
    (a = 1 ∧ b = 3) := by
  sorry

end sqrt_equation_solution_l2707_270780


namespace min_four_dollar_frisbees_min_four_dollar_frisbees_proof_l2707_270748

/-- Given 60 frisbees sold at either $3 or $4 each, with total receipts of $204,
    the minimum number of $4 frisbees sold is 24. -/
theorem min_four_dollar_frisbees : ℕ :=
  let total_frisbees : ℕ := 60
  let total_receipts : ℕ := 204
  let price_low : ℕ := 3
  let price_high : ℕ := 4
  24

/-- Proof that the minimum number of $4 frisbees sold is indeed 24. -/
theorem min_four_dollar_frisbees_proof :
  let total_frisbees : ℕ := 60
  let total_receipts : ℕ := 204
  let price_low : ℕ := 3
  let price_high : ℕ := 4
  let min_high_price_frisbees := min_four_dollar_frisbees
  (∃ (low_price_frisbees : ℕ),
    low_price_frisbees + min_high_price_frisbees = total_frisbees ∧
    low_price_frisbees * price_low + min_high_price_frisbees * price_high = total_receipts) ∧
  (∀ (high_price_frisbees : ℕ),
    high_price_frisbees < min_high_price_frisbees →
    ¬∃ (low_price_frisbees : ℕ),
      low_price_frisbees + high_price_frisbees = total_frisbees ∧
      low_price_frisbees * price_low + high_price_frisbees * price_high = total_receipts) :=
by
  sorry

#check min_four_dollar_frisbees
#check min_four_dollar_frisbees_proof

end min_four_dollar_frisbees_min_four_dollar_frisbees_proof_l2707_270748


namespace random_walk_properties_l2707_270765

/-- Represents a random walk on a line. -/
structure RandomWalk where
  a : ℕ  -- number of steps to the right
  b : ℕ  -- number of steps to the left
  h : a > b

/-- The maximum possible range of a random walk. -/
def max_range (w : RandomWalk) : ℕ := w.a

/-- The minimum possible range of a random walk. -/
def min_range (w : RandomWalk) : ℕ := w.a - w.b

/-- The number of sequences that achieve the maximum range. -/
def max_range_sequences (w : RandomWalk) : ℕ := w.b + 1

/-- Theorem stating the properties of the random walk. -/
theorem random_walk_properties (w : RandomWalk) :
  (max_range w = w.a) ∧
  (min_range w = w.a - w.b) ∧
  (max_range_sequences w = w.b + 1) := by
  sorry


end random_walk_properties_l2707_270765


namespace min_socks_for_pair_l2707_270760

/-- Represents the number of socks of each color in the drawer -/
def socksPerColor : ℕ := 24

/-- Represents the total number of colors of socks in the drawer -/
def numColors : ℕ := 2

/-- Represents the minimum number of socks that must be picked to guarantee a pair of the same color -/
def minSocksToPick : ℕ := 3

/-- Theorem stating that picking 3 socks guarantees at least one pair of the same color,
    and this is the minimum number required -/
theorem min_socks_for_pair :
  (∀ (picked : Finset ℕ), picked.card = minSocksToPick → 
    ∃ (color : Fin numColors), (picked.filter (λ sock => sock % numColors = color)).card ≥ 2) ∧
  (∀ (n : ℕ), n < minSocksToPick → 
    ∃ (picked : Finset ℕ), picked.card = n ∧ 
      ∀ (color : Fin numColors), (picked.filter (λ sock => sock % numColors = color)).card < 2) :=
sorry

end min_socks_for_pair_l2707_270760


namespace inequality_cube_l2707_270718

theorem inequality_cube (a b c : ℝ) (h1 : a > b) (h2 : b > c) (h3 : c > 0) :
  (a - c)^3 > (b - c)^3 := by
  sorry

end inequality_cube_l2707_270718


namespace quadratic_root_transform_l2707_270775

/-- Given a quadratic equation ax^2 + bx + c = 0 with roots x₁ and x₂,
    this theorem proves the equations with transformed roots. -/
theorem quadratic_root_transform (a b c : ℝ) (x₁ x₂ : ℝ) 
  (hroot : a * x₁^2 + b * x₁ + c = 0 ∧ a * x₂^2 + b * x₂ + c = 0) :
  (∃ y₁ y₂ : ℝ, y₁ = 1/x₁^3 ∧ y₂ = 1/x₂^3 ∧ 
    c^3 * y₁^2 + (b^3 - 3*a*b*c) * y₁ + a^3 = 0 ∧
    c^3 * y₂^2 + (b^3 - 3*a*b*c) * y₂ + a^3 = 0) ∧
  (∃ z₁ z₂ : ℝ, z₁ = (x₁ - x₂)^2 ∧ z₂ = (x₁ + x₂)^2 ∧
    a^4 * z₁^2 + 2*a^2*(2*a*c - b^2) * z₁ + b^2*(b^2 - 4*a*c) = 0 ∧
    a^4 * z₂^2 + 2*a^2*(2*a*c - b^2) * z₂ + b^2*(b^2 - 4*a*c) = 0) :=
by sorry

end quadratic_root_transform_l2707_270775


namespace final_cell_count_l2707_270781

/-- Calculates the number of cells after a given number of days, 
    where cells double every 3 days starting from an initial population. -/
def cell_count (initial_cells : ℕ) (days : ℕ) : ℕ :=
  initial_cells * 2^(days / 3)

/-- Theorem stating that given 4 initial cells and 9 days, 
    the final cell count is 32. -/
theorem final_cell_count : cell_count 4 9 = 32 := by
  sorry

end final_cell_count_l2707_270781


namespace largest_common_divisor_l2707_270755

def is_odd (n : ℕ) : Prop := ∃ k, n = 2*k + 1

def product_function (n : ℕ) : ℕ := (n+2)*(n+4)*(n+6)*(n+8)*(n+10)

theorem largest_common_divisor (n : ℕ) (h : is_odd n) :
  (∀ m : ℕ, m > 8 → ¬(m ∣ product_function n)) ∧
  (8 ∣ product_function n) :=
sorry

end largest_common_divisor_l2707_270755


namespace delta_value_l2707_270790

theorem delta_value (Δ : ℤ) (h : 4 * (-3) = Δ + 3) : Δ = -15 := by
  sorry

end delta_value_l2707_270790


namespace original_combined_cost_l2707_270770

/-- Represents the original prices of items --/
structure OriginalPrices where
  dress : ℝ
  shoes : ℝ
  handbag : ℝ
  necklace : ℝ

/-- Represents the discounted prices of items --/
structure DiscountedPrices where
  dress : ℝ
  shoes : ℝ
  handbag : ℝ
  necklace : ℝ

/-- Calculates the total savings before the coupon --/
def totalSavings (original : OriginalPrices) (discounted : DiscountedPrices) : ℝ :=
  (original.dress - discounted.dress) +
  (original.shoes - discounted.shoes) +
  (original.handbag - discounted.handbag) +
  (original.necklace - discounted.necklace)

/-- Calculates the total discounted price before the coupon --/
def totalDiscountedPrice (discounted : DiscountedPrices) : ℝ :=
  discounted.dress + discounted.shoes + discounted.handbag + discounted.necklace

/-- The main theorem --/
theorem original_combined_cost (original : OriginalPrices) (discounted : DiscountedPrices)
  (h1 : discounted.dress = original.dress / 2 - 10)
  (h2 : discounted.shoes = original.shoes * 0.85)
  (h3 : discounted.handbag = original.handbag - 30)
  (h4 : discounted.necklace = original.necklace)
  (h5 : discounted.necklace ≤ original.dress)
  (h6 : totalSavings original discounted = 120)
  (h7 : totalDiscountedPrice discounted * 0.9 = totalDiscountedPrice discounted - 120) :
  original.dress + original.shoes + original.handbag + original.necklace = 1200 := by
  sorry


end original_combined_cost_l2707_270770


namespace root_square_relation_l2707_270708

theorem root_square_relation (b c : ℝ) : 
  (∃ r s : ℝ, r^2 + s^2 = -b ∧ r^2 * s^2 = c ∧ 
   r + s = 5 ∧ r * s = 2) → 
  c / b = -4 / 21 :=
by sorry

end root_square_relation_l2707_270708


namespace student_group_equations_l2707_270706

/-- Given a number of students and groups, prove that the system of equations
    represents the given conditions. -/
theorem student_group_equations (x y : ℕ) : 
  (5 * y = x - 3 ∧ 6 * y = x + 3) ↔ 
  (x = 5 * y + 3 ∧ x = 6 * y - 3) := by
sorry

end student_group_equations_l2707_270706


namespace profit_ratio_l2707_270762

def investment_p : ℕ := 500000
def investment_q : ℕ := 1000000

theorem profit_ratio (p q : ℕ) (h : p = investment_p ∧ q = investment_q) :
  (p : ℚ) / (p + q : ℚ) = 1 / 3 ∧ (q : ℚ) / (p + q : ℚ) = 2 / 3 := by
  sorry

end profit_ratio_l2707_270762


namespace john_painting_time_l2707_270701

theorem john_painting_time (sally_time john_time combined_time : ℝ) : 
  sally_time = 4 →
  combined_time = 2.4 →
  1 / sally_time + 1 / john_time = 1 / combined_time →
  john_time = 6 := by
sorry

end john_painting_time_l2707_270701


namespace expression_equality_l2707_270721

theorem expression_equality : 150 * (150 - 8) - (150 * 150 - 8) = -1192 := by
  sorry

end expression_equality_l2707_270721


namespace missing_element_is_loop_l2707_270779

-- Define the basic elements of a flowchart
inductive FlowchartElement
| Input
| Output
| Condition
| Loop

-- Define the program structures
inductive ProgramStructure
| Sequence
| Condition
| Loop

-- Define the known basic elements
def known_elements : List FlowchartElement := [FlowchartElement.Input, FlowchartElement.Output, FlowchartElement.Condition]

-- Define the program structures
def program_structures : List ProgramStructure := [ProgramStructure.Sequence, ProgramStructure.Condition, ProgramStructure.Loop]

-- Theorem: The missing basic element of a flowchart is Loop
theorem missing_element_is_loop : 
  ∃ (e : FlowchartElement), e ∉ known_elements ∧ e = FlowchartElement.Loop :=
sorry

end missing_element_is_loop_l2707_270779


namespace power_product_three_six_l2707_270796

theorem power_product_three_six : (3^5 * 6^5 : ℕ) = 34012224 := by
  sorry

end power_product_three_six_l2707_270796


namespace other_number_proof_l2707_270719

theorem other_number_proof (a b : ℕ+) (h1 : Nat.lcm a b = 2310) (h2 : Nat.gcd a b = 26) (h3 : a = 210) : b = 286 := by
  sorry

end other_number_proof_l2707_270719


namespace sum_f_91_and_neg_91_l2707_270749

/-- A polynomial function of degree 6 -/
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^6 + b * x^4 - c * x^2 + 3

/-- Theorem: Given f(x) = ax^6 + bx^4 - cx^2 + 3 and f(91) = 1, prove f(91) + f(-91) = 2 -/
theorem sum_f_91_and_neg_91 (a b c : ℝ) (h : f a b c 91 = 1) : f a b c 91 + f a b c (-91) = 2 := by
  sorry

#check sum_f_91_and_neg_91

end sum_f_91_and_neg_91_l2707_270749


namespace debt_ratio_proof_l2707_270752

/-- Proves that the ratio of Aryan's debt to Kyro's debt is 2:1 given the problem conditions --/
theorem debt_ratio_proof (aryan_debt kyro_debt : ℝ) 
  (h1 : aryan_debt = 1200)
  (h2 : 0.6 * aryan_debt + 0.8 * kyro_debt + 300 = 1500) :
  aryan_debt / kyro_debt = 2 := by
  sorry


end debt_ratio_proof_l2707_270752


namespace twin_primes_divisibility_l2707_270793

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

theorem twin_primes_divisibility (a : ℤ) 
  (h1 : is_prime (a - 1).natAbs) 
  (h2 : is_prime (a + 1).natAbs) 
  (h3 : (a - 1).natAbs > 10) 
  (h4 : (a + 1).natAbs > 10) : 
  120 ∣ (a^3 - 4*a) :=
sorry

end twin_primes_divisibility_l2707_270793


namespace count_special_numbers_eq_266_l2707_270754

/-- The number of natural numbers from 1 to 1992 that are multiples of 3, but not multiples of 2 or 5 -/
def count_special_numbers : ℕ := 
  (Nat.floor (1992 / 3) : ℕ) - 
  (Nat.floor (1992 / 6) : ℕ) - 
  (Nat.floor (1992 / 15) : ℕ) + 
  (Nat.floor (1992 / 30) : ℕ)

theorem count_special_numbers_eq_266 : count_special_numbers = 266 := by
  sorry

end count_special_numbers_eq_266_l2707_270754


namespace water_container_percentage_l2707_270724

theorem water_container_percentage (initial_water : ℝ) (capacity : ℝ) (added_water : ℝ) :
  capacity = 40 →
  added_water = 14 →
  (initial_water + added_water) / capacity = 3/4 →
  initial_water / capacity = 2/5 :=
by sorry

end water_container_percentage_l2707_270724


namespace inequality_solution_l2707_270716

theorem inequality_solution (x : ℝ) : 
  1 / (x + 2) + 5 / (x + 4) ≥ 1 ↔ x ∈ Set.Icc (-4 : ℝ) (-3) ∪ Set.Icc (-2 : ℝ) 2 := by
  sorry

end inequality_solution_l2707_270716


namespace union_determines_m_l2707_270797

def A (m : ℝ) : Set ℝ := {1, 2, m}
def B : Set ℝ := {2, 3}

theorem union_determines_m (m : ℝ) (h : A m ∪ B = {1, 2, 3}) : m = 3 := by
  sorry

end union_determines_m_l2707_270797


namespace parabola_properties_l2707_270751

-- Define the parabola equation
def parabola_equation (x y : ℝ) : Prop :=
  4 * x^2 + 4 * x * y + y^2 - 10 * y - 15 = 0

-- Define the axis of symmetry
def axis_of_symmetry (x y : ℝ) : Prop :=
  2 * x + y - 1 = 0

-- Define the directrix
def directrix (x y : ℝ) : Prop :=
  x - 2 * y - 5 = 0

-- Define the tangent line
def tangent_line (y : ℝ) : Prop :=
  2 * y + 3 = 0

-- Theorem statement
theorem parabola_properties :
  ∀ (x y : ℝ),
    parabola_equation x y →
    (∃ (x₀ y₀ : ℝ), axis_of_symmetry x₀ y₀ ∧
                     directrix x₀ y₀ ∧
                     tangent_line y₀) :=
by sorry

end parabola_properties_l2707_270751


namespace arithmetic_sequence_general_term_l2707_270778

/-- An arithmetic sequence with its sum function and properties -/
structure ArithmeticSequence where
  /-- The general term of the sequence -/
  a : ℕ → ℝ
  /-- The sum of the first n terms -/
  S : ℕ → ℝ
  /-- The sum of the first 4 terms is 0 -/
  sum_4 : S 4 = 0
  /-- The 5th term is 5 -/
  term_5 : a 5 = 5
  /-- The sequence is arithmetic -/
  is_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)

/-- The general term of the arithmetic sequence is 2n - 5 -/
theorem arithmetic_sequence_general_term (seq : ArithmeticSequence) :
  ∀ n : ℕ, seq.a n = 2 * n - 5 := by
  sorry

end arithmetic_sequence_general_term_l2707_270778


namespace system_one_solution_system_two_solution_l2707_270787

-- System 1
theorem system_one_solution (x y : ℝ) : 
  x - y = 1 ∧ 2*x + y = 5 → x = 2 ∧ y = 1 := by sorry

-- System 2
theorem system_two_solution (x y : ℝ) : 
  x/2 - (y+1)/3 = 1 ∧ x + y = 1 → x = 2 ∧ y = -1 := by sorry

end system_one_solution_system_two_solution_l2707_270787


namespace inequality_proof_l2707_270784

theorem inequality_proof (a : ℝ) (h : a ≠ 2) :
  (1 : ℝ) / (a^2 - 4*a + 4) > 2 / (a^3 - 8) := by
sorry

end inequality_proof_l2707_270784


namespace rectangle_perimeter_l2707_270744

/-- A rectangle with integer dimensions satisfying the given condition has a perimeter of 26. -/
theorem rectangle_perimeter (a b : ℕ) : 
  a ≠ b →  -- not a square
  4 * (a + b) - a * b = 12 →  -- twice perimeter minus area equals 12
  2 * (a + b) = 26 := by  -- perimeter equals 26
  sorry

end rectangle_perimeter_l2707_270744


namespace bookstore_sales_ratio_l2707_270782

theorem bookstore_sales_ratio :
  -- Initial conditions
  let initial_inventory : ℕ := 743
  let saturday_instore : ℕ := 37
  let saturday_online : ℕ := 128
  let sunday_online_increase : ℕ := 34
  let shipment : ℕ := 160
  let final_inventory : ℕ := 502

  -- Define Sunday in-store sales
  let sunday_instore : ℕ := initial_inventory - final_inventory + shipment - 
    (saturday_instore + saturday_online + sunday_online_increase)

  -- Theorem statement
  (sunday_instore : ℚ) / (saturday_instore : ℚ) = 2 / 1 := by
  sorry

end bookstore_sales_ratio_l2707_270782


namespace xy_unique_values_l2707_270729

def X : Finset ℤ := {2, 3, 7}
def Y : Finset ℤ := {-31, -24, 4}

theorem xy_unique_values : 
  Finset.card (Finset.image (λ (p : ℤ × ℤ) => p.1 * p.2) (X.product Y)) = 9 := by
  sorry

end xy_unique_values_l2707_270729


namespace semi_circle_perimeter_l2707_270737

/-- The perimeter of a semi-circle with radius 6.4 cm is π * 6.4 + 12.8 -/
theorem semi_circle_perimeter :
  let r : ℝ := 6.4
  (2 * r * Real.pi / 2) + 2 * r = r * Real.pi + 2 * r := by
  sorry

end semi_circle_perimeter_l2707_270737


namespace sqrt_product_simplification_l2707_270768

theorem sqrt_product_simplification (x : ℝ) (h : x > 0) :
  Real.sqrt (100 * x) * Real.sqrt (3 * x) * Real.sqrt (18 * x) = 30 * x * Real.sqrt (6 * x) :=
by sorry

end sqrt_product_simplification_l2707_270768


namespace smallest_angle_when_largest_is_120_l2707_270700

/-- Represents a trapezoid with angles in arithmetic sequence -/
structure ArithmeticTrapezoid where
  /-- The smallest angle of the trapezoid -/
  smallest_angle : ℝ
  /-- The common difference between consecutive angles -/
  angle_difference : ℝ
  /-- The sum of all angles in the trapezoid is 360° -/
  angle_sum : smallest_angle + (smallest_angle + angle_difference) + 
              (smallest_angle + 2 * angle_difference) + 
              (smallest_angle + 3 * angle_difference) = 360

theorem smallest_angle_when_largest_is_120 (t : ArithmeticTrapezoid) 
  (h : t.smallest_angle + 3 * t.angle_difference = 120) : 
  t.smallest_angle = 60 := by
  sorry

#check smallest_angle_when_largest_is_120

end smallest_angle_when_largest_is_120_l2707_270700


namespace distance_opposite_points_l2707_270795

-- Define a point in polar coordinates
structure PolarPoint where
  r : ℝ
  θ : ℝ

-- Define the distance function between two polar points
def polarDistance (A B : PolarPoint) : ℝ :=
  sorry

-- Theorem statement
theorem distance_opposite_points (A B : PolarPoint) 
    (h : abs (B.θ - A.θ) = Real.pi) : 
  polarDistance A B = A.r + B.r := by
  sorry

end distance_opposite_points_l2707_270795


namespace swimmers_pass_count_l2707_270743

/-- Represents a swimmer in the pool --/
structure Swimmer where
  speed : ℝ
  turnDelay : ℝ

/-- Calculates the number of times swimmers pass each other --/
def countPasses (poolLength : ℝ) (swimmer1 : Swimmer) (swimmer2 : Swimmer) (totalTime : ℝ) : ℕ :=
  sorry

/-- Theorem stating the number of passes for the given problem --/
theorem swimmers_pass_count :
  let poolLength : ℝ := 120
  let swimmer1 : Swimmer := ⟨4, 2⟩
  let swimmer2 : Swimmer := ⟨3, 0⟩
  let totalTime : ℝ := 900
  countPasses poolLength swimmer1 swimmer2 totalTime = 26 :=
by sorry

end swimmers_pass_count_l2707_270743


namespace coterminal_pi_third_pi_equals_180_degrees_arc_length_pi_third_l2707_270753

-- Define the set of coterminal angles
def coterminalAngles (θ : ℝ) : Set ℝ := {α | ∃ k : ℤ, α = θ + 2 * k * Real.pi}

-- Statement 1: Coterminal angles with π/3
theorem coterminal_pi_third : 
  coterminalAngles (Real.pi / 3) = {α | ∃ k : ℤ, α = Real.pi / 3 + 2 * k * Real.pi} :=
sorry

-- Statement 2: π radians equals 180 degrees
theorem pi_equals_180_degrees : 
  Real.pi = 180 * (Real.pi / 180) :=
sorry

-- Statement 3: Arc length in a circle
theorem arc_length_pi_third : 
  let r : ℝ := 6
  let θ : ℝ := Real.pi / 3
  r * θ = 2 * Real.pi :=
sorry

end coterminal_pi_third_pi_equals_180_degrees_arc_length_pi_third_l2707_270753


namespace push_up_difference_l2707_270715

theorem push_up_difference (zachary_pushups david_pushups : ℕ) 
  (h1 : zachary_pushups = 19)
  (h2 : david_pushups = 58) :
  david_pushups - zachary_pushups = 39 := by
  sorry

end push_up_difference_l2707_270715


namespace election_votes_theorem_l2707_270732

theorem election_votes_theorem (total_votes : ℕ) : 
  (∃ (winner_votes loser_votes : ℕ),
    winner_votes + loser_votes = total_votes ∧
    winner_votes = (70 * total_votes) / 100 ∧
    winner_votes - loser_votes = 320) →
  total_votes = 800 := by
sorry

end election_votes_theorem_l2707_270732


namespace sin_sum_of_complex_exponentials_l2707_270750

theorem sin_sum_of_complex_exponentials (θ φ : ℝ) :
  Complex.exp (θ * Complex.I) = 4/5 + 3/5 * Complex.I →
  Complex.exp (φ * Complex.I) = -5/13 + 12/13 * Complex.I →
  Real.sin (θ + φ) = 84/65 := by
  sorry

end sin_sum_of_complex_exponentials_l2707_270750


namespace athlete_weight_problem_l2707_270742

theorem athlete_weight_problem (a b c : ℕ) : 
  (a + b + c) / 3 = 42 →
  (a + b) / 2 = 40 →
  (b + c) / 2 = 43 →
  ∃ k₁ k₂ k₃ : ℕ, a = 5 * k₁ ∧ b = 5 * k₂ ∧ c = 5 * k₃ →
  b = 40 := by
sorry

end athlete_weight_problem_l2707_270742


namespace umbrella_probability_l2707_270733

theorem umbrella_probability (p_forget : ℚ) (h1 : p_forget = 5/8) :
  1 - p_forget = 3/8 := by
  sorry

end umbrella_probability_l2707_270733
