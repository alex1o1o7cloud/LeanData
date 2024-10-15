import Mathlib

namespace NUMINAMATH_CALUDE_two_digit_number_theorem_l1034_103498

/-- Represents a two-digit integer with its tens and units digits. -/
structure TwoDigitNumber where
  tens : ℕ
  units : ℕ
  is_valid : tens ≥ 1 ∧ tens ≤ 9 ∧ units ≤ 9

/-- Theorem stating the relationship between j and k for a two-digit number. -/
theorem two_digit_number_theorem (n : TwoDigitNumber) (k j : ℚ) :
  (10 * n.tens + n.units : ℚ) = k * (n.tens + n.units) →
  (20 * n.units + n.tens : ℚ) = j * (n.tens + n.units) →
  j = (199 + k) / 10 := by
  sorry

end NUMINAMATH_CALUDE_two_digit_number_theorem_l1034_103498


namespace NUMINAMATH_CALUDE_intersection_A_B_union_B_C_implies_a_range_l1034_103411

-- Define the sets A, B, and C
def A : Set ℝ := {x | -1 ≤ x ∧ x < 3}
def B : Set ℝ := {x | 2*x - 4 ≥ x - 2}
def C (a : ℝ) : Set ℝ := {x | x ≥ a - 1}

-- Theorem for part (1)
theorem intersection_A_B : A ∩ B = {x | 2 ≤ x ∧ x < 3} := by sorry

-- Theorem for part (2)
theorem union_B_C_implies_a_range (a : ℝ) : B ∪ C a = C a → a ≤ 3 := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_union_B_C_implies_a_range_l1034_103411


namespace NUMINAMATH_CALUDE_first_term_of_arithmetic_sequence_l1034_103490

-- Define an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- State the theorem
theorem first_term_of_arithmetic_sequence 
  (a : ℕ → ℝ) 
  (h_arith : is_arithmetic_sequence a) 
  (h_a5 : a 5 = 9) 
  (h_a3_a2 : 2 * a 3 = a 2 + 6) : 
  a 1 = -3 := by
sorry

end NUMINAMATH_CALUDE_first_term_of_arithmetic_sequence_l1034_103490


namespace NUMINAMATH_CALUDE_positive_sum_greater_than_abs_difference_l1034_103495

theorem positive_sum_greater_than_abs_difference (x y : ℝ) :
  x + y > |x - y| ↔ x > 0 ∧ y > 0 := by sorry

end NUMINAMATH_CALUDE_positive_sum_greater_than_abs_difference_l1034_103495


namespace NUMINAMATH_CALUDE_complementary_event_is_both_red_l1034_103428

/-- Represents the color of a ball -/
inductive Color
| Red
| White

/-- Represents the outcome of drawing two balls -/
structure TwoBallDraw where
  first : Color
  second : Color

/-- The set of all possible outcomes when drawing two balls -/
def allOutcomes : Set TwoBallDraw :=
  {⟨Color.Red, Color.Red⟩, ⟨Color.Red, Color.White⟩, 
   ⟨Color.White, Color.Red⟩, ⟨Color.White, Color.White⟩}

/-- Event A: At least one white ball -/
def eventA : Set TwoBallDraw :=
  {draw ∈ allOutcomes | draw.first = Color.White ∨ draw.second = Color.White}

/-- The complementary event of A -/
def complementA : Set TwoBallDraw :=
  allOutcomes \ eventA

theorem complementary_event_is_both_red :
  complementA = {⟨Color.Red, Color.Red⟩} :=
by sorry

end NUMINAMATH_CALUDE_complementary_event_is_both_red_l1034_103428


namespace NUMINAMATH_CALUDE_square_floor_tiles_l1034_103485

/-- A square floor covered with congruent square tiles -/
structure SquareFloor :=
  (side_length : ℕ)

/-- The number of tiles along the diagonals of a square floor -/
def diagonal_tiles (floor : SquareFloor) : ℕ :=
  2 * floor.side_length - 1

/-- The total number of tiles covering a square floor -/
def total_tiles (floor : SquareFloor) : ℕ :=
  floor.side_length ^ 2

/-- Theorem: If the total number of tiles along the two diagonals is 49,
    then the number of tiles covering the entire floor is 625 -/
theorem square_floor_tiles (floor : SquareFloor) :
  diagonal_tiles floor = 49 → total_tiles floor = 625 :=
by
  sorry


end NUMINAMATH_CALUDE_square_floor_tiles_l1034_103485


namespace NUMINAMATH_CALUDE_matrix_cube_computation_l1034_103448

def A : Matrix (Fin 2) (Fin 2) ℤ := !![2, -1; 1, 1]

theorem matrix_cube_computation :
  A ^ 3 = !![3, -6; 6, -3] := by sorry

end NUMINAMATH_CALUDE_matrix_cube_computation_l1034_103448


namespace NUMINAMATH_CALUDE_parabola_directrix_l1034_103407

/-- The directrix of a parabola y^2 = 16x is x = -4 -/
theorem parabola_directrix (x y : ℝ) : y^2 = 16*x → (∃ (a : ℝ), a = 4 ∧ x = -a) :=
by sorry

end NUMINAMATH_CALUDE_parabola_directrix_l1034_103407


namespace NUMINAMATH_CALUDE_largest_integer_before_zero_l1034_103462

noncomputable def f (x : ℝ) := Real.log x + 2 * x - 6

theorem largest_integer_before_zero (x₀ : ℝ) (h : f x₀ = 0) :
  ∃ k : ℤ, k = 2 ∧ k ≤ ⌊x₀⌋ ∧ ∀ m : ℤ, m > k → m > ⌊x₀⌋ :=
sorry

end NUMINAMATH_CALUDE_largest_integer_before_zero_l1034_103462


namespace NUMINAMATH_CALUDE_john_tax_rate_l1034_103473

/-- Given the number of shirts, price per shirt, and total payment including tax,
    calculate the tax rate as a percentage. -/
def calculate_tax_rate (num_shirts : ℕ) (price_per_shirt : ℚ) (total_payment : ℚ) : ℚ :=
  let cost_before_tax := num_shirts * price_per_shirt
  let tax_amount := total_payment - cost_before_tax
  (tax_amount / cost_before_tax) * 100

/-- Theorem stating that for 3 shirts at $20 each and a total payment of $66,
    the tax rate is 10%. -/
theorem john_tax_rate :
  calculate_tax_rate 3 20 66 = 10 := by
  sorry

#eval calculate_tax_rate 3 20 66

end NUMINAMATH_CALUDE_john_tax_rate_l1034_103473


namespace NUMINAMATH_CALUDE_art_dealer_earnings_l1034_103408

/-- Calculates the total money made from selling etchings -/
def total_money_made (total_etchings : ℕ) (first_group_count : ℕ) (first_group_price : ℕ) (second_group_price : ℕ) : ℕ :=
  let second_group_count := total_etchings - first_group_count
  (first_group_count * first_group_price) + (second_group_count * second_group_price)

/-- Proves that the art dealer made $630 from selling the etchings -/
theorem art_dealer_earnings : total_money_made 16 9 35 45 = 630 := by
  sorry

end NUMINAMATH_CALUDE_art_dealer_earnings_l1034_103408


namespace NUMINAMATH_CALUDE_not_perfect_square_l1034_103402

theorem not_perfect_square (a b : ℕ) (h1 : a > b) (h2 : Nat.gcd (a - b) (a * b + 1) = 1) (h3 : Nat.gcd (a + b) (a * b - 1) = 1) :
  ¬ ∃ k : ℕ, (a - b)^2 + (a * b + 1)^2 = k^2 := by
  sorry

end NUMINAMATH_CALUDE_not_perfect_square_l1034_103402


namespace NUMINAMATH_CALUDE_cat_and_mouse_positions_l1034_103482

def cat_cycle_length : ℕ := 4
def mouse_cycle_length : ℕ := 8
def total_moves : ℕ := 247

theorem cat_and_mouse_positions :
  (total_moves % cat_cycle_length = 3) ∧
  (total_moves % mouse_cycle_length = 7) := by
  sorry

end NUMINAMATH_CALUDE_cat_and_mouse_positions_l1034_103482


namespace NUMINAMATH_CALUDE_continuity_at_three_l1034_103431

def f (x : ℝ) : ℝ := 2 * x^2 - 4

theorem continuity_at_three :
  ∀ ε > 0, ∃ δ > 0, ∀ x, |x - 3| < δ → |f x - f 3| < ε := by
  sorry

end NUMINAMATH_CALUDE_continuity_at_three_l1034_103431


namespace NUMINAMATH_CALUDE_sarah_age_l1034_103414

/-- Given the ages of Sarah, Mark, Billy, and Ana, prove Sarah's age -/
theorem sarah_age (sarah mark billy ana : ℕ) 
  (h1 : sarah = 3 * mark - 4)
  (h2 : mark = billy + 4)
  (h3 : billy = ana / 2)
  (h4 : ana + 3 = 15) : 
  sarah = 26 := by
  sorry

end NUMINAMATH_CALUDE_sarah_age_l1034_103414


namespace NUMINAMATH_CALUDE_cost_price_calculation_l1034_103449

/-- Proves that given a selling price of 400 Rs. and a profit percentage of 25%, the cost price is 320 Rs. -/
theorem cost_price_calculation (selling_price : ℝ) (profit_percentage : ℝ) :
  selling_price = 400 →
  profit_percentage = 25 →
  selling_price = (1 + profit_percentage / 100) * 320 :=
by
  sorry

#check cost_price_calculation

end NUMINAMATH_CALUDE_cost_price_calculation_l1034_103449


namespace NUMINAMATH_CALUDE_shaded_area_square_with_quarter_circles_l1034_103451

/-- The area of the shaded region in a square with quarter circles at corners -/
theorem shaded_area_square_with_quarter_circles 
  (side_length : ℝ) 
  (radius : ℝ) 
  (h1 : side_length = 12) 
  (h2 : radius = 6) : 
  side_length ^ 2 - π * radius ^ 2 = 144 - 36 * π := by
  sorry

#check shaded_area_square_with_quarter_circles

end NUMINAMATH_CALUDE_shaded_area_square_with_quarter_circles_l1034_103451


namespace NUMINAMATH_CALUDE_computer_price_reduction_l1034_103418

/-- The average percentage decrease per price reduction for a computer model -/
theorem computer_price_reduction (original_price final_price : ℝ) 
  (h1 : original_price = 5000)
  (h2 : final_price = 2560)
  (h3 : ∃ x : ℝ, original_price * (1 - x/100)^3 = final_price) :
  ∃ x : ℝ, x = 20 ∧ original_price * (1 - x/100)^3 = final_price := by
sorry


end NUMINAMATH_CALUDE_computer_price_reduction_l1034_103418


namespace NUMINAMATH_CALUDE_complex_condition_implies_a_value_l1034_103455

theorem complex_condition_implies_a_value (a : ℝ) :
  (((a : ℂ) + Complex.I) * (2 * Complex.I)).re > 0 → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_condition_implies_a_value_l1034_103455


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_l1034_103421

theorem quadratic_roots_sum (a b : ℝ) (ha : a^2 - 8*a + 5 = 0) (hb : b^2 - 8*b + 5 = 0) (hab : a ≠ b) :
  (b - 1) / (a - 1) + (a - 1) / (b - 1) = -20 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_l1034_103421


namespace NUMINAMATH_CALUDE_pirate_treasure_distribution_l1034_103417

theorem pirate_treasure_distribution (x : ℕ) : x > 0 → (x * (x + 1)) / 2 = 5 * x → x + 5 * x = 54 := by
  sorry

end NUMINAMATH_CALUDE_pirate_treasure_distribution_l1034_103417


namespace NUMINAMATH_CALUDE_sum_of_triangles_l1034_103446

-- Define the triangle operation
def triangle (a b c : ℤ) : ℤ := a + 2*b - c

-- Theorem statement
theorem sum_of_triangles : triangle 3 5 7 + triangle 6 1 8 = 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_triangles_l1034_103446


namespace NUMINAMATH_CALUDE_cube_volume_from_surface_area_l1034_103478

theorem cube_volume_from_surface_area (surface_area : ℝ) (volume : ℝ) : 
  surface_area = 1734 → volume = (((surface_area / 6) ^ (1/2 : ℝ)) ^ 3) → volume = 4913 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_surface_area_l1034_103478


namespace NUMINAMATH_CALUDE_milk_price_is_three_l1034_103475

/-- Represents the milk and butter selling scenario --/
structure MilkButterScenario where
  num_cows : ℕ
  milk_per_cow : ℕ
  num_customers : ℕ
  milk_per_customer : ℕ
  butter_sticks_per_gallon : ℕ
  butter_price_per_stick : ℚ
  total_earnings : ℚ

/-- Calculates the price per gallon of milk --/
def price_per_gallon (scenario : MilkButterScenario) : ℚ :=
  let total_milk := scenario.num_cows * scenario.milk_per_cow
  let sold_milk := scenario.num_customers * scenario.milk_per_customer
  let remaining_milk := total_milk - sold_milk
  let butter_sticks := remaining_milk * scenario.butter_sticks_per_gallon
  let butter_earnings := butter_sticks * scenario.butter_price_per_stick
  let milk_earnings := scenario.total_earnings - butter_earnings
  milk_earnings / sold_milk

/-- Theorem stating that the price per gallon of milk is $3 --/
theorem milk_price_is_three (scenario : MilkButterScenario) 
  (h1 : scenario.num_cows = 12)
  (h2 : scenario.milk_per_cow = 4)
  (h3 : scenario.num_customers = 6)
  (h4 : scenario.milk_per_customer = 6)
  (h5 : scenario.butter_sticks_per_gallon = 2)
  (h6 : scenario.butter_price_per_stick = 3/2)
  (h7 : scenario.total_earnings = 144) :
  price_per_gallon scenario = 3 := by
  sorry

end NUMINAMATH_CALUDE_milk_price_is_three_l1034_103475


namespace NUMINAMATH_CALUDE_set_equality_implies_difference_l1034_103472

theorem set_equality_implies_difference (a b : ℝ) :
  ({0, b/a, b} : Set ℝ) = ({1, a+b, a} : Set ℝ) →
  b - a = 2 := by
sorry

end NUMINAMATH_CALUDE_set_equality_implies_difference_l1034_103472


namespace NUMINAMATH_CALUDE_one_correct_description_l1034_103480

/-- Represents an experimental description --/
structure ExperimentalDescription where
  id : Nat
  isCorrect : Bool

/-- The set of all experimental descriptions --/
def experimentDescriptions : Finset ExperimentalDescription := sorry

/-- Theorem stating that there is exactly one correct experimental description --/
theorem one_correct_description :
  (experimentDescriptions.filter (λ d => d.isCorrect)).card = 1 := by sorry

end NUMINAMATH_CALUDE_one_correct_description_l1034_103480


namespace NUMINAMATH_CALUDE_abs_eq_sqrt_square_abs_sqrt_square_domain_l1034_103491

theorem abs_eq_sqrt_square (x : ℝ) : |x| = Real.sqrt (x^2) := by sorry

theorem abs_sqrt_square_domain : Set.univ = {x : ℝ | ∃ y, y = Real.sqrt (x^2)} := by sorry

end NUMINAMATH_CALUDE_abs_eq_sqrt_square_abs_sqrt_square_domain_l1034_103491


namespace NUMINAMATH_CALUDE_circle_center_transformation_l1034_103420

def reflect_across_x_axis (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

def translate_right (p : ℝ × ℝ) (d : ℝ) : ℝ × ℝ :=
  (p.1 + d, p.2)

theorem circle_center_transformation :
  let initial_center : ℝ × ℝ := (3, -4)
  let reflected_center := reflect_across_x_axis initial_center
  let final_center := translate_right reflected_center 5
  final_center = (8, 4) := by
sorry

end NUMINAMATH_CALUDE_circle_center_transformation_l1034_103420


namespace NUMINAMATH_CALUDE_m_range_l1034_103422

/-- The function f(x) = x³ - ax + 2 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - a*x + 2

/-- The function g(x) = f(x) + mx -/
def g (a m : ℝ) (x : ℝ) : ℝ := f a x + m*x

theorem m_range (a m : ℝ) : 
  (∃ x₀, ∀ x, f a x ≤ f a x₀ ∧ f a x₀ = 4) →
  (∃ x₁ ∈ Set.Ioo (-3) (a - 1), ∀ x ∈ Set.Ioo (-3) (a - 1), g a m x₁ ≤ g a m x ∧ g a m x₁ ≤ m - 1) →
  -9 < m ∧ m ≤ -15/4 := by sorry

end NUMINAMATH_CALUDE_m_range_l1034_103422


namespace NUMINAMATH_CALUDE_right_triangle_division_area_ratio_l1034_103419

/-- Given a right triangle divided into a rectangle and two smaller right triangles,
    where one smaller triangle has area n times the rectangle's area,
    and the rectangle's length is twice its width,
    prove that the ratio of the other small triangle's area to the rectangle's area is 1/(4n). -/
theorem right_triangle_division_area_ratio (n : ℝ) (h : n > 0) :
  ∃ (x : ℝ) (t s : ℝ),
    x > 0 ∧ 
    2 * x^2 > 0 ∧  -- Area of rectangle
    (1/2) * t * x = n * (2 * x^2) ∧  -- Area of one small triangle
    s / x = x / (2 * n * x) ∧  -- Similar triangles ratio
    ((1/2) * (2*x) * s) / (2 * x^2) = 1 / (4*n) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_division_area_ratio_l1034_103419


namespace NUMINAMATH_CALUDE_evaluate_expression_l1034_103439

theorem evaluate_expression (x : ℝ) (h : 3 * x - 2 = 13) : 6 * x - 4 = 26 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1034_103439


namespace NUMINAMATH_CALUDE_sum_of_sixth_powers_l1034_103481

theorem sum_of_sixth_powers (ζ₁ ζ₂ ζ₃ : ℂ) 
  (h1 : ζ₁ + ζ₂ + ζ₃ = 2)
  (h2 : ζ₁^2 + ζ₂^2 + ζ₃^2 = 5)
  (h3 : ζ₁^3 + ζ₂^3 + ζ₃^3 = 14) :
  ζ₁^6 + ζ₂^6 + ζ₃^6 = 128.75 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_sixth_powers_l1034_103481


namespace NUMINAMATH_CALUDE_functional_equation_solutions_l1034_103438

def functional_equation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (f x + x * y) = f x * f (y + 1)

theorem functional_equation_solutions :
  ∀ f : ℝ → ℝ, functional_equation f →
    (∀ x, f x = 0) ∨ (∀ x, f x = 1) ∨ (∀ x, f x = x) := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solutions_l1034_103438


namespace NUMINAMATH_CALUDE_abs_a_minus_b_ge_four_l1034_103459

theorem abs_a_minus_b_ge_four (a b : ℝ) : 
  (∀ x : ℝ, |x - a| < 1 → |x - b| > 3) → 
  |a - b| ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_abs_a_minus_b_ge_four_l1034_103459


namespace NUMINAMATH_CALUDE_square_sum_equals_one_l1034_103488

theorem square_sum_equals_one (a b : ℝ) 
  (h : a * Real.sqrt (1 - b^2) + b * Real.sqrt (1 - a^2) = 1) : 
  a^2 + b^2 = 1 := by
sorry

end NUMINAMATH_CALUDE_square_sum_equals_one_l1034_103488


namespace NUMINAMATH_CALUDE_system_solution_l1034_103443

theorem system_solution : 
  ∀ (a b c d : ℝ), 
    a + c = -7 ∧ 
    a * c + b + d = 18 ∧ 
    a * d + b * c = -22 ∧ 
    b * d = 12 → 
    ((a = -5 ∧ b = 6 ∧ c = -2 ∧ d = 2) ∨ 
     (a = -2 ∧ b = 2 ∧ c = -5 ∧ d = 6)) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l1034_103443


namespace NUMINAMATH_CALUDE_instantaneous_velocity_at_4_l1034_103409

/-- The position function of the object -/
def s (t : ℝ) : ℝ := 3 * t^2 + t + 4

/-- The velocity function of the object (derivative of s) -/
def v (t : ℝ) : ℝ := 6 * t + 1

theorem instantaneous_velocity_at_4 : v 4 = 25 := by
  sorry

end NUMINAMATH_CALUDE_instantaneous_velocity_at_4_l1034_103409


namespace NUMINAMATH_CALUDE_twelfth_nine_position_l1034_103410

/-- The position of the nth occurrence of a digit in the sequence of natural numbers written without spaces -/
def digitPosition (n : ℕ) (digit : ℕ) : ℕ :=
  sorry

/-- The sequence of natural numbers written without spaces -/
def naturalNumberSequence : List ℕ :=
  sorry

theorem twelfth_nine_position :
  digitPosition 12 9 = 174 :=
sorry

end NUMINAMATH_CALUDE_twelfth_nine_position_l1034_103410


namespace NUMINAMATH_CALUDE_geometric_series_sum_l1034_103497

theorem geometric_series_sum (a b : ℝ) (h : ∑' n, a / b^n = 5) : 
  ∑' n, a / (a + b)^n = 5/6 := by
sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l1034_103497


namespace NUMINAMATH_CALUDE_cone_volume_l1034_103483

/-- The volume of a cone with given slant height and height --/
theorem cone_volume (slant_height height : ℝ) (h1 : slant_height = 15) (h2 : height = 9) :
  (1 / 3 : ℝ) * Real.pi * (slant_height ^ 2 - height ^ 2) * height = 432 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_cone_volume_l1034_103483


namespace NUMINAMATH_CALUDE_soccer_team_lineup_combinations_l1034_103444

-- Define the total number of players
def total_players : ℕ := 16

-- Define the number of quadruplets
def num_quadruplets : ℕ := 4

-- Define the number of starters to choose
def num_starters : ℕ := 7

-- Define the number of quadruplets that must be in the starting lineup
def quadruplets_in_lineup : ℕ := 3

-- Theorem statement
theorem soccer_team_lineup_combinations :
  (Nat.choose num_quadruplets quadruplets_in_lineup) *
  (Nat.choose (total_players - num_quadruplets) (num_starters - quadruplets_in_lineup)) = 1980 :=
by sorry

end NUMINAMATH_CALUDE_soccer_team_lineup_combinations_l1034_103444


namespace NUMINAMATH_CALUDE_prime_equation_l1034_103458

theorem prime_equation (a b : ℕ) : 
  Prime a → Prime b → a^11 + b = 2089 → 49*b - a = 2007 := by sorry

end NUMINAMATH_CALUDE_prime_equation_l1034_103458


namespace NUMINAMATH_CALUDE_sum_of_eight_five_to_eight_l1034_103434

theorem sum_of_eight_five_to_eight (n : ℕ) :
  (Finset.range 8).sum (λ _ => 5^8) = 3125000 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_eight_five_to_eight_l1034_103434


namespace NUMINAMATH_CALUDE_product_of_fractions_l1034_103450

theorem product_of_fractions : (1 / 3) * (1 / 4) * (1 / 5) * (1 / 6) = 1 / 360 := by
  sorry

end NUMINAMATH_CALUDE_product_of_fractions_l1034_103450


namespace NUMINAMATH_CALUDE_log_equality_implies_n_fifth_power_l1034_103499

theorem log_equality_implies_n_fifth_power (n : ℝ) :
  n > 0 →
  (Real.log (675 * Real.sqrt 3)) / (Real.log (3 * n)) = (Real.log 75) / (Real.log n) →
  n^5 = 5625 := by
  sorry

end NUMINAMATH_CALUDE_log_equality_implies_n_fifth_power_l1034_103499


namespace NUMINAMATH_CALUDE_min_students_in_class_l1034_103437

theorem min_students_in_class (boys girls : ℕ) : 
  boys > 0 → 
  girls > 0 → 
  (3 * boys) % 4 = 0 → 
  (3 * boys) / 4 = girls / 2 → 
  5 ≤ boys + girls :=
sorry

end NUMINAMATH_CALUDE_min_students_in_class_l1034_103437


namespace NUMINAMATH_CALUDE_root_product_expression_l1034_103430

theorem root_product_expression (p q : ℝ) (α β γ δ : ℂ) : 
  (α^2 + p*α - 1 = 0) → 
  (β^2 + p*β - 1 = 0) → 
  (γ^2 + q*γ + 1 = 0) → 
  (δ^2 + q*δ + 1 = 0) → 
  (α - γ)*(β - γ)*(α + δ)*(β + δ) = p^2 - q^2 := by
sorry

end NUMINAMATH_CALUDE_root_product_expression_l1034_103430


namespace NUMINAMATH_CALUDE_circle_C_equation_l1034_103405

/-- Given circle is symmetric to (x-1)^2 + y^2 = 1 with respect to y = -x -/
def given_circle (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1

/-- Line of symmetry -/
def symmetry_line (x y : ℝ) : Prop := y = -x

/-- Circle C -/
def circle_C (x y : ℝ) : Prop := x^2 + (y + 1)^2 = 1

/-- Symmetry transformation -/
def symmetric_point (x y : ℝ) : ℝ × ℝ := (-y, -x)

theorem circle_C_equation :
  ∀ x y : ℝ, 
  (∃ x' y' : ℝ, given_circle x' y' ∧ 
   symmetric_point x y = (x', y') ∧
   symmetry_line x y) →
  circle_C x y :=
sorry

end NUMINAMATH_CALUDE_circle_C_equation_l1034_103405


namespace NUMINAMATH_CALUDE_stone_slab_length_l1034_103494

/-- Given a floor covered by square stone slabs, this theorem calculates the length of each slab. -/
theorem stone_slab_length
  (num_slabs : ℕ)
  (total_area : ℝ)
  (h_num_slabs : num_slabs = 30)
  (h_total_area : total_area = 50.7) :
  ∃ (slab_length : ℝ),
    slab_length = 130 ∧
    num_slabs * (slab_length / 100)^2 = total_area :=
by sorry

end NUMINAMATH_CALUDE_stone_slab_length_l1034_103494


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l1034_103442

theorem parallel_vectors_x_value (x : ℝ) (a b : ℝ × ℝ) :
  a = (2, x) →
  b = (4, -1) →
  (∃ (k : ℝ), a = k • b) →
  x = -1/2 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l1034_103442


namespace NUMINAMATH_CALUDE_number_problem_l1034_103440

theorem number_problem (A B C : ℝ) 
  (h1 : A - B = 1620)
  (h2 : 0.075 * A = 0.125 * B)
  (h3 : 0.06 * B = 0.10 * C) :
  A = 4050 ∧ B = 2430 ∧ C = 1458 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l1034_103440


namespace NUMINAMATH_CALUDE_no_solution_when_m_equals_negative_one_l1034_103447

theorem no_solution_when_m_equals_negative_one :
  ∀ x : ℝ, (3 - 2*x) / (x - 3) + (2 + (-1)*x) / (3 - x) ≠ -1 :=
by
  sorry

end NUMINAMATH_CALUDE_no_solution_when_m_equals_negative_one_l1034_103447


namespace NUMINAMATH_CALUDE_principal_amount_proof_l1034_103470

/-- Proves that given specific interest conditions, the principal amount is 6400 --/
theorem principal_amount_proof (rate : ℝ) (time : ℝ) (difference : ℝ) : rate = 0.05 → time = 2 → difference = 16 → 
  ∃ (principal : ℝ), principal * ((1 + rate)^time - 1 - rate * time) = difference ∧ principal = 6400 := by
  sorry

end NUMINAMATH_CALUDE_principal_amount_proof_l1034_103470


namespace NUMINAMATH_CALUDE_triangle_side_calculation_l1034_103424

theorem triangle_side_calculation (A B C : Real) (a b c : Real) :
  -- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively
  -- Given conditions
  (a = 2) →
  (A = 30 * π / 180) →  -- Convert degrees to radians
  (B = 45 * π / 180) →  -- Convert degrees to radians
  -- Law of Sines
  (a / Real.sin A = b / Real.sin B) →
  -- Conclusion
  b = 2 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_calculation_l1034_103424


namespace NUMINAMATH_CALUDE_complex_subtraction_simplification_l1034_103465

theorem complex_subtraction_simplification :
  (5 - 7 * Complex.I) - (3 - 2 * Complex.I) = 2 - 5 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_subtraction_simplification_l1034_103465


namespace NUMINAMATH_CALUDE_polygon_properties_l1034_103493

/-- The number of diagonals from a vertex in a polygon with n sides -/
def diagonals_from_vertex (n : ℕ) : ℕ := n - 3

/-- The number of triangles formed by diagonals in a polygon with n sides -/
def triangles_formed (n : ℕ) : ℕ := n - 2

/-- The sum of interior angles of a polygon with n sides -/
def sum_interior_angles (n : ℕ) : ℕ := (n - 2) * 180

/-- The sum of exterior angles of any polygon -/
def sum_exterior_angles : ℕ := 360

theorem polygon_properties :
  (diagonals_from_vertex 6 = 3) ∧
  (triangles_formed 6 = 4) ∧
  (sum_interior_angles 6 = 720) ∧
  (∃ n : ℕ, sum_interior_angles n = 2 * sum_exterior_angles - 180 ∧ n = 5) :=
sorry

end NUMINAMATH_CALUDE_polygon_properties_l1034_103493


namespace NUMINAMATH_CALUDE_unique_number_l1034_103456

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def is_odd (n : ℕ) : Prop := ∃ k : ℕ, n = 2*k + 1

def is_multiple_of_9 (n : ℕ) : Prop := ∃ k : ℕ, n = 9*k

def digit_product (n : ℕ) : ℕ := (n / 10) * (n % 10)

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, n = k^2

theorem unique_number : 
  ∃! n : ℕ, is_two_digit n ∧ 
            is_odd n ∧ 
            is_multiple_of_9 n ∧ 
            is_perfect_square (digit_product n) ∧
            n = 99 := by
  sorry

end NUMINAMATH_CALUDE_unique_number_l1034_103456


namespace NUMINAMATH_CALUDE_mixture_price_calculation_l1034_103404

/-- Calculates the price of a mixture given the prices of two components and their ratio -/
def mixturePricePerKg (pricePeas : ℚ) (priceSoybean : ℚ) (ratioPeas : ℕ) (ratioSoybean : ℕ) : ℚ :=
  let totalParts := ratioPeas + ratioSoybean
  let totalPrice := pricePeas * ratioPeas + priceSoybean * ratioSoybean
  totalPrice / totalParts

theorem mixture_price_calculation (pricePeas priceSoybean : ℚ) (ratioPeas ratioSoybean : ℕ) :
  pricePeas = 16 →
  priceSoybean = 25 →
  ratioPeas = 2 →
  ratioSoybean = 1 →
  mixturePricePerKg pricePeas priceSoybean ratioPeas ratioSoybean = 19 := by
  sorry

end NUMINAMATH_CALUDE_mixture_price_calculation_l1034_103404


namespace NUMINAMATH_CALUDE_division_relation_l1034_103452

theorem division_relation (D : ℝ) (h : D > 0) :
  let d := D / 35
  let q := D / 5
  q = D / 5 ∧ q = 7 * d := by sorry

end NUMINAMATH_CALUDE_division_relation_l1034_103452


namespace NUMINAMATH_CALUDE_cristine_lemons_l1034_103416

theorem cristine_lemons : ∀ (initial_lemons : ℕ),
  (3 / 4 : ℚ) * initial_lemons = 9 →
  initial_lemons = 12 := by
  sorry

end NUMINAMATH_CALUDE_cristine_lemons_l1034_103416


namespace NUMINAMATH_CALUDE_stratified_sampling_size_l1034_103436

theorem stratified_sampling_size 
  (high_school_students : ℕ) 
  (junior_high_students : ℕ) 
  (sampled_high_school : ℕ) 
  (h1 : high_school_students = 3500)
  (h2 : junior_high_students = 1500)
  (h3 : sampled_high_school = 70) :
  let total_students := high_school_students + junior_high_students
  let sampling_ratio := sampled_high_school / high_school_students
  let total_sample_size := total_students * sampling_ratio
  total_sample_size = 100 := by
sorry

end NUMINAMATH_CALUDE_stratified_sampling_size_l1034_103436


namespace NUMINAMATH_CALUDE_last_two_digits_product_l1034_103487

theorem last_two_digits_product (n : ℤ) : 
  (n % 100 ≥ 10) →  -- Ensure n has at least two digits
  (n % 8 = 0) →     -- n is divisible by 8
  ((n % 100) / 10 + n % 10 = 12) →  -- Sum of last two digits is 12
  ((n % 100) / 10 * (n % 10) = 32) :=  -- Product of last two digits is 32
by sorry

end NUMINAMATH_CALUDE_last_two_digits_product_l1034_103487


namespace NUMINAMATH_CALUDE_quadratic_inequality_l1034_103468

theorem quadratic_inequality (x : ℝ) : x^2 - 8*x + 12 < 0 ↔ 2 < x ∧ x < 6 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l1034_103468


namespace NUMINAMATH_CALUDE_unique_prime_solution_l1034_103464

theorem unique_prime_solution : ∃! (p : ℕ), Prime p ∧ (p^4 + 2*p^3 + 4*p^2 + 2*p + 1)^5 = 418195493 := by
  sorry

end NUMINAMATH_CALUDE_unique_prime_solution_l1034_103464


namespace NUMINAMATH_CALUDE_star_op_greater_star_op_commutative_l1034_103467

-- Define the new operation ※ for rational numbers
def star_op (a b : ℚ) : ℚ := (a + b + abs (a - b)) / 2

-- Theorem for part (2)
theorem star_op_greater (a b : ℚ) (h : a > b) : star_op a b = a := by sorry

-- Theorem for part (3)
theorem star_op_commutative (a b : ℚ) : star_op a b = star_op b a := by sorry

-- Examples for part (1)
example : star_op 2 3 = 3 := by sorry
example : star_op 3 3 = 3 := by sorry
example : star_op (-2) (-3) = -2 := by sorry

end NUMINAMATH_CALUDE_star_op_greater_star_op_commutative_l1034_103467


namespace NUMINAMATH_CALUDE_parallel_planes_sufficient_not_necessary_l1034_103457

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the parallel relation between planes
variable (plane_parallel : Plane → Plane → Prop)

-- Define the parallel relation between a line and a plane
variable (line_plane_parallel : Line → Plane → Prop)

-- Define the subset relation between a line and a plane
variable (line_in_plane : Line → Plane → Prop)

-- State the theorem
theorem parallel_planes_sufficient_not_necessary
  (α β : Plane) (a : Line)
  (h_a_in_α : line_in_plane a α) :
  (∀ α β a, plane_parallel α β → line_plane_parallel a β) ∧
  (∃ α β a, line_plane_parallel a β ∧ ¬ plane_parallel α β) :=
sorry

end NUMINAMATH_CALUDE_parallel_planes_sufficient_not_necessary_l1034_103457


namespace NUMINAMATH_CALUDE_traffic_light_theorem_l1034_103489

/-- Represents the probability of different traffic light combinations -/
structure TrafficLightProbabilities where
  p1 : ℝ  -- Both lights green
  p2 : ℝ  -- First green, second red
  p3 : ℝ  -- First red, second green
  p4 : ℝ  -- Both lights red

/-- The conditions of the traffic light problem -/
def traffic_light_conditions (p : TrafficLightProbabilities) : Prop :=
  0 ≤ p.p1 ∧ 0 ≤ p.p2 ∧ 0 ≤ p.p3 ∧ 0 ≤ p.p4 ∧  -- Probabilities are non-negative
  p.p1 + p.p2 + p.p3 + p.p4 = 1 ∧  -- Sum of probabilities is 1
  p.p1 + p.p2 = 2/3 ∧  -- First light is green 2/3 of the time
  p.p1 + p.p3 = 2/3 ∧  -- Second light is green 2/3 of the time
  p.p1 / (p.p1 + p.p2) = 3/4  -- Given first is green, second is green 3/4 of the time

/-- The theorem to be proved -/
theorem traffic_light_theorem (p : TrafficLightProbabilities) 
  (h : traffic_light_conditions p) : 
  p.p4 / (p.p3 + p.p4) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_traffic_light_theorem_l1034_103489


namespace NUMINAMATH_CALUDE_inequality_solutions_l1034_103454

theorem inequality_solutions :
  (∀ x : ℝ, x^2 - 2*x - 1 > 0 ↔ (x > Real.sqrt 2 + 1 ∨ x < -Real.sqrt 2 + 1)) ∧
  (∀ x : ℝ, x ≠ 3 → ((2*x - 1) / (x - 3) ≥ 3 ↔ 3 < x ∧ x ≤ 8)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solutions_l1034_103454


namespace NUMINAMATH_CALUDE_min_value_of_two_plus_y_l1034_103441

theorem min_value_of_two_plus_y (x y : ℝ) (h1 : y > 0) (h2 : x^2 + y - 3 = 0) :
  ∀ z, z = 2 + y → z ≥ 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_two_plus_y_l1034_103441


namespace NUMINAMATH_CALUDE_linear_equation_solution_l1034_103423

theorem linear_equation_solution : 
  ∃ x : ℚ, (x - 75) / 4 = (5 - 3 * x) / 7 ∧ x = 545 / 19 := by sorry

end NUMINAMATH_CALUDE_linear_equation_solution_l1034_103423


namespace NUMINAMATH_CALUDE_intersection_P_Q_l1034_103432

def P : Set ℝ := {x : ℝ | 1 < x ∧ x < 4}
def Q : Set ℝ := {x : ℝ | 2 < x ∧ x < 3}

theorem intersection_P_Q : P ∩ Q = {x : ℝ | 2 < x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_intersection_P_Q_l1034_103432


namespace NUMINAMATH_CALUDE_abs_sum_inequality_l1034_103425

theorem abs_sum_inequality (k : ℝ) : 
  (∀ x : ℝ, |x + 2| + |x + 1| > k) → k < 1 := by sorry

end NUMINAMATH_CALUDE_abs_sum_inequality_l1034_103425


namespace NUMINAMATH_CALUDE_total_vertices_eq_21_l1034_103401

/-- The number of vertices in a triangle -/
def triangle_vertices : ℕ := 3

/-- The number of vertices in a hexagon -/
def hexagon_vertices : ℕ := 6

/-- The number of triangles -/
def num_triangles : ℕ := 1

/-- The number of hexagons -/
def num_hexagons : ℕ := 3

/-- The total number of vertices in all shapes -/
def total_vertices : ℕ := num_triangles * triangle_vertices + num_hexagons * hexagon_vertices

theorem total_vertices_eq_21 : total_vertices = 21 := by
  sorry

end NUMINAMATH_CALUDE_total_vertices_eq_21_l1034_103401


namespace NUMINAMATH_CALUDE_solve_system_for_x_l1034_103427

theorem solve_system_for_x :
  ∀ x y : ℚ, 
  (2 * x - 3 * y = 18) → 
  (x + 2 * y = 8) → 
  x = 60 / 7 := by
sorry

end NUMINAMATH_CALUDE_solve_system_for_x_l1034_103427


namespace NUMINAMATH_CALUDE_min_value_theorem_l1034_103413

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 3) :
  (1 / x + 4 / y) ≥ 3 ∧ ∃ x0 y0 : ℝ, x0 > 0 ∧ y0 > 0 ∧ x0 + y0 = 3 ∧ 1 / x0 + 4 / y0 = 3 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1034_103413


namespace NUMINAMATH_CALUDE_circle_radius_with_perpendicular_chords_l1034_103463

/-- Given a circle with two perpendicular chords intersecting at the center,
    if two parallel sides of the formed quadrilateral have length 2,
    then the radius of the circle is √2. -/
theorem circle_radius_with_perpendicular_chords 
  (O : ℝ × ℝ) -- Center of the circle
  (K L M N : ℝ × ℝ) -- Points on the circle
  (h1 : (K.1 - M.1) * (L.2 - N.2) = 0) -- KM ⊥ LN
  (h2 : (K.2 - L.2) = (M.2 - N.2)) -- KL ∥ MN
  (h3 : Real.sqrt ((K.1 - L.1)^2 + (K.2 - L.2)^2) = 2) -- KL = 2
  (h4 : Real.sqrt ((M.1 - N.1)^2 + (M.2 - N.2)^2) = 2) -- MN = 2
  (h5 : O = (0, 0)) -- Center at origin
  (h6 : K.2 = 0 ∧ M.2 = 0) -- K and M on x-axis
  (h7 : L.1 = 0 ∧ N.1 = 0) -- L and N on y-axis
  : Real.sqrt (K.1^2 + N.2^2) = Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_circle_radius_with_perpendicular_chords_l1034_103463


namespace NUMINAMATH_CALUDE_compute_expression_l1034_103484

theorem compute_expression : 8 * (2 / 3)^4 = 128 / 81 := by
  sorry

end NUMINAMATH_CALUDE_compute_expression_l1034_103484


namespace NUMINAMATH_CALUDE_sum_of_x_and_y_equals_four_l1034_103486

theorem sum_of_x_and_y_equals_four (x y : ℝ) (i : ℂ) (h : i^2 = -1) 
  (eq : y + (2 - x) * i = 1 - i) : x + y = 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_x_and_y_equals_four_l1034_103486


namespace NUMINAMATH_CALUDE_intersection_sum_l1034_103400

theorem intersection_sum (c d : ℚ) : 
  (3 = (1/3) * (-1) + c) → 
  (-1 = (1/3) * 3 + d) → 
  c + d = 4/3 := by
sorry

end NUMINAMATH_CALUDE_intersection_sum_l1034_103400


namespace NUMINAMATH_CALUDE_circle_radius_is_5_l1034_103445

/-- The equation of a circle in the xy-plane -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 - 4*x + y^2 - 21 = 0

/-- The radius of the circle defined by circle_equation -/
def circle_radius : ℝ := 5

/-- Theorem: The radius of the circle defined by circle_equation is 5 -/
theorem circle_radius_is_5 : 
  ∀ x y : ℝ, circle_equation x y → 
  ∃ h k : ℝ, (x - h)^2 + (y - k)^2 = circle_radius^2 :=
sorry

end NUMINAMATH_CALUDE_circle_radius_is_5_l1034_103445


namespace NUMINAMATH_CALUDE_smallest_sum_of_reciprocals_l1034_103477

theorem smallest_sum_of_reciprocals (x y : ℕ+) :
  x ≠ y →
  (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / 18 →
  ∀ a b : ℕ+, a ≠ b → (1 : ℚ) / a + (1 : ℚ) / b = (1 : ℚ) / 18 →
  (x : ℕ) + (y : ℕ) ≤ (a : ℕ) + (b : ℕ) →
  (x : ℕ) + (y : ℕ) = 75 :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_of_reciprocals_l1034_103477


namespace NUMINAMATH_CALUDE_cookie_distribution_theorem_l1034_103471

/-- Represents the distribution of cookies in boxes -/
def CookieDistribution := List Nat

/-- Represents the process of taking cookies from boxes and placing them on plates -/
def distributeCookies (boxes : CookieDistribution) : List Nat :=
  let maxCookies := boxes.foldl max 0
  List.range maxCookies |>.map (fun i => boxes.filter (· > i) |>.length)

theorem cookie_distribution_theorem (boxes : CookieDistribution) :
  (boxes.toFinset |>.card) = ((distributeCookies boxes).toFinset |>.card) := by
  sorry

end NUMINAMATH_CALUDE_cookie_distribution_theorem_l1034_103471


namespace NUMINAMATH_CALUDE_cube_preserves_order_l1034_103426

theorem cube_preserves_order (a b : ℝ) (h : a > b) : a^3 > b^3 := by
  sorry

end NUMINAMATH_CALUDE_cube_preserves_order_l1034_103426


namespace NUMINAMATH_CALUDE_alligator_journey_time_l1034_103461

/-- The combined time of Paul's journey to the Nile Delta and back -/
def combined_journey_time (initial_time : ℕ) (additional_return_time : ℕ) : ℕ :=
  initial_time + (initial_time + additional_return_time)

/-- Theorem stating that the combined journey time is 10 hours -/
theorem alligator_journey_time : combined_journey_time 4 2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_alligator_journey_time_l1034_103461


namespace NUMINAMATH_CALUDE_base_with_five_digits_l1034_103476

theorem base_with_five_digits : ∃! b : ℕ+, b ≥ 2 ∧ b ^ 4 ≤ 500 ∧ 500 < b ^ 5 := by sorry

end NUMINAMATH_CALUDE_base_with_five_digits_l1034_103476


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l1034_103415

/-- Given that when x = 3, the value of px³ + qx + 3 is 2005, 
    prove that when x = -3, the value of px³ + qx + 3 is -1999 -/
theorem algebraic_expression_value (p q : ℝ) : 
  (3^3 * p + 3 * q + 3 = 2005) → ((-3)^3 * p + (-3) * q + 3 = -1999) := by sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l1034_103415


namespace NUMINAMATH_CALUDE_arrangement_count_is_120_l1034_103466

/-- The number of ways to arrange 4 distinct objects and 1 empty space in 5 positions -/
def arrangement_count : ℕ := 5 * 4 * 3 * 2 * 1

/-- Theorem: The number of ways to arrange 4 distinct objects and 1 empty space in 5 positions is 120 -/
theorem arrangement_count_is_120 : arrangement_count = 120 := by
  sorry

end NUMINAMATH_CALUDE_arrangement_count_is_120_l1034_103466


namespace NUMINAMATH_CALUDE_tom_twice_tim_age_l1034_103406

/-- Proves that Tom will be twice Tim's age in 3 years -/
theorem tom_twice_tim_age (tom_age tim_age : ℕ) (x : ℕ) : 
  tom_age + tim_age = 21 → 
  tom_age = 15 → 
  tom_age + x = 2 * (tim_age + x) → 
  x = 3 := by
  sorry

end NUMINAMATH_CALUDE_tom_twice_tim_age_l1034_103406


namespace NUMINAMATH_CALUDE_quadratic_integer_solution_l1034_103429

theorem quadratic_integer_solution (a : ℕ+) :
  (∃ x : ℤ, a * x^2 + 2*(2*a - 1)*x + 4*a - 7 = 0) ↔ (a = 1 ∨ a = 5) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_integer_solution_l1034_103429


namespace NUMINAMATH_CALUDE_amanda_kitchen_upgrade_l1034_103433

/-- The total cost of Amanda's kitchen upgrade after applying discounts -/
def kitchen_upgrade_cost (cabinet_knobs : ℕ) (knob_price : ℚ) (drawer_pulls : ℕ) (pull_price : ℚ) 
  (knob_discount : ℚ) (pull_discount : ℚ) : ℚ :=
  let knob_total := cabinet_knobs * knob_price
  let pull_total := drawer_pulls * pull_price
  let discounted_knob_total := knob_total * (1 - knob_discount)
  let discounted_pull_total := pull_total * (1 - pull_discount)
  discounted_knob_total + discounted_pull_total

/-- Amanda's kitchen upgrade cost is $67.70 -/
theorem amanda_kitchen_upgrade : 
  kitchen_upgrade_cost 18 (5/2) 8 4 (1/10) (3/20) = 677/10 := by
  sorry

end NUMINAMATH_CALUDE_amanda_kitchen_upgrade_l1034_103433


namespace NUMINAMATH_CALUDE_ellipse_line_intersection_collinearity_l1034_103492

-- Define the ellipse Γ
def Γ (x y : ℝ) : Prop := 3 * x^2 + 4 * y^2 = 12

-- Define the line l
def l (m q y : ℝ) : ℝ := m * y + q

-- Define the right focus of the ellipse
def F : ℝ × ℝ := (1, 0)

-- Define the condition for A₁, F, and B to be collinear
def collinear (A₁ F B : ℝ × ℝ) : Prop :=
  (F.2 - A₁.2) * (B.1 - F.1) = (B.2 - F.2) * (F.1 - A₁.1)

-- Main theorem
theorem ellipse_line_intersection_collinearity 
  (m q : ℝ) 
  (hm : m ≠ 0) 
  (A B : ℝ × ℝ) 
  (hA : Γ A.1 A.2 ∧ A.1 = l m q A.2)
  (hB : Γ B.1 B.2 ∧ B.1 = l m q B.2)
  (hAB : A ≠ B)
  (A₁ : ℝ × ℝ)
  (hA₁ : A₁ = (A.1, -A.2)) :
  (collinear A₁ F B ↔ q = 4) :=
sorry

end NUMINAMATH_CALUDE_ellipse_line_intersection_collinearity_l1034_103492


namespace NUMINAMATH_CALUDE_quadratic_form_ratio_l1034_103403

theorem quadratic_form_ratio (x : ℝ) : 
  let f : ℝ → ℝ := λ x => x^2 + 784*x + 500
  ∃ b c : ℝ, (∀ x, f x = (x + b)^2 + c) ∧ c / b = -391 := by
sorry

end NUMINAMATH_CALUDE_quadratic_form_ratio_l1034_103403


namespace NUMINAMATH_CALUDE_polynomial_factorization_l1034_103496

theorem polynomial_factorization (x : ℤ) :
  x^12 + x^6 + 1 = (x^2 + x + 1) * (x^10 - x^9 + x^8 - x^7 + x^6 - x^5 + x^4 - x^3 + x^2 - x + 1) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l1034_103496


namespace NUMINAMATH_CALUDE_train_length_l1034_103435

/-- Given a train crossing a bridge, calculate its length. -/
theorem train_length
  (train_speed : ℝ)
  (crossing_time : ℝ)
  (bridge_length : ℝ)
  (h1 : train_speed = 45)  -- km/hr
  (h2 : crossing_time = 30 / 3600)  -- convert seconds to hours
  (h3 : bridge_length = 220 / 1000)  -- convert meters to kilometers
  : (train_speed * crossing_time - bridge_length) * 1000 = 155 :=
by sorry

end NUMINAMATH_CALUDE_train_length_l1034_103435


namespace NUMINAMATH_CALUDE_discount_profit_calculation_l1034_103469

-- Define the discount percentage
def discount : ℝ := 0.05

-- Define the profit percentage without discount
def profit_without_discount : ℝ := 0.29

-- Define the function to calculate profit percentage with discount
def profit_with_discount (d : ℝ) (p : ℝ) : ℝ :=
  (1 - d) * (1 + p) - 1

-- Theorem statement
theorem discount_profit_calculation :
  abs (profit_with_discount discount profit_without_discount - 0.2255) < 0.0001 := by
  sorry

end NUMINAMATH_CALUDE_discount_profit_calculation_l1034_103469


namespace NUMINAMATH_CALUDE_overtaking_car_speed_l1034_103412

/-- Proves that given a red car traveling at 30 mph with a 20-mile head start,
    if another car overtakes it in 1 hour, the speed of the other car must be 50 mph. -/
theorem overtaking_car_speed
  (red_car_speed : ℝ)
  (red_car_lead : ℝ)
  (overtake_time : ℝ)
  (h1 : red_car_speed = 30)
  (h2 : red_car_lead = 20)
  (h3 : overtake_time = 1) :
  let other_car_speed := (red_car_speed * overtake_time + red_car_lead) / overtake_time
  other_car_speed = 50 := by
sorry

end NUMINAMATH_CALUDE_overtaking_car_speed_l1034_103412


namespace NUMINAMATH_CALUDE_line_parameterization_l1034_103474

/-- The line equation -/
def line_equation (x y : ℝ) : Prop := y = 2 * x - 7

/-- The parametric equation of the line -/
def parametric_equation (s n t x y : ℝ) : Prop :=
  x = s + 2 * t ∧ y = -3 + n * t

/-- The theorem stating the values of s and n -/
theorem line_parameterization :
  ∃ (s n : ℝ), (∀ (t x y : ℝ), parametric_equation s n t x y → line_equation x y) ∧ s = 2 ∧ n = 4 := by
  sorry

end NUMINAMATH_CALUDE_line_parameterization_l1034_103474


namespace NUMINAMATH_CALUDE_movie_replay_count_l1034_103479

theorem movie_replay_count (movie_length : Real) (ad_length : Real) (theater_hours : Real) :
  movie_length = 1.5 ∧ ad_length = 1/3 ∧ theater_hours = 11 →
  ⌊theater_hours * 60 / (movie_length * 60 + ad_length)⌋ = 6 := by
sorry

end NUMINAMATH_CALUDE_movie_replay_count_l1034_103479


namespace NUMINAMATH_CALUDE_algebraic_expression_simplification_l1034_103460

theorem algebraic_expression_simplification :
  let x : ℝ := (Real.sqrt 3) / 2 + 1 / 2
  (1 / x + (x + 1) / x) / ((x + 2) / (x^2 + x)) = (Real.sqrt 3 + 3) / 2 :=
by sorry

end NUMINAMATH_CALUDE_algebraic_expression_simplification_l1034_103460


namespace NUMINAMATH_CALUDE_simplify_nested_roots_l1034_103453

theorem simplify_nested_roots (b : ℝ) (hb : b > 0) :
  (((b^16)^(1/8))^(1/4))^3 * (((b^16)^(1/4))^(1/8))^3 = b^3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_nested_roots_l1034_103453
