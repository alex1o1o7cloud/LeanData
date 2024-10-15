import Mathlib

namespace NUMINAMATH_CALUDE_unique_solution_geometric_series_l4129_412908

theorem unique_solution_geometric_series :
  ∃! x : ℝ, x = x^3 * (1 / (1 + x)) ∧ |x| < 1 :=
by
  -- The unique solution is (√5 - 1) / 2
  sorry

end NUMINAMATH_CALUDE_unique_solution_geometric_series_l4129_412908


namespace NUMINAMATH_CALUDE_sum_of_squares_l4129_412906

theorem sum_of_squares (a b c : ℝ) 
  (h_arithmetic : (a + b + c) / 3 = 10)
  (h_geometric : (a * b * c) ^ (1/3 : ℝ) = 6)
  (h_harmonic : 3 / (1/a + 1/b + 1/c) = 4) :
  a^2 + b^2 + c^2 = 576 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_l4129_412906


namespace NUMINAMATH_CALUDE_system_solution_unique_l4129_412989

theorem system_solution_unique :
  ∃! (x y : ℝ), x + 2*y = 5 ∧ 3*x + y = 5 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_system_solution_unique_l4129_412989


namespace NUMINAMATH_CALUDE_smallest_sum_of_reciprocals_l4129_412935

theorem smallest_sum_of_reciprocals (a b : ℕ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b)
  (h : (1 : ℚ) / a + (1 : ℚ) / b = (1 : ℚ) / 15) :
  ∀ (x y : ℕ), x > 0 ∧ y > 0 ∧ x ≠ y ∧ (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / 15 → a + b ≤ x + y ∧ a + b = 64 := by
  sorry

end NUMINAMATH_CALUDE_smallest_sum_of_reciprocals_l4129_412935


namespace NUMINAMATH_CALUDE_bernardo_win_smallest_number_l4129_412955

theorem bernardo_win_smallest_number : ∃ M : ℕ, 
  (M ≤ 999) ∧ 
  (900 ≤ 72 * M) ∧ 
  (72 * M ≤ 999) ∧ 
  (∀ n : ℕ, n < M → (n ≤ 999 → 72 * n < 900 ∨ 999 < 72 * n)) ∧
  M = 13 := by
sorry

end NUMINAMATH_CALUDE_bernardo_win_smallest_number_l4129_412955


namespace NUMINAMATH_CALUDE_expected_heads_after_flips_l4129_412936

def num_coins : ℕ := 64
def max_flips : ℕ := 4

def prob_heads_single_flip : ℚ := 1 / 2

def prob_heads_multiple_flips (n : ℕ) : ℚ :=
  1 - (1 - prob_heads_single_flip) ^ n

theorem expected_heads_after_flips :
  (num_coins : ℚ) * prob_heads_multiple_flips max_flips = 60 := by
  sorry

end NUMINAMATH_CALUDE_expected_heads_after_flips_l4129_412936


namespace NUMINAMATH_CALUDE_acute_angle_sine_equivalence_l4129_412930

theorem acute_angle_sine_equivalence (α β : Real) 
  (h_α_acute : 0 < α ∧ α < Real.pi / 2)
  (h_β_acute : 0 < β ∧ β < Real.pi / 2) :
  (α > 2 * β) ↔ (Real.sin (α - β) > Real.sin β) := by
  sorry

end NUMINAMATH_CALUDE_acute_angle_sine_equivalence_l4129_412930


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l4129_412971

/-- Given a line and a circle with specific properties, prove the minimum value of 1/a + 1/b --/
theorem min_value_reciprocal_sum (a b : ℝ) : 
  a > 0 → b > 0 → 
  (∃ x y : ℝ, 2*a*x - b*y + 2 = 0 ∧ (x + 1)^2 + (y - 2)^2 = 4) →
  (∃ x₁ y₁ x₂ y₂ : ℝ, 
    2*a*x₁ - b*y₁ + 2 = 0 ∧ (x₁ + 1)^2 + (y₁ - 2)^2 = 4 ∧
    2*a*x₂ - b*y₂ + 2 = 0 ∧ (x₂ + 1)^2 + (y₂ - 2)^2 = 4 ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = 16) →
  (1/a + 1/b) ≥ 2 :=
by sorry


end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l4129_412971


namespace NUMINAMATH_CALUDE_strawberry_plants_l4129_412943

theorem strawberry_plants (initial : ℕ) : 
  (initial * 2 * 2 * 2 - 4 = 20) → initial = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_strawberry_plants_l4129_412943


namespace NUMINAMATH_CALUDE_g_x_squared_properties_l4129_412962

noncomputable def f (x : ℝ) : ℝ := (1/2)^x

def symmetric_wrt_y_eq_x (f g : ℝ → ℝ) : Prop :=
  ∀ x y, f x = y ↔ g y = x

theorem g_x_squared_properties
  (g : ℝ → ℝ)
  (h_sym : symmetric_wrt_y_eq_x f g) :
  (∀ x, g (x^2) = g ((-x)^2)) ∧
  (∀ x y, x < y → x < 0 → y < 0 → g (x^2) < g (y^2)) :=
sorry

end NUMINAMATH_CALUDE_g_x_squared_properties_l4129_412962


namespace NUMINAMATH_CALUDE_rectangle_cut_theorem_l4129_412986

theorem rectangle_cut_theorem (m : ℤ) (hm : m > 12) :
  ∃ (x y : ℕ+), (x.val : ℤ) * (y.val : ℤ) > m ∧ (x.val : ℤ) * ((y.val : ℤ) - 1) < m :=
sorry

end NUMINAMATH_CALUDE_rectangle_cut_theorem_l4129_412986


namespace NUMINAMATH_CALUDE_hyperbola_equation_l4129_412957

-- Define the parabola
def parabola (x y : ℝ) : Prop := x^2 = 20*y

-- Define the focus of the parabola
def parabola_focus : ℝ × ℝ := (0, 5)

-- Define the asymptotes of the hyperbola
def hyperbola_asymptotes (x y : ℝ) : Prop := (3*x + 4*y = 0) ∨ (3*x - 4*y = 0)

-- Define the standard form of a hyperbola
def hyperbola_standard_form (a b : ℝ) (x y : ℝ) : Prop := y^2/a^2 - x^2/b^2 = 1

-- Theorem statement
theorem hyperbola_equation :
  ∃ (x y : ℝ), parabola x y ∧
  (∃ (fx fy : ℝ), (fx, fy) = parabola_focus) ∧
  hyperbola_asymptotes x y →
  hyperbola_standard_form 3 4 x y :=
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l4129_412957


namespace NUMINAMATH_CALUDE_remainder_2753_div_98_l4129_412967

theorem remainder_2753_div_98 : 2753 % 98 = 9 := by
  sorry

end NUMINAMATH_CALUDE_remainder_2753_div_98_l4129_412967


namespace NUMINAMATH_CALUDE_brenda_spay_problem_l4129_412988

/-- The number of cats Brenda needs to spay -/
def num_cats : ℕ := 7

/-- The number of dogs Brenda needs to spay -/
def num_dogs : ℕ := 2 * num_cats

/-- The total number of animals Brenda needs to spay -/
def total_animals : ℕ := 21

theorem brenda_spay_problem :
  num_cats = 7 ∧ num_dogs = 2 * num_cats ∧ num_cats + num_dogs = total_animals :=
sorry

end NUMINAMATH_CALUDE_brenda_spay_problem_l4129_412988


namespace NUMINAMATH_CALUDE_three_squares_inequality_l4129_412972

/-- Given three equal squares arranged in a specific configuration, 
    this theorem proves that the length of the diagonal spanning two squares (AB) 
    is greater than the length of the diagonal spanning one square 
    and the side of another square (BC). -/
theorem three_squares_inequality (x : ℝ) (y : ℝ) (h1 : x > 0) (h2 : y > 0) : 
  Real.sqrt (5 * x^2 + 4 * x * y + y^2) > Real.sqrt (5 * x^2 + 2 * x * y + y^2) := by
  sorry


end NUMINAMATH_CALUDE_three_squares_inequality_l4129_412972


namespace NUMINAMATH_CALUDE_rectangular_garden_area_l4129_412925

/-- The area of a rectangular garden -/
def garden_area (length width : ℝ) : ℝ := length * width

/-- Theorem: The area of a rectangular garden with length 12 m and width 5 m is 60 square meters -/
theorem rectangular_garden_area :
  garden_area 12 5 = 60 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_garden_area_l4129_412925


namespace NUMINAMATH_CALUDE_factory_workers_count_l4129_412978

/-- Represents the number of factory workers in company J -/
def factory_workers : ℕ := sorry

/-- Represents the number of office workers in company J -/
def office_workers : ℕ := 30

/-- Represents the total monthly payroll for factory workers in dollars -/
def factory_payroll : ℕ := 30000

/-- Represents the total monthly payroll for office workers in dollars -/
def office_payroll : ℕ := 75000

/-- Represents the difference in average monthly salary between office and factory workers in dollars -/
def salary_difference : ℕ := 500

theorem factory_workers_count :
  factory_workers = 15 ∧
  factory_workers * (office_payroll / office_workers - salary_difference) = factory_payroll :=
by sorry

end NUMINAMATH_CALUDE_factory_workers_count_l4129_412978


namespace NUMINAMATH_CALUDE_rightmost_four_digits_of_7_to_2023_l4129_412981

theorem rightmost_four_digits_of_7_to_2023 :
  7^2023 ≡ 1359 [ZMOD 10000] := by
  sorry

end NUMINAMATH_CALUDE_rightmost_four_digits_of_7_to_2023_l4129_412981


namespace NUMINAMATH_CALUDE_lg_sum_equals_two_l4129_412910

-- Define lg as the logarithm with base 10
noncomputable def lg (x : ℝ) := Real.log x / Real.log 10

-- Theorem statement
theorem lg_sum_equals_two : lg 4 + lg 25 = 2 := by
  sorry

end NUMINAMATH_CALUDE_lg_sum_equals_two_l4129_412910


namespace NUMINAMATH_CALUDE_extra_hours_worked_l4129_412928

def hours_week1 : ℕ := 35
def hours_week2 : ℕ := 35
def hours_week3 : ℕ := 48
def hours_week4 : ℕ := 48

theorem extra_hours_worked : 
  (hours_week3 + hours_week4) - (hours_week1 + hours_week2) = 26 := by
  sorry

end NUMINAMATH_CALUDE_extra_hours_worked_l4129_412928


namespace NUMINAMATH_CALUDE_oliver_initial_money_l4129_412970

/-- The value of a quarter in dollars -/
def quarter_value : ℚ := 1/4

/-- Oliver's initial problem -/
theorem oliver_initial_money 
  (initial_quarters : ℕ) 
  (given_dollars : ℚ) 
  (given_quarters : ℕ) 
  (remaining_total : ℚ) :
  initial_quarters = 200 →
  given_dollars = 5 →
  given_quarters = 120 →
  remaining_total = 55 →
  (initial_quarters : ℚ) * quarter_value + 
    (given_dollars + (given_quarters : ℚ) * quarter_value + remaining_total) = 120 := by
  sorry

#eval quarter_value -- This line is to check if the definition is correct

end NUMINAMATH_CALUDE_oliver_initial_money_l4129_412970


namespace NUMINAMATH_CALUDE_segment_length_segment_length_is_eight_l4129_412979

theorem segment_length : ℝ → Prop :=
  fun length =>
    ∃ x₁ x₂ : ℝ,
      x₁ < x₂ ∧
      |x₁ - (27 : ℝ)^(1/3)| = 4 ∧
      |x₂ - (27 : ℝ)^(1/3)| = 4 ∧
      length = x₂ - x₁ ∧
      length = 8

theorem segment_length_is_eight : segment_length 8 := by
  sorry

end NUMINAMATH_CALUDE_segment_length_segment_length_is_eight_l4129_412979


namespace NUMINAMATH_CALUDE_gecko_lizard_insect_ratio_l4129_412901

theorem gecko_lizard_insect_ratio :
  let num_geckos : ℕ := 5
  let insects_per_gecko : ℕ := 6
  let num_lizards : ℕ := 3
  let total_insects : ℕ := 66
  let geckos_total := num_geckos * insects_per_gecko
  let lizards_total := total_insects - geckos_total
  let insects_per_lizard := lizards_total / num_lizards
  insects_per_lizard / insects_per_gecko = 2 :=
by sorry

end NUMINAMATH_CALUDE_gecko_lizard_insect_ratio_l4129_412901


namespace NUMINAMATH_CALUDE_jacket_final_price_l4129_412999

def original_price : ℝ := 240
def initial_discount : ℝ := 0.6
def holiday_discount : ℝ := 0.25

theorem jacket_final_price :
  let price_after_initial := original_price * (1 - initial_discount)
  let final_price := price_after_initial * (1 - holiday_discount)
  final_price = 72 := by sorry

end NUMINAMATH_CALUDE_jacket_final_price_l4129_412999


namespace NUMINAMATH_CALUDE_myrtle_has_three_hens_l4129_412912

/-- The number of hens Myrtle has -/
def num_hens : ℕ := sorry

/-- The number of eggs each hen lays per day -/
def eggs_per_hen_per_day : ℕ := 3

/-- The number of days Myrtle was gone -/
def days_gone : ℕ := 7

/-- The number of eggs the neighbor took -/
def eggs_taken_by_neighbor : ℕ := 12

/-- The number of eggs Myrtle dropped -/
def eggs_dropped : ℕ := 5

/-- The number of eggs Myrtle has remaining -/
def eggs_remaining : ℕ := 46

/-- Theorem stating that Myrtle has 3 hens -/
theorem myrtle_has_three_hens :
  num_hens = 3 :=
by sorry

end NUMINAMATH_CALUDE_myrtle_has_three_hens_l4129_412912


namespace NUMINAMATH_CALUDE_luke_weed_eating_earnings_l4129_412929

/-- Proves that Luke made $18 weed eating given the conditions of the problem -/
theorem luke_weed_eating_earnings :
  ∀ (weed_eating_earnings : ℕ),
    9 + weed_eating_earnings = 3 * 9 →
    weed_eating_earnings = 18 :=
by
  sorry

end NUMINAMATH_CALUDE_luke_weed_eating_earnings_l4129_412929


namespace NUMINAMATH_CALUDE_triple_of_negative_two_l4129_412942

theorem triple_of_negative_two : (3 : ℤ) * (-2 : ℤ) = -6 := by sorry

end NUMINAMATH_CALUDE_triple_of_negative_two_l4129_412942


namespace NUMINAMATH_CALUDE_middle_three_sum_is_twelve_l4129_412973

/-- Represents a card with a color and a number -/
inductive Card
  | red (n : Nat)
  | blue (n : Nat)

/-- Checks if a number divides another number -/
def divides (a b : Nat) : Bool :=
  b % a == 0

/-- Checks if a stack of cards satisfies the alternating color and division rules -/
def validStack (stack : List Card) : Bool :=
  match stack with
  | [] => true
  | [_] => true
  | (Card.blue b) :: (Card.red r) :: (Card.blue b') :: rest =>
      divides r b && divides r b' && validStack ((Card.red r) :: (Card.blue b') :: rest)
  | _ => false

/-- Returns the sum of the numbers on the middle three cards -/
def middleThreeSum (stack : List Card) : Nat :=
  let mid := stack.length / 2
  match (stack.get? (mid - 1), stack.get? mid, stack.get? (mid + 1)) with
  | (some (Card.blue b1), some (Card.red r), some (Card.blue b2)) => b1 + r + b2
  | _ => 0

/-- The main theorem -/
theorem middle_three_sum_is_twelve :
  ∃ (stack : List Card),
    stack.length = 9 ∧
    stack.head? = some (Card.blue 2) ∧
    (∀ n, 1 ≤ n ∧ n ≤ 4 → (Card.red n) ∈ stack) ∧
    (∀ n, 2 ≤ n ∧ n ≤ 6 → (Card.blue n) ∈ stack) ∧
    validStack stack ∧
    middleThreeSum stack = 12 :=
  sorry


end NUMINAMATH_CALUDE_middle_three_sum_is_twelve_l4129_412973


namespace NUMINAMATH_CALUDE_vertex_x_is_three_l4129_412932

/-- A quadratic function passing through specific points -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  point1 : (2 : ℝ)^2 * a + 2 * b + c = 8
  point2 : (4 : ℝ)^2 * a + 4 * b + c = 8
  point3 : c = 3

/-- The x-coordinate of the vertex of the quadratic function -/
def vertex_x (f : QuadraticFunction) : ℝ := sorry

/-- Theorem stating that the x-coordinate of the vertex is 3 -/
theorem vertex_x_is_three (f : QuadraticFunction) : vertex_x f = 3 := by sorry

end NUMINAMATH_CALUDE_vertex_x_is_three_l4129_412932


namespace NUMINAMATH_CALUDE_max_m_inequality_l4129_412996

theorem max_m_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ m : ℝ, ∀ a b : ℝ, a > 0 → b > 0 → 2/a + 1/b ≥ m/(2*a+b)) ∧
  (∀ m : ℝ, (∀ a b : ℝ, a > 0 → b > 0 → 2/a + 1/b ≥ m/(2*a+b)) → m ≤ 9) ∧
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ 2/a + 1/b = 9/(2*a+b)) :=
by sorry

end NUMINAMATH_CALUDE_max_m_inequality_l4129_412996


namespace NUMINAMATH_CALUDE_work_completion_time_l4129_412911

/-- The time it takes to complete a work given two workers with different rates and a delayed start for one worker. -/
theorem work_completion_time 
  (p_rate : ℝ) (q_rate : ℝ) (p_solo_days : ℝ) 
  (hp : p_rate = 1 / 80)
  (hq : q_rate = 1 / 48)
  (hp_solo : p_solo_days = 16) : 
  p_solo_days + (1 - p_rate * p_solo_days) / (p_rate + q_rate) = 40 := by
  sorry

#check work_completion_time

end NUMINAMATH_CALUDE_work_completion_time_l4129_412911


namespace NUMINAMATH_CALUDE_f_is_perfect_square_l4129_412924

/-- The number of ordered pairs (a,b) of positive integers such that ab/(a+b) divides N -/
def f (N : ℕ+) : ℕ := sorry

/-- f(N) is always a perfect square -/
theorem f_is_perfect_square (N : ℕ+) : ∃ (k : ℕ), f N = k ^ 2 := by sorry

end NUMINAMATH_CALUDE_f_is_perfect_square_l4129_412924


namespace NUMINAMATH_CALUDE_three_digit_addition_theorem_l4129_412926

/-- Represents a three-digit number in the form xyz --/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  ones : Nat
  h_hundreds : hundreds ≥ 1 ∧ hundreds ≤ 9
  h_tens : tens ≥ 0 ∧ tens ≤ 9
  h_ones : ones ≥ 0 ∧ ones ≤ 9

/-- Converts a ThreeDigitNumber to a natural number --/
def ThreeDigitNumber.toNat (n : ThreeDigitNumber) : Nat :=
  100 * n.hundreds + 10 * n.tens + n.ones

theorem three_digit_addition_theorem (a b : Nat) :
  let n1 : ThreeDigitNumber := ⟨4, a, 5, by sorry, by sorry, by sorry⟩
  let n2 : ThreeDigitNumber := ⟨4, 3, 8, by sorry, by sorry, by sorry⟩
  let result : ThreeDigitNumber := ⟨8, b, 3, by sorry, by sorry, by sorry⟩
  (n1.toNat + n2.toNat = result.toNat) →
  (result.toNat % 3 = 0) →
  a + b = 1 := by
  sorry

#check three_digit_addition_theorem

end NUMINAMATH_CALUDE_three_digit_addition_theorem_l4129_412926


namespace NUMINAMATH_CALUDE_golden_fish_catches_l4129_412947

theorem golden_fish_catches (x y z : ℕ) : 
  4 * x + 2 * z = 1000 →
  2 * y + z = 800 →
  x + y + z = 900 :=
by sorry

end NUMINAMATH_CALUDE_golden_fish_catches_l4129_412947


namespace NUMINAMATH_CALUDE_g_at_2_l4129_412974

-- Define the function g
def g (d : ℝ) (x : ℝ) : ℝ := 3 * x^5 - 2 * x^4 + d * x - 8

-- State the theorem
theorem g_at_2 (d : ℝ) : g d (-2) = 4 → g d 2 = -84 := by
  sorry

end NUMINAMATH_CALUDE_g_at_2_l4129_412974


namespace NUMINAMATH_CALUDE_min_sum_grid_l4129_412904

theorem min_sum_grid (a b c d : ℕ+) : 
  a + b + c + d + a * b + c * d + a * c + b * d = 2015 →
  ∃ (w x y z : ℕ+), w + x + y + z ≤ a + b + c + d ∧
                    w + x + y + z + w * x + y * z + w * y + x * z = 2015 ∧
                    w + x + y + z = 88 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_grid_l4129_412904


namespace NUMINAMATH_CALUDE_quadrant_I_condition_l4129_412995

theorem quadrant_I_condition (k : ℝ) :
  (∃ x y : ℝ, x + 2*y = 6 ∧ k*x - y = 2 ∧ x > 0 ∧ y > 0) ↔ k > 1/3 := by
  sorry

end NUMINAMATH_CALUDE_quadrant_I_condition_l4129_412995


namespace NUMINAMATH_CALUDE_second_plot_germination_rate_l4129_412966

/-- Calculates the germination rate of the second plot given the number of seeds in each plot,
    the germination rate of the first plot, and the overall germination rate. -/
theorem second_plot_germination_rate 
  (seeds_first_plot : ℕ)
  (seeds_second_plot : ℕ)
  (germination_rate_first_plot : ℚ)
  (overall_germination_rate : ℚ)
  (h1 : seeds_first_plot = 300)
  (h2 : seeds_second_plot = 200)
  (h3 : germination_rate_first_plot = 25 / 100)
  (h4 : overall_germination_rate = 27 / 100)
  : (overall_germination_rate * (seeds_first_plot + seeds_second_plot) - 
     germination_rate_first_plot * seeds_first_plot) / seeds_second_plot = 30 / 100 := by
  sorry

end NUMINAMATH_CALUDE_second_plot_germination_rate_l4129_412966


namespace NUMINAMATH_CALUDE_intersecting_linear_function_k_range_l4129_412980

/-- A linear function passing through (2, 2) and intersecting y = -x + 3 within [0, 3] -/
structure IntersectingLinearFunction where
  k : ℝ
  b : ℝ
  passes_through_2_2 : 2 * k + b = 2
  intersects_in_domain : ∃ x : ℝ, 0 ≤ x ∧ x ≤ 3 ∧ k * x + b = -x + 3

/-- The range of k values for the intersecting linear function -/
def k_range (f : IntersectingLinearFunction) : Prop :=
  (f.k ≤ -2 ∨ f.k ≥ -1/2) ∧ f.k ≠ 0

theorem intersecting_linear_function_k_range (f : IntersectingLinearFunction) :
  k_range f := by sorry

end NUMINAMATH_CALUDE_intersecting_linear_function_k_range_l4129_412980


namespace NUMINAMATH_CALUDE_circle_equation_l4129_412917

def ellipse (x y : ℝ) : Prop := x^2 / 16 + y^2 / 4 = 1

def circle_passes_through_vertices (cx cy r : ℝ) : Prop :=
  (cx - 4)^2 + cy^2 = r^2 ∧
  cx^2 + (cy - 2)^2 = r^2 ∧
  cx^2 + (cy + 2)^2 = r^2

def center_on_negative_x_axis (cx cy : ℝ) : Prop :=
  cx < 0 ∧ cy = 0

theorem circle_equation (cx cy r : ℝ) :
  ellipse 4 0 ∧ ellipse 0 2 ∧ ellipse 0 (-2) ∧
  circle_passes_through_vertices cx cy r ∧
  center_on_negative_x_axis cx cy →
  cx = -3/2 ∧ cy = 0 ∧ r = 5/2 :=
sorry

end NUMINAMATH_CALUDE_circle_equation_l4129_412917


namespace NUMINAMATH_CALUDE_solution_set_is_open_interval_l4129_412993

-- Define the quadratic function
def f (x : ℝ) := -x^2 - 3*x + 4

-- Define the solution set
def S : Set ℝ := {x | f x > 0}

-- Theorem statement
theorem solution_set_is_open_interval : S = Set.Ioo (-4 : ℝ) 1 := by sorry

end NUMINAMATH_CALUDE_solution_set_is_open_interval_l4129_412993


namespace NUMINAMATH_CALUDE_root_sum_theorem_l4129_412907

theorem root_sum_theorem (a b c : ℝ) : 
  a^3 - 30*a^2 + 65*a - 42 = 0 → 
  b^3 - 30*b^2 + 65*b - 42 = 0 → 
  c^3 - 30*c^2 + 65*c - 42 = 0 → 
  a / (1/a + b*c) + b / (1/b + c*a) + c / (1/c + a*b) = 770/43 := by
sorry

end NUMINAMATH_CALUDE_root_sum_theorem_l4129_412907


namespace NUMINAMATH_CALUDE_calculation_result_l4129_412933

theorem calculation_result : (0.0048 * 3.5) / (0.05 * 0.1 * 0.004) = 840 := by
  sorry

end NUMINAMATH_CALUDE_calculation_result_l4129_412933


namespace NUMINAMATH_CALUDE_opposite_sides_equal_implies_parallelogram_l4129_412939

/-- A quadrilateral is represented by four points in a 2D plane -/
structure Quadrilateral (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V] :=
  (A B C D : V)

/-- Definition of a parallelogram -/
def is_parallelogram {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] 
  (q : Quadrilateral V) : Prop :=
  q.A - q.B = q.D - q.C ∧ q.A - q.D = q.B - q.C

/-- Theorem: If opposite sides of a quadrilateral are equal, it is a parallelogram -/
theorem opposite_sides_equal_implies_parallelogram 
  {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] 
  (q : Quadrilateral V) 
  (h1 : q.A - q.B = q.D - q.C) 
  (h2 : q.A - q.D = q.B - q.C) : 
  is_parallelogram q :=
sorry

end NUMINAMATH_CALUDE_opposite_sides_equal_implies_parallelogram_l4129_412939


namespace NUMINAMATH_CALUDE_tens_digit_of_3_to_2013_l4129_412946

theorem tens_digit_of_3_to_2013 : ∃ n : ℕ, 3^2013 ≡ 40 + n [ZMOD 100] ∧ n < 10 := by
  sorry

end NUMINAMATH_CALUDE_tens_digit_of_3_to_2013_l4129_412946


namespace NUMINAMATH_CALUDE_parabola_focus_l4129_412918

-- Define the parabola
def parabola (x y : ℝ) : Prop := y = 4 * x^2 - 3

-- Define the focus of a parabola
def is_focus (x y : ℝ) (p : ℝ → ℝ → Prop) : Prop :=
  ∀ (px py : ℝ), p px py →
    (px - x)^2 + (py - y)^2 = (py - (y - 1/4))^2

-- Theorem statement
theorem parabola_focus :
  is_focus 0 (-47/16) parabola := by sorry

end NUMINAMATH_CALUDE_parabola_focus_l4129_412918


namespace NUMINAMATH_CALUDE_scheme2_more_cost_effective_l4129_412949

/-- The cost of a teapot in yuan -/
def teapot_cost : ℝ := 25

/-- The cost of a tea cup in yuan -/
def teacup_cost : ℝ := 5

/-- The number of teapots the customer needs to buy -/
def num_teapots : ℕ := 4

/-- The discount percentage for Scheme 2 -/
def discount_percentage : ℝ := 0.94

/-- The cost calculation for Scheme 1 -/
def scheme1_cost (x : ℝ) : ℝ := 5 * x + 80

/-- The cost calculation for Scheme 2 -/
def scheme2_cost (x : ℝ) : ℝ := (teapot_cost * num_teapots + teacup_cost * x) * discount_percentage

/-- The number of tea cups for which we want to compare the schemes -/
def x : ℝ := 47

theorem scheme2_more_cost_effective : scheme2_cost x < scheme1_cost x := by
  sorry

end NUMINAMATH_CALUDE_scheme2_more_cost_effective_l4129_412949


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l4129_412960

-- Define the quadratic function
def f (a x : ℝ) : ℝ := a * x^2 + a * x - 4

-- Define the property that the solution set is ℝ
def solution_set_is_reals (a : ℝ) : Prop :=
  ∀ x : ℝ, f a x < 0

-- Theorem statement
theorem quadratic_inequality_range :
  ∀ a : ℝ, solution_set_is_reals a → -16 < a ∧ a ≤ 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l4129_412960


namespace NUMINAMATH_CALUDE_min_n_for_infinite_moves_l4129_412937

/-- A move in the card game -/
structure Move where
  cards : Finset ℕ
  sum_equals_index : ℕ

/-- The card game setup -/
structure CardGame where
  n : ℕ
  card_count : ℕ → ℕ
  card_count_eq_n : ∀ l : ℕ, card_count l = n

/-- An infinite sequence of moves in the game -/
def InfiniteMoveSequence (game : CardGame) : Type :=
  ℕ → Move

/-- The theorem statement -/
theorem min_n_for_infinite_moves :
  ∀ n : ℕ,
  n ≥ 10000 →
  ∃ (game : CardGame) (moves : InfiniteMoveSequence game),
    (∀ k : ℕ, k > 0 → 
      (moves k).cards.card = 100 ∧
      (moves k).sum_equals_index = k) ∧
  ∀ m : ℕ,
  m < 10000 →
  ¬∃ (game : CardGame) (moves : InfiniteMoveSequence game),
    (∀ k : ℕ, k > 0 → 
      (moves k).cards.card = 100 ∧
      (moves k).sum_equals_index = k) :=
sorry

end NUMINAMATH_CALUDE_min_n_for_infinite_moves_l4129_412937


namespace NUMINAMATH_CALUDE_gcd_840_1764_l4129_412954

theorem gcd_840_1764 : Nat.gcd 840 1764 = 84 := by
  sorry

end NUMINAMATH_CALUDE_gcd_840_1764_l4129_412954


namespace NUMINAMATH_CALUDE_rectangular_solid_surface_area_l4129_412994

-- Define a structure for a rectangular solid
structure RectangularSolid where
  length : ℕ
  width : ℕ
  height : ℕ

-- Define properties of the rectangular solid
def isPrime (n : ℕ) : Prop := sorry

def volume (r : RectangularSolid) : ℕ :=
  r.length * r.width * r.height

def surfaceArea (r : RectangularSolid) : ℕ :=
  2 * (r.length * r.width + r.width * r.height + r.height * r.length)

-- Theorem statement
theorem rectangular_solid_surface_area :
  ∀ (r : RectangularSolid),
    isPrime r.length ∧ isPrime r.width ∧ isPrime r.height →
    volume r = 1155 →
    surfaceArea r = 142 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_solid_surface_area_l4129_412994


namespace NUMINAMATH_CALUDE_range_of_a_l4129_412923

def A : Set ℝ := {x | x ≤ 1 ∨ x ≥ 3}
def B (a : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ a + 1}

theorem range_of_a (a : ℝ) (h : A ∩ B a = B a) : a ≤ 0 ∨ a ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l4129_412923


namespace NUMINAMATH_CALUDE_complex_square_equality_l4129_412964

theorem complex_square_equality : (((3 : ℂ) - I) / ((1 : ℂ) + I))^2 = -3 - 4*I := by sorry

end NUMINAMATH_CALUDE_complex_square_equality_l4129_412964


namespace NUMINAMATH_CALUDE_care_package_weight_ratio_l4129_412931

/-- Represents the weight of the care package at different stages --/
structure CarePackage where
  initial_weight : ℝ
  after_brownies : ℝ
  before_gummies : ℝ
  final_weight : ℝ

/-- Theorem stating the ratio of final weight to weight before gummies is 2:1 --/
theorem care_package_weight_ratio (package : CarePackage) : 
  package.initial_weight = 2 →
  package.after_brownies = 3 * package.initial_weight →
  package.before_gummies = package.after_brownies + 2 →
  package.final_weight = 16 →
  package.final_weight / package.before_gummies = 2 := by
  sorry


end NUMINAMATH_CALUDE_care_package_weight_ratio_l4129_412931


namespace NUMINAMATH_CALUDE_tricycle_wheels_l4129_412915

theorem tricycle_wheels (num_bicycles : ℕ) (num_tricycles : ℕ) (total_wheels : ℕ) :
  num_bicycles = 24 →
  num_tricycles = 14 →
  total_wheels = 90 →
  ∃ (tricycle_wheels : ℕ),
    tricycle_wheels = 3 ∧
    total_wheels = num_bicycles * 2 + num_tricycles * tricycle_wheels :=
by
  sorry

end NUMINAMATH_CALUDE_tricycle_wheels_l4129_412915


namespace NUMINAMATH_CALUDE_unique_scalar_for_vector_equation_l4129_412913

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

def cross_product : V → V → V := sorry

theorem unique_scalar_for_vector_equation
  (cross_product : V → V → V)
  (h_cross_product : ∀ (x y z : V) (r : ℝ),
    cross_product (r • x) y = r • cross_product x y ∧
    cross_product x y = -cross_product y x ∧
    cross_product (x + y) z = cross_product x z + cross_product y z) :
  ∃! k : ℝ, ∀ (a b c d : V),
    a + b + c + d = 0 →
    k • (cross_product b a) + cross_product b c + cross_product c a + cross_product d a = 0 :=
by sorry

end NUMINAMATH_CALUDE_unique_scalar_for_vector_equation_l4129_412913


namespace NUMINAMATH_CALUDE_wedge_product_formula_l4129_412944

/-- The wedge product of two 2D vectors -/
def wedge_product (a b : ℝ × ℝ) : ℝ :=
  a.1 * b.2 - a.2 * b.1

/-- Theorem: The wedge product of two 2D vectors (a₁, a₂) and (b₁, b₂) is a₁b₂ - a₂b₁ -/
theorem wedge_product_formula (a b : ℝ × ℝ) : 
  wedge_product a b = a.1 * b.2 - a.2 * b.1 := by
  sorry

end NUMINAMATH_CALUDE_wedge_product_formula_l4129_412944


namespace NUMINAMATH_CALUDE_largest_remainder_2015_l4129_412987

theorem largest_remainder_2015 : 
  ∀ d : ℕ, 1 ≤ d ∧ d ≤ 1000 → (2015 % d) ≤ 671 ∧ ∃ d₀ : ℕ, 1 ≤ d₀ ∧ d₀ ≤ 1000 ∧ 2015 % d₀ = 671 :=
by sorry

end NUMINAMATH_CALUDE_largest_remainder_2015_l4129_412987


namespace NUMINAMATH_CALUDE_unique_solution_l4129_412990

/-- The function f(x) = x^2 + 4x + 3 -/
def f (x : ℤ) : ℤ := x^2 + 4*x + 3

/-- The function g(x) = x^2 + 2x - 1 -/
def g (x : ℤ) : ℤ := x^2 + 2*x - 1

/-- Theorem stating that x = -2 is the only integer solution to f(g(f(x))) = g(f(g(x))) -/
theorem unique_solution :
  ∃! x : ℤ, f (g (f x)) = g (f (g x)) ∧ x = -2 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l4129_412990


namespace NUMINAMATH_CALUDE_diane_age_is_16_l4129_412977

/-- Represents the current ages of Diane, Alex, and Allison -/
structure Ages where
  diane : ℕ
  alex : ℕ
  allison : ℕ

/-- Conditions of the problem -/
def problem_conditions (ages : Ages) : Prop :=
  ages.alex + ages.allison = 47 ∧
  ages.alex + (30 - ages.diane) = 60 ∧
  ages.allison + (30 - ages.diane) = 15

/-- Theorem stating that Diane's current age is 16 -/
theorem diane_age_is_16 :
  ∃ (ages : Ages), problem_conditions ages ∧ ages.diane = 16 :=
sorry

end NUMINAMATH_CALUDE_diane_age_is_16_l4129_412977


namespace NUMINAMATH_CALUDE_sum_of_x_and_y_l4129_412948

theorem sum_of_x_and_y (x y : ℝ) (hx : |x| = 3) (hy : |y| = 4) (hxy : x * y > 0) :
  x + y = 7 ∨ x + y = -7 := by sorry

end NUMINAMATH_CALUDE_sum_of_x_and_y_l4129_412948


namespace NUMINAMATH_CALUDE_value_of_a_l4129_412914

theorem value_of_a (a b c : ℝ) (h1 : a + b = c) (h2 : b + c = 6) (h3 : c = 4) : a = 2 := by
  sorry

end NUMINAMATH_CALUDE_value_of_a_l4129_412914


namespace NUMINAMATH_CALUDE_polynomial_sum_and_coefficient_sum_l4129_412922

theorem polynomial_sum_and_coefficient_sum (d : ℝ) (h : d ≠ 0) :
  (15 * d^3 + 12 * d + 7 + 18 * d^2) + (2 * d^3 + d - 6 + 3 * d^2) =
  17 * d^3 + 21 * d^2 + 13 * d + 1 ∧
  17 + 21 + 13 + 1 = 52 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_sum_and_coefficient_sum_l4129_412922


namespace NUMINAMATH_CALUDE_floor_difference_equals_five_l4129_412963

theorem floor_difference_equals_five (n : ℤ) : 
  (Int.floor (n^2 / 4 : ℚ) - Int.floor (n / 2 : ℚ)^2 = 5) → n = 11 := by
  sorry

end NUMINAMATH_CALUDE_floor_difference_equals_five_l4129_412963


namespace NUMINAMATH_CALUDE_code_decryption_probability_l4129_412927

theorem code_decryption_probability :
  let p := 1 / 5  -- probability of success for each person
  let n := 3      -- number of people
  let prob_at_least_two := 
    Finset.sum (Finset.range (n - 1 + 1)) (fun k => 
      if k ≥ 2 then Nat.choose n k * p^k * (1 - p)^(n - k) else 0)
  prob_at_least_two = 13 / 125 := by
sorry

end NUMINAMATH_CALUDE_code_decryption_probability_l4129_412927


namespace NUMINAMATH_CALUDE_triangle_cosine_B_l4129_412945

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- State the theorem
theorem triangle_cosine_B (abc : Triangle) 
  (h1 : abc.b * Real.sin abc.B - abc.a * Real.sin abc.A = (1/2) * abc.a * Real.sin abc.C)
  (h2 : (1/2) * abc.a * abc.c * Real.sin abc.B = abc.a^2 * Real.sin abc.B) :
  Real.cos abc.B = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_cosine_B_l4129_412945


namespace NUMINAMATH_CALUDE_opposing_team_score_l4129_412940

theorem opposing_team_score (chucks_team_score : ℕ) (lead : ℕ) (opposing_team_score : ℕ) :
  chucks_team_score = 72 →
  lead = 17 →
  chucks_team_score = opposing_team_score + lead →
  opposing_team_score = 55 := by
sorry

end NUMINAMATH_CALUDE_opposing_team_score_l4129_412940


namespace NUMINAMATH_CALUDE_divisibility_by_nineteen_l4129_412956

theorem divisibility_by_nineteen (n : ℕ+) :
  ∃ k : ℤ, (5 ^ (2 * n.val - 1) : ℤ) + (3 ^ (n.val - 2) : ℤ) * (2 ^ (n.val - 1) : ℤ) = 19 * k :=
by sorry

end NUMINAMATH_CALUDE_divisibility_by_nineteen_l4129_412956


namespace NUMINAMATH_CALUDE_scrabble_score_calculation_l4129_412903

/-- Scrabble game score calculation -/
theorem scrabble_score_calculation 
  (brenda_turn1 : ℕ) 
  (david_turn1 : ℕ) 
  (brenda_turn2 : ℕ) 
  (david_turn2 : ℕ) 
  (brenda_lead_before_turn3 : ℕ) 
  (brenda_turn3 : ℕ) 
  (david_turn3 : ℕ) 
  (h1 : brenda_turn1 = 18) 
  (h2 : david_turn1 = 10) 
  (h3 : brenda_turn2 = 25) 
  (h4 : david_turn2 = 35) 
  (h5 : brenda_lead_before_turn3 = 22) 
  (h6 : brenda_turn3 = 15) 
  (h7 : david_turn3 = 32) : 
  (david_turn1 + david_turn2 + david_turn3) - (brenda_turn1 + brenda_turn2 + brenda_turn3) = 19 := by
  sorry

end NUMINAMATH_CALUDE_scrabble_score_calculation_l4129_412903


namespace NUMINAMATH_CALUDE_roots_shifted_polynomial_l4129_412921

theorem roots_shifted_polynomial (a b c : ℂ) : 
  (∀ x : ℂ, x^3 - 5*x + 7 = 0 ↔ x = a ∨ x = b ∨ x = c) →
  (∀ x : ℂ, x^3 + 9*x^2 + 22*x + 19 = 0 ↔ x = a - 3 ∨ x = b - 3 ∨ x = c - 3) :=
by sorry

end NUMINAMATH_CALUDE_roots_shifted_polynomial_l4129_412921


namespace NUMINAMATH_CALUDE_candle_scent_ratio_l4129_412902

/-- Represents the number of candles made for each scent -/
structure CandleCounts where
  coconut : ℕ
  lavender : ℕ
  almond : ℕ

/-- Represents the amount of scent used for each type -/
structure ScentAmounts where
  coconut : ℝ
  lavender : ℝ
  almond : ℝ

/-- The theorem stating the relationship between candle counts and scent amounts -/
theorem candle_scent_ratio 
  (counts : CandleCounts) 
  (amounts : ScentAmounts) 
  (h1 : counts.lavender = 2 * counts.coconut) 
  (h2 : counts.almond = 10) 
  (h3 : ∀ (s : ScentAmounts), s.coconut = s.lavender ∧ s.coconut = s.almond) : 
  amounts.coconut / amounts.almond = counts.coconut / counts.almond :=
sorry

end NUMINAMATH_CALUDE_candle_scent_ratio_l4129_412902


namespace NUMINAMATH_CALUDE_arithmetic_sequence_a8_l4129_412916

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The theorem to be proved -/
theorem arithmetic_sequence_a8 (a : ℕ → ℚ) 
  (h_arith : arithmetic_sequence a)
  (h_sum : a 4 + a 6 = 8)
  (h_a2 : a 2 = 3) :
  a 8 = 5 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_a8_l4129_412916


namespace NUMINAMATH_CALUDE_ones_digit_of_large_power_l4129_412982

theorem ones_digit_of_large_power : ∃ n : ℕ, 34^(11^34) ≡ 4 [ZMOD 10] :=
  sorry

end NUMINAMATH_CALUDE_ones_digit_of_large_power_l4129_412982


namespace NUMINAMATH_CALUDE_different_color_probability_l4129_412975

def shorts_colors : ℕ := 3
def jersey_colors : ℕ := 4

theorem different_color_probability :
  let total_combinations := shorts_colors * jersey_colors
  let different_color_combinations := total_combinations - shorts_colors
  (different_color_combinations : ℚ) / total_combinations = 3 / 4 := by
sorry

end NUMINAMATH_CALUDE_different_color_probability_l4129_412975


namespace NUMINAMATH_CALUDE_tetrahedron_altitude_exsphere_relation_l4129_412976

/-- A tetrahedron with its altitudes and exsphere radii -/
structure Tetrahedron where
  h₁ : ℝ
  h₂ : ℝ
  h₃ : ℝ
  h₄ : ℝ
  r₁ : ℝ
  r₂ : ℝ
  r₃ : ℝ
  r₄ : ℝ
  h_pos : h₁ > 0 ∧ h₂ > 0 ∧ h₃ > 0 ∧ h₄ > 0
  r_pos : r₁ > 0 ∧ r₂ > 0 ∧ r₃ > 0 ∧ r₄ > 0

/-- The theorem about the relationship between altitudes and exsphere radii in a tetrahedron -/
theorem tetrahedron_altitude_exsphere_relation (t : Tetrahedron) :
  2 * (1 / t.h₁ + 1 / t.h₂ + 1 / t.h₃ + 1 / t.h₄) =
  1 / t.r₁ + 1 / t.r₂ + 1 / t.r₃ + 1 / t.r₄ := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_altitude_exsphere_relation_l4129_412976


namespace NUMINAMATH_CALUDE_quadratic_roots_reciprocal_sum_l4129_412938

theorem quadratic_roots_reciprocal_sum (x₁ x₂ : ℝ) : 
  x₁^2 - 2*x₁ - 5 = 0 → 
  x₂^2 - 2*x₂ - 5 = 0 → 
  x₁ ≠ 0 → 
  x₂ ≠ 0 → 
  1/x₁ + 1/x₂ = -2/5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_reciprocal_sum_l4129_412938


namespace NUMINAMATH_CALUDE_rational_absolute_difference_sum_l4129_412900

theorem rational_absolute_difference_sum (a b : ℚ) : 
  |a - b| = a + b → a ≥ 0 ∧ b ≥ 0 := by sorry

end NUMINAMATH_CALUDE_rational_absolute_difference_sum_l4129_412900


namespace NUMINAMATH_CALUDE_fred_bought_two_tickets_l4129_412965

/-- The number of tickets Fred bought -/
def num_tickets : ℕ := 2

/-- The price of each ticket in cents -/
def ticket_price : ℕ := 592

/-- The cost of borrowing a movie in cents -/
def movie_rental : ℕ := 679

/-- The amount Fred paid in cents -/
def amount_paid : ℕ := 2000

/-- The change Fred received in cents -/
def change_received : ℕ := 137

/-- Theorem stating that Fred bought 2 tickets given the conditions -/
theorem fred_bought_two_tickets :
  num_tickets * ticket_price + movie_rental = amount_paid - change_received :=
by sorry

end NUMINAMATH_CALUDE_fred_bought_two_tickets_l4129_412965


namespace NUMINAMATH_CALUDE_sum_of_fourth_powers_l4129_412920

theorem sum_of_fourth_powers (a b c : ℝ) 
  (h1 : (a^2 - b^2) + c^2 = 8)
  (h2 : a * b * c = 2) :
  a^4 + b^4 + c^4 = 70 := by
sorry

end NUMINAMATH_CALUDE_sum_of_fourth_powers_l4129_412920


namespace NUMINAMATH_CALUDE_swim_team_girls_count_l4129_412951

theorem swim_team_girls_count (total : ℕ) (ratio : ℕ) (boys : ℕ) (girls : ℕ) : 
  total = 96 → 
  ratio = 5 → 
  girls = ratio * boys → 
  total = girls + boys → 
  girls = 80 :=
by
  sorry

end NUMINAMATH_CALUDE_swim_team_girls_count_l4129_412951


namespace NUMINAMATH_CALUDE_f_properties_l4129_412991

noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x) - 2 * Real.sqrt 3 * Real.sin x * Real.cos x

theorem f_properties :
  (∀ x, f x ≤ 2) ∧ 
  (f (7 * Real.pi / 12) = 0) := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l4129_412991


namespace NUMINAMATH_CALUDE_xyz_equals_two_l4129_412919

theorem xyz_equals_two
  (a b c x y z : ℂ)
  (nonzero_a : a ≠ 0)
  (nonzero_b : b ≠ 0)
  (nonzero_c : c ≠ 0)
  (nonzero_x : x ≠ 0)
  (nonzero_y : y ≠ 0)
  (nonzero_z : z ≠ 0)
  (eq_a : a = (b + c) / (x - 3))
  (eq_b : b = (a + c) / (y - 3))
  (eq_c : c = (a + b) / (z - 3))
  (sum_xy_xz_yz : x * y + x * z + y * z = 7)
  (sum_x_y_z : x + y + z = 3) :
  x * y * z = 2 := by
sorry


end NUMINAMATH_CALUDE_xyz_equals_two_l4129_412919


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_equation_l4129_412953

theorem sum_of_roots_quadratic_equation :
  let a : ℝ := 6 + 3 * Real.sqrt 3
  let b : ℝ := 3 + Real.sqrt 3
  let c : ℝ := -3
  let sum_of_roots := -b / a
  sum_of_roots = -1 + Real.sqrt 3 / 3 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_equation_l4129_412953


namespace NUMINAMATH_CALUDE_complex_fraction_evaluation_l4129_412909

theorem complex_fraction_evaluation : 
  (0.125 / 0.25 + (1 + 9/16) / 2.5) / 
  ((10 - 22 / 2.3) * 0.46 + 1.6) + 
  (17/20 + 1.9) * 0.5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_evaluation_l4129_412909


namespace NUMINAMATH_CALUDE_sunny_cake_candles_l4129_412961

/-- Given the initial number of cakes, number of cakes given away, and total candles used,
    calculate the number of candles on each remaining cake. -/
def candles_per_cake (initial_cakes : ℕ) (cakes_given_away : ℕ) (total_candles : ℕ) : ℕ :=
  total_candles / (initial_cakes - cakes_given_away)

/-- Prove that given the specific values in the problem, 
    the number of candles on each remaining cake is 6. -/
theorem sunny_cake_candles : 
  candles_per_cake 8 2 36 = 6 := by sorry

end NUMINAMATH_CALUDE_sunny_cake_candles_l4129_412961


namespace NUMINAMATH_CALUDE_trig_identity_proof_l4129_412959

theorem trig_identity_proof : 
  Real.sin (20 * π / 180) * Real.cos (10 * π / 180) + 
  Real.sin (10 * π / 180) * Real.sin (70 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_proof_l4129_412959


namespace NUMINAMATH_CALUDE_complex_square_equality_l4129_412969

theorem complex_square_equality (a b : ℝ) (i : ℂ) (h : i^2 = -1) 
  (eq : (a : ℂ) + b*i - 2*i = 2 - b*i) : 
  (a + b*i)^2 = 3 + 4*i := by
  sorry

end NUMINAMATH_CALUDE_complex_square_equality_l4129_412969


namespace NUMINAMATH_CALUDE_problem_one_problem_two_l4129_412905

-- Problem 1
theorem problem_one : (-1/3)⁻¹ + Real.sqrt 9 + (2 - Real.pi)^0 = 1 := by sorry

-- Problem 2
theorem problem_two (a : ℝ) (ha : a ≠ 0) (ha' : a ≠ 1) : 
  ((1/a - 1) / ((a^2 - 2*a + 1) / a)) = 1 / (1 - a) := by sorry

end NUMINAMATH_CALUDE_problem_one_problem_two_l4129_412905


namespace NUMINAMATH_CALUDE_largest_common_divisor_l4129_412950

theorem largest_common_divisor : 
  let a := 924
  let b := 1386
  let c := 462
  Nat.gcd a (Nat.gcd b c) = 462 := by
sorry

end NUMINAMATH_CALUDE_largest_common_divisor_l4129_412950


namespace NUMINAMATH_CALUDE_quadratic_property_l4129_412997

/-- A quadratic function f(x) = ax² + bx + c with specific conditions -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  h1 : a > b
  h2 : b > c
  h3 : a + b + c = 0

/-- The set A of m where f(m) < 0 -/
def A (f : QuadraticFunction) : Set ℝ :=
  {m | f.a * m^2 + f.b * m + f.c < 0}

/-- Main theorem: For any m in A, f(m+3) > 0 -/
theorem quadratic_property (f : QuadraticFunction) :
  ∀ m ∈ A f, f.a * (m + 3)^2 + f.b * (m + 3) + f.c > 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_property_l4129_412997


namespace NUMINAMATH_CALUDE_sqrt_sum_rational_form_l4129_412952

theorem sqrt_sum_rational_form :
  ∃ (p q r : ℕ+), 
    (Real.sqrt 6 + (Real.sqrt 6)⁻¹ + Real.sqrt 8 + (Real.sqrt 8)⁻¹ = (p * Real.sqrt 6 + q * Real.sqrt 8) / r) ∧
    (∀ (p' q' r' : ℕ+), 
      (Real.sqrt 6 + (Real.sqrt 6)⁻¹ + Real.sqrt 8 + (Real.sqrt 8)⁻¹ = (p' * Real.sqrt 6 + q' * Real.sqrt 8) / r') →
      r ≤ r') ∧
    (p + q + r = 19) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_sum_rational_form_l4129_412952


namespace NUMINAMATH_CALUDE_triangle_inequality_violation_l4129_412992

theorem triangle_inequality_violation
  (a b c d e : ℝ)
  (distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e)
  (positive : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ 0 < e)
  (sum_equality : a^2 + b^2 + c^2 + d^2 + e^2 = a*b + a*c + a*d + a*e + b*c + b*d + b*e + c*d + c*e + d*e) :
  ∃ (x y z : ℝ), (x = a ∧ y = b ∧ z = c) ∨ (x = a ∧ y = b ∧ z = d) ∨ (x = a ∧ y = b ∧ z = e) ∨
                 (x = a ∧ y = c ∧ z = d) ∨ (x = a ∧ y = c ∧ z = e) ∨ (x = a ∧ y = d ∧ z = e) ∨
                 (x = b ∧ y = c ∧ z = d) ∨ (x = b ∧ y = c ∧ z = e) ∨ (x = b ∧ y = d ∧ z = e) ∨
                 (x = c ∧ y = d ∧ z = e) ∧
                 (x + y ≤ z ∨ y + z ≤ x ∨ z + x ≤ y) :=
by sorry

end NUMINAMATH_CALUDE_triangle_inequality_violation_l4129_412992


namespace NUMINAMATH_CALUDE_probability_of_selection_X_l4129_412998

theorem probability_of_selection_X (p_Y p_XY : ℝ) : 
  p_Y = 2/3 → p_XY = 0.13333333333333333 → ∃ p_X : ℝ, p_X = 0.2 ∧ p_XY = p_X * p_Y :=
by sorry

end NUMINAMATH_CALUDE_probability_of_selection_X_l4129_412998


namespace NUMINAMATH_CALUDE_complex_equation_solution_l4129_412984

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_equation_solution :
  ∃ (x : ℂ), (2 : ℂ) + 3 * i * x = (4 : ℂ) - 5 * i * x ∧ x = i / 4 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l4129_412984


namespace NUMINAMATH_CALUDE_walnut_trees_after_planting_l4129_412983

/-- The number of walnut trees in the park after planting -/
def total_walnut_trees (initial : ℕ) (planted : ℕ) : ℕ :=
  initial + planted

/-- Theorem stating the total number of walnut trees after planting -/
theorem walnut_trees_after_planting :
  total_walnut_trees 22 33 = 55 := by
  sorry

end NUMINAMATH_CALUDE_walnut_trees_after_planting_l4129_412983


namespace NUMINAMATH_CALUDE_quiz_logic_l4129_412985

-- Define the universe of discourse
variable (Student : Type)

-- Define predicates
variable (answers_all_correctly : Student → Prop)
variable (passes_quiz : Student → Prop)

-- State the theorem
theorem quiz_logic (s : Student) 
  (h : ∀ x : Student, answers_all_correctly x → passes_quiz x) :
  ¬(passes_quiz s) → ¬(answers_all_correctly s) :=
by
  sorry

end NUMINAMATH_CALUDE_quiz_logic_l4129_412985


namespace NUMINAMATH_CALUDE_lens_savings_l4129_412958

theorem lens_savings (original_price : ℝ) (discount_rate : ℝ) (cheaper_price : ℝ) : 
  original_price = 300 ∧ 
  discount_rate = 0.20 ∧ 
  cheaper_price = 220 → 
  original_price * (1 - discount_rate) - cheaper_price = 20 := by
sorry

end NUMINAMATH_CALUDE_lens_savings_l4129_412958


namespace NUMINAMATH_CALUDE_parallel_lines_c_value_l4129_412941

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_lines_equal_slopes {m₁ m₂ b₁ b₂ : ℝ} :
  (∀ x y : ℝ, y = m₁ * x + b₁ ↔ y = m₂ * x + b₂) ↔ m₁ = m₂

/-- The value of c that makes the given lines parallel -/
theorem parallel_lines_c_value :
  (∀ x y : ℝ, y = (5/2) * x + 5 ↔ y = (3 * c) * x + 3) → c = 5/6 :=
by sorry

end NUMINAMATH_CALUDE_parallel_lines_c_value_l4129_412941


namespace NUMINAMATH_CALUDE_set_representation_implies_sum_of_powers_l4129_412934

theorem set_representation_implies_sum_of_powers (a b : ℝ) : 
  ({a, 1, b/a} : Set ℝ) = {a+b, 0, a^2} → a^2010 + b^2010 = 1 := by
  sorry

end NUMINAMATH_CALUDE_set_representation_implies_sum_of_powers_l4129_412934


namespace NUMINAMATH_CALUDE_orange_cost_l4129_412968

/-- If 4 dozen oranges cost $24.00, then 6 dozen oranges at the same rate will cost $36.00. -/
theorem orange_cost (initial_cost : ℝ) (initial_quantity : ℕ) (target_quantity : ℕ) : 
  initial_cost = 24 ∧ initial_quantity = 4 ∧ target_quantity = 6 →
  (target_quantity : ℝ) * (initial_cost / initial_quantity) = 36 := by
sorry

end NUMINAMATH_CALUDE_orange_cost_l4129_412968
