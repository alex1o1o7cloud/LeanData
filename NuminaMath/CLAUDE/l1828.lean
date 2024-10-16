import Mathlib

namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l1828_182868

theorem sqrt_equation_solution (y : ℝ) : 
  (Real.sqrt 1.21) / (Real.sqrt y) + (Real.sqrt 1.44) / (Real.sqrt 0.49) = 2.9365079365079367 → 
  y = 0.81 := by
sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l1828_182868


namespace NUMINAMATH_CALUDE_trig_identity_l1828_182832

theorem trig_identity (α : Real) 
  (h : Real.cos (π / 6 - α) = Real.sqrt 3 / 3) : 
  Real.cos (5 * π / 6 + α) + Real.sin (α - π / 6) ^ 2 = (2 - Real.sqrt 3) / 3 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l1828_182832


namespace NUMINAMATH_CALUDE_circle_x_axis_intersection_sum_l1828_182848

/-- The sum of x-coordinates of intersection points between a circle and the x-axis -/
def sum_x_coordinates (h k r : ℝ) : ℝ :=
  2 * h

/-- Theorem: For a circle with center (3, -5) and radius 7, 
    the sum of x-coordinates of its intersection points with the x-axis is 6 -/
theorem circle_x_axis_intersection_sum :
  sum_x_coordinates 3 (-5) 7 = 6 := by
  sorry


end NUMINAMATH_CALUDE_circle_x_axis_intersection_sum_l1828_182848


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l1828_182826

theorem triangle_abc_properties (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ A < π →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  2 * Real.cos A * (b * Real.cos C + c * Real.cos B) = a →
  Real.cos B = 3 / 5 →
  A = π / 3 ∧ Real.sin (B - C) = (7 * Real.sqrt 3 - 24) / 50 := by
  sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l1828_182826


namespace NUMINAMATH_CALUDE_three_card_sequence_l1828_182824

-- Define the ranks and suits
inductive Rank
| Ace | King

inductive Suit
| Heart | Diamond

-- Define a card as a pair of rank and suit
structure Card :=
  (rank : Rank)
  (suit : Suit)

def is_king (c : Card) : Prop := c.rank = Rank.King
def is_ace (c : Card) : Prop := c.rank = Rank.Ace
def is_heart (c : Card) : Prop := c.suit = Suit.Heart
def is_diamond (c : Card) : Prop := c.suit = Suit.Diamond

-- Define the theorem
theorem three_card_sequence (c1 c2 c3 : Card) : 
  -- Condition 1
  (is_king c2 ∨ is_king c3) ∧ is_ace c1 →
  -- Condition 2
  (is_king c1 ∨ is_king c2) ∧ is_king c3 →
  -- Condition 3
  (is_heart c1 ∨ is_heart c2) ∧ is_diamond c3 →
  -- Condition 4
  is_heart c1 ∧ (is_heart c2 ∨ is_heart c3) →
  -- Conclusion
  is_heart c1 ∧ is_ace c1 ∧ 
  is_heart c2 ∧ is_king c2 ∧
  is_diamond c3 ∧ is_king c3 := by
  sorry


end NUMINAMATH_CALUDE_three_card_sequence_l1828_182824


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l1828_182854

theorem complex_fraction_simplification :
  let z : ℂ := (4 - 9*I) / (3 + 4*I)
  z = -24/25 - 43/25*I :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l1828_182854


namespace NUMINAMATH_CALUDE_horner_v2_value_l1828_182891

def horner_step (v : ℝ) (x : ℝ) (a : ℝ) : ℝ := v * x + a

def horner_method (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (λ acc coeff => horner_step acc x coeff) 0

def polynomial : List ℝ := [5, 2, 3.5, -2.6, 1.7, -0.8]

theorem horner_v2_value :
  let x : ℝ := 5
  let v0 : ℝ := polynomial.head!
  let v1 : ℝ := horner_step v0 x (polynomial.get! 1)
  let v2 : ℝ := horner_step v1 x (polynomial.get! 2)
  v2 = 138.5 := by sorry

end NUMINAMATH_CALUDE_horner_v2_value_l1828_182891


namespace NUMINAMATH_CALUDE_closest_integer_to_cube_root_closest_integer_to_cube_root_of_sum_of_cubes_l1828_182820

theorem closest_integer_to_cube_root (x : ℝ) : 
  ∃ n : ℤ, ∀ m : ℤ, |x - n| ≤ |x - m| := by sorry

theorem closest_integer_to_cube_root_of_sum_of_cubes : 
  ∃ n : ℤ, (∀ m : ℤ, |((7 : ℝ)^3 + 9^3)^(1/3) - n| ≤ |((7 : ℝ)^3 + 9^3)^(1/3) - m|) ∧ n = 10 := by sorry

end NUMINAMATH_CALUDE_closest_integer_to_cube_root_closest_integer_to_cube_root_of_sum_of_cubes_l1828_182820


namespace NUMINAMATH_CALUDE_simplify_expression_l1828_182843

/-- Proves that the simplified expression (√3 - 1)^(1 - √2) / (√3 + 1)^(1 + √2) equals 2^(1-√2)(4 - 2√3) -/
theorem simplify_expression :
  let x := (Real.sqrt 3 - 1)^(1 - Real.sqrt 2) / (Real.sqrt 3 + 1)^(1 + Real.sqrt 2)
  let y := 2^(1 - Real.sqrt 2) * (4 - 2 * Real.sqrt 3)
  x = y := by sorry

end NUMINAMATH_CALUDE_simplify_expression_l1828_182843


namespace NUMINAMATH_CALUDE_largest_prime_divisor_check_l1828_182838

theorem largest_prime_divisor_check (n : ℕ) : 
  1200 ≤ n ∧ n ≤ 1250 → 
  (∀ p : ℕ, p.Prime → p ≤ 31 → n % p ≠ 0) → 
  n.Prime := by
  sorry

end NUMINAMATH_CALUDE_largest_prime_divisor_check_l1828_182838


namespace NUMINAMATH_CALUDE_nadine_garage_sale_spend_l1828_182899

/-- The amount Nadine spent at the garage sale -/
def garage_sale_total (table_price chair_price num_chairs : ℕ) : ℕ :=
  table_price + chair_price * num_chairs

/-- Theorem: Nadine spent $56 at the garage sale -/
theorem nadine_garage_sale_spend :
  garage_sale_total 34 11 2 = 56 := by
  sorry

end NUMINAMATH_CALUDE_nadine_garage_sale_spend_l1828_182899


namespace NUMINAMATH_CALUDE_no_solution_for_equation_l1828_182846

theorem no_solution_for_equation :
  ∀ x : ℝ, x ≠ 1 → x ≠ 0 → (6 / (x - 1) - (x + 5) / (x^2 - x) ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_for_equation_l1828_182846


namespace NUMINAMATH_CALUDE_rectangle_area_l1828_182813

/-- Given a rectangle where the length is thrice the breadth and the perimeter is 40,
    prove that its area is 75. -/
theorem rectangle_area (b : ℝ) (h1 : b > 0) : 
  let l := 3 * b
  let p := 2 * (l + b)
  p = 40 →
  l * b = 75 := by sorry

end NUMINAMATH_CALUDE_rectangle_area_l1828_182813


namespace NUMINAMATH_CALUDE_adjacent_vertices_probability_l1828_182851

/-- A decagon is a polygon with 10 vertices -/
def Decagon := {n : ℕ // n = 10}

/-- The number of vertices in a decagon -/
def numVertices : Decagon → ℕ := fun _ ↦ 10

/-- The number of adjacent vertices for each vertex in a decagon -/
def numAdjacentVertices : Decagon → ℕ := fun _ ↦ 2

/-- The probability of selecting two adjacent vertices in a decagon -/
def probAdjacentVertices (d : Decagon) : ℚ :=
  (numAdjacentVertices d : ℚ) / ((numVertices d - 1) : ℚ)

theorem adjacent_vertices_probability (d : Decagon) :
  probAdjacentVertices d = 2/9 := by sorry

end NUMINAMATH_CALUDE_adjacent_vertices_probability_l1828_182851


namespace NUMINAMATH_CALUDE_fraction_decimal_digits_l1828_182867

/-- The fraction we're considering -/
def fraction : ℚ := 987654321 / (2^30 * 5^2 * 3)

/-- The minimum number of digits to the right of the decimal point -/
def min_decimal_digits : ℕ := 30

/-- Theorem stating that the minimum number of digits to the right of the decimal point
    needed to express the fraction as a decimal is equal to min_decimal_digits -/
theorem fraction_decimal_digits :
  (∀ n : ℕ, n < min_decimal_digits → ∃ m : ℕ, fraction * 10^n ≠ m) ∧
  (∃ m : ℕ, fraction * 10^min_decimal_digits = m) :=
sorry

end NUMINAMATH_CALUDE_fraction_decimal_digits_l1828_182867


namespace NUMINAMATH_CALUDE_negation_of_forall_positive_negation_of_greater_than_zero_l1828_182801

theorem negation_of_forall_positive (p : ℝ → Prop) : 
  (¬ ∀ x : ℝ, p x) ↔ (∃ x : ℝ, ¬ p x) :=
by sorry

theorem negation_of_greater_than_zero :
  (¬ ∀ x : ℝ, x^2 + x + 2 > 0) ↔ (∃ x : ℝ, x^2 + x + 2 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_forall_positive_negation_of_greater_than_zero_l1828_182801


namespace NUMINAMATH_CALUDE_age_difference_l1828_182857

theorem age_difference (tyson_age frederick_age julian_age kyle_age : ℕ) : 
  tyson_age = 20 →
  frederick_age = 2 * tyson_age →
  julian_age = frederick_age - 20 →
  kyle_age = 25 →
  kyle_age - julian_age = 5 := by
sorry

end NUMINAMATH_CALUDE_age_difference_l1828_182857


namespace NUMINAMATH_CALUDE_gold_coin_problem_l1828_182863

theorem gold_coin_problem (n : ℕ) (c : ℕ) : 
  (n = 10 * (c - 4)) →
  (n = 7 * c + 5) →
  n = 110 := by
sorry

end NUMINAMATH_CALUDE_gold_coin_problem_l1828_182863


namespace NUMINAMATH_CALUDE_coworker_lunch_pizzas_l1828_182889

/-- Calculates the number of pizzas needed for a group lunch -/
def pizzas_ordered (coworkers : ℕ) (slices_per_pizza : ℕ) (slices_per_person : ℕ) : ℕ :=
  (coworkers * slices_per_person) / slices_per_pizza

/-- Proves that 12 coworkers each getting 2 slices from pizzas with 8 slices each requires 3 pizzas -/
theorem coworker_lunch_pizzas :
  pizzas_ordered 12 8 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_coworker_lunch_pizzas_l1828_182889


namespace NUMINAMATH_CALUDE_most_cost_effective_plan_l1828_182852

-- Define the prices of A and B type devices
def price_A : ℕ := 12000
def price_B : ℕ := 10000

-- Define the production capacities
def capacity_A : ℕ := 240
def capacity_B : ℕ := 180

-- Define the total number of devices to purchase
def total_devices : ℕ := 10

-- Define the budget constraint
def budget : ℕ := 110000

-- Define the minimum required production capacity
def min_capacity : ℕ := 2040

-- Theorem statement
theorem most_cost_effective_plan :
  ∃ (num_A num_B : ℕ),
    -- The total number of devices is 10
    num_A + num_B = total_devices ∧
    -- The total cost is within budget
    num_A * price_A + num_B * price_B ≤ budget ∧
    -- The total production capacity meets the minimum requirement
    num_A * capacity_A + num_B * capacity_B ≥ min_capacity ∧
    -- This is the most cost-effective plan
    ∀ (other_A other_B : ℕ),
      other_A + other_B = total_devices →
      other_A * capacity_A + other_B * capacity_B ≥ min_capacity →
      other_A * price_A + other_B * price_B ≥ num_A * price_A + num_B * price_B :=
by
  -- The proof goes here
  sorry

#check most_cost_effective_plan

end NUMINAMATH_CALUDE_most_cost_effective_plan_l1828_182852


namespace NUMINAMATH_CALUDE_remainder_of_power_sum_l1828_182898

theorem remainder_of_power_sum (n : ℕ) : (6^83 + 8^83) % 49 = 35 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_power_sum_l1828_182898


namespace NUMINAMATH_CALUDE_max_y_over_x_l1828_182885

/-- Given that x and y satisfy (x-2)^2 + y^2 = 1, the maximum value of y/x is √3/3 -/
theorem max_y_over_x (x y : ℝ) (h : (x - 2)^2 + y^2 = 1) :
  ∃ (max : ℝ), max = Real.sqrt 3 / 3 ∧ ∀ (x' y' : ℝ), (x' - 2)^2 + y'^2 = 1 → |y' / x'| ≤ max := by
  sorry

end NUMINAMATH_CALUDE_max_y_over_x_l1828_182885


namespace NUMINAMATH_CALUDE_smaller_two_digit_factor_l1828_182862

theorem smaller_two_digit_factor (a b : ℕ) : 
  10 ≤ a ∧ a < 100 ∧ 
  10 ≤ b ∧ b < 100 ∧ 
  a * b = 4536 →
  min a b = 54 := by
sorry

end NUMINAMATH_CALUDE_smaller_two_digit_factor_l1828_182862


namespace NUMINAMATH_CALUDE_impossibleConsecutive_l1828_182829

/-- A move that replaces one number with the sum of both numbers -/
def move (a b : ℕ) : ℕ × ℕ := (a + b, b)

/-- The sequence of numbers obtained after applying moves -/
def boardSequence : ℕ → ℕ × ℕ
  | 0 => (2, 5)
  | n + 1 => let (a, b) := boardSequence n; move a b

/-- The difference between the two numbers on the board after n moves -/
def difference (n : ℕ) : ℕ :=
  let (a, b) := boardSequence n
  max a b - min a b

theorem impossibleConsecutive : ∀ n : ℕ, difference n ≠ 1 := by
  sorry

end NUMINAMATH_CALUDE_impossibleConsecutive_l1828_182829


namespace NUMINAMATH_CALUDE_min_socks_for_five_correct_min_socks_for_five_optimal_l1828_182837

/-- Represents the colors of socks --/
inductive Color
  | Red
  | White
  | Blue

/-- Represents a drawer of socks --/
structure SockDrawer where
  red : ℕ
  white : ℕ
  blue : ℕ
  red_min : red ≥ 5
  white_min : white ≥ 5
  blue_min : blue ≥ 5

/-- The minimum number of socks to guarantee 5 of the same color --/
def minSocksForFive (drawer : SockDrawer) : ℕ := 13

theorem min_socks_for_five_correct (drawer : SockDrawer) :
  ∀ n : ℕ, n < minSocksForFive drawer →
    ∃ (r w b : ℕ), r < 5 ∧ w < 5 ∧ b < 5 ∧ r + w + b = n :=
  sorry

theorem min_socks_for_five_optimal (drawer : SockDrawer) :
  ∃ (r w b : ℕ), (r = 5 ∨ w = 5 ∨ b = 5) ∧ r + w + b = minSocksForFive drawer :=
  sorry

end NUMINAMATH_CALUDE_min_socks_for_five_correct_min_socks_for_five_optimal_l1828_182837


namespace NUMINAMATH_CALUDE_pure_imaginary_square_l1828_182858

def complex (a b : ℝ) : ℂ := ⟨a, b⟩

def is_pure_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

theorem pure_imaginary_square (x : ℝ) :
  is_pure_imaginary ((complex x 1)^2) → x = 1 ∨ x = -1 :=
by sorry

end NUMINAMATH_CALUDE_pure_imaginary_square_l1828_182858


namespace NUMINAMATH_CALUDE_absolute_value_equation_product_l1828_182878

theorem absolute_value_equation_product (x : ℝ) : 
  (|15 / x + 4| = 3) → (∃ y : ℝ, (|15 / y + 4| = 3) ∧ (x * y = 225 / 7)) :=
by sorry

end NUMINAMATH_CALUDE_absolute_value_equation_product_l1828_182878


namespace NUMINAMATH_CALUDE_distance_between_points_l1828_182836

theorem distance_between_points (x : ℝ) : 
  let A := 3 + x
  let B := 3 - x
  |A - B| = 8 → |x| = 4 := by
sorry

end NUMINAMATH_CALUDE_distance_between_points_l1828_182836


namespace NUMINAMATH_CALUDE_junior_score_l1828_182887

theorem junior_score (n : ℝ) (h_pos : n > 0) : 
  let junior_count := 0.2 * n
  let senior_count := 0.8 * n
  let total_score := 86 * n
  let senior_score := 85 * senior_count
  junior_count * (total_score - senior_score) / junior_count = 90 :=
by sorry

end NUMINAMATH_CALUDE_junior_score_l1828_182887


namespace NUMINAMATH_CALUDE_quadratic_equation_1_l1828_182883

theorem quadratic_equation_1 : 
  ∃ x₁ x₂ : ℝ, x₁^2 - 2*x₁ - 99 = 0 ∧ x₂^2 - 2*x₂ - 99 = 0 ∧ x₁ ≠ x₂ :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_1_l1828_182883


namespace NUMINAMATH_CALUDE_highest_score_for_given_stats_l1828_182882

/-- Represents a batsman's statistics -/
structure BatsmanStats where
  totalInnings : ℕ
  overallAverage : ℚ
  scoreDifference : ℕ
  averageExcludingExtremes : ℚ

/-- Calculates the highest score given a batsman's statistics -/
def highestScore (stats : BatsmanStats) : ℕ :=
  sorry

/-- Theorem stating the highest score for the given conditions -/
theorem highest_score_for_given_stats :
  let stats : BatsmanStats := {
    totalInnings := 46,
    overallAverage := 59,
    scoreDifference := 150,
    averageExcludingExtremes := 58
  }
  highestScore stats = 151 := by
  sorry

end NUMINAMATH_CALUDE_highest_score_for_given_stats_l1828_182882


namespace NUMINAMATH_CALUDE_system_solution_independent_of_c_l1828_182873

theorem system_solution_independent_of_c :
  ∀ (c : ℝ),
    2 - 0 + 2*(-1) = 0 ∧
    -2*2 + 0 - 2*(-1) = -2 ∧
    2*2 + c*0 + 3*(-1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_independent_of_c_l1828_182873


namespace NUMINAMATH_CALUDE_max_value_of_f_l1828_182823

noncomputable section

variable (a : ℝ)
variable (x : ℝ)

def f (x : ℝ) : ℝ := a * x^2 * Real.exp x

theorem max_value_of_f (h : a ≠ 0) :
  (a > 0 → ∃ (M : ℝ), M = 4 * a * Real.exp (-2) ∧ ∀ x, f a x ≤ M) ∧
  (a < 0 → ∃ (M : ℝ), M = 0 ∧ ∀ x, f a x ≤ M) :=
sorry

end

end NUMINAMATH_CALUDE_max_value_of_f_l1828_182823


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l1828_182839

def M : Set ℤ := {1, 2, 3, 4}
def N : Set ℤ := {-2, 2}

theorem intersection_of_M_and_N : M ∩ N = {2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l1828_182839


namespace NUMINAMATH_CALUDE_quadratic_is_perfect_square_l1828_182849

theorem quadratic_is_perfect_square :
  ∃ (a b : ℝ), ∀ x, 9 * x^2 - 30 * x + 25 = (a * x + b)^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_is_perfect_square_l1828_182849


namespace NUMINAMATH_CALUDE_senate_committee_seating_l1828_182888

/-- The number of ways to arrange n distinguishable objects in a circle -/
def circularPermutations (n : ℕ) : ℕ := (n - 1).factorial

/-- The number of politicians in the committee -/
def committeeSize : ℕ := 4 + 4 + 3

theorem senate_committee_seating :
  circularPermutations committeeSize = 3628800 := by
  sorry

end NUMINAMATH_CALUDE_senate_committee_seating_l1828_182888


namespace NUMINAMATH_CALUDE_pencil_sorting_l1828_182833

theorem pencil_sorting (box2 box3 box4 box5 : ℕ) : 
  box2 = 87 →
  box3 = box2 + 9 →
  box4 = box3 + 9 →
  box5 = box4 + 9 →
  box5 = 114 →
  box2 - 9 = 78 := by
sorry

end NUMINAMATH_CALUDE_pencil_sorting_l1828_182833


namespace NUMINAMATH_CALUDE_field_width_proof_l1828_182884

/-- Proves that a rectangular field with given conditions has a width of 20 feet -/
theorem field_width_proof (total_tape : ℝ) (field_length : ℝ) (leftover_tape : ℝ) 
  (h1 : total_tape = 250)
  (h2 : field_length = 60)
  (h3 : leftover_tape = 90) :
  let used_tape := total_tape - leftover_tape
  let perimeter := used_tape
  let width := (perimeter - 2 * field_length) / 2
  width = 20 := by sorry

end NUMINAMATH_CALUDE_field_width_proof_l1828_182884


namespace NUMINAMATH_CALUDE_sum_of_roots_squared_diff_sum_of_roots_eq_fourteen_l1828_182841

theorem sum_of_roots_squared_diff (a b : ℝ) :
  (∃ x y : ℝ, (x - a)^2 = b^2 ∧ (y - a)^2 = b^2 ∧ x + y = 2 * a) :=
by
  sorry

theorem sum_of_roots_eq_fourteen :
  let roots := {x : ℝ | (x - 7)^2 = 16}
  (∃ x y : ℝ, x ∈ roots ∧ y ∈ roots ∧ x + y = 14) :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_squared_diff_sum_of_roots_eq_fourteen_l1828_182841


namespace NUMINAMATH_CALUDE_geometry_theorem_l1828_182814

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (subset : Line → Plane → Prop)  -- line is subset of plane
variable (parallel : Line → Line → Prop)  -- lines are parallel
variable (parallel_lp : Line → Plane → Prop)  -- line is parallel to plane
variable (parallel_pp : Plane → Plane → Prop)  -- planes are parallel
variable (perpendicular : Line → Line → Prop)  -- lines are perpendicular
variable (perpendicular_lp : Line → Plane → Prop)  -- line is perpendicular to plane

-- Define the theorem
theorem geometry_theorem 
  (m n : Line) (α β : Plane) 
  (h_diff_lines : m ≠ n) 
  (h_diff_planes : α ≠ β) :
  (perpendicular_lp m α ∧ parallel_lp n α → perpendicular m n) ∧ 
  (perpendicular_lp m α ∧ perpendicular_lp m β → parallel_pp α β) ∧
  ¬(∀ m n α, subset m α ∧ parallel_lp n α → parallel m n) ∧
  ¬(∀ m n α, parallel_lp m α ∧ parallel_lp n α → parallel m n) :=
by sorry

end NUMINAMATH_CALUDE_geometry_theorem_l1828_182814


namespace NUMINAMATH_CALUDE_smallest_winning_number_l1828_182881

theorem smallest_winning_number : ∃ N : ℕ, 
  (N = 6) ∧ 
  (8 * N + 450 < 500) ∧ 
  (N ≤ 499) ∧ 
  (∀ m : ℕ, m < N → (8 * m + 450 ≥ 500) ∨ m > 499) :=
by sorry

end NUMINAMATH_CALUDE_smallest_winning_number_l1828_182881


namespace NUMINAMATH_CALUDE_ceiling_minus_x_l1828_182876

theorem ceiling_minus_x (x : ℝ) (h : ⌈x⌉ - ⌊x⌋ = 1) : ⌈x⌉ - x = 1 - (x - ⌊x⌋) := by
  sorry

end NUMINAMATH_CALUDE_ceiling_minus_x_l1828_182876


namespace NUMINAMATH_CALUDE_remainder_invariance_l1828_182866

theorem remainder_invariance (n : ℤ) : (n + 22) % 9 = 2 → (n + 31) % 9 = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_invariance_l1828_182866


namespace NUMINAMATH_CALUDE_hcd_problem_l1828_182818

theorem hcd_problem : (Nat.gcd 12348 2448 * 3) - 14 = 94 := by
  sorry

end NUMINAMATH_CALUDE_hcd_problem_l1828_182818


namespace NUMINAMATH_CALUDE_simplify_expression_l1828_182895

theorem simplify_expression (x : ℝ) (hx : x ≠ 0) :
  (25 * x^3) * (8 * x^2) * (1 / (4 * x)^3) = (25 / 8) * x^2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1828_182895


namespace NUMINAMATH_CALUDE_problem_1_l1828_182850

theorem problem_1 : (1 : ℝ) * (1 - 2 * Real.sqrt 3) * (1 + 2 * Real.sqrt 3) - (1 + Real.sqrt 3)^2 = -15 - 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l1828_182850


namespace NUMINAMATH_CALUDE_marble_distribution_l1828_182875

theorem marble_distribution (x : ℚ) 
  (total_marbles : ℕ) 
  (first_boy : ℚ → ℚ) 
  (second_boy : ℚ → ℚ) 
  (third_boy : ℚ → ℚ) 
  (h1 : first_boy x = 4 * x + 2)
  (h2 : second_boy x = 2 * x)
  (h3 : third_boy x = 3 * x - 1)
  (h4 : total_marbles = 47)
  (h5 : (first_boy x + second_boy x + third_boy x : ℚ) = total_marbles) :
  (first_boy x, second_boy x, third_boy x) = (202/9, 92/9, 129/9) := by
sorry

end NUMINAMATH_CALUDE_marble_distribution_l1828_182875


namespace NUMINAMATH_CALUDE_pie_eating_contest_l1828_182830

theorem pie_eating_contest (first_student third_student : ℚ) : 
  first_student = 7/8 → third_student = 3/4 → first_student - third_student = 1/8 := by
  sorry

end NUMINAMATH_CALUDE_pie_eating_contest_l1828_182830


namespace NUMINAMATH_CALUDE_five_thirteenths_period_l1828_182817

def decimal_expansion_period (n d : ℕ) : ℕ :=
  sorry

theorem five_thirteenths_period :
  decimal_expansion_period 5 13 = 6 := by
  sorry

end NUMINAMATH_CALUDE_five_thirteenths_period_l1828_182817


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1828_182860

theorem complex_equation_solution (z : ℂ) : z = Complex.I * (2 + z) → z = -1 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1828_182860


namespace NUMINAMATH_CALUDE_percentage_difference_l1828_182825

theorem percentage_difference : 
  (0.12 * 24.2) - (0.10 * 14.2) = 1.484 := by
  sorry

end NUMINAMATH_CALUDE_percentage_difference_l1828_182825


namespace NUMINAMATH_CALUDE_equation_solutions_l1828_182879

theorem equation_solutions :
  (∃ x : ℚ, 3 * x - (x - 1) = 7 ∧ x = 3) ∧
  (∃ x : ℚ, (2 * x - 1) / 3 - (x - 3) / 6 = 1 ∧ x = 5 / 3) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l1828_182879


namespace NUMINAMATH_CALUDE_combined_mean_of_two_sets_l1828_182842

theorem combined_mean_of_two_sets (set1_count : ℕ) (set1_mean : ℚ) (set2_count : ℕ) (set2_mean : ℚ) :
  set1_count = 7 →
  set1_mean = 15 →
  set2_count = 8 →
  set2_mean = 20 →
  (set1_count * set1_mean + set2_count * set2_mean) / (set1_count + set2_count) = 265 / 15 := by
  sorry

end NUMINAMATH_CALUDE_combined_mean_of_two_sets_l1828_182842


namespace NUMINAMATH_CALUDE_sara_lunch_bill_total_l1828_182880

/-- The total cost of Sara's lunch bill --/
def lunch_bill (hotdog_cost salad_cost drink_cost side_item_cost : ℚ) : ℚ :=
  hotdog_cost + salad_cost + drink_cost + side_item_cost

/-- Theorem stating that Sara's lunch bill totals $16.71 --/
theorem sara_lunch_bill_total :
  lunch_bill 5.36 5.10 2.50 3.75 = 16.71 := by
  sorry

end NUMINAMATH_CALUDE_sara_lunch_bill_total_l1828_182880


namespace NUMINAMATH_CALUDE_sheilas_weekly_earnings_l1828_182871

/-- Sheila's weekly earnings calculation -/
theorem sheilas_weekly_earnings :
  let hourly_rate : ℕ := 12
  let hours_mon_wed_fri : ℕ := 8
  let hours_tue_thu : ℕ := 6
  let days_8_hours : ℕ := 3
  let days_6_hours : ℕ := 2
  let earnings_8_hour_days : ℕ := hourly_rate * hours_mon_wed_fri * days_8_hours
  let earnings_6_hour_days : ℕ := hourly_rate * hours_tue_thu * days_6_hours
  let total_earnings : ℕ := earnings_8_hour_days + earnings_6_hour_days
  total_earnings = 432 :=
by sorry

end NUMINAMATH_CALUDE_sheilas_weekly_earnings_l1828_182871


namespace NUMINAMATH_CALUDE_dogs_can_prevent_escape_l1828_182865

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a square field -/
structure SquareField where
  sideLength : ℝ
  center : Point
  corners : Fin 4 → Point

/-- Represents the game setup -/
structure WolfDogGame where
  field : SquareField
  wolfSpeed : ℝ
  dogSpeed : ℝ

/-- Checks if a point is on the perimeter of the square field -/
def isOnPerimeter (field : SquareField) (p : Point) : Prop :=
  let a := field.sideLength / 2
  (p.x = field.center.x - a ∨ p.x = field.center.x + a) ∧
  (p.y ≥ field.center.y - a ∧ p.y ≤ field.center.y + a) ∨
  (p.y = field.center.y - a ∨ p.y = field.center.y + a) ∧
  (p.x ≥ field.center.x - a ∧ p.x ≤ field.center.x + a)

/-- Theorem: Dogs can prevent the wolf from escaping -/
theorem dogs_can_prevent_escape (game : WolfDogGame) 
  (h1 : game.field.sideLength > 0)
  (h2 : game.dogSpeed = 1.5 * game.wolfSpeed)
  (h3 : game.wolfSpeed > 0) :
  ∀ (p : Point), isOnPerimeter game.field p →
    ∃ (t : ℝ), t ≥ 0 ∧ 
      (∃ (i : Fin 4), (t * game.dogSpeed)^2 ≥ 
        ((game.field.corners i).x - p.x)^2 + ((game.field.corners i).y - p.y)^2) ∧
      (t * game.wolfSpeed)^2 < 
        (game.field.center.x - p.x)^2 + (game.field.center.y - p.y)^2 :=
sorry

end NUMINAMATH_CALUDE_dogs_can_prevent_escape_l1828_182865


namespace NUMINAMATH_CALUDE_sum_digits_greatest_prime_divisor_8191_l1828_182812

/-- The greatest prime divisor of a natural number n -/
def greatest_prime_divisor (n : ℕ) : ℕ :=
  (Nat.factors n).foldl max 1

/-- The sum of digits of a natural number n -/
def sum_of_digits (n : ℕ) : ℕ :=
  (n.repr.toList.map (λ c => c.toNat - '0'.toNat)).sum

/-- Theorem stating that the sum of the digits of the greatest prime divisor of 8191 is 10 -/
theorem sum_digits_greatest_prime_divisor_8191 :
  sum_of_digits (greatest_prime_divisor 8191) = 10 := by
  sorry

end NUMINAMATH_CALUDE_sum_digits_greatest_prime_divisor_8191_l1828_182812


namespace NUMINAMATH_CALUDE_arrangement_count_is_518400_l1828_182835

/-- The number of ways to arrange 4 math books and 6 history books with specific conditions -/
def arrangement_count : ℕ :=
  let math_books : ℕ := 4
  let history_books : ℕ := 6
  let math_ends : ℕ := math_books * (math_books - 1)
  let consecutive_history : ℕ := Nat.choose history_books 2
  let remaining_units : ℕ := 5  -- 4 single history books + 1 double-history unit
  let middle_arrangements : ℕ := Nat.factorial remaining_units
  let remaining_math_placements : ℕ := Nat.choose remaining_units 2 * Nat.factorial 2
  math_ends * consecutive_history * middle_arrangements * remaining_math_placements

/-- Theorem stating that the number of arrangements is 518,400 -/
theorem arrangement_count_is_518400 : arrangement_count = 518400 := by
  sorry

end NUMINAMATH_CALUDE_arrangement_count_is_518400_l1828_182835


namespace NUMINAMATH_CALUDE_computer_table_markup_l1828_182808

/-- Calculate the percentage markup on a product's cost price. -/
def percentage_markup (selling_price cost_price : ℚ) : ℚ :=
  (selling_price - cost_price) / cost_price * 100

/-- The percentage markup on a computer table -/
theorem computer_table_markup :
  percentage_markup 8340 6672 = 25 := by
  sorry

end NUMINAMATH_CALUDE_computer_table_markup_l1828_182808


namespace NUMINAMATH_CALUDE_ratio_equality_l1828_182804

theorem ratio_equality : (2^2001 * 3^2003) / 6^2002 = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_ratio_equality_l1828_182804


namespace NUMINAMATH_CALUDE_y_to_x_equals_one_l1828_182855

theorem y_to_x_equals_one (x y : ℝ) (h : (y + 1)^2 + Real.sqrt (x - 2) = 0) : y^x = 1 := by
  sorry

end NUMINAMATH_CALUDE_y_to_x_equals_one_l1828_182855


namespace NUMINAMATH_CALUDE_triangle_max_value_l1828_182819

/-- Given a triangle ABC with side lengths a, b, c opposite to angles A, B, C respectively,
    prove that if a² + b² = √3ab + c² and AB = 1, then the maximum value of AC + √3BC is 2√7 -/
theorem triangle_max_value (a b c : ℝ) (A B C : ℝ) :
  a^2 + b^2 = Real.sqrt 3 * a * b + c^2 →
  a = 1 →  -- AB = 1
  ∃ (AC BC : ℝ), AC + Real.sqrt 3 * BC ≤ 2 * Real.sqrt 7 ∧
    ∃ (AC' BC' : ℝ), AC' + Real.sqrt 3 * BC' = 2 * Real.sqrt 7 :=
by sorry

end NUMINAMATH_CALUDE_triangle_max_value_l1828_182819


namespace NUMINAMATH_CALUDE_sin_cos_equation_solutions_l1828_182828

/-- The number of solutions to sin(π/4 * sin x) = cos(π/4 * cos x) in [0, 2π] -/
theorem sin_cos_equation_solutions :
  ∃! (s : Finset ℝ), s.card = 4 ∧ 
  (∀ x ∈ s, 0 ≤ x ∧ x ≤ 2 * π ∧ 
    Real.sin (π/4 * Real.sin x) = Real.cos (π/4 * Real.cos x)) ∧
  (∀ y, 0 ≤ y ∧ y ≤ 2 * π ∧ 
    Real.sin (π/4 * Real.sin y) = Real.cos (π/4 * Real.cos y) → y ∈ s) :=
by sorry

end NUMINAMATH_CALUDE_sin_cos_equation_solutions_l1828_182828


namespace NUMINAMATH_CALUDE_product_of_largest_primes_l1828_182896

/-- The largest two-digit prime number -/
def largest_two_digit_prime : ℕ := 97

/-- The largest four-digit prime number -/
def largest_four_digit_prime : ℕ := 9973

/-- Theorem stating that the product of the largest two-digit prime and the largest four-digit prime is 967781 -/
theorem product_of_largest_primes : 
  largest_two_digit_prime * largest_four_digit_prime = 967781 := by
  sorry

end NUMINAMATH_CALUDE_product_of_largest_primes_l1828_182896


namespace NUMINAMATH_CALUDE_geometric_sequences_theorem_l1828_182864

/-- Two geometric sequences satisfying given conditions -/
structure GeometricSequences where
  a : ℕ → ℝ
  b : ℕ → ℝ
  a_pos : a 1 > 0
  a_geom : ∀ n : ℕ, a (n + 1) = a n * (a 2 / a 1)
  b_geom : ∀ n : ℕ, b (n + 1) = b n * (b 2 / b 1)
  diff_1 : b 1 - a 1 = 1
  diff_2 : b 2 - a 2 = 2
  diff_3 : b 3 - a 3 = 3

/-- The unique value of a_1 in the geometric sequences -/
def unique_a (gs : GeometricSequences) : ℝ := gs.a 1

/-- The statement to be proved -/
theorem geometric_sequences_theorem (gs : GeometricSequences) :
  unique_a gs = 1/3 ∧
  ¬∃ (a b : ℕ → ℝ) (q₁ q₂ : ℝ),
    (∀ n, a (n + 1) = a n * q₁) ∧
    (∀ n, b (n + 1) = b n * q₂) ∧
    ∃ (d : ℝ), d ≠ 0 ∧
    ∀ n : ℕ, n ≤ 3 →
      (b (n + 1) - a (n + 1)) - (b n - a n) = d :=
sorry

end NUMINAMATH_CALUDE_geometric_sequences_theorem_l1828_182864


namespace NUMINAMATH_CALUDE_range_of_m_for_increasing_f_l1828_182869

/-- A quadratic function f(x) = 4x^2 - mx + 5 that is increasing on [-2, +∞) -/
def f (m : ℝ) (x : ℝ) : ℝ := 4 * x^2 - m * x + 5

/-- The property that f is increasing on [-2, +∞) -/
def is_increasing_on_interval (m : ℝ) : Prop :=
  ∀ x y, x ≥ -2 → y ≥ -2 → x < y → f m x < f m y

/-- The theorem stating the range of m for which f is increasing on [-2, +∞) -/
theorem range_of_m_for_increasing_f :
  ∀ m : ℝ, is_increasing_on_interval m → m ≤ -16 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_for_increasing_f_l1828_182869


namespace NUMINAMATH_CALUDE_square_side_length_l1828_182807

theorem square_side_length (s : ℝ) : s > 0 → (4 * s = 2 * s^2) → s = 2 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l1828_182807


namespace NUMINAMATH_CALUDE_volleyball_team_selection_l1828_182811

theorem volleyball_team_selection (n : ℕ) (k : ℕ) : n = 16 ∧ k = 7 → Nat.choose n k = 11440 := by
  sorry

end NUMINAMATH_CALUDE_volleyball_team_selection_l1828_182811


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l1828_182853

theorem triangle_abc_properties (a b c A B C : ℝ) (h1 : 0 < A ∧ A < π) 
  (h2 : 0 < B ∧ B < π) (h3 : 0 < C ∧ C < π) (h4 : A + B + C = π) 
  (h5 : a * Real.cos C + Real.sqrt 3 * a * Real.sin C - b - c = 0) 
  (h6 : a = Real.sqrt 13) (h7 : 1/2 * b * c * Real.sin A = 3 * Real.sqrt 3) : 
  A = π/3 ∧ a + b + c = 7 + Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l1828_182853


namespace NUMINAMATH_CALUDE_hannah_late_count_l1828_182897

-- Define the given conditions
def hourly_rate : ℕ := 30
def weekly_hours : ℕ := 18
def late_penalty : ℕ := 5
def actual_pay : ℕ := 525

-- Define the theorem
theorem hannah_late_count : 
  ∃ (late_count : ℕ), 
    hourly_rate * weekly_hours - late_penalty * late_count = actual_pay ∧ 
    late_count = 3 := by sorry

end NUMINAMATH_CALUDE_hannah_late_count_l1828_182897


namespace NUMINAMATH_CALUDE_study_group_composition_l1828_182844

def number_of_selections (n m : ℕ) : ℕ :=
  (Nat.choose n 2) * (Nat.choose m 1) * 6

theorem study_group_composition :
  ∃ (n m : ℕ),
    n + m = 8 ∧
    number_of_selections n m = 90 ∧
    n = 3 ∧
    m = 5 := by
  sorry

end NUMINAMATH_CALUDE_study_group_composition_l1828_182844


namespace NUMINAMATH_CALUDE_age_problem_l1828_182859

/-- The age problem involving Sebastian, his siblings, and their father. -/
theorem age_problem (sebastian_age : ℕ) (sister_age_diff : ℕ) (brother_age_diff : ℕ) : 
  sebastian_age = 40 →
  sister_age_diff = 10 →
  brother_age_diff = 7 →
  (sebastian_age - 5 + (sebastian_age - sister_age_diff - 5) + 
   (sebastian_age - sister_age_diff - brother_age_diff - 5) : ℚ) = 
   (3 / 4 : ℚ) * ((109 : ℕ) - 5) →
  109 = sebastian_age + 69 := by
  sorry

#check age_problem

end NUMINAMATH_CALUDE_age_problem_l1828_182859


namespace NUMINAMATH_CALUDE_salad_total_calories_l1828_182870

/-- Represents the total calories in a salad. -/
def saladCalories (lettuce_cal : ℕ) (cucumber_cal : ℕ) (crouton_count : ℕ) (crouton_cal : ℕ) : ℕ :=
  lettuce_cal + cucumber_cal + crouton_count * crouton_cal

/-- Proves that the total calories in the salad is 350. -/
theorem salad_total_calories :
  saladCalories 30 80 12 20 = 350 := by
  sorry

end NUMINAMATH_CALUDE_salad_total_calories_l1828_182870


namespace NUMINAMATH_CALUDE_integral_one_plus_sin_l1828_182845

theorem integral_one_plus_sin : ∫ x in -Real.pi..Real.pi, (1 + Real.sin x) = 2 * Real.pi := by sorry

end NUMINAMATH_CALUDE_integral_one_plus_sin_l1828_182845


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1828_182877

theorem quadratic_equation_solution :
  ∃ (a b : ℕ+), 
    (∀ x : ℝ, x > 0 → x^2 + 10*x = 34 ↔ x = Real.sqrt a - b) ∧
    a + b = 64 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1828_182877


namespace NUMINAMATH_CALUDE_sandy_shorts_cost_l1828_182856

/-- Represents the amount Sandy spent on shorts -/
def S : ℝ := sorry

/-- The amount Sandy spent on a shirt -/
def shirt_cost : ℝ := 12.14

/-- The amount Sandy received for returning a jacket -/
def jacket_return : ℝ := 7.43

/-- The net amount Sandy spent on clothes -/
def net_spent : ℝ := 18.7

/-- Theorem stating that Sandy spent $13.99 on shorts -/
theorem sandy_shorts_cost : S = 13.99 :=
  by
    have h : S + shirt_cost - jacket_return = net_spent := by sorry
    sorry


end NUMINAMATH_CALUDE_sandy_shorts_cost_l1828_182856


namespace NUMINAMATH_CALUDE_triangle_side_sum_l1828_182847

theorem triangle_side_sum (a b c : ℝ) (h_angles : a = 60 ∧ b = 30 ∧ c = 90) 
  (h_side : 8 * Real.sqrt 3 = b) : 
  a + b + c = 24 * Real.sqrt 3 + 24 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_sum_l1828_182847


namespace NUMINAMATH_CALUDE_infinite_geometric_series_first_term_l1828_182874

theorem infinite_geometric_series_first_term
  (r : ℝ) (S : ℝ) (a : ℝ)
  (h_r : r = (1 : ℝ) / 4)
  (h_S : S = 80)
  (h_sum : S = a / (1 - r)) :
  a = 60 := by
sorry

end NUMINAMATH_CALUDE_infinite_geometric_series_first_term_l1828_182874


namespace NUMINAMATH_CALUDE_chromosome_stability_l1828_182890

-- Define the number of chromosomes in somatic cells
def somaticChromosomes : ℕ := 46

-- Define the process of meiosis
def meiosis (n : ℕ) : ℕ := n / 2

-- Define the process of fertilization
def fertilization (n : ℕ) : ℕ := n * 2

-- Theorem: Meiosis and fertilization maintain chromosome stability across generations
theorem chromosome_stability :
  ∀ (generation : ℕ),
    fertilization (meiosis somaticChromosomes) = somaticChromosomes :=
by sorry

end NUMINAMATH_CALUDE_chromosome_stability_l1828_182890


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1828_182800

-- Define set A
def A : Set ℝ := {x | 1 < x ∧ x < 8}

-- Define set B
def B : Set ℝ := {x | x^2 - 5*x - 14 ≥ 0}

-- Theorem statement
theorem intersection_of_A_and_B : A ∩ B = {x | 7 ≤ x ∧ x < 8} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1828_182800


namespace NUMINAMATH_CALUDE_triangle_perimeter_l1828_182894

theorem triangle_perimeter : ∀ (a b c : ℝ),
  a = 4 ∧ b = 6 ∧ c^2 - 6*c + 8 = 0 ∧
  a + b > c ∧ a + c > b ∧ b + c > a →
  a + b + c = 14 := by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l1828_182894


namespace NUMINAMATH_CALUDE_not_divisible_by_101_l1828_182840

theorem not_divisible_by_101 (k : ℤ) : ¬(101 ∣ (k^2 + k + 1)) := by
  sorry

end NUMINAMATH_CALUDE_not_divisible_by_101_l1828_182840


namespace NUMINAMATH_CALUDE_rectangle_count_l1828_182805

/-- The number of ways to choose k items from n items --/
def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

/-- The number of horizontal lines --/
def horizontal_lines : ℕ := 5

/-- The number of vertical lines --/
def vertical_lines : ℕ := 4

/-- The number of lines needed to form a rectangle --/
def lines_for_rectangle : ℕ := 4

/-- The number of horizontal lines needed for a rectangle --/
def horizontal_needed : ℕ := 2

/-- The number of vertical lines needed for a rectangle --/
def vertical_needed : ℕ := 2

theorem rectangle_count : 
  (choose horizontal_lines horizontal_needed) * (choose vertical_lines vertical_needed) = 60 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_count_l1828_182805


namespace NUMINAMATH_CALUDE_egg_price_increase_l1828_182892

theorem egg_price_increase (E : ℝ) (h₁ : E > 0) : 
  ∃ (p : ℝ), 
    (p > 0) ∧ 
    (E * (1 + p) + E * 1.06 = E + E + 15) ∧
    (p = 15 / E - 0.06) := by
  sorry

end NUMINAMATH_CALUDE_egg_price_increase_l1828_182892


namespace NUMINAMATH_CALUDE_factor_implies_c_value_l1828_182831

theorem factor_implies_c_value (c : ℝ) : 
  (∀ x : ℝ, (x + 5) ∣ (c * x^3 + 23 * x^2 - 5 * c * x + 55)) → c = 6.3 := by
sorry

end NUMINAMATH_CALUDE_factor_implies_c_value_l1828_182831


namespace NUMINAMATH_CALUDE_quadratic_equation_for_complex_roots_l1828_182810

theorem quadratic_equation_for_complex_roots (ω : ℂ) (α β : ℂ) 
  (h1 : ω^8 = 1) 
  (h2 : ω ≠ 1) 
  (h3 : α = ω + ω^3 + ω^5) 
  (h4 : β = ω^2 + ω^4 + ω^6 + ω^7) :
  α^2 + α + 3 = 0 ∧ β^2 + β + 3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_for_complex_roots_l1828_182810


namespace NUMINAMATH_CALUDE_equal_tuesdays_thursdays_count_l1828_182893

/-- Represents the days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Counts the number of occurrences of a specific day in a 31-day month -/
def countDayInMonth (startDay : DayOfWeek) (targetDay : DayOfWeek) : Nat :=
  sorry

/-- Checks if Tuesdays and Thursdays occur equally in a 31-day month starting on the given day -/
def hasSameTuesdaysThursdays (startDay : DayOfWeek) : Bool :=
  countDayInMonth startDay DayOfWeek.Tuesday = countDayInMonth startDay DayOfWeek.Thursday

/-- The set of days that can start a 31-day month with equal Tuesdays and Thursdays -/
def validStartDays : Finset DayOfWeek :=
  sorry

theorem equal_tuesdays_thursdays_count :
  Finset.card validStartDays = 3 :=
sorry

end NUMINAMATH_CALUDE_equal_tuesdays_thursdays_count_l1828_182893


namespace NUMINAMATH_CALUDE_coloring_books_problem_l1828_182872

theorem coloring_books_problem (total_colored : ℕ) (total_left : ℕ) (num_books : ℕ) :
  total_colored = 20 →
  total_left = 68 →
  num_books = 2 →
  (total_colored + total_left) % num_books = 0 →
  (total_colored + total_left) / num_books = 44 :=
by
  sorry

end NUMINAMATH_CALUDE_coloring_books_problem_l1828_182872


namespace NUMINAMATH_CALUDE_savings_calculation_l1828_182821

/-- Calculates the total savings for a year given monthly expenses and average monthly income -/
def yearly_savings (expense1 expense2 expense3 : ℕ) (months1 months2 months3 : ℕ) (avg_income : ℕ) : ℕ :=
  let total_expense := expense1 * months1 + expense2 * months2 + expense3 * months3
  let total_income := avg_income * 12
  total_income - total_expense

/-- Proves that the yearly savings is 5200 given the specific expenses and income -/
theorem savings_calculation : yearly_savings 1700 1550 1800 3 4 5 2125 = 5200 := by
  sorry

#eval yearly_savings 1700 1550 1800 3 4 5 2125

end NUMINAMATH_CALUDE_savings_calculation_l1828_182821


namespace NUMINAMATH_CALUDE_no_perfect_squares_in_sequence_l1828_182816

def x : ℕ → ℤ
  | 0 => 1
  | 1 => 3
  | (n + 2) => 6 * x (n + 1) - x n

theorem no_perfect_squares_in_sequence : ∀ n : ℕ, ¬∃ m : ℕ, x n = m ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_no_perfect_squares_in_sequence_l1828_182816


namespace NUMINAMATH_CALUDE_symmetry_of_shifted_even_function_l1828_182886

-- Define a function f
variable (f : ℝ → ℝ)

-- Define what it means for a function to be even
def is_even (g : ℝ → ℝ) : Prop := ∀ x, g x = g (-x)

-- Define the axis of symmetry for a function
def axis_of_symmetry (g : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, g (a + x) = g (a - x)

-- State the theorem
theorem symmetry_of_shifted_even_function :
  is_even (fun x ↦ f (x - 2)) → axis_of_symmetry f (-2) :=
by sorry

end NUMINAMATH_CALUDE_symmetry_of_shifted_even_function_l1828_182886


namespace NUMINAMATH_CALUDE_chicago_bulls_wins_conditions_satisfied_l1828_182827

/-- The number of games won by the Chicago Bulls -/
def bulls_wins : ℕ := 70

/-- The number of games won by the Miami Heat -/
def heat_wins : ℕ := bulls_wins + 5

/-- The total number of games won by both teams -/
def total_wins : ℕ := 145

/-- Theorem stating that the Chicago Bulls won 70 games -/
theorem chicago_bulls_wins : bulls_wins = 70 := by sorry

/-- Theorem proving the conditions are satisfied -/
theorem conditions_satisfied :
  (heat_wins = bulls_wins + 5) ∧ (bulls_wins + heat_wins = total_wins) := by sorry

end NUMINAMATH_CALUDE_chicago_bulls_wins_conditions_satisfied_l1828_182827


namespace NUMINAMATH_CALUDE_tangent_ellipse_hyperbola_parameter_l1828_182834

/-- Given an ellipse and a hyperbola that are tangent, prove that the parameter m of the hyperbola is 5/9 -/
theorem tangent_ellipse_hyperbola_parameter (x y m : ℝ) : 
  (∃ x y, x^2 + 9*y^2 = 9 ∧ x^2 - m*(y+3)^2 = 4) →  -- Existence of points satisfying both equations
  (∀ x y, x^2 + 9*y^2 = 9 → x^2 - m*(y+3)^2 ≥ 4) →  -- Hyperbola does not intersect interior of ellipse
  (∃ x y, x^2 + 9*y^2 = 9 ∧ x^2 - m*(y+3)^2 = 4) →  -- Existence of a common point
  m = 5/9 := by sorry

end NUMINAMATH_CALUDE_tangent_ellipse_hyperbola_parameter_l1828_182834


namespace NUMINAMATH_CALUDE_joe_test_scores_l1828_182861

theorem joe_test_scores (initial_avg : ℚ) (lowest_score : ℚ) (new_avg : ℚ) 
  (h1 : initial_avg = 90)
  (h2 : lowest_score = 75)
  (h3 : new_avg = 85) :
  ∃ n : ℕ, n > 0 ∧ 
    (n : ℚ) * initial_avg - lowest_score = (n - 1 : ℚ) * new_avg ∧
    n = 13 := by
sorry

end NUMINAMATH_CALUDE_joe_test_scores_l1828_182861


namespace NUMINAMATH_CALUDE_x_eq_two_iff_quadratic_eq_zero_l1828_182809

theorem x_eq_two_iff_quadratic_eq_zero : ∀ x : ℝ, x = 2 ↔ x^2 - 4*x + 4 = 0 := by
  sorry

end NUMINAMATH_CALUDE_x_eq_two_iff_quadratic_eq_zero_l1828_182809


namespace NUMINAMATH_CALUDE_circle_center_and_radius_l1828_182822

/-- Given a circle C defined by the equation x^2 + y^2 - 2x + 6y = 0,
    prove that its center is (1, -3) and its radius is √10 -/
theorem circle_center_and_radius :
  ∀ (x y : ℝ), x^2 + y^2 - 2*x + 6*y = 0 →
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    center = (1, -3) ∧
    radius = Real.sqrt 10 ∧
    (x - center.1)^2 + (y - center.2)^2 = radius^2 :=
by sorry


end NUMINAMATH_CALUDE_circle_center_and_radius_l1828_182822


namespace NUMINAMATH_CALUDE_t_shape_area_is_20_l1828_182802

/-- Represents the structure inside the square WXYZ -/
structure InternalStructure where
  top_left_side : ℕ
  top_right_side : ℕ
  bottom_right_side : ℕ
  bottom_left_side : ℕ
  rectangle_width : ℕ
  rectangle_height : ℕ

/-- Calculates the area of the T-shaped region -/
def t_shape_area (s : InternalStructure) : ℕ :=
  s.top_left_side * s.top_left_side +
  s.bottom_right_side * s.bottom_right_side +
  s.bottom_left_side * s.bottom_left_side +
  s.rectangle_width * s.rectangle_height

/-- The theorem stating that the area of the T-shaped region is 20 -/
theorem t_shape_area_is_20 (s : InternalStructure)
  (h1 : s.top_left_side = 2)
  (h2 : s.top_right_side = 2)
  (h3 : s.bottom_right_side = 2)
  (h4 : s.bottom_left_side = 2)
  (h5 : s.rectangle_width = 4)
  (h6 : s.rectangle_height = 2) :
  t_shape_area s = 20 := by
  sorry

end NUMINAMATH_CALUDE_t_shape_area_is_20_l1828_182802


namespace NUMINAMATH_CALUDE_distance_walked_l1828_182803

/-- Given a walking speed and a total walking time, calculate the distance walked. -/
theorem distance_walked (speed : ℝ) (time : ℝ) (h1 : speed = 1 / 15) (h2 : time = 45) :
  speed * time = 3 := by
  sorry

end NUMINAMATH_CALUDE_distance_walked_l1828_182803


namespace NUMINAMATH_CALUDE_octal_subtraction_example_l1828_182806

/-- Represents a number in base 8 --/
def OctalNumber := ℕ

/-- Converts a natural number to its octal representation --/
def toOctal (n : ℕ) : OctalNumber :=
  sorry

/-- Performs subtraction in base 8 --/
def octalSubtract (a b : OctalNumber) : OctalNumber :=
  sorry

theorem octal_subtraction_example :
  octalSubtract (toOctal 641) (toOctal 324) = toOctal 317 := by
  sorry

end NUMINAMATH_CALUDE_octal_subtraction_example_l1828_182806


namespace NUMINAMATH_CALUDE_unique_number_with_conditions_l1828_182815

theorem unique_number_with_conditions : ∃! n : ℕ,
  50 < n ∧ n < 70 ∧
  n % 5 = 3 ∧
  n % 7 = 2 ∧
  n % 8 = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_unique_number_with_conditions_l1828_182815
