import Mathlib

namespace NUMINAMATH_CALUDE_equation_solution_l1562_156253

theorem equation_solution (x y : ℝ) 
  (eq1 : 3 * x + 2 * y = 9) 
  (eq2 : x + 3 * y = 8) : 
  3 * x^2 + 7 * x * y + 3 * y^2 = 145 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l1562_156253


namespace NUMINAMATH_CALUDE_sin_cos_power_relation_l1562_156203

theorem sin_cos_power_relation (x : ℝ) :
  (Real.sin x)^10 + (Real.cos x)^10 = 11/36 →
  (Real.sin x)^12 + (Real.cos x)^12 = 5/18 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_power_relation_l1562_156203


namespace NUMINAMATH_CALUDE_sally_seashell_money_l1562_156217

/-- The number of seashells Sally picks on Monday -/
def monday_seashells : ℕ := 30

/-- The number of seashells Sally picks on Tuesday -/
def tuesday_seashells : ℕ := monday_seashells / 2

/-- The price of each seashell in dollars -/
def seashell_price : ℚ := 6/5

/-- The total number of seashells Sally picks -/
def total_seashells : ℕ := monday_seashells + tuesday_seashells

/-- The total money Sally can make by selling all her seashells -/
def total_money : ℚ := (total_seashells : ℚ) * seashell_price

theorem sally_seashell_money : total_money = 54 := by
  sorry

end NUMINAMATH_CALUDE_sally_seashell_money_l1562_156217


namespace NUMINAMATH_CALUDE_cube_split_theorem_l1562_156214

/-- Given a natural number m > 1, returns the first odd number in the split of m³ -/
def firstSplitNumber (m : ℕ) : ℕ := m * (m - 1) + 1

/-- Given a natural number m > 1, returns the list of odd numbers in the split of m³ -/
def splitNumbers (m : ℕ) : List ℕ :=
  List.range m |>.map (λ i => firstSplitNumber m + 2 * i)

theorem cube_split_theorem (m : ℕ) (h1 : m > 1) (h2 : 333 ∈ splitNumbers m) : m = 18 := by
  sorry

end NUMINAMATH_CALUDE_cube_split_theorem_l1562_156214


namespace NUMINAMATH_CALUDE_fraction_numerator_l1562_156290

theorem fraction_numerator (y : ℝ) (x : ℝ) (h1 : y > 0) 
  (h2 : x / y * y + 3 * y / 10 = 1 / 2 * y) : x = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_numerator_l1562_156290


namespace NUMINAMATH_CALUDE_toothpick_grid_problem_l1562_156249

/-- Calculates the total number of toothpicks in a grid -/
def total_toothpicks (length width : ℕ) (has_divider : Bool) : ℕ :=
  let vertical_lines := length + 1 + (if has_divider then 1 else 0)
  let vertical_toothpicks := vertical_lines * width
  let horizontal_lines := width + 1
  let horizontal_toothpicks := horizontal_lines * length
  vertical_toothpicks + horizontal_toothpicks

/-- The problem statement -/
theorem toothpick_grid_problem :
  total_toothpicks 40 25 true = 2090 := by
  sorry


end NUMINAMATH_CALUDE_toothpick_grid_problem_l1562_156249


namespace NUMINAMATH_CALUDE_truck_distance_l1562_156200

/-- Proves the distance traveled by a truck in yards over 5 minutes -/
theorem truck_distance (b t : ℝ) (h1 : t > 0) : 
  let feet_per_t_seconds : ℝ := b / 4
  let feet_in_yard : ℝ := 2
  let minutes : ℝ := 5
  let seconds_in_minute : ℝ := 60
  let yards_traveled : ℝ := (feet_per_t_seconds * (minutes * seconds_in_minute) / t) / feet_in_yard
  yards_traveled = 37.5 * b / t := by
sorry

end NUMINAMATH_CALUDE_truck_distance_l1562_156200


namespace NUMINAMATH_CALUDE_scrabble_champions_years_l1562_156294

theorem scrabble_champions_years (total_champions : ℕ) 
  (women_percent : ℚ) (men_with_beard_percent : ℚ) (men_with_beard : ℕ) : 
  women_percent = 3/5 →
  men_with_beard_percent = 2/5 →
  men_with_beard = 4 →
  total_champions = 25 := by
sorry

end NUMINAMATH_CALUDE_scrabble_champions_years_l1562_156294


namespace NUMINAMATH_CALUDE_first_occurrence_is_lcm_l1562_156295

/-- Represents the cycle length of letters -/
def letter_cycle : ℕ := 8

/-- Represents the cycle length of digits -/
def digit_cycle : ℕ := 4

/-- Represents the first occurrence of the original sequence -/
def first_occurrence : ℕ := 8

/-- Theorem stating that the first occurrence is the least common multiple of the cycle lengths -/
theorem first_occurrence_is_lcm :
  first_occurrence = Nat.lcm letter_cycle digit_cycle := by sorry

end NUMINAMATH_CALUDE_first_occurrence_is_lcm_l1562_156295


namespace NUMINAMATH_CALUDE_sequence_difference_l1562_156271

def sequence1 (n : ℕ) : ℤ := 2 * (-2)^(n - 1)
def sequence2 (n : ℕ) : ℤ := sequence1 n - 1
def sequence3 (n : ℕ) : ℤ := (-2)^n - sequence2 n

theorem sequence_difference : sequence1 7 - sequence2 7 + sequence3 7 = -254 := by
  sorry

end NUMINAMATH_CALUDE_sequence_difference_l1562_156271


namespace NUMINAMATH_CALUDE_fourth_side_distance_l1562_156213

/-- A square with a point inside it -/
structure SquareWithPoint where
  side_length : ℝ
  d1 : ℝ
  d2 : ℝ
  d3 : ℝ
  d4 : ℝ
  h_positive : 0 < side_length
  h_inside : d1 + d2 + d3 + d4 = side_length
  h_d1 : 0 < d1
  h_d2 : 0 < d2
  h_d3 : 0 < d3
  h_d4 : 0 < d4

/-- The theorem stating the possible distances to the fourth side -/
theorem fourth_side_distance (s : SquareWithPoint) 
  (h1 : s.d1 = 4)
  (h2 : s.d2 = 7)
  (h3 : s.d3 = 13) :
  s.d4 = 10 ∨ s.d4 = 16 := by
  sorry

end NUMINAMATH_CALUDE_fourth_side_distance_l1562_156213


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1562_156287

open Set

-- Define the sets A and B
def A : Set ℝ := { x | 2 < x ∧ x < 4 }
def B : Set ℝ := { x | x < 3 ∨ x > 5 }

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = { x | 2 < x ∧ x < 3 } := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1562_156287


namespace NUMINAMATH_CALUDE_rectangle_strip_count_l1562_156248

theorem rectangle_strip_count 
  (outer_perimeter : ℕ) 
  (hole_perimeter : ℕ) 
  (horizontal_strips : ℕ) : 
  outer_perimeter = 50 → 
  hole_perimeter = 32 → 
  horizontal_strips = 20 → 
  ∃ (vertical_strips : ℕ), vertical_strips = 21 :=
by
  sorry

end NUMINAMATH_CALUDE_rectangle_strip_count_l1562_156248


namespace NUMINAMATH_CALUDE_gravitational_force_calculation_l1562_156280

/-- Gravitational force calculation -/
theorem gravitational_force_calculation
  (k : ℝ) -- Gravitational constant
  (d₁ d₂ : ℝ) -- Distances
  (f₁ : ℝ) -- Force at distance d₁
  (h₁ : d₁ = 8000)
  (h₂ : d₂ = 320000)
  (h₃ : f₁ = 150)
  (h₄ : k = f₁ * d₁^2) -- Inverse square law
  : ∃ f₂ : ℝ, f₂ = k / d₂^2 ∧ f₂ = 3 / 32 := by
  sorry

end NUMINAMATH_CALUDE_gravitational_force_calculation_l1562_156280


namespace NUMINAMATH_CALUDE_smallest_integer_with_20_divisors_l1562_156236

theorem smallest_integer_with_20_divisors : 
  ∃ n : ℕ+, (n = 240) ∧ 
  (∀ m : ℕ+, m < n → (Finset.card (Nat.divisors m) ≠ 20)) ∧ 
  (Finset.card (Nat.divisors n) = 20) := by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_with_20_divisors_l1562_156236


namespace NUMINAMATH_CALUDE_oliver_final_amount_l1562_156284

/-- Calculates the final amount of money Oliver has after all transactions. -/
def olivers_money (initial : ℕ) (saved : ℕ) (frisbee_cost : ℕ) (puzzle_cost : ℕ) (gift : ℕ) : ℕ :=
  initial + saved - frisbee_cost - puzzle_cost + gift

/-- Theorem stating that Oliver's final amount of money is $15. -/
theorem oliver_final_amount :
  olivers_money 9 5 4 3 8 = 15 := by
  sorry

#eval olivers_money 9 5 4 3 8

end NUMINAMATH_CALUDE_oliver_final_amount_l1562_156284


namespace NUMINAMATH_CALUDE_store_optimal_pricing_l1562_156277

/-- Represents the store's product information and pricing strategy. -/
structure Store where
  purchase_price_A : ℝ
  purchase_price_B : ℝ
  retail_price_A : ℝ
  retail_price_B : ℝ
  daily_sales : ℝ
  price_decrease : ℝ

/-- Conditions for the store's pricing and sales. -/
def store_conditions (s : Store) : Prop :=
  s.purchase_price_A + s.purchase_price_B = 3 ∧
  s.retail_price_A = s.purchase_price_A + 1 ∧
  s.retail_price_B = 2 * s.purchase_price_B - 1 ∧
  3 * s.retail_price_A + 2 * s.retail_price_B = 12 ∧
  s.daily_sales = 500 ∧
  s.price_decrease > 0

/-- The profit function for the store. -/
def profit (s : Store) : ℝ :=
  (s.retail_price_A - s.price_decrease) * (s.daily_sales + 1000 * s.price_decrease) + s.retail_price_B * s.daily_sales - (s.purchase_price_A + s.purchase_price_B) * s.daily_sales

/-- Theorem stating the correct retail prices and optimal price decrease for maximum profit. -/
theorem store_optimal_pricing (s : Store) (h : store_conditions s) :
  s.retail_price_A = 2 ∧ s.retail_price_B = 3 ∧ profit s = 1000 ↔ s.price_decrease = 0.5 := by
  sorry


end NUMINAMATH_CALUDE_store_optimal_pricing_l1562_156277


namespace NUMINAMATH_CALUDE_existence_of_g_l1562_156275

open Set
open Function
open ContinuousOn

theorem existence_of_g (a b : ℝ) (f : ℝ → ℝ) 
  (h_f_cont : ContinuousOn f (Icc a b))
  (h_f_deriv : DifferentiableOn ℝ f (Icc a b))
  (h_f_zero : ∀ x ∈ Icc a b, f x = 0 → deriv f x ≠ 0) :
  ∃ g : ℝ → ℝ, 
    ContinuousOn g (Icc a b) ∧ 
    DifferentiableOn ℝ g (Icc a b) ∧
    ∀ x ∈ Icc a b, f x * deriv g x > deriv f x * g x :=
sorry

end NUMINAMATH_CALUDE_existence_of_g_l1562_156275


namespace NUMINAMATH_CALUDE_wrapping_paper_area_l1562_156251

/-- A rectangular box with a square base -/
structure Box where
  base_side : ℝ
  height : ℝ
  height_eq_double_base : height = 2 * base_side

/-- A square sheet of wrapping paper -/
structure WrappingPaper where
  side_length : ℝ

/-- The configuration of the box on the wrapping paper -/
structure BoxWrappingConfiguration where
  box : Box
  paper : WrappingPaper
  box_centrally_placed : True
  vertices_on_midlines : True
  paper_folds_to_top_center : True

theorem wrapping_paper_area (config : BoxWrappingConfiguration) :
  config.paper.side_length ^ 2 = 16 * config.box.base_side ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_wrapping_paper_area_l1562_156251


namespace NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l1562_156202

def complex_number : ℂ := 2 + Complex.I

theorem complex_number_in_first_quadrant :
  Complex.re complex_number > 0 ∧ Complex.im complex_number > 0 :=
by sorry

end NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l1562_156202


namespace NUMINAMATH_CALUDE_abs_negative_six_l1562_156281

theorem abs_negative_six : |(-6 : ℤ)| = 6 := by sorry

end NUMINAMATH_CALUDE_abs_negative_six_l1562_156281


namespace NUMINAMATH_CALUDE_multiples_of_6_not_18_under_350_l1562_156210

def count_multiples (n : ℕ) (m : ℕ) : ℕ :=
  (n - 1) / m

theorem multiples_of_6_not_18_under_350 : 
  (count_multiples 350 6) - (count_multiples 350 18) = 39 := by
  sorry

end NUMINAMATH_CALUDE_multiples_of_6_not_18_under_350_l1562_156210


namespace NUMINAMATH_CALUDE_fiftieth_term_is_247_l1562_156298

/-- The nth term of an arithmetic sequence -/
def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  a₁ + (n - 1 : ℝ) * d

/-- Theorem: The 50th term of the arithmetic sequence starting with 2 and common difference 5 is 247 -/
theorem fiftieth_term_is_247 : arithmetic_sequence 2 5 50 = 247 := by
  sorry

end NUMINAMATH_CALUDE_fiftieth_term_is_247_l1562_156298


namespace NUMINAMATH_CALUDE_duodecimal_reversal_difference_divisibility_l1562_156243

/-- Represents a duodecimal digit (0 to 11) -/
def DuodecimalDigit := {n : ℕ // n ≤ 11}

/-- Converts a two-digit duodecimal number to its decimal representation -/
def toDecimal (a b : DuodecimalDigit) : ℤ :=
  12 * a.val + b.val

theorem duodecimal_reversal_difference_divisibility
  (a b : DuodecimalDigit)
  (h : a ≠ b) :
  ∃ k : ℤ, toDecimal a b - toDecimal b a = 11 * k := by
  sorry

end NUMINAMATH_CALUDE_duodecimal_reversal_difference_divisibility_l1562_156243


namespace NUMINAMATH_CALUDE_rowing_distance_l1562_156296

theorem rowing_distance (v_man : ℝ) (v_river : ℝ) (total_time : ℝ) :
  v_man = 8 →
  v_river = 2 →
  total_time = 1 →
  ∃ (distance : ℝ),
    distance / (v_man - v_river) + distance / (v_man + v_river) = total_time ∧
    2 * distance = 7.5 :=
by sorry

end NUMINAMATH_CALUDE_rowing_distance_l1562_156296


namespace NUMINAMATH_CALUDE_paintings_distribution_l1562_156237

theorem paintings_distribution (total_paintings : ℕ) (paintings_per_room : ℕ) (num_rooms : ℕ) :
  total_paintings = 32 →
  paintings_per_room = 8 →
  total_paintings = paintings_per_room * num_rooms →
  num_rooms = 4 := by
sorry

end NUMINAMATH_CALUDE_paintings_distribution_l1562_156237


namespace NUMINAMATH_CALUDE_probability_sum_greater_than_five_l1562_156228

def roll_die : Finset ℕ := Finset.range 6

theorem probability_sum_greater_than_five :
  let outcomes := (roll_die.product roll_die).filter (λ p => p.1 + p.2 > 5)
  (outcomes.card : ℚ) / (roll_die.card * roll_die.card) = 13 / 18 := by
sorry

end NUMINAMATH_CALUDE_probability_sum_greater_than_five_l1562_156228


namespace NUMINAMATH_CALUDE_polynomial_factorization_l1562_156278

theorem polynomial_factorization (a b : ℝ) : a^3*b - 9*a*b = a*b*(a+3)*(a-3) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l1562_156278


namespace NUMINAMATH_CALUDE_rectangle_area_rate_of_change_l1562_156233

/-- The rate of change of the area of a rectangle with varying sides -/
theorem rectangle_area_rate_of_change :
  let a : ℝ → ℝ := λ t => 2 * t + 1
  let b : ℝ → ℝ := λ t => 3 * t + 2
  let S : ℝ → ℝ := λ t => a t * b t
  (deriv S) 4 = 55 := by sorry

end NUMINAMATH_CALUDE_rectangle_area_rate_of_change_l1562_156233


namespace NUMINAMATH_CALUDE_intersection_point_is_unique_l1562_156211

/-- The line equation in R³ --/
def line (x y z : ℝ) : Prop :=
  (x + 2) / 1 = (y - 2) / 0 ∧ (x + 2) / 1 = (z + 3) / 0

/-- The plane equation in R³ --/
def plane (x y z : ℝ) : Prop :=
  2 * x - 3 * y - 5 * z - 7 = 0

/-- The intersection point of the line and the plane --/
def intersection_point : ℝ × ℝ × ℝ := (-1, 2, -3)

theorem intersection_point_is_unique :
  ∃! p : ℝ × ℝ × ℝ, 
    line p.1 p.2.1 p.2.2 ∧ 
    plane p.1 p.2.1 p.2.2 ∧ 
    p = intersection_point :=
by
  sorry

end NUMINAMATH_CALUDE_intersection_point_is_unique_l1562_156211


namespace NUMINAMATH_CALUDE_no_primes_divisible_by_77_l1562_156241

theorem no_primes_divisible_by_77 : ¬∃ p : ℕ, Nat.Prime p ∧ 77 ∣ p := by
  sorry

end NUMINAMATH_CALUDE_no_primes_divisible_by_77_l1562_156241


namespace NUMINAMATH_CALUDE_committee_seating_arrangements_l1562_156297

/-- The number of distinct arrangements of chairs and benches -/
def distinct_arrangements (total_positions : ℕ) (bench_count : ℕ) : ℕ :=
  Nat.choose total_positions bench_count

theorem committee_seating_arrangements :
  distinct_arrangements 14 4 = 1001 := by
  sorry

end NUMINAMATH_CALUDE_committee_seating_arrangements_l1562_156297


namespace NUMINAMATH_CALUDE_jason_seashells_theorem_l1562_156283

/-- Calculates the number of seashells Jason gave to Tim -/
def seashells_given_to_tim (initial_seashells current_seashells : ℕ) : ℕ :=
  initial_seashells - current_seashells

/-- Proves that the number of seashells Jason gave to Tim is correct -/
theorem jason_seashells_theorem (initial_seashells current_seashells : ℕ) 
  (h1 : initial_seashells = 49)
  (h2 : current_seashells = 36)
  (h3 : initial_seashells ≥ current_seashells) :
  seashells_given_to_tim initial_seashells current_seashells = 13 := by
  sorry

#eval seashells_given_to_tim 49 36

end NUMINAMATH_CALUDE_jason_seashells_theorem_l1562_156283


namespace NUMINAMATH_CALUDE_average_of_three_numbers_l1562_156221

theorem average_of_three_numbers (y : ℝ) : (15 + 25 + y) / 3 = 23 → y = 29 := by
  sorry

end NUMINAMATH_CALUDE_average_of_three_numbers_l1562_156221


namespace NUMINAMATH_CALUDE_john_necklaces_l1562_156255

/-- Given the number of wire spools, length of each spool, and wire required per necklace,
    calculate the number of necklaces that can be made. -/
def necklaces_from_wire (num_spools : ℕ) (spool_length : ℕ) (wire_per_necklace : ℕ) : ℕ :=
  (num_spools * spool_length) / wire_per_necklace

/-- Prove that John can make 15 necklaces with the given conditions. -/
theorem john_necklaces : necklaces_from_wire 3 20 4 = 15 := by
  sorry

end NUMINAMATH_CALUDE_john_necklaces_l1562_156255


namespace NUMINAMATH_CALUDE_parallel_perpendicular_implication_parallel_contained_implication_l1562_156254

structure GeometrySpace where
  Line : Type
  Plane : Type
  parallel_lines : Line → Line → Prop
  parallel_plane_line : Plane → Line → Prop
  parallel_planes : Plane → Plane → Prop
  perpendicular_plane_line : Plane → Line → Prop
  line_in_plane : Line → Plane → Prop
  line_not_in_plane : Line → Plane → Prop

variable (G : GeometrySpace)

theorem parallel_perpendicular_implication
  (m n : G.Line) (α β : G.Plane)
  (h1 : G.parallel_lines m n)
  (h2 : G.perpendicular_plane_line α m)
  (h3 : G.perpendicular_plane_line β n) :
  G.parallel_planes α β :=
sorry

theorem parallel_contained_implication
  (m n : G.Line) (α β : G.Plane)
  (h1 : G.parallel_lines m n)
  (h2 : G.line_in_plane n α)
  (h3 : G.parallel_planes α β)
  (h4 : G.line_not_in_plane m β) :
  G.parallel_plane_line β m :=
sorry

end NUMINAMATH_CALUDE_parallel_perpendicular_implication_parallel_contained_implication_l1562_156254


namespace NUMINAMATH_CALUDE_original_book_count_l1562_156276

/-- Represents a bookshelf with three layers of books -/
structure Bookshelf :=
  (layer1 : ℕ)
  (layer2 : ℕ)
  (layer3 : ℕ)

/-- The total number of books on the bookshelf -/
def total_books (b : Bookshelf) : ℕ := b.layer1 + b.layer2 + b.layer3

/-- The bookshelf after moving books between layers -/
def move_books (b : Bookshelf) : Bookshelf :=
  { layer1 := b.layer1 - 20,
    layer2 := b.layer2 + 20 + 17,
    layer3 := b.layer3 - 17 }

/-- Theorem stating the original number of books on each layer -/
theorem original_book_count :
  ∀ b : Bookshelf,
    total_books b = 270 →
    (let b' := move_books b
     b'.layer1 = b'.layer2 ∧ b'.layer2 = b'.layer3) →
    b.layer1 = 110 ∧ b.layer2 = 53 ∧ b.layer3 = 107 :=
by sorry


end NUMINAMATH_CALUDE_original_book_count_l1562_156276


namespace NUMINAMATH_CALUDE_number_of_factors_l1562_156252

theorem number_of_factors (n : ℕ+) : 
  (Finset.range n).card = n :=
by sorry

#check number_of_factors

end NUMINAMATH_CALUDE_number_of_factors_l1562_156252


namespace NUMINAMATH_CALUDE_sams_shirts_l1562_156288

theorem sams_shirts (short_sleeve : ℕ) (long_sleeve : ℕ) (washed : ℕ) (unwashed : ℕ) : 
  long_sleeve = 23 →
  washed = 29 →
  unwashed = 34 →
  short_sleeve + long_sleeve = washed + unwashed →
  short_sleeve = 40 := by
sorry

end NUMINAMATH_CALUDE_sams_shirts_l1562_156288


namespace NUMINAMATH_CALUDE_triangle_ABC_properties_l1562_156246

variable (a b c : ℝ)
variable (A B C : ℝ)

-- Define triangle ABC
def triangle_ABC (a b c A B C : ℝ) : Prop :=
  -- Sides a, b, c are opposite to angles A, B, C respectively
  true

-- Define the given equation
def given_equation (a b c A C : ℝ) : Prop :=
  (2 * b - c) * Real.cos A = a * Real.cos C

-- Define the given conditions
def given_conditions (a b c : ℝ) : Prop :=
  a = 2 ∧ b + c = 4

-- Theorem statement
theorem triangle_ABC_properties 
  (h_triangle : triangle_ABC a b c A B C)
  (h_equation : given_equation a b c A C)
  (h_conditions : given_conditions a b c) :
  A = Real.pi / 3 ∧ 
  (1/2) * b * c * Real.sin A = Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_triangle_ABC_properties_l1562_156246


namespace NUMINAMATH_CALUDE_range_of_a_l1562_156201

def P : Set ℝ := {x : ℝ | x^2 ≤ 4}
def M (a : ℝ) : Set ℝ := {a}

theorem range_of_a (a : ℝ) : P ∪ M a = P → a ∈ Set.Icc (-2) 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1562_156201


namespace NUMINAMATH_CALUDE_same_color_probability_l1562_156225

/-- The probability of drawing two balls of the same color from a bag with 6 green and 7 white balls -/
theorem same_color_probability (total_balls : ℕ) (green_balls : ℕ) (white_balls : ℕ)
  (h1 : total_balls = green_balls + white_balls)
  (h2 : green_balls = 6)
  (h3 : white_balls = 7) :
  (green_balls * (green_balls - 1) + white_balls * (white_balls - 1)) / (total_balls * (total_balls - 1)) = 6 / 13 := by
sorry

end NUMINAMATH_CALUDE_same_color_probability_l1562_156225


namespace NUMINAMATH_CALUDE_motorboat_trip_time_l1562_156274

theorem motorboat_trip_time (v_b : ℝ) (d : ℝ) (h1 : v_b > 0) (h2 : d > 0) : 
  let v_c := v_b / 3
  let t_no_current := 2 * d / v_b
  let v_down := v_b + v_c
  let v_up := v_b - v_c
  let t_actual := d / v_down + d / v_up
  t_no_current = 44 / 60 → t_actual = 49.5 / 60 := by
sorry

end NUMINAMATH_CALUDE_motorboat_trip_time_l1562_156274


namespace NUMINAMATH_CALUDE_equal_wins_losses_probability_l1562_156208

/-- Represents the result of a single match -/
inductive MatchResult
  | Win
  | Loss
  | Tie

/-- Probability distribution for match results -/
def matchProbability : MatchResult → Rat
  | MatchResult.Win => 1/4
  | MatchResult.Loss => 1/4
  | MatchResult.Tie => 1/2

/-- Total number of matches played -/
def totalMatches : Nat := 10

/-- Calculates the probability of having equal wins and losses in a season -/
def probabilityEqualWinsLosses : Rat :=
  63/262144

theorem equal_wins_losses_probability :
  probabilityEqualWinsLosses = 63/262144 := by
  sorry

#check equal_wins_losses_probability

end NUMINAMATH_CALUDE_equal_wins_losses_probability_l1562_156208


namespace NUMINAMATH_CALUDE_tan_half_angle_less_than_one_l1562_156218

theorem tan_half_angle_less_than_one (θ : Real) (h : 0 < θ ∧ θ < π / 2) : 
  Real.tan (θ / 2) < 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_half_angle_less_than_one_l1562_156218


namespace NUMINAMATH_CALUDE_range_of_a_l1562_156258

-- Define the propositions p and q
def p (x a : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0
def q (x : ℝ) : Prop := x^2 - x - 6 ≤ 0

-- Define the theorem
theorem range_of_a (a : ℝ) :
  a < 0 ∧
  (∀ x, p x a → q x) ∧
  (∃ x, q x ∧ ¬p x a) →
  -2/3 ≤ a ∧ a < 0 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l1562_156258


namespace NUMINAMATH_CALUDE_smallest_sum_of_reciprocals_l1562_156229

theorem smallest_sum_of_reciprocals (x y : ℕ+) : 
  x ≠ y → 
  (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / 20 → 
  (∀ a b : ℕ+, a ≠ b → (1 : ℚ) / a + (1 : ℚ) / b = (1 : ℚ) / 20 → (x : ℕ) + y ≤ (a : ℕ) + b) →
  (x : ℕ) + y = 81 :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_of_reciprocals_l1562_156229


namespace NUMINAMATH_CALUDE_hybrid_rice_yield_and_conversion_l1562_156235

-- Define the yield per acre of ordinary rice
def ordinary_yield : ℝ := 600

-- Define the yield per acre of hybrid rice
def hybrid_yield : ℝ := 1200

-- Define the acreage difference between fields
def acreage_difference : ℝ := 4

-- Define the harvest of field A (hybrid rice)
def field_A_harvest : ℝ := 9600

-- Define the harvest of field B (ordinary rice)
def field_B_harvest : ℝ := 7200

-- Define the total yield goal
def total_yield_goal : ℝ := 17700

-- Define the minimum acres to be converted
def min_acres_converted : ℝ := 1.5

-- Theorem statement
theorem hybrid_rice_yield_and_conversion :
  (hybrid_yield = 2 * ordinary_yield) ∧
  (field_B_harvest / ordinary_yield - field_A_harvest / hybrid_yield = acreage_difference) ∧
  (field_A_harvest + ordinary_yield * (field_B_harvest / ordinary_yield - min_acres_converted) + hybrid_yield * min_acres_converted ≥ total_yield_goal) := by
  sorry

end NUMINAMATH_CALUDE_hybrid_rice_yield_and_conversion_l1562_156235


namespace NUMINAMATH_CALUDE_cassidy_grounding_period_l1562_156260

/-- Calculate the total grounding period for Cassidy --/
theorem cassidy_grounding_period :
  let initial_grounding : ℕ := 14
  let below_b_penalty : ℕ := 3
  let main_below_b : ℕ := 4
  let extra_below_b : ℕ := 2
  let a_grades : ℕ := 2
  let main_penalty := (main_below_b * below_b_penalty ^ 2 : ℚ)
  let extra_penalty := (extra_below_b * (below_b_penalty / 2) ^ 2 : ℚ)
  let additional_days := main_penalty + extra_penalty
  let reduced_initial := initial_grounding - a_grades
  let total_days := reduced_initial + additional_days
  ⌈total_days⌉ = 53 := by sorry

end NUMINAMATH_CALUDE_cassidy_grounding_period_l1562_156260


namespace NUMINAMATH_CALUDE_rhombus_perimeter_l1562_156242

/-- The perimeter of a rhombus given its diagonals -/
theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 20) (h2 : d2 = 16) :
  4 * Real.sqrt ((d1/2)^2 + (d2/2)^2) = 8 * Real.sqrt 41 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_perimeter_l1562_156242


namespace NUMINAMATH_CALUDE_expression_evaluation_l1562_156257

theorem expression_evaluation (x y z : ℝ) 
  (hz : z = y - 11)
  (hy : y = x + 3)
  (hx : x = 5)
  (hd1 : x + 2 ≠ 0)
  (hd2 : y - 3 ≠ 0)
  (hd3 : z + 7 ≠ 0) :
  ((x + 3) / (x + 2)) * ((y - 1) / (y - 3)) * ((z + 9) / (z + 7)) = 2.4 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1562_156257


namespace NUMINAMATH_CALUDE_small_circle_radius_l1562_156268

/-- A design consisting of a small circle surrounded by four equal quarter-circle arcs -/
structure CircleDesign where
  /-- The radius of the large arcs -/
  R : ℝ
  /-- The radius of the small circle -/
  r : ℝ
  /-- The width of the design is 2 cm -/
  width_eq : R + r = 2

/-- The radius of the small circle in a CircleDesign with width 2 cm is 2 - √2 cm -/
theorem small_circle_radius (d : CircleDesign) : d.r = 2 - Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_small_circle_radius_l1562_156268


namespace NUMINAMATH_CALUDE_gaochun_population_eq_scientific_l1562_156227

/-- The population of Gaochun County -/
def gaochun_population : ℕ := 425000

/-- Scientific notation representation of Gaochun County's population -/
def gaochun_population_scientific : ℝ := 4.25 * (10 ^ 5)

/-- Theorem stating that the scientific notation representation is equal to the actual population -/
theorem gaochun_population_eq_scientific : ↑gaochun_population = gaochun_population_scientific := by
  sorry

end NUMINAMATH_CALUDE_gaochun_population_eq_scientific_l1562_156227


namespace NUMINAMATH_CALUDE_problem_solution_l1562_156291

theorem problem_solution (m : ℤ) (a b c : ℝ) 
  (h1 : ∃! (x : ℤ), |2 * (x : ℝ) - m| ≤ 1 ∧ x = 2)
  (h2 : 4 * a^4 + 4 * b^4 + 4 * c^4 = m) : 
  m = 4 ∧ a^2 + b^2 + c^2 ≤ Real.sqrt 3 ∧ 
  ∃ a₀ b₀ c₀ : ℝ, a₀^2 + b₀^2 + c₀^2 = Real.sqrt 3 ∧ 
  4 * a₀^4 + 4 * b₀^4 + 4 * c₀^4 = m := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1562_156291


namespace NUMINAMATH_CALUDE_happy_children_count_is_30_l1562_156240

/-- Represents the number of children in different categories -/
structure ChildrenCount where
  total : Nat
  sad : Nat
  neither : Nat
  boys : Nat
  girls : Nat
  happyBoys : Nat
  sadGirls : Nat
  neitherBoys : Nat

/-- Calculates the number of happy children given the conditions -/
def happyChildrenCount (c : ChildrenCount) : Nat :=
  c.total - c.sad - c.neither

/-- Theorem stating that the number of happy children is 30 -/
theorem happy_children_count_is_30 (c : ChildrenCount) 
  (h1 : c.total = 60)
  (h2 : c.sad = 10)
  (h3 : c.neither = 20)
  (h4 : c.boys = 22)
  (h5 : c.girls = 38)
  (h6 : c.happyBoys = 6)
  (h7 : c.sadGirls = 4)
  (h8 : c.neitherBoys = 10) :
  happyChildrenCount c = 30 := by
  sorry

#check happy_children_count_is_30

end NUMINAMATH_CALUDE_happy_children_count_is_30_l1562_156240


namespace NUMINAMATH_CALUDE_washington_high_ratio_l1562_156205

/-- The student-teacher ratio at Washington High School -/
def student_teacher_ratio (num_students : ℕ) (num_teachers : ℕ) : ℚ :=
  num_students / num_teachers

/-- Theorem: The student-teacher ratio at Washington High School is 27.5 to 1 -/
theorem washington_high_ratio :
  student_teacher_ratio 1155 42 = 27.5 := by
  sorry

end NUMINAMATH_CALUDE_washington_high_ratio_l1562_156205


namespace NUMINAMATH_CALUDE_smallest_positive_solution_l1562_156226

theorem smallest_positive_solution :
  ∃ (x : ℝ), x > 0 ∧ Real.sqrt (x + 1) = 3 * x - 1 ∧
  ∀ (y : ℝ), y > 0 ∧ Real.sqrt (y + 1) = 3 * y - 1 → x ≤ y :=
by
  -- The solution is x = 7/9
  use 7/9
  sorry

end NUMINAMATH_CALUDE_smallest_positive_solution_l1562_156226


namespace NUMINAMATH_CALUDE_factorization_of_polynomial_l1562_156250

theorem factorization_of_polynomial (x : ℝ) :
  x^2 + 6*x + 9 - 100*x^4 = (-10*x^2 + x + 3) * (10*x^2 + x + 3) :=
by sorry

end NUMINAMATH_CALUDE_factorization_of_polynomial_l1562_156250


namespace NUMINAMATH_CALUDE_integral_sqrt_one_minus_x_squared_plus_x_l1562_156273

theorem integral_sqrt_one_minus_x_squared_plus_x : 
  ∫ x in (-1)..1, (Real.sqrt (1 - x^2) + x) = π / 2 := by sorry

end NUMINAMATH_CALUDE_integral_sqrt_one_minus_x_squared_plus_x_l1562_156273


namespace NUMINAMATH_CALUDE_factor_t_squared_minus_144_l1562_156292

theorem factor_t_squared_minus_144 (t : ℝ) : t^2 - 144 = (t - 12) * (t + 12) := by
  sorry

end NUMINAMATH_CALUDE_factor_t_squared_minus_144_l1562_156292


namespace NUMINAMATH_CALUDE_max_value_3a_plus_b_l1562_156244

theorem max_value_3a_plus_b (a b : ℝ) (h : 9 * a^2 + b^2 - 6 * a - 2 * b = 0) :
  ∀ x y : ℝ, 9 * x^2 + y^2 - 6 * x - 2 * y = 0 → 3 * x + y ≤ 3 * a + b → 3 * a + b ≤ 4 :=
by sorry

end NUMINAMATH_CALUDE_max_value_3a_plus_b_l1562_156244


namespace NUMINAMATH_CALUDE_min_points_for_12_monochromatic_triangles_l1562_156286

/-- A coloring of edges in a complete graph with two colors -/
def TwoColoring (n : ℕ) := Fin n → Fin n → Bool

/-- The number of monochromatic triangles in a given coloring -/
def monochromaticTriangles (n : ℕ) (c : TwoColoring n) : ℕ := sorry

/-- The statement that for any two-coloring of Kn, there are at least 12 monochromatic triangles -/
def hasAtLeast12MonochromaticTriangles (n : ℕ) : Prop :=
  ∀ c : TwoColoring n, monochromaticTriangles n c ≥ 12

/-- The theorem stating that 9 is the minimum number of points satisfying the condition -/
theorem min_points_for_12_monochromatic_triangles :
  (hasAtLeast12MonochromaticTriangles 9) ∧ 
  (∀ m : ℕ, m < 9 → ¬(hasAtLeast12MonochromaticTriangles m)) :=
sorry

end NUMINAMATH_CALUDE_min_points_for_12_monochromatic_triangles_l1562_156286


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l1562_156234

theorem simplify_and_evaluate (a b : ℤ) (h1 : a = -2) (h2 : b = 1) :
  (a^2 + 2*a*b) - 2*(a^2 + 4*a*b - b) = 10 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l1562_156234


namespace NUMINAMATH_CALUDE_average_of_xyz_l1562_156204

theorem average_of_xyz (x y z : ℝ) (h : (5 / 4) * (x + y + z) = 15) : 
  (x + y + z) / 3 = 4 := by
sorry

end NUMINAMATH_CALUDE_average_of_xyz_l1562_156204


namespace NUMINAMATH_CALUDE_base4_sequence_implies_bcd_52_l1562_156279

/-- Represents a digit in base-4 --/
inductive Base4Digit
| A
| B
| C
| D

/-- Converts a Base4Digit to its numerical value --/
def base4DigitToInt (d : Base4Digit) : Nat :=
  match d with
  | Base4Digit.A => 0
  | Base4Digit.B => 1
  | Base4Digit.C => 2
  | Base4Digit.D => 3

/-- Represents a three-digit number in base-4 --/
structure Base4Number :=
  (hundreds : Base4Digit)
  (tens : Base4Digit)
  (ones : Base4Digit)

/-- Converts a Base4Number to its base-10 representation --/
def toBase10 (n : Base4Number) : Nat :=
  4 * 4 * (base4DigitToInt n.hundreds) + 4 * (base4DigitToInt n.tens) + (base4DigitToInt n.ones)

theorem base4_sequence_implies_bcd_52 
  (n1 n2 n3 : Base4Number)
  (h1 : toBase10 n2 = toBase10 n1 + 1)
  (h2 : toBase10 n3 = toBase10 n2 + 1)
  (h3 : n1.hundreds = n2.hundreds ∧ n1.tens = n2.tens)
  (h4 : n2.hundreds = n3.hundreds ∧ n3.tens = Base4Digit.C)
  (h5 : n1.hundreds = Base4Digit.A ∧ n2.hundreds = Base4Digit.A ∧ n3.hundreds = Base4Digit.A)
  (h6 : n1.tens = Base4Digit.B ∧ n2.tens = Base4Digit.B)
  (h7 : n1.ones = Base4Digit.C ∧ n2.ones = Base4Digit.D ∧ n3.ones = Base4Digit.A) :
  toBase10 { hundreds := Base4Digit.B, tens := Base4Digit.C, ones := Base4Digit.D } = 52 := by
  sorry

end NUMINAMATH_CALUDE_base4_sequence_implies_bcd_52_l1562_156279


namespace NUMINAMATH_CALUDE_cell_diameter_scientific_notation_l1562_156262

/-- Expresses a given number in scientific notation -/
def scientificNotation (n : ℝ) : ℝ × ℤ :=
  sorry

theorem cell_diameter_scientific_notation :
  scientificNotation 0.00065 = (6.5, -4) := by sorry

end NUMINAMATH_CALUDE_cell_diameter_scientific_notation_l1562_156262


namespace NUMINAMATH_CALUDE_xyz_mod_nine_l1562_156293

theorem xyz_mod_nine (x y z : ℕ) : 
  x < 9 → y < 9 → z < 9 →
  (x + 3*y + 2*z) % 9 = 0 →
  (2*x + 2*y + z) % 9 = 7 →
  (x + 2*y + 3*z) % 9 = 5 →
  (x*y*z) % 9 = 5 := by
  sorry

end NUMINAMATH_CALUDE_xyz_mod_nine_l1562_156293


namespace NUMINAMATH_CALUDE_divide_seven_students_three_groups_l1562_156206

/-- The number of ways to divide students into groups and send them to different places -/
def divideAndSend (n : ℕ) (k : ℕ) (ratio : List ℕ) : ℕ :=
  sorry

/-- The theorem stating the correct number of ways for the given problem -/
theorem divide_seven_students_three_groups : divideAndSend 7 3 [3, 2, 2] = 630 := by
  sorry

end NUMINAMATH_CALUDE_divide_seven_students_three_groups_l1562_156206


namespace NUMINAMATH_CALUDE_side_length_S2_correct_l1562_156224

/-- The side length of square S2 in a specific rectangular arrangement -/
def side_length_S2 : ℕ := 650

/-- The width of the overall rectangle -/
def total_width : ℕ := 3400

/-- The height of the overall rectangle -/
def total_height : ℕ := 2100

/-- Theorem stating that the side length of S2 is correct given the constraints -/
theorem side_length_S2_correct :
  ∃ (r : ℕ),
    (2 * r + side_length_S2 = total_height) ∧
    (2 * r + 3 * side_length_S2 = total_width) := by
  sorry

end NUMINAMATH_CALUDE_side_length_S2_correct_l1562_156224


namespace NUMINAMATH_CALUDE_ad_square_area_l1562_156272

/-- Given two joined right triangles ABC and ACD with squares on their sides -/
structure JoinedTriangles where
  /-- Area of square on side AB -/
  ab_square_area : ℝ
  /-- Area of square on side BC -/
  bc_square_area : ℝ
  /-- Area of square on side CD -/
  cd_square_area : ℝ
  /-- ABC is a right triangle -/
  abc_right : True
  /-- ACD is a right triangle -/
  acd_right : True

/-- The theorem stating the area of the square on AD -/
theorem ad_square_area (t : JoinedTriangles)
  (h1 : t.ab_square_area = 36)
  (h2 : t.bc_square_area = 9)
  (h3 : t.cd_square_area = 16) :
  ∃ (ad_square_area : ℝ), ad_square_area = 61 := by
  sorry

end NUMINAMATH_CALUDE_ad_square_area_l1562_156272


namespace NUMINAMATH_CALUDE_muffin_banana_cost_ratio_l1562_156265

theorem muffin_banana_cost_ratio :
  ∀ (muffin_cost banana_cost : ℝ),
  (6 * muffin_cost + 5 * banana_cost = (3 * muffin_cost + 20 * banana_cost) / 2) →
  (muffin_cost / banana_cost = 10 / 9) :=
by
  sorry

end NUMINAMATH_CALUDE_muffin_banana_cost_ratio_l1562_156265


namespace NUMINAMATH_CALUDE_sum_of_x_coordinates_l1562_156263

/-- Triangle XYZ -/
structure TriangleXYZ where
  X : ℝ × ℝ
  Y : ℝ × ℝ := (0, 0)
  Z : ℝ × ℝ := (150, 0)
  area : ℝ := 1200

/-- Triangle XWV -/
structure TriangleXWV where
  X : ℝ × ℝ
  W : ℝ × ℝ := (500, 300)
  V : ℝ × ℝ := (510, 290)
  area : ℝ := 3600

/-- The theorem stating that the sum of all possible x-coordinates of X is 3200 -/
theorem sum_of_x_coordinates (triangle_xyz : TriangleXYZ) (triangle_xwv : TriangleXWV) 
  (h : triangle_xyz.X = triangle_xwv.X) :
  ∃ (x₁ x₂ x₃ x₄ : ℝ), (x₁ + x₂ + x₃ + x₄ = 3200 ∧ 
    (triangle_xyz.X.1 = x₁ ∨ triangle_xyz.X.1 = x₂ ∨ triangle_xyz.X.1 = x₃ ∨ triangle_xyz.X.1 = x₄)) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_x_coordinates_l1562_156263


namespace NUMINAMATH_CALUDE_class_size_l1562_156256

theorem class_size (football : ℕ) (tennis : ℕ) (both : ℕ) (neither : ℕ) 
  (h1 : football = 26)
  (h2 : tennis = 20)
  (h3 : both = 17)
  (h4 : neither = 10) :
  football + tennis - both + neither = 39 := by
  sorry

end NUMINAMATH_CALUDE_class_size_l1562_156256


namespace NUMINAMATH_CALUDE_radar_arrangements_l1562_156289

def word_length : ℕ := 5
def r_count : ℕ := 2
def a_count : ℕ := 2

theorem radar_arrangements : 
  (word_length.factorial) / (r_count.factorial * a_count.factorial) = 30 := by
  sorry

end NUMINAMATH_CALUDE_radar_arrangements_l1562_156289


namespace NUMINAMATH_CALUDE_age_ratio_theorem_l1562_156261

/-- Represents the ages of John and Emily -/
structure Ages where
  john : ℕ
  emily : ℕ

/-- The conditions of the problem -/
def problem_conditions (ages : Ages) : Prop :=
  (ages.john - 3 = 5 * (ages.emily - 3)) ∧
  (ages.john - 7 = 6 * (ages.emily - 7))

/-- The theorem to be proved -/
theorem age_ratio_theorem (ages : Ages) :
  problem_conditions ages →
  ∃ x : ℕ, x = 17 ∧ (ages.john + x) / (ages.emily + x) = 3 :=
by
  sorry


end NUMINAMATH_CALUDE_age_ratio_theorem_l1562_156261


namespace NUMINAMATH_CALUDE_expression_perfect_square_iff_A_specific_values_l1562_156219

/-- A monomial is a term of the form cx^n where c is a constant and n is a non-negative integer. -/
def Monomial (x : ℝ) := ℝ → ℝ

/-- The expression x^6 + x^4 + xA -/
def Expression (x : ℝ) (A : Monomial x) : ℝ := x^6 + x^4 + x * A x

/-- A perfect square is a number that is the square of an integer. -/
def IsPerfectSquare (n : ℝ) : Prop := ∃ m : ℝ, n = m^2

theorem expression_perfect_square_iff_A_specific_values (x : ℝ) (A : Monomial x) :
  IsPerfectSquare (Expression x A) ↔ 
  (A = λ x => 2 * x^4) ∨ 
  (A = λ x => -2 * x^4) ∨ 
  (A = λ x => (1/4) * x^7) ∨ 
  (A = λ x => (1/4) * x) :=
sorry

end NUMINAMATH_CALUDE_expression_perfect_square_iff_A_specific_values_l1562_156219


namespace NUMINAMATH_CALUDE_total_ages_l1562_156247

/-- Given that Gabriel is 3 years younger than Frank and Frank is 10 years old,
    prove that the total of their ages is 17. -/
theorem total_ages (frank_age : ℕ) (gabriel_age : ℕ) : 
  frank_age = 10 → gabriel_age = frank_age - 3 → frank_age + gabriel_age = 17 := by
  sorry

end NUMINAMATH_CALUDE_total_ages_l1562_156247


namespace NUMINAMATH_CALUDE_painted_fraction_is_three_eighths_l1562_156285

/-- Represents a square plate with sides of length 4 meters -/
structure Plate :=
  (side_length : ℝ)
  (area : ℝ)
  (h_side : side_length = 4)
  (h_area : area = side_length * side_length)

/-- Represents the number of equal parts the plate is divided into -/
def total_parts : ℕ := 16

/-- Represents the number of painted parts -/
def painted_parts : ℕ := 6

/-- The theorem to be proved -/
theorem painted_fraction_is_three_eighths (plate : Plate) :
  (painted_parts : ℝ) / total_parts = 3 / 8 := by
  sorry


end NUMINAMATH_CALUDE_painted_fraction_is_three_eighths_l1562_156285


namespace NUMINAMATH_CALUDE_cheryl_pesto_production_l1562_156239

/-- Represents the pesto production scenario -/
structure PestoProduction where
  basil_per_pesto : ℕ  -- cups of basil needed for 1 cup of pesto
  basil_per_week : ℕ   -- cups of basil harvested per week
  harvest_weeks : ℕ    -- number of weeks of harvest

/-- Calculates the total cups of pesto that can be produced -/
def total_pesto (p : PestoProduction) : ℕ :=
  (p.basil_per_week * p.harvest_weeks) / p.basil_per_pesto

/-- Theorem: Given the conditions, Cheryl can make 32 cups of pesto -/
theorem cheryl_pesto_production :
  let p := PestoProduction.mk 4 16 8
  total_pesto p = 32 := by
  sorry

end NUMINAMATH_CALUDE_cheryl_pesto_production_l1562_156239


namespace NUMINAMATH_CALUDE_max_k_for_sqrt_inequality_l1562_156267

theorem max_k_for_sqrt_inequality : 
  (∃ (k : ℝ), ∀ (l : ℝ), 
    (∃ (x : ℝ), 3 ≤ x ∧ x ≤ 6 ∧ Real.sqrt (x - 3) + Real.sqrt (6 - x) ≥ l) → 
    k ≥ l) ∧ 
  (∃ (x : ℝ), 3 ≤ x ∧ x ≤ 6 ∧ Real.sqrt (x - 3) + Real.sqrt (6 - x) ≥ Real.sqrt 6) :=
by sorry

end NUMINAMATH_CALUDE_max_k_for_sqrt_inequality_l1562_156267


namespace NUMINAMATH_CALUDE_N_satisfies_equation_l1562_156209

def N : Matrix (Fin 2) (Fin 2) ℝ := !![2, 2; 1, 2]

theorem N_satisfies_equation : 
  N^3 - 3 • N^2 + 4 • N = !![6, 12; 3, 6] := by sorry

end NUMINAMATH_CALUDE_N_satisfies_equation_l1562_156209


namespace NUMINAMATH_CALUDE_prob_no_roots_l1562_156266

/-- A random variable following a normal distribution with mean 1 and variance s² -/
def normal_dist (s : ℝ) : Type := ℝ

/-- The probability density function of a normal distribution -/
noncomputable def pdf (s : ℝ) (x : ℝ) : ℝ := sorry

/-- The cumulative distribution function of a normal distribution -/
noncomputable def cdf (s : ℝ) (x : ℝ) : ℝ := sorry

/-- The quadratic function f(x) = x² + 2x + ξ -/
def f (ξ : ℝ) (x : ℝ) : ℝ := x^2 + 2*x + ξ

/-- The statement that f(x) has no roots -/
def no_roots (ξ : ℝ) : Prop := ∀ x, f ξ x ≠ 0

/-- The main theorem -/
theorem prob_no_roots (s : ℝ) (h : s > 0) : 
  (1 - cdf s 1) = 1/2 := by sorry

end NUMINAMATH_CALUDE_prob_no_roots_l1562_156266


namespace NUMINAMATH_CALUDE_hcf_problem_l1562_156238

/-- Given two positive integers with specific properties, prove their HCF is 24 -/
theorem hcf_problem (a b : ℕ) : 
  (a > 0) → 
  (b > 0) → 
  (a ≤ b) → 
  (b = 744) → 
  (Nat.lcm a b = Nat.gcd a b * 11 * 12) → 
  Nat.gcd a b = 24 := by
sorry

end NUMINAMATH_CALUDE_hcf_problem_l1562_156238


namespace NUMINAMATH_CALUDE_roberts_chocolates_l1562_156282

theorem roberts_chocolates (nickel_chocolates : ℕ) (robert_extra : ℕ) : 
  nickel_chocolates = 4 → robert_extra = 9 → nickel_chocolates + robert_extra = 13 :=
by
  sorry

end NUMINAMATH_CALUDE_roberts_chocolates_l1562_156282


namespace NUMINAMATH_CALUDE_chocolate_bar_count_l1562_156245

/-- Represents the number of chocolate bars in a crate -/
def chocolate_bars_in_crate (large_boxes : ℕ) (small_boxes_per_large : ℕ) (bars_per_small : ℕ) : ℕ :=
  large_boxes * small_boxes_per_large * bars_per_small

/-- Proves that the total number of chocolate bars in the crate is 116,640 -/
theorem chocolate_bar_count :
  chocolate_bars_in_crate 45 36 72 = 116640 := by
  sorry

#eval chocolate_bars_in_crate 45 36 72

end NUMINAMATH_CALUDE_chocolate_bar_count_l1562_156245


namespace NUMINAMATH_CALUDE_segment_shadow_ratio_constant_l1562_156230

/-- Given two segments and their shadows, prove that the ratio of segment length to shadow length is constant. -/
theorem segment_shadow_ratio_constant 
  (a b a' b' : ℝ) 
  (ha : a > 0) (hb : b > 0) (ha' : a' > 0) (hb' : b' > 0)
  (h_fixed_lines : FixedLines) 
  (h_fixed_projection : FixedProjection) : 
  a / a' = b / b' := by
  sorry

/-- Represents the condition that the lines on which the segments and shadows lie are fixed. -/
structure FixedLines where
  -- Additional properties can be added if needed

/-- Represents the condition that the direction of projection is fixed. -/
structure FixedProjection where
  -- Additional properties can be added if needed

end NUMINAMATH_CALUDE_segment_shadow_ratio_constant_l1562_156230


namespace NUMINAMATH_CALUDE_emilio_gifts_l1562_156231

theorem emilio_gifts (total gifts_from_jorge gifts_from_pedro : ℕ) 
  (h1 : gifts_from_jorge = 6)
  (h2 : gifts_from_pedro = 4)
  (h3 : total = 21) :
  total - gifts_from_jorge - gifts_from_pedro = 11 := by
  sorry

end NUMINAMATH_CALUDE_emilio_gifts_l1562_156231


namespace NUMINAMATH_CALUDE_max_value_constraint_l1562_156215

theorem max_value_constraint (x y : ℝ) (h : 9*x^2 + y^2 + x*y = 1) :
  ∃ (M : ℝ), M = (2 * Real.sqrt 21) / 7 ∧ 3*x + y ≤ M :=
sorry

end NUMINAMATH_CALUDE_max_value_constraint_l1562_156215


namespace NUMINAMATH_CALUDE_division_problem_l1562_156207

theorem division_problem (dividend : ℕ) (divisor : ℕ) (remainder : ℕ) (quotient : ℕ) 
  (h1 : dividend = 686)
  (h2 : divisor = 36)
  (h3 : remainder = 2)
  (h4 : dividend = divisor * quotient + remainder) :
  quotient = 19 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l1562_156207


namespace NUMINAMATH_CALUDE_sum_of_digits_999_base_7_l1562_156223

def base_7_representation (n : ℕ) : List ℕ :=
  sorry

def sum_of_digits (digits : List ℕ) : ℕ :=
  sorry

theorem sum_of_digits_999_base_7 :
  sum_of_digits (base_7_representation 999) = 15 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_999_base_7_l1562_156223


namespace NUMINAMATH_CALUDE_volleyball_team_math_count_l1562_156212

theorem volleyball_team_math_count (total_players : ℕ) (physics_players : ℕ) (both_players : ℕ) 
  (h1 : total_players = 25)
  (h2 : physics_players = 10)
  (h3 : both_players = 6)
  (h4 : physics_players ≥ both_players)
  (h5 : ∀ player, player ∈ Set.range (Fin.val : Fin total_players → ℕ) → 
    (player ∈ Set.range (Fin.val : Fin physics_players → ℕ) ∨ 
     player ∈ Set.range (Fin.val : Fin (total_players - physics_players + both_players) → ℕ))) :
  total_players - physics_players + both_players = 21 := by
  sorry

end NUMINAMATH_CALUDE_volleyball_team_math_count_l1562_156212


namespace NUMINAMATH_CALUDE_hyejin_math_score_l1562_156220

/-- Given Hyejin's scores and average, prove the mathematics score -/
theorem hyejin_math_score (ethics : ℕ) (korean : ℕ) (science : ℕ) (social : ℕ) 
  (h1 : ethics = 82) (h2 : korean = 90) (h3 : science = 88) (h4 : social = 84)
  (average : ℚ) (h5 : average = 88) :
  let total := ethics + korean + science + social
  let math := 5 * average - total
  math = 96 := by sorry

end NUMINAMATH_CALUDE_hyejin_math_score_l1562_156220


namespace NUMINAMATH_CALUDE_fifteen_sided_polygon_diagonals_l1562_156232

/-- A convex polygon is a polygon where every interior angle is less than 180 degrees. -/
structure ConvexPolygon where
  sides : ℕ
  is_convex : True

/-- A diagonal of a convex polygon is a segment not joining adjacent vertices. -/
def diagonals (p : ConvexPolygon) : ℕ :=
  (p.sides * (p.sides - 3)) / 2

/-- Theorem: A convex polygon with 15 sides has 90 diagonals. -/
theorem fifteen_sided_polygon_diagonals :
  ∀ p : ConvexPolygon, p.sides = 15 → diagonals p = 90 := by
  sorry

end NUMINAMATH_CALUDE_fifteen_sided_polygon_diagonals_l1562_156232


namespace NUMINAMATH_CALUDE_functional_equation_solution_l1562_156222

/-- Given a fixed positive integer N, prove that any function f satisfying
    the given conditions is identically zero. -/
theorem functional_equation_solution (N : ℕ+) (f : ℤ → ℝ)
  (h1 : ∀ k : ℤ, f (2 * k) = 2 * f k)
  (h2 : ∀ k : ℤ, f (N - k) = f k) :
  ∀ a : ℤ, f a = 0 := by sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l1562_156222


namespace NUMINAMATH_CALUDE_hyperbola_theorem_l1562_156299

/-- A hyperbola C that shares a common asymptote with x^2 - 2y^2 = 2 and passes through (2, -2) -/
structure Hyperbola where
  -- The equation of the hyperbola in the form y^2/a^2 - x^2/b^2 = 1
  a : ℝ
  b : ℝ
  -- The hyperbola passes through (2, -2)
  point_condition : (2 : ℝ)^2 / b^2 - (-2 : ℝ)^2 / a^2 = 1
  -- The hyperbola shares a common asymptote with x^2 - 2y^2 = 2
  asymptote_condition : a^2 / b^2 = 2

/-- Properties of the hyperbola C -/
def hyperbola_properties (C : Hyperbola) : Prop :=
  -- The equation of C is y^2/2 - x^2/4 = 1
  C.a^2 = 2 ∧ C.b^2 = 4 ∧
  -- The eccentricity of C is √3
  Real.sqrt ((C.a^2 + C.b^2) / C.a^2) = Real.sqrt 3 ∧
  -- The asymptotes of C are y = ±(√2/2)x
  ∀ (x y : ℝ), (y = Real.sqrt 2 / 2 * x ∨ y = -Real.sqrt 2 / 2 * x) ↔ 
    (y^2 / C.a^2 - x^2 / C.b^2 = 0)

/-- Main theorem: The hyperbola C satisfies the required properties -/
theorem hyperbola_theorem (C : Hyperbola) : hyperbola_properties C :=
sorry

end NUMINAMATH_CALUDE_hyperbola_theorem_l1562_156299


namespace NUMINAMATH_CALUDE_minimize_y_l1562_156269

/-- The function y in terms of x, a, and b -/
def y (x a b : ℝ) : ℝ := 2 * (x - a)^2 + 3 * (x - b)^2

/-- The theorem stating that (2a + 3b) / 5 minimizes y -/
theorem minimize_y (a b : ℝ) :
  let x_min := (2 * a + 3 * b) / 5
  ∀ x, y x_min a b ≤ y x a b :=
by sorry

end NUMINAMATH_CALUDE_minimize_y_l1562_156269


namespace NUMINAMATH_CALUDE_trapezoid_shorter_base_l1562_156216

/-- A trapezoid with the given properties -/
structure Trapezoid where
  longer_base : ℝ
  shorter_base : ℝ
  midpoint_line : ℝ
  longer_base_length : longer_base = 97
  midpoint_line_length : midpoint_line = 3
  midpoint_property : midpoint_line = (longer_base - shorter_base) / 2

theorem trapezoid_shorter_base (t : Trapezoid) : t.shorter_base = 91 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_shorter_base_l1562_156216


namespace NUMINAMATH_CALUDE_min_value_in_region_l1562_156264

-- Define the region
def in_region (x y : ℝ) : Prop :=
  y ≥ |x - 1| ∧ y ≤ 2

-- Define the function to be minimized
def f (x y : ℝ) : ℝ := 2*x - y

-- Theorem statement
theorem min_value_in_region :
  ∃ (min : ℝ), min = -4 ∧
  (∀ x y : ℝ, in_region x y → f x y ≥ min) ∧
  (∃ x y : ℝ, in_region x y ∧ f x y = min) :=
sorry

end NUMINAMATH_CALUDE_min_value_in_region_l1562_156264


namespace NUMINAMATH_CALUDE_rectangular_field_area_l1562_156259

/-- A rectangular field with breadth 60% of length and perimeter 800 m has area 37500 m² -/
theorem rectangular_field_area (length breadth : ℝ) : 
  breadth = 0.6 * length →
  2 * (length + breadth) = 800 →
  length * breadth = 37500 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_field_area_l1562_156259


namespace NUMINAMATH_CALUDE_distance_to_school_prove_distance_to_school_l1562_156270

theorem distance_to_school (time_with_traffic time_without_traffic : ℝ)
  (speed_difference : ℝ) (distance : ℝ) : Prop :=
  time_with_traffic = 20 / 60 →
  time_without_traffic = 15 / 60 →
  speed_difference = 15 →
  ∃ (speed_with_traffic : ℝ),
    distance = speed_with_traffic * time_with_traffic ∧
    distance = (speed_with_traffic + speed_difference) * time_without_traffic →
  distance = 15

-- The proof of the theorem
theorem prove_distance_to_school :
  ∀ (time_with_traffic time_without_traffic speed_difference distance : ℝ),
  distance_to_school time_with_traffic time_without_traffic speed_difference distance :=
by
  sorry

end NUMINAMATH_CALUDE_distance_to_school_prove_distance_to_school_l1562_156270
