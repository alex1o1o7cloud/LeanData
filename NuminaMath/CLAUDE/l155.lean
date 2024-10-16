import Mathlib

namespace NUMINAMATH_CALUDE_power_sum_integer_l155_15565

theorem power_sum_integer (x : ℝ) (h1 : x ≠ 0) (h2 : ∃ k : ℤ, x + 1/x = k) :
  ∀ n : ℕ, ∃ m : ℤ, x^n + 1/(x^n) = m :=
sorry

end NUMINAMATH_CALUDE_power_sum_integer_l155_15565


namespace NUMINAMATH_CALUDE_solution_part1_solution_part2_l155_15579

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| - 1

-- Part 1: Prove the solution set for f(x) + |2x-3| > 0 when a = 2
theorem solution_part1 : 
  {x : ℝ | f 2 x + |2*x - 3| > 0} = {x : ℝ | x ≥ 2 ∨ x ≤ 4/3} := by sorry

-- Part 2: Prove the range of a for which f(x) > |x-3| has solutions
theorem solution_part2 : 
  {a : ℝ | ∃ x, f a x > |x - 3|} = {a : ℝ | a < 2 ∨ a > 4} := by sorry

end NUMINAMATH_CALUDE_solution_part1_solution_part2_l155_15579


namespace NUMINAMATH_CALUDE_fluffy_arrangements_eq_72_l155_15515

/-- The number of distinct four-letter arrangements from the letters in "FLUFFY" -/
def fluffy_arrangements : ℕ :=
  let f_count := 3  -- Number of F's in FLUFFY
  let other_letters := 3  -- Number of other distinct letters (L, U, Y)
  let arrangement_size := 4  -- Size of each arrangement

  -- Case 1: Using 1 F
  let case1 := (arrangement_size.factorial) *
               (Nat.choose other_letters (arrangement_size - 1))

  -- Case 2: Using 2 F's
  let case2 := (arrangement_size.factorial / 2) *
               (Nat.choose other_letters (arrangement_size - 2))

  -- Case 3: Using 3 F's
  let case3 := (arrangement_size.factorial / 6) *
               (Nat.choose other_letters (arrangement_size - 3))

  -- Sum of all cases
  case1 + case2 + case3

/-- The number of distinct four-letter arrangements from the letters in "FLUFFY" is 72 -/
theorem fluffy_arrangements_eq_72 : fluffy_arrangements = 72 := by
  sorry

end NUMINAMATH_CALUDE_fluffy_arrangements_eq_72_l155_15515


namespace NUMINAMATH_CALUDE_sandra_leftover_money_l155_15563

def sandra_savings : ℕ := 10
def mother_gift : ℕ := 4
def father_gift : ℕ := 2 * mother_gift
def candy_cost : ℚ := 1/2
def jelly_bean_cost : ℚ := 1/5
def candy_count : ℕ := 14
def jelly_bean_count : ℕ := 20

theorem sandra_leftover_money :
  (sandra_savings + mother_gift + father_gift : ℚ) - 
  (candy_count * candy_cost + jelly_bean_count * jelly_bean_cost) = 11 :=
by sorry

end NUMINAMATH_CALUDE_sandra_leftover_money_l155_15563


namespace NUMINAMATH_CALUDE_modified_counting_game_l155_15596

theorem modified_counting_game (n : ℕ) (a₁ : ℕ) (d : ℕ) (aₙ : ℕ → ℕ) :
  a₁ = 1 →
  d = 2 →
  (∀ k, aₙ k = a₁ + (k - 1) * d) →
  aₙ 53 = 105 :=
by sorry

end NUMINAMATH_CALUDE_modified_counting_game_l155_15596


namespace NUMINAMATH_CALUDE_four_heads_before_three_tails_l155_15513

/-- The probability of getting heads in a fair coin flip -/
def p_heads : ℚ := 1/2

/-- The probability of getting tails in a fair coin flip -/
def p_tails : ℚ := 1/2

/-- The probability of encountering 4 heads before 3 tails in repeated fair coin flips -/
noncomputable def q : ℚ := sorry

theorem four_heads_before_three_tails : q = 15/23 := by sorry

end NUMINAMATH_CALUDE_four_heads_before_three_tails_l155_15513


namespace NUMINAMATH_CALUDE_max_value_real_complex_l155_15524

theorem max_value_real_complex (α β : ℝ) :
  (∃ (M : ℝ), ∀ (x y : ℝ), abs x ≤ 1 → abs y ≤ 1 →
    abs (α * x + β * y) + abs (α * x - β * y) ≤ M ∧
    M = 2 * Real.sqrt 2 * Real.sqrt (α^2 + β^2)) ∧
  (∃ (N : ℝ), ∀ (x y : ℂ), Complex.abs x ≤ 1 → Complex.abs y ≤ 1 →
    Complex.abs (α * x + β * y) + Complex.abs (α * x - β * y) ≤ N ∧
    N = 2 * abs α + 2 * abs β) :=
by sorry

end NUMINAMATH_CALUDE_max_value_real_complex_l155_15524


namespace NUMINAMATH_CALUDE_quadratic_roots_difference_l155_15560

theorem quadratic_roots_difference (p : ℝ) : 
  let a : ℝ := 1
  let b : ℝ := -(2*p + 1)
  let c : ℝ := p*(p + 1)
  let discriminant := b^2 - 4*a*c
  let root1 := (-b + Real.sqrt discriminant) / (2*a)
  let root2 := (-b - Real.sqrt discriminant) / (2*a)
  max root1 root2 - min root1 root2 = 1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_difference_l155_15560


namespace NUMINAMATH_CALUDE_fixed_point_on_line_l155_15597

theorem fixed_point_on_line (a b : ℝ) (h : a + 2 * b = 1) :
  a * (1/2) + 3 * (-1/6) + b = 0 := by
sorry

end NUMINAMATH_CALUDE_fixed_point_on_line_l155_15597


namespace NUMINAMATH_CALUDE_sum_excluding_20_formula_l155_15576

/-- The sum of natural numbers from 1 to n, excluding 20 -/
def sum_excluding_20 (n : ℕ) : ℕ := 
  (Finset.range n).sum id - if n ≥ 20 then 20 else 0

/-- Theorem: For any natural number n > 20, the sum of all natural numbers 
    from 1 to n, excluding 20, is equal to n(n+1)/2 - 20 -/
theorem sum_excluding_20_formula {n : ℕ} (h : n > 20) : 
  sum_excluding_20 n = n * (n + 1) / 2 - 20 := by
  sorry


end NUMINAMATH_CALUDE_sum_excluding_20_formula_l155_15576


namespace NUMINAMATH_CALUDE_unique_integer_angle_geometric_progression_l155_15538

theorem unique_integer_angle_geometric_progression :
  ∃! (a b c : ℕ+), a + b + c = 180 ∧ 
  ∃ (r : ℚ), r > 1 ∧ b = a * (r : ℚ) ∧ c = b * (r : ℚ) ∧
  a ≠ b ∧ b ≠ c ∧ a ≠ c :=
by sorry

end NUMINAMATH_CALUDE_unique_integer_angle_geometric_progression_l155_15538


namespace NUMINAMATH_CALUDE_last_two_digits_sum_factorials_50_l155_15549

/-- The last two digits of a natural number -/
def lastTwoDigits (n : ℕ) : ℕ := n % 100

/-- The sum of factorials from 1 to n -/
def sumFactorials (n : ℕ) : ℕ := (List.range n).map Nat.factorial |>.sum

/-- The last two digits of the sum of factorials from 1 to 50 are 13 -/
theorem last_two_digits_sum_factorials_50 :
  lastTwoDigits (sumFactorials 50) = 13 := by
  sorry

end NUMINAMATH_CALUDE_last_two_digits_sum_factorials_50_l155_15549


namespace NUMINAMATH_CALUDE_infinitely_many_prime_divisors_l155_15562

def a : ℕ → ℕ
  | 0 => 1
  | n + 1 => (a n)^2 + 1

def is_prime_divisor_of_sequence (p : ℕ) : Prop :=
  ∃ n : ℕ, p.Prime ∧ p ∣ a n

theorem infinitely_many_prime_divisors :
  ∀ m : ℕ, ∃ p : ℕ, p > m ∧ is_prime_divisor_of_sequence p :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_prime_divisors_l155_15562


namespace NUMINAMATH_CALUDE_intersection_distance_l155_15598

theorem intersection_distance : ∃ (p q : ℕ+), 
  (∀ (d : ℕ+), d ∣ p ∧ d ∣ q → d = 1) ∧ 
  (∃ (x₁ x₂ : ℝ), 
    2 = x₁^2 + 2*x₁ - 2 ∧ 
    2 = x₂^2 + 2*x₂ - 2 ∧ 
    (x₂ - x₁)^2 = 20 ∧
    (x₂ - x₁)^2 * q^2 = p) ∧
  p - q = 19 :=
sorry

end NUMINAMATH_CALUDE_intersection_distance_l155_15598


namespace NUMINAMATH_CALUDE_range_of_a_l155_15578

open Real

theorem range_of_a (x₁ x₂ a : ℝ) (h_pos₁ : x₁ > 0) (h_pos₂ : x₂ > 0) (h_distinct : x₁ ≠ x₂)
  (h_equation : x₁ + a * (x₂ - 2 * ℯ * x₁) * (log x₂ - log x₁) = 0) :
  a < 0 ∨ a ≥ 1 / ℯ := by sorry

end NUMINAMATH_CALUDE_range_of_a_l155_15578


namespace NUMINAMATH_CALUDE_minimum_blue_beads_l155_15545

/-- Represents the color of a bead -/
inductive BeadColor
  | Red
  | Blue
  | Green

/-- Represents a necklace as a cyclic list of bead colors -/
def Necklace := List BeadColor

/-- Returns true if the necklace satisfies the condition that each red bead has neighbors of different colors -/
def redBeadCondition (n : Necklace) : Prop := sorry

/-- Returns true if the necklace satisfies the condition that any segment between two green beads contains at least one blue bead -/
def greenSegmentCondition (n : Necklace) : Prop := sorry

/-- Counts the number of blue beads in the necklace -/
def countBlueBeads (n : Necklace) : Nat := sorry

theorem minimum_blue_beads (n : Necklace) :
  n.length = 175 →
  redBeadCondition n →
  greenSegmentCondition n →
  countBlueBeads n ≥ 30 ∧ ∃ (m : Necklace), m.length = 175 ∧ redBeadCondition m ∧ greenSegmentCondition m ∧ countBlueBeads m = 30 :=
sorry

end NUMINAMATH_CALUDE_minimum_blue_beads_l155_15545


namespace NUMINAMATH_CALUDE_imaginary_power_product_l155_15510

-- Define the imaginary unit i
def i : ℂ := Complex.I

-- State the theorem
theorem imaginary_power_product : (i^15) * (i^135) = -1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_power_product_l155_15510


namespace NUMINAMATH_CALUDE_expression_simplification_l155_15528

theorem expression_simplification (x y : ℝ) :
  3 * y + 4 * y^2 + 2 - (7 - 3 * y - 4 * y^2 + 2 * x) = 8 * y^2 + 6 * y - 2 * x - 5 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l155_15528


namespace NUMINAMATH_CALUDE_consecutive_page_numbers_l155_15581

theorem consecutive_page_numbers (n : ℕ) : 
  n > 0 ∧ n * (n + 1) = 20412 → n + (n + 1) = 283 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_page_numbers_l155_15581


namespace NUMINAMATH_CALUDE_no_three_distinct_real_roots_l155_15534

theorem no_three_distinct_real_roots (c : ℝ) : 
  ¬ ∃ (x₁ x₂ x₃ : ℝ), (x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃) ∧ 
    (x₁^3 + 4*x₁^2 + 6*x₁ + c = 0) ∧ 
    (x₂^3 + 4*x₂^2 + 6*x₂ + c = 0) ∧ 
    (x₃^3 + 4*x₃^2 + 6*x₃ + c = 0) :=
by sorry

end NUMINAMATH_CALUDE_no_three_distinct_real_roots_l155_15534


namespace NUMINAMATH_CALUDE_videocassette_recorder_fraction_l155_15518

theorem videocassette_recorder_fraction 
  (cable_fraction : Real) 
  (cable_and_vcr_fraction : Real) 
  (neither_fraction : Real) :
  cable_fraction = 1/5 →
  cable_and_vcr_fraction = 1/3 * cable_fraction →
  neither_fraction = 0.7666666666666667 →
  ∃ (vcr_fraction : Real),
    vcr_fraction = 1/10 ∧
    vcr_fraction + cable_fraction - cable_and_vcr_fraction + neither_fraction = 1 :=
by sorry

end NUMINAMATH_CALUDE_videocassette_recorder_fraction_l155_15518


namespace NUMINAMATH_CALUDE_union_equals_B_intersection_equals_B_l155_15507

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x^2 + 4*x = 0}
def B (a : ℝ) : Set ℝ := {x : ℝ | x^2 + 2*(a+1)*x + a^2 - 1 = 0}

-- Define the set C
def C : Set ℝ := {a : ℝ | a ≤ -1 ∨ a = 1}

-- Theorem 1
theorem union_equals_B (a : ℝ) : A ∪ B a = B a → a = 1 := by sorry

-- Theorem 2
theorem intersection_equals_B : A ∩ B a = B a → a ∈ C := by sorry

end NUMINAMATH_CALUDE_union_equals_B_intersection_equals_B_l155_15507


namespace NUMINAMATH_CALUDE_evaluate_expression_l155_15580

theorem evaluate_expression : -(16 / 4 * 7 - 50 + 5 * 7) = -13 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l155_15580


namespace NUMINAMATH_CALUDE_teacher_number_game_l155_15584

theorem teacher_number_game (x : ℝ) : 
  x = 5 → 3 * ((2 * (2 * x + 3)) + 2) = 84 := by
  sorry

end NUMINAMATH_CALUDE_teacher_number_game_l155_15584


namespace NUMINAMATH_CALUDE_sum_of_rectangle_areas_l155_15506

/-- The sum of the areas of six rectangles with specified dimensions -/
theorem sum_of_rectangle_areas : 
  let width : ℕ := 2
  let lengths : List ℕ := [1, 4, 9, 16, 25, 36]
  let areas := lengths.map (λ l => width * l)
  areas.sum = 182 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_rectangle_areas_l155_15506


namespace NUMINAMATH_CALUDE_random_opening_page_8_is_random_event_l155_15575

/-- Represents a math book with a specified number of pages. -/
structure MathBook where
  pages : ℕ
  pages_positive : pages > 0

/-- Represents the act of opening a book randomly. -/
def RandomOpening (book : MathBook) := Unit

/-- Represents the result of opening a book to a specific page. -/
structure OpeningResult (book : MathBook) where
  page : ℕ
  page_valid : page > 0 ∧ page ≤ book.pages

/-- Defines what it means for an event to be random. -/
def IsRandomEvent (P : Prop) : Prop :=
  ¬(P ↔ True) ∧ ¬(P ↔ False)

/-- Theorem stating that opening a 200-page math book randomly and landing on page 8 is a random event. -/
theorem random_opening_page_8_is_random_event (book : MathBook) 
  (h_pages : book.pages = 200) :
  IsRandomEvent (∃ (opening : RandomOpening book) (result : OpeningResult book), result.page = 8) :=
sorry

end NUMINAMATH_CALUDE_random_opening_page_8_is_random_event_l155_15575


namespace NUMINAMATH_CALUDE_max_red_socks_l155_15500

def is_valid_sock_distribution (r b y : ℕ) : Prop :=
  let t := r + b + y
  t ≤ 2300 ∧
  (r * (r - 1) * (r - 2) + b * (b - 1) * (b - 2) + y * (y - 1) * (y - 2)) * 3 =
  t * (t - 1) * (t - 2)

theorem max_red_socks :
  ∀ r b y : ℕ, is_valid_sock_distribution r b y → r ≤ 897 :=
by sorry

end NUMINAMATH_CALUDE_max_red_socks_l155_15500


namespace NUMINAMATH_CALUDE_throne_identity_l155_15568

/-- Represents the types of beings in this problem -/
inductive Being
| Knight  -- Always tells the truth
| Liar    -- Always lies
| Human   -- Can either tell the truth or lie
| Monkey  -- An animal

/-- Represents a statement made by a being -/
structure Statement where
  content : Prop
  speaker : Being

/-- The statement made by the being on the throne -/
def throneStatement : Statement :=
  { content := (∃ x : Being, x = Being.Liar ∧ x = Being.Monkey),
    speaker := Being.Human }

/-- Theorem stating that the being on the throne must be a human who is lying -/
theorem throne_identity :
  throneStatement.speaker = Being.Human ∧ 
  ¬throneStatement.content :=
sorry

end NUMINAMATH_CALUDE_throne_identity_l155_15568


namespace NUMINAMATH_CALUDE_f_decreasing_on_interval_l155_15542

-- Define the function
def f (x : ℝ) : ℝ := 2 * x^3 - 6 * x^2 + 11

-- State the theorem
theorem f_decreasing_on_interval :
  ∀ x ∈ Set.Ioo 0 2, StrictMonoOn f (Set.Ioo 0 2) := by
  sorry

end NUMINAMATH_CALUDE_f_decreasing_on_interval_l155_15542


namespace NUMINAMATH_CALUDE_approx_small_number_to_large_place_l155_15525

/-- Given a real number less than 10000, the highest meaningful place value
    for approximation is the hundreds place when attempting to approximate
    to the ten thousand place. -/
theorem approx_small_number_to_large_place (x : ℝ) : 
  x < 10000 → 
  ∃ (approx : ℝ), 
    (approx = 100 * ⌊x / 100⌋) ∧ 
    (∀ (y : ℝ), y = 1000 * ⌊x / 1000⌋ ∨ y = 10000 * ⌊x / 10000⌋ → |x - approx| ≤ |x - y|) :=
by sorry

end NUMINAMATH_CALUDE_approx_small_number_to_large_place_l155_15525


namespace NUMINAMATH_CALUDE_correct_sampling_methods_l155_15591

/-- Represents different sampling methods -/
inductive SamplingMethod
  | SimpleRandom
  | Systematic
  | Stratified

/-- Represents a survey with its characteristics -/
structure Survey where
  total_population : ℕ
  sample_size : ℕ
  has_subgroups : Bool
  subgroup_sizes : List ℕ

/-- Determines the most appropriate sampling method for a given survey -/
def best_sampling_method (s : Survey) : SamplingMethod :=
  if s.has_subgroups then
    SamplingMethod.Stratified
  else if s.total_population % s.sample_size = 0 then
    SamplingMethod.Systematic
  else
    SamplingMethod.SimpleRandom

/-- The three surveys from the problem -/
def survey1 : Survey := ⟨10, 3, false, []⟩
def survey2 : Survey := ⟨3000, 30, false, []⟩
def survey3 : Survey := ⟨160, 40, true, [120, 16, 24]⟩

/-- Theorem stating the correct sampling methods for the given surveys -/
theorem correct_sampling_methods :
  best_sampling_method survey1 = SamplingMethod.SimpleRandom ∧
  best_sampling_method survey2 = SamplingMethod.Systematic ∧
  best_sampling_method survey3 = SamplingMethod.Stratified :=
  sorry

end NUMINAMATH_CALUDE_correct_sampling_methods_l155_15591


namespace NUMINAMATH_CALUDE_parabola_point_distance_l155_15533

/-- Theorem: For a parabola y² = 4x with focus F(1, 0), and a point P(x₀, y₀) on the parabola 
    such that |PF| = 3/2 * x₀, the value of x₀ is 2. -/
theorem parabola_point_distance (x₀ y₀ : ℝ) : 
  y₀^2 = 4*x₀ →                             -- P(x₀, y₀) is on the parabola
  (x₀ - 1)^2 + y₀^2 = (3/2 * x₀)^2 →        -- |PF| = 3/2 * x₀
  x₀ = 2 := by
sorry

end NUMINAMATH_CALUDE_parabola_point_distance_l155_15533


namespace NUMINAMATH_CALUDE_system_a_solution_l155_15592

theorem system_a_solution (x y z t : ℝ) : 
  x - 3*y + 2*z - t = 3 ∧
  2*x + 4*y - 3*z + t = 5 ∧
  4*x - 2*y + z + t = 3 ∧
  3*x + y + z - 2*t = 10 →
  x = 2 ∧ y = -1 ∧ z = -3 ∧ t = -4 := by
sorry


end NUMINAMATH_CALUDE_system_a_solution_l155_15592


namespace NUMINAMATH_CALUDE_hexagon_perimeter_l155_15505

/-- The perimeter of a hexagon with side lengths in arithmetic sequence -/
theorem hexagon_perimeter (a b c d e f : ℕ) (h1 : b = a + 2) (h2 : c = b + 2) (h3 : d = c + 2) 
  (h4 : e = d + 2) (h5 : f = e + 2) (h6 : a = 10) : a + b + c + d + e + f = 90 :=
by sorry

end NUMINAMATH_CALUDE_hexagon_perimeter_l155_15505


namespace NUMINAMATH_CALUDE_seed_germination_percentage_l155_15548

/-- Calculates the percentage of total seeds that germinated given the number of seeds and germination rates for two plots. -/
theorem seed_germination_percentage 
  (seeds_plot1 : ℕ) 
  (seeds_plot2 : ℕ) 
  (germination_rate_plot1 : ℚ) 
  (germination_rate_plot2 : ℚ) 
  (h1 : seeds_plot1 = 300)
  (h2 : seeds_plot2 = 200)
  (h3 : germination_rate_plot1 = 25 / 100)
  (h4 : germination_rate_plot2 = 40 / 100) :
  (((seeds_plot1 : ℚ) * germination_rate_plot1 + (seeds_plot2 : ℚ) * germination_rate_plot2) / 
   ((seeds_plot1 : ℚ) + (seeds_plot2 : ℚ))) * 100 = 31 := by
  sorry

end NUMINAMATH_CALUDE_seed_germination_percentage_l155_15548


namespace NUMINAMATH_CALUDE_smallest_n_with_conditions_n_3584_satisfies_conditions_smallest_n_is_3584_l155_15516

/-- A function to check if a number contains the digit 9 -/
def contains_nine (n : ℕ) : Prop :=
  ∃ k : ℕ, n / (10^k) % 10 = 9

/-- A function to check if a fraction terminates -/
def is_terminating (n : ℕ) : Prop :=
  ∃ a b : ℕ, n = 2^a * 5^b

theorem smallest_n_with_conditions :
  ∀ n : ℕ, n > 0 →
    (is_terminating n ∧ contains_nine n ∧ n % 7 = 0) →
    n ≥ 3584 :=
sorry

theorem n_3584_satisfies_conditions :
  is_terminating 3584 ∧ contains_nine 3584 ∧ 3584 % 7 = 0 :=
sorry

theorem smallest_n_is_3584 :
  ∀ n : ℕ, n > 0 →
    (is_terminating n ∧ contains_nine n ∧ n % 7 = 0) →
    n = 3584 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_with_conditions_n_3584_satisfies_conditions_smallest_n_is_3584_l155_15516


namespace NUMINAMATH_CALUDE_final_sum_after_transformation_l155_15561

theorem final_sum_after_transformation (a b S : ℝ) (h : a + b = S) :
  3 * (a + 5) + 3 * (b + 5) = 3 * S + 30 := by sorry

end NUMINAMATH_CALUDE_final_sum_after_transformation_l155_15561


namespace NUMINAMATH_CALUDE_susan_book_count_l155_15514

theorem susan_book_count (susan_books : ℕ) (lidia_books : ℕ) : 
  lidia_books = 4 * susan_books → 
  susan_books + lidia_books = 3000 → 
  susan_books = 600 := by
sorry

end NUMINAMATH_CALUDE_susan_book_count_l155_15514


namespace NUMINAMATH_CALUDE_maximal_closely_related_interval_l155_15593

/-- Two functions are closely related on an interval if their difference is bounded by 1 -/
def closely_related (f g : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x ∈ Set.Icc a b, |f x - g x| ≤ 1

/-- The given functions f and g -/
def f (x : ℝ) : ℝ := x^2 - 3*x + 4
def g (x : ℝ) : ℝ := 2*x - 3

/-- The theorem stating that [2, 3] is the maximal closely related interval for f and g -/
theorem maximal_closely_related_interval :
  closely_related f g 2 3 ∧
  ∀ a b : ℝ, a < 2 ∨ b > 3 → ¬(closely_related f g a b) :=
sorry

end NUMINAMATH_CALUDE_maximal_closely_related_interval_l155_15593


namespace NUMINAMATH_CALUDE_adjacent_knights_probability_l155_15567

/-- The number of knights at the round table -/
def total_knights : ℕ := 30

/-- The number of knights selected for the quest -/
def selected_knights : ℕ := 4

/-- The probability that at least two of the selected knights were sitting next to each other -/
def adjacent_probability : ℚ := 943 / 1023

/-- Theorem stating the probability of at least two out of four randomly selected knights 
    sitting next to each other in a round table of 30 knights -/
theorem adjacent_knights_probability : 
  let total_ways := Nat.choose total_knights selected_knights
  let non_adjacent_ways := (total_knights - selected_knights) * 
                           (total_knights - selected_knights - 3) * 
                           (total_knights - selected_knights - 6) * 
                           (total_knights - selected_knights - 9)
  (1 : ℚ) - (non_adjacent_ways : ℚ) / total_ways = adjacent_probability := by
  sorry

#eval adjacent_probability.num + adjacent_probability.den

end NUMINAMATH_CALUDE_adjacent_knights_probability_l155_15567


namespace NUMINAMATH_CALUDE_inequality_system_solution_l155_15504

theorem inequality_system_solution (x : ℝ) :
  (-9 * x^2 + 12 * x + 5 > 0) ∧ (3 * x - 1 < 0) ↔ x < -1/3 :=
by sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l155_15504


namespace NUMINAMATH_CALUDE_base_b_difference_divisibility_l155_15572

def base_conversion (b : ℕ) : ℤ := 2 * b^3 - 2 * b^2 + b - 1

theorem base_b_difference_divisibility (b : ℕ) (h : 4 ≤ b ∧ b ≤ 8) :
  ¬(5 ∣ base_conversion b) ↔ b = 6 :=
by sorry

end NUMINAMATH_CALUDE_base_b_difference_divisibility_l155_15572


namespace NUMINAMATH_CALUDE_investment_schemes_count_l155_15509

/-- The number of projects to invest in -/
def num_projects : ℕ := 3

/-- The number of candidate cities -/
def num_cities : ℕ := 4

/-- The maximum number of projects allowed in a single city -/
def max_projects_per_city : ℕ := 2

/-- Calculates the number of investment schemes -/
def num_investment_schemes : ℕ := sorry

/-- Theorem stating that the number of investment schemes is 60 -/
theorem investment_schemes_count :
  num_investment_schemes = 60 := by sorry

end NUMINAMATH_CALUDE_investment_schemes_count_l155_15509


namespace NUMINAMATH_CALUDE_quilt_shaded_fraction_l155_15512

/-- Represents a square quilt block -/
structure QuiltBlock where
  size : Nat
  total_squares : Nat
  shaded_area : Rat

/-- Creates a quilt block with the given specifications -/
def create_quilt_block : QuiltBlock :=
  { size := 4,
    total_squares := 16,
    shaded_area := 2 }

/-- Calculates the fraction of the quilt that is shaded -/
def shaded_fraction (quilt : QuiltBlock) : Rat :=
  quilt.shaded_area / quilt.total_squares

theorem quilt_shaded_fraction :
  let quilt := create_quilt_block
  shaded_fraction quilt = 1 / 8 := by sorry

end NUMINAMATH_CALUDE_quilt_shaded_fraction_l155_15512


namespace NUMINAMATH_CALUDE_w_squared_value_l155_15570

theorem w_squared_value (w : ℝ) (h : (w + 10)^2 = (4*w + 6)*(w + 5)) : w^2 = 70/3 := by
  sorry

end NUMINAMATH_CALUDE_w_squared_value_l155_15570


namespace NUMINAMATH_CALUDE_sheila_attend_probability_l155_15521

-- Define the probabilities
def prob_rain : ℝ := 0.5
def prob_sunny : ℝ := 1 - prob_rain
def prob_attend_if_rain : ℝ := 0.3
def prob_attend_if_sunny : ℝ := 0.9
def prob_remember : ℝ := 0.9

-- Define the theorem
theorem sheila_attend_probability :
  prob_rain * prob_attend_if_rain * prob_remember +
  prob_sunny * prob_attend_if_sunny * prob_remember = 0.54 := by
  sorry

end NUMINAMATH_CALUDE_sheila_attend_probability_l155_15521


namespace NUMINAMATH_CALUDE_trigonometric_expression_evaluation_l155_15553

theorem trigonometric_expression_evaluation (α : Real) (h : Real.tan α = 3) :
  (2 * Real.sin (2 * α) - 3 * Real.cos (2 * α)) / (4 * Real.sin (2 * α) + 5 * Real.cos (2 * α)) = -9/4 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_expression_evaluation_l155_15553


namespace NUMINAMATH_CALUDE_special_matrix_determinant_l155_15520

/-- The determinant of a special n × n matrix A with elements a_{ij} = |i - j| -/
theorem special_matrix_determinant (n : ℕ) (hn : n > 0) :
  let A : Matrix (Fin n) (Fin n) ℤ := λ i j => |i.val - j.val|
  Matrix.det A = (-1 : ℤ)^(n-1) * (n - 1) * 2^(n-2) := by
  sorry

end NUMINAMATH_CALUDE_special_matrix_determinant_l155_15520


namespace NUMINAMATH_CALUDE_ratio_of_cubes_l155_15588

theorem ratio_of_cubes : (81000 : ℝ)^3 / (9000 : ℝ)^3 = 729 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_cubes_l155_15588


namespace NUMINAMATH_CALUDE_car_journey_initial_speed_l155_15577

/-- Represents the speed and position of a car on a journey --/
structure CarJourney where
  initial_speed : ℝ
  total_distance : ℝ
  distance_to_b : ℝ
  distance_to_c : ℝ
  time_remaining_at_b : ℝ
  speed_reduction : ℝ

/-- Theorem stating the conditions of the car journey and the initial speed to be proved --/
theorem car_journey_initial_speed (j : CarJourney) 
  (h1 : j.total_distance = 100)
  (h2 : j.time_remaining_at_b = 0.5)
  (h3 : j.speed_reduction = 10)
  (h4 : j.distance_to_c = 80)
  (h5 : (j.distance_to_b / (j.initial_speed - j.speed_reduction) - 
         (j.distance_to_c - j.distance_to_b) / (j.initial_speed - 2 * j.speed_reduction)) = 1/12)
  : j.initial_speed = 100 := by
  sorry

#check car_journey_initial_speed

end NUMINAMATH_CALUDE_car_journey_initial_speed_l155_15577


namespace NUMINAMATH_CALUDE_percentage_difference_l155_15535

theorem percentage_difference (x y : ℝ) 
  (hx : 3 = 0.15 * x) 
  (hy : 3 = 0.25 * y) : 
  x - y = 8 := by
  sorry

end NUMINAMATH_CALUDE_percentage_difference_l155_15535


namespace NUMINAMATH_CALUDE_sum_of_numbers_l155_15585

theorem sum_of_numbers (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x * y = 16) 
  (h4 : 1 / x = 3 * (1 / y)) : x + y = (16 * Real.sqrt 3) / 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_numbers_l155_15585


namespace NUMINAMATH_CALUDE_kens_climbing_pace_l155_15566

/-- Climbing problem -/
theorem kens_climbing_pace 
  (sari_head_start : ℝ)  -- Time Sari starts before Ken (in hours)
  (sari_initial_lead : ℝ)  -- Sari's lead when Ken starts (in meters)
  (ken_climbing_time : ℝ)  -- Time Ken spends climbing (in hours)
  (final_distance : ℝ)  -- Distance Ken is ahead of Sari at the summit (in meters)
  (h1 : sari_head_start = 2)
  (h2 : sari_initial_lead = 700)
  (h3 : ken_climbing_time = 5)
  (h4 : final_distance = 50) :
  (sari_initial_lead + final_distance) / ken_climbing_time + 
  (sari_initial_lead / sari_head_start) = 500 :=
sorry

end NUMINAMATH_CALUDE_kens_climbing_pace_l155_15566


namespace NUMINAMATH_CALUDE_visitor_increase_l155_15582

theorem visitor_increase (original_fee : ℝ) (fee_reduction : ℝ) (sale_increase : ℝ) :
  original_fee = 1 →
  fee_reduction = 0.25 →
  sale_increase = 0.20 →
  let new_fee := original_fee * (1 - fee_reduction)
  let visitor_increase := (1 + sale_increase) / (1 - fee_reduction) - 1
  visitor_increase = 0.60 := by sorry

end NUMINAMATH_CALUDE_visitor_increase_l155_15582


namespace NUMINAMATH_CALUDE_simplify_fraction_l155_15547

theorem simplify_fraction : (120 : ℚ) / 1320 = 1 / 11 := by sorry

end NUMINAMATH_CALUDE_simplify_fraction_l155_15547


namespace NUMINAMATH_CALUDE_okeydokey_earthworms_calculation_l155_15544

/-- The number of apples Okeydokey invested -/
def okeydokey_apples : ℕ := 5

/-- The number of apples Artichokey invested -/
def artichokey_apples : ℕ := 7

/-- The total number of earthworms in the box -/
def total_earthworms : ℕ := 60

/-- The number of earthworms Okeydokey should receive -/
def okeydokey_earthworms : ℕ := 25

/-- Theorem stating that Okeydokey should receive 25 earthworms -/
theorem okeydokey_earthworms_calculation :
  (okeydokey_apples : ℚ) / (okeydokey_apples + artichokey_apples : ℚ) * total_earthworms = okeydokey_earthworms := by
  sorry

end NUMINAMATH_CALUDE_okeydokey_earthworms_calculation_l155_15544


namespace NUMINAMATH_CALUDE_inverse_f_486_l155_15551

def f (x : ℝ) : ℝ := sorry

theorem inverse_f_486 (h1 : f 5 = 2) (h2 : ∀ x, f (3 * x) = 3 * f x) :
  f 1215 = 486 := by sorry

end NUMINAMATH_CALUDE_inverse_f_486_l155_15551


namespace NUMINAMATH_CALUDE_max_quadratic_equations_l155_15502

def is_valid_number (n : ℕ) : Prop :=
  300 ≤ n ∧ n ≤ 999 ∧ (n / 100) % 2 = 1 ∧ (n / 100) > 1

def has_real_roots (a b c : ℕ) : Prop :=
  b * b ≥ 4 * a * c

def valid_equation (a b c : ℕ) : Prop :=
  is_valid_number a ∧ is_valid_number b ∧ is_valid_number c ∧
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  has_real_roots a b c

theorem max_quadratic_equations :
  ∃ (equations : Finset (ℕ × ℕ × ℕ)),
    (∀ (e : ℕ × ℕ × ℕ), e ∈ equations → valid_equation e.1 e.2.1 e.2.2) ∧
    equations.card = 100 ∧
    (∀ (equations' : Finset (ℕ × ℕ × ℕ)),
      (∀ (e : ℕ × ℕ × ℕ), e ∈ equations' → valid_equation e.1 e.2.1 e.2.2) →
      equations'.card ≤ 100) :=
sorry

end NUMINAMATH_CALUDE_max_quadratic_equations_l155_15502


namespace NUMINAMATH_CALUDE_virtual_set_divisors_l155_15508

def isVirtual (A : Finset ℕ) : Prop :=
  A.card = 5 ∧
  (∀ (a b c : ℕ), a ∈ A → b ∈ A → c ∈ A → a ≠ b → b ≠ c → a ≠ c → Nat.gcd a (Nat.gcd b c) > 1) ∧
  (∀ (a b c d : ℕ), a ∈ A → b ∈ A → c ∈ A → d ∈ A → a ≠ b → b ≠ c → c ≠ d → a ≠ c → a ≠ d → b ≠ d → Nat.gcd a (Nat.gcd b (Nat.gcd c d)) = 1)

theorem virtual_set_divisors (A : Finset ℕ) (h : isVirtual A) :
  (Finset.prod A id).divisors.card ≥ 2020 := by
  sorry

end NUMINAMATH_CALUDE_virtual_set_divisors_l155_15508


namespace NUMINAMATH_CALUDE_min_c_for_unique_solution_l155_15587

/-- Given positive integers a, b, c with a < b < c, the minimum value of c for which the system
    of equations 2x + y = 2022 and y = |x-a| + |x-b| + |x-c| has exactly one solution is 1012. -/
theorem min_c_for_unique_solution (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hab : a < b) (hbc : b < c) :
  (∃! x y : ℝ, 2 * x + y = 2022 ∧ y = |x - a| + |x - b| + |x - c|) →
  c ≥ 1012 ∧ 
  (c = 1012 → ∃! x y : ℝ, 2 * x + y = 2022 ∧ y = |x - a| + |x - b| + |x - c|) :=
by sorry

end NUMINAMATH_CALUDE_min_c_for_unique_solution_l155_15587


namespace NUMINAMATH_CALUDE_largest_certain_divisor_l155_15519

/-- The set of numbers on the eight-sided die -/
def dieNumbers : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8}

/-- The product of seven numbers from the die -/
def Q (s : Finset ℕ) : ℕ :=
  if s.card = 7 ∧ s ⊆ dieNumbers then s.prod id else 0

/-- The theorem stating that 48 is the largest number certain to divide Q -/
theorem largest_certain_divisor :
  ∀ n : ℕ, (∀ s : Finset ℕ, s.card = 7 ∧ s ⊆ dieNumbers → n ∣ Q s) → n ≤ 48 :=
sorry

end NUMINAMATH_CALUDE_largest_certain_divisor_l155_15519


namespace NUMINAMATH_CALUDE_probability_two_defective_shipment_l155_15503

/-- The probability of selecting two defective smartphones at random from a shipment -/
def probability_two_defective (total : ℕ) (defective : ℕ) : ℝ :=
  let p1 := defective / total
  let p2 := (defective - 1) / (total - 1)
  p1 * p2

/-- Theorem stating the probability of selecting two defective smartphones -/
theorem probability_two_defective_shipment :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.0000001 ∧ 
  |probability_two_defective 250 76 - 0.0915632| < ε :=
sorry

end NUMINAMATH_CALUDE_probability_two_defective_shipment_l155_15503


namespace NUMINAMATH_CALUDE_max_value_expression_max_value_achievable_l155_15590

theorem max_value_expression (x y : ℝ) :
  (3 * x + 4 * y + 5) / Real.sqrt (5 * x^2 + 3 * y^2 + 4) ≤ 5 * Real.sqrt 2 :=
sorry

theorem max_value_achievable :
  ∃ (x y : ℝ), (3 * x + 4 * y + 5) / Real.sqrt (5 * x^2 + 3 * y^2 + 4) = 5 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_max_value_expression_max_value_achievable_l155_15590


namespace NUMINAMATH_CALUDE_fraction_difference_times_two_l155_15546

theorem fraction_difference_times_two :
  let a := 4 + 6 + 8 + 10
  let b := 3 + 5 + 7 + 9
  (a / b - b / a) * 2 = 13 / 21 := by sorry

end NUMINAMATH_CALUDE_fraction_difference_times_two_l155_15546


namespace NUMINAMATH_CALUDE_bird_count_l155_15557

theorem bird_count (crows : ℕ) (hawk_percentage : ℚ) : 
  crows = 30 → 
  hawk_percentage = 60 / 100 → 
  crows + (crows + hawk_percentage * crows) = 78 := by
  sorry

end NUMINAMATH_CALUDE_bird_count_l155_15557


namespace NUMINAMATH_CALUDE_triangle_existence_implies_m_greater_than_six_l155_15559

/-- The function f(x) = x^3 - 3x + m -/
def f (m : ℝ) (x : ℝ) : ℝ := x^3 - 3*x + m

/-- Theorem: If there exists a triangle with side lengths f(a), f(b), f(c) for a, b, c in [0, 2], then m > 6 -/
theorem triangle_existence_implies_m_greater_than_six (m : ℝ) : 
  (∃ a b c : ℝ, 0 ≤ a ∧ a ≤ 2 ∧ 0 ≤ b ∧ b ≤ 2 ∧ 0 ≤ c ∧ c ≤ 2 ∧ 
    f m a + f m b > f m c ∧ 
    f m b + f m c > f m a ∧ 
    f m c + f m a > f m b) → 
  m > 6 := by
  sorry


end NUMINAMATH_CALUDE_triangle_existence_implies_m_greater_than_six_l155_15559


namespace NUMINAMATH_CALUDE_polynomial_simplification_l155_15501

theorem polynomial_simplification (r : ℝ) :
  (2 * r^3 + 5 * r^2 - 4 * r + 8) - (r^3 + 9 * r^2 - 2 * r - 3) =
  r^3 - 4 * r^2 - 2 * r + 11 := by
sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l155_15501


namespace NUMINAMATH_CALUDE_percentage_difference_l155_15540

theorem percentage_difference (C : ℝ) (A B : ℝ) 
  (hA : A = 0.75 * C) 
  (hB : B = 0.63 * C) : 
  (A - B) / A = 0.16 := by
  sorry

end NUMINAMATH_CALUDE_percentage_difference_l155_15540


namespace NUMINAMATH_CALUDE_min_intersection_points_l155_15522

/-- Represents a configuration of circles on a plane -/
structure CircleConfiguration where
  num_circles : ℕ
  num_intersections : ℕ
  intersections_per_circle : ℕ → ℕ

/-- The minimum number of intersections for a valid configuration -/
def min_intersections (config : CircleConfiguration) : ℕ :=
  (config.num_circles * config.intersections_per_circle 0) / 2

/-- Predicate for a valid circle configuration -/
def valid_configuration (config : CircleConfiguration) : Prop :=
  config.num_circles = 2008 ∧
  (∀ i, config.intersections_per_circle i ≥ 3) ∧
  config.num_intersections ≥ min_intersections config

theorem min_intersection_points (config : CircleConfiguration) 
  (h : valid_configuration config) : 
  config.num_intersections ≥ 3012 :=
sorry

end NUMINAMATH_CALUDE_min_intersection_points_l155_15522


namespace NUMINAMATH_CALUDE_even_function_implies_a_value_l155_15574

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := (x + 1) * (2 * x + 3 * a)

-- State the theorem
theorem even_function_implies_a_value :
  (∀ x : ℝ, f a x = f a (-x)) → a = -2/3 := by
  sorry

end NUMINAMATH_CALUDE_even_function_implies_a_value_l155_15574


namespace NUMINAMATH_CALUDE_divisible_by_six_l155_15599

theorem divisible_by_six (n : ℕ) : 
  6 ∣ (n + 20) * (n + 201) * (n + 2020) := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_six_l155_15599


namespace NUMINAMATH_CALUDE_min_value_absolute_sum_l155_15541

theorem min_value_absolute_sum (x : ℝ) :
  |x + 1| + 2 * |x - 5| + |2 * x - 7| + |(x - 11) / 2| ≥ 45 / 8 ∧
  ∃ x : ℝ, |x + 1| + 2 * |x - 5| + |2 * x - 7| + |(x - 11) / 2| = 45 / 8 :=
by sorry

end NUMINAMATH_CALUDE_min_value_absolute_sum_l155_15541


namespace NUMINAMATH_CALUDE_problem_statement_l155_15586

-- Define the quadratic equation
def quadratic (x : ℝ) : Prop := x^2 - 4*x - 5 = 0

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 1

-- Theorem statement
theorem problem_statement :
  -- Part 1: x = 5 is sufficient but not necessary for quadratic
  (∃ x ≠ 5, quadratic x) ∧ (quadratic 5) ∧
  -- Part 2: (∃x, tan x = 1) ∧ (¬(∀x, x^2 - x + 1 > 0)) is false
  ¬((∃ x : ℝ, Real.tan x = 1) ∧ ¬(∀ x : ℝ, x^2 - x + 1 > 0)) ∧
  -- Part 3: Tangent line equation at (2, f(2)) is y = -3
  (∃ m : ℝ, f 2 = -3 ∧ (deriv f) 2 = m ∧ m = 0) :=
sorry

end NUMINAMATH_CALUDE_problem_statement_l155_15586


namespace NUMINAMATH_CALUDE_cos_2A_plus_cos_2B_l155_15517

theorem cos_2A_plus_cos_2B (A B : Real) 
  (h1 : Real.sin A + Real.sin B = 1)
  (h2 : Real.cos A + Real.cos B = 0) :
  12 * Real.cos (2 * A) + 4 * Real.cos (2 * B) = 8 := by
  sorry

end NUMINAMATH_CALUDE_cos_2A_plus_cos_2B_l155_15517


namespace NUMINAMATH_CALUDE_probability_two_present_one_absent_l155_15556

/-- The probability of a student being absent on a given day -/
def p_absent : ℚ := 2 / 50

/-- The probability of a student being present on a given day -/
def p_present : ℚ := 1 - p_absent

/-- The number of students we are considering -/
def n_students : ℕ := 3

/-- The number of students that should be present -/
def n_present : ℕ := 2

theorem probability_two_present_one_absent :
  (n_students.choose n_present : ℚ) * p_present ^ n_present * p_absent ^ (n_students - n_present) = 1728 / 15625 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_present_one_absent_l155_15556


namespace NUMINAMATH_CALUDE_cans_needed_proof_l155_15529

/-- The number of cans Martha collected -/
def martha_cans : ℕ := 90

/-- The number of cans Diego collected -/
def diego_cans : ℕ := martha_cans / 2 + 10

/-- The total number of cans needed for the project -/
def project_goal : ℕ := 150

/-- The number of additional cans needed -/
def additional_cans : ℕ := project_goal - (martha_cans + diego_cans)

theorem cans_needed_proof : additional_cans = 5 := by sorry

end NUMINAMATH_CALUDE_cans_needed_proof_l155_15529


namespace NUMINAMATH_CALUDE_inverse_functions_values_l155_15595

-- Define the inverse function relationship
def are_inverse_functions (f g : ℝ → ℝ) : Prop :=
  ∀ x, f (g x) = x ∧ g (f x) = x

-- Define the two linear functions
def f (a : ℝ) (x : ℝ) : ℝ := a * x + 2
def g (b : ℝ) (x : ℝ) : ℝ := 3 * x + b

-- State the theorem
theorem inverse_functions_values :
  ∀ a b : ℝ, are_inverse_functions (f a) (g b) → a = 1/3 ∧ b = -6 :=
by sorry

end NUMINAMATH_CALUDE_inverse_functions_values_l155_15595


namespace NUMINAMATH_CALUDE_matrix_power_sum_l155_15531

/-- Given a 3x3 matrix C and a natural number m, 
    if C^m equals a specific matrix and C has a specific form,
    then b + m = 310 where b is an element of C. -/
theorem matrix_power_sum (b m : ℕ) (C : Matrix (Fin 3) (Fin 3) ℕ) : 
  C^m = !![1, 33, 3080; 1, 1, 65; 1, 0, 1] ∧ 
  C = !![1, 3, b; 0, 1, 5; 1, 0, 1] → 
  b + m = 310 := by sorry

end NUMINAMATH_CALUDE_matrix_power_sum_l155_15531


namespace NUMINAMATH_CALUDE_simplify_fraction_l155_15550

theorem simplify_fraction : (75 : ℚ) / 100 = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l155_15550


namespace NUMINAMATH_CALUDE_backpack_price_increase_l155_15573

theorem backpack_price_increase 
  (original_backpack_price : ℕ)
  (original_binder_price : ℕ)
  (num_binders : ℕ)
  (binder_price_reduction : ℕ)
  (total_spent : ℕ)
  (h1 : original_backpack_price = 50)
  (h2 : original_binder_price = 20)
  (h3 : num_binders = 3)
  (h4 : binder_price_reduction = 2)
  (h5 : total_spent = 109)
  : ∃ (price_increase : ℕ), 
    original_backpack_price + price_increase + 
    num_binders * (original_binder_price - binder_price_reduction) = total_spent ∧
    price_increase = 5 := by
  sorry

end NUMINAMATH_CALUDE_backpack_price_increase_l155_15573


namespace NUMINAMATH_CALUDE_nancy_seeds_l155_15523

/-- Calculates the total number of seeds Nancy started with. -/
def total_seeds (big_garden_seeds : ℕ) (small_gardens : ℕ) (seeds_per_small_garden : ℕ) : ℕ :=
  big_garden_seeds + small_gardens * seeds_per_small_garden

/-- Proves that Nancy started with 52 seeds given the problem conditions. -/
theorem nancy_seeds :
  let big_garden_seeds : ℕ := 28
  let small_gardens : ℕ := 6
  let seeds_per_small_garden : ℕ := 4
  total_seeds big_garden_seeds small_gardens seeds_per_small_garden = 52 := by
  sorry

#eval total_seeds 28 6 4

end NUMINAMATH_CALUDE_nancy_seeds_l155_15523


namespace NUMINAMATH_CALUDE_birth_interval_l155_15552

/-- Given 5 children born at equal intervals, with the youngest being 6 years old
    and the sum of all ages being 60 years, the interval between births is 3.6 years. -/
theorem birth_interval (n : ℕ) (youngest_age sum_ages : ℝ) (h1 : n = 5) (h2 : youngest_age = 6)
    (h3 : sum_ages = 60) : ∃ interval : ℝ,
  interval = 3.6 ∧
  sum_ages = n * youngest_age + (interval * (n * (n - 1)) / 2) := by
  sorry

end NUMINAMATH_CALUDE_birth_interval_l155_15552


namespace NUMINAMATH_CALUDE_inequality_solution_and_parameters_l155_15530

theorem inequality_solution_and_parameters :
  ∀ (a b : ℝ),
  (∀ x : ℝ, x^2 - 3*a*x + b > 0 ↔ x < 1 ∨ x > 2) →
  (a = 1 ∧ b = 2) ∧
  (∀ m : ℝ, 
    (m < 2 → ∀ x : ℝ, (x - 2)*(x - m) < 0 ↔ m < x ∧ x < 2) ∧
    (m = 2 → ∀ x : ℝ, ¬((x - 2)*(x - m) < 0)) ∧
    (m > 2 → ∀ x : ℝ, (x - 2)*(x - m) < 0 ↔ 2 < x ∧ x < m)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_and_parameters_l155_15530


namespace NUMINAMATH_CALUDE_no_solution_system_l155_15589

/-- The system of linear equations has no solution -/
theorem no_solution_system (x₁ x₂ x₃ : ℝ) : 
  ¬ (x₁ + 4 * x₂ + 10 * x₃ = 1 ∧ 
     -5 * x₂ - 13 * x₃ = -1.25 ∧ 
     0 = 1.25) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_system_l155_15589


namespace NUMINAMATH_CALUDE_concentric_circle_through_point_l155_15532

/-- Given a circle with equation x^2 + y^2 - 4x + 6y + 3 = 0,
    prove that (x - 2)^2 + (y + 3)^2 = 25 represents a circle
    that is concentric with the given circle and passes through (-1, 1) -/
theorem concentric_circle_through_point
  (h : ∀ x y : ℝ, x^2 + y^2 - 4*x + 6*y + 3 = 0 → (x - 2)^2 + (y + 3)^2 = 10) :
  (∀ x y : ℝ, (x - 2)^2 + (y + 3)^2 = 25 →
    ∃ k : ℝ, k > 0 ∧ (x - 2)^2 + (y + 3)^2 = k * ((x - 2)^2 + (y + 3)^2 - 10)) ∧
  ((-1 - 2)^2 + (1 + 3)^2 = 25) :=
by sorry

end NUMINAMATH_CALUDE_concentric_circle_through_point_l155_15532


namespace NUMINAMATH_CALUDE_chicken_rabbit_problem_l155_15583

/-- The number of chickens and rabbits in the cage satisfying the given conditions -/
theorem chicken_rabbit_problem :
  ∃ (chickens rabbits : ℕ),
    chickens + rabbits = 35 ∧
    2 * chickens + 4 * rabbits = 94 ∧
    chickens = 23 ∧
    rabbits = 12 := by
  sorry

end NUMINAMATH_CALUDE_chicken_rabbit_problem_l155_15583


namespace NUMINAMATH_CALUDE_danny_soda_remaining_l155_15554

theorem danny_soda_remaining (bottles : ℕ) (consumed : ℚ) (given_away : ℚ) : 
  bottles = 3 → 
  consumed = 9/10 → 
  given_away = 7/10 → 
  (1 - consumed) + 2 * (1 - given_away) = 7/10 :=
by sorry

end NUMINAMATH_CALUDE_danny_soda_remaining_l155_15554


namespace NUMINAMATH_CALUDE_noodles_given_to_william_l155_15571

/-- Given that Daniel initially had 54.0 noodles and was left with 42 noodles after giving some to William,
    prove that the number of noodles Daniel gave to William is 12. -/
theorem noodles_given_to_william (initial_noodles : ℝ) (remaining_noodles : ℝ) 
    (h1 : initial_noodles = 54.0) 
    (h2 : remaining_noodles = 42) : 
  initial_noodles - remaining_noodles = 12 := by
  sorry

end NUMINAMATH_CALUDE_noodles_given_to_william_l155_15571


namespace NUMINAMATH_CALUDE_function_difference_implies_m_value_l155_15539

theorem function_difference_implies_m_value :
  ∀ (m : ℝ),
  let f : ℝ → ℝ := λ x ↦ 4 * x^2 - 3 * x + 5
  let g : ℝ → ℝ := λ x ↦ x^2 - m * x - 8
  (f 5 - g 5 = 15) → (m = -15) :=
by
  sorry

end NUMINAMATH_CALUDE_function_difference_implies_m_value_l155_15539


namespace NUMINAMATH_CALUDE_locus_of_tangent_points_theorem_l155_15558

/-- The locus of points for which an ellipse or hyperbola with center at the origin is tangent -/
def locus_of_tangent_points (x y a b c : ℝ) : Prop :=
  (a^2 * y^2 + b^2 * x^2 = x^2 * y^2 ∧ b^2 = a^2 - c^2) ∨
  (a^2 * y^2 - b^2 * x^2 = x^2 * y^2 ∧ b^2 = c^2 - a^2)

/-- Theorem stating the locus of points for ellipses and hyperbolas with center at the origin -/
theorem locus_of_tangent_points_theorem (x y a b c : ℝ) :
  locus_of_tangent_points x y a b c :=
sorry

end NUMINAMATH_CALUDE_locus_of_tangent_points_theorem_l155_15558


namespace NUMINAMATH_CALUDE_sequence_comparison_theorem_l155_15526

theorem sequence_comparison_theorem (a b : ℕ → ℕ) :
  ∃ r s : ℕ, r ≠ s ∧ a r ≥ a s ∧ b r ≥ b s := by
  sorry

end NUMINAMATH_CALUDE_sequence_comparison_theorem_l155_15526


namespace NUMINAMATH_CALUDE_concert_duration_l155_15511

theorem concert_duration (hours : ℕ) (minutes : ℕ) : 
  hours = 7 ∧ minutes = 45 → hours * 60 + minutes = 465 := by
  sorry

end NUMINAMATH_CALUDE_concert_duration_l155_15511


namespace NUMINAMATH_CALUDE_problem_solution_l155_15536

theorem problem_solution (x y : ℝ) (h1 : x > 0) (h2 : y > 0) 
  (h3 : 8 * x^2 + 16 * x * y = x^3 + 3 * x^2 * y) (h4 : y = 2 * x) : 
  x = 40 / 7 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l155_15536


namespace NUMINAMATH_CALUDE_largest_valid_number_l155_15569

def is_valid_number (n : ℕ) : Prop :=
  n ≥ 1000 ∧ n ≤ 9999 ∧
  (n % 100 = (n / 100 % 10 + n / 1000) % 10 * 10 + (n / 10 % 10 + n / 100 % 10) % 10)

theorem largest_valid_number : 
  is_valid_number 9099 ∧ ∀ m : ℕ, is_valid_number m → m ≤ 9099 :=
sorry

end NUMINAMATH_CALUDE_largest_valid_number_l155_15569


namespace NUMINAMATH_CALUDE_circle_equation_k_l155_15537

/-- The equation of a circle with center (h, k) and radius r is (x - h)^2 + (y - k)^2 = r^2 -/
def is_circle (h k r : ℝ) (f : ℝ → ℝ → ℝ) : Prop :=
  ∀ x y, f x y = 0 ↔ (x - h)^2 + (y - k)^2 = r^2

/-- The given equation -/
def equation (k : ℝ) (x y : ℝ) : ℝ := x^2 + 8*x + y^2 + 4*y - k

theorem circle_equation_k (k : ℝ) : 
  (∃ h v, is_circle h v 5 (equation k)) ↔ k = 5 := by sorry

end NUMINAMATH_CALUDE_circle_equation_k_l155_15537


namespace NUMINAMATH_CALUDE_barneys_restock_order_l155_15594

/-- Represents the number of items in Barney's grocery store --/
structure GroceryStore where
  sold : Nat        -- Number of items sold that day
  left : Nat        -- Number of items left in the store
  storeroom : Nat   -- Number of items in the storeroom

/-- Calculates the total number of items ordered to restock the shelves --/
def items_ordered (store : GroceryStore) : Nat :=
  store.sold + store.left + store.storeroom

/-- Theorem stating that for Barney's grocery store, the number of items
    ordered to restock the shelves is 5608 --/
theorem barneys_restock_order :
  let store : GroceryStore := {
    sold := 1561,
    left := 3472,
    storeroom := 575
  }
  items_ordered store = 5608 := by
  sorry

end NUMINAMATH_CALUDE_barneys_restock_order_l155_15594


namespace NUMINAMATH_CALUDE_inequality_proof_l155_15555

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_eq : a^2 + b^2 + 4*c^2 = 3) :
  (a + b + 2*c ≤ 3) ∧ (b = 2*c → 1/a + 1/c ≥ 3) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l155_15555


namespace NUMINAMATH_CALUDE_compound_propositions_truth_l155_15543

-- Define propositions p and q
def p : Prop := ∀ x y : ℝ, x > y → -x < -y
def q : Prop := ∀ x y : ℝ, x < y → x^2 < y^2

-- Theorem statement
theorem compound_propositions_truth :
  (p ∧ q = False) ∧
  (p ∨ q = True) ∧
  (p ∧ (¬q) = True) ∧
  ((¬p) ∨ q = False) :=
by sorry

end NUMINAMATH_CALUDE_compound_propositions_truth_l155_15543


namespace NUMINAMATH_CALUDE_house_rent_expenditure_l155_15564

-- Define the given parameters
def total_income : ℝ := 1000
def petrol_percentage : ℝ := 0.30
def house_rent_percentage : ℝ := 0.10
def petrol_expenditure : ℝ := 300

-- Define the theorem
theorem house_rent_expenditure :
  let remaining_income := total_income - petrol_expenditure
  remaining_income * house_rent_percentage = 70 := by
  sorry

end NUMINAMATH_CALUDE_house_rent_expenditure_l155_15564


namespace NUMINAMATH_CALUDE_cube_volume_problem_l155_15527

theorem cube_volume_problem (a : ℝ) : 
  (a + 2) * (a + 2) * (a - 2) = a^3 - 16 → a^3 = 9 + 12 * Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_cube_volume_problem_l155_15527
