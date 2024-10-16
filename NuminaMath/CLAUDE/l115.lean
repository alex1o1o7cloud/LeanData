import Mathlib

namespace NUMINAMATH_CALUDE_eight_digit_increasing_integers_mod_1000_l115_11511

theorem eight_digit_increasing_integers_mod_1000 : 
  (Nat.choose 17 8) % 1000 = 310 := by sorry

end NUMINAMATH_CALUDE_eight_digit_increasing_integers_mod_1000_l115_11511


namespace NUMINAMATH_CALUDE_box_area_ratio_l115_11566

/-- A rectangular box with specific properties -/
structure Box where
  length : ℝ
  width : ℝ
  height : ℝ
  volume_eq : length * width * height = 3000
  side_area_eq : length * height = 200
  front_top_relation : width * height = (1/2) * (length * width)

/-- The ratio of the top face area to the side face area is 3:2 -/
theorem box_area_ratio (b : Box) : (b.length * b.width) / (b.length * b.height) = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_box_area_ratio_l115_11566


namespace NUMINAMATH_CALUDE_total_collection_theorem_l115_11599

def group_size : ℕ := 77

def member_contribution (n : ℕ) : ℕ := n

def total_collection_paise (n : ℕ) : ℕ := n * member_contribution n

def paise_to_rupees (p : ℕ) : ℚ := p / 100

theorem total_collection_theorem :
  paise_to_rupees (total_collection_paise group_size) = 59.29 := by
  sorry

end NUMINAMATH_CALUDE_total_collection_theorem_l115_11599


namespace NUMINAMATH_CALUDE_truncated_cone_rope_theorem_l115_11530

/-- Represents a truncated cone with given dimensions -/
structure TruncatedCone where
  r₁ : ℝ  -- Upper base radius
  r₂ : ℝ  -- Lower base radius
  h : ℝ   -- Slant height

/-- Calculates the minimum length of the rope for a given truncated cone -/
def min_rope_length (cone : TruncatedCone) : ℝ := sorry

/-- Calculates the minimum distance from the rope to the upper base circumference -/
def min_distance_to_upper_base (cone : TruncatedCone) : ℝ := sorry

theorem truncated_cone_rope_theorem (cone : TruncatedCone) 
  (h₁ : cone.r₁ = 5)
  (h₂ : cone.r₂ = 10)
  (h₃ : cone.h = 20) :
  (min_rope_length cone = 50) ∧ 
  (min_distance_to_upper_base cone = 4) := by sorry

end NUMINAMATH_CALUDE_truncated_cone_rope_theorem_l115_11530


namespace NUMINAMATH_CALUDE_square_root_of_negative_two_fourth_power_l115_11596

theorem square_root_of_negative_two_fourth_power :
  Real.sqrt ((-2)^4) = 4 ∨ Real.sqrt ((-2)^4) = -4 := by
  sorry

end NUMINAMATH_CALUDE_square_root_of_negative_two_fourth_power_l115_11596


namespace NUMINAMATH_CALUDE_not_always_possible_to_reduce_box_dimension_counterexample_exists_l115_11554

/-- Represents a rectangular parallelepiped with dimensions length, width, and height -/
structure Parallelepiped where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Represents a box containing parallelepipeds -/
structure Box where
  length : ℝ
  width : ℝ
  height : ℝ
  contents : List Parallelepiped

/-- Predicate to check if a parallelepiped fits in a box -/
def fits_in_box (p : Parallelepiped) (b : Box) : Prop :=
  p.length ≤ b.length ∧ p.width ≤ b.width ∧ p.height ≤ b.height

/-- Predicate to check if all parallelepipeds in a list fit in a box -/
def all_fit_in_box (ps : List Parallelepiped) (b : Box) : Prop :=
  ∀ p ∈ ps, fits_in_box p b

/-- Function to reduce one dimension of each parallelepiped -/
def reduce_parallelepipeds (ps : List Parallelepiped) : List Parallelepiped :=
  ps.map fun p => 
    let reduced_length := p.length * 0.99
    let reduced_width := p.width * 0.99
    let reduced_height := p.height * 0.99
    ⟨reduced_length, reduced_width, reduced_height⟩

/-- Theorem stating that it's not always possible to reduce a box dimension -/
theorem not_always_possible_to_reduce_box_dimension 
  (original_box : Box) 
  (reduced_parallelepipeds : List Parallelepiped) : Prop :=
  ∃ (reduced_box : Box), 
    (reduced_box.length < original_box.length ∨ 
     reduced_box.width < original_box.width ∨ 
     reduced_box.height < original_box.height) ∧
    all_fit_in_box reduced_parallelepipeds reduced_box →
    False

/-- Main theorem -/
theorem counterexample_exists : ∃ (original_box : Box) (original_parallelepipeds : List Parallelepiped),
  all_fit_in_box original_parallelepipeds original_box ∧
  not_always_possible_to_reduce_box_dimension original_box (reduce_parallelepipeds original_parallelepipeds) := by
  sorry

end NUMINAMATH_CALUDE_not_always_possible_to_reduce_box_dimension_counterexample_exists_l115_11554


namespace NUMINAMATH_CALUDE_total_fruits_in_bowl_l115_11502

/-- The total number of fruits in a bowl, given the number of bananas, 
    apples (twice the number of bananas), and oranges. -/
theorem total_fruits_in_bowl (bananas : ℕ) (oranges : ℕ) : 
  bananas = 2 → oranges = 6 → bananas + 2 * bananas + oranges = 12 := by
  sorry

end NUMINAMATH_CALUDE_total_fruits_in_bowl_l115_11502


namespace NUMINAMATH_CALUDE_max_value_operation_l115_11541

theorem max_value_operation : 
  ∃ (max : ℕ), max = 600 ∧ 
  (∀ n : ℕ, 100 ≤ n ∧ n ≤ 999 → 3 * (300 - n) ≤ max) ∧
  (∃ n : ℕ, 100 ≤ n ∧ n ≤ 999 ∧ 3 * (300 - n) = max) :=
by sorry

end NUMINAMATH_CALUDE_max_value_operation_l115_11541


namespace NUMINAMATH_CALUDE_tutors_next_meeting_l115_11544

theorem tutors_next_meeting (elise_schedule fiona_schedule george_schedule harry_schedule : ℕ) 
  (h_elise : elise_schedule = 5)
  (h_fiona : fiona_schedule = 6)
  (h_george : george_schedule = 8)
  (h_harry : harry_schedule = 9) :
  Nat.lcm elise_schedule (Nat.lcm fiona_schedule (Nat.lcm george_schedule harry_schedule)) = 360 := by
  sorry

end NUMINAMATH_CALUDE_tutors_next_meeting_l115_11544


namespace NUMINAMATH_CALUDE_square_area_from_vertices_l115_11561

/-- The area of a square with adjacent vertices at (1,3) and (-4,6) is 34 -/
theorem square_area_from_vertices : 
  let p1 : ℝ × ℝ := (1, 3)
  let p2 : ℝ × ℝ := (-4, 6)
  let side_length := Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)
  side_length^2 = 34 := by sorry

end NUMINAMATH_CALUDE_square_area_from_vertices_l115_11561


namespace NUMINAMATH_CALUDE_unique_number_2008_l115_11535

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem unique_number_2008 :
  ∃! n : ℕ, n > 0 ∧ n * sum_of_digits n = 2008 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_unique_number_2008_l115_11535


namespace NUMINAMATH_CALUDE_solution_set_f_derivative_positive_l115_11592

noncomputable def f (x : ℝ) : ℝ := x^2 * Real.exp x

theorem solution_set_f_derivative_positive :
  {x : ℝ | (deriv f) x > 0} = {x : ℝ | x < -2 ∨ x > 0} := by sorry

end NUMINAMATH_CALUDE_solution_set_f_derivative_positive_l115_11592


namespace NUMINAMATH_CALUDE_quadratic_function_property_l115_11503

/-- A quadratic function with specific properties -/
def QuadraticFunction (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

theorem quadratic_function_property (a b c : ℝ) :
  QuadraticFunction a b c 0 = -1 →
  QuadraticFunction a b c 4 = QuadraticFunction a b c 5 →
  ∃ (n : ℤ), QuadraticFunction a b c 11 = n →
  QuadraticFunction a b c 11 = -1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_property_l115_11503


namespace NUMINAMATH_CALUDE_circle_radius_l115_11587

theorem circle_radius (x y : ℝ) (h : x + y = 100 * Real.pi) : 
  (∃ r : ℝ, x = Real.pi * r^2 ∧ y = 2 * Real.pi * r) → 
  (∃ r : ℝ, x = Real.pi * r^2 ∧ y = 2 * Real.pi * r ∧ r = 10) :=
by sorry

end NUMINAMATH_CALUDE_circle_radius_l115_11587


namespace NUMINAMATH_CALUDE_units_digit_of_7_to_1023_l115_11575

theorem units_digit_of_7_to_1023 : ∃ n : ℕ, 7^1023 ≡ 3 [ZMOD 10] :=
  sorry

end NUMINAMATH_CALUDE_units_digit_of_7_to_1023_l115_11575


namespace NUMINAMATH_CALUDE_sufficient_condition_not_necessary_condition_sufficient_but_not_necessary_l115_11531

/-- Two lines are parallel if and only if their slopes are equal and not equal to 1/2 -/
def are_parallel (m : ℝ) : Prop :=
  m / 1 = 1 / m ∧ m / 1 ≠ 1 / 2

/-- The condition m = -1 is sufficient for the lines to be parallel -/
theorem sufficient_condition (m : ℝ) :
  m = -1 → are_parallel m :=
sorry

/-- The condition m = -1 is not necessary for the lines to be parallel -/
theorem not_necessary_condition :
  ∃ m : ℝ, m ≠ -1 ∧ are_parallel m :=
sorry

/-- m = -1 is a sufficient but not necessary condition for the lines to be parallel -/
theorem sufficient_but_not_necessary :
  (∀ m : ℝ, m = -1 → are_parallel m) ∧
  (∃ m : ℝ, m ≠ -1 ∧ are_parallel m) :=
sorry

end NUMINAMATH_CALUDE_sufficient_condition_not_necessary_condition_sufficient_but_not_necessary_l115_11531


namespace NUMINAMATH_CALUDE_real_square_nonnegative_and_no_real_square_root_of_negative_one_l115_11519

theorem real_square_nonnegative_and_no_real_square_root_of_negative_one :
  (∀ x : ℝ, x^2 ≥ 0) ∧ ¬(∃ x : ℝ, x^2 = -1) := by
  sorry

end NUMINAMATH_CALUDE_real_square_nonnegative_and_no_real_square_root_of_negative_one_l115_11519


namespace NUMINAMATH_CALUDE_complex_pure_imaginary_l115_11533

/-- Given that i is the imaginary unit and (1-2i)(a+i) is a pure imaginary number, prove that a = -2 -/
theorem complex_pure_imaginary (a : ℝ) : 
  (∃ (b : ℝ), (1 - 2 * Complex.I) * (a + Complex.I) = b * Complex.I) → a = -2 := by
  sorry

end NUMINAMATH_CALUDE_complex_pure_imaginary_l115_11533


namespace NUMINAMATH_CALUDE_bucket_to_leak_ratio_l115_11506

def leak_rate : ℚ := 3/2
def max_time : ℚ := 12
def bucket_capacity : ℚ := 36

theorem bucket_to_leak_ratio : 
  bucket_capacity / (leak_rate * max_time) = 2 := by
  sorry

end NUMINAMATH_CALUDE_bucket_to_leak_ratio_l115_11506


namespace NUMINAMATH_CALUDE_quadratic_roots_opposite_signs_l115_11509

theorem quadratic_roots_opposite_signs (a : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ x * y < 0 ∧ 
   a * x^2 - (a + 3) * x + 2 = 0 ∧
   a * y^2 - (a + 3) * y + 2 = 0) ↔ 
  a < 0 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_opposite_signs_l115_11509


namespace NUMINAMATH_CALUDE_exist_four_numbers_perfect_squares_l115_11515

theorem exist_four_numbers_perfect_squares :
  ∃ (a b c d : ℕ), 
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    ∃ (m n : ℕ), a^2 + 2*c*d + b^2 = m^2 ∧ c^2 + 2*a*b + d^2 = n^2 := by
  sorry

end NUMINAMATH_CALUDE_exist_four_numbers_perfect_squares_l115_11515


namespace NUMINAMATH_CALUDE_geometric_sequence_increasing_condition_l115_11585

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) :=
  ∀ n, a (n + 1) = q * a n

def monotonically_increasing (a : ℕ → ℝ) :=
  ∀ n m, n < m → a n < a m

theorem geometric_sequence_increasing_condition (a : ℕ → ℝ) (q : ℝ) :
  geometric_sequence a q →
  (¬ (q > 1 → monotonically_increasing a) ∧ ¬ (monotonically_increasing a → q > 1)) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_increasing_condition_l115_11585


namespace NUMINAMATH_CALUDE_sum_of_repeating_decimals_l115_11540

-- Define the repeating decimals
def repeating_6 : ℚ := 2/3
def repeating_45 : ℚ := 5/11

-- State the theorem
theorem sum_of_repeating_decimals :
  repeating_6 + repeating_45 = 37/33 := by sorry

end NUMINAMATH_CALUDE_sum_of_repeating_decimals_l115_11540


namespace NUMINAMATH_CALUDE_equation_identity_l115_11597

theorem equation_identity (a b c x : ℝ) (hab : a ≠ b) (hbc : b ≠ c) (hac : a ≠ c) :
  a^2 * ((x-b)/(a-b)) * ((x-c)/(a-c)) + b^2 * ((x-a)/(b-a)) * ((x-c)/(b-c)) + c^2 * ((x-a)/(c-a)) * ((x-b)/(c-b)) = x^2 := by
  sorry

end NUMINAMATH_CALUDE_equation_identity_l115_11597


namespace NUMINAMATH_CALUDE_square_side_length_l115_11588

theorem square_side_length : ∃ (x : ℝ), x > 0 ∧ x^2 = 6^2 + 8^2 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_square_side_length_l115_11588


namespace NUMINAMATH_CALUDE_smallest_possible_a_l115_11547

theorem smallest_possible_a (P : ℤ → ℤ) (a : ℕ+) : 
  (∀ x, ∃ (c : ℤ), P x = c) →  -- P has integer coefficients
  (P 1 = a) →
  (P 3 = a) →
  (P 2 = -a) →
  (P 4 = -a) →
  (P 6 = -a) →
  (P 8 = -a) →
  (∀ b : ℕ+, b < 105 → 
    ¬(∃ Q : ℤ → ℤ, 
      (∀ x, ∃ (c : ℤ), Q x = c) ∧  -- Q has integer coefficients
      (Q 1 = b) ∧
      (Q 3 = b) ∧
      (Q 2 = -b) ∧
      (Q 4 = -b) ∧
      (Q 6 = -b) ∧
      (Q 8 = -b)
    )
  ) →
  a = 105 :=
by sorry

end NUMINAMATH_CALUDE_smallest_possible_a_l115_11547


namespace NUMINAMATH_CALUDE_sequential_draw_probability_l115_11523

/-- The number of cards in a standard deck -/
def standardDeckSize : ℕ := 52

/-- The number of cards of each suit in a standard deck -/
def cardsPerSuit : ℕ := 13

/-- The probability of drawing a club, then a diamond, then a heart in order from a standard deck -/
def sequentialDrawProbability : ℚ :=
  (cardsPerSuit : ℚ) / standardDeckSize *
  (cardsPerSuit : ℚ) / (standardDeckSize - 1) *
  (cardsPerSuit : ℚ) / (standardDeckSize - 2)

theorem sequential_draw_probability :
  sequentialDrawProbability = 2197 / 132600 := by
  sorry

end NUMINAMATH_CALUDE_sequential_draw_probability_l115_11523


namespace NUMINAMATH_CALUDE_two_numbers_problem_l115_11548

theorem two_numbers_problem (x y : ℕ+) : 
  x + y = 667 →
  Nat.lcm x y / Nat.gcd x y = 120 →
  ((x = 232 ∧ y = 435) ∨ (x = 552 ∧ y = 115)) :=
by sorry

end NUMINAMATH_CALUDE_two_numbers_problem_l115_11548


namespace NUMINAMATH_CALUDE_A_less_than_B_l115_11589

theorem A_less_than_B : ∀ x : ℝ, (x + 3) * (x + 7) < (x + 4) * (x + 6) := by
  sorry

end NUMINAMATH_CALUDE_A_less_than_B_l115_11589


namespace NUMINAMATH_CALUDE_cakes_sold_minus_bought_l115_11512

/-- Given the initial number of cakes, the number of cakes sold, and the number of cakes bought,
    prove that the difference between cakes sold and cakes bought is 274. -/
theorem cakes_sold_minus_bought (initial : ℕ) (sold : ℕ) (bought : ℕ) 
    (h1 : initial = 648) 
    (h2 : sold = 467) 
    (h3 : bought = 193) : 
    sold - bought = 274 := by
  sorry

end NUMINAMATH_CALUDE_cakes_sold_minus_bought_l115_11512


namespace NUMINAMATH_CALUDE_college_students_count_l115_11595

/-- Calculates the total number of students in a college given the ratio of boys to girls and the number of girls -/
def total_students (boys_ratio : ℕ) (girls_ratio : ℕ) (num_girls : ℕ) : ℕ :=
  let num_boys := boys_ratio * num_girls / girls_ratio
  num_boys + num_girls

/-- Proves that in a college with a boys to girls ratio of 8:4 and 200 girls, the total number of students is 600 -/
theorem college_students_count : total_students 8 4 200 = 600 := by
  sorry

end NUMINAMATH_CALUDE_college_students_count_l115_11595


namespace NUMINAMATH_CALUDE_factorial_calculation_l115_11525

theorem factorial_calculation : (4 * Nat.factorial 6 + 32 * Nat.factorial 5) / Nat.factorial 7 = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_factorial_calculation_l115_11525


namespace NUMINAMATH_CALUDE_min_value_x_l115_11582

theorem min_value_x (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h : Real.log x / Real.log 3 ≥ Real.log 5 / Real.log 3 + (1/3) * (Real.log y / Real.log 3)) :
  x ≥ 5 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_min_value_x_l115_11582


namespace NUMINAMATH_CALUDE_arithmetic_sequence_geometric_mean_l115_11543

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + d * (n - 1)

theorem arithmetic_sequence_geometric_mean 
  (d : ℝ) (k : ℕ) 
  (h_d : d ≠ 0) 
  (h_k : k > 0) :
  let a := arithmetic_sequence (9 * d) d
  (a k) ^ 2 = a 1 * a (2 * k) → k = 4 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_geometric_mean_l115_11543


namespace NUMINAMATH_CALUDE_inequality_proof_l115_11522

theorem inequality_proof (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  Real.sqrt ((a^2 + b^2 + c^2 + d^2) / 4) ≥ (((a*b*c + a*b*d + a*c*d + b*c*d) / 4) ^ (1/3)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l115_11522


namespace NUMINAMATH_CALUDE_prob_more_ones_than_eights_l115_11539

/-- The number of sides on each die -/
def numSides : ℕ := 8

/-- The number of dice rolled -/
def numDice : ℕ := 5

/-- The total number of possible outcomes when rolling numDice dice with numSides sides -/
def totalOutcomes : ℕ := numSides ^ numDice

/-- The number of ways to roll an equal number of 1's and 8's -/
def equalOnesAndEights : ℕ := 12276

/-- The probability of rolling more 1's than 8's when rolling numDice fair dice with numSides sides -/
def probMoreOnesThanEights : ℚ := 10246 / 32768

theorem prob_more_ones_than_eights :
  probMoreOnesThanEights = 1/2 * (1 - equalOnesAndEights / totalOutcomes) :=
sorry

end NUMINAMATH_CALUDE_prob_more_ones_than_eights_l115_11539


namespace NUMINAMATH_CALUDE_chips_price_increase_l115_11514

/-- The cost of a pack of pretzels in dollars -/
def pretzel_cost : ℝ := 4

/-- The number of packets of chips bought -/
def chips_bought : ℕ := 2

/-- The number of packets of pretzels bought -/
def pretzels_bought : ℕ := 2

/-- The total cost of the purchase in dollars -/
def total_cost : ℝ := 22

/-- The percentage increase in the price of chips compared to pretzels -/
def price_increase_percentage : ℝ := 75

theorem chips_price_increase :
  let chips_cost := pretzel_cost * (1 + price_increase_percentage / 100)
  chips_bought * chips_cost + pretzels_bought * pretzel_cost = total_cost :=
by sorry

end NUMINAMATH_CALUDE_chips_price_increase_l115_11514


namespace NUMINAMATH_CALUDE_benny_spent_amount_l115_11556

/-- Represents the total amount spent in US dollars -/
def total_spent (initial_amount : ℝ) (remaining_amount : ℝ) : ℝ :=
  initial_amount - remaining_amount

/-- Theorem stating that given the initial amount of 200 US dollars and
    the remaining amount of 45 US dollars, the total amount spent is 155 US dollars -/
theorem benny_spent_amount :
  total_spent 200 45 = 155 := by sorry

end NUMINAMATH_CALUDE_benny_spent_amount_l115_11556


namespace NUMINAMATH_CALUDE_max_cross_sectional_area_l115_11569

-- Define the prism
def prism_base_side_length : ℝ := 8

-- Define the cutting plane
def cutting_plane (x y z : ℝ) : Prop := 3 * x - 5 * y + 2 * z = 20

-- Define the cross-sectional area function
noncomputable def cross_sectional_area (h : ℝ) : ℝ := 
  let diagonal := (2 * prism_base_side_length ^ 2 + h ^ 2) ^ (1/2 : ℝ)
  let area := h * diagonal / 2
  area

-- Theorem statement
theorem max_cross_sectional_area :
  ∃ h : ℝ, h > 0 ∧ 
    cross_sectional_area h = 9 * (38 : ℝ).sqrt ∧
    ∀ h' : ℝ, h' > 0 → cross_sectional_area h' ≤ cross_sectional_area h :=
by sorry

end NUMINAMATH_CALUDE_max_cross_sectional_area_l115_11569


namespace NUMINAMATH_CALUDE_grid_arithmetic_sequence_l115_11564

theorem grid_arithmetic_sequence (row : Fin 7 → ℚ) (col1 col2 : Fin 5 → ℚ) :
  -- The row forms an arithmetic sequence
  (∀ i : Fin 6, row (i + 1) - row i = row 1 - row 0) →
  -- The first column forms an arithmetic sequence
  (∀ i : Fin 4, col1 (i + 1) - col1 i = col1 1 - col1 0) →
  -- The second column forms an arithmetic sequence
  (∀ i : Fin 4, col2 (i + 1) - col2 i = col2 1 - col2 0) →
  -- Given values
  row 0 = 25 →
  col1 2 = 16 →
  col1 3 = 20 →
  col2 4 = -21 →
  -- The fourth element in the row is the same as the first element in the first column
  row 3 = col1 0 →
  -- The last element in the row is the same as the first element in the second column
  row 6 = col2 0 →
  -- M is the first element in the second column
  col2 0 = 1021 / 12 := by
sorry

end NUMINAMATH_CALUDE_grid_arithmetic_sequence_l115_11564


namespace NUMINAMATH_CALUDE_milk_selection_l115_11578

theorem milk_selection (total : ℕ) (soda_count : ℕ) (milk_percent : ℚ) (soda_percent : ℚ) :
  soda_percent = 60 / 100 →
  milk_percent = 20 / 100 →
  soda_count = 72 →
  (milk_percent / soda_percent) * soda_count = 24 := by
  sorry

end NUMINAMATH_CALUDE_milk_selection_l115_11578


namespace NUMINAMATH_CALUDE_analects_reasoning_is_common_sense_l115_11549

/-- Represents the types of reasoning --/
inductive ReasoningType
  | CommonSense
  | Inductive
  | Analogical
  | Deductive

/-- Represents a step in the logical progression --/
structure LogicalStep where
  premise : String
  consequence : String

/-- Represents the characteristics of the reasoning in the Analects passage --/
structure AnalectsReasoning where
  steps : List LogicalStep
  alignsWithCommonSense : Bool
  followsLogicalProgression : Bool

/-- Determines the type of reasoning based on its characteristics --/
def determineReasoningType (reasoning : AnalectsReasoning) : ReasoningType :=
  if reasoning.alignsWithCommonSense && reasoning.followsLogicalProgression then
    ReasoningType.CommonSense
  else
    ReasoningType.Inductive -- Default to another type if conditions are not met

/-- The main theorem stating that the reasoning in the Analects passage is Common Sense reasoning --/
theorem analects_reasoning_is_common_sense (analectsReasoning : AnalectsReasoning) 
    (h1 : analectsReasoning.steps.length > 0)
    (h2 : analectsReasoning.alignsWithCommonSense = true)
    (h3 : analectsReasoning.followsLogicalProgression = true) :
  determineReasoningType analectsReasoning = ReasoningType.CommonSense := by
  sorry

#check analects_reasoning_is_common_sense

end NUMINAMATH_CALUDE_analects_reasoning_is_common_sense_l115_11549


namespace NUMINAMATH_CALUDE_investment_calculation_l115_11558

/-- Calculates the investment amount given dividend information -/
theorem investment_calculation (face_value premium dividend_rate dividend_received : ℚ) : 
  face_value = 100 →
  premium = 20 / 100 →
  dividend_rate = 5 / 100 →
  dividend_received = 600 →
  (dividend_received / (face_value * dividend_rate)) * (face_value * (1 + premium)) = 14400 := by
  sorry

end NUMINAMATH_CALUDE_investment_calculation_l115_11558


namespace NUMINAMATH_CALUDE_division_remainder_3500_74_l115_11583

theorem division_remainder_3500_74 : ∃ q : ℕ, 3500 = 74 * q + 22 ∧ 22 < 74 := by
  sorry

end NUMINAMATH_CALUDE_division_remainder_3500_74_l115_11583


namespace NUMINAMATH_CALUDE_fraction_evaluation_l115_11574

theorem fraction_evaluation : (5 * 6 - 3 * 4) / (6 + 3) = 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l115_11574


namespace NUMINAMATH_CALUDE_initial_marbles_l115_11579

-- Define the variables
def marbles_given : ℕ := 14
def marbles_left : ℕ := 50

-- State the theorem
theorem initial_marbles : marbles_given + marbles_left = 64 := by
  sorry

end NUMINAMATH_CALUDE_initial_marbles_l115_11579


namespace NUMINAMATH_CALUDE_divisibility_by_eight_l115_11571

theorem divisibility_by_eight (n : ℤ) (h : Even n) :
  ∃ k₁ k₂ k₃ k₄ : ℤ,
    n * (n^2 + 20) = 8 * k₁ ∧
    n * (n^2 - 20) = 8 * k₂ ∧
    n * (n^2 + 4) = 8 * k₃ ∧
    n * (n^2 - 4) = 8 * k₄ :=
by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_eight_l115_11571


namespace NUMINAMATH_CALUDE_yoga_studio_average_weight_l115_11552

theorem yoga_studio_average_weight
  (num_men : ℕ)
  (num_women : ℕ)
  (avg_weight_men : ℝ)
  (avg_weight_women : ℝ)
  (h1 : num_men = 8)
  (h2 : num_women = 6)
  (h3 : avg_weight_men = 190)
  (h4 : avg_weight_women = 120)
  : (num_men * avg_weight_men + num_women * avg_weight_women) / (num_men + num_women) = 160 :=
by
  sorry

end NUMINAMATH_CALUDE_yoga_studio_average_weight_l115_11552


namespace NUMINAMATH_CALUDE_B_join_time_l115_11517

/-- Represents the time (in months) when B joined the business -/
def time_B_joined : ℝ := 7.5

/-- A's initial investment -/
def A_investment : ℝ := 27000

/-- B's investment when joining -/
def B_investment : ℝ := 36000

/-- Total duration of the business in months -/
def total_duration : ℝ := 12

/-- Ratio of A's profit share to B's profit share -/
def profit_ratio : ℝ := 2

theorem B_join_time :
  (A_investment * total_duration) / (B_investment * (total_duration - time_B_joined)) = profit_ratio := by
  sorry

end NUMINAMATH_CALUDE_B_join_time_l115_11517


namespace NUMINAMATH_CALUDE_arrangement_theorem_l115_11504

def num_boys : ℕ := 3
def num_girls : ℕ := 4
def total_people : ℕ := num_boys + num_girls

def arrange_condition1 : ℕ := sorry

def arrange_condition2 : ℕ := sorry

def arrange_condition3 : ℕ := sorry

def arrange_condition4 : ℕ := sorry

theorem arrangement_theorem :
  arrange_condition1 = 2160 ∧
  arrange_condition2 = 720 ∧
  arrange_condition3 = 144 ∧
  arrange_condition4 = 720 :=
sorry

end NUMINAMATH_CALUDE_arrangement_theorem_l115_11504


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l115_11529

theorem partial_fraction_decomposition (N₁ N₂ : ℚ) :
  (∀ x : ℚ, x ≠ 2 → x ≠ 3 →
    (50 * x - 42) / (x^2 - 5*x + 6) = N₁ / (x - 2) + N₂ / (x - 3)) →
  N₁ * N₂ = -6264 := by
sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l115_11529


namespace NUMINAMATH_CALUDE_triangle_max_side_length_l115_11591

theorem triangle_max_side_length (P Q R : Real) (a b : Real) :
  -- Triangle angles
  P + Q + R = Real.pi →
  -- Given condition
  Real.cos (3 * P) + Real.cos (3 * Q) + Real.cos (3 * R) = 1 →
  -- Two sides have lengths 12 and 15
  a = 12 ∧ b = 15 →
  -- Maximum length of the third side
  ∃ c : Real, c ≤ 27 ∧ 
    ∀ c' : Real, (c' ^ 2 ≤ a ^ 2 + b ^ 2 - 2 * a * b * Real.cos R) → c' ≤ c :=
by sorry

end NUMINAMATH_CALUDE_triangle_max_side_length_l115_11591


namespace NUMINAMATH_CALUDE_angle_covered_in_three_layers_l115_11524

theorem angle_covered_in_three_layers 
  (total_angle : ℝ) 
  (sum_of_angles : ℝ) 
  (angle_three_layers : ℝ) 
  (h1 : total_angle = 90) 
  (h2 : sum_of_angles = 290) 
  (h3 : angle_three_layers * 3 + (total_angle - angle_three_layers) * 2 = sum_of_angles) :
  angle_three_layers = 20 := by
sorry

end NUMINAMATH_CALUDE_angle_covered_in_three_layers_l115_11524


namespace NUMINAMATH_CALUDE_class_survey_is_comprehensive_l115_11528

/-- Represents a survey population -/
structure SurveyPopulation where
  size : ℕ
  is_finite : Bool

/-- Defines what makes a survey comprehensive -/
def is_comprehensive_survey (pop : SurveyPopulation) : Prop :=
  pop.is_finite ∧ pop.size > 0

/-- Represents a class of students -/
def class_of_students : SurveyPopulation :=
  { size := 30,  -- Assuming an average class size
    is_finite := true }

/-- Theorem stating that a survey of a class is suitable for a comprehensive survey -/
theorem class_survey_is_comprehensive :
  is_comprehensive_survey class_of_students := by
  sorry


end NUMINAMATH_CALUDE_class_survey_is_comprehensive_l115_11528


namespace NUMINAMATH_CALUDE_cost_of_750_apples_l115_11577

/-- The cost of buying a given number of apples, given the price and quantity of a bag of apples -/
def cost_of_apples (apples_per_bag : ℕ) (price_per_bag : ℕ) (total_apples : ℕ) : ℕ :=
  (total_apples / apples_per_bag) * price_per_bag

/-- Theorem: The cost of 750 apples is $120, given that a bag of 50 apples costs $8 -/
theorem cost_of_750_apples :
  cost_of_apples 50 8 750 = 120 := by
  sorry

#eval cost_of_apples 50 8 750

end NUMINAMATH_CALUDE_cost_of_750_apples_l115_11577


namespace NUMINAMATH_CALUDE_fish_population_estimate_l115_11584

theorem fish_population_estimate 
  (initially_marked : ℕ) 
  (second_catch : ℕ) 
  (marked_in_second : ℕ) 
  (h1 : initially_marked = 30)
  (h2 : second_catch = 50)
  (h3 : marked_in_second = 2) :
  (initially_marked * second_catch) / marked_in_second = 750 :=
by
  sorry

#check fish_population_estimate

end NUMINAMATH_CALUDE_fish_population_estimate_l115_11584


namespace NUMINAMATH_CALUDE_comic_book_arrangements_l115_11526

def batman_comics : ℕ := 5
def superman_comics : ℕ := 3
def xmen_comics : ℕ := 6
def ironman_comics : ℕ := 4

def total_arrangements : ℕ := 2987520000

theorem comic_book_arrangements :
  (batman_comics.factorial * superman_comics.factorial * xmen_comics.factorial * ironman_comics.factorial) *
  (batman_comics + superman_comics + xmen_comics + ironman_comics).factorial =
  total_arrangements := by sorry

end NUMINAMATH_CALUDE_comic_book_arrangements_l115_11526


namespace NUMINAMATH_CALUDE_percentage_calculation_l115_11507

def total_students : ℕ := 40
def a_both_tests : ℕ := 4
def b_both_tests : ℕ := 6
def c_both_tests : ℕ := 3
def d_test1_c_test2 : ℕ := 2
def d_both_tests : ℕ := 1

theorem percentage_calculation : 
  (a_both_tests + b_both_tests + c_both_tests + d_test1_c_test2 : ℚ) / total_students * 100 = 37.5 := by
  sorry

end NUMINAMATH_CALUDE_percentage_calculation_l115_11507


namespace NUMINAMATH_CALUDE_montague_population_fraction_l115_11520

/-- The fraction of the population living in Montague province -/
def montague_fraction : ℝ := sorry

/-- The fraction of the population living in Capulet province -/
def capulet_fraction : ℝ := sorry

/-- The theorem stating the conditions and the result to be proved -/
theorem montague_population_fraction :
  -- Conditions
  (montague_fraction + capulet_fraction = 1) ∧
  (0.8 * montague_fraction + 0.3 * capulet_fraction = 0.7 * capulet_fraction / (7/11)) →
  -- Conclusion
  montague_fraction = 2/3 := by sorry

end NUMINAMATH_CALUDE_montague_population_fraction_l115_11520


namespace NUMINAMATH_CALUDE_abs_x_minus_y_equals_four_l115_11568

theorem abs_x_minus_y_equals_four (x y : ℝ) 
  (h1 : x^3 + y^3 = 26)
  (h2 : x*y*(x+y) = -6) : 
  |x - y| = 4 := by
sorry

end NUMINAMATH_CALUDE_abs_x_minus_y_equals_four_l115_11568


namespace NUMINAMATH_CALUDE_possible_values_of_y_l115_11565

theorem possible_values_of_y (x y : ℝ) :
  |x - Real.sin (Real.log y)| = x + Real.sin (Real.log y) →
  ∃ n : ℤ, y = Real.exp (2 * π * ↑n) :=
sorry

end NUMINAMATH_CALUDE_possible_values_of_y_l115_11565


namespace NUMINAMATH_CALUDE_sin_C_value_max_area_l115_11562

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.b = Real.sqrt 5 ∧ 2 * Real.sin t.A = t.a * Real.cos t.B

-- Theorem 1: If c = 2, then sin C = 2/3
theorem sin_C_value (t : Triangle) (h : triangle_conditions t) (hc : t.c = 2) :
  Real.sin t.C = 2/3 := by
  sorry

-- Theorem 2: Maximum area of triangle ABC
theorem max_area (t : Triangle) (h : triangle_conditions t) :
  ∃ (max_area : ℝ), max_area = (5 * Real.sqrt 5) / 4 ∧
  ∀ (actual_area : ℝ), actual_area = (1/2) * t.a * t.b * Real.sin t.C → actual_area ≤ max_area := by
  sorry

end NUMINAMATH_CALUDE_sin_C_value_max_area_l115_11562


namespace NUMINAMATH_CALUDE_andy_profit_per_cake_l115_11527

/-- Calculates the profit per cake for Andy's cake business -/
def profit_per_cake (ingredient_cost_two_cakes : ℚ) (packaging_cost_per_cake : ℚ) (selling_price : ℚ) : ℚ :=
  selling_price - (ingredient_cost_two_cakes / 2 + packaging_cost_per_cake)

/-- Proves that Andy's profit per cake is $8 -/
theorem andy_profit_per_cake :
  profit_per_cake 12 1 15 = 8 := by
  sorry

end NUMINAMATH_CALUDE_andy_profit_per_cake_l115_11527


namespace NUMINAMATH_CALUDE_bhanu_house_rent_expenditure_l115_11536

/-- Calculates Bhanu's expenditure on house rent given his spending patterns and petrol expense -/
def house_rent_expenditure (total_income : ℝ) (petrol_percentage : ℝ) (rent_percentage : ℝ) (petrol_expense : ℝ) : ℝ :=
  let remaining_income := total_income - petrol_expense
  rent_percentage * remaining_income

/-- Proves that Bhanu's expenditure on house rent is 210 given his spending patterns and petrol expense -/
theorem bhanu_house_rent_expenditure :
  ∀ (total_income : ℝ),
    total_income > 0 →
    house_rent_expenditure total_income 0.3 0.3 300 = 210 :=
by
  sorry

#eval house_rent_expenditure 1000 0.3 0.3 300

end NUMINAMATH_CALUDE_bhanu_house_rent_expenditure_l115_11536


namespace NUMINAMATH_CALUDE_painting_price_increase_l115_11573

theorem painting_price_increase (x : ℝ) : 
  (1 + x / 100) * (1 - 0.15) = 1.0625 → x = 25 := by
  sorry

end NUMINAMATH_CALUDE_painting_price_increase_l115_11573


namespace NUMINAMATH_CALUDE_lucy_second_round_cookies_l115_11581

/-- The number of cookies Lucy sold on her first round -/
def first_round : ℕ := 34

/-- The total number of cookies Lucy sold -/
def total : ℕ := 61

/-- The number of cookies Lucy sold on her second round -/
def second_round : ℕ := total - first_round

theorem lucy_second_round_cookies : second_round = 27 := by
  sorry

end NUMINAMATH_CALUDE_lucy_second_round_cookies_l115_11581


namespace NUMINAMATH_CALUDE_radio_cost_price_l115_11560

/-- The cost price of a radio given its selling price and loss percentage -/
def cost_price (selling_price : ℚ) (loss_percentage : ℚ) : ℚ :=
  selling_price / (1 - loss_percentage / 100)

/-- Theorem: The cost price of a radio sold for 1245 with 17% loss is 1500 -/
theorem radio_cost_price :
  cost_price 1245 17 = 1500 := by
  sorry

end NUMINAMATH_CALUDE_radio_cost_price_l115_11560


namespace NUMINAMATH_CALUDE_two_digit_number_problem_l115_11500

theorem two_digit_number_problem (t : ℕ) : 
  t ≥ 10 ∧ t < 100 ∧ (13 * t) % 100 = 52 → t = 12 := by
  sorry

end NUMINAMATH_CALUDE_two_digit_number_problem_l115_11500


namespace NUMINAMATH_CALUDE_ball_hitting_ground_time_l115_11580

theorem ball_hitting_ground_time : 
  let f (t : ℝ) := -4.9 * t^2 + 4.5 * t + 6
  ∃ t : ℝ, t > 0 ∧ f t = 0 ∧ t = 8121 / 4900 := by
  sorry

end NUMINAMATH_CALUDE_ball_hitting_ground_time_l115_11580


namespace NUMINAMATH_CALUDE_painted_cube_theorem_l115_11538

theorem painted_cube_theorem (n : ℕ) (h : n > 2) :
  6 * (n - 2)^2 = (n - 2)^3 ↔ n = 8 := by
  sorry

end NUMINAMATH_CALUDE_painted_cube_theorem_l115_11538


namespace NUMINAMATH_CALUDE_horner_method_v4_l115_11551

def f (x : ℝ) : ℝ := 3*x^5 + 5*x^4 + 6*x^3 - 8*x^2 + 35*x + 12

def horner_v4 (a₅ a₄ a₃ a₂ a₁ a₀ x : ℝ) : ℝ :=
  ((((a₅ * x + a₄) * x + a₃) * x + a₂) * x + a₁) * x + a₀

theorem horner_method_v4 :
  horner_v4 3 5 6 (-8) 35 12 (-2) = 83 :=
sorry

end NUMINAMATH_CALUDE_horner_method_v4_l115_11551


namespace NUMINAMATH_CALUDE_calculation_proof_l115_11510

theorem calculation_proof : 
  1.2008 * 0.2008 * 2.4016 - 1.2008^3 - 1.2008 * 0.2008^2 = -1.2008 := by
sorry

end NUMINAMATH_CALUDE_calculation_proof_l115_11510


namespace NUMINAMATH_CALUDE_dodecahedron_interior_diagonals_l115_11516

/-- Represents a dodecahedron -/
structure Dodecahedron where
  /-- The number of faces in a dodecahedron -/
  faces : ℕ
  /-- The number of vertices in a dodecahedron -/
  vertices : ℕ
  /-- The number of faces meeting at each vertex -/
  faces_per_vertex : ℕ
  /-- Assertion that the dodecahedron has 12 faces -/
  faces_eq : faces = 12
  /-- Assertion that the dodecahedron has 20 vertices -/
  vertices_eq : vertices = 20
  /-- Assertion that 3 faces meet at each vertex -/
  faces_per_vertex_eq : faces_per_vertex = 3

/-- Calculates the number of interior diagonals in a dodecahedron -/
def interior_diagonals (d : Dodecahedron) : ℕ :=
  (d.vertices * (d.vertices - d.faces_per_vertex - 1)) / 2

/-- Theorem stating that a dodecahedron has 160 interior diagonals -/
theorem dodecahedron_interior_diagonals (d : Dodecahedron) : 
  interior_diagonals d = 160 := by
  sorry

end NUMINAMATH_CALUDE_dodecahedron_interior_diagonals_l115_11516


namespace NUMINAMATH_CALUDE_pencil_length_l115_11532

theorem pencil_length : ∀ (L : ℝ),
  (1/8 : ℝ) * L +  -- Black part
  (1/2 : ℝ) * ((7/8 : ℝ) * L) +  -- White part
  (7/2 : ℝ) = L  -- Blue part
  → L = 8 := by
sorry

end NUMINAMATH_CALUDE_pencil_length_l115_11532


namespace NUMINAMATH_CALUDE_equation_solution_l115_11534

theorem equation_solution : ∃ x : ℝ, 5 * 5^x + Real.sqrt (25 * 25^x) = 50 ∧ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l115_11534


namespace NUMINAMATH_CALUDE_smallest_b_value_l115_11594

theorem smallest_b_value (a b : ℕ) : 
  (∃ x y z : ℕ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ 
    (2 * a - b = x^2) ∧ 
    (a - 2 * b = y^2) ∧ 
    (a + b = z^2)) →
  (∀ b' : ℕ, b' < b → 
    ¬(∃ a' x' y' z' : ℕ, x' ≠ y' ∧ y' ≠ z' ∧ x' ≠ z' ∧
      (2 * a' - b' = x'^2) ∧ 
      (a' - 2 * b' = y'^2) ∧ 
      (a' + b' = z'^2))) →
  b = 3 :=
sorry

end NUMINAMATH_CALUDE_smallest_b_value_l115_11594


namespace NUMINAMATH_CALUDE_odd_function_property_y_value_at_3_l115_11542

def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^5 + b * x^3 + c * x

theorem odd_function_property (a b c : ℝ) :
  ∀ x, f a b c (-x) = -(f a b c x) :=
sorry

theorem y_value_at_3 (a b c : ℝ) :
  f a b c (-3) - 5 = 7 → f a b c 3 - 5 = -17 :=
sorry

end NUMINAMATH_CALUDE_odd_function_property_y_value_at_3_l115_11542


namespace NUMINAMATH_CALUDE_book_arrangement_theorem_l115_11576

def total_books : ℕ := 7
def science_books : ℕ := 2
def math_books : ℕ := 2
def unique_books : ℕ := total_books - science_books - math_books

def arrangements : ℕ := (total_books.factorial) / (science_books.factorial * math_books.factorial)

def highlighted_arrangements : ℕ := arrangements * (total_books.choose 2)

theorem book_arrangement_theorem :
  arrangements = 1260 ∧ highlighted_arrangements = 26460 := by
  sorry

end NUMINAMATH_CALUDE_book_arrangement_theorem_l115_11576


namespace NUMINAMATH_CALUDE_simplest_fraction_of_decimal_l115_11550

theorem simplest_fraction_of_decimal (a b : ℕ+) (h : (a : ℚ) / b = 0.478125) :
  (∀ d : ℕ+, d ∣ a → d ∣ b → d = 1) →
  (a : ℕ) = 153 ∧ b = 320 ∧ a + b = 473 := by
  sorry

end NUMINAMATH_CALUDE_simplest_fraction_of_decimal_l115_11550


namespace NUMINAMATH_CALUDE_factorization_problems_l115_11586

theorem factorization_problems :
  (∀ x : ℝ, x^3 - 9*x = x*(x+3)*(x-3)) ∧
  (∀ a b : ℝ, a^3*b - 2*a^2*b + a*b = a*b*(a-1)^2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_problems_l115_11586


namespace NUMINAMATH_CALUDE_x_intercept_ratio_l115_11508

-- Define the slopes and y-intercept
def m₁ : ℝ := 8
def m₂ : ℝ := 4
def c : ℝ := 0  -- y-intercept, defined as non-zero in the theorem

-- Define the x-intercepts
def u : ℝ := 0  -- actual value doesn't matter, will be constrained in the theorem
def v : ℝ := 0  -- actual value doesn't matter, will be constrained in the theorem

-- Theorem statement
theorem x_intercept_ratio (h₁ : c ≠ 0) 
                          (h₂ : m₁ * u + c = 0) 
                          (h₃ : m₂ * v + c = 0) : 
  u / v = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_x_intercept_ratio_l115_11508


namespace NUMINAMATH_CALUDE_units_digit_G_5_l115_11590

def G (n : ℕ) : ℕ := 2^(3^n) + 2

theorem units_digit_G_5 : G 5 % 10 = 0 := by sorry

end NUMINAMATH_CALUDE_units_digit_G_5_l115_11590


namespace NUMINAMATH_CALUDE_quadratic_inequality_l115_11593

theorem quadratic_inequality (a b : ℝ) : ∃ x₀ ∈ Set.Icc (-1 : ℝ) 1, |x₀^2 + a*x₀ + b| + a ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l115_11593


namespace NUMINAMATH_CALUDE_pie_remainder_l115_11545

theorem pie_remainder (whole_pie : ℝ) (carlos_share : ℝ) (maria_share : ℝ) : 
  carlos_share = 0.8 * whole_pie →
  maria_share = 0.25 * (whole_pie - carlos_share) →
  whole_pie - carlos_share - maria_share = 0.15 * whole_pie :=
by sorry

end NUMINAMATH_CALUDE_pie_remainder_l115_11545


namespace NUMINAMATH_CALUDE_clusters_per_box_l115_11553

/-- Given the following conditions:
    1. There are 4 clusters of oats in each spoonful.
    2. There are 25 spoonfuls of cereal in each bowl.
    3. There are 5 bowlfuls of cereal in each box.
    Prove that the number of clusters of oats in each box is equal to 500. -/
theorem clusters_per_box (clusters_per_spoon : ℕ) (spoons_per_bowl : ℕ) (bowls_per_box : ℕ)
  (h1 : clusters_per_spoon = 4)
  (h2 : spoons_per_bowl = 25)
  (h3 : bowls_per_box = 5) :
  clusters_per_spoon * spoons_per_bowl * bowls_per_box = 500 := by
  sorry

end NUMINAMATH_CALUDE_clusters_per_box_l115_11553


namespace NUMINAMATH_CALUDE_reeboks_sold_count_l115_11521

def quota : ℕ := 1000
def adidas_price : ℕ := 45
def nike_price : ℕ := 60
def reebok_price : ℕ := 35
def nike_sold : ℕ := 8
def adidas_sold : ℕ := 6
def above_goal : ℕ := 65

theorem reeboks_sold_count :
  ∃ (reebok_sold : ℕ),
    reebok_sold * reebok_price + nike_sold * nike_price + adidas_sold * adidas_price = quota + above_goal ∧
    reebok_sold = 9 := by
  sorry

end NUMINAMATH_CALUDE_reeboks_sold_count_l115_11521


namespace NUMINAMATH_CALUDE_vector_equality_implies_norm_equality_l115_11570

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

theorem vector_equality_implies_norm_equality 
  (a b : E) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a = -2 • b) : 
  ‖a‖ - ‖b‖ = ‖a + b‖ := by
  sorry

end NUMINAMATH_CALUDE_vector_equality_implies_norm_equality_l115_11570


namespace NUMINAMATH_CALUDE_polygon_sides_possibility_l115_11559

theorem polygon_sides_possibility : ∃ n : ℕ, n ≥ 10 ∧ (n - 3) * 180 = 1620 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_possibility_l115_11559


namespace NUMINAMATH_CALUDE_winnie_kept_balloons_l115_11537

/-- The number of balloons Winnie keeps for herself after distributing
    as many as possible equally among her friends -/
def balloons_kept (total_balloons : ℕ) (num_friends : ℕ) : ℕ :=
  total_balloons % num_friends

theorem winnie_kept_balloons :
  balloons_kept 200 12 = 8 := by
  sorry

end NUMINAMATH_CALUDE_winnie_kept_balloons_l115_11537


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l115_11513

theorem geometric_sequence_ratio (a₁ : ℝ) (q : ℝ) (h₁ : a₁ ≠ 0) :
  let S : ℕ → ℝ
    | 1 => a₁
    | 2 => a₁ + a₁ * q
    | 3 => a₁ + a₁ * q + a₁ * q^2
    | _ => 0  -- We only need S₁, S₂, and S₃ for this problem
  (S 3 - S 2 = S 2 - S 1) → q = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l115_11513


namespace NUMINAMATH_CALUDE_last_two_digits_product_l115_11501

/-- Given an integer n that is divisible by 6 and whose last two digits sum to 15,
    the product of its last two digits is 54. -/
theorem last_two_digits_product (n : ℤ) : 
  (n % 100 ≥ 0) →  -- Ensure we're dealing with the last two positive digits
  (n % 6 = 0) →    -- n is divisible by 6
  ((n % 100) / 10 + n % 10 = 15) →  -- Sum of last two digits is 15
  ((n % 100) / 10) * (n % 10) = 54 :=
by sorry

end NUMINAMATH_CALUDE_last_two_digits_product_l115_11501


namespace NUMINAMATH_CALUDE_regular_polygon_150_deg_interior_has_12_sides_l115_11518

/-- A regular polygon with interior angles measuring 150 degrees has 12 sides -/
theorem regular_polygon_150_deg_interior_has_12_sides :
  ∀ n : ℕ, 
  n > 2 →
  (180 * (n - 2) : ℝ) = (150 * n : ℝ) →
  n = 12 := by
sorry


end NUMINAMATH_CALUDE_regular_polygon_150_deg_interior_has_12_sides_l115_11518


namespace NUMINAMATH_CALUDE_margie_change_l115_11563

theorem margie_change : 
  let banana_cost : ℚ := 0.30
  let orange_cost : ℚ := 0.40
  let banana_count : ℕ := 5
  let orange_count : ℕ := 3
  let paid_amount : ℚ := 10.00
  let total_cost : ℚ := banana_cost * banana_count + orange_cost * orange_count
  let change : ℚ := paid_amount - total_cost
  change = 7.30 := by sorry

end NUMINAMATH_CALUDE_margie_change_l115_11563


namespace NUMINAMATH_CALUDE_average_of_numbers_is_eleven_l115_11572

theorem average_of_numbers_is_eleven : ∃ (M N : ℝ), 
  10 < N ∧ N < 20 ∧ 
  M = N - 4 ∧ 
  (8 + M + N) / 3 = 11 := by
  sorry

end NUMINAMATH_CALUDE_average_of_numbers_is_eleven_l115_11572


namespace NUMINAMATH_CALUDE_angle_BDC_is_45_l115_11567

-- Define the quadrilateral BCDE
structure Quadrilateral :=
  (B C D E : Point)

-- Define the angles
def angle_E (q : Quadrilateral) : ℝ := 25
def angle_C (q : Quadrilateral) : ℝ := 20

-- Define the angle BDC
def angle_BDC (q : Quadrilateral) : ℝ := angle_E q + angle_C q

-- State the theorem
theorem angle_BDC_is_45 (q : Quadrilateral) :
  angle_BDC q = 45 :=
sorry

end NUMINAMATH_CALUDE_angle_BDC_is_45_l115_11567


namespace NUMINAMATH_CALUDE_quadratic_roots_l115_11546

theorem quadratic_roots : 
  let f : ℝ → ℝ := λ x ↦ x^2 - 3
  ∃ x₁ x₂ : ℝ, x₁ = Real.sqrt 3 ∧ x₂ = -Real.sqrt 3 ∧ f x₁ = 0 ∧ f x₂ = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_l115_11546


namespace NUMINAMATH_CALUDE_jacks_allowance_l115_11555

/-- Calculates Jack's weekly allowance given the initial amount, number of weeks, and final amount in his piggy bank -/
def calculate_allowance (initial_amount : ℚ) (weeks : ℕ) (final_amount : ℚ) : ℚ :=
  2 * (final_amount - initial_amount) / weeks

/-- Proves that Jack's weekly allowance is $10 given the problem conditions -/
theorem jacks_allowance :
  let initial_amount : ℚ := 43
  let weeks : ℕ := 8
  let final_amount : ℚ := 83
  calculate_allowance initial_amount weeks final_amount = 10 := by
  sorry

#eval calculate_allowance 43 8 83

end NUMINAMATH_CALUDE_jacks_allowance_l115_11555


namespace NUMINAMATH_CALUDE_peace_treaty_day_l115_11598

-- Define the days of the week
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

-- Define a function to get the next day of the week
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday
  | DayOfWeek.Sunday => DayOfWeek.Monday

-- Define a function to advance a day by n days
def advanceDay (d : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => d
  | Nat.succ m => nextDay (advanceDay d m)

-- Theorem statement
theorem peace_treaty_day (startDay : DayOfWeek) (daysPassed : Nat) :
  startDay = DayOfWeek.Monday ∧ daysPassed = 893 →
  advanceDay startDay daysPassed = DayOfWeek.Saturday :=
by
  sorry -- Proof omitted as per instructions


end NUMINAMATH_CALUDE_peace_treaty_day_l115_11598


namespace NUMINAMATH_CALUDE_solve_dirt_bike_problem_l115_11505

def dirt_bike_problem (dirt_bike_cost off_road_cost registration_cost total_paid : ℕ) 
  (num_off_road : ℕ) : Prop :=
  ∃ (num_dirt_bikes : ℕ),
    num_dirt_bikes * (dirt_bike_cost + registration_cost) + 
    num_off_road * (off_road_cost + registration_cost) = total_paid ∧
    num_dirt_bikes = 3

theorem solve_dirt_bike_problem :
  dirt_bike_problem 150 300 25 1825 4 := by
  sorry

end NUMINAMATH_CALUDE_solve_dirt_bike_problem_l115_11505


namespace NUMINAMATH_CALUDE_fruit_seller_loss_percentage_l115_11557

/-- Calculates the percentage loss for a fruit seller given selling price, break-even price, and profit percentage. -/
def calculate_loss_percentage (selling_price profit_price profit_percentage : ℚ) : ℚ :=
  let cost_price := profit_price / (1 + profit_percentage / 100)
  let loss := cost_price - selling_price
  (loss / cost_price) * 100

/-- Theorem stating that under given conditions, the fruit seller's loss percentage is 15%. -/
theorem fruit_seller_loss_percentage :
  let selling_price : ℚ := 12
  let profit_price : ℚ := 14823529411764707 / 1000000000000000
  let profit_percentage : ℚ := 5
  calculate_loss_percentage selling_price profit_price profit_percentage = 15 := by
  sorry

#eval calculate_loss_percentage 12 (14823529411764707 / 1000000000000000) 5

end NUMINAMATH_CALUDE_fruit_seller_loss_percentage_l115_11557
