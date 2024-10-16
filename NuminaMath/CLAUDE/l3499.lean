import Mathlib

namespace NUMINAMATH_CALUDE_limit_of_a_sequence_l3499_349980

def a (n : ℕ) : ℚ := (9 - n^3) / (1 + 2*n^3)

theorem limit_of_a_sequence :
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |a n - (-1/2)| < ε :=
by sorry

end NUMINAMATH_CALUDE_limit_of_a_sequence_l3499_349980


namespace NUMINAMATH_CALUDE_average_female_students_l3499_349932

theorem average_female_students (class_8A class_8B class_8C class_8D class_8E : ℕ) 
  (h1 : class_8A = 10)
  (h2 : class_8B = 14)
  (h3 : class_8C = 7)
  (h4 : class_8D = 9)
  (h5 : class_8E = 13) : 
  (class_8A + class_8B + class_8C + class_8D + class_8E : ℚ) / 5 = 10.6 := by
  sorry

end NUMINAMATH_CALUDE_average_female_students_l3499_349932


namespace NUMINAMATH_CALUDE_no_real_solutions_for_ratio_equation_l3499_349971

theorem no_real_solutions_for_ratio_equation :
  ¬∃ (x : ℝ), (x + 3) / (2*x + 5) = (5*x + 4) / (8*x + 5) :=
by sorry

end NUMINAMATH_CALUDE_no_real_solutions_for_ratio_equation_l3499_349971


namespace NUMINAMATH_CALUDE_quadratic_sum_l3499_349902

/-- Given a quadratic function f(x) = -3x^2 + 27x + 135, 
    prove that when written in the form a(x+b)^2 + c,
    the sum of a, b, and c is 197.75 -/
theorem quadratic_sum (f : ℝ → ℝ) (a b c : ℝ) :
  (∀ x, f x = -3*x^2 + 27*x + 135) →
  (∀ x, f x = a*(x+b)^2 + c) →
  a + b + c = 197.75 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_l3499_349902


namespace NUMINAMATH_CALUDE_quoted_poetry_mismatch_l3499_349988

-- Define a type for poetry quotes
inductive PoetryQuote
| A
| B
| C
| D

-- Define a function to check if a quote matches its context
def matchesContext (quote : PoetryQuote) : Prop :=
  match quote with
  | PoetryQuote.A => True
  | PoetryQuote.B => True
  | PoetryQuote.C => True
  | PoetryQuote.D => False

-- Theorem statement
theorem quoted_poetry_mismatch :
  ∃ (q : PoetryQuote), ¬(matchesContext q) ∧ ∀ (p : PoetryQuote), p ≠ q → matchesContext p :=
by
  sorry

end NUMINAMATH_CALUDE_quoted_poetry_mismatch_l3499_349988


namespace NUMINAMATH_CALUDE_probability_three_out_of_five_dice_less_than_six_l3499_349913

/-- The probability of exactly three out of five fair 10-sided dice showing a number less than 6 -/
theorem probability_three_out_of_five_dice_less_than_six :
  let n : ℕ := 5  -- number of dice
  let k : ℕ := 3  -- number of successes (dice showing less than 6)
  let p : ℚ := 1/2  -- probability of a single die showing less than 6
  Nat.choose n k * p^k * (1-p)^(n-k) = 5/16 := by sorry

end NUMINAMATH_CALUDE_probability_three_out_of_five_dice_less_than_six_l3499_349913


namespace NUMINAMATH_CALUDE_total_cost_is_5080_l3499_349967

/-- Represents the price of a single small pack of paper -/
def small_pack_price : ℚ := 387/100

/-- Represents the price of a single large pack of paper -/
def large_pack_price : ℚ := 549/100

/-- Calculates the price of n small packs with the best discount -/
def small_pack_total (n : ℕ) : ℚ :=
  let discount_5 := 5/100
  let discount_10 := 10/100
  if n ≥ 10 then
    (1 - discount_10) * (n : ℚ) * small_pack_price
  else if n ≥ 5 then
    (1 - discount_5) * 5 * small_pack_price + (n - 5 : ℚ) * small_pack_price
  else
    (n : ℚ) * small_pack_price

/-- Calculates the price of n large packs with the best discount -/
def large_pack_total (n : ℕ) : ℚ :=
  let discount_3 := 7/100
  let discount_6 := 15/100
  if n ≥ 6 then
    (1 - discount_6) * (n : ℚ) * large_pack_price
  else if n ≥ 3 then
    (1 - discount_3) * 3 * large_pack_price + (n - 3 : ℚ) * large_pack_price
  else
    (n : ℚ) * large_pack_price

/-- The total cost of buying 8 small packs and 4 large packs -/
def total_cost : ℚ := small_pack_total 8 + large_pack_total 4

theorem total_cost_is_5080 : ⌊total_cost * 100⌋ = 5080 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_is_5080_l3499_349967


namespace NUMINAMATH_CALUDE_complement_of_A_union_B_l3499_349919

def U : Set Nat := {1, 2, 3, 4, 5}

def A : Set Nat := {x ∈ U | x^2 - 3*x + 2 = 0}

def B : Set Nat := {x ∈ U | ∃ a ∈ A, x = 2*a}

theorem complement_of_A_union_B (x : Nat) : 
  x ∈ (U \ (A ∪ B)) ↔ x = 3 ∨ x = 5 := by
  sorry

end NUMINAMATH_CALUDE_complement_of_A_union_B_l3499_349919


namespace NUMINAMATH_CALUDE_equal_sum_sequence_property_l3499_349910

def is_equal_sum_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n + a (n + 1) = a (n + 1) + a (n + 2)

theorem equal_sum_sequence_property (a : ℕ → ℝ) (h : is_equal_sum_sequence a) :
  (∀ k m : ℕ, Odd k → Odd m → a k = a m) ∧
  (∀ k m : ℕ, Even k → Even m → a k = a m) :=
by sorry

end NUMINAMATH_CALUDE_equal_sum_sequence_property_l3499_349910


namespace NUMINAMATH_CALUDE_probability_three_same_tunes_l3499_349948

/-- A defective toy train that produces only two different tunes at random -/
structure DefectiveToyTrain where
  tunes : Fin 2

/-- The probability of a specific sequence of tunes occurring -/
def probability_of_sequence (n : ℕ) : ℚ :=
  (1 / 2) ^ n

/-- The probability of producing n music tunes of the same type in a row -/
def probability_same_tune (n : ℕ) : ℚ :=
  2 * probability_of_sequence n

theorem probability_three_same_tunes :
  probability_same_tune 3 = 1 / 4 := by sorry

end NUMINAMATH_CALUDE_probability_three_same_tunes_l3499_349948


namespace NUMINAMATH_CALUDE_min_value_expression_l3499_349916

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x + 1/y) * (x + 1/y - 100) + (y + 1/x) * (y + 1/x - 100) ≥ -2500 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l3499_349916


namespace NUMINAMATH_CALUDE_line_up_permutations_l3499_349968

def number_of_people : ℕ := 5

theorem line_up_permutations :
  let youngest_not_first := number_of_people - 1
  let eldest_not_last := number_of_people - 1
  let remaining_positions := number_of_people - 2
  youngest_not_first * eldest_not_last * (remaining_positions.factorial) = 96 :=
by sorry

end NUMINAMATH_CALUDE_line_up_permutations_l3499_349968


namespace NUMINAMATH_CALUDE_floor_product_equality_l3499_349923

theorem floor_product_equality (Y : ℝ) : ⌊(0.3242 * Y)⌋ = 0.3242 * Y := by
  sorry

end NUMINAMATH_CALUDE_floor_product_equality_l3499_349923


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3499_349911

/-- A point in the second quadrant has a negative x-coordinate and positive y-coordinate -/
def second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

/-- The solution set of the inequality (2-m)x + 2 > m with respect to x -/
def solution_set (m : ℝ) : Set ℝ := {x | (2 - m) * x + 2 > m}

theorem inequality_solution_set (m : ℝ) :
  second_quadrant (3 - m) 1 → solution_set m = {x | x < -1} := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3499_349911


namespace NUMINAMATH_CALUDE_chocolates_remaining_theorem_l3499_349903

/-- Number of chocolates remaining after 4 days -/
def chocolates_remaining (total : ℕ) (day1 : ℕ) : ℕ :=
  let day2 := 2 * day1 - 3
  let day3 := day1 - 2
  let day4 := day3 - 1
  total - (day1 + day2 + day3 + day4)

/-- Theorem stating that 12 chocolates remain uneaten after 4 days -/
theorem chocolates_remaining_theorem :
  chocolates_remaining 24 4 = 12 := by
  sorry

#eval chocolates_remaining 24 4

end NUMINAMATH_CALUDE_chocolates_remaining_theorem_l3499_349903


namespace NUMINAMATH_CALUDE_balloon_theorem_l3499_349914

def balloon_problem (brooke_initial : ℕ) (brooke_added : ℕ) (tracy_initial : ℕ) (tracy_added : ℕ) : ℕ :=
  let brooke_total := brooke_initial + brooke_added
  let tracy_total := tracy_initial + tracy_added
  let tracy_after_popping := tracy_total - (tracy_total / 5 * 2)
  let brooke_after_giving := brooke_total - (brooke_total / 4)
  (tracy_after_popping - 5) + (brooke_after_giving - 5)

theorem balloon_theorem :
  balloon_problem 25 22 16 42 = 61 := by
  sorry

end NUMINAMATH_CALUDE_balloon_theorem_l3499_349914


namespace NUMINAMATH_CALUDE_bridge_length_calculation_l3499_349950

/-- Calculates the length of a bridge given train parameters --/
theorem bridge_length_calculation (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 135 →
  train_speed_kmh = 45 →
  crossing_time = 30 →
  (train_speed_kmh * 1000 / 3600) * crossing_time - train_length = 240 := by
  sorry

end NUMINAMATH_CALUDE_bridge_length_calculation_l3499_349950


namespace NUMINAMATH_CALUDE_triangle_area_specific_l3499_349954

noncomputable def triangle_area (a b c : ℝ) (A B C : ℝ) : ℝ :=
  (1/2) * b * c * Real.sin A

theorem triangle_area_specific : 
  ∀ (a b c : ℝ) (A B C : ℝ),
    b = 2 →
    c = 2 * Real.sqrt 2 →
    C = π / 4 →
    triangle_area a b c A B C = Real.sqrt 3 + 1 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_specific_l3499_349954


namespace NUMINAMATH_CALUDE_relay_race_time_reduction_l3499_349955

theorem relay_race_time_reduction (T : ℝ) (T1 T2 T3 T4 T5 : ℝ) :
  T > 0 ∧ T1 > 0 ∧ T2 > 0 ∧ T3 > 0 ∧ T4 > 0 ∧ T5 > 0 ∧
  T = T1 + T2 + T3 + T4 + T5 ∧
  T1/2 + T2 + T3 + T4 + T5 = 0.95 * T ∧
  T1 + T2/2 + T3 + T4 + T5 = 0.9 * T ∧
  T1 + T2 + T3/2 + T4 + T5 = 0.88 * T ∧
  T1 + T2 + T3 + T4/2 + T5 = 0.85 * T →
  T1 + T2 + T3 + T4 + T5/2 = 0.92 * T := by
sorry

end NUMINAMATH_CALUDE_relay_race_time_reduction_l3499_349955


namespace NUMINAMATH_CALUDE_arithmetic_sequence_product_l3499_349922

theorem arithmetic_sequence_product (a : ℝ) (d : ℝ) : 
  (a + 6 * d = 20) → (d = 2) → (a * (a + d) = 80) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_product_l3499_349922


namespace NUMINAMATH_CALUDE_interest_difference_l3499_349909

/-- Calculate the loss when using simple interest instead of compound interest -/
theorem interest_difference (principal : ℝ) (rate : ℝ) (time : ℝ) : 
  principal = 2500 →
  rate = 0.04 →
  time = 2 →
  principal * (1 + rate) ^ time - principal - (principal * rate * time) = 4 := by
  sorry

end NUMINAMATH_CALUDE_interest_difference_l3499_349909


namespace NUMINAMATH_CALUDE_system_solution_is_e_l3499_349901

theorem system_solution_is_e (x y z : ℝ) : 
  x = Real.exp (Real.log y) ∧ 
  y = Real.exp (Real.log z) ∧ 
  z = Real.exp (Real.log x) → 
  x = y ∧ y = z ∧ x = Real.exp 1 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_is_e_l3499_349901


namespace NUMINAMATH_CALUDE_exists_dominating_pair_l3499_349966

/-- A type representing a student's scores for three problems -/
def StudentScores := Fin 8 × Fin 8 × Fin 8

/-- The theorem statement -/
theorem exists_dominating_pair (students : Finset StudentScores) 
  (h : students.card = 249) : 
  ∃ (a b : StudentScores), a ∈ students ∧ b ∈ students ∧ 
    a.1 ≥ b.1 ∧ a.2.1 ≥ b.2.1 ∧ a.2.2 ≥ b.2.2 := by
  sorry

end NUMINAMATH_CALUDE_exists_dominating_pair_l3499_349966


namespace NUMINAMATH_CALUDE_product_of_sums_geq_one_l3499_349941

theorem product_of_sums_geq_one (a b c d : ℝ) 
  (h1 : a + b = 1) (h2 : c * d = 1) : 
  (a * c + b * d) * (a * d + b * c) ≥ 1 := by sorry

end NUMINAMATH_CALUDE_product_of_sums_geq_one_l3499_349941


namespace NUMINAMATH_CALUDE_coin_placement_coloring_l3499_349942

theorem coin_placement_coloring (n : ℕ) (h1 : 1 < n) (h2 : n < 2010) :
  (∃ (coloring : Fin 2010 → Fin n) (initial_positions : Fin n → Fin 2010),
    ∀ (t : ℕ) (i j : Fin n),
      i ≠ j →
      coloring ((initial_positions i + t) % 2010) ≠
      coloring ((initial_positions j + t) % 2010)) ↔
  2010 % n = 0 :=
sorry

end NUMINAMATH_CALUDE_coin_placement_coloring_l3499_349942


namespace NUMINAMATH_CALUDE_mistaken_quotient_l3499_349933

theorem mistaken_quotient (D : ℕ) : 
  D % 21 = 0 ∧ D / 21 = 20 → D / 12 = 35 := by
  sorry

end NUMINAMATH_CALUDE_mistaken_quotient_l3499_349933


namespace NUMINAMATH_CALUDE_triangle_properties_l3499_349945

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def is_valid_triangle (t : Triangle) : Prop :=
  t.a > 0 ∧ t.b > 0 ∧ t.c > 0 ∧
  t.A > 0 ∧ t.B > 0 ∧ t.C > 0 ∧
  t.A + t.B + t.C = Real.pi

def satisfies_equation (x : ℝ) : Prop :=
  x^2 - 2 * Real.sqrt 3 * x + 2 = 0

theorem triangle_properties (t : Triangle) 
  (h1 : is_valid_triangle t)
  (h2 : satisfies_equation t.a)
  (h3 : satisfies_equation t.b)
  (h4 : Real.cos (t.A + t.B) = 1/2) :
  t.C = Real.pi/3 ∧ 
  t.c = Real.sqrt 6 ∧
  (1/2 * t.a * t.b * Real.sin t.C) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l3499_349945


namespace NUMINAMATH_CALUDE_expected_value_of_twelve_sided_die_l3499_349931

/-- A fair 12-sided die with faces numbered from 1 to 12 -/
def twelve_sided_die : Finset ℕ := Finset.range 12

/-- The expected value of rolling the die -/
def expected_value : ℚ :=
  (Finset.sum twelve_sided_die (fun i => i + 1)) / 12

/-- Theorem: The expected value of rolling a fair 12-sided die with faces numbered from 1 to 12 is 6.5 -/
theorem expected_value_of_twelve_sided_die :
  expected_value = 13/2 := by sorry

end NUMINAMATH_CALUDE_expected_value_of_twelve_sided_die_l3499_349931


namespace NUMINAMATH_CALUDE_jennifer_theorem_l3499_349946

def jennifer_problem (initial_amount : ℚ) (sandwich_fraction : ℚ) (ticket_fraction : ℚ) (book_fraction : ℚ) : Prop :=
  let sandwich_cost := initial_amount * sandwich_fraction
  let ticket_cost := initial_amount * ticket_fraction
  let book_cost := initial_amount * book_fraction
  let total_spent := sandwich_cost + ticket_cost + book_cost
  let remaining := initial_amount - total_spent
  initial_amount = 90 ∧ 
  sandwich_fraction = 1/5 ∧ 
  ticket_fraction = 1/6 ∧ 
  book_fraction = 1/2 ∧ 
  remaining = 12

theorem jennifer_theorem : 
  ∃ (initial_amount sandwich_fraction ticket_fraction book_fraction : ℚ),
    jennifer_problem initial_amount sandwich_fraction ticket_fraction book_fraction :=
by
  sorry

end NUMINAMATH_CALUDE_jennifer_theorem_l3499_349946


namespace NUMINAMATH_CALUDE_min_magnitude_linear_combination_l3499_349905

/-- Given vectors a and b in ℝ², prove that the minimum magnitude of their linear combination c = xa + yb is √3, under specific conditions. -/
theorem min_magnitude_linear_combination (a b : ℝ × ℝ) 
  (h1 : ‖a‖ = 1) (h2 : ‖b‖ = 1) (h3 : a • b = 1/2) :
  ∃ (min : ℝ), min = Real.sqrt 3 ∧ 
  ∀ (x y : ℝ), x > 0 → y > 0 → x + y = 2 → 
  ‖x • a + y • b‖ ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_magnitude_linear_combination_l3499_349905


namespace NUMINAMATH_CALUDE_video_streaming_cost_l3499_349951

/-- Represents the monthly cost of a video streaming subscription -/
def monthly_cost : ℝ := 14

/-- Represents the number of months in a year -/
def months_in_year : ℕ := 12

/-- Represents the total cost paid by one person for a year -/
def total_cost_per_person : ℝ := 84

theorem video_streaming_cost : 
  monthly_cost * months_in_year = 2 * total_cost_per_person :=
by sorry

end NUMINAMATH_CALUDE_video_streaming_cost_l3499_349951


namespace NUMINAMATH_CALUDE_smallest_third_term_of_geometric_progression_l3499_349959

def is_arithmetic_progression (a b c : ℝ) : Prop :=
  b - a = c - b

def is_geometric_progression (a b c : ℝ) : Prop :=
  a ≠ 0 ∧ b / a = c / b

theorem smallest_third_term_of_geometric_progression :
  ∀ a b c : ℝ,
  a = 5 →
  is_arithmetic_progression a b c →
  is_geometric_progression 5 (b + 3) (c + 12) →
  ∃ d : ℝ, d ≥ 0 ∧ is_geometric_progression 5 (b + 3) d ∧
    ∀ e : ℝ, e ≥ 0 → is_geometric_progression 5 (b + 3) e → d ≤ e :=
by sorry

end NUMINAMATH_CALUDE_smallest_third_term_of_geometric_progression_l3499_349959


namespace NUMINAMATH_CALUDE_final_number_after_ten_steps_l3499_349998

/-- Performs one step of the sequence operation -/
def step (n : ℕ) (i : ℕ) : ℕ :=
  if i % 2 = 0 then n * 3 else n / 4

/-- Performs n steps of the sequence operation -/
def iterate_steps (start : ℕ) (n : ℕ) : ℕ :=
  match n with
  | 0 => start
  | k + 1 => step (iterate_steps start k) k

theorem final_number_after_ten_steps :
  iterate_steps 800000 10 = 1518750 := by
  sorry

end NUMINAMATH_CALUDE_final_number_after_ten_steps_l3499_349998


namespace NUMINAMATH_CALUDE_problem_solution_l3499_349991

theorem problem_solution : 
  (1/2 - 1/4 + 1/12) * (-12) = -4 ∧ 
  -(3^2) + (-5)^2 * (4/5) - |(-6)| = 5 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l3499_349991


namespace NUMINAMATH_CALUDE_tracy_art_fair_sales_l3499_349936

/-- The number of paintings sold at Tracy's art fair booth --/
def paintings_sold (group1_count group1_paintings group2_count group2_paintings group3_count group3_paintings : ℕ) : ℕ :=
  group1_count * group1_paintings + group2_count * group2_paintings + group3_count * group3_paintings

/-- Theorem stating the total number of paintings sold at Tracy's art fair booth --/
theorem tracy_art_fair_sales : paintings_sold 4 2 12 1 4 4 = 36 := by
  sorry

end NUMINAMATH_CALUDE_tracy_art_fair_sales_l3499_349936


namespace NUMINAMATH_CALUDE_petes_backward_speed_l3499_349978

/-- Pete's backward walking speed problem -/
theorem petes_backward_speed (petes_hand_speed tracy_cartwheel_speed susans_speed petes_backward_speed : ℝ) : 
  petes_hand_speed = 2 →
  petes_hand_speed = (1 / 4) * tracy_cartwheel_speed →
  tracy_cartwheel_speed = 2 * susans_speed →
  petes_backward_speed = 3 * susans_speed →
  petes_backward_speed = 12 := by
  sorry

end NUMINAMATH_CALUDE_petes_backward_speed_l3499_349978


namespace NUMINAMATH_CALUDE_equation_solution_l3499_349990

-- Define the equation
def satisfies_equation (x y : ℝ) : Prop :=
  2 * x^2 + 3 * x * y + y^2 + x = 1

-- Define the two lines
def line1 (x y : ℝ) : Prop := y = -x - 1
def line2 (x y : ℝ) : Prop := y = -2*x + 1

-- Theorem statement
theorem equation_solution (x y : ℝ) :
  satisfies_equation x y → line1 x y ∨ line2 x y :=
by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3499_349990


namespace NUMINAMATH_CALUDE_solve_linear_equation_l3499_349987

theorem solve_linear_equation (y : ℚ) (h : -3*y - 8 = 5*y + 4) : y = -3/2 := by
  sorry

end NUMINAMATH_CALUDE_solve_linear_equation_l3499_349987


namespace NUMINAMATH_CALUDE_order_of_logarithms_and_root_l3499_349900

theorem order_of_logarithms_and_root (a b c : ℝ) : 
  a = 2 * Real.log 0.99 →
  b = Real.log 0.98 →
  c = Real.sqrt 0.96 - 1 →
  a > b ∧ b > c := by
  sorry

end NUMINAMATH_CALUDE_order_of_logarithms_and_root_l3499_349900


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_l3499_349973

/-- If x^2 + mx + 16 is a perfect square trinomial, then m = ±8 -/
theorem perfect_square_trinomial (m : ℝ) : 
  (∀ x, ∃ k, x^2 + m*x + 16 = k^2) → m = 8 ∨ m = -8 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_l3499_349973


namespace NUMINAMATH_CALUDE_common_factor_proof_l3499_349907

def expression (x y : ℝ) : ℝ := 9 * x^3 * y^2 + 12 * x^2 * y^3

def common_factor (x y : ℝ) : ℝ := 3 * x^2 * y^2

theorem common_factor_proof :
  ∀ x y : ℝ, ∃ k : ℝ, expression x y = common_factor x y * k :=
by sorry

end NUMINAMATH_CALUDE_common_factor_proof_l3499_349907


namespace NUMINAMATH_CALUDE_square_side_length_l3499_349964

/-- Given a square and a rectangle with specific properties, prove that the side length of the square is 15 cm. -/
theorem square_side_length (s : ℝ) : 
  s > 0 →  -- side length is positive
  4 * s = 2 * (18 + 216 / 18) →  -- perimeters are equal
  s = 15 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l3499_349964


namespace NUMINAMATH_CALUDE_expenditure_ratio_l3499_349938

theorem expenditure_ratio (anand_income balu_income anand_expenditure balu_expenditure : ℚ) :
  anand_income / balu_income = 5 / 4 →
  anand_income = 2000 →
  anand_income - anand_expenditure = 800 →
  balu_income - balu_expenditure = 800 →
  anand_expenditure / balu_expenditure = 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_expenditure_ratio_l3499_349938


namespace NUMINAMATH_CALUDE_units_digit_of_7_pow_2050_l3499_349904

-- Define the function that returns the units digit of 7^n
def units_digit_of_7_pow (n : ℕ) : ℕ :=
  match n % 4 with
  | 0 => 1
  | 1 => 7
  | 2 => 9
  | 3 => 3
  | _ => 0  -- This case should never occur

theorem units_digit_of_7_pow_2050 :
  units_digit_of_7_pow 2050 = 9 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_7_pow_2050_l3499_349904


namespace NUMINAMATH_CALUDE_probability_girl_grade4_l3499_349962

/-- The probability of selecting a girl from grade 4 in a school playground -/
theorem probability_girl_grade4 (g3 b3 g4 b4 g5 b5 : ℕ) : 
  g3 = 28 → b3 = 35 → g4 = 45 → b4 = 42 → g5 = 38 → b5 = 51 →
  (g4 : ℚ) / (g3 + b3 + g4 + b4 + g5 + b5) = 45 / 239 := by
  sorry

end NUMINAMATH_CALUDE_probability_girl_grade4_l3499_349962


namespace NUMINAMATH_CALUDE_welders_left_l3499_349953

/-- Proves that 12 welders left the project given the initial conditions and remaining time. -/
theorem welders_left (initial_welders : ℕ) (initial_days : ℝ) (remaining_days : ℝ) : 
  initial_welders = 36 →
  initial_days = 3 →
  remaining_days = 3.0000000000000004 →
  (initial_welders - (initial_welders - remaining_days * initial_welders / (initial_days + remaining_days - 1))) = 12 := by
sorry

end NUMINAMATH_CALUDE_welders_left_l3499_349953


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_when_m_is_one_union_A_B_equals_A_iff_m_nonpositive_l3499_349974

-- Define set A
def A : Set ℝ := {x | x^2 + 5*x - 6 < 0}

-- Define set B as a function of m
def B (m : ℝ) : Set ℝ := {x | m - 2 < x ∧ x < 2*m + 1}

-- Part 1
theorem intersection_A_complement_B_when_m_is_one :
  A ∩ (Set.univ \ B 1) = {x | -6 < x ∧ x ≤ -1} := by sorry

-- Part 2
theorem union_A_B_equals_A_iff_m_nonpositive (m : ℝ) :
  A ∪ B m = A ↔ m ≤ 0 := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_when_m_is_one_union_A_B_equals_A_iff_m_nonpositive_l3499_349974


namespace NUMINAMATH_CALUDE_mean_temperature_l3499_349981

def temperatures : List ℚ := [79, 78, 82, 86, 88, 90, 88, 90, 89]

theorem mean_temperature : 
  (temperatures.sum / temperatures.length : ℚ) = 770 / 9 := by sorry

end NUMINAMATH_CALUDE_mean_temperature_l3499_349981


namespace NUMINAMATH_CALUDE_two_digit_number_digit_difference_l3499_349939

/-- Given a two-digit number where the difference between the original number
    and the number with interchanged digits is 45, prove that the difference
    between its two digits is 5. -/
theorem two_digit_number_digit_difference (x y : ℕ) : 
  x < 10 → y < 10 → (10 * x + y) - (10 * y + x) = 45 → x - y = 5 := by
sorry

end NUMINAMATH_CALUDE_two_digit_number_digit_difference_l3499_349939


namespace NUMINAMATH_CALUDE_equation_solution_l3499_349992

theorem equation_solution :
  ∃ x : ℚ, (5 + 3.5 * x = 2.1 * x - 25) ∧ (x = -150 / 7) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3499_349992


namespace NUMINAMATH_CALUDE_sequence_periodicity_l3499_349986

def is_periodic (a : ℕ → ℝ) (p : ℕ) : Prop :=
  ∃ k : ℕ, ∀ n ≥ k, a n = a (n + p)

def smallest_period (a : ℕ → ℝ) (p : ℕ) : Prop :=
  is_periodic a p ∧ ∀ q < p, ¬ is_periodic a q

theorem sequence_periodicity (a : ℕ → ℝ) 
  (h1 : ∃ n, a n ≠ 0)
  (h2 : ∀ n : ℕ, a (n + 2) = |a (n + 1)| - a n) :
  smallest_period a 9 :=
sorry

end NUMINAMATH_CALUDE_sequence_periodicity_l3499_349986


namespace NUMINAMATH_CALUDE_no_solution_iff_parallel_k_value_l3499_349965

/-- Two 2D vectors are parallel if one is a scalar multiple of the other -/
def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ c : ℝ, v = (c * w.1, c * w.2)

/-- The equation has no solution if and only if the direction vectors are parallel -/
theorem no_solution_iff_parallel (k : ℝ) :
  (∀ t s : ℝ, (1 + 5*t, 4 - 3*t) ≠ (0 - 2*s, 1 + k*s)) ↔
  parallel (5, -3) (-2, k) := by sorry

theorem k_value :
  (∀ t s : ℝ, (1 + 5*t, 4 - 3*t) ≠ (0 - 2*s, 1 + k*s)) ↔ k = 6/5 := by sorry

end NUMINAMATH_CALUDE_no_solution_iff_parallel_k_value_l3499_349965


namespace NUMINAMATH_CALUDE_f_properties_l3499_349949

noncomputable section

def f (x : ℝ) : ℝ := (2*x - x^2) * Real.exp x

theorem f_properties :
  (∀ x, f x > 0 ↔ 0 < x ∧ x < 2) ∧
  (∃ max_x, ∀ x, f x ≤ f max_x ∧ max_x = Real.sqrt 2) ∧
  (¬ ∃ min_x, ∀ x, f min_x ≤ f x) ∧
  (∃ max_x, ∀ x, f x ≤ f max_x) :=
sorry

end

end NUMINAMATH_CALUDE_f_properties_l3499_349949


namespace NUMINAMATH_CALUDE_geometric_series_sum_l3499_349961

theorem geometric_series_sum : ∀ (a r : ℝ) (n : ℕ),
  a = 2 → r = 3 → n = 6 →
  a * (r^n - 1) / (r - 1) = 728 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l3499_349961


namespace NUMINAMATH_CALUDE_total_cleaning_time_is_180_l3499_349975

/-- The total time Matt and Alex spend cleaning their cars -/
def total_cleaning_time (matt_outside : ℕ) : ℕ :=
  let matt_inside := matt_outside / 4
  let matt_total := matt_outside + matt_inside
  let alex_outside := matt_outside / 2
  let alex_inside := matt_inside * 2
  let alex_total := alex_outside + alex_inside
  matt_total + alex_total

/-- Theorem stating that the total cleaning time is 180 minutes -/
theorem total_cleaning_time_is_180 :
  total_cleaning_time 80 = 180 := by sorry

end NUMINAMATH_CALUDE_total_cleaning_time_is_180_l3499_349975


namespace NUMINAMATH_CALUDE_grazing_area_difference_l3499_349977

/-- Proves that the area difference between two circular grazing arrangements is 35π square feet -/
theorem grazing_area_difference (rope_length : ℝ) (tank_radius : ℝ) : 
  rope_length = 12 → tank_radius = 10 → 
  π * rope_length^2 - (3/4 * π * rope_length^2 + 1/4 * π * (rope_length - tank_radius)^2) = 35 * π := by
  sorry

end NUMINAMATH_CALUDE_grazing_area_difference_l3499_349977


namespace NUMINAMATH_CALUDE_second_train_speed_l3499_349943

/-- Given two trains starting from the same station, traveling in the same direction for 10 hours,
    with the first train moving at 10 mph and the distance between them after 10 hours being 250 miles,
    prove that the speed of the second train is 35 mph. -/
theorem second_train_speed (first_train_speed : ℝ) (time : ℝ) (distance_between : ℝ) :
  first_train_speed = 10 →
  time = 10 →
  distance_between = 250 →
  ∃ second_train_speed : ℝ,
    second_train_speed * time - first_train_speed * time = distance_between ∧
    second_train_speed = 35 := by
  sorry

end NUMINAMATH_CALUDE_second_train_speed_l3499_349943


namespace NUMINAMATH_CALUDE_bus_journey_l3499_349989

theorem bus_journey (total_distance : ℝ) (speed1 speed2 : ℝ) (total_time : ℝ) 
  (h1 : total_distance = 250)
  (h2 : speed1 = 40)
  (h3 : speed2 = 60)
  (h4 : total_time = 5.5)
  (h5 : ∀ x : ℝ, x / speed1 + (total_distance - x) / speed2 = total_time → x = 160) :
  ∃ x : ℝ, x / speed1 + (total_distance - x) / speed2 = total_time ∧ x = 160 := by
sorry

end NUMINAMATH_CALUDE_bus_journey_l3499_349989


namespace NUMINAMATH_CALUDE_quadratic_roots_condition_l3499_349929

theorem quadratic_roots_condition (a b c : ℝ) :
  (∃ x y : ℝ, x < 0 ∧ y > 0 ∧ a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0) ↔ a * c < 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_condition_l3499_349929


namespace NUMINAMATH_CALUDE_four_digit_sum_problem_l3499_349956

def is_valid_digit (d : ℕ) : Prop := 1 ≤ d ∧ d ≤ 9

def to_number (a b c d : ℕ) : ℕ := 1000 * a + 100 * b + 10 * c + d

theorem four_digit_sum_problem (a b c d : ℕ) :
  is_valid_digit a ∧ is_valid_digit b ∧ is_valid_digit c ∧ is_valid_digit d →
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  to_number a b c d + to_number d c b a = 11990 →
  (a = 1 ∧ b = 9 ∧ c = 9 ∧ d = 9) ∨ (a = 9 ∧ b = 9 ∧ c = 9 ∧ d = 1) :=
by sorry

end NUMINAMATH_CALUDE_four_digit_sum_problem_l3499_349956


namespace NUMINAMATH_CALUDE_system_solution_set_l3499_349921

theorem system_solution_set (x : ℝ) : 
  (x - 1 < 1 ∧ x + 3 > 0) ↔ (-3 < x ∧ x < 2) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_set_l3499_349921


namespace NUMINAMATH_CALUDE_abc_inequality_l3499_349917

theorem abc_inequality (a b c : ℝ) 
  (ha : 0 < a ∧ a < 1) 
  (hb : 0 < b ∧ b < 1) 
  (hc : 0 < c ∧ c < 1)
  (eq_a : a = 2022 * Real.exp (a - 2022))
  (eq_b : b = 2023 * Real.exp (b - 2023))
  (eq_c : c = 2024 * Real.exp (c - 2024)) :
  c < b ∧ b < a := by sorry

end NUMINAMATH_CALUDE_abc_inequality_l3499_349917


namespace NUMINAMATH_CALUDE_pams_apple_bags_l3499_349944

theorem pams_apple_bags (gerald_apples_per_bag : ℕ) (pam_total_apples : ℕ) : 
  gerald_apples_per_bag = 40 →
  pam_total_apples = 1200 →
  ∃ (pam_bags : ℕ), pam_bags * (3 * gerald_apples_per_bag) = pam_total_apples ∧ pam_bags = 10 :=
by sorry

end NUMINAMATH_CALUDE_pams_apple_bags_l3499_349944


namespace NUMINAMATH_CALUDE_valid_factorization_l3499_349983

theorem valid_factorization (x : ℝ) : x^2 - 9 = (x - 3) * (x + 3) := by
  sorry

#check valid_factorization

end NUMINAMATH_CALUDE_valid_factorization_l3499_349983


namespace NUMINAMATH_CALUDE_isosceles_triangle_count_l3499_349920

-- Define the geoboard as a square grid
structure Geoboard :=
  (size : ℕ)

-- Define a point on the geoboard
structure Point :=
  (x : ℕ)
  (y : ℕ)

-- Define a triangle on the geoboard
structure Triangle :=
  (A : Point)
  (B : Point)
  (C : Point)

-- Function to check if a triangle is isosceles
def isIsosceles (t : Triangle) : Prop :=
  (t.A.x - t.C.x)^2 + (t.A.y - t.C.y)^2 = (t.B.x - t.C.x)^2 + (t.B.y - t.C.y)^2 ∨
  (t.A.x - t.C.x)^2 + (t.A.y - t.C.y)^2 = (t.A.x - t.B.x)^2 + (t.A.y - t.B.y)^2 ∨
  (t.B.x - t.C.x)^2 + (t.B.y - t.C.y)^2 = (t.A.x - t.B.x)^2 + (t.A.y - t.B.y)^2

-- Theorem statement
theorem isosceles_triangle_count (g : Geoboard) (A B : Point) 
  (h1 : A.y = B.y) -- A and B are on the same horizontal line
  (h2 : B.x - A.x = 3) -- Distance between A and B is 3 units
  (h3 : A.x > 0 ∧ A.y > 0 ∧ B.x < g.size ∧ B.y < g.size) -- A and B are within the grid
  : ∃ (S : Finset Point), 
    (∀ C ∈ S, C ≠ A ∧ C ≠ B ∧ C.x > 0 ∧ C.y > 0 ∧ C.x ≤ g.size ∧ C.y ≤ g.size) ∧ 
    (∀ C ∈ S, isIsosceles ⟨A, B, C⟩) ∧
    S.card = 3 :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_count_l3499_349920


namespace NUMINAMATH_CALUDE_power_of_negative_one_2010_l3499_349976

theorem power_of_negative_one_2010 : ∃ x : ℕ, ((-1 : ℤ) ^ 2010 : ℤ) = x ∧ ∀ y : ℕ, y ≥ x :=
by sorry

end NUMINAMATH_CALUDE_power_of_negative_one_2010_l3499_349976


namespace NUMINAMATH_CALUDE_expression_evaluation_l3499_349969

theorem expression_evaluation : 2 - (-3) * 2 - 4 - (-5) * 2 - 6 = 8 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3499_349969


namespace NUMINAMATH_CALUDE_range_of_m_for_perpendicular_vectors_l3499_349934

/-- Given two points M(-1,0) and N(1,0), and a line 3x - 4y + m = 0,
    if there exists a point P on the line such that PM · PN = 0,
    then the range of values for m is [-5, 5]. -/
theorem range_of_m_for_perpendicular_vectors :
  let M : ℝ × ℝ := (-1, 0)
  let N : ℝ × ℝ := (1, 0)
  let line (m : ℝ) := {(x, y) : ℝ × ℝ | 3 * x - 4 * y + m = 0}
  ∀ m : ℝ, (∃ P ∈ line m, (P.1 - M.1) * (N.1 - P.1) + (P.2 - M.2) * (N.2 - P.2) = 0) ↔ 
    m ∈ Set.Icc (-5 : ℝ) 5 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_for_perpendicular_vectors_l3499_349934


namespace NUMINAMATH_CALUDE_age_ratio_l3499_349937

theorem age_ratio (a b : ℕ) : 
  (a - 4 = b + 4) → 
  ((a + 4) = 3 * (b - 4)) → 
  (a : ℚ) / b = 5 / 3 := by
sorry

end NUMINAMATH_CALUDE_age_ratio_l3499_349937


namespace NUMINAMATH_CALUDE_cape_may_less_than_double_daytona_main_result_l3499_349925

/-- The number of shark sightings in Cape May -/
def cape_may_sightings : ℕ := 24

/-- The number of shark sightings in Daytona Beach -/
def daytona_beach_sightings : ℕ := 40 - cape_may_sightings

/-- The total number of shark sightings in both locations -/
def total_sightings : ℕ := 40

/-- Cape May has some less than double the number of shark sightings of Daytona Beach -/
theorem cape_may_less_than_double_daytona : cape_may_sightings < 2 * daytona_beach_sightings :=
sorry

/-- The difference between double the number of shark sightings in Daytona Beach and Cape May -/
def sightings_difference : ℕ := 2 * daytona_beach_sightings - cape_may_sightings

theorem main_result : sightings_difference = 8 := by
  sorry

end NUMINAMATH_CALUDE_cape_may_less_than_double_daytona_main_result_l3499_349925


namespace NUMINAMATH_CALUDE_luke_trivia_rounds_l3499_349960

/-- Given that Luke scored a total of 300 points in a trivia game, 
    gained the same number of points each round, and scored 60 points per round, 
    prove that he played 5 rounds. -/
theorem luke_trivia_rounds (total_points : ℕ) (points_per_round : ℕ) (rounds : ℕ) : 
  total_points = 300 ∧ 
  points_per_round = 60 ∧ 
  total_points = points_per_round * rounds → 
  rounds = 5 := by
sorry

end NUMINAMATH_CALUDE_luke_trivia_rounds_l3499_349960


namespace NUMINAMATH_CALUDE_problem_solution_l3499_349912

theorem problem_solution (x y : ℝ) (h1 : y > x) (h2 : x > 0) (h3 : x / y + y / x = 8) :
  (x + 2 * y) / (x - 2 * y) = -4 / Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3499_349912


namespace NUMINAMATH_CALUDE_equilateral_triangle_line_theorem_l3499_349935

/-- Given an equilateral triangle ABC with side length a and a line A₁B₁ passing through its center O,
    cutting segments x and y from sides AC and BC respectively, prove that 3xy - 2a(x + y) + a² = 0 --/
theorem equilateral_triangle_line_theorem
  (a x y : ℝ)
  (h_positive : a > 0)
  (h_x_positive : x > 0)
  (h_y_positive : y > 0)
  (h_x_bound : x < a)
  (h_y_bound : y < a) :
  3 * x * y - 2 * a * (x + y) + a^2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_line_theorem_l3499_349935


namespace NUMINAMATH_CALUDE_twenty_five_percent_problem_l3499_349924

theorem twenty_five_percent_problem (x : ℚ) : x + (1/4) * x = 80 - (1/4) * 80 ↔ x = 48 := by
  sorry

end NUMINAMATH_CALUDE_twenty_five_percent_problem_l3499_349924


namespace NUMINAMATH_CALUDE_contrapositive_false_l3499_349940

theorem contrapositive_false : 
  ¬(∀ x : ℝ, x^2 - 1 = 0 → x = 1) :=
sorry

end NUMINAMATH_CALUDE_contrapositive_false_l3499_349940


namespace NUMINAMATH_CALUDE_three_f_value_l3499_349993

axiom f : ℝ → ℝ
axiom f_def : ∀ x > 0, f (3 * x) = 3 / (3 + x)

theorem three_f_value : ∀ x > 0, 3 * f x = 27 / (9 + x) := by sorry

end NUMINAMATH_CALUDE_three_f_value_l3499_349993


namespace NUMINAMATH_CALUDE_polynomial_expansion_l3499_349996

theorem polynomial_expansion (z : ℝ) :
  (3 * z^3 + 4 * z^2 - 7 * z + 1) * (2 * z^4 - 3 * z^2 + 2) =
  6 * z^7 + 8 * z^6 - 23 * z^5 - 10 * z^4 + 27 * z^3 + 5 * z^2 - 14 * z + 2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_expansion_l3499_349996


namespace NUMINAMATH_CALUDE_smallest_composite_with_large_factors_l3499_349952

def is_composite (n : ℕ) : Prop := ∃ a b, 1 < a ∧ 1 < b ∧ n = a * b

def has_no_small_prime_factors (n : ℕ) : Prop := ∀ p, p < 20 → ¬(Nat.Prime p ∧ p ∣ n)

theorem smallest_composite_with_large_factors :
  ∃ n : ℕ, is_composite n ∧
           has_no_small_prime_factors n ∧
           (∀ m, m < n → ¬(is_composite m ∧ has_no_small_prime_factors m)) ∧
           n = 529 ∧
           520 < n ∧ n ≤ 530 :=
sorry

end NUMINAMATH_CALUDE_smallest_composite_with_large_factors_l3499_349952


namespace NUMINAMATH_CALUDE_fifteenth_term_of_sequence_l3499_349999

def inverse_proportional_sequence (a : ℕ → ℝ) : Prop :=
  ∃ k : ℝ, k > 0 ∧ ∀ n : ℕ, n > 0 → a n * a (n + 1) = k

theorem fifteenth_term_of_sequence 
  (a : ℕ → ℝ)
  (h_inv_prop : inverse_proportional_sequence a)
  (h_first_term : a 1 = 3)
  (h_second_term : a 2 = 4) :
  a 15 = 3 := by
  sorry

end NUMINAMATH_CALUDE_fifteenth_term_of_sequence_l3499_349999


namespace NUMINAMATH_CALUDE_bisection_method_approximation_l3499_349927

-- Define the function f
variable (f : ℝ → ℝ)

-- State the theorem
theorem bisection_method_approximation 
  (h_continuous : Continuous f)
  (h1 : f 0.64 < 0)
  (h2 : f 0.72 > 0)
  (h3 : f 0.68 < 0) :
  ∃ x : ℝ, x ∈ (Set.Ioo 0.68 0.72) ∧ f x = 0 ∧ |x - 0.7| < 0.1 := by
  sorry


end NUMINAMATH_CALUDE_bisection_method_approximation_l3499_349927


namespace NUMINAMATH_CALUDE_divisibility_implies_fraction_simplification_l3499_349994

theorem divisibility_implies_fraction_simplification (a b c : ℕ) :
  (100 * a + 10 * b + c) % 7 = 0 →
  ((10 * b + c + 16 * a) % 7 = 0 ∧ (10 * b + c - 61 * a) % 7 = 0) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_implies_fraction_simplification_l3499_349994


namespace NUMINAMATH_CALUDE_initial_pills_count_l3499_349963

/-- The number of pills Tony takes in the first two days -/
def pills_first_two_days : ℕ := 2 * 3 * 2

/-- The number of pills Tony takes in the next three days -/
def pills_next_three_days : ℕ := 1 * 3 * 3

/-- The number of pills Tony takes on the sixth day -/
def pills_sixth_day : ℕ := 2

/-- The number of pills left in the bottle after Tony's treatment -/
def pills_left : ℕ := 27

/-- The total number of pills Tony took during his treatment -/
def total_pills_taken : ℕ := pills_first_two_days + pills_next_three_days + pills_sixth_day

/-- Theorem: The initial number of pills in the bottle is 50 -/
theorem initial_pills_count : total_pills_taken + pills_left = 50 := by
  sorry

end NUMINAMATH_CALUDE_initial_pills_count_l3499_349963


namespace NUMINAMATH_CALUDE_xy_equals_three_l3499_349985

theorem xy_equals_three (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hdistinct : x ≠ y)
  (h : x + 3 / x = y + 3 / y) : x * y = 3 := by
  sorry

end NUMINAMATH_CALUDE_xy_equals_three_l3499_349985


namespace NUMINAMATH_CALUDE_marble_probability_difference_l3499_349979

theorem marble_probability_difference :
  let total_marbles : ℕ := 4000
  let red_marbles : ℕ := 1500
  let black_marbles : ℕ := 2500
  let p_same : ℚ := (red_marbles.choose 2 + black_marbles.choose 2) / total_marbles.choose 2
  let p_different : ℚ := (red_marbles * black_marbles) / total_marbles.choose 2
  |p_same - p_different| = 3 / 50 := by
sorry

end NUMINAMATH_CALUDE_marble_probability_difference_l3499_349979


namespace NUMINAMATH_CALUDE_binomial_prob_two_to_four_out_of_five_l3499_349906

/-- The probability of getting 2, 3, or 4 successes in 5 trials with probability 0.5 each -/
theorem binomial_prob_two_to_four_out_of_five (n : Nat) (p : Real) (X : Nat → Real) :
  n = 5 →
  p = 0.5 →
  (∀ k, X k = Nat.choose n k * p^k * (1 - p)^(n - k)) →
  X 2 + X 3 + X 4 = 25/32 :=
by sorry

end NUMINAMATH_CALUDE_binomial_prob_two_to_four_out_of_five_l3499_349906


namespace NUMINAMATH_CALUDE_unique_valid_ticket_l3499_349947

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def is_multiple_of_5 (n : ℕ) : Prop := n % 5 = 0

def is_even (n : ℕ) : Prop := n % 2 = 0

def ticket_valid (n : ℕ) : Prop :=
  n ≥ 1000 ∧ n < 10000 ∧
  is_prime (n % 10) ∧
  is_multiple_of_5 ((n / 10) % 10) ∧
  is_even ((n / 100) % 10) ∧
  n / 1000 = 3 * (n % 10)

theorem unique_valid_ticket : ∀ n : ℕ, ticket_valid n ↔ n = 9853 :=
sorry

end NUMINAMATH_CALUDE_unique_valid_ticket_l3499_349947


namespace NUMINAMATH_CALUDE_obtuse_triangle_necessary_not_sufficient_l3499_349957

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively --/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  pos_a : 0 < a
  pos_b : 0 < b
  pos_c : 0 < c
  triangle_ineq : a + b > c ∧ b + c > a ∧ c + a > b

/-- Definition of an obtuse triangle --/
def isObtuse (t : Triangle) : Prop :=
  t.a^2 + t.b^2 < t.c^2 ∨ t.b^2 + t.c^2 < t.a^2 ∨ t.c^2 + t.a^2 < t.b^2

theorem obtuse_triangle_necessary_not_sufficient :
  (∀ t : Triangle, isObtuse t → (t.a^2 + t.b^2 < t.c^2 ∨ t.b^2 + t.c^2 < t.a^2 ∨ t.c^2 + t.a^2 < t.b^2)) ∧
  (∃ t : Triangle, (t.a^2 + t.b^2 < t.c^2 ∨ t.b^2 + t.c^2 < t.a^2 ∨ t.c^2 + t.a^2 < t.b^2) ∧ ¬isObtuse t) :=
by sorry

end NUMINAMATH_CALUDE_obtuse_triangle_necessary_not_sufficient_l3499_349957


namespace NUMINAMATH_CALUDE_algorithm_output_l3499_349926

theorem algorithm_output (x : ℤ) (y z : ℕ) : 
  x = -3 → 
  y = Int.natAbs x → 
  z = 2^y - y → 
  z = 5 := by sorry

end NUMINAMATH_CALUDE_algorithm_output_l3499_349926


namespace NUMINAMATH_CALUDE_perfect_fruits_count_perfect_fruits_theorem_l3499_349928

/-- Represents the fruit types in the batch -/
inductive FruitType
| Apple
| Orange
| Mango

/-- Represents the size of a fruit -/
inductive Size
| Small
| Medium
| Large

/-- Represents the ripeness stage of a fruit -/
inductive Ripeness
| Unripe
| PartlyRipe
| FullyRipe

/-- Defines the characteristics of the fruit batch -/
structure FruitBatch where
  totalFruits : ℕ
  apples : ℕ
  oranges : ℕ
  mangoes : ℕ
  appleSizeDistribution : Size → ℚ
  appleRipenessDistribution : Ripeness → ℚ
  orangeSizeDistribution : Size → ℚ
  orangeRipenessDistribution : Ripeness → ℚ
  mangoSizeDistribution : Size → ℚ
  mangoRipenessDistribution : Ripeness → ℚ

/-- Defines what makes a fruit perfect based on its type -/
def isPerfect (t : FruitType) (s : Size) (r : Ripeness) : Prop :=
  match t with
  | FruitType.Apple => (s = Size.Medium ∨ s = Size.Large) ∧ r = Ripeness.FullyRipe
  | FruitType.Orange => s = Size.Large ∧ r = Ripeness.FullyRipe
  | FruitType.Mango => (s = Size.Medium ∨ s = Size.Large) ∧ (r = Ripeness.PartlyRipe ∨ r = Ripeness.FullyRipe)

/-- The main theorem to prove -/
theorem perfect_fruits_count (batch : FruitBatch) : ℕ :=
  sorry

/-- The theorem statement -/
theorem perfect_fruits_theorem (batch : FruitBatch) :
  batch.totalFruits = 120 ∧
  batch.apples = 60 ∧
  batch.oranges = 40 ∧
  batch.mangoes = 20 ∧
  batch.appleSizeDistribution Size.Small = 1/4 ∧
  batch.appleSizeDistribution Size.Medium = 1/2 ∧
  batch.appleSizeDistribution Size.Large = 1/4 ∧
  batch.appleRipenessDistribution Ripeness.Unripe = 1/3 ∧
  batch.appleRipenessDistribution Ripeness.PartlyRipe = 1/6 ∧
  batch.appleRipenessDistribution Ripeness.FullyRipe = 1/2 ∧
  batch.orangeSizeDistribution Size.Small = 1/3 ∧
  batch.orangeSizeDistribution Size.Medium = 1/3 ∧
  batch.orangeSizeDistribution Size.Large = 1/3 ∧
  batch.orangeRipenessDistribution Ripeness.Unripe = 1/2 ∧
  batch.orangeRipenessDistribution Ripeness.PartlyRipe = 1/4 ∧
  batch.orangeRipenessDistribution Ripeness.FullyRipe = 1/4 ∧
  batch.mangoSizeDistribution Size.Small = 1/5 ∧
  batch.mangoSizeDistribution Size.Medium = 2/5 ∧
  batch.mangoSizeDistribution Size.Large = 2/5 ∧
  batch.mangoRipenessDistribution Ripeness.Unripe = 1/4 ∧
  batch.mangoRipenessDistribution Ripeness.PartlyRipe = 1/2 ∧
  batch.mangoRipenessDistribution Ripeness.FullyRipe = 1/4 →
  perfect_fruits_count batch = 55 := by
  sorry


end NUMINAMATH_CALUDE_perfect_fruits_count_perfect_fruits_theorem_l3499_349928


namespace NUMINAMATH_CALUDE_binomial_12_11_l3499_349997

theorem binomial_12_11 : Nat.choose 12 11 = 12 := by
  sorry

end NUMINAMATH_CALUDE_binomial_12_11_l3499_349997


namespace NUMINAMATH_CALUDE_frost_39_cupcakes_in_6_minutes_l3499_349995

/-- The number of cupcakes frosted by three people in a given time -/
def cupcakes_frosted (bob_rate cagney_rate lacey_rate time : ℚ) : ℚ :=
  (bob_rate + cagney_rate + lacey_rate) * time

/-- Theorem stating that Bob, Cagney, and Lacey can frost 39 cupcakes in 6 minutes -/
theorem frost_39_cupcakes_in_6_minutes :
  cupcakes_frosted (1/40) (1/20) (1/30) 360 = 39 := by
  sorry

end NUMINAMATH_CALUDE_frost_39_cupcakes_in_6_minutes_l3499_349995


namespace NUMINAMATH_CALUDE_probability_two_girls_l3499_349930

/-- The probability of choosing two girls from a class with given composition -/
theorem probability_two_girls (total : ℕ) (girls : ℕ) (boys : ℕ) 
  (h1 : total = girls + boys) 
  (h2 : total = 8) 
  (h3 : girls = 5) 
  (h4 : boys = 3) : 
  (Nat.choose girls 2 : ℚ) / (Nat.choose total 2) = 5 / 14 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_girls_l3499_349930


namespace NUMINAMATH_CALUDE_column_arrangement_l3499_349958

theorem column_arrangement (total_people : ℕ) 
  (h1 : total_people = 30 * 16) 
  (h2 : ∃ (people_per_column : ℕ), total_people = people_per_column * 10) : 
  ∃ (people_per_column : ℕ), total_people = people_per_column * 10 ∧ people_per_column = 48 :=
by sorry

end NUMINAMATH_CALUDE_column_arrangement_l3499_349958


namespace NUMINAMATH_CALUDE_equation_solution_l3499_349972

theorem equation_solution (a b : ℕ+) :
  2 * a ^ 2 = 3 * b ^ 3 ↔ ∃ k : ℕ+, a = 18 * k ^ 3 ∧ b = 6 * k ^ 2 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l3499_349972


namespace NUMINAMATH_CALUDE_female_democrats_count_l3499_349915

theorem female_democrats_count (total : ℕ) (male female : ℕ) (male_democrats female_democrats : ℕ) :
  total = 870 →
  male + female = total →
  female_democrats = female / 2 →
  male_democrats = male / 4 →
  female_democrats + male_democrats = total / 3 →
  female_democrats = 145 := by
  sorry

end NUMINAMATH_CALUDE_female_democrats_count_l3499_349915


namespace NUMINAMATH_CALUDE_dogsled_race_distance_l3499_349908

/-- The distance of the dogsled race course -/
def distance : ℝ := sorry

/-- The time taken by Team W to complete the course -/
def time_W : ℝ := sorry

/-- The time taken by Team A to complete the course -/
def time_A : ℝ := sorry

/-- The average speed of Team W -/
def speed_W : ℝ := 20

/-- The average speed of Team A -/
def speed_A : ℝ := speed_W + 5

theorem dogsled_race_distance :
  (time_A = time_W - 3) →
  (distance = speed_W * time_W) →
  (distance = speed_A * time_A) →
  distance = 300 := by sorry

end NUMINAMATH_CALUDE_dogsled_race_distance_l3499_349908


namespace NUMINAMATH_CALUDE_walking_biking_time_difference_l3499_349970

/-- Proves that the difference between walking and biking time is 4 minutes -/
theorem walking_biking_time_difference :
  let blocks : ℕ := 6
  let walk_time_per_block : ℚ := 1
  let bike_time_per_block : ℚ := 20 / 60
  (blocks * walk_time_per_block) - (blocks * bike_time_per_block) = 4 := by
  sorry

end NUMINAMATH_CALUDE_walking_biking_time_difference_l3499_349970


namespace NUMINAMATH_CALUDE_smallest_divisible_by_10_13_14_l3499_349982

theorem smallest_divisible_by_10_13_14 : ∃ (n : ℕ), n > 0 ∧ 
  10 ∣ n ∧ 13 ∣ n ∧ 14 ∣ n ∧ 
  ∀ (m : ℕ), m > 0 → 10 ∣ m → 13 ∣ m → 14 ∣ m → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_10_13_14_l3499_349982


namespace NUMINAMATH_CALUDE_max_intersection_area_l3499_349918

/-- Represents a rectangle with integer dimensions -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- The area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℕ := r.width * r.height

theorem max_intersection_area :
  ∀ (r1 r2 : Rectangle),
    r1.height < r1.width →
    r2.height > r2.width →
    r1.area = 2015 →
    r2.area = 2016 →
    (∀ r : Rectangle,
      r.width ≤ min r1.width r2.width ∧
      r.height ≤ min r1.height r2.height →
      r.area ≤ 1302) ∧
    (∃ r : Rectangle,
      r.width ≤ min r1.width r2.width ∧
      r.height ≤ min r1.height r2.height ∧
      r.area = 1302) := by
  sorry

end NUMINAMATH_CALUDE_max_intersection_area_l3499_349918


namespace NUMINAMATH_CALUDE_points_on_same_line_l3499_349984

def point_A : ℝ × ℝ := (-1, 0.5)
def point_B : ℝ × ℝ := (3, -3.5)
def point_C : ℝ × ℝ := (7, -7.5)

def collinear (p q r : ℝ × ℝ) : Prop :=
  let (x₁, y₁) := p
  let (x₂, y₂) := q
  let (x₃, y₃) := r
  (y₂ - y₁) * (x₃ - x₁) = (y₃ - y₁) * (x₂ - x₁)

theorem points_on_same_line : collinear point_A point_B point_C := by
  sorry

end NUMINAMATH_CALUDE_points_on_same_line_l3499_349984
