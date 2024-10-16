import Mathlib

namespace NUMINAMATH_CALUDE_sprint_jog_difference_l802_80228

-- Define the distances
def sprint_distance : ℚ := 875 / 1000
def jog_distance : ℚ := 75 / 100

-- Theorem statement
theorem sprint_jog_difference :
  sprint_distance - jog_distance = 125 / 1000 := by
  sorry

end NUMINAMATH_CALUDE_sprint_jog_difference_l802_80228


namespace NUMINAMATH_CALUDE_solution_set_for_a_equals_neg_three_range_of_a_for_interval_condition_l802_80265

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x + a| + |x - 2|

-- Part 1
theorem solution_set_for_a_equals_neg_three :
  {x : ℝ | f (-3) x ≥ 3} = {x : ℝ | x ≤ 1 ∨ x ≥ 4} := by sorry

-- Part 2
theorem range_of_a_for_interval_condition :
  ∀ a : ℝ, (∀ x ∈ Set.Icc 1 2, f a x ≤ |x - 4|) → a ∈ Set.Icc (-3) 0 := by sorry

end NUMINAMATH_CALUDE_solution_set_for_a_equals_neg_three_range_of_a_for_interval_condition_l802_80265


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_l802_80251

theorem sum_of_roots_quadratic (m n : ℝ) : 
  (m^2 - 4*m - 2 = 0) → (n^2 - 4*n - 2 = 0) → m + n = 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_l802_80251


namespace NUMINAMATH_CALUDE_shorter_base_length_l802_80275

/-- A trapezoid with given properties -/
structure Trapezoid where
  long_base : ℝ
  short_base : ℝ
  midpoint_segment : ℝ

/-- The trapezoid satisfies the given conditions -/
def satisfies_conditions (t : Trapezoid) : Prop :=
  t.long_base = 85 ∧ t.midpoint_segment = 5

/-- Theorem: In a trapezoid satisfying the given conditions, the shorter base is 75 -/
theorem shorter_base_length (t : Trapezoid) (h : satisfies_conditions t) : 
  t.short_base = 75 := by
  sorry

end NUMINAMATH_CALUDE_shorter_base_length_l802_80275


namespace NUMINAMATH_CALUDE_fraction_closest_to_longest_side_specific_trapezoid_l802_80210

/-- Represents a trapezoid field -/
structure TrapezoidField where
  base1 : ℝ
  base2 : ℝ
  angle1 : ℝ
  angle2 : ℝ

/-- The fraction of area closer to the longest side of the trapezoid field -/
def fraction_closest_to_longest_side (field : TrapezoidField) : ℝ :=
  sorry

/-- Theorem stating the fraction of area closest to the longest side for the given trapezoid -/
theorem fraction_closest_to_longest_side_specific_trapezoid :
  let field : TrapezoidField := {
    base1 := 200,
    base2 := 100,
    angle1 := 45,
    angle2 := 135
  }
  fraction_closest_to_longest_side field = 7/12 := by
  sorry

end NUMINAMATH_CALUDE_fraction_closest_to_longest_side_specific_trapezoid_l802_80210


namespace NUMINAMATH_CALUDE_sum_base4_equals_l802_80212

/-- Converts a base 4 number represented as a list of digits to its decimal equivalent -/
def base4ToDecimal (digits : List Nat) : Nat :=
  digits.foldl (fun acc d => 4 * acc + d) 0

/-- Converts a decimal number to its base 4 representation as a list of digits -/
def decimalToBase4 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc else aux (m / 4) ((m % 4) :: acc)
    aux n []

theorem sum_base4_equals : 
  let a := base4ToDecimal [2, 0, 1]
  let b := base4ToDecimal [1, 3, 2]
  let c := base4ToDecimal [3, 0, 3]
  let d := base4ToDecimal [2, 2, 1]
  decimalToBase4 (a + b + c + d) = [0, 1, 1, 0, 1] := by
  sorry

end NUMINAMATH_CALUDE_sum_base4_equals_l802_80212


namespace NUMINAMATH_CALUDE_distinct_paintings_l802_80249

/-- The number of disks --/
def n : ℕ := 7

/-- The number of blue disks --/
def blue : ℕ := 4

/-- The number of red disks --/
def red : ℕ := 2

/-- The number of green disks --/
def green : ℕ := 1

/-- The number of symmetry operations (identity and reflection) --/
def symmetries : ℕ := 2

/-- The total number of colorings --/
def total_colorings : ℕ := (Nat.choose n blue) * (Nat.choose (n - blue) red) * (Nat.choose (n - blue - red) green)

/-- The number of colorings fixed by reflection --/
def fixed_colorings : ℕ := 3

/-- The theorem stating the number of distinct paintings --/
theorem distinct_paintings : (total_colorings + fixed_colorings) / symmetries = 54 := by
  sorry

end NUMINAMATH_CALUDE_distinct_paintings_l802_80249


namespace NUMINAMATH_CALUDE_simplify_2A_minus_3B_value_2A_minus_3B_specific_value_2A_minus_3B_independent_l802_80226

-- Define A and B as functions of x and y
def A (x y : ℝ) : ℝ := 3 * x^2 - x + 2 * y - 4 * x * y
def B (x y : ℝ) : ℝ := 2 * x^2 - 3 * x - y + x * y

-- Theorem 1: Simplification of 2A - 3B
theorem simplify_2A_minus_3B (x y : ℝ) :
  2 * A x y - 3 * B x y = 7 * x + 7 * y - 11 * x * y :=
by sorry

-- Theorem 2: Value of 2A - 3B under specific conditions
theorem value_2A_minus_3B_specific (x y : ℝ) 
  (h1 : x + y = 6/7) (h2 : x * y = -1) :
  2 * A x y - 3 * B x y = 17 :=
by sorry

-- Theorem 3: Value of 2A - 3B when independent of y
theorem value_2A_minus_3B_independent (x : ℝ) 
  (h : ∀ y : ℝ, 2 * A x y - 3 * B x y = 2 * A x 0 - 3 * B x 0) :
  2 * A x 0 - 3 * B x 0 = 49/11 :=
by sorry

end NUMINAMATH_CALUDE_simplify_2A_minus_3B_value_2A_minus_3B_specific_value_2A_minus_3B_independent_l802_80226


namespace NUMINAMATH_CALUDE_initial_distance_problem_l802_80267

theorem initial_distance_problem (speed_A speed_B : ℝ) (start_time end_time : ℝ) :
  speed_A = 5 →
  speed_B = 7 →
  start_time = 1 →
  end_time = 3 →
  let time_walked := end_time - start_time
  let distance_A := speed_A * time_walked
  let distance_B := speed_B * time_walked
  let initial_distance := distance_A + distance_B
  initial_distance = 24 := by
  sorry

end NUMINAMATH_CALUDE_initial_distance_problem_l802_80267


namespace NUMINAMATH_CALUDE_max_value_fraction_l802_80259

theorem max_value_fraction (x y : ℝ) (hx : -5 ≤ x ∧ x ≤ -1) (hy : 1 ≤ y ∧ y ≤ 3) :
  (∀ x' y', -5 ≤ x' ∧ x' ≤ -1 ∧ 1 ≤ y' ∧ y' ≤ 3 → (x' + y') / x' ≤ (x + y) / x) →
  (x + y) / x = -2 :=
sorry

end NUMINAMATH_CALUDE_max_value_fraction_l802_80259


namespace NUMINAMATH_CALUDE_total_digits_memorized_l802_80281

/-- The number of digits of pi memorized by each person --/
structure PiDigits where
  carlos : ℕ
  sam : ℕ
  mina : ℕ
  nina : ℕ

/-- The conditions given in the problem --/
def satisfies_conditions (p : PiDigits) : Prop :=
  p.sam = p.carlos + 6 ∧
  p.mina = 6 * p.carlos ∧
  p.nina = 4 * p.carlos ∧
  p.mina = 24

/-- The theorem to be proved --/
theorem total_digits_memorized (p : PiDigits) 
  (h : satisfies_conditions p) : 
  p.sam + p.carlos + p.mina + p.nina = 54 := by
  sorry


end NUMINAMATH_CALUDE_total_digits_memorized_l802_80281


namespace NUMINAMATH_CALUDE_science_book_pages_l802_80215

/-- Given information about the number of pages in different books -/
structure BookPages where
  history : ℕ
  novel : ℕ
  science : ℕ
  novel_half_of_history : novel = history / 2
  science_four_times_novel : science = 4 * novel
  history_pages : history = 300

/-- Theorem stating that the science book has 600 pages -/
theorem science_book_pages (b : BookPages) : b.science = 600 := by
  sorry

end NUMINAMATH_CALUDE_science_book_pages_l802_80215


namespace NUMINAMATH_CALUDE_li_family_cinema_cost_l802_80296

def adult_ticket_price : ℝ := 10
def child_discount : ℝ := 0.4
def senior_discount : ℝ := 0.3
def handling_fee : ℝ := 5
def num_adults : ℕ := 2
def num_children : ℕ := 1
def num_seniors : ℕ := 1

def total_cost : ℝ :=
  (num_adults * adult_ticket_price) +
  (num_children * adult_ticket_price * (1 - child_discount)) +
  (num_seniors * adult_ticket_price * (1 - senior_discount)) +
  handling_fee

theorem li_family_cinema_cost : total_cost = 38 := by
  sorry

end NUMINAMATH_CALUDE_li_family_cinema_cost_l802_80296


namespace NUMINAMATH_CALUDE_revenue_change_l802_80236

theorem revenue_change
  (T : ℝ) -- original tax rate (as a percentage)
  (C : ℝ) -- original consumption
  (h1 : T > 0)
  (h2 : C > 0) :
  let new_tax_rate := T * (1 - 0.16)
  let new_consumption := C * (1 + 0.15)
  let original_revenue := (T / 100) * C
  let new_revenue := (new_tax_rate / 100) * new_consumption
  (new_revenue - original_revenue) / original_revenue = -0.034 := by
sorry

end NUMINAMATH_CALUDE_revenue_change_l802_80236


namespace NUMINAMATH_CALUDE_a8_min_value_l802_80206

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  is_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_property : a 3 + a 6 = a 4 + 5
  a2_bound : a 2 ≤ 1

/-- The minimum value of the 8th term in the arithmetic sequence is 9 -/
theorem a8_min_value (seq : ArithmeticSequence) : seq.a 8 ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_a8_min_value_l802_80206


namespace NUMINAMATH_CALUDE_max_volume_side_length_l802_80218

def sheet_length : ℝ := 90
def sheet_width : ℝ := 48

def container_volume (x : ℝ) : ℝ :=
  (sheet_length - 2 * x) * (sheet_width - 2 * x) * x

theorem max_volume_side_length :
  ∃ (x : ℝ), x > 0 ∧ x < sheet_width / 2 ∧ x < sheet_length / 2 ∧
  ∀ (y : ℝ), y > 0 → y < sheet_width / 2 → y < sheet_length / 2 →
  container_volume y ≤ container_volume x ∧
  x = 10 :=
sorry

end NUMINAMATH_CALUDE_max_volume_side_length_l802_80218


namespace NUMINAMATH_CALUDE_fred_card_spending_l802_80276

-- Define the costs of each type of card
def football_pack_cost : ℝ := 2.73
def pokemon_pack_cost : ℝ := 4.01
def baseball_deck_cost : ℝ := 8.95

-- Define the number of packs/decks bought
def football_packs : ℕ := 2
def pokemon_packs : ℕ := 1
def baseball_decks : ℕ := 1

-- Define the total cost function
def total_cost : ℝ := 
  (football_pack_cost * football_packs) + 
  (pokemon_pack_cost * pokemon_packs) + 
  (baseball_deck_cost * baseball_decks)

-- Theorem statement
theorem fred_card_spending : total_cost = 18.42 := by
  sorry

end NUMINAMATH_CALUDE_fred_card_spending_l802_80276


namespace NUMINAMATH_CALUDE_green_to_red_ratio_is_three_to_one_l802_80214

/-- Represents the contents of a bag of mints -/
structure MintBag where
  green : ℕ
  red : ℕ

/-- The ratio of green mints to red mints -/
def mintRatio (bag : MintBag) : ℚ :=
  bag.green / bag.red

theorem green_to_red_ratio_is_three_to_one 
  (bag : MintBag) 
  (h_total : bag.green + bag.red > 0)
  (h_green_percent : (bag.green : ℚ) / (bag.green + bag.red) = 3/4) :
  mintRatio bag = 3/1 := by
sorry

end NUMINAMATH_CALUDE_green_to_red_ratio_is_three_to_one_l802_80214


namespace NUMINAMATH_CALUDE_students_not_enrolled_l802_80268

theorem students_not_enrolled (total : ℕ) (french : ℕ) (german : ℕ) (both : ℕ) 
  (h1 : total = 79)
  (h2 : french = 41)
  (h3 : german = 22)
  (h4 : both = 9) :
  total - (french + german - both) = 25 := by
  sorry

end NUMINAMATH_CALUDE_students_not_enrolled_l802_80268


namespace NUMINAMATH_CALUDE_probability_two_females_l802_80293

/-- The probability of selecting two females from a group of contestants -/
theorem probability_two_females (total : ℕ) (females : ℕ) (males : ℕ) :
  total = females + males →
  females = 5 →
  males = 3 →
  (Nat.choose females 2 : ℚ) / (Nat.choose total 2 : ℚ) = 5 / 14 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_females_l802_80293


namespace NUMINAMATH_CALUDE_exists_function_satisfying_condition_l802_80225

theorem exists_function_satisfying_condition : ∃ f : ℕ → ℕ, 
  (∀ m n : ℕ, f (m + n) ≥ f m + f (f n) - 1) ∧ 
  f 2019 = 2019 := by
  sorry

end NUMINAMATH_CALUDE_exists_function_satisfying_condition_l802_80225


namespace NUMINAMATH_CALUDE_equation_solution_l802_80294

theorem equation_solution (x : ℝ) (hx : x ≠ 0) :
  2 - 9 / x + 9 / x^2 = 0 → 2 / x = 2 / 3 ∨ 2 / x = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l802_80294


namespace NUMINAMATH_CALUDE_parallel_lines_a_value_l802_80244

/-- Two lines in the form Ax + By + C = 0 are parallel if and only if their slopes (-A/B) are equal -/
def parallel (A1 B1 C1 A2 B2 C2 : ℝ) : Prop :=
  B1 ≠ 0 ∧ B2 ≠ 0 ∧ A1 / B1 = A2 / B2

/-- Two lines in the form Ax + By + C = 0 are identical if and only if their coefficients are proportional -/
def identical (A1 B1 C1 A2 B2 C2 : ℝ) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ A1 = k * A2 ∧ B1 = k * B2 ∧ C1 = k * C2

theorem parallel_lines_a_value : 
  ∃! a : ℝ, parallel (a + 1) 3 3 1 (a - 1) 1 ∧ ¬identical (a + 1) 3 3 1 (a - 1) 1 ∧ a = -2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_a_value_l802_80244


namespace NUMINAMATH_CALUDE_three_circles_inscribed_l802_80211

theorem three_circles_inscribed (R : ℝ) (r : ℝ) : R = 9 → R = r * (1 + Real.sqrt 3) → r = (9 * (Real.sqrt 3 - 1)) / 2 := by
  sorry

end NUMINAMATH_CALUDE_three_circles_inscribed_l802_80211


namespace NUMINAMATH_CALUDE_polynomial_expansion_l802_80264

theorem polynomial_expansion (x : ℝ) : (2 - x^4) * (3 + x^5) = -x^9 - 3*x^4 + 2*x^5 + 6 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_expansion_l802_80264


namespace NUMINAMATH_CALUDE_comparison_of_fractions_l802_80222

theorem comparison_of_fractions :
  (1/2 : ℚ) < (2/2 : ℚ) →
  (1 - 5/6 : ℚ) > (1 - 7/6 : ℚ) →
  (-π : ℝ) < -3.14 →
  (-2/3 : ℚ) > (-4/5 : ℚ) :=
by
  sorry

end NUMINAMATH_CALUDE_comparison_of_fractions_l802_80222


namespace NUMINAMATH_CALUDE_tom_allowance_l802_80220

theorem tom_allowance (initial_allowance : ℝ) 
  (first_week_fraction : ℝ) (second_week_fraction : ℝ) : 
  initial_allowance = 12 →
  first_week_fraction = 1/3 →
  second_week_fraction = 1/4 →
  let remaining_after_first_week := initial_allowance - (initial_allowance * first_week_fraction)
  let final_remaining := remaining_after_first_week - (remaining_after_first_week * second_week_fraction)
  final_remaining = 6 := by
sorry

end NUMINAMATH_CALUDE_tom_allowance_l802_80220


namespace NUMINAMATH_CALUDE_seeds_in_second_plot_is_200_l802_80286

/-- The number of seeds planted in the second plot -/
def seeds_in_second_plot : ℕ := 200

/-- The number of seeds planted in the first plot -/
def seeds_in_first_plot : ℕ := 500

/-- The germination rate of seeds in the first plot -/
def germination_rate_first : ℚ := 30 / 100

/-- The germination rate of seeds in the second plot -/
def germination_rate_second : ℚ := 50 / 100

/-- The total germination rate of all seeds -/
def total_germination_rate : ℚ := 35714285714285715 / 100000000000000000

theorem seeds_in_second_plot_is_200 : 
  (germination_rate_first * seeds_in_first_plot + 
   germination_rate_second * seeds_in_second_plot) / 
  (seeds_in_first_plot + seeds_in_second_plot) = total_germination_rate :=
sorry

end NUMINAMATH_CALUDE_seeds_in_second_plot_is_200_l802_80286


namespace NUMINAMATH_CALUDE_percent_of_a_is_4b_l802_80262

theorem percent_of_a_is_4b (a b : ℝ) (h : a = 1.2 * b) :
  (4 * b) / a * 100 = 333.33 := by
  sorry

end NUMINAMATH_CALUDE_percent_of_a_is_4b_l802_80262


namespace NUMINAMATH_CALUDE_trajectory_of_midpoint_l802_80298

/-- The trajectory of point M given point P on a curve and M as the midpoint of OP -/
theorem trajectory_of_midpoint (x y x₀ y₀ : ℝ) : 
  (2 * x^2 - y^2 = 1) →  -- P is on the curve
  (x₀ = x / 2) →         -- M is the midpoint of OP (x-coordinate)
  (y₀ = y / 2) →         -- M is the midpoint of OP (y-coordinate)
  (8 * x₀^2 - 4 * y₀^2 = 1) := by
sorry

end NUMINAMATH_CALUDE_trajectory_of_midpoint_l802_80298


namespace NUMINAMATH_CALUDE_quadratic_polynomial_unique_l802_80230

theorem quadratic_polynomial_unique (q : ℝ → ℝ) :
  (q = λ x => (67/30) * x^2 - (39/10) * x - 2/15) ↔
  (q (-1) = 6 ∧ q 2 = 1 ∧ q 4 = 20) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_unique_l802_80230


namespace NUMINAMATH_CALUDE_city_population_increase_l802_80243

/-- Represents the net population increase in a city over one day -/
def net_population_increase (birth_rate : ℕ) (death_rate : ℕ) (time_interval : ℕ) (seconds_per_day : ℕ) : ℕ :=
  let net_rate_per_interval := birth_rate - death_rate
  let net_rate_per_second := net_rate_per_interval / time_interval
  net_rate_per_second * seconds_per_day

/-- Theorem stating the net population increase in a day given specific birth and death rates -/
theorem city_population_increase : 
  net_population_increase 6 2 2 86400 = 172800 := by
  sorry

#eval net_population_increase 6 2 2 86400

end NUMINAMATH_CALUDE_city_population_increase_l802_80243


namespace NUMINAMATH_CALUDE_fraction_simplification_l802_80269

theorem fraction_simplification : (1/4 + 1/6) / (3/8 - 1/3) = 10 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l802_80269


namespace NUMINAMATH_CALUDE_closer_to_d_probability_l802_80233

/-- Triangle DEF with side lengths -/
structure Triangle (DE EF FD : ℝ) where
  side_positive : 0 < DE ∧ 0 < EF ∧ 0 < FD
  triangle_inequality : DE + EF > FD ∧ EF + FD > DE ∧ FD + DE > EF

/-- The region closer to D than to E or F -/
def CloserToD (t : Triangle DE EF FD) : Set (ℝ × ℝ) := sorry

theorem closer_to_d_probability (t : Triangle 8 6 10) : 
  MeasureTheory.volume (CloserToD t) = (1/4) * MeasureTheory.volume (Set.univ : Set (ℝ × ℝ)) := by
  sorry

end NUMINAMATH_CALUDE_closer_to_d_probability_l802_80233


namespace NUMINAMATH_CALUDE_zoo_visitors_l802_80271

def num_cars : ℕ := 3
def people_per_car : ℕ := 21

theorem zoo_visitors : num_cars * people_per_car = 63 := by
  sorry

end NUMINAMATH_CALUDE_zoo_visitors_l802_80271


namespace NUMINAMATH_CALUDE_square_perimeter_with_area_9_l802_80255

theorem square_perimeter_with_area_9 (s : ℝ) (h1 : s^2 = 9) (h2 : ∃ k : ℕ, 4 * s = 4 * k) : 4 * s = 12 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_with_area_9_l802_80255


namespace NUMINAMATH_CALUDE_median_length_triangle_l802_80257

/-- Given a triangle ABC with sides CB = 7, AC = 8, and AB = 9, 
    the length of the median to side AC is 7. -/
theorem median_length_triangle (A B C : ℝ × ℝ) : 
  let d (p q : ℝ × ℝ) := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  d B C = 7 ∧ d A C = 8 ∧ d A B = 9 →
  let D := ((A.1 + C.1) / 2, (A.2 + C.2) / 2)  -- midpoint of AC
  d B D = 7 := by
sorry

end NUMINAMATH_CALUDE_median_length_triangle_l802_80257


namespace NUMINAMATH_CALUDE_division_of_powers_l802_80263

theorem division_of_powers (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (a * b^2) / (a * b) = b :=
by sorry

end NUMINAMATH_CALUDE_division_of_powers_l802_80263


namespace NUMINAMATH_CALUDE_probability_of_letter_in_mathematics_l802_80239

def alphabet : Finset Char := sorry

def mathematics : String := "MATHEMATICS"

def uniqueLetters (s : String) : Finset Char :=
  s.toList.toFinset

theorem probability_of_letter_in_mathematics :
  (uniqueLetters mathematics).card / alphabet.card = 4 / 13 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_letter_in_mathematics_l802_80239


namespace NUMINAMATH_CALUDE_not_all_vertices_lattice_points_l802_80256

/-- A polygon with 1994 sides where the length of the k-th side is √(4 + k^2) -/
structure Polygon1994 where
  vertices : Fin 1994 → ℤ × ℤ
  side_length : ∀ k : Fin 1994, Real.sqrt (4 + k.val ^ 2) = 
    Real.sqrt ((vertices (k + 1)).1 - (vertices k).1) ^ 2 + ((vertices (k + 1)).2 - (vertices k).2) ^ 2

/-- Theorem stating that it's impossible for all vertices of the polygon to be lattice points -/
theorem not_all_vertices_lattice_points (p : Polygon1994) : False := by
  sorry

end NUMINAMATH_CALUDE_not_all_vertices_lattice_points_l802_80256


namespace NUMINAMATH_CALUDE_highway_distance_l802_80278

/-- Proves the distance a car can travel on the highway given its city fuel efficiency and efficiency increase -/
theorem highway_distance (city_efficiency : ℝ) (efficiency_increase : ℝ) (highway_gas : ℝ) :
  city_efficiency = 30 →
  efficiency_increase = 0.2 →
  highway_gas = 7 →
  (city_efficiency * (1 + efficiency_increase)) * highway_gas = 252 := by
  sorry

end NUMINAMATH_CALUDE_highway_distance_l802_80278


namespace NUMINAMATH_CALUDE_parabola_equation_l802_80202

theorem parabola_equation (p : ℝ) (x₀ y₀ : ℝ) : 
  p > 0 → 
  y₀^2 = 2*p*x₀ → 
  (x₀ + p/2)^2 + y₀^2 = 100 → 
  y₀^2 = 36 → 
  (y^2 = 4*x ∨ y^2 = 36*x) := by
  sorry

end NUMINAMATH_CALUDE_parabola_equation_l802_80202


namespace NUMINAMATH_CALUDE_school_time_problem_l802_80266

/-- Given a boy who reaches school 3 minutes early when walking at 7/6 of his usual rate,
    his usual time to reach school is 21 minutes. -/
theorem school_time_problem (usual_rate : ℝ) (usual_time : ℝ) 
  (h1 : usual_rate > 0)
  (h2 : usual_time > 0)
  (h3 : usual_rate * usual_time = (7/6 * usual_rate) * (usual_time - 3)) :
  usual_time = 21 := by
  sorry

end NUMINAMATH_CALUDE_school_time_problem_l802_80266


namespace NUMINAMATH_CALUDE_sin_plus_cos_special_angle_l802_80240

/-- Given a point P(-3,4) on the terminal side of angle α, prove that sin α + cos α = 1/5 -/
theorem sin_plus_cos_special_angle (α : Real) :
  let P : ℝ × ℝ := (-3, 4)
  (∃ t : ℝ, t > 0 ∧ P.1 = t * Real.cos α ∧ P.2 = t * Real.sin α) →
  Real.sin α + Real.cos α = 1/5 := by
sorry

end NUMINAMATH_CALUDE_sin_plus_cos_special_angle_l802_80240


namespace NUMINAMATH_CALUDE_notebook_reorganization_theorem_l802_80248

/-- Represents the notebook reorganization problem --/
structure NotebookProblem where
  initial_notebooks : ℕ
  pages_per_notebook : ℕ
  initial_drawings_per_page : ℕ
  new_drawings_per_page : ℕ
  full_notebooks_after_reorg : ℕ
  full_pages_in_last_notebook : ℕ

/-- Calculates the number of drawings on the last page after reorganization --/
def drawings_on_last_page (p : NotebookProblem) : ℕ :=
  let total_drawings := p.initial_notebooks * p.pages_per_notebook * p.initial_drawings_per_page
  let full_pages := (p.full_notebooks_after_reorg * p.pages_per_notebook) + p.full_pages_in_last_notebook
  total_drawings - (full_pages * p.new_drawings_per_page)

/-- Theorem stating that for the given problem, the number of drawings on the last page is 4 --/
theorem notebook_reorganization_theorem (p : NotebookProblem) 
  (h1 : p.initial_notebooks = 10)
  (h2 : p.pages_per_notebook = 50)
  (h3 : p.initial_drawings_per_page = 5)
  (h4 : p.new_drawings_per_page = 8)
  (h5 : p.full_notebooks_after_reorg = 6)
  (h6 : p.full_pages_in_last_notebook = 40) :
  drawings_on_last_page p = 4 := by
  sorry

end NUMINAMATH_CALUDE_notebook_reorganization_theorem_l802_80248


namespace NUMINAMATH_CALUDE_probability_of_sum_five_l802_80224

def number_of_faces : ℕ := 6

def total_outcomes (n : ℕ) : ℕ := n * n

def favorable_outcomes : List (ℕ × ℕ) := [(1, 4), (2, 3), (3, 2), (4, 1)]

def probability (favorable : ℕ) (total : ℕ) : ℚ := favorable / total

theorem probability_of_sum_five :
  probability (favorable_outcomes.length) (total_outcomes number_of_faces) = 1 / 9 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_sum_five_l802_80224


namespace NUMINAMATH_CALUDE_vertex_angle_and_side_not_determine_equilateral_l802_80283

/-- A triangle with side lengths a, b, c and angles A, B, C (opposite to sides a, b, c respectively) -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- An equilateral triangle is a triangle with all sides equal -/
def IsEquilateral (t : Triangle) : Prop :=
  t.a = t.b ∧ t.b = t.c

/-- A vertex angle is any of the three angles in a triangle -/
def IsVertexAngle (t : Triangle) (angle : ℝ) : Prop :=
  angle = t.A ∨ angle = t.B ∨ angle = t.C

/-- Statement: Knowing a vertex angle and a side length is not sufficient to uniquely determine an equilateral triangle -/
theorem vertex_angle_and_side_not_determine_equilateral :
  ∃ (t1 t2 : Triangle) (angle side : ℝ),
    IsVertexAngle t1 angle ∧
    IsVertexAngle t2 angle ∧
    (t1.a = side ∨ t1.b = side ∨ t1.c = side) ∧
    (t2.a = side ∨ t2.b = side ∨ t2.c = side) ∧
    IsEquilateral t1 ∧
    ¬IsEquilateral t2 :=
  sorry

end NUMINAMATH_CALUDE_vertex_angle_and_side_not_determine_equilateral_l802_80283


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_exists_l802_80205

theorem quadratic_equation_solution_exists (a b c : ℝ) : 
  (∃ x : ℝ, a * x^2 + 2 * b * x + c = 0) ∨ 
  (∃ x : ℝ, b * x^2 + 2 * c * x + a = 0) ∨ 
  (∃ x : ℝ, c * x^2 + 2 * a * x + b = 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_exists_l802_80205


namespace NUMINAMATH_CALUDE_wendy_pictures_l802_80284

theorem wendy_pictures (total : ℕ) (num_albums : ℕ) (pics_per_album : ℕ) (first_album : ℕ) : 
  total = 79 →
  num_albums = 5 →
  pics_per_album = 7 →
  first_album + num_albums * pics_per_album = total →
  first_album = 44 := by
sorry

end NUMINAMATH_CALUDE_wendy_pictures_l802_80284


namespace NUMINAMATH_CALUDE_tom_hockey_games_this_year_l802_80289

/-- The number of hockey games Tom went to this year -/
def games_this_year (total_games : ℕ) (last_year_games : ℕ) : ℕ :=
  total_games - last_year_games

/-- Theorem stating that Tom went to 4 hockey games this year -/
theorem tom_hockey_games_this_year :
  games_this_year 13 9 = 4 := by
  sorry

end NUMINAMATH_CALUDE_tom_hockey_games_this_year_l802_80289


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l802_80254

theorem quadratic_equation_solution (h : (63 * (5/7)^2 + 36) = (100 * (5/7) - 9)) :
  (63 * 1^2 + 36) = (100 * 1 - 9) ∧ 
  ∀ x : ℚ, x ≠ 5/7 → x ≠ 1 → (63 * x^2 + 36) ≠ (100 * x - 9) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l802_80254


namespace NUMINAMATH_CALUDE_second_category_amount_is_720_l802_80273

/-- Represents a budget with three categories -/
structure Budget where
  total : ℕ
  ratio1 : ℕ
  ratio2 : ℕ
  ratio3 : ℕ

/-- Calculates the amount allocated to the second category in a budget -/
def amount_second_category (b : Budget) : ℕ :=
  b.total * b.ratio2 / (b.ratio1 + b.ratio2 + b.ratio3)

/-- Theorem stating that for a budget with ratio 5:4:1 and total $1800, 
    the amount allocated to the second category is $720 -/
theorem second_category_amount_is_720 :
  ∀ (b : Budget), b.total = 1800 ∧ b.ratio1 = 5 ∧ b.ratio2 = 4 ∧ b.ratio3 = 1 →
  amount_second_category b = 720 := by
  sorry

end NUMINAMATH_CALUDE_second_category_amount_is_720_l802_80273


namespace NUMINAMATH_CALUDE_total_distance_is_202_l802_80217

/-- Represents the driving data for a single day -/
structure DailyDrive where
  hours : Float
  speed : Float

/-- Calculates the distance traveled in a day given the driving data -/
def distanceTraveled (drive : DailyDrive) : Float :=
  drive.hours * drive.speed

/-- The week's driving schedule -/
def weekSchedule : List DailyDrive := [
  { hours := 3, speed := 12 },    -- Monday
  { hours := 3.5, speed := 8 },   -- Tuesday
  { hours := 2.5, speed := 12 },  -- Wednesday
  { hours := 4, speed := 6 },     -- Thursday
  { hours := 2, speed := 12 },    -- Friday
  { hours := 3, speed := 15 },    -- Saturday
  { hours := 1.5, speed := 10 }   -- Sunday
]

/-- Theorem: The total distance traveled during the week is 202 km -/
theorem total_distance_is_202 :
  (weekSchedule.map distanceTraveled).sum = 202 := by
  sorry

end NUMINAMATH_CALUDE_total_distance_is_202_l802_80217


namespace NUMINAMATH_CALUDE_charles_cleaning_time_l802_80229

theorem charles_cleaning_time 
  (alice_time : ℝ) 
  (bob_time : ℝ) 
  (charles_time : ℝ) 
  (h1 : alice_time = 20) 
  (h2 : bob_time = 3/4 * alice_time) 
  (h3 : charles_time = 2/3 * bob_time) : 
  charles_time = 10 := by
sorry

end NUMINAMATH_CALUDE_charles_cleaning_time_l802_80229


namespace NUMINAMATH_CALUDE_angle_measure_l802_80279

theorem angle_measure (x : ℝ) : 
  (180 - x = 4 * (90 - x)) → x = 60 := by
  sorry

end NUMINAMATH_CALUDE_angle_measure_l802_80279


namespace NUMINAMATH_CALUDE_opposite_of_negative_one_to_2023_l802_80203

theorem opposite_of_negative_one_to_2023 :
  ∀ n : ℕ, n = 2023 → Odd n → (-((-1)^n)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_one_to_2023_l802_80203


namespace NUMINAMATH_CALUDE_water_tank_capacity_l802_80221

theorem water_tank_capacity (tank_capacity : ℝ) : 
  (0.6 * tank_capacity - (0.7 * tank_capacity) = 45) → 
  tank_capacity = 450 := by
  sorry

end NUMINAMATH_CALUDE_water_tank_capacity_l802_80221


namespace NUMINAMATH_CALUDE_ratio_of_sum_and_difference_l802_80290

theorem ratio_of_sum_and_difference (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) (h : a + b = 7 * (a - b)) : a / b = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_sum_and_difference_l802_80290


namespace NUMINAMATH_CALUDE_quadratic_equation_two_distinct_roots_l802_80295

theorem quadratic_equation_two_distinct_roots (k : ℝ) :
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧
  x₁^2 - (k + 3) * x₁ + 2 * k + 1 = 0 ∧
  x₂^2 - (k + 3) * x₂ + 2 * k + 1 = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_two_distinct_roots_l802_80295


namespace NUMINAMATH_CALUDE_unique_function_satisfying_inequality_l802_80288

def satisfies_inequality (f : ℝ → ℝ) : Prop :=
  ∀ x y z : ℝ, f (x * y) + f (x * z) + f (y * z) - f x * f y * f z ≥ 1

theorem unique_function_satisfying_inequality :
  ∃! f : ℝ → ℝ, satisfies_inequality f ∧ ∀ x : ℝ, f x = 1 :=
sorry

end NUMINAMATH_CALUDE_unique_function_satisfying_inequality_l802_80288


namespace NUMINAMATH_CALUDE_existence_of_lower_bound_upper_bound_l802_80297

/-- The number of coefficients in (x+1)^a(x+2)^(n-a) divisible by 3 -/
def f (n a : ℕ) : ℕ :=
  sorry

/-- The minimum of f(n,a) for all valid a -/
def F (n : ℕ) : ℕ :=
  sorry

/-- There exist infinitely many positive integers n such that F(n) ≥ (n-1)/3 -/
theorem existence_of_lower_bound : ∃ (S : Set ℕ), Set.Infinite S ∧ ∀ n ∈ S, F n ≥ (n - 1) / 3 :=
  sorry

/-- For any positive integer n, F(n) ≤ (n-1)/3 -/
theorem upper_bound (n : ℕ) (hn : n > 0) : F n ≤ (n - 1) / 3 :=
  sorry

end NUMINAMATH_CALUDE_existence_of_lower_bound_upper_bound_l802_80297


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l802_80207

/-- A function f(x) = ax^2 + bx + 1 that is even and has domain [2a, 1-a] -/
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + 1

/-- The domain of f is [2a, 1-a] -/
def domain (a : ℝ) : Set ℝ := Set.Icc (2 * a) (1 - a)

/-- f is an even function -/
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

theorem sum_of_coefficients (a b : ℝ) :
  (∃ x, x ∈ domain a) →
  is_even (f a b) →
  a + b = -1 :=
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l802_80207


namespace NUMINAMATH_CALUDE_gcd_nine_factorial_seven_factorial_squared_l802_80219

theorem gcd_nine_factorial_seven_factorial_squared :
  Nat.gcd (Nat.factorial 9) ((Nat.factorial 7)^2) = 362880 := by
  sorry

end NUMINAMATH_CALUDE_gcd_nine_factorial_seven_factorial_squared_l802_80219


namespace NUMINAMATH_CALUDE_homer_candy_crush_ratio_l802_80277

/-- Proves that the ratio of points scored on the third try to points scored on the second try is 2:1 in Homer's Candy Crush game -/
theorem homer_candy_crush_ratio :
  ∀ (first_try second_try third_try : ℕ),
    first_try = 400 →
    second_try = first_try - 70 →
    ∃ (m : ℕ), third_try = m * second_try →
    first_try + second_try + third_try = 1390 →
    third_try = 2 * second_try :=
by
  sorry

#check homer_candy_crush_ratio

end NUMINAMATH_CALUDE_homer_candy_crush_ratio_l802_80277


namespace NUMINAMATH_CALUDE_last_two_average_l802_80285

theorem last_two_average (list : List ℝ) : 
  list.length = 7 →
  (list.sum / 7 : ℝ) = 60 →
  ((list.take 3).sum / 3 : ℝ) = 45 →
  ((list.drop 3).take 2).sum / 2 = 70 →
  ((list.drop 5).sum / 2 : ℝ) = 72.5 := by
sorry

end NUMINAMATH_CALUDE_last_two_average_l802_80285


namespace NUMINAMATH_CALUDE_partnership_capital_share_l802_80253

theorem partnership_capital_share :
  let total_profit : ℚ := 2430
  let a_profit_share : ℚ := 810
  let a_capital_share : ℚ := 1/3
  let b_capital_share : ℚ := 1/4
  let d_capital_share : ℚ := 1 - (a_capital_share + b_capital_share + c_capital_share)
  let c_capital_share : ℚ := 5/24
  a_profit_share / total_profit = a_capital_share ∧
  a_capital_share + b_capital_share + c_capital_share + d_capital_share = 1 →
  c_capital_share = 5/24 :=
by sorry

end NUMINAMATH_CALUDE_partnership_capital_share_l802_80253


namespace NUMINAMATH_CALUDE_sum_of_even_and_multiples_of_five_l802_80241

/-- The number of four-digit even numbers -/
def C : ℕ := 4500

/-- The number of four-digit multiples of 5 -/
def B : ℕ := 1800

/-- The sum of four-digit even numbers and four-digit multiples of 5 is 6300 -/
theorem sum_of_even_and_multiples_of_five : C + B = 6300 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_even_and_multiples_of_five_l802_80241


namespace NUMINAMATH_CALUDE_min_value_trig_expression_l802_80227

theorem min_value_trig_expression (α β : ℝ) : 
  (3 * Real.cos α + 4 * Real.sin β - 10)^2 + (3 * Real.sin α + 4 * Real.cos β - 20)^2 ≥ 236 := by
  sorry

end NUMINAMATH_CALUDE_min_value_trig_expression_l802_80227


namespace NUMINAMATH_CALUDE_johnson_prescription_l802_80204

/-- Represents a prescription with a fixed daily dose -/
structure Prescription where
  totalDays : ℕ
  remainingPills : ℕ
  daysElapsed : ℕ
  dailyDose : ℕ

/-- Calculates the daily dose given a prescription -/
def calculateDailyDose (p : Prescription) : ℕ :=
  (p.totalDays * p.dailyDose - p.remainingPills) / p.daysElapsed

/-- Theorem stating that for the given prescription, the daily dose is 2 pills -/
theorem johnson_prescription :
  ∃ (p : Prescription),
    p.totalDays = 30 ∧
    p.remainingPills = 12 ∧
    p.daysElapsed = 24 ∧
    calculateDailyDose p = 2 :=
by
  sorry


end NUMINAMATH_CALUDE_johnson_prescription_l802_80204


namespace NUMINAMATH_CALUDE_annual_pension_correct_l802_80246

/-- Represents the annual pension calculation for an employee -/
noncomputable def annual_pension 
  (a b p q : ℝ) 
  (h1 : b ≠ a) : ℝ :=
  (q * a^2 - p * b^2)^2 / (4 * (p * b - q * a)^2)

/-- Theorem stating the annual pension calculation is correct -/
theorem annual_pension_correct 
  (a b p q : ℝ) 
  (h1 : b ≠ a)
  (h2 : ∃ (k x : ℝ), 
    k * (x - a)^2 = k * x^2 - p ∧ 
    k * (x + b)^2 = k * x^2 + q) :
  ∃ (k x : ℝ), k * x^2 = annual_pension a b p q h1 := by
  sorry

end NUMINAMATH_CALUDE_annual_pension_correct_l802_80246


namespace NUMINAMATH_CALUDE_tan_eight_pi_thirds_l802_80270

theorem tan_eight_pi_thirds : Real.tan (8 * Real.pi / 3) = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_eight_pi_thirds_l802_80270


namespace NUMINAMATH_CALUDE_smallest_perfect_square_divisible_by_2_3_5_l802_80287

theorem smallest_perfect_square_divisible_by_2_3_5 : 
  ∀ n : ℕ, n > 0 → n.sqrt ^ 2 = n → n % 2 = 0 → n % 3 = 0 → n % 5 = 0 → n ≥ 225 := by
  sorry

end NUMINAMATH_CALUDE_smallest_perfect_square_divisible_by_2_3_5_l802_80287


namespace NUMINAMATH_CALUDE_cylinder_volume_ratio_l802_80242

/-- The ratio of volumes of two cylinders formed from a 6x9 rectangle -/
theorem cylinder_volume_ratio : 
  let rect_width : ℝ := 6
  let rect_height : ℝ := 9
  let cylinder1_height : ℝ := rect_height
  let cylinder1_circumference : ℝ := rect_width
  let cylinder2_height : ℝ := rect_width
  let cylinder2_circumference : ℝ := rect_height
  let cylinder1_volume : ℝ := π * (cylinder1_circumference / (2 * π))^2 * cylinder1_height
  let cylinder2_volume : ℝ := π * (cylinder2_circumference / (2 * π))^2 * cylinder2_height
  let max_volume : ℝ := max cylinder1_volume cylinder2_volume
  let min_volume : ℝ := min cylinder1_volume cylinder2_volume
  (max_volume / min_volume) = 3/4 := by
sorry

end NUMINAMATH_CALUDE_cylinder_volume_ratio_l802_80242


namespace NUMINAMATH_CALUDE_focus_of_our_parabola_l802_80223

/-- A parabola is defined by the equation y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The focus of a parabola is a point (h, k + 1/(4a)) where (h, k) is the vertex -/
def focus (p : Parabola) : ℝ × ℝ := sorry

/-- Our specific parabola -/
def our_parabola : Parabola :=
  { a := 4
    b := 8
    c := -1 }

theorem focus_of_our_parabola :
  focus our_parabola = (-1, -79/16) := by sorry

end NUMINAMATH_CALUDE_focus_of_our_parabola_l802_80223


namespace NUMINAMATH_CALUDE_division_powers_equality_l802_80247

theorem division_powers_equality (a : ℝ) (h : a ≠ 0) :
  a^6 / ((1/2) * a^2) = 2 * a^4 := by sorry

end NUMINAMATH_CALUDE_division_powers_equality_l802_80247


namespace NUMINAMATH_CALUDE_pizza_payment_difference_l802_80291

/-- Represents the pizza sharing scenario between Doug and Dave -/
structure PizzaSharing where
  total_slices : ℕ
  plain_cost : ℚ
  topping_cost : ℚ
  topped_slices : ℕ
  dave_plain_slices : ℕ

/-- Calculates the cost per slice given the total cost and number of slices -/
def cost_per_slice (total_cost : ℚ) (total_slices : ℕ) : ℚ :=
  total_cost / total_slices

/-- Calculates the payment difference between Dave and Doug -/
def payment_difference (ps : PizzaSharing) : ℚ :=
  let total_cost := ps.plain_cost + ps.topping_cost
  let per_slice_cost := cost_per_slice total_cost ps.total_slices
  let dave_slices := ps.topped_slices + ps.dave_plain_slices
  let doug_slices := ps.total_slices - dave_slices
  dave_slices * per_slice_cost - doug_slices * per_slice_cost

/-- Theorem stating that the payment difference is 2.8 under the given conditions -/
theorem pizza_payment_difference :
  let ps : PizzaSharing := {
    total_slices := 10,
    plain_cost := 10,
    topping_cost := 4,
    topped_slices := 4,
    dave_plain_slices := 2
  }
  payment_difference ps = 2.8 := by
  sorry


end NUMINAMATH_CALUDE_pizza_payment_difference_l802_80291


namespace NUMINAMATH_CALUDE_gumball_distribution_l802_80232

/-- Represents the number of gumballs each person has -/
structure Gumballs :=
  (joanna : ℕ)
  (jacques : ℕ)
  (julia : ℕ)

/-- Calculates the total number of gumballs -/
def total_gumballs (g : Gumballs) : ℕ :=
  g.joanna + g.jacques + g.julia

/-- Represents the purchase multipliers for each person -/
structure PurchaseMultipliers :=
  (joanna : ℕ)
  (jacques : ℕ)
  (julia : ℕ)

/-- Calculates the number of gumballs after purchases -/
def after_purchase (initial : Gumballs) (multipliers : PurchaseMultipliers) : Gumballs :=
  { joanna := initial.joanna + initial.joanna * multipliers.joanna,
    jacques := initial.jacques + initial.jacques * multipliers.jacques,
    julia := initial.julia + initial.julia * multipliers.julia }

/-- Theorem statement -/
theorem gumball_distribution 
  (initial : Gumballs) 
  (multipliers : PurchaseMultipliers) :
  initial.joanna = 40 ∧ 
  initial.jacques = 60 ∧ 
  initial.julia = 80 ∧
  multipliers.joanna = 5 ∧
  multipliers.jacques = 3 ∧
  multipliers.julia = 2 →
  let final := after_purchase initial multipliers
  (final.joanna = 240 ∧ 
   final.jacques = 240 ∧ 
   final.julia = 240) ∧
  (total_gumballs final / 3 = 240) :=
by sorry

end NUMINAMATH_CALUDE_gumball_distribution_l802_80232


namespace NUMINAMATH_CALUDE_converse_statement_l802_80237

theorem converse_statement (a b : ℝ) :
  (∀ a b, a > 1 ∧ b > 1 → a + b > 2) →
  (∀ a b, a + b ≤ 2 → a ≤ 1 ∨ b ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_converse_statement_l802_80237


namespace NUMINAMATH_CALUDE_investment_amount_proof_l802_80238

/-- Given an amount P and an interest rate y, proves that P = 5000 if the simple
    interest for 2 years is 500 and the compound interest for 2 years is 512.50 -/
theorem investment_amount_proof (P y : ℝ) 
    (h_simple : P * y * 2 / 100 = 500)
    (h_compound : P * ((1 + y / 100)^2 - 1) = 512.50) : 
    P = 5000 := by
  sorry

end NUMINAMATH_CALUDE_investment_amount_proof_l802_80238


namespace NUMINAMATH_CALUDE_division_equality_l802_80299

theorem division_equality (h : 29.94 / 1.45 = 17.1) : 2994 / 14.5 = 171 := by
  sorry

end NUMINAMATH_CALUDE_division_equality_l802_80299


namespace NUMINAMATH_CALUDE_price_relationship_total_cost_max_toy_A_l802_80272

/- Define the unit prices of toys A and B -/
def price_A : ℕ := 50
def price_B : ℕ := 75

/- Define the relationship between prices -/
theorem price_relationship : price_B = price_A + 25 := by sorry

/- Define the total cost of 2B and 1A -/
theorem total_cost : 2 * price_B + price_A = 200 := by sorry

/- Define the function for total cost given number of A -/
def total_cost_function (num_A : ℕ) : ℕ := price_A * num_A + price_B * (2 * num_A)

/- Define the maximum budget -/
def max_budget : ℕ := 20000

/- Theorem to prove the maximum number of toy A that can be purchased -/
theorem max_toy_A : 
  (∀ n : ℕ, total_cost_function n ≤ max_budget → n ≤ 100) ∧ 
  total_cost_function 100 ≤ max_budget := by sorry

end NUMINAMATH_CALUDE_price_relationship_total_cost_max_toy_A_l802_80272


namespace NUMINAMATH_CALUDE_sphere_volume_radius_3_l802_80292

/-- The volume of a sphere with radius 3 is 36π. -/
theorem sphere_volume_radius_3 :
  let r : ℝ := 3
  let volume := (4 / 3) * Real.pi * r ^ 3
  volume = 36 * Real.pi := by sorry

end NUMINAMATH_CALUDE_sphere_volume_radius_3_l802_80292


namespace NUMINAMATH_CALUDE_children_count_l802_80282

/-- The number of children required to assemble one small robot -/
def small_robot_children : ℕ := 2

/-- The number of children required to assemble one large robot -/
def large_robot_children : ℕ := 3

/-- The number of small robots assembled -/
def small_robots : ℕ := 18

/-- The number of large robots assembled -/
def large_robots : ℕ := 12

/-- The total number of children -/
def total_children : ℕ := small_robot_children * small_robots + large_robot_children * large_robots

theorem children_count : total_children = 72 := by sorry

end NUMINAMATH_CALUDE_children_count_l802_80282


namespace NUMINAMATH_CALUDE_select_four_with_both_genders_eq_34_l802_80260

/-- The number of ways to select 4 individuals from 4 boys and 3 girls,
    such that the selection includes both boys and girls. -/
def select_four_with_both_genders (num_boys : ℕ) (num_girls : ℕ) : ℕ :=
  Nat.choose (num_boys + num_girls) 4 - Nat.choose num_boys 4

/-- Theorem stating that selecting 4 individuals from 4 boys and 3 girls,
    such that the selection includes both boys and girls, results in 34 ways. -/
theorem select_four_with_both_genders_eq_34 :
  select_four_with_both_genders 4 3 = 34 := by
  sorry

#eval select_four_with_both_genders 4 3

end NUMINAMATH_CALUDE_select_four_with_both_genders_eq_34_l802_80260


namespace NUMINAMATH_CALUDE_common_chord_length_l802_80209

theorem common_chord_length (r : ℝ) (h : r = 12) :
  let chord_length := 2 * (r * Real.sqrt 3)
  chord_length = 12 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_common_chord_length_l802_80209


namespace NUMINAMATH_CALUDE_stratified_sampling_theorem_l802_80245

structure Department where
  total : ℕ
  males : ℕ
  females : ℕ

def sample_size : ℕ := 3

def dept_A : Department := ⟨10, 6, 4⟩
def dept_B : Department := ⟨5, 3, 2⟩

def total_staff : ℕ := dept_A.total + dept_B.total

def stratified_sample (d : Department) : ℕ :=
  (sample_size * d.total) / total_staff

def prob_at_least_one_female (d : Department) (n : ℕ) : ℚ :=
  1 - (Nat.choose d.males n : ℚ) / (Nat.choose d.total n : ℚ)

def prob_male_count (k : ℕ) : ℚ := 
  if k = 0 then 4 / 75
  else if k = 1 then 22 / 75
  else if k = 2 then 34 / 75
  else if k = 3 then 1 / 3
  else 0

def expected_male_count : ℚ := 2

theorem stratified_sampling_theorem :
  (stratified_sample dept_A = 2) ∧
  (stratified_sample dept_B = 1) ∧
  (prob_at_least_one_female dept_A 2 = 2 / 3) ∧
  (∀ k, 0 ≤ k ∧ k ≤ 3 → prob_male_count k = prob_male_count k) ∧
  (Finset.sum (Finset.range 4) (λ k => k * prob_male_count k) = expected_male_count) := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_theorem_l802_80245


namespace NUMINAMATH_CALUDE_four_digit_number_not_divisible_by_11_l802_80252

def is_not_divisible_by_11 (n : ℕ) : Prop := ¬(n % 11 = 0)

theorem four_digit_number_not_divisible_by_11 :
  ∀ B : ℕ, B < 10 →
  (∃ A : ℕ, A < 10 ∧ 
    (∀ B : ℕ, B < 10 → is_not_divisible_by_11 (9000 + 100 * A + 10 * B))) ↔ 
  (∃ A : ℕ, A = 1) :=
by sorry

end NUMINAMATH_CALUDE_four_digit_number_not_divisible_by_11_l802_80252


namespace NUMINAMATH_CALUDE_systematicSamplingExample_l802_80258

/-- Calculates the number of groups for systematic sampling -/
def systematicSamplingGroups (totalStudents : ℕ) (sampleSize : ℕ) : ℕ :=
  totalStudents / sampleSize

/-- Theorem stating that for 600 students and a sample size of 20, 
    the number of groups for systematic sampling is 30 -/
theorem systematicSamplingExample : 
  systematicSamplingGroups 600 20 = 30 := by
  sorry

end NUMINAMATH_CALUDE_systematicSamplingExample_l802_80258


namespace NUMINAMATH_CALUDE_exists_m_iff_n_power_of_two_l802_80274

theorem exists_m_iff_n_power_of_two (n : ℕ+) :
  (∃ m : ℤ, (2^n.val - 1) ∣ (m^2 + 9)) ↔ ∃ k : ℕ, n.val = 2^k :=
sorry

end NUMINAMATH_CALUDE_exists_m_iff_n_power_of_two_l802_80274


namespace NUMINAMATH_CALUDE_missing_figure_proof_l802_80280

theorem missing_figure_proof (x : ℝ) : (0.75 / 100) * x = 0.06 ↔ x = 8 := by sorry

end NUMINAMATH_CALUDE_missing_figure_proof_l802_80280


namespace NUMINAMATH_CALUDE_max_product_sum_2006_l802_80201

theorem max_product_sum_2006 : 
  (∃ (a b : ℤ), a + b = 2006 ∧ ∀ (x y : ℤ), x + y = 2006 → x * y ≤ a * b) ∧
  (∀ (a b : ℤ), a + b = 2006 → a * b ≤ 1006009) := by
  sorry

end NUMINAMATH_CALUDE_max_product_sum_2006_l802_80201


namespace NUMINAMATH_CALUDE_factors_of_50400_l802_80231

theorem factors_of_50400 : Nat.card (Nat.divisors 50400) = 108 := by
  sorry

end NUMINAMATH_CALUDE_factors_of_50400_l802_80231


namespace NUMINAMATH_CALUDE_bags_bought_l802_80200

def crayonPacks : ℕ := 5
def crayonPrice : ℚ := 5
def bookCount : ℕ := 10
def bookPrice : ℚ := 5
def calculatorCount : ℕ := 3
def calculatorPrice : ℚ := 5
def bookDiscount : ℚ := 0.2
def salesTax : ℚ := 0.05
def initialMoney : ℚ := 200
def bagPrice : ℚ := 10

def totalCost : ℚ :=
  crayonPacks * crayonPrice +
  bookCount * bookPrice * (1 - bookDiscount) +
  calculatorCount * calculatorPrice

def finalCost : ℚ := totalCost * (1 + salesTax)

def change : ℚ := initialMoney - finalCost

theorem bags_bought (h : change ≥ 0) : ⌊change / bagPrice⌋ = 11 := by
  sorry

#eval ⌊change / bagPrice⌋

end NUMINAMATH_CALUDE_bags_bought_l802_80200


namespace NUMINAMATH_CALUDE_unique_integer_property_l802_80216

theorem unique_integer_property : ∃! (n : ℕ), n > 0 ∧ 2000 * n + 1 = 33 * n := by
  sorry

end NUMINAMATH_CALUDE_unique_integer_property_l802_80216


namespace NUMINAMATH_CALUDE_sugar_in_house_l802_80213

/-- Given the total sugar needed and additional sugar needed, prove the amount of sugar stored in the house. -/
theorem sugar_in_house (total_sugar : ℕ) (additional_sugar : ℕ) 
  (h1 : total_sugar = 450)
  (h2 : additional_sugar = 163) :
  total_sugar - additional_sugar = 287 := by
  sorry

end NUMINAMATH_CALUDE_sugar_in_house_l802_80213


namespace NUMINAMATH_CALUDE_third_term_of_sequence_l802_80261

def arithmetic_sequence (a : ℤ) (d : ℤ) (n : ℕ) : ℤ := a + (n - 1) * d

theorem third_term_of_sequence (a : ℤ) (d : ℤ) :
  arithmetic_sequence a d 20 = 18 →
  arithmetic_sequence a d 21 = 20 →
  arithmetic_sequence a d 3 = -16 :=
by
  sorry

end NUMINAMATH_CALUDE_third_term_of_sequence_l802_80261


namespace NUMINAMATH_CALUDE_tournament_permutation_exists_l802_80234

/-- Represents the result of a match between two players -/
inductive MatchResult
  | Win
  | Loss

/-- Represents a tournament with n players -/
structure Tournament (n : Nat) where
  /-- The result of the match between player i and player j -/
  result : Fin n → Fin n → MatchResult

/-- A permutation of players -/
def PlayerPermutation (n : Nat) := Fin n → Fin n

/-- Checks if a player satisfies the condition with their neighbors -/
def satisfiesCondition (t : Tournament 1000) (p : PlayerPermutation 1000) (i : Fin 998) : Prop :=
  (t.result (p i) (p (i + 1)) = MatchResult.Win ∧ t.result (p i) (p (i + 2)) = MatchResult.Win) ∨
  (t.result (p i) (p (i + 1)) = MatchResult.Loss ∧ t.result (p i) (p (i + 2)) = MatchResult.Loss)

/-- The main theorem -/
theorem tournament_permutation_exists (t : Tournament 1000) :
  ∃ (p : PlayerPermutation 1000), ∀ (i : Fin 998), satisfiesCondition t p i := by
  sorry

end NUMINAMATH_CALUDE_tournament_permutation_exists_l802_80234


namespace NUMINAMATH_CALUDE_roses_picked_l802_80208

theorem roses_picked (initial : ℕ) (sold : ℕ) (final : ℕ) 
  (h1 : initial = 37) 
  (h2 : sold = 16) 
  (h3 : final = 40) : 
  final - (initial - sold) = 19 := by
sorry

end NUMINAMATH_CALUDE_roses_picked_l802_80208


namespace NUMINAMATH_CALUDE_root_equation_problem_l802_80235

/-- Given two constants p and q, if the specified equations have the given number of distinct roots
    and q = 8, then 50p - 10q = 20 -/
theorem root_equation_problem (p q : ℝ) : 
  (∃! x y z, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ 
    (x + p) * (x + q) * (x - 8) / (x + 4)^2 = 0) →
  (∃! x y, x ≠ y ∧ 
    (x + 4*p) * (x - 4) * (x - 10) / ((x + q) * (x - 8)) = 0) →
  q = 8 →
  50 * p - 10 * q = 20 := by
sorry

end NUMINAMATH_CALUDE_root_equation_problem_l802_80235


namespace NUMINAMATH_CALUDE_log_base_2_negative_range_l802_80250

-- Define the function f(x) = lg x
noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log 2

-- State the theorem
theorem log_base_2_negative_range :
  (∀ x, f (-x) = -f x) →  -- f is an odd function
  {x : ℝ | f x < 0} = Set.Ioo 0 1 := by sorry

end NUMINAMATH_CALUDE_log_base_2_negative_range_l802_80250
