import Mathlib

namespace NUMINAMATH_CALUDE_distance_between_points_l3003_300366

theorem distance_between_points (b : ℝ) :
  (∃ b₁ b₂ : ℝ, b = b₁ ∨ b = b₂) →
  (Real.sqrt ((3 * b - 1)^2 + (b + 1 - 4)^2) = 2 * Real.sqrt 13) →
  b * (5.47 - b) = 4.2 := by
sorry

end NUMINAMATH_CALUDE_distance_between_points_l3003_300366


namespace NUMINAMATH_CALUDE_function_range_condition_l3003_300302

/-- Given functions f and g, prove that m ≥ 3/2 under specified conditions -/
theorem function_range_condition (m : ℝ) (h_m : m > 0) : 
  (∀ x₁ ∈ Set.Icc (-1 : ℝ) 2, ∃ x₂ ∈ Set.Icc (-1 : ℝ) 2, 
    ((1/2 : ℝ) ^ x₁) = m * x₂ - 1) → 
  m ≥ 3/2 := by
  sorry


end NUMINAMATH_CALUDE_function_range_condition_l3003_300302


namespace NUMINAMATH_CALUDE_zoo_fraction_l3003_300322

/-- Given a zoo with various animals, prove that the fraction of elephants
    to the sum of parrots and snakes is 1/2. -/
theorem zoo_fraction (parrots snakes monkeys elephants zebras : ℕ) 
  (h1 : parrots = 8)
  (h2 : snakes = 3 * parrots)
  (h3 : monkeys = 2 * snakes)
  (h4 : ∃ f : ℚ, elephants = f * (parrots + snakes))
  (h5 : zebras + 3 = elephants)
  (h6 : monkeys - zebras = 35) :
  ∃ f : ℚ, elephants = f * (parrots + snakes) ∧ f = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_zoo_fraction_l3003_300322


namespace NUMINAMATH_CALUDE_knights_selection_l3003_300313

/-- The number of ways to select k non-adjacent elements from n elements in a circular arrangement -/
def circularNonAdjacentSelection (n k : ℕ) : ℕ :=
  Nat.choose (n - k + 1) k - Nat.choose (n - k - 1) (k - 2)

/-- The problem statement -/
theorem knights_selection :
  circularNonAdjacentSelection 50 15 = 463991880 := by
  sorry

end NUMINAMATH_CALUDE_knights_selection_l3003_300313


namespace NUMINAMATH_CALUDE_number_of_one_point_two_stamps_l3003_300363

/-- Represents the number of stamps of each denomination -/
structure StampCounts where
  half : ℕ
  eightyPercent : ℕ
  onePointTwo : ℕ

/-- The total value of all stamps in cents -/
def totalValue (s : StampCounts) : ℕ :=
  50 * s.half + 80 * s.eightyPercent + 120 * s.onePointTwo

/-- The theorem stating the number of 1.2 yuan stamps given the conditions -/
theorem number_of_one_point_two_stamps :
  ∃ (s : StampCounts),
    totalValue s = 6000 ∧
    s.eightyPercent = 4 * s.half ∧
    s.onePointTwo = 13 :=
by sorry

end NUMINAMATH_CALUDE_number_of_one_point_two_stamps_l3003_300363


namespace NUMINAMATH_CALUDE_pollywogs_disappear_in_44_days_l3003_300326

/-- The number of days it takes for all pollywogs to disappear from Elmer's pond -/
def days_until_empty (initial_pollywogs : ℕ) (maturation_rate : ℕ) (melvin_catch_rate : ℕ) (melvin_catch_days : ℕ) : ℕ :=
  let first_phase := melvin_catch_days
  let pollywogs_after_first_phase := initial_pollywogs - (maturation_rate + melvin_catch_rate) * first_phase
  let second_phase := pollywogs_after_first_phase / maturation_rate
  first_phase + second_phase

/-- Theorem stating that it takes 44 days for all pollywogs to disappear from Elmer's pond -/
theorem pollywogs_disappear_in_44_days :
  days_until_empty 2400 50 10 20 = 44 := by
  sorry

end NUMINAMATH_CALUDE_pollywogs_disappear_in_44_days_l3003_300326


namespace NUMINAMATH_CALUDE_percentage_studying_both_languages_l3003_300397

def english_percentage : ℝ := 90
def german_percentage : ℝ := 80

theorem percentage_studying_both_languages :
  let both_percentage := english_percentage + german_percentage - 100
  both_percentage = 70 := by sorry

end NUMINAMATH_CALUDE_percentage_studying_both_languages_l3003_300397


namespace NUMINAMATH_CALUDE_probability_three_same_color_l3003_300332

def white_balls : ℕ := 7
def black_balls : ℕ := 8
def total_balls : ℕ := white_balls + black_balls
def drawn_balls : ℕ := 3

def probability_same_color : ℚ :=
  (Nat.choose white_balls drawn_balls + Nat.choose black_balls drawn_balls) /
  Nat.choose total_balls drawn_balls

theorem probability_three_same_color :
  probability_same_color = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_probability_three_same_color_l3003_300332


namespace NUMINAMATH_CALUDE_Q_divisible_by_three_l3003_300321

def Q (x p q : ℤ) : ℤ := x^3 - x + (p+1)*x + q

theorem Q_divisible_by_three (p q : ℤ) 
  (h1 : 3 ∣ (p + 1)) 
  (h2 : 3 ∣ q) : 
  ∀ x : ℤ, 3 ∣ Q x p q := by
sorry

end NUMINAMATH_CALUDE_Q_divisible_by_three_l3003_300321


namespace NUMINAMATH_CALUDE_other_root_of_quadratic_l3003_300368

theorem other_root_of_quadratic (m : ℝ) : 
  (3 : ℝ) ^ 2 + m * 3 - 12 = 0 → 
  (-4 : ℝ) ^ 2 + m * (-4) - 12 = 0 := by
sorry

end NUMINAMATH_CALUDE_other_root_of_quadratic_l3003_300368


namespace NUMINAMATH_CALUDE_parabola_vertex_l3003_300300

/-- The parabola equation -/
def parabola (x y : ℝ) : Prop := y = (x - 2)^2

/-- The vertex of the parabola -/
def vertex : ℝ × ℝ := (2, 0)

/-- Theorem: The vertex coordinates of the parabola y = (x-2)² are (2, 0) -/
theorem parabola_vertex :
  ∀ x y : ℝ, parabola x y → (x, y) = vertex :=
sorry

end NUMINAMATH_CALUDE_parabola_vertex_l3003_300300


namespace NUMINAMATH_CALUDE_parabola_directrix_l3003_300301

/-- The parabola equation -/
def parabola_eq (x y : ℝ) : Prop :=
  y = (x^2 - 8*x + 12) / 16

/-- The directrix equation -/
def directrix_eq (y : ℝ) : Prop :=
  y = -5/4

/-- Theorem: The directrix of the given parabola is y = -5/4 -/
theorem parabola_directrix :
  ∀ x y : ℝ, parabola_eq x y → ∃ d : ℝ, directrix_eq d ∧ 
  (∀ p q : ℝ, parabola_eq p q → (p - x)^2 + (q - y)^2 = (q - d)^2) :=
sorry

end NUMINAMATH_CALUDE_parabola_directrix_l3003_300301


namespace NUMINAMATH_CALUDE_ceiling_floor_sum_l3003_300342

theorem ceiling_floor_sum : ⌈(7 : ℝ) / 3⌉ + ⌊-(7 : ℝ) / 3⌋ = 0 := by sorry

end NUMINAMATH_CALUDE_ceiling_floor_sum_l3003_300342


namespace NUMINAMATH_CALUDE_perfect_square_base_l3003_300304

theorem perfect_square_base : ∃! (d : ℕ), d > 1 ∧ ∃ (n : ℕ), d^4 + d^3 + d^2 + d + 1 = n^2 :=
sorry

end NUMINAMATH_CALUDE_perfect_square_base_l3003_300304


namespace NUMINAMATH_CALUDE_range_of_m_l3003_300323

/-- α is the condition that x ≤ -5 or x ≥ 1 -/
def α (x : ℝ) : Prop := x ≤ -5 ∨ x ≥ 1

/-- β is the condition that 2m-3 ≤ x ≤ 2m+1 -/
def β (m x : ℝ) : Prop := 2*m - 3 ≤ x ∧ x ≤ 2*m + 1

/-- α is a necessary condition for β -/
def α_necessary_for_β (m : ℝ) : Prop := ∀ x, β m x → α x

theorem range_of_m (m : ℝ) : α_necessary_for_β m → m ≥ 2 ∨ m ≤ -3 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l3003_300323


namespace NUMINAMATH_CALUDE_divisibility_criterion_1207_l3003_300386

-- Define a two-digit number
def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

-- Define the sum of cubes of digits
def sum_of_cubes_of_digits (n : ℕ) : ℕ :=
  (n / 10) ^ 3 + (n % 10) ^ 3

-- Theorem statement
theorem divisibility_criterion_1207 (x : ℕ) :
  is_two_digit x →
  sum_of_cubes_of_digits x = 344 →
  (1207 % x = 0 ↔ (x = 17 ∨ x = 71)) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_criterion_1207_l3003_300386


namespace NUMINAMATH_CALUDE_exists_claw_count_for_total_time_specific_grooming_problem_l3003_300335

/-- Represents the grooming time for a cat -/
structure GroomingTime where
  clipTime : ℕ -- Time to clip one nail in seconds
  earCleanTime : ℕ -- Time to clean one ear in seconds
  shampooTime : ℕ -- Time to shampoo in seconds

/-- Theorem stating that there exists a number of claws that results in the given total grooming time -/
theorem exists_claw_count_for_total_time 
  (g : GroomingTime) 
  (totalTime : ℕ) : 
  ∃ (clawCount : ℕ), 
    g.clipTime * clawCount + g.earCleanTime * 2 + g.shampooTime * 60 = totalTime :=
by
  sorry

/-- Application of the theorem to the specific problem -/
theorem specific_grooming_problem : 
  ∃ (clawCount : ℕ), 
    10 * clawCount + 90 * 2 + 5 * 60 = 640 :=
by
  sorry

end NUMINAMATH_CALUDE_exists_claw_count_for_total_time_specific_grooming_problem_l3003_300335


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l3003_300324

-- Define set A
def A : Set ℝ := {x | |x| < 1}

-- Define set B
def B : Set ℝ := {y | ∃ x, y = 2^x + 1}

-- State the theorem
theorem complement_A_intersect_B : (Set.compl A) ∩ B = Set.Ioi 1 := by
  sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l3003_300324


namespace NUMINAMATH_CALUDE_alcohol_percentage_original_mixture_l3003_300353

/-- Proves that the percentage of alcohol in the original mixture is 20% --/
theorem alcohol_percentage_original_mixture :
  let original_volume : ℝ := 15
  let added_water : ℝ := 5
  let new_alcohol_percentage : ℝ := 15
  let new_volume : ℝ := original_volume + added_water
  let original_alcohol_volume : ℝ := original_volume * (original_alcohol_percentage / 100)
  let new_alcohol_volume : ℝ := new_volume * (new_alcohol_percentage / 100)
  ∀ original_alcohol_percentage : ℝ,
    original_alcohol_volume = new_alcohol_volume →
    original_alcohol_percentage = 20 :=
by
  sorry


end NUMINAMATH_CALUDE_alcohol_percentage_original_mixture_l3003_300353


namespace NUMINAMATH_CALUDE_complement_A_union_B_subset_l3003_300378

-- Define the sets
def U : Set ℝ := Set.univ
def A : Set ℝ := {x | x > 1}
def B : Set ℝ := {x | x < 2}

-- State the theorem
theorem complement_A_union_B_subset :
  (Set.compl A ∪ B) ⊆ {x : ℝ | x < 2} := by sorry

end NUMINAMATH_CALUDE_complement_A_union_B_subset_l3003_300378


namespace NUMINAMATH_CALUDE_sum_inequality_l3003_300320

theorem sum_inequality (a b x y : ℝ) (ha : 0 < a) (hb : 0 < b) (hx : 0 < x) (hy : 0 < y)
  (hax : a * x ≤ 5) (hay : a * y ≤ 10) (hbx : b * x ≤ 10) (hby : b * y ≤ 10) :
  a * x + a * y + b * x + b * y ≤ 30 := by
  sorry

end NUMINAMATH_CALUDE_sum_inequality_l3003_300320


namespace NUMINAMATH_CALUDE_muffin_combinations_l3003_300346

/-- The number of ways to distribute n indistinguishable items into k distinguishable bins -/
def distribute (n k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- The number of muffin types -/
def muffin_types : ℕ := 4

/-- The number of additional muffins to distribute -/
def additional_muffins : ℕ := 4

theorem muffin_combinations :
  distribute additional_muffins muffin_types = 35 := by
  sorry

end NUMINAMATH_CALUDE_muffin_combinations_l3003_300346


namespace NUMINAMATH_CALUDE_sample_mean_estimates_population_mean_l3003_300373

/-- A type to represent statistical populations -/
structure Population where
  mean : ℝ

/-- A type to represent samples from a population -/
structure Sample where
  mean : ℝ

/-- Predicate to determine if a sample mean is an estimate of a population mean -/
def is_estimate_of (s : Sample) (p : Population) : Prop :=
  ∃ (ε : ℝ), ε > 0 ∧ |s.mean - p.mean| < ε

/-- Theorem stating that a sample mean is an estimate of the population mean -/
theorem sample_mean_estimates_population_mean (s : Sample) (p : Population) :
  is_estimate_of s p :=
sorry

end NUMINAMATH_CALUDE_sample_mean_estimates_population_mean_l3003_300373


namespace NUMINAMATH_CALUDE_cost_of_one_each_l3003_300398

/-- Represents the cost of goods A, B, and C -/
structure GoodsCost where
  A : ℝ
  B : ℝ
  C : ℝ

/-- The total cost of a combination of goods -/
def total_cost (g : GoodsCost) (a b c : ℝ) : ℝ :=
  a * g.A + b * g.B + c * g.C

theorem cost_of_one_each (g : GoodsCost) 
  (h1 : total_cost g 3 7 1 = 315)
  (h2 : total_cost g 4 10 1 = 420) :
  total_cost g 1 1 1 = 105 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_one_each_l3003_300398


namespace NUMINAMATH_CALUDE_shifted_function_point_l3003_300381

/-- A function whose graph passes through (1, -1) -/
def passes_through_point (f : ℝ → ℝ) : Prop :=
  f 1 = -1

/-- The horizontally shifted function -/
def shift_function (f : ℝ → ℝ) (x : ℝ) : ℝ :=
  f (x - 3)

theorem shifted_function_point (f : ℝ → ℝ) :
  passes_through_point f → passes_through_point (shift_function f) :=
by
  sorry

#check shifted_function_point

end NUMINAMATH_CALUDE_shifted_function_point_l3003_300381


namespace NUMINAMATH_CALUDE_min_abs_sum_l3003_300389

open Complex

variable (α γ : ℂ)

def f (z : ℂ) : ℂ := (2 + 3*I)*z^2 + α*z + γ

theorem min_abs_sum (h1 : (f α γ 1).im = 0) (h2 : (f α γ I).im = 0) :
  ∃ (α₀ γ₀ : ℂ), (abs α₀ + abs γ₀ = 3) ∧ 
    ∀ (α' γ' : ℂ), (f α' γ' 1).im = 0 → (f α' γ' I).im = 0 → abs α' + abs γ' ≥ 3 :=
sorry

end NUMINAMATH_CALUDE_min_abs_sum_l3003_300389


namespace NUMINAMATH_CALUDE_class_composition_l3003_300354

theorem class_composition (initial_girls : ℕ) (initial_boys : ℕ) (girls_left : ℕ) :
  initial_girls * 6 = initial_boys * 5 →
  (initial_girls - girls_left) * 3 = initial_boys * 2 →
  girls_left = 20 →
  initial_boys = 120 := by
  sorry

end NUMINAMATH_CALUDE_class_composition_l3003_300354


namespace NUMINAMATH_CALUDE_cost_of_20_pencils_15_notebooks_l3003_300314

/-- The cost of a pencil -/
def pencil_cost : ℝ := sorry

/-- The cost of a notebook -/
def notebook_cost : ℝ := sorry

/-- The first condition: 9 pencils and 10 notebooks cost $5.45 -/
axiom condition1 : 9 * pencil_cost + 10 * notebook_cost = 5.45

/-- The second condition: 6 pencils and 4 notebooks cost $2.50 -/
axiom condition2 : 6 * pencil_cost + 4 * notebook_cost = 2.50

/-- Theorem: The cost of 20 pencils and 15 notebooks is $9.04 -/
theorem cost_of_20_pencils_15_notebooks :
  20 * pencil_cost + 15 * notebook_cost = 9.04 := by sorry

end NUMINAMATH_CALUDE_cost_of_20_pencils_15_notebooks_l3003_300314


namespace NUMINAMATH_CALUDE_square_sum_equation_l3003_300340

theorem square_sum_equation (x y : ℝ) (h1 : x + 2*y = 8) (h2 : x*y = 1) : x^2 + 4*y^2 = 60 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_equation_l3003_300340


namespace NUMINAMATH_CALUDE_smallest_number_l3003_300399

theorem smallest_number (a b c d : ℚ) (ha : a = 0) (hb : b = 5) (hc : c = -0.3) (hd : d = -1/3) :
  min a (min b (min c d)) = d := by sorry

end NUMINAMATH_CALUDE_smallest_number_l3003_300399


namespace NUMINAMATH_CALUDE_geometric_sequence_constant_l3003_300361

/-- Represents a geometric sequence with sum S_n = 3^n + a -/
structure GeometricSequence where
  a : ℝ  -- The constant term in the sum formula
  -- Sequence definition: a_n = S_n - S_{n-1}
  seq : ℕ → ℝ := λ n => 3^n + a - (3^(n-1) + a)

/-- The first term of the sequence is 2 -/
axiom first_term (s : GeometricSequence) : s.seq 1 = 2

/-- The common ratio of the sequence is 3 -/
axiom common_ratio (s : GeometricSequence) : s.seq 2 = 3 * s.seq 1

/-- Theorem: The value of 'a' in the sum formula S_n = 3^n + a is -1 -/
theorem geometric_sequence_constant (s : GeometricSequence) : s.a = -1 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_constant_l3003_300361


namespace NUMINAMATH_CALUDE_total_turnips_l3003_300341

theorem total_turnips (melanie_turnips benny_turnips : ℕ) 
  (h1 : melanie_turnips = 139) 
  (h2 : benny_turnips = 113) : 
  melanie_turnips + benny_turnips = 252 := by
  sorry

end NUMINAMATH_CALUDE_total_turnips_l3003_300341


namespace NUMINAMATH_CALUDE_stick_difference_l3003_300392

/-- 
Given:
- Dave picked up 14 sticks
- Amy picked up 9 sticks
- There were initially 50 sticks in the yard

Prove that the difference between the number of sticks picked up by Dave and Amy
and the number of sticks left in the yard is 4.
-/
theorem stick_difference (dave_sticks amy_sticks initial_sticks : ℕ) 
  (h1 : dave_sticks = 14)
  (h2 : amy_sticks = 9)
  (h3 : initial_sticks = 50) :
  let picked_up := dave_sticks + amy_sticks
  let left_in_yard := initial_sticks - picked_up
  picked_up - left_in_yard = 4 := by
  sorry

end NUMINAMATH_CALUDE_stick_difference_l3003_300392


namespace NUMINAMATH_CALUDE_trailing_zeros_bound_l3003_300331

/-- The number of trailing zeros in the base-b representation of n! -/
def trailing_zeros (n : ℕ) (b : ℕ) : ℕ := sorry

/-- The largest prime factor of a natural number -/
def largest_prime_factor (b : ℕ) : ℕ := sorry

theorem trailing_zeros_bound {b : ℕ} (hb : b ≥ 2) :
  ∀ n : ℕ, trailing_zeros n b < n / (largest_prime_factor b - 1) := by sorry

end NUMINAMATH_CALUDE_trailing_zeros_bound_l3003_300331


namespace NUMINAMATH_CALUDE_same_color_probability_is_correct_l3003_300396

def white_balls : ℕ := 7
def black_balls : ℕ := 6
def red_balls : ℕ := 2

def total_balls : ℕ := white_balls + black_balls + red_balls

def same_color_probability : ℚ :=
  (Nat.choose white_balls 2 + Nat.choose black_balls 2 + Nat.choose red_balls 2) /
  Nat.choose total_balls 2

theorem same_color_probability_is_correct :
  same_color_probability = 37 / 105 := by
  sorry

end NUMINAMATH_CALUDE_same_color_probability_is_correct_l3003_300396


namespace NUMINAMATH_CALUDE_book_profit_rate_l3003_300385

/-- Given a cost price and a selling price, calculate the rate of profit -/
def rate_of_profit (cost_price selling_price : ℚ) : ℚ :=
  ((selling_price - cost_price) / cost_price) * 100

/-- Theorem: The rate of profit for a book bought at 50 Rs and sold at 70 Rs is 40% -/
theorem book_profit_rate : rate_of_profit 50 70 = 40 := by
  sorry

end NUMINAMATH_CALUDE_book_profit_rate_l3003_300385


namespace NUMINAMATH_CALUDE_light_bulbs_not_broken_l3003_300355

/-- The number of light bulbs not broken in both the foyer and kitchen -/
def num_not_broken (kitchen_total : ℕ) (foyer_broken : ℕ) : ℕ :=
  let kitchen_broken := (3 * kitchen_total) / 5
  let kitchen_not_broken := kitchen_total - kitchen_broken
  let foyer_total := foyer_broken * 3
  let foyer_not_broken := foyer_total - foyer_broken
  kitchen_not_broken + foyer_not_broken

/-- Theorem stating that the number of light bulbs not broken in both the foyer and kitchen is 34 -/
theorem light_bulbs_not_broken :
  num_not_broken 35 10 = 34 := by
  sorry

end NUMINAMATH_CALUDE_light_bulbs_not_broken_l3003_300355


namespace NUMINAMATH_CALUDE_complex_number_problem_l3003_300362

/-- Given a complex number z = b - 2i where b is real, and z / (2 - i) is real,
    prove that z = 4 - 2i and for (z + ai)² to be in the fourth quadrant, -2 < a < 2 -/
theorem complex_number_problem (b : ℝ) (z : ℂ) (h1 : z = b - 2*I) 
  (h2 : ∃ (r : ℝ), z / (2 - I) = r) :
  z = 4 - 2*I ∧ 
  ∀ (a : ℝ), (z + a*I)^2 ∈ {w : ℂ | w.re > 0 ∧ w.im < 0} → -2 < a ∧ a < 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_problem_l3003_300362


namespace NUMINAMATH_CALUDE_complex_magnitude_l3003_300348

theorem complex_magnitude (w z : ℂ) 
  (h1 : w * z + 2 * w - 3 * z = 10 - 6 * Complex.I)
  (h2 : Complex.abs w = 2)
  (h3 : Complex.abs (w + 2) = 3) :
  Complex.abs z = (2 * Real.sqrt 34 - 4) / 5 := by
sorry

end NUMINAMATH_CALUDE_complex_magnitude_l3003_300348


namespace NUMINAMATH_CALUDE_vertex_locus_is_partial_parabola_l3003_300338

/-- The locus of points (x_t, y_t) where x_t = -t / (t^2 + 1) and y_t = c - t^2 / (t^2 + 1),
    as t ranges over all real numbers, forms part, but not all, of a parabola. -/
theorem vertex_locus_is_partial_parabola (c : ℝ) (h : c > 0) :
  ∃ (a b d : ℝ), ∀ (t : ℝ),
    ∃ (x y : ℝ), x = -t / (t^2 + 1) ∧ y = c - t^2 / (t^2 + 1) ∧
    (y = a * x^2 + b * x + d ∨ y < a * x^2 + b * x + d) :=
sorry

end NUMINAMATH_CALUDE_vertex_locus_is_partial_parabola_l3003_300338


namespace NUMINAMATH_CALUDE_power_equality_l3003_300383

theorem power_equality : 32^3 * 8^4 = 2^27 := by
  sorry

end NUMINAMATH_CALUDE_power_equality_l3003_300383


namespace NUMINAMATH_CALUDE_ashton_sheets_l3003_300307

theorem ashton_sheets (jimmy_sheets : ℕ) (tommy_sheets : ℕ) (ashton_sheets : ℕ) : 
  jimmy_sheets = 32 →
  tommy_sheets = jimmy_sheets + 10 →
  jimmy_sheets + ashton_sheets = tommy_sheets + 30 →
  ashton_sheets = 40 :=
by
  sorry

end NUMINAMATH_CALUDE_ashton_sheets_l3003_300307


namespace NUMINAMATH_CALUDE_equation_roots_existence_l3003_300360

-- Define the equation
def equation (x a : ℝ) : Prop := |x^2 - x| - a = 0

-- Define the number of different real roots for a given 'a'
def num_roots (a : ℝ) : ℕ := sorry

-- Theorem statement
theorem equation_roots_existence :
  (∃ a : ℝ, num_roots a = 2) ∧
  (∃ a : ℝ, num_roots a = 3) ∧
  (∃ a : ℝ, num_roots a = 4) ∧
  (¬ ∃ a : ℝ, num_roots a = 6) :=
sorry

end NUMINAMATH_CALUDE_equation_roots_existence_l3003_300360


namespace NUMINAMATH_CALUDE_parabola_shift_l3003_300393

/-- Given a parabola y = x^2, shifting it 3 units right and 4 units up results in y = (x-3)^2 + 4 -/
theorem parabola_shift (x y : ℝ) : 
  (y = x^2) → 
  (∃ (y' : ℝ → ℝ), 
    (∀ x, y' x = (x - 3)^2) ∧ 
    (∀ x, y' x + 4 = (x - 3)^2 + 4)) := by
  sorry


end NUMINAMATH_CALUDE_parabola_shift_l3003_300393


namespace NUMINAMATH_CALUDE_paths_A_to_D_l3003_300375

/-- Represents a point in the graph --/
inductive Point : Type
| A : Point
| B : Point
| C : Point
| D : Point

/-- Represents a direct path between two points --/
inductive DirectPath : Point → Point → Type
| AB : DirectPath Point.A Point.B
| BC : DirectPath Point.B Point.C
| CD : DirectPath Point.C Point.D
| AC : DirectPath Point.A Point.C
| BD : DirectPath Point.B Point.D

/-- Counts the number of paths between two points --/
def countPaths (start finish : Point) : ℕ :=
  sorry

/-- The main theorem stating that there are 12 paths from A to D --/
theorem paths_A_to_D :
  countPaths Point.A Point.D = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_paths_A_to_D_l3003_300375


namespace NUMINAMATH_CALUDE_part_I_part_II_l3003_300377

-- Define sets A and B
def A : Set ℝ := {x | -1 < x ∧ x < 3}
def B (a b : ℝ) : Set ℝ := {x | x^2 - a*x + b < 0}

-- Theorem for part I
theorem part_I : ∀ a b : ℝ, A = B a b → a = 2 ∧ b = -3 := by sorry

-- Theorem for part II
theorem part_II : ∀ a : ℝ, (A ∩ B a 3) ⊇ B a 3 → -2 * Real.sqrt 3 ≤ a ∧ a ≤ 4 := by sorry

end NUMINAMATH_CALUDE_part_I_part_II_l3003_300377


namespace NUMINAMATH_CALUDE_profit_increase_after_cost_decrease_l3003_300336

theorem profit_increase_after_cost_decrease (x y : ℝ) (a : ℝ) 
  (h1 : y - x = x * (a / 100))  -- Initial profit percentage
  (h2 : y - 0.9 * x = 0.9 * x * ((a + 20) / 100))  -- New profit percentage
  : a = 80 := by
sorry

end NUMINAMATH_CALUDE_profit_increase_after_cost_decrease_l3003_300336


namespace NUMINAMATH_CALUDE_vector_projection_l3003_300382

/-- Given two 2D vectors a and b, prove that the projection of a onto b is √13/13 -/
theorem vector_projection (a b : ℝ × ℝ) : 
  a = (-2, 1) → b = (-2, -3) → 
  (a.1 * b.1 + a.2 * b.2) / Real.sqrt (b.1^2 + b.2^2) = Real.sqrt 13 / 13 := by
  sorry

end NUMINAMATH_CALUDE_vector_projection_l3003_300382


namespace NUMINAMATH_CALUDE_max_consecutive_sum_l3003_300379

theorem max_consecutive_sum (n : ℕ) : (n * (n + 1)) / 2 ≤ 1000 ↔ n ≤ 44 := by sorry

end NUMINAMATH_CALUDE_max_consecutive_sum_l3003_300379


namespace NUMINAMATH_CALUDE_area_between_squares_l3003_300303

/-- The area of the region between two squares, where a smaller square is entirely contained within a larger square -/
theorem area_between_squares (larger_side smaller_side : ℝ) 
  (h1 : larger_side = 8) 
  (h2 : smaller_side = 4) 
  (h3 : smaller_side ≤ larger_side) : 
  larger_side ^ 2 - smaller_side ^ 2 = 48 := by
  sorry

end NUMINAMATH_CALUDE_area_between_squares_l3003_300303


namespace NUMINAMATH_CALUDE_count_perfect_square_factors_l3003_300315

/-- The number of positive perfect square factors of (2^14)(3^18)(7^21) -/
def perfect_square_factors : ℕ := sorry

/-- The given number -/
def given_number : ℕ := 2^14 * 3^18 * 7^21

theorem count_perfect_square_factors :
  perfect_square_factors = 880 :=
sorry

end NUMINAMATH_CALUDE_count_perfect_square_factors_l3003_300315


namespace NUMINAMATH_CALUDE_empty_solution_set_implies_a_range_l3003_300316

/-- The function f(x) = x^2 + (1-a)x - a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + (1-a)*x - a

/-- Theorem stating that if the solution set of f(f(x)) < 0 is empty,
    then -3 ≤ a ≤ 2√2 - 3 -/
theorem empty_solution_set_implies_a_range (a : ℝ) :
  (∀ x : ℝ, f a (f a x) ≥ 0) → -3 ≤ a ∧ a ≤ 2 * Real.sqrt 2 - 3 :=
by sorry


end NUMINAMATH_CALUDE_empty_solution_set_implies_a_range_l3003_300316


namespace NUMINAMATH_CALUDE_interest_rate_difference_l3003_300359

theorem interest_rate_difference 
  (principal : ℝ) 
  (time : ℝ) 
  (interest_difference : ℝ) 
  (h1 : principal = 750) 
  (h2 : time = 2) 
  (h3 : interest_difference = 60) : 
  ∃ (original_rate higher_rate : ℝ),
    principal * higher_rate * time / 100 - principal * original_rate * time / 100 = interest_difference ∧ 
    higher_rate - original_rate = 4 := by
sorry

end NUMINAMATH_CALUDE_interest_rate_difference_l3003_300359


namespace NUMINAMATH_CALUDE_justin_bought_two_striped_jerseys_l3003_300380

-- Define the cost of each type of jersey
def long_sleeved_cost : ℕ := 15
def striped_cost : ℕ := 10

-- Define the number of long-sleeved jerseys bought
def long_sleeved_count : ℕ := 4

-- Define the total amount spent
def total_spent : ℕ := 80

-- Define the number of striped jerseys as a function
def striped_jerseys : ℕ := (total_spent - long_sleeved_cost * long_sleeved_count) / striped_cost

-- Theorem to prove
theorem justin_bought_two_striped_jerseys : striped_jerseys = 2 := by
  sorry

end NUMINAMATH_CALUDE_justin_bought_two_striped_jerseys_l3003_300380


namespace NUMINAMATH_CALUDE_consecutive_pair_divisible_by_five_l3003_300309

theorem consecutive_pair_divisible_by_five (a b : ℕ) : 
  a < 1500 → 
  b < 1500 → 
  b = a + 1 → 
  (a + b) % 5 = 0 → 
  a = 57 → 
  b = 58 := by
sorry

end NUMINAMATH_CALUDE_consecutive_pair_divisible_by_five_l3003_300309


namespace NUMINAMATH_CALUDE_unique_solution_l3003_300310

/-- Define a sequence of 100 real numbers satisfying given conditions -/
def SequenceOfHundred (a : Fin 100 → ℝ) : Prop :=
  (∀ i : Fin 99, a i - 4 * a (i + 1) + 3 * a (i + 2) ≥ 0) ∧
  (a 99 - 4 * a 0 + 3 * a 1 ≥ 0) ∧
  (a 0 = 1)

/-- Theorem stating that the sequence of all 1's is the unique solution -/
theorem unique_solution (a : Fin 100 → ℝ) (h : SequenceOfHundred a) :
  ∀ i : Fin 100, a i = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l3003_300310


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l3003_300334

def vector_a (x : ℝ) : Fin 2 → ℝ := ![2, x]
def vector_b (x : ℝ) : Fin 2 → ℝ := ![3, x + 1]

theorem parallel_vectors_x_value :
  ∀ x : ℝ, (∃ k : ℝ, k ≠ 0 ∧ vector_a x = k • vector_b x) → x = 2 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l3003_300334


namespace NUMINAMATH_CALUDE_playground_teachers_l3003_300365

theorem playground_teachers (boys girls : ℕ) (h1 : boys = 57) (h2 : girls = 82)
  (h3 : girls = boys + teachers + 13) : teachers = 12 := by
  sorry

end NUMINAMATH_CALUDE_playground_teachers_l3003_300365


namespace NUMINAMATH_CALUDE_trigonometric_expression_equality_l3003_300339

theorem trigonometric_expression_equality : 
  (2 * Real.sin (25 * π / 180) ^ 2 - 1) / (Real.sin (20 * π / 180) * Real.cos (20 * π / 180)) = -2 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_expression_equality_l3003_300339


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l3003_300325

theorem arithmetic_calculation : 12 - 10 + 15 / 5 * 8 + 7 - 6 * 4 + 3 - 2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l3003_300325


namespace NUMINAMATH_CALUDE_mother_age_l3003_300351

theorem mother_age (eunji_age_now : ℕ) (eunji_age_then : ℕ) (mother_age_then : ℕ) :
  eunji_age_now = 16 →
  eunji_age_then = 8 →
  mother_age_then = 35 →
  mother_age_then + (eunji_age_now - eunji_age_then) = 43 := by
  sorry

#check mother_age

end NUMINAMATH_CALUDE_mother_age_l3003_300351


namespace NUMINAMATH_CALUDE_bisecting_line_slope_intercept_sum_l3003_300311

/-- Triangle ABC with vertices A(0, 8), B(2, 0), C(8, 0) -/
structure Triangle :=
  (A : ℝ × ℝ)
  (B : ℝ × ℝ)
  (C : ℝ × ℝ)

/-- A line represented by its slope and y-intercept -/
structure Line :=
  (slope : ℝ)
  (y_intercept : ℝ)

/-- The area of a triangle -/
def area (t : Triangle) : ℝ := sorry

/-- Check if a line bisects the area of a triangle -/
def bisects_area (l : Line) (t : Triangle) : Prop := sorry

/-- The line through point B that bisects the area of the triangle -/
def bisecting_line (t : Triangle) : Line := sorry

/-- The theorem to be proved -/
theorem bisecting_line_slope_intercept_sum (t : Triangle) :
  t.A = (0, 8) ∧ t.B = (2, 0) ∧ t.C = (8, 0) →
  let l := bisecting_line t
  l.slope + l.y_intercept = -2 := by sorry

end NUMINAMATH_CALUDE_bisecting_line_slope_intercept_sum_l3003_300311


namespace NUMINAMATH_CALUDE_distance_equals_speed_times_time_l3003_300374

/-- The distance between Emily's house and Timothy's house -/
def distance : ℝ := 10

/-- Emily's speed in miles per hour -/
def speed : ℝ := 5

/-- Time taken for Emily to reach Timothy's house in hours -/
def time : ℝ := 2

/-- Theorem stating that the distance is equal to speed multiplied by time -/
theorem distance_equals_speed_times_time : distance = speed * time := by
  sorry

end NUMINAMATH_CALUDE_distance_equals_speed_times_time_l3003_300374


namespace NUMINAMATH_CALUDE_tetrahedron_volume_l3003_300384

/-- Given a tetrahedron with two faces of areas S₁ and S₂, sharing a common edge of length a,
    and with a dihedral angle α between these faces, the volume V of the tetrahedron is
    (2 * S₁ * S₂ * sin α) / (3 * a) -/
theorem tetrahedron_volume
  (S₁ S₂ a : ℝ)
  (α : ℝ)
  (h₁ : S₁ > 0)
  (h₂ : S₂ > 0)
  (h₃ : a > 0)
  (h₄ : 0 < α ∧ α < π) :
  ∃ V : ℝ, V = (2 * S₁ * S₂ * Real.sin α) / (3 * a) ∧ V > 0 := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_volume_l3003_300384


namespace NUMINAMATH_CALUDE_min_distance_sum_l3003_300343

theorem min_distance_sum (x y : ℝ) :
  Real.sqrt ((x - 1)^2 + y^2) + Real.sqrt ((x + 1)^2 + y^2) + |2 - y| ≥ 2 + Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_min_distance_sum_l3003_300343


namespace NUMINAMATH_CALUDE_problem_solution_l3003_300367

def g (n : ℤ) : ℤ :=
  if n % 2 = 1 then n + 5 else n / 2

theorem problem_solution (m : ℤ) (h_odd : m % 2 = 1) (h_eq : g (g (g m)) = 39) : m = 63 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3003_300367


namespace NUMINAMATH_CALUDE_distance_between_cities_l3003_300358

/-- The distance between City A and City B in miles -/
def distance : ℝ := 427.5

/-- The initial travel time from City A to City B in hours -/
def initial_time_AB : ℝ := 6

/-- The initial travel time from City B to City A in hours -/
def initial_time_BA : ℝ := 4.5

/-- The time saved on each trip in hours -/
def time_saved : ℝ := 0.5

/-- The average speed for the round trip after saving time in miles per hour -/
def average_speed : ℝ := 90

theorem distance_between_cities :
  2 * distance / (initial_time_AB + initial_time_BA - 2 * time_saved) = average_speed :=
sorry

end NUMINAMATH_CALUDE_distance_between_cities_l3003_300358


namespace NUMINAMATH_CALUDE_point_A_on_line_l_l3003_300337

/-- A line passing through the origin with slope -2 -/
def line_l (x y : ℝ) : Prop := y = -2 * x

/-- The point (1, -2) -/
def point_A : ℝ × ℝ := (1, -2)

/-- Theorem: The point (1, -2) lies on the line l -/
theorem point_A_on_line_l : line_l point_A.1 point_A.2 := by sorry

end NUMINAMATH_CALUDE_point_A_on_line_l_l3003_300337


namespace NUMINAMATH_CALUDE_probability_of_y_selection_l3003_300387

theorem probability_of_y_selection 
  (p_x : ℝ) 
  (p_both : ℝ) 
  (h1 : p_x = 1 / 7)
  (h2 : p_both = 0.05714285714285714) :
  p_both / p_x = 0.4 := by
sorry

end NUMINAMATH_CALUDE_probability_of_y_selection_l3003_300387


namespace NUMINAMATH_CALUDE_smallest_number_proof_l3003_300345

/-- The smallest natural number divisible by 21 with exactly 105 distinct divisors -/
def smallest_number_with_properties : ℕ := 254016

/-- The number of distinct divisors of n -/
def num_divisors (n : ℕ) : ℕ := sorry

theorem smallest_number_proof :
  (smallest_number_with_properties % 21 = 0) ∧
  (num_divisors smallest_number_with_properties = 105) ∧
  (∀ m : ℕ, m < smallest_number_with_properties →
    ¬(m % 21 = 0 ∧ num_divisors m = 105)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_proof_l3003_300345


namespace NUMINAMATH_CALUDE_abs_even_and_increasing_l3003_300372

-- Define the function
def f (x : ℝ) : ℝ := |x|

-- State the theorem
theorem abs_even_and_increasing : 
  (∀ x : ℝ, f x = f (-x)) ∧ 
  (∀ x y : ℝ, 0 < x → x < y → f x ≤ f y) :=
by sorry

end NUMINAMATH_CALUDE_abs_even_and_increasing_l3003_300372


namespace NUMINAMATH_CALUDE_complex_sum_problem_l3003_300390

theorem complex_sum_problem (u v w x y z : ℂ) : 
  v = 2 →
  y = -u - w →
  u + v * I + w + x * I + y + z * I = 2 * I →
  x + z = 0 := by
sorry

end NUMINAMATH_CALUDE_complex_sum_problem_l3003_300390


namespace NUMINAMATH_CALUDE_envelope_addressing_equation_verify_envelope_addressing_equation_l3003_300369

/-- The equation representing the relationship between the time taken by two machines to address 1000 envelopes -/
theorem envelope_addressing_equation (x : ℝ) : x > 0 → (1 / 12 + 1 / x = 1 / 3) ↔ 
  (1000 / 12 + 1000 / x = 1000 / 3) := by sorry

/-- Verifies that the equation holds for the given conditions -/
theorem verify_envelope_addressing_equation :
  ∃ x : ℝ, x > 0 ∧ (1 / 12 + 1 / x = 1 / 3) ∧ (1000 / 12 + 1000 / x = 1000 / 3) := by sorry

end NUMINAMATH_CALUDE_envelope_addressing_equation_verify_envelope_addressing_equation_l3003_300369


namespace NUMINAMATH_CALUDE_unique_four_digit_square_l3003_300305

/-- A function that checks if a number is a four-digit number -/
def isFourDigit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

/-- A function that checks if a number has its first two digits equal -/
def firstTwoDigitsEqual (n : ℕ) : Prop :=
  (n / 1000 = (n / 100) % 10)

/-- A function that checks if a number has its last two digits equal -/
def lastTwoDigitsEqual (n : ℕ) : Prop :=
  ((n / 10) % 10 = n % 10)

/-- The main theorem stating that 7744 is the only four-digit perfect square
    with equal first two digits and equal last two digits -/
theorem unique_four_digit_square :
  ∀ n : ℕ, isFourDigit n ∧ ∃ k : ℕ, n = k^2 ∧ firstTwoDigitsEqual n ∧ lastTwoDigitsEqual n
  ↔ n = 7744 := by
  sorry

end NUMINAMATH_CALUDE_unique_four_digit_square_l3003_300305


namespace NUMINAMATH_CALUDE_area_between_concentric_circles_l3003_300356

theorem area_between_concentric_circles :
  ∀ (r : ℝ),
  r > 0 →
  3 * r - r = 3 →
  π * (3 * r)^2 - π * r^2 = 18 * π := by
sorry

end NUMINAMATH_CALUDE_area_between_concentric_circles_l3003_300356


namespace NUMINAMATH_CALUDE_complement_of_union_sets_l3003_300391

open Set

theorem complement_of_union_sets (A B : Set ℝ) :
  A = {x : ℝ | x < 1} →
  B = {x : ℝ | x > 3} →
  (A ∪ B)ᶜ = {x : ℝ | 1 ≤ x ∧ x ≤ 3} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_union_sets_l3003_300391


namespace NUMINAMATH_CALUDE_average_running_time_l3003_300329

theorem average_running_time (f : ℕ) : 
  let third_graders := 9 * f
  let fourth_graders := 3 * f
  let fifth_graders := f
  let total_students := third_graders + fourth_graders + fifth_graders
  let total_minutes := 10 * third_graders + 18 * fourth_graders + 12 * fifth_graders
  (total_minutes : ℚ) / total_students = 12 := by
sorry

end NUMINAMATH_CALUDE_average_running_time_l3003_300329


namespace NUMINAMATH_CALUDE_franks_earnings_l3003_300395

/-- Represents Frank's work schedule and pay rates -/
structure WorkSchedule where
  totalHours : ℕ
  days : ℕ
  regularRate : ℚ
  overtimeRate : ℚ
  day1Hours : ℕ
  day2Hours : ℕ
  day3Hours : ℕ
  day4Hours : ℕ

/-- Calculates the total earnings based on the work schedule -/
def calculateEarnings (schedule : WorkSchedule) : ℚ :=
  let regularHours := min schedule.totalHours (schedule.days * 8)
  let overtimeHours := schedule.totalHours - regularHours
  regularHours * schedule.regularRate + overtimeHours * schedule.overtimeRate

/-- Frank's work schedule for the week -/
def franksSchedule : WorkSchedule :=
  { totalHours := 32
  , days := 4
  , regularRate := 15
  , overtimeRate := 22.5
  , day1Hours := 12
  , day2Hours := 8
  , day3Hours := 8
  , day4Hours := 12
  }

/-- Theorem stating that Frank's total earnings for the week are $660 -/
theorem franks_earnings : calculateEarnings franksSchedule = 660 := by
  sorry

end NUMINAMATH_CALUDE_franks_earnings_l3003_300395


namespace NUMINAMATH_CALUDE_pizza_toppings_l3003_300370

theorem pizza_toppings (total_slices cheese_slices onion_slices : ℕ) 
  (h_total : total_slices = 16)
  (h_cheese : cheese_slices = 9)
  (h_onion : onion_slices = 13)
  (h_at_least_one : cheese_slices + onion_slices ≥ total_slices) :
  ∃ (both_toppings : ℕ), 
    both_toppings = cheese_slices + onion_slices - total_slices ∧
    both_toppings = 6 :=
by sorry

end NUMINAMATH_CALUDE_pizza_toppings_l3003_300370


namespace NUMINAMATH_CALUDE_circle_equation_and_intersection_range_l3003_300327

-- Define the circle C
def circle_C (x y : ℝ) : Prop :=
  ∃ (h : ℝ), (x - h)^2 + (y - (3*h - 5))^2 = 1 ∧ (3 - h)^2 + (3 - (3*h - 5))^2 = 1 ∧ (2 - h)^2 + (4 - (3*h - 5))^2 = 1

-- Define the circle with diameter PQ
def circle_PQ (m x y : ℝ) : Prop := x^2 + y^2 = m^2

theorem circle_equation_and_intersection_range :
  ∃ (h : ℝ), 
    (∀ x y : ℝ, circle_C x y ↔ (x - 3)^2 + (y - 4)^2 = 1) ∧
    (∀ m : ℝ, m > 0 → 
      (∃ x y : ℝ, circle_C x y ∧ circle_PQ m x y) ↔ 
      (4 ≤ m ∧ m ≤ 6)) :=
sorry

end NUMINAMATH_CALUDE_circle_equation_and_intersection_range_l3003_300327


namespace NUMINAMATH_CALUDE_prob_odd_fair_die_l3003_300344

def die_outcomes : Finset Nat := {1, 2, 3, 4, 5, 6}
def odd_outcomes : Finset Nat := {1, 3, 5}

theorem prob_odd_fair_die :
  (Finset.card odd_outcomes : ℚ) / (Finset.card die_outcomes : ℚ) = 1 / 2 :=
by sorry

end NUMINAMATH_CALUDE_prob_odd_fair_die_l3003_300344


namespace NUMINAMATH_CALUDE_parallel_postulate_introduction_l3003_300352

-- Define the concept of a geometric theorem
def GeometricTheorem : Type := Unit

-- Define the concept of Euclid's parallel postulate
def EuclidParallelPostulate : Type := Unit

-- Define the property of a theorem being independent of the parallel postulate
def independent (t : GeometricTheorem) (p : EuclidParallelPostulate) : Prop := True

-- Define the concept of introducing a postulate in geometry
def introduced_later (p : EuclidParallelPostulate) : Prop := True

theorem parallel_postulate_introduction 
  (many_theorems : Set GeometricTheorem)
  (parallel_postulate : EuclidParallelPostulate)
  (h : ∀ t ∈ many_theorems, independent t parallel_postulate) :
  introduced_later parallel_postulate :=
by
  sorry

#check parallel_postulate_introduction

end NUMINAMATH_CALUDE_parallel_postulate_introduction_l3003_300352


namespace NUMINAMATH_CALUDE_solution_set_equality_l3003_300349

def f (x : ℤ) : ℤ := 2 * x^2 + x - 6

def is_prime_power (n : ℤ) : Prop :=
  ∃ (p : ℕ) (k : ℕ+), Nat.Prime p ∧ n = (p : ℤ) ^ (k : ℕ)

theorem solution_set_equality : 
  {x : ℤ | ∃ (y : ℤ), y > 0 ∧ is_prime_power y ∧ f x = y} = {-3, 2, 5} := by sorry

end NUMINAMATH_CALUDE_solution_set_equality_l3003_300349


namespace NUMINAMATH_CALUDE_reflected_ray_equation_l3003_300347

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space using the general form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The incident point of the light ray -/
def P : Point := ⟨5, 3⟩

/-- The point where the light ray intersects the x-axis -/
def Q : Point := ⟨2, 0⟩

/-- Function to calculate the reflected point across the x-axis -/
def reflect_across_x_axis (p : Point) : Point :=
  ⟨p.x, -p.y⟩

/-- The reflected point of P across the x-axis -/
def P' : Point := reflect_across_x_axis P

/-- Function to create a line from two points -/
def line_from_points (p1 p2 : Point) : Line :=
  let a := p2.y - p1.y
  let b := p1.x - p2.x
  let c := p2.x * p1.y - p1.x * p2.y
  ⟨a, b, c⟩

/-- The reflected ray line -/
def reflected_ray : Line := line_from_points Q P'

/-- Theorem stating that the reflected ray line has the equation x + y - 2 = 0 -/
theorem reflected_ray_equation :
  reflected_ray = ⟨1, 1, -2⟩ := by sorry

end NUMINAMATH_CALUDE_reflected_ray_equation_l3003_300347


namespace NUMINAMATH_CALUDE_final_pen_count_l3003_300394

def pen_collection (initial : ℕ) (mike_gave : ℕ) (sharon_took : ℕ) : ℕ :=
  ((initial + mike_gave) * 2) - sharon_took

theorem final_pen_count : pen_collection 25 22 19 = 75 := by
  sorry

end NUMINAMATH_CALUDE_final_pen_count_l3003_300394


namespace NUMINAMATH_CALUDE_round_repeating_decimal_to_hundredth_l3003_300306

/-- The repeating decimal 67.673673673... -/
def repeating_decimal : ℚ := 67 + 673 / 999

/-- Rounding a rational number to the nearest hundredth -/
def round_to_hundredth (q : ℚ) : ℚ := 
  ⌊q * 100 + 1/2⌋ / 100

/-- Theorem: Rounding 67.673673673... to the nearest hundredth equals 67.67 -/
theorem round_repeating_decimal_to_hundredth : 
  round_to_hundredth repeating_decimal = 6767 / 100 := by sorry

end NUMINAMATH_CALUDE_round_repeating_decimal_to_hundredth_l3003_300306


namespace NUMINAMATH_CALUDE_triangle_mapping_l3003_300333

theorem triangle_mapping :
  ∃ (f : ℂ → ℂ), 
    (∀ z w, f z = w ↔ w = (1 + Complex.I) * (1 - z)) ∧
    f 0 = 1 + Complex.I ∧
    f 1 = 0 ∧
    f Complex.I = 2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_mapping_l3003_300333


namespace NUMINAMATH_CALUDE_larger_number_proof_l3003_300317

theorem larger_number_proof (A B : ℝ) (h1 : A - B = 1650) (h2 : 0.075 * A = 0.125 * B) : A = 4125 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_proof_l3003_300317


namespace NUMINAMATH_CALUDE_fraction_equation_solution_l3003_300376

theorem fraction_equation_solution (n : ℚ) : 
  (1 : ℚ) / (n + 2) + 3 / (n + 2) + n / (n + 2) = 4 → n = -4/3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equation_solution_l3003_300376


namespace NUMINAMATH_CALUDE_change_calculation_l3003_300308

/-- Given the cost of milk and water, and the amount paid, calculate the change received. -/
theorem change_calculation (milk_cost water_cost paid : ℕ) 
  (h_milk : milk_cost = 350)
  (h_water : water_cost = 500)
  (h_paid : paid = 1000) :
  paid - (milk_cost + water_cost) = 150 := by
  sorry

end NUMINAMATH_CALUDE_change_calculation_l3003_300308


namespace NUMINAMATH_CALUDE_z_extrema_l3003_300318

-- Define the function z(x,y)
def z (x y : ℝ) : ℝ := 2 * x^3 - 6 * x * y + 3 * y^2

-- Define the region
def R : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 ≥ 0 ∧ p.2 ≥ 0 ∧ p.2 ≤ 2 ∧ p.2 ≤ p.1^2 / 2}

-- State the theorem
theorem z_extrema :
  ∃ (max min : ℝ), max = 12 ∧ min = -1 ∧
  (∀ p ∈ R, z p.1 p.2 ≤ max) ∧
  (∀ p ∈ R, z p.1 p.2 ≥ min) ∧
  (∃ p ∈ R, z p.1 p.2 = max) ∧
  (∃ p ∈ R, z p.1 p.2 = min) :=
sorry

end NUMINAMATH_CALUDE_z_extrema_l3003_300318


namespace NUMINAMATH_CALUDE_line_through_circle_center_l3003_300312

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 4*y = 0

-- Define the line equation
def line_equation (x y a : ℝ) : Prop := 3*x + y + a = 0

-- Define the center of a circle
def is_center (h k : ℝ) : Prop := ∀ x y, circle_equation x y ↔ (x - h)^2 + (y - k)^2 = 5

-- Theorem statement
theorem line_through_circle_center (a : ℝ) : 
  (∃ h k, is_center h k ∧ line_equation h k a) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_line_through_circle_center_l3003_300312


namespace NUMINAMATH_CALUDE_polynomial_simplification_l3003_300350

theorem polynomial_simplification (x : ℝ) :
  x * (x * (x * (3 - x) - 6) + 12) + 2 = -x^4 + 3*x^3 - 6*x^2 + 12*x + 2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l3003_300350


namespace NUMINAMATH_CALUDE_unique_solution_and_sum_l3003_300319

theorem unique_solution_and_sum : ∃! (a b c : ℕ), 
  ({a, b, c} : Set ℕ) = {0, 1, 2} ∧ 
  (((a ≠ 2) ∧ (b ≠ 2) ∧ (c = 0)) ∨ 
   ((a = 2) ∧ (b = 2) ∧ (c ≠ 0)) ∨ 
   ((a = 2) ∧ (b ≠ 2) ∧ (c ≠ 0))) ∧
  a = 2 ∧ b = 0 ∧ c = 1 ∧ 
  100 * c + 10 * b + a = 102 :=
sorry

end NUMINAMATH_CALUDE_unique_solution_and_sum_l3003_300319


namespace NUMINAMATH_CALUDE_population_growth_l3003_300371

theorem population_growth (p₀ : ℝ) : 
  let p₁ := p₀ * 1.1
  let p₂ := p₁ * 1.2
  let p₃ := p₂ * 1.3
  (p₃ - p₀) / p₀ * 100 = 71.6 := by
sorry

end NUMINAMATH_CALUDE_population_growth_l3003_300371


namespace NUMINAMATH_CALUDE_tenths_place_of_five_twelfths_l3003_300330

theorem tenths_place_of_five_twelfths (ε : ℚ) : 
  ε = 5 / 12 → 
  ∃ (n : ℕ) (r : ℚ), ε = (4 : ℚ) / 10 + n / 100 + r ∧ 0 ≤ r ∧ r < 1 / 100 :=
sorry

end NUMINAMATH_CALUDE_tenths_place_of_five_twelfths_l3003_300330


namespace NUMINAMATH_CALUDE_total_books_on_cart_l3003_300357

/-- The number of books Nancy shelved from the cart -/
structure BookCart where
  top_history : ℕ
  top_romance : ℕ
  top_poetry : ℕ
  bottom_western : ℕ
  bottom_biography : ℕ
  bottom_scifi : ℕ
  bottom_culinary : ℕ

/-- The theorem stating the total number of books on the cart -/
theorem total_books_on_cart (cart : BookCart) : 
  cart.top_history = 12 →
  cart.top_romance = 8 →
  cart.top_poetry = 4 →
  cart.bottom_western = 5 →
  cart.bottom_biography = 6 →
  cart.bottom_scifi = 3 →
  cart.bottom_culinary = 2 →
  ∃ (total : ℕ), total = 88 ∧ 
    total = cart.top_history + cart.top_romance + cart.top_poetry + 
            (cart.bottom_western + cart.bottom_biography + cart.bottom_scifi + cart.bottom_culinary) * 4 :=
by sorry


end NUMINAMATH_CALUDE_total_books_on_cart_l3003_300357


namespace NUMINAMATH_CALUDE_base12_divisibility_rule_l3003_300388

/-- 
Represents a number in base-12 as a list of digits, 
where each digit is between 0 and 11 (inclusive).
--/
def Base12Number := List Nat

/-- Converts a Base12Number to its decimal representation. --/
def toDecimal (n : Base12Number) : Nat :=
  n.enum.foldl (fun acc (i, digit) => acc + digit * (12 ^ i)) 0

/-- Calculates the sum of digits in a Base12Number. --/
def digitSum (n : Base12Number) : Nat :=
  n.sum

theorem base12_divisibility_rule (n : Base12Number) :
  11 ∣ digitSum n → 11 ∣ toDecimal n := by sorry

end NUMINAMATH_CALUDE_base12_divisibility_rule_l3003_300388


namespace NUMINAMATH_CALUDE_inner_square_area_ratio_is_one_fourth_l3003_300364

/-- The ratio of the area of a square formed by connecting the center of a larger square
    to the midpoints of its sides, to the area of the larger square. -/
def inner_square_area_ratio : ℚ := 1 / 4

/-- Theorem stating that the ratio of the area of the inner square to the outer square is 1/4. -/
theorem inner_square_area_ratio_is_one_fourth :
  inner_square_area_ratio = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_inner_square_area_ratio_is_one_fourth_l3003_300364


namespace NUMINAMATH_CALUDE_jake_eighth_week_hours_l3003_300328

def hours_worked : List ℕ := [14, 9, 12, 15, 11, 13, 10]
def total_weeks : ℕ := 8
def required_average : ℕ := 12

theorem jake_eighth_week_hours :
  ∃ (x : ℕ), 
    (List.sum hours_worked + x) / total_weeks = required_average ∧
    x = 12 := by
  sorry

end NUMINAMATH_CALUDE_jake_eighth_week_hours_l3003_300328
