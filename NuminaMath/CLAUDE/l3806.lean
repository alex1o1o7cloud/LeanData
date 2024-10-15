import Mathlib

namespace NUMINAMATH_CALUDE_hidden_cave_inventory_sum_l3806_380624

/-- Converts a number from base 5 to base 10 --/
def base5ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (5 ^ i)) 0

/-- The problem statement --/
theorem hidden_cave_inventory_sum : 
  let artifact := base5ToBase10 [3, 1, 2, 4]
  let sculpture := base5ToBase10 [1, 3, 4, 2]
  let coins := base5ToBase10 [3, 1, 2]
  artifact + sculpture + coins = 982 := by
sorry

end NUMINAMATH_CALUDE_hidden_cave_inventory_sum_l3806_380624


namespace NUMINAMATH_CALUDE_square_division_l3806_380684

theorem square_division (a : ℕ) (h1 : a > 0) :
  (a * a = 25) ∧
  (∃ b : ℕ, b > 0 ∧ a * a = 24 * 1 * 1 + b * b) ∧
  (a = 5) :=
by sorry

end NUMINAMATH_CALUDE_square_division_l3806_380684


namespace NUMINAMATH_CALUDE_min_abs_z_on_line_segment_l3806_380629

theorem min_abs_z_on_line_segment (z : ℂ) (h : Complex.abs (z - 6) + Complex.abs (z - Complex.I * 5) = 7) :
  ∃ (w : ℂ), Complex.abs w ≤ Complex.abs z ∧ Complex.abs (w - 6) + Complex.abs (w - Complex.I * 5) = 7 ∧ Complex.abs w = 30 / 7 :=
by sorry

end NUMINAMATH_CALUDE_min_abs_z_on_line_segment_l3806_380629


namespace NUMINAMATH_CALUDE_factory_sampling_is_systematic_l3806_380652

/-- Represents a sampling method --/
inductive SamplingMethod
  | Stratified
  | SimpleRandom
  | Systematic
  | Other

/-- Represents a sampling process --/
structure SamplingProcess where
  interval : ℕ  -- Time interval between samples
  continuous : Bool  -- Whether the process is continuous

/-- Determines if a sampling process is systematic --/
def is_systematic (process : SamplingProcess) : Prop :=
  process.interval > 0 ∧ process.continuous

/-- Theorem: A sampling process with a fixed positive time interval 
    from a continuous process is systematic sampling --/
theorem factory_sampling_is_systematic 
  (process : SamplingProcess) 
  (h1 : process.interval = 10)  -- 10-minute interval
  (h2 : process.continuous = true)  -- Conveyor belt implies continuous process
  : is_systematic process ∧ 
    (λ method : SamplingMethod => 
      is_systematic process → method = SamplingMethod.Systematic) SamplingMethod.Systematic := by
  sorry


end NUMINAMATH_CALUDE_factory_sampling_is_systematic_l3806_380652


namespace NUMINAMATH_CALUDE_parallelogram_area_example_l3806_380678

/-- The area of a parallelogram with given base and height -/
def parallelogram_area (base height : ℝ) : ℝ := base * height

/-- Theorem: The area of a parallelogram with base 26 cm and height 14 cm is 364 square centimeters -/
theorem parallelogram_area_example : parallelogram_area 26 14 = 364 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_example_l3806_380678


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_ratio_l3806_380695

theorem quadratic_equation_roots_ratio (k : ℝ) : 
  (∃ x y : ℝ, x ≠ 0 ∧ y ≠ 0 ∧ x / y = 3 ∧ 
   x^2 + 10*x + k = 0 ∧ y^2 + 10*y + k = 0) → 
  k = 18.75 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_ratio_l3806_380695


namespace NUMINAMATH_CALUDE_triangle_with_specific_properties_l3806_380643

/-- Represents a triangle with side lengths and circumradius -/
structure Triangle where
  a : ℕ
  b : ℕ
  c : ℕ
  r : ℕ

/-- Represents the distances from circumcenter to sides -/
structure CircumcenterDistances where
  d : ℕ
  e : ℕ

/-- The theorem statement -/
theorem triangle_with_specific_properties 
  (t : Triangle) 
  (dist : CircumcenterDistances) 
  (h1 : t.r = 25)
  (h2 : t.a > t.b)
  (h3 : t.a^2 + 4 * dist.d^2 = 2500)
  (h4 : t.b^2 + 4 * dist.e^2 = 2500) :
  t.a = 15 ∧ t.b = 7 ∧ t.c = 20 :=
by sorry

end NUMINAMATH_CALUDE_triangle_with_specific_properties_l3806_380643


namespace NUMINAMATH_CALUDE_point_A_not_in_square_l3806_380622

-- Define the points
def A : ℝ × ℝ := (-1, 3)
def B : ℝ × ℝ := (0, -4)
def C : ℝ × ℝ := (-2, -1)
def D : ℝ × ℝ := (1, 1)
def E : ℝ × ℝ := (3, -2)

-- Define a function to calculate the squared distance between two points
def squared_distance (p q : ℝ × ℝ) : ℝ :=
  (p.1 - q.1)^2 + (p.2 - q.2)^2

-- Define what it means for four points to form a square
def is_square (p q r s : ℝ × ℝ) : Prop :=
  let sides := [squared_distance p q, squared_distance q r, squared_distance r s, squared_distance s p]
  let diagonals := [squared_distance p r, squared_distance q s]
  (sides.all (· = sides.head!)) ∧ (diagonals.all (· = 2 * sides.head!))

-- Theorem statement
theorem point_A_not_in_square :
  ¬(is_square A B C D ∨ is_square A B C E ∨ is_square A B D E ∨ is_square A C D E) ∧
  (is_square B C D E) := by sorry

end NUMINAMATH_CALUDE_point_A_not_in_square_l3806_380622


namespace NUMINAMATH_CALUDE_different_color_probability_l3806_380692

def blue_chips : ℕ := 4
def yellow_chips : ℕ := 5
def green_chips : ℕ := 3

def total_chips : ℕ := blue_chips + yellow_chips + green_chips

theorem different_color_probability :
  let p_blue_first := blue_chips / total_chips
  let p_yellow_first := yellow_chips / total_chips
  let p_green_first := green_chips / total_chips
  let p_not_blue_second := (yellow_chips + green_chips) / (total_chips - 1)
  let p_not_yellow_second := (blue_chips + green_chips) / (total_chips - 1)
  let p_not_green_second := (blue_chips + yellow_chips) / (total_chips - 1)
  (p_blue_first * p_not_blue_second + 
   p_yellow_first * p_not_yellow_second + 
   p_green_first * p_not_green_second) = 47 / 66 :=
by
  sorry

end NUMINAMATH_CALUDE_different_color_probability_l3806_380692


namespace NUMINAMATH_CALUDE_johns_earnings_ratio_l3806_380685

def saturday_earnings : ℤ := 18
def previous_weekend_earnings : ℤ := 20
def pogo_stick_cost : ℤ := 60
def additional_needed : ℤ := 13

def total_earnings : ℤ := pogo_stick_cost - additional_needed

theorem johns_earnings_ratio :
  let sunday_earnings := total_earnings - saturday_earnings - previous_weekend_earnings
  saturday_earnings / sunday_earnings = 2 := by
  sorry

end NUMINAMATH_CALUDE_johns_earnings_ratio_l3806_380685


namespace NUMINAMATH_CALUDE_percent_problem_l3806_380654

theorem percent_problem (x : ℝ) : 
  (30 / 100 * 100 = 50 / 100 * x + 10) → x = 40 := by
  sorry

end NUMINAMATH_CALUDE_percent_problem_l3806_380654


namespace NUMINAMATH_CALUDE_no_solution_for_prime_factor_conditions_l3806_380653

/-- P(n) denotes the greatest prime factor of n -/
def greatest_prime_factor (n : ℕ) : ℕ :=
  sorry

theorem no_solution_for_prime_factor_conditions : 
  ∀ n : ℕ, n > 1 → 
  ¬(greatest_prime_factor n = Real.sqrt n ∧ 
    greatest_prime_factor (n + 54) = Real.sqrt (n + 54)) :=
by sorry

end NUMINAMATH_CALUDE_no_solution_for_prime_factor_conditions_l3806_380653


namespace NUMINAMATH_CALUDE_unique_number_with_three_prime_divisors_l3806_380621

theorem unique_number_with_three_prime_divisors (x : ℕ) (n : ℕ) :
  Odd n →
  x = 6^n + 1 →
  (∃ p q : ℕ, Prime p ∧ Prime q ∧ p ≠ q ∧ p ≠ 11 ∧ q ≠ 11 ∧
    x = 11 * p * q ∧
    ∀ r : ℕ, Prime r → r ∣ x → (r = 11 ∨ r = p ∨ r = q)) →
  x = 7777 := by
sorry

end NUMINAMATH_CALUDE_unique_number_with_three_prime_divisors_l3806_380621


namespace NUMINAMATH_CALUDE_daps_equivalent_to_dips_l3806_380683

/-- Represents the conversion rate between daps and dops -/
def daps_to_dops : ℚ := 5 / 4

/-- Represents the conversion rate between dops and dips -/
def dops_to_dips : ℚ := 3 / 8

/-- The number of dips we want to convert to daps -/
def target_dips : ℚ := 40

/-- Theorem stating the equivalence between daps and dips -/
theorem daps_equivalent_to_dips : 
  (target_dips * daps_to_dops * dops_to_dips)⁻¹ * target_dips = 18.75 := by
  sorry

end NUMINAMATH_CALUDE_daps_equivalent_to_dips_l3806_380683


namespace NUMINAMATH_CALUDE_last_eight_digits_of_product_l3806_380651

def product : ℕ := 11 * 101 * 1001 * 10001 * 1000001 * 111

theorem last_eight_digits_of_product : product % 100000000 = 87654321 := by
  sorry

end NUMINAMATH_CALUDE_last_eight_digits_of_product_l3806_380651


namespace NUMINAMATH_CALUDE_hanson_employees_count_l3806_380687

theorem hanson_employees_count :
  ∃ (E : ℕ) (M B B' : ℤ), 
    M = E * B + 2 ∧ 
    3 * M = E * B' + 1 → 
    E = 5 := by
  sorry

end NUMINAMATH_CALUDE_hanson_employees_count_l3806_380687


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l3806_380600

theorem min_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  (1 / a + 1 / b) ≥ 2 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ + b₀ = 2 ∧ 1 / a₀ + 1 / b₀ = 2 :=
by sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l3806_380600


namespace NUMINAMATH_CALUDE_bryans_books_l3806_380699

/-- Calculates the total number of books given the number of bookshelves and books per shelf. -/
def total_books (num_shelves : ℕ) (books_per_shelf : ℕ) : ℕ :=
  num_shelves * books_per_shelf

/-- Theorem stating that Bryan's total number of books is 504. -/
theorem bryans_books : 
  total_books 9 56 = 504 := by
  sorry

end NUMINAMATH_CALUDE_bryans_books_l3806_380699


namespace NUMINAMATH_CALUDE_roots_product_equals_squared_difference_l3806_380668

theorem roots_product_equals_squared_difference (m n : ℝ) 
  (α β γ δ : ℝ) : 
  (α^2 + m*α - 1 = 0) → 
  (β^2 + m*β - 1 = 0) → 
  (γ^2 + n*γ - 1 = 0) → 
  (δ^2 + n*δ - 1 = 0) → 
  (α - γ)*(β - γ)*(α - δ)*(β - δ) = (m - n)^2 := by
  sorry

end NUMINAMATH_CALUDE_roots_product_equals_squared_difference_l3806_380668


namespace NUMINAMATH_CALUDE_x_squared_vs_two_to_x_l3806_380669

theorem x_squared_vs_two_to_x (x : ℝ) :
  ¬(∀ x, x^2 < 1 → 2^x < 1) ∧ ¬(∀ x, 2^x < 1 → x^2 < 1) :=
sorry

end NUMINAMATH_CALUDE_x_squared_vs_two_to_x_l3806_380669


namespace NUMINAMATH_CALUDE_guppy_ratio_l3806_380648

/-- Represents the number of guppies each person has -/
structure Guppies where
  haylee : ℕ
  jose : ℕ
  charliz : ℕ
  nicolai : ℕ

/-- The conditions of the guppy problem -/
def guppy_conditions (g : Guppies) : Prop :=
  g.haylee = 36 ∧
  g.charliz = g.jose / 3 ∧
  g.nicolai = 4 * g.charliz ∧
  g.haylee + g.jose + g.charliz + g.nicolai = 84

/-- The theorem stating the ratio of Jose's guppies to Haylee's guppies -/
theorem guppy_ratio (g : Guppies) (h : guppy_conditions g) : 
  g.jose * 2 = g.haylee :=
sorry

end NUMINAMATH_CALUDE_guppy_ratio_l3806_380648


namespace NUMINAMATH_CALUDE_amaya_movie_watching_time_l3806_380696

/-- Calculates the total time spent watching a movie with interruptions and rewinds -/
def total_watching_time (segment1 segment2 segment3 rewind1 rewind2 : ℕ) : ℕ :=
  segment1 + segment2 + segment3 + rewind1 + rewind2

/-- Theorem stating that the total watching time for Amaya's movie is 120 minutes -/
theorem amaya_movie_watching_time :
  total_watching_time 35 45 20 5 15 = 120 := by
  sorry

#eval total_watching_time 35 45 20 5 15

end NUMINAMATH_CALUDE_amaya_movie_watching_time_l3806_380696


namespace NUMINAMATH_CALUDE_cone_base_circumference_l3806_380688

/-- 
Given a right circular cone with volume 27π cubic centimeters and height 9 cm,
prove that the circumference of the base is 6π cm.
-/
theorem cone_base_circumference (V : ℝ) (h : ℝ) (r : ℝ) :
  V = 27 * Real.pi ∧ h = 9 ∧ V = (1/3) * Real.pi * r^2 * h →
  2 * Real.pi * r = 6 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_cone_base_circumference_l3806_380688


namespace NUMINAMATH_CALUDE_students_passed_both_tests_l3806_380613

theorem students_passed_both_tests 
  (total : Nat) 
  (passed_long_jump : Nat) 
  (passed_shot_put : Nat) 
  (failed_both : Nat) : 
  total = 50 → 
  passed_long_jump = 40 → 
  passed_shot_put = 31 → 
  failed_both = 4 → 
  ∃ (passed_both : Nat), 
    passed_both = 25 ∧ 
    total = passed_both + (passed_long_jump - passed_both) + (passed_shot_put - passed_both) + failed_both :=
by sorry

end NUMINAMATH_CALUDE_students_passed_both_tests_l3806_380613


namespace NUMINAMATH_CALUDE_functional_equation_proof_l3806_380655

/-- Given a function f: ℝ → ℝ satisfying the functional equation
    f(x + y) = f(x) * f(y) for all real x and y, and f(3) = 4,
    prove that f(9) = 64. -/
theorem functional_equation_proof (f : ℝ → ℝ) 
    (h1 : ∀ x y : ℝ, f (x + y) = f x * f y) 
    (h2 : f 3 = 4) : 
  f 9 = 64 := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_proof_l3806_380655


namespace NUMINAMATH_CALUDE_no_real_solutions_l3806_380603

theorem no_real_solutions : ∀ x : ℝ, (x^3 - 8) / (x - 2) ≠ 3*x :=
sorry

end NUMINAMATH_CALUDE_no_real_solutions_l3806_380603


namespace NUMINAMATH_CALUDE_parabola_zeros_difference_l3806_380642

/-- A parabola with vertex (3, -2) passing through (5, 14) has zeros with difference √2 -/
theorem parabola_zeros_difference (a b c : ℝ) : 
  (∀ x, a * x^2 + b * x + c = a * (x - 3)^2 - 2) →  -- Vertex form
  (4 * 4 + b * 5 + c = 14) →  -- Point (5, 14) satisfies the equation
  (∃ m n : ℝ, m > n ∧ 
    a * m^2 + b * m + c = 0 ∧ 
    a * n^2 + b * n + c = 0 ∧ 
    m - n = Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_parabola_zeros_difference_l3806_380642


namespace NUMINAMATH_CALUDE_inequality_proof_l3806_380665

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b ≤ 1) :
  9 * a^2 * b + 9 * a * b^2 - a^2 - 10 * a * b - b^2 + a + b ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3806_380665


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l3806_380675

theorem solution_set_of_inequality (x : ℝ) :
  (x - 2) / (x + 3) > 0 ↔ x ∈ Set.Ioi (-3) ∪ Set.Ioi 2 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l3806_380675


namespace NUMINAMATH_CALUDE_candy_bar_cost_l3806_380691

/-- The cost of a candy bar, given that it costs $1 less than a chocolate that costs $3. -/
theorem candy_bar_cost : ℝ := by
  -- Define the cost of the candy bar
  let candy_cost : ℝ := 2

  -- Define the cost of the chocolate
  let chocolate_cost : ℝ := 3

  -- Assert that the chocolate costs $1 more than the candy bar
  have h1 : chocolate_cost = candy_cost + 1 := by sorry

  -- Prove that the candy bar costs $2
  have h2 : candy_cost = 2 := by sorry

  -- Return the cost of the candy bar
  exact candy_cost


end NUMINAMATH_CALUDE_candy_bar_cost_l3806_380691


namespace NUMINAMATH_CALUDE_same_result_different_parentheses_l3806_380650

-- Define the exponentiation operation
def power (a b : ℕ) : ℕ := a ^ b

-- Define the two different parenthesization methods
def method1 (n : ℕ) : ℕ := power (power n 7) (power 7 7)
def method2 (n : ℕ) : ℕ := power (power n (power 7 7)) 7

-- Theorem statement
theorem same_result_different_parentheses :
  ∃ (n : ℕ), method1 n = method2 n :=
sorry

end NUMINAMATH_CALUDE_same_result_different_parentheses_l3806_380650


namespace NUMINAMATH_CALUDE_earliest_meeting_time_l3806_380698

def anna_lap_time : ℕ := 5
def stephanie_lap_time : ℕ := 8
def james_lap_time : ℕ := 9
def tom_lap_time : ℕ := 10

theorem earliest_meeting_time :
  let lap_times := [anna_lap_time, stephanie_lap_time, james_lap_time, tom_lap_time]
  Nat.lcm (Nat.lcm (Nat.lcm anna_lap_time stephanie_lap_time) james_lap_time) tom_lap_time = 360 :=
by sorry

end NUMINAMATH_CALUDE_earliest_meeting_time_l3806_380698


namespace NUMINAMATH_CALUDE_max_b_in_box_l3806_380661

theorem max_b_in_box (a b c : ℕ) : 
  (a * b * c = 360) →
  (1 < c) → (c < b) → (b < a) →
  b ≤ 12 :=
by sorry

end NUMINAMATH_CALUDE_max_b_in_box_l3806_380661


namespace NUMINAMATH_CALUDE_characteristic_vector_of_g_sin_value_for_associated_function_l3806_380612

def associated_characteristic_vector (f : ℝ → ℝ) : ℝ × ℝ :=
  sorry

def associated_function (v : ℝ × ℝ) : ℝ → ℝ :=
  sorry

theorem characteristic_vector_of_g :
  let g : ℝ → ℝ := λ x => Real.sin (x + 5 * Real.pi / 6) - Real.sin (3 * Real.pi / 2 - x)
  associated_characteristic_vector g = (-Real.sqrt 3 / 2, 3 / 2) :=
sorry

theorem sin_value_for_associated_function :
  let f := associated_function (1, Real.sqrt 3)
  ∀ x, f x = 8 / 5 → x > -Real.pi / 3 → x < Real.pi / 6 →
    Real.sin x = (4 - 3 * Real.sqrt 3) / 10 :=
sorry

end NUMINAMATH_CALUDE_characteristic_vector_of_g_sin_value_for_associated_function_l3806_380612


namespace NUMINAMATH_CALUDE_fundraising_excess_l3806_380656

/-- Proves that Scott, Mary, and Ken exceeded their fundraising goal by $600 --/
theorem fundraising_excess (ken : ℕ) (mary scott : ℕ) (goal : ℕ) : 
  ken = 600 →
  mary = 5 * ken →
  mary = 3 * scott →
  goal = 4000 →
  mary + scott + ken - goal = 600 := by
sorry

end NUMINAMATH_CALUDE_fundraising_excess_l3806_380656


namespace NUMINAMATH_CALUDE_chocolate_bars_count_l3806_380647

/-- Given the total number of treats and the counts of chewing gums and candies,
    calculate the number of chocolate bars. -/
theorem chocolate_bars_count 
  (total_treats : ℕ) 
  (chewing_gums : ℕ) 
  (candies : ℕ) 
  (h1 : total_treats = 155) 
  (h2 : chewing_gums = 60) 
  (h3 : candies = 40) : 
  total_treats - (chewing_gums + candies) = 55 := by
  sorry

#eval 155 - (60 + 40)  -- Should output 55

end NUMINAMATH_CALUDE_chocolate_bars_count_l3806_380647


namespace NUMINAMATH_CALUDE_ratio_d_b_is_negative_four_l3806_380672

def f (a b c d : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + d

theorem ratio_d_b_is_negative_four
  (a b c d : ℝ)
  (h_even : ∀ x, f a b c d x = f a b c d (-x))
  (h_solution : ∀ x, f a b c d x < 0 ↔ -2 < x ∧ x < 2) :
  d / b = -4 := by
  sorry

end NUMINAMATH_CALUDE_ratio_d_b_is_negative_four_l3806_380672


namespace NUMINAMATH_CALUDE_certain_number_proof_l3806_380639

theorem certain_number_proof : 
  ∃ x : ℕ, (7899665 : ℕ) - (12 * 3 * x) = 7899593 ∧ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l3806_380639


namespace NUMINAMATH_CALUDE_max_sum_for_product_1386_l3806_380676

theorem max_sum_for_product_1386 :
  ∃ (A B C : ℕ+),
    A * B * C = 1386 ∧
    A ≠ B ∧ B ≠ C ∧ A ≠ C ∧
    ∀ (X Y Z : ℕ+),
      X * Y * Z = 1386 →
      X ≠ Y ∧ Y ≠ Z ∧ X ≠ Z →
      X + Y + Z ≤ A + B + C ∧
      A + B + C = 88 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_for_product_1386_l3806_380676


namespace NUMINAMATH_CALUDE_zinc_copper_mixture_weight_l3806_380660

/-- Proves that the total weight of a zinc-copper mixture is 70 kg,
    given a 9:11 ratio and 31.5 kg of zinc used. -/
theorem zinc_copper_mixture_weight
  (zinc_ratio : ℝ)
  (copper_ratio : ℝ)
  (zinc_weight : ℝ)
  (h_ratio : zinc_ratio / copper_ratio = 9 / 11)
  (h_zinc : zinc_weight = 31.5) :
  zinc_weight + (copper_ratio / zinc_ratio) * zinc_weight = 70 :=
by sorry

end NUMINAMATH_CALUDE_zinc_copper_mixture_weight_l3806_380660


namespace NUMINAMATH_CALUDE_perpendicular_slope_l3806_380664

/-- Given two points (3, -4) and (-2, 5) on a line, the slope of a line perpendicular to this line is 5/9. -/
theorem perpendicular_slope : 
  let p1 : ℝ × ℝ := (3, -4)
  let p2 : ℝ × ℝ := (-2, 5)
  let m : ℝ := (p2.2 - p1.2) / (p2.1 - p1.1)
  (- (1 / m)) = 5 / 9 := by sorry

end NUMINAMATH_CALUDE_perpendicular_slope_l3806_380664


namespace NUMINAMATH_CALUDE_equation_rewrite_l3806_380604

theorem equation_rewrite (x y : ℝ) : 
  (2 * x - y = 4) → (y = 2 * x - 4) := by
  sorry

end NUMINAMATH_CALUDE_equation_rewrite_l3806_380604


namespace NUMINAMATH_CALUDE_least_bananas_total_l3806_380640

/-- Represents the number of bananas taken by each monkey -/
structure BananaCounts where
  b₁ : ℕ
  b₂ : ℕ
  b₃ : ℕ

/-- Represents the final distribution of bananas for each monkey -/
structure FinalDistribution where
  m₁ : ℕ
  m₂ : ℕ
  m₃ : ℕ

/-- Calculates the final distribution based on the initial banana counts -/
def calculateFinalDistribution (counts : BananaCounts) : FinalDistribution :=
  { m₁ := counts.b₁ / 2 + counts.b₂ / 12 + counts.b₃ * 3 / 32
  , m₂ := counts.b₁ / 6 + counts.b₂ * 2 / 3 + counts.b₃ * 3 / 32
  , m₃ := counts.b₁ / 6 + counts.b₂ / 12 + counts.b₃ * 3 / 4 }

/-- Checks if the final distribution satisfies the 4:3:2 ratio -/
def satisfiesRatio (dist : FinalDistribution) : Prop :=
  3 * dist.m₁ = 4 * dist.m₂ ∧ 2 * dist.m₁ = 3 * dist.m₃

/-- The main theorem stating the least possible total number of bananas -/
theorem least_bananas_total (counts : BananaCounts) :
  (∀ (dist : FinalDistribution), dist = calculateFinalDistribution counts → satisfiesRatio dist) →
  counts.b₁ + counts.b₂ + counts.b₃ ≥ 148 :=
by sorry

end NUMINAMATH_CALUDE_least_bananas_total_l3806_380640


namespace NUMINAMATH_CALUDE_seventh_term_is_25_over_3_l3806_380637

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  -- First term
  a : ℚ
  -- Common difference
  d : ℚ
  -- Sum of first five terms is 15
  sum_first_five : a + (a + d) + (a + 2*d) + (a + 3*d) + (a + 4*d) = 15
  -- Sixth term is 7
  sixth_term : a + 5*d = 7

/-- The seventh term of the arithmetic sequence is 25/3 -/
theorem seventh_term_is_25_over_3 (seq : ArithmeticSequence) :
  seq.a + 6*seq.d = 25/3 := by
  sorry

end NUMINAMATH_CALUDE_seventh_term_is_25_over_3_l3806_380637


namespace NUMINAMATH_CALUDE_inequality_equality_l3806_380679

theorem inequality_equality (x : ℝ) : 
  x > 0 → (x * Real.sqrt (16 - x^2) + Real.sqrt (16*x - x^4) ≥ 16 ↔ x = 2 * Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_equality_l3806_380679


namespace NUMINAMATH_CALUDE_tan_30_squared_plus_sin_45_squared_l3806_380680

theorem tan_30_squared_plus_sin_45_squared : 
  (Real.tan (30 * π / 180))^2 + (Real.sin (45 * π / 180))^2 = 5/6 := by
  sorry

end NUMINAMATH_CALUDE_tan_30_squared_plus_sin_45_squared_l3806_380680


namespace NUMINAMATH_CALUDE_ratio_equality_l3806_380630

theorem ratio_equality (x y : ℝ) (h1 : 3 * x = 5 * y) (h2 : x ≠ 0) (h3 : y ≠ 0) :
  x / y = 5 / 3 := by sorry

end NUMINAMATH_CALUDE_ratio_equality_l3806_380630


namespace NUMINAMATH_CALUDE_identity_is_unique_satisfying_function_l3806_380657

/-- A function satisfying the given property -/
def SatisfyingFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y z : ℝ, x^3 + f y * x + f z = 0 → f x^3 + y * f x + z = 0

/-- The main theorem stating that the identity function is the only function satisfying the property -/
theorem identity_is_unique_satisfying_function :
  ∀ f : ℝ → ℝ, SatisfyingFunction f → f = id := by sorry

end NUMINAMATH_CALUDE_identity_is_unique_satisfying_function_l3806_380657


namespace NUMINAMATH_CALUDE_average_equation_l3806_380615

theorem average_equation (y : ℝ) : 
  (55 + 48 + 507 + 2 + 684 + y) / 6 = 223 → y = 42 := by
  sorry

end NUMINAMATH_CALUDE_average_equation_l3806_380615


namespace NUMINAMATH_CALUDE_ship_elevation_change_l3806_380697

/-- The average change in elevation per hour for a ship traveling between Lake Ontario and Lake Erie -/
theorem ship_elevation_change (lake_ontario_elevation lake_erie_elevation : ℝ) (travel_time : ℝ) :
  lake_ontario_elevation = 75 ∧ 
  lake_erie_elevation = 174.28 ∧ 
  travel_time = 8 →
  (lake_erie_elevation - lake_ontario_elevation) / travel_time = 12.41 :=
by sorry

end NUMINAMATH_CALUDE_ship_elevation_change_l3806_380697


namespace NUMINAMATH_CALUDE_cannot_fit_rectangles_l3806_380636

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Calculates the area of a rectangle -/
def area (r : Rectangle) : ℕ := r.width * r.height

/-- The large rectangle -/
def largeRectangle : Rectangle := { width := 13, height := 7 }

/-- The small rectangle -/
def smallRectangle : Rectangle := { width := 2, height := 3 }

/-- The number of small rectangles -/
def numSmallRectangles : ℕ := 15

/-- Theorem stating that it's not possible to fit 15 small rectangles into the large rectangle -/
theorem cannot_fit_rectangles : 
  area largeRectangle > numSmallRectangles * area smallRectangle :=
sorry

end NUMINAMATH_CALUDE_cannot_fit_rectangles_l3806_380636


namespace NUMINAMATH_CALUDE_journey_distance_correct_total_distance_l3806_380658

-- Define the journey parameters
def total_time : ℝ := 30
def speed_first_half : ℝ := 20
def speed_second_half : ℝ := 10

-- Define the total distance
def total_distance : ℝ := 400

-- Theorem statement
theorem journey_distance :
  (total_distance / 2 / speed_first_half) + (total_distance / 2 / speed_second_half) = total_time :=
by sorry

-- Proof that the total distance is correct
theorem correct_total_distance : total_distance = 400 :=
by sorry

end NUMINAMATH_CALUDE_journey_distance_correct_total_distance_l3806_380658


namespace NUMINAMATH_CALUDE_other_root_of_complex_equation_l3806_380659

theorem other_root_of_complex_equation (z : ℂ) :
  z^2 = -99 + 64*I ∧ (5 + 8*I)^2 = -99 + 64*I → z = 5 + 8*I ∨ z = -5 - 8*I :=
by sorry

end NUMINAMATH_CALUDE_other_root_of_complex_equation_l3806_380659


namespace NUMINAMATH_CALUDE_weight_loss_problem_l3806_380614

theorem weight_loss_problem (x : ℝ) : 
  (x - 12 = 2 * (x - 7) - 80) → x = 82 := by
  sorry

end NUMINAMATH_CALUDE_weight_loss_problem_l3806_380614


namespace NUMINAMATH_CALUDE_sqrt_sum_given_diff_l3806_380670

theorem sqrt_sum_given_diff (x : ℝ) :
  Real.sqrt (100 - x^2) - Real.sqrt (36 - x^2) = 5 →
  Real.sqrt (100 - x^2) + Real.sqrt (36 - x^2) = 12.8 := by
sorry

end NUMINAMATH_CALUDE_sqrt_sum_given_diff_l3806_380670


namespace NUMINAMATH_CALUDE_identical_asymptotes_hyperbolas_l3806_380606

theorem identical_asymptotes_hyperbolas (M : ℝ) : 
  (∀ x y : ℝ, x^2/9 - y^2/16 = 1 ↔ y^2/25 - x^2/M = 1) → M = 225/16 := by
  sorry

end NUMINAMATH_CALUDE_identical_asymptotes_hyperbolas_l3806_380606


namespace NUMINAMATH_CALUDE_coefficient_of_x_squared_l3806_380609

def polynomial (x : ℝ) : ℝ := 5*(x - x^4) - 4*(x^2 - 2*x^4 + x^6) + 3*(2*x^2 - x^8)

theorem coefficient_of_x_squared (x : ℝ) : 
  ∃ (a b c : ℝ), polynomial x = 2*x^2 + a*x + b*x^3 + c*x^4 + 
    (-5)*x^4 + 8*x^4 + (-4)*x^6 + (-3)*x^8 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_of_x_squared_l3806_380609


namespace NUMINAMATH_CALUDE_sqrt_inequality_l3806_380671

theorem sqrt_inequality (a : ℝ) (ha : a > 0) :
  Real.sqrt (a + 5) - Real.sqrt (a + 3) > Real.sqrt (a + 6) - Real.sqrt (a + 4) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_inequality_l3806_380671


namespace NUMINAMATH_CALUDE_probability_all_girls_chosen_l3806_380602

def total_members : ℕ := 15
def num_boys : ℕ := 8
def num_girls : ℕ := 7
def num_chosen : ℕ := 3

theorem probability_all_girls_chosen :
  (Nat.choose num_girls num_chosen : ℚ) / (Nat.choose total_members num_chosen) = 1 / 13 :=
by sorry

end NUMINAMATH_CALUDE_probability_all_girls_chosen_l3806_380602


namespace NUMINAMATH_CALUDE_ferry_travel_time_l3806_380677

/-- Represents the travel time of Ferry P in hours -/
def t : ℝ := 3

/-- Speed of Ferry P in km/h -/
def speed_p : ℝ := 6

/-- Speed of Ferry Q in km/h -/
def speed_q : ℝ := speed_p + 3

/-- Distance traveled by Ferry P in km -/
def distance_p : ℝ := speed_p * t

/-- Distance traveled by Ferry Q in km -/
def distance_q : ℝ := 2 * distance_p

/-- Travel time of Ferry Q in hours -/
def time_q : ℝ := t + 1

theorem ferry_travel_time :
  speed_q * time_q = distance_q ∧ t = 3 := by sorry

end NUMINAMATH_CALUDE_ferry_travel_time_l3806_380677


namespace NUMINAMATH_CALUDE_sum_25_terms_equals_625_l3806_380682

def arithmetic_sequence (n : ℕ) : ℕ := 2 * n - 1

def sum_arithmetic_sequence (n : ℕ) : ℕ := n * (arithmetic_sequence 1 + arithmetic_sequence n) / 2

theorem sum_25_terms_equals_625 : sum_arithmetic_sequence 25 = 625 := by sorry

end NUMINAMATH_CALUDE_sum_25_terms_equals_625_l3806_380682


namespace NUMINAMATH_CALUDE_b_completion_time_l3806_380641

/-- Given two workers a and b, this theorem proves how long it takes b to complete a job alone. -/
theorem b_completion_time (work : ℝ) (a b : ℝ → ℝ) :
  (∀ t, a t + b t = work / 16) →  -- a and b together complete the work in 16 days
  (∀ t, a t = work / 20) →        -- a alone completes the work in 20 days
  (∀ t, b t = work / 80) :=       -- b alone completes the work in 80 days
by sorry

end NUMINAMATH_CALUDE_b_completion_time_l3806_380641


namespace NUMINAMATH_CALUDE_simplify_fraction_l3806_380681

theorem simplify_fraction (x : ℝ) (h : x ≠ -1) :
  (x^2 - 1) / (x + 1) = x - 1 := by
sorry

end NUMINAMATH_CALUDE_simplify_fraction_l3806_380681


namespace NUMINAMATH_CALUDE_smallest_m_is_correct_l3806_380608

/-- The smallest positive value of m for which 15x^2 - mx + 630 = 0 has integral solutions -/
def smallest_m : ℕ := 195

/-- The equation 15x^2 - mx + 630 = 0 has integral solutions -/
def has_integral_solutions (m : ℕ) : Prop :=
  ∃ x : ℤ, 15 * x^2 - m * x + 630 = 0

/-- The main theorem: smallest_m is the smallest positive value of m for which
    the equation 15x^2 - mx + 630 = 0 has integral solutions -/
theorem smallest_m_is_correct :
  has_integral_solutions smallest_m ∧
  ∀ m : ℕ, 0 < m → m < smallest_m → ¬(has_integral_solutions m) :=
sorry

end NUMINAMATH_CALUDE_smallest_m_is_correct_l3806_380608


namespace NUMINAMATH_CALUDE_fraction_repetend_l3806_380633

/-- The repetend of the decimal representation of 7/29 -/
def repetend : Nat := 241379

/-- The length of the repetend -/
def repetend_length : Nat := 6

/-- The fraction we're considering -/
def fraction : Rat := 7 / 29

theorem fraction_repetend :
  ∃ (k : ℕ), (fraction * 10^repetend_length - fraction) * 10^k = repetend / (10^repetend_length - 1) :=
sorry

end NUMINAMATH_CALUDE_fraction_repetend_l3806_380633


namespace NUMINAMATH_CALUDE_simplify_expression_l3806_380618

theorem simplify_expression (a : ℝ) : a^4 * (-a)^3 = -a^7 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3806_380618


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l3806_380663

theorem complex_magnitude_problem (z w : ℂ) : 
  Complex.abs (3 * z - w) = 30 →
  Complex.abs (z + 3 * w) = 6 →
  Complex.abs (z + w) = 3 →
  ∃! (abs_z : ℝ), abs_z > 0 ∧ Complex.abs z = abs_z :=
by sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l3806_380663


namespace NUMINAMATH_CALUDE_number_problem_l3806_380666

theorem number_problem (a b c : ℕ) :
  Nat.gcd a b = 15 →
  Nat.gcd b c = 6 →
  b * c = 1800 →
  Nat.lcm a b = 3150 →
  a = 315 ∧ b = 150 ∧ c = 12 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l3806_380666


namespace NUMINAMATH_CALUDE_x_range_for_given_equation_l3806_380610

theorem x_range_for_given_equation (x y : ℝ) :
  x - 4 * Real.sqrt y = 2 * Real.sqrt (x - y) →
  x = 0 ∨ (4 ≤ x ∧ x ≤ 20) := by
  sorry

end NUMINAMATH_CALUDE_x_range_for_given_equation_l3806_380610


namespace NUMINAMATH_CALUDE_quadratic_equation_1_l3806_380694

theorem quadratic_equation_1 : 
  ∃ x₁ x₂ : ℝ, (x₁ + 1)^2 - 144 = 0 ∧ (x₂ + 1)^2 - 144 = 0 ∧ x₁ ≠ x₂ :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_1_l3806_380694


namespace NUMINAMATH_CALUDE_question_one_l3806_380674

theorem question_one (x : ℝ) (h : x^2 - 3*x = 2) :
  1 + 2*x^2 - 6*x = 5 := by
  sorry


end NUMINAMATH_CALUDE_question_one_l3806_380674


namespace NUMINAMATH_CALUDE_median_of_special_list_l3806_380693

/-- Represents the sum of integers from 1 to n -/
def triangularSum (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Represents our special list where each number n appears n times, up to 250 -/
def specialList : List ℕ := sorry

/-- The length of our special list -/
def listLength : ℕ := triangularSum 250

/-- The index of the median element in our list -/
def medianIndex : ℕ := (listLength + 1) / 2

/-- Function to find the smallest n such that triangularSum n ≥ target -/
def findSmallestN (target : ℕ) : ℕ := sorry

theorem median_of_special_list :
  let n := findSmallestN medianIndex
  n = 177 := by sorry

end NUMINAMATH_CALUDE_median_of_special_list_l3806_380693


namespace NUMINAMATH_CALUDE_green_blue_difference_l3806_380607

/-- Represents the number of tiles in a hexagonal figure -/
structure HexFigure where
  blue : Nat
  green : Nat

/-- Calculates the number of tiles needed for a double border -/
def doubleBorderTiles : Nat := 2 * 18

/-- The initial hexagonal figure -/
def initialFigure : HexFigure := { blue := 13, green := 6 }

/-- Creates a new figure with twice as many tiles -/
def doubleFigure (f : HexFigure) : HexFigure :=
  { blue := 2 * f.blue, green := 2 * f.green }

/-- Adds a double border of green tiles to a figure -/
def addGreenBorder (f : HexFigure) : HexFigure :=
  { blue := f.blue, green := f.green + doubleBorderTiles }

/-- Calculates the total tiles for two figures -/
def totalTiles (f1 f2 : HexFigure) : HexFigure :=
  { blue := f1.blue + f2.blue, green := f1.green + f2.green }

theorem green_blue_difference :
  let secondFigure := addGreenBorder (doubleFigure initialFigure)
  let totalFigure := totalTiles initialFigure secondFigure
  totalFigure.green - totalFigure.blue = 15 := by sorry

end NUMINAMATH_CALUDE_green_blue_difference_l3806_380607


namespace NUMINAMATH_CALUDE_player_a_not_losing_probability_l3806_380649

theorem player_a_not_losing_probability 
  (p_win : ℝ) 
  (p_draw : ℝ) 
  (h1 : p_win = 0.3) 
  (h2 : p_draw = 0.4) : 
  p_win + p_draw = 0.7 := by
  sorry

end NUMINAMATH_CALUDE_player_a_not_losing_probability_l3806_380649


namespace NUMINAMATH_CALUDE_expression_simplification_l3806_380619

theorem expression_simplification (x : ℝ) (h1 : x ≠ 0) (h2 : x ≠ 2) (h3 : x ≠ 4) :
  (x^2 - 4) * ((x + 2) / (x^2 - 2*x) - (x - 1) / (x^2 - 4*x + 4)) / ((x - 4) / x) = (x + 2) / (x - 2) := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3806_380619


namespace NUMINAMATH_CALUDE_problem_statement_l3806_380620

theorem problem_statement (P Q : ℝ) :
  (∀ x : ℝ, x ≠ 3 → P / (x - 3) + Q * (x + 2) = (-5 * x^2 + 18 * x + 40) / (x - 3)) →
  P + Q = 44 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3806_380620


namespace NUMINAMATH_CALUDE_min_max_problem_l3806_380662

theorem min_max_problem (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  (8 / x + 2 / y ≥ 18) ∧ (Real.sqrt (2 * x + 1) + Real.sqrt (2 * y + 1) ≤ 2 * Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_min_max_problem_l3806_380662


namespace NUMINAMATH_CALUDE_xyz_value_l3806_380611

theorem xyz_value (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 36)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 12)
  (h3 : (x + y + z)^2 = x^2 + y^2 + z^2 + 12) :
  x * y * z = 8 := by
  sorry

end NUMINAMATH_CALUDE_xyz_value_l3806_380611


namespace NUMINAMATH_CALUDE_lemonade_pitchers_sum_l3806_380627

theorem lemonade_pitchers_sum : 
  let first_intermission : ℚ := 0.25
  let second_intermission : ℚ := 0.42
  let third_intermission : ℚ := 0.25
  first_intermission + second_intermission + third_intermission = 0.92 := by
sorry

end NUMINAMATH_CALUDE_lemonade_pitchers_sum_l3806_380627


namespace NUMINAMATH_CALUDE_min_sum_first_two_terms_l3806_380689

/-- A sequence of positive integers satisfying the given recurrence relation -/
def ValidSequence (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, n ≥ 1 → a (n + 2) = (a n + 3024) / (1 + a (n + 1))

/-- The theorem stating the minimum possible value of a₁ + a₂ -/
theorem min_sum_first_two_terms (a : ℕ → ℕ) (h : ValidSequence a) :
    ∀ b : ℕ → ℕ, ValidSequence b → a 1 + a 2 ≤ b 1 + b 2 :=
  sorry

end NUMINAMATH_CALUDE_min_sum_first_two_terms_l3806_380689


namespace NUMINAMATH_CALUDE_range_of_a_l3806_380690

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, ((a - 5) * x > a - 5) ↔ x < 1) → a < 5 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l3806_380690


namespace NUMINAMATH_CALUDE_earliest_saturday_after_second_monday_after_second_thursday_l3806_380646

/-- Represents a day of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a date in a month -/
structure Date where
  day : Nat
  dayOfWeek : DayOfWeek

/-- Returns the next day of the week -/
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

/-- Returns the next date -/
def nextDate (d : Date) : Date :=
  { day := d.day + 1, dayOfWeek := nextDay d.dayOfWeek }

/-- Finds the nth occurrence of a specific day of the week, starting from a given date -/
def findNthDay (start : Date) (target : DayOfWeek) (n : Nat) : Date :=
  sorry

/-- Finds the first occurrence of a specific day of the week, starting from a given date -/
def findNextDay (start : Date) (target : DayOfWeek) : Date :=
  sorry

/-- Main theorem: The earliest possible date for the first Saturday after the second Monday 
    following the second Thursday of any month is the 17th -/
theorem earliest_saturday_after_second_monday_after_second_thursday (startDate : Date) : 
  (findNextDay 
    (findNthDay 
      (findNthDay startDate DayOfWeek.Thursday 2) 
      DayOfWeek.Monday 
      2) 
    DayOfWeek.Saturday).day ≥ 17 :=
  sorry

end NUMINAMATH_CALUDE_earliest_saturday_after_second_monday_after_second_thursday_l3806_380646


namespace NUMINAMATH_CALUDE_universiade_volunteer_count_l3806_380623

/-- Represents the result of a stratified sampling by gender -/
structure StratifiedSample where
  total_pool : ℕ
  selected_group : ℕ
  selected_male : ℕ
  selected_female : ℕ

/-- Calculates the number of female students in the pool based on stratified sampling -/
def femaleInPool (sample : StratifiedSample) : ℕ :=
  (sample.selected_female * sample.total_pool) / sample.selected_group

theorem universiade_volunteer_count :
  ∀ (sample : StratifiedSample),
    sample.total_pool = 200 →
    sample.selected_group = 30 →
    sample.selected_male = 12 →
    sample.selected_female = sample.selected_group - sample.selected_male →
    femaleInPool sample = 120 := by
  sorry

#eval femaleInPool { total_pool := 200, selected_group := 30, selected_male := 12, selected_female := 18 }

end NUMINAMATH_CALUDE_universiade_volunteer_count_l3806_380623


namespace NUMINAMATH_CALUDE_calculate_swimming_speed_triathlete_swimming_speed_l3806_380638

/-- Calculates the swimming speed given the total distance, running speed, and average speed -/
theorem calculate_swimming_speed 
  (total_distance : ℝ) 
  (running_distance : ℝ) 
  (running_speed : ℝ) 
  (average_speed : ℝ) : ℝ :=
  let swimming_distance := total_distance - running_distance
  let total_time := total_distance / average_speed
  let running_time := running_distance / running_speed
  let swimming_time := total_time - running_time
  let swimming_speed := swimming_distance / swimming_time
  swimming_speed

/-- Proves that the swimming speed is 6 miles per hour given the problem conditions -/
theorem triathlete_swimming_speed :
  calculate_swimming_speed 8 4 10 7.5 = 6 := by
  sorry

end NUMINAMATH_CALUDE_calculate_swimming_speed_triathlete_swimming_speed_l3806_380638


namespace NUMINAMATH_CALUDE_factorial_sum_equality_l3806_380634

theorem factorial_sum_equality : 6 * Nat.factorial 6 + 5 * Nat.factorial 5 + 3 * Nat.factorial 4 + Nat.factorial 4 = 1416 := by
  sorry

end NUMINAMATH_CALUDE_factorial_sum_equality_l3806_380634


namespace NUMINAMATH_CALUDE_english_score_is_67_l3806_380617

def mathematics_score : ℕ := 76
def science_score : ℕ := 65
def social_studies_score : ℕ := 82
def biology_score : ℕ := 85
def average_marks : ℕ := 75
def total_subjects : ℕ := 5

def english_score : ℕ := average_marks * total_subjects - (mathematics_score + science_score + social_studies_score + biology_score)

theorem english_score_is_67 : english_score = 67 := by
  sorry

end NUMINAMATH_CALUDE_english_score_is_67_l3806_380617


namespace NUMINAMATH_CALUDE_unique_seq_largest_gt_100_l3806_380605

/-- A sequence of 9 positive integers with unique sums property -/
def UniqueSeq (a : Fin 9 → ℕ) : Prop :=
  (∀ i j, i < j → a i < a j) ∧
  (∀ s₁ s₂ : Finset (Fin 9), s₁ ≠ s₂ → s₁.sum a ≠ s₂.sum a)

/-- Theorem: In a sequence with unique sums property, the largest element is greater than 100 -/
theorem unique_seq_largest_gt_100 (a : Fin 9 → ℕ) (h : UniqueSeq a) : a 8 > 100 := by
  sorry

end NUMINAMATH_CALUDE_unique_seq_largest_gt_100_l3806_380605


namespace NUMINAMATH_CALUDE_square_sum_of_reciprocal_and_sum_l3806_380631

theorem square_sum_of_reciprocal_and_sum (x₁ x₂ : ℝ) :
  x₁ = 2 / (Real.sqrt 5 + Real.sqrt 3) →
  x₂ = Real.sqrt 5 + Real.sqrt 3 →
  x₁^2 + x₂^2 = 16 := by
sorry

end NUMINAMATH_CALUDE_square_sum_of_reciprocal_and_sum_l3806_380631


namespace NUMINAMATH_CALUDE_traffic_light_probability_l3806_380616

/-- Represents a traffic light cycle -/
structure TrafficLightCycle where
  total_duration : ℕ
  red_duration : ℕ
  yellow_duration : ℕ
  green_duration : ℕ

/-- Calculates the probability of waiting no more than a given time -/
def probability_of_waiting (cycle : TrafficLightCycle) (max_wait : ℕ) : ℚ :=
  let proceed_duration := cycle.yellow_duration + cycle.green_duration
  let favorable_duration := min max_wait cycle.red_duration + proceed_duration
  favorable_duration / cycle.total_duration

/-- The main theorem to be proved -/
theorem traffic_light_probability (cycle : TrafficLightCycle) 
  (h1 : cycle.total_duration = 80)
  (h2 : cycle.red_duration = 40)
  (h3 : cycle.yellow_duration = 10)
  (h4 : cycle.green_duration = 30) :
  probability_of_waiting cycle 10 = 5/8 := by
  sorry

end NUMINAMATH_CALUDE_traffic_light_probability_l3806_380616


namespace NUMINAMATH_CALUDE_five_hour_pay_calculation_l3806_380601

/-- Represents the hourly pay rate in dollars -/
def hourly_rate (three_hour_pay six_hour_pay : ℚ) : ℚ :=
  three_hour_pay / 3

/-- Calculates the pay for a given number of hours -/
def calculate_pay (rate : ℚ) (hours : ℚ) : ℚ :=
  rate * hours

theorem five_hour_pay_calculation 
  (three_hour_pay six_hour_pay : ℚ) 
  (h1 : three_hour_pay = 24.75)
  (h2 : six_hour_pay = 49.50)
  (h3 : hourly_rate three_hour_pay six_hour_pay = hourly_rate three_hour_pay six_hour_pay) :
  calculate_pay (hourly_rate three_hour_pay six_hour_pay) 5 = 41.25 := by
  sorry

end NUMINAMATH_CALUDE_five_hour_pay_calculation_l3806_380601


namespace NUMINAMATH_CALUDE_profit_difference_exists_l3806_380628

/-- Represents the strategy of selling or renting a movie -/
inductive SaleStrategy
  | Forever
  | Rental

/-- Represents the economic factors affecting movie sales -/
structure EconomicFactors where
  price : ℝ
  customerBase : ℕ
  sharingRate : ℝ
  rentalFrequency : ℕ
  adminCosts : ℝ
  piracyRisk : ℝ

/-- Calculates the total profit for a given sale strategy and economic factors -/
def totalProfit (strategy : SaleStrategy) (factors : EconomicFactors) : ℝ :=
  sorry

/-- Theorem stating that the total profit from selling a movie "forever" 
    may be different from the total profit from temporary rentals -/
theorem profit_difference_exists :
  ∃ (f₁ f₂ : EconomicFactors), 
    totalProfit SaleStrategy.Forever f₁ ≠ totalProfit SaleStrategy.Rental f₂ :=
  sorry

end NUMINAMATH_CALUDE_profit_difference_exists_l3806_380628


namespace NUMINAMATH_CALUDE_paving_company_calculation_l3806_380673

/-- Represents the properties of a street paved with cement -/
structure Street where
  length : Real
  width : Real
  thickness : Real
  cement_used : Real

/-- Calculates the volume of cement used for a street -/
def cement_volume (s : Street) : Real :=
  s.length * s.width * s.thickness

/-- Cement density in tons per cubic meter -/
def cement_density : Real := 1

theorem paving_company_calculation (lexi_street tess_street : Street) 
  (h1 : lexi_street.length = 200)
  (h2 : lexi_street.width = 10)
  (h3 : lexi_street.thickness = 0.1)
  (h4 : lexi_street.cement_used = 10)
  (h5 : tess_street.length = 100)
  (h6 : tess_street.thickness = 0.1)
  (h7 : tess_street.cement_used = 5.1) :
  tess_street.width = 0.51 ∧ lexi_street.cement_used + tess_street.cement_used = 15.1 := by
  sorry


end NUMINAMATH_CALUDE_paving_company_calculation_l3806_380673


namespace NUMINAMATH_CALUDE_hot_chocolate_consumption_l3806_380625

/-- The number of cups of hot chocolate Tom can drink in 5 hours -/
def cups_in_five_hours : ℕ := 15

/-- The time interval between each cup of hot chocolate in minutes -/
def interval_minutes : ℕ := 20

/-- The total time in hours -/
def total_time_hours : ℕ := 5

/-- Theorem stating the number of cups Tom can drink in 5 hours -/
theorem hot_chocolate_consumption :
  cups_in_five_hours = (total_time_hours * 60) / interval_minutes :=
by sorry

end NUMINAMATH_CALUDE_hot_chocolate_consumption_l3806_380625


namespace NUMINAMATH_CALUDE_perpendicular_tangents_range_l3806_380626

open Real

theorem perpendicular_tangents_range (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧
    let f : ℝ → ℝ := λ x => a * x + sin x + cos x
    let f' : ℝ → ℝ := λ x => a + cos x - sin x
    (f' x₁) * (f' x₂) = -1) →
  -1 ≤ a ∧ a ≤ 1 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_tangents_range_l3806_380626


namespace NUMINAMATH_CALUDE_program_output_l3806_380645

-- Define the program steps as a function
def program (a b : Int) : (Int × Int × Int) :=
  let a' := if a < 0 then -a else a
  let b' := b * b
  let a'' := a' + b'
  let c := a'' - 2 * b'
  let a''' := a'' / c
  let b'' := b' * c + 1
  (a''', b'', c)

-- State the theorem
theorem program_output : program (-6) 2 = (5, 9, 2) := by
  sorry

end NUMINAMATH_CALUDE_program_output_l3806_380645


namespace NUMINAMATH_CALUDE_total_ice_cubes_l3806_380644

/-- The number of ice cubes Dave originally had -/
def original_cubes : ℕ := 2

/-- The number of new ice cubes Dave made -/
def new_cubes : ℕ := 7

/-- Theorem: The total number of ice cubes Dave had is 9 -/
theorem total_ice_cubes : original_cubes + new_cubes = 9 := by
  sorry

end NUMINAMATH_CALUDE_total_ice_cubes_l3806_380644


namespace NUMINAMATH_CALUDE_twenty_political_science_majors_l3806_380632

/-- The number of applicants who majored in political science -/
def political_science_majors (total : ℕ) (high_gpa : ℕ) (not_ps_low_gpa : ℕ) (ps_high_gpa : ℕ) : ℕ :=
  total - not_ps_low_gpa - (high_gpa - ps_high_gpa)

/-- Theorem stating that 20 applicants majored in political science -/
theorem twenty_political_science_majors :
  political_science_majors 40 20 10 5 = 20 := by
  sorry

#eval political_science_majors 40 20 10 5

end NUMINAMATH_CALUDE_twenty_political_science_majors_l3806_380632


namespace NUMINAMATH_CALUDE_three_digit_divisible_by_11_and_5_l3806_380667

theorem three_digit_divisible_by_11_and_5 : 
  (Finset.filter (fun n => n % 55 = 0) (Finset.range 900 ⊔ Finset.range 100)).card = 17 := by
  sorry

end NUMINAMATH_CALUDE_three_digit_divisible_by_11_and_5_l3806_380667


namespace NUMINAMATH_CALUDE_pure_imaginary_condition_l3806_380635

theorem pure_imaginary_condition (a : ℝ) : 
  (∃ (b : ℝ), (Complex.I : ℂ) * b = (1 + Complex.I) / (1 + a * Complex.I)) → a = -1 :=
by sorry

end NUMINAMATH_CALUDE_pure_imaginary_condition_l3806_380635


namespace NUMINAMATH_CALUDE_AAA_not_sufficient_for_congruence_l3806_380686

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  α : ℝ
  β : ℝ
  γ : ℝ
  sum_angles : α + β + γ = π

-- Define triangle congruence
def congruent (t1 t2 : Triangle) : Prop :=
  t1.a = t2.a ∧ t1.b = t2.b ∧ t1.c = t2.c

-- Define AAA criterion
def AAA_equal (t1 t2 : Triangle) : Prop :=
  t1.α = t2.α ∧ t1.β = t2.β ∧ t1.γ = t2.γ

-- Theorem: AAA criterion is not sufficient for triangle congruence
theorem AAA_not_sufficient_for_congruence :
  ¬(∀ (t1 t2 : Triangle), AAA_equal t1 t2 → congruent t1 t2) :=
sorry

end NUMINAMATH_CALUDE_AAA_not_sufficient_for_congruence_l3806_380686
