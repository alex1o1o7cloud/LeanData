import Mathlib

namespace NUMINAMATH_CALUDE_unit_vector_parallel_to_d_l4181_418126

def vector_d : Fin 2 → ℝ := ![12, 5]

theorem unit_vector_parallel_to_d :
  let magnitude : ℝ := Real.sqrt (12^2 + 5^2)
  let unit_vector_positive : Fin 2 → ℝ := ![12 / magnitude, 5 / magnitude]
  let unit_vector_negative : Fin 2 → ℝ := ![-12 / magnitude, -5 / magnitude]
  (∀ i, vector_d i = magnitude * unit_vector_positive i) ∧
  (∀ i, vector_d i = magnitude * unit_vector_negative i) ∧
  (∀ i, unit_vector_positive i * unit_vector_positive i + 
        unit_vector_negative i * unit_vector_negative i = 2) :=
by sorry

end NUMINAMATH_CALUDE_unit_vector_parallel_to_d_l4181_418126


namespace NUMINAMATH_CALUDE_toucan_count_l4181_418167

theorem toucan_count (initial_toucans : ℕ) (joining_toucans : ℕ) : 
  initial_toucans = 2 → joining_toucans = 1 → initial_toucans + joining_toucans = 3 := by
  sorry

end NUMINAMATH_CALUDE_toucan_count_l4181_418167


namespace NUMINAMATH_CALUDE_obtuse_triangle_x_range_l4181_418164

/-- A triangle with sides a, b, and c is obtuse if and only if 
    the square of the longest side is greater than the sum of 
    squares of the other two sides. -/
def is_obtuse_triangle (a b c : ℝ) : Prop :=
  (a ≤ b ∧ b ≤ c ∧ a^2 + b^2 < c^2) ∨
  (a ≤ c ∧ c ≤ b ∧ a^2 + c^2 < b^2) ∨
  (b ≤ a ∧ a ≤ c ∧ b^2 + a^2 < c^2) ∨
  (b ≤ c ∧ c ≤ a ∧ b^2 + c^2 < a^2) ∨
  (c ≤ a ∧ a ≤ b ∧ c^2 + a^2 < b^2) ∨
  (c ≤ b ∧ b ≤ a ∧ c^2 + b^2 < a^2)

theorem obtuse_triangle_x_range :
  ∀ x : ℝ, is_obtuse_triangle x (x + 1) (x + 2) → 1 < x ∧ x < 3 :=
by sorry

end NUMINAMATH_CALUDE_obtuse_triangle_x_range_l4181_418164


namespace NUMINAMATH_CALUDE_path_cost_calculation_l4181_418197

/-- Calculates the total cost of building paths around a rectangular plot -/
def calculate_path_cost (plot_length : Real) (plot_width : Real) 
                        (gravel_path_width : Real) (concrete_path_width : Real)
                        (gravel_cost_per_sqm : Real) (concrete_cost_per_sqm : Real) : Real :=
  let gravel_path_area := 2 * plot_length * gravel_path_width
  let concrete_path_area := 2 * plot_width * concrete_path_width
  let gravel_cost := gravel_path_area * gravel_cost_per_sqm
  let concrete_cost := concrete_path_area * concrete_cost_per_sqm
  gravel_cost + concrete_cost

/-- Theorem stating that the total cost of building the paths is approximately Rs. 9.78 -/
theorem path_cost_calculation :
  let plot_length := 120
  let plot_width := 0.85
  let gravel_path_width := 0.05
  let concrete_path_width := 0.07
  let gravel_cost_per_sqm := 0.80
  let concrete_cost_per_sqm := 1.50
  abs (calculate_path_cost plot_length plot_width gravel_path_width concrete_path_width
                           gravel_cost_per_sqm concrete_cost_per_sqm - 9.78) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_path_cost_calculation_l4181_418197


namespace NUMINAMATH_CALUDE_football_pack_cost_proof_l4181_418118

/-- The cost of a pack of football cards -/
def football_pack_cost : ℝ := 2.73

/-- The number of football card packs bought -/
def football_packs : ℕ := 2

/-- The cost of a pack of Pokemon cards -/
def pokemon_pack_cost : ℝ := 4.01

/-- The cost of a deck of baseball cards -/
def baseball_deck_cost : ℝ := 8.95

/-- The total amount spent on cards -/
def total_spent : ℝ := 18.42

theorem football_pack_cost_proof :
  (football_pack_cost * football_packs) + pokemon_pack_cost + baseball_deck_cost = total_spent := by
  sorry

end NUMINAMATH_CALUDE_football_pack_cost_proof_l4181_418118


namespace NUMINAMATH_CALUDE_problem_solution_l4181_418104

theorem problem_solution : ∃ x : ℚ, ((15 - 2 + 4 / 1) / x) * 8 = 77 ∧ x = 136 / 77 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l4181_418104


namespace NUMINAMATH_CALUDE_solution_set_f_range_of_m_l4181_418159

-- Define the functions f and g
def f (x : ℝ) : ℝ := |x - 1|
def g (x m : ℝ) : ℝ := -|x + 3| + m

-- Theorem for part (1)
theorem solution_set_f (x : ℝ) : f x + x^2 - 1 > 0 ↔ x > 1 ∨ x < 0 := by sorry

-- Theorem for part (2)
theorem range_of_m (m : ℝ) : (∃ x, f x < g x m) ↔ m > 4 := by sorry

end NUMINAMATH_CALUDE_solution_set_f_range_of_m_l4181_418159


namespace NUMINAMATH_CALUDE_tangent_line_slope_l4181_418141

/-- If the line y = kx is tangent to the curve y = x + exp(-x), then k = 1 - exp(1) -/
theorem tangent_line_slope (k : ℝ) : 
  (∃ x₀ : ℝ, k * x₀ = x₀ + Real.exp (-x₀) ∧ 
             k = 1 - Real.exp (-x₀)) → 
  k = 1 - Real.exp 1 := by
sorry

end NUMINAMATH_CALUDE_tangent_line_slope_l4181_418141


namespace NUMINAMATH_CALUDE_inequality_proof_l4181_418117

theorem inequality_proof (x : ℝ) : 
  -2 < (x^2 - 10*x + 21) / (x^2 - 6*x + 10) ∧ 
  (x^2 - 10*x + 21) / (x^2 - 6*x + 10) < 3 ↔ 
  3/2 < x ∧ x < 3 :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l4181_418117


namespace NUMINAMATH_CALUDE_infinite_geometric_series_first_term_l4181_418125

theorem infinite_geometric_series_first_term
  (r : ℝ) (S : ℝ) (a : ℝ)
  (h_r : r = (1 : ℝ) / 4)
  (h_S : S = 80)
  (h_sum : S = a / (1 - r))
  : a = 60 := by
  sorry

end NUMINAMATH_CALUDE_infinite_geometric_series_first_term_l4181_418125


namespace NUMINAMATH_CALUDE_sequence_length_l4181_418102

theorem sequence_length (a₁ : ℕ) (aₙ : ℕ) (d : ℤ) (n : ℕ) :
  a₁ = 150 ∧ aₙ = 30 ∧ d = -6 →
  n = 21 ∧ aₙ = a₁ + d * (n - 1) :=
by sorry

end NUMINAMATH_CALUDE_sequence_length_l4181_418102


namespace NUMINAMATH_CALUDE_square_sum_given_diff_and_product_l4181_418157

theorem square_sum_given_diff_and_product (x y : ℝ) 
  (h1 : x - y = 5) 
  (h2 : x * y = 4) : 
  x^2 + y^2 = 33 := by
sorry

end NUMINAMATH_CALUDE_square_sum_given_diff_and_product_l4181_418157


namespace NUMINAMATH_CALUDE_total_wool_is_82_l4181_418105

/-- The number of scarves Aaron makes -/
def aaron_scarves : ℕ := 10

/-- The number of sweaters Aaron makes -/
def aaron_sweaters : ℕ := 5

/-- The number of sweaters Enid makes -/
def enid_sweaters : ℕ := 8

/-- The number of balls of wool used for one scarf -/
def wool_per_scarf : ℕ := 3

/-- The number of balls of wool used for one sweater -/
def wool_per_sweater : ℕ := 4

/-- The total number of balls of wool used by Enid and Aaron -/
def total_wool : ℕ := aaron_scarves * wool_per_scarf + aaron_sweaters * wool_per_sweater + enid_sweaters * wool_per_sweater

theorem total_wool_is_82 : total_wool = 82 := by sorry

end NUMINAMATH_CALUDE_total_wool_is_82_l4181_418105


namespace NUMINAMATH_CALUDE_white_area_of_sign_l4181_418139

/-- Represents a block letter in the sign --/
structure BlockLetter where
  width : ℕ
  height : ℕ
  stroke_width : ℕ
  covered_area : ℕ

/-- Represents the sign --/
structure Sign where
  width : ℕ
  height : ℕ
  letters : List BlockLetter

def m_letter : BlockLetter := {
  width := 7,
  height := 8,
  stroke_width := 2,
  covered_area := 40
}

def a_letter : BlockLetter := {
  width := 7,
  height := 8,
  stroke_width := 2,
  covered_area := 40
}

def t_letter : BlockLetter := {
  width := 7,
  height := 8,
  stroke_width := 2,
  covered_area := 24
}

def h_letter : BlockLetter := {
  width := 7,
  height := 8,
  stroke_width := 2,
  covered_area := 40
}

def math_sign : Sign := {
  width := 28,
  height := 8,
  letters := [m_letter, a_letter, t_letter, h_letter]
}

theorem white_area_of_sign (s : Sign) : 
  s.width * s.height - (s.letters.map BlockLetter.covered_area).sum = 80 :=
by sorry

end NUMINAMATH_CALUDE_white_area_of_sign_l4181_418139


namespace NUMINAMATH_CALUDE_grace_september_earnings_l4181_418194

/-- Calculates Grace's earnings for landscaping in September --/
theorem grace_september_earnings :
  let mowing_rate : ℕ := 6
  let weeding_rate : ℕ := 11
  let mulching_rate : ℕ := 9
  let mowing_hours : ℕ := 63
  let weeding_hours : ℕ := 9
  let mulching_hours : ℕ := 10
  let total_earnings : ℕ := 
    mowing_rate * mowing_hours + 
    weeding_rate * weeding_hours + 
    mulching_rate * mulching_hours
  total_earnings = 567 := by
sorry

end NUMINAMATH_CALUDE_grace_september_earnings_l4181_418194


namespace NUMINAMATH_CALUDE_leap_year_1996_l4181_418153

def is_leap_year (year : ℕ) : Prop :=
  (year % 4 = 0 ∧ year % 100 ≠ 0) ∨ year % 400 = 0

theorem leap_year_1996 : is_leap_year 1996 := by
  sorry

end NUMINAMATH_CALUDE_leap_year_1996_l4181_418153


namespace NUMINAMATH_CALUDE_count_five_digit_palindromes_l4181_418188

/-- A five-digit palindrome is a number of the form abcba where a, b, c are digits and a ≠ 0. -/
def FiveDigitPalindrome (n : ℕ) : Prop :=
  ∃ a b c : ℕ, 
    a ≥ 1 ∧ a ≤ 9 ∧
    b ≥ 0 ∧ b ≤ 9 ∧
    c ≥ 0 ∧ c ≤ 9 ∧
    n = a * 10000 + b * 1000 + c * 100 + b * 10 + a

/-- The count of five-digit palindromes. -/
def CountFiveDigitPalindromes : ℕ := 
  (Finset.range 9).card * (Finset.range 10).card * (Finset.range 10).card

theorem count_five_digit_palindromes :
  CountFiveDigitPalindromes = 900 :=
sorry

end NUMINAMATH_CALUDE_count_five_digit_palindromes_l4181_418188


namespace NUMINAMATH_CALUDE_card_distribution_result_l4181_418106

/-- Represents the card distribution problem --/
def card_distribution (jimmy_initial bob_initial sarah_initial : ℕ)
  (jimmy_to_bob jimmy_to_mary : ℕ)
  (sarah_friends : ℕ) : Prop :=
  let bob_after_jimmy := bob_initial + jimmy_to_bob
  let bob_to_sarah := bob_after_jimmy / 3
  let sarah_after_bob := sarah_initial + bob_to_sarah
  let sarah_to_friends := (sarah_after_bob / sarah_friends) * sarah_friends
  let jimmy_final := jimmy_initial - jimmy_to_bob - jimmy_to_mary
  let sarah_final := sarah_after_bob - sarah_to_friends
  let friends_cards := sarah_to_friends / sarah_friends
  jimmy_final = 50 ∧ sarah_final = 1 ∧ friends_cards = 3

/-- The main theorem stating the result of the card distribution --/
theorem card_distribution_result :
  card_distribution 68 5 7 6 12 3 :=
sorry

end NUMINAMATH_CALUDE_card_distribution_result_l4181_418106


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l4181_418135

theorem quadratic_equation_solution : ∃ x₁ x₂ : ℝ, 
  (x₁^2 + 6*x₁ - 7 = 0) ∧ 
  (x₂^2 + 6*x₂ - 7 = 0) ∧ 
  x₁ = -7 ∧ 
  x₂ = 1 :=
by
  sorry

#check quadratic_equation_solution

end NUMINAMATH_CALUDE_quadratic_equation_solution_l4181_418135


namespace NUMINAMATH_CALUDE_multiply_add_equality_l4181_418180

theorem multiply_add_equality : 45 * 27 + 18 * 45 = 2025 := by
  sorry

end NUMINAMATH_CALUDE_multiply_add_equality_l4181_418180


namespace NUMINAMATH_CALUDE_kim_sweaters_theorem_l4181_418181

/-- The number of sweaters Kim knit on Monday -/
def monday_sweaters : ℕ := 8

/-- The number of sweaters Kim knit on Tuesday -/
def tuesday_sweaters : ℕ := monday_sweaters + 2

/-- The number of sweaters Kim knit on Wednesday -/
def wednesday_sweaters : ℕ := tuesday_sweaters - 4

/-- The number of sweaters Kim knit on Thursday -/
def thursday_sweaters : ℕ := tuesday_sweaters - 4

/-- The number of sweaters Kim knit on Friday -/
def friday_sweaters : ℕ := monday_sweaters / 2

/-- The total number of sweaters Kim knit in the week -/
def total_sweaters : ℕ := monday_sweaters + tuesday_sweaters + wednesday_sweaters + thursday_sweaters + friday_sweaters

theorem kim_sweaters_theorem : total_sweaters = 34 := by
  sorry

end NUMINAMATH_CALUDE_kim_sweaters_theorem_l4181_418181


namespace NUMINAMATH_CALUDE_absolute_difference_mn_l4181_418108

theorem absolute_difference_mn (m n : ℝ) 
  (h1 : m * n = 6)
  (h2 : m + n = 7)
  (h3 : m^2 - n^2 = 13) : 
  |m - n| = 13/7 := by sorry

end NUMINAMATH_CALUDE_absolute_difference_mn_l4181_418108


namespace NUMINAMATH_CALUDE_remainder_theorem_l4181_418133

theorem remainder_theorem (x : ℤ) : x % 63 = 25 → x % 8 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l4181_418133


namespace NUMINAMATH_CALUDE_klinker_double_age_in_15_years_l4181_418166

/-- The number of years it will take for Mr. Klinker to be twice as old as his daughter -/
def years_until_double_age (klinker_age : ℕ) (daughter_age : ℕ) : ℕ :=
  (klinker_age - 2 * daughter_age)

/-- Proof that it will take 15 years for Mr. Klinker to be twice as old as his daughter -/
theorem klinker_double_age_in_15_years :
  years_until_double_age 35 10 = 15 := by
  sorry

#eval years_until_double_age 35 10

end NUMINAMATH_CALUDE_klinker_double_age_in_15_years_l4181_418166


namespace NUMINAMATH_CALUDE_greatest_divisor_four_consecutive_integers_l4181_418121

theorem greatest_divisor_four_consecutive_integers :
  ∃ (n : ℕ), n > 0 ∧ 
  (∀ (k : ℕ), k > 0 → 12 ∣ (k * (k + 1) * (k + 2) * (k + 3))) ∧
  (∀ (m : ℕ), m > 12 → ∃ (l : ℕ), l > 0 ∧ ¬(m ∣ (l * (l + 1) * (l + 2) * (l + 3)))) := by
  sorry

end NUMINAMATH_CALUDE_greatest_divisor_four_consecutive_integers_l4181_418121


namespace NUMINAMATH_CALUDE_smallest_prime_divisor_of_sum_l4181_418103

theorem smallest_prime_divisor_of_sum : 
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ (3^19 + 11^13) ∧ ∀ q, Nat.Prime q → q ∣ (3^19 + 11^13) → p ≤ q :=
by sorry

end NUMINAMATH_CALUDE_smallest_prime_divisor_of_sum_l4181_418103


namespace NUMINAMATH_CALUDE_max_min_product_l4181_418182

theorem max_min_product (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c)
  (h4 : a + b + c = 8) (h5 : a * b + b * c + c * a = 16) :
  ∃ (m : ℝ), m = min (a * b) (min (b * c) (c * a)) ∧ m ≤ 16 / 9 ∧
  ∀ (m' : ℝ), m' = min (a * b) (min (b * c) (c * a)) → m' ≤ 16 / 9 :=
sorry

end NUMINAMATH_CALUDE_max_min_product_l4181_418182


namespace NUMINAMATH_CALUDE_cristina_photos_l4181_418186

theorem cristina_photos (total_slots : ℕ) (john_photos sarah_photos clarissa_photos : ℕ) 
  (h1 : total_slots = 40)
  (h2 : john_photos = 10)
  (h3 : sarah_photos = 9)
  (h4 : clarissa_photos = 14)
  (h5 : ∃ (cristina_photos : ℕ), cristina_photos + john_photos + sarah_photos + clarissa_photos = total_slots) :
  ∃ (cristina_photos : ℕ), cristina_photos = 7 := by
sorry

end NUMINAMATH_CALUDE_cristina_photos_l4181_418186


namespace NUMINAMATH_CALUDE_snake_paint_theorem_l4181_418107

/-- The amount of paint needed for a single cube -/
def paint_per_cube : ℕ := 60

/-- The number of cubes in the snake -/
def total_cubes : ℕ := 2016

/-- The number of cubes in each periodic fragment -/
def cubes_per_fragment : ℕ := 6

/-- The additional paint needed for adjustments -/
def additional_paint : ℕ := 20

/-- The total amount of paint needed for the snake -/
def total_paint_needed : ℕ :=
  (total_cubes / cubes_per_fragment) * (cubes_per_fragment * paint_per_cube) + additional_paint

theorem snake_paint_theorem :
  total_paint_needed = 120980 := by
  sorry

end NUMINAMATH_CALUDE_snake_paint_theorem_l4181_418107


namespace NUMINAMATH_CALUDE_set_properties_l4181_418100

def closed_under_transformation (A : Set ℝ) : Prop :=
  ∀ a ∈ A, (1 + a) / (1 - a) ∈ A

theorem set_properties (A : Set ℝ) (h : closed_under_transformation A) :
  (2 ∈ A → A = {2, -3, -1/2, 1/3}) ∧
  (0 ∉ A ∧ ∃ a ∈ A, A = {a, -a/(a+1), -1/(a+1), 1/(a-1)}) :=
sorry

end NUMINAMATH_CALUDE_set_properties_l4181_418100


namespace NUMINAMATH_CALUDE_systematic_sampling_last_id_l4181_418120

theorem systematic_sampling_last_id 
  (total_students : Nat) 
  (sample_size : Nat) 
  (first_id : Nat) 
  (h1 : total_students = 2000) 
  (h2 : sample_size = 50) 
  (h3 : first_id = 3) :
  let interval := total_students / sample_size
  let last_id := first_id + interval * (sample_size - 1)
  last_id = 1963 := by
sorry

end NUMINAMATH_CALUDE_systematic_sampling_last_id_l4181_418120


namespace NUMINAMATH_CALUDE_inscribed_rectangle_area_coefficient_l4181_418183

/-- Triangle ABC with inscribed rectangle PQRS --/
structure TriangleWithRectangle where
  /-- Side lengths of triangle ABC --/
  AB : ℝ
  BC : ℝ
  CA : ℝ
  /-- Coefficient of x in the area formula --/
  a : ℝ
  /-- Coefficient of x^2 in the area formula --/
  b : ℝ
  /-- The area of rectangle PQRS is given by a * x - b * x^2 --/
  area_formula : ∀ x, 0 ≤ x → x ≤ BC → 0 ≤ a * x - b * x^2

/-- The main theorem --/
theorem inscribed_rectangle_area_coefficient
  (t : TriangleWithRectangle)
  (h1 : t.AB = 13)
  (h2 : t.BC = 24)
  (h3 : t.CA = 15) :
  t.b = 13 / 48 :=
sorry

end NUMINAMATH_CALUDE_inscribed_rectangle_area_coefficient_l4181_418183


namespace NUMINAMATH_CALUDE_wedding_tables_l4181_418173

theorem wedding_tables (total_fish : ℕ) (fish_per_regular_table : ℕ) (fish_at_special_table : ℕ) :
  total_fish = 65 →
  fish_per_regular_table = 2 →
  fish_at_special_table = 3 →
  ∃ (num_tables : ℕ), num_tables * fish_per_regular_table + (fish_at_special_table - fish_per_regular_table) = total_fish ∧
                       num_tables = 32 := by
  sorry

end NUMINAMATH_CALUDE_wedding_tables_l4181_418173


namespace NUMINAMATH_CALUDE_point_on_circle_x_value_l4181_418185

/-- A circle in the xy-plane with diameter endpoints (-3,0) and (21,0) -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ
  h1 : center = (9, 0)
  h2 : radius = 12

/-- A point on the circle -/
structure PointOnCircle (c : Circle) where
  x : ℝ
  y : ℝ
  h : (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2

theorem point_on_circle_x_value (c : Circle) (p : PointOnCircle c) (h : p.y = 12) :
  p.x = 9 := by
  sorry

end NUMINAMATH_CALUDE_point_on_circle_x_value_l4181_418185


namespace NUMINAMATH_CALUDE_mod_congruence_unique_solution_l4181_418119

theorem mod_congruence_unique_solution : 
  ∃! n : ℤ, 1 ≤ n ∧ n ≤ 9 ∧ n ≡ -245 [ZMOD 10] ∧ n = 5 := by
  sorry

end NUMINAMATH_CALUDE_mod_congruence_unique_solution_l4181_418119


namespace NUMINAMATH_CALUDE_gcd_228_1995_l4181_418176

theorem gcd_228_1995 : Nat.gcd 228 1995 = 57 := by
  sorry

end NUMINAMATH_CALUDE_gcd_228_1995_l4181_418176


namespace NUMINAMATH_CALUDE_trigonometric_identity_l4181_418165

theorem trigonometric_identity (α β γ : Real) 
  (h : (Real.sin (β + γ) * Real.sin (γ + α)) / (Real.cos α * Real.cos γ) = 4/9) :
  (Real.sin (β + γ) * Real.sin (γ + α)) / (Real.cos (α + β + γ) * Real.cos γ) = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l4181_418165


namespace NUMINAMATH_CALUDE_total_passengers_taking_l4181_418198

/-- Represents a train type with its characteristics -/
structure TrainType where
  interval : ℕ  -- Arrival interval in minutes
  leaving : ℕ   -- Number of passengers leaving
  taking : ℕ    -- Number of passengers taking

/-- Calculates the number of trains per hour given the arrival interval -/
def trainsPerHour (interval : ℕ) : ℕ := 60 / interval

/-- Calculates the total passengers for a given operation (leaving or taking) per hour -/
def totalPassengers (t : TrainType) (op : TrainType → ℕ) : ℕ :=
  (trainsPerHour t.interval) * (op t)

/-- Theorem: The total number of unique passengers taking trains at each station during an hour is 4360 -/
theorem total_passengers_taking (stationCount : ℕ) (type1 type2 type3 : TrainType) :
  stationCount = 4 →
  type1 = { interval := 10, leaving := 200, taking := 320 } →
  type2 = { interval := 15, leaving := 300, taking := 400 } →
  type3 = { interval := 20, leaving := 150, taking := 280 } →
  (totalPassengers type1 TrainType.taking +
   totalPassengers type2 TrainType.taking +
   totalPassengers type3 TrainType.taking) = 4360 :=
by sorry

end NUMINAMATH_CALUDE_total_passengers_taking_l4181_418198


namespace NUMINAMATH_CALUDE_production_scaling_l4181_418156

theorem production_scaling (x z : ℝ) (h : x > 0) :
  let production (n : ℝ) := n * n * n * (2 / n)
  production x = 2 * x^2 →
  production z = 2 * z^3 / x :=
by sorry

end NUMINAMATH_CALUDE_production_scaling_l4181_418156


namespace NUMINAMATH_CALUDE_nonparallel_side_length_l4181_418145

/-- A trapezoid inscribed in a circle -/
structure InscribedTrapezoid where
  /-- Radius of the circle -/
  r : ℝ
  /-- Length of each parallel side -/
  a : ℝ
  /-- Length of each non-parallel side -/
  x : ℝ
  /-- The trapezoid is inscribed in the circle -/
  inscribed : True
  /-- The parallel sides are equal -/
  parallel_equal : True
  /-- The non-parallel sides are equal -/
  nonparallel_equal : True

/-- Theorem stating the length of non-parallel sides in the specific trapezoid -/
theorem nonparallel_side_length (t : InscribedTrapezoid) 
  (h1 : t.r = 300) 
  (h2 : t.a = 150) : 
  t.x = 300 := by
  sorry

end NUMINAMATH_CALUDE_nonparallel_side_length_l4181_418145


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_ratio_l4181_418149

theorem hyperbola_asymptote_ratio (a b : ℝ) (h1 : a > b) (h2 : b > 0) : 
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) → 
  (Real.arctan (2 * (b / a) / (1 - (b / a)^2)) = π / 4) →
  a / b = 1 / (-1 + Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_ratio_l4181_418149


namespace NUMINAMATH_CALUDE_hot_dogs_per_pack_l4181_418115

theorem hot_dogs_per_pack (total_hot_dogs : ℕ) (buns_per_pack : ℕ) (hot_dogs_per_pack : ℕ) : 
  total_hot_dogs = 36 →
  buns_per_pack = 9 →
  total_hot_dogs % buns_per_pack = 0 →
  total_hot_dogs % hot_dogs_per_pack = 0 →
  total_hot_dogs / buns_per_pack = total_hot_dogs / hot_dogs_per_pack →
  hot_dogs_per_pack = 9 := by
  sorry

end NUMINAMATH_CALUDE_hot_dogs_per_pack_l4181_418115


namespace NUMINAMATH_CALUDE_swimmers_pass_count_l4181_418114

/-- Represents a swimmer in the pool --/
structure Swimmer where
  speed : ℝ
  restTime : ℝ

/-- Calculates the number of times swimmers pass each other --/
def countPasses (poolLength : ℝ) (duration : ℝ) (swimmer1 : Swimmer) (swimmer2 : Swimmer) : ℕ :=
  sorry

/-- The main theorem --/
theorem swimmers_pass_count :
  let poolLength : ℝ := 120
  let duration : ℝ := 15 * 60  -- 15 minutes in seconds
  let swimmer1 : Swimmer := { speed := 4, restTime := 30 }
  let swimmer2 : Swimmer := { speed := 3, restTime := 0 }
  countPasses poolLength duration swimmer1 swimmer2 = 17 := by
  sorry

end NUMINAMATH_CALUDE_swimmers_pass_count_l4181_418114


namespace NUMINAMATH_CALUDE_min_cans_proof_l4181_418168

/-- The capacity of a special edition soda can in ounces -/
def can_capacity : ℕ := 15

/-- Half a gallon in ounces -/
def half_gallon : ℕ := 64

/-- The minimum number of cans needed to provide at least half a gallon of soda -/
def min_cans : ℕ := 5

theorem min_cans_proof :
  (∀ n : ℕ, n * can_capacity ≥ half_gallon → n ≥ min_cans) ∧
  (min_cans * can_capacity ≥ half_gallon) :=
sorry

end NUMINAMATH_CALUDE_min_cans_proof_l4181_418168


namespace NUMINAMATH_CALUDE_union_subset_intersection_implies_a_equals_one_l4181_418171

-- Define the sets A and B
def A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 1}
def B (a : ℝ) : Set ℝ := {x | -1 ≤ x ∧ x ≤ a}

-- State the theorem
theorem union_subset_intersection_implies_a_equals_one (a : ℝ) :
  (A ∪ B a) ⊆ (A ∩ B a) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_union_subset_intersection_implies_a_equals_one_l4181_418171


namespace NUMINAMATH_CALUDE_watermelon_cost_proof_l4181_418137

/-- The cost of one watermelon in rubles -/
def watermelon_cost : ℕ := 100

/-- The total number of fruits in the container -/
def total_fruits : ℕ := 150

/-- The total value of all fruits in rubles -/
def total_value : ℕ := 24000

/-- The number of melons that can fit in the container -/
def melon_capacity : ℕ := 120

/-- The number of watermelons that can fit in the container -/
def watermelon_capacity : ℕ := 160

theorem watermelon_cost_proof :
  ∃ (num_watermelons num_melons : ℕ) (melon_cost : ℕ),
    num_watermelons + num_melons = total_fruits ∧
    num_watermelons * watermelon_cost = num_melons * melon_cost ∧
    num_watermelons * watermelon_cost + num_melons * melon_cost = total_value ∧
    num_watermelons * melon_capacity = num_melons * watermelon_capacity :=
by sorry

end NUMINAMATH_CALUDE_watermelon_cost_proof_l4181_418137


namespace NUMINAMATH_CALUDE_sequence_squared_l4181_418196

def sequence_property (a : ℕ → ℝ) : Prop :=
  ∀ m n, m ≥ n → a (m + n) + a (m - n) = (a (2 * m) + a (2 * n)) / 2

theorem sequence_squared (a : ℕ → ℝ) (h : sequence_property a) (h1 : a 1 = 1) :
  ∀ n : ℕ, a n = n^2 := by
  sorry

#check sequence_squared

end NUMINAMATH_CALUDE_sequence_squared_l4181_418196


namespace NUMINAMATH_CALUDE_first_half_total_score_l4181_418111

/-- Represents the score of a team in a basketball game -/
structure Score where
  quarter1 : ℚ
  quarter2 : ℚ
  quarter3 : ℚ
  quarter4 : ℚ

/-- The Eagles' score -/
def eagles : Score :=
  { quarter1 := 1/2,
    quarter2 := 1/2 * 2,
    quarter3 := 1/2 * 2^2,
    quarter4 := 1/2 * 2^3 }

/-- The Tigers' score -/
def tigers : Score :=
  { quarter1 := 5,
    quarter2 := 5,
    quarter3 := 5,
    quarter4 := 5 }

/-- Total score for a team -/
def totalScore (s : Score) : ℚ :=
  s.quarter1 + s.quarter2 + s.quarter3 + s.quarter4

/-- First half score for a team -/
def firstHalfScore (s : Score) : ℚ :=
  s.quarter1 + s.quarter2

/-- Theorem stating the total first half score -/
theorem first_half_total_score :
  ⌈firstHalfScore eagles⌉ + ⌈firstHalfScore tigers⌉ = 19 ∧
  eagles.quarter1 = tigers.quarter1 ∧
  totalScore eagles = totalScore tigers + 2 ∧
  totalScore eagles ≤ 100 ∧
  totalScore tigers ≤ 100 :=
sorry


end NUMINAMATH_CALUDE_first_half_total_score_l4181_418111


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l4181_418148

theorem regular_polygon_sides (n : ℕ) (interior_angle : ℝ) (exterior_angle : ℝ) : 
  interior_angle = 150 →
  exterior_angle = 180 - interior_angle →
  n * exterior_angle = 360 →
  n = 12 := by
  sorry

#check regular_polygon_sides

end NUMINAMATH_CALUDE_regular_polygon_sides_l4181_418148


namespace NUMINAMATH_CALUDE_perpendicular_lines_a_equals_one_l4181_418128

/-- Given two lines l₁: ax + (3-a)y + 1 = 0 and l₂: 2x - y = 0,
    if l₁ is perpendicular to l₂, then a = 1 -/
theorem perpendicular_lines_a_equals_one (a : ℝ) :
  (∀ x y : ℝ, a * x + (3 - a) * y + 1 = 0 → 2 * x - y = 0 → 
    (a * 2 + (-1) * (3 - a) = 0)) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_lines_a_equals_one_l4181_418128


namespace NUMINAMATH_CALUDE_lemonade_pitchers_l4181_418134

/-- Represents the number of glasses a pitcher can serve -/
def glasses_per_pitcher : ℕ := 5

/-- Represents the total number of glasses served -/
def total_glasses_served : ℕ := 30

/-- Calculates the number of pitchers needed -/
def pitchers_needed : ℕ := total_glasses_served / glasses_per_pitcher

theorem lemonade_pitchers : pitchers_needed = 6 := by
  sorry

end NUMINAMATH_CALUDE_lemonade_pitchers_l4181_418134


namespace NUMINAMATH_CALUDE_tournament_27_teams_26_games_l4181_418195

/-- Represents a single-elimination tournament -/
structure Tournament where
  num_teams : ℕ
  no_ties : Bool

/-- Number of games needed to determine a winner in a single-elimination tournament -/
def games_to_winner (t : Tournament) : ℕ :=
  t.num_teams - 1

/-- Theorem: A single-elimination tournament with 27 teams requires 26 games to determine a winner -/
theorem tournament_27_teams_26_games :
  ∀ (t : Tournament), t.num_teams = 27 → t.no_ties = true → games_to_winner t = 26 := by
  sorry

end NUMINAMATH_CALUDE_tournament_27_teams_26_games_l4181_418195


namespace NUMINAMATH_CALUDE_discount_order_difference_l4181_418191

/-- Calculates the difference in final price when applying discounts in different orders -/
theorem discount_order_difference : 
  let original_price : ℚ := 30
  let fixed_discount : ℚ := 5
  let percentage_discount : ℚ := 0.25
  let scenario1 := (original_price - fixed_discount) * (1 - percentage_discount)
  let scenario2 := (original_price * (1 - percentage_discount)) - fixed_discount
  (scenario2 - scenario1) * 100 = 125 := by sorry

end NUMINAMATH_CALUDE_discount_order_difference_l4181_418191


namespace NUMINAMATH_CALUDE_trigonometric_equation_solution_l4181_418143

open Real

theorem trigonometric_equation_solution :
  ∀ x : ℝ, 2 * sin (2 * x) - cos (π / 2 + 3 * x) - cos (3 * x) * arccos (5 * x) * cos (π / 2 - 5 * x) = 0 ↔
  (∃ k : ℤ, x = k * π) ∨ (∃ n : ℤ, x = π / 15 + 2 * n * π / 5) ∨ (∃ n : ℤ, x = -π / 15 + 2 * n * π / 5) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_equation_solution_l4181_418143


namespace NUMINAMATH_CALUDE_profit_margin_calculation_l4181_418101

/-- Profit margin calculation -/
theorem profit_margin_calculation (n : ℝ) (C S M : ℝ) 
  (h1 : M = (1 / n) * (2 * C - S)) 
  (h2 : S - M = C) : 
  M = S / (n + 2) := by
  sorry

end NUMINAMATH_CALUDE_profit_margin_calculation_l4181_418101


namespace NUMINAMATH_CALUDE_two_consecutive_increases_l4181_418112

theorem two_consecutive_increases (initial : ℝ) (increase1 : ℝ) (increase2 : ℝ) : 
  let after_first_increase := initial * (1 + increase1 / 100)
  let final_number := after_first_increase * (1 + increase2 / 100)
  initial = 1256 ∧ increase1 = 325 ∧ increase2 = 147 → final_number = 6000.54 := by
sorry

end NUMINAMATH_CALUDE_two_consecutive_increases_l4181_418112


namespace NUMINAMATH_CALUDE_equation_holds_iff_m_equals_168_l4181_418172

theorem equation_holds_iff_m_equals_168 :
  ∀ m : ℤ, (4^4 : ℤ) - 7 = 9^2 + m ↔ m = 168 := by
  sorry

end NUMINAMATH_CALUDE_equation_holds_iff_m_equals_168_l4181_418172


namespace NUMINAMATH_CALUDE_investment_difference_is_1000_l4181_418178

/-- Represents the investment problem with three persons --/
structure InvestmentProblem where
  total_investment : ℕ
  total_gain : ℕ
  third_person_gain : ℕ

/-- Calculates the investment difference between the second and first person --/
def investment_difference (problem : InvestmentProblem) : ℕ :=
  let first_investment := problem.total_investment / 3
  let second_investment := first_investment + (problem.total_investment / 3 - first_investment)
  second_investment - first_investment

/-- Theorem stating that the investment difference is 1000 for the given problem --/
theorem investment_difference_is_1000 (problem : InvestmentProblem) 
  (h1 : problem.total_investment = 9000)
  (h2 : problem.total_gain = 1800)
  (h3 : problem.third_person_gain = 800) :
  investment_difference problem = 1000 := by
  sorry

#eval investment_difference ⟨9000, 1800, 800⟩

end NUMINAMATH_CALUDE_investment_difference_is_1000_l4181_418178


namespace NUMINAMATH_CALUDE_scientific_notation_216000_l4181_418163

theorem scientific_notation_216000 : 216000 = 2.16 * (10 ^ 5) := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_216000_l4181_418163


namespace NUMINAMATH_CALUDE_expression_change_l4181_418127

/-- The change in the expression x³ - 5x + 1 when x changes by b -/
def expressionChange (x b : ℝ) : ℝ :=
  let f := fun t => t^3 - 5*t + 1
  f (x + b) - f x

theorem expression_change (x b : ℝ) (h : b > 0) :
  expressionChange x b = 3*b*x^2 + 3*b^2*x + b^3 - 5*b ∨
  expressionChange x (-b) = -3*b*x^2 + 3*b^2*x - b^3 + 5*b :=
sorry

end NUMINAMATH_CALUDE_expression_change_l4181_418127


namespace NUMINAMATH_CALUDE_trig_identity_l4181_418136

theorem trig_identity (α : Real) (h : Real.tan α = 2) :
  7 * Real.sin α ^ 2 + 3 * Real.cos α ^ 2 = 31 / 5 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l4181_418136


namespace NUMINAMATH_CALUDE_cost_increase_percentage_l4181_418147

/-- Proves that given the initial profit is 320% of the cost, and after a cost increase
    (with constant selling price) the profit becomes 66.67% of the selling price,
    then the cost increase percentage is 40%. -/
theorem cost_increase_percentage (C : ℝ) (X : ℝ) : 
  C > 0 →                           -- Assuming positive initial cost
  let S := 4.2 * C                  -- Initial selling price
  let new_profit := 3.2 * C - (X / 100) * C  -- New profit after cost increase
  3.2 * C = 320 / 100 * C →         -- Initial profit is 320% of cost
  new_profit = 2 / 3 * S →          -- New profit is 66.67% of selling price
  X = 40 :=                         -- Cost increase percentage is 40%
by
  sorry


end NUMINAMATH_CALUDE_cost_increase_percentage_l4181_418147


namespace NUMINAMATH_CALUDE_exists_polygon_with_area_16_l4181_418154

/-- A polygon represented by a list of points in 2D space -/
def Polygon := List (Real × Real)

/-- Calculate the area of a polygon given its vertices -/
def polygonArea (p : Polygon) : Real := sorry

/-- Check if a polygon can be formed from given line segments -/
def canFormPolygon (segments : List Real) (p : Polygon) : Prop := sorry

/-- The main theorem stating that a polygon with area 16 can be formed from 12 segments of length 2 -/
theorem exists_polygon_with_area_16 :
  ∃ (p : Polygon), 
    polygonArea p = 16 ∧ 
    canFormPolygon (List.replicate 12 2) p :=
sorry

end NUMINAMATH_CALUDE_exists_polygon_with_area_16_l4181_418154


namespace NUMINAMATH_CALUDE_small_rectangle_perimeter_l4181_418161

/-- Given a square with perimeter 160 units, divided into two congruent rectangles,
    with one of those rectangles further divided into three smaller congruent rectangles,
    the perimeter of one of the three smaller congruent rectangles is equal to 2 * (20 + 40/3) units. -/
theorem small_rectangle_perimeter (s : ℝ) (h1 : s > 0) (h2 : 4 * s = 160) :
  2 * (s / 2 + s / 6) = 2 * (20 + 40 / 3) :=
sorry

end NUMINAMATH_CALUDE_small_rectangle_perimeter_l4181_418161


namespace NUMINAMATH_CALUDE_fraction_sum_integer_implies_fractions_integer_l4181_418179

theorem fraction_sum_integer_implies_fractions_integer
  (a b c : ℤ) (h : ∃ (m : ℤ), (a * b) / c + (a * c) / b + (b * c) / a = m) :
  (∃ (k : ℤ), (a * b) / c = k) ∧
  (∃ (l : ℤ), (a * c) / b = l) ∧
  (∃ (n : ℤ), (b * c) / a = n) :=
by sorry

end NUMINAMATH_CALUDE_fraction_sum_integer_implies_fractions_integer_l4181_418179


namespace NUMINAMATH_CALUDE_inequality_proof_l4181_418190

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x < y) :
  x + Real.sqrt (y^2 + 2) < y + Real.sqrt (x^2 + 2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l4181_418190


namespace NUMINAMATH_CALUDE_yoga_time_calculation_l4181_418132

/-- Calculates the yoga time given exercise ratios and bicycle riding time -/
theorem yoga_time_calculation (bicycle_time : ℚ) : 
  bicycle_time = 12 → (40 : ℚ) / 3 = 
    2 * (2 * bicycle_time / 3 + bicycle_time) / 3 := by
  sorry

#eval (40 : ℚ) / 3

end NUMINAMATH_CALUDE_yoga_time_calculation_l4181_418132


namespace NUMINAMATH_CALUDE_chosen_number_calculation_l4181_418123

theorem chosen_number_calculation (x : ℕ) (h : x = 30) : x * 8 - 138 = 102 := by
  sorry

end NUMINAMATH_CALUDE_chosen_number_calculation_l4181_418123


namespace NUMINAMATH_CALUDE_region_is_lower_left_l4181_418124

-- Define the line
def line (x y : ℝ) : Prop := x - 2*y + 6 = 0

-- Define the region
def region (x y : ℝ) : Prop := x - 2*y + 6 < 0

-- Define what it means to be on the lower left side of the line
def lower_left_side (x y : ℝ) : Prop := x - 2*y + 6 < 0

-- Theorem statement
theorem region_is_lower_left : 
  ∀ (x y : ℝ), region x y → lower_left_side x y :=
sorry

end NUMINAMATH_CALUDE_region_is_lower_left_l4181_418124


namespace NUMINAMATH_CALUDE_characterize_f_l4181_418122

def is_valid_f (f : ℕ → ℕ) : Prop :=
  ∀ n, f n ≠ 1 ∧ f n + f (n + 1) = f (n + 2) + f (n + 3) - 168

theorem characterize_f (f : ℕ → ℕ) (h : is_valid_f f) :
  ∃ (c d a : ℕ), (∀ n, f (2 * n) = c + n * d) ∧
                 (∀ n, f (2 * n + 1) = (168 - d) * n + a - c) ∧
                 c > 1 ∧
                 a > c + 1 :=
sorry

end NUMINAMATH_CALUDE_characterize_f_l4181_418122


namespace NUMINAMATH_CALUDE_only_b_opens_upwards_l4181_418189

def quadratic_a (x : ℝ) : ℝ := 1 - x - 6*x^2
def quadratic_b (x : ℝ) : ℝ := -8*x + x^2 + 1
def quadratic_c (x : ℝ) : ℝ := (1 - x)*(x + 5)
def quadratic_d (x : ℝ) : ℝ := 2 - (5 - x)^2

def opens_upwards (f : ℝ → ℝ) : Prop :=
  ∃ a > 0, ∃ b c : ℝ, ∀ x, f x = a*x^2 + b*x + c

theorem only_b_opens_upwards :
  opens_upwards quadratic_b ∧
  ¬opens_upwards quadratic_a ∧
  ¬opens_upwards quadratic_c ∧
  ¬opens_upwards quadratic_d :=
by sorry

end NUMINAMATH_CALUDE_only_b_opens_upwards_l4181_418189


namespace NUMINAMATH_CALUDE_minimum_seating_arrangement_l4181_418129

/-- Represents a circular seating arrangement -/
structure CircularSeating where
  total_chairs : ℕ
  seated_people : ℕ

/-- Checks if the seating arrangement is valid -/
def is_valid_seating (s : CircularSeating) : Prop :=
  s.seated_people > 0 ∧ 
  s.seated_people ≤ s.total_chairs ∧
  s.total_chairs % s.seated_people = 0

/-- Checks if any additional person must sit next to someone -/
def forces_adjacent_seating (s : CircularSeating) : Prop :=
  s.total_chairs / s.seated_people ≤ 4

/-- The main theorem to prove -/
theorem minimum_seating_arrangement :
  ∃ (s : CircularSeating), 
    s.total_chairs = 75 ∧
    is_valid_seating s ∧
    forces_adjacent_seating s ∧
    (∀ (t : CircularSeating), 
      t.total_chairs = 75 → 
      is_valid_seating t → 
      forces_adjacent_seating t → 
      s.seated_people ≤ t.seated_people) ∧
    s.seated_people = 19 :=
  sorry

end NUMINAMATH_CALUDE_minimum_seating_arrangement_l4181_418129


namespace NUMINAMATH_CALUDE_no_such_function_exists_l4181_418174

theorem no_such_function_exists :
  ¬ ∃ f : ℕ+ → ℕ+, ∀ n : ℕ+, (f^[n.val] n = n + 1) := by
  sorry

end NUMINAMATH_CALUDE_no_such_function_exists_l4181_418174


namespace NUMINAMATH_CALUDE_problem_solution_l4181_418109

-- Define the properties of functions f and g
def is_direct_proportion (f : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ x : ℝ, f x = k * x

def is_inverse_proportion (g : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ x : ℝ, x ≠ 0 → g x = k / x

-- Define the conditions given in the problem
def problem_conditions (f g : ℝ → ℝ) : Prop :=
  is_direct_proportion f ∧ is_inverse_proportion g ∧ f 1 = 1 ∧ g 1 = 2

-- Define what it means for a function to be odd
def is_odd_function (h : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, h (-x) = -h x

-- Theorem statement
theorem problem_solution (f g : ℝ → ℝ) (h : problem_conditions f g) :
  (∀ x : ℝ, f x = x) ∧
  (∀ x : ℝ, x ≠ 0 → g x = 2 / x) ∧
  is_odd_function (λ x => f x + g x) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l4181_418109


namespace NUMINAMATH_CALUDE_percentage_passed_all_topics_percentage_passed_all_topics_proof_l4181_418110

/-- The percentage of students who passed in all topics in a practice paper -/
theorem percentage_passed_all_topics : ℝ :=
  let total_students : ℕ := 2500
  let passed_three_topics : ℕ := 500
  let percent_no_pass : ℝ := 10
  let percent_one_topic : ℝ := 20
  let percent_two_topics : ℝ := 25
  let percent_four_topics : ℝ := 24
  let percent_three_topics : ℝ := (passed_three_topics : ℝ) / (total_students : ℝ) * 100

  1 -- This is the percentage we need to prove

theorem percentage_passed_all_topics_proof : percentage_passed_all_topics = 1 := by
  sorry

end NUMINAMATH_CALUDE_percentage_passed_all_topics_percentage_passed_all_topics_proof_l4181_418110


namespace NUMINAMATH_CALUDE_truck_rental_theorem_l4181_418130

/-- Represents the number of trucks on a rental lot -/
structure TruckLot where
  monday : ℕ
  rented : ℕ
  returned : ℕ
  saturday : ℕ

/-- Conditions for the truck rental problem -/
def truck_rental_conditions (lot : TruckLot) : Prop :=
  lot.monday = 20 ∧
  lot.rented ≤ 20 ∧
  lot.returned = lot.rented / 2 ∧
  lot.saturday = lot.monday - lot.rented + lot.returned

theorem truck_rental_theorem (lot : TruckLot) :
  truck_rental_conditions lot → lot.saturday = 10 :=
by
  sorry

#check truck_rental_theorem

end NUMINAMATH_CALUDE_truck_rental_theorem_l4181_418130


namespace NUMINAMATH_CALUDE_fraction_simplification_l4181_418150

theorem fraction_simplification :
  (5 : ℝ) / (2 * Real.sqrt 50 + 3 * Real.sqrt 8 + Real.sqrt 18) = (5 * Real.sqrt 2) / 38 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l4181_418150


namespace NUMINAMATH_CALUDE_power_of_x_in_product_l4181_418144

theorem power_of_x_in_product (x y z : ℕ) (hx : Prime x) (hy : Prime y) (hz : Prime z) 
  (hdiff : x ≠ y ∧ y ≠ z ∧ x ≠ z) :
  ∃ (a b c : ℕ), (a + 1) * (b + 1) * (c + 1) = 12 ∧ a = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_of_x_in_product_l4181_418144


namespace NUMINAMATH_CALUDE_product_expansion_l4181_418193

theorem product_expansion (a b c : ℝ) :
  (a + b) * (b + c) * (c + a) = (a + b + c) * (a * b + b * c + c * a) - a * b * c := by
  sorry

end NUMINAMATH_CALUDE_product_expansion_l4181_418193


namespace NUMINAMATH_CALUDE_antonios_weight_l4181_418138

theorem antonios_weight (A : ℝ) : 
  A + (A - 12) = 88 → A = 50 := by sorry

end NUMINAMATH_CALUDE_antonios_weight_l4181_418138


namespace NUMINAMATH_CALUDE_all_composites_reachable_l4181_418116

/-- A proper divisor of n is a positive integer that divides n and is not equal to 1 or n. -/
def ProperDivisor (d n : ℕ) : Prop :=
  d ∣ n ∧ d ≠ 1 ∧ d ≠ n

/-- The set of numbers that can be obtained by starting from 4 and repeatedly adding proper divisors. -/
inductive Reachable : ℕ → Prop
  | base : Reachable 4
  | step {n m : ℕ} : Reachable n → ProperDivisor m n → Reachable (n + m)

/-- A composite number is a natural number greater than 1 that is not prime. -/
def Composite (n : ℕ) : Prop :=
  n > 1 ∧ ¬ Nat.Prime n

/-- Theorem: Any composite number can be reached by starting from 4 and repeatedly adding proper divisors. -/
theorem all_composites_reachable : ∀ n : ℕ, Composite n → Reachable n := by
  sorry

end NUMINAMATH_CALUDE_all_composites_reachable_l4181_418116


namespace NUMINAMATH_CALUDE_snail_path_count_l4181_418177

theorem snail_path_count (n : ℕ) : 
  (number_of_paths : ℕ) = (Nat.choose (2 * n) n) ^ 2 :=
by
  sorry

where
  number_of_paths : ℕ := 
    count_closed_paths_on_graph_paper (2 * n)

  count_closed_paths_on_graph_paper (steps : ℕ) : ℕ := 
    -- Returns the number of distinct paths on graph paper
    -- that start and end at the same vertex
    -- and have a total length of 'steps'
    sorry

end NUMINAMATH_CALUDE_snail_path_count_l4181_418177


namespace NUMINAMATH_CALUDE_tenth_thousand_digit_is_seven_l4181_418151

def digit_sequence (n : ℕ) : ℕ :=
  let digits_1_to_9 := 9
  let digits_10_to_99 := 90 * 2
  let digits_100_to_999 := 900 * 3
  let digits_1_to_999 := digits_1_to_9 + digits_10_to_99 + digits_100_to_999
  let remaining_digits := n - digits_1_to_999
  let full_numbers_1000_onward := remaining_digits / 4
  let digits_from_full_numbers := full_numbers_1000_onward * 4
  let last_number := 1000 + full_numbers_1000_onward
  let remaining_digits_in_last_number := remaining_digits - digits_from_full_numbers
  if remaining_digits_in_last_number = 0 then
    (last_number - 1) % 10
  else
    (last_number / (10 ^ (4 - remaining_digits_in_last_number))) % 10

theorem tenth_thousand_digit_is_seven :
  digit_sequence 10000 = 7 := by
  sorry

end NUMINAMATH_CALUDE_tenth_thousand_digit_is_seven_l4181_418151


namespace NUMINAMATH_CALUDE_tank_cost_l4181_418142

theorem tank_cost (buy_price sell_price : ℚ) (num_sold : ℕ) (profit_percentage : ℚ) :
  buy_price = 0.25 →
  sell_price = 0.75 →
  num_sold = 110 →
  profit_percentage = 0.55 →
  (sell_price - buy_price) * num_sold = profit_percentage * 100 :=
by sorry

end NUMINAMATH_CALUDE_tank_cost_l4181_418142


namespace NUMINAMATH_CALUDE_tangent_line_at_one_tangent_lines_through_one_l4181_418162

noncomputable section

-- Define the function f(x) = x^3 + a*ln(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a * Real.log x

-- Part I
theorem tangent_line_at_one (a : ℝ) (h : a = 1) :
  ∃ (m b : ℝ), m * 1 - f a 1 + b = 0 ∧
    ∀ x, m * x - (f a x) + b = 0 ↔ 4 * x - (f a x) - 3 = 0 :=
sorry

-- Part II
theorem tangent_lines_through_one (a : ℝ) (h : a = 0) :
  ∃ (m₁ b₁ m₂ b₂ : ℝ), 
    (m₁ * 1 - f a 1 + b₁ = 0 ∧ ∀ x, m₁ * x - (f a x) + b₁ = 0 ↔ 3 * x - (f a x) - 2 = 0) ∧
    (m₂ * 1 - f a 1 + b₂ = 0 ∧ ∀ x, m₂ * x - (f a x) + b₂ = 0 ↔ 3 * x - 4 * (f a x) + 1 = 0) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_at_one_tangent_lines_through_one_l4181_418162


namespace NUMINAMATH_CALUDE_expression_equals_zero_l4181_418152

theorem expression_equals_zero :
  (1 - Real.sqrt 2) ^ 0 + |2 - Real.sqrt 5| + (-1) ^ 2022 - (1/3) * Real.sqrt 45 = 0 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_zero_l4181_418152


namespace NUMINAMATH_CALUDE_storage_box_faces_l4181_418169

theorem storage_box_faces : ∃ n : ℕ, n > 0 ∧ Nat.factorial n = 720 := by
  sorry

end NUMINAMATH_CALUDE_storage_box_faces_l4181_418169


namespace NUMINAMATH_CALUDE_translation_increases_y_l4181_418199

/-- Represents a quadratic function of the form y = ax^2 + bx + c -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a horizontal translation of a function -/
structure HorizontalTranslation where
  units : ℝ

/-- The original quadratic function y = -x^2 + 1 -/
def original_function : QuadraticFunction :=
  { a := -1, b := 0, c := 1 }

/-- The required translation -/
def translation : HorizontalTranslation :=
  { units := 2 }

/-- Theorem stating that the given translation makes y increase as x increases when x < 2 -/
theorem translation_increases_y (f : QuadraticFunction) (t : HorizontalTranslation) :
  f = original_function →
  t = translation →
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → x₂ < 2 →
  f.a * (x₁ - t.units)^2 + f.b * (x₁ - t.units) + f.c <
  f.a * (x₂ - t.units)^2 + f.b * (x₂ - t.units) + f.c :=
by sorry

end NUMINAMATH_CALUDE_translation_increases_y_l4181_418199


namespace NUMINAMATH_CALUDE_ellipse_condition_l4181_418146

def represents_ellipse (m : ℝ) : Prop :=
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a ≠ b ∧ 
  ∀ (x y : ℝ), x^2 / (5 - m) + y^2 / (m + 3) = 1 ↔ x^2 / a^2 + y^2 / b^2 = 1

theorem ellipse_condition (m : ℝ) : 
  represents_ellipse m → m > -3 ∧ m < 5 :=
sorry

end NUMINAMATH_CALUDE_ellipse_condition_l4181_418146


namespace NUMINAMATH_CALUDE_reflection_over_y_axis_l4181_418192

theorem reflection_over_y_axis :
  let reflect_matrix : Matrix (Fin 2) (Fin 2) ℝ := ![![-1, 0], ![0, 1]]
  ∀ (x y : ℝ), 
    reflect_matrix.mulVec ![x, y] = ![-x, y] := by sorry

end NUMINAMATH_CALUDE_reflection_over_y_axis_l4181_418192


namespace NUMINAMATH_CALUDE_same_expected_defects_l4181_418187

/-- Represents a worker's probability distribution of defective products -/
structure Worker where
  p0 : ℝ  -- Probability of 0 defective products
  p1 : ℝ  -- Probability of 1 defective product
  p2 : ℝ  -- Probability of 2 defective products
  p3 : ℝ  -- Probability of 3 defective products
  sum_to_one : p0 + p1 + p2 + p3 = 1
  non_negative : p0 ≥ 0 ∧ p1 ≥ 0 ∧ p2 ≥ 0 ∧ p3 ≥ 0

/-- Calculate the expected number of defective products for a worker -/
def expected_defects (w : Worker) : ℝ :=
  0 * w.p0 + 1 * w.p1 + 2 * w.p2 + 3 * w.p3

/-- Worker A's probability distribution -/
def worker_A : Worker := {
  p0 := 0.4
  p1 := 0.3
  p2 := 0.2
  p3 := 0.1
  sum_to_one := by norm_num
  non_negative := by norm_num
}

/-- Worker B's probability distribution -/
def worker_B : Worker := {
  p0 := 0.4
  p1 := 0.2
  p2 := 0.4
  p3 := 0
  sum_to_one := by norm_num
  non_negative := by norm_num
}

/-- Theorem stating that the expected number of defective products is the same for both workers -/
theorem same_expected_defects : expected_defects worker_A = expected_defects worker_B := by
  sorry

end NUMINAMATH_CALUDE_same_expected_defects_l4181_418187


namespace NUMINAMATH_CALUDE_problem_solution_l4181_418140

theorem problem_solution (x y z : ℝ) (hx : x = 7) (hy : y = -2) (hz : z = 4) :
  ((x - 2*y)^y) / z = 1 / 484 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l4181_418140


namespace NUMINAMATH_CALUDE_not_right_triangle_l4181_418113

theorem not_right_triangle (a b c : ℝ) (h : a = 3 ∧ b = 5 ∧ c = 7) : 
  ¬(a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2) :=
by sorry

end NUMINAMATH_CALUDE_not_right_triangle_l4181_418113


namespace NUMINAMATH_CALUDE_three_digit_swap_subtraction_l4181_418160

theorem three_digit_swap_subtraction (a b c : ℕ) : 
  a ≤ 9 → b ≤ 9 → c ≤ 9 → a ≠ 0 → a = c + 3 →
  (100 * a + 10 * b + c) - (100 * c + 10 * b + a) ≡ 7 [MOD 10] :=
by sorry

end NUMINAMATH_CALUDE_three_digit_swap_subtraction_l4181_418160


namespace NUMINAMATH_CALUDE_remaining_budget_calculation_l4181_418131

def total_budget : ℝ := 80000000
def infrastructure_percentage : ℝ := 0.30
def public_transportation : ℝ := 10000000
def healthcare_percentage : ℝ := 0.15

theorem remaining_budget_calculation :
  total_budget - (infrastructure_percentage * total_budget + public_transportation + healthcare_percentage * total_budget) = 34000000 := by
  sorry

end NUMINAMATH_CALUDE_remaining_budget_calculation_l4181_418131


namespace NUMINAMATH_CALUDE_median_is_212_l4181_418175

/-- Represents the list where each integer n from 1 to 300 appears n times -/
def special_list : List ℕ := sorry

/-- The sum of all elements in the special list -/
def total_elements : ℕ := (300 * (300 + 1)) / 2

/-- The position of the median in the special list -/
def median_position : ℕ × ℕ := (total_elements / 2, total_elements / 2 + 1)

/-- Theorem stating that the median of the special list is 212 -/
theorem median_is_212 : 
  ∃ (median : ℕ), median = 212 ∧ 
  (∃ (l1 l2 : List ℕ), special_list = l1 ++ [median] ++ [median] ++ l2 ∧ 
   l1.length = median_position.1 - 1 ∧
   l2.length = special_list.length - median_position.2) :=
sorry

end NUMINAMATH_CALUDE_median_is_212_l4181_418175


namespace NUMINAMATH_CALUDE_calcium_oxide_weight_l4181_418184

-- Define atomic weights
def atomic_weight_Ca : Real := 40.08
def atomic_weight_O : Real := 16.00

-- Define the compound
structure Compound where
  calcium : Nat
  oxygen : Nat

-- Define molecular weight calculation
def molecular_weight (c : Compound) : Real :=
  c.calcium * atomic_weight_Ca + c.oxygen * atomic_weight_O

-- Theorem to prove
theorem calcium_oxide_weight :
  molecular_weight { calcium := 1, oxygen := 1 } = 56.08 := by
  sorry

end NUMINAMATH_CALUDE_calcium_oxide_weight_l4181_418184


namespace NUMINAMATH_CALUDE_max_sum_of_xy_l4181_418170

theorem max_sum_of_xy (x y : ℕ+) : 
  (x * y : ℕ) - (x + y : ℕ) = Nat.gcd x y + Nat.lcm x y → 
  (∃ (c : ℕ), ∀ (a b : ℕ+), 
    (a * b : ℕ) - (a + b : ℕ) = Nat.gcd a b + Nat.lcm a b → 
    (a + b : ℕ) ≤ c) ∧ 
  (x + y : ℕ) ≤ 10 :=
sorry

end NUMINAMATH_CALUDE_max_sum_of_xy_l4181_418170


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l4181_418158

theorem sqrt_equation_solution :
  ∃! z : ℚ, Real.sqrt (7 - 5 * z) = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l4181_418158


namespace NUMINAMATH_CALUDE_ellipse_axis_ratio_l4181_418155

theorem ellipse_axis_ratio (k : ℝ) : 
  (∀ x y : ℝ, x^2 + k*y^2 = 1 → ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ x^2/a^2 + y^2/b^2 = 1) →
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ x^2/a^2 + y^2/b^2 = 1 ∧ a^2 = 1/k ∧ b^2 = 1) →
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ 2*a = 2*b) →
  k = 1/4 :=
sorry

end NUMINAMATH_CALUDE_ellipse_axis_ratio_l4181_418155
