import Mathlib

namespace NUMINAMATH_CALUDE_comic_book_problem_l1554_155418

theorem comic_book_problem (initial_books : ℕ) : 
  (initial_books / 2 + 6 = 17) → initial_books = 22 :=
by
  sorry

end NUMINAMATH_CALUDE_comic_book_problem_l1554_155418


namespace NUMINAMATH_CALUDE_at_most_one_perfect_square_l1554_155493

def sequence_a : ℕ → ℕ
  | 0 => 1  -- arbitrary starting value
  | n + 1 => (sequence_a n)^3 + 103

theorem at_most_one_perfect_square :
  ∃ k : ℕ, ∀ n m : ℕ, 
    (∃ i : ℕ, sequence_a n = i^2) → 
    (∃ j : ℕ, sequence_a m = j^2) → 
    n = m ∨ (n < k ∧ m < k) := by
  sorry

end NUMINAMATH_CALUDE_at_most_one_perfect_square_l1554_155493


namespace NUMINAMATH_CALUDE_article_word_limit_l1554_155467

/-- Calculates the word limit for an article given specific font and page constraints. -/
theorem article_word_limit 
  (total_pages : ℕ) 
  (large_font_pages : ℕ) 
  (large_font_words_per_page : ℕ) 
  (small_font_words_per_page : ℕ) 
  (h1 : total_pages = 21)
  (h2 : large_font_pages = 4)
  (h3 : large_font_words_per_page = 1800)
  (h4 : small_font_words_per_page = 2400) :
  large_font_pages * large_font_words_per_page + 
  (total_pages - large_font_pages) * small_font_words_per_page = 48000 :=
by sorry

end NUMINAMATH_CALUDE_article_word_limit_l1554_155467


namespace NUMINAMATH_CALUDE_container_capacity_l1554_155405

theorem container_capacity (container_volume : ℝ) (num_containers : ℕ) : 
  (8 : ℝ) = 0.2 * container_volume → 
  num_containers = 40 → 
  num_containers * container_volume = 1600 := by
sorry

end NUMINAMATH_CALUDE_container_capacity_l1554_155405


namespace NUMINAMATH_CALUDE_lulu_cupcakes_count_l1554_155446

/-- Represents the number of pastries baked by Lola and Lulu -/
structure Pastries where
  lola_cupcakes : ℕ
  lola_poptarts : ℕ
  lola_pies : ℕ
  lulu_cupcakes : ℕ
  lulu_poptarts : ℕ
  lulu_pies : ℕ

/-- The total number of pastries baked by Lola and Lulu -/
def total_pastries (p : Pastries) : ℕ :=
  p.lola_cupcakes + p.lola_poptarts + p.lola_pies +
  p.lulu_cupcakes + p.lulu_poptarts + p.lulu_pies

/-- Theorem stating that Lulu baked 16 mini cupcakes -/
theorem lulu_cupcakes_count (p : Pastries) 
  (h1 : p.lola_cupcakes = 13)
  (h2 : p.lola_poptarts = 10)
  (h3 : p.lola_pies = 8)
  (h4 : p.lulu_poptarts = 12)
  (h5 : p.lulu_pies = 14)
  (h6 : total_pastries p = 73) :
  p.lulu_cupcakes = 16 := by
  sorry

end NUMINAMATH_CALUDE_lulu_cupcakes_count_l1554_155446


namespace NUMINAMATH_CALUDE_complex_number_in_third_quadrant_l1554_155494

theorem complex_number_in_third_quadrant : 
  let z : ℂ := Complex.I * (-1 + Complex.I)
  (z.re < 0) ∧ (z.im < 0) := by
sorry

end NUMINAMATH_CALUDE_complex_number_in_third_quadrant_l1554_155494


namespace NUMINAMATH_CALUDE_five_segments_create_fifteen_sections_l1554_155496

/-- The maximum number of sections created by n line segments in a rectangle,
    where each new line intersects all previously drawn lines inside the rectangle. -/
def max_sections (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | 1 => 2
  | k + 2 => max_sections (k + 1) + k + 1

/-- The theorem stating that 5 line segments create a maximum of 15 sections. -/
theorem five_segments_create_fifteen_sections :
  max_sections 5 = 15 := by
  sorry

end NUMINAMATH_CALUDE_five_segments_create_fifteen_sections_l1554_155496


namespace NUMINAMATH_CALUDE_cos_power_sum_l1554_155484

theorem cos_power_sum (α : ℝ) (x : ℝ) (n : ℕ) (h : x ≠ 0) :
  x + 1/x = 2 * Real.cos α → x^n + 1/x^n = 2 * Real.cos (n * α) := by
  sorry

end NUMINAMATH_CALUDE_cos_power_sum_l1554_155484


namespace NUMINAMATH_CALUDE_store_goods_values_l1554_155460

/-- Given a store with two grades of goods, prove the initial values of the goods. -/
theorem store_goods_values (x y : ℝ) (a b : ℝ) (h1 : x + y = 450)
  (h2 : y / b * (a + b) = 400) (h3 : x / a * (a + b) = 480) :
  x = 300 ∧ y = 150 := by
  sorry


end NUMINAMATH_CALUDE_store_goods_values_l1554_155460


namespace NUMINAMATH_CALUDE_f_increasing_iff_m_range_l1554_155499

def f (x m : ℝ) : ℝ := |x^2 + (m-1)*x + (m^2 - 3*m + 1)|

theorem f_increasing_iff_m_range :
  ∀ m : ℝ, (∀ x₁ x₂ : ℝ, -1 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ 0 → f x₁ m < f x₂ m) ↔ (m = 1 ∨ m ≥ 3) :=
by sorry

end NUMINAMATH_CALUDE_f_increasing_iff_m_range_l1554_155499


namespace NUMINAMATH_CALUDE_consecutive_integers_sum_l1554_155437

theorem consecutive_integers_sum (x y z : ℤ) (w : ℤ) : 
  y = x + 1 → 
  z = x + 2 → 
  x + y + z = 150 → 
  w = 2*z - x → 
  x + y + z + w = 203 := by sorry

end NUMINAMATH_CALUDE_consecutive_integers_sum_l1554_155437


namespace NUMINAMATH_CALUDE_triangle_angle_calculation_l1554_155444

theorem triangle_angle_calculation (a b c : ℝ) (A B C : ℝ) :
  a = 4 * Real.sqrt 3 →
  c = 12 →
  C = π / 3 →
  0 < a ∧ 0 < b ∧ 0 < c →
  A + B + C = π →
  a / Real.sin A = b / Real.sin B →
  b / Real.sin B = c / Real.sin C →
  A = π / 6 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_calculation_l1554_155444


namespace NUMINAMATH_CALUDE_least_coins_coins_exist_l1554_155417

theorem least_coins (n : ℕ) : n > 0 ∧ n % 7 = 3 ∧ n % 4 = 2 → n ≥ 24 := by
  sorry

theorem coins_exist : ∃ n : ℕ, n > 0 ∧ n % 7 = 3 ∧ n % 4 = 2 ∧ n = 24 := by
  sorry

end NUMINAMATH_CALUDE_least_coins_coins_exist_l1554_155417


namespace NUMINAMATH_CALUDE_pipe_A_fill_time_l1554_155492

-- Define the flow rates of pipes A, B, and C
def flow_rate_A : ℝ := by sorry
def flow_rate_B : ℝ := 2 * flow_rate_A
def flow_rate_C : ℝ := 2 * flow_rate_B

-- Define the time it takes for all three pipes to fill the tank
def total_fill_time : ℝ := 4

-- Theorem stating that pipe A alone takes 28 hours to fill the tank
theorem pipe_A_fill_time :
  1 / flow_rate_A = 28 :=
by
  sorry

end NUMINAMATH_CALUDE_pipe_A_fill_time_l1554_155492


namespace NUMINAMATH_CALUDE_product_of_differences_l1554_155448

theorem product_of_differences (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) 
  (h₁ : x₁^3 - 3*x₁*y₁^2 = 2010) (h₂ : y₁^3 - 3*x₁^2*y₁ = 2009)
  (h₃ : x₂^3 - 3*x₂*y₂^2 = 2010) (h₄ : y₂^3 - 3*x₂^2*y₂ = 2009)
  (h₅ : x₃^3 - 3*x₃*y₃^2 = 2010) (h₆ : y₃^3 - 3*x₃^2*y₃ = 2009) :
  (1 - x₁/y₁) * (1 - x₂/y₂) * (1 - x₃/y₃) = 1/2010 := by
  sorry

end NUMINAMATH_CALUDE_product_of_differences_l1554_155448


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_l1554_155412

/-- A complex number is purely imaginary if its real part is zero -/
def PurelyImaginary (z : ℂ) : Prop := z.re = 0

/-- The imaginary unit -/
def i : ℂ := Complex.I

/-- The condition that (z+2)/(1-i) + z is a real number -/
def IsRealCondition (z : ℂ) : Prop := ((z + 2) / (1 - i) + z).im = 0

theorem purely_imaginary_complex : 
  ∀ z : ℂ, PurelyImaginary z → IsRealCondition z → z = -2/3 * i :=
sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_l1554_155412


namespace NUMINAMATH_CALUDE_arithmetic_seq_problem_l1554_155447

/-- Define an arithmetic sequence {aₙ/n} with common difference d -/
def arithmetic_seq (a : ℕ → ℚ) (d : ℚ) :=
  ∀ n m : ℕ, a m / m - a n / n = d * (m - n)

theorem arithmetic_seq_problem (a : ℕ → ℚ) (d : ℚ) 
  (h_seq : arithmetic_seq a d)
  (h_a3 : a 3 = 2)
  (h_a9 : a 9 = 12) :
  d = 1/9 ∧ a 12 = 20 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_seq_problem_l1554_155447


namespace NUMINAMATH_CALUDE_power_division_multiplication_l1554_155465

theorem power_division_multiplication (x : ℕ) : (3^18 / 27^2) * 7 = 3720087 := by
  sorry

end NUMINAMATH_CALUDE_power_division_multiplication_l1554_155465


namespace NUMINAMATH_CALUDE_student_count_l1554_155421

theorem student_count (total_erasers total_pencils leftover_erasers leftover_pencils : ℕ)
  (h1 : total_erasers = 49)
  (h2 : total_pencils = 66)
  (h3 : leftover_erasers = 4)
  (h4 : leftover_pencils = 6) :
  ∃ (students : ℕ),
    students > 0 ∧
    (total_erasers - leftover_erasers) % students = 0 ∧
    (total_pencils - leftover_pencils) % students = 0 ∧
    students = 15 := by
  sorry

end NUMINAMATH_CALUDE_student_count_l1554_155421


namespace NUMINAMATH_CALUDE_ratio_constraint_l1554_155450

theorem ratio_constraint (a b : ℝ) (h1 : 0 ≤ a) (h2 : a < b) 
  (h3 : ∀ x : ℝ, a + b * Real.cos x + (b / (2 * Real.sqrt 2)) * Real.cos (2 * x) ≥ 0) :
  (b + a) / (b - a) = 3 + 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ratio_constraint_l1554_155450


namespace NUMINAMATH_CALUDE_kendra_toy_purchase_l1554_155451

/-- The price of a wooden toy -/
def toy_price : ℕ := 20

/-- The price of a hat -/
def hat_price : ℕ := 10

/-- The number of hats Kendra bought -/
def hats_bought : ℕ := 3

/-- The amount of money Kendra started with -/
def initial_money : ℕ := 100

/-- The amount of change Kendra received -/
def change_received : ℕ := 30

/-- The number of wooden toys Kendra bought -/
def toys_bought : ℕ := 2

theorem kendra_toy_purchase :
  toy_price * toys_bought + hat_price * hats_bought = initial_money - change_received :=
by sorry

end NUMINAMATH_CALUDE_kendra_toy_purchase_l1554_155451


namespace NUMINAMATH_CALUDE_remaining_cube_volume_l1554_155429

/-- Calculates the remaining volume of a cube after removing a cylindrical section. -/
theorem remaining_cube_volume (cube_side : ℝ) (cylinder_radius : ℝ) (cylinder_height : ℝ) :
  cube_side = 6 →
  cylinder_radius = 3 →
  cylinder_height = 6 →
  cube_side^3 - π * cylinder_radius^2 * cylinder_height = 216 - 54 * π :=
by
  sorry

#check remaining_cube_volume

end NUMINAMATH_CALUDE_remaining_cube_volume_l1554_155429


namespace NUMINAMATH_CALUDE_cos_leq_half_range_l1554_155428

theorem cos_leq_half_range (x : Real) :
  x ∈ Set.Icc 0 (2 * Real.pi) →
  (Real.cos x ≤ 1/2 ↔ x ∈ Set.Icc (Real.pi/3) (5*Real.pi/3)) :=
by sorry

end NUMINAMATH_CALUDE_cos_leq_half_range_l1554_155428


namespace NUMINAMATH_CALUDE_point_groups_theorem_l1554_155435

theorem point_groups_theorem (n₁ n₂ : ℕ) : 
  n₁ + n₂ = 28 → 
  (n₁ * (n₁ - 1)) / 2 - (n₂ * (n₂ - 1)) / 2 = 81 → 
  (n₁ = 17 ∧ n₂ = 11) ∨ (n₁ = 11 ∧ n₂ = 17) := by
  sorry

end NUMINAMATH_CALUDE_point_groups_theorem_l1554_155435


namespace NUMINAMATH_CALUDE_dice_cube_properties_l1554_155474

/-- Represents a cube formed from 27 dice in a 3x3x3 configuration -/
structure DiceCube where
  size : Nat
  visible_dice : Nat
  faces_per_die : Nat

/-- Calculates the probability of exactly 25 sixes on the surface of the cube -/
def prob_25_sixes (cube : DiceCube) : ℚ :=
  31 / (2^13 * 3^18)

/-- Calculates the probability of at least one "one" on the surface of the cube -/
def prob_at_least_one_one (cube : DiceCube) : ℚ :=
  1 - (5^6 / (2^2 * 3^18))

/-- Calculates the expected number of sixes showing on the surface of the cube -/
def expected_sixes (cube : DiceCube) : ℚ :=
  9

/-- Calculates the expected sum of the numbers on the surface of the cube -/
def expected_sum (cube : DiceCube) : ℚ :=
  6 - (5^6 / (2 * 3^17))

/-- Main theorem stating the properties of the dice cube -/
theorem dice_cube_properties (cube : DiceCube) 
    (h1 : cube.size = 27) 
    (h2 : cube.visible_dice = 26) 
    (h3 : cube.faces_per_die = 6) : 
  (prob_25_sixes cube = 31 / (2^13 * 3^18)) ∧ 
  (prob_at_least_one_one cube = 1 - 5^6 / (2^2 * 3^18)) ∧ 
  (expected_sixes cube = 9) ∧ 
  (expected_sum cube = 6 - 5^6 / (2 * 3^17)) := by
  sorry

#check dice_cube_properties

end NUMINAMATH_CALUDE_dice_cube_properties_l1554_155474


namespace NUMINAMATH_CALUDE_num_machines_is_five_l1554_155478

/-- The number of machines in the first scenario -/
def num_machines : ℕ := 5

/-- The production rate of the machines in the first scenario -/
def production_rate_1 : ℚ := 20 / (10 * num_machines)

/-- The production rate of the machines in the second scenario -/
def production_rate_2 : ℚ := 200 / (25 * 20)

/-- Theorem stating that the number of machines in the first scenario is 5 -/
theorem num_machines_is_five :
  num_machines = 5 ∧ production_rate_1 = production_rate_2 :=
sorry

end NUMINAMATH_CALUDE_num_machines_is_five_l1554_155478


namespace NUMINAMATH_CALUDE_smallest_z_value_l1554_155477

theorem smallest_z_value (w x y z : ℕ) : 
  w^3 + x^3 + y^3 = z^3 →
  w < x ∧ x < y ∧ y < z →
  Odd w ∧ Odd x ∧ Odd y ∧ Odd z →
  (∀ a b c d : ℕ, a < b ∧ b < c ∧ c < d ∧ 
    Odd a ∧ Odd b ∧ Odd c ∧ Odd d ∧
    a^3 + b^3 + c^3 = d^3 → z ≤ d) →
  z = 9 :=
by sorry

end NUMINAMATH_CALUDE_smallest_z_value_l1554_155477


namespace NUMINAMATH_CALUDE_prime_cube_plus_two_l1554_155458

theorem prime_cube_plus_two (m : ℕ) : 
  Prime m → Prime (m^2 + 2) → m = 3 ∧ Prime (m^3 + 2) :=
by sorry

end NUMINAMATH_CALUDE_prime_cube_plus_two_l1554_155458


namespace NUMINAMATH_CALUDE_common_chord_equation_l1554_155443

-- Define the circles
def C1 (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 8*y - 8 = 0
def C2 (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 4*y - 2 = 0

-- Define the common chord line
def common_chord (x y : ℝ) : Prop := x + 2*y - 1 = 0

-- Theorem statement
theorem common_chord_equation :
  ∀ x y : ℝ, C1 x y ∧ C2 x y → common_chord x y :=
by sorry

end NUMINAMATH_CALUDE_common_chord_equation_l1554_155443


namespace NUMINAMATH_CALUDE_jogger_multiple_l1554_155406

/-- The number of joggers bought by each person -/
structure JoggerPurchase where
  tyson : ℕ
  alexander : ℕ
  christopher : ℕ

/-- The conditions of the jogger purchase problem -/
def JoggerProblem (jp : JoggerPurchase) : Prop :=
  jp.alexander = jp.tyson + 22 ∧
  jp.christopher = 80 ∧
  jp.christopher = jp.alexander + 54 ∧
  ∃ m : ℕ, jp.christopher = m * jp.tyson

theorem jogger_multiple (jp : JoggerPurchase) (h : JoggerProblem jp) :
  ∃ m : ℕ, jp.christopher = m * jp.tyson ∧ m = 20 := by
  sorry

#check jogger_multiple

end NUMINAMATH_CALUDE_jogger_multiple_l1554_155406


namespace NUMINAMATH_CALUDE_sum_of_factors_l1554_155425

theorem sum_of_factors (a b c d e : ℤ) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ 
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ 
  c ≠ d ∧ c ≠ e ∧ 
  d ≠ e →
  (7 - a) * (7 - b) * (7 - c) * (7 - d) * (7 - e) = -120 →
  a + b + c + d + e = 25 := by
sorry

end NUMINAMATH_CALUDE_sum_of_factors_l1554_155425


namespace NUMINAMATH_CALUDE_complex_fraction_real_l1554_155408

theorem complex_fraction_real (a : ℝ) : 
  (((a : ℂ) + Complex.I) / (1 + Complex.I)).im = 0 → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_real_l1554_155408


namespace NUMINAMATH_CALUDE_dividend_calculation_l1554_155411

theorem dividend_calculation (divisor quotient remainder : ℕ) : 
  divisor = 36 → quotient = 19 → remainder = 6 → 
  divisor * quotient + remainder = 690 := by
sorry

end NUMINAMATH_CALUDE_dividend_calculation_l1554_155411


namespace NUMINAMATH_CALUDE_fixed_point_of_f_l1554_155438

-- Define the set of valid 'a' values
def ValidA : Set ℝ := { x | (0 < x ∧ x < 1) ∨ (1 < x) }

-- Define the logarithm function
noncomputable def log (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

-- Define our function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := log a x + 2

-- State the theorem
theorem fixed_point_of_f (a : ℝ) (h : a ∈ ValidA) : f a 1 = 2 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_of_f_l1554_155438


namespace NUMINAMATH_CALUDE_seven_eighths_of_48_l1554_155439

theorem seven_eighths_of_48 : (7 / 8 : ℚ) * 48 = 42 := by
  sorry

end NUMINAMATH_CALUDE_seven_eighths_of_48_l1554_155439


namespace NUMINAMATH_CALUDE_johnson_family_seating_l1554_155471

/-- The number of ways to arrange 5 boys and 4 girls in a row of 9 chairs such that at least 2 boys are next to each other -/
def seating_arrangements (num_boys : ℕ) (num_girls : ℕ) : ℕ :=
  Nat.factorial (num_boys + num_girls) - 2 * (Nat.factorial num_boys * Nat.factorial num_girls)

/-- Theorem stating that the number of seating arrangements for 5 boys and 4 girls with at least 2 boys next to each other is 357120 -/
theorem johnson_family_seating :
  seating_arrangements 5 4 = 357120 := by
  sorry

end NUMINAMATH_CALUDE_johnson_family_seating_l1554_155471


namespace NUMINAMATH_CALUDE_parabola_ellipse_intersection_l1554_155475

theorem parabola_ellipse_intersection (p : ℝ) (m n k : ℝ) : 
  p > 0 → m > n → n > 0 → 
  ∃ (x₀ y₀ : ℝ), 
    y₀^2 = 2*p*x₀ ∧ 
    (x₀ + p/2)^2 + y₀^2 = 3^2 ∧ 
    x₀^2 + y₀^2 = 9 →
  ∃ (c : ℝ), 
    c = 2 ∧ 
    m^2 - n^2 = c^2 ∧
    2/m = 1/2 →
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    x₁^2/m^2 + y₁^2/n^2 = 1 ∧
    x₂^2/m^2 + y₂^2/n^2 = 1 ∧
    y₁ = k*x₁ - 4 ∧
    y₂ = k*x₂ - 4 ∧
    x₁ ≠ x₂ ∧
    x₁*x₂ + y₁*y₂ > 0 →
  (-2*Real.sqrt 3/3 < k ∧ k < -1/2) ∨ (1/2 < k ∧ k < 2*Real.sqrt 3/3) :=
by sorry

end NUMINAMATH_CALUDE_parabola_ellipse_intersection_l1554_155475


namespace NUMINAMATH_CALUDE_phoenix_hike_length_l1554_155454

/-- Represents the length of Phoenix's hike on the Rocky Path Trail -/
theorem phoenix_hike_length 
  (day1 day2 day3 day4 : ℝ) 
  (first_two_days : day1 + day2 = 22)
  (second_third_avg : (day2 + day3) / 2 = 13)
  (last_two_days : day3 + day4 = 30)
  (first_third_days : day1 + day3 = 26) :
  day1 + day2 + day3 + day4 = 52 :=
by
  sorry


end NUMINAMATH_CALUDE_phoenix_hike_length_l1554_155454


namespace NUMINAMATH_CALUDE_power_sum_equality_l1554_155426

theorem power_sum_equality (a b c d : ℝ) 
  (sum_eq : a + b = c + d) 
  (cube_sum_eq : a^3 + b^3 = c^3 + d^3) : 
  a^5 + b^5 = c^5 + d^5 ∧ 
  ∃ a b c d : ℝ, (a + b = c + d) ∧ (a^3 + b^3 = c^3 + d^3) ∧ (a^4 + b^4 ≠ c^4 + d^4) :=
by sorry

end NUMINAMATH_CALUDE_power_sum_equality_l1554_155426


namespace NUMINAMATH_CALUDE_simplify_expression_l1554_155400

theorem simplify_expression (x : ℝ) (h1 : 1 < x) (h2 : x < 4) :
  Real.sqrt ((1 - x)^2) + |x - 4| = 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1554_155400


namespace NUMINAMATH_CALUDE_cube_minus_self_div_by_six_l1554_155490

theorem cube_minus_self_div_by_six (n : ℕ) : 6 ∣ (n^3 - n) := by
  sorry

end NUMINAMATH_CALUDE_cube_minus_self_div_by_six_l1554_155490


namespace NUMINAMATH_CALUDE_arithmetic_sequence_squares_l1554_155410

theorem arithmetic_sequence_squares (x : ℚ) :
  (∃ (a d : ℚ), 
    (5 + x)^2 = a - d ∧
    (7 + x)^2 = a ∧
    (10 + x)^2 = a + d ∧
    d ≠ 0) →
  x = -31/8 ∧ (∃ d : ℚ, d^2 = 1/2) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_squares_l1554_155410


namespace NUMINAMATH_CALUDE_simplify_expression_l1554_155459

theorem simplify_expression (x : ℝ) : 
  2*x - 3*(2-x) + (1/2)*(3-2*x) - 5*(2+3*x) = -11*x - 15.5 := by
sorry

end NUMINAMATH_CALUDE_simplify_expression_l1554_155459


namespace NUMINAMATH_CALUDE_total_cost_is_543_l1554_155432

/-- Calculates the total amount John has to pay for earbuds and a smartwatch, including tax and discount. -/
def totalCost (earbudsCost smartwatchCost : ℝ) (earbudsTaxRate smartwatchTaxRate earbusDiscountRate : ℝ) : ℝ :=
  let discountedEarbudsCost := earbudsCost * (1 - earbusDiscountRate)
  let earbudsTax := discountedEarbudsCost * earbudsTaxRate
  let smartwatchTax := smartwatchCost * smartwatchTaxRate
  discountedEarbudsCost + earbudsTax + smartwatchCost + smartwatchTax

/-- Theorem stating that given the specific costs, tax rates, and discount, the total cost is $543. -/
theorem total_cost_is_543 :
  totalCost 200 300 0.15 0.12 0.10 = 543 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_is_543_l1554_155432


namespace NUMINAMATH_CALUDE_min_value_of_function_l1554_155455

theorem min_value_of_function (x : ℝ) (h : x > 0) :
  (4 * x^2 + 8 * x + 13) / (6 * (1 + x)) ≥ 2 ∧
  ∃ y > 0, (4 * y^2 + 8 * y + 13) / (6 * (1 + y)) = 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_function_l1554_155455


namespace NUMINAMATH_CALUDE_parabola_unique_intersection_l1554_155468

/-- A parabola defined by x = -4y^2 - 6y + 10 -/
def parabola (y : ℝ) : ℝ := -4 * y^2 - 6 * y + 10

/-- The condition for a vertical line x = m to intersect the parabola at exactly one point -/
def unique_intersection (m : ℝ) : Prop :=
  ∃! y, parabola y = m

theorem parabola_unique_intersection :
  ∀ m : ℝ, unique_intersection m → m = 49 / 4 := by sorry

end NUMINAMATH_CALUDE_parabola_unique_intersection_l1554_155468


namespace NUMINAMATH_CALUDE_correct_operation_l1554_155488

theorem correct_operation (m : ℝ) : (-m + 2) * (-m - 2) = m^2 - 4 := by
  sorry

end NUMINAMATH_CALUDE_correct_operation_l1554_155488


namespace NUMINAMATH_CALUDE_box_dimensions_sum_l1554_155404

/-- Given a rectangular box with dimensions A, B, and C, prove that if the surface areas of its faces
    are 30, 30, 60, 60, 90, and 90 square units, then A + B + C = 24. -/
theorem box_dimensions_sum (A B C : ℝ) : 
  A > 0 → B > 0 → C > 0 →
  A * B = 30 →
  A * C = 60 →
  B * C = 90 →
  A + B + C = 24 := by
  sorry

end NUMINAMATH_CALUDE_box_dimensions_sum_l1554_155404


namespace NUMINAMATH_CALUDE_quadratic_root_range_l1554_155423

theorem quadratic_root_range (m : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ < -1 ∧ x₂ > 1 ∧ 
   x₁^2 + (m-1)*x₁ + m^2 - 2 = 0 ∧ 
   x₂^2 + (m-1)*x₂ + m^2 - 2 = 0) → 
  0 < m ∧ m < 1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_range_l1554_155423


namespace NUMINAMATH_CALUDE_sin_graph_transformation_l1554_155486

theorem sin_graph_transformation (x : ℝ) :
  let f (x : ℝ) := Real.sin (2 * x)
  let g (x : ℝ) := f (x - π / 3)
  let h (x : ℝ) := g (-x)
  h x = Real.sin (-2 * x - 2 * π / 3) := by
  sorry

end NUMINAMATH_CALUDE_sin_graph_transformation_l1554_155486


namespace NUMINAMATH_CALUDE_infinitely_many_pairs_exist_l1554_155422

/-- Definition of triangular numbers -/
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Theorem stating the existence of infinitely many pairs (a, b) satisfying the property -/
theorem infinitely_many_pairs_exist :
  ∃ f : ℕ → ℕ × ℕ, ∀ k : ℕ,
    let (a, b) := f k
    ∀ n : ℕ, (∃ m : ℕ, a * triangular_number n + b = triangular_number m) ↔
              (∃ l : ℕ, triangular_number n = triangular_number l) :=
by sorry

end NUMINAMATH_CALUDE_infinitely_many_pairs_exist_l1554_155422


namespace NUMINAMATH_CALUDE_pentagonal_gcd_one_l1554_155453

theorem pentagonal_gcd_one (n : ℕ+) : 
  let P : ℕ+ → ℕ := fun m => (m * (3 * m - 1)) / 2
  Nat.gcd (5 * P n) (n + 1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_pentagonal_gcd_one_l1554_155453


namespace NUMINAMATH_CALUDE_smallest_covering_triangular_number_l1554_155414

def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

def all_remainders_covered (m : ℕ) : Prop :=
  ∀ r : Fin 7, ∃ k : ℕ, k ≤ m ∧ triangular_number k % 7 = r.val

theorem smallest_covering_triangular_number :
  (all_remainders_covered 10) ∧
  (∀ n < 10, ¬ all_remainders_covered n) :=
sorry

end NUMINAMATH_CALUDE_smallest_covering_triangular_number_l1554_155414


namespace NUMINAMATH_CALUDE_f_max_min_on_interval_l1554_155427

-- Define the function f(x)
def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 - 12 * x + 5

-- State the theorem
theorem f_max_min_on_interval :
  ∃ (a b : ℝ), a ∈ Set.Icc 0 2 ∧ b ∈ Set.Icc 0 2 ∧
  (∀ x, x ∈ Set.Icc 0 2 → f x ≤ f a) ∧
  (∀ x, x ∈ Set.Icc 0 2 → f x ≥ f b) ∧
  f a = 5 ∧ f b = -15 :=
sorry


end NUMINAMATH_CALUDE_f_max_min_on_interval_l1554_155427


namespace NUMINAMATH_CALUDE_fraction_equals_zero_l1554_155483

theorem fraction_equals_zero (x : ℝ) : (x + 1) / (x - 2) = 0 → x = -1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equals_zero_l1554_155483


namespace NUMINAMATH_CALUDE_bread_in_pond_l1554_155449

theorem bread_in_pond (total_bread : ℕ) (duck1_bread : ℕ) (duck2_bread : ℕ) (duck3_bread : ℕ) 
  (h1 : total_bread = 100)
  (h2 : duck1_bread = total_bread / 2)
  (h3 : duck2_bread = 13)
  (h4 : duck3_bread = 7) :
  total_bread - (duck1_bread + duck2_bread + duck3_bread) = 30 := by
  sorry

end NUMINAMATH_CALUDE_bread_in_pond_l1554_155449


namespace NUMINAMATH_CALUDE_cube_sphere_volume_l1554_155401

theorem cube_sphere_volume (cube_surface_area : ℝ) (h_surface_area : cube_surface_area = 18) :
  let cube_edge := Real.sqrt (cube_surface_area / 6)
  let sphere_radius := (Real.sqrt 3 * cube_edge) / 2
  let sphere_volume := (4 / 3) * Real.pi * sphere_radius ^ 3
  sphere_volume = 9 * Real.pi / 2 := by
  sorry

end NUMINAMATH_CALUDE_cube_sphere_volume_l1554_155401


namespace NUMINAMATH_CALUDE_rectangle_length_l1554_155470

/-- Proves that a rectangle with area 6 m² and width 150 cm has length 400 cm -/
theorem rectangle_length (area : ℝ) (width_cm : ℝ) (length_cm : ℝ) : 
  area = 6 → 
  width_cm = 150 → 
  area = (width_cm / 100) * (length_cm / 100) → 
  length_cm = 400 := by
sorry

end NUMINAMATH_CALUDE_rectangle_length_l1554_155470


namespace NUMINAMATH_CALUDE_ways_to_express_114_l1554_155403

/-- Represents the number of ways to express a given number as the sum of ones and threes with a minimum number of ones -/
def waysToExpress (total : ℕ) (minOnes : ℕ) : ℕ :=
  (total - minOnes) / 3 + 1

/-- The theorem stating that there are 35 ways to express 114 as the sum of ones and threes with at least 10 ones -/
theorem ways_to_express_114 : waysToExpress 114 10 = 35 := by
  sorry

#eval waysToExpress 114 10

end NUMINAMATH_CALUDE_ways_to_express_114_l1554_155403


namespace NUMINAMATH_CALUDE_intersection_point_property_l1554_155420

theorem intersection_point_property (α : ℝ) (h1 : α ≠ 0) (h2 : Real.tan α = -α) :
  (α^2 + 1) * (1 + Real.cos (2 * α)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_property_l1554_155420


namespace NUMINAMATH_CALUDE_midpoint_chain_l1554_155456

/-- Given a line segment XY with midpoints defined as follows:
    G is the midpoint of XY
    H is the midpoint of XG
    I is the midpoint of XH
    J is the midpoint of XI
    If XJ = 4, then XY = 64 -/
theorem midpoint_chain (X Y G H I J : ℝ) : 
  (G = (X + Y) / 2) →
  (H = (X + G) / 2) →
  (I = (X + H) / 2) →
  (J = (X + I) / 2) →
  (J - X = 4) →
  (Y - X = 64) := by
  sorry

end NUMINAMATH_CALUDE_midpoint_chain_l1554_155456


namespace NUMINAMATH_CALUDE_appears_in_31st_equation_l1554_155416

/-- The first term of the nth equation in the sequence -/
def first_term (n : ℕ) : ℕ := 2 * n^2

/-- The proposition that 2016 appears in the 31st equation -/
theorem appears_in_31st_equation : ∃ k : ℕ, k ≥ first_term 31 ∧ k ≤ first_term 32 ∧ k = 2016 :=
sorry

end NUMINAMATH_CALUDE_appears_in_31st_equation_l1554_155416


namespace NUMINAMATH_CALUDE_least_whole_number_subtraction_l1554_155479

-- Define the original ratio
def original_ratio : Rat := 6 / 7

-- Define the comparison ratio
def comparison_ratio : Rat := 16 / 21

-- Define the function that creates the new ratio after subtracting x
def new_ratio (x : ℕ) : Rat := (6 - x) / (7 - x)

-- Statement to prove
theorem least_whole_number_subtraction :
  ∀ x : ℕ, x < 3 → new_ratio x ≥ comparison_ratio ∧
  new_ratio 3 < comparison_ratio :=
by sorry

end NUMINAMATH_CALUDE_least_whole_number_subtraction_l1554_155479


namespace NUMINAMATH_CALUDE_frog_arrangement_count_l1554_155463

def frog_arrangements (n : ℕ) (g r : ℕ) (b : ℕ) : Prop :=
  n = g + r + b ∧
  g = 3 ∧
  r = 3 ∧
  b = 1

theorem frog_arrangement_count :
  ∀ (n g r b : ℕ),
    frog_arrangements n g r b →
    (n - 1) * 2 * (g.factorial * r.factorial) = 504 :=
by sorry

end NUMINAMATH_CALUDE_frog_arrangement_count_l1554_155463


namespace NUMINAMATH_CALUDE_four_men_absent_l1554_155481

/-- Represents the work completion scenario with absent workers -/
structure WorkScenario where
  totalMen : ℕ
  originalDays : ℕ
  actualDays : ℕ
  absentMen : ℕ

/-- Calculates the number of absent men given the work scenario -/
def calculateAbsentMen (scenario : WorkScenario) : ℕ :=
  scenario.totalMen - (scenario.totalMen * scenario.originalDays) / scenario.actualDays

/-- Theorem stating that 4 men became absent in the given scenario -/
theorem four_men_absent :
  let scenario := WorkScenario.mk 8 6 12 4
  calculateAbsentMen scenario = 4 := by
  sorry

#eval calculateAbsentMen (WorkScenario.mk 8 6 12 4)

end NUMINAMATH_CALUDE_four_men_absent_l1554_155481


namespace NUMINAMATH_CALUDE_partners_count_l1554_155480

/-- Represents the number of employees in each category -/
structure FirmComposition where
  partners : ℕ
  associates : ℕ
  managers : ℕ

/-- The initial ratio of partners : associates : managers -/
def initial_ratio : FirmComposition := ⟨2, 63, 20⟩

/-- The new ratio after hiring more employees -/
def new_ratio : FirmComposition := ⟨1, 34, 15⟩

/-- The number of additional associates hired -/
def additional_associates : ℕ := 35

/-- The number of additional managers hired -/
def additional_managers : ℕ := 10

/-- Theorem stating that the number of partners in the firm is 14 -/
theorem partners_count : ∃ (x : ℕ), 
  x * initial_ratio.partners = 14 ∧
  x * initial_ratio.associates + additional_associates = new_ratio.associates * 14 ∧
  x * initial_ratio.managers + additional_managers = new_ratio.managers * 14 :=
sorry

end NUMINAMATH_CALUDE_partners_count_l1554_155480


namespace NUMINAMATH_CALUDE_isosceles_triangle_exists_l1554_155445

/-- Represents an isosceles triangle with base a and leg b -/
structure IsoscelesTriangle where
  a : ℝ  -- base
  b : ℝ  -- leg
  ma : ℝ  -- height corresponding to base
  mb : ℝ  -- height corresponding to leg
  h_isosceles : b > 0 ∧ ma > 0 ∧ mb > 0 ∧ a * ma = b * mb

/-- Given the sums and differences of sides and heights, 
    prove the existence of an isosceles triangle -/
theorem isosceles_triangle_exists 
  (sum_sides : ℝ) 
  (sum_heights : ℝ) 
  (diff_sides : ℝ) 
  (diff_heights : ℝ) 
  (h_positive : sum_sides > 0 ∧ sum_heights > 0 ∧ diff_sides > 0 ∧ diff_heights > 0) :
  ∃ t : IsoscelesTriangle, 
    t.a + t.b = sum_sides ∧ 
    t.ma + t.mb = sum_heights ∧
    t.b - t.a = diff_sides ∧
    t.ma - t.mb = diff_heights :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_exists_l1554_155445


namespace NUMINAMATH_CALUDE_expression_evaluation_l1554_155498

theorem expression_evaluation (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ -1) :
  ((((x - 2)^2 * (x^2 + x + 1)^2) / (x^3 - 1)^2)^2 * 
   (((x + 2)^2 * (x^2 - x + 1)^2) / (x^3 + 1)^2)^2) = (x^2 - 4)^4 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1554_155498


namespace NUMINAMATH_CALUDE_distance_to_felix_l1554_155473

/-- The vertical distance David and Emma walk together to reach Felix -/
theorem distance_to_felix (david_x david_y emma_x emma_y felix_x felix_y : ℝ) 
  (h1 : david_x = 2 ∧ david_y = -25)
  (h2 : emma_x = -3 ∧ emma_y = 19)
  (h3 : felix_x = -1/2 ∧ felix_y = -6) :
  let midpoint_y := (david_y + emma_y) / 2
  |(midpoint_y - felix_y)| = 3 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_felix_l1554_155473


namespace NUMINAMATH_CALUDE_sum_of_equal_expressions_l1554_155466

theorem sum_of_equal_expressions 
  (a b c d e f g h i : ℤ) 
  (eq1 : a + b + c + d = d + e + f + g) 
  (eq2 : d + e + f + g = g + h + i) 
  (ha : a = 4) 
  (hg : g = 13) 
  (hh : h = 6) : 
  ∃ S : ℤ, (a + b + c + d = S) ∧ (d + e + f + g = S) ∧ (g + h + i = S) ∧ (S = 19 + i) :=
sorry

end NUMINAMATH_CALUDE_sum_of_equal_expressions_l1554_155466


namespace NUMINAMATH_CALUDE_lcm_gcd_relation_l1554_155431

theorem lcm_gcd_relation (a b : ℕ) : 
  (Nat.lcm a b + Nat.gcd a b = a * b / 5) ↔ 
  ((a = 10 ∧ b = 10) ∨ (a = 6 ∧ b = 30) ∨ (a = 30 ∧ b = 6)) :=
sorry

end NUMINAMATH_CALUDE_lcm_gcd_relation_l1554_155431


namespace NUMINAMATH_CALUDE_fraction_nonnegative_l1554_155442

theorem fraction_nonnegative (x : ℝ) : 
  (x^4 - 4*x^3 + 4*x^2) / (1 - x^3) ≥ 0 ↔ x ∈ Set.Ici 0 :=
by sorry

end NUMINAMATH_CALUDE_fraction_nonnegative_l1554_155442


namespace NUMINAMATH_CALUDE_no_natural_solution_l1554_155461

theorem no_natural_solution :
  ¬∃ (a b c : ℕ), (a^b - b^c) * (b^c - c^a) * (c^a - a^b) = 11713 := by
  sorry

end NUMINAMATH_CALUDE_no_natural_solution_l1554_155461


namespace NUMINAMATH_CALUDE_neg_cube_eq_cube_of_neg_l1554_155419

theorem neg_cube_eq_cube_of_neg (x : ℚ) : -x^3 = (-x)^3 := by
  sorry

end NUMINAMATH_CALUDE_neg_cube_eq_cube_of_neg_l1554_155419


namespace NUMINAMATH_CALUDE_inequality_pattern_l1554_155482

theorem inequality_pattern (x a : ℝ) : 
  x > 0 →
  x + 1/x ≥ 2 →
  x + 4/x^2 ≥ 3 →
  x + 27/x^3 ≥ 4 →
  x + a/x^4 ≥ 5 →
  a = 4^4 := by
sorry

end NUMINAMATH_CALUDE_inequality_pattern_l1554_155482


namespace NUMINAMATH_CALUDE_vendor_throw_away_percent_l1554_155424

-- Define the initial number of apples (100 for simplicity)
def initial_apples : ℝ := 100

-- Define the percentage of apples sold on the first day
def first_day_sale_percent : ℝ := 30

-- Define the percentage of apples sold on the second day
def second_day_sale_percent : ℝ := 50

-- Define the total percentage of apples thrown away
def total_thrown_away_percent : ℝ := 42

-- Define the percentage of remaining apples thrown away on the first day
def first_day_throw_away_percent : ℝ := 20

theorem vendor_throw_away_percent :
  let remaining_after_first_sale := initial_apples * (1 - first_day_sale_percent / 100)
  let remaining_after_first_throw := remaining_after_first_sale * (1 - first_day_throw_away_percent / 100)
  let sold_second_day := remaining_after_first_throw * (second_day_sale_percent / 100)
  let thrown_away_second_day := remaining_after_first_throw - sold_second_day
  let total_thrown_away := (remaining_after_first_sale - remaining_after_first_throw) + thrown_away_second_day
  total_thrown_away = initial_apples * (total_thrown_away_percent / 100) :=
by sorry

end NUMINAMATH_CALUDE_vendor_throw_away_percent_l1554_155424


namespace NUMINAMATH_CALUDE_prime_condition_equivalence_l1554_155433

/-- For a prime number p, this function returns true if for each integer a 
    such that 1 < a < p/2, there exists an integer b such that p/2 < b < p 
    and p divides ab - 1 -/
def satisfies_condition (p : ℕ) : Prop :=
  ∀ a : ℕ, 1 < a → a < p / 2 → ∃ b : ℕ, p / 2 < b ∧ b < p ∧ p ∣ (a * b - 1)

theorem prime_condition_equivalence (p : ℕ) (hp : Nat.Prime p) : 
  satisfies_condition p ↔ p ∈ ({5, 7, 13} : Set ℕ) := by
  sorry

end NUMINAMATH_CALUDE_prime_condition_equivalence_l1554_155433


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l1554_155491

theorem complex_fraction_equality : (1 - 2*I) / (1 + I) = (-1 - 3*I) / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l1554_155491


namespace NUMINAMATH_CALUDE_initial_rope_length_l1554_155487

theorem initial_rope_length 
  (r_initial : ℝ) 
  (h1 : r_initial > 0) 
  (h2 : π * (21^2 - r_initial^2) = 933.4285714285714) : 
  r_initial = 12 := by
sorry

end NUMINAMATH_CALUDE_initial_rope_length_l1554_155487


namespace NUMINAMATH_CALUDE_operation_result_l1554_155495

def operation (a b : ℝ) : ℝ := a * (b ^ (1/2))

theorem operation_result :
  ∀ x : ℝ, operation x 9 = 12 → x = 4 := by
sorry

end NUMINAMATH_CALUDE_operation_result_l1554_155495


namespace NUMINAMATH_CALUDE_pond_length_proof_l1554_155472

def field_length : ℝ := 80

theorem pond_length_proof (field_width : ℝ) (pond_side : ℝ) : 
  field_length = 2 * field_width →
  pond_side^2 = (field_length * field_width) / 50 →
  pond_side = 8 := by
sorry

end NUMINAMATH_CALUDE_pond_length_proof_l1554_155472


namespace NUMINAMATH_CALUDE_quadratic_inequality_no_solution_l1554_155452

theorem quadratic_inequality_no_solution :
  ¬∃ x : ℝ, x^2 - 2*x + 3 < 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_no_solution_l1554_155452


namespace NUMINAMATH_CALUDE_hyperbola_focus_m_value_l1554_155462

/-- Given a hyperbola with equation 3mx^2 - my^2 = 3 and one focus at (0, 2), prove that m = -1 -/
theorem hyperbola_focus_m_value (m : ℝ) : 
  (∃ (x y : ℝ), 3 * m * x^2 - m * y^2 = 3) →  -- Hyperbola equation
  (∃ (a b : ℝ), a^2 / (3/m) + b^2 / (1/m) = 1) →  -- Standard form of hyperbola
  (2 : ℝ)^2 = (3/m) + (1/m) →  -- Focus property
  m = -1 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_focus_m_value_l1554_155462


namespace NUMINAMATH_CALUDE_greatest_x_value_l1554_155415

theorem greatest_x_value (x : ℝ) : 
  x ≠ 2 → 
  (x^2 - 5*x - 14) / (x - 2) = 4 / (x + 4) → 
  x ≤ -2 :=
by sorry

end NUMINAMATH_CALUDE_greatest_x_value_l1554_155415


namespace NUMINAMATH_CALUDE_binomial_expansion_example_l1554_155413

theorem binomial_expansion_example : 50^4 + 4*(50^3) + 6*(50^2) + 4*50 + 1 = 6765201 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_example_l1554_155413


namespace NUMINAMATH_CALUDE_circle_radius_l1554_155489

theorem circle_radius (x y : Real) (h : x + y = 90 * Real.pi) :
  ∃ (r : Real), r > 0 ∧ x = Real.pi * r^2 ∧ y = 2 * Real.pi * r ∧ r = 9 := by
sorry

end NUMINAMATH_CALUDE_circle_radius_l1554_155489


namespace NUMINAMATH_CALUDE_system_solution_l1554_155457

theorem system_solution (x y z : ℝ) 
  (eq1 : x * y = 4 - x - 2 * y)
  (eq2 : y * z = 8 - 3 * y - 2 * z)
  (eq3 : x * z = 40 - 5 * x - 2 * z)
  (y_pos : y > 0) : y = 2 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l1554_155457


namespace NUMINAMATH_CALUDE_quadratic_completing_square_l1554_155485

/-- Given a quadratic equation 16x^2 - 32x - 512 = 0, when transformed
    to the form (x + p)^2 = q, the value of q is 33. -/
theorem quadratic_completing_square :
  ∃ (p : ℝ), ∀ (x : ℝ),
    16 * x^2 - 32 * x - 512 = 0 ↔ (x + p)^2 = 33 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_completing_square_l1554_155485


namespace NUMINAMATH_CALUDE_brick_width_calculation_l1554_155434

/-- Proves that given a courtyard of 25 meters by 15 meters, to be paved with 18750 bricks of length 20 cm, the width of each brick must be 10 cm. -/
theorem brick_width_calculation (courtyard_length : ℝ) (courtyard_width : ℝ) 
  (brick_length : ℝ) (total_bricks : ℕ) :
  courtyard_length = 25 →
  courtyard_width = 15 →
  brick_length = 0.2 →
  total_bricks = 18750 →
  ∃ (brick_width : ℝ), 
    brick_width = 0.1 ∧ 
    (courtyard_length * 100) * (courtyard_width * 100) = 
      total_bricks * brick_length * 100 * brick_width * 100 :=
by sorry

end NUMINAMATH_CALUDE_brick_width_calculation_l1554_155434


namespace NUMINAMATH_CALUDE_greatest_n_less_than_200_l1554_155476

theorem greatest_n_less_than_200 :
  ∃ (n : ℕ), n < 200 ∧ 
  (∃ (k : ℕ), n = 9 * k - 2) ∧
  (∃ (l : ℕ), n = 6 * l - 4) ∧
  (∀ (m : ℕ), m < 200 ∧ 
    (∃ (p : ℕ), m = 9 * p - 2) ∧ 
    (∃ (q : ℕ), m = 6 * q - 4) → 
    m ≤ n) ∧
  n = 194 := by
sorry

end NUMINAMATH_CALUDE_greatest_n_less_than_200_l1554_155476


namespace NUMINAMATH_CALUDE_parallelogram_D_coordinates_l1554_155464

structure Point where
  x : ℝ
  y : ℝ

def Parallelogram (A B C D : Point) : Prop :=
  (B.x - A.x, B.y - A.y) = (D.x - C.x, D.y - C.y) ∧
  (C.x - B.x, C.y - B.y) = (A.x - D.x, A.y - D.y)

theorem parallelogram_D_coordinates :
  let A : Point := ⟨-1, 2⟩
  let B : Point := ⟨0, 0⟩
  let C : Point := ⟨1, 7⟩
  let D : Point := ⟨0, 9⟩
  Parallelogram A B C D → D = ⟨0, 9⟩ := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_D_coordinates_l1554_155464


namespace NUMINAMATH_CALUDE_initial_solution_volume_l1554_155497

/-- Proves that the initial amount of solution is 6 litres, given that it is 25% alcohol
    and becomes 50% alcohol when 3 litres of pure alcohol are added. -/
theorem initial_solution_volume (x : ℝ) :
  (0.25 * x) / x = 0.25 →
  ((0.25 * x + 3) / (x + 3) = 0.5) →
  x = 6 := by
sorry

end NUMINAMATH_CALUDE_initial_solution_volume_l1554_155497


namespace NUMINAMATH_CALUDE_parabola_comparison_l1554_155407

theorem parabola_comparison :
  ∀ x : ℝ, x^2 - 3/4*x + 3 ≥ x^2 + 1/4*x + 1 := by
  sorry

end NUMINAMATH_CALUDE_parabola_comparison_l1554_155407


namespace NUMINAMATH_CALUDE_math_problems_l1554_155441

theorem math_problems :
  (32 * 3 = 96) ∧
  (43 / 9 = 4 ∧ 43 % 9 = 7) ∧
  (630 / 9 = 70) ∧
  (125 * 47 * 8 = 125 * 8 * 47) := by
  sorry

end NUMINAMATH_CALUDE_math_problems_l1554_155441


namespace NUMINAMATH_CALUDE_perpendicular_transitivity_l1554_155402

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between lines and planes
variable (perp : Line → Plane → Prop)

-- State the theorem
theorem perpendicular_transitivity
  (m n : Line) (α β : Plane)
  (hm : m ≠ n) (hαβ : α ≠ β)
  (hmβ : perp m β) (hnβ : perp n β) (hnα : perp n α) :
  perp m α :=
sorry

end NUMINAMATH_CALUDE_perpendicular_transitivity_l1554_155402


namespace NUMINAMATH_CALUDE_die_game_first_player_win_probability_l1554_155436

def game_win_probability : ℚ := 5/11

theorem die_game_first_player_win_probability :
  let n := 6  -- number of sides on the die
  let m := 7  -- winning condition (multiple of m)
  ∀ (k : ℕ), k < m →
    let p : ℚ := game_win_probability  -- probability of winning starting from state k
    (p = n / (2*n - 1) ∧
     p = (n-1) * (1 - p) / n + 1/n) :=
by sorry

end NUMINAMATH_CALUDE_die_game_first_player_win_probability_l1554_155436


namespace NUMINAMATH_CALUDE_tangent_slope_angle_at_zero_l1554_155469

noncomputable def f (x : ℝ) : ℝ := x * Real.cos x

theorem tangent_slope_angle_at_zero :
  let slope := (deriv f) 0
  Real.arctan slope = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_tangent_slope_angle_at_zero_l1554_155469


namespace NUMINAMATH_CALUDE_smallest_A_with_triple_factors_l1554_155440

def number_of_factors (n : ℕ) : ℕ := (Nat.divisors n).card

theorem smallest_A_with_triple_factors : 
  ∃ (A : ℕ), A > 0 ∧ 
  (∀ (k : ℕ), k > 0 ∧ k < A → number_of_factors (6 * k) ≠ 3 * number_of_factors k) ∧
  number_of_factors (6 * A) = 3 * number_of_factors A ∧
  A = 2 := by
  sorry

end NUMINAMATH_CALUDE_smallest_A_with_triple_factors_l1554_155440


namespace NUMINAMATH_CALUDE_fishing_competition_duration_l1554_155430

theorem fishing_competition_duration 
  (jackson_daily : ℕ) 
  (jonah_daily : ℕ) 
  (george_daily : ℕ) 
  (total_catch : ℕ) 
  (h1 : jackson_daily = 6)
  (h2 : jonah_daily = 4)
  (h3 : george_daily = 8)
  (h4 : total_catch = 90) :
  ∃ (days : ℕ), days * (jackson_daily + jonah_daily + george_daily) = total_catch ∧ days = 5 := by
  sorry

end NUMINAMATH_CALUDE_fishing_competition_duration_l1554_155430


namespace NUMINAMATH_CALUDE_smallest_marble_count_l1554_155409

/-- Represents the number of marbles of each color in the urn -/
structure MarbleCount where
  red : ℕ
  white : ℕ
  blue : ℕ

/-- Calculates the probability of drawing a specific combination of marbles -/
def probability (m : MarbleCount) (r w b : ℕ) : ℚ :=
  (m.red.choose r) * (m.white.choose w) * (m.blue.choose b) /
  ((m.red + m.white + m.blue).choose 4)

/-- Checks if the three specified events are equally likely -/
def events_equally_likely (m : MarbleCount) : Prop :=
  probability m 3 1 0 = probability m 2 1 1 ∧
  probability m 3 1 0 = probability m 2 1 1

/-- The theorem stating that 8 is the smallest number of marbles satisfying the conditions -/
theorem smallest_marble_count :
  ∃ (m : MarbleCount),
    m.red + m.white + m.blue = 8 ∧
    events_equally_likely m ∧
    ∀ (n : MarbleCount),
      n.red + n.white + n.blue < 8 →
      ¬(events_equally_likely n) :=
sorry

end NUMINAMATH_CALUDE_smallest_marble_count_l1554_155409
