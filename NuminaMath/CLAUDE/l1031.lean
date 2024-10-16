import Mathlib

namespace NUMINAMATH_CALUDE_complex_modulus_proof_l1031_103156

theorem complex_modulus_proof (z z₁ z₂ : ℂ) 
  (h₁ : z₁ ≠ z₂)
  (h₂ : z₁^2 = -2 - 2 * Complex.I * Real.sqrt 3)
  (h₃ : z₂^2 = -2 - 2 * Complex.I * Real.sqrt 3)
  (h₄ : Complex.abs (z - z₁) = 4)
  (h₅ : Complex.abs (z - z₂) = 4) :
  Complex.abs z = 2 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_complex_modulus_proof_l1031_103156


namespace NUMINAMATH_CALUDE_final_alcohol_percentage_l1031_103101

/-- Given a mixture of 15 litres with 25% alcohol, prove that after removing 2 litres of alcohol
    and adding 3 litres of water, the final alcohol percentage is approximately 10.94%. -/
theorem final_alcohol_percentage
  (initial_volume : ℝ)
  (initial_alcohol_percentage : ℝ)
  (alcohol_removed : ℝ)
  (water_added : ℝ)
  (h1 : initial_volume = 15)
  (h2 : initial_alcohol_percentage = 0.25)
  (h3 : alcohol_removed = 2)
  (h4 : water_added = 3) :
  let initial_alcohol := initial_volume * initial_alcohol_percentage
  let remaining_alcohol := initial_alcohol - alcohol_removed
  let final_volume := initial_volume - alcohol_removed + water_added
  let final_percentage := (remaining_alcohol / final_volume) * 100
  ∃ ε > 0, abs (final_percentage - 10.94) < ε :=
sorry

end NUMINAMATH_CALUDE_final_alcohol_percentage_l1031_103101


namespace NUMINAMATH_CALUDE_quadratic_range_condition_l1031_103168

/-- A quadratic function f(x) = mx^2 - 2x + m has a value range of [0, +∞) if and only if m = 1 -/
theorem quadratic_range_condition (m : ℝ) : 
  (∀ x : ℝ, ∃ y : ℝ, y ≥ 0 ∧ y = m * x^2 - 2 * x + m) ∧ 
  (∀ y : ℝ, y ≥ 0 → ∃ x : ℝ, y = m * x^2 - 2 * x + m) ↔ 
  m = 1 :=
sorry

end NUMINAMATH_CALUDE_quadratic_range_condition_l1031_103168


namespace NUMINAMATH_CALUDE_spring_outing_speeds_l1031_103173

theorem spring_outing_speeds (distance : ℝ) (bus_head_start : ℝ) (car_earlier_arrival : ℝ) :
  distance = 90 →
  bus_head_start = 0.5 →
  car_earlier_arrival = 0.25 →
  ∃ (bus_speed car_speed : ℝ),
    car_speed = 1.5 * bus_speed ∧
    distance / bus_speed - distance / car_speed = bus_head_start + car_earlier_arrival ∧
    bus_speed = 40 ∧
    car_speed = 60 := by
  sorry

end NUMINAMATH_CALUDE_spring_outing_speeds_l1031_103173


namespace NUMINAMATH_CALUDE_two_digit_numbers_problem_l1031_103199

theorem two_digit_numbers_problem :
  ∃ (x y : ℕ), 
    x > y ∧ 
    x ≥ 10 ∧ x < 100 ∧ 
    y ≥ 10 ∧ y < 100 ∧ 
    1000 * x + y = 2 * (1000 * y + 10 * x) + 590 ∧
    2 * x + 3 * y = 72 ∧
    x = 21 ∧ 
    y = 10 ∧
    ∀ (a b : ℕ), 
      (a > b ∧ 
       a ≥ 10 ∧ a < 100 ∧ 
       b ≥ 10 ∧ b < 100 ∧ 
       1000 * a + b = 2 * (1000 * b + 10 * a) + 590 ∧
       2 * a + 3 * b = 72) → 
      (a = 21 ∧ b = 10) :=
by sorry


end NUMINAMATH_CALUDE_two_digit_numbers_problem_l1031_103199


namespace NUMINAMATH_CALUDE_farm_animals_count_l1031_103104

theorem farm_animals_count :
  ∀ (num_hens num_cows : ℕ),
    num_hens = 28 →
    2 * num_hens + 4 * num_cows = 136 →
    num_hens + num_cows = 48 := by
  sorry

end NUMINAMATH_CALUDE_farm_animals_count_l1031_103104


namespace NUMINAMATH_CALUDE_sin_cos_sixth_power_sum_l1031_103141

theorem sin_cos_sixth_power_sum (θ : ℝ) (h : Real.sin (2 * θ) = 1 / 4) :
  Real.sin θ ^ 6 + Real.cos θ ^ 6 = 61 / 64 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_sixth_power_sum_l1031_103141


namespace NUMINAMATH_CALUDE_welders_left_correct_l1031_103143

/-- The number of welders who left for the other project after the first day -/
def welders_who_left : ℕ := 11

/-- The initial number of welders -/
def initial_welders : ℕ := 16

/-- The number of days to complete the order with all welders -/
def initial_days : ℕ := 8

/-- The additional days needed by remaining welders to complete the order -/
def additional_days : ℕ := 16

/-- The total amount of work to be done -/
def total_work : ℝ := initial_welders * initial_days

/-- The work done in the first day -/
def first_day_work : ℝ := initial_welders

/-- The remaining work after the first day -/
def remaining_work : ℝ := total_work - first_day_work

theorem welders_left_correct :
  (initial_welders - welders_who_left) * (initial_days + additional_days) = remaining_work :=
sorry

end NUMINAMATH_CALUDE_welders_left_correct_l1031_103143


namespace NUMINAMATH_CALUDE_guest_bathroom_towel_sets_l1031_103181

theorem guest_bathroom_towel_sets :
  let master_sets : ℕ := 4
  let guest_price : ℚ := 40
  let master_price : ℚ := 50
  let discount : ℚ := 20 / 100
  let total_spent : ℚ := 224
  let discounted_guest_price : ℚ := guest_price * (1 - discount)
  let discounted_master_price : ℚ := master_price * (1 - discount)
  ∃ guest_sets : ℕ,
    guest_sets * discounted_guest_price + master_sets * discounted_master_price = total_spent ∧
    guest_sets = 2 :=
by sorry

end NUMINAMATH_CALUDE_guest_bathroom_towel_sets_l1031_103181


namespace NUMINAMATH_CALUDE_line_tangent_to_parabola_l1031_103109

/-- A line with equation x - 2y = r is tangent to a parabola with equation y = x^2 - r
    if and only if r = -1/8 -/
theorem line_tangent_to_parabola (r : ℝ) :
  (∃ x y, x - 2*y = r ∧ y = x^2 - r ∧
    ∀ x' y', x' - 2*y' = r ∧ y' = x'^2 - r → (x', y') = (x, y)) ↔
  r = -1/8 := by
sorry

end NUMINAMATH_CALUDE_line_tangent_to_parabola_l1031_103109


namespace NUMINAMATH_CALUDE_tens_digit_of_2020_pow_2021_minus_2022_l1031_103171

theorem tens_digit_of_2020_pow_2021_minus_2022 : ∃ n : ℕ, 
  (2020^2021 - 2022) % 100 = 70 + n ∧ n < 10 :=
by sorry

end NUMINAMATH_CALUDE_tens_digit_of_2020_pow_2021_minus_2022_l1031_103171


namespace NUMINAMATH_CALUDE_sum_mod_seven_l1031_103154

theorem sum_mod_seven : (4123 + 4124 + 4125 + 4126 + 4127) % 7 = 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_mod_seven_l1031_103154


namespace NUMINAMATH_CALUDE_ellipse_condition_l1031_103126

def is_ellipse_with_y_axis_foci (m : ℝ) : Prop :=
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ b > a ∧
  ∀ (x y : ℝ), x^2 / (5 - m) + y^2 / (m - 1) = 1 ↔ 
    x^2 / a^2 + y^2 / b^2 = 1

theorem ellipse_condition (m : ℝ) : 
  is_ellipse_with_y_axis_foci m ↔ 3 < m ∧ m < 5 :=
sorry

end NUMINAMATH_CALUDE_ellipse_condition_l1031_103126


namespace NUMINAMATH_CALUDE_circle_chords_count_l1031_103197

/-- The number of combinations of n items taken k at a time -/
def choose (n k : ℕ) : ℕ := (n.factorial) / (k.factorial * (n - k).factorial)

/-- The number of points on the circle -/
def num_points : ℕ := 10

/-- The number of points needed to form a chord -/
def points_per_chord : ℕ := 2

theorem circle_chords_count :
  choose num_points points_per_chord = 45 := by sorry

end NUMINAMATH_CALUDE_circle_chords_count_l1031_103197


namespace NUMINAMATH_CALUDE_sum_x_y_l1031_103150

/-- The smallest positive integer x such that 480x is a perfect square -/
def x : ℕ := 30

/-- The smallest positive integer y such that 480y is a perfect cube -/
def y : ℕ := 450

/-- 480 * x is a perfect square -/
axiom x_square : ∃ n : ℕ, 480 * x = n^2

/-- 480 * y is a perfect cube -/
axiom y_cube : ∃ n : ℕ, 480 * y = n^3

/-- x is the smallest positive integer such that 480x is a perfect square -/
axiom x_smallest : ∀ z : ℕ, z > 0 → z < x → ¬∃ n : ℕ, 480 * z = n^2

/-- y is the smallest positive integer such that 480y is a perfect cube -/
axiom y_smallest : ∀ z : ℕ, z > 0 → z < y → ¬∃ n : ℕ, 480 * z = n^3

theorem sum_x_y : x + y = 480 := by sorry

end NUMINAMATH_CALUDE_sum_x_y_l1031_103150


namespace NUMINAMATH_CALUDE_f_positive_iff_x_range_l1031_103158

-- Define the function f
def f (a x : ℝ) : ℝ := x^2 + (a - 4) * x + 4 - 2 * a

-- State the theorem
theorem f_positive_iff_x_range (a : ℝ) (h : a ∈ Set.Icc (-1) 1) :
  (∀ x : ℝ, f a x > 0) ↔ (∀ x : ℝ, x < 1 ∨ x > 3) :=
sorry

end NUMINAMATH_CALUDE_f_positive_iff_x_range_l1031_103158


namespace NUMINAMATH_CALUDE_parabola_c_value_l1031_103115

/-- A parabola with equation x = ay^2 + by + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The x-coordinate of a point on the parabola -/
def Parabola.x_coord (p : Parabola) (y : ℝ) : ℝ :=
  p.a * y^2 + p.b * y + p.c

theorem parabola_c_value (p : Parabola) :
  p.x_coord 1 = -3 →   -- vertex at (-3, 1)
  p.x_coord 3 = -1 →   -- passes through (-1, 3)
  p.c = -5/2 := by
    sorry

end NUMINAMATH_CALUDE_parabola_c_value_l1031_103115


namespace NUMINAMATH_CALUDE_method1_saves_more_l1031_103139

/-- Represents the price of a badminton racket in yuan -/
def racket_price : ℕ := 20

/-- Represents the price of a shuttlecock in yuan -/
def shuttlecock_price : ℕ := 5

/-- Represents the number of rackets to be purchased -/
def num_rackets : ℕ := 4

/-- Represents the number of shuttlecocks to be purchased -/
def num_shuttlecocks : ℕ := 30

/-- Calculates the cost using discount method ① -/
def cost_method1 : ℕ := racket_price * num_rackets + shuttlecock_price * (num_shuttlecocks - num_rackets)

/-- Calculates the cost using discount method ② -/
def cost_method2 : ℚ := (racket_price * num_rackets + shuttlecock_price * num_shuttlecocks) * 92 / 100

/-- Theorem stating that discount method ① saves more money than method ② -/
theorem method1_saves_more : cost_method1 < cost_method2 := by
  sorry


end NUMINAMATH_CALUDE_method1_saves_more_l1031_103139


namespace NUMINAMATH_CALUDE_man_son_age_ratio_l1031_103123

/-- Proves that the ratio of a man's age to his son's age in two years is 2:1,
    given the man is 35 years older than his son and the son's present age is 33. -/
theorem man_son_age_ratio :
  let son_age : ℕ := 33
  let man_age : ℕ := son_age + 35
  let son_age_in_two_years : ℕ := son_age + 2
  let man_age_in_two_years : ℕ := man_age + 2
  man_age_in_two_years = 2 * son_age_in_two_years := by
  sorry

#check man_son_age_ratio

end NUMINAMATH_CALUDE_man_son_age_ratio_l1031_103123


namespace NUMINAMATH_CALUDE_multiple_of_eleven_with_specific_digits_l1031_103151

theorem multiple_of_eleven_with_specific_digits : ∃ (A B : ℕ), A < 10 ∧ B < 10 ∧
  (85 * 10^5 + A * 10^4 + 3 * 10^3 + 6 * 10^2 + B * 10 + 4) % 11 = 0 ∧
  (9 * 10^6 + 1 * 10^5 + 7 * 10^4 + B * 10^3 + A * 10^2 + 5 * 10 + 0) % 11 = 0 :=
by sorry

end NUMINAMATH_CALUDE_multiple_of_eleven_with_specific_digits_l1031_103151


namespace NUMINAMATH_CALUDE_tanner_money_left_l1031_103105

def september_savings : ℕ := 17
def october_savings : ℕ := 48
def november_savings : ℕ := 25
def video_game_cost : ℕ := 49

theorem tanner_money_left : 
  september_savings + october_savings + november_savings - video_game_cost = 41 := by
  sorry

end NUMINAMATH_CALUDE_tanner_money_left_l1031_103105


namespace NUMINAMATH_CALUDE_P_smallest_l1031_103185

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m^2
def is_perfect_cube (n : ℕ) : Prop := ∃ m : ℕ, n = m^3
def is_perfect_fifth_power (n : ℕ) : Prop := ∃ m : ℕ, n = m^5

def H : ℕ := sorry

axiom H_def : H > 0 ∧ 
  is_perfect_cube (H / 2) ∧ 
  is_perfect_fifth_power (H / 3) ∧ 
  is_perfect_square (H / 5)

axiom H_minimal : ∀ n : ℕ, n > 0 → 
  is_perfect_cube (n / 2) → 
  is_perfect_fifth_power (n / 3) → 
  is_perfect_square (n / 5) → 
  H ≤ n

def P : ℕ := sorry

axiom P_def : P > 0 ∧ 
  is_perfect_square (P / 2) ∧ 
  is_perfect_cube (P / 3) ∧ 
  is_perfect_fifth_power (P / 5)

axiom P_minimal : ∀ n : ℕ, n > 0 → 
  is_perfect_square (n / 2) → 
  is_perfect_cube (n / 3) → 
  is_perfect_fifth_power (n / 5) → 
  P ≤ n

def S : ℕ := sorry

axiom S_def : S > 0 ∧ 
  is_perfect_fifth_power (S / 2) ∧ 
  is_perfect_square (S / 3) ∧ 
  is_perfect_cube (S / 5)

axiom S_minimal : ∀ n : ℕ, n > 0 → 
  is_perfect_fifth_power (n / 2) → 
  is_perfect_square (n / 3) → 
  is_perfect_cube (n / 5) → 
  S ≤ n

theorem P_smallest : P < S ∧ P < H := by sorry

end NUMINAMATH_CALUDE_P_smallest_l1031_103185


namespace NUMINAMATH_CALUDE_factorial_equivalences_l1031_103122

/-- The number of arrangements of n objects taken k at a time -/
def A (n k : ℕ) : ℕ := sorry

/-- Factorial function -/
def factorial (n : ℕ) : ℕ := sorry

theorem factorial_equivalences (n : ℕ) : 
  (A n (n - 1) = factorial n) ∧ 
  ((1 / (n + 1 : ℚ)) * A (n + 1) (n + 1) = factorial n) := by sorry

end NUMINAMATH_CALUDE_factorial_equivalences_l1031_103122


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l1031_103110

theorem absolute_value_equation_solution :
  ∃! y : ℝ, |y - 8| + 3 * y = 12 :=
by
  -- The unique solution is y = 2
  use 2
  constructor
  · -- Prove that y = 2 satisfies the equation
    simp
    norm_num
  · -- Prove uniqueness
    intro z hz
    -- Proof goes here
    sorry

#check absolute_value_equation_solution

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l1031_103110


namespace NUMINAMATH_CALUDE_parabola_directrix_l1031_103195

open Real

-- Define the parabola
structure Parabola where
  p : ℝ
  eq : ℝ → ℝ → Prop
  h_pos : p > 0
  h_eq : ∀ x y, eq x y ↔ y^2 = 2*p*x

-- Define points
structure Point where
  x : ℝ
  y : ℝ

-- Define the problem setup
def problem_setup (C : Parabola) (O F P Q : Point) : Prop :=
  -- O is the coordinate origin
  O.x = 0 ∧ O.y = 0
  -- F is the focus of parabola C
  ∧ F.x = C.p/2 ∧ F.y = 0
  -- P is a point on C
  ∧ C.eq P.x P.y
  -- PF is perpendicular to the x-axis
  ∧ P.x = F.x
  -- Q is a point on the x-axis
  ∧ Q.y = 0
  -- PQ is perpendicular to OP
  ∧ (Q.y - P.y) * (P.x - O.x) + (Q.x - P.x) * (P.y - O.y) = 0
  -- |FQ| = 6
  ∧ |F.x - Q.x| = 6

-- Theorem statement
theorem parabola_directrix (C : Parabola) (O F P Q : Point) 
  (h : problem_setup C O F P Q) : 
  ∃ (x : ℝ), x = -3/2 ∧ ∀ (y : ℝ), C.eq x y ↔ False :=
sorry

end NUMINAMATH_CALUDE_parabola_directrix_l1031_103195


namespace NUMINAMATH_CALUDE_tiles_for_18_24_room_l1031_103180

/-- Calculates the number of tiles needed for a rectangular room with a double border --/
def tilesNeeded (length width : ℕ) : ℕ :=
  let borderTiles := 2 * (length - 2) + 2 * (length - 4) + 2 * (width - 2) + 2 * (width - 4) + 8
  let innerLength := length - 4
  let innerWidth := width - 4
  let innerArea := innerLength * innerWidth
  let innerTiles := (innerArea + 8) / 9  -- Ceiling division
  borderTiles + innerTiles

/-- The theorem states that for an 18 by 24 foot room, 183 tiles are needed --/
theorem tiles_for_18_24_room : tilesNeeded 24 18 = 183 := by
  sorry

end NUMINAMATH_CALUDE_tiles_for_18_24_room_l1031_103180


namespace NUMINAMATH_CALUDE_triangle_formation_l1031_103147

/-- Check if three lengths can form a triangle -/
def canFormTriangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- Given two stick lengths -/
def stick1 : ℝ := 3
def stick2 : ℝ := 5

theorem triangle_formation :
  ¬(canFormTriangle stick1 stick2 2) ∧
  (canFormTriangle stick1 stick2 3) ∧
  (canFormTriangle stick1 stick2 4) ∧
  (canFormTriangle stick1 stick2 6) := by
  sorry

end NUMINAMATH_CALUDE_triangle_formation_l1031_103147


namespace NUMINAMATH_CALUDE_exponential_function_sum_of_extrema_l1031_103146

theorem exponential_function_sum_of_extrema (a : ℝ) : 
  a > 0 → 
  a ≠ 1 → 
  (max (a^1) (a^2) + min (a^1) (a^2) = 6) → 
  a = 2 := by sorry

end NUMINAMATH_CALUDE_exponential_function_sum_of_extrema_l1031_103146


namespace NUMINAMATH_CALUDE_problem_solution_l1031_103174

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| + 3 * x

theorem problem_solution (a : ℝ) (h : a > 0) :
  -- Part 1
  (∀ x, f 1 x ≥ 3 * x + 2 ↔ x ≥ 3 ∨ x ≤ -1) ∧
  -- Part 2
  ((∀ x, f a x ≤ 0 ↔ x ≤ -1) → a = 2) :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l1031_103174


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l1031_103103

/-- An arithmetic sequence {a_n} -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property (a : ℕ → ℝ) (h : arithmetic_sequence a) :
  a 6 + a 8 = 16 → a 4 = 1 → a 10 = 15 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l1031_103103


namespace NUMINAMATH_CALUDE_scientific_notation_of_given_number_l1031_103175

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  property : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

/-- The given number in millions -/
def givenNumber : ℝ := 141260

theorem scientific_notation_of_given_number :
  toScientificNotation givenNumber = ScientificNotation.mk 1.4126 5 (by sorry) :=
sorry

end NUMINAMATH_CALUDE_scientific_notation_of_given_number_l1031_103175


namespace NUMINAMATH_CALUDE_salt_solution_mixture_l1031_103112

theorem salt_solution_mixture (x : ℝ) : 
  (0.6 * x = 0.1 * (x + 1)) → x = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_salt_solution_mixture_l1031_103112


namespace NUMINAMATH_CALUDE_multiple_births_quintuplets_l1031_103162

theorem multiple_births_quintuplets (total_babies : ℕ) 
  (h_total : total_babies = 1500)
  (h_triplets_quadruplets : ∃ (t q : ℕ), t = 3 * q)
  (h_twins_triplets : ∃ (w t : ℕ), w = 2 * t)
  (h_quintuplets_quadruplets : ∃ (q qu : ℕ), q = qu / 2)
  (h_sum : ∃ (w t q qu : ℕ), 2 * w + 3 * t + 4 * q + 5 * qu = total_babies) :
  ∃ (quintuplets : ℕ), quintuplets = 1500 / 11 ∧ 
    quintuplets * 5 = total_babies * 5 / 11 :=
by sorry

end NUMINAMATH_CALUDE_multiple_births_quintuplets_l1031_103162


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_square_roots_solution_l1031_103165

theorem quadratic_equation_solution (x : ℝ) :
  25 * x^2 - 36 = 0 → x = 6/5 ∨ x = -6/5 := by sorry

theorem square_roots_solution (x a : ℝ) :
  a > 0 ∧ (x + 2)^2 = a ∧ (3*x - 10)^2 = a → x = 2 ∧ a = 16 := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_square_roots_solution_l1031_103165


namespace NUMINAMATH_CALUDE_largest_root_is_three_l1031_103179

-- Define the cubic polynomial
def cubic (x : ℝ) : ℝ := x^3 - 3*x^2 - 8*x + 15

-- Define the conditions for p, q, and r
def root_conditions (p q r : ℝ) : Prop :=
  p + q + r = 3 ∧ p*q + p*r + q*r = -8 ∧ p*q*r = -15

-- Theorem statement
theorem largest_root_is_three :
  ∃ (p q r : ℝ), root_conditions p q r ∧
  (cubic p = 0 ∧ cubic q = 0 ∧ cubic r = 0) ∧
  (∀ x : ℝ, cubic x = 0 → x ≤ 3) :=
sorry

end NUMINAMATH_CALUDE_largest_root_is_three_l1031_103179


namespace NUMINAMATH_CALUDE_max_value_g_in_unit_interval_l1031_103127

-- Define the function g(x)
def g (x : ℝ) : ℝ := x * (x^2 - 1)

-- State the theorem
theorem max_value_g_in_unit_interval :
  ∃ (M : ℝ), M = 0 ∧ ∀ x, x ∈ Set.Icc 0 1 → g x ≤ M :=
by
  sorry

end NUMINAMATH_CALUDE_max_value_g_in_unit_interval_l1031_103127


namespace NUMINAMATH_CALUDE_max_value_of_f_l1031_103182

-- Define the quadratic function
def f (x : ℝ) : ℝ := -8 * x^2 + 32 * x - 1

-- State the theorem
theorem max_value_of_f :
  ∃ (M : ℝ), (∀ (x : ℝ), f x ≤ M) ∧ (∃ (x : ℝ), f x = M) ∧ M = 31 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_f_l1031_103182


namespace NUMINAMATH_CALUDE_root_property_l1031_103169

theorem root_property (a : ℝ) : 3 * a^2 - 4 * a + 1 = 0 → 6 * a^2 - 8 * a + 5 = 3 := by
  sorry

end NUMINAMATH_CALUDE_root_property_l1031_103169


namespace NUMINAMATH_CALUDE_calculation_proof_l1031_103131

theorem calculation_proof : 3 * 16 + 3 * 17 + 3 * 20 + 11 = 170 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l1031_103131


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1031_103161

theorem sufficient_not_necessary_condition (x : ℝ) :
  (x > 0 → x^2 + 4*x + 3 > 0) ∧
  ¬(x^2 + 4*x + 3 > 0 → x > 0) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1031_103161


namespace NUMINAMATH_CALUDE_prime_square_plus_two_l1031_103198

theorem prime_square_plus_two (p : ℕ) : 
  Prime p → Prime (p^2 + 2) → p = 3 :=
by sorry

end NUMINAMATH_CALUDE_prime_square_plus_two_l1031_103198


namespace NUMINAMATH_CALUDE_cube_vertex_shapes_l1031_103159

-- Define a cube
structure Cube where
  vertices : Fin 8 → Point3D

-- Define a selection of 4 vertices from a cube
def VertexSelection (c : Cube) := Fin 4 → Fin 8

-- Define geometric shapes that can be formed by 4 vertices
inductive Shape
  | Rectangle
  | TetrahedronIsoscelesRight
  | TetrahedronEquilateral
  | TetrahedronRight

-- Function to check if a selection of vertices forms a specific shape
def formsShape (c : Cube) (s : VertexSelection c) (shape : Shape) : Prop :=
  match shape with
  | Shape.Rectangle => sorry
  | Shape.TetrahedronIsoscelesRight => sorry
  | Shape.TetrahedronEquilateral => sorry
  | Shape.TetrahedronRight => sorry

-- Theorem stating that all these shapes can be formed by selecting 4 vertices from a cube
theorem cube_vertex_shapes (c : Cube) :
  ∃ (s₁ s₂ s₃ s₄ : VertexSelection c),
    formsShape c s₁ Shape.Rectangle ∧
    formsShape c s₂ Shape.TetrahedronIsoscelesRight ∧
    formsShape c s₃ Shape.TetrahedronEquilateral ∧
    formsShape c s₄ Shape.TetrahedronRight :=
  sorry

end NUMINAMATH_CALUDE_cube_vertex_shapes_l1031_103159


namespace NUMINAMATH_CALUDE_double_up_polynomial_properties_l1031_103129

/-- A double-up polynomial is a quadratic polynomial with two real roots, one of which is twice the other. -/
def DoubleUpPolynomial (p q : ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ (k^2 + p*k + q = 0) ∧ ((2*k)^2 + p*(2*k) + q = 0)

theorem double_up_polynomial_properties :
  (∀ p q : ℝ, DoubleUpPolynomial p q →
    (p = -15 → q = 50) ∧
    (∃ k : ℝ, (k = 4 ∨ k = 2) → p + q = 20 ∨ p + q = 2) ∧
    (p + q = 9 → ∃ k : ℝ, k = 3 ∨ k = -3/2)) := by
  sorry

end NUMINAMATH_CALUDE_double_up_polynomial_properties_l1031_103129


namespace NUMINAMATH_CALUDE_daps_equiv_48_dips_l1031_103189

/-- Conversion rate between daps and dops -/
def daps_to_dops : ℚ := 4 / 5

/-- Conversion rate between dops and dips -/
def dops_to_dips : ℚ := 8 / 3

/-- The number of daps equivalent to 48 dips -/
def daps_equiv_to_48_dips : ℚ := 22.5

theorem daps_equiv_48_dips :
  daps_equiv_to_48_dips = 48 * dops_to_dips * daps_to_dops := by
  sorry

end NUMINAMATH_CALUDE_daps_equiv_48_dips_l1031_103189


namespace NUMINAMATH_CALUDE_square_side_length_l1031_103100

theorem square_side_length 
  (x y : ℕ+) 
  (h1 : Nat.gcd x.val y.val = 5)
  (h2 : ∃ (s : ℝ), s > 0 ∧ x.val^2 + y.val^2 = 2 * s^2)
  (h3 : (169 : ℝ) / 6 * Nat.lcm x.val y.val = 2 * s^2) :
  s = 65 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_square_side_length_l1031_103100


namespace NUMINAMATH_CALUDE_mary_screw_ratio_l1031_103102

/-- The number of screws Mary initially has -/
def initial_screws : ℕ := 8

/-- The number of sections Mary needs to split the screws into -/
def num_sections : ℕ := 4

/-- The number of screws needed in each section -/
def screws_per_section : ℕ := 6

/-- The ratio of screws Mary needs to buy to the screws she initially has -/
def screw_ratio : ℚ := 2

theorem mary_screw_ratio : 
  (num_sections * screws_per_section - initial_screws) / initial_screws = screw_ratio := by
  sorry

end NUMINAMATH_CALUDE_mary_screw_ratio_l1031_103102


namespace NUMINAMATH_CALUDE_inequality_equivalence_l1031_103138

theorem inequality_equivalence (x : ℝ) : (x - 3) / 2 ≥ 1 ↔ x ≥ 5 := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l1031_103138


namespace NUMINAMATH_CALUDE_road_construction_cost_ratio_l1031_103172

theorem road_construction_cost_ratio (a m k : ℝ) (ha : a > 0) (hm : m > 0) (hk : k ≥ 3) :
  let x : ℝ := 0.2 * a
  let n : ℝ := a * x + 5
  let P : ℝ := (m * x) / (m * k * n)
  P ≤ 1 / 20 := by
  sorry

end NUMINAMATH_CALUDE_road_construction_cost_ratio_l1031_103172


namespace NUMINAMATH_CALUDE_boat_problem_l1031_103178

theorem boat_problem (total_students : ℕ) (big_boat_capacity small_boat_capacity : ℕ) (total_boats : ℕ) :
  total_students = 52 →
  big_boat_capacity = 8 →
  small_boat_capacity = 4 →
  total_boats = 9 →
  ∃ (big_boats small_boats : ℕ),
    big_boats + small_boats = total_boats ∧
    big_boats * big_boat_capacity + small_boats * small_boat_capacity = total_students ∧
    big_boats = 4 :=
by sorry

end NUMINAMATH_CALUDE_boat_problem_l1031_103178


namespace NUMINAMATH_CALUDE_parking_arrangements_l1031_103194

theorem parking_arrangements (total_spaces : ℕ) (cars : ℕ) (consecutive_empty : ℕ) 
  (h1 : total_spaces = 12) 
  (h2 : cars = 8) 
  (h3 : consecutive_empty = 4) : 
  (Nat.factorial cars) * (total_spaces - cars - consecutive_empty + 1) = 362880 := by
  sorry

end NUMINAMATH_CALUDE_parking_arrangements_l1031_103194


namespace NUMINAMATH_CALUDE_expression_evaluation_l1031_103108

theorem expression_evaluation : 
  3 + Real.sqrt 3 + (1 / (3 + Real.sqrt 3)) + (1 / (Real.sqrt 3 - 3)) = 3 + (2 * Real.sqrt 3) / 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1031_103108


namespace NUMINAMATH_CALUDE_square_difference_equality_l1031_103192

theorem square_difference_equality : (2 + 3)^2 - (2^2 + 3^2) = 12 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_equality_l1031_103192


namespace NUMINAMATH_CALUDE_solution_set_part_i_range_of_a_part_ii_l1031_103107

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| - 2

-- Part I
theorem solution_set_part_i :
  {x : ℝ | f 1 x + |2*x - 3| > 0} = Set.Ioi 2 ∪ Set.Iic (2/3) :=
sorry

-- Part II
theorem range_of_a_part_ii :
  {a : ℝ | ∀ x, f a x < |x - 3|} = Set.Ioo 1 5 :=
sorry

end NUMINAMATH_CALUDE_solution_set_part_i_range_of_a_part_ii_l1031_103107


namespace NUMINAMATH_CALUDE_complex_number_problem_l1031_103177

theorem complex_number_problem (z : ℂ) 
  (h1 : ∃ (r : ℝ), z + 2 * Complex.I = r)
  (h2 : ∃ (m : ℝ), z / (2 - Complex.I) = m) : 
  z = 4 - 2 * Complex.I := by
sorry

end NUMINAMATH_CALUDE_complex_number_problem_l1031_103177


namespace NUMINAMATH_CALUDE_arithmetic_geometric_mean_inequality_l1031_103186

theorem arithmetic_geometric_mean_inequality {x y : ℝ} (hx : x > 0) (hy : y > 0) :
  (x + y) / 2 ≥ Real.sqrt (x * y) ∧
  ((x + y) / 2 = Real.sqrt (x * y) ↔ x = y) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_mean_inequality_l1031_103186


namespace NUMINAMATH_CALUDE_max_value_sum_of_roots_l1031_103136

theorem max_value_sum_of_roots (x : ℝ) (h : -49 ≤ x ∧ x ≤ 49) :
  Real.sqrt (49 + x) + Real.sqrt (49 - x) ≤ 14 ∧
  (Real.sqrt (49 + x) + Real.sqrt (49 - x) = 14 ↔ x = 0) :=
by sorry

end NUMINAMATH_CALUDE_max_value_sum_of_roots_l1031_103136


namespace NUMINAMATH_CALUDE_distance_between_points_l1031_103157

/-- The distance between equidistant points A, B, and C, given specific travel conditions. -/
theorem distance_between_points (v_car v_train t : ℝ) (h1 : v_car = 80) (h2 : v_train = 50) (h3 : t = 7) :
  let S := v_car * t * (25800 / 210)
  S = 861 := by sorry

end NUMINAMATH_CALUDE_distance_between_points_l1031_103157


namespace NUMINAMATH_CALUDE_cricket_average_l1031_103187

theorem cricket_average (current_innings : ℕ) (next_innings_runs : ℕ) (average_increase : ℕ) :
  current_innings = 10 →
  next_innings_runs = 80 →
  average_increase = 4 →
  (current_innings * x + next_innings_runs) / (current_innings + 1) = x + average_increase →
  x = 36 :=
by sorry

end NUMINAMATH_CALUDE_cricket_average_l1031_103187


namespace NUMINAMATH_CALUDE_largest_decimal_l1031_103113

theorem largest_decimal : ∀ (a b c d e : ℝ), 
  a = 0.989 → b = 0.998 → c = 0.981 → d = 0.899 → e = 0.9801 →
  (b ≥ a ∧ b ≥ c ∧ b ≥ d ∧ b ≥ e) := by
  sorry

end NUMINAMATH_CALUDE_largest_decimal_l1031_103113


namespace NUMINAMATH_CALUDE_expression_simplification_l1031_103164

theorem expression_simplification (a : ℝ) (h : a = 1 - Real.sqrt 3) :
  (1 - (2 * a - 1) / (a ^ 2)) / ((a - 1) / (a ^ 2)) = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1031_103164


namespace NUMINAMATH_CALUDE_choir_size_l1031_103144

theorem choir_size :
  ∀ X : ℕ,
  (X / 2 : ℚ) - (X / 6 : ℚ) = 10 →
  X = 30 :=
by
  sorry

#check choir_size

end NUMINAMATH_CALUDE_choir_size_l1031_103144


namespace NUMINAMATH_CALUDE_days_without_email_is_244_l1031_103133

/-- Represents the number of days in a year -/
def days_in_year : ℕ := 365

/-- Represents the email frequency of the first niece -/
def niece1_frequency : ℕ := 4

/-- Represents the email frequency of the second niece -/
def niece2_frequency : ℕ := 6

/-- Represents the email frequency of the third niece -/
def niece3_frequency : ℕ := 8

/-- Calculates the number of days Mr. Thompson did not receive an email from any niece -/
def days_without_email : ℕ :=
  days_in_year - 
  (days_in_year / niece1_frequency + 
   days_in_year / niece2_frequency + 
   days_in_year / niece3_frequency - 
   days_in_year / (niece1_frequency * niece2_frequency) - 
   days_in_year / (niece1_frequency * niece3_frequency) - 
   days_in_year / (niece2_frequency * niece3_frequency) + 
   days_in_year / (niece1_frequency * niece2_frequency * niece3_frequency))

theorem days_without_email_is_244 : days_without_email = 244 := by
  sorry

end NUMINAMATH_CALUDE_days_without_email_is_244_l1031_103133


namespace NUMINAMATH_CALUDE_equation_solution_l1031_103121

theorem equation_solution : 
  ∀ x : ℝ, x * (2 * x - 1) = 4 * x - 2 ↔ x = 2 ∨ x = 1/2 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l1031_103121


namespace NUMINAMATH_CALUDE_equilateral_triangle_area_decrease_l1031_103188

/-- Prove that for an equilateral triangle with an area of 100√3 cm², 
    if each side is decreased by 6 cm, the decrease in area is 51√3 cm². -/
theorem equilateral_triangle_area_decrease 
  (original_area : ℝ) 
  (side_decrease : ℝ) :
  original_area = 100 * Real.sqrt 3 →
  side_decrease = 6 →
  let original_side := Real.sqrt ((4 * original_area) / Real.sqrt 3)
  let new_side := original_side - side_decrease
  let new_area := (new_side^2 * Real.sqrt 3) / 4
  original_area - new_area = 51 * Real.sqrt 3 := by
sorry


end NUMINAMATH_CALUDE_equilateral_triangle_area_decrease_l1031_103188


namespace NUMINAMATH_CALUDE_expression_simplification_l1031_103142

theorem expression_simplification (a b : ℝ) (h1 : a > b) (h2 : b > 1) :
  (a^(b+1) * b^(a+1)) / (b^(b+1) * a^(a+1)) = (a/b)^(b-a) := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1031_103142


namespace NUMINAMATH_CALUDE_y_investment_l1031_103128

/-- Represents the investment and profit share of a person in a business. -/
structure Investor where
  investment : ℕ
  profitShare : ℕ

/-- Represents the business with three investors. -/
structure Business where
  x : Investor
  y : Investor
  z : Investor

/-- The theorem stating that given the conditions, y's investment is 15000 rupees. -/
theorem y_investment (b : Business) : 
  b.x.investment = 5000 ∧ 
  b.z.investment = 7000 ∧ 
  b.x.profitShare = 2 ∧ 
  b.y.profitShare = 6 ∧ 
  b.z.profitShare = 7 → 
  b.y.investment = 15000 :=
by sorry

end NUMINAMATH_CALUDE_y_investment_l1031_103128


namespace NUMINAMATH_CALUDE_fraction_equality_l1031_103170

theorem fraction_equality (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : (4 * x + 2 * y) / (2 * x - 8 * y) = 3) : 
  (2 * x + 8 * y) / (8 * x - 2 * y) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1031_103170


namespace NUMINAMATH_CALUDE_triangle_property_l1031_103149

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ)  -- Angles of the triangle
  (a b c : ℝ)  -- Sides of the triangle opposite to angles A, B, C respectively

-- Define the property that makes a triangle right-angled
def isRightTriangle (t : Triangle) : Prop :=
  t.a^2 = t.b^2 + t.c^2

-- State the theorem
theorem triangle_property (t : Triangle) 
  (h : Real.sin t.C = Real.sin t.A * Real.cos t.B) : 
  isRightTriangle t :=
sorry

end NUMINAMATH_CALUDE_triangle_property_l1031_103149


namespace NUMINAMATH_CALUDE_markup_is_twenty_percent_l1031_103176

/-- Calculates the markup percentage given cost price, discount, and profit percentage. -/
def markup_percentage (cost_price discount : ℚ) (profit_percentage : ℚ) : ℚ :=
  let selling_price := cost_price * (1 + profit_percentage) - discount
  let markup := selling_price - cost_price
  (markup / cost_price) * 100

/-- Theorem stating that under the given conditions, the markup percentage is 20%. -/
theorem markup_is_twenty_percent :
  markup_percentage 180 50 (20/100) = 20 := by
sorry

end NUMINAMATH_CALUDE_markup_is_twenty_percent_l1031_103176


namespace NUMINAMATH_CALUDE_final_face_is_four_l1031_103134

/-- Represents a standard 6-sided die where opposite faces sum to 7 -/
structure StandardDie where
  faces : Fin 6 → Nat
  opposite_sum_seven : ∀ (f : Fin 6), faces f + faces (5 - f) = 7

/-- Represents a move direction -/
inductive Move
| Left
| Forward
| Right
| Back

/-- The sequence of moves in the path -/
def path : List Move := [Move.Left, Move.Forward, Move.Right, Move.Back, Move.Forward, Move.Back]

/-- Simulates rolling the die in a given direction -/
def roll (d : StandardDie) (m : Move) (top : Fin 6) : Fin 6 :=
  sorry

/-- Simulates rolling the die along the entire path -/
def rollPath (d : StandardDie) (initial : Fin 6) : Fin 6 :=
  sorry

/-- Theorem stating that the final top face is 4 regardless of initial state -/
theorem final_face_is_four (d : StandardDie) (initial : Fin 6) :
  d.faces (rollPath d initial) = 4 := by sorry

end NUMINAMATH_CALUDE_final_face_is_four_l1031_103134


namespace NUMINAMATH_CALUDE_variance_of_five_numbers_l1031_103130

theorem variance_of_five_numbers (m : ℝ) 
  (h : (1 + 2 + 3 + 4 + m) / 5 = 3) : 
  ((1 - 3)^2 + (2 - 3)^2 + (3 - 3)^2 + (4 - 3)^2 + (m - 3)^2) / 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_variance_of_five_numbers_l1031_103130


namespace NUMINAMATH_CALUDE_four_digit_difference_l1031_103160

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def digits_in_order (a b c d : ℕ) : Prop := a > b ∧ b > c ∧ c > d

def number_from_digits (a b c d : ℕ) : ℕ := 1000 * a + 100 * b + 10 * c + d

theorem four_digit_difference (a b c d : ℕ) :
  digits_in_order a b c d →
  is_four_digit (number_from_digits a b c d) →
  is_four_digit (number_from_digits a b c d - number_from_digits d c b a) →
  number_from_digits a b c d = 7641 := by
sorry

end NUMINAMATH_CALUDE_four_digit_difference_l1031_103160


namespace NUMINAMATH_CALUDE_triangle_inequality_theorem_l1031_103148

/-- Checks if three numbers can form a triangle --/
def canFormTriangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

theorem triangle_inequality_theorem :
  (¬ canFormTriangle 2 5 7) ∧
  (¬ canFormTriangle 9 3 5) ∧
  (canFormTriangle 4 5 6) ∧
  (¬ canFormTriangle 4 5 10) :=
sorry

end NUMINAMATH_CALUDE_triangle_inequality_theorem_l1031_103148


namespace NUMINAMATH_CALUDE_zoo_cost_theorem_l1031_103193

def zoo_cost (goat_price : ℚ) (goat_count : ℕ) (llama_price_factor : ℚ) 
              (kangaroo_price_factor : ℚ) (kangaroo_multiple : ℕ) 
              (discount_rate : ℚ) : ℚ :=
  let llama_count := 2 * goat_count
  let kangaroo_count := kangaroo_multiple * 5
  let llama_price := goat_price * (1 + llama_price_factor)
  let kangaroo_price := llama_price * (1 - kangaroo_price_factor)
  let goat_cost := goat_price * goat_count
  let llama_cost := llama_price * llama_count
  let kangaroo_cost := kangaroo_price * kangaroo_count
  let total_cost := goat_cost + llama_cost + kangaroo_cost
  let discounted_cost := total_cost * (1 - discount_rate)
  discounted_cost

theorem zoo_cost_theorem : 
  zoo_cost 400 3 (1/2) (1/4) 2 (1/10) = 8850 := by sorry

end NUMINAMATH_CALUDE_zoo_cost_theorem_l1031_103193


namespace NUMINAMATH_CALUDE_line_perp_plane_implies_perp_line_l1031_103152

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between a line and a plane
variable (perpendicular : Line → Plane → Prop)

-- Define the subset relation for a line in a plane
variable (subset : Line → Plane → Prop)

-- Define the perpendicular relation between two lines
variable (perpendicularLines : Line → Line → Prop)

-- State the theorem
theorem line_perp_plane_implies_perp_line 
  (m n : Line) (α : Plane) 
  (h1 : m ≠ n) 
  (h2 : perpendicular m α) 
  (h3 : subset n α) : 
  perpendicularLines m n :=
sorry

end NUMINAMATH_CALUDE_line_perp_plane_implies_perp_line_l1031_103152


namespace NUMINAMATH_CALUDE_quadratic_real_roots_l1031_103183

theorem quadratic_real_roots (a : ℝ) : 
  (∃ x : ℝ, x^2 - x + a = 0) ↔ a ≤ 1/4 := by sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_l1031_103183


namespace NUMINAMATH_CALUDE_rectangle_max_area_l1031_103184

/-- A rectangle with whole number dimensions and perimeter 40 has a maximum area of 100 -/
theorem rectangle_max_area :
  ∀ l w : ℕ,
  l + w = 20 →
  l * w ≤ 100 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_max_area_l1031_103184


namespace NUMINAMATH_CALUDE_correct_factorization_l1031_103118

theorem correct_factorization (x : ℝ) : -x^2 + 2*x - 1 = -(x - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_correct_factorization_l1031_103118


namespace NUMINAMATH_CALUDE_total_cost_is_17_l1031_103119

/-- The total cost of ingredients for Pauline's tacos -/
def total_cost (taco_shells_cost : ℝ) (bell_pepper_cost : ℝ) (bell_pepper_quantity : ℕ) (meat_cost_per_pound : ℝ) (meat_quantity : ℝ) : ℝ :=
  taco_shells_cost + bell_pepper_cost * bell_pepper_quantity + meat_cost_per_pound * meat_quantity

/-- Proof that the total cost of ingredients for Pauline's tacos is $17 -/
theorem total_cost_is_17 :
  total_cost 5 1.5 4 3 2 = 17 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_is_17_l1031_103119


namespace NUMINAMATH_CALUDE_arithmetic_equation_l1031_103114

theorem arithmetic_equation : 3 * 13 + 3 * 14 + 3 * 17 + 11 = 143 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_equation_l1031_103114


namespace NUMINAMATH_CALUDE_arithmetic_evaluation_l1031_103190

theorem arithmetic_evaluation : 4 * (9 - 6) - 8 = 4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_evaluation_l1031_103190


namespace NUMINAMATH_CALUDE_tv_power_consumption_l1031_103145

/-- Given a TV that runs for 4 hours a day, with electricity costing 14 cents per kWh,
    and the TV costing 49 cents to run for a week, prove that the TV uses 125 watts of electricity per hour. -/
theorem tv_power_consumption (hours_per_day : ℝ) (cost_per_kwh : ℝ) (weekly_cost : ℝ) :
  hours_per_day = 4 →
  cost_per_kwh = 0.14 →
  weekly_cost = 0.49 →
  ∃ (watts : ℝ), watts = 125 ∧ 
    (weekly_cost / cost_per_kwh) / (hours_per_day * 7) * 1000 = watts :=
by sorry

end NUMINAMATH_CALUDE_tv_power_consumption_l1031_103145


namespace NUMINAMATH_CALUDE_line_properties_l1031_103140

-- Define the line l
def line_l (x y : ℝ) : Prop := Real.sqrt 3 * x + y + 3 = 0

-- Define the point (0, -3)
def point : ℝ × ℝ := (0, -3)

-- Define the other line
def other_line (x y : ℝ) : Prop := x + (Real.sqrt 3 / 3) * y + Real.sqrt 3 = 0

theorem line_properties :
  (∀ x y, line_l x y ↔ other_line x y) ∧
  line_l point.1 point.2 ∧
  (∀ x y, line_l x y → y / x ≠ Real.tan (60 * π / 180)) ∧
  (∃ x, line_l x 0 ∧ x = -Real.sqrt 3) :=
sorry

end NUMINAMATH_CALUDE_line_properties_l1031_103140


namespace NUMINAMATH_CALUDE_work_completion_time_l1031_103117

theorem work_completion_time (D_A : ℝ) 
  (h1 : D_A > 0)
  (h2 : 1 / D_A + 2 / D_A = 1 / 4) : 
  D_A = 12 := by
sorry

end NUMINAMATH_CALUDE_work_completion_time_l1031_103117


namespace NUMINAMATH_CALUDE_spider_plant_babies_l1031_103191

/-- The number of baby plants produced by a spider plant in a given time period -/
def baby_plants (plants_per_time : ℕ) (times_per_year : ℕ) (years : ℕ) : ℕ :=
  plants_per_time * times_per_year * years

/-- Theorem: A spider plant producing 2 baby plants 2 times a year will have 16 baby plants after 4 years -/
theorem spider_plant_babies : baby_plants 2 2 4 = 16 := by
  sorry

end NUMINAMATH_CALUDE_spider_plant_babies_l1031_103191


namespace NUMINAMATH_CALUDE_area_of_triangle_DBG_l1031_103132

-- Define the triangle and squares
structure RightTriangle :=
  (A B C : ℝ × ℝ)
  (is_right_angle : (A.1 - B.1) * (C.1 - B.1) + (A.2 - B.2) * (C.2 - B.2) = 0)

def square_area (side : ℝ) : ℝ := side ^ 2

-- State the theorem
theorem area_of_triangle_DBG 
  (triangle : RightTriangle)
  (area_ABDE : square_area (Real.sqrt ((triangle.A.1 - triangle.B.1)^2 + (triangle.A.2 - triangle.B.2)^2)) = 8)
  (area_BCFG : square_area (Real.sqrt ((triangle.B.1 - triangle.C.1)^2 + (triangle.B.2 - triangle.C.2)^2)) = 26) :
  let D : ℝ × ℝ := (triangle.A.1 + (triangle.B.2 - triangle.A.2), triangle.A.2 - (triangle.B.1 - triangle.A.1))
  let G : ℝ × ℝ := (triangle.B.1 + (triangle.C.2 - triangle.B.2), triangle.B.2 - (triangle.C.1 - triangle.B.1))
  (1/2) * Real.sqrt ((D.1 - triangle.B.1)^2 + (D.2 - triangle.B.2)^2) * 
         Real.sqrt ((G.1 - triangle.B.1)^2 + (G.2 - triangle.B.2)^2) = 2 * Real.sqrt 13 :=
by sorry

end NUMINAMATH_CALUDE_area_of_triangle_DBG_l1031_103132


namespace NUMINAMATH_CALUDE_new_average_production_l1031_103125

theorem new_average_production (n : ℕ) (past_avg : ℝ) (today_prod : ℝ) :
  n = 12 ∧ past_avg = 50 ∧ today_prod = 115 →
  (n * past_avg + today_prod) / (n + 1) = 55 := by
  sorry

end NUMINAMATH_CALUDE_new_average_production_l1031_103125


namespace NUMINAMATH_CALUDE_distribute_6_4_l1031_103124

/-- The number of ways to distribute n distinguishable balls into k indistinguishable boxes -/
def distribute (n k : ℕ) : ℕ :=
  sorry

/-- The main theorem stating that there are 182 ways to distribute 6 distinguishable balls into 4 indistinguishable boxes -/
theorem distribute_6_4 : distribute 6 4 = 182 := by
  sorry

end NUMINAMATH_CALUDE_distribute_6_4_l1031_103124


namespace NUMINAMATH_CALUDE_root_product_l1031_103120

theorem root_product (d e : ℤ) : 
  (∀ s : ℂ, s^2 - 2*s - 1 = 0 → s^5 - d*s - e = 0) → 
  d * e = 348 := by
sorry

end NUMINAMATH_CALUDE_root_product_l1031_103120


namespace NUMINAMATH_CALUDE_measure_union_ge_sum_measures_l1031_103106

open MeasureTheory Set

-- Define the algebra structure
variable {α : Type*} [MeasurableSpace α]

-- Define the measure
variable (μ : Measure α)

-- Define the sequence of sets
variable (A : ℕ → Set α)

-- State the theorem
theorem measure_union_ge_sum_measures
  (h_algebra : ∀ n, MeasurableSet (A n))
  (h_disjoint : Pairwise (Disjoint on A))
  (h_union : MeasurableSet (⋃ n, A n)) :
  μ (⋃ n, A n) ≥ ∑' n, μ (A n) :=
sorry

end NUMINAMATH_CALUDE_measure_union_ge_sum_measures_l1031_103106


namespace NUMINAMATH_CALUDE_equivalent_discount_l1031_103137

theorem equivalent_discount (original_price : ℝ) (discount1 : ℝ) (discount2 : ℝ) (equivalent_discount : ℝ) : 
  original_price = 50 →
  discount1 = 0.15 →
  discount2 = 0.10 →
  equivalent_discount = 0.235 →
  original_price * (1 - equivalent_discount) = 
  original_price * (1 - discount1) * (1 - discount2) := by
sorry

end NUMINAMATH_CALUDE_equivalent_discount_l1031_103137


namespace NUMINAMATH_CALUDE_intersection_of_p_and_q_when_a_is_one_range_of_a_for_not_p_sufficient_for_not_q_l1031_103111

/-- Proposition p: x^2 - 5ax + 4a^2 < 0, where a > 0 -/
def p (x a : ℝ) : Prop := x^2 - 5*a*x + 4*a^2 < 0 ∧ a > 0

/-- Proposition q: x^2 - 2x - 8 ≤ 0 and x^2 + 3x - 10 > 0 -/
def q (x : ℝ) : Prop := x^2 - 2*x - 8 ≤ 0 ∧ x^2 + 3*x - 10 > 0

/-- The solution set of p -/
def solution_set_p (a : ℝ) : Set ℝ := {x | p x a}

/-- The solution set of q -/
def solution_set_q : Set ℝ := {x | q x}

theorem intersection_of_p_and_q_when_a_is_one :
  (solution_set_p 1) ∩ solution_set_q = Set.Ioo 2 4 := by sorry

theorem range_of_a_for_not_p_sufficient_for_not_q :
  {a : ℝ | ∀ x, ¬(p x a) → ¬(q x)} = Set.Icc 1 2 := by sorry

end NUMINAMATH_CALUDE_intersection_of_p_and_q_when_a_is_one_range_of_a_for_not_p_sufficient_for_not_q_l1031_103111


namespace NUMINAMATH_CALUDE_max_annual_profit_l1031_103166

noncomputable section

def fixed_cost : ℝ := 2.6

def additional_investment (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x < 40 then 10 * x^2 + 300 * x
  else (901 * x^2 - 9450 * x + 10000) / x

def selling_price : ℝ := 0.9

def annual_profit (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x < 40 then selling_price * x - additional_investment x - fixed_cost
  else (selling_price * x * x - (901 * x^2 - 9450 * x + 10000)) / x - fixed_cost

theorem max_annual_profit :
  ∃ (x : ℝ), x = 100 ∧ annual_profit x = 8990 ∧
  ∀ (y : ℝ), y ≥ 0 → annual_profit y ≤ annual_profit x :=
sorry

end NUMINAMATH_CALUDE_max_annual_profit_l1031_103166


namespace NUMINAMATH_CALUDE_car_cleaning_ratio_l1031_103155

theorem car_cleaning_ratio : 
  ∀ (outside_time inside_time total_time : ℕ),
  outside_time = 80 →
  total_time = 100 →
  total_time = outside_time + inside_time →
  (inside_time : ℚ) / (outside_time : ℚ) = 1 / 4 :=
by sorry

end NUMINAMATH_CALUDE_car_cleaning_ratio_l1031_103155


namespace NUMINAMATH_CALUDE_quadratic_equation_m_value_l1031_103196

theorem quadratic_equation_m_value (m : ℝ) :
  (∀ x : ℝ, x^2 + m*x + 9 = (x + 3)^2) → m = 6 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_m_value_l1031_103196


namespace NUMINAMATH_CALUDE_min_value_when_a_is_quarter_range_of_a_for_full_range_l1031_103135

-- Define the piecewise function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (2 - 4*a) * a^x + a else Real.log x

-- Theorem 1: Minimum value of f(x) when a = 1/4 is 0
theorem min_value_when_a_is_quarter :
  ∀ x : ℝ, f (1/4) x ≥ 0 ∧ ∃ x₀ : ℝ, f (1/4) x₀ = 0 :=
sorry

-- Theorem 2: Range of f(x) is R iff 1/2 < a ≤ 3/4
theorem range_of_a_for_full_range :
  ∀ a : ℝ, (a > 0 ∧ a ≠ 1) →
    (∀ y : ℝ, ∃ x : ℝ, f a x = y) ↔ (1/2 < a ∧ a ≤ 3/4) :=
sorry

end NUMINAMATH_CALUDE_min_value_when_a_is_quarter_range_of_a_for_full_range_l1031_103135


namespace NUMINAMATH_CALUDE_digit_equation_solutions_l1031_103153

theorem digit_equation_solutions (n : ℕ) (x y z : ℕ) :
  n ≥ 2 →
  let a : ℚ := x * (10^n - 1) / 9
  let b : ℚ := y * (10^n - 1) / 9
  let c : ℚ := z * (10^(2*n) - 1) / 9
  a^2 + b = c →
  ((x = 3 ∧ y = 2 ∧ z = 1) ∨
   (x = 6 ∧ y = 8 ∧ z = 4) ∨
   (x = 8 ∧ y = 3 ∧ z = 7 ∧ n = 2)) :=
by sorry

end NUMINAMATH_CALUDE_digit_equation_solutions_l1031_103153


namespace NUMINAMATH_CALUDE_fraction_power_equality_l1031_103167

theorem fraction_power_equality : (72000 ^ 5 : ℕ) / (9000 ^ 5) = 32768 := by sorry

end NUMINAMATH_CALUDE_fraction_power_equality_l1031_103167


namespace NUMINAMATH_CALUDE_book_division_proof_l1031_103163

def number_of_divisions (total : ℕ) (target : ℕ) : ℕ :=
  if total ≤ target then 0
  else 1 + number_of_divisions (total / 2) target

theorem book_division_proof :
  number_of_divisions 400 25 = 4 :=
by sorry

end NUMINAMATH_CALUDE_book_division_proof_l1031_103163


namespace NUMINAMATH_CALUDE_linear_function_composition_function_transformation_l1031_103116

-- Part 1
def is_linear (f : ℝ → ℝ) : Prop :=
  ∃ a b : ℝ, ∀ x, f x = a * x + b

theorem linear_function_composition (f : ℝ → ℝ) :
  is_linear f → (∀ x, f (f x) = 4 * x - 1) →
  (∀ x, f x = 2 * x - 1/3) ∨ (∀ x, f x = -2 * x + 1) :=
sorry

-- Part 2
theorem function_transformation (f : ℝ → ℝ) :
  (∀ x, f (1 - x) = 2 * x^2 - x + 1) →
  (∀ x, f x = 2 * x^2 - 3 * x + 2) :=
sorry

end NUMINAMATH_CALUDE_linear_function_composition_function_transformation_l1031_103116
