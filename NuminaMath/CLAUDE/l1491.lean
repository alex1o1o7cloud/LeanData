import Mathlib

namespace NUMINAMATH_CALUDE_first_hundred_complete_l1491_149147

/-- Represents a color of a number in the sequence -/
inductive Color
| Blue
| Red

/-- Represents the properties of the sequence of 200 numbers -/
structure NumberSequence :=
  (numbers : Fin 200 → ℕ)
  (colors : Fin 200 → Color)
  (blue_ascending : ∀ i j, i < j → colors i = Color.Blue → colors j = Color.Blue → numbers i < numbers j)
  (red_descending : ∀ i j, i < j → colors i = Color.Red → colors j = Color.Red → numbers i > numbers j)
  (blue_range : ∀ n, n ∈ Finset.range 100 → ∃ i, colors i = Color.Blue ∧ numbers i = n + 1)
  (red_range : ∀ n, n ∈ Finset.range 100 → ∃ i, colors i = Color.Red ∧ numbers i = 100 - n)

/-- The main theorem stating that the first 100 numbers contain all natural numbers from 1 to 100 -/
theorem first_hundred_complete (seq : NumberSequence) :
  ∀ n, n ∈ Finset.range 100 → ∃ i, i < 100 ∧ seq.numbers i = n + 1 :=
sorry

end NUMINAMATH_CALUDE_first_hundred_complete_l1491_149147


namespace NUMINAMATH_CALUDE_root_equation_m_value_l1491_149190

theorem root_equation_m_value (x m : ℝ) : 
  (3 / x = m / (x - 3)) → (x = 6) → (m = 3 / 2) := by
  sorry

end NUMINAMATH_CALUDE_root_equation_m_value_l1491_149190


namespace NUMINAMATH_CALUDE_gumball_probability_l1491_149124

/-- Given a box of gumballs with blue, green, red, and purple colors, 
    prove that the probability of selecting either a red or a purple gumball is 0.45, 
    given that the probability of selecting a blue gumball is 0.3 
    and the probability of selecting a green gumball is 0.25. -/
theorem gumball_probability (blue green red purple : ℝ) 
  (h1 : blue = 0.3) 
  (h2 : green = 0.25) 
  (h3 : blue + green + red + purple = 1) : 
  red + purple = 0.45 := by
  sorry

end NUMINAMATH_CALUDE_gumball_probability_l1491_149124


namespace NUMINAMATH_CALUDE_washing_machines_removed_l1491_149117

theorem washing_machines_removed (
  num_containers : ℕ) (crates_per_container : ℕ) (boxes_per_crate : ℕ)
  (machines_per_box : ℕ) (num_workers : ℕ) (machines_removed_per_box : ℕ)
  (h1 : num_containers = 100)
  (h2 : crates_per_container = 30)
  (h3 : boxes_per_crate = 15)
  (h4 : machines_per_box = 10)
  (h5 : num_workers = 6)
  (h6 : machines_removed_per_box = 4)
  : (num_containers * crates_per_container * boxes_per_crate * machines_removed_per_box * num_workers) = 180000 := by
  sorry

#check washing_machines_removed

end NUMINAMATH_CALUDE_washing_machines_removed_l1491_149117


namespace NUMINAMATH_CALUDE_trendy_haircut_cost_is_8_l1491_149112

/-- The cost of a trendy haircut -/
def trendy_haircut_cost : ℕ → Prop
| cost => 
  let normal_cost : ℕ := 5
  let special_cost : ℕ := 6
  let normal_per_day : ℕ := 5
  let special_per_day : ℕ := 3
  let trendy_per_day : ℕ := 2
  let days_per_week : ℕ := 7
  let total_weekly_earnings : ℕ := 413
  (normal_cost * normal_per_day + special_cost * special_per_day + cost * trendy_per_day) * days_per_week = total_weekly_earnings

theorem trendy_haircut_cost_is_8 : trendy_haircut_cost 8 := by
  sorry

end NUMINAMATH_CALUDE_trendy_haircut_cost_is_8_l1491_149112


namespace NUMINAMATH_CALUDE_points_on_circle_l1491_149121

theorem points_on_circle (t : ℝ) (h : t ≠ 0) :
  ∃ (a : ℝ), (((t^2 + 1) / t)^2 + ((t^2 - 1) / t)^2) = a := by
  sorry

end NUMINAMATH_CALUDE_points_on_circle_l1491_149121


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequences_l1491_149116

/-- An arithmetic sequence {a_n} with a_2 = 2 and a_5 = 8 -/
def a : ℕ → ℝ := sorry

/-- A geometric sequence {b_n} with all terms positive and b_1 = 1 -/
def b : ℕ → ℝ := sorry

/-- Sum of the first n terms of the geometric sequence {b_n} -/
def T (n : ℕ) : ℝ := sorry

theorem arithmetic_geometric_sequences :
  (∀ n : ℕ, n ≥ 1 → a n = 2 * n - 2) ∧
  (a 2 = 2) ∧
  (a 5 = 8) ∧
  (∀ n : ℕ, n ≥ 1 → b n > 0) ∧
  (b 1 = 1) ∧
  (b 2 + b 3 = a 4) ∧
  (∀ n : ℕ, n ≥ 1 → T n = 2^n - 1) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequences_l1491_149116


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1491_149126

theorem arithmetic_sequence_sum : ∀ (a₁ aₙ d n : ℕ),
  a₁ = 1 →
  aₙ = 28 →
  d = 3 →
  n * d = aₙ - a₁ + d →
  (n * (a₁ + aₙ)) / 2 = 145 :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1491_149126


namespace NUMINAMATH_CALUDE_smallest_prime_factor_of_sum_l1491_149130

theorem smallest_prime_factor_of_sum (n : ℕ) (m : ℕ) : 
  2 ∣ (2005^2007 + 2007^20015) ∧ 
  ∀ p : ℕ, p < 2 → p.Prime → ¬(p ∣ (2005^2007 + 2007^20015)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_prime_factor_of_sum_l1491_149130


namespace NUMINAMATH_CALUDE_carolyn_practice_days_l1491_149114

/-- Represents the practice schedule of a musician --/
structure PracticeSchedule where
  piano_time : ℕ  -- Daily piano practice time in minutes
  violin_ratio : ℕ  -- Ratio of violin practice time to piano practice time
  total_monthly_time : ℕ  -- Total practice time in a month (in minutes)
  weeks_in_month : ℕ  -- Number of weeks in a month

/-- Calculates the number of practice days per week --/
def practice_days_per_week (schedule : PracticeSchedule) : ℚ :=
  let daily_total := schedule.piano_time * (1 + schedule.violin_ratio)
  let monthly_days := schedule.total_monthly_time / daily_total
  monthly_days / schedule.weeks_in_month

/-- Theorem stating that Carolyn practices 6 days a week --/
theorem carolyn_practice_days (schedule : PracticeSchedule) 
  (h1 : schedule.piano_time = 20)
  (h2 : schedule.violin_ratio = 3)
  (h3 : schedule.total_monthly_time = 1920)
  (h4 : schedule.weeks_in_month = 4) :
  practice_days_per_week schedule = 6 := by
  sorry

#eval practice_days_per_week ⟨20, 3, 1920, 4⟩

end NUMINAMATH_CALUDE_carolyn_practice_days_l1491_149114


namespace NUMINAMATH_CALUDE_big_dig_copper_production_l1491_149136

/-- Represents a mine with its daily ore production and copper percentage -/
structure Mine where
  daily_production : ℝ
  copper_percentage : ℝ

/-- Calculates the total daily copper production from all mines -/
def total_copper_production (mines : List Mine) : ℝ :=
  mines.foldl (fun acc mine => acc + mine.daily_production * mine.copper_percentage) 0

/-- Theorem stating the total daily copper production from all four mines -/
theorem big_dig_copper_production :
  let mine_a : Mine := { daily_production := 4500, copper_percentage := 0.055 }
  let mine_b : Mine := { daily_production := 6000, copper_percentage := 0.071 }
  let mine_c : Mine := { daily_production := 5000, copper_percentage := 0.147 }
  let mine_d : Mine := { daily_production := 3500, copper_percentage := 0.092 }
  let all_mines : List Mine := [mine_a, mine_b, mine_c, mine_d]
  total_copper_production all_mines = 1730.5 := by
  sorry


end NUMINAMATH_CALUDE_big_dig_copper_production_l1491_149136


namespace NUMINAMATH_CALUDE_max_blocks_fit_l1491_149103

/-- Represents the dimensions of a rectangular solid -/
structure Dimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the volume of a rectangular solid given its dimensions -/
def volume (d : Dimensions) : ℕ := d.length * d.width * d.height

/-- The dimensions of the small block -/
def smallBlock : Dimensions := ⟨3, 2, 1⟩

/-- The dimensions of the box -/
def box : Dimensions := ⟨4, 6, 2⟩

/-- The maximum number of small blocks that can fit in the box -/
def maxBlocks : ℕ := 8

theorem max_blocks_fit :
  volume box / volume smallBlock = maxBlocks ∧
  maxBlocks * volume smallBlock ≤ volume box ∧
  (maxBlocks + 1) * volume smallBlock > volume box :=
sorry

end NUMINAMATH_CALUDE_max_blocks_fit_l1491_149103


namespace NUMINAMATH_CALUDE_jerry_cases_jerry_cases_proof_l1491_149105

/-- The number of cases Jerry has, given the following conditions:
  - Each case has 3 shelves
  - Each shelf can hold 20 records
  - Each vinyl record has 60 ridges
  - The shelves are 60% full
  - There are 8640 ridges on all records
-/
theorem jerry_cases : ℕ :=
  let shelves_per_case : ℕ := 3
  let records_per_shelf : ℕ := 20
  let ridges_per_record : ℕ := 60
  let shelf_fullness : ℚ := 3/5
  let total_ridges : ℕ := 8640
  
  4

/-- Proof that Jerry has 4 cases -/
theorem jerry_cases_proof : jerry_cases = 4 := by
  sorry

end NUMINAMATH_CALUDE_jerry_cases_jerry_cases_proof_l1491_149105


namespace NUMINAMATH_CALUDE_cup_volume_ratio_l1491_149184

/-- Given a bottle that can be filled with 10 pours of cup a or 5 pours of cup b,
    prove that the volume of cup b is twice the volume of cup a. -/
theorem cup_volume_ratio (V A B : ℝ) (hA : 10 * A = V) (hB : 5 * B = V) :
  B = 2 * A := by sorry

end NUMINAMATH_CALUDE_cup_volume_ratio_l1491_149184


namespace NUMINAMATH_CALUDE_equal_roots_iff_n_eq_neg_one_l1491_149186

/-- The equation has equal roots if and only if n = -1 -/
theorem equal_roots_iff_n_eq_neg_one (n : ℝ) : 
  (∃! x : ℝ, x ≠ 2 ∧ (x * (x - 2) - (n + 2)) / ((x - 2) * (n - 2)) = x / n) ↔ n = -1 := by
  sorry

end NUMINAMATH_CALUDE_equal_roots_iff_n_eq_neg_one_l1491_149186


namespace NUMINAMATH_CALUDE_integral_inequality_l1491_149150

theorem integral_inequality (a b c : ℝ) (h1 : a ≥ b) (h2 : b ≥ c) (h3 : c ≥ 1) :
  ∫ x in (0 : ℝ)..1, ((1 - a*x)^3 + (1 - b*x)^3 + (1 - c*x)^3 - 3*x) ≥ 
    a*b + b*c + c*a - 3/2*(a + b + c) - 3/4*a*b*c := by
  sorry

end NUMINAMATH_CALUDE_integral_inequality_l1491_149150


namespace NUMINAMATH_CALUDE_odd_function_properties_l1491_149109

/-- A function f: ℝ → ℝ is odd if f(-x) = -f(x) for all x ∈ ℝ -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem odd_function_properties (f : ℝ → ℝ) (h : IsOdd f) :
  (f 0 = 0) ∧
  (∀ x ≥ 0, f x ≥ -1) →
  (∃ x ≥ 0, f x = -1) →
  (∀ x ≤ 0, f x ≤ 1) ∧
  (∃ x ≤ 0, f x = 1) := by
  sorry

end NUMINAMATH_CALUDE_odd_function_properties_l1491_149109


namespace NUMINAMATH_CALUDE_polygon_with_108_degree_interior_angles_is_pentagon_l1491_149177

theorem polygon_with_108_degree_interior_angles_is_pentagon :
  ∀ (n : ℕ) (interior_angle : ℝ),
    interior_angle = 108 →
    (n : ℝ) * (180 - interior_angle) = 360 →
    n = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_polygon_with_108_degree_interior_angles_is_pentagon_l1491_149177


namespace NUMINAMATH_CALUDE_ab_range_l1491_149192

theorem ab_range (a b : ℝ) (h : a * b = a + b + 3) :
  (a * b ≤ 1) ∨ (a * b ≥ 9) := by sorry

end NUMINAMATH_CALUDE_ab_range_l1491_149192


namespace NUMINAMATH_CALUDE_monotonic_increasing_interval_l1491_149170

noncomputable def f (a x : ℝ) : ℝ := a^(x^2 - 3*x + 2)

theorem monotonic_increasing_interval 
  (a : ℝ) 
  (h : a > 1) :
  ∀ x₁ x₂ : ℝ, x₁ ≥ 3/2 ∧ x₂ ≥ 3/2 ∧ x₁ < x₂ → f a x₁ < f a x₂ :=
by sorry

end NUMINAMATH_CALUDE_monotonic_increasing_interval_l1491_149170


namespace NUMINAMATH_CALUDE_simon_age_proof_l1491_149142

/-- Alvin's age in years -/
def alvin_age : ℕ := 30

/-- Simon's age in years -/
def simon_age : ℕ := 10

/-- The difference between half of Alvin's age and Simon's age -/
def age_difference : ℕ := 5

theorem simon_age_proof :
  simon_age = alvin_age / 2 - age_difference :=
by sorry

end NUMINAMATH_CALUDE_simon_age_proof_l1491_149142


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l1491_149107

theorem least_subtraction_for_divisibility :
  ∃! x : ℕ, x ≤ 13 ∧ (427398 - x) % 14 = 0 ∧ ∀ y : ℕ, y < x → (427398 - y) % 14 ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l1491_149107


namespace NUMINAMATH_CALUDE_instructors_reunion_l1491_149133

/-- The number of weeks between Rita's teaching sessions -/
def rita_weeks : ℕ := 5

/-- The number of weeks between Pedro's teaching sessions -/
def pedro_weeks : ℕ := 8

/-- The number of weeks between Elaine's teaching sessions -/
def elaine_weeks : ℕ := 10

/-- The number of weeks between Moe's teaching sessions -/
def moe_weeks : ℕ := 9

/-- The number of weeks until all instructors teach together again -/
def weeks_until_reunion : ℕ := 360

theorem instructors_reunion :
  Nat.lcm rita_weeks (Nat.lcm pedro_weeks (Nat.lcm elaine_weeks moe_weeks)) = weeks_until_reunion :=
sorry

end NUMINAMATH_CALUDE_instructors_reunion_l1491_149133


namespace NUMINAMATH_CALUDE_max_sqrt_sum_l1491_149172

theorem max_sqrt_sum (x : ℝ) (h1 : 0 ≤ x) (h2 : x ≤ 17) :
  Real.sqrt (x + 31) + Real.sqrt (17 - x) + Real.sqrt x ≤ 12 ∧
  ∃ x₀, x₀ = 13 ∧ Real.sqrt (x₀ + 31) + Real.sqrt (17 - x₀) + Real.sqrt x₀ = 12 :=
by sorry

end NUMINAMATH_CALUDE_max_sqrt_sum_l1491_149172


namespace NUMINAMATH_CALUDE_usb_available_space_l1491_149164

theorem usb_available_space (total_capacity : ℝ) (occupied_percentage : ℝ) 
  (h1 : total_capacity = 128)
  (h2 : occupied_percentage = 75) :
  (1 - occupied_percentage / 100) * total_capacity = 32 := by
  sorry

end NUMINAMATH_CALUDE_usb_available_space_l1491_149164


namespace NUMINAMATH_CALUDE_ribbon_distribution_l1491_149135

theorem ribbon_distribution (total_ribbon : ℚ) (num_boxes : ℕ) :
  total_ribbon = 2 / 5 →
  num_boxes = 5 →
  (total_ribbon / num_boxes : ℚ) = 2 / 25 := by
sorry

end NUMINAMATH_CALUDE_ribbon_distribution_l1491_149135


namespace NUMINAMATH_CALUDE_cube_inequality_l1491_149118

theorem cube_inequality (a b : ℝ) : a^3 > b^3 → a > b := by
  sorry

end NUMINAMATH_CALUDE_cube_inequality_l1491_149118


namespace NUMINAMATH_CALUDE_interior_triangle_area_l1491_149146

theorem interior_triangle_area (a b c : ℝ) (ha : a = 16) (hb : b = 324) (hc : c = 100) :
  (1/2 : ℝ) * Real.sqrt a * Real.sqrt b = 36 := by
  sorry

end NUMINAMATH_CALUDE_interior_triangle_area_l1491_149146


namespace NUMINAMATH_CALUDE_least_possible_average_of_four_integers_l1491_149132

theorem least_possible_average_of_four_integers (a b c d : ℤ) : 
  a < b ∧ b < c ∧ c < d ∧  -- Four different integers
  d = 90 ∧                 -- Largest integer is 90
  a ≥ 21 →                 -- Smallest integer is at least 21
  (a + b + c + d) / 4 ≥ 39 ∧ 
  ∃ (x y z w : ℤ), x < y ∧ y < z ∧ z < w ∧ w = 90 ∧ x ≥ 21 ∧ (x + y + z + w) / 4 = 39 :=
by
  sorry

#check least_possible_average_of_four_integers

end NUMINAMATH_CALUDE_least_possible_average_of_four_integers_l1491_149132


namespace NUMINAMATH_CALUDE_smallest_constant_inequality_l1491_149188

theorem smallest_constant_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  Real.sqrt (x / (y + z + 2)) + Real.sqrt (y / (x + z + 2)) + Real.sqrt (z / (x + y + 2)) >
  (4 / Real.sqrt 3) * Real.cos (π / 6) := by
  sorry

end NUMINAMATH_CALUDE_smallest_constant_inequality_l1491_149188


namespace NUMINAMATH_CALUDE_book_distribution_l1491_149152

/-- The number of ways to distribute n distinct books among k people, 
    with each person receiving m books -/
def distribute_books (n k m : ℕ) : ℕ := sorry

/-- The number of ways to choose r items from n distinct items -/
def choose (n r : ℕ) : ℕ := sorry

theorem book_distribution :
  distribute_books 6 3 2 = 90 :=
by
  sorry

end NUMINAMATH_CALUDE_book_distribution_l1491_149152


namespace NUMINAMATH_CALUDE_a_is_irrational_l1491_149138

/-- The n-th digit after the decimal point of a real number -/
noncomputable def nthDigitAfterDecimal (a : ℝ) (n : ℕ) : ℕ := sorry

/-- The digit to the left of the decimal point of a real number -/
noncomputable def digitLeftOfDecimal (x : ℝ) : ℕ := sorry

/-- A real number a satisfying the given condition -/
noncomputable def a : ℝ := sorry

/-- The condition that relates a to √2 -/
axiom a_condition : ∀ n : ℕ, nthDigitAfterDecimal a n = digitLeftOfDecimal (n * Real.sqrt 2)

theorem a_is_irrational : Irrational a := by sorry

end NUMINAMATH_CALUDE_a_is_irrational_l1491_149138


namespace NUMINAMATH_CALUDE_smallest_five_digit_congruent_to_2_mod_37_l1491_149141

theorem smallest_five_digit_congruent_to_2_mod_37 : ∃ n : ℕ,
  (n ≥ 10000 ∧ n < 100000) ∧  -- Five-digit positive integer
  (n ≡ 2 [ZMOD 37]) ∧         -- Congruent to 2 modulo 37
  (∀ m : ℕ, (m ≥ 10000 ∧ m < 100000) ∧ (m ≡ 2 [ZMOD 37]) → n ≤ m) ∧  -- Smallest such number
  n = 10027 :=                -- The number is 10027
by sorry

end NUMINAMATH_CALUDE_smallest_five_digit_congruent_to_2_mod_37_l1491_149141


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l1491_149176

theorem complex_modulus_problem (z : ℂ) (h : (1 - Complex.I) * z = 1 + 3 * Complex.I) : 
  Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l1491_149176


namespace NUMINAMATH_CALUDE_tangent_point_exists_min_sum_of_squares_l1491_149151

noncomputable section

-- Define the parabola C: x^2 = 2y
def parabola (x y : ℝ) : Prop := x^2 = 2*y

-- Define the focus F(0, 1/2)
def focus : ℝ × ℝ := (0, 1/2)

-- Define the origin O(0, 0)
def origin : ℝ × ℝ := (0, 0)

-- Define a point M on the parabola in the first quadrant
def point_on_parabola (M : ℝ × ℝ) : Prop :=
  parabola M.1 M.2 ∧ M.1 > 0 ∧ M.2 > 0

-- Define the circle through M, F, and O with center Q
def circle_MFO (M Q : ℝ × ℝ) : Prop :=
  (M.1 - Q.1)^2 + (M.2 - Q.2)^2 = (focus.1 - Q.1)^2 + (focus.2 - Q.2)^2 ∧
  (origin.1 - Q.1)^2 + (origin.2 - Q.2)^2 = (focus.1 - Q.1)^2 + (focus.2 - Q.2)^2

-- Distance from Q to the directrix is 3/4
def Q_to_directrix (Q : ℝ × ℝ) : Prop := Q.2 + 1/2 = 3/4

-- Theorem 1: Existence of point M where MQ is tangent to C
theorem tangent_point_exists :
  ∃ M : ℝ × ℝ, point_on_parabola M ∧
  ∃ Q : ℝ × ℝ, circle_MFO M Q ∧ Q_to_directrix Q ∧
  (M.1 = Real.sqrt 2 ∧ M.2 = 1) :=
sorry

-- Define the line l: y = kx + 1/4
def line (k : ℝ) (x y : ℝ) : Prop := y = k*x + 1/4

-- Theorem 2: Minimum value of |AB|^2 + |DE|^2
theorem min_sum_of_squares (k : ℝ) (h : 1/2 ≤ k ∧ k ≤ 2) :
  ∃ A B D E : ℝ × ℝ,
  point_on_parabola A ∧ point_on_parabola B ∧
  line k A.1 A.2 ∧ line k B.1 B.2 ∧
  (∃ Q : ℝ × ℝ, circle_MFO (Real.sqrt 2, 1) Q ∧ Q_to_directrix Q ∧
    line k D.1 D.2 ∧ line k E.1 E.2 ∧
    circle_MFO D Q ∧ circle_MFO E Q) ∧
  (A.1 - B.1)^2 + (A.2 - B.2)^2 + (D.1 - E.1)^2 + (D.2 - E.2)^2 ≥ 13/2 :=
sorry

end NUMINAMATH_CALUDE_tangent_point_exists_min_sum_of_squares_l1491_149151


namespace NUMINAMATH_CALUDE_max_surrounding_sum_l1491_149137

/-- Represents a 3x3 grid of integers -/
def Grid := Matrix (Fin 3) (Fin 3) ℕ

/-- Checks if all elements in a list are distinct -/
def all_distinct (l : List ℕ) : Prop := l.Nodup

/-- Checks if the product of three numbers equals 3240 -/
def product_is_3240 (a b c : ℕ) : Prop := a * b * c = 3240

/-- Checks if a grid satisfies the problem conditions -/
def valid_grid (g : Grid) : Prop :=
  g 1 1 = 45 ∧
  (∀ i j k, (i = 0 ∧ j = k) ∨ (i = 2 ∧ j = k) ∨ (j = 0 ∧ i = k) ∨ (j = 2 ∧ i = k) ∨
            (i + j = 2 ∧ k = 1) ∨ (i = j ∧ k = 1) →
            product_is_3240 (g i j) (g i k) (g j k)) ∧
  all_distinct [g 0 0, g 0 1, g 0 2, g 1 0, g 1 2, g 2 0, g 2 1, g 2 2]

/-- Sum of the eight numbers surrounding the center in a grid -/
def surrounding_sum (g : Grid) : ℕ :=
  g 0 0 + g 0 1 + g 0 2 + g 1 0 + g 1 2 + g 2 0 + g 2 1 + g 2 2

/-- The theorem stating the maximum sum of surrounding numbers -/
theorem max_surrounding_sum :
  ∀ g : Grid, valid_grid g → surrounding_sum g ≤ 160 :=
by sorry

end NUMINAMATH_CALUDE_max_surrounding_sum_l1491_149137


namespace NUMINAMATH_CALUDE_distribute_seven_balls_two_boxes_l1491_149181

/-- The number of ways to distribute distinguishable balls into distinguishable boxes -/
def distribute_balls (num_balls : ℕ) (num_boxes : ℕ) : ℕ :=
  num_boxes ^ num_balls

/-- Theorem: The number of ways to distribute 7 distinguishable balls into 2 distinguishable boxes is 128 -/
theorem distribute_seven_balls_two_boxes : 
  distribute_balls 7 2 = 128 := by
  sorry

end NUMINAMATH_CALUDE_distribute_seven_balls_two_boxes_l1491_149181


namespace NUMINAMATH_CALUDE_scale_length_theorem_l1491_149193

/-- A scale divided into equal parts -/
structure Scale where
  num_parts : ℕ
  part_length : ℝ

/-- The total length of a scale -/
def total_length (s : Scale) : ℝ := s.num_parts * s.part_length

/-- Theorem stating that a scale with 2 parts of 40 inches each has a total length of 80 inches -/
theorem scale_length_theorem :
  ∀ (s : Scale), s.num_parts = 2 ∧ s.part_length = 40 → total_length s = 80 :=
by
  sorry

end NUMINAMATH_CALUDE_scale_length_theorem_l1491_149193


namespace NUMINAMATH_CALUDE_compute_expression_l1491_149139

theorem compute_expression : 12 + 10 * (4 - 9)^2 = 262 := by
  sorry

end NUMINAMATH_CALUDE_compute_expression_l1491_149139


namespace NUMINAMATH_CALUDE_ship_speed_comparison_ship_time_comparison_l1491_149198

/-- Prove that the harmonic mean of two speeds is less than their arithmetic mean -/
theorem ship_speed_comparison 
  (distance : ℝ) 
  (speed_forward : ℝ) 
  (speed_return : ℝ) 
  (h1 : 0 < distance)
  (h2 : 0 < speed_forward)
  (h3 : 0 < speed_return)
  (h4 : speed_forward ≠ speed_return) :
  (2 * speed_forward * speed_return) / (speed_forward + speed_return) < 
  (speed_forward + speed_return) / 2 := by
  sorry

/-- Prove that a ship with varying speeds takes longer than a ship with constant average speed -/
theorem ship_time_comparison 
  (distance : ℝ) 
  (speed_forward : ℝ) 
  (speed_return : ℝ) 
  (h1 : 0 < distance)
  (h2 : 0 < speed_forward)
  (h3 : 0 < speed_return)
  (h4 : speed_forward ≠ speed_return) :
  (2 * distance) / ((2 * speed_forward * speed_return) / (speed_forward + speed_return)) > 
  (2 * distance) / ((speed_forward + speed_return) / 2) := by
  sorry

end NUMINAMATH_CALUDE_ship_speed_comparison_ship_time_comparison_l1491_149198


namespace NUMINAMATH_CALUDE_solution_implies_difference_l1491_149179

theorem solution_implies_difference (m n : ℝ) : 
  (m - n = 2) → (n - m = -2) := by sorry

end NUMINAMATH_CALUDE_solution_implies_difference_l1491_149179


namespace NUMINAMATH_CALUDE_original_number_is_84_l1491_149185

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

def digit_sum (n : ℕ) : ℕ := n % 10 + n / 10

def swap_digits (n : ℕ) : ℕ := (n % 10) * 10 + n / 10

theorem original_number_is_84 (n : ℕ) 
  (h1 : is_two_digit n)
  (h2 : digit_sum n = 12)
  (h3 : n = swap_digits n + 36) :
  n = 84 := by
sorry

end NUMINAMATH_CALUDE_original_number_is_84_l1491_149185


namespace NUMINAMATH_CALUDE_not_all_greater_than_one_l1491_149174

theorem not_all_greater_than_one (a b c : ℝ) 
  (ha : 0 < a ∧ a < 2) 
  (hb : 0 < b ∧ b < 2) 
  (hc : 0 < c ∧ c < 2) : 
  ¬((2 - a) * b > 1 ∧ (2 - b) * c > 1 ∧ (2 - c) * a > 1) := by
  sorry

end NUMINAMATH_CALUDE_not_all_greater_than_one_l1491_149174


namespace NUMINAMATH_CALUDE_cubic_equation_solution_l1491_149129

theorem cubic_equation_solution (x y : ℝ) (h1 : x^(3*y) = 27) (h2 : x = 3) : y = 1 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_solution_l1491_149129


namespace NUMINAMATH_CALUDE_two_digit_number_puzzle_l1491_149119

theorem two_digit_number_puzzle (a b : ℕ) : 
  a < 10 → b < 10 → a ≠ 0 →
  (10 * a + b) - (10 * b + a) = 36 →
  2 * a = b →
  (a + b) - (a - b) = 16 := by
sorry

end NUMINAMATH_CALUDE_two_digit_number_puzzle_l1491_149119


namespace NUMINAMATH_CALUDE_line_intersects_circle_l1491_149189

/-- The line l: y = k(x - 1) intersects the circle C: x² + y² - 3x = 1 for any real number k -/
theorem line_intersects_circle (k : ℝ) : ∃ (x y : ℝ), 
  y = k * (x - 1) ∧ x^2 + y^2 - 3*x = 1 := by
  sorry

end NUMINAMATH_CALUDE_line_intersects_circle_l1491_149189


namespace NUMINAMATH_CALUDE_problem_statement_l1491_149199

theorem problem_statement (x : ℝ) : 3 * x - 1 = 8 → 150 * (1 / x) + 2 = 52 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1491_149199


namespace NUMINAMATH_CALUDE_solution_set_equivalence_l1491_149127

theorem solution_set_equivalence :
  ∀ x : ℝ, (x - 2 > 1 ∧ x < 4) ↔ (3 < x ∧ x < 4) := by sorry

end NUMINAMATH_CALUDE_solution_set_equivalence_l1491_149127


namespace NUMINAMATH_CALUDE_line_intersects_circle_iff_abs_b_le_sqrt2_l1491_149197

/-- The line y=x+b has common points with the circle x²+y²=1 if and only if |b| ≤ √2. -/
theorem line_intersects_circle_iff_abs_b_le_sqrt2 (b : ℝ) : 
  (∃ (x y : ℝ), y = x + b ∧ x^2 + y^2 = 1) ↔ |b| ≤ Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_line_intersects_circle_iff_abs_b_le_sqrt2_l1491_149197


namespace NUMINAMATH_CALUDE_fraction_meaningful_l1491_149148

theorem fraction_meaningful (x : ℝ) : 
  (∃ y : ℝ, y = (x + 2) / (x - 1)) ↔ x ≠ 1 := by
sorry

end NUMINAMATH_CALUDE_fraction_meaningful_l1491_149148


namespace NUMINAMATH_CALUDE_tournament_games_l1491_149168

/-- Calculates the number of games in a single-elimination tournament. -/
def gamesInSingleElimination (n : ℕ) : ℕ := n - 1

/-- Represents the structure of a two-stage tournament. -/
structure TwoStageTournament where
  totalTeams : ℕ
  firstStageGroups : ℕ
  teamsPerGroup : ℕ
  secondStageTeams : ℕ

/-- Calculates the total number of games in a two-stage tournament. -/
def totalGames (t : TwoStageTournament) : ℕ :=
  (t.firstStageGroups * gamesInSingleElimination t.teamsPerGroup) +
  gamesInSingleElimination t.secondStageTeams

/-- Theorem stating the total number of games in the specific tournament described. -/
theorem tournament_games :
  let t : TwoStageTournament := {
    totalTeams := 24,
    firstStageGroups := 4,
    teamsPerGroup := 6,
    secondStageTeams := 4
  }
  totalGames t = 23 := by sorry

end NUMINAMATH_CALUDE_tournament_games_l1491_149168


namespace NUMINAMATH_CALUDE_intersection_point_l1491_149169

/-- The line equation y = 5x - 6 -/
def line_equation (x y : ℝ) : Prop := y = 5 * x - 6

/-- The y-axis has the equation x = 0 -/
def y_axis (x : ℝ) : Prop := x = 0

theorem intersection_point : 
  ∃ (x y : ℝ), line_equation x y ∧ y_axis x ∧ x = 0 ∧ y = -6 :=
sorry

end NUMINAMATH_CALUDE_intersection_point_l1491_149169


namespace NUMINAMATH_CALUDE_noodle_problem_l1491_149195

theorem noodle_problem (x : ℚ) : 
  (2 / 3 : ℚ) * x = 54 → x = 81 := by
  sorry

end NUMINAMATH_CALUDE_noodle_problem_l1491_149195


namespace NUMINAMATH_CALUDE_quadratic_equations_integer_roots_l1491_149102

theorem quadratic_equations_integer_roots :
  ∃ (p q : ℤ), ∀ k : ℕ, k ≤ 9 →
    ∃ (x y : ℤ), x^2 + (p + k) * x + (q + k) = 0 ∧
                 y^2 + (p + k) * y + (q + k) = 0 ∧
                 x ≠ y :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equations_integer_roots_l1491_149102


namespace NUMINAMATH_CALUDE_parabola_directrix_l1491_149110

/-- Given a parabola y² = 2px and a point M(1, m) on it, 
    if the distance from M to its focus is 5, 
    then the equation of its directrix is x = -4 -/
theorem parabola_directrix (p : ℝ) (m : ℝ) :
  m^2 = 2*p  -- Point M(1, m) is on the parabola y² = 2px
  → (1 - p/2)^2 + m^2 = 5^2  -- Distance from M to focus is 5
  → (-p/2 : ℝ) = -4  -- Equation of directrix is x = -4
:= by sorry

end NUMINAMATH_CALUDE_parabola_directrix_l1491_149110


namespace NUMINAMATH_CALUDE_intersection_implies_a_value_l1491_149155

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 4 ≤ 0}
def B (a : ℝ) : Set ℝ := {x | 2*x + a ≤ 0}

-- State the theorem
theorem intersection_implies_a_value :
  ∀ a : ℝ, (A ∩ B a) = {x | -2 ≤ x ∧ x ≤ 1} → a = -2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_a_value_l1491_149155


namespace NUMINAMATH_CALUDE_units_digit_not_eight_l1491_149108

theorem units_digit_not_eight (a b : Nat) :
  a ∈ Finset.range 100 → b ∈ Finset.range 100 →
  (2^a + 5^b) % 10 ≠ 8 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_not_eight_l1491_149108


namespace NUMINAMATH_CALUDE_not_odd_implies_exists_neq_l1491_149100

/-- A function is odd if f(-x) = -f(x) for all x in its domain -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem not_odd_implies_exists_neq (f : ℝ → ℝ) (h : ¬IsOdd f) : 
  ∃ x, f (-x) ≠ -f x := by
  sorry

end NUMINAMATH_CALUDE_not_odd_implies_exists_neq_l1491_149100


namespace NUMINAMATH_CALUDE_inequality_implies_not_six_l1491_149165

theorem inequality_implies_not_six (m : ℝ) : m + 3 < (-m + 1) - (-13) → m ≠ 6 := by
  sorry

end NUMINAMATH_CALUDE_inequality_implies_not_six_l1491_149165


namespace NUMINAMATH_CALUDE_g_of_2_eq_8_l1491_149115

noncomputable def f (x : ℝ) : ℝ := 4 / (3 - x)

noncomputable def g (x : ℝ) : ℝ := 1 / (f.invFun x) + 7

theorem g_of_2_eq_8 : g 2 = 8 := by sorry

end NUMINAMATH_CALUDE_g_of_2_eq_8_l1491_149115


namespace NUMINAMATH_CALUDE_theresa_has_eleven_games_l1491_149162

/-- The number of video games Tory has -/
def tory_games : ℕ := 6

/-- The number of video games Julia has -/
def julia_games : ℕ := tory_games / 3

/-- The number of video games Theresa has -/
def theresa_games : ℕ := 3 * julia_games + 5

/-- Theorem stating that Theresa has 11 video games -/
theorem theresa_has_eleven_games : theresa_games = 11 := by
  sorry

end NUMINAMATH_CALUDE_theresa_has_eleven_games_l1491_149162


namespace NUMINAMATH_CALUDE_area_ratio_equals_side_ratio_l1491_149191

/-- Triangle PQR with angle bisector PS -/
structure AngleBisectorTriangle where
  /-- Length of side PQ -/
  PQ : ℝ
  /-- Length of side PR -/
  PR : ℝ
  /-- Length of side QR -/
  QR : ℝ
  /-- PS is an angle bisector -/
  PS_is_angle_bisector : Bool

/-- The ratio of areas of triangles formed by an angle bisector -/
def area_ratio (t : AngleBisectorTriangle) : ℝ :=
  sorry

/-- Theorem: The ratio of areas of triangles formed by an angle bisector
    is equal to the ratio of the lengths of the sides adjacent to the bisected angle -/
theorem area_ratio_equals_side_ratio (t : AngleBisectorTriangle) 
  (h : t.PS_is_angle_bisector = true) (h1 : t.PQ = 45) (h2 : t.PR = 75) (h3 : t.QR = 64) : 
  area_ratio t = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_area_ratio_equals_side_ratio_l1491_149191


namespace NUMINAMATH_CALUDE_cube_root_of_m_minus_n_l1491_149163

theorem cube_root_of_m_minus_n (m n : ℝ) : 
  (3 * m + 2 * n = 36) → 
  (3 * n + 2 * m = 9) → 
  (m - n)^(1/3) = 3 := by
sorry

end NUMINAMATH_CALUDE_cube_root_of_m_minus_n_l1491_149163


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_greater_than_3_l1491_149143

theorem negation_of_existence (p : ℝ → Prop) :
  (¬ ∃ x, p x) ↔ ∀ x, ¬ p x := by sorry

theorem negation_of_greater_than_3 :
  (¬ ∃ x : ℝ, x^2 > 3) ↔ (∀ x : ℝ, x^2 ≤ 3) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_greater_than_3_l1491_149143


namespace NUMINAMATH_CALUDE_walking_time_calculation_l1491_149194

/-- Given a man who walks and runs at different speeds, this theorem proves
    the time taken to walk a distance that he can run in 1.5 hours. -/
theorem walking_time_calculation (walk_speed run_speed : ℝ) (run_time : ℝ) 
    (h1 : walk_speed = 8)
    (h2 : run_speed = 16)
    (h3 : run_time = 1.5) : 
  (run_speed * run_time) / walk_speed = 3 := by
  sorry

end NUMINAMATH_CALUDE_walking_time_calculation_l1491_149194


namespace NUMINAMATH_CALUDE_salary_increase_l1491_149104

theorem salary_increase
  (num_employees : ℕ)
  (avg_salary : ℝ)
  (manager_salary : ℝ)
  (h1 : num_employees = 20)
  (h2 : avg_salary = 1500)
  (h3 : manager_salary = 3600) :
  let total_salary := num_employees * avg_salary
  let new_total_salary := total_salary + manager_salary
  let new_avg_salary := new_total_salary / (num_employees + 1)
  new_avg_salary - avg_salary = 100 := by
sorry

end NUMINAMATH_CALUDE_salary_increase_l1491_149104


namespace NUMINAMATH_CALUDE_min_value_of_expression_l1491_149125

theorem min_value_of_expression (x y : ℝ) : 
  (x*y - 2)^2 + (x - 1 + y)^2 ≥ 2 ∧ 
  ∃ (a b : ℝ), (a*b - 2)^2 + (a - 1 + b)^2 = 2 :=
by sorry


end NUMINAMATH_CALUDE_min_value_of_expression_l1491_149125


namespace NUMINAMATH_CALUDE_limit_of_sequence_a_l1491_149160

def a (n : ℕ) : ℚ := (1 + 3 * n) / (6 - n)

theorem limit_of_sequence_a :
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |a n - (-3)| < ε :=
by sorry

end NUMINAMATH_CALUDE_limit_of_sequence_a_l1491_149160


namespace NUMINAMATH_CALUDE_radical_conjugate_sum_product_l1491_149153

theorem radical_conjugate_sum_product (a b : ℝ) : 
  (a + Real.sqrt b) + (a - Real.sqrt b) = -6 ∧ 
  (a + Real.sqrt b) * (a - Real.sqrt b) = 4 → 
  a + b = 2 := by
sorry

end NUMINAMATH_CALUDE_radical_conjugate_sum_product_l1491_149153


namespace NUMINAMATH_CALUDE_bales_in_barn_l1491_149111

/-- The number of bales in the barn after Tim stacked new bales -/
def total_bales (initial_bales new_bales : ℕ) : ℕ :=
  initial_bales + new_bales

/-- Theorem stating that the total number of bales is 82 -/
theorem bales_in_barn : total_bales 54 28 = 82 := by
  sorry

end NUMINAMATH_CALUDE_bales_in_barn_l1491_149111


namespace NUMINAMATH_CALUDE_range_of_a_l1491_149171

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, x ∈ Set.Icc 0 1 ∧ x^2 + (1-a)*x + 3-a > 0) ↔ a < 3 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l1491_149171


namespace NUMINAMATH_CALUDE_probability_of_correct_dial_l1491_149154

def first_three_digits : ℕ := 3
def last_four_digits : ℕ := 24
def total_combinations : ℕ := first_three_digits * last_four_digits
def correct_numbers : ℕ := 1

theorem probability_of_correct_dial :
  (correct_numbers : ℚ) / total_combinations = 1 / 72 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_correct_dial_l1491_149154


namespace NUMINAMATH_CALUDE_machine_value_depletion_rate_l1491_149187

/-- The value depletion rate of a machine given its initial value and value after 2 years -/
theorem machine_value_depletion_rate 
  (initial_value : ℝ) 
  (value_after_two_years : ℝ) 
  (h1 : initial_value = 700) 
  (h2 : value_after_two_years = 567) : 
  ∃ (r : ℝ), 
    value_after_two_years = initial_value * (1 - r)^2 ∧ 
    r = 0.1 := by
  sorry

end NUMINAMATH_CALUDE_machine_value_depletion_rate_l1491_149187


namespace NUMINAMATH_CALUDE_shirt_to_wallet_ratio_l1491_149158

/-- The cost of food Mike bought --/
def food_cost : ℚ := 30

/-- The total amount Mike spent on shopping --/
def total_spent : ℚ := 150

/-- The cost of the wallet Mike bought --/
def wallet_cost : ℚ := food_cost + 60

/-- The cost of the shirt Mike bought --/
def shirt_cost : ℚ := total_spent - wallet_cost - food_cost

/-- The theorem stating the ratio of shirt cost to wallet cost --/
theorem shirt_to_wallet_ratio : 
  shirt_cost / wallet_cost = 1 / 3 := by sorry

end NUMINAMATH_CALUDE_shirt_to_wallet_ratio_l1491_149158


namespace NUMINAMATH_CALUDE_sam_pennies_total_l1491_149175

/-- Given that Sam had 98 pennies initially and found 93 more pennies,
    prove that he now has 191 pennies in total. -/
theorem sam_pennies_total (initial : ℕ) (found : ℕ) (h1 : initial = 98) (h2 : found = 93) :
  initial + found = 191 := by
  sorry

end NUMINAMATH_CALUDE_sam_pennies_total_l1491_149175


namespace NUMINAMATH_CALUDE_parabola_equation_l1491_149166

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := 16 * x^2 - 9 * y^2 = 144

-- Define the left vertex of the hyperbola
def left_vertex : ℝ × ℝ := (-3, 0)

-- Define the point P
def point_P : ℝ × ℝ := (2, -4)

-- Define the parabola equations
def parabola1 (x y : ℝ) : Prop := y^2 = 8 * x
def parabola2 (x y : ℝ) : Prop := x^2 = -y

-- Theorem statement
theorem parabola_equation :
  ∀ (f : ℝ × ℝ) (p : (ℝ × ℝ) → Prop),
    (f = left_vertex) →  -- The focus of the parabola is the left vertex of the hyperbola
    (p point_P) →  -- The parabola passes through point P
    (∀ (x y : ℝ), p (x, y) ↔ (parabola1 x y ∨ parabola2 x y)) :=
by sorry

end NUMINAMATH_CALUDE_parabola_equation_l1491_149166


namespace NUMINAMATH_CALUDE_parallel_vectors_k_value_l1491_149156

def vector_a (k : ℝ) : Fin 2 → ℝ := ![1, k]
def vector_b : Fin 2 → ℝ := ![-2, 6]

theorem parallel_vectors_k_value :
  (∃ (c : ℝ), c ≠ 0 ∧ (∀ i, vector_a k i = c * vector_b i)) →
  k = -3 :=
by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_k_value_l1491_149156


namespace NUMINAMATH_CALUDE_cubic_factorization_l1491_149182

theorem cubic_factorization (x : ℝ) : x^3 - 16*x = x*(x+4)*(x-4) := by
  sorry

end NUMINAMATH_CALUDE_cubic_factorization_l1491_149182


namespace NUMINAMATH_CALUDE_student_math_percentage_l1491_149101

/-- The percentage a student got in math, given their history score, third subject score,
    and desired overall average. -/
def math_percentage (history : ℝ) (third_subject : ℝ) (overall_average : ℝ) : ℝ :=
  3 * overall_average - history - third_subject

/-- Theorem stating that the student got 74% in math, given the conditions. -/
theorem student_math_percentage :
  math_percentage 81 70 75 = 74 := by
  sorry

end NUMINAMATH_CALUDE_student_math_percentage_l1491_149101


namespace NUMINAMATH_CALUDE_polynomial_coefficient_sum_l1491_149134

theorem polynomial_coefficient_sum (a a₁ a₂ a₃ a₄ : ℝ) :
  (∀ x : ℝ, (x + 1)^5 - x^5 = a + a₁*(x + 4)^4*x + a₂*(x + 1)^3*x^2 + a₃*(x + 1)^2*x^3 + a₄*(x + 1)*x^4) →
  a₁ + a₃ = 15 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_sum_l1491_149134


namespace NUMINAMATH_CALUDE_diagonal_passes_through_720_cubes_l1491_149140

/-- The number of unit cubes an internal diagonal passes through in a rectangular solid -/
def cubes_passed_by_diagonal (x y z : ℕ) : ℕ :=
  x + y + z - (Nat.gcd x y + Nat.gcd y z + Nat.gcd z x) + Nat.gcd x (Nat.gcd y z)

/-- Theorem: In a 180 × 360 × 450 rectangular solid made of unit cubes, 
    an internal diagonal passes through 720 cubes -/
theorem diagonal_passes_through_720_cubes :
  cubes_passed_by_diagonal 180 360 450 = 720 := by
  sorry

end NUMINAMATH_CALUDE_diagonal_passes_through_720_cubes_l1491_149140


namespace NUMINAMATH_CALUDE_beth_shopping_theorem_l1491_149180

def cans_of_peas : ℕ := 35

def cans_of_corn : ℕ := 10

theorem beth_shopping_theorem :
  cans_of_peas = 2 * cans_of_corn + 15 ∧ cans_of_corn = 10 := by
  sorry

end NUMINAMATH_CALUDE_beth_shopping_theorem_l1491_149180


namespace NUMINAMATH_CALUDE_reflection_of_S_l1491_149128

-- Define the reflection across the x-axis
def reflect_x_axis (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

-- Define the reflection across the line y = -x
def reflect_y_neg_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.2, -p.1)

-- Define the composition of both reflections
def double_reflection (p : ℝ × ℝ) : ℝ × ℝ :=
  reflect_y_neg_x (reflect_x_axis p)

-- Theorem statement
theorem reflection_of_S :
  double_reflection (5, 0) = (0, -5) := by
  sorry

end NUMINAMATH_CALUDE_reflection_of_S_l1491_149128


namespace NUMINAMATH_CALUDE_toby_money_left_l1491_149144

/-- The amount of money Toby received -/
def total_amount : ℚ := 343

/-- The number of brothers Toby has -/
def num_brothers : ℕ := 2

/-- The number of cousins Toby has -/
def num_cousins : ℕ := 4

/-- The percentage of money each brother receives -/
def brother_percentage : ℚ := 12 / 100

/-- The percentage of money each cousin receives -/
def cousin_percentage : ℚ := 7 / 100

/-- The percentage of money spent on mom's gift -/
def mom_gift_percentage : ℚ := 15 / 100

/-- The amount left for Toby after sharing and buying the gift -/
def amount_left : ℚ := 
  total_amount - 
  (num_brothers * (brother_percentage * total_amount) + 
   num_cousins * (cousin_percentage * total_amount) + 
   mom_gift_percentage * total_amount)

theorem toby_money_left : amount_left = 113.19 := by
  sorry

end NUMINAMATH_CALUDE_toby_money_left_l1491_149144


namespace NUMINAMATH_CALUDE_book_pages_sum_l1491_149159

/-- A book with two chapters -/
structure Book where
  chapter1_pages : ℕ
  chapter2_pages : ℕ

/-- The total number of pages in a book -/
def total_pages (b : Book) : ℕ := b.chapter1_pages + b.chapter2_pages

/-- Theorem: A book with 13 pages in the first chapter and 68 pages in the second chapter has 81 pages in total -/
theorem book_pages_sum : 
  ∀ (b : Book), b.chapter1_pages = 13 ∧ b.chapter2_pages = 68 → total_pages b = 81 := by
  sorry

end NUMINAMATH_CALUDE_book_pages_sum_l1491_149159


namespace NUMINAMATH_CALUDE_total_pupils_across_schools_l1491_149173

theorem total_pupils_across_schools (
  girls_A boys_A girls_B boys_B girls_C boys_C : ℕ
) (h1 : girls_A = 542) (h2 : boys_A = 387)
  (h3 : girls_B = 713) (h4 : boys_B = 489)
  (h5 : girls_C = 628) (h6 : boys_C = 361) :
  girls_A + boys_A + girls_B + boys_B + girls_C + boys_C = 3120 := by
  sorry

end NUMINAMATH_CALUDE_total_pupils_across_schools_l1491_149173


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_problem_solution_l1491_149157

theorem least_subtraction_for_divisibility (n : ℕ) (d : ℕ) (h : d > 0) :
  ∃ (x : ℕ), x < d ∧ (n - x) % d = 0 ∧ ∀ (y : ℕ), y < x → (n - y) % d ≠ 0 :=
sorry

theorem problem_solution :
  ∃ (x : ℕ), x = 14 ∧ (10154 - x) % 30 = 0 ∧ ∀ (y : ℕ), y < x → (10154 - y) % 30 ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_problem_solution_l1491_149157


namespace NUMINAMATH_CALUDE_roots_equation_value_l1491_149106

theorem roots_equation_value (α β : ℝ) : 
  α^2 - α - 1 = 0 → β^2 - β - 1 = 0 → α^2 + α * (β^2 - 2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_roots_equation_value_l1491_149106


namespace NUMINAMATH_CALUDE_jongkook_total_points_total_questions_sum_l1491_149145

/-- The number of English problems solved by each student -/
def total_problems : ℕ := 18

/-- The number of 6-point questions Jongkook got correct -/
def correct_six_point : ℕ := 8

/-- The number of 5-point questions Jongkook got correct -/
def correct_five_point : ℕ := 6

/-- The point value of the first type of question -/
def points_type_one : ℕ := 6

/-- The point value of the second type of question -/
def points_type_two : ℕ := 5

/-- Theorem stating that Jongkook's total points is 78 -/
theorem jongkook_total_points :
  correct_six_point * points_type_one + correct_five_point * points_type_two = 78 := by
  sorry

/-- Theorem stating that the sum of correct questions equals the total number of problems -/
theorem total_questions_sum :
  correct_six_point + correct_five_point + (total_problems - correct_six_point - correct_five_point) = total_problems := by
  sorry

end NUMINAMATH_CALUDE_jongkook_total_points_total_questions_sum_l1491_149145


namespace NUMINAMATH_CALUDE_log_identity_l1491_149120

-- Define the base-10 logarithm
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem log_identity : log10 2 ^ 2 + log10 2 * log10 5 + log10 5 = 1 := by
  sorry

end NUMINAMATH_CALUDE_log_identity_l1491_149120


namespace NUMINAMATH_CALUDE_linear_function_parallel_and_point_l1491_149161

-- Define a linear function
def linear_function (k b : ℝ) : ℝ → ℝ := λ x ↦ k * x + b

-- Define parallel lines
def parallel (f g : ℝ → ℝ) : Prop := ∃ c : ℝ, ∀ x : ℝ, f x = g x + c

theorem linear_function_parallel_and_point :
  ∀ k b : ℝ,
  parallel (linear_function k b) (linear_function 2 1) →
  linear_function k b (-3) = 4 →
  linear_function k b = linear_function 2 10 :=
by sorry

end NUMINAMATH_CALUDE_linear_function_parallel_and_point_l1491_149161


namespace NUMINAMATH_CALUDE_long_division_puzzle_l1491_149122

theorem long_division_puzzle : ∃! (a b c d : ℕ), 
  (a < 10) ∧ (b < 10) ∧ (c < 10) ∧ (d < 10) ∧
  (c ≠ 0) ∧ (d ≠ 0) ∧
  (1000 * a + 100 * b + 10 * c + d) / (10 * c + d) = (100 * b + 10 * c + d) ∧
  (10 * c + d) * b = (10 * c + d) ∧
  (a = 3) ∧ (b = 1) ∧ (c = 2) ∧ (d = 5) := by
sorry

end NUMINAMATH_CALUDE_long_division_puzzle_l1491_149122


namespace NUMINAMATH_CALUDE_problem_hexagon_area_l1491_149113

/-- Represents a hexagon formed by stretching a rubber band over pegs on a grid. -/
structure Hexagon where
  interior_points : ℕ
  boundary_points : ℕ

/-- Calculates the area of a hexagon using Pick's Theorem. -/
def area (h : Hexagon) : ℕ :=
  h.interior_points + h.boundary_points / 2 - 1

/-- The hexagon formed on the 5x5 grid as described in the problem. -/
def problem_hexagon : Hexagon :=
  { interior_points := 11
  , boundary_points := 6 }

/-- Theorem stating that the area of the problem hexagon is 13 square units. -/
theorem problem_hexagon_area :
  area problem_hexagon = 13 := by
  sorry

#eval area problem_hexagon  -- Should output 13

end NUMINAMATH_CALUDE_problem_hexagon_area_l1491_149113


namespace NUMINAMATH_CALUDE_total_oranges_l1491_149123

theorem total_oranges (children : ℕ) (oranges_per_child : ℕ) 
  (h1 : children = 4) 
  (h2 : oranges_per_child = 3) : 
  children * oranges_per_child = 12 := by
  sorry

end NUMINAMATH_CALUDE_total_oranges_l1491_149123


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1491_149149

def arithmetic_sequence (a : ℕ → ℝ) := ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)

theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℝ) 
  (h1 : arithmetic_sequence a) 
  (h2 : a 7 - 2 * a 4 = 6) 
  (h3 : a 3 = 2) : 
  ∃ d : ℝ, (∀ n, a (n + 1) = a n + d) ∧ d = 4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1491_149149


namespace NUMINAMATH_CALUDE_election_ratio_l1491_149196

theorem election_ratio :
  ∀ (R D : ℝ),
  R > 0 → D > 0 →
  (0.70 * R + 0.25 * D) - (0.30 * R + 0.75 * D) = 0.039999999999999853 * (R + D) →
  R / D = 1.5 := by
sorry

end NUMINAMATH_CALUDE_election_ratio_l1491_149196


namespace NUMINAMATH_CALUDE_bike_price_proof_l1491_149183

theorem bike_price_proof (upfront_percentage : ℝ) (upfront_payment : ℝ) (total_price : ℝ) :
  upfront_percentage = 0.20 →
  upfront_payment = 240 →
  upfront_percentage * total_price = upfront_payment →
  total_price = 1200 := by
  sorry

end NUMINAMATH_CALUDE_bike_price_proof_l1491_149183


namespace NUMINAMATH_CALUDE_inequality_proof_l1491_149167

theorem inequality_proof (x : ℝ) : 
  2 < x → x < 9/2 → (10*x^2 + 15*x - 75) / ((3*x - 6)*(x + 5)) < 4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1491_149167


namespace NUMINAMATH_CALUDE_three_glass_bottles_weight_l1491_149131

/-- The weight of a glass bottle in grams -/
def glass_bottle_weight : ℝ := sorry

/-- The weight of a plastic bottle in grams -/
def plastic_bottle_weight : ℝ := sorry

/-- The total weight of 4 glass bottles and 5 plastic bottles is 1050 grams -/
axiom total_weight : 4 * glass_bottle_weight + 5 * plastic_bottle_weight = 1050

/-- A glass bottle is 150 grams heavier than a plastic bottle -/
axiom weight_difference : glass_bottle_weight = plastic_bottle_weight + 150

/-- The weight of 3 glass bottles is 600 grams -/
theorem three_glass_bottles_weight : 3 * glass_bottle_weight = 600 := by sorry

end NUMINAMATH_CALUDE_three_glass_bottles_weight_l1491_149131


namespace NUMINAMATH_CALUDE_fourth_side_length_l1491_149178

/-- A quadrilateral inscribed in a circle with radius 150√3, where three sides are 150 units long -/
structure InscribedQuadrilateral where
  -- The radius of the circle
  r : ℝ
  -- The lengths of the four sides of the quadrilateral
  s₁ : ℝ
  s₂ : ℝ
  s₃ : ℝ
  s₄ : ℝ
  -- Conditions
  h_radius : r = 150 * Real.sqrt 3
  h_three_sides : s₁ = 150 ∧ s₂ = 150 ∧ s₃ = 150

/-- The theorem stating that the fourth side of the quadrilateral is 450 units long -/
theorem fourth_side_length (q : InscribedQuadrilateral) : q.s₄ = 450 := by
  sorry

end NUMINAMATH_CALUDE_fourth_side_length_l1491_149178
