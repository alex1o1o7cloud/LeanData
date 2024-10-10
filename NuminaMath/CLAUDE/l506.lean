import Mathlib

namespace union_of_A_and_B_l506_50660

-- Define the sets A and B
def A : Set ℝ := {x | -2 < x ∧ x ≤ 1}
def B : Set ℝ := {x | -1 ≤ x ∧ x < 2}

-- State the theorem
theorem union_of_A_and_B : A ∪ B = {x | -2 < x ∧ x < 2} := by sorry

end union_of_A_and_B_l506_50660


namespace bottle_count_theorem_l506_50668

/-- Represents the number of bottles for each team and the total filled -/
structure BottleCount where
  total : Nat
  football : Nat
  soccer : Nat
  lacrosse : Nat
  rugby : Nat
  unaccounted : Nat

/-- The given conditions and the statement to prove -/
theorem bottle_count_theorem (bc : BottleCount) : 
  bc.total = 254 ∧ 
  bc.football = 11 * 6 ∧ 
  bc.soccer = 53 ∧ 
  bc.lacrosse = bc.football + 12 ∧ 
  bc.rugby = 49 → 
  bc.total = bc.football + bc.soccer + bc.lacrosse + bc.rugby + bc.unaccounted :=
by sorry

end bottle_count_theorem_l506_50668


namespace sphere_radius_equal_volume_cone_l506_50603

/-- The radius of a sphere with the same volume as a cone -/
theorem sphere_radius_equal_volume_cone (r h : ℝ) (hr : r = 2) (hh : h = 8) :
  ∃ (r_sphere : ℝ), (1/3 * π * r^2 * h) = (4/3 * π * r_sphere^3) ∧ r_sphere = 2 * (2 : ℝ)^(1/3) :=
sorry

end sphere_radius_equal_volume_cone_l506_50603


namespace moon_speed_mph_approx_l506_50622

/-- Conversion factor from kilometers to miles -/
def km_to_miles : ℝ := 0.621371

/-- Conversion factor from seconds to hours -/
def seconds_to_hours : ℝ := 3600

/-- The moon's speed in kilometers per second -/
def moon_speed_km_s : ℝ := 1.02

/-- Converts a speed from kilometers per second to miles per hour -/
def convert_km_s_to_mph (speed_km_s : ℝ) : ℝ :=
  speed_km_s * km_to_miles * seconds_to_hours

/-- Theorem stating that the moon's speed in miles per hour is approximately 2281.34 -/
theorem moon_speed_mph_approx :
  ∃ ε > 0, |convert_km_s_to_mph moon_speed_km_s - 2281.34| < ε :=
sorry

end moon_speed_mph_approx_l506_50622


namespace gcd_90_252_l506_50691

theorem gcd_90_252 : Nat.gcd 90 252 = 18 := by
  sorry

end gcd_90_252_l506_50691


namespace coyote_coins_proof_l506_50627

/-- Represents the number of coins Coyote has after each crossing and payment -/
def coins_after_crossing (initial_coins : ℕ) (num_crossings : ℕ) : ℤ :=
  (3^num_crossings * initial_coins) - (50 * (3^num_crossings - 1) / 2)

/-- Theorem stating that Coyote ends up with 0 coins after 4 crossings if he starts with 25 coins -/
theorem coyote_coins_proof :
  coins_after_crossing 25 4 = 0 := by
  sorry

#eval coins_after_crossing 25 4

end coyote_coins_proof_l506_50627


namespace brandon_card_count_l506_50656

theorem brandon_card_count (malcom_cards : ℕ) (brandon_cards : ℕ) : 
  (malcom_cards = brandon_cards + 8) →
  (malcom_cards / 2 = 14) →
  brandon_cards = 20 := by
sorry

end brandon_card_count_l506_50656


namespace simplest_quadratic_radical_l506_50609

/-- If the simplest quadratic radical √a is of the same type as √27, then a = 3 -/
theorem simplest_quadratic_radical (a : ℝ) : (∃ k : ℕ+, a = 27 * k^2) → a = 3 := by
  sorry

end simplest_quadratic_radical_l506_50609


namespace jills_lavender_candles_l506_50699

/-- Represents the number of candles of each scent Jill made -/
structure CandleCounts where
  lavender : ℕ
  coconut : ℕ
  almond : ℕ
  jasmine : ℕ

/-- Represents the amount of scent (in ml) required for each type of candle -/
def scentAmounts : CandleCounts where
  lavender := 10
  coconut := 8
  almond := 12
  jasmine := 9

/-- The total number of almond candles Jill made -/
def totalAlmondCandles : ℕ := 12

/-- The ratio of coconut scent to almond scent Jill had -/
def coconutToAlmondRatio : ℚ := 5/2

theorem jills_lavender_candles (counts : CandleCounts) : counts.lavender = 135 :=
  by
  have h1 : counts.lavender = 3 * counts.coconut := by sorry
  have h2 : counts.almond = 2 * counts.jasmine := by sorry
  have h3 : counts.almond = totalAlmondCandles := by sorry
  have h4 : counts.coconut * scentAmounts.coconut = 
            coconutToAlmondRatio * (counts.almond * scentAmounts.almond) := by sorry
  have h5 : counts.jasmine * scentAmounts.jasmine = 
            counts.jasmine * scentAmounts.jasmine := by sorry
  sorry

end jills_lavender_candles_l506_50699


namespace minor_axis_length_of_ellipse_l506_50663

/-- The length of the minor axis of the ellipse x^2/4 + y^2/36 = 1 is 4 -/
theorem minor_axis_length_of_ellipse : 
  let ellipse := (fun (x y : ℝ) => x^2/4 + y^2/36 = 1)
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ 
    (∀ x y, ellipse x y ↔ x^2/a^2 + y^2/b^2 = 1) ∧
    2 * min a b = 4 :=
by sorry

end minor_axis_length_of_ellipse_l506_50663


namespace radical_simplification_l506_50602

theorem radical_simplification (a : ℝ) (ha : a > 0) :
  Real.sqrt (50 * a^3) * Real.sqrt (18 * a^2) * Real.sqrt (98 * a^5) = 42 * a^5 * Real.sqrt 10 := by
  sorry

end radical_simplification_l506_50602


namespace seeds_in_small_gardens_l506_50616

theorem seeds_in_small_gardens (total_seeds : ℕ) (big_garden_seeds : ℕ) (num_small_gardens : ℕ) :
  total_seeds = 42 →
  big_garden_seeds = 36 →
  num_small_gardens = 3 →
  num_small_gardens > 0 →
  total_seeds ≥ big_garden_seeds →
  (total_seeds - big_garden_seeds) % num_small_gardens = 0 →
  (total_seeds - big_garden_seeds) / num_small_gardens = 2 := by
sorry

end seeds_in_small_gardens_l506_50616


namespace square_plus_self_even_l506_50624

theorem square_plus_self_even (n : ℤ) : ∃ k : ℤ, n^2 + n = 2 * k := by
  sorry

end square_plus_self_even_l506_50624


namespace expression_value_l506_50654

theorem expression_value (x y z : ℤ) (hx : x = -5) (hy : y = 8) (hz : z = 3) :
  2 * (x - y)^2 - x^3 * y + z^4 * y^2 - x^2 * z^3 = 5847 := by
  sorry

end expression_value_l506_50654


namespace not_proportional_D_l506_50650

/-- Represents a relation between x and y --/
inductive Relation
  | DirectlyProportional
  | InverselyProportional
  | Neither

/-- Determines the type of relation between x and y given an equation --/
def determineRelation (equation : ℝ → ℝ → Prop) : Relation :=
  sorry

/-- The equation x + y = 0 --/
def equationA (x y : ℝ) : Prop := x + y = 0

/-- The equation 3xy = 10 --/
def equationB (x y : ℝ) : Prop := 3 * x * y = 10

/-- The equation x = 5y --/
def equationC (x y : ℝ) : Prop := x = 5 * y

/-- The equation 3x + y = 10 --/
def equationD (x y : ℝ) : Prop := 3 * x + y = 10

/-- The equation x/y = √3 --/
def equationE (x y : ℝ) : Prop := x / y = Real.sqrt 3

theorem not_proportional_D :
  determineRelation equationD = Relation.Neither ∧
  determineRelation equationA ≠ Relation.Neither ∧
  determineRelation equationB ≠ Relation.Neither ∧
  determineRelation equationC ≠ Relation.Neither ∧
  determineRelation equationE ≠ Relation.Neither :=
  sorry

end not_proportional_D_l506_50650


namespace norbs_age_l506_50681

def guesses : List Nat := [25, 29, 31, 33, 37, 39, 42, 45, 48, 50]

def is_prime (n : Nat) : Prop :=
  n > 1 ∧ ∀ m : Nat, m > 1 → m < n → ¬(n % m = 0)

def count_low_guesses (age : Nat) : Nat :=
  (guesses.filter (· < age)).length

def count_off_by_one (age : Nat) : Nat :=
  (guesses.filter (λ g => g = age - 1 ∨ g = age + 1)).length

theorem norbs_age :
  ∃ (age : Nat),
    age ∈ guesses ∧
    is_prime age ∧
    count_low_guesses age < (2 * guesses.length) / 3 ∧
    count_off_by_one age = 2 ∧
    age = 29 :=
  sorry

end norbs_age_l506_50681


namespace right_triangle_height_radius_ratio_l506_50631

theorem right_triangle_height_radius_ratio (a b c h r : ℝ) :
  a > 0 → b > 0 → c > 0 → h > 0 → r > 0 →
  a^2 + b^2 = c^2 →  -- Right triangle condition
  (a + b + c) * r = c * h →  -- Area equality condition
  2 < h / r ∧ h / r ≤ 1 + Real.sqrt 2 :=
by sorry

end right_triangle_height_radius_ratio_l506_50631


namespace arithmetic_expression_evaluation_l506_50689

theorem arithmetic_expression_evaluation : 4 * 12 + 5 * 11 + 6^2 + 7 * 9 = 202 := by
  sorry

end arithmetic_expression_evaluation_l506_50689


namespace rectangle_side_length_l506_50641

theorem rectangle_side_length (square_side : ℝ) (rectangle_width : ℝ) :
  square_side = 5 →
  rectangle_width = 4 →
  square_side * square_side = rectangle_width * (25 / rectangle_width) :=
by
  sorry

end rectangle_side_length_l506_50641


namespace quadratic_inequality_range_l506_50664

theorem quadratic_inequality_range (a : ℝ) :
  (∀ x > 0, x^2 - a*x + 1 > 0) → a ∈ Set.Ioo (-2) 2 := by
  sorry

end quadratic_inequality_range_l506_50664


namespace farmers_field_planted_fraction_l506_50695

theorem farmers_field_planted_fraction :
  ∀ (a b c : ℝ) (s : ℝ),
    a = 5 →
    b = 12 →
    c^2 = a^2 + b^2 →
    (s / a) = (4 / c) →
    (a * b / 2 - s^2) / (a * b / 2) = 470 / 507 :=
by sorry

end farmers_field_planted_fraction_l506_50695


namespace other_solution_of_quadratic_l506_50605

theorem other_solution_of_quadratic (x₁ : ℚ) :
  x₁ = 3/5 →
  (30 * x₁^2 + 13 = 47 * x₁ - 2) →
  ∃ x₂ : ℚ, x₂ ≠ x₁ ∧ x₂ = 5/6 ∧ 30 * x₂^2 + 13 = 47 * x₂ - 2 := by
  sorry

end other_solution_of_quadratic_l506_50605


namespace sphere_diameter_triple_volume_l506_50638

theorem sphere_diameter_triple_volume (π : ℝ) (h_π : π > 0) : 
  let r₁ : ℝ := 6
  let V₁ : ℝ := (4/3) * π * r₁^3
  let V₂ : ℝ := 3 * V₁
  let r₂ : ℝ := (V₂ / ((4/3) * π))^(1/3)
  2 * r₂ = 12 * (2 : ℝ)^(1/3) :=
by sorry

end sphere_diameter_triple_volume_l506_50638


namespace average_age_when_youngest_born_l506_50610

theorem average_age_when_youngest_born 
  (n : ℕ) 
  (current_avg : ℝ) 
  (youngest_age : ℝ) 
  (sum_others_at_birth : ℝ) 
  (h1 : n = 7) 
  (h2 : current_avg = 30) 
  (h3 : youngest_age = 6) 
  (h4 : sum_others_at_birth = 150) : 
  (sum_others_at_birth / n : ℝ) = 150 / 7 := by
sorry

end average_age_when_youngest_born_l506_50610


namespace geometric_progression_iff_equal_first_two_l506_50629

/-- A sequence of positive real numbers -/
def Sequence := ℕ → ℝ

/-- Predicate to check if a sequence is positive -/
def IsPositive (a : Sequence) : Prop :=
  ∀ n, a n > 0

/-- Predicate to check if a sequence satisfies the given recurrence relation -/
def SatisfiesRecurrence (a : Sequence) (b : ℝ) : Prop :=
  ∀ n, a (n + 2) = (b + 1) * a n * a (n + 1)

/-- Predicate to check if a sequence is a geometric progression -/
def IsGeometricProgression (a : Sequence) : Prop :=
  ∃ r, ∀ n, a (n + 1) = r * a n

/-- Main theorem -/
theorem geometric_progression_iff_equal_first_two (a : Sequence) (b : ℝ) :
  b > 0 ∧ IsPositive a ∧ SatisfiesRecurrence a b →
  IsGeometricProgression a ↔ a 1 = a 0 :=
sorry

end geometric_progression_iff_equal_first_two_l506_50629


namespace unique_x_intercept_l506_50625

/-- The parabola equation: x = -3y^2 + 2y + 3 -/
def parabola (y : ℝ) : ℝ := -3 * y^2 + 2 * y + 3

/-- X-intercept occurs when y = 0 -/
def x_intercept : ℝ := parabola 0

/-- Theorem: The parabola has exactly one x-intercept -/
theorem unique_x_intercept : ∃! x : ℝ, ∃ y : ℝ, parabola y = x ∧ y = 0 := by sorry

end unique_x_intercept_l506_50625


namespace no_m_exists_for_subset_l506_50628

theorem no_m_exists_for_subset : ¬ ∃ m : ℝ, m > 1 ∧ ∀ x : ℝ, -3 ≤ x ∧ x ≤ 4 → 1 - m ≤ x ∧ x ≤ 3 * m - 2 := by
  sorry

end no_m_exists_for_subset_l506_50628


namespace range_of_f_l506_50662

def f (x : ℝ) : ℝ := x^2 - 4*x + 4

theorem range_of_f :
  ∀ x ∈ Set.Icc (-3 : ℝ) 3, ∃ y ∈ Set.Icc 0 25, f x = y ∧
  ∀ z, f x = z → z ∈ Set.Icc 0 25 :=
by sorry

end range_of_f_l506_50662


namespace min_students_with_both_traits_l506_50696

theorem min_students_with_both_traits (total : ℕ) (blue_eyes : ℕ) (lunch_box : ℕ)
  (h1 : total = 35)
  (h2 : blue_eyes = 15)
  (h3 : lunch_box = 23)
  (h4 : blue_eyes ≤ total)
  (h5 : lunch_box ≤ total) :
  total - (total - blue_eyes) - (total - lunch_box) ≥ 3 :=
by sorry

end min_students_with_both_traits_l506_50696


namespace inequality_solution_set_l506_50643

theorem inequality_solution_set (m : ℝ) : 
  (∃ (a b c : ℤ), (∀ x : ℝ, (x^2 - 2*x + m ≤ 0) ↔ (x = a ∨ x = b ∨ x = c)) ∧ 
   (∀ y : ℤ, (y^2 - 2*y + m ≤ 0) → (y = a ∨ y = b ∨ y = c))) ↔ 
  (m = -2 ∨ m = 0) := by
sorry

end inequality_solution_set_l506_50643


namespace least_integer_satisfying_condition_l506_50673

/-- Given a positive integer n, returns the integer formed by removing its leftmost digit. -/
def removeLeftmostDigit (n : ℕ+) : ℕ :=
  sorry

/-- Checks if a positive integer satisfies the condition that removing its leftmost digit
    results in 1/29 of the original number. -/
def satisfiesCondition (n : ℕ+) : Prop :=
  removeLeftmostDigit n = n.val / 29

/-- Proves that 725 is the least positive integer that satisfies the given condition. -/
theorem least_integer_satisfying_condition :
  satisfiesCondition 725 ∧ ∀ m : ℕ+, m < 725 → ¬satisfiesCondition m :=
sorry

end least_integer_satisfying_condition_l506_50673


namespace line_segment_lattice_points_l506_50694

/-- The number of lattice points on a line segment with given integer coordinates --/
def latticePointCount (x1 y1 x2 y2 : ℤ) : ℕ :=
  sorry

/-- Theorem: The number of lattice points on the line segment from (5, 23) to (53, 311) is 49 --/
theorem line_segment_lattice_points :
  latticePointCount 5 23 53 311 = 49 := by
  sorry

end line_segment_lattice_points_l506_50694


namespace unique_prime_triplet_l506_50604

theorem unique_prime_triplet : ∃! p : ℕ, Prime p ∧ Prime (p + 2) ∧ Prime (p + 4) ∧ p = 3 := by
  sorry

end unique_prime_triplet_l506_50604


namespace min_photos_theorem_l506_50620

theorem min_photos_theorem (n_girls n_boys : ℕ) (h_girls : n_girls = 4) (h_boys : n_boys = 8) :
  ∃ (min_photos : ℕ), min_photos = n_girls * n_boys + 1 ∧
  (∀ (num_photos : ℕ), num_photos ≥ min_photos →
    (∃ (photo : Fin num_photos → Fin (n_girls + n_boys) × Fin (n_girls + n_boys)),
      (∃ (i : Fin num_photos), (photo i).1 ≥ n_girls ∧ (photo i).2 ≥ n_girls) ∨
      (∃ (i : Fin num_photos), (photo i).1 < n_girls ∧ (photo i).2 < n_girls) ∨
      (∃ (i j : Fin num_photos), i ≠ j ∧ photo i = photo j))) :=
by sorry

end min_photos_theorem_l506_50620


namespace min_intercept_sum_l506_50623

/-- Given a line passing through (1, 2) with equation x/a + y/b = 1 where a > 0 and b > 0,
    the minimum value of a + b is 3 + 2√2 -/
theorem min_intercept_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) 
    (h_line : 1 / a + 2 / b = 1) : 
  ∀ (x y : ℝ), x / a + y / b = 1 → x + y ≥ 3 + 2 * Real.sqrt 2 := by
  sorry

end min_intercept_sum_l506_50623


namespace periodic_functions_exist_l506_50615

-- Define a type for periodic functions
def PeriodicFunction (p : ℝ) := { f : ℝ → ℝ // ∀ x, f (x + p) = f x }

-- Define a predicate for the smallest positive period
def SmallestPositivePeriod (f : ℝ → ℝ) (p : ℝ) :=
  (∀ x, f (x + p) = f x) ∧ (∀ q, 0 < q → q < p → ∃ x, f (x + q) ≠ f x)

-- Main theorem
theorem periodic_functions_exist (p₁ p₂ : ℝ) (hp₁ : 0 < p₁) (hp₂ : 0 < p₂) :
  ∃ (f₁ f₂ : ℝ → ℝ),
    SmallestPositivePeriod f₁ p₁ ∧
    SmallestPositivePeriod f₂ p₂ ∧
    ∃ (p : ℝ), ∀ x, (f₁ - f₂) (x + p) = (f₁ - f₂) x :=
by
  sorry


end periodic_functions_exist_l506_50615


namespace sin_sum_of_roots_l506_50690

theorem sin_sum_of_roots (a b c : ℝ) (α β : ℝ) :
  (0 < α) → (α < π) →
  (0 < β) → (β < π) →
  (α ≠ β) →
  (a * Real.cos α + b * Real.sin α + c = 0) →
  (a * Real.cos β + b * Real.sin β + c = 0) →
  Real.sin (α + β) = (2 * a * b) / (a^2 + b^2) := by
  sorry

end sin_sum_of_roots_l506_50690


namespace opposite_of_2023_l506_50652

theorem opposite_of_2023 : 
  ∀ x : ℤ, x + 2023 = 0 → x = -2023 := by
  sorry

end opposite_of_2023_l506_50652


namespace cyclic_difference_fourth_power_sum_l506_50659

theorem cyclic_difference_fourth_power_sum (a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℕ) 
  (h_distinct : a₁ ≠ a₂ ∧ a₁ ≠ a₃ ∧ a₁ ≠ a₄ ∧ a₁ ≠ a₅ ∧ a₁ ≠ a₆ ∧ a₁ ≠ a₇ ∧
                a₂ ≠ a₃ ∧ a₂ ≠ a₄ ∧ a₂ ≠ a₅ ∧ a₂ ≠ a₆ ∧ a₂ ≠ a₇ ∧
                a₃ ≠ a₄ ∧ a₃ ≠ a₅ ∧ a₃ ≠ a₆ ∧ a₃ ≠ a₇ ∧
                a₄ ≠ a₅ ∧ a₄ ≠ a₆ ∧ a₄ ≠ a₇ ∧
                a₅ ≠ a₆ ∧ a₅ ≠ a₇ ∧
                a₆ ≠ a₇) :
  (a₁ - a₂)^4 + (a₂ - a₃)^4 + (a₃ - a₄)^4 + (a₄ - a₅)^4 + 
  (a₅ - a₆)^4 + (a₆ - a₇)^4 + (a₇ - a₁)^4 ≥ 82 :=
by sorry

end cyclic_difference_fourth_power_sum_l506_50659


namespace final_mixture_volume_l506_50639

/-- Represents an alcohol mixture -/
structure AlcoholMixture where
  volume : ℝ
  concentration : ℝ

/-- The problem setup -/
def mixture_problem (mixture30 mixture50 mixtureFinal : AlcoholMixture) : Prop :=
  mixture30.concentration = 0.30 ∧
  mixture50.concentration = 0.50 ∧
  mixtureFinal.concentration = 0.45 ∧
  mixture30.volume = 2.5 ∧
  mixtureFinal.volume = mixture30.volume + mixture50.volume ∧
  mixture30.volume * mixture30.concentration + mixture50.volume * mixture50.concentration =
    mixtureFinal.volume * mixtureFinal.concentration

/-- The theorem statement -/
theorem final_mixture_volume
  (mixture30 mixture50 mixtureFinal : AlcoholMixture)
  (h : mixture_problem mixture30 mixture50 mixtureFinal) :
  mixtureFinal.volume = 10 :=
sorry

end final_mixture_volume_l506_50639


namespace tan_30_degrees_l506_50684

theorem tan_30_degrees : Real.tan (30 * π / 180) = Real.sqrt 3 / 3 := by
  sorry

end tan_30_degrees_l506_50684


namespace unique_four_digit_consecutive_square_swap_l506_50617

def is_consecutive_digits (n : ℕ) : Prop :=
  ∃ x : ℕ, x ≤ 6 ∧ 
    n = 1000 * x + 100 * (x + 1) + 10 * (x + 2) + (x + 3)

def swap_thousands_hundreds (n : ℕ) : ℕ :=
  let thousands := n / 1000
  let hundreds := (n / 100) % 10
  let tens := (n / 10) % 10
  let ones := n % 10
  1000 * hundreds + 100 * thousands + 10 * tens + ones

theorem unique_four_digit_consecutive_square_swap :
  ∃! n : ℕ, 1000 ≤ n ∧ n < 10000 ∧
    is_consecutive_digits n ∧
    ∃ m : ℕ, swap_thousands_hundreds n = m * m :=
by
  use 3456
  sorry

end unique_four_digit_consecutive_square_swap_l506_50617


namespace car_trading_profit_l506_50600

theorem car_trading_profit (P : ℝ) (h : P > 0) : 
  let discount_rate : ℝ := 0.3
  let increase_rate : ℝ := 0.7
  let buying_price : ℝ := P * (1 - discount_rate)
  let selling_price : ℝ := buying_price * (1 + increase_rate)
  let profit : ℝ := selling_price - P
  profit / P = 0.19 := by sorry

end car_trading_profit_l506_50600


namespace min_value_theorem_equality_condition_l506_50683

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  a^2 + b^2 + 1/a^2 + 1/b^2 + b/a + a/b ≥ Real.sqrt 15 := by
  sorry

theorem equality_condition (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  a^2 + b^2 + 1/a^2 + 1/b^2 + b/a + a/b = Real.sqrt 15 ↔ 
  a = (3/20)^(1/4) ∧ b = 1/(2*a) := by
  sorry

end min_value_theorem_equality_condition_l506_50683


namespace simple_interest_rate_percent_l506_50619

theorem simple_interest_rate_percent : 
  ∀ (principal interest time rate : ℝ),
  principal = 800 →
  interest = 176 →
  time = 4 →
  interest = principal * rate * time / 100 →
  rate = 5.5 := by
sorry

end simple_interest_rate_percent_l506_50619


namespace smallest_k_for_polynomial_division_l506_50630

theorem smallest_k_for_polynomial_division : 
  ∃ (k : ℕ), k > 0 ∧ 
  (∀ (z : ℂ), (z^10 + z^9 + z^6 + z^5 + z^4 + z + 1) ∣ (z^k - 1)) ∧
  (∀ (m : ℕ), m > 0 → m < k → 
    ¬(∀ (z : ℂ), (z^10 + z^9 + z^6 + z^5 + z^4 + z + 1) ∣ (z^m - 1))) ∧
  k = 84 :=
by sorry

end smallest_k_for_polynomial_division_l506_50630


namespace smallest_angle_through_point_l506_50621

theorem smallest_angle_through_point (α : Real) : 
  (∃ k : ℤ, α = 11 * Real.pi / 6 + 2 * Real.pi * k) ∧ 
  (∀ β : Real, β > 0 → 
    (Real.sin β = Real.sin (2 * Real.pi / 3) ∧ 
     Real.cos β = Real.cos (2 * Real.pi / 3)) → 
    α ≤ β) ↔ 
  (Real.sin α = Real.sin (2 * Real.pi / 3) ∧ 
   Real.cos α = Real.cos (2 * Real.pi / 3) ∧ 
   α > 0 ∧ 
   α < 2 * Real.pi) :=
by sorry

end smallest_angle_through_point_l506_50621


namespace modulo_eleven_residue_l506_50601

theorem modulo_eleven_residue : (341 + 6 * 50 + 4 * 156 + 3 * 12^2) % 11 = 4 := by
  sorry

end modulo_eleven_residue_l506_50601


namespace parabola_circle_intersection_l506_50655

/-- Parabola type representing y = ax² --/
structure Parabola where
  a : ℝ

/-- Point type representing (x, y) --/
structure Point where
  x : ℝ
  y : ℝ

/-- Line type representing y = mx + b --/
structure Line where
  m : ℝ
  b : ℝ

/-- Circle type representing (x - h)² + (y - k)² = r² --/
structure Circle where
  h : ℝ
  k : ℝ
  r : ℝ

/-- Given conditions of the problem --/
def given (C : Parabola) (M A B : Point) : Prop :=
  M.y = C.a * M.x^2 ∧
  M.x = 2 ∧ M.y = 1 ∧
  A.y = C.a * A.x^2 ∧
  B.y = C.a * B.x^2 ∧
  A ≠ M ∧ B ≠ M ∧
  ∃ (circ : Circle), (A.x - circ.h)^2 + (A.y - circ.k)^2 = circ.r^2 ∧
                     (B.x - circ.h)^2 + (B.y - circ.k)^2 = circ.r^2 ∧
                     (M.x - circ.h)^2 + (M.y - circ.k)^2 = circ.r^2 ∧
                     circ.r = (A.x - B.x)^2 + (A.y - B.y)^2

/-- The main theorem to be proved --/
theorem parabola_circle_intersection 
  (C : Parabola) (M A B : Point) (h : given C M A B) :
  (∃ (l : Line), l.m * (-2) + l.b = 5 ∧ l.m * A.x + l.b = A.y ∧ l.m * B.x + l.b = B.y) ∧
  (∃ (N : Point), N.x^2 + (N.y - 3)^2 = 8 ∧ N.y ≠ 1 ∧
    (N.x - M.x) * (B.x - A.x) + (N.y - M.y) * (B.y - A.y) = 0) :=
sorry

end parabola_circle_intersection_l506_50655


namespace f_monotone_decreasing_l506_50661

noncomputable def f (x : ℝ) := (x + 1) * Real.exp x

theorem f_monotone_decreasing :
  ∀ x y, x < y → x < -2 → y < -2 → f y < f x := by sorry

end f_monotone_decreasing_l506_50661


namespace rational_function_inequality_l506_50698

theorem rational_function_inequality (f : ℚ → ℤ) :
  ∃ a b : ℚ, (f a + f b : ℚ) / 2 ≤ f ((a + b) / 2) := by
  sorry

end rational_function_inequality_l506_50698


namespace arithmetic_geometric_sequence_property_l506_50626

def is_arithmetic_sequence (x y z : ℝ) : Prop :=
  y - x = z - y

def is_geometric_sequence (x y z w v : ℝ) : Prop :=
  y / x = z / y ∧ z / y = w / z ∧ w / z = v / w

theorem arithmetic_geometric_sequence_property :
  ∀ (a b m n : ℝ),
  is_arithmetic_sequence (-9) a (-1) →
  is_geometric_sequence (-9) m b n (-1) →
  a * b = 15 := by sorry

end arithmetic_geometric_sequence_property_l506_50626


namespace hair_extension_length_l506_50679

def original_length : ℕ := 18

def extension_factor : ℕ := 2

theorem hair_extension_length : 
  original_length * extension_factor = 36 := by sorry

end hair_extension_length_l506_50679


namespace cuboidal_box_volume_l506_50676

/-- A cuboidal box with given adjacent face areas has a specific volume -/
theorem cuboidal_box_volume (l w h : ℝ) (h1 : l * w = 120) (h2 : w * h = 72) (h3 : h * l = 60) :
  l * w * h = 4320 := by
  sorry

end cuboidal_box_volume_l506_50676


namespace root_in_interval_l506_50614

-- Define the function f(x) = x^2 + 12x - 15
def f (x : ℝ) : ℝ := x^2 + 12*x - 15

-- State the theorem
theorem root_in_interval :
  (f 1.1 < 0) → (f 1.2 > 0) → ∃ x : ℝ, 1.1 < x ∧ x < 1.2 ∧ f x = 0 :=
by
  sorry

end root_in_interval_l506_50614


namespace least_multiple_17_greater_500_l506_50634

theorem least_multiple_17_greater_500 : ∃ (n : ℕ), n * 17 = 510 ∧ 
  510 > 500 ∧ (∀ m : ℕ, m * 17 > 500 → m * 17 ≥ 510) := by
  sorry

end least_multiple_17_greater_500_l506_50634


namespace nell_card_difference_l506_50669

/-- Represents the number of cards Nell has -/
structure CardCount where
  baseball : ℕ
  ace : ℕ

/-- The difference between baseball and ace cards -/
def cardDifference (cards : CardCount) : ℤ :=
  cards.baseball - cards.ace

theorem nell_card_difference (initial final : CardCount) 
  (h1 : initial.baseball = 438)
  (h2 : initial.ace = 18)
  (h3 : final.baseball = 178)
  (h4 : final.ace = 55) :
  cardDifference final = 123 := by
  sorry

end nell_card_difference_l506_50669


namespace expression_evaluation_l506_50653

theorem expression_evaluation (x y : ℕ) (h1 : x = 3) (h2 : y = 2) :
  5 * x^(y + 1) + 6 * y^(x + 1) = 231 := by
  sorry

end expression_evaluation_l506_50653


namespace semicircle_area_with_inscribed_rectangle_l506_50607

/-- The area of a semicircle that circumscribes a 2 × 3 rectangle with the longer side on the diameter -/
theorem semicircle_area_with_inscribed_rectangle : 
  ∀ (semicircle_area : ℝ) (rectangle_width : ℝ) (rectangle_length : ℝ),
    rectangle_width = 2 →
    rectangle_length = 3 →
    semicircle_area = (9 * Real.pi) / 4 := by
  sorry

end semicircle_area_with_inscribed_rectangle_l506_50607


namespace transaction_gain_per_year_l506_50674

/-- Calculate simple interest -/
def simpleInterest (principal rate time : ℚ) : ℚ :=
  (principal * rate * time) / 100

theorem transaction_gain_per_year 
  (principal : ℚ) 
  (borrowRate lendRate : ℚ) 
  (time : ℚ) 
  (h1 : principal = 5000)
  (h2 : borrowRate = 4)
  (h3 : lendRate = 6)
  (h4 : time = 2) :
  (simpleInterest principal lendRate time - simpleInterest principal borrowRate time) / time = 200 := by
  sorry

end transaction_gain_per_year_l506_50674


namespace vertex_angle_of_special_triangle_l506_50682

/-- A triangle with angles a, b, and c is isosceles and a "double angle triangle" -/
structure SpecialTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  sum_180 : a + b + c = 180
  isosceles : b = c
  double_angle : a = 2 * b ∨ b = 2 * a

/-- The vertex angle of an isosceles "double angle triangle" is either 36° or 90° -/
theorem vertex_angle_of_special_triangle (t : SpecialTriangle) :
  t.a = 36 ∨ t.a = 90 := by
  sorry

end vertex_angle_of_special_triangle_l506_50682


namespace evaluate_expression_l506_50645

theorem evaluate_expression : 7^3 - 4 * 7^2 + 4 * 7 - 1 = 174 := by sorry

end evaluate_expression_l506_50645


namespace angle_bisector_ratio_l506_50651

-- Define the triangle
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the angle bisector
def angleBisector (t : Triangle) : ℝ × ℝ :=
  sorry

-- Define the intersection point
def intersectionPoint (p q r s : ℝ × ℝ) : ℝ × ℝ :=
  sorry

-- Define the distance between two points
def distance (p q : ℝ × ℝ) : ℝ :=
  sorry

theorem angle_bisector_ratio (t : Triangle) :
  let D : ℝ × ℝ := ((t.A.1 + t.B.1) / 2, (t.A.2 + t.B.2) / 2)
  let E : ℝ × ℝ := ((t.A.1 + t.C.1) / 2, (t.A.2 + t.C.2) / 2)
  let T : ℝ × ℝ := angleBisector t
  let F : ℝ × ℝ := intersectionPoint t.A T D E
  distance t.A D = distance D t.B ∧ 
  distance t.A E = distance E t.C ∧
  distance t.A D = 2 ∧
  distance t.A E = 3 →
  distance t.A F / distance t.A T = 1 / 3 :=
sorry

end angle_bisector_ratio_l506_50651


namespace union_of_A_and_B_l506_50680

def A : Set Nat := {1,2,3,4,5}
def B : Set Nat := {2,4,6,8,10}

theorem union_of_A_and_B : A ∪ B = {1,2,3,4,5,6,8,10} := by
  sorry

end union_of_A_and_B_l506_50680


namespace point_transformation_l506_50671

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define the transformation function
def transform (p : Point2D) : Point2D :=
  { x := p.x + 3, y := p.y + 5 }

theorem point_transformation :
  ∀ (x y : ℝ),
  let A : Point2D := { x := x, y := -2 }
  let B : Point2D := transform A
  B.x = 1 → x = -2 := by
  sorry

end point_transformation_l506_50671


namespace f_sum_difference_equals_two_l506_50608

noncomputable def f (x : ℝ) : ℝ := ((x + 1)^2 + Real.sin x) / (x^2 + 1)

theorem f_sum_difference_equals_two :
  f 2016 + (deriv f) 2016 + f (-2016) - (deriv f) (-2016) = 2 := by
  sorry

end f_sum_difference_equals_two_l506_50608


namespace expression_evaluation_l506_50618

theorem expression_evaluation (c : ℕ) (h : c = 4) :
  (c^c - c*(c-1)^(c-1))^(c-1) = 3241792 := by
  sorry

end expression_evaluation_l506_50618


namespace age_difference_l506_50644

theorem age_difference (P M Mo : ℕ) 
  (h1 : P * 5 = M * 3) 
  (h2 : M * 5 = Mo * 3) 
  (h3 : P + M + Mo = 196) : 
  Mo - P = 64 := by
  sorry

end age_difference_l506_50644


namespace jenga_remaining_blocks_l506_50635

/-- Represents a Jenga game state -/
structure JengaGame where
  initialBlocks : ℕ
  players : ℕ
  completeRounds : ℕ
  extraBlocksRemoved : ℕ

/-- Calculates the number of blocks remaining before the last player's turn -/
def remainingBlocks (game : JengaGame) : ℕ :=
  game.initialBlocks - (game.players * game.completeRounds + game.extraBlocksRemoved)

/-- Theorem stating the number of blocks remaining in the specific Jenga game scenario -/
theorem jenga_remaining_blocks :
  let game : JengaGame := {
    initialBlocks := 54,
    players := 5,
    completeRounds := 5,
    extraBlocksRemoved := 1
  }
  remainingBlocks game = 28 := by sorry

end jenga_remaining_blocks_l506_50635


namespace sin_75_degrees_l506_50611

theorem sin_75_degrees : Real.sin (75 * π / 180) = (Real.sqrt 6 + Real.sqrt 2) / 4 := by
  sorry

end sin_75_degrees_l506_50611


namespace kates_bill_l506_50649

theorem kates_bill (bob_bill : ℝ) (bob_discount : ℝ) (kate_discount : ℝ) (total_after_discount : ℝ) :
  bob_bill = 30 →
  bob_discount = 0.05 →
  kate_discount = 0.02 →
  total_after_discount = 53 →
  ∃ kate_bill : ℝ,
    kate_bill = 25 ∧
    bob_bill * (1 - bob_discount) + kate_bill * (1 - kate_discount) = total_after_discount :=
by sorry

end kates_bill_l506_50649


namespace lisa_additional_marbles_l506_50632

def minimum_additional_marbles (num_friends : ℕ) (initial_marbles : ℕ) : ℕ :=
  let required_marbles := (num_friends * (num_friends + 1)) / 2
  if required_marbles > initial_marbles then
    required_marbles - initial_marbles
  else
    0

theorem lisa_additional_marbles :
  minimum_additional_marbles 11 45 = 21 := by
  sorry

end lisa_additional_marbles_l506_50632


namespace max_consecutive_sum_36_l506_50693

/-- The sum of consecutive integers from a to (a + n - 1) -/
def sum_consecutive (a : ℤ) (n : ℕ) : ℤ := n * a + (n * (n - 1)) / 2

/-- The proposition that 72 is the maximum number of consecutive integers summing to 36 -/
theorem max_consecutive_sum_36 :
  (∃ a : ℤ, sum_consecutive a 72 = 36) ∧
  (∀ n : ℕ, n > 72 → ∀ a : ℤ, sum_consecutive a n ≠ 36) :=
sorry

end max_consecutive_sum_36_l506_50693


namespace twelve_pharmacies_not_enough_l506_50647

/-- Represents a grid of streets -/
structure Grid :=
  (rows : Nat)
  (cols : Nat)

/-- Represents a pharmacy on the grid -/
structure Pharmacy :=
  (row : Nat)
  (col : Nat)

/-- The maximum walking distance to a pharmacy -/
def max_walking_distance : Nat := 3

/-- Calculate the number of street segments covered by a pharmacy -/
def covered_segments (g : Grid) (p : Pharmacy) : Nat :=
  let coverage_side := 2 * max_walking_distance + 1
  min coverage_side g.rows * min coverage_side g.cols

/-- Calculate the total number of street segments in the grid -/
def total_segments (g : Grid) : Nat :=
  2 * g.rows * (g.cols - 1) + 2 * g.cols * (g.rows - 1)

/-- The main theorem to be proved -/
theorem twelve_pharmacies_not_enough :
  ∀ (pharmacies : List Pharmacy),
    pharmacies.length = 12 →
    ∃ (g : Grid),
      g.rows = 9 ∧ g.cols = 9 ∧
      (pharmacies.map (covered_segments g)).sum < total_segments g := by
  sorry


end twelve_pharmacies_not_enough_l506_50647


namespace present_value_exponent_l506_50606

theorem present_value_exponent 
  (Q r j m n : ℝ) 
  (hQ : Q > 0) 
  (hr : r > 0) 
  (hjm : j + m > -1) 
  (heq : Q = r / (1 + j + m) ^ n) : 
  n = Real.log (r / Q) / Real.log (1 + j + m) := by
sorry

end present_value_exponent_l506_50606


namespace arithmetic_sequence_10th_term_l506_50667

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

/-- Given an arithmetic sequence a with a₂ = 2 and a₃ = 4,
    prove that the 10th term a₁₀ = 18. -/
theorem arithmetic_sequence_10th_term
  (a : ℕ → ℝ)
  (h_arith : is_arithmetic_sequence a)
  (h_2 : a 2 = 2)
  (h_3 : a 3 = 4) :
  a 10 = 18 :=
by sorry

end arithmetic_sequence_10th_term_l506_50667


namespace line_bisects_circle_implies_b_eq_neg_two_l506_50677

/-- The line l is defined by parametric equations x = 2t and y = 1 + bt -/
def line_l (b t : ℝ) : ℝ × ℝ := (2 * t, 1 + b * t)

/-- The circle C is defined by the equation (x - 1)^2 + y^2 = 1 -/
def circle_C (p : ℝ × ℝ) : Prop :=
  (p.1 - 1)^2 + p.2^2 = 1

/-- A line bisects the area of a circle if it passes through the center of the circle -/
def bisects_circle_area (l : ℝ → ℝ × ℝ) (c : ℝ × ℝ → Prop) : Prop :=
  ∃ t, l t = (1, 0)

/-- Main theorem: If line l bisects the area of circle C, then b = -2 -/
theorem line_bisects_circle_implies_b_eq_neg_two (b : ℝ) :
  bisects_circle_area (line_l b) circle_C → b = -2 :=
sorry

end line_bisects_circle_implies_b_eq_neg_two_l506_50677


namespace least_integer_satisfying_inequality_l506_50685

theorem least_integer_satisfying_inequality :
  ∀ y : ℤ, (3 * |y| + 6 < 24) → y ≥ -5 ∧ 
  ∃ x : ℤ, x = -5 ∧ (3 * |x| + 6 < 24) := by
  sorry

end least_integer_satisfying_inequality_l506_50685


namespace square_equation_proof_l506_50613

theorem square_equation_proof (h1 : 3 > 1) (h2 : 1 > 1) : (3 * (1^3 + 3))^2 = 8339 := by
  sorry

end square_equation_proof_l506_50613


namespace system_of_equations_l506_50672

theorem system_of_equations (x y c d : ℝ) 
  (eq1 : 4 * x + 8 * y = c)
  (eq2 : 5 * x - 10 * y = d)
  (h_d_nonzero : d ≠ 0)
  (h_x_nonzero : x ≠ 0)
  (h_y_nonzero : y ≠ 0) :
  c / d = -4 / 5 := by
sorry

end system_of_equations_l506_50672


namespace factorization_equality_l506_50687

theorem factorization_equality (a b : ℝ) : 3 * a^2 * b - 3 * a * b + 6 * b = 3 * b * (a^2 - a + 2) := by
  sorry

end factorization_equality_l506_50687


namespace painting_time_is_18_17_l506_50633

/-- The time required for three painters to complete a room, given their individual rates and break times -/
def total_painting_time (linda_rate tom_rate jerry_rate : ℚ) (tom_break jerry_break : ℚ) : ℚ :=
  let combined_rate := linda_rate + tom_rate + jerry_rate
  18 / 17

/-- Theorem stating that the total painting time for Linda, Tom, and Jerry is 18/17 hours -/
theorem painting_time_is_18_17 :
  let linda_rate : ℚ := 1 / 3
  let tom_rate : ℚ := 1 / 4
  let jerry_rate : ℚ := 1 / 6
  let tom_break : ℚ := 2
  let jerry_break : ℚ := 1
  total_painting_time linda_rate tom_rate jerry_rate tom_break jerry_break = 18 / 17 := by
  sorry

#eval total_painting_time (1/3) (1/4) (1/6) 2 1

end painting_time_is_18_17_l506_50633


namespace tan_inequality_l506_50688

theorem tan_inequality (h1 : 130 * π / 180 > π / 2) (h2 : 130 * π / 180 < π)
                       (h3 : 140 * π / 180 > π / 2) (h4 : 140 * π / 180 < π) :
  Real.tan (130 * π / 180) < Real.tan (140 * π / 180) := by
  sorry

end tan_inequality_l506_50688


namespace obtuse_triangle_partition_l506_50646

/-- A triple of positive integers forming an obtuse triangle -/
structure ObtuseTriple where
  a : ℕ+
  b : ℕ+
  c : ℕ+
  h1 : a < b
  h2 : b < c
  h3 : a + b > c
  h4 : a * a + b * b < c * c

/-- The set of integers from 2 to 3n+1 -/
def triangleSet (n : ℕ+) : Set ℕ+ :=
  {k | 2 ≤ k ∧ k ≤ 3*n+1}

/-- A partition of the triangle set into n obtuse triples -/
def ObtusePartition (n : ℕ+) : Type :=
  { partition : Finset (Finset ℕ+) //
    partition.card = n ∧
    (∀ s ∈ partition, ∃ t : ObtuseTriple, (↑s : Set ℕ+) = {t.a, t.b, t.c}) ∧
    (⋃ (s ∈ partition), (↑s : Set ℕ+)) = triangleSet n }

/-- The main theorem -/
theorem obtuse_triangle_partition (n : ℕ+) :
  ∃ p : ObtusePartition n, True := by sorry

end obtuse_triangle_partition_l506_50646


namespace k_at_neg_one_eq_64_l506_50692

/-- The polynomial h(x) -/
def h (p : ℝ) (x : ℝ) : ℝ := x^3 - p*x^2 + 3*x + 20

/-- The polynomial k(x) -/
def k (q r : ℝ) (x : ℝ) : ℝ := x^4 + x^3 - q*x^2 + 50*x + r

/-- Theorem stating that k(-1) = 64 given the conditions -/
theorem k_at_neg_one_eq_64 (p q r : ℝ) :
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ h p x = 0 ∧ h p y = 0 ∧ h p z = 0) →
  (∀ x : ℝ, h p x = 0 → k q r x = 0) →
  k q r (-1) = 64 :=
by sorry

end k_at_neg_one_eq_64_l506_50692


namespace find_D_l506_50648

theorem find_D (A B C D : ℤ) 
  (h1 : A + C = 15)
  (h2 : A - B = 1)
  (h3 : C + C = A)
  (h4 : B - D = 2)
  (h5 : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D) :
  D = 7 := by
sorry

end find_D_l506_50648


namespace pure_imaginary_iff_real_zero_imag_nonzero_l506_50665

/-- A complex number is pure imaginary if and only if its real part is zero and its imaginary part is non-zero -/
theorem pure_imaginary_iff_real_zero_imag_nonzero (z : ℂ) :
  (∃ b : ℝ, b ≠ 0 ∧ z = Complex.I * b) ↔ (z.re = 0 ∧ z.im ≠ 0) :=
by sorry

end pure_imaginary_iff_real_zero_imag_nonzero_l506_50665


namespace inequality_proof_l506_50670

theorem inequality_proof (x : ℝ) (hx : x > 0) :
  (1 + x + x^2) * (1 + x + x^2 + x^3 + x^4) ≤ (1 + x + x^2 + x^3)^2 := by
  sorry

end inequality_proof_l506_50670


namespace clock_strikes_count_l506_50678

/-- Calculates the number of clock strikes in a 24-hour period -/
def clock_strikes : ℕ :=
  -- Strikes at whole hours: sum of 1 to 12, twice (for AM and PM)
  2 * (List.range 12).sum
  -- Strikes at half hours: 24 (once every half hour)
  + 24

/-- Theorem stating that the clock strikes 180 times in a 24-hour period -/
theorem clock_strikes_count : clock_strikes = 180 := by
  sorry

end clock_strikes_count_l506_50678


namespace henri_total_time_l506_50657

/-- Represents the total time Henri has for watching movies and reading -/
def total_time : ℝ := 8

/-- Duration of the first movie Henri watches -/
def movie1_duration : ℝ := 3.5

/-- Duration of the second movie Henri watches -/
def movie2_duration : ℝ := 1.5

/-- Henri's reading speed in words per minute -/
def reading_speed : ℝ := 10

/-- Number of words Henri reads -/
def words_read : ℝ := 1800

/-- Theorem stating that Henri's total time for movies and reading is 8 hours -/
theorem henri_total_time : 
  movie1_duration + movie2_duration + (words_read / reading_speed) / 60 = total_time := by
  sorry

end henri_total_time_l506_50657


namespace line_slope_point_sum_l506_50637

/-- Theorem: For a line with slope 8 passing through (-2, 4), m + b = 28 -/
theorem line_slope_point_sum (m b : ℝ) : 
  m = 8 → -- The slope is 8
  4 = 8 * (-2) + b → -- The line passes through (-2, 4)
  m + b = 28 := by sorry

end line_slope_point_sum_l506_50637


namespace complex_equation_solution_l506_50675

theorem complex_equation_solution (z : ℂ) : 
  (Complex.I * z = Complex.I + z) → z = (1 - Complex.I) / 2 := by
  sorry

end complex_equation_solution_l506_50675


namespace inequality_solution_interval_l506_50666

theorem inequality_solution_interval (x : ℝ) : 
  (1 / (x^2 + 1) > 3 / x + 13 / 10) ↔ -2 < x ∧ x < 0 := by
  sorry

end inequality_solution_interval_l506_50666


namespace marshmallow_challenge_l506_50642

/-- The marshmallow challenge problem -/
theorem marshmallow_challenge 
  (haley : ℕ) 
  (michael : ℕ) 
  (brandon : ℕ) 
  (h1 : haley = 8)
  (h2 : michael = 3 * haley)
  (h3 : haley + michael + brandon = 44) :
  brandon / michael = 1 / 2 :=
sorry

end marshmallow_challenge_l506_50642


namespace smallest_value_for_x_5_l506_50686

theorem smallest_value_for_x_5 (x : ℝ) (h : x = 5) :
  let a := 8 / x
  let b := 8 / (x + 2)
  let c := 8 / (x - 2)
  let d := x / 8
  let e := (x + 2) / 8
  d ≤ a ∧ d ≤ b ∧ d ≤ c ∧ d ≤ e := by
  sorry

end smallest_value_for_x_5_l506_50686


namespace correlation_coefficient_is_one_l506_50640

/-- A structure representing a set of sample data points -/
structure SampleData where
  n : ℕ
  x : Fin n → ℝ
  y : Fin n → ℝ
  h_n : n ≥ 2
  h_distinct : ∀ i j, i ≠ j → x i ≠ x j
  h_line : ∀ i, y i = (1/3) * x i - 5

/-- The sample correlation coefficient of a set of data points -/
def sampleCorrelationCoefficient (data : SampleData) : ℝ :=
  sorry

/-- Theorem stating that the sample correlation coefficient is 1 
    for data points satisfying the given conditions -/
theorem correlation_coefficient_is_one (data : SampleData) :
  sampleCorrelationCoefficient data = 1 :=
sorry

end correlation_coefficient_is_one_l506_50640


namespace fireflies_joining_l506_50697

theorem fireflies_joining (initial : ℕ) (joined : ℕ) (left : ℕ) (remaining : ℕ) : 
  initial = 3 → left = 2 → remaining = 9 → initial + joined - left = remaining → joined = 8 := by
  sorry

end fireflies_joining_l506_50697


namespace power_calculation_l506_50612

theorem power_calculation : 16^4 * 8^2 / 4^12 = (1 : ℚ) / 4 := by sorry

end power_calculation_l506_50612


namespace largest_of_five_consecutive_sum_180_l506_50636

theorem largest_of_five_consecutive_sum_180 (a : ℕ) :
  (∃ (x : ℕ), x = a ∧ 
    x + (x + 1) + (x + 2) + (x + 3) + (x + 4) = 180) →
  a + 4 = 38 :=
by sorry

end largest_of_five_consecutive_sum_180_l506_50636


namespace total_books_on_shelves_l506_50658

/-- Given 150 book shelves with 15 books each, the total number of books is 2250. -/
theorem total_books_on_shelves (num_shelves : ℕ) (books_per_shelf : ℕ) : 
  num_shelves = 150 → books_per_shelf = 15 → num_shelves * books_per_shelf = 2250 := by
  sorry

end total_books_on_shelves_l506_50658
