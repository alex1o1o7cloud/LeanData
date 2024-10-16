import Mathlib

namespace NUMINAMATH_CALUDE_binary_multiplication_l4127_412750

-- Define binary numbers as natural numbers
def binary_1111 : ℕ := 15  -- 1111₂ in decimal
def binary_111 : ℕ := 7    -- 111₂ in decimal

-- Define the result in binary as a natural number
def result : ℕ := 79       -- 1001111₂ in decimal

-- Theorem statement
theorem binary_multiplication :
  binary_1111 * binary_111 = result := by
  sorry

end NUMINAMATH_CALUDE_binary_multiplication_l4127_412750


namespace NUMINAMATH_CALUDE_fahrenheit_for_40_celsius_l4127_412712

-- Define the relationship between C and F
def celsius_to_fahrenheit (C F : ℝ) : Prop :=
  C = (5/9) * (F - 32)

-- Theorem statement
theorem fahrenheit_for_40_celsius :
  ∃ F : ℝ, celsius_to_fahrenheit 40 F ∧ F = 104 :=
by
  sorry

end NUMINAMATH_CALUDE_fahrenheit_for_40_celsius_l4127_412712


namespace NUMINAMATH_CALUDE_flowerbed_fence_length_l4127_412736

/-- Calculates the perimeter of a rectangular flowerbed with given width and length rule -/
def flowerbed_perimeter (width : ℝ) : ℝ :=
  let length := 2 * width - 1
  2 * (width + length)

/-- Theorem stating that a rectangular flowerbed with width 4 meters and length 1 meter less than twice its width has a perimeter of 22 meters -/
theorem flowerbed_fence_length : flowerbed_perimeter 4 = 22 := by
  sorry

end NUMINAMATH_CALUDE_flowerbed_fence_length_l4127_412736


namespace NUMINAMATH_CALUDE_quadratic_equation_root_l4127_412731

/-- Given a quadratic equation (m+2)x^2 - x + m^2 - 4 = 0 where one root is 0,
    prove that the other root is 1/4 -/
theorem quadratic_equation_root (m : ℝ) :
  (∃ x : ℝ, (m + 2) * x^2 - x + m^2 - 4 = 0 ∧ x = 0) →
  (∃ y : ℝ, (m + 2) * y^2 - y + m^2 - 4 = 0 ∧ y = 1/4) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_root_l4127_412731


namespace NUMINAMATH_CALUDE_correct_equation_l4127_412721

theorem correct_equation (a b : ℝ) : 5 * a^2 * b - 6 * a^2 * b = -a^2 * b := by
  sorry

end NUMINAMATH_CALUDE_correct_equation_l4127_412721


namespace NUMINAMATH_CALUDE_sun_radius_scientific_notation_l4127_412758

/-- The radius of the Sun in meters -/
def sun_radius : ℝ := 696000000

/-- Scientific notation representation of the Sun's radius -/
def sun_radius_scientific : ℝ := 6.96 * (10 ^ 8)

/-- Theorem stating that the Sun's radius is correctly expressed in scientific notation -/
theorem sun_radius_scientific_notation : sun_radius = sun_radius_scientific :=
sorry

end NUMINAMATH_CALUDE_sun_radius_scientific_notation_l4127_412758


namespace NUMINAMATH_CALUDE_sqrt_meaningful_range_l4127_412709

theorem sqrt_meaningful_range (x : ℝ) : 
  (∃ y : ℝ, y^2 = x + 2) ↔ x ≥ -2 := by sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_range_l4127_412709


namespace NUMINAMATH_CALUDE_min_value_of_sum_l4127_412741

theorem min_value_of_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 3 * a + 2 * b = 1) :
  (3 / a + 2 / b) ≥ 25 ∧ ∃ a₀ b₀ : ℝ, a₀ > 0 ∧ b₀ > 0 ∧ 3 * a₀ + 2 * b₀ = 1 ∧ 3 / a₀ + 2 / b₀ = 25 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_sum_l4127_412741


namespace NUMINAMATH_CALUDE_stream_speed_l4127_412785

/-- Proves that the speed of the stream is 4 km/hr, given the boat's speed in still water
    and the time and distance traveled downstream. -/
theorem stream_speed (boat_speed : ℝ) (time : ℝ) (distance : ℝ) :
  boat_speed = 16 →
  time = 3 →
  distance = 60 →
  ∃ (stream_speed : ℝ), stream_speed = 4 ∧ distance = (boat_speed + stream_speed) * time :=
by sorry

end NUMINAMATH_CALUDE_stream_speed_l4127_412785


namespace NUMINAMATH_CALUDE_total_donation_l4127_412787

theorem total_donation (megan_inheritance dan_inheritance : ℕ) 
  (h1 : megan_inheritance = 1000000)
  (h2 : dan_inheritance = 10000)
  (donation_percentage : ℚ)
  (h3 : donation_percentage = 1/10) : 
  (megan_inheritance * donation_percentage).floor + 
  (dan_inheritance * donation_percentage).floor = 101000 := by
  sorry

end NUMINAMATH_CALUDE_total_donation_l4127_412787


namespace NUMINAMATH_CALUDE_gain_percent_calculation_l4127_412789

/-- If the cost price of 50 articles equals the selling price of 35 articles,
    then the gain percent is (3/7) * 100. -/
theorem gain_percent_calculation (C S : ℝ) (h : 50 * C = 35 * S) :
  (S - C) / C * 100 = (3 / 7) * 100 := by
  sorry

end NUMINAMATH_CALUDE_gain_percent_calculation_l4127_412789


namespace NUMINAMATH_CALUDE_intersection_empty_implies_a_geq_5_not_p_sufficient_not_necessary_implies_a_leq_2_l4127_412797

-- Define the sets A and B
def A : Set ℝ := {x | -1 < x ∧ x < 6}
def B (a : ℝ) : Set ℝ := {x | x^2 - 2*x + 1 - a^2 ≥ 0}

-- Define propositions p and q
def p (x : ℝ) : Prop := x ∈ A
def q (a : ℝ) (x : ℝ) : Prop := x ∈ B a

-- Theorem for part (1)
theorem intersection_empty_implies_a_geq_5 (a : ℝ) :
  (a > 0) → (A ∩ B a = ∅) → a ≥ 5 := by sorry

-- Theorem for part (2)
theorem not_p_sufficient_not_necessary_implies_a_leq_2 (a : ℝ) :
  (a > 0) → (∀ x, ¬(p x) → q a x) → (∃ x, q a x ∧ p x) → a ≤ 2 := by sorry

end NUMINAMATH_CALUDE_intersection_empty_implies_a_geq_5_not_p_sufficient_not_necessary_implies_a_leq_2_l4127_412797


namespace NUMINAMATH_CALUDE_sin_over_x_satisfies_equation_l4127_412772

open Real

theorem sin_over_x_satisfies_equation (x : ℝ) (hx : x ≠ 0) :
  let y : ℝ → ℝ := fun x => sin x / x
  let y' : ℝ → ℝ := fun x => (x * cos x - sin x) / (x^2)
  x * y' x + y x = cos x := by
  sorry

end NUMINAMATH_CALUDE_sin_over_x_satisfies_equation_l4127_412772


namespace NUMINAMATH_CALUDE_sector_central_angle_sine_l4127_412710

theorem sector_central_angle_sine (r : ℝ) (arc_length : ℝ) (h1 : r = 2) (h2 : arc_length = 8 * Real.pi / 3) :
  Real.sin (arc_length / r) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sector_central_angle_sine_l4127_412710


namespace NUMINAMATH_CALUDE_second_class_size_l4127_412711

theorem second_class_size (students1 : ℕ) (avg1 : ℝ) (avg2 : ℝ) (avg_total : ℝ) :
  students1 = 30 →
  avg1 = 40 →
  avg2 = 90 →
  avg_total = 71.25 →
  ∃ students2 : ℕ, 
    (students1 * avg1 + students2 * avg2) / (students1 + students2 : ℝ) = avg_total ∧
    students2 = 50 :=
by sorry

end NUMINAMATH_CALUDE_second_class_size_l4127_412711


namespace NUMINAMATH_CALUDE_must_divide_a_l4127_412749

theorem must_divide_a (a b c d : ℕ+) 
  (h1 : Nat.gcd a b = 30)
  (h2 : Nat.gcd b c = 45)
  (h3 : Nat.gcd c d = 60)
  (h4 : 80 < Nat.gcd d a)
  (h5 : Nat.gcd d a < 120) :
  7 ∣ a.val := by
  sorry

end NUMINAMATH_CALUDE_must_divide_a_l4127_412749


namespace NUMINAMATH_CALUDE_strength_increase_percentage_l4127_412742

/-- Calculates the percentage increase in strength due to a magical bracer --/
theorem strength_increase_percentage 
  (original_weight : ℝ) 
  (training_increase : ℝ) 
  (final_weight : ℝ) : 
  original_weight = 135 →
  training_increase = 265 →
  final_weight = 2800 →
  ((final_weight - (original_weight + training_increase)) / (original_weight + training_increase)) * 100 = 600 := by
  sorry

end NUMINAMATH_CALUDE_strength_increase_percentage_l4127_412742


namespace NUMINAMATH_CALUDE_greatest_multiple_of_nine_with_odd_unique_digits_l4127_412702

/-- A function that checks if a natural number has all odd and unique digits -/
def hasOddUniqueDigits (n : ℕ) : Prop := sorry

/-- A function that returns the remainder when n is divided by m -/
def remainder (n m : ℕ) : ℕ := sorry

/-- The theorem statement -/
theorem greatest_multiple_of_nine_with_odd_unique_digits :
  ∃ M : ℕ,
    M % 9 = 0 ∧
    hasOddUniqueDigits M ∧
    (∀ N : ℕ, N % 9 = 0 → hasOddUniqueDigits N → N ≤ M) ∧
    remainder M 1000 = 531 := by
  sorry

end NUMINAMATH_CALUDE_greatest_multiple_of_nine_with_odd_unique_digits_l4127_412702


namespace NUMINAMATH_CALUDE_inequality_proof_l4127_412748

theorem inequality_proof (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h1 : a ≤ 2 * b) (h2 : 2 * b ≤ 4 * a) :
  4 * a * b ≤ 2 * (a^2 + b^2) ∧ 2 * (a^2 + b^2) ≤ 5 * a * b := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l4127_412748


namespace NUMINAMATH_CALUDE_product_equality_l4127_412770

theorem product_equality : 3^2 * 5 * 7^2 * 11 = 24255 := by
  sorry

end NUMINAMATH_CALUDE_product_equality_l4127_412770


namespace NUMINAMATH_CALUDE_cubic_equation_roots_l4127_412734

/-- The equation x³ - x - 2/(3√3) = 0 has exactly three real roots: 2/√3, -1/√3, and -1/√3 -/
theorem cubic_equation_roots :
  ∃! (s : Finset ℝ), s.card = 3 ∧ ∀ x ∈ s, x^3 - x - 2/(3*Real.sqrt 3) = 0 ∧ 
  (2/Real.sqrt 3 ∈ s ∧ -1/Real.sqrt 3 ∈ s) :=
sorry

end NUMINAMATH_CALUDE_cubic_equation_roots_l4127_412734


namespace NUMINAMATH_CALUDE_bakery_storage_ratio_l4127_412776

/-- Given the conditions in a bakery storage room, prove the ratio of flour to baking soda --/
theorem bakery_storage_ratio : 
  ∀ (sugar flour baking_soda : ℕ),
  -- Conditions
  sugar = 900 ∧ 
  3 * flour = 8 * sugar ∧ 
  8 * (baking_soda + 60) = flour →
  -- Conclusion
  flour = 10 * baking_soda := by
  sorry

end NUMINAMATH_CALUDE_bakery_storage_ratio_l4127_412776


namespace NUMINAMATH_CALUDE_least_k_divisible_by_1680_l4127_412779

theorem least_k_divisible_by_1680 :
  ∃ (k : ℕ), k > 0 ∧
  (∃ (a b c d : ℕ), k = 2^a * 3^b * 5^c * 7^d) ∧
  (1680 ∣ k^4) ∧
  (∀ (m : ℕ), m > 0 →
    (∃ (x y z w : ℕ), m = 2^x * 3^y * 5^z * 7^w) →
    (1680 ∣ m^4) →
    m ≥ k) ∧
  k = 210 :=
sorry

end NUMINAMATH_CALUDE_least_k_divisible_by_1680_l4127_412779


namespace NUMINAMATH_CALUDE_difference_of_squares_special_case_l4127_412793

theorem difference_of_squares_special_case : (831 : ℤ) * 831 - 830 * 832 = 1 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_special_case_l4127_412793


namespace NUMINAMATH_CALUDE_cubic_sum_divided_by_quadratic_sum_l4127_412786

theorem cubic_sum_divided_by_quadratic_sum (a b c : ℚ) 
  (ha : a = 7) (hb : b = 5) (hc : c = -2) : 
  (a^3 + b^3 + c^3) / (a^2 - a*b + b^2 + c^2) = 460 / 43 := by
  sorry

end NUMINAMATH_CALUDE_cubic_sum_divided_by_quadratic_sum_l4127_412786


namespace NUMINAMATH_CALUDE_card_drawing_certainty_l4127_412720

theorem card_drawing_certainty (total : ℕ) (hearts clubs spades drawn : ℕ) 
  (h_total : total = hearts + clubs + spades)
  (h_hearts : hearts = 5)
  (h_clubs : clubs = 4)
  (h_spades : spades = 3)
  (h_drawn : drawn = 10) :
  ∀ (draw : Finset ℕ), draw.card = drawn → 
    (∃ (h c s : ℕ), h ∈ draw ∧ c ∈ draw ∧ s ∈ draw ∧ 
      h ≤ hearts ∧ c ≤ clubs ∧ s ≤ spades) :=
sorry

end NUMINAMATH_CALUDE_card_drawing_certainty_l4127_412720


namespace NUMINAMATH_CALUDE_christmas_games_l4127_412717

theorem christmas_games (C B : ℕ) (h1 : B = 8) (h2 : C + B + (C + B) / 2 = 30) : C = 12 := by
  sorry

end NUMINAMATH_CALUDE_christmas_games_l4127_412717


namespace NUMINAMATH_CALUDE_total_distance_calculation_l4127_412754

-- Define the distance walked per day
def distance_per_day : ℝ := 4.0

-- Define the number of days walked
def days_walked : ℝ := 3.0

-- Define the total distance walked
def total_distance : ℝ := distance_per_day * days_walked

-- Theorem statement
theorem total_distance_calculation :
  total_distance = 12.0 := by sorry

end NUMINAMATH_CALUDE_total_distance_calculation_l4127_412754


namespace NUMINAMATH_CALUDE_factorization_identities_l4127_412725

theorem factorization_identities (x y m n a b : ℝ) : 
  (3*x - 12*x^3 = 3*x*(1-2*x)*(1+2*x)) ∧ 
  (9*m^2 - 4*n^2 = (3*m+2*n)*(3*m-2*n)) ∧ 
  (a^2*(x-y) + b^2*(y-x) = (x-y)*(a+b)*(a-b)) ∧ 
  (x^2 - 4*x*y + 4*y^2 - 1 = (x-y+1)*(x-y-1)) := by sorry

end NUMINAMATH_CALUDE_factorization_identities_l4127_412725


namespace NUMINAMATH_CALUDE_equation_solution_l4127_412761

theorem equation_solution :
  let f : ℝ → ℝ := λ x => 1 / (x + 3) + 3 * x / (x + 3) - 5 / (x + 3) + 2 / (x - 1)
  ∀ x : ℝ, f x = 5 ↔ x = (-11 + Real.sqrt 257) / 4 ∨ x = (-11 - Real.sqrt 257) / 4 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l4127_412761


namespace NUMINAMATH_CALUDE_percentage_solution_l4127_412723

/-- The percentage that, when applied to 100 and added to 20, results in 100 -/
def percentage_problem (P : ℝ) : Prop :=
  100 * (P / 100) + 20 = 100

/-- The solution to the percentage problem is 80% -/
theorem percentage_solution : ∃ P : ℝ, percentage_problem P ∧ P = 80 := by
  sorry

end NUMINAMATH_CALUDE_percentage_solution_l4127_412723


namespace NUMINAMATH_CALUDE_sphere_volume_ratio_l4127_412783

theorem sphere_volume_ratio (r₁ r₂ : ℝ) (h : r₁ > 0 ∧ r₂ > 0) :
  (4 * Real.pi * r₁^2) / (4 * Real.pi * r₂^2) = 4 / 9 →
  ((4 / 3) * Real.pi * r₁^3) / ((4 / 3) * Real.pi * r₂^3) = 8 / 27 := by
sorry

end NUMINAMATH_CALUDE_sphere_volume_ratio_l4127_412783


namespace NUMINAMATH_CALUDE_correct_sunset_time_l4127_412743

/-- Represents time in hours and minutes -/
structure Time where
  hours : ℕ
  minutes : ℕ
  valid : minutes < 60

/-- Represents a duration in hours and minutes -/
structure Duration where
  hours : ℕ
  minutes : ℕ

/-- Adds a duration to a time -/
def addDurationToTime (t : Time) (d : Duration) : Time :=
  let totalMinutes := t.hours * 60 + t.minutes + d.hours * 60 + d.minutes
  let newHours := totalMinutes / 60
  let newMinutes := totalMinutes % 60
  { hours := newHours % 24, minutes := newMinutes, valid := by sorry }

theorem correct_sunset_time : 
  let sunrise : Time := { hours := 5, minutes := 35, valid := by sorry }
  let daylight : Duration := { hours := 14, minutes := 42 }
  let sunset := addDurationToTime sunrise daylight
  sunset.hours = 20 ∧ sunset.minutes = 17 := by sorry

end NUMINAMATH_CALUDE_correct_sunset_time_l4127_412743


namespace NUMINAMATH_CALUDE_pictures_picked_out_l4127_412791

def total_pictures : ℕ := 10
def jim_bought : ℕ := 3
def probability : ℚ := 7/15

theorem pictures_picked_out :
  ∃ n : ℕ, n > 0 ∧ n < total_pictures ∧
  (Nat.choose (total_pictures - jim_bought) n : ℚ) / (Nat.choose total_pictures n : ℚ) = probability ∧
  n = 2 := by
  sorry

end NUMINAMATH_CALUDE_pictures_picked_out_l4127_412791


namespace NUMINAMATH_CALUDE_misha_current_money_l4127_412762

/-- The amount of money Misha needs to earn -/
def additional_money : ℕ := 13

/-- The total amount Misha would have after earning the additional money -/
def total_money : ℕ := 47

/-- Misha's current money amount -/
def current_money : ℕ := total_money - additional_money

theorem misha_current_money : current_money = 34 := by
  sorry

end NUMINAMATH_CALUDE_misha_current_money_l4127_412762


namespace NUMINAMATH_CALUDE_juanita_contest_cost_l4127_412777

/-- Represents the drumming contest Juanita entered -/
structure DrummingContest where
  min_drums : ℕ  -- Minimum number of drums to hit before earning money
  earn_rate : ℚ  -- Amount earned per drum hit above min_drums
  time_limit : ℕ  -- Time limit in minutes

/-- Represents Juanita's performance in the contest -/
structure Performance where
  drums_hit : ℕ  -- Number of drums hit
  money_lost : ℚ  -- Amount of money lost (negative earnings)

def contest_entry_cost (contest : DrummingContest) (performance : Performance) : ℚ :=
  let earnings := max ((performance.drums_hit - contest.min_drums) * contest.earn_rate) 0
  earnings + performance.money_lost

theorem juanita_contest_cost :
  let contest := DrummingContest.mk 200 0.025 2
  let performance := Performance.mk 300 7.5
  contest_entry_cost contest performance = 10 := by
  sorry

end NUMINAMATH_CALUDE_juanita_contest_cost_l4127_412777


namespace NUMINAMATH_CALUDE_odd_integers_sum_21_to_65_l4127_412718

def arithmetic_sum (a₁ : ℕ) (aₙ : ℕ) (d : ℕ) : ℕ :=
  let n := (aₙ - a₁) / d + 1
  n * (a₁ + aₙ) / 2

theorem odd_integers_sum_21_to_65 :
  arithmetic_sum 21 65 2 = 989 := by
  sorry

end NUMINAMATH_CALUDE_odd_integers_sum_21_to_65_l4127_412718


namespace NUMINAMATH_CALUDE_crayons_lost_or_given_away_l4127_412768

theorem crayons_lost_or_given_away (initial_crayons remaining_crayons : ℕ) 
  (h1 : initial_crayons = 606)
  (h2 : remaining_crayons = 291) :
  initial_crayons - remaining_crayons = 315 := by
  sorry

end NUMINAMATH_CALUDE_crayons_lost_or_given_away_l4127_412768


namespace NUMINAMATH_CALUDE_sum_equation_implies_N_value_l4127_412753

theorem sum_equation_implies_N_value :
  481 + 483 + 485 + 487 + 489 + 491 = 3000 - N → N = 84 := by
  sorry

end NUMINAMATH_CALUDE_sum_equation_implies_N_value_l4127_412753


namespace NUMINAMATH_CALUDE_point_symmetry_l4127_412706

def f (x : ℝ) : ℝ := x^3

theorem point_symmetry (a b : ℝ) : 
  (f a = b) → (f (-a) = -b) := by
  sorry

end NUMINAMATH_CALUDE_point_symmetry_l4127_412706


namespace NUMINAMATH_CALUDE_product_expansion_sum_l4127_412796

theorem product_expansion_sum (a b c d : ℝ) : 
  (∀ x, (5*x^2 - 3*x + 2)*(9 - 3*x) = a*x^3 + b*x^2 + c*x + d) →
  27*a + 9*b + 3*c + d = 0 := by
sorry

end NUMINAMATH_CALUDE_product_expansion_sum_l4127_412796


namespace NUMINAMATH_CALUDE_B_2_2_l4127_412794

def B : ℕ → ℕ → ℕ
| 0, n => n + 2
| m+1, 0 => B m 2
| m+1, n+1 => B m (B (m+1) n)

theorem B_2_2 : B 2 2 = 8 := by sorry

end NUMINAMATH_CALUDE_B_2_2_l4127_412794


namespace NUMINAMATH_CALUDE_prime_count_in_range_l4127_412760

theorem prime_count_in_range (n : ℕ) (h : n > 2) :
  (n > 3 → ∀ p, Nat.Prime p → ¬((n - 1).factorial + 2 < p ∧ p < (n - 1).factorial + n)) ∧
  (n = 3 → ∃! p, Nat.Prime p ∧ ((n - 1).factorial + 2 < p ∧ p < (n - 1).factorial + n)) :=
sorry

end NUMINAMATH_CALUDE_prime_count_in_range_l4127_412760


namespace NUMINAMATH_CALUDE_contractor_work_completion_l4127_412792

/-- Represents the problem of determining when 1/4 of the work was completed. -/
theorem contractor_work_completion (total_days : ℕ) (initial_workers : ℕ) (remaining_days : ℕ) (fired_workers : ℕ) : 
  total_days = 100 →
  initial_workers = 10 →
  remaining_days = 75 →
  fired_workers = 2 →
  ∃ (x : ℕ), 
    (x * initial_workers = remaining_days * (initial_workers - fired_workers)) ∧
    x = 60 :=
by sorry

end NUMINAMATH_CALUDE_contractor_work_completion_l4127_412792


namespace NUMINAMATH_CALUDE_tournament_result_l4127_412756

/-- Represents a tennis tournament with the given rules --/
structure TennisTournament where
  participants : ℕ
  points_for_win : ℕ
  points_for_loss : ℕ

/-- Calculates the number of participants finishing with a given number of points --/
def participants_with_points (t : TennisTournament) (points : ℕ) : ℕ :=
  Nat.choose (Nat.log 2 t.participants) points

theorem tournament_result (t : TennisTournament) 
  (h1 : t.participants = 512)
  (h2 : t.points_for_win = 1)
  (h3 : t.points_for_loss = 0) :
  participants_with_points t 6 = 84 := by
  sorry

end NUMINAMATH_CALUDE_tournament_result_l4127_412756


namespace NUMINAMATH_CALUDE_students_per_table_l4127_412757

theorem students_per_table (num_tables : ℕ) (total_students : ℕ) 
  (h1 : num_tables = 34) (h2 : total_students = 204) : 
  total_students / num_tables = 6 := by
  sorry

end NUMINAMATH_CALUDE_students_per_table_l4127_412757


namespace NUMINAMATH_CALUDE_committee_choice_count_l4127_412767

/-- The number of members in the club -/
def total_members : ℕ := 18

/-- The minimum tenure required for eligibility -/
def min_tenure : ℕ := 10

/-- The number of members to be chosen for the committee -/
def committee_size : ℕ := 3

/-- The number of eligible members (those with tenure ≥ 10 years) -/
def eligible_members : ℕ := total_members - min_tenure + 1

/-- The number of ways to choose the committee -/
def committee_choices : ℕ := Nat.choose eligible_members committee_size

theorem committee_choice_count :
  committee_choices = 84 := by sorry

end NUMINAMATH_CALUDE_committee_choice_count_l4127_412767


namespace NUMINAMATH_CALUDE_unique_pair_divides_l4127_412714

theorem unique_pair_divides (a b : ℕ) (ha : a > 1) (hb : b > 1) :
  (b^a ∣ a^b - 1) ↔ (a = 3 ∧ b = 2) :=
sorry

end NUMINAMATH_CALUDE_unique_pair_divides_l4127_412714


namespace NUMINAMATH_CALUDE_quadratic_equations_solutions_l4127_412715

theorem quadratic_equations_solutions :
  (∃ x₁ x₂ : ℝ, x₁ = -5 ∧ x₂ = 1 ∧ x₁^2 + 4*x₁ - 5 = 0 ∧ x₂^2 + 4*x₂ - 5 = 0) ∧
  (∃ y₁ y₂ : ℝ, y₁ = 1/3 ∧ y₂ = -1 ∧ 3*y₁^2 + 2*y₁ = 1 ∧ 3*y₂^2 + 2*y₂ = 1) :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_equations_solutions_l4127_412715


namespace NUMINAMATH_CALUDE_probability_of_selection_X_l4127_412744

theorem probability_of_selection_X (p_Y p_XY : ℝ) : 
  p_Y = 2/3 → p_XY = 0.13333333333333333 → ∃ p_X : ℝ, p_X = 0.2 ∧ p_XY = p_X * p_Y :=
by sorry

end NUMINAMATH_CALUDE_probability_of_selection_X_l4127_412744


namespace NUMINAMATH_CALUDE_graduation_messages_l4127_412727

theorem graduation_messages (n : ℕ) (h : n = 45) : 
  (n * (n - 1)) / 2 = 990 := by
  sorry

end NUMINAMATH_CALUDE_graduation_messages_l4127_412727


namespace NUMINAMATH_CALUDE_set_intersection_condition_l4127_412713

theorem set_intersection_condition (m : ℝ) : 
  let A := {x : ℝ | x^2 - 3*x + 2 = 0}
  let C := {x : ℝ | x^2 - m*x + 2 = 0}
  (A ∩ C = C) ↔ (m = 3 ∨ (-2 * Real.sqrt 2 < m ∧ m < 2 * Real.sqrt 2)) := by
sorry

end NUMINAMATH_CALUDE_set_intersection_condition_l4127_412713


namespace NUMINAMATH_CALUDE_other_root_of_quadratic_l4127_412765

theorem other_root_of_quadratic (m : ℝ) : 
  ((-4 : ℝ)^2 + m * (-4) - 20 = 0) → (5^2 + m * 5 - 20 = 0) := by
  sorry

end NUMINAMATH_CALUDE_other_root_of_quadratic_l4127_412765


namespace NUMINAMATH_CALUDE_max_second_term_arithmetic_sequence_l4127_412778

theorem max_second_term_arithmetic_sequence : ∀ (a d : ℕ),
  a > 0 ∧ d > 0 ∧ 
  a + (a + d) + (a + 2*d) + (a + 3*d) = 52 →
  a + d ≤ 10 :=
by sorry

end NUMINAMATH_CALUDE_max_second_term_arithmetic_sequence_l4127_412778


namespace NUMINAMATH_CALUDE_puzzle_solutions_l4127_412700

-- Define a structure for the puzzle solution
structure PuzzleSolution where
  a : Nat
  b : Nat
  v : Nat
  h : a * 10 + b = b ^ v ∧ a ≠ b ∧ a ≠ v ∧ b ≠ v ∧ a > 0 ∧ b > 0 ∧ v > 0 ∧ a < 10 ∧ b < 10 ∧ v < 10

-- Define the theorem
theorem puzzle_solutions :
  {s : PuzzleSolution | True} =
  {⟨3, 2, 5, sorry⟩, ⟨3, 6, 2, sorry⟩, ⟨6, 4, 3, sorry⟩} := by sorry

end NUMINAMATH_CALUDE_puzzle_solutions_l4127_412700


namespace NUMINAMATH_CALUDE_river_current_speed_proof_l4127_412769

def river_current_speed (boat_speed : ℝ) (distance : ℝ) (total_time : ℝ) : ℝ :=
  let current_speed := 4
  current_speed

theorem river_current_speed_proof (boat_speed distance total_time : ℝ) 
  (h1 : boat_speed = 20)
  (h2 : distance = 60)
  (h3 : total_time = 6.25) :
  river_current_speed boat_speed distance total_time = 4 := by
  sorry

#check river_current_speed_proof

end NUMINAMATH_CALUDE_river_current_speed_proof_l4127_412769


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l4127_412707

open Set

def U : Set Nat := {1, 2, 3, 4, 5, 6}
def A : Set Nat := {2, 3}
def B : Set Nat := {3, 5}

theorem intersection_A_complement_B : A ∩ (U \ B) = {2} := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l4127_412707


namespace NUMINAMATH_CALUDE_max_value_expression_l4127_412780

theorem max_value_expression (a b c d x : ℝ) 
  (h1 : a + b = 0) 
  (h2 : c * d = 1) 
  (h3 : x^2 = 1) : 
  ∃ (m : ℝ), ∀ y, y^2 = 1 → x^2 + a + b + c * d * x ≤ m ∧ m = 2 := by
  sorry

end NUMINAMATH_CALUDE_max_value_expression_l4127_412780


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_l4127_412788

-- Define the right triangle XYZ
def XYZ : Set (ℝ × ℝ) := sorry

-- Define the lengths of the sides
def XZ : ℝ := 15
def YZ : ℝ := 8

-- Define that Z is a right angle
def Z_is_right_angle : sorry := sorry

-- Define the inscribed circle
def inscribed_circle : Set (ℝ × ℝ) := sorry

-- State the theorem
theorem inscribed_circle_radius :
  ∃ (r : ℝ), r = 3 ∧ ∀ (p : ℝ × ℝ), p ∈ inscribed_circle → 
    ∃ (c : ℝ × ℝ), c ∈ XYZ ∧ dist p c = r :=
sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_l4127_412788


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l4127_412719

theorem contrapositive_equivalence :
  (∀ a : ℝ, a > 0 → a^2 > 0) ↔ (∀ a : ℝ, a^2 ≤ 0 → a ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l4127_412719


namespace NUMINAMATH_CALUDE_binary_encodes_to_032239_l4127_412784

/-- Represents a mapping from characters to digits -/
def EncodeMap := Char → Nat

/-- The encoding scheme based on "MONITOR KEYBOARD" -/
def monitorKeyboardEncode : EncodeMap :=
  fun c => match c with
  | 'M' => 0
  | 'O' => 1
  | 'N' => 2
  | 'I' => 3
  | 'T' => 4
  | 'R' => 6
  | 'K' => 7
  | 'E' => 8
  | 'Y' => 9
  | 'B' => 0
  | 'A' => 2
  | 'D' => 4
  | _ => 0  -- Default case, should not be reached for valid inputs

/-- Encodes a string to a list of digits using the given encoding map -/
def encodeString (encode : EncodeMap) (s : String) : List Nat :=
  s.data.map encode

/-- Converts a list of digits to a natural number -/
def digitsToNat (digits : List Nat) : Nat :=
  digits.foldl (fun acc d => acc * 10 + d) 0

/-- The main theorem: BINARY encodes to 032239 -/
theorem binary_encodes_to_032239 :
  digitsToNat (encodeString monitorKeyboardEncode "BINARY") = 032239 := by
  sorry


end NUMINAMATH_CALUDE_binary_encodes_to_032239_l4127_412784


namespace NUMINAMATH_CALUDE_square_plus_one_geq_two_abs_l4127_412735

theorem square_plus_one_geq_two_abs (x : ℝ) : x^2 + 1 ≥ 2 * |x| := by
  sorry

end NUMINAMATH_CALUDE_square_plus_one_geq_two_abs_l4127_412735


namespace NUMINAMATH_CALUDE_graph_single_point_implies_d_eq_21_l4127_412737

/-- The equation of the graph -/
def graph_equation (x y d : ℝ) : Prop :=
  3 * x^2 + y^2 + 12 * x - 6 * y + d = 0

/-- The condition that the graph consists of a single point -/
def is_single_point (d : ℝ) : Prop :=
  ∃! (x y : ℝ), graph_equation x y d

/-- Theorem: If the graph consists of a single point, then d = 21 -/
theorem graph_single_point_implies_d_eq_21 :
  ∀ d : ℝ, is_single_point d → d = 21 := by
  sorry

end NUMINAMATH_CALUDE_graph_single_point_implies_d_eq_21_l4127_412737


namespace NUMINAMATH_CALUDE_picnic_attendance_theorem_l4127_412726

/-- Represents the number of people at a picnic -/
structure PicnicAttendance where
  men : ℕ
  women : ℕ
  adults : ℕ
  children : ℕ

/-- Theorem: Given the conditions of the picnic, the total number of attendees is 240 -/
theorem picnic_attendance_theorem (p : PicnicAttendance) 
  (h1 : p.men = p.women + 80)
  (h2 : p.adults = p.children + 80)
  (h3 : p.men = 120)
  : p.adults + p.children = 240 := by
  sorry

#check picnic_attendance_theorem

end NUMINAMATH_CALUDE_picnic_attendance_theorem_l4127_412726


namespace NUMINAMATH_CALUDE_birds_on_trees_l4127_412703

theorem birds_on_trees (n : ℕ) (h : n = 44) : 
  let initial_sum := n * (n + 1) / 2
  ∀ (current_sum : ℕ), current_sum % 4 ≠ 0 →
    ∃ (next_sum : ℕ), (next_sum = current_sum ∨ next_sum = current_sum + n - 1 ∨ next_sum = current_sum - (n - 1)) ∧
      next_sum % 4 ≠ 0 :=
by sorry

#check birds_on_trees

end NUMINAMATH_CALUDE_birds_on_trees_l4127_412703


namespace NUMINAMATH_CALUDE_plan_A_cost_per_text_l4127_412730

/-- The cost per text message for Plan A, in dollars -/
def cost_per_text_A : ℝ := 0.25

/-- The monthly fee for Plan A, in dollars -/
def monthly_fee_A : ℝ := 9

/-- The cost per text message for Plan B, in dollars -/
def cost_per_text_B : ℝ := 0.40

/-- The number of text messages at which both plans cost the same -/
def equal_cost_messages : ℕ := 60

theorem plan_A_cost_per_text :
  cost_per_text_A * equal_cost_messages + monthly_fee_A =
  cost_per_text_B * equal_cost_messages :=
by sorry

end NUMINAMATH_CALUDE_plan_A_cost_per_text_l4127_412730


namespace NUMINAMATH_CALUDE_percent_application_l4127_412795

theorem percent_application (x : ℝ) : x * 0.0002 = 2.4712 → x = 12356 := by sorry

end NUMINAMATH_CALUDE_percent_application_l4127_412795


namespace NUMINAMATH_CALUDE_sum_and_difference_bounds_l4127_412732

theorem sum_and_difference_bounds (a b : ℝ) 
  (ha : 60 ≤ a ∧ a ≤ 84) (hb : 28 ≤ b ∧ b ≤ 33) : 
  (88 ≤ a + b ∧ a + b ≤ 117) ∧ (27 ≤ a - b ∧ a - b ≤ 56) := by
  sorry

end NUMINAMATH_CALUDE_sum_and_difference_bounds_l4127_412732


namespace NUMINAMATH_CALUDE_continuous_and_strictly_monotone_function_l4127_412705

-- Define a function type from reals to reals
def RealFunction := ℝ → ℝ

-- Define the property of having limits at any point
def has_limits_at_any_point (f : RealFunction) : Prop :=
  ∀ a : ℝ, ∃ L : ℝ, ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, |x - a| < δ → |f x - L| < ε

-- Define the property of having no local extrema
def has_no_local_extrema (f : RealFunction) : Prop :=
  ∀ a : ℝ, ∀ ε > 0, ∃ x y : ℝ, |x - a| < ε ∧ |y - a| < ε ∧ f x < f a ∧ f a < f y

-- State the theorem
theorem continuous_and_strictly_monotone_function 
  (f : RealFunction) 
  (h1 : has_limits_at_any_point f) 
  (h2 : has_no_local_extrema f) : 
  Continuous f ∧ StrictMono f :=
by sorry

end NUMINAMATH_CALUDE_continuous_and_strictly_monotone_function_l4127_412705


namespace NUMINAMATH_CALUDE_number_of_balls_in_box_l4127_412746

theorem number_of_balls_in_box : ∃ n : ℕ, n - 44 = 70 - n ∧ n = 57 := by sorry

end NUMINAMATH_CALUDE_number_of_balls_in_box_l4127_412746


namespace NUMINAMATH_CALUDE_egg_price_calculation_l4127_412763

/-- Proves that the price of each egg is $0.20 given the conditions of the problem --/
theorem egg_price_calculation (total_eggs : ℕ) (crate_cost : ℚ) (eggs_left : ℕ) : 
  total_eggs = 30 → crate_cost = 5 → eggs_left = 5 → 
  (crate_cost / (total_eggs - eggs_left : ℚ)) = 0.20 := by
  sorry

#check egg_price_calculation

end NUMINAMATH_CALUDE_egg_price_calculation_l4127_412763


namespace NUMINAMATH_CALUDE_log_48_in_terms_of_a_and_b_l4127_412799

theorem log_48_in_terms_of_a_and_b (a b : ℝ) 
  (h1 : Real.log 3 / Real.log 7 = a) 
  (h2 : Real.log 4 / Real.log 7 = b) : 
  Real.log 48 / Real.log 7 = a + 2 * b := by sorry

end NUMINAMATH_CALUDE_log_48_in_terms_of_a_and_b_l4127_412799


namespace NUMINAMATH_CALUDE_vector_relation_l4127_412716

/-- Given points A, B, C, and D in a plane, where BC = 3CD, prove that AD = -1/3 AB + 4/3 AC -/
theorem vector_relation (A B C D : ℝ × ℝ) 
  (h : B - C = 3 * (C - D)) : 
  A - D = -1/3 * (A - B) + 4/3 * (A - C) := by
  sorry

end NUMINAMATH_CALUDE_vector_relation_l4127_412716


namespace NUMINAMATH_CALUDE_prime_divisibility_problem_l4127_412766

theorem prime_divisibility_problem (p q : ℕ) : 
  Prime p → Prime q → p < 2005 → q < 2005 → 
  (q ∣ p^2 + 4) → (p ∣ q^2 + 4) → 
  p = 2 ∧ q = 2 := by
sorry

end NUMINAMATH_CALUDE_prime_divisibility_problem_l4127_412766


namespace NUMINAMATH_CALUDE_current_speed_l4127_412701

/-- 
Given a man's speed with and against a current, this theorem proves 
the speed of the current.
-/
theorem current_speed 
  (speed_with_current : ℝ) 
  (speed_against_current : ℝ) 
  (h1 : speed_with_current = 12)
  (h2 : speed_against_current = 8) :
  ∃ (man_speed current_speed : ℝ),
    man_speed + current_speed = speed_with_current ∧
    man_speed - current_speed = speed_against_current ∧
    current_speed = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_current_speed_l4127_412701


namespace NUMINAMATH_CALUDE_roses_in_vase_after_actions_l4127_412747

/-- Represents the number of flowers in a vase -/
structure FlowerVase where
  roses : ℕ
  orchids : ℕ

/-- Represents the actions taken by Jessica -/
structure JessicaActions where
  addedRoses : ℕ
  addedOrchids : ℕ
  cutRoses : ℕ

def initial : FlowerVase := { roses := 15, orchids := 62 }

def actions : JessicaActions := { addedRoses := 0, addedOrchids := 34, cutRoses := 2 }

def final : FlowerVase := { roses := 96, orchids := initial.orchids + actions.addedOrchids }

theorem roses_in_vase_after_actions (R : ℕ) : 
  final.roses = 13 + R ↔ actions.addedRoses = R := by sorry

end NUMINAMATH_CALUDE_roses_in_vase_after_actions_l4127_412747


namespace NUMINAMATH_CALUDE_jacket_final_price_l4127_412745

def original_price : ℝ := 240
def initial_discount : ℝ := 0.6
def holiday_discount : ℝ := 0.25

theorem jacket_final_price :
  let price_after_initial := original_price * (1 - initial_discount)
  let final_price := price_after_initial * (1 - holiday_discount)
  final_price = 72 := by sorry

end NUMINAMATH_CALUDE_jacket_final_price_l4127_412745


namespace NUMINAMATH_CALUDE_equal_area_trapezoid_kp_l4127_412704

/-- Represents a trapezoid with two bases and a point that divides it into equal areas -/
structure EqualAreaTrapezoid where
  /-- Length of the longer base KL -/
  base_kl : ℝ
  /-- Length of the shorter base MN -/
  base_mn : ℝ
  /-- Length of segment KP, where P divides the trapezoid into equal areas when connected to N -/
  kp : ℝ
  /-- Assumption that base_kl is greater than base_mn -/
  h_base : base_kl > base_mn
  /-- Assumption that all lengths are positive -/
  h_positive : base_kl > 0 ∧ base_mn > 0 ∧ kp > 0

/-- Theorem stating that for a trapezoid with given dimensions, KP = 28 when P divides the area equally -/
theorem equal_area_trapezoid_kp
  (t : EqualAreaTrapezoid)
  (h_kl : t.base_kl = 40)
  (h_mn : t.base_mn = 16) :
  t.kp = 28 := by
  sorry

#check equal_area_trapezoid_kp

end NUMINAMATH_CALUDE_equal_area_trapezoid_kp_l4127_412704


namespace NUMINAMATH_CALUDE_inequality_proof_l4127_412738

theorem inequality_proof (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) :
  (y*z + z*x + x*y)^2 * (x + y + z) ≥ 4 * x*y*z * (x^2 + y^2 + z^2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l4127_412738


namespace NUMINAMATH_CALUDE_quadratic_composite_zeros_l4127_412773

/-- A quadratic function f(x) = ax^2 + bx + c where a > 0 -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  a_pos : a > 0

/-- The function f(x) -/
def f (q : QuadraticFunction) (x : ℝ) : ℝ :=
  q.a * x^2 + q.b * x + q.c

/-- The composite function f(f(x)) -/
def f_comp_f (q : QuadraticFunction) (x : ℝ) : ℝ :=
  f q (f q x)

/-- The number of distinct real zeros of a function -/
def num_distinct_real_zeros (g : ℝ → ℝ) : ℕ := sorry

theorem quadratic_composite_zeros
  (q : QuadraticFunction)
  (h : f q (1 / q.a) < 0) :
  num_distinct_real_zeros (f_comp_f q) = 4 :=
sorry

end NUMINAMATH_CALUDE_quadratic_composite_zeros_l4127_412773


namespace NUMINAMATH_CALUDE_greatest_x_value_l4127_412771

theorem greatest_x_value : ∃ (x_max : ℝ),
  (∀ x : ℝ, (x^2 - 3*x - 70) / (x - 10) = 5 / (x + 7) → x ≤ x_max) ∧
  ((x_max^2 - 3*x_max - 70) / (x_max - 10) = 5 / (x_max + 7)) ∧
  x_max = -2 :=
by sorry

end NUMINAMATH_CALUDE_greatest_x_value_l4127_412771


namespace NUMINAMATH_CALUDE_inequality_system_solution_l4127_412751

theorem inequality_system_solution (x : ℝ) :
  (6 * x + 1 ≤ 4 * (x - 1)) ∧ (1 - x / 4 > (x + 5) / 2) → x ≤ -5/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l4127_412751


namespace NUMINAMATH_CALUDE_base_6_to_base_3_conversion_l4127_412729

def base_6_to_decimal (n : ℕ) : ℕ :=
  2 * 6^2 + 1 * 6^1 + 0 * 6^0

def base_3_to_decimal (n : ℕ) : ℕ :=
  2 * 3^3 + 2 * 3^2 + 2 * 3^1 + 0 * 3^0

theorem base_6_to_base_3_conversion :
  base_6_to_decimal 210 = base_3_to_decimal 2220 := by
  sorry

end NUMINAMATH_CALUDE_base_6_to_base_3_conversion_l4127_412729


namespace NUMINAMATH_CALUDE_flower_stitches_l4127_412775

/-- Proves that given the conditions, the number of stitches required to embroider one flower is 60. -/
theorem flower_stitches (
  stitches_per_minute : ℕ)
  (unicorn_stitches : ℕ)
  (godzilla_stitches : ℕ)
  (num_unicorns : ℕ)
  (num_flowers : ℕ)
  (total_minutes : ℕ)
  (h1 : stitches_per_minute = 4)
  (h2 : unicorn_stitches = 180)
  (h3 : godzilla_stitches = 800)
  (h4 : num_unicorns = 3)
  (h5 : num_flowers = 50)
  (h6 : total_minutes = 1085)
  : (total_minutes * stitches_per_minute - (num_unicorns * unicorn_stitches + godzilla_stitches)) / num_flowers = 60 :=
sorry

end NUMINAMATH_CALUDE_flower_stitches_l4127_412775


namespace NUMINAMATH_CALUDE_crow_votes_l4127_412774

/-- Represents the number of votes for each participant -/
structure Votes where
  rooster : ℕ
  crow : ℕ
  cuckoo : ℕ

/-- Represents Woodpecker's counts -/
structure WoodpeckerCounts where
  total : ℕ
  roosterAndCrow : ℕ
  crowAndCuckoo : ℕ
  cuckooAndRooster : ℕ

/-- The maximum error in Woodpecker's counts -/
def maxError : ℕ := 13

/-- Check if a number is within the error range of another number -/
def withinErrorRange (actual : ℕ) (counted : ℕ) : Prop :=
  (actual ≤ counted + maxError) ∧ (counted ≤ actual + maxError)

/-- The theorem to be proved -/
theorem crow_votes (v : Votes) (w : WoodpeckerCounts) 
  (h1 : withinErrorRange (v.rooster + v.crow + v.cuckoo) w.total)
  (h2 : withinErrorRange (v.rooster + v.crow) w.roosterAndCrow)
  (h3 : withinErrorRange (v.crow + v.cuckoo) w.crowAndCuckoo)
  (h4 : withinErrorRange (v.cuckoo + v.rooster) w.cuckooAndRooster)
  (h5 : w.total = 59)
  (h6 : w.roosterAndCrow = 15)
  (h7 : w.crowAndCuckoo = 18)
  (h8 : w.cuckooAndRooster = 20) :
  v.crow = 13 := by
  sorry

end NUMINAMATH_CALUDE_crow_votes_l4127_412774


namespace NUMINAMATH_CALUDE_convention_handshakes_l4127_412755

-- Define the number of companies and representatives per company
def num_companies : ℕ := 5
def reps_per_company : ℕ := 4

-- Define the total number of people
def total_people : ℕ := num_companies * reps_per_company

-- Define the number of handshakes per person
def handshakes_per_person : ℕ := total_people - reps_per_company

-- Theorem statement
theorem convention_handshakes : 
  (total_people * handshakes_per_person) / 2 = 160 := by
  sorry


end NUMINAMATH_CALUDE_convention_handshakes_l4127_412755


namespace NUMINAMATH_CALUDE_p_shape_points_l4127_412724

/-- Represents a "P" shape formed from a square -/
structure PShape :=
  (side_length : ℕ)

/-- Counts the number of distinct points on a "P" shape -/
def count_points (p : PShape) : ℕ :=
  3 * (p.side_length + 1) - 2

/-- Theorem stating the number of points on a "P" shape with side length 10 -/
theorem p_shape_points :
  let p : PShape := { side_length := 10 }
  count_points p = 31 := by sorry

end NUMINAMATH_CALUDE_p_shape_points_l4127_412724


namespace NUMINAMATH_CALUDE_marble_prob_diff_l4127_412782

/-- The number of red marbles in the box -/
def red_marbles : ℕ := 1200

/-- The number of black marbles in the box -/
def black_marbles : ℕ := 800

/-- The total number of marbles in the box -/
def total_marbles : ℕ := red_marbles + black_marbles

/-- The probability of drawing two marbles of the same color -/
def prob_same_color : ℚ :=
  (Nat.choose red_marbles 2 + Nat.choose black_marbles 2) / Nat.choose total_marbles 2

/-- The probability of drawing two marbles of different colors -/
def prob_diff_color : ℚ :=
  (red_marbles * black_marbles) / Nat.choose total_marbles 2

/-- Theorem stating the absolute difference between the probabilities -/
theorem marble_prob_diff :
  |prob_same_color - prob_diff_color| = 7900 / 199900 := by
  sorry


end NUMINAMATH_CALUDE_marble_prob_diff_l4127_412782


namespace NUMINAMATH_CALUDE_faye_pencil_count_l4127_412759

/-- The number of rows of pencils and crayons --/
def num_rows : ℕ := 30

/-- The number of pencils in each row --/
def pencils_per_row : ℕ := 24

/-- The total number of pencils --/
def total_pencils : ℕ := num_rows * pencils_per_row

theorem faye_pencil_count : total_pencils = 720 := by
  sorry

end NUMINAMATH_CALUDE_faye_pencil_count_l4127_412759


namespace NUMINAMATH_CALUDE_quadratic_equation_from_means_l4127_412739

theorem quadratic_equation_from_means (a b : ℝ) : 
  (a + b) / 2 = 5 → 
  Real.sqrt (a * b) = 15 → 
  ∃ (x : ℝ), x^2 - 10*x + 225 = 0 ↔ (x = a ∨ x = b) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_from_means_l4127_412739


namespace NUMINAMATH_CALUDE_prob_no_consecutive_heads_is_half_l4127_412752

/-- The probability of heads not appearing consecutively when tossing a fair coin four times -/
def prob_no_consecutive_heads : ℚ := 1/2

/-- A fair coin is tossed four times -/
def num_tosses : ℕ := 4

/-- The total number of possible outcomes when tossing a fair coin four times -/
def total_outcomes : ℕ := 2^num_tosses

/-- The number of outcomes where heads do not appear consecutively -/
def favorable_outcomes : ℕ := 8

theorem prob_no_consecutive_heads_is_half :
  prob_no_consecutive_heads = favorable_outcomes / total_outcomes :=
by sorry

end NUMINAMATH_CALUDE_prob_no_consecutive_heads_is_half_l4127_412752


namespace NUMINAMATH_CALUDE_farm_plot_length_l4127_412722

/-- Proves that a rectangular plot with given width and area has a specific length -/
theorem farm_plot_length (width : ℝ) (area_acres : ℝ) (acre_sq_ft : ℝ) :
  width = 1210 →
  area_acres = 10 →
  acre_sq_ft = 43560 →
  (area_acres * acre_sq_ft) / width = 360 := by
  sorry

end NUMINAMATH_CALUDE_farm_plot_length_l4127_412722


namespace NUMINAMATH_CALUDE_machinery_cost_l4127_412733

def total_amount : ℝ := 7428.57
def raw_materials : ℝ := 5000
def cash_percentage : ℝ := 0.30

theorem machinery_cost :
  ∃ (machinery : ℝ),
    machinery = total_amount - raw_materials - (cash_percentage * total_amount) ∧
    machinery = 200 := by
  sorry

end NUMINAMATH_CALUDE_machinery_cost_l4127_412733


namespace NUMINAMATH_CALUDE_bottle_production_time_l4127_412728

/-- Given that 6 identical machines produce 270 bottles per minute at a constant rate,
    prove that 5 such machines will take 4 minutes to produce 900 bottles. -/
theorem bottle_production_time (rate : ℕ) (h1 : 6 * rate = 270) : 
  (900 : ℕ) / (5 * rate) = 4 := by
  sorry

end NUMINAMATH_CALUDE_bottle_production_time_l4127_412728


namespace NUMINAMATH_CALUDE_susan_peaches_in_knapsack_l4127_412764

/-- The number of peaches Susan bought -/
def total_peaches : ℕ := 5 * 12

/-- The number of cloth bags Susan has -/
def num_cloth_bags : ℕ := 2

/-- Represents the relationship between peaches in cloth bags and knapsack -/
def knapsack_ratio : ℚ := 1 / 2

/-- The number of peaches in the knapsack -/
def peaches_in_knapsack : ℕ := 12

theorem susan_peaches_in_knapsack :
  ∃ (x : ℕ), 
    (x : ℚ) * num_cloth_bags + (x : ℚ) * knapsack_ratio = total_peaches ∧
    peaches_in_knapsack = (x : ℚ) * knapsack_ratio := by
  sorry

end NUMINAMATH_CALUDE_susan_peaches_in_knapsack_l4127_412764


namespace NUMINAMATH_CALUDE_upstream_downstream_time_difference_l4127_412740

/-- Proves that the difference in time between traveling upstream and downstream is 90 minutes -/
theorem upstream_downstream_time_difference 
  (distance : ℝ) 
  (boat_speed : ℝ) 
  (stream_speed : ℝ) 
  (h1 : distance = 36) 
  (h2 : boat_speed = 10) 
  (h3 : stream_speed = 2) : 
  (distance / (boat_speed - stream_speed) - distance / (boat_speed + stream_speed)) * 60 = 90 := by
  sorry

#check upstream_downstream_time_difference

end NUMINAMATH_CALUDE_upstream_downstream_time_difference_l4127_412740


namespace NUMINAMATH_CALUDE_sphere_radius_ratio_l4127_412798

/-- The maximum ratio of the radius of the third sphere to the radius of the first sphere
    in a specific geometric configuration. -/
theorem sphere_radius_ratio (r x : ℝ) (h1 : r > 0) (h2 : x > 0) : 
  let R := 3 * r
  let t := x / r
  let α := π / 3
  let cone_height := R / 2
  let slant_height := R
  (∀ θ : ℝ, 0 ≤ θ ∧ θ ≤ π / 3 → 
    2 * Real.cos θ ≤ (3 - 2*t) / Real.sqrt (t^2 + 2*t)) →
  (3 * t^2 - 14 * t + 9 = 0) →
  t ≤ 3 / 2 →
  t = (7 - Real.sqrt 22) / 3 := by
sorry

end NUMINAMATH_CALUDE_sphere_radius_ratio_l4127_412798


namespace NUMINAMATH_CALUDE_isosceles_triangle_x_values_l4127_412708

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Calculates the square of the distance between two points in 3D space -/
def distanceSquared (p1 p2 : Point3D) : ℝ :=
  (p1.x - p2.x)^2 + (p1.y - p2.y)^2 + (p1.z - p2.z)^2

/-- Theorem: In an isosceles triangle ABC with vertices A(4, 1, 9), B(10, -1, 6), 
    and C(x, 4, 3), where BC is the base, the possible values of x are 2 and 6 -/
theorem isosceles_triangle_x_values :
  let A : Point3D := ⟨4, 1, 9⟩
  let B : Point3D := ⟨10, -1, 6⟩
  let C : Point3D := ⟨x, 4, 3⟩
  (distanceSquared A B = distanceSquared A C) → (x = 2 ∨ x = 6) :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_x_values_l4127_412708


namespace NUMINAMATH_CALUDE_some_number_value_l4127_412790

theorem some_number_value (x y n : ℝ) 
  (h1 : x / (2 * y) = 3 / n) 
  (h2 : (7 * x + 2 * y) / (x - 2 * y) = 23) : 
  n = 2 := by
  sorry

end NUMINAMATH_CALUDE_some_number_value_l4127_412790


namespace NUMINAMATH_CALUDE_quadratic_inequality_l4127_412781

/-- Represents a quadratic function of the form f(x) = -2ax^2 + ax - 4 where a > 0 -/
def QuadraticFunction (a : ℝ) : ℝ → ℝ := 
  fun x ↦ -2 * a * x^2 + a * x - 4

theorem quadratic_inequality (a : ℝ) (ha : a > 0) :
  let f := QuadraticFunction a
  f 2 < f (-1) ∧ f (-1) < f 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l4127_412781
