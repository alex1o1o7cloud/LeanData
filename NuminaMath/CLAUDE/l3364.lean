import Mathlib

namespace NUMINAMATH_CALUDE_james_flowers_per_day_l3364_336447

theorem james_flowers_per_day 
  (total_volunteers : ℕ) 
  (days_worked : ℕ) 
  (total_flowers : ℕ) 
  (h1 : total_volunteers = 5)
  (h2 : days_worked = 2)
  (h3 : total_flowers = 200)
  (h4 : total_flowers % (total_volunteers * days_worked) = 0) :
  total_flowers / (total_volunteers * days_worked) = 20 := by
sorry

end NUMINAMATH_CALUDE_james_flowers_per_day_l3364_336447


namespace NUMINAMATH_CALUDE_blue_paint_gallons_l3364_336473

theorem blue_paint_gallons (total : ℕ) (white : ℕ) (blue : ℕ) :
  total = 6689 →
  white + blue = total →
  8 * white = 5 * blue →
  blue = 4116 := by
sorry

end NUMINAMATH_CALUDE_blue_paint_gallons_l3364_336473


namespace NUMINAMATH_CALUDE_spherical_segment_max_volume_l3364_336419

/-- Given a spherical segment with surface area S, its maximum volume V is S √(S / (18π)) -/
theorem spherical_segment_max_volume (S : ℝ) (h : S > 0) :
  ∃ V : ℝ, V = S * Real.sqrt (S / (18 * Real.pi)) ∧
  ∀ (V' : ℝ), (∃ (R h : ℝ), R > 0 ∧ h > 0 ∧ h ≤ 2*R ∧ S = 2 * Real.pi * R * h ∧
                V' = Real.pi * h^2 * (3*R - h) / 3) →
  V' ≤ V :=
sorry

end NUMINAMATH_CALUDE_spherical_segment_max_volume_l3364_336419


namespace NUMINAMATH_CALUDE_intersection_point_y_coordinate_l3364_336442

-- Define the parabola
def parabola (x : ℝ) : ℝ := 2 * x^2

-- Define the slope of the tangent at a point
def tangent_slope (x : ℝ) : ℝ := 4 * x

-- Define the condition for perpendicular tangents
def perpendicular_tangents (a b : ℝ) : Prop :=
  tangent_slope a * tangent_slope b = -1

-- Define the y-coordinate of the intersection point
def intersection_y (a b : ℝ) : ℝ := 2 * a * b

-- Theorem statement
theorem intersection_point_y_coordinate 
  (a b : ℝ) 
  (ha : parabola a = 2 * a^2) 
  (hb : parabola b = 2 * b^2) 
  (hperp : perpendicular_tangents a b) :
  intersection_y a b = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_y_coordinate_l3364_336442


namespace NUMINAMATH_CALUDE_average_first_five_subjects_l3364_336476

/-- Given the average marks for 6 subjects and the marks for the 6th subject,
    calculate the average marks for the first 5 subjects. -/
theorem average_first_five_subjects
  (total_subjects : ℕ)
  (average_six_subjects : ℚ)
  (marks_sixth_subject : ℕ)
  (h1 : total_subjects = 6)
  (h2 : average_six_subjects = 78)
  (h3 : marks_sixth_subject = 98) :
  (average_six_subjects * total_subjects - marks_sixth_subject) / (total_subjects - 1) = 74 :=
by sorry

end NUMINAMATH_CALUDE_average_first_five_subjects_l3364_336476


namespace NUMINAMATH_CALUDE_pyramid_equal_volume_division_l3364_336425

theorem pyramid_equal_volume_division (m : ℝ) (hm : m > 0) :
  ∃ (x y z : ℝ),
    x > 0 ∧ y > 0 ∧ z > 0 ∧
    x + y + z = m ∧
    x^3 = (1/3) * m^3 ∧
    (x + y)^3 = (2/3) * m^3 ∧
    x = m / Real.rpow 3 (1/3) ∧
    y = (m / Real.rpow 3 (1/3)) * (Real.rpow 2 (1/3) - 1) ∧
    z = m * (1 - Real.rpow (2/3) (1/3)) :=
by sorry

end NUMINAMATH_CALUDE_pyramid_equal_volume_division_l3364_336425


namespace NUMINAMATH_CALUDE_adjacent_knights_probability_l3364_336495

def number_of_knights : ℕ := 30
def chosen_knights : ℕ := 4

def probability_adjacent_knights : ℚ :=
  1 - (number_of_knights - chosen_knights + 1) * (number_of_knights - chosen_knights - 1) * (number_of_knights - chosen_knights - 3) * (number_of_knights - chosen_knights - 5) / (number_of_knights.choose chosen_knights)

theorem adjacent_knights_probability :
  probability_adjacent_knights = 53 / 85 := by sorry

end NUMINAMATH_CALUDE_adjacent_knights_probability_l3364_336495


namespace NUMINAMATH_CALUDE_count_special_divisors_l3364_336466

/-- The number of positive integer divisors of 998^49999 that are not divisors of 998^49998 -/
def special_divisors : ℕ := 99999

/-- 998 as a product of its prime factors -/
def factor_998 : ℕ × ℕ := (2, 499)

theorem count_special_divisors :
  (factor_998.1 * factor_998.2)^49999 = 998^49999 →
  (∃ (d : ℕ → ℕ × ℕ),
    (∀ (n : ℕ), n < special_divisors →
      (factor_998.1^(d n).1 * factor_998.2^(d n).2 ∣ 998^49999) ∧
      ¬(factor_998.1^(d n).1 * factor_998.2^(d n).2 ∣ 998^49998)) ∧
    (∀ (n m : ℕ), n < special_divisors → m < special_divisors → n ≠ m →
      factor_998.1^(d n).1 * factor_998.2^(d n).2 ≠ factor_998.1^(d m).1 * factor_998.2^(d m).2) ∧
    (∀ (k : ℕ), (k ∣ 998^49999) ∧ ¬(k ∣ 998^49998) →
      ∃ (n : ℕ), n < special_divisors ∧ k = factor_998.1^(d n).1 * factor_998.2^(d n).2)) :=
by sorry

end NUMINAMATH_CALUDE_count_special_divisors_l3364_336466


namespace NUMINAMATH_CALUDE_ninth_grader_wins_l3364_336410

/-- Represents the grade of a student -/
inductive Grade
| Ninth
| Tenth

/-- Represents a chess tournament with ninth and tenth graders -/
structure ChessTournament where
  ninth_graders : ℕ
  tenth_graders : ℕ
  ninth_points : ℕ
  tenth_points : ℕ

/-- Chess tournament satisfying the given conditions -/
def valid_tournament (t : ChessTournament) : Prop :=
  t.tenth_graders = 9 * t.ninth_graders ∧
  t.tenth_points = 4 * t.ninth_points

/-- Maximum points a single player can score -/
def max_player_points (t : ChessTournament) (g : Grade) : ℕ :=
  match g with
  | Grade.Ninth => t.tenth_graders
  | Grade.Tenth => (t.tenth_graders - 1) / 2

/-- Theorem stating that a ninth grader wins the tournament with 9 points -/
theorem ninth_grader_wins (t : ChessTournament) 
  (h : valid_tournament t) (h_ninth : t.ninth_graders > 0) :
  ∃ (n : ℕ), n = 9 ∧ 
    n = max_player_points t Grade.Ninth ∧ 
    n > max_player_points t Grade.Tenth :=
  sorry

end NUMINAMATH_CALUDE_ninth_grader_wins_l3364_336410


namespace NUMINAMATH_CALUDE_bruce_initial_amount_l3364_336468

def crayons_cost : ℕ := 5 * 5
def books_cost : ℕ := 10 * 5
def calculators_cost : ℕ := 3 * 5
def total_spent : ℕ := crayons_cost + books_cost + calculators_cost
def bags_cost : ℕ := 11 * 10
def initial_amount : ℕ := total_spent + bags_cost

theorem bruce_initial_amount : initial_amount = 200 := by
  sorry

end NUMINAMATH_CALUDE_bruce_initial_amount_l3364_336468


namespace NUMINAMATH_CALUDE_infinitely_many_amiable_squares_l3364_336471

/-- A number is amiable if the set {1,2,...,N} can be partitioned into pairs
    of elements, each pair having the sum of its elements a perfect square. -/
def IsAmiable (N : ℕ) : Prop :=
  ∃ (partition : List (ℕ × ℕ)),
    (∀ (pair : ℕ × ℕ), pair ∈ partition → pair.1 ≤ N ∧ pair.2 ≤ N) ∧
    (∀ n : ℕ, n ≤ N → ∃ (pair : ℕ × ℕ), pair ∈ partition ∧ (n = pair.1 ∨ n = pair.2)) ∧
    (∀ (pair : ℕ × ℕ), pair ∈ partition → ∃ m : ℕ, pair.1 + pair.2 = m^2)

/-- There exist infinitely many amiable numbers which are themselves perfect squares. -/
theorem infinitely_many_amiable_squares :
  ∀ k : ℕ, ∃ N : ℕ, N > k ∧ ∃ m : ℕ, N = m^2 ∧ IsAmiable N :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_amiable_squares_l3364_336471


namespace NUMINAMATH_CALUDE_elvin_internet_charge_l3364_336475

/-- Represents Elvin's monthly telephone bill structure -/
structure MonthlyBill where
  callCharge : ℝ
  internetCharge : ℝ

/-- Calculates the total bill amount -/
def totalBill (bill : MonthlyBill) : ℝ :=
  bill.callCharge + bill.internetCharge

theorem elvin_internet_charge :
  ∀ (jan_bill feb_bill : MonthlyBill),
    totalBill jan_bill = 46 →
    totalBill feb_bill = 76 →
    feb_bill.callCharge = 2 * jan_bill.callCharge →
    jan_bill.internetCharge = feb_bill.internetCharge →
    jan_bill.internetCharge = 16 := by
  sorry

end NUMINAMATH_CALUDE_elvin_internet_charge_l3364_336475


namespace NUMINAMATH_CALUDE_total_books_two_months_l3364_336439

def books_last_month : ℕ := 4

def books_this_month (n : ℕ) : ℕ := 2 * n

theorem total_books_two_months : 
  books_last_month + books_this_month books_last_month = 12 := by
  sorry

end NUMINAMATH_CALUDE_total_books_two_months_l3364_336439


namespace NUMINAMATH_CALUDE_gcd_930_868_l3364_336406

theorem gcd_930_868 : Nat.gcd 930 868 = 62 := by
  sorry

end NUMINAMATH_CALUDE_gcd_930_868_l3364_336406


namespace NUMINAMATH_CALUDE_parabola_transformation_l3364_336485

/-- The original parabola function -/
def original_parabola (x : ℝ) : ℝ := x^2 - 1

/-- The transformed parabola function -/
def transformed_parabola (x : ℝ) : ℝ := (x - 1)^2 + 1

/-- Theorem stating that the transformation of the original parabola
    by shifting 2 units up and 1 unit right results in the transformed parabola -/
theorem parabola_transformation :
  ∀ x : ℝ, transformed_parabola x = original_parabola (x - 1) + 2 :=
by sorry

end NUMINAMATH_CALUDE_parabola_transformation_l3364_336485


namespace NUMINAMATH_CALUDE_ab_power_2013_l3364_336430

theorem ab_power_2013 (a b : ℚ) (h : |a - 2| + (2*b + 1)^2 = 0) : (a*b)^2013 = -1 := by
  sorry

end NUMINAMATH_CALUDE_ab_power_2013_l3364_336430


namespace NUMINAMATH_CALUDE_simplify_fraction_l3364_336488

theorem simplify_fraction (a : ℝ) (h1 : a ≠ 0) (h2 : a ≠ 1) :
  (a - 1 / a) / ((a - 1) / a) = a + 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l3364_336488


namespace NUMINAMATH_CALUDE_kellys_initial_games_prove_kellys_initial_games_l3364_336452

/-- Theorem: Kelly's initial number of Nintendo games
Given that Kelly gives away 250 Nintendo games and has 300 games left,
prove that she initially had 550 games. -/
theorem kellys_initial_games : ℕ → Prop :=
  fun initial_games =>
    let games_given_away : ℕ := 250
    let games_left : ℕ := 300
    initial_games = games_given_away + games_left ∧ initial_games = 550

/-- Proof of Kelly's initial number of Nintendo games -/
theorem prove_kellys_initial_games : ∃ (initial_games : ℕ), kellys_initial_games initial_games :=
  sorry

end NUMINAMATH_CALUDE_kellys_initial_games_prove_kellys_initial_games_l3364_336452


namespace NUMINAMATH_CALUDE_bowling_ball_weight_l3364_336491

theorem bowling_ball_weight :
  ∀ (bowling_ball_weight canoe_weight : ℝ),
    (9 * bowling_ball_weight = 6 * canoe_weight) →
    (4 * canoe_weight = 120) →
    bowling_ball_weight = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_bowling_ball_weight_l3364_336491


namespace NUMINAMATH_CALUDE_solution_correctness_l3364_336416

def solution_set : Set (ℝ × ℝ × ℝ) :=
  {(0, 1, 1), (0, -1, -1), (1, 1, 0), (-1, -1, 0), (1, 0, 1), (-1, 0, -1),
   (Real.sqrt 3 / 3, Real.sqrt 3 / 3, Real.sqrt 3 / 3),
   (-Real.sqrt 3 / 3, -Real.sqrt 3 / 3, -Real.sqrt 3 / 3)}

def satisfies_conditions (a b c : ℝ) : Prop :=
  a^2*b + c = b^2*c + a ∧ 
  b^2*c + a = c^2*a + b ∧
  a*b + b*c + c*a = 1

theorem solution_correctness :
  ∀ (a b c : ℝ), satisfies_conditions a b c ↔ (a, b, c) ∈ solution_set :=
by sorry

end NUMINAMATH_CALUDE_solution_correctness_l3364_336416


namespace NUMINAMATH_CALUDE_sodium_bisulfite_moles_required_l3364_336434

/-- Represents the balanced chemical equation for the reaction --/
structure ChemicalEquation :=
  (NaHSO3 : ℕ)
  (HCl : ℕ)
  (NaCl : ℕ)
  (H2O : ℕ)
  (SO2 : ℕ)

/-- The balanced equation for the reaction --/
def balanced_equation : ChemicalEquation :=
  { NaHSO3 := 1, HCl := 1, NaCl := 1, H2O := 1, SO2 := 1 }

/-- Theorem stating the number of moles of Sodium bisulfite required --/
theorem sodium_bisulfite_moles_required 
  (NaCl_produced : ℕ) 
  (HCl_used : ℕ) 
  (h1 : NaCl_produced = 2) 
  (h2 : HCl_used = 2) 
  (h3 : balanced_equation.NaHSO3 = balanced_equation.HCl) 
  (h4 : balanced_equation.NaHSO3 = balanced_equation.NaCl) :
  NaCl_produced = 2 := by
  sorry

end NUMINAMATH_CALUDE_sodium_bisulfite_moles_required_l3364_336434


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l3364_336480

/-- An arithmetic sequence with given first and last terms -/
structure ArithmeticSequence where
  a₁ : ℚ  -- First term
  a₃₀ : ℚ  -- 30th term
  is_arithmetic : a₃₀ = a₁ + 29 * ((a₃₀ - a₁) / 29)  -- Condition for arithmetic sequence

/-- Properties of the arithmetic sequence -/
theorem arithmetic_sequence_properties (seq : ArithmeticSequence) 
    (h₁ : seq.a₁ = 5)
    (h₂ : seq.a₃₀ = 100) : 
  let d := (seq.a₃₀ - seq.a₁) / 29
  let a₈ := seq.a₁ + 7 * d
  let a₁₅ := seq.a₁ + 14 * d
  let S₁₅ := 15 / 2 * (seq.a₁ + a₁₅)
  (a₈ = 25 + 1 / 29) ∧ (S₁₅ = 393 + 2 / 29) := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l3364_336480


namespace NUMINAMATH_CALUDE_remainder_not_composite_l3364_336444

theorem remainder_not_composite (p : Nat) (h_prime : Nat.Prime p) (h_gt_30 : p > 30) :
  ¬(∃ (a b : Nat), a > 1 ∧ b > 1 ∧ p % 30 = a * b) := by
  sorry

end NUMINAMATH_CALUDE_remainder_not_composite_l3364_336444


namespace NUMINAMATH_CALUDE_cucumber_weight_after_evaporation_l3364_336462

theorem cucumber_weight_after_evaporation 
  (initial_weight : ℝ) 
  (initial_water_percent : ℝ) 
  (final_water_percent : ℝ) 
  (h1 : initial_weight = 100) 
  (h2 : initial_water_percent = 0.99) 
  (h3 : final_water_percent = 0.98) : 
  ∃ (final_weight : ℝ), final_weight = 50 ∧ 
    (1 - initial_water_percent) * initial_weight = 
    (1 - final_water_percent) * final_weight := by
  sorry

end NUMINAMATH_CALUDE_cucumber_weight_after_evaporation_l3364_336462


namespace NUMINAMATH_CALUDE_remainder_of_large_number_l3364_336446

theorem remainder_of_large_number (p : Nat) (h_prime : Nat.Prime p) :
  123456789012 ≡ 71 [MOD p] :=
by
  sorry

end NUMINAMATH_CALUDE_remainder_of_large_number_l3364_336446


namespace NUMINAMATH_CALUDE_man_and_son_work_time_l3364_336486

/-- Given a task that takes a man 4 days and his son 12 days to complete individually, 
    prove that they can complete the task together in 3 days. -/
theorem man_and_son_work_time (task : ℝ) (man_rate son_rate combined_rate : ℝ) : 
  task > 0 ∧ 
  man_rate = task / 4 ∧ 
  son_rate = task / 12 ∧ 
  combined_rate = man_rate + son_rate → 
  task / combined_rate = 3 := by
sorry

end NUMINAMATH_CALUDE_man_and_son_work_time_l3364_336486


namespace NUMINAMATH_CALUDE_absolute_value_sum_zero_l3364_336409

theorem absolute_value_sum_zero (x y : ℝ) :
  |x - 2| + |4 - y| = 0 → x = 2 ∧ y = 4 := by
sorry

end NUMINAMATH_CALUDE_absolute_value_sum_zero_l3364_336409


namespace NUMINAMATH_CALUDE_bella_age_l3364_336417

theorem bella_age (bella_age : ℕ) (brother_age : ℕ) : 
  brother_age = bella_age + 9 →
  bella_age + brother_age = 19 →
  bella_age = 5 := by
sorry

end NUMINAMATH_CALUDE_bella_age_l3364_336417


namespace NUMINAMATH_CALUDE_joan_egg_count_l3364_336458

-- Define the number of dozens Joan bought
def dozen_count : ℕ := 6

-- Define the number of eggs in a dozen
def eggs_per_dozen : ℕ := 12

-- Theorem to prove
theorem joan_egg_count : dozen_count * eggs_per_dozen = 72 := by
  sorry

end NUMINAMATH_CALUDE_joan_egg_count_l3364_336458


namespace NUMINAMATH_CALUDE_resort_worker_period_l3364_336449

theorem resort_worker_period (average_tips : ℝ) (total_period : ℕ) : 
  (6 * average_tips = (1 / 2) * (6 * average_tips + (total_period - 1) * average_tips)) →
  total_period = 7 := by
  sorry

end NUMINAMATH_CALUDE_resort_worker_period_l3364_336449


namespace NUMINAMATH_CALUDE_ellipse_m_value_l3364_336492

/-- Definition of an ellipse with semi-major axis a and semi-minor axis b -/
def is_ellipse (a b : ℝ) (P : ℝ × ℝ) : Prop :=
  (P.1^2 / a^2) + (P.2^2 / b^2) = 1

/-- The sum of distances from any point on an ellipse to its foci is constant -/
axiom ellipse_foci_distance_sum (a b : ℝ) (P F₁ F₂ : ℝ × ℝ) :
  is_ellipse a b P → (dist P F₁ + dist P F₂ = 2 * a)

/-- Theorem: For an ellipse with equation x²/m + y²/16 = 1, 
    if the distances from any point to the foci are 3 and 7, then m = 25 -/
theorem ellipse_m_value (m : ℝ) (P F₁ F₂ : ℝ × ℝ) :
  is_ellipse (Real.sqrt m) 4 P →
  dist P F₁ = 3 →
  dist P F₂ = 7 →
  m = 25 := by sorry

end NUMINAMATH_CALUDE_ellipse_m_value_l3364_336492


namespace NUMINAMATH_CALUDE_hexagon_area_in_circle_l3364_336472

/-- The area of a regular hexagon inscribed in a circle with area 196π square units is 294√3 square units. -/
theorem hexagon_area_in_circle (circle_area : ℝ) (hexagon_area : ℝ) : 
  circle_area = 196 * Real.pi → hexagon_area = 294 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_area_in_circle_l3364_336472


namespace NUMINAMATH_CALUDE_jade_transactions_l3364_336424

theorem jade_transactions (mabel_transactions : ℕ) 
  (anthony_transactions : ℕ) (cal_transactions : ℕ) (jade_transactions : ℕ) : 
  mabel_transactions = 90 →
  anthony_transactions = mabel_transactions + (mabel_transactions * 10 / 100) →
  cal_transactions = anthony_transactions * 2 / 3 →
  jade_transactions = cal_transactions + 16 →
  jade_transactions = 82 := by
sorry

end NUMINAMATH_CALUDE_jade_transactions_l3364_336424


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l3364_336474

/-- The hyperbola C: mx² + ny² = 1 -/
structure Hyperbola where
  m : ℝ
  n : ℝ
  h_mn : m * n < 0

/-- The circle x² + y² - 6x - 2y + 9 = 0 -/
def Circle : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 - 6*p.1 - 2*p.2 + 9 = 0}

/-- The eccentricity of a hyperbola -/
def eccentricity (h : Hyperbola) : ℝ := sorry

/-- The asymptotes of a hyperbola -/
def asymptotes (h : Hyperbola) : Set (Set (ℝ × ℝ)) := sorry

/-- Predicate to check if a line is tangent to a circle -/
def is_tangent_to (line : Set (ℝ × ℝ)) (circle : Set (ℝ × ℝ)) : Prop := sorry

theorem hyperbola_eccentricity (h : Hyperbola) :
  (∃ a ∈ asymptotes h, is_tangent_to a Circle) →
  (eccentricity h = 5/3 ∨ eccentricity h = 5/4) := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l3364_336474


namespace NUMINAMATH_CALUDE_percent_relation_l3364_336407

theorem percent_relation (x y z : ℝ) (h1 : x = 1.3 * y) (h2 : y = 0.6 * z) : 
  x = 0.78 * z := by
  sorry

end NUMINAMATH_CALUDE_percent_relation_l3364_336407


namespace NUMINAMATH_CALUDE_frog_grasshopper_difference_l3364_336459

/-- Represents the jumping distances in the contest -/
structure JumpDistances where
  grasshopper : ℕ
  frog : ℕ
  mouse : ℕ

/-- The conditions of the jumping contest -/
def contest_conditions (j : JumpDistances) : Prop :=
  j.grasshopper = 19 ∧
  j.frog > j.grasshopper ∧
  j.mouse = j.frog + 20 ∧
  j.mouse = j.grasshopper + 30

/-- The theorem stating the difference between the frog's and grasshopper's jump distances -/
theorem frog_grasshopper_difference (j : JumpDistances) 
  (h : contest_conditions j) : j.frog - j.grasshopper = 10 := by
  sorry


end NUMINAMATH_CALUDE_frog_grasshopper_difference_l3364_336459


namespace NUMINAMATH_CALUDE_scientific_notation_of_five_nm_l3364_336479

theorem scientific_notation_of_five_nm :
  ∃ (a : ℝ) (n : ℤ), 0.000000005 = a * 10^n ∧ 1 ≤ a ∧ a < 10 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_five_nm_l3364_336479


namespace NUMINAMATH_CALUDE_min_value_absolute_sum_l3364_336477

theorem min_value_absolute_sum (x : ℝ) : 
  ∃ (m : ℝ), (∀ x, |x - 1| + |x + 2| ≥ m) ∧ (∃ x, |x - 1| + |x + 2| = m) ∧ m = 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_absolute_sum_l3364_336477


namespace NUMINAMATH_CALUDE_age_ratio_in_ten_years_l3364_336489

/-- Represents the age difference between Pete and Claire -/
structure AgeDifference where
  pete : ℕ
  claire : ℕ

/-- The conditions of the problem -/
def age_conditions (ad : AgeDifference) : Prop :=
  ∃ (x : ℕ),
    -- Claire's age 2 years ago
    ad.claire = x + 2 ∧
    -- Pete's age 2 years ago
    ad.pete = 3 * x + 2 ∧
    -- Four years ago condition
    3 * x - 4 = 4 * (x - 4)

/-- The theorem to be proved -/
theorem age_ratio_in_ten_years (ad : AgeDifference) :
  age_conditions ad →
  (ad.pete + 10) / (ad.claire + 10) = 2 := by
sorry

end NUMINAMATH_CALUDE_age_ratio_in_ten_years_l3364_336489


namespace NUMINAMATH_CALUDE_fishing_rod_price_l3364_336490

theorem fishing_rod_price (initial_price : ℝ) (saturday_increase : ℝ) (sunday_discount : ℝ) :
  initial_price = 50 ∧ 
  saturday_increase = 0.2 ∧ 
  sunday_discount = 0.15 →
  initial_price * (1 + saturday_increase) * (1 - sunday_discount) = 51 := by
  sorry

end NUMINAMATH_CALUDE_fishing_rod_price_l3364_336490


namespace NUMINAMATH_CALUDE_jade_rate_ratio_l3364_336467

/-- The "jade rate" for a shape is the constant k in the volume formula V = kD³,
    where D is the characteristic length of the shape. -/
def jade_rate (volume : Real → Real) : Real :=
  volume 1

theorem jade_rate_ratio :
  let sphere_volume (a : Real) := (4 / 3) * Real.pi * (a / 2)^3
  let cylinder_volume (a : Real) := Real.pi * (a / 2)^2 * a
  let cube_volume (a : Real) := a^3
  let k₁ := jade_rate sphere_volume
  let k₂ := jade_rate cylinder_volume
  let k₃ := jade_rate cube_volume
  k₁ / k₂ = (Real.pi / 6) / (Real.pi / 4) ∧ k₂ / k₃ = Real.pi / 4 := by
  sorry


end NUMINAMATH_CALUDE_jade_rate_ratio_l3364_336467


namespace NUMINAMATH_CALUDE_even_function_inequality_l3364_336426

/-- A function f is even if f(-x) = f(x) for all x in its domain -/
def EvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- A function f is increasing on [0, +∞) if f(x) ≤ f(y) for all 0 ≤ x ≤ y -/
def IncreasingOnNonnegative (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 ≤ x → x ≤ y → f x ≤ f y

theorem even_function_inequality (f : ℝ → ℝ) 
  (h_even : EvenFunction f) 
  (h_incr : IncreasingOnNonnegative f) : 
  f (-2) < f 3 ∧ f 3 < f (-Real.pi) := by
  sorry

end NUMINAMATH_CALUDE_even_function_inequality_l3364_336426


namespace NUMINAMATH_CALUDE_ellipse_m_value_l3364_336441

/-- An ellipse with equation x²/(10-m) + y²/(m-2) = 1, major axis along y-axis, and focal length 4 -/
structure Ellipse (m : ℝ) :=
  (eq : ∀ (x y : ℝ), x^2 / (10 - m) + y^2 / (m - 2) = 1)
  (major_axis : m - 2 > 10 - m)
  (focal_length : ℝ)
  (focal_length_eq : focal_length = 4)

/-- The value of m for the given ellipse is 8 -/
theorem ellipse_m_value (e : Ellipse m) : m = 8 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_m_value_l3364_336441


namespace NUMINAMATH_CALUDE_power_of_point_formula_l3364_336400

/-- The power of a point with respect to a circle -/
def power_of_point (d R : ℝ) : ℝ := d^2 - R^2

/-- Theorem: The power of a point with respect to a circle is d^2 - R^2,
    where d is the distance from the point to the center of the circle,
    and R is the radius of the circle. -/
theorem power_of_point_formula (d R : ℝ) :
  power_of_point d R = d^2 - R^2 := by sorry

end NUMINAMATH_CALUDE_power_of_point_formula_l3364_336400


namespace NUMINAMATH_CALUDE_equiangular_polygon_with_specific_angle_ratio_is_decagon_l3364_336433

theorem equiangular_polygon_with_specific_angle_ratio_is_decagon :
  ∀ (n : ℕ) (exterior_angle interior_angle : ℝ),
    n ≥ 3 →
    exterior_angle > 0 →
    interior_angle > 0 →
    exterior_angle + interior_angle = 180 →
    exterior_angle = (1 / 4) * interior_angle →
    360 / exterior_angle = 10 :=
by sorry

end NUMINAMATH_CALUDE_equiangular_polygon_with_specific_angle_ratio_is_decagon_l3364_336433


namespace NUMINAMATH_CALUDE_square_difference_equality_l3364_336438

theorem square_difference_equality : 1012^2 - 992^2 - 1009^2 + 995^2 = 12024 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_equality_l3364_336438


namespace NUMINAMATH_CALUDE_moon_speed_conversion_l3364_336443

/-- Converts a speed from kilometers per second to kilometers per hour -/
def km_per_second_to_km_per_hour (speed_km_per_second : ℝ) : ℝ :=
  speed_km_per_second * 3600

/-- The moon's speed in kilometers per second -/
def moon_speed_km_per_second : ℝ := 0.9

theorem moon_speed_conversion :
  km_per_second_to_km_per_hour moon_speed_km_per_second = 3240 := by
  sorry

end NUMINAMATH_CALUDE_moon_speed_conversion_l3364_336443


namespace NUMINAMATH_CALUDE_sqrt_x_minus_one_real_l3364_336456

theorem sqrt_x_minus_one_real (x : ℝ) : (∃ y : ℝ, y ^ 2 = x - 1) ↔ x ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_x_minus_one_real_l3364_336456


namespace NUMINAMATH_CALUDE_no_valid_labeling_l3364_336403

-- Define a type for the vertices of a tetrahedron
inductive Vertex : Type
| A : Vertex
| B : Vertex
| C : Vertex
| D : Vertex

-- Define a type for the faces of a tetrahedron
inductive Face : Type
| ABC : Face
| ABD : Face
| ACD : Face
| BCD : Face

-- Define a labeling function
def Labeling := Vertex → Fin 4

-- Define a function to get the sum of a face given a labeling
def faceSum (l : Labeling) (f : Face) : Nat :=
  match f with
  | Face.ABC => (l Vertex.A).val + (l Vertex.B).val + (l Vertex.C).val
  | Face.ABD => (l Vertex.A).val + (l Vertex.B).val + (l Vertex.D).val
  | Face.ACD => (l Vertex.A).val + (l Vertex.C).val + (l Vertex.D).val
  | Face.BCD => (l Vertex.B).val + (l Vertex.C).val + (l Vertex.D).val

-- Define a predicate for a valid labeling
def isValidLabeling (l : Labeling) : Prop :=
  (∀ (v1 v2 : Vertex), v1 ≠ v2 → l v1 ≠ l v2) ∧
  (∀ (f1 f2 : Face), faceSum l f1 = faceSum l f2)

-- Theorem: There are no valid labelings
theorem no_valid_labeling : ¬∃ (l : Labeling), isValidLabeling l := by
  sorry


end NUMINAMATH_CALUDE_no_valid_labeling_l3364_336403


namespace NUMINAMATH_CALUDE_ice_cream_arrangement_l3364_336413

theorem ice_cream_arrangement (n : ℕ) (h : n = 6) : Nat.factorial n = 720 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_arrangement_l3364_336413


namespace NUMINAMATH_CALUDE_stockholm_uppsala_distance_l3364_336405

/-- The distance between two cities on a map, in centimeters. -/
def map_distance : ℝ := 45

/-- The scale of the map, representing how many kilometers in reality correspond to one centimeter on the map. -/
def map_scale : ℝ := 20

/-- The actual distance between the two cities, in kilometers. -/
def actual_distance : ℝ := map_distance * map_scale

theorem stockholm_uppsala_distance : actual_distance = 900 := by
  sorry

end NUMINAMATH_CALUDE_stockholm_uppsala_distance_l3364_336405


namespace NUMINAMATH_CALUDE_z_in_third_quadrant_l3364_336414

def i : ℂ := Complex.I

def z : ℂ := i + 2 * i^2 + 3 * i^3

theorem z_in_third_quadrant : 
  Real.sign (z.re) = -1 ∧ Real.sign (z.im) = -1 :=
sorry

end NUMINAMATH_CALUDE_z_in_third_quadrant_l3364_336414


namespace NUMINAMATH_CALUDE_factorize_x_squared_minus_one_l3364_336422

theorem factorize_x_squared_minus_one (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_factorize_x_squared_minus_one_l3364_336422


namespace NUMINAMATH_CALUDE_max_automobiles_on_ferry_l3364_336402

/-- Represents the capacity of the ferry in tons -/
def ferry_capacity : ℝ := 50

/-- Represents the minimum weight of an automobile in pounds -/
def min_auto_weight : ℝ := 1600

/-- Represents the conversion factor from tons to pounds -/
def tons_to_pounds : ℝ := 2000

/-- Theorem stating the maximum number of automobiles that can be loaded onto the ferry -/
theorem max_automobiles_on_ferry :
  ⌊(ferry_capacity * tons_to_pounds) / min_auto_weight⌋ = 62 := by
  sorry

end NUMINAMATH_CALUDE_max_automobiles_on_ferry_l3364_336402


namespace NUMINAMATH_CALUDE_infinite_series_sum_equals_one_l3364_336420

/-- The sum of the infinite series ∑(n=1 to ∞) (3n^2 - 2n + 1) / (n^4 - n^3 + n^2 - n + 1) is equal to 1. -/
theorem infinite_series_sum_equals_one :
  let a : ℕ → ℚ := λ n => (3*n^2 - 2*n + 1) / (n^4 - n^3 + n^2 - n + 1)
  ∑' n, a n = 1 := by
  sorry

end NUMINAMATH_CALUDE_infinite_series_sum_equals_one_l3364_336420


namespace NUMINAMATH_CALUDE_circle_area_greater_than_rectangle_l3364_336457

theorem circle_area_greater_than_rectangle : ∀ (r : ℝ), r = 1 →
  π * r^2 ≥ 1 * 2.4 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_greater_than_rectangle_l3364_336457


namespace NUMINAMATH_CALUDE_range_of_a_l3364_336435

theorem range_of_a (x : ℝ) (a : ℝ) : 
  (∀ x, (0 < x ∧ x < a) → (|x - 2| < 3)) ∧ 
  (∃ x, |x - 2| < 3 ∧ ¬(0 < x ∧ x < a)) ∧
  (a > 0) →
  (0 < a ∧ a ≤ 5) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l3364_336435


namespace NUMINAMATH_CALUDE_correct_article_usage_l3364_336448

/-- Represents the possible article choices for each blank -/
inductive Article
  | A
  | The
  | None

/-- Represents the sentence structure with two article blanks -/
structure Sentence where
  first_blank : Article
  second_blank : Article

/-- Defines the correct article usage based on the given conditions -/
def correct_usage : Sentence :=
  { first_blank := Article.A,  -- Gottlieb Daimler is referred to generally
    second_blank := Article.The }  -- The car invention is referred to specifically

/-- Theorem stating that the correct usage is "a" for the first blank and "the" for the second -/
theorem correct_article_usage :
  correct_usage = { first_blank := Article.A, second_blank := Article.The } :=
by sorry

end NUMINAMATH_CALUDE_correct_article_usage_l3364_336448


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l3364_336494

theorem quadratic_equation_roots (c : ℝ) (x₁ x₂ : ℝ) : 
  (∀ x, x^2 + 2*x + c = 0 ↔ x = x₁ ∨ x = x₂) →
  x₁^2 + x₂^2 = c^2 - 2*c →
  c = -2 ∧ x₁ = -1 + Real.sqrt 3 ∧ x₂ = -1 - Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l3364_336494


namespace NUMINAMATH_CALUDE_expected_digits_20_sided_die_l3364_336401

/-- The expected number of digits when rolling a fair 20-sided die with numbers 1 to 20 -/
theorem expected_digits_20_sided_die : 
  let die_faces : Finset ℕ := Finset.range 20
  let one_digit_count : ℕ := (die_faces.filter (λ n => n < 10)).card
  let two_digit_count : ℕ := (die_faces.filter (λ n => n ≥ 10)).card
  let total_faces : ℕ := die_faces.card
  let expected_value : ℚ := (one_digit_count * 1 + two_digit_count * 2) / total_faces
  expected_value = 31 / 20 := by
sorry

end NUMINAMATH_CALUDE_expected_digits_20_sided_die_l3364_336401


namespace NUMINAMATH_CALUDE_min_value_of_expression_l3364_336415

theorem min_value_of_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  (x^2 / (x + 2) + y^2 / (y + 1)) ≥ 1/4 ∧ 
  ∃ x₀ y₀, x₀ > 0 ∧ y₀ > 0 ∧ x₀ + y₀ = 1 ∧ x₀^2 / (x₀ + 2) + y₀^2 / (y₀ + 1) = 1/4 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l3364_336415


namespace NUMINAMATH_CALUDE_largest_non_prime_sequence_l3364_336493

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

theorem largest_non_prime_sequence :
  ∃ (start : ℕ),
    (∀ i ∈ Finset.range 7, 
      let n := start + i
      10 ≤ n ∧ n < 40 ∧ ¬(is_prime n)) ∧
    (∀ j ≥ start + 7, 
      ¬(∀ i ∈ Finset.range 7, 
        let n := j + i
        10 ≤ n ∧ n < 40 ∧ ¬(is_prime n))) →
  start + 6 = 32 :=
sorry

end NUMINAMATH_CALUDE_largest_non_prime_sequence_l3364_336493


namespace NUMINAMATH_CALUDE_sandwich_cost_is_three_l3364_336429

/-- The cost of a sandwich given the total cost and number of items. -/
def sandwich_cost (total_cost : ℚ) (water_cost : ℚ) (num_sandwiches : ℕ) : ℚ :=
  (total_cost - water_cost) / num_sandwiches

/-- Theorem stating that the cost of each sandwich is 3 given the problem conditions. -/
theorem sandwich_cost_is_three :
  sandwich_cost 11 2 3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sandwich_cost_is_three_l3364_336429


namespace NUMINAMATH_CALUDE_power_sum_equality_l3364_336460

theorem power_sum_equality (a b c d : ℝ) 
  (sum_eq : a + b = c + d) 
  (cube_sum_eq : a^3 + b^3 = c^3 + d^3) : 
  (a^5 + b^5 = c^5 + d^5) ∧ 
  ∃ (a b c d : ℝ), (a + b = c + d) ∧ (a^3 + b^3 = c^3 + d^3) ∧ (a^4 + b^4 ≠ c^4 + d^4) :=
by sorry

end NUMINAMATH_CALUDE_power_sum_equality_l3364_336460


namespace NUMINAMATH_CALUDE_log_inequality_l3364_336437

theorem log_inequality (a b c : ℝ) : 
  a = Real.log 3 / Real.log 4 →
  b = Real.log 4 / Real.log 3 →
  c = Real.log 3 / Real.log 5 →
  b > a ∧ a > c :=
by sorry

end NUMINAMATH_CALUDE_log_inequality_l3364_336437


namespace NUMINAMATH_CALUDE_probability_at_least_two_same_l3364_336455

def num_dice : ℕ := 8
def num_sides : ℕ := 8

theorem probability_at_least_two_same :
  let total_outcomes := num_sides ^ num_dice
  let all_different := Nat.factorial num_sides
  (1 - (all_different : ℚ) / total_outcomes) = 2043 / 2048 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_two_same_l3364_336455


namespace NUMINAMATH_CALUDE_share_price_increase_l3364_336427

/-- 
Proves that a 30% increase followed by a 15.384615384615374% increase 
results in a 50% total increase
-/
theorem share_price_increase (initial_price : ℝ) : 
  let first_quarter_increase := 0.30
  let second_quarter_increase := 0.15384615384615374
  let first_quarter_price := initial_price * (1 + first_quarter_increase)
  let second_quarter_price := first_quarter_price * (1 + second_quarter_increase)
  second_quarter_price = initial_price * 1.50 := by
  sorry

end NUMINAMATH_CALUDE_share_price_increase_l3364_336427


namespace NUMINAMATH_CALUDE_custom_equation_solution_l3364_336451

-- Define the custom operation *
def star (a b : ℝ) : ℝ := a * b + a + b

-- State the theorem
theorem custom_equation_solution :
  ∀ x : ℝ, star 3 x = 27 → x = 6 := by
  sorry

end NUMINAMATH_CALUDE_custom_equation_solution_l3364_336451


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l3364_336412

theorem complex_magnitude_problem (z : ℂ) (h : z * (1 + Complex.I) = 2 * Complex.I) :
  Complex.abs (z + 2 * Complex.I) = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l3364_336412


namespace NUMINAMATH_CALUDE_edward_candy_purchase_l3364_336482

def whack_a_mole_tickets : ℕ := 3
def skee_ball_tickets : ℕ := 5
def candy_cost : ℕ := 4

theorem edward_candy_purchase :
  (whack_a_mole_tickets + skee_ball_tickets) / candy_cost = 2 := by
  sorry

end NUMINAMATH_CALUDE_edward_candy_purchase_l3364_336482


namespace NUMINAMATH_CALUDE_tims_number_l3364_336496

theorem tims_number (n : ℕ) : 
  (∃ k l : ℕ, n = 9 * k - 2 ∧ n = 8 * l - 4) ∧ 
  n < 150 ∧ 
  (∀ m : ℕ, (∃ p q : ℕ, m = 9 * p - 2 ∧ m = 8 * q - 4) ∧ m < 150 → m ≤ n) →
  n = 124 := by
sorry

end NUMINAMATH_CALUDE_tims_number_l3364_336496


namespace NUMINAMATH_CALUDE_problem_statements_l3364_336436

theorem problem_statements :
  (({0} : Set ℕ) ⊆ Set.univ) ∧
  (∀ (α : Type) (A B : Set α) (x : α), x ∈ A ∩ B → x ∈ A ∪ B) ∧
  (∃ (a b : ℝ), b^2 < a^2 ∧ ¬(a < b ∧ b < 0)) ∧
  (¬(∀ (x : ℤ), x^2 > 0) ↔ ∃ (x : ℤ), x^2 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_problem_statements_l3364_336436


namespace NUMINAMATH_CALUDE_four_digit_integer_problem_l3364_336445

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def digit_sum (n : ℕ) : ℕ :=
  (n / 1000) + ((n / 100) % 10) + ((n / 10) % 10) + (n % 10)

def middle_digit_sum (n : ℕ) : ℕ :=
  ((n / 100) % 10) + ((n / 10) % 10)

def thousands_minus_units (n : ℕ) : ℤ :=
  (n / 1000 : ℤ) - (n % 10 : ℤ)

theorem four_digit_integer_problem (n : ℕ) 
  (h1 : is_four_digit n)
  (h2 : digit_sum n = 18)
  (h3 : middle_digit_sum n = 11)
  (h4 : thousands_minus_units n = 1)
  (h5 : n % 11 = 0) :
  n = 4653 := by
sorry

end NUMINAMATH_CALUDE_four_digit_integer_problem_l3364_336445


namespace NUMINAMATH_CALUDE_dream_star_results_l3364_336453

/-- Represents the results of a football team in a league --/
structure TeamResults where
  games_played : ℕ
  games_won : ℕ
  games_drawn : ℕ
  games_lost : ℕ
  points : ℕ

/-- Calculates the points earned by a team based on their results --/
def calculate_points (r : TeamResults) : ℕ :=
  3 * r.games_won + r.games_drawn

/-- Theorem stating the unique solution for the given problem --/
theorem dream_star_results :
  ∃! r : TeamResults,
    r.games_played = 9 ∧
    r.games_lost = 2 ∧
    r.points = 17 ∧
    r.games_played = r.games_won + r.games_drawn + r.games_lost ∧
    r.points = calculate_points r ∧
    r.games_won = 5 ∧
    r.games_drawn = 2 := by
  sorry

#check dream_star_results

end NUMINAMATH_CALUDE_dream_star_results_l3364_336453


namespace NUMINAMATH_CALUDE_volume_of_specific_prism_l3364_336423

/-- Right triangular prism ABC-A₁B₁C₁ -/
structure RightTriangularPrism where
  -- Base triangle
  AB : ℝ
  BC : ℝ
  AC : ℝ
  -- Circumscribed sphere
  sphereSurfaceArea : ℝ

/-- Volume of a right triangular prism -/
def prismVolume (p : RightTriangularPrism) : ℝ := sorry

theorem volume_of_specific_prism :
  let p : RightTriangularPrism := {
    AB := 2,
    BC := 2,
    AC := 2 * Real.sqrt 3,
    sphereSurfaceArea := 32 * Real.pi
  }
  prismVolume p = 4 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_volume_of_specific_prism_l3364_336423


namespace NUMINAMATH_CALUDE_arithmetic_sequence_max_a5_l3364_336404

theorem arithmetic_sequence_max_a5 (a : ℕ → ℝ) (s : ℕ → ℝ) :
  (∀ n, s n = (n * (2 * a 1 + (n - 1) * (a 2 - a 1))) / 2) →
  (∀ n, a n = a 1 + (n - 1) * (a 2 - a 1)) →
  s 2 ≥ 4 →
  s 4 ≤ 16 →
  a 5 ≤ 9 ∧ ∃ (a' : ℕ → ℝ), (∀ n, s n = (n * (2 * a' 1 + (n - 1) * (a' 2 - a' 1))) / 2) ∧
                             (∀ n, a' n = a' 1 + (n - 1) * (a' 2 - a' 1)) ∧
                             s 2 ≥ 4 ∧
                             s 4 ≤ 16 ∧
                             a' 5 = 9 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_max_a5_l3364_336404


namespace NUMINAMATH_CALUDE_exists_number_with_specific_digit_sum_l3364_336499

/-- Sum of digits of a natural number -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- Theorem stating the existence of a number with specific digit sum properties -/
theorem exists_number_with_specific_digit_sum : 
  ∃ m : ℕ, digit_sum m = 1990 ∧ digit_sum (m^2) = 1990^2 := by sorry

end NUMINAMATH_CALUDE_exists_number_with_specific_digit_sum_l3364_336499


namespace NUMINAMATH_CALUDE_remainder_of_29_times_182_power_1000_mod_13_l3364_336461

theorem remainder_of_29_times_182_power_1000_mod_13 : 
  (29 * 182^1000) % 13 = 0 := by
sorry

end NUMINAMATH_CALUDE_remainder_of_29_times_182_power_1000_mod_13_l3364_336461


namespace NUMINAMATH_CALUDE_binomial_coefficient_congruence_l3364_336450

theorem binomial_coefficient_congruence 
  (p a b : ℕ) 
  (hp : Nat.Prime p) 
  (hp_pos : p > 0) 
  (hab : a > b) 
  (hb_pos : b > 0) : 
  Nat.choose (p * a) (p * b) ≡ Nat.choose a b [MOD p] := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_congruence_l3364_336450


namespace NUMINAMATH_CALUDE_product_of_roots_zero_l3364_336483

theorem product_of_roots_zero (a b c : ℝ) : 
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  a^3 - 4*a = 0 →
  b^3 - 4*b = 0 →
  c^3 - 4*c = 0 →
  a * b * c = 0 := by
sorry

end NUMINAMATH_CALUDE_product_of_roots_zero_l3364_336483


namespace NUMINAMATH_CALUDE_complex_square_eq_143_minus_48i_l3364_336481

theorem complex_square_eq_143_minus_48i :
  ∀ z : ℂ, z^2 = -143 - 48*I ↔ z = 2 - 12*I ∨ z = -2 + 12*I := by
  sorry

end NUMINAMATH_CALUDE_complex_square_eq_143_minus_48i_l3364_336481


namespace NUMINAMATH_CALUDE_cube_root_two_identity_l3364_336498

theorem cube_root_two_identity (x : ℝ) (h : 32 = x^6 + 1/x^6) :
  x^2 + 1/x^2 = 2 * Real.rpow 2 (1/3) := by
  sorry

end NUMINAMATH_CALUDE_cube_root_two_identity_l3364_336498


namespace NUMINAMATH_CALUDE_smallest_valid_amount_l3364_336465

/-- Represents the number of bags -/
def num_bags : List Nat := [8, 7, 6]

/-- Represents the types of currency -/
inductive Currency
| Dollar
| HalfDollar
| QuarterDollar

/-- Checks if a given amount can be equally distributed into the specified number of bags for all currency types -/
def is_valid_distribution (amount : Nat) (bags : Nat) : Prop :=
  ∀ c : Currency, ∃ n : Nat, n * bags = amount

/-- The main theorem stating the smallest valid amount -/
theorem smallest_valid_amount :
  (∀ bags ∈ num_bags, is_valid_distribution 294 bags) ∧
  (∀ amount < 294, ¬(∀ bags ∈ num_bags, is_valid_distribution amount bags)) :=
sorry

end NUMINAMATH_CALUDE_smallest_valid_amount_l3364_336465


namespace NUMINAMATH_CALUDE_quadratic_inequality_l3364_336411

theorem quadratic_inequality (x : ℝ) : -3 * x^2 + 5 * x + 4 < 0 ↔ -4/3 < x ∧ x < 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l3364_336411


namespace NUMINAMATH_CALUDE_water_usage_problem_l3364_336470

/-- Calculates the water charge based on usage --/
def water_charge (usage : ℕ) : ℚ :=
  if usage ≤ 24 then 1.8 * usage
  else 1.8 * 24 + 4 * (usage - 24)

/-- Represents the water usage problem --/
theorem water_usage_problem :
  ∃ (zhang_usage wang_usage : ℕ),
    zhang_usage > 24 ∧
    wang_usage ≤ 24 ∧
    water_charge zhang_usage - water_charge wang_usage = 19.2 ∧
    zhang_usage = 27 ∧
    wang_usage = 20 :=
by
  sorry

#eval water_charge 27  -- Should output 55.2
#eval water_charge 20  -- Should output 36

end NUMINAMATH_CALUDE_water_usage_problem_l3364_336470


namespace NUMINAMATH_CALUDE_johns_final_elevation_l3364_336440

/-- Calculates the final elevation after descending for a given time. -/
def finalElevation (startElevation : ℝ) (descentRate : ℝ) (time : ℝ) : ℝ :=
  startElevation - descentRate * time

/-- Proves that John's final elevation is 350 feet. -/
theorem johns_final_elevation :
  let startElevation : ℝ := 400
  let descentRate : ℝ := 10
  let time : ℝ := 5
  finalElevation startElevation descentRate time = 350 := by
  sorry

end NUMINAMATH_CALUDE_johns_final_elevation_l3364_336440


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3364_336454

theorem inequality_solution_set (x : ℝ) : 
  (x^2 - 2*x - 3)*(x^2 + 1) < 0 ↔ -1 < x ∧ x < 3 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3364_336454


namespace NUMINAMATH_CALUDE_ticket_difference_l3364_336484

theorem ticket_difference (initial_tickets : ℕ) (remaining_tickets : ℕ) : 
  initial_tickets = 48 → remaining_tickets = 32 → initial_tickets - remaining_tickets = 16 := by
  sorry

end NUMINAMATH_CALUDE_ticket_difference_l3364_336484


namespace NUMINAMATH_CALUDE_triangle_side_length_l3364_336421

noncomputable def f (x : ℝ) := Real.sin (7 * Real.pi / 6 - 2 * x) - 2 * Real.sin x ^ 2 + 1

theorem triangle_side_length 
  (A B C : ℝ) 
  (a b c : ℝ) 
  (h1 : f A = 1/2)
  (h2 : b - a = c - b)  -- arithmetic sequence condition
  (h3 : b * c * Real.cos A = 9) : 
  a = 3 * Real.sqrt 2 := by 
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3364_336421


namespace NUMINAMATH_CALUDE_egg_roll_count_l3364_336478

/-- The number of egg rolls Omar rolled -/
def omar_rolls : ℕ := 219

/-- The number of egg rolls Karen rolled -/
def karen_rolls : ℕ := 229

/-- The total number of egg rolls rolled by Omar and Karen -/
def total_rolls : ℕ := omar_rolls + karen_rolls

theorem egg_roll_count : total_rolls = 448 := by sorry

end NUMINAMATH_CALUDE_egg_roll_count_l3364_336478


namespace NUMINAMATH_CALUDE_age_difference_l3364_336487

-- Define the ages
def katie_daughter_age : ℕ := 12
def lavinia_daughter_age : ℕ := katie_daughter_age - 10
def lavinia_son_age : ℕ := 2 * katie_daughter_age

-- Theorem statement
theorem age_difference : lavinia_son_age - lavinia_daughter_age = 22 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l3364_336487


namespace NUMINAMATH_CALUDE_box_dimensions_sum_l3364_336408

/-- Given a rectangular box with dimensions A, B, and C, prove that if the surface areas of its faces
    are 30, 30, 60, 60, 90, and 90 square units, then A + B + C = 24. -/
theorem box_dimensions_sum (A B C : ℝ) : 
  A > 0 → B > 0 → C > 0 →
  A * B = 30 →
  A * C = 60 →
  B * C = 90 →
  A + B + C = 24 := by
  sorry

end NUMINAMATH_CALUDE_box_dimensions_sum_l3364_336408


namespace NUMINAMATH_CALUDE_egyptian_fraction_decomposition_l3364_336469

theorem egyptian_fraction_decomposition (n : ℕ) (h : n ≥ 2) :
  (2 : ℚ) / (2 * n + 1) = 1 / (n + 1) + 1 / ((n + 1) * (2 * n + 1)) := by
  sorry

end NUMINAMATH_CALUDE_egyptian_fraction_decomposition_l3364_336469


namespace NUMINAMATH_CALUDE_tan_210_degrees_l3364_336432

theorem tan_210_degrees : Real.tan (210 * π / 180) = -Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_210_degrees_l3364_336432


namespace NUMINAMATH_CALUDE_simplify_square_roots_l3364_336497

theorem simplify_square_roots : Real.sqrt (5 * 3) * Real.sqrt (3^3 * 5^3) = 225 := by
  sorry

end NUMINAMATH_CALUDE_simplify_square_roots_l3364_336497


namespace NUMINAMATH_CALUDE_unit_digit_product_l3364_336418

theorem unit_digit_product : ∃ n : ℕ, (3^68 * 6^59 * 7^71) % 10 = 8 ∧ n = (3^68 * 6^59 * 7^71) := by
  sorry

end NUMINAMATH_CALUDE_unit_digit_product_l3364_336418


namespace NUMINAMATH_CALUDE_bread_slices_proof_l3364_336428

def original_loaf_slices (andy_eaten : ℕ) (slices_per_toast : ℕ) (toast_made : ℕ) (remaining_slice : ℕ) : ℕ :=
  andy_eaten + slices_per_toast * toast_made + remaining_slice

theorem bread_slices_proof :
  original_loaf_slices 6 2 10 1 = 27 := by
  sorry

end NUMINAMATH_CALUDE_bread_slices_proof_l3364_336428


namespace NUMINAMATH_CALUDE_darias_piggy_bank_problem_l3364_336431

/-- The problem of calculating Daria's initial piggy bank balance. -/
theorem darias_piggy_bank_problem
  (vacuum_cost : ℕ)
  (weekly_savings : ℕ)
  (weeks_to_save : ℕ)
  (h1 : vacuum_cost = 120)
  (h2 : weekly_savings = 10)
  (h3 : weeks_to_save = 10)
  (h4 : vacuum_cost = weekly_savings * weeks_to_save + initial_balance) :
  initial_balance = 20 :=
by
  sorry

#check darias_piggy_bank_problem

end NUMINAMATH_CALUDE_darias_piggy_bank_problem_l3364_336431


namespace NUMINAMATH_CALUDE_volume_ratio_cylinder_cone_sphere_l3364_336463

noncomputable def sphere_volume (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3

noncomputable def cylinder_volume (r h : ℝ) : ℝ := Real.pi * r^2 * h

noncomputable def cone_volume (r h : ℝ) : ℝ := (1 / 3) * Real.pi * r^2 * h

theorem volume_ratio_cylinder_cone_sphere (r : ℝ) (h_pos : r > 0) :
  ∃ (k : ℝ), k > 0 ∧ 
    cylinder_volume r (2 * r) = 3 * k ∧
    cone_volume r (2 * r) = k ∧
    sphere_volume r = 2 * k := by
  sorry

end NUMINAMATH_CALUDE_volume_ratio_cylinder_cone_sphere_l3364_336463


namespace NUMINAMATH_CALUDE_motion_analysis_l3364_336464

-- Define the motion law
def s (t : ℝ) : ℝ := 4 * t + t^3

-- Define velocity as the derivative of s
noncomputable def v (t : ℝ) : ℝ := deriv s t

-- Define acceleration as the derivative of v
noncomputable def a (t : ℝ) : ℝ := deriv v t

-- Theorem statement
theorem motion_analysis :
  (∀ t, v t = 4 + 3 * t^2) ∧
  (∀ t, a t = 6 * t) ∧
  (v 0 = 4 ∧ a 0 = 0) ∧
  (v 1 = 7 ∧ a 1 = 6) ∧
  (v 2 = 16 ∧ a 2 = 12) := by sorry

end NUMINAMATH_CALUDE_motion_analysis_l3364_336464
