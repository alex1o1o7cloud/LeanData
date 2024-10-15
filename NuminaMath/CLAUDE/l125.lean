import Mathlib

namespace NUMINAMATH_CALUDE_carlas_daily_collection_l125_12525

/-- Represents the number of items Carla needs to collect each day for her project -/
def daily_collection_amount (leaves bugs days : ℕ) : ℕ :=
  (leaves + bugs) / days

/-- Proves that Carla needs to collect 5 items per day given the project conditions -/
theorem carlas_daily_collection :
  daily_collection_amount 30 20 10 = 5 := by
  sorry

#eval daily_collection_amount 30 20 10

end NUMINAMATH_CALUDE_carlas_daily_collection_l125_12525


namespace NUMINAMATH_CALUDE_birthday_money_ratio_l125_12554

theorem birthday_money_ratio : 
  ∀ (total_money video_game_cost goggles_cost money_left : ℚ),
    total_money = 100 →
    video_game_cost = total_money / 4 →
    money_left = 60 →
    goggles_cost = total_money - video_game_cost - money_left →
    goggles_cost / (total_money - video_game_cost) = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_birthday_money_ratio_l125_12554


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l125_12511

def M : Set ℕ := {0, 1, 3}

def N : Set ℕ := {x | ∃ a ∈ M, x = 3 * a}

theorem intersection_of_M_and_N : M ∩ N = {0, 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l125_12511


namespace NUMINAMATH_CALUDE_female_managers_count_l125_12563

/-- Represents a company with employees and managers -/
structure Company where
  total_employees : ℕ
  total_managers : ℕ
  male_employees : ℕ
  male_managers : ℕ
  female_employees : ℕ
  female_managers : ℕ

/-- The conditions of the company as described in the problem -/
def company_conditions (c : Company) : Prop :=
  c.total_managers = (2 * c.total_employees) / 5 ∧
  c.male_managers = (2 * c.male_employees) / 5 ∧
  c.female_employees = 750

/-- The theorem stating that under the given conditions, the number of female managers is 300 -/
theorem female_managers_count (c : Company) 
  (h : company_conditions c) : c.female_managers = 300 := by
  sorry


end NUMINAMATH_CALUDE_female_managers_count_l125_12563


namespace NUMINAMATH_CALUDE_mary_remaining_sheep_l125_12553

def initial_sheep : ℕ := 400

def sheep_after_sister (initial : ℕ) : ℕ :=
  initial - (initial / 4)

def sheep_after_brother (after_sister : ℕ) : ℕ :=
  after_sister - (after_sister / 2)

theorem mary_remaining_sheep :
  sheep_after_brother (sheep_after_sister initial_sheep) = 150 := by
  sorry

end NUMINAMATH_CALUDE_mary_remaining_sheep_l125_12553


namespace NUMINAMATH_CALUDE_orange_division_l125_12515

theorem orange_division (oranges : ℕ) (friends : ℕ) (pieces_per_friend : ℕ) 
  (h1 : oranges = 80) 
  (h2 : friends = 200) 
  (h3 : pieces_per_friend = 4) : 
  (friends * pieces_per_friend) / oranges = 10 := by
  sorry

end NUMINAMATH_CALUDE_orange_division_l125_12515


namespace NUMINAMATH_CALUDE_parabola_parameter_value_l125_12570

/-- Proves that for a parabola y^2 = 2px (p > 0) with axis of symmetry at distance 4 from the point (3, 0), the value of p is 2. -/
theorem parabola_parameter_value (p : ℝ) (h1 : p > 0) : 
  (∃ (x y : ℝ), y^2 = 2*p*x) →  -- Parabola equation
  (∃ (a : ℝ), ∀ (x y : ℝ), y^2 = 2*p*x → x = a) →  -- Axis of symmetry exists
  (|3 - (- p/2)| = 4) →  -- Distance from (3, 0) to axis of symmetry is 4
  p = 2 := by
sorry

end NUMINAMATH_CALUDE_parabola_parameter_value_l125_12570


namespace NUMINAMATH_CALUDE_top_z_conference_teams_l125_12581

theorem top_z_conference_teams (n : ℕ) : n * (n - 1) / 2 = 45 → n = 10 := by
  sorry

end NUMINAMATH_CALUDE_top_z_conference_teams_l125_12581


namespace NUMINAMATH_CALUDE_litter_count_sum_l125_12557

theorem litter_count_sum : 
  let glass_bottles : ℕ := 25
  let aluminum_cans : ℕ := 18
  let plastic_bags : ℕ := 12
  let paper_cups : ℕ := 7
  let cigarette_packs : ℕ := 5
  let face_masks : ℕ := 3
  glass_bottles + aluminum_cans + plastic_bags + paper_cups + cigarette_packs + face_masks = 70 := by
  sorry

end NUMINAMATH_CALUDE_litter_count_sum_l125_12557


namespace NUMINAMATH_CALUDE_no_rational_multiples_of_pi_l125_12510

theorem no_rational_multiples_of_pi (x y : ℚ) : 
  (∃ (m n : ℚ), x = m * Real.pi ∧ y = n * Real.pi) →
  0 < x → x < y → y < Real.pi / 2 →
  Real.tan x + Real.tan y = 2 →
  False :=
sorry

end NUMINAMATH_CALUDE_no_rational_multiples_of_pi_l125_12510


namespace NUMINAMATH_CALUDE_smallest_four_digit_in_pascal_triangle_l125_12541

def pascal_triangle (n : ℕ) (k : ℕ) : ℕ :=
  Nat.choose n k

def is_in_pascal_triangle (x : ℕ) : Prop :=
  ∃ n k, pascal_triangle n k = x

def is_four_digit (x : ℕ) : Prop :=
  1000 ≤ x ∧ x < 10000

theorem smallest_four_digit_in_pascal_triangle :
  (is_in_pascal_triangle 1000) ∧
  (∀ x, is_in_pascal_triangle x → is_four_digit x → 1000 ≤ x) :=
sorry

end NUMINAMATH_CALUDE_smallest_four_digit_in_pascal_triangle_l125_12541


namespace NUMINAMATH_CALUDE_xy_value_l125_12527

theorem xy_value (x y : ℝ) (h : x * (x + 2*y) = x^2 + 12) : x * y = 6 := by
  sorry

end NUMINAMATH_CALUDE_xy_value_l125_12527


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l125_12535

theorem sqrt_equation_solution (x : ℝ) : Real.sqrt (4 - 2 * x) = 5 → x = 21 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l125_12535


namespace NUMINAMATH_CALUDE_parabola_translation_l125_12577

/-- The original parabola function -/
def original_parabola (x : ℝ) : ℝ := 3 * x^2

/-- The translated parabola function -/
def translated_parabola (x : ℝ) : ℝ := 3 * (x - 1)^2 - 4

/-- Theorem stating that the translated_parabola is the result of
    translating the original_parabola 1 unit right and 4 units down -/
theorem parabola_translation :
  ∀ x : ℝ, translated_parabola x = original_parabola (x - 1) - 4 :=
by
  sorry

end NUMINAMATH_CALUDE_parabola_translation_l125_12577


namespace NUMINAMATH_CALUDE_log_half_decreasing_l125_12531

-- Define the function f(x) = log_(1/2)(x)
noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log (1/2)

-- State the theorem
theorem log_half_decreasing :
  ∀ x₁ x₂ : ℝ, x₁ > 0 → x₂ > 0 → x₁ < x₂ → f x₁ > f x₂ := by
  sorry

end NUMINAMATH_CALUDE_log_half_decreasing_l125_12531


namespace NUMINAMATH_CALUDE_expression_value_l125_12578

theorem expression_value (a b : ℤ) (h1 : a = 4) (h2 : b = -5) : 
  -a^2 - b^2 + a*b + b = -66 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l125_12578


namespace NUMINAMATH_CALUDE_cube_sum_greater_than_product_sufficient_l125_12503

theorem cube_sum_greater_than_product_sufficient (x y z : ℝ) : 
  x + y + z > 0 → x^3 + y^3 + z^3 > 3*x*y*z := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_greater_than_product_sufficient_l125_12503


namespace NUMINAMATH_CALUDE_smallest_two_digit_with_digit_product_12_l125_12593

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def digit_product (n : ℕ) : ℕ :=
  (n / 10) * (n % 10)

theorem smallest_two_digit_with_digit_product_12 :
  ∀ n : ℕ, is_two_digit n → digit_product n = 12 → 26 ≤ n :=
sorry

end NUMINAMATH_CALUDE_smallest_two_digit_with_digit_product_12_l125_12593


namespace NUMINAMATH_CALUDE_repeating_decimal_length_1_221_l125_12536

theorem repeating_decimal_length_1_221 : ∃ n : ℕ, n > 0 ∧ n = 48 ∧ ∀ k : ℕ, (10^k - 1) % 221 = 0 ↔ n ∣ k := by sorry

end NUMINAMATH_CALUDE_repeating_decimal_length_1_221_l125_12536


namespace NUMINAMATH_CALUDE_diamonds_formula_diamonds_G15_l125_12590

/-- The number of diamonds in figure G_n -/
def diamonds (n : ℕ+) : ℕ :=
  6 * n

/-- The theorem stating that the number of diamonds in G_n is 6n -/
theorem diamonds_formula (n : ℕ+) : diamonds n = 6 * n := by
  sorry

/-- Corollary: The number of diamonds in G_15 is 90 -/
theorem diamonds_G15 : diamonds 15 = 90 := by
  sorry

end NUMINAMATH_CALUDE_diamonds_formula_diamonds_G15_l125_12590


namespace NUMINAMATH_CALUDE_smallest_n_congruence_l125_12574

theorem smallest_n_congruence (n : ℕ) : n > 0 → (∀ k < n, (7^k : ℤ) % 5 ≠ k^7 % 5) → (7^n : ℤ) % 5 = n^7 % 5 → n = 2 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_congruence_l125_12574


namespace NUMINAMATH_CALUDE_shelter_dogs_l125_12558

/-- The number of dogs in an animal shelter given specific ratios -/
theorem shelter_dogs (d c : ℕ) (h1 : d * 7 = c * 15) (h2 : d * 11 = (c + 20) * 15) : d = 175 := by
  sorry

end NUMINAMATH_CALUDE_shelter_dogs_l125_12558


namespace NUMINAMATH_CALUDE_divisibility_condition_l125_12591

theorem divisibility_condition (a b c : ℝ) :
  ∀ n : ℕ, (∃ k : ℝ, a^n * (b - c) + b^n * (c - a) + c^n * (a - b) = k * (a^2 + b^2 + c^2 + a*b + b*c + c*a)) ↔ n = 4 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_condition_l125_12591


namespace NUMINAMATH_CALUDE_travis_bowls_problem_l125_12500

/-- Represents the problem of calculating the number of bowls Travis initially had --/
theorem travis_bowls_problem :
  let base_fee : ℕ := 100
  let safe_bowl_pay : ℕ := 3
  let lost_bowl_fee : ℕ := 4
  let lost_bowls : ℕ := 12
  let broken_bowls : ℕ := 15
  let total_payment : ℕ := 1825

  ∃ (total_bowls safe_bowls : ℕ),
    total_bowls = safe_bowls + lost_bowls + broken_bowls ∧
    total_payment = base_fee + safe_bowl_pay * safe_bowls - lost_bowl_fee * (lost_bowls + broken_bowls) ∧
    total_bowls = 638 :=
by
  sorry

end NUMINAMATH_CALUDE_travis_bowls_problem_l125_12500


namespace NUMINAMATH_CALUDE_integer_average_sum_l125_12540

theorem integer_average_sum (a b c d : ℤ) 
  (h1 : (a + b + c) / 3 + d = 29)
  (h2 : (b + c + d) / 3 + a = 23)
  (h3 : (a + c + d) / 3 + b = 21)
  (h4 : (a + b + d) / 3 + c = 17) :
  a = 21 ∨ b = 21 ∨ c = 21 ∨ d = 21 :=
by sorry

end NUMINAMATH_CALUDE_integer_average_sum_l125_12540


namespace NUMINAMATH_CALUDE_sum_of_specific_repeating_decimals_l125_12597

/-- Represents a repeating decimal with a repeating part and a period -/
def RepeatingDecimal (repeating_part : ℕ) (period : ℕ) : ℚ :=
  repeating_part / (10^period - 1)

/-- The sum of three specific repeating decimals -/
theorem sum_of_specific_repeating_decimals :
  RepeatingDecimal 12 2 + RepeatingDecimal 34 3 + RepeatingDecimal 567 5 = 16133 / 99999 := by
  sorry

#eval RepeatingDecimal 12 2 + RepeatingDecimal 34 3 + RepeatingDecimal 567 5

end NUMINAMATH_CALUDE_sum_of_specific_repeating_decimals_l125_12597


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l125_12524

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  geometric_sequence a →
  a 4 + a 7 = 2 →
  a 5 * a 6 = -8 →
  a 1 + a 10 = -7 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l125_12524


namespace NUMINAMATH_CALUDE_coin_flip_probability_l125_12583

/-- Represents the outcome of a coin flip -/
inductive CoinOutcome
| Heads
| Tails

/-- Represents the set of five coins -/
structure CoinSet :=
  (penny : CoinOutcome)
  (nickel : CoinOutcome)
  (dime : CoinOutcome)
  (quarter : CoinOutcome)
  (half_dollar : CoinOutcome)

/-- The total number of possible outcomes when flipping five coins -/
def total_outcomes : Nat := 32

/-- Predicate for the desired outcome (penny, nickel, and half dollar are heads) -/
def desired_outcome (cs : CoinSet) : Prop :=
  cs.penny = CoinOutcome.Heads ∧
  cs.nickel = CoinOutcome.Heads ∧
  cs.half_dollar = CoinOutcome.Heads

/-- The number of outcomes satisfying the desired condition -/
def successful_outcomes : Nat := 4

/-- The probability of the desired outcome -/
def probability : ℚ := 1 / 8

theorem coin_flip_probability :
  (successful_outcomes : ℚ) / total_outcomes = probability :=
sorry

end NUMINAMATH_CALUDE_coin_flip_probability_l125_12583


namespace NUMINAMATH_CALUDE_max_alpha_is_half_l125_12538

/-- The set of functions satisfying the given condition -/
def F : Set (ℝ → ℝ) :=
  {f | ∀ x > 0, f (3 * x) ≥ f (f (2 * x)) + x}

/-- The theorem stating that 1/2 is the maximum α -/
theorem max_alpha_is_half :
    (∃ α : ℝ, ∀ f ∈ F, ∀ x > 0, f x ≥ α * x) ∧
    (∀ β : ℝ, (∀ f ∈ F, ∀ x > 0, f x ≥ β * x) → β ≤ 1/2) :=
  sorry


end NUMINAMATH_CALUDE_max_alpha_is_half_l125_12538


namespace NUMINAMATH_CALUDE_incenter_is_intersection_of_angle_bisectors_l125_12506

/-- A triangle in a 2D plane -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- A point in a 2D plane -/
def Point := ℝ × ℝ

/-- The distance from a point to a line segment -/
noncomputable def distanceToSide (P : Point) (side : Point × Point) : ℝ := sorry

/-- The angle bisector of an angle in a triangle -/
noncomputable def angleBisector (vertex : Point) (side1 : Point) (side2 : Point) : Point × Point := sorry

/-- The intersection point of two lines -/
noncomputable def lineIntersection (line1 : Point × Point) (line2 : Point × Point) : Point := sorry

theorem incenter_is_intersection_of_angle_bisectors (T : Triangle) :
  ∃ (P : Point),
    (∀ (side : Point × Point), 
      side ∈ [(T.A, T.B), (T.B, T.C), (T.C, T.A)] → 
      distanceToSide P side = distanceToSide P (T.A, T.B)) ↔
    (P = lineIntersection 
      (angleBisector T.A T.B T.C) 
      (angleBisector T.B T.C T.A)) :=
by sorry

end NUMINAMATH_CALUDE_incenter_is_intersection_of_angle_bisectors_l125_12506


namespace NUMINAMATH_CALUDE_friends_games_l125_12523

theorem friends_games (katie_games : ℕ) (difference : ℕ) : 
  katie_games = 81 → difference = 22 → katie_games - difference = 59 := by
  sorry

end NUMINAMATH_CALUDE_friends_games_l125_12523


namespace NUMINAMATH_CALUDE_tenthDrawnNumber_l125_12582

/-- Represents the systematic sampling problem -/
def systematicSampling (totalStudents : Nat) (sampleSize : Nat) (firstDrawn : Nat) (nthDraw : Nat) : Nat :=
  let interval := totalStudents / sampleSize
  firstDrawn + interval * (nthDraw - 1)

/-- Theorem stating the 10th drawn number in the given systematic sampling scenario -/
theorem tenthDrawnNumber :
  systematicSampling 1000 50 15 10 = 195 := by
  sorry

end NUMINAMATH_CALUDE_tenthDrawnNumber_l125_12582


namespace NUMINAMATH_CALUDE_photos_to_cover_poster_l125_12596

def poster_length : ℕ := 3
def poster_width : ℕ := 5
def photo_length : ℕ := 3
def photo_width : ℕ := 5
def inches_per_foot : ℕ := 12

theorem photos_to_cover_poster :
  (poster_length * inches_per_foot * poster_width * inches_per_foot) / (photo_length * photo_width) = 144 := by
  sorry

end NUMINAMATH_CALUDE_photos_to_cover_poster_l125_12596


namespace NUMINAMATH_CALUDE_fraction_simplification_l125_12589

theorem fraction_simplification (a : ℝ) (h : a^2 ≠ 9) :
  3 / (a^2 - 9) - a / (9 - a^2) = 1 / (a - 3) := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l125_12589


namespace NUMINAMATH_CALUDE_find_divisor_l125_12575

theorem find_divisor (dividend quotient remainder : ℕ) (h : dividend = quotient * 23 + remainder) :
  dividend = 997 → quotient = 43 → remainder = 8 → 23 = dividend / quotient :=
by sorry

end NUMINAMATH_CALUDE_find_divisor_l125_12575


namespace NUMINAMATH_CALUDE_horner_method_v3_l125_12565

def f (x : ℝ) : ℝ := 5*x^5 + 2*x^4 + 3.5*x^3 - 2.6*x^2 + 1.7*x - 0.8

def horner_step (a : ℝ) (x : ℝ) (v : ℝ) : ℝ := a*x + v

def horner_v3 (x : ℝ) : ℝ :=
  let v0 := 5
  let v1 := horner_step 2 x v0
  let v2 := horner_step 3.5 x v1
  horner_step (-2.6) x v2

theorem horner_method_v3 :
  horner_v3 1 = 7.9 :=
sorry

end NUMINAMATH_CALUDE_horner_method_v3_l125_12565


namespace NUMINAMATH_CALUDE_john_pill_payment_john_pays_54_dollars_l125_12564

/-- The amount John pays for pills in a 30-day month, given the specified conditions. -/
theorem john_pill_payment (pills_per_day : ℕ) (cost_per_pill : ℚ) 
  (insurance_coverage_percent : ℚ) (days_in_month : ℕ) : ℚ :=
  let total_cost := (pills_per_day : ℚ) * cost_per_pill * days_in_month
  let insurance_coverage := total_cost * (insurance_coverage_percent / 100)
  total_cost - insurance_coverage

/-- Proof that John pays $54 for his pills in a 30-day month. -/
theorem john_pays_54_dollars : 
  john_pill_payment 2 (3/2) 40 30 = 54 := by
  sorry

end NUMINAMATH_CALUDE_john_pill_payment_john_pays_54_dollars_l125_12564


namespace NUMINAMATH_CALUDE_number_division_problem_l125_12555

theorem number_division_problem :
  ∃! n : ℕ, 
    n / (555 + 445) = 2 * (555 - 445) ∧ 
    n % (555 + 445) = 70 := by
  sorry

end NUMINAMATH_CALUDE_number_division_problem_l125_12555


namespace NUMINAMATH_CALUDE_ball_in_ice_l125_12530

theorem ball_in_ice (r : ℝ) (h : r = 16.25) :
  let d := 30  -- diameter of the hole
  let depth := 10  -- depth of the hole
  let x := r - depth  -- distance from center of ball to surface
  d^2 / 4 + x^2 = r^2 := by sorry

end NUMINAMATH_CALUDE_ball_in_ice_l125_12530


namespace NUMINAMATH_CALUDE_largest_n_for_unique_k_l125_12521

theorem largest_n_for_unique_k : ∃ (k : ℤ),
  (8 : ℚ) / 15 < (112 : ℚ) / (112 + k) ∧ (112 : ℚ) / (112 + k) < 7 / 13 ∧
  ∀ (m : ℕ) (k' : ℤ), m > 112 →
    ((8 : ℚ) / 15 < (m : ℚ) / (m + k') ∧ (m : ℚ) / (m + k') < 7 / 13 →
     ∃ (k'' : ℤ), k'' ≠ k' ∧ (8 : ℚ) / 15 < (m : ℚ) / (m + k'') ∧ (m : ℚ) / (m + k'') < 7 / 13) :=
by sorry

#check largest_n_for_unique_k

end NUMINAMATH_CALUDE_largest_n_for_unique_k_l125_12521


namespace NUMINAMATH_CALUDE_mean_score_is_94_5_l125_12550

structure ScoreData where
  score : ℕ
  count : ℕ

def total_students : ℕ := 120

def score_distribution : List ScoreData := [
  ⟨120, 12⟩,
  ⟨110, 19⟩,
  ⟨100, 33⟩,
  ⟨90, 30⟩,
  ⟨75, 15⟩,
  ⟨65, 9⟩,
  ⟨50, 2⟩
]

def total_score : ℕ := score_distribution.foldl (fun acc data => acc + data.score * data.count) 0

theorem mean_score_is_94_5 :
  (total_score : ℚ) / total_students = 94.5 := by
  sorry

end NUMINAMATH_CALUDE_mean_score_is_94_5_l125_12550


namespace NUMINAMATH_CALUDE_square_area_change_l125_12508

theorem square_area_change (original_area : ℝ) (decrease_percent : ℝ) (increase_percent : ℝ) : 
  original_area = 625 →
  decrease_percent = 0.2 →
  increase_percent = 0.2 →
  let original_side : ℝ := Real.sqrt original_area
  let new_side1 : ℝ := original_side * (1 - decrease_percent)
  let new_side2 : ℝ := original_side * (1 + increase_percent)
  new_side1 * new_side2 = 600 := by
sorry

end NUMINAMATH_CALUDE_square_area_change_l125_12508


namespace NUMINAMATH_CALUDE_exists_permutation_distinct_columns_l125_12529

/-- A table is represented as a function from pairs of indices to integers -/
def Table (n : ℕ) := Fin n → Fin n → ℤ

/-- A predicate stating that no two cells within a row share the same number -/
def DistinctInRows (t : Table n) : Prop :=
  ∀ i j₁ j₂, j₁ ≠ j₂ → t i j₁ ≠ t i j₂

/-- A permutation of a row is a bijection on Fin n -/
def RowPermutation (n : ℕ) := Fin n ≃ Fin n

/-- Apply a row permutation to a table -/
def ApplyRowPermutation (t : Table n) (p : Fin n → RowPermutation n) : Table n :=
  λ i j ↦ t i ((p i).toFun j)

/-- A predicate stating that all columns contain distinct numbers -/
def DistinctInColumns (t : Table n) : Prop :=
  ∀ j i₁ i₂, i₁ ≠ i₂ → t i₁ j ≠ t i₂ j

/-- The main theorem -/
theorem exists_permutation_distinct_columns (n : ℕ) (t : Table n) 
    (h : DistinctInRows t) : 
    ∃ p : Fin n → RowPermutation n, DistinctInColumns (ApplyRowPermutation t p) := by
  sorry

end NUMINAMATH_CALUDE_exists_permutation_distinct_columns_l125_12529


namespace NUMINAMATH_CALUDE_tropical_storm_sally_rainfall_l125_12543

theorem tropical_storm_sally_rainfall (day1 day2 day3 : ℝ) : 
  day2 = 5 * day1 →
  day3 = day1 + day2 - 6 →
  day3 = 18 →
  day1 = 4 := by
sorry

end NUMINAMATH_CALUDE_tropical_storm_sally_rainfall_l125_12543


namespace NUMINAMATH_CALUDE_average_salary_calculation_l125_12559

/-- Calculates the average salary of all employees in an office --/
theorem average_salary_calculation (officer_salary : ℕ) (non_officer_salary : ℕ) 
  (officer_count : ℕ) (non_officer_count : ℕ) :
  officer_salary = 470 →
  non_officer_salary = 110 →
  officer_count = 15 →
  non_officer_count = 525 →
  (officer_salary * officer_count + non_officer_salary * non_officer_count) / 
    (officer_count + non_officer_count) = 120 := by
  sorry

#check average_salary_calculation

end NUMINAMATH_CALUDE_average_salary_calculation_l125_12559


namespace NUMINAMATH_CALUDE_min_teams_for_players_l125_12537

theorem min_teams_for_players (total_players : ℕ) (max_per_team : ℕ) (min_teams : ℕ) : 
  total_players = 30 → 
  max_per_team = 7 → 
  min_teams = 5 → 
  (∀ t : ℕ, t < min_teams → t * max_per_team < total_players) ∧ 
  (min_teams * (total_players / min_teams) = total_players) := by
  sorry

end NUMINAMATH_CALUDE_min_teams_for_players_l125_12537


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_is_integer_l125_12551

theorem right_triangle_hypotenuse_is_integer (n : ℤ) :
  let a : ℤ := 2 * n + 1
  let b : ℤ := 2 * n * (n + 1)
  let c : ℤ := 2 * n^2 + 2 * n + 1
  c^2 = a^2 + b^2 := by sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_is_integer_l125_12551


namespace NUMINAMATH_CALUDE_valentines_dog_biscuits_l125_12544

theorem valentines_dog_biscuits (num_dogs : ℕ) (biscuits_per_dog : ℕ) : 
  num_dogs = 2 → biscuits_per_dog = 3 → num_dogs * biscuits_per_dog = 6 :=
by sorry

end NUMINAMATH_CALUDE_valentines_dog_biscuits_l125_12544


namespace NUMINAMATH_CALUDE_stickers_given_correct_l125_12562

/-- Represents the number of stickers Willie gave to Emily -/
def stickers_given (initial final : ℕ) : ℕ := initial - final

/-- Proves that the number of stickers Willie gave to Emily is correct -/
theorem stickers_given_correct (initial final : ℕ) (h : initial ≥ final) :
  stickers_given initial final = initial - final :=
by
  sorry

end NUMINAMATH_CALUDE_stickers_given_correct_l125_12562


namespace NUMINAMATH_CALUDE_circuit_probability_l125_12504

/-- The probability that a circuit with two independently controlled switches
    connected in parallel can operate normally. -/
theorem circuit_probability (p1 p2 : ℝ) (h1 : p1 = 0.5) (h2 : p2 = 0.7) :
  p1 * (1 - p2) + (1 - p1) * p2 + p1 * p2 = 0.85 := by
  sorry

end NUMINAMATH_CALUDE_circuit_probability_l125_12504


namespace NUMINAMATH_CALUDE_unique_rebus_solution_l125_12519

/-- Represents a four-digit number ABCD where A, B, C, D are distinct non-zero digits. -/
structure FourDigitNumber where
  a : Nat
  b : Nat
  c : Nat
  d : Nat
  a_nonzero : a ≠ 0
  b_nonzero : b ≠ 0
  c_nonzero : c ≠ 0
  d_nonzero : d ≠ 0
  all_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d
  a_digit : a < 10
  b_digit : b < 10
  c_digit : c < 10
  d_digit : d < 10

/-- The rebus equation ABCA = 182 * CD -/
def rebusEquation (n : FourDigitNumber) : Prop :=
  1000 * n.a + 100 * n.b + 10 * n.c + n.a = 182 * (10 * n.c + n.d)

/-- Theorem stating that 2916 is the only solution to the rebus equation -/
theorem unique_rebus_solution :
  ∃! n : FourDigitNumber, rebusEquation n ∧ n.a = 2 ∧ n.b = 9 ∧ n.c = 1 ∧ n.d = 6 :=
sorry

end NUMINAMATH_CALUDE_unique_rebus_solution_l125_12519


namespace NUMINAMATH_CALUDE_egg_count_problem_l125_12513

theorem egg_count_problem (count_sum : ℕ) (error_sum : ℤ) (actual_count : ℕ) : 
  count_sum = 3162 →
  (∃ (e1 e2 e3 : ℤ), (e1 = 1 ∨ e1 = -1) ∧ 
                     (e2 = 10 ∨ e2 = -10) ∧ 
                     (e3 = 100 ∨ e3 = -100) ∧ 
                     error_sum = e1 + e2 + e3) →
  7 * actual_count + error_sum = count_sum →
  actual_count = 439 := by
sorry

end NUMINAMATH_CALUDE_egg_count_problem_l125_12513


namespace NUMINAMATH_CALUDE_triangle_circumradius_l125_12533

/-- Given a triangle ABC with area S = (1/2) * sin A * sin B * sin C, 
    the radius R of its circumcircle is equal to 1/2. -/
theorem triangle_circumradius (A B C : ℝ) (a b c : ℝ) (S : ℝ) (R : ℝ) : 
  S = (1/2) * Real.sin A * Real.sin B * Real.sin C →
  R = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_circumradius_l125_12533


namespace NUMINAMATH_CALUDE_craigs_apples_l125_12566

/-- 
Given:
- Craig's initial number of apples
- The number of apples Craig shares with Eugene
Prove that Craig's final number of apples is equal to the initial number minus the shared number.
-/
theorem craigs_apples (initial_apples shared_apples : ℕ) :
  initial_apples - shared_apples = initial_apples - shared_apples :=
by sorry

end NUMINAMATH_CALUDE_craigs_apples_l125_12566


namespace NUMINAMATH_CALUDE_fraction_simplification_l125_12546

theorem fraction_simplification : (8 : ℚ) / (5 * 42) = 4 / 105 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l125_12546


namespace NUMINAMATH_CALUDE_total_cost_after_discount_l125_12587

def mountain_bike_initial : ℝ := 250
def helmet_initial : ℝ := 60
def gloves_initial : ℝ := 30

def mountain_bike_increase : ℝ := 0.08
def helmet_increase : ℝ := 0.15
def gloves_increase : ℝ := 0.10

def discount : ℝ := 0.05

theorem total_cost_after_discount : 
  let mountain_bike_new := mountain_bike_initial * (1 + mountain_bike_increase)
  let helmet_new := helmet_initial * (1 + helmet_increase)
  let gloves_new := gloves_initial * (1 + gloves_increase)
  let total_before_discount := mountain_bike_new + helmet_new + gloves_new
  let total_after_discount := total_before_discount * (1 - discount)
  total_after_discount = 353.4 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_after_discount_l125_12587


namespace NUMINAMATH_CALUDE_circle_intersection_range_l125_12595

theorem circle_intersection_range (a : ℝ) : 
  (∃ x y : ℝ, (x - a)^2 + (y - a)^2 = 8 ∧ x^2 + y^2 = 2) ↔ 
  (a < -1 ∧ a > -3) ∨ (a > 1 ∧ a < 3) :=
sorry

end NUMINAMATH_CALUDE_circle_intersection_range_l125_12595


namespace NUMINAMATH_CALUDE_complex_magnitude_equation_l125_12549

theorem complex_magnitude_equation (n : ℝ) (h : n > 0) :
  Complex.abs (5 + Complex.I * n) = 5 * Real.sqrt 13 → n = 10 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_equation_l125_12549


namespace NUMINAMATH_CALUDE_cd_player_cost_l125_12501

/-- The amount spent on the CD player, given the total amount spent and the amounts spent on speakers and tires. -/
theorem cd_player_cost (total spent_on_speakers spent_on_tires : ℚ) 
  (h_total : total = 387.85)
  (h_speakers : spent_on_speakers = 136.01)
  (h_tires : spent_on_tires = 112.46) :
  total - (spent_on_speakers + spent_on_tires) = 139.38 := by
  sorry

end NUMINAMATH_CALUDE_cd_player_cost_l125_12501


namespace NUMINAMATH_CALUDE_line_passes_through_point_l125_12556

/-- The line equation y = 2x - 1 passes through the point (0, -1) -/
theorem line_passes_through_point :
  let f : ℝ → ℝ := λ x => 2 * x - 1
  f 0 = -1 := by sorry

end NUMINAMATH_CALUDE_line_passes_through_point_l125_12556


namespace NUMINAMATH_CALUDE_placemats_length_l125_12572

theorem placemats_length (R : ℝ) (n : ℕ) (x : ℝ) : 
  R = 5 ∧ n = 8 → x = 2 * R * Real.sin (π / (2 * n)) := by
  sorry

end NUMINAMATH_CALUDE_placemats_length_l125_12572


namespace NUMINAMATH_CALUDE_grid_solution_l125_12584

/-- Represents a 3x3 grid with numbers from 1 to 9 -/
def Grid := Fin 3 → Fin 3 → Fin 9

/-- Check if two cells are adjacent in the grid -/
def adjacent (i j k l : Fin 3) : Prop :=
  (i = k ∧ (j.val + 1 = l.val ∨ l.val + 1 = j.val)) ∨
  (j = l ∧ (i.val + 1 = k.val ∨ k.val + 1 = i.val))

/-- The sum of any two numbers in adjacent cells is less than 12 -/
def valid_sum (g : Grid) : Prop :=
  ∀ i j k l, adjacent i j k l → (g i j).val + (g k l).val < 12

/-- The given positions of known numbers in the grid -/
def known_positions (g : Grid) : Prop :=
  g 0 1 = 1 ∧ g 1 0 = 3 ∧ g 1 1 = 5 ∧ g 2 2 = 7 ∧ g 0 2 = 9

/-- The theorem to be proved -/
theorem grid_solution (g : Grid) 
  (h1 : valid_sum g) 
  (h2 : known_positions g) : 
  g 0 0 = 8 ∧ g 2 0 = 6 ∧ g 2 1 = 4 ∧ g 1 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_grid_solution_l125_12584


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_property_l125_12505

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

theorem arithmetic_sequence_sum_property
  (a : ℕ → ℝ) (d : ℝ) 
  (h_arithmetic : arithmetic_sequence a d)
  (h_sum : a 3 + a 5 + a 7 + a 9 + a 11 = 100) :
  3 * a 9 - a 13 = 2 * (a 1 + 6 * d) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_property_l125_12505


namespace NUMINAMATH_CALUDE_birds_in_tree_l125_12588

theorem birds_in_tree (initial_birds : ℝ) (birds_flew_away : ℝ) : 
  initial_birds = initial_birds := by sorry

end NUMINAMATH_CALUDE_birds_in_tree_l125_12588


namespace NUMINAMATH_CALUDE_largest_fraction_l125_12507

theorem largest_fraction : 
  let fractions := [5/12, 7/15, 29/58, 151/303, 199/400]
  ∀ x ∈ fractions, (29:ℚ)/58 ≥ x := by sorry

end NUMINAMATH_CALUDE_largest_fraction_l125_12507


namespace NUMINAMATH_CALUDE_r₂_bound_bound_is_tight_l125_12580

-- Define the function f
def f (r₂ r₃ : ℝ) (x : ℝ) : ℝ := x^2 - r₂ * x + r₃

-- Define the sequence g
def g (r₂ r₃ : ℝ) : ℕ → ℝ
  | 0 => 0
  | n + 1 => f r₂ r₃ (g r₂ r₃ n)

-- Define the conditions on the sequence
def sequence_conditions (r₂ r₃ : ℝ) : Prop :=
  (∀ i ≤ 2011, g r₂ r₃ (2*i) < g r₂ r₃ (2*i + 1) ∧ g r₂ r₃ (2*i + 1) > g r₂ r₃ (2*i + 2)) ∧
  (∃ j : ℕ, ∀ i > j, g r₂ r₃ (i + 1) > g r₂ r₃ i) ∧
  (∀ M : ℝ, ∃ N : ℕ, g r₂ r₃ N > M)

theorem r₂_bound (r₂ r₃ : ℝ) (h : sequence_conditions r₂ r₃) : |r₂| > 2 :=
  sorry

theorem bound_is_tight : ∀ ε > 0, ∃ r₂ r₃ : ℝ, sequence_conditions r₂ r₃ ∧ |r₂| < 2 + ε :=
  sorry

end NUMINAMATH_CALUDE_r₂_bound_bound_is_tight_l125_12580


namespace NUMINAMATH_CALUDE_ellipse_k_range_l125_12568

/-- An ellipse with equation x^2 + ky^2 = 2 and foci on the y-axis -/
structure Ellipse (k : ℝ) where
  eq : ∀ (x y : ℝ), x^2 + k * y^2 = 2
  foci_on_y : True  -- This is a placeholder for the foci condition

/-- The range of k for an ellipse with equation x^2 + ky^2 = 2 and foci on the y-axis -/
theorem ellipse_k_range (k : ℝ) (e : Ellipse k) : 0 < k ∧ k < 1 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_k_range_l125_12568


namespace NUMINAMATH_CALUDE_det_A_equals_one_l125_12545

theorem det_A_equals_one (a b c : ℝ) (A : Matrix (Fin 2) (Fin 2) ℝ) :
  A = ![![2*a, b], ![c, -2*a]] →
  A + A⁻¹ = 0 →
  Matrix.det A = 1 := by sorry

end NUMINAMATH_CALUDE_det_A_equals_one_l125_12545


namespace NUMINAMATH_CALUDE_bella_steps_to_meet_l125_12502

/-- The number of steps Bella takes before meeting Ella -/
def steps_to_meet (total_distance : ℕ) (bella_step_length : ℕ) (ella_speed_multiplier : ℕ) : ℕ :=
  let distance_to_meet := total_distance / 2
  let bella_speed := 1
  let ella_speed := ella_speed_multiplier * bella_speed
  let combined_speed := bella_speed + ella_speed
  let distance_bella_walks := (distance_to_meet * bella_speed) / combined_speed
  distance_bella_walks / bella_step_length

/-- Theorem stating that Bella takes 528 steps before meeting Ella -/
theorem bella_steps_to_meet :
  steps_to_meet 15840 3 4 = 528 := by
  sorry

end NUMINAMATH_CALUDE_bella_steps_to_meet_l125_12502


namespace NUMINAMATH_CALUDE_terminal_side_in_third_quadrant_l125_12576

/-- Given a point P with coordinates (cosθ, tanθ) in the second quadrant,
    prove that the terminal side of angle θ is in the third quadrant. -/
theorem terminal_side_in_third_quadrant (θ : Real) :
  (cosθ < 0 ∧ tanθ > 0) →  -- Point P is in the second quadrant
  (cosθ < 0 ∧ sinθ < 0)    -- Terminal side of θ is in the third quadrant
:= by sorry

end NUMINAMATH_CALUDE_terminal_side_in_third_quadrant_l125_12576


namespace NUMINAMATH_CALUDE_functional_equation_properties_l125_12547

/-- A function satisfying the given functional equation -/
noncomputable def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) + f (x - y) = 2 * f x * f y

theorem functional_equation_properties (f : ℝ → ℝ) 
  (h_eq : FunctionalEquation f) (h_nonzero : f 0 ≠ 0) : 
  (f 0 = 1) ∧ 
  (∀ x : ℝ, f (-x) = f x) ∧ 
  (∃ c : ℝ, (∀ x : ℝ, f (x + 2*c) = f x) ∧ f c = -1) := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_properties_l125_12547


namespace NUMINAMATH_CALUDE_jons_laundry_loads_l125_12594

/-- Represents the laundry machine and Jon's clothes -/
structure LaundryProblem where
  machine_capacity : ℝ
  shirt_weight : ℝ
  pants_weight : ℝ
  sock_weight : ℝ
  jacket_weight : ℝ
  shirt_count : ℕ
  pants_count : ℕ
  sock_count : ℕ
  jacket_count : ℕ

/-- Calculates the minimum number of loads required -/
def minimum_loads (problem : LaundryProblem) : ℕ :=
  sorry

/-- Theorem stating that the minimum number of loads for Jon's laundry is 5 -/
theorem jons_laundry_loads :
  let problem : LaundryProblem :=
    { machine_capacity := 8
    , shirt_weight := 1/4
    , pants_weight := 1/2
    , sock_weight := 1/6
    , jacket_weight := 2
    , shirt_count := 20
    , pants_count := 20
    , sock_count := 18
    , jacket_count := 6
    }
  minimum_loads problem = 5 := by
  sorry

end NUMINAMATH_CALUDE_jons_laundry_loads_l125_12594


namespace NUMINAMATH_CALUDE_correct_calculation_l125_12520

theorem correct_calculation (x : ℤ) (h : x + 44 - 39 = 63) : x + 39 - 44 = 53 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l125_12520


namespace NUMINAMATH_CALUDE_museum_exhibit_count_l125_12552

def base5ToBase10 (n : ℕ) : ℕ := sorry

theorem museum_exhibit_count : 
  let clay_tablets := base5ToBase10 1432
  let bronze_sculptures := base5ToBase10 2041
  let stone_carvings := base5ToBase10 232
  clay_tablets + bronze_sculptures + stone_carvings = 580 := by sorry

end NUMINAMATH_CALUDE_museum_exhibit_count_l125_12552


namespace NUMINAMATH_CALUDE_product_213_16_l125_12542

theorem product_213_16 : 213 * 16 = 3408 := by
  sorry

end NUMINAMATH_CALUDE_product_213_16_l125_12542


namespace NUMINAMATH_CALUDE_ball_probabilities_l125_12534

/-- The total number of balls in the bag -/
def total_balls : ℕ := 12

/-- The number of red balls initially in the bag -/
def red_balls : ℕ := 4

/-- The number of black balls in the bag -/
def black_balls : ℕ := 8

/-- The probability of drawing a black ball after removing m red balls -/
def prob_black (m : ℕ) : ℚ :=
  black_balls / (total_balls - m)

/-- The probability of drawing a black ball after removing n red balls -/
def prob_black_n (n : ℕ) : ℚ :=
  black_balls / (total_balls - n)

theorem ball_probabilities :
  (prob_black 4 = 1) ∧
  (prob_black 2 > 0 ∧ prob_black 2 < 1) ∧
  (prob_black 3 > 0 ∧ prob_black 3 < 1) ∧
  (prob_black_n 3 = 8/9) :=
sorry

end NUMINAMATH_CALUDE_ball_probabilities_l125_12534


namespace NUMINAMATH_CALUDE_parallel_vectors_cos_2theta_l125_12509

/-- Given two parallel vectors a and b, prove that cos(2θ) = -1/3 -/
theorem parallel_vectors_cos_2theta (θ : ℝ) 
  (a b : ℝ × ℝ) 
  (ha : a = (Real.cos θ, 1)) 
  (hb : b = (1, 3 * Real.cos θ)) 
  (h_parallel : ∃ (k : ℝ), k ≠ 0 ∧ a = k • b) : 
  Real.cos (2 * θ) = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_cos_2theta_l125_12509


namespace NUMINAMATH_CALUDE_harriet_return_speed_harriet_return_speed_approx_145_l125_12585

/-- Calculates the return speed given the conditions of Harriet's trip -/
theorem harriet_return_speed (outbound_speed : ℝ) (total_time : ℝ) (outbound_time_minutes : ℝ) : ℝ :=
  let outbound_time : ℝ := outbound_time_minutes / 60
  let distance : ℝ := outbound_speed * outbound_time
  let return_time : ℝ := total_time - outbound_time
  distance / return_time

/-- Proves that Harriet's return speed is approximately 145 km/h -/
theorem harriet_return_speed_approx_145 :
  ∃ ε > 0, abs (harriet_return_speed 105 5 174 - 145) < ε :=
sorry

end NUMINAMATH_CALUDE_harriet_return_speed_harriet_return_speed_approx_145_l125_12585


namespace NUMINAMATH_CALUDE_qinJiushaoResult_l125_12526

/-- Qin Jiushao's algorithm for polynomial evaluation -/
def qinJiushaoAlgorithm (n : ℕ) (x : ℕ) : ℕ :=
  let rec loop : ℕ → ℕ → ℕ
    | 0, v => v
    | i+1, v => loop i (x * v + 1)
  loop n 1

/-- Theorem stating the result of Qin Jiushao's algorithm for n=5 and x=2 -/
theorem qinJiushaoResult : qinJiushaoAlgorithm 5 2 = 2^5 + 2^4 + 2^3 + 2^2 + 2 + 1 := by
  sorry

#eval qinJiushaoAlgorithm 5 2

end NUMINAMATH_CALUDE_qinJiushaoResult_l125_12526


namespace NUMINAMATH_CALUDE_cone_volume_l125_12532

/-- Given a cone with base area 2π and lateral area 4π, its volume is (2√6/3)π -/
theorem cone_volume (r l h : ℝ) (h_base_area : π * r^2 = 2) (h_lateral_area : π * r * l = 4) 
  (h_height : h^2 = l^2 - r^2) : 
  (1/3) * π * r^2 * h = (2 * Real.sqrt 6 / 3) * π := by
  sorry


end NUMINAMATH_CALUDE_cone_volume_l125_12532


namespace NUMINAMATH_CALUDE_circles_intersect_l125_12561

-- Define the circles
def C₁ (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 8*y - 8 = 0
def C₂ (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 4*y - 2 = 0

-- Theorem statement
theorem circles_intersect : ∃ (x y : ℝ), C₁ x y ∧ C₂ x y :=
sorry

end NUMINAMATH_CALUDE_circles_intersect_l125_12561


namespace NUMINAMATH_CALUDE_tailor_buttons_l125_12516

theorem tailor_buttons (green : ℕ) (yellow : ℕ) (blue : ℕ) 
  (h1 : yellow = green + 10)
  (h2 : blue = green - 5)
  (h3 : green + yellow + blue = 275) :
  green = 90 := by
sorry

end NUMINAMATH_CALUDE_tailor_buttons_l125_12516


namespace NUMINAMATH_CALUDE_prob_sum_eight_two_dice_l125_12599

def dice_outcomes : ℕ := 6 * 6

def favorable_outcomes : ℕ := 5

theorem prob_sum_eight_two_dice : 
  (favorable_outcomes : ℚ) / dice_outcomes = 5 / 36 := by sorry

end NUMINAMATH_CALUDE_prob_sum_eight_two_dice_l125_12599


namespace NUMINAMATH_CALUDE_angle_D_value_l125_12514

-- Define the angles as real numbers
variable (A B C D E : ℝ)

-- State the given conditions
axiom angle_sum : A + B = 180
axiom angle_C_eq_D : C = D
axiom angle_A_value : A = 50
axiom angle_E_value : E = 60
axiom triangle1_sum : A + B + E = 180
axiom triangle2_sum : B + C + D = 180

-- State the theorem to be proved
theorem angle_D_value : D = 55 := by
  sorry

end NUMINAMATH_CALUDE_angle_D_value_l125_12514


namespace NUMINAMATH_CALUDE_element_in_union_l125_12512

theorem element_in_union (M N : Set ℕ) (a : ℕ) 
  (h1 : M ∪ N = {1, 2, 3})
  (h2 : M ∩ N = {a}) : 
  a ∈ M ∪ N := by
  sorry

end NUMINAMATH_CALUDE_element_in_union_l125_12512


namespace NUMINAMATH_CALUDE_system_solution_l125_12518

theorem system_solution : 
  ∃ (x₁ y₁ x₂ y₂ : ℝ), 
    (x₁ = 0 ∧ y₁ = 1/3) ∧ 
    (x₂ = 19/2 ∧ y₂ = -6) ∧
    (∀ x y : ℝ, (5*x*(y + 6) = 0 ∧ 2*x + 3*y = 1) → 
      ((x = x₁ ∧ y = y₁) ∨ (x = x₂ ∧ y = y₂))) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l125_12518


namespace NUMINAMATH_CALUDE_arctan_equation_solution_l125_12567

theorem arctan_equation_solution :
  ∃ y : ℝ, 2 * Real.arctan (1/5) + 2 * Real.arctan (1/25) + Real.arctan (1/y) = π/4 ∧ y = 1 := by
  sorry

end NUMINAMATH_CALUDE_arctan_equation_solution_l125_12567


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l125_12522

theorem quadratic_inequality_range (m : ℝ) :
  (¬∃ x : ℝ, x^2 - 2*x + m ≤ 0) ↔ m > 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l125_12522


namespace NUMINAMATH_CALUDE_sin_cos_identity_l125_12592

theorem sin_cos_identity (α : Real) (h : Real.sin α + 2 * Real.cos α = 0) :
  2 * Real.sin α * Real.cos α - Real.cos α ^ 2 = -1 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_identity_l125_12592


namespace NUMINAMATH_CALUDE_coefficient_expansion_l125_12579

theorem coefficient_expansion (a : ℝ) : 
  (Nat.choose 5 3) * a^3 = 80 → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_expansion_l125_12579


namespace NUMINAMATH_CALUDE_triangle_properties_l125_12586

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the main theorem
theorem triangle_properties (t : Triangle) 
  (h1 : 2 * t.c * Real.sin t.C = (2 * t.b + t.a) * Real.sin t.B + (2 * t.a - 3 * t.b) * Real.sin t.A)
  (h2 : t.a > 0 ∧ t.b > 0 ∧ t.c > 0)
  (h3 : t.A > 0 ∧ t.B > 0 ∧ t.C > 0)
  (h4 : t.A + t.B + t.C = π) : 
  (t.C = π / 3) ∧ 
  (t.c = 4 → 4 < t.a + t.b ∧ t.a + t.b ≤ 8) :=
sorry


end NUMINAMATH_CALUDE_triangle_properties_l125_12586


namespace NUMINAMATH_CALUDE_angle_A_measure_l125_12573

-- Define the angles A and B
def angle_A : ℝ := sorry
def angle_B : ℝ := sorry

-- State the theorem
theorem angle_A_measure :
  (angle_A = 2 * angle_B - 15) →  -- Condition 1
  (angle_A + angle_B = 180) →     -- Condition 2 (supplementary angles)
  angle_A = 115 := by             -- Conclusion
sorry


end NUMINAMATH_CALUDE_angle_A_measure_l125_12573


namespace NUMINAMATH_CALUDE_line_translation_proof_l125_12598

/-- Represents a line in the form y = mx + b -/
structure Line where
  slope : ℝ
  yIntercept : ℝ

/-- The vertical translation distance between two lines with the same slope -/
def verticalTranslation (l1 l2 : Line) : ℝ :=
  l2.yIntercept - l1.yIntercept

theorem line_translation_proof (l1 l2 : Line) 
  (h1 : l1.slope = 3 ∧ l1.yIntercept = -1)
  (h2 : l2.slope = 3 ∧ l2.yIntercept = 6)
  : verticalTranslation l1 l2 = 7 := by
  sorry

end NUMINAMATH_CALUDE_line_translation_proof_l125_12598


namespace NUMINAMATH_CALUDE_even_function_inequality_l125_12560

def f (m : ℝ) (x : ℝ) : ℝ := (m - 1) * x^2 + 2 * m * x + 3

theorem even_function_inequality (m : ℝ) :
  (∀ x : ℝ, f m x = f m (-x)) →
  f m (Real.sqrt 3) < f m (-Real.sqrt 2) ∧ f m (-Real.sqrt 2) < f m (-1) :=
by sorry

end NUMINAMATH_CALUDE_even_function_inequality_l125_12560


namespace NUMINAMATH_CALUDE_increasing_f_implies_a_nonnegative_l125_12539

def f (a : ℝ) (x : ℝ) : ℝ := x^3 + 3*a*x

theorem increasing_f_implies_a_nonnegative (a : ℝ) : 
  (∀ x y : ℝ, x < y → f a x < f a y) → a ≥ 0 := by sorry

end NUMINAMATH_CALUDE_increasing_f_implies_a_nonnegative_l125_12539


namespace NUMINAMATH_CALUDE_not_good_pair_3_3_l125_12517

/-- A pair of natural numbers is good if there exists a polynomial with integer coefficients and distinct integers satisfying certain conditions. -/
def is_good_pair (r s : ℕ) : Prop :=
  ∃ (P : ℤ → ℤ) (a b : Fin r → ℤ) (c d : Fin s → ℤ),
    (∀ i j, i ≠ j → a i ≠ a j) ∧
    (∀ i j, i ≠ j → c i ≠ c j) ∧
    (∀ i j, a i ≠ c j) ∧
    (∀ i, P (a i) = 2) ∧
    (∀ i, P (c i) = 5) ∧
    (∀ x y : ℤ, (x - y) ∣ (P x - P y))

/-- Theorem stating that (3, 3) is not a good pair. -/
theorem not_good_pair_3_3 : ¬ is_good_pair 3 3 := by
  sorry

end NUMINAMATH_CALUDE_not_good_pair_3_3_l125_12517


namespace NUMINAMATH_CALUDE_chord_length_l125_12569

/-- The length of chord AB formed by the intersection of a line and a circle -/
theorem chord_length (A B : ℝ × ℝ) : 
  (∀ (x y : ℝ), x + Real.sqrt 3 * y - 2 = 0 → x^2 + y^2 = 4 → (x, y) = A ∨ (x, y) = B) →
  ((A.1 + Real.sqrt 3 * A.2 - 2 = 0 ∧ A.1^2 + A.2^2 = 4) ∧
   (B.1 + Real.sqrt 3 * B.2 - 2 = 0 ∧ B.1^2 + B.2^2 = 4)) →
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_chord_length_l125_12569


namespace NUMINAMATH_CALUDE_second_question_correct_percentage_l125_12528

theorem second_question_correct_percentage
  (first_correct : Real)
  (neither_correct : Real)
  (both_correct : Real)
  (h1 : first_correct = 0.63)
  (h2 : neither_correct = 0.20)
  (h3 : both_correct = 0.32) :
  ∃ (second_correct : Real),
    second_correct = 0.49 ∧
    first_correct + second_correct - both_correct = 1 - neither_correct :=
by sorry

end NUMINAMATH_CALUDE_second_question_correct_percentage_l125_12528


namespace NUMINAMATH_CALUDE_graph_not_in_third_quadrant_l125_12548

def f (x : ℝ) : ℝ := -x + 2

theorem graph_not_in_third_quadrant :
  ∀ x y : ℝ, f x = y → ¬(x < 0 ∧ y < 0) := by
  sorry

end NUMINAMATH_CALUDE_graph_not_in_third_quadrant_l125_12548


namespace NUMINAMATH_CALUDE_unique_n_reaches_16_l125_12571

def g (n : ℕ) : ℕ :=
  if n % 2 = 1 then n^2 + 1 else (n / 2)^2

theorem unique_n_reaches_16 :
  ∃! n : ℕ, n ∈ Finset.range 100 ∧
  ∃ k : ℕ, (k.iterate g n) = 16 :=
sorry

end NUMINAMATH_CALUDE_unique_n_reaches_16_l125_12571
