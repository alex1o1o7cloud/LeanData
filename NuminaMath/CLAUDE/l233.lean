import Mathlib

namespace NUMINAMATH_CALUDE_min_sum_m_n_l233_23376

theorem min_sum_m_n : ∃ (m n : ℕ+), 
  108 * (m : ℕ) = (n : ℕ)^3 ∧ 
  (∀ (m' n' : ℕ+), 108 * (m' : ℕ) = (n' : ℕ)^3 → (m : ℕ) + (n : ℕ) ≤ (m' : ℕ) + (n' : ℕ)) ∧
  (m : ℕ) + (n : ℕ) = 8 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_m_n_l233_23376


namespace NUMINAMATH_CALUDE_platform_length_platform_length_is_200_l233_23369

/-- Given a train traveling at 72 kmph that crosses a platform in 30 seconds and a man in 20 seconds, 
    the length of the platform is 200 meters. -/
theorem platform_length 
  (train_speed : ℝ) 
  (time_platform : ℝ) 
  (time_man : ℝ) 
  (h1 : train_speed = 72) 
  (h2 : time_platform = 30) 
  (h3 : time_man = 20) : ℝ := by
  
  -- Convert train speed from kmph to m/s
  let train_speed_ms := train_speed * 1000 / 3600

  -- Calculate length of train
  let train_length := train_speed_ms * time_man

  -- Calculate total distance (train + platform)
  let total_distance := train_speed_ms * time_platform

  -- Calculate platform length
  let platform_length := total_distance - train_length

  -- Prove that platform_length = 200
  sorry

/-- The length of the platform is 200 meters -/
theorem platform_length_is_200 : platform_length 72 30 20 rfl rfl rfl = 200 := by sorry

end NUMINAMATH_CALUDE_platform_length_platform_length_is_200_l233_23369


namespace NUMINAMATH_CALUDE_polynomial_root_ratio_l233_23300

theorem polynomial_root_ratio (a b c d e : ℝ) (h1 : a ≠ 0) :
  (∀ x : ℝ, a*x^4 + b*x^3 + c*x^2 + d*x + e = 0 ↔ x = 5 ∨ x = -3 ∨ x = 2 ∨ x = (-(b+d)/a - 5 - (-3) - 2)) →
  (b + d) / a = -12496 / 3173 := by
sorry

end NUMINAMATH_CALUDE_polynomial_root_ratio_l233_23300


namespace NUMINAMATH_CALUDE_geni_phone_expense_l233_23306

/-- Represents a telephone plan with fixed fee, free minutes, and per-minute rate -/
structure TelephonePlan where
  fixedFee : ℝ
  freeMinutes : ℕ
  ratePerMinute : ℝ

/-- Calculates the bill for a given usage in minutes -/
def calculateBill (plan : TelephonePlan) (usageMinutes : ℕ) : ℝ :=
  plan.fixedFee + max 0 (usageMinutes - plan.freeMinutes) * plan.ratePerMinute

/-- Converts hours and minutes to total minutes -/
def toMinutes (hours : ℕ) (minutes : ℕ) : ℕ := hours * 60 + minutes

theorem geni_phone_expense :
  let plan : TelephonePlan := { fixedFee := 18, freeMinutes := 600, ratePerMinute := 0.03 }
  let januaryUsage : ℕ := toMinutes 15 17
  let februaryUsage : ℕ := toMinutes 9 55
  calculateBill plan januaryUsage + calculateBill plan februaryUsage = 45.51 := by
  sorry

end NUMINAMATH_CALUDE_geni_phone_expense_l233_23306


namespace NUMINAMATH_CALUDE_quadratic_always_negative_l233_23357

theorem quadratic_always_negative (k : ℝ) :
  (∀ x : ℝ, (5 - k) * x^2 - 2 * (1 - k) * x + (2 - 2 * k) < 0) ↔ k > 9 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_always_negative_l233_23357


namespace NUMINAMATH_CALUDE_arithmetic_sequence_shared_prime_factor_l233_23394

theorem arithmetic_sequence_shared_prime_factor (a b : ℕ) (ha : a > 0) (hb : b > 0) :
  ∃ (p : ℕ) (hp : Prime p), ∀ n : ℕ, ∃ k ≥ n, p ∣ (a * k + b) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_shared_prime_factor_l233_23394


namespace NUMINAMATH_CALUDE_cement_bought_l233_23321

/-- The amount of cement bought, given the total amount, original amount, and son's contribution -/
theorem cement_bought (total : ℕ) (original : ℕ) (son_contribution : ℕ) 
  (h1 : total = 450)
  (h2 : original = 98)
  (h3 : son_contribution = 137) :
  total - (original + son_contribution) = 215 := by
  sorry

end NUMINAMATH_CALUDE_cement_bought_l233_23321


namespace NUMINAMATH_CALUDE_cone_base_area_l233_23372

/-- Given a cone whose unfolded lateral surface is a semicircle with area 2π,
    prove that the area of its base is π. -/
theorem cone_base_area (r : ℝ) (h : r > 0) : 
  (2 * π = π * r^2) → (π * r^2 / 2 = π) :=
by sorry

end NUMINAMATH_CALUDE_cone_base_area_l233_23372


namespace NUMINAMATH_CALUDE_slope_of_line_l233_23355

theorem slope_of_line (x₁ x₂ y₁ y₂ : ℝ) (h₁ : x₁ ≠ x₂) 
  (h₂ : (4 / x₁) + (5 / y₁) = 0) (h₃ : (4 / x₂) + (5 / y₂) = 0) :
  (y₂ - y₁) / (x₂ - x₁) = -5/4 := by
  sorry

end NUMINAMATH_CALUDE_slope_of_line_l233_23355


namespace NUMINAMATH_CALUDE_circle_area_theorem_l233_23330

theorem circle_area_theorem (r : ℝ) (A : ℝ) (h : r > 0) :
  8 * (1 / A) = r^2 → A = 2 * Real.sqrt (2 * Real.pi) :=
by sorry

end NUMINAMATH_CALUDE_circle_area_theorem_l233_23330


namespace NUMINAMATH_CALUDE_ellipse_and_line_properties_l233_23352

/-- An ellipse with given properties -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h : a > b ∧ b > 0
  eccentricity : ℝ
  h_eccentricity : eccentricity = Real.sqrt 6 / 3
  triangle_area : ℝ
  h_triangle_area : triangle_area = 5 * Real.sqrt 2 / 3

/-- The equation of the ellipse -/
def ellipse_equation (e : Ellipse) : ℝ × ℝ → Prop :=
  fun (x, y) ↦ x^2 / e.a^2 + y^2 / e.b^2 = 1

/-- The moving line that intersects the ellipse -/
def moving_line (k : ℝ) : ℝ → ℝ :=
  fun x ↦ k * (x + 1)

/-- Theorem stating the properties of the ellipse and the intersecting line -/
theorem ellipse_and_line_properties (e : Ellipse) :
  (∀ x y, ellipse_equation e (x, y) ↔ x^2 / 5 + y^2 / (5/3) = 1) ∧
  (∃ k : ℝ, k = Real.sqrt 3 / 3 ∨ k = -Real.sqrt 3 / 3 ∧
    ∃ x₁ x₂ : ℝ, 
      ellipse_equation e (x₁, moving_line k x₁) ∧
      ellipse_equation e (x₂, moving_line k x₂) ∧
      (x₁ + x₂) / 2 = -1/2) :=
sorry

end NUMINAMATH_CALUDE_ellipse_and_line_properties_l233_23352


namespace NUMINAMATH_CALUDE_size_relationship_l233_23380

theorem size_relationship : 
  let a : ℝ := 1 + Real.sqrt 7
  let b : ℝ := Real.sqrt 3 + Real.sqrt 5
  let c : ℝ := 4
  a < b ∧ b < c := by sorry

end NUMINAMATH_CALUDE_size_relationship_l233_23380


namespace NUMINAMATH_CALUDE_noah_jelly_beans_l233_23351

-- Define the total number of jelly beans
def total_jelly_beans : ℝ := 600

-- Define the percentages for Thomas and Sarah
def thomas_percentage : ℝ := 0.06
def sarah_percentage : ℝ := 0.10

-- Define the ratio for Barry, Emmanuel, and Miguel
def barry_ratio : ℝ := 4
def emmanuel_ratio : ℝ := 5
def miguel_ratio : ℝ := 6

-- Define the percentages for Chloe and Noah
def chloe_percentage : ℝ := 0.40
def noah_percentage : ℝ := 0.30

-- Theorem to prove
theorem noah_jelly_beans :
  let thomas_share := total_jelly_beans * thomas_percentage
  let sarah_share := total_jelly_beans * sarah_percentage
  let remaining_jelly_beans := total_jelly_beans - (thomas_share + sarah_share)
  let total_ratio := barry_ratio + emmanuel_ratio + miguel_ratio
  let emmanuel_share := (emmanuel_ratio / total_ratio) * remaining_jelly_beans
  let noah_share := emmanuel_share * noah_percentage
  noah_share = 50.4 := by
  sorry

end NUMINAMATH_CALUDE_noah_jelly_beans_l233_23351


namespace NUMINAMATH_CALUDE_inscribed_circles_radii_sum_l233_23339

/-- Given a triangle with an inscribed circle of radius r and three smaller triangles
    formed by tangent lines parallel to the sides of the original triangle, each with
    their own inscribed circles of radii r₁, r₂, and r₃, the sum of the radii of the
    smaller inscribed circles equals the radius of the original inscribed circle. -/
theorem inscribed_circles_radii_sum (r r₁ r₂ r₃ : ℝ) 
  (h : r > 0 ∧ r₁ > 0 ∧ r₂ > 0 ∧ r₃ > 0) : r₁ + r₂ + r₃ = r := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circles_radii_sum_l233_23339


namespace NUMINAMATH_CALUDE_candy_distribution_l233_23303

theorem candy_distribution (total : Nat) (friends : Nat) (h1 : total = 17) (h2 : friends = 5) :
  total % friends = 2 := by
  sorry

end NUMINAMATH_CALUDE_candy_distribution_l233_23303


namespace NUMINAMATH_CALUDE_cylinder_max_volume_l233_23301

/-- The maximum volume of a cylinder with total surface area 1 is achieved when 
    the radius and height are both equal to 1/√(6π) -/
theorem cylinder_max_volume (r h : ℝ) :
  r > 0 ∧ h > 0 ∧ 2 * Real.pi * r^2 + 2 * Real.pi * r * h = 1 →
  Real.pi * r^2 * h ≤ Real.pi * (1 / Real.sqrt (6 * Real.pi))^2 * (1 / Real.sqrt (6 * Real.pi)) :=
sorry

end NUMINAMATH_CALUDE_cylinder_max_volume_l233_23301


namespace NUMINAMATH_CALUDE_option2_higher_expectation_l233_23329

/-- Represents the number of red and white balls in the box -/
structure BallCount where
  red : ℕ
  white : ℕ

/-- Represents the two lottery options -/
inductive LotteryOption
  | Option1
  | Option2

/-- Calculates the expected value for Option 1 -/
def expectedValueOption1 (initial : BallCount) : ℚ :=
  sorry

/-- Calculates the expected value for Option 2 -/
def expectedValueOption2 (initial : BallCount) : ℚ :=
  sorry

/-- Theorem stating that Option 2 has a higher expected value -/
theorem option2_higher_expectation (initial : BallCount) :
  initial.red = 3 ∧ initial.white = 3 →
  expectedValueOption2 initial > expectedValueOption1 initial :=
sorry

end NUMINAMATH_CALUDE_option2_higher_expectation_l233_23329


namespace NUMINAMATH_CALUDE_delaney_travel_time_l233_23368

/-- The time (in minutes) when the bus leaves, relative to midnight -/
def bus_departure_time : ℕ := 8 * 60

/-- The time (in minutes) when Delaney left home, relative to midnight -/
def delaney_departure_time : ℕ := 7 * 60 + 50

/-- The time (in minutes) that Delaney missed the bus by -/
def missed_by : ℕ := 20

/-- The time (in minutes) it takes Delaney to reach the pick-up point -/
def travel_time : ℕ := bus_departure_time + missed_by - delaney_departure_time

theorem delaney_travel_time : travel_time = 30 := by
  sorry

end NUMINAMATH_CALUDE_delaney_travel_time_l233_23368


namespace NUMINAMATH_CALUDE_equation_with_positive_root_l233_23393

theorem equation_with_positive_root (x m : ℝ) : 
  ((x - 2) / (x + 1) = m / (x + 1) ∧ x > 0) → m = -3 :=
by sorry

end NUMINAMATH_CALUDE_equation_with_positive_root_l233_23393


namespace NUMINAMATH_CALUDE_fraction_operation_result_l233_23322

theorem fraction_operation_result (x : ℝ) : 
  x = 2.5 → ((x / (1 / 2)) * x) / ((x * (1 / 2)) / x) = 25 := by
  sorry

end NUMINAMATH_CALUDE_fraction_operation_result_l233_23322


namespace NUMINAMATH_CALUDE_ellipse_line_intersection_l233_23354

/-- An ellipse with given properties and a line intersecting it -/
structure EllipseWithLine where
  a : ℝ
  b : ℝ
  k : ℝ
  h_a_pos : 0 < a
  h_b_pos : 0 < b
  h_a_gt_b : b < a
  h_right_focus : (1 : ℝ) = a * (a^2 - b^2).sqrt / a
  h_eccentricity : (a^2 - b^2).sqrt / a = 1/2
  h_ellipse_eq : ∀ x y : ℝ, x^2/a^2 + y^2/b^2 = 1 ↔ (x, y) ∈ {p : ℝ × ℝ | p.1^2/a^2 + p.2^2/b^2 = 1}
  h_line_eq : ∀ x : ℝ, (x, k*x + 1) ∈ {p : ℝ × ℝ | p.1^2/a^2 + p.2^2/b^2 = 1}
  h_intersect : ∃ A B : ℝ × ℝ, A ∈ {p : ℝ × ℝ | p.1^2/a^2 + p.2^2/b^2 = 1} ∧ 
                                B ∈ {p : ℝ × ℝ | p.1^2/a^2 + p.2^2/b^2 = 1} ∧
                                A.2 = k * A.1 + 1 ∧ B.2 = k * B.1 + 1 ∧ A ≠ B
  h_midpoints : ∀ A B : ℝ × ℝ, A ∈ {p : ℝ × ℝ | p.1^2/a^2 + p.2^2/b^2 = 1} ∧ 
                                B ∈ {p : ℝ × ℝ | p.1^2/a^2 + p.2^2/b^2 = 1} ∧
                                A.2 = k * A.1 + 1 ∧ B.2 = k * B.1 + 1 ∧ A ≠ B →
                ∃ M N : ℝ × ℝ, M = ((A.1 + 1)/2, A.2/2) ∧ N = ((B.1 + 1)/2, B.2/2)
  h_origin_on_circle : ∀ M N : ℝ × ℝ, M.1 * N.1 + M.2 * N.2 = 0

/-- The main theorem: given the ellipse and line with specified properties, k = -1/2 -/
theorem ellipse_line_intersection (e : EllipseWithLine) : e.k = -1/2 :=
sorry

end NUMINAMATH_CALUDE_ellipse_line_intersection_l233_23354


namespace NUMINAMATH_CALUDE_banknote_replacement_theorem_l233_23326

/-- Represents the state of the banknote replacement process -/
structure BanknoteState where
  total_banknotes : ℕ
  remaining_banknotes : ℕ
  budget : ℕ
  days : ℕ

/-- Calculates the number of banknotes that can be replaced on a given day -/
def replace_banknotes (state : BanknoteState) (day : ℕ) : ℕ :=
  min state.remaining_banknotes (state.remaining_banknotes / (day + 1))

/-- Updates the state after a day of replacement -/
def update_state (state : BanknoteState) (day : ℕ) : BanknoteState :=
  let replaced := replace_banknotes state day
  { state with
    remaining_banknotes := state.remaining_banknotes - replaced
    budget := state.budget - 90000
    days := state.days + 1 }

/-- Checks if the budget is exceeded -/
def budget_exceeded (state : BanknoteState) : Prop :=
  state.budget < 0

/-- Checks if 80% of banknotes have been replaced -/
def eighty_percent_replaced (state : BanknoteState) : Prop :=
  state.remaining_banknotes ≤ state.total_banknotes / 5

/-- Main theorem statement -/
theorem banknote_replacement_theorem (initial_state : BanknoteState)
    (h_total : initial_state.total_banknotes = 3628800)
    (h_budget : initial_state.budget = 1000000) :
    ∃ (final_state : BanknoteState),
      final_state.days ≥ 4 ∧
      eighty_percent_replaced final_state ∧
      ¬∃ (complete_state : BanknoteState),
        complete_state.remaining_banknotes = 0 ∧
        ¬budget_exceeded complete_state :=
  sorry


end NUMINAMATH_CALUDE_banknote_replacement_theorem_l233_23326


namespace NUMINAMATH_CALUDE_kathys_candy_collection_l233_23365

theorem kathys_candy_collection (num_groups : ℕ) (candies_per_group : ℕ) 
  (h1 : num_groups = 10) (h2 : candies_per_group = 3) : 
  num_groups * candies_per_group = 30 := by
  sorry

end NUMINAMATH_CALUDE_kathys_candy_collection_l233_23365


namespace NUMINAMATH_CALUDE_real_numbers_greater_than_eight_is_set_real_numbers_greater_than_eight_definite_membership_real_numbers_greater_than_eight_fixed_standards_l233_23371

-- Define the property of being greater than 8
def GreaterThanEight (x : ℝ) : Prop := x > 8

-- Define the set of real numbers greater than 8
def RealNumbersGreaterThanEight : Set ℝ := {x : ℝ | GreaterThanEight x}

-- Theorem stating that RealNumbersGreaterThanEight is a well-defined set
theorem real_numbers_greater_than_eight_is_set :
  ∀ (x : ℝ), x ∈ RealNumbersGreaterThanEight ↔ GreaterThanEight x :=
by
  sorry

-- Theorem stating that RealNumbersGreaterThanEight has definite membership criteria
theorem real_numbers_greater_than_eight_definite_membership :
  ∀ (x : ℝ), Decidable (x ∈ RealNumbersGreaterThanEight) :=
by
  sorry

-- Theorem stating that RealNumbersGreaterThanEight has fixed standards for inclusion
theorem real_numbers_greater_than_eight_fixed_standards :
  ∀ (x y : ℝ), x > 8 ∧ y > 8 → (x ∈ RealNumbersGreaterThanEight ∧ y ∈ RealNumbersGreaterThanEight) :=
by
  sorry

end NUMINAMATH_CALUDE_real_numbers_greater_than_eight_is_set_real_numbers_greater_than_eight_definite_membership_real_numbers_greater_than_eight_fixed_standards_l233_23371


namespace NUMINAMATH_CALUDE_negation_existential_square_plus_one_less_than_zero_l233_23362

theorem negation_existential_square_plus_one_less_than_zero :
  (¬ ∃ x : ℝ, x^2 + 1 < 0) ↔ (∀ x : ℝ, x^2 + 1 ≥ 0) := by sorry

end NUMINAMATH_CALUDE_negation_existential_square_plus_one_less_than_zero_l233_23362


namespace NUMINAMATH_CALUDE_probability_at_least_four_same_l233_23331

/-- The number of sides on each die -/
def num_sides : ℕ := 6

/-- The number of dice rolled -/
def num_dice : ℕ := 5

/-- The probability of rolling a specific number on a fair die -/
def prob_single : ℚ := 1 / num_sides

/-- The probability that at least four out of five fair six-sided dice show the same value -/
def prob_at_least_four_same : ℚ := 13 / 648

/-- Theorem stating that the probability of at least four out of five fair six-sided dice 
    showing the same value is 13/648 -/
theorem probability_at_least_four_same : 
  prob_at_least_four_same = (1 / num_sides^4) + (5 * (1 / num_sides^3) * (5 / 6)) :=
by sorry

end NUMINAMATH_CALUDE_probability_at_least_four_same_l233_23331


namespace NUMINAMATH_CALUDE_tiles_in_row_l233_23361

/-- Given a rectangular room with area 144 sq ft and length twice the width,
    prove that 25 tiles of size 4 inches by 4 inches fit in a row along the width. -/
theorem tiles_in_row (room_area : ℝ) (tile_size : ℝ) : 
  room_area = 144 →
  tile_size = 4 →
  ⌊(12 * (144 / 2).sqrt) / tile_size⌋ = 25 := by sorry

end NUMINAMATH_CALUDE_tiles_in_row_l233_23361


namespace NUMINAMATH_CALUDE_deal_or_no_deal_probability_l233_23309

/-- The set of values in the Deal or No Deal game -/
def deal_values : Finset ℕ := {1, 5, 10, 25, 50, 75, 100, 200, 300, 400, 500, 750, 1000, 5000, 10000, 25000, 50000, 75000, 100000, 200000, 300000, 400000, 500000, 750000, 1000000}

/-- The number of boxes in the game -/
def total_boxes : ℕ := 26

/-- The threshold value for high-value boxes -/
def threshold : ℕ := 200000

/-- The set of high-value boxes -/
def high_value_boxes : Finset ℕ := deal_values.filter (λ x => x ≥ threshold)

/-- The number of boxes to eliminate -/
def boxes_to_eliminate : ℕ := 14

theorem deal_or_no_deal_probability :
  (total_boxes - boxes_to_eliminate) / 2 = high_value_boxes.card ∧
  (total_boxes - boxes_to_eliminate) % 2 = 0 :=
sorry

#eval deal_values.card
#eval total_boxes
#eval high_value_boxes
#eval boxes_to_eliminate

end NUMINAMATH_CALUDE_deal_or_no_deal_probability_l233_23309


namespace NUMINAMATH_CALUDE_stratified_sampling_male_athletes_l233_23391

theorem stratified_sampling_male_athletes 
  (total_male : ℕ) 
  (total_female : ℕ) 
  (sample_size : ℕ) 
  (h1 : total_male = 48) 
  (h2 : total_female = 36) 
  (h3 : sample_size = 21) : 
  ℕ :=
  12

#check stratified_sampling_male_athletes

end NUMINAMATH_CALUDE_stratified_sampling_male_athletes_l233_23391


namespace NUMINAMATH_CALUDE_cbd_represents_115_l233_23334

/-- Represents the encoding of a base 5 digit --/
inductive Encoding
| A
| B
| C
| D
| E

/-- Represents a coded number as a list of Encodings --/
def CodedNumber := List Encoding

/-- Converts a CodedNumber to its base 10 representation --/
def to_base_10 (code : CodedNumber) : ℕ := sorry

/-- Checks if two CodedNumbers are consecutive --/
def are_consecutive (a b : CodedNumber) : Prop := sorry

theorem cbd_represents_115 
  (h1 : are_consecutive [Encoding.A, Encoding.B, Encoding.C] [Encoding.A, Encoding.B, Encoding.D])
  (h2 : are_consecutive [Encoding.A, Encoding.B, Encoding.D] [Encoding.A, Encoding.C, Encoding.E])
  (h3 : are_consecutive [Encoding.A, Encoding.C, Encoding.E] [Encoding.A, Encoding.D, Encoding.A]) :
  to_base_10 [Encoding.C, Encoding.B, Encoding.D] = 115 := by sorry

end NUMINAMATH_CALUDE_cbd_represents_115_l233_23334


namespace NUMINAMATH_CALUDE_students_per_group_l233_23373

theorem students_per_group 
  (total_students : ℕ) 
  (num_groups : ℕ) 
  (h1 : total_students = 30) 
  (h2 : num_groups = 6) 
  (h3 : total_students % num_groups = 0) :
  total_students / num_groups = 5 := by
sorry

end NUMINAMATH_CALUDE_students_per_group_l233_23373


namespace NUMINAMATH_CALUDE_basketball_games_total_l233_23341

theorem basketball_games_total (games_won games_lost : ℕ) : 
  games_won - games_lost = 28 → games_won = 45 → games_lost = 17 → 
  games_won + games_lost = 62 := by
  sorry

end NUMINAMATH_CALUDE_basketball_games_total_l233_23341


namespace NUMINAMATH_CALUDE_eg_length_l233_23350

-- Define the quadrilateral EFGH
structure Quadrilateral :=
  (E F G H : ℝ × ℝ)

-- Define the properties of the quadrilateral
def is_valid_quadrilateral (q : Quadrilateral) : Prop :=
  let (ex, ey) := q.E
  let (fx, fy) := q.F
  let (gx, gy) := q.G
  let (hx, hy) := q.H
  -- EF = 6
  (ex - fx)^2 + (ey - fy)^2 = 36 ∧
  -- FG = 18
  (fx - gx)^2 + (fy - gy)^2 = 324 ∧
  -- GH = 6
  (gx - hx)^2 + (gy - hy)^2 = 36 ∧
  -- HE = 10
  (hx - ex)^2 + (hy - ey)^2 = 100 ∧
  -- Angle EFG is a right angle
  (ex - fx) * (gx - fx) + (ey - fy) * (gy - fy) = 0

-- Theorem statement
theorem eg_length (q : Quadrilateral) (h : is_valid_quadrilateral q) :
  let (ex, ey) := q.E
  let (gx, gy) := q.G
  (ex - gx)^2 + (ey - gy)^2 = 360 := by
  sorry

end NUMINAMATH_CALUDE_eg_length_l233_23350


namespace NUMINAMATH_CALUDE_quadratic_inequality_empty_solution_l233_23347

theorem quadratic_inequality_empty_solution (a : ℝ) :
  (∀ x : ℝ, x^2 + a*x + 4 ≥ 0) → a ∈ Set.Icc (-4 : ℝ) 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_empty_solution_l233_23347


namespace NUMINAMATH_CALUDE_conditional_probability_B_given_A_l233_23389

/-- The total number of zongzi -/
def total_zongzi : ℕ := 5

/-- The number of zongzi with pork filling -/
def pork_zongzi : ℕ := 2

/-- The number of zongzi with red bean paste filling -/
def red_bean_zongzi : ℕ := 3

/-- Event A: the two picked zongzi have the same filling -/
def event_A : Set (Fin total_zongzi × Fin total_zongzi) := sorry

/-- Event B: the two picked zongzi both have red bean paste filling -/
def event_B : Set (Fin total_zongzi × Fin total_zongzi) := sorry

/-- The probability measure on the sample space -/
def P : Set (Fin total_zongzi × Fin total_zongzi) → ℝ := sorry

theorem conditional_probability_B_given_A :
  P event_B / P event_A = 3/4 := by sorry

end NUMINAMATH_CALUDE_conditional_probability_B_given_A_l233_23389


namespace NUMINAMATH_CALUDE_complex_real_condition_l233_23379

theorem complex_real_condition (i : ℂ) (m : ℝ) : 
  i * i = -1 →
  (1 / (2 + i) + m * i).im = 0 →
  m = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_real_condition_l233_23379


namespace NUMINAMATH_CALUDE_students_in_chemistry_or_physics_not_both_l233_23305

theorem students_in_chemistry_or_physics_not_both (total_chemistry : ℕ) (both : ℕ) (only_physics : ℕ)
  (h1 : both = 15)
  (h2 : total_chemistry = 30)
  (h3 : only_physics = 12) :
  total_chemistry - both + only_physics = 27 :=
by sorry

end NUMINAMATH_CALUDE_students_in_chemistry_or_physics_not_both_l233_23305


namespace NUMINAMATH_CALUDE_fraction_equality_l233_23396

theorem fraction_equality (x y : ℝ) (h : x / y = 2 / 5) : 
  ((x + 3 * y) / (2 * y) ≠ 13 / 10) ∧ 
  ((2 * x) / (y - x) = 4 / 3) ∧ 
  ((x + 5 * y) / (2 * x) = 27 / 4) ∧ 
  ((2 * y - x) / (3 * y) ≠ 7 / 15) ∧ 
  (y / (3 * x) = 5 / 6) := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l233_23396


namespace NUMINAMATH_CALUDE_parallel_range_perpendicular_min_abs_product_l233_23383

-- Define the lines l₁ and l₂
def l₁ (a : ℝ) (x y : ℝ) : Prop := x + a^2 * y + 1 = 0
def l₂ (a b : ℝ) (x y : ℝ) : Prop := (a^2 + 1) * x - b * y + 3 = 0

-- Define parallel lines
def parallel (a b : ℝ) : Prop := ∀ x y, l₁ a x y ↔ l₂ a b x y

-- Define perpendicular lines
def perpendicular (a b : ℝ) : Prop := ∀ x₁ y₁ x₂ y₂, 
  l₁ a x₁ y₁ → l₂ a b x₂ y₂ → (x₁ - x₂) * (y₁ - y₂) = 0

-- Statement 1: If l₁ ∥ l₂, then b ∈ (-∞, -6) ∪ (-6, 0]
theorem parallel_range (a b : ℝ) : 
  parallel a b → b < -6 ∨ (-6 < b ∧ b ≤ 0) :=
sorry

-- Statement 2: If l₁ ⟂ l₂, then the minimum value of |ab| is 2
theorem perpendicular_min_abs_product (a b : ℝ) :
  perpendicular a b → |a * b| ≥ 2 :=
sorry

end NUMINAMATH_CALUDE_parallel_range_perpendicular_min_abs_product_l233_23383


namespace NUMINAMATH_CALUDE_alpine_school_math_players_l233_23358

/-- The number of players taking mathematics in Alpine School -/
def mathematics_players (total_players physics_players both_players : ℕ) : ℕ :=
  total_players - (physics_players - both_players)

/-- Theorem: Given the conditions, prove that 10 players are taking mathematics -/
theorem alpine_school_math_players :
  mathematics_players 15 9 4 = 10 := by
  sorry

end NUMINAMATH_CALUDE_alpine_school_math_players_l233_23358


namespace NUMINAMATH_CALUDE_sevenPeopleRoundTable_l233_23386

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def seatingArrangements (totalPeople : ℕ) (adjacentPair : ℕ) : ℕ :=
  if totalPeople ≤ 1 then 0
  else
    let effectiveUnits := totalPeople - adjacentPair + 1
    (factorial effectiveUnits * adjacentPair) / totalPeople

theorem sevenPeopleRoundTable :
  seatingArrangements 7 2 = 240 := by
  sorry

end NUMINAMATH_CALUDE_sevenPeopleRoundTable_l233_23386


namespace NUMINAMATH_CALUDE_apple_selling_price_l233_23363

/-- Calculates the selling price of an apple given its cost price and loss fraction. -/
def selling_price (cost_price : ℝ) (loss_fraction : ℝ) : ℝ :=
  cost_price * (1 - loss_fraction)

/-- Theorem stating the selling price of an apple given specific conditions. -/
theorem apple_selling_price :
  let cost_price : ℝ := 19
  let loss_fraction : ℝ := 1/6
  let calculated_price := selling_price cost_price loss_fraction
  ∃ ε > 0, |calculated_price - 15.83| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_apple_selling_price_l233_23363


namespace NUMINAMATH_CALUDE_middle_number_is_four_l233_23317

/-- Represents a triple of positive integers -/
structure Triple where
  left : Nat
  middle : Nat
  right : Nat
  left_pos : 0 < left
  middle_pos : 0 < middle
  right_pos : 0 < right

/-- Checks if a triple satisfies the problem conditions -/
def validTriple (t : Triple) : Prop :=
  t.left < t.middle ∧ t.middle < t.right ∧ t.left + t.middle + t.right = 15

/-- Casey cannot determine the other two numbers -/
def caseyUncertain (t : Triple) : Prop :=
  ∃ t' : Triple, t'.left = t.left ∧ validTriple t' ∧ t' ≠ t

/-- Tracy cannot determine the other two numbers -/
def tracyUncertain (t : Triple) : Prop :=
  ∃ t' : Triple, t'.right = t.right ∧ validTriple t' ∧ t' ≠ t

/-- Stacy cannot determine the other two numbers -/
def stacyUncertain (t : Triple) : Prop :=
  ∃ t' : Triple, t'.middle = t.middle ∧ validTriple t' ∧ t' ≠ t

/-- The main theorem stating that the middle number must be 4 -/
theorem middle_number_is_four :
  ∀ t : Triple,
    validTriple t →
    caseyUncertain t →
    tracyUncertain t →
    stacyUncertain t →
    t.middle = 4 := by
  sorry


end NUMINAMATH_CALUDE_middle_number_is_four_l233_23317


namespace NUMINAMATH_CALUDE_youtube_views_problem_l233_23312

/-- Calculates the additional views after the fourth day given the initial views,
    increase factor, and total views after 6 days. -/
def additional_views_after_fourth_day (initial_views : ℕ) (increase_factor : ℕ) (total_views_after_six_days : ℕ) : ℕ :=
  total_views_after_six_days - (initial_views + increase_factor * initial_views)

/-- Theorem stating that given the specific conditions of the problem,
    the additional views after the fourth day is 50000. -/
theorem youtube_views_problem :
  additional_views_after_fourth_day 4000 10 94000 = 50000 := by
  sorry

end NUMINAMATH_CALUDE_youtube_views_problem_l233_23312


namespace NUMINAMATH_CALUDE_continuous_function_integrable_l233_23366

theorem continuous_function_integrable 
  {a b : ℝ} (f : ℝ → ℝ) (h : ContinuousOn f (Set.Icc a b)) : 
  IntervalIntegrable f volume a b :=
sorry

end NUMINAMATH_CALUDE_continuous_function_integrable_l233_23366


namespace NUMINAMATH_CALUDE_side_bc_equation_proof_l233_23374

/-- A triangle with two known altitudes and one known vertex -/
structure Triangle where
  -- First altitude equation: 2x - 3y + 1 = 0
  altitude1 : ℝ → ℝ → Prop
  altitude1_eq : ∀ x y, altitude1 x y ↔ 2 * x - 3 * y + 1 = 0

  -- Second altitude equation: x + y = 0
  altitude2 : ℝ → ℝ → Prop
  altitude2_eq : ∀ x y, altitude2 x y ↔ x + y = 0

  -- Vertex A coordinates
  vertex_a : ℝ × ℝ
  vertex_a_def : vertex_a = (1, 2)

/-- The equation of the line on which side BC lies -/
def side_bc_equation (t : Triangle) (x y : ℝ) : Prop :=
  2 * x + 3 * y + 7 = 0

/-- Theorem stating that the equation of side BC is 2x + 3y + 7 = 0 -/
theorem side_bc_equation_proof (t : Triangle) :
  ∀ x y, side_bc_equation t x y ↔ 2 * x + 3 * y + 7 = 0 :=
sorry

end NUMINAMATH_CALUDE_side_bc_equation_proof_l233_23374


namespace NUMINAMATH_CALUDE_purchase_cost_l233_23315

/-- The cost of a single sandwich in dollars -/
def sandwich_cost : ℕ := 4

/-- The cost of a single soda in dollars -/
def soda_cost : ℕ := 3

/-- The cost of a single cookie in dollars -/
def cookie_cost : ℕ := 2

/-- The number of sandwiches purchased -/
def num_sandwiches : ℕ := 4

/-- The number of sodas purchased -/
def num_sodas : ℕ := 6

/-- The number of cookies purchased -/
def num_cookies : ℕ := 7

/-- The total cost of the purchase in dollars -/
def total_cost : ℕ := sandwich_cost * num_sandwiches + soda_cost * num_sodas + cookie_cost * num_cookies

theorem purchase_cost : total_cost = 48 := by
  sorry

end NUMINAMATH_CALUDE_purchase_cost_l233_23315


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l233_23345

theorem partial_fraction_decomposition (N₁ N₂ : ℚ) :
  (∀ x : ℚ, x ≠ 1 → x ≠ 3 →
    (56 * x - 14) / (x^2 - 4*x + 3) = N₁ / (x - 1) + N₂ / (x - 3)) →
  N₁ * N₂ = -1617 := by
sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l233_23345


namespace NUMINAMATH_CALUDE_original_equals_scientific_l233_23388

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- The number to be converted -/
def original_number : ℕ := 28000

/-- The scientific notation representation of the original number -/
def scientific_representation : ScientificNotation :=
  { coefficient := 2.8
    exponent := 4
    is_valid := by sorry }

/-- Theorem stating that the original number is equal to its scientific notation representation -/
theorem original_equals_scientific :
  (original_number : ℝ) = scientific_representation.coefficient * (10 : ℝ) ^ scientific_representation.exponent :=
by sorry

end NUMINAMATH_CALUDE_original_equals_scientific_l233_23388


namespace NUMINAMATH_CALUDE_log_243_between_consecutive_integers_l233_23308

theorem log_243_between_consecutive_integers (a b : ℤ) :
  (a : ℝ) < Real.log 243 / Real.log 5 ∧
  Real.log 243 / Real.log 5 < (b : ℝ) ∧
  b = a + 1 →
  a + b = 7 := by sorry

end NUMINAMATH_CALUDE_log_243_between_consecutive_integers_l233_23308


namespace NUMINAMATH_CALUDE_homework_problem_l233_23390

theorem homework_problem (a b c d : ℤ) 
  (h1 : a = -1) 
  (h2 : b = -c) 
  (h3 : d = -2) : 
  4*a + (b + c) - |3*d| = -10 := by
  sorry

end NUMINAMATH_CALUDE_homework_problem_l233_23390


namespace NUMINAMATH_CALUDE_committee_formation_proof_l233_23384

/-- The number of ways to choose k elements from n elements -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

theorem committee_formation_proof :
  let total_students : ℕ := 8
  let committee_size : ℕ := 5
  let always_included : ℕ := 2
  let remaining_students : ℕ := total_students - always_included
  let students_to_choose : ℕ := committee_size - always_included
  choose remaining_students students_to_choose = 20 := by
  sorry

end NUMINAMATH_CALUDE_committee_formation_proof_l233_23384


namespace NUMINAMATH_CALUDE_wall_building_time_l233_23340

/-- The number of days required for a group of workers to build a wall, given:
  * The number of workers in the reference group
  * The length of the wall built by the reference group
  * The number of days taken by the reference group
  * The number of workers in the new group
  * The length of the wall to be built by the new group
-/
def days_required (
  ref_workers : ℕ
  ) (ref_length : ℕ
  ) (ref_days : ℕ
  ) (new_workers : ℕ
  ) (new_length : ℕ
  ) : ℚ :=
  (ref_workers * ref_days * new_length : ℚ) / (new_workers * ref_length)

/-- Theorem stating that 30 workers will take 18 days to build a 100m wall,
    given that 18 workers can build a 140m wall in 42 days -/
theorem wall_building_time :
  days_required 18 140 42 30 100 = 18 := by
  sorry

end NUMINAMATH_CALUDE_wall_building_time_l233_23340


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l233_23338

-- Define arithmetic sequences a_n and b_n
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

-- Define the problem statement
theorem arithmetic_sequence_sum (a b : ℕ → ℝ) :
  arithmetic_sequence a →
  arithmetic_sequence b →
  a 1 = 25 →
  b 1 = 75 →
  a 2 + b 2 = 100 →
  a 37 + b 37 = 100 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l233_23338


namespace NUMINAMATH_CALUDE_min_moves_is_22_l233_23378

/-- A move consists of transferring one coin to an adjacent box. -/
def Move := ℕ

/-- The configuration of coins in the boxes. -/
def Configuration := Fin 7 → ℕ

/-- The initial configuration of coins in the boxes. -/
def initial_config : Configuration :=
  fun i => [5, 8, 11, 17, 20, 15, 10].get i

/-- A configuration is balanced if all boxes have the same number of coins. -/
def is_balanced (c : Configuration) : Prop :=
  ∀ i j : Fin 7, c i = c j

/-- The number of moves required to transform one configuration into another. -/
def moves_required (start finish : Configuration) : ℕ := sorry

/-- The minimum number of moves required to balance the configuration. -/
def min_moves_to_balance (c : Configuration) : ℕ := sorry

/-- The theorem stating that the minimum number of moves required to balance
    the initial configuration is 22. -/
theorem min_moves_is_22 :
  min_moves_to_balance initial_config = 22 := by sorry

end NUMINAMATH_CALUDE_min_moves_is_22_l233_23378


namespace NUMINAMATH_CALUDE_smallest_square_sum_12_consecutive_l233_23327

/-- The sum of 12 consecutive integers starting from n -/
def sum_12_consecutive (n : ℕ) : ℕ := 6 * (2 * n + 11)

/-- Predicate to check if a number is a perfect square -/
def is_perfect_square (m : ℕ) : Prop := ∃ k : ℕ, m = k * k

theorem smallest_square_sum_12_consecutive :
  (∀ n : ℕ, n > 0 → sum_12_consecutive n < 150 → ¬ is_perfect_square (sum_12_consecutive n)) ∧
  is_perfect_square 150 ∧
  (∃ n : ℕ, n > 0 ∧ sum_12_consecutive n = 150) :=
sorry

end NUMINAMATH_CALUDE_smallest_square_sum_12_consecutive_l233_23327


namespace NUMINAMATH_CALUDE_simplify_trig_expression_l233_23319

theorem simplify_trig_expression (α : Real) (h : 270 * π / 180 < α ∧ α < 360 * π / 180) :
  Real.sqrt ((1/2) + (1/2) * Real.sqrt ((1/2) + (1/2) * Real.cos (2 * α))) = -Real.cos (α / 2) := by
  sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_l233_23319


namespace NUMINAMATH_CALUDE_lemonade_water_amount_solution_is_correct_l233_23323

/-- Represents the recipe for lemonade --/
structure LemonadeRecipe where
  water : ℝ
  sugar : ℝ
  lemon_juice : ℝ

/-- Checks if the recipe satisfies the given ratios --/
def is_valid_recipe (r : LemonadeRecipe) : Prop :=
  r.water = 5 * r.sugar ∧ r.sugar = 3 * r.lemon_juice

/-- The main theorem: given the ratios and lemon juice amount, prove the water amount --/
theorem lemonade_water_amount (r : LemonadeRecipe) 
  (h1 : is_valid_recipe r) (h2 : r.lemon_juice = 5) : r.water = 75 := by
  sorry

/-- Proof that our solution is correct --/
theorem solution_is_correct : ∃ r : LemonadeRecipe, 
  is_valid_recipe r ∧ r.lemon_juice = 5 ∧ r.water = 75 := by
  sorry

end NUMINAMATH_CALUDE_lemonade_water_amount_solution_is_correct_l233_23323


namespace NUMINAMATH_CALUDE_intersection_collinearity_l233_23335

/-- A point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- A line in 2D space -/
structure Line :=
  (a : ℝ) (b : ℝ) (c : ℝ)

/-- Quadrilateral ABCD -/
structure Quadrilateral :=
  (A B C D : Point)

/-- Check if three points are collinear -/
def collinear (P Q R : Point) : Prop :=
  (Q.y - P.y) * (R.x - P.x) = (R.y - P.y) * (Q.x - P.x)

/-- The main theorem -/
theorem intersection_collinearity 
  (ABCD : Quadrilateral) 
  (P Q : Point) 
  (l : Line) 
  (E F : Point) 
  (R S T : Point) :
  (∃ (l1 : Line), l1.a * ABCD.A.x + l1.b * ABCD.A.y + l1.c = 0 ∧ 
                  l1.a * ABCD.B.x + l1.b * ABCD.B.y + l1.c = 0 ∧ 
                  l1.a * P.x + l1.b * P.y + l1.c = 0) →  -- AB extended through P
  (∃ (l2 : Line), l2.a * ABCD.C.x + l2.b * ABCD.C.y + l2.c = 0 ∧ 
                  l2.a * ABCD.D.x + l2.b * ABCD.D.y + l2.c = 0 ∧ 
                  l2.a * P.x + l2.b * P.y + l2.c = 0) →  -- CD extended through P
  (∃ (l3 : Line), l3.a * ABCD.B.x + l3.b * ABCD.B.y + l3.c = 0 ∧ 
                  l3.a * ABCD.C.x + l3.b * ABCD.C.y + l3.c = 0 ∧ 
                  l3.a * Q.x + l3.b * Q.y + l3.c = 0) →  -- BC extended through Q
  (∃ (l4 : Line), l4.a * ABCD.A.x + l4.b * ABCD.A.y + l4.c = 0 ∧ 
                  l4.a * ABCD.D.x + l4.b * ABCD.D.y + l4.c = 0 ∧ 
                  l4.a * Q.x + l4.b * Q.y + l4.c = 0) →  -- AD extended through Q
  (l.a * P.x + l.b * P.y + l.c = 0) →  -- P is on line l
  (l.a * E.x + l.b * E.y + l.c = 0) →  -- E is on line l
  (l.a * F.x + l.b * F.y + l.c = 0) →  -- F is on line l
  (∃ (l5 l6 : Line), l5.a * ABCD.A.x + l5.b * ABCD.A.y + l5.c = 0 ∧ 
                     l5.a * ABCD.C.x + l5.b * ABCD.C.y + l5.c = 0 ∧ 
                     l6.a * ABCD.B.x + l6.b * ABCD.B.y + l6.c = 0 ∧ 
                     l6.a * ABCD.D.x + l6.b * ABCD.D.y + l6.c = 0 ∧ 
                     l5.a * R.x + l5.b * R.y + l5.c = 0 ∧ 
                     l6.a * R.x + l6.b * R.y + l6.c = 0) →  -- R is intersection of AC and BD
  (∃ (l7 l8 : Line), l7.a * ABCD.A.x + l7.b * ABCD.A.y + l7.c = 0 ∧ 
                     l7.a * E.x + l7.b * E.y + l7.c = 0 ∧ 
                     l8.a * ABCD.B.x + l8.b * ABCD.B.y + l8.c = 0 ∧ 
                     l8.a * F.x + l8.b * F.y + l8.c = 0 ∧ 
                     l7.a * S.x + l7.b * S.y + l7.c = 0 ∧ 
                     l8.a * S.x + l8.b * S.y + l8.c = 0) →  -- S is intersection of AE and BF
  (∃ (l9 l10 : Line), l9.a * ABCD.C.x + l9.b * ABCD.C.y + l9.c = 0 ∧ 
                      l9.a * F.x + l9.b * F.y + l9.c = 0 ∧ 
                      l10.a * ABCD.D.x + l10.b * ABCD.D.y + l10.c = 0 ∧ 
                      l10.a * E.x + l10.b * E.y + l10.c = 0 ∧ 
                      l9.a * T.x + l9.b * T.y + l9.c = 0 ∧ 
                      l10.a * T.x + l10.b * T.y + l10.c = 0) →  -- T is intersection of CF and DE
  collinear R S T ∧ collinear R S Q :=
by sorry

end NUMINAMATH_CALUDE_intersection_collinearity_l233_23335


namespace NUMINAMATH_CALUDE_function_properties_l233_23360

noncomputable def f (x : ℝ) := Real.sqrt 3 * Real.sin x * Real.cos x + Real.sin x ^ 2 - 3 / 2

theorem function_properties :
  (∃ (p : ℝ), p > 0 ∧ ∀ (x : ℝ), f (x + p) = f x ∧ ∀ (q : ℝ), q > 0 ∧ (∀ (x : ℝ), f (x + q) = f x) → p ≤ q) ∧
  (∀ (A B C a b c : ℝ),
    f C = 0 →
    c = 3 →
    2 * Real.sin A - Real.sin B = 0 →
    a ^ 2 + b ^ 2 - 2 * a * b * Real.cos C = c ^ 2 →
    a = Real.sqrt 3 ∧ b = 2 * Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l233_23360


namespace NUMINAMATH_CALUDE_trigonometric_identity_l233_23314

theorem trigonometric_identity (α : ℝ) (h : Real.sin α + Real.cos α = 2/3) :
  (2 * Real.sin α ^ 2 + 2 * Real.sin α * Real.cos α) / (1 + Real.tan α) = -5/9 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l233_23314


namespace NUMINAMATH_CALUDE_famous_artists_not_set_l233_23346

/-- A structure representing a collection of objects -/
structure Collection where
  elements : Set α
  is_definite : Bool
  is_distinct : Bool
  is_unordered : Bool

/-- Definition of a set -/
def is_set (c : Collection) : Prop :=
  c.is_definite ∧ c.is_distinct ∧ c.is_unordered

/-- Famous artists collection -/
def famous_artists : Collection := sorry

/-- Theorem stating that famous artists cannot form a set -/
theorem famous_artists_not_set : ¬(is_set famous_artists) := by
  sorry

end NUMINAMATH_CALUDE_famous_artists_not_set_l233_23346


namespace NUMINAMATH_CALUDE_ricciana_long_jump_l233_23382

/-- Ricciana's long jump problem -/
theorem ricciana_long_jump (R : ℕ) : R = 20 :=
  let ricciana_jump := 4
  let margarita_run := 18
  let margarita_jump := 2 * ricciana_jump - 1
  let ricciana_total := R + ricciana_jump
  let margarita_total := margarita_run + margarita_jump
  have h1 : margarita_total = ricciana_total + 1 := by sorry
  sorry

#check ricciana_long_jump

end NUMINAMATH_CALUDE_ricciana_long_jump_l233_23382


namespace NUMINAMATH_CALUDE_tabletop_qualification_l233_23325

theorem tabletop_qualification (length width diagonal : ℝ) 
  (h_length : length = 60)
  (h_width : width = 32)
  (h_diagonal : diagonal = 68) : 
  length ^ 2 + width ^ 2 = diagonal ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_tabletop_qualification_l233_23325


namespace NUMINAMATH_CALUDE_rectangle_100_101_diagonal_segments_l233_23392

/-- The number of segments a diagonal is divided into by grid lines in a rectangle -/
def diagonal_segments (width : ℕ) (height : ℕ) : ℕ :=
  width + height - Nat.gcd width height

/-- Theorem: In a 100 × 101 rectangle, the diagonal is divided into 200 segments by grid lines -/
theorem rectangle_100_101_diagonal_segments :
  diagonal_segments 100 101 = 200 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_100_101_diagonal_segments_l233_23392


namespace NUMINAMATH_CALUDE_desk_chair_cost_l233_23387

theorem desk_chair_cost (cost_A cost_B : ℝ) : 
  (cost_B = cost_A + 40) →
  (4 * cost_A + 5 * cost_B = 1820) →
  (cost_A = 180 ∧ cost_B = 220) := by
sorry

end NUMINAMATH_CALUDE_desk_chair_cost_l233_23387


namespace NUMINAMATH_CALUDE_converse_and_inverse_false_l233_23307

-- Define the types
def Quadrilateral : Type := sorry
def Rhombus : Type := sorry
def Parallelogram : Type := sorry

-- Define the properties
def is_rhombus : Quadrilateral → Prop := sorry
def is_parallelogram : Quadrilateral → Prop := sorry

-- Given statement
axiom rhombus_is_parallelogram : ∀ q : Quadrilateral, is_rhombus q → is_parallelogram q

-- Theorem to prove
theorem converse_and_inverse_false : 
  (∃ q : Quadrilateral, is_parallelogram q ∧ ¬is_rhombus q) ∧ 
  (∃ q : Quadrilateral, ¬is_rhombus q ∧ is_parallelogram q) := by
  sorry

end NUMINAMATH_CALUDE_converse_and_inverse_false_l233_23307


namespace NUMINAMATH_CALUDE_correct_quotient_l233_23316

theorem correct_quotient (N : ℕ) (h1 : N % 21 = 0) (h2 : N / 12 = 63) : N / 21 = 36 := by
  sorry

end NUMINAMATH_CALUDE_correct_quotient_l233_23316


namespace NUMINAMATH_CALUDE_limit_x_minus_pi_half_times_tan_x_approaches_pi_half_l233_23333

/-- The limit of (x - π/2) * tan(x) as x approaches π/2 is -1. -/
theorem limit_x_minus_pi_half_times_tan_x_approaches_pi_half :
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 0 < |x - π/2| ∧ |x - π/2| < δ →
    |(x - π/2) * Real.tan x + 1| < ε :=
sorry

end NUMINAMATH_CALUDE_limit_x_minus_pi_half_times_tan_x_approaches_pi_half_l233_23333


namespace NUMINAMATH_CALUDE_product_of_four_integers_l233_23343

theorem product_of_four_integers (P Q R S : ℕ+) : 
  P + Q + R + S = 100 →
  (P : ℚ) + 5 = (Q : ℚ) - 5 →
  (P : ℚ) + 5 = (R : ℚ) * 2 →
  (P : ℚ) + 5 = (S : ℚ) / 2 →
  (P : ℚ) * (Q : ℚ) * (R : ℚ) * (S : ℚ) = 1509400000 / 6561 := by
  sorry

end NUMINAMATH_CALUDE_product_of_four_integers_l233_23343


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l233_23311

/-- Represents an ellipse with foci on the x-axis -/
def is_ellipse_x_axis (m : ℝ) : Prop :=
  m^2 - 1 > 3

/-- The condition m^2 > 5 is sufficient but not necessary for the equation to represent an ellipse with foci on the x-axis -/
theorem sufficient_not_necessary_condition :
  (∀ m : ℝ, m^2 > 5 → is_ellipse_x_axis m) ∧
  (∃ m : ℝ, m^2 ≤ 5 ∧ is_ellipse_x_axis m) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l233_23311


namespace NUMINAMATH_CALUDE_circle_center_trajectory_l233_23377

/-- A moving circle with center (x, y) passes through (1, 0) and is tangent to x = -1 -/
def MovingCircle (x y : ℝ) : Prop :=
  (x - 1)^2 + y^2 = (x + 1)^2

/-- The trajectory of the circle's center satisfies y^2 = 4x -/
theorem circle_center_trajectory (x y : ℝ) :
  MovingCircle x y → y^2 = 4*x := by
  sorry

end NUMINAMATH_CALUDE_circle_center_trajectory_l233_23377


namespace NUMINAMATH_CALUDE_container_volume_scaling_l233_23336

theorem container_volume_scaling (original_volume : ℝ) :
  let scale_factor : ℝ := 2
  let new_volume : ℝ := original_volume * scale_factor^3
  new_volume = 8 * original_volume := by sorry

end NUMINAMATH_CALUDE_container_volume_scaling_l233_23336


namespace NUMINAMATH_CALUDE_right_triangle_perimeter_l233_23302

theorem right_triangle_perimeter (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  a * b / 2 = 150 →
  a = 30 →
  c^2 = a^2 + b^2 →
  a + b + c = 40 + 10 * Real.sqrt 10 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_perimeter_l233_23302


namespace NUMINAMATH_CALUDE_existence_of_point_l233_23348

theorem existence_of_point :
  ∃ (x₀ y₀ z₀ : ℝ),
    (x₀ + y₀ + z₀ ≠ 0) ∧
    (0 < x₀^2 + y₀^2 + z₀^2) ∧
    (x₀^2 + y₀^2 + z₀^2 < 1 / 1999) ∧
    (1.999 < (x₀^2 + y₀^2 + z₀^2) / (x₀ + y₀ + z₀)) ∧
    ((x₀^2 + y₀^2 + z₀^2) / (x₀ + y₀ + z₀) < 2) :=
by sorry

end NUMINAMATH_CALUDE_existence_of_point_l233_23348


namespace NUMINAMATH_CALUDE_empty_solution_set_implies_positive_a_nonpositive_discriminant_l233_23332

/-- The discriminant of a quadratic equation ax^2 + bx + c = 0 -/
def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

/-- The solution set of a quadratic inequality ax^2 + bx + c < 0 -/
def solutionSet (a b c : ℝ) : Set ℝ := {x : ℝ | a*x^2 + b*x + c < 0}

theorem empty_solution_set_implies_positive_a_nonpositive_discriminant
  (a b c : ℝ) (h_a_nonzero : a ≠ 0) :
  IsEmpty (solutionSet a b c) → a > 0 ∧ discriminant a b c ≤ 0 :=
by sorry

end NUMINAMATH_CALUDE_empty_solution_set_implies_positive_a_nonpositive_discriminant_l233_23332


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l233_23320

theorem simplify_and_evaluate (x : ℤ) 
  (h1 : -1 ≤ x ∧ x ≤ 1) 
  (h2 : x ≠ 0) 
  (h3 : x ≠ 1) : 
  ((((x^2 - 1) / (x^2 - 2*x + 1) + 1 / (1 - x)) : ℚ) / (x^2 : ℚ) * (x - 1)) = -1 :=
by sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l233_23320


namespace NUMINAMATH_CALUDE_total_stones_l233_23359

/-- Represents the number of stones in each pile -/
structure StonePiles where
  pile1 : ℕ
  pile2 : ℕ
  pile3 : ℕ
  pile4 : ℕ
  pile5 : ℕ

/-- Defines the conditions for the stone piles -/
def ValidStonePiles (p : StonePiles) : Prop :=
  p.pile5 = 6 * p.pile3 ∧
  p.pile2 = 2 * (p.pile3 + p.pile5) ∧
  p.pile1 = p.pile5 / 3 ∧
  p.pile1 = p.pile4 - 10 ∧
  p.pile4 = 2 * p.pile2

/-- The theorem to be proved -/
theorem total_stones (p : StonePiles) (h : ValidStonePiles p) : 
  p.pile1 + p.pile2 + p.pile3 + p.pile4 + p.pile5 = 60 := by
  sorry

end NUMINAMATH_CALUDE_total_stones_l233_23359


namespace NUMINAMATH_CALUDE_inequality_proof_l233_23353

theorem inequality_proof (a b c d : ℝ) (ha : a ≥ -1) (hb : b ≥ -1) (hc : c ≥ -1) (hd : d ≥ -1) :
  ∃! k : ℝ, ∀ (a b c d : ℝ), a ≥ -1 → b ≥ -1 → c ≥ -1 → d ≥ -1 →
    a^3 + b^3 + c^3 + d^3 + 1 ≥ k * (a + b + c + d) ∧ k = 3/4 :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l233_23353


namespace NUMINAMATH_CALUDE_correct_propositions_l233_23342

-- Define the type for propositions
inductive Proposition
  | one
  | two
  | three
  | four
  | five
  | six
  | seven

-- Define a function to check if a proposition is correct
def is_correct (p : Proposition) : Prop :=
  match p with
  | .two => True
  | .six => True
  | .seven => True
  | _ => False

-- Define the theorem
theorem correct_propositions :
  ∀ p : Proposition, is_correct p ↔ (p = .two ∨ p = .six ∨ p = .seven) :=
by sorry

end NUMINAMATH_CALUDE_correct_propositions_l233_23342


namespace NUMINAMATH_CALUDE_kopeck_enough_for_kvass_l233_23344

/-- Represents the price of bread before any increase -/
def x : ℝ := sorry

/-- Represents the price of kvass before any increase -/
def y : ℝ := sorry

/-- The value of one kopeck -/
def kopeck : ℝ := 1

/-- Initial condition: total spending equals one kopeck -/
axiom initial_condition : x + y = kopeck

/-- Condition after first price increase -/
axiom first_increase : 0.6 * x + 1.2 * y = kopeck

/-- Theorem stating that one kopeck is enough for kvass after two 20% price increases -/
theorem kopeck_enough_for_kvass : kopeck > 1.44 * y := by sorry

end NUMINAMATH_CALUDE_kopeck_enough_for_kvass_l233_23344


namespace NUMINAMATH_CALUDE_quadratic_inequality_l233_23364

theorem quadratic_inequality (x : ℝ) : -x^2 - 2*x + 3 ≥ 0 ↔ -3 ≤ x ∧ x ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l233_23364


namespace NUMINAMATH_CALUDE_composite_fraction_theorem_l233_23310

def first_eight_composites : List Nat := [4, 6, 8, 9, 10, 12, 14, 15]
def next_eight_composites : List Nat := [16, 18, 20, 21, 22, 24, 25, 26]
def first_prime : Nat := 2
def second_prime : Nat := 3

theorem composite_fraction_theorem :
  let numerator := (List.prod first_eight_composites + first_prime)
  let denominator := (List.prod next_eight_composites + second_prime)
  (numerator : ℚ) / denominator = 
    (4 * 6 * 8 * 9 * 10 * 12 * 14 * 15 + 2 : ℚ) / 
    (16 * 18 * 20 * 21 * 22 * 24 * 25 * 26 + 3) := by
  sorry

end NUMINAMATH_CALUDE_composite_fraction_theorem_l233_23310


namespace NUMINAMATH_CALUDE_exchange_process_duration_l233_23381

/-- Represents the number of children of each gender -/
def n : ℕ := 10

/-- Calculates the sum of the first n even numbers -/
def sum_even (n : ℕ) : ℕ := n * (n + 1)

/-- Calculates the sum of the first n natural numbers -/
def sum_natural (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Represents the total number of swaps required to move boys from even positions to the first n positions -/
def total_swaps (n : ℕ) : ℕ := sum_even n - sum_natural n

theorem exchange_process_duration :
  total_swaps n = 55 ∧ total_swaps n < 60 := by sorry

end NUMINAMATH_CALUDE_exchange_process_duration_l233_23381


namespace NUMINAMATH_CALUDE_binomial_12_9_l233_23385

theorem binomial_12_9 : Nat.choose 12 9 = 220 := by sorry

end NUMINAMATH_CALUDE_binomial_12_9_l233_23385


namespace NUMINAMATH_CALUDE_line_bisecting_segment_l233_23375

/-- The equation of a line passing through a point and bisecting a segment between two other lines -/
theorem line_bisecting_segment (M : ℝ × ℝ) (l₁ l₂ : ℝ → ℝ → ℝ) :
  M = (3/2, -1/2) →
  (∀ x y, l₁ x y = 2*x - 5*y + 10) →
  (∀ x y, l₂ x y = 3*x + 8*y + 15) →
  ∃ P₁ P₂ : ℝ × ℝ,
    l₁ P₁.1 P₁.2 = 0 ∧
    l₂ P₂.1 P₂.2 = 0 ∧
    M = ((P₁.1 + P₂.1) / 2, (P₁.2 + P₂.2) / 2) →
  ∃ A B C : ℝ,
    A = 5 ∧ B = 3 ∧ C = -6 ∧
    ∀ x y, A*x + B*y + C = 0 ↔ (y - M.2) / (x - M.1) = -A / B :=
by sorry

end NUMINAMATH_CALUDE_line_bisecting_segment_l233_23375


namespace NUMINAMATH_CALUDE_largest_even_five_digit_number_with_square_and_cube_l233_23367

/-- A function that checks if a number is a perfect square --/
def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

/-- A function that checks if a number is a perfect cube --/
def is_perfect_cube (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m * m = n

/-- A function that returns the first three digits of a 5-digit number --/
def first_three_digits (n : ℕ) : ℕ :=
  n / 100

/-- A function that returns the last three digits of a 5-digit number --/
def last_three_digits (n : ℕ) : ℕ :=
  n % 1000

/-- Main theorem --/
theorem largest_even_five_digit_number_with_square_and_cube : 
  ∀ n : ℕ, 
    10000 ≤ n ∧ n < 100000 ∧  -- 5-digit number
    Even n ∧  -- even number
    is_perfect_square (first_three_digits n) ∧  -- first three digits form a perfect square
    is_perfect_cube (last_three_digits n)  -- last three digits form a perfect cube
    → n ≤ 62512 :=
by sorry

end NUMINAMATH_CALUDE_largest_even_five_digit_number_with_square_and_cube_l233_23367


namespace NUMINAMATH_CALUDE_crazy_silly_school_unwatched_movies_l233_23328

/-- Given a total number of movies and the number of watched movies,
    calculate the number of unwatched movies -/
def unwatched_movies (total : ℕ) (watched : ℕ) : ℕ :=
  total - watched

/-- Theorem: In the 'crazy silly school' series, with 8 total movies
    and 4 watched movies, there are 4 unwatched movies -/
theorem crazy_silly_school_unwatched_movies :
  unwatched_movies 8 4 = 4 := by
  sorry

end NUMINAMATH_CALUDE_crazy_silly_school_unwatched_movies_l233_23328


namespace NUMINAMATH_CALUDE_absolute_value_of_complex_fraction_l233_23324

theorem absolute_value_of_complex_fraction : 
  Complex.abs (2 / (1 + Complex.I)) = Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_absolute_value_of_complex_fraction_l233_23324


namespace NUMINAMATH_CALUDE_earrings_to_necklace_ratio_l233_23395

theorem earrings_to_necklace_ratio 
  (total_cost : ℝ) 
  (num_necklaces : ℕ) 
  (single_necklace_cost : ℝ) 
  (h1 : total_cost = 240000)
  (h2 : num_necklaces = 3)
  (h3 : single_necklace_cost = 40000) :
  (total_cost - num_necklaces * single_necklace_cost) / single_necklace_cost = 3 := by
  sorry

end NUMINAMATH_CALUDE_earrings_to_necklace_ratio_l233_23395


namespace NUMINAMATH_CALUDE_max_area_rectangle_l233_23399

/-- Represents a rectangle with width and length -/
structure Rectangle where
  width : ℝ
  length : ℝ

/-- Calculates the perimeter of a rectangle -/
def perimeter (r : Rectangle) : ℝ := 2 * (r.width + r.length)

/-- Calculates the area of a rectangle -/
def area (r : Rectangle) : ℝ := r.width * r.length

/-- Theorem: The maximum area of a rectangle with perimeter 60 and length 5 more than width -/
theorem max_area_rectangle :
  ∃ (r : Rectangle),
    perimeter r = 60 ∧
    r.length = r.width + 5 ∧
    area r = 218.75 ∧
    ∀ (r' : Rectangle),
      perimeter r' = 60 →
      r'.length = r'.width + 5 →
      area r' ≤ area r := by
  sorry

end NUMINAMATH_CALUDE_max_area_rectangle_l233_23399


namespace NUMINAMATH_CALUDE_polynomial_division_proof_l233_23356

theorem polynomial_division_proof (z : ℝ) : 
  ((4/3 : ℝ) * z^4 - (17/9 : ℝ) * z^3 + (56/27 : ℝ) * z^2 - (167/81 : ℝ) * z + 500/243) * (3 * z + 1) = 
  4 * z^5 - 5 * z^4 + 7 * z^3 - 15 * z^2 + 9 * z - 3 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_division_proof_l233_23356


namespace NUMINAMATH_CALUDE_arithmetic_sequence_tenth_term_l233_23313

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_tenth_term
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_sum : a 8 + a 9 = 32)
  (h_seventh : a 7 = 1) :
  a 10 = 31 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_tenth_term_l233_23313


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l233_23397

theorem sufficient_but_not_necessary (x : ℝ) : 
  (∀ x, x > 0 → x^2 + 4*x + 3 > 0) ∧ 
  (∃ x, x^2 + 4*x + 3 > 0 ∧ ¬(x > 0)) := by
sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l233_23397


namespace NUMINAMATH_CALUDE_part1_solution_part2_solution_l233_23337

-- Define A_n (falling factorial)
def A (n : ℕ) : ℕ := n * (n - 1) * (n - 2) * (n - 3)

-- Define C_n (binomial coefficient)
def C (n : ℕ) : ℕ := n * (n - 1) * (n - 2) * (n - 3) * (n - 4) * (n - 5) / 720

-- Part 1: Prove that the only positive integer solution to A_{2n+1}^4 = 140A_n^3 is n = 3
theorem part1_solution : {n : ℕ | n > 0 ∧ A (2*n + 1)^4 = 140 * A n^3} = {3} := by sorry

-- Part 2: Prove that the positive integer solutions to A_N^4 ≥ 24C_n^6 where n ≥ 6 are n = 6, 7, 8, 9, 10
theorem part2_solution : {n : ℕ | n ≥ 6 ∧ A n^4 ≥ 24 * C n^6} = {6, 7, 8, 9, 10} := by sorry

end NUMINAMATH_CALUDE_part1_solution_part2_solution_l233_23337


namespace NUMINAMATH_CALUDE_length_of_AB_is_two_l233_23304

-- Define the points A and B
def A (a : ℝ) : ℝ × ℝ := (3, a + 3)
def B (a : ℝ) : ℝ × ℝ := (a, 4)

-- Define the condition that AB is parallel to the x-axis
def parallel_to_x_axis (a : ℝ) : Prop :=
  (A a).2 = (B a).2

-- Define the length of segment AB
def length_AB (a : ℝ) : ℝ :=
  |((A a).1 - (B a).1)|

-- Theorem statement
theorem length_of_AB_is_two (a : ℝ) :
  parallel_to_x_axis a → length_AB a = 2 := by
  sorry

end NUMINAMATH_CALUDE_length_of_AB_is_two_l233_23304


namespace NUMINAMATH_CALUDE_cricket_team_age_difference_l233_23370

theorem cricket_team_age_difference (team_size : ℕ) (captain_age : ℕ) (team_avg_age : ℕ) (remaining_avg_age : ℕ) : 
  team_size = 11 → 
  captain_age = 24 → 
  team_avg_age = 23 → 
  remaining_avg_age = team_avg_age - 1 → 
  ∃ (wicket_keeper_age : ℕ), wicket_keeper_age = captain_age + 7 :=
by sorry

end NUMINAMATH_CALUDE_cricket_team_age_difference_l233_23370


namespace NUMINAMATH_CALUDE_obstacle_course_total_time_l233_23398

def first_part_minutes : ℕ := 7
def first_part_seconds : ℕ := 23
def second_part_seconds : ℕ := 73
def third_part_minutes : ℕ := 5
def third_part_seconds : ℕ := 58

def seconds_per_minute : ℕ := 60

theorem obstacle_course_total_time :
  (first_part_minutes * seconds_per_minute + first_part_seconds) +
  second_part_seconds +
  (third_part_minutes * seconds_per_minute + third_part_seconds) = 874 := by
  sorry

end NUMINAMATH_CALUDE_obstacle_course_total_time_l233_23398


namespace NUMINAMATH_CALUDE_p_q_contradictory_l233_23349

-- Define proposition p
def p : Prop := ∀ a : ℝ, a > 0 → a^2 ≠ 0

-- Define proposition q
def q : Prop := ∀ a : ℝ, a ≤ 0 → a^2 = 0

-- Theorem stating that p and q are contradictory
theorem p_q_contradictory : p ↔ ¬q := by
  sorry


end NUMINAMATH_CALUDE_p_q_contradictory_l233_23349


namespace NUMINAMATH_CALUDE_kamals_math_marks_l233_23318

def english_marks : ℕ := 66
def physics_marks : ℕ := 77
def chemistry_marks : ℕ := 62
def biology_marks : ℕ := 75
def average_marks : ℚ := 69
def total_subjects : ℕ := 5

theorem kamals_math_marks :
  let total_marks := average_marks * total_subjects
  let known_marks_sum := english_marks + physics_marks + chemistry_marks + biology_marks
  let math_marks := total_marks - known_marks_sum
  math_marks = 65 := by sorry

end NUMINAMATH_CALUDE_kamals_math_marks_l233_23318
