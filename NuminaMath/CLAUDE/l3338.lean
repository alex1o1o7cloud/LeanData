import Mathlib

namespace NUMINAMATH_CALUDE_min_beta_value_l3338_333870

theorem min_beta_value (α β : ℕ+) 
  (h1 : (43 : ℚ) / 197 < α / β)
  (h2 : α / β < (17 : ℚ) / 77) :
  ∀ β' : ℕ+, ((43 : ℚ) / 197 < α / β' ∧ α / β' < (17 : ℚ) / 77) → β' ≥ 32 := by
  sorry

end NUMINAMATH_CALUDE_min_beta_value_l3338_333870


namespace NUMINAMATH_CALUDE_square_root_sum_equality_l3338_333806

theorem square_root_sum_equality (n : ℕ) :
  (∃ (x : ℕ), (x : ℝ) * (2018 : ℝ)^2 = (2018 : ℝ)^20) ∧
  (Real.sqrt ((x : ℝ) * (2018 : ℝ)^2) = (2018 : ℝ)^10) →
  x = 2018^18 :=
by sorry

end NUMINAMATH_CALUDE_square_root_sum_equality_l3338_333806


namespace NUMINAMATH_CALUDE_jakes_test_average_l3338_333807

theorem jakes_test_average : 
  let first_test : ℕ := 80
  let second_test : ℕ := first_test + 10
  let third_test : ℕ := 65
  let fourth_test : ℕ := third_test
  let total_marks : ℕ := first_test + second_test + third_test + fourth_test
  let num_tests : ℕ := 4
  (total_marks : ℚ) / num_tests = 75 := by
  sorry

end NUMINAMATH_CALUDE_jakes_test_average_l3338_333807


namespace NUMINAMATH_CALUDE_merchant_profit_l3338_333837

theorem merchant_profit (cost_price : ℝ) (markup_percentage : ℝ) (discount_percentage : ℝ) : 
  markup_percentage = 50 →
  discount_percentage = 20 →
  let marked_price := cost_price * (1 + markup_percentage / 100)
  let selling_price := marked_price * (1 - discount_percentage / 100)
  let profit := selling_price - cost_price
  let profit_percentage := (profit / cost_price) * 100
  profit_percentage = 20 :=
by sorry

end NUMINAMATH_CALUDE_merchant_profit_l3338_333837


namespace NUMINAMATH_CALUDE_rectangle_area_l3338_333800

/-- The area of a rectangle with width 5.4 meters and height 2.5 meters is 13.5 square meters. -/
theorem rectangle_area : 
  let width : Real := 5.4
  let height : Real := 2.5
  width * height = 13.5 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l3338_333800


namespace NUMINAMATH_CALUDE_p_minus_q_value_l3338_333818

theorem p_minus_q_value (p q : ℚ) (hp : 3 / p = 4) (hq : 3 / q = 18) : p - q = 7 / 12 := by
  sorry

end NUMINAMATH_CALUDE_p_minus_q_value_l3338_333818


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3338_333856

theorem complex_equation_solution (x : ℝ) : 
  (↑x + 2 * Complex.I) * (↑x - Complex.I) = (6 : ℂ) + 2 * Complex.I → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3338_333856


namespace NUMINAMATH_CALUDE_shooter_hit_rate_l3338_333847

theorem shooter_hit_rate (p : ℝ) (h1 : 0 ≤ p ∧ p ≤ 1) : 
  (1 - (1 - p)^4 = 80/81) → p = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_shooter_hit_rate_l3338_333847


namespace NUMINAMATH_CALUDE_closest_cube_root_to_50_l3338_333819

theorem closest_cube_root_to_50 :
  ∀ n : ℤ, |((2:ℝ)^n)^(1/3) - 50| ≥ |((2:ℝ)^17)^(1/3) - 50| :=
by sorry

end NUMINAMATH_CALUDE_closest_cube_root_to_50_l3338_333819


namespace NUMINAMATH_CALUDE_remainder_nine_power_2023_mod_50_l3338_333801

theorem remainder_nine_power_2023_mod_50 : 9^2023 % 50 = 41 := by
  sorry

end NUMINAMATH_CALUDE_remainder_nine_power_2023_mod_50_l3338_333801


namespace NUMINAMATH_CALUDE_quadratic_always_nonnegative_implies_a_range_l3338_333890

theorem quadratic_always_nonnegative_implies_a_range (a : ℝ) :
  (∀ x : ℝ, x^2 + a*x + 1 ≥ 0) → a ∈ Set.Icc (-2) 2 :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_always_nonnegative_implies_a_range_l3338_333890


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l3338_333813

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, |x| + x^2 ≥ 0) ↔ (∃ x₀ : ℝ, |x₀| + x₀^2 < 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l3338_333813


namespace NUMINAMATH_CALUDE_eggs_remaining_l3338_333879

def dozen : ℕ := 12

def initial_eggs (num_dozens : ℕ) : ℕ := num_dozens * dozen

def remaining_after_half (total : ℕ) : ℕ := total / 2

def final_eggs (after_half : ℕ) (broken : ℕ) : ℕ := after_half - broken

theorem eggs_remaining (num_dozens : ℕ) (broken : ℕ) 
  (h1 : num_dozens = 6) 
  (h2 : broken = 15) : 
  final_eggs (remaining_after_half (initial_eggs num_dozens)) broken = 21 := by
  sorry

#check eggs_remaining

end NUMINAMATH_CALUDE_eggs_remaining_l3338_333879


namespace NUMINAMATH_CALUDE_boat_speed_specific_boat_speed_l3338_333851

/-- The speed of a boat in still water given its travel times with and against a current. -/
theorem boat_speed (distance : ℝ) (time_against : ℝ) (time_with : ℝ) :
  distance > 0 ∧ time_against > 0 ∧ time_with > 0 →
  ∃ (boat_speed current_speed : ℝ),
    (boat_speed - current_speed) * time_against = distance ∧
    (boat_speed + current_speed) * time_with = distance ∧
    boat_speed = 15.6 := by
  sorry

/-- The specific case mentioned in the problem -/
theorem specific_boat_speed :
  ∃ (boat_speed current_speed : ℝ),
    (boat_speed - current_speed) * 8 = 96 ∧
    (boat_speed + current_speed) * 5 = 96 ∧
    boat_speed = 15.6 := by
  sorry

end NUMINAMATH_CALUDE_boat_speed_specific_boat_speed_l3338_333851


namespace NUMINAMATH_CALUDE_card_collection_average_l3338_333855

def card_count (k : ℕ) : ℕ := 2 * k - 1

def total_cards (n : ℕ) : ℕ := n^2

def sum_of_values (n : ℕ) : ℕ := (n * (n + 1) / 2)^2 - (n * (n + 1) * (2 * n + 1) / 6)

def average_value (n : ℕ) : ℚ := (sum_of_values n : ℚ) / (total_cards n : ℚ)

theorem card_collection_average (n : ℕ) :
  n > 0 ∧ average_value n = 100 → n = 10 :=
by sorry

end NUMINAMATH_CALUDE_card_collection_average_l3338_333855


namespace NUMINAMATH_CALUDE_contractor_engagement_days_l3338_333820

/-- Represents the daily wage in rupees --/
def daily_wage : ℚ := 25

/-- Represents the daily fine in rupees --/
def daily_fine : ℚ := 7.5

/-- Represents the total amount received in rupees --/
def total_amount : ℚ := 685

/-- Represents the number of days absent --/
def days_absent : ℕ := 2

/-- Proves that the contractor was engaged for 28 days --/
theorem contractor_engagement_days : 
  ∃ (days_worked : ℕ), 
    (daily_wage * days_worked - daily_fine * days_absent = total_amount) ∧ 
    (days_worked + days_absent = 28) := by
  sorry

end NUMINAMATH_CALUDE_contractor_engagement_days_l3338_333820


namespace NUMINAMATH_CALUDE_max_product_853_l3338_333810

def Digits : Finset Nat := {3, 5, 6, 8, 9}

def IsValidPair (a b c d e : Nat) : Prop :=
  a ∈ Digits ∧ b ∈ Digits ∧ c ∈ Digits ∧ d ∈ Digits ∧ e ∈ Digits ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧
  c ≠ d ∧ c ≠ e ∧
  d ≠ e

def ThreeDigitNum (a b c : Nat) : Nat := 100 * a + 10 * b + c
def TwoDigitNum (d e : Nat) : Nat := 10 * d + e

theorem max_product_853 :
  ∀ a b c d e,
    IsValidPair a b c d e →
    ThreeDigitNum a b c * TwoDigitNum d e ≤ ThreeDigitNum 8 5 3 * TwoDigitNum 9 6 :=
by sorry

end NUMINAMATH_CALUDE_max_product_853_l3338_333810


namespace NUMINAMATH_CALUDE_jan_extra_miles_l3338_333864

/-- Represents the driving data for a person -/
structure DrivingData where
  time : ℝ
  speed : ℝ
  distance : ℝ

/-- The problem statement -/
theorem jan_extra_miles (ian : DrivingData) (han : DrivingData) (jan : DrivingData) : 
  han.time = ian.time + 2 →
  han.speed = ian.speed + 5 →
  jan.time = ian.time + 3 →
  jan.speed = ian.speed + 15 →
  han.distance = ian.distance + 110 →
  jan.distance = ian.distance + 195 := by
  sorry


end NUMINAMATH_CALUDE_jan_extra_miles_l3338_333864


namespace NUMINAMATH_CALUDE_prep_school_cost_l3338_333842

theorem prep_school_cost (cost_per_semester : ℕ) (semesters_per_year : ℕ) (years : ℕ) : 
  cost_per_semester = 20000 → semesters_per_year = 2 → years = 13 →
  cost_per_semester * semesters_per_year * years = 520000 := by
  sorry

end NUMINAMATH_CALUDE_prep_school_cost_l3338_333842


namespace NUMINAMATH_CALUDE_expected_rank_is_103_l3338_333880

/-- Represents a tennis tournament with the given conditions -/
structure TennisTournament where
  num_players : ℕ
  num_rounds : ℕ
  win_prob : ℚ

/-- Calculates the expected rank of the winner in a tennis tournament -/
def expected_rank (t : TennisTournament) : ℚ :=
  sorry

/-- The specific tournament described in the problem -/
def specific_tournament : TennisTournament :=
  { num_players := 256
  , num_rounds := 8
  , win_prob := 3/5 }

/-- Theorem stating that the expected rank of the winner in the specific tournament is 103 -/
theorem expected_rank_is_103 : expected_rank specific_tournament = 103 :=
  sorry

end NUMINAMATH_CALUDE_expected_rank_is_103_l3338_333880


namespace NUMINAMATH_CALUDE_sphere_volume_diameter_relation_l3338_333852

theorem sphere_volume_diameter_relation :
  ∀ (V₁ V₂ d₁ d₂ : ℝ),
  V₁ > 0 → d₁ > 0 →
  V₁ = (π * d₁^3) / 6 →
  V₂ = 2 * V₁ →
  V₂ = (π * d₂^3) / 6 →
  d₂ / d₁ = (2 : ℝ)^(1/3) :=
by sorry

end NUMINAMATH_CALUDE_sphere_volume_diameter_relation_l3338_333852


namespace NUMINAMATH_CALUDE_ceiling_squared_negative_fraction_l3338_333897

theorem ceiling_squared_negative_fraction : ⌈((-7/4 : ℚ)^2)⌉ = 4 := by sorry

end NUMINAMATH_CALUDE_ceiling_squared_negative_fraction_l3338_333897


namespace NUMINAMATH_CALUDE_fraction_evaluation_l3338_333849

theorem fraction_evaluation (x : ℝ) (h : x = 3) : (x^6 + 8*x^3 + 16) / (x^3 + 4) = 31 := by
  sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l3338_333849


namespace NUMINAMATH_CALUDE_binomial_probability_two_successes_l3338_333804

/-- A random variable X follows a binomial distribution with parameters n and p -/
structure BinomialDistribution (n : ℕ) (p : ℝ) where
  X : ℝ → ℝ

/-- The probability mass function of a binomial distribution -/
def probability_mass_function (n : ℕ) (p : ℝ) (k : ℕ) : ℝ :=
  (Nat.choose n k) * (p ^ k) * ((1 - p) ^ (n - k))

/-- Theorem: For a binomial distribution B(4, 1/2), P(X=2) = 3/8 -/
theorem binomial_probability_two_successes :
  ∀ (X : BinomialDistribution 4 (1/2)),
  probability_mass_function 4 (1/2) 2 = 3/8 := by
  sorry

end NUMINAMATH_CALUDE_binomial_probability_two_successes_l3338_333804


namespace NUMINAMATH_CALUDE_student_mistake_difference_l3338_333824

theorem student_mistake_difference (number : ℕ) (h : number = 192) : 
  (5 / 6 : ℚ) * number - (5 / 16 : ℚ) * number = 100 := by
  sorry

end NUMINAMATH_CALUDE_student_mistake_difference_l3338_333824


namespace NUMINAMATH_CALUDE_circle_center_sum_l3338_333859

/-- The sum of the x and y coordinates of the center of a circle with equation x^2 + y^2 = 4x - 6y + 9 is -1 -/
theorem circle_center_sum (x y : ℝ) : x^2 + y^2 = 4*x - 6*y + 9 → x + y = -1 := by
  sorry

end NUMINAMATH_CALUDE_circle_center_sum_l3338_333859


namespace NUMINAMATH_CALUDE_smallest_upper_bound_D_l3338_333872

def D (n : ℕ+) : ℚ := 5 - (2 * n.val + 5 : ℚ) / 2^n.val

theorem smallest_upper_bound_D :
  ∃ t : ℕ, (∀ n : ℕ+, D n < t) ∧ (∀ s : ℕ, s < t → ∃ m : ℕ+, D m ≥ s) :=
  sorry

end NUMINAMATH_CALUDE_smallest_upper_bound_D_l3338_333872


namespace NUMINAMATH_CALUDE_company_salary_change_l3338_333843

theorem company_salary_change 
  (E : ℕ) -- Original number of employees
  (S : ℝ) -- Original average salary
  (new_E : ℕ) -- New number of employees
  (new_S : ℝ) -- New average salary
  (h1 : new_E = (E * 4) / 5) -- 20% decrease in employees
  (h2 : new_S = S * 1.15) -- 15% increase in average salary
  : (new_E : ℝ) * new_S = 0.92 * ((E : ℝ) * S) :=
by sorry

end NUMINAMATH_CALUDE_company_salary_change_l3338_333843


namespace NUMINAMATH_CALUDE_y_intercept_after_translation_intersection_point_l3338_333838

/-- The y-intercept of a line after vertical translation -/
theorem y_intercept_after_translation (m b h : ℝ) :
  let original_line := fun x => m * x + b
  let translated_line := fun x => m * x + (b + h)
  (translated_line 0) = b + h :=
by
  sorry

/-- Proof that the translated line y = 2x - 1 + 3 intersects y-axis at (0, 2) -/
theorem intersection_point :
  let original_line := fun x => 2 * x - 1
  let translated_line := fun x => 2 * x - 1 + 3
  (translated_line 0) = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_y_intercept_after_translation_intersection_point_l3338_333838


namespace NUMINAMATH_CALUDE_steve_skylar_berry_ratio_l3338_333845

/-- Proves that Steve has half as many berries as Skylar given the conditions of the problem -/
theorem steve_skylar_berry_ratio :
  -- Define the number of berries for each person
  ∀ (steve_berries stacy_berries skylar_berries : ℕ),
  -- Stacy has 2 more than triple as many berries as Steve
  stacy_berries = 3 * steve_berries + 2 →
  -- Skylar has 20 berries
  skylar_berries = 20 →
  -- Stacy has 32 berries
  stacy_berries = 32 →
  -- Conclusion: Steve has 1/2 as many berries as Skylar
  (steve_berries : ℚ) / skylar_berries = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_steve_skylar_berry_ratio_l3338_333845


namespace NUMINAMATH_CALUDE_lotto_winning_percentage_l3338_333844

theorem lotto_winning_percentage :
  let total_tickets : ℕ := 200
  let cost_per_ticket : ℚ := 2
  let grand_prize : ℚ := 5000
  let profit : ℚ := 4830
  let five_dollar_win_ratio : ℚ := 4/5
  let ten_dollar_win_ratio : ℚ := 1/5
  let five_dollar_prize : ℚ := 5
  let ten_dollar_prize : ℚ := 10
  ∃ (winning_tickets : ℕ),
    (winning_tickets : ℚ) / total_tickets * 100 = 19 ∧
    profit = five_dollar_win_ratio * winning_tickets * five_dollar_prize +
             ten_dollar_win_ratio * winning_tickets * ten_dollar_prize +
             grand_prize -
             (total_tickets * cost_per_ticket) :=
by sorry

end NUMINAMATH_CALUDE_lotto_winning_percentage_l3338_333844


namespace NUMINAMATH_CALUDE_greatest_four_digit_number_l3338_333887

theorem greatest_four_digit_number (n : ℕ) : n ≤ 9999 ∧ n ≥ 1000 ∧ 
  ∃ k₁ k₂ : ℕ, n = 11 * k₁ + 2 ∧ n = 7 * k₂ + 4 → n ≤ 9973 :=
by sorry

end NUMINAMATH_CALUDE_greatest_four_digit_number_l3338_333887


namespace NUMINAMATH_CALUDE_tanα_eq_2_implies_reciprocal_sin2α_eq_5_4_l3338_333821

theorem tanα_eq_2_implies_reciprocal_sin2α_eq_5_4 (α : ℝ) (h : Real.tan α = 2) :
  1 / Real.sin (2 * α) = 5 / 4 := by
sorry

end NUMINAMATH_CALUDE_tanα_eq_2_implies_reciprocal_sin2α_eq_5_4_l3338_333821


namespace NUMINAMATH_CALUDE_committee_safe_configuration_l3338_333858

/-- Represents a lock-key system for a committee safe --/
structure CommitteeSafe where
  numMembers : Nat
  numLocks : Nat
  keysPerMember : Nat

/-- Checks if a given number of members can open the safe --/
def canOpen (safe : CommitteeSafe) (presentMembers : Nat) : Prop :=
  presentMembers ≥ 3 ∧ presentMembers ≤ safe.numMembers

/-- Checks if the safe system is secure --/
def isSecure (safe : CommitteeSafe) : Prop :=
  ∀ (presentMembers : Nat), presentMembers ≤ safe.numMembers →
    (canOpen safe presentMembers ↔ 
      presentMembers * safe.keysPerMember ≥ safe.numLocks)

/-- The theorem stating the correct configuration for a 5-member committee --/
theorem committee_safe_configuration :
  ∃ (safe : CommitteeSafe),
    safe.numMembers = 5 ∧
    safe.numLocks = 10 ∧
    safe.keysPerMember = 6 ∧
    isSecure safe :=
sorry

end NUMINAMATH_CALUDE_committee_safe_configuration_l3338_333858


namespace NUMINAMATH_CALUDE_complex_arithmetic_result_l3338_333884

def B : ℂ := 5 - 2 * Complex.I
def N : ℂ := -5 + 2 * Complex.I
def T : ℂ := 2 * Complex.I
def Q : ℂ := 3

theorem complex_arithmetic_result : B - N + T - Q = 7 - 2 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_arithmetic_result_l3338_333884


namespace NUMINAMATH_CALUDE_pizza_bill_friends_l3338_333830

theorem pizza_bill_friends (total_price : ℕ) (price_per_person : ℕ) (bob_included : Bool) : 
  total_price = 40 → price_per_person = 8 → bob_included = true → 
  (total_price / price_per_person) - 1 = 4 := by
  sorry

end NUMINAMATH_CALUDE_pizza_bill_friends_l3338_333830


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_l3338_333826

theorem necessary_but_not_sufficient_condition :
  (∀ x : ℝ, x > 0 → x > -2) ∧ 
  (∃ x : ℝ, x > -2 ∧ x ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_l3338_333826


namespace NUMINAMATH_CALUDE_intersection_lines_sum_l3338_333865

theorem intersection_lines_sum (c d : ℝ) : 
  (3 = (1/3) * 3 + c) → 
  (3 = (1/3) * 3 + d) → 
  c + d = 4 := by
sorry

end NUMINAMATH_CALUDE_intersection_lines_sum_l3338_333865


namespace NUMINAMATH_CALUDE_basketball_game_free_throws_l3338_333860

theorem basketball_game_free_throws :
  ∀ (three_pointers two_pointers free_throws : ℕ),
    three_pointers + two_pointers + free_throws = 32 →
    two_pointers = 4 * three_pointers + 3 →
    3 * three_pointers + 2 * two_pointers + free_throws = 65 →
    free_throws = 4 :=
by sorry

end NUMINAMATH_CALUDE_basketball_game_free_throws_l3338_333860


namespace NUMINAMATH_CALUDE_donuts_per_box_l3338_333875

theorem donuts_per_box (total_boxes : Nat) (boxes_given : Nat) (extra_donuts_given : Nat) (donuts_left : Nat) :
  total_boxes = 4 →
  boxes_given = 1 →
  extra_donuts_given = 6 →
  donuts_left = 30 →
  ∃ (donuts_per_box : Nat), 
    donuts_per_box * total_boxes = 
      donuts_per_box * boxes_given + extra_donuts_given + donuts_left ∧
    donuts_per_box = 12 := by
  sorry

end NUMINAMATH_CALUDE_donuts_per_box_l3338_333875


namespace NUMINAMATH_CALUDE_max_value_of_a_l3338_333841

theorem max_value_of_a : ∃ (a_max : ℝ), a_max = 16175 ∧
  ∀ (a : ℝ), (∀ (x : ℝ), 0 ≤ x ∧ x ≤ 1 →
    -2022 ≤ (a + 1) * x^2 - (a + 1) * x + 2022 ∧
    (a + 1) * x^2 - (a + 1) * x + 2022 ≤ 2022) →
  a ≤ a_max := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_a_l3338_333841


namespace NUMINAMATH_CALUDE_art_exhibition_problem_l3338_333867

/-- Art exhibition visitor and ticket problem -/
theorem art_exhibition_problem 
  (total_saturday : ℕ)
  (sunday_morning_increase : ℚ)
  (sunday_afternoon_increase : ℚ)
  (total_sunday_increase : ℕ)
  (sunday_morning_revenue : ℕ)
  (sunday_afternoon_revenue : ℕ)
  (sunday_morning_adults : ℕ)
  (sunday_afternoon_adults : ℕ)
  (h1 : total_saturday = 300)
  (h2 : sunday_morning_increase = 40 / 100)
  (h3 : sunday_afternoon_increase = 30 / 100)
  (h4 : total_sunday_increase = 100)
  (h5 : sunday_morning_revenue = 4200)
  (h6 : sunday_afternoon_revenue = 7200)
  (h7 : sunday_morning_adults = 70)
  (h8 : sunday_afternoon_adults = 100) :
  ∃ (sunday_morning sunday_afternoon adult_price student_price : ℕ),
    sunday_morning = 140 ∧
    sunday_afternoon = 260 ∧
    adult_price = 40 ∧
    student_price = 20 := by
  sorry


end NUMINAMATH_CALUDE_art_exhibition_problem_l3338_333867


namespace NUMINAMATH_CALUDE_fraction_simplification_l3338_333861

theorem fraction_simplification (b c d x y : ℝ) :
  (c * x * (b^2 * x^3 + 3 * b^2 * y^3 + c^3 * y^3) + d * y * (b^2 * x^3 + 3 * c^3 * x^3 + c^3 * y^3)) / (c * x + d * y) =
  b^2 * x^3 + 3 * c^2 * x * y^3 + c^3 * y^3 :=
by sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3338_333861


namespace NUMINAMATH_CALUDE_dogs_neither_long_furred_nor_brown_l3338_333833

/-- Prove that the number of dogs that are neither long-furred nor brown is 8 -/
theorem dogs_neither_long_furred_nor_brown
  (total_dogs : ℕ)
  (long_furred_dogs : ℕ)
  (brown_dogs : ℕ)
  (long_furred_brown_dogs : ℕ)
  (h1 : total_dogs = 45)
  (h2 : long_furred_dogs = 26)
  (h3 : brown_dogs = 22)
  (h4 : long_furred_brown_dogs = 11) :
  total_dogs - (long_furred_dogs + brown_dogs - long_furred_brown_dogs) = 8 := by
  sorry

#check dogs_neither_long_furred_nor_brown

end NUMINAMATH_CALUDE_dogs_neither_long_furred_nor_brown_l3338_333833


namespace NUMINAMATH_CALUDE_inequality_proof_l3338_333871

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x * y + 2 * y * z + 2 * z * x) / (x^2 + y^2 + z^2) ≤ (Real.sqrt 33 + 1) / 4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3338_333871


namespace NUMINAMATH_CALUDE_decimal_multiplication_and_composition_l3338_333832

theorem decimal_multiplication_and_composition : 
  (35 * 0.01 = 0.35) ∧ (0.875 = 875 * 0.001) := by
  sorry

end NUMINAMATH_CALUDE_decimal_multiplication_and_composition_l3338_333832


namespace NUMINAMATH_CALUDE_sand_box_fill_time_l3338_333814

/-- The time required to fill a rectangular box with sand -/
theorem sand_box_fill_time
  (length width height : ℝ)
  (fill_rate : ℝ)
  (h_length : length = 7)
  (h_width : width = 6)
  (h_height : height = 2)
  (h_fill_rate : fill_rate = 4)
  : (length * width * height) / fill_rate = 21 := by
  sorry

end NUMINAMATH_CALUDE_sand_box_fill_time_l3338_333814


namespace NUMINAMATH_CALUDE_square_equation_proof_l3338_333822

theorem square_equation_proof (a b c : ℤ) 
  (h : (a + 3)^2 + (b + 4)^2 - (c + 5)^2 = a^2 + b^2 - c^2) :
  ∃ (k : ℚ), (a + 3)^2 + (b + 4)^2 - (c + 5)^2 = k^2 ∧ 
              a^2 + b^2 - c^2 = k^2 ∧
              k = (4*a - 3*b : ℚ) / 5 := by
  sorry

end NUMINAMATH_CALUDE_square_equation_proof_l3338_333822


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3338_333899

theorem sufficient_not_necessary_condition (a b : ℝ) :
  (∀ a b : ℝ, a > b ∧ b > 0 → a^2 > b^2) ∧
  (∃ a b : ℝ, a^2 > b^2 ∧ ¬(a > b ∧ b > 0)) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3338_333899


namespace NUMINAMATH_CALUDE_percentage_calculation_l3338_333885

theorem percentage_calculation : 
  (789524.37 : ℝ) * (7.5 / 100) = 59214.32825 := by sorry

end NUMINAMATH_CALUDE_percentage_calculation_l3338_333885


namespace NUMINAMATH_CALUDE_jose_distance_l3338_333869

/-- Given a speed of 2 kilometers per hour and a time of 2 hours, 
    the distance traveled is equal to 4 kilometers. -/
theorem jose_distance (speed : ℝ) (time : ℝ) (distance : ℝ) : 
  speed = 2 → time = 2 → distance = speed * time → distance = 4 := by
  sorry


end NUMINAMATH_CALUDE_jose_distance_l3338_333869


namespace NUMINAMATH_CALUDE_square_difference_l3338_333886

theorem square_difference (x : ℤ) (h : x^2 = 9801) : (x + 3) * (x - 3) = 9792 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_l3338_333886


namespace NUMINAMATH_CALUDE_g36_values_product_l3338_333850

def is_valid_g (g : ℕ → ℕ) : Prop :=
  ∀ a b : ℕ, 3 * g (a^2 + b^2) = (g a)^2 + (g b)^2 + g a * g b

def possible_g36_values (g : ℕ → ℕ) : Set ℕ :=
  {x : ℕ | ∃ h : is_valid_g g, g 36 = x}

theorem g36_values_product (g : ℕ → ℕ) (h : is_valid_g g) :
  (Finset.card (Finset.image g {36})) * (Finset.sum (Finset.image g {36}) id) = 2 := by
  sorry

end NUMINAMATH_CALUDE_g36_values_product_l3338_333850


namespace NUMINAMATH_CALUDE_lost_bottle_caps_l3338_333862

/-- Represents the number of bottle caps Danny has now -/
def current_bottle_caps : ℕ := 25

/-- Represents the number of bottle caps Danny had at first -/
def initial_bottle_caps : ℕ := 91

/-- Theorem stating that the number of lost bottle caps is the difference between
    the initial number and the current number of bottle caps -/
theorem lost_bottle_caps : 
  initial_bottle_caps - current_bottle_caps = 66 := by
  sorry

end NUMINAMATH_CALUDE_lost_bottle_caps_l3338_333862


namespace NUMINAMATH_CALUDE_brandy_energy_drinks_l3338_333802

/-- The number of energy drinks Brandy drank -/
def num_drinks : ℕ := 4

/-- The maximum safe amount of caffeine per day in mg -/
def max_caffeine : ℕ := 500

/-- The amount of caffeine in each energy drink in mg -/
def caffeine_per_drink : ℕ := 120

/-- The amount of additional caffeine Brandy can safely consume after drinking the energy drinks in mg -/
def remaining_caffeine : ℕ := 20

theorem brandy_energy_drinks :
  num_drinks * caffeine_per_drink + remaining_caffeine = max_caffeine :=
sorry

end NUMINAMATH_CALUDE_brandy_energy_drinks_l3338_333802


namespace NUMINAMATH_CALUDE_removal_time_l3338_333893

/-- Represents a position in the 3D grid -/
structure Position :=
  (x : Nat) (y : Nat) (z : Nat)

/-- The dimensions of the rectangular prism -/
def prism_dimensions : Position :=
  ⟨3, 4, 5⟩

/-- Checks if a position is within the prism bounds -/
def is_valid_position (p : Position) : Prop :=
  1 ≤ p.x ∧ p.x ≤ prism_dimensions.x ∧
  1 ≤ p.y ∧ p.y ≤ prism_dimensions.y ∧
  1 ≤ p.z ∧ p.z ≤ prism_dimensions.z

/-- Calculates the layer sum for a position -/
def layer_sum (p : Position) : Nat :=
  p.x + p.y + p.z

/-- The maximum possible layer sum in the prism -/
def max_layer_sum : Nat :=
  prism_dimensions.x + prism_dimensions.y + prism_dimensions.z

/-- Theorem: It takes 10 minutes to remove all cubes -/
theorem removal_time : 
  ∀ (start : Position), 
  is_valid_position start → 
  layer_sum start = 3 → 
  max_layer_sum - layer_sum start + 1 = 10 := by
  sorry

end NUMINAMATH_CALUDE_removal_time_l3338_333893


namespace NUMINAMATH_CALUDE_triangle_angle_range_l3338_333831

theorem triangle_angle_range (A B C : Real) (h : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π) 
  (h_tan : Real.tan B ^ 2 = Real.tan A * Real.tan C) : 
  π / 3 ≤ B ∧ B < π / 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_range_l3338_333831


namespace NUMINAMATH_CALUDE_shorts_cost_l3338_333878

def football_cost : ℝ := 3.75
def shoes_cost : ℝ := 11.85
def zachary_money : ℝ := 10
def additional_money_needed : ℝ := 8

theorem shorts_cost : 
  ∃ (shorts_price : ℝ), 
    football_cost + shoes_cost + shorts_price = zachary_money + additional_money_needed ∧ 
    shorts_price = 2.40 := by
sorry

end NUMINAMATH_CALUDE_shorts_cost_l3338_333878


namespace NUMINAMATH_CALUDE_lisa_age_l3338_333803

theorem lisa_age :
  ∀ (L N : ℕ),
  L = N + 8 →
  L - 2 = 3 * (N - 2) →
  L = 14 :=
by sorry

end NUMINAMATH_CALUDE_lisa_age_l3338_333803


namespace NUMINAMATH_CALUDE_blue_flower_percentage_l3338_333895

/-- Given a total of 10 flowers, with 4 red and 2 white flowers,
    prove that 40% of the flowers are blue. -/
theorem blue_flower_percentage
  (total : ℕ)
  (red : ℕ)
  (white : ℕ)
  (h_total : total = 10)
  (h_red : red = 4)
  (h_white : white = 2) :
  (total - red - white : ℚ) / total * 100 = 40 := by
  sorry

#check blue_flower_percentage

end NUMINAMATH_CALUDE_blue_flower_percentage_l3338_333895


namespace NUMINAMATH_CALUDE_tangent_lines_through_M_line_intersects_circle_l3338_333894

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 4

-- Define point M
def point_M : ℝ × ℝ := (3, 1)

-- Define the line ax - y + 3 = 0
def line (a x y : ℝ) : Prop := a * x - y + 3 = 0

-- Theorem for part (I)
theorem tangent_lines_through_M :
  ∃ (k : ℝ), 
    (∀ x y : ℝ, (x = 3 ∨ 3 * x - 4 * y - 5 = 0) → 
      (circle_C x y ∧ (x = point_M.1 ∧ y = point_M.2 ∨ 
       (y - point_M.2) = k * (x - point_M.1)))) :=
sorry

-- Theorem for part (II)
theorem line_intersects_circle :
  ∀ a : ℝ, ∃ x y : ℝ, line a x y ∧ circle_C x y :=
sorry

end NUMINAMATH_CALUDE_tangent_lines_through_M_line_intersects_circle_l3338_333894


namespace NUMINAMATH_CALUDE_symmetric_point_coordinates_l3338_333846

/-- A point in a 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Symmetry with respect to the y-axis -/
def symmetric_y (a b : Point) : Prop :=
  b.x = -a.x ∧ b.y = a.y

theorem symmetric_point_coordinates :
  let a : Point := ⟨-1, 8⟩
  let b : Point := ⟨1, 8⟩
  symmetric_y a b → b = ⟨1, 8⟩ := by
  sorry

end NUMINAMATH_CALUDE_symmetric_point_coordinates_l3338_333846


namespace NUMINAMATH_CALUDE_total_dimes_l3338_333816

-- Define the initial number of dimes Melanie had
def initial_dimes : Nat := 19

-- Define the number of dimes given by her dad
def dimes_from_dad : Nat := 39

-- Define the number of dimes given by her mother
def dimes_from_mom : Nat := 25

-- Theorem to prove the total number of dimes
theorem total_dimes : 
  initial_dimes + dimes_from_dad + dimes_from_mom = 83 := by
  sorry

end NUMINAMATH_CALUDE_total_dimes_l3338_333816


namespace NUMINAMATH_CALUDE_fraction_comparison_l3338_333868

theorem fraction_comparison (a b m : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : m > 0) :
  b / a < (b + m) / (a + m) := by
  sorry

end NUMINAMATH_CALUDE_fraction_comparison_l3338_333868


namespace NUMINAMATH_CALUDE_sum_of_42_odd_numbers_l3338_333828

/-- The sum of the first n odd numbers -/
def sumOfOddNumbers (n : ℕ) : ℕ :=
  n * n

theorem sum_of_42_odd_numbers :
  sumOfOddNumbers 42 = 1764 := by
  sorry

#eval sumOfOddNumbers 42  -- This will output 1764

end NUMINAMATH_CALUDE_sum_of_42_odd_numbers_l3338_333828


namespace NUMINAMATH_CALUDE_sphere_radius_ratio_l3338_333823

theorem sphere_radius_ratio (v_large v_small : ℝ) (h1 : v_large = 324 * Real.pi) (h2 : v_small = 0.25 * v_large) :
  (v_small / v_large) ^ (1/3 : ℝ) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_sphere_radius_ratio_l3338_333823


namespace NUMINAMATH_CALUDE_M_intersect_N_l3338_333883

def M : Set ℝ := {x : ℝ | -4 < x ∧ x < 2}

def N : Set ℝ := {x : ℝ | x^2 - x - 6 < 0}

theorem M_intersect_N : M ∩ N = {x : ℝ | -2 < x ∧ x < 2} := by
  sorry

end NUMINAMATH_CALUDE_M_intersect_N_l3338_333883


namespace NUMINAMATH_CALUDE_tangent_line_intersection_l3338_333853

/-- The function f(x) = x^3 - x^2 + ax + 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - x^2 + a*x + 1

/-- The derivative of f(x) -/
def f_derivative (a : ℝ) (x : ℝ) : ℝ := 3*x^2 - 2*x + a

theorem tangent_line_intersection (a : ℝ) :
  ∃ (x₁ x₂ : ℝ), 
    (x₁ = 1 ∧ f a x₁ = a + 1) ∧ 
    (x₂ = -1 ∧ f a x₂ = -a - 1) ∧
    (∀ x : ℝ, x ≠ x₁ ∧ x ≠ x₂ → 
      f a x ≠ (f_derivative a x₁) * x) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_intersection_l3338_333853


namespace NUMINAMATH_CALUDE_positive_real_equalities_l3338_333882

theorem positive_real_equalities (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (a^2 + b^2 + c^2 = a*b + b*c + c*a → a = b ∧ b = c) ∧
  ((a + b + c) * (a^2 + b^2 + c^2 - a*b - b*c - a*c) = 0 → a = b ∧ b = c) ∧
  (a^4 + b^4 + c^4 + d^4 = 4*a*b*c*d → a = b ∧ b = c ∧ c = d) := by
  sorry

end NUMINAMATH_CALUDE_positive_real_equalities_l3338_333882


namespace NUMINAMATH_CALUDE_second_exam_score_l3338_333881

theorem second_exam_score (total_marks : ℕ) (num_exams : ℕ) (first_exam_percent : ℚ) 
  (third_exam_marks : ℕ) (overall_average_percent : ℚ) :
  total_marks = 500 →
  num_exams = 3 →
  first_exam_percent = 45 / 100 →
  third_exam_marks = 100 →
  overall_average_percent = 40 / 100 →
  (first_exam_percent * total_marks + (55 / 100) * total_marks + third_exam_marks) / 
    (num_exams * total_marks) = overall_average_percent :=
by sorry

end NUMINAMATH_CALUDE_second_exam_score_l3338_333881


namespace NUMINAMATH_CALUDE_fred_dimes_problem_l3338_333892

/-- Represents the number of dimes Fred's sister borrowed -/
def dimes_borrowed (initial_dimes remaining_dimes : ℕ) : ℕ :=
  initial_dimes - remaining_dimes

theorem fred_dimes_problem (initial_dimes remaining_dimes : ℕ) 
  (h1 : initial_dimes = 7)
  (h2 : remaining_dimes = 4) :
  dimes_borrowed initial_dimes remaining_dimes = 3 := by
  sorry

end NUMINAMATH_CALUDE_fred_dimes_problem_l3338_333892


namespace NUMINAMATH_CALUDE_tan_double_angle_l3338_333888

theorem tan_double_angle (α : ℝ) (h : Real.sin α - 2 * Real.cos α = 0) : 
  Real.tan (2 * α) = -4/3 := by
  sorry

end NUMINAMATH_CALUDE_tan_double_angle_l3338_333888


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_l3338_333808

theorem sum_of_roots_quadratic (a b : ℝ) : 
  (a^2 - 8*a + 5 = 0) → (b^2 - 8*b + 5 = 0) → (a + b = 8) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_l3338_333808


namespace NUMINAMATH_CALUDE_base9_multiplication_l3338_333889

/-- Converts a base 9 number represented as a list of digits to its decimal equivalent -/
def base9ToDecimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * 9^i) 0

/-- Converts a decimal number to its base 9 representation as a list of digits -/
def decimalToBase9 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) :=
    if m = 0 then acc
    else aux (m / 9) ((m % 9) :: acc)
  aux n []

/-- Multiplies two base 9 numbers -/
def multiplyBase9 (a b : List Nat) : List Nat :=
  decimalToBase9 ((base9ToDecimal a) * (base9ToDecimal b))

theorem base9_multiplication (a b : List Nat) :
  multiplyBase9 [3, 5, 4] [1, 2] = [1, 2, 5, 1] := by
  sorry

#eval multiplyBase9 [3, 5, 4] [1, 2]

end NUMINAMATH_CALUDE_base9_multiplication_l3338_333889


namespace NUMINAMATH_CALUDE_highlighter_box_cost_l3338_333857

theorem highlighter_box_cost (
  boxes : ℕ)
  (pens_per_box : ℕ)
  (rearranged_boxes : ℕ)
  (pens_per_package : ℕ)
  (package_price : ℚ)
  (pens_per_set : ℕ)
  (set_price : ℚ)
  (total_profit : ℚ)
  (h1 : boxes = 12)
  (h2 : pens_per_box = 30)
  (h3 : rearranged_boxes = 5)
  (h4 : pens_per_package = 6)
  (h5 : package_price = 3)
  (h6 : pens_per_set = 3)
  (h7 : set_price = 2)
  (h8 : total_profit = 115) :
  ∃ (cost_per_box : ℚ), abs (cost_per_box - 25/3) < 1/100 := by
  sorry

end NUMINAMATH_CALUDE_highlighter_box_cost_l3338_333857


namespace NUMINAMATH_CALUDE_kelly_apples_l3338_333812

/-- The number of apples Kelly needs to pick -/
def apples_to_pick : ℕ := 49

/-- The total number of apples Kelly will have after picking -/
def total_apples : ℕ := 105

/-- The number of apples Kelly has now -/
def current_apples : ℕ := total_apples - apples_to_pick

theorem kelly_apples : current_apples = 56 := by
  sorry

end NUMINAMATH_CALUDE_kelly_apples_l3338_333812


namespace NUMINAMATH_CALUDE_consecutive_integers_sum_l3338_333840

theorem consecutive_integers_sum (a b c d : ℝ) : 
  (b = a + 1 ∧ c = b + 1 ∧ d = c + 1) →  -- consecutive integers condition
  (a + d = 180) →                        -- sum of first and fourth is 180
  b = 90.5 :=                            -- second integer is 90.5
by sorry

end NUMINAMATH_CALUDE_consecutive_integers_sum_l3338_333840


namespace NUMINAMATH_CALUDE_find_z2_l3338_333835

def complex_i : ℂ := Complex.I

theorem find_z2 (z1 z2 : ℂ) : 
  ((z1 - 2) * (1 + complex_i) = 1 - complex_i) →
  (z2.im = 2) →
  ((z1 * z2).im = 0) →
  z2 = 4 + 2 * complex_i :=
by sorry

end NUMINAMATH_CALUDE_find_z2_l3338_333835


namespace NUMINAMATH_CALUDE_morse_code_symbols_l3338_333877

/-- The number of possible symbols for a given sequence length -/
def symbolCount (n : ℕ) : ℕ := 2^n

/-- The total number of distinct Morse code symbols with lengths 1 to 4 -/
def totalSymbols : ℕ := symbolCount 1 + symbolCount 2 + symbolCount 3 + symbolCount 4

theorem morse_code_symbols : totalSymbols = 30 := by
  sorry

end NUMINAMATH_CALUDE_morse_code_symbols_l3338_333877


namespace NUMINAMATH_CALUDE_point_in_fourth_quadrant_implies_a_squared_plus_one_l3338_333891

theorem point_in_fourth_quadrant_implies_a_squared_plus_one (a : ℤ) : 
  (3 * a - 9 > 0) → (2 * a - 10 < 0) → a^2 + 1 = 17 := by
  sorry

end NUMINAMATH_CALUDE_point_in_fourth_quadrant_implies_a_squared_plus_one_l3338_333891


namespace NUMINAMATH_CALUDE_man_against_stream_speed_l3338_333898

/-- Represents the speed of a man rowing a boat in different conditions -/
structure BoatSpeed where
  stillWaterRate : ℝ
  withStreamSpeed : ℝ

/-- Calculates the speed of the boat against the stream -/
def againstStreamSpeed (bs : BoatSpeed) : ℝ :=
  abs (2 * bs.stillWaterRate - bs.withStreamSpeed)

/-- Theorem: Given the man's rate in still water and speed with the stream,
    prove that his speed against the stream is 12 km/h -/
theorem man_against_stream_speed (bs : BoatSpeed)
    (h1 : bs.stillWaterRate = 7)
    (h2 : bs.withStreamSpeed = 26) :
    againstStreamSpeed bs = 12 := by
  sorry

end NUMINAMATH_CALUDE_man_against_stream_speed_l3338_333898


namespace NUMINAMATH_CALUDE_smallest_fraction_between_l3338_333854

theorem smallest_fraction_between (a₁ b₁ a₂ b₂ : ℕ) 
  (h₁ : a₁ < b₁) (h₂ : a₂ < b₂) 
  (h₃ : Nat.gcd a₁ b₁ = 1) (h₄ : Nat.gcd a₂ b₂ = 1)
  (h₅ : a₂ * b₁ - a₁ * b₂ = 1) :
  ∃ (n k : ℕ), 
    (∀ (n' k' : ℕ), a₁ * n' < b₁ * k' ∧ b₂ * k' < a₂ * n' → n ≤ n') ∧
    a₁ * n < b₁ * k ∧ b₂ * k < a₂ * n ∧
    n = b₁ + b₂ ∧ k = a₁ + a₂ := by
  sorry

end NUMINAMATH_CALUDE_smallest_fraction_between_l3338_333854


namespace NUMINAMATH_CALUDE_original_production_was_125_l3338_333805

/-- Represents the clothing production problem --/
structure ClothingProduction where
  plannedDays : ℕ
  actualDailyProduction : ℕ
  daysAheadOfSchedule : ℕ

/-- Calculates the original planned daily production --/
def originalPlannedProduction (cp : ClothingProduction) : ℚ :=
  (cp.actualDailyProduction * (cp.plannedDays - cp.daysAheadOfSchedule)) / cp.plannedDays

/-- Theorem stating that the original planned production was 125 sets per day --/
theorem original_production_was_125 (cp : ClothingProduction) 
  (h1 : cp.plannedDays = 30)
  (h2 : cp.actualDailyProduction = 150)
  (h3 : cp.daysAheadOfSchedule = 5) :
  originalPlannedProduction cp = 125 := by
  sorry

#eval originalPlannedProduction ⟨30, 150, 5⟩

end NUMINAMATH_CALUDE_original_production_was_125_l3338_333805


namespace NUMINAMATH_CALUDE_union_equals_A_iff_m_in_range_l3338_333839

-- Define sets A and B
def A : Set ℝ := {x : ℝ | -1 ≤ x ∧ x ≤ 2}
def B (m : ℝ) : Set ℝ := {x : ℝ | x^2 - (2*m + 1)*x + 2*m < 0}

-- State the theorem
theorem union_equals_A_iff_m_in_range (m : ℝ) : 
  A ∪ B m = A ↔ -1/2 ≤ m ∧ m ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_union_equals_A_iff_m_in_range_l3338_333839


namespace NUMINAMATH_CALUDE_right_triangle_leg_length_l3338_333848

theorem right_triangle_leg_length 
  (a b c : ℝ) 
  (right_triangle : a^2 + b^2 = c^2) 
  (leg_a : a = 7) 
  (hypotenuse : c = 25) : 
  b = 24 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_leg_length_l3338_333848


namespace NUMINAMATH_CALUDE_train_length_calculation_l3338_333829

/-- Calculates the length of a train given its speed and time to cross a point -/
theorem train_length_calculation (speed_km_hr : ℝ) (time_seconds : ℝ) : 
  speed_km_hr = 144 →
  time_seconds = 0.9999200063994881 →
  ∃ (length_meters : ℝ), abs (length_meters - 39.997) < 0.001 := by
  sorry

end NUMINAMATH_CALUDE_train_length_calculation_l3338_333829


namespace NUMINAMATH_CALUDE_min_value_theorem_l3338_333896

theorem min_value_theorem (x y : ℝ) (h1 : x * y + 1 = 4 * x + y) (h2 : x > 1) :
  (x + 1) * (y + 2) ≥ 15 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3338_333896


namespace NUMINAMATH_CALUDE_square_sum_given_diff_and_product_l3338_333873

theorem square_sum_given_diff_and_product (a b : ℝ) 
  (h1 : a - b = 3) 
  (h2 : a * b = 9) : 
  a^2 + b^2 = 27 := by
sorry

end NUMINAMATH_CALUDE_square_sum_given_diff_and_product_l3338_333873


namespace NUMINAMATH_CALUDE_f_positive_range_f_always_negative_l3338_333876

def f (a : ℝ) (x : ℝ) : ℝ := |x - 2| - |2*x - a|

theorem f_positive_range (x : ℝ) : 
  f 3 x > 0 ↔ 1 < x ∧ x < 5/3 := by sorry

theorem f_always_negative (a : ℝ) : 
  (∀ x < 2, f a x < 0) ↔ a ≥ 4 := by sorry

end NUMINAMATH_CALUDE_f_positive_range_f_always_negative_l3338_333876


namespace NUMINAMATH_CALUDE_line_l_properties_l3338_333825

/-- The line l is defined by the equation (a^2 + a + 1)x - y + 1 = 0, where a is a real number -/
def line_l (a : ℝ) (x y : ℝ) : Prop :=
  (a^2 + a + 1) * x - y + 1 = 0

/-- The perpendicular line is defined by the equation x + y = 0 -/
def perp_line (x y : ℝ) : Prop :=
  x + y = 0

theorem line_l_properties :
  (∀ a : ℝ, line_l a 0 1) ∧ 
  (∀ x y : ℝ, line_l 0 x y → perp_line x y) :=
by sorry

end NUMINAMATH_CALUDE_line_l_properties_l3338_333825


namespace NUMINAMATH_CALUDE_nancy_quarters_l3338_333874

/-- The number of quarters Nancy has -/
def number_of_quarters (total_amount : ℚ) (quarter_value : ℚ) : ℚ :=
  total_amount / quarter_value

theorem nancy_quarters : 
  let total_amount : ℚ := 3
  let quarter_value : ℚ := 1/4
  number_of_quarters total_amount quarter_value = 12 := by
sorry

end NUMINAMATH_CALUDE_nancy_quarters_l3338_333874


namespace NUMINAMATH_CALUDE_major_axis_endpoints_of_ellipse_l3338_333863

/-- The equation of the ellipse -/
def ellipse_equation (x y : ℝ) : Prop := 6 * x^2 + y^2 = 6

/-- The endpoints of the major axis -/
def major_axis_endpoints : Set (ℝ × ℝ) := {(0, -Real.sqrt 6), (0, Real.sqrt 6)}

/-- Theorem: The endpoints of the major axis of the ellipse 6x^2 + y^2 = 6 
    are (0, -√6) and (0, √6) -/
theorem major_axis_endpoints_of_ellipse :
  ∀ (p : ℝ × ℝ), p ∈ major_axis_endpoints ↔ 
    (ellipse_equation p.1 p.2 ∧ 
     ∀ (q : ℝ × ℝ), ellipse_equation q.1 q.2 → p.1^2 + p.2^2 ≥ q.1^2 + q.2^2) :=
by sorry

end NUMINAMATH_CALUDE_major_axis_endpoints_of_ellipse_l3338_333863


namespace NUMINAMATH_CALUDE_chicken_to_beef_ratio_is_two_to_one_l3338_333827

/-- Represents the order of beef and chicken --/
structure FoodOrder where
  beef_pounds : ℕ
  beef_price_per_pound : ℕ
  chicken_price_per_pound : ℕ
  total_cost : ℕ

/-- Calculates the ratio of chicken to beef in the order --/
def chicken_to_beef_ratio (order : FoodOrder) : ℚ :=
  let beef_cost := order.beef_pounds * order.beef_price_per_pound
  let chicken_cost := order.total_cost - beef_cost
  let chicken_pounds := chicken_cost / order.chicken_price_per_pound
  chicken_pounds / order.beef_pounds

/-- Theorem stating that the ratio of chicken to beef is 2:1 for the given order --/
theorem chicken_to_beef_ratio_is_two_to_one (order : FoodOrder) 
  (h1 : order.beef_pounds = 1000)
  (h2 : order.beef_price_per_pound = 8)
  (h3 : order.chicken_price_per_pound = 3)
  (h4 : order.total_cost = 14000) : 
  chicken_to_beef_ratio order = 2 := by
  sorry

#eval chicken_to_beef_ratio { 
  beef_pounds := 1000, 
  beef_price_per_pound := 8, 
  chicken_price_per_pound := 3, 
  total_cost := 14000 
}

end NUMINAMATH_CALUDE_chicken_to_beef_ratio_is_two_to_one_l3338_333827


namespace NUMINAMATH_CALUDE_x_eq_one_sufficient_not_necessary_for_x_squared_eq_one_l3338_333836

theorem x_eq_one_sufficient_not_necessary_for_x_squared_eq_one :
  (∃ x : ℝ, x = 1 → x^2 = 1) ∧ 
  (∃ x : ℝ, x^2 = 1 ∧ x ≠ 1) :=
sorry

end NUMINAMATH_CALUDE_x_eq_one_sufficient_not_necessary_for_x_squared_eq_one_l3338_333836


namespace NUMINAMATH_CALUDE_method_a_cheaper_for_18_hours_l3338_333834

/-- Calculates the cost of internet usage for Method A (Pay-per-use) -/
def costMethodA (hours : ℝ) : ℝ := 3 * hours + 1.2 * hours

/-- Calculates the cost of internet usage for Method B (Monthly subscription) -/
def costMethodB (hours : ℝ) : ℝ := 60 + 1.2 * hours

/-- Theorem stating that Method A is cheaper than Method B for 18 hours of usage -/
theorem method_a_cheaper_for_18_hours :
  costMethodA 18 < costMethodB 18 :=
sorry

end NUMINAMATH_CALUDE_method_a_cheaper_for_18_hours_l3338_333834


namespace NUMINAMATH_CALUDE_min_tiles_for_floor_l3338_333817

/-- Represents the dimensions of a rectangular shape in inches -/
structure Dimensions where
  length : ℕ
  width : ℕ

/-- Converts feet to inches -/
def feetToInches (feet : ℕ) : ℕ := feet * 12

/-- Calculates the area of a rectangular shape given its dimensions -/
def area (d : Dimensions) : ℕ := d.length * d.width

/-- Calculates the number of tiles needed to cover a floor -/
def tilesNeeded (floorDim : Dimensions) (tileDim : Dimensions) : ℕ :=
  (area floorDim) / (area tileDim)

theorem min_tiles_for_floor : 
  let tileDim : Dimensions := ⟨3, 4⟩
  let floorDimFeet : Dimensions := ⟨2, 5⟩
  let floorDimInches : Dimensions := ⟨feetToInches floorDimFeet.length, feetToInches floorDimFeet.width⟩
  tilesNeeded floorDimInches tileDim = 120 := by
  sorry

end NUMINAMATH_CALUDE_min_tiles_for_floor_l3338_333817


namespace NUMINAMATH_CALUDE_line_segment_length_l3338_333811

/-- Represents a line segment in 3D space -/
structure LineSegment3D where
  length : ℝ

/-- Represents a space region around a line segment -/
structure SpaceRegion where
  segment : LineSegment3D
  radius : ℝ
  volume : ℝ

/-- Theorem: If a space region containing all points within 5 units of a line segment 
    in three-dimensional space has a volume of 500π, then the length of the line segment is 40/3 units. -/
theorem line_segment_length (region : SpaceRegion) 
  (h1 : region.radius = 5)
  (h2 : region.volume = 500 * Real.pi) : 
  region.segment.length = 40 / 3 := by
  sorry

end NUMINAMATH_CALUDE_line_segment_length_l3338_333811


namespace NUMINAMATH_CALUDE_cos_squared_plus_half_sin_double_l3338_333815

theorem cos_squared_plus_half_sin_double (θ : ℝ) :
  3 * Real.cos (π / 2 - θ) + Real.cos (π + θ) = 0 →
  Real.cos θ ^ 2 + (1 / 2) * Real.sin (2 * θ) = 6 / 5 := by
  sorry

end NUMINAMATH_CALUDE_cos_squared_plus_half_sin_double_l3338_333815


namespace NUMINAMATH_CALUDE_least_addition_for_divisibility_l3338_333809

theorem least_addition_for_divisibility (n : ℕ) : 
  (∃ k : ℕ, k > 0 ∧ (625573 + k) % 3 = 0) → 
  (625573 + 2) % 3 = 0 ∧ ∀ m : ℕ, m < 2 → (625573 + m) % 3 ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_least_addition_for_divisibility_l3338_333809


namespace NUMINAMATH_CALUDE_handshakes_in_specific_convention_l3338_333866

/-- Represents a convention with companies and representatives -/
structure Convention where
  num_companies : ℕ
  reps_per_company : ℕ
  companies_to_shake : ℕ

/-- Calculates the total number of handshakes in the convention -/
def total_handshakes (conv : Convention) : ℕ :=
  let total_people := conv.num_companies * conv.reps_per_company
  let handshakes_per_person := (conv.companies_to_shake * conv.reps_per_company)
  (total_people * handshakes_per_person) / 2

/-- The specific convention described in the problem -/
def specific_convention : Convention :=
  { num_companies := 5
  , reps_per_company := 4
  , companies_to_shake := 2 }

theorem handshakes_in_specific_convention :
  total_handshakes specific_convention = 80 := by
  sorry


end NUMINAMATH_CALUDE_handshakes_in_specific_convention_l3338_333866
