import Mathlib

namespace discount_comparison_l197_19710

/-- The original bill amount in dollars -/
def original_bill : ℝ := 8000

/-- The single discount rate -/
def single_discount_rate : ℝ := 0.3

/-- The first successive discount rate -/
def first_successive_discount_rate : ℝ := 0.2

/-- The second successive discount rate -/
def second_successive_discount_rate : ℝ := 0.1

/-- The difference between the two discount scenarios -/
def discount_difference : ℝ := 160

theorem discount_comparison :
  let single_discounted := original_bill * (1 - single_discount_rate)
  let successive_discounted := original_bill * (1 - first_successive_discount_rate) * (1 - second_successive_discount_rate)
  successive_discounted - single_discounted = discount_difference := by
  sorry

end discount_comparison_l197_19710


namespace sqrt_equation_solution_l197_19725

theorem sqrt_equation_solution : ∃ x : ℝ, x = 225 / 16 ∧ Real.sqrt x + Real.sqrt (x + 4) = 8 := by
  sorry

end sqrt_equation_solution_l197_19725


namespace batsman_85_run_innings_l197_19700

/-- Represents a batsman's scoring record -/
structure Batsman where
  totalRuns : ℕ
  totalInnings : ℕ

/-- Calculate the average score of a batsman -/
def average (b : Batsman) : ℚ :=
  (b.totalRuns : ℚ) / b.totalInnings

/-- The innings in which the batsman scored 85 -/
def targetInnings (b : Batsman) : ℕ := b.totalInnings

theorem batsman_85_run_innings (b : Batsman) 
  (h1 : average b = 37)
  (h2 : average { totalRuns := b.totalRuns - 85, totalInnings := b.totalInnings - 1 } = 34) :
  targetInnings b = 17 := by
  sorry

end batsman_85_run_innings_l197_19700


namespace circle_radius_proof_l197_19746

theorem circle_radius_proof (x y : ℝ) (h : x + y = 100 * Real.pi) :
  ∃ (r : ℝ), r > 0 ∧ x = Real.pi * r^2 ∧ y = 2 * Real.pi * r ∧ r = 10 := by
sorry

end circle_radius_proof_l197_19746


namespace grocery_shopping_theorem_l197_19797

def initial_amount : ℝ := 100
def roast_price : ℝ := 17
def vegetables_price : ℝ := 11
def wine_price : ℝ := 12
def dessert_price : ℝ := 8
def bread_price : ℝ := 4
def milk_price : ℝ := 2
def discount_rate : ℝ := 0.15
def tax_rate : ℝ := 0.05

def total_purchase : ℝ := roast_price + vegetables_price + wine_price + dessert_price + bread_price + milk_price

def discounted_total : ℝ := total_purchase * (1 - discount_rate)

def final_amount : ℝ := discounted_total * (1 + tax_rate)

def remaining_amount : ℝ := initial_amount - final_amount

theorem grocery_shopping_theorem : 
  ∃ (ε : ℝ), abs (remaining_amount - 51.80) < ε ∧ ε > 0 :=
by sorry

end grocery_shopping_theorem_l197_19797


namespace seventh_root_of_unity_sum_l197_19771

theorem seventh_root_of_unity_sum (z : ℂ) (h1 : z^7 = 1) (h2 : z ≠ 1) :
  ∃ (sign : Bool), z + z^2 + z^4 = (-1 + (if sign then 1 else -1) * Complex.I * Real.sqrt 7) / 2 := by
  sorry

end seventh_root_of_unity_sum_l197_19771


namespace joe_paint_usage_l197_19766

theorem joe_paint_usage (total_paint : ℝ) (second_week_fraction : ℝ) (total_used : ℝ) :
  total_paint = 360 →
  second_week_fraction = 1 / 7 →
  total_used = 128.57 →
  ∃ (first_week_fraction : ℝ),
    first_week_fraction * total_paint +
    second_week_fraction * (total_paint - first_week_fraction * total_paint) = total_used ∧
    first_week_fraction = 1 / 4 := by
  sorry

end joe_paint_usage_l197_19766


namespace work_completion_time_l197_19755

/-- The number of days it takes A to complete the work alone -/
def a_days : ℝ := 40

/-- The number of days it takes A and B to complete the work together -/
def ab_days : ℝ := 24

/-- The number of days it takes B to complete the work alone -/
def b_days : ℝ := 60

/-- Theorem stating that if A can do the work in 40 days and A and B together can do it in 24 days, 
    then B can do the work alone in 60 days -/
theorem work_completion_time : 
  (1 / a_days + 1 / b_days = 1 / ab_days) ∧ (b_days = 60) :=
by sorry

end work_completion_time_l197_19755


namespace sum_of_squares_of_coefficients_eq_198_l197_19718

/-- The polynomial for which we calculate the sum of squares of coefficients -/
def p (x : ℝ) : ℝ := 3 * (x^5 + 4*x^3 + 2*x + 1)

/-- The sum of squares of coefficients of the polynomial p -/
def sum_of_squares_of_coefficients : ℝ :=
  (3^2) + (12^2) + (6^2) + (3^2) + (0^2) + (0^2)

theorem sum_of_squares_of_coefficients_eq_198 :
  sum_of_squares_of_coefficients = 198 := by sorry

end sum_of_squares_of_coefficients_eq_198_l197_19718


namespace min_p_value_l197_19761

/-- The probability that Alex and Dylan are on the same team, given that Alex picks one of the cards a or a+7, and Dylan picks the other. -/
def p (a : ℕ) : ℚ :=
  let total_combinations := (52 - 2).choose 2
  let lower_team_combinations := (44 - a).choose 2
  let higher_team_combinations := (a - 1).choose 2
  (lower_team_combinations + higher_team_combinations : ℚ) / total_combinations

/-- The minimum value of a for which p(a) ≥ 1/2 -/
def min_a : ℕ := 8

theorem min_p_value :
  p min_a = 73 / 137 ∧ 
  p min_a ≥ 1 / 2 ∧
  ∀ a : ℕ, a < min_a → p a < 1 / 2 := by sorry

end min_p_value_l197_19761


namespace rebecca_pie_slices_l197_19792

theorem rebecca_pie_slices (total_pies : ℕ) (slices_per_pie : ℕ) 
  (remaining_slices : ℕ) (rebecca_husband_slices : ℕ) 
  (family_friends_percent : ℚ) :
  total_pies = 2 →
  slices_per_pie = 8 →
  remaining_slices = 5 →
  rebecca_husband_slices = 2 →
  family_friends_percent = 1/2 →
  ∃ (rebecca_initial_slices : ℕ),
    rebecca_initial_slices = total_pies * slices_per_pie - 
      ((remaining_slices + rebecca_husband_slices) / family_friends_percent) :=
by sorry

end rebecca_pie_slices_l197_19792


namespace kates_hair_length_l197_19727

theorem kates_hair_length :
  ∀ (kate emily logan : ℝ),
  kate = (1/2) * emily →
  emily = logan + 6 →
  logan = 20 →
  kate = 13 := by
sorry

end kates_hair_length_l197_19727


namespace trip_duration_is_six_hours_l197_19791

/-- Represents the position of a clock hand in minutes (0-59) -/
def ClockPosition : Type := Fin 60

/-- Represents a time of day in hours and minutes -/
structure TimeOfDay where
  hours : Fin 24
  minutes : Fin 60

/-- Returns true if the hour and minute hands coincide at the given time -/
def hands_coincide (t : TimeOfDay) : Prop :=
  (t.hours.val * 5 + t.minutes.val / 12 : ℚ) = t.minutes.val

/-- Returns true if the hour and minute hands form a 180° angle at the given time -/
def hands_opposite (t : TimeOfDay) : Prop :=
  ((t.hours.val * 5 + t.minutes.val / 12 : ℚ) + 30) % 60 = t.minutes.val

/-- The start time of the trip -/
def start_time : TimeOfDay :=
  { hours := 8, minutes := 43 }

/-- The end time of the trip -/
def end_time : TimeOfDay :=
  { hours := 14, minutes := 43 }

theorem trip_duration_is_six_hours :
  hands_coincide start_time →
  hands_opposite end_time →
  start_time.hours.val < 9 →
  end_time.hours.val > 14 ∧ end_time.hours.val < 15 →
  (end_time.hours.val - start_time.hours.val : ℕ) = 6 :=
sorry

end trip_duration_is_six_hours_l197_19791


namespace outfits_count_l197_19740

/-- The number of shirts available --/
def num_shirts : ℕ := 8

/-- The number of pairs of pants available --/
def num_pants : ℕ := 5

/-- The number of ties available --/
def num_ties : ℕ := 4

/-- The number of hats available --/
def num_hats : ℕ := 2

/-- The total number of outfit combinations --/
def total_outfits : ℕ := num_shirts * num_pants * (num_ties + 1) * (num_hats + 1)

/-- Theorem stating that the total number of outfits is 600 --/
theorem outfits_count : total_outfits = 600 := by
  sorry

end outfits_count_l197_19740


namespace lcm_gcf_relation_l197_19788

theorem lcm_gcf_relation (n : ℕ) :
  (Nat.lcm n 12 = 42) ∧ (Nat.gcd n 12 = 6) → n = 21 := by
  sorry

end lcm_gcf_relation_l197_19788


namespace statue_model_ratio_l197_19781

/-- Given a statue of height 75 feet and a model of height 5 inches,
    prove that one inch of the model represents 15 feet of the statue. -/
theorem statue_model_ratio :
  let statue_height : ℝ := 75  -- statue height in feet
  let model_height : ℝ := 5    -- model height in inches
  statue_height / model_height = 15 := by
sorry


end statue_model_ratio_l197_19781


namespace gcf_and_lcm_of_numbers_l197_19790

def numbers : List Nat := [42, 126, 105]

theorem gcf_and_lcm_of_numbers :
  (Nat.gcd (Nat.gcd 42 126) 105 = 21) ∧
  (Nat.lcm (Nat.lcm 42 126) 105 = 630) := by
  sorry

end gcf_and_lcm_of_numbers_l197_19790


namespace imaginary_part_of_z_l197_19739

theorem imaginary_part_of_z (z : ℂ) (h : (3 - 4*I)*z = Complex.abs (3 - 4*I)) :
  z.im = 4/5 := by
  sorry

end imaginary_part_of_z_l197_19739


namespace cube_inequality_l197_19709

theorem cube_inequality (a x y : ℝ) (h1 : 0 < a) (h2 : a < 1) (h3 : a^x < a^y) : x^3 > y^3 := by
  sorry

end cube_inequality_l197_19709


namespace two_digit_number_digit_difference_l197_19719

/-- Given a two-digit number where the difference between the original number
    and the number with interchanged digits is 45, prove that the difference
    between its two digits is 5. -/
theorem two_digit_number_digit_difference (x y : ℕ) : 
  x < 10 → y < 10 → (10 * x + y) - (10 * y + x) = 45 → x - y = 5 := by
sorry

end two_digit_number_digit_difference_l197_19719


namespace circle_area_theorem_l197_19706

def A : ℝ × ℝ := (8, 15)
def B : ℝ × ℝ := (14, 9)

structure Circle where
  center : ℝ × ℝ
  radius : ℝ

def tangent_line (c : Circle) (p : ℝ × ℝ) : Set (ℝ × ℝ) := sorry

def on_circle (p : ℝ × ℝ) (c : Circle) : Prop := sorry

def intersect_x_axis (l₁ l₂ : Set (ℝ × ℝ)) : Prop := sorry

theorem circle_area_theorem (ω : Circle) :
  on_circle A ω →
  on_circle B ω →
  (∃ (p : ℝ × ℝ), p.2 = 0 ∧ p ∈ tangent_line ω A ∧ p ∈ tangent_line ω B) →
  ω.radius^2 * Real.pi = 306 * Real.pi :=
sorry

end circle_area_theorem_l197_19706


namespace log_inequality_l197_19751

theorem log_inequality : 
  Real.log 2 / Real.log 3 < 2/3 ∧ 
  2/3 < Real.log 75 / Real.log 625 ∧ 
  Real.log 75 / Real.log 625 < Real.log 3 / Real.log 5 := by
  sorry

end log_inequality_l197_19751


namespace food_relation_values_l197_19737

def A : Set ℝ := {-1, 1/2, 1}

def B (a : ℝ) : Set ℝ := {x : ℝ | a * x^2 = 1 ∧ a ≥ 0}

def is_full_food (X Y : Set ℝ) : Prop := X ⊆ Y ∨ Y ⊆ X

def is_partial_food (X Y : Set ℝ) : Prop :=
  (∃ x, x ∈ X ∧ x ∈ Y) ∧ ¬(X ⊆ Y) ∧ ¬(Y ⊆ X)

theorem food_relation_values :
  ∀ a : ℝ, (is_full_food A (B a) ∨ is_partial_food A (B a)) ↔ (a = 0 ∨ a = 1 ∨ a = 4) :=
sorry

end food_relation_values_l197_19737


namespace loan_duration_b_l197_19720

/-- Proves that the loan duration for B is 2 years given the problem conditions -/
theorem loan_duration_b (principal_b : ℕ) (principal_c : ℕ) (rate : ℚ) 
  (duration_c : ℕ) (total_interest : ℕ) :
  principal_b = 5000 →
  principal_c = 3000 →
  rate = 8/100 →
  duration_c = 4 →
  total_interest = 1760 →
  ∃ (n : ℕ), n = 2 ∧ 
    (principal_b * rate * n + principal_c * rate * duration_c = total_interest) :=
by sorry

end loan_duration_b_l197_19720


namespace min_attacking_pairs_8x8_16rooks_l197_19796

/-- Represents a chessboard configuration -/
structure ChessBoard where
  size : Nat
  rooks : Nat

/-- Calculates the minimum number of attacking rook pairs on a chessboard -/
def minAttackingPairs (board : ChessBoard) : Nat :=
  sorry

/-- Theorem: The minimum number of attacking rook pairs on an 8x8 board with 16 rooks is 16 -/
theorem min_attacking_pairs_8x8_16rooks :
  ∀ (board : ChessBoard),
    board.size = 8 ∧ board.rooks = 16 →
    minAttackingPairs board = 16 := by
  sorry

end min_attacking_pairs_8x8_16rooks_l197_19796


namespace minimal_point_in_rectangle_l197_19754

/-- Given positive real numbers a and b, the point (a/2, b/2) minimizes the sum of distances
    to the corners of the rectangle with vertices at (0,0), (a,0), (0,b), and (a,b). -/
theorem minimal_point_in_rectangle (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  ∀ x y, 0 < x → x < a → 0 < y → y < b →
  Real.sqrt (x^2 + y^2) + Real.sqrt (x^2 + (b-y)^2) + 
  Real.sqrt ((a-x)^2 + y^2) + Real.sqrt ((a-x)^2 + (b-y)^2) ≥
  Real.sqrt ((a/2)^2 + (b/2)^2) + Real.sqrt ((a/2)^2 + (b/2)^2) + 
  Real.sqrt ((a/2)^2 + (b/2)^2) + Real.sqrt ((a/2)^2 + (b/2)^2) :=
by sorry


end minimal_point_in_rectangle_l197_19754


namespace solution_exists_l197_19748

def f (x : ℝ) : ℝ := 2 * x - 3

def d : ℝ := 2

theorem solution_exists : ∃ x : ℝ, 2 * (f x) - 11 = f (x - d) :=
  sorry

end solution_exists_l197_19748


namespace variance_transformed_l197_19723

-- Define a random variable ξ
variable (ξ : ℝ → ℝ)

-- Define the variance operator D
noncomputable def D (X : ℝ → ℝ) : ℝ := sorry

-- Given condition
axiom variance_xi : D ξ = 2

-- Theorem to prove
theorem variance_transformed : D (fun ω => 2 * ξ ω + 3) = 8 := by sorry

end variance_transformed_l197_19723


namespace difference_of_squares_l197_19733

theorem difference_of_squares (a : ℝ) : a^2 - 100 = (a + 10) * (a - 10) := by
  sorry

end difference_of_squares_l197_19733


namespace min_players_distinct_scores_l197_19708

/-- A round robin chess tournament where each player plays every other player exactly once. -/
structure Tournament (n : ℕ) where
  scores : Fin n → ℚ

/-- Property P(m) for a tournament -/
def hasPropertyP (t : Tournament n) (m : ℕ) : Prop :=
  ∀ (S : Finset (Fin n)), S.card = m →
    (∃ (w : Fin n), w ∈ S ∧ ∀ (x : Fin n), x ∈ S ∧ x ≠ w → t.scores w > t.scores x) ∧
    (∃ (l : Fin n), l ∈ S ∧ ∀ (x : Fin n), x ∈ S ∧ x ≠ l → t.scores l < t.scores x)

/-- All scores in the tournament are distinct -/
def hasDistinctScores (t : Tournament n) : Prop :=
  ∀ (i j : Fin n), i ≠ j → t.scores i ≠ t.scores j

/-- The main theorem -/
theorem min_players_distinct_scores (m : ℕ) (h : m ≥ 4) :
  (∀ (n : ℕ), n ≥ 2*m - 3 →
    ∀ (t : Tournament n), hasPropertyP t m → hasDistinctScores t) ∧
  (∃ (t : Tournament (2*m - 4)), hasPropertyP t m ∧ ¬hasDistinctScores t) :=
sorry

end min_players_distinct_scores_l197_19708


namespace train_speed_theorem_l197_19703

-- Define the given constants
def train_length : ℝ := 110
def bridge_length : ℝ := 170
def crossing_time : ℝ := 16.7986561075114

-- Define the theorem
theorem train_speed_theorem :
  let total_distance := train_length + bridge_length
  let speed_ms := total_distance / crossing_time
  let speed_kmh := speed_ms * 3.6
  speed_kmh = 60 := by sorry

end train_speed_theorem_l197_19703


namespace principal_sum_from_interest_difference_l197_19756

/-- Proves that for a given interest rate and time period, if the difference between
    compound interest and simple interest is 41, then the principal sum is 4100. -/
theorem principal_sum_from_interest_difference
  (rate : ℝ) (time : ℝ) (diff : ℝ) (p : ℝ) :
  rate = 10 →
  time = 2 →
  diff = 41 →
  diff = p * ((1 + rate / 100) ^ time - 1) - p * (rate * time / 100) →
  p = 4100 := by
  sorry

#check principal_sum_from_interest_difference

end principal_sum_from_interest_difference_l197_19756


namespace roots_order_l197_19795

variables (a b m n : ℝ)

-- Define the equation
def f (x : ℝ) : ℝ := 1 - (x - a) * (x - b)

theorem roots_order (h1 : f m = 0) (h2 : f n = 0) (h3 : m < n) (h4 : a < b) :
  m < a ∧ a < b ∧ b < n := by
  sorry

end roots_order_l197_19795


namespace fraction_ordering_l197_19789

theorem fraction_ordering : (7 : ℚ) / 29 < 11 / 33 ∧ 11 / 33 < 13 / 31 := by
  sorry

end fraction_ordering_l197_19789


namespace two_times_choose_six_two_l197_19784

theorem two_times_choose_six_two : 2 * (Nat.choose 6 2) = 30 := by
  sorry

end two_times_choose_six_two_l197_19784


namespace shopkeeper_decks_l197_19732

theorem shopkeeper_decks (total_red_cards : ℕ) (cards_per_deck : ℕ) (colors_per_deck : ℕ) (red_suits_per_deck : ℕ) (cards_per_suit : ℕ) : 
  total_red_cards = 182 →
  cards_per_deck = 52 →
  colors_per_deck = 2 →
  red_suits_per_deck = 2 →
  cards_per_suit = 13 →
  (total_red_cards / (red_suits_per_deck * cards_per_suit) : ℕ) = 7 := by
sorry

end shopkeeper_decks_l197_19732


namespace pyramid_max_volume_l197_19793

theorem pyramid_max_volume (a b c h : ℝ) (angle : ℝ) :
  a = 5 ∧ b = 12 ∧ c = 13 →
  a^2 + b^2 = c^2 →
  angle ≥ 30 * π / 180 →
  (∃ (height : ℝ), height > 0 ∧
    (∀ (face_height : ℝ), face_height > 0 →
      Real.cos (Real.arccos (height / face_height)) ≥ Real.cos angle)) →
  (1/3 : ℝ) * (1/2 * a * b) * h ≤ 150 * Real.sqrt 3 :=
by sorry

end pyramid_max_volume_l197_19793


namespace rectangular_box_dimensions_l197_19747

theorem rectangular_box_dimensions (A B C : ℝ) : 
  A > 0 ∧ B > 0 ∧ C > 0 →
  A * B = 40 →
  A * C = 90 →
  B * C = 100 →
  A + B + C = 83 / 3 := by
sorry

end rectangular_box_dimensions_l197_19747


namespace line_intercepts_minimum_minimum_sum_of_intercepts_l197_19757

theorem line_intercepts_minimum (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_line : a + b = a * b) : 
  (b / a) + (a / b) ≥ 2 ∧ 
  (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + y = a * b ∧ x / a + y / b = 2) :=
by sorry

theorem minimum_sum_of_intercepts (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_line : a + b = a * b) :
  a + b ≥ 4 ∧ (a + b = 4 ↔ a = 2 ∧ b = 2) :=
by sorry

end line_intercepts_minimum_minimum_sum_of_intercepts_l197_19757


namespace complex_expression_equality_l197_19762

theorem complex_expression_equality (z : ℂ) (h : z = 1 + Complex.I) :
  5 / z + z^2 = 5/2 - (1/2) * Complex.I := by
  sorry

end complex_expression_equality_l197_19762


namespace waiter_tip_earnings_l197_19787

theorem waiter_tip_earnings (total_customers : ℕ) (non_tipping_customers : ℕ) (tip_amount : ℕ) (total_earnings : ℕ) : 
  total_customers = 10 →
  non_tipping_customers = 5 →
  tip_amount = 3 →
  total_earnings = (total_customers - non_tipping_customers) * tip_amount →
  total_earnings = 15 :=
by sorry

end waiter_tip_earnings_l197_19787


namespace complex_number_location_l197_19729

theorem complex_number_location (z : ℂ) (h : (2 - 3*Complex.I)*z = 1 + Complex.I) :
  z.re < 0 ∧ z.im > 0 :=
by sorry

end complex_number_location_l197_19729


namespace classroom_students_l197_19734

theorem classroom_students (T : ℕ) (S : ℕ) (n : ℕ) : 
  (T = S / n + 24) →  -- Teacher's age is 24 years more than average student age
  (T = (T + S) / (n + 1) + 20) →  -- Teacher's age is 20 years more than average age of everyone
  (n = 5) := by  -- Number of students is 5
sorry

end classroom_students_l197_19734


namespace teresa_black_pencils_l197_19724

/-- Given Teresa's pencil distribution problem, prove she has 35 black pencils. -/
theorem teresa_black_pencils : 
  (colored_pencils : ℕ) →
  (siblings : ℕ) →
  (pencils_per_sibling : ℕ) →
  (pencils_kept : ℕ) →
  colored_pencils = 14 →
  siblings = 3 →
  pencils_per_sibling = 13 →
  pencils_kept = 10 →
  (siblings * pencils_per_sibling + pencils_kept) - colored_pencils = 35 := by
sorry

end teresa_black_pencils_l197_19724


namespace cos_equality_l197_19713

theorem cos_equality (n : ℤ) (h1 : 0 ≤ n) (h2 : n ≤ 180) : 
  n = 43 → Real.cos (n * π / 180) = Real.cos (317 * π / 180) := by
  sorry

end cos_equality_l197_19713


namespace quadratic_real_solutions_implies_m_less_than_two_l197_19752

/-- A quadratic equation with parameter m -/
def quadratic_equation (x m : ℝ) : Prop :=
  x^2 + 2*x + m = 0

/-- The discriminant of the quadratic equation -/
def discriminant (m : ℝ) : ℝ :=
  4 - 4*m

theorem quadratic_real_solutions_implies_m_less_than_two (m : ℝ) :
  (∃ x : ℝ, quadratic_equation x m) → m < 2 := by
  sorry

end quadratic_real_solutions_implies_m_less_than_two_l197_19752


namespace sets_intersection_and_union_l197_19778

def A : Set ℝ := {x | x^2 + 2*x - 3 < 0}
def B : Set ℝ := {x | (x+2)*(x-3) < 0}

theorem sets_intersection_and_union :
  (A ∩ B = {x : ℝ | -2 < x ∧ x < 1}) ∧
  (A ∪ B = {x : ℝ | -3 < x ∧ x < 3}) := by
  sorry

end sets_intersection_and_union_l197_19778


namespace positive_x_axis_line_m_range_l197_19741

/-- A line passing through the positive half-axis of the x-axis -/
structure PositiveXAxisLine where
  m : ℝ
  equation : ℝ → ℝ
  equation_def : ∀ x, equation x = 2 * x + m - 3
  passes_positive_x : ∃ x > 0, equation x = 0

/-- The range of m for a line passing through the positive half-axis of the x-axis -/
theorem positive_x_axis_line_m_range (line : PositiveXAxisLine) : line.m < 3 := by
  sorry


end positive_x_axis_line_m_range_l197_19741


namespace g_g_2_equals_263_l197_19785

def g (x : ℝ) : ℝ := 2 * x^2 + 2 * x - 1

theorem g_g_2_equals_263 : g (g 2) = 263 := by
  sorry

end g_g_2_equals_263_l197_19785


namespace polynomial_value_at_8_l197_19721

def is_monic_degree_7 (p : ℝ → ℝ) : Prop :=
  ∃ a b c d e f : ℝ, p = λ x => x^7 + a*x^6 + b*x^5 + c*x^4 + d*x^3 + e*x^2 + f*x + (p 0)

theorem polynomial_value_at_8 
  (p : ℝ → ℝ) 
  (h_monic : is_monic_degree_7 p)
  (h1 : p 1 = 1) (h2 : p 2 = 2) (h3 : p 3 = 3) (h4 : p 4 = 4)
  (h5 : p 5 = 5) (h6 : p 6 = 6) (h7 : p 7 = 7) :
  p 8 = 5048 := by
sorry

end polynomial_value_at_8_l197_19721


namespace difference_of_squares_example_l197_19750

theorem difference_of_squares_example : (23 + 15)^2 - (23 - 15)^2 = 1380 := by
  sorry

end difference_of_squares_example_l197_19750


namespace find_principal_amount_l197_19745

/-- Given compound and simple interest for 2 years, find the principal amount -/
theorem find_principal_amount (compound_interest simple_interest : ℚ) : 
  compound_interest = 11730 → 
  simple_interest = 10200 → 
  ∃ (principal rate : ℚ), 
    principal > 0 ∧ 
    rate > 0 ∧ 
    rate < 100 ∧
    compound_interest = principal * ((1 + rate / 100) ^ 2 - 1) ∧
    simple_interest = principal * rate * 2 / 100 ∧
    principal = 1700 :=
by sorry

end find_principal_amount_l197_19745


namespace hyperbola_equation_l197_19711

/-- Represents a hyperbola -/
structure Hyperbola where
  equation : ℝ → ℝ → Prop

/-- Checks if a point lies on the hyperbola -/
def Hyperbola.contains (h : Hyperbola) (x y : ℝ) : Prop :=
  h.equation x y

/-- Checks if a line is an asymptote of the hyperbola -/
def Hyperbola.hasAsymptote (h : Hyperbola) (f : ℝ → ℝ) : Prop :=
  ∀ ε > 0, ∃ δ > 0, ∀ x y, h.contains x y → |y - f x| < ε ∨ |x| > δ

theorem hyperbola_equation (h : Hyperbola) :
  (h.hasAsymptote (fun x ↦ 2 * x) ∧ h.hasAsymptote (fun x ↦ -2 * x)) →
  h.contains 1 (2 * Real.sqrt 5) →
  h.equation = fun x y ↦ y^2 / 16 - x^2 / 4 = 1 :=
sorry

end hyperbola_equation_l197_19711


namespace three_cats_meowing_l197_19728

/-- The number of meows for three cats in a given time period -/
def total_meows (cat1_freq : ℕ) (time : ℕ) : ℕ :=
  let cat2_freq := 2 * cat1_freq
  let cat3_freq := cat2_freq / 3
  (cat1_freq + cat2_freq + cat3_freq) * time

/-- Theorem stating that the total number of meows for three cats in 5 minutes is 55 -/
theorem three_cats_meowing (cat1_freq : ℕ) (h : cat1_freq = 3) :
  total_meows cat1_freq 5 = 55 := by
  sorry

#eval total_meows 3 5

end three_cats_meowing_l197_19728


namespace storeroom_items_proof_l197_19779

/-- Calculates the number of items in the storeroom given the number of restocked items,
    sold items, and total items left in the store. -/
def items_in_storeroom (restocked : ℕ) (sold : ℕ) (total_left : ℕ) : ℕ :=
  total_left - (restocked - sold)

/-- Proves that the number of items in the storeroom is 575 given the specific conditions. -/
theorem storeroom_items_proof :
  items_in_storeroom 4458 1561 3472 = 575 := by
  sorry

#eval items_in_storeroom 4458 1561 3472

end storeroom_items_proof_l197_19779


namespace consecutive_integers_product_812_sum_l197_19769

theorem consecutive_integers_product_812_sum (x : ℕ) : 
  x > 0 ∧ x * (x + 1) = 812 → x + (x + 1) = 57 := by
  sorry

end consecutive_integers_product_812_sum_l197_19769


namespace counterexample_exists_l197_19775

theorem counterexample_exists : ∃ n : ℕ, 
  2 ∣ n ∧ ¬ Nat.Prime n ∧ Nat.Prime (n - 3) := by sorry

end counterexample_exists_l197_19775


namespace hotdog_cost_l197_19731

/-- Represents the cost of Sara's lunch items in dollars -/
structure LunchCost where
  total : ℝ
  salad : ℝ
  hotdog : ℝ

/-- Theorem stating that given the total lunch cost and salad cost, the hotdog cost can be determined -/
theorem hotdog_cost (lunch : LunchCost) 
  (h1 : lunch.total = 10.46)
  (h2 : lunch.salad = 5.1)
  (h3 : lunch.total = lunch.salad + lunch.hotdog) : 
  lunch.hotdog = 5.36 := by
  sorry

#check hotdog_cost

end hotdog_cost_l197_19731


namespace area_between_curves_l197_19722

theorem area_between_curves : 
  let f (x : ℝ) := x^2
  let g (x : ℝ) := x^3
  ∫ x in (0: ℝ)..(1: ℝ), f x - g x = 1/12 := by sorry

end area_between_curves_l197_19722


namespace line_equation_equivalence_l197_19705

theorem line_equation_equivalence (x y : ℝ) (h : 2*x - 5*y - 3 = 0) : 
  -4*x + 10*y + 3 = 0 := by sorry

end line_equation_equivalence_l197_19705


namespace grading_multiple_l197_19765

theorem grading_multiple (total_questions : ℕ) (score : ℕ) (correct_responses : ℕ) :
  total_questions = 100 →
  score = 70 →
  correct_responses = 90 →
  ∃ m : ℕ, score = correct_responses - m * (total_questions - correct_responses) →
  m = 2 := by
sorry

end grading_multiple_l197_19765


namespace polygon_interior_angles_increase_l197_19782

theorem polygon_interior_angles_increase (n : ℕ) :
  (n + 1 - 2) * 180 - (n - 2) * 180 = 180 → n + 1 - n = 1 := by
  sorry

end polygon_interior_angles_increase_l197_19782


namespace weekly_calorie_allowance_l197_19799

/-- The number of calories to reduce from the average daily allowance to hypothetically live to 100 years old -/
def calorie_reduction : ℕ := 500

/-- The average daily calorie allowance for a person in their 60's -/
def average_daily_allowance : ℕ := 2000

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The weekly calorie allowance for a person in their 60's to hypothetically live to 100 years old -/
theorem weekly_calorie_allowance :
  (average_daily_allowance - calorie_reduction) * days_in_week = 10500 := by
  sorry

end weekly_calorie_allowance_l197_19799


namespace daves_monday_hours_l197_19759

/-- 
Given:
- Dave's hourly rate is $6
- Dave worked on Monday and Tuesday
- On Tuesday, Dave worked 2 hours
- Dave made $48 in total for both days

Prove: Dave worked 6 hours on Monday
-/
theorem daves_monday_hours 
  (hourly_rate : ℕ) 
  (tuesday_hours : ℕ) 
  (total_earnings : ℕ) 
  (h1 : hourly_rate = 6)
  (h2 : tuesday_hours = 2)
  (h3 : total_earnings = 48) : 
  ∃ (monday_hours : ℕ), 
    hourly_rate * (monday_hours + tuesday_hours) = total_earnings ∧ 
    monday_hours = 6 := by
  sorry

#check daves_monday_hours

end daves_monday_hours_l197_19759


namespace art_gallery_theorem_l197_19749

theorem art_gallery_theorem (total : ℕ) 
  (h1 : total > 0)
  (h2 : (total / 3 : ℚ) = total / 3)  -- Ensures division is exact
  (h3 : ((total / 3) / 6 : ℚ) = (total / 3) / 6)  -- Ensures division is exact
  (h4 : ((2 * total / 3) / 3 : ℚ) = (2 * total / 3) / 3)  -- Ensures division is exact
  (h5 : 2 * (2 * total / 3) / 3 = 1200) :
  total = 2700 := by
sorry

end art_gallery_theorem_l197_19749


namespace smallest_x_value_l197_19773

theorem smallest_x_value (x : ℝ) : 
  ((((5 * x - 20) / (4 * x - 5)) ^ 2 + ((5 * x - 20) / (4 * x - 5))) = 6) →
  x ≥ 35 / 17 :=
by
  sorry

end smallest_x_value_l197_19773


namespace joneal_stops_in_quarter_A_l197_19715

/-- Represents the quarters of the circular track -/
inductive Quarter : Type
| A : Quarter
| B : Quarter
| C : Quarter
| D : Quarter

/-- Calculates the quarter in which a runner stops after running a given distance -/
def stopQuarter (trackCircumference : ℕ) (runDistance : ℕ) : Quarter :=
  match (runDistance % trackCircumference) / (trackCircumference / 4) with
  | 0 => Quarter.A
  | 1 => Quarter.B
  | 2 => Quarter.C
  | _ => Quarter.D

theorem joneal_stops_in_quarter_A :
  let trackCircumference : ℕ := 100
  let runDistance : ℕ := 10000
  stopQuarter trackCircumference runDistance = Quarter.A := by
  sorry

end joneal_stops_in_quarter_A_l197_19715


namespace sqrt_2_simplest_l197_19702

def is_simplest_sqrt (x : ℝ) (others : List ℝ) : Prop :=
  ∀ y ∈ others, ¬∃ (n : ℕ) (r : ℝ), n > 1 ∧ y = n * Real.sqrt r

theorem sqrt_2_simplest : is_simplest_sqrt (Real.sqrt 2) [Real.sqrt 8, Real.sqrt 12, Real.sqrt 18] := by
  sorry

end sqrt_2_simplest_l197_19702


namespace f_properties_l197_19714

-- Define the function f
def f (x : ℝ) : ℝ := x^3 + 3*x^2 - 9*x

-- State the theorem
theorem f_properties :
  -- Part 1: Monotonicity of f
  (∀ x < -3, ∀ y < -3, x < y → f x < f y) ∧
  (∀ x ∈ Set.Ioo (-3) 1, ∀ y ∈ Set.Ioo (-3) 1, x < y → f x > f y) ∧
  (∀ x > 1, ∀ y > 1, x < y → f x < f y) ∧
  -- Part 2: Minimum value condition
  (∀ c : ℝ, (∀ x ∈ Set.Icc (-4) c, f x ≥ -5) ∧ (∃ x ∈ Set.Icc (-4) c, f x = -5) ↔ c ≥ 1) :=
by sorry

end f_properties_l197_19714


namespace max_product_of_three_distinct_naturals_l197_19738

theorem max_product_of_three_distinct_naturals (a b c : ℕ) : 
  a ≠ b → b ≠ c → a ≠ c → a + b + c = 48 → a * b * c ≤ 4080 := by
  sorry

end max_product_of_three_distinct_naturals_l197_19738


namespace kindergarten_class_average_l197_19736

theorem kindergarten_class_average (giraffe elephant rabbit : ℕ) : 
  giraffe = 225 →
  elephant = giraffe + 48 →
  rabbit = giraffe - 24 →
  (giraffe + elephant + rabbit) / 3 = 233 :=
by sorry

end kindergarten_class_average_l197_19736


namespace nonzero_even_from_second_step_l197_19730

/-- Represents a bi-infinite sequence of integers -/
def BiInfiniteSequence := ℤ → ℤ

/-- The initial sequence with one 1 and all other elements 0 -/
def initial_sequence : BiInfiniteSequence :=
  fun i => if i = 0 then 1 else 0

/-- The next sequence after one step of evolution -/
def next_sequence (s : BiInfiniteSequence) : BiInfiniteSequence :=
  fun i => s (i - 1) + s i + s (i + 1)

/-- The sequence after n steps of evolution -/
def evolved_sequence (n : ℕ) : BiInfiniteSequence :=
  match n with
  | 0 => initial_sequence
  | m + 1 => next_sequence (evolved_sequence m)

/-- Predicate to check if a sequence contains a non-zero even number -/
def contains_nonzero_even (s : BiInfiniteSequence) : Prop :=
  ∃ i : ℤ, s i ≠ 0 ∧ s i % 2 = 0

/-- The main theorem to be proved -/
theorem nonzero_even_from_second_step :
  ∀ n : ℕ, n ≥ 2 → contains_nonzero_even (evolved_sequence n) :=
sorry

end nonzero_even_from_second_step_l197_19730


namespace value_of_m_l197_19743

theorem value_of_m (m : ℝ) : 
  let f : ℝ → ℝ := λ x => x^2 - 2*x + m
  let g : ℝ → ℝ := λ x => x^2 - 2*x + 2*m + 8
  3 * f 5 = g 5 → m = -22 := by
sorry

end value_of_m_l197_19743


namespace start_with_any_digits_l197_19763

theorem start_with_any_digits :
  ∀ (A : ℕ), ∃ (n m : ℕ), 10^m * A ≤ 2^n ∧ 2^n < 10^m * (A + 1) :=
sorry

end start_with_any_digits_l197_19763


namespace jack_mopping_time_l197_19776

/-- Calculates the total time Jack spends mopping and resting given the room sizes and mopping speeds -/
def total_mopping_time (bathroom_size kitchen_size living_room_size : ℕ) 
                       (bathroom_speed kitchen_speed living_room_speed : ℕ) : ℕ :=
  let bathroom_time := (bathroom_size + bathroom_speed - 1) / bathroom_speed
  let kitchen_time := (kitchen_size + kitchen_speed - 1) / kitchen_speed
  let living_room_time := (living_room_size + living_room_speed - 1) / living_room_speed
  let mopping_time := bathroom_time + kitchen_time + living_room_time
  let break_time := 3 * 5 + (bathroom_size + kitchen_size + living_room_size) / 40
  mopping_time + break_time

theorem jack_mopping_time :
  total_mopping_time 24 80 120 8 10 7 = 49 := by
  sorry

end jack_mopping_time_l197_19776


namespace sufficient_not_necessary_condition_l197_19707

theorem sufficient_not_necessary_condition (a b : ℝ) :
  (b < -1 → |a| + |b| > 1) ∧ 
  ∃ a b : ℝ, |a| + |b| > 1 ∧ b ≥ -1 := by
  sorry

end sufficient_not_necessary_condition_l197_19707


namespace phone_bill_calculation_l197_19770

def initial_balance : ℚ := 800
def rent_payment : ℚ := 450
def paycheck_deposit : ℚ := 1500
def electricity_bill : ℚ := 117
def internet_bill : ℚ := 100
def final_balance : ℚ := 1563

theorem phone_bill_calculation : 
  initial_balance - rent_payment + paycheck_deposit - electricity_bill - internet_bill - final_balance = 70 := by
  sorry

end phone_bill_calculation_l197_19770


namespace inequality_proof_l197_19768

theorem inequality_proof (a b c d : ℝ) (h1 : 0 < a) (h2 : a ≤ b) (h3 : b ≤ c) (h4 : c ≤ d) :
  a^b * b^c * c^d * d^a ≥ b^a * c^b * d^c * a^d := by
  sorry

end inequality_proof_l197_19768


namespace math_team_combinations_l197_19798

theorem math_team_combinations : 
  let total_girls : ℕ := 4
  let total_boys : ℕ := 6
  let girls_on_team : ℕ := 3
  let boys_on_team : ℕ := 2
  (total_girls.choose girls_on_team) * (total_boys.choose boys_on_team) = 60 := by
sorry

end math_team_combinations_l197_19798


namespace definite_integral_exp_plus_2x_l197_19777

theorem definite_integral_exp_plus_2x : ∫ x in (0:ℝ)..1, (Real.exp x + 2 * x) = Real.exp 1 - 1 := by sorry

end definite_integral_exp_plus_2x_l197_19777


namespace total_interest_is_350_l197_19712

/-- Calculates the total interest for two loans over a given time period. -/
def totalInterest (principal1 : ℝ) (rate1 : ℝ) (principal2 : ℝ) (rate2 : ℝ) (time : ℝ) : ℝ :=
  principal1 * rate1 * time + principal2 * rate2 * time

/-- Theorem stating that the total interest for the given loans and time period is 350. -/
theorem total_interest_is_350 :
  totalInterest 800 0.03 1400 0.05 3.723404255319149 = 350 := by
  sorry

end total_interest_is_350_l197_19712


namespace median_to_mean_l197_19753

theorem median_to_mean (m : ℝ) : 
  let set := [m, m + 3, m + 7, m + 10, m + 12]
  m + 7 = 12 → 
  (set.sum / set.length : ℝ) = 11.4 := by
sorry

end median_to_mean_l197_19753


namespace quadratic_equation_roots_l197_19780

theorem quadratic_equation_roots (p q r : ℝ) (h : p ≠ 0 ∧ q ≠ r) :
  let f : ℝ → ℝ := λ x => p * (q - r) * x^2 + q * (r - p) * x + r * (p - q)
  (f (-1) = 0) → (f (-r * (p - q) / (p * (q - r))) = 0) :=
by sorry

end quadratic_equation_roots_l197_19780


namespace final_turtle_count_l197_19744

/-- Number of turtle statues on Grandma Molly's lawn after four years -/
def turtle_statues : ℕ :=
  let year1 := 4
  let year2 := year1 * 4
  let year3_before_breakage := year2 + 12
  let year3_after_breakage := year3_before_breakage - 3
  let year4_new_statues := 3 * 2
  year3_after_breakage + year4_new_statues

theorem final_turtle_count : turtle_statues = 31 := by
  sorry

end final_turtle_count_l197_19744


namespace odd_even_sum_difference_l197_19764

def sum_odd (n : ℕ) : ℕ := n^2

def sum_even (n : ℕ) : ℕ := n * (n + 1)

theorem odd_even_sum_difference : 
  let n_odd : ℕ := (2023 - 1) / 2 + 1
  let n_even : ℕ := (2022 - 2) / 2 + 1
  sum_odd n_odd - sum_even n_even + 7 - 8 = 47 := by
  sorry

end odd_even_sum_difference_l197_19764


namespace perimeter_semicircular_square_l197_19774

/-- The perimeter of a region bounded by semicircular arcs constructed on the sides of a square with side length 1/π is equal to 2. -/
theorem perimeter_semicircular_square : 
  let side_length : ℝ := 1 / Real.pi
  let semicircle_length : ℝ := Real.pi * side_length / 2
  let num_semicircles : ℕ := 4
  semicircle_length * num_semicircles = 2 := by sorry

end perimeter_semicircular_square_l197_19774


namespace movie_tickets_correct_l197_19726

/-- The number of movie tickets sold for the given estimation --/
def movie_tickets : ℕ := 6

/-- The price of a pack of grain crackers --/
def cracker_price : ℚ := 2.25

/-- The price of a bottle of beverage --/
def beverage_price : ℚ := 1.5

/-- The price of a chocolate bar --/
def chocolate_price : ℚ := 1

/-- The average amount of estimated snack sales per movie ticket --/
def avg_sales_per_ticket : ℚ := 2.79

/-- Theorem stating that the number of movie tickets sold is correct --/
theorem movie_tickets_correct : 
  (3 * cracker_price + 4 * beverage_price + 4 * chocolate_price) / avg_sales_per_ticket = movie_tickets :=
by sorry

end movie_tickets_correct_l197_19726


namespace T_greater_than_N_l197_19704

/-- Represents an 8x8 chessboard -/
def Board := Fin 8 → Fin 8 → Bool

/-- Represents a domino placement on the board -/
def DominoPlacement := List (Fin 8 × Fin 8 × Bool)

/-- Returns true if the given domino placement is valid on the board -/
def isValidPlacement (board : Board) (placement : DominoPlacement) : Prop :=
  sorry

/-- Counts the number of valid domino placements for a given number of dominoes -/
def countPlacements (n : Nat) : Nat :=
  sorry

/-- The number of ways to place 32 dominoes -/
def N : Nat := countPlacements 32

/-- The number of ways to place 24 dominoes -/
def T : Nat := countPlacements 24

/-- Theorem stating that T is greater than N -/
theorem T_greater_than_N : T > N := by
  sorry

end T_greater_than_N_l197_19704


namespace sum_of_divisors_143_l197_19742

def sum_of_divisors (n : ℕ) : ℕ := sorry

theorem sum_of_divisors_143 : sum_of_divisors 143 = 168 := by sorry

end sum_of_divisors_143_l197_19742


namespace total_amount_spent_l197_19701

def num_pens : ℕ := 30
def num_pencils : ℕ := 75
def avg_price_pencil : ℚ := 2
def avg_price_pen : ℚ := 16

theorem total_amount_spent : 
  num_pens * avg_price_pen + num_pencils * avg_price_pencil = 630 := by
  sorry

end total_amount_spent_l197_19701


namespace fathers_contribution_l197_19735

/-- Given the costs of items, savings, and lacking amount, calculate the father's contribution --/
theorem fathers_contribution 
  (mp3_cost cd_cost savings lacking : ℕ) 
  (h1 : mp3_cost = 120)
  (h2 : cd_cost = 19)
  (h3 : savings = 55)
  (h4 : lacking = 64) :
  mp3_cost + cd_cost = savings + lacking + 148 := by
  sorry

#check fathers_contribution

end fathers_contribution_l197_19735


namespace only_d_is_odd_l197_19717

theorem only_d_is_odd : ∀ n : ℤ,
  (n = 3 * 5 + 1 ∨ n = 2 * (3 + 5) ∨ n = 3 * (3 + 5) ∨ n = (3 + 5) / 2) → ¬(Odd n) ∧
  Odd (3 + 5 + 1) :=
by
  sorry

end only_d_is_odd_l197_19717


namespace distribute_4_2_l197_19786

/-- The number of ways to distribute n indistinguishable objects into k indistinguishable containers -/
def distribute (n : ℕ) (k : ℕ) : ℕ :=
  sorry

/-- Theorem: There are 3 ways to distribute 4 indistinguishable balls into 2 indistinguishable boxes -/
theorem distribute_4_2 : distribute 4 2 = 3 := by
  sorry

end distribute_4_2_l197_19786


namespace cube_surface_area_l197_19794

/-- Given a cube with side length x and distance d between non-intersecting diagonals
    of adjacent lateral faces, prove that its total surface area is 18d^2. -/
theorem cube_surface_area (d : ℝ) (h : d > 0) :
  let x := d * Real.sqrt 3
  6 * x^2 = 18 * d^2 :=
by sorry

end cube_surface_area_l197_19794


namespace factorial_divisibility_l197_19760

theorem factorial_divisibility (n : ℕ) (h : 1 ≤ n ∧ n ≤ 100) :
  ∃ k : ℕ, (Nat.factorial (n^3 - 1)) = k * (Nat.factorial n)^(n + 1) :=
sorry

end factorial_divisibility_l197_19760


namespace wall_bricks_count_l197_19716

/-- Represents the wall construction scenario --/
structure WallConstruction where
  /-- Total number of bricks in the wall --/
  total_bricks : ℕ
  /-- Time taken by the first bricklayer alone (in hours) --/
  time_worker1 : ℕ
  /-- Time taken by the second bricklayer alone (in hours) --/
  time_worker2 : ℕ
  /-- Reduction in combined output when working together (in bricks per hour) --/
  output_reduction : ℕ
  /-- Actual time taken to complete the wall (in hours) --/
  actual_time : ℕ

/-- Theorem stating the number of bricks in the wall --/
theorem wall_bricks_count (w : WallConstruction) 
  (h1 : w.time_worker1 = 8)
  (h2 : w.time_worker2 = 12)
  (h3 : w.output_reduction = 15)
  (h4 : w.actual_time = 6) :
  w.total_bricks = 360 :=
sorry

end wall_bricks_count_l197_19716


namespace fraction_zero_implies_x_neg_two_l197_19767

theorem fraction_zero_implies_x_neg_two (x : ℝ) :
  (|x| - 2) / (x^2 - x - 2) = 0 → x = -2 := by
  sorry

end fraction_zero_implies_x_neg_two_l197_19767


namespace gumball_probability_l197_19783

/-- Given a jar with blue and pink gumballs, if the probability of drawing
    two blue gumballs with replacement is 25/49, then the probability of
    drawing a pink gumball is 2/7. -/
theorem gumball_probability (p_blue p_pink : ℝ) : 
  p_blue + p_pink = 1 →
  p_blue^2 = 25/49 →
  p_pink = 2/7 := by
sorry

end gumball_probability_l197_19783


namespace range_of_m_l197_19758

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | |x - 2| ≤ 4}
def B (m : ℝ) : Set ℝ := {x : ℝ | (x - 1 - m) * (x - 1 + m) ≤ 0}

-- State the theorem
theorem range_of_m (m : ℝ) : 
  (m > 0 ∧ A ⊂ B m) → m ≥ 5 := by sorry

end range_of_m_l197_19758


namespace systematic_sampling_in_school_l197_19772

/-- Represents a sampling method -/
inductive SamplingMethod
  | StratifiedSampling
  | RandomDraw
  | RandomSampling
  | SystematicSampling

/-- Represents a school with classes and students -/
structure School where
  num_classes : Nat
  students_per_class : Nat
  student_numbers : Finset Nat

/-- Represents a sampling scenario -/
structure SamplingScenario where
  school : School
  selected_number : Nat

/-- Determines the sampling method used in a given scenario -/
def determineSamplingMethod (scenario : SamplingScenario) : SamplingMethod :=
  sorry

theorem systematic_sampling_in_school (scenario : SamplingScenario) :
  scenario.school.num_classes = 35 →
  scenario.school.students_per_class = 56 →
  scenario.school.student_numbers = Finset.range 56 →
  scenario.selected_number = 14 →
  determineSamplingMethod scenario = SamplingMethod.SystematicSampling :=
  sorry

end systematic_sampling_in_school_l197_19772
