import Mathlib

namespace y_coordinate_range_l4086_408616

/-- The parabola equation y^2 = x + 4 -/
def parabola (x y : ℝ) : Prop := y^2 = x + 4

/-- Point A is at (0,2) -/
def point_A : ℝ × ℝ := (0, 2)

/-- B is on the parabola -/
def B_on_parabola (B : ℝ × ℝ) : Prop := parabola B.1 B.2

/-- C is on the parabola -/
def C_on_parabola (C : ℝ × ℝ) : Prop := parabola C.1 C.2

/-- AB is perpendicular to BC -/
def AB_perp_BC (A B C : ℝ × ℝ) : Prop :=
  (B.2 - A.2) * (C.2 - B.2) = -(B.1 - A.1) * (C.1 - B.1)

/-- The main theorem -/
theorem y_coordinate_range (B C : ℝ × ℝ) :
  B_on_parabola B → C_on_parabola C → AB_perp_BC point_A B C →
  C.2 ≤ 0 ∨ C.2 ≥ 4 := by sorry

end y_coordinate_range_l4086_408616


namespace gcd_of_powers_of_two_l4086_408654

theorem gcd_of_powers_of_two : Nat.gcd (2^2016 - 1) (2^2008 - 1) = 2^8 - 1 := by
  sorry

end gcd_of_powers_of_two_l4086_408654


namespace jane_dolls_l4086_408606

theorem jane_dolls (total : ℕ) (difference : ℕ) : total = 32 → difference = 6 → ∃ jane : ℕ, jane = 13 ∧ jane + (jane + difference) = total := by
  sorry

end jane_dolls_l4086_408606


namespace equation_equivalence_product_l4086_408646

theorem equation_equivalence_product (a b x y : ℝ) (m n p q : ℤ) :
  (a^7 * x * y - a^6 * y - a^5 * x = a^3 * (b^4 - 1)) →
  ((a^m * x - a^n) * (a^p * y - a^q) = a^3 * b^4) →
  m * n * p * q = 4 := by
sorry

end equation_equivalence_product_l4086_408646


namespace f_has_inverse_when_x_geq_2_l4086_408655

def f (x : ℝ) : ℝ := x^2 - 4*x + 5

theorem f_has_inverse_when_x_geq_2 :
  ∀ (a b : ℝ), a ≥ 2 → b ≥ 2 → a ≠ b → f a ≠ f b :=
by
  sorry

end f_has_inverse_when_x_geq_2_l4086_408655


namespace second_char_lines_relation_l4086_408668

/-- Represents a character in a script with a certain number of lines. -/
structure Character where
  lines : ℕ

/-- Represents a script with three characters. -/
structure Script where
  char1 : Character
  char2 : Character
  char3 : Character
  first_has_more : char1.lines = char2.lines + 8
  third_has_two : char3.lines = 2
  first_has_twenty : char1.lines = 20

/-- The theorem stating the relationship between the lines of the second and third characters. -/
theorem second_char_lines_relation (script : Script) : 
  script.char2.lines = 3 * script.char3.lines + 6 := by
  sorry

end second_char_lines_relation_l4086_408668


namespace remainder_after_addition_l4086_408631

theorem remainder_after_addition (m : ℤ) (h : m % 5 = 2) : (m + 2535) % 5 = 2 := by
  sorry

end remainder_after_addition_l4086_408631


namespace five_letter_words_count_l4086_408650

/-- The number of vowels in the alphabet -/
def num_vowels : ℕ := 5

/-- The number of letters in the alphabet -/
def alphabet_size : ℕ := 26

/-- The number of five-letter words where the first and last letters are the same vowel,
    and the remaining three letters can be any letters from the alphabet -/
def num_words : ℕ := num_vowels * alphabet_size^3

theorem five_letter_words_count : num_words = 87880 := by
  sorry

end five_letter_words_count_l4086_408650


namespace solution_of_exponential_equation_l4086_408658

theorem solution_of_exponential_equation :
  {x : ℝ | (4 : ℝ) ^ (x^2 + 1) = 16} = {-1, 1} := by sorry

end solution_of_exponential_equation_l4086_408658


namespace cos_18_degrees_l4086_408660

theorem cos_18_degrees :
  Real.cos (18 * π / 180) = Real.sqrt ((5 + Real.sqrt 5) / 8) := by
  sorry

end cos_18_degrees_l4086_408660


namespace cistern_width_l4086_408682

/-- Given a cistern with the following properties:
  * length: 10 meters
  * water depth: 1.35 meters
  * total wet surface area: 103.2 square meters
  Prove that the width of the cistern is 6 meters. -/
theorem cistern_width (length : ℝ) (water_depth : ℝ) (wet_surface_area : ℝ) :
  length = 10 →
  water_depth = 1.35 →
  wet_surface_area = 103.2 →
  ∃ (width : ℝ), 
    wet_surface_area = length * width + 2 * length * water_depth + 2 * width * water_depth ∧
    width = 6 :=
by sorry

end cistern_width_l4086_408682


namespace regular_ngon_rotation_forms_regular_2ngon_l4086_408665

/-- A regular n-gon -/
structure RegularNGon (n : ℕ) (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V] :=
  (vertices : Fin n → V)
  (center : V)
  (is_regular : ∀ i j : Fin n, ‖vertices i - center‖ = ‖vertices j - center‖)

/-- Rotation of a vector about a point -/
def rotate (θ : ℝ) (center : V) (v : V) [NormedAddCommGroup V] [InnerProductSpace ℝ V] : V :=
  sorry

/-- The theorem statement -/
theorem regular_ngon_rotation_forms_regular_2ngon
  (n : ℕ) (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V]
  (ngon : RegularNGon n V) (θ : ℝ) :
  θ < 2 * Real.pi / n →
  (∃ (m : ℕ), θ = 2 * Real.pi / m) →
  ∃ (circle_center : V) (radius : ℝ),
    ∀ (i : Fin n),
      ‖circle_center - ngon.vertices i‖ = radius ∧
      ‖circle_center - rotate θ ngon.center (ngon.vertices i)‖ = radius :=
sorry

end regular_ngon_rotation_forms_regular_2ngon_l4086_408665


namespace first_player_wins_98_max_n_first_player_wins_l4086_408675

/-- Represents the game board -/
def Board := Fin 1000 → Bool

/-- Represents a player's move -/
inductive Move
| Place (pos : Fin 1000) (num : Nat)
| Remove (start : Fin 1000) (len : Nat)

/-- Represents a player's strategy -/
def Strategy := Board → Move

/-- Applies a move to the board -/
def applyMove (b : Board) (m : Move) : Board :=
  sorry

/-- Checks if all tokens are placed in a row without gaps -/
def isWinningState (b : Board) : Prop :=
  sorry

/-- The game's rules and win condition -/
def gameRules (n : Nat) (s1 s2 : Strategy) : Prop :=
  sorry

/-- Theorem: First player can always win for n = 98 -/
theorem first_player_wins_98 :
  ∃ (s1 : Strategy), ∀ (s2 : Strategy), gameRules 98 s1 s2 → isWinningState (sorry : Board) :=
  sorry

/-- Theorem: 98 is the maximum n for which first player can always win -/
theorem max_n_first_player_wins :
  (∃ (s1 : Strategy), ∀ (s2 : Strategy), gameRules 98 s1 s2 → isWinningState (sorry : Board)) ∧
  (∀ n > 98, ∃ (s2 : Strategy), ∀ (s1 : Strategy), ¬(gameRules n s1 s2 → isWinningState (sorry : Board))) :=
  sorry

end first_player_wins_98_max_n_first_player_wins_l4086_408675


namespace jamie_quiz_performance_l4086_408680

theorem jamie_quiz_performance (y : ℕ) : 
  let total_questions : ℕ := 8 * y
  let missed_questions : ℕ := 2 * y
  let correct_questions : ℕ := total_questions - missed_questions
  (correct_questions : ℚ) / (total_questions : ℚ) * 100 = 75 := by
sorry

end jamie_quiz_performance_l4086_408680


namespace probability_information_both_clubs_l4086_408677

def total_students : ℕ := 30
def art_club_students : ℕ := 22
def music_club_students : ℕ := 25

def probability_both_clubs : ℚ := 397 / 435

theorem probability_information_both_clubs :
  let students_in_both := art_club_students + music_club_students - total_students
  let students_only_art := art_club_students - students_in_both
  let students_only_music := music_club_students - students_in_both
  let total_combinations := total_students.choose 2
  let incompatible_combinations := students_only_art.choose 2 + students_only_music.choose 2
  (1 : ℚ) - (incompatible_combinations : ℚ) / total_combinations = probability_both_clubs :=
by sorry

end probability_information_both_clubs_l4086_408677


namespace kendra_suvs_count_l4086_408635

/-- The number of SUVs Kendra saw in the afternoon -/
def afternoon_suvs : ℕ := 10

/-- The number of SUVs Kendra saw in the evening -/
def evening_suvs : ℕ := 5

/-- The total number of SUVs Kendra saw during her road trip -/
def total_suvs : ℕ := afternoon_suvs + evening_suvs

theorem kendra_suvs_count : total_suvs = 15 := by
  sorry

end kendra_suvs_count_l4086_408635


namespace window_purchase_savings_l4086_408617

def window_price : ℕ := 150
def alice_windows : ℕ := 9
def bob_windows : ℕ := 10

def discount (n : ℕ) : ℕ :=
  (n / 6) * window_price

def cost (n : ℕ) : ℕ :=
  n * window_price - discount n

def total_separate_cost : ℕ :=
  cost alice_windows + cost bob_windows

def joint_windows : ℕ :=
  alice_windows + bob_windows

def joint_cost : ℕ :=
  cost joint_windows

def savings : ℕ :=
  total_separate_cost - joint_cost

theorem window_purchase_savings :
  savings = 150 := by sorry

end window_purchase_savings_l4086_408617


namespace rectangle_side_length_l4086_408612

/-- Given two rectangles A and B, where A has sides of length 3 and 6,
    and the ratio of corresponding sides of A to B is 3/4,
    prove that the length of side c in Rectangle B is 4. -/
theorem rectangle_side_length (a b c d : ℝ) : 
  a = 3 → b = 6 → a / c = 3 / 4 → b / d = 3 / 4 → c = 4 := by
  sorry

end rectangle_side_length_l4086_408612


namespace inverse_proposition_absolute_values_l4086_408645

theorem inverse_proposition_absolute_values (a b : ℝ) :
  (∀ x y : ℝ, x = y → |x| = |y|) →
  (∀ x y : ℝ, |x| = |y| → x = y) :=
by
  sorry

end inverse_proposition_absolute_values_l4086_408645


namespace economics_test_correct_answers_l4086_408659

theorem economics_test_correct_answers 
  (total_students : ℕ) 
  (correct_q1 : ℕ) 
  (correct_q2 : ℕ) 
  (not_taken : ℕ) 
  (h1 : total_students = 25) 
  (h2 : correct_q1 = 22) 
  (h3 : correct_q2 = 20) 
  (h4 : not_taken = 3) :
  (correct_q1 + correct_q2) - (total_students - not_taken) = 20 := by
sorry

end economics_test_correct_answers_l4086_408659


namespace additional_earnings_calculation_l4086_408697

/-- Represents the financial data for a company's quarterly earnings and dividends. -/
structure CompanyFinancials where
  expectedEarnings : ℝ
  actualEarnings : ℝ
  additionalDividendRate : ℝ

/-- Calculates the additional earnings per share based on the company's financial data. -/
def additionalEarnings (cf : CompanyFinancials) : ℝ :=
  cf.actualEarnings - cf.expectedEarnings

/-- Theorem stating that the additional earnings per share is the difference between
    actual and expected earnings. -/
theorem additional_earnings_calculation (cf : CompanyFinancials) 
    (h1 : cf.expectedEarnings = 0.80)
    (h2 : cf.actualEarnings = 1.10)
    (h3 : cf.additionalDividendRate = 0.04) :
    additionalEarnings cf = 0.30 := by
  sorry

end additional_earnings_calculation_l4086_408697


namespace fruit_supply_theorem_l4086_408626

/-- Represents the weekly fruit requirements for a bakery -/
structure BakeryRequirement where
  strawberries : ℕ
  blueberries : ℕ
  raspberries : ℕ

/-- Calculates the total number of sacks needed for a given fruit over 10 weeks -/
def totalSacksFor10Weeks (weeklyRequirements : List BakeryRequirement) (getFruit : BakeryRequirement → ℕ) : ℕ :=
  10 * (weeklyRequirements.map getFruit).sum

/-- The list of weekly requirements for all bakeries -/
def allBakeries : List BakeryRequirement := [
  ⟨2, 3, 5⟩,
  ⟨4, 2, 8⟩,
  ⟨12, 10, 7⟩,
  ⟨8, 4, 3⟩,
  ⟨15, 6, 12⟩,
  ⟨5, 9, 11⟩
]

theorem fruit_supply_theorem :
  totalSacksFor10Weeks allBakeries (·.strawberries) = 460 ∧
  totalSacksFor10Weeks allBakeries (·.blueberries) = 340 ∧
  totalSacksFor10Weeks allBakeries (·.raspberries) = 460 := by
  sorry

end fruit_supply_theorem_l4086_408626


namespace square_of_1009_l4086_408623

theorem square_of_1009 : 1009 * 1009 = 1018081 := by
  sorry

end square_of_1009_l4086_408623


namespace expression_evaluation_l4086_408624

theorem expression_evaluation : (3 * 5 * 6) * (1/3 + 1/5 + 1/6) = 63 := by
  sorry

end expression_evaluation_l4086_408624


namespace stock_quoted_value_l4086_408634

/-- Proves that given an investment of 1620 in an 8% stock that earns 135, the stock is quoted at 96 --/
theorem stock_quoted_value (investment : ℝ) (dividend_rate : ℝ) (dividend_earned : ℝ) 
  (h1 : investment = 1620)
  (h2 : dividend_rate = 8 / 100)
  (h3 : dividend_earned = 135) :
  (investment / ((dividend_earned * 100) / dividend_rate)) * 100 = 96 := by
  sorry

end stock_quoted_value_l4086_408634


namespace two_numbers_with_given_means_l4086_408611

theorem two_numbers_with_given_means : ∃ a b : ℝ, 
  a > 0 ∧ b > 0 ∧ 
  Real.sqrt (a * b) = Real.sqrt 5 ∧
  2 / (1/a + 1/b) = 5/3 ∧
  a = (15 + Real.sqrt 145) / 4 ∧
  b = (15 - Real.sqrt 145) / 4 := by
  sorry

end two_numbers_with_given_means_l4086_408611


namespace triangle_point_coordinates_l4086_408696

/-- Given a triangle ABC with the following properties:
  - A has coordinates (2, 8)
  - M has coordinates (4, 11) and is the midpoint of AB
  - L has coordinates (6, 6) and BL is the angle bisector of angle ABC
  Prove that the coordinates of point C are (6, 14) -/
theorem triangle_point_coordinates (A B C M L : ℝ × ℝ) : 
  A = (2, 8) →
  M = (4, 11) →
  L = (6, 6) →
  M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) →  -- M is midpoint of AB
  (L.1 - B.1) * (C.2 - B.2) = (L.2 - B.2) * (C.1 - B.1) →  -- BL is angle bisector
  C = (6, 14) := by sorry

end triangle_point_coordinates_l4086_408696


namespace complex_magnitude_l4086_408622

theorem complex_magnitude (z : ℂ) : z = -2 + I → Complex.abs (z + 1) = Real.sqrt 2 := by
  sorry

end complex_magnitude_l4086_408622


namespace new_rectangle_area_comparison_l4086_408630

theorem new_rectangle_area_comparison (a b : ℝ) (h : 0 < a ∧ a < b) :
  let new_base := 2 * a * b
  let new_height := (a * Real.sqrt (a^2 + b^2)) / 2
  let new_area := new_base * new_height
  let circle_area := Real.pi * b^2
  new_area = a^2 * b * Real.sqrt (a^2 + b^2) ∧ 
  ∃ (a b : ℝ), new_area ≠ circle_area :=
by sorry

end new_rectangle_area_comparison_l4086_408630


namespace paco_cookies_l4086_408614

theorem paco_cookies (initial_cookies : ℕ) (eaten_cookies : ℕ) (given_cookies : ℕ) 
  (h1 : initial_cookies = 17)
  (h2 : eaten_cookies = 14)
  (h3 : eaten_cookies + given_cookies ≤ initial_cookies) :
  given_cookies = 3 := by
  sorry

end paco_cookies_l4086_408614


namespace cube_sum_and_reciprocal_l4086_408621

theorem cube_sum_and_reciprocal (x : ℝ) (h : x + 1/x = 7) : x^3 + 1/x^3 = 322 := by
  sorry

end cube_sum_and_reciprocal_l4086_408621


namespace system_solution_l4086_408605

theorem system_solution :
  let eq1 := (fun (x y : ℝ) ↦ x^2 + y^2 + 6*x*y = 68)
  let eq2 := (fun (x y : ℝ) ↦ 2*x^2 + 2*y^2 - 3*x*y = 16)
  (∀ x y, eq1 x y ∧ eq2 x y ↔ 
    ((x = 4 ∧ y = 2) ∨ (x = 2 ∧ y = 4) ∨ 
     (x = -4 ∧ y = -2) ∨ (x = -2 ∧ y = -4))) :=
by sorry

end system_solution_l4086_408605


namespace range_of_a_l4086_408607

open Real

theorem range_of_a (a : ℝ) :
  (∀ x ∈ Set.Icc 1 2, |2^x - a| < |5 - 2^x|) →
  3 < a ∧ a < 5 := by
  sorry

end range_of_a_l4086_408607


namespace negation_equivalence_l4086_408699

theorem negation_equivalence : 
  (¬ ∃ x : ℝ, x^2 + x + 1 < 0) ↔ (∀ x : ℝ, x^2 + x + 1 ≥ 0) :=
by sorry

end negation_equivalence_l4086_408699


namespace magnitude_of_z_l4086_408674

/-- The magnitude of the complex number z = (1-i)/i is √2 -/
theorem magnitude_of_z (i : ℂ) (h : i^2 = -1) : 
  Complex.abs ((1 - i) / i) = Real.sqrt 2 := by
  sorry

end magnitude_of_z_l4086_408674


namespace arithmetic_sequence_a12_l4086_408686

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_a12 (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a)
  (h_sum : a 7 + a 9 = 16)
  (h_a4 : a 4 = 1) :
  a 12 = 15 := by
  sorry

end arithmetic_sequence_a12_l4086_408686


namespace buffy_breath_holding_time_l4086_408625

/-- Represents the breath-holding times of Kelly, Brittany, and Buffy in seconds -/
structure BreathHoldingTimes where
  kelly : ℕ
  brittany : ℕ
  buffy : ℕ

/-- The breath-holding contest results -/
def contest : BreathHoldingTimes :=
  { kelly := 3 * 60,  -- Kelly's time in seconds
    brittany := 3 * 60 - 20,  -- Brittany's time is 20 seconds less than Kelly's
    buffy := (3 * 60 - 20) - 40  -- Buffy's time is 40 seconds less than Brittany's
  }

/-- Theorem stating that Buffy held her breath for 120 seconds -/
theorem buffy_breath_holding_time :
  contest.buffy = 120 := by
  sorry

end buffy_breath_holding_time_l4086_408625


namespace shoes_sold_l4086_408627

theorem shoes_sold (shoes sandals : ℕ) 
  (ratio : shoes / sandals = 9 / 5)
  (sandals_count : sandals = 40) : 
  shoes = 72 := by
  sorry

end shoes_sold_l4086_408627


namespace polynomial_remainder_theorem_l4086_408689

theorem polynomial_remainder_theorem (p : ℝ → ℝ) (hp1 : p 1 = 5) (hp3 : p 3 = 8) :
  ∃ (t : ℝ), ∃ (q : ℝ → ℝ), 
    ∀ x, p x = q x * ((x - 1) * (x - 3) * (x - 5)) + 
              (t * x^2 + (3 - 8*t)/2 * x + (7 + 6*t)/2) :=
by sorry

end polynomial_remainder_theorem_l4086_408689


namespace unique_mod_equivalence_l4086_408648

theorem unique_mod_equivalence :
  ∃! n : ℕ, 0 ≤ n ∧ n ≤ 9 ∧ n ≡ -2023 [ZMOD 10] := by
  sorry

end unique_mod_equivalence_l4086_408648


namespace average_visitors_theorem_l4086_408610

/-- Calculates the average number of visitors per day in a 30-day month starting on a Sunday -/
def averageVisitorsPerDay (sundayVisitors : ℕ) (otherDayVisitors : ℕ) : ℚ :=
  let numSundays := 4
  let numOtherDays := 30 - numSundays
  let totalVisitors := numSundays * sundayVisitors + numOtherDays * otherDayVisitors
  totalVisitors / 30

/-- Theorem stating that the average number of visitors per day is 292 -/
theorem average_visitors_theorem (sundayVisitors : ℕ) (otherDayVisitors : ℕ) 
  (h1 : sundayVisitors = 630) (h2 : otherDayVisitors = 240) : 
  averageVisitorsPerDay sundayVisitors otherDayVisitors = 292 := by
  sorry

#eval averageVisitorsPerDay 630 240

end average_visitors_theorem_l4086_408610


namespace product_abcd_is_zero_l4086_408676

theorem product_abcd_is_zero
  (a b c d : ℤ)
  (eq1 : 3*a + 2*b + 4*c + 8*d = 40)
  (eq2 : 4*(d+c) = b)
  (eq3 : 2*b + 2*c = a)
  (eq4 : c + 1 = d) :
  a * b * c * d = 0 := by
  sorry

end product_abcd_is_zero_l4086_408676


namespace sixth_year_fee_l4086_408685

def membership_fee (initial_fee : ℕ) (annual_increase : ℕ) (year : ℕ) : ℕ :=
  initial_fee + (year - 1) * annual_increase

theorem sixth_year_fee :
  membership_fee 80 10 6 = 130 := by
  sorry

end sixth_year_fee_l4086_408685


namespace lives_per_player_l4086_408608

theorem lives_per_player (initial_players : ℕ) (players_quit : ℕ) (total_lives : ℕ) : 
  initial_players = 8 → players_quit = 3 → total_lives = 15 → 
  (total_lives / (initial_players - players_quit) = 3) := by
  sorry

end lives_per_player_l4086_408608


namespace right_triangle_hypotenuse_l4086_408600

/-- Given a right triangle with legs a and b, if rotating about leg a produces a cone
    of volume 500π cm³ and rotating about leg b produces a cone of volume 1800π cm³,
    then the length of the hypotenuse is approximately 24.46 cm. -/
theorem right_triangle_hypotenuse (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (1/3 * π * a * b^2 = 500 * π) →
  (1/3 * π * b * a^2 = 1800 * π) →
  abs ((a^2 + b^2).sqrt - 24.46) < 0.01 := by
sorry

end right_triangle_hypotenuse_l4086_408600


namespace quadrilateral_area_is_84_l4086_408644

/-- Represents a quadrilateral ABCD -/
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

/-- The area of a quadrilateral -/
def area (q : Quadrilateral) : ℝ := sorry

/-- The length of a side of a quadrilateral -/
def side_length (q : Quadrilateral) (side : Fin 4) : ℝ := sorry

/-- The measure of an angle in a quadrilateral -/
def angle_measure (q : Quadrilateral) (vertex : Fin 4) : ℝ := sorry

/-- Whether a quadrilateral is convex -/
def is_convex (q : Quadrilateral) : Prop := sorry

theorem quadrilateral_area_is_84 (q : Quadrilateral) 
  (h_convex : is_convex q)
  (h_ab : side_length q 0 = 5)
  (h_bc : side_length q 1 = 12)
  (h_cd : side_length q 2 = 13)
  (h_ad : side_length q 3 = 15)
  (h_angle_abc : angle_measure q 1 = 90) :
  area q = 84 := by sorry

end quadrilateral_area_is_84_l4086_408644


namespace triangle_median_theorem_l4086_408637

-- Define the triangle and its medians
structure Triangle :=
  (D E F : ℝ × ℝ)
  (DP EQ : ℝ × ℝ)

-- Define the properties of the triangle
def is_valid_triangle (t : Triangle) : Prop :=
  -- DP and EQ are medians
  ∃ P Q : ℝ × ℝ,
    t.DP = P - t.D ∧
    t.EQ = Q - t.E ∧
    P = (t.E + t.F) / 2 ∧
    Q = (t.D + t.F) / 2 ∧
  -- DP and EQ are perpendicular
  t.DP.1 * t.EQ.1 + t.DP.2 * t.EQ.2 = 0 ∧
  -- Lengths of DP and EQ
  Real.sqrt (t.DP.1^2 + t.DP.2^2) = 18 ∧
  Real.sqrt (t.EQ.1^2 + t.EQ.2^2) = 24

-- Theorem statement
theorem triangle_median_theorem (t : Triangle) (h : is_valid_triangle t) :
  Real.sqrt ((t.F.1 - t.D.1)^2 + (t.F.2 - t.D.2)^2) = 8 * Real.sqrt 13 :=
sorry

end triangle_median_theorem_l4086_408637


namespace cubic_system_solution_l4086_408684

theorem cubic_system_solution (a b c : ℝ) : 
  a + b + c = 3 ∧ 
  a^2 + b^2 + c^2 = 35 ∧ 
  a^3 + b^3 + c^3 = 99 → 
  ({a, b, c} : Set ℝ) = {1, -3, 5} :=
sorry

end cubic_system_solution_l4086_408684


namespace cucumbers_for_apples_l4086_408640

-- Define the cost relationships
def apple_banana_ratio : ℚ := 10 / 5
def banana_cucumber_ratio : ℚ := 3 / 4

-- Define the number of apples we're interested in
def apples_of_interest : ℚ := 20

-- Theorem to prove
theorem cucumbers_for_apples :
  let bananas_for_apples : ℚ := apples_of_interest / apple_banana_ratio
  let cucumbers_for_bananas : ℚ := bananas_for_apples * (1 / banana_cucumber_ratio)
  cucumbers_for_bananas = 40 / 3 :=
by sorry

end cucumbers_for_apples_l4086_408640


namespace larger_number_proof_l4086_408638

/-- Given two positive integers with HCF 23 and LCM factors 13 and 19, prove the larger is 437 -/
theorem larger_number_proof (a b : ℕ+) 
  (hcf : Nat.gcd a b = 23)
  (lcm : Nat.lcm a b = 23 * 13 * 19) :
  max a b = 437 := by
  sorry

end larger_number_proof_l4086_408638


namespace parallel_intersections_l4086_408613

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the parallel relation for planes
variable (parallel_planes : Plane → Plane → Prop)

-- Define the intersection relation for planes and lines
variable (intersect : Plane → Plane → Line → Prop)

-- Define the parallel relation for lines
variable (parallel_lines : Line → Line → Prop)

-- State the theorem
theorem parallel_intersections
  (α β γ : Plane) (m n : Line)
  (h1 : parallel_planes α β)
  (h2 : intersect α γ m)
  (h3 : intersect β γ n) :
  parallel_lines m n :=
sorry

end parallel_intersections_l4086_408613


namespace geometric_sum_n1_l4086_408666

theorem geometric_sum_n1 (a : ℝ) (h : a ≠ 1) :
  1 + a + a^2 + a^3 = (1 - a^4) / (1 - a) := by
  sorry

end geometric_sum_n1_l4086_408666


namespace hyperbola_circle_intersection_l4086_408618

/-- The intersection points of a hyperbola and a circle -/
theorem hyperbola_circle_intersection :
  ∀ x y : ℝ, x^2 - 9*y^2 = 36 ∧ x^2 + y^2 = 36 → (x = 6 ∧ y = 0) ∨ (x = -6 ∧ y = 0) :=
by sorry

end hyperbola_circle_intersection_l4086_408618


namespace cleaning_assignment_cases_l4086_408656

def number_of_people : ℕ := 6
def people_for_floor : ℕ := 2
def people_for_window : ℕ := 1

theorem cleaning_assignment_cases :
  (Nat.choose (number_of_people - 1) (people_for_floor - 1)) *
  (Nat.choose (number_of_people - people_for_floor) people_for_window) = 12 := by
  sorry

end cleaning_assignment_cases_l4086_408656


namespace parabola_vertex_l4086_408693

/-- The equation of a parabola in the form 2y^2 + 8y + 3x + 7 = 0 -/
def parabola_equation (x y : ℝ) : Prop :=
  2 * y^2 + 8 * y + 3 * x + 7 = 0

/-- The vertex of a parabola -/
def is_vertex (x y : ℝ) (eq : ℝ → ℝ → Prop) : Prop :=
  eq x y ∧ ∀ x' y', eq x' y' → y ≤ y'

theorem parabola_vertex :
  is_vertex (1/3) (-2) parabola_equation := by
  sorry

end parabola_vertex_l4086_408693


namespace total_movies_count_l4086_408673

/-- The number of times Timothy and Theresa went to the movies in 2009 and 2010 -/
def total_movies (timothy_2009 timothy_2010 theresa_2009 theresa_2010 : ℕ) : ℕ :=
  timothy_2009 + timothy_2010 + theresa_2009 + theresa_2010

/-- Theorem stating the total number of movies Timothy and Theresa saw in 2009 and 2010 -/
theorem total_movies_count : 
  ∀ (timothy_2009 timothy_2010 theresa_2009 theresa_2010 : ℕ),
    timothy_2009 = 24 →
    timothy_2010 = timothy_2009 + 7 →
    theresa_2009 = timothy_2009 / 2 →
    theresa_2010 = 2 * timothy_2010 →
    total_movies timothy_2009 timothy_2010 theresa_2009 theresa_2010 = 129 := by
  sorry

#check total_movies_count

end total_movies_count_l4086_408673


namespace min_inequality_solution_set_l4086_408629

open Set Real

theorem min_inequality_solution_set (x : ℝ) (hx : x ≠ 0) :
  min 4 (x + 4 / x) ≥ 8 * min x (1 / x) ↔ x ∈ Iic 0 ∪ Ioo 0 (1 / 2) ∪ Ici 2 :=
sorry

end min_inequality_solution_set_l4086_408629


namespace stadium_entry_fee_l4086_408651

/-- Proves that the entry fee per person is $20 given the stadium conditions --/
theorem stadium_entry_fee (capacity : ℕ) (occupancy_ratio : ℚ) (fee_difference : ℕ) :
  capacity = 2000 →
  occupancy_ratio = 3/4 →
  fee_difference = 10000 →
  ∃ (fee : ℚ), fee = 20 ∧
    (capacity : ℚ) * fee - (capacity : ℚ) * occupancy_ratio * fee = fee_difference :=
by sorry

end stadium_entry_fee_l4086_408651


namespace probability_both_selected_l4086_408698

theorem probability_both_selected (prob_ram : ℚ) (prob_ravi : ℚ) 
  (h1 : prob_ram = 3/7) (h2 : prob_ravi = 1/5) : 
  prob_ram * prob_ravi = 3/35 := by
  sorry

end probability_both_selected_l4086_408698


namespace two_cars_meeting_time_l4086_408609

/-- Two cars traveling between cities problem -/
theorem two_cars_meeting_time 
  (distance : ℝ) 
  (speed1 speed2 : ℝ) 
  (h1 : distance = 450) 
  (h2 : speed1 = 45) 
  (h3 : speed2 = 30) :
  (2 * distance) / (speed1 + speed2) = 12 := by
sorry

end two_cars_meeting_time_l4086_408609


namespace quadratic_inequality_and_hyperbola_l4086_408662

-- Define the solution set condition
def solution_set (a b : ℝ) : Prop :=
  ∀ x : ℝ, (a * x^2 - 3*x + 2 > 0) ↔ (x < 1 ∨ x > b)

-- Define the main theorem
theorem quadratic_inequality_and_hyperbola (a b : ℝ) :
  solution_set a b →
  (∀ x y : ℝ, x > 0 → y > 0 → a/x + b/y = 1 →
    (∀ k : ℝ, (∀ x y : ℝ, x > 0 → y > 0 → a/x + b/y = 1 → 2*x + y ≥ k) → k ≤ 8)) →
  a = 1 ∧ b = 2 := by
sorry


end quadratic_inequality_and_hyperbola_l4086_408662


namespace isosceles_triangle_perimeter_l4086_408642

-- Define the side lengths of the isosceles triangle
def side_a : ℝ := 9
def side_b : ℝ := sorry  -- This will be either 3 or 5

-- Define the equation for side_b
axiom side_b_equation : side_b^2 - 8*side_b + 15 = 0

-- Define the perimeter of the triangle
def perimeter : ℝ := 2*side_a + side_b

-- Theorem statement
theorem isosceles_triangle_perimeter :
  perimeter = 19 ∨ perimeter = 21 ∨ perimeter = 23 :=
sorry

end isosceles_triangle_perimeter_l4086_408642


namespace final_price_after_two_reductions_l4086_408688

/-- Given an original price and two identical percentage reductions, 
    calculate the final price after the reductions. -/
def final_price (original_price : ℝ) (reduction_percentage : ℝ) : ℝ :=
  original_price * (1 - reduction_percentage)^2

/-- Theorem stating that for a product with original price $100 and 
    two reductions of percentage m, the final price is 100(1-m)^2 -/
theorem final_price_after_two_reductions (m : ℝ) :
  final_price 100 m = 100 * (1 - m)^2 := by
  sorry

end final_price_after_two_reductions_l4086_408688


namespace ways_to_buy_three_items_l4086_408669

/-- Represents the inventory of a store selling computer peripherals -/
structure StoreInventory where
  headphones : ℕ
  mice : ℕ
  keyboards : ℕ
  keyboard_mouse_sets : ℕ
  headphone_mouse_sets : ℕ

/-- Calculates the number of ways to buy a headphone, a keyboard, and a mouse -/
def waysToButThreeItems (inventory : StoreInventory) : ℕ :=
  inventory.headphones * inventory.mice * inventory.keyboards +
  inventory.keyboard_mouse_sets * inventory.headphones +
  inventory.headphone_mouse_sets * inventory.keyboards

/-- The store's actual inventory -/
def actualInventory : StoreInventory := {
  headphones := 9
  mice := 13
  keyboards := 5
  keyboard_mouse_sets := 4
  headphone_mouse_sets := 5
}

theorem ways_to_buy_three_items :
  waysToButThreeItems actualInventory = 646 := by
  sorry

end ways_to_buy_three_items_l4086_408669


namespace set_intersection_problem_l4086_408670

theorem set_intersection_problem (a : ℝ) : 
  let A : Set ℝ := {-4, 2*a-1, a^2}
  let B : Set ℝ := {a-5, 1-a, 9}
  9 ∈ (A ∩ B) → a = 5 ∨ a = -3 :=
by
  sorry


end set_intersection_problem_l4086_408670


namespace apple_difference_l4086_408661

def apple_contest (aaron bella claire daniel edward fiona george hannah : ℕ) : Prop :=
  aaron = 5 ∧ bella = 3 ∧ claire = 7 ∧ daniel = 2 ∧ edward = 4 ∧ fiona = 3 ∧ george = 1 ∧ hannah = 6 ∧
  claire ≥ aaron ∧ claire ≥ bella ∧ claire ≥ daniel ∧ claire ≥ edward ∧ claire ≥ fiona ∧ claire ≥ george ∧ claire ≥ hannah ∧
  aaron ≥ bella ∧ aaron ≥ daniel ∧ aaron ≥ edward ∧ aaron ≥ fiona ∧ aaron ≥ george ∧ aaron ≥ hannah ∧
  george ≤ aaron ∧ george ≤ bella ∧ george ≤ claire ∧ george ≤ daniel ∧ george ≤ edward ∧ george ≤ fiona ∧ george ≤ hannah

theorem apple_difference (aaron bella claire daniel edward fiona george hannah : ℕ) :
  apple_contest aaron bella claire daniel edward fiona george hannah →
  claire - george = 6 := by
  sorry

end apple_difference_l4086_408661


namespace next_multiple_remainder_l4086_408679

theorem next_multiple_remainder (N : ℕ) (h : N = 44 * 432) :
  (N + 432) % 39 = 12 := by
  sorry

end next_multiple_remainder_l4086_408679


namespace algebraic_expression_value_l4086_408601

theorem algebraic_expression_value (x y : ℝ) (h : x - y - 7 = 0) :
  x^2 - y^2 - 14*y = 49 := by
  sorry

end algebraic_expression_value_l4086_408601


namespace teachers_liking_beverages_l4086_408641

theorem teachers_liking_beverages 
  (total : ℕ) 
  (tea : ℕ) 
  (coffee : ℕ) 
  (h1 : total = 90)
  (h2 : tea = 66)
  (h3 : coffee = 42)
  (h4 : ∃ (both neither : ℕ), both = 3 * neither ∧ tea + coffee - both + neither = total) :
  ∃ (at_least_one : ℕ), at_least_one = 81 ∧ at_least_one = tea + coffee - (tea + coffee - total + (total - tea - coffee) / 2) :=
by sorry

end teachers_liking_beverages_l4086_408641


namespace yogurt_combinations_l4086_408632

def num_flavors : ℕ := 5
def num_toppings : ℕ := 8

def combinations_with_no_topping : ℕ := 1
def combinations_with_one_topping (n : ℕ) : ℕ := n
def combinations_with_two_toppings (n : ℕ) : ℕ := n * (n - 1) / 2

def total_topping_combinations (n : ℕ) : ℕ :=
  combinations_with_no_topping + 
  combinations_with_one_topping n + 
  combinations_with_two_toppings n

theorem yogurt_combinations : 
  num_flavors * total_topping_combinations num_toppings = 185 := by
  sorry

end yogurt_combinations_l4086_408632


namespace max_value_sum_of_roots_l4086_408694

theorem max_value_sum_of_roots (a b c : ℝ) : 
  0 ≤ a → 0 ≤ b → 0 ≤ c → a + b + c = 7 → 
  Real.sqrt (3 * a + 2) + Real.sqrt (3 * b + 2) + Real.sqrt (3 * c + 2) ≤ 3 * Real.sqrt 23 ∧
  ∃ a b c, 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ a + b + c = 7 ∧
    Real.sqrt (3 * a + 2) + Real.sqrt (3 * b + 2) + Real.sqrt (3 * c + 2) = 3 * Real.sqrt 23 :=
by sorry

end max_value_sum_of_roots_l4086_408694


namespace man_son_age_difference_l4086_408653

/-- Given a man and his son, proves that the man is 20 years older than his son. -/
theorem man_son_age_difference (man_age son_age : ℕ) : 
  son_age = 18 →
  man_age + 2 = 2 * (son_age + 2) →
  man_age - son_age = 20 := by
  sorry

end man_son_age_difference_l4086_408653


namespace function_value_at_four_l4086_408603

/-- Given a function f: ℝ → ℝ satisfying f(x) + 2f(1 - x) = 3x^2 for all x,
    prove that f(4) = 2 -/
theorem function_value_at_four (f : ℝ → ℝ) 
    (h : ∀ x : ℝ, f x + 2 * f (1 - x) = 3 * x^2) : 
    f 4 = 2 := by
  sorry

end function_value_at_four_l4086_408603


namespace softball_players_l4086_408691

/-- The number of softball players in a games hour -/
theorem softball_players (cricket hockey football total : ℕ) 
  (h1 : cricket = 10)
  (h2 : hockey = 12)
  (h3 : football = 16)
  (h4 : total = 51)
  (h5 : total = cricket + hockey + football + softball) : 
  softball = 13 := by
  sorry

end softball_players_l4086_408691


namespace sequence_general_term_l4086_408639

theorem sequence_general_term (a : ℕ → ℕ) :
  a 1 = 1 ∧
  (∀ n : ℕ, n ≥ 1 → a (n + 1) - 2 * a n = 2^n) →
  ∀ n : ℕ, n ≥ 1 → a n = n * 2^(n - 1) :=
by sorry

end sequence_general_term_l4086_408639


namespace binomial_expansion_and_specific_case_l4086_408619

theorem binomial_expansion_and_specific_case :
  ∀ (a b : ℝ),
    (a + b)^4 = a^4 + 4*a^3*b + 6*a^2*b^2 + 4*a*b^3 + b^4 ∧
    (2 - 1/3)^4 = 625/81 :=
by sorry

end binomial_expansion_and_specific_case_l4086_408619


namespace kate_stickers_l4086_408672

/-- Given that the ratio of Kate's stickers to Jenna's stickers is 7:4 and Jenna has 12 stickers,
    prove that Kate has 21 stickers. -/
theorem kate_stickers (kate_stickers : ℕ) (jenna_stickers : ℕ) 
    (h1 : jenna_stickers = 12)
    (h2 : kate_stickers * 4 = jenna_stickers * 7) : 
  kate_stickers = 21 := by
  sorry

end kate_stickers_l4086_408672


namespace hexagon_area_l4086_408692

-- Define the hexagon vertices
def hexagon_vertices : List (ℝ × ℝ) := [(0,0), (1,4), (3,4), (4,0), (3,-4), (1,-4)]

-- Function to calculate the area of a polygon given its vertices
def polygon_area (vertices : List (ℝ × ℝ)) : ℝ := sorry

-- Theorem statement
theorem hexagon_area : polygon_area hexagon_vertices = 24 := by sorry

end hexagon_area_l4086_408692


namespace polynomial_division_quotient_l4086_408667

theorem polynomial_division_quotient :
  let dividend : Polynomial ℚ := 10 * X^4 - 5 * X^3 + 3 * X^2 + 11 * X - 6
  let divisor : Polynomial ℚ := 5 * X^2 + 7
  let quotient : Polynomial ℚ := 2 * X^2 - X - 11/5
  (dividend : Polynomial ℚ).div divisor = quotient := by
  sorry

end polynomial_division_quotient_l4086_408667


namespace inequality_solution_l4086_408664

theorem inequality_solution (x : ℝ) (h1 : x > 0) 
  (h2 : x * Real.sqrt (16 - x^2) + Real.sqrt (16*x - x^4) ≥ 16) : 
  x = 2 * Real.sqrt 2 := by
  sorry

end inequality_solution_l4086_408664


namespace quadrilateral_with_equal_incircle_radii_is_rhombus_l4086_408652

/-- A point in a 2D plane -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- A quadrilateral defined by four points -/
structure Quadrilateral :=
  (A B C D : Point)

/-- The intersection point of the diagonals of a quadrilateral -/
def diagonalIntersection (q : Quadrilateral) : Point :=
  sorry

/-- The radius of the incircle of a triangle -/
def incircleRadius (p1 p2 p3 : Point) : ℝ :=
  sorry

/-- Checks if a quadrilateral is convex -/
def isConvex (q : Quadrilateral) : Prop :=
  sorry

/-- Checks if a quadrilateral is a rhombus -/
def isRhombus (q : Quadrilateral) : Prop :=
  sorry

theorem quadrilateral_with_equal_incircle_radii_is_rhombus
  (q : Quadrilateral)
  (h_convex : isConvex q)
  (O : Point)
  (h_O : O = diagonalIntersection q)
  (h_radii : incircleRadius q.A q.B O = incircleRadius q.B q.C O ∧
             incircleRadius q.B q.C O = incircleRadius q.C q.D O ∧
             incircleRadius q.C q.D O = incircleRadius q.D q.A O) :
  isRhombus q :=
sorry

end quadrilateral_with_equal_incircle_radii_is_rhombus_l4086_408652


namespace monotonic_quadratic_range_l4086_408695

/-- A quadratic function f(x) = x^2 + 2(a-1)x + 2 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*(a-1)*x + 2

/-- The function f is monotonic on the interval [-4, 4] -/
def is_monotonic_on_interval (a : ℝ) : Prop :=
  (∀ x y, -4 ≤ x ∧ x < y ∧ y ≤ 4 → f a x < f a y) ∨
  (∀ x y, -4 ≤ x ∧ x < y ∧ y ≤ 4 → f a x > f a y)

/-- If f(x) = x^2 + 2(a-1)x + 2 is monotonic on the interval [-4, 4], then a ≤ -3 or a ≥ 5 -/
theorem monotonic_quadratic_range (a : ℝ) : 
  is_monotonic_on_interval a → a ≤ -3 ∨ a ≥ 5 := by
  sorry

end monotonic_quadratic_range_l4086_408695


namespace max_area_difference_l4086_408657

/-- Represents a rectangle with integer dimensions -/
structure Rectangle where
  length : ℕ
  width : ℕ

/-- The perimeter of a rectangle -/
def perimeter (r : Rectangle) : ℕ := 2 * (r.length + r.width)

/-- The area of a rectangle -/
def area (r : Rectangle) : ℕ := r.length * r.width

/-- Theorem stating the maximum area difference between two rectangles -/
theorem max_area_difference :
  ∃ (r1 r2 : Rectangle),
    perimeter r1 = 200 ∧
    perimeter r2 = 200 ∧
    r2.width = 20 ∧
    ∀ (r3 r4 : Rectangle),
      perimeter r3 = 200 →
      perimeter r4 = 200 →
      r4.width = 20 →
      area r1 - area r2 ≥ area r3 - area r4 ∧
      area r1 - area r2 = 900 :=
sorry

end max_area_difference_l4086_408657


namespace quadratic_inequality_solution_set_l4086_408620

theorem quadratic_inequality_solution_set 
  (a b c x₁ x₂ : ℝ) 
  (h₁ : x₁ < x₂) 
  (h₂ : a < 0) 
  (h₃ : ∀ x, a * x^2 + b * x + c = 0 ↔ x = x₁ ∨ x = x₂) :
  ∀ x, a * x^2 + b * x + c > 0 ↔ x₁ < x ∧ x < x₂ := by
sorry

end quadratic_inequality_solution_set_l4086_408620


namespace backpack_player_prices_l4086_408671

/-- Represents the prices and discounts for the backpack and portable music player problem -/
structure PriceInfo where
  backpack_price : ℕ
  player_price : ℕ
  renmin_discount : Rat
  carrefour_voucher : ℕ
  carrefour_voucher_threshold : ℕ
  budget : ℕ

/-- Calculates the total price at Renmin Department Store after discount -/
def renmin_total (info : PriceInfo) : Rat :=
  (info.backpack_price + info.player_price : Rat) * info.renmin_discount

/-- Calculates the total price at Carrefour after applying vouchers -/
def carrefour_total (info : PriceInfo) : ℕ :=
  info.player_price + info.backpack_price - 
    ((info.player_price + info.backpack_price) / info.carrefour_voucher_threshold) * info.carrefour_voucher

/-- The main theorem stating the correct prices and the more cost-effective store -/
theorem backpack_player_prices (info : PriceInfo) : 
  info.backpack_price = 92 ∧ 
  info.player_price = 360 ∧ 
  info.backpack_price + info.player_price = 452 ∧
  info.player_price = 4 * info.backpack_price - 8 ∧
  info.renmin_discount = 4/5 ∧
  info.carrefour_voucher = 30 ∧
  info.carrefour_voucher_threshold = 100 ∧
  info.budget = 400 →
  renmin_total info < carrefour_total info ∧
  renmin_total info ≤ info.budget ∧
  (carrefour_total info : Rat) ≤ info.budget := by
  sorry

end backpack_player_prices_l4086_408671


namespace inscribed_rectangle_coefficient_l4086_408636

/-- Triangle ABC with inscribed rectangle PQRS --/
structure TriangleWithRectangle where
  /-- Side length AB --/
  ab : ℝ
  /-- Side length BC --/
  bc : ℝ
  /-- Side length CA --/
  ca : ℝ
  /-- Width of the inscribed rectangle (PQ) --/
  ω : ℝ
  /-- Coefficient α in the area formula --/
  α : ℝ
  /-- Coefficient β in the area formula --/
  β : ℝ
  /-- P is on AB, Q on AC, R and S on BC --/
  rectangle_inscribed : Bool
  /-- Area formula for rectangle PQRS --/
  area_formula : ℝ → ℝ := fun ω => α * ω - β * ω^2

/-- The main theorem --/
theorem inscribed_rectangle_coefficient
  (t : TriangleWithRectangle)
  (h1 : t.ab = 15)
  (h2 : t.bc = 26)
  (h3 : t.ca = 25)
  (h4 : t.rectangle_inscribed = true) :
  t.β = 33 / 28 := by
  sorry

end inscribed_rectangle_coefficient_l4086_408636


namespace min_fence_length_for_given_garden_l4086_408615

/-- Calculates the minimum fence length for a rectangular garden with one side against a wall -/
def min_fence_length (length width : ℝ) : ℝ :=
  2 * width + length

theorem min_fence_length_for_given_garden :
  min_fence_length 32 14 = 60 := by
  sorry

end min_fence_length_for_given_garden_l4086_408615


namespace one_third_of_360_l4086_408649

theorem one_third_of_360 : (360 : ℝ) * (1 / 3) = 120 := by
  sorry

end one_third_of_360_l4086_408649


namespace third_roll_five_prob_l4086_408690

/-- Represents a six-sided die --/
inductive Die
| Fair
| Biased

/-- Probability of rolling a five for a given die --/
def prob_five (d : Die) : ℚ :=
  match d with
  | Die.Fair => 1/6
  | Die.Biased => 3/4

/-- Probability of rolling a non-five for a given die --/
def prob_not_five (d : Die) : ℚ :=
  match d with
  | Die.Fair => 5/6
  | Die.Biased => 1/4

/-- Probability of choosing each die initially --/
def initial_prob : ℚ := 1/2

/-- Theorem stating the probability of rolling a five on the third roll --/
theorem third_roll_five_prob :
  let p_fair := initial_prob * (prob_five Die.Fair)^2
  let p_biased := initial_prob * (prob_five Die.Biased)^2
  let p_fair_given_two_fives := p_fair / (p_fair + p_biased)
  let p_biased_given_two_fives := p_biased / (p_fair + p_biased)
  p_fair_given_two_fives * (prob_five Die.Fair) + 
  p_biased_given_two_fives * (prob_five Die.Biased) = 223/74 := by
  sorry

end third_roll_five_prob_l4086_408690


namespace arithmetic_sequence_properties_l4086_408647

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ+ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ+, a (n + 1) = a n + d

/-- Partial sum of an arithmetic sequence -/
def partial_sum (a : ℕ+ → ℝ) (n : ℕ+) : ℝ :=
  (Finset.range n.val).sum (λ i => a ⟨i + 1, Nat.succ_pos i⟩)

theorem arithmetic_sequence_properties
  (a : ℕ+ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_sum : partial_sum a 6 > partial_sum a 7 ∧ partial_sum a 7 > partial_sum a 5) :
  (∃ d : ℝ, d < 0 ∧ ∀ n : ℕ+, a (n + 1) = a n + d) ∧
  partial_sum a 11 > 0 :=
sorry

end arithmetic_sequence_properties_l4086_408647


namespace special_rectangle_area_l4086_408683

/-- Represents a rectangle with a diagonal of length y and length three times its width -/
structure SpecialRectangle where
  y : ℝ  -- diagonal length
  w : ℝ  -- width
  h : ℝ  -- height (length)
  h_eq : h = 3 * w  -- length is three times the width
  diag_eq : y^2 = h^2 + w^2  -- Pythagorean theorem for the diagonal

/-- The area of a SpecialRectangle is 3y^2/10 -/
theorem special_rectangle_area (rect : SpecialRectangle) :
  rect.w * rect.h = (3 * rect.y^2) / 10 := by
  sorry

end special_rectangle_area_l4086_408683


namespace cos_equality_proof_l4086_408633

theorem cos_equality_proof (n : ℤ) : 
  0 ≤ n ∧ n ≤ 180 ∧ Real.cos (n * π / 180) = Real.cos (317 * π / 180) → n = 43 := by
  sorry

end cos_equality_proof_l4086_408633


namespace arccos_zero_l4086_408604

theorem arccos_zero : Real.arccos 0 = π / 2 := by sorry

end arccos_zero_l4086_408604


namespace jason_percentage_more_than_zachary_l4086_408687

/-- Proves that Jason received 30% more money than Zachary from selling video games -/
theorem jason_percentage_more_than_zachary 
  (zachary_games : ℕ) 
  (zachary_price : ℚ) 
  (ryan_extra : ℚ) 
  (total_amount : ℚ) 
  (h1 : zachary_games = 40)
  (h2 : zachary_price = 5)
  (h3 : ryan_extra = 50)
  (h4 : total_amount = 770)
  (h5 : zachary_games * zachary_price + 2 * (zachary_games * zachary_price + ryan_extra) / 2 = total_amount) :
  (((zachary_games * zachary_price + ryan_extra) / 2 - zachary_games * zachary_price) / (zachary_games * zachary_price)) * 100 = 30 := by
  sorry

end jason_percentage_more_than_zachary_l4086_408687


namespace xy_value_l4086_408678

theorem xy_value (x y : ℝ) (h : (x - 3)^2 + |y + 2| = 0) : x * y = -6 := by
  sorry

end xy_value_l4086_408678


namespace garden_dimensions_l4086_408643

/-- Represents a rectangular garden with given perimeter and length-width relationship --/
structure RectangularGarden where
  width : ℝ
  length : ℝ
  perimeter : ℝ
  length_width_relation : length = width + 3
  perimeter_formula : perimeter = 2 * (length + width)

/-- Theorem stating the dimensions of the garden given the conditions --/
theorem garden_dimensions (g : RectangularGarden) 
  (h : g.perimeter = 26) : g.width = 5 ∧ g.length = 8 := by
  sorry

#check garden_dimensions

end garden_dimensions_l4086_408643


namespace triangle_2_3_4_l4086_408663

-- Define the triangle operation
def triangle (a b c : ℝ) : ℝ := b^3 - 5*a*c

-- Theorem statement
theorem triangle_2_3_4 : triangle 2 3 4 = -13 := by
  sorry

end triangle_2_3_4_l4086_408663


namespace wine_equation_correct_l4086_408602

/-- Represents the value of clear wine in terms of grain -/
def clear_wine_value : ℝ := 10

/-- Represents the value of turbid wine in terms of grain -/
def turbid_wine_value : ℝ := 3

/-- Represents the total amount of grain available -/
def total_grain : ℝ := 30

/-- Represents the total amount of wine obtained -/
def total_wine : ℝ := 5

/-- Theorem stating that the equation 10x + 3(5-x) = 30 correctly represents
    the relationship between clear wine, turbid wine, and total grain value -/
theorem wine_equation_correct (x : ℝ) :
  0 ≤ x ∧ x ≤ total_wine →
  clear_wine_value * x + turbid_wine_value * (total_wine - x) = total_grain :=
by sorry

end wine_equation_correct_l4086_408602


namespace base_equality_l4086_408681

/-- Given a positive integer b, converts the base-b number 101ᵦ to base 10 -/
def base_b_to_decimal (b : ℕ) : ℕ := b^2 + 1

/-- Converts 24₅ to base 10 -/
def base_5_to_decimal : ℕ := 2 * 5 + 4

/-- The theorem states that 4 is the unique positive integer b that satisfies 24₅ = 101ᵦ -/
theorem base_equality : ∃! (b : ℕ), b > 0 ∧ base_5_to_decimal = base_b_to_decimal b :=
sorry

end base_equality_l4086_408681


namespace quadratic_polynomial_satisfies_conditions_l4086_408628

theorem quadratic_polynomial_satisfies_conditions :
  ∃ (q : ℝ → ℝ),
    (∀ x, q x = 2.5 * x^2 - 5.5 * x + 13) ∧
    q (-1) = 10 ∧
    q 2 = 1 ∧
    q 4 = 20 := by
  sorry

end quadratic_polynomial_satisfies_conditions_l4086_408628
