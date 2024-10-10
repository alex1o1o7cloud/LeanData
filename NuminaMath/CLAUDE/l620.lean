import Mathlib

namespace choir_robe_expenditure_is_36_l620_62045

/-- Calculates the total expenditure for additional choir robes. -/
def choir_robe_expenditure (total_singers : ℕ) (existing_robes : ℕ) (cost_per_robe : ℕ) : ℕ :=
  (total_singers - existing_robes) * cost_per_robe

/-- Proves that the expenditure for additional choir robes is $36 given the specified conditions. -/
theorem choir_robe_expenditure_is_36 :
  choir_robe_expenditure 30 12 2 = 36 := by
  sorry

end choir_robe_expenditure_is_36_l620_62045


namespace trapezoid_bases_solutions_l620_62053

theorem trapezoid_bases_solutions :
  let area : ℕ := 1800
  let altitude : ℕ := 60
  let base_sum : ℕ := 2 * area / altitude
  let valid_base_pair := λ b₁ b₂ : ℕ =>
    b₁ % 10 = 0 ∧ b₂ % 10 = 0 ∧ b₁ + b₂ = base_sum
  (∃! (solutions : Finset (ℕ × ℕ)), solutions.card = 4 ∧
    ∀ pair : ℕ × ℕ, pair ∈ solutions ↔ valid_base_pair pair.1 pair.2) :=
by sorry

end trapezoid_bases_solutions_l620_62053


namespace quadratic_inequality_properties_l620_62010

/-- Given that the solution set of ax² - bx + c > 0 is (-1, 2), prove the following properties -/
theorem quadratic_inequality_properties 
  (a b c : ℝ) 
  (h : ∀ x : ℝ, ax^2 - b*x + c > 0 ↔ -1 < x ∧ x < 2) : 
  (a + b + c = 0) ∧ (a < 0) := by
  sorry

end quadratic_inequality_properties_l620_62010


namespace polynomial_division_quotient_l620_62063

theorem polynomial_division_quotient : ∀ x : ℝ,
  (9 * x^3 - 5 * x^2 + 8 * x - 12) = (x - 3) * (9 * x^2 + 22 * x + 74) + 210 := by
  sorry

end polynomial_division_quotient_l620_62063


namespace a_lt_neg_four_sufficient_not_necessary_l620_62061

/-- The function f(x) = ax + 3 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x + 3

/-- The condition for f to have a zero point on [-1,1] -/
def has_zero_point (a : ℝ) : Prop := ∃ x : ℝ, x ∈ Set.Icc (-1) 1 ∧ f a x = 0

/-- The statement that a < -4 is sufficient but not necessary for f to have a zero point on [-1,1] -/
theorem a_lt_neg_four_sufficient_not_necessary :
  (∀ a : ℝ, a < -4 → has_zero_point a) ∧
  ¬(∀ a : ℝ, has_zero_point a → a < -4) :=
sorry

end a_lt_neg_four_sufficient_not_necessary_l620_62061


namespace compare_fractions_l620_62022

theorem compare_fractions : -4/3 < -5/4 := by
  sorry

end compare_fractions_l620_62022


namespace prob_different_cities_l620_62055

/-- The probability that student A attends university in city A -/
def prob_A_cityA : ℝ := 0.6

/-- The probability that student B attends university in city A -/
def prob_B_cityA : ℝ := 0.3

/-- The theorem stating that the probability of A and B not attending university 
    in the same city is 0.54, given the probabilities of each student 
    attending city A -/
theorem prob_different_cities (h1 : 0 ≤ prob_A_cityA ∧ prob_A_cityA ≤ 1) 
                               (h2 : 0 ≤ prob_B_cityA ∧ prob_B_cityA ≤ 1) : 
  prob_A_cityA * (1 - prob_B_cityA) + (1 - prob_A_cityA) * prob_B_cityA = 0.54 := by
  sorry

end prob_different_cities_l620_62055


namespace jonathan_book_purchase_l620_62011

-- Define the costs of the books and Jonathan's savings
def dictionary_cost : ℕ := 11
def dinosaur_book_cost : ℕ := 19
def cookbook_cost : ℕ := 7
def savings : ℕ := 8

-- Define the total cost of the books
def total_cost : ℕ := dictionary_cost + dinosaur_book_cost + cookbook_cost

-- Define the amount Jonathan needs
def amount_needed : ℕ := total_cost - savings

-- Theorem statement
theorem jonathan_book_purchase :
  amount_needed = 29 :=
by sorry

end jonathan_book_purchase_l620_62011


namespace five_in_C_l620_62083

def C : Set ℕ := {x | 1 ≤ x ∧ x < 10}

theorem five_in_C : 5 ∈ C := by sorry

end five_in_C_l620_62083


namespace badminton_players_count_l620_62050

/-- Represents a sports club with members playing badminton and tennis -/
structure SportsClub where
  total : ℕ
  tennis : ℕ
  neither : ℕ
  both : ℕ

/-- Calculates the number of members playing badminton in a sports club -/
def badminton_players (club : SportsClub) : ℕ :=
  club.total - club.neither - (club.tennis - club.both)

/-- Theorem stating the number of badminton players in the given sports club -/
theorem badminton_players_count (club : SportsClub) 
  (h_total : club.total = 30)
  (h_tennis : club.tennis = 19)
  (h_neither : club.neither = 3)
  (h_both : club.both = 9) :
  badminton_players club = 17 := by
  sorry

#eval badminton_players { total := 30, tennis := 19, neither := 3, both := 9 }

end badminton_players_count_l620_62050


namespace product_equals_3408_decimal_product_l620_62027

theorem product_equals_3408 : 213 * 16 = 3408 := by
  sorry

-- Additional fact (not used in the proof)
theorem decimal_product : 0.16 * 2.13 = 0.3408 := by
  sorry

end product_equals_3408_decimal_product_l620_62027


namespace canal_construction_efficiency_l620_62086

theorem canal_construction_efficiency (total_length : ℝ) (efficiency_multiplier : ℝ) (days_ahead : ℝ) 
  (original_daily_plan : ℝ) : 
  total_length = 3600 ∧ 
  efficiency_multiplier = 1.8 ∧ 
  days_ahead = 20 ∧
  (total_length / original_daily_plan - total_length / (efficiency_multiplier * original_daily_plan) = days_ahead) →
  original_daily_plan = 20 := by
sorry

end canal_construction_efficiency_l620_62086


namespace distribute_four_to_three_l620_62098

/-- The number of ways to distribute volunteers to venues -/
def distribute_volunteers (num_volunteers : ℕ) (num_venues : ℕ) : ℕ :=
  if num_venues > num_volunteers then 0
  else if num_venues = 1 then 1
  else (num_volunteers - 1).choose (num_venues - 1) * num_venues.factorial

/-- Theorem: Distributing 4 volunteers to 3 venues yields 36 schemes -/
theorem distribute_four_to_three :
  distribute_volunteers 4 3 = 36 := by
  sorry

#eval distribute_volunteers 4 3

end distribute_four_to_three_l620_62098


namespace transform_point_l620_62021

/-- Rotate a point 90 degrees clockwise around a center point -/
def rotate90Clockwise (p : ℝ × ℝ) (center : ℝ × ℝ) : ℝ × ℝ :=
  let (x, y) := p
  let (cx, cy) := center
  (cx + (y - cy), cy - (x - cx))

/-- Reflect a point over the x-axis -/
def reflectOverX (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

/-- The main theorem -/
theorem transform_point :
  let A : ℝ × ℝ := (-4, 1)
  let center : ℝ × ℝ := (1, 1)
  let rotated := rotate90Clockwise A center
  let final := reflectOverX rotated
  final = (1, -6) := by sorry

end transform_point_l620_62021


namespace norway_visitors_l620_62058

/-- Given a group of people with information about their visits to Iceland and Norway,
    calculate the number of people who visited Norway. -/
theorem norway_visitors
  (total : ℕ)
  (iceland : ℕ)
  (both : ℕ)
  (neither : ℕ)
  (h1 : total = 50)
  (h2 : iceland = 25)
  (h3 : both = 21)
  (h4 : neither = 23) :
  total = iceland + (norway : ℕ) - both + neither ∧ norway = 23 :=
by sorry

end norway_visitors_l620_62058


namespace johns_snack_spending_l620_62043

theorem johns_snack_spending (initial_amount : ℝ) (remaining_amount : ℝ) 
  (snack_fraction : ℝ) (necessity_fraction : ℝ) :
  initial_amount = 20 →
  remaining_amount = 4 →
  necessity_fraction = 3/4 →
  remaining_amount = initial_amount * (1 - snack_fraction) * (1 - necessity_fraction) →
  snack_fraction = 1/5 := by
  sorry

end johns_snack_spending_l620_62043


namespace return_speed_theorem_l620_62074

theorem return_speed_theorem (v : ℕ) : 
  v > 50 ∧ 
  v ≤ 100 ∧ 
  (∃ k : ℕ, k = (100 * v) / (50 + v)) → 
  v = 75 := by
sorry

end return_speed_theorem_l620_62074


namespace min_value_problem_l620_62001

theorem min_value_problem (a b c d e f g h : ℝ) 
  (h1 : a * b * c * d = 16) 
  (h2 : e * f * g * h = 36) : 
  (a*e)^2 + (b*f)^2 + (c*g)^2 + (d*h)^2 ≥ 576 := by
  sorry

end min_value_problem_l620_62001


namespace stating_arithmetic_sequence_iff_60_degree_l620_62088

/-- A triangle with interior angles A, B, and C. -/
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  sum_180 : A + B + C = 180
  positive : A > 0 ∧ B > 0 ∧ C > 0

/-- The interior angles of a triangle form an arithmetic sequence. -/
def arithmetic_sequence (t : Triangle) : Prop :=
  t.A + t.C = 2 * t.B ∨ t.A + t.B = 2 * t.C ∨ t.B + t.C = 2 * t.A

/-- One of the interior angles of a triangle is 60 degrees. -/
def has_60_degree (t : Triangle) : Prop :=
  t.A = 60 ∨ t.B = 60 ∨ t.C = 60

/-- 
Theorem stating that a triangle's interior angles form an arithmetic sequence 
if and only if one of its interior angles is 60 degrees.
-/
theorem arithmetic_sequence_iff_60_degree (t : Triangle) :
  arithmetic_sequence t ↔ has_60_degree t :=
sorry

end stating_arithmetic_sequence_iff_60_degree_l620_62088


namespace union_A_B_when_m_neg_two_intersection_A_B_equals_B_iff_l620_62044

-- Define sets A and B
def A : Set ℝ := {x | (x - 2) / (x + 1) ≤ 0}
def B (m : ℝ) : Set ℝ := {x | 2 * m + 3 < x ∧ x < m^2}

-- Theorem for part 1
theorem union_A_B_when_m_neg_two :
  A ∪ B (-2) = {x | -1 < x ∧ x < 4} := by sorry

-- Theorem for part 2
theorem intersection_A_B_equals_B_iff (m : ℝ) :
  A ∩ B m = B m ↔ m ∈ Set.Icc (-Real.sqrt 2) 3 := by sorry

end union_A_B_when_m_neg_two_intersection_A_B_equals_B_iff_l620_62044


namespace values_equal_l620_62090

/-- The value of the expression at point C -/
def value_at_C : ℝ := 5 * 5 + 6 * 8.73

/-- The value of the expression at point D -/
def value_at_D : ℝ := 105

/-- Theorem stating that the values at points C and D are equal -/
theorem values_equal : value_at_C = value_at_D := by
  sorry

#eval value_at_C
#eval value_at_D

end values_equal_l620_62090


namespace student_average_score_l620_62014

theorem student_average_score (M P C : ℕ) : 
  M + P = 50 → C = P + 20 → (M + C) / 2 = 35 := by
  sorry

end student_average_score_l620_62014


namespace yogurt_satisfaction_probability_l620_62047

theorem yogurt_satisfaction_probability 
  (total_sample : ℕ) 
  (satisfied_with_yogurt : ℕ) 
  (h1 : total_sample = 500) 
  (h2 : satisfied_with_yogurt = 370) : 
  (satisfied_with_yogurt : ℚ) / total_sample = 37 / 50 := by
  sorry

end yogurt_satisfaction_probability_l620_62047


namespace min_c_value_l620_62092

theorem min_c_value (a b c : ℕ) : 
  a > 0 → b > 0 → c > 0 →
  (a + b + c) / 3 = 20 →
  a ≤ b → b ≤ c →
  b = a + 13 →
  ∀ c' : ℕ, c' > 0 ∧ 
    (∃ a' b' : ℕ, a' > 0 ∧ b' > 0 ∧
      (a' + b' + c') / 3 = 20 ∧
      a' ≤ b' ∧ b' ≤ c' ∧
      b' = a' + 13) →
    c ≤ c' →
  c = 45 :=
sorry

end min_c_value_l620_62092


namespace brothers_difference_l620_62024

theorem brothers_difference (aaron_brothers : ℕ) (bennett_brothers : ℕ) : 
  aaron_brothers = 4 → bennett_brothers = 6 → 2 * aaron_brothers - bennett_brothers = 2 := by
  sorry

end brothers_difference_l620_62024


namespace exists_cubic_positive_l620_62015

theorem exists_cubic_positive : ∃ x : ℝ, x^3 - x^2 + 1 > 0 := by sorry

end exists_cubic_positive_l620_62015


namespace infinite_sum_evaluation_l620_62096

theorem infinite_sum_evaluation :
  (∑' n : ℕ, (3 : ℝ)^n / (1 + 3^n + 3^(n + 1) + 3^(2*n + 1))) = 1/4 := by
  sorry

end infinite_sum_evaluation_l620_62096


namespace divisors_half_of_n_l620_62099

theorem divisors_half_of_n (n : ℕ) : 
  (n > 0) → (Finset.card (Nat.divisors n) = n / 2) → (n = 8 ∨ n = 12) := by
  sorry

end divisors_half_of_n_l620_62099


namespace mistaken_multiplication_l620_62006

/-- Given a polynomial P(x) such that (-3x) * P(x) = 3x³ - 3x² + 3x,
    prove that P(x) - 3x = -x² - 2x - 1 -/
theorem mistaken_multiplication (P : ℝ → ℝ) :
  (∀ x, (-3 * x) * P x = 3 * x^3 - 3 * x^2 + 3 * x) →
  (∀ x, P x - 3 * x = -x^2 - 2 * x - 1) :=
by sorry

end mistaken_multiplication_l620_62006


namespace sufficient_not_necessary_condition_l620_62029

theorem sufficient_not_necessary_condition (a : ℝ) : 
  (∀ x, x > 1 → x > a) ∧ (∃ x, x > a ∧ x ≤ 1) → a < 1 := by
  sorry

end sufficient_not_necessary_condition_l620_62029


namespace quadratic_equation_solution_l620_62005

theorem quadratic_equation_solution (x : ℝ) : -x^2 - (-16 + 10)*x - 8 = -(x - 2)*(x - 4) := by
  sorry

end quadratic_equation_solution_l620_62005


namespace square_plus_abs_zero_implies_both_zero_l620_62068

theorem square_plus_abs_zero_implies_both_zero (a b : ℝ) : 
  a^2 + |b| = 0 → a = 0 ∧ b = 0 := by
  sorry

end square_plus_abs_zero_implies_both_zero_l620_62068


namespace odd_function_property_l620_62036

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem odd_function_property (f : ℝ → ℝ) (h1 : is_odd_function f) (h2 : f 3 - f 2 = 1) :
  f (-2) - f (-3) = 1 := by sorry

end odd_function_property_l620_62036


namespace sarah_christmas_shopping_l620_62067

/-- The amount of money Sarah started with for Christmas shopping. -/
def initial_amount : ℕ := 100

/-- The cost of each toy car. -/
def toy_car_cost : ℕ := 11

/-- The number of toy cars Sarah bought. -/
def num_toy_cars : ℕ := 2

/-- The cost of the scarf. -/
def scarf_cost : ℕ := 10

/-- The cost of the beanie. -/
def beanie_cost : ℕ := 14

/-- The cost of the necklace. -/
def necklace_cost : ℕ := 20

/-- The cost of the gloves. -/
def gloves_cost : ℕ := 12

/-- The cost of the book. -/
def book_cost : ℕ := 15

/-- The amount of money Sarah has remaining after purchasing all gifts. -/
def remaining_amount : ℕ := 7

/-- Theorem stating that the initial amount is equal to the sum of all gift costs plus the remaining amount. -/
theorem sarah_christmas_shopping :
  initial_amount = 
    num_toy_cars * toy_car_cost + 
    scarf_cost + 
    beanie_cost + 
    necklace_cost + 
    gloves_cost + 
    book_cost + 
    remaining_amount :=
by sorry

end sarah_christmas_shopping_l620_62067


namespace cube_sum_from_sum_and_square_sum_l620_62004

theorem cube_sum_from_sum_and_square_sum (x y : ℝ) 
  (h1 : x + y = 5) 
  (h2 : x^2 + y^2 = 17) : 
  x^3 + y^3 = 65 := by
sorry

end cube_sum_from_sum_and_square_sum_l620_62004


namespace third_player_games_l620_62034

/-- Represents a chess tournament with three players. -/
structure ChessTournament where
  total_games : ℕ
  player1_games : ℕ
  player2_games : ℕ
  player3_games : ℕ

/-- The theorem stating the number of games played by the third player. -/
theorem third_player_games (t : ChessTournament) 
  (h1 : t.total_games = 27)
  (h2 : t.player1_games = 13)
  (h3 : t.player2_games = 27)
  (h4 : t.player1_games + t.player2_games + t.player3_games = 2 * t.total_games) :
  t.player3_games = 14 := by
  sorry


end third_player_games_l620_62034


namespace phone_reselling_profit_l620_62059

theorem phone_reselling_profit (initial_investment : ℝ) (profit_ratio : ℝ) (selling_price : ℝ) :
  initial_investment = 3000 →
  profit_ratio = 1 / 3 →
  selling_price = 20 →
  (initial_investment * (1 + profit_ratio)) / selling_price = 200 := by
  sorry

end phone_reselling_profit_l620_62059


namespace cost_of_three_pencils_two_pens_l620_62095

/-- The cost of a single pencil -/
def pencil_cost : ℝ := sorry

/-- The cost of a single pen -/
def pen_cost : ℝ := sorry

/-- The total cost of three pencils and two pens is $4.15 -/
axiom three_pencils_two_pens : 3 * pencil_cost + 2 * pen_cost = 4.15

/-- The cost of two pencils and three pens is $3.70 -/
axiom two_pencils_three_pens : 2 * pencil_cost + 3 * pen_cost = 3.70

/-- The cost of three pencils and two pens is $4.15 -/
theorem cost_of_three_pencils_two_pens : 3 * pencil_cost + 2 * pen_cost = 4.15 := by
  sorry

end cost_of_three_pencils_two_pens_l620_62095


namespace hyperbola_asymptotes_l620_62082

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 and eccentricity e,
    prove that its asymptotes are √3x ± y = 0 when e = 2 -/
theorem hyperbola_asymptotes (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) :
  let e := Real.sqrt (a^2 + b^2) / a
  e = 2 →
  ∃ (k : ℝ), k = Real.sqrt 3 ∧
    (∀ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1 →
      (y = k * x ∨ y = -k * x)) :=
by sorry

end hyperbola_asymptotes_l620_62082


namespace whiteboard_washing_l620_62019

theorem whiteboard_washing (kids : ℕ) (whiteboards : ℕ) (time : ℕ) :
  kids = 4 →
  whiteboards = 3 →
  time = 20 →
  (1 : ℝ) * 160 * whiteboards = kids * time * 6 :=
by sorry

end whiteboard_washing_l620_62019


namespace work_completion_time_l620_62046

-- Define the rates of work for A and B
def rate_A : ℚ := 1 / 16
def rate_B : ℚ := rate_A / 3

-- Define the total rate when A and B work together
def total_rate : ℚ := rate_A + rate_B

-- Theorem statement
theorem work_completion_time :
  (1 : ℚ) / total_rate = 12 := by sorry

end work_completion_time_l620_62046


namespace fraction_with_buddies_l620_62066

/-- Represents the number of students in each grade --/
structure StudentCounts where
  ninth : ℚ
  sixth : ℚ
  seventh : ℚ

/-- Represents the pairing ratios --/
structure PairingRatios where
  ninth : ℚ
  sixth : ℚ

/-- Represents the school mentoring program --/
structure MentoringProgram where
  counts : StudentCounts
  ratios : PairingRatios

/-- The main theorem about the fraction of students with buddies --/
theorem fraction_with_buddies (program : MentoringProgram) 
  (h1 : program.counts.ninth = 5 * program.counts.sixth / 4)
  (h2 : program.counts.seventh = 3 * program.counts.sixth / 4)
  (h3 : program.ratios.ninth = 1/4)
  (h4 : program.ratios.sixth = 1/3)
  (h5 : program.ratios.ninth * program.counts.ninth = program.ratios.sixth * program.counts.sixth) :
  (program.ratios.ninth * program.counts.ninth) / 
  (program.counts.ninth + program.counts.sixth + program.counts.seventh) = 5/48 := by
  sorry

end fraction_with_buddies_l620_62066


namespace unique_krakozyabr_count_l620_62080

def Krakozyabr : Type := Unit

structure KrakozyabrPopulation where
  total : ℕ
  horns : ℕ
  wings : ℕ
  both : ℕ
  all_have_horns_or_wings : total = horns + wings - both
  horns_with_wings_ratio : both = horns / 5
  wings_with_horns_ratio : both = wings / 4
  total_range : 25 < total ∧ total < 35

theorem unique_krakozyabr_count : 
  ∀ (pop : KrakozyabrPopulation), pop.total = 32 := by
  sorry

end unique_krakozyabr_count_l620_62080


namespace cos_540_degrees_l620_62084

theorem cos_540_degrees : Real.cos (540 * π / 180) = -1 := by
  sorry

end cos_540_degrees_l620_62084


namespace complex_number_problem_l620_62008

theorem complex_number_problem (z : ℂ) :
  Complex.abs z = 5 ∧ (Complex.I * Complex.im ((3 + 4 * Complex.I) * z) = (3 + 4 * Complex.I) * z) →
  z = 4 + 3 * Complex.I ∨ z = -4 - 3 * Complex.I :=
by sorry

end complex_number_problem_l620_62008


namespace F_10_squares_l620_62017

/-- Represents the number of squares in figure F_n -/
def num_squares (n : ℕ) : ℕ :=
  1 + 3 * (n - 1) * n

/-- The theorem stating that F_10 contains 271 squares -/
theorem F_10_squares : num_squares 10 = 271 := by
  sorry

end F_10_squares_l620_62017


namespace almond_croissant_price_l620_62093

/-- The price of an almond croissant given Harrison's croissant buying habits -/
theorem almond_croissant_price :
  let regular_price : ℚ := 7/2  -- $3.50
  let weeks_in_year : ℕ := 52
  let total_spent : ℚ := 468
  let almond_price : ℚ := (total_spent - weeks_in_year * regular_price) / weeks_in_year
  almond_price = 11/2  -- $5.50
  := by sorry

end almond_croissant_price_l620_62093


namespace inequality_proof_l620_62048

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : Real.sqrt a + Real.sqrt b = 2) :
  (a * Real.sqrt b + b * Real.sqrt a ≤ 2) ∧ (2 ≤ a^2 + b^2) ∧ (a^2 + b^2 < 16) := by
  sorry

end inequality_proof_l620_62048


namespace perpendicular_condition_l620_62023

-- Define the lines l₁ and l₂
def l₁ (x y a : ℝ) : Prop := x + a * y - 2 = 0
def l₂ (x y a : ℝ) : Prop := x - a * y - 1 = 0

-- Define perpendicularity of lines
def perpendicular (a : ℝ) : Prop := 1 + a * (-a) = 0

-- Define sufficient condition
def sufficient (P Q : Prop) : Prop := P → Q

-- Define necessary condition
def necessary (P Q : Prop) : Prop := Q → P

theorem perpendicular_condition (a : ℝ) :
  sufficient (a = -1) (perpendicular a) ∧
  ¬ necessary (a = -1) (perpendicular a) :=
by sorry

end perpendicular_condition_l620_62023


namespace factor_decomposition_96_l620_62049

theorem factor_decomposition_96 : 
  ∃ (x y : ℤ), x * y = 96 ∧ x^2 + y^2 = 208 := by
  sorry

end factor_decomposition_96_l620_62049


namespace skidding_distance_speed_relation_l620_62057

theorem skidding_distance_speed_relation 
  (a b : ℝ) 
  (h1 : b = a * 60^2) 
  (h2 : 3 * b = a * x^2) : 
  x = 60 * Real.sqrt 3 := by
  sorry

end skidding_distance_speed_relation_l620_62057


namespace brads_running_speed_l620_62073

/-- Prove Brad's running speed given the conditions of the problem -/
theorem brads_running_speed 
  (total_distance : ℝ) 
  (maxwells_speed : ℝ) 
  (maxwells_distance : ℝ) 
  (h1 : total_distance = 40) 
  (h2 : maxwells_speed = 3) 
  (h3 : maxwells_distance = 15) : 
  ∃ (brads_speed : ℝ), brads_speed = 5 := by
  sorry


end brads_running_speed_l620_62073


namespace smallest_n_terminating_with_3_l620_62040

def is_terminating_decimal (n : ℕ) : Prop :=
  ∃ (a b : ℕ), n = 2^a * 5^b

def contains_digit_3 (n : ℕ) : Prop :=
  ∃ (d : ℕ), d < 10 ∧ (n / 10^d) % 10 = 3

theorem smallest_n_terminating_with_3 :
  ∀ n : ℕ, n > 0 →
    (is_terminating_decimal n ∧ contains_digit_3 n) →
    n ≥ 32 :=
by sorry

end smallest_n_terminating_with_3_l620_62040


namespace sequence_properties_l620_62078

def sequence_a (n : ℕ) : ℝ := 3 * (2^n - 1)

def sequence_b (n : ℕ) : ℝ := sequence_a n + 3

def sum_S (n : ℕ) : ℝ := 2 * sequence_a n - 3 * n

theorem sequence_properties :
  (∀ n : ℕ, sum_S (n + 1) = 2 * sequence_a (n + 1) - 3 * (n + 1)) ∧
  sequence_a 1 = 3 ∧
  sequence_a 2 = 9 ∧
  sequence_a 3 = 21 ∧
  (∀ n : ℕ, sequence_b (n + 1) = 2 * sequence_b n) ∧
  (∀ n : ℕ, sequence_a n = 3 * (2^n - 1)) :=
by sorry

end sequence_properties_l620_62078


namespace x_fourth_plus_inverse_x_fourth_l620_62091

theorem x_fourth_plus_inverse_x_fourth (x : ℝ) (h : x ≠ 0) :
  x^2 + 1/x^2 = 5 → x^4 + 1/x^4 = 23 := by
  sorry

end x_fourth_plus_inverse_x_fourth_l620_62091


namespace prob_ride_all_cars_l620_62042

/-- The number of cars in the roller coaster -/
def num_cars : ℕ := 4

/-- The number of times the passenger rides the roller coaster -/
def num_rides : ℕ := 4

/-- The probability of choosing any specific car for a single ride -/
def prob_single_car : ℚ := 1 / num_cars

/-- The probability of riding in each of the 4 cars exactly once in 4 rides -/
def prob_all_cars : ℚ := 3 / 32

/-- Theorem stating that the probability of riding in each car exactly once is 3/32 -/
theorem prob_ride_all_cars : 
  prob_all_cars = (num_cars.factorial : ℚ) / num_cars ^ num_rides :=
sorry

end prob_ride_all_cars_l620_62042


namespace megan_popsicle_consumption_l620_62003

/-- The number of Popsicles Megan can finish in 5 hours -/
def popsicles_in_5_hours : ℕ := 15

/-- The time in minutes it takes Megan to eat one Popsicle -/
def minutes_per_popsicle : ℕ := 20

/-- The number of hours given in the problem -/
def hours : ℕ := 5

/-- Theorem stating that Megan can finish 15 Popsicles in 5 hours -/
theorem megan_popsicle_consumption :
  popsicles_in_5_hours = (hours * 60) / minutes_per_popsicle :=
by sorry

end megan_popsicle_consumption_l620_62003


namespace squared_sum_minus_sum_of_squares_l620_62051

theorem squared_sum_minus_sum_of_squares : (37 + 12)^2 - (37^2 + 12^2) = 888 := by
  sorry

end squared_sum_minus_sum_of_squares_l620_62051


namespace slipper_discount_percentage_l620_62094

/-- Calculates the discount percentage on slippers given the original price, 
    embroidery cost per shoe, shipping cost, and final discounted price. -/
theorem slipper_discount_percentage 
  (original_price : ℝ) 
  (embroidery_cost_per_shoe : ℝ) 
  (shipping_cost : ℝ) 
  (final_price : ℝ) : 
  original_price = 50 ∧ 
  embroidery_cost_per_shoe = 5.5 ∧ 
  shipping_cost = 10 ∧ 
  final_price = 66 →
  (original_price - (final_price - shipping_cost - 2 * embroidery_cost_per_shoe)) / original_price * 100 = 10 := by
sorry

end slipper_discount_percentage_l620_62094


namespace geometric_sequence_first_term_range_l620_62002

/-- Given an infinite geometric sequence {a_n} with common ratio q,
    if the sum of all terms is equal to q, then the range of the first term a_1 is:
    -2 < a_1 ≤ 1/4 and a_1 ≠ 0 -/
theorem geometric_sequence_first_term_range (a : ℕ → ℝ) (q : ℝ) :
  (∀ n : ℕ, a (n + 1) = q * a n) →  -- Common ratio is q
  (∃ S : ℝ, S = q ∧ S = ∑' n, a n) →  -- Sum of all terms is q
  (-2 < a 0 ∧ a 0 ≤ 1/4 ∧ a 0 ≠ 0) :=
by sorry

end geometric_sequence_first_term_range_l620_62002


namespace unique_n_for_divisibility_by_15_l620_62054

def is_divisible_by (a b : ℕ) : Prop := ∃ k, a = b * k

theorem unique_n_for_divisibility_by_15 : 
  ∃! n : ℕ, n < 10 ∧ is_divisible_by (80000 + 10000 * n + 945) 15 :=
sorry

end unique_n_for_divisibility_by_15_l620_62054


namespace count_integer_lengths_specific_triangle_l620_62087

/-- Represents a right triangle with integer side lengths -/
structure RightTriangle where
  de : ℕ
  ef : ℕ
  df : ℕ
  is_right : de^2 + ef^2 = df^2

/-- Counts the number of distinct integer lengths of line segments from E to DF -/
def count_integer_lengths (t : RightTriangle) : ℕ :=
  let max_length := max t.de t.ef
  let min_length := min t.de t.ef
  max_length - min_length + 1

/-- The main theorem -/
theorem count_integer_lengths_specific_triangle :
  ∃ (t : RightTriangle), t.de = 12 ∧ t.ef = 16 ∧ count_integer_lengths t = 5 :=
sorry

end count_integer_lengths_specific_triangle_l620_62087


namespace pages_already_read_l620_62052

/-- Theorem: Number of pages Rich has already read
Given a book with 372 pages, where Rich skipped 16 pages of maps and has 231 pages left to read,
prove that Rich has already read 125 pages. -/
theorem pages_already_read
  (total_pages : ℕ)
  (skipped_pages : ℕ)
  (pages_left : ℕ)
  (h1 : total_pages = 372)
  (h2 : skipped_pages = 16)
  (h3 : pages_left = 231) :
  total_pages - skipped_pages - pages_left = 125 := by
  sorry

end pages_already_read_l620_62052


namespace quadratic_trinomial_exists_l620_62089

/-- A quadratic trinomial satisfying the given conditions -/
def f (a c : ℝ) (m : ℝ) : ℝ := a * m^2 - a * m + c

theorem quadratic_trinomial_exists :
  ∃ (a c : ℝ), a ≠ 0 ∧ f a c 4 = 13 :=
sorry

end quadratic_trinomial_exists_l620_62089


namespace unique_two_digit_integer_l620_62037

theorem unique_two_digit_integer (t : ℕ) : 
  (t ≥ 10 ∧ t < 100) ∧ (13 * t) % 100 = 45 ↔ t = 65 := by
  sorry

end unique_two_digit_integer_l620_62037


namespace math_team_combinations_l620_62020

def number_of_teams (n_girls m_boys k_girls l_boys : ℕ) : ℕ :=
  Nat.choose n_girls k_girls * Nat.choose m_boys l_boys

theorem math_team_combinations :
  number_of_teams 5 7 3 2 = 210 := by
  sorry

end math_team_combinations_l620_62020


namespace two_digit_powers_of_three_l620_62081

theorem two_digit_powers_of_three : 
  ∃! (count : ℕ), ∃ (S : Finset ℕ), 
    (∀ n ∈ S, 10 ≤ 3^n ∧ 3^n ≤ 99) ∧ 
    (∀ n ∉ S, 3^n < 10 ∨ 99 < 3^n) ∧ 
    Finset.card S = count ∧
    count = 2 := by
  sorry

end two_digit_powers_of_three_l620_62081


namespace smallest_number_with_remainder_l620_62028

theorem smallest_number_with_remainder (n : ℤ) : 
  (n % 5 = 2) ∧ 
  ((n + 1) % 5 = 2) ∧ 
  ((n + 2) % 5 = 2) ∧ 
  (n + (n + 1) + (n + 2) = 336) →
  n = 107 := by
sorry

end smallest_number_with_remainder_l620_62028


namespace sum_of_cubes_of_roots_l620_62060

theorem sum_of_cubes_of_roots (a b c : ℝ) (h : b^2 - 4*a*c > 0) :
  let x₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let x₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  x₁^3 + x₂^3 = 95/8 :=
by
  sorry

end sum_of_cubes_of_roots_l620_62060


namespace rectangle_area_increase_l620_62030

theorem rectangle_area_increase : 
  let original_length : ℕ := 13
  let original_width : ℕ := 10
  let increase : ℕ := 2
  let original_area := original_length * original_width
  let new_length := original_length + increase
  let new_width := original_width + increase
  let new_area := new_length * new_width
  new_area - original_area = 50 := by sorry

end rectangle_area_increase_l620_62030


namespace rostov_true_supporters_l620_62076

structure Island where
  total_population : ℕ
  knights : ℕ
  liars : ℕ
  rostov_yes : ℕ
  zenit_yes : ℕ
  lokomotiv_yes : ℕ
  cska_yes : ℕ

def percentage (n : ℕ) (total : ℕ) : ℚ :=
  (n : ℚ) / (total : ℚ) * 100

theorem rostov_true_supporters (i : Island) :
  i.knights + i.liars = i.total_population →
  percentage i.rostov_yes i.total_population = 40 →
  percentage i.zenit_yes i.total_population = 30 →
  percentage i.lokomotiv_yes i.total_population = 50 →
  percentage i.cska_yes i.total_population = 0 →
  percentage i.liars i.total_population = 10 →
  percentage (i.rostov_yes - i.liars) i.total_population = 30 := by
  sorry

#check rostov_true_supporters

end rostov_true_supporters_l620_62076


namespace aaron_guitar_loan_l620_62072

/-- Calculates the total amount owed for a loan with monthly payments and interest. -/
def totalAmountOwed (monthlyPayment : ℝ) (numberOfMonths : ℕ) (interestRate : ℝ) : ℝ :=
  let totalWithoutInterest := monthlyPayment * numberOfMonths
  let interestAmount := totalWithoutInterest * interestRate
  totalWithoutInterest + interestAmount

/-- Theorem stating that given the specific conditions of Aaron's guitar purchase,
    the total amount owed is $1320. -/
theorem aaron_guitar_loan :
  totalAmountOwed 100 12 0.1 = 1320 := by
  sorry

end aaron_guitar_loan_l620_62072


namespace solution_range_l620_62070

theorem solution_range (b : ℝ) :
  (∀ x : ℝ, x^2 - b*x - 5 = 5 → (x = -2 ∨ x = 5)) →
  (∀ x : ℝ, x^2 - b*x - 5 = -1 → (x = -1 ∨ x = 4)) →
  ∃ x₁ x₂ : ℝ, 
    (x₁^2 - b*x₁ - 5 = 0 ∧ -2 < x₁ ∧ x₁ < -1) ∧
    (x₂^2 - b*x₂ - 5 = 0 ∧ 4 < x₂ ∧ x₂ < 5) ∧
    (∀ x : ℝ, x^2 - b*x - 5 = 0 → ((-2 < x ∧ x < -1) ∨ (4 < x ∧ x < 5))) := by
  sorry

end solution_range_l620_62070


namespace integral_sum_equals_pi_over_four_plus_ln_two_l620_62041

theorem integral_sum_equals_pi_over_four_plus_ln_two :
  (∫ (x : ℝ) in (0)..(1), Real.sqrt (1 - x^2)) + (∫ (x : ℝ) in (1)..(2), 1/x) = π/4 + Real.log 2 := by
  sorry

end integral_sum_equals_pi_over_four_plus_ln_two_l620_62041


namespace notebook_distribution_l620_62077

theorem notebook_distribution (S : ℕ) 
  (h1 : S > 0)
  (h2 : S * (S / 8) = (S / 2) * 16) : 
  S * (S / 8) = 512 := by
sorry

end notebook_distribution_l620_62077


namespace mencius_reading_problem_l620_62065

theorem mencius_reading_problem (total_chars : ℕ) (days : ℕ) (first_day_chars : ℕ) : 
  total_chars = 34685 →
  days = 3 →
  first_day_chars + 2 * first_day_chars + 4 * first_day_chars = total_chars →
  first_day_chars = 4955 := by
sorry

end mencius_reading_problem_l620_62065


namespace function_range_l620_62032

theorem function_range (a : ℝ) : 
  (a > 0) →
  (∀ x₁ : ℝ, ∃ x₂ : ℝ, x₂ ≥ -2 ∧ (x₁^2 - 2*x₁) > (a*x₂ + 2)) →
  a > 3/2 := by
sorry

end function_range_l620_62032


namespace chord_line_equation_l620_62026

/-- Given an ellipse and a chord midpoint, prove the equation of the line containing the chord -/
theorem chord_line_equation (x y : ℝ) :
  (x^2 / 4 + y^2 / 3 = 1) →  -- Ellipse equation
  (∃ x1 y1 x2 y2 : ℝ,        -- Endpoints of the chord
    x1^2 / 4 + y1^2 / 3 = 1 ∧
    x2^2 / 4 + y2^2 / 3 = 1 ∧
    (x1 + x2) / 2 = -1 ∧     -- Midpoint x-coordinate
    (y1 + y2) / 2 = 1) →     -- Midpoint y-coordinate
  (∃ a b c : ℝ,              -- Line equation coefficients
    a * x + b * y + c = 0 ∧  -- General form of line equation
    a = 3 ∧ b = -4 ∧ c = 7)  -- Specific coefficients for the answer
  := by sorry

end chord_line_equation_l620_62026


namespace range_of_m_l620_62013

theorem range_of_m (m : ℝ) : 
  (∃ x : ℝ, x ∈ Set.Icc 2 4 ∧ x^2 - 2*x + 5 - m < 0) → m > 5 := by
  sorry

end range_of_m_l620_62013


namespace negation_of_existential_proposition_l620_62071

theorem negation_of_existential_proposition :
  (¬∃ x₀ : ℝ, x₀ ∈ Set.Icc (-3) 3 ∧ x₀^2 + 2*x₀ + 1 ≤ 0) ↔
  (∀ x₀ : ℝ, x₀ ∈ Set.Icc (-3) 3 → x₀^2 + 2*x₀ + 1 > 0) := by sorry

end negation_of_existential_proposition_l620_62071


namespace memorial_day_weather_probability_l620_62035

/-- The probability of exactly k successes in n independent Bernoulli trials --/
def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (n.choose k : ℝ) * p^k * (1 - p)^(n - k)

/-- The number of days in the Memorial Day weekend --/
def num_days : ℕ := 5

/-- The probability of rain on each day --/
def rain_probability : ℝ := 0.8

/-- The number of desired sunny days --/
def desired_sunny_days : ℕ := 2

theorem memorial_day_weather_probability :
  binomial_probability num_days desired_sunny_days (1 - rain_probability) = 51 / 250 := by
  sorry

end memorial_day_weather_probability_l620_62035


namespace equality_or_sum_zero_l620_62056

theorem equality_or_sum_zero (a b c d : ℝ) :
  (a + b) / (b + c) = (c + d) / (d + a) →
  (a = c ∨ a + b + c + d = 0) :=
by sorry

end equality_or_sum_zero_l620_62056


namespace five_digit_base10_to_base2_sum_l620_62007

theorem five_digit_base10_to_base2_sum : ∃ (min max : ℕ),
  (∀ n : ℕ, 10000 ≤ n ∧ n ≤ 99999 →
    min ≤ (Nat.log 2 n + 1) ∧ (Nat.log 2 n + 1) ≤ max) ∧
  (max - min + 1) * (min + max) / 2 = 62 := by
  sorry

end five_digit_base10_to_base2_sum_l620_62007


namespace range_of_f_l620_62033

def f (x : ℝ) : ℝ := x^2 - 2*x

def domain : Set ℝ := {0, 1, 2, 3}

theorem range_of_f : 
  {y | ∃ x ∈ domain, f x = y} = {-1, 0, 3} := by sorry

end range_of_f_l620_62033


namespace bob_profit_l620_62039

/-- Calculates the profit from breeding and selling show dogs -/
def dogBreedingProfit (numDogs : ℕ) (dogCost : ℕ) (numPuppies : ℕ) (puppyPrice : ℕ) 
                      (foodVaccinationCost : ℕ) (advertisingCost : ℕ) : ℤ :=
  (numPuppies * puppyPrice : ℤ) - (numDogs * dogCost + foodVaccinationCost + advertisingCost)

theorem bob_profit : 
  dogBreedingProfit 2 250 6 350 500 150 = 950 := by
  sorry

end bob_profit_l620_62039


namespace work_efficiency_ratio_l620_62009

/-- Given a road that can be repaired by A in 4 days or by B in 5 days,
    the ratio of A's work efficiency to B's work efficiency is 5/4. -/
theorem work_efficiency_ratio (road : ℝ) (days_A days_B : ℕ) 
  (h_A : road / days_A = road / 4)
  (h_B : road / days_B = road / 5) :
  (road / days_A) / (road / days_B) = 5 / 4 := by
  sorry

end work_efficiency_ratio_l620_62009


namespace f_properties_l620_62097

-- Define the function f
def f (x a b : ℝ) : ℝ := |x + a| - |x - b|

-- Main theorem
theorem f_properties (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  -- Part I: Solution set for f(x) > 2 when a = 1 and b = 2
  (∀ x, f x 1 2 > 2 ↔ x > 3/2) ∧
  -- Part II: If max(f) = 3, then min(1/a + 2/b) = (3 + 2√2)/3
  (∃ x, ∀ y, f y a b ≤ f x a b) ∧ (∀ y, f y a b ≤ 3) →
    ∀ a' b', a' > 0 → b' > 0 → 1/a' + 2/b' ≥ (3 + 2*Real.sqrt 2)/3 ∧
    ∃ a'' b'', a'' > 0 ∧ b'' > 0 ∧ 1/a'' + 2/b'' = (3 + 2*Real.sqrt 2)/3 :=
by sorry


end f_properties_l620_62097


namespace bridge_length_l620_62062

/-- The length of a bridge given train specifications and crossing time -/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 160 ∧ 
  train_speed_kmh = 45 ∧ 
  crossing_time = 30 →
  ∃ (bridge_length : ℝ), bridge_length = 215 := by
  sorry

end bridge_length_l620_62062


namespace prime_quadratic_roots_l620_62038

theorem prime_quadratic_roots (p : ℕ) : 
  Prime p → 
  (∃ x y : ℤ, x^2 + p*x - 222*p = 0 ∧ y^2 + p*y - 222*p = 0) → 
  31 < p ∧ p ≤ 41 := by
sorry

end prime_quadratic_roots_l620_62038


namespace inequality_proof_l620_62000

theorem inequality_proof (x y z w : ℝ) 
  (h_positive : x > 0 ∧ y > 0 ∧ z > 0 ∧ w > 0) 
  (h_sum_squares : x^2 + y^2 + z^2 + w^2 = 1) : 
  x^2 * y * z * w + x * y^2 * z * w + x * y * z^2 * w + x * y * z * w^2 ≤ 1/8 := by
sorry

end inequality_proof_l620_62000


namespace one_thirds_in_eleven_halves_l620_62064

theorem one_thirds_in_eleven_halves : (11 / 2) / (1 / 3) = 33 / 2 := by
  sorry

end one_thirds_in_eleven_halves_l620_62064


namespace fraction_sum_equals_thirteen_fourths_l620_62085

theorem fraction_sum_equals_thirteen_fourths (a b : ℝ) (h1 : a = 3) (h2 : b = 1) :
  5 / (a + b) + 2 = 13 / 4 := by
  sorry

end fraction_sum_equals_thirteen_fourths_l620_62085


namespace factorization_equality_l620_62025

theorem factorization_equality (m a : ℝ) : 3 * m * a^2 - 6 * m * a + 3 * m = 3 * m * (a - 1)^2 := by
  sorry

end factorization_equality_l620_62025


namespace function_equality_l620_62075

theorem function_equality (f g h k : ℝ → ℝ) (a b : ℝ) 
  (h1 : ∀ x, f x = (x - 1) * g x + 3)
  (h2 : ∀ x, f x = (x + 1) * h x + 1)
  (h3 : ∀ x, f x = (x^2 - 1) * k x + a * x + b) :
  a = 1 ∧ b = 2 := by
  sorry

end function_equality_l620_62075


namespace diamond_calculation_l620_62012

-- Define the diamond operation
def diamond (a b : ℚ) : ℚ := a - 1 / b

-- Theorem statement
theorem diamond_calculation :
  (diamond (diamond 2 3) 4) - (diamond 2 (diamond 3 4)) = -29/132 :=
by sorry

end diamond_calculation_l620_62012


namespace geric_bills_count_geric_bills_proof_l620_62031

theorem geric_bills_count : ℕ → ℕ → ℕ → Prop :=
  fun geric_bills kyla_bills jessa_bills =>
    (geric_bills = 2 * kyla_bills) ∧
    (kyla_bills = jessa_bills - 2) ∧
    (jessa_bills - 3 = 7) →
    geric_bills = 16

-- The proof goes here
theorem geric_bills_proof : ∃ g k j, geric_bills_count g k j :=
  sorry

end geric_bills_count_geric_bills_proof_l620_62031


namespace garden_roller_length_l620_62079

theorem garden_roller_length :
  let diameter : ℝ := 1.4
  let area_covered : ℝ := 66
  let revolutions : ℝ := 5
  let π : ℝ := 22 / 7
  let radius : ℝ := diameter / 2
  let length : ℝ := (area_covered / revolutions) / (2 * π * radius)
  length = 2.1 := by sorry

end garden_roller_length_l620_62079


namespace pizza_recipe_water_amount_l620_62016

theorem pizza_recipe_water_amount :
  ∀ (water flour salt : ℚ),
    flour = 16 →
    salt = (1/2) * flour →
    water + flour + salt = 34 →
    water = 10 :=
by sorry

end pizza_recipe_water_amount_l620_62016


namespace unique_base_solution_l620_62069

/-- Convert a number from base b to decimal --/
def to_decimal (digits : List Nat) (b : Nat) : Nat :=
  digits.foldl (fun acc d => acc * b + d) 0

/-- Check if the equation holds for a given base --/
def equation_holds (b : Nat) : Prop :=
  to_decimal [2, 5, 1] b + to_decimal [1, 7, 4] b = to_decimal [4, 3, 5] b

theorem unique_base_solution :
  ∃! b : Nat, b > 1 ∧ equation_holds b :=
sorry

end unique_base_solution_l620_62069


namespace triangle_properties_l620_62018

noncomputable def angle_A (A B C : ℝ) (a b c : ℝ) : Prop :=
  a * Real.cos C + (1/2) * c = b ∧ a = 1 → A = Real.pi / 3

def perimeter_range (A B C : ℝ) (a b c : ℝ) : Prop :=
  a * Real.cos C + (1/2) * c = b ∧ a = 1 →
  let l := a + b + c
  2 < l ∧ l ≤ 3

theorem triangle_properties (A B C : ℝ) (a b c : ℝ) :
  angle_A A B C a b c ∧ perimeter_range A B C a b c :=
sorry

end triangle_properties_l620_62018
