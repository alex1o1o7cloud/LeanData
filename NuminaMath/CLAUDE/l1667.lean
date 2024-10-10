import Mathlib

namespace average_score_theorem_l1667_166765

def perfect_score : ℕ := 30
def deduction_per_mistake : ℕ := 2

def madeline_mistakes : ℕ := 2
def leo_mistakes : ℕ := 2 * madeline_mistakes
def brent_mistakes : ℕ := leo_mistakes + 1
def nicholas_mistakes : ℕ := 3 * madeline_mistakes

def brent_score : ℕ := 25
def nicholas_score : ℕ := brent_score - 5

def student_score (mistakes : ℕ) : ℕ := perfect_score - mistakes * deduction_per_mistake

theorem average_score_theorem : 
  (student_score madeline_mistakes + student_score leo_mistakes + brent_score + nicholas_score) / 4 = 83 / 4 := by
  sorry

end average_score_theorem_l1667_166765


namespace average_stickers_per_pack_l1667_166724

def sticker_counts : List ℕ := [5, 8, 0, 12, 15, 20, 22, 25, 30, 35]

def num_packs : ℕ := 10

theorem average_stickers_per_pack :
  (sticker_counts.sum : ℚ) / num_packs = 17.2 := by
  sorry

end average_stickers_per_pack_l1667_166724


namespace inequality_proof_l1667_166711

theorem inequality_proof (a₁ a₂ a₃ a₄ a₅ : ℝ) 
  (h₁ : a₁ > 1) (h₂ : a₂ > 1) (h₃ : a₃ > 1) (h₄ : a₄ > 1) (h₅ : a₅ > 1) : 
  16 * (a₁ * a₂ * a₃ * a₄ * a₅ + 1) > (1 + a₁) * (1 + a₂) * (1 + a₃) * (1 + a₄) * (1 + a₅) := by
  sorry

end inequality_proof_l1667_166711


namespace caroline_citrus_drinks_l1667_166775

/-- The number of citrus drinks Caroline can make from a given number of oranges -/
def citrus_drinks (oranges : ℕ) : ℕ :=
  8 * oranges / 3

/-- Theorem stating that Caroline can make 56 citrus drinks from 21 oranges -/
theorem caroline_citrus_drinks :
  citrus_drinks 21 = 56 := by
  sorry

end caroline_citrus_drinks_l1667_166775


namespace john_pennies_l1667_166737

/-- Given that Kate has 223 pennies, John has more pennies than Kate,
    and the difference between their pennies is 165,
    prove that John has 388 pennies. -/
theorem john_pennies (kate_pennies : ℕ) (john_more : ℕ) (difference : ℕ)
    (h1 : kate_pennies = 223)
    (h2 : john_more > kate_pennies)
    (h3 : john_more - kate_pennies = difference)
    (h4 : difference = 165) :
    john_more = 388 := by
  sorry

end john_pennies_l1667_166737


namespace probability_between_C_and_D_l1667_166744

/-- Given a line segment AB with points C, D, and E such that AB = 4AD = 4BE
    and AD = DC = CE = EB, the probability of a random point on AB being
    between C and D is 1/4. -/
theorem probability_between_C_and_D (A B C D E : ℝ) : 
  A < C ∧ C < D ∧ D < E ∧ E < B →
  B - A = 4 * (D - A) →
  B - A = 4 * (B - E) →
  D - A = C - D →
  C - D = E - C →
  E - C = B - E →
  (D - C) / (B - A) = 1 / 4 := by
  sorry

end probability_between_C_and_D_l1667_166744


namespace average_score_five_subjects_l1667_166746

theorem average_score_five_subjects 
  (avg_three : ℝ) 
  (score_four : ℝ) 
  (score_five : ℝ) 
  (h1 : avg_three = 92) 
  (h2 : score_four = 90) 
  (h3 : score_five = 95) : 
  (3 * avg_three + score_four + score_five) / 5 = 92.2 := by
sorry

end average_score_five_subjects_l1667_166746


namespace problem_1_problem_2_problem_3_problem_4_l1667_166704

-- Problem 1
theorem problem_1 : 23 + (-16) - (-7) = 14 := by sorry

-- Problem 2
theorem problem_2 : (3/4 - 7/8 - 5/12) * (-24) = 13 := by sorry

-- Problem 3
theorem problem_3 : (7/4 - 7/8 - 7/12) / (-7/8) + (-7/8) / (7/4 - 7/8 - 7/12) = -10/3 := by sorry

-- Problem 4
theorem problem_4 : -(1^4) - (1 - 0.5) * (1/3) * (2 - (-3)^2) = 1/6 := by sorry

end problem_1_problem_2_problem_3_problem_4_l1667_166704


namespace sum_of_divisors_330_l1667_166734

def sum_of_divisors (n : ℕ) : ℕ :=
  (Finset.filter (· ∣ n) (Finset.range (n + 1))).sum id

theorem sum_of_divisors_330 : sum_of_divisors 330 = 864 := by
  sorry

end sum_of_divisors_330_l1667_166734


namespace bakery_bread_rolls_l1667_166713

/-- Given a bakery with a total of 90 items, 19 croissants, and 22 bagels,
    prove that the number of bread rolls is 49. -/
theorem bakery_bread_rolls :
  let total_items : ℕ := 90
  let croissants : ℕ := 19
  let bagels : ℕ := 22
  let bread_rolls : ℕ := total_items - croissants - bagels
  bread_rolls = 49 := by
  sorry

end bakery_bread_rolls_l1667_166713


namespace sqrt_35_between_5_and_6_l1667_166756

theorem sqrt_35_between_5_and_6 : 5 < Real.sqrt 35 ∧ Real.sqrt 35 < 6 := by
  sorry

end sqrt_35_between_5_and_6_l1667_166756


namespace correct_prediction_probability_l1667_166778

def num_monday_classes : ℕ := 5
def num_tuesday_classes : ℕ := 6
def total_classes : ℕ := num_monday_classes + num_tuesday_classes
def correct_predictions : ℕ := 7
def monday_correct_predictions : ℕ := 3

theorem correct_prediction_probability :
  (Nat.choose num_monday_classes monday_correct_predictions * (1/2)^num_monday_classes) *
  (Nat.choose num_tuesday_classes (correct_predictions - monday_correct_predictions) * (1/2)^num_tuesday_classes) /
  (Nat.choose total_classes correct_predictions * (1/2)^total_classes) = 5/11 := by
sorry

end correct_prediction_probability_l1667_166778


namespace arithmetic_sequence_l1667_166796

theorem arithmetic_sequence (a b c : ℝ) (h1 : b ≠ 0) 
  (h2 : ∃ x : ℝ, bx^2 - 4*b*x + 2*(a+c) = 0 ∧ (∀ y : ℝ, bx^2 - 4*b*x + 2*(a+c) = 0 → y = x)) :
  b - a = c - b := by
sorry

end arithmetic_sequence_l1667_166796


namespace function_max_value_l1667_166790

theorem function_max_value (x : ℝ) (h : x < 5/4) :
  (4 * x - 2 + 1 / (4 * x - 5)) ≤ 1 :=
by sorry

end function_max_value_l1667_166790


namespace simplify_expression_l1667_166764

theorem simplify_expression (a : ℝ) (ha : a ≠ 0) :
  (1 - a / (a + 1)) / ((a^2 - a) / (a^2 - 1)) = 1 / a :=
by
  sorry

end simplify_expression_l1667_166764


namespace max_value_implies_m_l1667_166701

/-- Given that the maximum value of f(x) = sin(x + π/2) + cos(x - π/2) + m is 2√2, prove that m = √2 -/
theorem max_value_implies_m (f : ℝ → ℝ) (m : ℝ) 
  (h : ∀ x, f x = Real.sin (x + π/2) + Real.cos (x - π/2) + m) 
  (h_max : ∃ x₀, ∀ x, f x ≤ f x₀ ∧ f x₀ = 2 * Real.sqrt 2) : 
  m = Real.sqrt 2 := by
  sorry

end max_value_implies_m_l1667_166701


namespace monster_perimeter_l1667_166769

theorem monster_perimeter (r : ℝ) (θ : ℝ) : 
  r = 2 → θ = 270 * π / 180 → 
  r * θ + 2 * r = 3 * π + 4 := by sorry

end monster_perimeter_l1667_166769


namespace sqrt_nine_equals_three_l1667_166750

theorem sqrt_nine_equals_three : Real.sqrt 9 = 3 := by
  sorry

end sqrt_nine_equals_three_l1667_166750


namespace restaurant_bill_problem_l1667_166763

theorem restaurant_bill_problem (kate_bill : ℝ) (bob_discount : ℝ) (kate_discount : ℝ) (total_after_discount : ℝ) :
  kate_bill = 25 →
  bob_discount = 0.05 →
  kate_discount = 0.02 →
  total_after_discount = 53 →
  ∃ bob_bill : ℝ, 
    bob_bill * (1 - bob_discount) + kate_bill * (1 - kate_discount) = total_after_discount ∧
    bob_bill = 30 := by
  sorry

end restaurant_bill_problem_l1667_166763


namespace smallest_solution_congruence_l1667_166733

theorem smallest_solution_congruence :
  ∃ (x : ℕ), x > 0 ∧ (5 * x) % 31 = 17 % 31 ∧ ∀ (y : ℕ), y > 0 → (5 * y) % 31 = 17 % 31 → x ≤ y :=
by sorry

end smallest_solution_congruence_l1667_166733


namespace solve_equation_l1667_166777

theorem solve_equation (x y : ℝ) : y = 1 / (4 * x + 2) → y = 2 → x = -3/8 := by
  sorry

end solve_equation_l1667_166777


namespace no_equidistant_points_l1667_166774

/-- A circle in a plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in a plane, represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Configuration of a circle and two parallel tangent lines -/
structure CircleWithTangents where
  circle : Circle
  tangent1 : Line
  tangent2 : Line
  d : ℝ  -- distance from circle center to each tangent

/-- Predicate for a point being equidistant from a circle and a line -/
def isEquidistant (p : ℝ × ℝ) (c : Circle) (l : Line) : Prop := sorry

/-- Main theorem: No equidistant points exist when d > r -/
theorem no_equidistant_points (config : CircleWithTangents) 
  (h : config.d > config.circle.radius) :
  ¬∃ p : ℝ × ℝ, isEquidistant p config.circle config.tangent1 ∧ 
                isEquidistant p config.circle config.tangent2 := by
  sorry

end no_equidistant_points_l1667_166774


namespace sum_of_roots_l1667_166725

theorem sum_of_roots (k m : ℝ) (y₁ y₂ : ℝ) 
  (h₁ : 2 * y₁^2 - k * y₁ - m = 0)
  (h₂ : 2 * y₂^2 - k * y₂ - m = 0)
  (h₃ : y₁ ≠ y₂) : 
  y₁ + y₂ = k / 2 := by
  sorry

end sum_of_roots_l1667_166725


namespace no_complete_set_in_matrix_l1667_166708

/-- Definition of the matrix A --/
def A (n : ℕ) (i j : ℕ) : ℕ :=
  if (i + j - 1) % n = 0 then n else (i + j - 1) % n

/-- Theorem statement --/
theorem no_complete_set_in_matrix (n : ℕ) (h_even : Even n) (h_pos : 0 < n) :
  ¬ ∃ σ : Fin n → Fin n, Function.Bijective σ ∧ (∀ i : Fin n, A n i.val (σ i).val = i.val + 1) :=
sorry

end no_complete_set_in_matrix_l1667_166708


namespace powers_of_two_start_with_any_digits_l1667_166794

theorem powers_of_two_start_with_any_digits (A m : ℕ) : 
  ∃ n : ℕ+, (10 ^ m * A : ℝ) < (2 : ℝ) ^ (n : ℝ) ∧ (2 : ℝ) ^ (n : ℝ) < (10 ^ m * (A + 1) : ℝ) := by
  sorry

end powers_of_two_start_with_any_digits_l1667_166794


namespace unique_polynomial_l1667_166745

/-- A polynomial satisfying the given conditions -/
def p : ℝ → ℝ := λ x => x^2 + 1

/-- The theorem stating that p is the unique polynomial satisfying the conditions -/
theorem unique_polynomial :
  (p 3 = 10) ∧
  (∀ x y : ℝ, p x * p y = p x + p y + p (x * y) - 3) ∧
  (∀ q : ℝ → ℝ, (q 3 = 10 ∧ ∀ x y : ℝ, q x * q y = q x + q y + q (x * y) - 3) → q = p) :=
by sorry

end unique_polynomial_l1667_166745


namespace value_of_w_l1667_166791

theorem value_of_w (j p t q s w : ℝ) 
  (h1 : j = 0.75 * p)
  (h2 : j = 0.80 * t)
  (h3 : t = p - (w / 100) * p)
  (h4 : q = 1.15 * p)
  (h5 : q = 0.70 * j)
  (h6 : s = 1.40 * t)
  (h7 : s = 0.90 * q) :
  w = 6.25 := by
sorry

end value_of_w_l1667_166791


namespace gcd_difference_is_perfect_square_l1667_166739

theorem gcd_difference_is_perfect_square (x y z : ℕ) (h : (1 : ℚ) / x - (1 : ℚ) / y = (1 : ℚ) / z) :
  ∃ (k : ℕ), Nat.gcd x (Nat.gcd y z) * (y - x) = k * k := by
  sorry

end gcd_difference_is_perfect_square_l1667_166739


namespace jacob_final_score_l1667_166738

/-- Represents the score for a quiz contest -/
structure QuizScore where
  correct : ℕ
  incorrect : ℕ
  unanswered : ℕ
  correct_points : ℚ
  incorrect_points : ℚ

/-- Calculates the final score for a quiz contest -/
def final_score (qs : QuizScore) : ℚ :=
  qs.correct * qs.correct_points + qs.incorrect * qs.incorrect_points

/-- Jacob's quiz score -/
def jacob_score : QuizScore :=
  { correct := 20
    incorrect := 10
    unanswered := 5
    correct_points := 1
    incorrect_points := -1/2 }

/-- Theorem: Jacob's final score is 15 points -/
theorem jacob_final_score :
  final_score jacob_score = 15 := by sorry

end jacob_final_score_l1667_166738


namespace gretchen_desk_work_time_l1667_166716

/-- Represents the ratio of walking time to sitting time -/
def walkingSittingRatio : ℚ := 10 / 90

/-- Represents the total walking time in minutes -/
def totalWalkingTime : ℕ := 40

/-- Represents the time spent working at the desk in hours -/
def deskWorkTime : ℚ := 6

theorem gretchen_desk_work_time :
  walkingSittingRatio * (deskWorkTime * 60) = totalWalkingTime :=
sorry

end gretchen_desk_work_time_l1667_166716


namespace ratio_q_p_l1667_166785

def total_slips : ℕ := 40
def distinct_numbers : ℕ := 10
def slips_per_number : ℕ := 4
def drawn_slips : ℕ := 4

def p : ℚ := (distinct_numbers * 1) / Nat.choose total_slips drawn_slips
def q : ℚ := (Nat.choose distinct_numbers 2 * Nat.choose slips_per_number 2 * Nat.choose slips_per_number 2) / Nat.choose total_slips drawn_slips

theorem ratio_q_p : q / p = 162 := by
  sorry

end ratio_q_p_l1667_166785


namespace cylinder_cone_sphere_volumes_l1667_166700

/-- Given a cylinder with volume 150π cm³, prove that:
    1. A cone with the same base radius and height as the cylinder has a volume of 50π cm³
    2. A sphere with the same radius as the cylinder has a volume of 200π cm³ -/
theorem cylinder_cone_sphere_volumes (r h : ℝ) (hr : r > 0) (hh : h > 0) : 
  π * r^2 * h = 150 * π →
  (1/3 : ℝ) * π * r^2 * h = 50 * π ∧ 
  (4/3 : ℝ) * π * r^3 = 200 * π := by
  sorry

#check cylinder_cone_sphere_volumes

end cylinder_cone_sphere_volumes_l1667_166700


namespace greatest_prime_factor_of_5_cubed_plus_10_to_4_l1667_166748

theorem greatest_prime_factor_of_5_cubed_plus_10_to_4 :
  ∃ p : ℕ, p.Prime ∧ p ∣ (5^3 + 10^4) ∧ ∀ q : ℕ, q.Prime → q ∣ (5^3 + 10^4) → q ≤ p :=
by sorry

end greatest_prime_factor_of_5_cubed_plus_10_to_4_l1667_166748


namespace tenth_grader_average_score_l1667_166795

/-- Represents a chess tournament between 9th and 10th graders -/
structure ChessTournament where
  ninth_graders : ℕ
  tenth_graders : ℕ
  tournament_points : ℕ

/-- The number of 10th graders is 10 times the number of 9th graders -/
axiom tenth_grader_count (t : ChessTournament) : t.tenth_graders = 10 * t.ninth_graders

/-- Each player plays every other player exactly once -/
axiom total_games (t : ChessTournament) : t.tournament_points = (t.ninth_graders + t.tenth_graders) * (t.ninth_graders + t.tenth_graders - 1) / 2

/-- The average score of a 10th grader is 10 points -/
theorem tenth_grader_average_score (t : ChessTournament) :
  t.tournament_points / t.tenth_graders = 10 :=
sorry

end tenth_grader_average_score_l1667_166795


namespace triangle_strike_interval_l1667_166758

def is_factor (a b : ℕ) : Prop := b % a = 0

theorem triangle_strike_interval (cymbal_interval triangle_interval coincidence_interval : ℕ) :
  cymbal_interval = 7 →
  is_factor triangle_interval coincidence_interval →
  is_factor cymbal_interval coincidence_interval →
  coincidence_interval = 14 →
  triangle_interval ≠ cymbal_interval →
  triangle_interval > 0 →
  triangle_interval = 2 :=
by
  sorry

end triangle_strike_interval_l1667_166758


namespace sum_digits_2_5_power_1997_l1667_166727

/-- The number of decimal digits in a positive integer -/
def num_digits (x : ℕ) : ℕ := sorry

/-- Theorem: The sum of the number of decimal digits in 2^1997 and 5^1997 is 1998 -/
theorem sum_digits_2_5_power_1997 : num_digits (2^1997) + num_digits (5^1997) = 1998 := by
  sorry

end sum_digits_2_5_power_1997_l1667_166727


namespace absolute_value_inequality_l1667_166757

theorem absolute_value_inequality (x : ℝ) :
  x ≠ 1 →
  (|(2 * x - 1) / (x - 1)| > 2) ↔ (x > 3/4 ∧ x < 1) ∨ x > 1 :=
by sorry

end absolute_value_inequality_l1667_166757


namespace person_speed_in_mph_l1667_166798

/-- Prove that a person crossing a 2500-meter street in 8 minutes has a speed of approximately 11.65 miles per hour. -/
theorem person_speed_in_mph : ∃ (speed : ℝ), abs (speed - 11.65) < 0.01 :=
  let street_length : ℝ := 2500 -- meters
  let crossing_time : ℝ := 8 -- minutes
  let meters_per_mile : ℝ := 1609.34
  let minutes_per_hour : ℝ := 60
  let distance_miles : ℝ := street_length / meters_per_mile
  let time_hours : ℝ := crossing_time / minutes_per_hour
  let speed : ℝ := distance_miles / time_hours
by
  -- Proof goes here
  sorry

end person_speed_in_mph_l1667_166798


namespace probability_one_white_ball_l1667_166722

/-- The probability of drawing exactly one white ball when randomly selecting two balls from a bag containing 2 white and 3 black balls -/
theorem probability_one_white_ball (total_balls : ℕ) (white_balls : ℕ) (black_balls : ℕ) :
  total_balls = white_balls + black_balls →
  total_balls = 5 →
  white_balls = 2 →
  black_balls = 3 →
  (Nat.choose white_balls 1 * Nat.choose black_balls 1 : ℚ) / Nat.choose total_balls 2 = 3/5 := by
  sorry

end probability_one_white_ball_l1667_166722


namespace teachers_not_adjacent_l1667_166723

/-- The number of ways to arrange 2 teachers and 3 students in a row, 
    such that the teachers are not adjacent -/
def arrangement_count : ℕ := 72

/-- The number of teachers -/
def teacher_count : ℕ := 2

/-- The number of students -/
def student_count : ℕ := 3

theorem teachers_not_adjacent : 
  arrangement_count = 
    (Nat.factorial student_count) * (Nat.factorial (student_count + 1)) / 
    (Nat.factorial (student_count + 1 - teacher_count)) := by
  sorry

end teachers_not_adjacent_l1667_166723


namespace smallest_base_for_150_base_6_works_for_150_smallest_base_is_6_l1667_166728

theorem smallest_base_for_150 :
  ∀ b : ℕ, b > 0 → (b^2 ≤ 150 ∧ 150 < b^3) → b ≥ 6 :=
by sorry

theorem base_6_works_for_150 :
  6^2 ≤ 150 ∧ 150 < 6^3 :=
by sorry

theorem smallest_base_is_6 :
  ∃! b : ℕ, b > 0 ∧ b^2 ≤ 150 ∧ 150 < b^3 ∧ ∀ c : ℕ, (c > 0 ∧ c^2 ≤ 150 ∧ 150 < c^3) → c ≥ b :=
by sorry

end smallest_base_for_150_base_6_works_for_150_smallest_base_is_6_l1667_166728


namespace tv_price_difference_l1667_166718

def budget : ℕ := 1000
def initial_discount : ℕ := 100
def additional_discount_percent : ℕ := 20

theorem tv_price_difference : 
  let price_after_initial_discount := budget - initial_discount
  let additional_discount := price_after_initial_discount * additional_discount_percent / 100
  let final_price := price_after_initial_discount - additional_discount
  budget - final_price = 280 := by sorry

end tv_price_difference_l1667_166718


namespace even_function_tangent_slope_l1667_166736

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 
  if x > 0 then a * x^2 / (x + 1) else a * x^2 / (-x + 1)

theorem even_function_tangent_slope (a : ℝ) :
  (∀ x, f a x = f a (-x)) →
  (∀ x > 0, f a x = a * x^2 / (x + 1)) →
  (deriv (f a)) (-1) = -1 →
  a = 4/3 := by sorry

end even_function_tangent_slope_l1667_166736


namespace larry_initial_amount_l1667_166719

def larry_problem (initial_amount lunch_cost brother_gift current_amount : ℕ) : Prop :=
  initial_amount = lunch_cost + brother_gift + current_amount

theorem larry_initial_amount :
  ∃ (initial_amount : ℕ), larry_problem initial_amount 5 2 15 ∧ initial_amount = 22 := by
  sorry

end larry_initial_amount_l1667_166719


namespace harkamal_payment_l1667_166792

/-- The total amount paid by Harkamal to the shopkeeper for grapes and mangoes -/
def total_amount (grape_quantity : ℕ) (grape_rate : ℕ) (mango_quantity : ℕ) (mango_rate : ℕ) : ℕ :=
  grape_quantity * grape_rate + mango_quantity * mango_rate

/-- Theorem stating that Harkamal paid 1100 to the shopkeeper -/
theorem harkamal_payment :
  total_amount 8 70 9 60 = 1100 := by
  sorry

#eval total_amount 8 70 9 60

end harkamal_payment_l1667_166792


namespace min_value_z_l1667_166749

/-- The minimum value of z = x - y given the specified constraints -/
theorem min_value_z (x y : ℝ) (h1 : x + y - 2 ≥ 0) (h2 : x ≤ 4) (h3 : y ≤ 5) :
  ∀ (x' y' : ℝ), x' + y' - 2 ≥ 0 → x' ≤ 4 → y' ≤ 5 → x - y ≤ x' - y' :=
by sorry

end min_value_z_l1667_166749


namespace physics_marks_l1667_166729

theorem physics_marks (P C M : ℝ) 
  (h1 : (P + C + M) / 3 = 85)
  (h2 : (P + M) / 2 = 90)
  (h3 : (P + C) / 2 = 70) :
  P = 65 := by
sorry

end physics_marks_l1667_166729


namespace sequence_properties_l1667_166779

def a : ℕ → ℕ
  | 0 => 2
  | n + 1 => (a n)^2 - a n + 1

theorem sequence_properties :
  (∀ m n : ℕ, m ≠ n → Nat.gcd (a m) (a n) = 1) ∧
  (∑' k : ℕ, (1 : ℝ) / (a k)) = 1 := by
  sorry

end sequence_properties_l1667_166779


namespace target_hit_probability_l1667_166715

theorem target_hit_probability (p_A p_B : ℝ) (h_A : p_A = 9/10) (h_B : p_B = 8/9) :
  1 - (1 - p_A) * (1 - p_B) = 89/90 := by
  sorry

end target_hit_probability_l1667_166715


namespace hilt_book_profit_l1667_166782

/-- The difference in total amount between selling and buying books -/
def book_profit (num_books : ℕ) (buy_price sell_price : ℚ) : ℚ :=
  num_books * sell_price - num_books * buy_price

/-- Theorem stating the profit from buying and selling books -/
theorem hilt_book_profit :
  book_profit 15 11 25 = 210 := by
  sorry

end hilt_book_profit_l1667_166782


namespace cone_volume_l1667_166742

/-- Given a cone with base radius 1 and lateral surface area √5π, its volume is 2π/3 -/
theorem cone_volume (r h : ℝ) (lateral_area : ℝ) : 
  r = 1 → 
  lateral_area = Real.sqrt 5 * Real.pi → 
  2 * Real.pi * r * h = lateral_area → 
  (1/3) * Real.pi * r^2 * h = (2/3) * Real.pi := by
  sorry

end cone_volume_l1667_166742


namespace smallest_non_prime_digit_divisible_by_all_single_digit_primes_l1667_166703

def is_prime (n : ℕ) : Prop := sorry

def single_digit_primes : List ℕ := [2, 3, 5, 7]

def digits (n : ℕ) : List ℕ := sorry

theorem smallest_non_prime_digit_divisible_by_all_single_digit_primes :
  ∃ (N : ℕ),
    (∀ d ∈ digits N, ¬ is_prime d) ∧
    (∀ p ∈ single_digit_primes, N % p = 0) ∧
    (∀ m < N, ¬(∀ d ∈ digits m, ¬ is_prime d) ∨ ¬(∀ p ∈ single_digit_primes, m % p = 0)) ∧
    N = 840 :=
sorry

end smallest_non_prime_digit_divisible_by_all_single_digit_primes_l1667_166703


namespace simplify_fraction_l1667_166735

theorem simplify_fraction : 3 * (11 / 4) * (16 / -55) = -12 / 5 := by
  sorry

end simplify_fraction_l1667_166735


namespace sum_of_three_numbers_l1667_166726

theorem sum_of_three_numbers (a b c : ℝ) 
  (sum1 : a + b = 35)
  (sum2 : b + c = 47)
  (sum3 : c + a = 58) : 
  a + b + c = 70 := by
sorry

end sum_of_three_numbers_l1667_166726


namespace quadratic_factorization_and_perfect_square_discriminant_l1667_166760

/-- A quadratic expression of the form 15x^2 + ax + 15 can be factored into two linear binomial 
factors with integer coefficients, and its discriminant is a perfect square when a = 34 -/
theorem quadratic_factorization_and_perfect_square_discriminant :
  ∃ (m n p q : ℤ), 
    (15 : ℤ) * m * p = 15 ∧ 
    m * q + n * p = 34 ∧ 
    n * q = 15 ∧
    ∃ (k : ℤ), 34^2 - 4 * 15 * 15 = k^2 := by
  sorry

end quadratic_factorization_and_perfect_square_discriminant_l1667_166760


namespace calories_consumed_l1667_166717

/-- Given a package of candy with 3 servings of 120 calories each,
    prove that eating half the package results in consuming 180 calories. -/
theorem calories_consumed (servings : ℕ) (calories_per_serving : ℕ) (portion_eaten : ℚ) : 
  servings = 3 → 
  calories_per_serving = 120 → 
  portion_eaten = 1/2 →
  (↑servings * ↑calories_per_serving : ℚ) * portion_eaten = 180 := by
sorry

end calories_consumed_l1667_166717


namespace pencil_eraser_cost_l1667_166743

theorem pencil_eraser_cost : ∃ (p e : ℕ), 
  15 * p + 5 * e = 200 ∧ 
  p ≥ 2 * e ∧ 
  p + e = 14 := by
sorry

end pencil_eraser_cost_l1667_166743


namespace lasagna_pieces_sum_to_six_l1667_166732

/-- Represents the number of lasagna pieces each person eats -/
structure LasagnaPieces where
  manny : ℚ
  aaron : ℚ
  kai : ℚ
  raphael : ℚ
  lisa : ℚ

/-- Calculates the total number of lasagna pieces eaten -/
def total_pieces (pieces : LasagnaPieces) : ℚ :=
  pieces.manny + pieces.aaron + pieces.kai + pieces.raphael + pieces.lisa

/-- Theorem stating the total number of lasagna pieces equals 6 -/
theorem lasagna_pieces_sum_to_six : ∃ (pieces : LasagnaPieces), 
  pieces.manny = 1 ∧ 
  pieces.aaron = 0 ∧ 
  pieces.kai = 2 * pieces.manny ∧ 
  pieces.raphael = pieces.manny / 2 ∧ 
  pieces.lisa = 2 + pieces.raphael ∧ 
  total_pieces pieces = 6 := by
  sorry

end lasagna_pieces_sum_to_six_l1667_166732


namespace gcd_polynomial_and_multiple_l1667_166767

theorem gcd_polynomial_and_multiple (b : ℤ) (h : ∃ k : ℤ, b = 342 * k) :
  Nat.gcd (Int.natAbs (5*b^3 + b^2 + 8*b + 38)) (Int.natAbs b) = 38 := by
  sorry

end gcd_polynomial_and_multiple_l1667_166767


namespace product_of_fractions_equals_one_l1667_166786

theorem product_of_fractions_equals_one :
  (7 / 4 : ℚ) * (8 / 14 : ℚ) * (21 / 12 : ℚ) * (16 / 28 : ℚ) *
  (49 / 28 : ℚ) * (24 / 42 : ℚ) * (63 / 36 : ℚ) * (32 / 56 : ℚ) = 1 := by
  sorry

end product_of_fractions_equals_one_l1667_166786


namespace ellipse_line_theorem_l1667_166799

/-- An ellipse with given properties -/
structure Ellipse where
  center : ℝ × ℝ
  left_focus : ℝ × ℝ
  right_vertex : ℝ × ℝ

/-- A line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Theorem: For an ellipse with given properties, if a line intersects a chord at its midpoint A(1, 1/2), then the line has equation x + 2y - 2 = 0 -/
theorem ellipse_line_theorem (e : Ellipse) (l : Line) :
  e.center = (0, 0) ∧
  e.left_focus = (-Real.sqrt 3, 0) ∧
  e.right_vertex = (2, 0) ∧
  (∃ (B C : ℝ × ℝ), B ≠ C ∧ (1, 1/2) = ((B.1 + C.1)/2, (B.2 + C.2)/2) ∧
    (∀ (x y : ℝ), x^2/4 + y^2 = 1 → l.a * x + l.b * y + l.c = 0 → (x, y) = B ∨ (x, y) = C)) →
  l.a = 1 ∧ l.b = 2 ∧ l.c = -2 :=
sorry

end ellipse_line_theorem_l1667_166799


namespace share_difference_l1667_166780

theorem share_difference (total amount_a amount_b amount_c : ℕ) : 
  total = 120 →
  amount_b = 20 →
  amount_a + amount_b + amount_c = total →
  amount_a = amount_c - 20 →
  amount_a - amount_b = 20 :=
by
  sorry

end share_difference_l1667_166780


namespace solution_when_m_is_one_solution_for_general_m_l1667_166759

-- Define the inequality
def inequality (m x : ℝ) : Prop := (2*m - m*x)/2 > x/2 - 1

-- Theorem for part 1
theorem solution_when_m_is_one :
  ∀ x : ℝ, inequality 1 x ↔ x < 2 := by sorry

-- Theorem for part 2
theorem solution_for_general_m :
  ∀ m x : ℝ, m ≠ -1 →
    (inequality m x ↔ (m > -1 ∧ x < 2) ∨ (m < -1 ∧ x > 2)) := by sorry

end solution_when_m_is_one_solution_for_general_m_l1667_166759


namespace coffee_maker_capacity_l1667_166710

/-- A cylindrical coffee maker with capacity x cups contains 30 cups when 25% full. -/
theorem coffee_maker_capacity (x : ℝ) (h : 0.25 * x = 30) : x = 120 := by
  sorry

end coffee_maker_capacity_l1667_166710


namespace range_of_x_l1667_166766

theorem range_of_x (x : ℝ) : 
  (|x - 1| + |x - 2| = 1) → (1 ≤ x ∧ x ≤ 2) := by
  sorry

end range_of_x_l1667_166766


namespace square_calculation_l1667_166720

theorem square_calculation :
  (41 ^ 2 = 40 ^ 2 + 81) ∧ (39 ^ 2 = 40 ^ 2 - 79) := by
  sorry

end square_calculation_l1667_166720


namespace escalator_problem_solution_l1667_166773

/-- The time taken to descend an escalator under different conditions -/
def EscalatorProblem (s : ℝ) : Prop :=
  let t_standing := (3/2 : ℝ)  -- Time taken when standing on moving escalator
  let t_running_stationary := (1 : ℝ)  -- Time taken when running on stationary escalator
  let v_escalator := s / t_standing  -- Speed of escalator
  let v_running := s / t_running_stationary  -- Speed of running
  let v_combined := v_escalator + v_running  -- Combined speed
  let t_running_moving := s / v_combined  -- Time taken when running on moving escalator
  t_running_moving = (3/5 : ℝ)

/-- Theorem stating the solution to the escalator problem -/
theorem escalator_problem_solution :
  ∀ s : ℝ, s > 0 → EscalatorProblem s :=
by
  sorry

end escalator_problem_solution_l1667_166773


namespace sock_pair_probability_l1667_166781

def number_of_socks : ℕ := 10
def number_of_colors : ℕ := 5
def socks_per_color : ℕ := 2
def socks_drawn : ℕ := 6

theorem sock_pair_probability :
  let total_combinations := Nat.choose number_of_socks socks_drawn
  let pair_combinations := Nat.choose number_of_colors 2 * Nat.choose (number_of_colors - 2) 2 * 4
  (pair_combinations : ℚ) / total_combinations = 4 / 7 := by
  sorry

end sock_pair_probability_l1667_166781


namespace partition_theorem_l1667_166751

theorem partition_theorem (a m n : ℝ) (h1 : 0 < m) (h2 : 0 < n) (h3 : n < 1/m) :
  let x := a * (n - 1) / (m * n - 1)
  let first_partition := (m * x, a - m * x)
  let second_partition := (x, n * (a - m * x))
  first_partition.1 + first_partition.2 = a ∧
  second_partition.1 + second_partition.2 = a ∧
  first_partition.1 = m * second_partition.1 ∧
  second_partition.2 = n * first_partition.2 :=
by sorry

end partition_theorem_l1667_166751


namespace d_range_l1667_166714

/-- Circle C with center (3,4) and radius 1 -/
def CircleC (x y : ℝ) : Prop := (x - 3)^2 + (y - 4)^2 = 1

/-- Point A -/
def A : ℝ × ℝ := (0, 1)

/-- Point B -/
def B : ℝ × ℝ := (0, -1)

/-- Distance squared between two points -/
def distanceSquared (p1 p2 : ℝ × ℝ) : ℝ :=
  (p1.1 - p2.1)^2 + (p1.2 - p2.2)^2

/-- The function d for a point P on the circle -/
def d (x y : ℝ) : ℝ := distanceSquared (x, y) A + distanceSquared (x, y) B

theorem d_range :
  ∀ x y : ℝ, CircleC x y → 34 ≤ d x y ∧ d x y ≤ 74 :=
sorry

end d_range_l1667_166714


namespace equilateral_triangle_side_l1667_166768

/-- The length of one side of the largest equilateral triangle created from a 78 cm string -/
def triangle_side_length : ℝ := 26

/-- The total length of the string used to create the triangle -/
def string_length : ℝ := 78

/-- Theorem stating that the length of one side of the largest equilateral triangle
    created from a 78 cm string is 26 cm -/
theorem equilateral_triangle_side (s : ℝ) :
  s = triangle_side_length ↔ s * 3 = string_length :=
by sorry

end equilateral_triangle_side_l1667_166768


namespace exists_h_not_divisible_l1667_166747

theorem exists_h_not_divisible : ∃ h : ℝ, ∀ n : ℕ, 
  ¬(⌊h * 1969^n⌋ ∣ ⌊h * 1969^(n-1)⌋) :=
sorry

end exists_h_not_divisible_l1667_166747


namespace average_visitors_theorem_l1667_166730

/-- Calculates the average number of visitors per day in a 30-day month starting with a Sunday -/
def averageVisitorsPerDay (sundayVisitors : ℕ) (otherDayVisitors : ℕ) : ℚ :=
  let totalDays : ℕ := 30
  let sundays : ℕ := 4
  let otherDays : ℕ := totalDays - sundays
  let totalVisitors : ℕ := sundayVisitors * sundays + otherDayVisitors * otherDays
  (totalVisitors : ℚ) / totalDays

theorem average_visitors_theorem (sundayVisitors otherDayVisitors : ℕ) 
  (h1 : sundayVisitors = 510) (h2 : otherDayVisitors = 240) : 
  averageVisitorsPerDay sundayVisitors otherDayVisitors = 276 := by
  sorry

#eval averageVisitorsPerDay 510 240

end average_visitors_theorem_l1667_166730


namespace geometric_sequence_fourth_term_l1667_166731

theorem geometric_sequence_fourth_term (a : ℕ → ℝ) (h1 : ∀ n, a (n + 1) / a n = a 2 / a 1) 
    (h2 : a 2 = 9) (h3 : a 5 = 243) : a 4 = 81 := by
  sorry

end geometric_sequence_fourth_term_l1667_166731


namespace lcm_gcd_difference_times_min_l1667_166770

theorem lcm_gcd_difference_times_min (a b : ℕ) (ha : a = 8) (hb : b = 12) :
  (Nat.lcm a b - Nat.gcd a b) * min a b = 160 := by
  sorry

end lcm_gcd_difference_times_min_l1667_166770


namespace hurricane_damage_calculation_l1667_166761

/-- Calculates the total hurricane damage in Canadian dollars, including a recovery tax -/
theorem hurricane_damage_calculation (damage_usd : ℝ) (assets_cad : ℝ) (exchange_rate : ℝ) (tax_rate : ℝ) :
  damage_usd = 45000000 →
  assets_cad = 15000000 →
  exchange_rate = 1.25 →
  tax_rate = 0.1 →
  let damage_cad := damage_usd * exchange_rate + assets_cad
  let total_with_tax := damage_cad * (1 + tax_rate)
  total_with_tax = 78375000 := by
sorry

end hurricane_damage_calculation_l1667_166761


namespace potato_distribution_l1667_166702

theorem potato_distribution (num_people : ℕ) (bag_weight : ℝ) (bag_cost : ℝ) (total_cost : ℝ) :
  num_people = 40 →
  bag_weight = 20 →
  bag_cost = 5 →
  total_cost = 15 →
  (total_cost / bag_cost * bag_weight) / num_people = 1.5 := by
  sorry

end potato_distribution_l1667_166702


namespace last_digit_fifth_power_l1667_166712

theorem last_digit_fifth_power (R : ℤ) : 10 ∣ (R^5 - R) := by
  sorry

end last_digit_fifth_power_l1667_166712


namespace balloon_problem_l1667_166771

/-- Represents the balloon counts for a person -/
structure BalloonCount where
  red : Nat
  blue : Nat

/-- Calculates the total cost of balloons -/
def totalCost (redCount blue_count : Nat) (redCost blueCost : Nat) : Nat :=
  redCount * redCost + blue_count * blueCost

/-- Theorem statement for the balloon problem -/
theorem balloon_problem 
  (fred sam dan : BalloonCount)
  (redCost blueCost : Nat)
  (h1 : fred = ⟨10, 5⟩)
  (h2 : sam = ⟨46, 20⟩)
  (h3 : dan = ⟨16, 12⟩)
  (h4 : redCost = 10)
  (h5 : blueCost = 5) :
  let totalRed := fred.red + sam.red + dan.red
  let totalBlue := fred.blue + sam.blue + dan.blue
  let totalCost := totalCost totalRed totalBlue redCost blueCost
  totalRed = 72 ∧ totalBlue = 37 ∧ totalCost = 905 := by
  sorry


end balloon_problem_l1667_166771


namespace exponent_equality_l1667_166783

theorem exponent_equality (a : ℝ) (m n : ℕ) (h : 0 < a) :
  (a^12 = (a^3)^m) ∧ (a^12 = a^2 * a^n) → m = 4 ∧ n = 10 := by
  sorry

end exponent_equality_l1667_166783


namespace min_votes_class_president_l1667_166753

/-- Represents the minimum number of votes needed to win an election -/
def min_votes_to_win (total_votes : ℕ) (num_candidates : ℕ) : ℕ :=
  (total_votes / num_candidates) + 1

/-- Theorem: In an election with 4 candidates and 61 votes, the minimum number of votes to win is 16 -/
theorem min_votes_class_president : min_votes_to_win 61 4 = 16 := by
  sorry

end min_votes_class_president_l1667_166753


namespace chris_money_before_birthday_l1667_166784

/-- The amount of money Chris had before his birthday. -/
def money_before_birthday (grandmother_gift aunt_uncle_gift parents_gift total_now : ℕ) : ℕ :=
  total_now - (grandmother_gift + aunt_uncle_gift + parents_gift)

/-- Theorem stating that Chris had $239 before his birthday. -/
theorem chris_money_before_birthday :
  money_before_birthday 25 20 75 359 = 239 := by
  sorry

end chris_money_before_birthday_l1667_166784


namespace sine_cosine_identity_l1667_166707

theorem sine_cosine_identity : 
  Real.sin (50 * π / 180) * Real.cos (170 * π / 180) - 
  Real.sin (40 * π / 180) * Real.sin (170 * π / 180) = 
  -Real.sqrt 3 / 2 := by
  sorry

end sine_cosine_identity_l1667_166707


namespace linear_function_k_value_l1667_166772

/-- A linear function passing through a specific point -/
def linear_function (k : ℝ) (x : ℝ) : ℝ := k * x + 3

/-- The point through which the function passes -/
def point : ℝ × ℝ := (2, 5)

theorem linear_function_k_value :
  ∃ k : ℝ, linear_function k (point.1) = point.2 ∧ k = 1 := by
  sorry

end linear_function_k_value_l1667_166772


namespace min_value_sum_of_reciprocals_l1667_166741

theorem min_value_sum_of_reciprocals (a b c d e f : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) 
  (pos_d : 0 < d) (pos_e : 0 < e) (pos_f : 0 < f)
  (sum_eq_10 : a + b + c + d + e + f = 10) : 
  (1/a + 9/b + 25/c + 49/d + 81/e + 121/f) ≥ 129.6 := by
sorry

end min_value_sum_of_reciprocals_l1667_166741


namespace angies_age_equation_angie_is_eight_years_old_l1667_166787

/-- Angie's age in years -/
def angiesAge : ℕ := 8

/-- The equation representing the given condition -/
theorem angies_age_equation : 2 * angiesAge + 4 = 20 := by sorry

/-- Proof that Angie's age is 8 years old -/
theorem angie_is_eight_years_old : angiesAge = 8 := by sorry

end angies_age_equation_angie_is_eight_years_old_l1667_166787


namespace third_smallest_prime_cubed_to_fourth_l1667_166740

-- Define a function to get the nth prime number
def nthPrime (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem third_smallest_prime_cubed_to_fourth : (nthPrime 3) ^ 3 ^ 4 = 244140625 := by
  sorry

end third_smallest_prime_cubed_to_fourth_l1667_166740


namespace repeating_decimal_proof_l1667_166793

theorem repeating_decimal_proof : ∃ (n : ℕ), n ≥ 10 ∧ n < 100 ∧ 
  (48 * (n / 99 : ℚ) - 48 * (n / 100 : ℚ) = 1 / 5) ∧
  (100 * (n / 99 : ℚ) - (n / 99 : ℚ) = n) :=
by sorry

end repeating_decimal_proof_l1667_166793


namespace triangle_inequality_third_side_length_l1667_166789

theorem triangle_inequality (a b c : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 → 
  (a + b > c ∧ b + c > a ∧ c + a > b) ↔ 
  (a < b + c ∧ b < c + a ∧ c < a + b) :=
sorry

theorem third_side_length (x : ℝ) : 
  x > 0 → 
  5 > 0 → 
  8 > 0 → 
  (5 + 8 > x ∧ 8 + x > 5 ∧ x + 5 > 8) → 
  (3 < x ∧ x < 13) :=
sorry

end triangle_inequality_third_side_length_l1667_166789


namespace line_intersects_ellipse_l1667_166709

/-- Given that x₁ and x₂ are extremal points of f(x) = (1/3)ax³ - (1/2)ax² - x,
    prove that the line passing through A(x₁, 1/x₁) and B(x₂, 1/x₂)
    intersects the ellipse x²/2 + y² = 1 --/
theorem line_intersects_ellipse (a : ℝ) (x₁ x₂ : ℝ) :
  (x₁ ≠ x₂) →
  (∀ x, (a*x^2 - a*x - 1 = 0) ↔ (x = x₁ ∨ x = x₂)) →
  ∃ x y : ℝ, (y - 1/x₁ = (1/x₂ - 1/x₁)/(x₂ - x₁) * (x - x₁)) ∧
             (x^2/2 + y^2 = 1) :=
by sorry

end line_intersects_ellipse_l1667_166709


namespace current_speed_l1667_166754

/-- Given a man's speed with and against a current, calculate the speed of the current. -/
theorem current_speed (speed_with_current speed_against_current : ℝ) 
  (h1 : speed_with_current = 15)
  (h2 : speed_against_current = 10) :
  ∃ (current_speed : ℝ), current_speed = 2.5 ∧ 
    speed_with_current = speed_against_current + 2 * current_speed :=
by sorry

end current_speed_l1667_166754


namespace two_digit_primes_with_prime_digits_l1667_166797

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

def is_two_digit (n : ℕ) : Prop := n ≥ 10 ∧ n ≤ 99

def digits_are_prime (n : ℕ) : Prop :=
  is_prime (n / 10) ∧ is_prime (n % 10)

theorem two_digit_primes_with_prime_digits :
  ∃! (s : Finset ℕ), 
    (∀ n ∈ s, is_two_digit n ∧ is_prime n ∧ digits_are_prime n) ∧
    (∀ n, is_two_digit n → is_prime n → digits_are_prime n → n ∈ s) ∧
    s.card = 4 :=
sorry

end two_digit_primes_with_prime_digits_l1667_166797


namespace joe_age_difference_l1667_166752

theorem joe_age_difference (joe_age : ℕ) (james_age : ℕ) : joe_age = 22 → 2 * (joe_age + 8) = 3 * (james_age + 8) → joe_age - james_age = 10 := by
  sorry

end joe_age_difference_l1667_166752


namespace total_food_amount_l1667_166705

-- Define the number of boxes
def num_boxes : ℕ := 388

-- Define the amount of food per box in kilograms
def food_per_box : ℕ := 2

-- Theorem to prove the total amount of food
theorem total_food_amount : num_boxes * food_per_box = 776 := by
  sorry

end total_food_amount_l1667_166705


namespace consecutive_integers_product_l1667_166762

theorem consecutive_integers_product (a b c d e : ℤ) : 
  (a + b + c + d + e) / 5 = 17 ∧ 
  d = 12 ∧ 
  e = 22 ∧ 
  (∃ n : ℤ, a = n ∧ b = n + 1 ∧ c = n + 2) →
  a * b * c = 4896 := by
sorry

end consecutive_integers_product_l1667_166762


namespace seventy_seven_base4_non_consecutive_digits_l1667_166706

/-- Converts a decimal number to its base 4 representation -/
def toBase4 (n : ℕ) : List ℕ :=
  sorry

/-- Counts the number of non-consecutive digits in a list of digits -/
def countNonConsecutiveDigits (digits : List ℕ) : ℕ :=
  sorry

theorem seventy_seven_base4_non_consecutive_digits :
  let base4Repr := toBase4 77
  countNonConsecutiveDigits base4Repr = 3 :=
by sorry

end seventy_seven_base4_non_consecutive_digits_l1667_166706


namespace mass_of_copper_sulfate_pentahydrate_l1667_166788

-- Define the constants
def volume : ℝ := 0.5 -- in L
def concentration : ℝ := 1 -- in mol/L
def molar_mass : ℝ := 250 -- in g/mol

-- Theorem statement
theorem mass_of_copper_sulfate_pentahydrate (volume concentration molar_mass : ℝ) : 
  volume * concentration * molar_mass = 125 := by
  sorry

#check mass_of_copper_sulfate_pentahydrate

end mass_of_copper_sulfate_pentahydrate_l1667_166788


namespace hawkeye_fewer_maine_coons_l1667_166721

/-- Proves that Hawkeye owns 1 fewer Maine Coon than Gordon --/
theorem hawkeye_fewer_maine_coons (jamie_persians jamie_maine_coons gordon_persians gordon_maine_coons hawkeye_maine_coons : ℕ) :
  jamie_persians = 4 →
  jamie_maine_coons = 2 →
  gordon_persians = jamie_persians / 2 →
  gordon_maine_coons = jamie_maine_coons + 1 →
  jamie_persians + jamie_maine_coons + gordon_persians + gordon_maine_coons + hawkeye_maine_coons = 13 →
  gordon_maine_coons - hawkeye_maine_coons = 1 := by
  sorry

end hawkeye_fewer_maine_coons_l1667_166721


namespace equation_solution_l1667_166755

theorem equation_solution :
  ∃ y : ℝ, (7 * (4 * y + 3) - 3 = -3 * (2 - 9 * y)) ∧ (y = -24) := by
  sorry

end equation_solution_l1667_166755


namespace min_students_same_choice_l1667_166776

theorem min_students_same_choice (n : ℕ) (m : ℕ) (h1 : n = 45) (h2 : m = 6) :
  ∃ k : ℕ, k ≥ 16 ∧ k * m ≥ n := by
  sorry

end min_students_same_choice_l1667_166776
