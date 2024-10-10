import Mathlib

namespace tomato_price_is_five_l2280_228097

/-- Represents the price per pound of tomatoes -/
def tomato_price : ℝ := sorry

/-- The number of pounds of tomatoes bought -/
def tomato_pounds : ℝ := 2

/-- The number of pounds of apples bought -/
def apple_pounds : ℝ := 5

/-- The price per pound of apples -/
def apple_price : ℝ := 6

/-- The total amount spent -/
def total_spent : ℝ := 40

/-- Theorem stating that the price per pound of tomatoes is $5 -/
theorem tomato_price_is_five :
  tomato_price * tomato_pounds + apple_price * apple_pounds = total_spent →
  tomato_price = 5 := by sorry

end tomato_price_is_five_l2280_228097


namespace intersection_P_Q_l2280_228073

-- Define the sets P and Q
def P : Set ℝ := {x | |x| < 2}
def Q : Set ℝ := {y | ∃ x, y = Real.sqrt (x^2) + 1}

-- State the theorem
theorem intersection_P_Q :
  P ∩ Q = {x | 1 ≤ x ∧ x < 2} := by sorry

end intersection_P_Q_l2280_228073


namespace stick_cutting_probability_l2280_228024

theorem stick_cutting_probability (stick_length : Real) (mark_position : Real) 
  (h1 : stick_length = 2)
  (h2 : mark_position = 0.6)
  (h3 : 0 < mark_position ∧ mark_position < stick_length) :
  let cut_range := stick_length - mark_position
  let valid_cut := min (stick_length / 4) cut_range
  (valid_cut / cut_range) = 5/14 := by
  sorry


end stick_cutting_probability_l2280_228024


namespace remainder_11_power_4001_mod_13_l2280_228046

theorem remainder_11_power_4001_mod_13 : 11^4001 % 13 = 7 := by
  sorry

end remainder_11_power_4001_mod_13_l2280_228046


namespace angle_D_measure_l2280_228048

-- Define the hexagon and its angles
structure Hexagon where
  A : ℝ
  B : ℝ
  C : ℝ
  D : ℝ
  E : ℝ
  F : ℝ

-- Define the properties of the hexagon
def is_convex_hexagon_with_properties (h : Hexagon) : Prop :=
  -- Angles A, B, and C are congruent
  h.A = h.B ∧ h.B = h.C
  -- Angles D and E are congruent
  ∧ h.D = h.E
  -- Angle A is 50 degrees less than angle D
  ∧ h.A + 50 = h.D
  -- Sum of angles in a hexagon is 720 degrees
  ∧ h.A + h.B + h.C + h.D + h.E + h.F = 720

-- Theorem statement
theorem angle_D_measure (h : Hexagon) 
  (h_props : is_convex_hexagon_with_properties h) : 
  h.D = 153.33 := by
  sorry

end angle_D_measure_l2280_228048


namespace range_of_a_l2280_228042

/-- The range of a satisfying the given conditions -/
theorem range_of_a : ∀ a : ℝ, 
  ((∀ x : ℝ, x^2 - 4*a*x + 3*a^2 < 0 → x^2 + 2*x - 8 > 0) ∧ 
   (∃ x : ℝ, x^2 + 2*x - 8 > 0 ∧ x^2 - 4*a*x + 3*a^2 ≥ 0)) ↔ 
  (a ≤ -4 ∨ a ≥ 2 ∨ a = 0) :=
by sorry

end range_of_a_l2280_228042


namespace symmetry_of_lines_l2280_228009

/-- Given two lines in a 2D plane represented by their equations,
    this function returns true if they are symmetric with respect to the line x+y=0 -/
def are_symmetric_lines (l1 l2 : ℝ → ℝ → Prop) : Prop :=
  ∀ x y, l1 x y ↔ l2 (-y) (-x)

/-- The equation of the original line -/
def original_line (x y : ℝ) : Prop := 3 * x - 4 * y + 5 = 0

/-- The equation of the supposedly symmetric line -/
def symmetric_line (x y : ℝ) : Prop := 4 * x - 3 * y - 5 = 0

/-- Theorem stating that the symmetric_line is indeed symmetric to the original_line
    with respect to the line x+y=0 -/
theorem symmetry_of_lines : are_symmetric_lines original_line symmetric_line :=
sorry

end symmetry_of_lines_l2280_228009


namespace charlyn_visible_area_l2280_228035

/-- The length of one side of the square in kilometers -/
def square_side : ℝ := 5

/-- The visibility range in kilometers -/
def visibility_range : ℝ := 1

/-- The area of the region Charlyn can see during her walk -/
noncomputable def visible_area : ℝ :=
  (square_side + 2 * visibility_range) ^ 2 - (square_side - 2 * visibility_range) ^ 2 + Real.pi * visibility_range ^ 2

theorem charlyn_visible_area :
  ‖visible_area - 43.14‖ < 0.01 :=
sorry

end charlyn_visible_area_l2280_228035


namespace circle_radius_from_longest_chord_l2280_228084

theorem circle_radius_from_longest_chord (c : Real) (h : c > 0) : 
  ∃ (r : Real), r > 0 ∧ r = c / 2 := by sorry

end circle_radius_from_longest_chord_l2280_228084


namespace min_value_of_expression_l2280_228094

theorem min_value_of_expression (x : ℝ) : (x^2 + 8) / Real.sqrt (x^2 + 4) ≥ 4 ∧
  ∃ x₀ : ℝ, (x₀^2 + 8) / Real.sqrt (x₀^2 + 4) = 4 := by
  sorry

end min_value_of_expression_l2280_228094


namespace impossible_arrangement_l2280_228044

-- Define a 3x3 grid as a function from (Fin 3 × Fin 3) to ℕ
def Grid := Fin 3 → Fin 3 → ℕ

-- Define a predicate to check if a number is between 1 and 9
def InRange (n : ℕ) : Prop := 1 ≤ n ∧ n ≤ 9

-- Define a predicate to check if a grid contains all numbers from 1 to 9
def ContainsAllNumbers (g : Grid) : Prop :=
  ∀ n, InRange n → ∃ i j, g i j = n

-- Define a predicate to check if the product of numbers in a row is a multiple of 4
def RowProductMultipleOf4 (g : Grid) : Prop :=
  ∀ i, (g i 0) * (g i 1) * (g i 2) % 4 = 0

-- Define a predicate to check if the product of numbers in a column is a multiple of 4
def ColProductMultipleOf4 (g : Grid) : Prop :=
  ∀ j, (g 0 j) * (g 1 j) * (g 2 j) % 4 = 0

-- The main theorem
theorem impossible_arrangement : ¬∃ (g : Grid),
  ContainsAllNumbers g ∧ 
  RowProductMultipleOf4 g ∧ 
  ColProductMultipleOf4 g :=
sorry

end impossible_arrangement_l2280_228044


namespace leo_has_more_leo_excess_marbles_l2280_228087

/-- The number of marbles Ben has -/
def ben_marbles : ℕ := 56

/-- The total number of marbles in the jar -/
def total_marbles : ℕ := 132

/-- Leo's marbles are the difference between the total and Ben's marbles -/
def leo_marbles : ℕ := total_marbles - ben_marbles

/-- The statement that Leo has more marbles than Ben -/
theorem leo_has_more : leo_marbles > ben_marbles := by sorry

/-- The main theorem: Leo has 20 more marbles than Ben -/
theorem leo_excess_marbles : leo_marbles - ben_marbles = 20 := by sorry

end leo_has_more_leo_excess_marbles_l2280_228087


namespace probability_of_three_successes_l2280_228004

def n : ℕ := 7
def k : ℕ := 3
def p : ℚ := 1/3

theorem probability_of_three_successes :
  Nat.choose n k * p^k * (1-p)^(n-k) = 560/2187 := by
  sorry

end probability_of_three_successes_l2280_228004


namespace cubic_minus_linear_factorization_l2280_228082

theorem cubic_minus_linear_factorization (a : ℝ) : a^3 - a = a * (a + 1) * (a - 1) := by
  sorry

end cubic_minus_linear_factorization_l2280_228082


namespace range_of_m_l2280_228065

open Set

-- Define the universal set U as ℝ
def U := ℝ

-- Define set A
def A : Set ℝ := {x | -7 ≤ 2*x - 1 ∧ 2*x - 1 ≤ 7}

-- Define set B (parameterized by m)
def B (m : ℝ) : Set ℝ := {x | m - 1 ≤ x ∧ x ≤ 3*m - 2}

-- Theorem statement
theorem range_of_m (m : ℝ) : A ∩ B m = B m → m ≤ 2 := by
  sorry

end range_of_m_l2280_228065


namespace smallest_n_for_zoe_play_l2280_228071

def can_zoe_play (n : ℕ) : Prop :=
  ∀ (yvan_first : ℕ) (h_yvan_first : yvan_first ≤ n),
    ∃ (zoe_first : ℕ) (zoe_last : ℕ),
      zoe_first < zoe_last ∧
      zoe_last ≤ n ∧
      zoe_first ≠ yvan_first ∧
      zoe_last ≠ yvan_first ∧
      ∀ (yvan_second : ℕ) (yvan_second_last : ℕ),
        yvan_second < yvan_second_last ∧
        yvan_second_last ≤ n ∧
        yvan_second ≠ yvan_first ∧
        yvan_second_last ≠ yvan_first ∧
        yvan_second ∉ Set.Icc zoe_first zoe_last ∧
        yvan_second_last ∉ Set.Icc zoe_first zoe_last →
        ∃ (zoe_second : ℕ) (zoe_second_last : ℕ),
          zoe_second < zoe_second_last ∧
          zoe_second_last ≤ n ∧
          zoe_second ∉ Set.Icc zoe_first zoe_last ∪ Set.Icc yvan_second yvan_second_last ∧
          zoe_second_last ∉ Set.Icc zoe_first zoe_last ∪ Set.Icc yvan_second yvan_second_last ∧
          zoe_second_last - zoe_second = 3

theorem smallest_n_for_zoe_play :
  (∀ k < 14, ¬ can_zoe_play k) ∧ can_zoe_play 14 := by sorry

end smallest_n_for_zoe_play_l2280_228071


namespace purely_imaginary_z_reciprocal_l2280_228053

theorem purely_imaginary_z_reciprocal (m : ℝ) :
  let z : ℂ := m^2 - 1 + (m + 1) * I
  (∃ (y : ℝ), z = y * I) → 2 / z = -I :=
by
  sorry

end purely_imaginary_z_reciprocal_l2280_228053


namespace marble_draw_probability_l2280_228067

/-- The probability of drawing one red marble followed by one blue marble without replacement -/
theorem marble_draw_probability (red blue yellow : ℕ) (h_red : red = 4) (h_blue : blue = 3) (h_yellow : yellow = 6) :
  (red : ℚ) / (red + blue + yellow) * blue / (red + blue + yellow - 1) = 1 / 13 := by sorry

end marble_draw_probability_l2280_228067


namespace initial_boarders_count_l2280_228062

theorem initial_boarders_count (initial_boarders day_students new_boarders : ℕ) : 
  initial_boarders > 0 ∧ 
  day_students > 0 ∧
  new_boarders = 66 ∧
  initial_boarders * 12 = day_students * 5 ∧
  (initial_boarders + new_boarders) * 2 = day_students * 1 →
  initial_boarders = 330 := by
sorry

end initial_boarders_count_l2280_228062


namespace one_greater_than_digit_squares_l2280_228089

def digit_squares_sum (n : ℕ) : ℕ :=
  (n.digits 10).map (λ d => d^2) |>.sum

theorem one_greater_than_digit_squares : {n : ℕ | n > 0 ∧ n = digit_squares_sum n + 1} = {35, 75} := by
  sorry

end one_greater_than_digit_squares_l2280_228089


namespace square_of_sum_l2280_228076

theorem square_of_sum (a b : ℝ) : (a + b)^2 = a^2 + 2*a*b + b^2 := by
  sorry

end square_of_sum_l2280_228076


namespace sqrt_23_bound_l2280_228085

theorem sqrt_23_bound : 4.5 < Real.sqrt 23 ∧ Real.sqrt 23 < 5 := by
  sorry

end sqrt_23_bound_l2280_228085


namespace smallest_base_perfect_square_l2280_228011

theorem smallest_base_perfect_square : 
  ∃ (b : ℕ), b > 3 ∧ b = 5 ∧ ∀ (x : ℕ), x > 3 ∧ x < b → ¬∃ (y : ℕ), 4 * x + 5 = y ^ 2 :=
by sorry

end smallest_base_perfect_square_l2280_228011


namespace cyclic_fraction_product_l2280_228061

theorem cyclic_fraction_product (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (h1 : (x + y) / z = (y + z) / x) (h2 : (y + z) / x = (z + x) / y) :
  ((x + y) * (y + z) * (z + x)) / (x * y * z) = -1 ∨
  ((x + y) * (y + z) * (z + x)) / (x * y * z) = 8 := by
sorry

end cyclic_fraction_product_l2280_228061


namespace donna_episodes_per_weekday_l2280_228005

theorem donna_episodes_per_weekday : 
  ∀ (weekday_episodes : ℕ),
  weekday_episodes > 0 →
  5 * weekday_episodes + 2 * (3 * weekday_episodes) = 88 →
  weekday_episodes = 8 := by
sorry

end donna_episodes_per_weekday_l2280_228005


namespace parabola_y_intercept_l2280_228020

/-- A parabola passing through two given points has a specific y-intercept -/
theorem parabola_y_intercept (a b : ℝ) : 
  (6 = 2^2 + 2*a + b) ∧ (-14 = (-2)^2 + (-2)*a + b) → b = -8 := by
  sorry

end parabola_y_intercept_l2280_228020


namespace marbles_lost_l2280_228031

theorem marbles_lost (initial : ℝ) (remaining : ℝ) (lost : ℝ) : 
  initial = 9.5 → remaining = 4.25 → lost = initial - remaining → lost = 5.25 := by
  sorry

end marbles_lost_l2280_228031


namespace sqrt_sum_problem_l2280_228022

theorem sqrt_sum_problem (y : ℝ) (h : Real.sqrt (64 - y^2) - Real.sqrt (36 - y^2) = 4) :
  Real.sqrt (64 - y^2) + Real.sqrt (36 - y^2) = 7 := by
  sorry

end sqrt_sum_problem_l2280_228022


namespace grocery_store_distance_l2280_228003

theorem grocery_store_distance (distance_house_to_park : ℝ) 
                               (distance_park_to_store : ℝ) 
                               (total_distance : ℝ) : ℝ := by
  have h1 : distance_house_to_park = 5 := by sorry
  have h2 : distance_park_to_store = 3 := by sorry
  have h3 : total_distance = 16 := by sorry
  
  let distance_store_to_house := total_distance - distance_house_to_park - distance_park_to_store
  
  have h4 : distance_store_to_house = 
            total_distance - distance_house_to_park - distance_park_to_store := by rfl
  
  exact distance_store_to_house

end grocery_store_distance_l2280_228003


namespace max_correct_answers_l2280_228037

/-- Represents an exam score. -/
structure ExamScore where
  correct : ℕ
  incorrect : ℕ
  unanswered : ℕ
  totalQuestions : ℕ
  score : ℤ

/-- Checks if the exam score is valid according to the rules. -/
def ExamScore.isValid (e : ExamScore) : Prop :=
  e.correct + e.incorrect + e.unanswered = e.totalQuestions ∧
  6 * e.correct - 3 * e.incorrect = e.score

/-- Theorem: The maximum number of correct answers for the given exam conditions is 14. -/
theorem max_correct_answers :
  ∀ e : ExamScore,
    e.totalQuestions = 25 →
    e.score = 57 →
    e.isValid →
    e.correct ≤ 14 ∧
    ∃ e' : ExamScore, e'.totalQuestions = 25 ∧ e'.score = 57 ∧ e'.isValid ∧ e'.correct = 14 :=
by sorry

end max_correct_answers_l2280_228037


namespace complex_arithmetic_simplification_l2280_228059

theorem complex_arithmetic_simplification :
  (7 - 3 * Complex.I) - 3 * (2 + 4 * Complex.I) = 1 - 15 * Complex.I := by
  sorry

end complex_arithmetic_simplification_l2280_228059


namespace expression_equality_l2280_228096

-- Define lg as the base-10 logarithm
noncomputable def lg (x : ℝ) := Real.log x / Real.log 10

-- State the theorem
theorem expression_equality : (-8)^(1/3) + π^0 + lg 4 + lg 25 = 1 := by sorry

end expression_equality_l2280_228096


namespace probability_two_heads_in_four_tosses_l2280_228055

-- Define the number of coin tosses
def n : ℕ := 4

-- Define the number of heads we're looking for
def k : ℕ := 2

-- Define the probability of getting heads on a single toss
def p : ℚ := 1/2

-- Define the probability of getting tails on a single toss
def q : ℚ := 1/2

-- Define the binomial coefficient function
def binomial_coefficient (n k : ℕ) : ℕ := Nat.choose n k

-- Define the probability of getting exactly k heads in n tosses
def probability_k_heads (n k : ℕ) (p q : ℚ) : ℚ :=
  (binomial_coefficient n k : ℚ) * p^k * q^(n-k)

-- Theorem statement
theorem probability_two_heads_in_four_tosses :
  probability_k_heads n k p q = 3/8 := by
  sorry

end probability_two_heads_in_four_tosses_l2280_228055


namespace part_1_part_2_l2280_228095

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x * (Real.exp x - 1) - a * x^2

-- Define the derivative of f
def f' (a : ℝ) (x : ℝ) : ℝ := Real.exp x + x * Real.exp x - 1 - 2 * a * x

-- Theorem for part 1
theorem part_1 (a : ℝ) : f' a 1 = 2 * Real.exp 1 - 2 → a = 1/2 := by sorry

-- Define the specific function f with a = 1/2
def f_half (x : ℝ) : ℝ := x * (Real.exp x - 1) - (1/2) * x^2

-- Define the derivative of f_half
def f_half' (x : ℝ) : ℝ := (x + 1) * (Real.exp x - 1)

-- Theorem for part 2
theorem part_2 (m : ℝ) : 
  (∀ x ∈ Set.Ioo (2*m - 3) (3*m - 2), f_half' x > 0) ↔ 
  (m ∈ Set.Ioc (-1) (1/3) ∪ Set.Ici (3/2)) := by sorry

end

end part_1_part_2_l2280_228095


namespace sequence_sum_l2280_228033

theorem sequence_sum (P Q R S T U V : ℝ) : 
  R = 7 ∧
  P + Q + R = 36 ∧
  Q + R + S = 36 ∧
  R + S + T = 36 ∧
  S + T + U = 36 ∧
  T + U + V = 36 →
  P + V = 29 := by
sorry

end sequence_sum_l2280_228033


namespace sum_of_squares_is_312_l2280_228013

/-- Represents the rates and distances for biking, jogging, and swimming activities. -/
structure ActivityRates where
  bike_rate : ℕ
  jog_rate : ℕ
  swim_rate : ℕ

/-- Calculates the total distance covered given rates and times. -/
def total_distance (rates : ActivityRates) (bike_time jog_time swim_time : ℕ) : ℕ :=
  rates.bike_rate * bike_time + rates.jog_rate * jog_time + rates.swim_rate * swim_time

/-- Theorem stating that given the conditions, the sum of squares of rates is 312. -/
theorem sum_of_squares_is_312 (rates : ActivityRates) : 
  total_distance rates 1 4 3 = 66 ∧ 
  total_distance rates 3 3 2 = 76 → 
  rates.bike_rate ^ 2 + rates.jog_rate ^ 2 + rates.swim_rate ^ 2 = 312 := by
  sorry

#check sum_of_squares_is_312

end sum_of_squares_is_312_l2280_228013


namespace existence_of_set_B_l2280_228030

theorem existence_of_set_B : ∃ (a : ℝ), 
  let A : Set ℝ := {1, 3, a^2 + 3*a - 4}
  let B : Set ℝ := {0, 6, a^2 + 4*a - 2, a + 3}
  (A ∩ B = {3}) ∧ (a = 0) := by
  sorry

end existence_of_set_B_l2280_228030


namespace monotonic_power_function_l2280_228032

theorem monotonic_power_function (m : ℝ) :
  (∀ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < x₂ → (m^2 - 5*m + 7) * x₁^(m^2 - 6) < (m^2 - 5*m + 7) * x₂^(m^2 - 6)) →
  m = 3 := by
sorry

end monotonic_power_function_l2280_228032


namespace three_digit_number_proof_l2280_228027

/-- A three-digit number is between 100 and 999 inclusive -/
def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

theorem three_digit_number_proof :
  ∃ (x : ℕ), is_three_digit x ∧ (7000 + x) - (10 * x + 7) = 3555 ∧ x = 382 := by
  sorry

end three_digit_number_proof_l2280_228027


namespace billboard_average_is_twenty_l2280_228070

/-- Calculates the average number of billboards seen per hour given the counts for three consecutive hours. -/
def average_billboards (hour1 hour2 hour3 : ℕ) : ℚ :=
  (hour1 + hour2 + hour3 : ℚ) / 3

/-- Theorem stating that the average number of billboards seen per hour is 20 given the specific counts. -/
theorem billboard_average_is_twenty :
  average_billboards 17 20 23 = 20 := by
  sorry

end billboard_average_is_twenty_l2280_228070


namespace algebra_test_male_students_l2280_228016

theorem algebra_test_male_students (M : ℕ) : 
  (90 * (M + 32) = 82 * M + 92 * 32) → M = 8 := by
sorry

end algebra_test_male_students_l2280_228016


namespace parabola_directrix_l2280_228078

/-- Given a parabola with equation y = -3x^2 + 6x - 5, its directrix is y = -23/12 -/
theorem parabola_directrix (x y : ℝ) : 
  y = -3 * x^2 + 6 * x - 5 →
  ∃ (k : ℝ), k = -23/12 ∧ k = y - (1/(4 * -3)) :=
by sorry

end parabola_directrix_l2280_228078


namespace quadratic_equation_result_l2280_228056

theorem quadratic_equation_result (x : ℝ) : 
  7 * x^2 - 2 * x - 4 = 4 * x + 11 → (5 * x - 7)^2 = 570 / 49 := by
  sorry

end quadratic_equation_result_l2280_228056


namespace tangent_circle_equation_l2280_228002

/-- A circle with radius 2, center on the positive x-axis, and tangent to the y-axis -/
structure TangentCircle where
  center : ℝ × ℝ
  radius : ℝ
  center_on_x_axis : center.2 = 0
  positive_x : center.1 > 0
  radius_is_two : radius = 2
  tangent_to_y : center.1 = radius

/-- The equation of the circle is x² + y² - 4x = 0 -/
theorem tangent_circle_equation (c : TangentCircle) :
  ∀ x y : ℝ, (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2 ↔ x^2 + y^2 - 4*x = 0 :=
by sorry

end tangent_circle_equation_l2280_228002


namespace zach_remaining_amount_l2280_228057

/-- Represents the financial situation for Zach's bike purchase --/
structure BikeSavings where
  bike_cost : ℕ
  weekly_allowance : ℕ
  lawn_mowing_pay : ℕ
  babysitting_rate : ℕ
  current_savings : ℕ
  babysitting_hours : ℕ

/-- Calculates the remaining amount needed to buy the bike --/
def remaining_amount (s : BikeSavings) : ℕ :=
  s.bike_cost - (s.current_savings + s.weekly_allowance + s.lawn_mowing_pay + s.babysitting_rate * s.babysitting_hours)

/-- Theorem stating the remaining amount Zach needs to earn --/
theorem zach_remaining_amount :
  let s : BikeSavings := {
    bike_cost := 100,
    weekly_allowance := 5,
    lawn_mowing_pay := 10,
    babysitting_rate := 7,
    current_savings := 65,
    babysitting_hours := 2
  }
  remaining_amount s = 6 := by sorry

end zach_remaining_amount_l2280_228057


namespace polynomial_inequality_polynomial_inequality_equality_condition_l2280_228025

/-- A polynomial of degree 3 with roots in (0, 1) -/
structure PolynomialWithRootsInUnitInterval where
  b : ℝ
  c : ℝ
  root_property : ∃ (x₁ x₂ x₃ : ℝ), 0 < x₁ ∧ x₁ < 1 ∧
                                    0 < x₂ ∧ x₂ < 1 ∧
                                    0 < x₃ ∧ x₃ < 1 ∧
                                    x₁ + x₂ + x₃ = 2 ∧
                                    x₁ * x₂ + x₂ * x₃ + x₃ * x₁ = b ∧
                                    x₁ * x₂ * x₃ = -c

/-- The main theorem stating the inequality for polynomials with roots in (0, 1) -/
theorem polynomial_inequality (P : PolynomialWithRootsInUnitInterval) :
  8 * P.b + 9 * P.c ≤ 8 := by
  sorry

/-- Conditions for equality in the polynomial inequality -/
theorem polynomial_inequality_equality_condition (P : PolynomialWithRootsInUnitInterval) :
  (8 * P.b + 9 * P.c = 8) ↔ 
  (∃ (x : ℝ), x = 2/3 ∧ 
   ∃ (x₁ x₂ x₃ : ℝ), x₁ = x ∧ x₂ = x ∧ x₃ = x ∧
                     0 < x₁ ∧ x₁ < 1 ∧
                     0 < x₂ ∧ x₂ < 1 ∧
                     0 < x₃ ∧ x₃ < 1 ∧
                     x₁ + x₂ + x₃ = 2 ∧
                     x₁ * x₂ + x₂ * x₃ + x₃ * x₁ = P.b ∧
                     x₁ * x₂ * x₃ = -P.c) := by
  sorry

end polynomial_inequality_polynomial_inequality_equality_condition_l2280_228025


namespace secret_spread_day_l2280_228010

/-- The number of people who know the secret on day n -/
def secret_spread (n : ℕ) : ℕ := (3^(n+1) - 1) / 2

/-- The day when the secret is known by 6560 people -/
def target_day : ℕ := 8

theorem secret_spread_day : secret_spread target_day = 6560 := by
  sorry

#eval secret_spread target_day

end secret_spread_day_l2280_228010


namespace sports_club_overlap_l2280_228081

theorem sports_club_overlap (total : ℕ) (badminton : ℕ) (tennis : ℕ) (neither : ℕ)
  (h_total : total = 150)
  (h_badminton : badminton = 75)
  (h_tennis : tennis = 60)
  (h_neither : neither = 25) :
  badminton + tennis - (total - neither) = 10 := by
  sorry

end sports_club_overlap_l2280_228081


namespace mersenne_prime_implies_prime_exponent_l2280_228023

theorem mersenne_prime_implies_prime_exponent (n : ℕ) :
  Nat.Prime (2^n - 1) → Nat.Prime n := by
sorry

end mersenne_prime_implies_prime_exponent_l2280_228023


namespace hemisphere_surface_area_l2280_228086

theorem hemisphere_surface_area (r : ℝ) (h : π * r^2 = 64 * π) :
  2 * π * r^2 + π * r^2 = 192 * π := by
  sorry

end hemisphere_surface_area_l2280_228086


namespace swimming_contest_proof_l2280_228007

def kelly_time : ℕ := 3 * 60  -- Kelly's time in seconds

def brittany_time (kelly : ℕ) : ℕ := kelly - 20

def buffy_time (brittany : ℕ) : ℕ := brittany - 40

def carmen_time (kelly : ℕ) : ℕ := kelly + 15

def denise_time (carmen : ℕ) : ℕ := carmen - 35

def total_time (kelly brittany buffy carmen denise : ℕ) : ℕ :=
  kelly + brittany + buffy + carmen + denise

def average_time (total : ℕ) (count : ℕ) : ℕ := total / count

theorem swimming_contest_proof :
  let kelly := kelly_time
  let brittany := brittany_time kelly
  let buffy := buffy_time brittany
  let carmen := carmen_time kelly
  let denise := denise_time carmen
  let total := total_time kelly brittany buffy carmen denise
  let avg := average_time total 5
  total = 815 ∧ avg = 163 := by
  sorry

end swimming_contest_proof_l2280_228007


namespace university_theater_sales_l2280_228038

/-- The total money made from ticket sales at University Theater --/
def total_money_made (total_tickets : ℕ) (adult_price senior_price : ℕ) (senior_tickets : ℕ) : ℕ :=
  let adult_tickets := total_tickets - senior_tickets
  adult_tickets * adult_price + senior_tickets * senior_price

/-- Theorem: The University Theater made $8748 from ticket sales --/
theorem university_theater_sales : total_money_made 510 21 15 327 = 8748 := by
  sorry

end university_theater_sales_l2280_228038


namespace ratio_transformation_l2280_228099

/-- Given an original ratio of 2:3, prove that adding 2 to each term results in a ratio of 4:5 -/
theorem ratio_transformation (x : ℚ) : x = 2 → (2 + x) / (3 + x) = 4 / 5 := by
  sorry

end ratio_transformation_l2280_228099


namespace exists_p_with_conditions_l2280_228092

theorem exists_p_with_conditions : ∃ p : ℕ+, 
  ∃ q r s : ℕ+,
  (Nat.gcd p q = 40) ∧
  (Nat.gcd q r = 45) ∧
  (Nat.gcd r s = 60) ∧
  (∃ k : ℕ+, Nat.gcd s p = 10 * k ∧ k ≥ 10 ∧ k < 100) ∧
  (∃ m : ℕ+, p = 7 * m) := by
sorry

end exists_p_with_conditions_l2280_228092


namespace opposite_value_implies_ab_zero_l2280_228054

/-- Given that for all x, a(-x) + b(-x)^2 = -(ax + bx^2), prove that ab = 0 -/
theorem opposite_value_implies_ab_zero (a b : ℝ) 
  (h : ∀ x : ℝ, a * (-x) + b * (-x)^2 = -(a * x + b * x^2)) : 
  a * b = 0 := by
  sorry

end opposite_value_implies_ab_zero_l2280_228054


namespace arithmetic_sequence_problem_l2280_228017

theorem arithmetic_sequence_problem (a : ℕ → ℤ) :
  (∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)) →  -- arithmetic sequence condition
  a 4 - a 2 = -2 →                                      -- given condition
  a 7 = -3 →                                            -- given condition
  a 9 = -5 := by                                        -- conclusion to prove
sorry

end arithmetic_sequence_problem_l2280_228017


namespace max_value_of_f_l2280_228079

noncomputable def f (t : ℝ) : ℝ := (3^t - 4*t)*t / 9^t

theorem max_value_of_f :
  ∃ (max : ℝ), max = 1/16 ∧ ∀ (t : ℝ), f t ≤ max :=
sorry

end max_value_of_f_l2280_228079


namespace hyperbola_equation_l2280_228080

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the hyperbola
def hyperbola (x y a b : ℝ) : Prop := (x^2 / a^2) - (y^2 / b^2) = 1

-- Define the asymptotes
def asymptotes (x y : ℝ) : Prop := y = Real.sqrt 3 * x ∨ y = -Real.sqrt 3 * x

-- State the theorem
theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ (x y : ℝ), parabola x y ∧ hyperbola x y a b) →
  (∀ (x y : ℝ), asymptotes x y) →
  (∀ (x y : ℝ), hyperbola x y 1 (Real.sqrt 3)) := by
  sorry

end hyperbola_equation_l2280_228080


namespace notebook_cost_l2280_228093

theorem notebook_cost (total_students : Nat) (total_cost : Nat) : ∃ (s n c : Nat),
  -- Total number of students
  total_students = 42 ∧
  -- Majority of students bought notebooks
  s > total_students / 2 ∧
  -- Number of notebooks per student is greater than 2
  n > 2 ∧
  -- Cost in cents is greater than number of notebooks
  c > n ∧
  -- Total cost equation
  s * n * c = total_cost ∧
  -- Given total cost
  total_cost = 2773 →
  -- Conclusion: cost of a notebook is 103 cents
  c = 103 :=
sorry

end notebook_cost_l2280_228093


namespace tax_percentage_calculation_l2280_228058

theorem tax_percentage_calculation (original_cost total_paid : ℝ) 
  (h1 : original_cost = 200)
  (h2 : total_paid = 230) :
  (total_paid - original_cost) / original_cost * 100 = 15 := by
  sorry

end tax_percentage_calculation_l2280_228058


namespace amy_garden_space_l2280_228045

/-- Calculates the total square footage of garden beds -/
def total_sq_ft (num_beds1 num_beds2 : ℕ) (length1 width1 length2 width2 : ℝ) : ℝ :=
  (num_beds1 * length1 * width1) + (num_beds2 * length2 * width2)

/-- Proves that Amy's garden beds have a total of 42 sq ft of growing space -/
theorem amy_garden_space : total_sq_ft 2 2 3 3 4 3 = 42 := by
  sorry

end amy_garden_space_l2280_228045


namespace initial_people_count_l2280_228036

/-- The number of people initially on the train -/
def initial_people : ℕ := sorry

/-- The number of people left on the train after the first stop -/
def people_left : ℕ := 31

/-- The number of people who got off at the first stop -/
def people_off : ℕ := 17

/-- Theorem stating that the initial number of people on the train was 48 -/
theorem initial_people_count : initial_people = people_left + people_off :=
by sorry

end initial_people_count_l2280_228036


namespace range_of_a_l2280_228066

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x > 0 → a * Real.exp (a * Real.exp x + a) ≥ Real.log (Real.exp x + 1)) → 
  a ≥ 1 / Real.exp 1 :=
sorry

end range_of_a_l2280_228066


namespace approximation_theorem_l2280_228098

theorem approximation_theorem (a b : ℝ) (ε : ℝ) (hε : ε > 0) :
  ∃ (k m : ℤ) (n : ℕ), |n • a - k| < ε ∧ |n • b - m| < ε := by
  sorry

end approximation_theorem_l2280_228098


namespace largest_pile_size_l2280_228043

theorem largest_pile_size (total : ℕ) (small medium large : ℕ) : 
  total = small + medium + large →
  medium = 2 * small →
  large = 3 * small →
  total = 240 →
  large = 120 := by
sorry

end largest_pile_size_l2280_228043


namespace sandcastle_heights_sum_l2280_228000

/-- Represents the height of a sandcastle in feet and fractions of a foot -/
structure SandcastleHeight where
  whole : ℕ
  numerator : ℕ
  denominator : ℕ

/-- Calculates the total height of four sandcastles -/
def total_height (janet : SandcastleHeight) (sister : SandcastleHeight) 
                 (tom : SandcastleHeight) (lucy : SandcastleHeight) : ℚ :=
  (janet.whole : ℚ) + (janet.numerator : ℚ) / (janet.denominator : ℚ) +
  (sister.whole : ℚ) + (sister.numerator : ℚ) / (sister.denominator : ℚ) +
  (tom.whole : ℚ) + (tom.numerator : ℚ) / (tom.denominator : ℚ) +
  (lucy.whole : ℚ) + (lucy.numerator : ℚ) / (lucy.denominator : ℚ)

theorem sandcastle_heights_sum :
  let janet := SandcastleHeight.mk 3 5 6
  let sister := SandcastleHeight.mk 2 7 12
  let tom := SandcastleHeight.mk 1 11 20
  let lucy := SandcastleHeight.mk 2 13 24
  total_height janet sister tom lucy = 10 + 61 / 120 := by sorry

end sandcastle_heights_sum_l2280_228000


namespace tv_cash_savings_l2280_228088

/-- Calculates the savings when buying a television by cash instead of installments -/
theorem tv_cash_savings 
  (cash_price : ℕ) 
  (down_payment : ℕ) 
  (monthly_payment : ℕ) 
  (num_months : ℕ) : 
  cash_price = 400 →
  down_payment = 120 →
  monthly_payment = 30 →
  num_months = 12 →
  down_payment + monthly_payment * num_months - cash_price = 80 := by
sorry

end tv_cash_savings_l2280_228088


namespace ice_cream_flavors_count_l2280_228063

/-- The number of basic ice cream flavors -/
def num_flavors : ℕ := 4

/-- The number of scoops used to create a new flavor -/
def num_scoops : ℕ := 4

/-- 
The number of ways to distribute indistinguishable objects into distinguishable categories
n: number of objects
k: number of categories
-/
def stars_and_bars (n k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- The total number of distinct ice cream flavors that can be created -/
def total_flavors : ℕ := stars_and_bars num_scoops num_flavors

theorem ice_cream_flavors_count : total_flavors = 35 := by
  sorry

end ice_cream_flavors_count_l2280_228063


namespace max_value_x_plus_inverse_l2280_228072

theorem max_value_x_plus_inverse (x : ℝ) (h : 13 = x^2 + 1/x^2) :
  (x + 1/x) ≤ Real.sqrt 15 ∧ ∃ y : ℝ, 13 = y^2 + 1/y^2 ∧ y + 1/y = Real.sqrt 15 :=
by sorry

end max_value_x_plus_inverse_l2280_228072


namespace sarah_and_tom_ages_l2280_228064

/-- Given the age relationship between Sarah and Tom, prove their current ages sum to 33 -/
theorem sarah_and_tom_ages : ∃ (s t : ℕ),
  (s = t + 7) ∧                   -- Sarah is seven years older than Tom
  (s + 10 = 3 * (t - 3)) ∧        -- Ten years from now, Sarah will be three times as old as Tom was three years ago
  (s + t = 33)                    -- The sum of their current ages is 33
:= by sorry

end sarah_and_tom_ages_l2280_228064


namespace rationalize_denominator_l2280_228050

theorem rationalize_denominator :
  ∃ (A B C D : ℕ),
    (A = 25 ∧ B = 2 ∧ C = 20 ∧ D = 17) ∧
    D > 0 ∧
    (∀ p : ℕ, Prime p → ¬(p^2 ∣ B)) ∧
    (Real.sqrt 50 / (Real.sqrt 25 - 2 * Real.sqrt 2) = (A * Real.sqrt B + C) / D) := by
  sorry

end rationalize_denominator_l2280_228050


namespace count_valid_numbers_l2280_228051

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def digit_product (n : ℕ) : ℕ :=
  (n / 1000) * ((n / 100) % 10) * ((n / 10) % 10) * (n % 10)

def valid_number (n : ℕ) : Prop :=
  is_four_digit n ∧ digit_product n = 18

theorem count_valid_numbers : 
  ∃ (S : Finset ℕ), (∀ n ∈ S, valid_number n) ∧ S.card = 24 ∧ 
  (∀ m : ℕ, valid_number m → m ∈ S) :=
sorry

end count_valid_numbers_l2280_228051


namespace complex_number_in_first_quadrant_l2280_228018

theorem complex_number_in_first_quadrant : ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ (2 - I) / (1 - 3*I) = a + b*I := by
  sorry

end complex_number_in_first_quadrant_l2280_228018


namespace valid_distributions_count_l2280_228052

def number_of_valid_distributions : ℕ :=
  (Finset.filter (fun d => d ≠ 1 ∧ d ≠ 360) (Nat.divisors 360)).card

theorem valid_distributions_count : number_of_valid_distributions = 22 := by
  sorry

end valid_distributions_count_l2280_228052


namespace percentage_of_fair_haired_women_l2280_228041

theorem percentage_of_fair_haired_women (total : ℝ) 
  (h1 : total > 0) 
  (fair_haired_ratio : ℝ) 
  (h2 : fair_haired_ratio = 0.75)
  (women_ratio_among_fair_haired : ℝ) 
  (h3 : women_ratio_among_fair_haired = 0.40) : 
  (fair_haired_ratio * women_ratio_among_fair_haired) * 100 = 30 := by
sorry

end percentage_of_fair_haired_women_l2280_228041


namespace train_platform_passing_time_l2280_228047

/-- Given a train of length 1200 meters that crosses a tree in 120 seconds,
    calculate the time required to pass a platform of length 900 meters. -/
theorem train_platform_passing_time 
  (train_length : ℝ) 
  (tree_passing_time : ℝ) 
  (platform_length : ℝ) 
  (h1 : train_length = 1200) 
  (h2 : tree_passing_time = 120) 
  (h3 : platform_length = 900) :
  (train_length + platform_length) / (train_length / tree_passing_time) = 210 := by
  sorry

#check train_platform_passing_time

end train_platform_passing_time_l2280_228047


namespace problem_solution_l2280_228040

theorem problem_solution :
  let x : ℝ := -39660 - 17280 * Real.sqrt 2
  (x + 720 * Real.sqrt 1152) / Real.rpow 15625 (1/3) = 7932 / (3^2 - Real.sqrt 196) := by
  sorry

end problem_solution_l2280_228040


namespace root_value_theorem_l2280_228008

theorem root_value_theorem (a : ℝ) : 
  (2 * a^2 + 3 * a - 4 = 0) → (2 * a^2 + 3 * a = 4) := by
  sorry

end root_value_theorem_l2280_228008


namespace mileage_pay_is_104_l2280_228083

/-- Calculates the mileage pay for a delivery driver given the distances for three packages and the pay rate per mile. -/
def calculate_mileage_pay (first_package : ℝ) (second_package : ℝ) (third_package : ℝ) (pay_rate : ℝ) : ℝ :=
  (first_package + second_package + third_package) * pay_rate

/-- Theorem stating that given specific package distances and pay rate, the mileage pay is $104. -/
theorem mileage_pay_is_104 :
  let first_package : ℝ := 10
  let second_package : ℝ := 28
  let third_package : ℝ := second_package / 2
  let pay_rate : ℝ := 2
  calculate_mileage_pay first_package second_package third_package pay_rate = 104 := by
  sorry

#check mileage_pay_is_104

end mileage_pay_is_104_l2280_228083


namespace min_value_problem_l2280_228034

theorem min_value_problem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a * b - a - 2 * b = 0) :
  ∃ (min : ℝ), min = 7 ∧ 
  (∀ (x y : ℝ), x > 0 → y > 0 → x * y - x - 2 * y = 0 → 
    x^2 / 4 - 2 / x + y^2 - 1 / y ≥ min) ∧
  (a^2 / 4 - 2 / a + b^2 - 1 / b = min) :=
sorry

end min_value_problem_l2280_228034


namespace partial_fraction_decomposition_l2280_228019

theorem partial_fraction_decomposition :
  let f (x : ℚ) := (7 * x - 4) / (x^2 - 9*x - 18)
  let g (x : ℚ) := 59 / (11 * (x - 9)) + 18 / (11 * (x + 2))
  ∀ x, x ≠ 9 ∧ x ≠ -2 → f x = g x :=
by
  sorry

end partial_fraction_decomposition_l2280_228019


namespace set_union_problem_l2280_228069

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {1, 2^a}
def B (a b : ℝ) : Set ℝ := {a, b}

-- State the theorem
theorem set_union_problem (a b : ℝ) : 
  A a ∩ B a b = {1/2} → A a ∪ B a b = {-1, 1/2, 1} := by
  sorry

end set_union_problem_l2280_228069


namespace production_system_l2280_228074

/-- Represents the profit functions and properties of a production system with two products. -/
theorem production_system (total_workers : ℕ) 
  (prod_rate_A prod_rate_B : ℕ) 
  (profit_per_A profit_per_B cost_increase_B : ℚ) : 
  total_workers = 65 → 
  prod_rate_A = 2 →
  prod_rate_B = 1 →
  profit_per_A = 15 →
  profit_per_B = 120 →
  cost_increase_B = 2 →
  ∃ (profit_A profit_B : ℚ → ℚ) (x : ℚ),
    (∀ x, profit_A x = 1950 - 30 * x) ∧
    (∀ x, profit_B x = 120 * x - 2 * x^2) ∧
    (profit_A x - profit_B x = 1250 → x = 5) ∧
    (∃ (total_profit : ℚ → ℚ),
      (∀ x, total_profit x = profit_A x + profit_B x) ∧
      (∀ y, total_profit y ≤ 2962) ∧
      (total_profit 22 = 2962 ∨ total_profit 23 = 2962)) :=
by sorry

end production_system_l2280_228074


namespace smallest_factor_of_36_l2280_228015

theorem smallest_factor_of_36 (a b c : ℤ) 
  (h1 : a * b * c = 36) 
  (h2 : a + b + c = 4) : 
  min a (min b c) = -4 := by
sorry

end smallest_factor_of_36_l2280_228015


namespace thirty_percent_less_than_eighty_l2280_228014

theorem thirty_percent_less_than_eighty (x : ℚ) : x + x / 2 = 80 - 80 * 3 / 10 → x = 112 / 3 := by
  sorry

end thirty_percent_less_than_eighty_l2280_228014


namespace y_intercept_of_line_l2280_228075

theorem y_intercept_of_line (x y : ℝ) :
  2 * x - 3 * y = 6 → x = 0 → y = -2 := by
  sorry

end y_intercept_of_line_l2280_228075


namespace randy_blocks_theorem_l2280_228077

/-- The number of blocks Randy used to build a tower -/
def blocks_used : ℕ := 25

/-- The number of blocks Randy has left -/
def blocks_left : ℕ := 72

/-- The initial number of blocks Randy had -/
def initial_blocks : ℕ := blocks_used + blocks_left

theorem randy_blocks_theorem : initial_blocks = 97 := by
  sorry

end randy_blocks_theorem_l2280_228077


namespace investment_return_rate_l2280_228029

theorem investment_return_rate 
  (total_investment : ℝ) 
  (total_interest : ℝ) 
  (known_rate : ℝ) 
  (known_investment : ℝ) 
  (h1 : total_investment = 33000)
  (h2 : total_interest = 970)
  (h3 : known_rate = 0.0225)
  (h4 : known_investment = 13000)
  : ∃ r : ℝ, 
    r * known_investment + known_rate * (total_investment - known_investment) = total_interest ∧ 
    r = 0.04 := by
  sorry

end investment_return_rate_l2280_228029


namespace food_boxes_l2280_228026

theorem food_boxes (total_food : ℝ) (food_per_box : ℝ) (h1 : total_food = 777.5) (h2 : food_per_box = 2.25) :
  ⌊total_food / food_per_box⌋ = 345 := by
  sorry

end food_boxes_l2280_228026


namespace fraction_exponent_equality_l2280_228090

theorem fraction_exponent_equality (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  (x / y)^(-3/4 : ℝ) = 4 * (y / x)^3 := by sorry

end fraction_exponent_equality_l2280_228090


namespace largest_quantity_l2280_228039

def A : ℚ := 2010 / 2009 + 2010 / 2011
def B : ℚ := 2010 / 2011 + 2012 / 2011
def C : ℚ := 2011 / 2010 + 2011 / 2012

theorem largest_quantity : A > B ∧ A > C := by
  sorry

end largest_quantity_l2280_228039


namespace exists_transformation_458_to_14_l2280_228091

-- Define the operations
def double (n : ℕ) : ℕ := 2 * n

def erase_last_digit (n : ℕ) : ℕ :=
  if n < 10 then n else n / 10

-- Define a single step transformation
inductive Step
| Double : Step
| EraseLastDigit : Step

def apply_step (n : ℕ) (s : Step) : ℕ :=
  match s with
  | Step.Double => double n
  | Step.EraseLastDigit => erase_last_digit n

-- Define a sequence of steps
def apply_steps (n : ℕ) (steps : List Step) : ℕ :=
  steps.foldl apply_step n

-- Theorem statement
theorem exists_transformation_458_to_14 :
  ∃ (steps : List Step), apply_steps 458 steps = 14 := by
  sorry

end exists_transformation_458_to_14_l2280_228091


namespace sixth_quiz_score_l2280_228006

theorem sixth_quiz_score (scores : List ℕ) (target_mean : ℕ) : 
  scores = [86, 90, 82, 84, 95] →
  target_mean = 95 →
  ∃ (sixth_score : ℕ), 
    sixth_score = 133 ∧ 
    (scores.sum + sixth_score) / 6 = target_mean :=
by sorry

end sixth_quiz_score_l2280_228006


namespace min_organizer_handshakes_l2280_228021

/-- The number of handshakes between players in a chess tournament where each player plays against every other player exactly once -/
def player_handshakes (n : ℕ) : ℕ := n * (n - 1) / 2

/-- The total number of handshakes including those of the organizer -/
def total_handshakes (n : ℕ) (k : ℕ) : ℕ := player_handshakes n + k

/-- Theorem stating that the minimum number of organizer handshakes is 0 given 406 total handshakes -/
theorem min_organizer_handshakes :
  ∃ (n : ℕ), total_handshakes n 0 = 406 ∧ 
  ∀ (m k : ℕ), total_handshakes m k = 406 → k ≥ 0 :=
sorry

end min_organizer_handshakes_l2280_228021


namespace certain_number_problem_l2280_228049

theorem certain_number_problem : ∃ x : ℕ, 3*15 + 3*16 + 3*19 + x = 161 ∧ x = 11 := by
  sorry

end certain_number_problem_l2280_228049


namespace toms_average_speed_l2280_228068

/-- Prove that Tom's average speed is 45 mph given the race conditions -/
theorem toms_average_speed (karen_speed : ℝ) (karen_delay : ℝ) (karen_win_margin : ℝ) (tom_distance : ℝ) :
  karen_speed = 60 →
  karen_delay = 1 / 15 →
  karen_win_margin = 4 →
  tom_distance = 24 →
  (tom_distance / ((tom_distance + karen_win_margin) / karen_speed + karen_delay)) = 45 :=
by sorry

end toms_average_speed_l2280_228068


namespace partial_fraction_decomposition_l2280_228028

theorem partial_fraction_decomposition :
  ∃! (A B C : ℚ),
    ∀ (x : ℚ), x ≠ 1 ∧ x ≠ 4 ∧ x ≠ 6 →
      (x^2 - 5*x + 6) / ((x - 1)*(x - 4)*(x - 6)) =
      A / (x - 1) + B / (x - 4) + C / (x - 6) ∧
      A = 2/15 ∧ B = -1/3 ∧ C = 3/5 := by
sorry

end partial_fraction_decomposition_l2280_228028


namespace complex_division_l2280_228001

/-- Given complex numbers z₁ and z₂ corresponding to points (1, -1) and (-2, 1) in the complex plane,
    prove that z₂/z₁ = -3/2 - 1/2i. -/
theorem complex_division (z₁ z₂ : ℂ) (h₁ : z₁ = Complex.mk 1 (-1)) (h₂ : z₂ = Complex.mk (-2) 1) :
  z₂ / z₁ = Complex.mk (-3/2) (-1/2) := by
  sorry

end complex_division_l2280_228001


namespace complex_product_of_three_l2280_228012

theorem complex_product_of_three (α₁ α₂ α₃ : ℝ) (z₁ z₂ z₃ : ℂ) :
  z₁ = Complex.exp (Complex.I * α₁) →
  z₂ = Complex.exp (Complex.I * α₂) →
  z₃ = Complex.exp (Complex.I * α₃) →
  z₁ * z₂ = Complex.exp (Complex.I * (α₁ + α₂)) →
  z₂ * z₃ = Complex.exp (Complex.I * (α₂ + α₃)) →
  z₁ * z₂ * z₃ = Complex.exp (Complex.I * (α₁ + α₂ + α₃)) :=
by sorry

end complex_product_of_three_l2280_228012


namespace problem_1_problem_2_l2280_228060

-- Problem 1
theorem problem_1 (f : ℝ → ℝ) (h : ∀ x ≤ 1, f (1 - Real.sqrt x) = x) :
  ∀ x ≤ 1, f x = x^2 - 2*x + 1 := by sorry

-- Problem 2
theorem problem_2 (f : ℝ → ℝ) 
  (h1 : ∃ a b : ℝ, ∀ x, f x = a * x + b) 
  (h2 : ∀ x, f (f x) = 4 * x + 3) :
  (∀ x, f x = 2 * x + 1) ∨ (∀ x, f x = -2 * x - 3) := by sorry

end problem_1_problem_2_l2280_228060
