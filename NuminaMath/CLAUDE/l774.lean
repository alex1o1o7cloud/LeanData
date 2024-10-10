import Mathlib

namespace intersection_point_sum_squares_l774_77420

-- Define the lines
def line1 (x y : ℝ) : Prop := 323 * x + 457 * y = 1103
def line2 (x y : ℝ) : Prop := 177 * x + 543 * y = 897

-- Define the intersection point
def intersection_point (a b : ℝ) : Prop := line1 a b ∧ line2 a b

-- Theorem statement
theorem intersection_point_sum_squares :
  ∀ a b : ℝ, intersection_point a b → a^2 + 2004 * b^2 = 2008 := by
  sorry

end intersection_point_sum_squares_l774_77420


namespace quadratic_equation_properties_l774_77440

theorem quadratic_equation_properties (m : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + 2*m*x₁ + m^2 - 1 = 0 ∧ x₂^2 + 2*m*x₂ + m^2 - 1 = 0) ∧
  ((-2)^2 + 2*m*(-2) + m^2 - 1 = 0 → 2023 - m^2 + 4*m = 2026) :=
by
  sorry

end quadratic_equation_properties_l774_77440


namespace ab_four_necessary_not_sufficient_l774_77494

/-- Two lines in the plane -/
structure TwoLines where
  a : ℝ
  b : ℝ

/-- The condition that the slopes are equal -/
def slopes_equal (l : TwoLines) : Prop :=
  l.a * l.b = 4

/-- The condition that the lines are parallel -/
def are_parallel (l : TwoLines) : Prop :=
  (2 * l.b = l.a * 2) ∧ ¬(2 * (l.b - 2) = l.a * (-1))

/-- The main theorem: ab = 4 is necessary but not sufficient for parallelism -/
theorem ab_four_necessary_not_sufficient :
  (∀ l : TwoLines, are_parallel l → slopes_equal l) ∧
  ¬(∀ l : TwoLines, slopes_equal l → are_parallel l) :=
sorry

end ab_four_necessary_not_sufficient_l774_77494


namespace product_of_special_numbers_l774_77483

theorem product_of_special_numbers (m n : ℕ) 
  (h1 : m + n = 20) 
  (h2 : (1 : ℚ) / m + (1 : ℚ) / n = 5 / 24) : 
  m * n = 96 := by
  sorry

end product_of_special_numbers_l774_77483


namespace f_is_even_count_f_eq_2016_l774_77401

/-- The smallest factor of n that is not 1 -/
def smallest_factor (n : ℕ) : ℕ := sorry

/-- The function f as defined in the problem -/
def f (n : ℕ) : ℕ := n + smallest_factor n

/-- Theorem stating that f(n) is always even for n > 1 -/
theorem f_is_even (n : ℕ) (h : n > 1) : Even (f n) := by sorry

/-- Theorem stating that there are exactly 3 positive integers n such that f(n) = 2016 -/
theorem count_f_eq_2016 : ∃! (s : Finset ℕ), (∀ n ∈ s, f n = 2016) ∧ s.card = 3 := by sorry

end f_is_even_count_f_eq_2016_l774_77401


namespace sqrt_equality_implies_one_three_l774_77491

theorem sqrt_equality_implies_one_three :
  ∀ a b : ℕ+,
  a < b →
  (Real.sqrt (1 + Real.sqrt (27 + 18 * Real.sqrt 3)) = Real.sqrt a + Real.sqrt b) →
  (a = 1 ∧ b = 3) := by
sorry

end sqrt_equality_implies_one_three_l774_77491


namespace candy_division_ways_l774_77408

def divide_candies (total : ℕ) (min_per_person : ℕ) : ℕ :=
  total - 2 * min_per_person + 1

theorem candy_division_ways :
  divide_candies 8 1 = 7 := by
  sorry

end candy_division_ways_l774_77408


namespace money_problem_l774_77454

theorem money_problem (a b : ℝ) 
  (h1 : 5 * a + 2 * b > 100)
  (h2 : 4 * a - b = 40) : 
  a > 180 / 13 ∧ b > 200 / 13 := by
  sorry

end money_problem_l774_77454


namespace arccos_one_half_eq_pi_third_l774_77400

theorem arccos_one_half_eq_pi_third : Real.arccos (1/2) = π/3 := by
  sorry

end arccos_one_half_eq_pi_third_l774_77400


namespace factorization_proof_l774_77439

theorem factorization_proof (c : ℝ) : 196 * c^2 + 42 * c - 14 = 14 * c * (14 * c + 2) := by
  sorry

end factorization_proof_l774_77439


namespace race_order_l774_77430

-- Define the participants
inductive Participant : Type
  | Jia : Participant
  | Yi : Participant
  | Bing : Participant
  | Ding : Participant
  | Wu : Participant

-- Define a relation for "finished before"
def finished_before (a b : Participant) : Prop := sorry

-- Define the conditions
axiom ding_faster_than_yi : finished_before Participant.Ding Participant.Yi
axiom wu_before_bing : finished_before Participant.Wu Participant.Bing
axiom jia_between_bing_and_ding : 
  finished_before Participant.Bing Participant.Jia ∧ 
  finished_before Participant.Jia Participant.Ding

-- State the theorem
theorem race_order : 
  finished_before Participant.Wu Participant.Bing ∧
  finished_before Participant.Bing Participant.Jia ∧
  finished_before Participant.Jia Participant.Ding ∧
  finished_before Participant.Ding Participant.Yi :=
by sorry

end race_order_l774_77430


namespace unique_solution_positive_root_l774_77403

theorem unique_solution_positive_root (x : ℝ) :
  x ≥ 0 ∧ 2021 * (x^2020)^(1/202) - 1 = 2020 * x ↔ x = 1 := by
  sorry

end unique_solution_positive_root_l774_77403


namespace profit_margin_in_terms_of_retail_price_l774_77407

/-- Given a profit margin P, production cost C, retail price P_R, and constants k and c,
    prove that P can be expressed in terms of P_R. -/
theorem profit_margin_in_terms_of_retail_price
  (P C P_R k c : ℝ) (hP : P = k * C) (hP_R : P_R = c * (P + C)) :
  P = (k / (c * (k + 1))) * P_R :=
sorry

end profit_margin_in_terms_of_retail_price_l774_77407


namespace pigs_joined_l774_77458

/-- Given an initial number of pigs and a final number of pigs,
    prove that the number of pigs that joined is equal to their difference. -/
theorem pigs_joined (initial final : ℕ) (h : final ≥ initial) :
  final - initial = final - initial :=
by sorry

end pigs_joined_l774_77458


namespace existence_of_n_good_not_n_plus_1_good_l774_77406

def sum_of_digits (k : ℕ+) : ℕ := sorry

def is_n_good (a n : ℕ+) : Prop :=
  ∃ (seq : Fin (n + 1) → ℕ+),
    seq (Fin.last n) = a ∧
    ∀ i : Fin n, seq i.succ = seq i - sum_of_digits (seq i)

theorem existence_of_n_good_not_n_plus_1_good :
  ∀ n : ℕ+, ∃ b : ℕ+, is_n_good b n ∧ ¬is_n_good b (n + 1) :=
sorry

end existence_of_n_good_not_n_plus_1_good_l774_77406


namespace train_passing_bridge_l774_77493

/-- Time for a train to pass a bridge -/
theorem train_passing_bridge 
  (train_length : ℝ) 
  (train_speed_kmh : ℝ) 
  (bridge_length : ℝ) 
  (h1 : train_length = 500)
  (h2 : train_speed_kmh = 72)
  (h3 : bridge_length = 200) : 
  (train_length + bridge_length) / (train_speed_kmh * 1000 / 3600) = 35 := by
  sorry

end train_passing_bridge_l774_77493


namespace fraction_say_dislike_actually_like_l774_77419

def TotalStudents : ℝ := 100

def LikeDancing : ℝ := 0.6 * TotalStudents
def DislikeDancing : ℝ := 0.4 * TotalStudents

def SayLikeActuallyLike : ℝ := 0.8 * LikeDancing
def SayDislikeActuallyLike : ℝ := 0.2 * LikeDancing
def SayDislikeActuallyDislike : ℝ := 0.9 * DislikeDancing
def SayLikeActuallyDislike : ℝ := 0.1 * DislikeDancing

def TotalSayDislike : ℝ := SayDislikeActuallyLike + SayDislikeActuallyDislike

theorem fraction_say_dislike_actually_like : 
  SayDislikeActuallyLike / TotalSayDislike = 0.25 := by
  sorry

end fraction_say_dislike_actually_like_l774_77419


namespace rectangle_area_is_72_l774_77475

/-- Represents a circle with a center point and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a rectangle with two corner points -/
structure Rectangle where
  topLeft : ℝ × ℝ
  bottomRight : ℝ × ℝ

def circleP : Circle := { center := (0, 3), radius := 3 }
def circleQ : Circle := { center := (3, 3), radius := 3 }
def circleR : Circle := { center := (6, 3), radius := 3 }
def circleS : Circle := { center := (9, 3), radius := 3 }

def rectangleABCD : Rectangle := { topLeft := (0, 6), bottomRight := (12, 0) }

theorem rectangle_area_is_72 
  (h1 : circleP.radius = circleQ.radius ∧ circleP.radius = circleR.radius ∧ circleP.radius = circleS.radius)
  (h2 : circleP.center.2 = circleQ.center.2 ∧ circleP.center.2 = circleR.center.2 ∧ circleP.center.2 = circleS.center.2)
  (h3 : circleP.center.1 + circleP.radius = circleQ.center.1 ∧ 
        circleQ.center.1 + circleQ.radius = circleR.center.1 ∧
        circleR.center.1 + circleR.radius = circleS.center.1)
  (h4 : rectangleABCD.topLeft.1 = circleP.center.1 - circleP.radius ∧
        rectangleABCD.bottomRight.1 = circleS.center.1 + circleS.radius)
  (h5 : rectangleABCD.topLeft.2 = circleP.center.2 + circleP.radius ∧
        rectangleABCD.bottomRight.2 = circleP.center.2 - circleP.radius)
  : (rectangleABCD.bottomRight.1 - rectangleABCD.topLeft.1) * 
    (rectangleABCD.topLeft.2 - rectangleABCD.bottomRight.2) = 72 := by
  sorry

end rectangle_area_is_72_l774_77475


namespace power_multiplication_correct_equation_l774_77487

theorem power_multiplication (a b : ℕ) : 2^a * 2^b = 2^(a + b) := by sorry

theorem correct_equation : 2^2 * 2^3 = 2^5 := by
  apply power_multiplication

end power_multiplication_correct_equation_l774_77487


namespace unique_p_q_for_f_bounded_l774_77488

def f (p q x : ℝ) := x^2 + p*x + q

theorem unique_p_q_for_f_bounded :
  ∃! p q : ℝ, ∀ x ∈ Set.Icc 1 5, |f p q x| ≤ 2 := by sorry

end unique_p_q_for_f_bounded_l774_77488


namespace arithmetic_equality_l774_77473

theorem arithmetic_equality : 57 * 44 + 13 * 44 = 3080 := by
  sorry

end arithmetic_equality_l774_77473


namespace triangle_existence_implies_m_greater_than_six_l774_77480

/-- The function f(x) = x^3 - 3x + m -/
def f (m : ℝ) (x : ℝ) : ℝ := x^3 - 3*x + m

/-- Theorem: If there exists a triangle with side lengths f(a), f(b), and f(c) for a, b, c in [0,2], then m > 6 -/
theorem triangle_existence_implies_m_greater_than_six (m : ℝ) :
  (∃ (a b c : ℝ), 0 ≤ a ∧ a ≤ 2 ∧ 0 ≤ b ∧ b ≤ 2 ∧ 0 ≤ c ∧ c ≤ 2 ∧ 
    f m a + f m b > f m c ∧ 
    f m b + f m c > f m a ∧ 
    f m c + f m a > f m b) →
  m > 6 := by
sorry


end triangle_existence_implies_m_greater_than_six_l774_77480


namespace separation_leads_to_growth_and_blessing_l774_77464

/-- Represents the separation experience between a child and their mother --/
structure SeparationExperience where
  duration : ℕ
  communication_frequency : ℕ
  visits : ℕ
  child_attitude : Bool
  mother_attitude : Bool

/-- Represents the outcome of the separation experience --/
inductive Outcome
  | PersonalGrowth
  | Blessing
  | Negative

/-- Function to determine the outcome of a separation experience --/
def determine_outcome (exp : SeparationExperience) : Outcome := sorry

/-- Theorem stating that a positive separation experience leads to personal growth and can be a blessing --/
theorem separation_leads_to_growth_and_blessing 
  (exp : SeparationExperience) 
  (h1 : exp.duration ≥ 3) 
  (h2 : exp.communication_frequency ≥ 300) 
  (h3 : exp.visits ≥ 1) 
  (h4 : exp.child_attitude = true) 
  (h5 : exp.mother_attitude = true) : 
  determine_outcome exp = Outcome.PersonalGrowth ∧ 
  determine_outcome exp = Outcome.Blessing := 
sorry

end separation_leads_to_growth_and_blessing_l774_77464


namespace max_value_of_trigonometric_expression_l774_77418

theorem max_value_of_trigonometric_expression :
  let f : ℝ → ℝ := λ x => Real.sin (x + 3 * Real.pi / 4) + Real.cos (x + Real.pi / 3) + Real.cos (x + Real.pi / 4)
  let max_value := 2 * Real.cos (-Real.pi / 24)
  ∀ x ∈ Set.Icc (-Real.pi / 2) (-Real.pi / 4), f x ≤ max_value ∧ ∃ x₀ ∈ Set.Icc (-Real.pi / 2) (-Real.pi / 4), f x₀ = max_value :=
by
  sorry

end max_value_of_trigonometric_expression_l774_77418


namespace arc_length_proof_l774_77444

open Real

noncomputable def curve (x : ℝ) : ℝ := Real.log (5 / (2 * x))

theorem arc_length_proof (a b : ℝ) (ha : a = Real.sqrt 3) (hb : b = Real.sqrt 8) :
  ∫ x in a..b, sqrt (1 + (deriv curve x) ^ 2) = 1 + (1 / 2) * log (3 / 2) :=
by sorry

end arc_length_proof_l774_77444


namespace delta_max_success_ratio_l774_77445

/-- Represents a player's score in a chess competition --/
structure PlayerScore where
  day1_score : ℕ
  day1_total : ℕ
  day2_score : ℕ
  day2_total : ℕ

/-- Calculate the success ratio for a given day --/
def day_success_ratio (score : ℕ) (total : ℕ) : ℚ :=
  ↑score / ↑total

/-- Calculate the overall success ratio --/
def overall_success_ratio (player : PlayerScore) : ℚ :=
  ↑(player.day1_score + player.day2_score) / ↑(player.day1_total + player.day2_total)

theorem delta_max_success_ratio 
  (gamma : PlayerScore)
  (delta : PlayerScore)
  (h1 : gamma.day1_score = 180 ∧ gamma.day1_total = 360)
  (h2 : gamma.day2_score = 150 ∧ gamma.day2_total = 240)
  (h3 : delta.day1_total + delta.day2_total = 600)
  (h4 : delta.day1_total ≠ 360)
  (h5 : delta.day1_score > 0 ∧ delta.day2_score > 0)
  (h6 : day_success_ratio delta.day1_score delta.day1_total < day_success_ratio gamma.day1_score gamma.day1_total)
  (h7 : day_success_ratio delta.day2_score delta.day2_total < day_success_ratio gamma.day2_score gamma.day2_total)
  (h8 : overall_success_ratio gamma = 11/20) :
  overall_success_ratio delta ≤ 599/600 :=
sorry

end delta_max_success_ratio_l774_77445


namespace money_sharing_l774_77417

theorem money_sharing (amanda ben carlos total : ℕ) : 
  amanda + ben + carlos = total →
  amanda = 3 * (ben / 5) →
  carlos = 9 * (ben / 5) →
  ben = 50 →
  total = 170 := by
sorry

end money_sharing_l774_77417


namespace max_value_of_f_l774_77478

-- Define the quadratic function
def f (x : ℝ) : ℝ := -3 * x^2 + 12 * x - 5

-- State the theorem
theorem max_value_of_f :
  ∃ (M : ℝ), (∀ x, f x ≤ M) ∧ (∃ x₀, f x₀ = M) ∧ M = 7 := by
  sorry

end max_value_of_f_l774_77478


namespace no_solution_for_equation_l774_77486

theorem no_solution_for_equation : 
  ¬ ∃ (a b : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ (1 / a + 1 / b = 2 / (a + b)) := by
  sorry

end no_solution_for_equation_l774_77486


namespace sum_of_roots_eq_fourteen_l774_77446

theorem sum_of_roots_eq_fourteen : ∀ x₁ x₂ : ℝ, (x₁ - 7)^2 = 16 ∧ (x₂ - 7)^2 = 16 → x₁ + x₂ = 14 := by
  sorry

end sum_of_roots_eq_fourteen_l774_77446


namespace tangent_cotangent_identity_l774_77402

theorem tangent_cotangent_identity (α : Real) 
  (h1 : 0 < α) (h2 : α < π/2) (h3 : α ≠ π/4) :
  (Real.sqrt (Real.tan α) + Real.sqrt (1 / Real.tan α)) / 
  (Real.sqrt (Real.tan α) - Real.sqrt (1 / Real.tan α)) = 
  1 / Real.tan (α - π/4) := by
  sorry

end tangent_cotangent_identity_l774_77402


namespace sin_cos_difference_equals_half_l774_77449

theorem sin_cos_difference_equals_half : 
  Real.sin (43 * π / 180) * Real.cos (13 * π / 180) - 
  Real.sin (13 * π / 180) * Real.cos (43 * π / 180) = 1/2 := by sorry

end sin_cos_difference_equals_half_l774_77449


namespace new_average_commission_is_550_l774_77415

/-- Represents a salesperson's commission data -/
structure SalespersonData where
  totalSales : ℕ
  lastCommission : ℝ
  averageIncrease : ℝ

/-- Calculates the new average commission for a salesperson -/
def newAverageCommission (data : SalespersonData) : ℝ :=
  -- The actual calculation is not implemented here
  sorry

/-- Theorem stating that under given conditions, the new average commission is $550 -/
theorem new_average_commission_is_550 (data : SalespersonData) 
  (h1 : data.totalSales = 6)
  (h2 : data.lastCommission = 1300)
  (h3 : data.averageIncrease = 150) :
  newAverageCommission data = 550 := by
  sorry

end new_average_commission_is_550_l774_77415


namespace livestream_sales_scientific_notation_l774_77497

/-- Proves that 1814 billion yuan is equal to 1.814 × 10^12 yuan -/
theorem livestream_sales_scientific_notation :
  (1814 : ℝ) * (10^9 : ℝ) = 1.814 * (10^12 : ℝ) := by
  sorry

end livestream_sales_scientific_notation_l774_77497


namespace smallest_positive_integer_ending_in_6_divisible_by_13_l774_77433

/-- A number ends in 6 if it's of the form 10n + 6 for some integer n -/
def ends_in_6 (x : ℕ) : Prop := ∃ n : ℕ, x = 10 * n + 6

/-- A number is divisible by 13 if there exists an integer k such that x = 13k -/
def divisible_by_13 (x : ℕ) : Prop := ∃ k : ℕ, x = 13 * k

theorem smallest_positive_integer_ending_in_6_divisible_by_13 :
  (ends_in_6 26 ∧ divisible_by_13 26) ∧
  ∀ x : ℕ, 0 < x ∧ x < 26 → ¬(ends_in_6 x ∧ divisible_by_13 x) :=
sorry

end smallest_positive_integer_ending_in_6_divisible_by_13_l774_77433


namespace equation_solution_l774_77431

theorem equation_solution :
  ∃ (x : ℚ), x ≠ -2 ∧ (x^2 + 2*x + 2) / (x + 2) = x + 3 ∧ x = -4/3 := by
  sorry

end equation_solution_l774_77431


namespace openai_robotics_competition_weight_l774_77448

/-- The weight of the standard robot in the OpenAI robotics competition. -/
def standard_robot_weight : ℝ := 100

/-- The maximum weight allowed for a robot in the competition. -/
def max_weight : ℝ := 210

/-- The minimum weight of a robot in the competition. -/
def min_weight : ℝ := standard_robot_weight + 5

theorem openai_robotics_competition_weight :
  standard_robot_weight = 100 ∧
  max_weight = 210 ∧
  min_weight = standard_robot_weight + 5 ∧
  max_weight ≤ 2 * min_weight :=
by sorry

end openai_robotics_competition_weight_l774_77448


namespace basketball_score_problem_l774_77455

theorem basketball_score_problem (total_points winning_margin : ℕ) 
  (h1 : total_points = 48) 
  (h2 : winning_margin = 18) : 
  ∃ (sharks_score dolphins_score : ℕ), 
    sharks_score + dolphins_score = total_points ∧ 
    sharks_score - dolphins_score = winning_margin ∧ 
    dolphins_score = 15 := by
sorry

end basketball_score_problem_l774_77455


namespace square_sum_identity_l774_77441

theorem square_sum_identity (y : ℝ) :
  (y - 2)^2 + 2*(y - 2)*(4 + y) + (4 + y)^2 = 4*(y + 1)^2 := by
  sorry

end square_sum_identity_l774_77441


namespace binary_representation_of_25_l774_77467

/-- Converts a natural number to its binary representation as a list of bits -/
def toBinary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
  let rec toBinaryAux (m : ℕ) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: toBinaryAux (m / 2)
  toBinaryAux n

/-- The binary representation of 25 -/
def binary25 : List Bool := [true, false, false, true, true]

/-- Theorem stating that the binary representation of 25 is [1,1,0,0,1] -/
theorem binary_representation_of_25 : toBinary 25 = binary25 := by
  sorry

end binary_representation_of_25_l774_77467


namespace money_left_over_correct_l774_77414

/-- Calculates the money left over after purchases given the specified conditions --/
def money_left_over (
  video_game_cost : ℚ)
  (video_game_discount : ℚ)
  (candy_cost : ℚ)
  (sales_tax : ℚ)
  (shipping_fee : ℚ)
  (babysitting_rate : ℚ)
  (bonus_rate : ℚ)
  (hours_worked : ℕ)
  (bonus_threshold : ℕ) : ℚ :=
  let discounted_game_cost := video_game_cost * (1 - video_game_discount)
  let total_before_tax := discounted_game_cost + shipping_fee + candy_cost
  let total_cost := total_before_tax * (1 + sales_tax)
  let regular_hours := min hours_worked bonus_threshold
  let bonus_hours := hours_worked - regular_hours
  let total_earnings := babysitting_rate * hours_worked + bonus_rate * bonus_hours
  total_earnings - total_cost

theorem money_left_over_correct :
  money_left_over 60 0.15 5 0.10 3 8 2 9 5 = 151/10 := by
  sorry

end money_left_over_correct_l774_77414


namespace complex_abs_one_plus_i_over_i_l774_77461

theorem complex_abs_one_plus_i_over_i (i : ℂ) : i * i = -1 → Complex.abs ((1 + i) / i) = Real.sqrt 2 := by
  sorry

end complex_abs_one_plus_i_over_i_l774_77461


namespace equal_roots_quadratic_l774_77472

theorem equal_roots_quadratic (k : ℝ) : 
  (∃ x : ℝ, x^2 - 2*x + k = 0 ∧ 
   ∀ y : ℝ, y^2 - 2*y + k = 0 → y = x) → 
  k = 1 := by
sorry

end equal_roots_quadratic_l774_77472


namespace trigonometric_identities_l774_77434

theorem trigonometric_identities (x : Real) 
  (h1 : -π < x ∧ x < 0) 
  (h2 : Real.sin x + Real.cos x = 1/5) : 
  (Real.sin x - Real.cos x = -7/5) ∧ 
  ((3 * (Real.sin (x/2))^2 - 2 * Real.sin (x/2) * Real.cos (x/2) + (Real.cos (x/2))^2) / 
   (Real.tan x + 1 / Real.tan x) = -132/125) := by
  sorry

end trigonometric_identities_l774_77434


namespace sector_angle_l774_77489

theorem sector_angle (r : ℝ) (α : ℝ) (h1 : α * r = 5) (h2 : (1/2) * α * r^2 = 5) : α = 5/2 := by
  sorry

end sector_angle_l774_77489


namespace quadratic_form_ratio_l774_77476

theorem quadratic_form_ratio (k : ℝ) :
  ∃ (c r s : ℝ), 10 * k^2 - 6 * k + 20 = c * (k + r)^2 + s ∧ s / r = -191 / 3 :=
by sorry

end quadratic_form_ratio_l774_77476


namespace max_min_f_a4_range_a_inequality_l774_77457

-- Define the function f
def f (a x : ℝ) : ℝ := x * abs (x - a) + 2 * x - 3

-- Theorem for part 1
theorem max_min_f_a4 :
  ∃ (max min : ℝ),
    (∀ x, 2 ≤ x ∧ x ≤ 5 → f 4 x ≤ max) ∧
    (∃ x, 2 ≤ x ∧ x ≤ 5 ∧ f 4 x = max) ∧
    (∀ x, 2 ≤ x ∧ x ≤ 5 → min ≤ f 4 x) ∧
    (∃ x, 2 ≤ x ∧ x ≤ 5 ∧ f 4 x = min) ∧
    max = 12 ∧ min = 5 :=
sorry

-- Theorem for part 2
theorem range_a_inequality :
  ∀ a : ℝ,
    (∀ x, 1 ≤ x ∧ x ≤ 2 → f a x ≤ 2 * x - 2) ↔
    (3 / 2 ≤ a ∧ a ≤ 2) :=
sorry

end max_min_f_a4_range_a_inequality_l774_77457


namespace complex_equation_imaginary_part_l774_77484

theorem complex_equation_imaginary_part :
  ∀ z : ℂ, (4 + 3*I)*z = Complex.abs (3 - 4*I) → Complex.im z = -3/5 :=
by sorry

end complex_equation_imaginary_part_l774_77484


namespace cubic_sequence_with_two_squares_exists_l774_77438

/-- A cubic sequence with integer coefficients -/
def cubic_sequence (b c d : ℤ) (n : ℤ) : ℤ :=
  n^3 + b*n^2 + c*n + d

/-- Predicate for perfect squares -/
def is_perfect_square (x : ℤ) : Prop :=
  ∃ k : ℤ, x = k^2

theorem cubic_sequence_with_two_squares_exists :
  ∃ (b c d : ℤ),
    (is_perfect_square (cubic_sequence b c d 2015)) ∧
    (is_perfect_square (cubic_sequence b c d 2016)) ∧
    (∀ n : ℤ, n ≠ 2015 → n ≠ 2016 → ¬(is_perfect_square (cubic_sequence b c d n))) ∧
    (cubic_sequence b c d 2015 * cubic_sequence b c d 2016 = 0) :=
sorry

end cubic_sequence_with_two_squares_exists_l774_77438


namespace max_min_values_on_interval_l774_77422

def f (x : ℝ) := 2 * x^3 - 3 * x^2 - 12 * x + 5

theorem max_min_values_on_interval :
  ∃ (max min : ℝ),
    (∀ x ∈ Set.Icc 0 3, f x ≤ max) ∧
    (∃ x ∈ Set.Icc 0 3, f x = max) ∧
    (∀ x ∈ Set.Icc 0 3, min ≤ f x) ∧
    (∃ x ∈ Set.Icc 0 3, f x = min) ∧
    max = 5 ∧ min = -15 := by
  sorry

end max_min_values_on_interval_l774_77422


namespace complex_sum_equals_negative_one_l774_77425

theorem complex_sum_equals_negative_one (z : ℂ) (h : z = Complex.exp (2 * Real.pi * Complex.I / 9)) :
  z^2 / (1 + z^3) + z^4 / (1 + z^6) + z^6 / (1 + z^9) = -1 := by
  sorry

end complex_sum_equals_negative_one_l774_77425


namespace limit_exp_sin_ratio_l774_77416

theorem limit_exp_sin_ratio : 
  ∀ ε > 0, ∃ δ > 0, ∀ x ≠ 0, |x| < δ → 
    |((Real.exp (2*x) - Real.exp x) / (Real.sin (2*x) - Real.sin x)) - 1| < ε := by
sorry

end limit_exp_sin_ratio_l774_77416


namespace smallest_distance_between_circles_l774_77462

theorem smallest_distance_between_circles (z w : ℂ) 
  (hz : Complex.abs (z - (2 - 4*I)) = 2)
  (hw : Complex.abs (w - (5 - 6*I)) = 4) :
  ∃ (min_dist : ℝ), min_dist = Real.sqrt 13 - 6 ∧
    ∀ (z' w' : ℂ), Complex.abs (z' - (2 - 4*I)) = 2 →
      Complex.abs (w' - (5 - 6*I)) = 4 →
        Complex.abs (z' - w') ≥ min_dist :=
by sorry

end smallest_distance_between_circles_l774_77462


namespace max_ab_internally_tangent_circles_l774_77436

/-- Two circles C₁ and C₂ are internally tangent if the distance between their centers
    is equal to the difference of their radii. -/
def internally_tangent (a b : ℝ) : Prop :=
  (a + b)^2 = 1

/-- The equation of circle C₁ -/
def C₁ (x y a : ℝ) : Prop :=
  (x - a)^2 + (y + 2)^2 = 4

/-- The equation of circle C₂ -/
def C₂ (x y b : ℝ) : Prop :=
  (x + b)^2 + (y + 2)^2 = 1

/-- The theorem stating that the maximum value of ab is 1/4 -/
theorem max_ab_internally_tangent_circles (a b : ℝ) :
  internally_tangent a b → a * b ≤ 1/4 ∧ ∃ a b, internally_tangent a b ∧ a * b = 1/4 :=
sorry

end max_ab_internally_tangent_circles_l774_77436


namespace blue_line_length_calculation_l774_77410

-- Define the length of the white line
def white_line_length : ℝ := 7.666666666666667

-- Define the difference between the white and blue lines
def length_difference : ℝ := 4.333333333333333

-- Define the length of the blue line
def blue_line_length : ℝ := white_line_length - length_difference

-- Theorem statement
theorem blue_line_length_calculation : 
  blue_line_length = 3.333333333333334 := by sorry

end blue_line_length_calculation_l774_77410


namespace max_product_with_sum_constraint_l774_77471

theorem max_product_with_sum_constraint :
  ∃ (x : ℤ), 
    (∀ y : ℤ, x * (340 - x) ≥ y * (340 - y)) ∧ 
    (x * (340 - x) > 2000) ∧
    (x * (340 - x) = 28900) := by
  sorry

end max_product_with_sum_constraint_l774_77471


namespace johns_purchase_cost_l774_77495

/-- Calculates the total cost of John's purchase of soap and shampoo. -/
def total_cost (soap_bars : ℕ) (soap_weight : ℝ) (soap_price : ℝ)
                (shampoo_bottles : ℕ) (shampoo_weight : ℝ) (shampoo_price : ℝ) : ℝ :=
  (soap_bars : ℝ) * soap_weight * soap_price +
  (shampoo_bottles : ℝ) * shampoo_weight * shampoo_price

/-- Proves that John's total spending on soap and shampoo is $41.40. -/
theorem johns_purchase_cost : 
  total_cost 20 1.5 0.5 15 2.2 0.8 = 41.40 := by
  sorry

end johns_purchase_cost_l774_77495


namespace two_year_compound_interest_l774_77424

/-- Calculates the final amount after two years of compound interest with variable rates -/
def final_amount (initial : ℝ) (rate1 : ℝ) (rate2 : ℝ) : ℝ :=
  initial * (1 + rate1) * (1 + rate2)

/-- Theorem stating that given the specific initial amount and interest rates, 
    the final amount after two years is 82432 -/
theorem two_year_compound_interest :
  final_amount 64000 0.12 0.15 = 82432 := by
  sorry

#eval final_amount 64000 0.12 0.15

end two_year_compound_interest_l774_77424


namespace hyperbola_eccentricity_l774_77412

/-- The eccentricity of a hyperbola with given conditions -/
theorem hyperbola_eccentricity (a b c : ℝ) (m n : ℝ) :
  a > 0 →
  b > 0 →
  c = (a^2 + b^2).sqrt →
  (∀ x y, x^2 / a^2 - y^2 / b^2 = 1 ↔ ((x, y) : ℝ × ℝ) ∈ {p | p.1^2 / a^2 - p.2^2 / b^2 = 1}) →
  (c, 0) ∈ {p | p.1^2 / a^2 - p.2^2 / b^2 = 1} →
  ((m + n) * c, (m - n) * b * c / a) ∈ {p | p.1^2 / a^2 - p.2^2 / b^2 = 1} →
  m * n = 2 / 9 →
  (a^2 + b^2) / a^2 = (3 * Real.sqrt 2 / 4)^2 := by
  sorry

end hyperbola_eccentricity_l774_77412


namespace calculate_expression_l774_77479

theorem calculate_expression : (30 / (10 - 2 * 3))^2 = 56.25 := by
  sorry

end calculate_expression_l774_77479


namespace coaching_charges_calculation_l774_77429

/-- Number of days from January 1 to November 4 in a non-leap year -/
def daysOfCoaching : Nat := 308

/-- Total payment for coaching in dollars -/
def totalPayment : Int := 7038

/-- Daily coaching charges in dollars -/
def dailyCharges : ℚ := totalPayment / daysOfCoaching

theorem coaching_charges_calculation :
  dailyCharges = 7038 / 308 := by sorry

end coaching_charges_calculation_l774_77429


namespace max_product_sum_2000_l774_77466

theorem max_product_sum_2000 :
  (∃ (a b : ℤ), a + b = 2000 ∧ ∀ (x y : ℤ), x + y = 2000 → x * y ≤ a * b) ∧
  (∀ (a b : ℤ), a + b = 2000 → a * b ≤ 1000000) ∧
  (∃ (a b : ℤ), a + b = 2000 ∧ a * b = 1000000) :=
by sorry

end max_product_sum_2000_l774_77466


namespace ellipse_properties_l774_77451

/-- Definition of the ellipse C -/
def ellipse_C (x y : ℝ) : Prop :=
  x^2 / 9 + y^2 / 5 = 1

/-- Definition of line l -/
def line_l (x y : ℝ) : Prop :=
  y = Real.sqrt 3 / 3 * (x + 2) ∨ y = -Real.sqrt 3 / 3 * (x + 2)

/-- Point on the ellipse -/
structure PointOnEllipse where
  x : ℝ
  y : ℝ
  on_ellipse : ellipse_C x y

/-- Theorem statement -/
theorem ellipse_properties :
  let a := 3
  let b := Real.sqrt 5
  let e := 2/3
  ∃ (F₁ F₂ : ℝ × ℝ) (A : ℝ × ℝ),
    -- C passes through (0, √5)
    ellipse_C 0 (Real.sqrt 5) ∧
    -- Eccentricity is 2/3
    Real.sqrt (F₁.1^2 + F₁.2^2) / a = e ∧
    -- A is on x = 4
    A.1 = 4 ∧
    -- When perpendicular bisector of F₁A passes through F₂, l has the given equation
    (∀ x y, line_l x y ↔ (y - F₁.2) / (x - F₁.1) = (A.2 - F₁.2) / (A.1 - F₁.1)) ∧
    -- Minimum length of AB
    ∃ (min_length : ℝ),
      min_length = Real.sqrt 21 ∧
      ∀ (B : PointOnEllipse),
        A.1 * B.x + A.2 * B.y = 0 →  -- OA ⊥ OB
        (A.1 - B.x)^2 + (A.2 - B.y)^2 ≥ min_length^2 :=
by sorry

end ellipse_properties_l774_77451


namespace product_of_x_and_y_l774_77421

theorem product_of_x_and_y (x y : ℝ) (h1 : 3 * x + 4 * y = 60) (h2 : 6 * x - 4 * y = 12) :
  x * y = 72 := by
  sorry

end product_of_x_and_y_l774_77421


namespace rectangle_area_l774_77426

theorem rectangle_area (width length perimeter area : ℝ) : 
  length = 4 * width →
  perimeter = 2 * (length + width) →
  perimeter = 200 →
  area = length * width →
  area = 1600 := by
sorry

end rectangle_area_l774_77426


namespace chicken_count_l774_77442

theorem chicken_count (coop run free_range : ℕ) : 
  coop = 14 →
  run = 2 * coop →
  free_range = 2 * run - 4 →
  free_range = 52 := by
sorry

end chicken_count_l774_77442


namespace production_target_is_1800_l774_77447

/-- Calculates the yearly production target for a car manufacturing company. -/
def yearly_production_target (current_monthly_production : ℕ) (monthly_increase : ℕ) : ℕ :=
  (current_monthly_production + monthly_increase) * 12

/-- Theorem: The yearly production target is 1800 cars. -/
theorem production_target_is_1800 :
  yearly_production_target 100 50 = 1800 := by
  sorry

end production_target_is_1800_l774_77447


namespace flash_drive_problem_l774_77409

/-- Represents the number of flash drives needed to store files -/
def min_flash_drives (total_files : ℕ) (drive_capacity : ℚ) 
  (file_sizes : List (ℕ × ℚ)) : ℕ :=
  sorry

/-- The problem statement -/
theorem flash_drive_problem :
  let total_files : ℕ := 40
  let drive_capacity : ℚ := 2
  let file_sizes : List (ℕ × ℚ) := [(4, 1.2), (16, 0.9), (20, 0.6)]
  min_flash_drives total_files drive_capacity file_sizes = 20 := by
  sorry

end flash_drive_problem_l774_77409


namespace total_dresses_l774_77450

theorem total_dresses (emily melissa debora sophia : ℕ) : 
  emily = 16 ∧ 
  melissa = emily / 2 ∧ 
  debora = melissa + 12 ∧ 
  sophia = debora - 5 → 
  emily + melissa + debora + sophia = 59 := by
sorry

end total_dresses_l774_77450


namespace partial_fraction_decomposition_l774_77474

theorem partial_fraction_decomposition :
  ∃ (P Q R : ℚ), 
    P = -3/5 ∧ Q = -1 ∧ R = 13/5 ∧
    ∀ (x : ℚ), x ≠ 1 → x ≠ 4 → x ≠ 6 →
      (x^2 - 10) / ((x - 1) * (x - 4) * (x - 6)) = 
      P / (x - 1) + Q / (x - 4) + R / (x - 6) := by
  sorry

end partial_fraction_decomposition_l774_77474


namespace z_in_fourth_quadrant_z_on_y_equals_x_l774_77404

/-- The real part of the complex number z -/
def real_part (m : ℝ) : ℝ := m^2 - 8*m + 15

/-- The imaginary part of the complex number z -/
def imag_part (m : ℝ) : ℝ := m^2 - 5*m - 14

/-- The complex number z -/
def z (m : ℝ) : ℂ := Complex.mk (real_part m) (imag_part m)

/-- Condition for z to be in the fourth quadrant -/
def in_fourth_quadrant (m : ℝ) : Prop :=
  real_part m > 0 ∧ imag_part m < 0

/-- Condition for z to be on the line y = x -/
def on_y_equals_x (m : ℝ) : Prop :=
  real_part m = imag_part m

theorem z_in_fourth_quadrant :
  ∀ m : ℝ, in_fourth_quadrant m ↔ (-2 < m ∧ m < 3) ∨ (5 < m ∧ m < 7) :=
sorry

theorem z_on_y_equals_x :
  ∀ m : ℝ, on_y_equals_x m ↔ m = 29/3 :=
sorry

end z_in_fourth_quadrant_z_on_y_equals_x_l774_77404


namespace T_divisibility_l774_77413

def T : Set ℕ := {s | ∃ n : ℕ, s = (n - 2)^2 + (n - 1)^2 + n^2 + (n + 1)^2}

theorem T_divisibility :
  (∀ s ∈ T, ¬(9 ∣ s)) ∧ (∃ s ∈ T, 4 ∣ s) := by sorry

end T_divisibility_l774_77413


namespace book_pair_count_l774_77453

theorem book_pair_count :
  let num_genres : ℕ := 4
  let books_per_genre : ℕ := 4
  let choose_genres : ℕ := 2
  num_genres.choose choose_genres * books_per_genre^choose_genres = 96 :=
by sorry

end book_pair_count_l774_77453


namespace recycle_128_cans_l774_77411

/-- The number of new cans that can be created through recycling, given an initial number of cans -/
def recycle_cans (initial_cans : ℕ) : ℕ :=
  if initial_cans < 2 then 0
  else (initial_cans / 2) + recycle_cans (initial_cans / 2)

/-- Theorem stating that recycling 128 cans produces 127 new cans -/
theorem recycle_128_cans :
  recycle_cans 128 = 127 := by
  sorry

end recycle_128_cans_l774_77411


namespace combined_work_theorem_l774_77477

/-- The time taken for three workers to complete a task together, given their individual completion times -/
def combined_completion_time (time_A time_B time_C : ℚ) : ℚ :=
  1 / (1 / time_A + 1 / time_B + 1 / time_C)

/-- Theorem: Given the individual completion times, the combined completion time is 72/13 days -/
theorem combined_work_theorem :
  combined_completion_time 12 18 24 = 72 / 13 := by
  sorry

end combined_work_theorem_l774_77477


namespace white_surface_area_fraction_l774_77459

/-- Represents a cube with side length and number of smaller cubes -/
structure Cube where
  side_length : ℕ
  num_smaller_cubes : ℕ

/-- Represents the composition of a larger cube -/
structure CubeComposition where
  large_cube : Cube
  small_cube : Cube
  num_red : ℕ
  num_white : ℕ

/-- Calculate the surface area of a cube -/
def surface_area (c : Cube) : ℕ := 6 * c.side_length^2

/-- Calculate the minimum number of visible faces for white cubes -/
def min_visible_white_faces (cc : CubeComposition) : ℕ :=
  cc.num_white - 1

/-- The theorem stating the fraction of white surface area -/
theorem white_surface_area_fraction (cc : CubeComposition) 
  (h1 : cc.large_cube.side_length = 4)
  (h2 : cc.small_cube.side_length = 1)
  (h3 : cc.large_cube.num_smaller_cubes = 64)
  (h4 : cc.num_red = 56)
  (h5 : cc.num_white = 8) :
  (min_visible_white_faces cc : ℚ) / (surface_area cc.large_cube : ℚ) = 7 / 96 := by
  sorry

end white_surface_area_fraction_l774_77459


namespace movie_of_the_year_fraction_l774_77468

/-- The required fraction for a film to be considered for "movie of the year" -/
def required_fraction (total_members : ℕ) (min_lists : ℚ) : ℚ :=
  min_lists / total_members

/-- Theorem stating the required fraction for the Cinematic Academy's "movie of the year" consideration -/
theorem movie_of_the_year_fraction :
  required_fraction 765 (191.25 : ℚ) = 0.25 := by
  sorry

end movie_of_the_year_fraction_l774_77468


namespace isosceles_triangle_base_length_l774_77492

/-- An isosceles triangle with perimeter 18 and one side 4 has base length 7 -/
theorem isosceles_triangle_base_length : 
  ∀ (a b c : ℝ), 
    a + b + c = 18 →  -- perimeter is 18
    (a = b ∨ b = c ∨ a = c) →  -- isosceles condition
    (a = 4 ∨ b = 4 ∨ c = 4) →  -- one side is 4
    (a + b > c ∧ b + c > a ∧ a + c > b) →  -- triangle inequality
    (if a = b then c else if b = c then a else b) = 7 :=
by sorry

end isosceles_triangle_base_length_l774_77492


namespace triangle_inequality_l774_77482

theorem triangle_inequality (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ b + c > a ∧ c + a > b) :
  |((a - b) / (a + b) + (b - c) / (b + c) + (c - a) / (c + a))| < 1 :=
sorry

end triangle_inequality_l774_77482


namespace bobs_age_problem_l774_77432

theorem bobs_age_problem :
  ∃ (n : ℕ), 
    (∃ (k : ℕ), n - 3 = k^2) ∧ 
    (∃ (j : ℕ), n + 4 = j^3) ∧ 
    n = 725 := by
  sorry

end bobs_age_problem_l774_77432


namespace quadratic_equation_equivalence_l774_77443

theorem quadratic_equation_equivalence :
  ∀ x : ℝ, (2 * x^2 - 12 * x + 1 = 0) ↔ ((x - 3)^2 = 17/2) :=
by sorry

end quadratic_equation_equivalence_l774_77443


namespace average_age_of_students_l774_77452

theorem average_age_of_students (num_students : ℕ) (teacher_age : ℕ) (total_average : ℕ) 
  (h1 : num_students = 40)
  (h2 : teacher_age = 56)
  (h3 : total_average = 16) :
  (num_students * total_average - teacher_age) / num_students = 15 := by
  sorry

end average_age_of_students_l774_77452


namespace juanitas_dessert_cost_l774_77481

/-- Represents the cost of a brownie dessert with various toppings -/
def brownieDessertCost (brownieCost iceCreamCost syrupCost nutsCost : ℚ)
  (iceCreamScoops syrupServings : ℕ) (includeNuts : Bool) : ℚ :=
  brownieCost +
  iceCreamCost * iceCreamScoops +
  syrupCost * syrupServings +
  (if includeNuts then nutsCost else 0)

/-- Proves that Juanita's dessert costs $7.00 given the prices and her order -/
theorem juanitas_dessert_cost :
  let brownieCost : ℚ := 5/2
  let iceCreamCost : ℚ := 1
  let syrupCost : ℚ := 1/2
  let nutsCost : ℚ := 3/2
  let iceCreamScoops : ℕ := 2
  let syrupServings : ℕ := 2
  let includeNuts : Bool := true
  brownieDessertCost brownieCost iceCreamCost syrupCost nutsCost
    iceCreamScoops syrupServings includeNuts = 7 :=
by
  sorry

end juanitas_dessert_cost_l774_77481


namespace ronald_store_visits_l774_77490

def store_visits (bananas_per_visit : ℕ) (total_bananas : ℕ) : ℕ :=
  total_bananas / bananas_per_visit

theorem ronald_store_visits :
  let bananas_per_visit := 10
  let total_bananas := 20
  store_visits bananas_per_visit total_bananas = 2 := by
  sorry

end ronald_store_visits_l774_77490


namespace sum_of_numbers_l774_77498

theorem sum_of_numbers (a b c : ℝ) : 
  a = 0.8 → b = 1/2 → c = 0.5 → a < 2 ∧ b < 2 ∧ c < 2 → a + b + c = 1.8 := by
sorry

end sum_of_numbers_l774_77498


namespace cookies_left_l774_77469

def cookies_problem (days : ℕ) (trays_per_day : ℕ) (cookies_per_tray : ℕ) 
  (frank_eats_per_day : ℕ) (ted_eats : ℕ) : ℕ :=
  days * trays_per_day * cookies_per_tray - days * frank_eats_per_day - ted_eats

theorem cookies_left : 
  cookies_problem 6 2 12 1 4 = 134 := by
  sorry

end cookies_left_l774_77469


namespace decimal_fraction_sum_equals_one_l774_77499

theorem decimal_fraction_sum_equals_one : ∃ (a b c d e f g h : Nat),
  (a = 2 ∨ a = 3) ∧ (b = 2 ∨ b = 3) ∧
  (c = 2 ∨ c = 3) ∧ (d = 2 ∨ d = 3) ∧
  (e = 2 ∨ e = 3) ∧ (f = 2 ∨ f = 3) ∧
  (g = 2 ∨ g = 3) ∧ (h = 2 ∨ h = 3) ∧
  (a * 10 + b) / 100 + (c * 10 + d) / 100 + (e * 10 + f) / 100 + (g * 10 + h) / 100 = 1 := by
  sorry

end decimal_fraction_sum_equals_one_l774_77499


namespace circle_equation_l774_77463

/-- A circle with center (2, -3) passing through the origin has the equation (x - 2)^2 + (y + 3)^2 = 13 -/
theorem circle_equation (x y : ℝ) : 
  (∃ (r : ℝ), r > 0 ∧ 
    ((x - 2)^2 + (y + 3)^2 = r^2) ∧ 
    (0 - 2)^2 + (0 + 3)^2 = r^2) ↔ 
  (x - 2)^2 + (y + 3)^2 = 13 :=
by sorry

end circle_equation_l774_77463


namespace min_q_for_three_solutions_l774_77428

theorem min_q_for_three_solutions (p q : ℝ) :
  (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧
    (∀ x : ℝ, |x^2 + p*x + q| = 3 ↔ (x = x₁ ∨ x = x₂ ∨ x = x₃))) →
  q ≥ -3 ∧ ∃ p₀ : ℝ, ∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧
    (∀ x : ℝ, |x^2 + p₀*x + (-3)| = 3 ↔ (x = x₁ ∨ x = x₂ ∨ x = x₃)) :=
by
  sorry

end min_q_for_three_solutions_l774_77428


namespace combined_squares_perimeter_l774_77470

/-- The perimeter of the resulting figure when combining two squares -/
theorem combined_squares_perimeter (p1 p2 : ℝ) (h1 : p1 = 40) (h2 : p2 = 100) : 
  p1 + p2 - 2 * (p1 / 4) = 120 :=
by sorry

end combined_squares_perimeter_l774_77470


namespace functional_equation_solution_l774_77423

/-- A function satisfying the given functional equation -/
def SatisfiesEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x^2 + y^2) = f (x^2 - y^2) + f (2*x*y)

/-- The main theorem -/
theorem functional_equation_solution :
  ∀ f : ℝ → ℝ, (∀ x, f x ≥ 0) → SatisfiesEquation f →
  ∃ a : ℝ, a ≥ 0 ∧ ∀ x, f x = a * x^2 :=
sorry

end functional_equation_solution_l774_77423


namespace max_a_value_l774_77465

theorem max_a_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + y + 6 = 4 * x * y) :
  ∃ (a : ℝ), ∀ (b : ℝ), (∀ (u v : ℝ), u > 0 → v > 0 → u + v + 6 = 4 * u * v →
    u^2 + 2 * u * v + v^2 - b * u - b * v + 1 ≥ 0) → b ≤ a ∧ a = 10 / 3 :=
sorry

end max_a_value_l774_77465


namespace rolling_semicircle_path_length_l774_77437

/-- The length of the path traveled by the center of a rolling semicircular arc -/
theorem rolling_semicircle_path_length (r : ℝ) (h : r > 0) :
  let path_length := 3 * Real.pi * r
  path_length = (Real.pi * (2 * r)) / 2 :=
by sorry

end rolling_semicircle_path_length_l774_77437


namespace platform_length_platform_length_is_210_l774_77405

/-- Given a train's speed and time to pass a platform and a man, calculate the platform length -/
theorem platform_length 
  (train_speed : ℝ) 
  (time_platform : ℝ) 
  (time_man : ℝ) : ℝ :=
  let train_speed_ms := train_speed * (1000 / 3600)
  let train_length := train_speed_ms * time_man
  let platform_length := train_speed_ms * time_platform - train_length
  platform_length

/-- The length of the platform is 210 meters -/
theorem platform_length_is_210 :
  platform_length 54 34 20 = 210 := by
  sorry

end platform_length_platform_length_is_210_l774_77405


namespace cos_odd_function_phi_l774_77460

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem cos_odd_function_phi (φ : ℝ) 
  (h1 : 0 ≤ φ) (h2 : φ ≤ π) 
  (h3 : is_odd_function (fun x ↦ Real.cos (x + φ))) : 
  φ = π / 2 := by
  sorry

end cos_odd_function_phi_l774_77460


namespace special_polynomial_B_value_l774_77427

/-- A polynomial of degree 5 with specific properties -/
structure SpecialPolynomial where
  A : ℤ
  B : ℤ
  C : ℤ
  roots : Finset ℤ
  roots_positive : ∀ r ∈ roots, r > 0
  roots_sum : (roots.sum id) = 15
  roots_card : roots.card = 5
  is_root : ∀ r ∈ roots, r^5 - 15*r^4 + A*r^3 + B*r^2 + C*r + 24 = 0

/-- The coefficient B in the special polynomial is -90 -/
theorem special_polynomial_B_value (p : SpecialPolynomial) : p.B = -90 := by
  sorry

end special_polynomial_B_value_l774_77427


namespace intersection_equals_open_interval_l774_77485

-- Define the sets M and N
def M : Set ℝ := {x | x > 1}
def N : Set ℝ := {x | x < 5}

-- State the theorem
theorem intersection_equals_open_interval :
  M ∩ N = Set.Ioo 1 5 := by sorry

end intersection_equals_open_interval_l774_77485


namespace f_is_quadratic_l774_77496

/-- Definition of a quadratic equation -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The equation 2x^2 - x - 3 = 0 -/
def f (x : ℝ) : ℝ := 2 * x^2 - x - 3

/-- Theorem: f is a quadratic equation -/
theorem f_is_quadratic : is_quadratic_equation f := by
  sorry


end f_is_quadratic_l774_77496


namespace line_point_x_value_l774_77456

/-- Given a line passing through points (x, -4) and (4, 1) with a slope of 1, prove that x = -1 -/
theorem line_point_x_value (x : ℝ) : 
  let p1 : ℝ × ℝ := (x, -4)
  let p2 : ℝ × ℝ := (4, 1)
  let slope : ℝ := (p2.2 - p1.2) / (p2.1 - p1.1)
  slope = 1 → x = -1 := by
  sorry

end line_point_x_value_l774_77456


namespace problem_statement_l774_77435

theorem problem_statement (a b : ℝ) 
  (h1 : |a| = 5)
  (h2 : |b| = 7)
  (h3 : |a + b| = a + b) :
  a - b = -2 := by
sorry

end problem_statement_l774_77435
