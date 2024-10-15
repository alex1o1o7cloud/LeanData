import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_and_inequality_solution_l413_41385

theorem quadratic_and_inequality_solution :
  (∃ x₁ x₂ : ℝ, x₁ = 1 + Real.sqrt 5 ∧ x₂ = 1 - Real.sqrt 5 ∧
    (x₁^2 - 2*x₁ - 4 = 0) ∧ (x₂^2 - 2*x₂ - 4 = 0)) ∧
  (∀ x : ℝ, (2*(x-1) ≥ -4 ∧ (3*x-6)/2 < x-1) ↔ (-1 ≤ x ∧ x < 4)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_and_inequality_solution_l413_41385


namespace NUMINAMATH_CALUDE_frequency_converges_to_probability_l413_41389

-- Define a random experiment
def RandomExperiment : Type := Unit

-- Define an event in the experiment
def Event (e : RandomExperiment) : Type := Unit

-- Define the probability of an event
def probability (e : RandomExperiment) (A : Event e) : ℝ := sorry

-- Define the frequency of an event after n trials
def frequency (e : RandomExperiment) (A : Event e) (n : ℕ) : ℝ := sorry

-- Theorem: As the number of trials approaches infinity, 
-- the frequency converges to the probability
theorem frequency_converges_to_probability 
  (e : RandomExperiment) (A : Event e) : 
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, 
  |frequency e A n - probability e A| < ε :=
sorry

end NUMINAMATH_CALUDE_frequency_converges_to_probability_l413_41389


namespace NUMINAMATH_CALUDE_cubic_roots_product_l413_41301

theorem cubic_roots_product (a b c : ℝ) : 
  (x^3 - 26*x^2 + 32*x - 15 = 0 → x = a ∨ x = b ∨ x = c) →
  (1 + a) * (1 + b) * (1 + c) = 74 := by
sorry

end NUMINAMATH_CALUDE_cubic_roots_product_l413_41301


namespace NUMINAMATH_CALUDE_french_exam_words_to_learn_l413_41316

/-- The least number of words to learn for a French exam -/
def least_words_to_learn : ℕ := 569

theorem french_exam_words_to_learn 
  (total_words : ℕ) 
  (recall_rate : ℚ) 
  (target_recall : ℚ) 
  (h1 : total_words = 600)
  (h2 : recall_rate = 95 / 100)
  (h3 : target_recall = 90 / 100) :
  (↑least_words_to_learn : ℚ) ≥ (target_recall * total_words) / recall_rate ∧ 
  (↑(least_words_to_learn - 1) : ℚ) < (target_recall * total_words) / recall_rate :=
sorry

end NUMINAMATH_CALUDE_french_exam_words_to_learn_l413_41316


namespace NUMINAMATH_CALUDE_shekars_science_marks_l413_41306

theorem shekars_science_marks
  (math_marks : ℕ)
  (social_marks : ℕ)
  (english_marks : ℕ)
  (biology_marks : ℕ)
  (average_marks : ℚ)
  (num_subjects : ℕ)
  (h1 : math_marks = 76)
  (h2 : social_marks = 82)
  (h3 : english_marks = 67)
  (h4 : biology_marks = 75)
  (h5 : average_marks = 73)
  (h6 : num_subjects = 5) :
  ∃ (science_marks : ℕ),
    (math_marks + social_marks + english_marks + biology_marks + science_marks) / num_subjects = average_marks ∧
    science_marks = 65 := by
  sorry

end NUMINAMATH_CALUDE_shekars_science_marks_l413_41306


namespace NUMINAMATH_CALUDE_arccos_sin_one_point_five_l413_41322

theorem arccos_sin_one_point_five :
  Real.arccos (Real.sin 1.5) = π / 2 - 1.5 := by
  sorry

end NUMINAMATH_CALUDE_arccos_sin_one_point_five_l413_41322


namespace NUMINAMATH_CALUDE_triangle_properties_l413_41382

-- Define the triangle ABC
def triangle_ABC (a b c : ℝ) (A B C : ℝ) : Prop :=
  -- Add conditions here
  c = Real.sqrt 2 ∧
  Real.cos C = 3/4 ∧
  2 * c * Real.sin A = b * Real.sin C

-- State the theorem
theorem triangle_properties (a b c : ℝ) (A B C : ℝ) 
  (h : triangle_ABC a b c A B C) : 
  b = 2 ∧ 
  Real.sin A = Real.sqrt 14 / 8 ∧ 
  Real.sin (2 * A + π/3) = (5 * Real.sqrt 7 + 9 * Real.sqrt 3) / 32 := by
  sorry


end NUMINAMATH_CALUDE_triangle_properties_l413_41382


namespace NUMINAMATH_CALUDE_coplanar_points_scalar_l413_41351

theorem coplanar_points_scalar (O E F G H : EuclideanSpace ℝ (Fin 3)) (m : ℝ) :
  (O = 0) →
  (4 • (E - O) - 3 • (F - O) + 2 • (G - O) + m • (H - O) = 0) →
  (∃ (a b c d : ℝ), a • (E - O) + b • (F - O) + c • (G - O) + d • (H - O) = 0 ∧ 
    (a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0 ∨ d ≠ 0)) →
  m = -3 :=
by sorry

end NUMINAMATH_CALUDE_coplanar_points_scalar_l413_41351


namespace NUMINAMATH_CALUDE_class_size_problem_l413_41356

theorem class_size_problem (x : ℕ) : 
  x ≥ 46 → 
  (7 : ℚ) / 24 * x < 15 → 
  x = 48 :=
by sorry

end NUMINAMATH_CALUDE_class_size_problem_l413_41356


namespace NUMINAMATH_CALUDE_blue_balls_count_l413_41373

theorem blue_balls_count (red_balls : ℕ) (blue_balls : ℕ) 
  (h1 : red_balls = 25)
  (h2 : red_balls = 2 * blue_balls + 3) :
  blue_balls = 11 := by
  sorry

end NUMINAMATH_CALUDE_blue_balls_count_l413_41373


namespace NUMINAMATH_CALUDE_three_intersection_range_l413_41370

def f (x : ℝ) := x^3 - 3*x

theorem three_intersection_range :
  ∃ (a_min a_max : ℝ), a_min < a_max ∧
  (∀ a : ℝ, (∃ (x₁ x₂ x₃ : ℝ), x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧
                               f x₁ = a ∧ f x₂ = a ∧ f x₃ = a) ↔
              a_min < a ∧ a < a_max) ∧
  a_min = -2 ∧ a_max = 2 :=
sorry

end NUMINAMATH_CALUDE_three_intersection_range_l413_41370


namespace NUMINAMATH_CALUDE_senior_ticket_cost_l413_41399

theorem senior_ticket_cost 
  (total_tickets : ℕ) 
  (adult_price : ℕ) 
  (total_receipts : ℕ) 
  (senior_tickets : ℕ) 
  (h1 : total_tickets = 510) 
  (h2 : adult_price = 21) 
  (h3 : total_receipts = 8748) 
  (h4 : senior_tickets = 327) :
  ∃ (senior_price : ℕ), 
    senior_price * senior_tickets + adult_price * (total_tickets - senior_tickets) = total_receipts ∧ 
    senior_price = 15 := by
sorry

end NUMINAMATH_CALUDE_senior_ticket_cost_l413_41399


namespace NUMINAMATH_CALUDE_problem_statement_l413_41315

-- Define the base 10 logarithm
noncomputable def log (x : ℝ) := Real.log x / Real.log 10

-- State the theorem
theorem problem_statement : (2/3)^0 + log 2 + log 5 = 2 := by sorry

end NUMINAMATH_CALUDE_problem_statement_l413_41315


namespace NUMINAMATH_CALUDE_unique_solution_quartic_equation_l413_41369

theorem unique_solution_quartic_equation :
  ∃! x : ℝ, x^4 + (2 - x)^4 + 2*x = 34 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_quartic_equation_l413_41369


namespace NUMINAMATH_CALUDE_largest_possible_a_l413_41357

theorem largest_possible_a (a b c d : ℕ+) 
  (h1 : a < 3 * b) 
  (h2 : b < 4 * c) 
  (h3 : c < 5 * d) 
  (h4 : d < 50) : 
  a ≤ 2924 ∧ ∃ (a' b' c' d' : ℕ+), 
    a' = 2924 ∧ 
    a' < 3 * b' ∧ 
    b' < 4 * c' ∧ 
    c' < 5 * d' ∧ 
    d' < 50 :=
by sorry

end NUMINAMATH_CALUDE_largest_possible_a_l413_41357


namespace NUMINAMATH_CALUDE_sum_of_cubes_divisible_by_three_l413_41362

theorem sum_of_cubes_divisible_by_three (n : ℤ) : 
  ∃ k : ℤ, n^3 + (n+1)^3 + (n+2)^3 = 3 * k := by
sorry

end NUMINAMATH_CALUDE_sum_of_cubes_divisible_by_three_l413_41362


namespace NUMINAMATH_CALUDE_student_score_proof_l413_41318

theorem student_score_proof (total_questions : Nat) (score : Int) 
  (h1 : total_questions = 100)
  (h2 : score = 79) : 
  ∃ (correct incorrect : Nat),
    correct + incorrect = total_questions ∧
    score = correct - 2 * incorrect ∧
    correct = 93 := by
  sorry

end NUMINAMATH_CALUDE_student_score_proof_l413_41318


namespace NUMINAMATH_CALUDE_power_30_mod_7_l413_41310

theorem power_30_mod_7 : 2^30 ≡ 1 [MOD 7] :=
by
  have h : 2^3 ≡ 1 [MOD 7] := by sorry
  sorry

end NUMINAMATH_CALUDE_power_30_mod_7_l413_41310


namespace NUMINAMATH_CALUDE_complex_unit_vector_l413_41360

theorem complex_unit_vector (z : ℂ) (h : z = 3 + 4*I) : z / Complex.abs z = 3/5 + 4/5*I := by
  sorry

end NUMINAMATH_CALUDE_complex_unit_vector_l413_41360


namespace NUMINAMATH_CALUDE_exponential_function_through_point_l413_41376

theorem exponential_function_through_point (a : ℝ) : 
  (∀ x : ℝ, (fun x => a^x) x = a^x) → 
  a^2 = 4 → 
  a > 0 → 
  a ≠ 1 → 
  a = 2 := by
sorry

end NUMINAMATH_CALUDE_exponential_function_through_point_l413_41376


namespace NUMINAMATH_CALUDE_sin_alpha_value_l413_41359

theorem sin_alpha_value (α : ℝ) (h : 3 * Real.sin (2 * α) = Real.cos α) : 
  Real.sin α = 1/6 := by
sorry

end NUMINAMATH_CALUDE_sin_alpha_value_l413_41359


namespace NUMINAMATH_CALUDE_normal_dist_probability_l413_41327

-- Define the normal distribution
def normal_dist (μ σ : ℝ) : Type := Unit

-- Define the probability function
noncomputable def P (X : normal_dist 4 1) (a b : ℝ) : ℝ := sorry

-- State the theorem
theorem normal_dist_probability 
  (X : normal_dist 4 1) 
  (h1 : P X (4 - 2) (4 + 2) = 0.9544) 
  (h2 : P X (4 - 1) (4 + 1) = 0.6826) : 
  P X 5 6 = 0.1359 := by sorry

end NUMINAMATH_CALUDE_normal_dist_probability_l413_41327


namespace NUMINAMATH_CALUDE_order_of_exponents_l413_41398

theorem order_of_exponents : 5^(1/5) > 0.5^(1/5) ∧ 0.5^(1/5) > 0.5^2 := by
  sorry

end NUMINAMATH_CALUDE_order_of_exponents_l413_41398


namespace NUMINAMATH_CALUDE_tree_purchase_equations_l413_41335

/-- Represents the cost of an A-type tree -/
def cost_A : ℕ := 100

/-- Represents the cost of a B-type tree -/
def cost_B : ℕ := 80

/-- Represents the total amount spent -/
def total_spent : ℕ := 8000

/-- Represents the difference in number between A-type and B-type trees -/
def tree_difference : ℕ := 8

theorem tree_purchase_equations (x y : ℕ) :
  (x - y = tree_difference ∧ cost_A * x + cost_B * y = total_spent) ↔
  (x - y = 8 ∧ 100 * x + 80 * y = 8000) :=
sorry

end NUMINAMATH_CALUDE_tree_purchase_equations_l413_41335


namespace NUMINAMATH_CALUDE_race_distance_multiple_of_360_l413_41343

/-- Represents a race between two contestants A and B -/
structure Race where
  speedRatio : Rat  -- Ratio of speeds of A to B
  headStart : ℕ     -- Head start distance for A in meters
  winMargin : ℕ     -- Distance by which A wins in meters

/-- The total distance of the race is a multiple of 360 meters -/
theorem race_distance_multiple_of_360 (race : Race) 
  (h1 : race.speedRatio = 3 / 4)
  (h2 : race.headStart = 140)
  (h3 : race.winMargin = 20) :
  ∃ (k : ℕ), race.headStart + race.winMargin + k * 360 = 
    race.headStart + (4 * (race.headStart + race.winMargin)) / 3 :=
sorry

end NUMINAMATH_CALUDE_race_distance_multiple_of_360_l413_41343


namespace NUMINAMATH_CALUDE_constant_term_position_l413_41350

/-- The position of the constant term in the expansion of (√a - 2/∛a)^30 -/
theorem constant_term_position (a : ℝ) (h : a > 0) : 
  ∃ (r : ℕ), r = 18 ∧ 
  (∀ (k : ℕ), k ≠ r → (90 - 5 * k : ℚ) / 6 ≠ 0) ∧
  (90 - 5 * r : ℚ) / 6 = 0 := by
  sorry

end NUMINAMATH_CALUDE_constant_term_position_l413_41350


namespace NUMINAMATH_CALUDE_power_division_sum_product_l413_41325

theorem power_division_sum_product : (-6)^6 / 6^4 + 4^3 - 7^2 * 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_power_division_sum_product_l413_41325


namespace NUMINAMATH_CALUDE_coupon_one_best_l413_41380

/-- Represents the discount amount for a given coupon and price -/
def discount (coupon : Nat) (price : ℝ) : ℝ :=
  match coupon with
  | 1 => 0.15 * price
  | 2 => 30
  | 3 => 0.25 * (price - 120)
  | _ => 0

/-- Theorem stating when coupon 1 is the best choice -/
theorem coupon_one_best (price : ℝ) :
  (∀ c : Nat, c ≠ 1 → discount 1 price > discount c price) ↔ 200 < price ∧ price < 300 := by
  sorry


end NUMINAMATH_CALUDE_coupon_one_best_l413_41380


namespace NUMINAMATH_CALUDE_total_complaints_over_five_days_l413_41377

/-- Represents the different staff shortage scenarios -/
inductive StaffShortage
  | Normal
  | TwentyPercent
  | FortyPercent

/-- Represents the different self-checkout states -/
inductive SelfCheckout
  | Working
  | PartiallyBroken
  | CompletelyBroken

/-- Represents the different weather conditions -/
inductive Weather
  | Clear
  | Rainy
  | Snowstorm

/-- Represents the different special events -/
inductive SpecialEvent
  | Normal
  | Holiday
  | OngoingSale

/-- Represents the conditions for a single day -/
structure DayConditions where
  staffShortage : StaffShortage
  selfCheckout : SelfCheckout
  weather : Weather
  specialEvent : SpecialEvent

/-- Calculates the number of complaints for a given day based on its conditions -/
def calculateComplaints (baseComplaints : ℕ) (conditions : DayConditions) : ℕ :=
  sorry

/-- The base number of complaints per day -/
def baseComplaints : ℕ := 120

/-- The conditions for each of the five days -/
def dayConditions : List DayConditions := [
  { staffShortage := StaffShortage.TwentyPercent, selfCheckout := SelfCheckout.CompletelyBroken, weather := Weather.Rainy, specialEvent := SpecialEvent.OngoingSale },
  { staffShortage := StaffShortage.FortyPercent, selfCheckout := SelfCheckout.PartiallyBroken, weather := Weather.Clear, specialEvent := SpecialEvent.Holiday },
  { staffShortage := StaffShortage.FortyPercent, selfCheckout := SelfCheckout.CompletelyBroken, weather := Weather.Snowstorm, specialEvent := SpecialEvent.Normal },
  { staffShortage := StaffShortage.Normal, selfCheckout := SelfCheckout.Working, weather := Weather.Rainy, specialEvent := SpecialEvent.OngoingSale },
  { staffShortage := StaffShortage.TwentyPercent, selfCheckout := SelfCheckout.CompletelyBroken, weather := Weather.Clear, specialEvent := SpecialEvent.Holiday }
]

/-- Theorem stating that the total number of complaints over the five days is 1038 -/
theorem total_complaints_over_five_days :
  (dayConditions.map (calculateComplaints baseComplaints)).sum = 1038 := by
  sorry

end NUMINAMATH_CALUDE_total_complaints_over_five_days_l413_41377


namespace NUMINAMATH_CALUDE_employee_discount_price_l413_41324

theorem employee_discount_price (wholesale_cost : ℝ) (markup_percentage : ℝ) (discount_percentage : ℝ) : 
  wholesale_cost = 200 →
  markup_percentage = 0.20 →
  discount_percentage = 0.25 →
  let retail_price := wholesale_cost * (1 + markup_percentage)
  let discounted_price := retail_price * (1 - discount_percentage)
  discounted_price = 180 := by
sorry


end NUMINAMATH_CALUDE_employee_discount_price_l413_41324


namespace NUMINAMATH_CALUDE_angle_D_value_l413_41347

theorem angle_D_value (A B C D : ℝ) 
  (h1 : A + B = 180)
  (h2 : C = D)
  (h3 : A = 2 * D - 10) :
  D = 70 := by
sorry

end NUMINAMATH_CALUDE_angle_D_value_l413_41347


namespace NUMINAMATH_CALUDE_percent_of_percent_l413_41308

theorem percent_of_percent : (3 / 100) / (5 / 100) * 100 = 60 := by
  sorry

end NUMINAMATH_CALUDE_percent_of_percent_l413_41308


namespace NUMINAMATH_CALUDE_translation_vector_exponential_l413_41342

/-- Given two functions f and g, where f(x) = 2^x + 1 and g(x) = 2^(x+1),
    prove that the translation vector (h, k) that transforms the graph of f
    into the graph of g is (-1, -1). -/
theorem translation_vector_exponential (f g : ℝ → ℝ)
  (hf : ∀ x, f x = 2^x + 1)
  (hg : ∀ x, g x = 2^(x+1))
  (h k : ℝ)
  (translation : ∀ x, g x = f (x - h) + k) :
  h = -1 ∧ k = -1 := by
  sorry

end NUMINAMATH_CALUDE_translation_vector_exponential_l413_41342


namespace NUMINAMATH_CALUDE_intersection_empty_implies_m_less_than_negative_one_l413_41312

theorem intersection_empty_implies_m_less_than_negative_one (m : ℝ) : 
  let M := {x : ℝ | x - m ≤ 0}
  let N := {y : ℝ | ∃ x : ℝ, y = (x - 1)^2 - 1}
  M ∩ N = ∅ → m < -1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_empty_implies_m_less_than_negative_one_l413_41312


namespace NUMINAMATH_CALUDE_workers_per_block_l413_41330

/-- Given a company with the following conditions:
  - The total amount for gifts is $6000
  - Each gift costs $2
  - There are 15 blocks in the company
  This theorem proves that there are 200 workers in each block. -/
theorem workers_per_block (total_amount : ℕ) (gift_worth : ℕ) (num_blocks : ℕ)
  (h1 : total_amount = 6000)
  (h2 : gift_worth = 2)
  (h3 : num_blocks = 15) :
  total_amount / gift_worth / num_blocks = 200 := by
  sorry

end NUMINAMATH_CALUDE_workers_per_block_l413_41330


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l413_41375

theorem contrapositive_equivalence (m : ℝ) : 
  (¬(∃ x : ℝ, x^2 = m) → m < 0) ↔ 
  (m ≥ 0 → ∃ x : ℝ, x^2 = m) := by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l413_41375


namespace NUMINAMATH_CALUDE_total_silver_dollars_l413_41314

/-- The number of silver dollars owned by Mr. Chiu -/
def chiu_dollars : ℕ := 56

/-- The number of silver dollars owned by Mr. Phung -/
def phung_dollars : ℕ := chiu_dollars + 16

/-- The number of silver dollars owned by Mr. Ha -/
def ha_dollars : ℕ := phung_dollars + 5

/-- The total number of silver dollars owned by all three -/
def total_dollars : ℕ := chiu_dollars + phung_dollars + ha_dollars

theorem total_silver_dollars : total_dollars = 205 := by
  sorry

end NUMINAMATH_CALUDE_total_silver_dollars_l413_41314


namespace NUMINAMATH_CALUDE_constant_term_expansion_l413_41384

/-- The constant term in the expansion of (x + 2/x)^6 -/
def constant_term : ℕ := 160

/-- The binomial coefficient (n choose k) -/
def binomial (n k : ℕ) : ℕ := sorry

theorem constant_term_expansion :
  constant_term = binomial 6 3 * 2^3 := by sorry

end NUMINAMATH_CALUDE_constant_term_expansion_l413_41384


namespace NUMINAMATH_CALUDE_total_hours_worked_is_48_l413_41329

/-- Calculates the total hours worked given basic pay, overtime rate, and total wage -/
def totalHoursWorked (basicPay : ℚ) (basicHours : ℚ) (overtimeRate : ℚ) (totalWage : ℚ) : ℚ :=
  let basicHourlyRate := basicPay / basicHours
  let overtimeHourlyRate := basicHourlyRate * (1 + overtimeRate)
  let overtimePay := totalWage - basicPay
  let overtimeHours := overtimePay / overtimeHourlyRate
  basicHours + overtimeHours

/-- Theorem stating that under given conditions, the total hours worked is 48 -/
theorem total_hours_worked_is_48 :
  totalHoursWorked 20 40 (1/4) 25 = 48 := by
  sorry

end NUMINAMATH_CALUDE_total_hours_worked_is_48_l413_41329


namespace NUMINAMATH_CALUDE_six_people_three_events_outcomes_l413_41374

/-- The number of possible outcomes for champions in a competition. -/
def championOutcomes (people : ℕ) (events : ℕ) : ℕ :=
  people ^ events

/-- Theorem stating the number of possible outcomes for 6 people in 3 events. -/
theorem six_people_three_events_outcomes :
  championOutcomes 6 3 = 216 := by
  sorry

end NUMINAMATH_CALUDE_six_people_three_events_outcomes_l413_41374


namespace NUMINAMATH_CALUDE_harolds_marbles_distribution_l413_41387

theorem harolds_marbles_distribution (total_marbles : ℕ) (kept_marbles : ℕ) 
  (best_friends : ℕ) (marbles_per_best_friend : ℕ) (cousins : ℕ) (marbles_per_cousin : ℕ) 
  (school_friends : ℕ) :
  total_marbles = 5000 →
  kept_marbles = 250 →
  best_friends = 3 →
  marbles_per_best_friend = 100 →
  cousins = 5 →
  marbles_per_cousin = 75 →
  school_friends = 10 →
  (total_marbles - (kept_marbles + best_friends * marbles_per_best_friend + 
    cousins * marbles_per_cousin)) / school_friends = 407 := by
  sorry

#check harolds_marbles_distribution

end NUMINAMATH_CALUDE_harolds_marbles_distribution_l413_41387


namespace NUMINAMATH_CALUDE_no_function_satisfies_equation_l413_41358

theorem no_function_satisfies_equation :
  ¬ ∃ f : ℕ → ℕ, ∀ x : ℕ, f (2 * f x) = x + 1998 := by
  sorry

end NUMINAMATH_CALUDE_no_function_satisfies_equation_l413_41358


namespace NUMINAMATH_CALUDE_complex_division_sum_l413_41333

theorem complex_division_sum (a b : ℝ) : 
  (Complex.I + 1) / (Complex.I - 1) = Complex.mk a b → a + b = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_division_sum_l413_41333


namespace NUMINAMATH_CALUDE_camping_trip_attendance_l413_41303

/-- The percentage of students who went to the camping trip -/
def camping_trip_percentage : ℝ := 14

/-- The percentage of students who went to the music festival -/
def music_festival_percentage : ℝ := 8

/-- The percentage of students who participated in the sports league -/
def sports_league_percentage : ℝ := 6

/-- The percentage of camping trip attendees who spent more than $100 -/
def camping_trip_high_cost_percentage : ℝ := 60

/-- The percentage of music festival attendees who spent more than $90 -/
def music_festival_high_cost_percentage : ℝ := 80

/-- The percentage of sports league participants who paid more than $70 -/
def sports_league_high_cost_percentage : ℝ := 75

theorem camping_trip_attendance : 
  camping_trip_percentage = 14 := by sorry

end NUMINAMATH_CALUDE_camping_trip_attendance_l413_41303


namespace NUMINAMATH_CALUDE_staircase_climbing_ways_l413_41339

/-- The number of ways to climb n steps, where one can go up by 1, 2, or 3 steps at a time. -/
def climbStairs (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | 1 => 1
  | 2 => 2
  | k + 3 => climbStairs k + climbStairs (k + 1) + climbStairs (k + 2)

/-- The number of steps in the staircase -/
def numSteps : ℕ := 10

/-- Theorem stating that there are 274 ways to climb a 10-step staircase -/
theorem staircase_climbing_ways : climbStairs numSteps = 274 := by
  sorry

end NUMINAMATH_CALUDE_staircase_climbing_ways_l413_41339


namespace NUMINAMATH_CALUDE_train_speed_l413_41355

/-- Calculates the speed of a train given its length, time to pass a person moving in the opposite direction, and the person's speed. -/
theorem train_speed (train_length : ℝ) (passing_time : ℝ) (person_speed : ℝ) : 
  train_length = 275 →
  passing_time = 15 →
  person_speed = 6 →
  (train_length / 1000) / (passing_time / 3600) - person_speed = 60 :=
by
  sorry

#check train_speed

end NUMINAMATH_CALUDE_train_speed_l413_41355


namespace NUMINAMATH_CALUDE_minimum_bottles_needed_l413_41390

def small_bottle_capacity : ℕ := 45
def large_bottle_capacity : ℕ := 600
def already_filled : ℕ := 90

theorem minimum_bottles_needed : 
  ∃ (n : ℕ), n * small_bottle_capacity + already_filled ≥ large_bottle_capacity ∧ 
  ∀ (m : ℕ), m * small_bottle_capacity + already_filled ≥ large_bottle_capacity → n ≤ m :=
by
  sorry

end NUMINAMATH_CALUDE_minimum_bottles_needed_l413_41390


namespace NUMINAMATH_CALUDE_shortest_altitude_right_triangle_l413_41305

/-- The shortest altitude of a triangle with sides 13, 84, and 85 --/
theorem shortest_altitude_right_triangle : 
  ∀ (a b c h : ℝ), 
    a = 13 → b = 84 → c = 85 →
    a^2 + b^2 = c^2 →
    h * c = 2 * (1/2 * a * b) →
    h = 1092 / 85 := by
  sorry

end NUMINAMATH_CALUDE_shortest_altitude_right_triangle_l413_41305


namespace NUMINAMATH_CALUDE_multiplication_equation_l413_41344

theorem multiplication_equation :
  ∀ (multiplier multiplicand product : ℕ),
    multiplier = 6 →
    multiplicand = product - 140 →
    multiplier * multiplicand = product →
    (multiplier = 6 ∧ multiplicand = 28 ∧ product = 168) :=
by
  sorry

end NUMINAMATH_CALUDE_multiplication_equation_l413_41344


namespace NUMINAMATH_CALUDE_manicure_cost_calculation_l413_41317

/-- The cost of a manicure in a nail salon. -/
def manicure_cost (total_revenue : ℚ) (total_fingers : ℕ) (fingers_per_person : ℕ) (non_clients : ℕ) : ℚ :=
  total_revenue / ((total_fingers / fingers_per_person) - non_clients)

/-- Theorem stating the cost of a manicure in the given scenario. -/
theorem manicure_cost_calculation :
  manicure_cost 200 210 10 11 = 952 / 100 := by sorry

end NUMINAMATH_CALUDE_manicure_cost_calculation_l413_41317


namespace NUMINAMATH_CALUDE_handshake_theorem_l413_41386

theorem handshake_theorem (n : ℕ) (h : n = 40) : 
  (n * (n - 1)) / 2 = 780 := by
  sorry

#check handshake_theorem

end NUMINAMATH_CALUDE_handshake_theorem_l413_41386


namespace NUMINAMATH_CALUDE_arccos_equation_solution_l413_41309

theorem arccos_equation_solution :
  ∀ x : ℝ, Real.arccos (3 * x) - Real.arccos x = π / 3 → x = -3 * Real.sqrt 21 / 28 := by
  sorry

end NUMINAMATH_CALUDE_arccos_equation_solution_l413_41309


namespace NUMINAMATH_CALUDE_missing_fraction_proof_l413_41397

theorem missing_fraction_proof (x : ℚ) : 
  1/2 + (-5/6) + 1/5 + 1/4 + (-9/20) + (-5/6) + x = 5/6 → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_missing_fraction_proof_l413_41397


namespace NUMINAMATH_CALUDE_complement_determines_interval_l413_41346

-- Define the set A
def A (a b : ℝ) : Set ℝ := { x | a ≤ x ∧ x ≤ b }

-- Define the complement of A
def C_U_A : Set ℝ := { x | x > 4 ∨ x < 3 }

-- Theorem statement
theorem complement_determines_interval :
  ∃ (a b : ℝ), A a b = (C_U_A)ᶜ ∧ a = 3 ∧ b = 4 := by
  sorry

end NUMINAMATH_CALUDE_complement_determines_interval_l413_41346


namespace NUMINAMATH_CALUDE_appropriate_units_l413_41365

-- Define the mass units
inductive MassUnit
| Kilogram
| Gram
| Ton

-- Define a structure for an object with a weight and unit
structure WeightedObject where
  weight : ℝ
  unit : MassUnit

-- Define the objects
def basketOfEggs : WeightedObject := { weight := 5, unit := MassUnit.Kilogram }
def honeybee : WeightedObject := { weight := 5, unit := MassUnit.Gram }
def tank : WeightedObject := { weight := 6, unit := MassUnit.Ton }

-- Function to determine if a unit is appropriate for an object
def isAppropriateUnit (obj : WeightedObject) : Prop :=
  match obj with
  | { weight := w, unit := MassUnit.Kilogram } => w ≥ 1 ∧ w < 1000
  | { weight := w, unit := MassUnit.Gram } => w ≥ 0.1 ∧ w < 1000
  | { weight := w, unit := MassUnit.Ton } => w ≥ 1 ∧ w < 1000

-- Theorem stating that the given units are appropriate for each object
theorem appropriate_units :
  isAppropriateUnit basketOfEggs ∧
  isAppropriateUnit honeybee ∧
  isAppropriateUnit tank := by
  sorry

end NUMINAMATH_CALUDE_appropriate_units_l413_41365


namespace NUMINAMATH_CALUDE_total_amount_distributed_l413_41304

-- Define the shares of A, B, and C
def share_A : ℕ := sorry
def share_B : ℕ := sorry
def share_C : ℕ := 495

-- Define the amounts to be decreased
def decrease_A : ℕ := 25
def decrease_B : ℕ := 10
def decrease_C : ℕ := 15

-- Define the ratio of remaining amounts
def ratio_A : ℕ := 3
def ratio_B : ℕ := 2
def ratio_C : ℕ := 5

-- Theorem to prove
theorem total_amount_distributed :
  share_A + share_B + share_C = 1010 :=
by
  sorry

-- Lemma to ensure the ratio condition is met
lemma ratio_condition :
  (share_A - decrease_A) * ratio_B * ratio_C = 
  (share_B - decrease_B) * ratio_A * ratio_C ∧
  (share_B - decrease_B) * ratio_A * ratio_C = 
  (share_C - decrease_C) * ratio_A * ratio_B :=
by
  sorry

end NUMINAMATH_CALUDE_total_amount_distributed_l413_41304


namespace NUMINAMATH_CALUDE_least_product_of_distinct_primes_above_50_l413_41353

theorem least_product_of_distinct_primes_above_50 :
  ∃ (p q : ℕ), 
    Prime p ∧ Prime q ∧ 
    p > 50 ∧ q > 50 ∧ 
    p ≠ q ∧
    p * q = 3127 ∧
    ∀ (r s : ℕ), Prime r → Prime s → r > 50 → s > 50 → r ≠ s → r * s ≥ p * q :=
by sorry

end NUMINAMATH_CALUDE_least_product_of_distinct_primes_above_50_l413_41353


namespace NUMINAMATH_CALUDE_fraction_sum_l413_41319

theorem fraction_sum : 2/5 + 4/50 + 3/500 + 8/5000 = 0.4876 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_l413_41319


namespace NUMINAMATH_CALUDE_intersection_of_tangents_l413_41352

/-- A curve defined by y = x + 1/x for x > 0 -/
def C : Set (ℝ × ℝ) := {p | p.2 = p.1 + 1 / p.1 ∧ p.1 > 0}

/-- A line passing through (0,1) with slope k -/
def line (k : ℝ) : Set (ℝ × ℝ) := {p | p.2 = k * p.1 + 1}

/-- The intersection points of the line with the curve C -/
def intersection_points (k : ℝ) : Set (ℝ × ℝ) := C ∩ line k

/-- The tangent line to C at a point (x, y) -/
def tangent_line (x : ℝ) : Set (ℝ × ℝ) := 
  {p | p.2 = (1 - 1/x^2) * p.1 + 2/x}

theorem intersection_of_tangents (k : ℝ) :
  ∀ M N : ℝ × ℝ, M ∈ intersection_points k → N ∈ intersection_points k → M ≠ N →
  ∃ P : ℝ × ℝ, P ∈ tangent_line M.1 ∧ P ∈ tangent_line N.1 ∧ 
  P.1 = 2 ∧ 2 < P.2 ∧ P.2 < 2.5 :=
sorry

end NUMINAMATH_CALUDE_intersection_of_tangents_l413_41352


namespace NUMINAMATH_CALUDE_total_nails_formula_nails_for_40_per_side_l413_41326

/-- The number of nails used to fix a square metal plate -/
def total_nails (nails_per_side : ℕ) : ℕ :=
  4 * nails_per_side - 4

/-- Theorem: The total number of nails used is equal to 4 times the number of nails on one side, minus 4 -/
theorem total_nails_formula (nails_per_side : ℕ) :
  total_nails nails_per_side = 4 * nails_per_side - 4 := by
  sorry

/-- Corollary: For a square with 40 nails on each side, the total number of nails used is 156 -/
theorem nails_for_40_per_side :
  total_nails 40 = 156 := by
  sorry

end NUMINAMATH_CALUDE_total_nails_formula_nails_for_40_per_side_l413_41326


namespace NUMINAMATH_CALUDE_angle_B_measure_side_b_value_l413_41361

-- Define a structure for a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  S : ℝ
  -- Add necessary conditions
  angle_sum : A + B + C = π
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c
  positive_angles : 0 < A ∧ 0 < B ∧ 0 < C

-- Part 1
theorem angle_B_measure (t : Triangle) (h : (Real.cos t.B) / (Real.cos t.C) = -t.b / (2 * t.a + t.c)) :
  t.B = 2 * π / 3 := by sorry

-- Part 2
theorem side_b_value (t : Triangle) (h1 : t.a = 4) (h2 : t.S = 5 * Real.sqrt 3) (h3 : t.B = 2 * π / 3) :
  t.b = Real.sqrt 61 := by sorry

end NUMINAMATH_CALUDE_angle_B_measure_side_b_value_l413_41361


namespace NUMINAMATH_CALUDE_crash_prob_equal_l413_41307

-- Define the probability of an engine failing
variable (p : ℝ)

-- Define the probability of crashing for the 3-engine plane
def crash_prob_3 (p : ℝ) : ℝ := 3 * p^2 * (1 - p) + p^3

-- Define the probability of crashing for the 5-engine plane
def crash_prob_5 (p : ℝ) : ℝ := 10 * p^3 * (1 - p)^2 + 5 * p^4 * (1 - p) + p^5

-- Theorem stating that the crash probabilities are equal for p = 0, 1/2, and 1
theorem crash_prob_equal : 
  (crash_prob_3 0 = crash_prob_5 0) ∧ 
  (crash_prob_3 (1/2) = crash_prob_5 (1/2)) ∧ 
  (crash_prob_3 1 = crash_prob_5 1) :=
sorry

end NUMINAMATH_CALUDE_crash_prob_equal_l413_41307


namespace NUMINAMATH_CALUDE_remainder_theorem_l413_41366

theorem remainder_theorem (n m p : ℤ) 
  (hn : n % 18 = 10)
  (hm : m % 27 = 16)
  (hp : p % 6 = 4) :
  (2*n + 3*m - p) % 9 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l413_41366


namespace NUMINAMATH_CALUDE_sum_of_x_values_l413_41368

theorem sum_of_x_values (x : ℝ) : 
  (|x - 25| = 50) → (∃ y : ℝ, |y - 25| = 50 ∧ x + y = 50) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_x_values_l413_41368


namespace NUMINAMATH_CALUDE_sum_of_combinations_l413_41354

theorem sum_of_combinations : Finset.sum (Finset.range 5) (fun k => Nat.choose 6 (k + 1)) = 62 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_combinations_l413_41354


namespace NUMINAMATH_CALUDE_expression_simplification_l413_41395

theorem expression_simplification (x : ℝ) : 
  ((((x + 2)^2 * (x^2 - 2*x + 2)^2) / (x^3 + 2)^2)^2 * 
   (((x - 2)^2 * (x^2 + 2*x + 2)^2) / (x^3 - 2)^2)^2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l413_41395


namespace NUMINAMATH_CALUDE_point_on_terminal_side_l413_41334

/-- Given a point P(m, m+1) on the terminal side of angle α where cos(α) = 3/5, prove that m = 3 -/
theorem point_on_terminal_side (m : ℝ) (α : ℝ) : 
  (∃ P : ℝ × ℝ, P = (m, m + 1) ∧ P.1 / Real.sqrt (P.1^2 + P.2^2) = 3/5) → 
  m = 3 := by
  sorry

end NUMINAMATH_CALUDE_point_on_terminal_side_l413_41334


namespace NUMINAMATH_CALUDE_quadratic_equations_solutions_l413_41393

theorem quadratic_equations_solutions :
  (∃ (s : Set ℝ), s = {x : ℝ | 2 * x^2 - x = 0} ∧ s = {0, 1/2}) ∧
  (∃ (t : Set ℝ), t = {x : ℝ | (2 * x + 1)^2 - 9 = 0} ∧ t = {1, -2}) := by
sorry

end NUMINAMATH_CALUDE_quadratic_equations_solutions_l413_41393


namespace NUMINAMATH_CALUDE_common_difference_unique_l413_41337

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum function
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_correct : ∀ n, S n = (n : ℝ) * (a 1 + a n) / 2

/-- The common difference of an arithmetic sequence -/
def common_difference (seq : ArithmeticSequence) : ℝ :=
  seq.a 2 - seq.a 1

theorem common_difference_unique
  (seq : ArithmeticSequence)
  (h1 : seq.a 3 = 3)
  (h2 : seq.S 4 = 14) :
  common_difference seq = -1 := by
  sorry

end NUMINAMATH_CALUDE_common_difference_unique_l413_41337


namespace NUMINAMATH_CALUDE_min_value_inequality_l413_41313

-- Define the function f
def f (x : ℝ) : ℝ := |x + 2| + |x - 1|

-- Theorem statement
theorem min_value_inequality (k a b c : ℝ) : 
  (∀ x, f x ≥ k) → -- k is the minimum value of f
  (a > 0 ∧ b > 0 ∧ c > 0) → -- a, b, c are positive
  (3 / (k * a) + 3 / (2 * k * b) + 1 / (k * c) = 1) → -- given equation
  a + 2 * b + 3 * c ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_min_value_inequality_l413_41313


namespace NUMINAMATH_CALUDE_class_distribution_l413_41363

theorem class_distribution (total : ℕ) (girls boys_carrots boys_apples : ℕ) : 
  total = 33 → 
  girls + boys_carrots + boys_apples = total →
  3 * boys_carrots + boys_apples = girls →
  boys_apples = girls →
  4 * boys_carrots = girls →
  girls = 15 ∧ boys_carrots = 6 ∧ boys_apples = 12 := by
  sorry

end NUMINAMATH_CALUDE_class_distribution_l413_41363


namespace NUMINAMATH_CALUDE_circumcircle_of_triangle_ABC_l413_41396

/-- The circumcircle of a triangle is the circle that passes through all three vertices of the triangle. -/
def is_circumcircle (a b c : ℝ × ℝ) (f : ℝ → ℝ → ℝ) : Prop :=
  f a.1 a.2 = 0 ∧ f b.1 b.2 = 0 ∧ f c.1 c.2 = 0

/-- The equation of a circle in general form is x^2 + y^2 + Dx + Ey + F = 0 -/
def circle_equation (D E F : ℝ) (x y : ℝ) : ℝ :=
  x^2 + y^2 + D*x + E*y + F

theorem circumcircle_of_triangle_ABC :
  let A : ℝ × ℝ := (-1, 5)
  let B : ℝ × ℝ := (5, 5)
  let C : ℝ × ℝ := (6, -2)
  let f (x y : ℝ) := circle_equation (-4) (-2) (-20) x y
  is_circumcircle A B C f :=
sorry

end NUMINAMATH_CALUDE_circumcircle_of_triangle_ABC_l413_41396


namespace NUMINAMATH_CALUDE_train_bridge_crossing_time_l413_41345

/-- Calculates the time taken for a train to cross a bridge -/
theorem train_bridge_crossing_time 
  (bridge_length : ℝ) 
  (train_length : ℝ) 
  (train_speed : ℝ) 
  (h1 : bridge_length = 200)
  (h2 : train_length = 100)
  (h3 : train_speed = 5) : 
  (bridge_length + train_length) / train_speed = 60 := by
  sorry

end NUMINAMATH_CALUDE_train_bridge_crossing_time_l413_41345


namespace NUMINAMATH_CALUDE_binomial_expansion_coefficient_sum_l413_41372

/-- The sum of the coefficients of the last three terms in the binomial expansion -/
def sum_of_last_three_coefficients (n : ℕ) : ℕ := 1 + n + n * (n - 1) / 2

/-- The theorem stating that if the sum of the coefficients of the last three terms 
    in the expansion of (√x + 2/√x)^n is 79, then n = 12 -/
theorem binomial_expansion_coefficient_sum (n : ℕ) : 
  sum_of_last_three_coefficients n = 79 → n = 12 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_coefficient_sum_l413_41372


namespace NUMINAMATH_CALUDE_ada_original_seat_l413_41371

/-- Represents the seats in the theater --/
inductive Seat
| one
| two
| three
| four
| five
| six

/-- Represents the friends --/
inductive Friend
| ada
| bea
| ceci
| dee
| edie
| fi

/-- Represents the direction of movement --/
inductive Direction
| left
| right

/-- Defines a movement of a friend --/
structure Movement where
  friend : Friend
  distance : Nat
  direction : Direction

/-- Defines the seating arrangement --/
def SeatingArrangement := Friend → Seat

/-- Defines the set of movements --/
def Movements := List Movement

/-- Function to apply a movement to a seating arrangement --/
def applyMovement (arrangement : SeatingArrangement) (move : Movement) : SeatingArrangement :=
  sorry

/-- Function to apply all movements to a seating arrangement --/
def applyMovements (arrangement : SeatingArrangement) (moves : Movements) : SeatingArrangement :=
  sorry

/-- Theorem stating Ada's original seat --/
theorem ada_original_seat 
  (initial_arrangement : SeatingArrangement)
  (moves : Movements)
  (final_arrangement : SeatingArrangement) :
  (moves = [
    ⟨Friend.bea, 3, Direction.right⟩,
    ⟨Friend.ceci, 1, Direction.left⟩,
    ⟨Friend.dee, 1, Direction.right⟩,
    ⟨Friend.edie, 1, Direction.left⟩
  ]) →
  (final_arrangement = applyMovements initial_arrangement moves) →
  (final_arrangement Friend.ada = Seat.one ∨ final_arrangement Friend.ada = Seat.six) →
  (initial_arrangement Friend.ada = Seat.three) :=
sorry

end NUMINAMATH_CALUDE_ada_original_seat_l413_41371


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l413_41321

theorem partial_fraction_decomposition :
  ∃! (P Q R : ℚ),
    (∀ x : ℚ, x ≠ 2 ∧ x ≠ 4 →
      (3 * x + 1) / ((x - 4) * (x - 2)^2) =
      P / (x - 4) + Q / (x - 2) + R / (x - 2)^2) ∧
    P = 13/4 ∧ Q = -13/4 ∧ R = -7/2 := by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l413_41321


namespace NUMINAMATH_CALUDE_cake_recipe_difference_l413_41338

theorem cake_recipe_difference (total_flour total_sugar flour_added : ℕ) :
  total_flour = 9 →
  total_sugar = 6 →
  flour_added = 2 →
  (total_flour - flour_added) - total_sugar = 1 := by
sorry

end NUMINAMATH_CALUDE_cake_recipe_difference_l413_41338


namespace NUMINAMATH_CALUDE_air_quality_probability_l413_41378

theorem air_quality_probability (p_good : ℝ) (p_consecutive : ℝ) 
  (h1 : p_good = 0.8) 
  (h2 : p_consecutive = 0.68) : 
  p_consecutive / p_good = 0.85 := by
  sorry

end NUMINAMATH_CALUDE_air_quality_probability_l413_41378


namespace NUMINAMATH_CALUDE_star_value_l413_41349

-- Define the * operation
def star (a b : ℝ) (x y : ℝ) : ℝ := a * x + b * y + 2010

-- State the theorem
theorem star_value (a b : ℝ) :
  (star a b 3 5 = 2011) → (star a b 4 9 = 2009) → (star a b 1 2 = 2010) := by
  sorry

end NUMINAMATH_CALUDE_star_value_l413_41349


namespace NUMINAMATH_CALUDE_floor_x_length_l413_41391

/-- Represents the dimensions of a rectangular floor -/
structure Floor where
  width : ℝ
  length : ℝ

/-- Calculates the area of a rectangular floor -/
def area (f : Floor) : ℝ := f.width * f.length

theorem floor_x_length
  (x y : Floor)
  (h1 : area x = area y)
  (h2 : x.width = 10)
  (h3 : y.width = 9)
  (h4 : y.length = 20) :
  x.length = 18 := by
  sorry

end NUMINAMATH_CALUDE_floor_x_length_l413_41391


namespace NUMINAMATH_CALUDE_sum_of_integers_l413_41392

theorem sum_of_integers (a b c d : ℤ) 
  (eq1 : a - b + c = 7)
  (eq2 : b - c + d = 8)
  (eq3 : c - d + a = 5)
  (eq4 : d - a + b = 4) :
  a + b + c + d = 12 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_integers_l413_41392


namespace NUMINAMATH_CALUDE_ezekiel_painted_faces_l413_41348

/-- The number of faces on a cuboid -/
def faces_per_cuboid : ℕ := 6

/-- The number of cuboids painted -/
def num_cuboids : ℕ := 5

/-- The total number of faces painted by Ezekiel -/
def total_faces_painted : ℕ := faces_per_cuboid * num_cuboids

theorem ezekiel_painted_faces :
  total_faces_painted = 30 :=
by sorry

end NUMINAMATH_CALUDE_ezekiel_painted_faces_l413_41348


namespace NUMINAMATH_CALUDE_range_of_a_l413_41341

/-- The range of a given the conditions in the problem -/
theorem range_of_a (p q : Prop) (a : ℝ) 
  (hp : p ↔ ∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0)
  (hq : q ↔ ∃ x : ℝ, x^2 + (a - 1)*x + 1 < 0)
  (hpq_or : p ∨ q)
  (hpq_not_and : ¬(p ∧ q)) :
  a ∈ Set.Icc (-1) 1 ∪ Set.Ioi 3 := by
  sorry


end NUMINAMATH_CALUDE_range_of_a_l413_41341


namespace NUMINAMATH_CALUDE_time_sum_after_duration_l413_41331

/-- Represents time in hours, minutes, and seconds -/
structure Time where
  hours : Nat
  minutes : Nat
  seconds : Nat

/-- Adds a duration to a given time and returns the resulting time -/
def addDuration (initial : Time) (durationHours durationMinutes durationSeconds : Nat) : Time :=
  sorry

/-- Converts a Time to its 12-hour clock representation -/
def to12HourClock (t : Time) : Time :=
  sorry

/-- Calculates the sum of digits in a Time -/
def sumDigits (t : Time) : Nat :=
  sorry

theorem time_sum_after_duration :
  let initialTime : Time := ⟨15, 0, 0⟩  -- 3:00:00 PM
  let finalTime := to12HourClock (addDuration initialTime 317 58 33)
  sumDigits finalTime = 99 := by
  sorry

end NUMINAMATH_CALUDE_time_sum_after_duration_l413_41331


namespace NUMINAMATH_CALUDE_sqrt_meaningful_iff_geq_two_l413_41332

theorem sqrt_meaningful_iff_geq_two (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = x - 2) ↔ x ≥ 2 := by sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_iff_geq_two_l413_41332


namespace NUMINAMATH_CALUDE_cone_radius_from_melted_cylinder_l413_41311

/-- The radius of a cone formed by melting a cylinder -/
theorem cone_radius_from_melted_cylinder (r_cylinder h_cylinder h_cone : ℝ) 
  (h_r : r_cylinder = 8)
  (h_h_cylinder : h_cylinder = 2)
  (h_h_cone : h_cone = 6) : 
  ∃ (r_cone : ℝ), r_cone = 8 ∧ 
  (π * r_cylinder^2 * h_cylinder = (1/3) * π * r_cone^2 * h_cone) :=
by
  sorry

#check cone_radius_from_melted_cylinder

end NUMINAMATH_CALUDE_cone_radius_from_melted_cylinder_l413_41311


namespace NUMINAMATH_CALUDE_angle5_is_36_degrees_l413_41388

-- Define the angles
variable (angle1 angle2 angle5 : ℝ)

-- Define the conditions
axiom parallel_lines : True  -- m ∥ n
axiom angle1_is_quarter_angle2 : angle1 = (1 / 4) * angle2
axiom alternate_interior_angles : angle5 = angle1
axiom straight_line : angle2 + angle5 = 180

-- Theorem to prove
theorem angle5_is_36_degrees : angle5 = 36 := by
  sorry

end NUMINAMATH_CALUDE_angle5_is_36_degrees_l413_41388


namespace NUMINAMATH_CALUDE_adult_ticket_cost_l413_41320

theorem adult_ticket_cost 
  (total_seats : ℕ) 
  (child_ticket_cost : ℚ) 
  (num_children : ℕ) 
  (total_revenue : ℚ) 
  (h1 : total_seats = 250) 
  (h2 : child_ticket_cost = 4) 
  (h3 : num_children = 188) 
  (h4 : total_revenue = 1124) :
  let num_adults : ℕ := total_seats - num_children
  let adult_ticket_cost : ℚ := (total_revenue - (↑num_children * child_ticket_cost)) / ↑num_adults
  adult_ticket_cost = 6 := by
sorry

end NUMINAMATH_CALUDE_adult_ticket_cost_l413_41320


namespace NUMINAMATH_CALUDE_family_savings_calculation_l413_41323

def tax_rate : Float := 0.13

def ivan_salary : Float := 55000
def vasilisa_salary : Float := 45000
def mother_salary : Float := 18000
def father_salary : Float := 20000
def son_scholarship : Float := 3000
def mother_pension : Float := 10000
def son_extra_scholarship : Float := 15000

def monthly_expenses : Float := 74000

def net_income (gross_income : Float) : Float :=
  gross_income * (1 - tax_rate)

def total_income_before_may2018 : Float :=
  net_income ivan_salary + net_income vasilisa_salary + 
  net_income mother_salary + net_income father_salary + son_scholarship

def total_income_may_to_aug2018 : Float :=
  net_income ivan_salary + net_income vasilisa_salary + 
  mother_pension + net_income father_salary + son_scholarship

def total_income_from_sept2018 : Float :=
  net_income ivan_salary + net_income vasilisa_salary + 
  mother_pension + net_income father_salary + 
  son_scholarship + net_income son_extra_scholarship

theorem family_savings_calculation :
  (total_income_before_may2018 - monthly_expenses = 49060) ∧
  (total_income_may_to_aug2018 - monthly_expenses = 43400) ∧
  (total_income_from_sept2018 - monthly_expenses = 56450) := by
  sorry

end NUMINAMATH_CALUDE_family_savings_calculation_l413_41323


namespace NUMINAMATH_CALUDE_orange_packing_l413_41336

theorem orange_packing (total_oranges : ℕ) (oranges_per_box : ℕ) (h1 : total_oranges = 94) (h2 : oranges_per_box = 8) :
  (total_oranges + oranges_per_box - 1) / oranges_per_box = 12 := by
  sorry

end NUMINAMATH_CALUDE_orange_packing_l413_41336


namespace NUMINAMATH_CALUDE_cosine_value_in_triangle_l413_41394

/-- In a triangle ABC, if b cos C = (3a - c) cos B, then cos B = 1/3 -/
theorem cosine_value_in_triangle (a b c : ℝ) (A B C : ℝ) :
  (0 < a) → (0 < b) → (0 < c) →
  (0 < A) → (A < π) →
  (0 < B) → (B < π) →
  (0 < C) → (C < π) →
  (A + B + C = π) →
  (b * Real.cos C = (3 * a - c) * Real.cos B) →
  Real.cos B = 1 / 3 := by
sorry


end NUMINAMATH_CALUDE_cosine_value_in_triangle_l413_41394


namespace NUMINAMATH_CALUDE_min_value_sum_squares_l413_41383

theorem min_value_sum_squares (x y z a : ℝ) (h : x + 2*y + 3*z = a) :
  x^2 + y^2 + z^2 ≥ a^2 / 14 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_squares_l413_41383


namespace NUMINAMATH_CALUDE_circle_equation_from_diameter_l413_41379

/-- Given two points as the endpoints of a circle's diameter, prove the equation of the circle -/
theorem circle_equation_from_diameter (p1 p2 : ℝ × ℝ) :
  p1 = (-1, 3) →
  p2 = (5, -5) →
  ∃ (a b c : ℝ), ∀ (x y : ℝ),
    (x^2 + y^2 + a*x + b*y + c = 0) ↔
    ((x - ((p1.1 + p2.1) / 2))^2 + (y - ((p1.2 + p2.2) / 2))^2 = ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2) / 4) :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_from_diameter_l413_41379


namespace NUMINAMATH_CALUDE_complement_of_intersection_l413_41300

def U : Set ℕ := {1, 2, 3, 4}
def M : Set ℕ := {1, 2, 3}
def N : Set ℕ := {2, 3, 4}

theorem complement_of_intersection (U M N : Set ℕ) 
  (hU : U = {1, 2, 3, 4})
  (hM : M = {1, 2, 3})
  (hN : N = {2, 3, 4}) :
  (U \ (M ∩ N)) = {1, 4} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_intersection_l413_41300


namespace NUMINAMATH_CALUDE_max_m_value_l413_41381

def f (x : ℝ) : ℝ := |x + 1| + |1 - 2*x|

theorem max_m_value (a b : ℝ) (h1 : 0 < b) (h2 : b < 1/2) (h3 : 1/2 < a) 
  (h4 : f a = 3 * f b) :
  ∃ m : ℤ, (∀ n : ℤ, a^2 + b^2 > ↑n → n ≤ m) ∧ a^2 + b^2 > ↑m :=
by sorry

end NUMINAMATH_CALUDE_max_m_value_l413_41381


namespace NUMINAMATH_CALUDE_original_group_size_l413_41364

theorem original_group_size 
  (total_work : ℝ) 
  (original_days : ℕ) 
  (remaining_days : ℕ) 
  (absent_men : ℕ) :
  let original_work_rate := total_work / original_days
  let remaining_work_rate := total_work / remaining_days
  original_days = 10 ∧ 
  remaining_days = 12 ∧ 
  absent_men = 5 →
  ∃ (original_size : ℕ),
    original_size * original_work_rate = (original_size - absent_men) * remaining_work_rate ∧
    original_size = 25 :=
by sorry

end NUMINAMATH_CALUDE_original_group_size_l413_41364


namespace NUMINAMATH_CALUDE_interest_problem_l413_41367

theorem interest_problem (P : ℝ) : 
  (P * 0.04 + P * 0.06 + P * 0.08 = 2700) → P = 15000 := by
  sorry

end NUMINAMATH_CALUDE_interest_problem_l413_41367


namespace NUMINAMATH_CALUDE_max_choir_members_choir_of_120_exists_l413_41302

/-- Represents a choir formation --/
structure ChoirFormation where
  rows : ℕ
  members_per_row : ℕ

/-- Represents the choir and its formations --/
structure Choir where
  total_members : ℕ
  original_formation : ChoirFormation
  new_formation : ChoirFormation

/-- The conditions of the choir problem --/
def choir_conditions (c : Choir) : Prop :=
  c.total_members < 120 ∧
  c.total_members = c.original_formation.rows * c.original_formation.members_per_row + 3 ∧
  c.total_members = (c.original_formation.rows - 1) * (c.original_formation.members_per_row + 2)

/-- The theorem stating the maximum number of choir members --/
theorem max_choir_members :
  ∀ c : Choir, choir_conditions c → c.total_members ≤ 120 :=
by sorry

/-- The theorem stating that 120 is achievable --/
theorem choir_of_120_exists :
  ∃ c : Choir, choir_conditions c ∧ c.total_members = 120 :=
by sorry

end NUMINAMATH_CALUDE_max_choir_members_choir_of_120_exists_l413_41302


namespace NUMINAMATH_CALUDE_production_average_l413_41328

theorem production_average (n : ℕ) : 
  (∀ (past_total : ℕ), past_total = n * 50 →
   ∀ (new_total : ℕ), new_total = past_total + 90 →
   (new_total : ℚ) / (n + 1 : ℚ) = 52) →
  n = 19 := by
sorry

end NUMINAMATH_CALUDE_production_average_l413_41328


namespace NUMINAMATH_CALUDE_square_inequality_l413_41340

theorem square_inequality (x y : ℝ) (h1 : x > y) (h2 : y > 3 / (x - y)) : x^2 > y^2 + 6 := by
  sorry

end NUMINAMATH_CALUDE_square_inequality_l413_41340
