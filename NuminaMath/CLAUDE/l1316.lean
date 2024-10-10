import Mathlib

namespace hill_climbing_speed_l1316_131666

/-- Proves that given a round trip journey with specified conditions, 
    the average speed for the upward journey is 2.625 km/h -/
theorem hill_climbing_speed 
  (upward_time : ℝ) 
  (downward_time : ℝ) 
  (total_avg_speed : ℝ) 
  (h1 : upward_time = 4) 
  (h2 : downward_time = 2) 
  (h3 : total_avg_speed = 3.5) : 
  (total_avg_speed * (upward_time + downward_time)) / (2 * upward_time) = 2.625 := by
sorry

end hill_climbing_speed_l1316_131666


namespace sum_congruence_l1316_131626

theorem sum_congruence : (1 + 22 + 333 + 4444 + 55555 + 666666 + 7777777 + 88888888) % 9 = 6 := by
  sorry

end sum_congruence_l1316_131626


namespace smallest_b_for_quadratic_inequality_l1316_131644

theorem smallest_b_for_quadratic_inequality :
  ∀ b : ℝ, b^2 - 16*b + 63 ≤ 0 → b ≥ 7 :=
by sorry

end smallest_b_for_quadratic_inequality_l1316_131644


namespace min_bullseyes_for_victory_l1316_131604

/-- Represents the archery competition scenario -/
structure ArcheryCompetition where
  total_shots : ℕ := 120
  halfway_point : ℕ := 60
  alex_lead_at_half : ℕ := 60
  bullseye_score : ℕ := 10
  alex_min_score : ℕ := 3

/-- Theorem stating the minimum number of bullseyes Alex needs to guarantee victory -/
theorem min_bullseyes_for_victory (comp : ArcheryCompetition) :
  ∃ n : ℕ, n = 52 ∧
  (∀ m : ℕ, -- m represents Alex's current score
    (comp.alex_lead_at_half + m = comp.halfway_point * comp.alex_min_score) →
    (m + n * comp.bullseye_score + (comp.halfway_point - n) * comp.alex_min_score >
     m - comp.alex_lead_at_half + comp.halfway_point * comp.bullseye_score) ∧
    (∀ k : ℕ, k < n →
      ∃ p : ℕ, p ≤ m - comp.alex_lead_at_half + comp.halfway_point * comp.bullseye_score ∧
      p ≥ m + k * comp.bullseye_score + (comp.halfway_point - k) * comp.alex_min_score)) :=
sorry

end min_bullseyes_for_victory_l1316_131604


namespace sum_of_roots_quadratic_sum_of_roots_specific_equation_l1316_131615

theorem sum_of_roots_quadratic (a b c : ℝ) (h : a ≠ 0) :
  let f : ℝ → ℝ := λ x => a * x^2 + b * x + c
  (∃ x₁ x₂, f x₁ = 0 ∧ f x₂ = 0 ∧ x₁ ≠ x₂) →
  (∃ s, s = x₁ + x₂ ∧ s = -b / a) :=
sorry

theorem sum_of_roots_specific_equation :
  let f : ℝ → ℝ := λ x => x^2 + 2010*x - (2011 + 18*x)
  (∃ x₁ x₂, f x₁ = 0 ∧ f x₂ = 0 ∧ x₁ ≠ x₂) →
  (∃ s, s = x₁ + x₂ ∧ s = -1992) :=
sorry

end sum_of_roots_quadratic_sum_of_roots_specific_equation_l1316_131615


namespace relationship_abc_l1316_131634

theorem relationship_abc :
  let a : ℤ := -2 * 3^2
  let b : ℤ := (-2 * 3)^2
  let c : ℤ := -(2 * 3)^2
  b > a ∧ a > c := by sorry

end relationship_abc_l1316_131634


namespace journey_speed_proof_l1316_131663

theorem journey_speed_proof (total_distance : ℝ) (total_time : ℝ) (second_half_speed : ℝ) :
  total_distance = 672 ∧ total_time = 30 ∧ second_half_speed = 24 →
  ∃ first_half_speed : ℝ,
    first_half_speed = 21 ∧
    first_half_speed * (total_time / 2) + second_half_speed * (total_time / 2) = total_distance :=
by
  sorry

end journey_speed_proof_l1316_131663


namespace nontrivial_solution_iff_l1316_131658

/-- A system of linear equations with coefficients a, b, c has a non-trivial solution -/
def has_nontrivial_solution (a b c : ℝ) : Prop :=
  ∃ x y z : ℝ, (x ≠ 0 ∨ y ≠ 0 ∨ z ≠ 0) ∧
    a * x + b * y + c * z = 0 ∧
    b * x + c * y + a * z = 0 ∧
    c * x + a * y + b * z = 0

/-- The main theorem characterizing when the system has a non-trivial solution -/
theorem nontrivial_solution_iff (a b c : ℝ) :
  has_nontrivial_solution a b c ↔ a + b + c = 0 ∨ (a = b ∧ b = c) :=
sorry

end nontrivial_solution_iff_l1316_131658


namespace randys_house_blocks_l1316_131639

/-- Given Randy's block building scenario, prove the number of blocks used for the house. -/
theorem randys_house_blocks (total : ℕ) (tower : ℕ) (difference : ℕ) (house : ℕ) : 
  total = 90 → tower = 63 → difference = 26 → house = tower + difference → house = 89 := by
  sorry

end randys_house_blocks_l1316_131639


namespace inequality_proof_l1316_131669

theorem inequality_proof (x y : ℝ) (n : ℕ+) 
  (hx : 0 < x ∧ x < 1) (hy : 0 < y ∧ y < 1) :
  (x^n.val / (1 - x^2) + y^n.val / (1 - y^2)) ≥ ((x^n.val + y^n.val) / (1 - x*y)) := by
  sorry

end inequality_proof_l1316_131669


namespace find_B_l1316_131679

/-- The number represented by 2B8, where B is a single digit -/
def number (B : ℕ) : ℕ := 200 + 10 * B + 8

/-- The sum of digits of the number 2B8 -/
def digit_sum (B : ℕ) : ℕ := 2 + B + 8

theorem find_B : ∃ B : ℕ, B < 10 ∧ number B % 3 = 0 ∧ B = 2 :=
sorry

end find_B_l1316_131679


namespace prop_q_not_necessary_nor_sufficient_l1316_131620

/-- Proposition P: The solution sets of two quadratic inequalities are the same -/
def PropP (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ) : Prop :=
  {x : ℝ | a₁ * x^2 + b₁ * x + c₁ > 0} = {x : ℝ | a₂ * x^2 + b₂ * x + c₂ > 0}

/-- Proposition Q: The coefficients of two quadratic expressions are proportional -/
def PropQ (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ a₁ = k * a₂ ∧ b₁ = k * b₂ ∧ c₁ = k * c₂

theorem prop_q_not_necessary_nor_sufficient :
  ¬(∀ a₁ b₁ c₁ a₂ b₂ c₂ : ℝ, PropQ a₁ b₁ c₁ a₂ b₂ c₂ → PropP a₁ b₁ c₁ a₂ b₂ c₂) ∧
  ¬(∀ a₁ b₁ c₁ a₂ b₂ c₂ : ℝ, PropP a₁ b₁ c₁ a₂ b₂ c₂ → PropQ a₁ b₁ c₁ a₂ b₂ c₂) :=
sorry

end prop_q_not_necessary_nor_sufficient_l1316_131620


namespace special_function_increasing_l1316_131660

/-- A function satisfying the given properties -/
class SpecialFunction (f : ℝ → ℝ) : Prop where
  pos_gt_one : ∀ x > 0, f x > 1
  multiplicative : ∀ x y, f (x + y) = f x * f y

/-- Theorem: f is increasing on ℝ -/
theorem special_function_increasing (f : ℝ → ℝ) [SpecialFunction f] :
  ∀ x₁ x₂, x₁ < x₂ → f x₁ < f x₂ := by
  sorry

end special_function_increasing_l1316_131660


namespace shopping_tax_calculation_l1316_131668

/-- Calculates the final tax percentage given spending percentages and tax rates --/
def final_tax_percentage (clothing_percent : ℝ) (food_percent : ℝ) (electronics_percent : ℝ) 
  (other_percent : ℝ) (clothing_tax : ℝ) (food_tax : ℝ) (electronics_tax : ℝ) 
  (other_tax : ℝ) (loyalty_discount : ℝ) : ℝ :=
  let total_tax := clothing_percent * clothing_tax + food_percent * food_tax + 
                   electronics_percent * electronics_tax + other_percent * other_tax
  let discounted_tax := total_tax * (1 - loyalty_discount)
  discounted_tax * 100

theorem shopping_tax_calculation :
  final_tax_percentage 0.4 0.25 0.2 0.15 0.05 0.02 0.1 0.08 0.03 = 5.529 := by
  sorry

end shopping_tax_calculation_l1316_131668


namespace square_of_difference_product_of_three_terms_l1316_131654

-- Problem 1
theorem square_of_difference (a b : ℝ) : (a^2 - b)^2 = a^4 - 2*a^2*b + b^2 := by sorry

-- Problem 2
theorem product_of_three_terms (x : ℝ) : (2*x + 1)*(4*x^2 - 1)*(2*x - 1) = 16*x^4 - 8*x^2 + 1 := by sorry

end square_of_difference_product_of_three_terms_l1316_131654


namespace sidney_adult_cats_l1316_131657

/-- Represents the number of adult cats Sidney has -/
def num_adult_cats : ℕ := sorry

/-- The number of kittens Sidney has -/
def num_kittens : ℕ := 4

/-- The number of cans of cat food Sidney has -/
def initial_cans : ℕ := 7

/-- The number of cans an adult cat eats per day -/
def adult_cat_consumption : ℚ := 1

/-- The number of cans a kitten eats per day -/
def kitten_consumption : ℚ := 3/4

/-- The number of additional cans Sidney needs to buy -/
def additional_cans : ℕ := 35

/-- The number of days Sidney needs to feed her cats -/
def days : ℕ := 7

theorem sidney_adult_cats : 
  num_adult_cats = 3 ∧
  (num_kittens : ℚ) * kitten_consumption * days + 
  (num_adult_cats : ℚ) * adult_cat_consumption * days = 
  (initial_cans : ℚ) + additional_cans :=
sorry

end sidney_adult_cats_l1316_131657


namespace rhombus_perimeter_l1316_131600

theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 10) (h2 : d2 = 12) :
  let side := Real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2)
  4 * side = 4 * Real.sqrt 61 := by
  sorry

end rhombus_perimeter_l1316_131600


namespace fraction_irreducible_l1316_131670

theorem fraction_irreducible (n : ℕ) : Nat.gcd (21 * n + 4) (14 * n + 3) = 1 := by
  sorry

end fraction_irreducible_l1316_131670


namespace randy_piggy_bank_theorem_l1316_131687

/-- Calculates the amount in Randy's piggy bank after a year -/
def piggy_bank_after_year (initial_amount : ℕ) (store_trip_cost : ℕ) (store_trips_per_month : ℕ)
  (internet_bill : ℕ) (extra_cost_third_trip : ℕ) (weekly_earnings : ℕ) (birthday_gift : ℕ) : ℕ :=
  let months_in_year : ℕ := 12
  let weeks_in_year : ℕ := 52
  let regular_store_expenses := store_trip_cost * store_trips_per_month * months_in_year
  let extra_expenses := extra_cost_third_trip * (months_in_year / 3)
  let internet_expenses := internet_bill * months_in_year
  let job_income := weekly_earnings * weeks_in_year
  let total_expenses := regular_store_expenses + extra_expenses + internet_expenses
  let total_income := job_income + birthday_gift
  initial_amount + total_income - total_expenses

theorem randy_piggy_bank_theorem :
  piggy_bank_after_year 200 2 4 20 1 15 100 = 740 := by
  sorry

end randy_piggy_bank_theorem_l1316_131687


namespace min_value_expression_l1316_131672

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 1) :
  a^2 + 8*a*b + 32*b^2 + 24*b*c + 8*c^2 ≥ 36 ∧
  ∃ (a₀ b₀ c₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ c₀ > 0 ∧ a₀ * b₀ * c₀ = 1 ∧
    a₀^2 + 8*a₀*b₀ + 32*b₀^2 + 24*b₀*c₀ + 8*c₀^2 = 36 :=
by sorry

end min_value_expression_l1316_131672


namespace twenty_students_no_math_l1316_131621

/-- Represents a class of students with information about their subject choices. -/
structure ClassInfo where
  total : ℕ
  no_science : ℕ
  no_either : ℕ
  both : ℕ

/-- Calculates the number of students who didn't opt for math. -/
def students_no_math (info : ClassInfo) : ℕ :=
  info.total - info.both - (info.no_science - info.no_either)

/-- Theorem stating that in a specific class, 20 students didn't opt for math. -/
theorem twenty_students_no_math :
  let info : ClassInfo := {
    total := 40,
    no_science := 15,
    no_either := 2,
    both := 7
  }
  students_no_math info = 20 := by
  sorry


end twenty_students_no_math_l1316_131621


namespace farm_animal_ratio_l1316_131685

theorem farm_animal_ratio (cows sheep pigs : ℕ) : 
  cows = 12 →
  sheep = 2 * cows →
  cows + sheep + pigs = 108 →
  pigs / sheep = 3 := by
  sorry

end farm_animal_ratio_l1316_131685


namespace shaded_volume_is_112_l1316_131683

/-- The volume of a rectangular prism with dimensions a, b, and c -/
def volume (a b c : ℕ) : ℕ := a * b * c

/-- The dimensions of the larger prism -/
def large_prism : Fin 3 → ℕ
| 0 => 4
| 1 => 5
| 2 => 6
| _ => 0

/-- The dimensions of the smaller prism -/
def small_prism : Fin 3 → ℕ
| 0 => 1
| 1 => 2
| 2 => 4
| _ => 0

theorem shaded_volume_is_112 :
  volume (large_prism 0) (large_prism 1) (large_prism 2) -
  volume (small_prism 0) (small_prism 1) (small_prism 2) = 112 := by
  sorry

end shaded_volume_is_112_l1316_131683


namespace quadratic_function_property_l1316_131676

theorem quadratic_function_property (a b c : ℝ) :
  let f : ℝ → ℝ := λ x ↦ a * x^2 + b * x + c
  (f 0 = f 4 ∧ f 0 > f 1) → (a > 0 ∧ 4 * a + b = 0) := by
  sorry

end quadratic_function_property_l1316_131676


namespace crayon_purchase_l1316_131612

def half_dozen : ℕ := 6

theorem crayon_purchase (total_cost : ℕ) (cost_per_crayon : ℕ) (half_dozens : ℕ) : 
  total_cost = 48 ∧ 
  cost_per_crayon = 2 ∧ 
  total_cost = half_dozens * half_dozen * cost_per_crayon →
  half_dozens = 4 := by
sorry

end crayon_purchase_l1316_131612


namespace length_AE_l1316_131607

/-- Represents a point on a line -/
structure Point where
  x : ℝ

/-- Calculates the distance between two points -/
def distance (p q : Point) : ℝ := abs (p.x - q.x)

/-- Theorem: Length of AE given specific conditions -/
theorem length_AE (a b c d e : Point) 
  (consecutive : a.x < b.x ∧ b.x < c.x ∧ c.x < d.x ∧ d.x < e.x)
  (bc_eq_3cd : distance b c = 3 * distance c d)
  (de_eq_7 : distance d e = 7)
  (ab_eq_5 : distance a b = 5)
  (ac_eq_11 : distance a c = 11) :
  distance a e = 18 := by
  sorry

end length_AE_l1316_131607


namespace roots_distinct_and_sum_integer_l1316_131628

/-- Given that a, b, c are roots of x^3 - x^2 - x - 1 = 0, prove they are distinct and
    that (a^1982 - b^1982)/(a - b) + (b^1982 - c^1982)/(b - c) + (c^1982 - a^1982)/(c - a) is an integer -/
theorem roots_distinct_and_sum_integer (a b c : ℂ) : 
  (a^3 - a^2 - a - 1 = 0) → 
  (b^3 - b^2 - b - 1 = 0) → 
  (c^3 - c^2 - c - 1 = 0) → 
  (a ≠ b ∧ b ≠ c ∧ c ≠ a) ∧ 
  (∃ k : ℤ, (a^1982 - b^1982)/(a - b) + (b^1982 - c^1982)/(b - c) + (c^1982 - a^1982)/(c - a) = k) := by
  sorry

end roots_distinct_and_sum_integer_l1316_131628


namespace detect_non_conforming_probability_l1316_131619

/-- The number of cans in a box -/
def total_cans : ℕ := 5

/-- The number of non-conforming cans in the box -/
def non_conforming_cans : ℕ := 2

/-- The number of cans selected for testing -/
def selected_cans : ℕ := 2

/-- The probability of detecting at least one non-conforming product -/
def probability_detect : ℚ := 7 / 10

theorem detect_non_conforming_probability :
  probability_detect = (Nat.choose non_conforming_cans 1 * Nat.choose (total_cans - non_conforming_cans) 1 + 
                        Nat.choose non_conforming_cans 2) / 
                       Nat.choose total_cans selected_cans :=
by sorry

end detect_non_conforming_probability_l1316_131619


namespace cube_root_condition_square_root_condition_integer_part_condition_main_result_l1316_131694

def a : ℝ := 4
def b : ℝ := 2
def c : ℤ := 3

theorem cube_root_condition : (3 * a - 4) ^ (1/3 : ℝ) = 2 := by sorry

theorem square_root_condition : (a + 2 * b + 1) ^ (1/2 : ℝ) = 3 := by sorry

theorem integer_part_condition : c = Int.floor (Real.sqrt 15) := by sorry

theorem main_result : (a + b + c : ℝ) ^ (1/2 : ℝ) = 3 ∨ (a + b + c : ℝ) ^ (1/2 : ℝ) = -3 := by sorry

end cube_root_condition_square_root_condition_integer_part_condition_main_result_l1316_131694


namespace complex_fraction_imaginary_l1316_131637

theorem complex_fraction_imaginary (a : ℝ) : 
  (∃ (k : ℝ), (2 + I) / (a - I) = k * I) → a = 1/2 := by
  sorry

end complex_fraction_imaginary_l1316_131637


namespace expression_evaluation_l1316_131602

theorem expression_evaluation :
  let x : ℕ := 3
  let y : ℕ := 4
  5 * x^(y+1) + 6 * y^(x+1) = 2751 := by
sorry

end expression_evaluation_l1316_131602


namespace correct_quotient_proof_l1316_131611

theorem correct_quotient_proof (N : ℕ) : 
  N % 21 = 0 →  -- remainder is 0 when divided by 21
  N / 12 = 56 → -- quotient is 56 when divided by 12
  N / 21 = 32   -- correct quotient when divided by 21
:= by sorry

end correct_quotient_proof_l1316_131611


namespace arithmetic_sequence_general_term_l1316_131627

/-- 
Given an arithmetic sequence {a_n} where the first three terms are a-1, a-1, and 2a+3,
prove that the general term formula is a_n = 2n-3.
-/
theorem arithmetic_sequence_general_term 
  (a_n : ℕ → ℝ) 
  (a : ℝ) 
  (h1 : a_n 1 = a - 1) 
  (h2 : a_n 2 = a - 1) 
  (h3 : a_n 3 = 2*a + 3) 
  (h_arithmetic : ∀ n : ℕ, a_n (n + 1) - a_n n = a_n (n + 2) - a_n (n + 1)) :
  ∀ n : ℕ, a_n n = 2*n - 3 :=
by sorry

end arithmetic_sequence_general_term_l1316_131627


namespace probability_spade_heart_spade_l1316_131601

/-- A standard deck of cards. -/
def StandardDeck : ℕ := 52

/-- The number of cards of each suit in a standard deck. -/
def CardsPerSuit : ℕ := 13

/-- The probability of drawing ♠, ♥, ♠ in sequence from a standard deck. -/
def ProbabilitySpadeHeartSpade : ℚ :=
  (CardsPerSuit : ℚ) / StandardDeck *
  (CardsPerSuit : ℚ) / (StandardDeck - 1) *
  (CardsPerSuit - 1 : ℚ) / (StandardDeck - 2)

theorem probability_spade_heart_spade :
  ProbabilitySpadeHeartSpade = 78 / 5100 := by
  sorry

end probability_spade_heart_spade_l1316_131601


namespace profit_share_difference_example_l1316_131641

/-- Given a total profit and a ratio of division between two parties, 
    calculate the difference between their shares. -/
def profit_share_difference (total_profit : ℚ) (ratio_x ratio_y : ℚ) : ℚ :=
  let total_ratio := ratio_x + ratio_y
  let share_x := (ratio_x / total_ratio) * total_profit
  let share_y := (ratio_y / total_ratio) * total_profit
  share_x - share_y

/-- Theorem stating that for a total profit of 500 and a ratio of 1/2 : 1/3, 
    the difference in profit shares is 100. -/
theorem profit_share_difference_example : 
  profit_share_difference 500 (1/2) (1/3) = 100 := by
  sorry

#eval profit_share_difference 500 (1/2) (1/3)

end profit_share_difference_example_l1316_131641


namespace middle_income_sample_size_l1316_131698

/-- Calculates the number of households to be drawn from a specific income group in a stratified sample. -/
def stratifiedSampleSize (totalHouseholds : ℕ) (groupHouseholds : ℕ) (sampleSize : ℕ) : ℕ :=
  (groupHouseholds * sampleSize) / totalHouseholds

/-- Proves that the number of middle-income households in the stratified sample is 60. -/
theorem middle_income_sample_size :
  let totalHouseholds : ℕ := 600
  let middleIncomeHouseholds : ℕ := 360
  let sampleSize : ℕ := 100
  stratifiedSampleSize totalHouseholds middleIncomeHouseholds sampleSize = 60 := by
  sorry


end middle_income_sample_size_l1316_131698


namespace test_score_for_three_hours_l1316_131610

/-- A model for a test score based on preparation time. -/
structure TestScore where
  maxPoints : ℝ
  scoreFunction : ℝ → ℝ
  knownScore : ℝ
  knownTime : ℝ

/-- Theorem: Given the conditions, prove that 3 hours of preparation results in a score of 202.5 -/
theorem test_score_for_three_hours 
  (test : TestScore)
  (h1 : test.maxPoints = 150)
  (h2 : ∀ t, test.scoreFunction t = (test.knownScore / test.knownTime^2) * t^2)
  (h3 : test.knownScore = 90)
  (h4 : test.knownTime = 2) :
  test.scoreFunction 3 = 202.5 := by
  sorry


end test_score_for_three_hours_l1316_131610


namespace fixed_point_of_line_family_l1316_131696

/-- The fixed point that a family of lines passes through -/
theorem fixed_point_of_line_family :
  ∃! p : ℝ × ℝ, ∀ m : ℝ, (2*m - 1) * p.1 + (m + 3) * p.2 - (m - 11) = 0 :=
by
  -- The unique point is (2, -3)
  use (2, -3)
  sorry

end fixed_point_of_line_family_l1316_131696


namespace probability_theorem_l1316_131686

-- Define the total number of children
def total_children : ℕ := 9

-- Define the number of children with green hats
def green_hats : ℕ := 3

-- Define the function to calculate the probability
def probability_no_adjacent_green_hats (n : ℕ) (k : ℕ) : ℚ :=
  -- The actual calculation would go here, but we'll use sorry to skip the proof
  5 / 14

-- State the theorem
theorem probability_theorem :
  probability_no_adjacent_green_hats total_children green_hats = 5 / 14 := by
  sorry

end probability_theorem_l1316_131686


namespace truck_max_load_l1316_131673

/-- The maximum load a truck can carry, given the mass of lemon bags and remaining capacity -/
theorem truck_max_load (mass_per_bag : ℕ) (num_bags : ℕ) (remaining_capacity : ℕ) :
  mass_per_bag = 8 →
  num_bags = 100 →
  remaining_capacity = 100 →
  mass_per_bag * num_bags + remaining_capacity = 900 := by
  sorry

end truck_max_load_l1316_131673


namespace triangle_angle_measure_l1316_131636

theorem triangle_angle_measure (X Y Z : ℝ) : 
  Y = 30 → Z = 3 * Y → X + Y + Z = 180 → X = 60 := by
  sorry

end triangle_angle_measure_l1316_131636


namespace three_prime_divisors_theorem_l1316_131650

theorem three_prime_divisors_theorem (x n : ℕ) :
  x = 2^n - 32 ∧
  (∃ p q : ℕ, Nat.Prime p ∧ Nat.Prime q ∧ p ≠ q ∧ p ≠ 2 ∧ q ≠ 2 ∧
    (∀ r : ℕ, Nat.Prime r → r ∣ x → r = 2 ∨ r = p ∨ r = q)) →
  x = 2016 ∨ x = 16352 := by
  sorry

end three_prime_divisors_theorem_l1316_131650


namespace max_subdivision_sides_l1316_131665

/-- Represents a convex polygon with n sides -/
structure ConvexPolygon (n : ℕ) where
  sides : n > 2

/-- Represents the maximum number of sides in a subdivision polygon -/
def maxSubdivisionSides (n : ℕ) : ℕ := n

/-- Theorem stating that the maximum number of sides in a subdivision polygon is n -/
theorem max_subdivision_sides (n : ℕ) (p : ConvexPolygon n) :
  maxSubdivisionSides n = n := by
  sorry

#eval maxSubdivisionSides 13    -- Should output 13
#eval maxSubdivisionSides 1950  -- Should output 1950

end max_subdivision_sides_l1316_131665


namespace triangle_square_diagonal_l1316_131661

/-- Given a triangle with base 6 and height 4, the length of the diagonal of a square 
    with the same area as the triangle is √24. -/
theorem triangle_square_diagonal : 
  ∀ (triangle_base triangle_height : ℝ),
  triangle_base = 6 →
  triangle_height = 4 →
  ∃ (square_diagonal : ℝ),
    (1/2 * triangle_base * triangle_height) = square_diagonal^2 / 2 ∧
    square_diagonal = Real.sqrt 24 :=
by sorry

end triangle_square_diagonal_l1316_131661


namespace geometric_progression_ratio_condition_l1316_131675

theorem geometric_progression_ratio_condition 
  (x y z w r : ℝ) 
  (h1 : x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧ w ≠ 0)
  (h2 : x * (y - z) ≠ y * (z - x) ∧ 
        y * (z - x) ≠ z * (x - y) ∧ 
        z * (x - y) ≠ w * (y - x))
  (h3 : ∃ (a : ℝ), a ≠ 0 ∧ 
        x * (y - z) = a ∧ 
        y * (z - x) = a * r ∧ 
        z * (x - y) = a * r^2 ∧ 
        w * (y - x) = a * r^3) :
  r^3 + r^2 + r + 1 = 0 := by
sorry

end geometric_progression_ratio_condition_l1316_131675


namespace odd_guess_probability_l1316_131624

theorem odd_guess_probability (n : ℕ) (hn : n = 2002) :
  (n - n / 3 : ℚ) / n > 2 / 3 := by
  sorry

end odd_guess_probability_l1316_131624


namespace annie_brownies_l1316_131633

def brownies_problem (total : ℕ) : Prop :=
  let after_admin : ℕ := total / 2
  let after_carl : ℕ := after_admin / 2
  let after_simon : ℕ := after_carl - 2
  (after_simon = 3) ∧ (total > 0)

theorem annie_brownies : ∃ (total : ℕ), brownies_problem total ∧ total = 20 := by
  sorry

end annie_brownies_l1316_131633


namespace yard_length_26_trees_l1316_131638

/-- The length of a yard with equally spaced trees -/
def yardLength (n : ℕ) (d : ℝ) : ℝ := (n - 1 : ℝ) * d

/-- Theorem: The length of a yard with 26 equally spaced trees, 
    one at each end, and 12 meters between consecutive trees, is 300 meters. -/
theorem yard_length_26_trees : yardLength 26 12 = 300 := by
  sorry

end yard_length_26_trees_l1316_131638


namespace fruit_juice_mixture_l1316_131695

/-- Given a 2-liter mixture that is 10% pure fruit juice, 
    adding 0.4 liters of pure fruit juice results in a 
    mixture that is 25% fruit juice -/
theorem fruit_juice_mixture : 
  let initial_volume : ℝ := 2
  let initial_percentage : ℝ := 0.1
  let added_volume : ℝ := 0.4
  let target_percentage : ℝ := 0.25
  let final_volume := initial_volume + added_volume
  let final_juice_volume := initial_volume * initial_percentage + added_volume
  final_juice_volume / final_volume = target_percentage :=
by sorry


end fruit_juice_mixture_l1316_131695


namespace apple_ratio_l1316_131689

theorem apple_ratio (jim_apples jane_apples jerry_apples : ℕ) 
  (h1 : jim_apples = 20)
  (h2 : jane_apples = 60)
  (h3 : jerry_apples = 40) :
  (jim_apples + jane_apples + jerry_apples) / 3 / jim_apples = 2 := by
  sorry

end apple_ratio_l1316_131689


namespace even_function_monotonicity_l1316_131697

def f (m : ℝ) (x : ℝ) : ℝ := (m - 1) * x^2 + 2 * m * x + 3

theorem even_function_monotonicity (m : ℝ) :
  (∀ x, f m x = f m (-x)) →
  ∃ a b c : ℝ, a < b ∧ b < c ∧
    (∀ x ∈ Set.Ioo (-3) 1, f m x = -x^2 + 3) ∧
    (∀ x y, -3 < x ∧ x < y ∧ y < a → f m x < f m y) ∧
    (∀ x y, a < x ∧ x < y ∧ y < 1 → f m x > f m y) :=
by sorry

end even_function_monotonicity_l1316_131697


namespace quadratic_root_theorem_l1316_131647

-- Define the quadratic equation
def quadratic (x k : ℝ) : ℝ := x^2 + 2*x + 3 - k

-- Define the condition for distinct real roots
def has_distinct_real_roots (k : ℝ) : Prop :=
  ∃ α β : ℝ, α ≠ β ∧ quadratic α k = 0 ∧ quadratic β k = 0

-- Define the relationship between k and the roots
def root_relationship (k α β : ℝ) : Prop :=
  k^2 = α * β + 3 * k

-- Theorem statement
theorem quadratic_root_theorem (k : ℝ) :
  has_distinct_real_roots k ∧ (∃ α β : ℝ, root_relationship k α β) → k = 3 :=
sorry

end quadratic_root_theorem_l1316_131647


namespace smallest_a_is_correct_l1316_131655

/-- The smallest positive integer a such that both 112 and 33 are factors of a * 43 * 62 * 1311 -/
def smallest_a : ℕ := 1848

/-- Predicate to check if a number divides the product a * 43 * 62 * 1311 -/
def is_factor (n : ℕ) (a : ℕ) : Prop :=
  (n : ℤ) ∣ (a * 43 * 62 * 1311 : ℤ)

theorem smallest_a_is_correct :
  (∀ a : ℕ, a > 0 → is_factor 112 a → is_factor 33 a → a ≥ smallest_a) ∧
  is_factor 112 smallest_a ∧
  is_factor 33 smallest_a :=
sorry

end smallest_a_is_correct_l1316_131655


namespace larger_number_proof_l1316_131605

theorem larger_number_proof (L S : ℕ) (hL : L > S) :
  L - S = 1365 → L = 6 * S + 20 → L = 1634 := by
  sorry

end larger_number_proof_l1316_131605


namespace rectangular_plot_breadth_l1316_131656

theorem rectangular_plot_breadth (length breadth area : ℝ) : 
  length = 3 * breadth → 
  area = length * breadth → 
  area = 588 → 
  breadth = 14 :=
by sorry

end rectangular_plot_breadth_l1316_131656


namespace joes_notebooks_l1316_131652

theorem joes_notebooks (initial_amount : ℕ) (notebook_cost : ℕ) (book_cost : ℕ) 
  (books_bought : ℕ) (amount_left : ℕ) : 
  initial_amount = 56 → 
  notebook_cost = 4 → 
  book_cost = 7 → 
  books_bought = 2 → 
  amount_left = 14 → 
  ∃ (notebooks_bought : ℕ), 
    notebooks_bought = 7 ∧ 
    initial_amount = notebook_cost * notebooks_bought + book_cost * books_bought + amount_left :=
by sorry

end joes_notebooks_l1316_131652


namespace volume_after_density_change_l1316_131632

/-- Given a substance with initial density and a density change factor, 
    calculate the new volume of a specified mass. -/
theorem volume_after_density_change 
  (initial_mass : ℝ) 
  (initial_volume : ℝ) 
  (density_change_factor : ℝ) 
  (mass_to_calculate : ℝ) 
  (h1 : initial_mass > 0)
  (h2 : initial_volume > 0)
  (h3 : density_change_factor > 0)
  (h4 : mass_to_calculate > 0)
  (h5 : initial_mass = 500)
  (h6 : initial_volume = 1)
  (h7 : density_change_factor = 1.25)
  (h8 : mass_to_calculate = 0.001) : 
  (mass_to_calculate / (initial_mass / initial_volume * density_change_factor)) * 1000000 = 1.6 := by
  sorry

#check volume_after_density_change

end volume_after_density_change_l1316_131632


namespace simplify_expression_l1316_131690

theorem simplify_expression : (2^8 + 4^5) * (1^3 - (-1)^3)^8 = 327680 := by
  sorry

end simplify_expression_l1316_131690


namespace alberts_cabbage_rows_l1316_131681

/-- Represents Albert's cabbage patch -/
structure CabbagePatch where
  total_heads : ℕ
  heads_per_row : ℕ

/-- Calculates the number of rows in the cabbage patch -/
def number_of_rows (patch : CabbagePatch) : ℕ :=
  patch.total_heads / patch.heads_per_row

/-- Theorem stating the number of rows in Albert's cabbage patch -/
theorem alberts_cabbage_rows :
  let patch : CabbagePatch := { total_heads := 180, heads_per_row := 15 }
  number_of_rows patch = 12 := by
  sorry

end alberts_cabbage_rows_l1316_131681


namespace power_of_power_l1316_131625

theorem power_of_power (a : ℝ) : (a^2)^3 = a^6 := by
  sorry

end power_of_power_l1316_131625


namespace not_always_perfect_square_l1316_131635

theorem not_always_perfect_square (d : ℕ) (h1 : d > 0) (h2 : d ≠ 2) (h3 : d ≠ 5) (h4 : d ≠ 13) :
  ∃ (a b : ℕ), a ∈ ({2, 5, 13, d} : Set ℕ) ∧ b ∈ ({2, 5, 13, d} : Set ℕ) ∧ a ≠ b ∧
  ¬∃ (k : ℕ), a * b - 1 = k * k :=
by sorry

end not_always_perfect_square_l1316_131635


namespace unique_solution_quadratic_positive_n_for_unique_solution_l1316_131616

/-- For a quadratic equation ax^2 + bx + c = 0, its discriminant is b^2 - 4ac --/
def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

/-- A quadratic equation has exactly one solution if and only if its discriminant is zero --/
def has_exactly_one_solution (a b c : ℝ) : Prop :=
  discriminant a b c = 0

theorem unique_solution_quadratic (n : ℝ) :
  has_exactly_one_solution 4 n 16 ↔ n = 16 ∨ n = -16 :=
sorry

theorem positive_n_for_unique_solution :
  ∃ n : ℝ, n > 0 ∧ has_exactly_one_solution 4 n 16 ∧ n = 16 :=
sorry

end unique_solution_quadratic_positive_n_for_unique_solution_l1316_131616


namespace min_baking_time_three_cakes_l1316_131614

/-- Represents a cake that needs to be baked on both sides -/
structure Cake where
  side1_baked : Bool
  side2_baked : Bool

/-- Represents a baking pan that can hold up to two cakes -/
structure Pan where
  capacity : Nat
  current_cakes : List Cake

/-- The time it takes to bake one side of a cake -/
def bake_time : Nat := 1

/-- The function to calculate the minimum baking time for all cakes -/
def min_baking_time (cakes : List Cake) (pan : Pan) : Nat :=
  sorry

/-- Theorem stating that the minimum baking time for three cakes is 3 minutes -/
theorem min_baking_time_three_cakes :
  let cakes := [Cake.mk false false, Cake.mk false false, Cake.mk false false]
  let pan := Pan.mk 2 []
  min_baking_time cakes pan = 3 := by
  sorry

end min_baking_time_three_cakes_l1316_131614


namespace monotonic_increasing_interval_of_f_l1316_131618

/-- The function f(x) = -x^2 + 2x - 2 -/
def f (x : ℝ) : ℝ := -x^2 + 2*x - 2

/-- The monotonic increasing interval of f(x) = -x^2 + 2x - 2 is (-∞, 1) -/
theorem monotonic_increasing_interval_of_f :
  ∀ x y : ℝ, x < y → y ≤ 1 → f x < f y :=
sorry

end monotonic_increasing_interval_of_f_l1316_131618


namespace six_digit_integers_count_l1316_131691

/-- The number of different positive six-digit integers formed using the digits 1, 1, 1, 5, 9, and 9 -/
def six_digit_integers : ℕ :=
  Nat.factorial 6 / (Nat.factorial 3 * Nat.factorial 2)

/-- Theorem stating that the number of different positive six-digit integers
    formed using the digits 1, 1, 1, 5, 9, and 9 is equal to 60 -/
theorem six_digit_integers_count : six_digit_integers = 60 := by
  sorry

end six_digit_integers_count_l1316_131691


namespace interest_rate_calculation_l1316_131693

theorem interest_rate_calculation (initial_amount loan_amount final_amount : ℚ) :
  initial_amount = 30 →
  loan_amount = 15 →
  final_amount = 33 →
  (final_amount - initial_amount) / loan_amount * 100 = 20 := by
sorry

end interest_rate_calculation_l1316_131693


namespace parallelogram_area_l1316_131609

/-- The area of a parallelogram with one angle of 150 degrees and two consecutive sides of lengths 10 and 20 is 100√3. -/
theorem parallelogram_area (a b : ℝ) (θ : ℝ) (h1 : a = 10) (h2 : b = 20) (h3 : θ = 150 * π / 180) :
  a * b * Real.sin (π - θ) = 100 * Real.sqrt 3 := by
  sorry

end parallelogram_area_l1316_131609


namespace inequality_proof_l1316_131682

theorem inequality_proof (x y : ℝ) (hx : |x| < 1) (hy : |y| < 1) :
  (1 / (1 - x^2)) + (1 / (1 - y^2)) ≥ 2 / (1 - x*y) := by
  sorry

end inequality_proof_l1316_131682


namespace number_exceeding_fraction_l1316_131662

theorem number_exceeding_fraction (x : ℝ) : x = (3/7 + 0.8 * (3/7)) * x → x = (35/27) * x := by
  sorry

end number_exceeding_fraction_l1316_131662


namespace geometric_sequence_property_l1316_131692

-- Define a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (r : ℝ) (a₁ : ℝ), ∀ n, a n = a₁ * r ^ (n - 1)

-- Define the property of three terms forming a geometric sequence
def form_geometric_sequence (x y z : ℝ) : Prop :=
  y * y = x * z

-- Theorem statement
theorem geometric_sequence_property (a : ℕ → ℝ) :
  is_geometric_sequence a →
  form_geometric_sequence (a 3) (a 6) (a 9) := by sorry

end geometric_sequence_property_l1316_131692


namespace min_value_theorem_equality_condition_l1316_131642

theorem min_value_theorem (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h_sum : x^2 + y^2 + z^2 = 1) : 
  (x^5 / (y^2 + z^2 - y*z)) + (y^5 / (z^2 + x^2 - z*x)) + (z^5 / (x^2 + y^2 - x*y)) ≥ Real.sqrt 3 / 3 :=
by sorry

theorem equality_condition (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h_sum : x^2 + y^2 + z^2 = 1) : 
  (x^5 / (y^2 + z^2 - y*z)) + (y^5 / (z^2 + x^2 - z*x)) + (z^5 / (x^2 + y^2 - x*y)) = Real.sqrt 3 / 3 ↔ 
  x = 1 / Real.sqrt 3 ∧ y = 1 / Real.sqrt 3 ∧ z = 1 / Real.sqrt 3 :=
by sorry

end min_value_theorem_equality_condition_l1316_131642


namespace gcd_of_three_numbers_l1316_131664

theorem gcd_of_three_numbers : Nat.gcd 12357 (Nat.gcd 15498 21726) = 3 := by
  sorry

end gcd_of_three_numbers_l1316_131664


namespace beckett_olaf_age_difference_l1316_131643

/-- Given the ages of four people satisfying certain conditions, prove that Beckett is 8 years younger than Olaf. -/
theorem beckett_olaf_age_difference :
  ∀ (beckett_age olaf_age shannen_age jack_age : ℕ),
    beckett_age = 12 →
    ∃ (x : ℕ), beckett_age + x = olaf_age →
    shannen_age + 2 = olaf_age →
    jack_age = 2 * shannen_age + 5 →
    beckett_age + olaf_age + shannen_age + jack_age = 71 →
    x = 8 := by
  sorry

end beckett_olaf_age_difference_l1316_131643


namespace solution_set_of_inequality_l1316_131677

theorem solution_set_of_inequality (x : ℝ) : 
  (x^2 - x < 0) ↔ (0 < x ∧ x < 1) :=
by sorry

end solution_set_of_inequality_l1316_131677


namespace square_root_of_sixteen_l1316_131629

theorem square_root_of_sixteen : 
  ∃ (x : ℝ), x^2 = 16 ∧ (x = 4 ∨ x = -4) :=
sorry

end square_root_of_sixteen_l1316_131629


namespace fraction_equality_implies_four_l1316_131674

theorem fraction_equality_implies_four (k n m : ℕ+) :
  (1 : ℚ) / n^2 + (1 : ℚ) / m^2 = (k : ℚ) / (n^2 + m^2) →
  k = 4 := by
  sorry

end fraction_equality_implies_four_l1316_131674


namespace number_of_divisors_30030_l1316_131680

theorem number_of_divisors_30030 : Nat.card (Nat.divisors 30030) = 64 := by
  sorry

end number_of_divisors_30030_l1316_131680


namespace student_allocation_arrangements_l1316_131608

theorem student_allocation_arrangements : 
  let n : ℕ := 4  -- number of students
  let m : ℕ := 3  -- number of locations
  let arrangements := {f : Fin n → Fin m | ∀ i : Fin m, ∃ j : Fin n, f j = i}
  Fintype.card arrangements = 36 := by
sorry

end student_allocation_arrangements_l1316_131608


namespace cubic_identity_l1316_131699

theorem cubic_identity (x : ℝ) : 
  (2*x - 1)^3 = 5*x^3 + (3*x + 1)*(x^2 - x - 1) - 10*x^2 + 10*x := by
  sorry

end cubic_identity_l1316_131699


namespace sarah_pencils_count_l1316_131603

/-- The number of pencils Sarah buys on Monday -/
def monday_pencils : ℕ := 20

/-- The number of pencils Sarah buys on Tuesday -/
def tuesday_pencils : ℕ := 18

/-- The number of pencils Sarah buys on Wednesday -/
def wednesday_pencils : ℕ := 3 * tuesday_pencils

/-- The total number of pencils Sarah has -/
def total_pencils : ℕ := monday_pencils + tuesday_pencils + wednesday_pencils

theorem sarah_pencils_count : total_pencils = 92 := by
  sorry

end sarah_pencils_count_l1316_131603


namespace tangent_line_at_zero_derivative_monotone_increasing_f_superadditive_l1316_131645

noncomputable def f (x : ℝ) : ℝ := Real.exp x * Real.log (x + 1)

theorem tangent_line_at_zero (x : ℝ) :
  (deriv f) 0 = 1 :=
sorry

theorem derivative_monotone_increasing :
  StrictMonoOn (deriv f) (Set.Icc 0 2) :=
sorry

theorem f_superadditive {s t : ℝ} (hs : s > 0) (ht : t > 0) :
  f (s + t) > f s + f t :=
sorry

end tangent_line_at_zero_derivative_monotone_increasing_f_superadditive_l1316_131645


namespace min_value_problem_min_value_attained_l1316_131684

theorem min_value_problem (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + y = 1) :
  (x^2 / (x + 2) + y^2 / (y + 1)) ≥ 1/4 :=
by sorry

theorem min_value_attained (ε : ℝ) (hε : ε > 0) :
  ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x + y = 1 ∧
    (x^2 / (x + 2) + y^2 / (y + 1)) < 1/4 + ε :=
by sorry

end min_value_problem_min_value_attained_l1316_131684


namespace eleven_girls_l1316_131667

/-- Represents a circular arrangement of girls -/
structure CircularArrangement where
  girls : ℕ  -- Total number of girls in the circle

/-- Defines the position of one girl relative to another in the circle -/
def position (c : CircularArrangement) (left right : ℕ) : Prop :=
  left + right + 2 = c.girls

/-- Theorem: If Florence is the 4th on the left and 7th on the right from Jess,
    then there are 11 girls in total -/
theorem eleven_girls (c : CircularArrangement) :
  position c 3 6 → c.girls = 11 := by
  sorry

#check eleven_girls

end eleven_girls_l1316_131667


namespace sally_balloons_l1316_131651

theorem sally_balloons (initial : ℕ) (lost : ℕ) (final : ℕ) : 
  initial = 20 → lost = 5 → final = (initial - lost) * 2 → final = 30 := by
sorry

end sally_balloons_l1316_131651


namespace distribute_four_into_two_l1316_131640

/-- The number of ways to distribute n distinguishable objects into k distinguishable containers -/
def distribute (n k : ℕ) : ℕ := k^n

/-- Theorem: There are 16 ways to distribute 4 distinguishable balls into 2 distinguishable boxes -/
theorem distribute_four_into_two : distribute 4 2 = 16 := by
  sorry

end distribute_four_into_two_l1316_131640


namespace line_through_point_l1316_131649

/-- Given a line ax + (a+1)y = a + 4 passing through the point (3, -7), prove that a = -11/5 --/
theorem line_through_point (a : ℚ) : 
  (a * 3 + (a + 1) * (-7) = a + 4) → a = -11/5 := by
  sorry

end line_through_point_l1316_131649


namespace equation_solution_l1316_131646

theorem equation_solution : ∃ x : ℝ, (3 / (x - 1) = 5 + 3 * x / (1 - x)) ∧ x = 4 := by
  sorry

end equation_solution_l1316_131646


namespace quadratic_root_property_l1316_131653

theorem quadratic_root_property (a : ℝ) : 
  (2 * a^2 = 6 * a - 4) → (a^2 - 3 * a + 2024 = 2022) := by
  sorry

end quadratic_root_property_l1316_131653


namespace combined_area_rhombus_circle_l1316_131648

/-- The combined area of a rhombus and a circle -/
theorem combined_area_rhombus_circle (d1 d2 r : ℝ) (h1 : d1 = 40) (h2 : d2 = 30) (h3 : r = 10) :
  (d1 * d2 / 2) + (π * r^2) = 600 + 100 * π := by
  sorry

end combined_area_rhombus_circle_l1316_131648


namespace four_digit_number_with_specific_factors_l1316_131631

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def count_prime_factors (n : ℕ) : ℕ := sorry
def count_non_prime_factors (n : ℕ) : ℕ := sorry
def count_total_factors (n : ℕ) : ℕ := sorry

theorem four_digit_number_with_specific_factors :
  ∃ (n : ℕ), is_four_digit n ∧ 
             count_prime_factors n = 3 ∧ 
             count_non_prime_factors n = 39 ∧ 
             count_total_factors n = 42 :=
sorry

end four_digit_number_with_specific_factors_l1316_131631


namespace cos_pi_minus_2alpha_l1316_131606

theorem cos_pi_minus_2alpha (α : ℝ) (h : Real.cos (π / 2 - α) = Real.sqrt 2 / 3) : 
  Real.cos (π - 2 * α) = -5 / 9 := by
  sorry

end cos_pi_minus_2alpha_l1316_131606


namespace gcd_of_three_numbers_l1316_131623

theorem gcd_of_three_numbers : Nat.gcd 9125 (Nat.gcd 4257 2349) = 1 := by
  sorry

end gcd_of_three_numbers_l1316_131623


namespace not_divisible_by_nine_l1316_131659

theorem not_divisible_by_nine (n : ℕ) : ¬ (9 ∣ (n^3 + 2)) := by
  sorry

end not_divisible_by_nine_l1316_131659


namespace binomial_12_3_l1316_131617

theorem binomial_12_3 : Nat.choose 12 3 = 220 := by
  sorry

end binomial_12_3_l1316_131617


namespace square_root_of_sixteen_l1316_131622

theorem square_root_of_sixteen : Real.sqrt 16 = 4 := by
  sorry

end square_root_of_sixteen_l1316_131622


namespace workout_difference_l1316_131630

/-- Represents Oliver's workout schedule over four days -/
structure WorkoutSchedule where
  monday : ℝ
  tuesday : ℝ
  wednesday : ℝ
  thursday : ℝ

/-- Checks if a workout schedule satisfies the given conditions -/
def is_valid_schedule (s : WorkoutSchedule) : Prop :=
  s.monday = 4 ∧
  s.tuesday < s.monday ∧
  s.wednesday = 2 * s.monday ∧
  s.thursday = 2 * s.tuesday ∧
  s.monday + s.tuesday + s.wednesday + s.thursday = 18

/-- Theorem stating that for any valid workout schedule, 
    the difference between Monday's and Tuesday's workout time is 2 hours -/
theorem workout_difference (s : WorkoutSchedule) 
  (h : is_valid_schedule s) : s.monday - s.tuesday = 2 := by
  sorry

end workout_difference_l1316_131630


namespace function_transformation_l1316_131688

theorem function_transformation (f : ℝ → ℝ) (h : f 1 = 3) : 
  f 1 + 1 = 4 := by sorry

end function_transformation_l1316_131688


namespace ariel_fish_count_l1316_131678

theorem ariel_fish_count (total : ℕ) (male_fraction : ℚ) (female_count : ℕ) : 
  total = 45 → male_fraction = 2/3 → female_count = total - (total * male_fraction).num → female_count = 15 := by
  sorry

end ariel_fish_count_l1316_131678


namespace binary_100_is_4_binary_101_is_5_binary_1100_is_12_l1316_131671

-- Define binary to decimal conversion function
def binaryToDecimal (b : List Bool) : ℕ :=
  b.foldl (fun acc x => 2 * acc + if x then 1 else 0) 0

-- Define decimal numbers
def four : ℕ := 4
def five : ℕ := 5
def twelve : ℕ := 12

-- Define binary numbers
def binary_100 : List Bool := [true, false, false]
def binary_101 : List Bool := [true, false, true]
def binary_1100 : List Bool := [true, true, false, false]

-- Theorem statements
theorem binary_100_is_4 : binaryToDecimal binary_100 = four := by sorry

theorem binary_101_is_5 : binaryToDecimal binary_101 = five := by sorry

theorem binary_1100_is_12 : binaryToDecimal binary_1100 = twelve := by sorry

end binary_100_is_4_binary_101_is_5_binary_1100_is_12_l1316_131671


namespace trig_expression_simplification_l1316_131613

theorem trig_expression_simplification :
  let left_numerator := Real.sin (15 * π / 180) + Real.sin (30 * π / 180) + 
                        Real.sin (45 * π / 180) + Real.sin (60 * π / 180) + 
                        Real.sin (75 * π / 180)
  let denominator := Real.cos (10 * π / 180) * Real.cos (20 * π / 180) * 
                     Real.cos (30 * π / 180) * 2
  let right_numerator := 2 * Real.sqrt 2 * Real.cos (22.5 * π / 180) * 
                         Real.cos (7.5 * π / 180)
  left_numerator / denominator = right_numerator / denominator := by
  sorry

end trig_expression_simplification_l1316_131613
