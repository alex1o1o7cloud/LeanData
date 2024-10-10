import Mathlib

namespace smallest_n_for_quadruplets_l2793_279315

def count_quadruplets (n : ℕ) : ℕ :=
  sorry

theorem smallest_n_for_quadruplets : 
  (∃ (n : ℕ), 
    n > 0 ∧ 
    count_quadruplets n = 50000 ∧
    (∀ (a b c d : ℕ), 
      (Nat.gcd a (Nat.gcd b (Nat.gcd c d)) = 65 ∧ 
       Nat.lcm a (Nat.lcm b (Nat.lcm c d)) = n) → 
      (a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0)) ∧
    (∀ (m : ℕ), m < n → 
      (count_quadruplets m ≠ 50000 ∨
       ∃ (a b c d : ℕ), 
         (Nat.gcd a (Nat.gcd b (Nat.gcd c d)) = 65 ∧ 
          Nat.lcm a (Nat.lcm b (Nat.lcm c d)) = m) ∧
         (a = 0 ∨ b = 0 ∨ c = 0 ∨ d = 0)))) ∧
  (∀ (n : ℕ), 
    n > 0 ∧ 
    count_quadruplets n = 50000 ∧
    (∀ (a b c d : ℕ), 
      (Nat.gcd a (Nat.gcd b (Nat.gcd c d)) = 65 ∧ 
       Nat.lcm a (Nat.lcm b (Nat.lcm c d)) = n) → 
      (a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0)) →
    n ≥ 4459000) ∧
  count_quadruplets 4459000 = 50000 ∧
  (∀ (a b c d : ℕ), 
    (Nat.gcd a (Nat.gcd b (Nat.gcd c d)) = 65 ∧ 
     Nat.lcm a (Nat.lcm b (Nat.lcm c d)) = 4459000) → 
    (a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0)) :=
by sorry

end smallest_n_for_quadruplets_l2793_279315


namespace expand_expression_l2793_279331

theorem expand_expression (x : ℝ) : 24 * (3 * x - 4) = 72 * x - 96 := by
  sorry

end expand_expression_l2793_279331


namespace min_perimeter_isosceles_triangles_l2793_279325

/-- Represents an isosceles triangle with integer side lengths -/
structure IsoscelesTriangle where
  side : ℕ  -- Length of equal sides
  base : ℕ  -- Length of the base
  isValid : 2 * side > base  -- Triangle inequality

/-- Check if two triangles have the same perimeter -/
def samePerimeter (t1 t2 : IsoscelesTriangle) : Prop :=
  2 * t1.side + t1.base = 2 * t2.side + t2.base

/-- Check if two triangles have the same area -/
def sameArea (t1 t2 : IsoscelesTriangle) : Prop :=
  t1.base * (t1.side ^ 2 - (t1.base / 2) ^ 2).sqrt = 
  t2.base * (t2.side ^ 2 - (t2.base / 2) ^ 2).sqrt

/-- Check if the base ratio of two triangles is 5:4 -/
def baseRatio54 (t1 t2 : IsoscelesTriangle) : Prop :=
  5 * t2.base = 4 * t1.base

/-- The main theorem -/
theorem min_perimeter_isosceles_triangles :
  ∃ (t1 t2 : IsoscelesTriangle),
    t1 ≠ t2 ∧
    samePerimeter t1 t2 ∧
    sameArea t1 t2 ∧
    baseRatio54 t1 t2 ∧
    (∀ (s1 s2 : IsoscelesTriangle),
      s1 ≠ s2 →
      samePerimeter s1 s2 →
      sameArea s1 s2 →
      baseRatio54 s1 s2 →
      2 * t1.side + t1.base ≤ 2 * s1.side + s1.base) ∧
    2 * t1.side + t1.base = 138 :=
sorry

end min_perimeter_isosceles_triangles_l2793_279325


namespace walk_group_legs_and_wheels_l2793_279390

/-- Calculates the total number of legs and wheels in a group of humans, dogs, and wheelchairs. -/
def total_legs_and_wheels (num_humans : ℕ) (num_dogs : ℕ) (num_wheelchairs : ℕ) : ℕ :=
  num_humans * 2 + num_dogs * 4 + num_wheelchairs * 4

/-- Proves that the total number of legs and wheels in the given group is 22. -/
theorem walk_group_legs_and_wheels :
  total_legs_and_wheels 3 3 1 = 22 := by
  sorry

end walk_group_legs_and_wheels_l2793_279390


namespace alberto_bjorn_bike_distance_l2793_279366

/-- The problem of comparing distances biked by Alberto and Bjorn -/
theorem alberto_bjorn_bike_distance :
  let alberto_rate : ℝ := 80 / 4  -- Alberto's constant rate in miles per hour
  let alberto_time : ℝ := 4  -- Alberto's total time in hours
  let bjorn_rate1 : ℝ := 20  -- Bjorn's first rate in miles per hour
  let bjorn_rate2 : ℝ := 25  -- Bjorn's second rate in miles per hour
  let bjorn_time1 : ℝ := 2  -- Bjorn's time at first rate in hours
  let bjorn_time2 : ℝ := 2  -- Bjorn's time at second rate in hours
  
  let alberto_distance : ℝ := alberto_rate * alberto_time
  let bjorn_distance : ℝ := bjorn_rate1 * bjorn_time1 + bjorn_rate2 * bjorn_time2
  
  alberto_distance - bjorn_distance = -10
  := by sorry

end alberto_bjorn_bike_distance_l2793_279366


namespace solution_set_when_a_is_one_range_of_a_l2793_279396

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| + |x + 3|

-- Part 1
theorem solution_set_when_a_is_one :
  {x : ℝ | f 1 x ≥ 6} = Set.Iic (-4) ∪ Set.Ici 2 :=
sorry

-- Part 2
theorem range_of_a :
  {a : ℝ | ∀ x, f a x > -a} = Set.Ioi (-3/2) :=
sorry

end solution_set_when_a_is_one_range_of_a_l2793_279396


namespace smallest_positive_integer_l2793_279361

theorem smallest_positive_integer (a : ℝ) : 
  ∃ (b : ℤ), (∀ (x : ℝ), (x + 2) * (x + 5) * (x + 8) * (x + 11) + b > 0) ∧ 
  (∀ (c : ℤ), c < b → ∃ (y : ℝ), (y + 2) * (y + 5) * (y + 8) * (y + 11) + c ≤ 0) ∧
  b = 82 :=
by sorry

end smallest_positive_integer_l2793_279361


namespace collinear_dots_probability_l2793_279302

/-- Represents a 5x5 grid of dots -/
def Grid := Fin 5 × Fin 5

/-- The total number of dots in the grid -/
def total_dots : Nat := 25

/-- The number of ways to choose 4 dots from the total dots -/
def total_choices : Nat := Nat.choose total_dots 4

/-- The number of sets of 4 collinear dots in the grid -/
def collinear_sets : Nat := 28

/-- The probability of choosing 4 collinear dots -/
def collinear_probability : Rat := collinear_sets / total_choices

theorem collinear_dots_probability :
  collinear_probability = 4 / 1807 := by sorry

end collinear_dots_probability_l2793_279302


namespace stream_speed_calculation_l2793_279388

/-- Proves that given a boat with a speed of 20 km/hr in still water,
    traveling 125 km downstream in 5 hours, the speed of the stream is 5 km/hr. -/
theorem stream_speed_calculation (boat_speed : ℝ) (downstream_distance : ℝ) (downstream_time : ℝ) :
  boat_speed = 20 →
  downstream_distance = 125 →
  downstream_time = 5 →
  (boat_speed + (downstream_distance / downstream_time - boat_speed)) = 25 :=
by
  sorry

#check stream_speed_calculation

end stream_speed_calculation_l2793_279388


namespace multiply_and_simplify_l2793_279384

theorem multiply_and_simplify (x : ℝ) : 
  (x^4 + 6*x^2 + 9) * (x^2 - 3) = x^4 + 6*x^2 := by
  sorry

end multiply_and_simplify_l2793_279384


namespace quadratic_function_satisfies_conditions_l2793_279324

/-- The quadratic equation x^2 - x + 1 = 0 has roots α and β -/
def has_roots (α β : ℂ) : Prop :=
  α^2 - α + 1 = 0 ∧ β^2 - β + 1 = 0

/-- The quadratic function f(x) = x^2 - 2x + 2 -/
def f (x : ℂ) : ℂ := x^2 - 2*x + 2

/-- Theorem stating that f(x) satisfies the required conditions -/
theorem quadratic_function_satisfies_conditions (α β : ℂ) 
  (h : has_roots α β) : f α = β ∧ f β = α ∧ f 1 = 1 := by
  sorry

end quadratic_function_satisfies_conditions_l2793_279324


namespace bob_rope_art_fraction_l2793_279365

theorem bob_rope_art_fraction (total_length : ℝ) (remaining_length : ℝ) (num_sections : ℕ) (section_length : ℝ) : 
  total_length = 50 ∧ 
  remaining_length = 20 ∧ 
  num_sections = 10 ∧ 
  section_length = 2 ∧ 
  remaining_length = num_sections * section_length →
  (total_length - remaining_length * 2) / total_length = 1 / 5 := by
  sorry

end bob_rope_art_fraction_l2793_279365


namespace least_common_period_is_36_l2793_279351

/-- A function satisfying the given functional equation -/
def SatisfiesFunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + 6) + f (x - 6) = f x

/-- The period of a function -/
def IsPeriod (f : ℝ → ℝ) (p : ℝ) : Prop :=
  p > 0 ∧ ∀ x, f (x + p) = f x

/-- The least common positive period for all functions satisfying the functional equation -/
def LeastCommonPeriod (p : ℝ) : Prop :=
  p > 0 ∧
  (∀ f : ℝ → ℝ, SatisfiesFunctionalEquation f → IsPeriod f p) ∧
  ∀ q, q > 0 → (∀ f : ℝ → ℝ, SatisfiesFunctionalEquation f → IsPeriod f q) → p ≤ q

theorem least_common_period_is_36 : LeastCommonPeriod 36 := by
  sorry

end least_common_period_is_36_l2793_279351


namespace class_size_is_ten_l2793_279327

/-- The number of students who scored 92 -/
def high_scorers : ℕ := 5

/-- The number of students who scored 80 -/
def mid_scorers : ℕ := 4

/-- The score of the last student -/
def last_score : ℕ := 70

/-- The minimum required average score -/
def min_average : ℕ := 85

/-- The total number of students in the class -/
def total_students : ℕ := high_scorers + mid_scorers + 1

theorem class_size_is_ten :
  total_students = 10 ∧
  (high_scorers * 92 + mid_scorers * 80 + last_score) / total_students ≥ min_average := by
  sorry

end class_size_is_ten_l2793_279327


namespace ambiguous_triangle_case_l2793_279395

/-- Given two sides and an angle of a triangle, proves the existence of conditions
    for obtaining two different values for the third side. -/
theorem ambiguous_triangle_case (a b : ℝ) (α : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a < b)
  (h4 : 0 < α) (h5 : α < π) :
  ∃ c1 c2 : ℝ, c1 ≠ c2 ∧ 
  (∃ β γ : ℝ, 
    0 < β ∧ 0 < γ ∧ 
    α + β + γ = π ∧
    a / Real.sin α = b / Real.sin β ∧
    a / Real.sin α = c1 / Real.sin γ) ∧
  (∃ β' γ' : ℝ, 
    0 < β' ∧ 0 < γ' ∧ 
    α + β' + γ' = π ∧
    a / Real.sin α = b / Real.sin β' ∧
    a / Real.sin α = c2 / Real.sin γ') :=
sorry

end ambiguous_triangle_case_l2793_279395


namespace circle_area_diameter_increase_l2793_279349

theorem circle_area_diameter_increase (A D A' D' : ℝ) :
  A' = 6 * A →
  A = (π / 4) * D^2 →
  A' = (π / 4) * D'^2 →
  D' = Real.sqrt 6 * D :=
by sorry

end circle_area_diameter_increase_l2793_279349


namespace squirrel_nuts_collected_l2793_279374

/-- Represents the number of nuts eaten on day k -/
def nutsEatenOnDay (k : ℕ) : ℕ := k

/-- Represents the fraction of remaining nuts eaten each day -/
def fractionEaten : ℚ := 1 / 100

/-- Represents the number of nuts remaining before eating on day k -/
def nutsRemaining (k : ℕ) (totalNuts : ℕ) : ℕ :=
  totalNuts - (k - 1) * (k - 1 + 1) / 2

/-- Represents the number of nuts eaten on day k including the fraction -/
def totalNutsEatenOnDay (k : ℕ) (totalNuts : ℕ) : ℚ :=
  nutsEatenOnDay k + fractionEaten * (nutsRemaining k totalNuts - nutsEatenOnDay k)

/-- The theorem stating the total number of nuts collected by the squirrel -/
theorem squirrel_nuts_collected :
  ∃ n : ℕ, n > 0 ∧
    (∀ k : ℕ, k < n → totalNutsEatenOnDay k 9801 < nutsRemaining k 9801) ∧
    nutsRemaining n 9801 = n :=
  sorry

end squirrel_nuts_collected_l2793_279374


namespace most_suitable_for_sample_survey_l2793_279336

/-- Represents a survey scenario -/
structure SurveyScenario where
  name : String
  quantity : Nat
  easySurvey : Bool

/-- Determines if a scenario is suitable for a sample survey -/
def suitableForSampleSurvey (scenario : SurveyScenario) : Prop :=
  scenario.quantity > 1000 ∧ ¬scenario.easySurvey

/-- The list of survey scenarios -/
def scenarios : List SurveyScenario := [
  { name := "Body temperature during H1N1", quantity := 100, easySurvey := false },
  { name := "Quality of Zongzi from Wufangzhai", quantity := 10000, easySurvey := false },
  { name := "Vision condition of classmates", quantity := 50, easySurvey := true },
  { name := "Mathematics learning in eighth grade", quantity := 200, easySurvey := true }
]

theorem most_suitable_for_sample_survey :
  ∃ (s : SurveyScenario), s ∈ scenarios ∧ 
  suitableForSampleSurvey s ∧ 
  (∀ (t : SurveyScenario), t ∈ scenarios → suitableForSampleSurvey t → s = t) :=
sorry

end most_suitable_for_sample_survey_l2793_279336


namespace equation_solutions_l2793_279343

theorem equation_solutions :
  (∃! x : ℝ, x^2 - 2*x = -1) ∧
  (∀ x : ℝ, (x + 3)^2 = 2*x*(x + 3) ↔ x = -3 ∨ x = 3) := by
  sorry

end equation_solutions_l2793_279343


namespace pascal_and_coin_toss_l2793_279356

/-- Pascal's Triangle row sum -/
def pascal_row_sum (n : ℕ) : ℕ := 2^n

/-- Binomial coefficient -/
def binomial (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

/-- Probability of k successes in n independent trials -/
def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (binomial n k : ℚ) * p^k * (1 - p)^(n - k)

theorem pascal_and_coin_toss :
  pascal_row_sum 10 = 1024 ∧
  binomial_probability 10 5 (1/2) = 63/256 := by sorry

end pascal_and_coin_toss_l2793_279356


namespace boys_from_maple_high_school_l2793_279322

theorem boys_from_maple_high_school (total_students : ℕ) (total_boys : ℕ) (total_girls : ℕ)
  (jonas_students : ℕ) (clay_students : ℕ) (maple_students : ℕ)
  (jonas_girls : ℕ) (clay_girls : ℕ) :
  total_students = 150 →
  total_boys = 85 →
  total_girls = 65 →
  jonas_students = 50 →
  clay_students = 70 →
  maple_students = 30 →
  jonas_girls = 25 →
  clay_girls = 30 →
  total_students = total_boys + total_girls →
  total_students = jonas_students + clay_students + maple_students →
  (maple_students - (total_girls - jonas_girls - clay_girls) : ℤ) = 20 := by
sorry

end boys_from_maple_high_school_l2793_279322


namespace gcd_of_squares_sum_l2793_279341

theorem gcd_of_squares_sum : Nat.gcd (123^2 + 235^2 + 347^2) (122^2 + 234^2 + 348^2) = 1 := by
  sorry

end gcd_of_squares_sum_l2793_279341


namespace interest_difference_implies_sum_l2793_279317

/-- Proves that if the difference between compound interest and simple interest
    on a sum at 5% per annum for 2 years is Rs. 60, then the sum is Rs. 24,000. -/
theorem interest_difference_implies_sum (P : ℝ) : 
  P * ((1 + 0.05)^2 - 1) - P * (0.05 * 2) = 60 → P = 24000 := by
  sorry

end interest_difference_implies_sum_l2793_279317


namespace unique_quadratic_solution_l2793_279386

theorem unique_quadratic_solution (a c : ℝ) : 
  (∃! x, a * x^2 + 10 * x + c = 0) →  -- exactly one solution
  (a + c = 12) →                      -- sum condition
  (a < c) →                           -- order condition
  (a = 6 - Real.sqrt 11 ∧ c = 6 + Real.sqrt 11) := by
sorry

end unique_quadratic_solution_l2793_279386


namespace hall_breadth_is_12_l2793_279342

def hall_length : ℝ := 15
def hall_volume : ℝ := 1200

theorem hall_breadth_is_12 (b h : ℝ) 
  (area_eq : 2 * (hall_length * b) = 2 * (hall_length * h + b * h))
  (volume_eq : hall_length * b * h = hall_volume) :
  b = 12 := by sorry

end hall_breadth_is_12_l2793_279342


namespace speed_of_train_b_l2793_279394

/-- Theorem: Speed of Train B
Given two trains A and B traveling in opposite directions, meeting at some point,
with train A reaching its destination 9 hours after meeting and traveling at 70 km/h,
and train B reaching its destination 4 hours after meeting,
prove that the speed of train B is 157.5 km/h. -/
theorem speed_of_train_b (speed_a : ℝ) (time_a time_b : ℝ) (speed_b : ℝ) :
  speed_a = 70 →
  time_a = 9 →
  time_b = 4 →
  speed_a * time_a = speed_b * time_b →
  speed_b = 157.5 := by
sorry

end speed_of_train_b_l2793_279394


namespace language_group_selection_l2793_279337

theorem language_group_selection (total : Nat) (english : Nat) (japanese : Nat)
  (h_total : total = 9)
  (h_english : english = 7)
  (h_japanese : japanese = 3)
  (h_at_least_one : english + japanese ≥ total) :
  (english * japanese) - (english + japanese - total) = 20 := by
  sorry

end language_group_selection_l2793_279337


namespace bracket_computation_l2793_279364

-- Define the operation [x, y, z]
def bracket (x y z : ℚ) : ℚ := (x + y) / z

-- Theorem statement
theorem bracket_computation :
  bracket (bracket 120 60 180) (bracket 4 2 6) (bracket 20 10 30) = 2 := by
  sorry

end bracket_computation_l2793_279364


namespace arithmetic_sequence_general_term_l2793_279339

def arithmeticSequence (a₁ : ℝ) (d : ℝ) : ℕ → ℝ :=
  λ n => a₁ + d * (n - 1)

theorem arithmetic_sequence_general_term :
  let a₁ : ℝ := -1
  let d : ℝ := 4
  ∀ n : ℕ, arithmeticSequence a₁ d n = 4 * n - 5 := by
sorry

end arithmetic_sequence_general_term_l2793_279339


namespace marks_score_is_46_l2793_279358

def highest_score : ℕ := 98
def score_range : ℕ := 75

def least_score : ℕ := highest_score - score_range

def marks_score : ℕ := 2 * least_score

theorem marks_score_is_46 : marks_score = 46 := by
  sorry

end marks_score_is_46_l2793_279358


namespace systematic_sampling_l2793_279323

theorem systematic_sampling (n : Nat) (groups : Nat) (last_group_num : Nat) :
  n = 100 ∧ groups = 5 ∧ last_group_num = 94 →
  ∃ (interval : Nat) (first_group_num : Nat),
    interval * (groups - 1) + first_group_num = last_group_num ∧
    interval * 1 + first_group_num = 34 :=
sorry

end systematic_sampling_l2793_279323


namespace triangle_problem_l2793_279308

theorem triangle_problem (A B C : Real) (a b c : Real) :
  (0 < A) ∧ (A < π) ∧
  (0 < B) ∧ (B < π) ∧
  (0 < C) ∧ (C < π) ∧
  (A + B + C = π) ∧
  (c * Real.cos B + b * Real.cos C = 2 * a * Real.cos A) ∧
  (a = 2) ∧
  (1/2 * b * c * Real.sin A = Real.sqrt 3) →
  (A = π/3) ∧ (a + b + c = 6) := by
sorry


end triangle_problem_l2793_279308


namespace max_rectangle_area_l2793_279316

/-- Given a rectangle with perimeter 160 feet and length twice its width,
    the maximum area that can be enclosed is 12800/9 square feet. -/
theorem max_rectangle_area (w : ℝ) (l : ℝ) (h1 : w > 0) (h2 : l > 0) 
    (h3 : 2 * w + 2 * l = 160) (h4 : l = 2 * w) : w * l ≤ 12800 / 9 := by
  sorry

end max_rectangle_area_l2793_279316


namespace repeating_decimal_to_fraction_l2793_279377

theorem repeating_decimal_to_fraction :
  ∃ (x : ℚ), x = 0.56 ∧ x = 56 / 99 :=
by
  sorry

end repeating_decimal_to_fraction_l2793_279377


namespace shoes_price_calculation_shoes_price_proof_l2793_279373

theorem shoes_price_calculation (initial_price : ℝ) 
  (price_increase_percentage : ℝ) (discount_percentage : ℝ) : ℝ :=
  let thursday_price := initial_price * (1 + price_increase_percentage)
  let friday_price := thursday_price * (1 - discount_percentage)
  friday_price

theorem shoes_price_proof :
  shoes_price_calculation 50 0.15 0.2 = 46 := by
  sorry

end shoes_price_calculation_shoes_price_proof_l2793_279373


namespace second_group_average_l2793_279303

theorem second_group_average (n₁ : ℕ) (n₂ : ℕ) (avg₁ : ℝ) (avg_total : ℝ) :
  n₁ = 30 →
  n₂ = 20 →
  avg₁ = 20 →
  avg_total = 24 →
  ∃ avg₂ : ℝ,
    (n₁ * avg₁ + n₂ * avg₂) / (n₁ + n₂) = avg_total ∧
    avg₂ = 30 := by
  sorry

end second_group_average_l2793_279303


namespace inequality_proof_l2793_279352

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : 1/a + 1/b = 1) :
  ∀ n : ℕ, (a + b)^n - a^n - b^n ≥ 2^(2*n) - 2^(n+1) :=
by sorry

end inequality_proof_l2793_279352


namespace cube_root_simplification_l2793_279334

theorem cube_root_simplification :
  let x : ℝ := 5488000
  let y : ℝ := 2744
  let z : ℝ := 343
  (1000 = 10^3) →
  (y = 2^3 * z) →
  (z = 7^3) →
  x^(1/3) = 140 * 2^(1/3) :=
by sorry

end cube_root_simplification_l2793_279334


namespace strawberry_harvest_l2793_279357

/-- Calculates the total number of strawberries harvested in a rectangular garden --/
theorem strawberry_harvest (length width : ℕ) (plants_per_sqft : ℕ) (strawberries_per_plant : ℕ) :
  length = 10 → width = 12 → plants_per_sqft = 5 → strawberries_per_plant = 8 →
  length * width * plants_per_sqft * strawberries_per_plant = 4800 := by
  sorry

#check strawberry_harvest

end strawberry_harvest_l2793_279357


namespace area_bound_l2793_279391

-- Define the points and circles
variable (A B C D K L M N : Point)
variable (I I_A I_B I_C I_D : Circle)

-- Define the convex quadrilateral ABCD
def is_convex_quadrilateral (A B C D : Point) : Prop := sorry

-- Define the inscribed circle I
def is_inscribed_circle (I : Circle) (A B C D : Point) : Prop := sorry

-- Define tangent points
def is_tangent_point (K L M N : Point) (I : Circle) (A B C D : Point) : Prop := sorry

-- Define incircles of triangles
def is_incircle (I_A I_B I_C I_D : Circle) (A B C D K L M N : Point) : Prop := sorry

-- Define common external tangent lines
def common_external_tangent (I_AB I_BC I_CD I_AD : Line) (I_A I_B I_C I_D : Circle) : Prop := sorry

-- Define the area S of the quadrilateral formed by I_AB, I_BC, I_CD, and I_AD
def area_S (I_AB I_BC I_CD I_AD : Line) : ℝ := sorry

-- Define the radius r of circle I
def radius_r (I : Circle) : ℝ := sorry

-- Theorem statement
theorem area_bound 
  (h1 : is_convex_quadrilateral A B C D)
  (h2 : is_inscribed_circle I A B C D)
  (h3 : is_tangent_point K L M N I A B C D)
  (h4 : is_incircle I_A I_B I_C I_D A B C D K L M N)
  (h5 : common_external_tangent I_AB I_BC I_CD I_AD I_A I_B I_C I_D)
  (S : ℝ)
  (h6 : S = area_S I_AB I_BC I_CD I_AD)
  (r : ℝ)
  (h7 : r = radius_r I) :
  S ≤ (12 - 8 * Real.sqrt 2) * r^2 := by sorry

end area_bound_l2793_279391


namespace arcsin_arccos_eq_arctan_pi_fourth_l2793_279354

theorem arcsin_arccos_eq_arctan_pi_fourth :
  ∃ x : ℝ, x = 0 ∧ Real.arcsin x + Real.arccos (1 - x) = Real.arctan x + π / 4 :=
by sorry

end arcsin_arccos_eq_arctan_pi_fourth_l2793_279354


namespace cost_of_dozen_pens_l2793_279368

/-- The cost of one dozen pens given the ratio of pen to pencil cost and the total cost of 3 pens and 5 pencils -/
theorem cost_of_dozen_pens (pen_cost pencil_cost : ℕ) : 
  pen_cost = 5 * pencil_cost →  -- Condition 1: pen cost is 5 times pencil cost
  3 * pen_cost + 5 * pencil_cost = 240 →  -- Condition 2: total cost of 3 pens and 5 pencils
  12 * pen_cost = 720 :=  -- Conclusion: cost of one dozen pens
by
  sorry  -- Proof is omitted as per instructions

end cost_of_dozen_pens_l2793_279368


namespace perimeter_semicircular_arcs_on_square_l2793_279305

/-- The perimeter of a region bounded by semicircular arcs on a square's sides -/
theorem perimeter_semicircular_arcs_on_square (side_length : Real) :
  side_length = 4 / Real.pi →
  (4 : Real) * (Real.pi * side_length / 2) = 8 := by
  sorry

end perimeter_semicircular_arcs_on_square_l2793_279305


namespace prob_three_red_standard_deck_l2793_279355

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : ℕ)
  (red_cards : ℕ)
  (h_total : total_cards = 52)
  (h_red : red_cards = 26)

/-- The probability of drawing three red cards from a standard deck -/
def prob_three_red (d : Deck) : ℚ :=
  (d.red_cards * (d.red_cards - 1) * (d.red_cards - 2)) / 
  (d.total_cards * (d.total_cards - 1) * (d.total_cards - 2))

/-- Theorem stating the probability of drawing three red cards from a standard deck -/
theorem prob_three_red_standard_deck :
  ∃ (d : Deck), prob_three_red d = 200 / 1701 :=
sorry

end prob_three_red_standard_deck_l2793_279355


namespace factors_of_48_l2793_279310

/-- The number of distinct positive factors of 48 is 10. -/
theorem factors_of_48 : Finset.card (Nat.divisors 48) = 10 := by
  sorry

end factors_of_48_l2793_279310


namespace max_value_of_f_in_interval_l2793_279344

def f (x : ℝ) : ℝ := x^3 - 3*x^2 - 9*x + 5

theorem max_value_of_f_in_interval :
  ∃ (c : ℝ), c ∈ Set.Icc (-4 : ℝ) (4 : ℝ) ∧
  ∀ (x : ℝ), x ∈ Set.Icc (-4 : ℝ) (4 : ℝ) → f x ≤ f c ∧ f c = 10 :=
sorry

end max_value_of_f_in_interval_l2793_279344


namespace sum_of_a_and_b_l2793_279371

-- Define the conversion rate from paise to rupees
def paise_to_rupees (paise : ℚ) : ℚ := paise / 100

-- Define variables a and b
def a : ℚ := sorry
def b : ℚ := sorry

-- State the theorem
theorem sum_of_a_and_b : 
  (0.5 / 100 * a = paise_to_rupees 65) → 
  (1.25 / 100 * b = paise_to_rupees 104) → 
  (a + b = 213.2) := by sorry

end sum_of_a_and_b_l2793_279371


namespace adam_shelf_capacity_l2793_279304

/-- The number of action figures that can fit on each shelf. -/
def figures_per_shelf : ℕ := 9

/-- The number of shelves in Adam's room. -/
def number_of_shelves : ℕ := 3

/-- The total number of action figures that can fit on all shelves. -/
def total_figures : ℕ := figures_per_shelf * number_of_shelves

theorem adam_shelf_capacity :
  total_figures = 27 :=
by sorry

end adam_shelf_capacity_l2793_279304


namespace polynomial_division_quotient_l2793_279320

theorem polynomial_division_quotient (z : ℝ) : 
  4 * z^5 - 3 * z^4 + 2 * z^3 - 5 * z^2 + 7 * z - 3 = 
  (z + 2) * (4 * z^4 - 11 * z^3 + 24 * z^2 - 53 * z + 113) + (-229) := by
  sorry

end polynomial_division_quotient_l2793_279320


namespace system_solution_l2793_279309

theorem system_solution :
  let solutions : List (ℤ × ℤ × ℤ) := [(0, 12, 0), (2, 7, 3), (4, 2, 6)]
  ∀ x y z : ℤ,
    (x + y + z = 12 ∧ 8*x + 5*y + 3*z = 60) ↔ (x, y, z) ∈ solutions :=
by sorry

end system_solution_l2793_279309


namespace simplify_fraction_product_l2793_279379

theorem simplify_fraction_product : 5 * (18 / 7) * (21 / -63) = -30 / 7 := by
  sorry

end simplify_fraction_product_l2793_279379


namespace sum_of_values_l2793_279311

/-- A discrete random variable with two possible values -/
structure DiscreteRV where
  x₁ : ℝ
  x₂ : ℝ
  p₁ : ℝ
  p₂ : ℝ
  h_prob_sum : p₁ + p₂ = 1
  h_prob_nonneg : 0 ≤ p₁ ∧ 0 ≤ p₂

/-- Expected value of the discrete random variable -/
def expectation (X : DiscreteRV) : ℝ := X.x₁ * X.p₁ + X.x₂ * X.p₂

/-- Variance of the discrete random variable -/
def variance (X : DiscreteRV) : ℝ :=
  X.p₁ * (X.x₁ - expectation X)^2 + X.p₂ * (X.x₂ - expectation X)^2

theorem sum_of_values (X : DiscreteRV)
  (h_p₁ : X.p₁ = 2/3)
  (h_p₂ : X.p₂ = 1/3)
  (h_order : X.x₁ < X.x₂)
  (h_expectation : expectation X = 4/9)
  (h_variance : variance X = 2) :
  X.x₁ + X.x₂ = 17/9 := by
  sorry

end sum_of_values_l2793_279311


namespace inscribed_circle_radius_l2793_279363

theorem inscribed_circle_radius (a b c : ℝ) (h1 : a = 6) (h2 : b = 12) (h3 : c = 18) :
  let r := (1 / a + 1 / b + 1 / c + 2 * Real.sqrt (1 / (a * b) + 1 / (a * c) + 1 / (b * c)))⁻¹
  r = 36 / 17 := by
  sorry

end inscribed_circle_radius_l2793_279363


namespace popcorn_soda_cost_l2793_279300

/-- Calculate the total cost of popcorn and soda purchases with discounts and tax --/
theorem popcorn_soda_cost : ∃ (total_cost : ℚ),
  (let popcorn_price : ℚ := 14.7 / 5
   let soda_price : ℚ := 2
   let popcorn_quantity : ℕ := 4
   let soda_quantity : ℕ := 3
   let popcorn_discount : ℚ := 0.1
   let soda_discount : ℚ := 0.05
   let popcorn_tax : ℚ := 0.06
   let soda_tax : ℚ := 0.07

   let popcorn_subtotal : ℚ := popcorn_price * popcorn_quantity
   let soda_subtotal : ℚ := soda_price * soda_quantity

   let popcorn_discounted : ℚ := popcorn_subtotal * (1 - popcorn_discount)
   let soda_discounted : ℚ := soda_subtotal * (1 - soda_discount)

   let popcorn_total : ℚ := popcorn_discounted * (1 + popcorn_tax)
   let soda_total : ℚ := soda_discounted * (1 + soda_tax)

   total_cost = popcorn_total + soda_total) ∧
  (total_cost ≥ 17.31 ∧ total_cost < 17.33) := by
  sorry

#eval (14.7 / 5 * 4 * 0.9 * 1.06 + 2 * 3 * 0.95 * 1.07 : ℚ)

end popcorn_soda_cost_l2793_279300


namespace largest_n_divisible_by_five_l2793_279392

def expression (n : ℕ) : ℤ :=
  8 * (n - 2)^6 - 3 * n^2 + 20 * n - 36

theorem largest_n_divisible_by_five :
  ∀ n : ℕ, n < 100000 →
    (expression n % 5 = 0 → n ≤ 99997) ∧
    (expression 99997 % 5 = 0) ∧
    99997 < 100000 :=
by sorry

end largest_n_divisible_by_five_l2793_279392


namespace no_solutions_for_diophantine_equation_l2793_279301

theorem no_solutions_for_diophantine_equation :
  ¬∃ (m : ℕ+) (p q : ℕ), 
    Nat.Prime p ∧ Nat.Prime q ∧ 2^(m : ℕ) * p^2 + 1 = q^7 := by
  sorry

end no_solutions_for_diophantine_equation_l2793_279301


namespace geometric_sum_eight_terms_l2793_279376

theorem geometric_sum_eight_terms :
  let a₀ : ℚ := 2/3
  let r : ℚ := 1/3
  let n : ℕ := 8
  let S := a₀ * (1 - r^n) / (1 - r)
  S = 6560/6561 := by sorry

end geometric_sum_eight_terms_l2793_279376


namespace losing_ticket_probability_l2793_279382

/-- Given the odds of drawing a winning ticket are 5:8, 
    the probability of drawing a losing ticket is 8/13 -/
theorem losing_ticket_probability (winning_odds : Rat) 
  (h : winning_odds = 5 / 8) : 
  (1 : Rat) - winning_odds * (13 : Rat) / ((5 : Rat) + (8 : Rat)) = 8 / 13 :=
sorry

end losing_ticket_probability_l2793_279382


namespace decimal_multiplication_l2793_279319

theorem decimal_multiplication (a b c : ℚ) (h1 : a = 0.025) (h2 : b = 3.84) (h3 : c = 0.096) 
  (h4 : (25 : ℕ) * 384 = 9600) : a * b = c := by
  sorry

end decimal_multiplication_l2793_279319


namespace exists_special_set_l2793_279328

/-- A function that checks if a natural number is a perfect power -/
def isPerfectPower (n : ℕ) : Prop :=
  ∃ (b k : ℕ), k > 1 ∧ n = b^k

/-- The existence of a set of 1992 positive integers with the required property -/
theorem exists_special_set : ∃ (S : Finset ℕ), 
  (S.card = 1992) ∧ 
  (∀ (T : Finset ℕ), T ⊆ S → isPerfectPower (T.sum id)) :=
sorry

end exists_special_set_l2793_279328


namespace arithmetic_sequence_properties_l2793_279367

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- General term of the sequence
  S : ℕ → ℝ  -- Sum of the first n terms
  b : ℕ → ℝ  -- Related sequence
  h1 : a 3 = 10
  h2 : S 6 = 72
  h3 : ∀ n, b n = (1/2) * a n - 30

/-- The minimum value of the sum of the first n terms of b_n -/
def T_min (seq : ArithmeticSequence) : ℝ :=
  Finset.sum (Finset.range 15) (λ i => seq.b (i + 1))

/-- Main theorem about the arithmetic sequence and its properties -/
theorem arithmetic_sequence_properties (seq : ArithmeticSequence) :
  (∀ n, seq.a n = 4 * n - 2) ∧ T_min seq = -225 := by
  sorry


end arithmetic_sequence_properties_l2793_279367


namespace expression_evaluation_l2793_279397

theorem expression_evaluation :
  let x : ℤ := 25
  let y : ℤ := 30
  let z : ℤ := 7
  (x - (y - z)) - ((x - y) - z) = 14 := by sorry

end expression_evaluation_l2793_279397


namespace pool_filling_time_l2793_279321

theorem pool_filling_time (R : ℝ) (h1 : R > 0) : 
  (R + 1.5 * R) * 5 = 1 → R * 12.5 = 1 := by
  sorry

end pool_filling_time_l2793_279321


namespace sum_of_roots_cubic_l2793_279389

theorem sum_of_roots_cubic (x : ℝ) : 
  let f : ℝ → ℝ := λ x => 6*x^3 - 7*x^2 + 2*x
  (∃ a b c : ℝ, f x = (x - a) * (x - b) * (x - c)) →
  a + b + c = 7/6 :=
by sorry

end sum_of_roots_cubic_l2793_279389


namespace four_black_faces_symmetry_l2793_279326

/-- Represents the symmetry types of a cube. -/
inductive CubeSymmetryType
  | A
  | B1
  | B2
  | C

/-- Represents a cube with some faces painted black. -/
structure PaintedCube where
  blackFaces : Finset (Fin 6)
  blackFaceCount : blackFaces.card = 4

/-- Returns the symmetry type of a painted cube. -/
def symmetryType (cube : PaintedCube) : CubeSymmetryType :=
  sorry

/-- Theorem stating that a cube with four black faces has a symmetry type equivalent to B1 or B2. -/
theorem four_black_faces_symmetry (cube : PaintedCube) :
  symmetryType cube = CubeSymmetryType.B1 ∨ symmetryType cube = CubeSymmetryType.B2 :=
sorry

end four_black_faces_symmetry_l2793_279326


namespace lcm_gcd_12_15_l2793_279362

theorem lcm_gcd_12_15 :
  (Nat.lcm 12 15 * Nat.gcd 12 15 = 180) ∧
  (Nat.lcm 12 15 + Nat.gcd 12 15 = 63) := by
  sorry

end lcm_gcd_12_15_l2793_279362


namespace complement_A_intersect_B_l2793_279360

-- Define the sets A and B
def A : Set ℝ := {x | 2 ≤ x ∧ x < 5}
def B : Set ℝ := {x | 3 * x - 7 ≥ 8 - 2 * x}

-- State the theorem
theorem complement_A_intersect_B :
  (Aᶜ ∩ B) = {x : ℝ | x ≥ 5} := by sorry

end complement_A_intersect_B_l2793_279360


namespace triangle_max_area_l2793_279375

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that the maximum area of the triangle is 2 + √3 when b²-2√3bc*sin(A)+c²=4 and a=2 -/
theorem triangle_max_area (a b c : ℝ) (A B C : ℝ) :
  a = 2 →
  b^2 - 2 * Real.sqrt 3 * b * c * Real.sin A + c^2 = 4 →
  0 < A ∧ A < Real.pi →
  0 < B ∧ B < Real.pi →
  0 < C ∧ C < Real.pi →
  A + B + C = Real.pi →
  (∃ (S : ℝ), S = (1/2) * b * c * Real.sin A ∧
    ∀ (S' : ℝ), S' = (1/2) * b * c * Real.sin A → S' ≤ 2 + Real.sqrt 3) :=
by sorry

end triangle_max_area_l2793_279375


namespace select_five_from_eight_l2793_279332

theorem select_five_from_eight : Nat.choose 8 5 = 56 := by
  sorry

end select_five_from_eight_l2793_279332


namespace correct_seating_arrangements_l2793_279338

/-- The number of ways to arrange n distinct objects -/
def factorial (n : ℕ) : ℕ := (List.range n).foldr (· * ·) 1

/-- The number of people to be seated -/
def total_people : ℕ := 10

/-- The number of people with seating restrictions -/
def restricted_people : ℕ := 4

/-- The number of ways to arrange 10 people in a row, where 4 specific people cannot sit in 4 consecutive seats -/
def seating_arrangements : ℕ := 
  factorial total_people - factorial (total_people - restricted_people + 1) * factorial restricted_people

theorem correct_seating_arrangements : seating_arrangements = 3507840 := by
  sorry

end correct_seating_arrangements_l2793_279338


namespace consecutive_product_theorem_l2793_279346

theorem consecutive_product_theorem (n : ℕ) : 
  (∃ m : ℕ, 9*n^2 + 5*n - 26 = m * (m + 1)) → n = 2 := by
  sorry

end consecutive_product_theorem_l2793_279346


namespace fraction_integer_condition_l2793_279330

theorem fraction_integer_condition (p : ℕ+) :
  (↑p : ℚ) ∈ ({3, 5, 9, 35} : Set ℚ) ↔ ∃ (k : ℤ), k > 0 ∧ (3 * p + 25 : ℚ) / (2 * p - 5 : ℚ) = k := by
  sorry

end fraction_integer_condition_l2793_279330


namespace different_color_probability_l2793_279359

def total_chips : ℕ := 7 + 5
def red_chips : ℕ := 7
def green_chips : ℕ := 5

theorem different_color_probability :
  (red_chips * green_chips : ℚ) / (total_chips * (total_chips - 1) / 2) = 35 / 66 := by
  sorry

end different_color_probability_l2793_279359


namespace switcheroo_period_l2793_279340

/-- Represents a word of length 2^n -/
def Word (n : ℕ) := Fin (2^n) → Char

/-- Performs a single switcheroo operation on a word -/
def switcheroo (n : ℕ) (w : Word n) : Word n :=
  sorry

/-- Returns true if two words are equal -/
def word_eq (n : ℕ) (w1 w2 : Word n) : Prop :=
  ∀ i, w1 i = w2 i

/-- Applies the switcheroo operation m times -/
def apply_switcheroo (n m : ℕ) (w : Word n) : Word n :=
  sorry

theorem switcheroo_period (n : ℕ) :
  ∀ w : Word n, word_eq n (apply_switcheroo n (2^n) w) w ∧
  ∀ m : ℕ, m < 2^n → ¬(word_eq n (apply_switcheroo n m w) w) :=
by sorry

end switcheroo_period_l2793_279340


namespace restaurant_bill_proof_l2793_279399

/-- The number of friends in the group -/
def total_friends : ℕ := 10

/-- The number of friends who paid -/
def paying_friends : ℕ := 9

/-- The extra amount each paying friend contributed -/
def extra_payment : ℚ := 3

/-- The total bill at the restaurant -/
def total_bill : ℚ := 270

/-- Theorem stating that the given scenario results in the correct total bill -/
theorem restaurant_bill_proof :
  (paying_friends : ℚ) * (total_bill / total_friends + extra_payment) = total_bill :=
by sorry

end restaurant_bill_proof_l2793_279399


namespace skating_minutes_on_eleventh_day_l2793_279370

def minutes_per_day_first_period : ℕ := 80
def days_first_period : ℕ := 6
def minutes_per_day_second_period : ℕ := 105
def days_second_period : ℕ := 4
def target_average : ℕ := 95
def total_days : ℕ := 11

theorem skating_minutes_on_eleventh_day :
  (minutes_per_day_first_period * days_first_period +
   minutes_per_day_second_period * days_second_period +
   145) / total_days = target_average :=
by sorry

end skating_minutes_on_eleventh_day_l2793_279370


namespace f_is_quadratic_l2793_279380

-- Define what a quadratic equation is
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

-- Define the function representing the equation x^2 = x + 1
def f (x : ℝ) : ℝ := x^2 - x - 1

-- Theorem stating that f is a quadratic equation
theorem f_is_quadratic : is_quadratic_equation f :=
  sorry

end f_is_quadratic_l2793_279380


namespace difference_of_squares_l2793_279385

theorem difference_of_squares : 535^2 - 465^2 = 70000 := by
  sorry

end difference_of_squares_l2793_279385


namespace smallest_odd_four_prime_factors_l2793_279348

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(m ∣ n)

def has_exactly_four_prime_factors (n : ℕ) : Prop :=
  ∃ (p q r s : ℕ), is_prime p ∧ is_prime q ∧ is_prime r ∧ is_prime s ∧
    p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s ∧
    n = p * q * r * s

theorem smallest_odd_four_prime_factors :
  (1155 % 2 = 1) ∧
  has_exactly_four_prime_factors 1155 ∧
  ∀ n : ℕ, n < 1155 → (n % 2 = 1 → ¬has_exactly_four_prime_factors n) :=
by sorry

end smallest_odd_four_prime_factors_l2793_279348


namespace attendant_claimed_two_shirts_l2793_279369

-- Define the given conditions
def trousers : ℕ := 10
def total_bill : ℕ := 140
def shirt_cost : ℕ := 5
def trouser_cost : ℕ := 9
def missing_shirts : ℕ := 8

-- Define the function to calculate the number of shirts the attendant initially claimed
def attendant_claim : ℕ :=
  let trouser_total : ℕ := trousers * trouser_cost
  let shirt_total : ℕ := total_bill - trouser_total
  let actual_shirts : ℕ := shirt_total / shirt_cost
  actual_shirts - missing_shirts

-- Theorem statement
theorem attendant_claimed_two_shirts :
  attendant_claim = 2 := by sorry

end attendant_claimed_two_shirts_l2793_279369


namespace misha_dog_savings_l2793_279387

theorem misha_dog_savings (current_amount target_amount : ℕ) 
  (h1 : current_amount = 34)
  (h2 : target_amount = 47) :
  target_amount - current_amount = 13 := by
sorry

end misha_dog_savings_l2793_279387


namespace rubber_band_difference_l2793_279318

theorem rubber_band_difference (total : ℕ) (aira_initial : ℕ) (samantha_extra : ℕ) (equal_share : ℕ)
  (h1 : total = 18)
  (h2 : aira_initial = 4)
  (h3 : samantha_extra = 5)
  (h4 : equal_share = 6) :
  let samantha_initial := aira_initial + samantha_extra
  let joe_initial := total - samantha_initial - aira_initial
  joe_initial - aira_initial = 1 := by sorry

end rubber_band_difference_l2793_279318


namespace unique_non_negative_one_result_l2793_279345

theorem unique_non_negative_one_result :
  (-1 * 1 = -1) ∧
  ((-1) / (-1) ≠ -1) ∧
  (-2015 / 2015 = -1) ∧
  ((-1)^9 * (-1)^2 = -1) :=
by sorry

end unique_non_negative_one_result_l2793_279345


namespace x_squared_plus_inverse_x_squared_l2793_279313

theorem x_squared_plus_inverse_x_squared (x : ℝ) (h : x - 1/x = 3) : x^2 + 1/x^2 = 11 := by
  sorry

end x_squared_plus_inverse_x_squared_l2793_279313


namespace ellipse_foci_distance_sum_l2793_279381

-- Define the ellipse
def ellipse (x y : ℝ) (m : ℝ) : Prop :=
  x^2 / 4 + y^2 / m = 1

-- Define the property that the ellipse passes through (0, 4)
def passes_through_B (m : ℝ) : Prop :=
  ellipse 0 4 m

-- Define the sum of distances from any point to the foci
def sum_distances_to_foci (m : ℝ) : ℝ := 8

-- Theorem statement
theorem ellipse_foci_distance_sum (m : ℝ) 
  (h : passes_through_B m) : 
  sum_distances_to_foci m = 8 := by sorry

end ellipse_foci_distance_sum_l2793_279381


namespace smith_family_seating_arrangement_l2793_279347

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem smith_family_seating_arrangement :
  let total_arrangements := factorial 7
  let no_adjacent_boys := factorial 4 * factorial 3
  total_arrangements - no_adjacent_boys = 4896 :=
by sorry

end smith_family_seating_arrangement_l2793_279347


namespace arithmetic_sequence_length_l2793_279372

theorem arithmetic_sequence_length : ∀ (a₁ d : ℤ) (n : ℕ),
  a₁ = 165 ∧ d = -6 ∧ (a₁ + d * (n - 1 : ℤ) ≤ 24) ∧ (a₁ + d * ((n - 1) - 1 : ℤ) > 24) →
  n = 24 := by
sorry

end arithmetic_sequence_length_l2793_279372


namespace trig_identity_l2793_279307

theorem trig_identity (α : ℝ) : 
  (3 - 4 * Real.cos (2 * α) + Real.cos (4 * α)) / 
  (3 + 4 * Real.cos (2 * α) + Real.cos (4 * α)) = 
  (Real.tan α) ^ 4 / 3.396 := by sorry

end trig_identity_l2793_279307


namespace sum_of_digits_for_special_triangle_l2793_279335

/-- Given a positive integer n, returns the sum of its digits -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- The sum of the first n natural numbers -/
def triangle_sum (n : ℕ) : ℕ := n * (n + 1) / 2

theorem sum_of_digits_for_special_triangle : 
  ∃ (N : ℕ), (triangle_sum N = 2145) ∧ (sum_of_digits N = 11) :=
sorry

end sum_of_digits_for_special_triangle_l2793_279335


namespace fish_filets_count_l2793_279312

/-- The number of fish filets Ben and his family will have -/
def fish_filets : ℕ :=
  let ben_fish := 4
  let judy_fish := 1
  let billy_fish := 3
  let jim_fish := 2
  let susie_fish := 5
  let total_caught := ben_fish + judy_fish + billy_fish + jim_fish + susie_fish
  let thrown_back := 3
  let kept_fish := total_caught - thrown_back
  let filets_per_fish := 2
  kept_fish * filets_per_fish

theorem fish_filets_count : fish_filets = 24 := by
  sorry

end fish_filets_count_l2793_279312


namespace prob_a_wins_l2793_279350

/-- Given a chess game between players A and B, this theorem proves
    the probability of player A winning, given the probabilities of
    a draw and A not losing. -/
theorem prob_a_wins (prob_draw prob_a_not_lose : ℚ)
  (h_draw : prob_draw = 1/2)
  (h_not_lose : prob_a_not_lose = 5/6) :
  prob_a_not_lose - prob_draw = 1/3 :=
by sorry

end prob_a_wins_l2793_279350


namespace candy_distribution_l2793_279383

/-- The number of pieces of candy in each of Wendy's boxes -/
def candy_per_box : ℕ := sorry

/-- The number of pieces of candy Wendy's brother has -/
def brother_candy : ℕ := 6

/-- The number of boxes Wendy has -/
def wendy_boxes : ℕ := 2

/-- The total number of pieces of candy -/
def total_candy : ℕ := 12

theorem candy_distribution :
  candy_per_box * wendy_boxes + brother_candy = total_candy ∧ candy_per_box = 3 :=
by sorry

end candy_distribution_l2793_279383


namespace no_intersection_l2793_279398

/-- The number of distinct points of intersection between two ellipses -/
def intersectionPoints (f g : ℝ → ℝ → Prop) : ℕ :=
  sorry

/-- First ellipse: 3x^2 + 2y^2 = 4 -/
def ellipse1 (x y : ℝ) : Prop :=
  3 * x^2 + 2 * y^2 = 4

/-- Second ellipse: 6x^2 + 3y^2 = 9 -/
def ellipse2 (x y : ℝ) : Prop :=
  6 * x^2 + 3 * y^2 = 9

/-- Theorem: The number of distinct points of intersection between the two given ellipses is 0 -/
theorem no_intersection : intersectionPoints ellipse1 ellipse2 = 0 :=
  sorry

end no_intersection_l2793_279398


namespace largest_product_sum_of_digits_l2793_279353

def is_prime (p : ℕ) : Prop := sorry

def sum_of_digits (n : ℕ) : ℕ := sorry

theorem largest_product_sum_of_digits :
  ∃ (n d e : ℕ),
    is_prime d ∧ is_prime e ∧ is_prime (10 * e + d) ∧
    d ∈ ({5, 7} : Set ℕ) ∧ e ∈ ({3, 7} : Set ℕ) ∧
    n = d * e * (10 * e + d) ∧
    (∀ (m d' e' : ℕ),
      is_prime d' ∧ is_prime e' ∧ is_prime (10 * e' + d') ∧
      d' ∈ ({5, 7} : Set ℕ) ∧ e' ∈ ({3, 7} : Set ℕ) ∧
      m = d' * e' * (10 * e' + d') →
      m ≤ n) ∧
    sum_of_digits n = 21 :=
by sorry

end largest_product_sum_of_digits_l2793_279353


namespace sum_of_powers_l2793_279314

theorem sum_of_powers (ω : ℂ) (h1 : ω^9 = 1) (h2 : ω ≠ 1) :
  ω^20 + ω^24 + ω^28 + ω^32 + ω^36 + ω^40 + ω^44 + ω^48 + ω^52 + ω^56 + ω^60 + ω^64 + ω^68 = (ω^2 - 1) / (ω^4 - 1) := by
  sorry

end sum_of_powers_l2793_279314


namespace line_transformation_theorem_l2793_279329

/-- Given a line with equation y = mx + b, returns a new line with half the slope and twice the y-intercept -/
def transform_line (m b : ℚ) : ℚ × ℚ := (m / 2, 2 * b)

theorem line_transformation_theorem :
  let original_line := ((2 : ℚ) / 3, 4)
  let transformed_line := transform_line original_line.1 original_line.2
  transformed_line = ((1 : ℚ) / 3, 8) := by sorry

end line_transformation_theorem_l2793_279329


namespace divisors_of_2_pow_n_minus_1_l2793_279378

theorem divisors_of_2_pow_n_minus_1 (n : ℕ) (d : ℕ) (h1 : Odd n) (h2 : d > 0) (h3 : d ∣ (2^n - 1)) :
  d % 8 = 1 ∨ d % 8 = 7 :=
sorry

end divisors_of_2_pow_n_minus_1_l2793_279378


namespace seventh_term_of_geometric_sequence_l2793_279306

def geometric_sequence (a : ℕ → ℝ) := ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n

theorem seventh_term_of_geometric_sequence 
  (a : ℕ → ℝ) 
  (h_geom : geometric_sequence a) 
  (h_sum1 : a 1 + a 2 = 3) 
  (h_sum2 : a 2 + a 3 = 6) : 
  a 7 = 64 := by
sorry

end seventh_term_of_geometric_sequence_l2793_279306


namespace expression_evaluation_l2793_279393

theorem expression_evaluation : 2 - 3 * (-4) + 5 - (-6) * 7 = 61 := by
  sorry

end expression_evaluation_l2793_279393


namespace construction_cost_difference_equals_profit_l2793_279333

/-- Represents the construction and sale details of houses in an area --/
structure HouseData where
  other_sale_price : ℕ
  certain_sale_multiplier : ℚ
  profit : ℕ

/-- Calculates the difference in construction cost between a certain house and other houses --/
def construction_cost_difference (data : HouseData) : ℕ :=
  data.profit

theorem construction_cost_difference_equals_profit (data : HouseData)
  (h1 : data.other_sale_price = 320000)
  (h2 : data.certain_sale_multiplier = 3/2)
  (h3 : data.profit = 60000) :
  construction_cost_difference data = data.profit := by
  sorry

#eval construction_cost_difference { other_sale_price := 320000, certain_sale_multiplier := 3/2, profit := 60000 }

end construction_cost_difference_equals_profit_l2793_279333
