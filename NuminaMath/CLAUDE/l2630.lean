import Mathlib

namespace sum_of_cubes_negative_l2630_263051

theorem sum_of_cubes_negative : 
  (Real.sqrt 2021 - Real.sqrt 2020)^3 + 
  (Real.sqrt 2020 - Real.sqrt 2019)^3 + 
  (Real.sqrt 2019 - Real.sqrt 2018)^3 < 0 := by
  sorry

end sum_of_cubes_negative_l2630_263051


namespace equation_has_six_roots_l2630_263036

noncomputable def f (x : ℝ) : ℝ := (x^2 - x + 1)^3 / (x^2 * (x - 1)^2)

def is_root (x : ℝ) : Prop := f x = f Real.pi

theorem equation_has_six_roots :
  ∃ (r1 r2 r3 r4 r5 r6 : ℝ),
    r1 ≠ r2 ∧ r1 ≠ r3 ∧ r1 ≠ r4 ∧ r1 ≠ r5 ∧ r1 ≠ r6 ∧
    r2 ≠ r3 ∧ r2 ≠ r4 ∧ r2 ≠ r5 ∧ r2 ≠ r6 ∧
    r3 ≠ r4 ∧ r3 ≠ r5 ∧ r3 ≠ r6 ∧
    r4 ≠ r5 ∧ r4 ≠ r6 ∧
    r5 ≠ r6 ∧
    is_root r1 ∧ is_root r2 ∧ is_root r3 ∧ is_root r4 ∧ is_root r5 ∧ is_root r6 ∧
    ∀ x : ℝ, is_root x → (x = r1 ∨ x = r2 ∨ x = r3 ∨ x = r4 ∨ x = r5 ∨ x = r6) :=
by
  sorry

end equation_has_six_roots_l2630_263036


namespace specific_building_height_l2630_263057

/-- Calculates the height of a building with specific floor heights -/
def building_height (total_stories : ℕ) (base_height : ℕ) (height_increase : ℕ) : ℕ :=
  let first_half := total_stories / 2
  let second_half := total_stories - first_half
  (first_half * base_height) + (second_half * (base_height + height_increase))

/-- Theorem stating the height of the specific building described in the problem -/
theorem specific_building_height :
  building_height 20 12 3 = 270 := by
  sorry

end specific_building_height_l2630_263057


namespace jack_waiting_time_l2630_263094

/-- The total waiting time for Jack's trip to Canada -/
def total_waiting_time (customs_hours : ℕ) (quarantine_days : ℕ) (hours_per_day : ℕ) : ℕ :=
  customs_hours + quarantine_days * hours_per_day

/-- Theorem stating that Jack's total waiting time is 356 hours -/
theorem jack_waiting_time :
  total_waiting_time 20 14 24 = 356 := by
  sorry

end jack_waiting_time_l2630_263094


namespace initial_amount_proof_l2630_263022

/-- 
Theorem: If an amount increases by 1/8th of itself each year for two years 
and results in 81000, then the initial amount was 64000.
-/
theorem initial_amount_proof (P : ℚ) : 
  (P * (1 + 1/8) * (1 + 1/8) = 81000) → P = 64000 := by
  sorry

end initial_amount_proof_l2630_263022


namespace adam_lawn_mowing_earnings_l2630_263013

/-- Adam's lawn mowing earnings problem -/
theorem adam_lawn_mowing_earnings 
  (dollars_per_lawn : ℕ) 
  (total_lawns : ℕ) 
  (forgotten_lawns : ℕ) 
  (h1 : dollars_per_lawn = 9)
  (h2 : total_lawns = 12)
  (h3 : forgotten_lawns = 8)
  : (total_lawns - forgotten_lawns) * dollars_per_lawn = 36 := by
  sorry

end adam_lawn_mowing_earnings_l2630_263013


namespace fiftieth_islander_is_knight_l2630_263096

/-- Represents the type of an islander: either a knight or a liar -/
inductive IslanderType
  | Knight
  | Liar

/-- Represents the statement made by an islander about their right neighbor -/
inductive Statement
  | Knight
  | Liar

/-- The number of islanders around the table -/
def n : ℕ := 50

/-- Function that returns the statement made by the islander at a given position -/
def statement (pos : ℕ) : Statement :=
  if pos % 2 = 1 then Statement.Knight else Statement.Liar

/-- Function that determines the actual type of the islander at a given position -/
def islanderType (firstType : IslanderType) (pos : ℕ) : IslanderType :=
  sorry

/-- Theorem stating that the 50th islander must be a knight -/
theorem fiftieth_islander_is_knight (firstType : IslanderType) :
  islanderType firstType n = IslanderType.Knight :=
  sorry

end fiftieth_islander_is_knight_l2630_263096


namespace probability_at_least_one_boy_l2630_263083

def total_students : ℕ := 5
def total_girls : ℕ := 3
def representatives : ℕ := 2

theorem probability_at_least_one_boy :
  let total_selections := Nat.choose total_students representatives
  let all_girl_selections := Nat.choose total_girls representatives
  (1 : ℚ) - (all_girl_selections : ℚ) / (total_selections : ℚ) = 7/10 := by
  sorry

end probability_at_least_one_boy_l2630_263083


namespace cubic_function_properties_l2630_263088

/-- A cubic function with specific properties -/
structure CubicFunction where
  b : ℝ
  c : ℝ
  d : ℝ
  f : ℝ → ℝ
  f_def : ∀ x, f x = x^3 + 3*b*x^2 + c*x + d
  increasing_neg : ∀ x y, x < y → y < 0 → f x < f y
  decreasing_pos : ∀ x y, 0 < x → x < y → y < 2 → f y < f x
  root_neg_b : f (-b) = 0

/-- Main theorem about the cubic function -/
theorem cubic_function_properties (cf : CubicFunction) :
  cf.c = 0 ∧
  (∃ x₁ x₂ : ℝ, x₁ ≠ -cf.b ∧ x₂ ≠ -cf.b ∧ cf.f x₁ = 0 ∧ cf.f x₂ = 0 ∧ x₂ - (-cf.b) = (-cf.b) - x₁) ∧
  (0 ≤ cf.f 1 ∧ cf.f 1 < 11) := by
  sorry

end cubic_function_properties_l2630_263088


namespace line_point_k_value_l2630_263060

/-- Given a line containing points (3,5), (-1,k), and (-7,2), prove that k = 3.8 -/
theorem line_point_k_value (k : ℝ) : 
  (∃ (line : ℝ → ℝ), line 3 = 5 ∧ line (-1) = k ∧ line (-7) = 2) → k = 3.8 := by
  sorry

end line_point_k_value_l2630_263060


namespace hyperbola_eccentricity_l2630_263041

/-- The eccentricity of a hyperbola with equation x^2 - y^2 = 1 is √2 -/
theorem hyperbola_eccentricity : ∃ e : ℝ, e = Real.sqrt 2 ∧ 
  ∀ x y : ℝ, x^2 - y^2 = 1 → 
    ∃ a b c : ℝ, a > 0 ∧ b > 0 ∧ 
      x^2 / a^2 - y^2 / b^2 = 1 ∧ 
      c^2 = a^2 + b^2 ∧ 
      e = c / a := by
  sorry

end hyperbola_eccentricity_l2630_263041


namespace magazine_subscription_issues_l2630_263042

/-- Proves that the number of issues in an 18-month magazine subscription is 36,
    given the normal price, promotional discount per issue, and total promotional discount. -/
theorem magazine_subscription_issues
  (normal_price : ℝ)
  (subscription_duration : ℝ)
  (discount_per_issue : ℝ)
  (total_discount : ℝ)
  (h1 : normal_price = 34)
  (h2 : subscription_duration = 18)
  (h3 : discount_per_issue = 0.25)
  (h4 : total_discount = 9) :
  (total_discount / discount_per_issue : ℝ) = 36 := by
sorry

end magazine_subscription_issues_l2630_263042


namespace booboo_arrangements_l2630_263079

def word_arrangements (n : ℕ) (r₁ : ℕ) (r₂ : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial r₁ * Nat.factorial r₂)

theorem booboo_arrangements :
  word_arrangements 6 2 4 = 15 := by
sorry

end booboo_arrangements_l2630_263079


namespace perpendicular_vectors_l2630_263084

/-- Given two vectors AB and CD in R², where AB is perpendicular to CD,
    prove that the y-coordinate of AB is 1. -/
theorem perpendicular_vectors (x : ℝ) : 
  let AB : ℝ × ℝ := (3, x)
  let CD : ℝ × ℝ := (-2, 6)
  (AB.1 * CD.1 + AB.2 * CD.2 = 0) → x = 1 := by
sorry

end perpendicular_vectors_l2630_263084


namespace new_average_score_is_correct_l2630_263020

/-- Represents the grace mark criteria for different score ranges -/
inductive GraceMarkCriteria where
  | below30 : GraceMarkCriteria
  | between30and40 : GraceMarkCriteria
  | above40 : GraceMarkCriteria

/-- Returns the grace marks for a given criteria -/
def graceMarks (c : GraceMarkCriteria) : ℕ :=
  match c with
  | GraceMarkCriteria.below30 => 5
  | GraceMarkCriteria.between30and40 => 3
  | GraceMarkCriteria.above40 => 1

/-- Calculates the new average score after applying grace marks -/
def newAverageScore (
  classSize : ℕ
  ) (initialAverage : ℚ
  ) (studentsPerRange : ℕ
  ) : ℚ :=
  let initialTotal := classSize * initialAverage
  let totalGraceMarks := 
    studentsPerRange * (graceMarks GraceMarkCriteria.below30 + 
                        graceMarks GraceMarkCriteria.between30and40 + 
                        graceMarks GraceMarkCriteria.above40)
  (initialTotal + totalGraceMarks) / classSize

/-- Theorem stating that the new average score is approximately 39.57 -/
theorem new_average_score_is_correct :
  let classSize := 35
  let initialAverage := 37
  let studentsPerRange := 10
  abs (newAverageScore classSize initialAverage studentsPerRange - 39.57) < 0.01 := by
  sorry

end new_average_score_is_correct_l2630_263020


namespace students_making_stars_l2630_263054

theorem students_making_stars (stars_per_student : ℕ) (total_stars : ℕ) (h1 : stars_per_student = 3) (h2 : total_stars = 372) :
  total_stars / stars_per_student = 124 := by
  sorry

end students_making_stars_l2630_263054


namespace big_suv_to_normal_car_ratio_l2630_263048

/-- Represents the time in minutes for each task when washing a normal car -/
structure NormalCarWashTime where
  windows : Nat
  body : Nat
  tires : Nat
  waxing : Nat

/-- Calculates the total time to wash a normal car -/
def normalCarTotalTime (t : NormalCarWashTime) : Nat :=
  t.windows + t.body + t.tires + t.waxing

/-- Represents the washing scenario -/
structure CarWashScenario where
  normalCarTime : NormalCarWashTime
  normalCarCount : Nat
  totalTime : Nat

/-- Theorem: The ratio of time taken to wash the big SUV to the time taken to wash a normal car is 2:1 -/
theorem big_suv_to_normal_car_ratio 
  (scenario : CarWashScenario) 
  (h1 : scenario.normalCarTime = ⟨4, 7, 4, 9⟩) 
  (h2 : scenario.normalCarCount = 2) 
  (h3 : scenario.totalTime = 96) : 
  (scenario.totalTime - scenario.normalCarCount * normalCarTotalTime scenario.normalCarTime) / 
  (normalCarTotalTime scenario.normalCarTime) = 2 := by
  sorry


end big_suv_to_normal_car_ratio_l2630_263048


namespace f_has_two_real_roots_l2630_263081

/-- The quadratic function we're analyzing -/
def f (x : ℝ) : ℝ := 2 * x^2 - 4 * x - 3

/-- Theorem stating that f has exactly two real roots -/
theorem f_has_two_real_roots : ∃! (s : Finset ℝ), s.card = 2 ∧ ∀ x ∈ s, f x = 0 := by
  sorry

end f_has_two_real_roots_l2630_263081


namespace sum_in_base_8_l2630_263076

/-- Converts a decimal number to its octal (base 8) representation -/
def toOctal (n : ℕ) : List ℕ :=
  sorry

/-- Converts an octal (base 8) representation to its decimal value -/
def fromOctal (l : List ℕ) : ℕ :=
  sorry

/-- Adds two numbers in their octal representations -/
def octalAdd (a b : List ℕ) : List ℕ :=
  sorry

theorem sum_in_base_8 :
  let a := 624
  let b := 112
  let expected_sum := [1, 3, 4, 0]
  octalAdd (toOctal a) (toOctal b) = expected_sum ∧
  fromOctal expected_sum = a + b :=
by sorry

end sum_in_base_8_l2630_263076


namespace third_term_of_geometric_sequence_l2630_263055

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem third_term_of_geometric_sequence
  (a : ℕ → ℝ)
  (h_geom : geometric_sequence a)
  (h_a1 : a 1 = 1)
  (h_a4 : a 4 = 8) :
  a 3 = 4 := by
sorry

end third_term_of_geometric_sequence_l2630_263055


namespace smallest_gcd_multiple_l2630_263024

theorem smallest_gcd_multiple (m n : ℕ+) (h : Nat.gcd m n = 7) :
  ∃ (k : ℕ+), k = Nat.gcd (8 * m) (6 * n) ∧ ∀ (l : ℕ+), l = Nat.gcd (8 * m) (6 * n) → k ≤ l :=
by
  sorry

end smallest_gcd_multiple_l2630_263024


namespace arithmetic_sequence_log_theorem_l2630_263006

-- Define the logarithm base 2
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 2

-- Define the arithmetic sequence condition
def is_arithmetic_sequence (a b c : ℝ) : Prop := b - a = c - b

-- State the theorem
theorem arithmetic_sequence_log_theorem (x : ℝ) :
  is_arithmetic_sequence (lg 2) (lg (2^x - 1)) (lg (2^x + 3)) →
  x = Real.log 5 / Real.log 2 :=
by sorry

end arithmetic_sequence_log_theorem_l2630_263006


namespace rectangle_area_l2630_263097

theorem rectangle_area (w : ℝ) (h₁ : w > 0) (h₂ : 10 * w = 200) : w * (4 * w) = 1600 := by
  sorry

end rectangle_area_l2630_263097


namespace fraction_problem_l2630_263043

theorem fraction_problem : ∃ x : ℚ, x < 20 / 100 * 180 ∧ x * 180 = 24 := by
  use 2 / 15
  sorry

end fraction_problem_l2630_263043


namespace quadratic_inequality_range_l2630_263008

theorem quadratic_inequality_range (m : ℝ) : 
  (¬ (1^2 + 2*1 - m > 0)) ∧ (2^2 + 2*2 - m > 0) → 3 ≤ m ∧ m < 8 := by
  sorry

end quadratic_inequality_range_l2630_263008


namespace greatest_integer_y_l2630_263071

theorem greatest_integer_y (y : ℕ+) : (y.val : ℝ)^4 / (y.val : ℝ)^2 < 18 ↔ y.val ≤ 4 :=
sorry

end greatest_integer_y_l2630_263071


namespace scientific_notation_of_116_million_l2630_263063

theorem scientific_notation_of_116_million :
  (116000000 : ℝ) = 1.16 * (10 ^ 8) := by sorry

end scientific_notation_of_116_million_l2630_263063


namespace number_of_students_in_section_B_l2630_263017

theorem number_of_students_in_section_B (students_A : ℕ) (avg_weight_A : ℚ) (avg_weight_B : ℚ) (avg_weight_total : ℚ) :
  students_A = 26 →
  avg_weight_A = 50 →
  avg_weight_B = 30 →
  avg_weight_total = 38.67 →
  ∃ (students_B : ℕ), 
    (students_A * avg_weight_A + students_B * avg_weight_B : ℚ) / (students_A + students_B : ℚ) = avg_weight_total ∧
    students_B = 34 :=
by sorry

end number_of_students_in_section_B_l2630_263017


namespace natural_number_triples_l2630_263093

theorem natural_number_triples (a b c : ℕ) :
  (∃ m n p : ℕ, (a + b : ℚ) / c = m ∧ (b + c : ℚ) / a = n ∧ (c + a : ℚ) / b = p) →
  (∃ k : ℕ, (a = k ∧ b = k ∧ c = k) ∨
            (a = k ∧ b = k ∧ c = 2 * k) ∨
            (a = k ∧ b = 2 * k ∧ c = 3 * k) ∨
            (a = k ∧ c = 2 * k ∧ b = 3 * k) ∨
            (b = k ∧ a = 2 * k ∧ c = 3 * k) ∨
            (b = k ∧ c = 2 * k ∧ a = 3 * k) ∨
            (c = k ∧ a = 2 * k ∧ b = 3 * k) ∨
            (c = k ∧ b = 2 * k ∧ a = 3 * k)) :=
sorry

end natural_number_triples_l2630_263093


namespace roots_of_quadratic_equation_l2630_263039

theorem roots_of_quadratic_equation :
  let f : ℝ → ℝ := λ x => x^2 - 2*x
  (f 0 = 0) ∧ (f 2 = 0) ∧ (∀ x : ℝ, f x = 0 → x = 0 ∨ x = 2) :=
by sorry

end roots_of_quadratic_equation_l2630_263039


namespace greatest_integer_less_than_negative_25_over_6_l2630_263053

theorem greatest_integer_less_than_negative_25_over_6 :
  Int.floor (-25 / 6 : ℚ) = -5 := by
  sorry

end greatest_integer_less_than_negative_25_over_6_l2630_263053


namespace certain_number_value_certain_number_value_proof_l2630_263033

theorem certain_number_value : ℝ → Prop :=
  fun y =>
    let x : ℝ := (390 - (48 + 62 + 98 + 124)) -- x from the second set
    let first_set : List ℝ := [28, x, 42, 78, y]
    let second_set : List ℝ := [48, 62, 98, 124, x]
    (List.sum first_set / first_set.length = 62) ∧
    (List.sum second_set / second_set.length = 78) →
    y = 104

-- The proof goes here
theorem certain_number_value_proof : certain_number_value 104 := by
  sorry

end certain_number_value_certain_number_value_proof_l2630_263033


namespace max_value_difference_l2630_263023

-- Define the function f(x)
def f (x : ℝ) : ℝ := 3 * x - x^3

-- Define a as the point where f(x) reaches its maximum value
def a : ℝ := 1

-- Define b as the maximum value of f(x)
def b : ℝ := f a

-- Theorem statement
theorem max_value_difference (x : ℝ) : a - b = -1 := by
  sorry

end max_value_difference_l2630_263023


namespace max_value_is_58_l2630_263062

/-- Represents a type of stone with its weight and value -/
structure Stone where
  weight : ℕ
  value : ℕ

/-- The problem setup -/
def cave_problem :=
  let stone7 : Stone := { weight := 7, value := 16 }
  let stone3 : Stone := { weight := 3, value := 9 }
  let stone2 : Stone := { weight := 2, value := 4 }
  let max_weight : ℕ := 20
  let max_stone7 : ℕ := 2
  (stone7, stone3, stone2, max_weight, max_stone7)

/-- The function to maximize the value of stones -/
def maximize_value (p : Stone × Stone × Stone × ℕ × ℕ) : ℕ :=
  let (stone7, stone3, stone2, max_weight, max_stone7) := p
  sorry -- The actual maximization logic would go here

/-- The theorem stating that the maximum value is 58 -/
theorem max_value_is_58 : maximize_value cave_problem = 58 := by
  sorry

end max_value_is_58_l2630_263062


namespace calculation_sum_l2630_263082

theorem calculation_sum (x : ℝ) (h : (x - 5) + 14 = 39) : (5 * x + 14) + 39 = 203 := by
  sorry

end calculation_sum_l2630_263082


namespace corner_spheres_sum_diameter_l2630_263011

-- Define a sphere in a corner
structure CornerSphere where
  radius : ℝ
  center : ℝ × ℝ × ℝ

-- Define the condition for a point on the sphere
def satisfiesCondition (s : CornerSphere) : Prop :=
  ∃ (x y z : ℝ), 
    (x - s.radius)^2 + (y - s.radius)^2 + (z - s.radius)^2 = s.radius^2 ∧
    x = 5 ∧ y = 5 ∧ z = 10

theorem corner_spheres_sum_diameter :
  ∀ (s1 s2 : CornerSphere),
    satisfiesCondition s1 → satisfiesCondition s2 →
    s1.center = (s1.radius, s1.radius, s1.radius) →
    s2.center = (s2.radius, s2.radius, s2.radius) →
    2 * (s1.radius + s2.radius) = 40 := by
  sorry

end corner_spheres_sum_diameter_l2630_263011


namespace product_parity_probabilities_l2630_263037

/-- The probability that the product of two arbitrary natural numbers is even -/
def prob_even_product : ℚ := 3/4

/-- The probability that the product of two arbitrary natural numbers is odd -/
def prob_odd_product : ℚ := 1/4

theorem product_parity_probabilities :
  (prob_even_product + prob_odd_product = 1) ∧
  (prob_even_product = 3/4) ∧
  (prob_odd_product = 1/4) := by
  sorry

end product_parity_probabilities_l2630_263037


namespace soccer_enjoyment_misreporting_l2630_263074

theorem soccer_enjoyment_misreporting (total : ℝ) (total_pos : 0 < total) :
  let enjoy := 0.7 * total
  let dont_enjoy := 0.3 * total
  let say_dont_but_do := 0.25 * enjoy
  let say_dont_and_dont := 0.85 * dont_enjoy
  say_dont_but_do / (say_dont_but_do + say_dont_and_dont) = 2/5 :=
by sorry

end soccer_enjoyment_misreporting_l2630_263074


namespace range_of_a_l2630_263080

-- Define the conditions
def p (x : ℝ) : Prop := 0 < x ∧ x < 2
def q (x a : ℝ) : Prop := x < a

-- Define the sufficient but not necessary condition
def sufficient_not_necessary (p q : ℝ → Prop) : Prop :=
  (∀ x, p x → q x) ∧ ¬(∀ x, q x → p x)

-- Theorem statement
theorem range_of_a (h : sufficient_not_necessary p (q · a)) :
  ∀ y : ℝ, y ≥ 2 ↔ ∃ x : ℝ, a = y := by
  sorry

end range_of_a_l2630_263080


namespace sum_753_326_base8_l2630_263067

/-- Converts a base-8 number represented as a list of digits to its decimal equivalent. -/
def base8ToDecimal (digits : List Nat) : Nat :=
  digits.foldr (fun d acc => 8 * acc + d) 0

/-- Converts a decimal number to its base-8 representation as a list of digits. -/
def decimalToBase8 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) : List Nat :=
    if m = 0 then acc else aux (m / 8) ((m % 8) :: acc)
  aux n []

/-- The theorem stating that the sum of 753₈ and 326₈ in base 8 is 1301₈. -/
theorem sum_753_326_base8 :
  decimalToBase8 (base8ToDecimal [7, 5, 3] + base8ToDecimal [3, 2, 6]) = [1, 3, 0, 1] := by
  sorry

end sum_753_326_base8_l2630_263067


namespace prob_all_even_four_dice_l2630_263044

/-- The probability of a single standard six-sided die showing an even number -/
def prob_even_single : ℚ := 1 / 2

/-- The number of dice being tossed simultaneously -/
def num_dice : ℕ := 4

/-- Theorem: The probability of all four standard six-sided dice showing even numbers
    when tossed simultaneously is 1/16 -/
theorem prob_all_even_four_dice :
  (prob_even_single ^ num_dice : ℚ) = 1 / 16 := by
  sorry

end prob_all_even_four_dice_l2630_263044


namespace function_domain_iff_m_range_l2630_263038

/-- The function f(x) = lg(x^2 - 2mx + m + 2) has domain R if and only if m ∈ (-1, 2) -/
theorem function_domain_iff_m_range (m : ℝ) : 
  (∀ x : ℝ, ∃ y : ℝ, y = Real.log (x^2 - 2*m*x + m + 2)) ↔ m > -1 ∧ m < 2 := by
sorry


end function_domain_iff_m_range_l2630_263038


namespace tricycle_count_l2630_263072

theorem tricycle_count (num_bicycles : ℕ) (bicycle_wheels : ℕ) (tricycle_wheels : ℕ) (total_wheels : ℕ) :
  num_bicycles = 24 →
  bicycle_wheels = 2 →
  tricycle_wheels = 3 →
  total_wheels = 90 →
  ∃ num_tricycles : ℕ, num_tricycles = 14 ∧ 
    num_bicycles * bicycle_wheels + num_tricycles * tricycle_wheels = total_wheels :=
by sorry

end tricycle_count_l2630_263072


namespace sum_of_amp_operations_l2630_263056

-- Define the operation &
def amp (a b : ℤ) : ℤ := (a + b) * (a - b)

-- Theorem statement
theorem sum_of_amp_operations : amp 12 5 + amp 8 3 = 174 := by
  sorry

end sum_of_amp_operations_l2630_263056


namespace second_question_percentage_l2630_263068

theorem second_question_percentage
  (first_correct : ℝ)
  (neither_correct : ℝ)
  (both_correct : ℝ)
  (h1 : first_correct = 75)
  (h2 : neither_correct = 20)
  (h3 : both_correct = 20) :
  ∃ (second_correct : ℝ),
    second_correct = 25 ∧
    first_correct + second_correct - both_correct = 100 - neither_correct :=
sorry

end second_question_percentage_l2630_263068


namespace shoes_cost_calculation_l2630_263025

def shopping_problem (initial_amount sweater_cost tshirt_cost amount_left : ℕ) : Prop :=
  let total_spent := initial_amount - amount_left
  let other_items_cost := sweater_cost + tshirt_cost
  let shoes_cost := total_spent - other_items_cost
  shoes_cost = 11

theorem shoes_cost_calculation :
  shopping_problem 91 24 6 50 := by sorry

end shoes_cost_calculation_l2630_263025


namespace age_difference_l2630_263085

/-- Represents the ages of three brothers -/
structure BrothersAges where
  richard : ℕ
  david : ℕ
  scott : ℕ

/-- The conditions of the problem -/
def problem_conditions (ages : BrothersAges) : Prop :=
  ages.richard = ages.david + 6 ∧
  ages.david > ages.scott ∧
  ∃ (y : ℕ), ages.richard + y = 2 * (ages.scott + y) ∧
  ages.david = 14

/-- The theorem to prove -/
theorem age_difference (ages : BrothersAges) :
  problem_conditions ages →
  ∃ (s : ℕ), s < 14 ∧ ages.david - ages.scott = 14 - s :=
by sorry

end age_difference_l2630_263085


namespace quadratic_factorization_l2630_263012

theorem quadratic_factorization (C D : ℤ) :
  (∀ y : ℝ, 15 * y^2 - 76 * y + 48 = (C * y - 16) * (D * y - 3)) →
  C * D + C = 20 := by
  sorry

end quadratic_factorization_l2630_263012


namespace harry_says_1111_l2630_263016

/-- Represents a student in the counting game -/
inductive Student
| Adam
| Beth
| Claire
| Debby
| Eva
| Frank
| Gina
| Harry

/-- Defines the rules for each student's counting pattern -/
def countingRule (s : Student) : ℕ → Prop :=
  match s with
  | Student.Adam => λ n => n % 4 ≠ 0
  | Student.Beth => λ n => (n % 4 = 0) ∧ (n % 3 ≠ 2)
  | Student.Claire => λ n => (n % 4 = 0) ∧ (n % 3 = 2) ∧ (n % 3 ≠ 0)
  | Student.Debby => λ n => (n % 4 = 0) ∧ (n % 3 = 2) ∧ (n % 3 = 0) ∧ (n % 2 ≠ 0)
  | Student.Eva => λ n => (n % 4 = 0) ∧ (n % 3 = 2) ∧ (n % 3 = 0) ∧ (n % 2 = 0) ∧ (n % 3 ≠ 0)
  | Student.Frank => λ n => (n % 4 = 0) ∧ (n % 3 = 2) ∧ (n % 3 = 0) ∧ (n % 2 = 0) ∧ (n % 3 = 0) ∧ (n % 2 ≠ 0)
  | Student.Gina => λ n => (n % 4 = 0) ∧ (n % 3 = 2) ∧ (n % 3 = 0) ∧ (n % 2 = 0) ∧ (n % 3 = 0) ∧ (n % 2 = 0) ∧ (n % 3 ≠ 0)
  | Student.Harry => λ n => (n % 4 = 0) ∧ (n % 3 = 2) ∧ (n % 3 = 0) ∧ (n % 2 = 0) ∧ (n % 3 = 0) ∧ (n % 2 = 0) ∧ (n % 3 = 0)

/-- The theorem stating that Harry says the number 1111 -/
theorem harry_says_1111 : countingRule Student.Harry 1111 := by
  sorry

end harry_says_1111_l2630_263016


namespace roses_apples_l2630_263066

/-- Rose's apple distribution problem -/
theorem roses_apples (num_friends : ℕ) (apples_per_friend : ℕ) : 
  num_friends = 3 → apples_per_friend = 3 → num_friends * apples_per_friend = 9 :=
by sorry

end roses_apples_l2630_263066


namespace log_inequality_l2630_263018

theorem log_inequality (m n : ℝ) (hm : m > 0) (hn : n > 0) :
  (Real.log 3 / Real.log m < Real.log 3 / Real.log n) ∧ (Real.log 3 / Real.log n < 0) →
  1 > m ∧ m > n ∧ n > 0 := by
sorry

end log_inequality_l2630_263018


namespace least_reducible_fraction_l2630_263003

theorem least_reducible_fraction (n : ℕ) : 
  (∀ m : ℕ, m > 0 ∧ m < 34 → ¬(Nat.gcd (m - 29) (3 * m + 8) > 1)) ∧ 
  (34 > 0) ∧ 
  (Nat.gcd (34 - 29) (3 * 34 + 8) > 1) :=
sorry

end least_reducible_fraction_l2630_263003


namespace max_sum_of_proportional_integers_l2630_263078

theorem max_sum_of_proportional_integers (x y z : ℤ) : 
  (x : ℚ) / 5 = 6 / (y : ℚ) → 
  (x : ℚ) / 5 = (z : ℚ) / 2 → 
  (∃ (a b c : ℤ), x = a ∧ y = b ∧ z = c) →
  (∀ (x' y' z' : ℤ), (x' : ℚ) / 5 = 6 / (y' : ℚ) → (x' : ℚ) / 5 = (z' : ℚ) / 2 → x + y + z ≥ x' + y' + z') →
  x + y + z = 43 :=
by sorry

end max_sum_of_proportional_integers_l2630_263078


namespace a_4_value_l2630_263059

def geometric_sequence (a : ℕ → ℝ) (r : ℝ) :=
  ∀ n, a (n + 1) = r * a n

theorem a_4_value (a : ℕ → ℝ) :
  geometric_sequence a 2 →
  a 4 / a 2 - a 3 = 0 →
  a 4 = 8 := by
sorry

end a_4_value_l2630_263059


namespace max_rides_both_days_l2630_263061

/-- Represents the prices of rides on a given day -/
structure RidePrices where
  ferrisWheel : ℕ
  rollerCoaster : ℕ
  bumperCars : ℕ
  carousel : ℕ
  logFlume : ℕ
  hauntedHouse : Option ℕ

/-- Calculates the maximum number of rides within a budget -/
def maxRides (prices : RidePrices) (budget : ℕ) : ℕ :=
  sorry

/-- The daily budget -/
def dailyBudget : ℕ := 10

/-- Ride prices for the first day -/
def firstDayPrices : RidePrices :=
  { ferrisWheel := 4
  , rollerCoaster := 5
  , bumperCars := 3
  , carousel := 2
  , logFlume := 6
  , hauntedHouse := none }

/-- Ride prices for the second day -/
def secondDayPrices : RidePrices :=
  { ferrisWheel := 4
  , rollerCoaster := 7
  , bumperCars := 3
  , carousel := 2
  , logFlume := 6
  , hauntedHouse := some 4 }

theorem max_rides_both_days :
  maxRides firstDayPrices dailyBudget = 3 ∧
  maxRides secondDayPrices dailyBudget = 3 :=
sorry

end max_rides_both_days_l2630_263061


namespace surface_area_of_seven_solid_arrangement_l2630_263014

/-- Represents a 1 × 1 × 2 solid -/
structure Solid :=
  (length : ℝ := 1)
  (width : ℝ := 1)
  (height : ℝ := 2)

/-- Represents the arrangement of solids as shown in the diagram -/
def Arrangement := List Solid

/-- Calculates the surface area of the arrangement -/
def surfaceArea (arr : Arrangement) : ℝ :=
  sorry

/-- The specific arrangement of seven solids as shown in the diagram -/
def sevenSolidArrangement : Arrangement :=
  List.replicate 7 { length := 1, width := 1, height := 2 }

theorem surface_area_of_seven_solid_arrangement :
  surfaceArea sevenSolidArrangement = 42 :=
by sorry

end surface_area_of_seven_solid_arrangement_l2630_263014


namespace dvd_cost_is_six_l2630_263047

/-- Represents the DVD production and sales scenario --/
structure DVDProduction where
  movieCost : ℕ
  dailySales : ℕ
  daysPerWeek : ℕ
  weeks : ℕ
  profit : ℕ
  sellingPriceFactor : ℚ

/-- Calculates the production cost of a single DVD --/
def calculateDVDCost (p : DVDProduction) : ℚ :=
  let totalSales := p.dailySales * p.daysPerWeek * p.weeks
  let revenue := p.profit + p.movieCost
  let costPerDVD := revenue / (totalSales * (p.sellingPriceFactor - 1))
  costPerDVD

/-- Theorem stating that the DVD production cost is $6 --/
theorem dvd_cost_is_six (p : DVDProduction) 
  (h1 : p.movieCost = 2000)
  (h2 : p.dailySales = 500)
  (h3 : p.daysPerWeek = 5)
  (h4 : p.weeks = 20)
  (h5 : p.profit = 448000)
  (h6 : p.sellingPriceFactor = 5/2) :
  calculateDVDCost p = 6 := by
  sorry

#eval calculateDVDCost {
  movieCost := 2000,
  dailySales := 500,
  daysPerWeek := 5,
  weeks := 20,
  profit := 448000,
  sellingPriceFactor := 5/2
}

end dvd_cost_is_six_l2630_263047


namespace ceiling_floor_product_l2630_263090

theorem ceiling_floor_product (y : ℝ) :
  y < 0 → ⌈y⌉ * ⌊y⌋ = 132 → -12 < y ∧ y < -11 := by
  sorry

end ceiling_floor_product_l2630_263090


namespace exam_candidates_count_l2630_263073

theorem exam_candidates_count : 
  ∀ (T P F : ℕ) (total_avg passed_avg failed_avg : ℚ),
    P = 100 →
    total_avg = 35 →
    passed_avg = 39 →
    failed_avg = 15 →
    T = P + F →
    (total_avg * T : ℚ) = (passed_avg * P : ℚ) + (failed_avg * F : ℚ) →
    T = 120 := by
  sorry

end exam_candidates_count_l2630_263073


namespace ace_in_top_probability_l2630_263040

/-- A standard deck of cards --/
def standard_deck : ℕ := 52

/-- The number of top cards we're considering --/
def top_cards : ℕ := 3

/-- The probability of the Ace of Spades being among the top cards --/
def prob_ace_in_top : ℚ := 3 / 52

theorem ace_in_top_probability :
  prob_ace_in_top = top_cards / standard_deck :=
by sorry

end ace_in_top_probability_l2630_263040


namespace fgh_supermarkets_in_us_l2630_263069

/-- The number of FGH supermarkets in the US, given the total number of supermarkets
    and the difference between US and Canadian supermarkets. -/
def us_supermarkets (total : ℕ) (difference : ℕ) : ℕ :=
  (total + difference) / 2

theorem fgh_supermarkets_in_us :
  us_supermarkets 60 22 = 41 := by
  sorry

#eval us_supermarkets 60 22

end fgh_supermarkets_in_us_l2630_263069


namespace lunch_gratuity_percentage_l2630_263035

/-- Given the conditions of a lunch bill, prove the gratuity percentage --/
theorem lunch_gratuity_percentage
  (total_price : ℝ)
  (num_people : ℕ)
  (avg_price_no_gratuity : ℝ)
  (h1 : total_price = 207)
  (h2 : num_people = 15)
  (h3 : avg_price_no_gratuity = 12) :
  (total_price - (↑num_people * avg_price_no_gratuity)) / (↑num_people * avg_price_no_gratuity) * 100 = 15 := by
  sorry

end lunch_gratuity_percentage_l2630_263035


namespace problem_1_l2630_263002

theorem problem_1 (x : ℝ) (hx : x ≠ 0) :
  (-2 * x^5 + 3 * x^3 - (1/2) * x^2) / ((-1/2 * x)^2) = -8 * x^3 + 12 * x - 2 := by
  sorry

end problem_1_l2630_263002


namespace impossibility_of_filling_l2630_263021

/-- A brick is made of four unit cubes: one unit cube with three unit cubes
    attached to three of its faces, all sharing a common vertex. -/
structure Brick :=
  (cubes : Fin 4 → Unit)

/-- A rectangular parallelepiped with dimensions 11 × 12 × 13 -/
def Parallelepiped := Fin 11 × Fin 12 × Fin 13

/-- A function that represents filling the parallelepiped with bricks -/
def FillParallelepiped := Parallelepiped → Brick

/-- Theorem stating that it's impossible to fill the 11 × 12 × 13 parallelepiped with the given bricks -/
theorem impossibility_of_filling :
  ¬ ∃ (f : FillParallelepiped), True :=
sorry

end impossibility_of_filling_l2630_263021


namespace max_y_value_l2630_263046

theorem max_y_value (x y : ℤ) (h : 3*x*y + 7*x + 6*y = 20) : 
  y ≤ 16 ∧ ∃ (x' y' : ℤ), 3*x'*y' + 7*x' + 6*y' = 20 ∧ y' = 16 :=
sorry

end max_y_value_l2630_263046


namespace quadratic_monotonicity_l2630_263086

/-- A quadratic function f(x) = ax² + bx + 3 -/
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + 3

/-- The derivative of f(x) -/
def f' (a b : ℝ) (x : ℝ) : ℝ := 2 * a * x + b

theorem quadratic_monotonicity (a b : ℝ) :
  (∀ x ≤ -1, 0 ≤ f' a b x) →
  (∀ x ≥ -1, f' a b x ≤ 0) →
  b = 2 * a ∧ a < 0 := by
sorry

end quadratic_monotonicity_l2630_263086


namespace min_abs_E_is_zero_l2630_263029

/-- Given a real-valued function E, prove that its minimum absolute value is 0
    when the minimum of |E(x)| + |x + 6| + |x - 5| is 11 for all real x. -/
theorem min_abs_E_is_zero (E : ℝ → ℝ) : 
  (∀ x, |E x| + |x + 6| + |x - 5| ≥ 11) → 
  (∃ x, |E x| + |x + 6| + |x - 5| = 11) → 
  ∃ x, |E x| = 0 :=
sorry

end min_abs_E_is_zero_l2630_263029


namespace brand_preference_ratio_l2630_263089

/-- Given a survey with 250 total respondents and 200 preferring brand X,
    prove that the ratio of people preferring brand X to those preferring brand Y is 4:1 -/
theorem brand_preference_ratio (total : ℕ) (brand_x : ℕ) (h1 : total = 250) (h2 : brand_x = 200) :
  (brand_x : ℚ) / (total - brand_x : ℚ) = 4 / 1 := by
  sorry

end brand_preference_ratio_l2630_263089


namespace black_marble_price_is_ten_cents_l2630_263070

/-- Represents the marble pricing problem --/
structure MarbleProblem where
  total_marbles : ℕ
  white_percentage : ℚ
  black_percentage : ℚ
  white_price : ℚ
  color_price : ℚ
  total_earnings : ℚ

/-- Calculates the price of each black marble --/
def black_marble_price (p : MarbleProblem) : ℚ :=
  let white_marbles := p.white_percentage * p.total_marbles
  let black_marbles := p.black_percentage * p.total_marbles
  let color_marbles := p.total_marbles - (white_marbles + black_marbles)
  let white_earnings := white_marbles * p.white_price
  let color_earnings := color_marbles * p.color_price
  let black_earnings := p.total_earnings - (white_earnings + color_earnings)
  black_earnings / black_marbles

/-- Theorem stating that the black marble price is $0.10 --/
theorem black_marble_price_is_ten_cents 
  (p : MarbleProblem) 
  (h1 : p.total_marbles = 100)
  (h2 : p.white_percentage = 1/5)
  (h3 : p.black_percentage = 3/10)
  (h4 : p.white_price = 1/20)
  (h5 : p.color_price = 1/5)
  (h6 : p.total_earnings = 14) :
  black_marble_price p = 1/10 := by
  sorry

#eval black_marble_price { 
  total_marbles := 100, 
  white_percentage := 1/5, 
  black_percentage := 3/10, 
  white_price := 1/5, 
  color_price := 1/5, 
  total_earnings := 14 
}

end black_marble_price_is_ten_cents_l2630_263070


namespace gcd_lcm_product_360_l2630_263019

theorem gcd_lcm_product_360 : 
  ∃! (s : Finset ℕ), 
    (∀ d ∈ s, d > 0 ∧ ∃ a b : ℕ, a > 0 ∧ b > 0 ∧ d = Nat.gcd a b ∧ d * Nat.lcm a b = 360) ∧ 
    s.card = 17 :=
by sorry

end gcd_lcm_product_360_l2630_263019


namespace greenhouse_path_area_l2630_263000

/-- Calculates the total area of paths in Joanna's greenhouse --/
theorem greenhouse_path_area :
  let num_rows : ℕ := 5
  let beds_per_row : ℕ := 3
  let bed_width : ℕ := 4
  let bed_height : ℕ := 3
  let path_width : ℕ := 2
  
  let total_width : ℕ := beds_per_row * bed_width + (beds_per_row + 1) * path_width
  let total_height : ℕ := num_rows * bed_height + (num_rows + 1) * path_width
  
  let total_area : ℕ := total_width * total_height
  let bed_area : ℕ := num_rows * beds_per_row * bed_width * bed_height
  
  total_area - bed_area = 360 :=
by sorry


end greenhouse_path_area_l2630_263000


namespace point_q_coordinates_l2630_263030

/-- Given two points P and Q in a 2D Cartesian coordinate system,
    prove that Q has coordinates (1, -3) under the given conditions. -/
theorem point_q_coordinates
  (P Q : ℝ × ℝ)  -- P and Q are points in 2D space
  (h1 : P = (1, 2))  -- P has coordinates (1, 2)
  (h2 : (Q.2 : ℝ) < 0)  -- Q is below the x-axis
  (h3 : P.1 = Q.1)  -- PQ is parallel to the y-axis
  (h4 : Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) = 5)  -- PQ = 5
  : Q = (1, -3) := by
  sorry

end point_q_coordinates_l2630_263030


namespace sum_of_squares_l2630_263098

theorem sum_of_squares (x y : ℕ+) 
  (h1 : x * y + x + y = 90)
  (h2 : x^2 * y + x * y^2 = 1122) : 
  x^2 + y^2 = 1044 := by
  sorry

end sum_of_squares_l2630_263098


namespace factor_polynomial_l2630_263065

theorem factor_polynomial (x y : ℝ) : 2 * x^2 - 2 * y^2 = 2 * (x + y) * (x - y) := by
  sorry

end factor_polynomial_l2630_263065


namespace valid_selections_count_l2630_263052

def male_teachers : ℕ := 5
def female_teachers : ℕ := 4
def total_teachers : ℕ := male_teachers + female_teachers
def selected_teachers : ℕ := 3

def all_selections : ℕ := (total_teachers.choose selected_teachers)
def all_male_selections : ℕ := (male_teachers.choose selected_teachers)
def all_female_selections : ℕ := (female_teachers.choose selected_teachers)

theorem valid_selections_count : 
  all_selections - (all_male_selections + all_female_selections) = 420 := by
  sorry

end valid_selections_count_l2630_263052


namespace permutations_of_eight_distinct_objects_l2630_263005

theorem permutations_of_eight_distinct_objects : Nat.factorial 8 = 40320 := by
  sorry

end permutations_of_eight_distinct_objects_l2630_263005


namespace floor_subtraction_inequality_l2630_263045

theorem floor_subtraction_inequality (x y : ℝ) : ⌊x - y⌋ ≤ ⌊x⌋ - ⌊y⌋ := by
  sorry

end floor_subtraction_inequality_l2630_263045


namespace ab_equals_two_l2630_263064

theorem ab_equals_two (a b : ℝ) (h : (a + 1)^2 + (b + 2)^2 = 0) : a * b = 2 := by
  sorry

end ab_equals_two_l2630_263064


namespace multiplication_subtraction_difference_l2630_263075

theorem multiplication_subtraction_difference : ∃ (x : ℝ), x = 10 ∧ (3 * x) - (26 - x) = 14 := by
  sorry

end multiplication_subtraction_difference_l2630_263075


namespace ellas_food_consumption_l2630_263099

/-- 
Given that:
1. Ella's dog eats 4 times as much food as Ella each day
2. Ella eats 20 pounds of food per day
3. The total food consumption for Ella and her dog over some number of days is 1000 pounds

This theorem proves that the number of days is 10.
-/
theorem ellas_food_consumption (dog_ratio : ℕ) (ella_daily : ℕ) (total_food : ℕ) :
  dog_ratio = 4 →
  ella_daily = 20 →
  total_food = 1000 →
  ∃ (days : ℕ), days = 10 ∧ total_food = (ella_daily + dog_ratio * ella_daily) * days :=
by sorry

end ellas_food_consumption_l2630_263099


namespace completing_square_quadratic_l2630_263095

theorem completing_square_quadratic (x : ℝ) : 
  x^2 - 4*x + 3 = 0 ↔ (x - 2)^2 = 1 := by
  sorry

end completing_square_quadratic_l2630_263095


namespace intersection_A_complement_B_is_empty_l2630_263004

-- Define the sets A and B
def A : Set ℝ := {x | Real.sqrt (x - 2) ≤ 0}
def B : Set ℝ := {x | 10^2 * 2 = 10^2}

-- State the theorem
theorem intersection_A_complement_B_is_empty :
  A ∩ Bᶜ = ∅ := by sorry

end intersection_A_complement_B_is_empty_l2630_263004


namespace chess_tournament_matches_l2630_263032

/-- Represents a single elimination tournament --/
structure Tournament :=
  (total_players : ℕ)
  (bye_players : ℕ)
  (h_bye : bye_players < total_players)

/-- Calculates the number of matches in a tournament --/
def matches_played (t : Tournament) : ℕ := t.total_players - 1

/-- Main theorem about the chess tournament --/
theorem chess_tournament_matches :
  ∃ (t : Tournament),
    t.total_players = 120 ∧
    t.bye_players = 40 ∧
    matches_played t = 119 ∧
    119 % 7 = 0 := by sorry

end chess_tournament_matches_l2630_263032


namespace proposition_is_false_l2630_263026

theorem proposition_is_false : ∃ m n : ℤ, m > n ∧ m^2 ≤ n^2 := by sorry

end proposition_is_false_l2630_263026


namespace appetizer_price_l2630_263092

def total_spent : ℚ := 50
def entree_percentage : ℚ := 80 / 100
def num_entrees : ℕ := 4
def num_appetizers : ℕ := 2

theorem appetizer_price :
  let entree_cost : ℚ := total_spent * entree_percentage
  let appetizer_total : ℚ := total_spent - entree_cost
  let single_appetizer_price : ℚ := appetizer_total / num_appetizers
  single_appetizer_price = 5 := by
sorry

end appetizer_price_l2630_263092


namespace equation_transformation_l2630_263015

theorem equation_transformation (x y : ℝ) : x - 3 = y - 3 → x - y = 0 := by
  sorry

end equation_transformation_l2630_263015


namespace product_equals_584638125_l2630_263007

theorem product_equals_584638125 : 625 * 935421 = 584638125 := by
  sorry

end product_equals_584638125_l2630_263007


namespace jellybean_count_l2630_263001

/-- The number of jellybeans in a dozen -/
def jellybeans_per_dozen : ℕ := 12

/-- Caleb's number of jellybeans -/
def caleb_jellybeans : ℕ := 3 * jellybeans_per_dozen

/-- Sophie's number of jellybeans -/
def sophie_jellybeans : ℕ := caleb_jellybeans / 2

/-- The total number of jellybeans Caleb and Sophie have together -/
def total_jellybeans : ℕ := caleb_jellybeans + sophie_jellybeans

theorem jellybean_count : total_jellybeans = 54 := by
  sorry

end jellybean_count_l2630_263001


namespace research_development_percentage_l2630_263091

/-- Represents the budget allocation of a company -/
structure BudgetAllocation where
  transportation : ℝ
  utilities : ℝ
  equipment : ℝ
  supplies : ℝ
  salaries : ℝ
  research_development : ℝ

/-- The theorem stating that the research and development budget is 9% -/
theorem research_development_percentage
  (budget : BudgetAllocation)
  (h1 : budget.transportation = 15)
  (h2 : budget.utilities = 5)
  (h3 : budget.equipment = 4)
  (h4 : budget.supplies = 2)
  (h5 : budget.salaries = 234 / 360 * 100)
  (h6 : budget.transportation + budget.utilities + budget.equipment + budget.supplies + budget.salaries + budget.research_development = 100) :
  budget.research_development = 9 := by
sorry


end research_development_percentage_l2630_263091


namespace geometric_sequence_sum_l2630_263058

/-- A geometric sequence with positive terms -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = q * a n

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  geometric_sequence a →
  a 1 = 1 →
  a 1 + a 3 + a 5 = 21 →
  a 2 + a 4 + a 6 = 42 :=
by
  sorry

end geometric_sequence_sum_l2630_263058


namespace complex_expression_equals_negative_five_l2630_263077

theorem complex_expression_equals_negative_five :
  Real.sqrt 27 + (-1/3)⁻¹ - |2 - Real.sqrt 3| - 8 * Real.cos (30 * π / 180) = -5 := by
  sorry

end complex_expression_equals_negative_five_l2630_263077


namespace platform_length_l2630_263034

/-- Given a train of length 300 meters that takes 39 seconds to cross a platform
    and 20 seconds to cross a signal pole, the length of the platform is 285 meters. -/
theorem platform_length (train_length : ℝ) (time_platform : ℝ) (time_pole : ℝ)
  (h1 : train_length = 300)
  (h2 : time_platform = 39)
  (h3 : time_pole = 20) :
  let speed := train_length / time_pole
  let platform_length := speed * time_platform - train_length
  platform_length = 285 := by sorry

end platform_length_l2630_263034


namespace sum_of_z_values_l2630_263049

-- Define the function f
def f (x : ℝ) : ℝ := (4*x)^2 - 3*(4*x) + 2

-- State the theorem
theorem sum_of_z_values (f : ℝ → ℝ) : 
  (f = λ x => (4*x)^2 - 3*(4*x) + 2) → 
  (∃ z₁ z₂ : ℝ, f z₁ = 9 ∧ f z₂ = 9 ∧ z₁ ≠ z₂ ∧ z₁ + z₂ = 3/16) := by
  sorry


end sum_of_z_values_l2630_263049


namespace triangle_similarity_condition_l2630_263010

theorem triangle_similarity_condition 
  (a b c a₁ b₁ c₁ : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (ha₁ : a₁ > 0) (hb₁ : b₁ > 0) (hc₁ : c₁ > 0) :
  (∃ (k : ℝ), a₁ = k * a ∧ b₁ = k * b ∧ c₁ = k * c) ↔ 
  (Real.sqrt (a * a₁) + Real.sqrt (b * b₁) + Real.sqrt (c * c₁) = 
   Real.sqrt ((a + b + c) * (a₁ + b₁ + c₁))) := by
sorry

end triangle_similarity_condition_l2630_263010


namespace factorization_proof_l2630_263028

variable (x y a b : ℝ)

theorem factorization_proof :
  (9*(3*x - 2*y)^2 - (x - y)^2 = (10*x - 7*y)*(8*x - 5*y)) ∧
  (a^2*b^2 + 4*a*b + 4 - b^2 = (a*b + 2 + b)*(a*b + 2 - b)) := by
  sorry


end factorization_proof_l2630_263028


namespace square_difference_l2630_263050

theorem square_difference : (40 : ℕ)^2 - (39 : ℕ)^2 = 79 := by sorry

end square_difference_l2630_263050


namespace max_xy_given_constraint_l2630_263031

theorem max_xy_given_constraint (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 7 * x + 8 * y = 112) :
  x * y ≤ 56 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 7 * x₀ + 8 * y₀ = 112 ∧ x₀ * y₀ = 56 := by
  sorry

end max_xy_given_constraint_l2630_263031


namespace infinite_occurrence_in_sequence_l2630_263009

/-- Represents a polynomial with integer coefficients -/
def IntPolynomial := List Int

/-- Computes the sum of digits of a natural number in decimal system -/
def sumOfDigits (n : Nat) : Nat :=
  sorry

/-- Computes a_n for a given polynomial and natural number n -/
def computeA (P : IntPolynomial) (n : Nat) : Nat :=
  sumOfDigits (sorry)  -- Evaluate P(n) and compute sum of digits

/-- The main theorem -/
theorem infinite_occurrence_in_sequence (P : IntPolynomial) :
  ∃ (k : Nat), Set.Infinite {n : Nat | computeA P n = k} :=
sorry

end infinite_occurrence_in_sequence_l2630_263009


namespace inequality_linear_iff_k_eq_two_l2630_263087

/-- The inequality (k+2)x^(|k|-1) + 5 < 0 is linear in x if and only if k = 2 -/
theorem inequality_linear_iff_k_eq_two (k : ℝ) : 
  (∃ (a b : ℝ), ∀ x, ((k + 2) * x^(|k| - 1) + 5 < 0) ↔ (a * x + b < 0)) ↔ k = 2 :=
sorry

end inequality_linear_iff_k_eq_two_l2630_263087


namespace max_purple_points_theorem_l2630_263027

/-- The maximum number of purple points in a configuration of blue and red lines -/
def max_purple_points (n : ℕ) : ℕ :=
  n * (n - 1) * (n - 2) * (n - 3) * (n - 4) / 8

/-- Theorem stating the maximum number of purple points given n blue lines -/
theorem max_purple_points_theorem (n : ℕ) (h : n ≥ 5) :
  let blue_lines := n
  let no_parallel := true
  let no_concurrent := true
  max_purple_points n = n * (n - 1) * (n - 2) * (n - 3) * (n - 4) / 8 :=
by
  sorry

#check max_purple_points_theorem

end max_purple_points_theorem_l2630_263027
