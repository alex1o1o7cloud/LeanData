import Mathlib

namespace attendance_probability_additional_A_tickets_needed_l2840_284086

def total_students : ℕ := 50
def tickets_A : ℕ := 3
def tickets_B : ℕ := 7
def tickets_C : ℕ := 10

def total_tickets : ℕ := tickets_A + tickets_B + tickets_C

theorem attendance_probability :
  (total_tickets : ℚ) / total_students = 2 / 5 := by sorry

theorem additional_A_tickets_needed (x : ℕ) :
  (tickets_A + x : ℚ) / total_students = 1 / 5 → x = 7 := by sorry

end attendance_probability_additional_A_tickets_needed_l2840_284086


namespace expression_factorization_l2840_284017

theorem expression_factorization (x : ℝ) :
  (16 * x^6 + 36 * x^4 - 9) - (4 * x^6 - 12 * x^4 + 3) = 12 * (x^6 + 4 * x^4 - 1) := by
  sorry

end expression_factorization_l2840_284017


namespace hcl_mixing_theorem_l2840_284078

/-- Represents a solution with a given volume and concentration -/
structure Solution where
  volume : ℝ
  concentration : ℝ

/-- Calculates the volume of pure HCL in a solution -/
def pureHCL (s : Solution) : ℝ := s.volume * s.concentration

/-- Theorem stating the correctness of the mixing process -/
theorem hcl_mixing_theorem (sol1 sol2 final : Solution) : 
  sol1.volume = 30.0 →
  sol1.concentration = 0.1 →
  sol2.volume = 20.0 →
  sol2.concentration = 0.6 →
  final.volume = sol1.volume + sol2.volume →
  final.concentration = 0.3 →
  pureHCL sol1 + pureHCL sol2 = pureHCL final ∧
  final.volume = 50.0 := by
  sorry

#check hcl_mixing_theorem

end hcl_mixing_theorem_l2840_284078


namespace unique_two_digit_number_l2840_284079

theorem unique_two_digit_number : ∃! n : ℕ, 
  10 ≤ n ∧ n < 100 ∧ 
  2 ∣ n ∧ 
  3 ∣ (n + 1) ∧ 
  4 ∣ (n + 2) ∧ 
  5 ∣ (n + 3) ∧ 
  n = 62 := by
  sorry

end unique_two_digit_number_l2840_284079


namespace x_plus_2y_equals_5_l2840_284003

theorem x_plus_2y_equals_5 (x y : ℝ) 
  (eq1 : 2 * x + y = 6) 
  (eq2 : (x + y) / 3 = 1.222222222222222) : 
  x + 2 * y = 5 := by
  sorry

end x_plus_2y_equals_5_l2840_284003


namespace pies_sold_in_week_l2840_284076

/-- The number of pies sold daily by the restaurant -/
def daily_sales : ℕ := 8

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The total number of pies sold in a week -/
def weekly_sales : ℕ := daily_sales * days_in_week

theorem pies_sold_in_week : weekly_sales = 56 := by
  sorry

end pies_sold_in_week_l2840_284076


namespace joyce_apple_count_l2840_284089

/-- The number of apples Joyce ends up with after receiving apples from Larry -/
def final_apple_count (initial : ℝ) (received : ℝ) : ℝ :=
  initial + received

/-- Theorem stating that Joyce ends up with 127.0 apples -/
theorem joyce_apple_count : final_apple_count 75.0 52.0 = 127.0 := by
  sorry

end joyce_apple_count_l2840_284089


namespace expression_factorization_l2840_284019

theorem expression_factorization (b : ℝ) :
  (9 * b^3 + 126 * b^2 - 11) - (-8 * b^3 + 2 * b^2 - 11) = b^2 * (17 * b + 124) := by
  sorry

end expression_factorization_l2840_284019


namespace freshmen_sample_size_is_20_l2840_284010

/-- Calculates the number of freshmen to be sampled in a stratified sampling scheme -/
def freshmenSampleSize (totalStudents sampleSize freshmen : ℕ) : ℕ :=
  (freshmen * sampleSize) / totalStudents

/-- Theorem stating that the number of freshmen to be sampled is 20 -/
theorem freshmen_sample_size_is_20 :
  freshmenSampleSize 900 45 400 = 20 := by
  sorry

end freshmen_sample_size_is_20_l2840_284010


namespace tomorrow_is_saturday_l2840_284025

-- Define the days of the week
inductive Day : Type
  | Sunday : Day
  | Monday : Day
  | Tuesday : Day
  | Wednesday : Day
  | Thursday : Day
  | Friday : Day
  | Saturday : Day

-- Define a function to get the next day
def nextDay (d : Day) : Day :=
  match d with
  | Day.Sunday => Day.Monday
  | Day.Monday => Day.Tuesday
  | Day.Tuesday => Day.Wednesday
  | Day.Wednesday => Day.Thursday
  | Day.Thursday => Day.Friday
  | Day.Friday => Day.Saturday
  | Day.Saturday => Day.Sunday

-- Define a function to add days
def addDays (d : Day) (n : Nat) : Day :=
  match n with
  | 0 => d
  | Nat.succ k => nextDay (addDays d k)

-- Define the given condition
def dayBeforeYesterday : Day := Day.Wednesday

-- Theorem to prove
theorem tomorrow_is_saturday : 
  addDays (nextDay (nextDay dayBeforeYesterday)) 1 = Day.Saturday :=
sorry

end tomorrow_is_saturday_l2840_284025


namespace inequality_for_three_positives_l2840_284088

theorem inequality_for_three_positives (x₁ x₂ x₃ : ℝ) (h₁ : x₁ > 0) (h₂ : x₂ > 0) (h₃ : x₃ > 0) :
  (x₁ * x₂ / x₃) + (x₂ * x₃ / x₁) + (x₃ * x₁ / x₂) ≥ x₁ + x₂ + x₃ ∧
  ((x₁ * x₂ / x₃) + (x₂ * x₃ / x₁) + (x₃ * x₁ / x₂) = x₁ + x₂ + x₃ ↔ x₁ = x₂ ∧ x₂ = x₃) :=
by sorry

end inequality_for_three_positives_l2840_284088


namespace roots_product_value_l2840_284040

theorem roots_product_value (x₁ x₂ : ℝ) : 
  (3 * x₁^2 - 9 * x₁ - 21 = 0) → 
  (3 * x₂^2 - 9 * x₂ - 21 = 0) → 
  (3 * x₁ - 4) * (6 * x₂ - 8) = -202 := by
  sorry

end roots_product_value_l2840_284040


namespace anthony_free_throw_improvement_l2840_284099

theorem anthony_free_throw_improvement :
  let initial_success : ℚ := 6 / 15
  let initial_attempts : ℕ := 15
  let additional_success : ℕ := 24
  let additional_attempts : ℕ := 32
  let final_success : ℚ := (6 + additional_success) / (initial_attempts + additional_attempts)
  (final_success - initial_success) * 100 = 24 := by
sorry

end anthony_free_throw_improvement_l2840_284099


namespace natural_number_pairs_l2840_284015

theorem natural_number_pairs : ∀ a b : ℕ, 
  (∃! (s1 s2 s3 s4 : Prop), 
    s1 = (∃ k : ℕ, a^2 + 4*a + 3 = k * b) ∧
    s2 = (a^2 + a*b - 6*b^2 - 2*a - 16*b - 8 = 0) ∧
    s3 = (∃ k : ℕ, a + 2*b + 1 = 4 * k) ∧
    s4 = Nat.Prime (a + 6*b + 1) ∧
    (s1 ∧ s2 ∧ s3 ∧ ¬s4 ∨
     s1 ∧ s2 ∧ ¬s3 ∧ s4 ∨
     s1 ∧ ¬s2 ∧ s3 ∧ s4 ∨
     ¬s1 ∧ s2 ∧ s3 ∧ s4)) →
  ((a = 6 ∧ b = 1) ∨ (a = 18 ∧ b = 7)) := by
sorry

end natural_number_pairs_l2840_284015


namespace M_greater_than_N_l2840_284090

theorem M_greater_than_N : ∀ x : ℝ, (x - 3) * (x - 7) > (x - 2) * (x - 8) := by
  sorry

end M_greater_than_N_l2840_284090


namespace master_bath_size_l2840_284080

theorem master_bath_size (bedroom_size new_room_size : ℝ) 
  (h1 : bedroom_size = 309)
  (h2 : new_room_size = 918)
  (h3 : new_room_size = 2 * (bedroom_size + bathroom_size)) :
  bathroom_size = 150 :=
by
  sorry

end master_bath_size_l2840_284080


namespace final_sum_theorem_l2840_284048

theorem final_sum_theorem (T x y : ℝ) (h : x + y = T) :
  (2 * (2 * x + 4) + 2 * (3 * y + 4)) = 6 * T + 16 := by
  sorry

end final_sum_theorem_l2840_284048


namespace radical_conjugate_sum_product_l2840_284087

theorem radical_conjugate_sum_product (x y : ℝ) : 
  (x + Real.sqrt y) + (x - Real.sqrt y) = 10 →
  (x + Real.sqrt y) * (x - Real.sqrt y) = 9 →
  x + y = 21 := by
sorry

end radical_conjugate_sum_product_l2840_284087


namespace female_to_male_ratio_l2840_284032

/-- Represents a dog breed with the number of female and male puppies -/
structure Breed where
  name : String
  females : Nat
  males : Nat

/-- The litter of puppies -/
def litter : List Breed := [
  { name := "Golden Retriever", females := 2, males := 4 },
  { name := "Labrador", females := 1, males := 3 },
  { name := "Poodle", females := 3, males := 2 },
  { name := "Beagle", females := 1, males := 2 }
]

/-- The total number of female puppies in the litter -/
def totalFemales : Nat := litter.foldl (fun acc breed => acc + breed.females) 0

/-- The total number of male puppies in the litter -/
def totalMales : Nat := litter.foldl (fun acc breed => acc + breed.males) 0

/-- Theorem stating that the ratio of female to male puppies is 7:11 -/
theorem female_to_male_ratio :
  totalFemales = 7 ∧ totalMales = 11 := by sorry

end female_to_male_ratio_l2840_284032


namespace crayon_ratio_l2840_284082

theorem crayon_ratio (total : ℕ) (broken_percent : ℚ) (slightly_used : ℕ) 
  (h1 : total = 120)
  (h2 : broken_percent = 1/5)
  (h3 : slightly_used = 56) : 
  (total - (broken_percent * total).num - slightly_used) / total = 1/3 := by
sorry

end crayon_ratio_l2840_284082


namespace three_person_job_time_specific_job_time_l2840_284070

/-- The time taken to complete a job when multiple people work together -/
def time_to_complete (rates : List ℚ) : ℚ :=
  1 / (rates.sum)

/-- Proof that the time taken by three people working together is correct -/
theorem three_person_job_time (man_days son_days father_days : ℚ) 
  (man_days_pos : 0 < man_days) 
  (son_days_pos : 0 < son_days) 
  (father_days_pos : 0 < father_days) : 
  time_to_complete [1/man_days, 1/son_days, 1/father_days] = 
  1 / (1/man_days + 1/son_days + 1/father_days) :=
by sorry

/-- Application to the specific problem -/
theorem specific_job_time : 
  time_to_complete [1/20, 1/25, 1/20] = 100/14 :=
by sorry

end three_person_job_time_specific_job_time_l2840_284070


namespace equal_intercepts_values_l2840_284098

-- Define the line equation
def line_equation (a x y : ℝ) : Prop := a * x + y - 2 + a = 0

-- Define the condition for equal intercepts
def equal_intercepts (a : ℝ) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ line_equation a k 0 ∧ line_equation a 0 k

-- Theorem statement
theorem equal_intercepts_values :
  ∀ a : ℝ, equal_intercepts a ↔ (a = 2 ∨ a = 1) :=
sorry

end equal_intercepts_values_l2840_284098


namespace smaller_rss_better_fit_l2840_284049

/-- A regression model -/
structure RegressionModel where
  /-- The residual sum of squares of the model -/
  rss : ℝ
  /-- A measure of the model's fit quality -/
  fit_quality : ℝ

/-- The relationship between residual sum of squares and fit quality -/
axiom better_fit (m1 m2 : RegressionModel) :
  m1.rss < m2.rss → m1.fit_quality > m2.fit_quality

/-- Theorem: A smaller residual sum of squares indicates a better fit -/
theorem smaller_rss_better_fit (m1 m2 : RegressionModel) :
  m1.rss < m2.rss → m1.fit_quality > m2.fit_quality := by
  sorry


end smaller_rss_better_fit_l2840_284049


namespace expression_simplification_l2840_284046

theorem expression_simplification (a b c x y : ℝ) (h : c^2*b*x + c*a*y ≠ 0) :
  (c^2*b*x*(a^3*x^3 + 3*a^2*y^2 + b^3*y^3) + c*a*y*(a^3*x^3 + 3*b^3*x^3 + b^3*y^3)) / (c^2*b*x + c*a*y) = 
  a^3*x^3 + 3*a*b^3*x^3 + b^3*y^3 := by sorry

end expression_simplification_l2840_284046


namespace circle_center_sum_l2840_284065

/-- Given a circle with equation x^2 + y^2 = 4x - 6y + 9, 
    the sum of the x and y coordinates of its center is -1 -/
theorem circle_center_sum (x y : ℝ) : 
  (x^2 + y^2 = 4*x - 6*y + 9) → 
  ∃ (h k : ℝ), (x - h)^2 + (y - k)^2 = (h^2 + k^2 + 9 - 4*h + 6*k) ∧ h + k = -1 := by
  sorry

end circle_center_sum_l2840_284065


namespace smallest_n_exceeding_million_l2840_284093

theorem smallest_n_exceeding_million (n : ℕ) : ∃ (n : ℕ), n > 0 ∧ 
  (∀ k < n, (12 : ℝ) ^ ((k * (k + 1) : ℝ) / (2 * 13)) ≤ 1000000) ∧
  (12 : ℝ) ^ ((n * (n + 1) : ℝ) / (2 * 13)) > 1000000 ∧ n = 12 := by
  sorry

end smallest_n_exceeding_million_l2840_284093


namespace whisker_ratio_proof_l2840_284028

/-- The number of whiskers Princess Puff has -/
def princess_puff_whiskers : ℕ := 14

/-- The number of whiskers Catman Do has -/
def catman_do_whiskers : ℕ := 22

/-- The number of whiskers Catman Do is missing compared to the ratio -/
def missing_whiskers : ℕ := 6

/-- The ratio of Catman Do's whiskers to Princess Puff's whiskers -/
def whisker_ratio : ℚ := 2 / 1

theorem whisker_ratio_proof :
  (catman_do_whiskers + missing_whiskers : ℚ) / princess_puff_whiskers = whisker_ratio :=
sorry

end whisker_ratio_proof_l2840_284028


namespace mixture_volume_is_four_liters_l2840_284008

/-- Represents the weight in grams of 1 liter of ghee for a specific brand. -/
structure GheeWeight where
  weight : ℝ
  weight_positive : weight > 0

/-- Represents the volume ratio between two brands of ghee. -/
structure MixingRatio where
  a : ℝ
  b : ℝ
  a_positive : a > 0
  b_positive : b > 0

/-- Calculates the total volume of a ghee mixture given the weights and mixing ratio. -/
def calculate_mixture_volume (weight_a weight_b : GheeWeight) (ratio : MixingRatio) (total_weight_kg : ℝ) : ℝ :=
  sorry

/-- Theorem stating that the mixture volume is 4 liters given the problem conditions. -/
theorem mixture_volume_is_four_liters 
  (weight_a : GheeWeight)
  (weight_b : GheeWeight)
  (ratio : MixingRatio)
  (total_weight_kg : ℝ)
  (ha : weight_a.weight = 900)
  (hb : weight_b.weight = 750)
  (hr : ratio.a / ratio.b = 3 / 2)
  (hw : total_weight_kg = 3.36) :
  calculate_mixture_volume weight_a weight_b ratio total_weight_kg = 4 :=
sorry

end mixture_volume_is_four_liters_l2840_284008


namespace stating_min_positive_temperatures_l2840_284081

/-- Represents the number of participants at the conference -/
def num_participants : ℕ := 9

/-- Represents the total number of positive records -/
def positive_records : ℕ := 36

/-- Represents the total number of negative records -/
def negative_records : ℕ := 36

/-- Represents the minimum number of participants with positive temperatures -/
def min_positive_temps : ℕ := 3

/-- 
Theorem stating that given the conditions of the meteorological conference,
the minimum number of participants with positive temperatures is 3.
-/
theorem min_positive_temperatures : 
  ∀ y : ℕ, 
  y ≤ num_participants →
  y * (y - 1) + (num_participants - y) * (num_participants - 1 - y) = positive_records →
  y ≥ min_positive_temps :=
by sorry

end stating_min_positive_temperatures_l2840_284081


namespace fraction_subtraction_l2840_284058

theorem fraction_subtraction : (3 + 6 + 9) / (2 + 4 + 8) - (2 + 4 + 8) / (3 + 6 + 9) = 32 / 63 := by
  sorry

end fraction_subtraction_l2840_284058


namespace correct_number_of_selections_l2840_284097

/-- The number of ways to choose k elements from a set of n elements -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The number of ways to choose 5 players from a team of 16 players, 
    including a set of twins, where both twins cannot be chosen together -/
def choosePlayersWithTwinRestriction : ℕ := sorry

theorem correct_number_of_selections :
  choosePlayersWithTwinRestriction = 4004 := by sorry

end correct_number_of_selections_l2840_284097


namespace stadium_attendance_l2840_284027

theorem stadium_attendance (total : ℕ) (girls : ℕ) 
  (h1 : total = 600) 
  (h2 : girls = 240) : 
  total - ((total - girls) / 4 + girls / 8) = 480 := by
  sorry

end stadium_attendance_l2840_284027


namespace schoolchildren_count_l2840_284083

/-- The number of schoolchildren in the group -/
def S : ℕ := 135

/-- The number of buses initially provided -/
def n : ℕ := 6

/-- The number of schoolchildren in each bus after redistribution -/
def m : ℕ := 27

theorem schoolchildren_count :
  -- Initially, 22 people per bus with 3 left over
  S = 22 * n + 3 ∧
  -- After redistribution
  S = (n - 1) * m ∧
  -- No more than 18 buses
  n ≤ 18 ∧
  -- Each bus can hold no more than 36 people
  m ≤ 36 ∧
  -- m is greater than 22 (implied by the redistribution)
  m > 22 :=
by sorry

end schoolchildren_count_l2840_284083


namespace line_through_first_third_quadrants_positive_slope_l2840_284002

/-- A line passing through the first and third quadrants has a positive slope -/
theorem line_through_first_third_quadrants_positive_slope (k : ℝ) 
  (h1 : k ≠ 0) 
  (h2 : ∀ (x y : ℝ), y = k * x → ((x > 0 ∧ y > 0) ∨ (x < 0 ∧ y < 0))) : 
  k > 0 := by
  sorry

end line_through_first_third_quadrants_positive_slope_l2840_284002


namespace ratio_bounds_l2840_284085

theorem ratio_bounds (A C : ℕ) (n : ℕ) :
  (10 ≤ A ∧ A ≤ 99) →
  (10 ≤ C ∧ C ≤ 99) →
  (100 * A + C) / (A + C) = n →
  11 ≤ n ∧ n ≤ 90 := by
sorry

end ratio_bounds_l2840_284085


namespace original_number_is_four_l2840_284009

def is_correct_number (x : ℕ) : Prop :=
  (x + 3) % 5 = 2 ∧ 
  ((x + 5) + 3) % 5 = 2

theorem original_number_is_four : 
  ∃ (x : ℕ), is_correct_number x ∧ x = 4 :=
sorry

end original_number_is_four_l2840_284009


namespace cost_per_box_l2840_284023

/-- Calculates the cost per box for packaging a fine arts collection --/
theorem cost_per_box (box_length box_width box_height : ℝ)
  (total_volume min_total_cost : ℝ) :
  box_length = 20 ∧ box_width = 20 ∧ box_height = 12 ∧
  total_volume = 2160000 ∧ min_total_cost = 180 →
  (min_total_cost / (total_volume / (box_length * box_width * box_height))) = 0.4 := by
  sorry

end cost_per_box_l2840_284023


namespace johns_distance_is_285_l2840_284037

/-- Calculates the distance traveled given speed and time -/
def distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- John's total driving distance -/
def johns_total_distance : ℝ :=
  distance 45 2 + distance 30 0.5 + distance 60 1 + distance 20 1 + distance 50 2

/-- Theorem stating that John's total driving distance is 285 miles -/
theorem johns_distance_is_285 : johns_total_distance = 285 := by
  sorry

end johns_distance_is_285_l2840_284037


namespace escalator_time_l2840_284068

/-- The time taken for a person to cover the entire length of an escalator -/
theorem escalator_time (escalator_speed : ℝ) (person_speed : ℝ) (escalator_length : ℝ)
  (h1 : escalator_speed = 12)
  (h2 : person_speed = 2)
  (h3 : escalator_length = 196) :
  escalator_length / (escalator_speed + person_speed) = 14 :=
by sorry

end escalator_time_l2840_284068


namespace f_is_even_l2840_284034

def F (f : ℝ → ℝ) (x : ℝ) : ℝ := (x^3 - 2*x) * f x

def OddFunction (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def EvenFunction (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

def NotIdenticallyZero (f : ℝ → ℝ) : Prop := ∃ x, f x ≠ 0

theorem f_is_even (f : ℝ → ℝ) (h1 : OddFunction (F f)) (h2 : NotIdenticallyZero f) :
  EvenFunction f := by sorry

end f_is_even_l2840_284034


namespace equation_solution_l2840_284072

theorem equation_solution :
  ∃ x : ℝ, (1/8 : ℝ)^(3*x + 12) = (64 : ℝ)^(x + 4) ↔ x = -4 := by
  sorry

end equation_solution_l2840_284072


namespace correct_probability_l2840_284035

def total_rolls : ℕ := 12
def rolls_per_type : ℕ := 3
def num_types : ℕ := 4
def rolls_per_guest : ℕ := 4

def probability_correct_selection : ℚ :=
  (rolls_per_type * (rolls_per_type - 1) * (rolls_per_type - 2)) /
  (total_rolls * (total_rolls - 1) * (total_rolls - 2) * (total_rolls - 3))

theorem correct_probability :
  probability_correct_selection = 9 / 55 := by sorry

end correct_probability_l2840_284035


namespace prime_divisors_of_29_pow_p_plus_1_l2840_284030

theorem prime_divisors_of_29_pow_p_plus_1 (p : Nat) :
  Nat.Prime p ∧ p ∣ 29^p + 1 ↔ p = 2 ∨ p = 3 ∨ p = 5 := by
  sorry

end prime_divisors_of_29_pow_p_plus_1_l2840_284030


namespace largest_710_triple_correct_l2840_284006

/-- Converts a base-10 number to its base-7 representation as a list of digits -/
def toBase7 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) : List ℕ :=
      if m = 0 then [] else (m % 7) :: aux (m / 7)
    aux n |>.reverse

/-- Interprets a list of digits as a base-10 number -/
def fromDigits (digits : List ℕ) : ℕ :=
  digits.foldl (fun acc d => acc * 10 + d) 0

/-- Checks if a number is a 7-10 triple -/
def is710Triple (n : ℕ) : Prop :=
  fromDigits (toBase7 n) = 3 * n

/-- The largest 7-10 triple -/
def largest710Triple : ℕ := 335

theorem largest_710_triple_correct :
  is710Triple largest710Triple ∧ 
  ∀ m : ℕ, m > largest710Triple → ¬is710Triple m := by
  sorry

end largest_710_triple_correct_l2840_284006


namespace triangle_angle_calculation_l2840_284013

theorem triangle_angle_calculation (a b : ℝ) (A B : ℝ) :
  a = 1 →
  b = Real.sqrt 3 →
  A = π / 6 →
  (B = π / 3 ∨ B = 2 * π / 3) →
  Real.sin B = (b * Real.sin A) / a :=
by
  sorry

end triangle_angle_calculation_l2840_284013


namespace women_work_hours_l2840_284062

/-- Represents the work rate of men and women -/
structure WorkRate where
  men : ℕ
  men_days : ℕ
  men_hours : ℕ
  women : ℕ
  women_days : ℕ
  women_hours : ℕ
  women_to_men_ratio : ℚ

/-- The given work scenario -/
def work_scenario : WorkRate where
  men := 15
  men_days := 21
  men_hours := 8
  women := 21
  women_days := 36
  women_hours := 0  -- This is what we need to prove
  women_to_men_ratio := 2/3

/-- Theorem stating that the women's work hours per day is 5 -/
theorem women_work_hours (w : WorkRate) (h : w = work_scenario) : w.women_hours = 5 := by
  sorry


end women_work_hours_l2840_284062


namespace characterize_S_l2840_284066

-- Define the set of possible values for 1/x + 1/y
def S : Set ℝ := { z | ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x + y = 2 ∧ z = 1/x + 1/y }

-- Theorem statement
theorem characterize_S : S = Set.Ici 2 := by sorry

end characterize_S_l2840_284066


namespace dodecahedron_regions_count_l2840_284060

/-- The number of regions formed by the planes of a dodecahedron's faces --/
def num_regions : ℕ := 185

/-- The Euler characteristic for 3D space --/
def euler_characteristic : ℤ := -1

/-- The number of vertices formed by the intersecting planes --/
def num_vertices : ℕ := 52

/-- The number of edges formed by the intersecting planes --/
def num_edges : ℕ := 300

/-- The number of faces formed by the intersecting planes --/
def num_faces : ℕ := 432

/-- Theorem stating that the number of regions is correct given the Euler characteristic and the numbers of vertices, edges, and faces --/
theorem dodecahedron_regions_count :
  (num_vertices : ℤ) - (num_edges : ℤ) + (num_faces : ℤ) - (num_regions : ℤ) = euler_characteristic :=
by sorry

end dodecahedron_regions_count_l2840_284060


namespace parabola_from_parametric_equations_l2840_284004

/-- Given parametric equations for x and y in terms of t, prove that the relationship between x and y is quadratic. -/
theorem parabola_from_parametric_equations (t x y : ℝ) :
  x = 3^t - 2 ∧ y = 9^t - 4 * 3^t + 2*t - 4 →
  ∃ (a b c : ℝ), y = a * x^2 + b * x + c := by
  sorry

end parabola_from_parametric_equations_l2840_284004


namespace square_root_problem_l2840_284051

theorem square_root_problem (x : ℝ) : 
  Real.sqrt x = 3.87 → x = 14.9769 := by
  sorry

end square_root_problem_l2840_284051


namespace sharadek_word_guessing_l2840_284020

theorem sharadek_word_guessing (n : ℕ) (h : n ≤ 1000000) :
  ∃ (q : ℕ), q ≤ 20 ∧ 2^q ≥ n := by
  sorry

end sharadek_word_guessing_l2840_284020


namespace pasture_rent_is_175_l2840_284050

/-- Represents the rent share of a person -/
structure RentShare where
  oxen : ℕ
  months : ℕ
  share : ℚ

/-- Calculates the total rent of a pasture given the rent shares of three people -/
def totalRent (a b c : RentShare) : ℚ :=
  let totalOxenMonths := a.oxen * a.months + b.oxen * b.months + c.oxen * c.months
  (c.share * totalOxenMonths) / (c.oxen * c.months)

/-- Theorem stating that the total rent is 175 given the problem conditions -/
theorem pasture_rent_is_175 (a b c : RentShare)
  (ha : a.oxen = 10 ∧ a.months = 7)
  (hb : b.oxen = 12 ∧ b.months = 5)
  (hc : c.oxen = 15 ∧ c.months = 3 ∧ c.share = 45) :
  totalRent a b c = 175 := by
  sorry

#eval totalRent
  { oxen := 10, months := 7, share := 0 }
  { oxen := 12, months := 5, share := 0 }
  { oxen := 15, months := 3, share := 45 }

end pasture_rent_is_175_l2840_284050


namespace drought_pond_fill_time_l2840_284069

/-- Proves the time required to fill a pond under drought conditions -/
theorem drought_pond_fill_time 
  (pond_capacity : ℝ) 
  (normal_rate : ℝ) 
  (drought_factor : ℝ) 
  (h1 : pond_capacity = 200) 
  (h2 : normal_rate = 6) 
  (h3 : drought_factor = 2/3) : 
  pond_capacity / (normal_rate * drought_factor) = 50 := by
  sorry

#check drought_pond_fill_time

end drought_pond_fill_time_l2840_284069


namespace bales_stored_is_difference_solution_l2840_284045

/-- The number of bales Tim stored in the barn -/
def bales_stored (initial_bales final_bales : ℕ) : ℕ :=
  final_bales - initial_bales

/-- Theorem stating that the number of bales Tim stored is the difference between the final and initial number of bales -/
theorem bales_stored_is_difference (initial_bales final_bales : ℕ) 
  (h : final_bales ≥ initial_bales) :
  bales_stored initial_bales final_bales = final_bales - initial_bales :=
by
  sorry

/-- The solution to the specific problem -/
theorem solution : bales_stored 28 54 = 26 :=
by
  sorry

end bales_stored_is_difference_solution_l2840_284045


namespace interview_probabilities_l2840_284016

structure InterviewScenario where
  prob_A_pass : ℝ
  prob_B_pass : ℝ
  prob_C_pass : ℝ

def exactly_one_pass (s : InterviewScenario) : ℝ :=
  s.prob_A_pass * (1 - s.prob_B_pass) * (1 - s.prob_C_pass) +
  (1 - s.prob_A_pass) * s.prob_B_pass * (1 - s.prob_C_pass) +
  (1 - s.prob_A_pass) * (1 - s.prob_B_pass) * s.prob_C_pass

def at_most_one_sign (s : InterviewScenario) : ℝ :=
  (1 - s.prob_A_pass) * (1 - s.prob_B_pass * s.prob_C_pass) +
  s.prob_A_pass * (1 - s.prob_B_pass * s.prob_C_pass)

theorem interview_probabilities (s : InterviewScenario) 
  (h1 : s.prob_A_pass = 1/4)
  (h2 : s.prob_B_pass = 1/3)
  (h3 : s.prob_C_pass = 1/3) :
  exactly_one_pass s = 4/9 ∧ at_most_one_sign s = 8/9 := by
  sorry

#check interview_probabilities

end interview_probabilities_l2840_284016


namespace odd_integers_sum_product_l2840_284047

theorem odd_integers_sum_product (p q : ℕ) : 
  (p < 16 ∧ q < 16 ∧ Odd p ∧ Odd q) →
  (∃ (S : Finset ℕ), S = {n | ∃ (a b : ℕ), a < 16 ∧ b < 16 ∧ Odd a ∧ Odd b ∧ n = a * b + a + b} ∧ 
   Finset.card S = 36) :=
by sorry

end odd_integers_sum_product_l2840_284047


namespace tire_circumference_l2840_284084

/-- The circumference of a tire given its rotational speed and the car's velocity -/
theorem tire_circumference (rpm : ℝ) (velocity_kmh : ℝ) : rpm = 400 → velocity_kmh = 72 → 
  ∃ (circumference : ℝ), circumference = 3 := by
  sorry

end tire_circumference_l2840_284084


namespace mutually_exclusive_complementary_l2840_284044

-- Define the sample space
def Ω : Type := Unit

-- Define the events
def A : Set Ω := sorry
def B : Set Ω := sorry
def C : Set Ω := sorry
def D : Set Ω := sorry

-- Theorem for mutually exclusive events
theorem mutually_exclusive :
  (A ∩ B = ∅) ∧ (A ∩ C = ∅) ∧ (B ∩ D = ∅) :=
sorry

-- Theorem for complementary events
theorem complementary :
  B ∪ D = Set.univ ∧ B ∩ D = ∅ :=
sorry

end mutually_exclusive_complementary_l2840_284044


namespace third_quadrant_trig_expression_l2840_284031

theorem third_quadrant_trig_expression (α : Real) : 
  (α > π ∧ α < 3*π/2) →  -- α is in the third quadrant
  (2 * Real.sin α) / Real.sqrt (1 - Real.cos α ^ 2) = -2 := by
  sorry

end third_quadrant_trig_expression_l2840_284031


namespace river_throw_count_l2840_284064

/-- Represents the number of objects thrown by a person -/
structure ThrowCount where
  sticks : ℕ
  rocks : ℕ

/-- Calculates the total number of objects thrown -/
def total_objects (tc : ThrowCount) : ℕ :=
  tc.sticks + tc.rocks

theorem river_throw_count :
  let ted : ThrowCount := { sticks := 12, rocks := 18 }
  let bill : ThrowCount := { sticks := ted.sticks + 6, rocks := ted.rocks / 2 }
  let alice : ThrowCount := { sticks := ted.sticks / 2, rocks := bill.rocks * 3 }
  total_objects bill + total_objects alice = 60 := by
  sorry

end river_throw_count_l2840_284064


namespace binomial_coefficient_product_l2840_284000

theorem binomial_coefficient_product (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x, (1 - x)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  (a₀ + a₂ + a₄) * (a₁ + a₃ + a₅) = -256 := by
  sorry

end binomial_coefficient_product_l2840_284000


namespace square_with_prime_quotient_and_remainder_four_l2840_284014

theorem square_with_prime_quotient_and_remainder_four (n : ℕ) : 
  (n ^ 2 % 11 = 4 ∧ Nat.Prime ((n ^ 2 - 4) / 11)) ↔ n = 9 :=
sorry

end square_with_prime_quotient_and_remainder_four_l2840_284014


namespace watch_price_after_discounts_l2840_284077

/-- Calculates the final price of a watch after three consecutive discounts -/
def finalPrice (originalPrice : ℝ) (discount1 discount2 discount3 : ℝ) : ℝ :=
  originalPrice * (1 - discount1) * (1 - discount2) * (1 - discount3)

/-- Theorem stating that the final price of a 25000 rs watch after 15%, 20%, and 10% discounts is 15300 rs -/
theorem watch_price_after_discounts :
  finalPrice 25000 0.15 0.20 0.10 = 15300 := by
  sorry

end watch_price_after_discounts_l2840_284077


namespace mothers_carrots_count_l2840_284018

/-- The number of carrots Olivia picked -/
def olivias_carrots : ℕ := 20

/-- The total number of good carrots -/
def good_carrots : ℕ := 19

/-- The total number of bad carrots -/
def bad_carrots : ℕ := 15

/-- The number of carrots Olivia's mother picked -/
def mothers_carrots : ℕ := (good_carrots + bad_carrots) - olivias_carrots

theorem mothers_carrots_count : mothers_carrots = 14 := by
  sorry

end mothers_carrots_count_l2840_284018


namespace letter_Y_satisfies_condition_l2840_284021

def date_C (d : ℕ) : ℕ := d
def date_A (d : ℕ) : ℕ := d + 2
def date_B (d : ℕ) : ℕ := d + 8
def date_Y (d : ℕ) : ℕ := d + 10

theorem letter_Y_satisfies_condition (d : ℕ) :
  date_A d + date_B d = date_C d + date_Y d :=
by sorry

end letter_Y_satisfies_condition_l2840_284021


namespace inductive_inequality_l2840_284011

theorem inductive_inequality (x : ℝ) (n : ℕ) (h1 : x > 0) (h2 : n > 0) 
  (h3 : x + 1 / x ≥ 2) (h4 : x + 4 / x^2 ≥ 3) : 
  x + n^2 / x^n ≥ n + 1 := by
  sorry

end inductive_inequality_l2840_284011


namespace boys_neither_happy_nor_sad_l2840_284041

theorem boys_neither_happy_nor_sad (total_children : ℕ) (happy_children : ℕ) (sad_children : ℕ) 
  (neither_happy_nor_sad : ℕ) (total_boys : ℕ) (total_girls : ℕ) (happy_boys : ℕ) (sad_girls : ℕ) :
  total_children = 60 →
  happy_children = 30 →
  sad_children = 10 →
  neither_happy_nor_sad = 20 →
  total_boys = 19 →
  total_girls = 41 →
  happy_boys = 6 →
  sad_girls = 4 →
  total_children = happy_children + sad_children + neither_happy_nor_sad →
  total_children = total_boys + total_girls →
  (total_boys - (happy_boys + (sad_children - sad_girls))) = 7 :=
by sorry

end boys_neither_happy_nor_sad_l2840_284041


namespace cool_drink_volume_cool_drink_volume_proof_l2840_284071

theorem cool_drink_volume : ℝ → Prop :=
  fun initial_volume =>
    let initial_jasmine_ratio := 0.1
    let added_jasmine := 8
    let added_water := 12
    let final_jasmine_ratio := 0.16
    let final_volume := initial_volume + added_jasmine + added_water
    initial_jasmine_ratio * initial_volume + added_jasmine = final_jasmine_ratio * final_volume →
    initial_volume = 80

theorem cool_drink_volume_proof : ∃ (v : ℝ), cool_drink_volume v :=
  sorry

end cool_drink_volume_cool_drink_volume_proof_l2840_284071


namespace min_cars_with_racing_stripes_l2840_284036

theorem min_cars_with_racing_stripes 
  (total_cars : ℕ) 
  (cars_without_ac : ℕ) 
  (max_ac_no_stripes : ℕ) 
  (h1 : total_cars = 100)
  (h2 : cars_without_ac = 37)
  (h3 : max_ac_no_stripes = 49) :
  ∃ (min_racing_stripes : ℕ), 
    min_racing_stripes = 14 ∧ 
    (∀ (racing_stripes : ℕ), 
      racing_stripes ≥ min_racing_stripes → 
      ∃ (ac_with_stripes ac_no_stripes no_ac_with_stripes no_ac_no_stripes : ℕ),
        ac_with_stripes + ac_no_stripes + no_ac_with_stripes + no_ac_no_stripes = total_cars ∧
        no_ac_with_stripes + no_ac_no_stripes = cars_without_ac ∧
        ac_no_stripes ≤ max_ac_no_stripes ∧
        racing_stripes = ac_with_stripes + no_ac_with_stripes) :=
sorry

end min_cars_with_racing_stripes_l2840_284036


namespace multiply_24_99_l2840_284053

theorem multiply_24_99 : 24 * 99 = 2376 := by
  sorry

end multiply_24_99_l2840_284053


namespace weeks_passed_l2840_284095

/-- Represents the number of weeks passed -/
def weeks : ℕ := sorry

/-- Weekly fixed allowance in dollars -/
def fixed_allowance : ℚ := 20

/-- Extra earning per chore in dollars -/
def extra_chore_earning : ℚ := 1.5

/-- Average number of extra chores per week -/
def avg_extra_chores : ℕ := 15

/-- Total money earned in dollars -/
def total_money : ℚ := 425

/-- Theorem stating that the number of weeks passed is 10 -/
theorem weeks_passed : weeks = 10 := by sorry

end weeks_passed_l2840_284095


namespace circle_area_reduction_l2840_284055

theorem circle_area_reduction (r : ℝ) (h1 : π * r^2 = 36 * π) (h2 : r > 2) : 
  π * (r - 2)^2 = 16 * π := by
  sorry

end circle_area_reduction_l2840_284055


namespace same_sign_l2840_284029

theorem same_sign (a b c : ℝ) (h1 : (b/a) * (c/a) > 1) (h2 : (b/a) + (c/a) ≥ -2) :
  (a > 0 ∧ b > 0 ∧ c > 0) ∨ (a < 0 ∧ b < 0 ∧ c < 0) :=
sorry

end same_sign_l2840_284029


namespace hotel_profit_maximization_l2840_284073

/-- Represents the hotel profit maximization problem -/
theorem hotel_profit_maximization
  (total_rooms : ℕ)
  (base_price : ℝ)
  (price_increment : ℝ)
  (vacancy_increment : ℕ)
  (expense_per_room : ℝ)
  (max_profit_price : ℝ) :
  total_rooms = 50 →
  base_price = 180 →
  price_increment = 10 →
  vacancy_increment = 1 →
  expense_per_room = 20 →
  max_profit_price = 350 →
  ∀ price : ℝ,
    price ≥ base_price →
    let occupied_rooms := total_rooms - (price - base_price) / price_increment * vacancy_increment
    let profit := (price - expense_per_room) * occupied_rooms
    profit ≤ (max_profit_price - expense_per_room) * (total_rooms - (max_profit_price - base_price) / price_increment * vacancy_increment) :=
by
  sorry

#check hotel_profit_maximization

end hotel_profit_maximization_l2840_284073


namespace factor_expression_l2840_284039

theorem factor_expression (x : ℝ) : 3 * x^2 - 75 = 3 * (x - 5) * (x + 5) := by
  sorry

end factor_expression_l2840_284039


namespace determinant_evaluation_l2840_284005

theorem determinant_evaluation (a b : ℝ) : 
  Matrix.det !![1, a, b; 1, a+b, b; 1, a, a+b] = a * b := by
  sorry

end determinant_evaluation_l2840_284005


namespace martian_calendar_reform_not_feasible_l2840_284007

theorem martian_calendar_reform_not_feasible :
  ∀ (x y p q : ℕ),
  (26 * x + 29 * y = 687) →
  (27 * p + 31 * q = 687) →
  (p + q ≥ x + y) :=
by sorry

end martian_calendar_reform_not_feasible_l2840_284007


namespace claras_weight_l2840_284042

theorem claras_weight (alice_weight clara_weight : ℝ) 
  (h1 : alice_weight + clara_weight = 240)
  (h2 : clara_weight - alice_weight = clara_weight / 3) : 
  clara_weight = 144 := by
sorry

end claras_weight_l2840_284042


namespace jill_study_time_l2840_284043

/-- Calculates the total minutes Jill studies over 3 days given her study pattern -/
def total_study_minutes (day1_hours : ℕ) : ℕ :=
  let day2_hours := 2 * day1_hours
  let day3_hours := day2_hours - 1
  let total_hours := day1_hours + day2_hours + day3_hours
  total_hours * 60

/-- Proves that Jill's study pattern results in 540 minutes of total study time -/
theorem jill_study_time : total_study_minutes 2 = 540 := by
  sorry

end jill_study_time_l2840_284043


namespace quadratic_function_bound_l2840_284024

theorem quadratic_function_bound (a b : ℝ) :
  (∃ m : ℝ, |m^2 + a*m + b| ≤ 1/4 ∧ |(m+1)^2 + a*(m+1) + b| ≤ 1/4) →
  0 ≤ a^2 - 4*b ∧ a^2 - 4*b ≤ 2 := by
  sorry

end quadratic_function_bound_l2840_284024


namespace min_value_reciprocal_sum_l2840_284054

theorem min_value_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 4) :
  1 / x + 4 / y ≥ 9 / 4 :=
by sorry

end min_value_reciprocal_sum_l2840_284054


namespace largest_integer_in_set_l2840_284012

theorem largest_integer_in_set (a b c d : ℤ) : 
  a < b ∧ b < c ∧ c < d →  -- Four different integers
  (a + b + c + d) / 4 = 68 →  -- Average is 68
  a ≥ 5 →  -- Smallest integer is at least 5
  d = 254 :=  -- Largest integer is 254
by sorry

end largest_integer_in_set_l2840_284012


namespace mixed_oil_cost_theorem_l2840_284033

/-- Calculates the cost per litre of a mixed oil blend --/
def cost_per_litre_mixed_oil (volume_A volume_B volume_C : ℚ) 
                             (price_A price_B price_C : ℚ) : ℚ :=
  let total_cost := volume_A * price_A + volume_B * price_B + volume_C * price_C
  let total_volume := volume_A + volume_B + volume_C
  total_cost / total_volume

/-- The cost per litre of the mixed oil is approximately 54.52 --/
theorem mixed_oil_cost_theorem :
  ∃ (ε : ℚ), ε > 0 ∧ ε < 1/100 ∧ 
  abs (cost_per_litre_mixed_oil 10 5 8 54 66 48 - 54.52) < ε :=
sorry

end mixed_oil_cost_theorem_l2840_284033


namespace average_salary_is_8000_l2840_284063

def salary_a : ℕ := 8000
def salary_b : ℕ := 5000
def salary_c : ℕ := 11000
def salary_d : ℕ := 7000
def salary_e : ℕ := 9000

def num_people : ℕ := 5

def total_salary : ℕ := salary_a + salary_b + salary_c + salary_d + salary_e

theorem average_salary_is_8000 : (total_salary : ℚ) / num_people = 8000 := by
  sorry

end average_salary_is_8000_l2840_284063


namespace tangent_line_condition_l2840_284056

/-- The curve function f(x) = x³ - 3ax -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 3*a*x

/-- The derivative of f(x) with respect to x -/
def f_derivative (a : ℝ) (x : ℝ) : ℝ := 3*x^2 - 3*a

theorem tangent_line_condition (a : ℝ) :
  (∀ b : ℝ, ¬∃ x : ℝ, f a x = -x + b ∧ f_derivative a x = -1) →
  a < 1/3 := by
  sorry

end tangent_line_condition_l2840_284056


namespace extreme_value_condition_l2840_284075

/-- A function f: ℝ → ℝ has an extreme value at x₀ -/
def has_extreme_value (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  ∃ ε > 0, ∀ x ∈ Set.Ioo (x₀ - ε) (x₀ + ε), f x₀ ≤ f x ∨ f x₀ ≥ f x

/-- The statement that f'(x₀) = 0 is a necessary but not sufficient condition
    for f to have an extreme value at x₀ -/
theorem extreme_value_condition (f : ℝ → ℝ) (hf : Differentiable ℝ f) :
  (∀ x₀ : ℝ, has_extreme_value f x₀ → (deriv f) x₀ = 0) ∧
  ¬(∀ x₀ : ℝ, (deriv f) x₀ = 0 → has_extreme_value f x₀) :=
sorry

end extreme_value_condition_l2840_284075


namespace integer_points_in_triangle_l2840_284038

def count_points (n : ℕ) : ℕ :=
  (n + 1) * (n + 2) / 2

theorem integer_points_in_triangle : count_points 7 = 36 := by
  sorry

end integer_points_in_triangle_l2840_284038


namespace trajectory_equation_l2840_284022

/-- The equation of the trajectory of the center of a circle that passes through point A(2,0) 
    and is internally tangent to the circle x^2 + 4x + y^2 - 32 = 0 is x^2/9 + y^2/5 = 1 -/
theorem trajectory_equation : 
  ∀ (x y : ℝ), 
  (∃ (r : ℝ), (x - 2)^2 + y^2 = r^2) ∧ 
  (∃ (t : ℝ), (x - (-2))^2 + y^2 = (6 + t)^2 ∧ x^2 + 4*x + y^2 - 32 = 0) →
  x^2/9 + y^2/5 = 1 := by
sorry

end trajectory_equation_l2840_284022


namespace school_travel_time_l2840_284061

theorem school_travel_time (usual_rate : ℝ) (usual_time : ℝ) : 
  (usual_time > 0) →
  (7/6 * usual_rate * (usual_time - 5) = usual_rate * usual_time) →
  usual_time = 35 := by
sorry

end school_travel_time_l2840_284061


namespace parallelogram_construction_l2840_284059

-- Define the angle XOY
def Angle (O X Y : Point) : Prop := sorry

-- Define that a point is inside an angle
def InsideAngle (P : Point) (O X Y : Point) : Prop := sorry

-- Define that a point is on a line
def OnLine (P : Point) (A B : Point) : Prop := sorry

-- Define a parallelogram
def Parallelogram (A B C D : Point) : Prop := sorry

-- Define the theorem
theorem parallelogram_construction (O X Y A B : Point) 
  (h1 : Angle O X Y)
  (h2 : InsideAngle A O X Y)
  (h3 : InsideAngle B O X Y) :
  ∃ (C D : Point), 
    OnLine C O X ∧ 
    OnLine D O Y ∧ 
    Parallelogram A B C D :=
sorry

end parallelogram_construction_l2840_284059


namespace smallest_equal_purchase_l2840_284001

theorem smallest_equal_purchase (nuts : Nat) (bolts : Nat) (washers : Nat)
  (h_nuts : nuts = 13)
  (h_bolts : bolts = 8)
  (h_washers : washers = 17) :
  Nat.lcm (Nat.lcm nuts bolts) washers = 1768 := by
  sorry

end smallest_equal_purchase_l2840_284001


namespace distance_for_50L_800cc_l2840_284074

/-- The distance traveled using a given amount of diesel and engine capacity -/
def distance_traveled (diesel : ℝ) (engine_capacity : ℝ) : ℝ := sorry

/-- The volume of diesel required varies directly as the capacity of the engine -/
axiom diesel_engine_relation (d1 d2 e1 e2 dist : ℝ) :
  d1 / e1 = d2 / e2 → distance_traveled d1 e1 = distance_traveled d2 e2

theorem distance_for_50L_800cc :
  distance_traveled 50 800 = 6 :=
by
  have h1 : distance_traveled 100 1200 = 800 := sorry
  sorry

end distance_for_50L_800cc_l2840_284074


namespace all_conditions_imply_right_triangle_l2840_284057

structure Triangle (A B C : ℝ) :=
  (angle_sum : A + B + C = 180)

def is_right_triangle (t : Triangle A B C) : Prop :=
  A = 90 ∨ B = 90 ∨ C = 90

theorem all_conditions_imply_right_triangle 
  (t : Triangle A B C) : 
  (A + B = C) ∨ 
  (∃ (k : ℝ), k > 0 ∧ A = k ∧ B = 2*k ∧ C = 3*k) ∨ 
  (A = 90 - B) ∨ 
  (A = B - C) → 
  is_right_triangle t :=
sorry

end all_conditions_imply_right_triangle_l2840_284057


namespace regular_polygon_nine_sides_l2840_284091

-- Define a regular polygon
structure RegularPolygon where
  n : ℕ  -- number of sides
  a : ℝ  -- side length
  b : ℝ  -- longest diagonal
  c : ℝ  -- shortest diagonal
  h1 : n > 2  -- n must be greater than 2 for a polygon
  h2 : a > 0  -- side length must be positive
  h3 : b > c  -- longest diagonal is greater than shortest diagonal
  h4 : a = b - c  -- given condition

-- Theorem statement
theorem regular_polygon_nine_sides (p : RegularPolygon) : p.n = 9 := by
  sorry

end regular_polygon_nine_sides_l2840_284091


namespace yard_trees_l2840_284096

/-- The number of trees in a yard with given specifications -/
def numTrees (yardLength : ℕ) (treeDist : ℕ) : ℕ :=
  (yardLength / treeDist) + 1

theorem yard_trees :
  numTrees 180 18 = 12 :=
by sorry

end yard_trees_l2840_284096


namespace janes_calculation_l2840_284092

theorem janes_calculation (x y z : ℝ) 
  (h1 : x - (y - z) = 23) 
  (h2 : x - y - z = 7) : 
  x - y = 15 := by
sorry

end janes_calculation_l2840_284092


namespace sandwich_non_condiment_percentage_l2840_284067

theorem sandwich_non_condiment_percentage :
  let total_weight : ℝ := 150
  let condiment_weight : ℝ := 45
  let non_condiment_weight : ℝ := total_weight - condiment_weight
  let non_condiment_fraction : ℝ := non_condiment_weight / total_weight
  non_condiment_fraction * 100 = 70 := by
sorry

end sandwich_non_condiment_percentage_l2840_284067


namespace sin_n_squared_not_converge_to_zero_l2840_284026

/-- The sequence x_n = sin(n^2) does not converge to zero. -/
theorem sin_n_squared_not_converge_to_zero :
  ¬ (∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |Real.sin (n^2)| < ε) := by
  sorry

end sin_n_squared_not_converge_to_zero_l2840_284026


namespace jackie_breaks_l2840_284094

/-- Calculates the number of breaks Jackie takes during push-ups -/
def number_of_breaks (pushups_per_10_seconds : ℕ) (pushups_with_breaks : ℕ) (break_duration : ℕ) : ℕ :=
  let pushups_per_minute : ℕ := pushups_per_10_seconds * 6
  let missed_pushups : ℕ := pushups_per_minute - pushups_with_breaks
  let time_not_pushing : ℕ := missed_pushups * 10 / pushups_per_10_seconds
  time_not_pushing / break_duration

theorem jackie_breaks :
  number_of_breaks 5 22 8 = 2 :=
by sorry

end jackie_breaks_l2840_284094


namespace mall_entrance_exit_ways_l2840_284052

theorem mall_entrance_exit_ways (n : Nat) (h : n = 4) : 
  (n * (n - 1) : Nat) = 12 := by
  sorry

end mall_entrance_exit_ways_l2840_284052
