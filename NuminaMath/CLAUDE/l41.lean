import Mathlib

namespace NUMINAMATH_CALUDE_erased_number_l41_4161

/-- Given nine consecutive integers where the sum of eight of them is 1703, prove that the missing number is 214. -/
theorem erased_number (a : ℤ) (b : ℤ) (h1 : -4 ≤ b ∧ b ≤ 4) (h2 : 8*a - b = 1703) : a + b = 214 := by
  sorry

end NUMINAMATH_CALUDE_erased_number_l41_4161


namespace NUMINAMATH_CALUDE_square_sum_ge_twice_product_l41_4189

theorem square_sum_ge_twice_product {x y : ℝ} (h : x ≥ y) : x^2 + y^2 ≥ 2*x*y := by
  sorry

end NUMINAMATH_CALUDE_square_sum_ge_twice_product_l41_4189


namespace NUMINAMATH_CALUDE_congruence_problem_l41_4130

theorem congruence_problem (a b n : ℤ) : 
  a ≡ 25 [ZMOD 60] →
  b ≡ 85 [ZMOD 60] →
  150 ≤ n →
  n ≤ 241 →
  (a - b ≡ n [ZMOD 60]) ↔ (n = 180 ∨ n = 240) :=
by sorry

end NUMINAMATH_CALUDE_congruence_problem_l41_4130


namespace NUMINAMATH_CALUDE_smallest_common_multiple_of_8_and_6_l41_4133

theorem smallest_common_multiple_of_8_and_6 : 
  ∃ (n : ℕ), n > 0 ∧ 8 ∣ n ∧ 6 ∣ n ∧ ∀ (m : ℕ), m > 0 → 8 ∣ m → 6 ∣ m → n ≤ m := by
  sorry

end NUMINAMATH_CALUDE_smallest_common_multiple_of_8_and_6_l41_4133


namespace NUMINAMATH_CALUDE_sufficient_condition_not_necessary_condition_sufficient_but_not_necessary_l41_4181

/-- Proposition p: For all real x, x^2 - 4x + 2m ≥ 0 -/
def proposition_p (m : ℝ) : Prop :=
  ∀ x : ℝ, x^2 - 4*x + 2*m ≥ 0

/-- m ≥ 3 is a sufficient condition for proposition p -/
theorem sufficient_condition (m : ℝ) :
  m ≥ 3 → proposition_p m :=
sorry

/-- m ≥ 3 is not a necessary condition for proposition p -/
theorem not_necessary_condition :
  ∃ m : ℝ, m < 3 ∧ proposition_p m :=
sorry

/-- m ≥ 3 is a sufficient but not necessary condition for proposition p -/
theorem sufficient_but_not_necessary :
  (∀ m : ℝ, m ≥ 3 → proposition_p m) ∧
  (∃ m : ℝ, m < 3 ∧ proposition_p m) :=
sorry

end NUMINAMATH_CALUDE_sufficient_condition_not_necessary_condition_sufficient_but_not_necessary_l41_4181


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l41_4148

theorem arithmetic_mean_problem : ∃ (x y : ℝ), 
  ((x + 12) + y + 3*x + 18 + (3*x + 6)) / 5 = 30 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l41_4148


namespace NUMINAMATH_CALUDE_sphere_wedge_volume_l41_4165

/-- Given a sphere with circumference 18π inches, cut into 8 congruent wedges,
    the volume of one wedge is 121.5π cubic inches. -/
theorem sphere_wedge_volume (circumference : ℝ) (num_wedges : ℕ) :
  circumference = 18 * Real.pi →
  num_wedges = 8 →
  (1 / num_wedges : ℝ) * (4 / 3 * Real.pi * (circumference / (2 * Real.pi))^3) = 121.5 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_sphere_wedge_volume_l41_4165


namespace NUMINAMATH_CALUDE_point_minimizing_distance_sum_l41_4108

/-- The point that minimizes the sum of distances to two fixed points on a line --/
theorem point_minimizing_distance_sum 
  (M N P : ℝ × ℝ) 
  (hM : M = (1, 2)) 
  (hN : N = (4, 6)) 
  (hP : P.2 = P.1 - 1) 
  (h_min : ∀ Q : ℝ × ℝ, Q.2 = Q.1 - 1 → 
    dist P M + dist P N ≤ dist Q M + dist Q N) : 
  P = (17/5, 12/5) := by
  sorry

-- where
-- dist : ℝ × ℝ → ℝ × ℝ → ℝ 
-- represents the Euclidean distance between two points

end NUMINAMATH_CALUDE_point_minimizing_distance_sum_l41_4108


namespace NUMINAMATH_CALUDE_scooter_depreciation_l41_4151

theorem scooter_depreciation (initial_value : ℝ) : 
  (((initial_value * (3/4)) * (3/4)) = 22500) → initial_value = 40000 := by
  sorry

end NUMINAMATH_CALUDE_scooter_depreciation_l41_4151


namespace NUMINAMATH_CALUDE_dacid_weighted_average_score_l41_4147

/-- Calculates the weighted average score for a student given their marks and subject weightages --/
def weighted_average_score (
  english_mark : ℚ)
  (math_mark : ℚ)
  (physics_mark : ℚ)
  (chemistry_mark : ℚ)
  (biology_mark : ℚ)
  (cs_mark : ℚ)
  (sports_mark : ℚ)
  (english_weight : ℚ)
  (math_weight : ℚ)
  (physics_weight : ℚ)
  (chemistry_weight : ℚ)
  (biology_weight : ℚ)
  (cs_weight : ℚ)
  (sports_weight : ℚ) : ℚ :=
  english_mark * english_weight +
  math_mark * math_weight +
  physics_mark * physics_weight +
  chemistry_mark * chemistry_weight +
  biology_mark * biology_weight +
  (cs_mark * 100 / 150) * cs_weight +
  (sports_mark * 100 / 150) * sports_weight

/-- Theorem stating that Dacid's weighted average score is approximately 86.82 --/
theorem dacid_weighted_average_score :
  ∃ ε > 0, abs (weighted_average_score 96 95 82 97 95 88 83 0.25 0.20 0.10 0.15 0.10 0.15 0.05 - 86.82) < ε :=
by
  sorry

end NUMINAMATH_CALUDE_dacid_weighted_average_score_l41_4147


namespace NUMINAMATH_CALUDE_bread_per_sandwich_proof_l41_4136

/-- The number of sandwiches Sally eats on Saturday -/
def saturday_sandwiches : ℕ := 2

/-- The number of sandwiches Sally eats on Sunday -/
def sunday_sandwiches : ℕ := 1

/-- The total number of pieces of bread Sally eats across Saturday and Sunday -/
def total_bread : ℕ := 6

/-- The number of pieces of bread per sandwich -/
def bread_per_sandwich : ℕ := 2

theorem bread_per_sandwich_proof :
  saturday_sandwiches * bread_per_sandwich + sunday_sandwiches * bread_per_sandwich = total_bread :=
by sorry

end NUMINAMATH_CALUDE_bread_per_sandwich_proof_l41_4136


namespace NUMINAMATH_CALUDE_martha_final_cards_l41_4150

-- Define the initial number of cards Martha has
def initial_cards : ℝ := 76.0

-- Define the number of cards Martha gives away
def cards_given_away : ℝ := 3.0

-- Theorem statement
theorem martha_final_cards : 
  initial_cards - cards_given_away = 73.0 := by
  sorry

end NUMINAMATH_CALUDE_martha_final_cards_l41_4150


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l41_4131

theorem geometric_sequence_product (a : ℕ → ℝ) : 
  (∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r) →  -- geometric sequence condition
  (a 3 * a 3 - 5 * a 3 + 4 = 0) →             -- a_3 is a root of x^2 - 5x + 4 = 0
  (a 5 * a 5 - 5 * a 5 + 4 = 0) →             -- a_5 is a root of x^2 - 5x + 4 = 0
  (a 2 * a 4 * a 6 = 8 ∨ a 2 * a 4 * a 6 = -8) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l41_4131


namespace NUMINAMATH_CALUDE_problem_solution_l41_4119

theorem problem_solution : (((3⁻¹ : ℚ) + 7^3 - 2)⁻¹ * 7 : ℚ) = 21 / 1024 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l41_4119


namespace NUMINAMATH_CALUDE_inequality_proof_l41_4183

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  1 < (a / Real.sqrt (a^2 + b^2)) + (b / Real.sqrt (b^2 + c^2)) + (c / Real.sqrt (c^2 + a^2)) ∧
  (a / Real.sqrt (a^2 + b^2)) + (b / Real.sqrt (b^2 + c^2)) + (c / Real.sqrt (c^2 + a^2)) ≤ (3 * Real.sqrt 2) / 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l41_4183


namespace NUMINAMATH_CALUDE_car_value_after_depreciation_l41_4123

/-- Calculates the current value of a car given its initial price and depreciation rate. -/
def currentCarValue (initialPrice : ℝ) (depreciationRate : ℝ) : ℝ :=
  initialPrice * (1 - depreciationRate)

/-- Theorem stating that a car initially priced at $4000 with 30% depreciation is now worth $2800. -/
theorem car_value_after_depreciation :
  currentCarValue 4000 0.3 = 2800 := by
  sorry

end NUMINAMATH_CALUDE_car_value_after_depreciation_l41_4123


namespace NUMINAMATH_CALUDE_sequence_convergence_l41_4129

def converges (a : ℕ → ℝ) : Prop :=
  ∃ (l : ℝ), ∀ ε > 0, ∃ N, ∀ n ≥ N, |a n - l| < ε

theorem sequence_convergence
  (a : ℕ → ℝ)
  (h_pos : ∀ n, a n > 0)
  (h_ineq : ∀ n ≥ 2, a (n + 1) ≤ (a n * (a (n - 1))^2)^(1/3)) :
  converges a :=
sorry

end NUMINAMATH_CALUDE_sequence_convergence_l41_4129


namespace NUMINAMATH_CALUDE_rectangle_area_value_l41_4134

theorem rectangle_area_value (y : ℝ) : 
  y > 1 → 
  (3 : ℝ) * (y - 1) = 36 → 
  y = 13 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_value_l41_4134


namespace NUMINAMATH_CALUDE_athletes_division_l41_4142

theorem athletes_division (n : ℕ) (k : ℕ) : n = 10 ∧ k = 5 → (n.choose k) / 2 = 126 := by
  sorry

end NUMINAMATH_CALUDE_athletes_division_l41_4142


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_property_l41_4101

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Theorem: In an arithmetic sequence, if a_4 + a_8 = 16, then a_2 + a_10 = 16 -/
theorem arithmetic_sequence_sum_property
  (a : ℕ → ℝ) (h_arithmetic : arithmetic_sequence a) (h_sum : a 4 + a 8 = 16) :
  a 2 + a 10 = 16 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_property_l41_4101


namespace NUMINAMATH_CALUDE_total_distance_run_l41_4103

theorem total_distance_run (num_students : ℕ) (avg_distance : ℕ) (h1 : num_students = 18) (h2 : avg_distance = 106) :
  num_students * avg_distance = 1908 := by
  sorry

end NUMINAMATH_CALUDE_total_distance_run_l41_4103


namespace NUMINAMATH_CALUDE_jordan_running_time_l41_4122

/-- Given Steve's running time and distance, and Jordan's relative speed,
    calculate Jordan's time to run a specified distance. -/
theorem jordan_running_time
  (steve_distance : ℝ)
  (steve_time : ℝ)
  (jordan_relative_speed : ℝ)
  (jordan_distance : ℝ)
  (h1 : steve_distance = 6)
  (h2 : steve_time = 36)
  (h3 : jordan_relative_speed = 3)
  (h4 : jordan_distance = 8) :
  (jordan_distance * steve_time) / (steve_distance * jordan_relative_speed) = 24 :=
by sorry

end NUMINAMATH_CALUDE_jordan_running_time_l41_4122


namespace NUMINAMATH_CALUDE_regression_prediction_l41_4169

/-- Represents the regression equation y = mx + b -/
structure RegressionLine where
  m : ℝ
  b : ℝ

/-- Calculates the y-value for a given x using the regression line -/
def RegressionLine.predict (line : RegressionLine) (x : ℝ) : ℝ :=
  line.m * x + line.b

theorem regression_prediction 
  (line : RegressionLine)
  (h1 : line.m = 9.4)
  (h2 : line.predict 3.5 = 42)
  : line.predict 6 = 65.5 := by
  sorry

end NUMINAMATH_CALUDE_regression_prediction_l41_4169


namespace NUMINAMATH_CALUDE_final_selling_price_l41_4171

/-- Calculate the final selling price of three items with given costs, profit/loss percentages, discount, and tax. -/
theorem final_selling_price (cycle_cost scooter_cost motorbike_cost : ℚ)
  (cycle_loss_percent scooter_profit_percent motorbike_profit_percent : ℚ)
  (discount_percent tax_percent : ℚ) :
  let cycle_price := cycle_cost * (1 - cycle_loss_percent)
  let scooter_price := scooter_cost * (1 + scooter_profit_percent)
  let motorbike_price := motorbike_cost * (1 + motorbike_profit_percent)
  let total_price := cycle_price + scooter_price + motorbike_price
  let discounted_price := total_price * (1 - discount_percent)
  let final_price := discounted_price * (1 + tax_percent)
  cycle_cost = 2300 ∧
  scooter_cost = 12000 ∧
  motorbike_cost = 25000 ∧
  cycle_loss_percent = 0.30 ∧
  scooter_profit_percent = 0.25 ∧
  motorbike_profit_percent = 0.15 ∧
  discount_percent = 0.10 ∧
  tax_percent = 0.05 →
  final_price = 41815.20 := by
sorry

end NUMINAMATH_CALUDE_final_selling_price_l41_4171


namespace NUMINAMATH_CALUDE_no_perfect_power_in_sequence_l41_4143

/-- Represents a triple in the sequence -/
structure Triple where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Generates the next triple from the current one -/
def nextTriple (t : Triple) : Triple :=
  { a := t.a * t.b,
    b := t.b * t.c,
    c := t.c * t.a }

/-- Checks if a number is a perfect power -/
def isPerfectPower (n : ℕ) : Prop :=
  ∃ k m : ℕ, m ≥ 2 ∧ n = k^m

/-- The sequence of triples starting with (2,3,5) -/
def tripleSequence : ℕ → Triple
  | 0 => { a := 2, b := 3, c := 5 }
  | n + 1 => nextTriple (tripleSequence n)

/-- Theorem: No number in any triple of the sequence is a perfect power -/
theorem no_perfect_power_in_sequence :
  ∀ n : ℕ, ¬(isPerfectPower (tripleSequence n).a ∨
            isPerfectPower (tripleSequence n).b ∨
            isPerfectPower (tripleSequence n).c) :=
by
  sorry


end NUMINAMATH_CALUDE_no_perfect_power_in_sequence_l41_4143


namespace NUMINAMATH_CALUDE_triangle_properties_l41_4179

theorem triangle_properties (A B C : ℝ) (a b c : ℝ) (R : ℝ) :
  R = Real.sqrt 3 →
  (2 * Real.sin A - Real.sin C) / Real.sin B = Real.cos C / Real.cos B →
  (∀ x y z, x + y + z = Real.pi → Real.sin x / a = Real.sin y / b) →
  (b = 2 * R * Real.sin B) →
  (b^2 = a^2 + c^2 - 2 * a * c * Real.cos B) →
  (∃ (S : ℝ), S = 1/2 * a * c * Real.sin B) →
  (B = Real.pi / 3 ∧ 
   b = 3 ∧ 
   (∃ (S_max : ℝ), S_max = 9 * Real.sqrt 3 / 4 ∧ 
     (∀ S, S ≤ S_max) ∧ 
     (S = S_max ↔ a = c ∧ a = 3))) := by sorry

end NUMINAMATH_CALUDE_triangle_properties_l41_4179


namespace NUMINAMATH_CALUDE_marks_reading_increase_l41_4118

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- Mark's current daily reading time in hours -/
def current_daily_reading : ℕ := 2

/-- Mark's desired weekly reading time in hours -/
def desired_weekly_reading : ℕ := 18

/-- Calculate the increase in Mark's weekly reading time -/
def reading_time_increase : ℕ :=
  desired_weekly_reading - (current_daily_reading * days_in_week)

/-- Theorem stating that Mark's weekly reading time increase is 4 hours -/
theorem marks_reading_increase : reading_time_increase = 4 := by
  sorry

end NUMINAMATH_CALUDE_marks_reading_increase_l41_4118


namespace NUMINAMATH_CALUDE_geometric_series_sum_l41_4104

def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem geometric_series_sum :
  geometric_sum (1/4) (1/4) 6 = 4095/12288 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l41_4104


namespace NUMINAMATH_CALUDE_range_of_k_l41_4194

def proposition_p (k : ℝ) : Prop :=
  ∀ x : ℝ, x^2 + k*x + 2*k + 5 ≥ 0

def proposition_q (k : ℝ) : Prop :=
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a ≠ b ∧
  ∀ x y : ℝ, x^2 / (4-k) + y^2 / (k-1) = 1 ↔ (x/a)^2 + (y/b)^2 = 1

theorem range_of_k (k : ℝ) :
  (proposition_q k ↔ k ∈ Set.Ioo 1 (5/2)) ∧
  ((proposition_p k ∨ proposition_q k) ∧ ¬(proposition_p k ∧ proposition_q k) ↔
   k ∈ Set.Icc (-2) 1 ∪ Set.Icc (5/2) 10) :=
sorry

end NUMINAMATH_CALUDE_range_of_k_l41_4194


namespace NUMINAMATH_CALUDE_fish_tank_water_calculation_l41_4173

theorem fish_tank_water_calculation (initial_water : ℝ) (added_water : ℝ) : 
  initial_water = 7.75 → added_water = 7 → initial_water + added_water = 14.75 := by
  sorry

end NUMINAMATH_CALUDE_fish_tank_water_calculation_l41_4173


namespace NUMINAMATH_CALUDE_shopping_money_l41_4191

theorem shopping_money (remaining_amount : ℝ) (spent_percentage : ℝ) (initial_amount : ℝ) :
  remaining_amount = 217 →
  spent_percentage = 30 →
  remaining_amount = initial_amount * (1 - spent_percentage / 100) →
  initial_amount = 310 := by
  sorry

end NUMINAMATH_CALUDE_shopping_money_l41_4191


namespace NUMINAMATH_CALUDE_sophia_age_in_eight_years_l41_4158

/-- Represents the ages of individuals in the problem -/
structure Ages where
  jeremy : ℕ
  sebastian : ℕ
  isabella : ℕ
  sophia : ℕ
  lucas : ℕ
  olivia : ℕ
  ethan : ℕ

/-- The conditions of the problem -/
def problem_conditions (ages : Ages) : Prop :=
  (ages.jeremy + ages.sebastian + ages.isabella + ages.sophia + ages.lucas + ages.olivia + ages.ethan + 42 = 495) ∧
  (ages.sebastian = ages.jeremy + 4) ∧
  (ages.isabella = ages.sebastian - 3) ∧
  (ages.sophia = 2 * ages.lucas) ∧
  (ages.lucas = ages.jeremy - 5) ∧
  (ages.olivia = ages.isabella) ∧
  (ages.ethan = ages.olivia / 2) ∧
  (ages.jeremy + ages.sebastian + ages.isabella + 6 = 150) ∧
  (ages.jeremy = 40)

/-- The theorem to be proved -/
theorem sophia_age_in_eight_years (ages : Ages) :
  problem_conditions ages → ages.sophia + 8 = 78 := by
  sorry


end NUMINAMATH_CALUDE_sophia_age_in_eight_years_l41_4158


namespace NUMINAMATH_CALUDE_point_on_inverse_proportion_in_first_quadrant_l41_4193

/-- Given that point M(3,m) lies on the graph of y = 6/x, prove that M is in the first quadrant -/
theorem point_on_inverse_proportion_in_first_quadrant (m : ℝ) : 
  m = 6 / 3 → m > 0 := by sorry

end NUMINAMATH_CALUDE_point_on_inverse_proportion_in_first_quadrant_l41_4193


namespace NUMINAMATH_CALUDE_square_root_of_a_minus_b_l41_4195

theorem square_root_of_a_minus_b (a b : ℝ) (h1 : |a| = 3) (h2 : Real.sqrt (b^2) = 4) (h3 : a > b) :
  Real.sqrt (a - b) = Real.sqrt 7 ∨ Real.sqrt (a - b) = 1 := by
  sorry

end NUMINAMATH_CALUDE_square_root_of_a_minus_b_l41_4195


namespace NUMINAMATH_CALUDE_men_in_room_l41_4145

theorem men_in_room (initial_men : ℕ) (initial_women : ℕ) : 
  initial_men * 5 = initial_women * 4 →
  (2 * (initial_women - 3) = 24) →
  (initial_men + 2 = 14) :=
by
  sorry

#check men_in_room

end NUMINAMATH_CALUDE_men_in_room_l41_4145


namespace NUMINAMATH_CALUDE_f_negative_five_equals_negative_five_l41_4190

/-- Given a function f(x) = a*sin(x) + b*tan(x) + 1 where f(5) = 7, prove that f(-5) = -5 -/
theorem f_negative_five_equals_negative_five 
  (a b : ℝ) 
  (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = a * Real.sin x + b * Real.tan x + 1) 
  (h2 : f 5 = 7) : 
  f (-5) = -5 := by
sorry

end NUMINAMATH_CALUDE_f_negative_five_equals_negative_five_l41_4190


namespace NUMINAMATH_CALUDE_figure_segments_length_l41_4146

theorem figure_segments_length 
  (rectangle_length : ℝ) 
  (rectangle_breadth : ℝ) 
  (square_side : ℝ) 
  (h1 : rectangle_length = 10) 
  (h2 : rectangle_breadth = 6) 
  (h3 : square_side = 4) :
  square_side + 2 * rectangle_length + rectangle_breadth / 2 = 27 :=
by sorry

end NUMINAMATH_CALUDE_figure_segments_length_l41_4146


namespace NUMINAMATH_CALUDE_weight_solution_l41_4172

def weight_problem (A B C D E : ℝ) : Prop :=
  let avg_ABC := (A + B + C) / 3
  let avg_ABCD := (A + B + C + D) / 4
  let avg_BCDE := (B + C + D + E) / 4
  avg_ABC = 50 ∧ 
  avg_ABCD = 53 ∧ 
  E = D + 3 ∧ 
  avg_BCDE = 51 →
  A = 8

theorem weight_solution :
  ∀ A B C D E : ℝ, weight_problem A B C D E :=
by sorry

end NUMINAMATH_CALUDE_weight_solution_l41_4172


namespace NUMINAMATH_CALUDE_evaluate_64_to_5_6_l41_4125

theorem evaluate_64_to_5_6 : (64 : ℝ) ^ (5/6) = 32 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_64_to_5_6_l41_4125


namespace NUMINAMATH_CALUDE_f_2023_of_5_eq_57_l41_4137

def f (x : ℚ) : ℚ := (2 + x) / (1 - 2 * x)

def f_n : ℕ → ℚ → ℚ
  | 0, x => x
  | n + 1, x => f (f_n n x)

theorem f_2023_of_5_eq_57 : f_n 2023 5 = 57 := by
  sorry

end NUMINAMATH_CALUDE_f_2023_of_5_eq_57_l41_4137


namespace NUMINAMATH_CALUDE_intersection_M_N_l41_4184

def M : Set ℝ := {x | x^2 > 1}
def N : Set ℝ := {-2, -1, 0, 1, 2}

theorem intersection_M_N : M ∩ N = {-2, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l41_4184


namespace NUMINAMATH_CALUDE_bag_probability_l41_4124

theorem bag_probability (n : ℕ) : 
  (5 : ℚ) / (n + 5) = 1 / 3 → n = 10 := by
  sorry

end NUMINAMATH_CALUDE_bag_probability_l41_4124


namespace NUMINAMATH_CALUDE_quadrilateral_area_l41_4135

/-- The area of a quadrilateral ABCD with given diagonal and offsets -/
theorem quadrilateral_area (BD AC : ℝ) (offset_A offset_C : ℝ) :
  BD = 28 →
  offset_A = 8 →
  offset_C = 2 →
  (1/2 * BD * offset_A) + (1/2 * BD * offset_C) = 140 :=
by sorry

end NUMINAMATH_CALUDE_quadrilateral_area_l41_4135


namespace NUMINAMATH_CALUDE_exponent_sum_l41_4185

theorem exponent_sum (x m n : ℝ) (hm : x^m = 6) (hn : x^n = 2) : x^(m+n) = 12 := by
  sorry

end NUMINAMATH_CALUDE_exponent_sum_l41_4185


namespace NUMINAMATH_CALUDE_rajas_income_l41_4141

theorem rajas_income (household_percent : ℝ) (clothes_percent : ℝ) (medicines_percent : ℝ)
  (transportation_percent : ℝ) (entertainment_percent : ℝ) (savings : ℝ) (income : ℝ) :
  household_percent = 0.45 →
  clothes_percent = 0.12 →
  medicines_percent = 0.08 →
  transportation_percent = 0.15 →
  entertainment_percent = 0.10 →
  savings = 5000 →
  household_percent * income + clothes_percent * income + medicines_percent * income +
    transportation_percent * income + entertainment_percent * income + savings = income →
  income = 50000 := by
  sorry

end NUMINAMATH_CALUDE_rajas_income_l41_4141


namespace NUMINAMATH_CALUDE_max_inscribed_right_triangles_l41_4144

/-- Represents an ellipse with equation x^2 + a^2 * y^2 = a^2 where a > 1 -/
structure Ellipse where
  a : ℝ
  h_a_gt_one : a > 1

/-- Represents a right triangle inscribed in the ellipse with C(0, 1) as the right angle -/
structure InscribedRightTriangle (e : Ellipse) where
  A : ℝ × ℝ
  B : ℝ × ℝ
  h_on_ellipse_A : A.1^2 + e.a^2 * A.2^2 = e.a^2
  h_on_ellipse_B : B.1^2 + e.a^2 * B.2^2 = e.a^2
  h_right_angle : (A.1 - 0) * (B.1 - 0) + (A.2 - 1) * (B.2 - 1) = 0

/-- The theorem stating the maximum number of inscribed right triangles -/
theorem max_inscribed_right_triangles (e : Ellipse) : 
  (∃ (n : ℕ), ∀ (m : ℕ), (∃ (f : Fin m → InscribedRightTriangle e), Function.Injective f) → m ≤ n) ∧ 
  (∃ (f : Fin 3 → InscribedRightTriangle e), Function.Injective f) := by
  sorry

end NUMINAMATH_CALUDE_max_inscribed_right_triangles_l41_4144


namespace NUMINAMATH_CALUDE_remainder_problem_l41_4188

theorem remainder_problem (D : ℕ) (R : ℕ) (h1 : D > 0) 
  (h2 : 242 % D = 4) 
  (h3 : 698 % D = R) 
  (h4 : 940 % D = 7) : R = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l41_4188


namespace NUMINAMATH_CALUDE_total_hours_worked_l41_4128

def ordinary_rate : ℚ := 60 / 100
def overtime_rate : ℚ := 90 / 100
def total_earnings : ℚ := 3240 / 100
def overtime_hours : ℕ := 8

theorem total_hours_worked : ℕ := by
  sorry

#check total_hours_worked = 50

end NUMINAMATH_CALUDE_total_hours_worked_l41_4128


namespace NUMINAMATH_CALUDE_range_of_a_l41_4163

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def monotone_increasing_on (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ {x y}, x ∈ s → y ∈ s → x ≤ y → f x ≤ f y

theorem range_of_a (f : ℝ → ℝ) (a : ℝ) 
  (h_even : is_even_function f)
  (h_mono : monotone_increasing_on f (Set.Ici 0))
  (h_ineq : f (a + 1) ≤ f 4) :
  -5 ≤ a ∧ a ≤ 3 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l41_4163


namespace NUMINAMATH_CALUDE_difference_is_perfect_square_l41_4117

theorem difference_is_perfect_square (m n : ℕ+) 
  (h : 2001 * m^2 + m = 2002 * n^2 + n) : 
  ∃ k : ℕ, (m : ℤ) - (n : ℤ) = k^2 :=
sorry

end NUMINAMATH_CALUDE_difference_is_perfect_square_l41_4117


namespace NUMINAMATH_CALUDE_least_xy_value_l41_4106

theorem least_xy_value (x y : ℕ+) (h : (1 : ℚ) / x + (1 : ℚ) / (3 * y) = (1 : ℚ) / 8) :
  (x * y : ℕ) ≥ 96 ∧ ∃ (a b : ℕ+), (a : ℚ) / b + (1 : ℚ) / (3 * b) = (1 : ℚ) / 8 ∧ (a * b : ℕ) = 96 :=
sorry

end NUMINAMATH_CALUDE_least_xy_value_l41_4106


namespace NUMINAMATH_CALUDE_pyramid_volume_scaling_l41_4110

theorem pyramid_volume_scaling (V₀ : ℝ) (l w h : ℝ) : 
  V₀ = (1/3) * l * w * h → 
  V₀ = 60 → 
  (1/3) * (3*l) * (4*w) * (2*h) = 1440 := by sorry

end NUMINAMATH_CALUDE_pyramid_volume_scaling_l41_4110


namespace NUMINAMATH_CALUDE_graph_intersection_sum_l41_4127

/-- The equation of the graph -/
def graph_equation (x y : ℝ) : Prop :=
  (x^2 + y^2 - 2*x)^2 = 2*(x^2 + y^2)^2

/-- The number of points where the graph meets the x-axis -/
def p : ℕ :=
  3  -- This is given as a fact from the problem, not derived from the solution

/-- The number of points where the graph meets the y-axis -/
def q : ℕ :=
  1  -- This is given as a fact from the problem, not derived from the solution

/-- The theorem to be proved -/
theorem graph_intersection_sum : 100 * p + 100 * q = 400 := by
  sorry

end NUMINAMATH_CALUDE_graph_intersection_sum_l41_4127


namespace NUMINAMATH_CALUDE_sqrt_difference_inequality_l41_4140

theorem sqrt_difference_inequality (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  Real.sqrt a - Real.sqrt b < Real.sqrt (a - b) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_inequality_l41_4140


namespace NUMINAMATH_CALUDE_precy_age_l41_4126

/-- Represents the ages of Alex and Precy -/
structure Ages where
  alex : ℕ
  precy : ℕ

/-- The conditions given in the problem -/
def problem_conditions (ages : Ages) : Prop :=
  ages.alex = 15 ∧
  (ages.alex + 3) = 3 * (ages.precy + 3) ∧
  (ages.alex - 1) = 7 * (ages.precy - 1)

/-- The theorem stating that under the given conditions, Precy's age is 3 -/
theorem precy_age (ages : Ages) : problem_conditions ages → ages.precy = 3 := by
  sorry

end NUMINAMATH_CALUDE_precy_age_l41_4126


namespace NUMINAMATH_CALUDE_f_value_at_negative_pi_third_l41_4160

noncomputable def f (a b x : ℝ) : ℝ :=
  a * (Real.cos x)^2 - b * Real.sin x * Real.cos x - a / 2

theorem f_value_at_negative_pi_third (a b : ℝ) :
  (∃ (x : ℝ), f a b x ≤ 1/2) ∧
  (f a b (π/3) = Real.sqrt 3 / 4) →
  (f a b (-π/3) = 0 ∨ f a b (-π/3) = -Real.sqrt 3 / 4) :=
by sorry

end NUMINAMATH_CALUDE_f_value_at_negative_pi_third_l41_4160


namespace NUMINAMATH_CALUDE_linear_equation_exponents_l41_4115

/-- A function to represent the linearity of an equation in two variables -/
def is_linear_two_var (f : ℝ → ℝ → ℝ) : Prop :=
  ∃ (m n c : ℝ), ∀ x y, f x y = m * x + n * y + c

/-- The main theorem -/
theorem linear_equation_exponents :
  ∀ a b : ℝ,
  (is_linear_two_var (λ x y => x^(a-3) + y^(b-1))) →
  (a = 4 ∧ b = 2) :=
by sorry

end NUMINAMATH_CALUDE_linear_equation_exponents_l41_4115


namespace NUMINAMATH_CALUDE_sum_of_numbers_l41_4199

theorem sum_of_numbers (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x * y = 9375) (h4 : y / x = 15) : x + y = 400 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_numbers_l41_4199


namespace NUMINAMATH_CALUDE_jos_number_l41_4154

theorem jos_number (n : ℕ) : 
  (∃ k l : ℕ, n = 9 * k - 2 ∧ n = 6 * l - 4) ∧ 
  n < 100 ∧ 
  (∀ m : ℕ, m < 100 → (∃ k' l' : ℕ, m = 9 * k' - 2 ∧ m = 6 * l' - 4) → m ≤ n) → 
  n = 86 := by
sorry

end NUMINAMATH_CALUDE_jos_number_l41_4154


namespace NUMINAMATH_CALUDE_light_distance_half_year_l41_4180

/-- The speed of light in kilometers per second -/
def speed_of_light : ℝ := 299792

/-- The number of days in half a year -/
def half_year_days : ℝ := 182.5

/-- The distance light travels in half a year -/
def light_distance : ℝ := speed_of_light * half_year_days * 24 * 3600

theorem light_distance_half_year :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 * 10^12 ∧ 
  |light_distance - 4.73 * 10^12| < ε :=
sorry

end NUMINAMATH_CALUDE_light_distance_half_year_l41_4180


namespace NUMINAMATH_CALUDE_normal_distribution_probability_l41_4162

/-- The normal distribution with mean μ and standard deviation σ -/
noncomputable def normal_distribution (μ σ : ℝ) : ℝ → ℝ := sorry

/-- The probability density function of a normal distribution -/
noncomputable def normal_pdf (μ σ : ℝ) : ℝ → ℝ := sorry

/-- The cumulative distribution function of a normal distribution -/
noncomputable def normal_cdf (μ σ : ℝ) : ℝ → ℝ := sorry

/-- The probability of a random variable X falling within an interval [a, b] -/
noncomputable def prob_interval (μ σ : ℝ) (a b : ℝ) : ℝ :=
  normal_cdf μ σ b - normal_cdf μ σ a

theorem normal_distribution_probability (μ σ : ℝ) :
  (normal_pdf μ σ 0 = 1 / (3 * Real.sqrt (2 * Real.pi))) →
  (prob_interval μ σ (μ - σ) (μ + σ) = 0.6826) →
  (prob_interval μ σ (μ - 2*σ) (μ + 2*σ) = 0.9544) →
  (prob_interval μ σ 3 6 = 0.1359) := by sorry

end NUMINAMATH_CALUDE_normal_distribution_probability_l41_4162


namespace NUMINAMATH_CALUDE_geometry_rhyme_probability_l41_4155

def geometry_letters : Finset Char := {'G', 'E', 'O', 'M', 'E', 'T', 'R', 'Y'}
def rhyme_letters : Finset Char := {'R', 'H', 'Y', 'M', 'E'}

theorem geometry_rhyme_probability :
  (geometry_letters ∩ rhyme_letters).card / geometry_letters.card = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_geometry_rhyme_probability_l41_4155


namespace NUMINAMATH_CALUDE_ruth_apples_l41_4121

/-- The number of apples Ruth ends up with after a series of events -/
def final_apples (initial : ℕ) (shared : ℕ) (gift : ℕ) : ℕ :=
  let remaining := initial - shared
  let after_sister := remaining - remaining / 2
  after_sister + gift

/-- Theorem stating that Ruth ends up with 105 apples -/
theorem ruth_apples : final_apples 200 5 7 = 105 := by
  sorry

end NUMINAMATH_CALUDE_ruth_apples_l41_4121


namespace NUMINAMATH_CALUDE_hyperbola_a_plus_h_l41_4192

/-- A hyperbola with given asymptotes and a point it passes through -/
structure Hyperbola where
  /-- First asymptote equation: y = m₁x + c₁ -/
  m₁ : ℝ
  c₁ : ℝ
  /-- Second asymptote equation: y = m₂x + c₂ -/
  m₂ : ℝ
  c₂ : ℝ
  /-- Point that the hyperbola passes through -/
  p : ℝ × ℝ

/-- The standard form of a hyperbola: (y-k)^2/a^2 - (x-h)^2/b^2 = 1 -/
structure StandardForm where
  a : ℝ
  b : ℝ
  h : ℝ
  k : ℝ
  a_pos : a > 0
  b_pos : b > 0

/-- Theorem stating the value of a + h for the given hyperbola -/
theorem hyperbola_a_plus_h (hyp : Hyperbola) 
    (h : hyp.m₁ = 3 ∧ hyp.c₁ = 6 ∧ hyp.m₂ = -3 ∧ hyp.c₂ = 2 ∧ hyp.p = (1, 8)) :
    ∃ (sf : StandardForm), sf.a + sf.h = (Real.sqrt 119 - 2) / 3 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_a_plus_h_l41_4192


namespace NUMINAMATH_CALUDE_purely_imaginary_m_l41_4176

theorem purely_imaginary_m (m : ℝ) : 
  (m^2 - m : ℂ) + 3*I = (0 : ℝ) + I * (3 : ℝ) → m = 0 ∨ m = 1 := by
  sorry

end NUMINAMATH_CALUDE_purely_imaginary_m_l41_4176


namespace NUMINAMATH_CALUDE_solve_product_equation_l41_4174

theorem solve_product_equation : 
  ∀ x : ℝ, 6 * (x - 3) * (x + 5) = 0 ↔ x = -5 ∨ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_product_equation_l41_4174


namespace NUMINAMATH_CALUDE_hexagon_area_l41_4105

theorem hexagon_area (s : ℝ) (t : ℝ) : 
  s > 0 → t > 0 →
  (4 * s = 6 * t) →  -- Equal perimeters
  (s^2 = 16) →       -- Area of square is 16
  (6 * (t^2 * Real.sqrt 3) / 4) = (64 * Real.sqrt 3) / 3 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_area_l41_4105


namespace NUMINAMATH_CALUDE_num_ways_to_achieve_18_with_5_dice_l41_4109

/-- The number of faces on each die -/
def numFaces : ℕ := 6

/-- The number of dice thrown -/
def numDice : ℕ := 5

/-- The target sum we're aiming for -/
def targetSum : ℕ := 18

/-- A function that calculates the number of ways to achieve the target sum -/
def numWaysToAchieveSum (faces : ℕ) (dice : ℕ) (sum : ℕ) : ℕ := sorry

theorem num_ways_to_achieve_18_with_5_dice : 
  numWaysToAchieveSum numFaces numDice targetSum = 651 := by sorry

end NUMINAMATH_CALUDE_num_ways_to_achieve_18_with_5_dice_l41_4109


namespace NUMINAMATH_CALUDE_neil_fraction_of_packs_l41_4187

def total_marbles : ℕ := 400
def marbles_per_pack : ℕ := 10
def packs_kept_by_leo : ℕ := 25
def fraction_to_manny : ℚ := 1/4

theorem neil_fraction_of_packs (total_packs : ℕ) (packs_given_away : ℕ) 
  (packs_to_manny : ℕ) (packs_to_neil : ℕ) :
  total_packs = total_marbles / marbles_per_pack →
  packs_given_away = total_packs - packs_kept_by_leo →
  packs_to_manny = ⌊(fraction_to_manny * packs_given_away : ℚ)⌋ →
  packs_to_neil = packs_given_away - packs_to_manny →
  (packs_to_neil : ℚ) / packs_given_away = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_neil_fraction_of_packs_l41_4187


namespace NUMINAMATH_CALUDE_amicable_pairs_theorem_l41_4120

/-- Sum of divisors of a number -/
def sum_of_divisors (n : ℕ) : ℕ := sorry

/-- Two numbers are amicable if the sum of proper divisors of each equals the other number -/
def is_amicable_pair (m n : ℕ) : Prop :=
  sum_of_divisors m = m + n ∧ sum_of_divisors n = m + n

/-- The main theorem stating that the given pairs are amicable -/
theorem amicable_pairs_theorem :
  let pair1_1 := 3^3 * 5 * 7 * 71
  let pair1_2 := 3^3 * 5 * 17 * 31
  let pair2_1 := 3^2 * 5 * 13 * 79 * 29
  let pair2_2 := 3^2 * 5 * 13 * 11 * 199
  is_amicable_pair pair1_1 pair1_2 ∧ is_amicable_pair pair2_1 pair2_2 := by
  sorry

end NUMINAMATH_CALUDE_amicable_pairs_theorem_l41_4120


namespace NUMINAMATH_CALUDE_tysons_three_pointers_l41_4156

/-- Tyson's basketball scoring problem -/
theorem tysons_three_pointers (x : ℕ) : 
  (3 * x + 2 * 12 + 1 * 6 = 75) → x = 15 := by
  sorry

end NUMINAMATH_CALUDE_tysons_three_pointers_l41_4156


namespace NUMINAMATH_CALUDE_weekly_social_media_time_l41_4182

/-- Charlotte's daily phone usage in hours -/
def daily_phone_usage : ℕ := 16

/-- The fraction of phone time spent on social media -/
def social_media_fraction : ℚ := 1/2

/-- Number of days in a week -/
def days_in_week : ℕ := 7

/-- Theorem: Charlotte spends 56 hours on social media in a week -/
theorem weekly_social_media_time : 
  (daily_phone_usage * social_media_fraction * days_in_week : ℚ) = 56 := by
sorry

end NUMINAMATH_CALUDE_weekly_social_media_time_l41_4182


namespace NUMINAMATH_CALUDE_triangle_quadratic_no_roots_l41_4177

/-- Given a, b, and c are side lengths of a triangle, 
    the quadratic equation (a+b)x^2 + 2cx + a+b = 0 has no real roots -/
theorem triangle_quadratic_no_roots (a b c : ℝ) 
  (h_triangle : a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ a + c > b) : 
  ∀ x : ℝ, (a + b) * x^2 + 2 * c * x + (a + b) ≠ 0 := by
  sorry

#check triangle_quadratic_no_roots

end NUMINAMATH_CALUDE_triangle_quadratic_no_roots_l41_4177


namespace NUMINAMATH_CALUDE_recreation_spending_comparison_l41_4164

theorem recreation_spending_comparison (W : ℝ) : 
  let last_week_recreation := 0.15 * W
  let this_week_wages := 0.8 * W
  let this_week_recreation := 0.5 * this_week_wages
  (this_week_recreation / last_week_recreation) * 100 = 267 := by
sorry

end NUMINAMATH_CALUDE_recreation_spending_comparison_l41_4164


namespace NUMINAMATH_CALUDE_min_value_abc_l41_4196

theorem min_value_abc (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (sum_eq_3 : a + b + c = 3) :
  (a + b) / (a * b * c) ≥ 16 / 9 :=
sorry

end NUMINAMATH_CALUDE_min_value_abc_l41_4196


namespace NUMINAMATH_CALUDE_simplify_expression_l41_4178

theorem simplify_expression : ((4 + 6) * 2) / 4 - 1 / 4 = 4.75 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l41_4178


namespace NUMINAMATH_CALUDE_two_sqrt_two_gt_sqrt_seven_l41_4116

theorem two_sqrt_two_gt_sqrt_seven : 2 * Real.sqrt 2 > Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_two_sqrt_two_gt_sqrt_seven_l41_4116


namespace NUMINAMATH_CALUDE_equal_intercept_line_equation_l41_4175

/-- A line passing through (1, 2) with equal intercepts on both axes -/
def EqualInterceptLine : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 = 1 ∧ p.2 = 2) ∨ 
               (p.2 = 2 * p.1) ∨ 
               (p.1 + p.2 = 3)}

theorem equal_intercept_line_equation :
  ∀ (x y : ℝ), (x, y) ∈ EqualInterceptLine ↔ (y = 2 * x ∨ x + y = 3) :=
by sorry

end NUMINAMATH_CALUDE_equal_intercept_line_equation_l41_4175


namespace NUMINAMATH_CALUDE_parabola_unique_values_l41_4138

/-- A parabola passing through three given points -/
structure Parabola where
  b : ℝ
  c : ℝ
  eq : ℝ → ℝ := fun x ↦ x^2 + b*x + c
  point1 : eq (-2) = -8
  point2 : eq 4 = 28
  point3 : eq 1 = 4

/-- The unique values of b and c for the parabola -/
theorem parabola_unique_values (p : Parabola) : p.b = 4 ∧ p.c = -1 := by
  sorry

end NUMINAMATH_CALUDE_parabola_unique_values_l41_4138


namespace NUMINAMATH_CALUDE_min_f_1998_l41_4111

/-- A function from positive integers to positive integers satisfying the given property -/
def SpecialFunction (f : ℕ+ → ℕ+) : Prop :=
  ∀ (s t : ℕ+), f (t^2 * f s) = s * (f t)^2

/-- The theorem stating the minimum value of f(1998) -/
theorem min_f_1998 (f : ℕ+ → ℕ+) (h : SpecialFunction f) :
  ∃ (m : ℕ+), f 1998 = m ∧ ∀ (g : ℕ+ → ℕ+), SpecialFunction g → m ≤ g 1998 :=
sorry

end NUMINAMATH_CALUDE_min_f_1998_l41_4111


namespace NUMINAMATH_CALUDE_divisibility_implies_equality_l41_4167

theorem divisibility_implies_equality (a b n : ℕ) :
  (∀ k : ℕ, k ≠ 0 → ∃ q : ℤ, a - k^n = (b - k) * q) →
  a = b^n := by sorry

end NUMINAMATH_CALUDE_divisibility_implies_equality_l41_4167


namespace NUMINAMATH_CALUDE_shortest_side_of_right_triangle_l41_4139

theorem shortest_side_of_right_triangle (a b c : ℝ) : 
  a = 7 → b = 24 → c^2 = a^2 + b^2 → a ≤ b ∧ a ≤ c := by
  sorry

end NUMINAMATH_CALUDE_shortest_side_of_right_triangle_l41_4139


namespace NUMINAMATH_CALUDE_wedding_fish_count_l41_4197

/-- The number of tables at Glenda's wedding reception. -/
def num_tables : ℕ := 32

/-- The number of fish in each fishbowl, except for one special table. -/
def fish_per_table : ℕ := 2

/-- The number of fish in the special table's fishbowl. -/
def fish_in_special_table : ℕ := 3

/-- The total number of fish at Glenda's wedding reception. -/
def total_fish : ℕ := (num_tables - 1) * fish_per_table + fish_in_special_table

theorem wedding_fish_count : total_fish = 65 := by
  sorry

end NUMINAMATH_CALUDE_wedding_fish_count_l41_4197


namespace NUMINAMATH_CALUDE_probability_of_mixed_team_l41_4100

def num_girls : ℕ := 3
def num_boys : ℕ := 2
def team_size : ℕ := 2
def total_group_size : ℕ := num_girls + num_boys

def num_total_combinations : ℕ := (total_group_size.choose team_size)
def num_mixed_combinations : ℕ := num_girls * num_boys

theorem probability_of_mixed_team :
  (num_mixed_combinations : ℚ) / num_total_combinations = 3 / 5 := by sorry

end NUMINAMATH_CALUDE_probability_of_mixed_team_l41_4100


namespace NUMINAMATH_CALUDE_recipe_fat_calculation_l41_4186

/-- Calculates the grams of fat per serving in a recipe -/
def fat_per_serving (servings : ℕ) (cream_cups : ℚ) (fat_per_cup : ℕ) : ℚ :=
  (cream_cups * fat_per_cup) / servings

theorem recipe_fat_calculation :
  fat_per_serving 4 (1/2) 88 = 11 := by
  sorry

end NUMINAMATH_CALUDE_recipe_fat_calculation_l41_4186


namespace NUMINAMATH_CALUDE_sector_angle_when_arc_equals_radius_l41_4159

theorem sector_angle_when_arc_equals_radius (r : ℝ) (θ : ℝ) :
  r > 0 → r * θ = r → θ = 1 := by sorry

end NUMINAMATH_CALUDE_sector_angle_when_arc_equals_radius_l41_4159


namespace NUMINAMATH_CALUDE_gain_percent_problem_l41_4168

/-- 
If the cost price of 50 articles is equal to the selling price of 25 articles, 
then the gain percent is 100%.
-/
theorem gain_percent_problem (C S : ℝ) (hpos : C > 0) : 
  50 * C = 25 * S → (S - C) / C * 100 = 100 :=
by
  sorry

#check gain_percent_problem

end NUMINAMATH_CALUDE_gain_percent_problem_l41_4168


namespace NUMINAMATH_CALUDE_tenth_difference_optimal_number_l41_4112

/-- A positive integer that can be expressed as the difference of squares of two positive integers m and n, where m - n > 1 -/
def DifferenceOptimalNumber (k : ℕ) : Prop :=
  ∃ m n : ℕ, m > n ∧ m - n > 1 ∧ k = m^2 - n^2

/-- The sequence of difference optimal numbers in ascending order -/
def DifferenceOptimalSequence : ℕ → ℕ :=
  sorry

theorem tenth_difference_optimal_number :
  DifferenceOptimalNumber (DifferenceOptimalSequence 10) ∧ 
  DifferenceOptimalSequence 10 = 32 := by
  sorry

end NUMINAMATH_CALUDE_tenth_difference_optimal_number_l41_4112


namespace NUMINAMATH_CALUDE_negation_of_implication_l41_4170

theorem negation_of_implication :
  (¬(∀ x : ℝ, x = 3 → x^2 - 2*x - 3 = 0)) ↔
  (∀ x : ℝ, x ≠ 3 → x^2 - 2*x - 3 ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_implication_l41_4170


namespace NUMINAMATH_CALUDE_max_second_term_arithmetic_sequence_l41_4149

theorem max_second_term_arithmetic_sequence (a d : ℕ) (h1 : 0 < a) (h2 : 0 < d) :
  (a + (a + d) + (a + 2*d) + (a + 3*d) = 58) →
  ∀ b e : ℕ, (0 < b) → (0 < e) →
  (b + (b + e) + (b + 2*e) + (b + 3*e) = 58) →
  (a + d ≤ 10) :=
by sorry

end NUMINAMATH_CALUDE_max_second_term_arithmetic_sequence_l41_4149


namespace NUMINAMATH_CALUDE_harmonic_sum_number_bounds_harmonic_number_digit_sum_even_l41_4166

/-- Represents a three-digit natural number -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  units : Nat
  is_valid : hundreds ≥ 1 ∧ hundreds ≤ 9 ∧ tens ≤ 9 ∧ units ≤ 9

/-- Checks if a number is a sum number -/
def is_sum_number (n : ThreeDigitNumber) : Prop :=
  n.hundreds = n.tens + n.units

/-- Checks if a number is a harmonic number -/
def is_harmonic_number (n : ThreeDigitNumber) : Prop :=
  n.hundreds = n.tens^2 - n.units^2

/-- Checks if a number is a harmonic sum number -/
def is_harmonic_sum_number (n : ThreeDigitNumber) : Prop :=
  is_sum_number n ∧ is_harmonic_number n

/-- Converts a ThreeDigitNumber to its numeric value -/
def to_nat (n : ThreeDigitNumber) : Nat :=
  100 * n.hundreds + 10 * n.tens + n.units

theorem harmonic_sum_number_bounds (n : ThreeDigitNumber) :
  is_harmonic_sum_number n → 110 ≤ to_nat n ∧ to_nat n ≤ 954 := by
  sorry

theorem harmonic_number_digit_sum_even (n : ThreeDigitNumber) :
  is_harmonic_number n → Even (n.hundreds + n.tens + n.units) := by
  sorry

end NUMINAMATH_CALUDE_harmonic_sum_number_bounds_harmonic_number_digit_sum_even_l41_4166


namespace NUMINAMATH_CALUDE_wednesday_to_monday_ratio_l41_4132

/-- Represents the number of cars passing through a toll booth on each day of the week -/
structure TollBoothWeek where
  total : ℕ
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ
  thursday : ℕ
  friday : ℕ
  saturday : ℕ
  sunday : ℕ

/-- The ratio of cars on Wednesday to cars on Monday is 2:1 -/
theorem wednesday_to_monday_ratio (week : TollBoothWeek)
  (h1 : week.total = 450)
  (h2 : week.monday = 50)
  (h3 : week.tuesday = 50)
  (h4 : week.wednesday = week.thursday)
  (h5 : week.friday = 50)
  (h6 : week.saturday = 50)
  (h7 : week.sunday = 50)
  (h8 : week.total = week.monday + week.tuesday + week.wednesday + week.thursday + 
                     week.friday + week.saturday + week.sunday) :
  week.wednesday = 2 * week.monday := by
  sorry

#check wednesday_to_monday_ratio

end NUMINAMATH_CALUDE_wednesday_to_monday_ratio_l41_4132


namespace NUMINAMATH_CALUDE_equation_solutions_l41_4157

theorem equation_solutions :
  (∃ x1 x2 : ℝ, x1^2 = 2 ∧ x2^2 = 2 ∧ x1 = Real.sqrt 2 ∧ x2 = -Real.sqrt 2) ∧
  (∃ x1 x2 : ℝ, 4*x1^2 - 1 = 0 ∧ 4*x2^2 - 1 = 0 ∧ x1 = 1/2 ∧ x2 = -1/2) ∧
  (∃ x1 x2 : ℝ, (x1-1)^2 - 4 = 0 ∧ (x2-1)^2 - 4 = 0 ∧ x1 = 3 ∧ x2 = -1) ∧
  (∃ x1 x2 : ℝ, 12*(3-x1)^2 - 48 = 0 ∧ 12*(3-x2)^2 - 48 = 0 ∧ x1 = 1 ∧ x2 = 5) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l41_4157


namespace NUMINAMATH_CALUDE_equation_condition_l41_4107

theorem equation_condition (a b c : ℕ) 
  (ha : 0 < a ∧ a < 20) 
  (hb : 0 < b ∧ b < 20) 
  (hc : 0 < c ∧ c < 20) : 
  (20 * a + b) * (20 * a + c) = 400 * a^2 + 200 * a + b * c ↔ b + c = 10 := by
sorry

end NUMINAMATH_CALUDE_equation_condition_l41_4107


namespace NUMINAMATH_CALUDE_final_card_values_card_game_2004_l41_4153

def card_game (n : ℕ) : ℕ :=
  3^(2*n) - 2 * 3^n + 2

theorem final_card_values (n : ℕ) :
  let initial_cards := 3^(2*n)
  let final_values := card_game n
  ∀ c : ℕ, c ≥ 3^n ∧ c ≤ 3^(2*n) - 3^n + 1 →
    c ∈ Finset.range final_values :=
by sorry

theorem card_game_2004 :
  card_game 1002 = 3^2004 - 2 * 3^1002 + 2 :=
by sorry

end NUMINAMATH_CALUDE_final_card_values_card_game_2004_l41_4153


namespace NUMINAMATH_CALUDE_symmetry_line_is_common_chord_l41_4198

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 8
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 4*x - 4*y = 0

-- Define the line of symmetry
def is_line_of_symmetry (l : ℝ → ℝ → Prop) : Prop :=
  ∀ (x y : ℝ), circle1 x y ↔ ∃ (x' y' : ℝ), circle2 x' y' ∧ l x y ∧ l x' y'

-- Define the common chord
def is_common_chord (l : ℝ → ℝ → Prop) : Prop :=
  ∀ (x y : ℝ), l x y → (circle1 x y ∧ circle2 x y)

-- Theorem statement
theorem symmetry_line_is_common_chord :
  ∀ (l : ℝ → ℝ → Prop), is_line_of_symmetry l → is_common_chord l :=
sorry

end NUMINAMATH_CALUDE_symmetry_line_is_common_chord_l41_4198


namespace NUMINAMATH_CALUDE_inequality_condition_l41_4113

-- Define the conditions
def has_solutions (a : ℝ) : Prop := ∃ x : ℝ, x^2 + 2*a*x - a ≤ 0

def condition_q (a : ℝ) : Prop := a > 0 ∨ a < -1

-- State the theorem
theorem inequality_condition :
  (∀ a : ℝ, condition_q a → has_solutions a) ∧
  ¬(∀ a : ℝ, has_solutions a → condition_q a) :=
sorry

end NUMINAMATH_CALUDE_inequality_condition_l41_4113


namespace NUMINAMATH_CALUDE_inequality_solution_range_l41_4102

theorem inequality_solution_range (a : ℝ) : 
  (∃ x : ℝ, |x - 2| + |x + 2| ≤ a^2 - 3*a) ↔ (a ≥ 4 ∨ a ≤ -1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l41_4102


namespace NUMINAMATH_CALUDE_discount_calculation_l41_4114

/-- Proves that a 20% discount followed by a 15% discount on an item 
    originally priced at 450 results in a final price of 306 -/
theorem discount_calculation (original_price : ℝ) (first_discount second_discount final_price : ℝ) :
  original_price = 450 ∧ 
  first_discount = 20 ∧ 
  second_discount = 15 ∧ 
  final_price = 306 →
  original_price * (1 - first_discount / 100) * (1 - second_discount / 100) = final_price :=
by sorry

end NUMINAMATH_CALUDE_discount_calculation_l41_4114


namespace NUMINAMATH_CALUDE_z_modulus_range_l41_4152

-- Define the complex number z
def z (a : ℝ) : ℂ := Complex.mk (a - 2) (a + 1)

-- Define the condition for z to be in the second quadrant
def second_quadrant (a : ℝ) : Prop := a - 2 < 0 ∧ a + 1 > 0

-- State the theorem
theorem z_modulus_range :
  ∃ (min max : ℝ), min = 3 * Real.sqrt 2 / 2 ∧ max = 3 ∧
  ∀ a : ℝ, second_quadrant a →
    Complex.abs (z a) ≥ min ∧ Complex.abs (z a) ≤ max ∧
    (∃ a₁ a₂ : ℝ, second_quadrant a₁ ∧ second_quadrant a₂ ∧
      Complex.abs (z a₁) = min ∧ Complex.abs (z a₂) = max) :=
by sorry

end NUMINAMATH_CALUDE_z_modulus_range_l41_4152
