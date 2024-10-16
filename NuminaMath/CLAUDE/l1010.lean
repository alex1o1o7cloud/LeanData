import Mathlib

namespace NUMINAMATH_CALUDE_min_value_implies_a_l1010_101039

def f (x a : ℝ) : ℝ := |x + 1| + 2 * |x - a|

theorem min_value_implies_a (a : ℝ) : 
  (∀ x, f x a ≥ 5) ∧ (∃ x, f x a = 5) → a = -6 ∨ a = 4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_implies_a_l1010_101039


namespace NUMINAMATH_CALUDE_reflection_sum_l1010_101096

/-- Reflection across a line y = mx + b --/
def reflect (m b x y : ℚ) : ℚ × ℚ :=
  let d := (x + (y - b) * m) / (1 + m^2)
  (2 * d - x, 2 * d * m - y + 2 * b)

theorem reflection_sum (m b : ℚ) : 
  reflect m b 2 3 = (10, 6) → m + b = 107/6 := by
sorry

end NUMINAMATH_CALUDE_reflection_sum_l1010_101096


namespace NUMINAMATH_CALUDE_highway_length_l1010_101084

theorem highway_length (speed1 speed2 time : ℝ) (h1 : speed1 = 13) (h2 : speed2 = 17) (h3 : time = 2) :
  (speed1 + speed2) * time = 60 := by
  sorry

end NUMINAMATH_CALUDE_highway_length_l1010_101084


namespace NUMINAMATH_CALUDE_modular_inverse_31_mod_45_l1010_101010

theorem modular_inverse_31_mod_45 : ∃ x : ℤ, 0 ≤ x ∧ x < 45 ∧ (31 * x) % 45 = 1 := by
  use 15
  sorry

end NUMINAMATH_CALUDE_modular_inverse_31_mod_45_l1010_101010


namespace NUMINAMATH_CALUDE_simplify_expression_l1010_101093

theorem simplify_expression : -(-3) - 4 + (-5) = 3 - 4 - 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1010_101093


namespace NUMINAMATH_CALUDE_minimum_opinion_change_l1010_101048

/-- Represents the number of students who like or dislike math at a given time --/
structure MathOpinion where
  like : Nat
  dislike : Nat

/-- Represents the change in students' opinions about math --/
structure OpinionChange where
  dislike_to_like : Nat
  like_to_dislike : Nat

theorem minimum_opinion_change (initial final : MathOpinion) (change : OpinionChange)
    (h1 : initial.like + initial.dislike = 40)
    (h2 : final.like + final.dislike = 40)
    (h3 : initial.like = 18)
    (h4 : initial.dislike = 22)
    (h5 : final.like = 28)
    (h6 : final.dislike = 12)
    (h7 : change.dislike_to_like = 10)
    (h8 : final.like = initial.like + change.dislike_to_like - change.like_to_dislike) :
    change.dislike_to_like + change.like_to_dislike = 10 := by
  sorry

#check minimum_opinion_change

end NUMINAMATH_CALUDE_minimum_opinion_change_l1010_101048


namespace NUMINAMATH_CALUDE_coin_collection_l1010_101061

theorem coin_collection (nickels dimes quarters : ℕ) (total_value : ℕ) : 
  nickels = dimes →
  quarters = 2 * nickels →
  total_value = 1950 →
  5 * nickels + 10 * dimes + 25 * quarters = total_value →
  nickels = 30 := by
sorry

end NUMINAMATH_CALUDE_coin_collection_l1010_101061


namespace NUMINAMATH_CALUDE_gcd_315_2016_l1010_101026

theorem gcd_315_2016 : Nat.gcd 315 2016 = 63 := by sorry

end NUMINAMATH_CALUDE_gcd_315_2016_l1010_101026


namespace NUMINAMATH_CALUDE_binomial_coefficient_ratio_l1010_101086

theorem binomial_coefficient_ratio (n : ℕ) : 4^n / 2^n = 64 → n = 6 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_ratio_l1010_101086


namespace NUMINAMATH_CALUDE_usual_weekly_salary_proof_l1010_101058

/-- Calculates the weekly salary given daily rate and work days per week -/
def weeklySalary (dailyRate : ℚ) (workDaysPerWeek : ℕ) : ℚ :=
  dailyRate * workDaysPerWeek

/-- Represents a worker with a daily rate and standard work week -/
structure Worker where
  dailyRate : ℚ
  workDaysPerWeek : ℕ

theorem usual_weekly_salary_proof (w : Worker) 
    (h1 : w.workDaysPerWeek = 5)
    (h2 : w.dailyRate * 2 = 745) :
    weeklySalary w.dailyRate w.workDaysPerWeek = 1862.5 := by
  sorry

#eval weeklySalary (745 / 2) 5

end NUMINAMATH_CALUDE_usual_weekly_salary_proof_l1010_101058


namespace NUMINAMATH_CALUDE_partition_12_exists_partition_22_not_exists_l1010_101020

def is_valid_partition (n : ℕ) (k : ℕ) : Prop :=
  ∃ (partition : List (ℕ × ℕ)),
    partition.length = k ∧
    (∀ (pair : ℕ × ℕ), pair ∈ partition → pair.1 ∈ Finset.range n ∧ pair.2 ∈ Finset.range n) ∧
    (∀ i j, i ≠ j → (partition.get i).1 ≠ (partition.get j).1 ∧ (partition.get i).1 ≠ (partition.get j).2 ∧
                    (partition.get i).2 ≠ (partition.get j).1 ∧ (partition.get i).2 ≠ (partition.get j).2) ∧
    (∀ (pair : ℕ × ℕ), pair ∈ partition → Nat.Prime (pair.1 + pair.2)) ∧
    (∀ i j, i ≠ j → (partition.get i).1 + (partition.get i).2 ≠ (partition.get j).1 + (partition.get j).2)

theorem partition_12_exists : is_valid_partition 12 6 :=
sorry

theorem partition_22_not_exists : ¬ is_valid_partition 22 11 :=
sorry

end NUMINAMATH_CALUDE_partition_12_exists_partition_22_not_exists_l1010_101020


namespace NUMINAMATH_CALUDE_S_intersect_T_eq_T_l1010_101043

def S : Set Int := {s | ∃ n : Int, s = 2*n + 1}
def T : Set Int := {t | ∃ n : Int, t = 4*n + 1}

theorem S_intersect_T_eq_T : S ∩ T = T := by
  sorry

end NUMINAMATH_CALUDE_S_intersect_T_eq_T_l1010_101043


namespace NUMINAMATH_CALUDE_candies_distribution_l1010_101079

def candies_a : ℕ := 17
def candies_b : ℕ := 19
def num_people : ℕ := 9

theorem candies_distribution :
  (candies_a + candies_b) / num_people = 4 := by
  sorry

end NUMINAMATH_CALUDE_candies_distribution_l1010_101079


namespace NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l1010_101014

theorem fixed_point_of_exponential_function (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(x - 2) + 1
  f 2 = 2 := by sorry

end NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l1010_101014


namespace NUMINAMATH_CALUDE_probability_of_n_in_polynomial_l1010_101075

def word : String := "polynomial"

def count_letter (s : String) (c : Char) : Nat :=
  s.toList.filter (· = c) |>.length

theorem probability_of_n_in_polynomial :
  (count_letter word 'n' : ℚ) / word.length = 1 / 10 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_n_in_polynomial_l1010_101075


namespace NUMINAMATH_CALUDE_benny_apples_l1010_101040

theorem benny_apples (total : ℕ) (dan_apples : ℕ) (benny_apples : ℕ) :
  total = 11 → dan_apples = 9 → total = dan_apples + benny_apples → benny_apples = 2 := by
  sorry

end NUMINAMATH_CALUDE_benny_apples_l1010_101040


namespace NUMINAMATH_CALUDE_equal_distribution_proof_l1010_101036

theorem equal_distribution_proof (isabella sam giselle : ℕ) : 
  isabella = sam + 45 →
  isabella = giselle + 15 →
  giselle = 120 →
  (isabella + sam + giselle) / 3 = 115 :=
by
  sorry

end NUMINAMATH_CALUDE_equal_distribution_proof_l1010_101036


namespace NUMINAMATH_CALUDE_percentage_of_three_digit_numbers_with_repeated_digit_l1010_101034

theorem percentage_of_three_digit_numbers_with_repeated_digit : 
  let total_three_digit_numbers : ℕ := 900
  let three_digit_numbers_without_repeat : ℕ := 9 * 9 * 8
  let three_digit_numbers_with_repeat : ℕ := total_three_digit_numbers - three_digit_numbers_without_repeat
  let percentage : ℚ := three_digit_numbers_with_repeat / total_three_digit_numbers
  ⌊percentage * 1000 + 5⌋ / 10 = 28 :=
by sorry

end NUMINAMATH_CALUDE_percentage_of_three_digit_numbers_with_repeated_digit_l1010_101034


namespace NUMINAMATH_CALUDE_avocado_cost_l1010_101046

theorem avocado_cost (initial_amount : ℕ) (num_avocados : ℕ) (change : ℕ) : 
  initial_amount = 20 → num_avocados = 3 → change = 14 → 
  (initial_amount - change) / num_avocados = 2 := by
  sorry

end NUMINAMATH_CALUDE_avocado_cost_l1010_101046


namespace NUMINAMATH_CALUDE_equation_solution_l1010_101008

theorem equation_solution (x y : ℝ) :
  (2 * x) / (1 + x^2) = (1 + y^2) / (2 * y) →
  ((x = 1 ∧ y = 1) ∨ (x = -1 ∧ y = -1)) := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l1010_101008


namespace NUMINAMATH_CALUDE_tenth_term_of_arithmetic_sequence_l1010_101035

/-- An arithmetic sequence with given second and third terms -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  a 2 = 2 ∧ a 3 = 4 ∧ ∀ n : ℕ, a (n + 1) - a n = a 2 - a 1

theorem tenth_term_of_arithmetic_sequence 
  (a : ℕ → ℝ) (h : arithmetic_sequence a) : a 10 = 18 := by
  sorry

end NUMINAMATH_CALUDE_tenth_term_of_arithmetic_sequence_l1010_101035


namespace NUMINAMATH_CALUDE_president_and_committee_from_eight_l1010_101088

/-- The number of ways to choose a president and a 2-person committee from a group of people. -/
def choose_president_and_committee (n : ℕ) : ℕ :=
  n * (n - 1).choose 2

/-- The theorem stating that choosing a president and a 2-person committee from 8 people results in 168 ways. -/
theorem president_and_committee_from_eight :
  choose_president_and_committee 8 = 168 := by
  sorry

#eval choose_president_and_committee 8

end NUMINAMATH_CALUDE_president_and_committee_from_eight_l1010_101088


namespace NUMINAMATH_CALUDE_regular_18gon_symmetry_sum_l1010_101024

/-- A regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  n_pos : n > 0

/-- The number of lines of symmetry in a regular polygon -/
def linesOfSymmetry (p : RegularPolygon n) : ℕ := n

/-- The smallest positive angle of rotational symmetry in degrees for a regular polygon -/
def rotationalSymmetryAngle (p : RegularPolygon n) : ℚ := 360 / n

theorem regular_18gon_symmetry_sum :
  ∀ (p : RegularPolygon 18),
  (linesOfSymmetry p : ℚ) + rotationalSymmetryAngle p = 38 := by sorry

end NUMINAMATH_CALUDE_regular_18gon_symmetry_sum_l1010_101024


namespace NUMINAMATH_CALUDE_principle_countable_noun_meaning_l1010_101072

/-- Define a type for English words -/
def EnglishWord : Type := String

/-- Define a type for word meanings -/
def WordMeaning : Type := String

/-- Function to get the meaning of a word when used as a countable noun -/
def countableNounMeaning (word : EnglishWord) : WordMeaning :=
  sorry

/-- Theorem stating that "principle" as a countable noun means "principle, criterion" -/
theorem principle_countable_noun_meaning :
  countableNounMeaning "principle" = "principle, criterion" :=
sorry

end NUMINAMATH_CALUDE_principle_countable_noun_meaning_l1010_101072


namespace NUMINAMATH_CALUDE_quotient_digits_l1010_101064

def dividend (n : ℕ) : ℕ := 100 * n + 38

theorem quotient_digits :
  (∀ n : ℕ, n ≤ 7 → (dividend n) / 8 < 100) ∧
  (dividend 7) / 8 ≥ 10 ∧
  (∀ n : ℕ, n ≥ 8 → (dividend n) / 8 ≥ 100) ∧
  (dividend 8) / 8 < 1000 :=
sorry

end NUMINAMATH_CALUDE_quotient_digits_l1010_101064


namespace NUMINAMATH_CALUDE_rectangle_area_l1010_101070

theorem rectangle_area (square_side : ℝ) (circle_radius : ℝ) (rectangle_length : ℝ) (rectangle_breadth : ℝ) :
  square_side ^ 2 = 1296 →
  circle_radius = square_side →
  rectangle_length = circle_radius / 6 →
  rectangle_breadth = 10 →
  rectangle_length * rectangle_breadth = 60 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_l1010_101070


namespace NUMINAMATH_CALUDE_lcm_132_315_l1010_101021

theorem lcm_132_315 : Nat.lcm 132 315 = 13860 := by sorry

end NUMINAMATH_CALUDE_lcm_132_315_l1010_101021


namespace NUMINAMATH_CALUDE_arithmetic_sequence_m_value_l1010_101067

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- Sum function
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_formula : ∀ n, S n = n * (a 1 + a n) / 2

/-- The main theorem -/
theorem arithmetic_sequence_m_value
  (seq : ArithmeticSequence)
  (h1 : seq.S (m - 1) = -2)
  (h2 : seq.S m = 0)
  (h3 : seq.S (m + 1) = 3)
  : m = 5 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_m_value_l1010_101067


namespace NUMINAMATH_CALUDE_inscribed_sphere_surface_area_l1010_101098

/-- Given a pyramid with volume V and surface area S, prove that when V = 2 and S = 3,
    the surface area of the inscribed sphere is 16π. -/
theorem inscribed_sphere_surface_area (V S : ℝ) (h1 : V = 2) (h2 : S = 3) :
  let r := 3 * V / S
  4 * Real.pi * r^2 = 16 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_inscribed_sphere_surface_area_l1010_101098


namespace NUMINAMATH_CALUDE_function_not_in_first_quadrant_l1010_101074

-- Define the linear function
def f (x : ℝ) : ℝ := -3 * x - 2

-- Theorem: The function f does not pass through the first quadrant
theorem function_not_in_first_quadrant :
  ∀ x y : ℝ, f x = y → ¬(x > 0 ∧ y > 0) := by
  sorry

end NUMINAMATH_CALUDE_function_not_in_first_quadrant_l1010_101074


namespace NUMINAMATH_CALUDE_petyas_calculation_error_l1010_101003

theorem petyas_calculation_error :
  ¬∃ (a : ℕ), a > 3 ∧ 
  ∃ (n : ℕ), ((a - 3) * (a + 4) - a = n) ∧ 
  (∃ (digits : List ℕ), 
    digits.length = 6069 ∧
    digits.count 8 = 2023 ∧
    digits.count 0 = 2023 ∧
    digits.count 3 = 2023 ∧
    (∀ d, d ∈ digits → d ∈ [8, 0, 3]) ∧
    n = digits.foldl (λ acc d => acc * 10 + d) 0) :=
sorry

end NUMINAMATH_CALUDE_petyas_calculation_error_l1010_101003


namespace NUMINAMATH_CALUDE_rectangle_side_ratio_l1010_101000

theorem rectangle_side_ratio (a b : ℝ) (h : b = 2 * a) : (b / a) ^ 2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_side_ratio_l1010_101000


namespace NUMINAMATH_CALUDE_ellipse_equation_l1010_101005

/-- Given an ellipse and a line passing through its upper vertex and right focus,
    prove that the equation of the ellipse is x^2/5 + y^2/4 = 1. -/
theorem ellipse_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ c : ℝ, c > 0 ∧ c < a ∧ a^2 = b^2 + c^2 ∧
   2*0 + b - 2 = 0 ∧ 2*c + 0 - 2 = 0) →
  ∀ x y : ℝ, x^2/a^2 + y^2/b^2 = 1 ↔ x^2/5 + y^2/4 = 1 :=
by sorry


end NUMINAMATH_CALUDE_ellipse_equation_l1010_101005


namespace NUMINAMATH_CALUDE_intersection_implies_a_zero_l1010_101051

theorem intersection_implies_a_zero (a : ℝ) : 
  let A : Set ℝ := {a^2, a+1, -1}
  let B : Set ℝ := {2*a-1, |a-2|, 3*a^2+4}
  (A ∩ B : Set ℝ) = {-1} → a = 0 := by
sorry

end NUMINAMATH_CALUDE_intersection_implies_a_zero_l1010_101051


namespace NUMINAMATH_CALUDE_quadratic_trinomial_minimum_l1010_101052

theorem quadratic_trinomial_minimum (a b : ℝ) (h1 : a > b) 
  (h2 : ∀ x : ℝ, a * x^2 + 2 * x + b ≥ 0)
  (h3 : ∃ x₀ : ℝ, a * x₀^2 + 2 * x₀ + b = 0) :
  ∀ x : ℝ, (a^2 + b^2) / (a - b) ≥ 2 * Real.sqrt 2 ∧
  ∃ y : ℝ, (a^2 + b^2) / (a - b) = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_trinomial_minimum_l1010_101052


namespace NUMINAMATH_CALUDE_batsman_average_problem_l1010_101012

/-- Represents a batsman's cricket statistics -/
structure BatsmanStats where
  innings : ℕ
  totalRuns : ℕ
  average : ℚ

/-- Calculates the new average after an innings -/
def newAverage (stats : BatsmanStats) (runsScored : ℕ) : ℚ :=
  (stats.totalRuns + runsScored) / (stats.innings + 1)

/-- Theorem statement for the batsman's average problem -/
theorem batsman_average_problem (stats : BatsmanStats) 
  (h1 : stats.innings = 11)
  (h2 : newAverage stats 55 = stats.average + 1) :
  newAverage stats 55 = 44 := by
  sorry

#check batsman_average_problem

end NUMINAMATH_CALUDE_batsman_average_problem_l1010_101012


namespace NUMINAMATH_CALUDE_tangent_line_at_pi_l1010_101057

/-- The equation of the tangent line to y = x sin x at (π, 0) is y = -πx + π² -/
theorem tangent_line_at_pi (x y : ℝ) : 
  let f : ℝ → ℝ := λ t => t * Real.sin t
  let f' : ℝ → ℝ := λ t => Real.sin t + t * Real.cos t
  let tangent_line : ℝ → ℝ := λ t => -π * t + π^2
  (∀ t, HasDerivAt f (f' t) t) →
  HasDerivAt f (f' π) π →
  tangent_line π = f π →
  tangent_line = λ t => -π * t + π^2 := by
sorry


end NUMINAMATH_CALUDE_tangent_line_at_pi_l1010_101057


namespace NUMINAMATH_CALUDE_median_salary_is_25000_l1010_101047

/-- Represents a position in the company with its count and salary --/
structure Position where
  title : String
  count : Nat
  salary : Nat

/-- The list of positions in the company --/
def positions : List Position := [
  ⟨"President", 1, 130000⟩,
  ⟨"Vice-President", 15, 90000⟩,
  ⟨"Director", 10, 80000⟩,
  ⟨"Associate Director", 8, 50000⟩,
  ⟨"Administrative Specialist", 37, 25000⟩
]

/-- The total number of employees --/
def totalEmployees : Nat := positions.foldl (fun acc p => acc + p.count) 0

/-- The median salary of the employees --/
def medianSalary : Nat := 25000

theorem median_salary_is_25000 :
  totalEmployees = 71 → medianSalary = 25000 := by
  sorry

end NUMINAMATH_CALUDE_median_salary_is_25000_l1010_101047


namespace NUMINAMATH_CALUDE_linear_function_and_inequality_l1010_101028

-- Define the linear function f
def f : ℝ → ℝ := fun x ↦ x + 2

-- Define the function g
def g (a : ℝ) : ℝ → ℝ := fun x ↦ (1 - a) * x^2 - x

theorem linear_function_and_inequality (a : ℝ) :
  (∀ x, f (f x) = x + 4) →
  (∀ x₁ ∈ Set.Icc (1/4 : ℝ) 4, ∃ x₂ ∈ Set.Icc (-3 : ℝ) (1/3 : ℝ), g a x₁ ≥ f x₂) →
  (∀ x, f x = x + 2) ∧ a ∈ Set.Iic (3/4 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_linear_function_and_inequality_l1010_101028


namespace NUMINAMATH_CALUDE_train_speed_l1010_101081

/-- Proves that a train crossing a bridge has a specific speed -/
theorem train_speed (train_length bridge_length : Real) (crossing_time : Real) :
  train_length = 110 ∧
  bridge_length = 112 ∧
  crossing_time = 11.099112071034318 →
  (train_length + bridge_length) / crossing_time * 3.6 = 72 := by
sorry

end NUMINAMATH_CALUDE_train_speed_l1010_101081


namespace NUMINAMATH_CALUDE_profit_calculation_l1010_101049

-- Define the buy rate
def buy_rate : ℚ := 15 / 4

-- Define the sell rate
def sell_rate : ℚ := 30 / 6

-- Define the target profit
def target_profit : ℚ := 200

-- Define the number of oranges to be sold
def oranges_to_sell : ℕ := 160

-- Theorem statement
theorem profit_calculation :
  (oranges_to_sell : ℚ) * (sell_rate - buy_rate) = target_profit :=
by sorry

end NUMINAMATH_CALUDE_profit_calculation_l1010_101049


namespace NUMINAMATH_CALUDE_cubic_function_symmetry_l1010_101004

theorem cubic_function_symmetry (a b : ℝ) :
  let f : ℝ → ℝ := λ x ↦ a * x^3 - b * x + 1
  f (-2) = 1 → f 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_cubic_function_symmetry_l1010_101004


namespace NUMINAMATH_CALUDE_unique_real_solution_l1010_101041

theorem unique_real_solution :
  ∃! x : ℝ, x + Real.sqrt (x - 2) = 4 := by sorry

end NUMINAMATH_CALUDE_unique_real_solution_l1010_101041


namespace NUMINAMATH_CALUDE_set_intersection_example_l1010_101019

theorem set_intersection_example :
  let A : Set ℕ := {0, 1, 2}
  let B : Set ℕ := {1, 2, 3, 4}
  A ∩ B = {1, 2} := by
sorry

end NUMINAMATH_CALUDE_set_intersection_example_l1010_101019


namespace NUMINAMATH_CALUDE_richard_david_age_difference_l1010_101025

/-- The ages of Richard, David, and Scott in a family -/
structure FamilyAges where
  R : ℕ  -- Richard's age
  D : ℕ  -- David's age
  S : ℕ  -- Scott's age

/-- The conditions of the family ages problem -/
def FamilyAgesProblem (ages : FamilyAges) : Prop :=
  ages.R > ages.D ∧                 -- Richard is older than David
  ages.D = ages.S + 8 ∧             -- David is 8 years older than Scott
  ages.R + 8 = 2 * (ages.S + 8) ∧   -- In 8 years, Richard will be twice as old as Scott
  ages.D = 14                       -- David was 9 years old 5 years ago

/-- The theorem stating that Richard is 6 years older than David -/
theorem richard_david_age_difference (ages : FamilyAges) 
  (h : FamilyAgesProblem ages) : ages.R = ages.D + 6 := by
  sorry


end NUMINAMATH_CALUDE_richard_david_age_difference_l1010_101025


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1010_101054

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℤ)
  (h_arithmetic : arithmetic_sequence a)
  (h_sum : a 1 + a 7 = -8)
  (h_second : a 2 = 2) :
  ∃ d : ℤ, d = -3 ∧ ∀ n : ℕ, a (n + 1) = a n + d :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1010_101054


namespace NUMINAMATH_CALUDE_yellow_red_difference_after_border_l1010_101015

/-- Represents a hexagonal figure with red and yellow tiles -/
structure HexagonalFigure where
  redTiles : ℕ
  yellowTiles : ℕ

/-- Adds a border of yellow tiles to a hexagonal figure -/
def addBorder (figure : HexagonalFigure) : HexagonalFigure :=
  { redTiles := figure.redTiles,
    yellowTiles := figure.yellowTiles + 24 }

theorem yellow_red_difference_after_border (figure : HexagonalFigure) 
  (h1 : figure.redTiles = 12)
  (h2 : figure.yellowTiles = 8) :
  let newFigure := addBorder figure
  newFigure.yellowTiles - newFigure.redTiles = 20 := by
  sorry

end NUMINAMATH_CALUDE_yellow_red_difference_after_border_l1010_101015


namespace NUMINAMATH_CALUDE_conference_theorem_l1010_101044

/-- Represents the state of knowledge among scientists at a conference --/
structure ConferenceState where
  total_scientists : Nat
  initial_knowers : Nat
  pairs : Nat

/-- Calculates the probability of a specific number of scientists knowing the news after pairing --/
def probability_of_knowers (state : ConferenceState) (final_knowers : Nat) : ℚ :=
  sorry

/-- Calculates the expected number of scientists knowing the news after pairing --/
def expected_knowers (state : ConferenceState) : ℚ :=
  sorry

/-- The main theorem about the conference scenario --/
theorem conference_theorem (state : ConferenceState) 
  (h1 : state.total_scientists = 18)
  (h2 : state.initial_knowers = 10)
  (h3 : state.pairs = 9) :
  (probability_of_knowers state 13 = 0) ∧
  (probability_of_knowers state 14 = 1120 / 2431) ∧
  (expected_knowers state = 14 + 12 / 17) :=
  sorry

end NUMINAMATH_CALUDE_conference_theorem_l1010_101044


namespace NUMINAMATH_CALUDE_range_of_a_l1010_101011

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, 0 < x ∧ x < 4 → |x - 1| < a) ↔ a ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1010_101011


namespace NUMINAMATH_CALUDE_curry_house_spicy_curries_l1010_101009

/-- Represents the curry house's pepper buying strategy -/
structure CurryHouse where
  very_spicy_peppers : ℕ := 3
  spicy_peppers : ℕ := 2
  mild_peppers : ℕ := 1
  prev_very_spicy : ℕ := 30
  prev_spicy : ℕ := 30
  prev_mild : ℕ := 10
  new_mild : ℕ := 90
  pepper_reduction : ℕ := 40

/-- Calculates the number of spicy curries the curry house now buys peppers for -/
def calculate_new_spicy_curries (ch : CurryHouse) : ℕ :=
  let prev_total := ch.very_spicy_peppers * ch.prev_very_spicy + 
                    ch.spicy_peppers * ch.prev_spicy + 
                    ch.mild_peppers * ch.prev_mild
  let new_total := prev_total - ch.pepper_reduction
  (new_total - ch.mild_peppers * ch.new_mild) / ch.spicy_peppers

/-- Proves that the curry house now buys peppers for 15 spicy curries -/
theorem curry_house_spicy_curries (ch : CurryHouse) : 
  calculate_new_spicy_curries ch = 15 := by
  sorry

end NUMINAMATH_CALUDE_curry_house_spicy_curries_l1010_101009


namespace NUMINAMATH_CALUDE_a_range_l1010_101016

theorem a_range (a : ℝ) : 
  (∀ x : ℝ, |x - a| - |x| < 2 - a^2) → 
  a > -1 ∧ a < 1 :=
sorry

end NUMINAMATH_CALUDE_a_range_l1010_101016


namespace NUMINAMATH_CALUDE_triangle_area_l1010_101083

/-- A triangle with integral sides and perimeter 12 has an area of 6 -/
theorem triangle_area (a b c : ℕ) : 
  a + b + c = 12 → 
  a + b > c → b + c > a → c + a > b → 
  (a * b : ℚ) / 2 = 6 :=
sorry

end NUMINAMATH_CALUDE_triangle_area_l1010_101083


namespace NUMINAMATH_CALUDE_inequality_solution_l1010_101059

theorem inequality_solution (x : ℝ) : 
  (1 / (x * (x + 1)) - 1 / ((x + 1) * (x + 2)) < 1 / 4) ↔ 
  (x < -2 ∨ (-1 < x ∧ x < 0) ∨ 1 < x) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l1010_101059


namespace NUMINAMATH_CALUDE_ten_steps_climb_l1010_101013

def climb_stairs (n : ℕ) : ℕ :=
  if n ≤ 1 then 1
  else if n = 2 then 2
  else climb_stairs (n - 1) + climb_stairs (n - 2)

theorem ten_steps_climb : climb_stairs 10 = 89 := by
  sorry

end NUMINAMATH_CALUDE_ten_steps_climb_l1010_101013


namespace NUMINAMATH_CALUDE_inverse_proportion_problem_l1010_101071

/-- Given that x and y are inversely proportional, and x + y = 30 and x - y = 10, 
    prove that y = 200/7 when x = 7. -/
theorem inverse_proportion_problem (x y : ℝ) (k : ℝ) 
  (h1 : x * y = k)  -- x and y are inversely proportional
  (h2 : x + y = 30) -- sum condition
  (h3 : x - y = 10) -- difference condition
  : (7 : ℝ) * (200 / 7) = k := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_problem_l1010_101071


namespace NUMINAMATH_CALUDE_four_jumps_reduction_l1010_101022

def jump_reduction (initial : ℕ) (jumps : ℕ) (reduction : ℕ) : ℕ :=
  initial - jumps * reduction

theorem four_jumps_reduction : jump_reduction 320 4 10 = 280 := by
  sorry

end NUMINAMATH_CALUDE_four_jumps_reduction_l1010_101022


namespace NUMINAMATH_CALUDE_vector_b_solution_l1010_101073

def vector_a : ℝ × ℝ := (1, -2)

theorem vector_b_solution (b : ℝ × ℝ) :
  (b.1 * vector_a.2 = b.2 * vector_a.1) →  -- parallel condition
  (b.1^2 + b.2^2 = 20) →                   -- magnitude condition
  (b = (2, -4) ∨ b = (-2, 4)) :=
by sorry

end NUMINAMATH_CALUDE_vector_b_solution_l1010_101073


namespace NUMINAMATH_CALUDE_marble_count_exceeds_200_l1010_101018

def marbles (n : ℕ) : ℕ := 3 * 2^n

theorem marble_count_exceeds_200 :
  (∃ k : ℕ, marbles k > 200) ∧ 
  (∀ j : ℕ, j < 8 → marbles j ≤ 200) ∧
  (marbles 8 > 200) := by
sorry

end NUMINAMATH_CALUDE_marble_count_exceeds_200_l1010_101018


namespace NUMINAMATH_CALUDE_field_length_width_difference_l1010_101042

/-- Proves that for a rectangular field with length 24 meters and width 13.5 meters,
    the difference between twice the width and the length is 3 meters. -/
theorem field_length_width_difference :
  let length : ℝ := 24
  let width : ℝ := 13.5
  2 * width - length = 3 := by sorry

end NUMINAMATH_CALUDE_field_length_width_difference_l1010_101042


namespace NUMINAMATH_CALUDE_min_max_values_on_interval_monotone_increasing_condition_l1010_101053

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a * abs (x - 1)

-- Part I
theorem min_max_values_on_interval (a : ℝ) (h : a = 2) :
  (∃ x ∈ Set.Icc 0 2, ∀ y ∈ Set.Icc 0 2, f a x ≤ f a y) ∧
  (∃ x ∈ Set.Icc 0 2, ∀ y ∈ Set.Icc 0 2, f a y ≤ f a x) ∧
  (∃ x ∈ Set.Icc 0 2, f a x = 1) ∧
  (∃ x ∈ Set.Icc 0 2, f a x = 6) := by
  sorry

-- Part II
theorem monotone_increasing_condition :
  (∀ a : ℝ, a ∈ Set.Icc (-2) 0 ↔ Monotone (f a)) := by
  sorry

end NUMINAMATH_CALUDE_min_max_values_on_interval_monotone_increasing_condition_l1010_101053


namespace NUMINAMATH_CALUDE_a_square_property_l1010_101082

def a : ℕ → ℤ
  | 0 => 1
  | 1 => 1
  | (n + 2) => 14 * a (n + 1) - a n

theorem a_square_property : ∃ k : ℕ → ℤ, ∀ n : ℕ, 2 * a n - 1 = k n ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_a_square_property_l1010_101082


namespace NUMINAMATH_CALUDE_hawks_score_l1010_101078

theorem hawks_score (eagles hawks ravens : ℕ) : 
  eagles + hawks + ravens = 120 →
  eagles = hawks + 20 →
  ravens = 2 * hawks →
  hawks = 25 :=
by
  sorry

end NUMINAMATH_CALUDE_hawks_score_l1010_101078


namespace NUMINAMATH_CALUDE_common_root_and_other_roots_l1010_101002

def f (x : ℝ) : ℝ := x^4 - x^3 - 22*x^2 + 16*x + 96
def g (x : ℝ) : ℝ := x^3 - 2*x^2 - 3*x + 10

theorem common_root_and_other_roots :
  (f (-2) = 0 ∧ g (-2) = 0) ∧
  (f 3 = 0 ∧ f (-4) = 0 ∧ f 4 = 0) :=
sorry

end NUMINAMATH_CALUDE_common_root_and_other_roots_l1010_101002


namespace NUMINAMATH_CALUDE_product_from_hcf_lcm_l1010_101055

theorem product_from_hcf_lcm (a b : ℕ+) : 
  Nat.gcd a b = 16 → Nat.lcm a b = 160 → a * b = 2560 := by
  sorry

end NUMINAMATH_CALUDE_product_from_hcf_lcm_l1010_101055


namespace NUMINAMATH_CALUDE_odd_sum_probability_redesigned_board_l1010_101091

/-- Represents the redesigned dartboard -/
structure Dartboard where
  outer_radius : ℝ
  inner_radius : ℝ
  inner_points : Fin 3 → ℕ
  outer_points : Fin 3 → ℕ

/-- The probability of getting an odd sum when throwing two darts -/
def odd_sum_probability (d : Dartboard) : ℝ :=
  sorry

/-- The redesigned dartboard as described in the problem -/
def redesigned_board : Dartboard :=
  { outer_radius := 8
    inner_radius := 4
    inner_points := λ i => if i = 0 then 3 else 1
    outer_points := λ i => if i = 0 then 2 else 3 }

/-- Theorem stating the probability of an odd sum on the redesigned board -/
theorem odd_sum_probability_redesigned_board :
    odd_sum_probability redesigned_board = 4 / 9 :=
  sorry

end NUMINAMATH_CALUDE_odd_sum_probability_redesigned_board_l1010_101091


namespace NUMINAMATH_CALUDE_logarithm_properties_l1010_101032

noncomputable def log (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem logarithm_properties (b : ℝ) (h1 : b > 0) (h2 : b ≠ 1) :
  (log b 1 = 0) ∧
  (log b b = 1) ∧
  (log b (1/b) = -1) ∧
  (∀ x : ℝ, 0 < x → x < 1 → log b x < 0) :=
by sorry

end NUMINAMATH_CALUDE_logarithm_properties_l1010_101032


namespace NUMINAMATH_CALUDE_cat_dog_ratio_l1010_101045

/-- Given a ratio of cats to dogs and the number of cats, calculate the number of dogs -/
theorem cat_dog_ratio (cat_ratio : ℕ) (dog_ratio : ℕ) (num_cats : ℕ) (num_dogs : ℕ) :
  cat_ratio ≠ 0 ∧ dog_ratio ≠ 0 →
  cat_ratio * num_dogs = dog_ratio * num_cats →
  cat_ratio = 4 ∧ dog_ratio = 5 ∧ num_cats = 24 →
  num_dogs = 30 := by
  sorry

#check cat_dog_ratio

end NUMINAMATH_CALUDE_cat_dog_ratio_l1010_101045


namespace NUMINAMATH_CALUDE_arithmetic_mean_special_set_l1010_101099

/-- Given a set of n numbers where n > 1, one number is 1 - 2/n and all others are 1,
    the arithmetic mean of these numbers is 1 - 2/n² -/
theorem arithmetic_mean_special_set (n : ℕ) (h : n > 1) :
  let s : Finset ℕ := Finset.range n
  let f : ℕ → ℝ := fun i => if i = 0 then 1 - 2 / n else 1
  (s.sum f) / n = 1 - 2 / n^2 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_mean_special_set_l1010_101099


namespace NUMINAMATH_CALUDE_max_value_theorem_l1010_101092

theorem max_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x^2 - x*y + y^2 = 8) :
  (∃ (z : ℝ), z = x^2 + x*y + y^2 ∧ z ≤ 24) ∧
  (∃ (a b c d : ℕ+), 24 = (a + b * Real.sqrt c) / d ∧ a + b + c + d = 26) :=
by sorry

end NUMINAMATH_CALUDE_max_value_theorem_l1010_101092


namespace NUMINAMATH_CALUDE_stating_line_triangle_intersection_count_l1010_101001

/-- Represents the number of intersection points between a line and a triangle's boundary. -/
inductive IntersectionCount
  | Zero
  | One
  | Two
  | Infinite

/-- A triangle in a 2D plane. -/
structure Triangle where
  -- Add necessary fields (e.g., vertices) here

/-- A line in a 2D plane. -/
structure Line where
  -- Add necessary fields (e.g., points or coefficients) here

/-- 
  Theorem stating that the number of intersection points between a line and 
  a triangle's boundary is either 0, 1, 2, or infinitely many.
-/
theorem line_triangle_intersection_count 
  (t : Triangle) (l : Line) : 
  ∃ (count : IntersectionCount), 
    (count = IntersectionCount.Zero) ∨ 
    (count = IntersectionCount.One) ∨ 
    (count = IntersectionCount.Two) ∨ 
    (count = IntersectionCount.Infinite) :=
by
  sorry


end NUMINAMATH_CALUDE_stating_line_triangle_intersection_count_l1010_101001


namespace NUMINAMATH_CALUDE_shooting_probabilities_l1010_101069

/-- Two shooters independently shoot at a target -/
structure ShootingScenario where
  /-- Probability of shooter A hitting the target -/
  prob_A : ℝ
  /-- Probability of shooter B hitting the target -/
  prob_B : ℝ
  /-- Assumption that probabilities are between 0 and 1 -/
  h_prob_A : 0 ≤ prob_A ∧ prob_A ≤ 1
  h_prob_B : 0 ≤ prob_B ∧ prob_B ≤ 1

/-- The probability that the target is hit in one shooting attempt -/
def prob_hit (s : ShootingScenario) : ℝ :=
  s.prob_A + s.prob_B - s.prob_A * s.prob_B

/-- The probability that the target is hit exactly by shooter A -/
def prob_hit_A (s : ShootingScenario) : ℝ :=
  s.prob_A * (1 - s.prob_B)

theorem shooting_probabilities (s : ShootingScenario) 
  (h_A : s.prob_A = 0.95) (h_B : s.prob_B = 0.9) : 
  prob_hit s = 0.995 ∧ prob_hit_A s = 0.095 := by
  sorry

#eval prob_hit ⟨0.95, 0.9, by norm_num, by norm_num⟩
#eval prob_hit_A ⟨0.95, 0.9, by norm_num, by norm_num⟩

end NUMINAMATH_CALUDE_shooting_probabilities_l1010_101069


namespace NUMINAMATH_CALUDE_quadratic_function_statements_l1010_101076

/-- A quadratic function y = ax^2 + bx + c where a ≠ 0 and x ∈ M (M is a non-empty set) -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  M : Set ℝ
  a_nonzero : a ≠ 0
  M_nonempty : M.Nonempty

/-- Statement 1: When a > 0, the function always has a minimum value of (4ac - b^2) / (4a) -/
def statement1 (f : QuadraticFunction) : Prop :=
  f.a > 0 → ∃ (min : ℝ), ∀ (x : ℝ), x ∈ f.M → f.a * x^2 + f.b * x + f.c ≥ min

/-- Statement 2: The existence of max/min depends on the range of x, and both can exist with values not necessarily (4ac - b^2) / (4a) -/
def statement2 (f : QuadraticFunction) : Prop :=
  ∃ (M1 M2 : Set ℝ), M1.Nonempty ∧ M2.Nonempty ∧
    (∃ (max min : ℝ), (∀ (x : ℝ), x ∈ M1 → f.a * x^2 + f.b * x + f.c ≤ max) ∧
                      (∀ (x : ℝ), x ∈ M2 → f.a * x^2 + f.b * x + f.c ≥ min) ∧
                      (max ≠ (4 * f.a * f.c - f.b^2) / (4 * f.a) ∨
                       min ≠ (4 * f.a * f.c - f.b^2) / (4 * f.a)))

/-- Statement 3: The method to find max/min involves finding the axis of symmetry and analyzing the graph -/
def statement3 (f : QuadraticFunction) : Prop :=
  ∃ (x : ℝ), x = -f.b / (2 * f.a) ∧
    ∀ (y : ℝ), y ∈ f.M → (f.a * y^2 + f.b * y + f.c = f.a * x^2 + f.b * x + f.c ↔ y = x)

theorem quadratic_function_statements (f : QuadraticFunction) :
  (statement1 f ∧ ¬statement2 f ∧ ¬statement3 f) ∨
  (¬statement1 f ∧ statement2 f ∧ ¬statement3 f) ∨
  (¬statement1 f ∧ ¬statement2 f ∧ statement3 f) ∨
  (¬statement1 f ∧ statement2 f ∧ statement3 f) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_statements_l1010_101076


namespace NUMINAMATH_CALUDE_fourteen_percent_of_seven_hundred_is_ninety_eight_l1010_101080

theorem fourteen_percent_of_seven_hundred_is_ninety_eight :
  ∀ x : ℝ, (14 / 100) * x = 98 → x = 700 := by
  sorry

end NUMINAMATH_CALUDE_fourteen_percent_of_seven_hundred_is_ninety_eight_l1010_101080


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_equality_l1010_101038

/-- Arithmetic sequence {a_n} -/
def a (n : ℕ) : ℝ := 2*n + 2

/-- Geometric sequence {b_n} -/
def b (n : ℕ) : ℝ := 8 * 2^(n-2)

theorem arithmetic_geometric_sequence_equality :
  (a 1 + a 2 = 10) →
  (a 4 - a 3 = 2) →
  (b 2 = a 3) →
  (b 3 = a 7) →
  (a 15 = b 4) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_equality_l1010_101038


namespace NUMINAMATH_CALUDE_sin_alpha_minus_2pi_over_3_l1010_101063

theorem sin_alpha_minus_2pi_over_3 (α : ℝ) (h : Real.cos (π / 6 - α) = 2 / 3) :
  Real.sin (α - 2 * π / 3) = -2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sin_alpha_minus_2pi_over_3_l1010_101063


namespace NUMINAMATH_CALUDE_probability_gary_paula_letters_l1010_101029

/-- The probability of drawing one letter from Gary's name and one from Paula's name -/
theorem probability_gary_paula_letters : 
  let total_letters : ℕ := 9
  let gary_letters : ℕ := 4
  let paula_letters : ℕ := 5
  let prob_gary_then_paula : ℚ := (gary_letters : ℚ) / total_letters * paula_letters / (total_letters - 1)
  let prob_paula_then_gary : ℚ := (paula_letters : ℚ) / total_letters * gary_letters / (total_letters - 1)
  prob_gary_then_paula + prob_paula_then_gary = 5 / 9 := by
  sorry

end NUMINAMATH_CALUDE_probability_gary_paula_letters_l1010_101029


namespace NUMINAMATH_CALUDE_simplify_expression_l1010_101062

theorem simplify_expression (x y : ℝ) :
  3 * x + 7 * x^2 + 4 * y - (5 - 3 * x - 7 * x^2 + 2 * y) = 14 * x^2 + 6 * x + 2 * y - 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1010_101062


namespace NUMINAMATH_CALUDE_water_mass_in_range_l1010_101068

/-- Represents the thermodynamic properties of a substance -/
structure ThermodynamicProperties where
  specific_heat_capacity : Real
  specific_latent_heat : Real

/-- Represents the initial state of a substance -/
structure InitialState where
  mass : Real
  temperature : Real

/-- Calculates the range of added water mass given the initial conditions and final temperature -/
def calculate_water_mass_range (ice_props : ThermodynamicProperties)
                               (water_props : ThermodynamicProperties)
                               (ice_initial : InitialState)
                               (water_initial : InitialState)
                               (final_temp : Real) : Set Real :=
  sorry

/-- Theorem stating that the mass of added water lies within the calculated range -/
theorem water_mass_in_range :
  let ice_props : ThermodynamicProperties := {
    specific_heat_capacity := 2100,
    specific_latent_heat := 3.3e5
  }
  let water_props : ThermodynamicProperties := {
    specific_heat_capacity := 4200,
    specific_latent_heat := 0
  }
  let ice_initial : InitialState := {
    mass := 0.1,
    temperature := -5
  }
  let water_initial : InitialState := {
    mass := 0,  -- mass to be determined
    temperature := 10
  }
  let final_temp : Real := 0
  let water_mass_range := calculate_water_mass_range ice_props water_props ice_initial water_initial final_temp
  ∀ m ∈ water_mass_range, 0.0028 ≤ m ∧ m ≤ 0.8119 :=
by sorry

end NUMINAMATH_CALUDE_water_mass_in_range_l1010_101068


namespace NUMINAMATH_CALUDE_regular_octahedron_has_six_vertices_l1010_101006

/-- A regular octahedron is a Platonic solid with equilateral triangular faces. -/
structure RegularOctahedron where
  -- We don't need to define the internal structure for this problem

/-- The number of vertices in a regular octahedron -/
def num_vertices (o : RegularOctahedron) : ℕ := 6

/-- Theorem: A regular octahedron has 6 vertices -/
theorem regular_octahedron_has_six_vertices (o : RegularOctahedron) : 
  num_vertices o = 6 := by
  sorry

end NUMINAMATH_CALUDE_regular_octahedron_has_six_vertices_l1010_101006


namespace NUMINAMATH_CALUDE_tahir_contribution_l1010_101097

/-- Proves that Tahir needs to contribute 50 Canadian dollars given the problem conditions -/
theorem tahir_contribution (headphone_cost : ℝ) (kenji_yen : ℝ) (exchange_rate : ℝ) 
  (h1 : headphone_cost = 200)
  (h2 : kenji_yen = 15000)
  (h3 : exchange_rate = 100) : 
  headphone_cost - (kenji_yen / exchange_rate) = 50 := by
  sorry

#check tahir_contribution

end NUMINAMATH_CALUDE_tahir_contribution_l1010_101097


namespace NUMINAMATH_CALUDE_three_collinear_sufficient_not_necessary_for_coplanar_l1010_101095

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Check if three points are collinear -/
def collinear (p q r : Point3D) : Prop := sorry

/-- Check if four points are coplanar -/
def coplanar (p q r s : Point3D) : Prop := sorry

/-- Main theorem: Three points on a line is sufficient but not necessary for four points to be coplanar -/
theorem three_collinear_sufficient_not_necessary_for_coplanar :
  (∀ p q r s : Point3D, (collinear p q r) → (coplanar p q r s)) ∧
  (∃ p q r s : Point3D, (coplanar p q r s) ∧ ¬(collinear p q r) ∧ ¬(collinear p q s) ∧ ¬(collinear p r s) ∧ ¬(collinear q r s)) :=
sorry

end NUMINAMATH_CALUDE_three_collinear_sufficient_not_necessary_for_coplanar_l1010_101095


namespace NUMINAMATH_CALUDE_f_neg_two_eq_four_l1010_101027

noncomputable def g (x : ℝ) : ℝ := Real.log x / Real.log (1/2)

def symmetric_about_y_eq_x (f g : ℝ → ℝ) : Prop :=
  ∀ x y, f x = y ↔ g y = x

theorem f_neg_two_eq_four 
  (f : ℝ → ℝ) 
  (h : symmetric_about_y_eq_x f g) : 
  f (-2) = 4 := by
sorry

end NUMINAMATH_CALUDE_f_neg_two_eq_four_l1010_101027


namespace NUMINAMATH_CALUDE_three_zeros_iff_a_in_open_interval_l1010_101060

/-- A cubic function with parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 3*x + a

/-- The property of having exactly three distinct real zeros -/
def has_three_distinct_zeros (a : ℝ) : Prop :=
  ∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ f a x = 0 ∧ f a y = 0 ∧ f a z = 0

/-- Theorem stating the equivalence between the function having three distinct zeros
    and the parameter a being in the open interval (-2, 2) -/
theorem three_zeros_iff_a_in_open_interval :
  ∀ a : ℝ, has_three_distinct_zeros a ↔ -2 < a ∧ a < 2 :=
sorry

end NUMINAMATH_CALUDE_three_zeros_iff_a_in_open_interval_l1010_101060


namespace NUMINAMATH_CALUDE_count_distinct_sums_l1010_101017

def S : Finset ℕ := {2, 5, 8, 11, 14, 17, 20, 23}

def sumOfFourDistinct (s : Finset ℕ) : Finset ℕ :=
  (s.powerset.filter (fun t => t.card = 4)).image (fun t => t.sum id)

theorem count_distinct_sums : (sumOfFourDistinct S).card = 49 := by
  sorry

end NUMINAMATH_CALUDE_count_distinct_sums_l1010_101017


namespace NUMINAMATH_CALUDE_rug_overlap_problem_l1010_101056

theorem rug_overlap_problem (total_area : ℝ) (covered_area : ℝ) (two_layer_area : ℝ)
  (h1 : total_area = 350)
  (h2 : covered_area = 250)
  (h3 : two_layer_area = 45) :
  total_area = covered_area + two_layer_area + 55 :=
by sorry

end NUMINAMATH_CALUDE_rug_overlap_problem_l1010_101056


namespace NUMINAMATH_CALUDE_ice_cream_melt_height_l1010_101089

/-- The height of a cylinder with radius 9 inches, having the same volume as a sphere with radius 3 inches, is 4/9 inches. -/
theorem ice_cream_melt_height : 
  let sphere_radius : ℝ := 3
  let cylinder_radius : ℝ := 9
  let sphere_volume := (4 / 3) * Real.pi * sphere_radius ^ 3
  let cylinder_volume (h : ℝ) := Real.pi * cylinder_radius ^ 2 * h
  ∃ h : ℝ, cylinder_volume h = sphere_volume ∧ h = 4 / 9 :=
by sorry

end NUMINAMATH_CALUDE_ice_cream_melt_height_l1010_101089


namespace NUMINAMATH_CALUDE_martian_calendar_months_l1010_101087

/-- Represents the number of days in a Martian month -/
inductive MartianMonth
  | long : MartianMonth  -- 100 days
  | short : MartianMonth -- 77 days

/-- Calculates the number of days in a Martian month -/
def daysInMonth (m : MartianMonth) : Nat :=
  match m with
  | MartianMonth.long => 100
  | MartianMonth.short => 77

/-- Represents a Martian calendar year -/
structure MartianYear where
  months : List MartianMonth
  total_days : Nat
  total_days_eq : total_days = List.sum (months.map daysInMonth)

/-- The theorem to be proved -/
theorem martian_calendar_months (year : MartianYear) 
    (h : year.total_days = 5882) : year.months.length = 74 := by
  sorry

#check martian_calendar_months

end NUMINAMATH_CALUDE_martian_calendar_months_l1010_101087


namespace NUMINAMATH_CALUDE_john_bought_two_shirts_l1010_101065

/-- The number of shirts John bought -/
def num_shirts : ℕ := 2

/-- The cost of the first shirt in dollars -/
def cost_first_shirt : ℕ := 15

/-- The cost of the second shirt in dollars -/
def cost_second_shirt : ℕ := cost_first_shirt - 6

/-- The total cost of the shirts in dollars -/
def total_cost : ℕ := 24

theorem john_bought_two_shirts :
  num_shirts = 2 ∧
  cost_first_shirt = cost_second_shirt + 6 ∧
  cost_first_shirt = 15 ∧
  cost_first_shirt + cost_second_shirt = total_cost :=
by sorry

end NUMINAMATH_CALUDE_john_bought_two_shirts_l1010_101065


namespace NUMINAMATH_CALUDE_delivery_ratio_l1010_101030

theorem delivery_ratio : 
  let meals : ℕ := 3
  let total : ℕ := 27
  let packages : ℕ := total - meals
  packages / meals = 8 := by
sorry

end NUMINAMATH_CALUDE_delivery_ratio_l1010_101030


namespace NUMINAMATH_CALUDE_pentagon_reconstruction_l1010_101085

-- Define the pentagon and its extended points
variable (A B C D E A' B' C' D' E' : ℝ × ℝ)

-- Define the conditions of the pentagon
axiom extend_A : A' - B = B - A
axiom extend_B : B' - C = C - B
axiom extend_C : C' - D = D - C
axiom extend_D : D' - E = E - D
axiom extend_E : E' - A = A - E

-- Define the theorem
theorem pentagon_reconstruction :
  E = (1/31 : ℝ) • A' + (1/31 : ℝ) • B' + (2/31 : ℝ) • C' + (4/31 : ℝ) • D' + (8/31 : ℝ) • E' :=
sorry

end NUMINAMATH_CALUDE_pentagon_reconstruction_l1010_101085


namespace NUMINAMATH_CALUDE_max_isosceles_triangles_correct_l1010_101090

/-- Represents a set of points on a line and a point not on the line -/
structure PointConfiguration where
  n : ℕ  -- number of points on the line
  h : n = 100

/-- The maximum number of isosceles triangles that can be formed -/
def max_isosceles_triangles (config : PointConfiguration) : ℕ := 150

/-- Theorem stating the maximum number of isosceles triangles -/
theorem max_isosceles_triangles_correct (config : PointConfiguration) :
  max_isosceles_triangles config = 150 := by
  sorry

end NUMINAMATH_CALUDE_max_isosceles_triangles_correct_l1010_101090


namespace NUMINAMATH_CALUDE_line_and_circle_problem_l1010_101033

/-- Line l: x - y + m = 0 -/
def line_l (m : ℝ) (x y : ℝ) : Prop := x - y + m = 0

/-- Point type -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point is on a line -/
def point_on_line (p : Point) (m : ℝ) : Prop := line_l m p.x p.y

/-- Rotate a line by 90 degrees counterclockwise around its x-axis intersection -/
def rotate_line (m : ℝ) (x y : ℝ) : Prop := y + x + m = 0

/-- Circle equation -/
def circle_equation (center : Point) (radius : ℝ) (x y : ℝ) : Prop :=
  (x - center.x)^2 + (y - center.y)^2 = radius^2

theorem line_and_circle_problem (m : ℝ) :
  (∃ (x y : ℝ), rotate_line m x y ∧ x = 2 ∧ y = -3) →
  (∃ (center : Point) (radius : ℝ),
    point_on_line center m ∧
    circle_equation center radius 1 1 ∧
    circle_equation center radius 2 (-2)) →
  m = 1 ∧
  (∃ (center : Point),
    point_on_line center 1 ∧
    circle_equation center 5 1 1 ∧
    circle_equation center 5 2 (-2) ∧
    center.x = -3 ∧
    center.y = -2) := by
  sorry

end NUMINAMATH_CALUDE_line_and_circle_problem_l1010_101033


namespace NUMINAMATH_CALUDE_no_solution_inequality_system_l1010_101094

theorem no_solution_inequality_system :
  ¬ ∃ (x y : ℝ), (4 * x^2 + 4 * x * y + 19 * y^2 ≤ 2) ∧ (x - y ≤ -1) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_inequality_system_l1010_101094


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l1010_101077

theorem quadratic_equation_roots : ∃! (r₁ r₂ : ℝ),
  (r₁ ≠ r₂) ∧ 
  (r₁^2 - 6*r₁ + 8 = 0) ∧ 
  (r₂^2 - 6*r₂ + 8 = 0) ∧
  (r₁ = 2 ∨ r₁ = 4) ∧
  (r₂ = 2 ∨ r₂ = 4) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l1010_101077


namespace NUMINAMATH_CALUDE_weight_of_doubled_cube_l1010_101023

/-- Given two cubes of the same material, if one cube has sides twice as long as the other,
    and the smaller cube weighs 4 pounds, then the larger cube weighs 32 pounds. -/
theorem weight_of_doubled_cube (s : ℝ) (weight : ℝ → ℝ) (volume : ℝ → ℝ) :
  (∀ x, weight x = (weight s / volume s) * volume x) →  -- weight is proportional to volume
  volume s = s^3 →  -- volume of a cube is side length cubed
  weight s = 4 →  -- weight of original cube is 4 pounds
  weight (2*s) = 32 :=  -- weight of new cube with doubled side length
by
  sorry


end NUMINAMATH_CALUDE_weight_of_doubled_cube_l1010_101023


namespace NUMINAMATH_CALUDE_stirring_ensures_representativeness_l1010_101031

/-- Represents the lottery method for sampling -/
structure LotteryMethod where
  /-- The action of stirring the lots -/
  stir : Bool

/-- Represents the representativeness of a sample -/
def representative (method : LotteryMethod) : Prop :=
  method.stir

/-- Theorem stating that stirring evenly is key to representativeness in the lottery method -/
theorem stirring_ensures_representativeness (method : LotteryMethod) :
  representative method ↔ method.stir :=
sorry

end NUMINAMATH_CALUDE_stirring_ensures_representativeness_l1010_101031


namespace NUMINAMATH_CALUDE_waiter_customers_l1010_101037

theorem waiter_customers (non_tipping : ℕ) (tip_amount : ℕ) (total_tips : ℕ) : 
  non_tipping = 5 → tip_amount = 3 → total_tips = 15 → 
  non_tipping + (total_tips / tip_amount) = 10 :=
by sorry

end NUMINAMATH_CALUDE_waiter_customers_l1010_101037


namespace NUMINAMATH_CALUDE_cone_surface_area_3_4_5_triangle_l1010_101066

/-- The total surface area of a cone formed by rotating a 3-4-5 right triangle around its shortest side. -/
theorem cone_surface_area_3_4_5_triangle : ∃ (S : ℝ), 
  S = π * 4 * (4 + 5) ∧ S = 36 * π := by
  sorry

end NUMINAMATH_CALUDE_cone_surface_area_3_4_5_triangle_l1010_101066


namespace NUMINAMATH_CALUDE_div_chain_equals_fraction_l1010_101007

theorem div_chain_equals_fraction : (132 / 6) / 3 = 22 / 3 := by
  sorry

end NUMINAMATH_CALUDE_div_chain_equals_fraction_l1010_101007


namespace NUMINAMATH_CALUDE_new_person_age_l1010_101050

theorem new_person_age (group_size : ℕ) (age_decrease : ℕ) (replaced_age : ℕ) : 
  group_size = 10 → 
  age_decrease = 3 → 
  replaced_age = 45 → 
  ∃ (original_avg : ℚ) (new_avg : ℚ),
    original_avg - new_avg = age_decrease ∧
    group_size * original_avg - replaced_age = group_size * new_avg - 15 :=
by sorry

end NUMINAMATH_CALUDE_new_person_age_l1010_101050
