import Mathlib

namespace tic_tac_toe_tie_probability_l2150_215078

theorem tic_tac_toe_tie_probability (max_win_prob zoe_win_prob : ℚ) :
  max_win_prob = 4/9 →
  zoe_win_prob = 5/12 →
  1 - (max_win_prob + zoe_win_prob) = 5/36 := by
sorry

end tic_tac_toe_tie_probability_l2150_215078


namespace a_range_theorem_l2150_215037

-- Define the line equation
def line_equation (x y a : ℝ) : ℝ := x + y - a

-- Define the condition for points being on opposite sides of the line
def opposite_sides (a : ℝ) : Prop :=
  (line_equation 1 1 a) * (line_equation 2 (-1) a) < 0

-- Theorem statement
theorem a_range_theorem :
  ∀ a : ℝ, opposite_sides a ↔ a ∈ Set.Ioo 1 2 :=
by sorry

end a_range_theorem_l2150_215037


namespace roots_of_equation_l2150_215013

/-- The polynomial equation whose roots we want to find -/
def f (x : ℝ) : ℝ := (x^3 - 4*x^2 - x + 4)*(x-3)*(x+2)

/-- The set of roots we claim to be correct -/
def root_set : Set ℝ := {-2, -1, 1, 3, 4}

/-- Theorem stating that the roots of the equation are exactly the elements of root_set -/
theorem roots_of_equation : 
  ∀ x : ℝ, f x = 0 ↔ x ∈ root_set :=
by sorry

end roots_of_equation_l2150_215013


namespace lcm_problem_l2150_215066

theorem lcm_problem (m : ℕ+) 
  (h1 : Nat.lcm 40 m = 120) 
  (h2 : Nat.lcm m 45 = 180) : 
  m = 12 := by
  sorry

end lcm_problem_l2150_215066


namespace calorie_calculation_l2150_215088

/-- Represents the daily calorie allowance for a certain age group -/
def average_daily_allowance : ℕ := 2000

/-- The number of calories to reduce daily to hypothetically live to 100 years -/
def calorie_reduction : ℕ := 500

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The allowed weekly calorie intake for the age group -/
def allowed_weekly_intake : ℕ := 10500

theorem calorie_calculation :
  (average_daily_allowance - calorie_reduction) * days_in_week = allowed_weekly_intake := by
  sorry

end calorie_calculation_l2150_215088


namespace equation_solution_l2150_215068

theorem equation_solution : ∃ a : ℝ, -6 * a^2 = 3 * (4 * a + 2) ∧ a = -1 := by
  sorry

end equation_solution_l2150_215068


namespace consecutive_sum_prime_iff_n_one_or_two_l2150_215090

/-- The sum of n consecutive natural numbers starting from k -/
def consecutiveSum (n k : ℕ) : ℕ := n * (2 * k + n - 1) / 2

/-- A natural number is prime if it's greater than 1 and its only divisors are 1 and itself -/
def isPrime (p : ℕ) : Prop :=
  p > 1 ∧ ∀ d : ℕ, d ∣ p → d = 1 ∨ d = p

theorem consecutive_sum_prime_iff_n_one_or_two :
  ∀ n : ℕ, (∃ k : ℕ, isPrime (consecutiveSum n k)) ↔ n = 1 ∨ n = 2 :=
sorry

end consecutive_sum_prime_iff_n_one_or_two_l2150_215090


namespace simplify_expression_l2150_215053

theorem simplify_expression (a b : ℝ) : 
  (-2 * a^2 * b^3) * (-a * b^2)^2 + (-1/2 * a^2 * b^3)^2 * 4 * b = -a^4 * b^7 := by
  sorry

end simplify_expression_l2150_215053


namespace square_equation_solution_l2150_215036

theorem square_equation_solution : ∃ (M : ℕ), M > 0 ∧ 33^2 * 66^2 = 15^2 * M^2 ∧ M = 726 := by
  sorry

end square_equation_solution_l2150_215036


namespace equation_solutions_count_l2150_215007

theorem equation_solutions_count : 
  ∃! (s : Finset ℝ), (∀ x ∈ s, (x^2 - 7)^2 = 49) ∧ s.card = 3 :=
sorry

end equation_solutions_count_l2150_215007


namespace min_first_row_sum_l2150_215035

/-- Represents a grid with 9 rows and 2004 columns -/
def Grid := Fin 9 → Fin 2004 → ℕ

/-- The condition that each integer from 1 to 2004 appears exactly 9 times in the grid -/
def validDistribution (g : Grid) : Prop :=
  ∀ n : Fin 2004, (Finset.univ.sum fun i => (Finset.univ.filter (fun j => g i j = n.val + 1)).card) = 9

/-- The condition that no integer appears more than 3 times in any column -/
def validColumn (g : Grid) : Prop :=
  ∀ j : Fin 2004, ∀ n : Fin 2004, (Finset.univ.filter (fun i => g i j = n.val + 1)).card ≤ 3

/-- The sum of the numbers in the first row -/
def firstRowSum (g : Grid) : ℕ :=
  Finset.univ.sum (fun j => g 0 j)

theorem min_first_row_sum :
  ∀ g : Grid, validDistribution g → validColumn g →
  firstRowSum g ≥ 2005004 :=
sorry

end min_first_row_sum_l2150_215035


namespace arithmetic_sequence_properties_l2150_215043

/-- An arithmetic sequence with given properties -/
structure ArithmeticSequence where
  a : ℕ → ℤ  -- The sequence
  s : ℕ → ℤ  -- The sum of the first n terms
  first_term : a 1 = -7
  third_sum : s 3 = -15

/-- Properties of the arithmetic sequence -/
theorem arithmetic_sequence_properties (seq : ArithmeticSequence) :
  (∀ n : ℕ, seq.a n = 2 * n - 9) ∧
  (∀ n : ℕ, seq.s n = (n - 4)^2 - 16) ∧
  (∀ n : ℕ, seq.s n ≥ -16) ∧
  seq.s 4 = -16 := by
  sorry

end arithmetic_sequence_properties_l2150_215043


namespace rita_butterfly_hours_l2150_215029

theorem rita_butterfly_hours : ∀ (total_required hours_backstroke hours_breaststroke monthly_freestyle_sidestroke months : ℕ),
  total_required = 1500 →
  hours_backstroke = 50 →
  hours_breaststroke = 9 →
  monthly_freestyle_sidestroke = 220 →
  months = 6 →
  total_required - (hours_backstroke + hours_breaststroke + monthly_freestyle_sidestroke * months) = 121 :=
by
  sorry

#check rita_butterfly_hours

end rita_butterfly_hours_l2150_215029


namespace trajectory_of_midpoint_l2150_215091

/-- The trajectory of point P given a moving point M on a circle and a fixed point B -/
theorem trajectory_of_midpoint (x y : ℝ) : 
  (∃ m n : ℝ, m^2 + n^2 = 1 ∧ 
              x = (m + 3) / 2 ∧ 
              y = n / 2) → 
  (2*x - 3)^2 + 4*y^2 = 1 :=
by sorry

end trajectory_of_midpoint_l2150_215091


namespace thirtiethDigitOf_1_11_plus_1_13_l2150_215096

/-- The decimal representation of a rational number -/
def decimalRepresentation (q : ℚ) : ℕ → ℕ := sorry

/-- The nth digit after the decimal point in the sum of the decimal representations of two rational numbers -/
def nthDigitAfterDecimal (q₁ q₂ : ℚ) (n : ℕ) : ℕ := sorry

/-- Theorem: The 30th digit after the decimal point in the sum of 1/11 and 1/13 is 2 -/
theorem thirtiethDigitOf_1_11_plus_1_13 : 
  nthDigitAfterDecimal (1/11) (1/13) 30 = 2 := by sorry

end thirtiethDigitOf_1_11_plus_1_13_l2150_215096


namespace similar_triangles_problem_l2150_215009

theorem similar_triangles_problem (A₁ A₂ : ℕ) (k : ℕ) (s : ℝ) :
  A₁ > A₂ →
  A₁ - A₂ = 18 →
  A₁ = k^2 * A₂ →
  s = 3 →
  (∃ (a b c : ℝ), A₂ = (a * b) / 2 ∧ c^2 = a^2 + b^2 ∧ s = c) →
  (∃ (a' b' c' : ℝ), A₁ = (a' * b') / 2 ∧ c'^2 = a'^2 + b'^2 ∧ 6 = c') :=
by sorry

end similar_triangles_problem_l2150_215009


namespace extreme_value_implies_ab_eq_neg_three_l2150_215012

/-- A function f(x) = ax³ + bx has an extreme value at x = 1/a -/
def has_extreme_value (a b : ℝ) : Prop :=
  let f := fun x : ℝ => a * x^3 + b * x
  ∃ (h : ℝ), h = (1 : ℝ) / a ∧ (deriv f) h = 0

/-- If f(x) = ax³ + bx has an extreme value at x = 1/a, then ab = -3 -/
theorem extreme_value_implies_ab_eq_neg_three (a b : ℝ) (h : a ≠ 0) :
  has_extreme_value a b → a * b = -3 := by
  sorry

end extreme_value_implies_ab_eq_neg_three_l2150_215012


namespace rope_ratio_proof_l2150_215084

theorem rope_ratio_proof (total_length shorter_length longer_length : ℝ) 
  (h1 : total_length = 60)
  (h2 : shorter_length = 20)
  (h3 : longer_length = total_length - shorter_length) :
  longer_length / shorter_length = 2 := by
  sorry

end rope_ratio_proof_l2150_215084


namespace sequence_fifth_b_l2150_215072

/-- Given a sequence {aₙ}, where 2aₙ and aₙ₊₁ are the roots of x² - 3x + bₙ = 0,
    and a₁ = 2, prove that b₅ = -1054 -/
theorem sequence_fifth_b (a : ℕ → ℝ) (b : ℕ → ℝ) : 
  (∀ n, (2 * a n) * (a (n + 1)) = b n) → 
  (∀ n, 2 * a n + a (n + 1) = 3) → 
  a 1 = 2 → 
  b 5 = -1054 :=
by sorry

end sequence_fifth_b_l2150_215072


namespace triangle_inequality_sum_l2150_215057

/-- Given a triangle with side lengths a, b, and c, the sum of the ratios of each side length
    to the square root of twice the sum of squares of the other two sides minus the square
    of the current side is greater than or equal to the square root of 3. -/
theorem triangle_inequality_sum (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) :
  (a / Real.sqrt (2 * b^2 + 2 * c^2 - a^2)) +
  (b / Real.sqrt (2 * c^2 + 2 * a^2 - b^2)) +
  (c / Real.sqrt (2 * a^2 + 2 * b^2 - c^2)) ≥ Real.sqrt 3 := by
  sorry

end triangle_inequality_sum_l2150_215057


namespace sum_of_operation_l2150_215093

def A : Finset ℕ := {1, 2, 3, 5}
def B : Finset ℕ := {1, 2}

def operation (A B : Finset ℕ) : Finset ℕ :=
  Finset.image (λ (x : ℕ × ℕ) => x.1 * x.2) (A.product B)

theorem sum_of_operation :
  (operation A B).sum id = 31 := by sorry

end sum_of_operation_l2150_215093


namespace black_square_area_proof_l2150_215073

/-- The edge length of the cube in feet -/
def cube_edge : ℝ := 12

/-- The total area covered by yellow paint in square feet -/
def yellow_area : ℝ := 432

/-- The number of faces on a cube -/
def num_faces : ℕ := 6

/-- The area of the black square on each face of the cube in square feet -/
def black_square_area : ℝ := 72

theorem black_square_area_proof :
  let total_surface_area := num_faces * cube_edge ^ 2
  let yellow_area_per_face := yellow_area / num_faces
  black_square_area = cube_edge ^ 2 - yellow_area_per_face := by
  sorry

end black_square_area_proof_l2150_215073


namespace potato_peeling_theorem_l2150_215038

def potato_peeling_problem (julie_rate ted_rate combined_time : ℝ) : Prop :=
  let julie_part := combined_time * julie_rate
  let ted_part := combined_time * ted_rate
  let remaining_part := 1 - (julie_part + ted_part)
  remaining_part / julie_rate = 1

theorem potato_peeling_theorem :
  potato_peeling_problem (1/10) (1/8) 4 := by
  sorry

end potato_peeling_theorem_l2150_215038


namespace population_after_four_years_l2150_215032

def population_after_n_years (initial_population : ℕ) (new_people : ℕ) (people_leaving : ℕ) (years : ℕ) : ℕ :=
  let population_after_changes := initial_population + new_people - people_leaving
  (population_after_changes / 2^years : ℕ)

theorem population_after_four_years :
  population_after_n_years 780 100 400 4 = 30 := by
  sorry

end population_after_four_years_l2150_215032


namespace property_satisfied_l2150_215064

theorem property_satisfied (n : ℕ) : 
  (∀ q : ℕ, n % q^2 < q^(q^2) / 2) ↔ (n = 1 ∨ n = 4) := by
  sorry

end property_satisfied_l2150_215064


namespace a_in_P_and_b_in_Q_l2150_215058

-- Define the sets P and Q
def P : Set ℤ := {x | ∃ m : ℤ, x = 2 * m + 1}
def Q : Set ℤ := {y | ∃ n : ℤ, y = 2 * n}

-- Define the theorem
theorem a_in_P_and_b_in_Q (x₀ y₀ : ℤ) (hx : x₀ ∈ P) (hy : y₀ ∈ Q) :
  let a := x₀ + y₀
  let b := x₀ * y₀
  a ∈ P ∧ b ∈ Q := by
  sorry

end a_in_P_and_b_in_Q_l2150_215058


namespace concentric_circles_properties_l2150_215045

/-- Two concentric circles with a width of 15 feet between them -/
structure ConcentricCircles where
  inner_diameter : ℝ
  width : ℝ
  width_is_15 : width = 15

theorem concentric_circles_properties (c : ConcentricCircles) :
  let outer_diameter := c.inner_diameter + 2 * c.width
  (π * outer_diameter - π * c.inner_diameter = 30 * π) ∧
  (π * (15 * c.inner_diameter + 225) = 
   π * ((outer_diameter / 2)^2 - (c.inner_diameter / 2)^2)) := by
  sorry

end concentric_circles_properties_l2150_215045


namespace fraction_inequality_solution_set_l2150_215095

theorem fraction_inequality_solution_set (x : ℝ) (h : x ≠ 0) :
  (x - 3) / x ≥ 0 ↔ x < 0 ∨ x ≥ 3 := by
  sorry

end fraction_inequality_solution_set_l2150_215095


namespace orange_slices_problem_l2150_215025

/-- The number of additional slices needed to fill the last container -/
def additional_slices_needed (total_slices : ℕ) (container_capacity : ℕ) : ℕ :=
  container_capacity - (total_slices % container_capacity)

/-- Theorem stating that given 329 slices and a container capacity of 4,
    3 additional slices are needed to fill the last container -/
theorem orange_slices_problem :
  additional_slices_needed 329 4 = 3 := by
  sorry

end orange_slices_problem_l2150_215025


namespace new_oarsman_weight_l2150_215075

theorem new_oarsman_weight (n : ℕ) (old_weight average_increase : ℝ) :
  n = 10 ∧ old_weight = 53 ∧ average_increase = 1.8 →
  ∃ new_weight : ℝ,
    new_weight = old_weight + n * average_increase ∧
    new_weight = 71 := by
  sorry

end new_oarsman_weight_l2150_215075


namespace sufficient_not_necessary_condition_l2150_215011

theorem sufficient_not_necessary_condition (x : ℝ) :
  (∀ x, x > 2 → (x + 1) * (x - 2) > 0) ∧
  (∃ x, (x + 1) * (x - 2) > 0 ∧ ¬(x > 2)) := by
  sorry

end sufficient_not_necessary_condition_l2150_215011


namespace roots_of_equation_l2150_215054

theorem roots_of_equation (x y : ℝ) (h1 : x + y = 10) (h2 : (x - y) * (x + y) = 48) :
  x^2 - 10*x + 19.24 = 0 ∧ y^2 - 10*y + 19.24 = 0 := by
  sorry

end roots_of_equation_l2150_215054


namespace no_solution_implies_a_leq_two_l2150_215014

theorem no_solution_implies_a_leq_two (a : ℝ) :
  (∀ x : ℝ, |x + 2| + |3 - x| ≥ 2*a + 1) → a ≤ 2 := by
  sorry

end no_solution_implies_a_leq_two_l2150_215014


namespace moon_speed_km_per_hour_l2150_215065

/-- The speed of the moon in kilometers per second -/
def moon_speed_km_per_sec : ℝ := 1.05

/-- The number of seconds in an hour -/
def seconds_per_hour : ℕ := 3600

/-- Theorem: The moon's speed in kilometers per hour -/
theorem moon_speed_km_per_hour :
  moon_speed_km_per_sec * seconds_per_hour = 3780 := by
  sorry

end moon_speed_km_per_hour_l2150_215065


namespace jaron_snickers_needed_l2150_215074

/-- The number of Snickers bars Jaron needs to sell to win the Nintendo Switch -/
def snickers_needed (total_points_needed : ℕ) (bunnies_sold : ℕ) (points_per_bunny : ℕ) (points_per_snickers : ℕ) : ℕ :=
  ((total_points_needed - bunnies_sold * points_per_bunny) + points_per_snickers - 1) / points_per_snickers

theorem jaron_snickers_needed :
  snickers_needed 2000 8 100 25 = 48 := by
  sorry

end jaron_snickers_needed_l2150_215074


namespace product_remainder_l2150_215019

theorem product_remainder (a b m : ℕ) (ha : a % m = 7) (hb : b % m = 1) (hm : m = 8) :
  (a * b) % m = 7 := by
  sorry

end product_remainder_l2150_215019


namespace cd_cost_calculation_l2150_215001

/-- The cost of the CD that Ibrahim wants to buy -/
def cd_cost : ℝ := 19

/-- The cost of the MP3 player -/
def mp3_cost : ℝ := 120

/-- Ibrahim's savings -/
def savings : ℝ := 55

/-- Money given by Ibrahim's father -/
def father_contribution : ℝ := 20

/-- The amount Ibrahim lacks after his savings and father's contribution -/
def amount_lacking : ℝ := 64

theorem cd_cost_calculation :
  cd_cost = mp3_cost + cd_cost - (savings + father_contribution) - amount_lacking :=
by sorry

end cd_cost_calculation_l2150_215001


namespace class_election_is_survey_conduction_l2150_215008

/-- Represents the steps in a survey process -/
inductive SurveyStep
  | DetermineObject
  | SelectMethod
  | Conduct
  | DrawConclusions

/-- Represents a voting process in a class election -/
structure ClassElection where
  students : Set Student
  candidates : Set Candidate
  ballot_box : Set Vote

/-- Definition of conducting a survey -/
def conducSurvey (process : ClassElection) : SurveyStep :=
  SurveyStep.Conduct

theorem class_election_is_survey_conduction (election : ClassElection) :
  conducSurvey election = SurveyStep.Conduct := by
  sorry

#check class_election_is_survey_conduction

end class_election_is_survey_conduction_l2150_215008


namespace min_value_constraint_l2150_215047

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := 4 * x^2 - 4 * a * x + a^2 - 2 * a + 2

-- Define the theorem
theorem min_value_constraint (a : ℝ) : 
  (∀ x ∈ Set.Icc 0 2, f a x ≥ 3) ∧ (∃ x ∈ Set.Icc 0 2, f a x = 3) ↔ 
  a = 1 - Real.sqrt 2 ∨ a = 5 + Real.sqrt 10 := by
  sorry

end min_value_constraint_l2150_215047


namespace quadratic_inequality_solution_set_l2150_215080

theorem quadratic_inequality_solution_set :
  ∀ x : ℝ, x^2 - 5*x + 6 > 0 ↔ x < 2 ∨ x > 3 := by
sorry

end quadratic_inequality_solution_set_l2150_215080


namespace inscribed_squares_perimeter_ratio_l2150_215017

theorem inscribed_squares_perimeter_ratio :
  let r : ℝ := 5
  let s₁ : ℝ := Real.sqrt ((2 * r^2) / 5)  -- side length of square in semicircle
  let s₂ : ℝ := r * Real.sqrt 2           -- side length of square in circle
  (4 * s₁) / (4 * s₂) = Real.sqrt 10 / 5 := by
sorry

end inscribed_squares_perimeter_ratio_l2150_215017


namespace remainder_theorem_l2150_215006

theorem remainder_theorem (x : Int) (h : x % 285 = 31) :
  (x % 17 = 14) ∧ (x % 23 = 8) ∧ (x % 19 = 12) := by
  sorry

end remainder_theorem_l2150_215006


namespace gum_pieces_per_package_l2150_215076

/-- The number of packages of gum Robin has -/
def num_packages : ℕ := 5

/-- The number of extra pieces of gum Robin has -/
def extra_pieces : ℕ := 6

/-- The total number of pieces of gum Robin has -/
def total_pieces : ℕ := 41

/-- The number of pieces of gum in each package -/
def pieces_per_package : ℕ := (total_pieces - extra_pieces) / num_packages

theorem gum_pieces_per_package : pieces_per_package = 7 := by
  sorry

end gum_pieces_per_package_l2150_215076


namespace min_value_theorem_l2150_215030

theorem min_value_theorem (a b c d : ℝ) (sum_constraint : a + b + c + d = 8) :
  20 * (a^2 + b^2 + c^2 + d^2) - (a^3 * b + a^3 * c + a^3 * d + b^3 * a + b^3 * c + b^3 * d + c^3 * a + c^3 * b + c^3 * d + d^3 * a + d^3 * b + d^3 * c) ≥ 112 :=
by sorry

end min_value_theorem_l2150_215030


namespace quadratic_inequality_solution_l2150_215015

theorem quadratic_inequality_solution (a b : ℝ) : 
  (∀ x : ℝ, ax^2 + bx + 2 > 0 ↔ -1/2 < x ∧ x < 1/3) → 
  a - b = -10 :=
by sorry

end quadratic_inequality_solution_l2150_215015


namespace sqrt_5_minus_1_bounds_l2150_215051

theorem sqrt_5_minus_1_bounds : 1 < Real.sqrt 5 - 1 ∧ Real.sqrt 5 - 1 < 2 := by
  sorry

end sqrt_5_minus_1_bounds_l2150_215051


namespace definite_integral_sqrt_minus_x_l2150_215028

open Set
open MeasureTheory
open Interval

theorem definite_integral_sqrt_minus_x :
  ∫ (x : ℝ) in (Icc 0 1), (Real.sqrt (1 - (x - 1)^2) - x) = π/4 - 1/2 := by
  sorry

end definite_integral_sqrt_minus_x_l2150_215028


namespace quadratic_inequality_equiv_interval_l2150_215031

theorem quadratic_inequality_equiv_interval (x : ℝ) :
  x^2 - 8*x + 15 < 0 ↔ 3 < x ∧ x < 5 :=
by sorry

end quadratic_inequality_equiv_interval_l2150_215031


namespace husband_additional_payment_l2150_215003

/-- Calculates the additional amount the husband needs to pay to split expenses equally for the house help -/
theorem husband_additional_payment (salary : ℝ) (medical_cost : ℝ) 
  (h1 : salary = 160)
  (h2 : medical_cost = 128)
  (h3 : salary ≥ medical_cost / 2) : 
  salary / 2 - medical_cost / 4 = 16 := by
  sorry

end husband_additional_payment_l2150_215003


namespace min_correct_answers_to_advance_l2150_215069

/-- Represents a math competition with specified rules -/
structure MathCompetition where
  total_questions : ℕ
  points_correct : ℕ
  points_incorrect : ℕ
  min_score : ℕ

/-- Calculates the score for a given number of correct answers in the competition -/
def calculate_score (comp : MathCompetition) (correct_answers : ℕ) : ℤ :=
  (correct_answers * comp.points_correct : ℤ) - 
  ((comp.total_questions - correct_answers) * comp.points_incorrect : ℤ)

/-- Theorem stating the minimum number of correct answers needed to advance -/
theorem min_correct_answers_to_advance (comp : MathCompetition) 
  (h1 : comp.total_questions = 25)
  (h2 : comp.points_correct = 4)
  (h3 : comp.points_incorrect = 1)
  (h4 : comp.min_score = 60) :
  ∃ (n : ℕ), n = 17 ∧ 
    (∀ (m : ℕ), m ≥ n → calculate_score comp m ≥ comp.min_score) ∧
    (∀ (m : ℕ), m < n → calculate_score comp m < comp.min_score) :=
  sorry

end min_correct_answers_to_advance_l2150_215069


namespace impossible_to_tile_with_sphinx_l2150_215070

/-- Represents a sphinx tile -/
structure SphinxTile :=
  (upward_triangles : Nat)
  (downward_triangles : Nat)

/-- Represents an equilateral triangle -/
structure EquilateralTriangle :=
  (side_length : Nat)

/-- Defines the properties of a sphinx tile -/
def is_valid_sphinx_tile (tile : SphinxTile) : Prop :=
  tile.upward_triangles + tile.downward_triangles = 6 ∧
  (tile.upward_triangles = 4 ∧ tile.downward_triangles = 2) ∨
  (tile.upward_triangles = 2 ∧ tile.downward_triangles = 4)

/-- Calculates the number of unit triangles in an equilateral triangle -/
def num_unit_triangles (triangle : EquilateralTriangle) : Nat :=
  triangle.side_length * (triangle.side_length + 1)

/-- Calculates the number of upward-pointing unit triangles -/
def num_upward_triangles (triangle : EquilateralTriangle) : Nat :=
  (triangle.side_length * (triangle.side_length - 1)) / 2

/-- Calculates the number of downward-pointing unit triangles -/
def num_downward_triangles (triangle : EquilateralTriangle) : Nat :=
  (triangle.side_length * (triangle.side_length + 1)) / 2

/-- Theorem stating the impossibility of tiling the triangle with sphinx tiles -/
theorem impossible_to_tile_with_sphinx (triangle : EquilateralTriangle) 
  (h1 : triangle.side_length = 6) : 
  ¬ ∃ (tiling : List SphinxTile), 
    (∀ tile ∈ tiling, is_valid_sphinx_tile tile) ∧ 
    (List.sum (tiling.map (λ tile => tile.upward_triangles)) = num_upward_triangles triangle) ∧
    (List.sum (tiling.map (λ tile => tile.downward_triangles)) = num_downward_triangles triangle) :=
sorry

end impossible_to_tile_with_sphinx_l2150_215070


namespace rectangle_area_increase_rectangle_area_percentage_increase_l2150_215094

theorem rectangle_area_increase (l w : ℝ) (hl : l > 0) (hw : w > 0) : 
  (1.3 * l) * (1.2 * w) = 1.56 * (l * w) := by
  sorry

theorem rectangle_area_percentage_increase (l w : ℝ) (hl : l > 0) (hw : w > 0) : 
  ((1.3 * l) * (1.2 * w) - l * w) / (l * w) = 0.56 := by
  sorry

end rectangle_area_increase_rectangle_area_percentage_increase_l2150_215094


namespace triangles_in_pentagon_l2150_215022

/-- The number of triangles formed when all diagonals are drawn in a pentagon -/
def num_triangles_in_pentagon : ℕ := 35

/-- Theorem stating that the number of triangles in a fully connected pentagon is 35 -/
theorem triangles_in_pentagon :
  num_triangles_in_pentagon = 35 := by
  sorry

#check triangles_in_pentagon

end triangles_in_pentagon_l2150_215022


namespace max_remainder_is_456_l2150_215024

/-- The maximum number on the board initially -/
def max_initial : ℕ := 2012

/-- The sum of numbers from 1 to n -/
def sum_to_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The number of operations performed -/
def num_operations : ℕ := max_initial - 1

/-- The final number N after all operations -/
def final_number : ℕ := sum_to_n max_initial * 2^num_operations

theorem max_remainder_is_456 : final_number % 1000 = 456 := by
  sorry

end max_remainder_is_456_l2150_215024


namespace star_3_7_l2150_215004

-- Define the star operation
def star (a b : ℕ) : ℕ := a^2 + 3*a*b + b^2

-- Theorem statement
theorem star_3_7 : star 3 7 = 121 := by
  sorry

end star_3_7_l2150_215004


namespace math_eng_only_is_five_l2150_215052

structure SubjectDistribution where
  total : ℕ
  mathEngOnly : ℕ
  mathHistOnly : ℕ
  engHistOnly : ℕ
  allThree : ℕ
  mathOnly : ℕ
  engOnly : ℕ
  histOnly : ℕ

def isValidDistribution (d : SubjectDistribution) : Prop :=
  d.total = 228 ∧
  d.mathEngOnly + d.mathHistOnly + d.engHistOnly + d.allThree + d.mathOnly + d.engOnly + d.histOnly = d.total ∧
  d.mathEngOnly = d.mathOnly ∧
  d.engOnly = 0 ∧
  d.histOnly = 0 ∧
  d.mathHistOnly = 6 ∧
  d.engHistOnly = 5 * d.allThree ∧
  d.allThree > 0 ∧
  d.allThree % 2 = 0

theorem math_eng_only_is_five (d : SubjectDistribution) (h : isValidDistribution d) : 
  d.mathEngOnly = 5 := by
  sorry

end math_eng_only_is_five_l2150_215052


namespace heartsuit_three_eight_l2150_215081

-- Define the ♥ operation
def heartsuit (x y : ℝ) : ℝ := 4 * x + 6 * y

-- Theorem statement
theorem heartsuit_three_eight : heartsuit 3 8 = 60 := by
  sorry

end heartsuit_three_eight_l2150_215081


namespace ilya_incorrect_l2150_215087

theorem ilya_incorrect : ¬∃ (s t : ℝ), s + t = s * t ∧ s + t = s / t := by
  sorry

end ilya_incorrect_l2150_215087


namespace magnitude_of_Z_l2150_215060

theorem magnitude_of_Z (Z : ℂ) (h : (1 - Complex.I) * Z = 1 + Complex.I) : Complex.abs Z = 1 := by
  sorry

end magnitude_of_Z_l2150_215060


namespace partition_positive_integers_l2150_215097

def is_arithmetic_sequence (x y z : ℕ) : Prop :=
  y - x = z - y ∧ x < y ∧ y < z

def has_infinite_arithmetic_subsequence (S : Set ℕ) : Prop :=
  ∃ (a d : ℕ), d ≠ 0 ∧ ∀ n : ℕ, (a + n * d) ∈ S

theorem partition_positive_integers :
  ∃ (A B : Set ℕ),
    (A ∪ B = {n : ℕ | n > 0}) ∧
    (A ∩ B = ∅) ∧
    (∀ x y z : ℕ, x ∈ A → y ∈ A → z ∈ A → x ≠ y → y ≠ z → x ≠ z →
      ¬is_arithmetic_sequence x y z) ∧
    ¬has_infinite_arithmetic_subsequence B :=
by sorry

end partition_positive_integers_l2150_215097


namespace aron_dusting_time_l2150_215002

/-- Represents the cleaning schedule and durations for Aron --/
structure CleaningSchedule where
  vacuum_duration : ℕ  -- Minutes spent vacuuming per day
  vacuum_frequency : ℕ  -- Days per week spent vacuuming
  dust_frequency : ℕ  -- Days per week spent dusting
  total_cleaning_time : ℕ  -- Total minutes spent cleaning per week

/-- Calculates the time spent dusting per day given a cleaning schedule --/
def dusting_time_per_day (schedule : CleaningSchedule) : ℕ :=
  let total_vacuum_time := schedule.vacuum_duration * schedule.vacuum_frequency
  let total_dust_time := schedule.total_cleaning_time - total_vacuum_time
  total_dust_time / schedule.dust_frequency

/-- Theorem stating that Aron spends 20 minutes dusting each day --/
theorem aron_dusting_time (schedule : CleaningSchedule) 
    (h1 : schedule.vacuum_duration = 30)
    (h2 : schedule.vacuum_frequency = 3)
    (h3 : schedule.dust_frequency = 2)
    (h4 : schedule.total_cleaning_time = 130) :
  dusting_time_per_day schedule = 20 := by
  sorry

end aron_dusting_time_l2150_215002


namespace consecutive_numbers_product_invariant_l2150_215033

theorem consecutive_numbers_product_invariant :
  ∃ (a : ℕ), 
    let original := [a, a+1, a+2, a+3, a+4, a+5, a+6]
    ∃ (modified : List ℕ),
      (∀ i, i ∈ original → ∃ j, j ∈ modified ∧ (j = i - 1 ∨ j = i ∨ j = i + 1)) ∧
      (modified.length = 7) ∧
      (original.prod = modified.prod) :=
by sorry

end consecutive_numbers_product_invariant_l2150_215033


namespace price_reduction_l2150_215041

theorem price_reduction (price_2010 : ℝ) (price_2011 price_2012 : ℝ) :
  price_2011 = price_2010 * (1 + 0.25) →
  price_2012 = price_2010 * (1 + 0.10) →
  price_2012 = price_2011 * (1 - 0.12) :=
by
  sorry

end price_reduction_l2150_215041


namespace hazel_eyed_brunettes_l2150_215034

/-- Represents the characteristics of students in a class -/
structure ClassCharacteristics where
  total_students : ℕ
  green_eyed_blondes : ℕ
  brunettes : ℕ
  hazel_eyed : ℕ

/-- Theorem: Number of hazel-eyed brunettes in the class -/
theorem hazel_eyed_brunettes (c : ClassCharacteristics) 
  (h1 : c.total_students = 60)
  (h2 : c.green_eyed_blondes = 20)
  (h3 : c.brunettes = 35)
  (h4 : c.hazel_eyed = 25) :
  c.total_students - (c.brunettes + c.green_eyed_blondes) = c.hazel_eyed - (c.total_students - c.brunettes) :=
by sorry

#check hazel_eyed_brunettes

end hazel_eyed_brunettes_l2150_215034


namespace moving_circle_trajectory_l2150_215021

/-- The locus of points satisfying the given conditions is one branch of a hyperbola -/
theorem moving_circle_trajectory (M : ℝ × ℝ) :
  (∃ (x y : ℝ), M = (x, y) ∧ x > 0) →
  (Real.sqrt (M.1^2 + M.2^2) - Real.sqrt ((M.1 - 3)^2 + M.2^2) = 2) →
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ M.1^2 / a^2 - M.2^2 / b^2 = 1 := by
  sorry

end moving_circle_trajectory_l2150_215021


namespace salt_mixture_percentage_l2150_215005

/-- The percentage of salt in the initial solution -/
def P : ℝ := sorry

/-- The amount of initial solution in ounces -/
def initial_amount : ℝ := 40

/-- The amount of 60% solution added in ounces -/
def added_amount : ℝ := 40

/-- The percentage of salt in the added solution -/
def added_percentage : ℝ := 60

/-- The total amount of the resulting mixture in ounces -/
def total_amount : ℝ := initial_amount + added_amount

/-- The percentage of salt in the resulting mixture -/
def result_percentage : ℝ := 40

theorem salt_mixture_percentage :
  P = 20 ∧
  (P / 100 * initial_amount + added_percentage / 100 * added_amount) / total_amount * 100 = result_percentage :=
sorry

end salt_mixture_percentage_l2150_215005


namespace problem_1_problem_2_l2150_215063

-- Problem 1
theorem problem_1 : 2 * Real.sqrt 18 - Real.sqrt 50 + (1/2) * Real.sqrt 32 = 3 * Real.sqrt 2 := by
  sorry

-- Problem 2
theorem problem_2 : (Real.sqrt 5 + Real.sqrt 6) * (Real.sqrt 5 - Real.sqrt 6) - (Real.sqrt 5 - 1)^2 = -7 + 2 * Real.sqrt 5 := by
  sorry

end problem_1_problem_2_l2150_215063


namespace carla_laundry_rate_l2150_215048

/-- Given a total number of laundry pieces and available hours, 
    calculate the number of pieces to be cleaned per hour. -/
def piecesPerHour (totalPieces : ℕ) (availableHours : ℕ) : ℕ :=
  totalPieces / availableHours

theorem carla_laundry_rate :
  piecesPerHour 80 4 = 20 := by
  sorry


end carla_laundry_rate_l2150_215048


namespace stating_gray_cube_count_gray_cube_count_3x3x3_gray_cube_count_5x5x5_l2150_215085

/-- 
Represents the number of 1x1x1 cubes with a specific number of gray faces 
in an nxnxn cube where all outer faces are painted gray.
-/
def grayCubes (n : ℕ) : ℕ × ℕ :=
  (6 * (n - 2)^2, (n - 2)^3)

/-- 
Theorem stating the correct number of 1x1x1 cubes with exactly one gray face 
and with no gray faces in an nxnxn cube where all outer faces are painted gray.
-/
theorem gray_cube_count (n : ℕ) (h : n ≥ 3) : 
  grayCubes n = (6 * (n - 2)^2, (n - 2)^3) := by
  sorry

/-- 
Corollary for the specific case of a 3x3x3 cube, giving the number of cubes 
with exactly one gray face and exactly two gray faces.
-/
theorem gray_cube_count_3x3x3 : 
  (grayCubes 3).1 = 6 ∧ 12 = 12 := by
  sorry

/-- 
Corollary for the specific case of a 5x5x5 cube, giving the number of cubes 
with exactly one gray face and with no gray faces.
-/
theorem gray_cube_count_5x5x5 : 
  (grayCubes 5).1 = 54 ∧ (grayCubes 5).2 = 27 := by
  sorry

end stating_gray_cube_count_gray_cube_count_3x3x3_gray_cube_count_5x5x5_l2150_215085


namespace manuscript_cost_theorem_l2150_215059

def manuscript_typing_cost (total_pages : ℕ) (initial_cost : ℕ) (revision_cost : ℕ) 
  (pages_revised_once : ℕ) (pages_revised_twice : ℕ) : ℕ :=
  (total_pages * initial_cost) + 
  (pages_revised_once * revision_cost) + 
  (pages_revised_twice * revision_cost * 2)

theorem manuscript_cost_theorem :
  manuscript_typing_cost 100 5 4 30 20 = 780 := by
  sorry

end manuscript_cost_theorem_l2150_215059


namespace principal_amount_l2150_215079

/-- Proves that given the conditions of the problem, the principal amount must be 600 --/
theorem principal_amount (P R : ℝ) : 
  (P * (R + 5) * 10) / 100 = (P * R * 10) / 100 + 300 →
  P = 600 := by
  sorry

end principal_amount_l2150_215079


namespace apple_banana_cost_l2150_215049

/-- The total cost of buying apples and bananas -/
def total_cost (a b : ℝ) : ℝ := 3 * a + 4 * b

/-- Theorem stating that the total cost of buying 3 kg of apples at 'a' yuan/kg
    and 4 kg of bananas at 'b' yuan/kg is (3a + 4b) yuan -/
theorem apple_banana_cost (a b : ℝ) :
  total_cost a b = 3 * a + 4 * b := by
  sorry

end apple_banana_cost_l2150_215049


namespace min_value_a_l2150_215067

theorem min_value_a (a : ℝ) : 
  (∀ x : ℝ, x > 0 → (1 / (x^2 + 1)) ≤ (a / x)) → 
  a ≥ 1/2 :=
sorry

end min_value_a_l2150_215067


namespace five_fourths_of_twelve_fifths_l2150_215020

theorem five_fourths_of_twelve_fifths : (5 / 4 : ℚ) * (12 / 5 : ℚ) = 3 := by
  sorry

end five_fourths_of_twelve_fifths_l2150_215020


namespace mike_spent_500_on_self_l2150_215046

def total_rose_bushes : ℕ := 6
def price_per_rose_bush : ℕ := 75
def rose_bushes_for_friend : ℕ := 2
def tiger_tooth_aloes : ℕ := 2
def price_per_aloe : ℕ := 100

def money_spent_on_self : ℕ :=
  (total_rose_bushes - rose_bushes_for_friend) * price_per_rose_bush +
  tiger_tooth_aloes * price_per_aloe

theorem mike_spent_500_on_self :
  money_spent_on_self = 500 := by
  sorry

end mike_spent_500_on_self_l2150_215046


namespace floor_neg_sqrt_64_over_9_l2150_215092

theorem floor_neg_sqrt_64_over_9 : ⌊-Real.sqrt (64 / 9)⌋ = -3 := by
  sorry

end floor_neg_sqrt_64_over_9_l2150_215092


namespace sophie_cookies_l2150_215042

/-- Represents the number of cookies Sophie bought -/
def num_cookies : ℕ := sorry

/-- The cost of a single cupcake -/
def cupcake_cost : ℚ := 2

/-- The cost of a single doughnut -/
def doughnut_cost : ℚ := 1

/-- The cost of a single slice of apple pie -/
def apple_pie_slice_cost : ℚ := 2

/-- The cost of a single cookie -/
def cookie_cost : ℚ := 6/10

/-- The total amount Sophie spent -/
def total_spent : ℚ := 33

/-- The number of cupcakes Sophie bought -/
def num_cupcakes : ℕ := 5

/-- The number of doughnuts Sophie bought -/
def num_doughnuts : ℕ := 6

/-- The number of apple pie slices Sophie bought -/
def num_apple_pie_slices : ℕ := 4

theorem sophie_cookies :
  num_cookies = 15 ∧
  (num_cupcakes : ℚ) * cupcake_cost +
  (num_doughnuts : ℚ) * doughnut_cost +
  (num_apple_pie_slices : ℚ) * apple_pie_slice_cost +
  (num_cookies : ℚ) * cookie_cost = total_spent :=
by sorry

end sophie_cookies_l2150_215042


namespace z_in_first_quadrant_l2150_215018

def complex_number (z : ℂ) : Prop :=
  z = (3 + Complex.I) / (1 - Complex.I)

theorem z_in_first_quadrant (z : ℂ) (h : complex_number z) :
  Complex.re z > 0 ∧ Complex.im z > 0 :=
sorry

end z_in_first_quadrant_l2150_215018


namespace power_three_2023_mod_10_l2150_215077

theorem power_three_2023_mod_10 : 3^2023 % 10 = 7 := by
  sorry

end power_three_2023_mod_10_l2150_215077


namespace geometric_and_arithmetic_sequence_problem_l2150_215089

-- Define the geometric sequence
def geometric_sequence (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ := a₁ * q^(n-1)

-- Define the b_n sequence
def b_sequence (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ := geometric_sequence a₁ q n + 2*n

-- Define the sum of the first n terms of b_n
def T_n (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ := 
  Finset.sum (Finset.range n) (λ i => b_sequence a₁ q (i+1))

theorem geometric_and_arithmetic_sequence_problem :
  ∀ a₁ q : ℝ,
  (a₁ > 0) →
  (q > 1) →
  (a₁ * (a₁*q) * (a₁*q^2) = 8) →
  (2*((a₁*q)+2) = (a₁+1) + ((a₁*q^2)+2)) →
  (a₁ = 1 ∧ q = 2) ∧
  (∀ n : ℕ, n > 0 → T_n 1 2 n = 2^n + n^2 + n - 1) :=
by sorry

end geometric_and_arithmetic_sequence_problem_l2150_215089


namespace box_volume_conversion_l2150_215056

/-- Converts cubic feet to cubic yards -/
def cubic_feet_to_cubic_yards (cubic_feet : ℚ) : ℚ :=
  cubic_feet / 27

theorem box_volume_conversion :
  let box_volume_cubic_feet : ℚ := 200
  let box_volume_cubic_yards : ℚ := cubic_feet_to_cubic_yards box_volume_cubic_feet
  box_volume_cubic_yards = 200 / 27 := by
  sorry

end box_volume_conversion_l2150_215056


namespace max_value_abc_l2150_215039

theorem max_value_abc (a b c : ℝ) (h : a + 3 * b + c = 5) :
  ∃ (max : ℝ), max = 25/3 ∧ ∀ (x y z : ℝ), x + 3 * y + z = 5 → x * y + x * z + y * z ≤ max :=
sorry

end max_value_abc_l2150_215039


namespace number_is_composite_l2150_215026

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- Represents the number formed by the given sequence of digits -/
def formNumber (digits : List Digit) : ℕ :=
  sorry

/-- Checks if a natural number is composite -/
def isComposite (n : ℕ) : Prop :=
  ∃ m, 1 < m ∧ m < n ∧ n % m = 0

/-- The main theorem to be proved -/
theorem number_is_composite (digits : List Digit) :
  isComposite (formNumber digits) :=
sorry

end number_is_composite_l2150_215026


namespace abc_product_l2150_215098

theorem abc_product (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (hab : a ≠ b) (hbc : b ≠ c) (hac : a ≠ c)
  (h1 : a + 1 = b + 2) (h2 : b + 2 = c + 3) :
  a * b * c = c * (c + 1) * (c + 2) := by
  sorry

end abc_product_l2150_215098


namespace gala_trees_count_l2150_215055

/-- Represents an apple orchard with Fuji and Gala trees -/
structure Orchard where
  total : ℕ
  fuji : ℕ
  gala : ℕ
  cross_pollinated : ℕ

/-- The conditions of the orchard problem -/
def orchard_conditions (o : Orchard) : Prop :=
  o.cross_pollinated = o.total / 10 ∧
  o.fuji + o.cross_pollinated = 204 ∧
  o.fuji = 3 * o.total / 4 ∧
  o.total = o.fuji + o.gala + o.cross_pollinated

theorem gala_trees_count (o : Orchard) (h : orchard_conditions o) : o.gala = 60 := by
  sorry

end gala_trees_count_l2150_215055


namespace arithmetic_sequence_common_difference_l2150_215062

theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℤ) -- The arithmetic sequence
  (d : ℤ) -- The common difference
  (h1 : a 0 = 23) -- First term is 23
  (h2 : ∀ n, a (n + 1) = a n + d) -- Arithmetic sequence definition
  (h3 : ∀ n, n < 6 → a n > 0) -- First 6 terms are positive
  (h4 : ∀ n, n ≥ 6 → a n < 0) -- Terms from 7th onward are negative
  : d = -4 := by
  sorry

end arithmetic_sequence_common_difference_l2150_215062


namespace cubic_equation_solution_l2150_215016

theorem cubic_equation_solution (a : ℝ) (h : a^2 - a - 1 = 0) : 
  a^3 - a^2 - a + 2023 = 2023 := by
  sorry

end cubic_equation_solution_l2150_215016


namespace minimum_packages_shipped_minimum_packages_value_l2150_215027

def sarahs_load : ℕ := 18
def ryans_load : ℕ := 11

theorem minimum_packages_shipped (n : ℕ) :
  (n % sarahs_load = 0) ∧ (n % ryans_load = 0) →
  n ≥ Nat.lcm sarahs_load ryans_load :=
by sorry

theorem minimum_packages_value :
  Nat.lcm sarahs_load ryans_load = 198 :=
by sorry

end minimum_packages_shipped_minimum_packages_value_l2150_215027


namespace hyperbola_eccentricity_l2150_215071

/-- The eccentricity of a hyperbola with equation x²/a² - y²/b² = 1 is √(1 + b²/a²) -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let e := Real.sqrt (1 + b^2 / a^2)
  (∀ x y, x^2 / a^2 - y^2 / b^2 = 1) → e = 3 / 2 :=
by sorry

end hyperbola_eccentricity_l2150_215071


namespace quadratic_points_order_l2150_215086

/-- Given a quadratic function f(x) = x² - 4x - m, prove that the y-coordinates
    of the points (-1, y₃), (3, y₂), and (2, y₁) on this function satisfy y₃ > y₂ > y₁ -/
theorem quadratic_points_order (m : ℝ) :
  let f : ℝ → ℝ := λ x ↦ x^2 - 4*x - m
  let y₁ : ℝ := f 2
  let y₂ : ℝ := f 3
  let y₃ : ℝ := f (-1)
  y₃ > y₂ ∧ y₂ > y₁ :=
by sorry

end quadratic_points_order_l2150_215086


namespace sequence_b_is_geometric_progression_l2150_215010

def sequence_a (a : ℝ) (n : ℕ) : ℝ := 
  if n = 1 then a else 3 * (4 ^ (n - 1)) + 2 * (a - 4) * (3 ^ (n - 2))

def sum_S (a : ℝ) (n : ℕ) : ℝ := 
  (4 ^ n) + (a - 4) * (3 ^ (n - 1))

def sequence_b (a : ℝ) (n : ℕ) : ℝ := 
  sum_S a n - (4 ^ n)

theorem sequence_b_is_geometric_progression (a : ℝ) (h : a ≠ 4) :
  ∀ n : ℕ, n ≥ 1 → sequence_b a (n + 1) = 3 * sequence_b a n := by
  sorry

end sequence_b_is_geometric_progression_l2150_215010


namespace hair_cut_calculation_l2150_215082

/-- Given the total amount of hair cut and the amount cut on the first day,
    calculate the amount cut on the second day. -/
theorem hair_cut_calculation (total : ℝ) (first_day : ℝ) (h1 : total = 0.88) (h2 : first_day = 0.38) :
  total - first_day = 0.50 := by
  sorry

end hair_cut_calculation_l2150_215082


namespace trig_product_equals_one_l2150_215040

theorem trig_product_equals_one :
  let cos30 : ℝ := Real.sqrt 3 / 2
  let sin60 : ℝ := Real.sqrt 3 / 2
  let sin30 : ℝ := 1 / 2
  let cos60 : ℝ := 1 / 2
  (1 - 1 / cos30) * (1 + 1 / sin60) * (1 - 1 / sin30) * (1 + 1 / cos60) = 1 := by
  sorry

end trig_product_equals_one_l2150_215040


namespace pure_imaginary_z_l2150_215023

theorem pure_imaginary_z (a : ℝ) : 
  (∃ (b : ℝ), (1 - a * Complex.I) / (1 + a * Complex.I) = Complex.I * b) → 
  (a = 1 ∨ a = -1) := by
sorry

end pure_imaginary_z_l2150_215023


namespace complex_exponential_sum_l2150_215044

theorem complex_exponential_sum (θ φ : ℝ) :
  Complex.exp (Complex.I * θ) + Complex.exp (Complex.I * φ) = (1 / 3 : ℂ) + (2 / 5 : ℂ) * Complex.I →
  Complex.exp (-Complex.I * θ) + Complex.exp (-Complex.I * φ) = (1 / 3 : ℂ) - (2 / 5 : ℂ) * Complex.I :=
by sorry

end complex_exponential_sum_l2150_215044


namespace range_of_Z_l2150_215050

theorem range_of_Z (a b : ℝ) (h : a^2 + 3*a*b + 9*b^2 = 4) :
  ∃ (z : ℝ), z = a^2 + 9*b^2 ∧ 8/3 ≤ z ∧ z ≤ 8 ∧
  (∀ (w : ℝ), w = a^2 + 9*b^2 → 8/3 ≤ w ∧ w ≤ 8) :=
by sorry

end range_of_Z_l2150_215050


namespace sum_of_bases_equality_l2150_215099

/-- Converts a number from base 13 to base 10 -/
def base13ToBase10 (n : ℕ) : ℕ := sorry

/-- Converts a number from base 14 to base 10 -/
def base14ToBase10 (n : ℕ) : ℕ := sorry

/-- Represents the digit C in base 14 -/
def C : ℕ := 12

theorem sum_of_bases_equality : 
  base13ToBase10 372 + base14ToBase10 (4 * 14^2 + C * 14 + 5) = 1557 := by sorry

end sum_of_bases_equality_l2150_215099


namespace seven_presenter_schedule_l2150_215000

/-- The number of ways to schedule n presenters with one specific presenter following another --/
def schedule_presenters (n : ℕ) : ℕ :=
  Nat.factorial n / 2

/-- Theorem: For 7 presenters, with one following another, there are 2520 ways to schedule --/
theorem seven_presenter_schedule :
  schedule_presenters 7 = 2520 := by
  sorry

end seven_presenter_schedule_l2150_215000


namespace quadrupled_bonus_remainder_l2150_215061

/-- Represents the bonus pool and its division among employees -/
structure BonusPool :=
  (total : ℕ)
  (employees : ℕ)
  (remainder : ℕ)

/-- Theorem stating the relationship between the original and quadrupled bonus pools -/
theorem quadrupled_bonus_remainder
  (original : BonusPool)
  (h1 : original.employees = 8)
  (h2 : original.remainder = 5)
  (quadrupled : BonusPool)
  (h3 : quadrupled.employees = original.employees)
  (h4 : quadrupled.total = 4 * original.total) :
  quadrupled.remainder = 4 := by
sorry

end quadrupled_bonus_remainder_l2150_215061


namespace subtraction_multiplication_fractions_l2150_215083

theorem subtraction_multiplication_fractions :
  (5 / 12 - 1 / 6) * (3 / 4) = 3 / 16 := by
  sorry

end subtraction_multiplication_fractions_l2150_215083
