import Mathlib

namespace locus_of_points_m_l430_43081

-- Define the given circle
structure GivenCircle where
  O : ℝ × ℝ  -- Center of the circle
  R : ℝ      -- Radius of the circle
  h : R > 0  -- Radius is positive

-- Define the point A on the given circle
def PointOnCircle (c : GivenCircle) (A : ℝ × ℝ) : Prop :=
  (A.1 - c.O.1)^2 + (A.2 - c.O.2)^2 = c.R^2

-- Define the tangent line at point A
def TangentLine (c : GivenCircle) (A : ℝ × ℝ) : ℝ × ℝ → Prop :=
  λ M => (M.1 - A.1) * (A.1 - c.O.1) + (M.2 - A.2) * (A.2 - c.O.2) = 0

-- Define the segment AM with length a
def SegmentAM (A M : ℝ × ℝ) (a : ℝ) : Prop :=
  (M.1 - A.1)^2 + (M.2 - A.2)^2 = a^2

-- Theorem: The locus of points M forms a circle concentric with the given circle
theorem locus_of_points_m (c : GivenCircle) (a : ℝ) (h : a > 0) :
  ∀ A M : ℝ × ℝ,
    PointOnCircle c A →
    TangentLine c A M →
    SegmentAM A M a →
    (M.1 - c.O.1)^2 + (M.2 - c.O.2)^2 = c.R^2 + a^2 :=
  sorry

end locus_of_points_m_l430_43081


namespace zeros_of_f_l430_43090

def f (x : ℝ) : ℝ := x^3 - 2*x^2 - x + 2

theorem zeros_of_f :
  ∀ x : ℝ, f x = 0 ↔ x = -1 ∨ x = 1 ∨ x = 2 := by
  sorry

end zeros_of_f_l430_43090


namespace gardner_brownies_l430_43098

theorem gardner_brownies :
  ∀ (students cookies cupcakes brownies total_treats : ℕ),
    students = 20 →
    cookies = 20 →
    cupcakes = 25 →
    total_treats = students * 4 →
    total_treats = cookies + cupcakes + brownies →
    brownies = 35 := by
  sorry

end gardner_brownies_l430_43098


namespace tan_alpha_value_l430_43024

theorem tan_alpha_value (α : ℝ) (h : (Real.sin α - 2 * Real.cos α) / (Real.sin α + Real.cos α) = -1) :
  Real.tan α = 1/2 := by
  sorry

end tan_alpha_value_l430_43024


namespace min_stamps_for_50_cents_l430_43069

/-- Represents the number of stamps of each denomination -/
structure StampCombination where
  threes : ℕ
  fours : ℕ
  sixes : ℕ

/-- Calculates the total value of stamps in cents -/
def totalValue (s : StampCombination) : ℕ :=
  3 * s.threes + 4 * s.fours + 6 * s.sixes

/-- Calculates the total number of stamps -/
def totalStamps (s : StampCombination) : ℕ :=
  s.threes + s.fours + s.sixes

/-- Checks if a stamp combination is valid (totals 50 cents) -/
def isValid (s : StampCombination) : Prop :=
  totalValue s = 50

/-- Theorem: The minimum number of stamps to make 50 cents is 10 -/
theorem min_stamps_for_50_cents :
  (∃ (s : StampCombination), isValid s ∧ totalStamps s = 10) ∧
  (∀ (s : StampCombination), isValid s → totalStamps s ≥ 10) :=
by sorry

end min_stamps_for_50_cents_l430_43069


namespace inequality_proof_l430_43066

theorem inequality_proof (a b c d : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (hmax : d = max a (max b c)) : 
  a * (d - b) + b * (d - c) + c * (d - a) ≤ d^2 := by
  sorry

end inequality_proof_l430_43066


namespace fourth_term_is_sixty_l430_43051

/-- Represents a stratified sample drawn from an arithmetic sequence of questionnaires. -/
structure StratifiedSample where
  total_questionnaires : ℕ
  sample_size : ℕ
  second_term : ℕ
  h_total : total_questionnaires = 1000
  h_sample : sample_size = 150
  h_second : second_term = 30

/-- The number of questionnaires drawn from the fourth term of the sequence. -/
def fourth_term (s : StratifiedSample) : ℕ := 60

/-- Theorem stating that the fourth term of the stratified sample is 60. -/
theorem fourth_term_is_sixty (s : StratifiedSample) : fourth_term s = 60 := by
  sorry

end fourth_term_is_sixty_l430_43051


namespace square_of_one_minus_i_l430_43068

theorem square_of_one_minus_i (i : ℂ) : i^2 = -1 → (1 - i)^2 = -2*i := by
  sorry

end square_of_one_minus_i_l430_43068


namespace roger_used_crayons_l430_43085

/-- The number of used crayons Roger had -/
def used_crayons : ℕ := 14 - 2 - 8

/-- The total number of crayons Roger had -/
def total_crayons : ℕ := 14

/-- The number of new crayons Roger had -/
def new_crayons : ℕ := 2

/-- The number of broken crayons Roger had -/
def broken_crayons : ℕ := 8

theorem roger_used_crayons : 
  used_crayons + new_crayons + broken_crayons = total_crayons ∧ used_crayons = 4 := by
  sorry

end roger_used_crayons_l430_43085


namespace tea_cost_price_l430_43048

/-- The cost price of 80 kg of tea per kg -/
def C : ℝ := 15

/-- The theorem stating the cost price of 80 kg of tea per kg -/
theorem tea_cost_price :
  -- 80 kg of tea is mixed with 20 kg of tea at cost price of 20 per kg
  -- The sale price of the mixed tea is 20 per kg
  -- The trader wants to earn a profit of 25%
  (80 * C + 20 * 20) * 1.25 = 100 * 20 :=
by
  sorry

end tea_cost_price_l430_43048


namespace power_difference_theorem_l430_43014

def solution_set : Set (ℕ × ℕ) := {(0, 1), (2, 1), (2, 2), (1, 2)}

theorem power_difference_theorem :
  {(m, n) : ℕ × ℕ | (3:ℤ)^m - (2:ℤ)^n ∈ ({-1, 5, 7} : Set ℤ)} = solution_set :=
by sorry

end power_difference_theorem_l430_43014


namespace circle_radius_from_parabola_tangency_l430_43009

/-- The radius of a circle given specific tangency conditions of a parabola -/
theorem circle_radius_from_parabola_tangency : ∃ (r : ℝ), 
  (∀ x y : ℝ, y = x^2 + r → y ≤ x) ∧ 
  (∃ x : ℝ, x^2 + r = x) ∧
  r = (1 : ℝ) / 4 :=
sorry

end circle_radius_from_parabola_tangency_l430_43009


namespace pythagorean_triple_double_l430_43046

theorem pythagorean_triple_double (a b c : ℤ) :
  (a^2 + b^2 = c^2) → ((2*a)^2 + (2*b)^2 = (2*c)^2) := by
  sorry

end pythagorean_triple_double_l430_43046


namespace circle_equation_l430_43077

/-- The equation of a circle passing through (0,0), (4,0), and (-1,1) -/
theorem circle_equation (x y : ℝ) : 
  (x^2 + y^2 - 4*x - 6*y = 0) ↔ 
  ((x = 0 ∧ y = 0) ∨ (x = 4 ∧ y = 0) ∨ (x = -1 ∧ y = 1)) := by
  sorry

end circle_equation_l430_43077


namespace max_perimeter_is_29_l430_43042

/-- Represents a triangle with two fixed sides of length 7 and 8, and a variable third side y --/
structure Triangle where
  y : ℤ
  is_valid : 0 < y ∧ y < 7 + 8 ∧ 7 < y + 8 ∧ 8 < y + 7

/-- The perimeter of the triangle --/
def perimeter (t : Triangle) : ℤ := 7 + 8 + t.y

/-- Theorem stating that the maximum perimeter is 29 --/
theorem max_perimeter_is_29 :
  ∀ t : Triangle, perimeter t ≤ 29 ∧ ∃ t' : Triangle, perimeter t' = 29 := by
  sorry

#check max_perimeter_is_29

end max_perimeter_is_29_l430_43042


namespace david_spent_half_ben_spent_more_total_spent_is_48_l430_43072

/-- The amount Ben spent at the bagel store -/
def ben_spent : ℝ := 32

/-- The amount David spent at the bagel store -/
def david_spent : ℝ := 16

/-- For every dollar Ben spent, David spent 50 cents less -/
theorem david_spent_half : david_spent = ben_spent / 2 := by sorry

/-- Ben paid $16.00 more than David -/
theorem ben_spent_more : ben_spent = david_spent + 16 := by sorry

/-- The total amount spent by Ben and David -/
def total_spent : ℝ := ben_spent + david_spent

theorem total_spent_is_48 : total_spent = 48 := by sorry

end david_spent_half_ben_spent_more_total_spent_is_48_l430_43072


namespace farm_land_allocation_l430_43074

theorem farm_land_allocation (total_land : ℕ) (reserved : ℕ) (cattle : ℕ) (crops : ℕ) 
  (h1 : total_land = 150)
  (h2 : reserved = 15)
  (h3 : cattle = 40)
  (h4 : crops = 70) :
  total_land - reserved - cattle - crops = 25 := by
  sorry

end farm_land_allocation_l430_43074


namespace intersection_M_N_l430_43050

def M : Set ℝ := {x | (x - 1)^2 < 4}

def N : Set ℝ := {-1, 0, 1, 2, 3}

theorem intersection_M_N : M ∩ N = {0, 1, 2} := by
  sorry

end intersection_M_N_l430_43050


namespace probability_under_20_l430_43099

theorem probability_under_20 (total : ℕ) (over_30 : ℕ) (under_20 : ℕ) :
  total = 100 →
  over_30 = 90 →
  under_20 = total - over_30 →
  (under_20 : ℚ) / (total : ℚ) = 1 / 10 := by
  sorry

end probability_under_20_l430_43099


namespace min_distance_to_line_l430_43043

theorem min_distance_to_line (x y : ℝ) :
  8 * x + 15 * y = 120 →
  ∃ (min_val : ℝ), min_val = 120 / 17 ∧
    ∀ (x' y' : ℝ), 8 * x' + 15 * y' = 120 →
      Real.sqrt (x'^2 + y'^2) ≥ min_val :=
by sorry

end min_distance_to_line_l430_43043


namespace tyson_basketball_scores_l430_43088

/-- Represents the number of times Tyson scored points in each category -/
structure BasketballScores where
  threePointers : Nat
  twoPointers : Nat
  onePointers : Nat

/-- Calculates the total points scored given a BasketballScores structure -/
def totalPoints (scores : BasketballScores) : Nat :=
  3 * scores.threePointers + 2 * scores.twoPointers + scores.onePointers

theorem tyson_basketball_scores :
  ∃ (scores : BasketballScores),
    scores.threePointers = 15 ∧
    scores.twoPointers = 12 ∧
    scores.onePointers % 2 = 0 ∧
    totalPoints scores = 75 ∧
    scores.onePointers = 6 := by
  sorry

end tyson_basketball_scores_l430_43088


namespace valid_student_totals_l430_43015

/-- Represents the distribution of students in groups -/
structure StudentDistribution where
  total_groups : Nat
  groups_with_13 : Nat
  total_students : Nat

/-- Checks if a given distribution is valid according to the problem conditions -/
def is_valid_distribution (d : StudentDistribution) : Prop :=
  d.total_groups = 6 ∧
  d.groups_with_13 = 4 ∧
  (d.total_students = 76 ∨ d.total_students = 80)

/-- Theorem stating that the only valid total numbers of students are 76 and 80 -/
theorem valid_student_totals :
  ∀ d : StudentDistribution,
    is_valid_distribution d →
    (d.total_students = 76 ∨ d.total_students = 80) :=
by
  sorry

#check valid_student_totals

end valid_student_totals_l430_43015


namespace second_difference_quadratic_l430_43039

theorem second_difference_quadratic 
  (f : ℕ → ℝ) 
  (h : ∀ n : ℕ, f (n + 2) - 2 * f (n + 1) + f n = 1) : 
  ∃ a b : ℝ, ∀ n : ℕ, f n = (1/2) * n^2 + a * n + b := by
sorry

end second_difference_quadratic_l430_43039


namespace line_segment_length_l430_43080

/-- Given two points M(-2, a) and N(a, 4) on a line with slope -1/2,
    prove that the distance between M and N is 6√3. -/
theorem line_segment_length (a : ℝ) : 
  (4 - a) / (a + 2) = -1/2 →
  Real.sqrt ((a + 2)^2 + (4 - a)^2) = 6 * Real.sqrt 3 := by
  sorry

end line_segment_length_l430_43080


namespace distinct_prime_factors_count_l430_43047

def product : ℕ := 77 * 79 * 81 * 83

theorem distinct_prime_factors_count :
  (Nat.factors product).toFinset.card = 5 := by sorry

end distinct_prime_factors_count_l430_43047


namespace smallest_number_with_given_remainders_l430_43023

theorem smallest_number_with_given_remainders : 
  ∃ (a : ℕ), a = 74 ∧ 
  (∀ (n : ℕ), n < a → 
    (n % 3 ≠ 2 ∨ n % 5 ≠ 4 ∨ n % 7 ≠ 4)) ∧
  74 % 3 = 2 ∧ 74 % 5 = 4 ∧ 74 % 7 = 4 := by
sorry

end smallest_number_with_given_remainders_l430_43023


namespace power_difference_equals_multiple_of_thirty_power_l430_43057

theorem power_difference_equals_multiple_of_thirty_power : 
  (5^1002 + 6^1001)^2 - (5^1002 - 6^1001)^2 = 24 * 30^1001 := by
sorry

end power_difference_equals_multiple_of_thirty_power_l430_43057


namespace product_of_roots_l430_43002

theorem product_of_roots (x : ℂ) : 
  (2 * x^3 - 3 * x^2 - 8 * x + 10 = 0) → 
  (∃ r₁ r₂ r₃ : ℂ, (x - r₁) * (x - r₂) * (x - r₃) = 2 * x^3 - 3 * x^2 - 8 * x + 10 ∧ r₁ * r₂ * r₃ = -5) :=
by sorry

end product_of_roots_l430_43002


namespace reverse_divisibility_implies_divides_99_l430_43096

-- Define a function to reverse the digits of a natural number
def reverse_digits (n : ℕ) : ℕ := sorry

-- Define the property of k
def has_reverse_divisibility_property (k : ℕ) : Prop :=
  ∀ n : ℕ, k ∣ n → k ∣ reverse_digits n

-- Theorem statement
theorem reverse_divisibility_implies_divides_99 (k : ℕ) :
  has_reverse_divisibility_property k → k ∣ 99 := by sorry

end reverse_divisibility_implies_divides_99_l430_43096


namespace largest_x_equation_l430_43063

theorem largest_x_equation (x : ℝ) : 
  (((14 * x^3 - 40 * x^2 + 20 * x - 4) / (4 * x - 3) + 6 * x = 8 * x - 3) ↔ 
  (14 * x^3 - 48 * x^2 + 38 * x - 13 = 0)) ∧ 
  (∀ y : ℝ, ((14 * y^3 - 40 * y^2 + 20 * y - 4) / (4 * y - 3) + 6 * y = 8 * y - 3) → y ≤ x) := by
  sorry

end largest_x_equation_l430_43063


namespace largest_n_for_square_sum_l430_43007

theorem largest_n_for_square_sum : ∃ (n : ℕ), n = 1490 ∧ 
  (∀ m : ℕ, m > n → ¬ ∃ k : ℕ, 4^995 + 4^1500 + 4^m = k^2) ∧
  (∃ k : ℕ, 4^995 + 4^1500 + 4^n = k^2) := by
  sorry

end largest_n_for_square_sum_l430_43007


namespace ratio_sum_theorem_l430_43052

theorem ratio_sum_theorem (a b c d : ℝ) : 
  b = 2 * a ∧ c = 4 * a ∧ d = 5 * a ∧ 
  a^2 + b^2 + c^2 + d^2 = 2460 →
  abs ((a + b + c + d) - 87.744) < 0.001 := by
  sorry

end ratio_sum_theorem_l430_43052


namespace percentage_problem_l430_43055

theorem percentage_problem : 
  ∃ p : ℝ, p * 24 = 0.12 ∧ p = 0.005 := by
  sorry

end percentage_problem_l430_43055


namespace problem_1_l430_43018

theorem problem_1 : Real.sqrt 32 - 3 * Real.sqrt (1/2) + Real.sqrt 2 = (7/2) * Real.sqrt 2 := by
  sorry

end problem_1_l430_43018


namespace p_true_and_q_false_l430_43034

-- Define proposition p
def p : Prop := ∀ x y : ℝ, x > y → -x < -y

-- Define proposition q
def q : Prop := ∀ x y : ℝ, x ≠ 0 ∧ y ≠ 0 → (1 / x > 1 / y → x < y)

-- Theorem to prove
theorem p_true_and_q_false : p ∧ ¬q := by
  sorry

end p_true_and_q_false_l430_43034


namespace quadratic_roots_sum_reciprocal_l430_43076

theorem quadratic_roots_sum_reciprocal (a b : ℝ) : 
  a^2 - 6*a - 5 = 0 → 
  b^2 - 6*b - 5 = 0 → 
  a ≠ 0 → 
  b ≠ 0 → 
  1/a + 1/b = -6/5 := by sorry

end quadratic_roots_sum_reciprocal_l430_43076


namespace problem_solution_l430_43086

theorem problem_solution (x y : ℚ) : 
  x / y = 12 / 5 → y = 25 → x = 60 := by sorry

end problem_solution_l430_43086


namespace words_with_vowels_count_l430_43033

def alphabet : Finset Char := {'A', 'B', 'C', 'D', 'E', 'F'}
def vowels : Finset Char := {'A', 'E'}
def word_length : Nat := 5

def total_words : Nat := alphabet.card ^ word_length
def words_without_vowels : Nat := (alphabet \ vowels).card ^ word_length

theorem words_with_vowels_count :
  total_words - words_without_vowels = 6752 :=
sorry

end words_with_vowels_count_l430_43033


namespace soap_amount_is_fifteen_l430_43083

/-- Represents the recipe for bubble mix -/
structure BubbleMixRecipe where
  soap_per_cup : ℚ  -- tablespoons of soap per cup of water
  ounces_per_cup : ℚ  -- ounces in a cup of water

/-- Represents a container for bubble mix -/
structure BubbleMixContainer where
  capacity : ℚ  -- capacity in ounces

/-- Calculates the amount of soap needed for a given container and recipe -/
def soap_needed (recipe : BubbleMixRecipe) (container : BubbleMixContainer) : ℚ :=
  (container.capacity / recipe.ounces_per_cup) * recipe.soap_per_cup

/-- Theorem: The amount of soap needed for the given recipe and container is 15 tablespoons -/
theorem soap_amount_is_fifteen (recipe : BubbleMixRecipe) (container : BubbleMixContainer) 
    (h1 : recipe.soap_per_cup = 3)
    (h2 : recipe.ounces_per_cup = 8)
    (h3 : container.capacity = 40) :
    soap_needed recipe container = 15 := by
  sorry

end soap_amount_is_fifteen_l430_43083


namespace parabola_focus_on_line_l430_43053

-- Define the line on which the focus lies
def focus_line (x y : ℝ) : Prop := x - 2*y - 4 = 0

-- Define the standard equations of parabolas
def parabola_eq1 (x y : ℝ) : Prop := y^2 = 16*x
def parabola_eq2 (x y : ℝ) : Prop := x^2 = -8*y

-- Theorem statement
theorem parabola_focus_on_line :
  ∀ (x y : ℝ), focus_line x y →
  (∃ (a b : ℝ), parabola_eq1 a b) ∨ (∃ (c d : ℝ), parabola_eq2 c d) :=
sorry

end parabola_focus_on_line_l430_43053


namespace jake_final_bitcoins_l430_43025

/-- Represents the number of bitcoins Jake has at each step -/
def bitcoin_transactions (initial : ℕ) (donation1 : ℕ) (donation2 : ℕ) : ℕ :=
  let after_donation1 := initial - donation1
  let after_brother := after_donation1 / 2
  let after_triple := after_brother * 3
  after_triple - donation2

/-- Theorem stating that Jake ends up with 80 bitcoins after all transactions -/
theorem jake_final_bitcoins :
  bitcoin_transactions 80 20 10 = 80 := by
  sorry

end jake_final_bitcoins_l430_43025


namespace three_times_work_days_l430_43026

/-- The number of days Aarti needs to complete one piece of work -/
def base_work_days : ℕ := 9

/-- The number of times the work is multiplied -/
def work_multiplier : ℕ := 3

/-- Theorem: The time required to complete three times the work is 27 days -/
theorem three_times_work_days : base_work_days * work_multiplier = 27 := by
  sorry

end three_times_work_days_l430_43026


namespace triangle_properties_l430_43059

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The main theorem about the triangle -/
theorem triangle_properties (t : Triangle) 
  (h1 : t.a * Real.sin t.B - t.b * Real.cos t.A = 0)
  (h2 : t.a = 2 * Real.sqrt 5)
  (h3 : t.b = 2) :
  t.A = π / 4 ∧ (1/2 * t.b * t.c * Real.sin t.A = 4) := by
  sorry

end triangle_properties_l430_43059


namespace battle_station_staffing_ways_l430_43001

/-- Represents the number of job openings -/
def num_jobs : ℕ := 5

/-- Represents the total number of candidates considered -/
def total_candidates : ℕ := 18

/-- Represents the number of candidates skilled in one area only -/
def specialized_candidates : ℕ := 6

/-- Represents the number of versatile candidates -/
def versatile_candidates : ℕ := total_candidates - specialized_candidates

/-- Represents the number of ways to select the specialized candidates -/
def specialized_selection_ways : ℕ := 2 * 2 * 1 * 1

/-- The main theorem stating the number of ways to staff the battle station -/
theorem battle_station_staffing_ways :
  specialized_selection_ways * versatile_candidates * (versatile_candidates - 1) = 528 := by
  sorry

end battle_station_staffing_ways_l430_43001


namespace siblings_ages_l430_43045

-- Define the ages of the siblings
def hans_age : ℕ := 8
def annika_age : ℕ := 25
def emil_age : ℕ := 5
def frida_age : ℕ := 20

-- Define the conditions
def condition1 : Prop := annika_age + 4 = 3 * (hans_age + 4)
def condition2 : Prop := emil_age + 4 = 2 * (hans_age + 4)
def condition3 : Prop := (emil_age + 4) - (hans_age + 4) = (frida_age + 4) / 2
def condition4 : Prop := hans_age + annika_age + emil_age + frida_age = 58
def condition5 : Prop := frida_age + 5 = annika_age

-- Theorem statement
theorem siblings_ages :
  condition1 ∧ condition2 ∧ condition3 ∧ condition4 ∧ condition5 →
  annika_age = 25 ∧ emil_age = 5 ∧ frida_age = 20 :=
by
  sorry

end siblings_ages_l430_43045


namespace equation_solution_range_l430_43012

theorem equation_solution_range (k : ℝ) :
  (∃ x : ℝ, (4 * (2015^x) - 2015^(-x)) / (2015^x - 3 * (2015^(-x))) = k) ↔ 
  (k < 1/3 ∨ k > 4) := by
sorry

end equation_solution_range_l430_43012


namespace unique_solution_when_a_is_one_l430_43030

-- Define the equation
def equation (a x : ℝ) : Prop :=
  (5 : ℝ) ^ (x^2 + 2*a*x + a^2) = a*x^2 + 2*a^2*x + a^3 + a^2 - 6*a + 6

-- Theorem statement
theorem unique_solution_when_a_is_one :
  ∃! a : ℝ, ∃! x : ℝ, equation a x :=
by
  sorry

end unique_solution_when_a_is_one_l430_43030


namespace domino_distribution_l430_43022

theorem domino_distribution (total_dominoes : Nat) (num_players : Nat) 
  (h1 : total_dominoes = 28) 
  (h2 : num_players = 4) : 
  total_dominoes / num_players = 7 := by
  sorry

end domino_distribution_l430_43022


namespace age_difference_l430_43094

theorem age_difference : ∀ (a b : ℕ), 
  (a < 10 ∧ b < 10) →  -- a and b are single digits
  (10 * a + b + 5 = 3 * (10 * b + a + 5)) →  -- In 5 years, Rachel's age will be three times Sam's age
  ((10 * a + b) - (10 * b + a) = 63) :=  -- The difference in their current ages is 63
by
  sorry

end age_difference_l430_43094


namespace ned_candy_boxes_l430_43010

/-- The number of candy pieces Ned gave to his little brother -/
def pieces_given : ℝ := 7.0

/-- The number of candy pieces in each box -/
def pieces_per_box : ℝ := 6.0

/-- The number of candy pieces Ned still has -/
def pieces_left : ℕ := 42

/-- The number of boxes Ned bought initially -/
def boxes_bought : ℕ := 8

theorem ned_candy_boxes : 
  ⌊(pieces_given + pieces_left : ℝ) / pieces_per_box⌋ = boxes_bought := by
  sorry

end ned_candy_boxes_l430_43010


namespace quadratic_inequality_solution_range_l430_43097

theorem quadratic_inequality_solution_range (a : ℝ) : 
  (∃ x : ℝ, 1 < x ∧ x < 4 ∧ 2 * x^2 - 8 * x - 4 - a > 0) ↔ 
  (-16 < a ∧ a < -8) := by
  sorry

end quadratic_inequality_solution_range_l430_43097


namespace circle_symmetry_theorem_l430_43071

/-- The equation of the circle C -/
def circle_equation (x y a : ℝ) : Prop :=
  x^2 + y^2 + 4*x + a*y - 5 = 0

/-- The equation of the line -/
def line_equation (x y : ℝ) : Prop :=
  2*x + y - 1 = 0

/-- The theorem stating the relationship between the circle and the line -/
theorem circle_symmetry_theorem (a : ℝ) : 
  (∀ x y : ℝ, circle_equation x y a → 
    ∃ x' y' : ℝ, circle_equation x' y' a ∧ 
    ((x + x')/2, (y + y')/2) ∈ {(x, y) | line_equation x y}) →
  a = -10 := by
  sorry


end circle_symmetry_theorem_l430_43071


namespace negation_of_universal_positive_square_not_equal_l430_43029

theorem negation_of_universal_positive_square_not_equal (P : ℝ → Prop) :
  (¬ ∀ x : ℝ, x > 0 → x^2 ≠ x) ↔ (∃ x : ℝ, x > 0 ∧ x^2 = x) :=
sorry

end negation_of_universal_positive_square_not_equal_l430_43029


namespace infinitely_many_common_terms_l430_43005

-- Define the arithmetic sequence
def a (n : ℕ) : ℤ := 3*n - 1

-- Define the geometric sequence
def b (n : ℕ) : ℕ := 2^n

-- State the properties of the sequences
axiom a2_eq_5 : a 2 = 5
axiom a8_eq_23 : a 8 = 23
axiom b1_eq_2 : b 1 = 2
axiom b_mul (s t : ℕ) : b (s + t) = b s * b t

-- Theorem statement
theorem infinitely_many_common_terms :
  ∀ m : ℕ, ∃ k : ℕ, k > m ∧ ∃ n : ℕ, b k = a n :=
sorry

end infinitely_many_common_terms_l430_43005


namespace remainder_b91_mod_50_l430_43070

theorem remainder_b91_mod_50 : ∃ k : ℤ, 7^91 + 9^91 = 50 * k + 16 := by sorry

end remainder_b91_mod_50_l430_43070


namespace xy_gt_one_necessary_not_sufficient_l430_43073

theorem xy_gt_one_necessary_not_sufficient :
  (∀ x y : ℝ, x > 1 ∧ y > 1 → x * y > 1) ∧
  (∃ x y : ℝ, x * y > 1 ∧ ¬(x > 1 ∧ y > 1)) :=
by sorry

end xy_gt_one_necessary_not_sufficient_l430_43073


namespace problem_solution_l430_43004

def problem (a b : ℝ × ℝ) : Prop :=
  let angle := 2 * Real.pi / 3
  let magnitude_b := Real.sqrt ((b.1 ^ 2) + (b.2 ^ 2))
  (a = (2, 0)) ∧ 
  (magnitude_b = 1) ∧
  (a.1 * b.1 + a.2 * b.2 = Real.cos angle * magnitude_b * 2) →
  Real.sqrt (((a.1 + 2 * b.1) ^ 2) + ((a.2 + 2 * b.2) ^ 2)) = 2

theorem problem_solution : ∃ (a b : ℝ × ℝ), problem a b := by sorry

end problem_solution_l430_43004


namespace modified_car_distance_increase_l430_43062

/-- Proves the increase in travel distance after modifying a car's fuel efficiency -/
theorem modified_car_distance_increase
  (original_efficiency : ℝ)
  (tank_capacity : ℝ)
  (fuel_reduction_factor : ℝ)
  (h1 : original_efficiency = 32)
  (h2 : tank_capacity = 12)
  (h3 : fuel_reduction_factor = 0.8)
  : (tank_capacity * (original_efficiency / fuel_reduction_factor) - tank_capacity * original_efficiency) = 76.8 := by
  sorry

end modified_car_distance_increase_l430_43062


namespace optimal_bus_rental_l430_43036

/-- Represents the optimal bus rental problem --/
theorem optimal_bus_rental
  (total_passengers : ℕ)
  (capacity_A capacity_B : ℕ)
  (cost_A cost_B : ℕ)
  (max_total_buses : ℕ)
  (max_B_minus_A : ℕ)
  (h_total_passengers : total_passengers = 900)
  (h_capacity_A : capacity_A = 36)
  (h_capacity_B : capacity_B = 60)
  (h_cost_A : cost_A = 1600)
  (h_cost_B : cost_B = 2400)
  (h_max_total_buses : max_total_buses = 21)
  (h_max_B_minus_A : max_B_minus_A = 7) :
  ∃ (x y : ℕ),
    x = 5 ∧ y = 12 ∧
    capacity_A * x + capacity_B * y ≥ total_passengers ∧
    x + y ≤ max_total_buses ∧
    y ≤ x + max_B_minus_A ∧
    ∀ (a b : ℕ),
      capacity_A * a + capacity_B * b ≥ total_passengers →
      a + b ≤ max_total_buses →
      b ≤ a + max_B_minus_A →
      cost_A * x + cost_B * y ≤ cost_A * a + cost_B * b :=
by sorry

end optimal_bus_rental_l430_43036


namespace divisors_of_72_l430_43006

def divisors (n : ℕ) : Set ℕ := {d | d ∣ n ∧ d > 0}

theorem divisors_of_72 : 
  divisors 72 = {1, 2, 3, 4, 6, 8, 9, 12, 18, 24, 36, 72} := by sorry

end divisors_of_72_l430_43006


namespace same_solution_implies_m_value_l430_43003

theorem same_solution_implies_m_value : 
  ∀ (x m : ℝ), 
  (2 * x - m = 1 ∧ 3 * x = 2 * (x - 1)) → 
  m = -5 := by
sorry

end same_solution_implies_m_value_l430_43003


namespace tangent_line_value_l430_43058

/-- Proves that if a line is tangent to both y = ln x and x² = ay at the same point, then a = 2e -/
theorem tangent_line_value (a : ℝ) (h : a > 0) : 
  (∃ (x y : ℝ), x > 0 ∧ 
    y = Real.log x ∧ 
    x^2 = a * y ∧ 
    (1 / x) = (2 / a) * x) → 
  a = 2 * Real.exp 1 := by
sorry

end tangent_line_value_l430_43058


namespace fencing_cost_per_meter_l430_43089

/-- Proves that for a rectangular plot with given dimensions and total fencing cost,
    the cost per meter of fencing is as calculated. -/
theorem fencing_cost_per_meter
  (length : ℝ) (breadth : ℝ) (total_cost : ℝ)
  (h1 : length = 60)
  (h2 : breadth = 40)
  (h3 : total_cost = 5300) :
  total_cost / (2 * (length + breadth)) = 26.5 := by
  sorry

end fencing_cost_per_meter_l430_43089


namespace alice_bracelet_profit_l430_43078

/-- Calculates Alice's profit from selling friendship bracelets -/
theorem alice_bracelet_profit :
  let total_design_a : ℕ := 30
  let total_design_b : ℕ := 22
  let cost_design_a : ℚ := 2
  let cost_design_b : ℚ := 4.5
  let given_away_design_a : ℕ := 5
  let given_away_design_b : ℕ := 3
  let bulk_price_design_a : ℚ := 0.2
  let bulk_price_design_b : ℚ := 0.4
  let total_cost := total_design_a * cost_design_a + total_design_b * cost_design_b
  let remaining_design_a := total_design_a - given_away_design_a
  let remaining_design_b := total_design_b - given_away_design_b
  let total_revenue := remaining_design_a * bulk_price_design_a + remaining_design_b * bulk_price_design_b
  let profit := total_revenue - total_cost
  profit = -146.4 := by sorry

end alice_bracelet_profit_l430_43078


namespace points_difference_l430_43084

def basketball_game (jon_points jack_points tom_points : ℕ) : Prop :=
  (jack_points = jon_points + 5) ∧
  (jon_points + jack_points + tom_points = 18) ∧
  (tom_points < jon_points + jack_points)

theorem points_difference (jon_points jack_points tom_points : ℕ) :
  basketball_game jon_points jack_points tom_points →
  jon_points = 3 →
  (jon_points + jack_points) - tom_points = 4 := by
  sorry

end points_difference_l430_43084


namespace line_parameterization_specific_line_parameterization_l430_43041

/-- A parameterization of a line is valid if it satisfies the line equation and has a correct direction vector -/
def IsValidParameterization (a b : ℝ) (p : ℝ × ℝ) (v : ℝ × ℝ) : Prop :=
  (∀ t, (p.1 + t * v.1, p.2 + t * v.2) ∈ {(x, y) | y = a * x + b}) ∧
  ∃ k ≠ 0, v = (k, a * k)

theorem line_parameterization (a b : ℝ) :
  let line := fun x y ↦ y = a * x + b
  IsValidParameterization a b (1, 8) (1, 3) ∧
  IsValidParameterization a b (2, 11) (-1/3, -1) ∧
  IsValidParameterization a b (-1.5, 0.5) (1, 3) ∧
  ¬IsValidParameterization a b (-5/3, 0) (3, 9) ∧
  ¬IsValidParameterization a b (0, 5) (6, 2) :=
by sorry

/-- The specific line y = 3x + 5 has valid parameterizations A, D, and E, but not B and C -/
theorem specific_line_parameterization :
  let line := fun x y ↦ y = 3 * x + 5
  IsValidParameterization 3 5 (1, 8) (1, 3) ∧
  IsValidParameterization 3 5 (2, 11) (-1/3, -1) ∧
  IsValidParameterization 3 5 (-1.5, 0.5) (1, 3) ∧
  ¬IsValidParameterization 3 5 (-5/3, 0) (3, 9) ∧
  ¬IsValidParameterization 3 5 (0, 5) (6, 2) :=
by sorry

end line_parameterization_specific_line_parameterization_l430_43041


namespace consecutive_product_plus_one_is_square_l430_43040

theorem consecutive_product_plus_one_is_square : ∃ m : ℕ, 
  2017 * 2018 * 2019 * 2020 + 1 = m^2 := by
  sorry

end consecutive_product_plus_one_is_square_l430_43040


namespace cistern_wet_surface_area_l430_43044

/-- Calculates the total wet surface area of a rectangular cistern -/
def total_wet_surface_area (length width depth : Real) : Real :=
  length * width + 2 * length * depth + 2 * width * depth

/-- Theorem stating the total wet surface area of a specific cistern -/
theorem cistern_wet_surface_area :
  total_wet_surface_area 8 6 1.85 = 99.8 := by
  sorry

end cistern_wet_surface_area_l430_43044


namespace equation_solution_l430_43032

theorem equation_solution :
  let f : ℝ → ℝ := λ x => -x^2 * (x + 5) - (5*x - 2)
  ∀ x : ℝ, f x = 0 ↔ x = -2 ∨ x = (-3 + Real.sqrt 5) / 2 ∨ x = (-3 - Real.sqrt 5) / 2 := by
  sorry

end equation_solution_l430_43032


namespace min_sum_squares_l430_43087

/-- Given five real numbers satisfying certain conditions, 
    the sum of their squares has a minimum value. -/
theorem min_sum_squares (a₁ a₂ a₃ a₄ a₅ : ℝ) 
    (h1 : a₁*a₂ + a₂*a₃ + a₃*a₄ + a₄*a₅ + a₅*a₁ = 20)
    (h2 : a₁*a₃ + a₂*a₄ + a₃*a₅ + a₄*a₁ + a₅*a₂ = 22) :
    ∃ (m : ℝ), m = a₁^2 + a₂^2 + a₃^2 + a₄^2 + a₅^2 ∧ 
    ∀ (b₁ b₂ b₃ b₄ b₅ : ℝ), 
    (b₁*b₂ + b₂*b₃ + b₃*b₄ + b₄*b₅ + b₅*b₁ = 20) →
    (b₁*b₃ + b₂*b₄ + b₃*b₅ + b₄*b₁ + b₅*b₂ = 22) →
    m ≤ b₁^2 + b₂^2 + b₃^2 + b₄^2 + b₅^2 ∧
    m = 21 + Real.sqrt 5 := by
  sorry

end min_sum_squares_l430_43087


namespace modular_inverse_32_mod_37_l430_43082

theorem modular_inverse_32_mod_37 :
  ∃ x : ℕ, x ≤ 36 ∧ (32 * x) % 37 = 1 :=
by
  use 15
  sorry

end modular_inverse_32_mod_37_l430_43082


namespace sandy_puppies_l430_43060

def total_puppies (initial : ℝ) (additional : ℝ) : ℝ :=
  initial + additional

theorem sandy_puppies : total_puppies 8 4 = 12 := by
  sorry

end sandy_puppies_l430_43060


namespace f_of_x_minus_3_l430_43011

theorem f_of_x_minus_3 (x : ℝ) : (fun (x : ℝ) => x^2) (x - 3) = x^2 - 6*x + 9 := by
  sorry

end f_of_x_minus_3_l430_43011


namespace distributive_property_negative_l430_43017

theorem distributive_property_negative (a b : ℝ) : -3 * (a - b) = -3 * a + 3 * b := by
  sorry

end distributive_property_negative_l430_43017


namespace complex_fraction_simplification_l430_43061

theorem complex_fraction_simplification :
  (1 + 2*Complex.I) / (3 - 4*Complex.I) = -1/5 + (2/5)*Complex.I :=
by sorry

end complex_fraction_simplification_l430_43061


namespace solve_equation_l430_43049

theorem solve_equation (X : ℝ) : (X^3).sqrt = 81 * (81^(1/9)) → X = 3^(80/27) := by
  sorry

end solve_equation_l430_43049


namespace marble_weight_problem_l430_43016

theorem marble_weight_problem (weight_piece1 weight_piece2 total_weight : ℝ) 
  (h1 : weight_piece1 = 0.33)
  (h2 : weight_piece2 = 0.33)
  (h3 : total_weight = 0.75) :
  total_weight - (weight_piece1 + weight_piece2) = 0.09 := by
  sorry

end marble_weight_problem_l430_43016


namespace min_voters_for_tall_to_win_l430_43054

/-- Represents the voting structure and rules of the giraffe beauty contest --/
structure GiraffeContest where
  total_voters : Nat
  num_districts : Nat
  sections_per_district : Nat
  voters_per_section : Nat
  (total_voters_eq : total_voters = num_districts * sections_per_district * voters_per_section)
  (num_districts_pos : num_districts > 0)
  (sections_per_district_pos : sections_per_district > 0)
  (voters_per_section_pos : voters_per_section > 0)

/-- Calculates the minimum number of voters required to win the contest --/
def minVotersToWin (contest : GiraffeContest) : Nat :=
  let districts_to_win := contest.num_districts / 2 + 1
  let sections_to_win := contest.sections_per_district / 2 + 1
  let voters_to_win_section := contest.voters_per_section / 2 + 1
  districts_to_win * sections_to_win * voters_to_win_section

/-- The main theorem stating the minimum number of voters required for Tall to win --/
theorem min_voters_for_tall_to_win (contest : GiraffeContest)
  (h_total : contest.total_voters = 105)
  (h_districts : contest.num_districts = 5)
  (h_sections : contest.sections_per_district = 7)
  (h_voters : contest.voters_per_section = 3) :
  minVotersToWin contest = 24 := by
  sorry


end min_voters_for_tall_to_win_l430_43054


namespace one_third_blue_faces_iff_three_l430_43092

/-- Represents a cube with side length n -/
structure Cube (n : ℕ) where
  side_length : n > 0

/-- The number of blue faces after cutting the cube into unit cubes -/
def blue_faces (c : Cube n) : ℕ :=
  6 * n^2

/-- The total number of faces of all unit cubes -/
def total_faces (c : Cube n) : ℕ :=
  6 * n^3

/-- Theorem stating that exactly one-third of the faces are blue iff n = 3 -/
theorem one_third_blue_faces_iff_three {n : ℕ} (c : Cube n) :
  3 * blue_faces c = total_faces c ↔ n = 3 :=
sorry

end one_third_blue_faces_iff_three_l430_43092


namespace hexagon_walk_distance_hexagon_walk_distance_proof_l430_43038

/-- The distance of a point from its starting position after moving 7 km along the perimeter of a regular hexagon with side length 3 km -/
theorem hexagon_walk_distance : ℝ :=
  let side_length : ℝ := 3
  let walk_distance : ℝ := 7
  let hexagon_angle : ℝ := 2 * Real.pi / 6
  let end_position : ℝ × ℝ := (1, Real.sqrt 3)
  2

theorem hexagon_walk_distance_proof :
  hexagon_walk_distance = 2 := by sorry

end hexagon_walk_distance_hexagon_walk_distance_proof_l430_43038


namespace number_square_puzzle_l430_43021

theorem number_square_puzzle : ∃ x : ℝ, x^2 + 95 = (x - 20)^2 ∧ x = 7.625 := by
  sorry

end number_square_puzzle_l430_43021


namespace nth_number_in_set_l430_43095

theorem nth_number_in_set (n : ℕ) : 
  (n + 1) * 19 + 13 = (499 * 19 + 13) → n = 498 := by
  sorry

#check nth_number_in_set

end nth_number_in_set_l430_43095


namespace abs_inequality_iff_l430_43028

theorem abs_inequality_iff (a b : ℝ) : a > b ↔ a * |a| > b * |b| := by sorry

end abs_inequality_iff_l430_43028


namespace f_has_no_boundary_point_l430_43000

-- Define the concept of a boundary point
def has_boundary_point (f : ℝ → ℝ) : Prop :=
  ∃ x₀ : ℝ, x₀ ≠ 0 ∧
    (∃ x₁ x₂ : ℝ, x₁ < x₀ ∧ x₀ < x₂ ∧ f x₁ = 0 ∧ f x₂ = 0)

-- Define the function f(x) = x³ + x
def f (x : ℝ) : ℝ := x^3 + x

-- Theorem stating that f does not have a boundary point
theorem f_has_no_boundary_point : ¬ has_boundary_point f := by
  sorry


end f_has_no_boundary_point_l430_43000


namespace john_weight_loss_l430_43079

/-- The number of calories John burns per day -/
def calories_burned_per_day : ℕ := 2300

/-- The number of calories needed to lose 1 pound -/
def calories_per_pound : ℕ := 4000

/-- The number of days it takes John to lose 10 pounds -/
def days_to_lose_10_pounds : ℕ := 80

/-- The number of pounds John wants to lose -/
def pounds_to_lose : ℕ := 10

/-- The number of calories John eats per day -/
def calories_eaten_per_day : ℕ := 1800

theorem john_weight_loss :
  calories_eaten_per_day =
    calories_burned_per_day -
    (pounds_to_lose * calories_per_pound) / days_to_lose_10_pounds :=
by
  sorry

end john_weight_loss_l430_43079


namespace smallest_constant_inequality_l430_43091

theorem smallest_constant_inequality (x y : ℝ) :
  ∃ (C : ℝ), C = 4/3 ∧ (∀ (x y : ℝ), 1 + (x + y)^2 ≤ C * (1 + x^2) * (1 + y^2)) ∧
  (∀ (D : ℝ), (∀ (x y : ℝ), 1 + (x + y)^2 ≤ D * (1 + x^2) * (1 + y^2)) → C ≤ D) :=
sorry

end smallest_constant_inequality_l430_43091


namespace compare_x_y_z_l430_43093

open Real

theorem compare_x_y_z (x y z : ℝ) (hx : x = log π) (hy : y = log 2 / log 5) (hz : z = exp (-1/2)) :
  y < z ∧ z < x := by sorry

end compare_x_y_z_l430_43093


namespace smallest_nonprime_with_large_factors_l430_43013

def is_prime (p : ℕ) : Prop := p > 1 ∧ ∀ m : ℕ, m > 1 → m < p → ¬(p % m = 0)

def has_no_prime_factors_less_than (n k : ℕ) : Prop :=
  ∀ p : ℕ, is_prime p → p < k → ¬(n % p = 0)

def is_nonprime (n : ℕ) : Prop := n > 1 ∧ ¬(is_prime n)

theorem smallest_nonprime_with_large_factors :
  ∃ n : ℕ, is_nonprime n ∧
            has_no_prime_factors_less_than n 20 ∧
            (∀ m : ℕ, m < n → ¬(is_nonprime m ∧ has_no_prime_factors_less_than m 20)) ∧
            500 < n ∧ n ≤ 550 :=
sorry

end smallest_nonprime_with_large_factors_l430_43013


namespace f_at_negative_two_equals_six_l430_43031

-- Define the functions f and g
def f (a b c x : ℝ) : ℝ := a * x^2 + 2 * b * x + c
def g (a b c x : ℝ) : ℝ := (a + 1) * x^2 + 2 * (b + 2) * x + (c + 4)

-- Define the discriminant function
def discriminant (a b c : ℝ) : ℝ := 4 * b^2 - 4 * a * c

-- State the theorem
theorem f_at_negative_two_equals_six (a b c : ℝ) :
  discriminant a b c - discriminant (a + 1) (b + 2) (c + 4) = 24 →
  f a b c (-2) = 6 := by
  sorry


end f_at_negative_two_equals_six_l430_43031


namespace accidental_addition_correction_l430_43075

theorem accidental_addition_correction (x : ℤ) (h : x + 5 = 43) : 5 * x = 190 := by
  sorry

end accidental_addition_correction_l430_43075


namespace jack_and_jill_meeting_point_l430_43056

/-- Represents the problem of Jack and Jill running up and down a hill -/
structure HillRun where
  total_distance : ℝ
  uphill_distance : ℝ
  jack_head_start : ℝ
  jack_uphill_speed : ℝ
  jack_downhill_speed : ℝ
  jill_uphill_speed : ℝ
  jill_downhill_speed : ℝ

/-- Calculates the meeting point of Jack and Jill -/
def meeting_point (run : HillRun) : ℝ :=
  sorry

/-- The main theorem stating the distance from the top where Jack and Jill meet -/
theorem jack_and_jill_meeting_point (run : HillRun) 
  (h1 : run.total_distance = 12)
  (h2 : run.uphill_distance = 6)
  (h3 : run.jack_head_start = 1/6)  -- 10 minutes in hours
  (h4 : run.jack_uphill_speed = 15)
  (h5 : run.jack_downhill_speed = 20)
  (h6 : run.jill_uphill_speed = 18)
  (h7 : run.jill_downhill_speed = 24) :
  run.uphill_distance - meeting_point run = 33/19 :=
sorry

end jack_and_jill_meeting_point_l430_43056


namespace subway_fare_cost_l430_43019

/-- The cost of the subway fare each way given Brian's spending and constraints -/
theorem subway_fare_cost (apple_cost : ℚ) (kiwi_cost : ℚ) (banana_cost : ℚ) 
  (initial_money : ℚ) (max_apples : ℕ) :
  apple_cost = 14 / 12 →
  kiwi_cost = 10 →
  banana_cost = 5 →
  initial_money = 50 →
  max_apples = 24 →
  (initial_money - kiwi_cost - banana_cost - (↑max_apples * apple_cost)) / 2 = 7/2 := by
  sorry

end subway_fare_cost_l430_43019


namespace sculpture_third_week_cut_percentage_l430_43037

/-- Calculates the percentage of marble cut away in the third week of sculpting. -/
theorem sculpture_third_week_cut_percentage
  (initial_weight : ℝ)
  (first_week_cut : ℝ)
  (second_week_cut : ℝ)
  (final_weight : ℝ)
  (h1 : initial_weight = 190)
  (h2 : first_week_cut = 0.25)
  (h3 : second_week_cut = 0.15)
  (h4 : final_weight = 109.0125) :
  let weight_after_first_week := initial_weight * (1 - first_week_cut)
  let weight_after_second_week := weight_after_first_week * (1 - second_week_cut)
  let third_week_cut_percentage := 1 - (final_weight / weight_after_second_week)
  ∃ ε > 0, |third_week_cut_percentage - 0.0999| < ε :=
by sorry

end sculpture_third_week_cut_percentage_l430_43037


namespace power_of_power_l430_43067

theorem power_of_power (a : ℝ) : (a^4)^2 = a^8 := by
  sorry

end power_of_power_l430_43067


namespace rectangles_4x2_grid_l430_43020

/-- The number of rectangles that can be formed on a grid of dots -/
def num_rectangles (cols : ℕ) (rows : ℕ) : ℕ :=
  (cols.choose 2) * (rows.choose 2)

/-- Theorem: The number of rectangles on a 4x2 grid is 6 -/
theorem rectangles_4x2_grid : num_rectangles 4 2 = 6 := by
  sorry

end rectangles_4x2_grid_l430_43020


namespace triangular_prism_is_pentahedron_l430_43064

-- Define the polyhedra types
inductive Polyhedron
| TriangularPyramid
| TriangularPrism
| QuadrangularPrism
| PentagonalPyramid

-- Define the function that returns the number of faces for each polyhedron
def numFaces (p : Polyhedron) : Nat :=
  match p with
  | Polyhedron.TriangularPyramid => 4    -- tetrahedron
  | Polyhedron.TriangularPrism => 5      -- pentahedron
  | Polyhedron.QuadrangularPrism => 6    -- hexahedron
  | Polyhedron.PentagonalPyramid => 6    -- hexahedron

-- Theorem: A triangular prism is a pentahedron
theorem triangular_prism_is_pentahedron :
  numFaces Polyhedron.TriangularPrism = 5 := by sorry

end triangular_prism_is_pentahedron_l430_43064


namespace painted_cube_probability_l430_43008

/-- Represents a cube with painted faces -/
structure PaintedCube where
  size : ℕ
  paintedFaces : ℕ

/-- The number of unit cubes in a larger cube -/
def totalUnitCubes (size : ℕ) : ℕ := size ^ 3

/-- The number of ways to choose 2 cubes from a set -/
def waysToChooseTwoCubes (n : ℕ) : ℕ := n * (n - 1) / 2

/-- The number of unit cubes with exactly three painted faces -/
def cubesWithThreePaintedFaces (size : ℕ) : ℕ := 8

/-- The number of unit cubes with no painted faces -/
def cubesWithNoPaintedFaces (size : ℕ) : ℕ := (size - 2) ^ 3

/-- The probability of selecting one cube with three painted faces and one with no painted faces -/
def probabilityOfSelection (size : ℕ) : ℚ :=
  let total := totalUnitCubes size
  let ways := waysToChooseTwoCubes total
  let threePainted := cubesWithThreePaintedFaces size
  let noPainted := cubesWithNoPaintedFaces size
  (threePainted * noPainted : ℚ) / ways

theorem painted_cube_probability :
  probabilityOfSelection 5 = 72 / 2583 := by
  sorry

end painted_cube_probability_l430_43008


namespace sum_of_roots_equals_one_l430_43065

theorem sum_of_roots_equals_one :
  let f : ℝ → ℝ := λ x => (x + 3) * (x - 4) - 21
  ∃ x₁ x₂ : ℝ, (f x₁ = 0 ∧ f x₂ = 0) ∧ x₁ + x₂ = 1 :=
by
  sorry

end sum_of_roots_equals_one_l430_43065


namespace min_value_complex_condition_l430_43027

theorem min_value_complex_condition (x y : ℝ) :
  Complex.abs (Complex.mk x y - Complex.I * 4) = Complex.abs (Complex.mk x y + 2) →
  2^x + 4^y ≥ 4 * Real.sqrt 2 ∧ 
  ∃ (x₀ y₀ : ℝ), Complex.abs (Complex.mk x₀ y₀ - Complex.I * 4) = Complex.abs (Complex.mk x₀ y₀ + 2) ∧
                 2^x₀ + 4^y₀ = 4 * Real.sqrt 2 :=
by sorry

end min_value_complex_condition_l430_43027


namespace ratio_of_linear_system_l430_43035

theorem ratio_of_linear_system (x y c d : ℝ) 
  (eq1 : 9 * x - 6 * y = c)
  (eq2 : 15 * x - 10 * y = d)
  (h1 : d ≠ 0)
  (h2 : x ≠ 0)
  (h3 : y ≠ 0) :
  c / d = -2 / 5 := by
  sorry

end ratio_of_linear_system_l430_43035
