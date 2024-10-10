import Mathlib

namespace age_difference_l3978_397853

theorem age_difference (son_age father_age : ℕ) 
  (h1 : son_age = 9)
  (h2 : father_age = 36) :
  father_age - son_age = 27 := by
  sorry

end age_difference_l3978_397853


namespace square_minus_circle_area_l3978_397892

theorem square_minus_circle_area (r : ℝ) (s : ℝ) : 
  r = 2 → s = 2 * Real.sqrt 2 → 
  s^2 - π * r^2 = 8 - 4 * π := by
sorry

end square_minus_circle_area_l3978_397892


namespace final_position_of_E_l3978_397830

-- Define the position of E as a pair of axes (base_axis, top_axis)
inductive Axis
  | PositiveX
  | NegativeX
  | PositiveY
  | NegativeY

def Position := Axis × Axis

-- Define the transformations
def rotateClockwise270 (p : Position) : Position :=
  match p with
  | (Axis.NegativeX, Axis.PositiveY) => (Axis.PositiveY, Axis.NegativeX)
  | _ => p  -- For completeness, though we only care about the initial position

def reflectXAxis (p : Position) : Position :=
  match p with
  | (base, top) => (
      match base with
      | Axis.PositiveY => Axis.NegativeY
      | Axis.NegativeY => Axis.PositiveY
      | _ => base,
      top
    )

def reflectYAxis (p : Position) : Position :=
  match p with
  | (base, top) => (
      base,
      match top with
      | Axis.PositiveX => Axis.NegativeX
      | Axis.NegativeX => Axis.PositiveX
      | _ => top
    )

def halfTurn (p : Position) : Position :=
  match p with
  | (base, top) => (
      match base with
      | Axis.PositiveY => Axis.NegativeY
      | Axis.NegativeY => Axis.PositiveY
      | Axis.PositiveX => Axis.NegativeX
      | Axis.NegativeX => Axis.PositiveX,
      match top with
      | Axis.PositiveY => Axis.NegativeY
      | Axis.NegativeY => Axis.PositiveY
      | Axis.PositiveX => Axis.NegativeX
      | Axis.NegativeX => Axis.PositiveX
    )

-- Theorem statement
theorem final_position_of_E :
  let initial_position : Position := (Axis.NegativeX, Axis.PositiveY)
  let final_position := halfTurn (reflectYAxis (reflectXAxis (rotateClockwise270 initial_position)))
  final_position = (Axis.NegativeY, Axis.NegativeX) :=
by
  sorry

end final_position_of_E_l3978_397830


namespace real_part_of_complex_power_l3978_397875

theorem real_part_of_complex_power : Complex.re ((1 - 2*Complex.I)^5) = 41 := by
  sorry

end real_part_of_complex_power_l3978_397875


namespace intersection_of_A_and_B_l3978_397890

def A : Set ℝ := {x | x^2 - x = 0}
def B : Set ℝ := {x | x^2 + x = 0}

theorem intersection_of_A_and_B : A ∩ B = {0} := by sorry

end intersection_of_A_and_B_l3978_397890


namespace hotel_guests_count_l3978_397827

/-- The number of guests attending at least one reunion -/
def total_guests (oates_guests hall_guests both_guests : ℕ) : ℕ :=
  oates_guests + hall_guests - both_guests

/-- Theorem stating the total number of guests attending at least one reunion -/
theorem hotel_guests_count :
  total_guests 70 52 28 = 94 := by
  sorry

end hotel_guests_count_l3978_397827


namespace remaining_area_formula_l3978_397814

/-- The area of the remaining part when a small square is removed from a larger square -/
def remaining_area (x : ℝ) : ℝ := 9 - x^2

/-- Theorem stating the area of the remaining part when a small square is removed from a larger square -/
theorem remaining_area_formula (x : ℝ) (h : 0 < x ∧ x < 3) : 
  remaining_area x = 9 - x^2 := by
  sorry

end remaining_area_formula_l3978_397814


namespace exists_unique_equal_power_point_equal_power_point_is_orthogonal_circle_center_l3978_397847

/-- Represents a circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The power of a point with respect to a circle -/
def powerOfPoint (p : ℝ × ℝ) (c : Circle) : ℝ :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 - c.radius^2

/-- Theorem: Given three circles, there exists a unique point with equal power to all three circles -/
theorem exists_unique_equal_power_point (c1 c2 c3 : Circle) :
  ∃! p : ℝ × ℝ, powerOfPoint p c1 = powerOfPoint p c2 ∧ powerOfPoint p c2 = powerOfPoint p c3 :=
sorry

/-- Theorem: The point with equal power to three circles is the center of a circle 
    that intersects the three given circles at right angles -/
theorem equal_power_point_is_orthogonal_circle_center 
  (c1 c2 c3 : Circle) (p : ℝ × ℝ) 
  (h : powerOfPoint p c1 = powerOfPoint p c2 ∧ powerOfPoint p c2 = powerOfPoint p c3) :
  ∃ r : ℝ, ∀ i : Fin 3, 
    let c := Circle.mk p r
    let ci := [c1, c2, c3].get i
    ∃ x : ℝ × ℝ, (x.1 - c.center.1)^2 + (x.2 - c.center.2)^2 = c.radius^2 ∧
                 (x.1 - ci.center.1)^2 + (x.2 - ci.center.2)^2 = ci.radius^2 ∧
                 ((x.1 - c.center.1) * (x.1 - ci.center.1) + (x.2 - c.center.2) * (x.2 - ci.center.2) = 0) :=
sorry

end exists_unique_equal_power_point_equal_power_point_is_orthogonal_circle_center_l3978_397847


namespace number_accurate_to_hundreds_l3978_397870

def number : ℝ := 1.45 * 10^4

def accurate_to_hundreds_place (x : ℝ) : Prop :=
  ∃ n : ℤ, x = (n * 100 : ℝ) ∧ n ≥ 0 ∧ n < 1000

theorem number_accurate_to_hundreds : accurate_to_hundreds_place number := by
  sorry

end number_accurate_to_hundreds_l3978_397870


namespace smaller_sphere_radius_l3978_397826

/-- The radius of a smaller sphere when a sphere of radius R is cast into two smaller spheres -/
theorem smaller_sphere_radius (R : ℝ) (R_pos : R > 0) : ℝ :=
  let smaller_radius := R / 3
  let larger_radius := 2 * smaller_radius
  have volume_conservation : (4 / 3) * Real.pi * R^3 = (4 / 3) * Real.pi * smaller_radius^3 + (4 / 3) * Real.pi * larger_radius^3 := by sorry
  have radius_ratio : larger_radius = 2 * smaller_radius := by sorry
  smaller_radius

#check smaller_sphere_radius

end smaller_sphere_radius_l3978_397826


namespace min_diff_composites_sum_96_l3978_397842

def is_composite (n : ℕ) : Prop := ∃ a b, 1 < a ∧ 1 < b ∧ n = a * b

theorem min_diff_composites_sum_96 : 
  ∃ (a b : ℕ), is_composite a ∧ is_composite b ∧ a + b = 96 ∧
  ∀ (c d : ℕ), is_composite c → is_composite d → c + d = 96 → 
  (c : ℤ) - (d : ℤ) ≥ 2 ∨ (d : ℤ) - (c : ℤ) ≥ 2 :=
sorry

end min_diff_composites_sum_96_l3978_397842


namespace absolute_value_inequality_l3978_397804

theorem absolute_value_inequality (a : ℝ) :
  (∀ x : ℝ, |x - 3| + |x + 2| > a) → a < 5 := by
  sorry

end absolute_value_inequality_l3978_397804


namespace min_voters_for_giraffe_contest_l3978_397829

/-- Represents the voting structure in the giraffe beauty contest -/
structure VotingStructure :=
  (total_voters : ℕ)
  (num_districts : ℕ)
  (num_sections_per_district : ℕ)
  (voters_per_section : ℕ)
  (h_total : total_voters = num_districts * num_sections_per_district * voters_per_section)

/-- Calculates the minimum number of voters required to win -/
def min_voters_to_win (vs : VotingStructure) : ℕ :=
  let districts_to_win := (vs.num_districts + 1) / 2
  let sections_to_win := (vs.num_sections_per_district + 1) / 2
  let voters_to_win_section := (vs.voters_per_section + 1) / 2
  districts_to_win * sections_to_win * voters_to_win_section

/-- Theorem stating the minimum number of voters required to win the contest -/
theorem min_voters_for_giraffe_contest :
  ∀ (vs : VotingStructure),
  vs.total_voters = 105 ∧
  vs.num_districts = 5 ∧
  vs.num_sections_per_district = 7 ∧
  vs.voters_per_section = 3 →
  min_voters_to_win vs = 24 := by
  sorry

#eval min_voters_to_win {
  total_voters := 105,
  num_districts := 5,
  num_sections_per_district := 7,
  voters_per_section := 3,
  h_total := rfl
}

end min_voters_for_giraffe_contest_l3978_397829


namespace triangle_property_l3978_397873

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the properties of the triangle
def is_right_triangle (t : Triangle) : Prop :=
  -- The angle at C is a right angle
  sorry

def altitude_meets_AB (t : Triangle) (D : ℝ × ℝ) : Prop :=
  -- The altitude from C meets AB at D
  sorry

def integer_sides (t : Triangle) : Prop :=
  -- The lengths of the sides of triangle ABC are integers
  sorry

def BD_length (t : Triangle) (D : ℝ × ℝ) : Prop :=
  -- BD = 29³
  sorry

def cos_B_fraction (t : Triangle) (m n : ℕ) : Prop :=
  -- cos B = m/n, where m and n are relatively prime positive integers
  sorry

theorem triangle_property (t : Triangle) (D : ℝ × ℝ) (m n : ℕ) :
  is_right_triangle t →
  altitude_meets_AB t D →
  integer_sides t →
  BD_length t D →
  cos_B_fraction t m n →
  m + n = 450 := by
  sorry

end triangle_property_l3978_397873


namespace arrangement_count_l3978_397833

-- Define the total number of people
def total_people : ℕ := 7

-- Define the number of people to be selected
def selected_people : ℕ := 5

-- Define a function to calculate the number of arrangements
def arrangements (n : ℕ) (k : ℕ) : ℕ := sorry

-- Theorem statement
theorem arrangement_count :
  arrangements total_people selected_people = 600 :=
sorry

end arrangement_count_l3978_397833


namespace minimum_cost_theorem_l3978_397848

/-- Represents the cost and survival properties of flower types -/
structure FlowerType where
  cost : ℝ
  survivalRate : ℝ

/-- Represents the planting scenario -/
structure PlantingScenario where
  typeA : FlowerType
  typeB : FlowerType
  totalPots : ℕ
  maxReplacement : ℕ

def minimumCost (scenario : PlantingScenario) : ℝ :=
  let m := scenario.totalPots / 2
  m * scenario.typeA.cost + (scenario.totalPots - m) * scenario.typeB.cost

theorem minimum_cost_theorem (scenario : PlantingScenario) :
  scenario.typeA.cost = 30 ∧
  scenario.typeB.cost = 60 ∧
  scenario.totalPots = 400 ∧
  scenario.typeA.survivalRate = 0.7 ∧
  scenario.typeB.survivalRate = 0.9 ∧
  scenario.maxReplacement = 80 ∧
  3 * scenario.typeA.cost + 4 * scenario.typeB.cost = 330 ∧
  4 * scenario.typeA.cost + 3 * scenario.typeB.cost = 300 →
  minimumCost scenario = 18000 :=
sorry

#check minimum_cost_theorem

end minimum_cost_theorem_l3978_397848


namespace unique_number_existence_l3978_397858

theorem unique_number_existence : ∃! N : ℕ, N / 1000 = 220 ∧ N % 1000 = 40 := by
  sorry

end unique_number_existence_l3978_397858


namespace number_calculation_l3978_397817

theorem number_calculation (N : ℝ) (h : (1/4) * (1/3) * (2/5) * N = 30) : 
  (40/100) * N = 360 := by
  sorry

end number_calculation_l3978_397817


namespace quadratic_roots_condition_l3978_397897

theorem quadratic_roots_condition (c : ℚ) : 
  (∀ x : ℚ, x^2 - 7*x + c = 0 ↔ ∃ s : ℤ, s^2 = 9*c ∧ x = (7 + s) / 2 ∨ x = (7 - s) / 2) →
  c = 49 / 13 :=
by sorry

end quadratic_roots_condition_l3978_397897


namespace purple_yellow_ratio_l3978_397845

/-- Represents the number of flowers of each color in the garden -/
structure GardenFlowers where
  yellow : ℕ
  purple : ℕ
  green : ℕ

/-- Conditions of the garden -/
def gardenConditions (g : GardenFlowers) : Prop :=
  g.yellow = 10 ∧
  g.green = (g.yellow + g.purple) / 4 ∧
  g.yellow + g.purple + g.green = 35

/-- Theorem stating the relationship between purple and yellow flowers -/
theorem purple_yellow_ratio (g : GardenFlowers) 
  (h : gardenConditions g) : g.purple * 10 = g.yellow * 18 := by
  sorry

#check purple_yellow_ratio

end purple_yellow_ratio_l3978_397845


namespace probability_heart_king_king_ace_l3978_397834

/-- Represents a standard deck of 52 playing cards -/
def StandardDeck : ℕ := 52

/-- Number of hearts excluding King and Ace of hearts -/
def HeartsExcludingKingAce : ℕ := 11

/-- Number of Kings in a standard deck -/
def KingsInDeck : ℕ := 4

/-- Number of Aces in a standard deck -/
def AcesInDeck : ℕ := 4

/-- Probability of drawing the specific sequence (Heart, King, King, Ace) -/
def probabilityHeartKingKingAce : ℚ :=
  (HeartsExcludingKingAce : ℚ) / StandardDeck *
  KingsInDeck / (StandardDeck - 1) *
  (KingsInDeck - 1) / (StandardDeck - 2) *
  AcesInDeck / (StandardDeck - 3)

theorem probability_heart_king_king_ace :
  probabilityHeartKingKingAce = 1 / 12317 := by
  sorry

end probability_heart_king_king_ace_l3978_397834


namespace cos_330_deg_l3978_397810

/-- Cosine of 330 degrees is equal to sqrt(3)/2 -/
theorem cos_330_deg : Real.cos (330 * π / 180) = Real.sqrt 3 / 2 := by
  sorry

end cos_330_deg_l3978_397810


namespace expression_value_l3978_397838

theorem expression_value (x y z : ℝ) (hx : x = 1) (hy : y = 1) (hz : z = 3) :
  x^2 * y * z - x * y * z^2 = -6 := by
  sorry

end expression_value_l3978_397838


namespace dining_group_size_l3978_397885

theorem dining_group_size (total_bill : ℝ) (tip_percentage : ℝ) (individual_payment : ℝ) : 
  total_bill = 139 ∧ tip_percentage = 0.1 ∧ individual_payment = 25.48 →
  Int.floor ((total_bill * (1 + tip_percentage)) / individual_payment) = 6 := by
sorry

end dining_group_size_l3978_397885


namespace absolute_value_inequality_supremum_l3978_397809

theorem absolute_value_inequality_supremum :
  (∀ k : ℝ, (∀ x : ℝ, |x + 3| + |x - 1| > k) → k < 4) ∧
  ∀ ε > 0, ∃ x : ℝ, |x + 3| + |x - 1| < 4 + ε :=
by sorry

end absolute_value_inequality_supremum_l3978_397809


namespace condition_implies_isosceles_l3978_397832

-- Define a structure for a triangle in a plane
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define a function to represent a vector from one point to another
def vector (p q : ℝ × ℝ) : ℝ × ℝ := (q.1 - p.1, q.2 - p.2)

-- Define dot product for 2D vectors
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Define the condition given in the problem
def satisfies_condition (t : Triangle) : Prop :=
  ∀ O : ℝ × ℝ, 
    let OB := vector O t.B
    let OC := vector O t.C
    let OA := vector O t.A
    dot_product (OB - OC) (OB + OC - 2 • OA) = 0

-- Define what it means for a triangle to be isosceles
def is_isosceles (t : Triangle) : Prop :=
  let AB := vector t.A t.B
  let AC := vector t.A t.C
  dot_product AB AB = dot_product AC AC

-- State the theorem
theorem condition_implies_isosceles (t : Triangle) :
  satisfies_condition t → is_isosceles t :=
by
  sorry

end condition_implies_isosceles_l3978_397832


namespace dividend_calculation_l3978_397896

theorem dividend_calculation (dividend divisor quotient : ℕ) 
  (h1 : dividend = 5 * divisor) 
  (h2 : divisor = 4 * quotient) : 
  dividend = 100 := by
  sorry

end dividend_calculation_l3978_397896


namespace mathborough_rainfall_2007_l3978_397883

/-- Calculates the total rainfall in Mathborough for the year 2007 given the rainfall data from 2005 to 2007. -/
theorem mathborough_rainfall_2007 (rainfall_2005 : ℝ) (increase_2006 : ℝ) (increase_2007 : ℝ) :
  rainfall_2005 = 40.5 →
  increase_2006 = 3 →
  increase_2007 = 4 →
  (rainfall_2005 + increase_2006 + increase_2007) * 12 = 570 := by
  sorry

end mathborough_rainfall_2007_l3978_397883


namespace arithmetic_sequence_50th_term_l3978_397868

/-- Given an arithmetic sequence with first term 2 and common difference 5,
    the 50th term of this sequence is 247. -/
theorem arithmetic_sequence_50th_term :
  let a : ℕ → ℕ := λ n => 2 + (n - 1) * 5
  a 50 = 247 := by sorry

end arithmetic_sequence_50th_term_l3978_397868


namespace absolute_difference_of_product_and_sum_l3978_397871

theorem absolute_difference_of_product_and_sum (p q : ℝ) 
  (h1 : p * q = 6) 
  (h2 : p + q = 7) : 
  |p - q| = 5 := by
sorry

end absolute_difference_of_product_and_sum_l3978_397871


namespace decreasing_function_inequality_l3978_397863

theorem decreasing_function_inequality (f : ℝ → ℝ) (a : ℝ) :
  (∀ x y, 0 < x ∧ x < y ∧ y < 4 → f x > f y) →  -- f is decreasing on (0,4)
  (0 < a^2 - a ∧ a^2 - a < 4) →                 -- domain condition
  f (a^2 - a) > f 2 →                           -- given inequality
  (-1 < a ∧ a < 0) ∨ (1 < a ∧ a < 2) :=         -- conclusion
by sorry

end decreasing_function_inequality_l3978_397863


namespace probability_at_least_two_correct_l3978_397802

/-- The probability of getting at least two correct answers out of five questions
    with four choices each, when guessing randomly. -/
theorem probability_at_least_two_correct : ℝ := by
  -- Define the number of questions and choices
  let n : ℕ := 5
  let choices : ℕ := 4

  -- Define the probability of a correct guess
  let p : ℝ := 1 / choices

  -- Define the binomial probability function
  let binomial_prob (k : ℕ) : ℝ := (n.choose k) * p^k * (1 - p)^(n - k)

  -- Calculate the probability of getting 0 or 1 correct
  let prob_zero_or_one : ℝ := binomial_prob 0 + binomial_prob 1

  -- The probability of at least two correct is 1 minus the probability of 0 or 1 correct
  let prob_at_least_two : ℝ := 1 - prob_zero_or_one

  -- Prove that this probability is equal to 47/128
  sorry

#eval (47 : ℚ) / 128

end probability_at_least_two_correct_l3978_397802


namespace new_person_age_l3978_397836

/-- Given a group of 10 persons, prove that if replacing a 40-year-old person
    with a new person decreases the average age by 3 years, then the age of
    the new person is 10 years. -/
theorem new_person_age (T : ℕ) (A : ℕ) : 
  (T / 10 : ℚ) - ((T - 40 + A) / 10 : ℚ) = 3 → A = 10 := by
  sorry

end new_person_age_l3978_397836


namespace angle_measure_proof_l3978_397821

theorem angle_measure_proof (x : ℝ) : 
  (180 - x = 6 * (90 - x)) → x = 72 := by
  sorry

end angle_measure_proof_l3978_397821


namespace floor_factorial_ratio_l3978_397813

open BigOperators

def factorial (n : ℕ) : ℕ := ∏ i in Finset.range n, i + 1

theorem floor_factorial_ratio : 
  ⌊(factorial 2007 + factorial 2004 : ℚ) / (factorial 2006 + factorial 2005)⌋ = 2006 := by
  sorry

end floor_factorial_ratio_l3978_397813


namespace extreme_points_when_a_neg_one_max_value_on_interval_l3978_397884

/-- The function f(x) = x³ + 3ax² -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + 3*a*x^2

/-- Theorem for extreme points and values when a = -1 -/
theorem extreme_points_when_a_neg_one :
  let f_neg_one := f (-1)
  ∃ (local_max local_min : ℝ),
    (local_max = 0 ∧ f_neg_one local_max = 0) ∧
    (local_min = 2 ∧ f_neg_one local_min = -4) ∧
    ∀ x, f_neg_one x ≤ f_neg_one local_max ∨ f_neg_one x ≥ f_neg_one local_min :=
sorry

/-- Theorem for maximum value on [0,2] -/
theorem max_value_on_interval (a : ℝ) :
  let max_value := if a ≥ 0 then f a 2
                   else if a > -1 then max (f a 0) (f a 2)
                   else f a 0
  ∀ x ∈ Set.Icc 0 2, f a x ≤ max_value :=
sorry

end extreme_points_when_a_neg_one_max_value_on_interval_l3978_397884


namespace sqrt_nine_factorial_over_72_l3978_397894

theorem sqrt_nine_factorial_over_72 : Real.sqrt (Nat.factorial 9 / 72) = 12 * Real.sqrt 35 := by
  sorry

end sqrt_nine_factorial_over_72_l3978_397894


namespace otimes_neg_two_neg_one_l3978_397825

-- Define the custom operation
def otimes (a b : ℝ) : ℝ := a^2 - |b|

-- Theorem statement
theorem otimes_neg_two_neg_one : otimes (-2) (-1) = 3 := by
  sorry

end otimes_neg_two_neg_one_l3978_397825


namespace percentage_markup_approx_l3978_397888

def selling_price : ℝ := 8337
def cost_price : ℝ := 6947.5

theorem percentage_markup_approx (ε : ℝ) (h : ε > 0) :
  ∃ (markup_percentage : ℝ),
    abs (markup_percentage - 19.99) < ε ∧
    markup_percentage = (selling_price - cost_price) / cost_price * 100 :=
by
  sorry

end percentage_markup_approx_l3978_397888


namespace regular_polygon_sides_l3978_397839

theorem regular_polygon_sides (central_angle : ℝ) : 
  central_angle = 40 → (360 : ℝ) / central_angle = 9 := by
  sorry

end regular_polygon_sides_l3978_397839


namespace benny_candy_bars_l3978_397879

/-- The number of candy bars Benny bought -/
def candy_bars : ℕ := 5

/-- The cost of each soft drink -/
def soft_drink_cost : ℕ := 4

/-- The number of soft drinks Benny bought -/
def soft_drinks : ℕ := 2

/-- The total amount Benny spent -/
def total_spent : ℕ := 28

/-- The cost of each candy bar -/
def candy_bar_cost : ℕ := 4

theorem benny_candy_bars : 
  candy_bars * candy_bar_cost + soft_drinks * soft_drink_cost = total_spent := by
  sorry

end benny_candy_bars_l3978_397879


namespace laundry_cost_is_11_l3978_397818

/-- The cost of Samantha's laundry given the specified conditions. -/
def laundry_cost : ℚ :=
  let washer_cost : ℚ := 4
  let dryer_cost_per_10_min : ℚ := 1/4
  let wash_loads : ℕ := 2
  let dryer_count : ℕ := 3
  let dryer_time : ℕ := 40

  let wash_cost : ℚ := washer_cost * wash_loads
  let dryer_intervals : ℕ := dryer_time / 10
  let dryer_cost : ℚ := (dryer_cost_per_10_min * dryer_intervals) * dryer_count

  wash_cost + dryer_cost

/-- Theorem stating that the total cost of Samantha's laundry is $11. -/
theorem laundry_cost_is_11 : laundry_cost = 11 := by
  sorry

end laundry_cost_is_11_l3978_397818


namespace g_ln_inverse_2017_l3978_397835

noncomputable section

variable (a : ℝ)
variable (f : ℝ → ℝ)
variable (g : ℝ → ℝ)

axiom a_positive : a > 0
axiom a_not_one : a ≠ 1
axiom f_property : ∀ m n : ℝ, f (m + n) = f m + f n - 1
axiom g_def : ∀ x : ℝ, g x = f x + a^x / (a^x + 1)
axiom g_ln_2017 : g (Real.log 2017) = 2018

theorem g_ln_inverse_2017 : g (Real.log (1 / 2017)) = -2015 := by
  sorry

end g_ln_inverse_2017_l3978_397835


namespace skyscraper_anniversary_l3978_397872

theorem skyscraper_anniversary (current_year : ℕ) : 
  let years_since_built := 100
  let years_to_event := 95
  let event_year := current_year + years_to_event
  let years_at_event := years_since_built + years_to_event
  ∃ (anniversary : ℕ), anniversary > years_at_event ∧ anniversary - years_at_event = 5 :=
by sorry

end skyscraper_anniversary_l3978_397872


namespace angle_measure_proof_l3978_397837

theorem angle_measure_proof (x : ℝ) : 
  (90 - x = 3 * x - 10) → x = 25 := by
sorry

end angle_measure_proof_l3978_397837


namespace fruit_basket_difference_l3978_397855

/-- Proof that the difference between oranges and apples is 2 in a fruit basket -/
theorem fruit_basket_difference : ∀ (apples bananas peaches : ℕ),
  apples + bananas + peaches + 6 = 28 →
  bananas = 3 * apples →
  peaches = bananas / 2 →
  6 - apples = 2 := by
  sorry

end fruit_basket_difference_l3978_397855


namespace sqrt_sum_quotient_l3978_397844

theorem sqrt_sum_quotient : (Real.sqrt 27 + Real.sqrt 243) / Real.sqrt 48 = 3 := by
  sorry

end sqrt_sum_quotient_l3978_397844


namespace volleyball_substitutions_remainder_l3978_397881

/-- Number of players in a volleyball team -/
def total_players : ℕ := 18

/-- Number of starting players -/
def starting_players : ℕ := 6

/-- Number of substitute players -/
def substitute_players : ℕ := total_players - starting_players

/-- Maximum number of substitutions allowed -/
def max_substitutions : ℕ := 5

/-- Calculate the number of ways to make k substitutions -/
def substitution_ways (k : ℕ) : ℕ :=
  if k = 0 then 1
  else starting_players * (substitute_players - k + 1) * substitution_ways (k - 1)

/-- Total number of ways to execute substitutions -/
def total_substitution_ways : ℕ :=
  List.sum (List.map substitution_ways (List.range (max_substitutions + 1)))

/-- The main theorem to prove -/
theorem volleyball_substitutions_remainder :
  total_substitution_ways % 1000 = 271 := by sorry

end volleyball_substitutions_remainder_l3978_397881


namespace natalia_cycling_distance_l3978_397850

/-- Represents the total distance cycled over four days given specific conditions --/
def total_distance (monday tuesday : ℕ) : ℕ :=
  let wednesday := tuesday / 2
  let thursday := monday + wednesday
  monday + tuesday + wednesday + thursday

/-- Theorem stating that given the specific conditions in the problem, 
    the total distance cycled is 180 km --/
theorem natalia_cycling_distance : total_distance 40 50 = 180 := by
  sorry

end natalia_cycling_distance_l3978_397850


namespace pizza_slice_difference_l3978_397862

/-- Given a pizza with 78 slices shared in a ratio of 5:8, prove that the difference
    between the waiter's share and 20 less than the waiter's share is 20 slices. -/
theorem pizza_slice_difference (total_slices : ℕ) (buzz_ratio waiter_ratio : ℕ) : 
  total_slices = 78 → 
  buzz_ratio = 5 → 
  waiter_ratio = 8 → 
  let waiter_share := (waiter_ratio * total_slices) / (buzz_ratio + waiter_ratio)
  waiter_share - (waiter_share - 20) = 20 :=
by sorry

end pizza_slice_difference_l3978_397862


namespace jacob_test_score_l3978_397864

theorem jacob_test_score (x : ℝ) : 
  (x + 79 + 92 + 84 + 85) / 5 = 85 → x = 85 := by
sorry

end jacob_test_score_l3978_397864


namespace average_weight_abc_l3978_397806

/-- Given three weights a, b, and c, prove that their average is 43 kg -/
theorem average_weight_abc (a b c : ℝ) 
  (hab : (a + b) / 2 = 40)  -- average of a and b is 40 kg
  (hbc : (b + c) / 2 = 43)  -- average of b and c is 43 kg
  (hb : b = 37)             -- weight of b is 37 kg
  : (a + b + c) / 3 = 43 := by
  sorry

end average_weight_abc_l3978_397806


namespace hotel_room_charges_l3978_397819

theorem hotel_room_charges (G : ℝ) (h1 : G > 0) : 
  let R := G * (1 + 0.19999999999999986)
  let P := R * (1 - 0.25)
  P = G * (1 - 0.1) :=
by sorry

end hotel_room_charges_l3978_397819


namespace experimental_primary_school_students_l3978_397876

/-- The total number of students in Experimental Primary School -/
def total_students (num_classes : ℕ) (boys_in_class1 : ℕ) (girls_in_class1 : ℕ) : ℕ :=
  num_classes * (boys_in_class1 + girls_in_class1)

/-- Theorem: The total number of students in Experimental Primary School is 896 -/
theorem experimental_primary_school_students :
  total_students 28 21 11 = 896 := by
  sorry

end experimental_primary_school_students_l3978_397876


namespace triangle_angle_determinant_l3978_397824

theorem triangle_angle_determinant (A B C : ℝ) (h : A + B + C = Real.pi) :
  let M : Matrix (Fin 3) (Fin 3) ℝ := λ i j =>
    if i = j then Real.sin (2 * (if i = 0 then A else if i = 1 then B else C))
    else 1
  Matrix.det M = 2 := by
sorry

end triangle_angle_determinant_l3978_397824


namespace unique_prime_solution_l3978_397800

theorem unique_prime_solution :
  ∀ p q r : ℕ,
  Prime p → Prime q → Prime r →
  p + q^2 = r^4 →
  p = 7 ∧ q = 3 ∧ r = 2 :=
by sorry

end unique_prime_solution_l3978_397800


namespace joan_family_distance_l3978_397854

/-- Calculates the distance traveled given the total time, driving speed, and break times. -/
def distance_traveled (total_time : ℝ) (speed : ℝ) (lunch_break : ℝ) (bathroom_break : ℝ) : ℝ :=
  (total_time - (lunch_break + 2 * bathroom_break)) * speed

/-- Theorem: Given Joan's travel conditions, her family lives 480 miles away. -/
theorem joan_family_distance :
  let total_time : ℝ := 9  -- 9 hours total trip time
  let speed : ℝ := 60      -- 60 mph driving speed
  let lunch_break : ℝ := 0.5  -- 30 minutes = 0.5 hours
  let bathroom_break : ℝ := 0.25  -- 15 minutes = 0.25 hours
  distance_traveled total_time speed lunch_break bathroom_break = 480 := by
  sorry

#eval distance_traveled 9 60 0.5 0.25

end joan_family_distance_l3978_397854


namespace complex_equation_implies_xy_equals_one_l3978_397880

theorem complex_equation_implies_xy_equals_one (x y : ℝ) :
  (x + 1 : ℂ) + y * I = -I + 2 * x →
  x ^ y = 1 :=
by
  sorry

end complex_equation_implies_xy_equals_one_l3978_397880


namespace range_of_k_l3978_397886

open Set

def A : Set ℝ := {x | x ≤ 1 ∨ x ≥ 3}

def B (k : ℝ) : Set ℝ := {x | k < x ∧ x < 2*k + 1}

theorem range_of_k : ∀ k : ℝ, (Aᶜ ∩ B k = ∅) ↔ (k ≤ 0 ∨ k ≥ 3) :=
sorry

end range_of_k_l3978_397886


namespace robs_money_total_l3978_397867

/-- Represents the value of coins in dollars -/
def coin_value (coin : String) : ℚ :=
  match coin with
  | "quarter" => 25 / 100
  | "dime" => 10 / 100
  | "nickel" => 5 / 100
  | "penny" => 1 / 100
  | _ => 0

/-- Calculates the total value of a given number of coins -/
def coin_total (coin : String) (count : ℕ) : ℚ :=
  (coin_value coin) * count

/-- Theorem: Rob's total money is $2.42 -/
theorem robs_money_total :
  let quarters := coin_total "quarter" 7
  let dimes := coin_total "dime" 3
  let nickels := coin_total "nickel" 5
  let pennies := coin_total "penny" 12
  quarters + dimes + nickels + pennies = 242 / 100 := by
  sorry

end robs_money_total_l3978_397867


namespace dani_pants_after_five_years_l3978_397874

/-- Calculates the number of pants after a given number of years -/
def pantsAfterYears (initialPants : ℕ) (pairsPerYear : ℕ) (pantsPerPair : ℕ) (years : ℕ) : ℕ :=
  initialPants + years * (pairsPerYear * pantsPerPair)

/-- Theorem: Dani will have 90 pants after 5 years -/
theorem dani_pants_after_five_years :
  pantsAfterYears 50 4 2 5 = 90 := by
  sorry

#eval pantsAfterYears 50 4 2 5

end dani_pants_after_five_years_l3978_397874


namespace incircle_radius_l3978_397807

/-- An isosceles triangle with base 10 and height 12 -/
structure IsoscelesTriangle where
  base : ℝ
  height : ℝ
  is_isosceles : base = 10 ∧ height = 12

/-- The incircle of a triangle -/
def incircle (t : IsoscelesTriangle) : ℝ := sorry

/-- Theorem: The radius of the incircle of the given isosceles triangle is 10/3 -/
theorem incircle_radius (t : IsoscelesTriangle) : incircle t = 10 / 3 := by sorry

end incircle_radius_l3978_397807


namespace circle_line_intersection_min_value_l3978_397852

/-- Given a circle with center (m,n) in the first quadrant and radius 3,
    intersected by a line to form a chord of length 4,
    the minimum value of (m+2n)/(mn) is 8/3 -/
theorem circle_line_intersection_min_value (m n : ℝ) :
  m > 0 →
  n > 0 →
  m + 2*n = 3 →
  (∀ x y : ℝ, (x - m)^2 + (y - n)^2 = 9 → x + 2*y + 2 = 0 → 
    ∃ x' y' : ℝ, x' ≠ x ∧ y' ≠ y ∧ (x' - m)^2 + (y' - n)^2 = 9 ∧ 
    x' + 2*y' + 2 = 0 ∧ (x - x')^2 + (y - y')^2 = 16) →
  (m + 2*n) / (m * n) ≥ 8/3 :=
sorry

end circle_line_intersection_min_value_l3978_397852


namespace units_digit_of_division_l3978_397889

theorem units_digit_of_division : 
  (30 * 31 * 32 * 33 * 34 * 35) / 14000 % 10 = 2 := by sorry

end units_digit_of_division_l3978_397889


namespace binary_1101_is_13_l3978_397856

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (λ acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_1101_is_13 : 
  binary_to_decimal [true, false, true, true] = 13 := by
  sorry

end binary_1101_is_13_l3978_397856


namespace prime_product_divisors_l3978_397843

theorem prime_product_divisors (p q : Nat) (x : Nat) :
  Prime p →
  Prime q →
  (Nat.divisors (p^x * q^5)).card = 30 →
  x = 4 := by
sorry

end prime_product_divisors_l3978_397843


namespace complex_equation_proof_l3978_397878

theorem complex_equation_proof (z : ℂ) (h : z = 1 + I) : z^2 - 2*z + 2 = 0 := by
  sorry

end complex_equation_proof_l3978_397878


namespace certain_number_proof_l3978_397811

theorem certain_number_proof (x y : ℤ) 
  (eq1 : 4 * x + y = 34) 
  (eq2 : y^2 = 4) : 
  2 * x - y = 14 := by
  sorry

end certain_number_proof_l3978_397811


namespace acid_dilution_l3978_397828

/-- Proves that adding 80 ounces of pure water to 50 ounces of a 26% acid solution
    results in a 10% acid solution. -/
theorem acid_dilution (initial_volume : ℝ) (initial_concentration : ℝ) 
    (water_added : ℝ) (final_concentration : ℝ) :
  initial_volume = 50 →
  initial_concentration = 0.26 →
  water_added = 80 →
  final_concentration = 0.10 →
  (initial_volume * initial_concentration) / (initial_volume + water_added) = final_concentration :=
by
  sorry

#check acid_dilution

end acid_dilution_l3978_397828


namespace g_value_at_10_l3978_397816

theorem g_value_at_10 (g : ℕ → ℝ) 
  (h1 : g 1 = 1)
  (h2 : ∀ (m n : ℕ), m ≥ n → g (m + n) + g (m - n) = (g (2*m) + g (2*n))/2 + 2) :
  g 10 = 102 := by
sorry

end g_value_at_10_l3978_397816


namespace petri_dishes_count_l3978_397861

/-- The number of petri dishes in a biology lab -/
def number_of_petri_dishes : ℕ :=
  let total_germs : ℕ := 3600  -- 0.036 * 10^5 = 3600
  let germs_per_dish : ℕ := 80 -- Approximating 79.99999999999999 to 80
  total_germs / germs_per_dish

theorem petri_dishes_count : number_of_petri_dishes = 45 := by
  sorry

end petri_dishes_count_l3978_397861


namespace square_plus_linear_equals_square_l3978_397866

theorem square_plus_linear_equals_square (x y : ℕ+) 
  (h : x^2 + 84*x + 2016 = y^2) : 
  x^3 + y^2 = 12096 := by
  sorry

end square_plus_linear_equals_square_l3978_397866


namespace milk_per_milkshake_l3978_397860

/-- The amount of milk needed for each milkshake, given:
  * Blake has 72 ounces of milk initially
  * Blake has 192 ounces of ice cream
  * Each milkshake needs 12 ounces of ice cream
  * After making milkshakes, Blake has 8 ounces of milk left
-/
theorem milk_per_milkshake (initial_milk : ℕ) (ice_cream : ℕ) (ice_cream_per_shake : ℕ) (milk_left : ℕ)
  (h1 : initial_milk = 72)
  (h2 : ice_cream = 192)
  (h3 : ice_cream_per_shake = 12)
  (h4 : milk_left = 8) :
  (initial_milk - milk_left) / (ice_cream / ice_cream_per_shake) = 4 := by
  sorry

end milk_per_milkshake_l3978_397860


namespace pasta_preference_ratio_l3978_397898

/-- Given a survey of pasta preferences among students, this theorem proves
    that the ratio of students preferring spaghetti to those preferring manicotti is 2. -/
theorem pasta_preference_ratio :
  let total_students : ℕ := 800
  let spaghetti_preference : ℕ := 320
  let manicotti_preference : ℕ := 160
  (spaghetti_preference : ℚ) / manicotti_preference = 2 := by
  sorry

end pasta_preference_ratio_l3978_397898


namespace factor_implies_p_value_l3978_397895

theorem factor_implies_p_value (p : ℚ) : 
  (∀ x : ℚ, (3 * x + 4 = 0) → (4 * x^3 + p * x^2 + 17 * x + 24 = 0)) → 
  p = 13/4 := by
sorry

end factor_implies_p_value_l3978_397895


namespace ellipse_to_circle_l3978_397899

theorem ellipse_to_circle 
  (a b : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hab : a ≠ b) 
  (x y : ℝ) 
  (h_ellipse : x^2 / a^2 + y^2 / b^2 = 1) :
  x^2 + ((a/b) * y)^2 = a^2 := by
sorry

end ellipse_to_circle_l3978_397899


namespace dihedral_angle_sum_l3978_397849

/-- A dihedral angle -/
structure DihedralAngle where
  /-- The linear angle of the dihedral angle -/
  linearAngle : ℝ
  /-- The angle between the external normals of the dihedral angle -/
  externalNormalAngle : ℝ
  /-- The linear angle is between 0 and π -/
  linearAngle_bounds : 0 < linearAngle ∧ linearAngle < π
  /-- The external normal angle is between 0 and π -/
  externalNormalAngle_bounds : 0 < externalNormalAngle ∧ externalNormalAngle < π

/-- The sum of the external normal angle and the linear angle of a dihedral angle is π -/
theorem dihedral_angle_sum (d : DihedralAngle) : 
  d.externalNormalAngle + d.linearAngle = π :=
sorry

end dihedral_angle_sum_l3978_397849


namespace solutions_sum_greater_than_two_l3978_397823

noncomputable section

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^2 + 2/x

-- State the theorem
theorem solutions_sum_greater_than_two 
  (t : ℝ) 
  (h_t : t > 3) 
  (x₁ x₂ : ℝ) 
  (h_x₁ : x₁ > 0) 
  (h_x₂ : x₂ > 0) 
  (h_x₁_neq_x₂ : x₁ ≠ x₂) 
  (h_f_x₁ : f x₁ = t) 
  (h_f_x₂ : f x₂ = t) : 
  x₁ + x₂ > 2 := by
  sorry

end

end solutions_sum_greater_than_two_l3978_397823


namespace root_sum_l3978_397882

theorem root_sum (p q : ℝ) (h1 : q ≠ 0) (h2 : q^2 + p*q + q = 0) : p + q = -1 := by
  sorry

end root_sum_l3978_397882


namespace sum_of_digits_R50_div_R8_l3978_397803

def R (k : ℕ) : ℕ := (10^k - 1) / 9

theorem sum_of_digits_R50_div_R8 : ∃ (q : ℕ), R 50 = q * R 8 ∧ (q.digits 10).sum = 6 :=
sorry

end sum_of_digits_R50_div_R8_l3978_397803


namespace triangle_max_perimeter_l3978_397801

theorem triangle_max_perimeter :
  ∀ (x : ℕ),
  x > 0 →
  x + 2*x > 15 →
  x + 15 > 2*x →
  2*x + 15 > x →
  (∀ y : ℕ, y > 0 → y + 2*y > 15 → y + 15 > 2*y → 2*y + 15 > y → x + 2*x + 15 ≥ y + 2*y + 15) →
  x + 2*x + 15 = 57 := by
sorry

end triangle_max_perimeter_l3978_397801


namespace parallel_vectors_x_value_l3978_397831

/-- Two vectors are parallel if their corresponding components are proportional -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_x_value :
  let a : ℝ × ℝ := (2, 3)
  let b : ℝ × ℝ := (6, x)
  parallel a b → x = 9 := by
  sorry

end parallel_vectors_x_value_l3978_397831


namespace polyhedron_edge_intersection_l3978_397840

/-- A polyhedron with a given number of edges -/
structure Polyhedron where
  edges : ℕ

/-- A plane that can intersect edges of a polyhedron -/
structure IntersectingPlane where
  intersectedEdges : ℕ

/-- Represents a convex polyhedron -/
def ConvexPolyhedron (p : Polyhedron) : Prop := sorry

/-- Represents a non-convex polyhedron -/
def NonConvexPolyhedron (p : Polyhedron) : Prop := sorry

/-- The maximum number of edges that can be intersected by a plane in a convex polyhedron -/
def maxIntersectedEdgesConvex (p : Polyhedron) (plane : IntersectingPlane) : Prop :=
  ConvexPolyhedron p ∧ plane.intersectedEdges ≤ 68

/-- The number of edges that can be intersected by a plane in a non-convex polyhedron -/
def intersectedEdgesNonConvex (p : Polyhedron) (plane : IntersectingPlane) : Prop :=
  NonConvexPolyhedron p ∧ plane.intersectedEdges = 96

/-- The impossibility of intersecting all edges in any polyhedron -/
def cannotIntersectAllEdges (p : Polyhedron) (plane : IntersectingPlane) : Prop :=
  plane.intersectedEdges < p.edges

theorem polyhedron_edge_intersection (p : Polyhedron) (plane : IntersectingPlane) 
    (h : p.edges = 100) : 
    (maxIntersectedEdgesConvex p plane) ∧ 
    (∃ (p' : Polyhedron) (plane' : IntersectingPlane), intersectedEdgesNonConvex p' plane') ∧ 
    (cannotIntersectAllEdges p plane) := by
  sorry

end polyhedron_edge_intersection_l3978_397840


namespace age_difference_proof_l3978_397815

/-- Represents the age difference between Petra's mother and twice Petra's age --/
def age_difference (petra_age : ℕ) (mother_age : ℕ) : ℕ :=
  mother_age - 2 * petra_age

/-- Theorem stating the age difference between Petra's mother and twice Petra's age --/
theorem age_difference_proof :
  let petra_age : ℕ := 11
  let mother_age : ℕ := 36
  age_difference petra_age mother_age = 14 ∧
  petra_age + mother_age = 47 ∧
  ∃ (n : ℕ), mother_age = 2 * petra_age + n :=
by sorry

end age_difference_proof_l3978_397815


namespace opposite_expressions_solution_l3978_397841

theorem opposite_expressions_solution (x : ℚ) : (8*x - 7 = -(6 - 2*x)) → x = 1/6 := by
  sorry

end opposite_expressions_solution_l3978_397841


namespace max_four_digit_quotient_l3978_397859

def is_digit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

def is_nonzero_digit (n : ℕ) : Prop := n > 0 ∧ n ≤ 9

def four_digit_number (a b c d : ℕ) : ℕ := 1000 * a + 100 * b + 10 * c + d

def digit_sum (a b c d : ℕ) : ℕ := a + b + c + d

theorem max_four_digit_quotient :
  ∀ (a b c d : ℕ),
    is_nonzero_digit a →
    is_digit b →
    is_nonzero_digit c →
    is_nonzero_digit d →
    (four_digit_number a b c d) / (digit_sum a b c d) ≤ 337 :=
by sorry

end max_four_digit_quotient_l3978_397859


namespace parakeet_consumption_l3978_397877

/-- Represents the daily birdseed consumption of each bird type and the total for a week -/
structure BirdseedConsumption where
  parakeet : ℝ
  parrot : ℝ
  finch : ℝ
  total_weekly : ℝ

/-- The number of each type of bird -/
structure BirdCounts where
  parakeets : ℕ
  parrots : ℕ
  finches : ℕ

/-- Calculates the total daily consumption for all birds -/
def total_daily_consumption (c : BirdseedConsumption) (b : BirdCounts) : ℝ :=
  c.parakeet * b.parakeets + c.parrot * b.parrots + c.finch * b.finches

/-- Theorem stating the daily consumption of each parakeet -/
theorem parakeet_consumption (c : BirdseedConsumption) (b : BirdCounts) :
  c.parakeet = 2 ∧
  c.parrot = 14 ∧
  c.finch = c.parakeet / 2 ∧
  b.parakeets = 3 ∧
  b.parrots = 2 ∧
  b.finches = 4 ∧
  c.total_weekly = 266 ∧
  c.total_weekly = 7 * total_daily_consumption c b :=
by sorry

end parakeet_consumption_l3978_397877


namespace profit_share_difference_example_l3978_397891

/-- Calculates the difference in profit shares between two partners given their investments and the profit share of a third partner. -/
def profit_share_difference (invest_a invest_b invest_c b_profit : ℚ) : ℚ :=
  let total_invest := invest_a + invest_b + invest_c
  let total_profit := (total_invest * b_profit) / invest_b
  let a_share := (invest_a * total_profit) / total_invest
  let c_share := (invest_c * total_profit) / total_invest
  c_share - a_share

/-- Given the investments of A, B, and C as 8000, 10000, and 12000 respectively,
    and B's profit share as 1800, the difference between A's and C's profit shares is 720. -/
theorem profit_share_difference_example :
  profit_share_difference 8000 10000 12000 1800 = 720 := by
  sorry

end profit_share_difference_example_l3978_397891


namespace max_value_of_f_l3978_397865

/-- The function f(x) = -2x^2 + 4x + 10 -/
def f (x : ℝ) : ℝ := -2 * x^2 + 4 * x + 10

/-- The maximum value of f(x) for x ≥ 0 is 12 -/
theorem max_value_of_f :
  ∃ (M : ℝ), M = 12 ∧ ∀ (x : ℝ), x ≥ 0 → f x ≤ M :=
by sorry

end max_value_of_f_l3978_397865


namespace savings_ratio_proof_l3978_397869

def husband_contribution : ℕ := 335
def wife_contribution : ℕ := 225
def savings_period_months : ℕ := 6
def weeks_per_month : ℕ := 4
def num_children : ℕ := 4
def amount_per_child : ℕ := 1680

def total_weekly_contribution : ℕ := husband_contribution + wife_contribution
def total_weeks : ℕ := savings_period_months * weeks_per_month
def total_savings : ℕ := total_weekly_contribution * total_weeks
def total_divided : ℕ := amount_per_child * num_children

theorem savings_ratio_proof : 
  (total_divided : ℚ) / total_savings = 1/2 := by sorry

end savings_ratio_proof_l3978_397869


namespace allens_mother_age_l3978_397812

theorem allens_mother_age (allen_age mother_age : ℕ) : 
  allen_age = mother_age - 25 →
  allen_age + mother_age + 6 = 41 →
  mother_age = 30 := by
sorry

end allens_mother_age_l3978_397812


namespace fold_coincide_points_l3978_397851

/-- Given a number line where folding causes -2 to coincide with 8, and two points A and B
    with a distance of 2024 between them (A to the left of B) that coincide after folding,
    the coordinate of point A is -1009. -/
theorem fold_coincide_points (A B : ℝ) : 
  (A < B) →  -- A is to the left of B
  (B - A = 2024) →  -- Distance between A and B is 2024
  (A + B) / 2 = (-2 + 8) / 2 →  -- Midpoint of A and B is the same as midpoint of -2 and 8
  A = -1009 := by
sorry

end fold_coincide_points_l3978_397851


namespace completing_square_l3978_397820

theorem completing_square (x : ℝ) : x^2 - 4*x - 8 = 0 ↔ (x - 2)^2 = 12 := by
  sorry

end completing_square_l3978_397820


namespace ellipse_equation_l3978_397805

/-- An ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- The equation of an ellipse -/
def Ellipse.equation (e : Ellipse) (x y : ℝ) : Prop :=
  x^2 / e.a^2 + y^2 / e.b^2 = 1

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The condition that an ellipse passes through a point -/
def passes_through (e : Ellipse) (p : Point) : Prop :=
  e.equation p.x p.y

/-- The condition that two foci and one endpoint of the minor axis form an isosceles right triangle -/
def isosceles_right_triangle (e : Ellipse) : Prop :=
  e.a = Real.sqrt 2 * e.b

theorem ellipse_equation (e : Ellipse) (p : Point) 
    (h1 : passes_through e p)
    (h2 : p.x = 1 ∧ p.y = Real.sqrt 2 / 2)
    (h3 : isosceles_right_triangle e) :
    ∀ x y : ℝ, e.equation x y ↔ x^2 / 2 + y^2 = 1 := by
  sorry

end ellipse_equation_l3978_397805


namespace least_possible_qr_l3978_397893

/-- Triangle PQR with side lengths -/
structure TrianglePQR where
  pq : ℝ
  pr : ℝ
  qr : ℝ
  pq_positive : 0 < pq
  pr_positive : 0 < pr
  qr_positive : 0 < qr
  triangle_inequality_1 : qr < pq + pr
  triangle_inequality_2 : pq < qr + pr
  triangle_inequality_3 : pr < pq + qr

/-- Triangle SQR with side lengths -/
structure TriangleSQR where
  sq : ℝ
  sr : ℝ
  qr : ℝ
  sq_positive : 0 < sq
  sr_positive : 0 < sr
  qr_positive : 0 < qr
  triangle_inequality_1 : qr < sq + sr
  triangle_inequality_2 : sq < qr + sr
  triangle_inequality_3 : sr < sq + qr

/-- The theorem stating the least possible integral length of QR -/
theorem least_possible_qr 
  (triangle_pqr : TrianglePQR)
  (triangle_sqr : TriangleSQR)
  (h_pq : triangle_pqr.pq = 7)
  (h_pr : triangle_pqr.pr = 10)
  (h_sq : triangle_sqr.sq = 24)
  (h_sr : triangle_sqr.sr = 15)
  (h_qr_same : triangle_pqr.qr = triangle_sqr.qr)
  (h_qr_int : ∃ n : ℕ, triangle_pqr.qr = n) :
  triangle_pqr.qr = 9 ∧ ∀ m : ℕ, (m : ℝ) = triangle_pqr.qr → m ≥ 9 :=
sorry

end least_possible_qr_l3978_397893


namespace peach_distribution_problem_l3978_397808

/-- Represents the distribution of peaches among monkeys -/
structure PeachDistribution where
  total_peaches : ℕ
  num_monkeys : ℕ

/-- Checks if the distribution satisfies the first condition -/
def satisfies_condition1 (d : PeachDistribution) : Prop :=
  2 * 4 + (d.num_monkeys - 2) * 2 + 4 = d.total_peaches

/-- Checks if the distribution satisfies the second condition -/
def satisfies_condition2 (d : PeachDistribution) : Prop :=
  1 * 6 + (d.num_monkeys - 1) * 4 = d.total_peaches + 12

/-- The theorem to be proved -/
theorem peach_distribution_problem :
  ∃ (d : PeachDistribution),
    d.total_peaches = 26 ∧
    d.num_monkeys = 9 ∧
    satisfies_condition1 d ∧
    satisfies_condition2 d :=
by sorry

end peach_distribution_problem_l3978_397808


namespace ceiling_sqrt_165_l3978_397822

theorem ceiling_sqrt_165 : ⌈Real.sqrt 165⌉ = 13 := by
  sorry

end ceiling_sqrt_165_l3978_397822


namespace stack_logs_count_l3978_397846

/-- The number of logs in a stack with arithmetic progression of rows -/
def logsInStack (bottomRow : ℕ) (topRow : ℕ) : ℕ :=
  let numRows := bottomRow - topRow + 1
  numRows * (bottomRow + topRow) / 2

theorem stack_logs_count : logsInStack 15 5 = 110 := by
  sorry

end stack_logs_count_l3978_397846


namespace smallest_packages_for_more_envelopes_l3978_397857

theorem smallest_packages_for_more_envelopes (n : ℕ) : 
  (∀ k : ℕ, k < n → 10 * k ≤ 8 * k + 7) ∧ 
  (10 * n > 8 * n + 7) → 
  n = 4 := by
  sorry

end smallest_packages_for_more_envelopes_l3978_397857


namespace kendras_cookies_l3978_397887

/-- Kendra's cookie baking problem -/
theorem kendras_cookies :
  ∀ (cookies_per_batch : ℕ)
    (family_size : ℕ)
    (chips_per_cookie : ℕ)
    (chips_per_person : ℕ),
  cookies_per_batch = 12 →
  family_size = 4 →
  chips_per_cookie = 2 →
  chips_per_person = 18 →
  (chips_per_person / chips_per_cookie * family_size) / cookies_per_batch = 3 :=
by
  sorry


end kendras_cookies_l3978_397887
