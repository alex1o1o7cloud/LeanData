import Mathlib

namespace m_3_sufficient_not_necessary_l3360_336024

def A (m : ℝ) : Set ℝ := {-1, m^2}
def B : Set ℝ := {2, 9}

theorem m_3_sufficient_not_necessary :
  (∀ m : ℝ, m = 3 → A m ∩ B = {9}) ∧
  ¬(∀ m : ℝ, A m ∩ B = {9} → m = 3) := by
  sorry

end m_3_sufficient_not_necessary_l3360_336024


namespace final_stamp_count_l3360_336008

/-- Represents the number of stamps in Tom's collection -/
def stamps_collection (initial : ℕ) (mike_gift : ℕ) : ℕ → ℕ
  | harry_gift => initial + mike_gift + harry_gift

/-- Theorem: Tom's final stamp collection contains 3,061 stamps -/
theorem final_stamp_count :
  let initial := 3000
  let mike_gift := 17
  let harry_gift := 2 * mike_gift + 10
  stamps_collection initial mike_gift harry_gift = 3061 := by
  sorry

#check final_stamp_count

end final_stamp_count_l3360_336008


namespace target_hit_probability_l3360_336077

theorem target_hit_probability 
  (prob_A : ℝ) 
  (prob_B : ℝ) 
  (h_prob_A : prob_A = 0.8) 
  (h_prob_B : prob_B = 0.7) 
  (h_independent : True) -- Assumption of independence
  : 1 - (1 - prob_A) * (1 - prob_B) = 0.94 := by
  sorry

end target_hit_probability_l3360_336077


namespace distance_from_blast_site_l3360_336032

/-- The speed of sound in meters per second -/
def speed_of_sound : ℝ := 330

/-- The time between the first blast and when the man heard the second blast, in seconds -/
def time_between_blasts : ℝ := 30 * 60 + 24

/-- The time between the first and second blasts, in seconds -/
def time_between_actual_blasts : ℝ := 30 * 60

/-- The distance the man traveled when he heard the second blast -/
def distance_traveled : ℝ := speed_of_sound * (time_between_blasts - time_between_actual_blasts)

theorem distance_from_blast_site :
  distance_traveled = 7920 := by sorry

end distance_from_blast_site_l3360_336032


namespace bens_savings_proof_l3360_336028

/-- Ben's daily savings before his parents' contributions --/
def daily_savings : ℕ := 50 - 15

/-- The number of days Ben saved money --/
def num_days : ℕ := 7

/-- Ben's total savings after his mom doubled it --/
def doubled_savings : ℕ := 2 * (daily_savings * num_days)

/-- Ben's final amount after 7 days --/
def final_amount : ℕ := 500

/-- The additional amount Ben's dad gave him --/
def dads_contribution : ℕ := final_amount - doubled_savings

theorem bens_savings_proof :
  dads_contribution = 10 :=
sorry

end bens_savings_proof_l3360_336028


namespace intersection_on_unit_circle_l3360_336042

theorem intersection_on_unit_circle (k₁ k₂ : ℝ) (h : k₁ * k₂ + 1 = 0) :
  ∃ (x y : ℝ), (y = k₁ * x + 1) ∧ (y = k₂ * x - 1) ∧ (x^2 + y^2 = 1) := by
  sorry

end intersection_on_unit_circle_l3360_336042


namespace bean_sprouts_and_dried_tofu_problem_l3360_336001

/-- Bean sprouts and dried tofu problem -/
theorem bean_sprouts_and_dried_tofu_problem 
  (bean_sprouts_price dried_tofu_price : ℚ)
  (bean_sprouts_sell_price dried_tofu_sell_price : ℚ)
  (total_units : ℕ)
  (max_cost : ℚ) :
  bean_sprouts_price = 60 →
  dried_tofu_price = 40 →
  bean_sprouts_sell_price = 80 →
  dried_tofu_sell_price = 55 →
  total_units = 200 →
  max_cost = 10440 →
  2 * bean_sprouts_price + 3 * dried_tofu_price = 240 →
  3 * bean_sprouts_price + 4 * dried_tofu_price = 340 →
  ∃ (bean_sprouts_units dried_tofu_units : ℕ),
    bean_sprouts_units + dried_tofu_units = total_units ∧
    bean_sprouts_price * bean_sprouts_units + dried_tofu_price * dried_tofu_units ≤ max_cost ∧
    (bean_sprouts_units : ℚ) ≥ (3/2) * dried_tofu_units ∧
    bean_sprouts_units = 122 ∧
    dried_tofu_units = 78 ∧
    (bean_sprouts_sell_price - bean_sprouts_price) * bean_sprouts_units +
    (dried_tofu_sell_price - dried_tofu_price) * dried_tofu_units = 3610 ∧
    ∀ (other_bean_sprouts_units other_dried_tofu_units : ℕ),
      other_bean_sprouts_units + other_dried_tofu_units = total_units →
      bean_sprouts_price * other_bean_sprouts_units + dried_tofu_price * other_dried_tofu_units ≤ max_cost →
      (other_bean_sprouts_units : ℚ) ≥ (3/2) * other_dried_tofu_units →
      (bean_sprouts_sell_price - bean_sprouts_price) * other_bean_sprouts_units +
      (dried_tofu_sell_price - dried_tofu_price) * other_dried_tofu_units ≤ 3610 :=
by
  sorry

end bean_sprouts_and_dried_tofu_problem_l3360_336001


namespace parabola_focus_equation_l3360_336090

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in the form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a parabola in standard form -/
inductive Parabola
  | VertexAtOrigin (p : ℝ) : Parabola
  | FocusOnXAxis (p : ℝ) : Parabola

/-- Function to check if a point is on a line -/
def isPointOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Function to check if a point is on the x-axis -/
def isPointOnXAxis (p : Point) : Prop :=
  p.y = 0

/-- Function to check if a point is on the y-axis -/
def isPointOnYAxis (p : Point) : Prop :=
  p.x = 0

/-- Theorem stating the relationship between the focus of a parabola and its equation -/
theorem parabola_focus_equation (l : Line) (f : Point) :
  (l.a = 3 ∧ l.b = -4 ∧ l.c = -12) →
  isPointOnLine f l →
  (isPointOnXAxis f ∨ isPointOnYAxis f) →
  (∃ p : Parabola, p = Parabola.VertexAtOrigin (-12) ∨ p = Parabola.FocusOnXAxis 8) :=
sorry

end parabola_focus_equation_l3360_336090


namespace inverse_f_sum_l3360_336094

-- Define the function f(x) = x|x|
def f (x : ℝ) : ℝ := x * abs x

-- State the theorem
theorem inverse_f_sum : ∃ y z : ℝ, f y = 9 ∧ f z = -81 ∧ y + z = -6 := by sorry

end inverse_f_sum_l3360_336094


namespace prob_at_least_one_event_l3360_336030

theorem prob_at_least_one_event (P₁ P₂ : ℝ) 
  (h₁ : 0 ≤ P₁ ∧ P₁ ≤ 1) 
  (h₂ : 0 ≤ P₂ ∧ P₂ ≤ 1) :
  P₁ + P₂ - P₁ * P₂ = 1 - (1 - P₁) * (1 - P₂) :=
by sorry

end prob_at_least_one_event_l3360_336030


namespace exists_quadratic_without_cyclic_solution_l3360_336084

/-- A quadratic polynomial function -/
def QuadraticPolynomial := ℝ → ℝ

/-- Property that checks if a function satisfies the cyclic condition for given a, b, c, d -/
def SatisfiesCyclicCondition (f : QuadraticPolynomial) (a b c d : ℝ) : Prop :=
  f a = b ∧ f b = c ∧ f c = d ∧ f d = a

/-- Theorem stating that there exists a quadratic polynomial for which no distinct a, b, c, d satisfy the cyclic condition -/
theorem exists_quadratic_without_cyclic_solution :
  ∃ f : QuadraticPolynomial, ∀ a b c d : ℝ, a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ a →
    ¬(SatisfiesCyclicCondition f a b c d) :=
sorry

end exists_quadratic_without_cyclic_solution_l3360_336084


namespace greatest_integer_less_than_negative_twenty_two_thirds_l3360_336041

theorem greatest_integer_less_than_negative_twenty_two_thirds :
  Int.floor (-22 / 3) = -8 := by
  sorry

end greatest_integer_less_than_negative_twenty_two_thirds_l3360_336041


namespace rectangle_area_and_diagonal_l3360_336027

/-- Represents a rectangle with length, width, perimeter, area, and diagonal --/
structure Rectangle where
  length : ℝ
  width : ℝ
  perimeter : ℝ
  area : ℝ
  diagonal : ℝ

/-- Theorem about the area and diagonal of a specific rectangle --/
theorem rectangle_area_and_diagonal (r : Rectangle) 
  (h1 : r.length = 4 * r.width)
  (h2 : r.perimeter = 200) :
  r.area = 1600 ∧ r.diagonal = Real.sqrt 6800 := by
  sorry

#check rectangle_area_and_diagonal

end rectangle_area_and_diagonal_l3360_336027


namespace total_matches_proof_l3360_336088

def grade1_classes : ℕ := 5
def grade2_classes : ℕ := 7
def grade3_classes : ℕ := 4

def matches_in_tournament (n : ℕ) : ℕ := n * (n - 1) / 2

theorem total_matches_proof :
  matches_in_tournament grade1_classes +
  matches_in_tournament grade2_classes +
  matches_in_tournament grade3_classes = 37 := by
sorry

end total_matches_proof_l3360_336088


namespace unique_solution_for_all_y_l3360_336010

theorem unique_solution_for_all_y :
  ∃! x : ℚ, ∀ y : ℚ, 10 * x * y - 15 * y + 3 * x - 9 / 2 = 0 :=
by
  -- The unique solution is x = 3/2
  use 3 / 2
  sorry

end unique_solution_for_all_y_l3360_336010


namespace marcus_percentage_of_team_points_l3360_336098

/-- Represents the number of points for each type of goal -/
def threePointValue : ℕ := 3
def twoPointValue : ℕ := 2

/-- Represents the number of goals Marcus scored -/
def marcusThreePointers : ℕ := 5
def marcusTwoPointers : ℕ := 10

/-- Represents the total points scored by the team -/
def teamTotalPoints : ℕ := 70

/-- Calculates the total points scored by Marcus -/
def marcusTotalPoints : ℕ :=
  marcusThreePointers * threePointValue + marcusTwoPointers * twoPointValue

/-- Theorem: Marcus scored 50% of the team's total points -/
theorem marcus_percentage_of_team_points :
  (marcusTotalPoints : ℚ) / teamTotalPoints = 1/2 := by
  sorry

end marcus_percentage_of_team_points_l3360_336098


namespace inverse_of_A_cubed_l3360_336078

theorem inverse_of_A_cubed (A : Matrix (Fin 2) (Fin 2) ℝ) 
  (h : A⁻¹ = !![3, 7; -2, -5]) : 
  (A^3)⁻¹ = !![13, -15; -14, -29] := by sorry

end inverse_of_A_cubed_l3360_336078


namespace lcm_factor_proof_l3360_336096

/-- Given two positive integers with specific HCF and LCM properties, prove the other factor of their LCM -/
theorem lcm_factor_proof (A B : ℕ) (hA : A = 460) (hHCF : Nat.gcd A B = 20) 
  (hLCM_factor : ∃ (k : ℕ), Nat.lcm A B = 20 * 21 * k) : 
  ∃ (k : ℕ), Nat.lcm A B = 20 * 21 * 23 := by
  sorry

end lcm_factor_proof_l3360_336096


namespace ten_point_six_trillion_scientific_notation_l3360_336082

-- Define a trillion
def trillion : ℝ := 10^12

-- State the theorem
theorem ten_point_six_trillion_scientific_notation :
  (10.6 * trillion) = 1.06 * 10^13 := by sorry

end ten_point_six_trillion_scientific_notation_l3360_336082


namespace escalator_speed_l3360_336062

/-- Given an escalator and a person walking on it, calculate the escalator's speed. -/
theorem escalator_speed (escalator_length : ℝ) (person_speed : ℝ) (time_taken : ℝ) : 
  escalator_length = 112 →
  person_speed = 4 →
  time_taken = 8 →
  (person_speed + (escalator_length / time_taken - person_speed)) * time_taken = escalator_length →
  escalator_length / time_taken - person_speed = 10 := by
  sorry

end escalator_speed_l3360_336062


namespace absolute_value_square_sum_zero_l3360_336070

theorem absolute_value_square_sum_zero (x y : ℝ) :
  |x + 2| + (y - 1)^2 = 0 → x = -2 ∧ y = 1 := by
sorry

end absolute_value_square_sum_zero_l3360_336070


namespace distance_to_external_point_specific_distance_to_external_point_l3360_336007

/-- Given a circle with radius r and two tangents drawn from a common external point P
    with a sum length of s, the distance from the center O to P is sqrt(r^2 + (s/2)^2). -/
theorem distance_to_external_point (r s : ℝ) (hr : r > 0) (hs : s > 0) :
  let d := Real.sqrt (r^2 + (s/2)^2)
  d = Real.sqrt (r^2 + (s/2)^2) := by
  sorry

/-- For a circle with radius 11 and two tangents with sum length 120,
    the distance from the center to the external point is 61. -/
theorem specific_distance_to_external_point :
  let r : ℝ := 11
  let s : ℝ := 120
  let d := Real.sqrt (r^2 + (s/2)^2)
  d = 61 := by
  sorry

end distance_to_external_point_specific_distance_to_external_point_l3360_336007


namespace absent_students_count_l3360_336083

/-- The number of classes at Webster Middle School -/
def num_classes : ℕ := 18

/-- The number of students in each class -/
def students_per_class : ℕ := 28

/-- The number of students present on Monday -/
def students_present : ℕ := 496

/-- The number of absent students -/
def absent_students : ℕ := num_classes * students_per_class - students_present

theorem absent_students_count : absent_students = 8 := by
  sorry

end absent_students_count_l3360_336083


namespace string_average_length_l3360_336058

theorem string_average_length : 
  let strings : List ℚ := [2, 5, 7]
  (strings.sum / strings.length : ℚ) = 14/3 := by
sorry

end string_average_length_l3360_336058


namespace brick_volume_l3360_336039

/-- The volume of a rectangular prism -/
def volume (length width height : ℝ) : ℝ := length * width * height

/-- Theorem: The volume of a brick with dimensions 9 cm × 4 cm × 7 cm is 252 cubic centimeters -/
theorem brick_volume :
  volume 4 9 7 = 252 := by
  sorry

end brick_volume_l3360_336039


namespace arithmetic_sequence_ratio_l3360_336046

/-- Sum of first n terms of an arithmetic sequence -/
def S (n : ℕ) (a : ℕ → ℝ) : ℝ := sorry

/-- The sequence a is arithmetic -/
def is_arithmetic (a : ℕ → ℝ) : Prop := sorry

theorem arithmetic_sequence_ratio (a : ℕ → ℝ) (h : is_arithmetic a) 
  (h1 : S 3 a / S 6 a = 1 / 3) : S 9 a / S 6 a = 2 := by sorry

end arithmetic_sequence_ratio_l3360_336046


namespace female_democrat_ratio_l3360_336061

theorem female_democrat_ratio (total_participants male_participants female_participants : ℕ)
  (female_democrats : ℕ) (h1 : total_participants = 870)
  (h2 : male_participants + female_participants = total_participants)
  (h3 : female_democrats = 145)
  (h4 : 4 * (male_participants / 4 + female_democrats) = total_participants) :
  2 * female_democrats = female_participants :=
by sorry

end female_democrat_ratio_l3360_336061


namespace function_minimum_implies_a_less_than_one_l3360_336013

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + a

-- State the theorem
theorem function_minimum_implies_a_less_than_one :
  ∀ a : ℝ, (∃ m : ℝ, ∀ x < 1, f a x ≥ f a m) → a < 1 := by
  sorry

end function_minimum_implies_a_less_than_one_l3360_336013


namespace tangent_point_abscissa_l3360_336051

noncomputable section

-- Define the function f(x) = x^2 + x - ln x
def f (x : ℝ) : ℝ := x^2 + x - Real.log x

-- Define the derivative of f(x)
def f_deriv (x : ℝ) : ℝ := 2*x + 1 - 1/x

-- Theorem statement
theorem tangent_point_abscissa (t : ℝ) (h : t > 0) :
  (f t / t = f_deriv t) → t = 1 :=
sorry


end tangent_point_abscissa_l3360_336051


namespace remainder_theorem_l3360_336089

theorem remainder_theorem (x y u v : ℕ) (hx : x > 0) (hy : y > 0) 
  (h_div : x = u * y + v) (h_rem : v < y) : 
  (x + 2 * u * y) % y = v := by
sorry

end remainder_theorem_l3360_336089


namespace certain_number_problem_l3360_336056

theorem certain_number_problem (x : ℝ) : 
  x - (1/4)*2 - (1/3)*3 - (1/7)*x = 27 → 
  (10/100) * x = 3.325 := by
sorry

end certain_number_problem_l3360_336056


namespace largest_multiple_of_15_under_500_l3360_336086

theorem largest_multiple_of_15_under_500 : ∃ (n : ℕ), n = 495 ∧ 
  (∀ m : ℕ, m % 15 = 0 → m < 500 → m ≤ n) := by
  sorry

end largest_multiple_of_15_under_500_l3360_336086


namespace point_four_units_from_negative_three_l3360_336018

theorem point_four_units_from_negative_three :
  {x : ℝ | |x - (-3)| = 4} = {1, -7} := by sorry

end point_four_units_from_negative_three_l3360_336018


namespace intersection_range_l3360_336074

/-- The range of k for which the line y = kx + 2 intersects the right branch of 
    the hyperbola x^2 - y^2 = 6 at two distinct points -/
theorem intersection_range :
  ∀ k : ℝ, 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
   x₁ > 0 ∧ x₂ > 0 ∧
   x₁^2 - (k * x₁ + 2)^2 = 6 ∧
   x₂^2 - (k * x₂ + 2)^2 = 6) →
  -Real.sqrt 15 / 3 < k ∧ k < -1 :=
by sorry

end intersection_range_l3360_336074


namespace prob_b_in_middle_l3360_336047

def number_of_people : ℕ := 3

def total_arrangements (n : ℕ) : ℕ := n.factorial

def middle_arrangements (n : ℕ) : ℕ := (n - 1).factorial

def probability_in_middle (n : ℕ) : ℚ :=
  (middle_arrangements n : ℚ) / (total_arrangements n : ℚ)

theorem prob_b_in_middle :
  probability_in_middle number_of_people = 1 / 3 := by
  sorry

end prob_b_in_middle_l3360_336047


namespace complex_fraction_sum_l3360_336031

theorem complex_fraction_sum (x y : ℝ) :
  (1 - Complex.I) / (2 + Complex.I) = Complex.mk x y →
  x + y = -2/5 := by sorry

end complex_fraction_sum_l3360_336031


namespace goat_roaming_area_specific_case_l3360_336075

/-- Represents the dimensions of a rectangular shed -/
structure ShedDimensions where
  length : ℝ
  width : ℝ

/-- Calculates the area a goat can roam when tied to the corner of a rectangular shed -/
def goatRoamingArea (shed : ShedDimensions) (leashLength : ℝ) : ℝ :=
  sorry

/-- Theorem stating the area a goat can roam under specific conditions -/
theorem goat_roaming_area_specific_case :
  let shed : ShedDimensions := { length := 5, width := 4 }
  let leashLength : ℝ := 4
  goatRoamingArea shed leashLength = 12.25 * Real.pi := by sorry

end goat_roaming_area_specific_case_l3360_336075


namespace triangle_side_and_area_l3360_336021

theorem triangle_side_and_area 
  (A B C : ℝ) -- Angles of the triangle
  (a b c : ℝ) -- Sides of the triangle
  (h1 : c = 2)
  (h2 : Real.sin A = 2 * Real.sin C)
  (h3 : Real.cos B = 1/4)
  (h4 : a = c * Real.sin A / Real.sin C) -- Sine law
  (h5 : 0 < Real.sin C) -- Assumption to avoid division by zero
  (h6 : 0 ≤ A ∧ A < π) -- Assumption for valid angle A
  (h7 : 0 ≤ B ∧ B < π) -- Assumption for valid angle B
  (h8 : 0 ≤ C ∧ C < π) -- Assumption for valid angle C
  : a = 4 ∧ (1/2 * a * c * Real.sin B = Real.sqrt 15) := by
  sorry

#check triangle_side_and_area

end triangle_side_and_area_l3360_336021


namespace quadratic_minimum_l3360_336011

theorem quadratic_minimum (x : ℝ) :
  ∃ (min : ℝ), ∀ y : ℝ, y = 4 * x^2 + 8 * x + 16 → y ≥ min ∧ ∃ x₀ : ℝ, 4 * x₀^2 + 8 * x₀ + 16 = min :=
by
  -- The proof goes here
  sorry

end quadratic_minimum_l3360_336011


namespace asha_money_problem_l3360_336076

/-- Asha's money problem -/
theorem asha_money_problem (brother_loan : ℕ) (father_loan : ℕ) (mother_loan : ℕ) (granny_gift : ℕ) (savings : ℕ) (spent_fraction : ℚ) :
  brother_loan = 20 →
  father_loan = 40 →
  mother_loan = 30 →
  granny_gift = 70 →
  savings = 100 →
  spent_fraction = 3 / 4 →
  ∃ (remaining : ℕ), remaining = 65 ∧ 
    remaining = (brother_loan + father_loan + mother_loan + granny_gift + savings) - 
                (spent_fraction * (brother_loan + father_loan + mother_loan + granny_gift + savings)).floor :=
by sorry

end asha_money_problem_l3360_336076


namespace no_real_solutions_l3360_336049

theorem no_real_solutions : ¬∃ x : ℝ, (2*x - 10*x + 24)^2 + 4 = -2*|x| := by
  sorry

end no_real_solutions_l3360_336049


namespace preimage_of_neg_one_two_l3360_336052

/-- A mapping f from ℝ² to ℝ² defined as f(x, y) = (2x, x - y) -/
def f : ℝ × ℝ → ℝ × ℝ := fun (x, y) ↦ (2 * x, x - y)

/-- Theorem stating that f(-1/2, -5/2) = (-1, 2) -/
theorem preimage_of_neg_one_two :
  f (-1/2, -5/2) = (-1, 2) := by
  sorry

end preimage_of_neg_one_two_l3360_336052


namespace quadratic_inequality_condition_l3360_336050

theorem quadratic_inequality_condition (a : ℝ) :
  ((∀ x : ℝ, x^2 - 2*a*x + a > 0) → (0 ≤ a ∧ a ≤ 1)) ∧
  ¬((0 ≤ a ∧ a ≤ 1) → (∀ x : ℝ, x^2 - 2*a*x + a > 0)) :=
by sorry

end quadratic_inequality_condition_l3360_336050


namespace marble_remainder_l3360_336022

theorem marble_remainder (n m k : ℤ) : ∃ q : ℤ, (8*n + 5) + (8*m + 3) + (8*k + 7) = 8*q + 7 := by
  sorry

end marble_remainder_l3360_336022


namespace smallest_positive_integer_satisfying_congruences_l3360_336005

theorem smallest_positive_integer_satisfying_congruences :
  ∃ x : ℕ+, 
    (45 * x.val + 15) % 25 = 5 ∧ 
    x.val % 4 = 3 ∧
    ∀ y : ℕ+, 
      ((45 * y.val + 15) % 25 = 5 ∧ y.val % 4 = 3) → 
      x ≤ y :=
by
  -- The proof goes here
  sorry

end smallest_positive_integer_satisfying_congruences_l3360_336005


namespace lcm_gcf_problem_l3360_336067

theorem lcm_gcf_problem (n : ℕ) (h1 : Nat.lcm n 16 = 48) (h2 : Nat.gcd n 16 = 4) : n = 12 := by
  sorry

end lcm_gcf_problem_l3360_336067


namespace blue_to_black_pen_ratio_l3360_336044

/-- Given the conditions of John's pen collection, prove the ratio of blue to black pens --/
theorem blue_to_black_pen_ratio :
  ∀ (blue black red : ℕ),
  blue + black + red = 31 →
  black = red + 5 →
  blue = 18 →
  blue / black = 2 :=
by
  sorry

end blue_to_black_pen_ratio_l3360_336044


namespace sum_congruence_l3360_336020

theorem sum_congruence (a b c : ℕ) : 
  a < 11 → b < 11 → c < 11 → a > 0 → b > 0 → c > 0 →
  (a * b * c) % 11 = 1 → 
  (7 * c) % 11 = 4 → 
  (8 * b) % 11 = (5 + b) % 11 → 
  (a + b + c) % 11 = 9 := by
sorry

end sum_congruence_l3360_336020


namespace green_hats_count_l3360_336093

/-- Proves that the number of green hats is 20 given the conditions of the problem -/
theorem green_hats_count (total_hats : ℕ) (blue_price green_price total_price : ℕ) 
  (h1 : total_hats = 85)
  (h2 : blue_price = 6)
  (h3 : green_price = 7)
  (h4 : total_price = 530) :
  ∃ (blue_hats green_hats : ℕ), 
    blue_hats + green_hats = total_hats ∧
    blue_price * blue_hats + green_price * green_hats = total_price ∧
    green_hats = 20 := by
  sorry

#check green_hats_count

end green_hats_count_l3360_336093


namespace continued_fraction_result_l3360_336012

/-- Given x satisfying the infinite continued fraction equation,
    prove that 1/((x+2)(x-3)) equals (-√3 - 2) / 2 -/
theorem continued_fraction_result (x : ℝ) 
  (hx : x = 2 + Real.sqrt 3 / (2 + Real.sqrt 3 / (2 + Real.sqrt 3 / x))) :
  1 / ((x + 2) * (x - 3)) = (-Real.sqrt 3 - 2) / 2 := by
  sorry

end continued_fraction_result_l3360_336012


namespace impossibleDivision_l3360_336035

/-- Represents an employee with their salary -/
structure Employee :=
  (salary : ℝ)

/-- Represents a region with its employees -/
structure Region :=
  (employees : List Employee)

/-- The total salary of a region -/
def totalSalary (r : Region) : ℝ :=
  (r.employees.map Employee.salary).sum

/-- The condition that 10% of employees get 90% of total salary -/
def salaryDistributionCondition (employees : List Employee) : Prop :=
  ∃ (highPaidEmployees : List Employee),
    highPaidEmployees.length = (employees.length / 10) ∧
    (highPaidEmployees.map Employee.salary).sum ≥ 0.9 * ((employees.map Employee.salary).sum)

/-- The condition for a valid region division -/
def validRegionDivision (regions : List Region) : Prop :=
  ∀ r ∈ regions, ∀ subset : List Employee,
    subset.length = (r.employees.length / 10) →
    (subset.map Employee.salary).sum ≤ 0.11 * totalSalary r

/-- The main theorem -/
theorem impossibleDivision :
  ∃ (employees : List Employee),
    salaryDistributionCondition employees ∧
    ¬∃ (regions : List Region),
      (regions.map Region.employees).join = employees ∧
      validRegionDivision regions :=
sorry

end impossibleDivision_l3360_336035


namespace lecture_scheduling_l3360_336019

theorem lecture_scheduling (n : ℕ) (h : n = 6) :
  (n! / 2 : ℕ) = 360 := by
  sorry

#check lecture_scheduling

end lecture_scheduling_l3360_336019


namespace least_multiple_33_greater_500_l3360_336060

theorem least_multiple_33_greater_500 : ∃ (n : ℕ), n * 33 = 528 ∧ 
  528 > 500 ∧ 
  (∀ (m : ℕ), m * 33 > 500 → m * 33 ≥ 528) := by
  sorry

end least_multiple_33_greater_500_l3360_336060


namespace smallest_number_property_l3360_336068

/-- The smallest natural number divisible by 5 with a digit sum of 100 -/
def smallest_number : ℕ := 599999999995

/-- Function to calculate the sum of digits of a natural number -/
def digit_sum (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + digit_sum (n / 10)

/-- Theorem stating that 599999999995 is the smallest natural number divisible by 5 with a digit sum of 100 -/
theorem smallest_number_property :
  (∀ m : ℕ, m < smallest_number → (m % 5 = 0 → digit_sum m ≠ 100)) ∧
  smallest_number % 5 = 0 ∧
  digit_sum smallest_number = 100 :=
by sorry

#eval smallest_number
#eval digit_sum smallest_number
#eval smallest_number % 5

end smallest_number_property_l3360_336068


namespace merry_and_brother_lambs_l3360_336004

/-- The number of lambs Merry and her brother have in total -/
def total_lambs (merry_lambs : ℕ) (brother_extra : ℕ) : ℕ :=
  merry_lambs + (merry_lambs + brother_extra)

/-- Theorem stating the total number of lambs Merry and her brother have -/
theorem merry_and_brother_lambs :
  total_lambs 10 3 = 23 := by
  sorry

end merry_and_brother_lambs_l3360_336004


namespace min_value_2x_plus_y_l3360_336059

theorem min_value_2x_plus_y :
  ∀ x y : ℝ, (|y| ≤ 2 - x ∧ x ≥ -1) → (∀ x' y' : ℝ, |y'| ≤ 2 - x' ∧ x' ≥ -1 → 2*x + y ≤ 2*x' + y') ∧ (∃ x₀ y₀ : ℝ, |y₀| ≤ 2 - x₀ ∧ x₀ ≥ -1 ∧ 2*x₀ + y₀ = -5) :=
by sorry

end min_value_2x_plus_y_l3360_336059


namespace product_of_sum_of_squares_l3360_336055

theorem product_of_sum_of_squares (a b c d : ℤ) :
  let m := a^2 + b^2
  let n := c^2 + d^2
  m * n = (a*c - b*d)^2 + (a*d + b*c)^2 := by sorry

end product_of_sum_of_squares_l3360_336055


namespace functional_equation_solution_l3360_336003

open Real

/-- Given a function g: ℝ → ℝ satisfying the functional equation
    (g(x) * g(y) - g(x*y)) / 5 = x + y + 4 for all x, y ∈ ℝ,
    prove that g(x) = x + 5 for all x ∈ ℝ. -/
theorem functional_equation_solution (g : ℝ → ℝ) 
    (h : ∀ x y : ℝ, (g x * g y - g (x * y)) / 5 = x + y + 4) :
  ∀ x : ℝ, g x = x + 5 := by
  sorry

end functional_equation_solution_l3360_336003


namespace convention_center_tables_l3360_336099

/-- Converts a number from base 7 to base 10 -/
def base7ToBase10 (n : Nat) : Nat :=
  (n / 100) * 7^2 + ((n / 10) % 10) * 7^1 + (n % 10) * 7^0

/-- Calculates the number of tables needed given the total number of people and people per table -/
def calculateTables (totalPeople : Nat) (peoplePerTable : Nat) : Nat :=
  totalPeople / peoplePerTable

theorem convention_center_tables :
  let seatingCapacityBase7 : Nat := 315
  let peoplePerTable : Nat := 3
  let totalPeopleBase10 : Nat := base7ToBase10 seatingCapacityBase7
  calculateTables totalPeopleBase10 peoplePerTable = 53 := by
  sorry

end convention_center_tables_l3360_336099


namespace sqrt_equation_solution_l3360_336054

theorem sqrt_equation_solution :
  ∃ y : ℝ, (Real.sqrt 27 + Real.sqrt y) / Real.sqrt 75 = 2.4 → y = 243 :=
by
  sorry

end sqrt_equation_solution_l3360_336054


namespace grid_coverage_iff_divisible_by_four_l3360_336087

/-- A T-tetromino is a set of four cells in the shape of a "T" -/
def TTetromino : Type := Unit

/-- Represents the property of an n × n grid being completely covered by T-tetrominoes without overlapping -/
def is_completely_covered (n : ℕ) : Prop := sorry

theorem grid_coverage_iff_divisible_by_four (n : ℕ) : 
  is_completely_covered n ↔ 4 ∣ n :=
sorry

end grid_coverage_iff_divisible_by_four_l3360_336087


namespace root_sum_squares_l3360_336073

theorem root_sum_squares (p q r : ℝ) : 
  (p + q + r = 15) → 
  (p * q + q * r + r * p = 22) →
  (p * q * r = 8) →
  (p + q)^2 + (q + r)^2 + (r + p)^2 = 406 := by
sorry

end root_sum_squares_l3360_336073


namespace pure_imaginary_modulus_l3360_336081

theorem pure_imaginary_modulus (a : ℝ) : 
  let z : ℂ := a^2 - 1 + (a + 1) * Complex.I
  (z.re = 0 ∧ z.im ≠ 0) → Complex.abs z = 2 := by
  sorry

end pure_imaginary_modulus_l3360_336081


namespace checkerboard_fraction_l3360_336017

/-- The number of rectangles formed on a 7x7 checkerboard with 8 horizontal and 8 vertical lines -/
def r : ℕ := 784

/-- The number of squares formed on a 7x7 checkerboard -/
def s : ℕ := 140

/-- m and n are relatively prime positive integers such that s/r = m/n -/
theorem checkerboard_fraction (m n : ℕ) (h1 : m.gcd n = 1) (h2 : m > 0) (h3 : n > 0) 
  (h4 : s * n = r * m) : m + n = 33 := by
  sorry

end checkerboard_fraction_l3360_336017


namespace exists_number_satisfying_condition_l3360_336000

theorem exists_number_satisfying_condition : ∃ X : ℝ, 0.60 * X = 0.30 * 800 + 370 := by
  sorry

end exists_number_satisfying_condition_l3360_336000


namespace gcd_of_three_numbers_l3360_336002

theorem gcd_of_three_numbers : Nat.gcd 4557 (Nat.gcd 1953 5115) = 93 := by
  sorry

end gcd_of_three_numbers_l3360_336002


namespace largest_number_with_equal_quotient_and_remainder_l3360_336063

theorem largest_number_with_equal_quotient_and_remainder :
  ∀ (A B C : ℕ),
    A = 8 * B + C →
    B = C →
    0 ≤ C ∧ C < 8 →
    A ≤ 63 ∧ (∃ (A' : ℕ), A' = 63 ∧ ∃ (B' C' : ℕ), A' = 8 * B' + C' ∧ B' = C' ∧ 0 ≤ C' ∧ C' < 8) :=
by sorry

end largest_number_with_equal_quotient_and_remainder_l3360_336063


namespace square_perimeter_from_area_l3360_336016

-- Define a square with a given area
def Square (area : ℝ) := {side : ℝ // side^2 = area}

-- Define the perimeter of a square
def perimeter (s : Square area) : ℝ := 4 * s.val

-- Theorem statement
theorem square_perimeter_from_area (s : Square 225) : 
  perimeter s = 60 := by sorry

end square_perimeter_from_area_l3360_336016


namespace range_of_y_over_x_l3360_336006

theorem range_of_y_over_x (x y : ℝ) (h : (x - 2)^2 + y^2 = 3) :
  ∃ (k : ℝ), y / x = k ∧ -Real.sqrt 3 ≤ k ∧ k ≤ Real.sqrt 3 ∧
  (∀ (m : ℝ), y / x = m → -Real.sqrt 3 ≤ m ∧ m ≤ Real.sqrt 3) :=
sorry

end range_of_y_over_x_l3360_336006


namespace polynomial_sum_l3360_336069

theorem polynomial_sum (d a b c e : ℤ) (h : d ≠ 0) :
  (10 * d + 15 + 12 * d^2 + 2 * d^3) + (4 * d - 3 + 2 * d^2) = a * d^3 + b * d^2 + c * d + e →
  a + b + c + e = 42 := by
  sorry

end polynomial_sum_l3360_336069


namespace pond_water_volume_l3360_336025

/-- Calculates the water volume in a pond after a given number of days -/
def water_volume (initial_volume : ℕ) (evaporation_rate : ℕ) (water_added : ℕ) (add_interval : ℕ) (days : ℕ) : ℕ :=
  initial_volume - evaporation_rate * days + (days / add_interval) * water_added

theorem pond_water_volume :
  water_volume 500 1 10 7 35 = 515 := by
  sorry

end pond_water_volume_l3360_336025


namespace exponent_rules_l3360_336029

theorem exponent_rules (a : ℝ) : 
  (a^2 * a^4 ≠ a^8) ∧ 
  ((-2*a^2)^3 ≠ -6*a^6) ∧ 
  (a^4 / a = a^3) ∧ 
  (2*a + 3*a ≠ 5*a^2) := by
sorry

end exponent_rules_l3360_336029


namespace line_equation_proof_l3360_336079

/-- A line in the xy-plane passing through two points -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a point lies on a line -/
def Line.contains (l : Line) (x y : ℝ) : Prop :=
  l.a * x + l.b * y + l.c = 0

/-- The line passing through (2,3) and (4,4) -/
def line_through_points : Line :=
  { a := 1, b := 8, c := -26 }

theorem line_equation_proof :
  (line_through_points.contains 2 3) ∧
  (line_through_points.contains 4 4) :=
by sorry

end line_equation_proof_l3360_336079


namespace abs_and_reciprocal_l3360_336097

theorem abs_and_reciprocal :
  (abs (-9 : ℝ) = 9) ∧ (((-3 : ℝ)⁻¹) = -1/3) := by
  sorry

end abs_and_reciprocal_l3360_336097


namespace red_bacon_bits_count_l3360_336080

def salad_problem (mushrooms cherry_tomatoes pickles bacon_bits red_bacon_bits : ℕ) : Prop :=
  mushrooms = 3 ∧
  cherry_tomatoes = 2 * mushrooms ∧
  pickles = 4 * cherry_tomatoes ∧
  bacon_bits = 4 * pickles ∧
  red_bacon_bits = bacon_bits / 3

theorem red_bacon_bits_count : ∃ (mushrooms cherry_tomatoes pickles bacon_bits red_bacon_bits : ℕ),
  salad_problem mushrooms cherry_tomatoes pickles bacon_bits red_bacon_bits ∧ red_bacon_bits = 32 :=
by
  sorry

end red_bacon_bits_count_l3360_336080


namespace even_sum_probability_l3360_336095

-- Define the properties of the wheels
def first_wheel_sections : ℕ := 5
def first_wheel_even_sections : ℕ := 2
def first_wheel_odd_sections : ℕ := 3

def second_wheel_sections : ℕ := 4
def second_wheel_even_sections : ℕ := 1
def second_wheel_odd_sections : ℕ := 2
def second_wheel_special_sections : ℕ := 1

-- Define the probability of getting an even sum
def prob_even_sum : ℚ := 1/2

-- Theorem statement
theorem even_sum_probability :
  let p_even_first : ℚ := first_wheel_even_sections / first_wheel_sections
  let p_odd_first : ℚ := first_wheel_odd_sections / first_wheel_sections
  let p_even_second : ℚ := second_wheel_even_sections / second_wheel_sections
  let p_odd_second : ℚ := second_wheel_odd_sections / second_wheel_sections
  let p_special_second : ℚ := second_wheel_special_sections / second_wheel_sections
  
  -- Probability of both numbers being even (including special section effect)
  let p_both_even : ℚ := p_even_first * p_even_second + p_even_first * p_special_second
  
  -- Probability of both numbers being odd
  let p_both_odd : ℚ := p_odd_first * p_odd_second
  
  -- Total probability of an even sum
  p_both_even + p_both_odd = prob_even_sum :=
by sorry

end even_sum_probability_l3360_336095


namespace cubic_tangent_max_l3360_336064

/-- A cubic function with real coefficients -/
def f (a b : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + b*x

/-- The derivative of f with respect to x -/
def f' (a b : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x + b

theorem cubic_tangent_max (a b m : ℝ) (hm : m ≠ 0) :
  f' a b m = 0 ∧                   -- Tangent condition (derivative = 0 at x = m)
  f a b m = 0 ∧                    -- Tangent condition (f(m) = 0)
  (∃ x, f a b x = (1/2 : ℝ)) ∧     -- Maximum value condition
  (∀ x, f a b x ≤ (1/2 : ℝ)) →     -- Maximum value condition
  m = (3/2 : ℝ) := by
sorry

end cubic_tangent_max_l3360_336064


namespace number_of_routes_l3360_336045

-- Define the cities
inductive City : Type
| A | B | C | D | F

-- Define the roads
inductive Road : Type
| AB | AD | AF | BC | BD | CD | DF

-- Define a route as a list of roads
def Route := List Road

-- Function to check if a route is valid (uses each road exactly once and starts at A and ends at B)
def isValidRoute (r : Route) : Prop := sorry

-- Function to count the number of valid routes
def countValidRoutes : Nat := sorry

-- Theorem to prove
theorem number_of_routes : countValidRoutes = 16 := by sorry

end number_of_routes_l3360_336045


namespace christinas_speed_limit_l3360_336014

def total_distance : ℝ := 210
def friend_driving_time : ℝ := 3
def friend_speed_limit : ℝ := 40
def christina_driving_time : ℝ := 3  -- 180 minutes converted to hours

theorem christinas_speed_limit :
  ∃ (christina_speed : ℝ),
    christina_speed * christina_driving_time + 
    friend_speed_limit * friend_driving_time = total_distance ∧
    christina_speed = 30 := by
  sorry

end christinas_speed_limit_l3360_336014


namespace naH_required_for_h2O_l3360_336048

-- Define the molecules and their molar ratios in the reactions
structure Reaction :=
  (naH : ℚ) (h2O : ℚ) (naOH : ℚ) (h2 : ℚ)

-- Define the first step reaction
def firstStepReaction : Reaction :=
  { naH := 1, h2O := 1, naOH := 1, h2 := 1 }

-- Theorem stating that 1 mole of NaH is required to react with 1 mole of H2O
theorem naH_required_for_h2O :
  firstStepReaction.naH = firstStepReaction.h2O := by sorry

end naH_required_for_h2O_l3360_336048


namespace cafeteria_seats_unseated_fraction_l3360_336085

theorem cafeteria_seats_unseated_fraction :
  let total_tables : ℕ := 15
  let seats_per_table : ℕ := 10
  let seats_taken : ℕ := 135
  let total_seats : ℕ := total_tables * seats_per_table
  let seats_unseated : ℕ := total_seats - seats_taken
  (seats_unseated : ℚ) / total_seats = 1 / 10 := by
sorry

end cafeteria_seats_unseated_fraction_l3360_336085


namespace parallel_vectors_x_value_l3360_336092

/-- Two vectors are parallel if their components are proportional -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

/-- Given two vectors a and b, where a = (3,1) and b = (x,-1), 
    if a is parallel to b, then x = -3 -/
theorem parallel_vectors_x_value :
  ∀ x : ℝ, 
  let a : ℝ × ℝ := (3, 1)
  let b : ℝ × ℝ := (x, -1)
  are_parallel a b → x = -3 := by
sorry

end parallel_vectors_x_value_l3360_336092


namespace exterior_angle_square_octagon_l3360_336053

-- Define the necessary structures
structure Polygon :=
  (sides : ℕ)

-- Define the square and octagon
def square : Polygon := ⟨4⟩
def octagon : Polygon := ⟨8⟩

-- Define the function to calculate interior angle of a regular polygon
def interior_angle (p : Polygon) : ℚ :=
  180 * (p.sides - 2) / p.sides

-- Define the theorem
theorem exterior_angle_square_octagon :
  let octagon_interior_angle := interior_angle octagon
  let square_interior_angle := 90
  360 - octagon_interior_angle - square_interior_angle = 135 := by sorry

end exterior_angle_square_octagon_l3360_336053


namespace max_stable_angle_l3360_336009

/-- A sign consisting of two uniform legs attached by a frictionless hinge -/
structure Sign where
  μ : ℝ  -- coefficient of friction between the ground and the legs
  θ : ℝ  -- angle between the legs

/-- The condition for the sign to be in equilibrium -/
def is_stable (s : Sign) : Prop :=
  Real.tan (s.θ / 2) = 2 * s.μ

/-- Theorem stating the maximum angle for stability -/
theorem max_stable_angle (s : Sign) :
  is_stable s ↔ s.θ = Real.arctan (2 * s.μ) * 2 :=
sorry

end max_stable_angle_l3360_336009


namespace fast_food_order_l3360_336023

/-- A problem about friends ordering fast food --/
theorem fast_food_order (num_friends : ℕ) (hamburger_cost : ℚ) 
  (fries_sets : ℕ) (fries_cost : ℚ) (soda_cups : ℕ) (soda_cost : ℚ)
  (spaghetti_platters : ℕ) (spaghetti_cost : ℚ) (individual_payment : ℚ) :
  num_friends = 5 →
  hamburger_cost = 3 →
  fries_sets = 4 →
  fries_cost = 6/5 →
  soda_cups = 5 →
  soda_cost = 1/2 →
  spaghetti_platters = 1 →
  spaghetti_cost = 27/10 →
  individual_payment = 5 →
  ∃ (num_hamburgers : ℕ), 
    num_hamburgers * hamburger_cost + 
    fries_sets * fries_cost + 
    soda_cups * soda_cost + 
    spaghetti_platters * spaghetti_cost = 
    num_friends * individual_payment ∧
    num_hamburgers = 5 := by
  sorry


end fast_food_order_l3360_336023


namespace polynomial_root_implies_k_l3360_336071

theorem polynomial_root_implies_k (k : ℝ) : 
  (3 : ℝ)^3 + k * 3 - 18 = 0 → k = -3 := by sorry

end polynomial_root_implies_k_l3360_336071


namespace triangle_area_with_consecutive_integer_sides_and_height_l3360_336015

theorem triangle_area_with_consecutive_integer_sides_and_height :
  ∀ (a b c h : ℕ),
    a + 1 = b →
    b + 1 = c →
    c + 1 = h →
    (1 / 2 : ℚ) * b * h = 84 := by
  sorry

end triangle_area_with_consecutive_integer_sides_and_height_l3360_336015


namespace number_difference_l3360_336026

theorem number_difference (A B : ℕ) (h1 : A + B = 1812) (h2 : A = 7 * B + 4) : A - B = 1360 := by
  sorry

end number_difference_l3360_336026


namespace point_relationship_l3360_336091

/-- Prove that for points A(-1/2, m) and B(2, n) lying on the line y = 3x + b, m < n. -/
theorem point_relationship (m n b : ℝ) : 
  ((-1/2 : ℝ), m) ∈ {(x, y) | y = 3*x + b} →
  ((2 : ℝ), n) ∈ {(x, y) | y = 3*x + b} →
  m < n := by sorry

end point_relationship_l3360_336091


namespace tv_screen_length_tv_screen_length_approx_l3360_336033

theorem tv_screen_length (diagonal : ℝ) (ratio_length_height : ℚ) : ℝ :=
  let length := Real.sqrt ((ratio_length_height ^ 2 * diagonal ^ 2) / (1 + ratio_length_height ^ 2))
  length

theorem tv_screen_length_approx :
  ∃ ε > 0, abs (tv_screen_length 27 (4/3) - 21.6) < ε :=
sorry

end tv_screen_length_tv_screen_length_approx_l3360_336033


namespace min_reciprocal_sum_l3360_336072

theorem min_reciprocal_sum (m n : ℝ) (hm : m > 0) (hn : n > 0) (h_sum : m + n = 2) :
  (1 / m + 1 / n) ≥ 2 ∧ ∃ (m₀ n₀ : ℝ), m₀ > 0 ∧ n₀ > 0 ∧ m₀ + n₀ = 2 ∧ 1 / m₀ + 1 / n₀ = 2 :=
by sorry

end min_reciprocal_sum_l3360_336072


namespace meteorological_forecast_probability_l3360_336038

theorem meteorological_forecast_probability 
  (p q : ℝ) 
  (hp : 0 ≤ p ∧ p ≤ 1) 
  (hq : 0 ≤ q ∧ q ≤ 1) : 
  (p * (1 - q) : ℝ) = 
  (p : ℝ) * (1 - (q : ℝ)) := by
sorry

end meteorological_forecast_probability_l3360_336038


namespace vector_magnitude_proof_l3360_336034

theorem vector_magnitude_proof (a b : ℝ × ℝ) :
  let angle := Real.pi / 3
  (a.1^2 + a.2^2 = 1) →
  (b.1^2 + b.2^2 = 4) →
  (a.1 * b.1 + a.2 * b.2 = 2 * Real.cos angle) →
  ((2*a.1 - b.1)^2 + (2*a.2 - b.2)^2 = 4) :=
by sorry

end vector_magnitude_proof_l3360_336034


namespace arrangement_of_six_objects_l3360_336057

theorem arrangement_of_six_objects (n : ℕ) (h : n = 6) : 
  Nat.factorial n = 720 :=
by
  sorry

end arrangement_of_six_objects_l3360_336057


namespace variables_related_probability_l3360_336037

/-- The k-value obtained from a 2×2 contingency table -/
def k : ℝ := 4.073

/-- The probability that k^2 is greater than or equal to 3.841 -/
def p_3841 : ℝ := 0.05

/-- The probability that k^2 is greater than or equal to 5.024 -/
def p_5024 : ℝ := 0.025

/-- The theorem stating the probability of two variables being related -/
theorem variables_related_probability : ℝ := by
  sorry

end variables_related_probability_l3360_336037


namespace first_term_is_two_l3360_336036

/-- An increasing arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  increasing : ∀ n, a n < a (n + 1)
  arithmetic : ∃ d : ℝ, ∀ n, a (n + 1) - a n = d
  sum_first_three : a 1 + a 2 + a 3 = 12
  product_first_three : a 1 * a 2 * a 3 = 48

/-- The first term of the arithmetic sequence is 2 -/
theorem first_term_is_two (seq : ArithmeticSequence) : seq.a 1 = 2 := by
  sorry

end first_term_is_two_l3360_336036


namespace meeting_time_on_circular_track_l3360_336040

/-- The time taken for two people to meet on a circular track -/
theorem meeting_time_on_circular_track 
  (track_circumference : ℝ)
  (speed1 : ℝ)
  (speed2 : ℝ)
  (h1 : track_circumference = 528)
  (h2 : speed1 = 4.5)
  (h3 : speed2 = 3.75) :
  (track_circumference / ((speed1 + speed2) * 1000 / 60)) = 3.84 := by
  sorry

end meeting_time_on_circular_track_l3360_336040


namespace six_balls_four_boxes_l3360_336043

/-- The number of ways to distribute n distinguishable balls into k indistinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ :=
  sorry

/-- Theorem stating that there are 187 ways to distribute 6 distinguishable balls into 4 indistinguishable boxes -/
theorem six_balls_four_boxes : distribute_balls 6 4 = 187 := by
  sorry

end six_balls_four_boxes_l3360_336043


namespace quadratic_roots_relation_l3360_336065

theorem quadratic_roots_relation (p q : ℝ) : 
  (∃ a b : ℝ, 
    (2 * a^2 - 6 * a + 1 = 0) ∧ 
    (2 * b^2 - 6 * b + 1 = 0) ∧
    ((3 * a - 1)^2 + p * (3 * a - 1) + q = 0) ∧
    ((3 * b - 1)^2 + p * (3 * b - 1) + q = 0)) →
  q = -0.5 := by
sorry

end quadratic_roots_relation_l3360_336065


namespace oranges_discarded_per_day_l3360_336066

theorem oranges_discarded_per_day 
  (harvest_per_day : ℕ) 
  (days : ℕ) 
  (remaining_sacks : ℕ) 
  (h1 : harvest_per_day = 74)
  (h2 : days = 51)
  (h3 : remaining_sacks = 153) :
  (harvest_per_day * days - remaining_sacks) / days = 71 := by
  sorry

end oranges_discarded_per_day_l3360_336066
