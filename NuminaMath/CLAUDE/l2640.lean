import Mathlib

namespace sacks_per_day_l2640_264034

/-- Given a harvest of oranges that lasts for a certain number of days and produces a total number of sacks, this theorem proves the number of sacks harvested per day. -/
theorem sacks_per_day (total_sacks : ℕ) (harvest_days : ℕ) (h1 : total_sacks = 56) (h2 : harvest_days = 4) :
  total_sacks / harvest_days = 14 := by
  sorry

end sacks_per_day_l2640_264034


namespace locus_of_centers_l2640_264019

-- Define the two circles
def C₁ (x y : ℝ) : Prop := x^2 + y^2 = 4
def C₂ (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 9

-- Define the property of being externally tangent to C₁ and internally tangent to C₂
def externally_internally_tangent (a b r : ℝ) : Prop :=
  ∃ (x y : ℝ), C₁ x y ∧ (x - a)^2 + (y - b)^2 = (r + 2)^2 ∧
                C₂ x y ∧ (x - a)^2 + (y - b)^2 = (3 - r)^2

-- State the theorem
theorem locus_of_centers : 
  ∀ (a b : ℝ), (∃ r : ℝ, externally_internally_tangent a b r) ↔ 
  16 * a^2 + 25 * b^2 - 48 * a - 64 = 0 := by sorry

end locus_of_centers_l2640_264019


namespace arithmetic_sequence_min_sum_l2640_264075

/-- An arithmetic sequence with a_4 = -14 and common difference d = 3 -/
def ArithmeticSequence (n : ℕ) : ℤ := 3*n - 26

/-- The sum of the first n terms of the arithmetic sequence -/
def SequenceSum (n : ℕ) : ℤ := n * (ArithmeticSequence 1 + ArithmeticSequence n) / 2

theorem arithmetic_sequence_min_sum :
  (∀ m : ℕ, SequenceSum m ≥ SequenceSum 8) ∧
  SequenceSum 8 = -100 := by
  sorry

end arithmetic_sequence_min_sum_l2640_264075


namespace cylindrical_containers_radius_l2640_264070

theorem cylindrical_containers_radius (h : ℝ) (r : ℝ) :
  h > 0 →
  (π * (8^2) * (4 * h) = π * r^2 * h) →
  r = 16 := by
sorry

end cylindrical_containers_radius_l2640_264070


namespace fourth_month_sale_l2640_264045

def average_sale : ℝ := 2500
def month1_sale : ℝ := 2435
def month2_sale : ℝ := 2920
def month3_sale : ℝ := 2855
def month5_sale : ℝ := 2560
def month6_sale : ℝ := 1000

theorem fourth_month_sale (x : ℝ) : 
  (month1_sale + month2_sale + month3_sale + x + month5_sale + month6_sale) / 6 = average_sale →
  x = 3230 := by
sorry

end fourth_month_sale_l2640_264045


namespace willey_farm_capital_l2640_264088

def total_land : ℕ := 4500
def corn_cost : ℕ := 42
def wheat_cost : ℕ := 35
def wheat_acres : ℕ := 3400

theorem willey_farm_capital :
  let corn_acres := total_land - wheat_acres
  let wheat_total_cost := wheat_cost * wheat_acres
  let corn_total_cost := corn_cost * corn_acres
  wheat_total_cost + corn_total_cost = 165200 := by sorry

end willey_farm_capital_l2640_264088


namespace problem_solution_l2640_264048

theorem problem_solution (x y : ℝ) (h1 : y > x) (h2 : x > 0) (h3 : x / y + y / x = 8) :
  (x + y) / (x - y) = -Real.sqrt (5 / 3) := by
  sorry

end problem_solution_l2640_264048


namespace inverse_73_mod_74_l2640_264024

theorem inverse_73_mod_74 : ∃ x : ℕ, 0 ≤ x ∧ x ≤ 73 ∧ (73 * x) % 74 = 1 := by
  sorry

end inverse_73_mod_74_l2640_264024


namespace polynomial_division_theorem_l2640_264068

theorem polynomial_division_theorem (x : ℝ) : 
  x^6 + 8 = (x - 1) * (x^5 + x^4 + x^3 + x^2 + x + 1) + 9 := by
  sorry

end polynomial_division_theorem_l2640_264068


namespace system_solution_l2640_264077

theorem system_solution : ∃! (x y : ℝ), 
  (2 * x + Real.sqrt (2 * x + 3 * y) - 3 * y = 5) ∧ 
  (4 * x^2 + 2 * x + 3 * y - 9 * y^2 = 32) ∧ 
  (x = 17/4) ∧ 
  (y = 5/2) := by
  sorry

end system_solution_l2640_264077


namespace train_length_l2640_264041

/-- The length of a train given its speed and time to cross an electric pole. -/
theorem train_length (speed_kmh : ℝ) (time_sec : ℝ) (length : ℝ) : 
  speed_kmh = 72 → time_sec = 15 → length = (speed_kmh * 1000 / 3600) * time_sec → length = 300 := by
  sorry

#check train_length

end train_length_l2640_264041


namespace sum_of_last_two_digits_9_20_plus_11_20_l2640_264049

theorem sum_of_last_two_digits_9_20_plus_11_20 :
  (9^20 + 11^20) % 100 = 1 := by
  sorry

end sum_of_last_two_digits_9_20_plus_11_20_l2640_264049


namespace girls_in_school_l2640_264017

theorem girls_in_school (total_people boys teachers : ℕ)
  (h1 : total_people = 1396)
  (h2 : boys = 309)
  (h3 : teachers = 772) :
  total_people - boys - teachers = 315 := by
sorry

end girls_in_school_l2640_264017


namespace expression_evaluation_l2640_264008

theorem expression_evaluation : 
  let c : ℕ := 4
  (c^c - 2*c*(c-2)^c + c^2)^c = 431441456 := by
  sorry

end expression_evaluation_l2640_264008


namespace melanie_grew_more_turnips_l2640_264036

/-- The number of turnips Melanie grew -/
def melanie_turnips : ℕ := 139

/-- The number of turnips Benny grew -/
def benny_turnips : ℕ := 113

/-- The difference in turnips grown between Melanie and Benny -/
def turnip_difference : ℕ := melanie_turnips - benny_turnips

theorem melanie_grew_more_turnips : turnip_difference = 26 := by
  sorry

end melanie_grew_more_turnips_l2640_264036


namespace biathlon_distance_l2640_264030

/-- Biathlon problem -/
theorem biathlon_distance (total_distance : ℝ) (run_velocity : ℝ) (bike_velocity : ℝ) (total_time : ℝ)
  (h1 : total_distance = 155)
  (h2 : run_velocity = 10)
  (h3 : bike_velocity = 29)
  (h4 : total_time = 6) :
  ∃ (bike_distance : ℝ), 
    bike_distance + (total_distance - bike_distance) = total_distance ∧
    bike_distance / bike_velocity + (total_distance - bike_distance) / run_velocity = total_time ∧
    bike_distance = 145 := by
  sorry

end biathlon_distance_l2640_264030


namespace congruence_systems_solutions_l2640_264096

theorem congruence_systems_solutions :
  (∃ x : ℤ, x % 7 = 3 ∧ (6 * x) % 8 = 10) ∧
  (∀ x : ℤ, x % 7 = 3 ∧ (6 * x) % 8 = 10 → x % 56 = 3 ∨ x % 56 = 31) ∧
  (¬ ∃ x : ℤ, (3 * x) % 10 = 1 ∧ (4 * x) % 15 = 7) :=
by sorry

end congruence_systems_solutions_l2640_264096


namespace system_solution_unique_l2640_264052

theorem system_solution_unique : 
  ∃! (x y : ℝ), 2 * x - y = 3 ∧ 3 * x + 2 * y = 8 :=
by
  -- The proof goes here
  sorry

end system_solution_unique_l2640_264052


namespace problem_solution_l2640_264037

def A : Set ℝ := {x | x^2 - 2*x - 15 > 0}
def B : Set ℝ := {x | x - 6 < 0}

theorem problem_solution :
  (∀ m : ℝ, m ∈ A ↔ (m < -3 ∨ m > 5)) ∧
  (∀ m : ℝ, (m ∈ A ∨ m ∈ B) ∧ (m ∈ A ∧ m ∈ B) ↔ (m < -3 ∨ (5 < m ∧ m < 6))) := by
  sorry

end problem_solution_l2640_264037


namespace xyz_inequality_l2640_264092

theorem xyz_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h : x * y * z ≥ x * y + y * z + z * x) : x * y * z ≥ 3 * (x + y + z) := by
  sorry

end xyz_inequality_l2640_264092


namespace committee_choices_theorem_l2640_264078

/-- The number of ways to choose a committee with constraints -/
def committee_choices (total : ℕ) (women : ℕ) (men : ℕ) (committee_size : ℕ) (min_women : ℕ) : ℕ :=
  (Nat.choose women min_women) * (Nat.choose (total - min_women) (committee_size - min_women))

/-- Theorem: The number of ways to choose a 5-person committee from a club of 12 people
    (7 women and 5 men), where the committee must include at least 2 women, is 2520 -/
theorem committee_choices_theorem :
  committee_choices 12 7 5 5 2 = 2520 := by
  sorry

#eval committee_choices 12 7 5 5 2

end committee_choices_theorem_l2640_264078


namespace union_of_sets_l2640_264058

theorem union_of_sets : 
  let M : Set Nat := {1, 2, 5}
  let N : Set Nat := {1, 3, 5, 7}
  M ∪ N = {1, 2, 3, 5, 7} := by
sorry

end union_of_sets_l2640_264058


namespace aa_existence_l2640_264084

theorem aa_existence : ∃ aa : ℕ, 1 ≤ aa ∧ aa ≤ 9 ∧ (7 * aa^3) % 100 ≥ 10 ∧ (7 * aa^3) % 100 < 20 :=
by sorry

end aa_existence_l2640_264084


namespace gcd_property_l2640_264001

theorem gcd_property (a b : ℤ) : Int.gcd a b = 1 → Int.gcd (2*a + b) (a*(a + b)) = 1 := by
  sorry

end gcd_property_l2640_264001


namespace find_m_l2640_264060

-- Define the inequality
def inequality (x m : ℝ) : Prop := -1/2 * x^2 + 2*x > -m*x

-- Define the solution set
def solution_set (m : ℝ) : Set ℝ := {x : ℝ | 0 < x ∧ x < 2}

-- Theorem statement
theorem find_m : 
  ∀ m : ℝ, (∀ x : ℝ, inequality x m ↔ x ∈ solution_set m) → m = -1 :=
sorry

end find_m_l2640_264060


namespace angle_D_measure_l2640_264027

-- Define a convex hexagon
structure ConvexHexagon where
  A : ℝ
  B : ℝ
  C : ℝ
  D : ℝ
  E : ℝ
  F : ℝ
  sum_of_angles : A + B + C + D + E + F = 720
  positive_angles : 0 < A ∧ 0 < B ∧ 0 < C ∧ 0 < D ∧ 0 < E ∧ 0 < F

-- Define the specific conditions of the hexagon
def SpecialHexagon (h : ConvexHexagon) : Prop :=
  h.A = h.B ∧ h.B = h.C ∧   -- Angles A, B, and C are congruent
  h.D = h.E ∧ h.E = h.F ∧   -- Angles D, E, and F are congruent
  h.A + 30 = h.D            -- Angle A is 30° less than angle D

-- State the theorem
theorem angle_D_measure (h : ConvexHexagon) (special : SpecialHexagon h) : h.D = 135 := by
  sorry

end angle_D_measure_l2640_264027


namespace quadratic_roots_condition_l2640_264056

theorem quadratic_roots_condition (c : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + 2*x₁ + 4*c = 0 ∧ x₂^2 + 2*x₂ + 4*c = 0) →
  c < (1/4 : ℝ) :=
by sorry

end quadratic_roots_condition_l2640_264056


namespace discount_calculation_l2640_264032

/-- Calculates the discount amount given the cost of a suit, shoes, and the final payment -/
theorem discount_calculation (suit_cost shoes_cost final_payment : ℕ) :
  suit_cost = 430 →
  shoes_cost = 190 →
  final_payment = 520 →
  suit_cost + shoes_cost - final_payment = 100 := by
  sorry

end discount_calculation_l2640_264032


namespace parabola_axis_of_symmetry_l2640_264061

/-- The axis of symmetry of a parabola y=(x-h)^2 is x=h -/
theorem parabola_axis_of_symmetry (h : ℝ) :
  let f : ℝ → ℝ := fun x ↦ (x - h)^2
  ∃! a : ℝ, ∀ x y : ℝ, f x = f y → (x + y) / 2 = a :=
by sorry

end parabola_axis_of_symmetry_l2640_264061


namespace divisible_by_10101010101_has_at_least_6_nonzero_digits_l2640_264094

/-- The number of non-zero digits in the decimal representation of a natural number -/
def num_nonzero_digits (n : ℕ) : ℕ := sorry

/-- Theorem: Any natural number divisible by 10101010101 has at least 6 non-zero digits -/
theorem divisible_by_10101010101_has_at_least_6_nonzero_digits (k : ℕ) :
  k % 10101010101 = 0 → num_nonzero_digits k ≥ 6 := by
  sorry

end divisible_by_10101010101_has_at_least_6_nonzero_digits_l2640_264094


namespace expected_sixes_is_half_l2640_264013

/-- The probability of rolling a 6 on a standard die -/
def prob_six : ℚ := 1 / 6

/-- The probability of not rolling a 6 on a standard die -/
def prob_not_six : ℚ := 1 - prob_six

/-- The number of dice rolled -/
def num_dice : ℕ := 3

/-- The expected number of 6's when rolling three standard dice -/
def expected_sixes : ℚ := 
  (0 : ℚ) * (prob_not_six ^ num_dice) +
  (1 : ℚ) * (num_dice.choose 1 * prob_six * prob_not_six^2) +
  (2 : ℚ) * (num_dice.choose 2 * prob_six^2 * prob_not_six) +
  (3 : ℚ) * (prob_six ^ num_dice)

theorem expected_sixes_is_half : expected_sixes = 1 / 2 := by
  sorry


end expected_sixes_is_half_l2640_264013


namespace andy_weight_change_l2640_264072

/-- Calculates Andy's weight change over the year -/
theorem andy_weight_change (initial_weight : ℝ) (weight_gain : ℝ) (months : ℕ) : 
  initial_weight = 156 →
  weight_gain = 36 →
  months = 3 →
  initial_weight - (initial_weight + weight_gain) * (1 - 1/8)^months = 36 := by
  sorry

#check andy_weight_change

end andy_weight_change_l2640_264072


namespace sum_of_solutions_quadratic_l2640_264009

theorem sum_of_solutions_quadratic (x : ℝ) : 
  (2 * x^2 - 8 * x - 10 = 5 * x + 20) → 
  ∃ (y : ℝ), (2 * y^2 - 8 * y - 10 = 5 * y + 20) ∧ (x + y = 13/2) :=
by sorry

end sum_of_solutions_quadratic_l2640_264009


namespace C_on_or_inside_circle_O_l2640_264066

-- Define the circle O and points A, B, C
variable (O : ℝ × ℝ) (A B C : ℝ × ℝ)

-- Define the radius of circle O
def radius_O : ℝ := 10

-- Define that A is on circle O
def A_on_circle_O : (A.1 - O.1)^2 + (A.2 - O.2)^2 = radius_O^2 := by sorry

-- Define B as the midpoint of OA
def B_midpoint_OA : B = ((O.1 + A.1)/2, (O.2 + A.2)/2) := by sorry

-- Define the distance between B and C
def BC_distance : (B.1 - C.1)^2 + (B.2 - C.2)^2 = 5^2 := by sorry

-- Theorem to prove
theorem C_on_or_inside_circle_O :
  (C.1 - O.1)^2 + (C.2 - O.2)^2 ≤ radius_O^2 := by sorry

end C_on_or_inside_circle_O_l2640_264066


namespace abc_inequality_l2640_264012

theorem abc_inequality : 
  let a : ℝ := (2/5)^(3/5)
  let b : ℝ := (2/5)^(2/5)
  let c : ℝ := (3/5)^(2/5)
  a < b ∧ b < c := by sorry

end abc_inequality_l2640_264012


namespace expression_equals_zero_l2640_264040

theorem expression_equals_zero :
  (π - 2023) ^ (0 : ℝ) - |1 - Real.sqrt 2| + 2 * Real.cos (π / 4) - (1 / 2) ^ (-1 : ℝ) = 0 := by
  sorry

end expression_equals_zero_l2640_264040


namespace class_size_proof_l2640_264031

theorem class_size_proof (total : ℕ) : 
  (1 / 4 : ℚ) * total = total - ((3 / 4 : ℚ) * total) →
  (1 / 3 : ℚ) * ((3 / 4 : ℚ) * total) = ((3 / 4 : ℚ) * total) - 10 →
  10 = (2 / 3 : ℚ) * ((3 / 4 : ℚ) * total) →
  total = 20 := by
  sorry

end class_size_proof_l2640_264031


namespace largest_sum_and_simplification_l2640_264015

theorem largest_sum_and_simplification : 
  let sums : List ℚ := [1/5 + 1/6, 1/5 + 1/7, 1/5 + 1/3, 1/5 + 1/8, 1/5 + 1/9]
  (∀ x ∈ sums, x ≤ (1/5 + 1/3)) ∧ (1/5 + 1/3 = 8/15) := by
  sorry

end largest_sum_and_simplification_l2640_264015


namespace john_quiz_goal_l2640_264000

theorem john_quiz_goal (total_quizzes : ℕ) (goal_percentage : ℚ) (completed_quizzes : ℕ) (current_as : ℕ) :
  total_quizzes = 60 →
  goal_percentage = 70 / 100 →
  completed_quizzes = 40 →
  current_as = 25 →
  ∃ (max_non_as : ℕ),
    max_non_as = 3 ∧
    (total_quizzes - completed_quizzes - max_non_as) + current_as ≥ ⌈(goal_percentage * total_quizzes : ℚ)⌉ ∧
    ∀ (n : ℕ), n > max_non_as →
      (total_quizzes - completed_quizzes - n) + current_as < ⌈(goal_percentage * total_quizzes : ℚ)⌉ :=
by sorry

end john_quiz_goal_l2640_264000


namespace eggs_per_student_l2640_264098

theorem eggs_per_student (total_eggs : ℕ) (num_students : ℕ) (eggs_per_student : ℕ)
  (h1 : total_eggs = 56)
  (h2 : num_students = 7)
  (h3 : total_eggs = num_students * eggs_per_student) :
  eggs_per_student = 8 := by
  sorry

end eggs_per_student_l2640_264098


namespace sum_coordinates_of_B_l2640_264005

/-- Given that M(6,8) is the midpoint of AB and A has coordinates (10,8), 
    prove that the sum of the coordinates of B is 10. -/
theorem sum_coordinates_of_B (A B M : ℝ × ℝ) : 
  M = (6, 8) → 
  A = (10, 8) → 
  M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) → 
  B.1 + B.2 = 10 := by
sorry

end sum_coordinates_of_B_l2640_264005


namespace charcoal_drawings_count_l2640_264029

/-- The total number of drawings Thomas has -/
def total_drawings : ℕ := 120

/-- The number of drawings made with colored pencils -/
def colored_pencil_drawings : ℕ := 35

/-- The number of drawings made with blending markers -/
def blending_marker_drawings : ℕ := 22

/-- The number of drawings made with pastels -/
def pastel_drawings : ℕ := 15

/-- The number of drawings made with watercolors -/
def watercolor_drawings : ℕ := 12

/-- The number of charcoal drawings -/
def charcoal_drawings : ℕ := total_drawings - (colored_pencil_drawings + blending_marker_drawings + pastel_drawings + watercolor_drawings)

theorem charcoal_drawings_count : charcoal_drawings = 36 := by
  sorry

end charcoal_drawings_count_l2640_264029


namespace sin_2x_minus_y_eq_neg_one_l2640_264079

theorem sin_2x_minus_y_eq_neg_one 
  (hx : x + Real.sin x * Real.cos x - 1 = 0)
  (hy : 2 * Real.cos y - 2 * y + Real.pi + 4 = 0) : 
  Real.sin (2 * x - y) = -1 := by
sorry

end sin_2x_minus_y_eq_neg_one_l2640_264079


namespace adult_meals_calculation_l2640_264026

/-- Given a ratio of kids meals to adult meals and the number of kids meals sold,
    calculate the number of adult meals sold. -/
def adult_meals_sold (kids_ratio : ℕ) (adult_ratio : ℕ) (kids_meals : ℕ) : ℕ :=
  (adult_ratio * kids_meals) / kids_ratio

/-- Theorem stating that given the specific ratio and number of kids meals,
    the number of adult meals sold is 49. -/
theorem adult_meals_calculation :
  adult_meals_sold 10 7 70 = 49 := by
  sorry

#eval adult_meals_sold 10 7 70

end adult_meals_calculation_l2640_264026


namespace not_always_determinable_l2640_264023

/-- Represents a weight with a mass -/
structure Weight where
  mass : ℝ

/-- Represents a question about the order of three weights -/
structure Question where
  a : Weight
  b : Weight
  c : Weight

/-- The set of all possible permutations of five weights -/
def AllPermutations : Finset (List Weight) :=
  sorry

/-- The number of questions we can ask -/
def NumQuestions : ℕ := 9

/-- A function that simulates asking a question -/
def askQuestion (q : Question) (perm : List Weight) : Bool :=
  sorry

/-- The main theorem stating that it's not always possible to determine the exact order -/
theorem not_always_determinable (weights : Finset Weight) 
  (h : weights.card = 5) :
  ∃ (perm₁ perm₂ : List Weight),
    perm₁ ∈ AllPermutations ∧ 
    perm₂ ∈ AllPermutations ∧ 
    perm₁ ≠ perm₂ ∧
    ∀ (questions : Finset Question),
      questions.card ≤ NumQuestions →
      ∀ (q : Question),
        q ∈ questions →
        askQuestion q perm₁ = askQuestion q perm₂ :=
  sorry

end not_always_determinable_l2640_264023


namespace ratio_equality_l2640_264095

theorem ratio_equality (x y z : ℝ) 
  (h_pos : x > 0 ∧ y > 0 ∧ z > 0)
  (h_diff : x ≠ y ∧ y ≠ z ∧ x ≠ z)
  (h_eq : y / (2*x - z) = (x + y) / (2*z) ∧ (x + y) / (2*z) = x / y) :
  x / y = 3 := by
sorry

end ratio_equality_l2640_264095


namespace vector_norm_difference_l2640_264086

theorem vector_norm_difference (a b : ℝ × ℝ) 
  (h1 : a + b = (2, 3)) 
  (h2 : a - b = (-2, 1)) : 
  ‖a‖^2 - ‖b‖^2 = -1 := by sorry

end vector_norm_difference_l2640_264086


namespace triangle_side_length_l2640_264010

theorem triangle_side_length 
  (A B C : ℝ) 
  (a b c : ℝ) 
  (h_area : (1/2) * a * c * Real.sin B = Real.sqrt 3)
  (h_angle : B = π/3)
  (h_sides : a^2 + c^2 = 3*a*c) :
  b = 2 * Real.sqrt 2 := by
sorry

end triangle_side_length_l2640_264010


namespace cookie_difference_l2640_264035

/-- The number of sweet cookies Paco had -/
def total_sweet : ℕ := 40

/-- The number of salty cookies Paco had -/
def total_salty : ℕ := 25

/-- The number of sweet cookies Paco ate -/
def sweet_eaten : ℕ := total_sweet * 2

/-- The number of salty cookies Paco ate -/
noncomputable def salty_eaten : ℕ := (total_salty * 5) / 3

theorem cookie_difference :
  (salty_eaten : ℤ) - sweet_eaten = -38 := by sorry

end cookie_difference_l2640_264035


namespace mateo_net_salary_proof_l2640_264076

/-- Calculate Mateo's net salary for a week with absences -/
def calculate_net_salary (regular_salary : ℝ) (absence_days : ℕ) : ℝ :=
  let absence_deduction := 
    if absence_days ≥ 1 then 0.01 * regular_salary else 0 +
    if absence_days ≥ 2 then 0.02 * regular_salary else 0 +
    if absence_days ≥ 3 then 0.03 * regular_salary else 0 +
    if absence_days ≥ 4 then 0.04 * regular_salary else 0
  let salary_after_absence := regular_salary - absence_deduction
  let income_tax := 0.07 * salary_after_absence
  salary_after_absence - income_tax

theorem mateo_net_salary_proof :
  let regular_salary : ℝ := 791
  let absence_days : ℕ := 4
  let net_salary := calculate_net_salary regular_salary absence_days
  ∃ ε > 0, |net_salary - 662.07| < ε ∧ ε < 0.01 :=
by sorry

end mateo_net_salary_proof_l2640_264076


namespace right_triangle_arctan_sum_l2640_264055

theorem right_triangle_arctan_sum (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) :
  a^2 + c^2 = b^2 →
  Real.arctan (a / (c + b)) + Real.arctan (c / (a + b)) = π / 4 := by
sorry

end right_triangle_arctan_sum_l2640_264055


namespace square_preserves_order_l2640_264002

theorem square_preserves_order (a b : ℝ) : a > b ∧ b > 0 → a^2 > b^2 := by
  sorry

end square_preserves_order_l2640_264002


namespace unit_vector_xy_plane_l2640_264082

theorem unit_vector_xy_plane (u : ℝ × ℝ × ℝ) : 
  let (x, y, z) := u
  (x^2 + y^2 = 1 ∧ z = 0) →  -- u is a unit vector in the xy-plane
  (x + 3*y = Real.sqrt 30 / 2) →  -- angle with (1, 3, 0) is 30°
  (3*x - y = Real.sqrt 20 / 2) →  -- angle with (3, -1, 0) is 45°
  x = (3 * Real.sqrt 20 + Real.sqrt 30) / 20 :=
by sorry

end unit_vector_xy_plane_l2640_264082


namespace pizza_slice_price_l2640_264081

theorem pizza_slice_price 
  (whole_pizza_price : ℝ)
  (slices_sold : ℕ)
  (whole_pizzas_sold : ℕ)
  (total_revenue : ℝ)
  (h1 : whole_pizza_price = 15)
  (h2 : slices_sold = 24)
  (h3 : whole_pizzas_sold = 3)
  (h4 : total_revenue = 117) :
  ∃ (price_per_slice : ℝ), 
    price_per_slice * slices_sold + whole_pizza_price * whole_pizzas_sold = total_revenue ∧ 
    price_per_slice = 3 := by
  sorry

end pizza_slice_price_l2640_264081


namespace floor_area_less_than_ten_l2640_264059

/-- Represents a rectangular room -/
structure Room where
  length : ℝ
  width : ℝ
  height : ℝ

/-- The condition that the room's height is 3 meters -/
def height_is_three (r : Room) : Prop :=
  r.height = 3

/-- The condition that each wall's area is greater than the floor area -/
def walls_larger_than_floor (r : Room) : Prop :=
  r.length * r.height > r.length * r.width ∧ 
  r.width * r.height > r.length * r.width

/-- The theorem stating that under the given conditions, 
    the floor area must be less than 10 square meters -/
theorem floor_area_less_than_ten (r : Room) 
  (h1 : height_is_three r) 
  (h2 : walls_larger_than_floor r) : 
  r.length * r.width < 10 := by
  sorry


end floor_area_less_than_ten_l2640_264059


namespace closest_point_l2640_264097

def v (t : ℝ) : Fin 3 → ℝ := fun i => 
  match i with
  | 0 => 3 + 8*t
  | 1 => -1 + 2*t
  | 2 => -2 - 3*t

def a : Fin 3 → ℝ := fun i =>
  match i with
  | 0 => 1
  | 1 => 7
  | 2 => 1

def direction : Fin 3 → ℝ := fun i =>
  match i with
  | 0 => 8
  | 1 => 2
  | 2 => -3

theorem closest_point (t : ℝ) : 
  (∀ s : ℝ, ‖v t - a‖ ≤ ‖v s - a‖) ↔ t = -1/7 := by sorry

end closest_point_l2640_264097


namespace train_length_train_length_approx_200m_l2640_264038

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed_kmph : ℝ) (crossing_time_sec : ℝ) : ℝ :=
  let speed_mps := speed_kmph * 1000 / 3600
  speed_mps * crossing_time_sec

/-- Proof that a train traveling at 120 kmph crossing a pole in 6 seconds is approximately 200 meters long -/
theorem train_length_approx_200m :
  ∃ ε > 0, |train_length 120 6 - 200| < ε :=
sorry

end train_length_train_length_approx_200m_l2640_264038


namespace unique_valid_number_l2640_264074

def is_valid_number (n : ℕ) : Prop :=
  765400 ≤ n ∧ n ≤ 765499 ∧ n % 24 = 0

theorem unique_valid_number :
  ∃! n : ℕ, is_valid_number n ∧ n = 765455 :=
sorry

end unique_valid_number_l2640_264074


namespace product_of_terms_l2640_264051

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- The roots of the quadratic equation 2x^2 + 5x + 1 = 0 -/
def roots_of_equation (x y : ℝ) : Prop :=
  2 * x^2 + 5 * x + 1 = 0 ∧ 2 * y^2 + 5 * y + 1 = 0

theorem product_of_terms (a : ℕ → ℝ) :
  geometric_sequence a →
  roots_of_equation (a 1) (a 10) →
  a 4 * a 7 = 1/2 := by sorry

end product_of_terms_l2640_264051


namespace central_angle_unchanged_l2640_264003

theorem central_angle_unchanged (r₁ r₂ arc_length₁ arc_length₂ angle₁ angle₂ : ℝ) :
  r₁ > 0 →
  r₂ = 2 * r₁ →
  arc_length₂ = 2 * arc_length₁ →
  angle₁ = arc_length₁ / r₁ →
  angle₂ = arc_length₂ / r₂ →
  angle₂ = angle₁ :=
by sorry

end central_angle_unchanged_l2640_264003


namespace rectangle_area_l2640_264043

theorem rectangle_area (length width : ℝ) (h1 : length = Real.sqrt 6) (h2 : width = Real.sqrt 3) :
  length * width = 3 * Real.sqrt 2 := by
  sorry

end rectangle_area_l2640_264043


namespace garden_area_l2640_264018

/-- Proves that the area of a rectangular garden with length three times its width
    and width of 14 meters is 588 square meters. -/
theorem garden_area (width : ℝ) (length : ℝ) (area : ℝ) : 
  width = 14 →
  length = 3 * width →
  area = length * width →
  area = 588 :=
by sorry

end garden_area_l2640_264018


namespace seans_soda_purchase_l2640_264033

/-- The number of cans of soda Sean bought -/
def num_sodas : ℕ := sorry

/-- The cost of one soup in dollars -/
def cost_soup : ℚ := sorry

/-- The cost of the sandwich in dollars -/
def cost_sandwich : ℚ := sorry

/-- The total cost of the purchase in dollars -/
def total_cost : ℚ := 18

theorem seans_soda_purchase :
  (num_sodas : ℚ) = cost_soup ∧
  cost_sandwich = 3 * cost_soup ∧
  (num_sodas : ℚ) * 1 + 2 * cost_soup + cost_sandwich = total_cost ∧
  num_sodas = 3 := by sorry

end seans_soda_purchase_l2640_264033


namespace initial_peaches_l2640_264067

/-- Given a basket of peaches, prove that the initial number of peaches is 20 
    when 25 more are added to make a total of 45. -/
theorem initial_peaches (initial : ℕ) : initial + 25 = 45 → initial = 20 := by
  sorry

end initial_peaches_l2640_264067


namespace no_number_decreases_by_1981_l2640_264091

theorem no_number_decreases_by_1981 : 
  ¬ ∃ (N : ℕ), 
    ∃ (M : ℕ), 
      (N ≠ 0) ∧ 
      (M ≠ 0) ∧
      (N = 1981 * M) ∧
      (∃ (d : ℕ) (k : ℕ), N = d * 10^k + M ∧ 1 ≤ d ∧ d ≤ 9) :=
by sorry

end no_number_decreases_by_1981_l2640_264091


namespace simplify_expression_1_simplify_expression_2_l2640_264057

-- Theorem 1
theorem simplify_expression_1 (a : ℝ) : a^2 - 2*a - 3*a^2 + 4*a = -2*a^2 + 2*a := by
  sorry

-- Theorem 2
theorem simplify_expression_2 (x : ℝ) : 4*(x^2 - 2) - 2*(2*x^2 + 3*x + 3) + 7*x = x - 14 := by
  sorry

end simplify_expression_1_simplify_expression_2_l2640_264057


namespace yoga_time_calculation_l2640_264007

/-- Calculates the yoga time given exercise ratios and bicycle riding time -/
def yoga_time (gym_bike_ratio : Rat) (yoga_exercise_ratio : Rat) (bike_time : ℕ) : ℕ :=
  let gym_time := (gym_bike_ratio.num * bike_time) / gym_bike_ratio.den
  let total_exercise_time := gym_time + bike_time
  ((yoga_exercise_ratio.num * total_exercise_time) / yoga_exercise_ratio.den).toNat

/-- Proves that given the specified ratios and bicycle riding time, the yoga time is 20 minutes -/
theorem yoga_time_calculation :
  yoga_time (2 / 3) (2 / 3) 18 = 20 := by
  sorry

#eval yoga_time (2 / 3) (2 / 3) 18

end yoga_time_calculation_l2640_264007


namespace rainfall_increase_l2640_264021

/-- Given the rainfall data for Rainville in 2010 and 2011, prove the increase in average monthly rainfall. -/
theorem rainfall_increase (average_2010 total_2011 : ℝ) (h1 : average_2010 = 35) 
  (h2 : total_2011 = 504) : ∃ x : ℝ, 
  12 * (average_2010 + x) = total_2011 ∧ x = 7 := by
  sorry

end rainfall_increase_l2640_264021


namespace hen_duck_speed_ratio_l2640_264090

/-- The number of leaps a hen makes for every 8 leaps of a duck -/
def hen_leaps : ℕ := 6

/-- The number of duck leaps that equal 3 hen leaps -/
def duck_leaps_equal : ℕ := 4

/-- The number of hen leaps that equal 4 duck leaps -/
def hen_leaps_equal : ℕ := 3

/-- The number of duck leaps for which we compare hen leaps -/
def duck_comparison_leaps : ℕ := 8

theorem hen_duck_speed_ratio :
  (hen_leaps : ℚ) / duck_comparison_leaps = 1 := by
  sorry

end hen_duck_speed_ratio_l2640_264090


namespace polynomial_value_equivalence_l2640_264011

theorem polynomial_value_equivalence (x y : ℝ) :
  3 * x^2 + 4 * y + 9 = 8 → 9 * x^2 + 12 * y + 8 = 5 := by
  sorry

end polynomial_value_equivalence_l2640_264011


namespace school_classrooms_l2640_264069

theorem school_classrooms 
  (total_students : ℕ) 
  (seats_per_bus : ℕ) 
  (buses_needed : ℕ) 
  (h1 : total_students = 58)
  (h2 : seats_per_bus = 2)
  (h3 : buses_needed = 29)
  (h4 : total_students = buses_needed * seats_per_bus)
  (h5 : ∃ (students_per_class : ℕ), total_students % students_per_class = 0) :
  ∃ (num_classrooms : ℕ), num_classrooms = 2 ∧ 
    total_students / num_classrooms = buses_needed := by
  sorry

end school_classrooms_l2640_264069


namespace sqrt_49_times_sqrt_25_squared_l2640_264087

theorem sqrt_49_times_sqrt_25_squared : (Real.sqrt (49 * Real.sqrt 25))^2 = 245 := by
  sorry

end sqrt_49_times_sqrt_25_squared_l2640_264087


namespace integer_triangle_exists_l2640_264044

/-- A triangle with integer side lengths forming an arithmetic progression and integer area -/
structure IntegerTriangle where
  a : ℕ
  b : ℕ
  c : ℕ
  area : ℕ
  arith_prog : b - a = c - b
  area_formula : area^2 = (a + b + c) * (b + c - a) * (a + c - b) * (a + b - c) / 16

/-- The existence of a specific integer triangle with sides 3, 4, 5 -/
theorem integer_triangle_exists : ∃ (t : IntegerTriangle), t.a = 3 ∧ t.b = 4 ∧ t.c = 5 := by
  sorry

end integer_triangle_exists_l2640_264044


namespace area_of_large_rectangle_l2640_264014

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Represents a square -/
structure Square where
  side : ℝ

/-- Calculates the area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℝ := r.width * r.height

/-- Calculates the area of a square -/
def Square.area (s : Square) : ℝ := s.side * s.side

/-- The theorem to be proved -/
theorem area_of_large_rectangle (shaded_square : Square) 
  (bottom_rect left_rect : Rectangle) :
  shaded_square.area = 4 →
  bottom_rect.width = shaded_square.side →
  bottom_rect.height + left_rect.height = shaded_square.side →
  left_rect.width + bottom_rect.width = shaded_square.side →
  (shaded_square.area + bottom_rect.area + left_rect.area = 12) := by
  sorry

end area_of_large_rectangle_l2640_264014


namespace domain_of_w_l2640_264099

-- Define the function w(y)
def w (y : ℝ) : ℝ := (y - 3) ^ (1/3) + (15 - y) ^ (1/3)

-- State the theorem about the domain of w
theorem domain_of_w :
  ∀ y : ℝ, ∃ z : ℝ, w y = z :=
sorry

end domain_of_w_l2640_264099


namespace sci_fi_readers_l2640_264065

theorem sci_fi_readers (total : ℕ) (literary : ℕ) (both : ℕ) (sci_fi : ℕ) : 
  total = 250 → literary = 88 → both = 18 → sci_fi = total + both - literary :=
by
  sorry

end sci_fi_readers_l2640_264065


namespace overlap_area_and_perimeter_l2640_264093

/-- Given two strips of widths 1 and 2 overlapping at an angle of π/4 radians,
    the area of the overlap region is √2 and the perimeter is 4√3. -/
theorem overlap_area_and_perimeter :
  ∀ (strip1_width strip2_width overlap_angle : ℝ),
    strip1_width = 1 →
    strip2_width = 2 →
    overlap_angle = π / 4 →
    ∃ (area perimeter : ℝ),
      area = Real.sqrt 2 ∧
      perimeter = 4 * Real.sqrt 3 :=
by sorry

end overlap_area_and_perimeter_l2640_264093


namespace fill_time_both_pipes_l2640_264085

-- Define the time it takes for Pipe A to fill the tank
def pipeA_time : ℝ := 12

-- Define the rate at which Pipe B fills the tank relative to Pipe A
def pipeB_rate_multiplier : ℝ := 3

-- Theorem stating the time it takes to fill the tank with both pipes open
theorem fill_time_both_pipes (pipeA_time : ℝ) (pipeB_rate_multiplier : ℝ) 
  (h1 : pipeA_time > 0) (h2 : pipeB_rate_multiplier > 0) :
  (1 / (1 / pipeA_time + pipeB_rate_multiplier / pipeA_time)) = 3 := by
  sorry

#check fill_time_both_pipes

end fill_time_both_pipes_l2640_264085


namespace all_signs_flippable_l2640_264080

/-- Represents a grid of +1 and -1 values -/
def Grid (m n : ℕ) := Fin m → Fin n → Int

/-- Represents the allowed sign-changing patterns -/
inductive Pattern
| horizontal : Pattern
| vertical : Pattern

/-- Applies a pattern to a specific location in the grid -/
def applyPattern (g : Grid m n) (p : Pattern) (i j : ℕ) : Grid m n :=
  sorry

/-- Checks if all signs in the grid have been flipped -/
def allSignsFlipped (g₁ g₂ : Grid m n) : Prop :=
  sorry

/-- Main theorem: All signs can be flipped iff m and n are multiples of 4 -/
theorem all_signs_flippable (m n : ℕ) :
  (∃ (g : Grid m n), ∃ (operations : List (Pattern × ℕ × ℕ)),
    allSignsFlipped g (operations.foldl (λ acc (p, i, j) => applyPattern acc p i j) g))
  ↔ (∃ (k₁ k₂ : ℕ), m = 4 * k₁ ∧ n = 4 * k₂) :=
sorry

end all_signs_flippable_l2640_264080


namespace square_playground_area_l2640_264022

theorem square_playground_area (w : ℝ) (s : ℝ) : 
  s = 3 * w + 10 →
  4 * s = 480 →
  s * s = 14400 := by
sorry

end square_playground_area_l2640_264022


namespace no_root_in_interval_l2640_264047

-- Define the function f(x) = x^5 - 3x - 1
def f (x : ℝ) : ℝ := x^5 - 3*x - 1

-- State the theorem
theorem no_root_in_interval :
  (∀ x ∈ Set.Ioo 2 3, f x ≠ 0) ∧ Continuous f := by sorry

end no_root_in_interval_l2640_264047


namespace baseball_soccer_difference_l2640_264004

def total_balls : ℕ := 145
def soccer_balls : ℕ := 20
def volleyball_balls : ℕ := 30

def basketball_balls : ℕ := soccer_balls + 5
def tennis_balls : ℕ := 2 * soccer_balls

def baseball_balls : ℕ := total_balls - (soccer_balls + basketball_balls + tennis_balls + volleyball_balls)

theorem baseball_soccer_difference :
  baseball_balls - soccer_balls = 10 :=
by sorry

end baseball_soccer_difference_l2640_264004


namespace equation_solution_l2640_264062

theorem equation_solution (x : ℝ) : 
  (x / 6) / 3 = 6 / (x / 3) → x = 18 ∨ x = -18 := by
  sorry

end equation_solution_l2640_264062


namespace angle_Q_measure_l2640_264028

-- Define a scalene triangle PQR
structure ScaleneTriangle where
  P : ℝ
  Q : ℝ
  R : ℝ
  scalene : P ≠ Q ∧ Q ≠ R ∧ R ≠ P
  sum_180 : P + Q + R = 180

-- Theorem statement
theorem angle_Q_measure (t : ScaleneTriangle) 
  (h1 : t.Q = 2 * t.P) 
  (h2 : t.R = 3 * t.P) : 
  t.Q = 60 := by
  sorry

end angle_Q_measure_l2640_264028


namespace train_speed_l2640_264039

/-- The speed of a train given its length and time to cross a fixed point -/
theorem train_speed (length : ℝ) (time : ℝ) (h1 : length = 300) (h2 : time = 10) :
  length / time = 30 := by
  sorry

#check train_speed

end train_speed_l2640_264039


namespace pencil_buyers_difference_l2640_264042

-- Define the cost of a pencil in cents
def pencil_cost : ℕ := 12

-- Define the total amount paid by seventh graders in cents
def seventh_grade_total : ℕ := 192

-- Define the total amount paid by sixth graders in cents
def sixth_grade_total : ℕ := 252

-- Define the number of sixth graders
def total_sixth_graders : ℕ := 35

-- Theorem statement
theorem pencil_buyers_difference : 
  (sixth_grade_total / pencil_cost) - (seventh_grade_total / pencil_cost) = 5 := by
  sorry

end pencil_buyers_difference_l2640_264042


namespace function_properties_l2640_264083

/-- Given a function f with parameter ω, prove properties about its graph -/
theorem function_properties (ω : ℝ) (h_ω_pos : ω > 0) :
  let f : ℝ → ℝ := λ x ↦ Real.sqrt 3 * Real.sin (ω * x) + 2 * (Real.sin (ω * x / 2))^2
  -- Assume the graph has exactly three symmetric centers on [0, π]
  (∃ (x₁ x₂ x₃ : ℝ), 0 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ < x₃ ∧ x₃ ≤ π ∧
    (∀ (y : ℝ), 0 ≤ y ∧ y ≤ π → (f y = f (2 * x₁ - y) ∨ f y = f (2 * x₂ - y) ∨ f y = f (2 * x₃ - y))) ∧
    (∀ (z : ℝ), 0 ≤ z ∧ z ≤ π → (z = x₁ ∨ z = x₂ ∨ z = x₃ ∨ f z ≠ f (2 * z - z)))) →
  -- Then prove:
  (13/6 ≤ ω ∧ ω < 19/6) ∧  -- 1. Range of ω
  (∃ (n : ℕ), n = 2 ∨ n = 3 ∧  -- 2. Number of axes of symmetry
    ∃ (x₁ x₂ x₃ : ℝ), 0 < x₁ ∧ x₁ < x₂ ∧ (n = 3 → x₂ < x₃) ∧ x₃ < π ∧
    (∀ (y : ℝ), 0 ≤ y ∧ y ≤ π → (f y = f (2 * x₁ - y) ∨ f y = f (2 * x₂ - y) ∨ (n = 3 → f y = f (2 * x₃ - y))))) ∧
  (∃ (x : ℝ), 0 < x ∧ x < π/4 ∧ f x = 3) ∧  -- 3. Maximum value on (0, π/4)
  (∀ (x y : ℝ), 0 < x ∧ x < y ∧ y < π/6 → f x < f y)  -- 4. Increasing on (0, π/6)
  := by sorry

end function_properties_l2640_264083


namespace num_parallelepipeds_is_29_l2640_264053

/-- A set of 4 points in 3D space -/
structure PointSet :=
  (points : Fin 4 → ℝ × ℝ × ℝ)
  (not_coplanar : ∀ (p : ℝ × ℝ × ℝ → ℝ), ¬(∀ i, p (points i) = 0))

/-- A parallelepiped formed by 4 vertices -/
structure Parallelepiped :=
  (vertices : Fin 8 → ℝ × ℝ × ℝ)

/-- The number of distinct parallelepipeds that can be formed from a set of 4 points -/
def num_parallelepipeds (ps : PointSet) : ℕ :=
  -- Definition here (not implemented)
  0

/-- Theorem: The number of distinct parallelepipeds formed by 4 non-coplanar points is 29 -/
theorem num_parallelepipeds_is_29 (ps : PointSet) : num_parallelepipeds ps = 29 := by
  sorry

end num_parallelepipeds_is_29_l2640_264053


namespace compute_alpha_l2640_264050

theorem compute_alpha (α β : ℂ) 
  (h1 : (α + β).re > 0)
  (h2 : (Complex.I * (2 * α - 3 * β)).re > 0)
  (h3 : β = 4 + 3 * Complex.I) :
  α = 6 + (3 / 2) * Complex.I := by sorry

end compute_alpha_l2640_264050


namespace all_statements_false_l2640_264025

def sharp (n : ℕ) : ℚ := 1 / (n + 1)

theorem all_statements_false :
  (sharp 4 + sharp 8 ≠ sharp 12) ∧
  (sharp 9 - sharp 3 ≠ sharp 6) ∧
  (sharp 5 * sharp 7 ≠ sharp 35) ∧
  (sharp 15 / sharp 3 ≠ sharp 5) := by
sorry

end all_statements_false_l2640_264025


namespace max_value_on_circle_l2640_264006

theorem max_value_on_circle (x y : ℝ) (h : x^2 + y^2 = 14*x + 6*y + 6) :
  3*x + 4*y ≤ 73 :=
sorry

end max_value_on_circle_l2640_264006


namespace class_a_win_probability_class_b_score_expectation_l2640_264054

/-- Represents the result of a single event --/
inductive EventResult
| Win
| Lose

/-- Represents the outcome of the three events for a class --/
structure ClassOutcome :=
  (event1 : EventResult)
  (event2 : EventResult)
  (event3 : EventResult)

/-- Calculates the score for a given ClassOutcome --/
def score (outcome : ClassOutcome) : Int :=
  let e1 := match outcome.event1 with
    | EventResult.Win => 2
    | EventResult.Lose => -1
  let e2 := match outcome.event2 with
    | EventResult.Win => 2
    | EventResult.Lose => -1
  let e3 := match outcome.event3 with
    | EventResult.Win => 2
    | EventResult.Lose => -1
  e1 + e2 + e3

/-- Probabilities of Class A winning each event --/
def probA1 : Float := 0.4
def probA2 : Float := 0.5
def probA3 : Float := 0.8

/-- Theorem stating the probability of Class A winning the championship --/
theorem class_a_win_probability :
  let p := probA1 * probA2 * probA3 +
           (1 - probA1) * probA2 * probA3 +
           probA1 * (1 - probA2) * probA3 +
           probA1 * probA2 * (1 - probA3)
  p = 0.6 := by sorry

/-- Theorem stating the expectation of Class B's total score --/
theorem class_b_score_expectation :
  let p_neg3 := probA1 * probA2 * probA3
  let p_0 := (1 - probA1) * probA2 * probA3 + probA1 * (1 - probA2) * probA3 + probA1 * probA2 * (1 - probA3)
  let p_3 := (1 - probA1) * (1 - probA2) * probA3 + (1 - probA1) * probA2 * (1 - probA3) + probA1 * (1 - probA2) * (1 - probA3)
  let p_6 := (1 - probA1) * (1 - probA2) * (1 - probA3)
  let expectation := -3 * p_neg3 + 0 * p_0 + 3 * p_3 + 6 * p_6
  expectation = 0.9 := by sorry

end class_a_win_probability_class_b_score_expectation_l2640_264054


namespace sum_of_roots_quadratic_l2640_264064

theorem sum_of_roots_quadratic (a b : ℝ) : 
  (∀ x : ℝ, x^2 - (a+b)*x + a*b + 1 = 0 ↔ x = a ∨ x = b) → 
  a + b = a + b :=
by sorry

end sum_of_roots_quadratic_l2640_264064


namespace factorization_a_squared_minus_ab_l2640_264020

theorem factorization_a_squared_minus_ab (a b : ℝ) : a^2 - a*b = a*(a - b) := by
  sorry

end factorization_a_squared_minus_ab_l2640_264020


namespace smallest_number_proof_l2640_264046

theorem smallest_number_proof (a b c : ℕ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →
  (a + b + c) / 3 = 24 →
  b = 23 →
  max a (max b c) = b + 4 →
  min a (min b c) = 22 :=
by sorry

end smallest_number_proof_l2640_264046


namespace embankment_build_time_l2640_264016

/-- Represents the time taken to build an embankment given a number of workers -/
def build_time (workers : ℕ) (days : ℚ) : Prop :=
  workers * days = 300

theorem embankment_build_time :
  build_time 75 4 → build_time 50 6 := by
  sorry

end embankment_build_time_l2640_264016


namespace average_age_problem_l2640_264089

theorem average_age_problem (devin_age eden_age mom_age : ℕ) : 
  devin_age = 12 →
  eden_age = 2 * devin_age →
  mom_age = 2 * eden_age →
  (devin_age + eden_age + mom_age) / 3 = 28 := by
  sorry

end average_age_problem_l2640_264089


namespace selection_theorem_l2640_264063

/-- The number of ways to select 3 people from 11, with at least one of A or B selected and C not selected -/
def selection_ways (n : ℕ) (k : ℕ) (total : ℕ) : ℕ :=
  (2 * Nat.choose (total - 3) (k - 1)) + Nat.choose (total - 3) (k - 2)

/-- Theorem stating that the number of selection ways is 64 -/
theorem selection_theorem : selection_ways 3 3 11 = 64 := by
  sorry

end selection_theorem_l2640_264063


namespace submerged_sphere_pressure_l2640_264071

/-- The total water pressure on a submerged sphere -/
theorem submerged_sphere_pressure
  (diameter : ℝ) (depth : ℝ) (ρ : ℝ) (g : ℝ) :
  diameter = 4 →
  depth = 3 →
  ρ > 0 →
  g > 0 →
  (∫ x in (-2 : ℝ)..2, 4 * π * ρ * g * (depth + x)) = 64 * π * ρ * g :=
by sorry

end submerged_sphere_pressure_l2640_264071


namespace four_percent_of_y_is_sixteen_l2640_264073

theorem four_percent_of_y_is_sixteen (y : ℝ) (h1 : y > 0) (h2 : 0.04 * y = 16) : y = 400 := by
  sorry

end four_percent_of_y_is_sixteen_l2640_264073
