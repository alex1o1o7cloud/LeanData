import Mathlib

namespace NUMINAMATH_CALUDE_cookies_per_bag_l3026_302648

theorem cookies_per_bag 
  (chocolate_chip : ℕ) 
  (oatmeal : ℕ) 
  (baggies : ℕ) 
  (h1 : chocolate_chip = 2) 
  (h2 : oatmeal = 16) 
  (h3 : baggies = 6) 
  : (chocolate_chip + oatmeal) / baggies = 3 := by
  sorry

end NUMINAMATH_CALUDE_cookies_per_bag_l3026_302648


namespace NUMINAMATH_CALUDE_infinite_nested_radical_sqrt_3_l3026_302682

theorem infinite_nested_radical_sqrt_3 :
  ∃! (x : ℝ), x > 0 ∧ x = Real.sqrt (3 - x) :=
by
  -- The unique positive solution is (-1 + √13) / 2
  have solution : ℝ := (-1 + Real.sqrt 13) / 2
  
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_infinite_nested_radical_sqrt_3_l3026_302682


namespace NUMINAMATH_CALUDE_three_numbers_sum_to_perfect_square_l3026_302644

def numbers : List Nat := [4784887, 2494651, 8595087, 1385287, 9042451, 9406087]

theorem three_numbers_sum_to_perfect_square :
  ∃ (a b c : Nat), a ∈ numbers ∧ b ∈ numbers ∧ c ∈ numbers ∧
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  ∃ (n : Nat), a + b + c = n * n :=
by
  sorry

end NUMINAMATH_CALUDE_three_numbers_sum_to_perfect_square_l3026_302644


namespace NUMINAMATH_CALUDE_fahrenheit_to_celsius_l3026_302619

theorem fahrenheit_to_celsius (F C : ℝ) : 
  F = 95 → F = (9/5) * C + 32 → C = 35 := by
  sorry

end NUMINAMATH_CALUDE_fahrenheit_to_celsius_l3026_302619


namespace NUMINAMATH_CALUDE_probability_end_multiple_of_three_is_31_90_l3026_302627

def is_multiple_of_three (n : ℕ) : Prop := ∃ k, n = 3 * k

def probability_end_multiple_of_three : ℚ :=
  let total_cards := 10
  let prob_left := 1 / 3
  let prob_right := 2 / 3
  let prob_start_multiple_3 := 3 / 10
  let prob_start_one_more := 4 / 10
  let prob_start_one_less := 3 / 10
  let prob_end_multiple_3_from_multiple_3 := prob_left * prob_right + prob_right * prob_left
  let prob_end_multiple_3_from_one_more := prob_right * prob_right
  let prob_end_multiple_3_from_one_less := prob_left * prob_left
  prob_start_multiple_3 * prob_end_multiple_3_from_multiple_3 +
  prob_start_one_more * prob_end_multiple_3_from_one_more +
  prob_start_one_less * prob_end_multiple_3_from_one_less

theorem probability_end_multiple_of_three_is_31_90 :
  probability_end_multiple_of_three = 31 / 90 := by
  sorry

end NUMINAMATH_CALUDE_probability_end_multiple_of_three_is_31_90_l3026_302627


namespace NUMINAMATH_CALUDE_triangle_max_value_l3026_302699

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    where the area is √3 and cos(C) / cos(B) = c / (2a - b),
    the maximum value of 1/(b+1) + 9/(a+9) is 3/5. -/
theorem triangle_max_value (a b c : ℝ) (A B C : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  0 < A → A < π →
  0 < B → B < π →
  0 < C → C < π →
  A + B + C = π →
  (1/2) * a * b * Real.sin C = Real.sqrt 3 →
  Real.cos C / Real.cos B = c / (2*a - b) →
  (∃ (x : ℝ), (1/(b+1) + 9/(a+9) ≤ x) ∧ 
   (∀ (y : ℝ), 1/(b+1) + 9/(a+9) ≤ y → x ≤ y)) →
  (1/(b+1) + 9/(a+9)) ≤ 3/5 :=
by sorry

end NUMINAMATH_CALUDE_triangle_max_value_l3026_302699


namespace NUMINAMATH_CALUDE_line_through_points_with_slope_one_l3026_302618

/-- Given a line passing through points M(-2, a) and N(a, 4) with a slope of 1, prove that a = 1 -/
theorem line_through_points_with_slope_one (a : ℝ) : 
  (let M := (-2, a)
   let N := (a, 4)
   (4 - a) / (a - (-2)) = 1) → 
  a = 1 := by
  sorry

end NUMINAMATH_CALUDE_line_through_points_with_slope_one_l3026_302618


namespace NUMINAMATH_CALUDE_student_assistant_sequences_l3026_302600

/-- The number of students in the class -/
def num_students : ℕ := 15

/-- The number of class meetings per week -/
def meetings_per_week : ℕ := 5

/-- The number of different sequences of student assistants possible in one week -/
def num_sequences : ℕ := num_students ^ meetings_per_week

theorem student_assistant_sequences :
  num_sequences = 759375 :=
by sorry

end NUMINAMATH_CALUDE_student_assistant_sequences_l3026_302600


namespace NUMINAMATH_CALUDE_quadruple_primes_l3026_302635

theorem quadruple_primes (p q r : ℕ) (n : ℕ+) : 
  (Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧ p^2 = q^2 + r^(n : ℕ)) ↔ 
  ((p = 3 ∧ q = 2 ∧ r = 5 ∧ n = 1) ∨ (p = 5 ∧ q = 3 ∧ r = 2 ∧ n = 4)) :=
sorry

end NUMINAMATH_CALUDE_quadruple_primes_l3026_302635


namespace NUMINAMATH_CALUDE_article_cost_l3026_302640

/-- Calculates the final cost of an article after two years of inflation and price changes -/
def finalCost (originalCost : ℝ) (inflationRate : ℝ) 
  (year1Increase year1Decrease year2Increase year2Decrease : ℝ) : ℝ :=
  let adjustedCost1 := originalCost * (1 + inflationRate)
  let afterYear1 := adjustedCost1 * (1 + year1Increase) * (1 - year1Decrease)
  let adjustedCost2 := afterYear1 * (1 + inflationRate)
  adjustedCost2 * (1 + year2Increase) * (1 - year2Decrease)

/-- Theorem stating the final cost of the article after two years -/
theorem article_cost : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  |finalCost 75 0.05 0.20 0.20 0.30 0.25 - 77.40| < ε :=
sorry

end NUMINAMATH_CALUDE_article_cost_l3026_302640


namespace NUMINAMATH_CALUDE_function_difference_theorem_l3026_302601

theorem function_difference_theorem (p q c : ℝ) (hp : p ≠ 0) (hq : q ≠ 0) :
  let f : ℝ → ℝ := λ x => p * x^6 + q * x^4 + 3 * x - Real.sqrt 2
  let d := f c - f (-c)
  d = 6 * c := by sorry

end NUMINAMATH_CALUDE_function_difference_theorem_l3026_302601


namespace NUMINAMATH_CALUDE_math_club_trips_l3026_302622

/-- Represents a math club with field trips -/
structure MathClub where
  total_students : ℕ
  students_per_trip : ℕ
  (total_students_pos : total_students > 0)
  (students_per_trip_pos : students_per_trip > 0)
  (students_per_trip_le_total : students_per_trip ≤ total_students)

/-- The minimum number of trips for one student to meet all others -/
def min_trips_for_one (club : MathClub) : ℕ :=
  (club.total_students - 1 + club.students_per_trip - 2) / (club.students_per_trip - 1)

/-- The minimum number of trips for all pairs to meet -/
def min_trips_for_all_pairs (club : MathClub) : ℕ :=
  (club.total_students * (club.total_students - 1)) / (club.students_per_trip * (club.students_per_trip - 1))

theorem math_club_trips (club : MathClub) 
  (h1 : club.total_students = 12) 
  (h2 : club.students_per_trip = 6) : 
  min_trips_for_one club = 3 ∧ min_trips_for_all_pairs club = 6 := by
  sorry

#eval min_trips_for_one ⟨12, 6, by norm_num, by norm_num, by norm_num⟩
#eval min_trips_for_all_pairs ⟨12, 6, by norm_num, by norm_num, by norm_num⟩

end NUMINAMATH_CALUDE_math_club_trips_l3026_302622


namespace NUMINAMATH_CALUDE_three_correct_statements_l3026_302660

theorem three_correct_statements : 
  (0 ∉ (∅ : Set ℕ)) ∧ 
  ((∅ : Set ℕ) ⊆ {1,2}) ∧ 
  ({(x,y) : ℝ × ℝ | 2*x + y = 10 ∧ 3*x - y = 5} ≠ {3,4}) ∧ 
  (∀ A B : Set α, A ⊆ B → A ∩ B = A) :=
by sorry

end NUMINAMATH_CALUDE_three_correct_statements_l3026_302660


namespace NUMINAMATH_CALUDE_n_has_nine_digits_l3026_302672

/-- The smallest positive integer satisfying the given conditions -/
def n : ℕ := sorry

/-- n is divisible by 30 -/
axiom n_div_30 : 30 ∣ n

/-- n^2 is a perfect fourth power -/
axiom n_sq_fourth_power : ∃ k : ℕ, n^2 = k^4

/-- n^4 is a perfect cube -/
axiom n_fourth_cube : ∃ k : ℕ, n^4 = k^3

/-- n is the smallest positive integer satisfying the conditions -/
axiom n_smallest : ∀ m : ℕ, m < n → ¬(30 ∣ m ∧ (∃ k : ℕ, m^2 = k^4) ∧ (∃ k : ℕ, m^4 = k^3))

/-- The number of digits in n -/
def num_digits (x : ℕ) : ℕ := sorry

/-- Theorem stating that n has 9 digits -/
theorem n_has_nine_digits : num_digits n = 9 := by sorry

end NUMINAMATH_CALUDE_n_has_nine_digits_l3026_302672


namespace NUMINAMATH_CALUDE_brick_height_l3026_302674

/-- The surface area of a rectangular prism -/
def surface_area (l w h : ℝ) : ℝ := 2 * l * w + 2 * l * h + 2 * w * h

/-- Proves that the height of a brick with given dimensions and surface area is 3 cm -/
theorem brick_height (l w sa : ℝ) (hl : l = 10) (hw : w = 4) (hsa : sa = 164) :
  ∃ h : ℝ, h = 3 ∧ surface_area l w h = sa :=
by sorry

end NUMINAMATH_CALUDE_brick_height_l3026_302674


namespace NUMINAMATH_CALUDE_log_seven_eighteen_l3026_302642

theorem log_seven_eighteen (a b : ℝ) (h1 : Real.log 2 / Real.log 10 = a) (h2 : Real.log 3 / Real.log 10 = b) :
  Real.log 18 / Real.log 7 = (a + 2 * b) / (1 - a) := by
  sorry

end NUMINAMATH_CALUDE_log_seven_eighteen_l3026_302642


namespace NUMINAMATH_CALUDE_binomial_expansion_example_l3026_302604

theorem binomial_expansion_example : 16^3 + 3*(16^2)*2 + 3*16*(2^2) + 2^3 = (16 + 2)^3 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_example_l3026_302604


namespace NUMINAMATH_CALUDE_only_negative_three_smaller_than_negative_two_l3026_302634

theorem only_negative_three_smaller_than_negative_two :
  (0 > -2) ∧ (-1 > -2) ∧ (-3 < -2) ∧ (1 > -2) :=
by sorry

end NUMINAMATH_CALUDE_only_negative_three_smaller_than_negative_two_l3026_302634


namespace NUMINAMATH_CALUDE_first_part_to_total_ratio_l3026_302697

theorem first_part_to_total_ratio : 
  ∃ (n : ℕ), (246.95 : ℝ) / 782 = (4939 : ℝ) / (15640 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_first_part_to_total_ratio_l3026_302697


namespace NUMINAMATH_CALUDE_simplify_expressions_l3026_302687

theorem simplify_expressions :
  (2 * Real.sqrt 12 - 6 * Real.sqrt (1/3) + Real.sqrt 48 = -2 * Real.sqrt 3) ∧
  ((Real.sqrt 3 - Real.pi)^0 - (Real.sqrt 20 - Real.sqrt 15) / Real.sqrt 5 + (-1)^2017 = Real.sqrt 3 - 2) := by
  sorry

end NUMINAMATH_CALUDE_simplify_expressions_l3026_302687


namespace NUMINAMATH_CALUDE_handshake_theorem_l3026_302639

def number_of_people : ℕ := 12
def handshakes_per_person : ℕ := 3

def handshake_arrangements (n : ℕ) (k : ℕ) : ℕ := sorry

theorem handshake_theorem :
  let M := handshake_arrangements number_of_people handshakes_per_person
  M = 6100940 ∧ M % 1000 = 940 := by sorry

end NUMINAMATH_CALUDE_handshake_theorem_l3026_302639


namespace NUMINAMATH_CALUDE_class_size_proof_l3026_302684

def class_composition (total : ℕ) (girls : ℕ) (boys : ℕ) : Prop :=
  girls + boys = total ∧ girls = (60 * total) / 100

def absent_composition (total : ℕ) (girls : ℕ) (boys : ℕ) : Prop :=
  (girls - 1) = (625 * (total - 3)) / 1000

theorem class_size_proof (total : ℕ) (girls : ℕ) (boys : ℕ) :
  class_composition total girls boys ∧ 
  absent_composition total girls boys →
  girls = 21 ∧ boys = 14 :=
by sorry

end NUMINAMATH_CALUDE_class_size_proof_l3026_302684


namespace NUMINAMATH_CALUDE_smallest_value_x_l3026_302683

/-- Given a system of linear equations, prove that x is the smallest value -/
theorem smallest_value_x (x y z : ℝ) 
  (eq1 : 3 * x - y = 20)
  (eq2 : 2 * z = 3 * y)
  (eq3 : x + y + z = 48) :
  x < y ∧ x < z :=
by sorry

end NUMINAMATH_CALUDE_smallest_value_x_l3026_302683


namespace NUMINAMATH_CALUDE_phone_bill_increase_l3026_302671

theorem phone_bill_increase (usual_bill : ℝ) (increase_rate : ℝ) (months : ℕ) : 
  usual_bill = 50 → 
  increase_rate = 0.1 → 
  months = 12 → 
  (usual_bill + usual_bill * increase_rate) * months = 660 := by
sorry

end NUMINAMATH_CALUDE_phone_bill_increase_l3026_302671


namespace NUMINAMATH_CALUDE_sprint_team_distance_l3026_302680

/-- Given a sprint team with a certain number of people, where each person runs a fixed distance,
    calculate the total distance run by the team. -/
def total_distance (team_size : ℝ) (distance_per_person : ℝ) : ℝ :=
  team_size * distance_per_person

/-- Theorem: A sprint team of 150.0 people, where each person runs 5.0 miles,
    will run a total of 750.0 miles. -/
theorem sprint_team_distance :
  total_distance 150.0 5.0 = 750.0 := by
  sorry

end NUMINAMATH_CALUDE_sprint_team_distance_l3026_302680


namespace NUMINAMATH_CALUDE_composite_function_equality_l3026_302607

theorem composite_function_equality (a : ℚ) : 
  let f (x : ℚ) := x / 5 + 4
  let g (x : ℚ) := 5 * x - 3
  f (g a) = 7 → a = 18 / 5 := by
sorry

end NUMINAMATH_CALUDE_composite_function_equality_l3026_302607


namespace NUMINAMATH_CALUDE_equality_implies_two_equal_l3026_302610

theorem equality_implies_two_equal (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : x^2/y + y^2/z + z^2/x = x^2/z + z^2/y + y^2/x) :
  x = y ∨ x = z ∨ y = z := by
  sorry

end NUMINAMATH_CALUDE_equality_implies_two_equal_l3026_302610


namespace NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l3026_302690

theorem quadratic_two_distinct_roots (k : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + 3*x₁ + k = 0 ∧ x₂^2 + 3*x₂ + k = 0) ↔ k < 9/4 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l3026_302690


namespace NUMINAMATH_CALUDE_theater_ticket_difference_l3026_302654

theorem theater_ticket_difference :
  ∀ (orchestra_price balcony_price : ℕ) 
    (total_tickets total_cost : ℕ) 
    (orchestra_tickets balcony_tickets : ℕ),
  orchestra_price = 12 →
  balcony_price = 8 →
  total_tickets = 360 →
  total_cost = 3320 →
  orchestra_tickets + balcony_tickets = total_tickets →
  orchestra_price * orchestra_tickets + balcony_price * balcony_tickets = total_cost →
  balcony_tickets - orchestra_tickets = 140 :=
by
  sorry

end NUMINAMATH_CALUDE_theater_ticket_difference_l3026_302654


namespace NUMINAMATH_CALUDE_h_function_proof_l3026_302603

theorem h_function_proof (x : ℝ) (h : ℝ → ℝ) : 
  (12 * x^4 + 4 * x^3 - 2 * x + 3 + h x = 6 * x^3 + 8 * x^2 - 10 * x + 6) →
  (h x = -12 * x^4 + 2 * x^3 + 8 * x^2 - 8 * x + 3) :=
by
  sorry

end NUMINAMATH_CALUDE_h_function_proof_l3026_302603


namespace NUMINAMATH_CALUDE_final_acid_concentration_l3026_302694

/-- Calculates the final acid concentration after removing water from an acidic solution -/
theorem final_acid_concentration
  (initial_volume : ℝ)
  (initial_concentration : ℝ)
  (water_removed : ℝ)
  (h1 : initial_volume = 12)
  (h2 : initial_concentration = 0.4)
  (h3 : water_removed = 4)
  : (initial_volume * initial_concentration) / (initial_volume - water_removed) = 0.6 := by
  sorry

#check final_acid_concentration

end NUMINAMATH_CALUDE_final_acid_concentration_l3026_302694


namespace NUMINAMATH_CALUDE_jane_daffodil_bulbs_l3026_302681

/-- The number of daffodil bulbs Jane planted -/
def daffodil_bulbs : ℕ := 30

theorem jane_daffodil_bulbs :
  let tulip_bulbs : ℕ := 20
  let iris_bulbs : ℕ := tulip_bulbs / 2
  let crocus_bulbs : ℕ := 3 * daffodil_bulbs
  let total_bulbs : ℕ := tulip_bulbs + iris_bulbs + daffodil_bulbs + crocus_bulbs
  let earnings_per_bulb : ℚ := 1/2
  let total_earnings : ℚ := 75
  total_earnings = earnings_per_bulb * total_bulbs :=
by sorry


end NUMINAMATH_CALUDE_jane_daffodil_bulbs_l3026_302681


namespace NUMINAMATH_CALUDE_smallest_p_for_multiple_of_ten_l3026_302685

theorem smallest_p_for_multiple_of_ten (n : ℕ) (h1 : n % 2 = 1) (h2 : n % 7 = 5) :
  ∃ p : ℕ, p > 0 ∧ (n + p) % 10 = 0 ∧ ∀ q : ℕ, 0 < q → (n + q) % 10 = 0 → p ≤ q :=
by sorry

end NUMINAMATH_CALUDE_smallest_p_for_multiple_of_ten_l3026_302685


namespace NUMINAMATH_CALUDE_intersection_line_slope_l3026_302632

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 6*x + 4*y - 8 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 16*x + 8*y + 40 = 0

-- Define the intersection points
def intersection_points (C D : ℝ × ℝ) : Prop :=
  circle1 C.1 C.2 ∧ circle2 C.1 C.2 ∧
  circle1 D.1 D.2 ∧ circle2 D.1 D.2 ∧
  C ≠ D

-- Theorem statement
theorem intersection_line_slope (C D : ℝ × ℝ) 
  (h : intersection_points C D) : 
  (D.2 - C.2) / (D.1 - C.1) = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_line_slope_l3026_302632


namespace NUMINAMATH_CALUDE_function_properties_l3026_302612

open Real

-- Define the function and its derivative
variable (f : ℝ → ℝ)
variable (f' : ℝ → ℝ)

-- Define the interval
variable (a b : ℝ)

-- State the theorem
theorem function_properties
  (hf : Continuous f)
  (hf' : Continuous f')
  (hderiv : ∀ x, HasDerivAt f (f' x) x)
  (hab : a < b)
  (hf'a : f' a > 0)
  (hf'b : f' b < 0) :
  (∃ x₀ ∈ Set.Icc a b, f x₀ > f b) ∧
  (∃ x₀ ∈ Set.Icc a b, f a - f b > f' x₀ * (a - b)) ∧
  ¬(∀ x₀ ∈ Set.Icc a b, f x₀ = 0 → False) ∧
  ¬(∀ x₀ ∈ Set.Icc a b, f x₀ > f a → False) := by
  sorry

end NUMINAMATH_CALUDE_function_properties_l3026_302612


namespace NUMINAMATH_CALUDE_eighteen_percent_of_x_is_ninety_l3026_302669

theorem eighteen_percent_of_x_is_ninety (x : ℝ) : (18 / 100) * x = 90 → x = 500 := by
  sorry

end NUMINAMATH_CALUDE_eighteen_percent_of_x_is_ninety_l3026_302669


namespace NUMINAMATH_CALUDE_equation_proof_l3026_302625

theorem equation_proof : (49 : ℚ) / (7 - 3 / 4) = 196 / 25 := by
  sorry

end NUMINAMATH_CALUDE_equation_proof_l3026_302625


namespace NUMINAMATH_CALUDE_inequality_proof_l3026_302631

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 1) :
  (1 / (a^3 * (b + c))) + (1 / (b^3 * (c + a))) + (1 / (c^3 * (a + b))) ≥ 3/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3026_302631


namespace NUMINAMATH_CALUDE_data_analysis_l3026_302662

def data : List ℕ := [11, 10, 11, 13, 11, 13, 15]

def mode (l : List ℕ) : ℕ := sorry

def mean (l : List ℕ) : ℚ := sorry

def variance (l : List ℕ) : ℚ := sorry

def median (l : List ℕ) : ℕ := sorry

theorem data_analysis (d : List ℕ) (h : d = data) : 
  mode d = 11 ∧ 
  mean d = 12 ∧ 
  variance d = 18/7 ∧ 
  median d = 11 := by sorry

end NUMINAMATH_CALUDE_data_analysis_l3026_302662


namespace NUMINAMATH_CALUDE_average_physics_chemistry_l3026_302645

/-- Given the scores in three subjects, prove the average of two subjects --/
theorem average_physics_chemistry 
  (total_average : ℝ) 
  (physics_math_average : ℝ) 
  (physics_score : ℝ) 
  (h1 : total_average = 60) 
  (h2 : physics_math_average = 90) 
  (h3 : physics_score = 140) : 
  (physics_score + (3 * total_average - physics_score - (2 * physics_math_average - physics_score))) / 2 = 70 := by
  sorry

end NUMINAMATH_CALUDE_average_physics_chemistry_l3026_302645


namespace NUMINAMATH_CALUDE_cos_330_degrees_l3026_302657

theorem cos_330_degrees : Real.cos (330 * π / 180) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_330_degrees_l3026_302657


namespace NUMINAMATH_CALUDE_total_sums_attempted_l3026_302688

theorem total_sums_attempted (right_sums wrong_sums total_sums : ℕ) : 
  wrong_sums = 2 * right_sums →
  right_sums = 8 →
  total_sums = right_sums + wrong_sums →
  total_sums = 24 := by
sorry

end NUMINAMATH_CALUDE_total_sums_attempted_l3026_302688


namespace NUMINAMATH_CALUDE_table_height_is_33_l3026_302620

/-- Represents a block of wood -/
structure Block where
  length : ℝ
  width : ℝ

/-- Represents a table -/
structure Table where
  height : ℝ

/-- Represents the configuration of blocks on the table -/
inductive Configuration
| A
| B

/-- Calculates the total visible length for a given configuration -/
def totalVisibleLength (block : Block) (table : Table) (config : Configuration) : ℝ :=
  match config with
  | Configuration.A => block.length + table.height - block.width
  | Configuration.B => block.width + table.height - block.length

/-- Theorem stating that under the given conditions, the table's height is 33 inches -/
theorem table_height_is_33 (block : Block) (table : Table) :
  totalVisibleLength block table Configuration.A = 36 →
  totalVisibleLength block table Configuration.B = 30 →
  table.height = 33 := by
  sorry

#check table_height_is_33

end NUMINAMATH_CALUDE_table_height_is_33_l3026_302620


namespace NUMINAMATH_CALUDE_completing_square_transform_l3026_302658

theorem completing_square_transform (x : ℝ) :
  (x^2 - 8*x + 5 = 0) ↔ ((x - 4)^2 = 11) :=
by sorry

end NUMINAMATH_CALUDE_completing_square_transform_l3026_302658


namespace NUMINAMATH_CALUDE_intercept_sum_modulo_13_l3026_302641

theorem intercept_sum_modulo_13 : ∃ (x₀ y₀ : ℕ), 
  x₀ < 13 ∧ y₀ < 13 ∧ 
  (4 * x₀ ≡ 1 [MOD 13]) ∧ 
  (3 * y₀ ≡ 12 [MOD 13]) ∧ 
  x₀ + y₀ = 14 := by
  sorry

end NUMINAMATH_CALUDE_intercept_sum_modulo_13_l3026_302641


namespace NUMINAMATH_CALUDE_money_ratio_to_anna_l3026_302696

def total_money : ℕ := 2000
def furniture_cost : ℕ := 400
def money_left : ℕ := 400

def money_after_furniture : ℕ := total_money - furniture_cost
def money_given_to_anna : ℕ := money_after_furniture - money_left

theorem money_ratio_to_anna : 
  (money_given_to_anna : ℚ) / (money_left : ℚ) = 3 := by sorry

end NUMINAMATH_CALUDE_money_ratio_to_anna_l3026_302696


namespace NUMINAMATH_CALUDE_cubic_equation_one_real_root_l3026_302626

theorem cubic_equation_one_real_root :
  ∃! x : ℝ, x^3 - Real.sqrt 3 * x^2 + x - (1 + Real.sqrt 3 / 9) = 0 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_one_real_root_l3026_302626


namespace NUMINAMATH_CALUDE_right_triangle_perimeter_l3026_302623

theorem right_triangle_perimeter (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  a * b / 2 = 150 →
  a = 30 →
  c^2 = a^2 + b^2 →
  a + b + c = 40 + 10 * Real.sqrt 10 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_perimeter_l3026_302623


namespace NUMINAMATH_CALUDE_debby_initial_bottles_l3026_302663

/-- The number of water bottles Debby bought initially -/
def initial_bottles : ℕ := sorry

/-- The number of bottles Debby drinks per day -/
def bottles_per_day : ℕ := 15

/-- The number of days Debby drank water -/
def days_drinking : ℕ := 11

/-- The number of bottles Debby has left -/
def bottles_left : ℕ := 99

/-- Theorem stating that Debby bought 264 water bottles initially -/
theorem debby_initial_bottles : initial_bottles = 264 := by
  sorry

end NUMINAMATH_CALUDE_debby_initial_bottles_l3026_302663


namespace NUMINAMATH_CALUDE_probability_is_four_ninths_l3026_302614

/-- A cube that has been cut into smaller cubes -/
structure CutCube where
  /-- The number of smaller cubes the original cube is cut into -/
  total_cubes : ℕ
  /-- The number of smaller cubes with exactly two faces painted -/
  two_faced_cubes : ℕ
  /-- The total number of smaller cubes is 27 -/
  total_is_27 : total_cubes = 27
  /-- The number of two-faced cubes is 12 -/
  two_faced_is_12 : two_faced_cubes = 12

/-- The probability of selecting a small cube with exactly two faces painted -/
def probability_two_faced (c : CutCube) : ℚ :=
  c.two_faced_cubes / c.total_cubes

theorem probability_is_four_ninths (c : CutCube) :
  probability_two_faced c = 4/9 := by
  sorry

end NUMINAMATH_CALUDE_probability_is_four_ninths_l3026_302614


namespace NUMINAMATH_CALUDE_odd_function_solution_set_l3026_302679

/-- A function is odd if f(-x) = -f(x) for all x -/
def OddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- The solution set of an inequality -/
def SolutionSet (f : ℝ → ℝ) : Set ℝ :=
  {x | x*f x - Real.exp (abs x) > 0}

theorem odd_function_solution_set
  (f : ℝ → ℝ)
  (hodd : OddFunction f)
  (hf1 : f 1 = Real.exp 1)
  (hineq : ∀ x ≥ 0, (x - 1) * f x < x * (deriv f x)) :
  SolutionSet f = Set.Ioi 1 ∪ Set.Iic (-1) :=
sorry

end NUMINAMATH_CALUDE_odd_function_solution_set_l3026_302679


namespace NUMINAMATH_CALUDE_distance_from_origin_l3026_302621

theorem distance_from_origin (z : ℂ) (h : (3 - 4*Complex.I)*z = Complex.abs (4 + 3*Complex.I)) : 
  Complex.abs z = 1 := by sorry

end NUMINAMATH_CALUDE_distance_from_origin_l3026_302621


namespace NUMINAMATH_CALUDE_complex_division_simplification_l3026_302665

theorem complex_division_simplification : 
  (2 + Complex.I) / (1 - 2 * Complex.I) = Complex.I := by sorry

end NUMINAMATH_CALUDE_complex_division_simplification_l3026_302665


namespace NUMINAMATH_CALUDE_tangent_line_parallel_point_exists_l3026_302667

/-- The curve function -/
def f (x : ℝ) : ℝ := x^3 + x - 2

/-- The derivative of the curve function -/
def f' (x : ℝ) : ℝ := 3*x^2 + 1

theorem tangent_line_parallel_point_exists :
  ∃ (x y : ℝ), f x = y ∧ f' x = 4 ∧ x = 1 := by sorry

end NUMINAMATH_CALUDE_tangent_line_parallel_point_exists_l3026_302667


namespace NUMINAMATH_CALUDE_ellipse_equation_l3026_302605

theorem ellipse_equation (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  (∃ c : ℝ, c = 2 ∧ c^2 = a^2 - b^2) →  -- Right focus coincides with parabola focus
  (a / 2 = c) →  -- Eccentricity is 1/2
  (∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1 ↔ x^2 / 16 + y^2 / 12 = 1) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_equation_l3026_302605


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l3026_302659

theorem least_subtraction_for_divisibility : 
  ∃ (x : ℕ), x = 98 ∧ 
  (∀ (y : ℕ), y < x → ¬(769 ∣ (157673 - y))) ∧ 
  (769 ∣ (157673 - x)) := by
  sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l3026_302659


namespace NUMINAMATH_CALUDE_jellybean_count_l3026_302673

theorem jellybean_count (steve matt matilda katy : ℕ) : 
  steve = 84 →
  matt = 10 * steve →
  matilda = matt / 2 →
  katy = 3 * matilda →
  katy = matt / 2 →
  katy = 1260 := by
sorry

end NUMINAMATH_CALUDE_jellybean_count_l3026_302673


namespace NUMINAMATH_CALUDE_stating_max_remaining_coins_product_mod_l3026_302698

/-- Represents the grid size -/
def gridSize : Nat := 418

/-- Represents the modulus for the final result -/
def modulus : Nat := 2007

/-- Represents the maximum number of coins that can remain in one quadrant -/
def maxCoinsPerQuadrant : Nat := (gridSize / 2) * (gridSize / 2)

/-- 
Theorem stating that the maximum value of bw (mod 2007) is 1999, 
where b and w are the number of remaining black and white coins respectively 
after applying the removal rules on a 418 × 418 grid.
-/
theorem max_remaining_coins_product_mod (b w : Nat) : 
  b ≤ maxCoinsPerQuadrant → 
  w ≤ maxCoinsPerQuadrant → 
  (b * w) % modulus ≤ 1999 ∧ 
  ∃ (b' w' : Nat), b' ≤ maxCoinsPerQuadrant ∧ w' ≤ maxCoinsPerQuadrant ∧ (b' * w') % modulus = 1999 := by
  sorry

end NUMINAMATH_CALUDE_stating_max_remaining_coins_product_mod_l3026_302698


namespace NUMINAMATH_CALUDE_rectangle_rotation_path_length_l3026_302630

/-- The length of the path traveled by point A in a rectangle ABCD undergoing three 90° rotations -/
theorem rectangle_rotation_path_length (AB BC : ℝ) (hAB : AB = 3) (hBC : BC = 8) :
  let diagonal := Real.sqrt (AB^2 + BC^2)
  let first_rotation := (1/2) * π * diagonal
  let second_rotation := (3/2) * π
  let third_rotation := 4 * π
  first_rotation + second_rotation + third_rotation = ((1/2) * Real.sqrt 73 + 11/2) * π :=
sorry

end NUMINAMATH_CALUDE_rectangle_rotation_path_length_l3026_302630


namespace NUMINAMATH_CALUDE_statement_contradiction_l3026_302633

-- Define the possible types of speakers
inductive Speaker
| Knight
| Liar

-- Define the statement made by A
def statement (s : Speaker) : Prop :=
  match s with
  | Speaker.Knight => s = Speaker.Liar ∨ 2 + 2 = 5
  | Speaker.Liar => ¬(s = Speaker.Liar ∨ 2 + 2 = 5)

-- Theorem stating that the conditions lead to a contradiction
theorem statement_contradiction :
  ¬∃ (s : Speaker), statement s :=
by
  sorry


end NUMINAMATH_CALUDE_statement_contradiction_l3026_302633


namespace NUMINAMATH_CALUDE_fraction_sum_approximation_l3026_302646

theorem fraction_sum_approximation : 
  let sum := (2007 : ℚ) / 2999 + 8001 / 5998 + 2001 / 3999 + 4013 / 7997 + 10007 / 15999 + 2803 / 11998
  5.99 < sum ∧ sum < 6.01 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_approximation_l3026_302646


namespace NUMINAMATH_CALUDE_banana_groups_l3026_302628

theorem banana_groups (total_bananas : ℕ) (bananas_per_group : ℕ) 
  (h1 : total_bananas = 203) 
  (h2 : bananas_per_group = 29) : 
  (total_bananas / bananas_per_group : ℕ) = 7 := by
  sorry

end NUMINAMATH_CALUDE_banana_groups_l3026_302628


namespace NUMINAMATH_CALUDE_units_digit_of_7_pow_3_pow_5_l3026_302652

theorem units_digit_of_7_pow_3_pow_5 : 7^(3^5) % 10 = 3 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_7_pow_3_pow_5_l3026_302652


namespace NUMINAMATH_CALUDE_centipede_sock_shoe_orders_l3026_302624

/-- Represents the number of legs of the centipede -/
def num_legs : ℕ := 10

/-- Represents the total number of items (socks and shoes) -/
def total_items : ℕ := 2 * num_legs

/-- Represents the number of valid orders to put on socks and shoes -/
def valid_orders : ℕ := Nat.factorial total_items / (2^num_legs)

/-- Theorem stating the number of valid orders for the centipede to put on socks and shoes -/
theorem centipede_sock_shoe_orders :
  valid_orders = Nat.factorial total_items / (2^num_legs) :=
by sorry

end NUMINAMATH_CALUDE_centipede_sock_shoe_orders_l3026_302624


namespace NUMINAMATH_CALUDE_prob_jill_draws_spade_l3026_302650

/-- Represents a standard deck of cards -/
def StandardDeck : ℕ := 52

/-- Represents the number of spades in a standard deck -/
def NumSpades : ℕ := 13

/-- Probability of drawing a spade from a standard deck -/
def ProbSpade : ℚ := NumSpades / StandardDeck

/-- Probability of not drawing a spade from a standard deck -/
def ProbNotSpade : ℚ := 1 - ProbSpade

/-- Probability that Jill draws a spade in a single round -/
def ProbJillSpadeInRound : ℚ := ProbNotSpade * ProbSpade

/-- Probability that neither Jack nor Jill draws a spade in a round -/
def ProbNoSpadeInRound : ℚ := ProbNotSpade * ProbNotSpade

theorem prob_jill_draws_spade :
  (ProbJillSpadeInRound / (1 - ProbNoSpadeInRound)) = 3 / 7 :=
sorry

end NUMINAMATH_CALUDE_prob_jill_draws_spade_l3026_302650


namespace NUMINAMATH_CALUDE_seconds_in_day_l3026_302686

/-- Represents the number of minutes in a day on the island of Misfortune -/
def minutes_per_day : ℕ := 77

/-- Represents the number of seconds in an hour on the island of Misfortune -/
def seconds_per_hour : ℕ := 91

/-- Theorem stating that there are 1001 seconds in a day on the island of Misfortune -/
theorem seconds_in_day : 
  ∃ (hours_per_day minutes_per_hour seconds_per_minute : ℕ), 
    hours_per_day * minutes_per_hour = minutes_per_day ∧
    minutes_per_hour * seconds_per_minute = seconds_per_hour ∧
    hours_per_day * minutes_per_hour * seconds_per_minute = 1001 :=
by
  sorry


end NUMINAMATH_CALUDE_seconds_in_day_l3026_302686


namespace NUMINAMATH_CALUDE_tv_weather_forecast_is_random_l3026_302668

/-- Represents an event in probability theory -/
structure Event where
  (description : String)

/-- Classifies an event as random, certain, or impossible -/
inductive EventClass
  | Random
  | Certain
  | Impossible

/-- An event is random if it can lead to different outcomes, doesn't have a guaranteed outcome, and is feasible to occur -/
def is_random_event (e : Event) : Prop :=
  (∃ (outcome1 outcome2 : String), outcome1 ≠ outcome2) ∧
  ¬(∃ (guaranteed_outcome : String), true) ∧
  (∃ (possible_occurrence : Bool), possible_occurrence = true)

/-- The main theorem: Turning on the TV and watching the weather forecast is a random event -/
theorem tv_weather_forecast_is_random :
  let e : Event := { description := "turning on the TV and watching the weather forecast" }
  is_random_event e → EventClass.Random = EventClass.Random :=
by
  sorry

end NUMINAMATH_CALUDE_tv_weather_forecast_is_random_l3026_302668


namespace NUMINAMATH_CALUDE_line_through_P_and_origin_line_through_P_perpendicular_to_l₃_l3026_302692

-- Define the lines
def l₁ (x y : ℝ) : Prop := 3*x + 4*y - 2 = 0
def l₂ (x y : ℝ) : Prop := 2*x + y + 2 = 0
def l₃ (x y : ℝ) : Prop := x - 2*y - 1 = 0

-- Define the intersection point P
def P : ℝ × ℝ := (-2, 2)

-- Theorem for the first line
theorem line_through_P_and_origin : 
  ∃ (m b : ℝ), ∀ (x y : ℝ), 
    (x = P.1 ∧ y = P.2 ∨ x = 0 ∧ y = 0) → 
    y = m*x + b ∧ 
    (∀ (x y : ℝ), y = m*x + b ↔ x + y = 0) :=
sorry

-- Theorem for the second line
theorem line_through_P_perpendicular_to_l₃ : 
  ∃ (m b : ℝ), ∀ (x y : ℝ),
    (x = P.1 ∧ y = P.2) →
    y = m*x + b ∧
    m * (1/2) = -1 ∧
    (∀ (x y : ℝ), y = m*x + b ↔ 2*x + y + 2 = 0) :=
sorry

end NUMINAMATH_CALUDE_line_through_P_and_origin_line_through_P_perpendicular_to_l₃_l3026_302692


namespace NUMINAMATH_CALUDE_solution_difference_l3026_302649

-- Define the function f
def f (c₁ c₂ c₃ : ℕ) (x : ℝ) : ℝ :=
  (x^2 - 6*x + c₁) * (x^2 - 6*x + c₂) * (x^2 - 6*x + c₃)

-- Define the set M
def M (c₁ c₂ c₃ : ℕ) : Set ℕ :=
  {x : ℕ | f c₁ c₂ c₃ x = 0}

-- State the theorem
theorem solution_difference (c₁ c₂ c₃ : ℕ) :
  (c₁ ≥ c₂) → (c₂ ≥ c₃) →
  (∃ x₁ x₂ x₃ x₄ x₅ : ℕ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₁ ≠ x₅ ∧
                         x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₂ ≠ x₅ ∧
                         x₃ ≠ x₄ ∧ x₃ ≠ x₅ ∧
                         x₄ ≠ x₅ ∧
                         M c₁ c₂ c₃ = {x₁, x₂, x₃, x₄, x₅}) →
  c₁ - c₃ = 4 :=
by sorry

end NUMINAMATH_CALUDE_solution_difference_l3026_302649


namespace NUMINAMATH_CALUDE_count_checkered_rectangles_l3026_302606

/-- The number of gray cells in the picture -/
def total_gray_cells : ℕ := 40

/-- The number of blue cells -/
def blue_cells : ℕ := 36

/-- The number of red cells -/
def red_cells : ℕ := 4

/-- The number of rectangles containing each blue cell -/
def rectangles_per_blue : ℕ := 4

/-- The number of rectangles containing each red cell -/
def rectangles_per_red : ℕ := 8

/-- The total number of checkered rectangles containing exactly one gray cell -/
def total_rectangles : ℕ := blue_cells * rectangles_per_blue + red_cells * rectangles_per_red

theorem count_checkered_rectangles : total_rectangles = 176 := by
  sorry

end NUMINAMATH_CALUDE_count_checkered_rectangles_l3026_302606


namespace NUMINAMATH_CALUDE_sandy_work_hours_l3026_302689

theorem sandy_work_hours (total_hours : ℕ) (num_days : ℕ) (hours_per_day : ℕ) : 
  total_hours = 45 → 
  num_days = 5 → 
  total_hours = num_days * hours_per_day → 
  hours_per_day = 9 := by
  sorry

end NUMINAMATH_CALUDE_sandy_work_hours_l3026_302689


namespace NUMINAMATH_CALUDE_alloy_mixture_l3026_302664

/-- The amount of alloy A in kg -/
def alloy_A : ℝ := 130

/-- The ratio of lead to tin in alloy A -/
def ratio_A : ℚ := 2/3

/-- The ratio of tin to copper in alloy B -/
def ratio_B : ℚ := 3/4

/-- The amount of tin in the new alloy in kg -/
def tin_new : ℝ := 146.57

/-- The amount of alloy B mixed with alloy A in kg -/
def alloy_B : ℝ := 160.33

theorem alloy_mixture :
  alloy_B * (ratio_B / (1 + ratio_B)) + alloy_A * (ratio_A / (1 + ratio_A)) = tin_new := by
  sorry

end NUMINAMATH_CALUDE_alloy_mixture_l3026_302664


namespace NUMINAMATH_CALUDE_distance_between_A_and_C_l3026_302655

-- Define a type for points on a line
structure Point := (x : ℝ)

-- Define a function to calculate distance between two points
def distance (p q : Point) : ℝ := |p.x - q.x|

-- State the theorem
theorem distance_between_A_and_C 
  (A B C : Point) 
  (on_same_line : ∃ (k : ℝ), B.x = k * A.x + (1 - k) * C.x)
  (AB_distance : distance A B = 5)
  (BC_distance : distance B C = 4) :
  distance A C = 1 ∨ distance A C = 9 := by
sorry


end NUMINAMATH_CALUDE_distance_between_A_and_C_l3026_302655


namespace NUMINAMATH_CALUDE_intersection_parallel_to_l_l3026_302666

structure GeometricSpace where
  Line : Type
  Plane : Type
  perpendicular : Line → Line → Prop
  perpendicular_line_plane : Line → Plane → Prop
  in_plane : Line → Plane → Prop
  skew : Line → Line → Prop
  intersect : Plane → Plane → Prop
  parallel : Line → Line → Prop
  intersection_line : Plane → Plane → Line

variable (S : GeometricSpace)

theorem intersection_parallel_to_l 
  (m n l : S.Line) (α β : S.Plane) :
  S.skew m n →
  S.perpendicular_line_plane m α →
  S.perpendicular_line_plane n β →
  S.perpendicular l m →
  S.perpendicular l n →
  ¬S.in_plane l α →
  ¬S.in_plane l β →
  S.intersect α β ∧ S.parallel l (S.intersection_line α β) :=
sorry

end NUMINAMATH_CALUDE_intersection_parallel_to_l_l3026_302666


namespace NUMINAMATH_CALUDE_smallest_winning_number_l3026_302695

def game_winner (M : ℕ) : Prop :=
  M ≤ 999 ∧ 
  2 * M < 1000 ∧ 
  6 * M < 1000 ∧ 
  12 * M < 1000 ∧ 
  36 * M > 999

theorem smallest_winning_number : 
  ∃ (M : ℕ), game_winner M ∧ ∀ (N : ℕ), N < M → ¬game_winner N :=
sorry

end NUMINAMATH_CALUDE_smallest_winning_number_l3026_302695


namespace NUMINAMATH_CALUDE_scout_troop_profit_l3026_302693

-- Define the number of candy bars
def num_bars : ℕ := 1500

-- Define the buying price
def buy_price : ℚ := 3 / 4

-- Define the selling price
def sell_price : ℚ := 2 / 3

-- Calculate the total cost
def total_cost : ℚ := num_bars * buy_price

-- Calculate the total revenue
def total_revenue : ℚ := num_bars * sell_price

-- Calculate the profit
def profit : ℚ := total_revenue - total_cost

-- Theorem to prove
theorem scout_troop_profit :
  profit = -125 := by sorry

end NUMINAMATH_CALUDE_scout_troop_profit_l3026_302693


namespace NUMINAMATH_CALUDE_integral_2x_over_half_pi_l3026_302613

theorem integral_2x_over_half_pi : ∫ x in (0)..(π/2), 2*x = π^2 / 4 := by
  sorry

end NUMINAMATH_CALUDE_integral_2x_over_half_pi_l3026_302613


namespace NUMINAMATH_CALUDE_pizza_group_size_l3026_302691

theorem pizza_group_size (slices_per_person : ℕ) (slices_per_pizza : ℕ) (num_pizzas : ℕ)
  (h1 : slices_per_person = 3)
  (h2 : slices_per_pizza = 9)
  (h3 : num_pizzas = 6) :
  (num_pizzas * (slices_per_pizza / slices_per_person)) = 18 :=
by sorry

end NUMINAMATH_CALUDE_pizza_group_size_l3026_302691


namespace NUMINAMATH_CALUDE_snowball_difference_l3026_302678

def charlie_snowballs : ℕ := 50
def lucy_snowballs : ℕ := 19

theorem snowball_difference : charlie_snowballs - lucy_snowballs = 31 := by
  sorry

end NUMINAMATH_CALUDE_snowball_difference_l3026_302678


namespace NUMINAMATH_CALUDE_power_of_two_ge_square_l3026_302609

theorem power_of_two_ge_square (n : ℕ) (h : n ≥ 4) : 2^n ≥ n^2 := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_ge_square_l3026_302609


namespace NUMINAMATH_CALUDE_system_solution_transformation_l3026_302602

theorem system_solution_transformation 
  (a₁ a₂ b₁ b₂ c₁ c₂ : ℝ) 
  (h : ∃ (x y : ℝ), x = 3 ∧ y = 4 ∧ a₁ * x + b₁ * y = c₁ ∧ a₂ * x + b₂ * y = c₂) :
  ∃ (x y : ℝ), x = 5 ∧ y = 5 ∧ 3 * a₁ * x + 4 * b₁ * y = 5 * c₁ ∧ 3 * a₂ * x + 4 * b₂ * y = 5 * c₂ :=
by sorry

end NUMINAMATH_CALUDE_system_solution_transformation_l3026_302602


namespace NUMINAMATH_CALUDE_max_a_value_l3026_302677

theorem max_a_value (a b : ℕ) (ha : 1 < a) (hb : a < b) :
  (∃! x : ℝ, -2 * x + 4033 = |x - 1| + |x + a| + |x - b|) →
  (∀ a' : ℕ, 1 < a' ∧ a' < b ∧ (∃! x : ℝ, -2 * x + 4033 = |x - 1| + |x + a'| + |x - b|) → a' ≤ a) ∧
  a = 4031 :=
by sorry

end NUMINAMATH_CALUDE_max_a_value_l3026_302677


namespace NUMINAMATH_CALUDE_solution_set_abs_inequality_l3026_302608

theorem solution_set_abs_inequality (x : ℝ) :
  (Set.Icc 1 3 : Set ℝ) = {x | |2 - x| ≤ 1} :=
by sorry

end NUMINAMATH_CALUDE_solution_set_abs_inequality_l3026_302608


namespace NUMINAMATH_CALUDE_book_cost_price_l3026_302637

theorem book_cost_price (selling_price : ℝ) (profit_percentage : ℝ) (cost_price : ℝ) : 
  selling_price = 260 ∧ profit_percentage = 20 → 
  selling_price = cost_price * (1 + profit_percentage / 100) →
  cost_price = 216.67 := by
  sorry

end NUMINAMATH_CALUDE_book_cost_price_l3026_302637


namespace NUMINAMATH_CALUDE_symmetric_points_sum_power_l3026_302651

/-- Two points are symmetric about the x-axis if their x-coordinates are equal
    and their y-coordinates are opposite. -/
def symmetric_about_x_axis (x1 y1 x2 y2 : ℝ) : Prop :=
  x1 = x2 ∧ y1 = -y2

theorem symmetric_points_sum_power (a b : ℝ) :
  symmetric_about_x_axis (a - 1) 5 2 (b - 1) →
  (a + b) ^ 2005 = -1 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_sum_power_l3026_302651


namespace NUMINAMATH_CALUDE_unique_number_with_special_divisor_property_l3026_302676

theorem unique_number_with_special_divisor_property :
  ∃! (N : ℕ), 
    N > 0 ∧
    (∃ (m : ℕ), 
      m > 0 ∧ 
      m < N ∧
      N % m = 0 ∧
      (∀ (d : ℕ), d > 0 → d < N → N % d = 0 → d ≤ m) ∧
      (∃ (k : ℕ), N + m = 10^k)) ∧
    N = 75 :=
by sorry

end NUMINAMATH_CALUDE_unique_number_with_special_divisor_property_l3026_302676


namespace NUMINAMATH_CALUDE_constant_function_inequality_l3026_302638

theorem constant_function_inequality (f : ℝ → ℝ) :
  (∀ x y z : ℝ, f (x + y) + f (y + z) + f (z + x) ≥ 3 * f (x + 2*y + 3*z)) →
  ∃ c : ℝ, ∀ x : ℝ, f x = c :=
by sorry

end NUMINAMATH_CALUDE_constant_function_inequality_l3026_302638


namespace NUMINAMATH_CALUDE_target_probability_l3026_302670

def prob_A : ℚ := 2/3
def prob_B : ℚ := 1/2
def num_shots : ℕ := 4

theorem target_probability : 
  let prob_A_2 := (num_shots.choose 2) * prob_A^2 * (1 - prob_A)^2
  let prob_B_3 := (num_shots.choose 3) * prob_B^3 * (1 - prob_B)
  prob_A_2 * prob_B_3 = 2/27 := by sorry

end NUMINAMATH_CALUDE_target_probability_l3026_302670


namespace NUMINAMATH_CALUDE_inequality_implies_k_bound_l3026_302643

theorem inequality_implies_k_bound (k : ℝ) : 
  (∀ x y : ℝ, x > 0 → y > 0 → Real.sqrt x + Real.sqrt y ≤ k * Real.sqrt (2*x + y)) → 
  k ≥ Real.sqrt 6 / 2 := by
sorry

end NUMINAMATH_CALUDE_inequality_implies_k_bound_l3026_302643


namespace NUMINAMATH_CALUDE_f_compose_three_equals_43_l3026_302636

-- Define the function f
def f (n : ℕ) : ℕ :=
  if n < 5 then n^2 + 1 else 2*n + 1

-- Theorem statement
theorem f_compose_three_equals_43 : f (f (f 3)) = 43 := by
  sorry

end NUMINAMATH_CALUDE_f_compose_three_equals_43_l3026_302636


namespace NUMINAMATH_CALUDE_min_sum_of_squares_for_sum_16_l3026_302647

theorem min_sum_of_squares_for_sum_16 :
  ∀ a b c : ℕ+,
  a + b + c = 16 →
  a^2 + b^2 + c^2 ≥ 86 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_of_squares_for_sum_16_l3026_302647


namespace NUMINAMATH_CALUDE_dot_product_sum_and_a_l3026_302616

/-- Given vectors a and b in ℝ², prove that the dot product of (a + b) and a equals 1. -/
theorem dot_product_sum_and_a (a b : ℝ × ℝ) (h1 : a = (1/2, Real.sqrt 3/2)) 
    (h2 : b = (-Real.sqrt 3/2, 1/2)) : 
    (a.1 + b.1) * a.1 + (a.2 + b.2) * a.2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_dot_product_sum_and_a_l3026_302616


namespace NUMINAMATH_CALUDE_parking_lot_wheels_l3026_302629

theorem parking_lot_wheels (num_cars num_bikes : ℕ) (wheels_per_car wheels_per_bike : ℕ) :
  num_cars = 14 →
  num_bikes = 5 →
  wheels_per_car = 4 →
  wheels_per_bike = 2 →
  num_cars * wheels_per_car + num_bikes * wheels_per_bike = 66 := by
  sorry

end NUMINAMATH_CALUDE_parking_lot_wheels_l3026_302629


namespace NUMINAMATH_CALUDE_triangle_reflection_area_l3026_302656

/-- The area of the union of a triangle and its reflection --/
theorem triangle_reflection_area : 
  let A : ℝ × ℝ := (3, 4)
  let B : ℝ × ℝ := (5, -2)
  let C : ℝ × ℝ := (6, 2)
  let reflect (p : ℝ × ℝ) : ℝ × ℝ := (p.1, 2 - p.2)
  let A' := reflect A
  let B' := reflect B
  let C' := reflect C
  let area (p q r : ℝ × ℝ) : ℝ := 
    (1/2) * abs (p.1 * (q.2 - r.2) + q.1 * (r.2 - p.2) + r.1 * (p.2 - q.2))
  area A B C + area A' B' C' = 11 := by
sorry


end NUMINAMATH_CALUDE_triangle_reflection_area_l3026_302656


namespace NUMINAMATH_CALUDE_population_after_three_years_l3026_302617

def population_growth (initial : ℕ) (rate : ℚ) (additional : ℕ) : ℕ :=
  ⌊(initial : ℚ) * (1 + rate) + additional⌋.toNat

def three_year_population (initial : ℕ) (rate1 rate2 rate3 : ℚ) (add1 add2 add3 : ℕ) : ℕ :=
  let year1 := population_growth initial rate1 add1
  let year2 := population_growth year1 rate2 add2
  population_growth year2 rate3 add3

theorem population_after_three_years :
  three_year_population 14000 (12/100) (8/100) (6/100) 150 100 500 = 18728 :=
by sorry

end NUMINAMATH_CALUDE_population_after_three_years_l3026_302617


namespace NUMINAMATH_CALUDE_rectangle_circle_tangent_l3026_302611

theorem rectangle_circle_tangent (r : ℝ) (w l : ℝ) : 
  r = 6 →  -- Circle radius is 6 cm
  w = 2 * r →  -- Width of rectangle is diameter of circle
  l * w = 3 * (π * r^2) →  -- Area of rectangle is 3 times area of circle
  l = 9 * π :=  -- Length of longer side is 9π cm
by
  sorry

end NUMINAMATH_CALUDE_rectangle_circle_tangent_l3026_302611


namespace NUMINAMATH_CALUDE_sequence_ratio_l3026_302615

/-- Given an arithmetic sequence and a geometric sequence with specific properties, 
    prove that the ratio of the difference of two terms in the arithmetic sequence 
    to a term in the geometric sequence is 1/2. -/
theorem sequence_ratio (a₁ a₂ b₁ b₂ b₃ : ℝ) : 
  ((-2 : ℝ) - a₁ = a₁ - a₂) ∧ (a₂ - (-8 : ℝ) = a₁ - a₂) ∧  -- Arithmetic sequence condition
  (b₁ / (-2 : ℝ) = b₂ / b₁) ∧ (b₂ / b₁ = b₃ / b₂) ∧ (b₃ / b₂ = (-8 : ℝ) / b₃) →  -- Geometric sequence condition
  (a₂ - a₁) / b₂ = (1 : ℝ) / 2 := by sorry

end NUMINAMATH_CALUDE_sequence_ratio_l3026_302615


namespace NUMINAMATH_CALUDE_quadratic_roots_property_l3026_302661

theorem quadratic_roots_property (m : ℝ) (x₁ x₂ : ℝ) : 
  (∀ x, x^2 + 2*m*x + m^2 - m = 0 ↔ x = x₁ ∨ x = x₂) →
  x₁ * x₂ = 2 →
  (x₁^2 + 2) * (x₂^2 + 2) = 32 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_property_l3026_302661


namespace NUMINAMATH_CALUDE_min_surface_area_cubic_pile_l3026_302675

/-- Represents the dimensions of a brick -/
structure BrickDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the surface area of a cube given its side length -/
def cubeSurfaceArea (sideLength : ℕ) : ℕ :=
  6 * sideLength * sideLength

/-- Theorem: The minimum surface area of a cubic pile of bricks -/
theorem min_surface_area_cubic_pile (brick : BrickDimensions)
  (h1 : brick.length = 25)
  (h2 : brick.width = 15)
  (h3 : brick.height = 5) :
  ∃ (sideLength : ℕ), cubeSurfaceArea sideLength = 33750 ∧
    ∀ (otherSideLength : ℕ), cubeSurfaceArea otherSideLength ≥ 33750 := by
  sorry

end NUMINAMATH_CALUDE_min_surface_area_cubic_pile_l3026_302675


namespace NUMINAMATH_CALUDE_triangle_area_l3026_302653

noncomputable def f (x : ℝ) : ℝ := Real.cos x * (Real.sin x - Real.sqrt 3 * Real.cos x)

theorem triangle_area (A B C : ℝ) (a b c : ℝ) :
  f (A / 2) = -Real.sqrt 3 / 2 →
  a = 3 →
  b + c = 2 * Real.sqrt 3 →
  (1 / 2) * b * c * Real.sin A = Real.sqrt 3 / 4 :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_l3026_302653
