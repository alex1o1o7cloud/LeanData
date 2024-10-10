import Mathlib

namespace expression_value_l862_86280

theorem expression_value : (3^2 - 3) - (5^2 - 5) * 2 + (6^2 - 6) = -4 := by
  sorry

end expression_value_l862_86280


namespace inequality_proof_l862_86276

theorem inequality_proof (x y a : ℝ) (hx : x > 0) (hy : y > 0) (ha : x + y = 2 * a) :
  x^3 * y^3 * (x^2 + y^2)^2 ≤ 4 * a^10 ∧
  (x^3 * y^3 * (x^2 + y^2)^2 = 4 * a^10 ↔ x = a ∧ y = a) :=
by sorry

end inequality_proof_l862_86276


namespace dormitory_to_city_distance_l862_86281

theorem dormitory_to_city_distance :
  ∀ (D : ℝ),
  (1 / 3 : ℝ) * D + (3 / 5 : ℝ) * D + 2 = D →
  D = 30 := by
sorry

end dormitory_to_city_distance_l862_86281


namespace first_test_score_l862_86224

theorem first_test_score (second_score average : ℝ) (h1 : second_score = 84) (h2 : average = 81) :
  let first_score := 2 * average - second_score
  first_score = 78 := by
sorry

end first_test_score_l862_86224


namespace odd_function_range_l862_86251

/-- A function f is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem odd_function_range (a : ℝ) (f : ℝ → ℝ) 
    (h_odd : IsOdd f)
    (h_neg : ∀ x, x < 0 → f x = 9*x + a^2/x + 7)
    (h_pos : ∀ x, x ≥ 0 → f x ≥ a + 1) :
  a ≤ -8/7 := by
  sorry

end odd_function_range_l862_86251


namespace ladies_walking_ratio_l862_86273

/-- Given two ladies walking in Central Park, prove that the ratio of their distances is 2:1 -/
theorem ladies_walking_ratio :
  ∀ (distance1 distance2 : ℝ),
  distance2 = 4 →
  distance1 + distance2 = 12 →
  distance1 / distance2 = 2 := by
sorry

end ladies_walking_ratio_l862_86273


namespace probability_in_specific_sequence_l862_86210

/-- Represents an arithmetic sequence with given parameters -/
structure ArithmeticSequence where
  first_term : ℕ
  common_difference : ℕ
  last_term : ℕ

/-- Calculates the number of terms in the arithmetic sequence -/
def number_of_terms (seq : ArithmeticSequence) : ℕ :=
  (seq.last_term - seq.first_term) / seq.common_difference + 1

/-- Calculates the number of terms divisible by 6 in the sequence -/
def divisible_by_six (seq : ArithmeticSequence) : ℕ :=
  (number_of_terms seq) / 3

/-- The probability of selecting a number divisible by 6 from the sequence -/
def probability_divisible_by_six (seq : ArithmeticSequence) : ℚ :=
  (divisible_by_six seq : ℚ) / (number_of_terms seq)

theorem probability_in_specific_sequence :
  let seq := ArithmeticSequence.mk 50 4 998
  probability_divisible_by_six seq = 79 / 238 := by
  sorry

end probability_in_specific_sequence_l862_86210


namespace object_length_doubles_on_day_two_l862_86246

/-- Calculates the length multiplier after n days -/
def lengthMultiplier (n : ℕ) : ℚ :=
  (n + 2 : ℚ) / 2

theorem object_length_doubles_on_day_two :
  ∃ n : ℕ, lengthMultiplier n = 2 ∧ n = 2 :=
sorry

end object_length_doubles_on_day_two_l862_86246


namespace article_supports_statements_l862_86214

/-- Represents the content of the given article about Chinese literature and Mo Yan's Nobel Prize -/
def ArticleContent : Type := sorry

/-- Represents the manifestations of the marginalization of literature since the 1990s -/
def LiteratureMarginalization (content : ArticleContent) : Prop := sorry

/-- Represents the effects of mentioning Mo Yan's award multiple times -/
def MoYanAwardEffects (content : ArticleContent) : Prop := sorry

/-- Represents ways to better develop pure literature -/
def PureLiteratureDevelopment (content : ArticleContent) : Prop := sorry

/-- The main theorem stating that the article supports the given statements -/
theorem article_supports_statements (content : ArticleContent) :
  LiteratureMarginalization content ∧
  MoYanAwardEffects content ∧
  PureLiteratureDevelopment content :=
sorry

end article_supports_statements_l862_86214


namespace second_group_size_l862_86200

/-- Represents a choir split into three groups -/
structure Choir :=
  (total : ℕ)
  (group1 : ℕ)
  (group2 : ℕ)
  (group3 : ℕ)
  (sum_eq_total : group1 + group2 + group3 = total)

/-- Theorem: Given a choir with 70 total members, 25 in the first group,
    and 15 in the third group, the second group must have 30 members -/
theorem second_group_size (c : Choir)
  (h1 : c.total = 70)
  (h2 : c.group1 = 25)
  (h3 : c.group3 = 15) :
  c.group2 = 30 := by
  sorry

end second_group_size_l862_86200


namespace buffer_water_requirement_l862_86239

/-- Given a buffer solution where water constitutes 1/3 of the total volume,
    prove that 0.72 liters of the buffer solution requires 0.24 liters of water. -/
theorem buffer_water_requirement (total_volume : ℝ) (water_fraction : ℝ) 
    (h1 : total_volume = 0.72)
    (h2 : water_fraction = 1/3) : 
  total_volume * water_fraction = 0.24 := by
  sorry

end buffer_water_requirement_l862_86239


namespace f_12_equals_190_l862_86294

def f (n : ℤ) : ℤ := n^2 + 2*n + 22

theorem f_12_equals_190 : f 12 = 190 := by
  sorry

end f_12_equals_190_l862_86294


namespace airplane_rows_l862_86289

/-- 
Given an airplane with the following conditions:
- Each row has 8 seats
- Only 3/4 of the seats in each row can be occupied
- There are 24 unoccupied seats on the plane

This theorem proves that the number of rows on the airplane is 12.
-/
theorem airplane_rows : 
  ∀ (rows : ℕ), 
  (8 : ℚ) * rows - (3 / 4 : ℚ) * 8 * rows = 24 → 
  rows = 12 := by
sorry

end airplane_rows_l862_86289


namespace nail_trimming_customers_l862_86211

/-- The number of customers who had their nails trimmed -/
def number_of_customers (total_sounds : ℕ) (nails_per_person : ℕ) : ℕ :=
  total_sounds / nails_per_person

/-- Theorem: Given 60 nail trimming sounds and 20 nails per person, 
    the number of customers who had their nails trimmed is 3 -/
theorem nail_trimming_customers :
  number_of_customers 60 20 = 3 := by
  sorry

end nail_trimming_customers_l862_86211


namespace ellipse_properties_l862_86234

/-- The ellipse C with given properties -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_ab : a > b
  h_b_pos : b > 0
  h_ecc : (a^2 - b^2) / a^2 = 5 / 9
  h_minor : b = 2

/-- The condition for the line x = m -/
def line_condition (C : Ellipse) (m : ℝ) : Prop :=
  ∀ x y : ℝ, x^2 / C.a^2 + y^2 / C.b^2 = 1 →
  x ≠ -C.a ∧ x ≠ C.a →
  (x - C.a) * (5 / 9 * m - 13 / 3) = 0

theorem ellipse_properties (C : Ellipse) :
  C.a^2 = 9 ∧ C.b^2 = 4 ∧ line_condition C (39 / 5) := by sorry

end ellipse_properties_l862_86234


namespace cone_volume_with_inscribed_sphere_l862_86201

/-- The volume of a cone with an inscribed sphere -/
theorem cone_volume_with_inscribed_sphere (r α : ℝ) (hr : r > 0) (hα : 0 < α ∧ α < π / 2) :
  ∃ V : ℝ, V = -π * r^3 * Real.tan (2 * α) / (24 * Real.cos α ^ 6) :=
by sorry

end cone_volume_with_inscribed_sphere_l862_86201


namespace complement_intersection_empty_l862_86202

def U : Set Nat := {1, 2, 3, 4, 5}
def M : Set Nat := {1, 3, 4}
def N : Set Nat := {2, 4, 5}

theorem complement_intersection_empty :
  (U \ M) ∩ (U \ N) = ∅ := by
  sorry

end complement_intersection_empty_l862_86202


namespace angle_expression_simplification_l862_86227

theorem angle_expression_simplification (a : ℝ) (α : ℝ) (h1 : a < 0) 
  (h2 : Real.tan α = 2) (h3 : Real.cos α = -Real.sqrt 5 / 5) :
  (Real.sin (π - α) * Real.cos (2 * π - α) * Real.sin (-α + 3 * π / 2)) / 
  (Real.tan (-α - π) * Real.sin (-π - α)) = 1 / 10 := by
sorry


end angle_expression_simplification_l862_86227


namespace smallest_n_complex_equation_l862_86247

theorem smallest_n_complex_equation (n : ℕ) (a b : ℝ) : 
  n > 3 ∧ 
  0 < a ∧ 
  0 < b ∧ 
  (∀ k : ℕ, 3 < k ∧ k < n → ¬∃ x y : ℝ, 0 < x ∧ 0 < y ∧ (x + y * I) ^ k + x = (x - y * I) ^ k + y) ∧
  (a + b * I) ^ n + a = (a - b * I) ^ n + b →
  b / a = 1 := by sorry

end smallest_n_complex_equation_l862_86247


namespace bill_drew_four_pentagons_l862_86257

/-- The number of pentagons Bill drew -/
def num_pentagons : ℕ := sorry

/-- The total number of lines Bill drew -/
def total_lines : ℕ := 88

/-- The number of triangles Bill drew -/
def num_triangles : ℕ := 12

/-- The number of squares Bill drew -/
def num_squares : ℕ := 8

/-- The number of sides in a triangle -/
def triangle_sides : ℕ := 3

/-- The number of sides in a square -/
def square_sides : ℕ := 4

/-- The number of sides in a pentagon -/
def pentagon_sides : ℕ := 5

theorem bill_drew_four_pentagons :
  num_pentagons = 4 :=
by
  sorry

end bill_drew_four_pentagons_l862_86257


namespace friend_meeting_point_l862_86213

theorem friend_meeting_point (trail_length : ℝ) (speed_difference : ℝ) 
  (h1 : trail_length = 60)
  (h2 : speed_difference = 0.4) : 
  let faster_friend_distance := trail_length * (1 + speed_difference) / (2 + speed_difference)
  faster_friend_distance = 35 := by
  sorry

end friend_meeting_point_l862_86213


namespace minimum_points_to_win_l862_86248

/-- Represents the points earned in a single race -/
inductive RaceResult
| First  : RaceResult
| Second : RaceResult
| Third  : RaceResult
| Other  : RaceResult

/-- Converts a race result to points -/
def pointsForResult (result : RaceResult) : Nat :=
  match result with
  | RaceResult.First  => 4
  | RaceResult.Second => 2
  | RaceResult.Third  => 1
  | RaceResult.Other  => 0

/-- Calculates total points for a series of race results -/
def totalPoints (results : List RaceResult) : Nat :=
  results.map pointsForResult |>.sum

/-- Represents all possible combinations of race results for four races -/
def allPossibleResults : List (List RaceResult) :=
  sorry

theorem minimum_points_to_win (results : List RaceResult) :
  (results.length = 4) →
  (totalPoints results ≥ 15) →
  (∀ other : List RaceResult, other.length = 4 → totalPoints other < totalPoints results) :=
sorry

end minimum_points_to_win_l862_86248


namespace complex_equation_solution_l862_86208

theorem complex_equation_solution (z : ℂ) : z * Complex.I = 3 - 4 * Complex.I → z = -4 - 3 * Complex.I := by
  sorry

end complex_equation_solution_l862_86208


namespace annual_increase_fraction_l862_86263

theorem annual_increase_fraction (initial_amount final_amount : ℝ) (f : ℝ) 
  (h1 : initial_amount = 65000)
  (h2 : final_amount = 82265.625)
  (h3 : final_amount = initial_amount * (1 + f)^2) :
  f = 0.125 := by
  sorry

end annual_increase_fraction_l862_86263


namespace square_value_l862_86271

theorem square_value (triangle circle square : ℕ) 
  (h1 : triangle + circle = square) 
  (h2 : triangle + circle + square = 100) : 
  square = 50 := by sorry

end square_value_l862_86271


namespace number_satisfying_equation_l862_86262

theorem number_satisfying_equation : ∃ x : ℝ, x^2 + 4 = 5*x ∧ (x = 4 ∨ x = 1) := by sorry

end number_satisfying_equation_l862_86262


namespace a4_value_l862_86218

theorem a4_value (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, x^5 = a₀ + a₁*(x+1) + a₂*(x+1)^2 + a₃*(x+1)^3 + a₄*(x+1)^4 + a₅*(x+1)^5) →
  a₄ = -5 := by
sorry

end a4_value_l862_86218


namespace age_problem_l862_86260

theorem age_problem (my_age : ℕ) : 
  (∃ (older_brother younger_sister youngest_brother : ℕ),
    -- Ten years ago, my older brother was exactly twice my age
    older_brother = 2 * (my_age - 10) ∧
    -- Ten years ago, my younger sister's age was half of mine
    younger_sister = (my_age - 10) / 2 ∧
    -- Ten years ago, my youngest brother was the same age as my sister
    youngest_brother = younger_sister ∧
    -- In fifteen years, the combined age of the four of us will be 110
    (my_age + 15) + (older_brother + 15) + (younger_sister + 15) + (youngest_brother + 15) = 110) →
  my_age = 16 :=
by sorry

end age_problem_l862_86260


namespace factor_expression_l862_86235

theorem factor_expression (x y : ℝ) : 60 * x^2 + 40 * y = 20 * (3 * x^2 + 2 * y) := by
  sorry

end factor_expression_l862_86235


namespace reduction_equivalence_original_value_proof_l862_86204

theorem reduction_equivalence (original : ℝ) (reduced : ℝ) : 
  reduced = original * (1 / 1000) ↔ reduced = original * 0.001 :=
by sorry

theorem original_value_proof : 
  ∃ (original : ℝ), 16.9 * (1 / 1000) = 0.0169 ∧ original = 16.9 :=
by sorry

end reduction_equivalence_original_value_proof_l862_86204


namespace misread_number_correction_l862_86244

theorem misread_number_correction (n : ℕ) (initial_avg correct_avg wrong_num : ℝ) (correct_num : ℝ) : 
  n = 10 →
  initial_avg = 15 →
  wrong_num = 26 →
  correct_avg = 16 →
  n * initial_avg + (correct_num - wrong_num) = n * correct_avg →
  correct_num = 36 := by
sorry

end misread_number_correction_l862_86244


namespace expression_simplification_l862_86207

theorem expression_simplification (x y : ℝ) : 
  3 * x + 4 * y^2 + 2 - (5 - 3 * x - 2 * y^2) = 6 * x + 6 * y^2 - 3 := by
  sorry

end expression_simplification_l862_86207


namespace grandmother_five_times_lingling_age_l862_86272

/-- Represents the current age of Lingling -/
def lingling_age : ℕ := 8

/-- Represents the current age of the grandmother -/
def grandmother_age : ℕ := 60

/-- Represents the number of years after which the grandmother's age will be 5 times Lingling's age -/
def years_until_five_times : ℕ := 5

/-- Proves that after 'years_until_five_times' years, the grandmother's age will be 5 times Lingling's age -/
theorem grandmother_five_times_lingling_age : 
  grandmother_age + years_until_five_times = 5 * (lingling_age + years_until_five_times) := by
  sorry

end grandmother_five_times_lingling_age_l862_86272


namespace hedge_sections_count_l862_86295

def section_blocks : ℕ := 30
def block_cost : ℚ := 2
def total_cost : ℚ := 480

theorem hedge_sections_count :
  (total_cost / (section_blocks * block_cost) : ℚ) = 8 := by
  sorry

end hedge_sections_count_l862_86295


namespace plate_on_square_table_l862_86203

/-- Given a square table with a round plate, if the distances from the plate's edge
    to two adjacent sides of the table are a and b, and the distance from the plate's edge
    to the opposite side of the a measurement is c, then the distance from the plate's edge
    to the opposite side of the b measurement is a + c - b. -/
theorem plate_on_square_table (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a + c = b + (a + c - b) :=
by sorry

end plate_on_square_table_l862_86203


namespace cylinder_surface_area_l862_86259

theorem cylinder_surface_area (r h : ℝ) (base_area : ℝ) : 
  base_area = π * r^2 →
  h = 2 * r →
  2 * base_area + 2 * π * r * h = 384 * π := by
  sorry

end cylinder_surface_area_l862_86259


namespace age_ratio_problem_l862_86241

theorem age_ratio_problem (ali_age yusaf_age umar_age : ℕ) : 
  ali_age = 8 →
  ali_age = yusaf_age + 3 →
  ∃ k : ℕ, umar_age = k * yusaf_age →
  umar_age = 10 →
  umar_age / yusaf_age = 2 := by
sorry

end age_ratio_problem_l862_86241


namespace tuesday_children_count_l862_86282

/-- Represents the number of children who went to the zoo on Tuesday -/
def tuesday_children : ℕ := sorry

/-- Theorem stating that the number of children who went to the zoo on Tuesday is 4 -/
theorem tuesday_children_count : tuesday_children = 4 := by
  have monday_revenue : ℕ := 7 * 3 + 5 * 4
  have tuesday_revenue : ℕ := tuesday_children * 3 + 2 * 4
  have total_revenue : ℕ := 61
  have revenue_equation : monday_revenue + tuesday_revenue = total_revenue := sorry
  sorry

end tuesday_children_count_l862_86282


namespace rice_division_l862_86245

theorem rice_division (total_weight : ℚ) (num_containers : ℕ) (ounces_per_pound : ℕ) :
  total_weight = 25 / 2 →
  num_containers = 4 →
  ounces_per_pound = 16 →
  (total_weight / num_containers) * ounces_per_pound = 50 := by
sorry

end rice_division_l862_86245


namespace half_circle_is_300_clerts_l862_86286

-- Define the number of clerts in a full circle
def full_circle_clerts : ℕ := 600

-- Define a half-circle as half of a full circle
def half_circle_clerts : ℕ := full_circle_clerts / 2

-- Theorem to prove
theorem half_circle_is_300_clerts : half_circle_clerts = 300 := by
  sorry

end half_circle_is_300_clerts_l862_86286


namespace water_speed_in_swimming_problem_l862_86279

/-- Proves that the speed of water is 4 km/h given the conditions of the swimming problem -/
theorem water_speed_in_swimming_problem : 
  ∀ (still_water_speed : ℝ) (distance : ℝ) (time : ℝ) (water_speed : ℝ),
    still_water_speed = 8 →
    distance = 8 →
    time = 2 →
    distance = (still_water_speed - water_speed) * time →
    water_speed = 4 := by
  sorry

end water_speed_in_swimming_problem_l862_86279


namespace subtraction_of_fractions_l862_86275

theorem subtraction_of_fractions : (8 : ℚ) / 15 - (11 : ℚ) / 20 = -1 / 60 := by sorry

end subtraction_of_fractions_l862_86275


namespace tan_sum_pi_twelfths_l862_86253

theorem tan_sum_pi_twelfths : 
  Real.tan (π / 12) + Real.tan (5 * π / 12) = 4 * Real.sqrt 3 / 3 := by sorry

end tan_sum_pi_twelfths_l862_86253


namespace sqrt_three_squared_l862_86254

theorem sqrt_three_squared : Real.sqrt 3 ^ 2 = 3 := by sorry

end sqrt_three_squared_l862_86254


namespace volume_of_sphere_with_radius_three_l862_86277

/-- The volume of a sphere with radius 3 is 36π. -/
theorem volume_of_sphere_with_radius_three : 
  (4 / 3 : ℝ) * Real.pi * 3^3 = 36 * Real.pi := by
  sorry

end volume_of_sphere_with_radius_three_l862_86277


namespace integer_fraction_pairs_l862_86219

theorem integer_fraction_pairs : 
  ∀ a b : ℕ+, 
    (((a : ℤ)^3 * (b : ℤ) - 1) % ((a : ℤ) + 1) = 0 ∧ 
     ((b : ℤ)^3 * (a : ℤ) + 1) % ((b : ℤ) - 1) = 0) ↔ 
    ((a = 2 ∧ b = 2) ∨ (a = 1 ∧ b = 3) ∨ (a = 3 ∧ b = 3)) :=
by sorry

end integer_fraction_pairs_l862_86219


namespace max_value_on_ellipse_l862_86278

theorem max_value_on_ellipse :
  ∃ (M : ℝ), M = 7 ∧
  ∀ (x y : ℝ), (x^2 / 4 + y^2 = 1) →
  ((3/4) * x^2 + 2*x - y^2 ≤ M) :=
sorry

end max_value_on_ellipse_l862_86278


namespace min_value_expression_l862_86243

theorem min_value_expression (a b : ℝ) (h : a^2 ≥ 8*b) :
  ∃ (min : ℝ), min = (9:ℝ)/8 ∧ ∀ (x y : ℝ), x^2 ≥ 8*y →
    (1 - x)^2 + (1 - 2*y)^2 + (x - 2*y)^2 ≥ min :=
by sorry

end min_value_expression_l862_86243


namespace square_sum_reciprocals_l862_86293

theorem square_sum_reciprocals (x y : ℝ) 
  (h : 1 / x - 1 / (2 * y) = 1 / (2 * x + y)) : 
  y^2 / x^2 + x^2 / y^2 = 9 / 4 := by
  sorry

end square_sum_reciprocals_l862_86293


namespace twenty_nine_free_travelers_l862_86231

/-- Represents the promotion scenario for a travel agency -/
structure TravelPromotion where
  /-- Number of tourists who came on their own -/
  self_arrivals : ℕ
  /-- Number of tourists who didn't bring anyone -/
  no_referrals : ℕ
  /-- Total number of tourists -/
  total_tourists : ℕ

/-- Calculates the number of tourists who traveled for free -/
def free_travelers (promo : TravelPromotion) : ℕ :=
  (promo.total_tourists - promo.self_arrivals - promo.no_referrals) / 4

/-- Theorem stating that 29 tourists traveled for free -/
theorem twenty_nine_free_travelers (promo : TravelPromotion)
  (h1 : promo.self_arrivals = 13)
  (h2 : promo.no_referrals = 100)
  (h3 : promo.total_tourists = promo.self_arrivals + promo.no_referrals + 4 * (free_travelers promo)) :
  free_travelers promo = 29 := by
  sorry

#eval free_travelers { self_arrivals := 13, no_referrals := 100, total_tourists := 229 }

end twenty_nine_free_travelers_l862_86231


namespace find_N_l862_86290

theorem find_N : ∀ N : ℕ, (1 + 2 + 3) / 6 = (1988 + 1989 + 1990) / N → N = 5967 := by
  sorry

end find_N_l862_86290


namespace consecutive_integers_sum_l862_86215

theorem consecutive_integers_sum (x : ℕ) (h1 : x > 0) (h2 : x * (x + 1) = 812) : x + (x + 1) = 57 := by
  sorry

end consecutive_integers_sum_l862_86215


namespace distance_between_points_l862_86266

theorem distance_between_points : 
  ∀ (A B : ℝ), A = -1 ∧ B = 2020 → |A - B| = 2021 :=
by
  sorry

end distance_between_points_l862_86266


namespace vector_magnitude_proof_l862_86238

/-- Given a vector a = (2, 1) and another vector b such that a · b = 10 and |a + b| = 5, prove that |b| = 2√10. -/
theorem vector_magnitude_proof (b : ℝ × ℝ) :
  let a : ℝ × ℝ := (2, 1)
  (a.1 * b.1 + a.2 * b.2 = 10) →
  ((a.1 + b.1)^2 + (a.2 + b.2)^2 = 25) →
  (b.1^2 + b.2^2 = 40) := by
sorry

end vector_magnitude_proof_l862_86238


namespace total_distance_calculation_l862_86291

/-- Calculates the total distance traveled given fuel efficiencies and fuel used for different driving conditions -/
theorem total_distance_calculation (city_efficiency highway_efficiency gravel_efficiency : ℝ)
  (city_fuel highway_fuel gravel_fuel : ℝ) : 
  city_efficiency = 15 →
  highway_efficiency = 25 →
  gravel_efficiency = 18 →
  city_fuel = 2.5 →
  highway_fuel = 3.8 →
  gravel_fuel = 1.7 →
  city_efficiency * city_fuel + highway_efficiency * highway_fuel + gravel_efficiency * gravel_fuel = 163.1 := by
  sorry

end total_distance_calculation_l862_86291


namespace brownie_division_l862_86230

/-- Represents the dimensions of a rectangular object -/
structure Dimensions where
  length : ℕ
  width : ℕ

/-- Calculates the area of a rectangular object given its dimensions -/
def area (d : Dimensions) : ℕ := d.length * d.width

/-- Represents a brownie tray with its dimensions -/
def tray : Dimensions := ⟨24, 30⟩

/-- Represents a single brownie piece with its dimensions -/
def piece : Dimensions := ⟨3, 4⟩

/-- Theorem stating that the tray can be divided into exactly 60 brownie pieces -/
theorem brownie_division :
  (area tray) / (area piece) = 60 := by sorry

end brownie_division_l862_86230


namespace conference_room_chairs_l862_86233

/-- The number of chairs in the conference room -/
def num_chairs : ℕ := 40

/-- The capacity of each chair -/
def chair_capacity : ℕ := 2

/-- The fraction of unoccupied chairs -/
def unoccupied_fraction : ℚ := 2/5

/-- The number of board members who attended the meeting -/
def attendees : ℕ := 48

theorem conference_room_chairs :
  (num_chairs : ℚ) * chair_capacity * (1 - unoccupied_fraction) = attendees ∧
  num_chairs * chair_capacity = num_chairs * 2 :=
sorry

end conference_room_chairs_l862_86233


namespace parabola_chord_midpoint_l862_86250

/-- Given a parabola y^2 = 2px and a chord with midpoint (3, 1) and slope 2, prove that p = 2 -/
theorem parabola_chord_midpoint (p : ℝ) : 
  (∀ x y : ℝ, y^2 = 2*p*x) →  -- Equation of the parabola
  (∃ x₁ y₁ x₂ y₂ : ℝ,        -- Existence of two points on the chord
    y₁^2 = 2*p*x₁ ∧          -- First point satisfies parabola equation
    y₂^2 = 2*p*x₂ ∧          -- Second point satisfies parabola equation
    (x₁ + x₂)/2 = 3 ∧        -- x-coordinate of midpoint is 3
    (y₁ + y₂)/2 = 1 ∧        -- y-coordinate of midpoint is 1
    (y₂ - y₁)/(x₂ - x₁) = 2  -- Slope of the chord is 2
  ) →
  p = 2 := by
sorry

end parabola_chord_midpoint_l862_86250


namespace aerith_win_probability_l862_86299

def coin_game (p_heads : ℚ) : ℚ :=
  (1 - p_heads) / (2 - p_heads)

theorem aerith_win_probability :
  let p_heads : ℚ := 4/7
  coin_game p_heads = 7/11 := by sorry

end aerith_win_probability_l862_86299


namespace lindas_coins_l862_86298

/-- Represents the number of coins Linda has initially and receives from her mother --/
structure CoinCounts where
  initial_dimes : Nat
  initial_quarters : Nat
  initial_nickels : Nat
  mother_dimes : Nat
  mother_quarters : Nat
  mother_nickels : Nat

/-- The theorem statement --/
theorem lindas_coins (c : CoinCounts) 
  (h1 : c.initial_dimes = 2)
  (h2 : c.initial_quarters = 6)
  (h3 : c.initial_nickels = 5)
  (h4 : c.mother_dimes = 2)
  (h5 : c.mother_nickels = 2 * c.initial_nickels)
  (h6 : c.initial_dimes + c.initial_quarters + c.initial_nickels + 
        c.mother_dimes + c.mother_quarters + c.mother_nickels = 35) :
  c.mother_quarters = 10 := by
  sorry

end lindas_coins_l862_86298


namespace sum_first_10_odd_integers_l862_86256

theorem sum_first_10_odd_integers : 
  (Finset.range 10).sum (fun i => 2 * i + 1) = 100 := by
  sorry

end sum_first_10_odd_integers_l862_86256


namespace function_inequality_solution_l862_86252

/-- Given a function f defined on positive integers and a constant a,
    prove that f(n) = a^(n*(n-1)/2) * f(1) satisfies f(n+1) ≥ a^n * f(n) for all positive integers n. -/
theorem function_inequality_solution (a : ℝ) (f : ℕ+ → ℝ) :
  (∀ n : ℕ+, f n = a^(n.val*(n.val-1)/2) * f 1) →
  (∀ n : ℕ+, f (n + 1) ≥ a^n.val * f n) :=
by sorry

end function_inequality_solution_l862_86252


namespace negation_equivalence_l862_86216

theorem negation_equivalence :
  (¬ ∀ x : ℝ, x > 0 → Real.sqrt x > x + 1) ↔ (∃ x : ℝ, x > 0 ∧ Real.sqrt x ≤ x + 1) := by
  sorry

end negation_equivalence_l862_86216


namespace power_of_three_mod_eleven_l862_86217

theorem power_of_three_mod_eleven : 3^1234 % 11 = 4 := by
  sorry

end power_of_three_mod_eleven_l862_86217


namespace length_of_PQ_l862_86288

-- Define the points
def O : ℝ × ℝ := (0, 0)
def A : ℝ × ℝ := (1, 0)
def B : ℝ × ℝ := (-1, 0)

-- Define the locus E
def E : Set (ℝ × ℝ) := {p | p.1^2 - p.2^2/4 = 1 ∧ p.1 ≠ 1 ∧ p.1 ≠ -1}

-- Define the slope product condition
def slope_product (M : ℝ × ℝ) : Prop :=
  (M.2 / (M.1 - 1)) * (M.2 / (M.1 + 1)) = 4

-- Define line l
def l : Set (ℝ × ℝ) := {p | ∃ (k : ℝ), p.2 = k * p.1 - 2}

-- Define the midpoint condition
def midpoint_condition (P Q : ℝ × ℝ) : Prop :=
  let D := ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)
  D.1 > 0 ∧ D.2 = 2

-- Main theorem
theorem length_of_PQ :
  ∀ (P Q : ℝ × ℝ),
  P ∈ E ∧ Q ∈ E ∧ P ∈ l ∧ Q ∈ l ∧
  midpoint_condition P Q ∧
  (∀ M ∈ E, slope_product M) →
  ‖P - Q‖ = 2 * Real.sqrt 14 :=
sorry

end length_of_PQ_l862_86288


namespace pharmacy_tubs_l862_86297

theorem pharmacy_tubs (total_needed : ℕ) (in_storage : ℕ) : 
  total_needed = 100 →
  in_storage = 20 →
  let to_buy := total_needed - in_storage
  let from_new_vendor := to_buy / 4
  let from_usual_vendor := to_buy - from_new_vendor
  from_usual_vendor = 60 := by
  sorry

end pharmacy_tubs_l862_86297


namespace circular_path_meeting_time_l862_86232

theorem circular_path_meeting_time (c : ℝ) : 
  c > 0 ∧ 
  (6⁻¹ : ℝ) > 0 ∧ 
  c⁻¹ > 0 ∧
  (((6 * c) / (c + 6) + 1) * c⁻¹ = 1) →
  c = 3 :=
by sorry

end circular_path_meeting_time_l862_86232


namespace circumscribed_circle_area_l862_86269

/-- The area of a circle circumscribed around an isosceles triangle -/
theorem circumscribed_circle_area (base lateral : ℝ) (h_base : base = 24) (h_lateral : lateral = 13) :
  let height : ℝ := Real.sqrt (lateral ^ 2 - (base / 2) ^ 2)
  let triangle_area : ℝ := (base * height) / 2
  let radius : ℝ := (base * lateral ^ 2) / (4 * triangle_area)
  let circle_area : ℝ := π * radius ^ 2
  circle_area = 285.61 * π :=
by sorry

end circumscribed_circle_area_l862_86269


namespace tuesday_to_monday_ratio_l862_86228

/-- Rainfall data for a week --/
structure RainfallData where
  monday_morning : ℝ
  monday_afternoon : ℝ
  tuesday : ℝ
  wednesday : ℝ
  thursday : ℝ
  friday : ℝ
  daily_average : ℝ
  num_days : ℕ

/-- Theorem stating the ratio of Tuesday's rainfall to Monday's total rainfall --/
theorem tuesday_to_monday_ratio (data : RainfallData) : 
  data.monday_morning = 2 ∧ 
  data.monday_afternoon = 1 ∧ 
  data.wednesday = 0 ∧ 
  data.thursday = 1 ∧ 
  data.friday = data.monday_morning + data.monday_afternoon + data.tuesday + data.wednesday + data.thursday ∧
  data.daily_average = 4 ∧
  data.num_days = 5 ∧
  data.daily_average * data.num_days = data.monday_morning + data.monday_afternoon + data.tuesday + data.wednesday + data.thursday + data.friday →
  data.tuesday / (data.monday_morning + data.monday_afternoon) = 2 := by
sorry

end tuesday_to_monday_ratio_l862_86228


namespace trishas_walk_distance_l862_86274

/-- The total distance Trisha walked during her vacation in New York City -/
theorem trishas_walk_distance :
  let distance_hotel_to_postcard : ℚ := 0.1111111111111111
  let distance_postcard_to_tshirt : ℚ := 0.1111111111111111
  let distance_tshirt_to_hotel : ℚ := 0.6666666666666666
  distance_hotel_to_postcard + distance_postcard_to_tshirt + distance_tshirt_to_hotel = 0.8888888888888888 := by
  sorry

end trishas_walk_distance_l862_86274


namespace minimum_bags_in_warehouse_A_minimum_bags_proof_l862_86296

theorem minimum_bags_in_warehouse_A : ℕ → ℕ → Prop :=
  fun x y =>
    (∃ k : ℕ, 
      (y + 90 = 2 * (x - 90)) ∧
      (x + k = 6 * (y - k)) ∧
      (x ≥ 139) ∧
      (∀ z : ℕ, z < x → 
        ¬(∃ w k : ℕ, 
          (w + 90 = 2 * (z - 90)) ∧
          (z + k = 6 * (w - k))))) →
    x = 139

-- The proof goes here
theorem minimum_bags_proof : 
  ∃ x y : ℕ, minimum_bags_in_warehouse_A x y :=
sorry

end minimum_bags_in_warehouse_A_minimum_bags_proof_l862_86296


namespace product_calculation_l862_86249

theorem product_calculation : 12 * 0.2 * 3 * 0.1 / 0.6 = 6 / 5 := by
  sorry

end product_calculation_l862_86249


namespace line_tangent_to_parabola_l862_86240

/-- A line y = 3x + d is tangent to the parabola y^2 = 12x if and only if d = 1 -/
theorem line_tangent_to_parabola (d : ℝ) :
  (∃ x y : ℝ, y = 3 * x + d ∧ y^2 = 12 * x ∧
    ∀ x' y' : ℝ, y' = 3 * x' + d → y'^2 ≤ 12 * x') ↔ d = 1 := by
  sorry

end line_tangent_to_parabola_l862_86240


namespace rectangle_perimeter_proof_l862_86223

def square_perimeter : ℝ := 24
def rectangle_width : ℝ := 4

theorem rectangle_perimeter_proof :
  let square_side := square_perimeter / 4
  let square_area := square_side ^ 2
  let rectangle_length := square_area / rectangle_width
  2 * (rectangle_length + rectangle_width) = 26 := by
sorry

end rectangle_perimeter_proof_l862_86223


namespace cost_difference_analysis_l862_86285

/-- Represents the cost difference between option 2 and option 1 -/
def cost_difference (x : ℝ) : ℝ := 54 * x + 9000 - (60 * x + 8800)

/-- Proves that the cost difference is 6x - 200 for x > 20, and positive when x = 30 -/
theorem cost_difference_analysis :
  (∀ x > 20, cost_difference x = 6 * x - 200) ∧
  (cost_difference 30 > 0) := by
  sorry

end cost_difference_analysis_l862_86285


namespace richard_needs_three_touchdowns_per_game_l862_86221

/-- Represents a football player's touchdown record --/
structure TouchdownRecord where
  player : String
  touchdowns : ℕ
  games : ℕ

/-- Calculates the number of touchdowns needed to beat a record --/
def touchdownsNeededToBeat (record : TouchdownRecord) : ℕ :=
  record.touchdowns + 1

/-- Theorem: Richard needs to average 3 touchdowns per game in the final two games to beat Archie's record --/
theorem richard_needs_three_touchdowns_per_game
  (archie : TouchdownRecord)
  (richard_current_touchdowns : ℕ)
  (richard_current_games : ℕ)
  (total_games : ℕ)
  (h1 : archie.player = "Archie")
  (h2 : archie.touchdowns = 89)
  (h3 : archie.games = 16)
  (h4 : richard_current_touchdowns = 6 * richard_current_games)
  (h5 : richard_current_games = 14)
  (h6 : total_games = 16) :
  (touchdownsNeededToBeat archie - richard_current_touchdowns) / (total_games - richard_current_games) = 3 :=
sorry

end richard_needs_three_touchdowns_per_game_l862_86221


namespace imaginary_part_of_complex_fraction_l862_86242

theorem imaginary_part_of_complex_fraction : 
  Complex.im ((2 * Complex.I - 5) / (2 - Complex.I)) = -1/5 := by
  sorry

end imaginary_part_of_complex_fraction_l862_86242


namespace books_sold_on_monday_l862_86261

theorem books_sold_on_monday (initial_stock : ℕ) (tuesday_sold : ℕ) (wednesday_sold : ℕ) (thursday_sold : ℕ) (friday_sold : ℕ) (unsold : ℕ) : 
  initial_stock = 800 →
  tuesday_sold = 10 →
  wednesday_sold = 20 →
  thursday_sold = 44 →
  friday_sold = 66 →
  unsold = 600 →
  initial_stock - (tuesday_sold + wednesday_sold + thursday_sold + friday_sold + unsold) = 60 := by
  sorry


end books_sold_on_monday_l862_86261


namespace coordinates_wrt_origin_l862_86225

/-- In a Cartesian coordinate system, the coordinates of a point (-1, 2) with respect to the origin are (-1, 2). -/
theorem coordinates_wrt_origin (x y : ℝ) : x = -1 ∧ y = 2 → (x, y) = (-1, 2) := by sorry

end coordinates_wrt_origin_l862_86225


namespace solution_set_M_range_of_a_l862_86206

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x + 2| - 2 * |x - 1|

-- Define the solution set M
def M : Set ℝ := {x | -2/3 ≤ x ∧ x ≤ 6}

-- Define the property for part (2)
def property (a : ℝ) : Prop := ∀ x, x ≥ a → f x ≤ x - a

-- Theorem for part (1)
theorem solution_set_M : {x : ℝ | f x ≥ -2} = M := by sorry

-- Theorem for part (2)
theorem range_of_a : {a : ℝ | property a} = {a | a ≤ -2 ∨ a ≥ 4} := by sorry

end solution_set_M_range_of_a_l862_86206


namespace least_positive_integer_with_remainders_l862_86229

theorem least_positive_integer_with_remainders : ∃! n : ℕ, 
  n > 0 ∧ 
  n % 2 = 0 ∧ 
  n % 5 = 1 ∧ 
  n % 4 = 2 ∧
  ∀ m : ℕ, m > 0 ∧ m % 2 = 0 ∧ m % 5 = 1 ∧ m % 4 = 2 → n ≤ m :=
by
  sorry

end least_positive_integer_with_remainders_l862_86229


namespace exists_twelve_digit_non_cube_l862_86287

theorem exists_twelve_digit_non_cube : ∃ n : ℕ, (10^11 ≤ n ∧ n < 10^12) ∧ ¬∃ k : ℕ, n = k^3 := by
  sorry

end exists_twelve_digit_non_cube_l862_86287


namespace not_all_face_sums_different_not_all_face_sums_different_b_l862_86270

/-- Represents the possible values that can be assigned to a vertex of the cube -/
inductive VertexValue
  | Zero
  | One

/-- Represents a cube with values assigned to its vertices -/
structure Cube :=
  (vertices : Fin 8 → VertexValue)

/-- Calculates the sum of values on a face of the cube -/
def faceSum (c : Cube) (face : Fin 6) : Nat :=
  sorry

/-- Theorem stating that it's impossible for all face sums to be different -/
theorem not_all_face_sums_different (c : Cube) : 
  ¬(∀ (i j : Fin 6), i ≠ j → faceSum c i ≠ faceSum c j) :=
sorry

/-- Represents the possible values that can be assigned to a vertex of the cube (for part b) -/
inductive VertexValueB
  | NegOne
  | PosOne

/-- Represents a cube with values assigned to its vertices (for part b) -/
structure CubeB :=
  (vertices : Fin 8 → VertexValueB)

/-- Calculates the sum of values on a face of the cube (for part b) -/
def faceSumB (c : CubeB) (face : Fin 6) : Int :=
  sorry

/-- Theorem stating that it's impossible for all face sums to be different (for part b) -/
theorem not_all_face_sums_different_b (c : CubeB) : 
  ¬(∀ (i j : Fin 6), i ≠ j → faceSumB c i ≠ faceSumB c j) :=
sorry

end not_all_face_sums_different_not_all_face_sums_different_b_l862_86270


namespace police_text_percentage_l862_86292

theorem police_text_percentage : 
  ∀ (total_texts grocery_texts response_texts police_texts : ℕ),
    total_texts = 33 →
    grocery_texts = 5 →
    response_texts = 5 * grocery_texts →
    police_texts = total_texts - (grocery_texts + response_texts) →
    (police_texts : ℚ) / (grocery_texts + response_texts : ℚ) * 100 = 10 :=
by sorry

end police_text_percentage_l862_86292


namespace quadratic_polynomial_with_complex_root_l862_86237

theorem quadratic_polynomial_with_complex_root :
  ∃ (a b c : ℝ), 
    (∀ x : ℂ, (3 : ℂ) * x^2 + (a : ℂ) * x + (b : ℂ) = 0 ↔ x = 5 + 2*I ∨ x = 5 - 2*I) ∧
    (3 : ℝ) * (5 + 2*I)^2 + a * (5 + 2*I) + b = 0 ∧
    a = -30 ∧ b = 87 := by
  sorry

end quadratic_polynomial_with_complex_root_l862_86237


namespace icosahedron_edge_probability_l862_86212

/-- A regular icosahedron -/
structure Icosahedron :=
  (vertices : Nat)
  (edges_per_vertex : Nat)
  (h_vertices : vertices = 12)
  (h_edges_per_vertex : edges_per_vertex = 5)

/-- The probability of selecting two vertices that form an edge in an icosahedron -/
def edge_probability (i : Icosahedron) : ℚ :=
  5 / 11

/-- Theorem: The probability of randomly selecting two vertices that form an edge in a regular icosahedron is 5/11 -/
theorem icosahedron_edge_probability (i : Icosahedron) : 
  edge_probability i = 5 / 11 := by
  sorry

end icosahedron_edge_probability_l862_86212


namespace perimeter_C_is_24_l862_86284

/-- Represents a polygon in the triangular grid -/
structure Polygon where
  perimeter : ℝ

/-- Represents the triangular grid with four polygons -/
structure TriangularGrid where
  A : Polygon
  B : Polygon
  C : Polygon
  D : Polygon

/-- The perimeter of triangle C in the given triangular grid -/
def perimeter_C (grid : TriangularGrid) : ℝ :=
  -- Definition to be proved
  24

/-- Theorem stating that the perimeter of triangle C is 24 cm -/
theorem perimeter_C_is_24 (grid : TriangularGrid) 
    (h1 : grid.A.perimeter = 56)
    (h2 : grid.B.perimeter = 34)
    (h3 : grid.D.perimeter = 42) :
  perimeter_C grid = 24 := by
  sorry

end perimeter_C_is_24_l862_86284


namespace quadratic_real_root_l862_86265

theorem quadratic_real_root (b : ℝ) : 
  (∃ x : ℝ, x^2 + b*x + 25 = 0) ↔ b ≤ -10 ∨ b ≥ 10 := by sorry

end quadratic_real_root_l862_86265


namespace printer_time_ratio_l862_86255

/-- Given four printers with their individual completion times, prove the ratio of time taken by printer x alone to the time taken by printers y, z, and w together. -/
theorem printer_time_ratio (x y z w : ℝ) (hx : x = 12) (hy : y = 10) (hz : z = 20) (hw : w = 15) :
  x / (1 / (1/y + 1/z + 1/w)) = 2.6 := by
  sorry

end printer_time_ratio_l862_86255


namespace fruit_mix_cherries_l862_86236

/-- Proves that in a fruit mix with the given conditions, the number of cherries is 167 -/
theorem fruit_mix_cherries (b r c : ℕ) : 
  b + r + c = 300 → 
  r = 3 * b → 
  c = 5 * b → 
  c = 167 := by
  sorry

end fruit_mix_cherries_l862_86236


namespace parallelogram_side_sum_l862_86222

/-- A parallelogram with side lengths 5, 10y-2, 3x+5, and 12 has x+y equal to 91/30 -/
theorem parallelogram_side_sum (x y : ℚ) : 
  (3 * x + 5 = 12) → (10 * y - 2 = 5) → x + y = 91 / 30 := by
  sorry

end parallelogram_side_sum_l862_86222


namespace treaty_of_paris_preliminary_articles_l862_86267

/-- Represents days of the week -/
inductive DayOfWeek
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday
| Sunday

/-- Calculates the day of the week given a number of days before a known day -/
def daysBefore (knownDay : DayOfWeek) (daysBefore : Nat) : DayOfWeek :=
  sorry

theorem treaty_of_paris_preliminary_articles :
  let treatyDay : DayOfWeek := DayOfWeek.Thursday
  let daysBetween : Nat := 621
  daysBefore treatyDay daysBetween = DayOfWeek.Tuesday :=
sorry

end treaty_of_paris_preliminary_articles_l862_86267


namespace inequality_proof_l862_86258

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a * b > 2007 * a + 2008 * b) :
  a + b > (Real.sqrt 2007 + Real.sqrt 2008) ^ 2 := by
  sorry

end inequality_proof_l862_86258


namespace quadratic_roots_nature_l862_86268

theorem quadratic_roots_nature (a b c : ℝ) (h : a ≠ 0) :
  let discriminant := b^2 - 4*a*c
  a * x^2 + b * x + c = 0 ∧ a = 1 ∧ b = -4 * Real.sqrt 3 ∧ c = 12 →
  discriminant = 0 ∧ ∃! x, a * x^2 + b * x + c = 0 :=
by sorry

end quadratic_roots_nature_l862_86268


namespace line_point_k_value_l862_86209

/-- A line contains the points (2, 4), (7, k), and (15, 8). The value of k is 72/13. -/
theorem line_point_k_value : ∀ (k : ℚ), 
  (∃ (m b : ℚ), 
    (4 : ℚ) = m * 2 + b ∧ 
    k = m * 7 + b ∧ 
    (8 : ℚ) = m * 15 + b) → 
  k = 72 / 13 := by
sorry

end line_point_k_value_l862_86209


namespace remaining_fun_is_1050_l862_86226

/-- Calculates the remaining amount for fun after a series of financial actions --/
def remaining_for_fun (initial_winnings : ℝ) (tax_rate : ℝ) (mortgage_rate : ℝ) 
  (retirement_rate : ℝ) (college_rate : ℝ) (savings : ℝ) : ℝ :=
  let after_tax := initial_winnings * (1 - tax_rate)
  let after_mortgage := after_tax * (1 - mortgage_rate)
  let after_retirement := after_mortgage * (1 - retirement_rate)
  let after_college := after_retirement * (1 - college_rate)
  after_college - savings

/-- Theorem stating that given the specific financial actions, 
    the remaining amount for fun is $1050 --/
theorem remaining_fun_is_1050 : 
  remaining_for_fun 20000 0.55 0.5 (1/3) 0.25 1200 = 1050 := by
  sorry

end remaining_fun_is_1050_l862_86226


namespace min_value_of_function_l862_86264

theorem min_value_of_function (x : ℝ) (h : x > 0) : 
  ∃ (y : ℝ), y = x + 4 / x^2 ∧ ∀ (z : ℝ), z = x + 4 / x^2 → y ≤ z ∧ y = 3 :=
sorry

end min_value_of_function_l862_86264


namespace abc_greater_than_28_l862_86220

-- Define the polynomials P and Q
def P (a b c x : ℝ) : ℝ := a * x^3 + (b - a) * x^2 - (c + b) * x + c
def Q (a b c x : ℝ) : ℝ := x^4 + (b - 1) * x^3 + (a - b) * x^2 - (c + a) * x + c

-- State the theorem
theorem abc_greater_than_28 
  (a b c : ℝ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) 
  (hb_pos : b > 0)
  (hP_roots : ∃ x₀ x₁ x₂ : ℝ, x₀ ≠ x₁ ∧ x₀ ≠ x₂ ∧ x₁ ≠ x₂ ∧ 
    P a b c x₀ = 0 ∧ P a b c x₁ = 0 ∧ P a b c x₂ = 0)
  (hQ_roots : ∃ x₀ x₁ x₂ : ℝ, 
    Q a b c x₀ = 0 ∧ Q a b c x₁ = 0 ∧ Q a b c x₂ = 0) :
  a * b * c > 28 :=
sorry

end abc_greater_than_28_l862_86220


namespace line_point_sum_l862_86283

/-- The line equation -/
def line_equation (x y : ℝ) : Prop := y = -5/3 * x + 15

/-- Point P is on the x-axis -/
def P : ℝ × ℝ := (9, 0)

/-- Point Q is on the y-axis -/
def Q : ℝ × ℝ := (0, 15)

/-- Point T is on the line segment PQ -/
def T_on_PQ (r s : ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ r = t * P.1 + (1 - t) * Q.1 ∧ s = t * P.2 + (1 - t) * Q.2

/-- The area of triangle POQ is four times the area of triangle TOP -/
def area_condition (r s : ℝ) : Prop :=
  abs ((P.1 - 0) * (Q.2 - 0) - (Q.1 - 0) * (P.2 - 0)) / 2 =
  4 * abs ((P.1 - 0) * (s - 0) - (r - 0) * (P.2 - 0)) / 2

/-- The main theorem -/
theorem line_point_sum (r s : ℝ) :
  line_equation r s →
  T_on_PQ r s →
  area_condition r s →
  r + s = 8.75 := by sorry

end line_point_sum_l862_86283


namespace school_transfer_percentage_l862_86205

/-- Proves the percentage of students from school A going to school C -/
theorem school_transfer_percentage
  (total_students : ℕ)
  (school_A_percentage : ℚ)
  (school_B_to_C_percentage : ℚ)
  (total_to_C_percentage : ℚ)
  (h1 : school_A_percentage = 60 / 100)
  (h2 : school_B_to_C_percentage = 40 / 100)
  (h3 : total_to_C_percentage = 34 / 100)
  : ∃ (school_A_to_C_percentage : ℚ),
    school_A_to_C_percentage = 30 / 100 ∧
    (school_A_percentage * total_students * school_A_to_C_percentage +
     (1 - school_A_percentage) * total_students * school_B_to_C_percentage =
     total_students * total_to_C_percentage) := by
  sorry


end school_transfer_percentage_l862_86205
