import Mathlib

namespace first_year_interest_rate_l2415_241596

/-- Given an initial amount, time period, interest rates, and final amount,
    calculate the interest rate for the first year. -/
theorem first_year_interest_rate
  (initial_amount : ℝ)
  (time_period : ℕ)
  (second_year_rate : ℝ)
  (final_amount : ℝ)
  (h1 : initial_amount = 5000)
  (h2 : time_period = 2)
  (h3 : second_year_rate = 0.25)
  (h4 : final_amount = 7500) :
  ∃ (first_year_rate : ℝ),
    first_year_rate = 0.20 ∧
    final_amount = initial_amount * (1 + first_year_rate) * (1 + second_year_rate) :=
by sorry

end first_year_interest_rate_l2415_241596


namespace duck_flock_size_l2415_241581

/-- Calculates the total number of ducks in a combined flock after a given number of years -/
def combined_flock_size (initial_size : ℕ) (annual_increase : ℕ) (years : ℕ) (joining_flock : ℕ) : ℕ :=
  initial_size + annual_increase * years + joining_flock

/-- Theorem stating the combined flock size after 5 years -/
theorem duck_flock_size :
  combined_flock_size 100 10 5 150 = 300 := by
  sorry

#eval combined_flock_size 100 10 5 150

end duck_flock_size_l2415_241581


namespace millet_percentage_in_mix_l2415_241540

/-- Theorem: Millet percentage in a birdseed mix -/
theorem millet_percentage_in_mix
  (brand_a_millet : ℝ)
  (brand_b_millet : ℝ)
  (mix_brand_a : ℝ)
  (h1 : brand_a_millet = 0.60)
  (h2 : brand_b_millet = 0.65)
  (h3 : mix_brand_a = 0.60)
  (h4 : 0 ≤ mix_brand_a ∧ mix_brand_a ≤ 1) :
  mix_brand_a * brand_a_millet + (1 - mix_brand_a) * brand_b_millet = 0.62 := by
  sorry


end millet_percentage_in_mix_l2415_241540


namespace discounted_cost_six_books_l2415_241553

/-- The cost of three identical books -/
def cost_three_books : ℚ := 45

/-- The number of books in the discounted purchase -/
def num_books_discounted : ℕ := 6

/-- The discount rate applied when buying six books -/
def discount_rate : ℚ := 1 / 10

/-- The cost of six books with a 10% discount, given that three identical books cost $45 -/
theorem discounted_cost_six_books : 
  (num_books_discounted : ℚ) * (cost_three_books / 3) * (1 - discount_rate) = 81 := by
  sorry

end discounted_cost_six_books_l2415_241553


namespace school_distance_proof_l2415_241522

/-- The distance to school in miles -/
def distance_to_school : ℝ := 5

/-- The speed of walking in miles per hour for the first scenario -/
def speed1 : ℝ := 4

/-- The speed of walking in miles per hour for the second scenario -/
def speed2 : ℝ := 5

/-- The time difference in hours between arriving early and late -/
def time_difference : ℝ := 0.25

theorem school_distance_proof :
  (distance_to_school / speed1 - distance_to_school / speed2 = time_difference) ∧
  (distance_to_school = 5) := by
  sorry

end school_distance_proof_l2415_241522


namespace carla_counted_books_thrice_l2415_241532

/-- Represents the counting scenario for Carla on Monday and Tuesday -/
structure CarlaCounting where
  monday_tiles : ℕ
  monday_books : ℕ
  tuesday_total : ℕ

/-- Calculates the number of times Carla counted the books on Tuesday -/
def books_count_tuesday (c : CarlaCounting) : ℕ :=
  let tuesday_tiles := c.monday_tiles * 2
  let tuesday_books := c.tuesday_total - tuesday_tiles
  tuesday_books / c.monday_books

/-- Theorem stating that given the conditions, Carla counted the books 3 times on Tuesday -/
theorem carla_counted_books_thrice (c : CarlaCounting) 
  (h1 : c.monday_tiles = 38)
  (h2 : c.monday_books = 75)
  (h3 : c.tuesday_total = 301) :
  books_count_tuesday c = 3 := by
  sorry

end carla_counted_books_thrice_l2415_241532


namespace fiftieth_term_of_sequence_l2415_241517

def arithmetic_sequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  a₁ + (n - 1) * d

theorem fiftieth_term_of_sequence : arithmetic_sequence 2 3 50 = 149 := by
  sorry

end fiftieth_term_of_sequence_l2415_241517


namespace zero_division_not_always_zero_l2415_241588

theorem zero_division_not_always_zero : ¬ (∀ a : ℝ, a ≠ 0 → 0 / a = 0) :=
sorry

end zero_division_not_always_zero_l2415_241588


namespace negative_negative_eight_properties_l2415_241589

theorem negative_negative_eight_properties :
  let x : ℤ := -8
  let y : ℤ := -(-x)
  (y = -x) ∧ 
  (y = -1 * x) ∧ 
  (y = |x|) ∧ 
  (y = 8) := by sorry

end negative_negative_eight_properties_l2415_241589


namespace total_travel_options_l2415_241539

/-- The number of train options from location A to location B -/
def train_options : ℕ := 3

/-- The number of ferry options from location B to location C -/
def ferry_options : ℕ := 2

/-- The number of direct flight options from location A to location C -/
def flight_options : ℕ := 2

/-- The total number of travel options from location A to location C -/
def total_options : ℕ := train_options * ferry_options + flight_options

theorem total_travel_options : total_options = 8 := by
  sorry

end total_travel_options_l2415_241539


namespace circles_M_N_common_tangents_l2415_241541

/-- Circle M with equation x^2 + y^2 - 4y = 0 -/
def circle_M : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 - 4 * p.2 = 0}

/-- Circle N with equation (x - 1)^2 + (y - 1)^2 = 1 -/
def circle_N : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - 1)^2 + (p.2 - 1)^2 = 1}

/-- The number of common tangents between two circles -/
def num_common_tangents (C1 C2 : Set (ℝ × ℝ)) : ℕ :=
  sorry

/-- Theorem stating that circles M and N have exactly 2 common tangents -/
theorem circles_M_N_common_tangents :
  num_common_tangents circle_M circle_N = 2 :=
sorry

end circles_M_N_common_tangents_l2415_241541


namespace sequence_median_l2415_241502

def sequence_sum (n : ℕ) : ℕ := n * (n + 1) / 2

theorem sequence_median : 
  let total_elements := sequence_sum 100
  let median_position := total_elements / 2
  ∃ k : ℕ, 
    k ≤ 100 ∧ 
    sequence_sum (k - 1) < median_position ∧ 
    median_position ≤ sequence_sum k ∧
    k = 71 := by sorry

end sequence_median_l2415_241502


namespace function_inequality_implies_a_range_l2415_241566

theorem function_inequality_implies_a_range (a : ℝ) : 
  (∀ x : ℝ, x > 0 → x * (2 * Real.log a - Real.log x) ≤ a) →
  (0 < a ∧ a ≤ Real.exp (-1)) := by
  sorry

end function_inequality_implies_a_range_l2415_241566


namespace range_of_fraction_l2415_241560

theorem range_of_fraction (x y : ℝ) (h : x + Real.sqrt (1 - y^2) = 0) :
  ∃ (a b : ℝ), a = -Real.sqrt 3 / 3 ∧ b = Real.sqrt 3 / 3 ∧
  ∀ (z : ℝ), (∃ (x' y' : ℝ), x' + Real.sqrt (1 - y'^2) = 0 ∧ z = y' / (x' - 2)) →
  a ≤ z ∧ z ≤ b :=
sorry

end range_of_fraction_l2415_241560


namespace video_votes_l2415_241545

theorem video_votes (score : ℤ) (like_percentage : ℚ) : 
  score = 130 ∧ like_percentage = 70 / 100 → 
  ∃ total_votes : ℕ, 
    (like_percentage * total_votes : ℚ) - ((1 - like_percentage) * total_votes : ℚ) = score ∧
    total_votes = 325 := by
  sorry

end video_votes_l2415_241545


namespace wall_cleaning_time_l2415_241550

/-- Represents the cleaning rate in minutes per section -/
def cleaning_rate (time_spent : ℕ) (sections_cleaned : ℕ) : ℚ :=
  (time_spent : ℚ) / sections_cleaned

/-- Calculates the remaining time to clean the wall -/
def remaining_time (total_sections : ℕ) (cleaned_sections : ℕ) (rate : ℚ) : ℚ :=
  ((total_sections - cleaned_sections) : ℚ) * rate

/-- Theorem stating the remaining time to clean the wall -/
theorem wall_cleaning_time (total_sections : ℕ) (cleaned_sections : ℕ) (time_spent : ℕ) :
  total_sections = 18 ∧ cleaned_sections = 3 ∧ time_spent = 33 →
  remaining_time total_sections cleaned_sections (cleaning_rate time_spent cleaned_sections) = 165 := by
  sorry

end wall_cleaning_time_l2415_241550


namespace intersection_complement_equals_singleton_l2415_241574

def U : Set Nat := {1, 2, 3, 4, 5, 6}
def A : Set Nat := {2, 3}
def B : Set Nat := {3, 5}

theorem intersection_complement_equals_singleton : A ∩ (U \ B) = {2} := by sorry

end intersection_complement_equals_singleton_l2415_241574


namespace square_diagonal_side_perimeter_l2415_241549

theorem square_diagonal_side_perimeter :
  ∀ (d s p : ℝ),
  d = 2 * Real.sqrt 2 →  -- diagonal is 2√2 inches
  d = s * Real.sqrt 2 →  -- relation between diagonal and side in a square
  s = 2 ∧                -- side length is 2 inches
  p = 4 * s              -- perimeter is 4 times the side length
  := by sorry

end square_diagonal_side_perimeter_l2415_241549


namespace patrick_less_than_twice_greg_l2415_241500

def homework_hours (jacob greg patrick : ℕ) : Prop :=
  jacob = 18 ∧ 
  greg = jacob - 6 ∧ 
  jacob + greg + patrick = 50

theorem patrick_less_than_twice_greg : 
  ∀ jacob greg patrick : ℕ, 
  homework_hours jacob greg patrick → 
  2 * greg - patrick = 4 := by
sorry

end patrick_less_than_twice_greg_l2415_241500


namespace equal_lengths_implies_k_value_l2415_241534

theorem equal_lengths_implies_k_value (AB AC : ℝ) (k : ℝ) :
  AB = AC → AB = 8 → AC = 5 - k → k = -3 := by
  sorry

end equal_lengths_implies_k_value_l2415_241534


namespace donna_has_40_bananas_l2415_241598

/-- The number of bananas Donna has -/
def donnas_bananas (total : ℕ) (dawns_extra : ℕ) (lydias : ℕ) : ℕ :=
  total - (lydias + dawns_extra) - lydias

/-- Proof that Donna has 40 bananas -/
theorem donna_has_40_bananas :
  donnas_bananas 200 40 60 = 40 := by
  sorry

end donna_has_40_bananas_l2415_241598


namespace min_value_inequality_l2415_241576

-- Define the function f
def f (x : ℝ) : ℝ := |x + 1| + |x - 3|

-- State the theorem
theorem min_value_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (∃ m : ℝ, (∀ x : ℝ, f x ≥ m) ∧ (∃ x₀ : ℝ, f x₀ = m) ∧ a + b + c = m) →
  a^2 + b^2 + c^2 ≥ 16/3 := by
sorry

end min_value_inequality_l2415_241576


namespace corrected_mean_l2415_241580

theorem corrected_mean (n : ℕ) (original_mean : ℝ) (incorrect_value correct_value : ℝ) 
  (h1 : n = 50) 
  (h2 : original_mean = 41) 
  (h3 : incorrect_value = 23) 
  (h4 : correct_value = 48) : 
  (n : ℝ) * original_mean - incorrect_value + correct_value = n * 41.5 := by
  sorry

#check corrected_mean

end corrected_mean_l2415_241580


namespace base_7_multiplication_l2415_241585

/-- Converts a number from base 7 to base 10 --/
def to_base_10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * 7^i) 0

/-- Converts a number from base 10 to base 7 --/
def to_base_7 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) :=
    if m = 0 then acc
    else aux (m / 7) ((m % 7) :: acc)
  aux n []

/-- Theorem statement --/
theorem base_7_multiplication :
  to_base_7 (to_base_10 [4, 2, 3] * to_base_10 [3]) = [5, 0, 3, 1] := by
  sorry

end base_7_multiplication_l2415_241585


namespace book_purchase_equation_l2415_241570

theorem book_purchase_equation (x : ℝ) : x > 0 →
  (∀ y : ℝ, y = x + 8 → y > 0) →
  (15000 : ℝ) / (x + 8) = (12000 : ℝ) / x :=
by
  sorry

end book_purchase_equation_l2415_241570


namespace triangle_areas_l2415_241523

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the point O (intersection of altitudes)
def O : ℝ × ℝ := sorry

-- Define the points P, Q, R on the sides of the triangle
def P : ℝ × ℝ := sorry
def Q : ℝ × ℝ := sorry
def R : ℝ × ℝ := sorry

-- Define the given conditions
axiom parallel_RP_AC : sorry
axiom AC_length : sorry
axiom sin_ABC : sorry

-- Define the areas of triangles ABC and ROC
noncomputable def area_ABC (t : Triangle) : ℝ := sorry
noncomputable def area_ROC (t : Triangle) : ℝ := sorry

-- State the theorem
theorem triangle_areas (t : Triangle) :
  (area_ABC t = 16/3 ∧ area_ROC t = 21/25) ∨
  (area_ABC t = 3 ∧ area_ROC t = 112/75) :=
sorry

end triangle_areas_l2415_241523


namespace intersection_area_is_sqrt_k_l2415_241529

/-- Regular tetrahedron with edge length 5 -/
structure RegularTetrahedron where
  edge_length : ℝ
  edge_length_eq : edge_length = 5

/-- Plane passing through specific points of a regular tetrahedron -/
structure IntersectionPlane (t : RegularTetrahedron) where
  -- Midpoint of edge VA
  point_R : ℝ × ℝ × ℝ
  -- Midpoint of edge AB
  point_S : ℝ × ℝ × ℝ
  -- Point one-third from C to B
  point_T : ℝ × ℝ × ℝ

/-- Area of the intersection between the tetrahedron and the plane -/
def intersection_area (t : RegularTetrahedron) (p : IntersectionPlane t) : ℝ := sorry

/-- The theorem to be proved -/
theorem intersection_area_is_sqrt_k (t : RegularTetrahedron) (p : IntersectionPlane t) :
  ∃ k : ℝ, k > 0 ∧ intersection_area t p = Real.sqrt k := by sorry

end intersection_area_is_sqrt_k_l2415_241529


namespace perfectSquareFactorsOf1800_l2415_241518

/-- The number of positive factors of 1800 that are perfect squares -/
def perfectSquareFactors : ℕ := 8

/-- 1800 as a natural number -/
def n : ℕ := 1800

/-- A function that returns the number of positive factors of a natural number that are perfect squares -/
def countPerfectSquareFactors (m : ℕ) : ℕ := sorry

theorem perfectSquareFactorsOf1800 : countPerfectSquareFactors n = perfectSquareFactors := by sorry

end perfectSquareFactorsOf1800_l2415_241518


namespace hyperbola_intersection_line_l2415_241501

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 / 3 = 1

-- Define the foci
def F₁ : ℝ × ℝ := (-2, 0)
def F₂ : ℝ × ℝ := (2, 0)

-- Define eccentricity
def e : ℝ := 2

-- Define point M
def M : ℝ × ℝ := (1, 3)

-- Define the line l
def line_l (x y : ℝ) : Prop := y = x + 2

-- Theorem statement
theorem hyperbola_intersection_line :
  ∀ A B : ℝ × ℝ,
  hyperbola A.1 A.2 ∧ hyperbola B.1 B.2 →  -- A and B are on the hyperbola
  M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) →  -- M is midpoint of AB
  line_l A.1 A.2 ∧ line_l B.1 B.2 →  -- A and B are on line l
  ∀ x y : ℝ, line_l x y ↔ y = x + 2 :=
by sorry

end hyperbola_intersection_line_l2415_241501


namespace cos_sin_sum_equals_sqrt2_over_2_l2415_241506

theorem cos_sin_sum_equals_sqrt2_over_2 :
  Real.cos (58 * π / 180) * Real.cos (13 * π / 180) + 
  Real.sin (58 * π / 180) * Real.sin (13 * π / 180) = 
  Real.sqrt 2 / 2 :=
by sorry

end cos_sin_sum_equals_sqrt2_over_2_l2415_241506


namespace square_sum_diff_product_l2415_241557

theorem square_sum_diff_product (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  ((a + b)^2 - (a - b)^2)^2 / (a * b)^2 = 16 := by
  sorry

end square_sum_diff_product_l2415_241557


namespace spade_problem_l2415_241510

def spade (x y : ℝ) : ℝ := (x + y) * (x - y)

theorem spade_problem : spade 5 (spade 3 2) = 0 := by
  sorry

end spade_problem_l2415_241510


namespace layoff_plans_count_l2415_241513

def staff_count : ℕ := 10
def layoff_count : ℕ := 4

/-- The number of ways to select 4 people out of 10 for layoff, 
    where two specific people (A and B) cannot both be kept -/
def layoff_plans : ℕ := Nat.choose (staff_count - 2) layoff_count + 
                        2 * Nat.choose (staff_count - 2) (layoff_count - 1)

theorem layoff_plans_count : layoff_plans = 182 := by
  sorry

end layoff_plans_count_l2415_241513


namespace johns_age_l2415_241584

/-- Given that John is 30 years younger than his dad and the sum of their ages is 80,
    prove that John is 25 years old. -/
theorem johns_age (john dad : ℕ) 
  (h1 : john = dad - 30)
  (h2 : john + dad = 80) : 
  john = 25 := by
  sorry

end johns_age_l2415_241584


namespace joes_lift_weight_l2415_241571

theorem joes_lift_weight (first_lift second_lift : ℕ) : 
  first_lift + second_lift = 600 →
  2 * first_lift = second_lift + 300 →
  first_lift = 300 := by
  sorry

end joes_lift_weight_l2415_241571


namespace cyclist_speed_l2415_241577

theorem cyclist_speed (distance : ℝ) (time_difference : ℝ) : 
  distance = 96 →
  time_difference = 16 →
  ∃ (speed : ℝ), 
    speed > 0 ∧
    distance / (speed - 4) = distance / (1.5 * speed) + time_difference ∧
    speed = 8 := by
  sorry

end cyclist_speed_l2415_241577


namespace tangent_sum_twelve_eighteen_equals_sqrt_three_over_three_l2415_241556

theorem tangent_sum_twelve_eighteen_equals_sqrt_three_over_three :
  (Real.tan (12 * π / 180) + Real.tan (18 * π / 180)) / 
  (1 - Real.tan (12 * π / 180) * Real.tan (18 * π / 180)) = Real.sqrt 3 / 3 := by
  sorry

end tangent_sum_twelve_eighteen_equals_sqrt_three_over_three_l2415_241556


namespace probability_sum_30_l2415_241568

/-- Represents a 20-faced die with specific numbering --/
structure Die :=
  (faces : Finset ℕ)
  (blank_face : Bool)
  (fair : Bool)
  (face_count : faces.card + (if blank_face then 1 else 0) = 20)

/-- Die 1 with faces numbered 1-18 and one blank face --/
def die1 : Die :=
  { faces := Finset.range 19 \ {0},
    blank_face := true,
    fair := true,
    face_count := sorry }

/-- Die 2 with faces numbered 1-9 and 11-20 and one blank face --/
def die2 : Die :=
  { faces := (Finset.range 21 \ {0, 10}),
    blank_face := true,
    fair := true,
    face_count := sorry }

/-- The probability of an event given the number of favorable outcomes and total outcomes --/
def probability (favorable : ℕ) (total : ℕ) : ℚ :=
  favorable / total

/-- The number of ways to roll a sum of 30 with the given dice --/
def favorable_outcomes : ℕ := 8

/-- The total number of possible outcomes when rolling two 20-faced dice --/
def total_outcomes : ℕ := 400

/-- The main theorem: probability of rolling a sum of 30 is 1/50 --/
theorem probability_sum_30 :
  probability favorable_outcomes total_outcomes = 1 / 50 :=
sorry

end probability_sum_30_l2415_241568


namespace line_triangle_area_theorem_l2415_241521

/-- Represents a line in the form y = mx + b -/
structure Line where
  m : ℝ
  b : ℝ

/-- Checks if a line forms a triangle with the coordinate axes -/
def formsTriangle (l : Line) : Prop :=
  l.b ≠ 0 ∧ l.b / l.m < 0

/-- Calculates the area of the triangle formed by a line and the coordinate axes -/
noncomputable def triangleArea (l : Line) : ℝ :=
  abs (l.b * (l.b / l.m)) / 2

/-- The main theorem -/
theorem line_triangle_area_theorem (k : ℝ) :
  let l : Line := { m := -2, b := k }
  formsTriangle l ∧ triangleArea l = 4 → k = 4 ∨ k = -4 := by
  sorry

end line_triangle_area_theorem_l2415_241521


namespace evaluate_expression_l2415_241592

theorem evaluate_expression (x y z : ℚ) (hx : x = 1/4) (hy : y = 3/4) (hz : z = -8) :
  x^2 * y^3 * z^2 = 108 := by
  sorry

end evaluate_expression_l2415_241592


namespace correct_simplification_l2415_241505

theorem correct_simplification (a b : ℝ) : 5*a - (b - 1) = 5*a - b + 1 := by
  sorry

end correct_simplification_l2415_241505


namespace point_slope_problem_l2415_241524

/-- If m > 0 and the points (m, 4) and (2, m) lie on a line with slope m², then m = 2. -/
theorem point_slope_problem (m : ℝ) (h1 : m > 0) 
  (h2 : (m - 4) / (2 - m) = m^2) : m = 2 := by
  sorry

end point_slope_problem_l2415_241524


namespace ball_attendees_l2415_241559

theorem ball_attendees :
  ∀ (n m : ℕ),
  n + m < 50 →
  (3 * n) / 4 = (5 * m) / 7 →
  n + m = 41 :=
by
  sorry

end ball_attendees_l2415_241559


namespace smallest_four_digit_mod_five_l2415_241593

theorem smallest_four_digit_mod_five : ∃ n : ℕ,
  (n ≥ 1000) ∧                 -- four-digit number
  (n < 10000) ∧                -- four-digit number
  (n % 5 = 4) ∧                -- equivalent to 4 mod 5
  (∀ m : ℕ, m ≥ 1000 ∧ m < 10000 ∧ m % 5 = 4 → m ≥ n) ∧  -- smallest such number
  (n = 1004) := by             -- the answer is 1004
sorry

end smallest_four_digit_mod_five_l2415_241593


namespace largest_common_term_l2415_241555

theorem largest_common_term (n m : ℕ) : 
  (147 = 2 + 5 * n) ∧ 
  (147 = 3 + 8 * m) ∧ 
  (147 ≤ 150) ∧ 
  (∀ k : ℕ, k > 147 → k ≤ 150 → (k - 2) % 5 ≠ 0 ∨ (k - 3) % 8 ≠ 0) :=
by sorry

end largest_common_term_l2415_241555


namespace complex_modulus_problem_l2415_241527

theorem complex_modulus_problem (z : ℂ) (h : z * (2 - Complex.I) = 3 + Complex.I) :
  Complex.abs z = Real.sqrt 2 := by
  sorry

end complex_modulus_problem_l2415_241527


namespace oil_price_reduction_l2415_241546

/-- Prove that the amount spent on oil is 1500 given the conditions --/
theorem oil_price_reduction (original_price reduced_price amount_spent : ℝ) 
  (h1 : reduced_price = original_price * (1 - 0.2))
  (h2 : reduced_price = 30)
  (h3 : amount_spent / reduced_price - amount_spent / original_price = 10) :
  amount_spent = 1500 := by
  sorry

end oil_price_reduction_l2415_241546


namespace equation_solutions_l2415_241515

theorem equation_solutions :
  (∃ x : ℝ, (x - 1)^3 = 64 ∧ x = 5) ∧
  (∃ x : ℝ, 25 * x^2 + 3 = 12 ∧ (x = 3/5 ∨ x = -3/5)) := by
  sorry

end equation_solutions_l2415_241515


namespace min_gumballs_for_four_same_color_l2415_241572

/-- Represents the number of gumballs of each color in the machine -/
structure GumballMachine where
  red : Nat
  white : Nat
  blue : Nat
  green : Nat

/-- Calculates the minimum number of gumballs needed to ensure four of the same color -/
def minGumballsForFourSameColor (machine : GumballMachine) : Nat :=
  sorry

/-- Theorem stating the minimum number of gumballs needed for the given machine -/
theorem min_gumballs_for_four_same_color 
  (machine : GumballMachine) 
  (h_red : machine.red = 10)
  (h_white : machine.white = 8)
  (h_blue : machine.blue = 9)
  (h_green : machine.green = 6) :
  minGumballsForFourSameColor machine = 13 := by
  sorry

end min_gumballs_for_four_same_color_l2415_241572


namespace complex_number_in_second_quadrant_l2415_241564

/-- The complex number z = i(1+i) is located in the second quadrant of the complex plane. -/
theorem complex_number_in_second_quadrant : 
  let z : ℂ := Complex.I * (1 + Complex.I)
  (z.re < 0) ∧ (z.im > 0) :=
by sorry

end complex_number_in_second_quadrant_l2415_241564


namespace some_number_proof_l2415_241562

def total_prime_factors (n : ℕ) : ℕ := sorry

theorem some_number_proof (x : ℕ) :
  total_prime_factors (x * 11^13 * 7^5) = 29 → x = 2^11 := by
  sorry

end some_number_proof_l2415_241562


namespace cuboid_height_from_volume_and_base_area_l2415_241582

/-- Represents the properties of a cuboid -/
structure Cuboid where
  volume : ℝ
  baseArea : ℝ
  height : ℝ

/-- Theorem stating that a cuboid with volume 144 and base area 18 has height 8 -/
theorem cuboid_height_from_volume_and_base_area :
  ∀ (c : Cuboid), c.volume = 144 → c.baseArea = 18 → c.height = 8 := by
  sorry

end cuboid_height_from_volume_and_base_area_l2415_241582


namespace intersection_point_is_solution_l2415_241520

/-- The intersection point of two lines -/
def intersection_point : ℚ × ℚ := (70/31, 135/31)

/-- First line equation -/
def line1 (x y : ℚ) : Prop := 8*x - 3*y = 5

/-- Second line equation -/
def line2 (x y : ℚ) : Prop := 5*x + 2*y = 20

theorem intersection_point_is_solution :
  let (x, y) := intersection_point
  line1 x y ∧ line2 x y ∧
  ∀ x' y', line1 x' y' ∧ line2 x' y' → x' = x ∧ y' = y := by
  sorry

end intersection_point_is_solution_l2415_241520


namespace point_coordinates_l2415_241591

/-- A point in the second quadrant with given distances from axes has coordinates (-1, 2) -/
theorem point_coordinates (P : ℝ × ℝ) :
  (P.1 < 0 ∧ P.2 > 0) →  -- P is in the second quadrant
  |P.2| = 2 →            -- Distance from P to x-axis is 2
  |P.1| = 1 →            -- Distance from P to y-axis is 1
  P = (-1, 2) :=
by sorry

end point_coordinates_l2415_241591


namespace trapezoid_shorter_base_l2415_241512

/-- A trapezoid with the given properties -/
structure Trapezoid where
  longer_base : ℝ
  shorter_base : ℝ
  midpoint_segment : ℝ

/-- The property that the line joining the midpoints of the diagonals is half the difference of the bases -/
def midpoint_property (t : Trapezoid) : Prop :=
  t.midpoint_segment = (t.longer_base - t.shorter_base) / 2

/-- The theorem to prove -/
theorem trapezoid_shorter_base (t : Trapezoid) 
  (h1 : t.longer_base = 120)
  (h2 : t.midpoint_segment = 7)
  (h3 : midpoint_property t) : 
  t.shorter_base = 106 := by
  sorry

end trapezoid_shorter_base_l2415_241512


namespace racket_price_proof_l2415_241563

/-- Given the total cost of items and the costs of sneakers and sports outfit, 
    prove that the price of the tennis racket is the difference between the total 
    cost and the sum of the other items' costs. -/
theorem racket_price_proof (total_cost sneakers_cost outfit_cost : ℕ) 
    (h1 : total_cost = 750)
    (h2 : sneakers_cost = 200)
    (h3 : outfit_cost = 250) : 
    total_cost - (sneakers_cost + outfit_cost) = 300 := by
  sorry

#check racket_price_proof

end racket_price_proof_l2415_241563


namespace sqrt_sum_expression_l2415_241537

theorem sqrt_sum_expression (a b c : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h_minimal : ∀ (a' b' c' : ℕ), a' > 0 → b' > 0 → c' > 0 → 
    (Real.sqrt 3 + 1 / Real.sqrt 3 + Real.sqrt 7 + 1 / Real.sqrt 7) * c' = a' * Real.sqrt 3 + b' * Real.sqrt 7 
    → c ≤ c')
  (h_equality : (Real.sqrt 3 + 1 / Real.sqrt 3 + Real.sqrt 7 + 1 / Real.sqrt 7) * c = a * Real.sqrt 3 + b * Real.sqrt 7) :
  a + b + c = 73 := by
sorry

end sqrt_sum_expression_l2415_241537


namespace intersection_of_M_and_N_l2415_241503

def M : Set ℕ := {1, 2, 5}
def N : Set ℕ := {x | x ≤ 2}

theorem intersection_of_M_and_N : M ∩ N = {1, 2} := by sorry

end intersection_of_M_and_N_l2415_241503


namespace quadratic_point_ordering_l2415_241516

/-- A quadratic function of the form y = -(x+1)² + c -/
def quadratic_function (c : ℝ) (x : ℝ) : ℝ := -(x + 1)^2 + c

theorem quadratic_point_ordering (c : ℝ) :
  let y₁ := quadratic_function c (-13/4)
  let y₂ := quadratic_function c (-1)
  let y₃ := quadratic_function c 0
  y₁ < y₃ ∧ y₃ < y₂ := by sorry

end quadratic_point_ordering_l2415_241516


namespace equation_solution_l2415_241599

theorem equation_solution :
  ∃! (a b c d : ℚ), 
    a^2 + b^2 + c^2 + d^2 - a*b - b*c - c*d - d + 2/5 = 0 ∧
    a = 1/5 ∧ b = 2/5 ∧ c = 3/5 ∧ d = 4/5 := by
  sorry

end equation_solution_l2415_241599


namespace maggie_bouncy_balls_indeterminate_l2415_241578

theorem maggie_bouncy_balls_indeterminate 
  (yellow_packs : ℝ) 
  (green_packs_given : ℝ) 
  (balls_per_pack : ℝ) 
  (total_kept : ℕ) 
  (h1 : yellow_packs = 8.0)
  (h2 : green_packs_given = 4.0)
  (h3 : balls_per_pack = 10.0)
  (h4 : total_kept = 80)
  (h5 : yellow_packs * balls_per_pack = total_kept) :
  ∃ (x y : ℝ), x ≠ y ∧ 
    (yellow_packs * balls_per_pack - green_packs_given * balls_per_pack + x * balls_per_pack = total_kept) ∧
    (yellow_packs * balls_per_pack - green_packs_given * balls_per_pack + y * balls_per_pack = total_kept) :=
by sorry

end maggie_bouncy_balls_indeterminate_l2415_241578


namespace pyramid_side_length_l2415_241536

/-- Regular triangular pyramid with specific properties -/
structure RegularPyramid where
  -- Base triangle side length
  a : ℝ
  -- Angle of inclination of face to base
  α : ℝ
  -- Height of the pyramid
  h : ℝ
  -- Condition that α is arctan(3/4)
  angle_condition : α = Real.arctan (3/4)
  -- Relation between height, side length, and angle
  height_relation : h = (a * Real.sqrt 3) / 2

/-- Polyhedron formed by intersecting prism with pyramid -/
structure Polyhedron (p : RegularPyramid) where
  -- Surface area of the polyhedron
  surface_area : ℝ
  -- Condition that surface area is 53√3
  area_condition : surface_area = 53 * Real.sqrt 3

/-- Theorem stating the side length of the base triangle -/
theorem pyramid_side_length (p : RegularPyramid) (poly : Polyhedron p) :
  p.a = 3 * Real.sqrt 3 := by
  sorry


end pyramid_side_length_l2415_241536


namespace two_digit_product_2210_l2415_241535

theorem two_digit_product_2210 (a b : ℕ) : 
  10 ≤ a ∧ a < 100 ∧ 
  10 ≤ b ∧ b < 100 ∧ 
  a * b = 2210 →
  min a b = 26 := by
sorry

end two_digit_product_2210_l2415_241535


namespace division_remainder_l2415_241558

theorem division_remainder : ∃ (A B : ℕ), 26 = 4 * A + B ∧ B < 4 ∧ B = 2 := by
  sorry

end division_remainder_l2415_241558


namespace roots_reality_l2415_241542

theorem roots_reality (p q : ℝ) (h : p^2 - 4*q > 0) :
  ∀ a : ℝ, (2*a + 3*p)^2 - 4*3*(q + a*p) > 0 := by
  sorry

end roots_reality_l2415_241542


namespace base4_divisibility_by_17_l2415_241548

def base4_to_decimal (a b c d : ℕ) : ℕ :=
  a * 4^3 + b * 4^2 + c * 4^1 + d * 4^0

def is_base4_digit (x : ℕ) : Prop :=
  x = 0 ∨ x = 1 ∨ x = 2 ∨ x = 3

theorem base4_divisibility_by_17 (x : ℕ) :
  is_base4_digit x →
  (base4_to_decimal 2 3 x 2 ∣ 17) ↔ x = 3 :=
by sorry

end base4_divisibility_by_17_l2415_241548


namespace cookies_in_box_proof_l2415_241594

/-- The number of cookies in each bag -/
def cookies_per_bag : ℕ := 7

/-- The number of boxes -/
def num_boxes : ℕ := 8

/-- The number of bags -/
def num_bags : ℕ := 9

/-- The additional number of cookies in boxes compared to bags -/
def additional_cookies : ℕ := 33

/-- The number of cookies in each box -/
def cookies_per_box : ℕ := 12

theorem cookies_in_box_proof :
  num_boxes * cookies_per_box = num_bags * cookies_per_bag + additional_cookies :=
sorry

end cookies_in_box_proof_l2415_241594


namespace stripe_area_on_cylindrical_tank_l2415_241533

/-- The area of a stripe on a cylindrical tank -/
theorem stripe_area_on_cylindrical_tank
  (diameter : ℝ)
  (stripe_width : ℝ)
  (revolutions : ℝ)
  (h_diameter : diameter = 20)
  (h_stripe_width : stripe_width = 4)
  (h_revolutions : revolutions = 3) :
  stripe_width * revolutions * (π * diameter) = 240 * π :=
by sorry

end stripe_area_on_cylindrical_tank_l2415_241533


namespace fraction_addition_l2415_241531

theorem fraction_addition (m n : ℚ) (h : m / n = 3 / 7) : (m + n) / n = 10 / 7 := by
  sorry

end fraction_addition_l2415_241531


namespace polyhedron_sum_l2415_241508

/-- A convex polyhedron with triangular, pentagonal, and hexagonal faces. -/
structure Polyhedron where
  T : ℕ  -- Number of triangular faces
  P : ℕ  -- Number of pentagonal faces
  H : ℕ  -- Number of hexagonal faces
  V : ℕ  -- Number of vertices
  E : ℕ  -- Number of edges

/-- Properties of the polyhedron -/
def is_valid_polyhedron (poly : Polyhedron) : Prop :=
  -- Total number of faces is 42
  poly.T + poly.P + poly.H = 42 ∧
  -- At each vertex, 3 triangular, 2 pentagonal, and 1 hexagonal face meet
  6 * poly.V = 3 * poly.T + 2 * poly.P + poly.H ∧
  -- Edge count
  2 * poly.E = 3 * poly.T + 5 * poly.P + 6 * poly.H ∧
  -- Euler's formula
  poly.V - poly.E + (poly.T + poly.P + poly.H) = 2

/-- Theorem statement -/
theorem polyhedron_sum (poly : Polyhedron) 
  (h : is_valid_polyhedron poly) : 
  100 * poly.H + 10 * poly.P + poly.T + poly.V = 714 := by
  sorry


end polyhedron_sum_l2415_241508


namespace x_value_proof_l2415_241538

theorem x_value_proof (x : ℝ) (h1 : x^2 - 3*x = 0) (h2 : x ≠ 0) : x = 3 := by
  sorry

end x_value_proof_l2415_241538


namespace prob_four_of_a_kind_after_reroll_l2415_241519

/-- Represents the outcome of rolling five dice -/
structure DiceRoll where
  pairs : Nat -- Number of pairs
  fourOfAKind : Bool -- Whether there's a four-of-a-kind

/-- Represents the possible outcomes after re-rolling the fifth die -/
inductive ReRollOutcome
  | fourOfAKind : ReRollOutcome
  | nothingSpecial : ReRollOutcome

/-- The probability of getting at least four of a kind after re-rolling -/
def probFourOfAKind (initialRoll : DiceRoll) : ℚ :=
  sorry

theorem prob_four_of_a_kind_after_reroll :
  ∀ (initialRoll : DiceRoll),
    initialRoll.pairs = 2 ∧ ¬initialRoll.fourOfAKind →
    probFourOfAKind initialRoll = 1 / 3 :=
  sorry

end prob_four_of_a_kind_after_reroll_l2415_241519


namespace minimal_divisible_number_l2415_241579

theorem minimal_divisible_number : ∃! n : ℕ,
  2007000 ≤ n ∧ n < 2008000 ∧
  n % 3 = 0 ∧ n % 5 = 0 ∧ n % 7 = 0 ∧
  (∀ m : ℕ, 2007000 ≤ m ∧ m < n → (m % 3 ≠ 0 ∨ m % 5 ≠ 0 ∨ m % 7 ≠ 0)) ∧
  n = 2007075 :=
sorry

end minimal_divisible_number_l2415_241579


namespace intersection_point_of_linear_system_l2415_241573

theorem intersection_point_of_linear_system (b : ℝ) :
  let eq1 : ℝ → ℝ → Prop := λ x y => x + y - b = 0
  let eq2 : ℝ → ℝ → Prop := λ x y => 3 * x + y - 2 = 0
  let line1 : ℝ → ℝ → Prop := λ x y => y = -x + b
  let line2 : ℝ → ℝ → Prop := λ x y => y = -3 * x + 2
  (∃ m, eq1 (-1) m ∧ eq2 (-1) m) →
  (∃! p : ℝ × ℝ, line1 p.1 p.2 ∧ line2 p.1 p.2 ∧ p = (-1, 5)) :=
by sorry

end intersection_point_of_linear_system_l2415_241573


namespace max_digit_sum_is_24_l2415_241528

/-- Represents a time in 24-hour format -/
structure Time24 where
  hours : Nat
  minutes : Nat
  hours_valid : hours < 24
  minutes_valid : minutes < 60

/-- Calculates the sum of digits for a given natural number -/
def sumOfDigits (n : Nat) : Nat :=
  if n < 10 then n else (n % 10) + sumOfDigits (n / 10)

/-- Calculates the sum of digits for a given time in 24-hour format -/
def timeDigitSum (t : Time24) : Nat :=
  sumOfDigits t.hours + sumOfDigits t.minutes

/-- The maximum possible sum of digits in a 24-hour format display -/
def maxDigitSum : Nat := 24

/-- Theorem stating that the maximum sum of digits in a 24-hour format display is 24 -/
theorem max_digit_sum_is_24 :
  ∀ t : Time24, timeDigitSum t ≤ maxDigitSum :=
by
  sorry

#check max_digit_sum_is_24

end max_digit_sum_is_24_l2415_241528


namespace all_parameterizations_valid_l2415_241526

/-- The slope of the line -/
def m : ℝ := -3

/-- The y-intercept of the line -/
def b : ℝ := 4

/-- The line equation: y = mx + b -/
def on_line (x y : ℝ) : Prop := y = m * x + b

/-- A parameterization is valid if it satisfies the line equation for all t -/
def valid_parameterization (p : ℝ × ℝ) (v : ℝ × ℝ) : Prop :=
  ∀ t : ℝ, on_line (p.1 + t * v.1) (p.2 + t * v.2)

/-- Theorem: All given parameterizations are valid -/
theorem all_parameterizations_valid :
  valid_parameterization (0, 4) (1, -3) ∧
  valid_parameterization (-2/3, 0) (3, -9) ∧
  valid_parameterization (-4/3, 8) (2, -6) ∧
  valid_parameterization (-2, 10) (1/2, -1) ∧
  valid_parameterization (1, 1) (4, -12) :=
sorry

end all_parameterizations_valid_l2415_241526


namespace Q_R_mutually_exclusive_l2415_241507

-- Define the sample space
structure Outcome :=
  (first : Bool) -- true for black, false for white
  (second : Bool)

-- Define the probability space
def Ω : Type := Outcome

-- Define the events
def P (ω : Ω) : Prop := ω.first ∧ ω.second
def Q (ω : Ω) : Prop := ¬ω.first ∧ ¬ω.second
def R (ω : Ω) : Prop := ω.first ∨ ω.second

-- State the theorem
theorem Q_R_mutually_exclusive : ∀ (ω : Ω), ¬(Q ω ∧ R ω) := by
  sorry

end Q_R_mutually_exclusive_l2415_241507


namespace G_1000_units_digit_l2415_241514

/-- Modified Fermat number -/
def G (n : ℕ) : ℕ := 3^(3^n) + 2

/-- Units digit of a natural number -/
def unitsDigit (n : ℕ) : ℕ := n % 10

theorem G_1000_units_digit :
  unitsDigit (G 1000) = 5 := by
  sorry

end G_1000_units_digit_l2415_241514


namespace average_value_sequence_l2415_241561

theorem average_value_sequence (x : ℝ) : 
  let sequence := [0, 3*x, 6*x, 12*x, 24*x]
  (sequence.sum / sequence.length : ℝ) = 9*x := by
  sorry

end average_value_sequence_l2415_241561


namespace total_differential_arcctg_l2415_241544

noncomputable def z (x y : ℝ) : ℝ := Real.arctan (y / x)

theorem total_differential_arcctg (x y dx dy : ℝ) (hx : x = 1) (hy : y = 3) (hdx : dx = 0.01) (hdy : dy = -0.05) :
  let dz := -(y / (x^2 + y^2)) * dx + (x / (x^2 + y^2)) * dy
  dz = -0.008 := by
  sorry

end total_differential_arcctg_l2415_241544


namespace function_equality_l2415_241575

theorem function_equality (f : ℤ → ℤ) :
  (∀ a b : ℤ, f (a^2 + b^2) + f (a * b) = f a ^ 2 + f b + 1) →
  (∀ a : ℤ, f a = 1) := by
sorry

end function_equality_l2415_241575


namespace basketball_game_properties_l2415_241583

/-- Represents the score of player A in a single round -/
inductive Score
  | Minus : Score  -- A loses the round
  | Zero : Score   -- Tie in the round
  | Plus : Score   -- A wins the round

/-- Represents the number of rounds played -/
inductive Rounds
  | Two : Rounds
  | Three : Rounds
  | Four : Rounds

/-- The basketball shooting game between A and B -/
structure BasketballGame where
  a_accuracy : ℝ
  b_accuracy : ℝ
  max_rounds : ℕ
  win_difference : ℕ

/-- The probability distribution of the score in a single round -/
def score_distribution (game : BasketballGame) : Score → ℝ
  | Score.Minus => game.b_accuracy * (1 - game.a_accuracy)
  | Score.Zero => game.a_accuracy * game.b_accuracy + (1 - game.a_accuracy) * (1 - game.b_accuracy)
  | Score.Plus => game.a_accuracy * (1 - game.b_accuracy)

/-- The probability of a tie in the game -/
def tie_probability (game : BasketballGame) : ℝ := sorry

/-- The probability distribution of the number of rounds played -/
def rounds_distribution (game : BasketballGame) : Rounds → ℝ
  | Rounds.Two => sorry
  | Rounds.Three => sorry
  | Rounds.Four => sorry

/-- The expected number of rounds played -/
def expected_rounds (game : BasketballGame) : ℝ := sorry

theorem basketball_game_properties (game : BasketballGame) 
  (h1 : game.a_accuracy = 0.5)
  (h2 : game.b_accuracy = 0.6)
  (h3 : game.max_rounds = 4)
  (h4 : game.win_difference = 4) :
  score_distribution game Score.Minus = 0.3 ∧ 
  score_distribution game Score.Zero = 0.5 ∧
  score_distribution game Score.Plus = 0.2 ∧
  tie_probability game = 0.2569 ∧
  rounds_distribution game Rounds.Two = 0.13 ∧
  rounds_distribution game Rounds.Three = 0.13 ∧
  rounds_distribution game Rounds.Four = 0.74 ∧
  expected_rounds game = 3.61 := by sorry

end basketball_game_properties_l2415_241583


namespace smallest_integer_quadratic_inequality_l2415_241595

theorem smallest_integer_quadratic_inequality :
  ∃ (n : ℤ), n^2 - 15*n + 56 ≤ 0 ∧ ∀ (m : ℤ), m^2 - 15*m + 56 ≤ 0 → n ≤ m :=
by sorry

end smallest_integer_quadratic_inequality_l2415_241595


namespace concentric_circles_area_ratio_l2415_241511

theorem concentric_circles_area_ratio :
  let d₁ : ℝ := 2  -- diameter of smaller circle
  let d₂ : ℝ := 6  -- diameter of larger circle
  let r₁ : ℝ := d₁ / 2  -- radius of smaller circle
  let r₂ : ℝ := d₂ / 2  -- radius of larger circle
  let area_small : ℝ := π * r₁^2  -- area of smaller circle
  let area_large : ℝ := π * r₂^2  -- area of larger circle
  let area_between : ℝ := area_large - area_small  -- area between circles
  (area_between / area_small) = 8 := by
  sorry

end concentric_circles_area_ratio_l2415_241511


namespace apple_count_l2415_241543

/-- Represents the total number of apples -/
def total_apples : ℕ := sorry

/-- Represents the price of a sweet apple in dollars -/
def sweet_price : ℚ := 1/2

/-- Represents the price of a sour apple in dollars -/
def sour_price : ℚ := 1/10

/-- Represents the proportion of sweet apples -/
def sweet_proportion : ℚ := 3/4

/-- Represents the proportion of sour apples -/
def sour_proportion : ℚ := 1/4

/-- Represents the total earnings in dollars -/
def total_earnings : ℚ := 40

theorem apple_count : 
  sweet_proportion * total_apples * sweet_price + 
  sour_proportion * total_apples * sour_price = total_earnings ∧
  total_apples = 100 := by sorry

end apple_count_l2415_241543


namespace problem_1_problem_2_l2415_241530

-- Problem 1
theorem problem_1 (x : ℝ) (h : x^2 - 2*x = 5) : 2*x^2 - 4*x + 2023 = 2033 := by
  sorry

-- Problem 2
theorem problem_2 (m n : ℝ) (h : m - n = -3) : 2*(m-n) - m + n + 5 = 2 := by
  sorry

end problem_1_problem_2_l2415_241530


namespace pipeline_theorem_l2415_241567

/-- Represents the pipeline construction problem -/
structure PipelineConstruction where
  total_length : ℝ
  daily_increase : ℝ
  days_ahead : ℝ
  actual_daily_length : ℝ

/-- The equation describing the pipeline construction problem -/
def pipeline_equation (p : PipelineConstruction) : Prop :=
  p.total_length / (p.actual_daily_length - p.daily_increase) -
  p.total_length / p.actual_daily_length = p.days_ahead

/-- Theorem stating that the equation holds for the given parameters -/
theorem pipeline_theorem (p : PipelineConstruction)
  (h1 : p.total_length = 4000)
  (h2 : p.daily_increase = 10)
  (h3 : p.days_ahead = 20) :
  pipeline_equation p :=
sorry

end pipeline_theorem_l2415_241567


namespace min_distance_to_line_l2415_241590

/-- The minimum distance from the origin to the line 2x - y + 1 = 0 is √5/5 -/
theorem min_distance_to_line : 
  let line := {(x, y) : ℝ × ℝ | 2 * x - y + 1 = 0}
  ∃ (d : ℝ), d = Real.sqrt 5 / 5 ∧ 
    (∀ (P : ℝ × ℝ), P ∈ line → d ≤ Real.sqrt (P.1^2 + P.2^2)) ∧
    (∃ (P : ℝ × ℝ), P ∈ line ∧ d = Real.sqrt (P.1^2 + P.2^2)) :=
by sorry


end min_distance_to_line_l2415_241590


namespace survey_respondents_l2415_241569

theorem survey_respondents (x y : ℕ) : 
  x = 60 → -- 60 people preferred brand X
  x = 3 * y → -- The ratio of preference for X to Y is 3:1
  x + y = 80 -- Total number of respondents
  :=
by
  sorry

end survey_respondents_l2415_241569


namespace no_double_application_increment_l2415_241597

theorem no_double_application_increment (f : ℕ → ℕ) : ∃ n : ℕ, n > 0 ∧ f (f n) ≠ n + 1 := by
  sorry

end no_double_application_increment_l2415_241597


namespace problem_2023_l2415_241525

theorem problem_2023 : (2023^2 - 2023) / 2023 = 2022 := by
  sorry

end problem_2023_l2415_241525


namespace unit_digit_of_large_exponentiation_l2415_241547

def unit_digit (n : ℕ) : ℕ := n % 10

theorem unit_digit_of_large_exponentiation : 
  unit_digit ((23^100000 * 56^150000) / Nat.gcd 23 56) = 6 := by
  sorry

end unit_digit_of_large_exponentiation_l2415_241547


namespace circle_area_increase_l2415_241586

theorem circle_area_increase (r : ℝ) (hr : r > 0) : 
  let new_radius := 2.5 * r
  let original_area := π * r^2
  let new_area := π * new_radius^2
  (new_area - original_area) / original_area = 5.25 := by
sorry

end circle_area_increase_l2415_241586


namespace season_games_count_l2415_241554

/-- The number of teams in the league -/
def num_teams : ℕ := 12

/-- The number of times each team plays every other team -/
def games_per_matchup : ℕ := 2

/-- The number of non-league games each team plays -/
def non_league_games : ℕ := 5

/-- The total number of games in a season -/
def total_games : ℕ := (num_teams * (num_teams - 1) / 2) * games_per_matchup + num_teams * non_league_games

theorem season_games_count : total_games = 192 := by
  sorry

end season_games_count_l2415_241554


namespace sufficiency_not_necessity_l2415_241587

theorem sufficiency_not_necessity (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (a + b = 2 → a * b ≤ 1) ∧
  ∃ (c d : ℝ), 0 < c ∧ 0 < d ∧ c * d ≤ 1 ∧ c + d ≠ 2 :=
by sorry

end sufficiency_not_necessity_l2415_241587


namespace isosceles_triangle_perimeter_isosceles_triangle_perimeter_proof_l2415_241551

/-- An isosceles triangle with side lengths 2 and 5 has a perimeter of 12 -/
theorem isosceles_triangle_perimeter : ℝ → Prop :=
  fun perimeter =>
    ∃ (a b : ℝ),
      a = 2 ∧ b = 5 ∧
      (a + b + b = perimeter ∨ a + a + b = perimeter) ∧
      perimeter = 12

/-- Proof of the theorem -/
theorem isosceles_triangle_perimeter_proof : isosceles_triangle_perimeter 12 := by
  sorry

end isosceles_triangle_perimeter_isosceles_triangle_perimeter_proof_l2415_241551


namespace triathlon_problem_l2415_241552

/-- Triathlon problem -/
theorem triathlon_problem (v₁ v₂ v₃ : ℝ) 
  (h1 : 1 / v₁ + 25 / v₂ + 4 / v₃ = 5 / 4)
  (h2 : v₁ / 16 + v₂ / 49 + v₃ / 49 = 5 / 4) :
  v₃ = 14 ∧ 4 / v₃ = 2 / 7 := by
  sorry

end triathlon_problem_l2415_241552


namespace farm_animals_count_l2415_241565

theorem farm_animals_count (total_animals : ℕ) (total_legs : ℕ) 
  (h1 : total_animals = 300) 
  (h2 : total_legs = 688) : 
  ∃ (ducks cows : ℕ), 
    ducks + cows = total_animals ∧ 
    2 * ducks + 4 * cows = total_legs ∧ 
    ducks = 256 := by
  sorry

end farm_animals_count_l2415_241565


namespace birds_landed_l2415_241509

/-- Given an initial number of birds on a fence and a final number of birds on the fence,
    this theorem proves that the number of birds that landed is equal to
    the difference between the final and initial numbers. -/
theorem birds_landed (initial final : ℕ) (h : initial ≤ final) :
  final - initial = final - initial :=
by sorry

end birds_landed_l2415_241509


namespace intersection_with_complement_l2415_241504

-- Define the sets
def P : Set ℝ := {1, 2, 3, 4}
def Q : Set ℝ := {3, 4, 5}
def U : Set ℝ := Set.univ

-- State the theorem
theorem intersection_with_complement :
  P ∩ (U \ Q) = {1, 2} := by sorry

end intersection_with_complement_l2415_241504
