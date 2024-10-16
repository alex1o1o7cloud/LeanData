import Mathlib

namespace NUMINAMATH_CALUDE_three_digit_divisibility_by_nine_l3019_301922

/-- Function to calculate the sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sumOfDigits (n / 10)

/-- Theorem stating that for all three-digit numbers, if the sum of digits is divisible by 9, then the number is divisible by 9 -/
theorem three_digit_divisibility_by_nine :
  ∀ n : ℕ, 100 ≤ n ∧ n < 1000 → (sumOfDigits n % 9 = 0 → n % 9 = 0) :=
by
  sorry

#check three_digit_divisibility_by_nine

end NUMINAMATH_CALUDE_three_digit_divisibility_by_nine_l3019_301922


namespace NUMINAMATH_CALUDE_lemonade_scaling_l3019_301987

/-- Represents a lemonade recipe -/
structure LemonadeRecipe where
  lemons : ℚ
  sugar : ℚ
  gallons : ℚ

/-- The original recipe -/
def originalRecipe : LemonadeRecipe :=
  { lemons := 30
  , sugar := 5
  , gallons := 40 }

/-- Calculate the amount of an ingredient needed for a given number of gallons -/
def calculateIngredient (original : LemonadeRecipe) (ingredient : ℚ) (targetGallons : ℚ) : ℚ :=
  (ingredient / original.gallons) * targetGallons

/-- The theorem to prove -/
theorem lemonade_scaling (recipe : LemonadeRecipe) (targetGallons : ℚ) :
  let scaledLemons := calculateIngredient recipe recipe.lemons targetGallons
  let scaledSugar := calculateIngredient recipe recipe.sugar targetGallons
  recipe.gallons = 40 ∧ recipe.lemons = 30 ∧ recipe.sugar = 5 ∧ targetGallons = 10 →
  scaledLemons = 7.5 ∧ scaledSugar = 1.25 := by
  sorry

end NUMINAMATH_CALUDE_lemonade_scaling_l3019_301987


namespace NUMINAMATH_CALUDE_binary_to_base_4_conversion_l3019_301903

-- Define the binary number
def binary_num : ℕ := 110110001

-- Define the base 4 number
def base_4_num : ℕ := 13201

-- Theorem statement
theorem binary_to_base_4_conversion :
  (binary_num : ℕ).digits 2 = [1, 1, 0, 1, 1, 0, 0, 0, 1] ∧
  (base_4_num : ℕ).digits 4 = [1, 3, 2, 0, 1] ∧
  binary_num = base_4_num :=
by sorry

end NUMINAMATH_CALUDE_binary_to_base_4_conversion_l3019_301903


namespace NUMINAMATH_CALUDE_triangle_dissection_theorem_l3019_301943

/-- A triangle in a 2D plane -/
structure Triangle where
  vertices : Fin 3 → ℝ × ℝ

/-- Represents a dissection of a triangle -/
def Dissection (t : Triangle) := List (List (ℝ × ℝ))

/-- Checks if two triangles are congruent -/
def are_congruent (t1 t2 : Triangle) : Prop := sorry

/-- Checks if one triangle is a reflection of another -/
def is_reflection (t1 t2 : Triangle) : Prop := sorry

/-- Checks if a dissection can transform one triangle to another using only translations -/
def can_transform_by_translation (d : Dissection t1) (t1 t2 : Triangle) : Prop := sorry

theorem triangle_dissection_theorem (t1 t2 : Triangle) :
  are_congruent t1 t2 → is_reflection t1 t2 →
  ∃ (d : Dissection t1), can_transform_by_translation d t1 t2 ∧ d.length ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_dissection_theorem_l3019_301943


namespace NUMINAMATH_CALUDE_vote_difference_is_42_l3019_301923

/-- Proves that the difference in votes for the bill between re-vote and original vote is 42 -/
theorem vote_difference_is_42 
  (total_members : ℕ) 
  (original_for original_against : ℕ) 
  (revote_for revote_against : ℕ) :
  total_members = 400 →
  original_for + original_against = total_members →
  original_against > original_for →
  revote_for + revote_against = total_members →
  revote_for > revote_against →
  (revote_for - revote_against) = 3 * (original_against - original_for) →
  revote_for = (11 * original_against) / 10 →
  revote_for - original_for = 42 := by
sorry


end NUMINAMATH_CALUDE_vote_difference_is_42_l3019_301923


namespace NUMINAMATH_CALUDE_jones_clothes_count_l3019_301963

/-- Represents the ratio of shirts to pants -/
def shirt_to_pants_ratio : ℕ := 6

/-- Represents the number of pants Mr. Jones owns -/
def pants_count : ℕ := 40

/-- Calculates the total number of clothes Mr. Jones owns -/
def total_clothes : ℕ := shirt_to_pants_ratio * pants_count + pants_count

/-- Proves that the total number of clothes Mr. Jones owns is 280 -/
theorem jones_clothes_count : total_clothes = 280 := by
  sorry

end NUMINAMATH_CALUDE_jones_clothes_count_l3019_301963


namespace NUMINAMATH_CALUDE_system_solution_l3019_301932

theorem system_solution (x y : ℝ) (h1 : x + 2*y = -1) (h2 : 2*x + y = 3) : x - y = 4 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l3019_301932


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l3019_301907

/-- The complex number i -/
def i : ℂ := Complex.I

/-- Predicate for the condition (a + bi)^2 = 2i -/
def condition (a b : ℝ) : Prop := (Complex.mk a b)^2 = 2*i

/-- Statement: a=b=1 is sufficient but not necessary for (a + bi)^2 = 2i -/
theorem sufficient_not_necessary :
  (∀ a b : ℝ, a = 1 ∧ b = 1 → condition a b) ∧
  (∃ a b : ℝ, condition a b ∧ (a ≠ 1 ∨ b ≠ 1)) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l3019_301907


namespace NUMINAMATH_CALUDE_two_balls_different_color_weight_l3019_301968

-- Define the types for color and weight
inductive Color : Type
| Red : Color
| Blue : Color

inductive Weight : Type
| Light : Weight
| Heavy : Weight

-- Define the Ball type
structure Ball :=
  (color : Color)
  (weight : Weight)

-- Define the theorem
theorem two_balls_different_color_weight 
  (balls : Set Ball)
  (h1 : ∀ b : Ball, b ∈ balls → (b.color = Color.Red ∨ b.color = Color.Blue))
  (h2 : ∀ b : Ball, b ∈ balls → (b.weight = Weight.Light ∨ b.weight = Weight.Heavy))
  (h3 : ∃ b : Ball, b ∈ balls ∧ b.color = Color.Red)
  (h4 : ∃ b : Ball, b ∈ balls ∧ b.color = Color.Blue)
  (h5 : ∃ b : Ball, b ∈ balls ∧ b.weight = Weight.Light)
  (h6 : ∃ b : Ball, b ∈ balls ∧ b.weight = Weight.Heavy)
  : ∃ b1 b2 : Ball, b1 ∈ balls ∧ b2 ∈ balls ∧ b1.color ≠ b2.color ∧ b1.weight ≠ b2.weight :=
by
  sorry

end NUMINAMATH_CALUDE_two_balls_different_color_weight_l3019_301968


namespace NUMINAMATH_CALUDE_last_digit_389_base5_is_4_l3019_301991

-- Define a function to convert a decimal number to base-5
def toBase5 (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec loop (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else loop (m / 5) ((m % 5) :: acc)
    loop n []

-- State the theorem
theorem last_digit_389_base5_is_4 :
  (toBase5 389).getLast? = some 4 :=
sorry

end NUMINAMATH_CALUDE_last_digit_389_base5_is_4_l3019_301991


namespace NUMINAMATH_CALUDE_min_abs_z_plus_2i_l3019_301925

theorem min_abs_z_plus_2i (z : ℂ) (h : Complex.abs (z^2 - 3) = Complex.abs (z * (z - 3*I))) :
  Complex.abs (z + 2*I) ≥ (7 + Real.sqrt 3) / 2 := by
  sorry

end NUMINAMATH_CALUDE_min_abs_z_plus_2i_l3019_301925


namespace NUMINAMATH_CALUDE_greatest_power_of_two_factor_l3019_301962

theorem greatest_power_of_two_factor (n : ℕ) : 
  ∃ (k : ℕ), (2^k : ℤ) ∣ (10^1004 - 4^502) ∧ 
  ∀ (m : ℕ), (2^m : ℤ) ∣ (10^1004 - 4^502) → m ≤ k :=
by
  use 1007
  sorry

#eval 1007  -- This will output the answer

end NUMINAMATH_CALUDE_greatest_power_of_two_factor_l3019_301962


namespace NUMINAMATH_CALUDE_a_minus_b_values_l3019_301965

theorem a_minus_b_values (a b : ℝ) (h1 : a < b) (h2 : |a| = 6) (h3 : |b| = 3) :
  a - b = -9 ∨ a - b = -3 := by
sorry

end NUMINAMATH_CALUDE_a_minus_b_values_l3019_301965


namespace NUMINAMATH_CALUDE_A_equals_set_l3019_301999

def A : Set ℕ := {x | 0 ≤ x ∧ x < 3}

theorem A_equals_set : A = {0, 1, 2} := by sorry

end NUMINAMATH_CALUDE_A_equals_set_l3019_301999


namespace NUMINAMATH_CALUDE_vasyas_birthday_l3019_301949

-- Define the days of the week
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

-- Define a function to get the next day
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

-- Define a function to get the day two days after
def twoDaysAfter (d : DayOfWeek) : DayOfWeek :=
  nextDay (nextDay d)

theorem vasyas_birthday (statementDay : DayOfWeek) 
  (h1 : twoDaysAfter statementDay = DayOfWeek.Sunday) :
  nextDay DayOfWeek.Thursday = statementDay :=
by
  sorry


end NUMINAMATH_CALUDE_vasyas_birthday_l3019_301949


namespace NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l3019_301931

theorem condition_necessary_not_sufficient :
  (∀ a b : ℝ, a + b ≠ 3 → (a ≠ 1 ∨ b ≠ 2)) ∧
  (∃ a b : ℝ, (a ≠ 1 ∨ b ≠ 2) ∧ a + b = 3) :=
by sorry

end NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l3019_301931


namespace NUMINAMATH_CALUDE_rectangle_length_given_perimeter_and_breadth_l3019_301939

/-- The perimeter of a rectangle given its length and breadth -/
def rectanglePerimeter (length width : ℝ) : ℝ := 2 * (length + width)

/-- Theorem: For a rectangular garden with perimeter 500 m and breadth 100 m, the length is 150 m -/
theorem rectangle_length_given_perimeter_and_breadth :
  ∀ length : ℝ, rectanglePerimeter length 100 = 500 → length = 150 := by
sorry

end NUMINAMATH_CALUDE_rectangle_length_given_perimeter_and_breadth_l3019_301939


namespace NUMINAMATH_CALUDE_exam_failure_rate_l3019_301941

theorem exam_failure_rate (total : ℝ) (failed_hindi : ℝ) (failed_both : ℝ) (passed_both : ℝ) :
  total = 100 →
  failed_hindi = 20 →
  failed_both = 10 →
  passed_both = 20 →
  ∃ failed_english : ℝ, failed_english = 70 :=
by sorry

end NUMINAMATH_CALUDE_exam_failure_rate_l3019_301941


namespace NUMINAMATH_CALUDE_both_hit_target_probability_l3019_301909

theorem both_hit_target_probability
  (prob_A : ℝ)
  (prob_B : ℝ)
  (h_A : prob_A = 0.8)
  (h_B : prob_B = 0.6) :
  prob_A * prob_B = 0.48 := by
sorry

end NUMINAMATH_CALUDE_both_hit_target_probability_l3019_301909


namespace NUMINAMATH_CALUDE_race_time_difference_l3019_301929

/-- The time difference between two runners in a race -/
def time_difference (distance : ℝ) (speed1 speed2 : ℝ) : ℝ :=
  distance * speed2 - distance * speed1

/-- Proof of the time difference in the race -/
theorem race_time_difference :
  let malcolm_speed : ℝ := 7
  let joshua_speed : ℝ := 8
  let race_distance : ℝ := 15
  time_difference race_distance malcolm_speed joshua_speed = 15 := by
  sorry

end NUMINAMATH_CALUDE_race_time_difference_l3019_301929


namespace NUMINAMATH_CALUDE_welders_who_left_l3019_301955

/-- Represents the problem of welders working on an order -/
structure WelderProblem where
  initial_welders : ℕ
  initial_days : ℕ
  remaining_days : ℕ
  welders_left : ℕ

/-- The specific problem instance -/
def problem : WelderProblem :=
  { initial_welders := 12
  , initial_days := 8
  , remaining_days := 28
  , welders_left := 3 }

/-- Theorem stating the number of welders who left for another project -/
theorem welders_who_left (p : WelderProblem) : 
  p.initial_welders - p.welders_left = 9 :=
by sorry

#check welders_who_left problem

end NUMINAMATH_CALUDE_welders_who_left_l3019_301955


namespace NUMINAMATH_CALUDE_least_clock_equivalent_hour_l3019_301926

theorem least_clock_equivalent_hour : ∃ (h : ℕ), 
  h > 3 ∧ 
  (∀ k : ℕ, k > 3 ∧ k < h → ¬(12 ∣ (k^2 - k))) ∧ 
  (12 ∣ (h^2 - h)) :=
by sorry

end NUMINAMATH_CALUDE_least_clock_equivalent_hour_l3019_301926


namespace NUMINAMATH_CALUDE_two_a_div_a_equals_two_l3019_301945

theorem two_a_div_a_equals_two (a : ℝ) (h : a ≠ 0) : 2 * a / a = 2 := by
  sorry

end NUMINAMATH_CALUDE_two_a_div_a_equals_two_l3019_301945


namespace NUMINAMATH_CALUDE_all_propositions_false_l3019_301952

-- Define the basic geometric concepts
def Line : Type := sorry
def Plane : Type := sorry

-- Define the geometric relations
def perpendicular (l1 l2 : Line) : Prop := sorry
def parallel (l1 l2 : Line) : Prop := sorry
def parallel_planes (p1 p2 : Plane) : Prop := sorry
def perpendicular_to_plane (l : Line) (p : Plane) : Prop := sorry
def angle_with_plane (l : Line) (p : Plane) : ℝ := sorry
def intersect (l1 l2 : Line) : Prop := sorry
def skew (l1 l2 : Line) : Prop := sorry

-- State the propositions
def proposition1 : Prop :=
  ∀ l1 l2 l3 : Line, perpendicular l1 l3 → perpendicular l2 l3 → parallel l1 l2

def proposition2 : Prop :=
  ∀ p1 p2 p3 : Plane, perpendicular_to_plane p1 p3 → perpendicular_to_plane p2 p3 → parallel_planes p1 p2

def proposition3 : Prop :=
  ∀ l1 l2 : Line, ∀ p : Plane, angle_with_plane l1 p = angle_with_plane l2 p → parallel l1 l2

def proposition4 : Prop :=
  ∀ l1 l2 l3 l4 : Line, skew l1 l2 → intersect l3 l1 → intersect l3 l2 → intersect l4 l1 → intersect l4 l2 → skew l3 l4

-- Theorem stating that all propositions are false
theorem all_propositions_false :
  ¬proposition1 ∧ ¬proposition2 ∧ ¬proposition3 ∧ ¬proposition4 := by sorry

end NUMINAMATH_CALUDE_all_propositions_false_l3019_301952


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l3019_301928

/-- The eccentricity of a hyperbola tangent to a specific circle -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∃ (x y : ℝ), (x - Real.sqrt 3)^2 + (y - 1)^2 = 3 ∧ 
    (x^2 / a^2 - y^2 / b^2 = 1) ∧ 
    ((Real.sqrt 3 * b - a)^2 = 3 * (b^2 + a^2) ∨ (Real.sqrt 3 * b + a)^2 = 3 * (b^2 + a^2))) →
  Real.sqrt (a^2 + b^2) / a = 2 * Real.sqrt 3 / 3 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l3019_301928


namespace NUMINAMATH_CALUDE_solution_set_real_iff_k_less_than_neg_three_l3019_301921

theorem solution_set_real_iff_k_less_than_neg_three (k : ℝ) :
  (∀ x : ℝ, |x + 1| - |x - 2| > k) ↔ k < -3 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_real_iff_k_less_than_neg_three_l3019_301921


namespace NUMINAMATH_CALUDE_elidas_name_length_l3019_301951

theorem elidas_name_length :
  ∀ (E A : ℕ),
  A = 2 * E - 2 →
  10 * ((E + A) / 2 : ℚ) = 65 →
  E = 5 :=
by sorry

end NUMINAMATH_CALUDE_elidas_name_length_l3019_301951


namespace NUMINAMATH_CALUDE_tori_test_score_l3019_301918

theorem tori_test_score (total : ℕ) (arithmetic : ℕ) (algebra : ℕ) (geometry : ℕ)
  (arithmetic_correct : ℚ) (algebra_correct : ℚ) (geometry_correct : ℚ)
  (passing_grade : ℚ) :
  total = 100 →
  arithmetic = 20 →
  algebra = 40 →
  geometry = 40 →
  arithmetic_correct = 4/5 →
  algebra_correct = 1/2 →
  geometry_correct = 7/10 →
  passing_grade = 13/20 →
  ↑⌈passing_grade * total⌉ - (↑⌊arithmetic_correct * arithmetic⌋ + 
    ↑⌊algebra_correct * algebra⌋ + ↑⌊geometry_correct * geometry⌋) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tori_test_score_l3019_301918


namespace NUMINAMATH_CALUDE_lcm_from_product_and_hcf_l3019_301904

theorem lcm_from_product_and_hcf (a b : ℕ+) 
  (h_product : a * b = 82500)
  (h_hcf : Nat.gcd a b = 55) :
  Nat.lcm a b = 1500 := by
  sorry

end NUMINAMATH_CALUDE_lcm_from_product_and_hcf_l3019_301904


namespace NUMINAMATH_CALUDE_g_50_solutions_l3019_301997

def g₀ (x : ℝ) : ℝ := x + |x - 50| - |x + 50|

def g (n : ℕ) (x : ℝ) : ℝ := 
  match n with
  | 0 => g₀ x
  | n + 1 => |g n x| - 2

theorem g_50_solutions : 
  ∃ (S : Finset ℝ), (∀ x ∈ S, g 50 x = 0) ∧ (∀ x ∉ S, g 50 x ≠ 0) ∧ Finset.card S = 4 := by
  sorry

end NUMINAMATH_CALUDE_g_50_solutions_l3019_301997


namespace NUMINAMATH_CALUDE_test_probabilities_l3019_301979

/-- The probability that exactly two out of three students pass their tests. -/
def prob_two_pass (pA pB pC : ℚ) : ℚ :=
  pA * pB * (1 - pC) + pA * (1 - pB) * pC + (1 - pA) * pB * pC

/-- The probability that at least one out of three students fails their test. -/
def prob_at_least_one_fail (pA pB pC : ℚ) : ℚ :=
  1 - pA * pB * pC

theorem test_probabilities (pA pB pC : ℚ) 
  (hA : pA = 4/5) (hB : pB = 3/5) (hC : pC = 7/10) : 
  prob_two_pass pA pB pC = 113/250 ∧ 
  prob_at_least_one_fail pA pB pC = 83/125 := by
  sorry

#eval prob_two_pass (4/5) (3/5) (7/10)
#eval prob_at_least_one_fail (4/5) (3/5) (7/10)

end NUMINAMATH_CALUDE_test_probabilities_l3019_301979


namespace NUMINAMATH_CALUDE_dogwood_tree_count_l3019_301944

/-- The number of dogwood trees in the park after planting and accounting for losses -/
def final_tree_count (initial_trees : ℕ) 
                     (worker_a_trees : ℕ) 
                     (worker_b_trees : ℕ) 
                     (worker_c_trees : ℕ) 
                     (worker_d_trees : ℕ) 
                     (worker_e_trees : ℕ) 
                     (worker_c_losses : ℕ) 
                     (worker_d_losses : ℕ) : ℕ :=
  initial_trees + 
  worker_a_trees + 
  worker_b_trees + 
  (worker_c_trees - worker_c_losses) + 
  (worker_d_trees - worker_d_losses) + 
  worker_e_trees

/-- Theorem stating that the final number of dogwood trees in the park is 80 -/
theorem dogwood_tree_count : 
  final_tree_count 34 12 10 15 8 4 2 1 = 80 := by
  sorry

end NUMINAMATH_CALUDE_dogwood_tree_count_l3019_301944


namespace NUMINAMATH_CALUDE_minimum_seats_for_adjacent_seating_l3019_301908

/-- Represents a seating arrangement in a row of seats. -/
structure SeatingArrangement where
  total_seats : ℕ
  occupied_seats : ℕ
  max_gap : ℕ

/-- Checks if a seating arrangement is valid. -/
def is_valid_arrangement (s : SeatingArrangement) : Prop :=
  s.occupied_seats ≤ s.total_seats ∧ 
  s.max_gap ≤ 2

/-- Checks if adding one more person would force them to sit next to someone. -/
def forces_adjacent_seating (s : SeatingArrangement) : Prop :=
  s.max_gap ≤ 1

/-- The main theorem to prove. -/
theorem minimum_seats_for_adjacent_seating :
  ∃ (s : SeatingArrangement),
    s.total_seats = 150 ∧
    s.occupied_seats = 30 ∧
    is_valid_arrangement s ∧
    forces_adjacent_seating s ∧
    (∀ (s' : SeatingArrangement),
      s'.total_seats = 150 →
      s'.occupied_seats < 30 →
      is_valid_arrangement s' →
      ¬forces_adjacent_seating s') :=
sorry

end NUMINAMATH_CALUDE_minimum_seats_for_adjacent_seating_l3019_301908


namespace NUMINAMATH_CALUDE_inverse_function_property_l3019_301970

theorem inverse_function_property (f g : ℝ → ℝ) (h_inverse : Function.RightInverse g f ∧ Function.LeftInverse g f)
  (h_property : ∀ a b : ℝ, f (a * b) = f a + f b) :
  ∀ a b : ℝ, g (a + b) = g a * g b :=
by sorry

end NUMINAMATH_CALUDE_inverse_function_property_l3019_301970


namespace NUMINAMATH_CALUDE_video_dislikes_calculation_l3019_301975

/-- Given a video with likes and dislikes, calculate the final number of dislikes after an increase -/
def final_dislikes (likes : ℕ) (initial_extra_dislikes : ℕ) (dislike_increase : ℕ) : ℕ :=
  likes / 2 + initial_extra_dislikes + dislike_increase

/-- Theorem stating that for a video with 3000 likes and 100 more than half as many dislikes,
    after getting 1000 more dislikes, the total number of dislikes is 2600 -/
theorem video_dislikes_calculation :
  final_dislikes 3000 100 1000 = 2600 := by
  sorry

#eval final_dislikes 3000 100 1000

end NUMINAMATH_CALUDE_video_dislikes_calculation_l3019_301975


namespace NUMINAMATH_CALUDE_towel_rate_proof_l3019_301980

/-- Proves that given the specified towel purchases and average price, the unknown rate must be 250. -/
theorem towel_rate_proof (num_towels_1 num_towels_2 num_towels_unknown : ℕ)
  (price_1 price_2 avg_price : ℚ) :
  num_towels_1 = 3 →
  num_towels_2 = 5 →
  num_towels_unknown = 2 →
  price_1 = 100 →
  price_2 = 150 →
  avg_price = 155 →
  let total_towels := num_towels_1 + num_towels_2 + num_towels_unknown
  let total_cost := num_towels_1 * price_1 + num_towels_2 * price_2 + num_towels_unknown * avg_price
  (total_cost / total_towels : ℚ) = avg_price →
  (((total_cost - (num_towels_1 * price_1 + num_towels_2 * price_2)) / num_towels_unknown) : ℚ) = 250 :=
by sorry

end NUMINAMATH_CALUDE_towel_rate_proof_l3019_301980


namespace NUMINAMATH_CALUDE_rect_to_cylindrical_coordinates_l3019_301947

/-- Given a point with rectangular coordinates (1, 1, 1), 
    its cylindrical coordinates are (√2, π/4, 1) -/
theorem rect_to_cylindrical_coordinates : 
  let x : ℝ := 1
  let y : ℝ := 1
  let z : ℝ := 1
  let ρ : ℝ := Real.sqrt 2
  let θ : ℝ := π / 4
  x = ρ * Real.cos θ ∧ 
  y = ρ * Real.sin θ ∧ 
  z = z := by sorry

end NUMINAMATH_CALUDE_rect_to_cylindrical_coordinates_l3019_301947


namespace NUMINAMATH_CALUDE_de_Bruijn_Erdos_l3019_301946

/-- A graph is a pair of a vertex set and an edge relation -/
structure Graph (V : Type) :=
  (edge : V → V → Prop)

/-- The chromatic number of a graph is the smallest number of colors needed to color the graph -/
def chromaticNumber {V : Type} (G : Graph V) : ℕ := sorry

/-- A subgraph of G induced by a subset of vertices -/
def inducedSubgraph {V : Type} (G : Graph V) (S : Set V) : Graph S := sorry

/-- A graph is finite if its vertex set is finite -/
def isFinite {V : Type} (G : Graph V) : Prop := sorry

theorem de_Bruijn_Erdos {V : Type} (G : Graph V) (k : ℕ) :
  (∀ (S : Set V), isFinite (inducedSubgraph G S) → chromaticNumber (inducedSubgraph G S) ≤ k) →
  chromaticNumber G ≤ k := by sorry

end NUMINAMATH_CALUDE_de_Bruijn_Erdos_l3019_301946


namespace NUMINAMATH_CALUDE_inequality_solution_l3019_301967

theorem inequality_solution (m n : ℝ) :
  (∀ x, 2 * m * x + 3 < 3 * x + n ↔
    ((2 * m - 3 > 0 ∧ x < (n - 3) / (2 * m - 3)) ∨
     (2 * m - 3 < 0 ∧ x > (n - 3) / (2 * m - 3)) ∨
     (m = 3 / 2 ∧ n > 3) ∨
     (m = 3 / 2 ∧ n ≤ 3 ∧ False))) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l3019_301967


namespace NUMINAMATH_CALUDE_intersection_condition_union_condition_l3019_301974

-- Define set A
def A : Set ℝ := {x | x^2 - 3*x + 2 = 0}

-- Define set B (parameterized by a)
def B (a : ℝ) : Set ℝ := {x | x^2 + 2*(a-1)*x + (a^2-5) = 0}

-- Theorem for part (1)
theorem intersection_condition (a : ℝ) : A ∩ B a = {2} → a = -5 ∨ a = 1 := by sorry

-- Theorem for part (2)
theorem union_condition (a : ℝ) : A ∪ B a = A → a > 3 := by sorry

end NUMINAMATH_CALUDE_intersection_condition_union_condition_l3019_301974


namespace NUMINAMATH_CALUDE_prism_volume_l3019_301938

/-- The volume of a right rectangular prism given its face areas -/
theorem prism_volume (a b c : ℝ) 
  (h1 : a * b = 30)
  (h2 : a * c = 40)
  (h3 : b * c = 60) :
  a * b * c = 120 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_prism_volume_l3019_301938


namespace NUMINAMATH_CALUDE_triangle_side_length_l3019_301990

/-- Given a triangle ABC with side lengths a, b, c opposite to angles A, B, C respectively,
    if a = 2, c = 2√3, and C = π/3, then b = 4 -/
theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) :
  a = 2 →
  c = 2 * Real.sqrt 3 →
  C = π / 3 →
  c^2 = a^2 + b^2 - 2*a*b*(Real.cos C) →
  b = 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3019_301990


namespace NUMINAMATH_CALUDE_binomial_unique_parameters_l3019_301913

/-- A random variable following a binomial distribution -/
structure BinomialRV where
  n : ℕ
  p : ℝ
  h1 : 0 ≤ p ∧ p ≤ 1

/-- The expectation of a linear transformation of a binomial random variable -/
def expectation (X : BinomialRV) (a b : ℝ) : ℝ := a * X.n * X.p + b

/-- The variance of a linear transformation of a binomial random variable -/
def variance (X : BinomialRV) (a : ℝ) : ℝ := a^2 * X.n * X.p * (1 - X.p)

/-- Theorem: If E(3X + 2) = 9.2 and D(3X + 2) = 12.96 for X ~ B(n, p), then n = 6 and p = 0.4 -/
theorem binomial_unique_parameters (X : BinomialRV) 
  (h2 : expectation X 3 2 = 9.2)
  (h3 : variance X 3 = 12.96) : 
  X.n = 6 ∧ X.p = 0.4 := by
  sorry

end NUMINAMATH_CALUDE_binomial_unique_parameters_l3019_301913


namespace NUMINAMATH_CALUDE_min_value_quadratic_roots_l3019_301981

theorem min_value_quadratic_roots (a b c : ℤ) (α β : ℝ) : 
  a > 0 → 
  (∃ x : ℝ, a * x^2 + b * x + c = 0) → 
  (α * α * a + b * α + c = 0) →
  (β * β * a + b * β + c = 0) →
  0 < α → α < β → β < 1 → 
  a ≥ 5 := by
sorry

end NUMINAMATH_CALUDE_min_value_quadratic_roots_l3019_301981


namespace NUMINAMATH_CALUDE_union_of_A_and_complement_of_B_l3019_301956

open Set

theorem union_of_A_and_complement_of_B (A B : Set ℝ) : 
  A = {x : ℝ | x^2 - 4*x - 12 < 0} →
  B = {x : ℝ | x < 2} →
  A ∪ (univ \ B) = {x : ℝ | x > -2} := by
sorry

end NUMINAMATH_CALUDE_union_of_A_and_complement_of_B_l3019_301956


namespace NUMINAMATH_CALUDE_parity_and_squares_equivalence_l3019_301976

theorem parity_and_squares_equivalence (a b : ℕ) (ha : 0 < a) (hb : 0 < b) :
  (a % 2 = b % 2) ↔ (∃ (c d : ℕ), 0 < c ∧ 0 < d ∧ a^2 + b^2 + c^2 + 1 = d^2) := by
  sorry

end NUMINAMATH_CALUDE_parity_and_squares_equivalence_l3019_301976


namespace NUMINAMATH_CALUDE_equal_distribution_l3019_301912

theorem equal_distribution (total_amount : ℕ) (num_persons : ℕ) (amount_per_person : ℕ) : 
  total_amount = 42900 →
  num_persons = 22 →
  amount_per_person = total_amount / num_persons →
  amount_per_person = 1950 := by
  sorry

end NUMINAMATH_CALUDE_equal_distribution_l3019_301912


namespace NUMINAMATH_CALUDE_multiply_three_negative_two_l3019_301959

theorem multiply_three_negative_two : 3 * (-2) = -6 := by
  sorry

end NUMINAMATH_CALUDE_multiply_three_negative_two_l3019_301959


namespace NUMINAMATH_CALUDE_imaginary_part_of_i_times_one_plus_i_l3019_301950

theorem imaginary_part_of_i_times_one_plus_i (i : ℂ) : i * i = -1 → Complex.im (i * (1 + i)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_i_times_one_plus_i_l3019_301950


namespace NUMINAMATH_CALUDE_negation_of_universal_positive_square_plus_two_l3019_301927

theorem negation_of_universal_positive_square_plus_two :
  (¬ ∀ x : ℝ, x^2 + 2 > 0) ↔ (∃ x : ℝ, x^2 + 2 ≤ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_universal_positive_square_plus_two_l3019_301927


namespace NUMINAMATH_CALUDE_liu_hui_perimeter_l3019_301948

-- Define the right triangle
def right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2

-- Define the parallelogram formed by Liu Hui block puzzle
def liu_hui_parallelogram (a b c : ℝ) : Prop :=
  right_triangle a b c ∧ a > 0 ∧ b > 0 ∧ c > 0

-- Theorem statement
theorem liu_hui_perimeter (a b c : ℝ) :
  liu_hui_parallelogram a b c → a = 3 → b = 4 → 2 * (c + a + b) = 24 := by
  sorry


end NUMINAMATH_CALUDE_liu_hui_perimeter_l3019_301948


namespace NUMINAMATH_CALUDE_mixed_number_calculation_l3019_301935

theorem mixed_number_calculation : 
  53 * ((3 + 1/5) - (4 + 1/2)) / ((2 + 3/4) + (1 + 2/3)) = -(15 + 3/5) := by sorry

end NUMINAMATH_CALUDE_mixed_number_calculation_l3019_301935


namespace NUMINAMATH_CALUDE_jane_daniel_difference_l3019_301992

/-- The width of the streets in Newville -/
def street_width : ℝ := 30

/-- The length of one side of a square block in Newville -/
def block_side : ℝ := 500

/-- The length of Daniel's path around one block -/
def daniel_lap : ℝ := 4 * block_side

/-- The length of Jane's path around one block -/
def jane_lap : ℝ := 4 * (block_side + street_width)

/-- The theorem stating the difference between Jane's and Daniel's lap distances -/
theorem jane_daniel_difference : jane_lap - daniel_lap = 120 := by
  sorry

end NUMINAMATH_CALUDE_jane_daniel_difference_l3019_301992


namespace NUMINAMATH_CALUDE_max_value_of_t_l3019_301937

theorem max_value_of_t (x y : ℝ) (hx : x > 0) (hy : y > 0) : 
  (∀ a b : ℝ, a > 0 → b > 0 → min a (b / (a^2 + b^2)) ≤ 1) ∧ 
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ min a (b / (a^2 + b^2)) = 1) :=
sorry

end NUMINAMATH_CALUDE_max_value_of_t_l3019_301937


namespace NUMINAMATH_CALUDE_marble_bag_total_l3019_301986

/-- Represents the total number of marbles in a bag with red, blue, and green marbles. -/
def total_marbles (red : ℕ) (blue : ℕ) (green : ℕ) : ℕ := red + blue + green

/-- Theorem: Given a bag of marbles with only red, blue, and green marbles,
    where the ratio of red to blue to green marbles is 2:3:4,
    and there are 36 blue marbles, the total number of marbles in the bag is 108. -/
theorem marble_bag_total :
  ∀ (red blue green : ℕ),
  red = 2 * n ∧ blue = 3 * n ∧ green = 4 * n →
  blue = 36 →
  total_marbles red blue green = 108 :=
by
  sorry

end NUMINAMATH_CALUDE_marble_bag_total_l3019_301986


namespace NUMINAMATH_CALUDE_fourth_power_difference_not_prime_l3019_301905

theorem fourth_power_difference_not_prime (p q : ℕ) (hp : Prime p) (hq : Prime q) (hne : p ≠ q) :
  ¬ Prime (p^4 - q^4) := by
  sorry

end NUMINAMATH_CALUDE_fourth_power_difference_not_prime_l3019_301905


namespace NUMINAMATH_CALUDE_total_snake_owners_l3019_301964

theorem total_snake_owners (total : Nat) (only_dogs : Nat) (only_cats : Nat) (only_birds : Nat) (only_snakes : Nat)
  (cats_and_dogs : Nat) (birds_and_dogs : Nat) (birds_and_cats : Nat) (snakes_and_dogs : Nat) (snakes_and_cats : Nat)
  (snakes_and_birds : Nat) (cats_dogs_snakes : Nat) (cats_dogs_birds : Nat) (cats_birds_snakes : Nat)
  (dogs_birds_snakes : Nat) (all_four : Nat)
  (h1 : total = 150)
  (h2 : only_dogs = 30)
  (h3 : only_cats = 25)
  (h4 : only_birds = 10)
  (h5 : only_snakes = 7)
  (h6 : cats_and_dogs = 15)
  (h7 : birds_and_dogs = 12)
  (h8 : birds_and_cats = 8)
  (h9 : snakes_and_dogs = 3)
  (h10 : snakes_and_cats = 4)
  (h11 : snakes_and_birds = 2)
  (h12 : cats_dogs_snakes = 5)
  (h13 : cats_dogs_birds = 4)
  (h14 : cats_birds_snakes = 6)
  (h15 : dogs_birds_snakes = 9)
  (h16 : all_four = 10) :
  only_snakes + snakes_and_dogs + snakes_and_cats + snakes_and_birds + cats_dogs_snakes + cats_birds_snakes + dogs_birds_snakes + all_four = 46 := by
  sorry

end NUMINAMATH_CALUDE_total_snake_owners_l3019_301964


namespace NUMINAMATH_CALUDE_exists_valid_matrix_l3019_301901

def is_valid_matrix (M : Matrix (Fin 4) (Fin 4) ℤ) : Prop :=
  (∀ i j, M i j ≠ 0) ∧
  (∀ i j, i + 1 < 4 → j + 1 < 4 →
    M i j + M (i + 1) j + M i (j + 1) + M (i + 1) (j + 1) = 0) ∧
  (∀ i j, i + 2 < 4 → j + 2 < 4 →
    M i j + M (i + 2) j + M i (j + 2) + M (i + 2) (j + 2) = 0) ∧
  (M 0 0 + M 0 3 + M 3 0 + M 3 3 = 0)

theorem exists_valid_matrix : ∃ M : Matrix (Fin 4) (Fin 4) ℤ, is_valid_matrix M := by
  sorry

end NUMINAMATH_CALUDE_exists_valid_matrix_l3019_301901


namespace NUMINAMATH_CALUDE_hippopotamus_crayons_l3019_301958

/-- The number of crayons eaten by a hippopotamus --/
def crayonsEaten (initial final : ℕ) : ℕ := initial - final

/-- Theorem: The number of crayons eaten by the hippopotamus is the difference between 
    the initial and final number of crayons --/
theorem hippopotamus_crayons (initial final : ℕ) (h : initial ≥ final) :
  crayonsEaten initial final = initial - final := by
  sorry

/-- Given Jane's initial and final crayon counts, calculate how many were eaten --/
def janesCrayons : ℕ := 
  let initial := 87
  let final := 80
  crayonsEaten initial final

#eval janesCrayons  -- Should output 7

end NUMINAMATH_CALUDE_hippopotamus_crayons_l3019_301958


namespace NUMINAMATH_CALUDE_cube_opposite_face_l3019_301919

-- Define a cube type
structure Cube :=
  (faces : Fin 6 → Char)

-- Define adjacency relation
def adjacent (c : Cube) (x y : Char) : Prop :=
  ∃ (i j : Fin 6), i ≠ j ∧ c.faces i = x ∧ c.faces j = y

-- Define opposite relation
def opposite (c : Cube) (x y : Char) : Prop :=
  ∃ (i j : Fin 6), i ≠ j ∧ c.faces i = x ∧ c.faces j = y ∧
  ∀ (k : Fin 6), k ≠ i → k ≠ j → ¬(adjacent c (c.faces i) (c.faces k) ∧ adjacent c (c.faces j) (c.faces k))

theorem cube_opposite_face (c : Cube) :
  (c.faces = λ i => ['А', 'Б', 'В', 'Г', 'Д', 'Е'][i]) →
  (adjacent c 'А' 'Б') →
  (adjacent c 'А' 'Г') →
  (adjacent c 'Г' 'Д') →
  (adjacent c 'Г' 'Е') →
  (adjacent c 'В' 'Д') →
  (adjacent c 'В' 'Б') →
  opposite c 'Д' 'Б' :=
by sorry

end NUMINAMATH_CALUDE_cube_opposite_face_l3019_301919


namespace NUMINAMATH_CALUDE_line_up_arrangements_l3019_301993

def number_of_people : ℕ := 5
def number_of_youngest : ℕ := 2

theorem line_up_arrangements :
  (number_of_people.factorial - 
   (number_of_youngest * (number_of_people - 1).factorial)) = 72 :=
by sorry

end NUMINAMATH_CALUDE_line_up_arrangements_l3019_301993


namespace NUMINAMATH_CALUDE_complex_equation_solutions_l3019_301977

theorem complex_equation_solutions :
  ∃ (s : Finset ℂ), (∀ z ∈ s, (z^4 + 1) / (z^2 - z - 2) = 0) ∧ s.card = 4 :=
by
  -- We define the numerator and denominator polynomials
  let num := fun (z : ℂ) ↦ z^4 + 1
  let den := fun (z : ℂ) ↦ z^2 - z - 2

  -- We assume the factorizations given in the problem
  have h_num : ∀ z, num z = (z^2 + Real.sqrt 2 * z + 1) * (z^2 - Real.sqrt 2 * z + 1) := by sorry
  have h_den : ∀ z, den z = (z - 2) * (z + 1) := by sorry

  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_complex_equation_solutions_l3019_301977


namespace NUMINAMATH_CALUDE_expression_a_equality_l3019_301906

theorem expression_a_equality : 7 * (2/3) + 16 * (5/12) = 11 + 1/3 := by
  sorry

end NUMINAMATH_CALUDE_expression_a_equality_l3019_301906


namespace NUMINAMATH_CALUDE_decreasing_f_implies_a_range_l3019_301920

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then x^2 - a*x + 5 else a/x

def is_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f y < f x

theorem decreasing_f_implies_a_range (a : ℝ) :
  is_decreasing (f a) → 2 ≤ a ∧ a ≤ 3 :=
by sorry

end NUMINAMATH_CALUDE_decreasing_f_implies_a_range_l3019_301920


namespace NUMINAMATH_CALUDE_hyperbola_foci_l3019_301910

/-- The hyperbola equation -/
def hyperbola_equation (x y k : ℝ) : Prop :=
  x^2 / (1 + k^2) - y^2 / (8 - k^2) = 1

/-- The foci coordinates -/
def foci_coordinates : Set (ℝ × ℝ) := {(-3, 0), (3, 0)}

theorem hyperbola_foci (k : ℝ) (h : 1 + k^2 > 0) :
  ∃ (x y : ℝ), hyperbola_equation x y k →
  (x, y) ∈ foci_coordinates :=
sorry

end NUMINAMATH_CALUDE_hyperbola_foci_l3019_301910


namespace NUMINAMATH_CALUDE_probability_two_zeros_not_adjacent_l3019_301969

/-- The number of ways to arrange n ones and k zeros in a row -/
def totalArrangements (n k : ℕ) : ℕ :=
  Nat.choose (n + k) k

/-- The number of ways to arrange n ones and k zeros in a row where the zeros are not adjacent -/
def favorableArrangements (n k : ℕ) : ℕ :=
  Nat.choose (n + 1) k

/-- The probability that k zeros are not adjacent when arranged with n ones in a row -/
def probabilityNonAdjacentZeros (n k : ℕ) : ℚ :=
  (favorableArrangements n k : ℚ) / (totalArrangements n k : ℚ)

theorem probability_two_zeros_not_adjacent :
  probabilityNonAdjacentZeros 4 2 = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_zeros_not_adjacent_l3019_301969


namespace NUMINAMATH_CALUDE_problem_statement_l3019_301998

theorem problem_statement (p : Prop) (q : Prop)
  (hp : p ↔ ∃ x₀ : ℝ, Real.exp x₀ ≤ 0)
  (hq : q ↔ ∀ x : ℝ, 2^x > x^2) :
  (¬p) ∨ q := by sorry

end NUMINAMATH_CALUDE_problem_statement_l3019_301998


namespace NUMINAMATH_CALUDE_tom_golf_performance_l3019_301924

/-- Represents a round of golf --/
structure GolfRound where
  holes : ℕ
  averageStrokes : ℚ
  parValue : ℕ

/-- Calculates the total strokes for a round --/
def totalStrokes (round : GolfRound) : ℚ :=
  round.averageStrokes * round.holes

/-- Calculates the par for a round --/
def parForRound (round : GolfRound) : ℕ :=
  round.parValue * round.holes

theorem tom_golf_performance :
  let rounds : List GolfRound := [
    { holes := 9, averageStrokes := 4, parValue := 3 },
    { holes := 9, averageStrokes := 3.5, parValue := 3 },
    { holes := 9, averageStrokes := 5, parValue := 3 },
    { holes := 9, averageStrokes := 3, parValue := 3 },
    { holes := 9, averageStrokes := 4.5, parValue := 3 }
  ]
  let totalStrokesTaken := (rounds.map totalStrokes).sum
  let totalPar := (rounds.map parForRound).sum
  totalStrokesTaken - totalPar = 45 := by sorry

end NUMINAMATH_CALUDE_tom_golf_performance_l3019_301924


namespace NUMINAMATH_CALUDE_min_value_x2_y2_l3019_301902

theorem min_value_x2_y2 (x y : ℝ) (h : (x + 5)^2 + (y - 12)^2 = 196) :
  ∃ (m : ℝ), m = 169 ∧ ∀ (a b : ℝ), (a + 5)^2 + (b - 12)^2 = 196 → x^2 + y^2 ≤ a^2 + b^2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_x2_y2_l3019_301902


namespace NUMINAMATH_CALUDE_constant_term_binomial_expansion_l3019_301989

/-- The constant term in the expansion of (x^2 - 2/√x)^5 is 80 -/
theorem constant_term_binomial_expansion :
  (∃ (c : ℝ), c = 80 ∧ 
   ∀ (x : ℝ), x > 0 → 
   ∃ (f : ℝ → ℝ), (λ x => (x^2 - 2/Real.sqrt x)^5) = (λ x => f x + c)) := by
sorry

end NUMINAMATH_CALUDE_constant_term_binomial_expansion_l3019_301989


namespace NUMINAMATH_CALUDE_prob_same_flavor_is_one_fourth_l3019_301995

/-- The number of flavors available -/
def num_flavors : ℕ := 4

/-- The probability of selecting two bags of biscuits with the same flavor -/
def prob_same_flavor : ℚ := 1 / 4

/-- Theorem: The probability of selecting two bags of biscuits with the same flavor
    out of four possible flavors is 1/4 -/
theorem prob_same_flavor_is_one_fourth :
  prob_same_flavor = 1 / 4 := by sorry

end NUMINAMATH_CALUDE_prob_same_flavor_is_one_fourth_l3019_301995


namespace NUMINAMATH_CALUDE_barefoot_kids_l3019_301972

theorem barefoot_kids (total : ℕ) (with_socks : ℕ) (with_shoes : ℕ) (with_both : ℕ) : 
  total = 22 →
  with_socks = 12 →
  with_shoes = 8 →
  with_both = 6 →
  total - (with_socks + with_shoes - with_both) = 8 :=
by sorry

end NUMINAMATH_CALUDE_barefoot_kids_l3019_301972


namespace NUMINAMATH_CALUDE_solution_set_for_a_4_range_of_a_for_f_leq_4_l3019_301934

-- Define the function f
def f (x a : ℝ) : ℝ := |x - 1| + |x - a|

-- Theorem for part (1)
theorem solution_set_for_a_4 :
  {x : ℝ | f x 4 ≥ 5} = {x : ℝ | x ≤ 0 ∨ x ≥ 5} := by sorry

-- Theorem for part (2)
theorem range_of_a_for_f_leq_4 :
  {a : ℝ | ∃ x, f x a ≤ 4} = {a : ℝ | -3 ≤ a ∧ a ≤ 5} := by sorry

end NUMINAMATH_CALUDE_solution_set_for_a_4_range_of_a_for_f_leq_4_l3019_301934


namespace NUMINAMATH_CALUDE_set_357_forms_triangle_l3019_301930

/-- Triangle inequality theorem: the sum of any two sides must be greater than the third side --/
def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- A set of three line segments can form a triangle if it satisfies the triangle inequality --/
def can_form_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ triangle_inequality a b c

/-- Theorem: The set (3, 5, 7) can form a triangle --/
theorem set_357_forms_triangle : can_form_triangle 3 5 7 := by
  sorry

end NUMINAMATH_CALUDE_set_357_forms_triangle_l3019_301930


namespace NUMINAMATH_CALUDE_combine_like_terms_to_zero_l3019_301917

theorem combine_like_terms_to_zero (x y : ℝ) : -2 * x * y^2 + 2 * x * y^2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_combine_like_terms_to_zero_l3019_301917


namespace NUMINAMATH_CALUDE_odd_square_not_representable_l3019_301914

def divisor_count (k : ℕ+) : ℕ := (Nat.divisors k.val).card

theorem odd_square_not_representable (M : ℕ+) (h_odd : Odd M.val) (h_square : ∃ k : ℕ+, M = k * k) :
  ¬∃ n : ℕ+, (M : ℚ) = (2 * Real.sqrt n.val / divisor_count n) ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_odd_square_not_representable_l3019_301914


namespace NUMINAMATH_CALUDE_value_of_a_l3019_301954

theorem value_of_a (A B : Set ℕ) (a : ℕ) 
  (hA : A = {1, 2})
  (hB : B = {2, a})
  (hUnion : A ∪ B = {1, 2, 4}) :
  a = 4 := by
sorry

end NUMINAMATH_CALUDE_value_of_a_l3019_301954


namespace NUMINAMATH_CALUDE_price_reduction_for_same_profit_no_solution_for_460_profit_l3019_301984

/-- Represents the fruit sales scenario at Huimin Fresh Supermarket -/
structure FruitSales where
  cost_price : ℝ
  initial_selling_price : ℝ
  initial_daily_sales : ℝ
  sales_increase_rate : ℝ

/-- Calculates the daily profit given a price reduction -/
def daily_profit (fs : FruitSales) (price_reduction : ℝ) : ℝ :=
  (fs.initial_selling_price - price_reduction - fs.cost_price) *
  (fs.initial_daily_sales + fs.sales_increase_rate * price_reduction)

/-- The scenario described in the problem -/
def huimin_scenario : FruitSales := {
  cost_price := 20
  initial_selling_price := 40
  initial_daily_sales := 20
  sales_increase_rate := 2
}

theorem price_reduction_for_same_profit :
  daily_profit huimin_scenario 10 = daily_profit huimin_scenario 0 := by sorry

theorem no_solution_for_460_profit :
  ∀ x : ℝ, daily_profit huimin_scenario x ≠ 460 := by sorry

end NUMINAMATH_CALUDE_price_reduction_for_same_profit_no_solution_for_460_profit_l3019_301984


namespace NUMINAMATH_CALUDE_rice_grains_difference_l3019_301916

def grains_on_square (k : ℕ) : ℕ := 3^k

def sum_first_n_squares (n : ℕ) : ℕ :=
  (List.range n).map grains_on_square |> List.sum

theorem rice_grains_difference : 
  grains_on_square 12 - sum_first_n_squares 9 = 501693 := by
  sorry

end NUMINAMATH_CALUDE_rice_grains_difference_l3019_301916


namespace NUMINAMATH_CALUDE_min_value_of_fraction_l3019_301966

theorem min_value_of_fraction (x y : ℝ) (h : x^2 + y^2 = 4) :
  ∃ m : ℝ, m = 1 - Real.sqrt 2 ∧ ∀ z : ℝ, z = x*y/(x+y-2) → m ≤ z :=
sorry

end NUMINAMATH_CALUDE_min_value_of_fraction_l3019_301966


namespace NUMINAMATH_CALUDE_negation_of_union_membership_l3019_301915

theorem negation_of_union_membership {α : Type*} (A B : Set α) (x : α) :
  ¬(x ∈ A ∪ B) ↔ x ∉ A ∧ x ∉ B :=
by sorry

end NUMINAMATH_CALUDE_negation_of_union_membership_l3019_301915


namespace NUMINAMATH_CALUDE_path_area_and_cost_l3019_301960

/-- Calculates the area of a path surrounding a rectangular field -/
def path_area (field_length field_width path_width : ℝ) : ℝ :=
  (field_length + 2 * path_width) * (field_width + 2 * path_width) - field_length * field_width

/-- Calculates the cost of constructing a path given its area and cost per square meter -/
def construction_cost (path_area cost_per_sqm : ℝ) : ℝ :=
  path_area * cost_per_sqm

theorem path_area_and_cost (field_length field_width path_width cost_per_sqm : ℝ)
  (h1 : field_length = 65)
  (h2 : field_width = 55)
  (h3 : path_width = 2.5)
  (h4 : cost_per_sqm = 2) :
  path_area field_length field_width path_width = 625 ∧
  construction_cost (path_area field_length field_width path_width) cost_per_sqm = 1250 := by
  sorry

end NUMINAMATH_CALUDE_path_area_and_cost_l3019_301960


namespace NUMINAMATH_CALUDE_quadratic_sum_l3019_301936

/-- A quadratic function g(x) = px^2 + qx + r passing through (0, 3) and (2, 3) -/
def QuadraticFunction (p q r : ℝ) : ℝ → ℝ := λ x ↦ p * x^2 + q * x + r

theorem quadratic_sum (p q r : ℝ) :
  QuadraticFunction p q r 0 = 3 ∧ QuadraticFunction p q r 2 = 3 →
  p + 2*q + r = 3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_sum_l3019_301936


namespace NUMINAMATH_CALUDE_fixed_cost_calculation_l3019_301933

/-- Represents the fixed cost to run the molding machine per week -/
def fixed_cost : ℝ := 7640

/-- Represents the cost to mold each handle -/
def variable_cost : ℝ := 0.60

/-- Represents the selling price per handle -/
def selling_price : ℝ := 4.60

/-- Represents the break-even point in number of handles per week -/
def break_even_point : ℝ := 1910

/-- Proves that the fixed cost is correct given the other parameters -/
theorem fixed_cost_calculation :
  fixed_cost = break_even_point * (selling_price - variable_cost) :=
by sorry

end NUMINAMATH_CALUDE_fixed_cost_calculation_l3019_301933


namespace NUMINAMATH_CALUDE_max_intersections_math_city_l3019_301961

/-- Represents the number of streets in Math City -/
def total_streets : ℕ := 10

/-- Represents the number of parallel streets -/
def parallel_streets : ℕ := 2

/-- Represents the number of non-parallel streets -/
def non_parallel_streets : ℕ := total_streets - parallel_streets

/-- 
  Theorem: Maximum number of intersections in Math City
  Given:
  - There are 10 streets in total
  - Exactly 2 streets are parallel to each other
  - No other pair of streets is parallel
  - No three streets meet at a single point
  Prove: The maximum number of intersections is 44
-/
theorem max_intersections_math_city : 
  (non_parallel_streets.choose 2) + (parallel_streets * non_parallel_streets) = 44 := by
  sorry

end NUMINAMATH_CALUDE_max_intersections_math_city_l3019_301961


namespace NUMINAMATH_CALUDE_isosceles_triangles_count_l3019_301985

/-- A triangle represented by its three vertices in 2D space -/
structure Triangle where
  v1 : (Int × Int)
  v2 : (Int × Int)
  v3 : (Int × Int)

/-- Check if a triangle is isosceles -/
def isIsosceles (t : Triangle) : Bool :=
  let d12 := (t.v1.1 - t.v2.1)^2 + (t.v1.2 - t.v2.2)^2
  let d23 := (t.v2.1 - t.v3.1)^2 + (t.v2.2 - t.v3.2)^2
  let d31 := (t.v3.1 - t.v1.1)^2 + (t.v3.2 - t.v1.2)^2
  d12 = d23 || d23 = d31 || d31 = d12

/-- The list of triangles from the problem -/
def triangles : List Triangle := [
  { v1 := (0, 8), v2 := (2, 8), v3 := (1, 6) },
  { v1 := (3, 5), v2 := (3, 8), v3 := (6, 5) },
  { v1 := (0, 2), v2 := (4, 3), v3 := (8, 2) },
  { v1 := (7, 5), v2 := (6, 8), v3 := (10, 5) },
  { v1 := (7, 2), v2 := (8, 4), v3 := (10, 1) },
  { v1 := (3, 1), v2 := (5, 1), v3 := (4, 3) }
]

theorem isosceles_triangles_count : 
  (triangles.filter isIsosceles).length = 5 := by sorry

end NUMINAMATH_CALUDE_isosceles_triangles_count_l3019_301985


namespace NUMINAMATH_CALUDE_hyperbola_min_eccentricity_asymptote_l3019_301957

/-- The asymptotic equation of a hyperbola with minimum eccentricity -/
theorem hyperbola_min_eccentricity_asymptote (m : ℝ) (h : m > 0) :
  let e := Real.sqrt (m + 4 / m + 1)
  let hyperbola := fun (x y : ℝ) => x^2 / m - y^2 / (m^2 + 4) = 1
  let asymptote := fun (x : ℝ) => (2 * x, -2 * x)
  (∀ m' > 0, e ≤ Real.sqrt (m' + 4 / m' + 1)) →
  (∃ t : ℝ, hyperbola (asymptote t).1 (asymptote t).2) :=
by sorry


end NUMINAMATH_CALUDE_hyperbola_min_eccentricity_asymptote_l3019_301957


namespace NUMINAMATH_CALUDE_sum_of_squares_175_l3019_301971

theorem sum_of_squares_175 (a b c d : ℕ+) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d → 
  a^2 + b^2 + c^2 + d^2 = 175 →
  (a : ℕ) + b + c + d = 23 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_175_l3019_301971


namespace NUMINAMATH_CALUDE_quartic_roots_max_value_l3019_301983

theorem quartic_roots_max_value (a b d : ℝ) (x₁ x₂ x₃ x₄ : ℝ) :
  (1/2 ≤ x₁ ∧ x₁ ≤ 2) →
  (1/2 ≤ x₂ ∧ x₂ ≤ 2) →
  (1/2 ≤ x₃ ∧ x₃ ≤ 2) →
  (1/2 ≤ x₄ ∧ x₄ ≤ 2) →
  x₁^4 - a*x₁^3 + b*x₁^2 - a*x₁ + d = 0 →
  x₂^4 - a*x₂^3 + b*x₂^2 - a*x₂ + d = 0 →
  x₃^4 - a*x₃^3 + b*x₃^2 - a*x₃ + d = 0 →
  x₄^4 - a*x₄^3 + b*x₄^2 - a*x₄ + d = 0 →
  (x₁ + x₂) * (x₁ + x₃) * x₄ / ((x₄ + x₂) * (x₄ + x₃) * x₁) ≤ 5/4 :=
by sorry

end NUMINAMATH_CALUDE_quartic_roots_max_value_l3019_301983


namespace NUMINAMATH_CALUDE_geometric_series_equality_l3019_301982

theorem geometric_series_equality (n : ℕ) : n ≥ 1 → (
  let C : ℕ → ℚ := λ k => 1320 * (1 - 1 / 3^k)
  let D : ℕ → ℚ := λ k => 1008 * (1 - 1 / (-3)^k)
  (∃ k ≥ 1, C k = D k) ∧ (∀ m ≥ 1, m < n → C m ≠ D m) → n = 2
) := by sorry

end NUMINAMATH_CALUDE_geometric_series_equality_l3019_301982


namespace NUMINAMATH_CALUDE_base_5_reversed_in_base_7_l3019_301996

/-- Converts a base 5 number to base 10 -/
def toBase10FromBase5 (a b c : Nat) : Nat :=
  25 * a + 5 * b + c

/-- Converts a base 7 number to base 10 -/
def toBase10FromBase7 (a b c : Nat) : Nat :=
  49 * c + 7 * b + a

/-- Checks if a number is a valid digit in base 5 -/
def isValidBase5Digit (n : Nat) : Prop :=
  n ≤ 4

theorem base_5_reversed_in_base_7 :
  ∃! (a₁ b₁ c₁ a₂ b₂ c₂ : Nat),
    isValidBase5Digit a₁ ∧ isValidBase5Digit b₁ ∧ isValidBase5Digit c₁ ∧
    isValidBase5Digit a₂ ∧ isValidBase5Digit b₂ ∧ isValidBase5Digit c₂ ∧
    toBase10FromBase5 a₁ b₁ c₁ = toBase10FromBase7 c₁ b₁ a₁ ∧
    toBase10FromBase5 a₂ b₂ c₂ = toBase10FromBase7 c₂ b₂ a₂ ∧
    a₁ ≠ 0 ∧ a₂ ≠ 0 ∧
    toBase10FromBase5 a₁ b₁ c₁ + toBase10FromBase5 a₂ b₂ c₂ = 153 :=
  sorry

end NUMINAMATH_CALUDE_base_5_reversed_in_base_7_l3019_301996


namespace NUMINAMATH_CALUDE_ribbon_leftover_l3019_301988

theorem ribbon_leftover (total : ℕ) (gifts : ℕ) (per_gift : ℕ) (h1 : total = 18) (h2 : gifts = 6) (h3 : per_gift = 2) :
  total - (gifts * per_gift) = 6 := by
  sorry

end NUMINAMATH_CALUDE_ribbon_leftover_l3019_301988


namespace NUMINAMATH_CALUDE_computer_ownership_increase_l3019_301994

/-- The percentage of families owning a personal computer in 1992 -/
def percentage_1992 : ℝ := 30

/-- The increase in the number of families owning a computer from 1992 to 1999 -/
def increase_1992_to_1999 : ℝ := 50

/-- The percentage of families owning at least one personal computer in 1999 -/
def percentage_1999 : ℝ := 45

theorem computer_ownership_increase :
  percentage_1999 = percentage_1992 * (1 + increase_1992_to_1999 / 100) := by
  sorry

end NUMINAMATH_CALUDE_computer_ownership_increase_l3019_301994


namespace NUMINAMATH_CALUDE_line_equation_l3019_301940

/-- A line in 2D space defined by its equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line -/
def Line.contains (l : Line) (p : Point) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if two lines are perpendicular -/
def Line.perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

/-- The problem statement -/
theorem line_equation : 
  ∃ (l : Line), 
    l.contains ⟨-1, 2⟩ ∧ 
    l.perpendicular ⟨2, -3, 0⟩ ∧
    l = ⟨3, 2, -1⟩ := by
  sorry

end NUMINAMATH_CALUDE_line_equation_l3019_301940


namespace NUMINAMATH_CALUDE_jodi_walking_schedule_l3019_301911

/-- Represents Jodi's walking schedule over 4 weeks -/
structure WalkingSchedule where
  week1_distance : ℝ
  week2_distance : ℝ
  week3_distance : ℝ
  week4_distance : ℝ
  days_per_week : ℕ
  total_distance : ℝ

/-- Theorem stating that given Jodi's walking schedule, she walked 2 miles per day in the second week -/
theorem jodi_walking_schedule (schedule : WalkingSchedule) 
  (h1 : schedule.week1_distance = 1)
  (h2 : schedule.week3_distance = 3)
  (h3 : schedule.week4_distance = 4)
  (h4 : schedule.days_per_week = 6)
  (h5 : schedule.total_distance = 60)
  : schedule.week2_distance = 2 := by
  sorry

end NUMINAMATH_CALUDE_jodi_walking_schedule_l3019_301911


namespace NUMINAMATH_CALUDE_product_sum_theorem_l3019_301978

theorem product_sum_theorem (a b c : ℕ+) : 
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  a * b * c = 5^4 →
  (a : ℕ) + (b : ℕ) + (c : ℕ) = 131 := by
  sorry

end NUMINAMATH_CALUDE_product_sum_theorem_l3019_301978


namespace NUMINAMATH_CALUDE_binomial_expansion_constant_term_l3019_301942

theorem binomial_expansion_constant_term (n : ℕ) : 
  (∃ k : ℚ, 2 * (n.choose 2) = (n.choose 1) + k ∧ (n.choose 3) = (n.choose 2) + k) →
  (∃ r : ℕ, r ≤ n ∧ 21 = 7 * r ∧ n.choose r = 35) := by
sorry

end NUMINAMATH_CALUDE_binomial_expansion_constant_term_l3019_301942


namespace NUMINAMATH_CALUDE_duty_shoes_price_l3019_301953

def full_price : ℝ → Prop :=
  λ price => 
    let discount1 := 0.2
    let discount2 := 0.25
    let price_after_discount1 := price * (1 - discount1)
    let price_after_discount2 := price_after_discount1 * (1 - discount2)
    price_after_discount2 = 51

theorem duty_shoes_price : ∃ (price : ℝ), full_price price ∧ price = 85 := by
  sorry

end NUMINAMATH_CALUDE_duty_shoes_price_l3019_301953


namespace NUMINAMATH_CALUDE_mixed_box_weight_l3019_301973

/-- The weight of a box with 100 aluminum balls -/
def weight_aluminum : ℝ := 510

/-- The weight of a box with 100 plastic balls -/
def weight_plastic : ℝ := 490

/-- The number of aluminum balls in the mixed box -/
def num_aluminum : ℕ := 20

/-- The number of plastic balls in the mixed box -/
def num_plastic : ℕ := 80

/-- The total number of balls in each box -/
def total_balls : ℕ := 100

theorem mixed_box_weight : 
  (num_aluminum : ℝ) / total_balls * weight_aluminum + 
  (num_plastic : ℝ) / total_balls * weight_plastic = 494 := by
  sorry

end NUMINAMATH_CALUDE_mixed_box_weight_l3019_301973


namespace NUMINAMATH_CALUDE_cards_13_and_38_lowest_probability_l3019_301900

/-- Represents the probability that a card is red side up after flips -/
def probability_red_up (k : ℕ) : ℚ :=
  if k ≤ 25 then
    (676 - 52 * k + 2 * k^2) / 676
  else
    (676 - 52 * (51 - k) + 2 * (51 - k)^2) / 676

/-- The total number of cards -/
def total_cards : ℕ := 50

/-- Theorem stating that cards 13 and 38 have the lowest probability of being red side up -/
theorem cards_13_and_38_lowest_probability :
  ∀ k : ℕ, 1 ≤ k ∧ k ≤ total_cards →
    probability_red_up 13 ≤ probability_red_up k ∧
    probability_red_up 38 ≤ probability_red_up k :=
sorry

end NUMINAMATH_CALUDE_cards_13_and_38_lowest_probability_l3019_301900
