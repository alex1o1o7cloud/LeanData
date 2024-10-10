import Mathlib

namespace total_points_earned_l2047_204770

def enemies_defeated : ℕ := 6
def points_per_enemy : ℕ := 9
def level_completion_points : ℕ := 8

theorem total_points_earned :
  enemies_defeated * points_per_enemy + level_completion_points = 62 := by
  sorry

end total_points_earned_l2047_204770


namespace intersection_of_A_and_B_l2047_204739

def A : Set ℕ := {1, 2}
def B : Set ℕ := {1, 2, 3}

theorem intersection_of_A_and_B : A ∩ B = {1, 2} := by sorry

end intersection_of_A_and_B_l2047_204739


namespace sum_of_twenty_and_ten_l2047_204732

theorem sum_of_twenty_and_ten : 20 + 10 = 30 := by sorry

end sum_of_twenty_and_ten_l2047_204732


namespace lcm_852_1491_l2047_204740

theorem lcm_852_1491 : Nat.lcm 852 1491 = 5961 := by
  sorry

end lcm_852_1491_l2047_204740


namespace thousandth_coprime_to_105_l2047_204734

/-- The sequence of positive integers coprime to 105, arranged in ascending order -/
def coprimeSeq : ℕ → ℕ := sorry

/-- The 1000th term of the sequence is 2186 -/
theorem thousandth_coprime_to_105 : coprimeSeq 1000 = 2186 := by sorry

end thousandth_coprime_to_105_l2047_204734


namespace steve_shared_oranges_l2047_204779

/-- The number of oranges Steve shared with Patrick -/
def oranges_shared (initial : ℕ) (remaining : ℕ) : ℕ :=
  initial - remaining

theorem steve_shared_oranges :
  oranges_shared 46 42 = 4 := by
  sorry

end steve_shared_oranges_l2047_204779


namespace product_plus_one_equals_square_l2047_204719

theorem product_plus_one_equals_square (n : ℕ) :
  n * (n + 1) * (n + 2) * (n + 3) + 1 = (n^2 + 3*n + 1)^2 := by
  sorry

end product_plus_one_equals_square_l2047_204719


namespace nested_subtraction_1999_always_true_l2047_204773

/-- The nested subtraction function with n levels of nesting -/
def nestedSubtraction (x : ℝ) : ℕ → ℝ
  | 0 => x - 1
  | n + 1 => x - nestedSubtraction x n

/-- Theorem stating that for 1999 levels of nesting, the equation is always true for any real x -/
theorem nested_subtraction_1999_always_true (x : ℝ) :
  nestedSubtraction x 1999 = 1 := by
  sorry

#check nested_subtraction_1999_always_true

end nested_subtraction_1999_always_true_l2047_204773


namespace point_coordinates_in_third_quadrant_l2047_204737

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The third quadrant of the 2D plane -/
def ThirdQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y < 0

/-- Distance from a point to the x-axis -/
def DistToXAxis (p : Point) : ℝ :=
  |p.y|

/-- Distance from a point to the y-axis -/
def DistToYAxis (p : Point) : ℝ :=
  |p.x|

theorem point_coordinates_in_third_quadrant :
  ∃ (p : Point), ThirdQuadrant p ∧ DistToXAxis p = 2 ∧ DistToYAxis p = 3 → p = Point.mk (-3) (-2) := by
  sorry

end point_coordinates_in_third_quadrant_l2047_204737


namespace farmer_ploughing_problem_l2047_204797

theorem farmer_ploughing_problem (planned_rate : ℕ) (actual_rate : ℕ) (extra_days : ℕ) (area_left : ℕ) (total_area : ℕ) :
  planned_rate = 120 →
  actual_rate = 85 →
  extra_days = 2 →
  area_left = 40 →
  total_area = 720 →
  ∃ (planned_days : ℕ), 
    planned_days * planned_rate = total_area ∧
    (planned_days + extra_days) * actual_rate + area_left = total_area ∧
    planned_days = 6 :=
by sorry

end farmer_ploughing_problem_l2047_204797


namespace last_three_digits_of_7_to_1992_l2047_204705

theorem last_three_digits_of_7_to_1992 : ∃ n : ℕ, 7^1992 ≡ 201 + 1000 * n [ZMOD 1000] := by
  sorry

end last_three_digits_of_7_to_1992_l2047_204705


namespace quadratic_m_gt_n_l2047_204778

/-- Represents a quadratic function y = ax^2 + bx + c -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  a_nonzero : a ≠ 0

/-- The y-value of a quadratic function at a given x -/
def eval (f : QuadraticFunction) (x : ℝ) : ℝ :=
  f.a * x^2 + f.b * x + f.c

theorem quadratic_m_gt_n 
  (f : QuadraticFunction)
  (h1 : eval f (-1) = 0)
  (h2 : eval f 0 = 2)
  (h3 : eval f 3 = 0)
  (m n : ℝ)
  (hm : eval f 1 = m)
  (hn : eval f 2 = n) :
  m > n := by
  sorry

end quadratic_m_gt_n_l2047_204778


namespace homework_pages_proof_l2047_204726

theorem homework_pages_proof (math_pages reading_pages total_pages : ℕ) : 
  math_pages = 8 → 
  math_pages = reading_pages + 3 → 
  total_pages = math_pages + reading_pages → 
  total_pages = 13 := by
sorry

end homework_pages_proof_l2047_204726


namespace triangle_formation_proof_l2047_204787

/-- Checks if three lengths can form a triangle -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

/-- Given sticks of lengths 4 and 10, proves which of the given lengths can form a triangle -/
theorem triangle_formation_proof :
  let a : ℝ := 4
  let b : ℝ := 10
  (¬ can_form_triangle a b 3) ∧
  (¬ can_form_triangle a b 5) ∧
  (can_form_triangle a b 8) ∧
  (¬ can_form_triangle a b 15) := by
  sorry

#check triangle_formation_proof

end triangle_formation_proof_l2047_204787


namespace square_sum_nonzero_iff_not_both_zero_l2047_204711

theorem square_sum_nonzero_iff_not_both_zero (a b : ℝ) :
  a^2 + b^2 ≠ 0 ↔ ¬(a = 0 ∧ b = 0) :=
sorry

end square_sum_nonzero_iff_not_both_zero_l2047_204711


namespace sqrt_180_equals_6_sqrt_5_l2047_204767

theorem sqrt_180_equals_6_sqrt_5 : Real.sqrt 180 = 6 * Real.sqrt 5 := by
  sorry

end sqrt_180_equals_6_sqrt_5_l2047_204767


namespace triangle_area_from_perimeter_and_inradius_l2047_204748

theorem triangle_area_from_perimeter_and_inradius 
  (perimeter : ℝ) (inradius : ℝ) (area : ℝ) : 
  perimeter = 36 → inradius = 2.5 → area = 45 → 
  area = inradius * (perimeter / 2) :=
by
  sorry

end triangle_area_from_perimeter_and_inradius_l2047_204748


namespace tyler_purchase_theorem_l2047_204706

def remaining_money (initial_amount scissors_cost eraser_cost scissors_quantity eraser_quantity : ℕ) : ℕ :=
  initial_amount - (scissors_cost * scissors_quantity + eraser_cost * eraser_quantity)

theorem tyler_purchase_theorem (initial_amount scissors_cost eraser_cost scissors_quantity eraser_quantity : ℕ) :
  initial_amount = 100 ∧ 
  scissors_cost = 5 ∧ 
  eraser_cost = 4 ∧ 
  scissors_quantity = 8 ∧ 
  eraser_quantity = 10 → 
  remaining_money initial_amount scissors_cost eraser_cost scissors_quantity eraser_quantity = 20 := by
  sorry

end tyler_purchase_theorem_l2047_204706


namespace line_equation_equivalence_l2047_204750

/-- Given a line in the form (3, -4) · ((x, y) - (2, 8)) = 0,
    prove that it's equivalent to y = (3/4)x + 6.5 -/
theorem line_equation_equivalence (x y : ℝ) :
  (3 * (x - 2) + (-4) * (y - 8) = 0) ↔ (y = (3/4) * x + 6.5) := by
  sorry

end line_equation_equivalence_l2047_204750


namespace sum_of_possible_distances_l2047_204774

theorem sum_of_possible_distances (a b c d : ℝ) 
  (hab : |a - b| = 2)
  (hbc : |b - c| = 3)
  (hcd : |c - d| = 4) :
  ∃ S : Finset ℝ, (∀ x ∈ S, ∃ a' b' c' d' : ℝ, 
    |a' - b'| = 2 ∧ |b' - c'| = 3 ∧ |c' - d'| = 4 ∧ |a' - d'| = x) ∧
  (∀ y : ℝ, (∃ a' b' c' d' : ℝ, 
    |a' - b'| = 2 ∧ |b' - c'| = 3 ∧ |c' - d'| = 4 ∧ |a' - d'| = y) → y ∈ S) ∧
  S.sum id = 18 :=
sorry

end sum_of_possible_distances_l2047_204774


namespace rational_inequality_solution_set_l2047_204799

theorem rational_inequality_solution_set :
  {x : ℝ | (x + 1) / (x + 2) < 0} = {x : ℝ | -2 < x ∧ x < -1} := by sorry

end rational_inequality_solution_set_l2047_204799


namespace cricketer_score_percentage_l2047_204795

/-- A cricketer's score breakdown and calculation of runs made by running between wickets --/
theorem cricketer_score_percentage (total_score : ℕ) (boundaries : ℕ) (sixes : ℕ)
  (singles : ℕ) (twos : ℕ) (threes : ℕ) :
  total_score = 138 →
  boundaries = 12 →
  sixes = 2 →
  singles = 25 →
  twos = 7 →
  threes = 3 →
  (((singles * 1 + twos * 2 + threes * 3) : ℚ) / total_score) * 100 = 48 / 138 * 100 := by
  sorry

#eval (48 : ℚ) / 138 * 100

end cricketer_score_percentage_l2047_204795


namespace cricket_team_age_difference_l2047_204743

theorem cricket_team_age_difference (team_size : ℕ) (captain_age : ℕ) (team_avg_age : ℚ) :
  team_size = 11 →
  captain_age = 27 →
  team_avg_age = 24 →
  ∃ (wicket_keeper_age : ℕ),
    (team_avg_age * team_size - captain_age - wicket_keeper_age) / (team_size - 2) = team_avg_age - 1 →
    wicket_keeper_age = captain_age + 3 :=
by sorry

end cricket_team_age_difference_l2047_204743


namespace equidistant_point_on_y_axis_l2047_204700

theorem equidistant_point_on_y_axis : 
  ∃ y : ℚ, y = 13/6 ∧ 
  (∀ (x : ℚ), x = 0 → 
    (x^2 + y^2) = ((x - 2)^2 + (y - 3)^2)) :=
by
  sorry

end equidistant_point_on_y_axis_l2047_204700


namespace complement_of_A_wrt_U_l2047_204713

-- Define the universal set U
def U : Set Nat := {1, 3, 5}

-- Define the set A
def A : Set Nat := {1, 5}

-- State the theorem
theorem complement_of_A_wrt_U :
  {x ∈ U | x ∉ A} = {3} := by sorry

end complement_of_A_wrt_U_l2047_204713


namespace average_price_reduction_l2047_204725

theorem average_price_reduction (initial_price final_price : ℝ) 
  (h1 : initial_price = 25)
  (h2 : final_price = 16)
  (h3 : final_price = initial_price * (1 - x)^2)
  (h4 : x ≥ 0 ∧ x ≤ 1) : 
  x = 0.2 :=
sorry

end average_price_reduction_l2047_204725


namespace loan_duration_l2047_204759

theorem loan_duration (principal_B principal_C interest_rate total_interest : ℚ) 
  (duration_C : ℕ) : 
  principal_B = 5000 →
  principal_C = 3000 →
  duration_C = 4 →
  interest_rate = 15 / 100 →
  total_interest = 3300 →
  principal_B * interest_rate * (duration_B : ℚ) + principal_C * interest_rate * (duration_C : ℚ) = total_interest →
  duration_B = 2 := by
  sorry

#check loan_duration

end loan_duration_l2047_204759


namespace square_sum_given_difference_and_product_l2047_204798

theorem square_sum_given_difference_and_product (a b : ℝ) 
  (h1 : a - b = 10) 
  (h2 : a * b = 55) : 
  a^2 + b^2 = 210 :=
by sorry

end square_sum_given_difference_and_product_l2047_204798


namespace pen_notebook_cost_l2047_204777

theorem pen_notebook_cost :
  ∀ (p n : ℕ), 
    p > n ∧ 
    p > 0 ∧ 
    n > 0 ∧ 
    17 * p + 5 * n = 200 →
    p + n = 16 := by
  sorry

end pen_notebook_cost_l2047_204777


namespace square_points_probability_l2047_204716

/-- The number of points around the square -/
def num_points : ℕ := 8

/-- The number of pairs of points that are one unit apart -/
def favorable_pairs : ℕ := 8

/-- The total number of ways to choose two points from the available points -/
def total_pairs : ℕ := num_points.choose 2

/-- The probability of choosing two points that are one unit apart -/
def probability : ℚ := favorable_pairs / total_pairs

theorem square_points_probability : probability = 2/7 := by sorry

end square_points_probability_l2047_204716


namespace average_age_of_ten_students_l2047_204791

theorem average_age_of_ten_students
  (total_students : Nat)
  (average_age_all : ℝ)
  (num_group1 : Nat)
  (average_age_group1 : ℝ)
  (age_last_student : ℝ)
  (h1 : total_students = 15)
  (h2 : average_age_all = 15)
  (h3 : num_group1 = 4)
  (h4 : average_age_group1 = 14)
  (h5 : age_last_student = 9)
  : (total_students * average_age_all - num_group1 * average_age_group1 - age_last_student) / (total_students - num_group1 - 1) = 16 := by
  sorry

#check average_age_of_ten_students

end average_age_of_ten_students_l2047_204791


namespace fraction_value_l2047_204780

theorem fraction_value : (3000 - 2883)^2 / 121 = 106.36 := by
  sorry

end fraction_value_l2047_204780


namespace sibling_age_difference_l2047_204764

/-- Given the ages of three siblings, prove the age difference between two of them. -/
theorem sibling_age_difference (juliet maggie ralph : ℕ) : 
  juliet > maggie ∧ 
  juliet = ralph - 2 ∧ 
  juliet = 10 ∧ 
  maggie + ralph = 19 → 
  juliet - maggie = 3 := by
sorry

end sibling_age_difference_l2047_204764


namespace toaster_sales_l2047_204710

/-- Represents the inverse proportionality between number of customers and toaster cost -/
def inverse_proportional (customers : ℕ) (cost : ℝ) (k : ℝ) : Prop :=
  (customers : ℝ) * cost = k

/-- Proves that if 12 customers buy a $500 toaster, then 8 customers will buy a $750 toaster,
    given the inverse proportionality relationship -/
theorem toaster_sales (k : ℝ) :
  inverse_proportional 12 500 k →
  inverse_proportional 8 750 k :=
by
  sorry

end toaster_sales_l2047_204710


namespace pages_left_to_read_total_annotated_pages_l2047_204714

-- Define the book and reading parameters
def total_pages : ℕ := 567
def pages_read_week1 : ℕ := 279
def pages_read_week2 : ℕ := 124
def pages_annotated_week1 : ℕ := 35
def pages_annotated_week2 : ℕ := 15

-- Theorem for pages left to read
theorem pages_left_to_read : 
  total_pages - (pages_read_week1 + pages_read_week2) = 164 := by sorry

-- Theorem for total annotated pages
theorem total_annotated_pages :
  pages_annotated_week1 + pages_annotated_week2 = 50 := by sorry

end pages_left_to_read_total_annotated_pages_l2047_204714


namespace min_area_is_three_l2047_204736

/-- Triangle ABC with A at origin, B at (30, 18), and C with integer coordinates -/
structure Triangle :=
  (c_x : ℤ)
  (c_y : ℤ)

/-- Area of triangle ABC given coordinates of C -/
def area (t : Triangle) : ℚ :=
  (1 / 2 : ℚ) * |30 * t.c_y - 18 * t.c_x|

/-- The minimum area of triangle ABC is 3 -/
theorem min_area_is_three :
  ∃ (t : Triangle), area t = 3 ∧ ∀ (t' : Triangle), area t' ≥ 3 :=
sorry

end min_area_is_three_l2047_204736


namespace probability_two_hearts_l2047_204746

theorem probability_two_hearts (total_cards : Nat) (heart_cards : Nat) (drawn_cards : Nat) :
  total_cards = 52 →
  heart_cards = 13 →
  drawn_cards = 2 →
  (Nat.choose heart_cards drawn_cards : ℚ) / (Nat.choose total_cards drawn_cards : ℚ) = 1 / 17 := by
  sorry

end probability_two_hearts_l2047_204746


namespace unique_root_quadratic_l2047_204723

theorem unique_root_quadratic (k : ℝ) :
  (∃! x : ℝ, k * x^2 - 3 * x + 2 = 0) → k = 0 ∨ k = 9/8 := by
  sorry

end unique_root_quadratic_l2047_204723


namespace stick_length_average_l2047_204762

theorem stick_length_average (total_sticks : ℕ) (all_avg : ℝ) (two_avg : ℝ) :
  total_sticks = 11 →
  all_avg = 145.7 →
  two_avg = 142.1 →
  let remaining_sticks := total_sticks - 2
  let total_length := all_avg * total_sticks
  let two_length := two_avg * 2
  let remaining_length := total_length - two_length
  remaining_length / remaining_sticks = 146.5 := by
sorry

end stick_length_average_l2047_204762


namespace final_antifreeze_ratio_l2047_204781

/-- Calculates the fraction of antifreeze in a tank after multiple replacements --/
def antifreezeRatio (tankCapacity : ℚ) (initialRatio : ℚ) (replacementAmount : ℚ) (replacements : ℕ) : ℚ :=
  let initialAntifreeze := tankCapacity * initialRatio
  let remainingRatio := (tankCapacity - replacementAmount) / tankCapacity
  initialAntifreeze * remainingRatio ^ replacements / tankCapacity

/-- Theorem stating the final antifreeze ratio after 4 replacements --/
theorem final_antifreeze_ratio :
  antifreezeRatio 20 (1/4) 4 4 = 1024/5000 := by
  sorry

#eval antifreezeRatio 20 (1/4) 4 4

end final_antifreeze_ratio_l2047_204781


namespace intersection_points_with_ellipse_l2047_204751

/-- The line equation mx - ny = 4 and circle x^2 + y^2 = 4 have no intersection points -/
def no_intersection (m n : ℝ) : Prop :=
  ∀ x y : ℝ, (m * x - n * y = 4) → (x^2 + y^2 ≠ 4)

/-- The ellipse equation x^2/9 + y^2/4 = 1 -/
def on_ellipse (x y : ℝ) : Prop :=
  x^2 / 9 + y^2 / 4 = 1

/-- A point (x, y) is on the line passing through (m, n) -/
def on_line_through_P (m n x y : ℝ) : Prop :=
  ∃ t : ℝ, x = m * t ∧ y = n * t

/-- The theorem statement -/
theorem intersection_points_with_ellipse (m n : ℝ) :
  no_intersection m n →
  (∃! (x1 y1 x2 y2 : ℝ), 
    x1 ≠ x2 ∧ 
    on_ellipse x1 y1 ∧ 
    on_ellipse x2 y2 ∧ 
    on_line_through_P m n x1 y1 ∧ 
    on_line_through_P m n x2 y2) :=
by sorry

end intersection_points_with_ellipse_l2047_204751


namespace min_value_implies_a_equals_9_l2047_204703

theorem min_value_implies_a_equals_9 (t a : ℝ) (h1 : 0 < t) (h2 : t < π / 2) (h3 : a > 0) :
  (∀ s, 0 < s ∧ s < π / 2 → (1 / Real.cos s + a / (1 - Real.cos s)) ≥ 16) ∧
  (∃ s, 0 < s ∧ s < π / 2 ∧ 1 / Real.cos s + a / (1 - Real.cos s) = 16) →
  a = 9 := by
sorry

end min_value_implies_a_equals_9_l2047_204703


namespace consecutive_roots_quadratic_l2047_204708

theorem consecutive_roots_quadratic (n : ℕ) (hn : n > 1) :
  let f : ℝ → ℝ := λ x => x^2 - (2*n - 1)*x + n*(n-1)
  (f (n - 1) = 0) ∧ (f n = 0) := by
  sorry

end consecutive_roots_quadratic_l2047_204708


namespace exhibition_survey_l2047_204758

/-- The percentage of visitors who liked the first part of the exhibition -/
def first_part_percentage : ℝ := 25

/-- The percentage of visitors who liked the second part of the exhibition -/
def second_part_percentage : ℝ := 40

theorem exhibition_survey (total_visitors : ℝ) (h_total_positive : total_visitors > 0) :
  let visitors_first_part := (first_part_percentage / 100) * total_visitors
  let visitors_second_part := (second_part_percentage / 100) * total_visitors
  (96 / 100 * visitors_first_part = 60 / 100 * visitors_second_part) ∧
  (59 / 100 * total_visitors = total_visitors - (visitors_first_part + visitors_second_part - 96 / 100 * visitors_first_part)) →
  first_part_percentage = 25 := by
sorry


end exhibition_survey_l2047_204758


namespace binary_representation_of_500_l2047_204702

/-- Converts a natural number to its binary representation as a list of bits -/
def to_binary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
  let rec aux (m : ℕ) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: aux (m / 2)
  aux n

/-- Converts a list of bits to its decimal representation -/
def from_binary (bits : List Bool) : ℕ :=
  bits.foldl (fun acc b => 2 * acc + if b then 1 else 0) 0

theorem binary_representation_of_500 :
  to_binary 500 = [true, false, false, true, true, true, true, true, true] :=
by sorry

end binary_representation_of_500_l2047_204702


namespace x_eq_y_sufficient_not_necessary_l2047_204760

theorem x_eq_y_sufficient_not_necessary :
  (∀ x y : ℝ, x = y → |x| = |y|) ∧
  (∃ x y : ℝ, |x| = |y| ∧ x ≠ y) :=
by sorry

end x_eq_y_sufficient_not_necessary_l2047_204760


namespace oddSumProbability_l2047_204752

/-- Represents an unfair die where even numbers are 5 times as likely as odd numbers -/
structure UnfairDie where
  /-- Probability of rolling an odd number -/
  oddProb : ℝ
  /-- Probability of rolling an even number -/
  evenProb : ℝ
  /-- Even probability is 5 times odd probability -/
  evenOddRatio : evenProb = 5 * oddProb
  /-- Total probability is 1 -/
  totalProb : oddProb + evenProb = 1

/-- The probability of rolling an odd sum with two rolls of the unfair die -/
def oddSumProb (d : UnfairDie) : ℝ :=
  2 * d.oddProb * d.evenProb

theorem oddSumProbability (d : UnfairDie) : oddSumProb d = 5 / 18 := by
  sorry


end oddSumProbability_l2047_204752


namespace factor_implies_k_value_l2047_204796

/-- Given a quadratic trinomial 2x^2 + 3x - k with a factor (2x - 5), k equals 20 -/
theorem factor_implies_k_value (k : ℝ) : 
  (∃ (q : ℝ → ℝ), ∀ x, 2*x^2 + 3*x - k = (2*x - 5) * q x) → 
  k = 20 := by
sorry

end factor_implies_k_value_l2047_204796


namespace sqrt_seven_expressions_l2047_204715

theorem sqrt_seven_expressions (a b : ℝ) 
  (ha : a = Real.sqrt 7 + 2) 
  (hb : b = Real.sqrt 7 - 2) : 
  a^2 * b + b^2 * a = 6 * Real.sqrt 7 ∧ 
  a^2 + a * b + b^2 = 25 := by
  sorry

end sqrt_seven_expressions_l2047_204715


namespace endpoint_coordinate_sum_l2047_204761

/-- Given a line segment with one endpoint at (10, 4) and midpoint at (7, -5),
    the sum of coordinates of the other endpoint is -10. -/
theorem endpoint_coordinate_sum : 
  ∀ (x y : ℝ), 
  (10 + x) / 2 = 7 → 
  (4 + y) / 2 = -5 → 
  x + y = -10 := by
sorry

end endpoint_coordinate_sum_l2047_204761


namespace tuesday_steps_l2047_204754

/-- The number of steps Toby walked on each day of the week --/
structure WeekSteps where
  sunday : ℕ
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ
  thursday : ℕ
  friday : ℕ
  saturday : ℕ

/-- Theorem stating that given the conditions, Toby walked 8300 steps on Tuesday --/
theorem tuesday_steps (w : WeekSteps) : 
  w.sunday = 9400 ∧ 
  w.monday = 9100 ∧ 
  (w.wednesday = 9200 ∨ w.thursday = 9200) ∧
  (w.wednesday = 8900 ∨ w.thursday = 8900) ∧
  w.friday + w.saturday = 18100 ∧
  w.sunday + w.monday + w.tuesday + w.wednesday + w.thursday + w.friday + w.saturday = 63000 
  → w.tuesday = 8300 := by
  sorry

#check tuesday_steps

end tuesday_steps_l2047_204754


namespace intersection_of_A_and_B_l2047_204741

def A : Set ℝ := {-1, 0, 1}
def B : Set ℝ := {x | 0 < x ∧ x < 2}

theorem intersection_of_A_and_B : A ∩ B = {1} := by sorry

end intersection_of_A_and_B_l2047_204741


namespace sum_of_four_consecutive_integers_divisible_by_two_l2047_204765

theorem sum_of_four_consecutive_integers_divisible_by_two (n : ℤ) : 
  2 ∣ ((n - 1) + n + (n + 1) + (n + 2)) := by
  sorry

end sum_of_four_consecutive_integers_divisible_by_two_l2047_204765


namespace shortcut_rectangle_ratio_l2047_204749

/-- A rectangle where the diagonal shortcut saves 1/3 of the longer side -/
structure ShortcutRectangle where
  x : ℝ  -- shorter side
  y : ℝ  -- longer side
  x_pos : 0 < x
  y_pos : 0 < y
  x_lt_y : x < y
  shortcut_saves : x + y - Real.sqrt (x^2 + y^2) = (1/3) * y

theorem shortcut_rectangle_ratio (r : ShortcutRectangle) : r.x / r.y = 5/12 := by
  sorry

end shortcut_rectangle_ratio_l2047_204749


namespace egyptian_triangle_bisecting_line_exists_l2047_204728

/-- Represents a right triangle with sides 3, 4, and 5 -/
structure EgyptianTriangle where
  a : Real
  b : Real
  c : Real
  ha : a = 3
  hb : b = 4
  hc : c = 5
  right_angle : a^2 + b^2 = c^2

/-- Represents a line that intersects the triangle -/
structure BisectingLine where
  x : Real -- intersection point on shorter leg
  y : Real -- intersection point on hypotenuse
  hx : x = 3 - Real.sqrt 6 / 2
  hy : y = 3 + Real.sqrt 6 / 2

/-- Theorem stating the existence of a bisecting line for an Egyptian triangle -/
theorem egyptian_triangle_bisecting_line_exists (t : EgyptianTriangle) :
  ∃ (l : BisectingLine),
    (l.x + l.y = t.a + t.b) ∧                          -- Bisects perimeter
    (l.x * l.y * (t.b / t.c) / 2 = t.a * t.b / 4) :=   -- Bisects area
by sorry

end egyptian_triangle_bisecting_line_exists_l2047_204728


namespace quadratic_completion_l2047_204788

theorem quadratic_completion (y : ℝ) : ∃ (k : ℤ) (a : ℝ), y^2 + 10*y + 47 = (y + a)^2 + k ∧ k = 22 := by
  sorry

end quadratic_completion_l2047_204788


namespace attendance_difference_l2047_204771

def football_game_attendance (saturday monday wednesday friday thursday expected_total : ℕ) : Prop :=
  let total := saturday + monday + wednesday + friday + thursday
  saturday = 80 ∧
  monday = saturday - 20 ∧
  wednesday = monday + 50 ∧
  friday = saturday + monday ∧
  thursday = 45 ∧
  expected_total = 350 ∧
  total - expected_total = 85

theorem attendance_difference :
  ∃ (saturday monday wednesday friday thursday expected_total : ℕ),
    football_game_attendance saturday monday wednesday friday thursday expected_total :=
by
  sorry

end attendance_difference_l2047_204771


namespace fourth_grade_students_l2047_204793

/-- The total number of students at the end of the year in fourth grade -/
def total_students (initial : ℝ) (added : ℝ) (new : ℝ) : ℝ :=
  initial + added + new

/-- Theorem: The total number of students at the end of the year is 56.0 -/
theorem fourth_grade_students :
  total_students 10.0 4.0 42.0 = 56.0 := by
  sorry

end fourth_grade_students_l2047_204793


namespace smallest_prime_eight_less_than_square_l2047_204720

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

theorem smallest_prime_eight_less_than_square : 
  (∀ n : ℕ, n > 0 ∧ is_prime n ∧ (∃ m : ℕ, n = m * m - 8) → n ≥ 17) ∧ 
  (17 > 0 ∧ is_prime 17 ∧ ∃ m : ℕ, 17 = m * m - 8) :=
sorry

end smallest_prime_eight_less_than_square_l2047_204720


namespace line_circle_intersection_k_range_l2047_204731

/-- Given a line y = kx + 3 intersecting a circle (x - 2)² + (y - 3)² = 4 at points M and N,
    if |MN| ≥ 2√3, then -√3/3 ≤ k ≤ √3/3 -/
theorem line_circle_intersection_k_range (k : ℝ) (M N : ℝ × ℝ) :
  (∀ x y, y = k * x + 3 → (x - 2)^2 + (y - 3)^2 = 4 → (x, y) = M ∨ (x, y) = N) →
  (M.1 - N.1)^2 + (M.2 - N.2)^2 ≥ 12 →
  -Real.sqrt 3 / 3 ≤ k ∧ k ≤ Real.sqrt 3 / 3 :=
by sorry

end line_circle_intersection_k_range_l2047_204731


namespace total_marbles_count_l2047_204729

/-- The total number of marbles Mary and Joan have -/
def total_marbles : ℕ :=
  let mary_yellow := 9
  let mary_blue := 7
  let mary_green := 6
  let joan_yellow := 3
  let joan_blue := 5
  let joan_green := 4
  mary_yellow + mary_blue + mary_green + joan_yellow + joan_blue + joan_green

theorem total_marbles_count : total_marbles = 34 := by
  sorry

end total_marbles_count_l2047_204729


namespace least_positive_integer_multiple_59_l2047_204721

theorem least_positive_integer_multiple_59 : 
  ∃ (x : ℕ+), (∀ (y : ℕ+), y < x → ¬(59 ∣ (2 * y + 51)^2)) ∧ (59 ∣ (2 * x + 51)^2) ∧ x = 4 := by
  sorry

end least_positive_integer_multiple_59_l2047_204721


namespace intersection_M_N_l2047_204772

def M : Set ℝ := {x | |x| ≤ 2}
def N : Set ℝ := {-1, 0, 2, 3}

theorem intersection_M_N : M ∩ N = {-1, 0, 2} := by
  sorry

end intersection_M_N_l2047_204772


namespace dogsled_race_speed_difference_l2047_204701

theorem dogsled_race_speed_difference 
  (course_length : ℝ) 
  (team_w_speed : ℝ) 
  (time_difference : ℝ) :
  course_length = 300 →
  team_w_speed = 20 →
  time_difference = 3 →
  let team_w_time := course_length / team_w_speed
  let team_a_time := team_w_time - time_difference
  let team_a_speed := course_length / team_a_time
  team_a_speed - team_w_speed = 5 := by
  sorry

end dogsled_race_speed_difference_l2047_204701


namespace prob_HTTH_is_one_sixteenth_l2047_204785

/-- The probability of obtaining the sequence HTTH in four consecutive fair coin tosses -/
def prob_HTTH : ℚ := 1 / 16

/-- A fair coin toss is modeled as a probability space with two outcomes -/
structure FairCoin where
  sample_space : Type
  prob : sample_space → ℚ
  head : sample_space
  tail : sample_space
  fair_head : prob head = 1 / 2
  fair_tail : prob tail = 1 / 2
  total_prob : prob head + prob tail = 1

/-- Four consecutive fair coin tosses -/
def four_tosses (c : FairCoin) : Type := c.sample_space × c.sample_space × c.sample_space × c.sample_space

/-- The probability of a specific sequence of four tosses -/
def sequence_prob (c : FairCoin) (s : four_tosses c) : ℚ :=
  c.prob s.1 * c.prob s.2.1 * c.prob s.2.2.1 * c.prob s.2.2.2

/-- Theorem: The probability of obtaining HTTH in four consecutive fair coin tosses is 1/16 -/
theorem prob_HTTH_is_one_sixteenth (c : FairCoin) :
  sequence_prob c (c.head, c.tail, c.tail, c.head) = prob_HTTH := by
  sorry

end prob_HTTH_is_one_sixteenth_l2047_204785


namespace garbage_collection_theorem_l2047_204790

/-- The amount of garbage collected by four people given specific relationships between their collections. -/
def total_garbage_collected (daliah_amount : ℝ) : ℝ := 
  let dewei_amount := daliah_amount - 2
  let zane_amount := 4 * dewei_amount
  let bela_amount := zane_amount + 3.75
  daliah_amount + dewei_amount + zane_amount + bela_amount

/-- Theorem stating that the total amount of garbage collected is 160.75 pounds when Daliah collects 17.5 pounds. -/
theorem garbage_collection_theorem : 
  total_garbage_collected 17.5 = 160.75 := by
  sorry

#eval total_garbage_collected 17.5

end garbage_collection_theorem_l2047_204790


namespace abs_x_minus_one_l2047_204775

theorem abs_x_minus_one (x : ℚ) (h : |1 - x| = 1 + |x|) : |x - 1| = 1 - x := by
  sorry

end abs_x_minus_one_l2047_204775


namespace fourth_term_of_geometric_sequence_l2047_204724

/-- A positive geometric sequence -/
def PositiveGeometricSequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a n > 0) ∧ (∃ r > 0, ∀ n, a (n + 1) = r * a n)

/-- The fourth term of a positive geometric sequence is 2 if the product of the third and fifth terms is 4 -/
theorem fourth_term_of_geometric_sequence
  (a : ℕ → ℝ)
  (h_geo : PositiveGeometricSequence a)
  (h_prod : a 3 * a 5 = 4) :
  a 4 = 2 := by
sorry

end fourth_term_of_geometric_sequence_l2047_204724


namespace correct_sampling_methods_l2047_204735

/-- Represents different sampling methods -/
inductive SamplingMethod
  | Random
  | Systematic
  | Stratified

/-- Represents income levels -/
inductive IncomeLevel
  | High
  | Middle
  | Low

/-- Represents a community with households of different income levels -/
structure Community where
  totalHouseholds : Nat
  highIncomeHouseholds : Nat
  middleIncomeHouseholds : Nat
  lowIncomeHouseholds : Nat

/-- Represents a group of senior soccer players -/
structure SoccerTeam where
  totalPlayers : Nat

/-- Determines the best sampling method for a given community and sample size -/
def bestSamplingMethodForCommunity (c : Community) (sampleSize : Nat) : SamplingMethod :=
  sorry

/-- Determines the best sampling method for a given soccer team and sample size -/
def bestSamplingMethodForSoccerTeam (t : SoccerTeam) (sampleSize : Nat) : SamplingMethod :=
  sorry

theorem correct_sampling_methods 
  (community : Community)
  (soccerTeam : SoccerTeam)
  (communitySampleSize : Nat)
  (soccerSampleSize : Nat)
  (h1 : community.totalHouseholds = 500)
  (h2 : community.highIncomeHouseholds = 125)
  (h3 : community.middleIncomeHouseholds = 280)
  (h4 : community.lowIncomeHouseholds = 95)
  (h5 : communitySampleSize = 100)
  (h6 : soccerTeam.totalPlayers = 12)
  (h7 : soccerSampleSize = 3) :
  bestSamplingMethodForCommunity community communitySampleSize = SamplingMethod.Stratified ∧
  bestSamplingMethodForSoccerTeam soccerTeam soccerSampleSize = SamplingMethod.Random :=
sorry

end correct_sampling_methods_l2047_204735


namespace piglet_gave_two_balloons_l2047_204722

/-- The number of balloons Piglet eventually gave to Eeyore -/
def piglet_balloons : ℕ := 2

/-- The number of balloons Winnie-the-Pooh prepared -/
def pooh_balloons (n : ℕ) : ℕ := 2 * n

/-- The number of balloons Owl prepared -/
def owl_balloons (n : ℕ) : ℕ := 4 * n

/-- The total number of balloons Eeyore received -/
def total_balloons : ℕ := 44

/-- Theorem stating that Piglet gave 2 balloons to Eeyore -/
theorem piglet_gave_two_balloons :
  ∃ (n : ℕ), 
    piglet_balloons + pooh_balloons n + owl_balloons n = total_balloons ∧
    n > piglet_balloons ∧
    piglet_balloons = 2 := by
  sorry


end piglet_gave_two_balloons_l2047_204722


namespace square_difference_plus_fifty_l2047_204707

theorem square_difference_plus_fifty : (312^2 - 288^2) / 24 + 50 = 650 := by
  sorry

end square_difference_plus_fifty_l2047_204707


namespace alligator_count_theorem_l2047_204757

/-- The total number of alligators seen by Samara and her friends -/
def total_alligators (samara_count : ℕ) (friends_count : ℕ) (friends_average : ℕ) : ℕ :=
  samara_count + friends_count * friends_average

/-- Theorem stating the total number of alligators seen by Samara and her friends -/
theorem alligator_count_theorem : 
  total_alligators 35 6 15 = 125 := by
  sorry

#eval total_alligators 35 6 15

end alligator_count_theorem_l2047_204757


namespace fraction_floor_value_l2047_204789

theorem fraction_floor_value : ⌊(1500^2 : ℝ) / ((500^2 : ℝ) - (496^2 : ℝ))⌋ = 564 := by
  sorry

end fraction_floor_value_l2047_204789


namespace two_days_satisfy_l2047_204776

/-- Represents the days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- The number of days in the month -/
def monthLength : Nat := 30

/-- Function to check if a given day results in equal Tuesdays and Thursdays -/
def equalTuesdaysThursdays (startDay : DayOfWeek) : Bool :=
  sorry -- Implementation details omitted

/-- Count the number of days that satisfy the condition -/
def countSatisfyingDays : Nat :=
  sorry -- Implementation details omitted

/-- Theorem stating that exactly two days satisfy the condition -/
theorem two_days_satisfy :
  countSatisfyingDays = 2 :=
sorry

end two_days_satisfy_l2047_204776


namespace cost_of_apples_l2047_204756

/-- The cost of apples given the total cost of groceries and the costs of other items -/
theorem cost_of_apples (total cost_bananas cost_bread cost_milk : ℕ) 
  (h1 : total = 42)
  (h2 : cost_bananas = 12)
  (h3 : cost_bread = 9)
  (h4 : cost_milk = 7) :
  total - (cost_bananas + cost_bread + cost_milk) = 14 :=
by sorry

end cost_of_apples_l2047_204756


namespace blood_expiration_time_l2047_204794

/-- Represents a time of day in hours and minutes -/
structure TimeOfDay where
  hours : Nat
  minutes : Nat
  hle24 : hours < 24
  mle59 : minutes < 60

/-- Represents a date with a day and a time -/
structure Date where
  day : Nat
  time : TimeOfDay

def factorial (n : Nat) : Nat :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

def secondsToTime (seconds : Nat) : TimeOfDay :=
  let totalMinutes := seconds / 60
  let hours := totalMinutes / 60
  let minutes := totalMinutes % 60
  ⟨hours % 24, minutes, by sorry, by sorry⟩

def addTimeToDate (d : Date) (seconds : Nat) : Date :=
  let newTime := secondsToTime ((d.time.hours * 60 + d.time.minutes) * 60 + seconds)
  ⟨d.day + (if newTime.hours < d.time.hours then 1 else 0), newTime⟩

theorem blood_expiration_time :
  let donationDate : Date := ⟨1, ⟨12, 0, by sorry, by sorry⟩⟩
  let expirationSeconds := factorial 8
  let expirationDate := addTimeToDate donationDate expirationSeconds
  expirationDate = ⟨1, ⟨23, 13, by sorry, by sorry⟩⟩ :=
by sorry

end blood_expiration_time_l2047_204794


namespace special_square_side_length_l2047_204745

/-- Square with special points -/
structure SpecialSquare where
  /-- Side length of the square -/
  side : ℝ
  /-- Point M on side CD -/
  m : ℝ × ℝ
  /-- Point E where AM intersects the circumscribed circle -/
  e : ℝ × ℝ

/-- The theorem statement -/
theorem special_square_side_length (s : SpecialSquare) :
  /- Point M is on side CD -/
  s.m.1 = s.side ∧ 0 ≤ s.m.2 ∧ s.m.2 ≤ s.side ∧
  /- CM:MD = 1:3 -/
  s.m.2 = s.side / 4 ∧
  /- E is on the circumscribed circle -/
  (s.e.1 - s.side / 2)^2 + (s.e.2 - s.side / 2)^2 = 2 * (s.side / 2)^2 ∧
  /- Area of triangle ACE is 14 -/
  1/2 * s.e.1 * s.e.2 = 14 →
  /- The side length of the square is 10 -/
  s.side = 10 := by
  sorry

end special_square_side_length_l2047_204745


namespace yanni_toy_cost_l2047_204753

/-- The cost of the toy Yanni bought -/
def toy_cost (initial_money mother_gift found_money money_left : ℚ) : ℚ :=
  initial_money + mother_gift + found_money - money_left

/-- Theorem stating the cost of the toy Yanni bought -/
theorem yanni_toy_cost :
  toy_cost 0.85 0.40 0.50 0.15 = 1.60 := by
  sorry

end yanni_toy_cost_l2047_204753


namespace cos_150_plus_cos_neg_150_l2047_204763

theorem cos_150_plus_cos_neg_150 : Real.cos (150 * π / 180) + Real.cos (-150 * π / 180) = -Real.sqrt 3 := by
  sorry

end cos_150_plus_cos_neg_150_l2047_204763


namespace no_integer_solutions_l2047_204718

theorem no_integer_solutions (m s : ℤ) (h : m * s = 2000^2001) :
  ¬∃ (x y : ℤ), m * x^2 - s * y^2 = 3 := by
  sorry

end no_integer_solutions_l2047_204718


namespace hyperbola_eccentricity_range_l2047_204727

/-- Given a circle centered at (0,b) with radius a, and a hyperbola C: y²/a² - x²/b² = 1 (a > 0, b > 0),
    if the circle and the asymptotes of the hyperbola C are disjoint, 
    then the eccentricity e of C satisfies 1 < e < (√5 + 1)/2. -/
theorem hyperbola_eccentricity_range (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let circle := {(x, y) : ℝ × ℝ | x^2 + (y - b)^2 = a^2}
  let hyperbola := {(x, y) : ℝ × ℝ | y^2 / a^2 - x^2 / b^2 = 1}
  let asymptotes := {(x, y) : ℝ × ℝ | b * y = a * x ∨ b * y = -a * x}
  let e := Real.sqrt (1 + b^2 / a^2)  -- eccentricity of the hyperbola
  (circle ∩ asymptotes = ∅) → 1 < e ∧ e < (Real.sqrt 5 + 1) / 2 :=
by sorry

end hyperbola_eccentricity_range_l2047_204727


namespace subset_implies_m_range_l2047_204782

theorem subset_implies_m_range (m : ℝ) : 
  let A : Set ℝ := {x | 4 ≤ x ∧ x ≤ 8}
  let B : Set ℝ := {x | m + 1 < x ∧ x < 2*m - 2}
  B ⊆ A → m ≤ 5 := by
  sorry

end subset_implies_m_range_l2047_204782


namespace min_value_sqrt_sum_min_value_sqrt_sum_equals_l2047_204747

theorem min_value_sqrt_sum (x y z : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) (h4 : x + y + z = 4) :
  ∀ a b c : ℝ, a ≥ 0 → b ≥ 0 → c ≥ 0 → a + b + c = 4 →
    Real.sqrt (2 * x + 1) + Real.sqrt (3 * y + 1) + Real.sqrt (4 * z + 1) ≤
    Real.sqrt (2 * a + 1) + Real.sqrt (3 * b + 1) + Real.sqrt (4 * c + 1) :=
by
  sorry

theorem min_value_sqrt_sum_equals (x y z : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) (h4 : x + y + z = 4) :
  ∃ x y z : ℝ, x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧ x + y + z = 4 ∧
    Real.sqrt (2 * x + 1) + Real.sqrt (3 * y + 1) + Real.sqrt (4 * z + 1) =
    Real.sqrt (61 / 27) + Real.sqrt (183 / 36) + Real.sqrt (976 / 108) :=
by
  sorry

end min_value_sqrt_sum_min_value_sqrt_sum_equals_l2047_204747


namespace log_13_3x_bounds_l2047_204755

-- Define the logarithm function
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- State the theorem
theorem log_13_3x_bounds (x : ℝ) (h : log 7 (x + 6) = 2) : 
  1 < log 13 (3 * x) ∧ log 13 (3 * x) < 2 := by
  sorry

end log_13_3x_bounds_l2047_204755


namespace value_of_2a_plus_3b_l2047_204712

theorem value_of_2a_plus_3b (a b : ℚ) 
  (eq1 : 3 * a + 6 * b = 48) 
  (eq2 : 8 * a + 4 * b = 84) : 
  2 * a + 3 * b = 85 / 3 := by
  sorry

end value_of_2a_plus_3b_l2047_204712


namespace least_common_multiple_15_36_l2047_204769

theorem least_common_multiple_15_36 : Nat.lcm 15 36 = 180 := by
  sorry

end least_common_multiple_15_36_l2047_204769


namespace absolute_value_problem_l2047_204730

theorem absolute_value_problem (y q : ℝ) (h1 : |y - 3| = q) (h2 : y < 3) : 
  y - 2*q = 3 - 3*q := by
  sorry

end absolute_value_problem_l2047_204730


namespace mork_and_mindy_tax_rate_l2047_204717

/-- Calculates the combined tax rate for Mork and Mindy given their individual tax rates and income ratio. -/
theorem mork_and_mindy_tax_rate 
  (mork_rate : ℝ) 
  (mindy_rate : ℝ) 
  (income_ratio : ℝ) 
  (h1 : mork_rate = 0.1) 
  (h2 : mindy_rate = 0.2) 
  (h3 : income_ratio = 3) : 
  (mork_rate + mindy_rate * income_ratio) / (1 + income_ratio) = 0.175 := by
sorry

#eval (0.1 + 0.2 * 3) / (1 + 3)

end mork_and_mindy_tax_rate_l2047_204717


namespace difference_of_squares_l2047_204709

theorem difference_of_squares (a b : ℝ) (h1 : a + b = 5) (h2 : a - b = 3) :
  a^2 - b^2 = 15 := by
  sorry

end difference_of_squares_l2047_204709


namespace smallest_an_correct_l2047_204786

def smallest_an (n : ℕ) : ℕ :=
  if n = 1 then 2
  else if n = 3 then 11
  else 4 * n + 1

theorem smallest_an_correct (n : ℕ) (h : n ≥ 1) :
  ∀ (a : ℕ → ℕ),
  (∀ i j, 0 ≤ i → i < j → j ≤ n → a i < a j) →
  (∀ i j, 0 ≤ i → i < j → j ≤ n → ¬ Nat.Prime (a j - a i)) →
  a n ≥ smallest_an n :=
sorry

end smallest_an_correct_l2047_204786


namespace problem_1_l2047_204784

theorem problem_1 : 
  4 + 1/4 - 19/5 + 4/5 + 11/4 = 4 := by sorry

end problem_1_l2047_204784


namespace phoenix_flight_l2047_204768

def geometric_sequence (a₁ : ℝ) (r : ℝ) (n : ℕ) : ℝ := a₁ * r^(n - 1)

theorem phoenix_flight :
  let a₁ := 3
  let r := 3
  ∀ n : ℕ, n < 8 → geometric_sequence a₁ r n ≤ 6560 ∧
  geometric_sequence a₁ r 8 > 6560 :=
by sorry

end phoenix_flight_l2047_204768


namespace mountain_elevation_difference_l2047_204783

/-- The elevation difference between two mountains -/
def elevation_difference (h b : ℕ) : ℕ := h - b

/-- Proves that the elevation difference between two mountains is 2500 feet -/
theorem mountain_elevation_difference :
  ∃ (h b : ℕ),
    h = 10000 ∧
    3 * h = 4 * b ∧
    elevation_difference h b = 2500 := by
  sorry

end mountain_elevation_difference_l2047_204783


namespace negation_of_universal_proposition_l2047_204766

theorem negation_of_universal_proposition :
  (¬ ∀ (x : ℕ), x > 0 → (x - 1)^2 > 0) ↔ (∃ (x : ℕ), x > 0 ∧ (x - 1)^2 ≤ 0) := by
  sorry

end negation_of_universal_proposition_l2047_204766


namespace basketball_not_table_tennis_count_l2047_204792

/-- Represents the class of students and their sports preferences -/
structure ClassSports where
  total : ℕ
  basketball : ℕ
  tableTennis : ℕ
  neither : ℕ

/-- The number of students who like basketball but not table tennis -/
def basketballNotTableTennis (c : ClassSports) : ℕ :=
  c.basketball - (c.total - c.tableTennis - c.neither)

/-- Theorem stating the number of students who like basketball but not table tennis -/
theorem basketball_not_table_tennis_count (c : ClassSports) 
  (h1 : c.total = 30)
  (h2 : c.basketball = 15)
  (h3 : c.tableTennis = 10)
  (h4 : c.neither = 8) :
  basketballNotTableTennis c = 12 := by
  sorry

end basketball_not_table_tennis_count_l2047_204792


namespace min_beacons_required_l2047_204744

/-- Represents a room in the maze --/
structure Room where
  x : Nat
  y : Nat

/-- Represents the maze structure --/
def Maze := List Room

/-- Calculates the distance between two rooms in the maze --/
def distance (maze : Maze) (r1 r2 : Room) : Nat :=
  sorry

/-- Checks if a set of beacons can uniquely identify all rooms --/
def can_identify_all_rooms (maze : Maze) (beacons : List Room) : Prop :=
  sorry

/-- The main theorem stating that at least 3 beacons are required --/
theorem min_beacons_required (maze : Maze) :
  ∀ (beacons : List Room),
    can_identify_all_rooms maze beacons →
    beacons.length ≥ 3 :=
  sorry

end min_beacons_required_l2047_204744


namespace smallest_dual_base_number_l2047_204738

/-- Represents a number in a given base -/
def BaseRepresentation (n : ℕ) (base : ℕ) : Prop :=
  ∃ (digit1 digit2 : ℕ), 
    n = digit1 * base + digit2 ∧
    digit1 < base ∧
    digit2 < base

/-- The smallest number representable in both base 6 and base 8 as AA and BB respectively -/
def SmallestDualBaseNumber : ℕ := 63

theorem smallest_dual_base_number :
  (BaseRepresentation SmallestDualBaseNumber 6) ∧
  (BaseRepresentation SmallestDualBaseNumber 8) ∧
  (∀ m : ℕ, m < SmallestDualBaseNumber →
    ¬(BaseRepresentation m 6 ∧ BaseRepresentation m 8)) :=
by sorry

end smallest_dual_base_number_l2047_204738


namespace no_linear_term_implies_a_equals_negative_four_l2047_204742

theorem no_linear_term_implies_a_equals_negative_four (a : ℝ) : 
  (∀ x : ℝ, ∃ b c : ℝ, (x + 4) * (x + a) = x^2 + b*x + c) → a = -4 := by
  sorry

end no_linear_term_implies_a_equals_negative_four_l2047_204742


namespace closest_point_l2047_204704

def v (t : ℝ) : Fin 3 → ℝ := fun i =>
  match i with
  | 0 => 3 + 5*t
  | 1 => -2 + 4*t
  | 2 => 1 + 2*t

def a : Fin 3 → ℝ := fun i =>
  match i with
  | 0 => -1
  | 1 => 1
  | 2 => -3

def direction : Fin 3 → ℝ := fun i =>
  match i with
  | 0 => 5
  | 1 => 4
  | 2 => 2

theorem closest_point (t : ℝ) :
  (∀ s : ℝ, ‖v t - a‖ ≤ ‖v s - a‖) ↔ t = -16/45 := by sorry

end closest_point_l2047_204704


namespace cos_squared_alpha_minus_pi_fourth_l2047_204733

theorem cos_squared_alpha_minus_pi_fourth (α : ℝ) (h : Real.sin (2 * α) = 1/3) :
  Real.cos (α - π/4)^2 = 2/3 := by
sorry

end cos_squared_alpha_minus_pi_fourth_l2047_204733
