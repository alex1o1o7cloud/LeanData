import Mathlib

namespace square_gt_abs_l587_58704

theorem square_gt_abs (a b : ℝ) : a > |b| → a^2 > b^2 := by
  sorry

end square_gt_abs_l587_58704


namespace abes_age_l587_58748

theorem abes_age (present_age : ℕ) : 
  present_age + (present_age - 7) = 35 → present_age = 21 :=
by sorry

end abes_age_l587_58748


namespace ellen_lego_problem_l587_58752

/-- Represents Ellen's Lego collection and calculations -/
theorem ellen_lego_problem (initial : ℕ) (lost : ℕ) (found : ℕ) 
  (h1 : initial = 12560) (h2 : lost = 478) (h3 : found = 342) :
  let current := initial - lost + found
  (current = 12424) ∧ 
  (((lost : ℚ) / (initial : ℚ)) * 100 = 381 / 100) := by
  sorry

#check ellen_lego_problem

end ellen_lego_problem_l587_58752


namespace common_root_condition_l587_58791

theorem common_root_condition (a : ℝ) : 
  (∃ x : ℝ, x^2 + a*x + 1 = 0 ∧ x^2 + x + a = 0) ↔ (a = 1 ∨ a = -2) :=
by sorry

end common_root_condition_l587_58791


namespace min_value_product_l587_58711

theorem min_value_product (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_prod : x * y * z = 1) :
  (2 * x + 3 * y) * (2 * y + 3 * z) * (2 * x * z + 1) ≥ 24 ∧
  ∃ (x₀ y₀ z₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ z₀ > 0 ∧ x₀ * y₀ * z₀ = 1 ∧
    (2 * x₀ + 3 * y₀) * (2 * y₀ + 3 * z₀) * (2 * x₀ * z₀ + 1) = 24 :=
by sorry

end min_value_product_l587_58711


namespace cosine_ratio_equals_one_l587_58785

theorem cosine_ratio_equals_one :
  (Real.cos (66 * π / 180) * Real.cos (6 * π / 180) + Real.cos (84 * π / 180) * Real.cos (24 * π / 180)) /
  (Real.cos (65 * π / 180) * Real.cos (5 * π / 180) + Real.cos (85 * π / 180) * Real.cos (25 * π / 180)) = 1 := by
  sorry

end cosine_ratio_equals_one_l587_58785


namespace hypergeometric_prob_and_max_likelihood_l587_58746

/-- Hypergeometric probability distribution -/
def hypergeometric_prob (N M n m : ℕ) : ℚ :=
  (Nat.choose M m * Nat.choose (N - M) (n - m)) / Nat.choose N n

/-- Maximum likelihood estimate for population size -/
def max_likelihood_estimate (M n m : ℕ) : ℕ :=
  (M * n) / m

theorem hypergeometric_prob_and_max_likelihood 
  (N M n m : ℕ) (h1 : M ≤ N) (h2 : n ≤ N) (h3 : m ≤ M) (h4 : m ≤ n) :
  (∀ N', hypergeometric_prob N' M n m ≤ hypergeometric_prob N M n m) →
  N = max_likelihood_estimate M n m := by
  sorry


end hypergeometric_prob_and_max_likelihood_l587_58746


namespace john_bought_three_reels_l587_58700

/-- The number of reels John bought -/
def num_reels : ℕ := sorry

/-- The length of fishing line in each reel (in meters) -/
def reel_length : ℕ := 100

/-- The length of each section after cutting (in meters) -/
def section_length : ℕ := 10

/-- The number of sections John got after cutting -/
def num_sections : ℕ := 30

/-- Theorem: John bought 3 reels of fishing line -/
theorem john_bought_three_reels :
  num_reels = 3 :=
by
  sorry

end john_bought_three_reels_l587_58700


namespace intersection_of_A_and_B_l587_58776

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - x - 6 < 0}
def B : Set ℝ := {x | x^2 + 2*x - 8 > 0}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | 2 < x ∧ x < 3} := by sorry

end intersection_of_A_and_B_l587_58776


namespace ellipse_intersection_theorem_l587_58759

/-- Ellipse C: x²/9 + y²/8 = 1 -/
def ellipse_C (x y : ℝ) : Prop := x^2/9 + y^2/8 = 1

/-- Line l: x = my + 1 -/
def line_l (m x y : ℝ) : Prop := x = m*y + 1

/-- Point on ellipse C -/
structure PointOnC where
  x : ℝ
  y : ℝ
  on_C : ellipse_C x y

/-- Foci and vertices of ellipse C -/
structure EllipseCPoints where
  F₁ : ℝ × ℝ
  F₂ : ℝ × ℝ
  A : ℝ × ℝ
  B : ℝ × ℝ

/-- Line l intersects ellipse C at points M and N -/
structure Intersection where
  m : ℝ
  M : PointOnC
  N : PointOnC
  M_on_l : line_l m M.x M.y
  N_on_l : line_l m N.x N.y
  y_conditions : M.y > 0 ∧ N.y < 0

/-- MA is perpendicular to NF₁ -/
def perpendicular (A M N F₁ : ℝ × ℝ) : Prop :=
  (M.2 - A.2) * (N.2 - F₁.2) = -(M.1 - A.1) * (N.1 - F₁.1)

/-- Theorem statement -/
theorem ellipse_intersection_theorem 
  (C : EllipseCPoints) 
  (I : Intersection) 
  (h_perp : perpendicular C.A (I.M.x, I.M.y) (I.N.x, I.N.y) C.F₁) :
  I.m = Real.sqrt 3 / 12 := by
  sorry

end ellipse_intersection_theorem_l587_58759


namespace triangle_expression_simplification_l587_58717

-- Define a triangle with side lengths a, b, c
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  -- Triangle inequality conditions
  ab_gt_c : a + b > c
  ac_gt_b : a + c > b
  bc_gt_a : b + c > a

-- Define the theorem
theorem triangle_expression_simplification (t : Triangle) :
  |t.a - t.b - t.c| + |t.b - t.a - t.c| - |t.c - t.a + t.b| = t.a - t.b + t.c :=
by sorry

end triangle_expression_simplification_l587_58717


namespace simplify_expression_1_simplify_expression_2_l587_58739

-- Define variables
variable (x y : ℝ)

-- Theorem for the first expression
theorem simplify_expression_1 : (2*x + 1) - (3 - x) = 3*x - 2 := by sorry

-- Theorem for the second expression
theorem simplify_expression_2 : x^2*y - (2*x*y^2 - 5*x^2*y) + 3*x*y^2 - y^3 = 6*x^2*y + x*y^2 - y^3 := by sorry

end simplify_expression_1_simplify_expression_2_l587_58739


namespace square_sum_equality_l587_58784

-- Define the problem statement
theorem square_sum_equality (x y : ℝ) 
  (h1 : y + 9 = (x - 3)^2) 
  (h2 : x + 9 = (y - 3)^2) 
  (h3 : x ≠ y) : 
  x^2 + y^2 = 49 := by
sorry

-- Additional helper lemmas if needed
lemma helper_lemma (x y : ℝ) 
  (h1 : y + 9 = (x - 3)^2) 
  (h2 : x + 9 = (y - 3)^2) 
  (h3 : x ≠ y) : 
  x + y = 7 := by
sorry

end square_sum_equality_l587_58784


namespace chocolate_bar_cost_l587_58705

theorem chocolate_bar_cost (total_cost : ℚ) (num_chocolate_bars : ℕ) (num_gummy_packs : ℕ) (num_chip_bags : ℕ) 
  (gummy_pack_cost : ℚ) (chip_bag_cost : ℚ) :
  total_cost = 150 →
  num_chocolate_bars = 10 →
  num_gummy_packs = 10 →
  num_chip_bags = 20 →
  gummy_pack_cost = 2 →
  chip_bag_cost = 5 →
  (total_cost - (num_gummy_packs * gummy_pack_cost + num_chip_bags * chip_bag_cost)) / num_chocolate_bars = 3 := by
  sorry

end chocolate_bar_cost_l587_58705


namespace ellipse_foci_l587_58754

/-- The ellipse equation -/
def ellipse_equation (x y : ℝ) : Prop := 2 * x^2 + y^2 = 8

/-- The foci coordinates -/
def foci_coordinates : Set (ℝ × ℝ) := {(0, 2), (0, -2)}

/-- Theorem: The foci of the ellipse 2x^2 + y^2 = 8 are at (0, ±2) -/
theorem ellipse_foci :
  ∀ (f : ℝ × ℝ), f ∈ foci_coordinates ↔ 
  (∃ (a b c : ℝ), 
    (∀ x y, ellipse_equation x y ↔ (x^2 / a^2 + y^2 / b^2 = 1)) ∧
    (a > b) ∧
    (c^2 = a^2 - b^2) ∧
    (f = (0, c) ∨ f = (0, -c))) :=
sorry

end ellipse_foci_l587_58754


namespace ball_box_theorem_l587_58783

/-- Given a box with 60 balls where the probability of picking a white ball is 0.25,
    this theorem proves the number of white and black balls, and the number of
    additional white balls needed to change the probability to 2/5. -/
theorem ball_box_theorem (total_balls : ℕ) (prob_white : ℚ) :
  total_balls = 60 →
  prob_white = 1/4 →
  ∃ (white_balls black_balls additional_balls : ℕ),
    white_balls = 15 ∧
    black_balls = 45 ∧
    additional_balls = 15 ∧
    white_balls + black_balls = total_balls ∧
    (white_balls : ℚ) / total_balls = prob_white ∧
    ((white_balls + additional_balls : ℚ) / (total_balls + additional_balls) = 2/5) :=
by sorry

end ball_box_theorem_l587_58783


namespace grandmothers_age_is_77_l587_58736

/-- The grandmother's age is obtained by writing the Latin grade twice in a row -/
def grandmothers_age (latin_grade : ℕ) : ℕ := 11 * latin_grade

/-- The morning grade is obtained by dividing the grandmother's age by the number of kittens and subtracting fourteen-thirds -/
def morning_grade (age : ℕ) (kittens : ℕ) : ℚ := age / kittens - 14 / 3

theorem grandmothers_age_is_77 :
  ∃ (latin_grade : ℕ) (kittens : ℕ),
    latin_grade < 10 ∧
    kittens % 3 = 0 ∧
    grandmothers_age latin_grade = 77 ∧
    morning_grade (grandmothers_age latin_grade) kittens = latin_grade :=
by sorry

end grandmothers_age_is_77_l587_58736


namespace base_6_divisibility_l587_58742

def base_6_to_10 (a b c d : ℕ) : ℕ := a * 6^3 + b * 6^2 + c * 6 + d

theorem base_6_divisibility :
  ∃! (d : ℕ), d < 6 ∧ (base_6_to_10 3 d d 7) % 13 = 0 :=
sorry

end base_6_divisibility_l587_58742


namespace sample_size_calculation_l587_58755

/-- Represents the sample size for each school level -/
structure SampleSize where
  elementary : ℕ
  middle : ℕ
  high : ℕ

/-- Calculates the total sample size -/
def totalSampleSize (s : SampleSize) : ℕ :=
  s.elementary + s.middle + s.high

/-- The ratio of elementary:middle:high school students -/
def schoolRatio : Fin 3 → ℕ
  | 0 => 2
  | 1 => 3
  | 2 => 5

theorem sample_size_calculation (s : SampleSize) 
  (h_ratio : s.elementary * schoolRatio 1 = s.middle * schoolRatio 0 ∧ 
             s.middle * schoolRatio 2 = s.high * schoolRatio 1)
  (h_middle : s.middle = 150) : 
  totalSampleSize s = 500 := by
sorry

end sample_size_calculation_l587_58755


namespace different_color_probability_l587_58708

/-- The probability of drawing two chips of different colors from a bag containing 
    7 red chips and 4 green chips, when drawing with replacement. -/
theorem different_color_probability :
  let total_chips : ℕ := 7 + 4
  let red_chips : ℕ := 7
  let green_chips : ℕ := 4
  let prob_red : ℚ := red_chips / total_chips
  let prob_green : ℚ := green_chips / total_chips
  prob_red * prob_green + prob_green * prob_red = 56 / 121 := by
sorry

end different_color_probability_l587_58708


namespace factorial_square_root_l587_58732

theorem factorial_square_root (n : ℕ) (h : n = 5) : 
  Real.sqrt (n.factorial * (n.factorial ^ 2)) = 240 * Real.sqrt 30 := by
  sorry

end factorial_square_root_l587_58732


namespace spring_percentage_is_ten_percent_l587_58731

/-- The percentage of students who chose Spring -/
def spring_percentage (total : ℕ) (spring : ℕ) : ℚ :=
  (spring : ℚ) / (total : ℚ) * 100

/-- Theorem: The percentage of students who chose Spring is 10% -/
theorem spring_percentage_is_ten_percent :
  spring_percentage 10 1 = 10 := by
  sorry

end spring_percentage_is_ten_percent_l587_58731


namespace inequality_proof_l587_58796

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^2 / b) + (b^3 / c^2) + (c^4 / a^3) ≥ -a + 2*b + 2*c := by
  sorry

end inequality_proof_l587_58796


namespace orange_ratio_l587_58767

theorem orange_ratio (good_oranges bad_oranges : ℕ) 
  (h1 : good_oranges = 24) 
  (h2 : bad_oranges = 8) : 
  (good_oranges : ℚ) / bad_oranges = 3 / 1 := by
  sorry

end orange_ratio_l587_58767


namespace max_distinct_pairs_l587_58750

theorem max_distinct_pairs (n : ℕ) (h : n = 2023) :
  let S := Finset.range n
  ∃ (k : ℕ) (pairs : Finset (ℕ × ℕ)),
    k = 809 ∧
    pairs.card = k ∧
    (∀ (p : ℕ × ℕ), p ∈ pairs → p.1 ∈ S ∧ p.2 ∈ S ∧ p.1 < p.2) ∧
    (∀ (p q : ℕ × ℕ), p ∈ pairs → q ∈ pairs → p ≠ q → p.1 ≠ q.1 ∧ p.1 ≠ q.2 ∧ p.2 ≠ q.1 ∧ p.2 ≠ q.2) ∧
    (∀ (p q : ℕ × ℕ), p ∈ pairs → q ∈ pairs → p ≠ q → p.1 + p.2 ≠ q.1 + q.2) ∧
    (∀ (p : ℕ × ℕ), p ∈ pairs → p.1 + p.2 ≤ n) ∧
    (∀ (m : ℕ) (larger_pairs : Finset (ℕ × ℕ)),
      m > k →
      (larger_pairs.card = m →
        ¬((∀ (p : ℕ × ℕ), p ∈ larger_pairs → p.1 ∈ S ∧ p.2 ∈ S ∧ p.1 < p.2) ∧
          (∀ (p q : ℕ × ℕ), p ∈ larger_pairs → q ∈ larger_pairs → p ≠ q → p.1 ≠ q.1 ∧ p.1 ≠ q.2 ∧ p.2 ≠ q.1 ∧ p.2 ≠ q.2) ∧
          (∀ (p q : ℕ × ℕ), p ∈ larger_pairs → q ∈ larger_pairs → p ≠ q → p.1 + p.2 ≠ q.1 + q.2) ∧
          (∀ (p : ℕ × ℕ), p ∈ larger_pairs → p.1 + p.2 ≤ n)))) :=
by
  sorry

end max_distinct_pairs_l587_58750


namespace earbuds_tickets_proof_l587_58756

/-- The number of tickets Connie spent on earbuds -/
def tickets_on_earbuds (total_tickets : ℕ) (tickets_on_koala : ℕ) (tickets_on_bracelets : ℕ) : ℕ :=
  total_tickets - tickets_on_koala - tickets_on_bracelets

theorem earbuds_tickets_proof :
  let total_tickets : ℕ := 50
  let tickets_on_koala : ℕ := total_tickets / 2
  let tickets_on_bracelets : ℕ := 15
  tickets_on_earbuds total_tickets tickets_on_koala tickets_on_bracelets = 10 := by
  sorry

#eval tickets_on_earbuds 50 25 15

end earbuds_tickets_proof_l587_58756


namespace min_amount_for_equal_distribution_l587_58701

/-- Given initial sheets of paper, number of students, and cost per sheet,
    calculate the minimum amount needed to buy additional sheets for equal distribution. -/
def min_amount_needed (initial_sheets : ℕ) (num_students : ℕ) (cost_per_sheet : ℕ) : ℕ :=
  let total_sheets_needed := (num_students * ((initial_sheets + num_students - 1) / num_students))
  let additional_sheets := total_sheets_needed - initial_sheets
  additional_sheets * cost_per_sheet

/-- Theorem stating that given 98 sheets of paper, 12 students, and a cost of 450 won per sheet,
    the minimum amount needed to buy additional sheets for equal distribution is 4500 won. -/
theorem min_amount_for_equal_distribution :
  min_amount_needed 98 12 450 = 4500 := by
  sorry

end min_amount_for_equal_distribution_l587_58701


namespace area_is_14_4_l587_58719

/-- An isosceles trapezoid with an inscribed circle -/
structure IsoscelesTrapezoidWithInscribedCircle where
  /-- Distance from circle center to one end of a non-parallel side -/
  d1 : ℝ
  /-- Distance from circle center to the other end of the same non-parallel side -/
  d2 : ℝ
  /-- Assumption that d1 and d2 are positive -/
  d1_pos : d1 > 0
  d2_pos : d2 > 0

/-- The area of the isosceles trapezoid with an inscribed circle -/
def area (t : IsoscelesTrapezoidWithInscribedCircle) : ℝ :=
  14.4

/-- Theorem stating that the area of the isosceles trapezoid with an inscribed circle is 14.4 cm² -/
theorem area_is_14_4 (t : IsoscelesTrapezoidWithInscribedCircle) 
    (h1 : t.d1 = 2) (h2 : t.d2 = 4) : area t = 14.4 := by
  sorry

end area_is_14_4_l587_58719


namespace roots_properties_l587_58722

theorem roots_properties (a b m : ℝ) (h1 : 2 * a^2 - 8 * a + m = 0)
                                    (h2 : 2 * b^2 - 8 * b + m = 0)
                                    (h3 : m > 0) :
  (a^2 + b^2 ≥ 8) ∧
  (Real.sqrt a + Real.sqrt b ≤ 2 * Real.sqrt 2) ∧
  (1 / (a + 2) + 1 / (2 * b) ≥ (3 + 2 * Real.sqrt 2) / 12) := by
  sorry

end roots_properties_l587_58722


namespace probability_of_odd_product_l587_58703

def A : Finset ℕ := {1, 2, 3}
def B : Finset ℕ := {0, 1, 3}

def is_product_odd (a b : ℕ) : Bool := (a * b) % 2 = 1

def favorable_outcomes : Finset (ℕ × ℕ) :=
  A.product B |>.filter (fun (a, b) => is_product_odd a b)

theorem probability_of_odd_product :
  (favorable_outcomes.card : ℚ) / ((A.card * B.card) : ℚ) = 4 / 9 := by
  sorry

#eval favorable_outcomes -- To check the favorable outcomes
#eval favorable_outcomes.card -- To check the number of favorable outcomes
#eval A.card * B.card -- To check the total number of outcomes

end probability_of_odd_product_l587_58703


namespace officer_selection_theorem_l587_58798

def club_size : ℕ := 25
def num_officers : ℕ := 3

def ways_to_choose_officers : ℕ :=
  let ways_without_alice_bob := (club_size - 2) * (club_size - 3) * (club_size - 4)
  let ways_with_alice_bob := 3 * 2 * (club_size - 2)
  ways_without_alice_bob + ways_with_alice_bob

theorem officer_selection_theorem :
  ways_to_choose_officers = 10764 := by sorry

end officer_selection_theorem_l587_58798


namespace limit_of_sequence_a_l587_58789

def a (n : ℕ) : ℚ := (4*n - 3) / (2*n + 1)

theorem limit_of_sequence_a : 
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |a n - 2| < ε :=
sorry

end limit_of_sequence_a_l587_58789


namespace night_day_worker_loading_ratio_l587_58792

theorem night_day_worker_loading_ratio
  (day_workers : ℚ)
  (night_workers : ℚ)
  (total_boxes : ℚ)
  (h1 : night_workers = (4/5) * day_workers)
  (h2 : (5/6) * total_boxes = day_workers * (boxes_per_day_worker : ℚ))
  (h3 : (1/6) * total_boxes = night_workers * (boxes_per_night_worker : ℚ)) :
  boxes_per_night_worker / boxes_per_day_worker = 25/4 := by
  sorry

end night_day_worker_loading_ratio_l587_58792


namespace solution_values_l587_58733

-- Define the solution sets A and B
def A : Set ℝ := {x : ℝ | x^2 - 2*x - 3 < 0}
def B : Set ℝ := {x : ℝ | x^2 + x - 6 < 0}

-- Define the intersection of A and B
def A_intersect_B : Set ℝ := A ∩ B

-- Define the solution set of x^2 + ax + b < 0
def solution_set (a b : ℝ) : Set ℝ := {x : ℝ | x^2 + a*x + b < 0}

-- Theorem statement
theorem solution_values :
  ∃ (a b : ℝ), solution_set a b = A_intersect_B ∧ a = -1 ∧ b = -2 :=
sorry

end solution_values_l587_58733


namespace cherry_price_proof_l587_58787

-- Define the discount rate
def discount_rate : ℝ := 0.3

-- Define the discounted price for a quarter-pound package
def discounted_quarter_pound_price : ℝ := 2

-- Define the weight of a full pound in terms of quarter-pounds
def full_pound_weight : ℝ := 4

-- Define the regular price for a full pound of cherries
def regular_full_pound_price : ℝ := 11.43

theorem cherry_price_proof :
  (1 - discount_rate) * regular_full_pound_price / full_pound_weight = discounted_quarter_pound_price := by
  sorry

end cherry_price_proof_l587_58787


namespace largest_angle_in_pentagon_l587_58762

/-- Given a pentagon ABCDE with the following properties:
  - Angle A measures 80°
  - Angle B measures 95°
  - Angles C and D are equal
  - Angle E is 10° less than three times angle C
  Prove that the largest angle in the pentagon measures 221° -/
theorem largest_angle_in_pentagon (A B C D E : ℝ) : 
  A = 80 ∧ 
  B = 95 ∧ 
  C = D ∧ 
  E = 3 * C - 10 ∧ 
  A + B + C + D + E = 540 →
  max A (max B (max C (max D E))) = 221 := by
  sorry

end largest_angle_in_pentagon_l587_58762


namespace triangle_side_validity_l587_58749

/-- Checks if three lengths can form a valid triangle -/
def is_valid_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

theorem triangle_side_validity :
  let side1 := 5
  let side2 := 7
  (is_valid_triangle side1 side2 6) ∧
  ¬(is_valid_triangle side1 side2 2) ∧
  ¬(is_valid_triangle side1 side2 17) :=
by sorry

end triangle_side_validity_l587_58749


namespace john_outfit_cost_l587_58740

/-- Calculates the final cost of John's outfit in Euros -/
def outfit_cost_in_euros (pants_cost shirt_percent_increase shirt_discount outfit_tax
                          hat_cost hat_discount hat_tax
                          shoes_cost shoes_discount shoes_tax
                          usd_to_eur_rate : ℝ) : ℝ :=
  let shirt_cost := pants_cost * (1 + shirt_percent_increase)
  let shirt_discounted := shirt_cost * (1 - shirt_discount)
  let outfit_cost := (pants_cost + shirt_discounted) * (1 + outfit_tax)
  let hat_discounted := hat_cost * (1 - hat_discount)
  let hat_with_tax := hat_discounted * (1 + hat_tax)
  let shoes_discounted := shoes_cost * (1 - shoes_discount)
  let shoes_with_tax := shoes_discounted * (1 + shoes_tax)
  let total_usd := outfit_cost + hat_with_tax + shoes_with_tax
  total_usd * usd_to_eur_rate

/-- The final cost of John's outfit in Euros is approximately 175.93 -/
theorem john_outfit_cost :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.005 ∧ 
  |outfit_cost_in_euros 50 0.6 0.15 0.07 25 0.1 0.06 70 0.2 0.08 0.85 - 175.93| < ε :=
sorry

end john_outfit_cost_l587_58740


namespace parabola_through_point_l587_58772

/-- 
If a parabola with equation y = ax^2 - 2x + 3 passes through the point (1, 2), 
then the value of a is 1.
-/
theorem parabola_through_point (a : ℝ) : 
  (2 : ℝ) = a * (1 : ℝ)^2 - 2 * (1 : ℝ) + 3 → a = 1 := by
  sorry

end parabola_through_point_l587_58772


namespace quadratic_inequality_range_l587_58797

theorem quadratic_inequality_range (m : ℝ) :
  (∀ x : ℝ, x^2 + m*x + 2*m - 3 ≥ 0) ↔ (2 ≤ m ∧ m ≤ 6) :=
sorry

end quadratic_inequality_range_l587_58797


namespace store_sale_profit_store_sale_result_l587_58718

/-- Calculates the money left after a store's inventory sale --/
theorem store_sale_profit (total_items : ℕ) (retail_price : ℚ) (discount_percent : ℚ) 
  (sold_percent : ℚ) (debt : ℚ) : ℚ :=
  let items_sold := total_items * sold_percent
  let discount_amount := retail_price * discount_percent
  let sale_price := retail_price - discount_amount
  let total_revenue := items_sold * sale_price
  let profit := total_revenue - debt
  profit

/-- Proves that the store has $3000 left after the sale --/
theorem store_sale_result : 
  store_sale_profit 2000 50 0.8 0.9 15000 = 3000 := by
  sorry

end store_sale_profit_store_sale_result_l587_58718


namespace month_mean_profit_l587_58782

/-- Calculates the mean daily profit for a month given the mean profits of two equal periods -/
def mean_daily_profit (days : ℕ) (mean_profit1 : ℚ) (mean_profit2 : ℚ) : ℚ :=
  (mean_profit1 + mean_profit2) / 2

theorem month_mean_profit : 
  let days : ℕ := 30
  let first_half_mean : ℚ := 275
  let second_half_mean : ℚ := 425
  mean_daily_profit days first_half_mean second_half_mean = 350 := by
sorry

end month_mean_profit_l587_58782


namespace sum_of_ages_age_difference_l587_58723

/-- Tyler's age -/
def tyler_age : ℕ := 7

/-- Tyler's brother's age -/
def brother_age : ℕ := 11 - tyler_age

/-- The sum of Tyler's and his brother's ages -/
theorem sum_of_ages : tyler_age + brother_age = 11 := by sorry

/-- The difference between Tyler's brother's age and Tyler's age -/
theorem age_difference : brother_age - tyler_age = 4 := by sorry

end sum_of_ages_age_difference_l587_58723


namespace number_exceeds_16_percent_l587_58763

theorem number_exceeds_16_percent : ∃ x : ℝ, x = 100 ∧ x = 0.16 * x + 84 := by
  sorry

end number_exceeds_16_percent_l587_58763


namespace group_purchase_equation_system_l587_58745

/-- Represents a group purchase scenario -/
structure GroupPurchase where
  people : ℕ
  price : ℕ
  excess_9 : ℕ
  shortage_6 : ℕ

/-- The group purchase scenario satisfies the given conditions -/
def satisfies_conditions (gp : GroupPurchase) : Prop :=
  9 * gp.people - gp.price = gp.excess_9 ∧
  gp.price - 6 * gp.people = gp.shortage_6

/-- The system of equations correctly represents the group purchase scenario -/
theorem group_purchase_equation_system (gp : GroupPurchase) 
  (h : satisfies_conditions gp) (h_excess : gp.excess_9 = 4) (h_shortage : gp.shortage_6 = 5) :
  9 * gp.people - gp.price = 4 ∧ gp.price - 6 * gp.people = 5 := by
  sorry

#check group_purchase_equation_system

end group_purchase_equation_system_l587_58745


namespace candy_duration_l587_58743

theorem candy_duration (neighbors_candy : ℝ) (sister_candy : ℝ) (daily_consumption : ℝ) :
  neighbors_candy = 11.0 →
  sister_candy = 5.0 →
  daily_consumption = 8.0 →
  (neighbors_candy + sister_candy) / daily_consumption = 2.0 := by
  sorry

end candy_duration_l587_58743


namespace equation_solution_l587_58734

theorem equation_solution : 
  ∃ x : ℝ, (1 / 7 + 7 / x = 15 / x + 1 / 15) ∧ (x = 105) := by
  sorry

end equation_solution_l587_58734


namespace bacteria_states_l587_58709

-- Define the bacteria types
inductive BacteriaType
| Red
| Blue

-- Define the state of the bacteria population
structure BacteriaState where
  red : ℕ
  blue : ℕ

-- Define the transformation rules
def transform (state : BacteriaState) : Set BacteriaState :=
  { BacteriaState.mk (state.red - 2) (state.blue + 1),  -- Two red to one blue
    BacteriaState.mk (state.red + 4) (state.blue - 2),  -- Two blue to four red
    BacteriaState.mk (state.red + 2) (state.blue - 1) } -- One red and one blue to three red

-- Define the initial state
def initial_state (r b : ℕ) : BacteriaState :=
  BacteriaState.mk r b

-- Define the set of possible states
def possible_states (n : ℕ) : Set BacteriaState :=
  {state | ∃ m : ℕ, state.red = n - 2 * m ∧ state.blue = m}

-- Theorem statement
theorem bacteria_states (r b : ℕ) :
  let init := initial_state r b
  let n := r + b
  ∀ state, state ∈ possible_states n ↔ 
    ∃ sequence : ℕ → BacteriaState, 
      sequence 0 = init ∧
      (∀ i, sequence (i + 1) ∈ transform (sequence i)) ∧
      (∃ j, sequence j = state) :=
by sorry

end bacteria_states_l587_58709


namespace sum_of_special_integers_l587_58766

/-- A positive integer with exactly two positive divisors -/
def smallest_two_divisor_integer : ℕ+ := sorry

/-- The largest integer less than 150 with exactly three positive divisors -/
def largest_three_divisor_integer_below_150 : ℕ+ := sorry

/-- The sum of the smallest integer with two positive divisors and 
    the largest integer less than 150 with three positive divisors -/
theorem sum_of_special_integers : 
  (smallest_two_divisor_integer : ℕ) + (largest_three_divisor_integer_below_150 : ℕ) = 123 := by sorry

end sum_of_special_integers_l587_58766


namespace bobby_deadlift_increase_l587_58726

/-- Represents Bobby's deadlift progression --/
structure DeadliftProgress where
  initial_weight : ℕ
  initial_age : ℕ
  final_age : ℕ
  percentage_increase : ℕ
  additional_weight : ℕ

/-- Calculates the average yearly increase in Bobby's deadlift --/
def average_yearly_increase (d : DeadliftProgress) : ℚ :=
  let final_weight := d.initial_weight * (d.percentage_increase : ℚ) / 100 + d.additional_weight
  let total_increase := final_weight - d.initial_weight
  let years := d.final_age - d.initial_age
  total_increase / years

/-- Theorem stating that Bobby's average yearly increase in deadlift is 110 pounds --/
theorem bobby_deadlift_increase :
  let bobby := DeadliftProgress.mk 300 13 18 250 100
  average_yearly_increase bobby = 110 := by
  sorry

end bobby_deadlift_increase_l587_58726


namespace ad_transmission_cost_l587_58786

/-- The cost of transmitting advertisements during a race -/
theorem ad_transmission_cost
  (num_ads : ℕ)
  (ad_duration : ℕ)
  (cost_per_minute : ℕ)
  (h1 : num_ads = 5)
  (h2 : ad_duration = 3)
  (h3 : cost_per_minute = 4000) :
  num_ads * ad_duration * cost_per_minute = 60000 :=
by sorry

end ad_transmission_cost_l587_58786


namespace smallest_square_area_l587_58790

/-- A square in the plane --/
structure RotatedSquare where
  center : ℤ × ℤ
  sideLength : ℝ
  rotation : ℝ

/-- Count the number of lattice points on the boundary of a rotated square --/
def countBoundaryLatticePoints (s : RotatedSquare) : ℕ :=
  sorry

/-- The area of a square --/
def squareArea (s : RotatedSquare) : ℝ :=
  s.sideLength ^ 2

/-- The theorem stating the area of the smallest square meeting the conditions --/
theorem smallest_square_area : 
  ∃ (s : RotatedSquare), 
    (∀ (s' : RotatedSquare), 
      countBoundaryLatticePoints s' = 5 → squareArea s ≤ squareArea s') ∧ 
    countBoundaryLatticePoints s = 5 ∧ 
    squareArea s = 32 :=
  sorry

end smallest_square_area_l587_58790


namespace unit_square_quadrilateral_inequalities_l587_58793

/-- A quadrilateral formed by selecting one point on each side of a unit square -/
structure UnitSquareQuadrilateral where
  a : Real
  b : Real
  c : Real
  d : Real
  a_nonneg : 0 ≤ a
  b_nonneg : 0 ≤ b
  c_nonneg : 0 ≤ c
  d_nonneg : 0 ≤ d
  a_le_one : a ≤ 1
  b_le_one : b ≤ 1
  c_le_one : c ≤ 1
  d_le_one : d ≤ 1

theorem unit_square_quadrilateral_inequalities (q : UnitSquareQuadrilateral) :
  2 ≤ q.a ^ 2 + q.b ^ 2 + q.c ^ 2 + q.d ^ 2 ∧
  q.a ^ 2 + q.b ^ 2 + q.c ^ 2 + q.d ^ 2 ≤ 4 ∧
  2 * Real.sqrt 2 ≤ q.a + q.b + q.c + q.d ∧
  q.a + q.b + q.c + q.d ≤ 4 := by
  sorry

end unit_square_quadrilateral_inequalities_l587_58793


namespace gcd_n4_plus_125_and_n_plus_5_l587_58707

theorem gcd_n4_plus_125_and_n_plus_5 (n : ℕ) (h1 : n > 0) (h2 : ¬ 7 ∣ n) :
  (Nat.gcd (n^4 + 5^3) (n + 5) = 1) ∨ (Nat.gcd (n^4 + 5^3) (n + 5) = 3) := by
sorry

end gcd_n4_plus_125_and_n_plus_5_l587_58707


namespace right_triangle_hypotenuse_l587_58751

theorem right_triangle_hypotenuse (a b c : ℝ) :
  a = 15 ∧ b = 36 ∧ c^2 = a^2 + b^2 → c = 39 := by
  sorry

end right_triangle_hypotenuse_l587_58751


namespace parabola_linear_function_relationship_l587_58779

-- Define the parabola
def parabola (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x

-- Define the linear function
def linear_function (a b : ℝ) (x : ℝ) : ℝ := (a - b) * x + b

theorem parabola_linear_function_relationship 
  (a b m : ℝ) 
  (h1 : a < 0)  -- parabola opens downwards
  (h2 : m < 0)  -- P(-1, m) is in the third quadrant
  (h3 : parabola a b (-1) = m)  -- parabola passes through P(-1, m)
  (h4 : -b / (2*a) < 0)  -- axis of symmetry is negative (P and origin on opposite sides)
  : ∀ x y : ℝ, x > 0 ∧ y > 0 → linear_function a b x ≠ y :=
by sorry

end parabola_linear_function_relationship_l587_58779


namespace only_C_is_certain_l587_58799

-- Define the event type
inductive Event
  | A  -- The temperature in Aojiang on June 1st this year is 30 degrees
  | B  -- There are 10 red balls in a box, and any ball taken out must be a white ball
  | C  -- Throwing a stone, the stone will eventually fall
  | D  -- In this math competition, every participating student will score full marks

-- Define what it means for an event to be certain
def is_certain (e : Event) : Prop :=
  match e with
  | Event.C => True
  | _ => False

-- Theorem statement
theorem only_C_is_certain :
  ∀ e : Event, is_certain e ↔ e = Event.C :=
by sorry

end only_C_is_certain_l587_58799


namespace negative_cube_divided_by_base_l587_58764

theorem negative_cube_divided_by_base (a : ℝ) (h : a ≠ 0) : -a^3 / a = -a^2 := by
  sorry

end negative_cube_divided_by_base_l587_58764


namespace arithmetic_sequence_general_term_l587_58710

/-- An arithmetic sequence {a_n} with specific conditions -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  (∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d) ∧ 
  a 3 = 7 ∧
  a 5 = a 2 + 6

/-- The general term formula for the arithmetic sequence -/
def general_term (n : ℕ) : ℝ := 2 * n + 1

/-- Theorem stating that the general term formula is correct for the given arithmetic sequence -/
theorem arithmetic_sequence_general_term (a : ℕ → ℝ) :
  arithmetic_sequence a → ∀ n : ℕ, a n = general_term n := by
  sorry

end arithmetic_sequence_general_term_l587_58710


namespace gum_distribution_l587_58744

theorem gum_distribution (num_cousins : ℕ) (gum_per_cousin : ℕ) (total_gum : ℕ) : 
  num_cousins = 4 → gum_per_cousin = 5 → total_gum = num_cousins * gum_per_cousin → total_gum = 20 := by
  sorry

end gum_distribution_l587_58744


namespace grants_test_score_l587_58721

theorem grants_test_score (hunter_score john_score grant_score : ℕ) :
  hunter_score = 45 →
  john_score = 2 * hunter_score →
  grant_score = john_score + 10 →
  grant_score = 100 := by
sorry

end grants_test_score_l587_58721


namespace nancy_bottle_caps_l587_58725

theorem nancy_bottle_caps (initial final found : ℕ) : 
  initial = 91 → final = 179 → found = final - initial :=
by sorry

end nancy_bottle_caps_l587_58725


namespace premier_pups_count_l587_58716

theorem premier_pups_count :
  let fetch : ℕ := 70
  let jump : ℕ := 40
  let bark : ℕ := 45
  let fetch_and_jump : ℕ := 25
  let jump_and_bark : ℕ := 15
  let fetch_and_bark : ℕ := 20
  let all_three : ℕ := 12
  let none : ℕ := 15
  
  let fetch_only : ℕ := fetch - (fetch_and_jump + fetch_and_bark - all_three)
  let jump_only : ℕ := jump - (fetch_and_jump + jump_and_bark - all_three)
  let bark_only : ℕ := bark - (fetch_and_bark + jump_and_bark - all_three)
  let fetch_jump_only : ℕ := fetch_and_jump - all_three
  let jump_bark_only : ℕ := jump_and_bark - all_three
  let fetch_bark_only : ℕ := fetch_and_bark - all_three

  fetch_only + jump_only + bark_only + fetch_jump_only + jump_bark_only + fetch_bark_only + all_three + none = 122 := by
  sorry

end premier_pups_count_l587_58716


namespace percentage_excess_l587_58727

theorem percentage_excess (x y : ℝ) (h : x = 0.38 * y) :
  (y - x) / x = 0.62 := by
  sorry

end percentage_excess_l587_58727


namespace fraction_calculation_l587_58758

theorem fraction_calculation : (0.5 ^ 4) / (0.05 ^ 3) = 500 := by
  sorry

end fraction_calculation_l587_58758


namespace divisors_of_power_minus_one_l587_58729

theorem divisors_of_power_minus_one (a b r : ℕ) (ha : a ≥ 2) (hb : b > 0) (hb_composite : ∃ x y, 1 < x ∧ 1 < y ∧ b = x * y) (hr : ∃ (S : Finset ℕ), S.card = r ∧ ∀ x ∈ S, x > 0 ∧ x ∣ b) :
  ∃ (T : Finset ℕ), T.card ≥ r ∧ ∀ x ∈ T, x > 0 ∧ x ∣ (a^b - 1) :=
sorry

end divisors_of_power_minus_one_l587_58729


namespace max_value_cube_root_sum_and_sum_l587_58760

theorem max_value_cube_root_sum_and_sum (x y : ℝ) :
  (x^(1/3) + y^(1/3) = 2) →
  (x + y = 20) →
  max x y = 10 + 6 * Real.sqrt 3 :=
by sorry

end max_value_cube_root_sum_and_sum_l587_58760


namespace apple_weight_l587_58769

/-- Given a bag containing apples, prove the weight of one apple. -/
theorem apple_weight (total_weight : ℝ) (empty_bag_weight : ℝ) (apple_count : ℕ) :
  total_weight = 1.82 →
  empty_bag_weight = 0.5 →
  apple_count = 6 →
  (total_weight - empty_bag_weight) / apple_count = 0.22 := by
  sorry

end apple_weight_l587_58769


namespace problem_solution_l587_58777

def U : Set ℝ := Set.univ

def A : Set ℝ := {x | x^2 + 3*x + 2 = 0}

def B (m : ℝ) : Set ℝ := {x | x^2 + (m+1)*x + m = 0}

theorem problem_solution (m : ℝ) : (U \ A) ∩ B m = ∅ → m = 1 ∨ m = 2 := by
  sorry

end problem_solution_l587_58777


namespace price_reduction_percentage_l587_58773

theorem price_reduction_percentage (original_price new_price : ℝ) 
  (h1 : original_price = 250)
  (h2 : new_price = 200) :
  (original_price - new_price) / original_price * 100 = 20 := by
sorry

end price_reduction_percentage_l587_58773


namespace reflection_of_circle_center_l587_58706

/-- Reflects a point (x, y) about the line y = -x -/
def reflect_about_y_neg_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.2, -p.1)

theorem reflection_of_circle_center :
  let original_center : ℝ × ℝ := (8, -3)
  let reflected_center : ℝ × ℝ := reflect_about_y_neg_x original_center
  reflected_center = (3, -8) := by sorry

end reflection_of_circle_center_l587_58706


namespace lemon_pie_degrees_l587_58778

/-- The number of degrees in a circle --/
def circle_degrees : ℕ := 360

/-- The total number of students in the class --/
def total_students : ℕ := 45

/-- The number of students preferring chocolate pie --/
def chocolate_pref : ℕ := 15

/-- The number of students preferring apple pie --/
def apple_pref : ℕ := 10

/-- The number of students preferring blueberry pie --/
def blueberry_pref : ℕ := 9

/-- Calculate the number of students preferring lemon pie --/
def lemon_pref : ℚ :=
  (total_students - (chocolate_pref + apple_pref + blueberry_pref)) / 2

/-- Theorem: The number of degrees for lemon pie on a pie chart is 44° --/
theorem lemon_pie_degrees : 
  (lemon_pref / total_students) * circle_degrees = 44 := by
  sorry

end lemon_pie_degrees_l587_58778


namespace line_equation_proof_l587_58795

/-- A line in 2D space represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if two lines are parallel -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

/-- Check if a point lies on a line -/
def point_on_line (x y : ℝ) (l : Line) : Prop :=
  l.a * x + l.b * y + l.c = 0

/-- The given line 2x - 3y + 4 = 0 -/
def given_line : Line :=
  { a := 2, b := -3, c := 4 }

/-- The point (-1, 2) -/
def point : (ℝ × ℝ) :=
  (-1, 2)

/-- The equation of the line we want to prove -/
def target_line : Line :=
  { a := 2, b := -3, c := 8 }

theorem line_equation_proof :
  parallel target_line given_line ∧
  point_on_line point.1 point.2 target_line :=
by sorry

end line_equation_proof_l587_58795


namespace quadratic_root_m_value_l587_58770

theorem quadratic_root_m_value (m : ℝ) : 
  (∃ x : ℝ, x^2 + 3*x - m = 0 ∧ x = 1) → m = 4 := by
  sorry

end quadratic_root_m_value_l587_58770


namespace other_number_is_twenty_l587_58780

theorem other_number_is_twenty (a b : ℕ) (h1 : a + b = 30) (h2 : a = 10 ∨ b = 10) : 
  (a = 20 ∨ b = 20) :=
by sorry

end other_number_is_twenty_l587_58780


namespace root_shift_polynomial_l587_58775

theorem root_shift_polynomial (a b c : ℂ) : 
  (∀ x : ℂ, x^3 - 5*x + 7 = 0 ↔ x = a ∨ x = b ∨ x = c) →
  (∀ x : ℂ, x^3 - 9*x^2 + 22*x - 5 = 0 ↔ x = a + 3 ∨ x = b + 3 ∨ x = c + 3) :=
by sorry

end root_shift_polynomial_l587_58775


namespace coefficient_sum_l587_58713

theorem coefficient_sum (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, (x - 2)^5 = a₅*x^5 + a₄*x^4 + a₃*x^3 + a₂*x^2 + a₁*x + a₀) →
  a₀ + a₁ + a₂ + a₃ + a₄ + a₅ = -1 := by
sorry

end coefficient_sum_l587_58713


namespace salary_calculation_l587_58794

/-- Represents the monthly salary in Rupees -/
def monthly_salary : ℝ := 1375

/-- Represents the savings rate as a decimal -/
def savings_rate : ℝ := 0.20

/-- Represents the expense increase rate as a decimal -/
def expense_increase_rate : ℝ := 0.20

/-- Represents the new savings amount after expense increase in Rupees -/
def new_savings : ℝ := 220

theorem salary_calculation :
  monthly_salary * savings_rate * (1 - expense_increase_rate) = new_savings :=
by sorry

end salary_calculation_l587_58794


namespace min_like_both_l587_58757

theorem min_like_both (total : ℕ) (like_mozart : ℕ) (like_beethoven : ℕ)
  (h_total : total = 120)
  (h_mozart : like_mozart = 102)
  (h_beethoven : like_beethoven = 85)
  : ∃ (like_both : ℕ), like_both ≥ 67 ∧ 
    (∀ (x : ℕ), x < like_both → 
      ∃ (only_mozart only_beethoven : ℕ),
        x + only_mozart + only_beethoven ≤ total ∧
        x + only_mozart ≤ like_mozart ∧
        x + only_beethoven ≤ like_beethoven) :=
by sorry

end min_like_both_l587_58757


namespace altitude_scientific_notation_l587_58768

/-- The altitude of a medium-high orbit satellite in China's Beidou satellite navigation system -/
def altitude : ℝ := 21500000

/-- The scientific notation representation of the altitude -/
def scientific_notation : ℝ := 2.15 * (10 ^ 7)

/-- Theorem stating that the altitude is equal to its scientific notation representation -/
theorem altitude_scientific_notation : altitude = scientific_notation := by
  sorry

end altitude_scientific_notation_l587_58768


namespace equation_solutions_l587_58728

theorem equation_solutions : 
  {x : ℝ | (1 / (x^2 + 11*x + 12) + 1 / (x^2 + 2*x + 3) + 1 / (x^2 - 13*x + 14) = 0)} = 
  {-4, -3, 3, 4} := by sorry

end equation_solutions_l587_58728


namespace percentage_of_1000_l587_58753

theorem percentage_of_1000 : (66.2 / 1000) * 100 = 6.62 := by
  sorry

end percentage_of_1000_l587_58753


namespace sufficient_not_necessary_l587_58747

/-- Two lines are perpendicular if and only if the product of their slopes is -1 -/
def perpendicular (m : ℝ) : Prop :=
  (m + 2) * (m - 2) + 3 * m * (m + 2) = 0

/-- The condition m = 1/2 -/
def condition (m : ℝ) : Prop := m = 1/2

/-- The statement that m = 1/2 is sufficient but not necessary for perpendicularity -/
theorem sufficient_not_necessary :
  (∀ m : ℝ, condition m → perpendicular m) ∧
  ¬(∀ m : ℝ, perpendicular m → condition m) := by
  sorry

end sufficient_not_necessary_l587_58747


namespace min_sum_of_tangent_product_l587_58702

theorem min_sum_of_tangent_product (x y : ℝ) :
  (Real.tan x - 2) * (Real.tan y - 2) = 5 →
  ∃ (min_sum : ℝ), min_sum = Real.pi - Real.arctan (1 / 2) ∧
    ∀ (a b : ℝ), (Real.tan a - 2) * (Real.tan b - 2) = 5 →
      a + b ≥ min_sum := by
  sorry

end min_sum_of_tangent_product_l587_58702


namespace smallest_multiple_of_6_and_15_l587_58771

theorem smallest_multiple_of_6_and_15 : 
  ∃ b : ℕ+, (∀ n : ℕ+, 6 ∣ n ∧ 15 ∣ n → b ≤ n) ∧ 6 ∣ b ∧ 15 ∣ b ∧ b = 30 := by
  sorry

end smallest_multiple_of_6_and_15_l587_58771


namespace tan_sum_reciprocal_l587_58714

theorem tan_sum_reciprocal (u v : ℝ) 
  (h1 : (Real.sin u / Real.cos v) + (Real.sin v / Real.cos u) = 2)
  (h2 : (Real.cos u / Real.sin v) + (Real.cos v / Real.sin u) = 3) :
  (Real.tan u / Real.tan v) + (Real.tan v / Real.tan u) = 3/2 := by
  sorry

end tan_sum_reciprocal_l587_58714


namespace perfect_squares_ending_in_444_and_4444_l587_58774

def ends_in_444 (n : ℕ) : Prop := n % 1000 = 444

def ends_in_4444 (n : ℕ) : Prop := n % 10000 = 4444

theorem perfect_squares_ending_in_444_and_4444 :
  (∀ a : ℕ, (∃ k : ℕ, a * a = k) ∧ ends_in_444 (a * a) ↔ ∃ n : ℕ, a = 500 * n + 38 ∨ a = 500 * n - 38) ∧
  (¬ ∃ a : ℕ, (∃ k : ℕ, a * a = k) ∧ ends_in_4444 (a * a)) :=
by sorry

end perfect_squares_ending_in_444_and_4444_l587_58774


namespace order_of_a_b_c_l587_58715

theorem order_of_a_b_c :
  let a := (2 : ℝ) ^ (9/10)
  let b := (3 : ℝ) ^ (2/3)
  let c := Real.log 3 / Real.log (1/2)
  b > a ∧ a > c := by sorry

end order_of_a_b_c_l587_58715


namespace fruit_sales_problem_l587_58735

/-- Fruit sales problem -/
theorem fruit_sales_problem 
  (ponkan_cost fuji_cost : ℚ)
  (h1 : 30 * ponkan_cost + 20 * fuji_cost = 2700)
  (h2 : 50 * ponkan_cost + 40 * fuji_cost = 4800)
  (ponkan_price fuji_price : ℚ)
  (h3 : ponkan_price = 80)
  (h4 : fuji_price = 60)
  (fuji_price_red1 fuji_price_red2 : ℚ)
  (h5 : fuji_price_red1 = fuji_price * (1 - 1/10))
  (h6 : fuji_price_red2 = fuji_price_red1 * (1 - 1/10))
  (profit : ℚ)
  (h7 : profit = 50 * (ponkan_price - ponkan_cost) + 
                 20 * (fuji_price - fuji_cost) +
                 10 * (fuji_price_red1 - fuji_cost) +
                 10 * (fuji_price_red2 - fuji_cost)) :
  ponkan_cost = 60 ∧ fuji_cost = 45 ∧ profit = 1426 := by
sorry

end fruit_sales_problem_l587_58735


namespace tangent_intersection_y_coordinate_l587_58720

/-- 
Given a parabola y = x^2 + 1 and two points A and B on it with perpendicular tangents,
this theorem states that the y-coordinate of the intersection point P of these tangents is 3/4.
-/
theorem tangent_intersection_y_coordinate (a b : ℝ) : 
  let f : ℝ → ℝ := λ x => x^2 + 1
  let A : ℝ × ℝ := (a, f a)
  let B : ℝ × ℝ := (b, f b)
  let tangent_A : ℝ → ℝ := λ x => 2*a*x - a^2 + 1
  let tangent_B : ℝ → ℝ := λ x => 2*b*x - b^2 + 1
  -- Perpendicularity condition: product of slopes is -1
  2*a * 2*b = -1 →
  -- P is the intersection point of tangents
  let P : ℝ × ℝ := ((a + b) / 2, tangent_A ((a + b) / 2))
  -- The y-coordinate of P is 3/4
  P.2 = 3/4 := by sorry


end tangent_intersection_y_coordinate_l587_58720


namespace special_numbers_count_l587_58724

def count_special_numbers (n : ℕ) : ℕ :=
  (n / 12) - (n / 60)

theorem special_numbers_count :
  count_special_numbers 2017 = 135 := by
  sorry

end special_numbers_count_l587_58724


namespace function_properties_l587_58761

noncomputable def f (x φ : ℝ) : ℝ := Real.cos (2 * x + φ)

theorem function_properties (φ : ℝ) (h : φ > 0) :
  (∀ x, f x φ = f (x + π) φ) ∧ 
  (∃ φ', ∀ x, f x φ' = f (-x) φ') ∧
  (∀ x ∈ Set.Icc (π - φ/2) (3*π/2 - φ/2), ∀ y ∈ Set.Icc (π - φ/2) (3*π/2 - φ/2), 
    x < y → f x φ > f y φ) ∧
  (∀ x, f x φ = Real.cos (2 * (x - φ/2))) :=
by sorry

end function_properties_l587_58761


namespace kates_retirement_fund_l587_58737

/-- Given a retirement fund with an initial value and a decrease amount, 
    calculate the current value of the fund. -/
def current_fund_value (initial_value decrease : ℕ) : ℕ :=
  initial_value - decrease

/-- Theorem: Kate's retirement fund's current value -/
theorem kates_retirement_fund : 
  current_fund_value 1472 12 = 1460 := by
  sorry

end kates_retirement_fund_l587_58737


namespace range_x_when_p_false_range_m_when_p_sufficient_for_q_l587_58712

-- Define propositions p and q
def p (x : ℝ) : Prop := |x - 3| < 1
def q (x m : ℝ) : Prop := m - 2 < x ∧ x < m + 1

-- Part 1: Range of x when p is false
theorem range_x_when_p_false (x : ℝ) :
  ¬(p x) → x ≤ 2 ∨ x ≥ 4 :=
sorry

-- Part 2: Range of m when p is a sufficient condition for q
theorem range_m_when_p_sufficient_for_q (m : ℝ) :
  (∀ x, p x → q x m) → 3 ≤ m ∧ m ≤ 4 :=
sorry

end range_x_when_p_false_range_m_when_p_sufficient_for_q_l587_58712


namespace fraction_multiplication_l587_58765

theorem fraction_multiplication : (2 : ℚ) / 3 * 5 / 7 * 11 / 13 = 110 / 273 := by
  sorry

end fraction_multiplication_l587_58765


namespace field_trip_buses_l587_58730

theorem field_trip_buses (total_people : ℕ) (num_vans : ℕ) (people_per_van : ℕ) (people_per_bus : ℕ) 
  (h1 : total_people = 76)
  (h2 : num_vans = 2)
  (h3 : people_per_van = 8)
  (h4 : people_per_bus = 20) :
  (total_people - num_vans * people_per_van) / people_per_bus = 3 := by
  sorry

end field_trip_buses_l587_58730


namespace four_lattice_points_l587_58738

-- Define the equation
def equation (x y : ℤ) : Prop := x^2 - y^2 = 53

-- Define a lattice point as a pair of integers
def LatticePoint : Type := ℤ × ℤ

-- Define a function to check if a lattice point satisfies the equation
def satisfies_equation (p : LatticePoint) : Prop :=
  equation p.1 p.2

-- Theorem: There are exactly 4 lattice points satisfying the equation
theorem four_lattice_points : 
  ∃! (s : Finset LatticePoint), (∀ p ∈ s, satisfies_equation p) ∧ s.card = 4 :=
sorry

end four_lattice_points_l587_58738


namespace engagement_ring_saving_time_l587_58781

/-- Proves the time required to save for an engagement ring based on annual salary and monthly savings -/
theorem engagement_ring_saving_time 
  (annual_salary : ℕ) 
  (monthly_savings : ℕ) 
  (h1 : annual_salary = 60000)
  (h2 : monthly_savings = 1000) : 
  (2 * (annual_salary / 12)) / monthly_savings = 10 := by
  sorry

end engagement_ring_saving_time_l587_58781


namespace jim_driven_distance_l587_58788

theorem jim_driven_distance (total_journey : ℕ) (remaining : ℕ) (driven : ℕ) : 
  total_journey = 1200 →
  remaining = 432 →
  driven = total_journey - remaining →
  driven = 768 := by
sorry

end jim_driven_distance_l587_58788


namespace unique_solution_for_equation_l587_58741

theorem unique_solution_for_equation : 
  ∀ m n : ℕ+, 1 + 5 * 2^(m : ℕ) = (n : ℕ)^2 ↔ m = 4 ∧ n = 9 := by sorry

end unique_solution_for_equation_l587_58741
