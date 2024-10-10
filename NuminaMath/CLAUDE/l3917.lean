import Mathlib

namespace four_heads_before_three_tails_l3917_391729

/-- The probability of getting heads or tails in a fair coin flip -/
def p_head : ℚ := 1/2
def p_tail : ℚ := 1/2

/-- The probability of encountering 4 heads before 3 tails in repeated fair coin flips -/
noncomputable def q : ℚ := sorry

/-- Theorem stating that q is equal to 28/47 -/
theorem four_heads_before_three_tails : q = 28/47 := by sorry

end four_heads_before_three_tails_l3917_391729


namespace min_distance_line_parabola_l3917_391747

/-- The minimum distance between a point on the line y = (12/5)x - 5 and a point on the parabola y = x^2 is 89/65 -/
theorem min_distance_line_parabola :
  let line := fun (x : ℝ) => (12/5) * x - 5
  let parabola := fun (x : ℝ) => x^2
  let distance := fun (x₁ x₂ : ℝ) => 
    Real.sqrt ((x₂ - x₁)^2 + (parabola x₂ - line x₁)^2)
  (∃ (x₁ x₂ : ℝ), ∀ (y₁ y₂ : ℝ), distance x₁ x₂ ≤ distance y₁ y₂) ∧
  (∃ (x₁ x₂ : ℝ), distance x₁ x₂ = 89/65) := by
  sorry

end min_distance_line_parabola_l3917_391747


namespace mean_equality_problem_l3917_391795

theorem mean_equality_problem (x : ℚ) : 
  (8 + 10 + 22) / 3 = (15 + x) / 2 → x = 35 / 3 :=
by sorry

end mean_equality_problem_l3917_391795


namespace prob_at_least_two_pass_written_is_0_6_expected_students_with_advantage_is_0_96_l3917_391736

-- Define the probabilities for each student passing the written test
def prob_written_A : ℝ := 0.4
def prob_written_B : ℝ := 0.8
def prob_written_C : ℝ := 0.5

-- Define the probabilities for each student passing the interview
def prob_interview_A : ℝ := 0.8
def prob_interview_B : ℝ := 0.4
def prob_interview_C : ℝ := 0.64

-- Function to calculate the probability of at least two students passing the written test
def prob_at_least_two_pass_written : ℝ :=
  prob_written_A * prob_written_B * (1 - prob_written_C) +
  prob_written_A * (1 - prob_written_B) * prob_written_C +
  (1 - prob_written_A) * prob_written_B * prob_written_C +
  prob_written_A * prob_written_B * prob_written_C

-- Function to calculate the probability of a student receiving admission advantage
def prob_admission_advantage (written_prob interview_prob : ℝ) : ℝ :=
  written_prob * interview_prob

-- Function to calculate the mathematical expectation of students receiving admission advantage
def expected_students_with_advantage : ℝ :=
  3 * (prob_admission_advantage prob_written_A prob_interview_A)

-- Theorem statements
theorem prob_at_least_two_pass_written_is_0_6 :
  prob_at_least_two_pass_written = 0.6 := by sorry

theorem expected_students_with_advantage_is_0_96 :
  expected_students_with_advantage = 0.96 := by sorry

end prob_at_least_two_pass_written_is_0_6_expected_students_with_advantage_is_0_96_l3917_391736


namespace function_max_min_implies_m_range_l3917_391751

/-- The function f(x) = x^2 - 2x + 3 on [0, m] with max 3 and min 2 implies m ∈ [1, 2] -/
theorem function_max_min_implies_m_range 
  (f : ℝ → ℝ) 
  (h_f : ∀ x, f x = x^2 - 2*x + 3) 
  (m : ℝ) 
  (h_max : ∃ x ∈ Set.Icc 0 m, ∀ y ∈ Set.Icc 0 m, f y ≤ f x)
  (h_min : ∃ x ∈ Set.Icc 0 m, ∀ y ∈ Set.Icc 0 m, f x ≤ f y)
  (h_max_val : ∃ x ∈ Set.Icc 0 m, f x = 3)
  (h_min_val : ∃ x ∈ Set.Icc 0 m, f x = 2) :
  m ∈ Set.Icc 1 2 :=
sorry

end function_max_min_implies_m_range_l3917_391751


namespace water_volume_ratio_in_cone_l3917_391748

/-- The ratio of the volume of water in a cone filled to 2/3 of its height to the total volume of the cone is 8/27. -/
theorem water_volume_ratio_in_cone (h r : ℝ) (h_pos : h > 0) (r_pos : r > 0) : 
  let water_height := (2 : ℝ) / 3 * h
  let water_radius := (2 : ℝ) / 3 * r
  let cone_volume := (1 : ℝ) / 3 * π * r^2 * h
  let water_volume := (1 : ℝ) / 3 * π * water_radius^2 * water_height
  water_volume / cone_volume = 8 / 27 := by
sorry


end water_volume_ratio_in_cone_l3917_391748


namespace special_sequence_properties_l3917_391761

/-- A sequence satisfying certain conditions -/
structure SpecialSequence where
  a : ℕ → ℝ
  S : ℕ → ℝ
  p : ℝ
  h1 : a 1 = 2
  h2 : ∀ n, a n ≠ 0
  h3 : ∀ n, a n * a (n + 1) = p * S n + 2
  h4 : ∀ n, S (n + 1) = S n + a (n + 1)

/-- The main theorem about the special sequence -/
theorem special_sequence_properties (seq : SpecialSequence) :
  (∀ n, seq.a (n + 2) - seq.a n = seq.p) ∧
  (∃ p : ℝ, p = 2 ∧ 
    (∃ d : ℝ, ∀ n, |seq.a (n + 1)| - |seq.a n| = d)) :=
sorry

end special_sequence_properties_l3917_391761


namespace intersecting_line_slope_angle_l3917_391716

/-- A line passing through (2,0) and intersecting y = √(2-x^2) -/
structure IntersectingLine where
  k : ℝ
  intersects_curve : ∃ (x y : ℝ), y = k * (x - 2) ∧ y = Real.sqrt (2 - x^2)

/-- The area of triangle AOB formed by the intersecting line -/
def triangleArea (l : IntersectingLine) : ℝ := sorry

/-- The slope angle of the line -/
def slopeAngle (l : IntersectingLine) : ℝ := sorry

theorem intersecting_line_slope_angle 
  (l : IntersectingLine) 
  (h : triangleArea l = 1) : 
  slopeAngle l = 150 * π / 180 := by sorry

end intersecting_line_slope_angle_l3917_391716


namespace probability_of_event_B_l3917_391770

theorem probability_of_event_B 
  (A B : Set ℝ) 
  (P : Set ℝ → ℝ) 
  (h1 : P (A ∩ B) = 0.25)
  (h2 : P (A ∪ B) = 0.6)
  (h3 : P A = 0.45) :
  P B = 0.4 := by
sorry

end probability_of_event_B_l3917_391770


namespace profit_without_discount_l3917_391782

/-- Represents the profit percentage and discount percentage as rational numbers -/
def ProfitWithDiscount : ℚ := 44 / 100
def DiscountPercentage : ℚ := 4 / 100

/-- Theorem: If a shopkeeper earns a 44% profit after offering a 4% discount, 
    they would earn a 50% profit without the discount -/
theorem profit_without_discount 
  (cost_price : ℚ) 
  (selling_price : ℚ) 
  (marked_price : ℚ) 
  (h1 : selling_price = cost_price * (1 + ProfitWithDiscount))
  (h2 : selling_price = marked_price * (1 - DiscountPercentage))
  : (marked_price - cost_price) / cost_price = 1 / 2 := by
  sorry


end profit_without_discount_l3917_391782


namespace circle_radius_equation_l3917_391715

/-- The value of d that makes the circle with equation x^2 - 8x + y^2 + 10y + d = 0 have a radius of 5 -/
theorem circle_radius_equation (x y : ℝ) (d : ℝ) : 
  (∀ x y, x^2 - 8*x + y^2 + 10*y + d = 0 ↔ (x - 4)^2 + (y + 5)^2 = 25) → 
  d = 16 := by
  sorry

end circle_radius_equation_l3917_391715


namespace square_with_removed_triangles_l3917_391783

/-- Given a square with side length s, from which two pairs of identical isosceles right triangles
    are removed to form a rectangle, if the total area removed is 180 m², then the diagonal of the
    remaining rectangle is 18 m. -/
theorem square_with_removed_triangles (s : ℝ) (x y : ℝ) : 
  x ≥ 0 → y ≥ 0 → x + y = s → x^2 + y^2 = 180 → 
  Real.sqrt (2 * (x^2 + y^2)) = 18 := by
  sorry

end square_with_removed_triangles_l3917_391783


namespace student_circle_circumference_l3917_391784

/-- The circumference of a circle formed by people standing with overlapping arms -/
def circle_circumference (n : ℕ) (arm_span : ℝ) (overlap : ℝ) : ℝ :=
  n * (arm_span - overlap)

/-- Proof that the circumference of the circle formed by 16 students is 110.4 cm -/
theorem student_circle_circumference :
  circle_circumference 16 10.4 3.5 = 110.4 := by
  sorry

end student_circle_circumference_l3917_391784


namespace alpha_plus_beta_value_l3917_391723

theorem alpha_plus_beta_value (α β : Real) 
  (h1 : α ∈ Set.Ioo 0 (π/4))
  (h2 : β ∈ Set.Ioo 0 (π/4))
  (h3 : α.sin * (3*π/2 + α).cos - (π/2 + α).sin * α.cos = -3/5)
  (h4 : 3 * β.sin = (2*α + β).sin) :
  α + β = π/4 := by sorry

end alpha_plus_beta_value_l3917_391723


namespace correct_observation_value_l3917_391762

theorem correct_observation_value
  (n : ℕ)
  (original_mean : ℚ)
  (incorrect_value : ℚ)
  (corrected_mean : ℚ)
  (h1 : n = 50)
  (h2 : original_mean = 36)
  (h3 : incorrect_value = 23)
  (h4 : corrected_mean = 36.5) :
  (n : ℚ) * corrected_mean - ((n : ℚ) * original_mean - incorrect_value) = 48 :=
by sorry

end correct_observation_value_l3917_391762


namespace twelve_point_polygons_l3917_391766

/-- The number of distinct convex polygons with three or more sides
    that can be formed from 12 points on a circle's circumference. -/
def num_polygons (n : ℕ) : ℕ :=
  2^n - (Nat.choose n 0 + Nat.choose n 1 + Nat.choose n 2)

/-- Theorem stating that the number of distinct convex polygons
    with three or more sides formed from 12 points on a circle
    is equal to 4017. -/
theorem twelve_point_polygons :
  num_polygons 12 = 4017 := by
  sorry

end twelve_point_polygons_l3917_391766


namespace team_selection_theorem_l3917_391754

def internal_medicine_doctors : ℕ := 12
def surgeons : ℕ := 8
def team_size : ℕ := 5

def select_team_with_restrictions (n : ℕ) (k : ℕ) : ℕ := Nat.choose n k

theorem team_selection_theorem :
  (select_team_with_restrictions (internal_medicine_doctors + surgeons - 2) (team_size - 1) = 3060) ∧
  (Nat.choose (internal_medicine_doctors + surgeons) team_size - 
   Nat.choose internal_medicine_doctors team_size - 
   Nat.choose surgeons team_size = 14656) :=
by sorry

end team_selection_theorem_l3917_391754


namespace x_y_relation_existence_of_k_l3917_391727

def x : ℕ → ℤ
  | 0 => 1
  | 1 => 4
  | (n + 2) => 3 * x (n + 1) - x n

def y : ℕ → ℤ
  | 0 => 1
  | 1 => 2
  | (n + 2) => 3 * y (n + 1) - y n

theorem x_y_relation (n : ℕ) : (x n)^2 - 5*(y n)^2 + 4 = 0 := by
  sorry

theorem existence_of_k (a b : ℕ) (h : a^2 - 5*b^2 + 4 = 0) :
  ∃ k : ℕ, x k = a ∧ y k = b := by
  sorry

end x_y_relation_existence_of_k_l3917_391727


namespace largest_x_value_l3917_391725

theorem largest_x_value : 
  let f : ℝ → ℝ := λ x => 7 * (9 * x^2 + 8 * x + 12) - x * (9 * x - 45)
  ∃ (x : ℝ), f x = 0 ∧ ∀ (y : ℝ), f y = 0 → y ≤ x ∧ x = -7/6 :=
by sorry

end largest_x_value_l3917_391725


namespace armands_guessing_game_l3917_391718

theorem armands_guessing_game : ∃ x : ℕ, x = 33 ∧ 3 * x = 2 * 51 - 3 := by
  sorry

end armands_guessing_game_l3917_391718


namespace stock_sale_percentage_l3917_391731

/-- Proves that the percentage of stock sold is 100% given the provided conditions -/
theorem stock_sale_percentage
  (cash_realized : ℝ)
  (brokerage_rate : ℝ)
  (cash_after_brokerage : ℝ)
  (h1 : cash_realized = 109.25)
  (h2 : brokerage_rate = 1 / 400)
  (h3 : cash_after_brokerage = 109)
  (h4 : cash_after_brokerage = cash_realized * (1 - brokerage_rate)) :
  cash_realized / (cash_after_brokerage / (1 - brokerage_rate)) = 1 :=
by sorry

end stock_sale_percentage_l3917_391731


namespace total_study_time_l3917_391764

def study_time (wednesday thursday friday weekend : ℕ) : Prop :=
  (wednesday = 2) ∧
  (thursday = 3 * wednesday) ∧
  (friday = thursday / 2) ∧
  (weekend = wednesday + thursday + friday) ∧
  (wednesday + thursday + friday + weekend = 22)

theorem total_study_time :
  ∃ (wednesday thursday friday weekend : ℕ),
    study_time wednesday thursday friday weekend :=
by sorry

end total_study_time_l3917_391764


namespace a_1_value_c_is_arithmetic_l3917_391735

def sequence_a (n : ℕ) : ℝ := sorry

def sum_S (n : ℕ) : ℝ := sorry

def sequence_c (n : ℕ) : ℝ := sorry

axiom sum_relation (n : ℕ) : sum_S n / 2 = sequence_a n - 2^n

axiom a_relation (n : ℕ) : sequence_a n = 2^n * sequence_c n

theorem a_1_value : sequence_a 1 = 4 := sorry

theorem c_is_arithmetic : ∃ (d : ℝ), ∀ (n : ℕ), n > 0 → sequence_c (n + 1) - sequence_c n = d := sorry

end a_1_value_c_is_arithmetic_l3917_391735


namespace commute_time_difference_l3917_391769

/-- Proves that the difference in commute time between walking and taking the train is 25 minutes -/
theorem commute_time_difference
  (distance : Real)
  (walking_speed : Real)
  (train_speed : Real)
  (additional_train_time : Real)
  (h1 : distance = 1.5)
  (h2 : walking_speed = 3)
  (h3 : train_speed = 20)
  (h4 : additional_train_time = 0.5 / 60) :
  (distance / walking_speed - (distance / train_speed + additional_train_time)) * 60 = 25 := by
  sorry

end commute_time_difference_l3917_391769


namespace track_circumference_l3917_391743

/-- The circumference of a circular track given two people walking in opposite directions -/
theorem track_circumference (v1 v2 : ℝ) (t : ℝ) (h1 : v1 = 20) (h2 : v2 = 13) (h3 : t = 33 / 60) :
  v1 * t + v2 * t = 18.15 := by
  sorry

end track_circumference_l3917_391743


namespace books_for_girls_l3917_391724

theorem books_for_girls (num_girls : ℕ) (num_boys : ℕ) (total_books : ℕ) : 
  num_girls = 15 → 
  num_boys = 10 → 
  total_books = 375 → 
  (num_girls * (total_books / (num_girls + num_boys))) = 225 := by
sorry

end books_for_girls_l3917_391724


namespace average_of_remaining_numbers_l3917_391797

theorem average_of_remaining_numbers
  (total : ℕ)
  (avg_all : ℚ)
  (avg_first_two : ℚ)
  (avg_next_two : ℚ)
  (h_total : total = 6)
  (h_avg_all : avg_all = 5/2)
  (h_avg_first_two : avg_first_two = 11/10)
  (h_avg_next_two : avg_next_two = 14/10) :
  (total * avg_all - 2 * avg_first_two - 2 * avg_next_two) / 2 = 5 := by
sorry

end average_of_remaining_numbers_l3917_391797


namespace min_mozart_bach_not_beethoven_l3917_391740

def Universe := 200
def Mozart := 150
def Bach := 120
def Beethoven := 90

theorem min_mozart_bach_not_beethoven :
  ∃ (m b e mb mbe : ℕ),
    m ≤ Mozart ∧
    b ≤ Bach ∧
    e ≤ Beethoven ∧
    mb ≤ m ∧
    mb ≤ b ∧
    mbe ≤ mb ∧
    mbe ≤ e ∧
    m + b - mb ≤ Universe ∧
    m + b + e - mb - mbe ≤ Universe ∧
    mb - mbe ≥ 10 :=
  sorry

end min_mozart_bach_not_beethoven_l3917_391740


namespace driver_comparison_l3917_391702

theorem driver_comparison (d : ℝ) (h : d > 0) : d / 40 < 8 * d / 315 := by
  sorry

#check driver_comparison

end driver_comparison_l3917_391702


namespace prime_cube_difference_equation_l3917_391728

theorem prime_cube_difference_equation :
  ∃! (p q r : ℕ), 
    Prime p ∧ Prime q ∧ Prime r ∧
    p^3 - q^3 = 11*r ∧
    p = 13 ∧ q = 2 ∧ r = 199 := by
  sorry

end prime_cube_difference_equation_l3917_391728


namespace p_necessary_not_sufficient_l3917_391734

-- Define a triangle
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  sum_180 : A + B + C = 180

-- Define condition p
def condition_p (t : Triangle) : Prop :=
  t.A = 60 ∨ t.B = 60 ∨ t.C = 60

-- Define condition q
def condition_q (t : Triangle) : Prop :=
  t.A - t.B = t.B - t.C

-- Theorem stating p is necessary but not sufficient for q
theorem p_necessary_not_sufficient :
  (∀ t : Triangle, condition_q t → condition_p t) ∧
  ¬(∀ t : Triangle, condition_p t → condition_q t) := by
  sorry


end p_necessary_not_sufficient_l3917_391734


namespace watch_correction_l3917_391717

/-- The number of days from April 1 at 12 noon to April 10 at 6 P.M. -/
def days_passed : ℚ := 9 + 6 / 24

/-- The rate at which the watch loses time, in minutes per day -/
def loss_rate : ℚ := 3

/-- The positive correction in minutes to be added to the watch -/
def correction (d : ℚ) (r : ℚ) : ℚ := d * r

theorem watch_correction :
  correction days_passed loss_rate = 27.75 := by
  sorry

end watch_correction_l3917_391717


namespace four_point_ratio_l3917_391753

/-- Given four distinct points on a plane with segment lengths a, a, a, 2a, 2a, and b,
    prove that the ratio of b to a is 2√2 -/
theorem four_point_ratio (a b : ℝ) (h : a > 0) :
  ∃ (A B C D : ℝ × ℝ),
    A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧
    ({dist A B, dist A C, dist A D, dist B C, dist B D, dist C D} : Finset ℝ) =
      {a, a, a, 2*a, 2*a, b} →
    b / a = 2 * Real.sqrt 2 :=
by sorry

end four_point_ratio_l3917_391753


namespace igloo_construction_l3917_391779

def igloo_bricks (n : ℕ) : ℕ :=
  if n ≤ 6 then
    14 + 2 * (n - 1)
  else
    24 - 3 * (n - 6)

def total_bricks : ℕ := (List.range 10).map (λ i => igloo_bricks (i + 1)) |>.sum

theorem igloo_construction :
  total_bricks = 170 := by
  sorry

end igloo_construction_l3917_391779


namespace solution_set_when_a_is_one_range_of_a_when_solution_set_is_real_l3917_391771

-- Define the inequality function
def f (a : ℝ) (x : ℝ) : ℝ := |a*x - 2| + |a*x - a|

-- Theorem 1: Solution set when a = 1
theorem solution_set_when_a_is_one :
  ∀ x : ℝ, f 1 x ≥ 2 ↔ x ≥ 2.5 ∨ x ≤ 0.5 := by sorry

-- Theorem 2: Range of a when solution set is ℝ
theorem range_of_a_when_solution_set_is_real :
  (∀ a : ℝ, a > 0 → (∀ x : ℝ, f a x ≥ 2)) → (∀ a : ℝ, a > 0 → a ≥ 4) := by sorry

end solution_set_when_a_is_one_range_of_a_when_solution_set_is_real_l3917_391771


namespace quadratic_inequality_problem_l3917_391752

/-- Given that ax^2 + 5x - 2 > 0 has solution set {x | 1/2 < x < 2}, prove:
    1. a = -2
    2. The solution set of ax^2 - 5x + a^2 - 1 > 0 is {x | -3 < x < 1/2} -/
theorem quadratic_inequality_problem 
  (h : ∀ x, ax^2 + 5*x - 2 > 0 ↔ 1/2 < x ∧ x < 2) :
  (a = -2) ∧ 
  (∀ x, a*x^2 - 5*x + a^2 - 1 > 0 ↔ -3 < x ∧ x < 1/2) :=
by sorry

end quadratic_inequality_problem_l3917_391752


namespace area_of_XYZW_l3917_391768

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Calculates the area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℝ := r.width * r.height

/-- Represents the larger rectangle XYZW -/
def XYZW : Rectangle := { width := 14, height := 28 }

/-- Represents one of the smaller identical rectangles -/
def smallRect : Rectangle := { width := 7, height := 14 }

theorem area_of_XYZW :
  XYZW.width = smallRect.height ∧
  XYZW.height = 3 * smallRect.width + smallRect.width ∧
  XYZW.area = 392 := by
  sorry

#check area_of_XYZW

end area_of_XYZW_l3917_391768


namespace pin_permutations_l3917_391741

theorem pin_permutations : 
  let n : ℕ := 4
  ∀ (digits : Finset ℕ), Finset.card digits = n → Finset.card (Finset.powersetCard n digits) = n.factorial :=
by
  sorry

end pin_permutations_l3917_391741


namespace square_of_product_divided_by_square_l3917_391765

theorem square_of_product_divided_by_square (m n : ℝ) :
  (2 * m * n)^2 / n^2 = 4 * m^2 := by
  sorry

end square_of_product_divided_by_square_l3917_391765


namespace race_outcomes_l3917_391790

/-- The number of participants in the race -/
def num_participants : ℕ := 6

/-- The number of top positions we're considering -/
def top_positions : ℕ := 3

/-- Calculate the number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

/-- Calculate the number of permutations of n items -/
def permutations (n : ℕ) : ℕ := Nat.factorial n

/-- The number of different 1st-2nd-3rd place outcomes in a race with 6 participants,
    where one specific participant is guaranteed to be in the top 3 and there are no ties -/
theorem race_outcomes : 
  top_positions * choose (num_participants - 1) (top_positions - 1) * permutations (top_positions - 1) = 60 := by
  sorry

end race_outcomes_l3917_391790


namespace car_average_speed_l3917_391778

/-- Proves that the average speed of a car is 36 km/hr given specific uphill and downhill conditions -/
theorem car_average_speed :
  let uphill_speed : ℝ := 30
  let downhill_speed : ℝ := 60
  let uphill_distance : ℝ := 100
  let downhill_distance : ℝ := 50
  let total_distance : ℝ := uphill_distance + downhill_distance
  let uphill_time : ℝ := uphill_distance / uphill_speed
  let downhill_time : ℝ := downhill_distance / downhill_speed
  let total_time : ℝ := uphill_time + downhill_time
  let average_speed : ℝ := total_distance / total_time
  average_speed = 36 := by
sorry

end car_average_speed_l3917_391778


namespace geometric_series_equality_l3917_391793

def C (n : ℕ) : ℚ := (1024 / 3) * (1 - 1 / (4 ^ n))

def D (n : ℕ) : ℚ := (2048 / 3) * (1 - 1 / ((-2) ^ n))

theorem geometric_series_equality (n : ℕ) (h : n ≥ 1) : 
  (C n = D n) ↔ n = 1 :=
sorry

end geometric_series_equality_l3917_391793


namespace nikolai_wins_l3917_391745

/-- Represents a mountain goat with its jump distance and number of jumps per unit time -/
structure Goat where
  name : String
  jumpDistance : ℕ
  jumpsPerUnitTime : ℕ

/-- Calculates the distance covered by a goat in one unit of time -/
def distancePerUnitTime (g : Goat) : ℕ :=
  g.jumpDistance * g.jumpsPerUnitTime

/-- Calculates the number of jumps needed to cover a given distance -/
def jumpsNeeded (g : Goat) (distance : ℕ) : ℕ :=
  (distance + g.jumpDistance - 1) / g.jumpDistance

/-- The theorem stating that Nikolai completes the journey faster -/
theorem nikolai_wins (gennady nikolai : Goat) (totalDistance : ℕ) : 
  gennady.name = "Gennady" →
  nikolai.name = "Nikolai" →
  gennady.jumpDistance = 6 →
  gennady.jumpsPerUnitTime = 2 →
  nikolai.jumpDistance = 4 →
  nikolai.jumpsPerUnitTime = 3 →
  totalDistance = 2000 →
  distancePerUnitTime gennady = distancePerUnitTime nikolai →
  jumpsNeeded nikolai totalDistance < jumpsNeeded gennady totalDistance :=
by sorry

#check nikolai_wins

end nikolai_wins_l3917_391745


namespace mark_young_fish_count_l3917_391760

/-- The number of tanks Mark has for pregnant fish -/
def num_tanks : ℕ := 3

/-- The number of pregnant fish in each tank -/
def fish_per_tank : ℕ := 4

/-- The number of young fish each pregnant fish gives birth to -/
def young_per_fish : ℕ := 20

/-- The total number of young fish Mark has at the end -/
def total_young_fish : ℕ := num_tanks * fish_per_tank * young_per_fish

theorem mark_young_fish_count : total_young_fish = 240 := by
  sorry

end mark_young_fish_count_l3917_391760


namespace x_squared_eq_5_is_quadratic_l3917_391757

/-- Definition of a quadratic equation -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The equation x^2 = 5 -/
def f (x : ℝ) : ℝ := x^2 - 5

theorem x_squared_eq_5_is_quadratic : is_quadratic_equation f := by
  sorry


end x_squared_eq_5_is_quadratic_l3917_391757


namespace equal_savings_time_l3917_391707

def sara_initial_savings : ℕ := 4100
def sara_weekly_savings : ℕ := 10
def jim_weekly_savings : ℕ := 15

theorem equal_savings_time : 
  ∃ w : ℕ, w = 820 ∧ 
  sara_initial_savings + sara_weekly_savings * w = jim_weekly_savings * w :=
by sorry

end equal_savings_time_l3917_391707


namespace circle_area_radius_decrease_l3917_391714

theorem circle_area_radius_decrease (r : ℝ) (h : r > 0) :
  let A := π * r^2
  let A' := 0.64 * A
  let r' := Real.sqrt (A' / π)
  r' / r = 0.8 := by sorry

end circle_area_radius_decrease_l3917_391714


namespace polynomial_value_l3917_391726

/-- A polynomial of degree 5 with integer coefficients -/
def polynomial (a₁ a₂ a₃ a₄ a₅ : ℤ) (x : ℝ) : ℝ :=
  x^5 + a₁*x^4 + a₂*x^3 + a₃*x^2 + a₄*x + a₅

theorem polynomial_value (a₁ a₂ a₃ a₄ a₅ : ℤ) :
  let f := polynomial a₁ a₂ a₃ a₄ a₅
  (f (Real.sqrt 3 + Real.sqrt 2) = 0) →
  (f 1 + f 3 = 0) →
  (f (-1) = 24) := by
  sorry

end polynomial_value_l3917_391726


namespace right_triangle_with_hypotenuse_65_l3917_391750

theorem right_triangle_with_hypotenuse_65 :
  ∃ (a b : ℕ), 
    a < b ∧ 
    a^2 + b^2 = 65^2 ∧ 
    a = 16 := by
  sorry

end right_triangle_with_hypotenuse_65_l3917_391750


namespace fuel_cost_savings_l3917_391792

theorem fuel_cost_savings (old_efficiency : ℝ) (old_fuel_cost : ℝ) 
  (h1 : old_efficiency > 0) (h2 : old_fuel_cost > 0) : 
  let new_efficiency := 1.5 * old_efficiency
  let new_fuel_cost := 1.2 * old_fuel_cost
  let old_trip_cost := (1 / old_efficiency) * old_fuel_cost
  let new_trip_cost := (1 / new_efficiency) * new_fuel_cost
  (old_trip_cost - new_trip_cost) / old_trip_cost = 0.2 := by
  sorry

end fuel_cost_savings_l3917_391792


namespace triangle_side_length_l3917_391712

theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) : 
  2 * b = a + c →
  B = π / 6 →
  (1 / 2) * a * c * Real.sin B = 3 / 2 →
  b = 1 + Real.sqrt 3 := by
  sorry

end triangle_side_length_l3917_391712


namespace quadratic_roots_property_l3917_391701

theorem quadratic_roots_property : ∀ x₁ x₂ : ℝ, 
  x₁^2 - 3*x₁ - 5 = 0 →
  x₂^2 - 3*x₂ - 5 = 0 →
  x₁ + x₂ - x₁ * x₂ = 8 := by
  sorry

end quadratic_roots_property_l3917_391701


namespace probability_at_least_one_correct_l3917_391737

theorem probability_at_least_one_correct (p : ℝ) (h1 : p = 1/2) :
  1 - (1 - p)^3 = 7/8 := by
  sorry

end probability_at_least_one_correct_l3917_391737


namespace chocolate_count_l3917_391739

/-- The number of large boxes in the massive crate -/
def large_boxes : ℕ := 54

/-- The number of small boxes in each large box -/
def small_boxes_per_large : ℕ := 24

/-- The number of chocolate bars in each small box -/
def chocolates_per_small : ℕ := 37

/-- The total number of chocolate bars in the massive crate -/
def total_chocolates : ℕ := large_boxes * small_boxes_per_large * chocolates_per_small

theorem chocolate_count : total_chocolates = 47952 := by
  sorry

end chocolate_count_l3917_391739


namespace inequality_proof_l3917_391742

theorem inequality_proof (n : ℕ) (hn : n > 0) :
  (2 * n^2 + 3 * n + 1)^n ≥ 6^n * (n!)^2 :=
by sorry

end inequality_proof_l3917_391742


namespace base_8_first_digit_of_395_l3917_391732

def base_8_first_digit (n : ℕ) : ℕ :=
  if n = 0 then 0
  else
    let p := Nat.log 8 n
    (n / 8^p) % 8

theorem base_8_first_digit_of_395 :
  base_8_first_digit 395 = 6 := by
sorry

end base_8_first_digit_of_395_l3917_391732


namespace candy_packing_problem_l3917_391773

theorem candy_packing_problem (a : ℕ) : 
  (a % 10 = 6) ∧ 
  (a % 15 = 11) ∧ 
  (200 ≤ a) ∧ 
  (a ≤ 250) ↔ 
  (a = 206 ∨ a = 236) :=
sorry

end candy_packing_problem_l3917_391773


namespace fraction_numerator_l3917_391776

theorem fraction_numerator (y : ℝ) (x : ℝ) (h1 : y > 0) 
  (h2 : y / 20 + x = 0.35 * y) : x = 3 := by
  sorry

end fraction_numerator_l3917_391776


namespace lawnmower_depreciation_l3917_391700

theorem lawnmower_depreciation (initial_value : ℝ) (first_depreciation_rate : ℝ) (second_depreciation_rate : ℝ) :
  initial_value = 100 →
  first_depreciation_rate = 0.25 →
  second_depreciation_rate = 0.20 →
  initial_value * (1 - first_depreciation_rate) * (1 - second_depreciation_rate) = 60 := by
sorry

end lawnmower_depreciation_l3917_391700


namespace matrix_determinant_l3917_391781

def matrix : Matrix (Fin 3) (Fin 3) ℤ :=
  ![![1, -3, 3],
    ![0,  5, -1],
    ![4, -2, 1]]

theorem matrix_determinant :
  Matrix.det matrix = -45 := by sorry

end matrix_determinant_l3917_391781


namespace negative_one_odd_power_l3917_391785

theorem negative_one_odd_power (n : ℕ) (h : Odd n) : (-1 : ℤ) ^ n = -1 := by
  sorry

end negative_one_odd_power_l3917_391785


namespace min_distance_circle_to_line_l3917_391794

/-- The minimum distance from a point on the circle ρ = 2 to the line ρ(cos(θ) + √3 sin(θ)) = 6 is 1 -/
theorem min_distance_circle_to_line : 
  let circle := {p : ℝ × ℝ | p.1^2 + p.2^2 = 4}
  let line := {p : ℝ × ℝ | p.1 + Real.sqrt 3 * p.2 = 6}
  ∃ (d : ℝ), d = 1 ∧ ∀ (p : ℝ × ℝ), p ∈ circle → 
    (∀ (q : ℝ × ℝ), q ∈ line → d ≤ Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)) :=
by sorry

end min_distance_circle_to_line_l3917_391794


namespace probability_three_primes_six_dice_l3917_391791

-- Define a 12-sided die
def die := Finset.range 12

-- Define prime numbers on a 12-sided die
def primes : Finset ℕ := {2, 3, 5, 7, 11}

-- Define the probability of rolling a prime number on one die
def prob_prime : ℚ := (primes.card : ℚ) / (die.card : ℚ)

-- Define the probability of not rolling a prime number on one die
def prob_not_prime : ℚ := 1 - prob_prime

-- Define the number of ways to choose 3 dice from 6
def choose_3_from_6 : ℕ := Nat.choose 6 3

-- Statement of the theorem
theorem probability_three_primes_six_dice :
  (choose_3_from_6 : ℚ) * prob_prime^3 * prob_not_prime^3 = 857500 / 2985984 := by
  sorry

end probability_three_primes_six_dice_l3917_391791


namespace sum_of_permutations_unique_l3917_391733

/-- Represents a positive integer with at least two digits as a list of its digits. -/
def PositiveInteger := {l : List Nat // l.length ≥ 2 ∧ l.head! ≠ 0}

/-- Calculates the sum of all permutations of a number's digits, excluding the original number. -/
def sumOfPermutations (n : PositiveInteger) : Nat :=
  sorry

/-- Theorem stating that the sum of permutations is unique for each number. -/
theorem sum_of_permutations_unique (x y : PositiveInteger) :
  x ≠ y → sumOfPermutations x ≠ sumOfPermutations y := by
  sorry

end sum_of_permutations_unique_l3917_391733


namespace marble_ratio_l3917_391786

theorem marble_ratio (total : ℕ) (red : ℕ) (dark_blue : ℕ) 
  (h1 : total = 63) 
  (h2 : red = 38) 
  (h3 : dark_blue = 6) :
  (total - red - dark_blue) / red = 1 / 2 := by
  sorry

end marble_ratio_l3917_391786


namespace geometric_sequence_identity_l3917_391709

/-- 
Given a geometric sequence and three of its terms L, M, N at positions l, m, n respectively,
prove that L^(m-n) * M^(n-l) * N^(l-m) = 1.
-/
theorem geometric_sequence_identity 
  {α : Type*} [Field α] 
  (a : ℕ → α) 
  (q : α) 
  (h_geom : ∀ n, a (n + 1) = a n * q) 
  (l m n : ℕ) :
  (a l) ^ (m - n) * (a m) ^ (n - l) * (a n) ^ (l - m) = 1 :=
sorry

end geometric_sequence_identity_l3917_391709


namespace parallel_line_y_intercept_l3917_391787

/-- A line parallel to y = -3x + 6 passing through (3, -2) has y-intercept 7 -/
theorem parallel_line_y_intercept :
  ∀ (b : ℝ → ℝ),
  (∀ x y, b y = -3 * x + b 0) →  -- b is a linear function with slope -3
  b (-2) = 3 →                   -- b passes through (3, -2)
  b 0 = 7 :=                     -- y-intercept of b is 7
by
  sorry

end parallel_line_y_intercept_l3917_391787


namespace correct_stratified_sample_l3917_391799

/-- Represents the number of students in each grade -/
structure GradePopulation where
  first : ℕ
  second : ℕ
  third : ℕ

/-- Represents the number of students to be sampled from each grade -/
structure SampleSize where
  first : ℕ
  second : ℕ
  third : ℕ

/-- Calculates the stratified sample size for each grade -/
def stratifiedSample (pop : GradePopulation) (totalSample : ℕ) : SampleSize :=
  let totalPop := pop.first + pop.second + pop.third
  { first := (totalSample * pop.first + totalPop - 1) / totalPop,
    second := (totalSample * pop.second + totalPop - 1) / totalPop,
    third := (totalSample * pop.third + totalPop - 1) / totalPop }

theorem correct_stratified_sample :
  let pop := GradePopulation.mk 600 680 720
  let sample := stratifiedSample pop 50
  sample.first = 15 ∧ sample.second = 17 ∧ sample.third = 18 := by
  sorry


end correct_stratified_sample_l3917_391799


namespace log_function_passes_through_point_l3917_391767

theorem log_function_passes_through_point 
  (a : ℝ) (ha : a > 0 ∧ a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ Real.log x / Real.log a - 2
  f 1 = -2 := by
  sorry

end log_function_passes_through_point_l3917_391767


namespace sum_range_l3917_391789

theorem sum_range (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : a + b + 1/a + 9/b = 10) : 2 ≤ a + b ∧ a + b ≤ 8 := by
  sorry

end sum_range_l3917_391789


namespace product_of_three_numbers_l3917_391780

theorem product_of_three_numbers (a b c : ℝ) : 
  a + b + c = 300 ∧ 
  5 * a = c - 14 ∧ 
  5 * a = b + 14 → 
  a * b * c = 664500 := by
sorry

end product_of_three_numbers_l3917_391780


namespace divisibility_by_17_l3917_391759

theorem divisibility_by_17 (a b : ℤ) : 
  let x : ℤ := 3 * b - 5 * a
  let y : ℤ := 9 * a - 2 * b
  (17 ∣ (2 * x + 3 * y)) ∧ (17 ∣ (9 * x + 5 * y)) := by
  sorry

end divisibility_by_17_l3917_391759


namespace sum_of_reciprocals_l3917_391711

/-- Given two positive real numbers with sum 55, HCF 5, and LCM 120, 
    prove that the sum of their reciprocals is 11/120 -/
theorem sum_of_reciprocals (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (sum : a + b = 55) (hcf : Int.gcd (Int.floor a) (Int.floor b) = 5) 
  (lcm : Int.lcm (Int.floor a) (Int.floor b) = 120) : 
  1 / a + 1 / b = 11 / 120 := by
  sorry

end sum_of_reciprocals_l3917_391711


namespace fraction_difference_l3917_391706

theorem fraction_difference (a b : ℝ) (h : a - b = 2 * a * b) : 1 / a - 1 / b = -2 := by
  sorry

end fraction_difference_l3917_391706


namespace complement_of_A_in_U_l3917_391755

universe u

def U : Finset ℕ := {1, 2, 3, 4, 5, 6, 7}
def A : Finset ℕ := {2, 4, 5}

theorem complement_of_A_in_U :
  (U \ A : Finset ℕ) = {1, 3, 6, 7} := by sorry

end complement_of_A_in_U_l3917_391755


namespace number_problem_l3917_391758

theorem number_problem : ∃ x : ℝ, (x / 5 - 5 = 5) ∧ (x = 50) := by
  sorry

end number_problem_l3917_391758


namespace num_sequences_mod_1000_l3917_391710

/-- The number of increasing sequences of positive integers satisfying the given conditions -/
def num_sequences : ℕ := sorry

/-- The upper bound for the sequence elements -/
def upper_bound : ℕ := 1007

/-- The length of the sequences -/
def sequence_length : ℕ := 12

/-- Predicate to check if a sequence satisfies the given conditions -/
def valid_sequence (b : Fin sequence_length → ℕ) : Prop :=
  (∀ i j : Fin sequence_length, i ≤ j → b i ≤ b j) ∧
  (∀ i : Fin sequence_length, b i ≤ upper_bound) ∧
  (∀ i : Fin sequence_length, Even (b i - i.val))

theorem num_sequences_mod_1000 :
  num_sequences % 1000 = 508 := by sorry

end num_sequences_mod_1000_l3917_391710


namespace quadratic_lower_bound_l3917_391704

theorem quadratic_lower_bound 
  (f : ℝ → ℝ) 
  (a b : ℤ) 
  (h1 : ∀ x, f x = x^2 + a*x + b) 
  (h2 : ∀ x, f x ≥ -9/10) : 
  ∀ x, f x ≥ -1/4 := by
  sorry

end quadratic_lower_bound_l3917_391704


namespace same_color_probability_l3917_391721

/-- Represents the number of jelly beans of each color for a person -/
structure JellyBeans where
  blue : ℕ
  green : ℕ
  yellow : ℕ
  red : ℕ

/-- Calculates the total number of jelly beans a person has -/
def JellyBeans.total (jb : JellyBeans) : ℕ :=
  jb.blue + jb.green + jb.yellow + jb.red

/-- Represents the jelly bean distribution for each person -/
def abe : JellyBeans := { blue := 2, green := 1, yellow := 0, red := 0 }
def bob : JellyBeans := { blue := 1, green := 2, yellow := 1, red := 0 }
def cara : JellyBeans := { blue := 3, green := 2, yellow := 0, red := 1 }

/-- Calculates the probability of picking a specific color for a person -/
def prob_pick_color (jb : JellyBeans) (color : ℕ) : ℚ :=
  color / jb.total

/-- Theorem: The probability of all three people picking jelly beans of the same color is 5/36 -/
theorem same_color_probability :
  (prob_pick_color abe abe.blue * prob_pick_color bob bob.blue * prob_pick_color cara cara.blue) +
  (prob_pick_color abe abe.green * prob_pick_color bob bob.green * prob_pick_color cara cara.green) =
  5 / 36 := by
  sorry

end same_color_probability_l3917_391721


namespace quadratic_equation_solutions_quartic_equation_solutions_l3917_391738

theorem quadratic_equation_solutions :
  let f : ℝ → ℝ := λ x ↦ 2*x^2 + 4*x - 1
  ∃ x₁ x₂ : ℝ, x₁ = -1 - Real.sqrt 6 / 2 ∧ 
             x₂ = -1 + Real.sqrt 6 / 2 ∧ 
             f x₁ = 0 ∧ f x₂ = 0 :=
by sorry

theorem quartic_equation_solutions :
  let g : ℝ → ℝ := λ x ↦ 4*(2*x - 1)^2 - 9*(x + 4)^2
  ∃ x₁ x₂ : ℝ, x₁ = -8/11 ∧ 
             x₂ = 16/5 ∧ 
             g x₁ = 0 ∧ g x₂ = 0 :=
by sorry

end quadratic_equation_solutions_quartic_equation_solutions_l3917_391738


namespace interchangeable_statements_l3917_391788

-- Define the concept of geometric objects
inductive GeometricObject
| Line
| Plane

-- Define the relationships between geometric objects
inductive Relationship
| Perpendicular
| Parallel

-- Define a geometric statement
structure GeometricStatement where
  obj1 : GeometricObject
  obj2 : GeometricObject
  rel1 : Relationship
  obj3 : GeometricObject
  rel2 : Relationship

-- Define the concept of an interchangeable statement
def isInterchangeable (s : GeometricStatement) : Prop :=
  (s.obj1 = GeometricObject.Line ∧ s.obj2 = GeometricObject.Plane) ∨
  (s.obj1 = GeometricObject.Plane ∧ s.obj2 = GeometricObject.Line)

-- Define the four statements
def statement1 : GeometricStatement :=
  { obj1 := GeometricObject.Line
  , obj2 := GeometricObject.Line
  , rel1 := Relationship.Perpendicular
  , obj3 := GeometricObject.Plane
  , rel2 := Relationship.Parallel }

def statement2 : GeometricStatement :=
  { obj1 := GeometricObject.Plane
  , obj2 := GeometricObject.Plane
  , rel1 := Relationship.Perpendicular
  , obj3 := GeometricObject.Plane
  , rel2 := Relationship.Parallel }

def statement3 : GeometricStatement :=
  { obj1 := GeometricObject.Line
  , obj2 := GeometricObject.Line
  , rel1 := Relationship.Parallel
  , obj3 := GeometricObject.Line
  , rel2 := Relationship.Parallel }

def statement4 : GeometricStatement :=
  { obj1 := GeometricObject.Line
  , obj2 := GeometricObject.Line
  , rel1 := Relationship.Parallel
  , obj3 := GeometricObject.Plane
  , rel2 := Relationship.Parallel }

-- Theorem to prove
theorem interchangeable_statements :
  isInterchangeable statement1 ∧ isInterchangeable statement3 ∧
  ¬isInterchangeable statement2 ∧ ¬isInterchangeable statement4 :=
sorry

end interchangeable_statements_l3917_391788


namespace f_is_even_l3917_391772

-- Define an even function
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

-- Define the function f
def f (h : ℝ → ℝ) (x : ℝ) : ℝ := |h (x^5)|

-- Theorem statement
theorem f_is_even (h : ℝ → ℝ) (h_even : IsEven h) : IsEven (f h) := by
  sorry

end f_is_even_l3917_391772


namespace perfect_square_trinomial_expansion_l3917_391777

theorem perfect_square_trinomial_expansion (x : ℝ) : 
  let a : ℝ := x
  let b : ℝ := (1 : ℝ) / 2
  2 * a * b = x := by sorry

end perfect_square_trinomial_expansion_l3917_391777


namespace largest_solution_and_fraction_l3917_391703

theorem largest_solution_and_fraction (a b c d : ℤ) : 
  (∃ x : ℚ, (5 * x) / 6 + 1 = 3 / x ∧ 
             x = (a + b * Real.sqrt c) / d ∧ 
             ∀ y : ℚ, (5 * y) / 6 + 1 = 3 / y → y ≤ x) →
  a * c * d / b = -55 := by
  sorry

end largest_solution_and_fraction_l3917_391703


namespace sine_rule_application_l3917_391775

theorem sine_rule_application (A B C : ℝ) (a b : ℝ) :
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π →
  A + B + C = π →
  a > 0 ∧ b > 0 →
  a = 3 * b * Real.sin A →
  Real.sin B = 1 / 3 := by
sorry

end sine_rule_application_l3917_391775


namespace triangle_vector_property_l3917_391744

theorem triangle_vector_property (A B C : ℝ) (hAcute : 0 < A ∧ A < π/2) 
    (hBcute : 0 < B ∧ B < π/2) (hCcute : 0 < C ∧ C < π/2) 
    (hSum : A + B + C = π) :
  let a : ℝ × ℝ := (Real.sin C + Real.cos C, 2 - 2 * Real.sin C)
  let b : ℝ × ℝ := (1 + Real.sin C, Real.sin C - Real.cos C)
  (a.1 * b.1 + a.2 * b.2 = 0) → 
  2 * Real.sin A ^ 2 + Real.cos B = 1 := by
sorry

end triangle_vector_property_l3917_391744


namespace negation_of_statement_l3917_391798

theorem negation_of_statement :
  (¬ ∀ x : ℝ, x > 0 → x - Real.log x > 0) ↔ (∃ x₀ : ℝ, x₀ > 0 ∧ x₀ - Real.log x₀ ≤ 0) := by
  sorry

end negation_of_statement_l3917_391798


namespace fraction_calculation_l3917_391756

theorem fraction_calculation : 
  (2 * 4.6 * 9 + 4 * 9.2 * 18) / (1 * 2.3 * 4.5 + 3 * 6.9 * 13.5) = 18/7 := by
  sorry

end fraction_calculation_l3917_391756


namespace total_marbles_relation_l3917_391774

/-- Represents the number of marbles of each color -/
structure MarbleCollection where
  red : ℝ
  blue : ℝ
  green : ℝ

/-- Conditions for the marble collection -/
def validCollection (c : MarbleCollection) : Prop :=
  c.red = 1.4 * c.blue ∧ c.green = 1.5 * c.red

/-- Total number of marbles in the collection -/
def totalMarbles (c : MarbleCollection) : ℝ :=
  c.red + c.blue + c.green

/-- Theorem stating the relationship between total marbles and red marbles -/
theorem total_marbles_relation (c : MarbleCollection) (h : validCollection c) :
    totalMarbles c = 3.21 * c.red := by
  sorry

#check total_marbles_relation

end total_marbles_relation_l3917_391774


namespace intersection_complement_equal_l3917_391730

def A : Set ℝ := {-3, -1, 1, 3}
def B : Set ℝ := {x : ℝ | x^2 + 2*x - 3 = 0}

theorem intersection_complement_equal : A ∩ (Set.univ \ B) = {-1, 3} := by
  sorry

end intersection_complement_equal_l3917_391730


namespace gretchen_earnings_l3917_391746

/-- Gretchen's earnings from drawing caricatures over a weekend -/
def weekend_earnings (price_per_drawing : ℕ) (saturday_sales : ℕ) (sunday_sales : ℕ) : ℕ :=
  price_per_drawing * (saturday_sales + sunday_sales)

/-- Theorem stating Gretchen's earnings for the given weekend -/
theorem gretchen_earnings :
  weekend_earnings 20 24 16 = 800 := by
  sorry

end gretchen_earnings_l3917_391746


namespace max_rounds_four_teams_one_match_l3917_391720

/-- Represents a round-robin tournament with 18 teams -/
structure Tournament :=
  (teams : Finset (Fin 18))
  (rounds : Fin 17 → Finset (Fin 18 × Fin 18))
  (round_valid : ∀ r, (rounds r).card = 9)
  (round_pairs : ∀ r t, (t ∈ teams) → (∃! u, (t, u) ∈ rounds r ∨ (u, t) ∈ rounds r))
  (all_play_all : ∀ t u, t ≠ u → (∃! r, (t, u) ∈ rounds r ∨ (u, t) ∈ rounds r))

/-- The property that there exist 4 teams with exactly 1 match played among them -/
def has_four_teams_one_match (T : Tournament) (n : ℕ) : Prop :=
  ∃ (a b c d : Fin 18), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    (∃! (i j : Fin 18) (r : Fin n), 
      ((i = a ∧ j = b) ∨ (i = a ∧ j = c) ∨ (i = a ∧ j = d) ∨
       (i = b ∧ j = c) ∨ (i = b ∧ j = d) ∨ (i = c ∧ j = d)) ∧
      ((i, j) ∈ T.rounds r ∨ (j, i) ∈ T.rounds r))

/-- The main theorem statement -/
theorem max_rounds_four_teams_one_match (T : Tournament) :
  (∀ n ≤ 7, has_four_teams_one_match T n) ∧
  (∃ n > 7, ¬has_four_teams_one_match T n) :=
sorry

end max_rounds_four_teams_one_match_l3917_391720


namespace quadratic_factorization_l3917_391705

theorem quadratic_factorization (y a b : ℤ) : 
  (3 * y^2 - 7 * y - 6 = (3 * y + a) * (y + b)) → (a - b = 5) := by
sorry

end quadratic_factorization_l3917_391705


namespace prize_winning_beverage_probabilities_l3917_391719

/-- The probability of success for each independent event -/
def p : ℚ := 1 / 6

/-- The probability of failure for each independent event -/
def q : ℚ := 1 - p

theorem prize_winning_beverage_probabilities :
  let prob_all_fail := q ^ 3
  let prob_at_least_two_fail := 1 - (3 * p^2 * q + p^3)
  (prob_all_fail = 125 / 216) ∧ (prob_at_least_two_fail = 25 / 27) := by
  sorry

end prize_winning_beverage_probabilities_l3917_391719


namespace sum_of_abs_values_l3917_391749

theorem sum_of_abs_values (a b : ℝ) (ha : |a| = 4) (hb : |b| = 5) :
  (a + b = 9) ∨ (a + b = -9) ∨ (a + b = 1) ∨ (a + b = -1) := by
  sorry

end sum_of_abs_values_l3917_391749


namespace solution_system_equations_l3917_391722

theorem solution_system_equations (x y : ℝ) :
  x ≠ 0 ∧
  |y - x| - |x| / x + 1 = 0 ∧
  |2 * x - y| + |x + y - 1| + |x - y| + y - 1 = 0 →
  y = x ∧ 0 < x ∧ x ≤ 0.5 :=
by sorry

end solution_system_equations_l3917_391722


namespace remainder_problem_l3917_391796

theorem remainder_problem (n : ℕ) 
  (h1 : n^2 % 7 = 1) 
  (h2 : n^3 % 7 = 6) : 
  n % 7 = 6 := by
sorry

end remainder_problem_l3917_391796


namespace min_value_of_sum_l3917_391708

theorem min_value_of_sum (m n : ℝ) (hm : m > 0) (hn : n > 0) (h : 1/m + 2/n = 1) :
  m + n ≥ (Real.sqrt 2 + 1)^2 :=
sorry

end min_value_of_sum_l3917_391708


namespace calculation_proof_l3917_391713

theorem calculation_proof :
  (1/5 - 2/3 - 3/10) * (-60) = 46 ∧
  (-1)^2024 + 24 / (-2)^3 - 15^2 * (1/15)^2 = -3 :=
by sorry

end calculation_proof_l3917_391713


namespace hiking_team_participants_l3917_391763

/-- The number of gloves needed for the hiking team -/
def total_gloves : ℕ := 126

/-- The number of gloves each participant needs -/
def gloves_per_participant : ℕ := 2

/-- The number of participants in the hiking team -/
def num_participants : ℕ := total_gloves / gloves_per_participant

theorem hiking_team_participants : num_participants = 63 := by
  sorry

end hiking_team_participants_l3917_391763
