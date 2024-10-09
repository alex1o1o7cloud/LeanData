import Mathlib

namespace larger_integer_value_l930_93062

theorem larger_integer_value (a b : ℕ) (h₁ : a / b = 7 / 3) (h₂ : a * b = 189) : max a b = 21 :=
sorry

end larger_integer_value_l930_93062


namespace tetrahedron_volume_l930_93030

theorem tetrahedron_volume (a b c : ℝ)
  (h₁ : a + b > c) (h₂ : a + c > b) (h₃ : b + c > a) :
  ∃ V : ℝ, 
    V = (1 / (6 * Real.sqrt 2)) * 
        Real.sqrt ((a^2 + b^2 - c^2) * (a^2 + c^2 - b^2) * (b^2 + c^2 - a^2)) :=
sorry

end tetrahedron_volume_l930_93030


namespace maximum_elevation_l930_93086

-- Define the elevation function
def elevation (t : ℝ) : ℝ := 200 * t - 17 * t^2 - 3 * t^3

-- State that the maximum elevation is 368.1 feet
theorem maximum_elevation :
  ∃ t : ℝ, t > 0 ∧ (∀ t' : ℝ, t' ≠ t → elevation t ≤ elevation t') ∧ elevation t = 368.1 :=
by
  sorry

end maximum_elevation_l930_93086


namespace cost_of_one_box_of_paper_clips_l930_93025

theorem cost_of_one_box_of_paper_clips (p i : ℝ) 
  (h1 : 15 * p + 7 * i = 55.40) 
  (h2 : 12 * p + 10 * i = 61.70) : 
  p = 1.835 := 
by 
  sorry

end cost_of_one_box_of_paper_clips_l930_93025


namespace A_investment_l930_93068

variable (x : ℕ)
variable (A_share : ℕ := 3780)
variable (Total_profit : ℕ := 12600)
variable (B_invest : ℕ := 4200)
variable (C_invest : ℕ := 10500)

theorem A_investment :
  (A_share : ℝ) / (Total_profit : ℝ) = (x : ℝ) / (x + B_invest + C_invest) →
  x = 6300 :=
by
  sorry

end A_investment_l930_93068


namespace balance_three_diamonds_l930_93024

-- Define the problem conditions
variables (a b c : ℕ)

-- Four Δ's and two ♦'s will balance twelve ●'s
def condition1 : Prop :=
  4 * a + 2 * b = 12 * c

-- One Δ will balance a ♦ and two ●'s
def condition2 : Prop :=
  a = b + 2 * c

-- Theorem to prove how many ●'s will balance three ♦'s
theorem balance_three_diamonds (h1 : condition1 a b c) (h2 : condition2 a b c) : 3 * b = 2 * c :=
by sorry

end balance_three_diamonds_l930_93024


namespace luke_total_score_l930_93041

theorem luke_total_score (points_per_round : ℕ) (number_of_rounds : ℕ) (total_score : ℕ) : 
  points_per_round = 146 ∧ number_of_rounds = 157 ∧ total_score = points_per_round * number_of_rounds → 
  total_score = 22822 := by 
  sorry

end luke_total_score_l930_93041


namespace arithmetic_sequence_a2015_l930_93047

theorem arithmetic_sequence_a2015 :
  ∀ {a : ℕ → ℤ}, (a 1 = 2 ∧ a 5 = 6 ∧ (∀ n, a (n + 1) = a n + a 2 - a 1)) → a 2015 = 2016 :=
by
  sorry

end arithmetic_sequence_a2015_l930_93047


namespace solve_for_y_l930_93095

theorem solve_for_y (y : ℝ) : y^2 - 6 * y + 5 = 0 ↔ y = 1 ∨ y = 5 :=
by
  sorry

end solve_for_y_l930_93095


namespace line_form_x_eq_ky_add_b_perpendicular_y_line_form_x_eq_ky_add_b_perpendicular_x_l930_93015

theorem line_form_x_eq_ky_add_b_perpendicular_y {k b : ℝ} : 
  ¬ ∃ c : ℝ, x = c ∧ ∀ y : ℝ, x = k*y + b :=
sorry

theorem line_form_x_eq_ky_add_b_perpendicular_x {b : ℝ} : 
  ∃ k : ℝ, k = 0 ∧ ∀ y : ℝ, x = k*y + b :=
sorry

end line_form_x_eq_ky_add_b_perpendicular_y_line_form_x_eq_ky_add_b_perpendicular_x_l930_93015


namespace smallest_a_value_l930_93042

theorem smallest_a_value :
  ∃ (a : ℝ), (∀ x : ℝ, -3 ≤ x ∧ x ≤ 3 →
    2 * (Real.sin (Real.pi - (Real.pi * x^2 / 12))) * (Real.cos (Real.pi / 6 * Real.sqrt (9 - x^2))) + 1 = a + 2 * (Real.sin (Real.pi / 6 * Real.sqrt (9 - x^2))) * (Real.cos (Real.pi * x^2 / 12))) ∧
    ∀ a' : ℝ, (∀ x : ℝ, -3 ≤ x ∧ x ≤ 3 →
      2 * (Real.sin (Real.pi - (Real.pi * x^2 / 12))) * (Real.cos (Real.pi / 6 * Real.sqrt (9 - x^2))) + 1 = a' + 2 * (Real.sin (Real.pi / 6 * Real.sqrt (9 - x^2))) * (Real.cos (Real.pi * x^2 / 12))) →
      a ≤ a'
  := sorry

end smallest_a_value_l930_93042


namespace base_four_odd_last_digit_l930_93049

theorem base_four_odd_last_digit :
  ∃ b : ℕ, b = 4 ∧ (b^4 ≤ 625 ∧ 625 < b^5) ∧ (625 % b % 2 = 1) :=
by
  sorry

end base_four_odd_last_digit_l930_93049


namespace find_number_l930_93071

theorem find_number (x : ℝ) (h : x = 0.16 * x + 21) : x = 25 :=
by
  sorry

end find_number_l930_93071


namespace least_possible_value_of_z_minus_x_l930_93010

theorem least_possible_value_of_z_minus_x 
  (x y z : ℤ) 
  (h1 : x < y) 
  (h2 : y < z) 
  (h3 : y - x > 5) 
  (h4 : ∃ n : ℤ, x = 2 * n)
  (h5 : ∃ m : ℤ, y = 2 * m + 1) 
  (h6 : ∃ k : ℤ, z = 2 * k + 1) : 
  z - x = 9 := 
sorry

end least_possible_value_of_z_minus_x_l930_93010


namespace tan_of_perpendicular_vectors_l930_93081

theorem tan_of_perpendicular_vectors (θ : ℝ) (hθ1 : 0 < θ) (hθ2 : θ < π / 2)
  (ha : ℝ × ℝ := (Real.cos θ, 2)) (hb : ℝ × ℝ := (-1, Real.sin θ))
  (h_perpendicular : ha.1 * hb.1 + ha.2 * hb.2 = 0) :
  Real.tan θ = 1 / 2 := 
sorry

end tan_of_perpendicular_vectors_l930_93081


namespace proof_by_contradiction_x_gt_y_implies_x3_gt_y3_l930_93005

theorem proof_by_contradiction_x_gt_y_implies_x3_gt_y3
  (x y: ℝ) (h: x > y) : ¬ (x^3 ≤ y^3) :=
by
  -- We need to show that assuming x^3 <= y^3 leads to a contradiction
  sorry

end proof_by_contradiction_x_gt_y_implies_x3_gt_y3_l930_93005


namespace prime_product_sum_91_l930_93034

theorem prime_product_sum_91 (p1 p2 : ℕ) (h1 : Nat.Prime p1) (h2 : Nat.Prime p2) (h3 : p1 + p2 = 91) : p1 * p2 = 178 :=
sorry

end prime_product_sum_91_l930_93034


namespace find_f_x_l930_93092

theorem find_f_x (f : ℝ → ℝ) (h : ∀ x : ℝ, f (2 * x - 1) = x^2 - x) : ∀ x : ℝ, f x = (1/4) * (x^2 - 1) := 
sorry

end find_f_x_l930_93092


namespace trig_identity_l930_93072

theorem trig_identity {α : ℝ} (h : Real.tan α = 2) : 
  (Real.sin (π + α) - Real.cos (π - α)) / 
  (Real.sin (π / 2 + α) - Real.cos (3 * π / 2 - α)) 
  = -1 / 3 := 
by 
  sorry

end trig_identity_l930_93072


namespace maxwell_walking_speed_l930_93009

theorem maxwell_walking_speed :
  ∃ v : ℝ, (8 * v + 6 * 7 = 74) ∧ v = 4 :=
by
  exists 4
  constructor
  { norm_num }
  rfl

end maxwell_walking_speed_l930_93009


namespace rectangle_ratio_l930_93063

theorem rectangle_ratio (w : ℝ) (h : ℝ)
  (hw : h = 10)   -- Length is 10
  (hp : 2 * w + 2 * h = 30) :  -- Perimeter is 30
  w / h = 1 / 2 :=             -- Ratio of width to length is 1/2
by
  -- Pending proof
  sorry

end rectangle_ratio_l930_93063


namespace find_p_l930_93080

theorem find_p (m n p : ℝ) 
  (h1 : m = 3 * n + 5) 
  (h2 : m + 2 = 3 * (n + p) + 5) : p = 2 / 3 :=
by
  sorry

end find_p_l930_93080


namespace magazine_page_height_l930_93085

theorem magazine_page_height
  (charge_per_sq_inch : ℝ := 8)
  (half_page_cost : ℝ := 432)
  (page_width : ℝ := 12) : 
  ∃ h : ℝ, (1/2) * h * page_width * charge_per_sq_inch = half_page_cost :=
by sorry

end magazine_page_height_l930_93085


namespace flowers_bouquets_l930_93000

theorem flowers_bouquets (tulips: ℕ) (roses: ℕ) (extra: ℕ) (total: ℕ) (used_for_bouquets: ℕ) 
(h1: tulips = 36) 
(h2: roses = 37) 
(h3: extra = 3) 
(h4: total = tulips + roses)
(h5: used_for_bouquets = total - extra) :
used_for_bouquets = 70 := by
  sorry

end flowers_bouquets_l930_93000


namespace all_positive_integers_are_nice_l930_93040

def isNice (n : ℕ) : Prop :=
  ∃ (k : ℕ) (a : Fin k → ℕ), (∀ i, ∃ m : ℕ, a i = 2 ^ m) ∧ n = (Finset.univ.sum a) / k

theorem all_positive_integers_are_nice : ∀ n : ℕ, 0 < n → isNice n := sorry

end all_positive_integers_are_nice_l930_93040


namespace one_positive_real_solution_l930_93059

noncomputable def f (x : ℝ) : ℝ := x^4 + 5 * x^3 + 10 * x^2 + 2023 * x - 2021

theorem one_positive_real_solution : 
  ∃! x : ℝ, 0 < x ∧ f x = 0 :=
by
  -- Proof goes here
  sorry

end one_positive_real_solution_l930_93059


namespace train_lengths_l930_93061

theorem train_lengths (L_A L_P L_B : ℕ) (speed_A_km_hr speed_B_km_hr : ℕ) (time_A_seconds : ℕ)
                      (h1 : L_P = L_A)
                      (h2 : speed_A_km_hr = 72)
                      (h3 : speed_B_km_hr = 80)
                      (h4 : time_A_seconds = 60)
                      (h5 : L_B = L_P / 2)
                      (h6 : L_A + L_P = (speed_A_km_hr * 1000 / 3600) * time_A_seconds) :
  L_A = 600 ∧ L_B = 300 :=
by
  sorry

end train_lengths_l930_93061


namespace max_ab_value_l930_93026

noncomputable def max_ab (a b : ℝ) : ℝ := a * b

theorem max_ab_value (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + 4 * b = 8) : max_ab a b ≤ 4 :=
by
  sorry

end max_ab_value_l930_93026


namespace RachelStillToColor_l930_93019

def RachelColoringBooks : Prop :=
  let initial_books := 23 + 32
  let colored := 44
  initial_books - colored = 11

theorem RachelStillToColor : RachelColoringBooks := 
  by
    let initial_books := 23 + 32
    let colored := 44
    show initial_books - colored = 11
    sorry

end RachelStillToColor_l930_93019


namespace largest_multiple_of_11_less_than_100_l930_93050

theorem largest_multiple_of_11_less_than_100 : 
  ∀ n, n < 100 → (∃ k, n = k * 11) → n ≤ 99 :=
by
  intro n hn hmul
  sorry

end largest_multiple_of_11_less_than_100_l930_93050


namespace inequality_proof_l930_93001

theorem inequality_proof (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  (a / (b + 2*c + 3*d)) + (b / (c + 2*d + 3*a)) + (c / (d + 2*a + 3*b)) + (d / (a + 2*b + 3*c)) ≥ 2 / 3 :=
sorry

end inequality_proof_l930_93001


namespace solve_equation_1_solve_equation_2_l930_93027

theorem solve_equation_1 (x : ℝ) : x^2 - 3 * x = 4 ↔ x = 4 ∨ x = -1 :=
by
  sorry

theorem solve_equation_2 (x : ℝ) : x * (x - 2) + x - 2 = 0 ↔ x = 2 ∨ x = -1 :=
by
  sorry

end solve_equation_1_solve_equation_2_l930_93027


namespace minimal_circle_intersect_l930_93018

noncomputable def circle_eq := 
  ∀ (x y : ℝ), 
    (x^2 + y^2 + 4 * x + y + 1 = 0) ∧
    (x^2 + y^2 + 2 * x + 2 * y + 1 = 0) → 
    (x^2 + y^2 + (6/5) * x + (3/5) * y + 1 = 0)

theorem minimal_circle_intersect :
  circle_eq :=
by
  sorry

end minimal_circle_intersect_l930_93018


namespace real_cube_inequality_l930_93074

theorem real_cube_inequality (a b : ℝ) (h : a > b) : a^3 > b^3 :=
sorry

end real_cube_inequality_l930_93074


namespace gcd_ab_l930_93033

def a : ℕ := 130^2 + 215^2 + 310^2
def b : ℕ := 131^2 + 216^2 + 309^2

theorem gcd_ab : Nat.gcd a b = 1 := by
  sorry

end gcd_ab_l930_93033


namespace cosine_angle_is_zero_l930_93088

-- Define the structure of an equilateral triangle
structure EquilateralTriangle where
  side_length : ℝ
  angle_60_deg : Prop

-- Define the structure of a parallelogram built from 6 equilateral triangles
structure Parallelogram where
  composed_of_6_equilateral_triangles : Prop
  folds_into_hexahedral_shape : Prop

-- Define the angle and its cosine computation between two specific directions in the folded hexahedral shape
def cosine_of_angle_between_AB_and_CD (parallelogram : Parallelogram) : ℝ := sorry

-- The condition that needs to be proved
axiom parallelogram_conditions : Parallelogram
axiom cosine_angle_proof : cosine_of_angle_between_AB_and_CD parallelogram_conditions = 0

-- Final proof statement
theorem cosine_angle_is_zero : cosine_of_angle_between_AB_and_CD parallelogram_conditions = 0 :=
cosine_angle_proof

end cosine_angle_is_zero_l930_93088


namespace probability_at_most_one_A_B_selected_l930_93022

def total_employees : ℕ := 36
def ratio_3_2_1 : (ℕ × ℕ × ℕ) := (3, 2, 1)
def sample_size : ℕ := 12
def youth_group_size : ℕ := 6
def total_combinations_youth : ℕ := Nat.choose 6 2
def event_complementary : ℕ := Nat.choose 2 2

theorem probability_at_most_one_A_B_selected :
  let prob := 1 - event_complementary / total_combinations_youth
  prob = (14 : ℚ) / 15 := sorry

end probability_at_most_one_A_B_selected_l930_93022


namespace eggs_left_after_taking_l930_93038

def eggs_in_box_initial : Nat := 47
def eggs_taken_by_Harry : Nat := 5
theorem eggs_left_after_taking : eggs_in_box_initial - eggs_taken_by_Harry = 42 := 
by
  -- Proof placeholder
  sorry

end eggs_left_after_taking_l930_93038


namespace relay_team_permutations_l930_93073

-- Definitions of conditions
def runners := ["Tony", "Leah", "Nina"]
def fixed_positions := ["Maria runs the third lap", "Jordan runs the fifth lap"]

-- Proof statement
theorem relay_team_permutations : 
  ∃ permutations, permutations = 6 := by
sorry

end relay_team_permutations_l930_93073


namespace isosceles_triangle_congruent_side_length_l930_93089

theorem isosceles_triangle_congruent_side_length (BC : ℝ) (BM : ℝ) :
  BC = 4 * Real.sqrt 2 → BM = 5 → ∃ (AB : ℝ), AB = Real.sqrt 34 :=
by
  -- sorry is used here to indicate proof is not provided, but the statement is expected to build successfully.
  sorry

end isosceles_triangle_congruent_side_length_l930_93089


namespace slope_of_line_of_intersections_l930_93008

theorem slope_of_line_of_intersections : 
  ∀ s : ℝ, let x := (41 * s + 13) / 11
           let y := -((2 * s + 6) / 11)
           ∃ m : ℝ, m = -22 / 451 :=
sorry

end slope_of_line_of_intersections_l930_93008


namespace inequality_and_equality_conditions_l930_93053

theorem inequality_and_equality_conditions
    {a b c d : ℝ}
    (ha : 0 < a)
    (hb : 0 < b)
    (hc : 0 < c)
    (hd : 0 < d) :
  (a ^ (1/3) * b ^ (1/3) + c ^ (1/3) * d ^ (1/3) ≤ (a + b + c) ^ (1/3) * (a + c + d) ^ (1/3)) ↔ 
  (b = (a / c) * (a + c) ∧ d = (c / a) * (a + c)) :=
  sorry

end inequality_and_equality_conditions_l930_93053


namespace correct_answer_l930_93017

-- Definitions of the groups
def group_1_well_defined : Prop := false -- Smaller numbers
def group_2_well_defined : Prop := true  -- Non-negative even numbers not greater than 10
def group_3_well_defined : Prop := true  -- All triangles
def group_4_well_defined : Prop := false -- Tall male students

-- Propositions representing the options
def option_A : Prop := group_1_well_defined ∧ group_4_well_defined
def option_B : Prop := group_2_well_defined ∧ group_3_well_defined
def option_C : Prop := group_2_well_defined
def option_D : Prop := group_3_well_defined

-- Theorem stating Option B is the correct answer
theorem correct_answer : option_B ∧ ¬option_A ∧ ¬option_C ∧ ¬option_D := by
  sorry

end correct_answer_l930_93017


namespace inequality_solution_set_l930_93021

noncomputable def solution_set := { x : ℝ | (x < -1 ∨ 1 < x) ∧ x ≠ 4 }

theorem inequality_solution_set : 
  { x : ℝ | (x^2 - 1) / (4 - x)^2 ≥ 0 } = solution_set :=
  by 
    sorry

end inequality_solution_set_l930_93021


namespace unique_prime_sum_diff_l930_93098

theorem unique_prime_sum_diff :
  ∀ p : ℕ, Prime p ∧ (∃ p1 p2 p3 : ℕ, Prime p1 ∧ Prime p2 ∧ Prime p3 ∧ (p = p1 + 2) ∧ (p = p3 - 2)) → p = 5 :=
by
  sorry

end unique_prime_sum_diff_l930_93098


namespace number_of_n_divisible_by_prime_lt_20_l930_93028

theorem number_of_n_divisible_by_prime_lt_20 (N : ℕ) : 
  (N = 69) :=
by
  sorry

end number_of_n_divisible_by_prime_lt_20_l930_93028


namespace solution_set_a_range_m_l930_93044

theorem solution_set_a (a : ℝ) :
  (∀ x : ℝ, |x - a| ≤ 3 ↔ -6 ≤ x ∧ x ≤ 0) ↔ a = -3 :=
by
  sorry

theorem range_m (m : ℝ) :
  (∀ x : ℝ, |x + 3| + |x + 8| ≥ 2 * m) ↔ m ≤ 5 / 2 :=
by
  sorry

end solution_set_a_range_m_l930_93044


namespace population_of_town_l930_93055

theorem population_of_town (F : ℝ) (males : ℕ) (female_glasses : ℝ) (percentage_glasses : ℝ) (total_population : ℝ) 
  (h1 : males = 2000) 
  (h2 : percentage_glasses = 0.30) 
  (h3 : female_glasses = 900) 
  (h4 : percentage_glasses * F = female_glasses) 
  (h5 : total_population = males + F) :
  total_population = 5000 :=
sorry

end population_of_town_l930_93055


namespace overlapping_area_zero_l930_93023

-- Definition of the points and triangles
structure Point where
  x : ℝ
  y : ℝ

structure Triangle where
  p1 : Point
  p2 : Point
  p3 : Point

def point0 : Point := { x := 0, y := 0 }
def point1 : Point := { x := 2, y := 2 }
def point2 : Point := { x := 2, y := 0 }
def point3 : Point := { x := 0, y := 2 }
def point4 : Point := { x := 1, y := 1 }

def triangle1 : Triangle := { p1 := point0, p2 := point1, p3 := point2 }
def triangle2 : Triangle := { p1 := point3, p2 := point1, p3 := point0 }

-- Function to calculate the area of a triangle
def area (t : Triangle) : ℝ :=
  0.5 * abs (t.p1.x * (t.p2.y - t.p3.y) + t.p2.x * (t.p3.y - t.p1.y) + t.p3.x * (t.p1.y - t.p2.y))

-- Using collinear points theorem to prove that the area of the overlapping region is zero
theorem overlapping_area_zero : area { p1 := point0, p2 := point1, p3 := point4 } = 0 := 
by 
  -- This follows directly from the fact that the points (0,0), (2,2), and (1,1) are collinear
  -- skipping the actual geometric proof for conciseness
  sorry

end overlapping_area_zero_l930_93023


namespace lisa_balls_count_l930_93029

def stepNumber := 1729

def base7DigitsSum(x : Nat) : Nat :=
  x / 7 ^ 3 + (x % 343) / 7 ^ 2 + (x % 49) / 7 + x % 7

theorem lisa_balls_count (h1 : stepNumber = 1729) : base7DigitsSum stepNumber = 11 := by
  sorry

end lisa_balls_count_l930_93029


namespace number_of_books_is_10_l930_93003

def costPerBookBeforeDiscount : ℝ := 5
def discountPerBook : ℝ := 0.5
def totalPayment : ℝ := 45

theorem number_of_books_is_10 (n : ℕ) (h : (costPerBookBeforeDiscount - discountPerBook) * n = totalPayment) : n = 10 := by
  sorry

end number_of_books_is_10_l930_93003


namespace cost_of_fencing_irregular_pentagon_l930_93016

noncomputable def total_cost_fencing (AB BC CD DE AE : ℝ) (cost_per_meter : ℝ) : ℝ := 
  (AB + BC + CD + DE + AE) * cost_per_meter

theorem cost_of_fencing_irregular_pentagon :
  total_cost_fencing 20 25 30 35 40 2 = 300 := 
by
  sorry

end cost_of_fencing_irregular_pentagon_l930_93016


namespace carries_average_speed_is_approx_34_29_l930_93035

noncomputable def CarriesActualAverageSpeed : ℝ :=
  let jerry_speed := 40 -- in mph
  let jerry_time := 1/2 -- in hours, 30 minutes = 0.5 hours
  let jerry_distance := jerry_speed * jerry_time

  let beth_distance := jerry_distance + 5
  let beth_time := jerry_time + (20 / 60) -- converting 20 minutes to hours

  let carrie_distance := 2 * jerry_distance
  let carrie_time := 1 + (10 / 60) -- converting 10 minutes to hours

  carrie_distance / carrie_time

theorem carries_average_speed_is_approx_34_29 : 
  |CarriesActualAverageSpeed - 34.29| < 0.01 :=
sorry

end carries_average_speed_is_approx_34_29_l930_93035


namespace problem_statement_l930_93043

open Classical

variable (p q : Prop)

theorem problem_statement (h1 : p ∨ q) (h2 : ¬(p ∧ q)) (h3 : ¬ p) : (p = (5 + 2 = 6) ∧ q = (6 > 2)) :=
by
  have hp : p = False := by sorry
  have hq : q = True := by sorry
  exact ⟨by sorry, by sorry⟩

end problem_statement_l930_93043


namespace collinear_points_sum_l930_93020

theorem collinear_points_sum (p q : ℝ) 
  (h1 : p = 2) (h2 : q = 4) 
  (collinear : ∃ (s : ℝ), 
     (2, p, q) = (2, s*p, s*q) ∧ 
     (p, 3, q) = (s*p, 3, s*q) ∧ 
     (p, q, 4) = (s*p, s*q, 4)): 
  p + q = 6 := by
  sorry

end collinear_points_sum_l930_93020


namespace arithmetic_sequence_geometric_sequence_l930_93093

-- Problem 1
theorem arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) (Sₙ : ℝ) 
  (h₁ : a₁ = 3 / 2) (h₂ : d = -1 / 2) (h₃ : Sₙ = -15) :
  n = 12 ∧ (a₁ + (n - 1) * d) = -4 := 
sorry

-- Problem 2
theorem geometric_sequence (a₁ : ℝ) (q : ℝ) (n : ℕ) (aₙ Sₙ : ℝ) 
  (h₁ : q = 2) (h₂ : aₙ = 96) (h₃ : Sₙ = 189) :
  a₁ = 3 ∧ n = 6 := 
sorry

end arithmetic_sequence_geometric_sequence_l930_93093


namespace number_of_black_and_white_films_l930_93084

theorem number_of_black_and_white_films (B x y : ℤ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h_fraction : (6 * y : ℚ) / ((y / (x : ℚ))/100 * (B : ℚ) + 6 * y) = 20 / 21) :
  B = 30 * x :=
sorry

end number_of_black_and_white_films_l930_93084


namespace train_length_is_400_l930_93032

-- Define the conditions
def time := 40 -- seconds
def speed_kmh := 36 -- km/h

-- Conversion factor from km/h to m/s
def kmh_to_ms (v : ℕ) := (v * 5) / 18

def speed_ms := kmh_to_ms speed_kmh -- convert speed to m/s

-- Definition of length of the train using the given conditions
def train_length := speed_ms * time

-- Theorem to prove the length of the train is 400 meters
theorem train_length_is_400 : train_length = 400 := by
  sorry

end train_length_is_400_l930_93032


namespace complex_division_evaluation_l930_93039

open Complex

theorem complex_division_evaluation :
  (2 : ℂ) / (I * (3 - I)) = (1 / 5 : ℂ) - (3 / 5) * I :=
by
  sorry

end complex_division_evaluation_l930_93039


namespace complete_square_transformation_l930_93048

theorem complete_square_transformation (x : ℝ) : 
  2 * x^2 - 4 * x - 3 = 0 ↔ (x - 1)^2 - (5 / 2) = 0 :=
sorry

end complete_square_transformation_l930_93048


namespace polynomial_at_x_neg_four_l930_93007

noncomputable def f (x : ℝ) : ℝ :=
  12 + 35 * x - 8 * x^2 + 79 * x^3 + 6 * x^4 + 5 * x^5 + 3 * x^6

theorem polynomial_at_x_neg_four : 
  f (-4) = 220 := by
  sorry

end polynomial_at_x_neg_four_l930_93007


namespace inequality_proof_l930_93045

open Real

theorem inequality_proof (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h_sum : a + b + c = 1) : 
  (a - b * c) / (a + b * c) + (b - c * a) / (b + c * a) + (c - a * b) / (c + a * b) ≤ 3 / 2 :=
sorry

end inequality_proof_l930_93045


namespace toby_photos_l930_93082

variable (p0 d c e x : ℕ)
def photos_remaining : ℕ := p0 - d + c + x - e

theorem toby_photos (h1 : p0 = 63) (h2 : d = 7) (h3 : c = 15) (h4 : e = 3) : photos_remaining p0 d c e x = 68 + x :=
by
  rw [h1, h2, h3, h4]
  sorry

end toby_photos_l930_93082


namespace number_of_boundaries_l930_93065

def total_runs : ℕ := 120
def sixes : ℕ := 4
def runs_per_six : ℕ := 6
def percentage_runs_by_running : ℚ := 0.60
def runs_per_boundary : ℕ := 4

theorem number_of_boundaries :
  let runs_by_running := (percentage_runs_by_running * total_runs : ℚ)
  let runs_by_sixes := (sixes * runs_per_six)
  let runs_by_boundaries := (total_runs - runs_by_running - runs_by_sixes : ℚ)
  (runs_by_boundaries / runs_per_boundary) = 6 := by
  sorry

end number_of_boundaries_l930_93065


namespace triangle_sides_square_perfect_l930_93052

theorem triangle_sides_square_perfect (x y z : ℕ) (h : ∃ h_x h_y h_z, 
  h_x = h_y + h_z ∧ 
  2 * h_x * x = 2 * h_y * y ∧ 
  2 * h_x * x = 2 * h_z * z ) :
  ∃ k : ℕ, x^2 + y^2 + z^2 = k^2 :=
by
  sorry

end triangle_sides_square_perfect_l930_93052


namespace base6_sum_correct_l930_93012

theorem base6_sum_correct {S H E : ℕ} (hS : S < 6) (hH : H < 6) (hE : E < 6) 
  (dist : S ≠ H ∧ H ≠ E ∧ S ≠ E) 
  (rightmost : (E + E) % 6 = S) 
  (second_rightmost : (H + H + if E + E < 6 then 0 else 1) % 6 = E) :
  S + H + E = 11 := 
by sorry

end base6_sum_correct_l930_93012


namespace area_to_be_painted_correct_l930_93083

-- Define the dimensions and areas involved
def wall_height : ℕ := 10
def wall_length : ℕ := 15
def window_height : ℕ := 3
def window_length : ℕ := 5
def painting_height : ℕ := 2
def painting_length : ℕ := 2

-- Calculate the areas
def wall_area : ℕ := wall_height * wall_length
def window_area : ℕ := window_height * window_length
def painting_area : ℕ := painting_height * painting_length
def area_not_painted : ℕ := window_area + painting_area
def area_to_be_painted : ℕ := wall_area - area_not_painted

-- Theorem: The area to be painted is 131 square feet
theorem area_to_be_painted_correct : area_to_be_painted = 131 := by
  sorry

end area_to_be_painted_correct_l930_93083


namespace gcd_456_357_l930_93002

theorem gcd_456_357 : Nat.gcd 456 357 = 3 := by
  sorry

end gcd_456_357_l930_93002


namespace total_robots_correct_l930_93090

def number_of_shapes : ℕ := 3
def number_of_colors : ℕ := 4
def total_types_of_robots : ℕ := number_of_shapes * number_of_colors

theorem total_robots_correct : total_types_of_robots = 12 := by
  sorry

end total_robots_correct_l930_93090


namespace number_of_perfect_square_factors_l930_93057

theorem number_of_perfect_square_factors :
  let n := (2^14) * (3^9) * (5^20)
  ∃ (count : ℕ), 
  (∀ (a : ℕ) (h : a ∣ n), (∃ k, a = k^2) → true) →
  count = 440 :=
by
  sorry

end number_of_perfect_square_factors_l930_93057


namespace odd_function_f_x_pos_l930_93013

variable (f : ℝ → ℝ)

theorem odd_function_f_x_pos {x : ℝ} (h1 : ∀ x < 0, f x = x^2 + x)
  (h2 : ∀ x, f x = -f (-x)) (hx : 0 < x) :
  f x = -x^2 + x := by
  sorry

end odd_function_f_x_pos_l930_93013


namespace find_sales_discount_l930_93069

noncomputable def salesDiscountPercentage (P N : ℝ) (D : ℝ): Prop :=
  let originalGrossIncome := P * N
  let newPrice := P * (1 - D / 100)
  let newNumberOfItems := N * 1.20
  let newGrossIncome := newPrice * newNumberOfItems
  newGrossIncome = originalGrossIncome * 1.08

theorem find_sales_discount (P N : ℝ) (hP : P > 0) (hN : N > 0) (h: ∃ D, salesDiscountPercentage P N D) :
  ∃ D, D = 10 :=
sorry

end find_sales_discount_l930_93069


namespace solve_floor_trig_eq_l930_93004

-- Define the floor function
def floor (x : ℝ) : ℤ := by 
  sorry

-- Define the condition and theorem
theorem solve_floor_trig_eq (x : ℝ) (n : ℤ) : 
  floor (Real.sin x + Real.cos x) = 1 ↔ (∃ n : ℤ, 2 * Real.pi * n ≤ x ∧ x ≤ (2 * Real.pi * n + Real.pi / 2)) := 
  by 
  sorry

end solve_floor_trig_eq_l930_93004


namespace temperature_difference_l930_93078

theorem temperature_difference (H L : ℝ) (hH : H = 8) (hL : L = -2) :
  H - L = 10 :=
by
  rw [hH, hL]
  norm_num

end temperature_difference_l930_93078


namespace divisible_by_10_l930_93067

theorem divisible_by_10 : (11 * 21 * 31 * 41 * 51 - 1) % 10 = 0 := by
  sorry

end divisible_by_10_l930_93067


namespace wooden_easel_cost_l930_93031

noncomputable def cost_paintbrush : ℝ := 1.5
noncomputable def cost_set_of_paints : ℝ := 4.35
noncomputable def amount_already_have : ℝ := 6.5
noncomputable def additional_amount_needed : ℝ := 12
noncomputable def total_cost_items : ℝ := cost_paintbrush + cost_set_of_paints
noncomputable def total_amount_needed : ℝ := amount_already_have + additional_amount_needed

theorem wooden_easel_cost :
  total_amount_needed - total_cost_items = 12.65 :=
by
  sorry

end wooden_easel_cost_l930_93031


namespace right_isosceles_hypotenuse_angle_l930_93077

theorem right_isosceles_hypotenuse_angle (α β : ℝ) (γ : ℝ)
  (h1 : α = 45) (h2 : β = 45) (h3 : γ = 90)
  (triangle_isosceles : α = β)
  (triangle_right : γ = 90) :
  γ = 90 :=
by
  sorry

end right_isosceles_hypotenuse_angle_l930_93077


namespace birds_flew_away_l930_93087

-- Define the initial and remaining birds
def original_birds : ℕ := 12
def remaining_birds : ℕ := 4

-- Define the number of birds that flew away
noncomputable def flew_away_birds : ℕ := original_birds - remaining_birds

-- State the theorem that the number of birds that flew away is 8
theorem birds_flew_away : flew_away_birds = 8 := by
  -- Lean expects a proof here. For now, we use sorry to indicate the proof is skipped.
  sorry

end birds_flew_away_l930_93087


namespace red_marbles_count_l930_93066

theorem red_marbles_count :
  ∀ (total marbles white yellow green red : ℕ),
    total = 50 →
    white = total / 2 →
    yellow = 12 →
    green = yellow / 2 →
    red = total - (white + yellow + green) →
    red = 7 :=
by
  intros total marbles white yellow green red Htotal Hwhite Hyellow Hgreen Hred
  sorry

end red_marbles_count_l930_93066


namespace find_m_l930_93054

variable {a_n : ℕ → ℤ}
variable {S : ℕ → ℤ}

def isArithmeticSeq (a_n : ℕ → ℤ) (d : ℤ) : Prop :=
∀ n, a_n (n + 1) = a_n n + d

def sumSeq (S : ℕ → ℤ) (a_n : ℕ → ℤ) : Prop :=
∀ n, S n = (n * (a_n 1 + a_n n)) / 2

theorem find_m
  (d : ℤ)
  (a_1 : ℤ)
  (a_n : ∀ n, ℤ)
  (S : ℕ → ℤ)
  (h_arith : isArithmeticSeq a_n d)
  (h_sum : sumSeq S a_n)
  (h1 : S (m - 1) = -2)
  (h2 : S m = 0)
  (h3 : S (m + 1) = 3) :
  m = 5 :=
sorry

end find_m_l930_93054


namespace false_proposition_l930_93079

theorem false_proposition :
  ¬ (∀ x : ℕ, (x > 0) → (x - 2)^2 > 0) :=
by
  sorry

end false_proposition_l930_93079


namespace solve_linear_system_l930_93046

theorem solve_linear_system (m x y : ℝ) 
  (h1 : x + y = 3 * m) 
  (h2 : x - y = 5 * m)
  (h3 : 2 * x + 3 * y = 10) : 
  m = 2 := 
by 
  sorry

end solve_linear_system_l930_93046


namespace find_range_of_r_l930_93099

noncomputable def range_of_r : Set ℝ :=
  {r : ℝ | 3 * Real.sqrt 5 - 3 * Real.sqrt 2 ≤ r ∧ r ≤ 3 * Real.sqrt 5 + 3 * Real.sqrt 2}

theorem find_range_of_r 
  (O : ℝ × ℝ) (A : ℝ × ℝ) (r : ℝ) (h : r > 0)
  (hA : A = (0, 3))
  (C : Set (ℝ × ℝ)) (hC : C = {M : ℝ × ℝ | (M.1 - 3)^2 + (M.2 - 3)^2 = r^2})
  (M : ℝ × ℝ) (hM : M ∈ C)
  (h_cond : (M.1 - 0)^2 + (M.2 - 3)^2 = 2 * ((M.1 - 0)^2 + (M.2 - 0)^2)) :
  r ∈ range_of_r :=
sorry

end find_range_of_r_l930_93099


namespace seven_power_units_digit_l930_93051

def units_digit (n : ℕ) : ℕ := (7^n) % 10

theorem seven_power_units_digit : units_digit 2023 = 3 := by
  -- Proof omitted for simplification
  sorry

end seven_power_units_digit_l930_93051


namespace transportation_degrees_correct_l930_93058

-- Define the percentages for the different categories.
def salaries_percent := 0.60
def research_development_percent := 0.09
def utilities_percent := 0.05
def equipment_percent := 0.04
def supplies_percent := 0.02

-- Define the total percentage of non-transportation categories.
def non_transportation_percent := 
  salaries_percent + research_development_percent + utilities_percent + equipment_percent + supplies_percent

-- Define the full circle in degrees.
def full_circle_degrees := 360.0

-- Total percentage which must sum to 1 (i.e., 100%).
def total_budget_percent := 1.0

-- Calculate the percentage for transportation.
def transportation_percent := total_budget_percent - non_transportation_percent

-- Define the result for degrees allocated to transportation.
def transportation_degrees := transportation_percent * full_circle_degrees

-- Prove that the transportation degrees are 72.
theorem transportation_degrees_correct : transportation_degrees = 72.0 :=
by
  unfold transportation_degrees transportation_percent non_transportation_percent
  sorry

end transportation_degrees_correct_l930_93058


namespace inequality_proof_l930_93076

open Real

theorem inequality_proof (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxyz : x + y + z = 1) :
    (x^4 / (y * (1 - y^2))) + (y^4 / (z * (1 - z^2))) + (z^4 / (x * (1 - x^2))) ≥ 1 / 8 :=
sorry

end inequality_proof_l930_93076


namespace smallest_rectangles_required_l930_93014

theorem smallest_rectangles_required :
  ∀ (r h : ℕ) (area_square length_square : ℕ),
  r = 3 → h = 4 →
  (∀ k, (k: ℕ) ∣ (r * h) → (k: ℕ) = r * h) →
  length_square = 12 →
  area_square = length_square * length_square →
  (area_square / (r * h) = 12) :=
by
  intros
  /- The mathematical proof steps will be filled here -/
  sorry

end smallest_rectangles_required_l930_93014


namespace simplify_expression_l930_93094

theorem simplify_expression (m n : ℝ) (h : m^2 + 3 * m * n = 5) : 
  5 * m^2 - 3 * m * n - (-9 * m * n + 3 * m^2) = 10 :=
by 
  sorry

end simplify_expression_l930_93094


namespace points_product_l930_93060

def f (n : ℕ) : ℕ :=
  if n % 6 == 0 then 6
  else if n % 2 == 0 then 2
  else 0

def total_points (rolls : List ℕ) : ℕ :=
  rolls.map f |>.sum

def allie_rolls := [5, 4, 1, 2]
def betty_rolls := [6, 3, 3, 2]

def allie_points := total_points allie_rolls
def betty_points := total_points betty_rolls

theorem points_product : allie_points * betty_points = 32 := by
  sorry

end points_product_l930_93060


namespace number_of_ways_l930_93097

-- Define the conditions
def num_people : ℕ := 3
def num_sports : ℕ := 4

-- Prove the total number of different ways
theorem number_of_ways : num_sports ^ num_people = 64 := by
  sorry

end number_of_ways_l930_93097


namespace avg_daily_distance_third_dog_summer_l930_93036

theorem avg_daily_distance_third_dog_summer :
  ∀ (total_days weekends miles_walked_weekday : ℕ), 
    total_days = 30 → weekends = 8 → miles_walked_weekday = 3 →
    (66 / 30 : ℝ) = 2.2 :=
by
  intros total_days weekends miles_walked_weekday h_total h_weekends h_walked
  -- proof goes here
  sorry

end avg_daily_distance_third_dog_summer_l930_93036


namespace largest_whole_number_lt_150_l930_93056

theorem largest_whole_number_lt_150 : 
  ∃ x : ℕ, (9 * x < 150) ∧ (∀ y : ℕ, 9 * y < 150 → y ≤ x) :=
  sorry

end largest_whole_number_lt_150_l930_93056


namespace correct_average_is_15_l930_93064

theorem correct_average_is_15 (n incorrect_avg correct_num wrong_num : ℕ) 
  (h1 : n = 10) (h2 : incorrect_avg = 14) (h3 : correct_num = 36) (h4 : wrong_num = 26) : 
  (incorrect_avg * n + (correct_num - wrong_num)) / n = 15 := 
by 
  sorry

end correct_average_is_15_l930_93064


namespace train_speed_is_40_kmh_l930_93096

noncomputable def speed_of_train (train_length_m : ℝ) 
                                   (man_speed_kmh : ℝ) 
                                   (pass_time_s : ℝ) : ℝ :=
  let train_length_km := train_length_m / 1000
  let pass_time_h := pass_time_s / 3600
  let relative_speed_kmh := train_length_km / pass_time_h
  relative_speed_kmh - man_speed_kmh
  
theorem train_speed_is_40_kmh :
  speed_of_train 110 4 9 = 40 := 
by
  sorry

end train_speed_is_40_kmh_l930_93096


namespace jame_weeks_tearing_cards_l930_93037

def cards_tears_per_time : ℕ := 30
def cards_per_deck : ℕ := 55
def tears_per_week : ℕ := 3
def decks_bought : ℕ := 18

theorem jame_weeks_tearing_cards :
  (cards_tears_per_time * tears_per_week * decks_bought * cards_per_deck) / (cards_tears_per_time * tears_per_week) = 11 := by
  sorry

end jame_weeks_tearing_cards_l930_93037


namespace one_clerk_forms_per_hour_l930_93011

theorem one_clerk_forms_per_hour
  (total_forms : ℕ)
  (total_hours : ℕ)
  (total_clerks : ℕ) 
  (h1 : total_forms = 2400)
  (h2 : total_hours = 8)
  (h3 : total_clerks = 12) :
  (total_forms / total_hours) / total_clerks = 25 :=
by
  have forms_per_hour := total_forms / total_hours
  have forms_per_clerk_per_hour := forms_per_hour / total_clerks
  sorry

end one_clerk_forms_per_hour_l930_93011


namespace expression_evaluation_l930_93070

def e1 : ℤ := 72 + (120 / 15) + (15 * 12) - 250 - (480 / 8)

theorem expression_evaluation : e1 = -50 :=
by
  sorry

end expression_evaluation_l930_93070


namespace cost_price_per_meter_l930_93091

theorem cost_price_per_meter (number_of_meters : ℕ) (selling_price : ℝ) (profit_per_meter : ℝ) (total_cost_price : ℝ) (cost_per_meter : ℝ) :
  number_of_meters = 85 →
  selling_price = 8925 →
  profit_per_meter = 15 →
  total_cost_price = selling_price - (profit_per_meter * number_of_meters) →
  cost_per_meter = total_cost_price / number_of_meters →
  cost_per_meter = 90 :=
by
  intros h1 h2 h3 h4 h5 
  sorry

end cost_price_per_meter_l930_93091


namespace smallest_possible_degree_p_l930_93006

theorem smallest_possible_degree_p (p : Polynomial ℝ) :
  (∀ x, 0 < |x| → ∃ C, |((3 * x^7 + 2 * x^6 - 4 * x^3 + x - 5) / (p.eval x)) - C| < ε)
  → (Polynomial.degree p) ≥ 7 := by
  sorry

end smallest_possible_degree_p_l930_93006


namespace two_y_minus_three_x_l930_93075

variable (x y : ℝ)

noncomputable def x_val : ℝ := 1.2 * 98
noncomputable def y_val : ℝ := 0.9 * (x_val + 35)

theorem two_y_minus_three_x : 2 * y_val - 3 * x_val = -78.12 := by
  sorry

end two_y_minus_three_x_l930_93075
