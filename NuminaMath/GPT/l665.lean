import Mathlib

namespace NUMINAMATH_GPT_inverse_proposition_of_square_positive_l665_66589

theorem inverse_proposition_of_square_positive :
  (∀ x : ℝ, x < 0 → x^2 > 0) →
  (∀ x : ℝ, ¬ (x^2 > 0) → ¬ (x < 0)) :=
by
  intro h
  intros x h₁
  sorry

end NUMINAMATH_GPT_inverse_proposition_of_square_positive_l665_66589


namespace NUMINAMATH_GPT_negation_of_exists_abs_lt_one_l665_66564

theorem negation_of_exists_abs_lt_one :
  (¬ ∃ x : ℝ, |x| < 1) ↔ (∀ x : ℝ, x ≤ -1 ∨ x ≥ 1) :=
by
  sorry

end NUMINAMATH_GPT_negation_of_exists_abs_lt_one_l665_66564


namespace NUMINAMATH_GPT_Christine_savings_l665_66574

theorem Christine_savings 
  (commission_rate: ℝ) 
  (total_sales: ℝ) 
  (personal_needs_percentage: ℝ) 
  (savings: ℝ) 
  (h1: commission_rate = 0.12) 
  (h2: total_sales = 24000) 
  (h3: personal_needs_percentage = 0.60) 
  (h4: savings = total_sales * commission_rate * (1 - personal_needs_percentage)) : 
  savings = 1152 := by 
  sorry

end NUMINAMATH_GPT_Christine_savings_l665_66574


namespace NUMINAMATH_GPT_value_of_a_l665_66540

theorem value_of_a (a : ℚ) (h : 2 * a + a / 2 = 9 / 2) : a = 9 / 5 :=
by
  sorry

end NUMINAMATH_GPT_value_of_a_l665_66540


namespace NUMINAMATH_GPT_math_team_combinations_l665_66576

def numGirls : ℕ := 4
def numBoys : ℕ := 7
def girlsToChoose : ℕ := 2
def boysToChoose : ℕ := 3

def comb (n k : ℕ) : ℕ := n.choose k

theorem math_team_combinations : 
  comb numGirls girlsToChoose * comb numBoys boysToChoose = 210 := 
by
  sorry

end NUMINAMATH_GPT_math_team_combinations_l665_66576


namespace NUMINAMATH_GPT_tanA_tanB_eq_thirteen_div_four_l665_66567

-- Define the triangle and its properties
variables {A B C : Type}
variables (a b c : ℝ)  -- sides BC, AC, AB
variables (HF HC : ℝ)  -- segments of altitude CF
variables (tanA tanB : ℝ)

-- Given conditions
def orthocenter_divide_altitude (HF HC : ℝ) : Prop :=
  HF = 8 ∧ HC = 18

-- The result we want to prove
theorem tanA_tanB_eq_thirteen_div_four (h : orthocenter_divide_altitude HF HC) : 
  tanA * tanB = 13 / 4 :=
  sorry

end NUMINAMATH_GPT_tanA_tanB_eq_thirteen_div_four_l665_66567


namespace NUMINAMATH_GPT_pizza_left_for_Wally_l665_66553

theorem pizza_left_for_Wally (a b c : ℚ) (ha : a = 1/3) (hb : b = 1/6) (hc : c = 1/4) :
  1 - (a + b + c) = 1/4 :=
by
  sorry

end NUMINAMATH_GPT_pizza_left_for_Wally_l665_66553


namespace NUMINAMATH_GPT_value_of_t_l665_66527

theorem value_of_t (k : ℤ) (t : ℤ) (h1 : t = 5 / 9 * (k - 32)) (h2 : k = 68) : t = 20 :=
by
  sorry

end NUMINAMATH_GPT_value_of_t_l665_66527


namespace NUMINAMATH_GPT_region_midpoint_area_equilateral_triangle_52_36_l665_66507

noncomputable def equilateral_triangle (A B C: ℝ × ℝ) : Prop :=
  dist A B = 2 ∧ dist B C = 2 ∧ dist C A = 2

def midpoint_region_area (a b c : ℝ × ℝ) : ℝ := sorry

theorem region_midpoint_area_equilateral_triangle_52_36 (A B C: ℝ × ℝ) (h: equilateral_triangle A B C) :
  let m := (midpoint_region_area A B C)
  100 * m = 52.36 :=
sorry

end NUMINAMATH_GPT_region_midpoint_area_equilateral_triangle_52_36_l665_66507


namespace NUMINAMATH_GPT_water_lilies_half_pond_l665_66524

theorem water_lilies_half_pond (growth_rate : ℕ → ℕ) (start_day : ℕ) (full_covered_day : ℕ) 
  (h_growth : ∀ n, growth_rate (n + 1) = 2 * growth_rate n) 
  (h_start : growth_rate start_day = 1) 
  (h_full_covered : growth_rate full_covered_day = 2^(full_covered_day - start_day)) : 
  growth_rate (full_covered_day - 1) = 2^(full_covered_day - start_day - 1) :=
by
  sorry

end NUMINAMATH_GPT_water_lilies_half_pond_l665_66524


namespace NUMINAMATH_GPT_farmer_plough_remaining_area_l665_66530

theorem farmer_plough_remaining_area :
  ∀ (x R : ℕ),
  (90 * x = 3780) →
  (85 * (x + 2) + R = 3780) →
  R = 40 :=
by
  intros x R h1 h2
  sorry

end NUMINAMATH_GPT_farmer_plough_remaining_area_l665_66530


namespace NUMINAMATH_GPT_proof_problem_l665_66506

theorem proof_problem (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^2 + b + (3/4)) * (b^2 + c + (3/4)) * (c^2 + a + (3/4)) ≥ (2 * a + (1/2)) * (2 * b + (1/2)) * (2 * c + (1/2)) := 
by
  sorry

end NUMINAMATH_GPT_proof_problem_l665_66506


namespace NUMINAMATH_GPT_fewer_cans_l665_66531

theorem fewer_cans (sarah_yesterday lara_more alex_yesterday sarah_today lara_today alex_today : ℝ)
  (H1 : sarah_yesterday = 50.5)
  (H2 : lara_more = 30.3)
  (H3 : alex_yesterday = 90.2)
  (H4 : sarah_today = 40.7)
  (H5 : lara_today = 70.5)
  (H6 : alex_today = 55.3) :
  (sarah_yesterday + (sarah_yesterday + lara_more) + alex_yesterday) - (sarah_today + lara_today + alex_today) = 55 :=
by {
  -- Sorry to skip the proof
  sorry
}

end NUMINAMATH_GPT_fewer_cans_l665_66531


namespace NUMINAMATH_GPT_simplify_expression_l665_66556

variable (x : ℝ)

theorem simplify_expression :
  3 - 5 * x - 7 * x^2 + 9 + 11 * x - 13 * x^2 - 15 + 17 * x + 19 * x^2 = -x^2 + 23 * x - 3 :=
sorry

end NUMINAMATH_GPT_simplify_expression_l665_66556


namespace NUMINAMATH_GPT_chord_slope_of_ellipse_l665_66522

theorem chord_slope_of_ellipse :
  (∃ (x1 y1 x2 y2 : ℝ), (x1 + x2)/2 = 4 ∧ (y1 + y2)/2 = 2 ∧
    (x1^2)/36 + (y1^2)/9 = 1 ∧ (x2^2)/36 + (y2^2)/9 = 1) →
    (∃ k : ℝ, k = (y1 - y2)/(x1 - x2) ∧ k = -1/2) :=
sorry

end NUMINAMATH_GPT_chord_slope_of_ellipse_l665_66522


namespace NUMINAMATH_GPT_Peter_bought_5_kilos_of_cucumbers_l665_66569

/-- 
Peter carried $500 to the market. 
He bought 6 kilos of potatoes for $2 per kilo, 
9 kilos of tomato for $3 per kilo, 
some kilos of cucumbers for $4 per kilo, 
and 3 kilos of bananas for $5 per kilo. 
After buying all these items, Peter has $426 remaining. 
How many kilos of cucumbers did Peter buy? 
-/
theorem Peter_bought_5_kilos_of_cucumbers : 
   ∃ (kilos_cucumbers : ℕ),
   (500 - (6 * 2 + 9 * 3 + 3 * 5 + kilos_cucumbers * 4) = 426) →
   kilos_cucumbers = 5 :=
sorry

end NUMINAMATH_GPT_Peter_bought_5_kilos_of_cucumbers_l665_66569


namespace NUMINAMATH_GPT_wrapping_paper_area_l665_66590

variable (l w h : ℝ)
variable (l_gt_w : l > w)

def area_wrapping_paper (l w h : ℝ) : ℝ :=
  3 * (l + w) * h

theorem wrapping_paper_area :
  area_wrapping_paper l w h = 3 * (l + w) * h :=
sorry

end NUMINAMATH_GPT_wrapping_paper_area_l665_66590


namespace NUMINAMATH_GPT_pentagon_area_l665_66575

/-- Given a convex pentagon ABCDE where BE and CE are angle bisectors at vertices B and C 
respectively, with ∠A = 35 degrees, ∠D = 145 degrees, and the area of triangle BCE is 11, 
prove that the area of the pentagon ABCDE is 22. -/
theorem pentagon_area (ABCDE : Type) (angle_A : ℝ) (angle_D : ℝ) (area_BCE : ℝ)
  (h_A : angle_A = 35) (h_D : angle_D = 145) (h_area_BCE : area_BCE = 11) :
  ∃ (area_ABCDE : ℝ), area_ABCDE = 22 :=
by
  sorry

end NUMINAMATH_GPT_pentagon_area_l665_66575


namespace NUMINAMATH_GPT_gcd_of_17934_23526_51774_l665_66597

-- Define the three integers
def a : ℕ := 17934
def b : ℕ := 23526
def c : ℕ := 51774

-- State the theorem
theorem gcd_of_17934_23526_51774 : Int.gcd a (Int.gcd b c) = 2 := by
  sorry

end NUMINAMATH_GPT_gcd_of_17934_23526_51774_l665_66597


namespace NUMINAMATH_GPT_num_true_statements_l665_66512

theorem num_true_statements :
  (∀ x y a, a ≠ 0 → (a^2 * x > a^2 * y → x > y)) ∧
  (∀ x y a, a ≠ 0 → (a^2 * x ≥ a^2 * y → x ≥ y)) ∧
  (∀ x y a, a ≠ 0 → (x / a^2 ≥ y / a^2 → x ≥ y)) ∧
  (∀ x y a, a ≠ 0 → (x ≥ y → x / a^2 ≥ y / a^2)) →
  ((∀ x y a, a ≠ 0 → (a^2 * x > a^2 * y → x > y)) →
   (∀ x y a, a ≠ 0 → (x / a^2 ≥ y / a^2 → x ≥ y))) :=
sorry

end NUMINAMATH_GPT_num_true_statements_l665_66512


namespace NUMINAMATH_GPT_max_height_l665_66593

noncomputable def ball_height (t : ℝ) : ℝ :=
  -4.9 * t^2 + 50 * t + 15

theorem max_height : ∃ t : ℝ, t < 50 / 4.9 ∧ ball_height t = 142.65 :=
sorry

end NUMINAMATH_GPT_max_height_l665_66593


namespace NUMINAMATH_GPT_factor_expression_l665_66585

theorem factor_expression (y : ℝ) : 
  3 * y * (2 * y + 5) + 4 * (2 * y + 5) = (3 * y + 4) * (2 * y + 5) :=
by
  sorry

end NUMINAMATH_GPT_factor_expression_l665_66585


namespace NUMINAMATH_GPT_distance_eq_l665_66547

open Real

variables (a b c d p q: ℝ)

-- Conditions from step a)
def onLine1 : Prop := b = (p-1)*a + q
def onLine2 : Prop := d = (p-1)*c + q

-- Theorem about the distance between points (a, b) and (c, d)
theorem distance_eq : 
  onLine1 a b p q → 
  onLine2 c d p q → 
  dist (a, b) (c, d) = abs (a - c) * sqrt (1 + (p - 1)^2) := 
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_distance_eq_l665_66547


namespace NUMINAMATH_GPT_trip_total_time_trip_average_speed_l665_66572

structure Segment where
  distance : ℝ -- in kilometers
  speed : ℝ -- average speed in km/hr
  break_time : ℝ -- in minutes

def seg1 := Segment.mk 12 13 15
def seg2 := Segment.mk 18 16 30
def seg3 := Segment.mk 25 20 45
def seg4 := Segment.mk 35 25 60
def seg5 := Segment.mk 50 22 0

noncomputable def total_time_minutes (segs : List Segment) : ℝ :=
  segs.foldl (λ acc s => acc + (s.distance / s.speed) * 60 + s.break_time) 0

noncomputable def total_distance (segs : List Segment) : ℝ :=
  segs.foldl (λ acc s => acc + s.distance) 0

noncomputable def overall_average_speed (segs : List Segment) : ℝ :=
  total_distance segs / (total_time_minutes segs / 60)

def segments := [seg1, seg2, seg3, seg4, seg5]

theorem trip_total_time : total_time_minutes segments = 568.24 := by sorry
theorem trip_average_speed : overall_average_speed segments = 14.78 := by sorry

end NUMINAMATH_GPT_trip_total_time_trip_average_speed_l665_66572


namespace NUMINAMATH_GPT_time_A_problems_60_l665_66591

variable (t : ℕ) -- time in minutes per type B problem

def time_per_A_problem := 2 * t
def time_per_C_problem := t / 2
def total_time_for_A_problems := 20 * time_per_A_problem

theorem time_A_problems_60 (hC : 80 * time_per_C_problem = 60) : total_time_for_A_problems = 60 := by
  sorry

end NUMINAMATH_GPT_time_A_problems_60_l665_66591


namespace NUMINAMATH_GPT_graphs_symmetric_respect_to_x_equals_1_l665_66537

-- Define the function f
variable (f : ℝ → ℝ)

-- Define g(x) = f(x-1)
def g (x : ℝ) : ℝ := f (x - 1)

-- Define h(x) = f(1 - x)
def h (x : ℝ) : ℝ := f (1 - x)

-- The theorem that their graphs are symmetric with respect to the line x = 1
theorem graphs_symmetric_respect_to_x_equals_1 :
  ∀ x : ℝ, g f x = h f x ↔ f x = f (2 - x) :=
sorry

end NUMINAMATH_GPT_graphs_symmetric_respect_to_x_equals_1_l665_66537


namespace NUMINAMATH_GPT_freight_cost_minimization_l665_66541

-- Define the main parameters: tonnage and costs for the trucks.
def freight_cost (num_seven_ton_trucks : ℕ) (num_five_ton_trucks : ℕ) : ℕ :=
  65 * num_seven_ton_trucks + 50 * num_five_ton_trucks

-- Define the total transported capacity by the two types of trucks.
def total_capacity (num_seven_ton_trucks : ℕ) (num_five_ton_trucks : ℕ) : ℕ :=
  7 * num_seven_ton_trucks + 5 * num_five_ton_trucks

-- Define the minimum freight cost given the conditions.
def minimum_freight_cost := 685

-- The theorem we want to prove.
theorem freight_cost_minimization : ∃ x y : ℕ, total_capacity x y ≥ 73 ∧
  (freight_cost x y = minimum_freight_cost) :=
by
  sorry

end NUMINAMATH_GPT_freight_cost_minimization_l665_66541


namespace NUMINAMATH_GPT_pigeons_percentage_l665_66528

theorem pigeons_percentage (total_birds pigeons sparrows crows doves non_sparrows : ℕ)
  (h_total : total_birds = 100)
  (h_pigeons : pigeons = 40)
  (h_sparrows : sparrows = 20)
  (h_crows : crows = 15)
  (h_doves : doves = 25)
  (h_non_sparrows : non_sparrows = total_birds - sparrows) :
  (pigeons / non_sparrows : ℚ) * 100 = 50 :=
sorry

end NUMINAMATH_GPT_pigeons_percentage_l665_66528


namespace NUMINAMATH_GPT_calculate_total_interest_rate_l665_66513

noncomputable def total_investment : ℝ := 10000
noncomputable def amount_invested_11_percent : ℝ := 3750
noncomputable def amount_invested_9_percent : ℝ := total_investment - amount_invested_11_percent
noncomputable def interest_rate_9_percent : ℝ := 0.09
noncomputable def interest_rate_11_percent : ℝ := 0.11

noncomputable def interest_from_9_percent : ℝ := interest_rate_9_percent * amount_invested_9_percent
noncomputable def interest_from_11_percent : ℝ := interest_rate_11_percent * amount_invested_11_percent

noncomputable def total_interest : ℝ := interest_from_9_percent + interest_from_11_percent

noncomputable def total_interest_rate : ℝ := (total_interest / total_investment) * 100

theorem calculate_total_interest_rate :
  total_interest_rate = 9.75 :=
by 
  sorry

end NUMINAMATH_GPT_calculate_total_interest_rate_l665_66513


namespace NUMINAMATH_GPT_max_xy_value_l665_66563

theorem max_xy_value (x y : ℝ) (h : x^2 + y^2 + 3 * x * y = 2015) : xy <= 403 :=
sorry

end NUMINAMATH_GPT_max_xy_value_l665_66563


namespace NUMINAMATH_GPT_find_a1_in_geometric_sequence_l665_66568

noncomputable def geometric_sequence_first_term (a : ℕ → ℝ) (r : ℝ) (h : ∀ n : ℕ, a (n + 1) = a n * r) : ℝ :=
  a 0

theorem find_a1_in_geometric_sequence (a : ℕ → ℝ) (h_geo : ∀ n : ℕ, a (n + 1) = a n * (1 / 2)) :
  a 2 = 16 → a 3 = 8 → geometric_sequence_first_term a (1 / 2) h_geo = 64 :=
by
  intros h2 h3
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_find_a1_in_geometric_sequence_l665_66568


namespace NUMINAMATH_GPT_unique_solution_of_pair_of_equations_l665_66503

-- Definitions and conditions
def pair_of_equations (x k : ℝ) : Prop :=
  (x^2 + 1 = 4 * x + k)

-- Theorem to prove
theorem unique_solution_of_pair_of_equations :
  ∃ k : ℝ, (∀ x : ℝ, pair_of_equations x k -> x = 2) ∧ k = 0 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_unique_solution_of_pair_of_equations_l665_66503


namespace NUMINAMATH_GPT_exists_a_b_l665_66514

theorem exists_a_b (n : ℕ) (hn : 0 < n) : ∃ a b : ℤ, (4 * a^2 + 9 * b^2 - 1) % n = 0 := by
  sorry

end NUMINAMATH_GPT_exists_a_b_l665_66514


namespace NUMINAMATH_GPT_Polynomial_has_root_l665_66599

noncomputable def P : ℝ → ℝ := sorry

variables (a1 a2 a3 b1 b2 b3 : ℝ)

axiom h1 : a1 * a2 * a3 ≠ 0
axiom h2 : ∀ x : ℝ, P (a1 * x + b1) + P (a2 * x + b2) = P (a3 * x + b3)

theorem Polynomial_has_root : ∃ x : ℝ, P x = 0 :=
sorry

end NUMINAMATH_GPT_Polynomial_has_root_l665_66599


namespace NUMINAMATH_GPT_square_pyramid_sum_l665_66539

def square_pyramid_faces : Nat := 5
def square_pyramid_edges : Nat := 8
def square_pyramid_vertices : Nat := 5

theorem square_pyramid_sum : square_pyramid_faces + square_pyramid_edges + square_pyramid_vertices = 18 := by
  sorry

end NUMINAMATH_GPT_square_pyramid_sum_l665_66539


namespace NUMINAMATH_GPT_intersection_of_A_and_B_range_of_a_l665_66561

open Set

namespace ProofProblem

def A : Set ℝ := {x | -1 ≤ x ∧ x < 3}
def B : Set ℝ := {x | x ≥ 2}
def C (a : ℝ) : Set ℝ := {x | x ≥ a - 1}

theorem intersection_of_A_and_B :
  A ∩ B = {x | 2 ≤ x ∧ x < 3} := 
sorry

theorem range_of_a (a : ℝ) :
  (B ∪ C a) = C a → a ≤ 3 :=
sorry

end ProofProblem

end NUMINAMATH_GPT_intersection_of_A_and_B_range_of_a_l665_66561


namespace NUMINAMATH_GPT_min_copy_paste_actions_l665_66546

theorem min_copy_paste_actions :
  ∀ (n : ℕ), (n ≥ 10) ∧ (n ≤ n) → (2^n ≥ 1001) :=
by sorry

end NUMINAMATH_GPT_min_copy_paste_actions_l665_66546


namespace NUMINAMATH_GPT_area_shaded_quad_correct_l665_66594

-- Define the side lengths of the squares
def side_length_small : ℕ := 3
def side_length_middle : ℕ := 5
def side_length_large : ℕ := 7

-- Define the total base length
def total_base_length : ℕ := side_length_small + side_length_middle + side_length_large

-- The height of triangle T3, equal to the side length of the largest square
def height_T3 : ℕ := side_length_large

-- The height-to-base ratio for each triangle
def height_to_base_ratio : ℚ := height_T3 / total_base_length

-- The heights of T1 and T2
def height_T1 : ℚ := side_length_small * height_to_base_ratio
def height_T2 : ℚ := (side_length_small + side_length_middle) * height_to_base_ratio

-- The height of the trapezoid, which is the side length of the middle square
def trapezoid_height : ℕ := side_length_middle

-- The bases of the trapezoid
def base1 : ℚ := height_T1
def base2 : ℚ := height_T2

-- The area of the trapezoid formula
def area_shaded_quad : ℚ := (trapezoid_height * (base1 + base2)) / 2

-- Assertion that the area of the shaded quadrilateral is equal to 77/6
theorem area_shaded_quad_correct : area_shaded_quad = 77 / 6 := by sorry

end NUMINAMATH_GPT_area_shaded_quad_correct_l665_66594


namespace NUMINAMATH_GPT_remainder_of_product_divided_by_10_l665_66551

theorem remainder_of_product_divided_by_10 :
  let a := 2457
  let b := 6273
  let c := 91409
  (a * b * c) % 10 = 9 :=
by
  sorry

end NUMINAMATH_GPT_remainder_of_product_divided_by_10_l665_66551


namespace NUMINAMATH_GPT_inequality_abc_l665_66562

theorem inequality_abc (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a^2 + 2) * (b^2 + 2) * (c^2 + 2) ≥ 9 * (a * b + b * c + c * a) :=
by
  sorry

end NUMINAMATH_GPT_inequality_abc_l665_66562


namespace NUMINAMATH_GPT_grandson_age_l665_66598

-- Define the ages of Markus, his son, and his grandson
variables (M S G : ℕ)

-- Conditions given in the problem
axiom h1 : M = 2 * S
axiom h2 : S = 2 * G
axiom h3 : M + S + G = 140

-- Theorem to prove that the age of Markus's grandson is 20 years
theorem grandson_age : G = 20 :=
by
  sorry

end NUMINAMATH_GPT_grandson_age_l665_66598


namespace NUMINAMATH_GPT_student_adjustment_l665_66596

noncomputable def binomial (n k : ℕ) : ℕ := Nat.choose n k

noncomputable def permutation (n k : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - k)

theorem student_adjustment : 
  let front_row_size := 4
  let back_row_size := 8
  let total_students := 12
  let num_to_select := 2
  let ways_to_select := binomial back_row_size num_to_select
  let ways_to_permute := permutation (front_row_size + num_to_select) num_to_select
  ways_to_select * ways_to_permute = 840 :=
  by {
    let front_row_size := 4
    let back_row_size := 8
    let total_students := 12
    let num_to_select := 2
    let ways_to_select := binomial back_row_size num_to_select
    let ways_to_permute := permutation (front_row_size + num_to_select) num_to_select
    exact sorry
  }

end NUMINAMATH_GPT_student_adjustment_l665_66596


namespace NUMINAMATH_GPT_range_of_a_l665_66500

def p (a : ℝ) := ∀ x : ℝ, x^2 + 2 * a * x + 4 > 0

def q (a : ℝ) := ∀ x₁ x₂ : ℝ, x₁ < x₂ → -(5 - 2 * a)^x₁ > -(5 - 2 * a)^x₂

theorem range_of_a (a : ℝ) : (p a ∨ q a) → ¬ (p a ∧ q a) → a ≤ -2 := by 
  sorry

end NUMINAMATH_GPT_range_of_a_l665_66500


namespace NUMINAMATH_GPT_rectangle_side_multiple_of_6_l665_66509

theorem rectangle_side_multiple_of_6 (a b : ℕ) (h : ∃ n : ℕ, a * b = n * 6) : a % 6 = 0 ∨ b % 6 = 0 :=
sorry

end NUMINAMATH_GPT_rectangle_side_multiple_of_6_l665_66509


namespace NUMINAMATH_GPT_selina_sold_shirts_l665_66571

/-- Selina's selling problem -/
theorem selina_sold_shirts :
  let pants_price := 5
  let shorts_price := 3
  let shirts_price := 4
  let num_pants := 3
  let num_shorts := 5
  let remaining_money := 30 + (2 * 10)
  let money_from_pants := num_pants * pants_price
  let money_from_shorts := num_shorts * shorts_price
  let total_money_from_pants_and_shorts := money_from_pants + money_from_shorts
  let total_money_from_shirts := remaining_money - total_money_from_pants_and_shorts
  let num_shirts := total_money_from_shirts / shirts_price
  num_shirts = 5 := by
{
  sorry
}

end NUMINAMATH_GPT_selina_sold_shirts_l665_66571


namespace NUMINAMATH_GPT_polynomial_divisibility_l665_66532

theorem polynomial_divisibility (r s : ℝ) :
  (∀ x, 10 * x^4 - 15 * x^3 - 55 * x^2 + 85 * x - 51 = 10 * (x - r)^2 * (x - s)) →
  r = 3 / 2 ∧ s = -5 / 2 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_polynomial_divisibility_l665_66532


namespace NUMINAMATH_GPT_probability_of_other_note_being_counterfeit_l665_66548

def total_notes := 20
def counterfeit_notes := 5

-- Binomial coefficient (n choose k)
noncomputable def binom (n k : ℕ) : ℚ := n.choose k

-- Probability of event A: both notes are counterfeit
noncomputable def P_A : ℚ :=
  binom counterfeit_notes 2 / binom total_notes 2

-- Probability of event B: at least one note is counterfeit
noncomputable def P_B : ℚ :=
  (binom counterfeit_notes 2 + binom counterfeit_notes 1 * binom (total_notes - counterfeit_notes) 1) / binom total_notes 2

-- Conditional probability P(A|B)
noncomputable def P_A_given_B : ℚ :=
  P_A / P_B

theorem probability_of_other_note_being_counterfeit :
  P_A_given_B = 2/17 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_other_note_being_counterfeit_l665_66548


namespace NUMINAMATH_GPT_conversion_proofs_l665_66518

-- Define the necessary constants for unit conversion
def cm_to_dm2 (cm2: ℚ) : ℚ := cm2 / 100
def m3_to_dm3 (m3: ℚ) : ℚ := m3 * 1000
def dm3_to_liters (dm3: ℚ) : ℚ := dm3
def liters_to_ml (liters: ℚ) : ℚ := liters * 1000

theorem conversion_proofs :
  (cm_to_dm2 628 = 6.28) ∧
  (m3_to_dm3 4.5 = 4500) ∧
  (dm3_to_liters 3.6 = 3.6) ∧
  (liters_to_ml 0.6 = 600) :=
by
  sorry

end NUMINAMATH_GPT_conversion_proofs_l665_66518


namespace NUMINAMATH_GPT_Margo_James_pairs_probability_l665_66554

def total_students : ℕ := 32
def Margo_pairs_prob : ℚ := 1 / 31
def James_pairs_prob : ℚ := 1 / 30
def total_prob : ℚ := Margo_pairs_prob * James_pairs_prob

theorem Margo_James_pairs_probability :
  total_prob = 1 / 930 := 
by
  -- sorry allows us to skip the proof steps, only statement needed
  sorry

end NUMINAMATH_GPT_Margo_James_pairs_probability_l665_66554


namespace NUMINAMATH_GPT_value_of_e_over_f_l665_66525

theorem value_of_e_over_f 
    (a b c d e f : ℝ) 
    (h1 : a * b * c = 1.875 * d * e * f)
    (h2 : a / b = 5 / 2)
    (h3 : b / c = 1 / 2)
    (h4 : c / d = 1)
    (h5 : d / e = 3 / 2) : 
    e / f = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_value_of_e_over_f_l665_66525


namespace NUMINAMATH_GPT_no_solution_natural_p_q_r_l665_66558

theorem no_solution_natural_p_q_r :
  ¬ ∃ (p q r : ℕ), 2^p + 5^q = 19^r := sorry

end NUMINAMATH_GPT_no_solution_natural_p_q_r_l665_66558


namespace NUMINAMATH_GPT_geometric_sequence_fourth_term_l665_66595

theorem geometric_sequence_fourth_term (x : ℝ) (r : ℝ) 
  (h1 : 3 * x + 3 = r * x)
  (h2 : 6 * x + 6 = r * (3 * x + 3)) :
  x = -3 ∧ r = 2 → (x * r^3 = -24) :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_fourth_term_l665_66595


namespace NUMINAMATH_GPT_cube_root_of_sum_powers_l665_66508

noncomputable def cube_root (x : ℝ) : ℝ := x ^ (1 / 3)

theorem cube_root_of_sum_powers :
  cube_root (2^7 + 2^7 + 2^7) = 4 * cube_root 2 :=
by
  sorry

end NUMINAMATH_GPT_cube_root_of_sum_powers_l665_66508


namespace NUMINAMATH_GPT_ternary_to_decimal_l665_66533

theorem ternary_to_decimal (k : ℕ) (hk : k > 0) : (1 * 3^3 + k * 3^1 + 2 = 35) → k = 2 :=
by
  sorry

end NUMINAMATH_GPT_ternary_to_decimal_l665_66533


namespace NUMINAMATH_GPT_max_square_plots_l665_66549

theorem max_square_plots (length width available_fencing : ℕ) 
(h : length = 30 ∧ width = 60 ∧ available_fencing = 2500) : 
  ∃ n : ℕ, n = 72 ∧ ∀ s : ℕ, ((30 * (60 / s - 1)) + (60 * (30 / s - 1)) ≤ 2500) → ((30 / s) * (60 / s) = n) := by
  sorry

end NUMINAMATH_GPT_max_square_plots_l665_66549


namespace NUMINAMATH_GPT_avg_salary_l665_66544

-- Conditions as definitions
def number_of_technicians : Nat := 7
def salary_per_technician : Nat := 10000
def number_of_workers : Nat := 14
def salary_per_non_technician : Nat := 6000

-- Total salary of technicians
def total_salary_technicians : Nat := number_of_technicians * salary_per_technician

-- Number of non-technicians
def number_of_non_technicians : Nat := number_of_workers - number_of_technicians

-- Total salary of non-technicians
def total_salary_non_technicians : Nat := number_of_non_technicians * salary_per_non_technician

-- Total salary
def total_salary_all_workers : Nat := total_salary_technicians + total_salary_non_technicians

-- Average salary of all workers
def avg_salary_all_workers : Nat := total_salary_all_workers / number_of_workers

-- Theorem to prove
theorem avg_salary (A : Nat) (h : A = avg_salary_all_workers) : A = 8000 := by
  sorry

end NUMINAMATH_GPT_avg_salary_l665_66544


namespace NUMINAMATH_GPT_more_birds_than_nests_l665_66505

theorem more_birds_than_nests (birds nests : Nat) (h_birds : birds = 6) (h_nests : nests = 3) : birds - nests = 3 :=
by
  sorry

end NUMINAMATH_GPT_more_birds_than_nests_l665_66505


namespace NUMINAMATH_GPT_mean_first_set_l665_66519

noncomputable def mean (s : List ℚ) : ℚ := s.sum / s.length

theorem mean_first_set (x : ℚ) (h : mean [128, 255, 511, 1023, x] = 423) :
  mean [28, x, 42, 78, 104] = 90 :=
sorry

end NUMINAMATH_GPT_mean_first_set_l665_66519


namespace NUMINAMATH_GPT_plane_equation_correct_l665_66573

def plane_equation (x y z : ℝ) : ℝ := 10 * x - 5 * y + 4 * z - 141

noncomputable def gcd (a b c d : ℤ) : ℤ := Int.gcd (Int.gcd a b) (Int.gcd c d)

theorem plane_equation_correct :
  (∀ x y z, plane_equation x y z = 0 ↔ 10 * x - 5 * y + 4 * z - 141 = 0)
  ∧ gcd 10 (-5) 4 (-141) = 1
  ∧ 10 > 0 := by
  sorry

end NUMINAMATH_GPT_plane_equation_correct_l665_66573


namespace NUMINAMATH_GPT_sum_a1_a4_l665_66534

variables (S : ℕ → ℕ) (a : ℕ → ℕ)

-- Define the sum of the first n terms of the sequence
def sum_seq (n : ℕ) : ℕ := n^2 + n + 1

-- Define the individual terms of the sequence
def term_seq (n : ℕ) : ℕ :=
if n = 1 then sum_seq 1 else sum_seq n - sum_seq (n - 1)

-- Prove that the sum of the first and fourth terms equals 11
theorem sum_a1_a4 : 
  (term_seq 1) + (term_seq 4) = 11 :=
by
  -- to be completed with proof steps
  sorry

end NUMINAMATH_GPT_sum_a1_a4_l665_66534


namespace NUMINAMATH_GPT_crayons_in_drawer_before_l665_66578

theorem crayons_in_drawer_before (m c : ℕ) (h1 : m = 3) (h2 : c = 10) : c - m = 7 := 
  sorry

end NUMINAMATH_GPT_crayons_in_drawer_before_l665_66578


namespace NUMINAMATH_GPT_sin_B_value_l665_66555

variable {A B C : Real}
variable {a b c : Real}
variable {sin_A sin_B sin_C : Real}

-- Given conditions as hypotheses
axiom h1 : c = 2 * a
axiom h2 : b * sin_B - a * sin_A = (1 / 2) * a * sin_C

-- The statement to prove
theorem sin_B_value : sin_B = Real.sqrt 7 / 4 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_sin_B_value_l665_66555


namespace NUMINAMATH_GPT_passengers_at_station_in_an_hour_l665_66584

-- Define the conditions
def train_interval_minutes := 5
def passengers_off_per_train := 200
def passengers_on_per_train := 320

-- Define the time period we're considering
def time_period_minutes := 60

-- Calculate the expected values based on conditions
def expected_trains_per_hour := time_period_minutes / train_interval_minutes
def expected_passengers_off_per_hour := passengers_off_per_train * expected_trains_per_hour
def expected_passengers_on_per_hour := passengers_on_per_train * expected_trains_per_hour
def expected_total_passengers := expected_passengers_off_per_hour + expected_passengers_on_per_hour

theorem passengers_at_station_in_an_hour :
  expected_total_passengers = 6240 :=
by
  -- Structure of the proof omitted. Just ensuring conditions and expected value defined.
  sorry

end NUMINAMATH_GPT_passengers_at_station_in_an_hour_l665_66584


namespace NUMINAMATH_GPT_limit_at_2_l665_66521

noncomputable def delta (ε : ℝ) : ℝ := ε / 3

theorem limit_at_2 (ε : ℝ) (hε : ε > 0) : 
  ∃ δ > 0, ∀ x : ℝ, (0 < |x - 2| ∧ |x - 2| < δ) → |(3 * x^2 - 5 * x - 2) / (x - 2) - 7| < ε :=
by
  let δ := delta ε
  have hδ : δ > 0 := by
    sorry
  use δ, hδ
  intros x hx
  sorry

end NUMINAMATH_GPT_limit_at_2_l665_66521


namespace NUMINAMATH_GPT_inequality_proof_l665_66559

theorem inequality_proof (x y : ℝ) (hx: 0 < x) (hy: 0 < y) : 
  1 ≤ (x + y) * (x^3 + y^3) / (x^2 + y^2)^2 ∧ 
  (x + y) * (x^3 + y^3) / (x^2 + y^2)^2 ≤ 9 / 8 := 
by 
  sorry

end NUMINAMATH_GPT_inequality_proof_l665_66559


namespace NUMINAMATH_GPT_right_triangle_side_length_l665_66517

theorem right_triangle_side_length (a c b : ℕ) (h₁ : a = 6) (h₂ : c = 10) (h₃ : c * c = a * a + b * b) : b = 8 :=
by {
  sorry
}

end NUMINAMATH_GPT_right_triangle_side_length_l665_66517


namespace NUMINAMATH_GPT_intersection_point_of_lines_l665_66515

theorem intersection_point_of_lines : ∃ (x y : ℝ), x + y = 5 ∧ x - y = 1 ∧ x = 3 ∧ y = 2 :=
by
  sorry

end NUMINAMATH_GPT_intersection_point_of_lines_l665_66515


namespace NUMINAMATH_GPT_remaining_movies_to_watch_l665_66511

theorem remaining_movies_to_watch (total_movies watched_movies remaining_movies : ℕ) 
  (h1 : total_movies = 8) 
  (h2 : watched_movies = 4) 
  (h3 : remaining_movies = total_movies - watched_movies) : 
  remaining_movies = 4 := 
by
  sorry

end NUMINAMATH_GPT_remaining_movies_to_watch_l665_66511


namespace NUMINAMATH_GPT_circles_are_externally_tangent_l665_66542

-- Conditions given in the problem
def r1 (r2 : ℝ) : Prop := ∃ r1 : ℝ, r1 * r2 = 10 ∧ r1 + r2 = 7
def distance := 7

-- The positional relationship proof problem statement
theorem circles_are_externally_tangent (r1 r2 : ℝ) (h : r1 * r2 = 10 ∧ r1 + r2 = 7) (d : ℝ) (h_d : d = distance) : 
  d = r1 + r2 :=
sorry

end NUMINAMATH_GPT_circles_are_externally_tangent_l665_66542


namespace NUMINAMATH_GPT_johnny_weekly_earnings_l665_66592

-- Define the conditions mentioned in the problem.
def number_of_dogs_at_once : ℕ := 3
def thirty_minute_walk_payment : ℝ := 15
def sixty_minute_walk_payment : ℝ := 20
def work_hours_per_day : ℝ := 4
def sixty_minute_walks_needed_per_day : ℕ := 6
def work_days_per_week : ℕ := 5

-- Prove Johnny's weekly earnings given the conditions
theorem johnny_weekly_earnings :
  let sixty_minute_walks_per_day := sixty_minute_walks_needed_per_day / number_of_dogs_at_once
  let sixty_minute_earnings_per_day := sixty_minute_walks_per_day * number_of_dogs_at_once * sixty_minute_walk_payment
  let remaining_hours_per_day := work_hours_per_day - sixty_minute_walks_per_day
  let thirty_minute_walks_per_day := remaining_hours_per_day * 2 -- each 30-minute walk takes 0.5 hours
  let thirty_minute_earnings_per_day := thirty_minute_walks_per_day * number_of_dogs_at_once * thirty_minute_walk_payment
  let daily_earnings := sixty_minute_earnings_per_day + thirty_minute_earnings_per_day
  let weekly_earnings := daily_earnings * work_days_per_week
  weekly_earnings = 1500 :=
by
  sorry

end NUMINAMATH_GPT_johnny_weekly_earnings_l665_66592


namespace NUMINAMATH_GPT_share_difference_l665_66535

variables {x : ℕ}

theorem share_difference (h1: 12 * x - 7 * x = 5000) : 7 * x - 3 * x = 4000 :=
by
  sorry

end NUMINAMATH_GPT_share_difference_l665_66535


namespace NUMINAMATH_GPT_problem_sequence_k_term_l665_66557

theorem problem_sequence_k_term (a : ℕ → ℤ) (S : ℕ → ℤ) (h₀ : ∀ n, S n = n^2 - 9 * n)
    (h₁ : ∀ n, a n = S n - S (n - 1)) (h₂ : 5 < a 8 ∧ a 8 < 8) : 8 = 8 :=
sorry

end NUMINAMATH_GPT_problem_sequence_k_term_l665_66557


namespace NUMINAMATH_GPT_false_converse_implication_l665_66538

theorem false_converse_implication : ∃ x : ℝ, (0 < x) ∧ (x - 3 ≤ 0) := by
  sorry

end NUMINAMATH_GPT_false_converse_implication_l665_66538


namespace NUMINAMATH_GPT_smallest_common_multiple_of_10_11_18_l665_66516

theorem smallest_common_multiple_of_10_11_18 : 
  ∃ (n : ℕ), (n % 10 = 0) ∧ (n % 11 = 0) ∧ (n % 18 = 0) ∧ (n = 990) :=
by
  sorry

end NUMINAMATH_GPT_smallest_common_multiple_of_10_11_18_l665_66516


namespace NUMINAMATH_GPT_least_number_to_add_for_divisibility_by_11_l665_66545

theorem least_number_to_add_for_divisibility_by_11 : ∃ k : ℕ, 11002 + k ≡ 0 [MOD 11] ∧ k = 9 := by
  sorry

end NUMINAMATH_GPT_least_number_to_add_for_divisibility_by_11_l665_66545


namespace NUMINAMATH_GPT_total_female_students_l665_66552

def total_students : ℕ := 1600
def sample_size : ℕ := 200
def fewer_girls : ℕ := 10

theorem total_female_students (x : ℕ) (sampled_girls sampled_boys : ℕ) (h_total_sample : sampled_girls + sampled_boys = sample_size)
                             (h_fewer_girls : sampled_girls + fewer_girls = sampled_boys) :
  sampled_girls * 8 = 760 :=
by
  sorry

end NUMINAMATH_GPT_total_female_students_l665_66552


namespace NUMINAMATH_GPT_middle_number_l665_66570

theorem middle_number (a b c : ℕ) (h1 : a < b) (h2 : b < c) 
  (h3 : a + b = 18) (h4 : a + c = 23) (h5 : b + c = 27) : b = 11 := by
  sorry

end NUMINAMATH_GPT_middle_number_l665_66570


namespace NUMINAMATH_GPT_a_range_iff_l665_66536

theorem a_range_iff (a x : ℝ) (h1 : x < 3) (h2 : (a - 1) * x < a + 3) : 
  1 ≤ a ∧ a < 3 := 
by
  sorry

end NUMINAMATH_GPT_a_range_iff_l665_66536


namespace NUMINAMATH_GPT_compute_expression_l665_66565

theorem compute_expression : (5 + 9)^2 + Real.sqrt (5^2 + 9^2) = 196 + Real.sqrt 106 := 
by sorry

end NUMINAMATH_GPT_compute_expression_l665_66565


namespace NUMINAMATH_GPT_smallest_value_of_expression_l665_66566

noncomputable def f (x : ℝ) : ℝ := x^4 + 14*x^3 + 52*x^2 + 56*x + 16

theorem smallest_value_of_expression :
  ∀ z : Fin 4 → ℝ, (∀ i, f (z i) = 0) → 
  ∃ (a b c d : Fin 4), a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ a ∧ d ≠ b ∧ a ≠ c ∧ 
  |(z a * z b) + (z c * z d)| = 8 :=
by
  sorry

end NUMINAMATH_GPT_smallest_value_of_expression_l665_66566


namespace NUMINAMATH_GPT_vector_dot_product_l665_66581

-- Definitions
def vec_a : ℝ × ℝ := (1, 3)
def vec_b : ℝ × ℝ := (-2, -1)

-- Theorem to prove
theorem vector_dot_product : 
  ((vec_a.1 + vec_b.1, vec_a.2 + vec_b.2) : ℝ × ℝ) • (2 * vec_a.1 + vec_b.1, 2 * vec_a.2 + vec_b.2) = 10 :=
by
  sorry

end NUMINAMATH_GPT_vector_dot_product_l665_66581


namespace NUMINAMATH_GPT_smallest_angle_WYZ_l665_66587

-- Define the given angle measures.
def angle_XYZ : ℝ := 40
def angle_XYW : ℝ := 15

-- The theorem statement proving the smallest possible degree measure for ∠WYZ
theorem smallest_angle_WYZ : angle_XYZ - angle_XYW = 25 :=
by
  -- Add the proof here
  sorry

end NUMINAMATH_GPT_smallest_angle_WYZ_l665_66587


namespace NUMINAMATH_GPT_true_propositions_l665_66504

def p : Prop :=
  ∀ a b : ℝ, (a > 2 ∧ b > 2) → a + b > 4

def q : Prop :=
  ¬ ∃ x : ℝ, x^2 - x > 0 → ∀ x : ℝ, x^2 - x ≤ 0

theorem true_propositions :
  (¬ p ∨ ¬ q) ∧ (p ∨ ¬ q) := by
  sorry

end NUMINAMATH_GPT_true_propositions_l665_66504


namespace NUMINAMATH_GPT_part1_part2_part3_l665_66580

section Part1

variables (a b : Real)

theorem part1 : 2 * (a + b)^2 - 8 * (a + b)^2 + 3 * (a + b)^2 = -3 * (a + b)^2 :=
by
  sorry

end Part1

section Part2

variables (x y : Real)

theorem part2 (h : x^2 + 2 * y = 4) : -3 * x^2 - 6 * y + 17 = 5 :=
by
  sorry

end Part2

section Part3

variables (a b c d : Real)

theorem part3 (h1 : a - 3 * b = 3) (h2 : 2 * b - c = -5) (h3 : c - d = 9) :
  (a - c) + (2 * b - d) - (2 * b - c) = 7 :=
by
  sorry

end Part3

end NUMINAMATH_GPT_part1_part2_part3_l665_66580


namespace NUMINAMATH_GPT_solve_quadratics_l665_66510

theorem solve_quadratics (p q u v : ℤ)
  (h1 : p ≠ 0 ∧ q ≠ 0 ∧ p ≠ q)
  (h2 : u ≠ 0 ∧ v ≠ 0 ∧ u ≠ v)
  (h3 : p + q = -u)
  (h4 : pq = -v)
  (h5 : u + v = -p)
  (h6 : uv = -q) :
  p = -1 ∧ q = 2 ∧ u = -1 ∧ v = 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_solve_quadratics_l665_66510


namespace NUMINAMATH_GPT_primes_with_consecutives_l665_66520

-- Define what it means for a number to be prime
def is_prime (n : Nat) := n > 1 ∧ ∀ m : Nat, m > 1 → m < n → ¬ (n % m = 0)

-- Define the main theorem to prove
theorem primes_with_consecutives (p : Nat) : is_prime p ∧ is_prime (p + 2) ∧ is_prime (p + 4) → p = 3 :=
by
  sorry

end NUMINAMATH_GPT_primes_with_consecutives_l665_66520


namespace NUMINAMATH_GPT_simplify_expression_l665_66579

variable (q : ℚ)

theorem simplify_expression :
  (2 * q^3 - 7 * q^2 + 3 * q - 4) + (5 * q^2 - 4 * q + 8) = 2 * q^3 - 2 * q^2 - q + 4 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l665_66579


namespace NUMINAMATH_GPT_original_sequence_polynomial_of_degree_3_l665_66583

def is_polynomial_of_degree (u : ℕ → ℤ) (n : ℕ) :=
  ∃ a b c d : ℤ, u n = a * n^3 + b * n^2 + c * n + d

def fourth_difference_is_zero (u : ℕ → ℤ) :=
  ∀ n : ℕ, (u (n + 4) - 4 * u (n + 3) + 6 * u (n + 2) - 4 * u (n + 1) + u n) = 0

theorem original_sequence_polynomial_of_degree_3 (u : ℕ → ℤ)
  (h : fourth_difference_is_zero u) : 
  ∃ (a b c d : ℤ), ∀ n : ℕ, u n = a * n^3 + b * n^2 + c * n + d := sorry

end NUMINAMATH_GPT_original_sequence_polynomial_of_degree_3_l665_66583


namespace NUMINAMATH_GPT_weight_mixture_is_correct_l665_66526

noncomputable def weight_mixture_in_kg (weight_a_per_liter weight_b_per_liter : ℝ)
  (ratio_a ratio_b total_volume_liters weight_conversion : ℝ) : ℝ :=
  let total_parts := ratio_a + ratio_b
  let volume_per_part := total_volume_liters / total_parts
  let volume_a := ratio_a * volume_per_part
  let volume_b := ratio_b * volume_per_part
  let weight_a := volume_a * weight_a_per_liter
  let weight_b := volume_b * weight_b_per_liter
  let total_weight_gm := weight_a + weight_b
  total_weight_gm / weight_conversion

theorem weight_mixture_is_correct :
  weight_mixture_in_kg 900 700 3 2 4 1000 = 3.280 :=
by
  -- Calculation should follow from the def
  sorry

end NUMINAMATH_GPT_weight_mixture_is_correct_l665_66526


namespace NUMINAMATH_GPT_shaded_area_correct_l665_66586

noncomputable def shaded_area (side_large side_small : ℝ) (pi_value : ℝ) : ℝ :=
  let area_large_square := side_large^2
  let area_large_circle := pi_value * (side_large / 2)^2
  let area_large_heart := area_large_square + area_large_circle
  let area_small_square := side_small^2
  let area_small_circle := pi_value * (side_small / 2)^2
  let area_small_heart := area_small_square + area_small_circle
  area_large_heart - area_small_heart

theorem shaded_area_correct : shaded_area 40 20 3.14 = 2142 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_shaded_area_correct_l665_66586


namespace NUMINAMATH_GPT_johnnys_age_l665_66543

theorem johnnys_age (x : ℤ) (h : x + 2 = 2 * (x - 3)) : x = 8 := sorry

end NUMINAMATH_GPT_johnnys_age_l665_66543


namespace NUMINAMATH_GPT_quadratic_general_form_l665_66588

theorem quadratic_general_form (x : ℝ) :
    (x + 3)^2 = x * (3 * x - 1) →
    2 * x^2 - 7 * x - 9 = 0 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_quadratic_general_form_l665_66588


namespace NUMINAMATH_GPT_red_side_probability_l665_66550

theorem red_side_probability
  (num_cards : ℕ)
  (num_black_black : ℕ)
  (num_black_red : ℕ)
  (num_red_red : ℕ)
  (num_red_sides_total : ℕ)
  (num_red_sides_with_red_other_side : ℕ) :
  num_cards = 8 →
  num_black_black = 4 →
  num_black_red = 2 →
  num_red_red = 2 →
  num_red_sides_total = (num_red_red * 2 + num_black_red) →
  num_red_sides_with_red_other_side = (num_red_red * 2) →
  (num_red_sides_with_red_other_side / num_red_sides_total : ℝ) = 2 / 3 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end NUMINAMATH_GPT_red_side_probability_l665_66550


namespace NUMINAMATH_GPT_odd_function_of_power_l665_66577

noncomputable def f (a b x : ℝ) : ℝ := (a - 1) * x ^ b

theorem odd_function_of_power (a b : ℝ) (h : f a b a = 1/2) : 
  ∀ x : ℝ, f a b (-x) = -f a b x := 
by
  sorry

end NUMINAMATH_GPT_odd_function_of_power_l665_66577


namespace NUMINAMATH_GPT_brick_weight_l665_66582

theorem brick_weight (b s : ℕ) (h1 : 5 * b = 4 * s) (h2 : 2 * s = 80) : b = 32 :=
by {
  sorry
}

end NUMINAMATH_GPT_brick_weight_l665_66582


namespace NUMINAMATH_GPT_ticket_sales_total_cost_l665_66529

noncomputable def total_ticket_cost (O B : ℕ) : ℕ :=
  12 * O + 8 * B

theorem ticket_sales_total_cost (O B : ℕ) (h1 : O + B = 350) (h2 : B = O + 90) :
  total_ticket_cost O B = 3320 :=
by
  -- the proof steps calculating the total cost will go here
  sorry

end NUMINAMATH_GPT_ticket_sales_total_cost_l665_66529


namespace NUMINAMATH_GPT_linear_eq_find_m_l665_66502

theorem linear_eq_find_m (m : ℤ) (x : ℝ) 
  (h : (m - 5) * x^(|m| - 4) + 5 = 0) 
  (h_linear : |m| - 4 = 1) 
  (h_nonzero : m - 5 ≠ 0) : m = -5 :=
by
  sorry

end NUMINAMATH_GPT_linear_eq_find_m_l665_66502


namespace NUMINAMATH_GPT_field_dimension_solution_l665_66523

theorem field_dimension_solution (m : ℝ) (h₁ : (3 * m + 10) * (m - 5) = 72) : m = 7 :=
sorry

end NUMINAMATH_GPT_field_dimension_solution_l665_66523


namespace NUMINAMATH_GPT_sets_are_equal_l665_66501

def int : Type := ℤ  -- Redefine integer as ℤ for clarity

def SetA : Set ℤ := {x | ∃ k : ℤ, x = 2 * k + 1}
def SetB : Set ℤ := {x | ∃ k : ℤ, x = 2 * k - 1}

theorem sets_are_equal : SetA = SetB := by
  -- implement the proof here
  sorry

end NUMINAMATH_GPT_sets_are_equal_l665_66501


namespace NUMINAMATH_GPT_complex_number_real_implies_m_is_5_l665_66560

theorem complex_number_real_implies_m_is_5 (m : ℝ) (h : m^2 - 2 * m - 15 = 0) : m = 5 :=
  sorry

end NUMINAMATH_GPT_complex_number_real_implies_m_is_5_l665_66560
