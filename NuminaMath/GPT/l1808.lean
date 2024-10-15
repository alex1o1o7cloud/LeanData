import Mathlib

namespace NUMINAMATH_GPT_inequality_proof_l1808_180855

theorem inequality_proof (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  (2 * x * y) / (x + y) + Real.sqrt ((x ^ 2 + y ^ 2) / 2) ≥ (x + y) / 2 + Real.sqrt (x * y) :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l1808_180855


namespace NUMINAMATH_GPT_part1_part2_l1808_180816

-- Definitions for sets A and B
def A : Set ℝ := {x : ℝ | -3 < x ∧ x < 2}
def B : Set ℝ := {x : ℝ | x < -5 ∨ x > 1}

-- Prove (1): A ∪ B
theorem part1 : A ∪ B = {x : ℝ | x < -5 ∨ x > -3} :=
by
  sorry

-- Prove (2): A ∩ (ℝ \ B)
theorem part2 : A ∩ (Set.compl B) = {x : ℝ | -3 < x ∧ x ≤ 1} :=
by
  sorry

end NUMINAMATH_GPT_part1_part2_l1808_180816


namespace NUMINAMATH_GPT_estimate_fish_in_pond_l1808_180859

theorem estimate_fish_in_pond
  (n m k : ℕ)
  (h_pr: k = 200)
  (h_cr: k = 8)
  (h_m: n = 200):
  n / (m / k) = 5000 := sorry

end NUMINAMATH_GPT_estimate_fish_in_pond_l1808_180859


namespace NUMINAMATH_GPT_gwen_books_collection_l1808_180856

theorem gwen_books_collection :
  let mystery_books := 8 * 6
  let picture_books := 5 * 4
  let science_books := 4 * 7
  let non_fiction_books := 3 * 5
  let lent_mystery_books := 2
  let lent_science_books := 3
  let borrowed_picture_books := 5
  mystery_books - lent_mystery_books + picture_books - borrowed_picture_books + borrowed_picture_books + science_books - lent_science_books + non_fiction_books = 106 := by
  sorry

end NUMINAMATH_GPT_gwen_books_collection_l1808_180856


namespace NUMINAMATH_GPT_only_value_of_k_l1808_180861

def A (k a b : ℕ) : ℚ := (a + b : ℚ) / (a^2 + k^2 * b^2 - k^2 * a * b : ℚ)

theorem only_value_of_k : (∀ a b : ℕ, 0 < a → 0 < b → ¬ (∃ c d : ℕ, 1 < c ∧ A 1 a b = (c : ℚ) / (d : ℚ))) → k = 1 := 
    by sorry  -- proof omitted

-- Note: 'only_value_of_k' states that given the conditions, there is no k > 1 that makes A(k, a, b) a composite number, hence k must be 1.

end NUMINAMATH_GPT_only_value_of_k_l1808_180861


namespace NUMINAMATH_GPT_Steve_bakes_more_apple_pies_l1808_180870

def Steve_bakes (days_apple days_cherry pies_per_day : ℕ) : ℕ :=
  (days_apple * pies_per_day) - (days_cherry * pies_per_day)

theorem Steve_bakes_more_apple_pies :
  Steve_bakes 3 2 12 = 12 :=
by
  sorry

end NUMINAMATH_GPT_Steve_bakes_more_apple_pies_l1808_180870


namespace NUMINAMATH_GPT_incorrect_judgment_D_l1808_180809

theorem incorrect_judgment_D (p q : Prop) (hp : p = (2 + 3 = 5)) (hq : q = (5 < 4)) : 
  ¬((p ∧ q) ∧ (p ∨ q)) := by 
    sorry

end NUMINAMATH_GPT_incorrect_judgment_D_l1808_180809


namespace NUMINAMATH_GPT_sum_is_odd_prob_l1808_180869

-- A type representing the spinner results, which can be either 1, 2, 3 or 4.
inductive SpinnerResult
| one : SpinnerResult
| two : SpinnerResult
| three : SpinnerResult
| four : SpinnerResult

open SpinnerResult

-- Function to determine if a spinner result is odd.
def isOdd (r : SpinnerResult) : Bool :=
  match r with
  | one => true
  | three => true
  | two => false
  | four => false

-- Defining the spinners P, Q, R, and S.
noncomputable def P : SpinnerResult := SpinnerResult.one -- example, could vary
noncomputable def Q : SpinnerResult := SpinnerResult.two -- example, could vary
noncomputable def R : SpinnerResult := SpinnerResult.three -- example, could vary
noncomputable def S : SpinnerResult := SpinnerResult.four -- example, could vary

-- Probability calculation function
def probabilityOddSum : ℚ :=
  let probOdd := 1 / 2
  let probEven := 1 / 2
  let scenario1 := 4 * probOdd * probEven^3
  let scenario2 := 4 * probOdd^3 * probEven
  scenario1 + scenario2

-- The theorem to be stated
theorem sum_is_odd_prob :
  probabilityOddSum = 1 / 2 := by
  sorry

end NUMINAMATH_GPT_sum_is_odd_prob_l1808_180869


namespace NUMINAMATH_GPT_associate_professors_bring_one_chart_l1808_180846

theorem associate_professors_bring_one_chart
(A B C : ℕ) (h1 : 2 * A + B = 7) (h2 : A * C + 2 * B = 11) (h3 : A + B = 6) : C = 1 :=
by sorry

end NUMINAMATH_GPT_associate_professors_bring_one_chart_l1808_180846


namespace NUMINAMATH_GPT_range_of_k_l1808_180882

open BigOperators

theorem range_of_k
  {f : ℝ → ℝ}
  (k : ℝ)
  (h : ∀ x : ℝ, f x = 32 * x - (k + 1) * 3^x + 2)
  (H : ∀ x : ℝ, f x > 0) :
  k < 1 /2 := 
sorry

end NUMINAMATH_GPT_range_of_k_l1808_180882


namespace NUMINAMATH_GPT_min_both_attendees_l1808_180821

-- Defining the parameters and conditions
variable (n : ℕ) -- total number of attendees
variable (glasses name_tags both : ℕ) -- attendees wearing glasses, name tags, and both

-- Conditions provided in the problem
def wearing_glasses_condition (n : ℕ) (glasses : ℕ) : Prop := glasses = n / 3
def wearing_name_tags_condition (n : ℕ) (name_tags : ℕ) : Prop := name_tags = n / 2
def total_attendees_condition (n : ℕ) : Prop := n = 6

-- Theorem to prove the minimum attendees wearing both glasses and name tags is 1
theorem min_both_attendees (n glasses name_tags both : ℕ) (h1 : wearing_glasses_condition n glasses) 
  (h2 : wearing_name_tags_condition n name_tags) (h3 : total_attendees_condition n) : 
  both = 1 :=
sorry

end NUMINAMATH_GPT_min_both_attendees_l1808_180821


namespace NUMINAMATH_GPT_boat_distance_downstream_l1808_180824

theorem boat_distance_downstream (speed_boat_still: ℕ) (speed_stream: ℕ) (time: ℕ)
    (h1: speed_boat_still = 25)
    (h2: speed_stream = 5)
    (h3: time = 4) :
    (speed_boat_still + speed_stream) * time = 120 := 
sorry

end NUMINAMATH_GPT_boat_distance_downstream_l1808_180824


namespace NUMINAMATH_GPT_weight_of_empty_box_l1808_180874

theorem weight_of_empty_box (w12 w8 w : ℝ) (h1 : w12 = 11.48) (h2 : w8 = 8.12) (h3 : ∀ b : ℕ, b > 0 → w = 0.84) :
  w8 - 8 * w = 1.40 :=
by
  sorry

end NUMINAMATH_GPT_weight_of_empty_box_l1808_180874


namespace NUMINAMATH_GPT_isabel_reading_homework_pages_l1808_180803

-- Definitions for the given problem
def num_math_pages := 2
def problems_per_page := 5
def total_problems := 30

-- Calculation based on conditions
def math_problems := num_math_pages * problems_per_page
def reading_problems := total_problems - math_problems

-- The statement to be proven
theorem isabel_reading_homework_pages : (reading_problems / problems_per_page) = 4 :=
by
  -- The proof would go here.
  sorry

end NUMINAMATH_GPT_isabel_reading_homework_pages_l1808_180803


namespace NUMINAMATH_GPT_initial_ratio_of_milk_to_water_l1808_180880

theorem initial_ratio_of_milk_to_water (M W : ℕ) (h1 : M + W = 60) (h2 : 2 * M = W + 60) : M / W = 2 :=
by
  sorry

end NUMINAMATH_GPT_initial_ratio_of_milk_to_water_l1808_180880


namespace NUMINAMATH_GPT_fraction_of_is_l1808_180830

theorem fraction_of_is (a b c d e : ℚ) (h1 : a = 2) (h2 : b = 9) (h3 : c = 3) (h4 : d = 4) (h5 : e = 8/27) :
  (a / b) = e * (c / d) := 
sorry

end NUMINAMATH_GPT_fraction_of_is_l1808_180830


namespace NUMINAMATH_GPT_sandwiches_final_count_l1808_180872

def sandwiches_left (initial : ℕ) (eaten_by_ruth : ℕ) (given_to_brother : ℕ) (eaten_by_first_cousin : ℕ) (eaten_by_other_cousins : ℕ) : ℕ :=
  initial - (eaten_by_ruth + given_to_brother + eaten_by_first_cousin + eaten_by_other_cousins)

theorem sandwiches_final_count :
  sandwiches_left 10 1 2 2 2 = 3 := by
  sorry

end NUMINAMATH_GPT_sandwiches_final_count_l1808_180872


namespace NUMINAMATH_GPT_sum_of_real_solutions_l1808_180885

theorem sum_of_real_solutions :
  ∀ x : ℝ, (x - 3) / (x^2 + 5 * x + 2) = (x - 6) / (x^2 - 11 * x) →
  (∃ r1 r2 : ℝ, r1 + r2 = 46 / 13) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_real_solutions_l1808_180885


namespace NUMINAMATH_GPT_multiples_of_7_with_unit_digit_7_and_lt_150_l1808_180888

/--
Given a positive integer n,
Let \(7n\) be a multiple of 7,
\(7n < 150\),
and \(7n \mod 10 = 7\),
prove there are 11 such multiples of 7.
-/
theorem multiples_of_7_with_unit_digit_7_and_lt_150 :
  ∃! (n : ℕ), n ≤ 11 ∧ ∀ (m : ℕ), (1 ≤ m ∧ m ≤ n) → (7 * (2 * m - 1) < 150 ∧ (7 * (2 * m - 1)) % 10 = 7) := by
sorry

end NUMINAMATH_GPT_multiples_of_7_with_unit_digit_7_and_lt_150_l1808_180888


namespace NUMINAMATH_GPT_intersection_of_P_and_Q_l1808_180835
-- Import the entire math library

-- Define the conditions for sets P and Q
def P := {x : ℝ | 1 ≤ x ∧ x ≤ 3}
def Q := {x : ℝ | (x - 1)^2 ≤ 4}

-- Define the theorem to prove that P ∩ Q = {x | 1 ≤ x ∧ x ≤ 3}
theorem intersection_of_P_and_Q : P ∩ Q = {x | 1 ≤ x ∧ x ≤ 3} :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_intersection_of_P_and_Q_l1808_180835


namespace NUMINAMATH_GPT_share_of_a_120_l1808_180879

theorem share_of_a_120 (A B C : ℝ) 
  (h1 : A = (2 / 3) * (B + C)) 
  (h2 : B = (6 / 9) * (A + C)) 
  (h3 : A + B + C = 300) : 
  A = 120 := 
by 
  sorry

end NUMINAMATH_GPT_share_of_a_120_l1808_180879


namespace NUMINAMATH_GPT_compare_logarithms_l1808_180814

theorem compare_logarithms (a b c : ℝ) (h1 : a = Real.log 2 / Real.log 3) 
                           (h2 : b = (Real.log 2 / Real.log 3)^2) 
                           (h3 : c = Real.log (2/3) / Real.log 4) : c < b ∧ b < a :=
by
  sorry

end NUMINAMATH_GPT_compare_logarithms_l1808_180814


namespace NUMINAMATH_GPT_travel_distance_l1808_180883

theorem travel_distance (x t : ℕ) (h : t = 14400) (h_eq : 12 * x + 12 * (2 * x) = t) : x = 400 :=
by
  sorry

end NUMINAMATH_GPT_travel_distance_l1808_180883


namespace NUMINAMATH_GPT_cos_arcsin_l1808_180897

theorem cos_arcsin (h : (7:ℝ) / 25 ≤ 1) : Real.cos (Real.arcsin ((7:ℝ) / 25)) = (24:ℝ) / 25 := by
  -- Proof to be provided
  sorry

end NUMINAMATH_GPT_cos_arcsin_l1808_180897


namespace NUMINAMATH_GPT_combined_volume_cone_hemisphere_cylinder_l1808_180873

theorem combined_volume_cone_hemisphere_cylinder (r h : ℝ)
  (vol_cylinder : ℝ) (vol_cone : ℝ) (vol_hemisphere : ℝ)
  (H1 : vol_cylinder = 72 * π)
  (H2 : vol_cylinder = π * r^2 * h)
  (H3 : vol_cone = (1/3) * π * r^2 * h)
  (H4 : vol_hemisphere = (2/3) * π * r^3)
  (H5 : vol_cylinder = vol_cone + vol_hemisphere) :
  vol_cylinder = 72 * π :=
by
  sorry

end NUMINAMATH_GPT_combined_volume_cone_hemisphere_cylinder_l1808_180873


namespace NUMINAMATH_GPT_rectangle_area_l1808_180884

-- Define the problem in Lean
theorem rectangle_area (l w : ℕ) (h1 : l = 4 * w) (h2 : 2 * l + 2 * w = 200) : l * w = 1600 := by
  sorry

end NUMINAMATH_GPT_rectangle_area_l1808_180884


namespace NUMINAMATH_GPT_probability_A_inter_B_l1808_180826

def set_A (x : ℝ) : Prop := -1 < x ∧ x < 5
def set_B (x : ℝ) : Prop := (x-2)/(3-x) > 0

def A_inter_B (x : ℝ) : Prop := set_A x ∧ set_B x

theorem probability_A_inter_B :
  let length_A := 5 - (-1)
  let length_A_inter_B := 3 - 2 
  length_A > 0 ∧ length_A_inter_B > 0 →
  length_A_inter_B / length_A = 1 / 6 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_probability_A_inter_B_l1808_180826


namespace NUMINAMATH_GPT_birds_more_than_half_sunflower_seeds_l1808_180845

theorem birds_more_than_half_sunflower_seeds :
  ∃ (n : ℕ), n = 3 ∧ ((4 / 5)^n * (2 / 5) + (2 / 5) > 1 / 2) :=
by
  sorry

end NUMINAMATH_GPT_birds_more_than_half_sunflower_seeds_l1808_180845


namespace NUMINAMATH_GPT_raja_monthly_income_l1808_180802

noncomputable def monthly_income (household_percentage clothes_percentage medicines_percentage savings : ℝ) : ℝ :=
  let spending_percentage := household_percentage + clothes_percentage + medicines_percentage
  let savings_percentage := 1 - spending_percentage
  savings / savings_percentage

theorem raja_monthly_income :
  monthly_income 0.35 0.20 0.05 15000 = 37500 :=
by
  sorry

end NUMINAMATH_GPT_raja_monthly_income_l1808_180802


namespace NUMINAMATH_GPT_area_of_intersection_of_two_circles_l1808_180877

open Real

noncomputable def area_intersection (r : ℝ) (c1 c2 : ℝ × ℝ) : ℝ :=
  let quarter_circle_area := (1/4) * π * r^2
  let triangle_area := (1/2) * r^2
  let segment_area := quarter_circle_area - triangle_area
  2 * segment_area

theorem area_of_intersection_of_two_circles :
  area_intersection 3 (3, 0) (0, 3) = (9 * π - 18) / 2 :=
by
  -- This will be proven by the steps of the provided solution.
  sorry

end NUMINAMATH_GPT_area_of_intersection_of_two_circles_l1808_180877


namespace NUMINAMATH_GPT_opposite_of_five_l1808_180813

theorem opposite_of_five : -5 = -5 :=
by
sorry

end NUMINAMATH_GPT_opposite_of_five_l1808_180813


namespace NUMINAMATH_GPT_target_annual_revenue_l1808_180858

-- Given conditions as definitions
def monthly_sales : ℕ := 4000
def additional_sales : ℕ := 1000

-- The proof problem in Lean statement form
theorem target_annual_revenue : (monthly_sales + additional_sales) * 12 = 60000 := by
  sorry

end NUMINAMATH_GPT_target_annual_revenue_l1808_180858


namespace NUMINAMATH_GPT_find_point_N_l1808_180850

-- Definition of symmetrical reflection across the x-axis
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

-- Given condition
def point_M : ℝ × ℝ := (1, 3)

-- Theorem statement
theorem find_point_N : reflect_x point_M = (1, -3) :=
by
  sorry

end NUMINAMATH_GPT_find_point_N_l1808_180850


namespace NUMINAMATH_GPT_triangle_expression_l1808_180867

open Real

variable (D E F : ℝ)
variable (DE DF EF : ℝ)

-- conditions
def triangleDEF : Prop := DE = 7 ∧ DF = 9 ∧ EF = 8

theorem triangle_expression (h : triangleDEF DE DF EF) :
  (cos ((D - E)/2) / sin (F/2) - sin ((D - E)/2) / cos (F/2)) = 81/28 :=
by
  have h1 : DE = 7 := h.1
  have h2 : DF = 9 := h.2.1
  have h3 : EF = 8 := h.2.2
  sorry

end NUMINAMATH_GPT_triangle_expression_l1808_180867


namespace NUMINAMATH_GPT_price_change_on_eggs_and_apples_l1808_180887

theorem price_change_on_eggs_and_apples :
  let initial_egg_price := 1.00
  let initial_apple_price := 1.00
  let egg_drop_percent := 0.10
  let apple_increase_percent := 0.02
  let new_egg_price := initial_egg_price * (1 - egg_drop_percent)
  let new_apple_price := initial_apple_price * (1 + apple_increase_percent)
  let initial_total := initial_egg_price + initial_apple_price
  let new_total := new_egg_price + new_apple_price
  let percent_change := ((new_total - initial_total) / initial_total) * 100
  percent_change = -4 :=
by
  sorry

end NUMINAMATH_GPT_price_change_on_eggs_and_apples_l1808_180887


namespace NUMINAMATH_GPT_male_worker_ants_percentage_l1808_180851

theorem male_worker_ants_percentage 
  (total_ants : ℕ) 
  (half_ants : ℕ) 
  (female_worker_ants : ℕ) 
  (h1 : total_ants = 110) 
  (h2 : half_ants = total_ants / 2) 
  (h3 : female_worker_ants = 44) :
  (half_ants - female_worker_ants) * 100 / half_ants = 20 := by
  sorry

end NUMINAMATH_GPT_male_worker_ants_percentage_l1808_180851


namespace NUMINAMATH_GPT_apples_count_l1808_180829

def mangoes_oranges_apples_ratio (mangoes oranges apples : Nat) : Prop :=
  mangoes / 10 = oranges / 2 ∧ mangoes / 10 = apples / 3

theorem apples_count (mangoes oranges apples : Nat) (h_ratio : mangoes_oranges_apples_ratio mangoes oranges apples) (h_mangoes : mangoes = 120) : apples = 36 :=
by
  sorry

end NUMINAMATH_GPT_apples_count_l1808_180829


namespace NUMINAMATH_GPT_tan_pi_over_4_plus_alpha_eq_two_l1808_180865

theorem tan_pi_over_4_plus_alpha_eq_two
  (α : ℂ) 
  (h : Complex.tan ((π / 4) + α) = 2) : 
  (1 / (2 * Complex.sin α * Complex.cos α + (Complex.cos α)^2)) = (2 / 3) :=
by
  sorry

end NUMINAMATH_GPT_tan_pi_over_4_plus_alpha_eq_two_l1808_180865


namespace NUMINAMATH_GPT_andrew_ruined_planks_l1808_180863

variable (b L k g h leftover plank_total ruin_bedroom ruin_guest : ℕ)

-- Conditions
def bedroom_planks := b
def living_room_planks := L
def kitchen_planks := k
def guest_bedroom_planks := g
def hallway_planks := h
def planks_leftover := leftover

-- Values
axiom bedroom_planks_val : bedroom_planks = 8
axiom living_room_planks_val : living_room_planks = 20
axiom kitchen_planks_val : kitchen_planks = 11
axiom guest_bedroom_planks_val : guest_bedroom_planks = bedroom_planks - 2
axiom hallway_planks_val : hallway_planks = 4
axiom planks_leftover_val : planks_leftover = 6

-- Total planks used and total planks had
def total_planks_used := bedroom_planks + living_room_planks + kitchen_planks + guest_bedroom_planks + (2 * hallway_planks)
def total_planks_had := total_planks_used + planks_leftover

-- Planks ruined
def planks_ruined_in_bedroom := ruin_bedroom
def planks_ruined_in_guest_bedroom := ruin_guest

-- Theorem to be proven
theorem andrew_ruined_planks :
  (planks_ruined_in_bedroom = total_planks_had - total_planks_used) ∧
  (planks_ruined_in_guest_bedroom = planks_ruined_in_bedroom) :=
by
  sorry

end NUMINAMATH_GPT_andrew_ruined_planks_l1808_180863


namespace NUMINAMATH_GPT_weight_of_second_triangle_l1808_180828

theorem weight_of_second_triangle :
  let side_len1 := 4
  let density1 := 0.9
  let weight1 := 10.8
  let side_len2 := 6
  let density2 := 1.2
  let weight2 := 18.7
  let area1 := (side_len1 ^ 2 * Real.sqrt 3) / 4
  let area2 := (side_len2 ^ 2 * Real.sqrt 3) / 4
  let calc_weight1 := area1 * density1
  let calc_weight2 := area2 * density2
  calc_weight1 = weight1 → calc_weight2 = weight2 := 
by
  intros
  -- Proof logic goes here
  sorry

end NUMINAMATH_GPT_weight_of_second_triangle_l1808_180828


namespace NUMINAMATH_GPT_max_distance_circle_ellipse_l1808_180841

theorem max_distance_circle_ellipse:
  (∀ P Q : ℝ × ℝ, 
     (P.1^2 + (P.2 - 3)^2 = 1 / 4) → 
     (Q.1^2 + 4 * Q.2^2 = 4) → 
     ∃ Q_max : ℝ × ℝ, 
         Q_max = (0, -1) ∧ 
         (∀ P : ℝ × ℝ, P.1^2 + (P.2 - 3)^2 = 1 / 4 →
         |dist P Q_max| = 9 / 2)) := 
sorry

end NUMINAMATH_GPT_max_distance_circle_ellipse_l1808_180841


namespace NUMINAMATH_GPT_ratio_Bill_to_Bob_l1808_180811

-- Define the shares
def Bill_share : ℕ := 300
def Bob_share : ℕ := 900

-- The theorem statement
theorem ratio_Bill_to_Bob : Bill_share / Bob_share = 1 / 3 := by
  sorry

end NUMINAMATH_GPT_ratio_Bill_to_Bob_l1808_180811


namespace NUMINAMATH_GPT_math_problem_l1808_180818

theorem math_problem (x : ℝ) (h : x = 0.18 * 4750) : 1.5 * x = 1282.5 :=
by
  sorry

end NUMINAMATH_GPT_math_problem_l1808_180818


namespace NUMINAMATH_GPT_intersection_equivalence_l1808_180823

open Set

noncomputable def U : Set ℤ := {-2, -1, 0, 1, 2}
noncomputable def M : Set ℤ := {-1, 0, 1}
noncomputable def N : Set ℤ := {x | x * x - x - 2 = 0}
noncomputable def complement_M_in_U : Set ℤ := U \ M

theorem intersection_equivalence : (complement_M_in_U ∩ N) = {2} := 
by
  sorry

end NUMINAMATH_GPT_intersection_equivalence_l1808_180823


namespace NUMINAMATH_GPT_find_number_l1808_180843

theorem find_number (x : ℝ) (h : 0.15 * 0.30 * 0.50 * x = 99) : x = 4400 :=
sorry

end NUMINAMATH_GPT_find_number_l1808_180843


namespace NUMINAMATH_GPT_sunny_ahead_in_second_race_l1808_180876

theorem sunny_ahead_in_second_race
  (s w : ℝ)
  (h1 : s / w = 8 / 7) :
  let sunny_new_speed := 0.9 * s
  let distance_sunny_runs := 450
  let distance_windy_runs := 400
  let time_sunny := distance_sunny_runs / sunny_new_speed
  let distance_windy_in_time_sunny := w * time_sunny
  450 - distance_windy_in_time_sunny = 12.5 :=
by
  let sunny_new_speed := 0.9 * s
  let distance_sunny_runs := 450
  let distance_windy_runs := 400
  let time_sunny := distance_sunny_runs / sunny_new_speed
  let distance_windy_in_time_sunny := w * time_sunny
  sorry

end NUMINAMATH_GPT_sunny_ahead_in_second_race_l1808_180876


namespace NUMINAMATH_GPT_bushes_needed_for_60_zucchinis_l1808_180847

-- Each blueberry bush yields 10 containers of blueberries.
def containers_per_bush : ℕ := 10

-- 6 containers of blueberries can be traded for 3 zucchinis.
def containers_to_zucchinis (containers zucchinis : ℕ) : Prop := containers = 6 ∧ zucchinis = 3

theorem bushes_needed_for_60_zucchinis (bushes containers zucchinis : ℕ) :
  containers_per_bush = 10 →
  containers_to_zucchinis 6 3 →
  zucchinis = 60 →
  bushes = 12 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_bushes_needed_for_60_zucchinis_l1808_180847


namespace NUMINAMATH_GPT_sixth_term_of_arithmetic_sequence_l1808_180860

noncomputable def sum_first_n_terms (a d : ℕ) (n : ℕ) : ℕ :=
  n * a + (n * (n - 1) / 2) * d

theorem sixth_term_of_arithmetic_sequence
  (a d : ℕ)
  (h₁ : sum_first_n_terms a d 4 = 10)
  (h₂ : a + 4 * d = 5) :
  a + 5 * d = 6 :=
by {
  sorry
}

end NUMINAMATH_GPT_sixth_term_of_arithmetic_sequence_l1808_180860


namespace NUMINAMATH_GPT_same_terminal_angle_l1808_180840

theorem same_terminal_angle (k : ℤ) :
  ∃ α : ℝ, α = k * 360 + 40 :=
by
  sorry

end NUMINAMATH_GPT_same_terminal_angle_l1808_180840


namespace NUMINAMATH_GPT_clara_weight_l1808_180899

theorem clara_weight (a c : ℝ) (h1 : a + c = 220) (h2 : c - a = c / 3) : c = 88 :=
by
  sorry

end NUMINAMATH_GPT_clara_weight_l1808_180899


namespace NUMINAMATH_GPT_grooming_time_l1808_180807

theorem grooming_time (time_per_dog : ℕ) (num_dogs : ℕ) (days : ℕ) (minutes_per_hour : ℕ) :
  time_per_dog = 20 →
  num_dogs = 2 →
  days = 30 →
  minutes_per_hour = 60 →
  (time_per_dog * num_dogs * days) / minutes_per_hour = 20 := 
by
  intros
  exact sorry

end NUMINAMATH_GPT_grooming_time_l1808_180807


namespace NUMINAMATH_GPT_unique_solution_for_lines_intersection_l1808_180890

theorem unique_solution_for_lines_intersection (n : ℕ) (h : n * (n - 1) / 2 = 2) : n = 2 :=
by
  sorry

end NUMINAMATH_GPT_unique_solution_for_lines_intersection_l1808_180890


namespace NUMINAMATH_GPT_divisor_inequality_l1808_180827

variable (k p : Nat)

-- Conditions
def is_prime (p : Nat) : Prop := Nat.Prime p
def is_divisor_of (k d : Nat) : Prop := ∃ m : Nat, d = k * m

-- Given conditions and the theorem to be proved
theorem divisor_inequality (h1 : k > 3) (h2 : is_prime p) (h3 : is_divisor_of k (2^p + 1)) : k ≥ 2 * p + 1 :=
  sorry

end NUMINAMATH_GPT_divisor_inequality_l1808_180827


namespace NUMINAMATH_GPT_derivative_of_y_l1808_180875

noncomputable def y (x : ℝ) : ℝ :=
  (4 * x + 1) / (16 * x^2 + 8 * x + 3) + (1 / Real.sqrt 2) * Real.arctan ((4 * x + 1) / Real.sqrt 2)

theorem derivative_of_y (x : ℝ) : 
  (deriv y x) = 16 / (16 * x^2 + 8 * x + 3)^2 :=
by 
  sorry

end NUMINAMATH_GPT_derivative_of_y_l1808_180875


namespace NUMINAMATH_GPT_exists_integers_x_y_z_l1808_180898

theorem exists_integers_x_y_z (n : ℕ) : 
  ∃ x y z : ℤ, (x^2 + y^2 + z^2 = 3^(2^n)) ∧ (Int.gcd x (Int.gcd y z) = 1) :=
sorry

end NUMINAMATH_GPT_exists_integers_x_y_z_l1808_180898


namespace NUMINAMATH_GPT_total_practice_hours_l1808_180866

def weekly_practice_hours : ℕ := 4
def weeks_per_month : ℕ := 4
def months : ℕ := 5

theorem total_practice_hours :
  weekly_practice_hours * weeks_per_month * months = 80 := by
  sorry

end NUMINAMATH_GPT_total_practice_hours_l1808_180866


namespace NUMINAMATH_GPT_original_price_of_trouser_l1808_180838

-- Define conditions
def sale_price : ℝ := 20
def discount : ℝ := 0.80

-- Define what the proof aims to show
theorem original_price_of_trouser (P : ℝ) (h : sale_price = P * (1 - discount)) : P = 100 :=
sorry

end NUMINAMATH_GPT_original_price_of_trouser_l1808_180838


namespace NUMINAMATH_GPT_period_of_3sin_minus_4cos_l1808_180854

theorem period_of_3sin_minus_4cos (x : ℝ) : 
  ∃ T : ℝ, T = 2 * Real.pi ∧ (∀ x, 3 * Real.sin x - 4 * Real.cos x = 3 * Real.sin (x + T) - 4 * Real.cos (x + T)) :=
sorry

end NUMINAMATH_GPT_period_of_3sin_minus_4cos_l1808_180854


namespace NUMINAMATH_GPT_san_francisco_superbowl_probability_l1808_180891

theorem san_francisco_superbowl_probability
  (P_play P_not_play : ℝ)
  (k : ℝ)
  (h1 : P_play = k * P_not_play)
  (h2 : P_play + P_not_play = 1) :
  k > 0 :=
sorry

end NUMINAMATH_GPT_san_francisco_superbowl_probability_l1808_180891


namespace NUMINAMATH_GPT_problem_solution_l1808_180804

theorem problem_solution (f : ℝ → ℝ)
    (h : ∀ x y : ℝ, f ((x - y) ^ 2) = f x ^ 2 - 2 * x * f y + y ^ 2) :
    ∃ n s : ℕ, 
    (n = 2) ∧ 
    (s = 3) ∧
    (n * s = 6) :=
sorry

end NUMINAMATH_GPT_problem_solution_l1808_180804


namespace NUMINAMATH_GPT_angle_diff_l1808_180868

-- Given conditions as definitions
def angle_A : ℝ := 120
def angle_B : ℝ := 50
def angle_D : ℝ := 60
def angle_E : ℝ := 140

-- Prove the difference between angle BCD and angle AFE is 10 degrees
theorem angle_diff (AB_parallel_DE : ∀ (A B D E : ℝ), AB_parallel_DE)
                 (angle_A_def : angle_A = 120)
                 (angle_B_def : angle_B = 50)
                 (angle_D_def : angle_D = 60)
                 (angle_E_def : angle_E = 140) :
    let angle_3 : ℝ := 180 - angle_A
    let angle_4 : ℝ := 180 - angle_E
    let angle_BCD : ℝ := angle_B + angle_D
    let angle_AFE : ℝ := angle_3 + angle_4
    angle_BCD - angle_AFE = 10 :=
by {
  sorry
}

end NUMINAMATH_GPT_angle_diff_l1808_180868


namespace NUMINAMATH_GPT_positive_numbers_l1808_180815

theorem positive_numbers 
    (a b c : ℝ) 
    (h1 : a + b + c > 0) 
    (h2 : ab + bc + ca > 0) 
    (h3 : abc > 0) 
    : a > 0 ∧ b > 0 ∧ c > 0 :=
sorry

end NUMINAMATH_GPT_positive_numbers_l1808_180815


namespace NUMINAMATH_GPT_length_RS_l1808_180834

open Real

-- Given definitions and conditions
def PQ : ℝ := 10
def PR : ℝ := 10
def QR : ℝ := 5
def PS : ℝ := 13

-- Prove the length of RS
theorem length_RS : ∃ (RS : ℝ), RS = 6.17362 := by
  sorry

end NUMINAMATH_GPT_length_RS_l1808_180834


namespace NUMINAMATH_GPT_count_restricted_arrangements_l1808_180817

theorem count_restricted_arrangements (n : ℕ) (hn : n = 5) : 
  (n.factorial - 2 * (n - 1).factorial) = 72 := 
by 
  sorry

end NUMINAMATH_GPT_count_restricted_arrangements_l1808_180817


namespace NUMINAMATH_GPT_charlie_extra_charge_l1808_180844

-- Define the data plan and cost structure
def data_plan_limit : ℕ := 8  -- GB
def extra_cost_per_gb : ℕ := 10  -- $ per GB

-- Define Charlie's data usage over each week
def usage_week_1 : ℕ := 2  -- GB
def usage_week_2 : ℕ := 3  -- GB
def usage_week_3 : ℕ := 5  -- GB
def usage_week_4 : ℕ := 10  -- GB

-- Calculate the total data usage and the extra data used
def total_usage : ℕ := usage_week_1 + usage_week_2 + usage_week_3 + usage_week_4
def extra_usage : ℕ := if total_usage > data_plan_limit then total_usage - data_plan_limit else 0
def extra_charge : ℕ := extra_usage * extra_cost_per_gb

-- Theorem to prove the extra charge
theorem charlie_extra_charge : extra_charge = 120 := by
  -- Skipping the proof
  sorry

end NUMINAMATH_GPT_charlie_extra_charge_l1808_180844


namespace NUMINAMATH_GPT_many_people_sharing_car_l1808_180810

theorem many_people_sharing_car (x y : ℤ) 
  (h1 : 3 * (y - 2) = x) 
  (h2 : 2 * y + 9 = x) : 
  3 * (y - 2) = 2 * y + 9 := 
by
  -- by assumption h1 and h2, we already have the setup, refute/validate consistency
  sorry

end NUMINAMATH_GPT_many_people_sharing_car_l1808_180810


namespace NUMINAMATH_GPT_tricycle_wheel_count_l1808_180820

theorem tricycle_wheel_count (bicycles wheels_per_bicycle tricycles total_wheels : ℕ)
  (h1 : bicycles = 16)
  (h2 : wheels_per_bicycle = 2)
  (h3 : tricycles = 7)
  (h4 : total_wheels = 53)
  (h5 : total_wheels = (bicycles * wheels_per_bicycle) + (tricycles * (3 : ℕ))) : 
  (3 : ℕ) = 3 := by
  sorry

end NUMINAMATH_GPT_tricycle_wheel_count_l1808_180820


namespace NUMINAMATH_GPT_parallel_lines_m_values_l1808_180825

theorem parallel_lines_m_values (m : ℝ) :
  (∀ x y : ℝ, 2 * x + (m + 1) * y + 4 = 0 ↔ mx + 3 * y - 2 = 0) → (m = -3 ∨ m = 2) :=
by
  sorry

end NUMINAMATH_GPT_parallel_lines_m_values_l1808_180825


namespace NUMINAMATH_GPT_problem_equivalence_l1808_180831

theorem problem_equivalence : 4 * 299 + 3 * 299 + 2 * 299 + 298 = 2989 := by
  sorry

end NUMINAMATH_GPT_problem_equivalence_l1808_180831


namespace NUMINAMATH_GPT_evaluate_expression_l1808_180800

def g (x : ℝ) : ℝ := 3 * x^2 - 5 * x + 7

theorem evaluate_expression : 3 * g 2 + 2 * g (-2) = 85 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1808_180800


namespace NUMINAMATH_GPT_smallest_integer_value_of_x_l1808_180832

theorem smallest_integer_value_of_x (x : ℤ) (h : 7 + 3 * x < 26) : x = 6 :=
sorry

end NUMINAMATH_GPT_smallest_integer_value_of_x_l1808_180832


namespace NUMINAMATH_GPT_factor_z4_minus_81_l1808_180871

theorem factor_z4_minus_81 :
  (z^4 - 81) = (z - 3) * (z + 3) * (z^2 + 9) :=
by
  sorry

end NUMINAMATH_GPT_factor_z4_minus_81_l1808_180871


namespace NUMINAMATH_GPT_custom_op_evaluation_l1808_180848

def custom_op (x y : ℤ) : ℤ := x * y - 3 * x + y

theorem custom_op_evaluation : (custom_op 9 6) - (custom_op 6 9) = -12 := by
  sorry

end NUMINAMATH_GPT_custom_op_evaluation_l1808_180848


namespace NUMINAMATH_GPT_min_value_of_f_l1808_180886

noncomputable def f (x : ℝ) : ℝ := 4 * x + 2 / x

theorem min_value_of_f (x : ℝ) (hx : x > 0) : ∃ y : ℝ, (∀ z : ℝ, z > 0 → f z ≥ y) ∧ y = 4 * Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_min_value_of_f_l1808_180886


namespace NUMINAMATH_GPT_line_product_l1808_180812

theorem line_product (b m : Int) (h_b : b = -2) (h_m : m = 3) : m * b = -6 :=
by
  rw [h_b, h_m]
  norm_num

end NUMINAMATH_GPT_line_product_l1808_180812


namespace NUMINAMATH_GPT_m_add_n_equals_19_l1808_180836

theorem m_add_n_equals_19 (n m : ℕ) (A_n_m : ℕ) (C_n_m : ℕ) (h1 : A_n_m = 272) (h2 : C_n_m = 136) :
  m + n = 19 :=
by
  sorry

end NUMINAMATH_GPT_m_add_n_equals_19_l1808_180836


namespace NUMINAMATH_GPT_total_present_ages_l1808_180853

theorem total_present_ages (P Q : ℕ) 
    (h1 : P - 12 = (1 / 2) * (Q - 12))
    (h2 : P = (3 / 4) * Q) : P + Q = 42 :=
by
  sorry

end NUMINAMATH_GPT_total_present_ages_l1808_180853


namespace NUMINAMATH_GPT_inverse_proposition_false_l1808_180878

theorem inverse_proposition_false (a b c : ℝ) : 
  ¬ (a > b → ((c ≠ 0) ∧ (a / (c * c)) > (b / (c * c))))
:= 
by 
  -- Outline indicating that the proof will follow from checking cases
  sorry

end NUMINAMATH_GPT_inverse_proposition_false_l1808_180878


namespace NUMINAMATH_GPT_find_ab_l1808_180857

theorem find_ab (a b : ℝ) (h1 : a + b = 4) (h2 : a^3 + b^3 = 136) : a * b = -6 :=
by
  sorry

end NUMINAMATH_GPT_find_ab_l1808_180857


namespace NUMINAMATH_GPT_children_play_time_equal_l1808_180808

-- Definitions based on the conditions in the problem
def totalChildren := 7
def totalPlayingTime := 140
def playersAtATime := 2

-- The statement to be proved
theorem children_play_time_equal :
  (playersAtATime * totalPlayingTime) / totalChildren = 40 := by
sorry

end NUMINAMATH_GPT_children_play_time_equal_l1808_180808


namespace NUMINAMATH_GPT_sum_of_fractions_to_decimal_l1808_180893

theorem sum_of_fractions_to_decimal :
  ((2 / 40 : ℚ) + (4 / 80) + (6 / 120) + (9 / 180) : ℚ) = 0.2 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_fractions_to_decimal_l1808_180893


namespace NUMINAMATH_GPT_line_passing_through_points_l1808_180896

theorem line_passing_through_points (a_1 b_1 a_2 b_2 : ℝ) 
  (h1 : 2 * a_1 + 3 * b_1 + 1 = 0)
  (h2 : 2 * a_2 + 3 * b_2 + 1 = 0) : 
  ∃ (m n : ℝ), (∀ x y : ℝ, (y - b_1) * (x - a_2) = (y - b_2) * (x - a_1)) → (m = 2 ∧ n = 3) :=
by { sorry }

end NUMINAMATH_GPT_line_passing_through_points_l1808_180896


namespace NUMINAMATH_GPT_cylinder_volume_ratio_l1808_180852

theorem cylinder_volume_ratio (w h : ℝ) (w_pos : w = 6) (h_pos : h = 9) :
  let r1 := w / (2 * Real.pi)
  let h1 := h
  let V1 := Real.pi * r1^2 * h1
  let r2 := h / (2 * Real.pi)
  let h2 := w
  let V2 := Real.pi * r2^2 * h2
  V2 / V1 = 3 / 4 :=
by
  sorry

end NUMINAMATH_GPT_cylinder_volume_ratio_l1808_180852


namespace NUMINAMATH_GPT_fraction_equality_l1808_180862

theorem fraction_equality
  (a b c d : ℝ) 
  (h1 : b ≠ c)
  (h2 : (a * c - b^2) / (a - 2 * b + c) = (b * d - c^2) / (b - 2 * c + d)) : 
  (a * c - b^2) / (a - 2 * b + c) = (a * d - b * c) / (a - b - c + d) ∧
  (b * d - c^2) / (b - 2 * c + d) = (a * d - b * c) / (a - b - c + d) := 
by
  sorry

end NUMINAMATH_GPT_fraction_equality_l1808_180862


namespace NUMINAMATH_GPT_find_first_number_l1808_180895

theorem find_first_number (sum_is_33 : ∃ x y : ℕ, x + y = 33) (second_is_twice_first : ∃ x y : ℕ, y = 2 * x) (second_is_22 : ∃ y : ℕ, y = 22) : ∃ x : ℕ, x = 11 :=
by
  sorry

end NUMINAMATH_GPT_find_first_number_l1808_180895


namespace NUMINAMATH_GPT_original_example_intended_l1808_180801

theorem original_example_intended (x : ℝ) : (3 * x - 4 = x / 3 + 4) → x = 3 :=
by
  sorry

end NUMINAMATH_GPT_original_example_intended_l1808_180801


namespace NUMINAMATH_GPT_inradius_of_triangle_l1808_180864

/-- Given conditions for the triangle -/
def perimeter : ℝ := 32
def area : ℝ := 40

/-- The theorem to prove the inradius of the triangle -/
theorem inradius_of_triangle (h : area = (r * perimeter) / 2) : r = 2.5 :=
by
  sorry

end NUMINAMATH_GPT_inradius_of_triangle_l1808_180864


namespace NUMINAMATH_GPT_quadratic_inequality_solution_l1808_180833

theorem quadratic_inequality_solution (a : ℝ) :
  (∃ x : ℝ, x^2 + (a - 1) * x + 1 < 0) ↔ (a < -1 ∨ a > 3) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_l1808_180833


namespace NUMINAMATH_GPT_speed_in_still_water_l1808_180849

theorem speed_in_still_water (upstream_speed : ℝ) (downstream_speed : ℝ) 
  (h_upstream : upstream_speed = 45) (h_downstream : downstream_speed = 55) : 
  (upstream_speed + downstream_speed) / 2 = 50 := 
by
  rw [h_upstream, h_downstream] 
  norm_num  -- simplifies the numeric expression
  done

end NUMINAMATH_GPT_speed_in_still_water_l1808_180849


namespace NUMINAMATH_GPT_outdoor_tables_count_l1808_180819

theorem outdoor_tables_count (num_indoor_tables : ℕ) (chairs_per_indoor_table : ℕ) (chairs_per_outdoor_table : ℕ) (total_chairs : ℕ) : ℕ :=
  let num_outdoor_tables := (total_chairs - (num_indoor_tables * chairs_per_indoor_table)) / chairs_per_outdoor_table
  num_outdoor_tables

example (h₁ : num_indoor_tables = 9)
        (h₂ : chairs_per_indoor_table = 10)
        (h₃ : chairs_per_outdoor_table = 3)
        (h₄ : total_chairs = 123) :
        outdoor_tables_count 9 10 3 123 = 11 :=
by
  -- Only the statement has to be provided; proof steps are not needed
  sorry

end NUMINAMATH_GPT_outdoor_tables_count_l1808_180819


namespace NUMINAMATH_GPT_unique_function_property_l1808_180889

theorem unique_function_property (f : ℕ → ℕ) (h : ∀ m n : ℕ, f m + f n ∣ m + n) :
  ∀ m : ℕ, f m = m :=
by
  sorry

end NUMINAMATH_GPT_unique_function_property_l1808_180889


namespace NUMINAMATH_GPT_parabola_properties_l1808_180894

-- Define the conditions
def vertex (f : ℝ → ℝ) (v : ℝ × ℝ) : Prop :=
  ∀ (x : ℝ), f (v.1) ≤ f x

def vertical_axis_of_symmetry (f : ℝ → ℝ) (h : ℝ) : Prop :=
  ∀ (x : ℝ), f x = f (2 * h - x)

def contains_point (f : ℝ → ℝ) (p : ℝ × ℝ) : Prop :=
  f p.1 = p.2

-- Define f as the given parabola equation
def f (x : ℝ) : ℝ := 2 * x^2 - 12 * x + 16

-- The main statement to prove
theorem parabola_properties :
  vertex f (3, -2) ∧ vertical_axis_of_symmetry f 3 ∧ contains_point f (6, 16) := sorry

end NUMINAMATH_GPT_parabola_properties_l1808_180894


namespace NUMINAMATH_GPT_share_difference_l1808_180822

theorem share_difference (x : ℕ) (p q r : ℕ) 
  (h1 : 3 * x = p) 
  (h2 : 7 * x = q) 
  (h3 : 12 * x = r) 
  (h4 : q - p = 2800) : 
  r - q = 3500 := by {
  sorry
}

end NUMINAMATH_GPT_share_difference_l1808_180822


namespace NUMINAMATH_GPT_trig_problem_1_trig_problem_2_l1808_180806

noncomputable def trig_expr_1 : ℝ :=
  Real.cos (-11 * Real.pi / 6) + Real.sin (12 * Real.pi / 5) * Real.tan (6 * Real.pi)

noncomputable def trig_expr_2 : ℝ :=
  Real.sin (420 * Real.pi / 180) * Real.cos (750 * Real.pi / 180) +
  Real.sin (-330 * Real.pi / 180) * Real.cos (-660 * Real.pi / 180)

theorem trig_problem_1 : trig_expr_1 = Real.sqrt 3 / 2 :=
by
  sorry

theorem trig_problem_2 : trig_expr_2 = 1 :=
by
  sorry

end NUMINAMATH_GPT_trig_problem_1_trig_problem_2_l1808_180806


namespace NUMINAMATH_GPT_gemstone_necklaces_sold_correct_l1808_180839

-- Define the conditions
def bead_necklaces_sold : Nat := 4
def necklace_cost : Nat := 3
def total_earnings : Nat := 21
def bead_necklaces_earnings : Nat := bead_necklaces_sold * necklace_cost
def gemstone_necklaces_earnings : Nat := total_earnings - bead_necklaces_earnings
def gemstone_necklaces_sold : Nat := gemstone_necklaces_earnings / necklace_cost

-- Theorem to prove the number of gem stone necklaces sold
theorem gemstone_necklaces_sold_correct :
  gemstone_necklaces_sold = 3 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_gemstone_necklaces_sold_correct_l1808_180839


namespace NUMINAMATH_GPT_parallel_lines_slope_l1808_180892

theorem parallel_lines_slope (m : ℝ) (h : (x + (1 + m) * y + m - 2 = 0) ∧ (m * x + 2 * y + 6 = 0)) :
  m = 1 ∨ m = -2 :=
  sorry

end NUMINAMATH_GPT_parallel_lines_slope_l1808_180892


namespace NUMINAMATH_GPT_amount_borrowed_from_bank_l1808_180881

-- Definitions of the conditions
def car_price : ℝ := 35000
def total_payment : ℝ := 38000
def interest_rate : ℝ := 0.15

theorem amount_borrowed_from_bank :
  total_payment - car_price = interest_rate * (total_payment - car_price) / interest_rate := sorry

end NUMINAMATH_GPT_amount_borrowed_from_bank_l1808_180881


namespace NUMINAMATH_GPT_decks_left_is_3_l1808_180805

-- Given conditions
def price_per_deck := 2
def total_decks_start := 5
def money_earned := 4

-- The number of decks sold
def decks_sold := money_earned / price_per_deck

-- The number of decks left
def decks_left := total_decks_start - decks_sold

-- The theorem to prove 
theorem decks_left_is_3 : decks_left = 3 :=
by
  -- Here we put the steps to prove
  sorry

end NUMINAMATH_GPT_decks_left_is_3_l1808_180805


namespace NUMINAMATH_GPT_expression_is_integer_l1808_180837

theorem expression_is_integer (n : ℤ) : (∃ k : ℤ, n * (n + 1) * (n + 2) * (n + 3) = 24 * k) := 
sorry

end NUMINAMATH_GPT_expression_is_integer_l1808_180837


namespace NUMINAMATH_GPT_fraction_percent_l1808_180842

theorem fraction_percent (x : ℝ) (h : x > 0) : ((x / 10 + x / 25) / x) * 100 = 14 :=
by
  sorry

end NUMINAMATH_GPT_fraction_percent_l1808_180842
