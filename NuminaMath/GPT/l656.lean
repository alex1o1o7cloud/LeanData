import Mathlib

namespace NUMINAMATH_GPT_washing_machine_capacity_l656_65640

-- Definitions of the conditions
def total_pounds_per_day : ℕ := 200
def number_of_machines : ℕ := 8

-- Main theorem to prove the question == answer given the conditions
theorem washing_machine_capacity :
  total_pounds_per_day / number_of_machines = 25 :=
by
  sorry

end NUMINAMATH_GPT_washing_machine_capacity_l656_65640


namespace NUMINAMATH_GPT_boys_girls_ratio_l656_65669

-- Definitions used as conditions
variable (B G : ℕ)

-- Conditions
def condition1 : Prop := B + G = 32
def condition2 : Prop := B = 2 * (G - 8)

-- Proof that the ratio of boys to girls initially is 1:1
theorem boys_girls_ratio (h1 : condition1 B G) (h2 : condition2 B G) : (B : ℚ) / G = 1 := by
  sorry

end NUMINAMATH_GPT_boys_girls_ratio_l656_65669


namespace NUMINAMATH_GPT_area_of_black_region_l656_65603

def side_length_square : ℝ := 10
def length_rectangle : ℝ := 5
def width_rectangle : ℝ := 2

theorem area_of_black_region :
  (side_length_square * side_length_square) - (length_rectangle * width_rectangle) = 90 := by
sorry

end NUMINAMATH_GPT_area_of_black_region_l656_65603


namespace NUMINAMATH_GPT_shorter_leg_length_l656_65667

theorem shorter_leg_length (a b c : ℝ) (h1 : b = 10) (h2 : a^2 + b^2 = c^2) (h3 : c = 2 * a) : 
  a = 10 * Real.sqrt 3 / 3 :=
by
  sorry

end NUMINAMATH_GPT_shorter_leg_length_l656_65667


namespace NUMINAMATH_GPT_Lois_books_total_l656_65675

-- Definitions based on the conditions
def initial_books : ℕ := 150
def books_given_to_nephew : ℕ := initial_books / 4
def remaining_books : ℕ := initial_books - books_given_to_nephew
def non_fiction_books : ℕ := remaining_books * 60 / 100
def kept_non_fiction_books : ℕ := non_fiction_books / 2
def fiction_books : ℕ := remaining_books - non_fiction_books
def lent_fiction_books : ℕ := fiction_books / 3
def remaining_fiction_books : ℕ := fiction_books - lent_fiction_books
def newly_purchased_books : ℕ := 12

-- The total number of books Lois has now
def total_books_now : ℕ := kept_non_fiction_books + remaining_fiction_books + newly_purchased_books

-- Theorem statement
theorem Lois_books_total : total_books_now = 76 := by
  sorry

end NUMINAMATH_GPT_Lois_books_total_l656_65675


namespace NUMINAMATH_GPT_total_cost_correct_l656_65653

noncomputable def camera_old_cost : ℝ := 4000
noncomputable def camera_new_cost := camera_old_cost * 1.30
noncomputable def lens_cost := 400
noncomputable def lens_discount := 200
noncomputable def lens_discounted_price := lens_cost - lens_discount
noncomputable def total_cost := camera_new_cost + lens_discounted_price

theorem total_cost_correct :
  total_cost = 5400 := by
  sorry

end NUMINAMATH_GPT_total_cost_correct_l656_65653


namespace NUMINAMATH_GPT_math_club_problem_l656_65649

theorem math_club_problem :
  ((7^2 - 5^2) - 2)^3 = 10648 := by
  sorry

end NUMINAMATH_GPT_math_club_problem_l656_65649


namespace NUMINAMATH_GPT_Talia_father_age_l656_65638

def Talia_age (T : ℕ) : Prop := T + 7 = 20
def Talia_mom_age (M T : ℕ) : Prop := M = 3 * T
def Talia_father_age_in_3_years (F M : ℕ) : Prop := F + 3 = M

theorem Talia_father_age (T F M : ℕ) 
    (hT : Talia_age T)
    (hM : Talia_mom_age M T)
    (hF : Talia_father_age_in_3_years F M) :
    F = 36 :=
by 
  sorry

end NUMINAMATH_GPT_Talia_father_age_l656_65638


namespace NUMINAMATH_GPT_maximum_marks_l656_65687

theorem maximum_marks (M : ℝ) (h : 0.5 * M = 50 + 10) : M = 120 :=
by
  sorry

end NUMINAMATH_GPT_maximum_marks_l656_65687


namespace NUMINAMATH_GPT_shift_quadratic_function_left_l656_65634

-- Define the original quadratic function
def original_function (x : ℝ) : ℝ := x^2

-- Define the shifted quadratic function
def shifted_function (x : ℝ) : ℝ := (x + 1)^2

-- Theorem statement
theorem shift_quadratic_function_left :
  ∀ x : ℝ, shifted_function x = original_function (x + 1) := by
  sorry

end NUMINAMATH_GPT_shift_quadratic_function_left_l656_65634


namespace NUMINAMATH_GPT_not_divisible_by_121_l656_65661

theorem not_divisible_by_121 (n : ℤ) : ¬ (121 ∣ (n^2 + 3 * n + 5)) :=
by
  sorry

end NUMINAMATH_GPT_not_divisible_by_121_l656_65661


namespace NUMINAMATH_GPT_line_interparabola_length_l656_65655

theorem line_interparabola_length :
  (∀ (x y : ℝ), y = x - 2 → y^2 = 4 * x) →
  ∃ (A B : ℝ × ℝ), (∃ (x1 y1 x2 y2 : ℝ), A = (x1, y1) ∧ B = (x2, y2)) →
  (dist A B = 4 * Real.sqrt 6) :=
by
  intros
  sorry

end NUMINAMATH_GPT_line_interparabola_length_l656_65655


namespace NUMINAMATH_GPT_range_of_a_and_m_l656_65676

open Set

-- Definitions of the sets A, B, and C
def A : Set ℝ := {x | x^2 - 3 * x + 2 = 0}
def B (a : ℝ) : Set ℝ := {x | x^2 - a * x + a - 1 = 0}
def C (m : ℝ) : Set ℝ := {x | x^2 - m * x + 1 = 0}

-- Conditions as hypotheses
def condition1 : A ∪ B a = A := sorry
def condition2 : A ∩ C m = C m := sorry

-- Theorem to prove the correct range of a and m
theorem range_of_a_and_m : (a = 2 ∨ a = 3) ∧ (-2 < m ∧ m ≤ 2) :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_range_of_a_and_m_l656_65676


namespace NUMINAMATH_GPT_jerky_remaining_after_giving_half_l656_65630

-- Define the main conditions as variables
def days := 5
def initial_jerky := 40
def jerky_per_day := 1 + 1 + 2

-- Calculate total consumption
def total_consumption := jerky_per_day * days

-- Calculate remaining jerky
def remaining_jerky := initial_jerky - total_consumption

-- Calculate final jerky after giving half to her brother
def jerky_left := remaining_jerky / 2

-- Statement to be proved
theorem jerky_remaining_after_giving_half :
  jerky_left = 10 :=
by
  -- Proof will go here
  sorry

end NUMINAMATH_GPT_jerky_remaining_after_giving_half_l656_65630


namespace NUMINAMATH_GPT_maximum_value_a_over_b_plus_c_l656_65610

open Real

noncomputable def max_frac_a_over_b_plus_c (a b c : ℝ) (h_pos: 0 < a ∧ 0 < b ∧ 0 < c) (h_eq: a * (a + b + c) = b * c) : ℝ :=
  if (b = c) then (Real.sqrt 2 - 1) / 2 else -1 -- placeholder for irrelevant case

theorem maximum_value_a_over_b_plus_c 
  (a b c : ℝ) 
  (h_pos: 0 < a ∧ 0 < b ∧ 0 < c)
  (h_eq: a * (a + b + c) = b * c) :
  max_frac_a_over_b_plus_c a b c h_pos h_eq = (Real.sqrt 2 - 1) / 2 :=
sorry

end NUMINAMATH_GPT_maximum_value_a_over_b_plus_c_l656_65610


namespace NUMINAMATH_GPT_quadratic_inequality_condition_l656_65616

theorem quadratic_inequality_condition
  (a b c : ℝ)
  (h1 : b^2 - 4 * a * c < 0)
  (h2 : ∀ x : ℝ, a * x^2 + b * x + c < 0) :
  False :=
sorry

end NUMINAMATH_GPT_quadratic_inequality_condition_l656_65616


namespace NUMINAMATH_GPT_cook_weave_l656_65692

theorem cook_weave (Y C W OC CY CYW : ℕ) (hY : Y = 25) (hC : C = 15) (hW : W = 8) (hOC : OC = 2)
  (hCY : CY = 7) (hCYW : CYW = 3) : 
  ∃ (CW : ℕ), CW = 9 :=
by 
  have CW : ℕ := C - OC - (CY - CYW) 
  use CW
  sorry

end NUMINAMATH_GPT_cook_weave_l656_65692


namespace NUMINAMATH_GPT_find_value_of_fraction_l656_65604

variable (x y : ℝ)
variable (h1 : y > x)
variable (h2 : x > 0)
variable (h3 : (x / y) + (y / x) = 8)

theorem find_value_of_fraction (h1 : y > x) (h2 : x > 0) (h3 : (x / y) + (y / x) = 8) :
  (x + y) / (x - y) = Real.sqrt (5 / 3) :=
sorry

end NUMINAMATH_GPT_find_value_of_fraction_l656_65604


namespace NUMINAMATH_GPT_min_queries_to_determine_parity_l656_65641

def num_bags := 100
def num_queries := 3
def bags := Finset (Fin num_bags)

def can_query_parity (bags : Finset (Fin num_bags)) : Prop :=
  bags.card = 15

theorem min_queries_to_determine_parity :
  ∀ (query : Fin num_queries → Finset (Fin num_bags)),
  (∀ i, can_query_parity (query i)) →
  (∀ i j k, query i ∪ query j ∪ query k = {a : Fin num_bags | a.val = 1}) →
  num_queries ≥ 3 :=
  sorry

end NUMINAMATH_GPT_min_queries_to_determine_parity_l656_65641


namespace NUMINAMATH_GPT_jason_text_messages_per_day_l656_65608

theorem jason_text_messages_per_day
  (monday_messages : ℕ)
  (tuesday_messages : ℕ)
  (total_messages : ℕ)
  (average_per_day : ℕ)
  (messages_wednesday_friday_per_day : ℕ) :
  monday_messages = 220 →
  tuesday_messages = monday_messages / 2 →
  average_per_day = 96 →
  total_messages = 5 * average_per_day →
  total_messages - (monday_messages + tuesday_messages) = 3 * messages_wednesday_friday_per_day →
  messages_wednesday_friday_per_day = 50 :=
by
  intros
  sorry

end NUMINAMATH_GPT_jason_text_messages_per_day_l656_65608


namespace NUMINAMATH_GPT_base7_to_base10_l656_65683

-- Define the base-7 number 521 in base-7
def base7_num : Nat := 5 * 7^2 + 2 * 7^1 + 1 * 7^0

-- State the theorem that needs to be proven
theorem base7_to_base10 : base7_num = 260 :=
by
  -- Proof steps will go here, but we'll skip and insert a sorry for now
  sorry

end NUMINAMATH_GPT_base7_to_base10_l656_65683


namespace NUMINAMATH_GPT_tetrahedron_through_hole_tetrahedron_cannot_through_hole_l656_65679

/--
A regular tetrahedron with edge length 1 can pass through a circular hole if and only if the radius \( R \) is at least 0.4478, given that the thickness of the hole can be neglected.
-/

theorem tetrahedron_through_hole (R : ℝ) (h1 : R = 0.45) : true :=
by sorry

theorem tetrahedron_cannot_through_hole (R : ℝ) (h1 : R = 0.44) : false :=
by sorry

end NUMINAMATH_GPT_tetrahedron_through_hole_tetrahedron_cannot_through_hole_l656_65679


namespace NUMINAMATH_GPT_find_value_of_y_l656_65671

theorem find_value_of_y (x y : ℚ) 
  (h1 : x = 51) 
  (h2 : x^3 * y - 2 * x^2 * y + x * y = 63000) : 
  y = 8 / 17 := 
by 
  sorry

end NUMINAMATH_GPT_find_value_of_y_l656_65671


namespace NUMINAMATH_GPT_Sarah_consumed_one_sixth_l656_65633

theorem Sarah_consumed_one_sixth (total_slices : ℕ) (slices_sarah_ate : ℕ) (shared_slices : ℕ) :
  total_slices = 20 → slices_sarah_ate = 3 → shared_slices = 1 → 
  ((slices_sarah_ate + shared_slices / 3) / total_slices : ℚ) = 1 / 6 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_Sarah_consumed_one_sixth_l656_65633


namespace NUMINAMATH_GPT_statement2_true_l656_65686

def digit : ℕ := sorry

def statement1 : Prop := digit = 2
def statement2 : Prop := digit ≠ 3
def statement3 : Prop := digit = 5
def statement4 : Prop := digit ≠ 6

def condition : Prop := (statement1 ∨ statement2 ∨ statement3 ∨ statement4) ∧
                        (statement1 ∨ statement2 ∨ statement3 ∨ statement4) ∧
                        (statement1 ∨ statement2 ∨ statement3 ∨ statement4) ∧
                        (¬ statement1 ∨ ¬ statement2 ∨ ¬ statement3 ∨ ¬ statement4)

theorem statement2_true (h : condition) : statement2 :=
sorry

end NUMINAMATH_GPT_statement2_true_l656_65686


namespace NUMINAMATH_GPT_percentage_increase_l656_65639

theorem percentage_increase 
    (P : ℝ)
    (buying_price : ℝ) (h1 : buying_price = 0.80 * P)
    (selling_price : ℝ) (h2 : selling_price = 1.24 * P) :
    ((selling_price - buying_price) / buying_price) * 100 = 55 := by 
  sorry

end NUMINAMATH_GPT_percentage_increase_l656_65639


namespace NUMINAMATH_GPT_coronavirus_transmission_l656_65663

theorem coronavirus_transmission:
  (∃ x: ℝ, (1 + x)^2 = 225) :=
by
  sorry

end NUMINAMATH_GPT_coronavirus_transmission_l656_65663


namespace NUMINAMATH_GPT_magnitude_of_b_l656_65690

open Real

noncomputable def a : ℝ × ℝ := (-sqrt 3, 1)

theorem magnitude_of_b (b : ℝ × ℝ)
    (h1 : (a.1 + 2 * b.1, a.2 + 2 * b.2) = (a.1, a.2))
    (h2 : (a.1 + b.1, a.2 + b.2) = (b.1, b.2)) :
    sqrt (b.1 ^ 2 + b.2 ^ 2) = sqrt 2 :=
sorry

end NUMINAMATH_GPT_magnitude_of_b_l656_65690


namespace NUMINAMATH_GPT_percent_problem_l656_65606

theorem percent_problem (x : ℝ) (h : 0.30 * 0.40 * x = 36) : 0.40 * 0.30 * x = 36 :=
by
  sorry

end NUMINAMATH_GPT_percent_problem_l656_65606


namespace NUMINAMATH_GPT_convex_pentagon_largest_angle_l656_65696

theorem convex_pentagon_largest_angle 
  (x : ℝ)
  (h1 : (x + 2) + (2 * x + 3) + (3 * x + 6) + (4 * x + 5) + (5 * x + 4) = 540) :
  5 * x + 4 = 532 / 3 :=
by
  sorry

end NUMINAMATH_GPT_convex_pentagon_largest_angle_l656_65696


namespace NUMINAMATH_GPT_trigonometric_identity_l656_65684

theorem trigonometric_identity
  (x : ℝ) 
  (h_tan : Real.tan x = -1/2) :
  (3 * Real.sin x ^ 2 - 2) / (Real.sin x * Real.cos x) = 7 / 2 := 
by
  sorry

end NUMINAMATH_GPT_trigonometric_identity_l656_65684


namespace NUMINAMATH_GPT_cat_litter_cost_l656_65647

theorem cat_litter_cost 
    (container_weight : ℕ) (container_cost : ℕ)
    (litter_box_capacity : ℕ) (change_interval : ℕ) 
    (days_needed : ℕ) (cost : ℕ) :
  container_weight = 45 → 
  container_cost = 21 → 
  litter_box_capacity = 15 → 
  change_interval = 7 →
  days_needed = 210 → 
  cost = 210 :=
by
  intros h1 h2 h3 h4 h5
  /- Here we would add the proof steps, but this is not required. -/
  sorry

end NUMINAMATH_GPT_cat_litter_cost_l656_65647


namespace NUMINAMATH_GPT_urn_gold_coins_percent_l656_65685

theorem urn_gold_coins_percent (perc_beads : ℝ) (perc_silver_coins : ℝ) (perc_gold_coins : ℝ) :
  perc_beads = 0.2 →
  perc_silver_coins = 0.4 →
  perc_gold_coins = 0.48 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_urn_gold_coins_percent_l656_65685


namespace NUMINAMATH_GPT_exists_integers_gcd_eq_one_addition_l656_65660

theorem exists_integers_gcd_eq_one_addition 
  (n k : ℕ) 
  (hnk_pos : n > 0 ∧ k > 0) 
  (hn_even_or_nk_even : (¬ n % 2 = 0) ∨ (n % 2 = 0 ∧ k % 2 = 0)) :
  ∃ a b : ℤ, Int.gcd a ↑n = 1 ∧ Int.gcd b ↑n = 1 ∧ k = a + b :=
by
  sorry

end NUMINAMATH_GPT_exists_integers_gcd_eq_one_addition_l656_65660


namespace NUMINAMATH_GPT_power_of_integer_is_two_l656_65691

-- Definitions based on conditions
def is_power_of_integer (n : ℕ) : Prop :=
  ∃ (k : ℕ) (m : ℕ), n = m^k

-- Given conditions translated to Lean definitions
def g : ℕ := 14
def n : ℕ := 3150 * g

-- The proof problem statement in Lean
theorem power_of_integer_is_two (h : g = 14) : is_power_of_integer n :=
sorry

end NUMINAMATH_GPT_power_of_integer_is_two_l656_65691


namespace NUMINAMATH_GPT_maria_towels_l656_65645

theorem maria_towels (green_towels white_towels given_towels : ℕ) (h1 : green_towels = 35) (h2 : white_towels = 21) (h3 : given_towels = 34) :
  green_towels + white_towels - given_towels = 22 :=
by
  sorry

end NUMINAMATH_GPT_maria_towels_l656_65645


namespace NUMINAMATH_GPT_total_oil_leakage_l656_65650

def oil_leaked_before : ℕ := 6522
def oil_leaked_during : ℕ := 5165
def total_oil_leaked : ℕ := oil_leaked_before + oil_leaked_during

theorem total_oil_leakage : total_oil_leaked = 11687 := by
  sorry

end NUMINAMATH_GPT_total_oil_leakage_l656_65650


namespace NUMINAMATH_GPT_tetrahedron_volume_ratio_l656_65644

theorem tetrahedron_volume_ratio
  (a b : ℝ)
  (larger_tetrahedron : a = 6)
  (smaller_tetrahedron : b = a / 2) :
  (b^3 / a^3) = 1 / 8 := 
by 
  sorry

end NUMINAMATH_GPT_tetrahedron_volume_ratio_l656_65644


namespace NUMINAMATH_GPT_simplify_sqrt_90000_l656_65623

theorem simplify_sqrt_90000 : Real.sqrt 90000 = 300 :=
by
  /- Proof goes here -/
  sorry

end NUMINAMATH_GPT_simplify_sqrt_90000_l656_65623


namespace NUMINAMATH_GPT_inequality_abc_l656_65626

theorem inequality_abc (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  (a + 1) * (b + 1) * (a + c) * (b + c) ≥ 16 * a * b * c :=
by
  sorry

end NUMINAMATH_GPT_inequality_abc_l656_65626


namespace NUMINAMATH_GPT_cyclic_quadrilateral_l656_65668

theorem cyclic_quadrilateral (T : ℕ) (S : ℕ) (AB BC CD DA : ℕ) (M N : ℝ × ℝ) (AC BD PQ MN : ℝ) (m n : ℕ) :
  T = 2378 → 
  S = 2 + 3 + 7 + 8 → 
  AB = S - 11 → 
  BC = 2 → 
  CD = 3 → 
  DA = 10 → 
  AC * BD = 47 → 
  PQ / MN = 1/2 → 
  m + n = 3 :=
by
  sorry

end NUMINAMATH_GPT_cyclic_quadrilateral_l656_65668


namespace NUMINAMATH_GPT_ratio_of_triangle_areas_l656_65670

theorem ratio_of_triangle_areas (a k : ℝ) (h_pos_a : 0 < a) (h_pos_k : 0 < k)
    (h_triangle_division : true) (h_square_area : ∃ s, s = a^2) (h_area_one_triangle : ∃ t, t = k * a^2) :
    ∃ r, r = (1 / (4 * k)) :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_triangle_areas_l656_65670


namespace NUMINAMATH_GPT_triangle_side_solution_l656_65681

/-- 
Given \( a \geq b \geq c > 0 \) and \( a < b + c \), a solution to the equation 
\( b \sqrt{x^{2} - c^{2}} + c \sqrt{x^{2} - b^{2}} = a x \) is provided by 
\( x = \frac{abc}{2 \sqrt{p(p-a)(p-b)(p-c)}} \) where \( p = \frac{1}{2}(a+b+c) \).
-/

theorem triangle_side_solution (a b c x : ℝ) (h1 : a ≥ b) (h2 : b ≥ c) (h3 : c > 0) (h4 : a < b + c) :
  b * (Real.sqrt (x^2 - c^2)) + c * (Real.sqrt (x^2 - b^2)) = a * x → 
  x = (a * b * c) / (2 * Real.sqrt ((a + b + c) / 2 * ((a + b + c) / 2 - a) * ((a + b + c) / 2 - b) * ((a + b + c) / 2 - c))) :=
sorry

end NUMINAMATH_GPT_triangle_side_solution_l656_65681


namespace NUMINAMATH_GPT_projection_cardinal_inequality_l656_65602

variables {Point : Type} [Fintype Point] [DecidableEq Point]

def projection_Oyz (S : Finset Point) : Finset Point := sorry
def projection_Ozx (S : Finset Point) : Finset Point := sorry
def projection_Oxy (S : Finset Point) : Finset Point := sorry

theorem projection_cardinal_inequality
  (S : Finset Point)
  (S_x := projection_Oyz S)
  (S_y := projection_Ozx S)
  (S_z := projection_Oxy S)
  : (Finset.card S)^2 ≤ (Finset.card S_x) * (Finset.card S_y) * (Finset.card S_z) :=
sorry

end NUMINAMATH_GPT_projection_cardinal_inequality_l656_65602


namespace NUMINAMATH_GPT_eggs_per_chicken_per_day_l656_65631

-- Define the conditions
def chickens : ℕ := 8
def price_per_dozen : ℕ := 5
def total_revenue : ℕ := 280
def weeks : ℕ := 4
def eggs_per_dozen : ℕ := 12
def days_per_week : ℕ := 7

-- Theorem statement on how many eggs each chicken lays per day
theorem eggs_per_chicken_per_day :
  (chickens * ((total_revenue / price_per_dozen * eggs_per_dozen) / (weeks * days_per_week))) / chickens = 3 :=
by
  sorry

end NUMINAMATH_GPT_eggs_per_chicken_per_day_l656_65631


namespace NUMINAMATH_GPT_salary_increase_l656_65648

theorem salary_increase (S0 S3 : ℕ) (r : ℕ) : 
  S0 = 3000 ∧ S3 = 8232 ∧ (S0 * (1 + r / 100)^3 = S3) → r = 40 :=
by
  sorry

end NUMINAMATH_GPT_salary_increase_l656_65648


namespace NUMINAMATH_GPT_polygon_sides_from_diagonals_l656_65697

theorem polygon_sides_from_diagonals (n : ℕ) (h : 20 = n * (n - 3) / 2) : n = 8 :=
sorry

end NUMINAMATH_GPT_polygon_sides_from_diagonals_l656_65697


namespace NUMINAMATH_GPT_calculate_expression_l656_65637

theorem calculate_expression : (3^3 * 4^3)^2 = 2985984 :=
  by
  sorry

end NUMINAMATH_GPT_calculate_expression_l656_65637


namespace NUMINAMATH_GPT_total_songs_correct_l656_65624

-- Define the conditions of the problem
def num_country_albums := 2
def songs_per_country_album := 12
def num_pop_albums := 8
def songs_per_pop_album := 7
def num_rock_albums := 5
def songs_per_rock_album := 10
def num_jazz_albums := 2
def songs_per_jazz_album := 15

-- Define the total number of songs
def total_songs :=
  num_country_albums * songs_per_country_album +
  num_pop_albums * songs_per_pop_album +
  num_rock_albums * songs_per_rock_album +
  num_jazz_albums * songs_per_jazz_album

-- Proposition stating the correct total number of songs
theorem total_songs_correct : total_songs = 160 :=
by {
  sorry -- Proof not required
}

end NUMINAMATH_GPT_total_songs_correct_l656_65624


namespace NUMINAMATH_GPT_cells_after_10_days_l656_65622

theorem cells_after_10_days :
  let a := 4
  let r := 2
  let n := 10 / 2
  let a_n := a * r ^ (n - 1)
  a_n = 64 :=
by
  let a := 4
  let r := 2
  let n := 10 / 2
  let a_n := a * r ^ (n - 1)
  show a_n = 64
  sorry

end NUMINAMATH_GPT_cells_after_10_days_l656_65622


namespace NUMINAMATH_GPT_integer_roots_abs_sum_l656_65695

theorem integer_roots_abs_sum (p q r n : ℤ) :
  (∃ n : ℤ, (∀ x : ℤ, x^3 - 2023 * x + n = 0) ∧ p + q + r = 0 ∧ p * q + q * r + r * p = -2023) →
  |p| + |q| + |r| = 102 :=
by
  sorry

end NUMINAMATH_GPT_integer_roots_abs_sum_l656_65695


namespace NUMINAMATH_GPT_minimum_value_of_f_l656_65642

noncomputable def f (x : ℝ) : ℝ :=
  x^2 + 1/x + 1/(x^2 + 1/x)

theorem minimum_value_of_f : 
  ∃ x > 0, f x = 2.5 :=
by 
  sorry

end NUMINAMATH_GPT_minimum_value_of_f_l656_65642


namespace NUMINAMATH_GPT_perfect_square_of_d_l656_65618

theorem perfect_square_of_d (a b c d : ℤ) (h : d = (a + (2:ℝ)^(1/3) * b + (4:ℝ)^(1/3) * c)^2) : ∃ k : ℤ, d = k^2 :=
by
  sorry

end NUMINAMATH_GPT_perfect_square_of_d_l656_65618


namespace NUMINAMATH_GPT_quadrilateral_inequality_l656_65613

theorem quadrilateral_inequality 
  (A B C D : Type)
  (AB AC AD BC BD CD : ℝ)
  (hAB_pos : 0 < AB)
  (hBC_pos : 0 < BC)
  (hCD_pos : 0 < CD)
  (hDA_pos : 0 < DA)
  (hAC_pos : 0 < AC)
  (hBD_pos : 0 < BD): 
  AC * BD ≤ AB * CD + BC * AD := 
sorry

end NUMINAMATH_GPT_quadrilateral_inequality_l656_65613


namespace NUMINAMATH_GPT_range_of_x_l656_65693

-- Problem Statement
theorem range_of_x (x : ℝ) (h : 0 ≤ x - 8) : 8 ≤ x :=
by {
  sorry
}

end NUMINAMATH_GPT_range_of_x_l656_65693


namespace NUMINAMATH_GPT_length_of_room_l656_65607

theorem length_of_room (b : ℕ) (t : ℕ) (L : ℕ) (blue_tiles : ℕ) (tile_area : ℕ) (total_area : ℕ) (effective_area : ℕ) (blue_area : ℕ) :
  b = 10 →
  t = 2 →
  blue_tiles = 16 →
  tile_area = t * t →
  total_area = (L - 4) * (b - 4) →
  blue_area = blue_tiles * tile_area →
  2 * blue_area = 3 * total_area →
  L = 20 :=
by
  intros h_b h_t h_blue_tiles h_tile_area h_total_area h_blue_area h_proportion
  sorry

end NUMINAMATH_GPT_length_of_room_l656_65607


namespace NUMINAMATH_GPT_only_one_positive_integer_n_l656_65614

theorem only_one_positive_integer_n (k : ℕ) (hk : 0 < k) (m : ℕ) (hm : k + 2 ≤ m) :
  ∃! (n : ℕ), 0 < n ∧ n^m ∣ 5^(n^k) + 1 :=
sorry

end NUMINAMATH_GPT_only_one_positive_integer_n_l656_65614


namespace NUMINAMATH_GPT_min_value_fraction_l656_65636

theorem min_value_fraction (x : ℝ) (hx : x < 2) : ∃ y : ℝ, y = (5 - 4 * x + x^2) / (2 - x) ∧ y = 2 :=
by sorry

end NUMINAMATH_GPT_min_value_fraction_l656_65636


namespace NUMINAMATH_GPT_max_value_g_l656_65665

def g (n : ℕ) : ℕ :=
  if n < 12 then n + 12 else g (n - 7)

theorem max_value_g : ∃ M, ∀ n, g n ≤ M ∧ M = 23 :=
  sorry

end NUMINAMATH_GPT_max_value_g_l656_65665


namespace NUMINAMATH_GPT_smallest_s_plus_d_l656_65666

theorem smallest_s_plus_d (s d : ℕ) (h_pos_s : s > 0) (h_pos_d : d > 0)
  (h_eq : 1 / s + 1 / (2 * s) + 1 / (3 * s) = 1 / (d^2 - 2 * d)) :
  s + d = 50 :=
sorry

end NUMINAMATH_GPT_smallest_s_plus_d_l656_65666


namespace NUMINAMATH_GPT_parabola_focus_l656_65612

noncomputable def focus_of_parabola (a b : ℝ) : ℝ × ℝ :=
  (0, 1 / (4 * a) - b)

theorem parabola_focus : focus_of_parabola 4 3 = (0, -47 / 16) :=
by
  -- Function definition: focus_of_parabola a b gives the focus of y = ax^2 - b
  -- Given: a = 4, b = 3
  -- Focus: (0, 1 / (4 * 4) - 3)
  -- Proof: Skipping detailed algebraic manipulation, assume function correctness
  sorry

end NUMINAMATH_GPT_parabola_focus_l656_65612


namespace NUMINAMATH_GPT_work_completion_time_l656_65617

theorem work_completion_time 
(w : ℝ)  -- total amount of work
(A B : ℝ)  -- work rate of a and b per day
(h1 : A + B = w / 30)  -- combined work rate
(h2 : 20 * (A + B) + 20 * A = w) : 
  (1 / A = 60) :=
sorry

end NUMINAMATH_GPT_work_completion_time_l656_65617


namespace NUMINAMATH_GPT_roger_allowance_spend_l656_65674

variable (A m s : ℝ)

-- Conditions from the problem
def condition1 : Prop := m = 0.25 * (A - 2 * s)
def condition2 : Prop := s = 0.10 * (A - 0.5 * m)
def goal : Prop := m + s = 0.59 * A

theorem roger_allowance_spend (h1 : condition1 A m s) (h2 : condition2 A m s) : goal A m s :=
  sorry

end NUMINAMATH_GPT_roger_allowance_spend_l656_65674


namespace NUMINAMATH_GPT_circle_center_radius_l656_65621

theorem circle_center_radius (x y : ℝ) :
  (x ^ 2 + y ^ 2 + 2 * x - 4 * y - 6 = 0) →
  ((x + 1) ^ 2 + (y - 2) ^ 2 = 11) :=
by sorry

end NUMINAMATH_GPT_circle_center_radius_l656_65621


namespace NUMINAMATH_GPT_ellipse_eccentricity_l656_65620

theorem ellipse_eccentricity (a b : ℝ) (h1 : a > b) (h2 : b > 0) 
    (B F A C : ℝ × ℝ) 
    (h3 : (B.1 ^ 2 / a ^ 2 + B.2 ^ 2 / b ^ 2 = 1))
    (h4 : (C.1 ^ 2 / a ^ 2 + C.2 ^ 2 / b ^ 2 = 1))
    (h5 : B.1 > 0 ∧ B.2 > 0)
    (h6 : C.1 > 0 ∧ C.2 > 0)
    (h7 : ∃ M : ℝ × ℝ, M = ((B.1 + C.1) / 2, (B.2 + C.2) / 2) ∧ (F = M)) :
    ∃ e : ℝ, e = (1 / 3) := 
  sorry

end NUMINAMATH_GPT_ellipse_eccentricity_l656_65620


namespace NUMINAMATH_GPT_derivative_given_limit_l656_65664

open Real

variable {f : ℝ → ℝ} {x₀ : ℝ}

theorem derivative_given_limit (h : ∀ ε > 0, ∃ δ > 0, ∀ Δx, abs Δx < δ → abs ((f (x₀ - 2 * Δx) - f x₀) / Δx + 2) < ε) :
  deriv f x₀ = -1 := by
  sorry

end NUMINAMATH_GPT_derivative_given_limit_l656_65664


namespace NUMINAMATH_GPT_power_function_value_l656_65609

theorem power_function_value (f : ℝ → ℝ) (h : ∀ x : ℝ, f x = x ^ (1 / 2)) (H : f 9 = 3) : f 25 = 5 :=
by
  sorry

end NUMINAMATH_GPT_power_function_value_l656_65609


namespace NUMINAMATH_GPT_find_number_l656_65657

def hash (a b : ℕ) : ℕ := a * b - b + b^2

theorem find_number :
  (∃ x : ℕ, hash 3 x = 63 ∧ x = 7) :=
sorry

end NUMINAMATH_GPT_find_number_l656_65657


namespace NUMINAMATH_GPT_quadratic_roots_value_r_l656_65694

theorem quadratic_roots_value_r
  (a b m p r : ℝ)
  (h1 : a ≠ 0)
  (h2 : b ≠ 0)
  (h_root1 : a^2 - m*a + 3 = 0)
  (h_root2 : b^2 - m*b + 3 = 0)
  (h_ab : a * b = 3)
  (h_root3 : (a + 1/b) * (b + 1/a) = r) :
  r = 16 / 3 :=
sorry

end NUMINAMATH_GPT_quadratic_roots_value_r_l656_65694


namespace NUMINAMATH_GPT_number_of_linear_eqs_l656_65688

def is_linear_eq_in_one_var (eq : String) : Bool :=
  match eq with
  | "0.3x = 1" => true
  | "x/2 = 5x + 1" => true
  | "x = 6" => true
  | _ => false

theorem number_of_linear_eqs :
  let eqs := ["x - 2 = 2 / x", "0.3x = 1", "x/2 = 5x + 1", "x^2 - 4x = 3", "x = 6", "x + 2y = 0"]
  (eqs.filter is_linear_eq_in_one_var).length = 3 :=
by
  sorry

end NUMINAMATH_GPT_number_of_linear_eqs_l656_65688


namespace NUMINAMATH_GPT_max_f_geq_l656_65646

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.sin (2 * x) + Real.sin (3 * x)

theorem max_f_geq (x : ℝ) : ∃ x, f x ≥ (3 + Real.sqrt 3) / 2 := sorry

end NUMINAMATH_GPT_max_f_geq_l656_65646


namespace NUMINAMATH_GPT_find_divisor_l656_65601

theorem find_divisor (d : ℕ) : 15 = (d * 4) + 3 → d = 3 := by
  intros h
  have h1 : 15 - 3 = 4 * d := by
    linarith
  have h2 : 12 = 4 * d := by
    linarith
  have h3 : d = 3 := by
    linarith
  exact h3

end NUMINAMATH_GPT_find_divisor_l656_65601


namespace NUMINAMATH_GPT_crescent_moon_area_l656_65659

theorem crescent_moon_area :
  let big_quarter_circle := (4 * 4 * Real.pi) / 4
  let small_semi_circle := (2 * 2 * Real.pi) / 2
  let crescent_area := big_quarter_circle - small_semi_circle
  crescent_area = 2 * Real.pi :=
by
  let big_quarter_circle := (4 * 4 * Real.pi) / 4
  let small_semi_circle := (2 * 2 * Real.pi) / 2
  let crescent_area := big_quarter_circle - small_semi_circle
  have h_bqc : big_quarter_circle = 4 * Real.pi := by
    sorry
  have h_ssc : small_semi_circle = 2 * Real.pi := by
    sorry
  have h_ca : crescent_area = 2 * Real.pi := by
    sorry
  exact h_ca

end NUMINAMATH_GPT_crescent_moon_area_l656_65659


namespace NUMINAMATH_GPT_principal_amount_l656_65600

theorem principal_amount (P : ℝ) (h : (P * 0.1236) - (P * 0.12) = 36) : P = 10000 := 
sorry

end NUMINAMATH_GPT_principal_amount_l656_65600


namespace NUMINAMATH_GPT_find_a_l656_65627

theorem find_a (x y a : ℤ) (h1 : x = 1) (h2 : y = 3) (h3 : a * x - 2 * y = 4) : a = 10 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_a_l656_65627


namespace NUMINAMATH_GPT_fx_properties_l656_65615

-- Definition of the function
def f (x : ℝ) : ℝ := x * |x|

-- Lean statement for the proof problem
theorem fx_properties :
  (∀ x : ℝ, f (-x) = -f x) ∧ (∀ x y : ℝ, 0 ≤ x → x ≤ y → f x ≤ f y) :=
by
  -- Definition used directly from the conditions
  sorry

end NUMINAMATH_GPT_fx_properties_l656_65615


namespace NUMINAMATH_GPT_expected_value_is_7_l656_65652

def win (n : ℕ) : ℕ :=
  if n % 2 = 1 then 2 * (10 - n) else 10 - n

def fair_die_values := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

def expected_value (values : List ℕ) (win : ℕ → ℕ) : ℚ :=
  (values.map (λ n => win n)).sum / values.length

theorem expected_value_is_7 :
  expected_value fair_die_values win = 7 := 
sorry

end NUMINAMATH_GPT_expected_value_is_7_l656_65652


namespace NUMINAMATH_GPT_arccos_sin_2_equals_l656_65625

theorem arccos_sin_2_equals : Real.arccos (Real.sin 2) = 2 - Real.pi / 2 := by
  sorry

end NUMINAMATH_GPT_arccos_sin_2_equals_l656_65625


namespace NUMINAMATH_GPT_point_reflection_l656_65682

-- Define the original point and the reflection function
structure Point where
  x : ℝ
  y : ℝ

def reflect_y_axis (p : Point) : Point :=
  ⟨-p.x, p.y⟩

-- Define the original point
def M : Point := ⟨-5, 2⟩

-- State the theorem to prove the reflection
theorem point_reflection : reflect_y_axis M = ⟨5, 2⟩ :=
  sorry

end NUMINAMATH_GPT_point_reflection_l656_65682


namespace NUMINAMATH_GPT_inequality_relationship_l656_65698

variable (a b : ℝ)

theorem inequality_relationship
  (h1 : a < 0)
  (h2 : -1 < b ∧ b < 0) : a < a * b^2 ∧ a * b^2 < a * b :=
by
  sorry

end NUMINAMATH_GPT_inequality_relationship_l656_65698


namespace NUMINAMATH_GPT_fraction_to_decimal_l656_65677

theorem fraction_to_decimal :
  (51 / 160 : ℝ) = 0.31875 := 
by
  sorry

end NUMINAMATH_GPT_fraction_to_decimal_l656_65677


namespace NUMINAMATH_GPT_slope_y_intercept_sum_l656_65654

theorem slope_y_intercept_sum 
  (m b : ℝ) 
  (h1 : (2 : ℝ) * m + b = -1) 
  (h2 : (5 : ℝ) * m + b = 2) : 
  m + b = -2 := 
sorry

end NUMINAMATH_GPT_slope_y_intercept_sum_l656_65654


namespace NUMINAMATH_GPT_max_areas_in_disk_l656_65629

noncomputable def max_non_overlapping_areas (n : ℕ) : ℕ := 5 * n + 1

theorem max_areas_in_disk (n : ℕ) : 
  let disk_divided_by_2n_radii_and_two_secant_lines_areas  := (5 * n + 1)
  disk_divided_by_2n_radii_and_two_secant_lines_areas = max_non_overlapping_areas n := by sorry

end NUMINAMATH_GPT_max_areas_in_disk_l656_65629


namespace NUMINAMATH_GPT_solve_in_primes_l656_65632

theorem solve_in_primes (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) :
  p^2 - 6 * p * q + q^2 + 3 * q - 1 = 0 ↔ (p = 17 ∧ q = 3) := 
sorry

end NUMINAMATH_GPT_solve_in_primes_l656_65632


namespace NUMINAMATH_GPT_amount_paid_to_shopkeeper_l656_65643

theorem amount_paid_to_shopkeeper :
  let price_of_grapes := 8 * 70
  let price_of_mangoes := 9 * 55
  price_of_grapes + price_of_mangoes = 1055 :=
by
  sorry

end NUMINAMATH_GPT_amount_paid_to_shopkeeper_l656_65643


namespace NUMINAMATH_GPT_apron_more_than_recipe_book_l656_65651

-- Define the prices and the total spent
def recipe_book_cost : ℕ := 6
def baking_dish_cost : ℕ := 2 * recipe_book_cost
def ingredient_cost : ℕ := 3
def total_ingredient_cost : ℕ := 5 * ingredient_cost
def total_spent : ℕ := 40

-- Define the condition that the total cost including the apron is $40
def total_without_apron : ℕ := recipe_book_cost + baking_dish_cost + total_ingredient_cost
def apron_cost : ℕ := total_spent - total_without_apron

-- Prove that the apron cost $1 more than the recipe book
theorem apron_more_than_recipe_book : apron_cost - recipe_book_cost = 1 := by
  -- The proof goes here
  sorry

end NUMINAMATH_GPT_apron_more_than_recipe_book_l656_65651


namespace NUMINAMATH_GPT_total_number_of_crayons_l656_65678

def number_of_blue_crayons := 3
def number_of_red_crayons := 4 * number_of_blue_crayons
def number_of_green_crayons := 2 * number_of_red_crayons
def number_of_yellow_crayons := number_of_green_crayons / 2

theorem total_number_of_crayons :
  number_of_blue_crayons + number_of_red_crayons + number_of_green_crayons + number_of_yellow_crayons = 51 :=
by 
  -- Proof is not required
  sorry

end NUMINAMATH_GPT_total_number_of_crayons_l656_65678


namespace NUMINAMATH_GPT_union_of_intervals_l656_65672

open Set

variable {α : Type*}

theorem union_of_intervals : 
  let A := Ioc (-1 : ℝ) 1
  let B := Ioo (0 : ℝ) 2
  A ∪ B = Ioo (-1 : ℝ) 2 := 
by
  let A := Ioc (-1 : ℝ) 1
  let B := Ioo (0 : ℝ) 2
  sorry

end NUMINAMATH_GPT_union_of_intervals_l656_65672


namespace NUMINAMATH_GPT_find_fraction_abs_l656_65662

-- Define the conditions and the main proof problem
theorem find_fraction_abs (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x^2 + y^2 = 5 * x * y) :
  abs ((x + y) / (x - y)) = Real.sqrt ((7 : ℝ) / 3) :=
by
  sorry

end NUMINAMATH_GPT_find_fraction_abs_l656_65662


namespace NUMINAMATH_GPT_solution_set_equality_of_abs_x_minus_2_gt_1_and_quadratic_gt_0_l656_65611

variable (a b : ℝ)

theorem solution_set_equality_of_abs_x_minus_2_gt_1_and_quadratic_gt_0 :
  (∀ x : ℝ, (|x - 2| > 1 ↔ x^2 + a * x + b > 0)) → a + b = -1 :=
by
  sorry

end NUMINAMATH_GPT_solution_set_equality_of_abs_x_minus_2_gt_1_and_quadratic_gt_0_l656_65611


namespace NUMINAMATH_GPT_train_speed_l656_65619

def train_length : ℝ := 400  -- Length of the train in meters
def crossing_time : ℝ := 40  -- Time to cross the electric pole in seconds

theorem train_speed : train_length / crossing_time = 10 := by
  sorry  -- Proof to be completed

end NUMINAMATH_GPT_train_speed_l656_65619


namespace NUMINAMATH_GPT_arithmetic_to_geometric_progression_l656_65605

theorem arithmetic_to_geometric_progression (x y z : ℝ) 
  (hAP : 2 * y^2 - y * x = z^2) : 
  z^2 = y * (2 * y - x) := 
  by 
  sorry

end NUMINAMATH_GPT_arithmetic_to_geometric_progression_l656_65605


namespace NUMINAMATH_GPT_non_real_roots_b_range_l656_65628

theorem non_real_roots_b_range (b : ℝ) : 
  ∃ (x : ℂ), x^2 + (b : ℂ) * x + 16 = 0 ∧ (¬ ∃ (x : ℝ), x^2 + b * x + 16 = 0) ↔ -8 < b ∧ b < 8 := 
by
  sorry

end NUMINAMATH_GPT_non_real_roots_b_range_l656_65628


namespace NUMINAMATH_GPT_value_of_x_l656_65680

theorem value_of_x (x y : ℝ) (h1 : x - y = 6) (h2 : x + y = 12) : x = 9 :=
by
  sorry

end NUMINAMATH_GPT_value_of_x_l656_65680


namespace NUMINAMATH_GPT_lying_dwarf_possible_numbers_l656_65635

theorem lying_dwarf_possible_numbers (a_1 a_2 a_3 a_4 a_5 a_6 a_7 : ℕ)
  (h2 : a_2 = a_1) 
  (h3 : a_3 = a_1 + a_2) 
  (h4 : a_4 = a_1 + a_2 + a_3) 
  (h5 : a_5 = a_1 + a_2 + a_3 + a_4) 
  (h6 : a_6 = a_1 + a_2 + a_3 + a_4 + a_5) 
  (h7 : a_7 = a_1 + a_2 + a_3 + a_4 + a_5 + a_6)
  (h_sum : a_1 + a_2 + a_3 + a_4 + a_5 + a_6 + a_7 = 58)
  (h_lied : ∃ (d : ℕ), d ∈ [a_1, a_2, a_3, a_4, a_5, a_6, a_7] ∧ (d ≠ a_1 ∧ d ≠ a_2 ∧ d ≠ a_3 ∧ d ≠ a_4 ∧ d ≠ a_5 ∧ d ≠ a_6 ∧ d ≠ a_7)) : 
  ∃ x : ℕ, x = 13 ∨ x = 26 :=
by
  sorry

end NUMINAMATH_GPT_lying_dwarf_possible_numbers_l656_65635


namespace NUMINAMATH_GPT_find_equation_line_l656_65656

noncomputable def line_through_point_area (A : Real × Real) (S : Real) : Prop :=
  ∃ (k : Real), (k < 0) ∧ (2 * A.1 + A.2 - 4 = 0) ∧
    (1 / 2 * (2 - k) * (1 - 2 / k) = S)

theorem find_equation_line (A : ℝ × ℝ) (S : ℝ) (hA : A = (1, 2)) (hS : S = 4) :
  line_through_point_area A S →
  ∃ l : ℝ → ℝ, ∀ x y : ℝ, y = l x ↔ 2 * x + y - 4 = 0 :=
by
  sorry

end NUMINAMATH_GPT_find_equation_line_l656_65656


namespace NUMINAMATH_GPT_not_perfect_cube_of_N_l656_65699

-- Define a twelve-digit number
def N : ℕ := 100000000000

-- Define the condition that a number is a perfect cube
def is_perfect_cube (n : ℕ) : Prop :=
  ∃ k : ℤ, n = k ^ 3

-- Problem statement: Prove that 100000000000 is not a perfect cube
theorem not_perfect_cube_of_N : ¬ is_perfect_cube N :=
by sorry

end NUMINAMATH_GPT_not_perfect_cube_of_N_l656_65699


namespace NUMINAMATH_GPT_age_sum_l656_65673

variable {S R K : ℝ}

theorem age_sum 
  (h1 : S = R + 10)
  (h2 : S + 12 = 3 * (R - 5))
  (h3 : K = R / 2) :
  S + R + K = 56.25 := 
by 
  sorry

end NUMINAMATH_GPT_age_sum_l656_65673


namespace NUMINAMATH_GPT_sarah_meets_vegetable_requirement_l656_65689

def daily_vegetable_requirement : ℝ := 2
def total_days : ℕ := 5
def weekly_requirement : ℝ := daily_vegetable_requirement * total_days

def sunday_consumption : ℝ := 3
def monday_consumption : ℝ := 1.5
def tuesday_consumption : ℝ := 1.5
def wednesday_consumption : ℝ := 1.5
def thursday_consumption : ℝ := 2.5

def total_consumption : ℝ := sunday_consumption + monday_consumption + tuesday_consumption + wednesday_consumption + thursday_consumption

theorem sarah_meets_vegetable_requirement : total_consumption = weekly_requirement :=
by
  sorry

end NUMINAMATH_GPT_sarah_meets_vegetable_requirement_l656_65689


namespace NUMINAMATH_GPT_solve_real_roots_in_intervals_l656_65658

noncomputable def real_roots_intervals (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : Prop :=
  ∃ x₁ x₂ : ℝ,
    (3 * x₁^2 - 2 * (a - b) * x₁ - a * b = 0) ∧
    (3 * x₂^2 - 2 * (a - b) * x₂ - a * b = 0) ∧
    (a / 3 < x₁ ∧ x₁ < 2 * a / 3) ∧
    (-2 * b / 3 < x₂ ∧ x₂ < -b / 3)

-- Statement of the problem:
theorem solve_real_roots_in_intervals (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  real_roots_intervals a b ha hb :=
sorry

end NUMINAMATH_GPT_solve_real_roots_in_intervals_l656_65658
