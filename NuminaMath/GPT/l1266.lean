import Mathlib

namespace NUMINAMATH_GPT_range_of_a_l1266_126673

noncomputable def f (a x : ℝ) : ℝ :=
  x^3 - 3 * a * x^2 + (2 * a + 1) * x

noncomputable def f' (a x : ℝ) : ℝ :=
  3 * x^2 - 6 * a * x + (2 * a + 1)

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, f' a x = 0 ∧ ∀ y : ℝ, f' a y ≠ 0) →
  (a > 1 ∨ a < -1 / 3) :=
sorry

end NUMINAMATH_GPT_range_of_a_l1266_126673


namespace NUMINAMATH_GPT_simplify_fraction_120_1800_l1266_126699

theorem simplify_fraction_120_1800 :
  (120 : ℚ) / 1800 = (1 : ℚ) / 15 := by
  sorry

end NUMINAMATH_GPT_simplify_fraction_120_1800_l1266_126699


namespace NUMINAMATH_GPT_smallest_k_equals_26_l1266_126637

open Real

-- Define the condition
def cos_squared_eq_one (θ : ℝ) : Prop :=
  cos θ ^ 2 = 1

-- Define the requirement for θ to be in the form 180°n
def theta_condition (n : ℤ) : Prop :=
  ∃ (k : ℤ), k ^ 2 + k + 81 = 180 * n

-- The problem statement in Lean: Find the smallest positive integer k such that
-- cos squared of (k^2 + k + 81) degrees = 1
noncomputable def smallest_k_satisfying_cos (k : ℤ) : Prop :=
  (∃ n : ℤ, theta_condition n ∧
   cos_squared_eq_one (k ^ 2 + k + 81)) ∧ (∀ m : ℤ, m > 0 ∧ m < k → 
   (∃ n : ℤ, theta_condition n ∧
   cos_squared_eq_one (m ^ 2 + m + 81)) → false)

theorem smallest_k_equals_26 : smallest_k_satisfying_cos 26 := 
  sorry

end NUMINAMATH_GPT_smallest_k_equals_26_l1266_126637


namespace NUMINAMATH_GPT_find_m_range_l1266_126617

def vector_a : ℝ × ℝ := (1, 2)
def dot_product (a b : ℝ × ℝ) : ℝ := (a.1 * b.1) + (a.2 * b.2)
def is_acute (a b : ℝ × ℝ) : Prop := dot_product a b > 0

theorem find_m_range (m : ℝ) :
  is_acute vector_a (4, m) → m ∈ Set.Ioo (-2 : ℝ) 8 ∪ Set.Ioi 8 := 
by
  sorry

end NUMINAMATH_GPT_find_m_range_l1266_126617


namespace NUMINAMATH_GPT_probability_A_l1266_126664

variable (A B : Prop)
variable (P : Prop → ℝ)

axiom prob_B : P B = 0.4
axiom prob_A_and_B : P (A ∧ B) = 0.15
axiom prob_notA_and_notB : P (¬ A ∧ ¬ B) = 0.5499999999999999

theorem probability_A : P A = 0.20 :=
by sorry

end NUMINAMATH_GPT_probability_A_l1266_126664


namespace NUMINAMATH_GPT_roger_forgot_lawns_l1266_126644

theorem roger_forgot_lawns
  (dollars_per_lawn : ℕ)
  (total_lawns : ℕ)
  (total_earned : ℕ)
  (actual_mowed_lawns : ℕ)
  (forgotten_lawns : ℕ)
  (h1 : dollars_per_lawn = 9)
  (h2 : total_lawns = 14)
  (h3 : total_earned = 54)
  (h4 : actual_mowed_lawns = total_earned / dollars_per_lawn) :
  forgotten_lawns = total_lawns - actual_mowed_lawns :=
  sorry

end NUMINAMATH_GPT_roger_forgot_lawns_l1266_126644


namespace NUMINAMATH_GPT_speed_against_current_l1266_126608

theorem speed_against_current (V_m V_c : ℝ) (h1 : V_m + V_c = 20) (h2 : V_c = 1) :
  V_m - V_c = 18 :=
by
  sorry

end NUMINAMATH_GPT_speed_against_current_l1266_126608


namespace NUMINAMATH_GPT_expand_expression_l1266_126630

theorem expand_expression (y z : ℝ) : 
  -2 * (5 * y^3 - 3 * y^2 * z + 4 * y * z^2 - z^3) = -10 * y^3 + 6 * y^2 * z - 8 * y * z^2 + 2 * z^3 :=
by sorry

end NUMINAMATH_GPT_expand_expression_l1266_126630


namespace NUMINAMATH_GPT_find_range_of_a_l1266_126668

noncomputable def range_of_a : Set ℝ :=
  {a | (∀ x : ℝ, x^2 - 2 * x > a) ∨ (∃ x0 : ℝ, x0^2 + 2 * a * x0 + 2 - a = 0)}

theorem find_range_of_a :
  {a : ℝ | (∀ x : ℝ, x^2 - 2 * x > a) ∨ (∃ x0 : ℝ, x0^2 + 2 * a * x0 + 2 - a = 0)} = 
  {a | (-2 < a ∧ a < -1) ∨ (1 ≤ a)} :=
by
  sorry

end NUMINAMATH_GPT_find_range_of_a_l1266_126668


namespace NUMINAMATH_GPT_triangle_side_b_range_l1266_126635

noncomputable def sin60 := Real.sin (Real.pi / 3)

theorem triangle_side_b_range (a b : ℝ) (A : ℝ)
  (ha : a = 2)
  (hA : A = 60 * Real.pi / 180)
  (h_2solutions : b * sin60 < a ∧ a < b) :
  (2 < b ∧ b < 4 * Real.sqrt 3 / 3) :=
by
  sorry

end NUMINAMATH_GPT_triangle_side_b_range_l1266_126635


namespace NUMINAMATH_GPT_chandra_valid_pairings_l1266_126696

def valid_pairings (num_bowls : ℕ) (num_glasses : ℕ) : ℕ :=
  num_bowls * num_glasses

theorem chandra_valid_pairings : valid_pairings 6 6 = 36 :=
  by sorry

end NUMINAMATH_GPT_chandra_valid_pairings_l1266_126696


namespace NUMINAMATH_GPT_ellipse_foci_distance_l1266_126659

noncomputable def distance_between_foci : ℝ :=
  let a := 20
  let b := 10
  2 * Real.sqrt (a ^ 2 - b ^ 2)

theorem ellipse_foci_distance : distance_between_foci = 20 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_GPT_ellipse_foci_distance_l1266_126659


namespace NUMINAMATH_GPT_exists_integers_cubes_sum_product_l1266_126624

theorem exists_integers_cubes_sum_product :
  ∃ (a b : ℤ), a^3 + b^3 = 91 ∧ a * b = 12 :=
by
  sorry

end NUMINAMATH_GPT_exists_integers_cubes_sum_product_l1266_126624


namespace NUMINAMATH_GPT_paint_mixture_replacement_l1266_126660

theorem paint_mixture_replacement :
  ∃ x y : ℝ,
    (0.5 * (1 - x) + 0.35 * x = 0.45) ∧
    (0.6 * (1 - y) + 0.45 * y = 0.55) ∧
    (x = 1 / 3) ∧
    (y = 1 / 3) :=
sorry

end NUMINAMATH_GPT_paint_mixture_replacement_l1266_126660


namespace NUMINAMATH_GPT_parabola_ratio_l1266_126680

-- Define the conditions and question as a theorem statement
theorem parabola_ratio
  (V₁ V₃ : ℝ × ℝ)
  (F₁ F₃ : ℝ × ℝ)
  (hV₁ : V₁ = (0, 0))
  (hF₁ : F₁ = (0, 1/8))
  (hV₃ : V₃ = (0, -1/2))
  (hF₃ : F₃ = (0, -1/4)) :
  dist F₁ F₃ / dist V₁ V₃ = 3 / 4 :=
  by
  sorry

end NUMINAMATH_GPT_parabola_ratio_l1266_126680


namespace NUMINAMATH_GPT_Richard_remaining_distance_l1266_126663

theorem Richard_remaining_distance
  (total_distance : ℕ)
  (day1_distance : ℕ)
  (day2_distance : ℕ)
  (day3_distance : ℕ)
  (half_and_subtract : day2_distance = (day1_distance / 2) - 6)
  (total_distance_to_walk : total_distance = 70)
  (distance_day1 : day1_distance = 20)
  (distance_day3 : day3_distance = 10)
  : total_distance - (day1_distance + day2_distance + day3_distance) = 36 :=
  sorry

end NUMINAMATH_GPT_Richard_remaining_distance_l1266_126663


namespace NUMINAMATH_GPT_symmetric_about_line_5pi12_l1266_126683

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 3)

theorem symmetric_about_line_5pi12 :
  ∀ x : ℝ, f (5 * Real.pi / 12 - x) = f (5 * Real.pi / 12 + x) :=
by
  intros x
  sorry

end NUMINAMATH_GPT_symmetric_about_line_5pi12_l1266_126683


namespace NUMINAMATH_GPT_Cary_height_is_72_l1266_126613

variable (Cary_height Bill_height Jan_height : ℕ)

-- Conditions
axiom Bill_height_is_half_Cary_height : Bill_height = Cary_height / 2
axiom Jan_height_is_6_inches_taller_than_Bill : Jan_height = Bill_height + 6
axiom Jan_height_is_42 : Jan_height = 42

-- Theorem statement
theorem Cary_height_is_72 : Cary_height = 72 := 
by
  sorry

end NUMINAMATH_GPT_Cary_height_is_72_l1266_126613


namespace NUMINAMATH_GPT_sampling_method_is_stratified_l1266_126678

/-- There are 500 boys and 400 girls in the high school senior year.
The total population consists of 900 students.
A random sample of 25 boys and 20 girls was taken.
Prove that the sampling method used is stratified sampling method. -/
theorem sampling_method_is_stratified :
    let boys := 500
    let girls := 400
    let total_students := 900
    let sample_boys := 25
    let sample_girls := 20
    let sampling_method := "Stratified sampling"
    sample_boys < boys ∧ sample_girls < girls → sampling_method = "Stratified sampling"
:=
sorry

end NUMINAMATH_GPT_sampling_method_is_stratified_l1266_126678


namespace NUMINAMATH_GPT_units_sold_at_original_price_l1266_126665

-- Define the necessary parameters and assumptions
variables (a x y : ℝ)
variables (total_units sold_original sold_discount sold_offseason : ℝ)
variables (purchase_price sell_price discount_price clearance_price : ℝ)

-- Define specific conditions
def purchase_units := total_units = 1000
def selling_price := sell_price = 1.25 * a
def discount_cond := discount_price = 1.25 * 0.9 * a
def clearance_cond := clearance_price = 1.25 * 0.60 * a
def holiday_limit := y ≤ 100
def profitability_condition := 1.25 * x + 1.25 * 0.9 * y + 1.25 * 0.60 * (1000 - x - y) > 1000 * a

-- The theorem asserting at least 426 units sold at the original price ensures profitability
theorem units_sold_at_original_price (h1 : total_units = 1000)
  (h2 : sell_price = 1.25 * a) (h3 : discount_price = 1.25 * 0.9 * a)
  (h4 : clearance_price = 1.25 * 0.60 * a) (h5 : y ≤ 100)
  (h6 : 1.25 * x + 1.25 * 0.9 * y + 1.25 * 0.60 * (1000 - x - y) > 1000 * a) :
  x ≥ 426 :=
by
  sorry

end NUMINAMATH_GPT_units_sold_at_original_price_l1266_126665


namespace NUMINAMATH_GPT_binary_calculation_l1266_126656

theorem binary_calculation :
  let b1 := 0b110110
  let b2 := 0b101110
  let b3 := 0b100
  let expected_result := 0b11100011110
  ((b1 * b2) / b3) = expected_result := by
  sorry

end NUMINAMATH_GPT_binary_calculation_l1266_126656


namespace NUMINAMATH_GPT_product_of_binomials_l1266_126666

theorem product_of_binomials (x : ℝ) : 
  (4 * x - 3) * (2 * x + 7) = 8 * x^2 + 22 * x - 21 := by
  sorry

end NUMINAMATH_GPT_product_of_binomials_l1266_126666


namespace NUMINAMATH_GPT_jiujiang_liansheng_sampling_l1266_126642

def bag_numbers : List ℕ := [7, 17, 27, 37, 47]

def systematic_sampling (N n : ℕ) (selected_bags : List ℕ) : Prop :=
  ∃ k i, k = N / n ∧ ∀ j, j < List.length selected_bags → selected_bags.get? j = some (i + k * j)

theorem jiujiang_liansheng_sampling :
  systematic_sampling 50 5 bag_numbers :=
by
  sorry

end NUMINAMATH_GPT_jiujiang_liansheng_sampling_l1266_126642


namespace NUMINAMATH_GPT_output_is_three_l1266_126605

-- Define the initial values
def initial_a : ℕ := 1
def initial_b : ℕ := 2

-- Define the final value of a after the computation
def final_a : ℕ := initial_a + initial_b

-- The theorem stating that the final value of a is 3
theorem output_is_three : final_a = 3 := by
  sorry

end NUMINAMATH_GPT_output_is_three_l1266_126605


namespace NUMINAMATH_GPT_find_c_eq_neg_9_over_4_l1266_126609

theorem find_c_eq_neg_9_over_4 (c x : ℚ) (h₁ : 3 * x + 5 = 1) (h₂ : c * x - 8 = -5) :
  c = -9 / 4 :=
sorry

end NUMINAMATH_GPT_find_c_eq_neg_9_over_4_l1266_126609


namespace NUMINAMATH_GPT_Gwen_still_has_money_in_usd_l1266_126674

open Real

noncomputable def exchange_rate : ℝ := 0.85
noncomputable def usd_gift : ℝ := 5.00
noncomputable def eur_gift : ℝ := 20.00
noncomputable def usd_spent_on_candy : ℝ := 3.25
noncomputable def eur_spent_on_toy : ℝ := 5.50

theorem Gwen_still_has_money_in_usd :
  let eur_conversion_to_usd := eur_gift / exchange_rate
  let total_usd_received := usd_gift + eur_conversion_to_usd
  let usd_spent_on_toy := eur_spent_on_toy / exchange_rate
  let total_usd_spent := usd_spent_on_candy + usd_spent_on_toy
  total_usd_received - total_usd_spent = 18.81 :=
by
  sorry

end NUMINAMATH_GPT_Gwen_still_has_money_in_usd_l1266_126674


namespace NUMINAMATH_GPT_cylinder_height_comparison_l1266_126640

theorem cylinder_height_comparison (r1 h1 r2 h2 : ℝ)
  (volume_eq : π * r1^2 * h1 = π * r2^2 * h2)
  (radius_relation : r2 = 1.2 * r1) :
  h1 = 1.44 * h2 :=
by {
  -- Proof steps here, not required per instruction
  sorry
}

end NUMINAMATH_GPT_cylinder_height_comparison_l1266_126640


namespace NUMINAMATH_GPT_spoon_less_than_fork_l1266_126622

-- Define the initial price of spoon and fork in kopecks
def initial_price (x : ℕ) : Prop :=
  x > 100 -- ensuring the spoon's sale price remains positive

-- Define the sale price of the spoon
def spoon_sale_price (x : ℕ) : ℕ :=
  x - 100

-- Define the sale price of the fork
def fork_sale_price (x : ℕ) : ℕ :=
  x / 10

-- Prove that the spoon's sale price can be less than the fork's sale price
theorem spoon_less_than_fork (x : ℕ) (h : initial_price x) : 
  spoon_sale_price x < fork_sale_price x :=
by
  sorry

end NUMINAMATH_GPT_spoon_less_than_fork_l1266_126622


namespace NUMINAMATH_GPT_clever_calculation_part1_clever_calculation_part2_clever_calculation_part3_l1266_126681

-- Prove that 46.3 * 0.56 + 5.37 * 5.6 + 1 * 0.056 equals 56.056
theorem clever_calculation_part1 : 46.3 * 0.56 + 5.37 * 5.6 + 1 * 0.056 = 56.056 :=
by
sorry

-- Prove that 101 * 92 - 92 equals 9200
theorem clever_calculation_part2 : 101 * 92 - 92 = 9200 :=
by
sorry

-- Prove that 36000 / 125 / 8 equals 36
theorem clever_calculation_part3 : 36000 / 125 / 8 = 36 :=
by
sorry

end NUMINAMATH_GPT_clever_calculation_part1_clever_calculation_part2_clever_calculation_part3_l1266_126681


namespace NUMINAMATH_GPT_problem1_problem2_l1266_126651

-- First proof problem
theorem problem1 (a b : ℝ) : a^4 + 6 * a^2 * b^2 + b^4 ≥ 4 * a * b * (a^2 + b^2) :=
by sorry

-- Second proof problem
theorem problem2 (a b : ℝ) : ∃ (x : ℝ), 
  (∀ (x : ℝ), |2 * x - a^4 + (1 - 6 * a^2 * b^2 - b^4)| + 2 * |x - (2 * a^3 * b + 2 * a * b^3 - 1)| ≥ 1) ∧
  ∃ (x : ℝ), |2 * x - a^4 + (1 - 6 * a^2 * b^2 - b^4)| + 2 * |x - (2 * a^3 * b + 2 * a * b^3 - 1)| = 1 :=
by sorry

end NUMINAMATH_GPT_problem1_problem2_l1266_126651


namespace NUMINAMATH_GPT_determine_lunch_break_duration_lunch_break_duration_in_minutes_l1266_126634

noncomputable def painter_lunch_break_duration (j h L : ℝ) : Prop :=
  (10 - L) * (j + h) = 0.6 ∧
  (8 - L) * h = 0.3 ∧
  (5 - L) * j = 0.1

theorem determine_lunch_break_duration (j h : ℝ) :
  ∃ L : ℝ, painter_lunch_break_duration j h L ∧ L = 0.8 :=
by sorry

theorem lunch_break_duration_in_minutes (j h : ℝ) :
  ∃ L : ℝ, painter_lunch_break_duration j h L ∧ L * 60 = 48 :=
by sorry

end NUMINAMATH_GPT_determine_lunch_break_duration_lunch_break_duration_in_minutes_l1266_126634


namespace NUMINAMATH_GPT_solve_for_x_l1266_126601

theorem solve_for_x (x : ℝ) (h : |3990 * x + 1995| = 1995) : x = 0 ∨ x = -1 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l1266_126601


namespace NUMINAMATH_GPT_surface_area_of_cube_l1266_126627

-- Define the volume condition
def volume_of_cube (s : ℝ) := s^3 = 125

-- Define the conversion from decimeters to centimeters
def decimeters_to_centimeters (d : ℝ) := d * 10

-- Define the surface area formula for one side of the cube
def surface_area_one_side (s_cm : ℝ) := s_cm^2

-- Prove that given the volume condition, the surface area of one side is 2500 cm²
theorem surface_area_of_cube
  (s : ℝ)
  (h : volume_of_cube s)
  (s_cm : ℝ := decimeters_to_centimeters s) :
  surface_area_one_side s_cm = 2500 :=
by
  sorry

end NUMINAMATH_GPT_surface_area_of_cube_l1266_126627


namespace NUMINAMATH_GPT_car_a_has_higher_avg_speed_l1266_126638

-- Definitions of the conditions for Car A
def distance_car_a : ℕ := 120
def speed_segment_1_car_a : ℕ := 60
def distance_segment_1_car_a : ℕ := 40
def speed_segment_2_car_a : ℕ := 40
def distance_segment_2_car_a : ℕ := 40
def speed_segment_3_car_a : ℕ := 80
def distance_segment_3_car_a : ℕ := distance_car_a - distance_segment_1_car_a - distance_segment_2_car_a

-- Definitions of the conditions for Car B
def distance_car_b : ℕ := 120
def time_segment_1_car_b : ℕ := 1
def speed_segment_1_car_b : ℕ := 60
def time_segment_2_car_b : ℕ := 1
def speed_segment_2_car_b : ℕ := 40
def total_time_car_b : ℕ := 3
def distance_segment_1_car_b := speed_segment_1_car_b * time_segment_1_car_b
def distance_segment_2_car_b := speed_segment_2_car_b * time_segment_2_car_b
def time_segment_3_car_b := total_time_car_b - time_segment_1_car_b - time_segment_2_car_b
def distance_segment_3_car_b := distance_car_b - distance_segment_1_car_b - distance_segment_2_car_b
def speed_segment_3_car_b := distance_segment_3_car_b / time_segment_3_car_b

-- Total Time for Car A
def time_car_a := distance_segment_1_car_a / speed_segment_1_car_a
                + distance_segment_2_car_a / speed_segment_2_car_a
                + distance_segment_3_car_a / speed_segment_3_car_a

-- Average Speed for Car A
def avg_speed_car_a := distance_car_a / time_car_a

-- Total Time for Car B
def time_car_b := total_time_car_b

-- Average Speed for Car B
def avg_speed_car_b := distance_car_b / time_car_b

-- Proof that Car A has a higher average speed than Car B
theorem car_a_has_higher_avg_speed : avg_speed_car_a > avg_speed_car_b := by sorry

end NUMINAMATH_GPT_car_a_has_higher_avg_speed_l1266_126638


namespace NUMINAMATH_GPT_solve_system_l1266_126614

theorem solve_system (x y : ℝ) :
  (x^3 - x + 1 = y^2 ∧ y^3 - y + 1 = x^2) ↔ ((x = 1 ∨ x = -1) ∧ (y = 1 ∨ y = -1)) :=
by
  sorry

end NUMINAMATH_GPT_solve_system_l1266_126614


namespace NUMINAMATH_GPT_walking_running_ratio_l1266_126682

theorem walking_running_ratio (d_w d_r : ℝ) (h1 : d_w / 4 + d_r / 8 = 3) (h2 : d_w + d_r = 16) :
  d_w / d_r = 1 := by
  sorry

end NUMINAMATH_GPT_walking_running_ratio_l1266_126682


namespace NUMINAMATH_GPT_school_orchestra_members_l1266_126602

theorem school_orchestra_members (total_members can_play_violin can_play_keyboard neither : ℕ)
    (h1 : total_members = 42)
    (h2 : can_play_violin = 25)
    (h3 : can_play_keyboard = 22)
    (h4 : neither = 3) :
    (can_play_violin + can_play_keyboard) - (total_members - neither) = 8 :=
by
  sorry

end NUMINAMATH_GPT_school_orchestra_members_l1266_126602


namespace NUMINAMATH_GPT_problem1_problem2_l1266_126629

-- Lean statement for Problem 1
theorem problem1 (x : ℝ) : x^2 * x^3 - x^5 = 0 := 
by sorry

-- Lean statement for Problem 2
theorem problem2 (a : ℝ) : (a + 1)^2 + 2 * a * (a - 1) = 3 * a^2 + 1 :=
by sorry

end NUMINAMATH_GPT_problem1_problem2_l1266_126629


namespace NUMINAMATH_GPT_intersection_area_two_circles_l1266_126671

theorem intersection_area_two_circles :
  let r : ℝ := 3
  let center1 : ℝ × ℝ := (3, 0)
  let center2 : ℝ × ℝ := (0, 3)
  let intersection_area := (9 * Real.pi - 18) / 2
  (∃ x y : ℝ, (x - center1.1)^2 + y^2 = r^2 ∧ x^2 + (y - center2.2)^2 = r^2) →
  (∃ (a : ℝ), a = intersection_area) :=
by
  sorry

end NUMINAMATH_GPT_intersection_area_two_circles_l1266_126671


namespace NUMINAMATH_GPT_min_le_one_fourth_sum_max_ge_four_ninths_sum_l1266_126677

variable (a b c : ℝ)

theorem min_le_one_fourth_sum
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0)
  (h_roots : b^2 - 4 * a * c ≥ 0) :
  min a (min b c) ≤ 1 / 4 * (a + b + c) :=
sorry

theorem max_ge_four_ninths_sum
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0)
  (h_roots : b^2 - 4 * a * c ≥ 0) :
  max a (max b c) ≥ 4 / 9 * (a + b + c) :=
sorry

end NUMINAMATH_GPT_min_le_one_fourth_sum_max_ge_four_ninths_sum_l1266_126677


namespace NUMINAMATH_GPT_motorboat_speeds_l1266_126661

theorem motorboat_speeds (v a x : ℝ) (d : ℝ)
  (h1 : ∀ t1 t2 t1' t2', 
        t1 = d / (v - a) ∧ t1' = d / (v + x - a) ∧ 
        t2 = d / (v + a) ∧ t2' = d / (v + a - x) ∧ 
        (t1 - t1' = t2' - t2)) 
        : x = 2 * a := 
sorry

end NUMINAMATH_GPT_motorboat_speeds_l1266_126661


namespace NUMINAMATH_GPT_find_number_l1266_126655

theorem find_number (y : ℝ) (h : 0.25 * 820 = 0.15 * y - 20) : y = 1500 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l1266_126655


namespace NUMINAMATH_GPT_shaded_square_cover_columns_l1266_126645

def triangular_number (n : Nat) : Nat := n * (n + 1) / 2

theorem shaded_square_cover_columns :
  ∃ n : Nat, 
    triangular_number n = 136 ∧ 
    ∀ i : Fin 10, ∃ k ≤ n, (triangular_number k) % 10 = i.val :=
sorry

end NUMINAMATH_GPT_shaded_square_cover_columns_l1266_126645


namespace NUMINAMATH_GPT_product_of_fractions_l1266_126676

-- Define the fractions
def one_fourth : ℚ := 1 / 4
def one_half : ℚ := 1 / 2
def one_eighth : ℚ := 1 / 8

-- State the theorem we are proving
theorem product_of_fractions :
  one_fourth * one_half = one_eighth :=
by
  sorry

end NUMINAMATH_GPT_product_of_fractions_l1266_126676


namespace NUMINAMATH_GPT_painters_work_l1266_126618

theorem painters_work (w1 w2 : ℕ) (d1 d2 : ℚ) (C : ℚ) (h1 : w1 * d1 = C) (h2 : w2 * d2 = C) (p : w1 = 5) (t : d1 = 1.6) (a : w2 = 4) : d2 = 2 := 
by
  sorry

end NUMINAMATH_GPT_painters_work_l1266_126618


namespace NUMINAMATH_GPT_opposite_sides_range_l1266_126641

theorem opposite_sides_range (a : ℝ) : (2 * 1 + 3 * a + 1) * (2 * a - 3 * 1 + 1) < 0 ↔ -1 < a ∧ a < 1 := sorry

end NUMINAMATH_GPT_opposite_sides_range_l1266_126641


namespace NUMINAMATH_GPT_sequence_formula_l1266_126623

theorem sequence_formula (a : ℕ → ℕ) (n : ℕ) (h : ∀ n ≥ 1, a n = a (n - 1) + n^3) : 
  a n = (n * (n + 1) / 2) ^ 2 := sorry

end NUMINAMATH_GPT_sequence_formula_l1266_126623


namespace NUMINAMATH_GPT_sequence_sum_l1266_126646

theorem sequence_sum : (1 - 3 + 5 - 7 + 9 - 11 + 13 - 15 + 17 - 19) = -10 :=
by
  sorry

end NUMINAMATH_GPT_sequence_sum_l1266_126646


namespace NUMINAMATH_GPT_minimum_m_plus_n_l1266_126632

theorem minimum_m_plus_n
  (m n : ℝ)
  (h_m_pos : m > 0)
  (h_n_pos : n > 0)
  (h_ellipse : 1 / m + 4 / n = 1) :
  m + n = 9 :=
sorry

end NUMINAMATH_GPT_minimum_m_plus_n_l1266_126632


namespace NUMINAMATH_GPT_number_of_possible_k_values_l1266_126658

theorem number_of_possible_k_values : 
  ∃ k_values : Finset ℤ, 
    (∀ k ∈ k_values, ∃ (x y : ℤ), y = x - 3 ∧ y = k * x - k) ∧
    k_values.card = 3 := 
sorry

end NUMINAMATH_GPT_number_of_possible_k_values_l1266_126658


namespace NUMINAMATH_GPT_compare_y1_y2_l1266_126672

def parabola (x : ℝ) (c : ℝ) : ℝ := -x^2 + 4 * x + c

theorem compare_y1_y2 (c y1 y2 : ℝ) :
  parabola (-1) c = y1 →
  parabola 1 c = y2 →
  y1 < y2 :=
by
  intro h1 h2
  sorry

end NUMINAMATH_GPT_compare_y1_y2_l1266_126672


namespace NUMINAMATH_GPT_call_center_agents_ratio_l1266_126648

theorem call_center_agents_ratio
  (a b : ℕ) -- Number of agents in teams A and B
  (x : ℝ) -- Calls each member of team B processes
  (h1 : (a : ℝ) / (b : ℝ) = 5 / 8)
  (h2 : b * x * 4 / 7 + a * 6 / 5 * x * 3 / 7 = b * x + a * 6 / 5 * x) :
  (a : ℝ) / (b : ℝ) = 5 / 8 :=
by
  sorry

end NUMINAMATH_GPT_call_center_agents_ratio_l1266_126648


namespace NUMINAMATH_GPT_jerry_bought_one_pound_of_pasta_sauce_l1266_126650

-- Definitions of the given conditions
def cost_mustard_oil_per_liter : ℕ := 13
def liters_mustard_oil : ℕ := 2
def cost_pasta_per_pound : ℕ := 4
def pounds_pasta : ℕ := 3
def cost_pasta_sauce_per_pound : ℕ := 5
def leftover_amount : ℕ := 7
def initial_amount : ℕ := 50

-- The goal to prove
theorem jerry_bought_one_pound_of_pasta_sauce :
  (initial_amount - leftover_amount - liters_mustard_oil * cost_mustard_oil_per_liter 
  - pounds_pasta * cost_pasta_per_pound) / cost_pasta_sauce_per_pound = 1 :=
by
  sorry

end NUMINAMATH_GPT_jerry_bought_one_pound_of_pasta_sauce_l1266_126650


namespace NUMINAMATH_GPT_new_person_weight_l1266_126639

theorem new_person_weight :
  (8 * 2.5 + 75 = 95) :=
by sorry

end NUMINAMATH_GPT_new_person_weight_l1266_126639


namespace NUMINAMATH_GPT_paint_left_after_two_coats_l1266_126684

theorem paint_left_after_two_coats :
  let initial_paint := 3 -- liters
  let first_coat_paint := initial_paint / 2
  let paint_after_first_coat := initial_paint - first_coat_paint
  let second_coat_paint := (2 / 3) * paint_after_first_coat
  let paint_after_second_coat := paint_after_first_coat - second_coat_paint
  (paint_after_second_coat * 1000) = 500 := by
  sorry

end NUMINAMATH_GPT_paint_left_after_two_coats_l1266_126684


namespace NUMINAMATH_GPT_solve_x_value_l1266_126657
-- Import the necessary libraries

-- Define the problem and the main theorem
theorem solve_x_value (x : ℝ) (h : 3 / x^2 = x / 27) : x = 3 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_solve_x_value_l1266_126657


namespace NUMINAMATH_GPT_original_price_of_cycle_l1266_126647

theorem original_price_of_cycle (P : ℝ) (h1 : 1440 = P + 0.6 * P) : P = 900 :=
by
  sorry

end NUMINAMATH_GPT_original_price_of_cycle_l1266_126647


namespace NUMINAMATH_GPT_solve_fractional_equation_l1266_126636

theorem solve_fractional_equation : ∀ x : ℝ, (2 * x / (x - 1) = 3) ↔ x = 3 := 
by
  sorry

end NUMINAMATH_GPT_solve_fractional_equation_l1266_126636


namespace NUMINAMATH_GPT_sufficient_not_necessary_condition_l1266_126695

noncomputable def setA (x : ℝ) : Prop := 
  (Real.log x / Real.log 2 - 1) * (Real.log x / Real.log 2 - 3) ≤ 0

noncomputable def setB (x : ℝ) (a : ℝ) : Prop := 
  (2 * x - a) / (x + 1) > 1

theorem sufficient_not_necessary_condition (a : ℝ) : 
  (∀ x, setA x → setB x a) ∧ (¬ ∀ x, setB x a → setA x) ↔ 
  -2 < a ∧ a < 1 := 
  sorry

end NUMINAMATH_GPT_sufficient_not_necessary_condition_l1266_126695


namespace NUMINAMATH_GPT_quadratic_inequality_solution_set_l1266_126625

theorem quadratic_inequality_solution_set :
  {x : ℝ | x^2 - 2 * x - 3 < 0} = {x : ℝ | -1 < x ∧ x < 3} :=
sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_set_l1266_126625


namespace NUMINAMATH_GPT_A_share_of_profit_l1266_126691

theorem A_share_of_profit
  (A_investment : ℤ) (B_investment : ℤ) (C_investment : ℤ)
  (A_profit_share : ℚ) (B_profit_share : ℚ) (C_profit_share : ℚ)
  (total_profit : ℤ) :
  A_investment = 6300 ∧ B_investment = 4200 ∧ C_investment = 10500 ∧
  A_profit_share = 0.45 ∧ B_profit_share = 0.3 ∧ C_profit_share = 0.25 ∧ 
  total_profit = 12200 →
  A_profit_share * total_profit = 5490 :=
by sorry

end NUMINAMATH_GPT_A_share_of_profit_l1266_126691


namespace NUMINAMATH_GPT_student_factor_l1266_126688

theorem student_factor (x : ℤ) : (121 * x - 138 = 104) → x = 2 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_student_factor_l1266_126688


namespace NUMINAMATH_GPT_roots_conditions_l1266_126669

theorem roots_conditions (α β m n : ℝ) (h_pos : β > 0)
  (h1 : α + 2 * β = -m)
  (h2 : 2 * α * β + β^2 = -3)
  (h3 : α * β^2 = -n)
  (h4 : α^2 + 2 * β^2 = 6) : 
  m = 0 ∧ n = 2 := by
  sorry

end NUMINAMATH_GPT_roots_conditions_l1266_126669


namespace NUMINAMATH_GPT_muffins_divide_equally_l1266_126620

theorem muffins_divide_equally (friends : ℕ) (total_muffins : ℕ) (Jessie_and_friends : ℕ) (muffins_per_person : ℕ) :
  friends = 6 →
  total_muffins = 35 →
  Jessie_and_friends = friends + 1 →
  muffins_per_person = total_muffins / Jessie_and_friends →
  muffins_per_person = 5 :=
by
  intros h_friends h_muffins h_people h_division
  sorry

end NUMINAMATH_GPT_muffins_divide_equally_l1266_126620


namespace NUMINAMATH_GPT_sum_last_two_digits_15_pow_25_plus_5_pow_25_mod_100_l1266_126697

theorem sum_last_two_digits_15_pow_25_plus_5_pow_25_mod_100 : 
  (15^25 + 5^25) % 100 = 0 := 
by
  sorry

end NUMINAMATH_GPT_sum_last_two_digits_15_pow_25_plus_5_pow_25_mod_100_l1266_126697


namespace NUMINAMATH_GPT_find_point_C_l1266_126604

def point := ℝ × ℝ
def is_midpoint (M A B : point) : Prop := (2 * M.1 = A.1 + B.1) ∧ (2 * M.2 = A.2 + B.2)

-- Variables for known points
def A : point := (2, 8)
def M : point := (4, 11)
def L : point := (6, 6)

-- The proof problem: Prove the coordinates of point C
theorem find_point_C (C : point) (B : point) :
  is_midpoint M A B →
  -- (additional conditions related to the angle bisector can be added if specified)
  C = (14, 2) :=
sorry

end NUMINAMATH_GPT_find_point_C_l1266_126604


namespace NUMINAMATH_GPT_fruit_order_count_l1266_126621

-- Define the initial conditions
def apples := 3
def oranges := 2
def bananas := 2
def totalFruits := apples + oranges + bananas -- which is 7

-- Calculate the factorial of a number
def fact : ℕ → ℕ
| 0       => 1
| (n + 1) => (n + 1) * fact n

-- Noncomputable definition to skip proof
noncomputable def distinctOrders : ℕ :=
  fact totalFruits / (fact apples * fact oranges * fact bananas)

-- Lean statement expressing that the number of distinct orders is 210
theorem fruit_order_count : distinctOrders = 210 :=
by
  sorry

end NUMINAMATH_GPT_fruit_order_count_l1266_126621


namespace NUMINAMATH_GPT_days_of_harvest_l1266_126686

-- Conditions
def ripeOrangesPerDay : ℕ := 82
def totalRipeOranges : ℕ := 2050

-- Problem statement: Prove the number of days of harvest
theorem days_of_harvest : (totalRipeOranges / ripeOrangesPerDay) = 25 :=
by
  sorry

end NUMINAMATH_GPT_days_of_harvest_l1266_126686


namespace NUMINAMATH_GPT_average_rounds_rounded_is_3_l1266_126649

-- Definitions based on conditions
def golfers : List ℕ := [3, 4, 3, 6, 2, 4]
def rounds : List ℕ := [0, 1, 2, 3, 4, 5]

noncomputable def total_rounds : ℕ :=
  List.sum (List.zipWith (λ g r => g * r) golfers rounds)

def total_golfers : ℕ := List.sum golfers

noncomputable def average_rounds : ℕ :=
  Int.natAbs (Int.ofNat total_rounds / total_golfers).toNat

theorem average_rounds_rounded_is_3 : average_rounds = 3 := by
  sorry

end NUMINAMATH_GPT_average_rounds_rounded_is_3_l1266_126649


namespace NUMINAMATH_GPT_max_value_expression_l1266_126626

theorem max_value_expression (a b c : ℝ) (h : a^2 + b^2 + c^2 = 9) : 
  (a - b)^2 + (b - c)^2 + (c - a)^2 ≤ 27 := 
by sorry

end NUMINAMATH_GPT_max_value_expression_l1266_126626


namespace NUMINAMATH_GPT_ladder_length_l1266_126692

variable (x y : ℝ)

theorem ladder_length :
  (x^2 = 15^2 + y^2) ∧ (x^2 = 24^2 + (y - 13)^2) → x = 25 := by
  sorry

end NUMINAMATH_GPT_ladder_length_l1266_126692


namespace NUMINAMATH_GPT_regression_decrease_by_three_l1266_126667

-- Given a regression equation \hat y = 2 - 3 \hat x
def regression_equation (x : ℝ) : ℝ :=
  2 - 3 * x

-- Prove that when x increases by one unit, \hat y decreases by 3 units
theorem regression_decrease_by_three (x : ℝ) :
  regression_equation (x + 1) - regression_equation x = -3 :=
by
  -- proof
  sorry

end NUMINAMATH_GPT_regression_decrease_by_three_l1266_126667


namespace NUMINAMATH_GPT_black_equals_sum_of_white_l1266_126662

theorem black_equals_sum_of_white :
  ∃ (a b c d : ℤ) (a_neq_zero : a ≠ 0) (b_neq_zero : b ≠ 0) (c_neq_zero : c ≠ 0) (d_neq_zero : d ≠ 0),
    (c + d * Real.sqrt 7 = (Real.sqrt (a + b * Real.sqrt 2) + Real.sqrt (a - b * Real.sqrt 2))^2) :=
by
  sorry

end NUMINAMATH_GPT_black_equals_sum_of_white_l1266_126662


namespace NUMINAMATH_GPT_constant_value_l1266_126653

noncomputable def find_constant (p q : ℚ) (h : p / q = 4 / 5) : ℚ :=
    let C := 0.5714285714285714 - (2 * q - p) / (2 * q + p)
    C

theorem constant_value (p q : ℚ) (h : p / q = 4 / 5) :
    find_constant p q h = 0.14285714285714285 := by
    sorry

end NUMINAMATH_GPT_constant_value_l1266_126653


namespace NUMINAMATH_GPT_least_value_of_q_minus_p_l1266_126654

variables (y p q : ℝ)

/-- Triangle side lengths -/
def BC := y + 7
def AC := y + 3
def AB := 2 * y + 1

/-- Given conditions for triangle inequalities and angle B being the largest -/
def triangle_inequality_conditions :=
  (y + 7 + (y + 3) > 2 * y + 1) ∧
  (y + 7 + (2 * y + 1) > y + 3) ∧
  ((y + 3) + (2 * y + 1) > y + 7)

def angle_largest_conditions :=
  (2 * y + 1 > y + 3) ∧
  (2 * y + 1 > y + 7)

/-- Prove the least possible value of q - p given the conditions -/
theorem least_value_of_q_minus_p
  (h1 : triangle_inequality_conditions y)
  (h2 : angle_largest_conditions y)
  (h3 : 6 < y)
  (h4 : y < 8) :
  q - p = 2 := sorry

end NUMINAMATH_GPT_least_value_of_q_minus_p_l1266_126654


namespace NUMINAMATH_GPT_pen_and_notebook_cost_l1266_126631

theorem pen_and_notebook_cost (pen_cost : ℝ) (notebook_cost : ℝ) 
  (h1 : pen_cost = 4.5) 
  (h2 : pen_cost = notebook_cost + 1.8) : 
  pen_cost + notebook_cost = 7.2 := 
  by
    sorry

end NUMINAMATH_GPT_pen_and_notebook_cost_l1266_126631


namespace NUMINAMATH_GPT_trapezium_area_l1266_126687

-- Definitions based on the problem conditions
def length_side_a : ℝ := 20
def length_side_b : ℝ := 18
def distance_between_sides : ℝ := 15

-- Statement of the proof problem
theorem trapezium_area :
  (1 / 2 * (length_side_a + length_side_b) * distance_between_sides) = 285 := by
  sorry

end NUMINAMATH_GPT_trapezium_area_l1266_126687


namespace NUMINAMATH_GPT_solution_set_a_eq_1_no_positive_a_for_all_x_l1266_126698

-- Define the original inequality for a given a.
def inequality (a x : ℝ) : Prop := |a * x - 1| + |a * x - a| ≥ 2

-- Part 1: For a = 1
theorem solution_set_a_eq_1 :
  {x : ℝ | inequality 1 x } = {x : ℝ | x ≤ 0 ∨ x ≥ 2} :=
sorry

-- Part 2: There is no positive a such that the inequality holds for all x ∈ ℝ
theorem no_positive_a_for_all_x :
  ¬ ∃ a > 0, ∀ x : ℝ, inequality a x :=
sorry

end NUMINAMATH_GPT_solution_set_a_eq_1_no_positive_a_for_all_x_l1266_126698


namespace NUMINAMATH_GPT_average_in_all_6_subjects_l1266_126603

-- Definitions of the conditions
def average_in_5_subjects : ℝ := 74
def marks_in_6th_subject : ℝ := 104
def num_subjects_total : ℝ := 6

-- Proof that the average in all 6 subjects is 79
theorem average_in_all_6_subjects :
  (average_in_5_subjects * 5 + marks_in_6th_subject) / num_subjects_total = 79 := by
  sorry

end NUMINAMATH_GPT_average_in_all_6_subjects_l1266_126603


namespace NUMINAMATH_GPT_math_problem_l1266_126628

theorem math_problem (n d : ℕ) (h1 : 0 < n) (h2 : d < 10)
  (h3 : 3 * n^2 + 2 * n + d = 263)
  (h4 : 3 * n^2 + 2 * n + 4 = 396 + 7 * d) :
  n + d = 11 :=
by {
  sorry
}

end NUMINAMATH_GPT_math_problem_l1266_126628


namespace NUMINAMATH_GPT_p_sufficient_not_necessary_l1266_126689

theorem p_sufficient_not_necessary:
  (∀ a b : ℝ, a > b ∧ b > 0 → (1 / a^2 < 1 / b^2)) ∧ 
  (∃ a b : ℝ, (1 / a^2 < 1 / b^2) ∧ ¬ (a > b ∧ b > 0)) :=
sorry

end NUMINAMATH_GPT_p_sufficient_not_necessary_l1266_126689


namespace NUMINAMATH_GPT_linda_change_l1266_126612

-- Defining the conditions
def cost_per_banana : ℝ := 0.30
def number_of_bananas : ℕ := 5
def amount_paid : ℝ := 10.00

-- Proving the statement
theorem linda_change :
  amount_paid - (number_of_bananas * cost_per_banana) = 8.50 :=
by
  sorry

end NUMINAMATH_GPT_linda_change_l1266_126612


namespace NUMINAMATH_GPT_prime_count_60_to_70_l1266_126615

theorem prime_count_60_to_70 : ∃ primes : Finset ℕ, primes.card = 2 ∧ ∀ p ∈ primes, 60 < p ∧ p < 70 ∧ Nat.Prime p :=
by
  sorry

end NUMINAMATH_GPT_prime_count_60_to_70_l1266_126615


namespace NUMINAMATH_GPT_abs_has_min_at_zero_l1266_126679

def f (x : ℝ) : ℝ := abs x

theorem abs_has_min_at_zero : ∃ m, (∀ x : ℝ, f x ≥ m) ∧ f 0 = m := by
  sorry

end NUMINAMATH_GPT_abs_has_min_at_zero_l1266_126679


namespace NUMINAMATH_GPT_find_sum_of_x_and_y_l1266_126611

theorem find_sum_of_x_and_y (x y : ℝ) (h : x^2 + y^2 = 8 * x - 4 * y - 20) : x + y = 2 := 
by
  sorry

end NUMINAMATH_GPT_find_sum_of_x_and_y_l1266_126611


namespace NUMINAMATH_GPT_oranges_thrown_away_l1266_126616

theorem oranges_thrown_away (original_oranges: ℕ) (new_oranges: ℕ) (total_oranges: ℕ) (x: ℕ)
  (h1: original_oranges = 5) (h2: new_oranges = 28) (h3: total_oranges = 31) :
  original_oranges - x + new_oranges = total_oranges → x = 2 :=
by
  intros h_eq
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_oranges_thrown_away_l1266_126616


namespace NUMINAMATH_GPT_correct_result_l1266_126694

-- Define the original number
def original_number := 51 + 6

-- Define the correct calculation using multiplication
def correct_calculation (x : ℕ) : ℕ := x * 6

-- Theorem to prove the correct calculation
theorem correct_result : correct_calculation original_number = 342 := by
  -- Skip the actual proof steps
  sorry

end NUMINAMATH_GPT_correct_result_l1266_126694


namespace NUMINAMATH_GPT_ivar_total_water_needed_l1266_126685

-- Define the initial number of horses
def initial_horses : ℕ := 3

-- Define the added horses
def added_horses : ℕ := 5

-- Define the total number of horses
def total_horses : ℕ := initial_horses + added_horses

-- Define water consumption per horse per day for drinking
def water_consumption_drinking : ℕ := 5

-- Define water consumption per horse per day for bathing
def water_consumption_bathing : ℕ := 2

-- Define total water consumption per horse per day
def total_water_consumption_per_horse_per_day : ℕ := 
    water_consumption_drinking + water_consumption_bathing

-- Define total daily water consumption for all horses
def daily_water_consumption_all_horses : ℕ := 
    total_horses * total_water_consumption_per_horse_per_day

-- Define total water consumption over 28 days
def total_water_consumption_28_days : ℕ := 
    daily_water_consumption_all_horses * 28

-- State the theorem
theorem ivar_total_water_needed : 
    total_water_consumption_28_days = 1568 := 
by
  sorry

end NUMINAMATH_GPT_ivar_total_water_needed_l1266_126685


namespace NUMINAMATH_GPT_union_M_N_l1266_126607

-- Definitions based on conditions
def M : Set ℕ := {0, 1, 2}
def N : Set ℕ := {x | ∃ a, a ∈ M ∧ x = 2 * a}

-- The theorem to be proven
theorem union_M_N : M ∪ N = {0, 1, 2, 4} := by
  sorry

end NUMINAMATH_GPT_union_M_N_l1266_126607


namespace NUMINAMATH_GPT_find_base_of_denominator_l1266_126670

theorem find_base_of_denominator 
  (some_base : ℕ)
  (h1 : (1/2)^16 * (1/81)^8 = 1 / some_base^16) : 
  some_base = 18 :=
sorry

end NUMINAMATH_GPT_find_base_of_denominator_l1266_126670


namespace NUMINAMATH_GPT_price_of_child_ticket_l1266_126619

theorem price_of_child_ticket (C : ℝ) 
  (adult_ticket_price : ℝ := 8) 
  (total_tickets_sold : ℕ := 34) 
  (adult_tickets_sold : ℕ := 12) 
  (total_revenue : ℝ := 236) 
  (h1 : 12 * adult_ticket_price + (34 - 12) * C = total_revenue) :
  C = 6.36 :=
by
  sorry

end NUMINAMATH_GPT_price_of_child_ticket_l1266_126619


namespace NUMINAMATH_GPT_squares_form_acute_triangle_l1266_126606

theorem squares_form_acute_triangle (a b c x y z d : ℝ)
    (h_triangle : ∀ x y z : ℝ, (x > 0 ∧ y > 0 ∧ z > 0) → (x + y > z) ∧ (x + z > y) ∧ (y + z > x))
    (h_acute : ∀ x y z : ℝ, (x^2 + y^2 > z^2) ∧ (x^2 + z^2 > y^2) ∧ (y^2 + z^2 > x^2))
    (h_inscribed_squares : x = a ^ 2 * b * c / (d * a + b * c) ∧
                           y = b ^ 2 * a * c / (d * b + a * c) ∧
                           z = c ^ 2 * a * b / (d * c + a * b)) :
    (x + y > z) ∧ (x + z > y) ∧ (y + z > x) ∧
    (x^2 + y^2 > z^2) ∧ (x^2 + z^2 > y^2) ∧ (y^2 + z^2 > x^2) :=
sorry

end NUMINAMATH_GPT_squares_form_acute_triangle_l1266_126606


namespace NUMINAMATH_GPT_intersecting_point_value_l1266_126633

theorem intersecting_point_value
  (b a : ℤ)
  (h1 : a = -2 * 2 + b)
  (h2 : 2 = -2 * a + b) :
  a = 2 :=
by
  sorry

end NUMINAMATH_GPT_intersecting_point_value_l1266_126633


namespace NUMINAMATH_GPT_union_A_B_intersection_complement_A_B_l1266_126690

def A := {x : ℝ | 3 ≤ x ∧ x < 7}
def B := {x : ℝ | 4 < x ∧ x < 10}

theorem union_A_B :
  A ∪ B = {x : ℝ | 3 ≤ x ∧ x < 10} :=
sorry

def complement_A := {x : ℝ | x < 3 ∨ x ≥ 7}

theorem intersection_complement_A_B :
  (complement_A ∩ B) = {x : ℝ | 7 ≤ x ∧ x < 10} :=
sorry

end NUMINAMATH_GPT_union_A_B_intersection_complement_A_B_l1266_126690


namespace NUMINAMATH_GPT_find_i_value_for_S_i_l1266_126643

theorem find_i_value_for_S_i :
  ∃ (i : ℕ), (3 * 6 - 2 ≤ i ∧ i < 3 * 6 + 1) ∧ (1000 ≤ 31 * 2^6) ∧ (31 * 2^6 ≤ 3000) ∧ i = 2 :=
by sorry

end NUMINAMATH_GPT_find_i_value_for_S_i_l1266_126643


namespace NUMINAMATH_GPT_polynomial_simplification_l1266_126675

theorem polynomial_simplification (x : ℝ) :
    (3 * x - 2) * (5 * x^12 - 3 * x^11 + 4 * x^9 - 2 * x^8)
    = 15 * x^13 - 19 * x^12 + 6 * x^11 + 12 * x^10 - 14 * x^9 - 4 * x^8 := by
  sorry

end NUMINAMATH_GPT_polynomial_simplification_l1266_126675


namespace NUMINAMATH_GPT_total_weight_correct_l1266_126600

-- Define the constant variables as per the conditions
def jug1_capacity : ℝ := 2
def jug2_capacity : ℝ := 3
def fill_percentage : ℝ := 0.7
def jug1_density : ℝ := 4
def jug2_density : ℝ := 5

-- Define the volumes of sand in each jug
def jug1_sand_volume : ℝ := fill_percentage * jug1_capacity
def jug2_sand_volume : ℝ := fill_percentage * jug2_capacity

-- Define the weights of sand in each jug
def jug1_weight : ℝ := jug1_sand_volume * jug1_density
def jug2_weight : ℝ := jug2_sand_volume * jug2_density

-- State the theorem that combines the weights
theorem total_weight_correct : jug1_weight + jug2_weight = 16.1 := sorry

end NUMINAMATH_GPT_total_weight_correct_l1266_126600


namespace NUMINAMATH_GPT_convex_100gon_distinct_numbers_l1266_126693

theorem convex_100gon_distinct_numbers :
  ∀ (vertices : Fin 100 → (ℕ × ℕ)),
  (∀ i, (vertices i).1 ≠ (vertices i).2) →
  ∃ (erase_one_number : ∀ (i : Fin 100), ℕ),
  (∀ i, erase_one_number i = (vertices i).1 ∨ erase_one_number i = (vertices i).2) ∧
  (∀ i j, i ≠ j → (i = j + 1 ∨ (i = 0 ∧ j = 99)) → erase_one_number i ≠ erase_one_number j) :=
by sorry

end NUMINAMATH_GPT_convex_100gon_distinct_numbers_l1266_126693


namespace NUMINAMATH_GPT_weekend_price_is_correct_l1266_126610

-- Define the original price of the jacket
def original_price : ℝ := 250

-- Define the first discount rate (40%)
def first_discount_rate : ℝ := 0.40

-- Define the additional weekend discount rate (10%)
def additional_discount_rate : ℝ := 0.10

-- Define a function to apply the first discount
def apply_first_discount (price : ℝ) (discount : ℝ) : ℝ :=
  price * (1 - discount)

-- Define a function to apply the additional discount
def apply_additional_discount (price : ℝ) (discount : ℝ) : ℝ :=
  price * (1 - discount)

-- Using both discounts, calculate the final weekend price
def weekend_price : ℝ :=
  apply_additional_discount (apply_first_discount original_price first_discount_rate) additional_discount_rate

-- The final theorem stating the expected weekend price is $135
theorem weekend_price_is_correct : weekend_price = 135 := by
  sorry

end NUMINAMATH_GPT_weekend_price_is_correct_l1266_126610


namespace NUMINAMATH_GPT_geometric_sequence_angle_count_l1266_126652

theorem geometric_sequence_angle_count :
  (∃ θs : Finset ℝ, (∀ θ ∈ θs, 0 < θ ∧ θ < 2 * π ∧ ¬ ∃ k : ℕ, θ = k * (π / 2)) 
                    ∧ θs.card = 4
                    ∧ ∀ θ ∈ θs, ∃ a b c : ℝ, (a, b, c) = (Real.sin θ, Real.cos θ, Real.tan θ) 
                                             ∨ (a, b) = (Real.sin θ, Real.tan θ) 
                                             ∨ (a, b) = (Real.cos θ, Real.tan θ)
                                             ∧ b = a * c) :=
sorry

end NUMINAMATH_GPT_geometric_sequence_angle_count_l1266_126652
