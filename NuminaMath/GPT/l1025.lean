import Mathlib

namespace NUMINAMATH_GPT_problem_part_1_problem_part_2_l1025_102597

variable (θ : Real)
variable (m : Real)
variable (h_θ : θ ∈ Ioc 0 (2 * Real.pi))
variable (h_eq : ∀ x, 2 * x^2 - (Real.sqrt 3 + 1) * x + m = 0 ↔ (x = Real.sin θ ∨ x = Real.cos θ))

theorem problem_part_1 : 
  (Real.sin θ)^2 / (Real.sin θ - Real.cos θ) + (Real.cos θ)^2 / (Real.cos θ - Real.sin θ) = (Real.sqrt 3 + 1) / 2 := 
by
  sorry

theorem problem_part_2 : 
  m = Real.sqrt 3 / 2 := 
by 
  sorry

end NUMINAMATH_GPT_problem_part_1_problem_part_2_l1025_102597


namespace NUMINAMATH_GPT_line_plane_intersection_l1025_102502

theorem line_plane_intersection 
  (t : ℝ)
  (x_eq : ∀ t: ℝ, x = 5 - t)
  (y_eq : ∀ t: ℝ, y = -3 + 5 * t)
  (z_eq : ∀ t: ℝ, z = 1 + 2 * t)
  (plane_eq : 3 * x + 7 * y - 5 * z - 11 = 0)
  : x = 4 ∧ y = 2 ∧ z = 3 :=
by
  sorry

end NUMINAMATH_GPT_line_plane_intersection_l1025_102502


namespace NUMINAMATH_GPT_x_squared_minus_y_squared_l1025_102513

theorem x_squared_minus_y_squared (x y : ℝ) (h1 : x + y = 9) (h2 : x - y = 3) : x^2 - y^2 = 27 := 
sorry

end NUMINAMATH_GPT_x_squared_minus_y_squared_l1025_102513


namespace NUMINAMATH_GPT_greatest_common_divisor_84_n_l1025_102577

theorem greatest_common_divisor_84_n :
  ∃ (n : ℕ), (∀ (d : ℕ), d ∣ 84 ∧ d ∣ n → d = 1 ∨ d = 2 ∨ d = 4) ∧ (∀ (x y : ℕ), x ∣ 84 ∧ x ∣ n ∧ y ∣ 84 ∧ y ∣ n → x ≤ y → y = 4) :=
sorry

end NUMINAMATH_GPT_greatest_common_divisor_84_n_l1025_102577


namespace NUMINAMATH_GPT_hyperbola_chord_line_eq_l1025_102514

theorem hyperbola_chord_line_eq (m n s t : ℝ) (h_mn_pos : m > 0 ∧ n > 0 ∧ s > 0 ∧ t > 0)
  (h_mn_sum : m + n = 2)
  (h_m_n_s_t : m / s + n / t = 9)
  (h_s_t_min : s + t = 4 / 9)
  (h_midpoint : (2 : ℝ) = (m + n)) :
  ∃ (c : ℝ), (∀ (x1 y1 x2 y2 : ℝ), 
    (x1 + x2) / 2 = m ∧ (y1 + y2) / 2 = n ∧ 
    (x1 ^ 2 / 4 - y1 ^ 2 / 2 = 1 ∧ x2 ^ 2 / 4 - y2 ^ 2 / 2 = 1) → 
    y2 - y1 = c * (x2 - x1)) ∧ (c = 1 / 2) →
  ∀ (x y : ℝ), x - 2 * y + 1 = 0 :=
by sorry

end NUMINAMATH_GPT_hyperbola_chord_line_eq_l1025_102514


namespace NUMINAMATH_GPT_weightOfEachPacket_l1025_102539

/-- Definition for the number of pounds in one ton --/
def poundsPerTon : ℕ := 2100

/-- Total number of packets filling the 13-ton capacity --/
def numPackets : ℕ := 1680

/-- Capacity of the gunny bag in tons --/
def capacityInTons : ℕ := 13

/-- Total weight of the gunny bag in pounds --/
def totalWeightInPounds : ℕ := capacityInTons * poundsPerTon

/-- Statement that each packet weighs 16.25 pounds --/
theorem weightOfEachPacket : (totalWeightInPounds / numPackets : ℚ) = 16.25 :=
sorry

end NUMINAMATH_GPT_weightOfEachPacket_l1025_102539


namespace NUMINAMATH_GPT_volleyball_team_total_score_l1025_102576

-- Define the conditions
def LizzieScore := 4
def NathalieScore := LizzieScore + 3
def CombinedLizzieNathalieScore := LizzieScore + NathalieScore
def AimeeScore := 2 * CombinedLizzieNathalieScore
def TeammatesScore := 17

-- Prove that the total team score is 50
theorem volleyball_team_total_score :
  LizzieScore + NathalieScore + AimeeScore + TeammatesScore = 50 :=
by
  sorry

end NUMINAMATH_GPT_volleyball_team_total_score_l1025_102576


namespace NUMINAMATH_GPT_milk_production_days_l1025_102567

variable {x : ℕ}

def daily_cow_production (x : ℕ) : ℚ := (x + 4) / ((x + 2) * (x + 3))

def total_daily_production (x : ℕ) : ℚ := (x + 5) * daily_cow_production x

def required_days (x : ℕ) : ℚ := (x + 9) / total_daily_production x

theorem milk_production_days : 
  required_days x = (x + 9) * (x + 2) * (x + 3) / ((x + 5) * (x + 4)) := 
by 
  sorry

end NUMINAMATH_GPT_milk_production_days_l1025_102567


namespace NUMINAMATH_GPT_product_closest_to_l1025_102526

def is_closest_to (n target : ℝ) (options : List ℝ) : Prop :=
  ∀ o ∈ options, |n - target| ≤ |n - o|

theorem product_closest_to : is_closest_to ((2.5) * (50.5 + 0.25)) 127 [120, 125, 127, 130, 140] :=
by
  sorry

end NUMINAMATH_GPT_product_closest_to_l1025_102526


namespace NUMINAMATH_GPT_hats_in_box_total_l1025_102522

theorem hats_in_box_total : 
  (∃ (n : ℕ), (∀ (r b y : ℕ), r + y = n - 2 ∧ r + b = n - 2 ∧ b + y = n - 2)) → (∃ n, n = 3) :=
by
  sorry

end NUMINAMATH_GPT_hats_in_box_total_l1025_102522


namespace NUMINAMATH_GPT_compare_probabilities_l1025_102565

theorem compare_probabilities 
  (M N : ℕ)
  (m n : ℝ)
  (h_m_pos : m > 0)
  (h_n_pos : n > 0)
  (h_m_million : m > 10^6)
  (h_n_million : n ≤ 10^6) :
  (M : ℝ) / (M + n / m * N) > (M : ℝ) / (M + N) :=
by
  sorry

end NUMINAMATH_GPT_compare_probabilities_l1025_102565


namespace NUMINAMATH_GPT_expand_binomials_l1025_102558

theorem expand_binomials (x : ℝ) : 
  (1 + x + x^3) * (1 - x^4) = 1 + x + x^3 - x^4 - x^5 - x^7 :=
by
  sorry

end NUMINAMATH_GPT_expand_binomials_l1025_102558


namespace NUMINAMATH_GPT_product_roots_positive_real_part_l1025_102517

open Complex

theorem product_roots_positive_real_part :
    (∃ (roots : Fin 6 → ℂ),
       (∀ k, roots k ^ 6 = -64) ∧
       (∀ k, (roots k).re > 0 → (roots 0).re > 0 ∧ (roots 0).im > 0 ∧
                               (roots 1).re > 0 ∧ (roots 1).im < 0) ∧
       (roots 0 * roots 1 = 4)
    ) :=
sorry

end NUMINAMATH_GPT_product_roots_positive_real_part_l1025_102517


namespace NUMINAMATH_GPT_attendance_ratio_3_to_1_l1025_102505

variable (x y : ℕ)
variable (total_attendance : ℕ := 2700)
variable (second_day_attendance : ℕ := 300)

/-- 
Prove that the ratio of the number of people attending the third day to the number of people attending the first day is 3:1
-/
theorem attendance_ratio_3_to_1
  (h1 : total_attendance = 2700)
  (h2 : second_day_attendance = x / 2)
  (h3 : second_day_attendance = 300)
  (h4 : y = total_attendance - x - second_day_attendance) :
  y / x = 3 :=
by
  sorry

end NUMINAMATH_GPT_attendance_ratio_3_to_1_l1025_102505


namespace NUMINAMATH_GPT_circle_properties_l1025_102509

def circle_center_line (x y : ℝ) : Prop := x + y - 1 = 0

def point_A_on_circle (x y : ℝ) : Prop := (x, y) = (-1, 4)
def point_B_on_circle (x y : ℝ) : Prop := (x, y) = (1, 2)

def circle_equation (x y : ℝ) : Prop := (x + 1)^2 + (y - 2)^2 = 4

def slope_range_valid (k : ℝ) : Prop :=
  k ≤ 0 ∨ k ≥ 4 / 3

theorem circle_properties
  (x y : ℝ)
  (center_x center_y : ℝ)
  (h_center_line : circle_center_line center_x center_y)
  (h_point_A_on_circle : point_A_on_circle x y)
  (h_point_B_on_circle : point_B_on_circle x y)
  (h_circle_equation : circle_equation x y)
  (k : ℝ) :
  circle_equation center_x center_y ∧ slope_range_valid k :=
sorry

end NUMINAMATH_GPT_circle_properties_l1025_102509


namespace NUMINAMATH_GPT_quadratic_inequality_solution_l1025_102557

theorem quadratic_inequality_solution (a b c : ℝ) 
  (h1 : a < 0) 
  (h2 : a * 2^2 + b * 2 + c = 0) 
  (h3 : a * (-1)^2 + b * (-1) + c = 0) :
  ∀ x, ax^2 + bx + c ≥ 0 ↔ (-1 ≤ x ∧ x ≤ 2) :=
by 
  sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_l1025_102557


namespace NUMINAMATH_GPT_hypotenuse_length_l1025_102512

theorem hypotenuse_length (a b c : ℝ) (h1 : a^2 + b^2 + c^2 = 1450) (h2 : c^2 = a^2 + b^2) : 
  c = Real.sqrt 725 :=
by
  sorry

end NUMINAMATH_GPT_hypotenuse_length_l1025_102512


namespace NUMINAMATH_GPT_donation_to_second_home_l1025_102586

-- Definitions of the conditions
def total_donation := 700.00
def first_home_donation := 245.00
def third_home_donation := 230.00

-- Define the unknown donation to the second home
noncomputable def second_home_donation := total_donation - first_home_donation - third_home_donation

-- The theorem to prove
theorem donation_to_second_home :
  second_home_donation = 225.00 :=
by sorry

end NUMINAMATH_GPT_donation_to_second_home_l1025_102586


namespace NUMINAMATH_GPT_quadrilateral_area_is_22_5_l1025_102572

-- Define the vertices of the quadrilateral
def vertex1 : ℝ × ℝ := (3, -1)
def vertex2 : ℝ × ℝ := (-1, 4)
def vertex3 : ℝ × ℝ := (2, 3)
def vertex4 : ℝ × ℝ := (9, 9)

-- Define the function to calculate the area using the Shoelace Theorem
noncomputable def shoelace_area (v1 v2 v3 v4 : ℝ × ℝ) : ℝ :=
  0.5 * (abs ((v1.1 * v2.2 + v2.1 * v3.2 + v3.1 * v4.2 + v4.1 * v1.2) 
        - (v1.2 * v2.1 + v2.2 * v3.1 + v3.2 * v4.1 + v4.2 * v1.1)))

-- State that the area of the quadrilateral with given vertices is 22.5
theorem quadrilateral_area_is_22_5 :
  shoelace_area vertex1 vertex2 vertex3 vertex4 = 22.5 :=
by 
  -- We skip the proof here.
  sorry

end NUMINAMATH_GPT_quadrilateral_area_is_22_5_l1025_102572


namespace NUMINAMATH_GPT_tylenol_tablet_mg_l1025_102528

/-- James takes 2 Tylenol tablets every 6 hours and consumes 3000 mg a day.
    Prove the mg of each Tylenol tablet. -/
theorem tylenol_tablet_mg (t : ℕ) (h1 : t = 2) (h2 : 24 / 6 = 4) (h3 : 3000 / (4 * t) = 375) : t * (4 * t) = 3000 :=
by
  sorry

end NUMINAMATH_GPT_tylenol_tablet_mg_l1025_102528


namespace NUMINAMATH_GPT_f_of_1_l1025_102579

theorem f_of_1 (f : ℕ+ → ℕ+) (h_mono : ∀ {a b : ℕ+}, a < b → f a < f b)
  (h_fn_prop : ∀ n : ℕ+, f (f n) = 3 * n) : f 1 = 2 :=
sorry

end NUMINAMATH_GPT_f_of_1_l1025_102579


namespace NUMINAMATH_GPT_cost_of_two_pencils_and_one_pen_l1025_102553

variables (a b : ℝ)

theorem cost_of_two_pencils_and_one_pen
  (h1 : 3 * a + b = 3.00)
  (h2 : 3 * a + 4 * b = 7.50) :
  2 * a + b = 2.50 :=
sorry

end NUMINAMATH_GPT_cost_of_two_pencils_and_one_pen_l1025_102553


namespace NUMINAMATH_GPT_starting_number_of_sequence_l1025_102590

theorem starting_number_of_sequence :
  ∃ (start : ℤ), 
    (∀ n, 0 ≤ n ∧ n < 8 → start + n * 11 ≤ 119) ∧ 
    (∃ k, 1 ≤ k ∧ k ≤ 8 ∧ 119 = start + (k - 1) * 11) ↔ start = 33 :=
by
  sorry

end NUMINAMATH_GPT_starting_number_of_sequence_l1025_102590


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_l1025_102543

theorem necessary_but_not_sufficient (x : ℝ) : (x^2 + 2 * x - 8 > 0) ↔ (x > 2) ∨ (x < -4) := by
sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_l1025_102543


namespace NUMINAMATH_GPT_option_c_l1025_102523

theorem option_c (a b : ℝ) (h : a > |b|) : a^2 > b^2 := sorry

end NUMINAMATH_GPT_option_c_l1025_102523


namespace NUMINAMATH_GPT_parabola_ellipse_sum_distances_l1025_102525

noncomputable def sum_distances_intersection_points (b c : ℝ) : ℝ :=
  2 * Real.sqrt b + 2 * Real.sqrt c

theorem parabola_ellipse_sum_distances
  (A B : ℝ)
  (h1 : A > 0) -- semi-major axis condition implied
  (h2 : B > 0) -- semi-minor axis condition implied
  (ellipse_eq : ∀ x y, (x^2) / A^2 + (y^2) / B^2 = 1)
  (focus_shared : ∃ f : ℝ, f = Real.sqrt (A^2 - B^2))
  (directrix_parabola : ∃ d : ℝ, d = B) -- directrix condition
  (intersections : ∃ (b c : ℝ), (b > 0 ∧ c > 0)) -- existence of such intersection points
  : sum_distances_intersection_points b c = 2 * Real.sqrt b + 2 * Real.sqrt c :=
sorry  -- proof omitted

end NUMINAMATH_GPT_parabola_ellipse_sum_distances_l1025_102525


namespace NUMINAMATH_GPT_length_of_inner_rectangle_is_4_l1025_102548

-- Defining the conditions and the final proof statement
theorem length_of_inner_rectangle_is_4 :
  ∃ y : ℝ, y = 4 ∧
  let inner_width := 2
  let second_width := inner_width + 4
  let largest_width := second_width + 4
  let inner_area := inner_width * y
  let second_area := 6 * second_width
  let largest_area := 10 * largest_width
  let first_shaded_area := second_area - inner_area
  let second_shaded_area := largest_area - second_area
  (first_shaded_area - inner_area = second_shaded_area - first_shaded_area)
:= sorry

end NUMINAMATH_GPT_length_of_inner_rectangle_is_4_l1025_102548


namespace NUMINAMATH_GPT_blue_stripe_area_l1025_102591

def cylinder_diameter : ℝ := 20
def cylinder_height : ℝ := 60
def stripe_width : ℝ := 4
def stripe_revolutions : ℕ := 3

theorem blue_stripe_area : 
  let circumference := Real.pi * cylinder_diameter
  let stripe_length := stripe_revolutions * circumference
  let expected_area := stripe_width * stripe_length
  expected_area = 240 * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_blue_stripe_area_l1025_102591


namespace NUMINAMATH_GPT_acres_used_for_corn_l1025_102549

theorem acres_used_for_corn (total_acres : ℕ) (ratio_beans ratio_wheat ratio_corn : ℕ)
    (h_total : total_acres = 1034)
    (h_ratio : ratio_beans = 5 ∧ ratio_wheat = 2 ∧ ratio_corn = 4) : 
    ratio_corn * (total_acres / (ratio_beans + ratio_wheat + ratio_corn)) = 376 := 
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_acres_used_for_corn_l1025_102549


namespace NUMINAMATH_GPT_factorize_expression_polygon_sides_l1025_102560

-- Problem 1: Factorize 2x^3 - 8x
theorem factorize_expression (x : ℝ) : 2 * x^3 - 8 * x = 2 * x * (x - 2) * (x + 2) :=
by
  sorry

-- Problem 2: Find the number of sides of a polygon with interior angle sum 1080 degrees
theorem polygon_sides (n : ℕ) (h : (n - 2) * 180 = 1080) : n = 8 :=
by
  sorry

end NUMINAMATH_GPT_factorize_expression_polygon_sides_l1025_102560


namespace NUMINAMATH_GPT_find_a_of_parabola_l1025_102534

theorem find_a_of_parabola
  (a b c : ℝ)
  (h_point : 2 = c)
  (h_vertex : -2 = a * (2 - 2)^2 + b * 2 + c) :
  a = 1 :=
by
  sorry

end NUMINAMATH_GPT_find_a_of_parabola_l1025_102534


namespace NUMINAMATH_GPT_exists_divisible_by_3_on_circle_l1025_102542

theorem exists_divisible_by_3_on_circle :
  ∃ a : ℕ → ℕ, (∀ i, a i ≥ 1) ∧
               (∀ i, i < 99 → (a (i + 1) < 99 → (a (i + 1) - a i = 1 ∨ a (i + 1) - a i = 2 ∨ a (i + 1) = 2 * a i))) ∧
               (∃ i, i < 99 ∧ a i % 3 = 0) := 
sorry

end NUMINAMATH_GPT_exists_divisible_by_3_on_circle_l1025_102542


namespace NUMINAMATH_GPT_triangle_area_l1025_102547

theorem triangle_area :
  ∀ (k : ℝ), ∃ (area : ℝ), 
  (∃ (r : ℝ) (a b c : ℝ), 
      r = 2 * Real.sqrt 3 ∧
      a / b = 3 / 5 ∧ a / c = 3 / 7 ∧ b / c = 5 / 7 ∧
      (∃ (A B C : ℝ),
          A = 3 * k ∧ B = 5 * k ∧ C = 7 * k ∧
          area = (1/2) * a * b * Real.sin (2 * Real.pi / 3))) →
  area = (135 * Real.sqrt 3 / 49) :=
sorry

end NUMINAMATH_GPT_triangle_area_l1025_102547


namespace NUMINAMATH_GPT_false_proposition_among_given_l1025_102556

theorem false_proposition_among_given (a b c : Prop) : 
  (a = ∀ x : ℝ, ∃ y : ℝ, x = y) ∧
  (b = (∃ a b c : ℕ, a = 3 ∧ b = 4 ∧ c = 5 ∧ a^2 + b^2 = c^2)) ∧
  (c = ∀ α β : ℝ, α = β ∧ ∃ P : Type, ∃ vertices : P, α = β ) → ¬c := by
  sorry

end NUMINAMATH_GPT_false_proposition_among_given_l1025_102556


namespace NUMINAMATH_GPT_minimum_possible_value_l1025_102510

-- Define the set of distinct elements
def distinct_elems : Set ℤ := {-8, -6, -4, -1, 1, 3, 7, 12}

-- Define the existence of distinct elements
def elem_distinct (p q r s t u v w : ℤ) : Prop :=
  p ∈ distinct_elems ∧ q ∈ distinct_elems ∧ r ∈ distinct_elems ∧ s ∈ distinct_elems ∧ 
  t ∈ distinct_elems ∧ u ∈ distinct_elems ∧ v ∈ distinct_elems ∧ w ∈ distinct_elems ∧ 
  p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ p ≠ t ∧ p ≠ u ∧ p ≠ v ∧ p ≠ w ∧ 
  q ≠ r ∧ q ≠ s ∧ q ≠ t ∧ q ≠ u ∧ q ≠ v ∧ q ≠ w ∧
  r ≠ s ∧ r ≠ t ∧ r ≠ u ∧ r ≠ v ∧ r ≠ w ∧ 
  s ≠ t ∧ s ≠ u ∧ s ≠ v ∧ s ≠ w ∧ 
  t ≠ u ∧ t ≠ v ∧ t ≠ w ∧ 
  u ≠ v ∧ u ≠ w ∧ 
  v ≠ w

-- The main proof problem
theorem minimum_possible_value :
  ∀ (p q r s t u v w : ℤ), elem_distinct p q r s t u v w ->
  (p + q + r + s)^2 + (t + u + v + w)^2 = 10 := 
sorry

end NUMINAMATH_GPT_minimum_possible_value_l1025_102510


namespace NUMINAMATH_GPT_log_base_eq_l1025_102527

theorem log_base_eq (x : ℝ) (h₁ : x > 0) (h₂ : x ≠ 1) : 
  (Real.log x / Real.log 4) * (Real.log 8 / Real.log x) = Real.log 8 / Real.log 4 := 
by 
  sorry

end NUMINAMATH_GPT_log_base_eq_l1025_102527


namespace NUMINAMATH_GPT_complement_union_l1025_102568

def A : Set ℝ := {x | x^2 - 1 < 0}
def B : Set ℝ := {x | x > 0}

theorem complement_union (x : ℝ) : (x ∈ Aᶜ ∪ B) ↔ (x ∈ Set.Iic (-1) ∪ Set.Ioi 0) := by
  sorry

end NUMINAMATH_GPT_complement_union_l1025_102568


namespace NUMINAMATH_GPT_find_other_number_l1025_102563

theorem find_other_number (HCF LCM num1 num2 : ℕ) 
    (h_hcf : HCF = 14)
    (h_lcm : LCM = 396)
    (h_num1 : num1 = 36)
    (h_prod : HCF * LCM = num1 * num2)
    : num2 = 154 := by
  sorry

end NUMINAMATH_GPT_find_other_number_l1025_102563


namespace NUMINAMATH_GPT_martha_total_payment_l1025_102559

noncomputable def cheese_kg : ℝ := 1.5
noncomputable def meat_kg : ℝ := 0.55
noncomputable def pasta_kg : ℝ := 0.28
noncomputable def tomatoes_kg : ℝ := 2.2

noncomputable def cheese_price_per_kg : ℝ := 6.30
noncomputable def meat_price_per_kg : ℝ := 8.55
noncomputable def pasta_price_per_kg : ℝ := 2.40
noncomputable def tomatoes_price_per_kg : ℝ := 1.79

noncomputable def total_cost :=
  cheese_kg * cheese_price_per_kg +
  meat_kg * meat_price_per_kg +
  pasta_kg * pasta_price_per_kg +
  tomatoes_kg * tomatoes_price_per_kg

theorem martha_total_payment : total_cost = 18.76 := by
  sorry

end NUMINAMATH_GPT_martha_total_payment_l1025_102559


namespace NUMINAMATH_GPT_rectangle_perimeter_divided_into_six_congruent_l1025_102562

theorem rectangle_perimeter_divided_into_six_congruent (l w : ℕ) (h1 : 2 * (w + l / 6) = 40) (h2 : l = 120 - 6 * w) : 
  2 * (l + w) = 280 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_perimeter_divided_into_six_congruent_l1025_102562


namespace NUMINAMATH_GPT_find_angle_B_l1025_102508

variable (a b c A B C : ℝ)

-- Assuming all the necessary conditions and givens
axiom triangle_condition1 : a * (Real.sin B * Real.cos C) + c * (Real.sin B * Real.cos A) = (1 / 2) * b
axiom triangle_condition2 : a > b

-- We need to prove B = π / 6
theorem find_angle_B : B = π / 6 :=
by
  sorry

end NUMINAMATH_GPT_find_angle_B_l1025_102508


namespace NUMINAMATH_GPT_exists_line_equidistant_from_AB_CD_l1025_102584

noncomputable def Line : Type := sorry  -- This would be replaced with an appropriate definition of a line in space

def Point : Type := sorry  -- Similarly, a point in space type definition

variables (A B C D : Point)

def perpendicularBisector (P Q : Point) : Type := sorry  -- Definition for perpendicular bisector plane of two points

def is_perpendicularBisector_of (e : Line) (P Q : Point) : Prop := sorry  -- e is perpendicular bisector plane of P and Q

theorem exists_line_equidistant_from_AB_CD (A B C D : Point) :
  ∃ e : Line, is_perpendicularBisector_of e A C ∧ is_perpendicularBisector_of e B D :=
by
  sorry

end NUMINAMATH_GPT_exists_line_equidistant_from_AB_CD_l1025_102584


namespace NUMINAMATH_GPT_expr_divisible_by_120_l1025_102504

theorem expr_divisible_by_120 (m : ℕ) : 120 ∣ (m^5 - 5 * m^3 + 4 * m) :=
sorry

end NUMINAMATH_GPT_expr_divisible_by_120_l1025_102504


namespace NUMINAMATH_GPT_gardener_cabbages_this_year_l1025_102550

-- Definitions for the conditions
def side_length_last_year (x : ℕ) := true
def area_last_year (x : ℕ) := x * x
def increase_in_output := 197

-- Proposition to prove the number of cabbages this year
theorem gardener_cabbages_this_year (x : ℕ) (hx : side_length_last_year x) : 
  (area_last_year x + increase_in_output) = 9801 :=
by 
  sorry

end NUMINAMATH_GPT_gardener_cabbages_this_year_l1025_102550


namespace NUMINAMATH_GPT_point_on_parabola_dist_3_from_focus_l1025_102589

def parabola (p : ℝ × ℝ) : Prop := (p.snd)^2 = 4 * p.fst

def focus : ℝ × ℝ := (1, 0)

theorem point_on_parabola_dist_3_from_focus :
  ∃ y: ℝ, ∃ x: ℝ, (parabola (x, y) ∧ (x = 2) ∧ (y = 2 * Real.sqrt 2 ∨ y = -2 * Real.sqrt 2) ∧ (Real.sqrt ((x - focus.fst)^2 + (y - focus.snd)^2) = 3)) :=
by
  sorry

end NUMINAMATH_GPT_point_on_parabola_dist_3_from_focus_l1025_102589


namespace NUMINAMATH_GPT_cloth_sales_worth_l1025_102529

/--
An agent gets a commission of 2.5% on the sales of cloth. If on a certain day, he gets Rs. 15 as commission, 
proves that the worth of the cloth sold through him on that day is Rs. 600.
-/
theorem cloth_sales_worth (commission : ℝ) (rate : ℝ) (total_sales : ℝ) 
  (h_commission : commission = 15) (h_rate : rate = 2.5) (h_commission_formula : commission = (rate / 100) * total_sales) : 
  total_sales = 600 := 
by
  sorry

end NUMINAMATH_GPT_cloth_sales_worth_l1025_102529


namespace NUMINAMATH_GPT_determine_roles_l1025_102540

/-
We have three inhabitants K, M, R.
One of them is a truth-teller (tt), one is a liar (l), 
and one is a trickster (tr).
K states: "I am a trickster."
M states: "That is true."
R states: "I am not a trickster."
A truth-teller always tells the truth.
A liar always lies.
A trickster sometimes lies and sometimes tells the truth.
-/

inductive Role
| truth_teller | liar | trickster

open Role

def inhabitant_role (K M R : Role) : Prop :=
  ((K = liar) ∧ (M = trickster) ∧ (R = truth_teller)) ∧
  (K = trickster → K ≠ K) ∧
  (M = truth_teller → M = truth_teller) ∧
  (R = trickster → R ≠ R)

theorem determine_roles (K M R : Role) : inhabitant_role K M R :=
sorry

end NUMINAMATH_GPT_determine_roles_l1025_102540


namespace NUMINAMATH_GPT_calculate_expression_l1025_102588

variable (a : ℝ)

theorem calculate_expression : 2 * a - 7 * a + 4 * a = -a := by
  sorry

end NUMINAMATH_GPT_calculate_expression_l1025_102588


namespace NUMINAMATH_GPT_simplify_expression_l1025_102554

theorem simplify_expression (x : ℝ) : 4 * x - 3 * x^2 + 6 + (8 - 5 * x + 2 * x^2) = - x^2 - x + 14 := by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1025_102554


namespace NUMINAMATH_GPT_certain_number_l1025_102581

theorem certain_number (x : ℤ) (h : 12 + x = 27) : x = 15 :=
by
  sorry

end NUMINAMATH_GPT_certain_number_l1025_102581


namespace NUMINAMATH_GPT_time_to_complete_together_l1025_102536

theorem time_to_complete_together (sylvia_time carla_time combined_time : ℕ) (h_sylvia : sylvia_time = 45) (h_carla : carla_time = 30) :
  let sylvia_rate := 1 / (sylvia_time : ℚ)
  let carla_rate := 1 / (carla_time : ℚ)
  let combined_rate := sylvia_rate + carla_rate
  let time_to_complete := 1 / combined_rate
  time_to_complete = (combined_time : ℚ) :=
by
  sorry

end NUMINAMATH_GPT_time_to_complete_together_l1025_102536


namespace NUMINAMATH_GPT_shoe_count_l1025_102569

theorem shoe_count 
  (pairs : ℕ)
  (total_shoes : ℕ)
  (prob : ℝ)
  (h_pairs : pairs = 12)
  (h_prob : prob = 0.043478260869565216)
  (h_total_shoes : total_shoes = pairs * 2) :
  total_shoes = 24 :=
by
  sorry

end NUMINAMATH_GPT_shoe_count_l1025_102569


namespace NUMINAMATH_GPT_quadrilateral_count_correct_triangle_count_correct_intersection_triangle_count_correct_l1025_102520

-- 1. Problem: Count of quadrilaterals from 12 points in a semicircle
def semicircle_points : ℕ := 12
def quadrilaterals_from_semicircle_points : ℕ :=
  let points_on_semicircle := 8
  let points_on_diameter := 4
  360 -- This corresponds to the final computed count, skipping calculation details

theorem quadrilateral_count_correct :
  quadrilaterals_from_semicircle_points = 360 := sorry

-- 2. Problem: Count of triangles from 10 points along an angle
def angle_points : ℕ := 10
def triangles_from_angle_points : ℕ :=
  let points_on_one_side := 5
  let points_on_other_side := 4
  90 -- This corresponds to the final computed count, skipping calculation details

theorem triangle_count_correct :
  triangles_from_angle_points = 90 := sorry

-- 3. Problem: Count of triangles from intersection points of parallel lines
def intersection_points : ℕ := 12
def triangles_from_intersections : ℕ :=
  let line_set_1_count := 3
  let line_set_2_count := 4
  200 -- This corresponds to the final computed count, skipping calculation details

theorem intersection_triangle_count_correct :
  triangles_from_intersections = 200 := sorry

end NUMINAMATH_GPT_quadrilateral_count_correct_triangle_count_correct_intersection_triangle_count_correct_l1025_102520


namespace NUMINAMATH_GPT_find_positive_integers_l1025_102573

theorem find_positive_integers
  (a b c : ℕ) 
  (h : a ≥ b ∧ b ≥ c ∧ a ≥ c)
  (ha : a > 0)
  (hb : b > 0)
  (hc : c > 0) :
  (1 + 1 / (a : ℚ)) * (1 + 1 / (b : ℚ)) * (1 + 1 / (c : ℚ)) = 2 →
  (a, b, c) ∈ [(15, 4, 2), (9, 5, 2), (7, 6, 2), (8, 3, 3), (5, 4, 3)] :=
by
  sorry

end NUMINAMATH_GPT_find_positive_integers_l1025_102573


namespace NUMINAMATH_GPT_binary_to_decimal_l1025_102516

theorem binary_to_decimal :
  (1 * 2^0 + 1 * 2^1 + 0 * 2^2 + 1 * 2^3 + 1 * 2^4 + 1 * 2^5 + 1 * 2^6 + 0 * 2^7 + 1 * 2^8) = 379 := 
by
  sorry

end NUMINAMATH_GPT_binary_to_decimal_l1025_102516


namespace NUMINAMATH_GPT_initial_water_amount_l1025_102544

theorem initial_water_amount (W : ℝ) (h1 : ∀ t, t = 50 -> 0.008 * t = 0.4) (h2 : 0.04 * W = 0.4) : W = 10 :=
by
  sorry

end NUMINAMATH_GPT_initial_water_amount_l1025_102544


namespace NUMINAMATH_GPT_sum_of_sequences_is_43_l1025_102541

theorem sum_of_sequences_is_43
  (A B C D : ℕ)
  (hA_pos : 0 < A)
  (hB_pos : 0 < B)
  (hC_pos : 0 < C)
  (hD_pos : 0 < D)
  (h_arith : A + (C - B) = B)
  (h_geom : C = (4 * B) / 3)
  (hD_def : D = (4 * C) / 3) :
  A + B + C + D = 43 :=
sorry

end NUMINAMATH_GPT_sum_of_sequences_is_43_l1025_102541


namespace NUMINAMATH_GPT_solve_furniture_factory_l1025_102587

variable (num_workers : ℕ) (tables_per_worker : ℕ) (legs_per_worker : ℕ) 
variable (tabletop_workers legs_workers : ℕ)

axiom worker_capacity : tables_per_worker = 3 ∧ legs_per_worker = 6
axiom total_workers : num_workers = 60
axiom table_leg_ratio : ∀ (x : ℕ), tabletop_workers = x → legs_workers = (num_workers - x)
axiom daily_production_eq : ∀ (x : ℕ), (4 * tables_per_worker * x = 6 * legs_per_worker * (num_workers - x))

theorem solve_furniture_factory : 
  ∃ (x y : ℕ), num_workers = x + y ∧ 
            4 * 3 * x = 6 * (num_workers - x) ∧ 
            x = 20 ∧ y = (num_workers - 20) := by
  sorry

end NUMINAMATH_GPT_solve_furniture_factory_l1025_102587


namespace NUMINAMATH_GPT_probability_divisible_by_3_l1025_102578

theorem probability_divisible_by_3 :
  ∀ (n : ℤ), (1 ≤ n) ∧ (n ≤ 99) → 3 ∣ (n * (n + 1)) :=
by
  intros n hn
  -- Detailed proof would follow here
  sorry

end NUMINAMATH_GPT_probability_divisible_by_3_l1025_102578


namespace NUMINAMATH_GPT_water_left_in_bathtub_l1025_102545

theorem water_left_in_bathtub :
  (40 * 60 * 9 - 200 * 9 - 12000 = 7800) :=
by
  -- Dripping rate per minute * number of minutes in an hour * number of hours
  let inflow_rate := 40 * 60
  let total_inflow := inflow_rate * 9
  -- Evaporation rate per hour * number of hours
  let total_evaporation := 200 * 9
  -- Water dumped out
  let water_dumped := 12000
  -- Final amount of water
  let final_amount := total_inflow - total_evaporation - water_dumped
  have h : final_amount = 7800 := by
    sorry
  exact h

end NUMINAMATH_GPT_water_left_in_bathtub_l1025_102545


namespace NUMINAMATH_GPT_expression_evaluation_l1025_102524

noncomputable def evaluate_expression : ℝ :=
  (Real.sin (38 * Real.pi / 180) * Real.sin (38 * Real.pi / 180) 
  + Real.cos (38 * Real.pi / 180) * Real.sin (52 * Real.pi / 180) 
  - Real.tan (15 * Real.pi / 180) ^ 2) / (3 * Real.tan (15 * Real.pi / 180))

theorem expression_evaluation : 
  evaluate_expression = (2 * Real.sqrt 3) / 3 :=
by
  sorry

end NUMINAMATH_GPT_expression_evaluation_l1025_102524


namespace NUMINAMATH_GPT_unique_solution_l1025_102575

theorem unique_solution (x : ℝ) : (2:ℝ)^x + (3:ℝ)^x + (6:ℝ)^x = (7:ℝ)^x ↔ x = 2 :=
by
  sorry

end NUMINAMATH_GPT_unique_solution_l1025_102575


namespace NUMINAMATH_GPT_negation_of_universal_l1025_102571

theorem negation_of_universal :
  ¬ (∀ x : ℝ, x^2 > 0) ↔ ∃ x : ℝ, x^2 ≤ 0 :=
by
  sorry

end NUMINAMATH_GPT_negation_of_universal_l1025_102571


namespace NUMINAMATH_GPT_max_neg_expr_l1025_102583

theorem max_neg_expr (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 1) : 
  (- (1 / (2 * a)) - (2 / b)) ≤ - (9 / 2) :=
sorry

end NUMINAMATH_GPT_max_neg_expr_l1025_102583


namespace NUMINAMATH_GPT_fraction_irreducible_l1025_102503

theorem fraction_irreducible (n : ℕ) : Nat.gcd (12 * n + 1) (30 * n + 2) = 1 :=
sorry

end NUMINAMATH_GPT_fraction_irreducible_l1025_102503


namespace NUMINAMATH_GPT_tangency_point_is_ln2_l1025_102521

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x + a * Real.exp (-x)

theorem tangency_point_is_ln2 (a : ℝ) :
  (∀ x : ℝ, f a x = f a (-x)) →
  (∃ m : ℝ, Real.exp m - Real.exp (-m) = 3 / 2) →
  m = Real.log 2 :=
by
  intro h1 h2
  sorry

end NUMINAMATH_GPT_tangency_point_is_ln2_l1025_102521


namespace NUMINAMATH_GPT_decreasing_function_inequality_l1025_102537

theorem decreasing_function_inequality (f : ℝ → ℝ) (a : ℝ)
  (h_decreasing : ∀ x1 x2 : ℝ, x1 < x2 → f x1 > f x2)
  (h_condition : f (3 * a) < f (-2 * a + 10)) :
  a > 2 :=
sorry

end NUMINAMATH_GPT_decreasing_function_inequality_l1025_102537


namespace NUMINAMATH_GPT_sum_of_digits_of_largest_five_digit_number_with_product_120_l1025_102551

theorem sum_of_digits_of_largest_five_digit_number_with_product_120 
  (a b c d e : ℕ)
  (h_digit_a : 0 ≤ a ∧ a ≤ 9)
  (h_digit_b : 0 ≤ b ∧ b ≤ 9)
  (h_digit_c : 0 ≤ c ∧ c ≤ 9)
  (h_digit_d : 0 ≤ d ∧ d ≤ 9)
  (h_digit_e : 0 ≤ e ∧ e ≤ 9)
  (h_product : a * b * c * d * e = 120)
  (h_largest : ∀ f g h i j : ℕ, 
                0 ≤ f ∧ f ≤ 9 → 
                0 ≤ g ∧ g ≤ 9 → 
                0 ≤ h ∧ h ≤ 9 → 
                0 ≤ i ∧ i ≤ 9 → 
                0 ≤ j ∧ j ≤ 9 → 
                f * g * h * i * j = 120 → 
                f * 10000 + g * 1000 + h * 100 + i * 10 + j ≤ a * 10000 + b * 1000 + c * 100 + d * 10 + e) :
  a + b + c + d + e = 18 :=
by sorry

end NUMINAMATH_GPT_sum_of_digits_of_largest_five_digit_number_with_product_120_l1025_102551


namespace NUMINAMATH_GPT_johns_age_fraction_l1025_102532

theorem johns_age_fraction (F M J : ℕ) 
  (hF : F = 40) 
  (hFM : F = M + 4) 
  (hJM : J = M - 16) : 
  J / F = 1 / 2 := 
by
  -- We don't need to fill in the proof, adding sorry to skip it
  sorry

end NUMINAMATH_GPT_johns_age_fraction_l1025_102532


namespace NUMINAMATH_GPT_value_of_nested_radical_l1025_102530

def nested_radical : ℝ := 
  sorry -- Definition of the recurring expression is needed here, let's call it x
  
theorem value_of_nested_radical :
  (nested_radical = 5) :=
sorry -- The actual proof steps will be written here.

end NUMINAMATH_GPT_value_of_nested_radical_l1025_102530


namespace NUMINAMATH_GPT_total_stamps_collected_l1025_102518

-- Conditions
def harry_stamps : ℕ := 180
def sister_stamps : ℕ := 60
def harry_three_times_sister : harry_stamps = 3 * sister_stamps := 
  by
  sorry  -- Proof will show that 180 = 3 * 60 (provided for completeness)

-- Statement to prove
theorem total_stamps_collected : harry_stamps + sister_stamps = 240 :=
  by
  sorry

end NUMINAMATH_GPT_total_stamps_collected_l1025_102518


namespace NUMINAMATH_GPT_tan_value_sin_cos_ratio_sin_squared_expression_l1025_102552

theorem tan_value (α : ℝ) (h1 : 3 * Real.pi / 4 < α) (h2 : α < Real.pi) (h3 : Real.tan α + 1 / Real.tan α = -10 / 3) : 
  Real.tan α = -1 / 3 :=
sorry

theorem sin_cos_ratio (α : ℝ) (h1 : 3 * Real.pi / 4 < α) (h2 : α < Real.pi) (h3 : Real.tan α + 1 / Real.tan α = -10 / 3) (h4 : Real.tan α = -1 / 3) : 
  (Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) = -1 / 2 :=
sorry

theorem sin_squared_expression (α : ℝ) (h1 : 3 * Real.pi / 4 < α) (h2 : α < Real.pi) (h3 : Real.tan α + 1 / Real.tan α = -10 / 3) (h4 : Real.tan α = -1 / 3) : 
  2 * Real.sin α ^ 2 - Real.sin α * Real.cos α - 3 * Real.cos α ^ 2 = -11 / 5 :=
sorry

end NUMINAMATH_GPT_tan_value_sin_cos_ratio_sin_squared_expression_l1025_102552


namespace NUMINAMATH_GPT_percentage_less_than_l1025_102582

theorem percentage_less_than (x y : ℝ) (h1 : y = x * 1.8181818181818181) : (∃ P : ℝ, P = 45) :=
by
  sorry

end NUMINAMATH_GPT_percentage_less_than_l1025_102582


namespace NUMINAMATH_GPT_total_number_of_rulers_l1025_102546

-- Given conditions
def initial_rulers : ℕ := 11
def rulers_added_by_tim : ℕ := 14

-- Given question and desired outcome
def total_rulers (initial_rulers rulers_added_by_tim : ℕ) : ℕ :=
  initial_rulers + rulers_added_by_tim

-- The proof problem statement
theorem total_number_of_rulers : total_rulers 11 14 = 25 := by
  sorry

end NUMINAMATH_GPT_total_number_of_rulers_l1025_102546


namespace NUMINAMATH_GPT_find_incorrect_statement_l1025_102555

def is_opposite (a b : ℝ) := a = -b

theorem find_incorrect_statement :
  ¬∀ (a b : ℝ), (a * b < 0) → is_opposite a b := sorry

end NUMINAMATH_GPT_find_incorrect_statement_l1025_102555


namespace NUMINAMATH_GPT_problem_solution_l1025_102566

theorem problem_solution :
  ∃ x y z : ℕ,
    0 < x ∧ 0 < y ∧ 0 < z ∧
    x^2 + y^2 + z^2 = 2 * (y * z + 1) ∧
    x + y + z = 4032 ∧
    x^2 * y + z = 4031 :=
by
  sorry

end NUMINAMATH_GPT_problem_solution_l1025_102566


namespace NUMINAMATH_GPT_arctan_sum_eq_pi_over_4_l1025_102507

theorem arctan_sum_eq_pi_over_4 : 
  Real.arctan (1/3) + Real.arctan (1/4) + Real.arctan (1/5) + Real.arctan (1/47) = Real.pi / 4 :=
by
  sorry

end NUMINAMATH_GPT_arctan_sum_eq_pi_over_4_l1025_102507


namespace NUMINAMATH_GPT_no_nat_nums_gt_one_divisibility_conditions_l1025_102561

theorem no_nat_nums_gt_one_divisibility_conditions :
  ¬ ∃ (a b c : ℕ), 
    1 < a ∧ 1 < b ∧ 1 < c ∧
    (c ∣ a^2 - 1) ∧ (b ∣ a^2 - 1) ∧ 
    (a ∣ c^2 - 1) ∧ (b ∣ c^2 - 1) :=
by 
  sorry

end NUMINAMATH_GPT_no_nat_nums_gt_one_divisibility_conditions_l1025_102561


namespace NUMINAMATH_GPT_g_of_f_roots_reciprocal_l1025_102593

noncomputable def f (a b c x : ℝ) : ℝ := x^3 + 2 * a * x^2 + 3 * b * x + 4 * c

theorem g_of_f_roots_reciprocal
  (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : a + b + c = 1) :
  ∃ g : ℝ → ℝ, g 1 = (4 - a) / (4 * c) :=
sorry

end NUMINAMATH_GPT_g_of_f_roots_reciprocal_l1025_102593


namespace NUMINAMATH_GPT_a2_range_l1025_102596

open Nat

noncomputable def a_seq (a : ℕ → ℝ) := ∀ (n : ℕ), n > 0 → (n + 1) * a n ≥ n * a (2 * n)

theorem a2_range (a : ℕ → ℝ) 
  (h1 : ∀ (n : ℕ), n > 0 → (n + 1) * a n ≥ n * a (2 * n)) 
  (h2 : ∀ (m n : ℕ), m < n → a m ≤ a n) 
  (h3 : a 1 = 2) :
  (2 < a 2) ∧ (a 2 ≤ 4) :=
sorry

end NUMINAMATH_GPT_a2_range_l1025_102596


namespace NUMINAMATH_GPT_Toby_change_l1025_102585

def change (orders_cost per_person total_cost given_amount : ℝ) : ℝ :=
  given_amount - per_person

def total_cost (cheeseburgers milkshake coke fries cookies tax : ℝ) : ℝ :=
  cheeseburgers + milkshake + coke + fries + cookies + tax

theorem Toby_change :
  let cheeseburger_cost := 3.65
  let milkshake_cost := 2.0
  let coke_cost := 1.0
  let fries_cost := 4.0
  let cookie_cost := 3 * 0.5 -- Total cost for three cookies
  let tax := 0.2
  let total := total_cost (2 * cheeseburger_cost) milkshake_cost coke_cost fries_cost cookie_cost tax
  let per_person := total / 2
  let toby_arrival := 15.0
  change total per_person total toby_arrival = 7 :=
by
  sorry

end NUMINAMATH_GPT_Toby_change_l1025_102585


namespace NUMINAMATH_GPT_num_possible_values_for_n_l1025_102598

open Real

noncomputable def count_possible_values_for_n : ℕ :=
  let log2 := log 2
  let log2_9 := log 9 / log2
  let log2_50 := log 50 / log2
  let range_n := ((6 : ℕ), 450)
  let count := range_n.2 - range_n.1 + 1
  count

theorem num_possible_values_for_n :
  count_possible_values_for_n = 445 :=
by
  sorry

end NUMINAMATH_GPT_num_possible_values_for_n_l1025_102598


namespace NUMINAMATH_GPT_possible_values_of_r_l1025_102594

noncomputable def r : ℝ := sorry

def is_four_place_decimal (x : ℝ) : Prop := 
  ∃ (a b c d : ℕ), a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧ x = a / 10 + b / 100 + c / 1000 + d / 10000

def is_closest_fraction (x : ℝ) : Prop := 
  abs (x - 3 / 11) < abs (x - 3 / 10) ∧ abs (x - 3 / 11) < abs (x - 1 / 4)

theorem possible_values_of_r :
  (0.2614 <= r ∧ r <= 0.2864) ∧ is_four_place_decimal r ∧ is_closest_fraction r →
  ∃ n : ℕ, n = 251 := 
sorry

end NUMINAMATH_GPT_possible_values_of_r_l1025_102594


namespace NUMINAMATH_GPT_find_second_term_l1025_102511

-- Define the terms and common ratio in the geometric sequence
def geometric_sequence (a r : ℚ) (n : ℕ) : ℚ := a * r^(n-1)

-- Given the fifth and sixth terms
variables (a r : ℚ)
axiom fifth_term : geometric_sequence a r 5 = 48
axiom sixth_term : geometric_sequence a r 6 = 72

-- Prove that the second term is 128/9
theorem find_second_term : geometric_sequence a r 2 = 128 / 9 :=
sorry

end NUMINAMATH_GPT_find_second_term_l1025_102511


namespace NUMINAMATH_GPT_range_of_lg_x_l1025_102501

variable {f : ℝ → ℝ}

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def is_decreasing_on_nonneg (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 ≤ x → x ≤ y → f y ≤ f x

theorem range_of_lg_x {f : ℝ → ℝ} (h_even : is_even f)
    (h_decreasing : is_decreasing_on_nonneg f)
    (h_condition : f (Real.log x) > f 1) :
    x ∈ Set.Ioo (1/10 : ℝ) (10 : ℝ) :=
  sorry

end NUMINAMATH_GPT_range_of_lg_x_l1025_102501


namespace NUMINAMATH_GPT_no_valid_number_l1025_102574

theorem no_valid_number (x : ℕ) (h1 : 0 ≤ x) (h2 : x ≤ 9) : ¬ ∃ (y : ℕ), (x * 100 + 3 * 10 + y) % 11 = 0 :=
by
  sorry

end NUMINAMATH_GPT_no_valid_number_l1025_102574


namespace NUMINAMATH_GPT_parabola_focus_equals_ellipse_focus_l1025_102592

theorem parabola_focus_equals_ellipse_focus (p : ℝ) : 
  let parabola_focus := (p / 2, 0)
  let ellipse_focus := (2, 0)
  parabola_focus = ellipse_focus → p = 4 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_parabola_focus_equals_ellipse_focus_l1025_102592


namespace NUMINAMATH_GPT_subscription_total_eq_14036_l1025_102580

noncomputable def total_subscription (x : ℕ) : ℕ :=
  3 * x + 14000

theorem subscription_total_eq_14036 (c : ℕ) (profit_b : ℕ) (total_profit : ℕ) 
  (h1 : profit_b = 10200)
  (h2 : total_profit = 30000) 
  (h3 : (profit_b : ℝ) / (total_profit : ℝ) = (c + 5000 : ℝ) / (total_subscription c : ℝ)) :
  total_subscription c = 14036 :=
by
  sorry

end NUMINAMATH_GPT_subscription_total_eq_14036_l1025_102580


namespace NUMINAMATH_GPT_tessellation_solutions_l1025_102538

theorem tessellation_solutions (m n : ℕ) (h : 60 * m + 90 * n = 360) : m = 3 ∧ n = 2 :=
by
  sorry

end NUMINAMATH_GPT_tessellation_solutions_l1025_102538


namespace NUMINAMATH_GPT_jay_savings_in_a_month_is_correct_l1025_102535

-- Definitions for the conditions
def initial_savings : ℕ := 20
def weekly_increase : ℕ := 10

-- Define the savings for each week
def savings_after_week (week : ℕ) : ℕ :=
  initial_savings + (week - 1) * weekly_increase

-- Define the total savings over 4 weeks
def total_savings_after_4_weeks : ℕ :=
  savings_after_week 1 + savings_after_week 2 + savings_after_week 3 + savings_after_week 4

-- Proposition statement 
theorem jay_savings_in_a_month_is_correct :
  total_savings_after_4_weeks = 140 :=
  by
  -- proof will go here
  sorry

end NUMINAMATH_GPT_jay_savings_in_a_month_is_correct_l1025_102535


namespace NUMINAMATH_GPT_polygon_sides_l1025_102564

def sum_interior_angles (n : ℕ) : ℝ := (n - 2) * 180
def sum_exterior_angles : ℝ := 360

theorem polygon_sides (n : ℕ) (h : 1/4 * sum_interior_angles n - sum_exterior_angles = 90) : n = 12 := 
by
  -- sorry to skip the proof
  sorry

end NUMINAMATH_GPT_polygon_sides_l1025_102564


namespace NUMINAMATH_GPT_polynomial_factorization_l1025_102531

variable (a b c : ℝ)

theorem polynomial_factorization :
  2 * a * (b - c)^3 + 3 * b * (c - a)^3 + 2 * c * (a - b)^3 =
  (a - b) * (b - c) * (c - a) * (5 * b - c) :=
by sorry

end NUMINAMATH_GPT_polynomial_factorization_l1025_102531


namespace NUMINAMATH_GPT_log_expression_identity_l1025_102570

theorem log_expression_identity :
  (Real.log 5 / Real.log 10)^2 + (Real.log 2 / Real.log 10) * (Real.log 50 / Real.log 10) = 1 :=
by
  sorry

end NUMINAMATH_GPT_log_expression_identity_l1025_102570


namespace NUMINAMATH_GPT_log_expression_value_l1025_102595

theorem log_expression_value :
  (Real.log 3 / Real.log 4 + Real.log 3 / Real.log 8) * 
  (Real.log 2 / Real.log 3 + Real.log 8 / Real.log 9) = 25 / 12 :=
by
  sorry

end NUMINAMATH_GPT_log_expression_value_l1025_102595


namespace NUMINAMATH_GPT_majority_owner_percentage_l1025_102506

theorem majority_owner_percentage (profit total_profit : ℝ)
    (majority_owner_share : ℝ) (partner_share : ℝ) 
    (combined_share : ℝ) 
    (num_partners : ℕ) 
    (total_profit_value : total_profit = 80000) 
    (partner_share_value : partner_share = 0.25 * (1 - majority_owner_share)) 
    (combined_share_value : combined_share = profit)
    (combined_share_amount : combined_share = 50000) 
    (num_partners_value : num_partners = 4) :
  majority_owner_share = 0.25 :=
by
  sorry

end NUMINAMATH_GPT_majority_owner_percentage_l1025_102506


namespace NUMINAMATH_GPT_triangle_side_count_l1025_102599

theorem triangle_side_count (x : ℤ) (h1 : 3 < x) (h2 : x < 13) : ∃ n, n = 9 := by
  sorry

end NUMINAMATH_GPT_triangle_side_count_l1025_102599


namespace NUMINAMATH_GPT_solve_x_in_equation_l1025_102515

theorem solve_x_in_equation (a b x : ℝ) (h1 : a ≠ b) (h2 : a ≠ -b) (h3 : x ≠ 0) : 
  (b ≠ 0 ∧ (1 / (a + b) + (a - b) / x = 1 / (a - b) + (a - b) / x) → x = a^2 - b^2) ∧ 
  (b = 0 ∧ a ≠ 0 ∧ (1 / a + a / x = 1 / a + a / x) → x ≠ 0) := 
by
  sorry

end NUMINAMATH_GPT_solve_x_in_equation_l1025_102515


namespace NUMINAMATH_GPT_sequence_general_formula_and_max_n_l1025_102519

theorem sequence_general_formula_and_max_n {a : ℕ → ℝ} {S : ℕ → ℝ} {T : ℕ → ℝ}
  (hS2 : S 2 = (3 / 2) * a 2 - 1) 
  (hS3 : S 3 = (3 / 2) * a 3 - 1) :
  (∀ n, a n = 2 * 3^(n - 1)) ∧ 
  (∃ n : ℕ, (8 / 5) * T n + n / (5 * 3 ^ (n - 1)) ≤ 40 / 27 ∧ ∀ k > n, 
    (8 / 5) * T k + k / (5 * 3 ^ (k - 1)) > 40 / 27) :=
by
  sorry

end NUMINAMATH_GPT_sequence_general_formula_and_max_n_l1025_102519


namespace NUMINAMATH_GPT_number_of_sacks_after_49_days_l1025_102533

def sacks_per_day : ℕ := 38
def days_of_harvest : ℕ := 49
def total_sacks_after_49_days : ℕ := 1862

theorem number_of_sacks_after_49_days :
  sacks_per_day * days_of_harvest = total_sacks_after_49_days :=
by
  sorry

end NUMINAMATH_GPT_number_of_sacks_after_49_days_l1025_102533


namespace NUMINAMATH_GPT_necessarily_negative_b_ab_l1025_102500

theorem necessarily_negative_b_ab (a b : ℝ) (h1 : 0 < a) (h2 : a < 2) (h3 : -2 < b) (h4 : b < 0) : 
  b + a * b < 0 := by 
  sorry

end NUMINAMATH_GPT_necessarily_negative_b_ab_l1025_102500
