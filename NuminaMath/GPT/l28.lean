import Mathlib

namespace option_D_is_correct_option_A_is_incorrect_option_B_is_incorrect_option_C_is_incorrect_l28_2809

variable (a b x : ℝ)

theorem option_D_is_correct :
  (2 * x + 1) * (x - 2) = 2 * x^2 - 3 * x - 2 :=
by sorry

theorem option_A_is_incorrect :
  2 * a^2 * b * 3 * a^2 * b^2 ≠ 6 * a^6 * b^3 :=
by sorry

theorem option_B_is_incorrect :
  0.00076 ≠ 7.6 * 10^4 :=
by sorry

theorem option_C_is_incorrect :
  -2 * a * (a + b) ≠ -2 * a^2 + 2 * a * b :=
by sorry

end option_D_is_correct_option_A_is_incorrect_option_B_is_incorrect_option_C_is_incorrect_l28_2809


namespace total_tiles_is_1352_l28_2847

noncomputable def side_length_of_floor := 39

noncomputable def total_tiles_covering_floor (n : ℕ) : ℕ :=
  (n ^ 2) - ((n / 3) ^ 2)

theorem total_tiles_is_1352 :
  total_tiles_covering_floor side_length_of_floor = 1352 := by
  sorry

end total_tiles_is_1352_l28_2847


namespace old_camera_model_cost_l28_2893

theorem old_camera_model_cost (C new_model_cost discounted_lens_cost : ℝ)
  (h1 : new_model_cost = 1.30 * C)
  (h2 : discounted_lens_cost = 200)
  (h3 : new_model_cost + discounted_lens_cost = 5400)
  : C = 4000 := by
sorry

end old_camera_model_cost_l28_2893


namespace find_c_l28_2879

-- Define the problem
def parabola (x y : ℝ) (a : ℝ) : Prop := 
  x = a * (y - 3) ^ 2 + 5

def point (x y : ℝ) (a : ℝ) : Prop := 
  7 = a * (6 - 3) ^ 2 + 5

-- Theorem to be proved
theorem find_c (a : ℝ) (c : ℝ) (h1 : parabola 7 6 a) (h2 : point 7 6 a) : c = 7 :=
by
  sorry

end find_c_l28_2879


namespace speed_of_goods_train_l28_2836

open Real

theorem speed_of_goods_train
  (V_girl : ℝ := 100) -- The speed of the girl's train in km/h
  (t : ℝ := 6/3600)  -- The passing time in hours
  (L : ℝ := 560/1000) -- The length of the goods train in km
  (V_g : ℝ) -- The speed of the goods train in km/h
  : V_g = 236 := sorry

end speed_of_goods_train_l28_2836


namespace sticks_form_equilateral_triangle_l28_2880

theorem sticks_form_equilateral_triangle (n : ℕ) (h1 : n ≥ 5) (h2 : n % 6 = 0 ∨ n % 6 = 2 ∨ n % 6 = 3 ∨ n % 6 = 5) :
  ∃ k, k * 3 = (n * (n + 1)) / 2 :=
by
  sorry

end sticks_form_equilateral_triangle_l28_2880


namespace batsman_average_after_17th_inning_l28_2823

theorem batsman_average_after_17th_inning
  (A : ℝ)
  (h1 : A + 10 = (16 * A + 200) / 17)
  : (A = 30 ∧ (A + 10) = 40) :=
by
  sorry

end batsman_average_after_17th_inning_l28_2823


namespace total_fruit_count_l28_2854

theorem total_fruit_count :
  let gerald_apple_bags := 5
  let gerald_orange_bags := 4
  let apples_per_gerald_bag := 30
  let oranges_per_gerald_bag := 25
  let pam_apple_bags := 6
  let pam_orange_bags := 4
  let sue_apple_bags := 2 * gerald_apple_bags
  let sue_orange_bags := gerald_orange_bags / 2
  let apples_per_sue_bag := apples_per_gerald_bag - 10
  let oranges_per_sue_bag := oranges_per_gerald_bag + 5
  
  let gerald_apples := gerald_apple_bags * apples_per_gerald_bag
  let gerald_oranges := gerald_orange_bags * oranges_per_gerald_bag
  
  let pam_apples := pam_apple_bags * (3 * apples_per_gerald_bag)
  let pam_oranges := pam_orange_bags * (2 * oranges_per_gerald_bag)
  
  let sue_apples := sue_apple_bags * apples_per_sue_bag
  let sue_oranges := sue_orange_bags * oranges_per_sue_bag

  let total_apples := gerald_apples + pam_apples + sue_apples
  let total_oranges := gerald_oranges + pam_oranges + sue_oranges
  total_apples + total_oranges = 1250 :=

by
  sorry

end total_fruit_count_l28_2854


namespace ellipse_line_intersection_l28_2839

theorem ellipse_line_intersection (m : ℝ) : 
  (m > 0 ∧ m ≠ 3) →
  (∃ x y : ℝ, (x^2 / 3 + y^2 / m = 1) ∧ (x + 2 * y - 2 = 0)) ↔ 
  ((1 / 4 < m ∧ m < 3) ∨ (m > 3)) := 
by 
  sorry

end ellipse_line_intersection_l28_2839


namespace ceil_neg_seven_fourths_cubed_eq_neg_five_l28_2828

noncomputable def ceil_of_neg_seven_fourths_cubed : ℤ :=
  Int.ceil ((-7 / 4 : ℚ)^3)

theorem ceil_neg_seven_fourths_cubed_eq_neg_five :
  ceil_of_neg_seven_fourths_cubed = -5 := by
  sorry

end ceil_neg_seven_fourths_cubed_eq_neg_five_l28_2828


namespace sequence_general_term_l28_2898

theorem sequence_general_term (a : ℕ → ℕ) (h1 : a 1 = 1) (h2 : ∀ n, a (n + 1) = 3 * a n - 2 * n ^ 2 + 4 * n + 4) :
  ∀ n, a n = 3^n + n^2 - n - 2 :=
sorry

end sequence_general_term_l28_2898


namespace expression_of_f_l28_2804

theorem expression_of_f (f : ℤ → ℤ) (h : ∀ x, f (x - 1) = x^2 + 4 * x - 5) : ∀ x, f x = x^2 + 6 * x :=
by
  sorry

end expression_of_f_l28_2804


namespace volume_of_rectangular_prism_l28_2822

    theorem volume_of_rectangular_prism (height base_perimeter: ℝ) (h: height = 5) (b: base_perimeter = 16) :
      ∃ volume, volume = 80 := 
    by
      -- Mathematically equivalent proof goes here
      sorry
    
end volume_of_rectangular_prism_l28_2822


namespace john_total_money_l28_2888

-- Variables representing the prices and quantities.
def chip_price : ℝ := 2
def corn_chip_price : ℝ := 1.5
def chips_quantity : ℕ := 15
def corn_chips_quantity : ℕ := 10

-- Hypothesis representing the total money John has.
theorem john_total_money : 
    (chips_quantity * chip_price + corn_chips_quantity * corn_chip_price) = 45 := by
  sorry

end john_total_money_l28_2888


namespace length_of_segment_from_vertex_to_center_of_regular_hexagon_is_16_l28_2812

def hexagon_vertex_to_center_length (a : ℝ) (h : a = 16) (regular_hexagon : Prop) : Prop :=
∃ (O A : ℝ), (a = 16) → (regular_hexagon = true) → (O = 0) ∧ (A = a) ∧ (dist O A = 16)

theorem length_of_segment_from_vertex_to_center_of_regular_hexagon_is_16 :
  hexagon_vertex_to_center_length 16 (by rfl) true :=
sorry

end length_of_segment_from_vertex_to_center_of_regular_hexagon_is_16_l28_2812


namespace equation_II_consecutive_integers_l28_2843

theorem equation_II_consecutive_integers :
  ∃ x y z w : ℕ, x + y + z + w = 46 ∧ [x, x+1, x+2, x+3] = [x, y, z, w] :=
by
  sorry

end equation_II_consecutive_integers_l28_2843


namespace bus_students_after_fifth_stop_l28_2857

theorem bus_students_after_fifth_stop :
  let initial := 72
  let firstStop := (2 / 3 : ℚ) * initial
  let secondStop := (2 / 3 : ℚ) * firstStop
  let thirdStop := (2 / 3 : ℚ) * secondStop
  let fourthStop := (2 / 3 : ℚ) * thirdStop
  let fifthStop := fourthStop + 12
  fifthStop = 236 / 9 :=
by
  sorry

end bus_students_after_fifth_stop_l28_2857


namespace proof_part_1_proof_part_2_l28_2802

variable {α : ℝ}

/-- Given tan(α) = 3, prove
  (1) (3 * sin(α) + 2 * cos(α))/(sin(α) - 4 * cos(α)) = -11 -/
theorem proof_part_1
  (h : Real.tan α = 3) :
  (3 * Real.sin α + 2 * Real.cos α) / (Real.sin α - 4 * Real.cos α) = -11 := 
by
  sorry

/-- Given tan(α) = 3, prove
  (2) (5 * cos^2(α) - 3 * sin^2(α))/(1 + sin^2(α)) = -11/5 -/
theorem proof_part_2
  (h : Real.tan α = 3) :
  (5 * (Real.cos α)^2 - 3 * (Real.sin α)^2) / (1 + (Real.sin α)^2) = -11 / 5 :=
by
  sorry

end proof_part_1_proof_part_2_l28_2802


namespace initial_number_of_fruits_l28_2891

theorem initial_number_of_fruits (oranges apples limes : ℕ) (h_oranges : oranges = 50)
  (h_apples : apples = 72) (h_oranges_limes : oranges = 2 * limes) (h_apples_limes : apples = 3 * limes) :
  (oranges + apples + limes) * 2 = 288 :=
by
  sorry

end initial_number_of_fruits_l28_2891


namespace circle_inside_triangle_l28_2897

-- Define the problem conditions
def triangle_sides : ℕ × ℕ × ℕ := (3, 4, 5)
def circle_area : ℚ := 25 / 8

-- Define the problem statement
theorem circle_inside_triangle (a b c : ℕ) (area : ℚ)
    (h1 : (a, b, c) = triangle_sides)
    (h2 : area = circle_area) :
    ∃ r R : ℚ, R < r ∧ 2 * r = a + b - c ∧ R^2 = area / π := sorry

end circle_inside_triangle_l28_2897


namespace solve_for_x_l28_2874

theorem solve_for_x :
  ∃ x : ℝ, 5 * (x - 9) = 3 * (3 - 3 * x) + 9 ∧ x = 4.5 :=
by
  use 4.5
  sorry

end solve_for_x_l28_2874


namespace inequality_solution_l28_2892

theorem inequality_solution (a : ℝ) :
  (∀ x : ℝ, a * x^2 + 4 * x + a ≥ -2 * x^2 + 1) → a ≥ 2 :=
by
  sorry

end inequality_solution_l28_2892


namespace eval_g_l28_2831

def g (x : ℝ) : ℝ := 3 * x^3 - 2 * x^2 + x + 1

theorem eval_g : 3 * g 2 + 2 * g (-2) = -9 := 
by {
  sorry
}

end eval_g_l28_2831


namespace similar_triangles_side_length_l28_2827

theorem similar_triangles_side_length (A1 A2 : ℕ) (k : ℕ)
  (h1 : A1 - A2 = 32)
  (h2 : A1 = k^2 * A2)
  (h3 : A2 > 0)
  (side2 : ℕ) (h4 : side2 = 5) :
  ∃ side1 : ℕ, side1 = 3 * side2 ∧ side1 = 15 :=
by
  sorry

end similar_triangles_side_length_l28_2827


namespace career_preference_degrees_l28_2819

variable (M F : ℕ)
variable (h1 : M / F = 2 / 3)
variable (preferred_males : ℚ := M / 4)
variable (preferred_females : ℚ := F / 2)
variable (total_students : ℚ := M + F)
variable (preferred_career_students : ℚ := preferred_males + preferred_females)
variable (career_fraction : ℚ := preferred_career_students / total_students)
variable (degrees : ℚ := 360 * career_fraction)

theorem career_preference_degrees :
  degrees = 144 :=
sorry

end career_preference_degrees_l28_2819


namespace AB_ratio_CD_l28_2881

variable (AB CD : ℝ)
variable (h : ℝ)
variable (O : Point)
variable (ABCD_isosceles : IsIsoscelesTrapezoid AB CD)
variable (areas_condition : List ℝ) 
-- where the list areas_condition represents: [S_OCD, S_OBC, S_OAB, S_ODA]

theorem AB_ratio_CD : 
  ABCD_isosceles ∧ areas_condition = [2, 3, 4, 5] → AB = 2 * CD :=
by
  sorry

end AB_ratio_CD_l28_2881


namespace analysis_error_l28_2846

theorem analysis_error (x : ℝ) (h1 : x + 1 / x ≥ 2) : 
  x + 1 / x ≥ 2 :=
by {
  sorry
}

end analysis_error_l28_2846


namespace inclination_angle_between_given_planes_l28_2830

noncomputable def Point (α : Type*) := α × α × α 

structure Plane (α : Type*) :=
(point : Point α)
(normal_vector : Point α)

def inclination_angle_between_planes (α : Type*) [Field α] (P1 P2 : Plane α) : α := 
  sorry

theorem inclination_angle_between_given_planes 
  (α : Type*) [Field α] 
  (A : Point α) 
  (n1 n2 : Point α) 
  (P1 : Plane α := Plane.mk A n1) 
  (P2 : Plane α := Plane.mk (1,0,0) n2) : 
  inclination_angle_between_planes α P1 P2 = sorry :=
sorry

end inclination_angle_between_given_planes_l28_2830


namespace equation_solutions_l28_2858

noncomputable def solve_equation (x : ℝ) : Prop :=
  x - 3 = 4 * (x - 3)^2

theorem equation_solutions :
  ∀ x : ℝ, solve_equation x ↔ x = 3 ∨ x = 3.25 :=
by sorry

end equation_solutions_l28_2858


namespace correct_conclusions_l28_2842

-- Given function f with the specified domain and properties
variable {f : ℝ → ℝ}

-- Given conditions
axiom functional_eq (x y : ℝ) : f (x + y) + f (x - y) = 2 * f x * f y
axiom f_one_half : f (1/2) = 0
axiom f_zero_not_zero : f 0 ≠ 0

-- Proving our conclusions
theorem correct_conclusions :
  f 0 = 1 ∧ (∀ y : ℝ, f (1/2 + y) = -f (1/2 - y))
:=
by
  sorry

end correct_conclusions_l28_2842


namespace circumference_of_cone_base_l28_2833

theorem circumference_of_cone_base (V : ℝ) (h : ℝ) (C : ℝ) (π := Real.pi) 
  (volume_eq : V = 24 * π) (height_eq : h = 6) 
  (circumference_eq : C = 4 * Real.sqrt 3 * π) :
  ∃ r : ℝ, (V = (1 / 3) * π * r^2 * h) ∧ (C = 2 * π * r) :=
by
  sorry

end circumference_of_cone_base_l28_2833


namespace area_of_ABCD_l28_2851

noncomputable def AB := 6
noncomputable def BC := 8
noncomputable def CD := 15
noncomputable def DA := 17
def right_angle_BCD := true
def convex_ABCD := true

theorem area_of_ABCD : ∃ area : ℝ, area = 110 := by
  -- Given conditions
  have hAB : AB = 6 := rfl
  have hBC : BC = 8 := rfl
  have hCD : CD = 15 := rfl
  have hDA : DA = 17 := rfl
  have hAngle : right_angle_BCD = true := rfl
  have hConvex : convex_ABCD = true := rfl

  -- skip the proof
  sorry

end area_of_ABCD_l28_2851


namespace min_value_of_y_l28_2841

noncomputable def y (x : ℝ) : ℝ := abs (x - 1) + abs (x - 2) - abs (x - 3)

theorem min_value_of_y : ∃ x : ℝ, (∀ x' : ℝ, y x' ≥ y x) ∧ y x = -1 :=
sorry

end min_value_of_y_l28_2841


namespace arithmetic_sequence_30th_term_l28_2807

theorem arithmetic_sequence_30th_term :
  let a₁ := 3
  let d := 4
  let n := 30
  a₁ + (n - 1) * d = 119 :=
by
  let a₁ := 3
  let d := 4
  let n := 30
  show a₁ + (n - 1) * d = 119
  sorry

end arithmetic_sequence_30th_term_l28_2807


namespace cube_vertex_adjacency_l28_2864

noncomputable def beautiful_face (a b c d : ℕ) : Prop :=
  a = b + c + d ∨ b = a + c + d ∨ c = a + b + d ∨ d = a + b + c

theorem cube_vertex_adjacency :
  ∀ (v1 v2 v3 v4 v5 v6 v7 v8 : ℕ), 
  v1 ≠ v2 ∧ v1 ≠ v3 ∧ v1 ≠ v4 ∧ v1 ≠ v5 ∧ v1 ≠ v6 ∧ v1 ≠ v7 ∧ v1 ≠ v8 ∧
  v2 ≠ v3 ∧ v2 ≠ v4 ∧ v2 ≠ v5 ∧ v2 ≠ v6 ∧ v2 ≠ v7 ∧ v2 ≠ v8 ∧
  v3 ≠ v4 ∧ v3 ≠ v5 ∧ v3 ≠ v6 ∧ v3 ≠ v7 ∧ v3 ≠ v8 ∧
  v4 ≠ v5 ∧ v4 ≠ v6 ∧ v4 ≠ v7 ∧ v4 ≠ v8 ∧
  v5 ≠ v6 ∧ v5 ≠ v7 ∧ v5 ≠ v8 ∧
  v6 ≠ v7 ∧ v6 ≠ v8 ∧
  v7 ≠ v8 ∧
  beautiful_face v1 v2 v3 v4 ∧ beautiful_face v5 v6 v7 v8 ∧
  beautiful_face v1 v3 v5 v7 ∧ beautiful_face v2 v4 v6 v8 ∧
  beautiful_face v1 v2 v5 v6 ∧ beautiful_face v3 v4 v7 v8 →
  (v6 = 6 → (v1 = 2 ∧ v2 = 3 ∧ v3 = 5) ∨ 
   (v1 = 3 ∧ v2 = 5 ∧ v3 = 7) ∨ 
   (v1 = 2 ∧ v2 = 3 ∧ v3 = 7)) :=
sorry

end cube_vertex_adjacency_l28_2864


namespace solve_for_y_l28_2849

theorem solve_for_y : ∀ y : ℝ, (y - 5)^3 = (1 / 27)⁻¹ → y = 8 :=
by
  intro y
  intro h
  sorry

end solve_for_y_l28_2849


namespace greatest_possible_sum_of_consecutive_integers_prod_lt_200_l28_2890

theorem greatest_possible_sum_of_consecutive_integers_prod_lt_200 :
  ∃ n : ℤ, (n * (n + 1) < 200) ∧ ( ∀ m : ℤ, (m * (m + 1) < 200) → m ≤ n) ∧ (n + (n + 1) = 27) :=
by
  sorry

end greatest_possible_sum_of_consecutive_integers_prod_lt_200_l28_2890


namespace recurring_division_l28_2876

def recurring_36_as_fraction : ℚ := 36 / 99
def recurring_12_as_fraction : ℚ := 12 / 99

theorem recurring_division :
  recurring_36_as_fraction / recurring_12_as_fraction = 3 := 
sorry

end recurring_division_l28_2876


namespace length_of_median_in_right_triangle_l28_2845

noncomputable def length_of_median (DE DF : ℝ) : ℝ :=
  let EF := Real.sqrt (DE^2 + DF^2)
  EF / 2

theorem length_of_median_in_right_triangle (DE DF : ℝ) (h1 : DE = 5) (h2 : DF = 12) :
  length_of_median DE DF = 6.5 :=
by
  -- Conditions
  rw [h1, h2]
  -- Proof (to be completed)
  sorry

end length_of_median_in_right_triangle_l28_2845


namespace fill_tank_time_l28_2894

-- Define the rates at which the pipes fill or empty the tank
def rateA : ℚ := 1 / 16
def rateB : ℚ := - (1 / 24)  -- Since pipe B empties the tank, it's negative.

-- Define the time after which pipe B is closed
def timeBClosed : ℚ := 21

-- Define the initial combined rate of both pipes
def combinedRate : ℚ := rateA + rateB

-- Define the proportion of the tank filled in the initial 21 minutes
def filledIn21Minutes : ℚ := combinedRate * timeBClosed

-- Define the remaining tank to be filled after pipe B is closed
def remainingTank : ℚ := 1 - filledIn21Minutes

-- Define the additional time required to fill the remaining part of the tank with only pipe A
def additionalTime : ℚ := remainingTank / rateA

-- Total time is the sum of the initial time and additional time
def totalTime : ℚ := timeBClosed + additionalTime

theorem fill_tank_time : totalTime = 30 :=
by
  -- Proof omitted
  sorry

end fill_tank_time_l28_2894


namespace maci_red_pens_l28_2848

def cost_blue_pens (b : ℕ) (cost_blue : ℕ) : ℕ := b * cost_blue

def cost_red_pen (cost_blue : ℕ) : ℕ := 2 * cost_blue

def total_cost (cost_blue : ℕ) (n_blue : ℕ) (n_red : ℕ) : ℕ := 
  n_blue * cost_blue + n_red * (2 * cost_blue)

theorem maci_red_pens :
  ∀ (n_blue cost_blue n_red total : ℕ),
  n_blue = 10 →
  cost_blue = 10 →
  total = 400 →
  total_cost cost_blue n_blue n_red = total →
  n_red = 15 := 
by
  intros n_blue cost_blue n_red total h1 h2 h3 h4
  sorry

end maci_red_pens_l28_2848


namespace carmen_sold_1_box_of_fudge_delights_l28_2867

noncomputable def boxes_of_fudge_delights (total_earned: ℝ) (samoas_price: ℝ) (thin_mints_price: ℝ) (fudge_delights_price: ℝ) (sugar_cookies_price: ℝ) (samoas_sold: ℝ) (thin_mints_sold: ℝ) (sugar_cookies_sold: ℝ): ℝ :=
  let samoas_total := samoas_price * samoas_sold
  let thin_mints_total := thin_mints_price * thin_mints_sold
  let sugar_cookies_total := sugar_cookies_price * sugar_cookies_sold
  let other_cookies_total := samoas_total + thin_mints_total + sugar_cookies_total
  (total_earned - other_cookies_total) / fudge_delights_price

theorem carmen_sold_1_box_of_fudge_delights: boxes_of_fudge_delights 42 4 3.5 5 2 3 2 9 = 1 :=
by
  sorry

end carmen_sold_1_box_of_fudge_delights_l28_2867


namespace baez_marble_loss_l28_2895

theorem baez_marble_loss :
  ∃ p : ℚ, (p > 0 ∧ (p / 100) * 25 * 2 = 60) ∧ p = 20 :=
by
  sorry

end baez_marble_loss_l28_2895


namespace find_integers_l28_2859

theorem find_integers (a b c d : ℕ) (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) 
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d) 
  (h1 : a + b + c = 6) 
  (h2 : a + b + d = 7) 
  (h3 : a + c + d = 8) 
  (h4 : b + c + d = 9) : 
  (a, b, c, d) = (1, 2, 3, 4) ∨ (a, b, c, d) = (1, 2, 4, 3) ∨ (a, b, c, d) = (1, 3, 2, 4) ∨ (a, b, c, d) = (1, 3, 4, 2) ∨ (a, b, c, d) = (1, 4, 2, 3) ∨ (a, b, c, d) = (1, 4, 3, 2) ∨ (a, b, c, d) = (2, 1, 3, 4) ∨ (a, b, c, d) = (2, 1, 4, 3) ∨ (a, b, c, d) = (2, 3, 1, 4) ∨ (a, b, c, d) = (2, 3, 4, 1) ∨ (a, b, c, d) = (2, 4, 1, 3) ∨ (a, b, c, d) = (2, 4, 3, 1) ∨ (a, b, c, d) = (3, 1, 2, 4) ∨ (a, b, c, d) = (3, 1, 4, 2) ∨ (a, b, c, d) = (3, 2, 1, 4) ∨ (a, b, c, d) = (3, 2, 4, 1) ∨ (a, b, c, d) = (3, 4, 1, 2) ∨ (a, b, c, d) = (3, 4, 2, 1) ∨ (a, b, c, d) = (4, 1, 2, 3) ∨ (a, b, c, d) = (4, 1, 3, 2) ∨ (a, b, c, d) = (4, 2, 1, 3) ∨ (a, b, c, d) = (4, 2, 3, 1) ∨ (a, b, c, d) = (4, 3, 1, 2) ∨ (a, b, c, d) = (4, 3, 2, 1) :=
sorry

end find_integers_l28_2859


namespace circle_sum_value_l28_2872

-- Define the problem
theorem circle_sum_value (a b x : ℕ) (h1 : a = 35) (h2 : b = 47) : x = a + b :=
by
  -- Given conditions
  have ha : a = 35 := h1
  have hb : b = 47 := h2
  -- Prove that the value of x is the sum of a and b
  have h_sum : x = a + b := sorry
  -- Assert the value of x is 82 based on given a and b
  exact h_sum

end circle_sum_value_l28_2872


namespace trigonometric_identity_l28_2835

theorem trigonometric_identity :
  (Real.cos (Real.pi / 3)) - (Real.tan (Real.pi / 4)) + (3 / 4) * (Real.tan (Real.pi / 6))^2 - (Real.sin (Real.pi / 6)) + (Real.cos (Real.pi / 6))^2 = 0 :=
by
  sorry

end trigonometric_identity_l28_2835


namespace Reese_initial_savings_l28_2820

theorem Reese_initial_savings (F M A R : ℝ) (savings : ℝ) :
  F = 0.2 * savings →
  M = 0.4 * savings →
  A = 1500 →
  R = 2900 →
  savings = 11000 :=
by
  sorry

end Reese_initial_savings_l28_2820


namespace simplify_fraction_l28_2885

theorem simplify_fraction (a b : ℕ) (h : a = 2020) (h2 : b = 2018) :
  (2 ^ a - 2 ^ b) / (2 ^ a + 2 ^ b) = 3 / 5 := by
  sorry

end simplify_fraction_l28_2885


namespace least_number_with_remainder_l28_2811

variable (x : ℕ)

theorem least_number_with_remainder (x : ℕ) : 
  (x % 16 = 11) ∧ (x % 27 = 11) ∧ (x % 34 = 11) ∧ (x % 45 = 11) ∧ (x % 144 = 11) → x = 36731 := by
  sorry

end least_number_with_remainder_l28_2811


namespace alejandro_candies_l28_2837

theorem alejandro_candies (n : ℕ) (S_n : ℕ) :
  (S_n = 2^n - 1 ∧ S_n ≥ 2007) → ((2^11 - 1 - 2007 = 40) ∧ (∃ k, k = 11)) :=
  by
    sorry

end alejandro_candies_l28_2837


namespace x_condition_sufficient_not_necessary_l28_2803

theorem x_condition_sufficient_not_necessary (x : ℝ) : (x < -1 → x^2 - 1 > 0) ∧ (¬ (∀ x, x^2 - 1 > 0 → x < -1)) :=
by
  sorry

end x_condition_sufficient_not_necessary_l28_2803


namespace part1_part2_l28_2889

open Real

variable {x y a: ℝ}

-- Condition for the second proof to avoid division by zero
variable (h1 : a ≠ 1) (h2 : a ≠ 4) (h3 : a ≠ -4)

theorem part1 : (x + y)^2 + y * (3 * x - y) = x^2 + 5 * (x * y) := 
by sorry

theorem part2 (h1: a ≠ 1) (h2: a ≠ 4) (h3: a ≠ -4) : 
  ((4 - a^2) / (a - 1) + a) / ((a^2 - 16) / (a - 1)) = -1 / (a + 4) := 
by sorry

end part1_part2_l28_2889


namespace square_area_given_equal_perimeters_l28_2869

theorem square_area_given_equal_perimeters 
  (a b c : ℝ) (a_eq : a = 7.5) (b_eq : b = 9.5) (c_eq : c = 12) 
  (sq_perimeter_eq_tri : 4 * s = a + b + c) : 
  s^2 = 52.5625 :=
by
  sorry

end square_area_given_equal_perimeters_l28_2869


namespace greatest_number_of_groups_l28_2863

theorem greatest_number_of_groups (s a t b n : ℕ) (hs : s = 10) (ha : a = 15) (ht : t = 12) (hb : b = 18) :
  (∀ n, n ≤ n ∧ n ∣ s ∧ n ∣ a ∧ n ∣ t ∧ n ∣ b ∧ n > 1 → 
  (s / n < (a / n) + (t / n) + (b / n))
  ∧ (∃ groups, groups = n)) → n = 3 :=
sorry

end greatest_number_of_groups_l28_2863


namespace number_of_employees_excluding_manager_l28_2815

theorem number_of_employees_excluding_manager 
  (avg_salary : ℕ)
  (manager_salary : ℕ)
  (new_avg_salary : ℕ)
  (n : ℕ)
  (T : ℕ)
  (h1 : avg_salary = 1600)
  (h2 : manager_salary = 3700)
  (h3 : new_avg_salary = 1700)
  (h4 : T = n * avg_salary)
  (h5 : T + manager_salary = (n + 1) * new_avg_salary) :
  n = 20 :=
by
  sorry

end number_of_employees_excluding_manager_l28_2815


namespace carpets_triple_overlap_area_l28_2878

theorem carpets_triple_overlap_area {W H : ℕ} (hW : W = 10) (hH : H = 10) 
    {w1 h1 w2 h2 w3 h3 : ℕ} 
    (h1_w1 : w1 = 6) (h1_h1 : h1 = 8)
    (h2_w2 : w2 = 6) (h2_h2 : h2 = 6)
    (h3_w3 : w3 = 5) (h3_h3 : h3 = 7) :
    ∃ (area : ℕ), area = 6 := by
  sorry

end carpets_triple_overlap_area_l28_2878


namespace correct_calculation_l28_2852

theorem correct_calculation (x : ℝ) (h : 3 * x - 12 = 60) : (x / 3) + 12 = 20 :=
by 
  sorry

end correct_calculation_l28_2852


namespace eraser_ratio_l28_2899

-- Define the variables and conditions
variables (c j g : ℕ)
variables (total : ℕ := 35)
variables (c_erasers : ℕ := 10)
variables (gabriel_erasers : ℕ := c_erasers / 2)
variables (julian_erasers : ℕ := c_erasers)

-- The proof statement
theorem eraser_ratio (hc : c_erasers = 10)
                      (h1 : c_erasers = 2 * gabriel_erasers)
                      (h2 : julian_erasers = c_erasers)
                      (h3 : c_erasers + gabriel_erasers + julian_erasers = total) :
                      julian_erasers / c_erasers = 1 :=
by
  sorry

end eraser_ratio_l28_2899


namespace molecular_weight_H_of_H2CrO4_is_correct_l28_2826

-- Define the atomic weight of hydrogen
def atomic_weight_H : ℝ := 1.008

-- Define the number of hydrogen atoms in H2CrO4
def num_H_atoms_in_H2CrO4 : ℕ := 2

-- Define the molecular weight of the compound H2CrO4
def molecular_weight_H2CrO4 : ℝ := 118

-- Define the molecular weight of the hydrogen part (H2)
def molecular_weight_H2 : ℝ := atomic_weight_H * num_H_atoms_in_H2CrO4

-- The statement to prove
theorem molecular_weight_H_of_H2CrO4_is_correct : molecular_weight_H2 = 2.016 :=
by
  sorry

end molecular_weight_H_of_H2CrO4_is_correct_l28_2826


namespace stack_trays_height_l28_2816

theorem stack_trays_height
  (thickness : ℕ)
  (top_diameter : ℕ)
  (bottom_diameter : ℕ)
  (decrement_step : ℕ)
  (base_height : ℕ)
  (cond1 : thickness = 2)
  (cond2 : top_diameter = 30)
  (cond3 : bottom_diameter = 8)
  (cond4 : decrement_step = 2)
  (cond5 : base_height = 2) :
  (bottom_diameter + decrement_step * (top_diameter - bottom_diameter) / decrement_step * thickness + base_height) = 26 :=
by
  sorry

end stack_trays_height_l28_2816


namespace part1_intersection_1_part1_union_1_part2_range_a_l28_2808

open Set

def U := ℝ
def A (x : ℝ) := -1 < x ∧ x < 3
def B (a x : ℝ) := a - 1 ≤ x ∧ x ≤ a + 6

noncomputable def part1_a : ℝ → Prop := sorry
noncomputable def part1_b : ℝ → Prop := sorry

-- part (1)
theorem part1_intersection_1 (a : ℝ) : A x ∧ B a x := sorry

theorem part1_union_1 (a : ℝ) : A x ∨ B a x := sorry

-- part (2)
theorem part2_range_a : {a : ℝ | -3 ≤ a ∧ a ≤ 0} := sorry

end part1_intersection_1_part1_union_1_part2_range_a_l28_2808


namespace garden_fencing_needed_l28_2856

/-- Given a rectangular garden where the length is 300 yards and the length is twice the width,
prove that the total amount of fencing needed to enclose the garden is 900 yards. -/
theorem garden_fencing_needed :
  ∃ (W L P : ℝ), L = 300 ∧ L = 2 * W ∧ P = 2 * (L + W) ∧ P = 900 :=
by
  sorry

end garden_fencing_needed_l28_2856


namespace correct_product_l28_2887

theorem correct_product (a b : ℚ) (calc_incorrect : a = 52 ∧ b = 735)
                        (incorrect_product : a * b = 38220) :
  (0.52 * 7.35 = 3.822) :=
by
  sorry

end correct_product_l28_2887


namespace quentavious_gum_count_l28_2884

def initial_nickels : Nat := 5
def remaining_nickels : Nat := 2
def gum_per_nickel : Nat := 2
def traded_nickels (initial remaining : Nat) : Nat := initial - remaining
def total_gum (trade_n gum_per_n : Nat) : Nat := trade_n * gum_per_n

theorem quentavious_gum_count : total_gum (traded_nickels initial_nickels remaining_nickels) gum_per_nickel = 6 := by
  sorry

end quentavious_gum_count_l28_2884


namespace circle_line_tangent_l28_2838

theorem circle_line_tangent (m : ℝ) :
  (∃ x y : ℝ, x^2 + y^2 = 4 * m ∧ x + y = 2 * m) ↔ m = 2 :=
sorry

end circle_line_tangent_l28_2838


namespace matrix_power_15_l28_2818

-- Define the matrix B
def B : Matrix (Fin 3) (Fin 3) ℝ :=
  !![ 0, -1,  0;
      1,  0,  0;
      0,  0,  1]

-- Define what we want to prove
theorem matrix_power_15 :
  B^15 = !![ 0,  1,  0;
            -1,  0,  0;
             0,  0,  1] :=
sorry

end matrix_power_15_l28_2818


namespace Jack_can_form_rectangle_l28_2896

theorem Jack_can_form_rectangle : 
  ∃ (a b : ℕ), 
  3 * a = 2016 ∧ 
  4 * a = 2016 ∧ 
  4 * b = 2016 ∧ 
  3 * b = 2016 ∧ 
  (503 * 4 + 3 * 9 = 2021) ∧ 
  (2 * 3 = 4) :=
by 
  sorry

end Jack_can_form_rectangle_l28_2896


namespace find_a_l28_2886

theorem find_a (a b c : ℝ) (h1 : a * b * c = (Real.sqrt ((a + 2) * (b + 3))) / (c + 1))
                 (h2 : a * 15 * 7 = 1.5) : a = 6 :=
sorry

end find_a_l28_2886


namespace max_frac_a_S_l28_2868

def S (n : ℕ) : ℕ := 2^n - 1

def a (n : ℕ) : ℕ :=
  if n = 1 then 1
  else S n - S (n - 1)

theorem max_frac_a_S (n : ℕ) (h : S n = 2^n - 1) : 
  let frac := (a n) / (a n * S n + a 6)
  ∃ N : ℕ, N > 0 ∧ (frac ≤ 1 / 15) := by
  sorry

end max_frac_a_S_l28_2868


namespace flood_damage_in_euros_l28_2829

variable (yen_damage : ℕ) (yen_per_euro : ℕ) (tax_rate : ℝ)

theorem flood_damage_in_euros : 
  yen_damage = 4000000000 →
  yen_per_euro = 110 →
  tax_rate = 1.05 →
  (yen_damage / yen_per_euro : ℝ) * tax_rate = 38181818 :=
by {
  -- We could include necessary lean proof steps here, but we use sorry to skip the proof.
  sorry
}

end flood_damage_in_euros_l28_2829


namespace ratio_S6_S3_l28_2821

theorem ratio_S6_S3 (a : ℝ) (q : ℝ) (h : a + 8 * a * q^3 = 0) : 
  (a * (1 - q^6) / (1 - q)) / (a * (1 - q^3) / (1 - q)) = 9 / 8 :=
by
  sorry

end ratio_S6_S3_l28_2821


namespace total_number_of_drivers_l28_2834

theorem total_number_of_drivers (N : ℕ) (A_drivers : ℕ) (B_sample : ℕ) (C_sample : ℕ) (D_sample : ℕ)
  (A_sample : ℕ)
  (hA : A_drivers = 96)
  (hA_sample : A_sample = 12)
  (hB_sample : B_sample = 21)
  (hC_sample : C_sample = 25)
  (hD_sample : D_sample = 43) :
  N = 808 :=
by
  -- skipping the proof here
  sorry

end total_number_of_drivers_l28_2834


namespace simplify_and_evaluate_l28_2825

theorem simplify_and_evaluate :
  ∀ (x : ℝ), x = -3 → 7 * x^2 - 3 * (2 * x^2 - 1) - 4 = 8 :=
by
  intros x hx
  rw [hx]
  sorry

end simplify_and_evaluate_l28_2825


namespace find_y_given_conditions_l28_2817

theorem find_y_given_conditions (t : ℚ) (x y : ℚ) (h1 : x = 3 - 2 * t) (h2 : y = 5 * t + 9) (h3 : x = 0) : y = 33 / 2 := by
  sorry

end find_y_given_conditions_l28_2817


namespace sum_integer_solutions_l28_2882

theorem sum_integer_solutions (n : ℤ) (h1 : |n^2| < |n - 5|^2) (h2 : |n - 5|^2 < 16) : n = 2 := 
sorry

end sum_integer_solutions_l28_2882


namespace round_2748397_542_nearest_integer_l28_2866

theorem round_2748397_542_nearest_integer :
  let n := 2748397.542
  let int_part := 2748397
  let decimal_part := 0.542
  (n.round = 2748398) :=
by
  sorry

end round_2748397_542_nearest_integer_l28_2866


namespace optionA_optionC_l28_2870

noncomputable def f (x : ℝ) : ℝ := Real.log (|x - 2| + 1)

theorem optionA : ∀ x : ℝ, f (x + 2) = f (-x + 2) := 
by sorry

theorem optionC : (∀ x : ℝ, x < 2 → f x > f (x + 0.01)) ∧ (∀ x : ℝ, x > 2 → f x < f (x - 0.01)) := 
by sorry

end optionA_optionC_l28_2870


namespace expression_divisible_by_10_l28_2832

theorem expression_divisible_by_10 (n : ℕ) : 10 ∣ (3 ^ (n + 2) - 2 ^ (n + 2) + 3 ^ n - 2 ^ n) :=
  sorry

end expression_divisible_by_10_l28_2832


namespace solve_inequality_l28_2875

theorem solve_inequality : {x : ℝ | 3 * x ^ 2 - 7 * x - 6 < 0} = {x : ℝ | -2 / 3 < x ∧ x < 3} :=
sorry

end solve_inequality_l28_2875


namespace arithmetic_expression_l28_2810

theorem arithmetic_expression :
  (((15 - 2) + (4 / (1 / 2)) - (6 * 8)) * (100 - 24)) / 38 = -54 := by
  sorry

end arithmetic_expression_l28_2810


namespace minimum_weights_l28_2800

variable {α : Type} [LinearOrderedField α]

theorem minimum_weights (weights : Finset α)
  (h_unique : weights.card = 5)
  (h_balanced : ∀ {x y : α}, x ∈ weights → y ∈ weights → x ≠ y →
    ∃ a b : α, a ∈ weights ∧ b ∈ weights ∧ x + y = a + b) :
  ∃ (n : ℕ), n = 13 ∧ ∀ S : Finset α, S.card = n ∧
    (∀ {x y : α}, x ∈ S → y ∈ S → x ≠ y → ∃ a b : α, a ∈ S ∧ b ∈ S ∧ x + y = a + b) :=
by
  sorry

end minimum_weights_l28_2800


namespace intersection_of_cylinders_within_sphere_l28_2824

theorem intersection_of_cylinders_within_sphere (a b c d e f : ℝ) :
    ∀ (x y z : ℝ), 
      (x - a)^2 + (y - b)^2 < 1 ∧ 
      (y - c)^2 + (z - d)^2 < 1 ∧ 
      (z - e)^2 + (x - f)^2 < 1 → 
      (x - (a + f) / 2)^2 + (y - (b + c) / 2)^2 + (z - (d + e) / 2)^2 < 3 / 2 :=
by
  sorry

end intersection_of_cylinders_within_sphere_l28_2824


namespace find_a_10_l28_2855

/-- 
a_n is an arithmetic sequence
-/
def a (n : ℕ) : ℝ := sorry

/-- 
Given conditions:
- Condition 1: a_2 + a_5 = 19
- Condition 2: S_5 = 40, where S_5 is the sum of the first five terms
-/
axiom condition1 : a 2 + a 5 = 19
axiom condition2 : (a 1 + a 2 + a 3 + a 4 + a 5) = 40

noncomputable def a_10 : ℝ := a 10

theorem find_a_10 : a_10 = 29 :=
by
  sorry

end find_a_10_l28_2855


namespace simplify_expression_l28_2806

theorem simplify_expression (tan_60 cot_60 : ℝ) (h1 : tan_60 = Real.sqrt 3) (h2 : cot_60 = 1 / Real.sqrt 3) :
  (tan_60^3 + cot_60^3) / (tan_60 + cot_60) = 31 / 3 :=
by
  -- proof will go here
  sorry

end simplify_expression_l28_2806


namespace total_sweaters_knit_l28_2877

-- Definitions from condition a)
def monday_sweaters : ℕ := 8
def tuesday_sweaters : ℕ := monday_sweaters + 2
def wednesday_sweaters : ℕ := tuesday_sweaters - 4
def thursday_sweaters : ℕ := wednesday_sweaters
def friday_sweaters : ℕ := monday_sweaters / 2

-- Theorem statement
theorem total_sweaters_knit : 
  monday_sweaters + tuesday_sweaters + wednesday_sweaters + thursday_sweaters + friday_sweaters = 34 :=
  by
    sorry

end total_sweaters_knit_l28_2877


namespace initial_students_per_class_l28_2805

theorem initial_students_per_class (students_per_class initial_classes additional_classes total_students : ℕ) 
  (h1 : initial_classes = 15) 
  (h2 : additional_classes = 5) 
  (h3 : total_students = 400) 
  (h4 : students_per_class * (initial_classes + additional_classes) = total_students) : 
  students_per_class = 20 := 
by 
  -- Proof goes here
  sorry

end initial_students_per_class_l28_2805


namespace train_crosses_signal_pole_in_18_seconds_l28_2850

-- Define the given conditions
def train_length := 300  -- meters
def platform_length := 450  -- meters
def time_to_cross_platform := 45  -- seconds

-- Define the question and the correct answer
def time_to_cross_signal_pole := 18  -- seconds (this is what we need to prove)

-- Define the total distance the train covers when crossing the platform
def total_distance_crossing_platform := train_length + platform_length  -- meters

-- Define the speed of the train
def train_speed := total_distance_crossing_platform / time_to_cross_platform  -- meters per second

theorem train_crosses_signal_pole_in_18_seconds :
  300 / train_speed = time_to_cross_signal_pole :=
by
  -- train_speed is defined directly in terms of the given conditions
  unfold train_speed total_distance_crossing_platform train_length platform_length time_to_cross_platform
  sorry

end train_crosses_signal_pole_in_18_seconds_l28_2850


namespace probability_not_finish_l28_2801

theorem probability_not_finish (p : ℝ) (h : p = 5 / 8) : 1 - p = 3 / 8 := 
by 
  rw [h]
  norm_num

end probability_not_finish_l28_2801


namespace survival_rate_is_100_percent_l28_2813

-- Definitions of conditions
def planted_trees : ℕ := 99
def survived_trees : ℕ := 99

-- Definition of survival rate
def survival_rate : ℕ := (survived_trees * 100) / planted_trees

-- Proof statement
theorem survival_rate_is_100_percent : survival_rate = 100 := by
  sorry

end survival_rate_is_100_percent_l28_2813


namespace circle_intersection_range_l28_2844

theorem circle_intersection_range (r : ℝ) (H : r > 0) :
  (∃ (x y : ℝ), x^2 + y^2 = r^2 ∧ (x+3)^2 + (y-4)^2 = 36) → (1 < r ∧ r < 11) := 
by
  sorry

end circle_intersection_range_l28_2844


namespace difference_of_squares_l28_2861

theorem difference_of_squares (x y : ℝ) (h1 : x + y = 10) (h2 : x - y = 8) : x^2 - y^2 = 80 := 
sorry

end difference_of_squares_l28_2861


namespace jack_finishes_book_in_13_days_l28_2814

def total_pages : ℕ := 285
def pages_per_day : ℕ := 23

theorem jack_finishes_book_in_13_days : (total_pages + pages_per_day - 1) / pages_per_day = 13 := by
  sorry

end jack_finishes_book_in_13_days_l28_2814


namespace boxes_neither_markers_nor_crayons_l28_2853

theorem boxes_neither_markers_nor_crayons (total boxes_markers boxes_crayons boxes_both: ℕ)
  (htotal : total = 15)
  (hmarkers : boxes_markers = 9)
  (hcrayons : boxes_crayons = 4)
  (hboth : boxes_both = 5) :
  total - (boxes_markers + boxes_crayons - boxes_both) = 7 := by
  sorry

end boxes_neither_markers_nor_crayons_l28_2853


namespace hyperbola_vertex_distance_l28_2860

theorem hyperbola_vertex_distance :
  ∀ (x y : ℝ),
  4 * x^2 + 24 * x - 4 * y^2 + 16 * y + 44 = 0 →
  2 = 2 :=
by
  intros x y h
  sorry

end hyperbola_vertex_distance_l28_2860


namespace lottery_ticket_not_necessarily_win_l28_2840

/-- Given a lottery with 1,000,000 tickets and a winning rate of 0.001, buying 1000 tickets may not necessarily win. -/
theorem lottery_ticket_not_necessarily_win (total_tickets : ℕ) (winning_rate : ℚ) (n_tickets : ℕ) :
  total_tickets = 1000000 →
  winning_rate = 1 / 1000 →
  n_tickets = 1000 →
  ∃ (p : ℚ), 0 < p ∧ p < 1 ∧ (p ^ n_tickets) < (1 / total_tickets) := 
by
  intros h_total h_rate h_n
  sorry

end lottery_ticket_not_necessarily_win_l28_2840


namespace inverse_g_of_87_l28_2871

noncomputable def g (x : ℝ) : ℝ := 3 * x^3 + 6

theorem inverse_g_of_87 : (g x = 87) → (x = 3) :=
by
  intro h
  sorry

end inverse_g_of_87_l28_2871


namespace justin_current_age_l28_2862

theorem justin_current_age (angelina_future_age : ℕ) (years_until_future : ℕ) (age_difference : ℕ)
  (h_future_age : angelina_future_age = 40) (h_years_until_future : years_until_future = 5)
  (h_age_difference : age_difference = 4) : 
  (angelina_future_age - years_until_future) - age_difference = 31 :=
by
  -- This is where the proof would go.
  sorry

end justin_current_age_l28_2862


namespace square_paintings_size_l28_2883

theorem square_paintings_size (total_area : ℝ) (small_paintings_count : ℕ) (small_painting_area : ℝ) 
                              (large_painting_area : ℝ) (square_paintings_count : ℕ) (square_paintings_total_area : ℝ) : 
  total_area = small_paintings_count * small_painting_area + large_painting_area + square_paintings_total_area → 
  square_paintings_count = 3 → 
  small_paintings_count = 4 → 
  small_painting_area = 2 * 3 → 
  large_painting_area = 10 * 15 → 
  square_paintings_total_area = 3 * 6^2 → 
  ∃ side_length, side_length^2 = (square_paintings_total_area / square_paintings_count) ∧ side_length = 6 := 
by
  intro h_total h_square_count h_small_count h_small_area h_large_area h_square_total 
  use 6
  sorry

end square_paintings_size_l28_2883


namespace original_cost_price_of_car_l28_2865

theorem original_cost_price_of_car (x : ℝ) (y : ℝ) (h1 : y = 0.87 * x) (h2 : 1.20 * y = 54000) :
  x = 54000 / 1.044 :=
by
  sorry

end original_cost_price_of_car_l28_2865


namespace red_button_probability_l28_2873

-- Definitions of the initial state
def initial_red_buttons : ℕ := 8
def initial_blue_buttons : ℕ := 12
def total_buttons := initial_red_buttons + initial_blue_buttons

-- Condition of removal and remaining buttons
def removed_buttons := total_buttons - (5 / 8 : ℚ) * total_buttons

-- Equal number of red and blue buttons removed
def removed_red_buttons := removed_buttons / 2
def removed_blue_buttons := removed_buttons / 2

-- State after removal
def remaining_red_buttons := initial_red_buttons - removed_red_buttons
def remaining_blue_buttons := initial_blue_buttons - removed_blue_buttons

-- Jars after removal
def jar_X := remaining_red_buttons + remaining_blue_buttons
def jar_Y := removed_red_buttons + removed_blue_buttons

-- Probability calculations
def probability_red_X : ℚ := remaining_red_buttons / jar_X
def probability_red_Y : ℚ := removed_red_buttons / jar_Y

-- Final probability
def final_probability : ℚ := probability_red_X * probability_red_Y

theorem red_button_probability :
  final_probability = 4 / 25 := 
  sorry

end red_button_probability_l28_2873
