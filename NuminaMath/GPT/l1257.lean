import Mathlib

namespace exists_nat_n_gt_one_sqrt_expr_nat_l1257_125761

theorem exists_nat_n_gt_one_sqrt_expr_nat (n : ℕ) : ∃ (n : ℕ), n > 1 ∧ ∃ (m : ℕ), n^(7 / 8) = m :=
by
  sorry

end exists_nat_n_gt_one_sqrt_expr_nat_l1257_125761


namespace ferry_distance_l1257_125703

theorem ferry_distance 
  (x : ℝ)
  (v_w : ℝ := 3)  -- speed of water flow in km/h
  (t_downstream : ℝ := 5)  -- time taken to travel downstream in hours
  (t_upstream : ℝ := 7)  -- time taken to travel upstream in hours
  (eqn : x / t_downstream - v_w = x / t_upstream + v_w) :
  x = 105 :=
sorry

end ferry_distance_l1257_125703


namespace grace_age_l1257_125798

theorem grace_age 
  (H : ℕ) 
  (I : ℕ) 
  (J : ℕ) 
  (G : ℕ)
  (h1 : H = I - 5)
  (h2 : I = J + 7)
  (h3 : G = 2 * J)
  (h4 : H = 18) : 
  G = 32 := 
sorry

end grace_age_l1257_125798


namespace rectangle_area_l1257_125769

theorem rectangle_area (P l w : ℕ) (h_perimeter: 2 * l + 2 * w = 60) (h_aspect: l = 3 * w / 2) : l * w = 216 :=
sorry

end rectangle_area_l1257_125769


namespace no_rational_roots_l1257_125782

theorem no_rational_roots (p q : ℤ) (h1 : p % 3 = 2) (h2 : q % 3 = 2) :
  ¬ ∃ (a b : ℤ), b ≠ 0 ∧ a * a = b * b * (p^2 - 4 * q) :=
by
  sorry

end no_rational_roots_l1257_125782


namespace probability_of_composite_l1257_125744

def is_composite (n : ℕ) : Prop :=
  ∃ m k : ℕ, 1 < m ∧ m < n ∧ 1 < k ∧ k < n ∧ m * k = n

def dice_outcomes (faces : ℕ) (rolls : ℕ) : ℕ :=
  faces ^ rolls

def non_composite_product_ways : ℕ :=
  1 + (3 * 4)  -- one way for all 1s, plus combinations of (1,1,1,{2,3,5})

def total_outcomes : ℕ :=
  dice_outcomes 6 4  -- 6^4 total possible outcomes

def probability_composite : ℚ :=
  1 - (non_composite_product_ways / total_outcomes)

theorem probability_of_composite:
  probability_composite = 1283 / 1296 := 
by
  sorry

end probability_of_composite_l1257_125744


namespace find_original_number_l1257_125774

theorem find_original_number (x : ℤ) (h : 3 * (2 * x + 8) = 84) : x = 10 :=
by
  sorry

end find_original_number_l1257_125774


namespace journey_total_distance_l1257_125732

theorem journey_total_distance (D : ℝ) 
  (h1 : (D / 3) / 21 + (D / 3) / 14 + (D / 3) / 6 = 12) : 
  D = 126 :=
sorry

end journey_total_distance_l1257_125732


namespace surface_area_after_removing_corners_l1257_125754

-- Define the dimensions of the cubes
def original_cube_side : ℝ := 4
def corner_cube_side : ℝ := 2

-- The surface area function for a cube with given side length
def surface_area (side : ℝ) : ℝ := 6 * side * side

theorem surface_area_after_removing_corners :
  surface_area original_cube_side = 96 :=
by
  sorry

end surface_area_after_removing_corners_l1257_125754


namespace largest_real_number_l1257_125735

theorem largest_real_number (x : ℝ) (h : (⌊x⌋ : ℝ) / x = 8 / 9) : x = 63 / 8 := sorry

end largest_real_number_l1257_125735


namespace number_of_triangles_l1257_125764

/-- 
  This statement defines and verifies the number of triangles 
  in the given geometric figure.
-/
theorem number_of_triangles (rectangle : Set ℝ) : 
  (exists lines : Set (List (ℝ × ℝ)), -- assuming a set of lines dividing the rectangle
    let small_right_triangles := 40
    let intermediate_isosceles_triangles := 8
    let intermediate_triangles := 10
    let larger_right_triangles := 20
    let largest_isosceles_triangles := 5
    small_right_triangles + intermediate_isosceles_triangles + intermediate_triangles + larger_right_triangles + largest_isosceles_triangles = 83) :=
sorry

end number_of_triangles_l1257_125764


namespace balls_in_boxes_ways_l1257_125737

theorem balls_in_boxes_ways : ∃ (ways : ℕ), ways = 56 :=
by
  let n := 5
  let m := 4
  let ways := 56
  sorry

end balls_in_boxes_ways_l1257_125737


namespace symmetry_axes_condition_l1257_125707

/-- Define the property of having axes of symmetry for a geometric figure -/
def has_symmetry_axes (bounded : Bool) (two_parallel_axes : Bool) : Prop :=
  if bounded then 
    ¬ two_parallel_axes 
  else 
    true

/-- Main theorem stating the condition on symmetry axes for bounded and unbounded geometric figures -/
theorem symmetry_axes_condition (bounded : Bool) : 
  ∃ two_parallel_axes : Bool, has_symmetry_axes bounded two_parallel_axes :=
by
  -- The proof itself is not necessary as per the problem statement
  sorry

end symmetry_axes_condition_l1257_125707


namespace candy_problem_l1257_125714

theorem candy_problem (
  a : ℤ
) : (a % 10 = 6) →
    (a % 15 = 11) →
    (200 ≤ a ∧ a ≤ 250) →
    (a = 206 ∨ a = 236) :=
sorry

end candy_problem_l1257_125714


namespace grill_runtime_l1257_125762

theorem grill_runtime
    (burn_rate : ℕ)
    (burn_time : ℕ)
    (bags : ℕ)
    (coals_per_bag : ℕ)
    (total_burnt_coals : ℕ)
    (total_time : ℕ)
    (h1 : burn_rate = 15)
    (h2 : burn_time = 20)
    (h3 : bags = 3)
    (h4 : coals_per_bag = 60)
    (h5 : total_burnt_coals = bags * coals_per_bag)
    (h6 : total_time = (total_burnt_coals / burn_rate) * burn_time) :
    total_time = 240 :=
by sorry

end grill_runtime_l1257_125762


namespace curves_intersect_at_three_points_l1257_125723

theorem curves_intersect_at_three_points (b : ℝ) :
  (∃ x y : ℝ, x^2 + y^2 = b^2 ∧ y = 2 * x^2 - b) ∧ 
  (∀ x₁ y₁ x₂ y₂ x₃ y₃ : ℝ,
    (x₁ ≠ x₂ ∨ y₁ ≠ y₂) ∧ (x₁ ≠ x₃ ∨ y₁ ≠ y₃) ∧ (x₂ ≠ x₃ ∨ y₂ ≠ y₃) ∧
    (x₁^2 + y₁^2 = b^2) ∧ (x₂^2 + y₂^2 = b^2) ∧ (x₃^2 + y₃^2 = b^2) ∧
    (y₁ = 2 * x₁^2 - b) ∧ (y₂ = 2 * x₂^2 - b) ∧ (y₃ = 2 * x₃^2 - b)) ↔ b > 1 / 4 :=
by
  sorry

end curves_intersect_at_three_points_l1257_125723


namespace initial_ants_count_l1257_125791

theorem initial_ants_count (n : ℕ) (h1 : ∀ x : ℕ, x ≠ n - 42 → x ≠ 42) : n = 42 :=
sorry

end initial_ants_count_l1257_125791


namespace neg_P_is_univ_l1257_125708

noncomputable def P : Prop :=
  ∃ x0 : ℝ, x0^2 + 2 * x0 + 2 ≤ 0

theorem neg_P_is_univ :
  ¬ P ↔ ∀ x : ℝ, x^2 + 2 * x + 2 > 0 :=
by {
  sorry
}

end neg_P_is_univ_l1257_125708


namespace tiger_distance_traveled_l1257_125726

theorem tiger_distance_traveled :
  let distance1 := 25 * 1
  let distance2 := 35 * 2
  let distance3 := 20 * 1.5
  let distance4 := 10 * 1
  let distance5 := 50 * 0.5
  distance1 + distance2 + distance3 + distance4 + distance5 = 160 := by
sorry

end tiger_distance_traveled_l1257_125726


namespace savings_example_l1257_125709

def window_cost : ℕ → ℕ := λ n => n * 120

def discount_windows (n : ℕ) : ℕ := (n / 6) * 2 + n

def effective_cost (needed : ℕ) : ℕ := 
  let free_windows := (needed / 8) * 2
  (needed - free_windows) * 120

def combined_cost (n m : ℕ) : ℕ :=
  effective_cost (n + m)

def separate_cost (needed1 needed2 : ℕ) : ℕ :=
  effective_cost needed1 + effective_cost needed2

def savings_if_combined (n m : ℕ) : ℕ :=
  separate_cost n m - combined_cost n m

theorem savings_example : savings_if_combined 12 9 = 360 := by
  sorry

end savings_example_l1257_125709


namespace class_speeds_relationship_l1257_125718

theorem class_speeds_relationship (x : ℝ) (hx : 0 < x) :
    (15 / (1.2 * x)) = ((15 / x) - (1 / 2)) :=
sorry

end class_speeds_relationship_l1257_125718


namespace find_a_l1257_125779

noncomputable def A : Set ℝ := {1, 2, 3}
noncomputable def B (a : ℝ) : Set ℝ := { x | x^2 - (a + 1) * x + a = 0 }

theorem find_a (a : ℝ) (h : A ∪ B a = A) : a = 1 ∨ a = 2 ∨ a = 3 :=
by
  sorry

end find_a_l1257_125779


namespace promotional_price_difference_l1257_125784

theorem promotional_price_difference
  (normal_price : ℝ)
  (months : ℕ)
  (issues_per_month : ℕ)
  (discount_per_issue : ℝ)
  (h1 : normal_price = 34)
  (h2 : months = 18)
  (h3 : issues_per_month = 2)
  (h4 : discount_per_issue = 0.25) : 
  normal_price - (months * issues_per_month * discount_per_issue) = 9 := 
by 
  sorry

end promotional_price_difference_l1257_125784


namespace raritet_meets_ferries_l1257_125773

theorem raritet_meets_ferries :
  (∀ (n : ℕ), ∃ (ferry_departure : Nat), ferry_departure = n ∧ ferry_departure + 8 = 8) →
  (∀ (m : ℕ), ∃ (raritet_departure : Nat), raritet_departure = m ∧ raritet_departure + 8 = 8) →
  ∃ (total_meetings : Nat), total_meetings = 17 := 
by
  sorry

end raritet_meets_ferries_l1257_125773


namespace distinct_values_of_c_l1257_125785

theorem distinct_values_of_c {c p q : ℂ} 
  (h_distinct : p ≠ q) 
  (h_eq : ∀ z : ℂ, (z - p) * (z - q) = (z - c * p) * (z - c * q)) :
  (∃ c_values : ℕ, c_values = 2) :=
sorry

end distinct_values_of_c_l1257_125785


namespace correct_operation_l1257_125733

variable (a b : ℝ)

theorem correct_operation (h1 : a^2 + a^3 ≠ a^5)
                          (h2 : (-a^2)^3 ≠ a^6)
                          (h3 : -2*a^3*b / (a*b) ≠ -2*a^2*b) :
                          a^2 * a^3 = a^5 :=
by sorry

end correct_operation_l1257_125733


namespace students_in_A_and_D_combined_l1257_125786

theorem students_in_A_and_D_combined (AB BC CD : ℕ) (hAB : AB = 83) (hBC : BC = 86) (hCD : CD = 88) : (AB + CD - BC = 85) :=
by
  sorry

end students_in_A_and_D_combined_l1257_125786


namespace prime_divides_expression_l1257_125729

theorem prime_divides_expression (p : ℕ) (hp : Nat.Prime p) : ∃ n : ℕ, p ∣ (2^n + 3^n + 6^n - 1) := 
by
  sorry

end prime_divides_expression_l1257_125729


namespace parabola_sum_is_neg_fourteen_l1257_125742

noncomputable def parabola_sum (a b c : ℝ) : ℝ := a + b + c

theorem parabola_sum_is_neg_fourteen :
  ∃ (a b c : ℝ), 
    (∀ x : ℝ, a * x^2 + b * x + c = -(x + 3)^2 + 2) ∧
    ((-1)^2 = a * (-1 + 3)^2 + 6) ∧ 
    ((-3)^2 = a * (-3 + 3)^2 + 2) ∧
    (parabola_sum a b c = -14) :=
sorry

end parabola_sum_is_neg_fourteen_l1257_125742


namespace solid_is_cone_l1257_125783

-- Define what it means for a solid to have a given view as an isosceles triangle or a circle.
structure Solid :=
(front_view : ℝ → ℝ → Prop)
(left_view : ℝ → ℝ → Prop)
(top_view : ℝ → ℝ → Prop)

-- Definition of isosceles triangle view
def isosceles_triangle (x y : ℝ) : Prop := 
  -- not specifying details of this relationship as a placeholder
  sorry

-- Definition of circle view with a center
def circle_with_center (x y : ℝ) : Prop := 
  -- not specifying details of this relationship as a placeholder
  sorry

-- Define the solid that satisfies the conditions in the problem
def specified_solid (s : Solid) : Prop :=
  (∀ x y, s.front_view x y → isosceles_triangle x y) ∧
  (∀ x y, s.left_view x y → isosceles_triangle x y) ∧
  (∀ x y, s.top_view x y → circle_with_center x y)

-- Given proof problem statement
theorem solid_is_cone (s : Solid) (h : specified_solid s) : 
  ∃ cone, cone = s :=
sorry

end solid_is_cone_l1257_125783


namespace temperature_difference_l1257_125717

theorem temperature_difference (T_high T_low : ℝ) (h1 : T_high = 8) (h2 : T_low = -2) : T_high - T_low = 10 :=
by
  sorry

end temperature_difference_l1257_125717


namespace probability_ratio_l1257_125790

-- Conditions definitions
def total_choices := Nat.choose 50 5
def p := 10 / total_choices
def q := (Nat.choose 10 2 * Nat.choose 5 2 * Nat.choose 5 3) / total_choices

-- Statement to prove
theorem probability_ratio : q / p = 450 := by
  sorry  -- proof is omitted

end probability_ratio_l1257_125790


namespace sum_of_first_ten_nicely_odd_numbers_is_775_l1257_125778

def is_nicely_odd (n : ℕ) : Prop :=
  (∃ p q : ℕ, Nat.Prime p ∧ Nat.Prime q ∧ p ≠ q ∧ (Odd p ∧ Odd q) ∧ n = p * q)
  ∨ (∃ p : ℕ, Nat.Prime p ∧ Odd p ∧ n = p ^ 3)

theorem sum_of_first_ten_nicely_odd_numbers_is_775 :
  let nicely_odd_nums := [15, 27, 21, 35, 125, 33, 77, 343, 55, 39]
  ∃ (nums : List ℕ), List.length nums = 10 ∧
  (∀ n ∈ nums, is_nicely_odd n) ∧ List.sum nums = 775 := by
  sorry

end sum_of_first_ten_nicely_odd_numbers_is_775_l1257_125778


namespace problem_statement_l1257_125751

theorem problem_statement (x : ℝ) (h : x^3 - 3 * x = 7) : x^7 + 27 * x^2 = 76 * x^2 + 270 * x + 483 :=
sorry

end problem_statement_l1257_125751


namespace geometric_series_sum_l1257_125794

theorem geometric_series_sum : 
  let a := 1
  let r := 2
  let n := 21
  a * ((r^n - 1) / (r - 1)) = 2097151 :=
by
  sorry

end geometric_series_sum_l1257_125794


namespace sin_sum_of_acute_l1257_125724

open Real

theorem sin_sum_of_acute (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2) :
  sin (α + β) ≤ sin α + sin β := 
by
  sorry

end sin_sum_of_acute_l1257_125724


namespace belindas_age_l1257_125756

theorem belindas_age (T B : ℕ) (h1 : T + B = 56) (h2 : B = 2 * T + 8) (h3 : T = 16) : B = 40 :=
by
  sorry

end belindas_age_l1257_125756


namespace problem_l1257_125741

theorem problem
  (x y : ℝ)
  (h₁ : x - 2 * y = -5)
  (h₂ : x * y = -2) :
  2 * x^2 * y - 4 * x * y^2 = 20 := 
by
  sorry

end problem_l1257_125741


namespace a1d1_a2d2_a3d3_eq_neg1_l1257_125706

theorem a1d1_a2d2_a3d3_eq_neg1 (a1 a2 a3 d1 d2 d3 : ℝ) (h : ∀ x : ℝ, 
  x^8 - x^6 + x^4 - x^2 + 1 = (x^2 + a1 * x + d1) * (x^2 + a2 * x + d2) * (x^2 + a3 * x + d3) * (x^2 + 1)) : 
  a1 * d1 + a2 * d2 + a3 * d3 = -1 := 
sorry

end a1d1_a2d2_a3d3_eq_neg1_l1257_125706


namespace chocolates_sold_l1257_125767

theorem chocolates_sold (C S : ℝ) (n : ℝ)
  (h1 : 65 * C = n * S)
  (h2 : S = 1.3 * C) :
  n = 50 :=
by
  sorry

end chocolates_sold_l1257_125767


namespace Alex_is_26_l1257_125736

-- Define the ages as integers
variable (Alex Jose Zack Inez : ℤ)

-- Conditions of the problem
variable (h1 : Alex = Jose + 6)
variable (h2 : Zack = Inez + 5)
variable (h3 : Inez = 18)
variable (h4 : Jose = Zack - 3)

-- Theorem we need to prove
theorem Alex_is_26 (h1: Alex = Jose + 6) (h2 : Zack = Inez + 5) (h3 : Inez = 18) (h4 : Jose = Zack - 3) : Alex = 26 :=
by
  sorry

end Alex_is_26_l1257_125736


namespace james_bike_ride_total_distance_l1257_125722

theorem james_bike_ride_total_distance 
  (d1 d2 d3 : ℝ)
  (H1 : d2 = 12)
  (H2 : d2 = 1.2 * d1)
  (H3 : d3 = 1.25 * d2) :
  d1 + d2 + d3 = 37 :=
by
  -- additional proof steps would go here
  sorry

end james_bike_ride_total_distance_l1257_125722


namespace greatest_k_for_factorial_div_l1257_125797

-- Definitions for conditions in the problem
def a : Nat := Nat.factorial 100
noncomputable def b (k : Nat) : Nat := 100^k

-- Statement to prove the greatest value of k for which b is a factor of a
theorem greatest_k_for_factorial_div (k : Nat) : 
  (∀ m : Nat, (m ≤ k → b m ∣ a) ↔ m ≤ 12) := 
by
  sorry

end greatest_k_for_factorial_div_l1257_125797


namespace girl_travel_distance_l1257_125747

def speed : ℝ := 6 -- meters per second
def time : ℕ := 16 -- seconds

def distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

theorem girl_travel_distance : distance speed time = 96 :=
by 
  unfold distance
  sorry

end girl_travel_distance_l1257_125747


namespace k_for_circle_radius_7_l1257_125728

theorem k_for_circle_radius_7 (k : ℝ) :
  (∃ x y : ℝ, x^2 + 8*x + y^2 + 4*y - k = 0) →
  (∃ x y : ℝ, (x + 4)^2 + (y + 2)^2 = 49) →
  k = 29 :=
by
  sorry

end k_for_circle_radius_7_l1257_125728


namespace complex_number_eq_l1257_125743

theorem complex_number_eq (a b : ℝ) (i : ℂ) (h1 : i * i = -1) (h2 : (a - 2 * i) * i = b - i) : a^2 + b^2 = 5 :=
sorry

end complex_number_eq_l1257_125743


namespace simplify_expression_l1257_125763

variables (a b : ℝ)

theorem simplify_expression : 
  (2 * a^2 - 3 * a * b + 8) - (-a * b - a^2 + 8) = 3 * a^2 - 2 * a * b :=
by sorry

-- Note:
-- ℝ denotes real numbers. Adjust types accordingly if using different numerical domains (e.g., ℚ, ℂ).

end simplify_expression_l1257_125763


namespace arctan_combination_l1257_125771

noncomputable def find_m : ℕ :=
  133

theorem arctan_combination :
  (Real.arctan (1/7) + Real.arctan (1/8) + Real.arctan (1/9) + Real.arctan (1/find_m)) = (Real.pi / 4) :=
by
  sorry

end arctan_combination_l1257_125771


namespace train_departure_time_l1257_125760

theorem train_departure_time 
(distance speed : ℕ) (arrival_time_chicago difference : ℕ) (arrival_time_new_york departure_time : ℕ) 
(h_dist : distance = 480) 
(h_speed : speed = 60)
(h_arrival_chicago : arrival_time_chicago = 17) 
(h_difference : difference = 1)
(h_arrival_new_york : arrival_time_new_york = arrival_time_chicago + difference) : 
  departure_time = arrival_time_new_york - distance / speed :=
by
  sorry

end train_departure_time_l1257_125760


namespace ceil_of_neg_frac_squared_l1257_125768

-- Define the negated fraction
def neg_frac : ℚ := -7 / 4

-- Define the squared value of the negated fraction
def squared_value : ℚ := neg_frac ^ 2

-- Define the ceiling function applied to the squared value
def ceil_squared_value : ℤ := Int.ceil squared_value

-- Prove that the ceiling of the squared value is 4
theorem ceil_of_neg_frac_squared : ceil_squared_value = 4 := 
by sorry

end ceil_of_neg_frac_squared_l1257_125768


namespace percentage_by_which_x_is_more_than_y_l1257_125793

variable {z : ℝ} 

-- Define x and y based on the given conditions
def x (z : ℝ) : ℝ := 0.78 * z
def y (z : ℝ) : ℝ := 0.60 * z

-- The main theorem we aim to prove
theorem percentage_by_which_x_is_more_than_y (z : ℝ) : x z = y z + 0.30 * y z := by
  sorry

end percentage_by_which_x_is_more_than_y_l1257_125793


namespace sad_girls_count_l1257_125749

-- Given definitions
def total_children : ℕ := 60
def happy_children : ℕ := 30
def sad_children : ℕ := 10
def neither_happy_nor_sad_children : ℕ := 20
def boys : ℕ := 22
def girls : ℕ := 38
def happy_boys : ℕ := 6
def boys_neither_happy_nor_sad : ℕ := 10

-- Intermediate definitions
def sad_boys : ℕ := boys - happy_boys - boys_neither_happy_nor_sad
def sad_girls : ℕ := sad_children - sad_boys

-- Theorem to prove that the number of sad girls is 4
theorem sad_girls_count : sad_girls = 4 := by
  sorry

end sad_girls_count_l1257_125749


namespace max_a_value_l1257_125720

theorem max_a_value : 
  (∀ (x : ℝ), (x - 1) * x - (a - 2) * (a + 1) ≥ 1) → a ≤ 3 / 2 :=
sorry

end max_a_value_l1257_125720


namespace c_finishes_work_in_18_days_l1257_125775

theorem c_finishes_work_in_18_days (A B C : ℝ) 
  (h1 : A = 1 / 12) 
  (h2 : B = 1 / 9) 
  (h3 : A + B + C = 1 / 4) : 
  1 / C = 18 := 
    sorry

end c_finishes_work_in_18_days_l1257_125775


namespace area_ratio_of_squares_l1257_125753

theorem area_ratio_of_squares (hA : ∃ sA : ℕ, 4 * sA = 16)
                             (hB : ∃ sB : ℕ, 4 * sB = 20)
                             (hC : ∃ sC : ℕ, 4 * sC = 40) :
  (∃ aB aC : ℕ, aB = sB * sB ∧ aC = sC * sC ∧ aB * 4 = aC) := by
  sorry

end area_ratio_of_squares_l1257_125753


namespace M_minus_N_l1257_125700

theorem M_minus_N (a b c d : ℕ) (h1 : a + b = 20) (h2 : a + c = 24) (h3 : a + d = 22) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d) : 
  let M := 2 * b + 26
  let N := 2 * 1 + 26
  (M - N) = 36 :=
by
  sorry

end M_minus_N_l1257_125700


namespace c_work_rate_l1257_125727

variable {W : ℝ} -- Denoting the work by W
variable {a_rate : ℝ} -- Work rate of a
variable {b_rate : ℝ} -- Work rate of b
variable {c_rate : ℝ} -- Work rate of c
variable {combined_rate : ℝ} -- Combined work rate of a, b, and c

theorem c_work_rate (W a_rate b_rate c_rate combined_rate : ℝ)
  (h1 : a_rate = W / 12)
  (h2 : b_rate = W / 24)
  (h3 : combined_rate = W / 4)
  (h4 : combined_rate = a_rate + b_rate + c_rate) :
  c_rate = W / 4.5 :=
by
  sorry

end c_work_rate_l1257_125727


namespace third_angle_in_triangle_sum_of_angles_in_triangle_l1257_125738

theorem third_angle_in_triangle (a b : ℝ) (h₁ : a = 50) (h₂ : b = 80) : 180 - a - b = 50 :=
by
  rw [h₁, h₂]
  norm_num

-- Adding this to demonstrate the constraint of the problem: Sum of angles in a triangle is 180°
theorem sum_of_angles_in_triangle (a b c : ℝ) (h₁: a + b + c = 180) : true :=
by
  trivial

end third_angle_in_triangle_sum_of_angles_in_triangle_l1257_125738


namespace minimum_passing_rate_l1257_125755

-- Define the conditions as hypotheses
variable (total_students : ℕ)
variable (correct_q1 : ℕ)
variable (correct_q2 : ℕ)
variable (correct_q3 : ℕ)
variable (correct_q4 : ℕ)
variable (correct_q5 : ℕ)
variable (pass_threshold : ℕ)

-- Assume all percentages are converted to actual student counts based on total_students
axiom students_answered_q1_correctly : correct_q1 = total_students * 81 / 100
axiom students_answered_q2_correctly : correct_q2 = total_students * 91 / 100
axiom students_answered_q3_correctly : correct_q3 = total_students * 85 / 100
axiom students_answered_q4_correctly : correct_q4 = total_students * 79 / 100
axiom students_answered_q5_correctly : correct_q5 = total_students * 74 / 100
axiom passing_criteria : pass_threshold = 3

-- Define the main theorem statement to be proven
theorem minimum_passing_rate (total_students : ℕ) :
  (total_students - (total_students * 19 / 100 + total_students * 9 / 100 + 
  total_students * 15 / 100 + total_students * 21 / 100 + 
  total_students * 26 / 100) / pass_threshold) / total_students * 100 ≥ 70 :=
  by sorry

end minimum_passing_rate_l1257_125755


namespace ceil_floor_difference_is_3_l1257_125765

noncomputable def ceil_floor_difference : ℤ :=
  Int.ceil ((14:ℚ) / 5 * (-31 / 3)) - Int.floor ((14 / 5) * Int.floor ((-31:ℚ) / 3))

theorem ceil_floor_difference_is_3 : ceil_floor_difference = 3 :=
  sorry

end ceil_floor_difference_is_3_l1257_125765


namespace plane_intercept_equation_l1257_125734

-- Define the conditions in Lean 4
variable (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)

-- State the main theorem
theorem plane_intercept_equation :
  ∃ (p : ℝ → ℝ → ℝ → ℝ), (∀ x y z, p x y z = x / a + y / b + z / c) :=
sorry

end plane_intercept_equation_l1257_125734


namespace polar_bear_trout_l1257_125701

/-
Question: How many buckets of trout does the polar bear eat daily?
Conditions:
  1. The polar bear eats some amount of trout and 0.4 bucket of salmon daily.
  2. The polar bear eats a total of 0.6 buckets of fish daily.
Answer: 0.2 buckets of trout daily.
-/

theorem polar_bear_trout (trout salmon total : ℝ) 
  (h1 : salmon = 0.4)
  (h2 : total = 0.6)
  (h3 : trout + salmon = total) :
  trout = 0.2 :=
by
  -- The proof will be provided here
  sorry

end polar_bear_trout_l1257_125701


namespace number_of_sides_of_polygon_l1257_125748

theorem number_of_sides_of_polygon (n : ℕ) (h : 3 * (n * (n - 3) / 2) - n = 21) : n = 6 :=
by sorry

end number_of_sides_of_polygon_l1257_125748


namespace complement_of_A_relative_to_I_l1257_125766

def I : Set ℤ := {-2, -1, 0, 1, 2}

def A : Set ℤ := {x : ℤ | x^2 < 3}

def complement_I_A : Set ℤ := {x ∈ I | x ∉ A}

theorem complement_of_A_relative_to_I :
  complement_I_A = {-2, 2} := by
  sorry

end complement_of_A_relative_to_I_l1257_125766


namespace Ryan_hours_learning_Spanish_is_4_l1257_125750

-- Definitions based on conditions
def hoursLearningChinese : ℕ := 5
def hoursLearningSpanish := ∃ x : ℕ, hoursLearningChinese = x + 1

-- Proof Statement
theorem Ryan_hours_learning_Spanish_is_4 : ∃ x : ℕ, hoursLearningSpanish ∧ x = 4 :=
by
  sorry

end Ryan_hours_learning_Spanish_is_4_l1257_125750


namespace innings_question_l1257_125719

theorem innings_question (n : ℕ) (runs_in_inning : ℕ) (avg_increase : ℕ) (new_avg : ℕ) 
  (h_runs_in_inning : runs_in_inning = 88) 
  (h_avg_increase : avg_increase = 3) 
  (h_new_avg : new_avg = 40)
  (h_eq : 37 * n + runs_in_inning = new_avg * (n + 1)): n + 1 = 17 :=
by
  -- Proof to be filled in here
  sorry

end innings_question_l1257_125719


namespace area_of_circle_l1257_125781

theorem area_of_circle (r : ℝ) : 
  (S = π * r^2) :=
sorry

end area_of_circle_l1257_125781


namespace meaningful_range_l1257_125795

theorem meaningful_range (x : ℝ) : (x < 4) ↔ (4 - x > 0) := 
by sorry

end meaningful_range_l1257_125795


namespace g_g_g_g_2_eq_16_l1257_125739

def g (x : ℕ) : ℕ :=
if x % 2 = 0 then x / 2 else 5 * x + 1

theorem g_g_g_g_2_eq_16 : g (g (g (g 2))) = 16 := by
  sorry

end g_g_g_g_2_eq_16_l1257_125739


namespace sequence_fifth_number_l1257_125789

theorem sequence_fifth_number : (5^2 - 1) = 24 :=
by {
  sorry
}

end sequence_fifth_number_l1257_125789


namespace muffin_count_l1257_125740

theorem muffin_count (doughnuts cookies muffins : ℕ) (h1 : doughnuts = 50) (h2 : cookies = (3 * doughnuts) / 5) (h3 : muffins = (1 * doughnuts) / 5) : muffins = 10 :=
by sorry

end muffin_count_l1257_125740


namespace prob_2_lt_X_lt_4_l1257_125757

noncomputable def normal_dist_p (μ σ : ℝ) (x : ℝ) : ℝ := sorry -- Assume this computes the CDF at x for a normal distribution

variable {X : ℝ → ℝ}
variable {σ : ℝ}

-- Condition: X follows a normal distribution with mean 3 and variance σ^2
axiom normal_distribution_X : ∀ x, X x = normal_dist_p 3 σ x

-- Condition: P(X ≤ 4) = 0.84
axiom prob_X_leq_4 : normal_dist_p 3 σ 4 = 0.84

-- Goal: Prove P(2 < X < 4) = 0.68
theorem prob_2_lt_X_lt_4 : normal_dist_p 3 σ 4 - normal_dist_p 3 σ 2 = 0.68 := by
  sorry

end prob_2_lt_X_lt_4_l1257_125757


namespace three_correct_deliveries_probability_l1257_125752

theorem three_correct_deliveries_probability (n : ℕ) (h1 : n = 5) :
  (∃ p : ℚ, p = 1/6 ∧ 
   (∃ choose3 : ℕ, choose3 = Nat.choose n 3) ∧ 
   (choose3 * 1/5 * 1/4 * 1/3 = p)) :=
by 
  sorry

end three_correct_deliveries_probability_l1257_125752


namespace chef_michel_total_pies_l1257_125799

theorem chef_michel_total_pies 
  (shepherd_pie_pieces : ℕ) 
  (chicken_pot_pie_pieces : ℕ)
  (shepherd_pie_customers : ℕ) 
  (chicken_pot_pie_customers : ℕ) 
  (h1 : shepherd_pie_pieces = 4)
  (h2 : chicken_pot_pie_pieces = 5)
  (h3 : shepherd_pie_customers = 52)
  (h4 : chicken_pot_pie_customers = 80) :
  (shepherd_pie_customers / shepherd_pie_pieces) +
  (chicken_pot_pie_customers / chicken_pot_pie_pieces) = 29 :=
by {
  sorry
}

end chef_michel_total_pies_l1257_125799


namespace prove_A_plus_B_l1257_125710

variable (A B : ℝ)

theorem prove_A_plus_B (h : ∀ x : ℝ, x ≠ 2 → (A / (x - 2) + B * (x + 3) = (-5 * x^2 + 20 * x + 34) / (x - 2))) : A + B = 9 := by
  sorry

end prove_A_plus_B_l1257_125710


namespace cos_alpha_beta_value_l1257_125788

noncomputable def cos_alpha_beta (α β : ℝ) : ℝ :=
  Real.cos (α + β)

theorem cos_alpha_beta_value (α β : ℝ)
  (h1 : Real.cos α - Real.cos β = -3/5)
  (h2 : Real.sin α + Real.sin β = 7/4) :
  cos_alpha_beta α β = -569/800 :=
by
  sorry

end cos_alpha_beta_value_l1257_125788


namespace hash_four_times_l1257_125713

noncomputable def hash (N : ℝ) : ℝ := 0.6 * N + 2

theorem hash_four_times (N : ℝ) : hash (hash (hash (hash N))) = 11.8688 :=
  sorry

end hash_four_times_l1257_125713


namespace parabola_point_dot_product_eq_neg4_l1257_125715

-- Definition of the parabola
def is_parabola_point (A : ℝ × ℝ) : Prop :=
  A.2 ^ 2 = 4 * A.1

-- Definition of the focus of the parabola y^2 = 4x
def focus : ℝ × ℝ := (1, 0)

-- Dot product of two vectors
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

-- Coordinates of origin
def origin : ℝ × ℝ := (0, 0)

-- Vector from origin to point A
def vector_OA (A : ℝ × ℝ) : ℝ × ℝ :=
  (A.1, A.2)

-- Vector from point A to the focus
def vector_AF (A : ℝ × ℝ) : ℝ × ℝ :=
  (focus.1 - A.1, focus.2 - A.2)

-- Theorem statement
theorem parabola_point_dot_product_eq_neg4 (A : ℝ × ℝ) 
  (hA : is_parabola_point A) 
  (h_dot : dot_product (vector_OA A) (vector_AF A) = -4) :
  A = (1, 2) ∨ A = (1, -2) :=
sorry

end parabola_point_dot_product_eq_neg4_l1257_125715


namespace Nicole_fish_tanks_l1257_125745

-- Definition to express the conditions
def first_tank_water := 8 -- gallons
def second_tank_difference := 2 -- fewer gallons than first tanks
def num_first_tanks := 2
def num_second_tanks := 2
def total_water_four_weeks := 112 -- gallons
def weeks := 4

-- Calculate the total water per week
def water_per_week := (num_first_tanks * first_tank_water) + (num_second_tanks * (first_tank_water - second_tank_difference))

-- Calculate the total number of tanks
def total_tanks := num_first_tanks + num_second_tanks

-- Proof statement
theorem Nicole_fish_tanks : total_water_four_weeks / water_per_week = weeks → total_tanks = 4 := by
  -- Proof goes here
  sorry

end Nicole_fish_tanks_l1257_125745


namespace saved_percentage_this_year_l1257_125759

variable (S : ℝ) -- Annual salary last year

-- Conditions
def saved_last_year := 0.06 * S
def salary_this_year := 1.20 * S
def saved_this_year := saved_last_year

-- The goal is to prove that the percentage saved this year is 5%
theorem saved_percentage_this_year :
  (saved_this_year / salary_this_year) * 100 = 5 :=
by sorry

end saved_percentage_this_year_l1257_125759


namespace range_of_a_l1257_125780

theorem range_of_a (a : ℝ) :
  (abs (15 - 3 * a) / 5 ≤ 3) → (0 ≤ a ∧ a ≤ 10) :=
by
  intro h
  sorry

end range_of_a_l1257_125780


namespace garden_dimensions_l1257_125731

theorem garden_dimensions (l b : ℝ) (walkway_width total_area perimeter : ℝ) 
  (h1 : l = 3 * b)
  (h2 : perimeter = 2 * l + 2 * b)
  (h3 : walkway_width = 1)
  (h4 : total_area = (l + 2 * walkway_width) * (b + 2 * walkway_width))
  (h5 : perimeter = 40)
  (h6 : total_area = 120) : 
  l = 15 ∧ b = 5 ∧ total_area - l * b = 45 :=  
  by
  sorry

end garden_dimensions_l1257_125731


namespace cube_rolling_impossible_l1257_125746

-- Definitions
def paintedCube : Type := sorry   -- Define a painted black-and-white cube.
def chessboard : Type := sorry    -- Define the chessboard.
def roll (c : paintedCube) (b : chessboard) : Prop := sorry   -- Define the rolling over the board visiting each square exactly once.
def matchColors (c : paintedCube) (b : chessboard) : Prop := sorry   -- Define the condition that colors match on contact.

-- Theorem
theorem cube_rolling_impossible (c : paintedCube) (b : chessboard)
  (h1 : roll c b) : ¬ matchColors c b := sorry

end cube_rolling_impossible_l1257_125746


namespace initial_number_of_girls_l1257_125792

theorem initial_number_of_girls (n : ℕ) (A : ℝ) 
  (h1 : (n + 1) * (A + 3) - 70 = n * A + 94) :
  n = 8 :=
by {
  sorry
}

end initial_number_of_girls_l1257_125792


namespace max_students_l1257_125770

-- Define the constants for pens and pencils
def pens : ℕ := 1802
def pencils : ℕ := 1203

-- State that the GCD of pens and pencils is 1
theorem max_students : Nat.gcd pens pencils = 1 :=
by sorry

end max_students_l1257_125770


namespace largest_possible_value_b_l1257_125787

theorem largest_possible_value_b : 
  ∃ b : ℚ, (3 * b + 7) * (b - 2) = 4 * b ∧ b = 40 / 15 := 
by 
  sorry

end largest_possible_value_b_l1257_125787


namespace sqrt_mul_sqrt_l1257_125711

theorem sqrt_mul_sqrt (h1 : Real.sqrt 25 = 5) : Real.sqrt (49 * Real.sqrt 25) = 7 * Real.sqrt 5 :=
by
  sorry

end sqrt_mul_sqrt_l1257_125711


namespace exists_fixed_point_sequence_l1257_125796

theorem exists_fixed_point_sequence (N : ℕ) (hN : 0 < N) (a : ℕ → ℕ)
  (ha_conditions : ∀ i < N, a i % 2^(N+1) ≠ 0) :
  ∃ M, ∀ n ≥ M, a n = a M :=
sorry

end exists_fixed_point_sequence_l1257_125796


namespace jackson_star_fish_count_l1257_125725

def total_starfish_per_spiral_shell (hermit_crabs : ℕ) (shells_per_crab : ℕ) (total_souvenirs : ℕ) : ℕ :=
  (total_souvenirs - (hermit_crabs + hermit_crabs * shells_per_crab)) / (hermit_crabs * shells_per_crab)

theorem jackson_star_fish_count :
  total_starfish_per_spiral_shell 45 3 450 = 2 :=
by
  -- The proof will be filled in here
  sorry

end jackson_star_fish_count_l1257_125725


namespace perimeter_of_square_l1257_125712

-- Defining the context and proving the equivalence.
theorem perimeter_of_square (x y : ℕ) (h : Nat.gcd x y = 3) (area : ℕ) :
  let lcm_xy := Nat.lcm x y
  let side_length := Real.sqrt (20 * lcm_xy)
  let perimeter := 4 * side_length
  perimeter = 24 * Real.sqrt 5 :=
by
  let lcm_xy := Nat.lcm x y
  let side_length := Real.sqrt (20 * lcm_xy)
  let perimeter := 4 * side_length
  sorry

end perimeter_of_square_l1257_125712


namespace value_of_k_for_square_of_binomial_l1257_125721

theorem value_of_k_for_square_of_binomial (a k : ℝ) : (x : ℝ) → x^2 - 14 * x + k = (x - a)^2 → k = 49 :=
by
  intro x h
  sorry

end value_of_k_for_square_of_binomial_l1257_125721


namespace flour_ratio_correct_l1257_125758

-- Definitions based on conditions
def initial_sugar : ℕ := 13
def initial_flour : ℕ := 25
def initial_baking_soda : ℕ := 35
def initial_cocoa_powder : ℕ := 60

def added_sugar : ℕ := 12
def added_flour : ℕ := 8
def added_cocoa_powder : ℕ := 15

-- Calculate remaining ingredients
def remaining_flour : ℕ := initial_flour - added_flour
def remaining_sugar : ℕ := initial_sugar - added_sugar
def remaining_cocoa_powder : ℕ := initial_cocoa_powder - added_cocoa_powder

-- Calculate ratio
def total_remaining_sugar_and_cocoa : ℕ := remaining_sugar + remaining_cocoa_powder
def flour_to_sugar_cocoa_ratio : ℕ × ℕ := (remaining_flour, total_remaining_sugar_and_cocoa)

-- Proposition stating the desired ratio
theorem flour_ratio_correct : flour_to_sugar_cocoa_ratio = (17, 46) := by
  sorry

end flour_ratio_correct_l1257_125758


namespace avg_of_first_5_multiples_of_5_l1257_125705

theorem avg_of_first_5_multiples_of_5 : (5 + 10 + 15 + 20 + 25) / 5 = 15 := 
by {
  sorry
}

end avg_of_first_5_multiples_of_5_l1257_125705


namespace arithmetic_sequence_problem_l1257_125704

variable (a : ℕ → ℝ) (d : ℝ) (m : ℕ)

noncomputable def a_seq := ∀ n, a n = a 1 + (n - 1) * d

theorem arithmetic_sequence_problem
  (h1 : a 1 = 0)
  (h2 : d ≠ 0)
  (h3 : a m = a 1 + a 2 + a 3 + a 4 + a 5) :
  m = 11 :=
sorry

end arithmetic_sequence_problem_l1257_125704


namespace percentage_broken_in_second_set_l1257_125777

-- Define the given conditions
def first_set_total : ℕ := 50
def first_set_broken_percent : ℚ := 0.10
def second_set_total : ℕ := 60
def total_broken : ℕ := 17

-- The proof problem statement
theorem percentage_broken_in_second_set :
  let first_set_broken := first_set_broken_percent * first_set_total
  let second_set_broken := total_broken - first_set_broken
  (second_set_broken / second_set_total) * 100 = 20 := 
sorry

end percentage_broken_in_second_set_l1257_125777


namespace find_n_in_arithmetic_sequence_l1257_125702

noncomputable def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n, a (n + 2) = a (n + 1) + d

theorem find_n_in_arithmetic_sequence (x : ℝ) (n : ℕ) (b : ℕ → ℝ)
  (h1 : b 1 = Real.exp x) 
  (h2 : b 2 = x) 
  (h3 : is_arithmetic_sequence b) : 
  b n = 1 + Real.exp x ↔ n = (1 + x) / (x - Real.exp x) :=
sorry

end find_n_in_arithmetic_sequence_l1257_125702


namespace intersection_empty_l1257_125776

open Set

-- Definition of set A
def A : Set (ℝ × ℝ) := { p | ∃ x : ℝ, p = (x, 2 * x + 3) }

-- Definition of set B
def B : Set (ℝ × ℝ) := { p | ∃ x : ℝ, p = (x, 4 * x + 1) }

-- The proof problem statement in Lean
theorem intersection_empty : A ∩ B = ∅ := sorry

end intersection_empty_l1257_125776


namespace percentage_failed_in_english_l1257_125716

theorem percentage_failed_in_english (total_students : ℕ) (hindi_failed : ℕ) (both_failed : ℕ) (both_passed : ℕ) 
  (H1 : hindi_failed = total_students * 25 / 100)
  (H2 : both_failed = total_students * 25 / 100)
  (H3 : both_passed = total_students * 50 / 100)
  : (total_students * 50 / 100) = (total_students * 75 / 100) + (both_failed) - both_passed
:= sorry

end percentage_failed_in_english_l1257_125716


namespace trinomial_identity_l1257_125730

theorem trinomial_identity :
  let a := 23
  let b := 15
  let c := 7
  (a + b + c)^2 - (a^2 + b^2 + c^2) = 1222 :=
by
  let a := 23
  let b := 15
  let c := 7
  sorry

end trinomial_identity_l1257_125730


namespace fx_greater_than_2_l1257_125772

noncomputable def f (x : ℝ) : ℝ := Real.exp x - Real.log x

theorem fx_greater_than_2 :
  ∀ x : ℝ, x > 0 → f x > 2 :=
by {
  sorry
}

end fx_greater_than_2_l1257_125772
