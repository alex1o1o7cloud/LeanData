import Mathlib

namespace NUMINAMATH_GPT_sector_angle_measure_l888_88871

-- Define the variables
variables (r α : ℝ)

-- Define the conditions
def perimeter_condition := (2 * r + r * α = 4)
def area_condition := (1 / 2 * α * r^2 = 1)

-- State the theorem
theorem sector_angle_measure (h1 : perimeter_condition r α) (h2 : area_condition r α) : α = 2 :=
sorry

end NUMINAMATH_GPT_sector_angle_measure_l888_88871


namespace NUMINAMATH_GPT_triangle_altitude_length_l888_88884

-- Define the problem
theorem triangle_altitude_length (l w h : ℝ) (hl : l = 2 * w) 
  (h_triangle_area : 0.5 * l * h = 0.5 * (l * w)) : h = w := 
by 
  -- Use the provided conditions and the equation setup to continue the proof
  sorry

end NUMINAMATH_GPT_triangle_altitude_length_l888_88884


namespace NUMINAMATH_GPT_binomial_coeff_divisibility_l888_88874

theorem binomial_coeff_divisibility (n k : ℕ) (hn : 0 < n) (hk : 0 < k) : n ∣ (Nat.choose n k) * Nat.gcd n k :=
sorry

end NUMINAMATH_GPT_binomial_coeff_divisibility_l888_88874


namespace NUMINAMATH_GPT_three_monotonic_intervals_l888_88888

open Real

noncomputable def f (b : ℝ) (x : ℝ) : ℝ := - (4 / 3) * x ^ 3 + (b - 1) * x

noncomputable def f' (b : ℝ) (x : ℝ) : ℝ := -4 * x ^ 2 + (b - 1)

theorem three_monotonic_intervals (b : ℝ) (h : (b - 1) > 0) : b > 1 := 
by
  have discriminant : 16 * (b - 1) > 0 := sorry
  sorry

end NUMINAMATH_GPT_three_monotonic_intervals_l888_88888


namespace NUMINAMATH_GPT_range_of_m_l888_88896

open Real

noncomputable def x (y : ℝ) : ℝ := 2 / (1 - 1 / y)

theorem range_of_m (y : ℝ) (m : ℝ) (h1 : y > 0) (h2 : 1 - 1 / y > 0) (h3 : -4 < m) (h4 : m < 2) : 
  x y + 2 * y > m^2 + 2 * m := 
by 
  have hx_pos : x y > 0 := sorry
  have hxy_eq : 2 / x y + 1 / y = 1 := sorry
  have hxy_ge : x y + 2 * y ≥ 8 := sorry
  have h_m_le : 8 > m^2 + 2 * m := sorry
  exact sorry

end NUMINAMATH_GPT_range_of_m_l888_88896


namespace NUMINAMATH_GPT_triangle_side_lengths_l888_88819

noncomputable def radius_inscribed_circle := 4/3
def sum_of_heights := 13

theorem triangle_side_lengths :
  ∃ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  ∃ (h_a h_b h_c : ℕ), h_a ≠ h_b ∧ h_b ≠ h_c ∧ h_a ≠ h_c ∧
  h_a + h_b + h_c = sum_of_heights ∧
  r * (a + b + c) = 8 ∧ -- (since Δ = r * s, where s = (a + b + c)/2)
  1 / 2 * a * h_a = 1 / 2 * b * h_b ∧
  1 / 2 * b * h_b = 1 / 2 * c * h_c ∧
  a = 6 ∧ b = 4 ∧ c = 3 :=
sorry

end NUMINAMATH_GPT_triangle_side_lengths_l888_88819


namespace NUMINAMATH_GPT_overlapping_area_of_thirty_sixty_ninety_triangles_l888_88808

-- Definitions for 30-60-90 triangle and the overlapping region
def thirty_sixty_ninety_triangle (hypotenuse : ℝ) := 
  (hypotenuse > 0) ∧ 
  (exists (short_leg long_leg : ℝ), short_leg = hypotenuse / 2 ∧ long_leg = short_leg * (Real.sqrt 3))

-- Area of a parallelogram given base and height
def parallelogram_area (base height : ℝ) : ℝ :=
  base * height

theorem overlapping_area_of_thirty_sixty_ninety_triangles :
  ∀ (hypotenuse : ℝ), thirty_sixty_ninety_triangle hypotenuse →
  hypotenuse = 10 →
  (∃ (base height : ℝ), base = height ∧ base * height = parallelogram_area (5 * Real.sqrt 3) (5 * Real.sqrt 3)) →
  parallelogram_area (5 * Real.sqrt 3) (5 * Real.sqrt 3) = 75 :=
by
  sorry

end NUMINAMATH_GPT_overlapping_area_of_thirty_sixty_ninety_triangles_l888_88808


namespace NUMINAMATH_GPT_inequality_solution_I_inequality_solution_II_l888_88848

noncomputable def f (x a : ℝ) : ℝ := |2 * x - a| - |x + 1|

theorem inequality_solution_I (x : ℝ) : f x 1 > 2 ↔ x < -2 / 3 ∨ x > 4 :=
sorry 

noncomputable def g (x a : ℝ) : ℝ := f x a + |x + 1| + x

theorem inequality_solution_II (a : ℝ) : (∀ x, g x a > a ^ 2 - 1 / 2) ↔ (-1 / 2 < a ∧ a < 1) :=
sorry

end NUMINAMATH_GPT_inequality_solution_I_inequality_solution_II_l888_88848


namespace NUMINAMATH_GPT_set_union_example_l888_88814

variable (A B : Set ℝ)

theorem set_union_example :
  A = {x | -2 < x ∧ x ≤ 1} ∧ B = {x | -1 ≤ x ∧ x < 2} →
  (A ∪ B) = {x | -2 < x ∧ x < 2} := 
by
  sorry

end NUMINAMATH_GPT_set_union_example_l888_88814


namespace NUMINAMATH_GPT_problem_quadratic_inequality_l888_88854

theorem problem_quadratic_inequality
  (a : ℝ) (b : ℝ) (c : ℝ)
  (h1 : 0 < a)
  (h2 : a ≤ 4/9)
  (h3 : b = -a)
  (h4 : c = -2*a + 1)
  (h5 : ∀ x : ℝ, -1 ≤ x ∧ x ≤ 2 → 0 ≤ a*x^2 + b*x + c ∧ a*x^2 + b*x + c ≤ 1) :
  3*a + 2*b + c ≠ 1/3 ∧ 3*a + 2*b + c ≠ 5/4 :=
by
  sorry

end NUMINAMATH_GPT_problem_quadratic_inequality_l888_88854


namespace NUMINAMATH_GPT_monochromatic_regions_lower_bound_l888_88807

theorem monochromatic_regions_lower_bound (n : ℕ) (h_n_ge_2 : n ≥ 2) :
  ∀ (blue_lines red_lines : ℕ) (conditions :
    blue_lines = 2 * n ∧ red_lines = n ∧ 
    (∀ (i j k l : ℕ), i ≠ j ∧ j ≠ k ∧ k ≠ l ∧ l ≠ i → 
      (blue_lines = 2 * n ∧ red_lines = n))) 
  , ∃ (monochromatic_regions : ℕ), 
      monochromatic_regions ≥ (n - 1) * (n - 2) / 2 :=
sorry

end NUMINAMATH_GPT_monochromatic_regions_lower_bound_l888_88807


namespace NUMINAMATH_GPT_seating_arrangement_correct_l888_88849

noncomputable def seatingArrangements (committee : Fin 10) : Nat :=
  Nat.factorial 9

theorem seating_arrangement_correct :
  seatingArrangements committee = 362880 :=
by sorry

end NUMINAMATH_GPT_seating_arrangement_correct_l888_88849


namespace NUMINAMATH_GPT_three_digit_number_property_l888_88826

theorem three_digit_number_property :
  (∃ a b c : ℕ, 100 ≤ 100 * a + 10 * b + c ∧ 100 * a + 10 * b + c ≤ 999 ∧ 100 * a + 10 * b + c = (a + b + c)^3) ↔
  (∃ a b c : ℕ, a = 5 ∧ b = 1 ∧ c = 2 ∧ 100 * a + 10 * b + c = 512) := sorry

end NUMINAMATH_GPT_three_digit_number_property_l888_88826


namespace NUMINAMATH_GPT_Jhon_payment_per_day_l888_88860

theorem Jhon_payment_per_day
  (total_days : ℕ)
  (present_days : ℕ)
  (absent_pay : ℝ)
  (total_pay : ℝ)
  (Jhon_present_days : total_days = 60)
  (Jhon_presence : present_days = 35)
  (Jhon_absent_payment : absent_pay = 3.0)
  (Jhon_total_payment : total_pay = 170) :
  ∃ (P : ℝ), 
    P = 2.71 ∧ 
    total_pay = (present_days * P + (total_days - present_days) * absent_pay) := 
sorry

end NUMINAMATH_GPT_Jhon_payment_per_day_l888_88860


namespace NUMINAMATH_GPT_membership_percentage_change_l888_88868

theorem membership_percentage_change :
  let initial_membership := 100.0
  let first_fall_membership := initial_membership * 1.04
  let first_spring_membership := first_fall_membership * 0.95
  let second_fall_membership := first_spring_membership * 1.07
  let second_spring_membership := second_fall_membership * 0.97
  let third_fall_membership := second_spring_membership * 1.05
  let third_spring_membership := third_fall_membership * 0.81
  let final_membership := third_spring_membership
  let total_percentage_change := ((final_membership - initial_membership) / initial_membership) * 100.0
  total_percentage_change = -12.79 :=
by
  sorry

end NUMINAMATH_GPT_membership_percentage_change_l888_88868


namespace NUMINAMATH_GPT_num_toys_purchased_min_selling_price_l888_88820

variable (x m : ℕ)

-- Given conditions
axiom cond1 : 1500 / x + 5 = 3500 / (2 * x)
axiom cond2 : 150 * m - 5000 >= 1150

-- Required proof
theorem num_toys_purchased : x = 50 :=
by
  sorry

theorem min_selling_price : m >= 41 :=
by
  sorry

end NUMINAMATH_GPT_num_toys_purchased_min_selling_price_l888_88820


namespace NUMINAMATH_GPT_incircle_radius_of_right_triangle_l888_88898

noncomputable def radius_of_incircle (a b c : ℝ) : ℝ := (a + b - c) / 2

theorem incircle_radius_of_right_triangle
  (a : ℝ) (b_proj_hypotenuse : ℝ) (r : ℝ) :
  a = 15 ∧ b_proj_hypotenuse = 16 ∧ r = 5 :=
by
  sorry

end NUMINAMATH_GPT_incircle_radius_of_right_triangle_l888_88898


namespace NUMINAMATH_GPT_optimal_pricing_for_max_profit_l888_88839

noncomputable def sales_profit (x : ℝ) : ℝ :=
  -5 * x^3 + 45 * x^2 - 75 * x + 675

theorem optimal_pricing_for_max_profit :
  ∃ x : ℝ, 0 ≤ x ∧ x < 9 ∧ ∀ y : ℝ, 0 ≤ y ∧ y < 9 → sales_profit y ≤ sales_profit 5 ∧ (14 - 5 = 9) :=
by
  sorry

end NUMINAMATH_GPT_optimal_pricing_for_max_profit_l888_88839


namespace NUMINAMATH_GPT_function_periodicity_l888_88841

theorem function_periodicity
  (f : ℝ → ℝ)
  (H_odd : ∀ x, f (-x) = -f x)
  (H_even_shift : ∀ x, f (x + 2) = f (-x + 2))
  (H_val_neg1 : f (-1) = -1)
  : f 2017 + f 2016 = 1 := 
sorry

end NUMINAMATH_GPT_function_periodicity_l888_88841


namespace NUMINAMATH_GPT_parabola_equation_l888_88851

theorem parabola_equation (p : ℝ) (hp : 0 < p) (F : ℝ × ℝ) (Q : ℝ × ℝ) (PQ QF : ℝ)
  (hPQ : PQ = 8 / p) (hQF : QF = 8 / p + p / 2) (hDist : QF = 5 / 4 * PQ) : 
  ∃ x, y^2 = 4 * x :=
by
  sorry

end NUMINAMATH_GPT_parabola_equation_l888_88851


namespace NUMINAMATH_GPT_periodic_odd_fn_calc_l888_88864

theorem periodic_odd_fn_calc :
  ∀ (f : ℝ → ℝ),
  (∀ x, f (x + 2) = f x) ∧ (∀ x, f (-x) = -f x) ∧ (∀ x, 0 < x ∧ x < 1 → f x = 4^x) →
  f (-5 / 2) + f 2 = -2 :=
by
  intros f h
  sorry

end NUMINAMATH_GPT_periodic_odd_fn_calc_l888_88864


namespace NUMINAMATH_GPT_part1_part2_l888_88845

theorem part1 (a : ℝ) (x : ℝ) (h : a > 0) :
  (|x + 1/a| + |x - a + 1|) ≥ 1 :=
sorry

theorem part2 (a : ℝ) (h1 : a > 0) (h2 : |3 + 1/a| + |3 - a + 1| < 11/2) :
  2 < a ∧ a < (13 + 3 * Real.sqrt 17) / 4 :=
sorry

end NUMINAMATH_GPT_part1_part2_l888_88845


namespace NUMINAMATH_GPT_taxi_ride_cost_l888_88811

-- Definitions given in the conditions
def base_fare : ℝ := 2.00
def cost_per_mile : ℝ := 0.30
def distance_traveled : ℝ := 10

-- The theorem we need to prove
theorem taxi_ride_cost : base_fare + (cost_per_mile * distance_traveled) = 5.00 :=
by
  sorry

end NUMINAMATH_GPT_taxi_ride_cost_l888_88811


namespace NUMINAMATH_GPT_simplify_expression_l888_88824

theorem simplify_expression : 4 * Real.sqrt (1 / 2) + 3 * Real.sqrt (1 / 3) - Real.sqrt 8 = Real.sqrt 3 := 
by 
  sorry

end NUMINAMATH_GPT_simplify_expression_l888_88824


namespace NUMINAMATH_GPT_white_surface_area_fraction_l888_88830

theorem white_surface_area_fraction :
  let larger_cube_edge := 4
  let smaller_cube_edge := 1
  let total_smaller_cubes := 64
  let white_cubes := 48
  let black_cubes := 16
  let total_faces := 6
  let black_cubes_per_face := 2
  let surface_area := total_faces * larger_cube_edge^2
  let black_faces_exposed := total_faces * black_cubes_per_face
  let white_faces_exposed := surface_area - black_faces_exposed
  (white_faces_exposed / surface_area) = (7 / 8) :=
by
  let larger_cube_edge := 4
  let smaller_cube_edge := 1
  let total_smaller_cubes := 64
  let white_cubes := 48
  let black_cubes := 16
  let total_faces := 6
  let black_cubes_per_face := 2
  let surface_area := total_faces * larger_cube_edge^2
  let black_faces_exposed := total_faces * black_cubes_per_face
  let white_faces_exposed := surface_area - black_faces_exposed
  have h_white_fraction : (white_faces_exposed / surface_area) = (7 / 8) := sorry
  exact h_white_fraction

end NUMINAMATH_GPT_white_surface_area_fraction_l888_88830


namespace NUMINAMATH_GPT_constant_speed_total_distance_l888_88827

def travel_time : ℝ := 5.5
def distance_per_hour : ℝ := 100
def speed := distance_per_hour

theorem constant_speed : ∀ t : ℝ, (1 ≤ t) ∧ (t ≤ travel_time) → speed = distance_per_hour := 
by sorry

theorem total_distance : speed * travel_time = 550 :=
by sorry

end NUMINAMATH_GPT_constant_speed_total_distance_l888_88827


namespace NUMINAMATH_GPT_part1_part2_l888_88870

noncomputable def f (x : ℝ) : ℝ := |x - 1| - 1
noncomputable def g (x : ℝ) : ℝ := -|x + 1| - 4

theorem part1 (x : ℝ) : f x ≤ 1 ↔ -1 ≤ x ∧ x ≤ 3 :=
by
  sorry

theorem part2 (m : ℝ) : (∀ x : ℝ, f x - g x ≥ m + 1) ↔ m ≤ 4 :=
by
  sorry

end NUMINAMATH_GPT_part1_part2_l888_88870


namespace NUMINAMATH_GPT_actual_distance_between_mountains_l888_88803

theorem actual_distance_between_mountains (D_map : ℝ) (d_map_ram : ℝ) (d_real_ram : ℝ)
  (hD_map : D_map = 312) (hd_map_ram : d_map_ram = 25) (hd_real_ram : d_real_ram = 10.897435897435898) :
  D_map / d_map_ram * d_real_ram = 136 :=
by
  -- Theorem statement is proven based on the given conditions.
  sorry

end NUMINAMATH_GPT_actual_distance_between_mountains_l888_88803


namespace NUMINAMATH_GPT_length_of_tunnel_l888_88897

theorem length_of_tunnel (time : ℝ) (speed : ℝ) (train_length : ℝ) (total_distance : ℝ) (tunnel_length : ℝ) 
  (h1 : time = 30) (h2 : speed = 100 / 3) (h3 : train_length = 400) (h4 : total_distance = speed * time) 
  (h5 : tunnel_length = total_distance - train_length) : 
  tunnel_length = 600 :=
by
  sorry

end NUMINAMATH_GPT_length_of_tunnel_l888_88897


namespace NUMINAMATH_GPT_bret_total_spend_l888_88887

/-- Bret and his team are working late along with another team of 4 co-workers.
He decides to order dinner for everyone. -/

def team_A : ℕ := 4 -- Bret’s team
def team_B : ℕ := 4 -- Other team

def main_meal_cost : ℕ := 12
def team_A_appetizers_cost : ℕ := 2 * 6  -- Two appetizers at $6 each
def team_B_appetizers_cost : ℕ := 3 * 8  -- Three appetizers at $8 each
def sharing_plates_cost : ℕ := 4 * 10    -- Four sharing plates at $10 each

def tip_percentage : ℝ := 0.20           -- Tip is 20%
def rush_order_fee : ℕ := 5              -- Rush order fee is $5
def sales_tax : ℝ := 0.07                -- Local sales tax is 7%

def total_cost_without_tip_and_tax : ℕ :=
  team_A * main_meal_cost + team_B * main_meal_cost + team_A_appetizers_cost +
  team_B_appetizers_cost + sharing_plates_cost

def total_cost_with_tip : ℝ :=
  total_cost_without_tip_and_tax + 
  (tip_percentage * total_cost_without_tip_and_tax)

def total_cost_before_tax : ℝ :=
  total_cost_with_tip + rush_order_fee

def final_total_cost : ℝ :=
  total_cost_before_tax + (sales_tax * total_cost_with_tip)


theorem bret_total_spend : final_total_cost = 225.85 := by
  sorry

end NUMINAMATH_GPT_bret_total_spend_l888_88887


namespace NUMINAMATH_GPT_circle_radius_l888_88885

noncomputable def circle_problem (rD rE : ℝ) (m n : ℝ) :=
  rD = 2 * rE ∧
  rD = (Real.sqrt m) - n ∧
  m ≥ 0 ∧ n ≥ 0

theorem circle_radius (rE rD : ℝ) (m n : ℝ) (h : circle_problem rD rE m n) :
  m + n = 5.76 :=
by
  sorry

end NUMINAMATH_GPT_circle_radius_l888_88885


namespace NUMINAMATH_GPT_width_of_each_glass_pane_l888_88840

noncomputable def width_of_pane (num_panes : ℕ) (total_area : ℝ) (length_of_pane : ℝ) : ℝ :=
  total_area / num_panes / length_of_pane

theorem width_of_each_glass_pane :
  width_of_pane 8 768 12 = 8 := by
  sorry

end NUMINAMATH_GPT_width_of_each_glass_pane_l888_88840


namespace NUMINAMATH_GPT_minimum_fencing_l888_88881

variable (a b z : ℝ)

def area_condition : Prop := a * b = 50
def length_condition : Prop := a + 2 * b = z

theorem minimum_fencing (h1 : area_condition a b) (h2 : length_condition a b z) : z ≥ 20 := 
  sorry

end NUMINAMATH_GPT_minimum_fencing_l888_88881


namespace NUMINAMATH_GPT_length_of_FD_l888_88817

/-- Square ABCD with side length 8 cm, corner C is folded to point E on AD such that AE = 2 cm and ED = 6 cm. Find the length of FD. -/
theorem length_of_FD 
  (A B C D E F G : Type)
  (square_length : Float)
  (AD_length AE_length ED_length : Float)
  (hyp1 : square_length = 8)
  (hyp2 : AE_length = 2)
  (hyp3 : ED_length = 6)
  (hyp4 : AD_length = AE_length + ED_length)
  (FD_length : Float) :
  FD_length = 7 / 4 := 
  by 
  sorry

end NUMINAMATH_GPT_length_of_FD_l888_88817


namespace NUMINAMATH_GPT_boat_stream_speeds_l888_88877

variable (x y : ℝ)

theorem boat_stream_speeds (h1 : 20 + x ≠ 0) (h2 : 40 - y ≠ 0) :
  380 = 7 * x + 13 * y ↔ 
  26 * (40 - y) = 14 * (20 + x) :=
by { sorry }

end NUMINAMATH_GPT_boat_stream_speeds_l888_88877


namespace NUMINAMATH_GPT_sequence_factorial_l888_88876

theorem sequence_factorial (a : ℕ → ℕ) (h₁ : a 1 = 1) (h₂ : ∀ n : ℕ, n > 0 → a n = n * a (n - 1)) :
  ∀ n : ℕ, a n = Nat.factorial n :=
by
  sorry

end NUMINAMATH_GPT_sequence_factorial_l888_88876


namespace NUMINAMATH_GPT_distance_between_vertices_l888_88844

-- Define the equations of the parabolas
def C_eq (x : ℝ) : ℝ := x^2 + 6 * x + 13
def D_eq (x : ℝ) : ℝ := -x^2 + 2 * x + 8

-- Define the vertices of the parabolas
def vertex_C : (ℝ × ℝ) := (-3, 4)
def vertex_D : (ℝ × ℝ) := (1, 9)

-- Prove that the distance between the vertices is sqrt 41
theorem distance_between_vertices : 
  dist (vertex_C) (vertex_D) = Real.sqrt 41 := 
by
  sorry

end NUMINAMATH_GPT_distance_between_vertices_l888_88844


namespace NUMINAMATH_GPT_roots_interlaced_l888_88889

variable {α : Type*} [LinearOrderedField α]
variables {f g : α → α}

theorem roots_interlaced
    (x1 x2 x3 x4 : α)
    (h1 : x1 < x2) (h2 : x3 < x4)
    (hfx1 : f x1 = 0) (hfx2 : f x2 = 0)
    (hfx_distinct : x1 ≠ x2)
    (hgx3 : g x3 = 0) (hgx4 : g x4 = 0)
    (hgx_distinct : x3 ≠ x4)
    (hgx1_ne_0 : g x1 ≠ 0) (hgx2_ne_0 : g x2 ≠ 0)
    (hgx1_gx2_lt_0 : g x1 * g x2 < 0) :
    (x1 < x3 ∧ x3 < x2 ∧ x2 < x4) ∨ (x3 < x1 ∧ x1 < x4 ∧ x4 < x2) :=
sorry

end NUMINAMATH_GPT_roots_interlaced_l888_88889


namespace NUMINAMATH_GPT_more_customers_left_than_stayed_l888_88882

-- Define the initial number of customers.
def initial_customers : ℕ := 11

-- Define the number of customers who stayed behind.
def customers_stayed : ℕ := 3

-- Define the number of customers who left.
def customers_left : ℕ := initial_customers - customers_stayed

-- Prove that the number of customers who left is 5 more than those who stayed behind.
theorem more_customers_left_than_stayed : customers_left - customers_stayed = 5 := by
  -- Sorry to skip the proof 
  sorry

end NUMINAMATH_GPT_more_customers_left_than_stayed_l888_88882


namespace NUMINAMATH_GPT_solve_equation1_solve_equation2_l888_88821

def equation1 (x : ℝ) := (x - 1) ^ 2 = 4
def equation2 (x : ℝ) := 2 * x ^ 3 = -16

theorem solve_equation1 (x : ℝ) (h : equation1 x) : x = 3 ∨ x = -1 := 
sorry

theorem solve_equation2 (x : ℝ) (h : equation2 x) : x = -2 := 
sorry

end NUMINAMATH_GPT_solve_equation1_solve_equation2_l888_88821


namespace NUMINAMATH_GPT_value_of_fraction_l888_88843

variables (w x y : ℝ)

theorem value_of_fraction (h1 : w / x = 1 / 3) (h2 : w / y = 3 / 4) : (x + y) / y = 13 / 4 :=
sorry

end NUMINAMATH_GPT_value_of_fraction_l888_88843


namespace NUMINAMATH_GPT_solve_equation_l888_88802

theorem solve_equation {x : ℝ} (hx : x = 1) : 9 - 3 / x / 3 + 3 = 3 := by
  rw [hx] -- Substitute x = 1
  norm_num -- Simplify the numerical expression
  sorry -- to be proved

end NUMINAMATH_GPT_solve_equation_l888_88802


namespace NUMINAMATH_GPT_find_real_solutions_l888_88863

theorem find_real_solutions (x : ℝ) :
  (1 / ((x - 2) * (x - 3)) + 1 / ((x - 3) * (x - 4)) + 1 / ((x - 4) * (x - 5)) = 1 / 8) ↔ (x = 7 ∨ x = -2) := 
by
  sorry

end NUMINAMATH_GPT_find_real_solutions_l888_88863


namespace NUMINAMATH_GPT_cos_of_complementary_angle_l888_88831

theorem cos_of_complementary_angle (Y Z : ℝ) (h : Y + Z = π / 2) 
  (sin_Y : Real.sin Y = 3 / 5) : Real.cos Z = 3 / 5 := 
  sorry

end NUMINAMATH_GPT_cos_of_complementary_angle_l888_88831


namespace NUMINAMATH_GPT_geo_seq_sum_monotone_l888_88894

theorem geo_seq_sum_monotone (q a1 : ℝ) (n : ℕ) (S : ℕ → ℝ) :
  (∀ n, S (n + 1) > S n) ↔ (a1 > 0 ∧ q > 0) :=
sorry -- Proof of the theorem (omitted)

end NUMINAMATH_GPT_geo_seq_sum_monotone_l888_88894


namespace NUMINAMATH_GPT_range_of_a_l888_88867

-- Define the even function property
def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

-- Define the monotonically increasing property on [0, ∞)
def mono_increasing_on_nonneg (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 ≤ x → 0 ≤ y → x ≤ y → f x ≤ f y

theorem range_of_a (f : ℝ → ℝ) (a : ℝ) :
  even_function f →
  mono_increasing_on_nonneg f →
  (f (Real.log a / Real.log 2) + f (Real.log a / Real.log (1/2)) ≤ 2 * f 1) →
  (0 < a ∧ a ≤ 2) :=
by
  intros h_even h_mono h_ineq
  sorry

end NUMINAMATH_GPT_range_of_a_l888_88867


namespace NUMINAMATH_GPT_tribe_leadership_choices_l888_88834

open Nat

theorem tribe_leadership_choices (n m k l : ℕ) (h : n = 15) : 
  (choose 14 2 * choose 12 3 * choose 9 3 * 15 = 27392400) := 
  by sorry

end NUMINAMATH_GPT_tribe_leadership_choices_l888_88834


namespace NUMINAMATH_GPT_probability_X_greater_than_2_l888_88862

noncomputable def probability_distribution (i : ℕ) : ℝ :=
  if h : 1 ≤ i ∧ i ≤ 4 then i / 10 else 0

theorem probability_X_greater_than_2 :
  (probability_distribution 3 + probability_distribution 4) = 0.7 := by 
  sorry

end NUMINAMATH_GPT_probability_X_greater_than_2_l888_88862


namespace NUMINAMATH_GPT_remainder_sum_modulo_eleven_l888_88842

theorem remainder_sum_modulo_eleven :
  (88132 + 88133 + 88134 + 88135 + 88136 + 88137 + 88138 + 88139 + 88140 + 88141) % 11 = 1 :=
by
  sorry

end NUMINAMATH_GPT_remainder_sum_modulo_eleven_l888_88842


namespace NUMINAMATH_GPT_calc_result_l888_88829

-- Define the operation and conditions
def my_op (a b c : ℝ) : ℝ :=
  3 * (a - b - c)^2

theorem calc_result (x y z : ℝ) : 
  my_op ((x - y - z)^2) ((y - x - z)^2) ((z - x - y)^2) = 0 :=
by
  sorry

end NUMINAMATH_GPT_calc_result_l888_88829


namespace NUMINAMATH_GPT_description_of_T_l888_88890

def T (x y : ℝ) : Prop :=
  (5 = x+3 ∧ y-6 ≤ 5) ∨
  (5 = y-6 ∧ x+3 ≤ 5) ∨
  ((x+3 = y-6) ∧ 5 ≤ x+3)

theorem description_of_T :
  ∀ (x y : ℝ), T x y ↔ (x = 2 ∧ y ≤ 11) ∨ (y = 11 ∧ x ≤ 2) ∨ (y = x + 9 ∧ x ≥ 2) :=
sorry

end NUMINAMATH_GPT_description_of_T_l888_88890


namespace NUMINAMATH_GPT_radian_measure_of_acute_angle_l888_88833

theorem radian_measure_of_acute_angle 
  (r1 r2 r3 : ℝ) (h1 : r1 = 4) (h2 : r2 = 3) (h3 : r3 = 2)
  (θ : ℝ) (S U : ℝ) 
  (hS : S = U * 9 / 14) (h_total_area : (π * r1^2) + (π * r2^2) + (π * r3^2) = S + U) :
  θ = 1827 * π / 3220 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_radian_measure_of_acute_angle_l888_88833


namespace NUMINAMATH_GPT_bridge_length_l888_88892

theorem bridge_length 
  (train_length : ℝ) 
  (train_speed_kmph : ℝ) 
  (cross_time_seconds : ℝ)
  (train_length_eq : train_length = 150)
  (train_speed_kmph_eq : train_speed_kmph = 45)
  (cross_time_seconds_eq : cross_time_seconds = 30) : 
  ∃ (bridge_length : ℝ), bridge_length = 225 := 
  by
  sorry

end NUMINAMATH_GPT_bridge_length_l888_88892


namespace NUMINAMATH_GPT_find_bottle_caps_l888_88891

variable (B : ℕ) -- Number of bottle caps Danny found at the park.

-- Conditions
variable (current_wrappers : ℕ := 67) -- Danny has 67 wrappers in his collection now.
variable (current_bottle_caps : ℕ := 35) -- Danny has 35 bottle caps in his collection now.
variable (found_wrappers : ℕ := 18) -- Danny found 18 wrappers at the park.
variable (more_wrappers_than_bottle_caps : ℕ := 32) -- Danny has 32 more wrappers than bottle caps.

-- Given the conditions, prove that Danny found 18 bottle caps at the park.
theorem find_bottle_caps (h1 : current_wrappers = current_bottle_caps + more_wrappers_than_bottle_caps)
                         (h2 : current_bottle_caps - B + found_wrappers = current_wrappers - more_wrappers_than_bottle_caps - B) :
  B = 18 :=
by
  sorry

end NUMINAMATH_GPT_find_bottle_caps_l888_88891


namespace NUMINAMATH_GPT_setB_is_correct_l888_88875

def setA : Set ℤ := {-1, 0, 1, 2}
def f (x : ℤ) : ℤ := x^2 - 2*x
def setB : Set ℤ := {y | ∃ x ∈ setA, f x = y}

theorem setB_is_correct : setB = {-1, 0, 3} := by
  sorry

end NUMINAMATH_GPT_setB_is_correct_l888_88875


namespace NUMINAMATH_GPT_new_pressure_of_nitrogen_gas_l888_88858

variable (p1 p2 v1 v2 k : ℝ)

theorem new_pressure_of_nitrogen_gas :
  (∀ p v, p * v = k) ∧ (p1 = 8) ∧ (v1 = 3) ∧ (p1 * v1 = k) ∧ (v2 = 7.5) →
  p2 = 3.2 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_new_pressure_of_nitrogen_gas_l888_88858


namespace NUMINAMATH_GPT_total_students_is_30_l888_88846

def students_per_bed : ℕ := 2 

def beds_per_room : ℕ := 2 

def students_per_couch : ℕ := 1 

def rooms_booked : ℕ := 6 

def total_students := (students_per_bed * beds_per_room + students_per_couch) * rooms_booked

theorem total_students_is_30 : total_students = 30 := by
  sorry

end NUMINAMATH_GPT_total_students_is_30_l888_88846


namespace NUMINAMATH_GPT_pics_per_album_eq_five_l888_88838

-- Definitions based on conditions
def pics_from_phone : ℕ := 5
def pics_from_camera : ℕ := 35
def total_pics : ℕ := pics_from_phone + pics_from_camera
def num_albums : ℕ := 8

-- Statement to prove
theorem pics_per_album_eq_five : total_pics / num_albums = 5 := by
  sorry

end NUMINAMATH_GPT_pics_per_album_eq_five_l888_88838


namespace NUMINAMATH_GPT_days_between_dates_l888_88836

-- Define the starting and ending dates
def start_date : Nat := 1990 * 365 + (19 + 2 * 31 + 28) -- March 19, 1990 (accounting for leap years before the start date)
def end_date : Nat   := 1996 * 365 + (23 + 2 * 31 + 29 + 366 * 2 + 365 * 3) -- March 23, 1996 (accounting for leap years)

-- Define the number of leap years between the dates
def leap_years : Nat := 2 -- 1992 and 1996

-- Total number of days
def total_days : Nat := (end_date - start_date + 1)

theorem days_between_dates : total_days = 2197 :=
by
  sorry

end NUMINAMATH_GPT_days_between_dates_l888_88836


namespace NUMINAMATH_GPT_votes_difference_l888_88878

theorem votes_difference (V : ℝ) (h1 : 0.62 * V = 899) :
  |(0.62 * V) - (0.38 * V)| = 348 :=
by
  -- The solution goes here
  sorry

end NUMINAMATH_GPT_votes_difference_l888_88878


namespace NUMINAMATH_GPT_solve_problem_l888_88835

open Real

noncomputable def problem (x : ℝ) : Prop :=
  (cos (2 * x / 5) - cos (2 * π / 15)) ^ 2 + (sin (2 * x / 3) - sin (4 * π / 9)) ^ 2 = 0

theorem solve_problem : ∀ t : ℤ, problem ((29 * π / 3) + 15 * π * t) :=
by
  intro t
  sorry

end NUMINAMATH_GPT_solve_problem_l888_88835


namespace NUMINAMATH_GPT_tan_beta_solution_l888_88837

theorem tan_beta_solution
  (α β : ℝ)
  (h₁ : Real.tan α = 2)
  (h₂ : Real.tan (α + β) = -1) :
  Real.tan β = 3 := 
sorry

end NUMINAMATH_GPT_tan_beta_solution_l888_88837


namespace NUMINAMATH_GPT_number_of_students_joined_l888_88899

theorem number_of_students_joined
  (A : ℝ)
  (x : ℕ)
  (h1 : A = 50)
  (h2 : (100 + x) * (A - 10) = 5400) 
  (h3 : 100 * A + 400 = 5400) :
  x = 35 := 
by 
  -- all conditions in a) are used as definitions in Lean 4 statement
  sorry

end NUMINAMATH_GPT_number_of_students_joined_l888_88899


namespace NUMINAMATH_GPT_david_reading_time_l888_88801

theorem david_reading_time (total_time : ℕ) (math_time : ℕ) (spelling_time : ℕ) 
  (reading_time : ℕ) (h1 : total_time = 60) (h2 : math_time = 15) 
  (h3 : spelling_time = 18) (h4 : reading_time = total_time - (math_time + spelling_time)) : 
  reading_time = 27 := by
  sorry

end NUMINAMATH_GPT_david_reading_time_l888_88801


namespace NUMINAMATH_GPT_xy_in_N_l888_88832

def M := {x : ℤ | ∃ m : ℤ, x = 3 * m + 1}
def N := {y : ℤ | ∃ n : ℤ, y = 3 * n + 2}

theorem xy_in_N (x y : ℤ) (hx : x ∈ M) (hy : y ∈ N) : (x * y) ∈ N :=
by
  sorry

end NUMINAMATH_GPT_xy_in_N_l888_88832


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l888_88822

theorem arithmetic_sequence_sum (a b c : ℤ)
  (h1 : ∃ d : ℤ, a = 3 + d)
  (h2 : ∃ d : ℤ, b = 3 + 2 * d)
  (h3 : ∃ d : ℤ, c = 3 + 3 * d)
  (h4 : 3 + 3 * (c - 3) = 15) : a + b + c = 27 :=
by 
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l888_88822


namespace NUMINAMATH_GPT_maximum_at_vertex_l888_88859

def quadratic_function (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

theorem maximum_at_vertex (a b c x_0 : ℝ) (h_a : a < 0) (h_x0 : 2 * a * x_0 + b = 0) :
  ∀ x : ℝ, quadratic_function a b c x ≤ quadratic_function a b c x_0 :=
sorry

end NUMINAMATH_GPT_maximum_at_vertex_l888_88859


namespace NUMINAMATH_GPT_brick_height_l888_88873

variable {l w : ℕ} (SA : ℕ)

theorem brick_height (h : ℕ) (l_eq : l = 10) (w_eq : w = 4) (SA_eq : SA = 136) 
    (surface_area_eq : SA = 2 * (l * w + l * h + w * h)) : h = 2 :=
by
  sorry

end NUMINAMATH_GPT_brick_height_l888_88873


namespace NUMINAMATH_GPT_katy_books_l888_88856

theorem katy_books (x : ℕ) (h : x + 2 * x + (2 * x - 3) = 37) : x = 8 :=
by
  sorry

end NUMINAMATH_GPT_katy_books_l888_88856


namespace NUMINAMATH_GPT_initial_blue_balls_l888_88804

theorem initial_blue_balls (total_balls : ℕ) (remaining_balls : ℕ) (B : ℕ) :
  total_balls = 18 → remaining_balls = total_balls - 3 → (B - 3) / remaining_balls = 1 / 5 → B = 6 :=
by 
  intros htotal hremaining hprob
  sorry

end NUMINAMATH_GPT_initial_blue_balls_l888_88804


namespace NUMINAMATH_GPT_pinecones_left_l888_88818

theorem pinecones_left (initial_pinecones : ℕ)
    (percent_eaten_by_reindeer : ℝ)
    (percent_collected_for_fires : ℝ)
    (twice_eaten_by_squirrels : ℕ → ℕ)
    (eaten_by_reindeer : ℕ → ℝ → ℕ)
    (collected_for_fires : ℕ → ℝ → ℕ)
    (h_initial : initial_pinecones = 2000)
    (h_percent_reindeer : percent_eaten_by_reindeer = 0.20)
    (h_twice_squirrels : ∀ n, twice_eaten_by_squirrels n = 2 * n)
    (h_percent_fires : percent_collected_for_fires = 0.25)
    (h_eaten_reindeer : ∀ n p, eaten_by_reindeer n p = n * p)
    (h_collected_fires : ∀ n p, collected_for_fires n p = n * p) :
  let reindeer_eat := eaten_by_reindeer initial_pinecones percent_eaten_by_reindeer
  let squirrel_eat := twice_eaten_by_squirrels reindeer_eat
  let after_eaten := initial_pinecones - reindeer_eat - squirrel_eat
  let fire_collect := collected_for_fires after_eaten percent_collected_for_fires
  let final_pinecones := after_eaten - fire_collect
  final_pinecones = 600 :=
by sorry

end NUMINAMATH_GPT_pinecones_left_l888_88818


namespace NUMINAMATH_GPT_p_nonnegative_iff_equal_l888_88847

def p (a b c x : ℝ) : ℝ := (x - a) * (x - b) + (x - b) * (x - c) + (x - c) * (x - a)

theorem p_nonnegative_iff_equal (a b c : ℝ) : (∀ x : ℝ, p a b c x ≥ 0) ↔ a = b ∧ b = c :=
by
  sorry

end NUMINAMATH_GPT_p_nonnegative_iff_equal_l888_88847


namespace NUMINAMATH_GPT_range_of_m_l888_88861

theorem range_of_m (m : ℝ) : (∃ x y : ℝ, 2 * x^2 - 3 * x + m = 0 ∧ 2 * y^2 - 3 * y + m = 0) → m ≤ 9 / 8 :=
by
  intro h
  -- We need to implement the proof here
  sorry

end NUMINAMATH_GPT_range_of_m_l888_88861


namespace NUMINAMATH_GPT_arithmetic_sequence_a2_a4_a9_eq_18_l888_88813

theorem arithmetic_sequence_a2_a4_a9_eq_18 (a : ℕ → ℕ) (S : ℕ → ℕ) 
  (h1 : S 9 = 54) 
  (h2 : ∀ n, S n = n * (a 1 + a n) / 2) :
  a 2 + a 4 + a 9 = 18 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_a2_a4_a9_eq_18_l888_88813


namespace NUMINAMATH_GPT_range_of_function_l888_88855

open Real

theorem range_of_function (x : ℝ) (h : 0 < x ∧ x < π / 2) :
  ∃ y, y = sin x - 2 * cos x + 32 / (125 * sin x * (1 - cos x)) ∧ y ≥ 2 / 5 :=
sorry

end NUMINAMATH_GPT_range_of_function_l888_88855


namespace NUMINAMATH_GPT_record_jump_l888_88816

theorem record_jump (standard_jump jump : Float) (h_standard : standard_jump = 4.00) (h_jump : jump = 3.85) : (jump - standard_jump : Float) = -0.15 := 
by
  rw [h_standard, h_jump]
  simp
  sorry

end NUMINAMATH_GPT_record_jump_l888_88816


namespace NUMINAMATH_GPT_range_of_a_l888_88879

theorem range_of_a (a : ℝ) (x : ℝ) :
  (x^2 - 4 * a * x + 3 * a^2 < 0 → (x^2 - x - 6 ≤ 0 ∨ x^2 + 2 * x - 8 > 0)) → a < 0 → 
  (a ≤ -4 ∨ -2 / 3 ≤ a ∧ a < 0) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l888_88879


namespace NUMINAMATH_GPT_bottle_total_height_l888_88872

theorem bottle_total_height (r1 r2 water_height_up water_height_down : ℝ) (h_r1 : r1 = 1) (h_r2 : r2 = 3) (h_water_height_up : water_height_up = 20) (h_water_height_down : water_height_down = 28) : 
    ∃ x : ℝ, (π * r1^2 * (x - water_height_up) = 9 * π * (x - water_height_down) ∧ x = 29) := 
by 
    sorry

end NUMINAMATH_GPT_bottle_total_height_l888_88872


namespace NUMINAMATH_GPT_problem_1_problem_2_l888_88812

theorem problem_1 (a_0 a_1 a_2 a_3 a_4 a_5 a_6 : ℝ) (x : ℝ)
  (h : (2 * x - 1) ^ 6 = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5 + a_6 * x^6) :
  a_0 + a_1 + a_2 + a_3 + a_4 + a_5 + a_6 = 1 :=
sorry

theorem problem_2 (a_0 a_1 a_2 a_3 a_4 a_5 a_6 : ℝ) (x : ℝ)
  (h : (2 * x - 1) ^ 6 = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5 + a_6 * x^6) :
  a_0 + a_2 + a_4 + a_6 = 365 :=
sorry

end NUMINAMATH_GPT_problem_1_problem_2_l888_88812


namespace NUMINAMATH_GPT_BoatCrafters_boats_total_l888_88866

theorem BoatCrafters_boats_total
  (n_february: ℕ)
  (h_february: n_february = 5)
  (h_march: 3 * n_february = 15)
  (h_april: 3 * 15 = 45) :
  n_february + 15 + 45 = 65 := 
sorry

end NUMINAMATH_GPT_BoatCrafters_boats_total_l888_88866


namespace NUMINAMATH_GPT_find_x_l888_88883

def integers_x_y (x y : ℤ) : Prop :=
  x > y ∧ y > 0 ∧ x + y + x * y = 110

theorem find_x (x y : ℤ) (h : integers_x_y x y) : x = 36 := sorry

end NUMINAMATH_GPT_find_x_l888_88883


namespace NUMINAMATH_GPT_battery_charge_to_60_percent_l888_88880

noncomputable def battery_charge_time (initial_charge_percent : ℝ) (initial_time_minutes : ℕ) (additional_time_minutes : ℕ) : ℕ :=
  let rate_per_minute := initial_charge_percent / initial_time_minutes
  let additional_charge_percent := additional_time_minutes * rate_per_minute
  let total_percent := initial_charge_percent + additional_charge_percent
  if total_percent = 60 then
    initial_time_minutes + additional_time_minutes
  else
    sorry

theorem battery_charge_to_60_percent : battery_charge_time 20 60 120 = 180 :=
by
  -- The formal proof will be provided here.
  sorry

end NUMINAMATH_GPT_battery_charge_to_60_percent_l888_88880


namespace NUMINAMATH_GPT_neg_70kg_represents_subtract_70kg_l888_88857

theorem neg_70kg_represents_subtract_70kg (add_30kg : Int) (concept_opposite : ∀ (x : Int), x = -(-x)) :
  -70 = -70 := 
by
  sorry

end NUMINAMATH_GPT_neg_70kg_represents_subtract_70kg_l888_88857


namespace NUMINAMATH_GPT_total_weight_gain_l888_88893

def orlando_gained : ℕ := 5

def jose_gained (orlando : ℕ) : ℕ :=
  2 * orlando + 2

def fernando_gained (jose : ℕ) : ℕ :=
  jose / 2 - 3

theorem total_weight_gain (O J F : ℕ) 
  (ho : O = orlando_gained) 
  (hj : J = jose_gained O) 
  (hf : F = fernando_gained J) :
  O + J + F = 20 :=
by
  sorry

end NUMINAMATH_GPT_total_weight_gain_l888_88893


namespace NUMINAMATH_GPT_reciprocal_of_mixed_number_l888_88850

def mixed_number := -1 - (4 / 5)

def reciprocal (x : ℚ) : ℚ := 1 / x

theorem reciprocal_of_mixed_number : reciprocal mixed_number = -5 / 9 := 
by
  sorry

end NUMINAMATH_GPT_reciprocal_of_mixed_number_l888_88850


namespace NUMINAMATH_GPT_mistaken_multiplication_l888_88805

theorem mistaken_multiplication (x : ℕ) : 
  let a := 139
  let b := 43
  let incorrect_result := 1251
  (a * b - a * x = incorrect_result) ↔ (x = 34) := 
by 
  let a := 139
  let b := 43
  let incorrect_result := 1251
  sorry

end NUMINAMATH_GPT_mistaken_multiplication_l888_88805


namespace NUMINAMATH_GPT_size_relationship_l888_88809

variable (a1 a2 b1 b2 : ℝ)

theorem size_relationship (h1 : a1 < a2) (h2 : b1 < b2) : a1 * b1 + a2 * b2 > a1 * b2 + a2 * b1 := 
sorry

end NUMINAMATH_GPT_size_relationship_l888_88809


namespace NUMINAMATH_GPT_eventually_periodic_l888_88815

variable (u : ℕ → ℤ)

def bounded (u : ℕ → ℤ) : Prop :=
  ∃ (m M : ℤ), ∀ (n : ℕ), m ≤ u n ∧ u n ≤ M

def recurrence (u : ℕ → ℤ) (n : ℕ) : Prop := 
  u (n) = (u (n-1) + u (n-2) + u (n-3) * u (n-4)) / (u (n-1) * u (n-2) + u (n-3) + u (n-4))

theorem eventually_periodic (hu_bounded : bounded u) (hu_recurrence : ∀ n ≥ 4, recurrence u n) :
  ∃ N M, ∀ k ≥ 0, u (N + k) = u (N + M + k) :=
sorry

end NUMINAMATH_GPT_eventually_periodic_l888_88815


namespace NUMINAMATH_GPT_find_number_l888_88800

theorem find_number (x : ℝ) (h : 0.65 * x = 0.05 * 60 + 23) : x = 40 :=
sorry

end NUMINAMATH_GPT_find_number_l888_88800


namespace NUMINAMATH_GPT_min_matches_to_win_champion_min_total_matches_if_wins_11_l888_88865

-- Define the conditions and problem in Lean 4
def teams := ["A", "B", "C"]
def players_per_team : ℕ := 9
def initial_matches : ℕ := 0

-- The minimum number of matches the champion team must win
theorem min_matches_to_win_champion (H : ∀ t ∈ teams, t ≠ "Champion" → players_per_team = 0) :
  initial_matches + 19 = 19 :=
by
  sorry

-- The minimum total number of matches if the champion team wins 11 matches
theorem min_total_matches_if_wins_11 (wins_by_champion : ℕ := 11) (H : wins_by_champion = 11) :
  initial_matches + wins_by_champion + (players_per_team * 2 - wins_by_champion) + 4 = 24 :=
by
  sorry

end NUMINAMATH_GPT_min_matches_to_win_champion_min_total_matches_if_wins_11_l888_88865


namespace NUMINAMATH_GPT_division_result_is_correct_l888_88853

def division_result : ℚ := 132 / 6 / 3

theorem division_result_is_correct : division_result = 22 / 3 :=
by
  -- here, we would include the proof steps, but for now, we'll put sorry
  sorry

end NUMINAMATH_GPT_division_result_is_correct_l888_88853


namespace NUMINAMATH_GPT_maria_cartons_needed_l888_88823

theorem maria_cartons_needed : 
  ∀ (total_needed strawberries blueberries raspberries blackberries : ℕ), 
  total_needed = 36 →
  strawberries = 4 →
  blueberries = 8 →
  raspberries = 3 →
  blackberries = 5 →
  (total_needed - (strawberries + blueberries + raspberries + blackberries) = 16) :=
by
  intros total_needed strawberries blueberries raspberries blackberries ht hs hb hr hb
  -- ... the proof would go here
  sorry

end NUMINAMATH_GPT_maria_cartons_needed_l888_88823


namespace NUMINAMATH_GPT_find_value_of_A_l888_88806

theorem find_value_of_A (M T A E : ℕ) (H : ℕ := 8) 
  (h1 : M + A + T + H = 28) 
  (h2 : T + E + A + M = 34) 
  (h3 : M + E + E + T = 30) : 
  A = 16 :=
by 
  sorry

end NUMINAMATH_GPT_find_value_of_A_l888_88806


namespace NUMINAMATH_GPT_jane_project_time_l888_88869

theorem jane_project_time
  (J : ℝ)
  (work_rate_jane_ashley : ℝ := 1 / J + 1 / 40)
  (time_together : ℝ := 15.2 - 8)
  (work_done_together : ℝ := time_together * work_rate_jane_ashley)
  (ashley_alone_time : ℝ := 8)
  (work_done_ashley : ℝ := ashley_alone_time / 40)
  (jane_alone_time : ℝ := 4)
  (work_done_jane_alone : ℝ := jane_alone_time / J) :
  7.2 * (1 / J + 1 / 40) + 8 / 40 + 4 / J = 1 ↔ J = 18.06 :=
by 
  sorry

end NUMINAMATH_GPT_jane_project_time_l888_88869


namespace NUMINAMATH_GPT_movie_revenue_multiple_correct_l888_88828

-- Definitions from the conditions
def opening_weekend_revenue : ℝ := 120 * 10^6
def company_share_fraction : ℝ := 0.60
def profit : ℝ := 192 * 10^6
def production_cost : ℝ := 60 * 10^6

-- The statement to prove
theorem movie_revenue_multiple_correct : 
  ∃ M : ℝ, (company_share_fraction * (opening_weekend_revenue * M) - production_cost = profit) ∧ M = 3.5 :=
by
  sorry

end NUMINAMATH_GPT_movie_revenue_multiple_correct_l888_88828


namespace NUMINAMATH_GPT_ned_time_left_to_diffuse_bomb_l888_88825

-- Conditions
def building_flights : Nat := 20
def time_per_flight : Nat := 11
def bomb_timer : Nat := 72
def time_spent_running : Nat := 165

-- Main statement
theorem ned_time_left_to_diffuse_bomb : 
  (bomb_timer - (building_flights - (time_spent_running / time_per_flight)) * time_per_flight) = 17 :=
by
  sorry

end NUMINAMATH_GPT_ned_time_left_to_diffuse_bomb_l888_88825


namespace NUMINAMATH_GPT_rectangle_length_l888_88810

theorem rectangle_length (w l : ℝ) (hP : (2 * l + 2 * w) / w = 5) (hA : l * w = 150) : l = 15 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_length_l888_88810


namespace NUMINAMATH_GPT_factor_expression_l888_88886

variables (a : ℝ)

theorem factor_expression : (45 * a^2 + 135 * a + 90 * a^3) = 45 * a * (90 * a^2 + a + 3) :=
by sorry

end NUMINAMATH_GPT_factor_expression_l888_88886


namespace NUMINAMATH_GPT_intersection_of_A_and_B_l888_88895

open Set

-- Definitions of sets A and B as per conditions in the problem
def A := {x : ℝ | -1 < x ∧ x < 2}
def B := {x : ℝ | -3 < x ∧ x ≤ 1}

-- The proof statement that A ∩ B = {x | -1 < x ∧ x ≤ 1}
theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | -1 < x ∧ x ≤ 1} := by
  sorry

end NUMINAMATH_GPT_intersection_of_A_and_B_l888_88895


namespace NUMINAMATH_GPT_intersection_point_of_diagonals_l888_88852

noncomputable def intersection_of_diagonals (k m b : Real) : Real × Real :=
  let A := (0, b)
  let B := (0, -b)
  let C := (2 * b / (k - m), 2 * b * k / (k - m) - b)
  let D := (-2 * b / (k - m), -2 * b * k / (k - m) + b)
  (0, 0)

theorem intersection_point_of_diagonals (k m b : Real) :
  intersection_of_diagonals k m b = (0, 0) :=
sorry

end NUMINAMATH_GPT_intersection_point_of_diagonals_l888_88852
