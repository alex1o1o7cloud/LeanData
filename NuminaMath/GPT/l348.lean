import Mathlib

namespace NUMINAMATH_GPT_george_total_blocks_l348_34840

-- Definitions (conditions).
def large_boxes : ℕ := 5
def small_boxes_per_large_box : ℕ := 8
def blocks_per_small_box : ℕ := 9
def individual_blocks : ℕ := 6

-- Mathematical proof problem statement.
theorem george_total_blocks :
  (large_boxes * small_boxes_per_large_box * blocks_per_small_box + individual_blocks) = 366 :=
by
  -- Placeholder for proof.
  sorry

end NUMINAMATH_GPT_george_total_blocks_l348_34840


namespace NUMINAMATH_GPT_f_difference_l348_34870

noncomputable def f (x : ℝ) : ℝ := sorry

-- Define the conditions
axiom odd_f : ∀ x : ℝ, f (-x) = -f x
axiom periodic_f : ∀ x : ℝ, f (x + 4) = f x
axiom local_f : ∀ x : ℝ, -2 < x ∧ x < 0 → f x = 2^x

-- State the problem
theorem f_difference :
  f 2012 - f 2011 = -1 / 2 := sorry

end NUMINAMATH_GPT_f_difference_l348_34870


namespace NUMINAMATH_GPT_cost_to_feed_turtles_l348_34816

-- Define the conditions
def ounces_per_half_pound : ℝ := 1 
def total_weight_turtles : ℝ := 30
def food_per_half_pound : ℝ := 0.5
def ounces_per_jar : ℝ := 15
def cost_per_jar : ℝ := 2

-- Define the statement to prove
theorem cost_to_feed_turtles : (total_weight_turtles / food_per_half_pound) / ounces_per_jar * cost_per_jar = 8 := by
  sorry

end NUMINAMATH_GPT_cost_to_feed_turtles_l348_34816


namespace NUMINAMATH_GPT_volume_of_cone_l348_34862

theorem volume_of_cone (d : ℝ) (h : ℝ) (r : ℝ) : 
  d = 10 ∧ h = 0.6 * d ∧ r = d / 2 → (1 / 3) * π * r^2 * h = 50 * π :=
by
  intro h1
  rcases h1 with ⟨h_d, h_h, h_r⟩
  sorry

end NUMINAMATH_GPT_volume_of_cone_l348_34862


namespace NUMINAMATH_GPT_halfway_between_frac_l348_34864

theorem halfway_between_frac : (1 / 7 + 1 / 9) / 2 = 8 / 63 := by
  sorry

end NUMINAMATH_GPT_halfway_between_frac_l348_34864


namespace NUMINAMATH_GPT_Stephen_total_distance_l348_34898

theorem Stephen_total_distance 
  (round_trips : ℕ := 10) 
  (mountain_height : ℕ := 40000) 
  (fraction_of_height : ℚ := 3/4) :
  (round_trips * (2 * (fraction_of_height * mountain_height))) = 600000 :=
by
  sorry

end NUMINAMATH_GPT_Stephen_total_distance_l348_34898


namespace NUMINAMATH_GPT_license_plate_combinations_l348_34802

theorem license_plate_combinations : 
  let letters := 26 
  let letters_and_digits := 36 
  let middle_character_choices := 2
  3 * letters * letters_and_digits * middle_character_choices = 1872 :=
by
  sorry

end NUMINAMATH_GPT_license_plate_combinations_l348_34802


namespace NUMINAMATH_GPT_inequality_l348_34805

-- Given three distinct positive real numbers a, b, c
variables {a b c : ℝ}

-- Assume a, b, and c are distinct and positive
axiom distinct_positive (h: a ≠ b ∧ b ≠ c ∧ a ≠ c) (ha: a > 0) (hb: b > 0) (hc: c > 0) : 
  ∃ a b c, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a > 0 ∧ b > 0 ∧ c > 0

-- The inequality to be proven
theorem inequality (h: a ≠ b ∧ b ≠ c ∧ a ≠ c) (ha: a > 0) (hb: b > 0) (hc: c > 0) :
  (a / b) + (b / c) > (a / c) + (c / a) := 
sorry

end NUMINAMATH_GPT_inequality_l348_34805


namespace NUMINAMATH_GPT_horner_method_evaluation_l348_34801

def f (x : ℝ) := 0.5 * x^5 + 4 * x^4 + 0 * x^3 - 3 * x^2 + x - 1

theorem horner_method_evaluation : f 3 = 1 :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_horner_method_evaluation_l348_34801


namespace NUMINAMATH_GPT_value_of_N_l348_34803

theorem value_of_N (a b c N : ℚ) 
  (h1 : a + b + c = 120)
  (h2 : a + 8 = N)
  (h3 : 8 * b = N)
  (h4 : c / 8 = N) :
  N = 960 / 73 :=
by
  sorry

end NUMINAMATH_GPT_value_of_N_l348_34803


namespace NUMINAMATH_GPT_regular_tetrahedron_surface_area_l348_34874

theorem regular_tetrahedron_surface_area {h : ℝ} (h_pos : h > 0) :
  ∃ (S : ℝ), S = (3 * h^2 * Real.sqrt 3) / 2 :=
sorry

end NUMINAMATH_GPT_regular_tetrahedron_surface_area_l348_34874


namespace NUMINAMATH_GPT_probability_green_given_not_red_l348_34880

theorem probability_green_given_not_red :
  let total_balls := 20
  let red_balls := 5
  let yellow_balls := 5
  let green_balls := 10
  let non_red_balls := total_balls - red_balls

  let probability_green_given_not_red := (green_balls : ℚ) / (non_red_balls : ℚ)

  probability_green_given_not_red = 2 / 3 :=
by
  sorry

end NUMINAMATH_GPT_probability_green_given_not_red_l348_34880


namespace NUMINAMATH_GPT_mike_picked_32_limes_l348_34888

theorem mike_picked_32_limes (total_limes : ℕ) (alyssa_limes : ℕ) (mike_limes : ℕ) 
  (h1 : total_limes = 57) (h2 : alyssa_limes = 25) (h3 : mike_limes = total_limes - alyssa_limes) : 
  mike_limes = 32 :=
by
  sorry

end NUMINAMATH_GPT_mike_picked_32_limes_l348_34888


namespace NUMINAMATH_GPT_area_of_region_l348_34879

theorem area_of_region :
  ∫ y in (0:ℝ)..(1:ℝ), y ^ (2 / 3) = 3 / 5 :=
by
  sorry

end NUMINAMATH_GPT_area_of_region_l348_34879


namespace NUMINAMATH_GPT_intersection_M_N_l348_34824

-- Define the sets M and N
def M : Set ℝ := { x : ℝ | x^2 - x - 6 < 0 }
def N : Set ℝ := { x : ℝ | 1 < x }

-- State the problem in terms of Lean definitions and theorem
theorem intersection_M_N : M ∩ N = {x : ℝ | 1 < x ∧ x < 3} :=
by
  sorry

end NUMINAMATH_GPT_intersection_M_N_l348_34824


namespace NUMINAMATH_GPT_randy_biscuits_l348_34884

theorem randy_biscuits (F : ℕ) (initial_biscuits mother_biscuits brother_ate remaining_biscuits : ℕ) 
  (h_initial : initial_biscuits = 32)
  (h_mother : mother_biscuits = 15)
  (h_brother : brother_ate = 20)
  (h_remaining : remaining_biscuits = 40)
  : ((initial_biscuits + mother_biscuits + F) - brother_ate) = remaining_biscuits → F = 13 := 
by
  intros h_eq
  sorry

end NUMINAMATH_GPT_randy_biscuits_l348_34884


namespace NUMINAMATH_GPT_marvin_next_birthday_monday_l348_34834

def is_leap_year (y : ℕ) : Prop :=
  (y % 4 = 0 ∧ y % 100 ≠ 0) ∨ (y % 400 = 0)

def day_of_week_after_leap_years (start_day : ℕ) (leap_years : ℕ) : ℕ :=
  (start_day + 2 * leap_years) % 7

def next_birthday_on_monday (year : ℕ) (start_day : ℕ) : ℕ :=
  let next_day := day_of_week_after_leap_years start_day ((year - 2012)/4)
  year + 4 * ((7 - next_day + 1) / 2)

theorem marvin_next_birthday_monday : next_birthday_on_monday 2012 3 = 2016 :=
by sorry

end NUMINAMATH_GPT_marvin_next_birthday_monday_l348_34834


namespace NUMINAMATH_GPT_fuel_capacity_ratio_l348_34887

noncomputable def oldCost : ℝ := 200
noncomputable def newCost : ℝ := 480
noncomputable def priceIncreaseFactor : ℝ := 1.20

theorem fuel_capacity_ratio (C C_new : ℝ) (h1 : newCost = C_new * oldCost * priceIncreaseFactor / C) : 
  C_new / C = 2 :=
sorry

end NUMINAMATH_GPT_fuel_capacity_ratio_l348_34887


namespace NUMINAMATH_GPT_tan_C_in_triangle_l348_34848

theorem tan_C_in_triangle (A B C : ℝ) (hA : Real.tan A = 1 / 2) (hB : Real.cos B = 3 * Real.sqrt 10 / 10) :
  Real.tan C = -1 :=
sorry

end NUMINAMATH_GPT_tan_C_in_triangle_l348_34848


namespace NUMINAMATH_GPT_sin_double_angle_value_l348_34859

open Real

theorem sin_double_angle_value (x : ℝ) 
  (h1 : sin (x + π/3) * cos (x - π/6) + sin (x - π/6) * cos (x + π/3) = 5 / 13)
  (h2 : -π/3 ≤ x ∧ x ≤ π/6) :
  sin (2 * x) = (5 * sqrt 3 - 12) / 26 :=
by
  sorry

end NUMINAMATH_GPT_sin_double_angle_value_l348_34859


namespace NUMINAMATH_GPT_max_cells_intersected_10_radius_circle_l348_34869

noncomputable def max_cells_intersected_by_circle (radius : ℝ) (cell_size : ℝ) : ℕ :=
  if radius = 10 ∧ cell_size = 1 then 80 else 0

theorem max_cells_intersected_10_radius_circle :
  max_cells_intersected_by_circle 10 1 = 80 :=
sorry

end NUMINAMATH_GPT_max_cells_intersected_10_radius_circle_l348_34869


namespace NUMINAMATH_GPT_simplify_sqrt_450_l348_34863

noncomputable def sqrt_450 : ℝ := Real.sqrt 450
noncomputable def expected_value : ℝ := 15 * Real.sqrt 2

theorem simplify_sqrt_450 : sqrt_450 = expected_value := 
by sorry

end NUMINAMATH_GPT_simplify_sqrt_450_l348_34863


namespace NUMINAMATH_GPT_measure_of_angle_A_l348_34852

theorem measure_of_angle_A (A B : ℝ) (h1 : A + B = 180) (h2 : A = 5 * B) : A = 150 := 
by 
  sorry

end NUMINAMATH_GPT_measure_of_angle_A_l348_34852


namespace NUMINAMATH_GPT_volume_displacement_square_l348_34832

-- Define the given conditions
def radius_cylinder := 5
def height_cylinder := 12
def side_length_cube := 10

theorem volume_displacement_square :
  let r := radius_cylinder
  let h := height_cylinder
  let s := side_length_cube
  let cube_diagonal := s * Real.sqrt 3
  let w := (125 * Real.sqrt 6) / 8
  w^2 = 1464.0625 :=
by
  sorry

end NUMINAMATH_GPT_volume_displacement_square_l348_34832


namespace NUMINAMATH_GPT_find_north_speed_l348_34814

-- Define the variables and conditions
variables (v : ℝ)  -- the speed of the cyclist going towards the north
def south_speed : ℝ := 25  -- the speed of the cyclist going towards the south is 25 km/h
def time_taken : ℝ := 1.4285714285714286  -- time taken to be 50 km apart
def distance_apart : ℝ := 50  -- distance apart after given time

-- Define the hypothesis based on the conditions
def relative_speed (v : ℝ) : ℝ := v + south_speed
def distance_formula (v : ℝ) : Prop :=
  distance_apart = relative_speed v * time_taken

-- The statement to prove
theorem find_north_speed : distance_formula v → v = 10 :=
  sorry

end NUMINAMATH_GPT_find_north_speed_l348_34814


namespace NUMINAMATH_GPT_problem_solution_l348_34828

theorem problem_solution (a b c : ℤ)
  (h1 : ∀ x : ℤ, |x| ≠ |a|)
  (h2 : ∀ x : ℤ, x^2 ≠ b^2)
  (h3 : ∀ x : ℤ, x * c ≤ 1):
  a + b + c = 0 :=
by sorry

end NUMINAMATH_GPT_problem_solution_l348_34828


namespace NUMINAMATH_GPT_square_of_binomial_l348_34885

theorem square_of_binomial (a : ℝ) :
  (∃ b : ℝ, (9:ℝ) * x^2 + 24 * x + a = (3 * x + b)^2) → a = 16 :=
by
  sorry

end NUMINAMATH_GPT_square_of_binomial_l348_34885


namespace NUMINAMATH_GPT_doug_age_l348_34846

theorem doug_age (Qaddama Jack Doug : ℕ) 
  (h1 : Qaddama = Jack + 6)
  (h2 : Jack = Doug - 3)
  (h3 : Qaddama = 19) : 
  Doug = 16 := 
by 
  sorry

end NUMINAMATH_GPT_doug_age_l348_34846


namespace NUMINAMATH_GPT_cyclist_rate_l348_34807

theorem cyclist_rate 
  (rate_hiker : ℝ := 4)
  (wait_time_1 : ℝ := 5 / 60)
  (wait_time_2 : ℝ := 10.000000000000002 / 60)
  (hiker_distance : ℝ := rate_hiker * wait_time_2)
  (cyclist_distance : ℝ := hiker_distance)
  (cyclist_rate := cyclist_distance / wait_time_1) :
  cyclist_rate = 8 := by 
sorry

end NUMINAMATH_GPT_cyclist_rate_l348_34807


namespace NUMINAMATH_GPT_compute_a_l348_34826

theorem compute_a 
  (a b : ℚ) 
  (h : ∃ (x : ℝ), x^3 + (a : ℝ) * x^2 + (b : ℝ) * x - 37 = 0 ∧ x = 2 - 3 * Real.sqrt 3) : 
  a = -55 / 23 :=
by 
  sorry

end NUMINAMATH_GPT_compute_a_l348_34826


namespace NUMINAMATH_GPT_evaluate_f_difference_l348_34818

def f (x : ℝ) : ℝ := x^6 - 2 * x^4 + 7 * x

theorem evaluate_f_difference :
  f 3 - f (-3) = 42 := by
  sorry

end NUMINAMATH_GPT_evaluate_f_difference_l348_34818


namespace NUMINAMATH_GPT_smallest_n_for_identity_matrix_l348_34837

noncomputable def rotation_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  ![
    ![ 1 / 2, -Real.sqrt 3 / 2 ],
    ![ Real.sqrt 3 / 2, 1 / 2]
  ]

theorem smallest_n_for_identity_matrix : ∃ (n : ℕ), n > 0 ∧ 
  ∃ (k : ℕ), rotation_matrix ^ n = 1 ∧ n = 3 :=
by
  sorry

end NUMINAMATH_GPT_smallest_n_for_identity_matrix_l348_34837


namespace NUMINAMATH_GPT_inequality_a_b_c_d_l348_34896

theorem inequality_a_b_c_d
  (a b c d : ℝ)
  (h₀ : 0 ≤ a) (h₁ : 0 ≤ b) (h₂ : 0 ≤ c) (h₃ : 0 ≤ d)
  (h₄ : a * b + b * c + c * d + d * a = 1) :
  (a ^ 3 / (b + c + d) + b ^ 3 / (c + d + a) + c ^ 3 / (a + b + d) + d ^ 3 / (a + b + c)) ≥ (1 / 3) :=
by
  sorry

end NUMINAMATH_GPT_inequality_a_b_c_d_l348_34896


namespace NUMINAMATH_GPT_common_difference_of_consecutive_multiples_l348_34822

/-- The sides of a rectangular prism are consecutive multiples of a certain number n. The base area is 450.
    Prove that the common difference between the consecutive multiples is 15. -/
theorem common_difference_of_consecutive_multiples (n d : ℕ) (h₁ : n * (n + d) = 450) : d = 15 :=
sorry

end NUMINAMATH_GPT_common_difference_of_consecutive_multiples_l348_34822


namespace NUMINAMATH_GPT_ratio_of_x_l348_34893

theorem ratio_of_x (x : ℝ) (h : x = Real.sqrt 7 + Real.sqrt 6) :
    ((x + 1 / x) / (x - 1 / x)) = (Real.sqrt 7 / Real.sqrt 6) :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_x_l348_34893


namespace NUMINAMATH_GPT_average_disk_space_per_minute_l348_34812

theorem average_disk_space_per_minute 
  (days : ℕ := 15) 
  (disk_space : ℕ := 36000) 
  (minutes_per_day : ℕ := 1440) 
  (total_minutes := days * minutes_per_day) 
  (average_space_per_minute := disk_space / total_minutes) :
  average_space_per_minute = 2 :=
sorry

end NUMINAMATH_GPT_average_disk_space_per_minute_l348_34812


namespace NUMINAMATH_GPT_union_sets_eq_l348_34853

-- Definitions of the sets M and N according to the conditions.
def M : Set ℝ := {x | x^2 = x}
def N : Set ℝ := {x | Real.log x ≤ 0}

-- The theorem we want to prove
theorem union_sets_eq :
  (M ∪ N) = Set.Icc 0 1 :=
by
  sorry

end NUMINAMATH_GPT_union_sets_eq_l348_34853


namespace NUMINAMATH_GPT_probability_of_sum_16_with_duplicates_l348_34891

namespace DiceProbability

def is_valid_die_roll (n : ℕ) : Prop :=
  n ≥ 1 ∧ n ≤ 6

def is_valid_combination (x y z : ℕ) : Prop :=
  x + y + z = 16 ∧ 
  is_valid_die_roll x ∧ 
  is_valid_die_roll y ∧ 
  is_valid_die_roll z ∧ 
  (x = y ∨ y = z ∨ z = x)

theorem probability_of_sum_16_with_duplicates (P : ℚ) :
  (∃ x y z : ℕ, is_valid_combination x y z) → 
  P = 1 / 36 :=
sorry

end DiceProbability

end NUMINAMATH_GPT_probability_of_sum_16_with_duplicates_l348_34891


namespace NUMINAMATH_GPT_traffic_lights_states_l348_34836

theorem traffic_lights_states (n k : ℕ) : 
  (k ≤ n) → 
  (∃ (ways : ℕ), ways = 3^k * 2^(n - k)) :=
by
  sorry

end NUMINAMATH_GPT_traffic_lights_states_l348_34836


namespace NUMINAMATH_GPT_sequence_expression_l348_34813

theorem sequence_expression (a : ℕ → ℚ)
  (h1 : a 1 = 2 / 3)
  (h2 : ∀ n : ℕ, a (n + 1) = (n / (n + 1)) * a n) :
  ∀ n : ℕ, a n = 2 / (3 * n) :=
sorry

end NUMINAMATH_GPT_sequence_expression_l348_34813


namespace NUMINAMATH_GPT_total_rainfall_recorded_l348_34817

-- Define the conditions based on the rainfall amounts for each day
def rainfall_monday : ℝ := 0.16666666666666666
def rainfall_tuesday : ℝ := 0.4166666666666667
def rainfall_wednesday : ℝ := 0.08333333333333333

-- State the theorem: the total rainfall recorded over the three days is 0.6666666666666667 cm.
theorem total_rainfall_recorded :
  (rainfall_monday + rainfall_tuesday + rainfall_wednesday) = 0.6666666666666667 := by
  sorry

end NUMINAMATH_GPT_total_rainfall_recorded_l348_34817


namespace NUMINAMATH_GPT_equidistant_xaxis_point_l348_34810

theorem equidistant_xaxis_point {x : ℝ} :
  (∃ x : ℝ, ∀ A B : ℝ × ℝ, A = (-3, 0) ∧ B = (2, 5) →
    ∀ P : ℝ × ℝ, P = (x, 0) →
      (dist A P = dist B P) → x = 2) := sorry

end NUMINAMATH_GPT_equidistant_xaxis_point_l348_34810


namespace NUMINAMATH_GPT_problem_1_problem_2_l348_34854

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * (x + Real.log x)
def g (x : ℝ) : ℝ := x^2

theorem problem_1 (a : ℝ) (ha : a ≠ 0) : 
  (∀ (x : ℝ), f a x = a * (x + Real.log x)) →
  deriv (f a) 1 = deriv g 1 → a = 1 := 
by 
  sorry

theorem problem_2 (a : ℝ) (ha : 0 < a) (hb : a < 1) (x1 x2 : ℝ) 
  (hx1 : 1 ≤ x1) (hx2 : x2 ≤ 2) (hx12 : x1 ≠ x2) : 
  |f a x1 - f a x2| < |g x1 - g x2| := 
by 
  sorry

end NUMINAMATH_GPT_problem_1_problem_2_l348_34854


namespace NUMINAMATH_GPT_percentage_enclosed_by_hexagons_is_50_l348_34895

noncomputable def hexagon_area (s : ℝ) : ℝ :=
  (3 * Real.sqrt 3 / 2) * s^2

noncomputable def square_area (s : ℝ) : ℝ :=
  s^2

noncomputable def total_tiling_unit_area (s : ℝ) : ℝ :=
  hexagon_area s + 3 * square_area s

noncomputable def percentage_enclosed_by_hexagons (s : ℝ) : ℝ :=
  (hexagon_area s / total_tiling_unit_area s) * 100

theorem percentage_enclosed_by_hexagons_is_50 (s : ℝ) : percentage_enclosed_by_hexagons s = 50 := by
  sorry

end NUMINAMATH_GPT_percentage_enclosed_by_hexagons_is_50_l348_34895


namespace NUMINAMATH_GPT_log_suff_nec_l348_34875

theorem log_suff_nec (a b : ℝ) (ha : a > 0) (hb : b > 0) : ¬ ((a > b) ↔ (Real.log b / Real.log a < 1)) := 
sorry

end NUMINAMATH_GPT_log_suff_nec_l348_34875


namespace NUMINAMATH_GPT_max_area_of_right_angled_isosceles_triangle_l348_34841

theorem max_area_of_right_angled_isosceles_triangle (a b : ℝ) (h₁ : a = 12) (h₂ : b = 15) :
  ∃ A : ℝ, A = 72 ∧ 
  (∀ (x : ℝ), x ≤ min a b → (1 / 2) * x^2 ≤ A) :=
by
  use 72
  sorry

end NUMINAMATH_GPT_max_area_of_right_angled_isosceles_triangle_l348_34841


namespace NUMINAMATH_GPT_notable_features_points_l348_34825

namespace Points3D

def is_first_octant (x y z : ℝ) : Prop := (x > 0) ∧ (y > 0) ∧ (z > 0)
def is_second_octant (x y z : ℝ) : Prop := (x < 0) ∧ (y > 0) ∧ (z > 0)
def is_eighth_octant (x y z : ℝ) : Prop := (x > 0) ∧ (y < 0) ∧ (z < 0)
def lies_in_YOZ_plane (x y z : ℝ) : Prop := (x = 0) ∧ (y ≠ 0) ∧ (z ≠ 0)
def lies_on_OY_axis (x y z : ℝ) : Prop := (x = 0) ∧ (y ≠ 0) ∧ (z = 0)
def is_origin (x y z : ℝ) : Prop := (x = 0) ∧ (y = 0) ∧ (z = 0)

theorem notable_features_points :
  is_first_octant 3 2 6 ∧
  is_second_octant (-2) 3 1 ∧
  is_eighth_octant 1 (-4) (-2) ∧
  is_eighth_octant 1 (-2) (-1) ∧
  lies_in_YOZ_plane 0 4 1 ∧
  lies_on_OY_axis 0 2 0 ∧
  is_origin 0 0 0 :=
by
  sorry

end Points3D

end NUMINAMATH_GPT_notable_features_points_l348_34825


namespace NUMINAMATH_GPT_dice_sum_four_l348_34871

def possible_outcomes (x : Nat) : Set (Nat × Nat) :=
  { (d1, d2) | d1 + d2 = x ∧ 1 ≤ d1 ∧ d1 ≤ 6 ∧ 1 ≤ d2 ∧ d2 ≤ 6 }

theorem dice_sum_four :
  possible_outcomes 4 = {(3, 1), (1, 3), (2, 2)} :=
by
  sorry -- We acknowledge that this outline is equivalent to the provided math problem.

end NUMINAMATH_GPT_dice_sum_four_l348_34871


namespace NUMINAMATH_GPT_largest_crate_dimension_l348_34830

def largest_dimension_of_crate : ℝ := 10

theorem largest_crate_dimension (length width : ℝ) (r : ℝ) (h : ℝ) 
  (h_length : length = 5) (h_width : width = 8) (h_radius : r = 5) (h_height : h >= 10) :
  h = largest_dimension_of_crate :=
by 
  sorry

end NUMINAMATH_GPT_largest_crate_dimension_l348_34830


namespace NUMINAMATH_GPT_fixed_point_l348_34883

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^(2*x - 1)

theorem fixed_point (a : ℝ) (h_pos : a > 0) (h_ne_one : a ≠ 1) : f a (1/2) = 1 :=
by
  sorry

end NUMINAMATH_GPT_fixed_point_l348_34883


namespace NUMINAMATH_GPT_original_price_of_cycle_l348_34866

noncomputable def original_price_given_gain (SP : ℝ) (gain : ℝ) : ℝ :=
  SP / (1 + gain)

theorem original_price_of_cycle (SP : ℝ) (HSP : SP = 1350) (Hgain : gain = 0.5) : 
  original_price_given_gain SP gain = 900 := 
by
  sorry

end NUMINAMATH_GPT_original_price_of_cycle_l348_34866


namespace NUMINAMATH_GPT_parallel_lines_slope_l348_34833

-- Define the given conditions
def line1_slope (x : ℝ) : ℝ := 6
def line2_slope (c : ℝ) (x : ℝ) : ℝ := 3 * c

-- State the proof problem
theorem parallel_lines_slope (c : ℝ) : 
  (∀ x : ℝ, line1_slope x = line2_slope c x) → c = 2 :=
by
  intro h
  -- Intro provides a human-readable variable and corresponding proof obligation
  -- The remainder of the proof would follow here, but instead,
  -- we use "sorry" to indicate an incomplete proof
  sorry

end NUMINAMATH_GPT_parallel_lines_slope_l348_34833


namespace NUMINAMATH_GPT_sequence_general_term_l348_34819

theorem sequence_general_term (a : ℕ → ℕ) (h1 : a 1 = 1) (h2 : ∀ n, n ≥ 1 → a (n + 1) = a n + 2) : ∀ n, a n = 2 * n - 1 :=
by
  sorry

end NUMINAMATH_GPT_sequence_general_term_l348_34819


namespace NUMINAMATH_GPT_no_a_satisfy_quadratic_equation_l348_34838

theorem no_a_satisfy_quadratic_equation :
  ∀ (a : ℕ), (a > 0) ∧ (a ≤ 100) ∧
  (∃ x₁ x₂ : ℤ, x₁ ≠ x₂ ∧ x₁ * x₂ = 2 * a^2 ∧ x₁ + x₂ = -(3*a + 1)) → false := by
  sorry

end NUMINAMATH_GPT_no_a_satisfy_quadratic_equation_l348_34838


namespace NUMINAMATH_GPT_multiplicative_inverse_mod_l348_34881

-- We define our variables
def a := 154
def m := 257
def inv_a := 20

-- Our main theorem stating that inv_a is indeed the multiplicative inverse of a modulo m
theorem multiplicative_inverse_mod : (a * inv_a) % m = 1 := by
  sorry

end NUMINAMATH_GPT_multiplicative_inverse_mod_l348_34881


namespace NUMINAMATH_GPT_factorize_expression_l348_34868

theorem factorize_expression (x : ℝ) : 3 * x^2 - 12 = 3 * (x + 2) * (x - 2) := 
by 
  sorry

end NUMINAMATH_GPT_factorize_expression_l348_34868


namespace NUMINAMATH_GPT_solution_is_13_l348_34827

def marbles_in_jars : Prop :=
  let jar1 := (5, 3, 1)  -- (red, blue, green)
  let jar2 := (1, 5, 3)  -- (red, blue, green)
  let jar3 := (3, 1, 5)  -- (red, blue, green)
  let total_ways := 125 + 15 + 15 + 3 + 27 + 15
  let favorable_ways := 125
  let probability := favorable_ways / total_ways
  let simplified_probability := 5 / 8
  let m := 5
  let n := 8
  m + n = 13

theorem solution_is_13 : marbles_in_jars :=
by {
  sorry
}

end NUMINAMATH_GPT_solution_is_13_l348_34827


namespace NUMINAMATH_GPT_bread_cost_each_is_3_l348_34877

-- Define the given conditions
def initial_amount : ℕ := 86
def bread_quantity : ℕ := 3
def orange_juice_quantity : ℕ := 3
def orange_juice_cost_each : ℕ := 6
def remaining_amount : ℕ := 59

-- Define the variable for bread cost
variable (B : ℕ)

-- Lean 4 statement to prove the cost of each loaf of bread
theorem bread_cost_each_is_3 :
  initial_amount - remaining_amount = (bread_quantity * B + orange_juice_quantity * orange_juice_cost_each) →
  B = 3 :=
by
  sorry

end NUMINAMATH_GPT_bread_cost_each_is_3_l348_34877


namespace NUMINAMATH_GPT_simplify_and_evaluate_l348_34811

theorem simplify_and_evaluate :
  let a := 1
  let b := 2
  (a - b) ^ 2 - a * (a - b) + (a + b) * (a - b) = -1 := by
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_l348_34811


namespace NUMINAMATH_GPT_no_real_roots_poly_l348_34821

theorem no_real_roots_poly (a b c : ℝ) (h : |a| + |b| + |c| ≤ Real.sqrt 2) :
  ∀ x : ℝ, x^4 + a*x^3 + b*x^2 + c*x + 1 > 0 := 
  sorry

end NUMINAMATH_GPT_no_real_roots_poly_l348_34821


namespace NUMINAMATH_GPT_differential_savings_l348_34892

theorem differential_savings (income : ℝ) (tax_rate1 tax_rate2 : ℝ) 
                            (old_tax_rate_eq : tax_rate1 = 0.40) 
                            (new_tax_rate_eq : tax_rate2 = 0.33) 
                            (income_eq : income = 45000) :
    ((tax_rate1 - tax_rate2) * income) = 3150 :=
by
  rw [old_tax_rate_eq, new_tax_rate_eq, income_eq]
  norm_num

end NUMINAMATH_GPT_differential_savings_l348_34892


namespace NUMINAMATH_GPT_prove_inequality_l348_34831

noncomputable def inequality_holds (x y : ℝ) : Prop :=
  x^3 * (y + 1) + y^3 * (x + 1) ≥ x^2 * (y + y^2) + y^2 * (x + x^2)

theorem prove_inequality (x y : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) : inequality_holds x y :=
  sorry

end NUMINAMATH_GPT_prove_inequality_l348_34831


namespace NUMINAMATH_GPT_nonnegative_integer_solution_count_l348_34865

theorem nonnegative_integer_solution_count :
  ∃ n : ℕ, (∀ x : ℕ, x^2 + 6 * x = 0 → x = 0) ∧ n = 1 :=
by
  sorry

end NUMINAMATH_GPT_nonnegative_integer_solution_count_l348_34865


namespace NUMINAMATH_GPT_time_between_shark_sightings_l348_34890

def earnings_per_photo : ℕ := 15
def fuel_cost_per_hour : ℕ := 50
def hunting_hours : ℕ := 5
def expected_profit : ℕ := 200

theorem time_between_shark_sightings :
  (hunting_hours * 60) / ((expected_profit + (fuel_cost_per_hour * hunting_hours)) / earnings_per_photo) = 10 :=
by 
  sorry

end NUMINAMATH_GPT_time_between_shark_sightings_l348_34890


namespace NUMINAMATH_GPT_find_interest_rate_l348_34886
noncomputable def annualInterestRate (P A : ℝ) (n t : ℕ) (r : ℝ) : Prop :=
  P * (1 + r / n)^(n * t) = A

theorem find_interest_rate :
  annualInterestRate 5000 6050.000000000001 1 2 0.1 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_GPT_find_interest_rate_l348_34886


namespace NUMINAMATH_GPT_sequence_a2017_l348_34878

theorem sequence_a2017 (a : ℕ → ℤ)
  (h1 : a 1 = 2)
  (h2 : a 2 = 3)
  (h_rec : ∀ n, a (n + 2) = a (n + 1) - a n) :
  a 2017 = 2 :=
sorry

end NUMINAMATH_GPT_sequence_a2017_l348_34878


namespace NUMINAMATH_GPT_determine_a_square_binomial_l348_34800

theorem determine_a_square_binomial (a : ℝ) :
  (∃ r s : ℝ, ∀ x : ℝ, ax^2 + 24*x + 9 = (r*x + s)^2) → a = 16 :=
by
  sorry

end NUMINAMATH_GPT_determine_a_square_binomial_l348_34800


namespace NUMINAMATH_GPT_quadratic_inequality_iff_abs_a_le_two_l348_34857

-- Definitions from the condition
variable (a : ℝ)
def quadratic_expr (x : ℝ) : ℝ := x^2 + a * x + 1

-- Statement of the problem as a Lean 4 statement
theorem quadratic_inequality_iff_abs_a_le_two :
  (∀ x : ℝ, quadratic_expr a x ≥ 0) ↔ (|a| ≤ 2) := sorry

end NUMINAMATH_GPT_quadratic_inequality_iff_abs_a_le_two_l348_34857


namespace NUMINAMATH_GPT_betty_sugar_l348_34849

theorem betty_sugar (f s : ℝ) (hf1 : f ≥ 8 + (3 / 4) * s) (hf2 : f ≤ 3 * s) : s ≥ 4 := 
sorry

end NUMINAMATH_GPT_betty_sugar_l348_34849


namespace NUMINAMATH_GPT_point_2023_0_cannot_lie_on_line_l348_34897

-- Define real numbers a and c with the condition ac > 0
variables (a c : ℝ)

-- The condition ac > 0
def ac_positive := (a * c > 0)

-- The statement that (2023, 0) cannot be on the line y = ax + c given the condition a * c > 0
theorem point_2023_0_cannot_lie_on_line (h : ac_positive a c) : ¬ (0 = 2023 * a + c) :=
sorry

end NUMINAMATH_GPT_point_2023_0_cannot_lie_on_line_l348_34897


namespace NUMINAMATH_GPT_find_m_value_l348_34815

theorem find_m_value (x y m : ℝ) 
  (h1 : 2 * x + y = 5) 
  (h2 : x - 2 * y = m)
  (h3 : 2 * x - 3 * y = 1) : 
  m = 0 := 
sorry

end NUMINAMATH_GPT_find_m_value_l348_34815


namespace NUMINAMATH_GPT_latus_rectum_parabola_l348_34899

theorem latus_rectum_parabola : 
  ∀ (x y : ℝ), (x = 4 * y^2) → (x = -1/16) :=
by 
  sorry

end NUMINAMATH_GPT_latus_rectum_parabola_l348_34899


namespace NUMINAMATH_GPT_number_of_people_in_group_l348_34861

theorem number_of_people_in_group :
  ∀ (N : ℕ), (75 - 35) = 5 * N → N = 8 :=
by
  intros N h
  sorry

end NUMINAMATH_GPT_number_of_people_in_group_l348_34861


namespace NUMINAMATH_GPT_relay_race_total_time_l348_34876

-- Definitions based on the problem conditions
def athlete1_time : ℕ := 55
def athlete2_time : ℕ := athlete1_time + 10
def athlete3_time : ℕ := athlete2_time - 15
def athlete4_time : ℕ := athlete1_time - 25

-- Problem statement
theorem relay_race_total_time : 
  athlete1_time + athlete2_time + athlete3_time + athlete4_time = 200 := 
by 
  sorry

end NUMINAMATH_GPT_relay_race_total_time_l348_34876


namespace NUMINAMATH_GPT_inequality_l348_34835

theorem inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a + b + c = 3) : 
  (1 / (8 * a^2 - 18 * a + 11)) + (1 / (8 * b^2 - 18 * b + 11)) + (1 / (8 * c^2 - 18 * c + 11)) ≤ 3 := 
sorry

end NUMINAMATH_GPT_inequality_l348_34835


namespace NUMINAMATH_GPT_find_range_of_a_l348_34850

noncomputable def f (x : ℝ) : ℝ := (1 / Real.exp x) - (Real.exp x) + 2 * x - (1 / 3) * x ^ 3

theorem find_range_of_a (a : ℝ) (h : f (3 * a ^ 2) + f (2 * a - 1) ≥ 0) : a ∈ Set.Icc (-1 : ℝ) (1 / 3) :=
sorry

end NUMINAMATH_GPT_find_range_of_a_l348_34850


namespace NUMINAMATH_GPT_bike_ride_ratio_l348_34847

theorem bike_ride_ratio (J : ℕ) (B : ℕ) (M : ℕ) (hB : B = 17) (hM : M = J + 10) (hTotal : B + J + M = 95) :
  J / B = 2 :=
by
  sorry

end NUMINAMATH_GPT_bike_ride_ratio_l348_34847


namespace NUMINAMATH_GPT_sinA_value_find_b_c_l348_34851

-- Define the conditions
def triangle (A B C : Type) (a b c : ℝ) : Prop :=
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0

variable {A B C : Type} (a b c : ℝ)
variable {S_triangle_ABC : ℝ}
variable {cosB : ℝ}

-- Given conditions
axiom cosB_val : cosB = 3 / 5
axiom a_val : a = 2

-- Problem 1: Prove sinA = 2/5 given additional condition b = 4
axiom b_val : b = 4

theorem sinA_value (h_triangle : triangle A B C a b c) (h_cosB : cosB = 3/5) (h_a : a = 2) (h_b : b = 4) : 
  ∃ sinA : ℝ, sinA = 2 / 5 :=
sorry

-- Problem 2: Prove b = sqrt(17) and c = 5 given the area
axiom area_val : S_triangle_ABC = 4

theorem find_b_c (h_triangle : triangle A B C a b c) (h_cosB : cosB = 3/5) (h_a : a = 2) (h_area : S_triangle_ABC = 4) : 
  ∃ b c : ℝ, b = Real.sqrt 17 ∧ c = 5 :=
sorry

end NUMINAMATH_GPT_sinA_value_find_b_c_l348_34851


namespace NUMINAMATH_GPT_cubics_inequality_l348_34882

theorem cubics_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) : a^3 + b^3 ≥ a^2 * b + a * b^2 :=
by
  sorry

end NUMINAMATH_GPT_cubics_inequality_l348_34882


namespace NUMINAMATH_GPT_pet_store_dogs_count_l348_34806

def initial_dogs : ℕ := 2
def sunday_received_dogs : ℕ := 5
def sunday_sold_dogs : ℕ := 2
def monday_received_dogs : ℕ := 3
def monday_returned_dogs : ℕ := 1
def tuesday_received_dogs : ℕ := 4
def tuesday_sold_dogs : ℕ := 3

theorem pet_store_dogs_count :
  initial_dogs 
  + sunday_received_dogs - sunday_sold_dogs
  + monday_received_dogs + monday_returned_dogs
  + tuesday_received_dogs - tuesday_sold_dogs = 10 := 
sorry

end NUMINAMATH_GPT_pet_store_dogs_count_l348_34806


namespace NUMINAMATH_GPT_total_games_attended_l348_34889

theorem total_games_attended 
  (games_this_month : ℕ)
  (games_last_month : ℕ)
  (games_next_month : ℕ)
  (total_games : ℕ) 
  (h : games_this_month = 11)
  (h2 : games_last_month = 17)
  (h3 : games_next_month = 16) 
  (htotal : total_games = 44) :
  games_this_month + games_last_month + games_next_month = total_games :=
by sorry

end NUMINAMATH_GPT_total_games_attended_l348_34889


namespace NUMINAMATH_GPT_parallel_vectors_y_value_l348_34820

theorem parallel_vectors_y_value (y : ℝ) :
  let a := (2, 3)
  let b := (4, y)
  ∃ y : ℝ, (2 : ℝ) / 4 = 3 / y → y = 6 :=
sorry

end NUMINAMATH_GPT_parallel_vectors_y_value_l348_34820


namespace NUMINAMATH_GPT_true_prices_for_pie_and_mead_l348_34804

-- Definitions for true prices
variable (k m : ℕ)

-- Definitions for conditions
def honest_pravdoslav (k m : ℕ) : Prop :=
  4*k = 3*(m + 2) ∧ 4*(m+2) = 3*k + 14

theorem true_prices_for_pie_and_mead (k m : ℕ) (h : honest_pravdoslav k m) : k = 6 ∧ m = 6 := sorry

end NUMINAMATH_GPT_true_prices_for_pie_and_mead_l348_34804


namespace NUMINAMATH_GPT_min_sum_a_b_l348_34858

theorem min_sum_a_b (a b : ℕ) (h1 : a ≠ b) (h2 : 0 < a ∧ 0 < b) (h3 : (1/a + 1/b) = 1/12) : a + b = 54 :=
sorry

end NUMINAMATH_GPT_min_sum_a_b_l348_34858


namespace NUMINAMATH_GPT_total_flowering_bulbs_count_l348_34823

-- Definitions for the problem conditions
def crocus_cost : ℝ := 0.35
def daffodil_cost : ℝ := 0.65
def total_budget : ℝ := 29.15
def crocus_count : ℕ := 22

-- Theorem stating the total number of bulbs that can be bought
theorem total_flowering_bulbs_count : 
  ∃ daffodil_count : ℕ, (crocus_count + daffodil_count = 55) ∧ (total_budget = crocus_cost * crocus_count + daffodil_count * daffodil_cost) :=
  sorry

end NUMINAMATH_GPT_total_flowering_bulbs_count_l348_34823


namespace NUMINAMATH_GPT_prove_f_10_l348_34855

variable (f : ℝ → ℝ)

-- Conditions from the problem
def condition : Prop := ∀ x : ℝ, f (3 ^ x) = x

-- Statement of the problem
theorem prove_f_10 (h : condition f) : f 10 = Real.log 10 / Real.log 3 :=
by
  sorry

end NUMINAMATH_GPT_prove_f_10_l348_34855


namespace NUMINAMATH_GPT_road_length_l348_34867

theorem road_length (L : ℝ) (h1 : 300 = 200 + 100)
  (h2 : 50 * 100 = 2.5 / (L / 300))
  (h3 : 75 + 50 = 125)
  (h4 : (125 / 50) * (2.5 / 100) * 200 = L - 2.5) : L = 15 := 
by
  sorry

end NUMINAMATH_GPT_road_length_l348_34867


namespace NUMINAMATH_GPT_problem_1_problem_2_l348_34842

noncomputable section

variables {A B C : ℝ} {a b c : ℝ}

-- Condition definitions
def triangle_conditions (A B : ℝ) (a b c : ℝ) : Prop :=
  b = 3 ∧ c = 1 ∧ A = 2 * B

-- Problem 1: Prove that a = 2 * sqrt(3)
theorem problem_1 {A B C a : ℝ} (h : triangle_conditions A B a b c) : a = 2 * Real.sqrt 3 := sorry

-- Problem 2: Prove the value of cos(2A + π/6)
theorem problem_2 {A B C a : ℝ} (h : triangle_conditions A B a b c) : 
  Real.cos (2 * A + Real.pi / 6) = (4 * Real.sqrt 2 - 7 * Real.sqrt 3) / 18 := sorry

end NUMINAMATH_GPT_problem_1_problem_2_l348_34842


namespace NUMINAMATH_GPT_distance_interval_l348_34843

-- Define the conditions based on the false statements:
variable (d : ℝ)

def false_by_alice : Prop := d < 8
def false_by_bob : Prop := d > 7
def false_by_charlie : Prop := d ≠ 6

theorem distance_interval (h_alice : false_by_alice d) (h_bob : false_by_bob d) (h_charlie : false_by_charlie d) :
  7 < d ∧ d < 8 :=
by
  sorry

end NUMINAMATH_GPT_distance_interval_l348_34843


namespace NUMINAMATH_GPT_social_media_phone_ratio_l348_34844

/-- 
Given that Jonathan spends 8 hours on his phone daily and 28 hours on social media in a week, 
prove that the ratio of the time spent on social media to the total time spent on his phone daily is \( 1 : 2 \).
-/
theorem social_media_phone_ratio (daily_phone_hours : ℕ) (weekly_social_media_hours : ℕ) 
  (h1 : daily_phone_hours = 8) (h2 : weekly_social_media_hours = 28) :
  (weekly_social_media_hours / 7) / daily_phone_hours = 1 / 2 := 
by
  sorry

end NUMINAMATH_GPT_social_media_phone_ratio_l348_34844


namespace NUMINAMATH_GPT_petya_max_votes_difference_l348_34845

theorem petya_max_votes_difference :
  ∃ (P1 P2 V1 V2 : ℕ), 
    P1 = V1 + 9 ∧ 
    V2 = P2 + 9 ∧ 
    P1 + P2 + V1 + V2 = 27 ∧ 
    P1 + P2 > V1 + V2 ∧ 
    (P1 + P2) - (V1 + V2) = 9 := 
by
  sorry

end NUMINAMATH_GPT_petya_max_votes_difference_l348_34845


namespace NUMINAMATH_GPT_mod_exp_value_l348_34839

theorem mod_exp_value (m : ℕ) (h1: 0 ≤ m) (h2: m < 9) (h3: 14^4 ≡ m [MOD 9]) : m = 5 :=
by
  sorry

end NUMINAMATH_GPT_mod_exp_value_l348_34839


namespace NUMINAMATH_GPT_parabola_directrix_eq_l348_34829

theorem parabola_directrix_eq (p : ℝ) (h : y^2 = 2 * x ∧ p = 1) : x = -p / 2 := by
  sorry

end NUMINAMATH_GPT_parabola_directrix_eq_l348_34829


namespace NUMINAMATH_GPT_sum_of_x_and_y_l348_34872

theorem sum_of_x_and_y 
  (x y : ℝ)
  (h : ((x + 1) + (y-1)) / 2 = 10) : x + y = 20 :=
sorry

end NUMINAMATH_GPT_sum_of_x_and_y_l348_34872


namespace NUMINAMATH_GPT_find_line_through_and_perpendicular_l348_34809

def point (x y : ℝ) := (x, y)

def passes_through (P : ℝ × ℝ) (a b c : ℝ) :=
  a * P.1 + b * P.2 + c = 0

def is_perpendicular (a1 b1 a2 b2 : ℝ) :=
  a1 * a2 + b1 * b2 = 0

theorem find_line_through_and_perpendicular :
  ∃ c : ℝ, passes_through (1, -1) 1 1 c ∧ is_perpendicular 1 (-1) 1 1 → 
  c = 0 :=
by
  sorry

end NUMINAMATH_GPT_find_line_through_and_perpendicular_l348_34809


namespace NUMINAMATH_GPT_milk_students_l348_34873

theorem milk_students (T : ℕ) (h1 : (1 / 4) * T = 80) : (3 / 4) * T = 240 := by
  sorry

end NUMINAMATH_GPT_milk_students_l348_34873


namespace NUMINAMATH_GPT_quadratic_polynomial_value_l348_34860

noncomputable def f (p q r : ℝ) (x : ℝ) : ℝ := p * x^2 + q * x + r

theorem quadratic_polynomial_value (a b c : ℝ) (hab : a ≠ b) (hbc : b ≠ c) (hca : c ≠ a)
  (h1 : f 1 (-(a + b + c)) (a*b + b*c + a*c) a = b * c)
  (h2 : f 1 (-(a + b + c)) (a*b + b*c + a*c) b = c * a)
  (h3 : f 1 (-(a + b + c)) (a*b + b*c + a*c) c = a * b) :
  f 1 (-(a + b + c)) (a*b + b*c + a*c) (a + b + c) = a * b + b * c + a * c := sorry

end NUMINAMATH_GPT_quadratic_polynomial_value_l348_34860


namespace NUMINAMATH_GPT_M1M2_product_l348_34894

theorem M1M2_product :
  ∀ (M1 M2 : ℝ),
  (∀ x : ℝ, x^2 - 5 * x + 6 ≠ 0 →
    (45 * x - 55) / (x^2 - 5 * x + 6) = M1 / (x - 2) + M2 / (x - 3)) →
  (M1 + M2 = 45) →
  (3 * M1 + 2 * M2 = 55) →
  M1 * M2 = 200 :=
by
  sorry

end NUMINAMATH_GPT_M1M2_product_l348_34894


namespace NUMINAMATH_GPT_max_elevation_l348_34856

def elevation (t : ℝ) : ℝ := 144 * t - 18 * t^2

theorem max_elevation : ∃ t : ℝ, elevation t = 288 :=
by
  use 4
  sorry

end NUMINAMATH_GPT_max_elevation_l348_34856


namespace NUMINAMATH_GPT_max_value_sequence_l348_34808

theorem max_value_sequence (a : ℕ → ℝ)
  (h1 : ∀ n : ℕ, a (n + 1) = (-1 : ℝ)^n * n - a n)
  (h2 : a 10 = a 1) :
  ∃ n, a n * a (n + 1) = 33 / 4 :=
sorry

end NUMINAMATH_GPT_max_value_sequence_l348_34808
