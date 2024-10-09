import Mathlib

namespace shaded_region_area_l375_37557

noncomputable def line1 (x : ℝ) : ℝ := -(3 / 10) * x + 5
noncomputable def line2 (x : ℝ) : ℝ := -(5 / 7) * x + 47 / 7

noncomputable def intersection_x : ℝ := 17 / 5

noncomputable def area_under_curve (f g : ℝ → ℝ) (a b : ℝ) : ℝ :=
  ∫ x in a..b, (g x - f x)

theorem shaded_region_area : 
  area_under_curve line1 line2 0 intersection_x = 1.91 :=
sorry

end shaded_region_area_l375_37557


namespace muffin_sum_l375_37574

theorem muffin_sum (N : ℕ) : 
  (N % 13 = 3) → 
  (N % 8 = 5) → 
  (N < 120) → 
  (N = 16 ∨ N = 81 ∨ N = 107) → 
  (16 + 81 + 107 = 204) := 
by sorry

end muffin_sum_l375_37574


namespace smallest_a_plus_b_l375_37513

theorem smallest_a_plus_b : ∃ a b : ℕ, a > 0 ∧ b > 0 ∧ 2^3 * 3^7 * 7^2 = a^b ∧ a + b = 380 :=
sorry

end smallest_a_plus_b_l375_37513


namespace ratio_of_areas_of_squares_l375_37585

theorem ratio_of_areas_of_squares (sideC sideD : ℕ) (hC : sideC = 45) (hD : sideD = 60) : 
  (sideC ^ 2) / (sideD ^ 2) = 9 / 16 := 
by
  sorry

end ratio_of_areas_of_squares_l375_37585


namespace probability_five_common_correct_l375_37580

-- Define the conditions
def compulsory_subjects : ℕ := 3  -- Chinese, Mathematics, and English
def elective_from_physics_history : ℕ := 1  -- Physics and History
def elective_from_four : ℕ := 4  -- Politics, Geography, Chemistry, Biology

def chosen_subjects_by_xiaoming_xiaofang : ℕ := 2  -- two subjects from the four electives

-- Calculate total combinations
noncomputable def total_combinations : ℕ := Nat.choose 4 2 * Nat.choose 4 2

-- Calculate combinations to have exactly five subjects in common
noncomputable def combinations_five_common : ℕ := Nat.choose 4 2 * Nat.choose 2 1 * Nat.choose 2 1

-- Calculate the probability
noncomputable def probability_five_common : ℚ := combinations_five_common / total_combinations

-- The theorem to be proved
theorem probability_five_common_correct : probability_five_common = 2 / 3 := by
  sorry

end probability_five_common_correct_l375_37580


namespace age_of_15th_student_l375_37550

theorem age_of_15th_student (avg_age_all : ℝ) (avg_age_4 : ℝ) (avg_age_10 : ℝ) 
  (total_students : ℕ) (group_4_students : ℕ) (group_10_students : ℕ) 
  (h1 : avg_age_all = 15) (h2 : avg_age_4 = 14) (h3 : avg_age_10 = 16) 
  (h4 : total_students = 15) (h5 : group_4_students = 4) (h6 : group_10_students = 10) : 
  ∃ x : ℝ, x = 9 := 
by 
  sorry

end age_of_15th_student_l375_37550


namespace factor_x_squared_minus_sixtyfour_l375_37554

theorem factor_x_squared_minus_sixtyfour (x : ℝ) : x^2 - 64 = (x - 8) * (x + 8) :=
by sorry

end factor_x_squared_minus_sixtyfour_l375_37554


namespace angle_between_line_and_plane_l375_37535

open Real

def plane1 (x y z : ℝ) : Prop := 2*x - y - 3*z + 5 = 0
def plane2 (x y z : ℝ) : Prop := x + y - 2 = 0

def point_M : ℝ × ℝ × ℝ := (-2, 0, 3)
def point_N : ℝ × ℝ × ℝ := (0, 2, 2)
def point_K : ℝ × ℝ × ℝ := (3, -3, 1)

theorem angle_between_line_and_plane :
  ∃ α : ℝ, α = arcsin (22 / (3 * sqrt 102)) :=
by sorry

end angle_between_line_and_plane_l375_37535


namespace joels_age_when_dad_twice_l375_37593

theorem joels_age_when_dad_twice
  (joel_age_now : ℕ)
  (dad_age_now : ℕ)
  (years : ℕ)
  (H1 : joel_age_now = 5)
  (H2 : dad_age_now = 32)
  (H3 : years = 22)
  (H4 : dad_age_now + years = 2 * (joel_age_now + years))
  : joel_age_now + years = 27 := 
by sorry

end joels_age_when_dad_twice_l375_37593


namespace find_f_3_l375_37549

noncomputable def f : ℝ → ℝ := sorry

axiom functional_eq (x y : ℝ) : f (x + y) = f x + f y
axiom f_4_eq_6 : f 4 = 6

theorem find_f_3 : f 3 = 9 / 2 :=
by sorry

end find_f_3_l375_37549


namespace proportion_solution_l375_37524

theorem proportion_solution (x : ℝ) (h : 0.75 / x = 7 / 8) : x = 6 / 7 :=
by {
  sorry
}

end proportion_solution_l375_37524


namespace graph_description_l375_37547

theorem graph_description : ∀ x y : ℝ, (x + y)^2 = 2 * (x^2 + y^2) → x = 0 ∧ y = 0 :=
by 
  sorry

end graph_description_l375_37547


namespace number_of_siblings_l375_37583

-- Definitions for the given conditions
def total_height : ℕ := 330
def sibling1_height : ℕ := 66
def sibling2_height : ℕ := 66
def sibling3_height : ℕ := 60
def last_sibling_height : ℕ := 70  -- Derived from the solution steps
def eliza_height : ℕ := last_sibling_height - 2

-- The final question to validate
theorem number_of_siblings (h : 2 * sibling1_height + sibling3_height + last_sibling_height + eliza_height = total_height) :
  4 = 4 :=
by {
  -- Condition h states that the total height is satisfied
  -- Therefore, it directly justifies our claim without further computation here.
  sorry
}

end number_of_siblings_l375_37583


namespace find_largest_angle_l375_37591

noncomputable def largest_angle_in_convex_pentagon (x : ℝ) : Prop :=
  let angle1 := 2 * x + 2
  let angle2 := 3 * x - 3
  let angle3 := 4 * x + 4
  let angle4 := 6 * x - 6
  let angle5 := x + 5
  angle1 + angle2 + angle3 + angle4 + angle5 = 540 ∧
  max (max angle1 (max angle2 (max angle3 angle4))) angle5 = angle4 ∧
  angle4 = 195.75

theorem find_largest_angle (x : ℝ) : largest_angle_in_convex_pentagon x := by
  sorry

end find_largest_angle_l375_37591


namespace find_x_l375_37501

theorem find_x (x : ℝ) (hx_pos : x > 0) (hx_ceil_eq : ⌈x⌉ = 15) : x = 14 :=
by
  -- Define the condition
  have h_eq : ⌈x⌉ * x = 210 := sorry
  -- Prove that the only solution is x = 14
  sorry

end find_x_l375_37501


namespace solve_system_of_equations_l375_37569

theorem solve_system_of_equations
  (x y : ℝ)
  (h1 : 1 / x + 1 / (2 * y) = (x^2 + 3 * y^2) * (3 * x^2 + y^2))
  (h2 : 1 / x - 1 / (2 * y) = 2 * (y^4 - x^4)) :
  x = (3 ^ (1 / 5) + 1) / 2 ∧ y = (3 ^ (1 / 5) - 1) / 2 :=
by
  sorry

end solve_system_of_equations_l375_37569


namespace time_passed_since_midnight_l375_37565

theorem time_passed_since_midnight (h : ℝ) :
  h = (12 - h) + (2/5) * h → h = 7.5 :=
by
  sorry

end time_passed_since_midnight_l375_37565


namespace value_of_y_l375_37531

theorem value_of_y (x y : ℝ) (h1 : 3 * (x - y) = 18) (h2 : x + y = 20) : y = 7 := by
  sorry

end value_of_y_l375_37531


namespace james_collected_on_first_day_l375_37558

-- Conditions
variables (x : ℕ) -- the number of tins collected on the first day
variable (h1 : 500 = x + 3 * x + (3 * x - 50) + 4 * 50) -- total number of tins collected

-- Theorem to be proved
theorem james_collected_on_first_day : x = 50 :=
by
  sorry

end james_collected_on_first_day_l375_37558


namespace ratio_of_speeds_l375_37584

theorem ratio_of_speeds (D H : ℕ) (h1 : D = 2 * H) (h2 : 10 * D = 20 * H) :
  10 * D = 2 * (10 * H) :=
by
  sorry

example (D H : ℕ) (h1 : D = 2 * H) (h2 : 10 * D = 20 * H) :
  10 = 10 :=
by
  sorry

end ratio_of_speeds_l375_37584


namespace sin_two_alpha_l375_37525

theorem sin_two_alpha (alpha : ℝ) (h : Real.cos (π / 4 - alpha) = 4 / 5) : 
  Real.sin (2 * alpha) = 7 / 25 :=
by
  sorry

end sin_two_alpha_l375_37525


namespace linda_original_amount_l375_37546

theorem linda_original_amount (L L2 : ℕ) 
  (h1 : L = 20) 
  (h2 : L - 5 = L2) : 
  L2 + 5 = 15 := 
sorry

end linda_original_amount_l375_37546


namespace Kiarra_age_l375_37563

variable (Kiarra Bea Job Figaro Harry : ℕ)

theorem Kiarra_age 
  (h1 : Kiarra = 2 * Bea)
  (h2 : Job = 3 * Bea)
  (h3 : Figaro = Job + 7)
  (h4 : Harry = Figaro / 2)
  (h5 : Harry = 26) : 
  Kiarra = 30 := sorry

end Kiarra_age_l375_37563


namespace area_triangle_formed_by_line_l375_37503

theorem area_triangle_formed_by_line (b : ℝ) (h : (1 / 2) * |b * (-b / 2)| > 1) : b < -2 ∨ b > 2 :=
by 
  sorry

end area_triangle_formed_by_line_l375_37503


namespace sum_of_roots_l375_37576

theorem sum_of_roots (a b c : ℝ) (x1 x2 x3 : ℝ) (h_eq: 6*x1^3 + 7*x2^2 - 12*x3 = 0) :
  (x1 + x2 + x3) = -1.17 :=
sorry

end sum_of_roots_l375_37576


namespace money_together_l375_37538

variable (Billy Sam : ℕ)

theorem money_together (h1 : Billy = 2 * Sam - 25) (h2 : Sam = 75) : Billy + Sam = 200 := by
  sorry

end money_together_l375_37538


namespace proposal_spreading_problem_l375_37506

theorem proposal_spreading_problem (n : ℕ) : 1 + n + n^2 = 1641 := 
sorry

end proposal_spreading_problem_l375_37506


namespace find_a_b_c_l375_37567

theorem find_a_b_c :
  ∃ a b c : ℕ, a = 1 ∧ b = 17 ∧ c = 2 ∧ (Nat.gcd a c = 1) ∧ a + b + c = 20 :=
by {
  -- the proof would go here
  sorry
}

end find_a_b_c_l375_37567


namespace find_x_l375_37555

theorem find_x (x : ℤ) (h : (2 + 76 + x) / 3 = 5) : x = -63 := 
sorry

end find_x_l375_37555


namespace quadratic_equation_roots_l375_37577

theorem quadratic_equation_roots {x y : ℝ}
  (h1 : x + y = 10)
  (h2 : |x - y| = 4)
  (h3 : x * y = 21) : (x - 7) * (x - 3) = 0 ∨ (x - 3) * (x - 7) = 0 :=
by
  sorry

end quadratic_equation_roots_l375_37577


namespace find_a_l375_37542

def setA (a : ℤ) : Set ℤ := {a, 0}

def setB : Set ℤ := {x : ℤ | 3 * x^2 - 10 * x < 0}

theorem find_a (a : ℤ) (h : (setA a ∩ setB).Nonempty) : a = 1 ∨ a = 2 ∨ a = 3 :=
sorry

end find_a_l375_37542


namespace min_containers_needed_l375_37556

theorem min_containers_needed 
  (total_boxes1 : ℕ) 
  (weight_box1 : ℕ) 
  (total_boxes2 : ℕ) 
  (weight_box2 : ℕ) 
  (weight_limit : ℕ) :
  total_boxes1 = 90000 →
  weight_box1 = 3300 →
  total_boxes2 = 5000 →
  weight_box2 = 200 →
  weight_limit = 100000 →
  (total_boxes1 * weight_box1 + total_boxes2 * weight_box2 + weight_limit - 1) / weight_limit = 3000 :=
by
  sorry

end min_containers_needed_l375_37556


namespace evaluate_composite_function_l375_37520

def g (x : ℝ) : ℝ := 3 * x^2 - 4

def h (x : ℝ) : ℝ := 5 * x^3 + 2

theorem evaluate_composite_function : g (h 2) = 5288 := by
  sorry

end evaluate_composite_function_l375_37520


namespace reciprocal_of_neg_2023_l375_37575

theorem reciprocal_of_neg_2023 : (-2023: ℝ) * (-1 / 2023) = 1 := 
by sorry

end reciprocal_of_neg_2023_l375_37575


namespace third_side_triangle_max_l375_37526

theorem third_side_triangle_max (a b c : ℝ) (h1 : a = 5) (h2 : b = 10) (h3 : a + b > c) (h4 : a + c > b) (h5 : b + c > a) : c = 14 :=
by
  sorry

end third_side_triangle_max_l375_37526


namespace paint_used_l375_37573

theorem paint_used (total_paint : ℚ) (first_week_fraction : ℚ) (second_week_fraction : ℚ) 
  (first_week_paint : ℚ) (remaining_paint : ℚ) (second_week_paint : ℚ) (total_used_paint : ℚ) :
  total_paint = 360 →
  first_week_fraction = 1/6 →
  second_week_fraction = 1/5 →
  first_week_paint = first_week_fraction * total_paint →
  remaining_paint = total_paint - first_week_paint →
  second_week_paint = second_week_fraction * remaining_paint →
  total_used_paint = first_week_paint + second_week_paint →
  total_used_paint = 120 := sorry

end paint_used_l375_37573


namespace parallelogram_area_l375_37571

theorem parallelogram_area (base height : ℝ) (h_base : base = 10) (h_height : height = 7) :
  base * height = 70 := by
  rw [h_base, h_height]
  norm_num

end parallelogram_area_l375_37571


namespace cos_three_pi_over_four_l375_37517

theorem cos_three_pi_over_four :
  Real.cos (3 * Real.pi / 4) = -1 / Real.sqrt 2 :=
by
  sorry

end cos_three_pi_over_four_l375_37517


namespace volume_of_pyramid_l375_37514

/--
Rectangle ABCD is the base of pyramid PABCD. Let AB = 10, BC = 6, PA is perpendicular to AB, and PB = 20. 
If PA makes an angle θ = 30° with the diagonal AC of the base, prove the volume of the pyramid PABCD is 200 cubic units.
-/
theorem volume_of_pyramid (AB BC PB : ℝ) (θ : ℝ) (hAB : AB = 10) (hBC : BC = 6)
  (hPB : PB = 20) (hθ : θ = 30) (PA_is_perpendicular_to_AB : true) (PA_makes_angle_with_AC : true) : 
  ∃ V, V = 1 / 3 * (AB * BC) * 10 ∧ V = 200 := 
by
  exists 1 / 3 * (AB * BC) * 10
  sorry

end volume_of_pyramid_l375_37514


namespace fiona_hoodies_l375_37523

theorem fiona_hoodies (F C : ℕ) (h1 : F + C = 8) (h2 : C = F + 2) : F = 3 :=
by
  sorry

end fiona_hoodies_l375_37523


namespace last_digit_322_pow_369_l375_37528

theorem last_digit_322_pow_369 : (322^369) % 10 = 2 := by
  sorry

end last_digit_322_pow_369_l375_37528


namespace smallest_number_divisible_by_18_70_100_84_increased_by_3_l375_37515

theorem smallest_number_divisible_by_18_70_100_84_increased_by_3 :
  ∃ n : ℕ, (n + 3) % 18 = 0 ∧ (n + 3) % 70 = 0 ∧ (n + 3) % 100 = 0 ∧ (n + 3) % 84 = 0 ∧ n = 6297 :=
by
  sorry

end smallest_number_divisible_by_18_70_100_84_increased_by_3_l375_37515


namespace shape_is_spiral_l375_37562

-- Assume cylindrical coordinates and constants.
variables (c : ℝ)
-- Define cylindrical coordinate properties.
variables (r θ z : ℝ)

-- Define the equation rθ = c.
def cylindrical_equation : Prop := r * θ = c

theorem shape_is_spiral (h : cylindrical_equation c r θ):
  ∃ f : ℝ → ℝ, ∀ θ > 0, r = f θ ∧ (∀ θ₁ θ₂, θ₁ < θ₂ ↔ f θ₁ > f θ₂) :=
sorry

end shape_is_spiral_l375_37562


namespace largest_prime_divisor_of_sum_of_squares_l375_37566

def largest_prime_divisor (n : ℕ) : ℕ := sorry

theorem largest_prime_divisor_of_sum_of_squares :
  largest_prime_divisor (11^2 + 90^2) = 89 :=
by sorry

end largest_prime_divisor_of_sum_of_squares_l375_37566


namespace proof_line_eq_l375_37599

variable (a T : ℝ) (line : ℝ × ℝ → Prop)

def line_eq (point : ℝ × ℝ) : Prop := 
  point.2 = (-2 * T / a^2) * point.1 + (2 * T / a)

def correct_line_eq (point : ℝ × ℝ) : Prop :=
  -2 * T * point.1 + a^2 * point.2 + 2 * a * T = 0

theorem proof_line_eq :
  ∀ point : ℝ × ℝ, line_eq a T point ↔ correct_line_eq a T point :=
by
  sorry

end proof_line_eq_l375_37599


namespace find_largest_element_l375_37581

noncomputable def increasing_sequence (a : ℕ → ℝ) : Prop :=
∀ i j, 1 ≤ i → i < j → j ≤ 8 → a i < a j

noncomputable def arithmetic_progression (a : ℕ → ℝ) (d : ℝ) (i : ℕ) : Prop :=
a (i+1) - a i = d ∧ a (i+2) - a (i+1) = d ∧ a (i+3) - a (i+2) = d

noncomputable def geometric_progression (a : ℕ → ℝ) (i : ℕ) : Prop :=
a (i+1) / a i = a (i+2) / a (i+1) ∧ a (i+2) / a (i+1) = a (i+3) / a (i+2)

theorem find_largest_element
  (a : ℕ → ℝ)
  (h_inc : increasing_sequence a)
  (h_ap1 : ∃ i, 1 ≤ i ∧ i ≤ 5 ∧ arithmetic_progression a 4 i)
  (h_ap2 : ∃ j, 1 ≤ j ∧ j ≤ 5 ∧ arithmetic_progression a 36 j)
  (h_gp : ∃ k, 1 ≤ k ∧ k ≤ 5 ∧ geometric_progression a k) :
  a 8 = 126 :=
sorry

end find_largest_element_l375_37581


namespace circle_E_radius_sum_l375_37530

noncomputable def radius_A := 15
noncomputable def radius_B := 5
noncomputable def radius_C := 3
noncomputable def radius_D := 3

-- We need to find that the sum of m and n for the radius of circle E is 131.
theorem circle_E_radius_sum (m n : ℕ) (h1 : Nat.gcd m n = 1) (radius_E : ℚ := (m / n)) :
  m + n = 131 :=
  sorry

end circle_E_radius_sum_l375_37530


namespace sum_first_100_even_numbers_divisible_by_6_l375_37522

-- Define the sequence of even numbers divisible by 6 between 100 and 300 inclusive.
def even_numbers_divisible_by_6 (n : ℕ) : ℕ := 102 + n * 6

-- Define the sum of the first 100 even numbers divisible by 6.
def sum_even_numbers_divisible_by_6 (k : ℕ) : ℕ := k / 2 * (102 + (102 + (k - 1) * 6))

-- Define the problem statement as a theorem.
theorem sum_first_100_even_numbers_divisible_by_6 :
  sum_even_numbers_divisible_by_6 100 = 39900 :=
by
  sorry

end sum_first_100_even_numbers_divisible_by_6_l375_37522


namespace parts_rate_relation_l375_37502

theorem parts_rate_relation
  (x : ℝ)
  (total_parts_per_hour : ℝ)
  (master_parts : ℝ)
  (apprentice_parts : ℝ)
  (h_total : total_parts_per_hour = 40)
  (h_master : master_parts = 300)
  (h_apprentice : apprentice_parts = 100)
  (h : total_parts_per_hour = x + (40 - x)) :
  (master_parts / x) = (apprentice_parts / (40 - x)) := 
by
  sorry

end parts_rate_relation_l375_37502


namespace proof_problem_l375_37559

variable {a b c : ℝ}

theorem proof_problem (h1 : ∀ x : ℝ, 4 * x^2 - 3 * x + 1 = a * (x - 1)^2 + b * (x - 1) + c) : 
  (4 * a + 2 * b + c = 28) := by
  -- The proof goes here. The goal statement is what we need.
  sorry

end proof_problem_l375_37559


namespace total_rooms_booked_l375_37597

variable (S D : ℕ)

theorem total_rooms_booked (h1 : 35 * S + 60 * D = 14000) (h2 : D = 196) : S + D = 260 :=
by
  sorry

end total_rooms_booked_l375_37597


namespace height_of_balcony_l375_37507

variable (t : ℝ) (v₀ : ℝ) (g : ℝ) (h₀ : ℝ)

axiom cond1 : t = 6
axiom cond2 : v₀ = 20
axiom cond3 : g = 10

theorem height_of_balcony : h₀ + v₀ * t - (1/2 : ℝ) * g * t^2 = 0 → h₀ = 60 :=
by
  intro h'
  sorry

end height_of_balcony_l375_37507


namespace sculpture_cost_in_inr_l375_37552

def convert_currency (n_cost : ℕ) (n_to_b_rate : ℕ) (b_to_i_rate : ℕ) : ℕ := 
  (n_cost / n_to_b_rate) * b_to_i_rate

theorem sculpture_cost_in_inr (n_cost : ℕ) (n_to_b_rate : ℕ) (b_to_i_rate : ℕ) :
  n_cost = 360 → 
  n_to_b_rate = 18 → 
  b_to_i_rate = 20 →
  convert_currency n_cost n_to_b_rate b_to_i_rate = 400 := by
  intros h1 h2 h3
  rw [h1, h2, h3]
  -- turns 360 / 18 * 20 = 400
  sorry

end sculpture_cost_in_inr_l375_37552


namespace binary_to_base4_conversion_l375_37541

theorem binary_to_base4_conversion :
  let b := 110110100
  let b_2 := Nat.ofDigits 2 [1, 1, 0, 1, 1, 0, 1, 0, 0]
  let b_4 := Nat.ofDigits 4 [3, 1, 2, 2, 0]
  b_2 = b → b_4 = 31220 :=
by
  intros b b_2 b_4 h
  sorry

end binary_to_base4_conversion_l375_37541


namespace solve_system_l375_37594

theorem solve_system (X Y Z : ℝ)
  (h1 : 0.15 * 40 = 0.25 * X + 2)
  (h2 : 0.30 * 60 = 0.20 * Y + 3)
  (h3 : 0.10 * Z = X - Y) :
  X = 16 ∧ Y = 75 ∧ Z = -590 :=
by
  sorry

end solve_system_l375_37594


namespace smallest_leading_coefficient_l375_37511

theorem smallest_leading_coefficient :
  ∀ (P : ℤ → ℤ), (∃ (a b c : ℚ), ∀ (x : ℤ), P x = a * (x^2 : ℚ) + b * (x : ℚ) + c) →
  (∀ x : ℤ, ∃ k : ℤ, P x = k) →
  (∃ a : ℚ, (∀ x : ℤ, ∃ k : ℤ, a * (x^2 : ℚ) + b * (x : ℚ) + c = k) ∧ a > 0 ∧ (∀ a' : ℚ, (∀ x : ℤ, ∃ k : ℤ, a' * (x^2 : ℚ) + b * (x : ℚ) + c = k) → a' ≥ a) ∧ a = 1 / 2) := 
sorry

end smallest_leading_coefficient_l375_37511


namespace part1_part2_l375_37596

def f (x : ℝ) : ℝ := abs (x + 2) - 2 * abs (x - 1)

theorem part1 : { x : ℝ | f x ≥ -2 } = { x : ℝ | -2/3 ≤ x ∧ x ≤ 6 } :=
by
  sorry

theorem part2 (a : ℝ) :
  (∀ x ≥ a, f x ≤ x - a) ↔ a ≤ -2 ∨ a ≥ 4 :=
by
  sorry

end part1_part2_l375_37596


namespace probability_of_winning_set_l375_37579

def winning_probability : ℚ :=
  let total_cards := 9
  let total_draws := 3
  let same_color_sets := 3
  let same_letter_sets := 3
  let total_ways_to_draw := Nat.choose total_cards total_draws
  let total_favorable_outcomes := same_color_sets + same_letter_sets
  let probability := total_favorable_outcomes / total_ways_to_draw
  probability

theorem probability_of_winning_set :
  winning_probability = 1 / 14 :=
by
  sorry

end probability_of_winning_set_l375_37579


namespace evaluate_x_squared_plus_y_squared_l375_37598

theorem evaluate_x_squared_plus_y_squared
  (x y : ℝ)
  (h1 : x + y = 12)
  (h2 : 3 * x + y = 20) :
  x^2 + y^2 = 80 := by
  sorry

end evaluate_x_squared_plus_y_squared_l375_37598


namespace marble_arrangement_count_l375_37534
noncomputable def countValidMarbleArrangements : Nat := 
  let totalArrangements := 120
  let restrictedPairsCount := 24
  totalArrangements - restrictedPairsCount

theorem marble_arrangement_count :
  countValidMarbleArrangements = 96 :=
  by
    sorry

end marble_arrangement_count_l375_37534


namespace find_a_b_find_tangent_line_l375_37545

-- Define the function f(x)
def f (a b x : ℝ) : ℝ := 2 * x ^ 3 + 3 * a * x ^ 2 + 3 * b * x + 8

-- Define the derivative of the function f(x)
def f' (a b x : ℝ) : ℝ := 6 * x ^ 2 + 6 * a * x + 3 * b

-- Define the conditions for extreme values at x=1 and x=2
def extreme_conditions (a b : ℝ) : Prop :=
  f' a b 1 = 0 ∧ f' a b 2 = 0

-- Prove the values of a and b
theorem find_a_b (a b : ℝ) (h : extreme_conditions a b) : a = -3 ∧ b = 4 :=
by sorry

-- Find the equation of the tangent line at x=0
def tangent_equation (a b : ℝ) (x y : ℝ) : Prop :=
  12 * x - y + 8 = 0

-- Prove the equation of the tangent line
theorem find_tangent_line (a b : ℝ) (h : extreme_conditions a b) : tangent_equation a b 0 8 :=
by sorry

end find_a_b_find_tangent_line_l375_37545


namespace cylinder_volume_triple_quadruple_l375_37564

theorem cylinder_volume_triple_quadruple (r h : ℝ) (V : ℝ) (π : ℝ) (original_volume : V = π * r^2 * h) 
                                         (original_volume_value : V = 8):
  ∃ V', V' = π * (3 * r)^2 * (4 * h) ∧ V' = 288 :=
by
  sorry

end cylinder_volume_triple_quadruple_l375_37564


namespace linear_system_k_value_l375_37536

theorem linear_system_k_value (x y k : ℝ) (h1 : x + 3 * y = 2 * k + 1) (h2 : x - y = 1) (h3 : x = -y) : k = -1 :=
sorry

end linear_system_k_value_l375_37536


namespace athletes_and_probability_l375_37551

-- Given conditions and parameters
def total_athletes_a := 27
def total_athletes_b := 9
def total_athletes_c := 18
def total_selected := 6
def athletes := ["A1", "A2", "A3", "A4", "A5", "A6"]

-- Definitions based on given conditions and solution steps
def selection_ratio := total_selected / (total_athletes_a + total_athletes_b + total_athletes_c)

def selected_from_a := total_athletes_a * selection_ratio
def selected_from_b := total_athletes_b * selection_ratio
def selected_from_c := total_athletes_c * selection_ratio

def pairs (l : List String) : List (String × String) :=
  (List.bind l (λ x => List.map (λ y => (x, y)) l)).filter (λ (x,y) => x < y)

def all_pairs := pairs athletes

def event_A (pair : String × String) : Bool :=
  pair.fst = "A5" ∨ pair.snd = "A5" ∨ pair.fst = "A6" ∨ pair.snd = "A6"

def favorable_event_A := all_pairs.filter event_A

noncomputable def probability_event_A := favorable_event_A.length / all_pairs.length

-- The main theorem: Number of athletes selected from each association and probability of event A
theorem athletes_and_probability : selected_from_a = 3 ∧ selected_from_b = 1 ∧ selected_from_c = 2 ∧ probability_event_A = 3/5 := by
  sorry

end athletes_and_probability_l375_37551


namespace paint_area_correct_l375_37508

-- Definitions for the conditions of the problem
def wall_height : ℕ := 10
def wall_length : ℕ := 15
def door_height : ℕ := 3
def door_length : ℕ := 5

-- Define the total area of the wall (without considering the door)
def wall_area : ℕ := wall_height * wall_length

-- Define the area of the door
def door_area : ℕ := door_height * door_length

-- Define the area that needs to be painted
def area_to_paint : ℕ := wall_area - door_area

-- The proof problem: Prove that Sandy needs to paint 135 square feet
theorem paint_area_correct : area_to_paint = 135 := 
by
  -- Sorry will be replaced with an actual proof
  sorry

end paint_area_correct_l375_37508


namespace fraction_doubled_l375_37532

theorem fraction_doubled (x y : ℝ) (h_nonzero : x + y ≠ 0) : (4 * x^2) / (2 * (x + y)) = 2 * (x^2 / (x + y)) :=
by
  sorry

end fraction_doubled_l375_37532


namespace man_reaches_home_at_11_pm_l375_37588

theorem man_reaches_home_at_11_pm :
  let start_time := 15 -- represents 3 pm in 24-hour format
  let level_speed := 4 -- km/hr
  let uphill_speed := 3 -- km/hr
  let downhill_speed := 6 -- km/hr
  let total_distance := 12 -- km
  let level_distance := 4 -- km
  let uphill_distance := 4 -- km
  let downhill_distance := 4 -- km
  let level_time := level_distance / level_speed -- time for 4 km on level ground
  let uphill_time := uphill_distance / uphill_speed -- time for 4 km uphill
  let downhill_time := downhill_distance / downhill_speed -- time for 4 km downhill
  let total_time_one_way := level_time + uphill_time + downhill_time + level_time
  let destination_time := start_time + total_time_one_way
  let return_time := destination_time + total_time_one_way
  return_time = 23 := -- represents 11 pm in 24-hour format
by
  sorry

end man_reaches_home_at_11_pm_l375_37588


namespace common_ratio_l375_37582

variable {G : Type} [LinearOrderedField G]

-- Definitions based on conditions
def geometric_seq (a₁ q : G) (n : ℕ) : G := a₁ * q^(n-1)
def sum_geometric_seq (a₁ q : G) (n : ℕ) : G :=
  if q = 1 then a₁ * n else a₁ * (1 - q^n) / (1 - q)

-- Hypotheses from conditions
variable {a₁ q : G}
variable (h1 : sum_geometric_seq a₁ q 3 = 7)
variable (h2 : sum_geometric_seq a₁ q 6 = 63)

theorem common_ratio (a₁ q : G) (h1 : sum_geometric_seq a₁ q 3 = 7)
  (h2 : sum_geometric_seq a₁ q 6 = 63) : q = 2 :=
by
  -- Proof to be completed
  sorry

end common_ratio_l375_37582


namespace biased_coin_prob_three_heads_l375_37539

def prob_heads := 1/3

theorem biased_coin_prob_three_heads : prob_heads^3 = 1/27 :=
by
  sorry

end biased_coin_prob_three_heads_l375_37539


namespace second_group_work_days_l375_37595

theorem second_group_work_days (M B : ℕ) (d1 d2 : ℕ) (H1 : M = 2 * B) 
  (H2 : (12 * M + 16 * B) * 5 = d1) (H3 : (13 * M + 24 * B) * d2 = d1) : 
  d2 = 4 :=
by
  sorry

end second_group_work_days_l375_37595


namespace even_numbers_average_l375_37521

theorem even_numbers_average (n : ℕ) (h : (n / 2 * (2 + 2 * n)) / n = 16) : n = 15 :=
by
  have hn : n ≠ 0 := sorry -- n > 0 because the first some even numbers were mentioned
  have hn_pos : 0 < n / 2 * (2 + 2 * n) := sorry -- n / 2 * (2 + 2n) > 0
  sorry

end even_numbers_average_l375_37521


namespace max_radius_of_circle_in_triangle_inscribed_l375_37510

theorem max_radius_of_circle_in_triangle_inscribed (ω : Set (ℝ × ℝ)) (hω : ∀ (P : ℝ × ℝ), P ∈ ω → P.1^2 + P.2^2 = 1)
  (O : ℝ × ℝ) (hO : O = (0, 0)) (P : ℝ × ℝ) (hP : P ∈ ω) (A : ℝ × ℝ) 
  (hA : A = (P.1, 0)) : 
  (∃ r : ℝ, r = (Real.sqrt 2 - 1) / 2) :=
by
  sorry

end max_radius_of_circle_in_triangle_inscribed_l375_37510


namespace find_x_plus_y_l375_37578

noncomputable def det3x3 (a b c d e f g h i : ℝ) : ℝ :=
  a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)

noncomputable def det2x2 (a b c d : ℝ) : ℝ :=
  a * d - b * c

theorem find_x_plus_y (x y : ℝ) (h1 : x ≠ y)
  (h2 : det3x3 2 5 10 4 x y 4 y x = 0)
  (h3 : det2x2 x y y x = 16) : x + y = 30 := by
  sorry

end find_x_plus_y_l375_37578


namespace distribution_of_balls_l375_37533

-- Definition for the problem conditions
inductive Ball : Type
| one : Ball
| two : Ball
| three : Ball
| four : Ball

inductive Box : Type
| box1 : Box
| box2 : Box
| box3 : Box

-- Function to count the number of ways to distribute the balls according to the conditions
noncomputable def num_ways_to_distribute_balls : Nat := 18

-- Theorem statement
theorem distribution_of_balls :
  num_ways_to_distribute_balls = 18 := by
  sorry

end distribution_of_balls_l375_37533


namespace rectangle_length_width_difference_l375_37519

noncomputable def difference_between_length_and_width : ℝ :=
  let x := by sorry
  let y := by sorry
  (x - y)

theorem rectangle_length_width_difference {x y : ℝ}
  (h₁ : 2 * (x + y) = 20) (h₂ : x^2 + y^2 = 10^2) :
  difference_between_length_and_width = 10 :=
  by sorry

end rectangle_length_width_difference_l375_37519


namespace rightmost_three_digits_of_7_pow_2023_l375_37553

theorem rightmost_three_digits_of_7_pow_2023 :
  (7 ^ 2023) % 1000 = 343 :=
sorry

end rightmost_three_digits_of_7_pow_2023_l375_37553


namespace murtha_total_items_at_day_10_l375_37570

-- Define terms and conditions
def num_pebbles (n : ℕ) : ℕ := n
def num_seashells (n : ℕ) : ℕ := 1 + 2 * (n - 1)

def total_pebbles (n : ℕ) : ℕ :=
  (n * (1 + n)) / 2

def total_seashells (n : ℕ) : ℕ :=
  (n * (1 + num_seashells n)) / 2

-- Define main proposition
theorem murtha_total_items_at_day_10 : total_pebbles 10 + total_seashells 10 = 155 := by
  -- Placeholder for proof
  sorry

end murtha_total_items_at_day_10_l375_37570


namespace sphere_intersection_circle_radius_l375_37548

theorem sphere_intersection_circle_radius
  (x1 y1 z1: ℝ) (x2 y2 z2: ℝ) (r1 r2: ℝ)
  (hyp1: x1 = 3) (hyp2: y1 = 5) (hyp3: z1 = 0) 
  (hyp4: r1 = 2) 
  (hyp5: x2 = 0) (hyp6: y2 = 5) (hyp7: z2 = -8) :
  r2 = Real.sqrt 59 := 
by
  sorry

end sphere_intersection_circle_radius_l375_37548


namespace find_a_given_solution_set_l375_37590

theorem find_a_given_solution_set :
  (∀ x : ℝ, 2 ≤ x ∧ x ≤ 3 ↔ x^2 + a * x + 6 ≤ 0) → a = -5 :=
by
  sorry

end find_a_given_solution_set_l375_37590


namespace probability_two_queens_or_at_least_one_king_l375_37568

/-- Prove that the probability of either drawing two queens or drawing at least one king 
    when 2 cards are selected randomly from a standard deck of 52 cards is 2/13. -/
theorem probability_two_queens_or_at_least_one_king :
  (∃ (kq pk pq : ℚ), kq = 4 ∧
                     pk = 4 ∧
                     pq = 52 ∧
                     (∃ (p : ℚ), p = (kq*(kq-1))/(pq*(pq-1)) + (pk/pq)*(pq-pk)/(pq-1) + (kq*(kq-1))/(pq*(pq-1)) ∧
                            p = 2/13)) :=
by {
  sorry
}

end probability_two_queens_or_at_least_one_king_l375_37568


namespace total_apartments_in_building_l375_37504

theorem total_apartments_in_building (A k m n : ℕ)
  (cond1 : 5 = A)
  (cond2 : 636 = (m-1) * k + n)
  (cond3 : 242 = (A-m) * k + n) :
  A * k = 985 :=
by
  sorry

end total_apartments_in_building_l375_37504


namespace root_of_unity_product_l375_37586

theorem root_of_unity_product (ω : ℂ) (h1 : ω^3 = 1) (h2 : ω ≠ 1) :
  (1 - ω + ω^2) * (1 + ω - ω^2) = 1 :=
  sorry

end root_of_unity_product_l375_37586


namespace find_divisor_for_multiple_l375_37505

theorem find_divisor_for_multiple (d : ℕ) :
  (∃ k : ℕ, k * d % 1821 = 710 ∧ k * d % 24 = 13 ∧ k * d = 3024) →
  d = 23 :=
by
  intros h
  sorry

end find_divisor_for_multiple_l375_37505


namespace find_m_l375_37543

theorem find_m (m : ℤ) (h1 : -180 ≤ m ∧ m ≤ 180) (h2 : Real.sin (m * Real.pi / 180) = Real.cos (810 * Real.pi / 180)) :
  m = 0 ∨ m = 180 :=
sorry

end find_m_l375_37543


namespace student_average_less_than_actual_average_l375_37529

variable {a b c : ℝ}

theorem student_average_less_than_actual_average (h : a < b) (h2 : b < c) :
  (a + (b + c) / 2) / 2 < (a + b + c) / 3 :=
by
  sorry

end student_average_less_than_actual_average_l375_37529


namespace range_of_b_l375_37592

variable (a b c : ℝ)

theorem range_of_b (h1 : a + b + c = 9) (h2 : a * b + b * c + c * a = 24) : 1 ≤ b ∧ b ≤ 5 :=
by
  sorry

end range_of_b_l375_37592


namespace rectangle_to_square_area_ratio_is_24_25_l375_37589

noncomputable def rectangle_to_square_area_ratio
  (s : ℝ) -- length of side of square S
  (longer_side : ℝ := 1.2 * s) -- longer side of rectangle R
  (shorter_side : ℝ := 0.8 * s) -- shorter side of rectangle R
  (area_R : ℝ := longer_side * shorter_side) -- area of rectangle R
  (area_S : ℝ := s^2) -- area of square S
  : ℝ := 
  area_R / area_S

theorem rectangle_to_square_area_ratio_is_24_25 
  (s : ℝ)
  : rectangle_to_square_area_ratio s = 24 / 25 :=
by 
  sorry

end rectangle_to_square_area_ratio_is_24_25_l375_37589


namespace min_guests_at_banquet_l375_37544

theorem min_guests_at_banquet (total_food : ℕ) (max_food_per_guest : ℕ) : 
  total_food = 323 ∧ max_food_per_guest = 2 → 
  (∀ guests : ℕ, guests * max_food_per_guest >= total_food) → 
  (∃ g : ℕ, g = 162) :=
by
  -- Assuming total food and max food per guest
  intro h_cons
  -- Mathematical proof steps would go here, skipping with sorry
  sorry

end min_guests_at_banquet_l375_37544


namespace production_line_B_units_l375_37537

theorem production_line_B_units (total_units : ℕ) (A_units B_units C_units : ℕ) 
  (h1 : total_units = 16800)
  (h2 : ∃ d : ℕ, A_units + d = B_units ∧ B_units + d = C_units) :
  B_units = 5600 := 
sorry

end production_line_B_units_l375_37537


namespace discount_comparison_l375_37500

noncomputable def final_price (P : ℝ) (d1 d2 d3 : ℝ) : ℝ :=
  P * (1 - d1) * (1 - d2) * (1 - d3)

theorem discount_comparison (P : ℝ) (d11 d12 d13 d21 d22 d23 : ℝ) :
  P = 20000 →
  d11 = 0.25 → d12 = 0.15 → d13 = 0.10 →
  d21 = 0.30 → d22 = 0.10 → d23 = 0.10 →
  final_price P d11 d12 d13 - final_price P d21 d22 d23 = 135 :=
by
  intros
  sorry

end discount_comparison_l375_37500


namespace additional_fertilizer_on_final_day_l375_37587

noncomputable def normal_usage_per_day : ℕ := 2
noncomputable def total_days : ℕ := 9
noncomputable def total_fertilizer_used : ℕ := 22

theorem additional_fertilizer_on_final_day :
  total_fertilizer_used - (normal_usage_per_day * total_days) = 4 := by
  sorry

end additional_fertilizer_on_final_day_l375_37587


namespace fruit_seller_original_apples_l375_37518

variable (x : ℝ)

theorem fruit_seller_original_apples (h : 0.60 * x = 420) : x = 700 := by
  sorry

end fruit_seller_original_apples_l375_37518


namespace percentage_difference_between_M_and_J_is_34_74_percent_l375_37512

-- Definitions of incomes and relationships
variables (J T M : ℝ)
variables (h1 : T = 0.80 * J)
variables (h2 : M = 1.60 * T)

-- Definitions of savings and expenses
variables (Msavings : ℝ := 0.15 * M)
variables (Mexpenses : ℝ := 0.25 * M)
variables (Tsavings : ℝ := 0.12 * T)
variables (Texpenses : ℝ := 0.30 * T)
variables (Jsavings : ℝ := 0.18 * J)
variables (Jexpenses : ℝ := 0.20 * J)

-- Total savings and expenses
variables (Mtotal : ℝ := Msavings + Mexpenses)
variables (Jtotal : ℝ := Jsavings + Jexpenses)

-- Prove the percentage difference between Mary's and Juan's total savings and expenses combined
theorem percentage_difference_between_M_and_J_is_34_74_percent :
  M = 1.28 * J → 
  Mtotal = 0.40 * M →
  Jtotal = 0.38 * J →
  ( (Mtotal - Jtotal) / Jtotal ) * 100 = 34.74 :=
by
  sorry

end percentage_difference_between_M_and_J_is_34_74_percent_l375_37512


namespace percentage_passed_both_l375_37572

-- Define the percentages of failures
def percentage_failed_hindi : ℕ := 34
def percentage_failed_english : ℕ := 44
def percentage_failed_both : ℕ := 22

-- Statement to prove
theorem percentage_passed_both : 
  (100 - (percentage_failed_hindi + percentage_failed_english - percentage_failed_both)) = 44 := by
  sorry

end percentage_passed_both_l375_37572


namespace sum_of_four_triangles_l375_37560

theorem sum_of_four_triangles (x y : ℝ) (h1 : 3 * x + 2 * y = 27) (h2 : 2 * x + 3 * y = 23) : 4 * y = 12 :=
sorry

end sum_of_four_triangles_l375_37560


namespace field_dimension_m_l375_37516

theorem field_dimension_m (m : ℝ) (h : (3 * m + 8) * (m - 3) = 80) : m = 6.057 := by
  sorry

end field_dimension_m_l375_37516


namespace one_leg_divisible_by_3_l375_37509

theorem one_leg_divisible_by_3 (a b c : ℕ) (h : a^2 + b^2 = c^2) : (3 ∣ a) ∨ (3 ∣ b) :=
by sorry

end one_leg_divisible_by_3_l375_37509


namespace problem_statement_l375_37527

variable (m n : ℝ)
noncomputable def sqrt_2_minus_1_inv := (Real.sqrt 2 - 1)⁻¹
noncomputable def sqrt_2_plus_1_inv := (Real.sqrt 2 + 1)⁻¹

theorem problem_statement 
  (hm : m = sqrt_2_minus_1_inv) 
  (hn : n = sqrt_2_plus_1_inv) : 
  m + n = 2 * Real.sqrt 2 := 
sorry

end problem_statement_l375_37527


namespace find_m_l375_37561

theorem find_m (S : ℕ → ℝ) (m : ℝ) (h : ∀ n, S n = m * 2^(n-1) - 3) : m = 6 :=
by
  sorry

end find_m_l375_37561


namespace solve_system_of_inequalities_l375_37540

theorem solve_system_of_inequalities (x : ℝ) :
  ( (x - 2) / (x - 1) < 1 ) ∧ ( -x^2 + x + 2 < 0 ) → x > 2 :=
by
  sorry

end solve_system_of_inequalities_l375_37540
