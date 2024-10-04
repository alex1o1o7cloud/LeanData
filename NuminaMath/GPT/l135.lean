import Mathlib

namespace hexagon_perimeter_l135_135175

theorem hexagon_perimeter (side_length : ℝ) (sides : ℕ) (h_sides : sides = 6) (h_side_length : side_length = 10) :
  sides * side_length = 60 :=
by
  rw [h_sides, h_side_length]
  norm_num

end hexagon_perimeter_l135_135175


namespace arrangement_count_l135_135605

def number_of_arrangements (slots total_geometry total_number_theory : ℕ) : ℕ :=
  Nat.choose slots total_geometry

theorem arrangement_count :
  number_of_arrangements 8 5 3 = 56 := 
by
  sorry

end arrangement_count_l135_135605


namespace students_at_year_end_l135_135116

theorem students_at_year_end (initial_students left_students new_students end_students : ℕ)
  (h_initial : initial_students = 31)
  (h_left : left_students = 5)
  (h_new : new_students = 11)
  (h_end : end_students = initial_students - left_students + new_students) :
  end_students = 37 :=
by
  sorry

end students_at_year_end_l135_135116


namespace chemistry_class_size_l135_135492

theorem chemistry_class_size
  (total_students : ℕ)
  (chem_bio_both : ℕ)
  (bio_students : ℕ)
  (chem_students : ℕ)
  (both_students : ℕ)
  (H1 : both_students = 8)
  (H2 : bio_students + chem_students + both_students = total_students)
  (H3 : total_students = 70)
  (H4 : chem_students = 2 * (bio_students + both_students)) :
  chem_students + both_students = 52 :=
by
  sorry

end chemistry_class_size_l135_135492


namespace mean_value_of_angles_mean_value_of_interior_angles_of_quadrilateral_l135_135347

-- Defining the sum of interior angles of a quadrilateral.
def sum_of_interior_angles (quad : Type) [IsQuadrilateral quad] : ℝ :=
  360

-- Defining the number of angles in a quadrilateral.
def number_of_angles (quad : Type) [IsQuadrilateral quad] : ℕ :=
  4

-- The theorem we want to prove: the mean value of the measures of the four interior angles of any quadrilateral.
theorem mean_value_of_angles (quad : Type) [IsQuadrilateral quad] : ℝ :=
  sum_of_interior_angles quad / number_of_angles quad
 
-- Proving the theorem.
theorem mean_value_of_interior_angles_of_quadrilateral (quad : Type) [IsQuadrilateral quad]:
  mean_value_of_angles quad = 90 :=
by
  sorry

end mean_value_of_angles_mean_value_of_interior_angles_of_quadrilateral_l135_135347


namespace smallest_possible_time_for_travel_l135_135210

theorem smallest_possible_time_for_travel :
  ∃ t : ℝ, (∀ D M P : ℝ, D = 6 → M = 6 → P = 6 → 
    ∀ motorcycle_speed distance : ℝ, motorcycle_speed = 90 → distance = 135 → 
    t < 3.9) :=
  sorry

end smallest_possible_time_for_travel_l135_135210


namespace eighth_box_contains_65_books_l135_135315

theorem eighth_box_contains_65_books (total_books boxes first_seven_books per_box eighth_box : ℕ) :
  total_books = 800 →
  boxes = 8 →
  first_seven_books = 7 →
  per_box = 105 →
  eighth_box = total_books - (first_seven_books * per_box) →
  eighth_box = 65 := by
  sorry

end eighth_box_contains_65_books_l135_135315


namespace perimeter_C_is_40_l135_135313

noncomputable def perimeter_of_figure_C (x y : ℝ) : ℝ :=
  2 * x + 6 * y

theorem perimeter_C_is_40 (x y : ℝ) (h1 : 6 * x + 2 * y = 56) (h2 : 4 * x + 6 * y = 56) :
  perimeter_of_figure_C x y = 40 :=
by
  -- Define initial conditions
  have eq1 : 3 * x + y = 28, by { rw [mul_assoc, mul_comm 3 x, add_assoc], exact (eq.div h1 2) }
  have eq2 : 2 * x + 3 * y = 28, by { rw [mul_assoc, mul_comm 2 x, add_assoc], exact (eq.div h2 2) }
  -- Assume the solutions are obtained from here
  have sol_x : x = 8, by sorry
  have sol_y : y = 4, by sorry
  -- Calculate the perimeter of figure C
  rw [perimeter_of_figure_C, sol_x, sol_y]
  norm_num
  trivial

-- Test case to ensure the code builds successfully
#eval perimeter_of_figure_C 8 4  -- Expected output: 40

end perimeter_C_is_40_l135_135313


namespace find_m_l135_135238

theorem find_m (m : ℕ) (h1 : 0 ≤ m ∧ m ≤ 9) (h2 : (8 + 4 + 5 + 9) - (6 + m + 3 + 7) % 11 = 0) : m = 9 :=
by
  sorry

end find_m_l135_135238


namespace julian_needs_more_legos_l135_135249

-- Definitions based on the conditions
def legos_julian_has := 400
def legos_per_airplane := 240
def number_of_airplanes := 2

-- Calculate the total number of legos required for two airplane models
def total_legos_needed := legos_per_airplane * number_of_airplanes

-- Calculate the number of additional legos Julian needs
def additional_legos_needed := total_legos_needed - legos_julian_has

-- Statement that needs to be proven
theorem julian_needs_more_legos : additional_legos_needed = 80 := by
  sorry

end julian_needs_more_legos_l135_135249


namespace water_used_for_plates_and_clothes_is_48_l135_135625

noncomputable def waterUsedToWashPlatesAndClothes : ℕ := 
  let barrel1 := 65 
  let barrel2 := (75 * 80) / 100 
  let barrel3 := (45 * 60) / 100 
  let totalCollected := barrel1 + barrel2 + barrel3
  let usedForCars := 7 * 2
  let usedForPlants := 15
  let usedForDog := 10
  let usedForCooking := 5
  let usedForBathing := 12
  let totalUsed := usedForCars + usedForPlants + usedForDog + usedForCooking + usedForBathing
  let remainingWater := totalCollected - totalUsed
  remainingWater / 2

theorem water_used_for_plates_and_clothes_is_48 : 
  waterUsedToWashPlatesAndClothes = 48 :=
by
  sorry

end water_used_for_plates_and_clothes_is_48_l135_135625


namespace mean_value_of_quadrilateral_angles_l135_135448

theorem mean_value_of_quadrilateral_angles :
  let sum_of_angles := 360 in
  let number_of_angles := 4 in
  sum_of_angles / number_of_angles = 90 :=
by
  sorry

end mean_value_of_quadrilateral_angles_l135_135448


namespace geometric_sequence_sum_l135_135794

-- Definition of a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

-- Main statement to prove
theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ)
  (h1 : is_geometric_sequence a q)
  (h2 : a 1 + a 2 = 40)
  (h3 : a 3 + a 4 = 60) :
  a 7 + a 8 = 135 :=
sorry

end geometric_sequence_sum_l135_135794


namespace quadratic_has_one_real_root_l135_135949

theorem quadratic_has_one_real_root (m : ℝ) (h : (6 * m)^2 - 4 * 1 * 4 * m = 0) : m = 4 / 9 :=
by sorry

end quadratic_has_one_real_root_l135_135949


namespace largest_common_in_range_l135_135189

-- Definitions for the problem's conditions
def first_seq (n : ℕ) : ℕ := 3 + 8 * n
def second_seq (m : ℕ) : ℕ := 5 + 9 * m

-- Statement of the theorem we are proving
theorem largest_common_in_range : 
  ∃ n m : ℕ, first_seq n = second_seq m ∧ 1 ≤ first_seq n ∧ first_seq n ≤ 200 ∧ first_seq n = 131 := by
  sorry

end largest_common_in_range_l135_135189


namespace cafeteria_extra_fruits_l135_135565

def num_apples_red := 75
def num_apples_green := 35
def num_oranges := 40
def num_bananas := 20
def num_students := 17

def total_fruits := num_apples_red + num_apples_green + num_oranges + num_bananas
def fruits_taken_by_students := num_students
def extra_fruits := total_fruits - fruits_taken_by_students

theorem cafeteria_extra_fruits : extra_fruits = 153 := by
  -- proof goes here
  sorry

end cafeteria_extra_fruits_l135_135565


namespace perimeter_C_correct_l135_135302

variables (x y : ℕ)

def perimeter_A (x y : ℕ) := 6 * x + 2 * y
def perimeter_B (x y : ℕ) := 4 * x + 6 * y
def perimeter_C (x y : ℕ) := 2 * x + 6 * y

theorem perimeter_C_correct (x y : ℕ) (h1 : 6 * x + 2 * y = 56) (h2 : 4 * x + 6 * y = 56) :
  2 * x + 6 * y = 40 :=
sorry

end perimeter_C_correct_l135_135302


namespace sum_abcd_l135_135251

variable (a b c d x : ℝ)

axiom eq1 : a + 2 = x
axiom eq2 : b + 3 = x
axiom eq3 : c + 4 = x
axiom eq4 : d + 5 = x
axiom eq5 : a + b + c + d + 10 = x

theorem sum_abcd : a + b + c + d = -26 / 3 :=
by
  -- We state the condition given in the problem
  sorry

end sum_abcd_l135_135251


namespace seashells_total_l135_135025

def seashells_sam : ℕ := 18
def seashells_mary : ℕ := 47
def seashells_john : ℕ := 32
def seashells_emily : ℕ := 26

theorem seashells_total : seashells_sam + seashells_mary + seashells_john + seashells_emily = 123 := by
    sorry

end seashells_total_l135_135025


namespace lcm_of_fractions_is_correct_l135_135047

-- Define denominators
def denom1 (x : ℚ) : ℚ := 5 * x
def denom2 (x : ℚ) : ℚ := 10 * x
def denom3 (x : ℚ) : ℚ := 15 * x

-- Define the fractions
def frac1 (x : ℚ) : ℚ := 1 / denom1 x
def frac2 (x : ℚ) : ℚ := 1 / denom2 x
def frac3 (x : ℚ) : ℚ := 1 / denom3 x

-- Define the least common multiple (LCM) of the denominators
def lcm_den (x : ℚ) : ℚ := 30 * x

-- The statement to be proven
theorem lcm_of_fractions_is_correct (x : ℚ) :
  ∃ y : ℚ, y = lcm_den x ∧ (frac1 x ≤ 1 / y) ∧ (frac2 x ≤ 1 / y) ∧ (frac3 x ≤ 1 / y) :=
sorry

end lcm_of_fractions_is_correct_l135_135047


namespace mean_value_interior_angles_quadrilateral_l135_135373

theorem mean_value_interior_angles_quadrilateral :
  let n := 4 in
  (n - 2) * 180 / n = 90 :=
by
  let n := 4
  sorry

end mean_value_interior_angles_quadrilateral_l135_135373


namespace g_five_eq_248_l135_135842

-- We define g, and assume it meets the conditions described.
variable (g : ℤ → ℤ)

-- Condition 1: g(1) > 1
axiom g_one_gt_one : g 1 > 1

-- Condition 2: Functional equation for g
axiom g_funct_eq (x y : ℤ) : g (x + y) + x * g y + y * g x = g x * g y + x + y + x * y

-- Condition 3: Recursive relationship for g
axiom g_recur_eq (x : ℤ) : 3 * g x = g (x + 1) + 2 * x - 1

-- Theorem we want to prove
theorem g_five_eq_248 : g 5 = 248 := by
  sorry

end g_five_eq_248_l135_135842


namespace parabola_x_intercept_unique_l135_135653

theorem parabola_x_intercept_unique : ∃! (x : ℝ), ∀ (y : ℝ), x = -y^2 + 2*y + 3 → x = 3 :=
by
  sorry

end parabola_x_intercept_unique_l135_135653


namespace parallel_vectors_y_value_l135_135514

theorem parallel_vectors_y_value 
  (y : ℝ) 
  (a : ℝ × ℝ := (6, 2)) 
  (b : ℝ × ℝ := (y, 3)) 
  (h : ∃ k : ℝ, b = k • a) : y = 9 :=
sorry

end parallel_vectors_y_value_l135_135514


namespace place_value_ratio_l135_135924

theorem place_value_ratio :
  let val_6 := 1000
  let val_2 := 0.1
  val_6 / val_2 = 10000 :=
by
  -- the proof would go here
  sorry

end place_value_ratio_l135_135924


namespace houses_in_lawrence_county_l135_135750

theorem houses_in_lawrence_county 
  (houses_before_boom : ℕ := 1426) 
  (houses_built_during_boom : ℕ := 574) 
  : houses_before_boom + houses_built_during_boom = 2000 := 
by 
  sorry

end houses_in_lawrence_county_l135_135750


namespace problem_statement_l135_135589

noncomputable def two_arccos_equals_arcsin : Prop :=
  2 * Real.arccos (3 / 5) = Real.arcsin (24 / 25)

theorem problem_statement : two_arccos_equals_arcsin :=
  sorry

end problem_statement_l135_135589


namespace cos_beta_value_l135_135177

theorem cos_beta_value (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2)
  (hα_cos : Real.cos α = 4 / 5) (hαβ_cos : Real.cos (α + β) = -16 / 65) : 
  Real.cos β = 5 / 13 := 
sorry

end cos_beta_value_l135_135177


namespace truncated_trigonal_pyramid_circumscribed_sphere_l135_135742

theorem truncated_trigonal_pyramid_circumscribed_sphere
  (h R_1 R_2 : ℝ)
  (O_1 T_1 O_2 T_2 : ℝ)
  (circumscribed : ∃ r : ℝ, h = 2 * r)
  (sphere_touches_lower_base : ∀ P, dist P T_1 = r)
  (sphere_touches_upper_base : ∀ Q, dist Q T_2 = r)
  (dist_O1_T1 : ℝ)
  (dist_O2_T2 : ℝ) :
  R_1 * R_2 * h^2 = (R_1^2 - dist_O1_T1^2) * (R_2^2 - dist_O2_T2^2) :=
sorry

end truncated_trigonal_pyramid_circumscribed_sphere_l135_135742


namespace mean_of_quadrilateral_angles_l135_135352

theorem mean_of_quadrilateral_angles :
  ∀ (angles : List ℝ), angles.length = 4 → angles.sum = 360 → angles.sum / angles.length = 90 :=
by
  intros
  sorry

end mean_of_quadrilateral_angles_l135_135352


namespace points_on_curve_is_parabola_l135_135268

theorem points_on_curve_is_parabola (X Y : ℝ) (h : Real.sqrt X + Real.sqrt Y = 1) :
  ∃ a b c : ℝ, Y = a * X^2 + b * X + c :=
sorry

end points_on_curve_is_parabola_l135_135268


namespace coordinates_of_C_l135_135160

structure Point :=
  (x : Int)
  (y : Int)

def reflect_over_x_axis (p : Point) : Point :=
  {x := p.x, y := -p.y}

def reflect_over_y_axis (p : Point) : Point :=
  {x := -p.x, y := p.y}

def C : Point := {x := 2, y := 2}

noncomputable def C'_reflected_x := reflect_over_x_axis C
noncomputable def C''_reflected_y := reflect_over_y_axis C'_reflected_x

theorem coordinates_of_C'' : C''_reflected_y = {x := -2, y := -2} :=
by
  sorry

end coordinates_of_C_l135_135160


namespace value_of_y_l135_135237

theorem value_of_y (x y : ℤ) (h1 : x^2 - 3 * x + 6 = y + 2) (h2 : x = -8) : y = 92 :=
by
  sorry

end value_of_y_l135_135237


namespace mean_of_quadrilateral_angles_l135_135406

theorem mean_of_quadrilateral_angles : 
  (∀ (a b c d : ℝ), a + b + c + d = 360) → 90 :=
by
  intro h
  sorry

end mean_of_quadrilateral_angles_l135_135406


namespace beth_friends_l135_135613

theorem beth_friends (F : ℝ) (h1 : 4 / F + 6 = 6.4) : F = 10 :=
by
  sorry

end beth_friends_l135_135613


namespace bill_score_l135_135584

variable {J B S : ℕ}

theorem bill_score (h1 : B = J + 20) (h2 : B = S / 2) (h3 : J + B + S = 160) : B = 45 :=
sorry

end bill_score_l135_135584


namespace intersection_point_of_line_and_plane_l135_135216

/-- The coordinates of the intersection point of the line and the plane are (4, -3, 8) -/
theorem intersection_point_of_line_and_plane :
  let x := 4, y := -3, z := 8 in
  (∃ t : ℝ, 
    x = 3 + t ∧ 
    y = -2 - t ∧ 
    z = 8) ∧ 
  5 * x + 9 * y + 4 * z - 25 = 0 :=
by 
  sorry

end intersection_point_of_line_and_plane_l135_135216


namespace arrangement_count_l135_135606

def number_of_arrangements (slots total_geometry total_number_theory : ℕ) : ℕ :=
  Nat.choose slots total_geometry

theorem arrangement_count :
  number_of_arrangements 8 5 3 = 56 := 
by
  sorry

end arrangement_count_l135_135606


namespace mean_value_of_quadrilateral_angles_l135_135387

theorem mean_value_of_quadrilateral_angles (a b c d : ℝ) (h_angles_sum : a + b + c + d = 360) : (a + b + c + d) / 4 = 90 := 
by
  sorry

end mean_value_of_quadrilateral_angles_l135_135387


namespace solve_for_a4b4_l135_135211

theorem solve_for_a4b4 (
    a1 a2 a3 a4 b1 b2 b3 b4 : ℝ
) (h1 : a1 * b1 + a2 * b3 = 1) 
  (h2 : a1 * b2 + a2 * b4 = 0) 
  (h3 : a3 * b1 + a4 * b3 = 0)
  (h4 : a3 * b2 + a4 * b4 = 1)
  (h5 : a2 * b3 = 7) : 
  a4 * b4 = -6 :=
sorry

end solve_for_a4b4_l135_135211


namespace find_g5_l135_135857

-- Define the function g on integers
def g : ℤ → ℤ := sorry

-- Define the conditions
axiom cond1 : g 1 > 1
axiom cond2 : ∀ (x y : ℤ), g (x + y) + x * g y + y * g x = g x * g y + x + y + x * y
axiom cond3 : ∀ (x : ℤ), 3 * g x = g (x + 1) + 2 * x - 1

-- The statement we need to prove
theorem find_g5 : g 5 = 248 := by
  sorry

end find_g5_l135_135857


namespace arrival_time_l135_135051

def minutes_to_hours (minutes : ℕ) : ℕ := minutes / 60

theorem arrival_time (departure_time : ℕ) (stop1 stop2 stop3 travel_hours : ℕ) (stops_total_time := stop1 + stop2 + stop3) (stops_total_hours := minutes_to_hours stops_total_time) : 
  departure_time = 7 → 
  stop1 = 25 → 
  stop2 = 10 → 
  stop3 = 25 → 
  travel_hours = 12 → 
  (departure_time + (travel_hours - stops_total_hours)) % 24 = 18 :=
by
  sorry

end arrival_time_l135_135051


namespace smallest_constant_N_l135_135092

-- Given that a, b, c are sides of a triangle and in arithmetic progression, prove that
-- (a^2 + b^2 + c^2) / (ab + bc + ca) ≥ 1.

theorem smallest_constant_N
  (a b c : ℝ)
  (habc : a + b > c ∧ a + c > b ∧ b + c > a) -- Triangle inequality
  (hap : ∃ d : ℝ, b = a + d ∧ c = a + 2 * d) -- Arithmetic progression
  : (a^2 + b^2 + c^2) / (a * b + b * c + c * a) ≥ 1 := 
sorry

end smallest_constant_N_l135_135092


namespace escalator_steps_l135_135490

theorem escalator_steps (T : ℝ) (E : ℝ) (N : ℝ) (h1 : N - 11 = 2 * (N - 29)) : N = 47 :=
by
  sorry

end escalator_steps_l135_135490


namespace cupcakes_left_l135_135013

def num_packages : ℝ := 3.5
def cupcakes_per_package : ℝ := 7
def cupcakes_eaten : ℝ := 5.75

theorem cupcakes_left :
  num_packages * cupcakes_per_package - cupcakes_eaten = 18.75 :=
by
  sorry

end cupcakes_left_l135_135013


namespace June_sweets_count_l135_135076

variable (A M J : ℕ)

-- condition: May has three-quarters of the number of sweets that June has
def May_sweets := M = (3/4) * J

-- condition: April has two-thirds of the number of sweets that May has
def April_sweets := A = (2/3) * M

-- condition: April, May, and June have 90 sweets between them
def Total_sweets := A + M + J = 90

-- proof problem: How many sweets does June have?
theorem June_sweets_count : 
  May_sweets M J ∧ April_sweets A M ∧ Total_sweets A M J → J = 40 :=
by
  sorry

end June_sweets_count_l135_135076


namespace cannot_be_square_of_binomial_B_l135_135576

theorem cannot_be_square_of_binomial_B (x y m n : ℝ) :
  (∃ (a b : ℝ), (3*x + 7*y) * (3*x - 7*y) = a^2 - b^2) ∧
  (∃ (a b : ℝ), ( -0.2*x - 0.3) * ( -0.2*x + 0.3) = a^2 - b^2) ∧
  (∃ (a b : ℝ), ( -3*n - m*n) * ( 3*n - m*n) = a^2 - b^2) ∧
  ¬(∃ (a b : ℝ), ( 5*m - n) * ( n - 5*m) = a^2 - b^2) :=
by
  sorry

end cannot_be_square_of_binomial_B_l135_135576


namespace inequality_proof_l135_135537

theorem inequality_proof (x y z : ℝ) (hx : 1 ≤ x) (hy : 1 ≤ y) (hz : 1 ≤ z) 
  (h : 1 / (x^2 - 1) + 1 / (y^2 - 1) + 1 / (z^2 - 1) = 1) :
  1 / (x + 1) + 1 / (y + 1) + 1 / (z + 1) ≤ 1 := 
by {
  sorry
}

end inequality_proof_l135_135537


namespace p_necessary_not_sufficient_for_q_l135_135658

def condition_p (x : ℝ) : Prop := x > 2
def condition_q (x : ℝ) : Prop := x > 3

theorem p_necessary_not_sufficient_for_q (x : ℝ) :
  (∀ (x : ℝ), condition_q x → condition_p x) ∧ ¬(∀ (x : ℝ), condition_p x → condition_q x) :=
by
  sorry

end p_necessary_not_sufficient_for_q_l135_135658


namespace arrange_books_l135_135607

-- Definition of the problem
def total_books : ℕ := 5 + 3

-- Definition of the combination function
def combination (n k : ℕ) : ℕ :=
  n.choose k

-- Prove that arranging 5 copies of Introduction to Geometry and 
-- 3 copies of Introduction to Number Theory into total_books positions can be done in 56 ways.
theorem arrange_books : combination total_books 5 = 56 := by
  sorry

end arrange_books_l135_135607


namespace joe_new_average_score_after_dropping_lowest_l135_135001

theorem joe_new_average_score_after_dropping_lowest 
  (initial_average : ℕ)
  (lowest_score : ℕ)
  (num_tests : ℕ)
  (new_num_tests : ℕ)
  (total_points : ℕ)
  (new_total_points : ℕ)
  (new_average : ℕ) :
  initial_average = 70 →
  lowest_score = 55 →
  num_tests = 4 →
  new_num_tests = 3 →
  total_points = num_tests * initial_average →
  new_total_points = total_points - lowest_score →
  new_average = new_total_points / new_num_tests →
  new_average = 75 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end joe_new_average_score_after_dropping_lowest_l135_135001


namespace hours_worked_each_day_l135_135190

-- Given conditions
def total_hours_worked : ℕ := 18
def number_of_days_worked : ℕ := 6

-- Statement to prove
theorem hours_worked_each_day : total_hours_worked / number_of_days_worked = 3 := by
  sorry

end hours_worked_each_day_l135_135190


namespace mean_value_of_quadrilateral_angles_l135_135332

-- Statement of the problem: mean value of interior angles in any quadrilateral is 90°
theorem mean_value_of_quadrilateral_angles : 
  (∀ (a b c d : ℝ), a + b + c + d = 360 → (a + b + c + d) / 4 = 90) :=
by
  sorry

end mean_value_of_quadrilateral_angles_l135_135332


namespace PolyCoeffInequality_l135_135580

open Real

variable (p q : ℝ[X])
variable (a : ℝ)
variable (n : ℕ)
variable (h k : ℝ)
variable (deg_p : p.degree = n)
variable (deg_q : q.degree = n - 1)
variable (hp : ∀ i, i ≤ n → |p.coeff i| ≤ h)
variable (hq : ∀ i, i < n → |q.coeff i| ≤ k)
variable (hpq : p = (X + C a) * q)

theorem PolyCoeffInequality : k ≤ h^n := by
  sorry

end PolyCoeffInequality_l135_135580


namespace ratio_of_horses_to_cows_l135_135823

/-- Let H and C be the initial number of horses and cows respectively.
Given that:
1. (H - 15) / (C + 15) = 7 / 3,
2. H - 15 = C + 75,
prove that the initial ratio of horses to cows is 4:1. -/
theorem ratio_of_horses_to_cows (H C : ℕ) 
  (h1 : (H - 15 : ℚ) / (C + 15 : ℚ) = 7 / 3)
  (h2 : H - 15 = C + 75) :
  H / C = 4 :=
by
  sorry

end ratio_of_horses_to_cows_l135_135823


namespace counterexample_not_prime_implies_prime_l135_135622

theorem counterexample_not_prime_implies_prime (n : ℕ) (h₁ : ¬Nat.Prime n) (h₂ : n = 27) : ¬Nat.Prime (n - 2) :=
by
  sorry

end counterexample_not_prime_implies_prime_l135_135622


namespace third_candle_remaining_fraction_l135_135042

theorem third_candle_remaining_fraction (t : ℝ) 
  (h1 : 0 < t)
  (second_candle_fraction_remaining : ℝ := 2/5)
  (third_candle_fraction_remaining : ℝ := 3/7)
  (second_candle_burned_fraction : ℝ := 3/5)
  (third_candle_burned_fraction : ℝ := 4/7)
  (second_candle_burn_rate : ℝ := 3 / (5 * t))
  (third_candle_burn_rate : ℝ := 4 / (7 * t))
  (remaining_burn_time_second : ℝ := (2 * t) / 3)
  (third_candle_burned_in_remaining_time : ℝ := (2 * t * 4) / (3 * 7 * t))
  (common_denominator_third : ℝ := 21)
  (converted_third_candle_fraction_remaining : ℝ := 9 / 21)
  (third_candle_fraction_subtracted : ℝ := 8 / 21) :
  (converted_third_candle_fraction_remaining - third_candle_fraction_subtracted) = 1 / 21 := by
  sorry

end third_candle_remaining_fraction_l135_135042


namespace train_speed_is_300_kmph_l135_135488

noncomputable def train_speed (length : ℝ) (time : ℝ) : ℝ :=
  (length / 1000) / (time / 3600)

theorem train_speed_is_300_kmph :
  train_speed 1250 15 = 300 := by
  sorry

end train_speed_is_300_kmph_l135_135488


namespace arithmetic_sequence_sum_l135_135566

theorem arithmetic_sequence_sum :
  ∀ (a_2 a_3 : ℤ) (S : ℕ → ℤ),
  a_2 = 1 → a_3 = 3 →
  (∀ n, S n = n * (2 * a_1 + (n - 1) * d) / 2) →
  S 4 = 8 :=
by
  intros a_2 a_3 S h_a2 h_a3 h_S
  sorry

end arithmetic_sequence_sum_l135_135566


namespace trader_profit_l135_135053

theorem trader_profit (P : ℝ) :
  let buy_price := 0.80 * P
  let sell_price := 1.20 * P
  sell_price - P = 0.20 * P := 
by
  sorry

end trader_profit_l135_135053


namespace mean_of_quadrilateral_angles_l135_135358

theorem mean_of_quadrilateral_angles :
  ∀ (angles : List ℝ), angles.length = 4 → angles.sum = 360 → angles.sum / angles.length = 90 :=
by
  intros
  sorry

end mean_of_quadrilateral_angles_l135_135358


namespace sin_2alpha_plus_pi_over_3_cos_beta_minus_pi_over_6_l135_135100

variable (α β : ℝ)
variable (hα : α < π/2) (hβ : β < π/2) -- acute angles
variable (h1 : Real.cos (α + π/6) = 3/5)
variable (h2 : Real.cos (α + β) = -Real.sqrt 5 / 5)

theorem sin_2alpha_plus_pi_over_3 :
  Real.sin (2 * α + π/3) = 24 / 25 :=
by
  sorry

theorem cos_beta_minus_pi_over_6 :
  Real.cos (β - π/6) = Real.sqrt 5 / 5 :=
by
  sorry

end sin_2alpha_plus_pi_over_3_cos_beta_minus_pi_over_6_l135_135100


namespace billiard_ball_returns_l135_135071

theorem billiard_ball_returns
  (w h : ℕ)
  (launch_angle : ℝ)
  (reflect_angle : ℝ)
  (start_A : ℝ × ℝ)
  (h_w : w = 2021)
  (h_h : h = 4300)
  (h_launch : launch_angle = 45)
  (h_reflect : reflect_angle = 45)
  (h_in_rect : ∀ (x y : ℝ), 0 ≤ x ∧ x ≤ 2021 ∧ 0 ≤ y ∧ y ≤ 4300) :
  ∃ (bounces : ℕ), bounces = 294 :=
by
  sorry

end billiard_ball_returns_l135_135071


namespace molecular_weight_of_ammonium_bromide_l135_135752

-- Define the atomic weights for the elements.
def atomic_weight_N : ℝ := 14.01
def atomic_weight_H : ℝ := 1.01
def atomic_weight_Br : ℝ := 79.90

-- Define the molecular weight of ammonium bromide based on its formula NH4Br.
def molecular_weight_NH4Br : ℝ := (1 * atomic_weight_N) + (4 * atomic_weight_H) + (1 * atomic_weight_Br)

-- State the theorem that the molecular weight of ammonium bromide is approximately 97.95 g/mol.
theorem molecular_weight_of_ammonium_bromide :
  molecular_weight_NH4Br = 97.95 :=
by
  -- The following line is a placeholder to mark the theorem as correct without proving it.
  sorry

end molecular_weight_of_ammonium_bromide_l135_135752


namespace jane_average_speed_l135_135277

theorem jane_average_speed :
  let total_distance := 200
  let total_time := 6
  total_distance / total_time = 100 / 3 :=
by
  sorry

end jane_average_speed_l135_135277


namespace walkway_area_correct_l135_135615

-- Define the dimensions of one flower bed
def flower_bed_width := 8
def flower_bed_height := 3

-- Define the number of flower beds and the width of the walkways
def num_flowers_horizontal := 3
def num_flowers_vertical := 4
def walkway_width := 2

-- Calculate the total dimension of the garden including both flower beds and walkways
def total_garden_width := (num_flowers_horizontal * flower_bed_width) + ((num_flowers_horizontal + 1) * walkway_width)
def total_garden_height := (num_flowers_vertical * flower_bed_height) + ((num_flowers_vertical + 1) * walkway_width)

-- Calculate the total area of the garden and the total area of the flower beds
def total_garden_area := total_garden_width * total_garden_height
def total_flower_bed_area := (flower_bed_width * flower_bed_height) * (num_flowers_horizontal * num_flowers_vertical)

-- Calculate the total area of the walkways in the garden
def total_walkway_area := total_garden_area - total_flower_bed_area

-- The statement to be proven:
theorem walkway_area_correct : total_walkway_area = 416 := by
  sorry

end walkway_area_correct_l135_135615


namespace find_value_of_expression_l135_135779

theorem find_value_of_expression (x y : ℝ) (h1 : |x| = 2) (h2 : |y| = 3) (h3 : x / y < 0) :
  (2 * x - y = 7) ∨ (2 * x - y = -7) :=
by
  sorry

end find_value_of_expression_l135_135779


namespace trig_identity_solution_l135_135581

-- Define the necessary trigonometric functions
noncomputable def tan (x : ℝ) : ℝ := Real.sin x / Real.cos x
noncomputable def cot (x : ℝ) : ℝ := Real.cos x / Real.sin x

-- Statement of the theorem
theorem trig_identity_solution (x : ℝ) (k : ℤ) (hcos : Real.cos x ≠ 0) (hsin : Real.sin x ≠ 0) :
  (Real.sin x) ^ 2 * tan x + (Real.cos x) ^ 2 * cot x + 2 * Real.sin x * Real.cos x = (4 * Real.sqrt 3) / 3 →
  ∃ k : ℤ, x = (-1) ^ k * (Real.pi / 6) + (Real.pi / 2) :=
sorry

end trig_identity_solution_l135_135581


namespace triangle_inequality_area_equality_condition_l135_135643

theorem triangle_inequality_area (a b c S : ℝ) (h_area : S = (a * b * Real.sin (Real.arccos ((a*a + b*b - c*c) / (2*a*b)))) / 2) :
  a^2 + b^2 + c^2 ≥ 4 * Real.sqrt 3 * S :=
by
  sorry

theorem equality_condition (a b c : ℝ) (h_eq : a = b ∧ b = c) : 
  a^2 + b^2 + c^2 = 4 * Real.sqrt 3 * (a^2 * (Real.sqrt 3 / 4)) :=
by
  sorry

end triangle_inequality_area_equality_condition_l135_135643


namespace arrangement_books_not_adjacent_distribution_books_students_l135_135578

open Finset

-- Given 5 different books including 2 mathematics books
variable (books : Finset Nat) (math_books : Finset Nat)
variable (mathematic_books : books ⊆ math_books)

-- We need to prove:
-- 1. There are 60 ways to arrange the books such that the mathematics books are neither adjacent nor at both ends simultaneously.
theorem arrangement_books_not_adjacent (h1 : books.card = 5) (h2 : math_books.card = 2) :
  (arrangements books math_books) = 60 := sorry

-- 2. There are 150 ways to distribute the books to 3 students such that each student receives at least 1 book.
theorem distribution_books_students (h1 : books.card = 5) (h3 : students.card = 3) (every_student_one : ∀ (s ∈ students), (s.card ≥ 1)):
  (distribution books students) = 150 := sorry

end arrangement_books_not_adjacent_distribution_books_students_l135_135578


namespace volume_of_remaining_sphere_after_hole_l135_135597

noncomputable def volume_of_remaining_sphere (R : ℝ) : ℝ :=
  let volume_sphere := (4 / 3) * Real.pi * R^3
  let volume_cylinder := (4 / 3) * Real.pi * (R / 2)^3
  volume_sphere - volume_cylinder

theorem volume_of_remaining_sphere_after_hole : 
  volume_of_remaining_sphere 5 = (500 * Real.pi) / 3 :=
by
  sorry

end volume_of_remaining_sphere_after_hole_l135_135597


namespace mean_of_quadrilateral_angles_is_90_l135_135451

theorem mean_of_quadrilateral_angles_is_90 :
  let n := 4 in
  let sum_of_interior_angles (n : ℕ) := (n - 2) * 180 in
  let quadrilateral_interior_angle_sum := sum_of_interior_angles 4 in
  quadrilateral_interior_angle_sum = 360 →
  360 / 4 = 90 :=
by
  intros n sum_of_interior_angles quadrilateral_interior_angle_sum h
  exact sorry

end mean_of_quadrilateral_angles_is_90_l135_135451


namespace coefficient_of_x_in_expansion_l135_135694

theorem coefficient_of_x_in_expansion : 
  (let T_r := λ r : ℕ, (-2)^r * (Nat.choose 8 r) * x^(4 - (3 * r) / 2) in
   (T_r 2) = 112) := sorry

end coefficient_of_x_in_expansion_l135_135694


namespace intersection_of_M_and_complement_of_N_l135_135663

open Set

variable (U : Type) [Nonempty U] [TopologicaUnionSpace U]

-- Conditions
def M : Set ℝ := {x : ℝ | x^2 > 4}
def N : Set ℝ := {x : ℝ | x^2 - 2 * x - 3 ≤ 0}

-- Complement of N in U
def CU_N : Set ℝ := {x : ℝ | x > 3 ∨ x < -1}

-- Question and correct answer assertion
theorem intersection_of_M_and_complement_of_N :
  M ∩ CU_N = {x : ℝ | x > 3 ∨ x < -2} :=
by
  sorry

end intersection_of_M_and_complement_of_N_l135_135663


namespace find_abc_l135_135162

open_locale big_operators

noncomputable def sum_k := ∑ k in finset.range 100, (-1)^(k+1) * (k^3 + k^2 + k + 1) / k.factorial

theorem find_abc : 
  sum_k = 1010101 / 100.factorial - 0 ∧ 1010101 + 100 + 0 = 1010201 :=
by { sorry }

end find_abc_l135_135162


namespace building_height_l135_135471

theorem building_height
    (flagpole_height : ℝ)
    (flagpole_shadow_length : ℝ)
    (building_shadow_length : ℝ)
    (h : ℝ)
    (h_eq : flagpole_height / flagpole_shadow_length = h / building_shadow_length)
    (flagpole_height_eq : flagpole_height = 18)
    (flagpole_shadow_length_eq : flagpole_shadow_length = 45)
    (building_shadow_length_eq : building_shadow_length = 65) :
  h = 26 := by
  sorry

end building_height_l135_135471


namespace ratio_m_n_l135_135133

theorem ratio_m_n (m n : ℕ) (h1 : m > n) (h2 : ¬ (m % n = 0)) (h3 : (m % n) = ((m + n) % (m - n))) : (m : ℚ) / n = 5 / 2 := by
  sorry

end ratio_m_n_l135_135133


namespace gondor_repaired_3_phones_on_monday_l135_135106

theorem gondor_repaired_3_phones_on_monday :
  ∃ P : ℕ, 
    (10 * P + 10 * 5 + 20 * 2 + 20 * 4 = 200) ∧
    P = 3 :=
by
  sorry

end gondor_repaired_3_phones_on_monday_l135_135106


namespace adam_has_9_apples_l135_135610

def jackie_apples : ℕ := 6
def difference : ℕ := 3

def adam_apples (j : ℕ) (d : ℕ) : ℕ := 
  j + d

theorem adam_has_9_apples : adam_apples jackie_apples difference = 9 := 
by 
  sorry

end adam_has_9_apples_l135_135610


namespace calculate_total_shaded_area_l135_135532

theorem calculate_total_shaded_area
(smaller_square_side larger_square_side smaller_circle_radius larger_circle_radius : ℝ)
(h1 : smaller_square_side = 6)
(h2 : larger_square_side = 12)
(h3 : smaller_circle_radius = 3)
(h4 : larger_circle_radius = 6) :
  (smaller_square_side^2 - π * smaller_circle_radius^2) + 
  (larger_square_side^2 - π * larger_circle_radius^2) = 180 - 45 * π :=
by
  sorry

end calculate_total_shaded_area_l135_135532


namespace depth_of_first_hole_l135_135591

-- Conditions as definitions in Lean 4
def number_of_workers_first_hole : Nat := 45
def hours_worked_first_hole : Nat := 8

def number_of_workers_second_hole : Nat := 110  -- 45 existing workers + 65 extra workers
def hours_worked_second_hole : Nat := 6
def depth_second_hole : Nat := 55

-- The key assumption that work done (W) is proportional to the depth of the hole (D)
theorem depth_of_first_hole :
  let work_first_hole := number_of_workers_first_hole * hours_worked_first_hole
  let work_second_hole := number_of_workers_second_hole * hours_worked_second_hole
  let depth_first_hole := (work_first_hole * depth_second_hole) / work_second_hole
  depth_first_hole = 30 := sorry

end depth_of_first_hole_l135_135591


namespace compute_fraction_product_l135_135917

-- Definitions based on conditions
def one_third_pow_four : ℚ := (1 / 3) ^ 4
def one_fifth : ℚ := 1 / 5

-- Main theorem to prove the problem question == answer
theorem compute_fraction_product : (one_third_pow_four * one_fifth) = 1 / 405 :=
by
  sorry

end compute_fraction_product_l135_135917


namespace pairs_a_eq_b_l135_135060

theorem pairs_a_eq_b 
  (n : ℕ) (h_n : ¬ ∃ k : ℕ, k^2 = n) (a b : ℕ) 
  (r : ℝ) (h_r_pos : 0 < r) (h_ra_rational : ∃ q₁ : ℚ, r^a + (n:ℝ)^(1/2) = q₁) 
  (h_rb_rational : ∃ q₂ : ℚ, r^b + (n:ℝ)^(1/2) = q₂) : 
  a = b :=
sorry

end pairs_a_eq_b_l135_135060


namespace terminating_decimal_contains_digit_3_l135_135714

theorem terminating_decimal_contains_digit_3 :
  ∃ n : ℕ, n > 0 ∧ (∃ a b : ℕ, n = 2 ^ a * 5 ^ b) ∧ (∃ d, n = d * 10 ^ 0 + 3) ∧ n = 32 :=
by sorry

end terminating_decimal_contains_digit_3_l135_135714


namespace mean_value_interior_angles_quadrilateral_l135_135377

theorem mean_value_interior_angles_quadrilateral :
  let n := 4 in
  (n - 2) * 180 / n = 90 :=
by
  let n := 4
  sorry

end mean_value_interior_angles_quadrilateral_l135_135377


namespace smallest_n_for_unity_root_l135_135720

theorem smallest_n_for_unity_root (z : ℂ) (hz : z^6 - z^3 + 1 = 0) : ∃ n : ℕ, n > 0 ∧ (∀ z, (z^6 - z^3 + 1 = 0) → (∃ k, z = exp(2 * real.pi * complex.I * k / n))) ∧ n = 9 :=
by
  sorry

end smallest_n_for_unity_root_l135_135720


namespace mean_of_quadrilateral_angles_l135_135353

theorem mean_of_quadrilateral_angles :
  ∀ (angles : List ℝ), angles.length = 4 → angles.sum = 360 → angles.sum / angles.length = 90 :=
by
  intros
  sorry

end mean_of_quadrilateral_angles_l135_135353


namespace find_k_l135_135695

-- Define the number and compute the sum of its digits
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

-- Define the problem
theorem find_k :
  ∃ k : ℕ, sum_of_digits (9 * (10^k - 1)) = 1111 ∧ k = 124 :=
sorry

end find_k_l135_135695


namespace largest_possible_N_l135_135815

theorem largest_possible_N (N : ℕ) :
  let divisors := Nat.divisors N
  in (1 ∈ divisors) ∧ (N ∈ divisors) ∧ (divisors.length ≥ 3) ∧ (divisors[divisors.length - 3] = 21 * divisors[1]) → N = 441 := 
by
  sorry

end largest_possible_N_l135_135815


namespace rudolph_stop_signs_per_mile_l135_135136

theorem rudolph_stop_signs_per_mile :
  let distance := 5 + 2
  let stop_signs := 17 - 3
  (stop_signs / distance) = 2 :=
by
  let distance := 5 + 2
  let stop_signs := 17 - 3
  calc
    (stop_signs / distance) = (14 / 7) : by rw [stop_signs, distance]
                          ... = 2 : by norm_num

end rudolph_stop_signs_per_mile_l135_135136


namespace dasha_flags_proof_l135_135929

variable (Tata_flags_right Yasha_flags_right Vera_flags_right Maxim_flags_right : ℕ)
variable (Total_flags : ℕ)

theorem dasha_flags_proof 
  (hTata: Tata_flags_right = 14)
  (hYasha: Yasha_flags_right = 32)
  (hVera: Vera_flags_right = 20)
  (hMaxim: Maxim_flags_right = 8)
  (hTotal: Total_flags = 37) :
  ∃ (Dasha_flags : ℕ), Dasha_flags = 8 :=
by
  sorry

end dasha_flags_proof_l135_135929


namespace carpet_shaded_area_l135_135604

theorem carpet_shaded_area
  (side_length_carpet : ℝ)
  (S : ℝ)
  (T : ℝ)
  (h1 : side_length_carpet = 12)
  (h2 : 12 / S = 4)
  (h3 : S / T = 2) :
  let area_big_square := S^2
  let area_small_squares := 4 * T^2
  area_big_square + area_small_squares = 18 := by
  sorry

end carpet_shaded_area_l135_135604


namespace Events_B_and_C_mutex_l135_135756

-- Definitions of events based on scores
def EventA (score : ℕ) := score ≥ 1 ∧ score ≤ 10
def EventB (score : ℕ) := score > 5 ∧ score ≤ 10
def EventC (score : ℕ) := score > 1 ∧ score < 6
def EventD (score : ℕ) := score > 0 ∧ score < 6

-- Mutually exclusive definition:
def mutually_exclusive (P Q : ℕ → Prop) := ∀ (x : ℕ), ¬ (P x ∧ Q x)

-- The proof statement:
theorem Events_B_and_C_mutex : mutually_exclusive EventB EventC :=
by
  sorry

end Events_B_and_C_mutex_l135_135756


namespace range_of_a_for_local_min_l135_135507

noncomputable def f (a x : ℝ) : ℝ := (x - 2 * a) * (x^2 + a^2 * x + 2 * a^3)

theorem range_of_a_for_local_min :
  (∀ a : ℝ, (∃ δ > 0, ∀ ε ∈ Set.Ioo (-δ) δ, f a ε > f a 0) → a < 0 ∨ a > 2) :=
by
  sorry

end range_of_a_for_local_min_l135_135507


namespace temperature_decrease_l135_135117

theorem temperature_decrease (current_temp : ℝ) (future_temp : ℝ) : 
  current_temp = 84 → future_temp = (3 / 4) * current_temp → (current_temp - future_temp) = 21 :=
by
  intros h1 h2
  rw [h1, h2]
  norm_num
  sorry

end temperature_decrease_l135_135117


namespace estimate_nearsighted_students_l135_135595

theorem estimate_nearsighted_students (sample_size total_students nearsighted_sample : ℕ) 
  (h_sample_size : sample_size = 30)
  (h_total_students : total_students = 400)
  (h_nearsighted_sample : nearsighted_sample = 12):
  (total_students * nearsighted_sample) / sample_size = 160 := by
  sorry

end estimate_nearsighted_students_l135_135595


namespace ratio_of_time_l135_135577

theorem ratio_of_time (tX tY tZ : ℕ) (h1 : tX = 16) (h2 : tY = 12) (h3 : tZ = 8) :
  (tX : ℚ) / (tY * tZ / (tY + tZ) : ℚ) = 10 / 3 := 
by 
  sorry

end ratio_of_time_l135_135577


namespace transformation_identity_l135_135134

theorem transformation_identity (n : Nat) (h : 2 ≤ n) : 
  n * Real.sqrt (n / (n ^ 2 - 1)) = Real.sqrt (n + n / (n ^ 2 - 1)) := 
sorry

end transformation_identity_l135_135134


namespace mean_value_of_quadrilateral_angles_l135_135333

theorem mean_value_of_quadrilateral_angles : 
  (∀ (A B C D : ℕ), A + B + C + D = 360 → (A + B + C + D) / 4 = 90) :=
by {
  sorry
}

end mean_value_of_quadrilateral_angles_l135_135333


namespace ratio_value_l135_135521

theorem ratio_value (c d : ℝ) (h1 : c = 15 - 4 * d) (h2 : c / d = 4) : d = 15 / 8 :=
by sorry

end ratio_value_l135_135521


namespace cars_on_river_road_l135_135057

-- Define the number of buses and cars
variables (B C : ℕ)

-- Given conditions
def ratio_condition : Prop := (B : ℚ) / C = 1 / 17
def fewer_buses_condition : Prop := B = C - 80

-- Problem statement
theorem cars_on_river_road (h_ratio : ratio_condition B C) (h_fewer : fewer_buses_condition B C) : C = 85 :=
by
  sorry

end cars_on_river_road_l135_135057


namespace mean_of_quadrilateral_angles_l135_135410

theorem mean_of_quadrilateral_angles : 
  (∀ (a b c d : ℝ), a + b + c + d = 360) → 90 :=
by
  intro h
  sorry

end mean_of_quadrilateral_angles_l135_135410


namespace length_of_AC_l135_135529

-- Define the conditions: lengths and angle
def AB : ℝ := 10
def BC : ℝ := 10
def CD : ℝ := 15
def DA : ℝ := 15
def angle_ADC : ℝ := 120

-- Prove the length of diagonal AC is 15*sqrt(3)
theorem length_of_AC : 
  (CD ^ 2 + DA ^ 2 - 2 * CD * DA * Real.cos (angle_ADC * Real.pi / 180)) = (15 * Real.sqrt 3) ^ 2 :=
by
  sorry

end length_of_AC_l135_135529


namespace solve_equation_l135_135096

theorem solve_equation {x y z : ℝ} (h₁ : x + 95 / 12 * y + 4 * z = 0)
  (h₂ : 4 * x + 95 / 12 * y - 3 * z = 0)
  (h₃ : 3 * x + 5 * y - 4 * z = 0)
  (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
  x^2 * z / y^3 = -60 :=
sorry

end solve_equation_l135_135096


namespace james_main_game_time_l135_135960

-- Define the download time in minutes
def download_time := 10

-- Define the installation time as half the download time
def installation_time := download_time / 2

-- Define the combined time for download and installation
def combined_time := download_time + installation_time

-- Define the tutorial time as triple the combined time
def tutorial_time := combined_time * 3

-- Define the total time as the combined time plus the tutorial time
def total_time := combined_time + tutorial_time

-- Statement of the problem to prove
theorem james_main_game_time : total_time = 60 := by
  sorry

end james_main_game_time_l135_135960


namespace smallest_multiple_of_4_and_14_is_28_l135_135633

theorem smallest_multiple_of_4_and_14_is_28 :
  ∃ (a : ℕ), a > 0 ∧ (4 ∣ a) ∧ (14 ∣ a) ∧ ∀ b : ℕ, b > 0 → (4 ∣ b) → (14 ∣ b) → a ≤ b := 
sorry

end smallest_multiple_of_4_and_14_is_28_l135_135633


namespace number_of_rational_points_in_region_l135_135560

/-- The statement to prove the number of points with positive rational coordinates
within the prescribed region -/
theorem number_of_rational_points_in_region : 
  let region := {p : ℚ × ℚ | 1 ≤ p.1 ∧ 0 < p.1 ∧ 0 < p.2 ∧ p.1 + p.2 ≤ 4}
  ∃ (s : Finset (ℚ × ℚ)), s.card = 10 ∧ ∀ p ∈ s, p ∈ region :=
sorry

end number_of_rational_points_in_region_l135_135560


namespace rowers_voted_l135_135114

variable (R : ℕ)

/-- Each rower votes for exactly 4 coaches out of 50 coaches,
and each coach receives exactly 7 votes.
Prove that the number of rowers is 88. -/
theorem rowers_voted (h1 : 50 * 7 = 4 * R) : R = 88 := by 
  sorry

end rowers_voted_l135_135114


namespace difference_max_min_eq_2log2_minus_1_l135_135109

noncomputable def f (x : ℝ) : ℝ := x * Real.log x - x

theorem difference_max_min_eq_2log2_minus_1 :
  let M := max (f (1 / 2)) (max (f 1) (f 2))
  let N := min (f (1 / 2)) (min (f 1) (f 2))
  M - N = 2 * Real.log 2 - 1 :=
by
  let M := max (f (1 / 2)) (max (f 1) (f 2))
  let N := min (f (1 / 2)) (min (f 1) (f 2))
  sorry

end difference_max_min_eq_2log2_minus_1_l135_135109


namespace find_g5_l135_135835

def g : ℤ → ℤ := sorry

axiom g_cond1 : g 1 > 1
axiom g_cond2 : ∀ x y : ℤ, g (x + y) + x * g y + y * g x = g x * g y + x + y + x * y
axiom g_cond3 : ∀ x : ℤ, 3 * g x = g (x + 1) + 2 * x - 1

theorem find_g5 : g 5 = 248 :=
by
  sorry

end find_g5_l135_135835


namespace solve_eqs_l135_135270

theorem solve_eqs (x y : ℤ) (h1 : 7 - x = 15) (h2 : y - 3 = 4 + x) : x = -8 ∧ y = -1 := 
by
  sorry

end solve_eqs_l135_135270


namespace possible_values_for_p_t_l135_135538

theorem possible_values_for_p_t (p q r s t : ℝ)
(h₁ : |p - q| = 3)
(h₂ : |q - r| = 4)
(h₃ : |r - s| = 5)
(h₄ : |s - t| = 6) :
  ∃ (v : Finset ℝ), v = {0, 2, 4, 6, 8, 10, 12, 18} ∧ |p - t| ∈ v :=
sorry

end possible_values_for_p_t_l135_135538


namespace geometric_sequence_sum_l135_135121

theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) (h1 : ∀ n, a (n + 1) = a n * q)
  (h2 : q = 2) (h3 : a 0 + a 1 + a 2 = 21) : 
  a 2 + a 3 + a 4 = 84 :=
sorry

end geometric_sequence_sum_l135_135121


namespace g_value_at_2_l135_135968

def g (x : ℝ) (d : ℝ) : ℝ := 3 * x^5 - 2 * x^4 + d * x - 8

theorem g_value_at_2 (d : ℝ) (h : g (-2) d = 4) : g 2 d = -84 := by
  sorry

end g_value_at_2_l135_135968


namespace max_fruits_is_15_l135_135205

def maxFruits (a m p : ℕ) : Prop :=
  3 * a + 4 * m + 5 * p = 50 ∧ a ≥ 1 ∧ m ≥ 1 ∧ p ≥ 1

theorem max_fruits_is_15 : ∃ a m p : ℕ, maxFruits a m p ∧ a + m + p = 15 := 
  sorry

end max_fruits_is_15_l135_135205


namespace fraction_power_mult_correct_l135_135912

noncomputable def fraction_power_mult : Prop :=
  (\left(\frac{1}{3} \right)^{4}) * \left(\frac{1}{5} \right) = \left(\frac{1}{405} \right)

theorem fraction_power_mult_correct : fraction_power_mult :=
by
  -- The complete proof will be here.
  sorry

end fraction_power_mult_correct_l135_135912


namespace mean_value_of_quadrilateral_angles_l135_135392

theorem mean_value_of_quadrilateral_angles (a b c d : ℝ) (h_angles_sum : a + b + c + d = 360) : (a + b + c + d) / 4 = 90 := 
by
  sorry

end mean_value_of_quadrilateral_angles_l135_135392


namespace choir_average_age_l135_135997

-- Each condition as a definition in Lean 4
def avg_age_females := 28
def num_females := 12
def avg_age_males := 32
def num_males := 18
def total_people := num_females + num_males

-- The total sum of ages calculated from the given conditions
def sum_ages_females := avg_age_females * num_females
def sum_ages_males := avg_age_males * num_males
def total_sum_ages := sum_ages_females + sum_ages_males

-- The final proof statement to be proved
theorem choir_average_age : 
  (total_sum_ages : ℝ) / (total_people : ℝ) = 30.4 := by
  sorry

end choir_average_age_l135_135997


namespace geometric_inequality_l135_135954

variable {q : ℝ} {b : ℕ → ℝ}

def geometric_sequence (b : ℕ → ℝ) (q : ℝ) : Prop := ∀ n : ℕ, b (n + 1) = b n * q

theorem geometric_inequality
  (h_geometric : geometric_sequence b q)
  (h_q_gt_one : q > 1)
  (h_pos : ∀ n : ℕ, b n > 0) :
  b 4 + b 8 > b 5 + b 7 :=
by
  sorry

end geometric_inequality_l135_135954


namespace sum_zero_l135_135257

noncomputable def f : ℝ → ℝ := sorry

theorem sum_zero :
  (∀ x : ℝ, f (-x) = -f x) →
  (∀ x : ℝ, f (x + 5) = f x) →
  f (1 / 3) = 1 →
  f (16 / 3) + f (29 / 3) + f 12 + f (-7) = 0 :=
by
  intros hodd hperiod hvalue
  sorry

end sum_zero_l135_135257


namespace aunt_may_milk_left_l135_135612

theorem aunt_may_milk_left
  (morning_milk : ℕ)
  (evening_milk : ℕ)
  (sold_milk : ℕ)
  (leftover_milk : ℕ)
  (h1 : morning_milk = 365)
  (h2 : evening_milk = 380)
  (h3 : sold_milk = 612)
  (h4 : leftover_milk = 15) :
  morning_milk + evening_milk + leftover_milk - sold_milk = 148 :=
by
  sorry

end aunt_may_milk_left_l135_135612


namespace aquafaba_needed_for_cakes_l135_135198

def tablespoons_of_aquafaba_for_egg_whites (n_egg_whites : ℕ) : ℕ :=
  2 * n_egg_whites

def total_egg_whites_needed (cakes : ℕ) (egg_whites_per_cake : ℕ) : ℕ :=
  cakes * egg_whites_per_cake

theorem aquafaba_needed_for_cakes (cakes : ℕ) (egg_whites_per_cake : ℕ) :
  tablespoons_of_aquafaba_for_egg_whites (total_egg_whites_needed cakes egg_whites_per_cake) = 32 :=
by
  have h1 : cakes = 2 := sorry
  have h2 : egg_whites_per_cake = 8 := sorry
  sorry

end aquafaba_needed_for_cakes_l135_135198


namespace solve_logarithmic_eq_l135_135986

-- Define the conditions
def condition1 (x : ℝ) : Prop := 9^x - 5 > 0
def condition2 (x : ℝ) : Prop := 3^x - 2 > 0

-- Translate the problem to a Lean statement
theorem solve_logarithmic_eq (x : ℝ) (h1 : condition1 x) (h2 : condition2 x) :
  log 2 (9^x - 5) = 2 + log 2 (3^x - 2) ↔ x = 1 :=
sorry

end solve_logarithmic_eq_l135_135986


namespace number_of_passed_candidates_l135_135032

theorem number_of_passed_candidates :
  ∀ (P F : ℕ),
  (P + F = 500) →
  (P * 80 + F * 15 = 500 * 60) →
  P = 346 :=
by
  intros P F h1 h2
  sorry

end number_of_passed_candidates_l135_135032


namespace fraction_equals_half_l135_135754

def numerator : ℤ := 1 - 2 + 4 - 8 + 16 - 32 + 64
def denominator : ℤ := 2 - 4 + 8 - 16 + 32 - 64 + 128

theorem fraction_equals_half : (numerator : ℚ) / (denominator : ℚ) = 1 / 2 :=
by
  sorry

end fraction_equals_half_l135_135754


namespace mean_value_of_angles_mean_value_of_interior_angles_of_quadrilateral_l135_135345

-- Defining the sum of interior angles of a quadrilateral.
def sum_of_interior_angles (quad : Type) [IsQuadrilateral quad] : ℝ :=
  360

-- Defining the number of angles in a quadrilateral.
def number_of_angles (quad : Type) [IsQuadrilateral quad] : ℕ :=
  4

-- The theorem we want to prove: the mean value of the measures of the four interior angles of any quadrilateral.
theorem mean_value_of_angles (quad : Type) [IsQuadrilateral quad] : ℝ :=
  sum_of_interior_angles quad / number_of_angles quad
 
-- Proving the theorem.
theorem mean_value_of_interior_angles_of_quadrilateral (quad : Type) [IsQuadrilateral quad]:
  mean_value_of_angles quad = 90 :=
by
  sorry

end mean_value_of_angles_mean_value_of_interior_angles_of_quadrilateral_l135_135345


namespace stewart_farm_sheep_l135_135732

theorem stewart_farm_sheep (S H : ℕ)
  (h1 : S / H = 2 / 7)
  (h2 : H * 230 = 12880) :
  S = 16 :=
by sorry

end stewart_farm_sheep_l135_135732


namespace consecutive_ints_square_l135_135977

theorem consecutive_ints_square (a b : ℤ) (h : b = a + 1) : 
  a^2 + b^2 + (a * b)^2 = (a * b + 1)^2 := 
by sorry

end consecutive_ints_square_l135_135977


namespace Kishore_misc_expense_l135_135743

theorem Kishore_misc_expense:
  let savings := 2400
  let percent_saved := 0.10
  let rent := 5000
  let milk := 1500
  let groceries := 4500
  let education := 2500
  let petrol := 2000
  let total_salary := savings / percent_saved 
  let total_spent := rent + milk + groceries + education + petrol
  total_salary - (total_spent + savings) = 6100 := 
by
  sorry

end Kishore_misc_expense_l135_135743


namespace folded_triangle_square_length_l135_135898

theorem folded_triangle_square_length (side_length folded_distance length_squared : ℚ) 
(h1: side_length = 15) 
(h2: folded_distance = 11) 
(h3: length_squared = 1043281/31109) :
∃ (PQ : ℚ), PQ^2 = length_squared := 
by 
  sorry

end folded_triangle_square_length_l135_135898


namespace radius_increase_is_0_31_l135_135922

noncomputable def increase_in_radius (initial_radius : ℝ) (odometer_summer : ℝ) (odometer_winter : ℝ) (miles_to_inches : ℝ) : ℝ :=
  let circumference_summer := 2 * Real.pi * initial_radius
  let distance_per_rotation_summer := circumference_summer / miles_to_inches
  let rotations_summer := odometer_summer / distance_per_rotation_summer
  let rotations_winter := odometer_winter / distance_per_rotation_summer
  let distance_winter := rotations_winter * distance_per_rotation_summer
  let new_radius := (distance_winter * miles_to_inches) / (2 * rotations_winter * Real.pi)
  new_radius - initial_radius

theorem radius_increase_is_0_31 : 
    increase_in_radius 16 530 520 63360 = 0.31 := 
by
    sorry

end radius_increase_is_0_31_l135_135922


namespace triangle_QR_length_l135_135665

noncomputable def length_PM : ℝ := 6 -- PM = 6 cm
noncomputable def length_MA : ℝ := 12 -- MA = 12 cm
noncomputable def length_NB : ℝ := 9 -- NB = 9 cm
def MN_parallel_PQ : Prop := true -- MN ∥ PQ

theorem triangle_QR_length 
  (h1 : MN_parallel_PQ)
  (h2 : length_PM = 6)
  (h3 : length_MA = 12)
  (h4 : length_NB = 9) : 
  length_QR = 27 :=
sorry

end triangle_QR_length_l135_135665


namespace find_period_l135_135038

variable (x : ℕ)
variable (theo_daily : ℕ := 8)
variable (mason_daily : ℕ := 7)
variable (roxy_daily : ℕ := 9)
variable (total_water : ℕ := 168)

theorem find_period (h : (theo_daily + mason_daily + roxy_daily) * x = total_water) : x = 7 :=
by
  sorry

end find_period_l135_135038


namespace rectangle_k_value_l135_135485

theorem rectangle_k_value (x d : ℝ)
  (h_ratio : ∃ x, ∀ l w, l = 5 * x ∧ w = 4 * x)
  (h_diagonal : ∀ l w, l = 5 * x ∧ w = 4 * x → d^2 = (5 * x)^2 + (4 * x)^2)
  (h_area_written : ∃ k, ∀ A, A = (5 * x) * (4 * x) → A = k * d^2) :
  ∃ k, k = 20 / 41 := sorry

end rectangle_k_value_l135_135485


namespace figure_C_perimeter_l135_135307

def is_perimeter (figure : Type) (perimeter : ℕ) : Prop :=
∃ x y : ℕ, (figure = 'A' → 6*x + 2*y = perimeter) ∧ 
           (figure = 'B' → 4*x + 6*y = perimeter) ∧
           (figure = 'C' → 2*x + 6*y = perimeter)

theorem figure_C_perimeter (hA : is_perimeter 'A' 56) (hB : is_perimeter 'B' 56) : 
  is_perimeter 'C' 40 :=
by
  sorry

end figure_C_perimeter_l135_135307


namespace solve_cubic_eq_solve_quadratic_eq_l135_135988

-- Define the first equation and prove its solution
theorem solve_cubic_eq (x : ℝ) (h : x^3 + 64 = 0) : x = -4 :=
by
  -- skipped proof
  sorry

-- Define the second equation and prove its solutions
theorem solve_quadratic_eq (x : ℝ) (h : (x - 2)^2 = 81) : x = 11 ∨ x = -7 :=
by
  -- skipped proof
  sorry

end solve_cubic_eq_solve_quadratic_eq_l135_135988


namespace mean_value_of_quadrilateral_angles_l135_135416

theorem mean_value_of_quadrilateral_angles (angles : Fin 4 → ℝ) 
  (h_sum : ∑ i, angles i = 360) : ∑ i, angles i / 4 = 90 :=
by
  sorry

end mean_value_of_quadrilateral_angles_l135_135416


namespace unique_real_solution_k_eq_35_over_4_l135_135220

theorem unique_real_solution_k_eq_35_over_4 :
  ∃ k : ℚ, (∀ x : ℝ, (x + 5) * (x + 3) = k + 3 * x) ↔ (k = 35 / 4) :=
by
  sorry

end unique_real_solution_k_eq_35_over_4_l135_135220


namespace greatest_odd_factors_le_150_l135_135016

theorem greatest_odd_factors_le_150 : 
  let n := 144 in
  n < 150 ∧ (∃ k, k * k = n) :=
by
  sorry

end greatest_odd_factors_le_150_l135_135016


namespace larger_angle_measure_l135_135878

theorem larger_angle_measure (x : ℝ) (h : 4 * x + 5 * x = 180) : 5 * x = 100 :=
by
  sorry

end larger_angle_measure_l135_135878


namespace largest_common_term_l135_135993

theorem largest_common_term (b : ℕ) : 
  (b < 1000) ∧ (b % 5 = 4) ∧ (b % 11 = 7) → b = 964 :=
by
  intros h
  sorry

end largest_common_term_l135_135993


namespace g_5_is_248_l135_135848

def g : ℤ → ℤ := sorry

theorem g_5_is_248 :
  (g(1) > 1) ∧
  (∀ x y : ℤ, g(x + y) + x * g(y) + y * g(x) = g(x) * g(y) + x + y + x * y) ∧
  (∀ x : ℤ, 3 * g(x) = g(x + 1) + 2 * x - 1) →
  g(5) = 248 :=
by
  -- proof omitted
  sorry

end g_5_is_248_l135_135848


namespace smallest_number_of_students_l135_135744

theorem smallest_number_of_students
  (n : ℕ)
  (h1 : 3 * 90 + (n - 3) * 65 ≤ n * 80)
  (h2 : ∀ k, k ≤ n - 3 → 65 ≤ k)
  (h3 : (3 * 90) + ((n - 3) * 65) / n = 80) : n = 5 :=
sorry

end smallest_number_of_students_l135_135744


namespace trains_pass_each_other_time_l135_135709

theorem trains_pass_each_other_time :
  ∃ t : ℝ, t = 240 / 191.171 := 
sorry

end trains_pass_each_other_time_l135_135709


namespace triangle_with_altitudes_is_obtuse_l135_135602

theorem triangle_with_altitudes_is_obtuse (h1 h2 h3 : ℝ) (h_pos1 : h1 > 0) (h_pos2 : h2 > 0) (h_pos3 : h3 > 0)
    (h_triangle_ineq1 : 1 / h2 + 1 / h3 > 1 / h1)
    (h_triangle_ineq2 : 1 / h1 + 1 / h3 > 1 / h2)
    (h_triangle_ineq3 : 1 / h1 + 1 / h2 > 1 / h3) : 
    ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
    a * h1 = b * h2 ∧ b * h2 = c * h3 ∧
    (a^2 + b^2 < c^2 ∨ b^2 + c^2 < a^2 ∨ c^2 + a^2 < b^2) :=
sorry

end triangle_with_altitudes_is_obtuse_l135_135602


namespace find_g5_l135_135836

def g : ℤ → ℤ := sorry

axiom g_cond1 : g 1 > 1
axiom g_cond2 : ∀ x y : ℤ, g (x + y) + x * g y + y * g x = g x * g y + x + y + x * y
axiom g_cond3 : ∀ x : ℤ, 3 * g x = g (x + 1) + 2 * x - 1

theorem find_g5 : g 5 = 248 :=
by
  sorry

end find_g5_l135_135836


namespace even_sum_count_l135_135522

theorem even_sum_count (x y : ℕ) 
  (hx : x = (40 + 42 + 44 + 46 + 48 + 50 + 52 + 54 + 56 + 58 + 60)) 
  (hy : y = ((60 - 40) / 2 + 1)) : 
  x + y = 561 := 
by 
  sorry

end even_sum_count_l135_135522


namespace positions_after_347_moves_l135_135113

-- Define the possible positions for the cat
inductive CatPosition
| top_vertex
| right_upper_vertex
| right_lower_vertex
| left_lower_vertex
| left_upper_vertex

-- Define the possible positions for the mouse
inductive MousePosition
| top_left_edge
| left_upper_vertex
| left_middle_edge
| left_lower_vertex
| bottom_edge
| right_lower_vertex
| right_middle_edge
| right_upper_vertex
| top_right_edge
| top_vertex

-- Define the movement function for the cat
def cat_position_after_moves (moves : Nat) : CatPosition :=
  match moves % 5 with
  | 0 => CatPosition.top_vertex
  | 1 => CatPosition.right_upper_vertex
  | 2 => CatPosition.right_lower_vertex
  | 3 => CatPosition.left_lower_vertex
  | 4 => CatPosition.left_upper_vertex
  | _ => CatPosition.top_vertex  -- This case is unreachable due to % 5

-- Define the movement function for the mouse
def mouse_position_after_moves (moves : Nat) : MousePosition :=
  match moves % 10 with
  | 0 => MousePosition.top_left_edge
  | 1 => MousePosition.left_upper_vertex
  | 2 => MousePosition.left_middle_edge
  | 3 => MousePosition.left_lower_vertex
  | 4 => MousePosition.bottom_edge
  | 5 => MousePosition.right_lower_vertex
  | 6 => MousePosition.right_middle_edge
  | 7 => MousePosition.right_upper_vertex
  | 8 => MousePosition.top_right_edge
  | 9 => MousePosition.top_vertex
  | _ => MousePosition.top_left_edge  -- This case is unreachable due to % 10

-- Prove the positions after 347 moves
theorem positions_after_347_moves :
  cat_position_after_moves 347 = CatPosition.right_upper_vertex ∧
  mouse_position_after_moves 347 = MousePosition.right_middle_edge :=
by
  sorry

end positions_after_347_moves_l135_135113


namespace mean_of_quadrilateral_angles_l135_135386

theorem mean_of_quadrilateral_angles : 
  (∀ (a b c d : ℝ), a + b + c + d = 360 → (a + b + c + d) / 4 = 90) :=
begin
  intros a b c d h,
  calc
    (a + b + c + d) / 4 = 360 / 4 : by rw h
                    ... = 90 : by norm_num,
end

end mean_of_quadrilateral_angles_l135_135386


namespace circle_parametric_eq_l135_135950

theorem circle_parametric_eq 
  (a b r : ℝ) (θ : ℝ) (hθ : 0 ≤ θ ∧ θ < 2 * Real.pi):
  (∃ (x y : ℝ), (x = r * Real.cos θ + a ∧ y = r * Real.sin θ + b)) ↔ 
  (∃ (x' y' : ℝ), (x' = r * Real.cos θ ∧ y' = r * Real.sin θ)) :=
sorry

end circle_parametric_eq_l135_135950


namespace largest_possible_N_l135_135814

theorem largest_possible_N : 
  ∃ (N : ℕ), (∀ (d : ℕ), d ∣ N → d = 1 ∨ d = N ∨ ∃ (k : ℕ), k * d = N)
    ∧ (∃ (p q r : ℕ), 1 < p ∧ p < q ∧ q < r ∧ q * 21 = r 
      ∧ [1, p, q, r, N / p, N / q, N / r, N] = multiset.sort has_dvd.dvd [1, p, q, N / q, (N / r), N / p, r, N])
    ∧ N = 441 := sorry

end largest_possible_N_l135_135814


namespace car_speed_is_48_l135_135891

theorem car_speed_is_48 {v : ℝ} : (3600 / v = 75) → v = 48 := 
by {
  sorry
}

end car_speed_is_48_l135_135891


namespace compute_fraction_power_mul_l135_135910

theorem compute_fraction_power_mul : ((1 / 3: ℚ) ^ 4) * (1 / 5) = (1 / 405) := by
  -- proof goes here
  sorry

end compute_fraction_power_mul_l135_135910


namespace largest_common_value_lt_1000_l135_135995

theorem largest_common_value_lt_1000 :
  ∃ a : ℕ, ∃ n m : ℕ, a = 4 + 5 * n ∧ a = 7 + 11 * m ∧ a < 1000 ∧ 
  (∀ b : ℕ, ∀ p q : ℕ, b = 4 + 5 * p ∧ b = 7 + 11 * q ∧ b < 1000 → b ≤ a) :=
sorry

end largest_common_value_lt_1000_l135_135995


namespace mean_value_of_quadrilateral_angles_l135_135441

theorem mean_value_of_quadrilateral_angles :
  let sum_of_angles := 360 in
  let number_of_angles := 4 in
  sum_of_angles / number_of_angles = 90 :=
by
  sorry

end mean_value_of_quadrilateral_angles_l135_135441


namespace mean_value_of_interior_angles_of_quadrilateral_l135_135403

theorem mean_value_of_interior_angles_of_quadrilateral 
  (sum_of_interior_angles : ℝ := 360) 
  (number_of_angles : ℝ := 4) : 
  sum_of_interior_angles / number_of_angles = 90 := 
by 
  sorry

end mean_value_of_interior_angles_of_quadrilateral_l135_135403


namespace ratio_b_to_c_l135_135278

theorem ratio_b_to_c (x a b c : ℤ) 
    (h1 : x = 100 * a + 10 * b + c)
    (h2 : a > 0)
    (h3 : 999 - x = 241) : (b : ℚ) / c = 5 / 8 :=
by
  sorry

end ratio_b_to_c_l135_135278


namespace find_a100_l135_135641

noncomputable def arithmetic_sequence (n : ℕ) (a : ℕ → ℤ) : Prop :=
∀ n : ℕ, a n - a (n + 1) = 2

theorem find_a100 (a : ℕ → ℤ) (h1 : arithmetic_sequence 3 a) (h2 : a 3 = 6) :
  a 100 = -188 :=
sorry

end find_a100_l135_135641


namespace mean_value_of_interior_angles_of_quadrilateral_l135_135397

theorem mean_value_of_interior_angles_of_quadrilateral 
  (sum_of_interior_angles : ℝ := 360) 
  (number_of_angles : ℝ := 4) : 
  sum_of_interior_angles / number_of_angles = 90 := 
by 
  sorry

end mean_value_of_interior_angles_of_quadrilateral_l135_135397


namespace sum_a3_a4_a5_eq_84_l135_135120

-- Definitions based on the conditions in the problem
def is_geometric_sequence (a : ℕ → ℕ) :=
  ∃ q : ℕ, ∀ n : ℕ, a (n+1) = a n * q

variables {a : ℕ → ℕ}

-- Given conditions
axiom q_eq_2 : ∃ q, q = 2
axiom sum_first_three_eq_21 : (a 1) + (a 2) + (a 3) = 21

-- Required proof statement
theorem sum_a3_a4_a5_eq_84 (h : is_geometric_sequence a) (hq : q_eq_2) (h_sum : sum_first_three_eq_21) : 
  let q := 2 in (a 3) + (a 4) + (a 5) = 84 :=
by
  sorry

end sum_a3_a4_a5_eq_84_l135_135120


namespace cos_theta_plus_pi_over_3_l135_135225

theorem cos_theta_plus_pi_over_3 {θ : ℝ} (h : Real.sin (θ / 2 + π / 6) = 2 / 3) :
  Real.cos (θ + π / 3) = 1 / 9 :=
by
  sorry

end cos_theta_plus_pi_over_3_l135_135225


namespace g_five_eq_248_l135_135843

-- We define g, and assume it meets the conditions described.
variable (g : ℤ → ℤ)

-- Condition 1: g(1) > 1
axiom g_one_gt_one : g 1 > 1

-- Condition 2: Functional equation for g
axiom g_funct_eq (x y : ℤ) : g (x + y) + x * g y + y * g x = g x * g y + x + y + x * y

-- Condition 3: Recursive relationship for g
axiom g_recur_eq (x : ℤ) : 3 * g x = g (x + 1) + 2 * x - 1

-- Theorem we want to prove
theorem g_five_eq_248 : g 5 = 248 := by
  sorry

end g_five_eq_248_l135_135843


namespace perimeter_C_is_40_l135_135312

noncomputable def perimeter_of_figure_C (x y : ℝ) : ℝ :=
  2 * x + 6 * y

theorem perimeter_C_is_40 (x y : ℝ) (h1 : 6 * x + 2 * y = 56) (h2 : 4 * x + 6 * y = 56) :
  perimeter_of_figure_C x y = 40 :=
by
  -- Define initial conditions
  have eq1 : 3 * x + y = 28, by { rw [mul_assoc, mul_comm 3 x, add_assoc], exact (eq.div h1 2) }
  have eq2 : 2 * x + 3 * y = 28, by { rw [mul_assoc, mul_comm 2 x, add_assoc], exact (eq.div h2 2) }
  -- Assume the solutions are obtained from here
  have sol_x : x = 8, by sorry
  have sol_y : y = 4, by sorry
  -- Calculate the perimeter of figure C
  rw [perimeter_of_figure_C, sol_x, sol_y]
  norm_num
  trivial

-- Test case to ensure the code builds successfully
#eval perimeter_of_figure_C 8 4  -- Expected output: 40

end perimeter_C_is_40_l135_135312


namespace amount_paid_is_correct_l135_135801

-- Define the conditions
def time_painting_house : ℕ := 8
def time_fixing_counter := 3 * time_painting_house
def time_mowing_lawn : ℕ := 6
def hourly_rate : ℕ := 15

-- Define the total time worked
def total_time_worked := time_painting_house + time_fixing_counter + time_mowing_lawn

-- Define the total amount paid
def total_amount_paid := total_time_worked * hourly_rate

-- Formalize the goal
theorem amount_paid_is_correct : total_amount_paid = 570 :=
by
  -- Proof steps to be filled in
  sorry  -- Placeholder for the proof

end amount_paid_is_correct_l135_135801


namespace jihyae_initial_money_l135_135963

variables {M : ℕ}

def spent_on_supplies (M : ℕ) := M / 2 + 200
def left_after_buying (M : ℕ) := M - spent_on_supplies M
def saved (M : ℕ) := left_after_buying M / 2 + 300
def final_leftover (M : ℕ) := left_after_buying M - saved M

theorem jihyae_initial_money : final_leftover M = 350 → M = 3000 :=
by
  sorry

end jihyae_initial_money_l135_135963


namespace triangle_angle_measure_l135_135624

/-- Proving the measure of angle x in a defined triangle -/
theorem triangle_angle_measure (A B C x : ℝ) (hA : A = 85) (hB : B = 35) (hC : C = 30) : x = 150 :=
by
  sorry

end triangle_angle_measure_l135_135624


namespace minNumberOfGloves_l135_135555

-- Define the number of participants
def numParticipants : ℕ := 43

-- Define the number of gloves needed per participant
def glovesPerParticipant : ℕ := 2

-- Define the total number of gloves
def totalGloves (participants glovesPerParticipant : ℕ) : ℕ := 
  participants * glovesPerParticipant

-- Theorem proving the minimum number of gloves required
theorem minNumberOfGloves : totalGloves numParticipants glovesPerParticipant = 86 :=
by
  sorry

end minNumberOfGloves_l135_135555


namespace prob_one_tails_in_three_consecutive_flips_l135_135731

-- Define the probability of heads and tails
def P_H : ℝ := 0.5
def P_T : ℝ := 0.5

-- Define the probability of a sequence of coin flips resulting in exactly one tails in three flips
def P_one_tails_in_three_flips : ℝ :=
  P_H * P_H * P_T + P_H * P_T * P_H + P_T * P_H * P_H

-- The statement we need to prove
theorem prob_one_tails_in_three_consecutive_flips :
  P_one_tails_in_three_flips = 0.375 :=
by
  sorry

end prob_one_tails_in_three_consecutive_flips_l135_135731


namespace inequality_proof_l135_135809

theorem inequality_proof (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : x + y > 2) : (1 + x) / y < 2 ∨ (1 + y) / x < 2 :=
sorry

end inequality_proof_l135_135809


namespace mean_value_of_interior_angles_of_quadrilateral_l135_135360

theorem mean_value_of_interior_angles_of_quadrilateral :
  (360 / 4) = 90 := 
by
  sorry

end mean_value_of_interior_angles_of_quadrilateral_l135_135360


namespace total_area_inf_series_l135_135830

noncomputable def S (n : ℕ) : ℝ := 1 / 2 ^ (n - 1)
noncomputable def T (n : ℕ) : ℝ := 1 / 4 * (1 / 2 ^ (n - 1))

theorem total_area_inf_series : (∑ n in (Finset.range ∞), S n + T n) = 5 / 2 :=
by
  sorry

end total_area_inf_series_l135_135830


namespace fair_attendance_l135_135547

theorem fair_attendance (P_this_year P_next_year P_last_year : ℕ) 
  (h1 : P_this_year = 600) 
  (h2 : P_next_year = 2 * P_this_year) 
  (h3 : P_last_year = P_next_year - 200) : 
  P_this_year = 600 ∧ P_next_year = 1200 ∧ P_last_year = 1000 :=
by
  split
  . exact h1
  split
  . rw [h1]
    rw [h2]
    norm_num
  . rw [h1]
    rw [h2]
    rw [h3]
    norm_num

example : fair_attendance 600 1200 1000 :=
by sorry

end fair_attendance_l135_135547


namespace autumn_grain_purchase_exceeds_1_8_billion_tons_l135_135822

variable (x : ℝ)

theorem autumn_grain_purchase_exceeds_1_8_billion_tons 
  (h : x > 0.18) : 
  x > 1.8 := 
by 
  sorry

end autumn_grain_purchase_exceeds_1_8_billion_tons_l135_135822


namespace mean_value_of_quadrilateral_interior_angles_l135_135459

theorem mean_value_of_quadrilateral_interior_angles (a b c d : ℝ) 
  (h_sum : a + b + c + d = 360) : 
  (a + b + c + d) / 4 = 90 :=
by
  sorry

end mean_value_of_quadrilateral_interior_angles_l135_135459


namespace max_a_plus_b_l135_135541

theorem max_a_plus_b (a b c d e : ℕ) (h1 : a < b) (h2 : b < c) (h3 : c < d) (h4 : d < e) (h5 : a + 2*b + 3*c + 4*d + 5*e = 300) : a + b ≤ 35 :=
sorry

end max_a_plus_b_l135_135541


namespace max_chord_length_l135_135648

theorem max_chord_length (x1 y1 x2 y2 : ℝ) (h_parabola1 : x1^2 = 8 * y1) (h_parabola2 : x2^2 = 8 * y2)
  (h_midpoint_ordinate : (y1 + y2) / 2 = 4) :
  abs ((y1 + y2) + 4) = 12 :=
by
  sorry

end max_chord_length_l135_135648


namespace car_average_speed_l135_135892

noncomputable def average_speed (speeds : List ℝ) (distances : List ℝ) (times : List ℝ) : ℝ :=
  (distances.sum + times.sum) / times.sum

theorem car_average_speed :
  let distances := [30, 35, 35, 52 / 3, 15]
  let times := [30 / 45, 35 / 55, 30 / 60, 20 / 60, 15 / 65]
  average_speed [45, 55, 70, 52, 65] distances times = 64.82 := by
  sorry

end car_average_speed_l135_135892


namespace correct_statements_count_l135_135865

theorem correct_statements_count :
  (∀ x > 0, x > Real.sin x) ∧
  (¬ (∀ x > 0, x - Real.log x > 0) ↔ (∃ x > 0, x - Real.log x ≤ 0)) ∧
  ¬ (∀ p q : Prop, (p ∨ q) → (p ∧ q)) →
  2 = 2 :=
by sorry

end correct_statements_count_l135_135865


namespace vector_definition_l135_135711

-- Definition of a vector's characteristics
def hasCharacteristics (vector : Type) := ∃ (magnitude : ℝ) (direction : ℂ), true

-- The statement to prove: a vector is defined by having both magnitude and direction
theorem vector_definition (vector : Type) : hasCharacteristics vector := 
sorry

end vector_definition_l135_135711


namespace no_real_roots_x_squared_minus_x_plus_nine_l135_135866

theorem no_real_roots_x_squared_minus_x_plus_nine :
  ∀ x : ℝ, ¬ (x^2 - x + 9 = 0) :=
by 
  intro x 
  sorry

end no_real_roots_x_squared_minus_x_plus_nine_l135_135866


namespace find_a6_l135_135119

open Nat

noncomputable def arith_seq (a1 d : ℤ) (n : ℕ) : ℤ :=
  a1 + (n - 1) * d

def a2 := 4
def a4 := 2

theorem find_a6 (a1 d : ℤ) (h_a2 : arith_seq a1 d 2 = a2) (h_a4 : arith_seq a1 d 4 = a4) : 
  arith_seq a1 d 6 = 0 := by
  sorry

end find_a6_l135_135119


namespace mean_of_quadrilateral_angles_l135_135359

theorem mean_of_quadrilateral_angles :
  ∀ (angles : List ℝ), angles.length = 4 → angles.sum = 360 → angles.sum / angles.length = 90 :=
by
  intros
  sorry

end mean_of_quadrilateral_angles_l135_135359


namespace mean_of_quadrilateral_angles_l135_135356

theorem mean_of_quadrilateral_angles :
  ∀ (angles : List ℝ), angles.length = 4 → angles.sum = 360 → angles.sum / angles.length = 90 :=
by
  intros
  sorry

end mean_of_quadrilateral_angles_l135_135356


namespace correct_cost_per_piece_l135_135551

-- Definitions for the given conditions
def totalPaid : ℝ := 20700
def reimbursement : ℝ := 600
def numberOfPieces : ℝ := 150
def correctTotal := totalPaid - reimbursement

-- Theorem stating the correct cost per piece of furniture
theorem correct_cost_per_piece : correctTotal / numberOfPieces = 134 := 
by
  sorry

end correct_cost_per_piece_l135_135551


namespace functional_equation_solution_l135_135586

theorem functional_equation_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (x^2 * y) = f (x * y) + y * f (f x + y)) →
  (∀ y : ℝ, f y = 0) :=
by
  intro h
  sorry

end functional_equation_solution_l135_135586


namespace pages_needed_l135_135978

def cards_per_page : ℕ := 3
def new_cards : ℕ := 2
def old_cards : ℕ := 10

theorem pages_needed : (new_cards + old_cards) / cards_per_page = 4 := by
  sorry

end pages_needed_l135_135978


namespace probability_of_odd_m_n_l135_135812

def count_odds (l : List ℕ) : ℕ :=
  List.length (List.filter (λ x => x % 2 = 1) l)

def possible_combinations : ℕ := 7 * 9

def favorable_combinations : ℕ :=
  (count_odds [1, 2, 3, 4, 5, 6, 7]) * (count_odds [1, 2, 3, 4, 5, 6, 7, 8, 9])

theorem probability_of_odd_m_n : (favorable_combinations : ℚ) / (possible_combinations : ℚ) = 20/63 :=
by
  -- Proof to be filled
  sorry

end probability_of_odd_m_n_l135_135812


namespace remainder_of_n_mod_7_l135_135634

theorem remainder_of_n_mod_7 (n : ℕ) : (n^2 ≡ 1 [MOD 7]) → (n^3 ≡ 6 [MOD 7]) → (n ≡ 6 [MOD 7]) :=
by
  sorry

end remainder_of_n_mod_7_l135_135634


namespace number_of_chairs_is_40_l135_135667

-- Define the conditions
variables (C : ℕ) -- Total number of chairs
variables (capacity_per_chair : ℕ := 2) -- Each chair's capacity is 2 people
variables (occupied_ratio : ℚ := 3 / 5) -- Ratio of occupied chairs
variables (attendees : ℕ := 48) -- Number of attendees

theorem number_of_chairs_is_40
  (h1 : ∀ c : ℕ, capacity_per_chair * c = attendees)
  (h2 : occupied_ratio * C * capacity_per_chair = attendees) : 
  C = 40 := sorry

end number_of_chairs_is_40_l135_135667


namespace point_A_is_minus_five_l135_135021

theorem point_A_is_minus_five 
  (A B C : ℝ)
  (h1 : A + 4 = B)
  (h2 : B - 2 = C)
  (h3 : C = -3) : 
  A = -5 := 
by 
  sorry

end point_A_is_minus_five_l135_135021


namespace mean_value_of_quadrilateral_angles_l135_135446

theorem mean_value_of_quadrilateral_angles :
  let sum_of_angles := 360 in
  let number_of_angles := 4 in
  sum_of_angles / number_of_angles = 90 :=
by
  sorry

end mean_value_of_quadrilateral_angles_l135_135446


namespace incenter_circumcenter_coincide_if_and_only_if_skew_edges_equal_l135_135982

structure Tetrahedron (α : Type*) [MetricSpace α] :=
(A B C D : α)

def Incenter {α : Type*} [MetricSpace α] (T : Tetrahedron α) : α := sorry
def Circumcenter {α : Type*} [MetricSpace α] (T : Tetrahedron α) : α := sorry

def equidistant_from_faces {α : Type*} [MetricSpace α] (T : Tetrahedron α) (I : α) : Prop := sorry
def equidistant_from_vertices {α : Type*} [MetricSpace α] (T : Tetrahedron α) (O : α) : Prop := sorry
def skew_edges_equal {α : Type*} [MetricSpace α] (T : Tetrahedron α) : Prop := sorry

theorem incenter_circumcenter_coincide_if_and_only_if_skew_edges_equal
  {α : Type*} [MetricSpace α] (T : Tetrahedron α) :
  (∃ I, ∃ O, (Incenter T = I) ∧ (Circumcenter T = O) ∧ 
            (equidistant_from_faces T I) ∧ (equidistant_from_vertices T O)) ↔ (skew_edges_equal T) := 
sorry

end incenter_circumcenter_coincide_if_and_only_if_skew_edges_equal_l135_135982


namespace visitors_on_saturday_l135_135183

theorem visitors_on_saturday (S : ℕ) (h1 : S + (S + 40) = 440) : S = 200 := by
  sorry

end visitors_on_saturday_l135_135183


namespace julia_garden_area_l135_135118

theorem julia_garden_area
  (length perimeter walk_distance : ℝ)
  (h_length : length * 30 = walk_distance)
  (h_perimeter : perimeter * 12 = walk_distance)
  (h_perimeter_def : perimeter = 2 * (length + width))
  (h_walk_distance : walk_distance = 1500) :
  (length * width = 625) :=
by
  sorry

end julia_garden_area_l135_135118


namespace mean_value_of_angles_mean_value_of_interior_angles_of_quadrilateral_l135_135350

-- Defining the sum of interior angles of a quadrilateral.
def sum_of_interior_angles (quad : Type) [IsQuadrilateral quad] : ℝ :=
  360

-- Defining the number of angles in a quadrilateral.
def number_of_angles (quad : Type) [IsQuadrilateral quad] : ℕ :=
  4

-- The theorem we want to prove: the mean value of the measures of the four interior angles of any quadrilateral.
theorem mean_value_of_angles (quad : Type) [IsQuadrilateral quad] : ℝ :=
  sum_of_interior_angles quad / number_of_angles quad
 
-- Proving the theorem.
theorem mean_value_of_interior_angles_of_quadrilateral (quad : Type) [IsQuadrilateral quad]:
  mean_value_of_angles quad = 90 :=
by
  sorry

end mean_value_of_angles_mean_value_of_interior_angles_of_quadrilateral_l135_135350


namespace circle_equation_AB_diameter_l135_135938

theorem circle_equation_AB_diameter (A B : ℝ × ℝ) :
  A = (1, -4) → B = (-5, 4) →
  ∃ C : ℝ × ℝ, C = (-2, 0) ∧ ∃ r : ℝ, r = 5 ∧ (∀ x y : ℝ, (x + 2)^2 + y^2 = 25) :=
by intros h1 h2; sorry

end circle_equation_AB_diameter_l135_135938


namespace mean_value_of_quadrilateral_angles_l135_135445

theorem mean_value_of_quadrilateral_angles :
  let sum_of_angles := 360 in
  let number_of_angles := 4 in
  sum_of_angles / number_of_angles = 90 :=
by
  sorry

end mean_value_of_quadrilateral_angles_l135_135445


namespace mole_can_sustain_l135_135115

noncomputable def mole_winter_sustainability : Prop :=
  ∃ (grain millet : ℕ), 
    grain = 8 ∧ 
    millet = 0 ∧ 
    ∀ (month : ℕ), 1 ≤ month ∧ month ≤ 3 → 
      ((grain ≥ 3 ∧ (grain - 3) + millet <= 12) ∨ 
      (grain ≥ 1 ∧ millet ≥ 3 ∧ (grain - 1) + (millet - 3) <= 12)) ∧
      ((∃ grain_exchanged millet_gained : ℕ, 
         grain_exchanged ≤ grain ∧
         millet_gained = 2 * grain_exchanged ∧
         grain - grain_exchanged + millet_gained <= 12 ∧
         grain = grain - grain_exchanged) → 
      (grain = 0 ∧ millet = 0))

theorem mole_can_sustain : mole_winter_sustainability := 
sorry 

end mole_can_sustain_l135_135115


namespace Sam_bought_cards_l135_135975

theorem Sam_bought_cards (original_cards current_cards : ℕ) 
  (h1 : original_cards = 87) (h2 : current_cards = 74) : 
  original_cards - current_cards = 13 :=
by
  -- The 'sorry' here means the proof is omitted.
  sorry

end Sam_bought_cards_l135_135975


namespace mean_value_of_quadrilateral_angles_l135_135433

theorem mean_value_of_quadrilateral_angles 
  (a1 a2 a3 a4 : ℝ) 
  (h_sum : a1 + a2 + a3 + a4 = 360) :
  (a1 + a2 + a3 + a4) / 4 = 90 :=
by
  sorry

end mean_value_of_quadrilateral_angles_l135_135433


namespace trapezium_perimeters_l135_135316

theorem trapezium_perimeters (AB BC AD AF : ℝ)
  (h1 : AB = 30) (h2 : BC = 30) (h3 : AD = 25) (h4 : AF = 24) :
  ∃ p : ℝ, (p = 90 ∨ p = 104) :=
by
  sorry

end trapezium_perimeters_l135_135316


namespace counterexample_exists_l135_135621

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_composite (n : ℕ) : Prop := n > 1 ∧ ¬ is_prime n

def problem_set : set ℕ := {11, 15, 19, 21, 27}

theorem counterexample_exists : ∃ n ∈ problem_set, is_composite n ∧ is_composite (n - 2) :=
by
  sorry

end counterexample_exists_l135_135621


namespace radius_of_semicircle_l135_135153

theorem radius_of_semicircle (P : ℝ) (π_val : ℝ) (h1 : P = 162) (h2 : π_val = Real.pi) : 
  ∃ r : ℝ, r = 162 / (π + 2) :=
by
  use 162 / (Real.pi + 2)
  sorry

end radius_of_semicircle_l135_135153


namespace min_inequality_l135_135007

theorem min_inequality (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h_sum : x + y + z = 2) :
  ∃ L, L = 9 / 4 ∧ (1 / (x + y) + 1 / (x + z) + 1 / (y + z) ≥ L) :=
sorry

end min_inequality_l135_135007


namespace range_a_l135_135314

theorem range_a (a : ℝ) : (∀ x, x > 0 → x^2 - a * x + 1 > 0) → -2 < a ∧ a < 2 := by
  sorry

end range_a_l135_135314


namespace mean_value_of_angles_mean_value_of_interior_angles_of_quadrilateral_l135_135342

-- Defining the sum of interior angles of a quadrilateral.
def sum_of_interior_angles (quad : Type) [IsQuadrilateral quad] : ℝ :=
  360

-- Defining the number of angles in a quadrilateral.
def number_of_angles (quad : Type) [IsQuadrilateral quad] : ℕ :=
  4

-- The theorem we want to prove: the mean value of the measures of the four interior angles of any quadrilateral.
theorem mean_value_of_angles (quad : Type) [IsQuadrilateral quad] : ℝ :=
  sum_of_interior_angles quad / number_of_angles quad
 
-- Proving the theorem.
theorem mean_value_of_interior_angles_of_quadrilateral (quad : Type) [IsQuadrilateral quad]:
  mean_value_of_angles quad = 90 :=
by
  sorry

end mean_value_of_angles_mean_value_of_interior_angles_of_quadrilateral_l135_135342


namespace daily_rental_cost_l135_135734

theorem daily_rental_cost (rental_fee_per_day : ℝ) (mileage_rate : ℝ) (budget : ℝ) (max_miles : ℝ) 
  (h1 : mileage_rate = 0.20) 
  (h2 : budget = 88.0) 
  (h3 : max_miles = 190.0) :
  rental_fee_per_day = 50.0 := 
by
  sorry

end daily_rental_cost_l135_135734


namespace geometric_prog_y_90_common_ratio_l135_135104

theorem geometric_prog_y_90_common_ratio :
  ∀ (y : ℝ), y = 90 → ∃ r : ℝ, r = (90 + y) / (30 + y) ∧ r = (180 + y) / (90 + y) ∧ r = 3 / 2 :=
by
  intros
  sorry

end geometric_prog_y_90_common_ratio_l135_135104


namespace mean_value_of_quadrilateral_angles_l135_135437

theorem mean_value_of_quadrilateral_angles 
  (a1 a2 a3 a4 : ℝ) 
  (h_sum : a1 + a2 + a3 + a4 = 360) :
  (a1 + a2 + a3 + a4) / 4 = 90 :=
by
  sorry

end mean_value_of_quadrilateral_angles_l135_135437


namespace figure_C_perimeter_l135_135308

def is_perimeter (figure : Type) (perimeter : ℕ) : Prop :=
∃ x y : ℕ, (figure = 'A' → 6*x + 2*y = perimeter) ∧ 
           (figure = 'B' → 4*x + 6*y = perimeter) ∧
           (figure = 'C' → 2*x + 6*y = perimeter)

theorem figure_C_perimeter (hA : is_perimeter 'A' 56) (hB : is_perimeter 'B' 56) : 
  is_perimeter 'C' 40 :=
by
  sorry

end figure_C_perimeter_l135_135308


namespace profit_percentage_l135_135603

theorem profit_percentage (SP : ℝ) (CP : ℝ) (hSP : SP = 100) (hCP : CP = 83.33) :
    (SP - CP) / CP * 100 = 20 :=
by
  rw [hSP, hCP]
  norm_num
  sorry

end profit_percentage_l135_135603


namespace find_square_subtraction_l135_135508

theorem find_square_subtraction (x y : ℝ) (h1 : x = Real.sqrt 5) (h2 : y = Real.sqrt 2) : (x - y)^2 = 7 - 2 * Real.sqrt 10 :=
by
  sorry

end find_square_subtraction_l135_135508


namespace resistance_parallel_l135_135785

theorem resistance_parallel (x y r : ℝ) (hy : y = 6) (hr : r = 2.4) 
  (h : 1 / r = 1 / x + 1 / y) : x = 4 :=
  sorry

end resistance_parallel_l135_135785


namespace wire_length_from_sphere_volume_l135_135886

theorem wire_length_from_sphere_volume
  (r_sphere : ℝ) (r_cylinder : ℝ) (h : ℝ)
  (h_sphere : r_sphere = 12)
  (h_cylinder : r_cylinder = 4)
  (volume_conservation : (4/3 * Real.pi * r_sphere^3) = (Real.pi * r_cylinder^2 * h)) :
  h = 144 :=
by {
  sorry
}

end wire_length_from_sphere_volume_l135_135886


namespace grain_milling_necessary_pounds_l135_135943

theorem grain_milling_necessary_pounds (x : ℝ) (h : 0.90 * x = 100) : x = 111 + 1 / 9 := 
by
  sorry

end grain_milling_necessary_pounds_l135_135943


namespace mean_value_of_quadrilateral_interior_angles_l135_135466

theorem mean_value_of_quadrilateral_interior_angles (a b c d : ℝ) 
  (h_sum : a + b + c + d = 360) : 
  (a + b + c + d) / 4 = 90 :=
by
  sorry

end mean_value_of_quadrilateral_interior_angles_l135_135466


namespace max_valid_committees_l135_135157

-- Define the conditions
def community_size : ℕ := 20
def english_speakers : ℕ := 10
def german_speakers : ℕ := 10
def french_speakers : ℕ := 10
def total_subsets : ℕ := Nat.choose community_size 3
def invalid_subsets_per_language : ℕ := Nat.choose 10 3

-- Lean statement to verify the number of valid committees
theorem max_valid_committees :
  total_subsets - 3 * invalid_subsets_per_language = 1020 :=
by
  simp [community_size, total_subsets, invalid_subsets_per_language]
  sorry

end max_valid_committees_l135_135157


namespace unique_solution_l135_135618

theorem unique_solution (k : ℝ) (h : k + 1 ≠ 0) : 
  (∀ x y : ℝ, ((x + 3) / (k * x + x - 3) = x) → ((y + 3) / (k * y + y - 3) = y) → x = y) ↔ k = -7/3 :=
by sorry

end unique_solution_l135_135618


namespace positive_number_property_l135_135739

theorem positive_number_property (x : ℝ) (h : x > 0) (hx : (x / 100) * x = 9) : x = 30 := by
  sorry

end positive_number_property_l135_135739


namespace mean_value_of_quadrilateral_interior_angles_l135_135462

theorem mean_value_of_quadrilateral_interior_angles (a b c d : ℝ) 
  (h_sum : a + b + c + d = 360) : 
  (a + b + c + d) / 4 = 90 :=
by
  sorry

end mean_value_of_quadrilateral_interior_angles_l135_135462


namespace problem_equivalence_l135_135497

theorem problem_equivalence : 4 * 4^3 - 16^60 / 16^57 = -3840 := by
  sorry

end problem_equivalence_l135_135497


namespace quadratic_roots_m_value_l135_135254

theorem quadratic_roots_m_value
  (x1 x2 m : ℝ)
  (h1 : x1^2 + 2 * x1 + m = 0)
  (h2 : x2^2 + 2 * x2 + m = 0)
  (h3 : x1 + x2 = x1 * x2 - 1) :
  m = -1 :=
sorry

end quadratic_roots_m_value_l135_135254


namespace rectangle_area_l135_135531

noncomputable def area_of_rectangle (radius : ℝ) (ab ad : ℝ) : ℝ :=
  ab * ad

theorem rectangle_area (radius : ℝ) (ad : ℝ) (ab : ℝ) 
  (h_radius : radius = Real.sqrt 5)
  (h_ab_ad_relation : ab = 4 * ad) : 
  area_of_rectangle radius ab ad = 16 / 5 :=
by
  sorry

end rectangle_area_l135_135531


namespace mean_value_of_quadrilateral_angles_l135_135444

theorem mean_value_of_quadrilateral_angles :
  let sum_of_angles := 360 in
  let number_of_angles := 4 in
  sum_of_angles / number_of_angles = 90 :=
by
  sorry

end mean_value_of_quadrilateral_angles_l135_135444


namespace min_sum_of_fractions_is_43_over_72_l135_135125

noncomputable def digits := {2, 3, 4, 5, 6, 7, 8, 9}
def distinct_digits (A B C D : ℕ) : Prop := 
  A ∈ digits ∧ B ∈ digits ∧ C ∈ digits ∧ D ∈ digits ∧ 
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D

theorem min_sum_of_fractions_is_43_over_72 (A B C D : ℕ) 
  (h_distinct : distinct_digits A B C D) :
  (A : ℚ) / B + (C : ℚ) / D = 43 / 72 :=
sorry

end min_sum_of_fractions_is_43_over_72_l135_135125


namespace mean_of_quadrilateral_angles_l135_135405

theorem mean_of_quadrilateral_angles : 
  (∀ (a b c d : ℝ), a + b + c + d = 360) → 90 :=
by
  intro h
  sorry

end mean_of_quadrilateral_angles_l135_135405


namespace minimum_trucks_required_l135_135690

-- Definitions for the problem
def total_weight_stones : ℝ := 10
def max_stone_weight : ℝ := 1
def truck_capacity : ℝ := 3

-- The theorem to prove
theorem minimum_trucks_required : ∃ (n : ℕ), n = 5 ∧ (n * truck_capacity) ≥ total_weight_stones := by
  sorry

end minimum_trucks_required_l135_135690


namespace mean_value_of_quadrilateral_angles_l135_135422

theorem mean_value_of_quadrilateral_angles (angles : Fin 4 → ℝ) 
  (h_sum : ∑ i, angles i = 360) : ∑ i, angles i / 4 = 90 :=
by
  sorry

end mean_value_of_quadrilateral_angles_l135_135422


namespace calculate_altitude_l135_135069

-- Define the conditions
def Speed_up : ℕ := 18
def Speed_down : ℕ := 24
def Avg_speed : ℝ := 20.571428571428573

-- Define what we want to prove
theorem calculate_altitude : 
  2 * Speed_up * Speed_down / (Speed_up + Speed_down) = Avg_speed →
  (864 : ℝ) / 2 = 432 :=
by
  sorry

end calculate_altitude_l135_135069


namespace type_2004_A_least_N_type_B_diff_2004_l135_135128

def game_type_A (N : ℕ) : Prop :=
  ∀ n, (1 ≤ n ∧ n ≤ N) → (n % 2 = 0 → false) 

def game_type_B (N : ℕ) : Prop :=
  ∃ n, (1 ≤ n ∧ n ≤ N) ∧ (n % 2 = 0 → true)


theorem type_2004_A : game_type_A 2004 :=
sorry

theorem least_N_type_B_diff_2004 : ∀ N, N > 2004 → game_type_B N → N = 2048 :=
sorry

end type_2004_A_least_N_type_B_diff_2004_l135_135128


namespace matrix_det_evaluation_l135_135761

noncomputable def matrix_det (x y z : ℝ) : ℝ :=
  Matrix.det ![
    ![1,   x,     y,     z],
    ![1, x + y,   y,     z],
    ![1,   x, x + y,     z],
    ![1,   x,     y, x + y + z]
  ]

theorem matrix_det_evaluation (x y z : ℝ) :
  matrix_det x y z = y * x * x + y * y * x :=
by sorry

end matrix_det_evaluation_l135_135761


namespace orthocenter_PQR_l135_135787

structure Point3D :=
  (x : ℚ)
  (y : ℚ)
  (z : ℚ)

def orthocenter (P Q R : Point3D) : Point3D :=
  sorry

theorem orthocenter_PQR :
  orthocenter ⟨2, 3, 4⟩ ⟨6, 4, 2⟩ ⟨4, 5, 6⟩ = ⟨1/2, 13/2, 15/2⟩ :=
by {
  sorry
}

end orthocenter_PQR_l135_135787


namespace find_g5_l135_135852

def g : ℤ → ℤ := sorry

axiom g1 : g 1 > 1
axiom g2 : ∀ x y : ℤ, g (x + y) + x * g y + y * g x = g x * g y + x + y + x * y
axiom g3 : ∀ x : ℤ, 3 * g x = g (x + 1) + 2 * x - 1

theorem find_g5 : g 5 = 248 :=
sorry

end find_g5_l135_135852


namespace blue_to_yellow_ratio_is_half_l135_135135

noncomputable section

def yellow_fish := 12
def blue_fish : ℕ := by 
  have total_fish := 42
  have green_fish := 2 * yellow_fish
  exact total_fish - (yellow_fish + green_fish)
def fish_ratio (x y : ℕ) := x / y

theorem blue_to_yellow_ratio_is_half : fish_ratio blue_fish yellow_fish = 1 / 2 := by
  sorry

end blue_to_yellow_ratio_is_half_l135_135135


namespace probability_of_monochromatic_triangle_l135_135784

noncomputable def problem_statement : ℕ → ℕ → ℕ → real := sorry

theorem probability_of_monochromatic_triangle:
  ∀ (pentagon_sides diagonals red_diagonals : ℕ), 
  pentagon_sides = 5 → 
  diagonals = 5 → 
  red_diagonals = 3 → 
  problem_statement pentagon_sides diagonals red_diagonals = 0.9979 :=
begin
  sorry
end

end probability_of_monochromatic_triangle_l135_135784


namespace curve_cross_intersection_l135_135904

theorem curve_cross_intersection : 
  ∃ (t_a t_b : ℝ), t_a ≠ t_b ∧ 
  (3 * t_a^2 + 1 = 3 * t_b^2 + 1) ∧
  (t_a^3 - 6 * t_a^2 + 4 = t_b^3 - 6 * t_b^2 + 4) ∧
  (3 * t_a^2 + 1 = 109 ∧ t_a^3 - 6 * t_a^2 + 4 = -428) := by
  sorry

end curve_cross_intersection_l135_135904


namespace max_stickers_single_player_l135_135481

noncomputable def max_stickers (num_players : ℕ) (average_stickers : ℕ) : ℕ :=
  let total_stickers := num_players * average_stickers
  let min_stickers_one_player := 1
  let min_stickers_others := (num_players - 1) * min_stickers_one_player
  total_stickers - min_stickers_others

theorem max_stickers_single_player : 
  ∀ (num_players average_stickers : ℕ), 
    num_players = 25 → 
    average_stickers = 4 →
    ∀ player_stickers : ℕ, player_stickers ≤ max_stickers num_players average_stickers → player_stickers = 76 :=
    by
      intro num_players average_stickers players_eq avg_eq player_stickers player_le_max
      sorry

end max_stickers_single_player_l135_135481


namespace smallest_n_for_unity_root_l135_135719

theorem smallest_n_for_unity_root (z : ℂ) (hz : z^6 - z^3 + 1 = 0) : ∃ n : ℕ, n > 0 ∧ (∀ z, (z^6 - z^3 + 1 = 0) → (∃ k, z = exp(2 * real.pi * complex.I * k / n))) ∧ n = 9 :=
by
  sorry

end smallest_n_for_unity_root_l135_135719


namespace nina_math_homework_l135_135820

theorem nina_math_homework (x : ℕ) :
  let ruby_math := 6
  let ruby_read := 2
  let nina_math := x * ruby_math
  let nina_read := 8 * ruby_read
  let nina_total := nina_math + nina_read
  nina_total = 48 → x = 5 :=
by
  intros
  sorry

end nina_math_homework_l135_135820


namespace mean_value_of_quadrilateral_angles_l135_135424

theorem mean_value_of_quadrilateral_angles : 
  ∀ (a b c d : ℝ), a + b + c + d = 360 → (a + b + c + d) / 4 = 90 :=
by
  intro a b c d h
  sorry

end mean_value_of_quadrilateral_angles_l135_135424


namespace mean_value_of_quadrilateral_angles_l135_135447

theorem mean_value_of_quadrilateral_angles :
  let sum_of_angles := 360 in
  let number_of_angles := 4 in
  sum_of_angles / number_of_angles = 90 :=
by
  sorry

end mean_value_of_quadrilateral_angles_l135_135447


namespace exists_integer_square_with_three_identical_digits_l135_135080

theorem exists_integer_square_with_three_identical_digits:
  ∃ x: ℤ, (x^2 % 1000 = 444) := by
  sorry

end exists_integer_square_with_three_identical_digits_l135_135080


namespace meeting_time_l135_135052

-- Definitions for the problem conditions.
def track_length : ℕ := 1800
def speed_A_kmph : ℕ := 36
def speed_B_kmph : ℕ := 54

-- Conversion factor from kmph to mps.
def kmph_to_mps (speed_kmph : ℕ) : ℕ := (speed_kmph * 1000) / 3600

-- Calculate the speeds in mps.
def speed_A_mps : ℕ := kmph_to_mps speed_A_kmph
def speed_B_mps : ℕ := kmph_to_mps speed_B_kmph

-- Calculate the time to complete one lap for A and B.
def time_lap_A : ℕ := track_length / speed_A_mps
def time_lap_B : ℕ := track_length / speed_B_mps

-- Prove the time to meet at the starting point.
theorem meeting_time : (Nat.lcm time_lap_A time_lap_B) = 360 := by
  -- Skipping the proof with sorry placeholder
  sorry

end meeting_time_l135_135052


namespace perimeter_C_correct_l135_135303

variables (x y : ℕ)

def perimeter_A (x y : ℕ) := 6 * x + 2 * y
def perimeter_B (x y : ℕ) := 4 * x + 6 * y
def perimeter_C (x y : ℕ) := 2 * x + 6 * y

theorem perimeter_C_correct (x y : ℕ) (h1 : 6 * x + 2 * y = 56) (h2 : 4 * x + 6 * y = 56) :
  2 * x + 6 * y = 40 :=
sorry

end perimeter_C_correct_l135_135303


namespace geom_seq_prod_of_terms_l135_135795

theorem geom_seq_prod_of_terms (a : ℕ → ℝ) (r : ℝ) (h_geom : ∀ n, a (n + 1) = r * a n) (h_a5 : a 5 = 2) : a 1 * a 9 = 4 := by
  sorry

end geom_seq_prod_of_terms_l135_135795


namespace mean_value_of_interior_angles_of_quadrilateral_l135_135365

theorem mean_value_of_interior_angles_of_quadrilateral :
  (360 / 4) = 90 := 
by
  sorry

end mean_value_of_interior_angles_of_quadrilateral_l135_135365


namespace compare_neg_rationals_l135_135083

theorem compare_neg_rationals : - (3 / 4 : ℚ) > - (6 / 5 : ℚ) :=
by sorry

end compare_neg_rationals_l135_135083


namespace integral_of_continuous_function_interval_l135_135973

open Set Filter IntervalIntegral
open scoped Topology BigOperators

noncomputable def integral_of_continuous_function (f : ℝ → ℝ) (hf1 : ∀ x ∈ (Icc 0 1), f x ∈ (Icc 0 1))
  (hf2 : ∀ x ∈ (Icc 0 1), f (f x) = 1) : Icc (3 / 4 : ℝ) 1 :=
  sorry

theorem integral_of_continuous_function_interval (f : ℝ → ℝ) (hf1 : ∀ x ∈ (Icc 0 1), f x ∈ (Icc 0 1)) 
  (hf2 : ∀ x ∈ (Icc 0 1), f (f x) = 1) : 
  3 / 4 < ∫ x in (0 : ℝ)..1, f x ∧ ∫ x in (0 : ℝ)..1, f x ≤ 1 :=
sorry

end integral_of_continuous_function_interval_l135_135973


namespace new_triangle_area_l135_135770

theorem new_triangle_area (a b : ℝ) (x y : ℝ) (hypotenuse : x = a ∧ y = b ∧ x^2 + y^2 = (a + b)^2) : 
    (3  * (1 / 2) * a * b) = (3 / 2) * a * b :=
by
  sorry

end new_triangle_area_l135_135770


namespace find_slope_and_intercept_l135_135896

noncomputable def line_equation_to_slope_intercept_form 
  (x y : ℝ) : Prop :=
  (3 * (x - 2) - 4 * (y + 3) = 0) ↔ (y = (3 / 4) * x - 4.5)

theorem find_slope_and_intercept : 
  ∃ (m b : ℝ), 
    (∀ (x y : ℝ), (line_equation_to_slope_intercept_form x y) → m = 3/4 ∧ b = -4.5) :=
sorry

end find_slope_and_intercept_l135_135896


namespace general_term_of_arithmetic_seq_sum_of_first_n_terms_b_n_l135_135791

theorem general_term_of_arithmetic_seq
  (a_n : ℕ → ℕ)
  (S_n : ℕ → ℕ)
  (a_2_eq_3 : a_n 2 = 3)
  (S_4_eq_16 : S_n 4 = 16) :
  (∀ n, a_n n = 2 * n - 1) :=
sorry

theorem sum_of_first_n_terms_b_n
  (a_n : ℕ → ℕ)
  (b_n : ℕ → ℝ)
  (T_n : ℕ → ℝ)
  (general_formula_a_n : ∀ n, a_n n = 2 * n - 1)
  (b_n_definition : ∀ n, b_n n = 1 / (a_n n * a_n (n + 1))) :
  (∀ n, T_n n = n / (2 * n + 1)) :=
sorry

end general_term_of_arithmetic_seq_sum_of_first_n_terms_b_n_l135_135791


namespace mike_total_earning_l135_135686

theorem mike_total_earning 
  (first_job : ℕ := 52)
  (hours : ℕ := 12)
  (wage_per_hour : ℕ := 9) :
  first_job + (hours * wage_per_hour) = 160 :=
by
  sorry

end mike_total_earning_l135_135686


namespace marble_problem_l135_135110

theorem marble_problem {r b : ℕ} 
  (h1 : 9 * r - b = 27) 
  (h2 : 3 * r - b = 3) : r + b = 13 := 
by
  sorry

end marble_problem_l135_135110


namespace peaches_thrown_away_l135_135905

variables (total_peaches fresh_percentage peaches_left : ℕ) (thrown_away : ℕ)
variables (h1 : total_peaches = 250) (h2 : fresh_percentage = 60) (h3 : peaches_left = 135)

theorem peaches_thrown_away :
  thrown_away = (total_peaches * (fresh_percentage / 100)) - peaches_left :=
sorry

end peaches_thrown_away_l135_135905


namespace a3_value_l135_135639

-- Define the geometric sequence
def geom_seq (r : ℝ) (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n = a 0 * r ^ n

-- Given conditions
variables (a : ℕ → ℝ) (r : ℝ)
axiom h_geom : geom_seq r a
axiom h_a1 : a 1 = 1
axiom h_a5 : a 5 = 4

-- Goal to prove
theorem a3_value : a 3 = 2 ∨ a 3 = -2 := by
  sorry

end a3_value_l135_135639


namespace second_offset_length_l135_135927

-- Definitions based on the given conditions.
def diagonal : ℝ := 24
def offset1 : ℝ := 9
def area_quad : ℝ := 180

-- Statement to prove the length of the second offset.
theorem second_offset_length :
  ∃ h : ℝ, (1 / 2) * diagonal * offset1 + (1 / 2) * diagonal * h = area_quad ∧ h = 6 :=
by
  sorry

end second_offset_length_l135_135927


namespace remainder_when_n_add_3006_divided_by_6_l135_135511

theorem remainder_when_n_add_3006_divided_by_6 (n : ℤ) (h : n % 6 = 1) : (n + 3006) % 6 = 1 := by
  sorry

end remainder_when_n_add_3006_divided_by_6_l135_135511


namespace perimeter_C_l135_135287

theorem perimeter_C (x y : ℕ) 
  (h1 : 6 * x + 2 * y = 56)
  (h2 : 4 * x + 6 * y = 56) :
  2 * x + 6 * y = 40 := 
by
  sorry

end perimeter_C_l135_135287


namespace gas_cost_l135_135094

theorem gas_cost 
  (x : ℝ)
  (h1 : 5 * (x / 5) = x)
  (h2 : 8 * (x / 8) = x)
  (h3 : (x / 5) - 15.50 = (x / 8)) : 
  x = 206.67 :=
by
  sorry

end gas_cost_l135_135094


namespace mean_of_quadrilateral_angles_is_90_l135_135452

theorem mean_of_quadrilateral_angles_is_90 :
  let n := 4 in
  let sum_of_interior_angles (n : ℕ) := (n - 2) * 180 in
  let quadrilateral_interior_angle_sum := sum_of_interior_angles 4 in
  quadrilateral_interior_angle_sum = 360 →
  360 / 4 = 90 :=
by
  intros n sum_of_interior_angles quadrilateral_interior_angle_sum h
  exact sorry

end mean_of_quadrilateral_angles_is_90_l135_135452


namespace g_5_is_248_l135_135846

def g : ℤ → ℤ := sorry

theorem g_5_is_248 :
  (g(1) > 1) ∧
  (∀ x y : ℤ, g(x + y) + x * g(y) + y * g(x) = g(x) * g(y) + x + y + x * y) ∧
  (∀ x : ℤ, 3 * g(x) = g(x + 1) + 2 * x - 1) →
  g(5) = 248 :=
by
  -- proof omitted
  sorry

end g_5_is_248_l135_135846


namespace circumscribed_circle_area_l135_135480

/-- 
Statement: The area of the circle circumscribed about an equilateral triangle with side lengths of 9 units is 27π square units.
-/
theorem circumscribed_circle_area (s : ℕ) (h : s = 9) : 
  let radius := (2/3) * (4.5 * Real.sqrt 3)
  let area := Real.pi * (radius ^ 2)
  area = 27 * Real.pi :=
by
  -- Axis and conditions definitions
  have := h

  -- Definition for the area based on the radius
  let radius := (2/3) * (4.5 * Real.sqrt 3)
  let area := Real.pi * (radius ^ 2)

  -- Statement of the equality to be proven
  show area = 27 * Real.pi
  sorry

end circumscribed_circle_area_l135_135480


namespace quadratic_roots_inverse_sum_l135_135498

theorem quadratic_roots_inverse_sum (t q α β : ℝ) (h1 : α + β = t) (h2 : α * β = q) 
  (h3 : ∀ n : ℕ, n ≥ 1 → α^n + β^n = t) : (1 / α^2011 + 1 / β^2011) = 2 := 
by 
  sorry

end quadratic_roots_inverse_sum_l135_135498


namespace points_on_opposite_sides_of_line_l135_135947

theorem points_on_opposite_sides_of_line (m : ℝ) (h1 : 2 - 1 + m > 0) (h2 : 1 - 3 + m < 0) : -1 < m ∧ m < 2 :=
by
  have h : (m + 1) * (m - 2) < 0 := sorry
  exact sorry

end points_on_opposite_sides_of_line_l135_135947


namespace gasoline_price_percent_increase_l135_135276

theorem gasoline_price_percent_increase 
  (highest_price : ℕ) (lowest_price : ℕ) 
  (h_highest : highest_price = 17) 
  (h_lowest : lowest_price = 10) : 
  (highest_price - lowest_price) * 100 / lowest_price = 70 := 
by 
  sorry

end gasoline_price_percent_increase_l135_135276


namespace michael_ratio_l135_135685

-- Definitions
def Michael_initial := 42
def Brother_initial := 17

-- Conditions
def Brother_after_candy_purchase := 35
def Candy_cost := 3
def Brother_before_candy := Brother_after_candy_purchase + Candy_cost
def x := Brother_before_candy - Brother_initial

-- Prove the ratio of the money Michael gave to his brother to his initial amount is 1:2
theorem michael_ratio :
  x * 2 = Michael_initial := by
  sorry

end michael_ratio_l135_135685


namespace problem_1_problem_2_l135_135079

variable {m n x : ℝ}

theorem problem_1 (m n : ℝ) : (m + n) * (2 * m + n) + n * (m - n) = 2 * m ^ 2 + 4 * m * n := 
by
  sorry

theorem problem_2 (x : ℝ) (h : x ≠ 0) : ((x + 3) / x - 2) / ((x ^ 2 - 9) / (4 * x)) = -(4  / (x + 3)) :=
by
  sorry

end problem_1_problem_2_l135_135079


namespace base_of_second_fraction_l135_135107

theorem base_of_second_fraction (base : ℝ) (h1 : (1/2) ^ 16 * (1/base) ^ 8 = 1 / (18 ^ 16)): base = 81 :=
sorry

end base_of_second_fraction_l135_135107


namespace geometric_sequence_seventh_term_l135_135275

theorem geometric_sequence_seventh_term (a r : ℝ) 
    (h1 : a * r^3 = 8) 
    (h2 : a * r^9 = 2) : 
    a * r^6 = 1 := 
by 
    sorry

end geometric_sequence_seventh_term_l135_135275


namespace perpendicular_case_parallel_case_l135_135933

noncomputable def vector_a : ℝ × ℝ := (1, 2)
noncomputable def vector_b : ℝ × ℝ := (-3, 2)
noncomputable def k_perpendicular : ℝ := 19
noncomputable def k_parallel : ℝ := -1/3

-- Define the operations used:
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

def scalar_mult (k : ℝ) (v : ℝ × ℝ) : ℝ × ℝ :=
  (k * v.1, k * v.2)

def vector_add (v1 v2 : ℝ × ℝ) : ℝ × ℝ :=
  (v1.1 + v2.1, v1.2 + v2.2)

def vector_sub (v1 v2 : ℝ × ℝ) : ℝ × ℝ :=
  (v1.1 - v2.1, v1.2 - v2.2)

-- Perpendicular case: 
theorem perpendicular_case : dot_product (vector_add (scalar_mult k_perpendicular vector_a) vector_b) (vector_sub vector_a (scalar_mult 3 vector_b)) = 0 := sorry

-- Parallel case:
theorem parallel_case : ∃ c : ℝ, vector_add (scalar_mult k_parallel vector_a) vector_b = scalar_mult c (vector_sub vector_a (scalar_mult 3 vector_b)) ∧ c < 0 := sorry

end perpendicular_case_parallel_case_l135_135933


namespace proof_problem_l135_135099

noncomputable def real_numbers (a x y : ℝ) (h₁ : 0 < a ∧ a < 1) (h₂ : a^x < a^y) : Prop :=
  x^3 > y^3

-- The theorem statement
theorem proof_problem (a x y : ℝ) (h₁ : 0 < a) (h₂ : a < 1) (h₃ : a^x < a^y) : x^3 > y^3 :=
by
  sorry

end proof_problem_l135_135099


namespace sum_five_smallest_primes_l135_135883

theorem sum_five_smallest_primes : (2 + 3 + 5 + 7 + 11) = 28 := by
  -- We state the sum of the known five smallest prime numbers.
  sorry

end sum_five_smallest_primes_l135_135883


namespace diagonal_AC_length_l135_135530

variable (A B C D : Type)
variables [InnerProductSpace ℝ V] [CompleteSpace V]
variables (AB BC CD DA : ℝ) (angle_ADC : ℝ)

-- Conditions
notation "length_AB" => AB = 10
notation "length_BC" => BC = 10
notation "length_CD" => CD = 15
notation "length_DA" => DA = 15
notation "angle_ADC_120" => angle_ADC = 120

-- The proof problem statement
theorem diagonal_AC_length : 
  length_AB ∧ length_BC ∧ length_CD ∧ length_DA ∧ angle_ADC_120 → 
  ∃ AC : ℝ, AC = 15 * Real.sqrt 3 :=
by
  intros
  sorry

end diagonal_AC_length_l135_135530


namespace orthocenter_PQR_is_correct_l135_135788

def Point := (ℝ × ℝ × ℝ)

def P : Point := (2, 3, 4)
def Q : Point := (6, 4, 2)
def R : Point := (4, 5, 6)

def orthocenter (P Q R : Point) : Point := sorry

theorem orthocenter_PQR_is_correct : orthocenter P Q R = (3 / 2, 13 / 2, 5) :=
sorry

end orthocenter_PQR_is_correct_l135_135788


namespace mean_value_interior_angles_quadrilateral_l135_135369

theorem mean_value_interior_angles_quadrilateral :
  let n := 4 in
  (n - 2) * 180 / n = 90 :=
by
  let n := 4
  sorry

end mean_value_interior_angles_quadrilateral_l135_135369


namespace gain_percent_l135_135473

variables (MP CP SP : ℝ)

-- problem conditions
axiom h1 : CP = 0.64 * MP
axiom h2 : SP = 0.84 * MP

-- To prove: Gain percent is 31.25%
theorem gain_percent (CP MP SP : ℝ) (h1 : CP = 0.64 * MP) (h2 : SP = 0.84 * MP) :
  ((SP - CP) / CP) * 100 = 31.25 :=
by sorry

end gain_percent_l135_135473


namespace solve_for_ratio_l135_135957

noncomputable def slope_tangent_y_equals_x_squared (x1 : ℝ) : ℝ :=
  2 * x1

noncomputable def slope_tangent_y_equals_x_cubed (x2 : ℝ) : ℝ :=
  3 * x2 * x2

noncomputable def y1_compute (x1 : ℝ) : ℝ :=
  x1 * x1

noncomputable def y2_compute (x2 : ℝ) : ℝ :=
  x2 * x2 * x2

theorem solve_for_ratio (x1 x2 : ℝ)
    (tangent_l_same : slope_tangent_y_equals_x_squared x1 = slope_tangent_y_equals_x_cubed x2)
    (y_tangent_l_same : y1_compute x1 = y2_compute x2) :
  x1 / x2 = 4 / 3 :=
by
  sorry

end solve_for_ratio_l135_135957


namespace optimal_option_l135_135594

theorem optimal_option :
  let initial_investment := 160 : ℝ
  let sales_revenue := 98 : ℝ
  let cost_function (n : ℕ) := 10 * (n ^ 2) - 2 * n : ℝ
  let total_profit_function (n : ℕ) := sales_revenue * n - cost_function n - initial_investment
  let sell_one := 200 : ℝ
  let sell_two := 300 : ℝ
  let option_one_profit := total_profit_function 5 + sell_one - initial_investment
  let option_two_profit := total_profit_function 4 + sell_two - initial_investment
  option_one_profit = 130 ∧ option_two_profit = 220 ∧ option_two_profit > option_one_profit :=
by
  sorry

end optimal_option_l135_135594


namespace mean_value_of_quadrilateral_angles_l135_135432

theorem mean_value_of_quadrilateral_angles 
  (a1 a2 a3 a4 : ℝ) 
  (h_sum : a1 + a2 + a3 + a4 = 360) :
  (a1 + a2 + a3 + a4) / 4 = 90 :=
by
  sorry

end mean_value_of_quadrilateral_angles_l135_135432


namespace sum_mod_17_l135_135218

theorem sum_mod_17 :
  (78 + 79 + 80 + 81 + 82 + 83 + 84 + 85) % 17 = 6 :=
by
  sorry

end sum_mod_17_l135_135218


namespace smallest_class_size_l135_135112

theorem smallest_class_size (n : ℕ) 
  (eight_students_scored_120 : 8 * 120 ≤ n * 92)
  (three_students_scored_115 : 3 * 115 ≤ n * 92)
  (min_score_70 : 70 * n ≤ n * 92)
  (mean_score_92 : (8 * 120 + 3 * 115 + 70 * (n - 11)) / n = 92) :
  n = 25 :=
by
  sorry

end smallest_class_size_l135_135112


namespace scientific_notation_equivalence_l135_135012

theorem scientific_notation_equivalence : 3 * 10^(-7) = 0.0000003 :=
by
  sorry

end scientific_notation_equivalence_l135_135012


namespace triangle_inequality_l135_135679

theorem triangle_inequality
  (a b c : ℝ)
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c)
  (h_ineq : 5 * (a^2 + b^2 + c^2) < 6 * (a * b + b * c + c * a)) :
  a + b > c ∧ b + c > a ∧ c + a > b :=
by
  sorry

end triangle_inequality_l135_135679


namespace aquafaba_needed_for_cakes_l135_135200

def tablespoons_of_aquafaba_for_egg_whites (n_egg_whites : ℕ) : ℕ :=
  2 * n_egg_whites

def total_egg_whites_needed (cakes : ℕ) (egg_whites_per_cake : ℕ) : ℕ :=
  cakes * egg_whites_per_cake

theorem aquafaba_needed_for_cakes (cakes : ℕ) (egg_whites_per_cake : ℕ) :
  tablespoons_of_aquafaba_for_egg_whites (total_egg_whites_needed cakes egg_whites_per_cake) = 32 :=
by
  have h1 : cakes = 2 := sorry
  have h2 : egg_whites_per_cake = 8 := sorry
  sorry

end aquafaba_needed_for_cakes_l135_135200


namespace fraction_identity_l135_135006

variables {a b c x : ℝ}

theorem fraction_identity (h1 : x = a / b) (h2 : a ≠ b) (h3 : b ≠ c) (h4 : c ≠ a) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  (a + 2 * b + 3 * c) / (a - b - 3 * c) = (b * (x + 2) + 3 * c) / (b * (x - 1) - 3 * c) :=
by {
  sorry
}

end fraction_identity_l135_135006


namespace find_eccentricity_of_ellipse_l135_135771

noncomputable def ellipseEccentricity (k : ℝ) : ℝ :=
  let a := Real.sqrt (k + 2)
  let b := Real.sqrt (k + 1)
  let c := Real.sqrt (a^2 - b^2)
  c / a

theorem find_eccentricity_of_ellipse (k : ℝ) (h1 : k + 2 = 4) (h2 : Real.sqrt (k + 2) = 2) :
  ellipseEccentricity k = 1 / 2 := by
  sorry

end find_eccentricity_of_ellipse_l135_135771


namespace find_g5_l135_135849

def g : ℤ → ℤ := sorry

axiom g1 : g 1 > 1
axiom g2 : ∀ x y : ℤ, g (x + y) + x * g y + y * g x = g x * g y + x + y + x * y
axiom g3 : ∀ x : ℤ, 3 * g x = g (x + 1) + 2 * x - 1

theorem find_g5 : g 5 = 248 :=
sorry

end find_g5_l135_135849


namespace divisor_iff_even_l135_135124

noncomputable def hasDivisor (k : ℕ) : Prop := 
  ∃ n : ℕ, n > 0 ∧ (8 * k * n - 1) ∣ (4 * k ^ 2 - 1) ^ 2

theorem divisor_iff_even (k : ℕ) (h : k > 0) : hasDivisor k ↔ (k % 2 = 0) :=
by
  sorry

end divisor_iff_even_l135_135124


namespace attendance_proof_l135_135546

noncomputable def next_year_attendance (this_year: ℕ) := 2 * this_year
noncomputable def last_year_attendance (next_year: ℕ) := next_year - 200
noncomputable def total_attendance (last_year this_year next_year: ℕ) := last_year + this_year + next_year

theorem attendance_proof (this_year: ℕ) (h1: this_year = 600):
    total_attendance (last_year_attendance (next_year_attendance this_year)) this_year (next_year_attendance this_year) = 2800 :=
by
  sorry

end attendance_proof_l135_135546


namespace mean_of_quadrilateral_angles_l135_135384

theorem mean_of_quadrilateral_angles : 
  (∀ (a b c d : ℝ), a + b + c + d = 360 → (a + b + c + d) / 4 = 90) :=
begin
  intros a b c d h,
  calc
    (a + b + c + d) / 4 = 360 / 4 : by rw h
                    ... = 90 : by norm_num,
end

end mean_of_quadrilateral_angles_l135_135384


namespace sum_of_possible_values_of_y_l135_135043

-- Definitions of the conditions
variables (y : ℝ)
-- Angle measures in degrees
variables (a b c : ℝ)
variables (isosceles : Bool)

-- Given conditions
def is_isosceles_triangle (a b c : ℝ) (isosceles : Bool) : Prop :=
  isosceles = true ∧ (a = b ∨ b = c ∨ c = a)

-- Sum of angles in any triangle
def sum_of_angles_in_triangle (a b c : ℝ) : Prop :=
  a + b + c = 180

-- Main statement to be proven
theorem sum_of_possible_values_of_y (y : ℝ) (a b c : ℝ) (isosceles : Bool) :
  is_isosceles_triangle a b c isosceles →
  sum_of_angles_in_triangle a b c →
  ((y = 60) → (a = y ∨ b = y ∨ c = y)) →
  isosceles = true → a = 60 ∨ b = 60 ∨ c = 60 →
  y + y + y = 180 :=
by
  intros h1 h2 h3 h4 h5
  sorry  -- Proof will be provided here

end sum_of_possible_values_of_y_l135_135043


namespace sunny_bakes_initial_cakes_l135_135029

theorem sunny_bakes_initial_cakes (cakes_after_giving_away : ℕ) (total_candles : ℕ) (candles_per_cake : ℕ) (given_away_cakes : ℕ) (initial_cakes : ℕ) :
  cakes_after_giving_away = total_candles / candles_per_cake →
  given_away_cakes = 2 →
  total_candles = 36 →
  candles_per_cake = 6 →
  initial_cakes = cakes_after_giving_away + given_away_cakes →
  initial_cakes = 8 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end sunny_bakes_initial_cakes_l135_135029


namespace part_a_l135_135054

-- Define the sequences and their properties
variables {n : ℕ} (h1 : n ≥ 3)
variables (a b : ℕ → ℝ)
variables (h_arith : ∀ k, a (k+1) = a k + d)
variables (h_geom : ∀ k, b (k+1) = b k * q)
variables (h_a1_b1 : a 1 = b 1)
variables (h_an_bn : a n = b n)

-- State the theorem to be proven
theorem part_a (k : ℕ) (h_k : 2 ≤ k ∧ k ≤ n - 1) : a k > b k :=
  sorry

end part_a_l135_135054


namespace max_cookies_Andy_can_eat_l135_135491

theorem max_cookies_Andy_can_eat 
  (x y : ℕ) 
  (h1 : x + y = 36)
  (h2 : y ≥ 2 * x) : 
  x ≤ 12 := by
  sorry

end max_cookies_Andy_can_eat_l135_135491


namespace arithmetic_sequence_sum_l135_135793

-- Define the arithmetic sequence and the given conditions
def arithmetic_seq (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

-- Define the specific values for the sequence a_1 = 2 and a_2 + a_3 = 13
variables {a : ℕ → ℤ} (d : ℤ)
axiom h1 : a 1 = 2
axiom h2 : a 2 + a 3 = 13

-- Conclude the value of a_4 + a_5 + a_6
theorem arithmetic_sequence_sum : a 4 + a 5 + a 6 = 42 :=
sorry

end arithmetic_sequence_sum_l135_135793


namespace number_of_blue_crayons_given_to_Becky_l135_135014

-- Definitions based on the conditions
def initial_green_crayons : ℕ := 5
def initial_blue_crayons : ℕ := 8
def given_out_green_crayons : ℕ := 3
def total_crayons_left : ℕ := 9

-- Statement of the problem and expected proof
theorem number_of_blue_crayons_given_to_Becky (initial_green_crayons initial_blue_crayons given_out_green_crayons total_crayons_left : ℕ) : 
  initial_green_crayons = 5 →
  initial_blue_crayons = 8 →
  given_out_green_crayons = 3 →
  total_crayons_left = 9 →
  ∃ num_blue_crayons_given_to_Becky, num_blue_crayons_given_to_Becky = 1 :=
by
  intros h1 h2 h3 h4
  sorry

end number_of_blue_crayons_given_to_Becky_l135_135014


namespace larger_triangle_perimeter_is_126_l135_135671

noncomputable def smaller_triangle_side1 : ℝ := 12
noncomputable def smaller_triangle_side2 : ℝ := 12
noncomputable def smaller_triangle_base : ℝ := 18
noncomputable def larger_triangle_longest_side : ℝ := 54
noncomputable def similarity_ratio : ℝ := larger_triangle_longest_side / smaller_triangle_base
noncomputable def larger_triangle_side1 : ℝ := smaller_triangle_side1 * similarity_ratio
noncomputable def larger_triangle_side2 : ℝ := smaller_triangle_side2 * similarity_ratio
noncomputable def larger_triangle_perimeter : ℝ := larger_triangle_side1 + larger_triangle_side2 + larger_triangle_longest_side

theorem larger_triangle_perimeter_is_126 :
  larger_triangle_perimeter = 126 := by
  sorry

end larger_triangle_perimeter_is_126_l135_135671


namespace edward_initial_amount_l135_135088

theorem edward_initial_amount (spent received final_amount : ℤ) 
  (h_spent : spent = 17) 
  (h_received : received = 10) 
  (h_final : final_amount = 7) : 
  ∃ initial_amount : ℤ, (initial_amount - spent + received = final_amount) ∧ (initial_amount = 14) :=
by
  sorry

end edward_initial_amount_l135_135088


namespace even_and_odd_implies_zero_l135_135097

theorem even_and_odd_implies_zero (f : ℝ → ℝ) (h1 : ∀ x : ℝ, f x = -f x) (h2 : ∀ x : ℝ, f (-x) = f x) : ∀ x : ℝ, f x = 0 :=
by
  sorry

end even_and_odd_implies_zero_l135_135097


namespace multiples_of_10_between_11_and_103_l135_135234

def countMultiplesOf10 (lower_bound upper_bound : Nat) : Nat :=
  Nat.div (upper_bound - lower_bound) 10 + 1

theorem multiples_of_10_between_11_and_103 : 
  countMultiplesOf10 11 103 = 9 :=
by
  sorry

end multiples_of_10_between_11_and_103_l135_135234


namespace symmetric_circle_with_respect_to_origin_l135_135946

theorem symmetric_circle_with_respect_to_origin :
  ∀ x y : ℝ, (x + 2) ^ 2 + (y - 1) ^ 2 = 1 → (x - 2) ^ 2 + (y + 1) ^ 2 = 1 :=
by
  intros x y h
  -- Symmetric transformation and verification will be implemented here
  sorry

end symmetric_circle_with_respect_to_origin_l135_135946


namespace mean_of_quadrilateral_angles_l135_135357

theorem mean_of_quadrilateral_angles :
  ∀ (angles : List ℝ), angles.length = 4 → angles.sum = 360 → angles.sum / angles.length = 90 :=
by
  intros
  sorry

end mean_of_quadrilateral_angles_l135_135357


namespace problem_l135_135583

theorem problem (w : ℝ) (h : 13 = 13 * w / (1 - w)) : w^2 = 1 / 4 :=
by
  sorry

end problem_l135_135583


namespace tutors_all_work_together_after_360_days_l135_135669

theorem tutors_all_work_together_after_360_days :
  ∀ (n : ℕ), (n > 0) → 
    (∃ k, k > 0 ∧ k = Nat.lcm 5 (Nat.lcm 8 (Nat.lcm 9 10)) ∧ 
     k % 7 = 3) := by
  sorry

end tutors_all_work_together_after_360_days_l135_135669


namespace ultramen_defeat_monster_in_5_minutes_l135_135710

theorem ultramen_defeat_monster_in_5_minutes :
  ∀ (attacksRequired : ℕ) (attackRate1 attackRate2 : ℕ),
    (attacksRequired = 100) →
    (attackRate1 = 12) →
    (attackRate2 = 8) →
    (attacksRequired / (attackRate1 + attackRate2) = 5) :=
by
  intros
  sorry

end ultramen_defeat_monster_in_5_minutes_l135_135710


namespace numberOfKidsInOtherClass_l135_135556

-- Defining the conditions as given in the problem
def kidsInSwansonClass := 25
def averageZitsSwansonClass := 5
def averageZitsOtherClass := 6
def additionalZitsInOtherClass := 67

-- Total number of zits in Ms. Swanson's class
def totalZitsSwansonClass := kidsInSwansonClass * averageZitsSwansonClass

-- Total number of zits in the other class
def totalZitsOtherClass := totalZitsSwansonClass + additionalZitsInOtherClass

-- Proof that the number of kids in the other class is 32
theorem numberOfKidsInOtherClass : 
  (totalZitsOtherClass / averageZitsOtherClass = 32) :=
by
  -- Proof is left as an exercise.
  sorry

end numberOfKidsInOtherClass_l135_135556


namespace quadratic_complete_square_l135_135684

theorem quadratic_complete_square (b m : ℝ) (h1 : b > 0)
    (h2 : (x : ℝ) → (x + m)^2 + 8 = x^2 + bx + 20) : b = 4 * Real.sqrt 3 :=
by
  sorry

end quadratic_complete_square_l135_135684


namespace mean_of_quadrilateral_angles_is_90_l135_135454

theorem mean_of_quadrilateral_angles_is_90 :
  let n := 4 in
  let sum_of_interior_angles (n : ℕ) := (n - 2) * 180 in
  let quadrilateral_interior_angle_sum := sum_of_interior_angles 4 in
  quadrilateral_interior_angle_sum = 360 →
  360 / 4 = 90 :=
by
  intros n sum_of_interior_angles quadrilateral_interior_angle_sum h
  exact sorry

end mean_of_quadrilateral_angles_is_90_l135_135454


namespace sequence_general_formula_l135_135527

theorem sequence_general_formula (n : ℕ) (hn : n > 0) 
  (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (hS : ∀ n, S n = 1 - n * a n) 
  (hpos : ∀ n, a n > 0) : 
  (a n = 1 / (n * (n + 1))) :=
sorry

end sequence_general_formula_l135_135527


namespace area_of_parallelogram_l135_135126

-- Define the vectors
def v : ℝ × ℝ := (7, -5)
def w : ℝ × ℝ := (14, -4)

-- Prove the area of the parallelogram
theorem area_of_parallelogram : 
  abs (v.1 * w.2 - v.2 * w.1) = 42 :=
by
  sorry

end area_of_parallelogram_l135_135126


namespace f_one_zero_inequality_solution_l135_135579

noncomputable def f : ℝ → ℝ := sorry

axiom increasing_f : ∀ x y, 0 < x → 0 < y → x < y → f x < f y
axiom functional_eq : ∀ x y, 0 < x → 0 < y → f (x / y) = f x - f y
axiom f_six : f 6 = 1

-- Part 1: Prove that f(1) = 0
theorem f_one_zero : f 1 = 0 := sorry

-- Part 2: Prove that ∀ x ∈ (0, (-3 + sqrt 153) / 2), f(x + 3) - f(1 / x) < 2
theorem inequality_solution : ∀ x, 0 < x → x < (-3 + Real.sqrt 153) / 2 → f (x + 3) - f (1 / x) < 2 := sorry

end f_one_zero_inequality_solution_l135_135579


namespace area_rectangle_l135_135563

theorem area_rectangle 
    (x y : ℝ)
    (h1 : 5 * x + 4 * y = 10)
    (h2 : 3 * x = 2 * y) :
    5 * (x * y) = 3000 / 121 :=
by
  sorry

end area_rectangle_l135_135563


namespace mean_value_of_quadrilateral_angles_l135_135335

theorem mean_value_of_quadrilateral_angles : 
  (∀ (A B C D : ℕ), A + B + C + D = 360 → (A + B + C + D) / 4 = 90) :=
by {
  sorry
}

end mean_value_of_quadrilateral_angles_l135_135335


namespace train_length_l135_135484

theorem train_length (speed_kmph : ℝ) (time_sec : ℝ) (h_speed : speed_kmph = 36) (h_time : time_sec = 6.5) : 
  (speed_kmph * 1000 / 3600) * time_sec = 65 := 
by {
  -- Placeholder for proof
  sorry
}

end train_length_l135_135484


namespace circle_condition_l135_135999

theorem circle_condition (m : ℝ) : (∀ x y : ℝ, x^2 + y^2 - x + y + m = 0 → (m < 1 / 2)) :=
by {
-- Skipping the proof here
sorry
}

end circle_condition_l135_135999


namespace sum_and_count_even_l135_135523

-- Sum of integers from a to b (inclusive)
def sum_of_integers (a b : ℕ) : ℕ :=
  (b - a + 1) * (a + b) / 2

-- Number of even integers from a to b (inclusive)
def count_even_integers (a b : ℕ) : ℕ :=
  ((b - if b % 2 == 0 then 0 else 1) - (a + if a % 2 == 0 then 0 else 1)) / 2 + 1

theorem sum_and_count_even (x y : ℕ) :
  x = sum_of_integers 20 40 →
  y = count_even_integers 20 40 →
  x + y = 641 :=
by
  intros
  sorry

end sum_and_count_even_l135_135523


namespace trig_identity_l135_135504

theorem trig_identity (α : ℝ) (h : Real.tan (π - α) = 2) :
  (Real.sin α - Real.cos α) / (Real.sin α + Real.cos α) = 3 :=
by
  sorry

end trig_identity_l135_135504


namespace first_month_sale_eq_6435_l135_135598

theorem first_month_sale_eq_6435 (s2 s3 s4 s5 s6 : ℝ)
  (h2 : s2 = 6927) (h3 : s3 = 6855) (h4 : s4 = 7230) (h5 : s5 = 6562) (h6 : s6 = 7391)
  (avg : ℝ) (h_avg : avg = 6900) :
  let total_sales := 6 * avg
  let other_months_sales := s2 + s3 + s4 + s5 + s6
  let first_month_sale := total_sales - other_months_sales
  first_month_sale = 6435 :=
by
  sorry

end first_month_sale_eq_6435_l135_135598


namespace quadratic_distinct_roots_l135_135797

theorem quadratic_distinct_roots
  (a b c : ℝ)
  (h1 : 5 * a + 3 * b + 2 * c = 0)
  (h2 : a ≠ 0) :
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (a * x1 ^ 2 + b * x1 + c = 0) ∧ (a * x2 ^ 2 + b * x2 + c = 0) :=
by
  sorry

end quadratic_distinct_roots_l135_135797


namespace find_m_find_min_value_l135_135662

-- Conditions
def A (m : ℤ) : Set ℝ := { x | abs (x + 1) + abs (x - m) < 5 }

-- First Problem: Prove m = 3 given 3 ∈ A
theorem find_m (m : ℤ) (h : 3 ∈ A m) : m = 3 := sorry

-- Second Problem: Prove a^2 + b^2 + c^2 ≥ 1 given a + 2b + 2c = 3
theorem find_min_value (a b c : ℝ) (h : a + 2 * b + 2 * c = 3) : (a^2 + b^2 + c^2) ≥ 1 := sorry

end find_m_find_min_value_l135_135662


namespace christine_needs_32_tablespoons_l135_135202

-- Define the conditions
def tablespoons_per_egg_white : ℕ := 2
def egg_whites_per_cake : ℕ := 8
def number_of_cakes : ℕ := 2

-- Define the calculation for total tablespoons of aquafaba needed
def total_tbs_aquafaba : ℕ :=
  tablespoons_per_egg_white * (egg_whites_per_cake * number_of_cakes)

-- The theorem to prove
theorem christine_needs_32_tablespoons :
  total_tbs_aquafaba = 32 :=
by 
  -- Placeholder for proof, as proof steps are not required
  sorry

end christine_needs_32_tablespoons_l135_135202


namespace raffle_prize_l135_135708

theorem raffle_prize (P : ℝ) :
  (0.80 * P = 80) → (P = 100) :=
by
  intro h1
  sorry

end raffle_prize_l135_135708


namespace mean_value_interior_angles_quadrilateral_l135_135372

theorem mean_value_interior_angles_quadrilateral :
  let n := 4 in
  (n - 2) * 180 / n = 90 :=
by
  let n := 4
  sorry

end mean_value_interior_angles_quadrilateral_l135_135372


namespace eqn_has_real_root_in_interval_l135_135520

noncomputable def f (x : ℝ) : ℝ := Real.log (x + 1) + x - 3

theorem eqn_has_real_root_in_interval (k : ℤ) :
  (∃ (x : ℝ), x > k ∧ x < (k + 1) ∧ f x = 0) → k = 2 :=
by
  sorry

end eqn_has_real_root_in_interval_l135_135520


namespace campaign_funds_total_l135_135031

variable (X : ℝ)

def campaign_funds (friends family remaining : ℝ) : Prop :=
  friends = 0.40 * X ∧
  family = 0.30 * (X - friends) ∧
  remaining = X - (friends + family) ∧
  remaining = 4200

theorem campaign_funds_total (X_val : ℝ) (friends family remaining : ℝ)
    (h : campaign_funds X friends family remaining) : X = 10000 :=
by
  have h_friends : friends = 0.40 * X := h.1
  have h_family : family = 0.30 * (X - friends) := h.2.1
  have h_remaining : remaining = X - (friends + family) := h.2.2.1
  have h_remaining_amount : remaining = 4200 := h.2.2.2
  sorry

end campaign_funds_total_l135_135031


namespace product_of_digits_in_base7_7891_is_zero_l135_135048

/-- The function to compute the base 7 representation. -/
def to_base7 (n : ℕ) : List ℕ :=
  if n < 7 then [n]
  else 
    let rest := to_base7 (n / 7)
    rest ++ [n % 7]

/-- The function to compute the product of the digits of a list. -/
def product_of_digits (digits : List ℕ) : ℕ :=
  digits.foldl (λ acc d => acc * d) 1

theorem product_of_digits_in_base7_7891_is_zero :
  product_of_digits (to_base7 7891) = 0 := by
  sorry

end product_of_digits_in_base7_7891_is_zero_l135_135048


namespace mean_value_of_angles_mean_value_of_interior_angles_of_quadrilateral_l135_135348

-- Defining the sum of interior angles of a quadrilateral.
def sum_of_interior_angles (quad : Type) [IsQuadrilateral quad] : ℝ :=
  360

-- Defining the number of angles in a quadrilateral.
def number_of_angles (quad : Type) [IsQuadrilateral quad] : ℕ :=
  4

-- The theorem we want to prove: the mean value of the measures of the four interior angles of any quadrilateral.
theorem mean_value_of_angles (quad : Type) [IsQuadrilateral quad] : ℝ :=
  sum_of_interior_angles quad / number_of_angles quad
 
-- Proving the theorem.
theorem mean_value_of_interior_angles_of_quadrilateral (quad : Type) [IsQuadrilateral quad]:
  mean_value_of_angles quad = 90 :=
by
  sorry

end mean_value_of_angles_mean_value_of_interior_angles_of_quadrilateral_l135_135348


namespace sufficient_condition_for_lg_m_lt_1_l135_135188

theorem sufficient_condition_for_lg_m_lt_1 (m : ℝ) (h1 : m ∈ ({1, 2} : Set ℝ)) : Real.log m < 1 :=
sorry

end sufficient_condition_for_lg_m_lt_1_l135_135188


namespace moles_of_HCl_formed_l135_135215

-- Define the reaction
def balancedReaction (CH4 Cl2 CH3Cl HCl : ℕ) : Prop :=
  CH4 + Cl2 = CH3Cl + HCl

-- Number of moles given
def molesCH4 := 2
def molesCl2 := 4

-- Theorem statement
theorem moles_of_HCl_formed :
  ∀ CH4 Cl2 CH3Cl HCl : ℕ, balancedReaction CH4 Cl2 CH3Cl HCl →
  CH4 = molesCH4 →
  Cl2 = molesCl2 →
  HCl = 2 := sorry

end moles_of_HCl_formed_l135_135215


namespace sufficient_condition_B_is_proper_subset_of_A_l135_135544

def A : Set ℝ := {x | x^2 + x = 6}
def B (m : ℝ) : Set ℝ := {-1 / m}

theorem sufficient_condition_B_is_proper_subset_of_A (m : ℝ) : 
  m = -1/2 → B m ⊆ A ∧ B m ≠ A :=
by
  sorry

end sufficient_condition_B_is_proper_subset_of_A_l135_135544


namespace raft_time_l135_135702

-- Defining the conditions
def distance_between_villages := 1 -- unit: ed (arbitrary unit)

def steamboat_time := 1 -- unit: hours
def motorboat_time := 3 / 4 -- unit: hours
def motorboat_speed_ratio := 2 -- Motorboat speed is twice steamboat speed in still water

-- Speed equations with the current
def steamboat_speed_with_current (v_s v_c : ℝ) := v_s + v_c = 1 -- unit: ed/hr
def motorboat_speed_with_current (v_s v_c : ℝ) := 2 * v_s + v_c = 4 / 3 -- unit: ed/hr

-- Goal: Prove the time it takes for the raft to travel the same distance downstream
theorem raft_time : ∃ t : ℝ, t = 90 :=
by
  -- Definitions
  let v_s := 1 / 3 -- Speed of the steamboat in still water (derived)
  let v_c := 2 / 3 -- Speed of the current (derived)
  let raft_speed := v_c -- Raft speed equals the speed of the current
  
  -- Calculate the time for the raft to travel the distance
  let raft_time := distance_between_villages / raft_speed
  
  -- Convert time to minutes
  let raft_time_minutes := raft_time * 60
  
  -- Prove the raft time is 90 minutes
  existsi raft_time_minutes
  exact sorry

end raft_time_l135_135702


namespace constant_function_odd_iff_zero_l135_135660

theorem constant_function_odd_iff_zero (k : ℝ) (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = k) 
  (h2 : ∀ x, f (-x) = -f x) : 
  k = 0 :=
sorry

end constant_function_odd_iff_zero_l135_135660


namespace geometric_series_sum_l135_135204

theorem geometric_series_sum : 
  let a := 2 in
  let r := 3 in
  let n := 8 in
  ∑ k in Finset.range n, a * r^k = 6560 :=
by 
  let a := 2
  let r := 3
  let n := 8
  have sum_formula : ∑ k in Finset.range n, a * r^k = a * (r^n - 1) / (r - 1) := sorry
  rw sum_formula
  calc
    a * (r^n - 1) / (r - 1) = 2 * (3^8 - 1) / (3 - 1) : by sorry
                            ... = 2 * 6560 / 2 : by sorry
                            ... = 6560 : by sorry

end geometric_series_sum_l135_135204


namespace perimeter_of_C_l135_135298

theorem perimeter_of_C (x y : ℝ) 
  (h₁ : 6 * x + 2 * y = 56) 
  (h₂ : 4 * x + 6 * y = 56) : 
  2 * x + 6 * y = 40 :=
sorry

end perimeter_of_C_l135_135298


namespace range_of_a_for_two_critical_points_l135_135513

noncomputable def f (a x : ℝ) : ℝ := 2 * a^x - Real.exp 1 * x^2 + 18

theorem range_of_a_for_two_critical_points (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ deriv (f a) x1 = 0 ∧ deriv (f a) x2 = 0) ↔ (a ∈ Set.Ioo (1 / Real.exp 1) 1 ∪ Set.Ioo 1 (Real.exp 1)) :=
sorry

end range_of_a_for_two_critical_points_l135_135513


namespace problem1_problem2_problem3_l135_135024

-- (1) Prove 1 - 2(x - y) + (x - y)^2 = (1 - x + y)^2
theorem problem1 (x y : ℝ) : 1 - 2 * (x - y) + (x - y)^2 = (1 - x + y)^2 :=
sorry

-- (2) Prove 25(a - 1)^2 - 10(a - 1) + 1 = (5a - 6)^2
theorem problem2 (a : ℝ) : 25 * (a - 1)^2 - 10 * (a - 1) + 1 = (5 * a - 6)^2 :=
sorry

-- (3) Prove (y^2 - 4y)(y^2 - 4y + 8) + 16 = (y - 2)^4
theorem problem3 (y : ℝ) : (y^2 - 4 * y) * (y^2 - 4 * y + 8) + 16 = (y - 2)^4 :=
sorry

end problem1_problem2_problem3_l135_135024


namespace power_modulo_l135_135882

theorem power_modulo (a b c n : ℕ) (h1 : a = 17) (h2 : b = 1999) (h3 : c = 29) (h4 : n = a^b % c) : 
  n = 17 := 
by
  -- Note: Additional assumptions and intermediate calculations could be provided as needed
  sorry

end power_modulo_l135_135882


namespace function_monotonically_increasing_range_l135_135150

theorem function_monotonically_increasing_range (a : ℝ) :
  (∀ x y : ℝ, x ≤ 1 ∧ y ≤ 1 ∧ x ≤ y → ((4 - a / 2) * x + 2) ≤ ((4 - a / 2) * y + 2)) ∧
  (∀ x y : ℝ, x > 1 ∧ y > 1 ∧ x ≤ y → a^x ≤ a^y) ∧
  (∀ x : ℝ, if x = 1 then a^1 ≥ (4 - a / 2) * 1 + 2 else true) ↔
  4 ≤ a ∧ a < 8 :=
sorry

end function_monotonically_increasing_range_l135_135150


namespace total_payment_l135_135803

/--
  Jerry took 8 hours painting the house. 
  The time to fix the kitchen counter was three times longer than painting the house.
  Jerry took 6 hours mowing the lawn.
  Jerry charged $15 per hour of work.
  Prove that the total amount of money Miss Stevie paid Jerry is $570.
-/
theorem total_payment (h_paint: ℕ := 8) (h_counter: ℕ := 3 * h_paint) (h_mow: ℕ := 6) (rate: ℕ := 15) :
  let total_hours := h_paint + h_counter + h_mow
  in total_hours * rate = 570 :=
by
  -- provide proof here
  sorry

end total_payment_l135_135803


namespace no_such_constant_l135_135256

noncomputable def f : ℚ → ℚ := sorry

theorem no_such_constant (h : ∀ x y : ℚ, ∃ k : ℤ, f (x + y) - f x - f y = k) :
  ¬ ∃ c : ℚ, ∀ x : ℚ, ∃ k : ℤ, f x - c * x = k := 
sorry

end no_such_constant_l135_135256


namespace frisbee_total_distance_l135_135495

-- Definitions for the conditions
def bess_initial_distance : ℝ := 20
def bess_throws : ℕ := 4
def bess_reduction : ℝ := 0.90
def holly_initial_distance : ℝ := 8
def holly_throws : ℕ := 5
def holly_reduction : ℝ := 0.95

-- Function to calculate the total distance for Bess
def total_distance_bess : ℝ :=
  let distances := List.range bess_throws |>.map (λ i => bess_initial_distance * bess_reduction ^ i)
  (distances.sum) * 2

-- Function to calculate the total distance for Holly
def total_distance_holly : ℝ :=
  let distances := List.range holly_throws |>.map (λ i => holly_initial_distance * holly_reduction ^ i)
  distances.sum

-- Proof statement
theorem frisbee_total_distance : 
  total_distance_bess + total_distance_holly = 173.76 :=
by
  sorry

end frisbee_total_distance_l135_135495


namespace first_divisor_is_six_l135_135500

theorem first_divisor_is_six {d : ℕ} 
  (h1: (1394 - 14) % d = 0)
  (h2: (2535 - 1929) % d = 0)
  (h3: (40 - 34) % d = 0)
  : d = 6 :=
sorry

end first_divisor_is_six_l135_135500


namespace perimeter_C_l135_135283

theorem perimeter_C : 
  ∀ {x y : ℕ}, 
    (6 * x + 2 * y = 56) → (4 * x + 6 * y = 56) → 
    (2 * x + 6 * y = 40) :=
by
  intros x y h1 h2
  sorry

end perimeter_C_l135_135283


namespace exists_infinite_bisecting_circles_l135_135778

-- Define circle and bisecting condition
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

def bisects (B C : Circle) : Prop :=
  let chord_len := (B.center.1 - C.center.1)^2 + (B.center.2 - C.center.2)^2 + C.radius^2
  (B.center.1 - C.center.1)^2 + (B.center.2 - C.center.2)^2 + C.radius^2 = B.radius^2

-- Define the theorem statement
theorem exists_infinite_bisecting_circles (C1 C2 : Circle) (h : C1.center ≠ C2.center) :
  ∃ (B : Circle), bisects B C1 ∧ bisects B C2 ∧
  ∀ (b_center : ℝ × ℝ), (∃ (B : Circle), bisects B C1 ∧ bisects B C2 ∧ B.center = b_center) ↔
  2 * (C2.center.1 - C1.center.1) * b_center.1 + 2 * (C2.center.2 - C1.center.2) * b_center.2 =
  (C2.center.1^2 - C1.center.1^2) + (C2.center.2^2 - C1.center.2^2) + (C2.radius^2 - C1.radius^2) := 
sorry

end exists_infinite_bisecting_circles_l135_135778


namespace mean_value_of_quadrilateral_angles_l135_135389

theorem mean_value_of_quadrilateral_angles (a b c d : ℝ) (h_angles_sum : a + b + c + d = 360) : (a + b + c + d) / 4 = 90 := 
by
  sorry

end mean_value_of_quadrilateral_angles_l135_135389


namespace mean_of_quadrilateral_angles_is_90_l135_135457

theorem mean_of_quadrilateral_angles_is_90 :
  let n := 4 in
  let sum_of_interior_angles (n : ℕ) := (n - 2) * 180 in
  let quadrilateral_interior_angle_sum := sum_of_interior_angles 4 in
  quadrilateral_interior_angle_sum = 360 →
  360 / 4 = 90 :=
by
  intros n sum_of_interior_angles quadrilateral_interior_angle_sum h
  exact sorry

end mean_of_quadrilateral_angles_is_90_l135_135457


namespace find_g_x_f_y_l135_135965

-- Definition of the functions and conditions
variable (f g : ℝ → ℝ)
variable (h : ∀ x y : ℝ, f (x + g y) = -x + y + 1)

-- The theorem to prove
theorem find_g_x_f_y (x y : ℝ) : g (x + f y) = -x + y - 1 := 
sorry

end find_g_x_f_y_l135_135965


namespace miniVanTankCapacity_is_65_l135_135668

noncomputable def miniVanTankCapacity : ℝ :=
  let serviceCostPerVehicle := 2.10
  let fuelCostPerLiter := 0.60
  let numMiniVans := 3
  let numTrucks := 2
  let totalCost := 299.1
  let truckFactor := 1.2
  let V := (totalCost - serviceCostPerVehicle * (numMiniVans + numTrucks)) /
            (fuelCostPerLiter * (numMiniVans + numTrucks * (1 + truckFactor)))
  V

theorem miniVanTankCapacity_is_65 : miniVanTankCapacity = 65 :=
  sorry

end miniVanTankCapacity_is_65_l135_135668


namespace mean_of_quadrilateral_angles_l135_135412

theorem mean_of_quadrilateral_angles : 
  (∀ (a b c d : ℝ), a + b + c + d = 360) → 90 :=
by
  intro h
  sorry

end mean_of_quadrilateral_angles_l135_135412


namespace num_of_cows_is_7_l135_135055

variables (C H : ℕ)

-- Define the conditions
def cow_legs : ℕ := 4 * C
def chicken_legs : ℕ := 2 * H
def cow_heads : ℕ := C
def chicken_heads : ℕ := H

def total_legs : ℕ := cow_legs C + chicken_legs H
def total_heads : ℕ := cow_heads C + chicken_heads H
def legs_condition : Prop := total_legs C H = 2 * total_heads C H + 14

-- The theorem to be proved
theorem num_of_cows_is_7 (h : legs_condition C H) : C = 7 :=
by sorry

end num_of_cows_is_7_l135_135055


namespace equivalent_exponentiation_l135_135907

theorem equivalent_exponentiation (h : 64 = 8^2) : 8^15 / 64^3 = 8^9 :=
by
  sorry

end equivalent_exponentiation_l135_135907


namespace mean_value_of_quadrilateral_angles_l135_135426

theorem mean_value_of_quadrilateral_angles : 
  ∀ (a b c d : ℝ), a + b + c + d = 360 → (a + b + c + d) / 4 = 90 :=
by
  intro a b c d h
  sorry

end mean_value_of_quadrilateral_angles_l135_135426


namespace mean_value_of_quadrilateral_angles_l135_135440

theorem mean_value_of_quadrilateral_angles 
  (a1 a2 a3 a4 : ℝ) 
  (h_sum : a1 + a2 + a3 + a4 = 360) :
  (a1 + a2 + a3 + a4) / 4 = 90 :=
by
  sorry

end mean_value_of_quadrilateral_angles_l135_135440


namespace arrange_books_l135_135608

-- Definition of the problem
def total_books : ℕ := 5 + 3

-- Definition of the combination function
def combination (n k : ℕ) : ℕ :=
  n.choose k

-- Prove that arranging 5 copies of Introduction to Geometry and 
-- 3 copies of Introduction to Number Theory into total_books positions can be done in 56 ways.
theorem arrange_books : combination total_books 5 = 56 := by
  sorry

end arrange_books_l135_135608


namespace problem_l135_135780

theorem problem (a b c d : ℝ) (h₁ : a + b = 0) (h₂ : c * d = 1) : 
  (5 * a + 5 * b - 7 * c * d) / (-(c * d) ^ 3) = 7 := 
by
  sorry

end problem_l135_135780


namespace tangent_line_equation_at_x_1_intervals_of_monotonic_increase_l135_135229

noncomputable def f (x : ℝ) := x^3 - x + 3
noncomputable def df (x : ℝ) := 3 * x^2 - 1

theorem tangent_line_equation_at_x_1 : 
  let k := df 1
  let y := f 1
  (2 = k) ∧ (y = 3) ∧ ∀ x y, y - 3 = 2 * (x - 1) ↔ 2 * x - y + 1 = 0 := 
by 
  sorry

theorem intervals_of_monotonic_increase : 
  let x1 := - (Real.sqrt 3) / 3
  let x2 := (Real.sqrt 3) / 3
  ∀ x, (df x > 0 ↔ (x < x1) ∨ (x > x2)) ∧ 
       (df x < 0 ↔ (x1 < x ∧ x < x2)) := 
by 
  sorry

end tangent_line_equation_at_x_1_intervals_of_monotonic_increase_l135_135229


namespace find_k_l135_135503

theorem find_k (x y k : ℝ) (h_line : 2 - k * x = -4 * y) (h_point : x = 3 ∧ y = -2) : k = -2 :=
by
  -- Given the conditions that the point (3, -2) lies on the line 2 - kx = -4y, 
  -- we want to prove that k = -2
  sorry

end find_k_l135_135503


namespace evaluate_expression_l135_135755

theorem evaluate_expression : 
  |-2| + (1 / 4) - 1 - 4 * Real.cos (Real.pi / 4) + Real.sqrt 8 = 5 / 4 :=
by
  sorry

end evaluate_expression_l135_135755


namespace distance_from_P_to_y_axis_l135_135033

theorem distance_from_P_to_y_axis (P : ℝ × ℝ) :
  (P.2 ^ 2 = -12 * P.1) → (dist P (-3, 0) = 9) → abs P.1 = 6 :=
by
  sorry

end distance_from_P_to_y_axis_l135_135033


namespace integral_f_values_l135_135130

open Set Function IntervalIntegral

theorem integral_f_values (f : ℝ → ℝ) 
  (h_cont : ContinuousOn f (Icc 0 1)) 
  (h_range : ∀ x ∈ Icc 0 1, f x ∈ Icc 0 1)
  (h_prop : ∀ x ∈ Icc 0 1, f (f x) = 1) : 
  (3/4 : ℝ) < ∫ x in (0 : ℝ)..(1 : ℝ), f x ∧ ∫ x in (0 : ℝ)..(1 : ℝ), f x ≤ (1 : ℝ) := 
sorry

end integral_f_values_l135_135130


namespace larger_exceeds_smaller_by_5_l135_135696

-- Define the problem's parameters and conditions.
variables (x n m : ℕ)
variables (subtracted : ℕ := 5)

-- Define the two numbers based on the given ratio.
def larger_number := 6 * x
def smaller_number := 5 * x

-- Condition when a number is subtracted
def new_ratio_condition := (larger_number - subtracted) * 4 = (smaller_number - subtracted) * 5

-- The main goal
theorem larger_exceeds_smaller_by_5 (hx : new_ratio_condition) : larger_number - smaller_number = 5 :=
sorry

end larger_exceeds_smaller_by_5_l135_135696


namespace inconsistent_mixture_volume_l135_135596

theorem inconsistent_mixture_volume :
  ∀ (diesel petrol water total_volume : ℚ),
    diesel = 4 →
    petrol = 4 →
    total_volume = 2.666666666666667 →
    diesel + petrol + water = total_volume →
    false :=
by
  intros diesel petrol water total_volume diesel_eq petrol_eq total_volume_eq volume_eq
  rw [diesel_eq, petrol_eq] at volume_eq
  sorry

end inconsistent_mixture_volume_l135_135596


namespace solution_set_inequality_l135_135867

theorem solution_set_inequality : 
  {x : ℝ | abs ((x - 3) / x) > ((x - 3) / x)} = {x : ℝ | 0 < x ∧ x < 3} :=
sorry

end solution_set_inequality_l135_135867


namespace prize_distribution_l135_135095

theorem prize_distribution 
  (total_winners : ℕ)
  (score1 score2 score3 : ℕ)
  (total_points : ℕ) 
  (winners1 winners2 winners3 : ℕ) :
  total_winners = 5 →
  score1 = 20 →
  score2 = 19 →
  score3 = 18 →
  total_points = 94 →
  score1 * winners1 + score2 * winners2 + score3 * winners3 = total_points →
  winners1 + winners2 + winners3 = total_winners →
  winners1 = 1 ∧ winners2 = 2 ∧ winners3 = 2 :=
by
  intros
  sorry

end prize_distribution_l135_135095


namespace find_number_l135_135590

theorem find_number (x : ℝ) (h : 2 = 0.04 * x) : x = 50 := 
sorry

end find_number_l135_135590


namespace minimum_value_of_f_minimum_value_achieved_sum_of_squares_ge_three_l135_135774

noncomputable def f (x : ℝ) : ℝ := |x + 1| + |x - 2|

theorem minimum_value_of_f : ∀ x : ℝ, f x ≥ 3 := by
  sorry

theorem minimum_value_achieved : ∃ x : ℝ, f x = 3 := by
  sorry

theorem sum_of_squares_ge_three (p q r : ℝ) (hp : 0 < p) (hq : 0 < q) (hr : 0 < r) (h : p + q + r = 3) : p^2 + q^2 + r^2 ≥ 3 := by
  sorry

end minimum_value_of_f_minimum_value_achieved_sum_of_squares_ge_three_l135_135774


namespace find_g5_l135_135834

def g : ℤ → ℤ := sorry

axiom g_cond1 : g 1 > 1
axiom g_cond2 : ∀ x y : ℤ, g (x + y) + x * g y + y * g x = g x * g y + x + y + x * y
axiom g_cond3 : ∀ x : ℤ, 3 * g x = g (x + 1) + 2 * x - 1

theorem find_g5 : g 5 = 248 :=
by
  sorry

end find_g5_l135_135834


namespace abs_frac_lt_one_l135_135022

theorem abs_frac_lt_one (x y : ℝ) (hx : |x| < 1) (hy : |y| < 1) :
  |(x - y) / (1 - x * y)| < 1 :=
sorry

end abs_frac_lt_one_l135_135022


namespace banana_cost_l135_135749

theorem banana_cost (pounds: ℕ) (rate: ℕ) (per_pounds: ℕ) : 
 (pounds = 18) → (rate = 3) → (per_pounds = 3) → 
  (pounds / per_pounds * rate = 18) := by
  intros
  sorry

end banana_cost_l135_135749


namespace sum_of_reciprocals_of_roots_l135_135192

theorem sum_of_reciprocals_of_roots 
  (r₁ r₂ : ℝ)
  (h_roots : ∀ (x : ℝ), x^2 - 17*x + 8 = 0 → (∃ r, (r = r₁ ∨ r = r₂) ∧ x = r))
  (h_sum : r₁ + r₂ = 17)
  (h_prod : r₁ * r₂ = 8) :
  1/r₁ + 1/r₂ = 17/8 := 
by
  sorry

end sum_of_reciprocals_of_roots_l135_135192


namespace total_money_before_spending_l135_135476

-- Define the amounts for each friend
variables (J P Q A: ℝ)

-- Define the conditions from the problem
def condition1 := P = 2 * J
def condition2 := Q = P + 20
def condition3 := A = 1.15 * Q
def condition4 := J + P + Q + A = 1211
def cost_of_item : ℝ := 1200

-- The total amount before buying the item
theorem total_money_before_spending (J P Q A : ℝ)
  (h1 : condition1 J P)
  (h2 : condition2 P Q)
  (h3 : condition3 Q A)
  (h4 : condition4 J P Q A) : 
  J + P + Q + A - cost_of_item = 11 :=
by
  sorry

end total_money_before_spending_l135_135476


namespace car_speed_l135_135593

theorem car_speed (t_60 : ℝ := 60) (t_12 : ℝ := 12) (t_dist : ℝ := 1) :
  ∃ v : ℝ, v = 50 ∧ (t_60 / 60 + t_12 = 3600 / v) := 
by
  sorry

end car_speed_l135_135593


namespace range_of_f_l135_135654

noncomputable def f (x : ℝ) : ℝ := 1 / x - 4 / Real.sqrt x + 3

theorem range_of_f : ∀ y, (∃ x, (1/16 : ℝ) ≤ x ∧ x ≤ 1 ∧ f x = y) ↔ -1 ≤ y ∧ y ≤ 3 := by
  sorry

end range_of_f_l135_135654


namespace perimeter_C_l135_135279

theorem perimeter_C : 
  ∀ {x y : ℕ}, 
    (6 * x + 2 * y = 56) → (4 * x + 6 * y = 56) → 
    (2 * x + 6 * y = 40) :=
by
  intros x y h1 h2
  sorry

end perimeter_C_l135_135279


namespace find_prime_pair_l135_135091

noncomputable def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ ∀ m : ℕ, m > 1 ∧ m < n → n % m ≠ 0

def has_integer_root (p q : ℕ) : Prop :=
  ∃ x : ℤ, x^4 + p * x^3 - q = 0

theorem find_prime_pair :
  ∃ (p q : ℕ), is_prime p ∧ is_prime q ∧ has_integer_root p q ∧ p = 2 ∧ q = 3 := by
  sorry

end find_prime_pair_l135_135091


namespace log_sum_even_l135_135059

noncomputable def f (A ω φ x : ℝ) : ℝ := A * Real.sin (ω * x + φ)

-- Define the condition for maximum value at x = 1
def has_max_value_at (f : ℝ → ℝ) (x : ℝ) : Prop :=
  ∀ y : ℝ, f y ≤ f x

-- Main theorem statement: Prove that lg x + lg y is an even function
theorem log_sum_even (A ω φ : ℝ) (hA : 0 < A) (hω : 0 < ω) 
  (hf_max : has_max_value_at (f A ω φ) 1) : 
  ∀ x y : ℝ, Real.log x + Real.log y = Real.log y + Real.log x := by
  sorry

end log_sum_even_l135_135059


namespace mean_value_of_quadrilateral_interior_angles_l135_135460

theorem mean_value_of_quadrilateral_interior_angles (a b c d : ℝ) 
  (h_sum : a + b + c + d = 360) : 
  (a + b + c + d) / 4 = 90 :=
by
  sorry

end mean_value_of_quadrilateral_interior_angles_l135_135460


namespace tommy_profit_l135_135320

noncomputable def total_cost : ℝ := 220 + 375 + 180 + 50 + 30

noncomputable def tomatoes_A : ℝ := 2 * (20 - 4)
noncomputable def oranges_A : ℝ := 2 * (10 - 2)

noncomputable def tomatoes_B : ℝ := 3 * (25 - 5)
noncomputable def oranges_B : ℝ := 3 * (15 - 3)
noncomputable def apples_B : ℝ := 3 * (5 - 1)

noncomputable def tomatoes_C : ℝ := 1 * (30 - 3)
noncomputable def apples_C : ℝ := 1 * (20 - 2)

noncomputable def revenue_A : ℝ := tomatoes_A * 5 + oranges_A * 4
noncomputable def revenue_B : ℝ := tomatoes_B * 6 + oranges_B * 4.5 + apples_B * 3
noncomputable def revenue_C : ℝ := tomatoes_C * 7 + apples_C * 3.5

noncomputable def total_revenue : ℝ := revenue_A + revenue_B + revenue_C

noncomputable def profit : ℝ := total_revenue - total_cost

theorem tommy_profit : profit = 179 :=
by
    sorry

end tommy_profit_l135_135320


namespace number_of_valid_seating_arrangements_l135_135875

def valid_permutations (p : Perm (Fin 6)) : Prop :=
  ¬((p 0 = 1 ∧ p 1 = 2) ∨ (p 1 = 1 ∧ p 0 = 2)) ∧
  ¬((p 2 = 3 ∧ p 3 = 4) ∨ (p 3 = 3 ∧ p 2 = 4)) ∧
  ¬((p 4 = 5 ∧ p 5 = 6) ∨ (p 5 = 5 ∧ p 4 = 6))

theorem number_of_valid_seating_arrangements : 
  let p_set := {p : Perm (Fin 6) // valid_permutations p ∧ p 0 = 0} 
  in Fintype.card p_set = 16 := by
sorry

end number_of_valid_seating_arrangements_l135_135875


namespace find_t_l135_135832

theorem find_t (k m r s t : ℕ) (h1 : k < m) (h2 : m < r) (h3 : r < s) (h4 : s < t)
    (havg : (k + m + r + s + t) / 5 = 18)
    (hmed : r = 23) 
    (hpos_k : 0 < k)
    (hpos_m : 0 < m)
    (hpos_r : 0 < r)
    (hpos_s : 0 < s)
    (hpos_t : 0 < t) :
  t = 40 := sorry

end find_t_l135_135832


namespace total_bees_l135_135582

theorem total_bees 
    (B : ℕ) 
    (h1 : (1/5 : ℚ) * B + (1/3 : ℚ) * B + (2/5 : ℚ) * B + 1 = B) : 
    B = 15 := sorry

end total_bees_l135_135582


namespace marbles_given_by_Joan_l135_135827

def initial_yellow_marbles : ℝ := 86.0
def final_yellow_marbles : ℝ := 111.0

theorem marbles_given_by_Joan :
  final_yellow_marbles - initial_yellow_marbles = 25 := by
  sorry

end marbles_given_by_Joan_l135_135827


namespace figure_C_perimeter_l135_135305

def is_perimeter (figure : Type) (perimeter : ℕ) : Prop :=
∃ x y : ℕ, (figure = 'A' → 6*x + 2*y = perimeter) ∧ 
           (figure = 'B' → 4*x + 6*y = perimeter) ∧
           (figure = 'C' → 2*x + 6*y = perimeter)

theorem figure_C_perimeter (hA : is_perimeter 'A' 56) (hB : is_perimeter 'B' 56) : 
  is_perimeter 'C' 40 :=
by
  sorry

end figure_C_perimeter_l135_135305


namespace largest_possible_value_of_N_l135_135817

theorem largest_possible_value_of_N :
  ∃ N : ℕ, (∀ d : ℕ, (d ∣ N) → (d = 1 ∨ d = N ∨ (∃ k : ℕ, d = 3 ∨ d=k ∨ d=441 / k))) ∧
            ((21 * 3) ∣ N) ∧
            (N = 441) :=
by
  sorry

end largest_possible_value_of_N_l135_135817


namespace rectangle_y_value_l135_135899

theorem rectangle_y_value (y : ℝ) (h1 : -2 < 6) (h2 : y > 2) 
    (h3 : 8 * (y - 2) = 64) : y = 10 :=
by
  sorry

end rectangle_y_value_l135_135899


namespace capsule_depth_equation_l135_135974

theorem capsule_depth_equation (x y z : ℝ) (h : y = 4 * x + z) : y = 4 * x + z := 
by 
  exact h

end capsule_depth_equation_l135_135974


namespace ribbon_left_l135_135267

theorem ribbon_left (gifts : ℕ) (ribbon_per_gift Tom_ribbon_total : ℝ) (h1 : gifts = 8) (h2 : ribbon_per_gift = 1.5) (h3 : Tom_ribbon_total = 15) : Tom_ribbon_total - (gifts * ribbon_per_gift) = 3 := 
by
  sorry

end ribbon_left_l135_135267


namespace average_speed_correct_l135_135677

-- Definitions of distances and speeds
def distance1 := 50 -- miles
def speed1 := 20 -- miles per hour
def distance2 := 20 -- miles
def speed2 := 40 -- miles per hour
def distance3 := 30 -- miles
def speed3 := 15 -- miles per hour

-- Definition of total distance and total time
def total_distance := distance1 + distance2 + distance3
def time1 := distance1 / speed1
def time2 := distance2 / speed2
def time3 := distance3 / speed3
def total_time := time1 + time2 + time3

-- Definition of average speed
def average_speed := total_distance / total_time

-- Statement to be proven
theorem average_speed_correct : average_speed = 20 := 
by 
  -- Proof will be provided here
  sorry

end average_speed_correct_l135_135677


namespace mean_value_of_interior_angles_of_quadrilateral_l135_135401

theorem mean_value_of_interior_angles_of_quadrilateral 
  (sum_of_interior_angles : ℝ := 360) 
  (number_of_angles : ℝ := 4) : 
  sum_of_interior_angles / number_of_angles = 90 := 
by 
  sorry

end mean_value_of_interior_angles_of_quadrilateral_l135_135401


namespace smallest_nth_root_of_unity_l135_135715

theorem smallest_nth_root_of_unity (n : ℕ) (h_pos : n > 0) :
  (∀ z : ℂ, z^6 - z^3 + 1 = 0 → ∃ k : ℕ, z = exp ((2 * ↑k * π * I) / ↑n)) ↔ n = 9 :=
by
  sorry

end smallest_nth_root_of_unity_l135_135715


namespace region_transformation_area_l135_135680

-- Define the region T with area 15
def region_T : ℝ := 15

-- Define the transformation matrix
def matrix_M : Matrix (Fin 2) (Fin 2) ℝ := ![
  ![ 3, 4 ],
  ![ 5, -2 ]
]

-- The determinant of the matrix
def det_matrix_M : ℝ := 3 * (-2) - 4 * 5

-- The proven target statement to show that after the transformation, the area of T' is 390
theorem region_transformation_area :
  ∃ (area_T' : ℝ), area_T' = |det_matrix_M| * region_T ∧ area_T' = 390 :=
by
  sorry

end region_transformation_area_l135_135680


namespace fritz_has_40_dollars_l135_135689

variable (F S R : ℝ)
variable (h1 : S = (1 / 2) * F + 4)
variable (h2 : R = 3 * S)
variable (h3 : R + S = 96)

theorem fritz_has_40_dollars : F = 40 :=
by
  sorry

end fritz_has_40_dollars_l135_135689


namespace flat_fee_first_night_l135_135483

theorem flat_fee_first_night :
  ∃ f n : ℚ, (f + 3 * n = 195) ∧ (f + 6 * n = 350) ∧ (f = 40) :=
by
  -- Skipping the detailed proof:
  sorry

end flat_fee_first_night_l135_135483


namespace cat_mouse_position_after_moves_l135_135953

-- Define the total number of moves
def total_moves : ℕ := 360

-- Define cat's cycle length and position calculation
def cat_cycle_length : ℕ := 5
def cat_final_position := total_moves % cat_cycle_length

-- Define mouse's cycle length and actual moves per cycle
def mouse_cycle_length : ℕ := 10
def mouse_effective_moves_per_cycle : ℕ := 9
def total_mouse_effective_moves := (total_moves / mouse_cycle_length) * mouse_effective_moves_per_cycle
def mouse_final_position := total_mouse_effective_moves % mouse_cycle_length

theorem cat_mouse_position_after_moves :
  cat_final_position = 0 ∧ mouse_final_position = 4 :=
by
  sorry

end cat_mouse_position_after_moves_l135_135953


namespace range_of_a_l135_135972

noncomputable def f (x : ℝ) : ℝ := 
  if x ≥ 0 then x^2 else -x^2

theorem range_of_a (a : ℝ) : (∀ x : ℝ, a ≤ x ∧ x ≤ a + 2 → f (x + a) ≥ 2 * f x) → a ≥ Real.sqrt 2 :=
by
  -- provided condition
  intros h
  sorry

end range_of_a_l135_135972


namespace mean_of_quadrilateral_angles_l135_135385

theorem mean_of_quadrilateral_angles : 
  (∀ (a b c d : ℝ), a + b + c + d = 360 → (a + b + c + d) / 4 = 90) :=
begin
  intros a b c d h,
  calc
    (a + b + c + d) / 4 = 360 / 4 : by rw h
                    ... = 90 : by norm_num,
end

end mean_of_quadrilateral_angles_l135_135385


namespace problem_statement_l135_135039

def horse_lap_times : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

noncomputable def LCM (a b : ℕ) : ℕ := Nat.lcm a b

-- Least common multiple of a set of numbers
noncomputable def LCM_set (s : List ℕ) : ℕ :=
s.foldl LCM 1

-- Calculate the sum of the digits of a number
def sum_of_digits (n : ℕ) : ℕ :=
n.digits 10 |>.sum

theorem problem_statement :
  let T := LCM_set [2, 3, 5, 7, 11, 13]
  sum_of_digits T = 6 := by
  sorry

end problem_statement_l135_135039


namespace problem_statement_l135_135142

theorem problem_statement (x : ℝ) : 
  (∀ (x : ℝ), (100 / x = 80 / (x - 4)) → true) :=
by
  intro x
  sorry

end problem_statement_l135_135142


namespace mean_value_of_interior_angles_of_quadrilateral_l135_135404

theorem mean_value_of_interior_angles_of_quadrilateral 
  (sum_of_interior_angles : ℝ := 360) 
  (number_of_angles : ℝ := 4) : 
  sum_of_interior_angles / number_of_angles = 90 := 
by 
  sorry

end mean_value_of_interior_angles_of_quadrilateral_l135_135404


namespace gain_percentage_is_five_percent_l135_135902

variables (CP SP New_SP Loss Loss_Percentage Gain Gain_Percentage : ℝ)
variables (H1 : Loss_Percentage = 10)
variables (H2 : CP = 933.33)
variables (H3 : Loss = (Loss_Percentage / 100) * CP)
variables (H4 : SP = CP - Loss)
variables (H5 : New_SP = SP + 140)
variables (H6 : Gain = New_SP - CP)
variables (H7 : Gain_Percentage = (Gain / CP) * 100)

theorem gain_percentage_is_five_percent :
  Gain_Percentage = 5 :=
by
  -- Proof goes here
  sorry

end gain_percentage_is_five_percent_l135_135902


namespace actual_total_discount_discount_difference_l135_135073

variable {original_price : ℝ}
variable (first_discount second_discount claimed_discount actual_discount : ℝ)

-- Definitions based on the problem conditions
def discount_1 (p : ℝ) : ℝ := (1 - first_discount) * p
def discount_2 (p : ℝ) : ℝ := (1 - second_discount) * discount_1 first_discount p

-- Statements we need to prove
theorem actual_total_discount (original_price : ℝ)
  (first_discount : ℝ := 0.40) (second_discount : ℝ := 0.30) (claimed_discount : ℝ := 0.70) :
  actual_discount = 1 - discount_2 first_discount second_discount original_price := 
by 
  sorry

theorem discount_difference (original_price : ℝ)
  (first_discount : ℝ := 0.40) (second_discount : ℝ := 0.30) (claimed_discount : ℝ := 0.70)
  (actual_discount : ℝ := 0.58) :
  claimed_discount - actual_discount = 0.12 := 
by 
  sorry

end actual_total_discount_discount_difference_l135_135073


namespace simplify_expression_l135_135762

theorem simplify_expression :
  (∃ (a b c d e f : ℝ), 
    a = (7)^(1/4) ∧ 
    b = (3)^(1/3) ∧ 
    c = (7)^(1/2) ∧ 
    d = (3)^(1/6) ∧ 
    e = (a / b) / (c / d) ∧ 
    f = ((1 / 7)^(1/4)) * ((1 / 3)^(1/6))
    → e = f) :=
by {
  sorry
}

end simplify_expression_l135_135762


namespace fraction_decomposition_roots_sum_l135_135543

theorem fraction_decomposition_roots_sum :
  ∀ (p q r A B C : ℝ),
  p ≠ q → p ≠ r → q ≠ r →
  (∀ (s : ℝ), s ≠ p → s ≠ q → s ≠ r →
          1 / (s^3 - 15 * s^2 + 50 * s - 56) = A / (s - p) + B / (s - q) + C / (s - r)) →
  1 / A + 1 / B + 1 / C = 225 :=
by
  intros p q r A B C hpq hpr hqr hDecomp
  -- Skip proof
  sorry

end fraction_decomposition_roots_sum_l135_135543


namespace canoe_kayak_problem_l135_135712

theorem canoe_kayak_problem (C K : ℕ) 
  (h1 : 9 * C + 12 * K = 432)
  (h2 : C = (4 * K) / 3) : 
  C - K = 6 := by
sorry

end canoe_kayak_problem_l135_135712


namespace expression_equals_6_l135_135574

-- Define the expression as a Lean definition.
def expression : ℤ := 1 - (-2) - 3 - (-4) - 5 - (-6) - 7 - (-8)

-- The statement to prove that the expression equals 6.
theorem expression_equals_6 : expression = 6 := 
by 
  -- This is a placeholder for the actual proof.
  sorry

end expression_equals_6_l135_135574


namespace mean_value_of_angles_mean_value_of_interior_angles_of_quadrilateral_l135_135349

-- Defining the sum of interior angles of a quadrilateral.
def sum_of_interior_angles (quad : Type) [IsQuadrilateral quad] : ℝ :=
  360

-- Defining the number of angles in a quadrilateral.
def number_of_angles (quad : Type) [IsQuadrilateral quad] : ℕ :=
  4

-- The theorem we want to prove: the mean value of the measures of the four interior angles of any quadrilateral.
theorem mean_value_of_angles (quad : Type) [IsQuadrilateral quad] : ℝ :=
  sum_of_interior_angles quad / number_of_angles quad
 
-- Proving the theorem.
theorem mean_value_of_interior_angles_of_quadrilateral (quad : Type) [IsQuadrilateral quad]:
  mean_value_of_angles quad = 90 :=
by
  sorry

end mean_value_of_angles_mean_value_of_interior_angles_of_quadrilateral_l135_135349


namespace total_amount_divided_l135_135063

theorem total_amount_divided (A B C : ℝ) (h1 : A = 2/3 * (B + C)) (h2 : B = 2/3 * (A + C)) (h3 : A = 80) : 
  A + B + C = 200 :=
by
  sorry

end total_amount_divided_l135_135063


namespace tim_meditation_time_l135_135707

-- Definitions of the conditions:
def time_reading_week (t_reading : ℕ) : Prop := t_reading = 14
def twice_as_much_reading (t_reading t_meditate : ℕ) : Prop := t_reading = 2 * t_meditate

-- The theorem to prove:
theorem tim_meditation_time (t_reading t_meditate_per_day : ℕ) 
  (h1 : time_reading_week t_reading)
  (h2 : twice_as_much_reading t_reading (7 * t_meditate_per_day)) :
  t_meditate_per_day = 1 :=
by
  sorry

end tim_meditation_time_l135_135707


namespace mean_value_of_quadrilateral_angles_l135_135439

theorem mean_value_of_quadrilateral_angles 
  (a1 a2 a3 a4 : ℝ) 
  (h_sum : a1 + a2 + a3 + a4 = 360) :
  (a1 + a2 + a3 + a4) / 4 = 90 :=
by
  sorry

end mean_value_of_quadrilateral_angles_l135_135439


namespace average_minutes_per_day_l135_135493

-- Definitions based on the conditions
variables (f : ℕ)
def third_graders := 6 * f
def fourth_graders := 2 * f
def fifth_graders := f

def third_graders_time := 10 * third_graders f
def fourth_graders_time := 12 * fourth_graders f
def fifth_graders_time := 15 * fifth_graders f

def total_students := third_graders f + fourth_graders f + fifth_graders f
def total_time := third_graders_time f + fourth_graders_time f + fifth_graders_time f

-- Proof statement
theorem average_minutes_per_day : total_time f / total_students f = 11 := sorry

end average_minutes_per_day_l135_135493


namespace number_of_doubles_players_l135_135951

theorem number_of_doubles_players (x y : ℕ) 
  (h1 : x + y = 13) 
  (h2 : 4 * x - 2 * y = 4) : 
  4 * x = 20 :=
by sorry

end number_of_doubles_players_l135_135951


namespace chloe_total_score_l135_135889

def points_per_treasure : ℕ := 9
def treasures_first_level : ℕ := 6
def treasures_second_level : ℕ := 3

def score_first_level : ℕ := treasures_first_level * points_per_treasure
def score_second_level : ℕ := treasures_second_level * points_per_treasure
def total_score : ℕ := score_first_level + score_second_level

theorem chloe_total_score : total_score = 81 := by
  sorry

end chloe_total_score_l135_135889


namespace six_digit_quotient_l135_135233

def six_digit_number (A B : ℕ) : ℕ := 100000 * A + 97860 + B

def divisible_by_99 (n : ℕ) : Prop := n % 99 = 0

theorem six_digit_quotient (A B : ℕ) (hA : A = 5) (hB : B = 1)
  (h9786B : divisible_by_99 (six_digit_number A B)) : 
  six_digit_number A B / 99 = 6039 := by
  sorry

end six_digit_quotient_l135_135233


namespace log_inequality_l135_135129

theorem log_inequality (a b c : ℝ) (ha : a ≥ 2) (hb : b ≥ 2) (hc : c ≥ 2) :
    (Real.log (c ^ 2) / Real.log (a + b) + Real.log (a ^ 2) / Real.log (b + c) + Real.log (b ^ 2) / Real.log (c + a)) ≥ 3 :=
sorry

end log_inequality_l135_135129


namespace interest_difference_l135_135515

theorem interest_difference (P P_B : ℝ) (R_A R_B T : ℝ)
    (h₁ : P = 10000)
    (h₂ : P_B = 4000.0000000000005)
    (h₃ : R_A = 15)
    (h₄ : R_B = 18)
    (h₅ : T = 2) :
    let P_A := P - P_B
    let I_A := (P_A * R_A * T) / 100
    let I_B := (P_B * R_B * T) / 100
    I_A - I_B = 359.99999999999965 := 
by
  sorry

end interest_difference_l135_135515


namespace expression_range_l135_135632

theorem expression_range (a b c x : ℝ) (h : a^2 + b^2 + c^2 ≠ 0) :
  ∃ y : ℝ, y = (a * Real.cos x + b * Real.sin x + c) / (Real.sqrt (a^2 + b^2 + c^2)) 
           ∧ y ∈ Set.Icc (-Real.sqrt 3) (Real.sqrt 3) :=
sorry

end expression_range_l135_135632


namespace perimeter_C_l135_135280

theorem perimeter_C : 
  ∀ {x y : ℕ}, 
    (6 * x + 2 * y = 56) → (4 * x + 6 * y = 56) → 
    (2 * x + 6 * y = 40) :=
by
  intros x y h1 h2
  sorry

end perimeter_C_l135_135280


namespace lara_flowers_l135_135005

theorem lara_flowers (M : ℕ) : 52 - M - (M + 6) - 16 = 0 → M = 15 :=
by
  sorry

end lara_flowers_l135_135005


namespace customers_sampling_candy_l135_135172

theorem customers_sampling_candy (total_customers caught fined not_caught : ℝ) 
    (h1 : total_customers = 100) 
    (h2 : caught = 0.22 * total_customers) 
    (h3 : not_caught / (caught / 0.9) = 0.1) :
    (not_caught + caught) / total_customers = 0.2444 := 
by sorry

end customers_sampling_candy_l135_135172


namespace ribbon_left_after_wrapping_l135_135265

def total_ribbon_needed (gifts : ℕ) (ribbon_per_gift : ℝ) : ℝ :=
  gifts * ribbon_per_gift

def remaining_ribbon (initial_ribbon : ℝ) (used_ribbon : ℝ) : ℝ :=
  initial_ribbon - used_ribbon

theorem ribbon_left_after_wrapping : 
  ∀ (gifts : ℕ) (ribbon_per_gift initial_ribbon : ℝ),
  gifts = 8 →
  ribbon_per_gift = 1.5 →
  initial_ribbon = 15 →
  remaining_ribbon initial_ribbon (total_ribbon_needed gifts ribbon_per_gift) = 3 :=
by
  intros gifts ribbon_per_gift initial_ribbon h1 h2 h3
  rw [h1, h2, h3]
  simp [total_ribbon_needed, remaining_ribbon]
  sorry

end ribbon_left_after_wrapping_l135_135265


namespace mean_value_of_quadrilateral_angles_l135_135415

theorem mean_value_of_quadrilateral_angles (angles : Fin 4 → ℝ) 
  (h_sum : ∑ i, angles i = 360) : ∑ i, angles i / 4 = 90 :=
by
  sorry

end mean_value_of_quadrilateral_angles_l135_135415


namespace mean_value_of_quadrilateral_angles_l135_135435

theorem mean_value_of_quadrilateral_angles 
  (a1 a2 a3 a4 : ℝ) 
  (h_sum : a1 + a2 + a3 + a4 = 360) :
  (a1 + a2 + a3 + a4) / 4 = 90 :=
by
  sorry

end mean_value_of_quadrilateral_angles_l135_135435


namespace arithmetic_sequence_sum_l135_135870

noncomputable def sum_first_ten_terms (a d : ℕ) : ℕ :=
  (10 / 2) * (2 * a + (10 - 1) * d)

theorem arithmetic_sequence_sum 
  (a d : ℕ) 
  (h1 : a + 2 * d = 8) 
  (h2 : a + 5 * d = 14) :
  sum_first_ten_terms a d = 130 :=
by
  sorry

end arithmetic_sequence_sum_l135_135870


namespace lcm_5_711_is_3555_l135_135165

theorem lcm_5_711_is_3555 : Nat.lcm 5 711 = 3555 := by
  sorry

end lcm_5_711_is_3555_l135_135165


namespace mean_value_of_quadrilateral_angles_l135_135417

theorem mean_value_of_quadrilateral_angles (angles : Fin 4 → ℝ) 
  (h_sum : ∑ i, angles i = 360) : ∑ i, angles i / 4 = 90 :=
by
  sorry

end mean_value_of_quadrilateral_angles_l135_135417


namespace mean_value_of_quadrilateral_angles_l135_135338

theorem mean_value_of_quadrilateral_angles : 
  (∀ (A B C D : ℕ), A + B + C + D = 360 → (A + B + C + D) / 4 = 90) :=
by {
  sorry
}

end mean_value_of_quadrilateral_angles_l135_135338


namespace tangent_line_equation_l135_135102

theorem tangent_line_equation :
  ∃ (P : ℝ × ℝ) (m : ℝ), 
  P = (-2, 15) ∧ m = 2 ∧ 
  (∀ (x y : ℝ), (y = x^3 - 10 * x + 3) → (y - 15 = 2 * (x + 2))) :=
sorry

end tangent_line_equation_l135_135102


namespace algebraic_expression_correct_l135_135222

variable (x y : ℤ)

theorem algebraic_expression_correct (h : (x - y) / (x + y) = 3) : (2 * (x - y)) / (x + y) - (x + y) / (3 * (x - y)) = 53 / 9 := 
by  
  sorry

end algebraic_expression_correct_l135_135222


namespace mean_value_of_quadrilateral_angles_l135_135434

theorem mean_value_of_quadrilateral_angles 
  (a1 a2 a3 a4 : ℝ) 
  (h_sum : a1 + a2 + a3 + a4 = 360) :
  (a1 + a2 + a3 + a4) / 4 = 90 :=
by
  sorry

end mean_value_of_quadrilateral_angles_l135_135434


namespace lunch_break_duration_l135_135828

def rate_sandra : ℝ := 0 -- Sandra's painting rate in houses per hour
def rate_helpers : ℝ := 0 -- Combined rate of the three helpers in houses per hour
def lunch_break : ℝ := 0 -- Lunch break duration in hours

axiom monday_condition : (8 - lunch_break) * (rate_sandra + rate_helpers) = 0.6
axiom tuesday_condition : (6 - lunch_break) * rate_helpers = 0.3
axiom wednesday_condition : (2 - lunch_break) * rate_sandra = 0.1

theorem lunch_break_duration : lunch_break = 0.5 :=
by {
  sorry
}

end lunch_break_duration_l135_135828


namespace kolya_start_time_l135_135494

-- Definitions of conditions as per the initial problem statement
def angle_moved_by_minute_hand (x : ℝ) : ℝ := 6 * x
def angle_moved_by_hour_hand (x : ℝ) : ℝ := 30 + 0.5 * x

theorem kolya_start_time (x : ℝ) :
  (angle_moved_by_minute_hand x = (angle_moved_by_hour_hand x + angle_moved_by_hour_hand x + 60) / 2) ∨
  (angle_moved_by_minute_hand x - 180 = (angle_moved_by_hour_hand x + angle_moved_by_hour_hand x + 60) / 2) :=
sorry

end kolya_start_time_l135_135494


namespace bird_cages_count_l135_135070

/-- 
If each bird cage contains 2 parrots and 2 parakeets,
and the total number of birds is 36,
then the number of bird cages is 9.
-/
theorem bird_cages_count (parrots_per_cage parakeets_per_cage total_birds cages : ℕ)
  (h1 : parrots_per_cage = 2)
  (h2 : parakeets_per_cage = 2)
  (h3 : total_birds = 36)
  (h4 : total_birds = (parrots_per_cage + parakeets_per_cage) * cages) :
  cages = 9 := 
by 
  sorry

end bird_cages_count_l135_135070


namespace orthocenter_PQR_is_correct_l135_135789

def Point := (ℝ × ℝ × ℝ)

def P : Point := (2, 3, 4)
def Q : Point := (6, 4, 2)
def R : Point := (4, 5, 6)

def orthocenter (P Q R : Point) : Point := sorry

theorem orthocenter_PQR_is_correct : orthocenter P Q R = (3 / 2, 13 / 2, 5) :=
sorry

end orthocenter_PQR_is_correct_l135_135789


namespace arithmetic_seq_product_of_first_two_terms_l135_135149

theorem arithmetic_seq_product_of_first_two_terms
    (a d : ℤ)
    (h1 : a + 4 * d = 17)
    (h2 : d = 2) :
    (a * (a + d) = 99) := 
by
    -- Proof to be done
    sorry

end arithmetic_seq_product_of_first_two_terms_l135_135149


namespace mean_value_of_quadrilateral_angles_l135_135414

theorem mean_value_of_quadrilateral_angles (angles : Fin 4 → ℝ) 
  (h_sum : ∑ i, angles i = 360) : ∑ i, angles i / 4 = 90 :=
by
  sorry

end mean_value_of_quadrilateral_angles_l135_135414


namespace least_positive_integer_l135_135880

theorem least_positive_integer (n : ℕ) :
  n % 3 = 1 ∧
  n % 5 = 3 ∧
  n % 6 = 5 ∧
  n % 7 = 2 ↔
  n = 83 :=
by
  sorry

end least_positive_integer_l135_135880


namespace quadratic_inequality_solution_l135_135221

theorem quadratic_inequality_solution (x : ℝ) : 3 * x^2 - 5 * x - 8 > 0 ↔ x < -4/3 ∨ x > 2 :=
by
  sorry

end quadratic_inequality_solution_l135_135221


namespace hot_dogs_served_today_l135_135486

-- Define the number of hot dogs served during lunch
def h_dogs_lunch : ℕ := 9

-- Define the number of hot dogs served during dinner
def h_dogs_dinner : ℕ := 2

-- Define the total number of hot dogs served today
def total_h_dogs : ℕ := h_dogs_lunch + h_dogs_dinner

-- Theorem stating that the total number of hot dogs served today is 11
theorem hot_dogs_served_today : total_h_dogs = 11 := by
  sorry

end hot_dogs_served_today_l135_135486


namespace right_triangle_area_perimeter_l135_135783

theorem right_triangle_area_perimeter (a b : ℕ) (h₁ : a = 36) (h₂ : b = 48) : 
  (1/2) * (a * b) = 864 ∧ a + b + Nat.sqrt (a * a + b * b) = 144 := by
  sorry

end right_triangle_area_perimeter_l135_135783


namespace perimeter_of_C_l135_135294

theorem perimeter_of_C (x y : ℝ) 
  (h₁ : 6 * x + 2 * y = 56) 
  (h₂ : 4 * x + 6 * y = 56) : 
  2 * x + 6 * y = 40 :=
sorry

end perimeter_of_C_l135_135294


namespace earnings_last_friday_l135_135535

theorem earnings_last_friday 
  (price_per_kg : ℕ := 2)
  (earnings_wednesday : ℕ := 30)
  (earnings_today : ℕ := 42)
  (total_kg_sold : ℕ := 48)
  (total_earnings : ℕ := total_kg_sold * price_per_kg) 
  (F : ℕ) :
  earnings_wednesday + F + earnings_today = total_earnings → F = 24 := by
  sorry

end earnings_last_friday_l135_135535


namespace problem_statement_l135_135141

theorem problem_statement (x : ℝ) : 
  (∀ (x : ℝ), (100 / x = 80 / (x - 4)) → true) :=
by
  intro x
  sorry

end problem_statement_l135_135141


namespace range_of_m_l135_135772

variable (f : ℝ → ℝ) (m : ℝ)

-- Given conditions
def condition1 := ∀ x, f (-x) = -f x -- f(x) is an odd function
def condition2 := ∀ x, f (x + 3) = f x -- f(x) has a minimum positive period of 3
def condition3 := f 2015 > 1 -- f(2015) > 1
def condition4 := f 1 = (2 * m + 3) / (m - 1) -- f(1) = (2m + 3) / (m - 1)

-- We aim to prove that -2/3 < m < 1 given these conditions.
theorem range_of_m : condition1 f → condition2 f → condition3 f → condition4 f m → -2 / 3 < m ∧ m < 1 := by
  intros
  sorry

end range_of_m_l135_135772


namespace amount_paid_is_correct_l135_135800

-- Define the conditions
def time_painting_house : ℕ := 8
def time_fixing_counter := 3 * time_painting_house
def time_mowing_lawn : ℕ := 6
def hourly_rate : ℕ := 15

-- Define the total time worked
def total_time_worked := time_painting_house + time_fixing_counter + time_mowing_lawn

-- Define the total amount paid
def total_amount_paid := total_time_worked * hourly_rate

-- Formalize the goal
theorem amount_paid_is_correct : total_amount_paid = 570 :=
by
  -- Proof steps to be filled in
  sorry  -- Placeholder for the proof

end amount_paid_is_correct_l135_135800


namespace mean_value_of_quadrilateral_angles_l135_135395

theorem mean_value_of_quadrilateral_angles (a b c d : ℝ) (h_angles_sum : a + b + c + d = 360) : (a + b + c + d) / 4 = 90 := 
by
  sorry

end mean_value_of_quadrilateral_angles_l135_135395


namespace total_students_correct_l135_135156

-- Given conditions
def number_of_buses : ℕ := 95
def number_of_seats_per_bus : ℕ := 118

-- Definition for the total number of students
def total_number_of_students : ℕ := number_of_buses * number_of_seats_per_bus

-- Problem statement
theorem total_students_correct :
  total_number_of_students = 11210 :=
by
  -- Proof is omitted, hence we use sorry.
  sorry

end total_students_correct_l135_135156


namespace least_non_lucky_multiple_of_10_l135_135599

def is_lucky (n : ℕ) : Prop :=
  n % (n.digits 10).sum = 0

theorem least_non_lucky_multiple_of_10 : 
  ∃ n : ℕ, n % 10 = 0 ∧ ¬is_lucky n ∧ (∀ m : ℕ, m % 10 = 0 ∧ ¬is_lucky m → m ≥ n) ∧ n = 110 :=
by
  sorry

end least_non_lucky_multiple_of_10_l135_135599


namespace evaluate_expression_l135_135923

theorem evaluate_expression :
  let x := (1/4 : ℚ)
  let y := (1/3 : ℚ)
  let z := (-12 : ℚ)
  let w := (5 : ℚ)
  x^2 * y^3 * z + w = (179/36 : ℚ) :=
by
  sorry

end evaluate_expression_l135_135923


namespace jason_books_is_21_l135_135002

def keith_books : ℕ := 20
def total_books : ℕ := 41

theorem jason_books_is_21 (jason_books : ℕ) : 
  jason_books + keith_books = total_books → 
  jason_books = 21 := 
by 
  intro h
  sorry

end jason_books_is_21_l135_135002


namespace factorization_of_a_square_minus_one_l135_135212

theorem factorization_of_a_square_minus_one (a : ℤ) : a^2 - 1 = (a + 1) * (a - 1) := 
  by sorry

end factorization_of_a_square_minus_one_l135_135212


namespace find_a_5_l135_135942

theorem find_a_5 (a : ℕ → ℤ) (h₁ : ∀ n : ℕ, n > 0 → a (n + 1) = a n - 1)
  (h₂ : a 2 + a 4 + a 6 = 18) : a 5 = 5 := 
sorry

end find_a_5_l135_135942


namespace curve_not_parabola_l135_135944

theorem curve_not_parabola (k : ℝ) : ¬ (∃ (a b c : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ c = 1 ∧ a * x^2 + b * y = c) :=
sorry

end curve_not_parabola_l135_135944


namespace find_b_value_l135_135034

theorem find_b_value (b : ℚ) (x : ℚ) (h1 : 3 * x + 9 = 0) (h2 : b * x + 15 = 5) : b = 10 / 3 :=
by
  sorry

end find_b_value_l135_135034


namespace mean_of_quadrilateral_angles_l135_135378

theorem mean_of_quadrilateral_angles : 
  (∀ (a b c d : ℝ), a + b + c + d = 360 → (a + b + c + d) / 4 = 90) :=
begin
  intros a b c d h,
  calc
    (a + b + c + d) / 4 = 360 / 4 : by rw h
                    ... = 90 : by norm_num,
end

end mean_of_quadrilateral_angles_l135_135378


namespace mean_value_of_angles_mean_value_of_interior_angles_of_quadrilateral_l135_135346

-- Defining the sum of interior angles of a quadrilateral.
def sum_of_interior_angles (quad : Type) [IsQuadrilateral quad] : ℝ :=
  360

-- Defining the number of angles in a quadrilateral.
def number_of_angles (quad : Type) [IsQuadrilateral quad] : ℕ :=
  4

-- The theorem we want to prove: the mean value of the measures of the four interior angles of any quadrilateral.
theorem mean_value_of_angles (quad : Type) [IsQuadrilateral quad] : ℝ :=
  sum_of_interior_angles quad / number_of_angles quad
 
-- Proving the theorem.
theorem mean_value_of_interior_angles_of_quadrilateral (quad : Type) [IsQuadrilateral quad]:
  mean_value_of_angles quad = 90 :=
by
  sorry

end mean_value_of_angles_mean_value_of_interior_angles_of_quadrilateral_l135_135346


namespace mean_value_of_interior_angles_of_quadrilateral_l135_135362

theorem mean_value_of_interior_angles_of_quadrilateral :
  (360 / 4) = 90 := 
by
  sorry

end mean_value_of_interior_angles_of_quadrilateral_l135_135362


namespace R_depends_on_d_and_n_l135_135969

variable (n a d : ℕ)

noncomputable def s1 : ℕ := (n * (2 * a + (n - 1) * d)) / 2
noncomputable def s2 : ℕ := (2 * n * (2 * a + (2 * n - 1) * d)) / 2
noncomputable def s3 : ℕ := (3 * n * (2 * a + (3 * n - 1) * d)) / 2
noncomputable def R : ℕ := s3 n a d - s2 n a d - s1 n a d

theorem R_depends_on_d_and_n : R n a d = 2 * d * n^2 :=
by
  sorry

end R_depends_on_d_and_n_l135_135969


namespace tan_double_angle_l135_135659

theorem tan_double_angle (theta : ℝ) (h : 2 * Real.sin theta + Real.cos theta = 0) :
  Real.tan (2 * theta) = - 4 / 3 :=
sorry

end tan_double_angle_l135_135659


namespace find_fruit_juice_amount_l135_135082

def total_punch : ℕ := 14 * 10
def mountain_dew : ℕ := 6 * 12
def ice : ℕ := 28
def fruit_juice : ℕ := total_punch - mountain_dew - ice

theorem find_fruit_juice_amount : fruit_juice = 40 := by
  sorry

end find_fruit_juice_amount_l135_135082


namespace simplify_expression_l135_135984

variable (a b : ℚ)

theorem simplify_expression (ha : a = -2) (hb : b = 1/5) :
  2 * (a^2 * b - 2 * a * b) - 3 * (a^2 * b - 3 * a * b) + a^2 * b = -2 := by
  -- Proof can be filled here
  sorry

end simplify_expression_l135_135984


namespace max_squares_no_common_point_l135_135626

theorem max_squares_no_common_point :
  let n := 7 in
  let total_cells := 6 * (n * n * 4) in
  let max_squares_corner := 8 in
  let cells_per_corner_square := 15 in
  let cells_per_noncorner_square := 16 in
  ∃ (x y : ℕ), x + y = 74 ∧ cells_per_corner_square * x + cells_per_noncorner_square * y ≤ total_cells ∧ x ≤ max_squares_corner :=
begin
  sorry
end

end max_squares_no_common_point_l135_135626


namespace molecular_weight_cao_is_correct_l135_135496

-- Define the atomic weights of calcium and oxygen
def atomic_weight_ca : ℝ := 40.08
def atomic_weight_o : ℝ := 16.00

-- Define the molecular weight of CaO
def molecular_weight_cao : ℝ := atomic_weight_ca + atomic_weight_o

-- State the theorem to prove
theorem molecular_weight_cao_is_correct : molecular_weight_cao = 56.08 :=
by
  sorry

end molecular_weight_cao_is_correct_l135_135496


namespace value_of_f_1_plus_g_4_l135_135971

def f (x : Int) : Int := 2 * x - 1
def g (x : Int) : Int := x + 1

theorem value_of_f_1_plus_g_4 : f (1 + g 4) = 11 := by
  sorry

end value_of_f_1_plus_g_4_l135_135971


namespace integer_pairs_summing_to_six_l135_135763

theorem integer_pairs_summing_to_six :
  ∃ m n : ℤ, m + n + m * n = 6 ∧ ((m = 0 ∧ n = 6) ∨ (m = 6 ∧ n = 0)) :=
by
  sorry

end integer_pairs_summing_to_six_l135_135763


namespace sequence_bound_l135_135158

theorem sequence_bound
  (a : ℕ → ℕ)
  (h_base0 : a 0 < a 1)
  (h_base1 : 0 < a 0 ∧ 0 < a 1)
  (h_recur : ∀ n, 2 ≤ n → a n = 3 * a (n - 1) - 2 * a (n - 2)) :
  a 100 > 2^99 :=
by
  sorry

end sequence_bound_l135_135158


namespace op_example_l135_135085

def myOp (c d : Int) : Int :=
  c * (d + 1) + c * d

theorem op_example : myOp 5 (-2) = -15 := 
  by
    sorry

end op_example_l135_135085


namespace wicket_keeper_older_than_captain_l135_135553

-- Define the team and various ages
def captain_age : ℕ := 28
def average_age_team : ℕ := 25
def number_of_players : ℕ := 11
def number_of_remaining_players : ℕ := number_of_players - 2
def average_age_remaining_players : ℕ := average_age_team - 1

theorem wicket_keeper_older_than_captain :
  ∃ (W : ℕ), W = captain_age + 3 ∧
  275 = number_of_players * average_age_team ∧
  216 = number_of_remaining_players * average_age_remaining_players ∧
  59 = 275 - 216 ∧
  W = 59 - captain_age :=
by
  sorry

end wicket_keeper_older_than_captain_l135_135553


namespace mean_value_of_quadrilateral_angles_l135_135420

theorem mean_value_of_quadrilateral_angles (angles : Fin 4 → ℝ) 
  (h_sum : ∑ i, angles i = 360) : ∑ i, angles i / 4 = 90 :=
by
  sorry

end mean_value_of_quadrilateral_angles_l135_135420


namespace mean_value_of_angles_mean_value_of_interior_angles_of_quadrilateral_l135_135344

-- Defining the sum of interior angles of a quadrilateral.
def sum_of_interior_angles (quad : Type) [IsQuadrilateral quad] : ℝ :=
  360

-- Defining the number of angles in a quadrilateral.
def number_of_angles (quad : Type) [IsQuadrilateral quad] : ℕ :=
  4

-- The theorem we want to prove: the mean value of the measures of the four interior angles of any quadrilateral.
theorem mean_value_of_angles (quad : Type) [IsQuadrilateral quad] : ℝ :=
  sum_of_interior_angles quad / number_of_angles quad
 
-- Proving the theorem.
theorem mean_value_of_interior_angles_of_quadrilateral (quad : Type) [IsQuadrilateral quad]:
  mean_value_of_angles quad = 90 :=
by
  sorry

end mean_value_of_angles_mean_value_of_interior_angles_of_quadrilateral_l135_135344


namespace find_g5_l135_135850

def g : ℤ → ℤ := sorry

axiom g1 : g 1 > 1
axiom g2 : ∀ x y : ℤ, g (x + y) + x * g y + y * g x = g x * g y + x + y + x * y
axiom g3 : ∀ x : ℤ, 3 * g x = g (x + 1) + 2 * x - 1

theorem find_g5 : g 5 = 248 :=
sorry

end find_g5_l135_135850


namespace first_discount_percentage_l135_135611

theorem first_discount_percentage (original_price final_price : ℝ)
  (first_discount second_discount : ℝ) (h_orig : original_price = 200)
  (h_final : final_price = 144) (h_second_disc : second_discount = 0.20) :
  first_discount = 0.10 :=
by
  sorry

end first_discount_percentage_l135_135611


namespace germination_estimate_l135_135245

theorem germination_estimate (germination_rate : ℝ) (total_pounds : ℝ) 
  (hrate_nonneg : 0 ≤ germination_rate) (hrate_le_one : germination_rate ≤ 1) 
  (h_germination_value : germination_rate = 0.971) 
  (h_total_pounds_value : total_pounds = 1000) : 
  total_pounds * (1 - germination_rate) = 29 := 
by 
  sorry

end germination_estimate_l135_135245


namespace numberOfWaysToPlaceCoinsSix_l135_135030

def numberOfWaysToPlaceCoins (n : ℕ) : ℕ :=
  if n = 0 then 1 else 2 * numberOfWaysToPlaceCoins (n - 1)

theorem numberOfWaysToPlaceCoinsSix : numberOfWaysToPlaceCoins 6 = 32 :=
by
  sorry

end numberOfWaysToPlaceCoinsSix_l135_135030


namespace problem_l135_135637

theorem problem (m n : ℚ) (h : m - n = -2/3) : 7 - 3 * m + 3 * n = 9 := 
by {
  -- Place a sorry here as we do not provide the proof 
  sorry
}

end problem_l135_135637


namespace perimeter_C_correct_l135_135300

variables (x y : ℕ)

def perimeter_A (x y : ℕ) := 6 * x + 2 * y
def perimeter_B (x y : ℕ) := 4 * x + 6 * y
def perimeter_C (x y : ℕ) := 2 * x + 6 * y

theorem perimeter_C_correct (x y : ℕ) (h1 : 6 * x + 2 * y = 56) (h2 : 4 * x + 6 * y = 56) :
  2 * x + 6 * y = 40 :=
sorry

end perimeter_C_correct_l135_135300


namespace pasha_wins_9_games_l135_135138

theorem pasha_wins_9_games :
  ∃ w l : ℕ, (w + l = 12) ∧ (2^w * (2^l - 1) - (2^l - 1) * 2^(w - 1) = 2023) ∧ (w = 9) :=
by
  sorry

end pasha_wins_9_games_l135_135138


namespace inequality_proof_l135_135009

noncomputable def inequality (x y z : ℝ) : Prop :=
  (x + y) / z + (y + z) / x + (z + x) / y ≥ 7

theorem inequality_proof (x y z : ℝ) (hx : x ≥ y + z) (hx_pos: 0 < x) (hy_pos: 0 < y) (hz_pos: 0 < z) :
  inequality x y z :=
by
  sorry

end inequality_proof_l135_135009


namespace total_time_before_playing_game_l135_135962

theorem total_time_before_playing_game : 
  ∀ (d i t_t t : ℕ), 
  d = 10 → 
  i = d / 2 → 
  t_t = 3 * (d + i) → 
  t = d + i + t_t → 
  t = 60 := 
by
  intros d i t_t t h1 h2 h3 h4
  sorry

end total_time_before_playing_game_l135_135962


namespace eval_expression_l135_135089

theorem eval_expression : 3 ^ 4 - 4 * 3 ^ 3 + 6 * 3 ^ 2 - 4 * 3 + 1 = 16 := 
by 
  sorry

end eval_expression_l135_135089


namespace tickets_problem_l135_135592

theorem tickets_problem (A C : ℝ) 
  (h1 : A + C = 200) 
  (h2 : 3 * A + 1.5 * C = 510) : C = 60 :=
by
  sorry

end tickets_problem_l135_135592


namespace parabola_distance_l135_135230

def parabola_focus (y : ℝ) (x : ℝ) : Prop :=
  y^2 = 8 * x ∧ 
  ∃ (F : ℝ × ℝ), F = (2, 0)

def directrix (x : ℝ → ℝ) : Prop :=
  ∀ x, x = -2

def point_on_parabola (P : ℝ × ℝ) : Prop :=
  P.1^2 = 4 * P.2 

def slope_AF (A P : ℝ × ℝ) : Prop :=
  ∀ A P, A = (-2, 4) ∧ P = (2, 4) ∧ (P.2 - A.2) / (P.1 - A.1) = -1

def PF_distance (P F : ℝ × ℝ) : ℝ :=
  dist P F

theorem parabola_distance :
  ∀ P : ℝ × ℝ, ∀ F : ℝ × ℝ,
    parabola_focus P.1 P.2 → 
    directrix (λ x => x) → 
    point_on_parabola P → 
    slope_AF F P → 
    PF_distance P F = 4 := 
by 
  sorry

end parabola_distance_l135_135230


namespace reyn_pieces_l135_135976

-- Define the conditions
variables (total_pieces : ℕ) (pieces_each : ℕ) (pieces_left : ℕ)
variables (R : ℕ) (Rhys : ℕ) (Rory : ℕ)

-- Initial Conditions
def mrs_young_conditions :=
  total_pieces = 300 ∧
  pieces_each = total_pieces / 3 ∧
  Rhys = 2 * R ∧
  Rory = 3 * R ∧
  6 * R + pieces_left = total_pieces ∧
  pieces_left = 150

-- The statement of our proof goal
theorem reyn_pieces (h : mrs_young_conditions total_pieces pieces_each pieces_left R Rhys Rory) : R = 25 :=
sorry

end reyn_pieces_l135_135976


namespace perimeter_of_C_l135_135296

theorem perimeter_of_C (x y : ℝ) 
  (h₁ : 6 * x + 2 * y = 56) 
  (h₂ : 4 * x + 6 * y = 56) : 
  2 * x + 6 * y = 40 :=
sorry

end perimeter_of_C_l135_135296


namespace probability_check_on_friday_l135_135322

theorem probability_check_on_friday
    (P_no_check : ℝ) (P_check_per_day : ℝ) (P_check_friday : ℝ)
    (A : ℝ) (B : ℝ)
    (h1 : P_no_check = 1 / 2)
    (h2 : ∀ d, P_check_per_day = 1 / 2 * 1 / 5)
    (h3 : P_check_friday = 1 / 10)
    (h4 : A = P_no_check + P_check_friday)
    (h5 : A = 3 / 5)
    (h6 : A = 1)
    : (\frac{1 / 10} {3 / 5}) = 1 / 6 :=
by
    sorry

end probability_check_on_friday_l135_135322


namespace Claire_photos_l135_135683

variable (C : ℕ)

def Lisa_photos := 3 * C
def Robert_photos := C + 28

theorem Claire_photos :
  Lisa_photos C = Robert_photos C → C = 14 :=
by
  sorry

end Claire_photos_l135_135683


namespace expected_number_of_inspections_is_seven_halves_l135_135103

open ProbabilityTheory

noncomputable def expected_value_of_number_of_inspections
  (inspection_results : List Bool)
  (defective_count : ℕ) : ℚ :=
let total_items := inspection_results.length in
let inspections := (1:ℚ) + (1:ℚ) / 2 + (1:ℚ) / 2 * (defective_count:ℚ) * (total_items-defective_count:ℚ) in
if total_items = 5 ∧ defective_count = 2 then inspections else sorry

theorem expected_number_of_inspections_is_seven_halves :
  expected_value_of_number_of_inspections [true, true, false, false, false] 2 = 7 / 2 :=
sorry

end expected_number_of_inspections_is_seven_halves_l135_135103


namespace survey_preference_l135_135550

theorem survey_preference (X Y : ℕ) 
  (ratio_condition : X / Y = 5)
  (total_respondents : X + Y = 180) :
  X = 150 := 
sorry

end survey_preference_l135_135550


namespace heights_proportional_l135_135231

-- Define the problem conditions
def sides_ratio (a b c : ℕ) : Prop := a / b = 3 / 4 ∧ b / c = 4 / 5

-- Define the heights
def heights_ratio (h1 h2 h3 : ℕ) : Prop := h1 / h2 = 20 / 15 ∧ h2 / h3 = 15 / 12

-- Problem statement: Given the sides ratio, prove the heights ratio
theorem heights_proportional {a b c h1 h2 h3 : ℕ} (h : sides_ratio a b c) :
  heights_ratio h1 h2 h3 :=
sorry

end heights_proportional_l135_135231


namespace cars_sold_l135_135478

theorem cars_sold (sales_Mon sales_Tue sales_Wed cars_Thu_Fri_Sat : ℕ) 
  (mean : ℝ) (h1 : sales_Mon = 8) 
  (h2 : sales_Tue = 3) 
  (h3 : sales_Wed = 10) 
  (h4 : mean = 5.5) 
  (h5 : mean * 6 = sales_Mon + sales_Tue + sales_Wed + cars_Thu_Fri_Sat):
  cars_Thu_Fri_Sat = 12 :=
sorry

end cars_sold_l135_135478


namespace candle_height_comparison_l135_135569

def first_candle_height (t : ℝ) : ℝ := 10 - 2 * t
def second_candle_height (t : ℝ) : ℝ := 8 - 2 * t

theorem candle_height_comparison (t : ℝ) :
  first_candle_height t = 3 * second_candle_height t → t = 3.5 :=
by
  -- the main proof steps would be here
  sorry

end candle_height_comparison_l135_135569


namespace range_of_m_l135_135552

-- Definition of the propositions and conditions
def p (m : ℝ) : Prop := m > 2
def q (m : ℝ) : Prop := 1 ≤ m ∧ m ≤ 3
def prop (m : ℝ) : Prop := (¬(p m ∧ q m) ∧ (p m ∨ q m))

-- The proof statement showing the range of m
theorem range_of_m (m : ℝ) : prop m ↔ (1 ≤ m ∧ m ≤ 2) ∨ (m > 3) :=
by
  sorry

end range_of_m_l135_135552


namespace geese_problem_l135_135074

theorem geese_problem 
  (G : ℕ)  -- Total number of geese in the original V formation
  (T : ℕ)  -- Number of geese that flew up from the trees to join the new V formation
  (h1 : G / 2 + T = 12)  -- Final number of geese flying in the V formation was 12 
  (h2 : T = G / 2)  -- Number of geese that flew out from the trees is the same as the number of geese that landed initially
: T = 6 := 
sorry

end geese_problem_l135_135074


namespace initial_crayons_l135_135706

theorem initial_crayons {C : ℕ} (h : C + 12 = 53) : C = 41 :=
by
  -- This is where the proof would go, but we use sorry to skip it.
  sorry

end initial_crayons_l135_135706


namespace angle_terminal_side_equivalence_l135_135146

theorem angle_terminal_side_equivalence (k : ℤ) : 
    ∃ k : ℤ, 405 = k * 360 + 45 :=
by
  sorry

end angle_terminal_side_equivalence_l135_135146


namespace parts_per_hour_equality_l135_135139

variable {x : ℝ}

theorem parts_per_hour_equality (h1 : x - 4 > 0) :
  (100 / x) = (80 / (x - 4)) :=
sorry

end parts_per_hour_equality_l135_135139


namespace shopkeeper_total_profit_percentage_l135_135487

noncomputable def profit_percentage (actual_weight faulty_weight ratio : ℕ) : ℝ :=
  (actual_weight - faulty_weight) / actual_weight * 100 * ratio

noncomputable def total_profit_percentage (ratios profits : List ℝ) : ℝ :=
  (List.sum (List.zipWith (· * ·) ratios profits)) / (List.sum ratios)

theorem shopkeeper_total_profit_percentage :
  let actual_weight := 1000
  let faulty_weights := [900, 850, 950]
  let profit_percentages := [10, 15, 5]
  let ratios := [3, 2, 1]
  total_profit_percentage ratios profit_percentages = 10.83 :=
by
  sorry

end shopkeeper_total_profit_percentage_l135_135487


namespace second_discount_percentage_l135_135479

theorem second_discount_percentage 
    (original_price : ℝ) (final_price : ℝ) (first_discount : ℝ) (third_discount : ℝ) (second_discount : ℝ) :
      original_price = 9795.3216374269 →
      final_price = 6700 →
      first_discount = 0.20 →
      third_discount = 0.05 →
      (original_price * (1 - first_discount) * (1 - second_discount / 100) * (1 - third_discount) = final_price) →
      second_discount = 10 :=
by
  intros h_orig h_final h_first h_third h_eq
  sorry

end second_discount_percentage_l135_135479


namespace raft_travel_time_l135_135699

-- Define the problem conditions:
def steamboat_time (distance : ℕ) := 1 -- in hours
def motorboat_time (distance : ℕ) : ℚ := 3 / 4 -- in hours
def speed_ratio := 2 -- motorboat speed is twice the speed of steamboat

-- Define the time for the raft to travel the distance:
def raft_time (distance : ℕ) (current_speed : ℚ) := distance / current_speed

-- Given the conditions, prove that the raft time equals to 90 minutes
theorem raft_travel_time (distance : ℕ) (rafter_speed : ℚ) (current_speed : ℚ) :
  steamboat_time distance = 1 ∧ motorboat_time distance = 3 / 4 ∧ rafter_speed = current_speed →
  rafter_speed = current_speed ∧ raft_time distance current_speed = 3 / 2 → -- hours
  raft_time distance current_speed * 60 = 90 := -- convert hours to minutes
by
  intros h1 h2
  sorry

end raft_travel_time_l135_135699


namespace no_valid_n_l135_135501

theorem no_valid_n (n : ℕ) : (100 ≤ n / 4 ∧ n / 4 ≤ 999) → (100 ≤ 4 * n ∧ 4 * n ≤ 999) → false :=
by
  intro h1 h2
  sorry

end no_valid_n_l135_135501


namespace least_positive_integer_l135_135879

theorem least_positive_integer (n : ℕ) (h1 : n > 1)
  (h2 : n % 3 = 2) (h3 : n % 4 = 2) (h4 : n % 5 = 2) (h5 : n % 11 = 2) :
  n = 662 :=
sorry

end least_positive_integer_l135_135879


namespace arithmetic_seq_k_l135_135258

theorem arithmetic_seq_k (a : ℕ → ℝ) (S : ℕ → ℝ) (k : ℕ) 
  (h1 : a 1 = -3)
  (h2 : a (k + 1) = 3 / 2)
  (h3 : S k = -12)
  (h4 : ∀ n, S n = n * (a 1 + a (n+1)) / 2):
  k = 13 :=
sorry

end arithmetic_seq_k_l135_135258


namespace find_g5_l135_135838

def g : ℤ → ℤ := sorry

axiom g_cond1 : g 1 > 1
axiom g_cond2 : ∀ x y : ℤ, g (x + y) + x * g y + y * g x = g x * g y + x + y + x * y
axiom g_cond3 : ∀ x : ℤ, 3 * g x = g (x + 1) + 2 * x - 1

theorem find_g5 : g 5 = 248 :=
by
  sorry

end find_g5_l135_135838


namespace mrs_hilt_walks_240_feet_l135_135819

-- Define the distances and trips as given conditions
def distance_to_fountain : ℕ := 30
def trips_to_fountain : ℕ := 4
def round_trip_distance : ℕ := 2 * distance_to_fountain
def total_distance_walked (round_trip_distance trips_to_fountain : ℕ) : ℕ :=
  round_trip_distance * trips_to_fountain

-- State the theorem
theorem mrs_hilt_walks_240_feet :
  total_distance_walked round_trip_distance trips_to_fountain = 240 :=
by
  sorry

end mrs_hilt_walks_240_feet_l135_135819


namespace mean_of_quadrilateral_angles_l135_135407

theorem mean_of_quadrilateral_angles : 
  (∀ (a b c d : ℝ), a + b + c + d = 360) → 90 :=
by
  intro h
  sorry

end mean_of_quadrilateral_angles_l135_135407


namespace ratio_of_ages_l135_135733

theorem ratio_of_ages (S F : Nat) 
  (h1 : F = 3 * S) 
  (h2 : (S + 6) + (F + 6) = 156) : 
  (F + 6) / (S + 6) = 19 / 7 := 
by 
  sorry

end ratio_of_ages_l135_135733


namespace is_triangle_inequality_set_B_valid_triangle_set_A_not_triangle_set_C_not_triangle_set_D_not_triangle_l135_135730

theorem is_triangle_inequality (a b c: ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem set_B_valid_triangle :
  is_triangle_inequality 5 5 6 := by
  sorry

theorem set_A_not_triangle :
  ¬ is_triangle_inequality 7 4 2 := by
  sorry

theorem set_C_not_triangle :
  ¬ is_triangle_inequality 3 4 8 := by
  sorry

theorem set_D_not_triangle :
  ¬ is_triangle_inequality 2 3 5 := by
  sorry

end is_triangle_inequality_set_B_valid_triangle_set_A_not_triangle_set_C_not_triangle_set_D_not_triangle_l135_135730


namespace mean_of_quadrilateral_angles_is_90_l135_135456

theorem mean_of_quadrilateral_angles_is_90 :
  let n := 4 in
  let sum_of_interior_angles (n : ℕ) := (n - 2) * 180 in
  let quadrilateral_interior_angle_sum := sum_of_interior_angles 4 in
  quadrilateral_interior_angle_sum = 360 →
  360 / 4 = 90 :=
by
  intros n sum_of_interior_angles quadrilateral_interior_angle_sum h
  exact sorry

end mean_of_quadrilateral_angles_is_90_l135_135456


namespace find_first_discount_l135_135564

theorem find_first_discount (price_initial : ℝ) (price_final : ℝ) (discount_additional : ℝ) (x : ℝ) :
  price_initial = 350 → price_final = 266 → discount_additional = 5 →
  price_initial * (1 - x / 100) * (1 - discount_additional / 100) = price_final →
  x = 20 :=
by
  intros h1 h2 h3 h4
  -- skippable in proofs, just holds the place
  sorry

end find_first_discount_l135_135564


namespace ellipse_hyperbola_tangent_l135_135087

def ellipse (x y : ℝ) : Prop := x^2 + 9 * y^2 = 9
def hyperbola (x y m : ℝ) : Prop := x^2 - m * (y + 1)^2 = 1

theorem ellipse_hyperbola_tangent (m : ℝ) :
  (∀ x y : ℝ, ellipse x y → hyperbola x y m) → m = 2 :=
by sorry

end ellipse_hyperbola_tangent_l135_135087


namespace line_through_center_eq_line_chord_len_eq_l135_135936

noncomputable def circle_eq (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 9

noncomputable def point_P : ℝ × ℝ := (2, 2)

def line_through_center (x y : ℝ) : Prop := 2 * x - y - 2 = 0

def line_chord_len (x y : ℝ) : Prop := 3 * x - 4 * y + 2 = 0 ∨ x = 2

theorem line_through_center_eq (x y : ℝ) (hC : circle_eq x y) :
  line_through_center x y :=
sorry

theorem line_chord_len_eq (x y : ℝ) (hC : circle_eq x y) (hP : x = 2 ∧ y = 2 ∧ (line_through_center x y)) :
  line_chord_len x y :=
sorry

end line_through_center_eq_line_chord_len_eq_l135_135936


namespace sum_A_B_l135_135008

noncomputable def num_four_digit_odd_numbers_divisible_by_3 : ℕ := 1500
noncomputable def num_four_digit_multiples_of_7 : ℕ := 1286

theorem sum_A_B (A B : ℕ) :
  A = num_four_digit_odd_numbers_divisible_by_3 →
  B = num_four_digit_multiples_of_7 →
  A + B = 2786 :=
by
  intros hA hB
  rw [hA, hB]
  exact rfl

end sum_A_B_l135_135008


namespace find_x_coordinate_l135_135672

-- Define the center and radius of the circle
structure Circle where
  center : (ℝ × ℝ)
  radius : ℝ

-- Define the points on the circle
def lies_on_circle (C : Circle) (P : ℝ × ℝ) : Prop :=
  let (x_c, y_c) := C.center
  let (x_p, y_p) := P
  (x_p - x_c)^2 + (y_p - y_c)^2 = C.radius^2

-- Lean 4 statement
theorem find_x_coordinate :
  ∀ (C : Circle), C.radius = 2 → lies_on_circle C (2, 0) ∧ lies_on_circle C (-2, 0) → 2 = 2 := by
  intro C h_radius ⟨h_lies_on_2_0, h_lies_on__2_0⟩
  sorry

end find_x_coordinate_l135_135672


namespace perimeter_C_l135_135288

theorem perimeter_C (x y : ℕ) 
  (h1 : 6 * x + 2 * y = 56)
  (h2 : 4 * x + 6 * y = 56) :
  2 * x + 6 * y = 40 := 
by
  sorry

end perimeter_C_l135_135288


namespace mean_value_of_interior_angles_of_quadrilateral_l135_135400

theorem mean_value_of_interior_angles_of_quadrilateral 
  (sum_of_interior_angles : ℝ := 360) 
  (number_of_angles : ℝ := 4) : 
  sum_of_interior_angles / number_of_angles = 90 := 
by 
  sorry

end mean_value_of_interior_angles_of_quadrilateral_l135_135400


namespace remainder_of_sum_of_consecutive_days_l135_135768

theorem remainder_of_sum_of_consecutive_days :
  (100045 + 100046 + 100047 + 100048 + 100049 + 100050 + 100051 + 100052) % 5 = 3 :=
by
  sorry

end remainder_of_sum_of_consecutive_days_l135_135768


namespace mean_value_interior_angles_quadrilateral_l135_135371

theorem mean_value_interior_angles_quadrilateral :
  let n := 4 in
  (n - 2) * 180 / n = 90 :=
by
  let n := 4
  sorry

end mean_value_interior_angles_quadrilateral_l135_135371


namespace total_area_of_squares_l135_135041

theorem total_area_of_squares (x : ℝ) (hx : 4 * x^2 = 240) : 
  let small_square_area := x^2
  let large_square_area := (2 * x)^2
  2 * small_square_area + large_square_area = 360 :=
by
  let small_square_area := x^2
  let large_square_area := (2 * x)^2
  sorry

end total_area_of_squares_l135_135041


namespace roots_sum_product_l135_135127

variable {a b : ℝ}

theorem roots_sum_product (ha : a + b = 6) (hp : a * b = 8) : 
  a^4 + b^4 + a^3 * b + a * b^3 = 432 :=
by
  sorry

end roots_sum_product_l135_135127


namespace residents_ticket_price_l135_135062

theorem residents_ticket_price
  (total_attendees : ℕ)
  (resident_count : ℕ)
  (non_resident_price : ℝ)
  (total_revenue : ℝ)
  (R : ℝ)
  (h1 : total_attendees = 586)
  (h2 : resident_count = 219)
  (h3 : non_resident_price = 17.95)
  (h4 : total_revenue = 9423.70)
  (total_residents_pay : ℝ := resident_count * R)
  (total_non_residents_pay : ℝ := (total_attendees - resident_count) * non_resident_price)
  (h5 : total_revenue = total_residents_pay + total_non_residents_pay) :
  R = 12.95 := by
  sorry

end residents_ticket_price_l135_135062


namespace projection_area_rectangular_board_l135_135077

noncomputable def projection_area (AB BC NE MN : ℝ) (ABCD_perp_ground : Prop) (E_mid_AB : Prop) (light_at_M : Prop) : ℝ :=
  let width := AB
  let height := BC
  let shadow_width := 5
  (1 / 2) * (width + shadow_width) * height

theorem projection_area_rectangular_board (AB BC NE MN : ℝ) (ABCD_perp_ground : Prop) (E_mid_AB : Prop) (light_at_M : Prop) :
  AB = 3 → BC = 2 → NE = 3 → MN = 5 → projection_area AB BC NE MN ABCD_perp_ground E_mid_AB light_at_M = 8 :=
by
  intros
  sorry

end projection_area_rectangular_board_l135_135077


namespace mean_value_of_interior_angles_of_quadrilateral_l135_135399

theorem mean_value_of_interior_angles_of_quadrilateral 
  (sum_of_interior_angles : ℝ := 360) 
  (number_of_angles : ℝ := 4) : 
  sum_of_interior_angles / number_of_angles = 90 := 
by 
  sorry

end mean_value_of_interior_angles_of_quadrilateral_l135_135399


namespace number_of_female_students_l135_135996

theorem number_of_female_students 
  (average_all : ℝ)
  (num_males : ℝ) 
  (average_males : ℝ)
  (average_females : ℝ) 
  (h_avg_all : average_all = 88)
  (h_num_males : num_males = 15)
  (h_avg_males : average_males = 80)
  (h_avg_females : average_females = 94) :
  ∃ F : ℝ, 1200 + 94 * F = 88 * (15 + F) ∧ F = 20 :=
by
  use 20
  sorry

end number_of_female_students_l135_135996


namespace greatest_odd_factors_under_150_l135_135018

theorem greatest_odd_factors_under_150 : ∃ (n : ℕ), n < 150 ∧ ( ∃ (k : ℕ), n = k * k ) ∧ (∀ m : ℕ, m < 150 ∧ ( ∃ (k : ℕ), m = k * k ) → m ≤ 144) :=
by
  sorry

end greatest_odd_factors_under_150_l135_135018


namespace ratio_of_almonds_to_walnuts_l135_135064

theorem ratio_of_almonds_to_walnuts
  (A W : ℝ)
  (weight_almonds : ℝ)
  (total_weight : ℝ)
  (weight_walnuts : ℝ)
  (ratio : 2 * W = total_weight - weight_almonds)
  (given_almonds : weight_almonds = 107.14285714285714)
  (given_total_weight : total_weight = 150)
  (computed_weight_walnuts : weight_walnuts = 42.85714285714286)
  (proportion : A / (2 * W) = weight_almonds / weight_walnuts) :
  A / W = 5 :=
by
  sorry

end ratio_of_almonds_to_walnuts_l135_135064


namespace flowers_per_bouquet_l135_135068

theorem flowers_per_bouquet (narcissus chrysanthemums bouquets : ℕ) 
  (h1: narcissus = 75) 
  (h2: chrysanthemums = 90) 
  (h3: bouquets = 33) 
  : (narcissus + chrysanthemums) / bouquets = 5 := 
by 
  sorry

end flowers_per_bouquet_l135_135068


namespace solve_for_x_l135_135691

theorem solve_for_x (x : ℚ) (h : (x + 4) / (x - 3) = (x - 2) / (x + 2)) : x = -2 / 11 :=
by
  sorry

end solve_for_x_l135_135691


namespace parts_per_hour_equality_l135_135140

variable {x : ℝ}

theorem parts_per_hour_equality (h1 : x - 4 > 0) :
  (100 / x) = (80 / (x - 4)) :=
sorry

end parts_per_hour_equality_l135_135140


namespace max_profit_30000_l135_135067

noncomputable def max_profit (type_A : ℕ) (type_B : ℕ) : ℝ := 
  10000 * type_A + 5000 * type_B

theorem max_profit_30000 :
  ∃ (type_A type_B : ℕ), 
  (4 * type_A + 1 * type_B ≤ 10) ∧
  (18 * type_A + 15 * type_B ≤ 66) ∧
  max_profit type_A type_B = 30000 :=
sorry

end max_profit_30000_l135_135067


namespace intersecting_lines_l135_135035

theorem intersecting_lines (x y : ℝ) : x ^ 2 - y ^ 2 = 0 ↔ (y = x ∨ y = -x) := by
  sorry

end intersecting_lines_l135_135035


namespace mean_value_of_quadrilateral_angles_l135_135428

theorem mean_value_of_quadrilateral_angles : 
  ∀ (a b c d : ℝ), a + b + c + d = 360 → (a + b + c + d) / 4 = 90 :=
by
  intro a b c d h
  sorry

end mean_value_of_quadrilateral_angles_l135_135428


namespace mean_value_of_quadrilateral_angles_l135_135429

theorem mean_value_of_quadrilateral_angles : 
  ∀ (a b c d : ℝ), a + b + c + d = 360 → (a + b + c + d) / 4 = 90 :=
by
  intro a b c d h
  sorry

end mean_value_of_quadrilateral_angles_l135_135429


namespace weight_distribution_l135_135568

theorem weight_distribution (x y z : ℕ) 
  (h1 : x + y + z = 100) 
  (h2 : x + 10 * y + 50 * z = 500) : 
  x = 60 ∧ y = 39 ∧ z = 1 :=
by {
  sorry
}

end weight_distribution_l135_135568


namespace rowing_distance_l135_135737

def man_rowing_speed_still_water : ℝ := 10
def stream_speed : ℝ := 8
def rowing_time_downstream : ℝ := 5
def effective_speed_downstream : ℝ := man_rowing_speed_still_water + stream_speed

theorem rowing_distance :
  effective_speed_downstream * rowing_time_downstream = 90 := 
by 
  sorry

end rowing_distance_l135_135737


namespace levels_for_blocks_l135_135805

theorem levels_for_blocks (S : ℕ → ℕ) (n : ℕ) (h1 : S n = n * (n + 1)) (h2 : S 10 = 110) : n = 10 :=
by {
  sorry
}

end levels_for_blocks_l135_135805


namespace solve_fractional_eq_l135_135271

theorem solve_fractional_eq (x: ℝ) (h1: x ≠ -11) (h2: x ≠ -8) (h3: x ≠ -12) (h4: x ≠ -7) :
  (1 / (x + 11) + 1 / (x + 8) = 1 / (x + 12) + 1 / (x + 7)) → (x = -19 / 2) :=
by
  sorry

end solve_fractional_eq_l135_135271


namespace isosceles_triangle_base_angle_l135_135956

theorem isosceles_triangle_base_angle
    (X : ℝ)
    (h1 : 0 < X)
    (h2 : 2 * X + X + X = 180)
    (h3 : X + X + 2 * X = 180) :
    X = 45 ∨ X = 72 :=
by sorry

end isosceles_triangle_base_angle_l135_135956


namespace value_of_m_l135_135239

theorem value_of_m (m : ℝ) :
  (∃ x : ℝ, x > 0 ∧ (3 / (x - 2) + (x + m) / (2 - x) = 1)) → m = 1 :=
by
  sorry

end value_of_m_l135_135239


namespace closest_point_on_line_l135_135766

theorem closest_point_on_line :
  ∀ (x y : ℝ), (4, -2) = (4, -2) →
    y = 3 * x - 1 →
    (∃ (p : ℝ × ℝ), p = (-0.5, -2.5) ∧ p = (-0.5, -2.5))
  := by
    -- The proof of the theorem goes here
    sorry

end closest_point_on_line_l135_135766


namespace g_5_is_248_l135_135845

def g : ℤ → ℤ := sorry

theorem g_5_is_248 :
  (g(1) > 1) ∧
  (∀ x y : ℤ, g(x + y) + x * g(y) + y * g(x) = g(x) * g(y) + x + y + x * y) ∧
  (∀ x : ℤ, 3 * g(x) = g(x + 1) + 2 * x - 1) →
  g(5) = 248 :=
by
  -- proof omitted
  sorry

end g_5_is_248_l135_135845


namespace mean_of_quadrilateral_angles_is_90_l135_135455

theorem mean_of_quadrilateral_angles_is_90 :
  let n := 4 in
  let sum_of_interior_angles (n : ℕ) := (n - 2) * 180 in
  let quadrilateral_interior_angle_sum := sum_of_interior_angles 4 in
  quadrilateral_interior_angle_sum = 360 →
  360 / 4 = 90 :=
by
  intros n sum_of_interior_angles quadrilateral_interior_angle_sum h
  exact sorry

end mean_of_quadrilateral_angles_is_90_l135_135455


namespace total_flour_used_l135_135050

theorem total_flour_used :
  let wheat_flour := 0.2
  let white_flour := 0.1
  let rye_flour := 0.15
  let almond_flour := 0.05
  let oat_flour := 0.1
  wheat_flour + white_flour + rye_flour + almond_flour + oat_flour = 0.6 :=
by
  sorry

end total_flour_used_l135_135050


namespace min_reciprocal_sum_l135_135252

theorem min_reciprocal_sum (a b x y : ℝ) (h1 : 8 * x - y - 4 ≤ 0) (h2 : x + y + 1 ≥ 0) (h3 : y - 4 * x ≤ 0) 
    (ha : a > 0) (hb : b > 0) (hz : a * x + b * y = 2) : 
    1 / a + 1 / b = 9 / 2 := 
    sorry

end min_reciprocal_sum_l135_135252


namespace sophia_lost_pawns_l135_135991

theorem sophia_lost_pawns
    (total_pawns : ℕ := 16)
    (start_pawns_each : ℕ := 8)
    (chloe_lost : ℕ := 1)
    (pawns_left : ℕ := 10)
    (chloe_pawns_left : ℕ := start_pawns_each - chloe_lost) :
    total_pawns = 2 * start_pawns_each → 
    ∃ (sophia_lost : ℕ), sophia_lost = start_pawns_each - (pawns_left - chloe_pawns_left) :=
by 
    intros _ 
    use 5 
    sorry

end sophia_lost_pawns_l135_135991


namespace perimeter_C_is_40_l135_135310

noncomputable def perimeter_of_figure_C (x y : ℝ) : ℝ :=
  2 * x + 6 * y

theorem perimeter_C_is_40 (x y : ℝ) (h1 : 6 * x + 2 * y = 56) (h2 : 4 * x + 6 * y = 56) :
  perimeter_of_figure_C x y = 40 :=
by
  -- Define initial conditions
  have eq1 : 3 * x + y = 28, by { rw [mul_assoc, mul_comm 3 x, add_assoc], exact (eq.div h1 2) }
  have eq2 : 2 * x + 3 * y = 28, by { rw [mul_assoc, mul_comm 2 x, add_assoc], exact (eq.div h2 2) }
  -- Assume the solutions are obtained from here
  have sol_x : x = 8, by sorry
  have sol_y : y = 4, by sorry
  -- Calculate the perimeter of figure C
  rw [perimeter_of_figure_C, sol_x, sol_y]
  norm_num
  trivial

-- Test case to ensure the code builds successfully
#eval perimeter_of_figure_C 8 4  -- Expected output: 40

end perimeter_C_is_40_l135_135310


namespace find_g5_l135_135853

def g : ℤ → ℤ := sorry

axiom g1 : g 1 > 1
axiom g2 : ∀ x y : ℤ, g (x + y) + x * g y + y * g x = g x * g y + x + y + x * y
axiom g3 : ∀ x : ℤ, 3 * g x = g (x + 1) + 2 * x - 1

theorem find_g5 : g 5 = 248 :=
sorry

end find_g5_l135_135853


namespace exists_xyz_l135_135970

noncomputable def sum_of_digits (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.sum

theorem exists_xyz :
  ∃ x y z : ℕ, x > 0 ∧ y > 0 ∧ z > 0 ∧
  (x + sum_of_digits x = y + sum_of_digits y ∧ y + sum_of_digits y = z + sum_of_digits z) :=
by {
  sorry
}

end exists_xyz_l135_135970


namespace perimeter_C_is_40_l135_135309

noncomputable def perimeter_of_figure_C (x y : ℝ) : ℝ :=
  2 * x + 6 * y

theorem perimeter_C_is_40 (x y : ℝ) (h1 : 6 * x + 2 * y = 56) (h2 : 4 * x + 6 * y = 56) :
  perimeter_of_figure_C x y = 40 :=
by
  -- Define initial conditions
  have eq1 : 3 * x + y = 28, by { rw [mul_assoc, mul_comm 3 x, add_assoc], exact (eq.div h1 2) }
  have eq2 : 2 * x + 3 * y = 28, by { rw [mul_assoc, mul_comm 2 x, add_assoc], exact (eq.div h2 2) }
  -- Assume the solutions are obtained from here
  have sol_x : x = 8, by sorry
  have sol_y : y = 4, by sorry
  -- Calculate the perimeter of figure C
  rw [perimeter_of_figure_C, sol_x, sol_y]
  norm_num
  trivial

-- Test case to ensure the code builds successfully
#eval perimeter_of_figure_C 8 4  -- Expected output: 40

end perimeter_C_is_40_l135_135309


namespace last_digit_of_3_pow_2004_l135_135821

theorem last_digit_of_3_pow_2004 : (3 ^ 2004) % 10 = 1 := by
  sorry

end last_digit_of_3_pow_2004_l135_135821


namespace mean_value_of_quadrilateral_angles_l135_135328

-- Statement of the problem: mean value of interior angles in any quadrilateral is 90°
theorem mean_value_of_quadrilateral_angles : 
  (∀ (a b c d : ℝ), a + b + c + d = 360 → (a + b + c + d) / 4 = 90) :=
by
  sorry

end mean_value_of_quadrilateral_angles_l135_135328


namespace find_a_range_l135_135512

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then -x^2 - a*x - 5 else (a + 1) / x

theorem find_a_range (a : ℝ) :
  (∀ x y : ℝ, x ≤ y → f a x ≤ f a y) → 
  - (7 / 2) ≤ a ∧ a ≤ -2 :=
by
  sorry

end find_a_range_l135_135512


namespace taxi_fare_l135_135174

theorem taxi_fare :
  ∀ (initial_fee rate_per_increment increment_distance total_distance : ℝ),
    initial_fee = 2.35 →
    rate_per_increment = 0.35 →
    increment_distance = (2 / 5) →
    total_distance = 3.6 →
    (initial_fee + rate_per_increment * (total_distance / increment_distance)) = 5.50 :=
by
  intros initial_fee rate_per_increment increment_distance total_distance
  intro h1 h2 h3 h4
  sorry -- Proof is not required.

end taxi_fare_l135_135174


namespace software_price_l135_135065

theorem software_price (copies total_revenue : ℝ) (P : ℝ) 
  (h1 : copies = 1200)
  (h2 : 0.5 * copies * P + 0.6 * (2 / 3) * (copies - 0.5 * copies) * P + 0.25 * (copies - 0.5 * copies - (2 / 3) * (copies - 0.5 * copies)) * P = total_revenue)
  (h3 : total_revenue = 72000) :
  P = 80.90 :=
by
  sorry

end software_price_l135_135065


namespace mean_value_of_quadrilateral_angles_l135_135421

theorem mean_value_of_quadrilateral_angles (angles : Fin 4 → ℝ) 
  (h_sum : ∑ i, angles i = 360) : ∑ i, angles i / 4 = 90 :=
by
  sorry

end mean_value_of_quadrilateral_angles_l135_135421


namespace traveler_meets_truck_at_15_48_l135_135609

noncomputable def timeTravelerMeetsTruck : ℝ := 15 + 48 / 60

theorem traveler_meets_truck_at_15_48 {S Vp Vm Vg : ℝ}
  (h_travel_covered : Vp = S / 4)
  (h_motorcyclist_catch : 1 = (S / 4) / (Vm - Vp))
  (h_motorcyclist_meet_truck : 1.5 = S / (Vm + Vg)) :
  (S / 4 + (12 / 5) * (Vg + Vp)) / (12 / 5) = timeTravelerMeetsTruck := sorry

end traveler_meets_truck_at_15_48_l135_135609


namespace mean_value_of_quadrilateral_angles_l135_135443

theorem mean_value_of_quadrilateral_angles :
  let sum_of_angles := 360 in
  let number_of_angles := 4 in
  sum_of_angles / number_of_angles = 90 :=
by
  sorry

end mean_value_of_quadrilateral_angles_l135_135443


namespace jim_loan_inequality_l135_135000

noncomputable def A (t : ℕ) : ℝ := 1500 * (1.06 ^ t)

theorem jim_loan_inequality : ∃ t : ℕ, A t > 3000 ∧ ∀ t' : ℕ, t' < t → A t' ≤ 3000 :=
by
  sorry

end jim_loan_inequality_l135_135000


namespace mean_value_of_quadrilateral_angles_l135_135324

-- Statement of the problem: mean value of interior angles in any quadrilateral is 90°
theorem mean_value_of_quadrilateral_angles : 
  (∀ (a b c d : ℝ), a + b + c + d = 360 → (a + b + c + d) / 4 = 90) :=
by
  sorry

end mean_value_of_quadrilateral_angles_l135_135324


namespace not_perpendicular_to_vA_not_perpendicular_to_vB_not_perpendicular_to_vD_l135_135745

def vector_a : ℝ × ℝ := (3, 2)
def vector_vA : ℝ × ℝ := (3, -2)
def vector_vB : ℝ × ℝ := (2, 3)
def vector_vD : ℝ × ℝ := (-3, 2)

def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

theorem not_perpendicular_to_vA : dot_product vector_a vector_vA ≠ 0 := by sorry
theorem not_perpendicular_to_vB : dot_product vector_a vector_vB ≠ 0 := by sorry
theorem not_perpendicular_to_vD : dot_product vector_a vector_vD ≠ 0 := by sorry

end not_perpendicular_to_vA_not_perpendicular_to_vB_not_perpendicular_to_vD_l135_135745


namespace min_value_of_sum_of_squares_l135_135811

theorem min_value_of_sum_of_squares (a b c : ℕ) (h₀ : a ≠ 0) (h₁ : b ≠ 0) (h₂ : c ≠ 0) 
  (h : a^2 + b^2 - c = 2022) : 
  a^2 + b^2 + c^2 = 2034 ∧ a = 27 ∧ b = 36 ∧ c = 3 := 
sorry

end min_value_of_sum_of_squares_l135_135811


namespace mean_value_of_quadrilateral_angles_l135_135431

theorem mean_value_of_quadrilateral_angles : 
  ∀ (a b c d : ℝ), a + b + c + d = 360 → (a + b + c + d) / 4 = 90 :=
by
  intro a b c d h
  sorry

end mean_value_of_quadrilateral_angles_l135_135431


namespace total_time_before_playing_game_l135_135961

theorem total_time_before_playing_game : 
  ∀ (d i t_t t : ℕ), 
  d = 10 → 
  i = d / 2 → 
  t_t = 3 * (d + i) → 
  t = d + i + t_t → 
  t = 60 := 
by
  intros d i t_t t h1 h2 h3 h4
  sorry

end total_time_before_playing_game_l135_135961


namespace symmetric_circle_eq_l135_135764

theorem symmetric_circle_eq :
  (∃ x y : ℝ, (x + 2)^2 + (y - 1)^2 = 4) :=
by
  sorry

end symmetric_circle_eq_l135_135764


namespace calc_value_l135_135808

def f (x : ℤ) : ℤ := x^2 + 5 * x + 4
def g (x : ℤ) : ℤ := 2 * x - 3

theorem calc_value :
  f (g (-3)) - 2 * g (f 2) = -26 := by
  sorry

end calc_value_l135_135808


namespace compute_xy_l135_135044

theorem compute_xy (x y : ℝ) (h1 : x - y = 6) (h2 : x^3 - y^3 = 54) : x * y = -9 := 
by 
  sorry

end compute_xy_l135_135044


namespace mean_value_of_quadrilateral_angles_l135_135427

theorem mean_value_of_quadrilateral_angles : 
  ∀ (a b c d : ℝ), a + b + c + d = 360 → (a + b + c + d) / 4 = 90 :=
by
  intro a b c d h
  sorry

end mean_value_of_quadrilateral_angles_l135_135427


namespace container_alcohol_amount_l135_135066

theorem container_alcohol_amount
  (A : ℚ) -- Amount of alcohol in quarts
  (initial_water : ℚ) -- Initial amount of water in quarts
  (added_water : ℚ) -- Amount of water added in quarts
  (final_ratio_alcohol_to_water : ℚ) -- Final ratio of alcohol to water
  (h_initial_water : initial_water = 4) -- Container initially contains 4 quarts of water.
  (h_added_water : added_water = 8/3) -- 2.666666666666667 quarts of water added.
  (h_final_ratio : final_ratio_alcohol_to_water = 3/5) -- Final ratio is 3 parts alcohol to 5 parts water.
  (h_final_water : initial_water + added_water = 20/3) -- Total final water quarts after addition.
  : A = 4 := 
sorry

end container_alcohol_amount_l135_135066


namespace range_of_m_if_forall_x_gt_0_l135_135661

open Real

theorem range_of_m_if_forall_x_gt_0 (m : ℝ) :
  (∀ x : ℝ, 0 < x → x + 1/x - m > 0) ↔ m < 2 :=
by
  -- Placeholder proof
  sorry

end range_of_m_if_forall_x_gt_0_l135_135661


namespace sum_digits_B_of_4444_4444_l135_135681

noncomputable def sum_digits (n : ℕ) : ℕ :=
  n.digits 10 |> List.sum

theorem sum_digits_B_of_4444_4444 :
  let A : ℕ := sum_digits (4444 ^ 4444)
  let B : ℕ := sum_digits A
  sum_digits B = 7 :=
by
  sorry

end sum_digits_B_of_4444_4444_l135_135681


namespace soccer_lineup_count_l135_135824

theorem soccer_lineup_count :
  let total_players : ℕ := 16
  let total_starters : ℕ := 7
  let m_j_players : ℕ := 2 -- Michael and John
  let other_players := total_players - m_j_players
  let total_ways : ℕ :=
    2 * Nat.choose other_players (total_starters - 1) + Nat.choose other_players (total_starters - 2)
  total_ways = 8008
:= sorry

end soccer_lineup_count_l135_135824


namespace mean_of_quadrilateral_angles_l135_135383

theorem mean_of_quadrilateral_angles : 
  (∀ (a b c d : ℝ), a + b + c + d = 360 → (a + b + c + d) / 4 = 90) :=
begin
  intros a b c d h,
  calc
    (a + b + c + d) / 4 = 360 / 4 : by rw h
                    ... = 90 : by norm_num,
end

end mean_of_quadrilateral_angles_l135_135383


namespace find_g5_l135_135854

-- Define the function g on integers
def g : ℤ → ℤ := sorry

-- Define the conditions
axiom cond1 : g 1 > 1
axiom cond2 : ∀ (x y : ℤ), g (x + y) + x * g y + y * g x = g x * g y + x + y + x * y
axiom cond3 : ∀ (x : ℤ), 3 * g x = g (x + 1) + 2 * x - 1

-- The statement we need to prove
theorem find_g5 : g 5 = 248 := by
  sorry

end find_g5_l135_135854


namespace area_of_portion_of_circle_l135_135163

open Real

def circle_eq (x y : ℝ) : Prop := x^2 - 16 * x + y^2 - 8 * y = 32
def line_eq (x y : ℝ) : Prop := y = 2 * x - 20

theorem area_of_portion_of_circle :
  let r := 4 * Real.sqrt 5,
      total_area := pi * r^2,
      area_of_interest := total_area / 4
  in circle_eq x y → line_eq x y → x < (y + 20) / 2 → y < 0 → area_of_interest = 20 * pi :=
sorry

end area_of_portion_of_circle_l135_135163


namespace Joey_age_is_six_l135_135804

theorem Joey_age_is_six (ages: Finset ℕ) (a1 a2 a3 a4 : ℕ) (h1: ages = {4, 6, 8, 10})
  (h2: a1 + a2 = 14 ∨ a2 + a3 = 14 ∨ a3 + a4 = 14) (h3: a1 > 7 ∨ a2 > 7 ∨ a3 > 7 ∨ a4 > 7)
  (h4: (6 ∈ ages ∧ a1 ∈ ages) ∨ (6 ∈ ages ∧ a2 ∈ ages) ∨ 
      (6 ∈ ages ∧ a3 ∈ ages) ∨ (6 ∈ ages ∧ a4 ∈ ages)): 
  (a1 = 6 ∨ a2 = 6 ∨ a3 = 6 ∨ a4 = 6) :=
by
  sorry

end Joey_age_is_six_l135_135804


namespace mean_value_of_quadrilateral_interior_angles_l135_135464

theorem mean_value_of_quadrilateral_interior_angles (a b c d : ℝ) 
  (h_sum : a + b + c + d = 360) : 
  (a + b + c + d) / 4 = 90 :=
by
  sorry

end mean_value_of_quadrilateral_interior_angles_l135_135464


namespace ratio_of_boys_to_girls_l135_135666

/-- 
  Given 200 girls and a total of 600 students in a college,
  the ratio of the number of boys to the number of girls is 2:1.
--/
theorem ratio_of_boys_to_girls 
  (num_girls : ℕ) (total_students : ℕ) (h_girls : num_girls = 200) 
  (h_total : total_students = 600) : 
  (total_students - num_girls) / num_girls = 2 :=
by
  sorry

end ratio_of_boys_to_girls_l135_135666


namespace find_S_l135_135939

noncomputable def A := { x : ℝ | x^2 - 7 * x + 10 ≤ 0 }
noncomputable def B (a b : ℝ) := { x : ℝ | x^2 + a * x + b < 0 }
def A_inter_B_is_empty (a b : ℝ) := A ∩ B a b = ∅
def A_union_B_condition := { x : ℝ | x - 3 < 4 ∧ 4 ≤ 2 * x }

theorem find_S :
  A ∪ B (-12) 35 = { x : ℝ | 2 ≤ x ∧ x < 7 } →
  A ∩ B (-12) 35 = ∅ →
  { x : ℝ | x = -12 + 35 } = { 23 } :=
by
  intro h1 h2
  sorry

end find_S_l135_135939


namespace ribbon_left_l135_135266

theorem ribbon_left (gifts : ℕ) (ribbon_per_gift Tom_ribbon_total : ℝ) (h1 : gifts = 8) (h2 : ribbon_per_gift = 1.5) (h3 : Tom_ribbon_total = 15) : Tom_ribbon_total - (gifts * ribbon_per_gift) = 3 := 
by
  sorry

end ribbon_left_l135_135266


namespace arc_length_of_curve_l135_135887

noncomputable def arc_length : ℝ :=
∫ t in (0 : ℝ)..(Real.pi / 3),
  (Real.sqrt ((t^2 * Real.cos t)^2 + (t^2 * Real.sin t)^2))

theorem arc_length_of_curve :
  arc_length = (Real.pi^3 / 81) :=
by
  sorry

end arc_length_of_curve_l135_135887


namespace smallest_nth_root_of_unity_l135_135716

theorem smallest_nth_root_of_unity (n : ℕ) (h_pos : n > 0) :
  (∀ z : ℂ, z^6 - z^3 + 1 = 0 → ∃ k : ℕ, z = exp ((2 * ↑k * π * I) / ↑n)) ↔ n = 9 :=
by
  sorry

end smallest_nth_root_of_unity_l135_135716


namespace triangle_is_isosceles_l135_135241

open Real

-- Define the basic setup of the triangle and the variables involved
variables {A B C : ℝ} -- Angles of the triangle
variables {a b c : ℝ} -- Sides opposite to A, B, and C respectively
variables (h1 : a * cos B = b * cos A) -- Given condition: a * cos B = b * cos A

-- The theorem stating that the given condition implies the triangle is isosceles
theorem triangle_is_isosceles (h1 : a * cos B = b * cos A) : A = B :=
sorry

end triangle_is_isosceles_l135_135241


namespace smallest_sum_a_b_l135_135223

theorem smallest_sum_a_b (a b: ℕ) (h₀: 0 < a) (h₁: 0 < b) (h₂: a ≠ b) (h₃: 1 / (a: ℝ) + 1 / (b: ℝ) = 1 / 15) : a + b = 64 :=
sorry

end smallest_sum_a_b_l135_135223


namespace intersection_eq_l135_135011

def M (x : ℝ) : Prop := (x + 3) * (x - 2) < 0

def N (x : ℝ) : Prop := 1 ≤ x ∧ x ≤ 3

def intersection (x : ℝ) : Prop := M x ∧ N x

theorem intersection_eq : ∀ x, intersection x ↔ (1 ≤ x ∧ x < 2) :=
by sorry

end intersection_eq_l135_135011


namespace mean_value_of_quadrilateral_angles_l135_135418

theorem mean_value_of_quadrilateral_angles (angles : Fin 4 → ℝ) 
  (h_sum : ∑ i, angles i = 360) : ∑ i, angles i / 4 = 90 :=
by
  sorry

end mean_value_of_quadrilateral_angles_l135_135418


namespace count_valid_arrays_l135_135502

-- Define the integer array condition
def valid_array (x1 x2 x3 x4 : ℕ) : Prop :=
  0 < x1 ∧ x1 ≤ x2 ∧ x2 < x3 ∧ x3 ≤ x4 ∧ x4 < 7

-- State the theorem that proves the number of valid arrays is 70
theorem count_valid_arrays : ∃ (n : ℕ), n = 70 ∧ 
    ∀ (x1 x2 x3 x4 : ℕ), valid_array x1 x2 x3 x4 -> ∃ (n : ℕ), n = 70 :=
by
  -- The proof can be filled in later
  sorry

end count_valid_arrays_l135_135502


namespace compute_fraction_pow_mult_l135_135914

def frac_1_3 := (1 : ℝ) / (3 : ℝ)
def frac_1_5 := (1 : ℝ) / (5 : ℝ)
def target := (1 : ℝ) / (405 : ℝ)

theorem compute_fraction_pow_mult :
  (frac_1_3^4 * frac_1_5) = target :=
by
  sorry

end compute_fraction_pow_mult_l135_135914


namespace smallest_n_for_poly_l135_135723

-- Defining the polynomial equation
def poly_eq_zero (z : ℂ) : Prop := z^6 - z^3 + 1 = 0

-- Definition of n-th roots of unity
def nth_roots_of_unity (n : ℕ) : set ℂ := {z : ℂ | z^n = 1}

-- Definition to check if all roots are n-th roots of unity
def all_roots_are_nth_roots_of_unity (n : ℕ) : Prop :=
  ∀ z : ℂ, poly_eq_zero z → z ∈ nth_roots_of_unity n

-- The theorem statement
theorem smallest_n_for_poly : ∃ n : ℕ, all_roots_are_nth_roots_of_unity n ∧ ∀ m : ℕ, all_roots_are_nth_roots_of_unity m → n ≤ m :=
begin
  sorry
end

end smallest_n_for_poly_l135_135723


namespace num_pos_int_values_l135_135219

theorem num_pos_int_values
  (N : ℕ) 
  (h₀ : 0 < N)
  (h₁ : ∃ (k : ℕ), 0 < k ∧ 48 = k * (N + 3)) :
  ∃ (n : ℕ), n = 7 :=
sorry

end num_pos_int_values_l135_135219


namespace smallest_n_for_roots_l135_135717

noncomputable def smallest_n_roots_of_unity (f : ℂ → ℂ) (n : ℕ) : Prop :=
  ∀ z, f z = 0 → ∃ k, z = exp (2 * Real.pi * Complex.I * k / n)

theorem smallest_n_for_roots (n : ℕ) : smallest_n_roots_of_unity (λ z, z^6 - z^3 + 1) n ↔ n = 9 := 
by
  sorry

end smallest_n_for_roots_l135_135717


namespace product_profit_equation_l135_135180

theorem product_profit_equation (purchase_price selling_price : ℝ) 
                                (initial_units units_decrease_per_dollar_increase : ℝ)
                                (profit : ℝ)
                                (hx : purchase_price = 35)
                                (hy : selling_price = 40)
                                (hz : initial_units = 200)
                                (hs : units_decrease_per_dollar_increase = 5)
                                (hp : profit = 1870) :
  ∃ x : ℝ, (x + (selling_price - purchase_price)) * (initial_units - units_decrease_per_dollar_increase * x) = profit :=
by { sorry }

end product_profit_equation_l135_135180


namespace M_inter_N_M_union_not_N_l135_135777

def M : Set ℝ := {x | -1 ≤ x ∧ x ≤ 3}
def N : Set ℝ := {x | x > 0}

theorem M_inter_N :
  M ∩ N = {x | 0 < x ∧ x ≤ 3} := 
sorry

theorem M_union_not_N :
  M ∪ {x | x ≤ 0} = {x | x ≤ 3} := 
sorry

end M_inter_N_M_union_not_N_l135_135777


namespace ride_count_l135_135557

noncomputable def initial_tickets : ℕ := 287
noncomputable def spent_on_games : ℕ := 134
noncomputable def earned_tickets : ℕ := 32
noncomputable def cost_per_ride : ℕ := 17

theorem ride_count (initial_tickets : ℕ) (spent_on_games : ℕ) (earned_tickets : ℕ) (cost_per_ride : ℕ) : 
  initial_tickets = 287 ∧ spent_on_games = 134 ∧ earned_tickets = 32 ∧ cost_per_ride = 17 → (initial_tickets - spent_on_games + earned_tickets) / cost_per_ride = 10 :=
by
  intros
  sorry

end ride_count_l135_135557


namespace carrie_expected_strawberries_l135_135616

noncomputable def calculate_strawberries (base height : ℝ) (plants_per_sq_ft strawberries_per_plant : ℝ) : ℝ :=
  let area := (1/2) * base * height
  let total_plants := plants_per_sq_ft * area
  total_plants * strawberries_per_plant

theorem carrie_expected_strawberries : calculate_strawberries 10 12 5 8 = 2400 :=
by
  /-
  Given: base = 10, height = 12, plants_per_sq_ft = 5, strawberries_per_plant = 8
  - calculate the area of the right triangle garden
  - calculate the total number of plants
  - calculate the total number of strawberries
  -/
  sorry

end carrie_expected_strawberries_l135_135616


namespace xyz_value_l135_135253

variables {x y z : ℂ}

theorem xyz_value (h1 : x * y + 2 * y = -8)
                  (h2 : y * z + 2 * z = -8)
                  (h3 : z * x + 2 * x = -8) :
  x * y * z = 32 :=
by
  sorry

end xyz_value_l135_135253


namespace part1_part2_l135_135940

variable (x k : ℝ)

-- Part (1)
theorem part1 (h1 : x = 3) : ∀ k : ℝ, (1 + k) * 3 ≤ k^2 + k + 4 := sorry

-- Part (2)
theorem part2 (h2 : ∀ k : ℝ, -4 ≤ k → (1 + k) * x ≤ k^2 + k + 4) : -5 ≤ x ∧ x ≤ 3 := sorry

end part1_part2_l135_135940


namespace part1_solution_set_part2_range_of_a_l135_135084

-- Define the function f
def f (a x : ℝ) : ℝ := |3 * x - 1| + a * x + 3

-- Problem 1: When a = 1, solve the inequality f(x) ≤ 5
theorem part1_solution_set : 
  { x : ℝ | f 1 x ≤ 5 } = {x : ℝ | -1 / 2 ≤ x ∧ x ≤ 3 / 4} := 
  by 
  sorry

-- Problem 2: Determine the range of a for which f(x) has a minimum
theorem part2_range_of_a :
  ∀ a : ℝ, (∀ x : ℝ, x < 1/3 → f a x ≤ f a 1/3) → 
           (∀ x : ℝ, x ≥ 1/3 → f a x ≥ f a 1/3) ↔ 
           (-3 ≤ a ∧ a ≤ 3) := 
  by
  sorry

end part1_solution_set_part2_range_of_a_l135_135084


namespace mean_value_of_angles_mean_value_of_interior_angles_of_quadrilateral_l135_135343

-- Defining the sum of interior angles of a quadrilateral.
def sum_of_interior_angles (quad : Type) [IsQuadrilateral quad] : ℝ :=
  360

-- Defining the number of angles in a quadrilateral.
def number_of_angles (quad : Type) [IsQuadrilateral quad] : ℕ :=
  4

-- The theorem we want to prove: the mean value of the measures of the four interior angles of any quadrilateral.
theorem mean_value_of_angles (quad : Type) [IsQuadrilateral quad] : ℝ :=
  sum_of_interior_angles quad / number_of_angles quad
 
-- Proving the theorem.
theorem mean_value_of_interior_angles_of_quadrilateral (quad : Type) [IsQuadrilateral quad]:
  mean_value_of_angles quad = 90 :=
by
  sorry

end mean_value_of_angles_mean_value_of_interior_angles_of_quadrilateral_l135_135343


namespace zero_point_in_interval_l135_135765

open Real

noncomputable def f (x : ℝ) : ℝ :=
  log (x + 1) / log 2 - 2 / x

theorem zero_point_in_interval :
  f 1 < 0 ∧ f 2 > 0 → ∃ x ∈ Ioo 1 2, f x = 0 :=
by
  have f1 : f 1 < 0 := sorry
  have f2 : f 2 > 0 := sorry
  sorry

end zero_point_in_interval_l135_135765


namespace average_marks_two_classes_correct_l135_135170

axiom average_marks_first_class : ℕ → ℕ → ℕ
axiom average_marks_second_class : ℕ → ℕ → ℕ
axiom combined_average_marks_correct : ℕ → ℕ → Prop

theorem average_marks_two_classes_correct :
  average_marks_first_class 39 45 = 39 * 45 →
  average_marks_second_class 35 70 = 35 * 70 →
  combined_average_marks_correct (average_marks_first_class 39 45) (average_marks_second_class 35 70) :=
by
  intros h1 h2
  sorry

end average_marks_two_classes_correct_l135_135170


namespace CapeMay_more_than_twice_Daytona_l135_135194

def Daytona_sharks : ℕ := 12
def CapeMay_sharks : ℕ := 32

theorem CapeMay_more_than_twice_Daytona : CapeMay_sharks - 2 * Daytona_sharks = 8 := by
  sorry

end CapeMay_more_than_twice_Daytona_l135_135194


namespace find_g5_l135_135863

noncomputable def g (x : ℤ) : ℤ := sorry

axiom condition1 : g 1 > 1
axiom condition2 : ∀ x y : ℤ, g (x + y) + x * g y + y * g x = g x * g y + x + y + x * y
axiom condition3 : ∀ x : ℤ, 3 * g x = g (x + 1) + 2 * x - 1

theorem find_g5 : g 5 = 248 := 
by
  sorry

end find_g5_l135_135863


namespace equations_solutions_l135_135829

-- Definition and statement for Equation 1
noncomputable def equation1_solution1 : ℝ :=
  (-3 + Real.sqrt 17) / 4

noncomputable def equation1_solution2 : ℝ :=
  (-3 - Real.sqrt 17) / 4

-- Definition and statement for Equation 2
def equation2_solution : ℝ :=
  -6

-- Theorem proving the solutions to the given equations
theorem equations_solutions :
  (∃ x : ℝ, 2 * x^2 + 3 * x = 1 ∧ (x = equation1_solution1 ∨ x = equation1_solution2)) ∧
  (∃ x : ℝ, 3 / (x - 2) = 5 / (2 - x) - 1 ∧ x = equation2_solution) :=
by
  sorry

end equations_solutions_l135_135829


namespace percent_problem_l135_135628

theorem percent_problem
  (X : ℝ)
  (h1 : 0.28 * 400 = 112)
  (h2 : 0.45 * X + 112 = 224.5) :
  X = 250 := 
sorry

end percent_problem_l135_135628


namespace perimeter_C_is_40_l135_135311

noncomputable def perimeter_of_figure_C (x y : ℝ) : ℝ :=
  2 * x + 6 * y

theorem perimeter_C_is_40 (x y : ℝ) (h1 : 6 * x + 2 * y = 56) (h2 : 4 * x + 6 * y = 56) :
  perimeter_of_figure_C x y = 40 :=
by
  -- Define initial conditions
  have eq1 : 3 * x + y = 28, by { rw [mul_assoc, mul_comm 3 x, add_assoc], exact (eq.div h1 2) }
  have eq2 : 2 * x + 3 * y = 28, by { rw [mul_assoc, mul_comm 2 x, add_assoc], exact (eq.div h2 2) }
  -- Assume the solutions are obtained from here
  have sol_x : x = 8, by sorry
  have sol_y : y = 4, by sorry
  -- Calculate the perimeter of figure C
  rw [perimeter_of_figure_C, sol_x, sol_y]
  norm_num
  trivial

-- Test case to ensure the code builds successfully
#eval perimeter_of_figure_C 8 4  -- Expected output: 40

end perimeter_C_is_40_l135_135311


namespace mean_of_quadrilateral_angles_l135_135354

theorem mean_of_quadrilateral_angles :
  ∀ (angles : List ℝ), angles.length = 4 → angles.sum = 360 → angles.sum / angles.length = 90 :=
by
  intros
  sorry

end mean_of_quadrilateral_angles_l135_135354


namespace unique_corresponding_point_l135_135045

-- Define the points for the squares
structure Point := (x : ℝ) (y : ℝ)

structure Square :=
  (a b c d : Point)

def contains (sq1 sq2: Square) : Prop :=
  sq2.a.x >= sq1.a.x ∧ sq2.a.y >= sq1.a.y ∧
  sq2.b.x <= sq1.b.x ∧ sq2.b.y >= sq1.b.y ∧
  sq2.c.x <= sq1.c.x ∧ sq2.c.y <= sq1.c.y ∧
  sq2.d.x >= sq1.d.x ∧ sq2.d.y <= sq1.d.y

theorem unique_corresponding_point
  (sq1 sq2 : Square)
  (h1 : contains sq1 sq2)
  (h2 : sq1.a.x - sq1.c.x = sq2.a.x - sq2.c.x ∧ sq1.a.y - sq1.c.y = sq2.a.y - sq2.c.y):
  ∃! (O : Point), ∃ O' : Point, contains sq1 sq2 ∧ 
  (O.x - sq1.a.x) / (sq1.b.x - sq1.a.x) = (O'.x - sq2.a.x) / (sq2.b.x - sq2.a.x) ∧ 
  (O.y - sq1.a.y) / (sq1.d.y - sq1.a.y) = (O'.y - sq2.a.y) / (sq2.d.y - sq2.a.y) := 
sorry

end unique_corresponding_point_l135_135045


namespace mean_of_quadrilateral_angles_l135_135355

theorem mean_of_quadrilateral_angles :
  ∀ (angles : List ℝ), angles.length = 4 → angles.sum = 360 → angles.sum / angles.length = 90 :=
by
  intros
  sorry

end mean_of_quadrilateral_angles_l135_135355


namespace find_g5_l135_135837

def g : ℤ → ℤ := sorry

axiom g_cond1 : g 1 > 1
axiom g_cond2 : ∀ x y : ℤ, g (x + y) + x * g y + y * g x = g x * g y + x + y + x * y
axiom g_cond3 : ∀ x : ℤ, 3 * g x = g (x + 1) + 2 * x - 1

theorem find_g5 : g 5 = 248 :=
by
  sorry

end find_g5_l135_135837


namespace largest_possible_N_l135_135816

theorem largest_possible_N (N : ℕ) (divisors : List ℕ) 
  (h1 : divisors = divisors.filter (λ d, N % d = 0)) 
  (h2 : divisors.length > 2) 
  (h3 : List.nth divisors (divisors.length - 3) = some (21 * (List.nth! divisors 1))) :
  N = 441 :=
sorry

end largest_possible_N_l135_135816


namespace christine_aquafaba_needed_l135_135196

-- Define the number of tablespoons per egg white
def tablespoons_per_egg_white : ℕ := 2

-- Define the number of egg whites per cake
def egg_whites_per_cake : ℕ := 8

-- Define the number of cakes
def number_of_cakes : ℕ := 2

-- Express the total amount of aquafaba needed
def aquafaba_needed : ℕ :=
  tablespoons_per_egg_white * egg_whites_per_cake * number_of_cakes

-- Statement asserting the amount of aquafaba needed is 32
theorem christine_aquafaba_needed : aquafaba_needed = 32 := by
  sorry

end christine_aquafaba_needed_l135_135196


namespace smallest_integer_remainder_l135_135698

theorem smallest_integer_remainder (n : ℕ) 
  (h5 : n ≡ 1 [MOD 5]) (h7 : n ≡ 1 [MOD 7]) (h8 : n ≡ 1 [MOD 8]) :
  80 < n ∧ n < 299 := 
sorry

end smallest_integer_remainder_l135_135698


namespace max_area_triangle_after_t_seconds_l135_135137

-- Define the problem conditions and question
def second_hand_rotation_rate : ℝ := 6 -- degrees per second
def minute_hand_rotation_rate : ℝ := 0.1 -- degrees per second
def perpendicular_angle : ℝ := 90 -- degrees

theorem max_area_triangle_after_t_seconds : 
  ∃ (t : ℝ), (second_hand_rotation_rate - minute_hand_rotation_rate) * t = perpendicular_angle ∧ t = 15 + 15 / 59 :=
by
  -- This is a statement of the proof problem; the proof itself is omitted.
  sorry

end max_area_triangle_after_t_seconds_l135_135137


namespace odd_function_sum_l135_135644

noncomputable def f : ℝ → ℝ := sorry

theorem odd_function_sum :
  (∀ x, f x = -f (-x)) ∧ 
  (∀ x y (hx : 3 ≤ x) (hy : y ≤ 7), x < y → f x < f y) ∧ 
  ( ∃ x, 3 ≤ x ∧ x ≤ 6 ∧ f x = 8) ∧ 
  ( ∃ x, 3 ≤ x ∧ x ≤ 6 ∧ f x = -1) →
  (2 * f (-6) + f (-3) = -15) :=
by
  intros
  sorry

end odd_function_sum_l135_135644


namespace Doris_spent_6_l135_135874

variable (D : ℝ)

theorem Doris_spent_6 (h0 : 24 - (D + D / 2) = 15) : D = 6 :=
by
  sorry

end Doris_spent_6_l135_135874


namespace mean_of_quadrilateral_angles_l135_135379

theorem mean_of_quadrilateral_angles : 
  (∀ (a b c d : ℝ), a + b + c + d = 360 → (a + b + c + d) / 4 = 90) :=
begin
  intros a b c d h,
  calc
    (a + b + c + d) / 4 = 360 / 4 : by rw h
                    ... = 90 : by norm_num,
end

end mean_of_quadrilateral_angles_l135_135379


namespace mean_value_of_interior_angles_of_quadrilateral_l135_135402

theorem mean_value_of_interior_angles_of_quadrilateral 
  (sum_of_interior_angles : ℝ := 360) 
  (number_of_angles : ℝ := 4) : 
  sum_of_interior_angles / number_of_angles = 90 := 
by 
  sorry

end mean_value_of_interior_angles_of_quadrilateral_l135_135402


namespace convert_octal_127_to_binary_l135_135758

def octal_to_binary (n : ℕ) : ℕ :=
  match n with
  | 1 => 3  -- 001 in binary
  | 2 => 2  -- 010 in binary
  | 7 => 7  -- 111 in binary
  | _ => 0  -- No other digits are used in this example

theorem convert_octal_127_to_binary :
  octal_to_binary 1 * 1000000 + octal_to_binary 2 * 1000 + octal_to_binary 7 = 1010111 :=
by
  -- Proof would go here
  sorry

end convert_octal_127_to_binary_l135_135758


namespace greatest_whole_number_with_odd_factors_less_than_150_l135_135017

theorem greatest_whole_number_with_odd_factors_less_than_150 :
  ∃ (n : ℕ), (∀ (m : ℕ), m < 150 ∧ odd_factors m → m ≤ n) ∧ n = 144 :=
by
  sorry

def odd_factors (k : ℕ) : Prop :=
  ∃ (m : ℕ), m * m = k

end greatest_whole_number_with_odd_factors_less_than_150_l135_135017


namespace f_zero_f_pos_f_decreasing_solve_inequality_l135_135967

open Real

noncomputable def f : ℝ → ℝ := sorry

axiom f_mul_add (m n : ℝ) : f m * f n = f (m + n)
axiom f_pos_neg (x : ℝ) : x < 0 → 1 < f x

theorem f_zero : f 0 = 1 :=
sorry

theorem f_pos (x : ℝ) : 0 < x → 0 < f x ∧ f x < 1 :=
sorry

theorem f_decreasing (x₁ x₂ : ℝ) : x₁ < x₂ → f x₁ > f x₂ :=
sorry

theorem solve_inequality (a x : ℝ) :
  f (x^2 - 3 * a * x + 1) * f (-3 * x + 6 * a + 1) ≥ 1 ↔
  (a > 1/3 ∧ 2 ≤ x ∧ x ≤ 3 * a + 1) ∨
  (a = 1/3 ∧ x = 2) ∨
  (a < 1/3 ∧ 3 * a + 1 ≤ x ∧ x ≤ 2) :=
sorry

end f_zero_f_pos_f_decreasing_solve_inequality_l135_135967


namespace equilateral_triangle_area_outside_circle_ratio_l135_135773

noncomputable def equilateral_triangle_area_ratio (r : ℝ) : ℝ :=
  let s := r * (real.sqrt 3) in
  let area_triangle := (real.sqrt 3 / 4) * s^2 in
  let area_circle := real.pi * r^2 in
  (area_triangle - area_circle) / area_triangle

theorem equilateral_triangle_area_outside_circle_ratio (r : ℝ) (h : r > 0) :
  equilateral_triangle_area_ratio r = (4/3 : ℝ) - (4 * real.sqrt 3 * real.pi / 27) :=
by
  -- proof omitted
  sorry

end equilateral_triangle_area_outside_circle_ratio_l135_135773


namespace solve_system_of_equations_l135_135989

theorem solve_system_of_equations : ∃ x y : ℤ, 3 * x - 2 * y = 6 ∧ 2 * x + 3 * y = 17 ∧ x = 4 ∧ y = 3 :=
by
  sorry

end solve_system_of_equations_l135_135989


namespace largest_common_value_lt_1000_l135_135994

theorem largest_common_value_lt_1000 :
  ∃ a : ℕ, ∃ n m : ℕ, a = 4 + 5 * n ∧ a = 7 + 11 * m ∧ a < 1000 ∧ 
  (∀ b : ℕ, ∀ p q : ℕ, b = 4 + 5 * p ∧ b = 7 + 11 * q ∧ b < 1000 → b ≤ a) :=
sorry

end largest_common_value_lt_1000_l135_135994


namespace fixed_point_of_line_l135_135549

theorem fixed_point_of_line (m : ℝ) : 
  ∀ (x y : ℝ), (3 * x - 2 * y + 7 = 0) ∧ (4 * x + 5 * y - 6 = 0) → x = -1 ∧ y = 2 :=
sorry

end fixed_point_of_line_l135_135549


namespace concentric_circle_chord_ratio_l135_135105

theorem concentric_circle_chord_ratio 
  (r R : ℝ) (h1 : 0 < r) (h2 : 0 < R) (h3 : r < R) :
  ∃ (A B C D: ℝ × ℝ), 
    (A ≠ B ∧ C ≠ D) ∧ 
    (dist A B = 2 * dist C D) := sorry

end concentric_circle_chord_ratio_l135_135105


namespace ratio_of_side_lengths_sum_l135_135155

theorem ratio_of_side_lengths_sum (a b c : ℕ) (ha : a = 4) (hb : b = 15) (hc : c = 25) :
  a + b + c = 44 := 
by
  sorry

end ratio_of_side_lengths_sum_l135_135155


namespace number_of_solutions_l135_135588

noncomputable def system_of_equations (a b c : ℕ) : Prop :=
  a * b + b * c = 44 ∧ a * c + b * c = 23

theorem number_of_solutions : ∃! (a b c : ℕ), system_of_equations a b c :=
by
  sorry

end number_of_solutions_l135_135588


namespace mean_of_quadrilateral_angles_l135_135409

theorem mean_of_quadrilateral_angles : 
  (∀ (a b c d : ℝ), a + b + c + d = 360) → 90 :=
by
  intro h
  sorry

end mean_of_quadrilateral_angles_l135_135409


namespace mean_of_quadrilateral_angles_l135_135351

theorem mean_of_quadrilateral_angles :
  ∀ (angles : List ℝ), angles.length = 4 → angles.sum = 360 → angles.sum / angles.length = 90 :=
by
  intros
  sorry

end mean_of_quadrilateral_angles_l135_135351


namespace strawberries_picking_problem_l135_135545

noncomputable def StrawberriesPicked : Prop :=
  let kg_to_lb := 2.2
  let marco_pounds := 1 + 3 * kg_to_lb
  let sister_pounds := 1.5 * marco_pounds
  let father_pounds := 2 * sister_pounds
  marco_pounds = 7.6 ∧ sister_pounds = 11.4 ∧ father_pounds = 22.8

theorem strawberries_picking_problem : StrawberriesPicked :=
  sorry

end strawberries_picking_problem_l135_135545


namespace people_on_bus_before_stop_l135_135078

variable (P_before P_after P_got_on : ℕ)
variable (h1 : P_got_on = 13)
variable (h2 : P_after = 17)

theorem people_on_bus_before_stop : P_before = 4 :=
by
  -- Given that P_after = 17 and P_got_on = 13
  -- We need to prove P_before = P_after - P_got_on = 4
  sorry

end people_on_bus_before_stop_l135_135078


namespace sum_of_squares_five_consecutive_not_perfect_square_l135_135869

theorem sum_of_squares_five_consecutive_not_perfect_square 
  (x : ℤ) : ¬ ∃ k : ℤ, (x-2)^2 + (x-1)^2 + x^2 + (x+1)^2 + (x+2)^2 = k^2 :=
by 
  sorry

end sum_of_squares_five_consecutive_not_perfect_square_l135_135869


namespace total_people_is_120_l135_135831

def num_children : ℕ := 80

def num_adults (num_children : ℕ) : ℕ := num_children / 2

def total_people (num_children num_adults : ℕ) : ℕ := num_children + num_adults

theorem total_people_is_120 : total_people num_children (num_adults num_children) = 120 := by
  sorry

end total_people_is_120_l135_135831


namespace mean_value_of_quadrilateral_angles_l135_135327

-- Statement of the problem: mean value of interior angles in any quadrilateral is 90°
theorem mean_value_of_quadrilateral_angles : 
  (∀ (a b c d : ℝ), a + b + c + d = 360 → (a + b + c + d) / 4 = 90) :=
by
  sorry

end mean_value_of_quadrilateral_angles_l135_135327


namespace mean_value_of_interior_angles_of_quadrilateral_l135_135368

theorem mean_value_of_interior_angles_of_quadrilateral :
  (360 / 4) = 90 := 
by
  sorry

end mean_value_of_interior_angles_of_quadrilateral_l135_135368


namespace figure_C_perimeter_l135_135304

def is_perimeter (figure : Type) (perimeter : ℕ) : Prop :=
∃ x y : ℕ, (figure = 'A' → 6*x + 2*y = perimeter) ∧ 
           (figure = 'B' → 4*x + 6*y = perimeter) ∧
           (figure = 'C' → 2*x + 6*y = perimeter)

theorem figure_C_perimeter (hA : is_perimeter 'A' 56) (hB : is_perimeter 'B' 56) : 
  is_perimeter 'C' 40 :=
by
  sorry

end figure_C_perimeter_l135_135304


namespace correct_operation_l135_135049

variable (a b : ℝ)

theorem correct_operation : 3 * a^2 * b - b * a^2 = 2 * a^2 * b := 
sorry

end correct_operation_l135_135049


namespace third_year_award_count_l135_135980

-- Define the variables and conditions
variables (x x1 x2 x3 x4 x5 : ℕ)

-- The conditions and definition for the problem
def conditions : Prop :=
  (x1 = x) ∧
  (x5 = 3 * x) ∧
  (x1 < x2) ∧
  (x2 < x3) ∧
  (x3 < x4) ∧
  (x4 < x5) ∧
  (x1 + x2 + x3 + x4 + x5 = 27)

-- The theorem statement
theorem third_year_award_count (h : conditions x x1 x2 x3 x4 x5) : x3 = 5 :=
sorry

end third_year_award_count_l135_135980


namespace tshirt_cost_l135_135528

-- Definitions based on conditions
def pants_cost : ℝ := 80
def shoes_cost : ℝ := 150
def discount : ℝ := 0.1
def total_paid : ℝ := 558

-- Variables based on the problem
variable (T : ℝ) -- Cost of one T-shirt
def num_tshirts : ℝ := 4
def num_pants : ℝ := 3
def num_shoes : ℝ := 2

-- Theorem: The cost of one T-shirt is $20
theorem tshirt_cost : T = 20 :=
by
  have total_cost : ℝ := (num_tshirts * T) + (num_pants * pants_cost) + (num_shoes * shoes_cost)
  have discounted_total : ℝ := (1 - discount) * total_cost
  have payment_condition : discounted_total = total_paid := sorry
  sorry -- detailed proof

end tshirt_cost_l135_135528


namespace even_stones_fraction_odd_stones_fraction_l135_135571

/-- The fraction of the distributions of 12 indistinguishable stones to 4 distinguishable boxes where every box contains an even number of stones is 12/65. -/
theorem even_stones_fraction : (∀ (B1 B2 B3 B4 : ℕ), B1 % 2 = 0 ∧ B2 % 2 = 0 ∧ B3 % 2 = 0 ∧ B4 % 2 = 0 ∧ B1 + B2 + B3 + B4 = 12) → (84 / 455 = 12 / 65) := 
by sorry

/-- The fraction of the distributions of 12 indistinguishable stones to 4 distinguishable boxes where every box contains an odd number of stones is 1/13. -/
theorem odd_stones_fraction : (∀ (B1 B2 B3 B4 : ℕ), B1 % 2 = 1 ∧ B2 % 2 = 1 ∧ B3 % 2 = 1 ∧ B4 % 2 = 1 ∧ B1 + B2 + B3 + B4 = 12) → (35 / 455 = 1 / 13) := 
by sorry

end even_stones_fraction_odd_stones_fraction_l135_135571


namespace mean_value_of_quadrilateral_angles_l135_135340

theorem mean_value_of_quadrilateral_angles : 
  (∀ (A B C D : ℕ), A + B + C + D = 360 → (A + B + C + D) / 4 = 90) :=
by {
  sorry
}

end mean_value_of_quadrilateral_angles_l135_135340


namespace mean_value_of_interior_angles_of_quadrilateral_l135_135361

theorem mean_value_of_interior_angles_of_quadrilateral :
  (360 / 4) = 90 := 
by
  sorry

end mean_value_of_interior_angles_of_quadrilateral_l135_135361


namespace smallest_M_bound_l135_135587

theorem smallest_M_bound {f : ℕ → ℝ} (hf1 : f 1 = 2) 
  (hf2 : ∀ n : ℕ, f (n + 1) ≥ f n ∧ f n ≥ (n / (n + 1)) * f (2 * n)) : 
  ∃ M : ℕ, (∀ n : ℕ, f n < M) ∧ M = 10 :=
by
  sorry

end smallest_M_bound_l135_135587


namespace cos_double_angle_l135_135790

-- Definition of the terminal condition
def terminal_side_of_angle (α : ℝ) (x y : ℝ) : Prop :=
  (x = 1) ∧ (y = Real.sqrt 3) ∧ (x^2 + y^2 = 1)

-- Prove the required statement
theorem cos_double_angle (α : ℝ) :
  (terminal_side_of_angle α 1 (Real.sqrt 3)) →
  Real.cos (2 * α + Real.pi / 2) = - Real.sqrt 3 / 2 :=
by
  sorry

end cos_double_angle_l135_135790


namespace min_value_S_l135_135682

noncomputable def S (x y : ℝ) : ℝ := 2 * x ^ 2 - x * y + y ^ 2 + 2 * x + 3 * y

theorem min_value_S : ∃ x y : ℝ, S x y = -4 ∧ ∀ (a b : ℝ), S a b ≥ -4 := 
by
  sorry

end min_value_S_l135_135682


namespace correct_calculation_l135_135167

variable (a b : ℚ)

theorem correct_calculation :
  (a / b) ^ 4 = a ^ 4 / b ^ 4 := 
by
  sorry

end correct_calculation_l135_135167


namespace numerator_of_fraction_l135_135274

theorem numerator_of_fraction (x : ℤ) (h : (x : ℚ) / (4 * x - 5) = 3 / 7) : x = 3 := 
sorry

end numerator_of_fraction_l135_135274


namespace scientific_notation_86560_l135_135925

theorem scientific_notation_86560 : ∃ a n, (86560 : ℝ) = a * 10 ^ n ∧ 1 ≤ a ∧ a < 10 ∧ a = 8.656 ∧ n = 4 :=
by {
  sorry
}

end scientific_notation_86560_l135_135925


namespace rose_spent_on_food_l135_135261

theorem rose_spent_on_food (T : ℝ) 
  (h_clothing : 0.5 * T = 0.5 * T)
  (h_other_items : 0.3 * T = 0.3 * T)
  (h_total_tax : 0.044 * T = 0.044 * T)
  (h_tax_clothing : 0.04 * 0.5 * T = 0.02 * T)
  (h_tax_other_items : 0.08 * 0.3 * T = 0.024 * T) :
  (0.2 * T = T - (0.5 * T + 0.3 * T)) :=
by sorry

end rose_spent_on_food_l135_135261


namespace smallest_n_for_roots_of_unity_l135_135722

theorem smallest_n_for_roots_of_unity (z : ℂ) (h : z^6 - z^3 + 1 = 0) : ∃ (n : ℕ), n = 18 ∧ z^n = 1 :=
by {
  sorry
}

end smallest_n_for_roots_of_unity_l135_135722


namespace smallest_n_for_root_unity_l135_135727

noncomputable def smallestPositiveInteger (p : Polynomial ℂ) (n : ℕ) : Prop :=
  n > 0 ∧ ∀ z : ℂ, z^n = 1 → p.eval z = 0 → (∃ k : ℕ, k < n ∧ z = Complex.exp (2 * Real.pi * Complex.I * k / n))

theorem smallest_n_for_root_unity (n : ℕ) : 
  smallestPositiveInteger (Polynomial.map (algebraMap ℂ ℂ) (X^6 - X^3 + 1)) n ↔ n = 9 :=
begin
  sorry
end

end smallest_n_for_root_unity_l135_135727


namespace remaining_distance_l135_135678

-- Definitions for the given conditions
def total_distance : ℕ := 436
def first_stopover_distance : ℕ := 132
def second_stopover_distance : ℕ := 236

-- Prove that the remaining distance from the second stopover to the island is 68 miles.
theorem remaining_distance : total_distance - (first_stopover_distance + second_stopover_distance) = 68 := by
  -- The proof (details) will go here
  sorry

end remaining_distance_l135_135678


namespace num_topping_combinations_l135_135760

-- Define the conditions as constants in Lean
constant cheese_options : ℕ := 3
constant meat_options : ℕ := 4
constant vegetable_options : ℕ := 5
constant pepperoni_option : ℕ := 1 -- Only one option for pepperoni
constant restricted_vegetable_options : ℕ := 1 -- Only one restricted option (peppers)

-- Define the total number of combinations without restrictions
def total_combinations : ℕ := cheese_options * meat_options * vegetable_options

-- Define the number of restricted combinations (pepperoni and peppers)
def restricted_combinations : ℕ := cheese_options * pepperoni_option * restricted_vegetable_options

-- Define the allowed combinations
def allowed_combinations : ℕ := total_combinations - restricted_combinations

-- The theorem stating the problem question and expected answer
theorem num_topping_combinations : allowed_combinations = 57 := by
  sorry

end num_topping_combinations_l135_135760


namespace perimeter_C_l135_135290

theorem perimeter_C :
  ∀ (x y : ℕ),
  (6 * x + 2 * y = 56) →
  (4 * x + 6 * y = 56) →
  (2 * x + 6 * y = 40) :=
by
  intros x y hA hB
  sorry

end perimeter_C_l135_135290


namespace exists_two_numbers_with_gcd_quotient_ge_p_plus_one_l135_135542

theorem exists_two_numbers_with_gcd_quotient_ge_p_plus_one (p : ℕ) (hp : Nat.Prime p)
  (l : List ℕ) (hl_len : l.length = p + 1) (hl_distinct : l.Nodup) :
  ∃ (a b : ℕ), a ≠ b ∧ a ∈ l ∧ b ∈ l ∧ a > b ∧ a / (Nat.gcd a b) ≥ p + 1 := sorry

end exists_two_numbers_with_gcd_quotient_ge_p_plus_one_l135_135542


namespace odd_positive_int_divides_3pow_n_plus_1_l135_135213

theorem odd_positive_int_divides_3pow_n_plus_1 (n : ℕ) (hn_odd : n % 2 = 1) (hn_pos : n > 0) : 
  n ∣ (3^n + 1) ↔ n = 1 := 
by
  sorry

end odd_positive_int_divides_3pow_n_plus_1_l135_135213


namespace area_of_triangle_is_sqrt3_l135_135674

theorem area_of_triangle_is_sqrt3
  (a b c : ℝ)
  (B : ℝ)
  (h_geom_prog : b^2 = a * c)
  (h_b : b = 2)
  (h_B : B = Real.pi / 3) :
  (1/2) * a * c * Real.sin B = Real.sqrt 3 := 
by
  sorry

end area_of_triangle_is_sqrt3_l135_135674


namespace petya_catch_bus_l135_135825

theorem petya_catch_bus 
    (v_p v_b d : ℝ) 
    (h1 : v_b = 5 * v_p)
    (h2 : ∀ t : ℝ, 5 * v_p * t ≤ 0.6) 
    : d = 0.12 := 
sorry

end petya_catch_bus_l135_135825


namespace largest_angle_smallest_angle_middle_angle_l135_135729

-- Definitions for angles of a triangle in degrees
variable (α β γ : ℝ)
variable (h_sum : α + β + γ = 180)

-- Largest angle condition
theorem largest_angle (h1 : α ≥ β) (h2 : α ≥ γ) : (60 ≤ α ∧ α < 180) :=
  sorry

-- Smallest angle condition
theorem smallest_angle (h1 : α ≤ β) (h2 : α ≤ γ) : (0 < α ∧ α ≤ 60) :=
  sorry

-- Middle angle condition
theorem middle_angle (h1 : α > β ∧ α < γ ∨ α < β ∧ α > γ) : (0 < α ∧ α < 90) :=
  sorry

end largest_angle_smallest_angle_middle_angle_l135_135729


namespace trains_meet_in_2067_seconds_l135_135046

def length_of_train1 : ℝ := 100  -- Length of Train 1 in meters
def length_of_train2 : ℝ := 200  -- Length of Train 2 in meters
def initial_distance : ℝ := 630  -- Initial distance between trains in meters
def speed_of_train1_kmh : ℝ := 90  -- Speed of Train 1 in km/h
def speed_of_train2_kmh : ℝ := 72  -- Speed of Train 2 in km/h

noncomputable def speed_of_train1_ms : ℝ := speed_of_train1_kmh * (1000 / 3600)
noncomputable def speed_of_train2_ms : ℝ := speed_of_train2_kmh * (1000 / 3600)
noncomputable def relative_speed : ℝ := speed_of_train1_ms + speed_of_train2_ms
noncomputable def total_distance : ℝ := initial_distance + length_of_train1 + length_of_train2
noncomputable def time_to_meet : ℝ := total_distance / relative_speed

theorem trains_meet_in_2067_seconds : time_to_meet = 20.67 := 
by
  sorry

end trains_meet_in_2067_seconds_l135_135046


namespace PQRS_product_l135_135224

noncomputable def P : ℝ := (Real.sqrt 2023 + Real.sqrt 2024)
noncomputable def Q : ℝ := (-Real.sqrt 2023 - Real.sqrt 2024)
noncomputable def R : ℝ := (Real.sqrt 2023 - Real.sqrt 2024)
noncomputable def S : ℝ := (Real.sqrt 2024 - Real.sqrt 2023)

theorem PQRS_product : (P * Q * R * S) = 1 := 
by 
  sorry

end PQRS_product_l135_135224


namespace jogging_days_in_second_week_l135_135123

theorem jogging_days_in_second_week
  (daily_jogging_time : ℕ) (first_week_days : ℕ) (total_jogging_time : ℕ) :
  daily_jogging_time = 30 →
  first_week_days = 3 →
  total_jogging_time = 240 →
  ∃ second_week_days : ℕ, second_week_days = 5 :=
by
  intros
  -- Conditions
  have h1 := daily_jogging_time = 30
  have h2 := first_week_days = 3
  have h3 := total_jogging_time = 240
  -- Calculations
  have first_week_time := first_week_days * daily_jogging_time
  have second_week_time := total_jogging_time - first_week_time
  have second_week_days := second_week_time / daily_jogging_time
  -- Conclusion
  use second_week_days
  sorry

end jogging_days_in_second_week_l135_135123


namespace ribbon_left_after_wrapping_l135_135264

def total_ribbon_needed (gifts : ℕ) (ribbon_per_gift : ℝ) : ℝ :=
  gifts * ribbon_per_gift

def remaining_ribbon (initial_ribbon : ℝ) (used_ribbon : ℝ) : ℝ :=
  initial_ribbon - used_ribbon

theorem ribbon_left_after_wrapping : 
  ∀ (gifts : ℕ) (ribbon_per_gift initial_ribbon : ℝ),
  gifts = 8 →
  ribbon_per_gift = 1.5 →
  initial_ribbon = 15 →
  remaining_ribbon initial_ribbon (total_ribbon_needed gifts ribbon_per_gift) = 3 :=
by
  intros gifts ribbon_per_gift initial_ribbon h1 h2 h3
  rw [h1, h2, h3]
  simp [total_ribbon_needed, remaining_ribbon]
  sorry

end ribbon_left_after_wrapping_l135_135264


namespace birds_problem_l135_135990

theorem birds_problem 
  (x y z : ℕ) 
  (h1 : x + y + z = 30) 
  (h2 : (1 / 3 : ℚ) * x + (1 / 2 : ℚ) * y + 2 * z = 30) 
  : x = 9 ∧ y = 10 ∧ z = 11 := 
  by {
  -- Proof steps would go here
  sorry
}

end birds_problem_l135_135990


namespace james_main_game_time_l135_135959

-- Define the download time in minutes
def download_time := 10

-- Define the installation time as half the download time
def installation_time := download_time / 2

-- Define the combined time for download and installation
def combined_time := download_time + installation_time

-- Define the tutorial time as triple the combined time
def tutorial_time := combined_time * 3

-- Define the total time as the combined time plus the tutorial time
def total_time := combined_time + tutorial_time

-- Statement of the problem to prove
theorem james_main_game_time : total_time = 60 := by
  sorry

end james_main_game_time_l135_135959


namespace expected_total_rainfall_over_week_l135_135489

noncomputable def daily_rain_expectation : ℝ :=
  (0.5 * 0) + (0.2 * 2) + (0.3 * 5)

noncomputable def total_rain_expectation (days: ℕ) : ℝ :=
  days * daily_rain_expectation

theorem expected_total_rainfall_over_week : total_rain_expectation 7 = 13.3 :=
by 
  -- calculation of expected value here
  -- daily_rain_expectation = 1.9
  -- total_rain_expectation 7 = 7 * 1.9 = 13.3
  sorry

end expected_total_rainfall_over_week_l135_135489


namespace all_positive_integers_in_A_l135_135642

variable (A : Set ℕ)

-- Conditions
def has_at_least_three_elements : Prop :=
  ∃ a b c : ℕ, a ∈ A ∧ b ∈ A ∧ c ∈ A ∧ a ≠ b ∧ a ≠ c ∧ b ≠ c

def all_divisors_in_set : Prop :=
  ∀ m : ℕ, m ∈ A → (∀ d : ℕ, d ∣ m → d ∈ A)

def  bc_plus_one_in_set : Prop :=
  ∀ b c : ℕ, 1 < b → b < c → b ∈ A → c ∈ A → 1 + b * c ∈ A

-- Theorem statement
theorem all_positive_integers_in_A
  (h1 : has_at_least_three_elements A)
  (h2 : all_divisors_in_set A)
  (h3 : bc_plus_one_in_set A) : ∀ n : ℕ, n > 0 → n ∈ A := 
by
  -- proof steps would go here
  sorry

end all_positive_integers_in_A_l135_135642


namespace mean_value_of_quadrilateral_angles_l135_135329

-- Statement of the problem: mean value of interior angles in any quadrilateral is 90°
theorem mean_value_of_quadrilateral_angles : 
  (∀ (a b c d : ℝ), a + b + c + d = 360 → (a + b + c + d) / 4 = 90) :=
by
  sorry

end mean_value_of_quadrilateral_angles_l135_135329


namespace max_marks_l135_135472

theorem max_marks (M : ℝ) (h1 : 0.33 * M = 59 + 40) : M = 300 :=
by
  sorry

end max_marks_l135_135472


namespace largest_spherical_ball_radius_in_torus_l135_135900

theorem largest_spherical_ball_radius_in_torus 
    (inner_radius outer_radius : ℝ) 
    (circle_center : ℝ × ℝ × ℝ) 
    (circle_radius : ℝ) 
    (r : ℝ)
    (h0 : inner_radius = 2)
    (h1 : outer_radius = 4)
    (h2 : circle_center = (3, 0, 1))
    (h3 : circle_radius = 1)
    (h4 : 3^2 + (r - 1)^2 = (r + 1)^2) :
    r = 9 / 4 :=
by
  sorry

end largest_spherical_ball_radius_in_torus_l135_135900


namespace min_value_frac_f1_f_l135_135640

theorem min_value_frac_f1_f'0 (a b c : ℝ) (h_a : a > 0) (h_b : b > 0) (h_c : c > 0) (h_discriminant : b^2 ≤ 4 * a * c) :
  (a + b + c) / b ≥ 2 := 
by
  -- Here goes the proof
  sorry

end min_value_frac_f1_f_l135_135640


namespace probability_product_multiple_of_10_l135_135664

-- Set definition
def num_set : set ℕ := {5, 7, 9, 10}

-- Definition of the condition: selection of 2 numbers without replacement
def choose_2_without_replacement (s : set ℕ) : set (set ℕ) :=
  {x | x ⊆ s ∧ x.card = 2 }

-- Definition of a successful pair (product being a multiple of 10)
def successful_pair (x : set ℕ) : Prop :=
  ∃ a b, a ∈ x ∧ b ∈ x ∧ a * b % 10 = 0

-- Our main goal
theorem probability_product_multiple_of_10 :
  (choose_2_without_replacement num_set).count successful_pair = 1 / 2 :=
sorry

end probability_product_multiple_of_10_l135_135664


namespace condition_suff_and_nec_l135_135539

def p (x : ℝ) : Prop := |x + 2| ≤ 3
def q (x : ℝ) : Prop := x < -8

theorem condition_suff_and_nec (x : ℝ) : p x ↔ ¬ q x :=
by
  sorry

end condition_suff_and_nec_l135_135539


namespace mean_value_of_interior_angles_of_quadrilateral_l135_135364

theorem mean_value_of_interior_angles_of_quadrilateral :
  (360 / 4) = 90 := 
by
  sorry

end mean_value_of_interior_angles_of_quadrilateral_l135_135364


namespace determine_marbles_l135_135318

noncomputable def marbles_total (x : ℚ) := (4 * x + 2) + (2 * x) + (3 * x - 1)

theorem determine_marbles (x : ℚ) (h1 : marbles_total x = 47) :
  (4 * x + 2 = 202 / 9) ∧ (2 * x = 92 / 9) ∧ (3 * x - 1 = 129 / 9) :=
by
  sorry

end determine_marbles_l135_135318


namespace mean_value_of_quadrilateral_interior_angles_l135_135467

theorem mean_value_of_quadrilateral_interior_angles (a b c d : ℝ) 
  (h_sum : a + b + c + d = 360) : 
  (a + b + c + d) / 4 = 90 :=
by
  sorry

end mean_value_of_quadrilateral_interior_angles_l135_135467


namespace hockey_league_games_l135_135317

theorem hockey_league_games (n t : ℕ) (h1 : n = 15) (h2 : t = 1050) :
  ∃ k, ∀ team1 team2 : ℕ, team1 ≠ team2 → k = 10 :=
by
  -- Declare k as the number of times each team faces the other teams
  let k := 10
  -- Verify the total number of teams and games
  have hn : n = 15 := h1
  have ht : t = 1050 := h2
  -- For any two distinct teams, they face each other k times
  use k
  intros team1 team2 hneq
  -- Show that k equals 10 under given conditions
  exact rfl

end hockey_league_games_l135_135317


namespace orthocenter_PQR_l135_135786

structure Point3D :=
  (x : ℚ)
  (y : ℚ)
  (z : ℚ)

def orthocenter (P Q R : Point3D) : Point3D :=
  sorry

theorem orthocenter_PQR :
  orthocenter ⟨2, 3, 4⟩ ⟨6, 4, 2⟩ ⟨4, 5, 6⟩ = ⟨1/2, 13/2, 15/2⟩ :=
by {
  sorry
}

end orthocenter_PQR_l135_135786


namespace kim_boxes_on_thursday_l135_135004

theorem kim_boxes_on_thursday (Tues Wed Thurs : ℕ) 
(h1 : Tues = 4800)
(h2 : Tues = 2 * Wed)
(h3 : Wed = 2 * Thurs) : Thurs = 1200 :=
by
  sorry

end kim_boxes_on_thursday_l135_135004


namespace james_problem_l135_135798

def probability_at_least_two_green_apples (total: ℕ) (red: ℕ) (green: ℕ) (yellow: ℕ) (choices: ℕ) : ℚ :=
  let favorable_outcomes := (Nat.choose green 2) * (Nat.choose (total - green) 1) + (Nat.choose green 3)
  let total_outcomes := Nat.choose total choices
  favorable_outcomes / total_outcomes

theorem james_problem : probability_at_least_two_green_apples 10 5 3 2 3 = 11 / 60 :=
by sorry

end james_problem_l135_135798


namespace college_application_distributions_l135_135144

theorem college_application_distributions : 
  let total_students := 6
  let colleges := 3
  ∃ n : ℕ, n = 540 ∧ 
    (n = (colleges^total_students - colleges * (2^total_students) + 
      (colleges.choose 2) * 1)) := sorry

end college_application_distributions_l135_135144


namespace mean_of_quadrilateral_angles_l135_135413

theorem mean_of_quadrilateral_angles : 
  (∀ (a b c d : ℝ), a + b + c + d = 360) → 90 :=
by
  intro h
  sorry

end mean_of_quadrilateral_angles_l135_135413


namespace coefficient_x_squared_l135_135098

variable {a w c d : ℝ}

/-- The coefficient of x^2 in the expanded form of the equation (ax + w)(cx + d) = 6x^2 + x - 12 -/
theorem coefficient_x_squared (h1 : (a * x + w) * (c * x + d) = 6 * x^2 + x - 12)
                             (h2 : abs a + abs w + abs c + abs d = 12) :
  a * c = 6 :=
  sorry

end coefficient_x_squared_l135_135098


namespace mean_value_of_quadrilateral_angles_l135_135394

theorem mean_value_of_quadrilateral_angles (a b c d : ℝ) (h_angles_sum : a + b + c + d = 360) : (a + b + c + d) / 4 = 90 := 
by
  sorry

end mean_value_of_quadrilateral_angles_l135_135394


namespace xiaodong_election_l135_135470

theorem xiaodong_election (V : ℕ) (h1 : 0 < V) :
  let total_needed := (3 : ℚ) / 4 * V
  let votes_obtained := (5 : ℚ) / 6 * (2 : ℚ) / 3 * V
  let remaining_votes := V - (2 : ℚ) / 3 * V
  total_needed - votes_obtained = (7 : ℚ) / 12 * remaining_votes :=
by 
  sorry

end xiaodong_election_l135_135470


namespace part_i_part_ii_l135_135010

-- Define the variables and conditions
variable (a b : ℝ)
variable (h₁ : a > 0)
variable (h₂ : b > 0)
variable (h₃ : a + b = 1 / a + 1 / b)

-- Prove the first part: a + b ≥ 2
theorem part_i : a + b ≥ 2 := by
  sorry

-- Prove the second part: It is impossible for both a² + a < 2 and b² + b < 2 simultaneously
theorem part_ii : ¬(a^2 + a < 2 ∧ b^2 + b < 2) := by
  sorry

end part_i_part_ii_l135_135010


namespace octal_to_binary_conversion_l135_135757

theorem octal_to_binary_conversion :
  ∃ b : ℕ, octal_to_decimal 127 = b ∧ decimal_to_binary b = 1010111 :=
by
  sorry

-- Supporting definitions that capture the concepts used in the problem
def octal_to_decimal (o : ℕ) : ℕ :=
  -- Implement the conversion of an octal number (represented as a natural number) to a decimal number
  sorry

def decimal_to_binary (d : ℕ) : ℕ :=
  -- Implement the conversion of a decimal number to a binary number (represented as a natural number)
  sorry

end octal_to_binary_conversion_l135_135757


namespace perimeter_C_l135_135284

theorem perimeter_C (x y : ℕ) 
  (h1 : 6 * x + 2 * y = 56)
  (h2 : 4 * x + 6 * y = 56) :
  2 * x + 6 * y = 40 := 
by
  sorry

end perimeter_C_l135_135284


namespace mean_value_of_quadrilateral_angles_l135_135339

theorem mean_value_of_quadrilateral_angles : 
  (∀ (A B C D : ℕ), A + B + C + D = 360 → (A + B + C + D) / 4 = 90) :=
by {
  sorry
}

end mean_value_of_quadrilateral_angles_l135_135339


namespace time_both_pipes_opened_l135_135161

def fill_rate_p := 1 / 10
def fill_rate_q := 1 / 15
def total_fill_rate := fill_rate_p + fill_rate_q -- Combined fill rate of both pipes

def remaining_fill_rate := 10 * fill_rate_q -- Fill rate of pipe q in 10 minutes

theorem time_both_pipes_opened (t : ℝ) :
  (t / 6) + (2 / 3) = 1 → t = 2 :=
by
  sorry

end time_both_pipes_opened_l135_135161


namespace ratio_night_to_day_l135_135893

-- Definitions based on conditions
def birds_day : ℕ := 8
def birds_total : ℕ := 24
def birds_night : ℕ := birds_total - birds_day

-- Theorem statement
theorem ratio_night_to_day : birds_night / birds_day = 2 := by
  sorry

end ratio_night_to_day_l135_135893


namespace mean_value_of_quadrilateral_angles_l135_135449

theorem mean_value_of_quadrilateral_angles :
  let sum_of_angles := 360 in
  let number_of_angles := 4 in
  sum_of_angles / number_of_angles = 90 :=
by
  sorry

end mean_value_of_quadrilateral_angles_l135_135449


namespace christine_aquafaba_needed_l135_135197

-- Define the number of tablespoons per egg white
def tablespoons_per_egg_white : ℕ := 2

-- Define the number of egg whites per cake
def egg_whites_per_cake : ℕ := 8

-- Define the number of cakes
def number_of_cakes : ℕ := 2

-- Express the total amount of aquafaba needed
def aquafaba_needed : ℕ :=
  tablespoons_per_egg_white * egg_whites_per_cake * number_of_cakes

-- Statement asserting the amount of aquafaba needed is 32
theorem christine_aquafaba_needed : aquafaba_needed = 32 := by
  sorry

end christine_aquafaba_needed_l135_135197


namespace identical_digits_divisible_l135_135182

  theorem identical_digits_divisible (n : ℕ) (hn : n > 0) : 
    ∀ a : ℕ, (10^(3^n - 1) * a / 9) % 3^n = 0 := 
  by
    intros
    sorry
  
end identical_digits_divisible_l135_135182


namespace restaurant_total_cost_l135_135243

theorem restaurant_total_cost (burger_cost pizza_cost : ℕ)
    (h1 : burger_cost = 9)
    (h2 : pizza_cost = 2 * burger_cost) :
    pizza_cost + 3 * burger_cost = 45 := 
by
  sorry

end restaurant_total_cost_l135_135243


namespace probability_positive_slopes_l135_135255

-- Define the points as independent and uniformly chosen within the unit square
structure Point :=
  (x y : ℝ)
  (prop : 0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 1)

def chosen_points : list Point := [A, B, C, D]

-- Condition for positive slopes between all pairs
def positive_slope (p1 p2 : Point) : Prop :=
  p1.x < p2.x ∧ p1.y < p2.y

-- The main theorem statement
theorem probability_positive_slopes :
  ∀ (A B C D : Point),
    (independent [A, B, C, D] ∧ uniform [A.x, B.x, C.x, D.x] ∧ uniform [A.y, B.y, C.y, D.y]) →
    (probability (positive_slope A B ∧ positive_slope A C ∧ positive_slope A D ∧ 
                  positive_slope B C ∧ positive_slope B D ∧ positive_slope C D)) = (1 / 24) :=
by sorry

end probability_positive_slopes_l135_135255


namespace letter_ratio_l135_135652

theorem letter_ratio (G B M : ℕ) (h1 : G = B + 10) 
                     (h2 : B = 40) 
                     (h3 : G + B + M = 270) : 
                     M / (G + B) = 2 := 
by 
  sorry

end letter_ratio_l135_135652


namespace find_g5_l135_135851

def g : ℤ → ℤ := sorry

axiom g1 : g 1 > 1
axiom g2 : ∀ x y : ℤ, g (x + y) + x * g y + y * g x = g x * g y + x + y + x * y
axiom g3 : ∀ x : ℤ, 3 * g x = g (x + 1) + 2 * x - 1

theorem find_g5 : g 5 = 248 :=
sorry

end find_g5_l135_135851


namespace campers_afternoon_l135_135061

theorem campers_afternoon (x : ℕ) 
  (h1 : 44 = x + 5) : 
  x = 39 := 
by
  sorry

end campers_afternoon_l135_135061


namespace find_g5_l135_135861

noncomputable def g (x : ℤ) : ℤ := sorry

axiom condition1 : g 1 > 1
axiom condition2 : ∀ x y : ℤ, g (x + y) + x * g y + y * g x = g x * g y + x + y + x * y
axiom condition3 : ∀ x : ℤ, 3 * g x = g (x + 1) + 2 * x - 1

theorem find_g5 : g 5 = 248 := 
by
  sorry

end find_g5_l135_135861


namespace jason_books_is_21_l135_135003

def keith_books : ℕ := 20
def total_books : ℕ := 41

theorem jason_books_is_21 (jason_books : ℕ) : 
  jason_books + keith_books = total_books → 
  jason_books = 21 := 
by 
  intro h
  sorry

end jason_books_is_21_l135_135003


namespace particle_max_height_and_time_l135_135601

theorem particle_max_height_and_time (t : ℝ) (s : ℝ) 
  (height_eq : s = 180 * t - 18 * t^2) :
  ∃ t₁ : ℝ, ∃ s₁ : ℝ, s₁ = 450 ∧ t₁ = 5 ∧ s = 180 * t₁ - 18 * t₁^2 :=
sorry

end particle_max_height_and_time_l135_135601


namespace complex_arithmetic_problem_l135_135191
open Complex

theorem complex_arithmetic_problem : (2 - 3 * Complex.I) * (2 + 3 * Complex.I) + (4 - 5 * Complex.I)^2 = 4 - 40 * Complex.I := by
  sorry

end complex_arithmetic_problem_l135_135191


namespace find_g5_l135_135859

noncomputable def g (x : ℤ) : ℤ := sorry

axiom condition1 : g 1 > 1
axiom condition2 : ∀ x y : ℤ, g (x + y) + x * g y + y * g x = g x * g y + x + y + x * y
axiom condition3 : ∀ x : ℤ, 3 * g x = g (x + 1) + 2 * x - 1

theorem find_g5 : g 5 = 248 := 
by
  sorry

end find_g5_l135_135859


namespace smallest_n_for_roots_l135_135718

noncomputable def smallest_n_roots_of_unity (f : ℂ → ℂ) (n : ℕ) : Prop :=
  ∀ z, f z = 0 → ∃ k, z = exp (2 * Real.pi * Complex.I * k / n)

theorem smallest_n_for_roots (n : ℕ) : smallest_n_roots_of_unity (λ z, z^6 - z^3 + 1) n ↔ n = 9 := 
by
  sorry

end smallest_n_for_roots_l135_135718


namespace greatest_possible_remainder_l135_135651

theorem greatest_possible_remainder (x : ℕ) : ∃ r : ℕ, r < 12 ∧ r ≠ 0 ∧ x % 12 = r ∧ r = 11 :=
by 
  sorry

end greatest_possible_remainder_l135_135651


namespace points_per_game_without_bonus_l135_135132

-- Definition of the conditions
def b : ℕ := 82
def n : ℕ := 79
def P : ℕ := 15089

-- Theorem statement
theorem points_per_game_without_bonus :
  (P - b * n) / n = 109 :=
by
  -- Proof will be filled in here
  sorry

end points_per_game_without_bonus_l135_135132


namespace three_children_meet_l135_135979

theorem three_children_meet 
  (children : Finset ℕ)
  (visited_times : ℕ → ℕ)
  (meet_at_stand : ℕ → ℕ → Prop)
  (h_children_count : children.card = 7)
  (h_visited_times : ∀ c ∈ children, visited_times c = 3)
  (h_meet_pairwise : ∀ (c1 c2 : ℕ), c1 ∈ children → c2 ∈ children → c1 ≠ c2 → meet_at_stand c1 c2) :
  ∃ (t : ℕ), ∃ (c1 c2 c3 : ℕ), c1 ≠ c2 ∧ c2 ≠ c3 ∧ c1 ≠ c3 ∧ 
  c1 ∈ children ∧ c2 ∈ children ∧ c3 ∈ children ∧ 
  meet_at_stand c1 t ∧ meet_at_stand c2 t ∧ meet_at_stand c3 t := 
sorry

end three_children_meet_l135_135979


namespace establish_model_steps_correct_l135_135148

-- Define each step as a unique identifier
inductive Step : Type
| observe_pose_questions
| propose_assumptions
| express_properties
| test_or_revise

open Step

-- The sequence of steps to establish a mathematical model for population change
def correct_model_steps : List Step :=
  [observe_pose_questions, propose_assumptions, express_properties, test_or_revise]

-- The correct answer is the sequence of steps in the correct order
theorem establish_model_steps_correct :
  correct_model_steps = [observe_pose_questions, propose_assumptions, express_properties, test_or_revise] :=
  by sorry

end establish_model_steps_correct_l135_135148


namespace solve_equation_l135_135217

theorem solve_equation :
  ∀ x y : ℝ, (x + y)^2 = (x + 1) * (y - 1) → x = -1 ∧ y = 1 :=
by
  intro x y
  sorry

end solve_equation_l135_135217


namespace total_sections_formed_l135_135704

theorem total_sections_formed (boys girls : ℕ) (hb : boys = 408) (hg : girls = 264) :
  let gcd := Nat.gcd boys girls
  let boys_sections := boys / gcd
  let girls_sections := girls / gcd
  boys_sections + girls_sections = 28 := 
by
  -- Note: this will assert the theorem, but the proof is omitted with sorry.
  sorry

end total_sections_formed_l135_135704


namespace polynomial_min_value_l135_135166

theorem polynomial_min_value (x : ℝ) : x = -3 → x^2 + 6 * x + 10 = 1 :=
by
  intro h
  sorry

end polynomial_min_value_l135_135166


namespace perimeter_C_correct_l135_135301

variables (x y : ℕ)

def perimeter_A (x y : ℕ) := 6 * x + 2 * y
def perimeter_B (x y : ℕ) := 4 * x + 6 * y
def perimeter_C (x y : ℕ) := 2 * x + 6 * y

theorem perimeter_C_correct (x y : ℕ) (h1 : 6 * x + 2 * y = 56) (h2 : 4 * x + 6 * y = 56) :
  2 * x + 6 * y = 40 :=
sorry

end perimeter_C_correct_l135_135301


namespace delta_five_three_l135_135086

def Δ (a b : ℕ) : ℕ := 4 * a - 6 * b

theorem delta_five_three :
  Δ 5 3 = 2 := by
  sorry

end delta_five_three_l135_135086


namespace age_of_15th_student_l135_135056

noncomputable def average_age_15_students := 15
noncomputable def average_age_7_students_1 := 14
noncomputable def average_age_7_students_2 := 16
noncomputable def total_students := 15
noncomputable def group_students := 7

theorem age_of_15th_student :
  let total_age_15_students := total_students * average_age_15_students
  let total_age_7_students_1 := group_students * average_age_7_students_1
  let total_age_7_students_2 := group_students * average_age_7_students_2
  let total_age_14_students := total_age_7_students_1 + total_age_7_students_2
  let age_15th_student := total_age_15_students - total_age_14_students
  age_15th_student = 15 :=
by
  sorry

end age_of_15th_student_l135_135056


namespace mean_value_of_quadrilateral_interior_angles_l135_135465

theorem mean_value_of_quadrilateral_interior_angles (a b c d : ℝ) 
  (h_sum : a + b + c + d = 360) : 
  (a + b + c + d) / 4 = 90 :=
by
  sorry

end mean_value_of_quadrilateral_interior_angles_l135_135465


namespace mean_value_of_quadrilateral_angles_l135_135436

theorem mean_value_of_quadrilateral_angles 
  (a1 a2 a3 a4 : ℝ) 
  (h_sum : a1 + a2 + a3 + a4 = 360) :
  (a1 + a2 + a3 + a4) / 4 = 90 :=
by
  sorry

end mean_value_of_quadrilateral_angles_l135_135436


namespace probability_real_l135_135983
 
noncomputable def problem : ℚ × ℚ → Prop
| (a, b) => (cos (a * real.pi) + complex.I * sin (b * real.pi)) ^ 2 = 0

def conditions (a b : ℚ) : Prop :=
  0 ≤ a ∧ a < 3 ∧ 0 ≤ b ∧ b < 3 ∧ ∃ n d : ℤ, n * a = d ∧ n * b = d ∧ 1 ≤ d ∧ d ≤ 10

theorem probability_real (a b : ℚ) (h : conditions a b) :
  ∃ p : ℚ, p = 1 / 10 :=
sorry

end probability_real_l135_135983


namespace cubic_eq_one_real_root_l135_135981

/-- The equation x^3 - 4x^2 + 9x + c = 0 has exactly one real root for any real number c. -/
theorem cubic_eq_one_real_root (c : ℝ) : 
  ∃! x : ℝ, x^3 - 4 * x^2 + 9 * x + c = 0 :=
sorry

end cubic_eq_one_real_root_l135_135981


namespace smallest_integer_in_odd_set_l135_135152

theorem smallest_integer_in_odd_set (is_odd: ℤ → Prop)
  (median: ℤ) (greatest: ℤ) (smallest: ℤ) 
  (h1: median = 126)
  (h2: greatest = 153) 
  (h3: ∀ x, is_odd x ↔ ∃ k: ℤ, x = 2*k + 1)
  (h4: ∀ a b c, median = (a+b) / 2 → c = a → a ≤ b)
  : 
  smallest = 100 :=
sorry

end smallest_integer_in_odd_set_l135_135152


namespace perimeter_of_C_l135_135297

theorem perimeter_of_C (x y : ℝ) 
  (h₁ : 6 * x + 2 * y = 56) 
  (h₂ : 4 * x + 6 * y = 56) : 
  2 * x + 6 * y = 40 :=
sorry

end perimeter_of_C_l135_135297


namespace system_solution_l135_135028

theorem system_solution (x y z : ℝ) :
    x + y + z = 2 ∧ 
    x^2 + y^2 + z^2 = 26 ∧
    x^3 + y^3 + z^3 = 38 →
    (x = 1 ∧ y = 4 ∧ z = -3) ∨
    (x = 1 ∧ y = -3 ∧ z = 4) ∨
    (x = 4 ∧ y = 1 ∧ z = -3) ∨
    (x = 4 ∧ y = -3 ∧ z = 1) ∨
    (x = -3 ∧ y = 1 ∧ z = 4) ∨
    (x = -3 ∧ y = 4 ∧ z = 1) := by
  sorry

end system_solution_l135_135028


namespace no_integer_solutions_l135_135207

theorem no_integer_solutions (x y z : ℤ) (h1 : x > y) (h2 : y > z) : 
  x * (x - y) + y * (y - z) + z * (z - x) ≠ 3 := 
by
  sorry

end no_integer_solutions_l135_135207


namespace eve_walked_distance_l135_135629

-- Defining the distances Eve ran and walked
def distance_ran : ℝ := 0.7
def distance_walked : ℝ := distance_ran - 0.1

-- Proving that the distance Eve walked is 0.6 mile
theorem eve_walked_distance : distance_walked = 0.6 := by
  -- The proof is omitted.
  sorry

end eve_walked_distance_l135_135629


namespace mean_value_of_quadrilateral_angles_l135_135425

theorem mean_value_of_quadrilateral_angles : 
  ∀ (a b c d : ℝ), a + b + c + d = 360 → (a + b + c + d) / 4 = 90 :=
by
  intro a b c d h
  sorry

end mean_value_of_quadrilateral_angles_l135_135425


namespace cube_sum_l135_135655

theorem cube_sum (x y : ℝ) (h1 : x * y = 15) (h2 : x + y = 11) : x^3 + y^3 = 836 := 
by
  sorry

end cube_sum_l135_135655


namespace sum_union_eq_34_l135_135776

open Finset

def A : Finset ℕ := {2, 0, 1, 9}
def B : Finset ℕ := { x | ∃ a ∈ A, x = 2 * a }

noncomputable def union_sum : ℕ := (A ∪ B).sum id

theorem sum_union_eq_34 : union_sum = 34 := by
  sorry

end sum_union_eq_34_l135_135776


namespace mean_value_of_quadrilateral_angles_l135_135388

theorem mean_value_of_quadrilateral_angles (a b c d : ℝ) (h_angles_sum : a + b + c + d = 360) : (a + b + c + d) / 4 = 90 := 
by
  sorry

end mean_value_of_quadrilateral_angles_l135_135388


namespace find_a_l135_135236

theorem find_a (a x : ℝ) (h : x = 3) (eqn : a * x + 4 = 1) : a = -1 :=
by
  -- Placeholder to indicate where the proof would go
  sorry

end find_a_l135_135236


namespace find_cost_price_l135_135178

theorem find_cost_price (C : ℝ) (SP : ℝ) (M : ℝ) (h1 : SP = 1.25 * C) (h2 : 0.90 * M = SP) (h3 : SP = 65.97) : 
  C = 52.776 :=
by
  sorry

end find_cost_price_l135_135178


namespace mean_of_quadrilateral_angles_l135_135380

theorem mean_of_quadrilateral_angles : 
  (∀ (a b c d : ℝ), a + b + c + d = 360 → (a + b + c + d) / 4 = 90) :=
begin
  intros a b c d h,
  calc
    (a + b + c + d) / 4 = 360 / 4 : by rw h
                    ... = 90 : by norm_num,
end

end mean_of_quadrilateral_angles_l135_135380


namespace estimate_fish_population_l135_135931

theorem estimate_fish_population :
  ∀ (x : ℕ), (1200 / x = 100 / 1000) → x = 12000 := by
  sorry

end estimate_fish_population_l135_135931


namespace expression_eval_l135_135193

theorem expression_eval : (3^2 - 3) - (5^2 - 5) * 2 + (6^2 - 6) = -4 :=
by sorry

end expression_eval_l135_135193


namespace regional_frisbee_committees_l135_135673

theorem regional_frisbee_committees : 
  let teams_count := 5
  let members_per_team := 8
  let captain_count := 1
  let host_selection := choose (members_per_team - captain_count) 3
  let non_host_selection := choose members_per_team 3
  teams_count * (host_selection * non_host_selection ^ (teams_count - 1)) = 1723286800 := by
    sorry

end regional_frisbee_committees_l135_135673


namespace solve_for_x_l135_135585

theorem solve_for_x (x y : ℤ) (h1 : x + 2 * y = 10) (h2 : y = 3) : x = 4 := by
  sorry

end solve_for_x_l135_135585


namespace solve_equation_l135_135987

theorem solve_equation (x : ℝ) (h : (x - 7) / 2 - (1 + x) / 3 = 1) : x = 29 :=
sorry

end solve_equation_l135_135987


namespace order_A_C_B_l135_135516

noncomputable def A (a b : ℝ) : ℝ := Real.log ((a + b) / 2)
noncomputable def B (a b : ℝ) : ℝ := Real.sqrt (Real.log a * Real.log b)
noncomputable def C (a b : ℝ) : ℝ := (Real.log a + Real.log b) / 2

theorem order_A_C_B (a b : ℝ) (h1 : 1 < b) (h2 : b < a) :
  A a b > C a b ∧ C a b > B a b :=
by 
  sorry

end order_A_C_B_l135_135516


namespace billy_laundry_loads_l135_135081

-- Define constants based on problem conditions
def sweeping_minutes_per_room := 3
def washing_minutes_per_dish := 2
def laundry_minutes_per_load := 9

def anna_rooms := 10
def billy_dishes := 6

-- Calculate total time spent by Anna and the time Billy spends washing dishes
def anna_total_time := sweeping_minutes_per_room * anna_rooms
def billy_dishwashing_time := washing_minutes_per_dish * billy_dishes

-- Define the time difference Billy needs to make up with laundry
def time_difference := anna_total_time - billy_dishwashing_time
def billy_required_laundry_loads := time_difference / laundry_minutes_per_load

-- The theorem to prove
theorem billy_laundry_loads : billy_required_laundry_loads = 2 := by 
  sorry

end billy_laundry_loads_l135_135081


namespace perimeter_C_correct_l135_135299

variables (x y : ℕ)

def perimeter_A (x y : ℕ) := 6 * x + 2 * y
def perimeter_B (x y : ℕ) := 4 * x + 6 * y
def perimeter_C (x y : ℕ) := 2 * x + 6 * y

theorem perimeter_C_correct (x y : ℕ) (h1 : 6 * x + 2 * y = 56) (h2 : 4 * x + 6 * y = 56) :
  2 * x + 6 * y = 40 :=
sorry

end perimeter_C_correct_l135_135299


namespace mean_value_of_quadrilateral_angles_l135_135390

theorem mean_value_of_quadrilateral_angles (a b c d : ℝ) (h_angles_sum : a + b + c + d = 360) : (a + b + c + d) / 4 = 90 := 
by
  sorry

end mean_value_of_quadrilateral_angles_l135_135390


namespace frame_dimension_ratio_l135_135738

theorem frame_dimension_ratio (W H x : ℕ) (h1 : W = 20) (h2 : H = 30) (h3 : 2 * (W + 2 * x) * (H + 6 * x) - W * H = 2 * (W * H)) :
  (W + 2 * x) / (H + 6 * x) = 1/2 :=
by sorry

end frame_dimension_ratio_l135_135738


namespace compute_fraction_power_mul_l135_135911

theorem compute_fraction_power_mul : ((1 / 3: ℚ) ^ 4) * (1 / 5) = (1 / 405) := by
  -- proof goes here
  sorry

end compute_fraction_power_mul_l135_135911


namespace common_ratio_geometric_sequence_l135_135736

theorem common_ratio_geometric_sequence (a b c d : ℤ) (h1 : a = 10) (h2 : b = -20) (h3 : c = 40) (h4 : d = -80) :
    b / a = -2 ∧ c = b * -2 ∧ d = c * -2 := by
  sorry

end common_ratio_geometric_sequence_l135_135736


namespace mean_value_of_quadrilateral_angles_l135_135391

theorem mean_value_of_quadrilateral_angles (a b c d : ℝ) (h_angles_sum : a + b + c + d = 360) : (a + b + c + d) / 4 = 90 := 
by
  sorry

end mean_value_of_quadrilateral_angles_l135_135391


namespace original_length_of_tape_l135_135617

-- Given conditions
variables (L : Real) (used_by_Remaining_yesterday : L * (1 - 1 / 5) = 4 / 5 * L)
          (remaining_after_today : 1.5 = 4 / 5 * L * 1 / 4)

-- The theorem to prove
theorem original_length_of_tape (L : Real) 
  (used_by_Remaining_yesterday : L * (1 - 1 / 5) = 4 / 5 * L)
  (remaining_after_today : 1.5 = 4 / 5 * L * 1 / 4) :
  L = 7.5 :=
by
  sorry

end original_length_of_tape_l135_135617


namespace target_hit_probability_l135_135895

-- Defining the probabilities for A, B, and C hitting the target.
def P_A_hit := 1 / 2
def P_B_hit := 1 / 3
def P_C_hit := 1 / 4

-- Defining the probability that A, B, and C miss the target.
def P_A_miss := 1 - P_A_hit
def P_B_miss := 1 - P_B_hit
def P_C_miss := 1 - P_C_hit

-- Calculating the combined probability that none of them hit the target.
def P_none_hit := P_A_miss * P_B_miss * P_C_miss

-- Now, calculating the probability that at least one of them hits the target.
def P_hit := 1 - P_none_hit

-- Statement of the theorem.
theorem target_hit_probability : P_hit = 3 / 4 := by
  sorry

end target_hit_probability_l135_135895


namespace second_player_wins_l135_135871

open Nat

noncomputable def game_piles := List Nat -- a list representing piles with one nut each.

def relatively_prime (a b : Nat) : Prop :=
  gcd a b = 1

def valid_move (state : game_piles) (p1 p2 : Nat) : Prop :=
  p1 ∈ state ∧ p2 ∈ state ∧ relatively_prime p1 p2

def new_state (state : game_piles) (p1 p2 : Nat) : game_piles :=
  (state.erase p1).erase p2 ++ [(p1 + p2)] -- combine piles

theorem second_player_wins (N : Nat) (hN : N > 2) :
  ∃ strategy : (game_piles → game_piles → Prop), ∀ init_state, game_piles.length init_state = N → 
    strategy init_state (new_state init_state + 1 1) →
    (strategy ∷ second_player_wins N (N - 1) init_state ∧ valid_move init_state 
    sorry

end second_player_wins_l135_135871


namespace min_colors_needed_l135_135482

theorem min_colors_needed (n : ℕ) : 
  (n + (n * (n - 1)) / 2 ≥ 12) → (n = 5) :=
by
  sorry

end min_colors_needed_l135_135482


namespace petya_run_12_seconds_l135_135687

-- Define the conditions
variable (petya_speed classmates_speed : ℕ → ℕ) -- speeds of Petya and his classmates
variable (total_distance : ℕ := 100) -- each participant needs to run 100 meters
variable (initial_total_distance_run : ℕ := 288) -- total distance run by all in the first 12 seconds
variable (remaining_distance_when_petya_finished : ℕ := 40) -- remaining distance for others when Petya finished
variable (time_to_first_finish : ℕ) -- the time Petya takes to finish the race

-- Assume constant speeds for all participants
axiom constant_speed_petya (t : ℕ) : petya_speed t = petya_speed 0
axiom constant_speed_classmates (t : ℕ) : classmates_speed t = classmates_speed 0

-- Summarized total distances run by participants
axiom total_distance_run_all (t : ℕ) :
  petya_speed t * t + classmates_speed t * t = initial_total_distance_run + remaining_distance_when_petya_finished + (total_distance - remaining_distance_when_petya_finished) * 3

-- Given conditions converted to Lean
axiom initial_distance_run (t : ℕ) :
  t = 12 → petya_speed t * t + classmates_speed t * t = initial_total_distance_run

axiom petya_completion (t : ℕ) :
  t = time_to_first_finish → petya_speed t * t = total_distance

axiom remaining_distance_classmates (t : ℕ) :
  t = time_to_first_finish → classmates_speed t * (t - time_to_first_finish) = remaining_distance_when_petya_finished
  
-- Define the proof goal using the conditions
theorem petya_run_12_seconds (d : ℕ) :
  (∃ t, t = 12 ∧ d = petya_speed t * t) → d = 80 :=
by
  sorry

end petya_run_12_seconds_l135_135687


namespace complement_intersection_l135_135131

-- Definitions
def U : Set ℕ := {x | x ≤ 4 ∧ 0 < x}
def A : Set ℕ := {1, 4}
def B : Set ℕ := {2, 4}
def complement (s : Set ℕ) := {x | x ∈ U ∧ x ∉ s}

-- The theorem to prove
theorem complement_intersection :
  complement (A ∩ B) = {1, 2, 3} :=
by
  sorry

end complement_intersection_l135_135131


namespace perimeter_of_C_l135_135295

theorem perimeter_of_C (x y : ℝ) 
  (h₁ : 6 * x + 2 * y = 56) 
  (h₂ : 4 * x + 6 * y = 56) : 
  2 * x + 6 * y = 40 :=
sorry

end perimeter_of_C_l135_135295


namespace compute_expression_l135_135918

-- Definition of the operation "minus the reciprocal of"
def diamond (a b : ℚ) : ℚ := a - 1 / b

-- Theorem statement to prove the given problem
theorem compute_expression :
  ((diamond (diamond 3 4) 5) - (diamond 3 (diamond 4 5))) = -71 / 380 := 
sorry

end compute_expression_l135_135918


namespace ratio_of_ages_l135_135319

-- Define the conditions and the main proof goal
theorem ratio_of_ages (R J : ℕ) (Tim_age : ℕ) (h1 : Tim_age = 5) (h2 : J = R + 2) (h3 : J = Tim_age + 12) :
  R / Tim_age = 3 := 
by
  sorry

end ratio_of_ages_l135_135319


namespace perimeter_C_l135_135293

theorem perimeter_C :
  ∀ (x y : ℕ),
  (6 * x + 2 * y = 56) →
  (4 * x + 6 * y = 56) →
  (2 * x + 6 * y = 40) :=
by
  intros x y hA hB
  sorry

end perimeter_C_l135_135293


namespace denmark_pizza_combinations_l135_135759

theorem denmark_pizza_combinations :
  (let cheese_options := 3
   let meat_options := 4
   let vegetable_options := 5
   let invalid_combinations := 1
   let total_combinations := cheese_options * meat_options * vegetable_options
   let valid_combinations := total_combinations - invalid_combinations
   valid_combinations = 59) :=
by
  sorry

end denmark_pizza_combinations_l135_135759


namespace cost_of_shorts_l135_135799

-- Define the given conditions and quantities
def initial_money : ℕ := 50
def jerseys_cost : ℕ := 5 * 2
def basketball_cost : ℕ := 18
def remaining_money : ℕ := 14

-- The total amount spent
def total_spent : ℕ := initial_money - remaining_money

-- The total cost of the jerseys and basketball
def jerseys_basketball_cost : ℕ := jerseys_cost + basketball_cost

-- The cost of the shorts
def shorts_cost : ℕ := total_spent - jerseys_basketball_cost

theorem cost_of_shorts : shorts_cost = 8 := sorry

end cost_of_shorts_l135_135799


namespace mushroom_mistake_l135_135143

theorem mushroom_mistake (p k v : ℝ) (hk : k = p + v - 10) (hp : p = k + v - 7) : 
  ∃ p k : ℝ, ∀ v : ℝ, (p = k + v - 7) ∧ (k = p + v - 10) → false :=
by
  sorry

end mushroom_mistake_l135_135143


namespace solution_set_a_eq_2_solution_set_a_eq_neg_2_solution_set_neg_2_lt_a_lt_2_solution_set_a_lt_neg_2_or_a_gt_2_l135_135272

def solve_inequality (a x : ℝ) : Prop :=
  a^2 * x - 6 < 4 * x + 3 * a

theorem solution_set_a_eq_2 :
  ∀ x : ℝ, solve_inequality 2 x ↔ true :=
sorry

theorem solution_set_a_eq_neg_2 :
  ∀ x : ℝ, ¬ solve_inequality (-2) x :=
sorry

theorem solution_set_neg_2_lt_a_lt_2 (a : ℝ) (h : -2 < a ∧ a < 2) :
  ∀ x : ℝ, solve_inequality a x ↔ x > 3 / (a - 2) :=
sorry

theorem solution_set_a_lt_neg_2_or_a_gt_2 (a : ℝ) (h : a < -2 ∨ a > 2) :
  ∀ x : ℝ, solve_inequality a x ↔ x < 3 / (a - 2) :=
sorry

end solution_set_a_eq_2_solution_set_a_eq_neg_2_solution_set_neg_2_lt_a_lt_2_solution_set_a_lt_neg_2_or_a_gt_2_l135_135272


namespace mean_of_quadrilateral_angles_is_90_l135_135453

theorem mean_of_quadrilateral_angles_is_90 :
  let n := 4 in
  let sum_of_interior_angles (n : ℕ) := (n - 2) * 180 in
  let quadrilateral_interior_angle_sum := sum_of_interior_angles 4 in
  quadrilateral_interior_angle_sum = 360 →
  360 / 4 = 90 :=
by
  intros n sum_of_interior_angles quadrilateral_interior_angle_sum h
  exact sorry

end mean_of_quadrilateral_angles_is_90_l135_135453


namespace cube_inequality_of_greater_l135_135506

theorem cube_inequality_of_greater {a b : ℝ} (h : a > b) : a^3 > b^3 :=
sorry

end cube_inequality_of_greater_l135_135506


namespace john_ate_cookies_l135_135676

-- Definitions for conditions
def dozen := 12

-- Given conditions
def initial_cookies : ℕ := 2 * dozen
def cookies_left : ℕ := 21

-- Problem statement
theorem john_ate_cookies : initial_cookies - cookies_left = 3 :=
by
  -- Solution steps omitted, only statement provided
  sorry

end john_ate_cookies_l135_135676


namespace mean_value_of_interior_angles_of_quadrilateral_l135_135363

theorem mean_value_of_interior_angles_of_quadrilateral :
  (360 / 4) = 90 := 
by
  sorry

end mean_value_of_interior_angles_of_quadrilateral_l135_135363


namespace gcd_of_lcm_l135_135619

noncomputable def gcd (A B C : ℕ) : ℕ := Nat.gcd (Nat.gcd A B) C
noncomputable def lcm (A B C : ℕ) : ℕ := Nat.lcm (Nat.lcm A B) C

theorem gcd_of_lcm (A B C : ℕ) (LCM_ABC : ℕ) (Product_ABC : ℕ) :
  lcm A B C = LCM_ABC →
  A * B * C = Product_ABC →
  gcd A B C = 20 :=
by
  intros lcm_eq product_eq
  sorry

end gcd_of_lcm_l135_135619


namespace correct_factorization_l135_135884

-- Define the polynomial expressions
def polyA (x : ℝ) := x^3 - x
def factorA1 (x : ℝ) := x * (x^2 - 1)
def factorA2 (x : ℝ) := x * (x + 1) * (x - 1)

def polyB (a : ℝ) := 4 * a^2 - 4 * a + 1
def factorB (a : ℝ) := 4 * a * (a - 1) + 1

def polyC (x y : ℝ) := x^2 + y^2
def factorC (x y : ℝ) := (x + y)^2

def polyD (x : ℝ) := -3 * x + 6 * x^2 - 3 * x^3
def factorD (x : ℝ) := -3 * x * (x - 1)^2

-- Statement of the correctness of factorization D
theorem correct_factorization : ∀ (x : ℝ), polyD x = factorD x :=
by
  intro x
  sorry

end correct_factorization_l135_135884


namespace sets_of_earrings_l135_135263

namespace EarringsProblem

variables (magnets buttons gemstones earrings : ℕ)

theorem sets_of_earrings (h1 : gemstones = 24)
                         (h2 : gemstones = 3 * buttons)
                         (h3 : buttons = magnets / 2)
                         (h4 : earrings = magnets / 2)
                         (h5 : ∀ n : ℕ, n % 2 = 0 → ∃ k, n = 2 * k) :
  earrings = 8 :=
by
  sorry

end EarringsProblem

end sets_of_earrings_l135_135263


namespace prop_false_iff_a_lt_neg_13_over_2_l135_135948

theorem prop_false_iff_a_lt_neg_13_over_2 :
  (¬ ∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → x^2 + a * x + 9 ≥ 0) ↔ a < -13 / 2 := 
sorry

end prop_false_iff_a_lt_neg_13_over_2_l135_135948


namespace find_g5_l135_135855

-- Define the function g on integers
def g : ℤ → ℤ := sorry

-- Define the conditions
axiom cond1 : g 1 > 1
axiom cond2 : ∀ (x y : ℤ), g (x + y) + x * g y + y * g x = g x * g y + x + y + x * y
axiom cond3 : ∀ (x : ℤ), 3 * g x = g (x + 1) + 2 * x - 1

-- The statement we need to prove
theorem find_g5 : g 5 = 248 := by
  sorry

end find_g5_l135_135855


namespace operation_value_l135_135206

def operation (a b : ℤ) : ℤ := 3 * a - 3 * b + 4

theorem operation_value : operation 6 8 = -2 := by
  sorry

end operation_value_l135_135206


namespace seahawks_touchdowns_l135_135159

theorem seahawks_touchdowns (total_points : ℕ) (points_per_touchdown : ℕ) (points_per_field_goal : ℕ) (field_goals : ℕ) (touchdowns : ℕ) :
  total_points = 37 →
  points_per_touchdown = 7 →
  points_per_field_goal = 3 →
  field_goals = 3 →
  total_points = (touchdowns * points_per_touchdown) + (field_goals * points_per_field_goal) →
  touchdowns = 4 :=
by
  intros h_total_points h_points_per_touchdown h_points_per_field_goal h_field_goals h_equation
  sorry

end seahawks_touchdowns_l135_135159


namespace find_g5_l135_135862

noncomputable def g (x : ℤ) : ℤ := sorry

axiom condition1 : g 1 > 1
axiom condition2 : ∀ x y : ℤ, g (x + y) + x * g y + y * g x = g x * g y + x + y + x * y
axiom condition3 : ∀ x : ℤ, 3 * g x = g (x + 1) + 2 * x - 1

theorem find_g5 : g 5 = 248 := 
by
  sorry

end find_g5_l135_135862


namespace mean_value_of_quadrilateral_angles_l135_135334

theorem mean_value_of_quadrilateral_angles : 
  (∀ (A B C D : ℕ), A + B + C + D = 360 → (A + B + C + D) / 4 = 90) :=
by {
  sorry
}

end mean_value_of_quadrilateral_angles_l135_135334


namespace g_5_is_248_l135_135847

def g : ℤ → ℤ := sorry

theorem g_5_is_248 :
  (g(1) > 1) ∧
  (∀ x y : ℤ, g(x + y) + x * g(y) + y * g(x) = g(x) * g(y) + x + y + x * y) ∧
  (∀ x : ℤ, 3 * g(x) = g(x + 1) + 2 * x - 1) →
  g(5) = 248 :=
by
  -- proof omitted
  sorry

end g_5_is_248_l135_135847


namespace cat_food_customers_l135_135705

/-
Problem: There was a big sale on cat food at the pet store. Some people bought cat food that day. The first 8 customers bought 3 cases each. The next four customers bought 2 cases each. The last 8 customers of the day only bought 1 case each. In total, 40 cases of cat food were sold. How many people bought cat food that day?
-/

theorem cat_food_customers:
  (8 * 3) + (4 * 2) + (8 * 1) = 40 →
  8 + 4 + 8 = 20 :=
by
  intro h
  linarith

end cat_food_customers_l135_135705


namespace compute_fraction_pow_mult_l135_135915

def frac_1_3 := (1 : ℝ) / (3 : ℝ)
def frac_1_5 := (1 : ℝ) / (5 : ℝ)
def target := (1 : ℝ) / (405 : ℝ)

theorem compute_fraction_pow_mult :
  (frac_1_3^4 * frac_1_5) = target :=
by
  sorry

end compute_fraction_pow_mult_l135_135915


namespace smallest_n_roots_of_unity_l135_135726

open Complex Polynomial

theorem smallest_n_roots_of_unity :
  (∀ z : ℂ, z^6 - z^3 + 1 = 0 → ∃ k : ℤ, z = exp(2 * Real.pi * Complex.I * k / 18)) ∧
  (∀ m : ℕ, (∀ z : ℂ, z^6 - z^3 + 1 = 0 → (∃ k : ℤ, z = exp(2 * Real.pi * Complex.I * k / m))) → 18 ≤ m) :=
sorry

end smallest_n_roots_of_unity_l135_135726


namespace mean_value_of_quadrilateral_angles_l135_135423

theorem mean_value_of_quadrilateral_angles : 
  ∀ (a b c d : ℝ), a + b + c + d = 360 → (a + b + c + d) / 4 = 90 :=
by
  intro a b c d h
  sorry

end mean_value_of_quadrilateral_angles_l135_135423


namespace seven_segments_impossible_l135_135675

theorem seven_segments_impossible :
  ¬(∃(segments : Fin 7 → Set (Fin 7)), (∀i, ∃ (S : Finset (Fin 7)), S.card = 3 ∧ ∀ j ∈ S, i ≠ j ∧ segments i j) ∧ (∀ i j, i ≠ j → segments i j → segments j i)) :=
sorry

end seven_segments_impossible_l135_135675


namespace peter_son_is_nikolay_l135_135548

variable (x y : ℕ)

/-- Within the stated scenarios of Nikolai/Peter paired fishes caught -/
theorem peter_son_is_nikolay :
  (∀ n p ns ps : ℕ, (
    n = ns ∧              -- Nikolai caught as many fish as his son
    p = 3 * ps ∧          -- Peter caught three times more fish than his son
    n + ns + p + ps = 25  -- A total of 25 fish were caught
  ) → ("Nikolay" = "Peter's son")) := 
sorry

end peter_son_is_nikolay_l135_135548


namespace mean_value_of_quadrilateral_angles_l135_135330

-- Statement of the problem: mean value of interior angles in any quadrilateral is 90°
theorem mean_value_of_quadrilateral_angles : 
  (∀ (a b c d : ℝ), a + b + c + d = 360 → (a + b + c + d) / 4 = 90) :=
by
  sorry

end mean_value_of_quadrilateral_angles_l135_135330


namespace mean_value_of_quadrilateral_interior_angles_l135_135461

theorem mean_value_of_quadrilateral_interior_angles (a b c d : ℝ) 
  (h_sum : a + b + c + d = 360) : 
  (a + b + c + d) / 4 = 90 :=
by
  sorry

end mean_value_of_quadrilateral_interior_angles_l135_135461


namespace xyz_value_l135_135645

theorem xyz_value (x y z : ℝ)
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 30)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 10) :
  x * y * z = 20 / 3 :=
by
  sorry

end xyz_value_l135_135645


namespace ratio_of_areas_of_concentric_circles_l135_135876

theorem ratio_of_areas_of_concentric_circles
  (C1 C2 : ℝ)
  (h : (30 / 360) * C1 = (24 / 360) * C2) :
  (π * (C1 / (2 * π))^2) / (π * (C2 / (2 * π))^2) = 16 / 25 := by
  sorry

end ratio_of_areas_of_concentric_circles_l135_135876


namespace find_constants_l135_135926

theorem find_constants (A B C : ℚ) :
  (∀ x : ℚ, (8 * x + 1) / ((x - 4) * (x - 2)^2) = A / (x - 4) + B / (x - 2) + C / (x - 2)^2) → 
  A = 33 / 4 ∧ B = -19 / 4 ∧ C = -17 / 2 :=
by 
  intro h
  sorry

end find_constants_l135_135926


namespace car_speed_conversion_l135_135868

theorem car_speed_conversion :
  let speed_mps := 10 -- speed of the car in meters per second
  let conversion_factor := 3.6 -- conversion factor from m/s to km/h
  let speed_kmph := speed_mps * conversion_factor -- speed of the car in kilometers per hour
  speed_kmph = 36 := 
by
  sorry

end car_speed_conversion_l135_135868


namespace pipe_fill_time_without_leak_l135_135184

theorem pipe_fill_time_without_leak (T : ℝ) (h1 : (1 / 9 : ℝ) = 1 / T - 1 / 4.5) : T = 3 := 
by
  sorry

end pipe_fill_time_without_leak_l135_135184


namespace mean_of_quadrilateral_angles_l135_135408

theorem mean_of_quadrilateral_angles : 
  (∀ (a b c d : ℝ), a + b + c + d = 360) → 90 :=
by
  intro h
  sorry

end mean_of_quadrilateral_angles_l135_135408


namespace find_center_radius_sum_l135_135093

theorem find_center_radius_sum :
    let x := x
    let y := y
    let a := 2
    let b := 3
    let r := 2 * Real.sqrt 6
    (x^2 - 4 * x + y^2 - 6 * y = 11) →
    (a + b + r = 5 + 2 * Real.sqrt 6) :=
by
  intros x y a b r
  sorry

end find_center_radius_sum_l135_135093


namespace mean_of_quadrilateral_angles_is_90_l135_135450

theorem mean_of_quadrilateral_angles_is_90 :
  let n := 4 in
  let sum_of_interior_angles (n : ℕ) := (n - 2) * 180 in
  let quadrilateral_interior_angle_sum := sum_of_interior_angles 4 in
  quadrilateral_interior_angle_sum = 360 →
  360 / 4 = 90 :=
by
  intros n sum_of_interior_angles quadrilateral_interior_angle_sum h
  exact sorry

end mean_of_quadrilateral_angles_is_90_l135_135450


namespace smallest_n_roots_of_unity_l135_135725

open Complex Polynomial

theorem smallest_n_roots_of_unity :
  (∀ z : ℂ, z^6 - z^3 + 1 = 0 → ∃ k : ℤ, z = exp(2 * Real.pi * Complex.I * k / 18)) ∧
  (∀ m : ℕ, (∀ z : ℂ, z^6 - z^3 + 1 = 0 → (∃ k : ℤ, z = exp(2 * Real.pi * Complex.I * k / m))) → 18 ≤ m) :=
sorry

end smallest_n_roots_of_unity_l135_135725


namespace interest_rate_l135_135998

noncomputable def compoundInterest (P : ℕ) (r : ℕ) (t : ℕ) : ℚ :=
  P * ((1 + r / 100 : ℚ) ^ t) - P

noncomputable def simpleInterest (P : ℕ) (r : ℕ) (t : ℕ) : ℚ :=
  P * r * t / 100

theorem interest_rate (P t : ℕ) (D : ℚ) (r : ℕ) :
  P = 10000 → t = 2 → D = 49 →
  compoundInterest P r t - simpleInterest P r t = D → r = 7 := by
  sorry

end interest_rate_l135_135998


namespace average_letters_per_day_l135_135688

theorem average_letters_per_day:
  let letters_per_day := [7, 10, 3, 5, 12]
  (letters_per_day.sum / letters_per_day.length : ℝ) = 7.4 :=
by
  sorry

end average_letters_per_day_l135_135688


namespace correct_mark_l135_135147

theorem correct_mark
  (n : ℕ)
  (initial_avg : ℝ)
  (wrong_mark : ℝ)
  (correct_avg : ℝ)
  (correct_total_marks : ℝ)
  (actual_total_marks : ℝ)
  (final_mark : ℝ) :
  n = 25 →
  initial_avg = 100 →
  wrong_mark = 60 →
  correct_avg = 98 →
  correct_total_marks = (n * correct_avg) →
  actual_total_marks = (n * initial_avg - wrong_mark + final_mark) →
  correct_total_marks = actual_total_marks →
  final_mark = 10 :=
by
  intros h_n h_initial_avg h_wrong_mark h_correct_avg h_correct_total_marks h_actual_total_marks h_eq
  sorry

end correct_mark_l135_135147


namespace sector_area_l135_135510

-- Define radius and central angle as conditions
def radius : ℝ := 1
def central_angle : ℝ := 2

-- Define the theorem to prove that the area of the sector is 1 cm² given the conditions
theorem sector_area : (1 / 2) * radius * central_angle = 1 := 
by 
  -- sorry is used to skip the actual proof
  sorry

end sector_area_l135_135510


namespace find_concentration_of_second_mixture_l135_135735

noncomputable def concentration_of_second_mixture (total_volume : ℝ) (final_percent : ℝ) (pure_antifreeze : ℝ) (pure_antifreeze_amount : ℝ) : ℝ :=
  let remaining_volume := total_volume - pure_antifreeze_amount
  let final_pure_amount := final_percent * total_volume
  let required_pure_antifreeze := final_pure_amount - pure_antifreeze
  (required_pure_antifreeze / remaining_volume) * 100

theorem find_concentration_of_second_mixture :
  concentration_of_second_mixture 55 0.20 6.11 6.11 = 10 :=
by
  simp [concentration_of_second_mixture]
  sorry

end find_concentration_of_second_mixture_l135_135735


namespace tan_phi_eq_sqrt3_l135_135636

theorem tan_phi_eq_sqrt3
  (φ : ℝ)
  (h1 : Real.cos (Real.pi / 2 - φ) = Real.sqrt 3 / 2)
  (h2 : abs φ < Real.pi / 2) :
  Real.tan φ = Real.sqrt 3 :=
sorry

end tan_phi_eq_sqrt3_l135_135636


namespace cylinder_base_ratio_l135_135966

variable (O : Point) -- origin
variable (a b c : ℝ) -- fixed point
variable (p q : ℝ) -- center of circular base
variable (α β : ℝ) -- intersection points with axis

-- Let O be the origin
-- Let (a, b, c) be the fixed point through which the cylinder passes
-- The cylinder's axis is parallel to the z-axis and the center of its base is (p, q)
-- The cylinder intersects the x-axis at (α, 0, 0) and the y-axis at (0, β, 0)
-- Let α = 2p and β = 2q

theorem cylinder_base_ratio : 
  α = 2 * p ∧ β = 2 * q → (a / p + b / q = 4) := by
  sorry

end cylinder_base_ratio_l135_135966


namespace christine_aquafaba_needed_l135_135195

-- Define the number of tablespoons per egg white
def tablespoons_per_egg_white : ℕ := 2

-- Define the number of egg whites per cake
def egg_whites_per_cake : ℕ := 8

-- Define the number of cakes
def number_of_cakes : ℕ := 2

-- Express the total amount of aquafaba needed
def aquafaba_needed : ℕ :=
  tablespoons_per_egg_white * egg_whites_per_cake * number_of_cakes

-- Statement asserting the amount of aquafaba needed is 32
theorem christine_aquafaba_needed : aquafaba_needed = 32 := by
  sorry

end christine_aquafaba_needed_l135_135195


namespace christine_needs_32_tablespoons_l135_135201

-- Define the conditions
def tablespoons_per_egg_white : ℕ := 2
def egg_whites_per_cake : ℕ := 8
def number_of_cakes : ℕ := 2

-- Define the calculation for total tablespoons of aquafaba needed
def total_tbs_aquafaba : ℕ :=
  tablespoons_per_egg_white * (egg_whites_per_cake * number_of_cakes)

-- The theorem to prove
theorem christine_needs_32_tablespoons :
  total_tbs_aquafaba = 32 :=
by 
  -- Placeholder for proof, as proof steps are not required
  sorry

end christine_needs_32_tablespoons_l135_135201


namespace mean_value_of_quadrilateral_angles_l135_135326

-- Statement of the problem: mean value of interior angles in any quadrilateral is 90°
theorem mean_value_of_quadrilateral_angles : 
  (∀ (a b c d : ℝ), a + b + c + d = 360 → (a + b + c + d) / 4 = 90) :=
by
  sorry

end mean_value_of_quadrilateral_angles_l135_135326


namespace total_height_increase_l135_135151

def height_increase_per_decade : ℕ := 90
def decades_in_two_centuries : ℕ := (2 * 100) / 10

theorem total_height_increase :
  height_increase_per_decade * decades_in_two_centuries = 1800 := by
  sorry

end total_height_increase_l135_135151


namespace valid_number_count_eq_102_l135_135620

open Nat
open Finset

def is_valid_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 3000 ∧
  let d := n % 10
  in d = (n / 10) % 10 + (n / 100) % 10 + (n / 1000)

noncomputable def count_valid_numbers : ℕ := 
  (Finset.range 3000).filter is_valid_number |>.card

theorem valid_number_count_eq_102 :
  count_valid_numbers = 102 :=
by 
  sorry -- The detailed proof goes here.

end valid_number_count_eq_102_l135_135620


namespace total_weight_new_group_l135_135693

variable (W : ℝ) -- Total weight of the original group of 20 people
variable (weights_old : List ℝ) 
variable (weights_new : List ℝ)

-- Given conditions
def five_weights_old : List ℝ := [40, 55, 60, 75, 80]
def average_weight_increase : ℝ := 2
def group_size : ℕ := 20
def num_replaced : ℕ := 5

-- Define theorem
theorem total_weight_new_group :
(W - five_weights_old.sum + group_size * average_weight_increase) -
(W - five_weights_old.sum) = weights_new.sum → 
weights_new.sum = 350 := 
by
  sorry

end total_weight_new_group_l135_135693


namespace avg_diff_noah_liam_l135_135036

-- Define the daily differences over 14 days
def daily_differences : List ℤ := [5, 0, 15, -5, 10, 10, -10, 5, 5, 10, -5, 15, 0, 5]

-- Define the function to calculate the average difference
def average_daily_difference (daily_diffs : List ℤ) : ℚ :=
  (daily_diffs.sum : ℚ) / daily_diffs.length

-- The proposition we want to prove
theorem avg_diff_noah_liam : average_daily_difference daily_differences = 60 / 14 := by
  sorry

end avg_diff_noah_liam_l135_135036


namespace part1_proof_l135_135269

def a : ℚ := 1 / 2
def b : ℚ := -2
def expr : ℚ := 2 * (3 * a^2 * b - a * b^2) - 3 * (2 * a^2 * b - a * b^2 + a * b)

theorem part1_proof : expr = 5 := by
  unfold expr
  unfold a
  unfold b
  sorry

end part1_proof_l135_135269


namespace mean_value_of_quadrilateral_angles_l135_135325

-- Statement of the problem: mean value of interior angles in any quadrilateral is 90°
theorem mean_value_of_quadrilateral_angles : 
  (∀ (a b c d : ℝ), a + b + c + d = 360 → (a + b + c + d) / 4 = 90) :=
by
  sorry

end mean_value_of_quadrilateral_angles_l135_135325


namespace relation_of_exponents_l135_135657

theorem relation_of_exponents
  (a b c d : ℝ)
  (x y p z : ℝ)
  (h1 : a^x = c)
  (h2 : b^p = c)
  (h3 : b^y = d)
  (h4 : a^z = d) :
  py = xz :=
sorry

end relation_of_exponents_l135_135657


namespace no_n_divisible_by_1955_l135_135209

theorem no_n_divisible_by_1955 : ∀ n : ℕ, ¬ (1955 ∣ (n^2 + n + 1)) := by
  sorry

end no_n_divisible_by_1955_l135_135209


namespace proposition_true_and_negation_false_l135_135941

theorem proposition_true_and_negation_false (a b : ℝ) : 
  (a + b ≥ 2 → (a ≥ 1 ∨ b ≥ 1)) ∧ ¬(a + b ≥ 2 → (a < 1 ∧ b < 1)) :=
by {
  sorry
}

end proposition_true_and_negation_false_l135_135941


namespace perimeter_C_l135_135289

theorem perimeter_C :
  ∀ (x y : ℕ),
  (6 * x + 2 * y = 56) →
  (4 * x + 6 * y = 56) →
  (2 * x + 6 * y = 40) :=
by
  intros x y hA hB
  sorry

end perimeter_C_l135_135289


namespace mean_value_of_quadrilateral_angles_l135_135341

theorem mean_value_of_quadrilateral_angles : 
  (∀ (A B C D : ℕ), A + B + C + D = 360 → (A + B + C + D) / 4 = 90) :=
by {
  sorry
}

end mean_value_of_quadrilateral_angles_l135_135341


namespace combi_sum_l135_135614

theorem combi_sum : (Nat.choose 8 2) + (Nat.choose 8 3) + (Nat.choose 9 2) = 120 :=
by
  sorry

end combi_sum_l135_135614


namespace natural_pairs_prime_l135_135214

theorem natural_pairs_prime (x y : ℕ) (p : ℕ) (hp : Nat.Prime p) (h_eq : p = xy^2 / (x + y))
  : (x, y) = (2, 2) ∨ (x, y) = (6, 2) :=
sorry

end natural_pairs_prime_l135_135214


namespace problem1_problem2_l135_135058

noncomputable def problem1_set_a : Set ℝ := {0, 1/3, -1/2}

theorem problem1 (a : ℝ) (P S : Set ℝ) (hP : P = {x | x^2 + x - 6 = 0}) (hS : S = {x | a*x + 1 = 0}) :
    (S ⊆ P) ↔ a ∈ problem1_set_a :=
by
  sorry

noncomputable def problem2_set_m : Set ℝ := {m | m ≤ 3}

theorem problem2 (m : ℝ) (A B : Set ℝ) (hA : A = {x | -2 ≤ x ∧ x ≤ 5}) (hB : B = {x | (m + 1) ≤ x ∧ x ≤ (2*m - 1)}) :
    (B ⊆ A) ↔ m ∈ problem2_set_m :=
by
  sorry

end problem1_problem2_l135_135058


namespace largest_N_satisfying_cond_l135_135818

theorem largest_N_satisfying_cond :
  ∃ N : ℕ, (∀ d1 d2 d3 d4 d5 d6 d7 d8 d9 : ℕ, 
  list.in_list [1, d1, d2, d3, d4, d5, d6, d7, N],
  d8 = 21 * d2 ∧ N = 441) :=
sorry

end largest_N_satisfying_cond_l135_135818


namespace triangle_ABC_properties_l135_135122

theorem triangle_ABC_properties 
  (a b c : ℝ) (A B C : ℝ)
  (h1 : a * Real.sin C = Real.sqrt 3 * c * Real.cos A)
  (h2 : a = Real.sqrt 13)
  (h3 : c = 3)
  (h_angle_range : A > 0 ∧ A < Real.pi) : 
  A = Real.pi / 3 ∧ (1 / 2) * b * c * Real.sin A = 3 * Real.sqrt 3 := 
by
  sorry

end triangle_ABC_properties_l135_135122


namespace erdos_problem_l135_135964

variable (X : Type) [Infinite X] (𝓗 : Set (Set X))
variable (h1 : ∀ (A : Set X) (hA : A.Finite), ∃ (H1 H2 : Set X) (hH1 : H1 ∈ 𝓗) (hH2 : H2 ∈ 𝓗), H1 ∩ H2 = ∅ ∧ H1 ∪ H2 = A)

theorem erdos_problem (k : ℕ) (hk : k > 0) : 
  ∃ (A : Set X) (ways : Finset (Set X × Set X)), A.Finite ∧ (∀ (p : Set X × Set X), p ∈ ways → p.1 ∈ 𝓗 ∧ p.2 ∈ 𝓗 ∧ p.1 ∩ p.2 = ∅ ∧ p.1 ∪ p.2 = A) ∧ ways.card ≥ k :=
by
  sorry

end erdos_problem_l135_135964


namespace find_t_l135_135499

def utility (hours_math hours_reading hours_painting : ℕ) : ℕ :=
  hours_math^2 + hours_reading * hours_painting

def utility_wednesday (t : ℕ) : ℕ :=
  utility 4 t (12 - t)

def utility_thursday (t : ℕ) : ℕ :=
  utility 3 (t + 1) (11 - t)

theorem find_t (t : ℕ) (h : utility_wednesday t = utility_thursday t) : t = 2 :=
by
  sorry

end find_t_l135_135499


namespace common_ratio_of_geometric_series_l135_135631

-- Definitions of the first two terms of the geometric series
def term1 : ℚ := 4 / 7
def term2 : ℚ := -8 / 3

-- Theorem to prove the common ratio
theorem common_ratio_of_geometric_series : (term2 / term1 = -14 / 3) := by
  sorry

end common_ratio_of_geometric_series_l135_135631


namespace mod_remainder_l135_135767

theorem mod_remainder :
  ((85^70 + 19^32)^16) % 21 = 16 := by
  -- Given conditions
  have h1 : 85^70 % 21 = 1 := sorry
  have h2 : 19^32 % 21 = 4 := sorry
  -- Conclusion
  sorry

end mod_remainder_l135_135767


namespace binary_multiplication_binary_result_l135_135928

-- Definitions for binary numbers
def bin_11011 : ℕ := 27 -- 11011 in binary is 27 in decimal
def bin_101 : ℕ := 5 -- 101 in binary is 5 in decimal

-- Theorem statement to prove the product of two binary numbers
theorem binary_multiplication : (bin_11011 * bin_101) = 135 := by
  sorry

-- Convert the result back to binary, expected to be 10000111
theorem binary_result : 135 = 8 * 16 + 7 := by
  sorry

end binary_multiplication_binary_result_l135_135928


namespace min_sum_x_y_l135_135935

theorem min_sum_x_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h_xy : 4 * x + y = x * y) : x + y ≥ 9 :=
by sorry

example (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h_xy : 4 * x + y = x * y) : x + y = 9 ↔ (x = 3 ∧ y = 6) :=
by sorry

end min_sum_x_y_l135_135935


namespace carpet_size_l135_135247

def length := 5
def width := 2
def area := length * width

theorem carpet_size : area = 10 := by
  sorry

end carpet_size_l135_135247


namespace range_of_c_l135_135656

theorem range_of_c (a c : ℝ) (ha : a ≥ 1 / 8)
  (h : ∀ x > 0, 2 * x + a / x ≥ c) : c ≤ 1 :=
sorry

end range_of_c_l135_135656


namespace no_solution_eq_eight_diff_l135_135533

theorem no_solution_eq_eight_diff (k : ℕ) (h1 : k > 0) (h2 : k ≤ 99) 
  (h3 : ∀ x y : ℕ, x^2 - k * y^2 ≠ 8) : 
  (99 - 3 = 96) := 
by 
  sorry

end no_solution_eq_eight_diff_l135_135533


namespace greatest_possible_remainder_l135_135650

theorem greatest_possible_remainder (x : ℕ) : ∃ r : ℕ, r < 12 ∧ r ≠ 0 ∧ x % 12 = r ∧ r = 11 :=
by 
  sorry

end greatest_possible_remainder_l135_135650


namespace g_five_eq_248_l135_135841

-- We define g, and assume it meets the conditions described.
variable (g : ℤ → ℤ)

-- Condition 1: g(1) > 1
axiom g_one_gt_one : g 1 > 1

-- Condition 2: Functional equation for g
axiom g_funct_eq (x y : ℤ) : g (x + y) + x * g y + y * g x = g x * g y + x + y + x * y

-- Condition 3: Recursive relationship for g
axiom g_recur_eq (x : ℤ) : 3 * g x = g (x + 1) + 2 * x - 1

-- Theorem we want to prove
theorem g_five_eq_248 : g 5 = 248 := by
  sorry

end g_five_eq_248_l135_135841


namespace mean_value_of_quadrilateral_interior_angles_l135_135463

theorem mean_value_of_quadrilateral_interior_angles (a b c d : ℝ) 
  (h_sum : a + b + c + d = 360) : 
  (a + b + c + d) / 4 = 90 :=
by
  sorry

end mean_value_of_quadrilateral_interior_angles_l135_135463


namespace mean_value_of_quadrilateral_angles_l135_135331

-- Statement of the problem: mean value of interior angles in any quadrilateral is 90°
theorem mean_value_of_quadrilateral_angles : 
  (∀ (a b c d : ℝ), a + b + c + d = 360 → (a + b + c + d) / 4 = 90) :=
by
  sorry

end mean_value_of_quadrilateral_angles_l135_135331


namespace number_of_possible_winning_scores_l135_135524

noncomputable def sum_of_first_n_integers (n : ℕ) : ℕ :=
  n * (n + 1) / 2

theorem number_of_possible_winning_scores : 
  let total_sum := sum_of_first_n_integers 12
  let max_possible_score := total_sum / 2
  let min_possible_score := sum_of_first_n_integers 6
  39 - 21 + 1 = 19 := 
by
  sorry

end number_of_possible_winning_scores_l135_135524


namespace range_of_m_l135_135864

theorem range_of_m (m : ℝ) : 
  (∀ x y : ℝ, (x - (m^2 - 2 * m + 4) * y - 6 > 0) ↔ (x, y) ≠ (-1, -1)) →
  -1 ≤ m ∧ m ≤ 3 :=
by
  intro h
  sorry

end range_of_m_l135_135864


namespace westward_fish_caught_fraction_l135_135176

theorem westward_fish_caught_fraction :
  let westward := 1800 in
  let eastward := 3200 in
  let north := 500 in
  let eastward_caught_fraction := (2 / 5 : ℚ) in
  let total_initial := westward + eastward + north in
  let remaining := 2870 in
  let total_caught := total_initial - remaining in
  let eastward_caught := eastward_caught_fraction * eastward in
  ∃ (x : ℚ), eastward_caught + x * westward = total_caught ∧ x = 3 / 4 :=
by
  sorry

end westward_fish_caught_fraction_l135_135176


namespace mean_of_quadrilateral_angles_l135_135411

theorem mean_of_quadrilateral_angles : 
  (∀ (a b c d : ℝ), a + b + c + d = 360) → 90 :=
by
  intro h
  sorry

end mean_of_quadrilateral_angles_l135_135411


namespace perimeter_C_l135_135292

theorem perimeter_C :
  ∀ (x y : ℕ),
  (6 * x + 2 * y = 56) →
  (4 * x + 6 * y = 56) →
  (2 * x + 6 * y = 40) :=
by
  intros x y hA hB
  sorry

end perimeter_C_l135_135292


namespace number_of_ordered_triples_l135_135187

theorem number_of_ordered_triples (a b c : ℕ) (h1 : a ≤ b) (h2 : b ≤ c) (h3 : b = 3969) (h4 : a * c = 3969^2) :
    ∃ n : ℕ, n = 12 := sorry

end number_of_ordered_triples_l135_135187


namespace range_of_a_l135_135769

variable (a : ℝ)

def p : Prop := ∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → x ^ 2 - a ≥ 0
def q : Prop := ∃ x : ℝ, x ^ 2 + 2 * a * x + 2 - a = 0

theorem range_of_a (h₁ : p a) (h₂ : q a) : a ≤ -2 ∨ a = 1 := 
sorry

end range_of_a_l135_135769


namespace repeating_decimal_transform_l135_135796

theorem repeating_decimal_transform (n : ℕ) (s : String) (k : ℕ) (m : ℕ)
  (original : s = "2345678") (len : k = 7) (position : n = 2011)
  (effective_position : m = n - 1) (mod_position : m % k = 3) :
  "0.1" ++ s = "0.12345678" :=
sorry

end repeating_decimal_transform_l135_135796


namespace designer_suit_size_l135_135919

theorem designer_suit_size : ∀ (waist_in_inches : ℕ) (comfort_in_inches : ℕ) 
  (inches_per_foot : ℕ) (cm_per_foot : ℝ), 
  waist_in_inches = 34 →
  comfort_in_inches = 2 →
  inches_per_foot = 12 →
  cm_per_foot = 30.48 →
  (((waist_in_inches + comfort_in_inches) / inches_per_foot : ℝ) * cm_per_foot) = 91.4 :=
by
  intros waist_in_inches comfort_in_inches inches_per_foot cm_per_foot
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_cast
  norm_num
  sorry

end designer_suit_size_l135_135919


namespace veg_eaters_l135_135173

variable (n_veg_only n_both : ℕ)

theorem veg_eaters
  (h1 : n_veg_only = 15)
  (h2 : n_both = 11) :
  n_veg_only + n_both = 26 :=
by sorry

end veg_eaters_l135_135173


namespace andrew_made_35_sandwiches_l135_135746

-- Define the number of friends and sandwiches per friend
def num_friends : ℕ := 7
def sandwiches_per_friend : ℕ := 5

-- Define the total number of sandwiches and prove it equals 35
def total_sandwiches : ℕ := num_friends * sandwiches_per_friend

theorem andrew_made_35_sandwiches : total_sandwiches = 35 := by
  sorry

end andrew_made_35_sandwiches_l135_135746


namespace total_sugar_weight_l135_135877

theorem total_sugar_weight (x y : ℝ) (h1 : y - x = 8) (h2 : x - 1 = 0.6 * (y + 1)) : x + y = 40 := by
  sorry

end total_sugar_weight_l135_135877


namespace shaded_area_of_joined_squares_l135_135747

theorem shaded_area_of_joined_squares:
  ∀ (a b : ℕ) (area_of_shaded : ℝ),
  (a = 6) → (b = 8) → 
  (area_of_shaded = (6 * 6 : ℝ) + (8 * 8 : ℝ) / 2) →
  area_of_shaded = 50.24 := 
by
  intros a b area_of_shaded h1 h2 h3
  -- skipping the proof for now
  sorry

end shaded_area_of_joined_squares_l135_135747


namespace inequality_solution_l135_135227

noncomputable def f (x : ℝ) : ℝ := abs (x + 1) + abs (x - 3)

theorem inequality_solution :
  -3 < x ∧ x < -1 ↔ f (x^2 - 3) < f (x - 1) :=
sorry

end inequality_solution_l135_135227


namespace R_depends_on_a_d_m_l135_135540

theorem R_depends_on_a_d_m (a d m : ℝ) :
    let s1 := (m / 2) * (2 * a + (m - 1) * d)
    let s2 := m * (2 * a + (2 * m - 1) * d)
    let s3 := 2 * m * (2 * a + (4 * m - 1) * d)
    let R := s3 - 2 * s2 + s1
    R = m * (a + 12 * m * d - (d / 2)) := by
  sorry

end R_depends_on_a_d_m_l135_135540


namespace intersection_A_B_subset_A_B_l135_135232

-- Definitions for the sets A and B
def set_A (a : ℝ) : Set ℝ := {x | 2 * a - 1 ≤ x ∧ x ≤ a + 3}
def set_B : Set ℝ := {x | x < -1 ∨ x > 5}

-- First proof problem: Intersection
theorem intersection_A_B (a : ℝ) (ha : a = -2) :
  set_A a ∩ set_B = {x | -5 ≤ x ∧ x < -1} :=
sorry

-- Second proof problem: Subset
theorem subset_A_B (a : ℝ) :
  set_A a ⊆ set_B ↔ (a ≤ -4 ∨ a ≥ 3) :=
sorry

end intersection_A_B_subset_A_B_l135_135232


namespace merchant_printer_count_l135_135600

theorem merchant_printer_count (P : ℕ) 
  (cost_keyboards : 15 * 20 = 300)
  (total_cost : 300 + 70 * P = 2050) :
  P = 25 := 
by
  sorry

end merchant_printer_count_l135_135600


namespace ratio_of_values_l135_135075

-- Define the geometric sequence with first term and common ratio
def geom_seq_term (a₁ : ℚ) (r : ℚ) (n : ℕ) : ℚ := a₁ * r^(n-1)

-- Define the sum of the first n terms of the geometric sequence
def geom_seq_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  if r = 1 then a * n else a * (1 - r^n) / (1 - r)

-- Sum of the first n terms for given sequence
noncomputable def S_n (n : ℕ) : ℚ :=
  geom_seq_sum (3/2) (-1/2) n

-- Define the function f(t) = t - 1/t
def f (t : ℚ) : ℚ := t - 1 / t

-- Define the maximum and minimum values of f(S_n) and their ratio
noncomputable def ratio_max_min_values : ℚ :=
  let max_val := f (3/2)
  let min_val := f (3/4)
  max_val / min_val

-- The theorem to prove the ratio of the maximum and minimum values
theorem ratio_of_values :
  ratio_max_min_values = -10/7 := by
  sorry

end ratio_of_values_l135_135075


namespace B_work_time_l135_135179

noncomputable def workRateA (W : ℝ): ℝ := W / 14
noncomputable def combinedWorkRate (W : ℝ): ℝ := W / 10

theorem B_work_time (W : ℝ) :
  ∃ T : ℝ, (W / T) = (combinedWorkRate W) - (workRateA W) ∧ T = 35 :=
by {
  use 35,
  sorry
}

end B_work_time_l135_135179


namespace min_value_4x_3y_l135_135519

theorem min_value_4x_3y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 3 * x + y = 5 * x * y) : 
  4 * x + 3 * y ≥ 5 :=
sorry

end min_value_4x_3y_l135_135519


namespace unique_acute_triangulation_l135_135262

-- Definitions for the proof problem
def is_convex (polygon : Type) : Prop := sorry
def is_acute_triangle (triangle : Type) : Prop := sorry
def is_triangulation (polygon : Type) (triangulation : List Type) : Prop := sorry
def is_acute_triangulation (polygon : Type) (triangulation : List Type) : Prop :=
  is_triangulation polygon triangulation ∧ ∀ triangle ∈ triangulation, is_acute_triangle triangle

-- Proposition to be proved
theorem unique_acute_triangulation (n : ℕ) (polygon : Type) 
  (h₁ : is_convex polygon) (h₂ : n ≥ 3) :
  ∃! triangulation : List Type, is_acute_triangulation polygon triangulation := 
sorry

end unique_acute_triangulation_l135_135262


namespace perimeter_C_l135_135285

theorem perimeter_C (x y : ℕ) 
  (h1 : 6 * x + 2 * y = 56)
  (h2 : 4 * x + 6 * y = 56) :
  2 * x + 6 * y = 40 := 
by
  sorry

end perimeter_C_l135_135285


namespace coefficient_of_m5n4_in_expansion_l135_135572

def binomial_coefficient (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem coefficient_of_m5n4_in_expansion : binomial_coefficient 9 5 = 126 := by
  sorry

end coefficient_of_m5n4_in_expansion_l135_135572


namespace prime_roots_range_l135_135518

theorem prime_roots_range (p : ℕ) (hp : Prime p) (h : ∃ x₁ x₂ : ℤ, x₁ + x₂ = -p ∧ x₁ * x₂ = -444 * p) : 31 < p ∧ p ≤ 41 :=
by sorry

end prime_roots_range_l135_135518


namespace smallest_n_for_poly_l135_135724

-- Defining the polynomial equation
def poly_eq_zero (z : ℂ) : Prop := z^6 - z^3 + 1 = 0

-- Definition of n-th roots of unity
def nth_roots_of_unity (n : ℕ) : set ℂ := {z : ℂ | z^n = 1}

-- Definition to check if all roots are n-th roots of unity
def all_roots_are_nth_roots_of_unity (n : ℕ) : Prop :=
  ∀ z : ℂ, poly_eq_zero z → z ∈ nth_roots_of_unity n

-- The theorem statement
theorem smallest_n_for_poly : ∃ n : ℕ, all_roots_are_nth_roots_of_unity n ∧ ∀ m : ℕ, all_roots_are_nth_roots_of_unity m → n ≤ m :=
begin
  sorry
end

end smallest_n_for_poly_l135_135724


namespace sum_C_D_eq_one_fifth_l135_135208

theorem sum_C_D_eq_one_fifth (D C : ℚ) :
  (∀ x : ℚ, (Dx - 13) / (x^2 - 9 * x + 20) = C / (x - 4) + 5 / (x - 5)) →
  (C + D) = 1/5 :=
by
  sorry

end sum_C_D_eq_one_fifth_l135_135208


namespace second_train_length_l135_135570

noncomputable def length_of_second_train (speed1_kmph speed2_kmph time_sec length1_m : ℝ) : ℝ :=
  let relative_speed_mps := (speed1_kmph + speed2_kmph) * (1000 / 3600)
  let total_distance := relative_speed_mps * time_sec
  total_distance - length1_m

theorem second_train_length :
  length_of_second_train 60 48 9.99920006399488 140 = 159.9760019198464 :=
by
  sorry

end second_train_length_l135_135570


namespace seventh_term_geometric_sequence_l135_135958

theorem seventh_term_geometric_sequence :
  ∃ (a₁ a₁₀ a₇ : ℕ) (r : ℕ),
    a₁ = 6 ∧ a₁₀ = 93312 ∧
    a₁₀ = a₁ * r^9 ∧
    a₇ = a₁ * r^6 ∧
    a₇ = 279936 :=
by
  sorry

end seventh_term_geometric_sequence_l135_135958


namespace line_through_circle_center_l135_135558

theorem line_through_circle_center {m : ℝ} :
  (∃ (x y : ℝ), x - 2*y + m = 0 ∧ x^2 + y^2 + 2*x - 4*y = 0) → m = 5 :=
by
  sorry

end line_through_circle_center_l135_135558


namespace mean_value_of_quadrilateral_angles_l135_135438

theorem mean_value_of_quadrilateral_angles 
  (a1 a2 a3 a4 : ℝ) 
  (h_sum : a1 + a2 + a3 + a4 = 360) :
  (a1 + a2 + a3 + a4) / 4 = 90 :=
by
  sorry

end mean_value_of_quadrilateral_angles_l135_135438


namespace pow_expression_eq_l135_135753

theorem pow_expression_eq : (-3)^(4^2) + 2^(3^2) = 43047233 := by
  sorry

end pow_expression_eq_l135_135753


namespace probability_sum_equals_5_l135_135932

/-
From the numbers 1, 2, 3, and 4, two numbers are randomly selected at once. 
Prove that the probability that the sum of these two numbers equals 5 is 1/3.
-/
theorem probability_sum_equals_5 :
  let numbers := {1, 2, 3, 4}
  let pairs := { (a, b) | a ∈ numbers ∧ b ∈ numbers ∧ a < b }
  let favorable_pairs := { (1, 4), (2, 3) }
  P(favorable_pairs, pairs) = 1 / 3 :=
by
  sorry

end probability_sum_equals_5_l135_135932


namespace volume_of_intersection_l135_135921

noncomputable section

open Real

def region1 (x y z : ℝ) : Prop :=
  abs x + abs y + abs z ≤ 1

def region2 (x y z : ℝ) : Prop :=
  abs x + abs y + abs (z - 1.5) ≤ 1

def intersection_volume_of_regions : ℝ :=
  0.1839

theorem volume_of_intersection : 
  (volume {(x, y, z) | region1 x y z ∧ region2 x y z}) = intersection_volume_of_regions :=
by
  sorry

end volume_of_intersection_l135_135921


namespace perimeter_C_l135_135282

theorem perimeter_C : 
  ∀ {x y : ℕ}, 
    (6 * x + 2 * y = 56) → (4 * x + 6 * y = 56) → 
    (2 * x + 6 * y = 40) :=
by
  intros x y h1 h2
  sorry

end perimeter_C_l135_135282


namespace axis_of_symmetry_parabola_l135_135833

theorem axis_of_symmetry_parabola : 
  ∀ (x : ℝ), ∃ y : ℝ, y = (x - 5)^2 → x = 5 := 
by 
  sorry

end axis_of_symmetry_parabola_l135_135833


namespace find_g5_l135_135860

noncomputable def g (x : ℤ) : ℤ := sorry

axiom condition1 : g 1 > 1
axiom condition2 : ∀ x y : ℤ, g (x + y) + x * g y + y * g x = g x * g y + x + y + x * y
axiom condition3 : ∀ x : ℤ, 3 * g x = g (x + 1) + 2 * x - 1

theorem find_g5 : g 5 = 248 := 
by
  sorry

end find_g5_l135_135860


namespace directrix_of_parabola_l135_135554

theorem directrix_of_parabola (x y : ℝ) (h : y = x^2) : y = -1 / 4 :=
sorry

end directrix_of_parabola_l135_135554


namespace g_five_eq_248_l135_135840

-- We define g, and assume it meets the conditions described.
variable (g : ℤ → ℤ)

-- Condition 1: g(1) > 1
axiom g_one_gt_one : g 1 > 1

-- Condition 2: Functional equation for g
axiom g_funct_eq (x y : ℤ) : g (x + y) + x * g y + y * g x = g x * g y + x + y + x * y

-- Condition 3: Recursive relationship for g
axiom g_recur_eq (x : ℤ) : 3 * g x = g (x + 1) + 2 * x - 1

-- Theorem we want to prove
theorem g_five_eq_248 : g 5 = 248 := by
  sorry

end g_five_eq_248_l135_135840


namespace area_increase_l135_135185

theorem area_increase (r₁ r₂: ℝ) (A₁ A₂: ℝ) (side1 side2: ℝ) 
  (h1: side1 = 8) (h2: side2 = 12) (h3: r₁ = side2 / 2) (h4: r₂ = side1 / 2)
  (h5: A₁ = 2 * (1/2 * Real.pi * r₁ ^ 2) + 2 * (1/2 * Real.pi * r₂ ^ 2))
  (h6: A₂ = 4 * (Real.pi * r₂ ^ 2))
  (h7: A₁ = 52 * Real.pi) (h8: A₂ = 64 * Real.pi) :
  ((A₁ + A₂) - A₁) / A₁ * 100 = 123 :=
by
  sorry

end area_increase_l135_135185


namespace pell_solution_unique_l135_135810

theorem pell_solution_unique 
  (x_0 y_0 x y : ℤ) 
  (h_fundamental : x_0^2 - 2003 * y_0^2 = 1)
  (h_pos_x : 0 < x) 
  (h_pos_y : 0 < y)
  (h_prime_div : ∀ p, Prime p → p ∣ x → p ∣ x_0) :
  x^2 - 2003 * y^2 = 1 → (x, y) = (x_0, y_0) :=
sorry

end pell_solution_unique_l135_135810


namespace sum_div_by_24_l135_135250

theorem sum_div_by_24 (m n : ℕ) (h : ∃ k : ℤ, mn + 1 = 24 * k): (m + n) % 24 = 0 := 
by
  sorry

end sum_div_by_24_l135_135250


namespace smallest_prime_x_l135_135873

-- Define prime number checker
def is_prime (n : ℕ) : Prop := n ≠ 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the problem conditions and proof goal
theorem smallest_prime_x 
  (x y z : ℕ) 
  (hx : is_prime x)
  (hy : is_prime y)
  (hz : is_prime z)
  (hxy : x ≠ y)
  (hxz : x ≠ z)
  (hyz : y ≠ z)
  (hd : ∀ d : ℕ, d ∣ (x * x * y * z) ↔ (d = 1 ∨ d = x ∨ d = x * x ∨ d = y ∨ d = x * y ∨ d = x * x * y ∨ d = z ∨ d = x * z ∨ d = x * x * z ∨ d = y * z ∨ d = x * y * z ∨ d = x * x * y * z)) 
  : x = 2 := 
sorry

end smallest_prime_x_l135_135873


namespace students_diff_l135_135985

-- Define the conditions
def M : ℕ := 457
def B : ℕ := 394

-- Prove the final answer
theorem students_diff : M - B = 63 := by
  -- The proof is omitted here with a sorry placeholder
  sorry

end students_diff_l135_135985


namespace mean_value_of_quadrilateral_angles_l135_135337

theorem mean_value_of_quadrilateral_angles : 
  (∀ (A B C D : ℕ), A + B + C + D = 360 → (A + B + C + D) / 4 = 90) :=
by {
  sorry
}

end mean_value_of_quadrilateral_angles_l135_135337


namespace solve_system_eq_l135_135813

theorem solve_system_eq (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) (hxy : x + y ≠ 0) 
  (hyz : y + z ≠ 0) (hzx : z + x ≠ 0) :
  (xy / (x + y) = 1 / 3) ∧ (yz / (y + z) = 1 / 4) ∧ (zx / (z + x) = 1 / 5) →
  (x = 1 / 2) ∧ (y = 1) ∧ (z = 1 / 3) :=
  sorry

end solve_system_eq_l135_135813


namespace passengers_got_off_l135_135872

theorem passengers_got_off :
  ∀ (initial_boarded new_boarded final_left got_off : ℕ),
    initial_boarded = 28 →
    new_boarded = 7 →
    final_left = 26 →
    got_off = initial_boarded + new_boarded - final_left →
    got_off = 9 :=
by
  intros initial_boarded new_boarded final_left got_off h_initial h_new h_final h_got_off
  rw [h_initial, h_new, h_final] at h_got_off
  exact h_got_off

end passengers_got_off_l135_135872


namespace students_not_enrolled_in_biology_class_l135_135171

theorem students_not_enrolled_in_biology_class (total_students : ℕ) (percent_biology : ℕ) 
  (h1 : total_students = 880) (h2 : percent_biology = 35) : 
  total_students - (percent_biology * total_students / 100) = 572 := by
  sorry

end students_not_enrolled_in_biology_class_l135_135171


namespace regular_polygon_enclosure_l135_135186

theorem regular_polygon_enclosure (m n : ℕ) (h : m = 12)
    (h_enc : ∀ p : ℝ, p = 360 / ↑n → (2 * (180 / ↑n)) = (360 / ↑m)) :
    n = 12 :=
by
  sorry

end regular_polygon_enclosure_l135_135186


namespace raft_time_l135_135701

-- Defining the conditions
def distance_between_villages := 1 -- unit: ed (arbitrary unit)

def steamboat_time := 1 -- unit: hours
def motorboat_time := 3 / 4 -- unit: hours
def motorboat_speed_ratio := 2 -- Motorboat speed is twice steamboat speed in still water

-- Speed equations with the current
def steamboat_speed_with_current (v_s v_c : ℝ) := v_s + v_c = 1 -- unit: ed/hr
def motorboat_speed_with_current (v_s v_c : ℝ) := 2 * v_s + v_c = 4 / 3 -- unit: ed/hr

-- Goal: Prove the time it takes for the raft to travel the same distance downstream
theorem raft_time : ∃ t : ℝ, t = 90 :=
by
  -- Definitions
  let v_s := 1 / 3 -- Speed of the steamboat in still water (derived)
  let v_c := 2 / 3 -- Speed of the current (derived)
  let raft_speed := v_c -- Raft speed equals the speed of the current
  
  -- Calculate the time for the raft to travel the distance
  let raft_time := distance_between_villages / raft_speed
  
  -- Convert time to minutes
  let raft_time_minutes := raft_time * 60
  
  -- Prove the raft time is 90 minutes
  existsi raft_time_minutes
  exact sorry

end raft_time_l135_135701


namespace watch_correction_l135_135901

noncomputable def correction_time (loss_per_day : ℕ) (start_date : ℕ) (end_date : ℕ) (spring_forward_hour : ℕ) (correction_time_hour : ℕ) : ℝ :=
  let n_days := end_date - start_date
  let total_hours_watch := n_days * 24 + correction_time_hour - spring_forward_hour
  let loss_rate_per_hour := (loss_per_day : ℝ) / 24
  let total_loss := loss_rate_per_hour * total_hours_watch
  total_loss

theorem watch_correction :
  correction_time 3 1 5 1 6 = 6.625 :=
by
  sorry

end watch_correction_l135_135901


namespace remainder_of_3_pow_17_mod_7_l135_135164

theorem remainder_of_3_pow_17_mod_7 :
  (3^17 % 7) = 5 :=
by 
  sorry

end remainder_of_3_pow_17_mod_7_l135_135164


namespace mean_of_quadrilateral_angles_l135_135381

theorem mean_of_quadrilateral_angles : 
  (∀ (a b c d : ℝ), a + b + c + d = 360 → (a + b + c + d) / 4 = 90) :=
begin
  intros a b c d h,
  calc
    (a + b + c + d) / 4 = 360 / 4 : by rw h
                    ... = 90 : by norm_num,
end

end mean_of_quadrilateral_angles_l135_135381


namespace larger_integer_l135_135321

theorem larger_integer (a b : ℕ) (h_diff : a - b = 8) (h_prod : a * b = 224) : a = 16 :=
by
  sorry

end larger_integer_l135_135321


namespace fountains_fill_pool_together_l135_135019

-- Define the times in hours for each fountain to fill the pool
def time_fountain1 : ℚ := 5 / 2  -- 2.5 hours
def time_fountain2 : ℚ := 15 / 4 -- 3.75 hours

-- Define the rates at which each fountain can fill the pool
def rate_fountain1 : ℚ := 1 / time_fountain1
def rate_fountain2 : ℚ := 1 / time_fountain2

-- Calculate the combined rate
def combined_rate : ℚ := rate_fountain1 + rate_fountain2

-- Define the time for both fountains working together to fill the pool
def combined_time : ℚ := 1 / combined_rate

-- Prove that the combined time is indeed 1.5 hours
theorem fountains_fill_pool_together : combined_time = 3 / 2 := by
  sorry

end fountains_fill_pool_together_l135_135019


namespace mean_value_of_interior_angles_of_quadrilateral_l135_135367

theorem mean_value_of_interior_angles_of_quadrilateral :
  (360 / 4) = 90 := 
by
  sorry

end mean_value_of_interior_angles_of_quadrilateral_l135_135367


namespace sum_convex_polygon_valid_n_l135_135040

theorem sum_convex_polygon_valid_n :
  let valid_n_values := {n : ℕ | ∃ (a d : ℕ), 
                            a > 0 ∧ d > 0 ∧ 
                            (∀ i j : ℕ, i < j ∧ j < n → 
                              (a + i * d) < (a + j * d) ∧ 
                              (n - 2) * 180 ∣ (a + (n - 1) * d - a)) ∧
                            (∀ i : ℕ, i < n → a + i * d < 180)} 
  in ∑ n in valid_n_values, n = 106 :=
by
  -- The proof would go here
  sorry

end sum_convex_polygon_valid_n_l135_135040


namespace largest_divisor_of_expression_l135_135517

theorem largest_divisor_of_expression
  (x : ℤ) (h_odd : x % 2 = 1) : 
  ∃ k : ℤ, k = 40 ∧ 40 ∣ (12 * x + 2) * (8 * x + 14) * (10 * x + 10) :=
by
  sorry

end largest_divisor_of_expression_l135_135517


namespace perimeter_C_l135_135281

theorem perimeter_C : 
  ∀ {x y : ℕ}, 
    (6 * x + 2 * y = 56) → (4 * x + 6 * y = 56) → 
    (2 * x + 6 * y = 40) :=
by
  intros x y h1 h2
  sorry

end perimeter_C_l135_135281


namespace coeff_x6_expansion_l135_135713

noncomputable def coefficient_x6 : ℤ :=
  ∑ k m n in finset.range 10, 
    if k + m + n = 9 ∧ 3 * m + n = 6 then 
      nat.multinomial ![k, m, n] * (1^k) * ((-3)^m) * (1^n) 
    else 0

theorem coeff_x6_expansion : coefficient_x6 = -216 := 
by {
  sorry
}

end coeff_x6_expansion_l135_135713


namespace line_properties_l135_135920

theorem line_properties : 
  ∃ (m b : ℝ), 
  (∀ x : ℝ, ∀ y : ℝ, (x = 1 ∧ y = 3) ∨ (x = 3 ∧ y = 7) → y = m * x + b) ∧
  m + b = 3 ∧
  (∀ x : ℝ, ∀ y : ℝ, (x = 0 ∧ y = 1) → y = m * x + b) :=
sorry

end line_properties_l135_135920


namespace laundry_loads_needed_l135_135477

theorem laundry_loads_needed
  (families : ℕ) (people_per_family : ℕ)
  (towels_per_person_per_day : ℕ) (days : ℕ)
  (washing_machine_capacity : ℕ)
  (h_f : families = 7)
  (h_p : people_per_family = 6)
  (h_t : towels_per_person_per_day = 2)
  (h_d : days = 10)
  (h_w : washing_machine_capacity = 10) : 
  ((families * people_per_family * towels_per_person_per_day * days) / washing_machine_capacity) = 84 := 
by
  sorry

end laundry_loads_needed_l135_135477


namespace winnie_keeps_balloons_l135_135168

theorem winnie_keeps_balloons : 
  let red := 20
  let white := 40
  let green := 70
  let yellow := 90
  let total_balloons := red + white + green + yellow
  let friends := 9
  let remainder := total_balloons % friends
  remainder = 4 :=
by
  let red := 20
  let white := 40
  let green := 70
  let yellow := 90
  let total_balloons := red + white + green + yellow
  let friends := 9
  let remainder := total_balloons % friends
  show remainder = 4
  sorry

end winnie_keeps_balloons_l135_135168


namespace find_t_l135_135026

-- Define the utility on both days
def utility_monday (t : ℝ) := t * (10 - t)
def utility_tuesday (t : ℝ) := (4 - t) * (t + 5)

-- Define the total hours spent on activities condition for both days
def total_hours_monday (t : ℝ) := t + (10 - t)
def total_hours_tuesday (t : ℝ) := (4 - t) + (t + 5)

theorem find_t : ∃ t : ℝ, t * (10 - t) = (4 - t) * (t + 5) ∧ 
                            total_hours_monday t ≥ 8 ∧ 
                            total_hours_tuesday t ≥ 8 :=
by
  sorry

end find_t_l135_135026


namespace product_of_roots_l135_135573

theorem product_of_roots :
  ∀ α β : ℝ, (Polynomial.roots (Polynomial.C 2 * Polynomial.X^2 + Polynomial.C 5 * Polynomial.X + Polynomial.C (-12))).prod = -6 :=
by
  sorry

end product_of_roots_l135_135573


namespace value_of_expression_l135_135781

theorem value_of_expression (x y : ℕ) (h1 : x = 3) (h2 : y = 4) : 
  (x^4 + 3 * y^3 + 10) / 7 = 283 / 7 := by
  sorry

end value_of_expression_l135_135781


namespace mean_value_of_quadrilateral_angles_l135_135442

theorem mean_value_of_quadrilateral_angles :
  let sum_of_angles := 360 in
  let number_of_angles := 4 in
  sum_of_angles / number_of_angles = 90 :=
by
  sorry

end mean_value_of_quadrilateral_angles_l135_135442


namespace partition_pos_integers_100_subsets_l135_135826

theorem partition_pos_integers_100_subsets :
  ∃ (P : (ℕ+ → Fin 100)), ∀ a b c : ℕ+, (a + 99 * b = c) → P a = P c ∨ P a = P b ∨ P b = P c :=
sorry

end partition_pos_integers_100_subsets_l135_135826


namespace math_crackers_initial_l135_135015

def crackers_initial (gave_each : ℕ) (left : ℕ) (num_friends : ℕ) : ℕ :=
  (gave_each * num_friends) + left

theorem math_crackers_initial :
  crackers_initial 7 17 3 = 38 :=
by
  -- The definition of crackers_initial and the theorem statement should be enough.
  -- The exact proof is left as a sorry placeholder.
  sorry

end math_crackers_initial_l135_135015


namespace perimeter_of_triangle_l135_135509

noncomputable def point_on_ellipse (P : ℝ × ℝ) : Prop := 
  (P.1^2 / 9) + (P.2^2 / 5) = 1

noncomputable def foci_position (F1 F2 : ℝ × ℝ) : Prop := 
  F1 = (-2, 0) ∧ F2 = (2, 0)

theorem perimeter_of_triangle :
  ∀ (P F1 F2 : ℝ × ℝ),
    point_on_ellipse P →
    foci_position F1 F2 →
    dist P F1 + dist P F2 + dist F1 F2 = 10 :=
by
  sorry

end perimeter_of_triangle_l135_135509


namespace mean_value_of_interior_angles_of_quadrilateral_l135_135398

theorem mean_value_of_interior_angles_of_quadrilateral 
  (sum_of_interior_angles : ℝ := 360) 
  (number_of_angles : ℝ := 4) : 
  sum_of_interior_angles / number_of_angles = 90 := 
by 
  sorry

end mean_value_of_interior_angles_of_quadrilateral_l135_135398


namespace correct_number_of_conclusions_l135_135930

def y (x : ℝ) := -5 * x + 1

def conclusion1 := y (-1) = 5
def conclusion2 := ∃ x1 x2 x3 : ℝ, y x1 > 0 ∧ y x2 > 0 ∧ y (x3) < 0 ∧ (x1 < 0) ∧ (x2 > 0) ∧ (x3 < x2)
def conclusion3 := ∀ x : ℝ, x > 1 → y x < 0
def conclusion4 := ∀ x1 x2 : ℝ, x1 < x2 → y x1 < y x2

-- We want to prove that exactly 2 of these conclusions are correct
theorem correct_number_of_conclusions : (¬ conclusion1 ∧ conclusion2 ∧ conclusion3 ∧ ¬ conclusion4) :=
by
  sorry

end correct_number_of_conclusions_l135_135930


namespace calculate_expression_l135_135474

theorem calculate_expression : 5 * 399 + 4 * 399 + 3 * 399 + 397 = 5185 := 
by
  -- proof goes here
  sorry

end calculate_expression_l135_135474


namespace diameterOuterBoundary_l135_135952

-- Definitions based on the conditions in the problem
def widthWalkingPath : ℝ := 10
def widthGardenRing : ℝ := 12
def diameterPond : ℝ := 16

-- The main theorem that proves the diameter of the circle that forms the outer boundary of the walking path
theorem diameterOuterBoundary : 2 * ((diameterPond / 2) + widthGardenRing + widthWalkingPath) = 60 :=
by
  sorry

end diameterOuterBoundary_l135_135952


namespace mean_value_of_quadrilateral_angles_l135_135393

theorem mean_value_of_quadrilateral_angles (a b c d : ℝ) (h_angles_sum : a + b + c + d = 360) : (a + b + c + d) / 4 = 90 := 
by
  sorry

end mean_value_of_quadrilateral_angles_l135_135393


namespace compute_fraction_product_l135_135916

-- Definitions based on conditions
def one_third_pow_four : ℚ := (1 / 3) ^ 4
def one_fifth : ℚ := 1 / 5

-- Main theorem to prove the problem question == answer
theorem compute_fraction_product : (one_third_pow_four * one_fifth) = 1 / 405 :=
by
  sorry

end compute_fraction_product_l135_135916


namespace probability_sum_A1_A2_le_0_5_l135_135526

-- Declare three events with given probabilities
constants {A1 A2 A3 : Event}
variable (P : Event → ℝ)
axiom P_A1 : P A1 = 0.2
axiom P_A2 : P A2 = 0.3
axiom P_A3 : P A3 = 0.5

-- State the goal to prove
theorem probability_sum_A1_A2_le_0_5 : P (A1 ∪ A2) ≤ 0.5 :=
sorry

end probability_sum_A1_A2_le_0_5_l135_135526


namespace complex_number_solution_l135_135226

theorem complex_number_solution (a b : ℤ) (z : ℂ) (h1 : z = a + b * Complex.I) (h2 : z^3 = 2 + 11 * Complex.I) : a + b = 3 :=
sorry

end complex_number_solution_l135_135226


namespace highest_probability_ksi_expected_value_ksi_equals_l135_135525

noncomputable def probability_ksi_equals (k : ℕ) : ℚ :=
  match k with
  | 2 => 9 / 64
  | 3 => 18 / 64
  | 4 => 21 / 64
  | 5 => 12 / 64
  | 6 => 4 / 64
  | _ => 0

noncomputable def expected_value_ksi : ℚ :=
  2 * (9 / 64) + 3 * (18 / 64) + 4 * (21 / 64) + 5 * (12 / 64) + 6 * (4 / 64)

theorem highest_probability_ksi :
  ∃ k : ℕ, (∀ m : ℕ, probability_ksi_equals k ≥ probability_ksi_equals m) ∧ k = 4 :=
by
  sorry

theorem expected_value_ksi_equals :
  expected_value_ksi = 15 / 4 :=
by
  sorry

end highest_probability_ksi_expected_value_ksi_equals_l135_135525


namespace double_recipe_total_l135_135154

theorem double_recipe_total 
  (butter_ratio : ℕ) (flour_ratio : ℕ) (sugar_ratio : ℕ) 
  (flour_cups : ℕ) 
  (h_ratio : butter_ratio = 2) 
  (h_flour : flour_ratio = 5) 
  (h_sugar : sugar_ratio = 3) 
  (h_flour_cups : flour_cups = 15) : 
  2 * ((butter_ratio * (flour_cups / flour_ratio)) + flour_cups + (sugar_ratio * (flour_cups / flour_ratio))) = 60 := 
by 
  sorry

end double_recipe_total_l135_135154


namespace compute_f_of_1_plus_g_of_3_l135_135807

def f (x : ℝ) : ℝ := 3 * x - 4
def g (x : ℝ) : ℝ := x^2 + 1

theorem compute_f_of_1_plus_g_of_3 : f (1 + g 3) = 29 := by 
  sorry

end compute_f_of_1_plus_g_of_3_l135_135807


namespace smallest_n_satisfying_conditions_l135_135023

-- We need variables and statements
variables (n : ℕ)

-- Define the conditions
def condition1 : Prop := n % 6 = 4
def condition2 : Prop := n % 7 = 3
def condition3 : Prop := n > 20

-- The main theorem statement to be proved
theorem smallest_n_satisfying_conditions (h1 : condition1 n) (h2 : condition2 n) (h3 : condition3 n) : n = 52 :=
by 
  sorry

end smallest_n_satisfying_conditions_l135_135023


namespace jinsu_work_per_hour_l135_135534

theorem jinsu_work_per_hour (t : ℝ) (h : t = 4) : (1 / t = 1 / 4) :=
by {
    sorry
}

end jinsu_work_per_hour_l135_135534


namespace ax0_eq_b_condition_l135_135505

theorem ax0_eq_b_condition (a b x0 : ℝ) (h : a < 0) : (ax0 = b) ↔ (∀ x : ℝ, (1/2 * a * x^2 - b * x) ≤ (1/2 * a * x0^2 - b * x0)) :=
sorry

end ax0_eq_b_condition_l135_135505


namespace lena_calculation_l135_135806

def round_to_nearest_ten (n : ℕ) : ℕ :=
  if n % 10 < 5 then n - n % 10 else n + (10 - n % 10)

theorem lena_calculation :
  round_to_nearest_ten (63 + 2 * 29) = 120 :=
by
  sorry

end lena_calculation_l135_135806


namespace mean_value_interior_angles_quadrilateral_l135_135370

theorem mean_value_interior_angles_quadrilateral :
  let n := 4 in
  (n - 2) * 180 / n = 90 :=
by
  let n := 4
  sorry

end mean_value_interior_angles_quadrilateral_l135_135370


namespace percentage_more_than_cost_price_l135_135037

noncomputable def SP : ℝ := 7350
noncomputable def CP : ℝ := 6681.818181818181

theorem percentage_more_than_cost_price : 
  (SP - CP) / CP * 100 = 10 :=
by
  sorry

end percentage_more_than_cost_price_l135_135037


namespace eval_f_nested_l135_135647

noncomputable def f (x : ℝ) : ℝ :=
if h : x < 0 then x + 1 else x ^ 2

theorem eval_f_nested : f (f (-2)) = 0 := by
  sorry

end eval_f_nested_l135_135647


namespace system_solution_l135_135145

theorem system_solution (x y : ℚ) (h1 : 2 * x - 3 * y = 1) (h2 : (y + 1) / 4 + 1 = (x + 2) / 3) : x = 3 ∧ y = 5 / 3 :=
by
  sorry

end system_solution_l135_135145


namespace aquafaba_needed_for_cakes_l135_135199

def tablespoons_of_aquafaba_for_egg_whites (n_egg_whites : ℕ) : ℕ :=
  2 * n_egg_whites

def total_egg_whites_needed (cakes : ℕ) (egg_whites_per_cake : ℕ) : ℕ :=
  cakes * egg_whites_per_cake

theorem aquafaba_needed_for_cakes (cakes : ℕ) (egg_whites_per_cake : ℕ) :
  tablespoons_of_aquafaba_for_egg_whites (total_egg_whites_needed cakes egg_whites_per_cake) = 32 :=
by
  have h1 : cakes = 2 := sorry
  have h2 : egg_whites_per_cake = 8 := sorry
  sorry

end aquafaba_needed_for_cakes_l135_135199


namespace distinct_pos_real_ints_l135_135273

noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

theorem distinct_pos_real_ints (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a ≠ b) (h4 : ∀ n : ℕ, (floor (n * a)) ∣ (floor (n * b))) : ∃ k l : ℤ, a = k ∧ b = l :=
by
  sorry

end distinct_pos_real_ints_l135_135273


namespace inequality_AM_GM_HM_l135_135108

variable {x y k : ℝ}

-- Define the problem conditions
def is_positive (a : ℝ) : Prop := a > 0
def is_unequal (a b : ℝ) : Prop := a ≠ b
def positive_constant_lessthan_two (c : ℝ) : Prop := c > 0 ∧ c < 2

-- State the theorem to be proven
theorem inequality_AM_GM_HM (h₁ : is_positive x) 
                             (h₂ : is_positive y) 
                             (h₃ : is_unequal x y) 
                             (h₄ : positive_constant_lessthan_two k) :
  ( ( ( (x + y) / 2 )^k > ( (x * y)^(1/2) )^k ) ∧ 
    ( ( (x * y)^(1/2) )^k > ( ( 2 * x * y ) / ( x + y ) )^k ) ) :=
by
  sorry

end inequality_AM_GM_HM_l135_135108


namespace mean_value_interior_angles_quadrilateral_l135_135374

theorem mean_value_interior_angles_quadrilateral :
  let n := 4 in
  (n - 2) * 180 / n = 90 :=
by
  let n := 4
  sorry

end mean_value_interior_angles_quadrilateral_l135_135374


namespace problem_statement_l135_135090

def digit_sum (k : ℕ) : ℕ :=
  k.digits 10 |>.sum

theorem problem_statement :
  ∀ n : ℕ, (∃ a b : ℕ, n = digit_sum a ∧ n = digit_sum b ∧ n = digit_sum (a + b)) ↔ (∃ k : ℕ, n = 9 * k) :=
by
  sorry

end problem_statement_l135_135090


namespace find_constant_l135_135169

-- Given function f satisfying the conditions
variable (f : ℝ → ℝ)

-- Define the given conditions
variable (h1 : ∀ x : ℝ, f x + 3 * f (c - x) = x)
variable (h2 : f 2 = 2)

-- Statement to prove the constant c
theorem find_constant (c : ℝ) : (f x + 3 * f (c - x) = x) → (f 2 = 2) → c = 8 :=
by
  intro h1 h2
  sorry

end find_constant_l135_135169


namespace mean_value_of_quadrilateral_angles_l135_135430

theorem mean_value_of_quadrilateral_angles : 
  ∀ (a b c d : ℝ), a + b + c + d = 360 → (a + b + c + d) / 4 = 90 :=
by
  intro a b c d h
  sorry

end mean_value_of_quadrilateral_angles_l135_135430


namespace calculate_probability_l135_135627

theorem calculate_probability :
  let letters_in_bag : List Char := ['C', 'A', 'L', 'C', 'U', 'L', 'A', 'T', 'E']
  let target_letters : List Char := ['C', 'U', 'T']
  let total_outcomes := letters_in_bag.length
  let favorable_outcomes := (letters_in_bag.filter (λ c => c ∈ target_letters)).length
  (favorable_outcomes : ℚ) / (total_outcomes : ℚ) = 4 / 9 := sorry

end calculate_probability_l135_135627


namespace problem1_correct_solution_problem2_correct_solution_l135_135775

noncomputable def g (x a : ℝ) : ℝ := |x| + 2 * |x + 2 - a|

/-- 
    Prove that the set {x | -2/3 ≤ x ≤ 2} satisfies g(x) ≤ 4 when a = 3 
--/
theorem problem1_correct_solution (x : ℝ) : g x 3 ≤ 4 ↔ -2/3 ≤ x ∧ x ≤ 2 :=
by
  sorry

noncomputable def f (x a : ℝ) : ℝ := g (x - 2) a

/-- 
    Prove that the range of a such that f(x) ≥ 1 for all x ∈ ℝ 
    is a ≤ 1 or a ≥ 3
--/
theorem problem2_correct_solution (a : ℝ) : 
  (∀ x : ℝ, f x a ≥ 1) ↔ a ≤ 1 ∨ a ≥ 3 :=
by
  sorry

end problem1_correct_solution_problem2_correct_solution_l135_135775


namespace largest_value_of_x_l135_135692

noncomputable def find_largest_x : ℝ :=
  let a := 10
  let b := 39
  let c := 18
  let discriminant := b^2 - 4 * a * c
  let sqrt_discriminant := Real.sqrt discriminant
  let x1 := (-b + sqrt_discriminant) / (2 * a)
  let x2 := (-b - sqrt_discriminant) / (2 * a)
  if x1 > x2 then x1 else x2

theorem largest_value_of_x :
  ∃ x : ℝ, 3 * (10 * x^2 + 11 * x + 12) = x * (10 * x - 45) ∧
  x = find_largest_x := by
  exists find_largest_x
  sorry

end largest_value_of_x_l135_135692


namespace ratio_of_percentage_change_l135_135894

theorem ratio_of_percentage_change
  (P U U' : ℝ)
  (h_price_decrease : U' = 4 * U)
  : (300 / 75) = 4 := 
by
  sorry

end ratio_of_percentage_change_l135_135894


namespace fraction_power_mult_correct_l135_135913

noncomputable def fraction_power_mult : Prop :=
  (\left(\frac{1}{3} \right)^{4}) * \left(\frac{1}{5} \right) = \left(\frac{1}{405} \right)

theorem fraction_power_mult_correct : fraction_power_mult :=
by
  -- The complete proof will be here.
  sorry

end fraction_power_mult_correct_l135_135913


namespace arithmetic_sequence_a3_l135_135792

theorem arithmetic_sequence_a3 (a : ℕ → ℝ) (h : a 2 + a 4 = 8) (h_seq : a 2 + a 4 = 2 * a 3) :
  a 3 = 4 :=
by
  sorry

end arithmetic_sequence_a3_l135_135792


namespace g_5_is_248_l135_135844

def g : ℤ → ℤ := sorry

theorem g_5_is_248 :
  (g(1) > 1) ∧
  (∀ x y : ℤ, g(x + y) + x * g(y) + y * g(x) = g(x) * g(y) + x + y + x * y) ∧
  (∀ x : ℤ, 3 * g(x) = g(x + 1) + 2 * x - 1) →
  g(5) = 248 :=
by
  -- proof omitted
  sorry

end g_5_is_248_l135_135844


namespace smallest_n_for_roots_of_unity_l135_135721

theorem smallest_n_for_roots_of_unity (z : ℂ) (h : z^6 - z^3 + 1 = 0) : ∃ (n : ℕ), n = 18 ∧ z^n = 1 :=
by {
  sorry
}

end smallest_n_for_roots_of_unity_l135_135721


namespace average_salary_rest_l135_135670

theorem average_salary_rest (total_workers : ℕ) (avg_salary_all : ℝ)
  (num_technicians : ℕ) (avg_salary_technicians : ℝ) :
  total_workers = 21 →
  avg_salary_all = 8000 →
  num_technicians = 7 →
  avg_salary_technicians = 12000 →
  (avg_salary_all * total_workers - avg_salary_technicians * num_technicians) / (total_workers - num_technicians) = 6000 :=
by intros h1 h2 h3 h4; sorry

end average_salary_rest_l135_135670


namespace number_of_solutions_l135_135101

theorem number_of_solutions : 
  ∃ S : Finset ℤ, (∀ x ∈ S, 20 ≤ x ∧ x ≤ 150 ∧ Odd x ∧ (x + 17) % 29 = 65 % 29) ∧ S.card = 3 :=
by
  sorry

end number_of_solutions_l135_135101


namespace mean_value_interior_angles_quadrilateral_l135_135376

theorem mean_value_interior_angles_quadrilateral :
  let n := 4 in
  (n - 2) * 180 / n = 90 :=
by
  let n := 4
  sorry

end mean_value_interior_angles_quadrilateral_l135_135376


namespace calculate_binom_l135_135906

theorem calculate_binom : 2 * Nat.choose 30 3 = 8120 := 
by 
  sorry

end calculate_binom_l135_135906


namespace mean_value_of_interior_angles_of_quadrilateral_l135_135396

theorem mean_value_of_interior_angles_of_quadrilateral 
  (sum_of_interior_angles : ℝ := 360) 
  (number_of_angles : ℝ := 4) : 
  sum_of_interior_angles / number_of_angles = 90 := 
by 
  sorry

end mean_value_of_interior_angles_of_quadrilateral_l135_135396


namespace john_text_messages_per_day_l135_135248

theorem john_text_messages_per_day (m n : ℕ) (h1 : m = 20) (h2 : n = 245) : 
  m + n / 7 = 55 :=
by
  sorry

end john_text_messages_per_day_l135_135248


namespace mean_value_of_quadrilateral_angles_l135_135419

theorem mean_value_of_quadrilateral_angles (angles : Fin 4 → ℝ) 
  (h_sum : ∑ i, angles i = 360) : ∑ i, angles i / 4 = 90 :=
by
  sorry

end mean_value_of_quadrilateral_angles_l135_135419


namespace isosceles_triangle_perimeter_l135_135955

variable (a b c : ℝ)
variable (h1 : a = 4 ∨ a = 8)
variable (h2 : b = 4 ∨ b = 8)
variable (h3 : a = b ∨ c = 8)

theorem isosceles_triangle_perimeter (h : a + b + c = 20) : a = b ∨ b = 8 ∧ (a = 8 ∧ c = 4 ∨ b = c) := 
  by
  sorry

end isosceles_triangle_perimeter_l135_135955


namespace find_g5_l135_135858

-- Define the function g on integers
def g : ℤ → ℤ := sorry

-- Define the conditions
axiom cond1 : g 1 > 1
axiom cond2 : ∀ (x y : ℤ), g (x + y) + x * g y + y * g x = g x * g y + x + y + x * y
axiom cond3 : ∀ (x : ℤ), 3 * g x = g (x + 1) + 2 * x - 1

-- The statement we need to prove
theorem find_g5 : g 5 = 248 := by
  sorry

end find_g5_l135_135858


namespace smallest_n_for_root_unity_l135_135728

noncomputable def smallestPositiveInteger (p : Polynomial ℂ) (n : ℕ) : Prop :=
  n > 0 ∧ ∀ z : ℂ, z^n = 1 → p.eval z = 0 → (∃ k : ℕ, k < n ∧ z = Complex.exp (2 * Real.pi * Complex.I * k / n))

theorem smallest_n_for_root_unity (n : ℕ) : 
  smallestPositiveInteger (Polynomial.map (algebraMap ℂ ℂ) (X^6 - X^3 + 1)) n ↔ n = 9 :=
begin
  sorry
end

end smallest_n_for_root_unity_l135_135728


namespace final_amount_is_75139_84_l135_135630

noncomputable def compoundInterest (P : ℝ) (r : ℝ) (t : ℝ) (n : ℝ) : ℝ :=
  P * (1 + r/n)^(n * t)

theorem final_amount_is_75139_84 (P : ℝ) (r : ℝ) (t : ℝ) (n : ℕ) :
  P = 64000 → r = 1/12 → t = 2 → n = 12 → compoundInterest P r t n = 75139.84 :=
by
  intros hP hr ht hn
  sorry

end final_amount_is_75139_84_l135_135630


namespace largest_common_term_l135_135992

theorem largest_common_term (b : ℕ) : 
  (b < 1000) ∧ (b % 5 = 4) ∧ (b % 11 = 7) → b = 964 :=
by
  intros h
  sorry

end largest_common_term_l135_135992


namespace mean_of_quadrilateral_angles_l135_135382

theorem mean_of_quadrilateral_angles : 
  (∀ (a b c d : ℝ), a + b + c + d = 360 → (a + b + c + d) / 4 = 90) :=
begin
  intros a b c d h,
  calc
    (a + b + c + d) / 4 = 360 / 4 : by rw h
                    ... = 90 : by norm_num,
end

end mean_of_quadrilateral_angles_l135_135382


namespace running_speed_proof_l135_135703

-- Definitions used in the conditions
def num_people : ℕ := 4
def stretch_km : ℕ := 300
def bike_speed_kmph : ℕ := 50
def total_time_hours : ℚ := 19 + (1/3)

-- The running speed to be proven
def running_speed_kmph : ℚ := 15.52

-- The main statement
theorem running_speed_proof
  (num_people_eq : num_people = 4)
  (stretch_eq : stretch_km = 300)
  (bike_speed_eq : bike_speed_kmph = 50)
  (total_time_eq : total_time_hours = 19.333333333333332) :
  running_speed_kmph = 15.52 :=
sorry

end running_speed_proof_l135_135703


namespace question1_solution_question2_solution_l135_135638

-- Define the function f for any value of a
def f (a : ℝ) (x : ℝ) : ℝ :=
  abs (x + 1) - abs (a * x - 1)

-- Definition specifically for question (1) setting a = 1
def f1 (x : ℝ) : ℝ :=
  f 1 x

-- Definition of the set for the inequality in (1)
def solution_set_1 : Set ℝ :=
  { x | f1 x > 1 }

-- Theorem for question (1)
theorem question1_solution :
  solution_set_1 = { x : ℝ | x > 1/2 } :=
sorry

-- Condition for question (2)
def inequality_condition (a : ℝ) (x : ℝ) : Prop :=
  f a x > x

-- Define the interval for x in question (2)
def interval_0_1 (x : ℝ) : Prop :=
  0 < x ∧ x < 1

-- Theorem for question (2)
theorem question2_solution {a : ℝ} :
  (∀ x ∈ {x | interval_0_1 x}, inequality_condition a x) ↔ (0 < a ∧ a ≤ 2) :=
sorry

end question1_solution_question2_solution_l135_135638


namespace jeremys_school_distance_l135_135246

def distance_to_school (rush_hour_time : ℚ) (no_traffic_time : ℚ) (speed_increase : ℚ) (distance : ℚ) : Prop :=
  ∃ v : ℚ, distance = v * rush_hour_time ∧ distance = (v + speed_increase) * no_traffic_time

theorem jeremys_school_distance :
  distance_to_school (3/10 : ℚ) (1/5 : ℚ) 20 12 :=
sorry

end jeremys_school_distance_l135_135246


namespace parametric_to_ellipse_parametric_to_line_l135_135623

-- Define the conditions and the corresponding parametric equations
variable (φ t : ℝ) (x y : ℝ)

-- The first parametric equation converted to the ordinary form
theorem parametric_to_ellipse (h1 : x = 5 * Real.cos φ) (h2 : y = 4 * Real.sin φ) :
  (x ^ 2 / 25) + (y ^ 2 / 16) = 1 := sorry

-- The second parametric equation converted to the ordinary form
theorem parametric_to_line (h3 : x = 1 - 3 * t) (h4 : y = 4 * t) :
  4 * x + 3 * y - 4 = 0 := sorry

end parametric_to_ellipse_parametric_to_line_l135_135623


namespace mean_value_of_interior_angles_of_quadrilateral_l135_135366

theorem mean_value_of_interior_angles_of_quadrilateral :
  (360 / 4) = 90 := 
by
  sorry

end mean_value_of_interior_angles_of_quadrilateral_l135_135366


namespace dinosaur_book_cost_l135_135536

-- Define the constants for costs and savings/needs
def dict_cost : ℕ := 11
def cookbook_cost : ℕ := 7
def savings : ℕ := 8
def needed : ℕ := 29
def total_cost : ℕ := savings + needed
def dino_cost : ℕ := 19

-- Mathematical statement to prove
theorem dinosaur_book_cost :
  dict_cost + dino_cost + cookbook_cost = total_cost :=
by
  -- The proof steps would go here
  sorry

end dinosaur_book_cost_l135_135536


namespace b_and_c_work_days_l135_135885

theorem b_and_c_work_days
  (A B C : ℝ)
  (h1 : A + B = 1 / 8)
  (h2 : A + C = 1 / 8)
  (h3 : A + B + C = 1 / 6) :
  B + C = 1 / 24 :=
sorry

end b_and_c_work_days_l135_135885


namespace wanda_crayons_l135_135323

variable (Dina Jacob Wanda : ℕ)

theorem wanda_crayons : Dina = 28 ∧ Jacob = Dina - 2 ∧ Dina + Jacob + Wanda = 116 → Wanda = 62 :=
by
  intro h
  sorry

end wanda_crayons_l135_135323


namespace complex_expression_equals_zero_l135_135934

def i : ℂ := Complex.I

theorem complex_expression_equals_zero : 2 * i^5 + (1 - i)^2 = 0 := 
by
  sorry

end complex_expression_equals_zero_l135_135934


namespace coin_same_side_probability_l135_135575

noncomputable def probability_same_side_5_tosses (p : ℚ) := (p ^ 5) + (p ^ 5)

theorem coin_same_side_probability : probability_same_side_5_tosses (1/2) = 1/16 := by
  sorry

end coin_same_side_probability_l135_135575


namespace coordinates_of_point_B_l135_135937

def A : ℝ × ℝ × ℝ := (2, -1, 7)
def a : ℝ × ℝ × ℝ := (8, 9, -12)
def segment_length : ℝ := 34

theorem coordinates_of_point_B
  (A : ℝ × ℝ × ℝ) (a : ℝ × ℝ × ℝ) (segment_length : ℝ)
  (hA : A = (2, -1, 7)) (ha : a = (8, 9, -12)) (h_length : segment_length = 34) :
  let λ := segment_length / real.sqrt (8^2 + 9^2 + (-12)^2) in
  let B := (A.1 + λ * a.1, A.2 + λ * a.2, A.3 + λ * a.3) in
  B = (18, 17, -17) :=
  by
  sorry

end coordinates_of_point_B_l135_135937


namespace chloe_boxes_of_clothing_l135_135909

theorem chloe_boxes_of_clothing (total_clothing pieces_per_box : ℕ) (h1 : total_clothing = 32) (h2 : pieces_per_box = 2 + 6) :
  ∃ B : ℕ, B = total_clothing / pieces_per_box ∧ B = 4 :=
by
  -- Proof can be filled in here
   sorry

end chloe_boxes_of_clothing_l135_135909


namespace smallest_integer_l135_135469

-- Define a function to calculate the LCM of a list of numbers
def lcm_list (l : List ℕ) : ℕ :=
  l.foldl Nat.lcm 1

-- List of divisors
def divisors : List ℕ := [4, 5, 6, 7, 8, 9, 10]

-- Calculating the required integer
noncomputable def required_integer : ℕ := lcm_list divisors + 1

-- The proof statement
theorem smallest_integer : required_integer = 2521 :=
  by 
  sorry

end smallest_integer_l135_135469


namespace gum_total_l135_135259

theorem gum_total (x y : ℕ) (hx : 10 ≤ x ∧ x < 100) (hy : 10 ≤ y ∧ y < 100) : 
  58 + x + y = 58 + x + y :=
by sorry

end gum_total_l135_135259


namespace probability_neither_event_l135_135562

-- Definitions and conditions
variable (P : Set (Set Prop) → ℝ)
variable (A B : Prop)
variable [ProbabilityMeasure P] 

-- Additional given conditions
axiom PA : P {A} = 0.20
axiom PB : P {B} = 0.40
axiom PAB : P {A, B} = 0.15

-- The theorem to prove
theorem probability_neither_event : P (neither A nor B) = 0.55 :=
by
  -- Definitions for 'or' and 'not' in terms of event A and event B
  let not_A := P {¬A}
  let not_B := P {¬B}
  let A_or_B := P {A ∨ B}
  let not_A_and_not_B := P {¬A ∧ ¬B}
  have h1 : A_or_B = PA + PB - PAB, from sorry
  have h2 : not_A_and_not_B = 1 - A_or_B, from sorry
  show not_A_and_not_B = 0.55, from sorry

end probability_neither_event_l135_135562


namespace train_length_is_correct_l135_135741

-- Definitions of speeds and time
def speedTrain_kmph := 100
def speedMotorbike_kmph := 64
def overtakingTime_s := 20

-- Calculate speeds in m/s
def speedTrain_mps := speedTrain_kmph * 1000 / 3600
def speedMotorbike_mps := speedMotorbike_kmph * 1000 / 3600

-- Calculate relative speed
def relativeSpeed_mps := speedTrain_mps - speedMotorbike_mps

-- Calculate the length of the train
def length_of_train := relativeSpeed_mps * overtakingTime_s

-- Theorem: Verifying the length of the train is 200 meters
theorem train_length_is_correct : length_of_train = 200 := by
  -- Sorry placeholder for proof
  sorry

end train_length_is_correct_l135_135741


namespace raft_travel_time_l135_135700

-- Define the problem conditions:
def steamboat_time (distance : ℕ) := 1 -- in hours
def motorboat_time (distance : ℕ) : ℚ := 3 / 4 -- in hours
def speed_ratio := 2 -- motorboat speed is twice the speed of steamboat

-- Define the time for the raft to travel the distance:
def raft_time (distance : ℕ) (current_speed : ℚ) := distance / current_speed

-- Given the conditions, prove that the raft time equals to 90 minutes
theorem raft_travel_time (distance : ℕ) (rafter_speed : ℚ) (current_speed : ℚ) :
  steamboat_time distance = 1 ∧ motorboat_time distance = 3 / 4 ∧ rafter_speed = current_speed →
  rafter_speed = current_speed ∧ raft_time distance current_speed = 3 / 2 → -- hours
  raft_time distance current_speed * 60 = 90 := -- convert hours to minutes
by
  intros h1 h2
  sorry

end raft_travel_time_l135_135700


namespace mean_value_of_quadrilateral_angles_l135_135336

theorem mean_value_of_quadrilateral_angles : 
  (∀ (A B C D : ℕ), A + B + C + D = 360 → (A + B + C + D) / 4 = 90) :=
by {
  sorry
}

end mean_value_of_quadrilateral_angles_l135_135336


namespace negation_proposition_l135_135559

theorem negation_proposition (x y : ℝ) :
  (¬ (x^2 + y^2 = 0 → x = 0 ∧ y = 0)) ↔ (x^2 + y^2 ≠ 0 → (x ≠ 0 ∨ y ≠ 0)) := 
sorry

end negation_proposition_l135_135559


namespace union_of_M_and_N_l135_135649

open Set

theorem union_of_M_and_N :
  let M := {x : ℝ | x^2 - 4 * x < 0}
  let N := {x : ℝ | |x| ≤ 2}
  M ∪ N = {x : ℝ | -2 ≤ x ∧ x < 4} :=
by
  sorry

end union_of_M_and_N_l135_135649


namespace problem_solution_l135_135240

theorem problem_solution (m : ℤ) (x : ℤ) (h : 4 * x + 2 * m = 14) : x = 2 → m = 3 :=
by sorry

end problem_solution_l135_135240


namespace problem_statement_l135_135646

theorem problem_statement (a b : ℝ) (h1 : 1 + b = 0) (h2 : a - 3 = 0) : 
  3 * (a^2 - 2 * a * b + b^2) - (4 * a^2 - 2 * (1 / 2 * a^2 + a * b - 3 / 2 * b^2)) = 12 :=
by
  sorry

end problem_statement_l135_135646


namespace unique_solution_values_l135_135635

open Real

theorem unique_solution_values (a : ℝ) : (∀ x : ℝ, 2 * log10 (x + 3) = log10 (a * x) → x > -3) → 
  (a ∈ set.Iic 0 ∪ set.Icc 12 12) := 
sorry

end unique_solution_values_l135_135635


namespace second_number_is_90_l135_135567

theorem second_number_is_90 (a b c : ℕ) 
  (h1 : a + b + c = 330) 
  (h2 : a = 2 * b) 
  (h3 : c = (1 / 3) * a) : 
  b = 90 := 
by
  sorry

end second_number_is_90_l135_135567


namespace diff_of_squares_l135_135908

variable (a : ℝ)

theorem diff_of_squares (a : ℝ) : (a + 3) * (a - 3) = a^2 - 9 := by
  sorry

end diff_of_squares_l135_135908


namespace find_g5_l135_135856

-- Define the function g on integers
def g : ℤ → ℤ := sorry

-- Define the conditions
axiom cond1 : g 1 > 1
axiom cond2 : ∀ (x y : ℤ), g (x + y) + x * g y + y * g x = g x * g y + x + y + x * y
axiom cond3 : ∀ (x : ℤ), 3 * g x = g (x + 1) + 2 * x - 1

-- The statement we need to prove
theorem find_g5 : g 5 = 248 := by
  sorry

end find_g5_l135_135856


namespace evaluate_power_l135_135235

theorem evaluate_power (x : ℝ) (hx : (8:ℝ)^(2 * x) = 11) : 
  2^(x + 1.5) = 11^(1 / 6) * 2 * Real.sqrt 2 :=
by 
  sorry

end evaluate_power_l135_135235


namespace Brians_trip_distance_l135_135751

theorem Brians_trip_distance (miles_per_gallon : ℕ) (gallons_used : ℕ) (distance_traveled : ℕ) 
  (h1 : miles_per_gallon = 20) (h2 : gallons_used = 3) : 
  distance_traveled = 60 :=
by
  sorry

end Brians_trip_distance_l135_135751


namespace geometric_sequence_min_l135_135111

theorem geometric_sequence_min (a : ℕ → ℝ) (q : ℝ)
  (h_positive : ∀ n, a n > 0)
  (h_condition : 2 * (a 4) + (a 3) - 2 * (a 2) - (a 1) = 8)
  (h_geometric : ∀ n, a (n+1) = a n * q) :
  ∃ min_val, min_val = 12 * Real.sqrt 3 ∧ min_val = 2 * (a 5) + (a 4) :=
sorry

end geometric_sequence_min_l135_135111


namespace min_sum_of_gcd_and_lcm_eq_three_times_sum_l135_135890

theorem min_sum_of_gcd_and_lcm_eq_three_times_sum (a b d : ℕ) (h1 : d = Nat.gcd a b)
  (h2 : Nat.gcd a b + Nat.lcm a b = 3 * (a + b)) :
  a + b = 12 :=
by
sorry

end min_sum_of_gcd_and_lcm_eq_three_times_sum_l135_135890


namespace related_sequence_exists_l135_135475

theorem related_sequence_exists :
  ∃ b : Fin 5 → ℕ, b = ![11, 10, 9, 8, 7] :=
by
  let a : Fin 5 → ℕ := ![1, 5, 9, 13, 17]
  let b : Fin 5 → ℕ := ![
    (a 0 + a 1 + a 2 + a 3 + a 4 - a 0) / 4,
    (a 0 + a 1 + a 2 + a 3 + a 4 - a 1) / 4,
    (a 0 + a 1 + a 2 + a 3 + a 4 - a 2) / 4,
    (a 0 + a 1 + a 2 + a 3 + a 4 - a 3) / 4,
    (a 0 + a 1 + a 2 + a 3 + a 4 - a 4) / 4
  ]
  existsi b
  sorry

end related_sequence_exists_l135_135475


namespace christine_needs_32_tablespoons_l135_135203

-- Define the conditions
def tablespoons_per_egg_white : ℕ := 2
def egg_whites_per_cake : ℕ := 8
def number_of_cakes : ℕ := 2

-- Define the calculation for total tablespoons of aquafaba needed
def total_tbs_aquafaba : ℕ :=
  tablespoons_per_egg_white * (egg_whites_per_cake * number_of_cakes)

-- The theorem to prove
theorem christine_needs_32_tablespoons :
  total_tbs_aquafaba = 32 :=
by 
  -- Placeholder for proof, as proof steps are not required
  sorry

end christine_needs_32_tablespoons_l135_135203


namespace ratio_correct_l135_135181

def cost_of_flasks := 150
def remaining_budget := 25
def total_budget := 325
def spent_budget := total_budget - remaining_budget
def cost_of_test_tubes := 100
def cost_of_safety_gear := cost_of_test_tubes / 2
def ratio_test_tubes_flasks := cost_of_test_tubes / cost_of_flasks

theorem ratio_correct :
  spent_budget = cost_of_flasks + cost_of_test_tubes + cost_of_safety_gear → 
  ratio_test_tubes_flasks = 2 / 3 :=
by
  sorry

end ratio_correct_l135_135181


namespace probability_edge_within_five_hops_l135_135260

def is_edge_square (n : ℕ) (coord : ℕ × ℕ) : Prop := 
  coord.1 = 1 ∨ coord.1 = n ∨ coord.2 = 1 ∨ coord.2 = n

def is_central_square (coord : ℕ × ℕ) : Prop :=
  (coord = (2, 2)) ∨ (coord = (2, 3)) ∨ (coord = (3, 2)) ∨ (coord = (3, 3))

noncomputable def probability_of_edge_in_n_hops (n : ℕ) : ℚ := sorry

theorem probability_edge_within_five_hops : probability_of_edge_in_n_hops 4 = 7 / 8 :=
sorry

end probability_edge_within_five_hops_l135_135260


namespace mean_of_quadrilateral_angles_is_90_l135_135458

theorem mean_of_quadrilateral_angles_is_90 :
  let n := 4 in
  let sum_of_interior_angles (n : ℕ) := (n - 2) * 180 in
  let quadrilateral_interior_angle_sum := sum_of_interior_angles 4 in
  quadrilateral_interior_angle_sum = 360 →
  360 / 4 = 90 :=
by
  intros n sum_of_interior_angles quadrilateral_interior_angle_sum h
  exact sorry

end mean_of_quadrilateral_angles_is_90_l135_135458


namespace perimeter_C_l135_135286

theorem perimeter_C (x y : ℕ) 
  (h1 : 6 * x + 2 * y = 56)
  (h2 : 4 * x + 6 * y = 56) :
  2 * x + 6 * y = 40 := 
by
  sorry

end perimeter_C_l135_135286


namespace rectangular_solid_volume_l135_135740

theorem rectangular_solid_volume 
  (a b c : ℝ) 
  (h1 : a * b = 18) 
  (h2 : b * c = 50) 
  (h3 : a * c = 45) : 
  a * b * c = 150 * Real.sqrt 3 := 
by 
  sorry

end rectangular_solid_volume_l135_135740


namespace cafeteria_dish_problem_l135_135697

theorem cafeteria_dish_problem (a : ℕ → ℕ) (a_1 : a 1 = 428)
    (h : ∀ n, a (n + 1) = (a n)/2 + 150) : a 8 = 301 :=
by 
    sorry

end cafeteria_dish_problem_l135_135697


namespace mean_value_interior_angles_quadrilateral_l135_135375

theorem mean_value_interior_angles_quadrilateral :
  let n := 4 in
  (n - 2) * 180 / n = 90 :=
by
  let n := 4
  sorry

end mean_value_interior_angles_quadrilateral_l135_135375


namespace men_work_problem_l135_135782

theorem men_work_problem (x : ℕ) (h1 : x * 70 = 40 * 63) : x = 36 := 
by
  sorry

end men_work_problem_l135_135782


namespace value_of_x_squared_y_plus_xy_squared_l135_135945

variable {R : Type} [CommRing R] (x y : R)

-- Given conditions
def cond1 : Prop := x + y = 3
def cond2 : Prop := x * y = 2

-- The main theorem to prove
theorem value_of_x_squared_y_plus_xy_squared (h1 : cond1 x y) (h2 : cond2 x y) : x^2 * y + x * y^2 = 6 :=
by
  sorry

end value_of_x_squared_y_plus_xy_squared_l135_135945


namespace simplify_expression_l135_135027

theorem simplify_expression (x y : ℝ) :
  3 * x + 6 * x + 9 * x + 12 * y + 15 * y + 18 + 21 = 18 * x + 27 * y + 39 :=
by
  sorry

end simplify_expression_l135_135027


namespace pages_left_l135_135020

-- Define the conditions
def initial_books := 10
def pages_per_book := 100
def books_lost := 2

-- The total pages Phil had initially
def initial_pages := initial_books * pages_per_book

-- The number of books left after losing some during the move
def books_left := initial_books - books_lost

-- Prove the number of pages worth of books Phil has left
theorem pages_left : books_left * pages_per_book = 800 := by
  sorry

end pages_left_l135_135020


namespace find_f_2011_l135_135888

open Function

variable {R : Type} [Field R]

def functional_equation (f : R → R) : Prop :=
  ∀ a b : R, f (a * f b) = a * b

theorem find_f_2011 (f : ℝ → ℝ) (h : functional_equation f) : f 2011 = 2011 :=
sorry

end find_f_2011_l135_135888


namespace figure_C_perimeter_l135_135306

def is_perimeter (figure : Type) (perimeter : ℕ) : Prop :=
∃ x y : ℕ, (figure = 'A' → 6*x + 2*y = perimeter) ∧ 
           (figure = 'B' → 4*x + 6*y = perimeter) ∧
           (figure = 'C' → 2*x + 6*y = perimeter)

theorem figure_C_perimeter (hA : is_perimeter 'A' 56) (hB : is_perimeter 'B' 56) : 
  is_perimeter 'C' 40 :=
by
  sorry

end figure_C_perimeter_l135_135306


namespace area_of_rectangle_l135_135561

theorem area_of_rectangle (l w : ℝ) (h_perimeter : 2 * (l + w) = 126) (h_difference : l - w = 37) : l * w = 650 :=
sorry

end area_of_rectangle_l135_135561


namespace least_value_is_one_l135_135881

noncomputable def least_possible_value (x y : ℝ) : ℝ := (x^2 * y - 1)^2 + (x^2 + y)^2

theorem least_value_is_one : ∀ x y : ℝ, (least_possible_value x y) ≥ 1 :=
by
  sorry

end least_value_is_one_l135_135881


namespace total_payment_l135_135802

/--
  Jerry took 8 hours painting the house. 
  The time to fix the kitchen counter was three times longer than painting the house.
  Jerry took 6 hours mowing the lawn.
  Jerry charged $15 per hour of work.
  Prove that the total amount of money Miss Stevie paid Jerry is $570.
-/
theorem total_payment (h_paint: ℕ := 8) (h_counter: ℕ := 3 * h_paint) (h_mow: ℕ := 6) (rate: ℕ := 15) :
  let total_hours := h_paint + h_counter + h_mow
  in total_hours * rate = 570 :=
by
  -- provide proof here
  sorry

end total_payment_l135_135802


namespace problem_I_problem_II_l135_135228

noncomputable def f (a x : ℝ) : ℝ := x^2 - (2 * a + 1) * x + a * Real.log x
noncomputable def g (a x : ℝ) : ℝ := (1 - a) * x
noncomputable def h (x : ℝ) : ℝ := (x^2 - 2 * x) / (x - Real.log x)

theorem problem_I (a : ℝ) (ha : a > 1 / 2) :
  (∀ x : ℝ, 0 < x ∧ x < 1 / 2 → deriv (f a) x > 0) ∧
  (∀ x : ℝ, 1 / 2 < x ∧ x < a → deriv (f a) x < 0) ∧
  (∀ x : ℝ, a < x → deriv (f a) x > 0) :=
sorry

theorem problem_II (a : ℝ) :
  (∃ x₀ : ℝ, 1 ≤ x₀ ∧ x₀ ≤ Real.exp 1 ∧ f a x₀ ≥ g a x₀) ↔ a ≤ (Real.exp 1 * (Real.exp 1 - 2)) / (Real.exp 1 - 1) :=
sorry

end problem_I_problem_II_l135_135228


namespace difference_in_students_specific_case_diff_l135_135242

-- Define the variables and conditions
variables (a b : ℕ)

-- Condition: a > b
axiom h1 : a > b

-- Definition of eighth grade students
def eighth_grade_students := (3 * a + b) * (2 * a + 2 * b)

-- Definition of seventh grade students
def seventh_grade_students := (2 * (a + b)) ^ 2

-- Theorem for the difference in the number of students
theorem difference_in_students : (eighth_grade_students a b) - (seventh_grade_students a b) = 2 * a^2 - 2 * b^2 :=
sorry

-- Theorem for the specific example when a = 10 and b = 2
theorem specific_case_diff : eighth_grade_students 10 2 - seventh_grade_students 10 2 = 192 :=
sorry

end difference_in_students_specific_case_diff_l135_135242


namespace number_of_paperback_books_l135_135748

variables (P H : ℕ)

theorem number_of_paperback_books (h1 : H = 4) (h2 : P / 3 + 2 * H = 10) : P = 6 := 
by
  sorry

end number_of_paperback_books_l135_135748


namespace force_of_water_pressure_on_plate_l135_135072

noncomputable def force_on_plate_under_water (γ : ℝ) (g : ℝ) (a b : ℝ) : ℝ :=
  γ * g * (b^2 - a^2) / 2

theorem force_of_water_pressure_on_plate :
  let γ : ℝ := 1000 -- kg/m^3
  let g : ℝ := 9.81  -- m/s^2
  let a : ℝ := 0.5   -- top depth
  let b : ℝ := 2.5   -- bottom depth
  force_on_plate_under_water γ g a b = 29430 := sorry

end force_of_water_pressure_on_plate_l135_135072


namespace rowing_speed_downstream_correct_l135_135897

/-- Given:
- The speed of the man upstream V_upstream is 20 kmph.
- The speed of the man in still water V_man is 40 kmph.
Prove:
- The speed of the man rowing downstream V_downstream is 60 kmph.
-/
def rowing_speed_downstream : Prop :=
  let V_upstream := 20
  let V_man := 40
  let V_s := V_man - V_upstream
  let V_downstream := V_man + V_s
  V_downstream = 60

theorem rowing_speed_downstream_correct : rowing_speed_downstream := by
  sorry

end rowing_speed_downstream_correct_l135_135897


namespace sequence_general_term_l135_135244

noncomputable def sequence_condition (a : ℕ → ℝ) : Prop :=
  (∀ n, 2 * a n = 3 * a (n + 1)) ∧ 
  (a 2 * a 5 = 8 / 27) ∧ 
  (∀ n, 0 < a n)

theorem sequence_general_term (a : ℕ → ℝ) (h : sequence_condition a) : 
  ∀ n, a n = (2 / 3)^(n - 2) :=
by 
  sorry

end sequence_general_term_l135_135244


namespace doubled_money_is_1_3_l135_135903

-- Define the amounts of money Alice and Bob have
def alice_money := (2 : ℚ) / 5
def bob_money := (1 : ℚ) / 4

-- Define the total money before doubling
def total_money_before_doubling := alice_money + bob_money

-- Define the total money after doubling
def total_money_after_doubling := 2 * total_money_before_doubling

-- State the proposition to prove
theorem doubled_money_is_1_3 : total_money_after_doubling = 1.3 := by
  -- The proof will be filled in here
  sorry

end doubled_money_is_1_3_l135_135903


namespace power_mod_2040_l135_135468

theorem power_mod_2040 : (6^2040) % 13 = 1 := by
  -- Skipping the proof as the problem only requires the statement
  sorry

end power_mod_2040_l135_135468


namespace g_five_eq_248_l135_135839

-- We define g, and assume it meets the conditions described.
variable (g : ℤ → ℤ)

-- Condition 1: g(1) > 1
axiom g_one_gt_one : g 1 > 1

-- Condition 2: Functional equation for g
axiom g_funct_eq (x y : ℤ) : g (x + y) + x * g y + y * g x = g x * g y + x + y + x * y

-- Condition 3: Recursive relationship for g
axiom g_recur_eq (x : ℤ) : 3 * g x = g (x + 1) + 2 * x - 1

-- Theorem we want to prove
theorem g_five_eq_248 : g 5 = 248 := by
  sorry

end g_five_eq_248_l135_135839


namespace perimeter_C_l135_135291

theorem perimeter_C :
  ∀ (x y : ℕ),
  (6 * x + 2 * y = 56) →
  (4 * x + 6 * y = 56) →
  (2 * x + 6 * y = 40) :=
by
  intros x y hA hB
  sorry

end perimeter_C_l135_135291
