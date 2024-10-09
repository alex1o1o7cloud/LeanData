import Mathlib

namespace max_y_difference_l479_47946

noncomputable def f (x : ℝ) : ℝ := 4 - x^2 + x^3
noncomputable def g (x : ℝ) : ℝ := 2 + x^2 + x^3

theorem max_y_difference : 
  ∃ x1 x2 : ℝ, 
    f x1 = g x1 ∧ f x2 = g x2 ∧ 
    (∀ x : ℝ, f x = g x → x = x1 ∨ x = x2) ∧ 
    abs ((f x1) - (f x2)) = 2 := 
by
  sorry

end max_y_difference_l479_47946


namespace probability_of_point_in_spheres_l479_47964

noncomputable def radius_of_inscribed_sphere (R : ℝ) : ℝ := 2 * R / 3
noncomputable def radius_of_tangent_spheres (R : ℝ) : ℝ := 2 * R / 3

theorem probability_of_point_in_spheres
  (R : ℝ)  -- Radius of the circumscribed sphere
  (r : ℝ := radius_of_inscribed_sphere R)  -- Radius of the inscribed sphere
  (r_t : ℝ := radius_of_tangent_spheres R)  -- Radius of each tangent sphere
  (volume : ℝ := 4/3 * Real.pi * r^3)  -- Volume of each smaller sphere
  (total_small_volume : ℝ := 5 * volume)  -- Total volume of smaller spheres
  (circumsphere_volume : ℝ := 4/3 * Real.pi * (2 * R)^3)  -- Volume of the circumscribed sphere
  : 
  total_small_volume / circumsphere_volume = 5 / 27 :=
by
  sorry

end probability_of_point_in_spheres_l479_47964


namespace multiple_of_Roseville_population_l479_47933

noncomputable def Willowdale_population : ℕ := 2000

noncomputable def Roseville_population : ℕ :=
  (3 * Willowdale_population) - 500

noncomputable def SunCity_population : ℕ := 12000

theorem multiple_of_Roseville_population :
  ∃ m : ℕ, SunCity_population = (m * Roseville_population) + 1000 ∧ m = 2 :=
by
  sorry

end multiple_of_Roseville_population_l479_47933


namespace supplement_of_complementary_angle_of_35_deg_l479_47908

theorem supplement_of_complementary_angle_of_35_deg :
  let A := 35
  let C := 90 - A
  let S := 180 - C
  S = 125 :=
by
  let A := 35
  let C := 90 - A
  let S := 180 - C
  -- we need to prove S = 125
  sorry

end supplement_of_complementary_angle_of_35_deg_l479_47908


namespace original_number_increased_l479_47991

theorem original_number_increased (x : ℝ) (h : (1.10 * x) * 1.15 = 632.5) : x = 500 :=
sorry

end original_number_increased_l479_47991


namespace find_x_l479_47918

noncomputable def a : ℝ × ℝ := (2, 1)
noncomputable def b (x : ℝ) : ℝ × ℝ := (x, 2)
noncomputable def vec_add (v w : ℝ × ℝ) : ℝ × ℝ := (v.1 + w.1, v.2 + w.2)
noncomputable def scalar_vec_mul (c : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (c * v.1, c * v.2)
noncomputable def vec_sub (v w : ℝ × ℝ) : ℝ × ℝ := (v.1 - w.1, v.2 - w.2)

theorem find_x (x : ℝ) :
  (vec_add a (b x)).1 * (vec_sub a (scalar_vec_mul 2 (b x))).2 =
  (vec_add a (b x)).2 * (vec_sub a (scalar_vec_mul 2 (b x))).1 →
  x = 4 :=
by sorry

end find_x_l479_47918


namespace PQ_length_l479_47915

theorem PQ_length (BC AD : ℝ) (angle_A angle_D : ℝ) (P Q : ℝ) 
  (H1 : BC = 700) (H2 : AD = 1400) (H3 : angle_A = 45) (H4 : angle_D = 45) 
  (mid_BC : P = BC / 2) (mid_AD : Q = AD / 2) :
  abs (Q - P) = 350 :=
by
  sorry

end PQ_length_l479_47915


namespace expand_polynomials_l479_47984

def p (z : ℝ) : ℝ := 3 * z ^ 2 + 4 * z - 7
def q (z : ℝ) : ℝ := 4 * z ^ 3 - 3 * z + 2

theorem expand_polynomials :
  (p z) * (q z) = 12 * z ^ 5 + 16 * z ^ 4 - 37 * z ^ 3 - 6 * z ^ 2 + 29 * z - 14 := by
  sorry

end expand_polynomials_l479_47984


namespace hours_per_day_l479_47968

-- Conditions
def days_worked : ℝ := 3
def total_hours_worked : ℝ := 7.5

-- Theorem to prove the number of hours worked each day
theorem hours_per_day : total_hours_worked / days_worked = 2.5 :=
by
  sorry

end hours_per_day_l479_47968


namespace find_c_l479_47995

-- Definitions based on the conditions in the problem
def is_vertex (h k : ℝ) := (5, 1) = (h, k)
def passes_through (x y : ℝ) := (2, 3) = (x, y)

-- Lean theorem statement
theorem find_c (a b c : ℝ) (h k x y : ℝ) (hv : is_vertex h k) (hp : passes_through x y)
  (heq : ∀ y, x = a * y^2 + b * y + c) : c = 17 / 4 :=
by
  sorry

end find_c_l479_47995


namespace find_1993_star_1935_l479_47923

axiom star (x y : ℕ) : ℕ

axiom star_self {x : ℕ} : star x x = 0
axiom star_assoc {x y z : ℕ} : star x (star y z) = star x y + z

theorem find_1993_star_1935 : star 1993 1935 = 58 :=
by
  sorry

end find_1993_star_1935_l479_47923


namespace inequality_proof_l479_47982

theorem inequality_proof (x y : ℝ) (h1 : y ≥ 0) (h2 : y * (y + 1) ≤ (x + 1)^2) : y * (y - 1) ≤ x^2 :=
sorry

end inequality_proof_l479_47982


namespace mode_and_median_of_survey_l479_47902

/-- A data structure representing the number of students corresponding to each sleep time. -/
structure SleepSurvey :=
  (time7 : ℕ)
  (time8 : ℕ)
  (time9 : ℕ)
  (time10 : ℕ)

def survey : SleepSurvey := { time7 := 6, time8 := 9, time9 := 11, time10 := 4 }

theorem mode_and_median_of_survey (s : SleepSurvey) :
  (mode=9 ∧ median = 8.5) :=
by
  -- proof would go here
  sorry

end mode_and_median_of_survey_l479_47902


namespace probability_of_picking_letter_from_mathematics_l479_47990

-- Definition of the problem conditions
def extended_alphabet_size := 30
def distinct_letters_in_mathematics := 8

-- Theorem statement
theorem probability_of_picking_letter_from_mathematics :
  (distinct_letters_in_mathematics / extended_alphabet_size : ℚ) = 4 / 15 := 
by 
  sorry

end probability_of_picking_letter_from_mathematics_l479_47990


namespace proposition_4_l479_47929

theorem proposition_4 (x y ε : ℝ) (h1 : |x - 2| < ε) (h2 : |y - 2| < ε) : |x - y| < 2 * ε :=
by
  sorry

end proposition_4_l479_47929


namespace remainder_of_expression_l479_47973

theorem remainder_of_expression (n : ℤ) (h : n % 8 = 3) : (4 * n - 10) % 8 = 2 :=
sorry

end remainder_of_expression_l479_47973


namespace division_by_repeating_decimal_l479_47993

-- Define the repeating decimal as a fraction
def repeating_decimal := 4 / 9

-- Prove the main theorem
theorem division_by_repeating_decimal : 8 / repeating_decimal = 18 :=
by
  -- lean implementation steps
  sorry

end division_by_repeating_decimal_l479_47993


namespace speed_of_first_car_l479_47966

theorem speed_of_first_car 
  (distance_highway : ℕ)
  (time_to_meet : ℕ)
  (speed_second_car : ℕ)
  (total_distance_covered : distance_highway = time_to_meet * 40 + time_to_meet * speed_second_car): 
  5 * 40 + 5 * 60 = distance_highway := 
by
  /-
    Given:
      - distance_highway : ℕ (The length of the highway, which is 500 miles)
      - time_to_meet : ℕ (The time after which the two cars meet, which is 5 hours)
      - speed_second_car : ℕ (The speed of the second car, which is 60 mph)
      - total_distance_covered : distance_highway = time_to_meet * speed_of_first_car + time_to_meet * speed_second_car

    We need to prove:
      - 5 * 40 + 5 * 60 = distance_highway
  -/

  sorry

end speed_of_first_car_l479_47966


namespace hexagon_area_l479_47961

-- Define the area of a triangle
def triangle_area (base height: ℝ) : ℝ := 0.5 * base * height

-- Given dimensions for each triangle
def base_unit := 1
def original_height := 3
def new_height := 4

-- Calculate areas of each triangle in the new configuration
def single_triangle_area := triangle_area base_unit new_height
def total_triangle_area := 4 * single_triangle_area

-- The area of the rectangular region formed by the hexagon and triangles
def rectangular_region_area := (base_unit + original_height + original_height) * new_height

-- Prove the area of the hexagon
theorem hexagon_area : rectangular_region_area - total_triangle_area = 32 :=
by
  -- We will provide the proof here
  sorry

end hexagon_area_l479_47961


namespace unique_solution_p_l479_47971

theorem unique_solution_p (p : ℚ) :
  (∀ x : ℝ, (2 * x + 3) / (p * x - 2) = x) ↔ p = -4 / 3 := sorry

end unique_solution_p_l479_47971


namespace min_value_of_expression_l479_47953

theorem min_value_of_expression (x y : ℝ) (hx : x > y) (hy : y > 0) (hxy : x + y ≤ 2) :
  ∃ m : ℝ, m = (2 / (x + 3 * y) + 1 / (x - y)) ∧ m = (3 + 2 * Real.sqrt 2) / 4 :=
by
  sorry

end min_value_of_expression_l479_47953


namespace max_distance_from_ellipse_to_line_l479_47941

theorem max_distance_from_ellipse_to_line :
  let ellipse (x y : ℝ) := (x^2 / 16) + (y^2 / 4) = 1
  let line (x y : ℝ) := x + 2 * y - Real.sqrt 2 = 0
  ∃ (d : ℝ), (∀ (x y : ℝ), ellipse x y → line x y → d = Real.sqrt 10) :=
sorry

end max_distance_from_ellipse_to_line_l479_47941


namespace vanessa_scored_27_points_l479_47904

variable (P : ℕ) (number_of_players : ℕ) (average_points_per_player : ℚ) (vanessa_points : ℕ)

axiom team_total_points : P = 48
axiom other_players : number_of_players = 6
axiom average_points_per_other_player : average_points_per_player = 3.5

theorem vanessa_scored_27_points 
  (h1 : P = 48)
  (h2 : number_of_players = 6)
  (h3 : average_points_per_player = 3.5)
: vanessa_points = 27 :=
sorry

end vanessa_scored_27_points_l479_47904


namespace simplify_complex_fraction_l479_47903

theorem simplify_complex_fraction : 
  ∀ (i : ℂ), 
  i^2 = -1 → 
  (2 - 2 * i) / (3 + 4 * i) = -(2 / 25 : ℝ) - (14 / 25) * i :=
by
  intros
  sorry

end simplify_complex_fraction_l479_47903


namespace area_of_rectangle_l479_47956

noncomputable def length := 44.4
noncomputable def width := 29.6

theorem area_of_rectangle (h1 : width = 2 / 3 * length) (h2 : 2 * (length + width) = 148) : 
  (length * width) = 1314.24 := 
by 
  sorry

end area_of_rectangle_l479_47956


namespace pencil_eraser_cost_l479_47942

theorem pencil_eraser_cost (p e : ℕ) (h_eq : 10 * p + 4 * e = 120) (h_gt : p > e) : p + e = 15 :=
by sorry

end pencil_eraser_cost_l479_47942


namespace find_second_number_l479_47944

variable (n : ℕ)

theorem find_second_number (h : 8000 * n = 480 * 10^5) : n = 6000 :=
by
  sorry

end find_second_number_l479_47944


namespace tommys_family_members_l479_47916

-- Definitions
def ounces_per_member : ℕ := 16
def ounces_per_steak : ℕ := 20
def steaks_needed : ℕ := 4

-- Theorem statement
theorem tommys_family_members : (steaks_needed * ounces_per_steak) / ounces_per_member = 5 :=
by
  -- Proof goes here
  sorry

end tommys_family_members_l479_47916


namespace sum_real_imaginary_part_l479_47965

noncomputable def imaginary_unit : ℂ := Complex.I

theorem sum_real_imaginary_part {z : ℂ} (h : z * imaginary_unit = 1 + imaginary_unit) :
  z.re + z.im = 2 := 
sorry

end sum_real_imaginary_part_l479_47965


namespace finite_pos_int_set_condition_l479_47960

theorem finite_pos_int_set_condition (X : Finset ℕ) 
  (hX : ∀ a ∈ X, 0 < a) 
  (h2 : 2 ≤ X.card) 
  (hcond : ∀ {a b : ℕ}, a ∈ X → b ∈ X → a > b → b^2 / (a - b) ∈ X) :
  ∃ a : ℕ, X = {a, 2 * a} :=
by
  sorry

end finite_pos_int_set_condition_l479_47960


namespace hiring_probability_l479_47989

noncomputable def combinatorics (n k : ℕ) : ℕ := Nat.choose n k

theorem hiring_probability (n : ℕ) (h1 : combinatorics 2 2 = 1)
                          (h2 : combinatorics (n - 2) 1 = n - 2)
                          (h3 : combinatorics n 3 = n * (n - 1) * (n - 2) / 6)
                          (h4 : (6 : ℕ) / (n * (n - 1) : ℚ) = 1 / 15) :
  n = 10 :=
by
  sorry

end hiring_probability_l479_47989


namespace best_shooter_l479_47906

noncomputable def avg_A : ℝ := 9
noncomputable def avg_B : ℝ := 8
noncomputable def avg_C : ℝ := 9
noncomputable def avg_D : ℝ := 9

noncomputable def var_A : ℝ := 1.2
noncomputable def var_B : ℝ := 0.4
noncomputable def var_C : ℝ := 1.8
noncomputable def var_D : ℝ := 0.4

theorem best_shooter :
  (avg_A = 9 ∧ var_A = 1.2) →
  (avg_B = 8 ∧ var_B = 0.4) →
  (avg_C = 9 ∧ var_C = 1.8) →
  (avg_D = 9 ∧ var_D = 0.4) →
  avg_D = 9 ∧ var_D = 0.4 :=
by {
  sorry
}

end best_shooter_l479_47906


namespace length_ratio_is_correct_width_ratio_is_correct_l479_47954

-- Definitions based on the conditions
def room_length : ℕ := 25
def room_width : ℕ := 15

-- Calculated perimeter
def room_perimeter : ℕ := 2 * (room_length + room_width)

-- Ratios to be proven
def length_to_perimeter_ratio : ℚ := room_length / room_perimeter
def width_to_perimeter_ratio : ℚ := room_width / room_perimeter

-- Stating the theorems to be proved
theorem length_ratio_is_correct : length_to_perimeter_ratio = 5 / 16 :=
by sorry

theorem width_ratio_is_correct : width_to_perimeter_ratio = 3 / 16 :=
by sorry

end length_ratio_is_correct_width_ratio_is_correct_l479_47954


namespace find_second_number_l479_47930

theorem find_second_number 
  (x y z : ℕ)
  (h1 : x + y + z = 120)
  (h2 : x = (3 * y) / 4)
  (h3 : z = (9 * y) / 7) : 
  y = 40 :=
sorry

end find_second_number_l479_47930


namespace sum_of_angles_eq_62_l479_47958

noncomputable def Φ (x : ℝ) : ℝ := Real.sin x
noncomputable def Ψ (x : ℝ) : ℝ := Real.cos x
def θ : List ℝ := [31, 30, 1, 0]

theorem sum_of_angles_eq_62 :
  θ.sum = 62 := by
  sorry

end sum_of_angles_eq_62_l479_47958


namespace compare_abc_l479_47938

noncomputable def a := Real.sin (15 * Real.pi / 180) * Real.cos (15 * Real.pi / 180)
noncomputable def b := Real.cos (Real.pi / 6) ^ 2 - Real.sin (Real.pi / 6) ^ 2
noncomputable def c := Real.tan (30 * Real.pi / 180) / (1 - Real.tan (30 * Real.pi / 180) ^ 2)

theorem compare_abc : a < b ∧ b < c :=
by
  sorry

end compare_abc_l479_47938


namespace remainder_777_777_mod_13_l479_47999

theorem remainder_777_777_mod_13 : (777 ^ 777) % 13 = 12 := 
by 
  -- Proof steps would go here
  sorry

end remainder_777_777_mod_13_l479_47999


namespace total_games_in_conference_l479_47943

-- Definitions based on the conditions
def numTeams := 16
def divisionTeams := 8
def gamesWithinDivisionPerTeam := 21
def gamesAcrossDivisionPerTeam := 16
def totalGamesPerTeam := 37
def totalGameCount := 592
def actualGameCount := 296

-- Proof statement
theorem total_games_in_conference : actualGameCount = (totalGameCount / 2) :=
  by sorry

end total_games_in_conference_l479_47943


namespace problem_statement_l479_47992

noncomputable def G (x : ℝ) : ℝ := ((x + 1) ^ 2) / 2 - 4

theorem problem_statement : G (G (G 0)) = -3.9921875 :=
by
  sorry

end problem_statement_l479_47992


namespace eleven_power_five_mod_nine_l479_47934

theorem eleven_power_five_mod_nine : ∃ n : ℕ, (11^5 ≡ n [MOD 9]) ∧ (0 ≤ n ∧ n < 9) ∧ (n = 5) := 
  by 
    sorry

end eleven_power_five_mod_nine_l479_47934


namespace power_division_l479_47952

theorem power_division : (19^11 / 19^6 = 247609) := sorry

end power_division_l479_47952


namespace find_number_of_members_l479_47922

variable (n : ℕ)

-- We translate the conditions into Lean 4 definitions
def total_collection := 9216
def per_member_contribution := n

-- The goal is to prove that n = 96 given the total collection
theorem find_number_of_members (h : n * n = total_collection) : n = 96 := 
sorry

end find_number_of_members_l479_47922


namespace differential_equation_solution_l479_47914

theorem differential_equation_solution (x y : ℝ) (C : ℝ) :
  (∀ dx dy, 2 * x * y * dx + x^2 * dy = 0) → x^2 * y = C :=
sorry

end differential_equation_solution_l479_47914


namespace probability_wheel_l479_47986

theorem probability_wheel (P : ℕ → ℚ) 
  (hA : P 0 = 1/4) 
  (hB : P 1 = 1/3) 
  (hC : P 2 = 1/6) 
  (hSum : P 0 + P 1 + P 2 + P 3 = 1) : 
  P 3 = 1/4 := 
by 
  -- Proof here
  sorry

end probability_wheel_l479_47986


namespace quadrilateral_is_square_l479_47963

-- Define a structure for a quadrilateral with side lengths and diagonal lengths
structure Quadrilateral :=
  (side_a side_b side_c side_d diag_e diag_f : ℝ)

-- Define what it means for a quadrilateral to be a square
def is_square (quad : Quadrilateral) : Prop :=
  quad.side_a = quad.side_b ∧ 
  quad.side_b = quad.side_c ∧ 
  quad.side_c = quad.side_d ∧  
  quad.diag_e = quad.diag_f

-- Define the problem to prove that the given quadrilateral is a square given the conditions
theorem quadrilateral_is_square (quad : Quadrilateral) 
  (h_sides : quad.side_a = quad.side_b ∧ 
             quad.side_b = quad.side_c ∧ 
             quad.side_c = quad.side_d)
  (h_diagonals : quad.diag_e = quad.diag_f) :
  is_square quad := 
  by
  -- This is where the proof would go
  sorry

end quadrilateral_is_square_l479_47963


namespace line_equation_l479_47985

theorem line_equation {m : ℤ} :
  (∀ x y : ℤ, 2 * x + y + m = 0) →
  (∀ x y : ℤ, 2 * x + y - 10 = 0) →
  (2 * 1 + 0 + m = 0) →
  m = -2 :=
by
  sorry

end line_equation_l479_47985


namespace find_n_l479_47912

theorem find_n (n : ℤ) (hn_range : -150 < n ∧ n < 150) (h_tan : Real.tan (n * Real.pi / 180) = Real.tan (286 * Real.pi / 180)) : 
  n = -74 :=
sorry

end find_n_l479_47912


namespace ducks_at_Lake_Michigan_l479_47926

variable (D : ℕ)

def ducks_condition := 2 * D + 6 = 206

theorem ducks_at_Lake_Michigan (h : ducks_condition D) : D = 100 :=
by
  sorry

end ducks_at_Lake_Michigan_l479_47926


namespace probability_even_sum_l479_47979

def p_even_first_wheel : ℚ := 1 / 3
def p_odd_first_wheel : ℚ := 2 / 3
def p_even_second_wheel : ℚ := 3 / 5
def p_odd_second_wheel : ℚ := 2 / 5

theorem probability_even_sum : 
  (p_even_first_wheel * p_even_second_wheel) + (p_odd_first_wheel * p_odd_second_wheel) = 7 / 15 :=
by
  sorry

end probability_even_sum_l479_47979


namespace min_value_l479_47998

noncomputable def minimum_value (a b c : ℝ) : ℝ :=
3 * a + 6 * b + 12 * c

theorem min_value (a b c : ℝ) (h : 9 * a ^ 2 + 4 * b ^ 2 + 36 * c ^ 2 = 4) :
  minimum_value a b c = -2 * Real.sqrt 14 := sorry

end min_value_l479_47998


namespace productivity_increase_l479_47939

/-- 
The original workday is 8 hours. 
During the first 6 hours, productivity is at the planned level (1 unit/hour). 
For the next 2 hours, productivity falls by 25% (0.75 units/hour). 
The workday is extended by 1 hour (now 9 hours). 
During the first 6 hours of the extended shift, productivity remains at the planned level (1 unit/hour). 
For the remaining 3 hours of the extended shift, productivity falls by 30% (0.7 units/hour). 
Prove that the overall productivity for the shift increased by 8% as a result of extending the workday.
-/
theorem productivity_increase
  (planned_productivity : ℝ)
  (initial_work_hours : ℝ)
  (initial_productivity_drop : ℝ)
  (extended_work_hours : ℝ)
  (extended_productivity_drop : ℝ)
  (initial_total_work : ℝ)
  (extended_total_work : ℝ)
  (percentage_increase : ℝ) :
  planned_productivity = 1 →
  initial_work_hours = 8 →
  initial_productivity_drop = 0.25 →
  extended_work_hours = 9 →
  extended_productivity_drop = 0.30 →
  initial_total_work = 7.5 →
  extended_total_work = 8.1 →
  percentage_increase = 8 →
  ((extended_total_work - initial_total_work) / initial_total_work * 100) = percentage_increase :=
sorry

end productivity_increase_l479_47939


namespace remainder_of_modified_division_l479_47987

theorem remainder_of_modified_division (x y u v : ℕ) (hx : 0 ≤ v ∧ v < y) (hxy : x = u * y + v) :
  ((x + 3 * u * y) % y) = v := by
  sorry

end remainder_of_modified_division_l479_47987


namespace red_balloon_probability_l479_47959

-- Define the conditions
def initial_red_balloons := 2
def initial_blue_balloons := 4
def inflated_red_balloons := 2
def inflated_blue_balloons := 2

-- Define the total number of balloons after inflation
def total_red_balloons := initial_red_balloons + inflated_red_balloons
def total_blue_balloons := initial_blue_balloons + inflated_blue_balloons
def total_balloons := total_red_balloons + total_blue_balloons

-- Define the probability calculation
def red_probability := (total_red_balloons : ℚ) / total_balloons * 100

-- The theorem to prove
theorem red_balloon_probability : red_probability = 40 := by
  sorry -- Skipping the proof itself

end red_balloon_probability_l479_47959


namespace not_possible_total_47_l479_47949

open Nat

theorem not_possible_total_47 (h c : ℕ) : ¬ (13 * h + 5 * c = 47) :=
  sorry

end not_possible_total_47_l479_47949


namespace bob_equals_alice_l479_47911

-- Define conditions as constants
def original_price : ℝ := 120.00
def tax_rate : ℝ := 0.08
def discount_rate : ℝ := 0.25

-- Bob's total calculation
def bob_total : ℝ := (original_price * (1 + tax_rate)) * (1 - discount_rate)

-- Alice's total calculation
def alice_total : ℝ := (original_price * (1 - discount_rate)) * (1 + tax_rate)

-- Theorem statement to be proved
theorem bob_equals_alice : bob_total = alice_total := by sorry

end bob_equals_alice_l479_47911


namespace animal_legs_l479_47962

theorem animal_legs (dogs chickens spiders octopus : Nat) (legs_dog legs_chicken legs_spider legs_octopus : Nat)
  (h1 : dogs = 3)
  (h2 : chickens = 4)
  (h3 : spiders = 2)
  (h4 : octopus = 1)
  (h5 : legs_dog = 4)
  (h6 : legs_chicken = 2)
  (h7 : legs_spider = 8)
  (h8 : legs_octopus = 8) :
  dogs * legs_dog + chickens * legs_chicken + spiders * legs_spider + octopus * legs_octopus = 44 := by
    sorry

end animal_legs_l479_47962


namespace present_age_ratio_l479_47951

theorem present_age_ratio (D J : ℕ) (h1 : Dan = 24) (h2 : James = 20) : Dan / James = 6 / 5 := by
  sorry

end present_age_ratio_l479_47951


namespace least_multiple_greater_than_500_l479_47994

theorem least_multiple_greater_than_500 : ∃ n : ℕ, n > 500 ∧ n % 32 = 0 := by
  let n := 512
  have h1 : n > 500 := by 
    -- proof omitted, as we're not solving the problem here
    sorry
  have h2 : n % 32 = 0 := by 
    -- proof omitted
    sorry
  exact ⟨n, h1, h2⟩

end least_multiple_greater_than_500_l479_47994


namespace a1_lt_a3_iff_an_lt_an1_l479_47917

-- Define arithmetic sequence and required properties
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop := 
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

variables (a : ℕ → ℝ)

-- Define the necessary and sufficient condition theorem
theorem a1_lt_a3_iff_an_lt_an1 (h_arith : is_arithmetic_sequence a) :
  (a 1 < a 3) ↔ (∀ n : ℕ, a n < a (n + 1)) :=
sorry

end a1_lt_a3_iff_an_lt_an1_l479_47917


namespace willie_initial_bananas_l479_47996

/-- Given that Willie will have 13 bananas, we need to prove that the initial number of bananas Willie had was some specific number X. --/
theorem willie_initial_bananas (initial_bananas : ℕ) (final_bananas : ℕ) 
    (h : final_bananas = 13) : initial_bananas = initial_bananas :=
by
  sorry

end willie_initial_bananas_l479_47996


namespace find_A_l479_47940

variable {A B C : ℚ}

theorem find_A (h1 : A = 1/2 * B) (h2 : B = 3/4 * C) (h3 : A + C = 55) : A = 15 :=
by
  sorry

end find_A_l479_47940


namespace find_n_l479_47967

theorem find_n :
  let a := (6 + 12 + 18 + 24 + 30 + 36 + 42) / 7
  let b := (2 * n : ℕ)
  (a*a - b*b = 0) -> (n = 12) := 
by 
  let a := 24
  let b := 2*n
  sorry

end find_n_l479_47967


namespace part1_part2_l479_47937

-- Definitions of sets A, B, and C
def setA : Set ℝ := { x | -2 ≤ x ∧ x ≤ 4 }
def setB : Set ℝ := { x | 1 < x ∧ x < 5 }
def setC (a : ℝ) : Set ℝ := { x | a - 1 < x ∧ x < 2 * a + 3 }

-- part (1)
theorem part1 (x : ℝ) : (x ∈ setA ∨ x ∈ setB) ↔ (-2 ≤ x ∧ x < 5) :=
sorry

-- part (2)
theorem part2 (a : ℝ) : ((setA ∩ setC a) = setC a) ↔ (a ≤ -4 ∨ (-1 ≤ a ∧ a ≤ 1/2)) :=
sorry

end part1_part2_l479_47937


namespace fraction_simplification_l479_47974

theorem fraction_simplification (x y z : ℝ) (h : x + y + z ≠ 0) :
  (x^2 + y^2 - z^2 + 2 * x * y) / (x^2 + z^2 - y^2 + 2 * x * z) = (x + y - z) / (x + z - y) :=
by
  sorry

end fraction_simplification_l479_47974


namespace bill_sunday_miles_l479_47900

variables (B J M S : ℝ)

-- Conditions
def condition_1 := B + 4
def condition_2 := 2 * (B + 4)
def condition_3 := J = 0 ∧ M = 5 ∧ (M + 2 = 7)
def condition_4 := (B + 5) + (B + 4) + 2 * (B + 4) + 7 = 50

-- The main theorem to prove the number of miles Bill ran on Sunday
theorem bill_sunday_miles (h1 : S = B + 4) (h2 : ∀ B, J = 0 → M = 5 → S + 2 = 7 → (B + 5) + S + 2 * S + 7 = 50) : S = 10.5 :=
by {
  sorry
}

end bill_sunday_miles_l479_47900


namespace mangoes_harvested_l479_47969

theorem mangoes_harvested (neighbors : ℕ) (mangoes_per_neighbor : ℕ) (total_mangoes_distributed : ℕ) (total_mangoes : ℕ) :
  neighbors = 8 ∧ mangoes_per_neighbor = 35 ∧ total_mangoes_distributed = neighbors * mangoes_per_neighbor ∧ total_mangoes = 2 * total_mangoes_distributed →
  total_mangoes = 560 :=
by {
  sorry
}

end mangoes_harvested_l479_47969


namespace expanded_figure_perimeter_l479_47932

def side_length : ℕ := 2
def bottom_row_squares : ℕ := 3
def total_squares : ℕ := 4

def perimeter (side_length : ℕ) (bottom_row_squares : ℕ) (total_squares: ℕ) : ℕ :=
  2 * side_length * (bottom_row_squares + 1)

theorem expanded_figure_perimeter : perimeter side_length bottom_row_squares total_squares = 20 :=
by
  sorry

end expanded_figure_perimeter_l479_47932


namespace solve_diophantine_equation_l479_47928

theorem solve_diophantine_equation :
  ∃ (x y : ℤ), x^2 - x * y - 6 * y^2 + 2 * x + 19 * y = 18 ∧ (x = 2 ∧ y = 2 ∨ x = -2 ∧ y = 2) :=
  sorry

end solve_diophantine_equation_l479_47928


namespace isosceles_trapezoid_base_ratio_correct_l479_47970

def isosceles_trapezoid_ratio (x y a b : ℝ) : Prop :=
  b = 2 * x ∧ a = 2 * y ∧ a + b = 10 ∧ (y * (Real.sqrt 2 + 1) = 5) →

  (a / b = (2 * (Real.sqrt 2) - 1) / 2)

theorem isosceles_trapezoid_base_ratio_correct: ∃ (x y a b : ℝ), 
  isosceles_trapezoid_ratio x y a b := sorry

end isosceles_trapezoid_base_ratio_correct_l479_47970


namespace price_of_kid_ticket_l479_47919

theorem price_of_kid_ticket (k a : ℤ) (hk : k = 6) (ha : a = 2)
  (price_kid price_adult : ℤ)
  (hprice_adult : price_adult = 2 * price_kid)
  (hcost_total : 6 * price_kid + 2 * price_adult = 50) :
  price_kid = 5 :=
by
  sorry

end price_of_kid_ticket_l479_47919


namespace Doris_needs_3_weeks_l479_47924

-- Definitions based on conditions
def hourly_wage : ℕ := 20
def monthly_expenses : ℕ := 1200
def weekday_hours_per_day : ℕ := 3
def saturdays_hours : ℕ := 5
def weekdays_per_week : ℕ := 5

-- Total hours per week
def total_hours_per_week := (weekday_hours_per_day * weekdays_per_week) + saturdays_hours

-- Weekly earnings
def weekly_earnings := hourly_wage * total_hours_per_week

-- Number of weeks needed for monthly expenses
def weeks_needed := monthly_expenses / weekly_earnings

-- Proposition to prove
theorem Doris_needs_3_weeks :
  weeks_needed = 3 := 
by
  sorry

end Doris_needs_3_weeks_l479_47924


namespace prime_sequence_constant_l479_47910

open Nat

-- Define a predicate for prime numbers
def is_prime (p : ℕ) : Prop := Nat.Prime p

-- Define the recurrence relation
def recurrence_relation (p : ℕ → ℕ) (k : ℤ) : Prop :=
  ∀ n : ℕ, p (n + 2) = p (n + 1) + p n + k

-- Define the proof problem
theorem prime_sequence_constant (p : ℕ → ℕ) (k : ℤ) : 
  (∀ n, is_prime (p n)) →
  recurrence_relation p k →
  ∃ (q : ℕ), is_prime q ∧ (∀ n, p n = q) ∧ k = -q :=
by
  -- Sorry proof here
  sorry

end prime_sequence_constant_l479_47910


namespace general_term_l479_47957

open Nat

noncomputable def S (n : ℕ) : ℕ := 2 * n^2 - 3 * n

theorem general_term (n : ℕ) (hn : n > 0) : (S n - S (n - 1)) = 4 * n - 5 := by
  sorry

end general_term_l479_47957


namespace total_value_is_84_l479_47945

-- Definitions based on conditions
def number_of_stamps : ℕ := 21
def value_of_7_stamps : ℕ := 28
def stamps_per_7 : ℕ := 7
def stamp_value : ℤ := value_of_7_stamps / stamps_per_7
def total_value_of_collection : ℤ := number_of_stamps * stamp_value

-- Statement to prove the total value of the stamp collection
theorem total_value_is_84 : total_value_of_collection = 84 := by
  sorry

end total_value_is_84_l479_47945


namespace remaining_family_member_age_l479_47988

variable (total_age father_age sister_age : ℕ) (remaining_member_age : ℕ)

def mother_age := father_age - 2
def brother_age := father_age / 2
def known_total_age := father_age + mother_age + brother_age + sister_age

theorem remaining_family_member_age : 
  total_age = 200 ∧ 
  father_age = 60 ∧ 
  sister_age = 40 ∧ 
  known_total_age = total_age - remaining_member_age → 
  remaining_member_age = 12 := by
  sorry

end remaining_family_member_age_l479_47988


namespace odd_function_condition_l479_47936

noncomputable def f (x a b : ℝ) : ℝ :=
  x * abs (x + a) + b

theorem odd_function_condition (a b : ℝ) :
  (∀ x : ℝ, f x a b = f (-x) a b) ↔ (a = 0 ∧ b = 0) := 
by
  sorry

end odd_function_condition_l479_47936


namespace irrational_roots_of_odd_coeff_quad_l479_47976

theorem irrational_roots_of_odd_coeff_quad (a b c : ℤ) (ha : a % 2 = 1) (hb : b % 2 = 1) (hc : c % 2 = 1) :
  ¬ ∃ r : ℚ, a * r^2 + b * r + c = 0 := 
sorry

end irrational_roots_of_odd_coeff_quad_l479_47976


namespace find_other_number_l479_47972

theorem find_other_number (LCM HCF num1 num2 : ℕ) 
  (h1 : LCM = 2310) 
  (h2 : HCF = 30) 
  (h3 : num1 = 330) 
  (h4 : LCM * HCF = num1 * num2) : 
  num2 = 210 := by 
  sorry

end find_other_number_l479_47972


namespace count_valid_x_satisfying_heartsuit_condition_l479_47925

def digit_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem count_valid_x_satisfying_heartsuit_condition :
  (∃ n, ∀ x, 1 ≤ x ∧ x < 1000 → digit_sum (digit_sum x) = 4 → n = 36) :=
by
  sorry

end count_valid_x_satisfying_heartsuit_condition_l479_47925


namespace degree_to_radian_radian_to_degree_l479_47955

theorem degree_to_radian (d : ℝ) (rad : ℝ) (deg : ℝ) :
  (180 * rad = π) → (d = 210) → rad = (π / 180) → d * rad = 7 * π / 6 :=
by sorry 

theorem radian_to_degree (r : ℝ) (rad : ℝ) (deg : ℝ) :
  (180 * rad = π) → (r = -5 * π / 2) → deg = (180 / π) → r * deg = -450 :=
by sorry

end degree_to_radian_radian_to_degree_l479_47955


namespace Derrick_yard_length_l479_47948

variables (Alex_yard Derrick_yard Brianne_yard Carla_yard Derek_yard : ℝ)

-- Given conditions as hypotheses
theorem Derrick_yard_length :
  (Alex_yard = Derrick_yard / 2) →
  (Brianne_yard = 6 * Alex_yard) →
  (Carla_yard = 3 * Brianne_yard + 5) →
  (Derek_yard = Carla_yard / 2 - 10) →
  (Brianne_yard = 30) →
  Derrick_yard = 10 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end Derrick_yard_length_l479_47948


namespace weight_difference_l479_47950

-- Defining the weights of the individuals
variables (a b c d e : ℝ)

-- Given conditions as hypotheses
def conditions :=
  (a = 75) ∧
  ((a + b + c) / 3 = 84) ∧
  ((a + b + c + d) / 4 = 80) ∧
  ((b + c + d + e) / 4 = 79)

-- Theorem statement to prove the desired result
theorem weight_difference (h : conditions a b c d e) : e - d = 3 :=
by
  sorry

end weight_difference_l479_47950


namespace expression_divisible_by_7_l479_47997

theorem expression_divisible_by_7 (n : ℕ) (hn : n > 0) :
  7 ∣ (3^(3*n+1) + 5^(3*n+2) + 7^(3*n+3)) :=
sorry

end expression_divisible_by_7_l479_47997


namespace overall_average_marks_l479_47947

theorem overall_average_marks 
  (num_candidates : ℕ) 
  (num_passed : ℕ) 
  (avg_passed : ℕ) 
  (avg_failed : ℕ)
  (h1 : num_candidates = 120) 
  (h2 : num_passed = 100)
  (h3 : avg_passed = 39)
  (h4 : avg_failed = 15) :
  (num_passed * avg_passed + (num_candidates - num_passed) * avg_failed) / num_candidates = 35 := 
by
  sorry

end overall_average_marks_l479_47947


namespace find_x_given_y_l479_47927

-- Given that x and y are always positive and x^2 and y vary inversely.
-- i.e., we have a relationship x^2 * y = k for a constant k,
-- and given that y = 8 when x = 3, find the value of x when y = 648.

theorem find_x_given_y
  (x y : ℝ)
  (h_pos_x : 0 < x)
  (h_pos_y : 0 < y)
  (h_inv : ∀ x y, x^2 * y = 72)
  (h_y : y = 648) : x = 1 / 3 :=
by
  sorry

end find_x_given_y_l479_47927


namespace sufficient_but_not_necessary_l479_47920

theorem sufficient_but_not_necessary (x : ℝ) : (x = -1 → x^2 - 5 * x - 6 = 0) ∧ (∃ y : ℝ, y ≠ -1 ∧ y^2 - 5 * y - 6 = 0) :=
by
  sorry

end sufficient_but_not_necessary_l479_47920


namespace teacups_count_l479_47931

theorem teacups_count (total_people teacup_capacity : ℕ) (H1 : total_people = 63) (H2 : teacup_capacity = 9) : total_people / teacup_capacity = 7 :=
by
  sorry

end teacups_count_l479_47931


namespace build_time_40_workers_l479_47905

theorem build_time_40_workers (r : ℝ) : 
  (60 * r) * 5 = 1 → (40 * r) * t = 1 → t = 7.5 :=
by
  intros h1 h2
  sorry

end build_time_40_workers_l479_47905


namespace power_of_two_l479_47935

theorem power_of_two (b m n : ℕ) (hb : b > 1) (hmn : m ≠ n) 
  (hprime_divisors : ∀ p : ℕ, p.Prime → (p ∣ b ^ m - 1 ↔ p ∣ b ^ n - 1)) : 
  ∃ k : ℕ, b + 1 = 2 ^ k :=
by
  sorry

end power_of_two_l479_47935


namespace wendy_total_sales_l479_47983

noncomputable def apple_price : ℝ := 1.50
noncomputable def orange_price : ℝ := 1.00
noncomputable def morning_apples : ℕ := 40
noncomputable def morning_oranges : ℕ := 30
noncomputable def afternoon_apples : ℕ := 50
noncomputable def afternoon_oranges : ℕ := 40

theorem wendy_total_sales :
  (morning_apples * apple_price + morning_oranges * orange_price) +
  (afternoon_apples * apple_price + afternoon_oranges * orange_price) = 205 := by
  sorry

end wendy_total_sales_l479_47983


namespace estimated_value_at_28_l479_47975

-- Definitions based on the conditions
def regression_equation (x : ℝ) : ℝ := 4.75 * x + 257

-- Problem statement
theorem estimated_value_at_28 : regression_equation 28 = 390 :=
by
  -- Sorry is used to skip the proof
  sorry

end estimated_value_at_28_l479_47975


namespace square_perimeter_l479_47921

theorem square_perimeter (area : ℝ) (h : area = 144) : ∃ perimeter : ℝ, perimeter = 48 :=
by
  sorry

end square_perimeter_l479_47921


namespace dragon_jewels_l479_47901

theorem dragon_jewels (x : ℕ) (h1 : (x / 3 = 6)) : x + 6 = 24 :=
sorry

end dragon_jewels_l479_47901


namespace kids_on_soccer_field_l479_47977

theorem kids_on_soccer_field (n f : ℕ) (h1 : n = 14) (h2 : f = 3) :
  n + n * f = 56 :=
by
  sorry

end kids_on_soccer_field_l479_47977


namespace distinct_solutions_eq_four_l479_47981

theorem distinct_solutions_eq_four : ∃! (x : ℝ), abs (x - abs (3 * x + 2)) = 4 :=
by sorry

end distinct_solutions_eq_four_l479_47981


namespace find_values_of_a2_b2_l479_47909

-- Define the conditions
variables {a b : ℝ}
variable (h1 : a > b)
variable (h2 : b > 0)
variable (hP : (-2, (Real.sqrt 14) / 2) ∈ { p : ℝ × ℝ | (p.1^2) / (a^2) + (p.2^2) / (b^2) = 1 })
variable (hCircle : ∀ Q : ℝ × ℝ, (Q ∈ { p : ℝ × ℝ | p.1^2 + p.2^2 = 2 }) → (∃ tA tB : ℝ × ℝ, (tA ∈ { p : ℝ × ℝ | (p.1^2) / (a^2) + (p.2^2) / (b^2) = 1 }) ∧ (tB ∈ { p : ℝ × ℝ | (p.1^2) / (a^2) + (p.2^2) / (b^2) = 1 }) ∧ (tA = - tB ∨ tB = - tA) ∧ ((tA.1 + tB.1)/2 = (-2 + tA.1)/2) ))

-- The theorem to be proven
theorem find_values_of_a2_b2 : a^2 + b^2 = 15 :=
sorry

end find_values_of_a2_b2_l479_47909


namespace problem_a_l479_47913

theorem problem_a (k l m : ℝ) : 
  (k + l + m) ^ 2 >= 3 * (k * l + l * m + m * k) :=
by sorry

end problem_a_l479_47913


namespace distinguishable_octahedrons_l479_47980

noncomputable def number_of_distinguishable_octahedrons (total_colors : ℕ) (used_colors : ℕ) : ℕ :=
  let num_ways_choose_colors := Nat.choose total_colors (used_colors - 1)
  let num_permutations := (used_colors - 1).factorial
  let num_rotations := 3
  (num_ways_choose_colors * num_permutations) / num_rotations

theorem distinguishable_octahedrons (h : number_of_distinguishable_octahedrons 9 8 = 13440) : true := sorry

end distinguishable_octahedrons_l479_47980


namespace negation_of_p_l479_47907

theorem negation_of_p :
  ¬(∀ x : ℝ, x > 0 → (x + 1) * Real.exp x > 1) ↔ ∃ x : ℝ, x > 0 ∧ (x + 1) * Real.exp x ≤ 1 :=
by
  sorry

end negation_of_p_l479_47907


namespace factorization_of_2210_l479_47978

theorem factorization_of_2210 : 
  ∃! (a b : ℕ), (10 ≤ a ∧ a < 100) ∧ (10 ≤ b ∧ b < 100) ∧ (a * b = 2210) :=
sorry

end factorization_of_2210_l479_47978
