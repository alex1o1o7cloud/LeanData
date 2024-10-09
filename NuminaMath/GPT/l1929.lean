import Mathlib

namespace find_f2_l1929_192941

def f (x : ℝ) : ℝ := sorry

theorem find_f2 : (∀ x, f (x-1) = x / (x-1)) → f 2 = 3 / 2 :=
by
  sorry

end find_f2_l1929_192941


namespace cone_lateral_surface_area_l1929_192933

noncomputable def lateralSurfaceArea (r l : ℝ) : ℝ := Real.pi * r * l

theorem cone_lateral_surface_area : 
  ∀ (r l : ℝ), r = 2 → l = 5 → lateralSurfaceArea r l = 10 * Real.pi :=
by 
  intros r l hr hl
  rw [hr, hl]
  unfold lateralSurfaceArea
  norm_num
  sorry

end cone_lateral_surface_area_l1929_192933


namespace find_r_value_l1929_192943

theorem find_r_value (m : ℕ) (h_m : m = 3) (t : ℕ) (h_t : t = 3^m + 2) (r : ℕ) (h_r : r = 4^t - 2 * t) : r = 4^29 - 58 := by
  sorry

end find_r_value_l1929_192943


namespace intersect_complement_l1929_192956

-- Definition of the universal set U
def U : Set ℕ := {1, 2, 3, 4, 5}

-- Definition of set A
def A : Set ℕ := {1, 2, 3}

-- Definition of set B
def B : Set ℕ := {3, 4}

-- Definition of the complement of B in U
def CU (U : Set ℕ) (B : Set ℕ) : Set ℕ := {x | x ∈ U ∧ x ∉ B}

-- Expected result of the intersection
def result : Set ℕ := {1, 2}

-- The proof statement
theorem intersect_complement :
  A ∩ CU U B = result :=
sorry

end intersect_complement_l1929_192956


namespace RiversideAcademy_statistics_l1929_192973

theorem RiversideAcademy_statistics (total_students physics_students both_subjects : ℕ)
  (h1 : total_students = 25)
  (h2 : physics_students = 10)
  (h3 : both_subjects = 6) :
  total_students - (physics_students - both_subjects) = 21 :=
by
  sorry

end RiversideAcademy_statistics_l1929_192973


namespace maximize_product_minimize_product_l1929_192947

-- Define the numbers that need to be arranged
def numbers : List ℕ := [2, 4, 6, 8]

-- Prove that 82 * 64 is the maximum product arrangement
theorem maximize_product : ∃ a b c d : ℕ, (a = 8) ∧ (b = 2) ∧ (c = 6) ∧ (d = 4) ∧ 
  (a * 10 + b) * (c * 10 + d) = 5248 :=
by
  existsi 8, 2, 6, 4
  constructor; constructor
  repeat {assumption}
  sorry

-- Prove that 28 * 46 is the minimum product arrangement
theorem minimize_product : ∃ a b c d : ℕ, (a = 2) ∧ (b = 8) ∧ (c = 4) ∧ (d = 6) ∧ 
  (a * 10 + b) * (c * 10 + d) = 1288 :=
by
  existsi 2, 8, 4, 6
  constructor; constructor
  repeat {assumption}
  sorry

end maximize_product_minimize_product_l1929_192947


namespace constant_term_in_expansion_l1929_192934

theorem constant_term_in_expansion (n k : ℕ) (x : ℝ) (choose : ℕ → ℕ → ℕ):
  (choose 12 3) * (6 ^ 3) = 47520 :=
by
  sorry

end constant_term_in_expansion_l1929_192934


namespace milk_leftover_l1929_192950

def milk (milkshake_num : ℕ) := 4 * milkshake_num
def ice_cream (milkshake_num : ℕ) := 12 * milkshake_num
def possible_milkshakes (ice_cream_amount : ℕ) := ice_cream_amount / 12

theorem milk_leftover (total_milk total_ice_cream : ℕ) (h1 : total_milk = 72) (h2 : total_ice_cream = 192) :
  total_milk - milk (possible_milkshakes total_ice_cream) = 8 :=
by
  sorry

end milk_leftover_l1929_192950


namespace quadrilateral_area_proof_l1929_192995

-- Assume we have a rectangle with area 24 cm^2 and two triangles with total area 7.5 cm^2.
-- We want to prove the area of the quadrilateral ABCD is 16.5 cm^2 inside this rectangle.

def rectangle_area : ℝ := 24
def triangles_area : ℝ := 7.5
def quadrilateral_area : ℝ := rectangle_area - triangles_area

theorem quadrilateral_area_proof : quadrilateral_area = 16.5 := 
by
  exact sorry

end quadrilateral_area_proof_l1929_192995


namespace arithmetic_sequence_problem_l1929_192911

variable (d a1 : ℝ)
variable (h1 : a1 ≠ d)
variable (h2 : d ≠ 0)

theorem arithmetic_sequence_problem (S20 M : ℝ)
  (h3 : S20 = 10 * M)
  (x y : ℝ)
  (h4 : M = x * (a1 + 9 * d) + y * d) :
  x = 2 ∧ y = 1 := 
by 
  sorry

end arithmetic_sequence_problem_l1929_192911


namespace equation_of_line_l1929_192970

noncomputable def line_equation_parallel (x y : ℝ) : Prop :=
  ∃ (m : ℝ), (3 * x - 6 * y = 9) ∧ (m = 1/2)

theorem equation_of_line (m : ℝ) (b : ℝ) :
  line_equation_parallel 3 9 →
  (m = 1/2) →
  (∀ (x y : ℝ), (y = m * x + b) ↔ (y - (-1) = m * (x - 2))) →
  b = -2 :=
by
  intros h_eq h_m h_line
  sorry

end equation_of_line_l1929_192970


namespace trig_identity_l1929_192985

theorem trig_identity (α : ℝ) (h : 3 * Real.sin α + Real.cos α = 0) : 
  (1 / (Real.cos α ^ 2 + Real.sin (2 * α))) = 10 / 3 := 
by 
  sorry

end trig_identity_l1929_192985


namespace cube_side_length_l1929_192969

theorem cube_side_length (n : ℕ) (h1 : 6 * (n^2) = 1/3 * 6 * (n^3)) : n = 3 := 
sorry

end cube_side_length_l1929_192969


namespace original_fraction_eq_2_5_l1929_192957

theorem original_fraction_eq_2_5 (a b : ℤ) (h : (a + 4) * b = a * (b + 10)) : (a / b) = (2 / 5) := by
  sorry

end original_fraction_eq_2_5_l1929_192957


namespace M_supseteq_P_l1929_192908

def M : Set ℝ := {y | ∃ x : ℝ, y = x^2 - 4}
def P : Set ℝ := {y | |y - 3| ≤ 1}

theorem M_supseteq_P : M ⊇ P := 
sorry

end M_supseteq_P_l1929_192908


namespace value_of_p_minus_q_plus_r_l1929_192901

theorem value_of_p_minus_q_plus_r
  (p q r : ℚ)
  (h1 : 3 / p = 6)
  (h2 : 3 / q = 18)
  (h3 : 5 / r = 15) :
  p - q + r = 2 / 3 :=
by
  sorry

end value_of_p_minus_q_plus_r_l1929_192901


namespace prob_both_shoot_in_one_round_prob_specified_shots_in_two_rounds_l1929_192966

noncomputable def P_A := 4 / 5
noncomputable def P_B := 3 / 4

def independent (P_X P_Y : ℚ) := P_X * P_Y

theorem prob_both_shoot_in_one_round : independent P_A P_B = 3 / 5 := by
  sorry

noncomputable def P_A_1 := 2 * (4 / 5) * (1 / 5)
noncomputable def P_A_2 := (4 / 5) * (4 / 5)
noncomputable def P_B_1 := 2 * (3 / 4) * (1 / 4)
noncomputable def P_B_2 := (3 / 4) * (3 / 4)

def event_A (P_A_1 P_A_2 P_B_1 P_B_2 : ℚ) := (P_A_1 * P_B_2) + (P_A_2 * P_B_1)

theorem prob_specified_shots_in_two_rounds : event_A P_A_1 P_A_2 P_B_1 P_B_2 = 3 / 10 := by
  sorry

end prob_both_shoot_in_one_round_prob_specified_shots_in_two_rounds_l1929_192966


namespace total_distance_correct_l1929_192932

-- Given conditions
def fuel_efficiency_city : Float := 15
def fuel_efficiency_highway : Float := 25
def fuel_efficiency_gravel : Float := 18

def gallons_used_city : Float := 2.5
def gallons_used_highway : Float := 3.8
def gallons_used_gravel : Float := 1.7

-- Define distances
def distance_city := fuel_efficiency_city * gallons_used_city
def distance_highway := fuel_efficiency_highway * gallons_used_highway
def distance_gravel := fuel_efficiency_gravel * gallons_used_gravel

-- Define total distance
def total_distance := distance_city + distance_highway + distance_gravel

-- Prove the total distance traveled is 163.1 miles
theorem total_distance_correct : total_distance = 163.1 := by
  -- Proof to be filled in
  sorry

end total_distance_correct_l1929_192932


namespace tanks_fill_l1929_192996

theorem tanks_fill
  (c : ℕ) -- capacity of each tank
  (h1 : 300 < c) -- first tank is filled with 300 liters, thus c > 300
  (h2 : 450 < c) -- second tank is filled with 450 liters, thus c > 450
  (h3 : (45 : ℝ) / 100 = (450 : ℝ) / c) -- second tank is 45% filled, thus 0.45 * c = 450
  (h4 : 300 + 450 < 2 * c) -- the two tanks have the same capacity, thus they must have enough capacity to be filled more than 750 liters
  : c - 300 + (c - 450) = 1250 :=
sorry

end tanks_fill_l1929_192996


namespace sin_triangle_sides_l1929_192939

theorem sin_triangle_sides (a b c : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : c > 0)
  (h₃ : a + b + c ≤ 2 * Real.pi) (h₄ : a + b > c) (h₅ : b + c > a) (h₆ : c + a > b) :
  ∃ x y z : ℝ, x = Real.sin a ∧ y = Real.sin b ∧ z = Real.sin c ∧ x + y > z ∧ y + z > x ∧ z + x > y := 
by
  sorry

end sin_triangle_sides_l1929_192939


namespace candy_left_proof_l1929_192968

def candy_left (d_candy : ℕ) (s_candy : ℕ) (eaten_candy : ℕ) : ℕ :=
  d_candy + s_candy - eaten_candy

theorem candy_left_proof :
  candy_left 32 42 35 = 39 :=
by
  sorry

end candy_left_proof_l1929_192968


namespace ratio_of_x_to_y_l1929_192904

theorem ratio_of_x_to_y (x y : ℚ) (h : (8*x - 5*y)/(10*x - 3*y) = 4/7) : x/y = 23/16 :=
by 
  sorry

end ratio_of_x_to_y_l1929_192904


namespace min_distance_between_tracks_l1929_192949

noncomputable def min_distance : ℝ :=
  (Real.sqrt 163 - 6) / 3

theorem min_distance_between_tracks :
  let RationalManTrack := {p : ℝ × ℝ | p.1^2 + p.2^2 = 4}
  let IrrationalManTrack := {p : ℝ × ℝ | (p.1 - 2)^2 / 9 + p.2^2 / 25 = 1}
  ∀ pA ∈ RationalManTrack, ∀ pB ∈ IrrationalManTrack,
  dist pA pB = min_distance :=
sorry

end min_distance_between_tracks_l1929_192949


namespace eval_product_eq_1093_l1929_192960

noncomputable def z : ℂ := Complex.exp (2 * Real.pi * Complex.I / 7)

theorem eval_product_eq_1093 : (3 - z) * (3 - z^2) * (3 - z^3) * (3 - z^4) * (3 - z^5) * (3 - z^6) = 1093 := by
  sorry

end eval_product_eq_1093_l1929_192960


namespace range_of_m_l1929_192948

theorem range_of_m (f : ℝ → ℝ) (m : ℝ) 
  (h1 : ∀ x : ℝ, f x = x^2 + 4 * x + 5)
  (h2 : ∀ x : ℝ, f (-2 + x) = f (-2 - x))
  (h3 : ∀ x : ℝ, m ≤ x ∧ x ≤ 0 → 1 ≤ f x ∧ f x ≤ 5)
  : -4 ≤ m ∧ m ≤ -2 :=
  sorry

end range_of_m_l1929_192948


namespace find_fraction_l1929_192963

theorem find_fraction (c d : ℕ) (h1 : 435 = 2 * 100 + c * 10 + d) :
  (c + d) / 12 = 5 / 6 :=
by sorry

end find_fraction_l1929_192963


namespace odd_function_neg_value_l1929_192999

theorem odd_function_neg_value (f : ℝ → ℝ) (h_odd : ∀ x : ℝ, f (-x) = -f x) (h_value : f 1 = 1) : f (-1) = -1 :=
by
  sorry

end odd_function_neg_value_l1929_192999


namespace infinitely_many_H_points_l1929_192983

-- Define the curve C as (x^2 / 4) + y^2 = 1
def is_on_curve (x y : ℝ) : Prop :=
  (x^2 / 4) + y^2 = 1

-- Define point P on curve C
def is_H_point (P : ℝ × ℝ) : Prop :=
  is_on_curve P.1 P.2 ∧
  ∃ (A B : ℝ × ℝ), is_on_curve A.1 A.2 ∧ B.1 = 4 ∧
  (dist (P.1, P.2) (A.1, A.2) = dist (P.1, P.2) (B.1, B.2) ∨
   dist (P.1, P.2) (A.1, A.2) = dist (A.1, A.2) (B.1, B.2))

-- Theorem to prove the existence of infinitely many H points
theorem infinitely_many_H_points : ∃ (P : ℝ × ℝ), is_H_point P ∧ ∀ (Q : ℝ × ℝ), Q ≠ P → is_H_point Q :=
sorry


end infinitely_many_H_points_l1929_192983


namespace length_second_train_is_125_l1929_192989

noncomputable def length_second_train (speed_faster speed_slower distance1 : ℕ) (time_minutes : ℝ) : ℝ :=
  let relative_speed_m_per_minute := (speed_faster - speed_slower) * 1000 / 60
  let total_distance_covered := relative_speed_m_per_minute * time_minutes
  total_distance_covered - distance1

theorem length_second_train_is_125 :
  length_second_train 50 40 125 1.5 = 125 :=
  by sorry

end length_second_train_is_125_l1929_192989


namespace total_ice_cubes_correct_l1929_192976

/-- Each tray holds 48 ice cubes -/
def cubes_per_tray : Nat := 48

/-- Billy has 24 trays -/
def number_of_trays : Nat := 24

/-- Calculate the total number of ice cubes -/
def total_ice_cubes (cubes_per_tray : Nat) (number_of_trays : Nat) : Nat :=
  cubes_per_tray * number_of_trays

/-- Proof that the total number of ice cubes is 1152 given the conditions -/
theorem total_ice_cubes_correct : total_ice_cubes cubes_per_tray number_of_trays = 1152 := by
  /- Here we state the main theorem, but we leave the proof as sorry per the instructions -/
  sorry

end total_ice_cubes_correct_l1929_192976


namespace zoo_individuals_remaining_l1929_192988

noncomputable def initial_students_class1 := 10
noncomputable def initial_students_class2 := 10
noncomputable def chaperones := 5
noncomputable def teachers := 2
noncomputable def students_left := 10
noncomputable def chaperones_left := 2

theorem zoo_individuals_remaining :
  let total_initial_individuals := initial_students_class1 + initial_students_class2 + chaperones + teachers
  let total_left := students_left + chaperones_left
  total_initial_individuals - total_left = 15 := by
  sorry

end zoo_individuals_remaining_l1929_192988


namespace intersection_points_area_l1929_192951

noncomputable def C (x : ℝ) : ℝ := (Real.log x)^2

noncomputable def L (α : ℝ) (x : ℝ) : ℝ :=
  (2 * Real.log α / α) * x - (Real.log α)^2

noncomputable def n (α : ℝ) : ℕ :=
  if α < 1 then 0 else if α = 1 then 1 else 2

noncomputable def S (α : ℝ) : ℝ :=
  2 - 2 * α - (1 / 2) * α * (Real.log α)^2 + 2 * α * Real.log α

theorem intersection_points (α : ℝ) (h : 0 < α) : n α = if α < 1 then 0 else if α = 1 then 1 else 2 := by
  sorry

theorem area (α : ℝ) (h : 0 < α ∧ α < 1) : S α = 2 - 2 * α - (1 / 2) * α * (Real.log α)^2 + 2 * α * Real.log α := by
  sorry

end intersection_points_area_l1929_192951


namespace expression_equals_4034_l1929_192945

theorem expression_equals_4034 : 6 * 2017 - 4 * 2017 = 4034 := by
  sorry

end expression_equals_4034_l1929_192945


namespace movies_in_series_l1929_192981

theorem movies_in_series :
  -- conditions 
  let number_books := 10
  let books_read := 14
  let book_read_vs_movies_extra := 5
  (∀ number_movies : ℕ, 
  (books_read = number_movies + book_read_vs_movies_extra) →
  -- question
  number_movies = 9) := sorry

end movies_in_series_l1929_192981


namespace greatest_drop_is_third_quarter_l1929_192916

def priceStart (quarter : ℕ) : ℕ :=
  match quarter with
  | 1 => 10
  | 2 => 7
  | 3 => 9
  | 4 => 5
  | _ => 0 -- default case for invalid quarters

def priceEnd (quarter : ℕ) : ℕ :=
  match quarter with
  | 1 => 7
  | 2 => 9
  | 3 => 5
  | 4 => 6
  | _ => 0 -- default case for invalid quarters

def priceChange (quarter : ℕ) : ℤ :=
  priceStart quarter - priceEnd quarter

def greatestDropInQuarter : ℕ :=
  if priceChange 1 > priceChange 3 then 1
  else if priceChange 2 > priceChange 1 then 2
  else if priceChange 3 > priceChange 4 then 3
  else 4

theorem greatest_drop_is_third_quarter :
  greatestDropInQuarter = 3 :=
by
  -- proof goes here
  sorry

end greatest_drop_is_third_quarter_l1929_192916


namespace distance_between_trees_l1929_192919

theorem distance_between_trees (num_trees : ℕ) (length_yard : ℝ)
  (h1 : num_trees = 26) (h2 : length_yard = 800) : 
  (length_yard / (num_trees - 1)) = 32 :=
by
  sorry

end distance_between_trees_l1929_192919


namespace find_x_l1929_192971

theorem find_x (x : ℝ) 
  (h1 : x = (1 / x * -x) - 5) 
  (h2 : x^2 - 3 * x + 2 ≥ 0) : 
  x = -6 := 
sorry

end find_x_l1929_192971


namespace LaShawn_twice_Kymbrea_after_25_months_l1929_192902

theorem LaShawn_twice_Kymbrea_after_25_months : 
  ∀ (x : ℕ), (10 + 6 * x = 2 * (30 + 2 * x)) → x = 25 :=
by
  intro x
  sorry

end LaShawn_twice_Kymbrea_after_25_months_l1929_192902


namespace functions_from_M_to_N_l1929_192928

def M : Set ℤ := { -1, 1, 2, 3 }
def N : Set ℤ := { 0, 1, 2, 3, 4 }
def f2 (x : ℤ) := x + 1
def f4 (x : ℤ) := (x - 1)^2

theorem functions_from_M_to_N :
  (∀ x ∈ M, f2 x ∈ N) ∧ (∀ x ∈ M, f4 x ∈ N) :=
by
  sorry

end functions_from_M_to_N_l1929_192928


namespace machine_A_produces_7_sprockets_per_hour_l1929_192977

theorem machine_A_produces_7_sprockets_per_hour
    (A B : ℝ)
    (h1 : B = 1.10 * A)
    (h2 : ∃ t : ℝ, 770 = A * (t + 10) ∧ 770 = B * t) : 
    A = 7 := 
by 
    sorry

end machine_A_produces_7_sprockets_per_hour_l1929_192977


namespace john_fouled_per_game_l1929_192927

theorem john_fouled_per_game
  (hit_rate : ℕ) (shots_per_foul : ℕ) (total_games : ℕ) (participation_rate : ℚ) (total_free_throws : ℕ) :
  hit_rate = 70 → shots_per_foul = 2 → total_games = 20 → participation_rate = 0.8 → total_free_throws = 112 →
  (total_free_throws / (participation_rate * total_games)) / shots_per_foul = 3.5 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end john_fouled_per_game_l1929_192927


namespace inversely_proportional_y_ratio_l1929_192930

variable {k : ℝ}
variable {x₁ x₂ y₁ y₂ : ℝ}
variable (h_inv_prop : ∀ (x y : ℝ), x * y = k)
variable (hx₁x₂ : x₁ ≠ 0 ∧ x₂ ≠ 0)
variable (hy₁y₂ : y₁ ≠ 0 ∧ y₂ ≠ 0)
variable (hx_ratio : x₁ / x₂ = 3 / 4)

theorem inversely_proportional_y_ratio :
  y₁ / y₂ = 4 / 3 :=
by
  sorry

end inversely_proportional_y_ratio_l1929_192930


namespace solve_xy_l1929_192907

theorem solve_xy (x y a b : ℝ) (h1 : x * y = 2 * b) (h2 : (1 / x^2) + (1 / y^2) = a) : 
  (x + y)^2 = 4 * a * b^2 + 4 * b := 
by 
  sorry

end solve_xy_l1929_192907


namespace destroyed_cakes_l1929_192922

theorem destroyed_cakes (initial_cakes : ℕ) (half_falls : ℕ) (half_saved : ℕ)
  (h1 : initial_cakes = 12)
  (h2 : half_falls = initial_cakes / 2)
  (h3 : half_saved = half_falls / 2) :
  initial_cakes - half_falls / 2 = 3 :=
by
  sorry

end destroyed_cakes_l1929_192922


namespace gcd_of_78_and_36_l1929_192987

theorem gcd_of_78_and_36 :
  Nat.gcd 78 36 = 6 :=
by
  sorry

end gcd_of_78_and_36_l1929_192987


namespace value_of_T_l1929_192961

variables {A M T E H : ℕ}

theorem value_of_T (H : ℕ) (MATH : ℕ) (MEET : ℕ) (TEAM : ℕ) (H_eq : H = 8) (MATH_eq : MATH = 47) (MEET_eq : MEET = 62) (TEAM_eq : TEAM = 58) :
  T = 9 :=
by
  sorry

end value_of_T_l1929_192961


namespace december_sales_fraction_l1929_192931

theorem december_sales_fraction (A : ℚ) : 
  let sales_jan_to_nov := 11 * A
  let sales_dec := 5 * A
  let total_sales := sales_jan_to_nov + sales_dec
  (sales_dec / total_sales) = 5 / 16 :=
by
  sorry

end december_sales_fraction_l1929_192931


namespace induction_step_l1929_192982

theorem induction_step (k : ℕ) : ((k + 1 + k) * (k + 1 + k + 1) / (k + 1)) = 2 * (2 * k + 1) := by
  sorry

end induction_step_l1929_192982


namespace smallest_b_factors_l1929_192965

theorem smallest_b_factors (b : ℕ) (m n : ℤ) (h : m * n = 2023 ∧ m + n = b) : b = 136 :=
sorry

end smallest_b_factors_l1929_192965


namespace train_passing_tree_time_l1929_192990

theorem train_passing_tree_time
  (train_length : ℝ) (train_speed_kmhr : ℝ) (conversion_factor : ℝ)
  (train_speed_ms : train_speed_ms = train_speed_kmhr * conversion_factor) :
  train_length = 500 → train_speed_kmhr = 72 → conversion_factor = 5 / 18 →
  500 / (72 * (5 / 18)) = 25 := 
by
  intros h1 h2 h3
  sorry

end train_passing_tree_time_l1929_192990


namespace find_first_term_l1929_192935

theorem find_first_term
  (a r : ℝ)
  (h1 : a / (1 - r) = 20)
  (h2 : a^2 / (1 - r^2) = 80) :
  a = 20 / 3 :=
by
  -- Proof is omitted for brevity
  sorry

end find_first_term_l1929_192935


namespace price_of_mixture_l1929_192921

theorem price_of_mixture :
  (1 * 64 + 1 * 74) / (1 + 1) = 69 :=
by
  sorry

end price_of_mixture_l1929_192921


namespace inequality_condition_l1929_192903

theorem inequality_condition {x : ℝ} (h : -1/2 ≤ x ∧ x < 1) : (2 * x + 1) / (1 - x) ≥ 0 :=
sorry

end inequality_condition_l1929_192903


namespace karen_cases_pickup_l1929_192926

theorem karen_cases_pickup (total_boxes cases_per_box: ℕ) (h1 : total_boxes = 36) (h2 : cases_per_box = 12):
  total_boxes / cases_per_box = 3 :=
by
  -- We insert a placeholder to skip the proof here
  sorry

end karen_cases_pickup_l1929_192926


namespace fixed_point_line_l1929_192938

theorem fixed_point_line (m x y : ℝ) (h : (2 * m + 1) * x + (m + 1) * y - 7 * m - 4 = 0) :
  x = 3 ∧ y = 1 :=
sorry

end fixed_point_line_l1929_192938


namespace smallest_prime_with_digit_sum_23_l1929_192997

def digits_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem smallest_prime_with_digit_sum_23 : ∃ p > 0, Prime p ∧ digits_sum p = 23 ∧ ∀ q, Prime q ∧ digits_sum q = 23 → q ≥ p :=
by
  exists 887
  sorry -- The detailed proof steps will be provided here.

end smallest_prime_with_digit_sum_23_l1929_192997


namespace selling_price_of_radio_l1929_192900

theorem selling_price_of_radio (CP LP : ℝ) (hCP : CP = 1500) (hLP : LP = 14.000000000000002) : 
  CP - (LP / 100 * CP) = 1290 :=
by
  -- Given definitions
  have h1 : CP - (LP / 100 * CP) = 1290 := sorry
  exact h1

end selling_price_of_radio_l1929_192900


namespace eggs_per_chicken_l1929_192962

theorem eggs_per_chicken (num_chickens : ℕ) (eggs_per_carton : ℕ) (num_cartons : ℕ) (total_eggs : ℕ) 
  (h1 : num_chickens = 20) (h2 : eggs_per_carton = 12) (h3 : num_cartons = 10) (h4 : total_eggs = num_cartons * eggs_per_carton) : 
  total_eggs / num_chickens = 6 :=
by
  sorry

end eggs_per_chicken_l1929_192962


namespace cost_price_of_cloth_l1929_192992

theorem cost_price_of_cloth:
  ∀ (meters_sold profit_per_meter : ℕ) (selling_price : ℕ),
  meters_sold = 45 →
  profit_per_meter = 12 →
  selling_price = 4500 →
  (selling_price - (profit_per_meter * meters_sold)) / meters_sold = 88 :=
by
  intros meters_sold profit_per_meter selling_price h1 h2 h3
  sorry

end cost_price_of_cloth_l1929_192992


namespace tangent_line_at_point_A_l1929_192909

noncomputable def curve (x : ℝ) : ℝ := Real.exp x

def point : ℝ × ℝ := (0, 1)

theorem tangent_line_at_point_A :
  ∃ m b : ℝ, (∀ x : ℝ, (curve x - (m * x + b))^2 = 0) ∧  
  m = 1 ∧ b = 1 :=
by
  sorry

end tangent_line_at_point_A_l1929_192909


namespace students_in_donnelly_class_l1929_192954

-- Conditions
def initial_cupcakes : ℕ := 40
def cupcakes_to_delmont_class : ℕ := 18
def cupcakes_to_staff : ℕ := 4
def leftover_cupcakes : ℕ := 2

-- Question: How many students are in Mrs. Donnelly's class?
theorem students_in_donnelly_class : 
  let cupcakes_given_to_students := initial_cupcakes - (cupcakes_to_delmont_class + cupcakes_to_staff)
  let cupcakes_given_to_donnelly_class := cupcakes_given_to_students - leftover_cupcakes
  cupcakes_given_to_donnelly_class = 16 :=
by
  sorry

end students_in_donnelly_class_l1929_192954


namespace janet_extra_flowers_l1929_192984

-- Define the number of flowers Janet picked for each type
def tulips : ℕ := 5
def roses : ℕ := 10
def daisies : ℕ := 8
def lilies : ℕ := 4

-- Define the number of flowers Janet used
def used : ℕ := 19

-- Calculate the total number of flowers Janet picked
def total_picked : ℕ := tulips + roses + daisies + lilies

-- Calculate the number of extra flowers
def extra_flowers : ℕ := total_picked - used

-- The theorem to be proven
theorem janet_extra_flowers : extra_flowers = 8 :=
by
  -- You would provide the proof here, but it's not required as per instructions
  sorry

end janet_extra_flowers_l1929_192984


namespace trapezoid_circle_ratio_l1929_192979

variable (P R : ℝ)

def is_isosceles_trapezoid_inscribed_in_circle (P R : ℝ) : Prop :=
  ∃ m A, 
    m = P / 4 ∧
    A = m * 2 * R ∧
    A = (P * R) / 2

theorem trapezoid_circle_ratio (P R : ℝ) 
  (h : is_isosceles_trapezoid_inscribed_in_circle P R) :
  (P / 2 * π * R) = (P / 2 * π * R) :=
by
  -- Use the given condition to prove the statement
  sorry

end trapezoid_circle_ratio_l1929_192979


namespace find_x_squared_plus_y_squared_l1929_192980

open Real

theorem find_x_squared_plus_y_squared (x y : ℝ) (h1 : (x + y)^2 = 4) (h2 : x * y = -6) : x^2 + y^2 = 16 := by
  sorry

end find_x_squared_plus_y_squared_l1929_192980


namespace total_cards_l1929_192906

theorem total_cards (H F B : ℕ) (hH : H = 200) (hF : F = 4 * H) (hB : B = F - 50) : H + F + B = 1750 := 
by 
  sorry

end total_cards_l1929_192906


namespace f_800_value_l1929_192944

theorem f_800_value (f : ℝ → ℝ) (f_condition : ∀ x y : ℝ, 0 < x → 0 < y → f (x * y) = f x / y) (f_400 : f 400 = 4) : f 800 = 2 :=
  sorry

end f_800_value_l1929_192944


namespace smallest_possible_value_l1929_192952

theorem smallest_possible_value (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (⌊(a + b + c) / d⌋ + ⌊(a + b + d) / c⌋ + ⌊(a + c + d) / b⌋ + ⌊(b + c + d) / a⌋) ≥ 8 :=
sorry

end smallest_possible_value_l1929_192952


namespace boys_planted_more_by_62_percent_girls_fraction_of_total_l1929_192972

-- Define the number of trees planted by boys and girls
def boys_trees : ℕ := 130
def girls_trees : ℕ := 80

-- Statement 1: Boys planted 62% more trees than girls
theorem boys_planted_more_by_62_percent : (boys_trees - girls_trees) * 100 / girls_trees = 62 := by
  sorry

-- Statement 2: The number of trees planted by girls represents 4/7 of the total number of trees
theorem girls_fraction_of_total : girls_trees * 7 = 4 * (boys_trees + girls_trees) := by
  sorry

end boys_planted_more_by_62_percent_girls_fraction_of_total_l1929_192972


namespace sum_of_remainders_l1929_192915

theorem sum_of_remainders (n : ℤ) (h : n % 18 = 11) : 
  (n % 3 + n % 6) = 7 :=
by
  sorry

end sum_of_remainders_l1929_192915


namespace convert_20121_base3_to_base10_l1929_192937

/- Define the base conversion function for base 3 to base 10 -/
def base3_to_base10 (d4 d3 d2 d1 d0 : ℕ) :=
  d4 * 3^4 + d3 * 3^3 + d2 * 3^2 + d1 * 3^1 + d0 * 3^0

/- Define the specific number in base 3 -/
def num20121_base3 := (2, 0, 1, 2, 1)

/- The theorem stating the equivalence of the base 3 number 20121_3 to its base 10 equivalent -/
theorem convert_20121_base3_to_base10 :
  base3_to_base10 2 0 1 2 1 = 178 :=
by
  sorry

end convert_20121_base3_to_base10_l1929_192937


namespace charlie_contribution_l1929_192994

theorem charlie_contribution (a b c : ℝ) (h₁ : a + b + c = 72) (h₂ : a = 1/4 * (b + c)) (h₃ : b = 1/5 * (a + c)) :
  c = 49 :=
by sorry

end charlie_contribution_l1929_192994


namespace total_uniform_cost_l1929_192929

theorem total_uniform_cost :
  let pants_cost := 20
  let shirt_cost := 2 * pants_cost
  let tie_cost := shirt_cost / 5
  let socks_cost := 3
  let uniform_cost := pants_cost + shirt_cost + tie_cost + socks_cost
  let total_cost := 5 * uniform_cost
  total_cost = 355 :=
by 
  let pants_cost := 20
  let shirt_cost := 2 * pants_cost
  let tie_cost := shirt_cost / 5
  let socks_cost := 3
  let uniform_cost := pants_cost + shirt_cost + tie_cost + socks_cost
  let total_cost := 5 * uniform_cost
  sorry

end total_uniform_cost_l1929_192929


namespace sin_minus_cos_eq_neg_sqrt_10_over_5_l1929_192920

theorem sin_minus_cos_eq_neg_sqrt_10_over_5 (θ : ℝ) (h1 : 0 < θ ∧ θ < π / 2) (h2 : Real.tan θ = 1 / 3) : 
  Real.sin θ - Real.cos θ = - ((Real.sqrt 10) / 5) :=
by
  sorry

end sin_minus_cos_eq_neg_sqrt_10_over_5_l1929_192920


namespace parakeets_per_cage_l1929_192998

-- Define total number of cages
def num_cages: Nat := 6

-- Define number of parrots per cage
def parrots_per_cage: Nat := 2

-- Define total number of birds in the store
def total_birds: Nat := 54

-- Theorem statement: prove the number of parakeets per cage
theorem parakeets_per_cage : (total_birds - num_cages * parrots_per_cage) / num_cages = 7 :=
by
  sorry

end parakeets_per_cage_l1929_192998


namespace T_53_eq_38_l1929_192940

def T (x y : ℕ) : ℕ := 4 * x + 6 * y

theorem T_53_eq_38 : T 5 3 = 38 := by
  sorry

end T_53_eq_38_l1929_192940


namespace tank_fill_time_l1929_192946

theorem tank_fill_time (R L : ℝ) (h1 : (R - L) * 8 = 1) (h2 : L * 56 = 1) :
  (1 / R) = 7 :=
by
  sorry

end tank_fill_time_l1929_192946


namespace penniless_pete_dime_difference_l1929_192967

theorem penniless_pete_dime_difference :
  ∃ a b c : ℕ, 
  (a + b + c = 100) ∧ 
  (5 * a + 10 * b + 50 * c = 1350) ∧ 
  (b = 170 ∨ b = 8) ∧ 
  (b - 8 = 162 ∨ 170 - b = 162) :=
sorry

end penniless_pete_dime_difference_l1929_192967


namespace smallest_positive_period_max_min_values_l1929_192993

noncomputable def a (x : ℝ) : ℝ × ℝ := (Real.cos x, -1 / 2)
noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.sin x, Real.cos (2 * x))
noncomputable def f (x : ℝ) : ℝ := (a x).1 * (b x).1 + (a x).2 * (b x).2

theorem smallest_positive_period (x : ℝ) :
  ∃ T, T > 0 ∧ ∀ x, f (x + T) = f x ∧ ∀ T', T' > 0 ∧ ∀ x, f (x + T') = f x → T ≤ T' :=
  sorry

theorem max_min_values : ∃ max min : ℝ, (max = 1) ∧ (min = -1 / 2) ∧
  ∀ x, 0 ≤ x ∧ x ≤ Real.pi / 2 →
  min ≤ f x ∧ f x ≤ max :=
  sorry

end smallest_positive_period_max_min_values_l1929_192993


namespace min_abs_sum_l1929_192910

theorem min_abs_sum (x : ℝ) : ∃ y : ℝ, y = min ((|x+1| + |x-2| + |x-3|)) 4 :=
sorry

end min_abs_sum_l1929_192910


namespace smaller_square_perimeter_l1929_192917

theorem smaller_square_perimeter (s : ℕ) (h1 : 4 * s = 144) : 
  let smaller_s := s / 3 
  let smaller_perimeter := 4 * smaller_s 
  smaller_perimeter = 48 :=
by
  let smaller_s := s / 3
  let smaller_perimeter := 4 * smaller_s 
  sorry

end smaller_square_perimeter_l1929_192917


namespace max_sum_numbered_cells_max_zero_number_cell_l1929_192923

-- Part 1
theorem max_sum_numbered_cells (n : ℕ) (grid : Matrix (Fin (2*n+1)) (Fin (2*n+1)) Cell) (mines : Finset (Fin (2*n+1) × Fin (2*n+1))) 
  (h1 : mines.card = n^2 + 1) :
  ∃ sum : ℕ, sum = 8 * n^2 + 4 := sorry

-- Part 2
theorem max_zero_number_cell (n k : ℕ) (grid : Matrix (Fin n) (Fin n) Cell) (mines : Finset (Fin n × Fin n)) 
  (h1 : mines.card = k) :
  ∃ (k_max : ℕ), k_max = (Nat.floor ((n + 2) / 3) ^ 2) - 1 := sorry

end max_sum_numbered_cells_max_zero_number_cell_l1929_192923


namespace xiaozhi_needs_median_for_top_10_qualification_l1929_192936

-- Define a set of scores as a list of integers
def scores : List ℕ := sorry

-- Assume these scores are unique (this is a condition given in the problem)
axiom unique_scores : ∀ (a b : ℕ), a ∈ scores → b ∈ scores → a ≠ b → scores.indexOf a ≠ scores.indexOf b

-- Define the median function (in practice, you would implement this, but we're just outlining it here)
def median (scores: List ℕ) : ℕ := sorry

-- Define the position of Xiao Zhi's score
def xiaozhi_score : ℕ := sorry

-- Given that the top 10 scores are needed to advance
def top_10 (scores: List ℕ) : List ℕ := scores.take 10

-- Proposition that Xiao Zhi needs median to determine his rank in top 10
theorem xiaozhi_needs_median_for_top_10_qualification 
    (scores_median : ℕ) (zs_score : ℕ) : 
    (∀ (s: List ℕ), s = scores → scores_median = median s → zs_score ≤ scores_median → zs_score ∉ top_10 s) ∧ 
    (exists (s: List ℕ), s = scores → zs_score ∉ top_10 s → zs_score ≤ scores_median) := 
sorry

end xiaozhi_needs_median_for_top_10_qualification_l1929_192936


namespace time_spent_on_Type_A_problems_l1929_192959

theorem time_spent_on_Type_A_problems (t : ℝ) (h1 : 25 * (8 * t) + 100 * (2 * t) = 120) : 
  25 * (8 * t) = 60 := by
  sorry

-- Conditions
-- t is the time spent on a Type C problem in minutes
-- 25 * (8 * t) + 100 * (2 * t) = 120 (time spent on Type A and B problems combined equals 120 minutes)

end time_spent_on_Type_A_problems_l1929_192959


namespace avg_remaining_two_l1929_192925

variables {A B C D E : ℝ}

-- Conditions
def avg_five (A B C D E : ℝ) : Prop := (A + B + C + D + E) / 5 = 10
def avg_three (A B C : ℝ) : Prop := (A + B + C) / 3 = 4

-- Theorem to prove
theorem avg_remaining_two (A B C D E : ℝ) (h1 : avg_five A B C D E) (h2 : avg_three A B C) : ((D + E) / 2) = 19 := 
sorry

end avg_remaining_two_l1929_192925


namespace cos_alpha_correct_l1929_192955

-- Define the point P
def P : ℝ × ℝ := (3, -4)

-- Define the hypotenuse using the Pythagorean theorem
noncomputable def r : ℝ :=
  Real.sqrt (P.1 * P.1 + P.2 * P.2)

-- Define x-coordinate of point P
def x : ℝ := P.1

-- Define the cosine of the angle
noncomputable def cos_alpha : ℝ :=
  x / r

-- Prove that cos_alpha equals 3/5 given the conditions
theorem cos_alpha_correct : cos_alpha = 3 / 5 :=
by
  sorry

end cos_alpha_correct_l1929_192955


namespace domain_of_function_l1929_192953

theorem domain_of_function :
  { x : ℝ // (6 - x - x^2) > 0 } = { x : ℝ // -3 < x ∧ x < 2 } :=
by
  sorry

end domain_of_function_l1929_192953


namespace geometric_sequence_min_value_l1929_192905

theorem geometric_sequence_min_value
  (q : ℝ) (a : ℕ → ℝ)
  (m n : ℕ) (h_m : 0 < m) (h_n : 0 < n)
  (h_geom : ∀ k, a k = q ^ k)
  (h_eq : a m * (a n) ^ 2 = (a 4) ^ 2)
  (h_sum : m + 2 * n = 8) :
  ∀ (f : ℝ), f = (2 / m + 1 / n) → f ≥ 1 :=
by
  sorry

end geometric_sequence_min_value_l1929_192905


namespace matrix_solution_property_l1929_192964

theorem matrix_solution_property (N : Matrix (Fin 2) (Fin 2) ℝ) 
    (h : N = Matrix.of ![![2, 4], ![1, 4]]) :
    N ^ 4 - 5 * N ^ 3 + 9 * N ^ 2 - 5 * N = Matrix.of ![![6, 12], ![3, 6]] :=
by 
  sorry

end matrix_solution_property_l1929_192964


namespace initial_amount_l1929_192991

theorem initial_amount (P R : ℝ) (h1 : 956 = P * (1 + (3 * R) / 100)) (h2 : 1052 = P * (1 + (3 * (R + 4)) / 100)) : P = 800 := 
by
  -- We would provide the proof steps here normally
  sorry

end initial_amount_l1929_192991


namespace find_smaller_integer_l1929_192958

theorem find_smaller_integer (x : ℤ) (h1 : ∃ y : ℤ, y = 2 * x) (h2 : x + 2 * x = 96) : x = 32 :=
sorry

end find_smaller_integer_l1929_192958


namespace divisibility_problem_l1929_192974

theorem divisibility_problem (n : ℕ) : 2016 ∣ ((n^2 + n)^2 - (n^2 - n)^2) * (n^6 - 1) := 
sorry

end divisibility_problem_l1929_192974


namespace rice_less_than_beans_by_30_l1929_192942

noncomputable def GB : ℝ := 60
noncomputable def S : ℝ := 50

theorem rice_less_than_beans_by_30 (R : ℝ) (x : ℝ) (h1 : R = 60 - x) (h2 : (2/3) * R + (4/5) * S + GB = 120) : 60 - R = 30 :=
by 
  -- Proof steps would go here, but they are not required for this task.
  sorry

end rice_less_than_beans_by_30_l1929_192942


namespace determine_constants_l1929_192914

theorem determine_constants (P Q R : ℚ) :
  (∀ x : ℚ, x ≠ 1 → x ≠ 4 → x ≠ 6 → 
    (x^2 - 5*x + 6) / ((x - 1) * (x - 4) * (x - 6)) =
    P / (x - 1) + Q / (x - 4) + R / (x - 6)) →
  P = 2 / 15 ∧ Q = 1 / 3 ∧ R = 0 :=
by {
  sorry
}

end determine_constants_l1929_192914


namespace average_of_six_numbers_l1929_192986

theorem average_of_six_numbers :
  (∀ a b : ℝ, (a + b) / 2 = 6.2) →
  (∀ c d : ℝ, (c + d) / 2 = 6.1) →
  (∀ e f : ℝ, (e + f) / 2 = 6.9) →
  ((a + b + c + d + e + f) / 6 = 6.4) :=
by
  intros h1 h2 h3
  -- Proof goes here, but will be skipped with sorry.
  sorry

end average_of_six_numbers_l1929_192986


namespace james_initial_friends_l1929_192912

theorem james_initial_friends (x : ℕ) (h1 : 19 = x - 2 + 1) : x = 20 :=
  by sorry

end james_initial_friends_l1929_192912


namespace gala_arrangements_l1929_192924

theorem gala_arrangements :
  let original_programs := 10
  let added_programs := 3
  let total_positions := original_programs + 1 - 2 -- Excluding first and last
  (total_positions * (total_positions - 1) * (total_positions - 2)) / 6 = 165 :=
by sorry

end gala_arrangements_l1929_192924


namespace find_m_l1929_192918

theorem find_m (m : ℝ) (h1 : m^2 - 3 * m + 2 = 0) (h2 : m ≠ 1) : m = 2 :=
sorry

end find_m_l1929_192918


namespace geom_progression_lines_common_point_l1929_192975

theorem geom_progression_lines_common_point
  (a c b : ℝ) (r : ℝ)
  (h_geom_prog : c = a * r ∧ b = a * r^2) :
  ∃ (P : ℝ × ℝ), ∀ (a c b : ℝ), c = a * r ∧ b = a * r^2 → (P = (0, 0) ∧ a ≠ 0) :=
by
  sorry

end geom_progression_lines_common_point_l1929_192975


namespace candies_leftover_l1929_192913

theorem candies_leftover (n : ℕ) : 31254389 % 6 = 5 :=
by {
  sorry
}

end candies_leftover_l1929_192913


namespace smallest_n_l1929_192978

def is_perfect_fourth (m : ℕ) : Prop := ∃ x : ℕ, m = x^4
def is_perfect_fifth (m : ℕ) : Prop := ∃ y : ℕ, m = y^5

theorem smallest_n :
  ∃ n : ℕ, n > 0 ∧ is_perfect_fourth (3 * n) ∧ is_perfect_fifth (2 * n) ∧ n = 6912 :=
by {
  sorry
}

end smallest_n_l1929_192978
