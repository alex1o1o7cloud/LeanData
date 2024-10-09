import Mathlib

namespace problem_1_problem_2_l1499_149904

noncomputable def f (a b x : ℝ) := a * (x - 1)^2 + b * Real.log x

theorem problem_1 (a : ℝ) (h_deriv : ∀ x ≥ 2, (2 * a * x^2 - 2 * a * x + 1) / x ≤ 0) : 
  a ≤ -1 / 4 :=
sorry

theorem problem_2 (a : ℝ) (h_ineq : ∀ x ≥ 1, a * (x - 1)^2 + Real.log x ≤ x - 1) : 
  a ≤ 0 :=
sorry

end problem_1_problem_2_l1499_149904


namespace h_at_8_l1499_149919

noncomputable def f (x : ℝ) : ℝ := x^3 - 3 * x + 2

noncomputable def h (x : ℝ) : ℝ :=
  let a := 1
  let b := 1
  let c := 2
  (1/2) * (x - a^3) * (x - b^3) * (x - c^3)

theorem h_at_8 : h 8 = 147 := 
by 
  sorry

end h_at_8_l1499_149919


namespace Iesha_num_books_about_school_l1499_149982

theorem Iesha_num_books_about_school (total_books sports_books : ℕ) (h1 : total_books = 58) (h2 : sports_books = 39) : total_books - sports_books = 19 :=
by
  sorry

end Iesha_num_books_about_school_l1499_149982


namespace ball_height_25_l1499_149941

theorem ball_height_25 (t : ℝ) (h : ℝ) 
  (h_eq : h = 45 - 7 * t - 6 * t^2) : 
  h = 25 ↔ t = 4 / 3 := 
by 
  sorry

end ball_height_25_l1499_149941


namespace koala_fiber_consumption_l1499_149990

theorem koala_fiber_consumption
  (absorbed_fiber : ℝ) (total_fiber : ℝ) 
  (h1 : absorbed_fiber = 0.40 * total_fiber)
  (h2 : absorbed_fiber = 12) :
  total_fiber = 30 := 
by
  sorry

end koala_fiber_consumption_l1499_149990


namespace initial_pineapple_sweets_l1499_149909

-- Define constants for initial number of flavored sweets and actions taken
def initial_cherry_sweets : ℕ := 30
def initial_strawberry_sweets : ℕ := 40
def total_remaining_sweets : ℕ := 55

-- Define Aaron's actions
def aaron_eats_half_sweets (n : ℕ) : ℕ := n / 2
def aaron_gives_to_friend : ℕ := 5

-- Calculate remaining sweets after Aaron's actions
def remaining_cherry_sweets : ℕ := initial_cherry_sweets - (aaron_eats_half_sweets initial_cherry_sweets) - aaron_gives_to_friend
def remaining_strawberry_sweets : ℕ := initial_strawberry_sweets - (aaron_eats_half_sweets initial_strawberry_sweets)

-- Define the problem to prove
theorem initial_pineapple_sweets :
  (total_remaining_sweets - (remaining_cherry_sweets + remaining_strawberry_sweets)) * 2 = 50 :=
by sorry -- Placeholder for the actual proof

end initial_pineapple_sweets_l1499_149909


namespace Hezekiah_age_l1499_149969

variable (H : ℕ)
variable (R : ℕ) -- Ryanne's age

-- Defining the conditions
def condition1 : Prop := R = H + 7
def condition2 : Prop := H + R = 15

-- The main theorem we want to prove
theorem Hezekiah_age : condition1 H R → condition2 H R → H = 4 :=
by  -- proof will be here
  sorry

end Hezekiah_age_l1499_149969


namespace sum_of_three_numbers_eq_16_l1499_149924

variable {a b c : ℝ}

theorem sum_of_three_numbers_eq_16
  (h1 : a^2 + b^2 + c^2 = 156)
  (h2 : a * b + b * c + c * a = 50) :
  a + b + c = 16 :=
by
  sorry

end sum_of_three_numbers_eq_16_l1499_149924


namespace find_a_l1499_149903

def A := { x : ℝ | x^2 + 4 * x = 0 }
def B (a : ℝ) := { x : ℝ | x^2 + 2 * (a + 1) * x + (a^2 - 1) = 0 }

theorem find_a (a : ℝ) :
  (∀ x : ℝ, x ∈ (A ∩ B a) ↔ x ∈ B a) → (a = 1 ∨ a ≤ -1) :=
by 
  sorry

end find_a_l1499_149903


namespace part_a_part_b_l1499_149913

-- Part (a)

theorem part_a : ∃ (a b : ℕ), 2015^2 + 2017^2 = 2 * (a^2 + b^2) :=
by
  -- The proof will go here
  sorry

-- Part (b)

theorem part_b (k n : ℕ) : ∃ (a b : ℕ), (2 * k + 1)^2 + (2 * n + 1)^2 = 2 * (a^2 + b^2) :=
by
  -- The proof will go here
  sorry

end part_a_part_b_l1499_149913


namespace number_of_customers_l1499_149974

theorem number_of_customers (total_sandwiches : ℕ) (office_orders : ℕ) (customers_half : ℕ) (num_offices : ℕ) (num_sandwiches_per_office : ℕ) 
  (sandwiches_per_customer : ℕ) (group_sandwiches : ℕ) (total_customers : ℕ) :
  total_sandwiches = 54 →
  num_offices = 3 →
  num_sandwiches_per_office = 10 →
  group_sandwiches = total_sandwiches - num_offices * num_sandwiches_per_office →
  customers_half * sandwiches_per_customer = group_sandwiches →
  sandwiches_per_customer = 4 →
  customers_half = total_customers / 2 →
  total_customers = 12 :=
by
  intros
  sorry

end number_of_customers_l1499_149974


namespace remainder_of_A_div_by_9_l1499_149916

theorem remainder_of_A_div_by_9 (A B : ℕ) (h : A = B * 9 + 13) : A % 9 = 4 := by
  sorry

end remainder_of_A_div_by_9_l1499_149916


namespace M_subsetneq_P_l1499_149984

def U : Set ℝ := Set.univ
def M : Set ℝ := {x : ℝ | x > 1}
def P : Set ℝ := {x : ℝ | x^2 > 1}

theorem M_subsetneq_P : M ⊂ P :=
by sorry

end M_subsetneq_P_l1499_149984


namespace jack_round_trip_speed_l1499_149952

noncomputable def jack_average_speed (d1 d2 : ℕ) (t1 t2 : ℕ) : ℕ :=
  let total_distance := d1 + d2
  let total_time := t1 + t2
  let total_time_hours := total_time / 60
  total_distance / total_time_hours

theorem jack_round_trip_speed : jack_average_speed 3 3 45 15 = 6 := by
  -- Import necessary library
  sorry

end jack_round_trip_speed_l1499_149952


namespace fraction_white_surface_area_l1499_149947

-- Definitions of the given conditions
def cube_side_length : ℕ := 4
def small_cubes : ℕ := 64
def black_cubes : ℕ := 34
def white_cubes : ℕ := 30
def total_surface_area : ℕ := 6 * cube_side_length^2
def black_faces_exposed : ℕ := 32 
def white_faces_exposed : ℕ := total_surface_area - black_faces_exposed

-- The proof statement
theorem fraction_white_surface_area (cube_side_length_eq : cube_side_length = 4)
                                    (small_cubes_eq : small_cubes = 64)
                                    (black_cubes_eq : black_cubes = 34)
                                    (white_cubes_eq : white_cubes = 30)
                                    (black_faces_eq : black_faces_exposed = 32)
                                    (total_surface_area_eq : total_surface_area = 96)
                                    (white_faces_eq : white_faces_exposed = 64) : 
                                    (white_faces_exposed : ℚ) / (total_surface_area : ℚ) = 2 / 3 :=
by
  sorry

end fraction_white_surface_area_l1499_149947


namespace right_triangle_median_l1499_149976

variable (A B C M N : Type) [LinearOrder B] [LinearOrder C] [LinearOrder A] [LinearOrder M] [LinearOrder N]
variable (AC BC AM BN AB : ℝ)
variable (right_triangle : AC * AC + BC * BC = AB * AB)
variable (median_A : AC * AC + (1 / 4) * BC * BC = 81)
variable (median_B : BC * BC + (1 / 4) * AC * AC = 99)

theorem right_triangle_median :
  ∀ (AC BC AB : ℝ),
  (AC * AC + BC * BC = 144) → (AC * AC + BC * BC = AB * AB) → AB = 12 :=
by
  intros
  sorry

end right_triangle_median_l1499_149976


namespace min_time_to_cook_cakes_l1499_149921

theorem min_time_to_cook_cakes (cakes : ℕ) (pot_capacity : ℕ) (time_per_side : ℕ) 
  (h1 : cakes = 3) (h2 : pot_capacity = 2) (h3 : time_per_side = 5) : 
  ∃ t, t = 15 := by
  sorry

end min_time_to_cook_cakes_l1499_149921


namespace sqrt_of_9_eq_3_l1499_149970

theorem sqrt_of_9_eq_3 : Real.sqrt 9 = 3 := by
  sorry

end sqrt_of_9_eq_3_l1499_149970


namespace average_of_five_digits_l1499_149987

theorem average_of_five_digits 
  (S : ℝ)
  (S3 : ℝ)
  (h_avg8 : S / 8 = 20)
  (h_avg3 : S3 / 3 = 33.333333333333336) :
  (S - S3) / 5 = 12 := 
by
  sorry

end average_of_five_digits_l1499_149987


namespace unique_not_in_range_l1499_149953

noncomputable def f (a b c d x : ℝ) : ℝ := (a * x + b) / (c * x + d)

theorem unique_not_in_range (a b c d : ℝ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) (h₃ : c ≠ 0) (h₄ : d ≠ 0)
  (h₅ : f a b c d 23 = 23) (h₆ : f a b c d 101 = 101) (h₇ : ∀ x ≠ -d / c, f a b c d (f a b c d x) = x) :
  (a / c) = 62 := 
 sorry

end unique_not_in_range_l1499_149953


namespace two_positive_numbers_inequality_three_positive_numbers_inequality_l1499_149900

theorem two_positive_numbers_inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (a + b) * (1 / a + 1 / b) ≥ 4 :=
by sorry

theorem three_positive_numbers_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a + b + c) * (1 / a + 1 / b + 1 / c) ≥ 9 :=
by sorry

end two_positive_numbers_inequality_three_positive_numbers_inequality_l1499_149900


namespace roots_separation_condition_l1499_149918

theorem roots_separation_condition (m n p q : ℝ)
  (h_1 : ∃ (x1 x2 : ℝ), x1 + x2 = -m ∧ x1 * x2 = n ∧ x1 ≠ x2)
  (h_2 : ∃ (x3 x4 : ℝ), x3 + x4 = -p ∧ x3 * x4 = q ∧ x3 ≠ x4)
  (h_3 : (∀ x1 x2 x3 x4 : ℝ, x1 + x2 = -m ∧ x1 * x2 = n ∧ x3 + x4 = -p ∧ x3 * x4 = q → 
         (x3 - x1) * (x3 - x2) * (x4 - x1) * (x4 - x2) < 0)) : 
  (n - q)^2 + (m - p) * (m * q - n * p) < 0 :=
sorry

end roots_separation_condition_l1499_149918


namespace largest_sum_product_l1499_149945

theorem largest_sum_product (p q : ℕ) (h1 : p * q = 100) (h2 : 0 < p) (h3 : 0 < q) : p + q ≤ 101 :=
sorry

end largest_sum_product_l1499_149945


namespace train_length_l1499_149981

theorem train_length (bridge_length time_seconds speed_kmh : ℝ) (S : speed_kmh = 64) (T : time_seconds = 45) (B : bridge_length = 300) : 
  ∃ (train_length : ℝ), train_length = 500 := 
by
  -- Add your proof here 
  sorry

end train_length_l1499_149981


namespace balloons_left_l1499_149948

def total_balloons (r w g c: Nat) : Nat := r + w + g + c

def num_friends : Nat := 10

theorem balloons_left (r w g c : Nat) (total := total_balloons r w g c) (h_r : r = 24) (h_w : w = 38) (h_g : g = 68) (h_c : c = 75) :
  total % num_friends = 5 := by
  sorry

end balloons_left_l1499_149948


namespace chord_length_l1499_149910

theorem chord_length (r : ℝ) (h : r = 15) :
  ∃ (cd : ℝ), cd = 13 * Real.sqrt 3 :=
by
  sorry

end chord_length_l1499_149910


namespace sum_of_squares_99_in_distinct_ways_l1499_149983

theorem sum_of_squares_99_in_distinct_ways : 
  ∃ a b c d e f g h i j k l : ℕ, 
    (a^2 + b^2 + c^2 + d^2 = 99) ∧ (e^2 + f^2 + g^2 + h^2 = 99) ∧ (i^2 + j^2 + k^2 + l^2 = 99) ∧ 
    (a ≠ e ∨ b ≠ f ∨ c ≠ g ∨ d ≠ h) ∧ 
    (a ≠ i ∨ b ≠ j ∨ c ≠ k ∨ d ≠ l) ∧ 
    (i ≠ e ∨ j ≠ f ∨ k ≠ g ∨ l ≠ h) 
    :=
sorry

end sum_of_squares_99_in_distinct_ways_l1499_149983


namespace area_PST_correct_l1499_149993

noncomputable def area_of_triangle_PST : ℚ :=
  let P : ℚ × ℚ := (0, 0)
  let Q : ℚ × ℚ := (4, 0)
  let R : ℚ × ℚ := (0, 4)
  let S : ℚ × ℚ := (0, 2)
  let T : ℚ × ℚ := (8 / 3, 4 / 3)
  1 / 2 * (|P.1 * (S.2 - T.2) + S.1 * (T.2 - P.2) + T.1 * (P.2 - S.2)|)

theorem area_PST_correct : area_of_triangle_PST = 8 / 3 := sorry

end area_PST_correct_l1499_149993


namespace find_scalars_l1499_149922

open Matrix

def B : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![2, -1],
    ![4, 3]]

def I : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![1, 0],
    ![0, 1]]

theorem find_scalars (r s : ℤ) (h : B^6 = r • B + s • I) :
  (r = 1125) ∧ (s = -1875) :=
sorry

end find_scalars_l1499_149922


namespace regular_octagon_interior_angle_l1499_149927

theorem regular_octagon_interior_angle:
  let n := 8
  let sum_interior_angles (n: ℕ) := 180 * (n - 2)
  let interior_angle (n: ℕ) := sum_interior_angles n / n
  interior_angle n = 135 :=
by
  sorry

end regular_octagon_interior_angle_l1499_149927


namespace part1_part2_l1499_149923

-- Condition: x = -1 is a solution to 2a + 4x = x + 5a
def is_solution_x (a x : ℤ) : Prop := 2 * a + 4 * x = x + 5 * a

-- Part 1: Prove a = -1 given x = -1
theorem part1 (x : ℤ) (h1 : x = -1) (h2 : is_solution_x a x) : a = -1 :=
by sorry

-- Condition: a = -1
def a_value (a : ℤ) : Prop := a = -1

-- Condition: ay + 6 = 6a + 2y
def equation_in_y (a y : ℤ) : Prop := a * y + 6 = 6 * a + 2 * y

-- Part 2: Prove y = 4 given a = -1
theorem part2 (a y : ℤ) (h1 : a_value a) (h2 : equation_in_y a y) : y = 4 :=
by sorry

end part1_part2_l1499_149923


namespace striped_shorts_difference_l1499_149943

variable (students : ℕ)
variable (striped_shirts checkered_shirts shorts : ℕ)

-- Conditions
variable (Hstudents : students = 81)
variable (Hstriped : striped_shirts = 2 * checkered_shirts)
variable (Hcheckered : checkered_shirts = students / 3)
variable (Hshorts : shorts = checkered_shirts + 19)

-- Goal
theorem striped_shorts_difference :
  striped_shirts - shorts = 8 :=
sorry

end striped_shorts_difference_l1499_149943


namespace complement_of_A_in_U_l1499_149962

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 4}
def complement_set (U A : Set ℕ) : Set ℕ := U \ A

theorem complement_of_A_in_U :
  complement_set U A = {2, 3, 5} :=
by
  apply Set.ext
  intro x
  simp [complement_set, U, A]
  sorry

end complement_of_A_in_U_l1499_149962


namespace find_g50_l1499_149999

noncomputable def g (x : ℝ) : ℝ := sorry

theorem find_g50 (g : ℝ → ℝ) (h : ∀ x y : ℝ, g (x * y) = y * g x)
  (h1 : g 1 = 10) : g 50 = 50 * 10 :=
by
  -- The proof sketch here; the detailed proof is omitted
  sorry

end find_g50_l1499_149999


namespace find_a2018_l1499_149961

-- Given Conditions
def initial_condition (a : ℕ → ℤ) : Prop :=
  a 1 = -1

def absolute_difference (a : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, n ≥ 2 → abs (a n - a (n-1)) = 2^(n-1)

def subseq_decreasing (a : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, a (2*n-1) > a (2*(n+1)-1)

def subseq_increasing (a : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, a (2*n) < a (2*(n+1))

-- Theorem to Prove
theorem find_a2018 (a : ℕ → ℤ)
  (h1 : initial_condition a)
  (h2 : absolute_difference a)
  (h3 : subseq_decreasing a)
  (h4 : subseq_increasing a) :
  a 2018 = (2^2018 - 1) / 3 :=
sorry

end find_a2018_l1499_149961


namespace cos_double_beta_eq_24_over_25_l1499_149989

theorem cos_double_beta_eq_24_over_25
  (α β : ℝ)
  (h1 : Real.sin (α - β) = 3 / 5)
  (h2 : Real.cos (α + β) = -3 / 5)
  (h3 : α - β ∈ Set.Ioo (π / 2) π)
  (h4 : α + β ∈ Set.Ioo (π / 2) π) :
  Real.cos (2 * β) = 24 / 25 := sorry

end cos_double_beta_eq_24_over_25_l1499_149989


namespace speeds_and_time_l1499_149958

theorem speeds_and_time (x s : ℕ) (t : ℝ)
  (h1 : ∀ {t : ℝ}, t = 2 → x * t > s * t + 24)
  (h2 : ∀ {t : ℝ}, t = 0.5 → x * t = 8) :
  x = 16 ∧ s = 4 ∧ t = 8 :=
by {
  sorry
}

end speeds_and_time_l1499_149958


namespace factorial_mod_13_l1499_149971

open Nat

theorem factorial_mod_13 :
  let n := 10
  let p := 13
  n! % p = 6 := by
sorry

end factorial_mod_13_l1499_149971


namespace B_N_Q_collinear_l1499_149929

/-- Define point positions -/
structure Point where
  x : ℝ
  y : ℝ

def M : Point := ⟨-1, 0⟩
def N : Point := ⟨1, 0⟩

/-- Define the curve C -/
def on_curve_C (P : Point) : Prop :=
  P.x^2 + P.y^2 - 6 * P.x + 1 = 0

/-- Define reflection of point A across the x-axis -/
def reflection_across_x (A : Point) : Point :=
  ⟨A.x, -A.y⟩

/-- Define the condition that line l passes through M and intersects curve C at two distinct points A and B -/
def line_l_condition (A B: Point) (k : ℝ) (hk : k ≠ 0) : Prop :=
  A.y = k * (A.x + 1) ∧ B.y = k * (B.x + 1) ∧ on_curve_C A ∧ on_curve_C B

/-- Main theorem to prove collinearity of B, N, Q -/
theorem B_N_Q_collinear (A B : Point) (k : ℝ) (hk : k ≠ 0)
  (hA : on_curve_C A) (hB : on_curve_C B)
  (h_l : line_l_condition A B k hk) :
  let Q := reflection_across_x A
  (B.x - N.x) * (Q.y - N.y) = (B.y - N.y) * (Q.x - N.x) :=
sorry

end B_N_Q_collinear_l1499_149929


namespace no_real_solution_l1499_149986

theorem no_real_solution :
    ∀ x : ℝ, (5 * x^2 - 3 * x + 2) / (x + 2) ≠ 2 * x - 3 :=
by
  intro x
  sorry

end no_real_solution_l1499_149986


namespace proposition_C_is_true_l1499_149950

theorem proposition_C_is_true :
  (∀ θ : ℝ, 90 < θ ∧ θ < 180 → θ > 90) :=
by
  sorry

end proposition_C_is_true_l1499_149950


namespace train_crossing_time_l1499_149994

-- Definitions based on conditions from the problem
def length_of_train_and_platform := 900 -- in meters
def speed_km_per_hr := 108 -- in km/hr
def distance := 2 * length_of_train_and_platform -- distance to be covered
def speed_m_per_s := (speed_km_per_hr * 1000) / 3600 -- converted speed

-- Theorem stating the time to cross the platform is 60 seconds
theorem train_crossing_time : distance / speed_m_per_s = 60 := by
  sorry

end train_crossing_time_l1499_149994


namespace triangle_geometry_l1499_149915

theorem triangle_geometry 
  (A : ℝ × ℝ) 
  (hA : A = (5,1))
  (median_CM : ∀ x y : ℝ, 2 * x - y - 5 = 0)
  (altitude_BH : ∀ x y : ℝ, x - 2 * y - 5 = 0):
  (∀ x y : ℝ, 2 * x + y - 11 = 0) ∧
  (4, 3) ∈ {(x, y) | 2 * x + y = 11 ∧ 2 * x - y = 5} :=
by
  sorry

end triangle_geometry_l1499_149915


namespace upstream_distance_l1499_149912

variable (Vb Vs Vdown Vup Dup : ℕ)

def boatInStillWater := Vb = 36
def speedStream := Vs = 12
def downstreamSpeed := Vdown = Vb + Vs
def upstreamSpeed := Vup = Vb - Vs
def timeEquality := 80 / Vdown = Dup / Vup

theorem upstream_distance (Vb Vs Vdown Vup Dup : ℕ) 
  (h1 : boatInStillWater Vb)
  (h2 : speedStream Vs)
  (h3 : downstreamSpeed Vb Vs Vdown)
  (h4 : upstreamSpeed Vb Vs Vup)
  (h5 : timeEquality Vdown Vup Dup) : Dup = 40 := 
sorry

end upstream_distance_l1499_149912


namespace simplify_expression_l1499_149942

theorem simplify_expression :
  (2 * (Real.sqrt 2 + Real.sqrt 6)) / (3 * Real.sqrt (2 + Real.sqrt 3)) = 4 / 3 :=
by
  sorry

end simplify_expression_l1499_149942


namespace intersection_S_T_eq_U_l1499_149964

def S : Set ℝ := {x | abs x < 5}
def T : Set ℝ := {x | (x + 7) * (x - 3) < 0}
def U : Set ℝ := {x | -5 < x ∧ x < 3}

theorem intersection_S_T_eq_U : (S ∩ T) = U := 
by 
  sorry

end intersection_S_T_eq_U_l1499_149964


namespace ccamathbonanza_2016_2_1_l1499_149995

-- Definitions of the speeds of the runners
def bhairav_speed := 28 -- in miles per hour
def daniel_speed := 15 -- in miles per hour
def tristan_speed := 10 -- in miles per hour

-- Distance of the race
def race_distance := 15 -- in miles

-- Time conversion from hours to minutes
def hours_to_minutes (hours : ℚ) : ℚ := hours * 60

-- Time taken by each runner to complete the race (in hours)
def time_bhairav := race_distance / bhairav_speed
def time_daniel := race_distance / daniel_speed
def time_tristan := race_distance / tristan_speed

-- Time taken by each runner to complete the race (in minutes)
def time_bhairav_minutes := hours_to_minutes time_bhairav
def time_daniel_minutes := hours_to_minutes time_daniel
def time_tristan_minutes := hours_to_minutes time_tristan

-- Time differences between consecutive runners' finishes (in minutes)
def time_diff_bhairav_daniel := time_daniel_minutes - time_bhairav_minutes
def time_diff_daniel_tristan := time_tristan_minutes - time_daniel_minutes

-- Greatest length of time between consecutive runners' finishes
def greatest_time_diff := max time_diff_bhairav_daniel time_diff_daniel_tristan

-- The theorem we need to prove
theorem ccamathbonanza_2016_2_1 : greatest_time_diff = 30 := by
  sorry

end ccamathbonanza_2016_2_1_l1499_149995


namespace find_cookies_on_second_plate_l1499_149997

theorem find_cookies_on_second_plate (a : ℕ → ℕ) :
  (a 1 = 5) ∧ (a 3 = 10) ∧ (a 4 = 14) ∧ (a 5 = 19) ∧ (a 6 = 25) ∧
  (∀ n, a (n + 2) - a (n + 1) = if (n + 1) % 2 = 0 then 5 else 4) →
  a 2 = 5 :=
by
  sorry

end find_cookies_on_second_plate_l1499_149997


namespace solution_amount_of_solution_A_l1499_149905

-- Define the conditions
variables (x y : ℝ)
variables (h1 : x + y = 140)
variables (h2 : 0.40 * x + 0.90 * y = 0.80 * 140)

-- State the theorem
theorem solution_amount_of_solution_A : x = 28 :=
by
  -- Here, the proof would be provided, but we replace it with sorry
  sorry

end solution_amount_of_solution_A_l1499_149905


namespace calc_4_op_3_l1499_149936

def specific_op (m n : ℕ) : ℕ := n^2 - m

theorem calc_4_op_3 :
  specific_op 4 3 = 5 :=
by
  sorry

end calc_4_op_3_l1499_149936


namespace trapezium_shorter_side_l1499_149935

theorem trapezium_shorter_side (a b h : ℝ) (H1 : a = 10) (H2 : b = 18) (H3 : h = 10.00001) : a = 10 :=
by
  sorry

end trapezium_shorter_side_l1499_149935


namespace departure_of_30_tons_of_grain_l1499_149951

-- Define positive as an arrival of grain.
def positive_arrival (x : ℤ) : Prop := x > 0

-- Define negative as a departure of grain.
def negative_departure (x : ℤ) : Prop := x < 0

-- The given conditions and question translated to a Lean statement.
theorem departure_of_30_tons_of_grain :
  (positive_arrival 30) → (negative_departure (-30)) :=
by
  intro pos30
  sorry

end departure_of_30_tons_of_grain_l1499_149951


namespace sum_of_distances_l1499_149937

theorem sum_of_distances (a b : ℤ) (k : ℕ) 
  (h1 : |k - a| + |(k + 1) - a| + |(k + 2) - a| + |(k + 3) - a| + |(k + 4) - a| + |(k + 5) - a| + |(k + 6) - a| = 609)
  (h2 : |k - b| + |(k + 1) - b| + |(k + 2) - b| + |(k + 3) - b| + |(k + 4) - b| + |(k + 5) - b| + |(k + 6) - b| = 721)
  (h3 : a + b = 192) :
  a = 1 ∨ a = 104 ∨ a = 191 := 
sorry

end sum_of_distances_l1499_149937


namespace mode_is_necessary_characteristic_of_dataset_l1499_149925

-- Define a dataset as a finite set of elements from any type.
variable {α : Type*} [Fintype α]

-- Define a mode for a dataset as an element that occurs most frequently.
def mode (dataset : Multiset α) : α :=
sorry  -- Mode definition and computation are omitted for this high-level example.

-- Define the theorem that mode is a necessary characteristic of a dataset.
theorem mode_is_necessary_characteristic_of_dataset (dataset : Multiset α) : 
  exists mode_elm : α, mode_elm = mode dataset :=
sorry

end mode_is_necessary_characteristic_of_dataset_l1499_149925


namespace sextuple_angle_terminal_side_on_xaxis_l1499_149944

-- Define angle and conditions
variable (α : ℝ)
variable (isPositiveAngle : 0 < α ∧ α < 360)
variable (sextupleAngleOnXAxis : ∃ k : ℕ, 6 * α = k * 360)

-- Prove the possible values of the angle
theorem sextuple_angle_terminal_side_on_xaxis :
  α = 60 ∨ α = 120 ∨ α = 180 ∨ α = 240 ∨ α = 300 :=
  sorry

end sextuple_angle_terminal_side_on_xaxis_l1499_149944


namespace regular_price_one_bag_l1499_149933

theorem regular_price_one_bag (p : ℕ) (h : 3 * p + 5 = 305) : p = 100 :=
by
  sorry

end regular_price_one_bag_l1499_149933


namespace fg_difference_l1499_149917

def f (x : ℝ) : ℝ := 2 * x + 5
def g (x : ℝ) : ℝ := 4 * x - 1

theorem fg_difference : f (g 3) - g (f 3) = -16 := by
  sorry

end fg_difference_l1499_149917


namespace ratio_tuesday_monday_l1499_149902

-- Define the conditions
variables (M T W : ℕ) (hM : M = 450) (hW : W = 300) (h_rel : W = T + 75)

-- Define the theorem
theorem ratio_tuesday_monday : (T : ℚ) / M = 1 / 2 :=
by
  -- Sorry means the proof has been omitted in Lean.
  sorry

end ratio_tuesday_monday_l1499_149902


namespace number_divided_by_3_equals_subtract_3_l1499_149960

theorem number_divided_by_3_equals_subtract_3 (x : ℝ) (h : x / 3 = x - 3) : x = 4.5 :=
sorry

end number_divided_by_3_equals_subtract_3_l1499_149960


namespace isosceles_triangle_perimeter_l1499_149931

theorem isosceles_triangle_perimeter (m x₁ x₂ : ℝ) (h₁ : 1^2 + m * 1 + 5 = 0) 
  (hx : x₁^2 + m * x₁ + 5 = 0 ∧ x₂^2 + m * x₂ + 5 = 0)
  (isosceles : (x₁ = x₂ ∨ x₁ = 1 ∨ x₂ = 1)) : 
  ∃ (P : ℝ), P = 11 :=
by 
  -- Here, you'd prove that under these conditions, the perimeter must be 11.
  sorry

end isosceles_triangle_perimeter_l1499_149931


namespace minimum_passed_l1499_149949

def total_participants : Nat := 100
def num_questions : Nat := 10
def correct_answers : List Nat := [93, 90, 86, 91, 80, 83, 72, 75, 78, 59]
def passing_criteria : Nat := 6

theorem minimum_passed (total_participants : ℕ) (num_questions : ℕ) (correct_answers : List ℕ) (passing_criteria : ℕ) :
  100 = total_participants → 10 = num_questions → correct_answers = [93, 90, 86, 91, 80, 83, 72, 75, 78, 59] →
  passing_criteria = 6 → 
  ∃ p : ℕ, p = 62 := 
by
  sorry

end minimum_passed_l1499_149949


namespace rational_sqrt_condition_l1499_149978

variable (r q n : ℚ)

theorem rational_sqrt_condition
  (h : (1 / (r + q * n) + 1 / (q + r * n) = 1 / (r + q))) : 
  ∃ x : ℚ, x^2 = (n - 3) / (n + 1) :=
sorry

end rational_sqrt_condition_l1499_149978


namespace negation_of_proposition_l1499_149992

theorem negation_of_proposition :
  (¬ (∀ x : ℝ, x > 1 → x - 1 > Real.log x)) ↔ ∃ x : ℝ, x > 1 ∧ x - 1 ≤ Real.log x :=
sorry

end negation_of_proposition_l1499_149992


namespace sum_of_polynomials_l1499_149956

-- Define the given polynomials f, g, and h
def f (x : ℝ) : ℝ := -6 * x^3 - 4 * x^2 + 2 * x - 5
def g (x : ℝ) : ℝ := -7 * x^2 + 6 * x - 9
def h (x : ℝ) : ℝ := 6 * x^2 + 7 * x + 3

-- Prove that the sum of f(x), g(x), and h(x) is a specific polynomial
theorem sum_of_polynomials (x : ℝ) : 
  f x + g x + h x = -6 * x^3 - 5 * x^2 + 15 * x - 11 := 
by {
  -- Proof is omitted
  sorry
}

end sum_of_polynomials_l1499_149956


namespace division_of_fractions_l1499_149901

theorem division_of_fractions :
  (5 / 6) / (9 / 10) = 25 / 27 :=
by sorry

end division_of_fractions_l1499_149901


namespace trig_identity_l1499_149911

theorem trig_identity (f : ℝ → ℝ) (x : ℝ) (h : f (Real.sin x) = 3 - Real.cos (2 * x)) : f (Real.cos x) = 3 + Real.cos (2 * x) :=
sorry

end trig_identity_l1499_149911


namespace sahil_machine_purchase_price_l1499_149932

theorem sahil_machine_purchase_price
  (repair_cost : ℕ)
  (transportation_cost : ℕ)
  (selling_price : ℕ)
  (profit_percent : ℤ)
  (purchase_price : ℕ)
  (total_cost : ℕ)
  (profit_ratio : ℚ)
  (h1 : repair_cost = 5000)
  (h2 : transportation_cost = 1000)
  (h3 : selling_price = 30000)
  (h4 : profit_percent = 50)
  (h5 : total_cost = purchase_price + repair_cost + transportation_cost)
  (h6 : profit_ratio = profit_percent / 100)
  (h7 : selling_price = (1 + profit_ratio) * total_cost) :
  purchase_price = 14000 :=
by
  sorry

end sahil_machine_purchase_price_l1499_149932


namespace no_real_solutions_l1499_149914

theorem no_real_solutions (x : ℝ) : (x - 3 * x + 7)^2 + 1 ≠ -|x| :=
by
  -- The statement of the theorem is sufficient; the proof is not needed as per indicated instructions.
  sorry

end no_real_solutions_l1499_149914


namespace find_b_l1499_149928

noncomputable def g (b : ℝ) (x : ℝ) : ℝ := b * x ^ 3 + b * x - 3

theorem find_b (b : ℝ) (h : g b (g b 1) = 1) : b = 1 / 2 :=
by
  sorry

end find_b_l1499_149928


namespace a_5_eq_14_l1499_149957

def S (n : ℕ) : ℚ := (3 / 2) * n ^ 2 + (1 / 2) * n

def a (n : ℕ) : ℚ := S n - S (n - 1)

theorem a_5_eq_14 : a 5 = 14 := by {
  -- Proof steps go here
  sorry
}

end a_5_eq_14_l1499_149957


namespace train_length_is_225_m_l1499_149965

noncomputable def speed_kmph : ℝ := 90
noncomputable def time_s : ℝ := 9

noncomputable def speed_ms : ℝ := speed_kmph / 3.6
noncomputable def distance_m (speed : ℝ) (time : ℝ) : ℝ := speed * time

theorem train_length_is_225_m :
  distance_m speed_ms time_s = 225 := by
  sorry

end train_length_is_225_m_l1499_149965


namespace volume_pyramid_correct_l1499_149991

noncomputable def volume_of_regular_triangular_pyramid 
  (R : ℝ) (β : ℝ) (a : ℝ) : ℝ :=
  (a^3 * (Real.tan β)) / 24

theorem volume_pyramid_correct 
  (R : ℝ) (β : ℝ) (a : ℝ) : 
  volume_of_regular_triangular_pyramid R β a = (a^3 * (Real.tan β)) / 24 :=
sorry

end volume_pyramid_correct_l1499_149991


namespace range_of_a_l1499_149934

theorem range_of_a (a : ℝ) : (-1/Real.exp 1 < a ∧ a < 0) ∨ (0 < a ∧ a < 1/Real.exp 1) :=
  sorry

end range_of_a_l1499_149934


namespace suitable_comprehensive_survey_l1499_149975

def investigate_service_life_of_lamps : Prop := 
  -- This would typically involve checking a subset rather than every lamp
  sorry

def investigate_water_quality : Prop := 
  -- This would typically involve sampling rather than checking every point
  sorry

def investigate_sports_activities : Prop := 
  -- This would typically involve sampling rather than collecting data on every student
  sorry

def test_components_of_rocket : Prop := 
  -- Given the critical importance and manageable number of components, this requires comprehensive examination
  sorry

def most_suitable_for_comprehensive_survey : Prop :=
  test_components_of_rocket ∧ ¬investigate_service_life_of_lamps ∧ 
  ¬investigate_water_quality ∧ ¬investigate_sports_activities

theorem suitable_comprehensive_survey : most_suitable_for_comprehensive_survey :=
  sorry

end suitable_comprehensive_survey_l1499_149975


namespace min_sum_abs_l1499_149940

theorem min_sum_abs (x : ℝ) : ∃ m, m = 4 ∧ ∀ x : ℝ, |x + 2| + |x - 2| + |x - 1| ≥ m := 
sorry

end min_sum_abs_l1499_149940


namespace problem_solution_l1499_149980

theorem problem_solution (m n : ℤ) (h : m + 1 = (n - 2) / 3) : 3 * m - n = -5 :=
by
  sorry

end problem_solution_l1499_149980


namespace number_of_dogs_l1499_149954

variable (D C : ℕ)
variable (x : ℚ)

-- Conditions
def ratio_dogs_to_cats := D = (x * (C: ℚ) / 7)
def new_ratio_dogs_to_cats := D = (15 / 11) * (C + 8)

theorem number_of_dogs (h1 : ratio_dogs_to_cats D C x) (h2 : new_ratio_dogs_to_cats D C) : D = 77 := 
by sorry

end number_of_dogs_l1499_149954


namespace complex_point_in_fourth_quadrant_l1499_149938

theorem complex_point_in_fourth_quadrant (z : ℂ) (h : z = 1 / (1 + I)) :
  z.re > 0 ∧ z.im < 0 :=
by
  -- Here we would provide the proof, but it is omitted as per the instructions.
  sorry

end complex_point_in_fourth_quadrant_l1499_149938


namespace interval_solution_length_l1499_149908

theorem interval_solution_length (a b : ℝ) (h : (b - a) / 3 = 8) : b - a = 24 := by
  sorry

end interval_solution_length_l1499_149908


namespace problem1_problem2_l1499_149963

theorem problem1 (x : ℚ) (h : x ≠ -4) : (3 - x) / (x + 4) = 1 / 2 → x = 2 / 3 :=
by
  sorry

theorem problem2 (x : ℚ) (h : x ≠ 1) : x / (x - 1) - 2 * x / (3 * (x - 1)) = 1 → x = 3 / 2 :=
by
  sorry

end problem1_problem2_l1499_149963


namespace ratio_of_areas_l1499_149939

-- Defining the variables for sides of rectangles
variables {a b c d : ℝ}

-- Given conditions
axiom h1 : a / c = 4 / 5
axiom h2 : b / d = 4 / 5

-- Statement to prove the ratio of areas
theorem ratio_of_areas (h1 : a / c = 4 / 5) (h2 : b / d = 4 / 5) : (a * b) / (c * d) = 16 / 25 :=
sorry

end ratio_of_areas_l1499_149939


namespace show_watching_days_l1499_149967

def numberOfEpisodes := 20
def lengthOfEachEpisode := 30
def dailyWatchingTime := 2

theorem show_watching_days:
  (numberOfEpisodes * lengthOfEachEpisode) / 60 / dailyWatchingTime = 5 := 
by
  sorry

end show_watching_days_l1499_149967


namespace intersection_point_is_neg3_l1499_149972

def f (x : ℝ) : ℝ := x^3 + 6 * x^2 + 9 * x + 15

theorem intersection_point_is_neg3 :
  ∃ a b : ℝ, (f a = b) ∧ (f b = a) ∧ (a, b) = (-3, -3) := sorry

end intersection_point_is_neg3_l1499_149972


namespace trig_inequality_l1499_149988

open Real

theorem trig_inequality (x : ℝ) (n m : ℕ) (hx : 0 < x ∧ x < π / 2) (hnm : n > m) : 
  2 * abs (sin x ^ n - cos x ^ n) ≤ 3 * abs (sin x ^ m - cos x ^ m) := 
sorry

end trig_inequality_l1499_149988


namespace total_rainfall_january_l1499_149946

theorem total_rainfall_january (R1 R2 T : ℝ) (h1 : R2 = 1.5 * R1) (h2 : R2 = 21) : T = 35 :=
by 
  let R1 := 14
  let R2 := 21
  let T := R1 + R2
  sorry

end total_rainfall_january_l1499_149946


namespace line_tangent_to_circle_l1499_149998

theorem line_tangent_to_circle (l : ℝ → ℝ) (P : ℝ × ℝ) 
  (hP1 : P = (0, 1)) (hP2 : ∀ x y : ℝ, x^2 + y^2 = 1 -> l x = y)
  (hTangent : ∀ x y : ℝ, l x = y ↔ x^2 + y^2 = 1 ∧ y = 1):
  l x = 1 := by
  sorry

end line_tangent_to_circle_l1499_149998


namespace seats_not_occupied_l1499_149977

theorem seats_not_occupied (seats_per_row : ℕ) (rows : ℕ) (fraction_allowed : ℚ) (total_seats : ℕ) (allowed_seats_per_row : ℕ) (allowed_total : ℕ) (unoccupied_seats : ℕ) :
  seats_per_row = 8 →
  rows = 12 →
  fraction_allowed = 3 / 4 →
  total_seats = seats_per_row * rows →
  allowed_seats_per_row = seats_per_row * fraction_allowed →
  allowed_total = allowed_seats_per_row * rows →
  unoccupied_seats = total_seats - allowed_total →
  unoccupied_seats = 24 :=
by sorry

end seats_not_occupied_l1499_149977


namespace bus_speed_including_stoppages_l1499_149955

theorem bus_speed_including_stoppages 
  (speed_excl_stoppages : ℚ) 
  (ten_minutes_per_hour : ℚ) 
  (bus_stops_for_10_minutes : ten_minutes_per_hour = 10/60) 
  (speed_is_54_kmph : speed_excl_stoppages = 54) : 
  (speed_excl_stoppages * (1 - ten_minutes_per_hour)) = 45 := 
by 
  sorry

end bus_speed_including_stoppages_l1499_149955


namespace evaluate_f_at_minus_2_l1499_149926

def f (x : ℚ) : ℚ := (2 * x - 3) / (4 * x + 5)

theorem evaluate_f_at_minus_2 : f (-2) = 7 / 3 := by
  -- Proof is omitted
  sorry

end evaluate_f_at_minus_2_l1499_149926


namespace samantha_last_name_length_l1499_149920

/-
Given:
1. Jamie’s last name "Grey" has 4 letters.
2. If Bobbie took 2 letters off her last name, her last name would have twice the length of Jamie’s last name.
3. Samantha’s last name has 3 fewer letters than Bobbie’s last name.

Prove:
- Samantha's last name contains 7 letters.
-/

theorem samantha_last_name_length : 
  ∀ (Jamie Bobbie Samantha : ℕ),
    Jamie = 4 →
    Bobbie - 2 = 2 * Jamie →
    Samantha = Bobbie - 3 →
    Samantha = 7 :=
by
  intros Jamie Bobbie Samantha hJamie hBobbie hSamantha
  sorry

end samantha_last_name_length_l1499_149920


namespace negation_of_universal_proposition_l1499_149968

theorem negation_of_universal_proposition :
  (¬ ∀ (x : ℝ), x^2 ≥ 0) ↔ ∃ (x : ℝ), x^2 < 0 :=
by sorry

end negation_of_universal_proposition_l1499_149968


namespace inequality_solution_l1499_149973

theorem inequality_solution (x : ℝ) :
  (2 / (x - 3) ≤ 5) ↔ (x < 3 ∨ x ≥ 17 / 5) := 
sorry

end inequality_solution_l1499_149973


namespace anne_more_drawings_l1499_149979

/-- Anne's markers problem setup. -/
structure MarkerProblem :=
  (markers : ℕ)
  (drawings_per_marker : ℚ)
  (drawings_made : ℕ)

-- Given conditions
def anne_conditions : MarkerProblem :=
  { markers := 12, drawings_per_marker := 1.5, drawings_made := 8 }

-- Equivalent proof problem statement in Lean
theorem anne_more_drawings(conditions : MarkerProblem) : 
  conditions.markers * conditions.drawings_per_marker - conditions.drawings_made = 10 :=
by
  -- The proof of this theorem is omitted
  sorry

end anne_more_drawings_l1499_149979


namespace lion_king_box_office_earnings_l1499_149930

-- Definitions and conditions
def cost_lion_king : ℕ := 10  -- Lion King cost 10 million
def cost_star_wars : ℕ := 25  -- Star Wars cost 25 million
def earnings_star_wars : ℕ := 405  -- Star Wars earned 405 million

-- Calculate profit of Star Wars
def profit_star_wars : ℕ := earnings_star_wars - cost_star_wars

-- Define the profit of The Lion King, given it's half of Star Wars' profit
def profit_lion_king : ℕ := profit_star_wars / 2

-- Calculate the earnings of The Lion King
def earnings_lion_king : ℕ := cost_lion_king + profit_lion_king

-- Theorem to prove
theorem lion_king_box_office_earnings : earnings_lion_king = 200 :=
by
  sorry

end lion_king_box_office_earnings_l1499_149930


namespace equilateral_triangle_area_with_inscribed_circle_l1499_149996

theorem equilateral_triangle_area_with_inscribed_circle
  (r : ℝ) (area_circle : ℝ) (area_triangle : ℝ) 
  (h_inscribed_circle_area : area_circle = 9 * Real.pi)
  (h_radius : r = 3) :
  area_triangle = 27 * Real.sqrt 3 :=
by
  sorry

end equilateral_triangle_area_with_inscribed_circle_l1499_149996


namespace harry_started_with_79_l1499_149966

-- Definitions using the conditions
def harry_initial_apples (x : ℕ) : Prop :=
  (x + 5 = 84)

-- Theorem statement proving the initial number of apples Harry started with
theorem harry_started_with_79 : ∃ x : ℕ, harry_initial_apples x ∧ x = 79 :=
by
  sorry

end harry_started_with_79_l1499_149966


namespace smallest_positive_root_l1499_149907

noncomputable def alpha : ℝ := Real.arctan (2 / 9)
noncomputable def beta : ℝ := Real.arctan (6 / 7)

theorem smallest_positive_root :
  ∃ x > 0, (2 * Real.sin (6 * x) + 9 * Real.cos (6 * x) = 6 * Real.sin (2 * x) + 7 * Real.cos (2 * x))
    ∧ x = (alpha + beta) / 8 := sorry

end smallest_positive_root_l1499_149907


namespace number_of_employees_l1499_149959

def fixed_time_coffee : ℕ := 5
def time_per_status_update : ℕ := 2
def time_per_payroll_update : ℕ := 3
def total_morning_routine : ℕ := 50

def time_per_employee : ℕ := time_per_status_update + time_per_payroll_update
def time_spent_on_employees : ℕ := total_morning_routine - fixed_time_coffee

theorem number_of_employees : (time_spent_on_employees / time_per_employee) = 9 := by
  sorry

end number_of_employees_l1499_149959


namespace unique_flavors_l1499_149985

noncomputable def distinctFlavors : Nat :=
  let redCandies := 5
  let greenCandies := 4
  let blueCandies := 2
  (90 - 15 - 18 - 30 + 3 + 5 + 6) / 3  -- Adjustments and consideration for equivalent ratios.
  
theorem unique_flavors :
  distinctFlavors = 11 :=
  by
    sorry

end unique_flavors_l1499_149985


namespace smallest_positive_m_l1499_149906

theorem smallest_positive_m {m p q : ℤ} (h_eq : 12 * p^2 - m * p - 360 = 0) (h_pq : p * q = -30) :
  (m = 12 * (p + q)) → 0 < m → m = 12 :=
by
  sorry

end smallest_positive_m_l1499_149906
