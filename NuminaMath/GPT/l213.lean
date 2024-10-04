import Mathlib

namespace bridge_length_is_correct_l213_213547

variable (train_length : ℝ) (train_speed_kmph : ℝ) (time_to_cross : ℝ)

def train_speed_mps : ℝ := train_speed_kmph * 1000 / 3600

def total_distance_covered : ℝ := train_speed_mps * time_to_cross

def bridge_length (train_length total_distance_covered : ℝ) : ℝ :=
  total_distance_covered - train_length

theorem bridge_length_is_correct :
  let train_length : ℝ := 145
  let train_speed_kmph : ℝ := 54
  let time_to_cross : ℝ := 53.66237367677253
  let bridge_length_calculated := bridge_length train_length (total_distance_covered train_length train_speed_kmph time_to_cross) 
  abs (bridge_length_calculated - 659.94) < 0.01 :=
by {
  sorry  -- This will be where the proof goes
}

end bridge_length_is_correct_l213_213547


namespace intersection_of_rectangle_and_circle_l213_213900

noncomputable def intersection_area : ℝ :=
  let rect := [(3, 14), (18, 14), (18, 3), (3, 3)]
  let circle_center := (3, 3)
  let radius := 4
  4 * Real.pi

theorem intersection_of_rectangle_and_circle :
  let rect := [(3, 14), (18, 14), (18, 3), (3, 3)]
  let circle_center := (3, 3)
  let radius := 4
  ∃ r : ℝ, r = 4 * Real.pi ∧
    (area_of_intersection rect circle_center radius = r) :=
begin
  sorry
end

end intersection_of_rectangle_and_circle_l213_213900


namespace symmedian_point_is_midpoint_of_altitude_l213_213435

/-- In a right-angled triangle ABC with a right angle at C, prove that the symmedian point of the triangle is the midpoint of the altitude CH. -/
theorem symmedian_point_is_midpoint_of_altitude (A B C H N : Point) (H_right_angle : angle A C B = 90) (H_orthocenter : orthocenter A B C = H) (H_midpoint : midpoint C H = N) : 
  is_symmedian_point A B C N → N = midpoint C H := 
by 
  sorry

end symmedian_point_is_midpoint_of_altitude_l213_213435


namespace exists_a_lt_0_l213_213135

noncomputable def f : ℝ → ℝ :=
sorry

theorem exists_a_lt_0 (f : ℝ → ℝ) (h1 : ∀ x y : ℝ, 0 < x → 0 < y → f (Real.sqrt (x * y)) = (f x + f y) / 2)
  (h2 : ∀ x y : ℝ, 0 < x → 0 < y → x < y → f x < f y) :
  ∃ a : ℝ, 0 < a ∧ f a < 0 :=
sorry

end exists_a_lt_0_l213_213135


namespace find_years_for_total_interest_l213_213533

-- Define variables and constants
variables (n : ℝ)
constant principal1 : ℝ := 1000
constant rate1 : ℝ := 3 / 100
constant principal2 : ℝ := 1400
constant rate2 : ℝ := 5 / 100
constant total_interest : ℝ := 350

-- Define expressions for interest calculations
def interest1 := principal1 * rate1 * n
def interest2 := principal2 * rate2 * n

-- State the theorem
theorem find_years_for_total_interest :
  interest1 + interest2 = total_interest → n = 3.5 :=
by 
  sorry

end find_years_for_total_interest_l213_213533


namespace bart_earns_020_per_question_l213_213563

noncomputable def bart_income_per_question (questions_per_survey : ℕ)
                                            (surveys_monday : ℕ)
                                            (surveys_tuesday : ℕ)
                                            (total_earning : ℝ) : ℝ :=
(total_earning / (questions_per_survey * (surveys_monday + surveys_tuesday)))

theorem bart_earns_020_per_question :
  bart_income_per_question 10 3 4 14 = 0.20 :=
by
  unfold bart_income_per_question
  norm_num
  sorry

end bart_earns_020_per_question_l213_213563


namespace hypotenuse_length_l213_213967

theorem hypotenuse_length (a b c : ℝ) (h₁ : a + b + c = 40) (h₂ : 0.5 * a * b = 24) (h₃ : a^2 + b^2 = c^2) : c = 18.8 := sorry

end hypotenuse_length_l213_213967


namespace smallest_x_value_l213_213001

theorem smallest_x_value : ∃ x : ℝ, (x = 0) ∧ (∀ y : ℝ, (left_side y x) = 0 → x ≤ y)
where
  left_side y x : ℝ := (y^2 + y - 20)

limit smaller

end smallest_x_value_l213_213001


namespace intersection_chord_length_l213_213751

-- Conditions
def parametric_line (t : ℝ) : ℝ × ℝ :=
  (1 + t, 2 + t)

def polar_circle (ρ θ : ℝ) : Prop :=
  ρ = -2 * Real.cos θ + 2 * Real.sin θ

-- Question and Proof requirements.
theorem intersection_chord_length :
  (∀ x y t, parametric_line t = (x, y) → x - y + 1 = 0) ∧
  (∀ x y, polar_circle (Real.sqrt (x^2 + y^2)) (Real.atan2 y x) ↔ (x + 1)^2 + (y - Real.sqrt 3)^2 = 4) ∧
  (∃ A B : ℝ × ℝ, parametric_line A.1 = A ∧ parametric_line B.1 = B ∧
                  polar_circle (Real.sqrt (A.1^2 + A.2^2)) (Real.atan2 A.2 A.1) ∧
                  polar_circle (Real.sqrt (B.1^2 + B.2^2)) (Real.atan2 B.2 B.1) ∧ 
                  Real.dist A B = Real.sqrt 10) :=
by sorry -- Proof is omitted, only the statement is required.

end intersection_chord_length_l213_213751


namespace train_speed_l213_213066

noncomputable def speed_of_train_kmph (length_train length_tunnel : ℝ) (time_taken : ℝ) : ℝ :=
  let total_distance := length_train + length_tunnel
  let speed_m_per_s := total_distance / time_taken
  speed_m_per_s * 3.6

theorem train_speed (length_train length_tunnel : ℝ) (time_taken : ℝ)
  (h_train : length_train = 100) (h_tunnel : length_tunnel = 1400) (h_time : time_taken = 74.994) :
  speed_of_train_kmph length_train length_tunnel time_taken ≈ 72 :=
by
  sorry

end train_speed_l213_213066


namespace sum_of_geometric_areas_l213_213126

theorem sum_of_geometric_areas (n : ℕ) : 
  let r : ℕ → ℝ := λ n, (1:ℝ) / (Real.sqrt 2)^(n-1)
  let A : ℕ → ℝ := λ n, π * (r n)^2
  ∑' n, A n = 2 * π :=
by sorry

end sum_of_geometric_areas_l213_213126


namespace pyramid_transport_volume_l213_213553

-- Define the conditions of the problem
def pyramid_height : ℝ := 15
def pyramid_base_side_length : ℝ := 8
def box_length : ℝ := 10
def box_width : ℝ := 10
def box_height : ℝ := 15

-- Define the volume of the box
def box_volume : ℝ := box_length * box_width * box_height

-- State the theorem
theorem pyramid_transport_volume : box_volume = 1500 := by
  sorry

end pyramid_transport_volume_l213_213553


namespace simplify_radicals_l213_213443

open Real

theorem simplify_radicals : sqrt 72 + sqrt 32 = 10 * sqrt 2 := by
  sorry

end simplify_radicals_l213_213443


namespace linear_function_l213_213055

theorem linear_function (f : ℝ → ℝ)
  (h : ∀ x, f (f x) = 4 * x + 6) :
  (∀ x, f x = 2 * x + 2) ∨ (∀ x, f x = -2 * x - 6) :=
sorry

end linear_function_l213_213055


namespace jonathan_tax_per_hour_l213_213773

-- Given conditions
def wage : ℝ := 25          -- wage in dollars per hour
def tax_rate : ℝ := 0.024    -- tax rate in decimal

-- Prove statement
theorem jonathan_tax_per_hour :
  (wage * 100) * tax_rate = 60 :=
sorry

end jonathan_tax_per_hour_l213_213773


namespace max_sum_value_l213_213789

noncomputable def max_sum (x y : ℝ) (h : 3 * (x^2 + y^2) = x - y) : ℝ :=
  x + y

theorem max_sum_value :
  ∃ x y : ℝ, ∃ h : 3 * (x^2 + y^2) = x - y, max_sum x y h = 1/3 :=
sorry

end max_sum_value_l213_213789


namespace bananas_to_oranges_l213_213857

theorem bananas_to_oranges :
  (3 / 4) * 12 * b = 9 * o →
  ((3 / 5) * 15 * b) = 9 * o := 
by
  sorry

end bananas_to_oranges_l213_213857


namespace fraction_sum_divided_by_2_equals_decimal_l213_213572

theorem fraction_sum_divided_by_2_equals_decimal :
  let f1 := (3 : ℚ) / 20
  let f2 := (5 : ℚ) / 200
  let f3 := (7 : ℚ) / 2000
  let sum := f1 + f2 + f3
  let result := sum / 2
  result = 0.08925 := 
by
  sorry

end fraction_sum_divided_by_2_equals_decimal_l213_213572


namespace problem_l213_213300

theorem problem 
  (a : ℝ) 
  (h_a : ∀ x : ℝ, |x + 1| - |2 - x| ≤ a ∧ a ≤ |x + 1| + |2 - x|)
  {m n : ℝ} 
  (h_mn : m > n) 
  (h_n : n > 0)
  (h: a = 3) 
  : 2 * m + 1 / (m^2 - 2 * m * n + n^2) ≥ 2 * n + a :=
by
  sorry

end problem_l213_213300


namespace sum_a_1_to_15_l213_213787

def a (n : ℕ) : ℕ :=
  if n = 1 then 1
  else if n = 2 then 2
  else if n = 3 then 3
  else if n > 3 then
    let p := a (n - 1)
    let q := n - 3
    let r := a (n - 2) * a (n - 3)
    sorry -- Placeholder for the actual computation of the number of real roots
  else 0

theorem sum_a_1_to_15 : (Finset.range 15).sum (λ n, a (n + 1)) = 30 :=
sorry

end sum_a_1_to_15_l213_213787


namespace power_func_values_l213_213329

def power_func_even (m : ℝ) : Prop := 
  let f := (m^2 - 5*m + 7) * x^(-m-1)
  ∀ x : ℝ, f x = f (-x)

theorem power_func_values (m : ℝ) (h : power_func_even m) :
  (m = 3 ∧ f(1/2) = 16) ∧ ((∀ a : ℝ, f (2*a+1) = f a) → (a = -1 ∨ a = -1/3)) :=
by
  sorry

end power_func_values_l213_213329


namespace ceil_square_neg_fraction_l213_213174

theorem ceil_square_neg_fraction :
  let x := (-7) / 4
  let y := x^2
  let z := Real.ceil y
  z = 4 := 
by
  sorry

end ceil_square_neg_fraction_l213_213174


namespace max_value_expression_l213_213629

theorem max_value_expression (x y : ℝ) :
  ∃ N : ℝ, N = sqrt 26 ∧ (∀ x y : ℝ, (x + 3 * y + 4) / (sqrt (x^2 + y^2 + x + 1)) ≤ N) :=
begin
  use sqrt 26,
  split,
  { refl },
  { intros x y,
    sorry
  }

end max_value_expression_l213_213629


namespace martha_black_butterflies_l213_213811

theorem martha_black_butterflies
    (total_butterflies : ℕ)
    (total_blue_butterflies : ℕ)
    (total_yellow_butterflies : ℕ)
    (total_black_butterflies : ℕ)
    (h1 : total_butterflies = 19)
    (h2 : total_blue_butterflies = 6)
    (h3 : total_blue_butterflies = 2 * total_yellow_butterflies)
    (h4 : total_black_butterflies = total_butterflies - (total_blue_butterflies + total_yellow_butterflies))
    : total_black_butterflies = 10 :=
  sorry

end martha_black_butterflies_l213_213811


namespace train_speed_without_stoppages_l213_213548

theorem train_speed_without_stoppages 
  (distance_with_stoppages : ℝ)
  (avg_speed_with_stoppages : ℝ)
  (stoppage_time_per_hour : ℝ)
  (distance_without_stoppages : ℝ)
  (avg_speed_without_stoppages : ℝ) :
  avg_speed_with_stoppages = 200 → 
  stoppage_time_per_hour = 20 / 60 →
  distance_without_stoppages = distance_with_stoppages * avg_speed_without_stoppages →
  distance_with_stoppages = avg_speed_with_stoppages →
  avg_speed_without_stoppages == 300 := 
by
  intros
  sorry

end train_speed_without_stoppages_l213_213548


namespace max_gangsters_chicago_max_gangsters_l213_213477

/-- Define the basic conditions from the problem statement --/
structure GangsterProblem :=
  (gangs : ℕ)                                    -- Number of gangs
  (gangster_belongs : ℕ → ℕ → Prop)               -- Gangster i belongs to gang j
  (hostile : ℕ → ℕ → Prop)                        -- Gang i is hostile to gang j
  (no_two_same_gangs : ∀ (g1 g2 : ℕ), g1 ≠ g2 → 
    ∃ (j : ℕ), gangster_belongs g1 j ≠ gangster_belongs g2 j) -- No two gangsters belong to the same set of gangs

/-- Our goal is to compute the maximum number of distinct gangsters under given conditions --/
theorem max_gangsters (G : GangsterProblem) : ℕ :=
  sorry

def ChicagoGangsters : GangsterProblem :=
  { gangs := 36,
    gangster_belongs := λ i j, sorry,
    hostile := λ i j, sorry,
    no_two_same_gangs := λ _ _ _, sorry
  }

theorem chicago_max_gangsters : max_gangsters ChicagoGangsters = 531441 :=
  sorry

end max_gangsters_chicago_max_gangsters_l213_213477


namespace ceiling_of_square_frac_l213_213221

theorem ceiling_of_square_frac : 
  let x : ℚ := -7 / 4
  let y : ℚ := x^2
  in Int.ceil y = 4 := 
by 
  let x : ℚ := -7 / 4
  let y : ℚ := x^2
  exact sorry

end ceiling_of_square_frac_l213_213221


namespace incorrect_analogies_l213_213013

theorem incorrect_analogies 
  (cond1 : ∀ a b c : ℝ, (a * b) * c = a * (b * c))
  (cond2 : ∀ (n : ℕ) (a : ℕ → ℝ), let S := λ n, a n * (n + 1) / 2 in (S 4, S 8 - S 4, S 12 - S 8) ∈ {s : ℕ × ℕ × ℕ | s.2 = s.1 + s.0})
  (cond3 : ∀ (l1 l2 l3 : ℝ → ℝ), (l1 1 = l2 1) → (l1 2 = l3 2) → (l2 1 ≠ l3 2))
  (cond4 : ∀ (A B P : ℝ) (kPA kPB : ℝ), let f := λ A B P, B = P * A - A in kPA * kPB / kPA = kPB * A)
  : ¬ (cond1 ∧ cond4) ∧ ¬ (cond3) :=
sorry

end incorrect_analogies_l213_213013


namespace regular_polygon_radius_l213_213473

-- The statement only; no proof is required.
theorem regular_polygon_radius
  (n : ℕ) -- the number of sides of the regular polygon
  (side_length : ℝ) -- the side length of the polygon
  (exterior_sum : ℝ) -- the sum of exterior angles (always 360 degrees for any polygon)
  (interior_sum : ℝ) -- the sum of interior angles of the polygon
  (h1 : side_length = 2)
  (h2 : exterior_sum = 360)
  (h3 : interior_sum = 2 * exterior_sum)
  (h4 : (n - 2) * 180 = interior_sum) :
  (radius : ℝ) -- the radius of the regular polygon
  (radius = 2) :=
sorry

end regular_polygon_radius_l213_213473


namespace train_length_l213_213972

-- Definitions and conditions
variable (L : ℕ)
def condition1 (L : ℕ) : Prop := L + 100 = 15 * (L + 100) / 15
def condition2 (L : ℕ) : Prop := L + 250 = 20 * (L + 250) / 20

-- Theorem statement
theorem train_length (h1 : condition1 L) (h2 : condition2 L) : L = 350 := 
by 
  sorry

end train_length_l213_213972


namespace min_sum_distances_l213_213296

open Real

def hyperbola_eq (x y : ℝ) (m : ℝ) : Prop := x^2 - y^2 / m = 1

theorem min_sum_distances {P : ℝ × ℝ} {m : ℝ} 
  (h1 : P.1^2 - P.2^2 / m = 1) 
  (h2 : m = 3) 
  (h3 : eccentricity = 2) 
  (h4 : ∃l : ℝ → ℝ → ℝ, l = λ x y, 2 * x - 5 * y)
  (h5 : A = (-2,0)) :
  min_value_of_sum_of_distances P (λ x y, 2 * x - 5 * y) (-2, 0) = 50 / 13 := 
sorry

end min_sum_distances_l213_213296


namespace profit_share_difference_l213_213929

theorem profit_share_difference (capital_A capital_B capital_C profit_B : ℝ) 
  (hA : capital_A = 8000) 
  (hB : capital_B = 10000) 
  (hC : capital_C = 12000) 
  (hProfitB : profit_B = 2000) : 
  let ratio_A := capital_A / 2000;
      ratio_B := capital_B / 2000;
      ratio_C := capital_C / 2000;
      part_value := profit_B / ratio_B; 
      share_A := ratio_A * part_value; 
      share_C := ratio_C * part_value; 
      difference := share_C - share_A 
  in difference = 800 := 
by 
  -- Proof goes here
  sorry

end profit_share_difference_l213_213929


namespace total_slices_l213_213774

def cheesecake_slices : Nat := 6
def tiramisu_slices : Nat := 8
def chocolate_cake_slices : Nat := 12

def cheesecake_eaten : Nat := 3
def tiramisu_eaten : Nat := 4
def chocolate_cake_eaten : Nat := 6

def total_slices_eaten : Nat := cheesecake_eaten + tiramisu_eaten + chocolate_cake_eaten

theorem total_slices : total_slices_eaten = 13 := by
  have cheesecake_part : cheesecake_slices / 2 = cheesecake_eaten := by simp
  have tiramisu_part : tiramisu_slices / 2 = tiramisu_eaten := by simp
  have chocolate_cake_part : chocolate_cake_slices / 2 = chocolate_cake_eaten := by simp
  simp [total_slices_eaten, cheesecake_part, tiramisu_part, chocolate_cake_part]
  sorry

end total_slices_l213_213774


namespace find_eccentricity_l213_213688

-- Definitions from the conditions
def is_ellipse (a c : ℝ) (e : ℝ) : Prop :=
  (e * a = c) ∧ (a^2 - c^2 - a * c = 0)

def eccentricity (a c : ℝ) : ℝ :=
  c / a

-- Theorem statement
theorem find_eccentricity (a c : ℝ) (e : ℝ) 
  (h1 : is_ellipse a c e)
  (h2 : eccentricity a c = e) :
  e = (Real.sqrt 5 - 1) / 2 :=
 sorry

end find_eccentricity_l213_213688


namespace circumradius_of_sector_l213_213061

theorem circumradius_of_sector (theta : ℝ) (theta_half : theta / 2 = θ / 2) : 
  let radius := 9 in
  let R := 4.5 * real.sec (theta / 2) in
  true :=
by sorry

end circumradius_of_sector_l213_213061


namespace ceiling_of_square_frac_l213_213224

theorem ceiling_of_square_frac : 
  let x : ℚ := -7 / 4
  let y : ℚ := x^2
  in Int.ceil y = 4 := 
by 
  let x : ℚ := -7 / 4
  let y : ℚ := x^2
  exact sorry

end ceiling_of_square_frac_l213_213224


namespace circumcircle_tangent_to_S1_S2_l213_213777

/-- E is a point on the median CD of triangle ABC. 
    S1 is a circle passing through E and touching AB at A, meeting AC again at M. 
    S2 is a circle passing through E and touching AB at B, meeting BC again at N. 
    -/
theorem circumcircle_tangent_to_S1_S2 
  (A B C D E M N : Point) 
  (h1 : median_on_triangle A B C D) 
  (h2 : point_on_median E D) 
  (S1 S2 : Circle) 
  (h3 : circle_touches_line_at S1 A B) 
  (h4 : circle_passes_through S1 E) 
  (h5 : circle_meets_line_at S1 A C M) 
  (h6 : circle_touches_line_at S2 B A) 
  (h7 : circle_passes_through S2 E) 
  (h8 : circle_meets_line_at S2 B C N) : 
  is_tangent_to_circumcircle_of_triangle S1 (circumcircle C M N) ∧ 
  is_tangent_to_circumcircle_of_triangle S2 (circumcircle C M N) := 
sorry

end circumcircle_tangent_to_S1_S2_l213_213777


namespace number_square_roots_l213_213360

theorem number_square_roots (a x : ℤ) (h1 : x = (2 * a + 3) ^ 2) (h2 : x = (a - 18) ^ 2) : x = 169 :=
by 
  sorry

end number_square_roots_l213_213360


namespace percentage_paid_X_vs_Y_l213_213905

theorem percentage_paid_X_vs_Y (X Y : ℝ) (h1 : X + Y = 528) (h2 : Y = 240) :
  ((X / Y) * 100) = 120 :=
by
  sorry

end percentage_paid_X_vs_Y_l213_213905


namespace polynomial_evaluation_l213_213354

-- Given the value of y
def y : ℤ := 4

-- Our goal is to prove this mathematical statement
theorem polynomial_evaluation : (3 * (y ^ 2) + 4 * y + 2 = 66) := 
by 
    sorry

end polynomial_evaluation_l213_213354


namespace ceil_square_eq_four_l213_213151

theorem ceil_square_eq_four : (⌈(-7 / 4: ℚ)^2⌉ : ℤ) = 4 := by
  sorry

end ceil_square_eq_four_l213_213151


namespace greatest_integer_b_not_in_range_l213_213489

theorem greatest_integer_b_not_in_range :
  let f (x : ℝ) (b : ℝ) := x^2 + b*x + 20
  let g (x : ℝ) (b : ℝ) := x^2 + b*x + 24
  (¬ (∃ (x : ℝ), g x b = 0)) → (b = 9) :=
by
  sorry

end greatest_integer_b_not_in_range_l213_213489


namespace five_digit_integers_count_l213_213716
open BigOperators

noncomputable def permutations_with_repetition (n : ℕ) (reps : List ℕ) : ℕ :=
  n.factorial / ((reps.map (λ x => x.factorial)).prod)

theorem five_digit_integers_count :
  permutations_with_repetition 5 [2, 2] = 30 :=
by
  sorry

end five_digit_integers_count_l213_213716


namespace intersection_eq_l213_213720

-- Conditions
def M : set ℝ := {x | abs (x - 1) < 2}
def N : set ℝ := {x | x * (x - 3) < 0}

-- Theorem statement
theorem intersection_eq : M ∩ N = {x | 0 < x ∧ x < 3} := 
by sorry

end intersection_eq_l213_213720


namespace complex_conjugate_of_z_l213_213315

theorem complex_conjugate_of_z:
  let i := Complex.i in
  let z := (1 + Real.sqrt 3 * i) / (Real.sqrt 3 + i) in
  Complex.conj z = Real.sqrt 3 / 2 - 1 / 2 * i := 
by
  sorry

end complex_conjugate_of_z_l213_213315


namespace surface_area_and_volume_l213_213539

-- Define the given conditions
def right_angled_triangle_sides (a b c : ℝ) : Prop :=
  a = 3 ∧ b = 4 ∧ c = 5 ∧ a^2 + b^2 = c^2

def radius_of_base : ℝ := 12 / 5
def height1 : ℝ := 9 / 5
def height2 : ℝ := 16 / 5

-- Calculate surface area and volume based on given conditions
noncomputable def solid_surface_area : ℝ := π * (3 + 4) * radius_of_base
noncomputable def solid_volume : ℝ := 1 / 3 * π * radius_of_base^2 * 5

-- The proof statement
theorem surface_area_and_volume
  (a b c : ℝ)
  (h : right_angled_triangle_sides a b c)
  : solid_surface_area = 84 / 5 * π ∧ solid_volume = 48 / 5 * π :=
by
  sorry

end surface_area_and_volume_l213_213539


namespace sin_X_in_right_triangle_l213_213750

structure Triangle :=
(X Y Z : Point)
(right_angle_Y : right_angle Y)
(rel_4sinX_5cosX : 4 * sin X = 5 * cos X)
(tan_X_def : tan X = XY / YZ)

theorem sin_X_in_right_triangle {T : Triangle} : 
  sin T.X = 5 * sqrt 41 / 41 :=
sorry

end sin_X_in_right_triangle_l213_213750


namespace OM_perp_MN_iff_SNT_collinear_l213_213984

theorem OM_perp_MN_iff_SNT_collinear
  (O O₁ O₂ M N S T : Point)
  (r₁ r₂ r : ℝ)
  (h_intersect : intersect_circles O₁ O₂ M N)
  (h_unequal_radii : r₁ ≠ r₂)
  (h_tangent₁ : tangent_to_circle O₁ O S)
  (h_tangent₂ : tangent_to_circle O₂ O T) :
  (perp OM MN) ↔ collinear S N T :=
sorry

end OM_perp_MN_iff_SNT_collinear_l213_213984


namespace geometric_series_six_terms_l213_213921

theorem geometric_series_six_terms :
  (1/4 - 1/16 + 1/64 - 1/256 + 1/1024 - 1/4096 : ℚ) = 4095 / 20480 :=
by
  sorry

end geometric_series_six_terms_l213_213921


namespace find_m_for_local_minimum_l213_213704

noncomputable def f (x m : ℝ) := x * (x - m) ^ 2

theorem find_m_for_local_minimum :
  ∃ m, (∀ x, x ≠ 2 → (f x m > f 2 m) ∨ (f x m < f 2 m)) → m = 2 :=
begin
  sorry
end

end find_m_for_local_minimum_l213_213704


namespace periodic_product_and_quotient_l213_213405

theorem periodic_product_and_quotient 
  (f g : ℝ → ℝ) (T₁ T₂ : ℝ)
  (hf_periodic : ∀ x, f(x + T₁) = f(x))
  (hg_periodic : ∀ x, g(x + T₂) = g(x))
  (hf_pos : ∀ x, f(x) > 0)
  (hg_pos : ∀ x, g(x) > 0)
  : ∃ (T : ℝ), (∀ x, f(x) * g(x) = f(x + T) * g(x + T)) ∧ (∀ x, f(x) / g(x) = f(x + T) / g(x + T)) :=
sorry

end periodic_product_and_quotient_l213_213405


namespace sum_of_squares_of_odd_divisors_of_240_l213_213633

theorem sum_of_squares_of_odd_divisors_of_240 :
  let divisors := [1, 3, 5, 15]
  in (divisors.map (λ x => x * x)).sum = 260 :=
by
  let divisors := [1, 3, 5, 15]
  have h1 : divisors.map (λ x => x * x) = [1, 9, 25, 225] := by sorry
  have h2 : [1, 9, 25, 225].sum = 260 := by sorry
  exact h2

end sum_of_squares_of_odd_divisors_of_240_l213_213633


namespace maximum_cos_sum_l213_213140

theorem maximum_cos_sum
  (A B C : ℝ)
  (h₁ : A + B + C = π)
  (h₂ : 0 ≤ A)
  (h₃ : 0 ≤ B)
  (h₄ : 0 ≤ C) :
  cos A + cos B * cos C ≤ real.sqrt 2 :=
sorry

end maximum_cos_sum_l213_213140


namespace imaginary_part_of_conjugate_l213_213696

-- Defining the given complex number z
def z : ℂ := 2 + 5 * complex.I

-- The target proof statement
theorem imaginary_part_of_conjugate (z : ℂ) (h : z = 2 + 5 * complex.I) : complex.im (conj z) = -5 :=
by
  sorry

end imaginary_part_of_conjugate_l213_213696


namespace sqrt_neg_cube_real_l213_213640

theorem sqrt_neg_cube_real : ∀ x : ℝ, x ≤ -1 → ∃ y : ℝ, y = sqrt (-(x + 1)^3) :=
by
  sorry

end sqrt_neg_cube_real_l213_213640


namespace martha_black_butterflies_l213_213808

theorem martha_black_butterflies (total_butterflies blue_butterflies yellow_butterflies black_butterflies : ℕ) 
    (h1 : total_butterflies = 19)
    (h2 : blue_butterflies = 2 * yellow_butterflies)
    (h3 : blue_butterflies = 6) :
    black_butterflies = 10 :=
by
  -- Prove the theorem assuming the conditions are met
  sorry

end martha_black_butterflies_l213_213808


namespace probability_at_least_two_consecutive_heads_l213_213956

/--
Prove that the probability of getting at least two consecutive heads 
when a fair coin is tossed 4 times is 1/2.
--/
theorem probability_at_least_two_consecutive_heads : 
  (probability (λ s : Fin 16 → Fin 2, ∃ i, s i = 1 ∧ s (i + 1) = 1)) = 1 / 2 := 
sorry

end probability_at_least_two_consecutive_heads_l213_213956


namespace log_inequality_range_l213_213725

theorem log_inequality_range (a : ℝ) (h : Real.logBase a 2 < 1) : (a > 2 ∨ (0 < a ∧ a < 1)) :=
sorry

end log_inequality_range_l213_213725


namespace probability_at_least_two_same_l213_213829

theorem probability_at_least_two_same (n : ℕ) (H : n = 8) : 
  (∃ i j, (1 ≤ i ∧ i ≤ n) ∧ (1 ≤ j ∧ j ≤ n) ∧ i ≠ j ∧ ∀ (x : ℕ), x ∈ {i, j}) :=
by
  sorry

end probability_at_least_two_same_l213_213829


namespace cori_age_proof_l213_213134

theorem cori_age_proof:
  ∃ (x : ℕ), (3 + x = (1 / 3) * (19 + x)) ∧ x = 5 :=
by
  sorry

end cori_age_proof_l213_213134


namespace boxes_left_for_Sonny_l213_213854

def initial_boxes : ℕ := 45
def boxes_given_to_brother : ℕ := 12
def boxes_given_to_sister : ℕ := 9
def boxes_given_to_cousin : ℕ := 7

def total_given_away : ℕ := boxes_given_to_brother + boxes_given_to_sister + boxes_given_to_cousin

def remaining_boxes : ℕ := initial_boxes - total_given_away

theorem boxes_left_for_Sonny : remaining_boxes = 17 := by
  sorry

end boxes_left_for_Sonny_l213_213854


namespace ceiling_of_square_of_neg_7_over_4_is_4_l213_213169

theorem ceiling_of_square_of_neg_7_over_4_is_4 : 
  Real.ceil ((-7 / 4 : Real) ^ 2) = 4 := by
  sorry

end ceiling_of_square_of_neg_7_over_4_is_4_l213_213169


namespace max_abs_xk_value_l213_213279

noncomputable def max_abs_xk (n : ℕ) (x : Fin n → ℝ) (k : ℕ) : ℝ :=
  if h : 1 ≤ k ∧ k ≤ n then
    Real.sqrt (2 * k * (n - k + 1) / (n + 1))
  else
    0

theorem max_abs_xk_value (n : ℕ) (x : Fin n → ℝ) (k : ℕ) (h_n : 2 ≤ n)
  (h_sum : (Finset.univ.sum (λ i : Fin n, (x i) ^ 2) + 
    Finset.univ.sum (λ i : Fin (n - 1), (x i) * (x ⟨i + 1, Nat.lt_of_lt_pred (Fin.is_lt (⟨i, by simp⟩))⟩)) = 1)) :
  max_abs_xk n x k = √((2 * k * (n - k + 1)) / (n + 1)) :=
by
  -- Proof omitted
  sorry

end max_abs_xk_value_l213_213279


namespace trig_identity_evaluation_l213_213144

theorem trig_identity_evaluation :
  real.sin (-17 * real.pi / 6) + real.cos (-20 * real.pi / 3) + real.tan (-53 * real.pi / 6) = -1 + real.sqrt 3 :=
by sorry

end trig_identity_evaluation_l213_213144


namespace f_one_eq_zero_l213_213584

-- Define the function f on ℝ
variable (f : ℝ → ℝ)

-- Conditions for the problem
axiom odd_function : ∀ x : ℝ, f (-x) = -f (x)
axiom periodic_function : ∀ x : ℝ, f (x + 2) = f (x)

-- Goal: Prove that f(1) = 0
theorem f_one_eq_zero : f 1 = 0 :=
by
  sorry

end f_one_eq_zero_l213_213584


namespace visit_attractions_permutation_count_l213_213389

theorem visit_attractions_permutation_count :
  let n := 5 in
  n! = 120 :=
by
  let n := 5
  sorry

end visit_attractions_permutation_count_l213_213389


namespace peter_total_cost_l213_213104

-- Define the conditions as Lean definitions
def cost_of_one_pants : ℕ := 6
def cost_of_shirts (n : ℕ) : ℕ := 20 / n -- 2 shirts cost $20 implies each shirt costs $10 when n = 2

-- State the problem in Lean
theorem peter_total_cost :
  let P := cost_of_one_pants,
      S := cost_of_shirts 2 in
  (2 * P) + (5 * S) = 62 :=
by
  let P := cost_of_one_pants
  let S := cost_of_shirts 2
  sorry

end peter_total_cost_l213_213104


namespace calculation_of_cube_exponent_l213_213987

theorem calculation_of_cube_exponent (a : ℤ) : (-2 * a^3)^3 = -8 * a^9 := by
  sorry

end calculation_of_cube_exponent_l213_213987


namespace curves_intersect_exactly_three_points_l213_213495

theorem curves_intersect_exactly_three_points (a : ℝ) :
  (∃! (p : ℝ × ℝ), p.1 ^ 2 + p.2 ^ 2 = a ^ 2 ∧ p.2 = p.1 ^ 2 - a) ↔ a > (1 / 2) :=
by sorry

end curves_intersect_exactly_three_points_l213_213495


namespace linear_function_passing_through_points_has_given_formula_value_of_y_when_x_is_negative_four_l213_213691

variable (k b : ℝ)
def linear_function (x : ℝ) : ℝ := k * x + b

theorem linear_function_passing_through_points_has_given_formula :
  (linear_function k b 1 = 5) ∧ (linear_function k b (-1) = 1) →
  (∀ x, linear_function k b x = 2 * x + 3) := 
by {
  intros h,
  
  -- Extract individual conditions from the hypothesis
  cases h with h1 h2,
  
  -- Use the conditions to prove k = 2 and b = 3
  have eq1 : k + b = 5 := h1,
  have eq2 : -k + b = 1 := h2,
  linarith,

  sorry
}

theorem value_of_y_when_x_is_negative_four : 
  linear_function 2 3 (-4) = -5 := 
by {
  calculate,
  sorry
}

end linear_function_passing_through_points_has_given_formula_value_of_y_when_x_is_negative_four_l213_213691


namespace solve_for_q_l213_213449

theorem solve_for_q :
  ∀ (k l q : ℚ),
    (3 / 4 = k / 108) →
    (3 / 4 = (l + k) / 126) →
    (3 / 4 = (q - l) / 180) →
    q = 148.5 :=
by
  intros k l q hk hl hq
  sorry

end solve_for_q_l213_213449


namespace smallest_largest_prime_sum_between_50_and_100_l213_213582

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_between (a b : ℕ) : List ℕ :=
  List.filter is_prime (List.range' a (b - a + 1))

theorem smallest_largest_prime_sum_between_50_and_100 :
  let primes := primes_between 50 100
  ∃ smallest largest, List.minimum primes = some smallest ∧ List.maximum primes = some largest ∧ smallest + largest = 150 :=
by
  let primes := primes_between 50 100
  have primes_sorted : primes = [53, 59, 61, 67, 71, 73, 79, 83, 89, 97] := sorry
  have h_min : List.minimum primes = some 53 := sorry
  have h_max : List.maximum primes = some 97 := sorry
  exists 53
  exists 97
  rw [h_min, h_max]
  exact rfl
  sorry

end smallest_largest_prime_sum_between_50_and_100_l213_213582


namespace complement_intersection_l213_213332

def setM : Set ℝ := { x | 2 / x < 1 }
def setN : Set ℝ := { y | ∃ x, y = Real.sqrt (x - 1) }

theorem complement_intersection 
  (R : Set ℝ) : ((R \ setM) ∩ setN = { y | 0 ≤ y ∧ y ≤ 2 }) :=
  sorry

end complement_intersection_l213_213332


namespace profit_distribution_l213_213059

theorem profit_distribution (x : ℕ) (hx : 2 * x = 4000) :
  let A := 2 * x
  let B := 3 * x
  let C := 5 * x
  A + B + C = 20000 := by
  sorry

end profit_distribution_l213_213059


namespace sandy_marks_loss_l213_213439

theorem sandy_marks_loss (n m c p : ℕ) (h1 : n = 30) (h2 : m = 65) (h3 : c = 25) (h4 : p = 3) :
  ∃ x : ℕ, (c * p - m) / (n - c) = x ∧ x = 2 := by
  sorry

end sandy_marks_loss_l213_213439


namespace ellipse_a_value_l213_213668

-- Define the conditions
def ellipse_equation (x y a : ℝ) : Prop := (x^2)/(a^2) + (y^2)/8 = 1
def focal_length (c : ℝ) : Prop := 2 * c = 4

-- Define the proof statement
theorem ellipse_a_value (a : ℝ) (h1 : focal_length 2) (h2 : ∀ x y, ellipse_equation x y a) : a = 2 * real.sqrt 3 :=
sorry

end ellipse_a_value_l213_213668


namespace ceil_square_of_neg_seven_fourths_l213_213236

/-- Evaluate the ceiling of the square of -7/4 --/
theorem ceil_square_of_neg_seven_fourths : (Int.ceil ((-7/4 : ℚ)^2 : ℚ) = 4) :=
sorry

end ceil_square_of_neg_seven_fourths_l213_213236


namespace point_P_coordinates_parallel_to_chord_l213_213868

noncomputable def curve (x : ℝ) : ℝ := 4 * x - x^2

def point_A : ℝ × ℝ := (4, 0)
def point_B : ℝ × ℝ := (2, 4)

def slope (p1 p2 : ℝ × ℝ) : ℝ :=
  (p2.2 - p1.2) / (p2.1 - p1.1)

theorem point_P_coordinates_parallel_to_chord :
  ∃ (P : ℝ × ℝ), 
    curve P.1 = P.2 ∧ 
    slope point_A point_B = -2 ∧ 
    (curve' P.1 = -2) → P = (3, 3) :=
by
  sorry

end point_P_coordinates_parallel_to_chord_l213_213868


namespace right_triangle_third_side_length_l213_213362

-- Given conditions
def side1 : ℝ := real.sqrt 2
def side2 : ℝ := real.sqrt 3

-- Statement to prove
theorem right_triangle_third_side_length (a b : ℝ) (h1 : a = real.sqrt 2) (h2 : b = real.sqrt 3) :
    ∃ c : ℝ, (a ^ 2 + b ^ 2 = c ^ 2 ∨ a ^ 2 + c ^ 2 = b ^ 2 ∨ b ^ 2 + c ^ 2 = a ^ 2) ∧ 
               (c = real.sqrt 5 ∨ c = 1) :=
by
  sorry

end right_triangle_third_side_length_l213_213362


namespace time_to_cross_bridge_l213_213056

theorem time_to_cross_bridge 
  (speed_kmhr : ℕ) 
  (bridge_length_m : ℕ) 
  (h1 : speed_kmhr = 10)
  (h2 : bridge_length_m = 2500) :
  (bridge_length_m / (speed_kmhr * 1000 / 60) = 15) :=
by
  sorry

end time_to_cross_bridge_l213_213056


namespace isosceles_trapezoid_area_l213_213100

theorem isosceles_trapezoid_area 
  (a b x y : ℝ) 
  (h1 : a = 20) 
  (h2 : x = 11.11)
  (h3 : y = 2.22)
  (h4 : b = y + 20)
  (h5 : h = 0.6 * x)
  (h6 : arcsin 0.6 = θ) : 
  trapezoid_area a b h = 74.07 := 
  sorry

end isosceles_trapezoid_area_l213_213100


namespace find_third_number_l213_213627
open BigOperators

def LCM (a b : ℕ) : ℕ := a * b / (Nat.gcd a b)

def LCM_of_three (a b c : ℕ) : ℕ := LCM (LCM a b) c

theorem find_third_number (n : ℕ) (h₁ : LCM 15 25 = 75) (h₂ : LCM_of_three 15 25 n = 525) : n = 7 :=
by 
  sorry

end find_third_number_l213_213627


namespace ceil_square_eq_four_l213_213160

theorem ceil_square_eq_four : (⌈(-7 / 4: ℚ)^2⌉ : ℤ) = 4 := by
  sorry

end ceil_square_eq_four_l213_213160


namespace bruno_score_is_correct_l213_213847

theorem bruno_score_is_correct (richard_score : ℕ) (point_difference : ℕ) (bruno_score : ℕ) :
  richard_score = 62 → point_difference = 14 → bruno_score = richard_score - point_difference → bruno_score = 48 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  rw h3
  exact rfl

end bruno_score_is_correct_l213_213847


namespace isosceles_triangle_constant_sum_l213_213412

/-- Given isosceles triangle ABC with AB = AC, and P point on BC.
    Drop perpendicular PX from P to AB and perpendicular PY from P to AC.
    Prove that PX + PY is constant as P moves. -/
theorem isosceles_triangle_constant_sum
  (A B C P X Y : Point)
  (hABAC : dist A B = dist A C)
  (hP_on_BC : is_on_line P B C)
  (hPX_perp_AB : is_perpendicular P X A B)
  (hPY_perp_AC : is_perpendicular P Y A C) :
  (dist P X + dist P Y) = constant := by
  sorry

end isosceles_triangle_constant_sum_l213_213412


namespace intersection_points_unique_count_l213_213877

noncomputable def log_4 (x : ℝ) : ℝ := log x / log 4
noncomputable def log_inv_4 (x : ℝ) : ℝ := log x / log (1/4)

theorem intersection_points_unique_count :
  let y1 := λ x : ℝ, log_4 x
  let y2 := λ x : ℝ, 1 / log_4 x
  let y3 := λ x : ℝ, -log_4 x
  let y4 := λ x : ℝ, -1 / log_4 x
  ∃ unique_points, unique_points = [(1, 0), (2, log_4 2), (1/2, -log_4 2)] ∧
                    ∀ p ∈ unique_points, p.1 > 0 ∧ p.2 = y1 p.1 ∨ p.2 = y2 p.1 ∨ p.2 = y3 p.1 ∨ p.2 = y4 p.1 :=
by {
  sorry
}

end intersection_points_unique_count_l213_213877


namespace parabola_slopes_l213_213661

theorem parabola_slopes (k : ℝ) (A B : ℝ × ℝ) (C : ℝ × ℝ) 
    (hC : C = (0, -2)) (hA : A.1^2 = 2 * A.2) (hB : B.1^2 = 2 * B.2) 
    (hA_eq : A.2 = k * A.1 + 2) (hB_eq : B.2 = k * B.1 + 2) :
  ((C.2 - A.2) / (C.1 - A.1))^2 + ((C.2 - B.2) / (C.1 - B.1))^2 - 2 * k^2 = 8 := 
sorry

end parabola_slopes_l213_213661


namespace trapezoid_area_l213_213089

-- Given definitions and conditions for the problem
def isosceles_trapezoid_circumscribed_around_circle (a b h : ℝ) : Prop :=
  a > b ∧ h > 0 ∧ ∀ (x y : ℝ), x = h / 0.6 ∧ y = (2 * x - h) / 8 → a = b + 2 * √((h^2 - ((a - b) / 2)^2))

-- Definitions derived from conditions
def longer_base := 20
def base_angle := Real.arcsin 0.6

-- The proposition we need to prove (area == 74)
theorem trapezoid_area : 
  ∀ (a b h : ℝ), isosceles_trapezoid_circumscribed_around_circle a b h → base_angle = Real.arcsin 0.6 → 
  a = 20 → (1 / 2) * (b + 20) * h = 74 :=
sorry

end trapezoid_area_l213_213089


namespace covered_squares_l213_213954

theorem covered_squares (s r : ℝ) (h_r : r = 7 * s / 2) : 
  let checkerboard_size := 10
  let disc_center := (0 : ℝ, s / 2)
  let square_side_length := s
  let middle_square := ((checkerboard_size / 2 - 1) * s, (checkerboard_size / 2 - 1) * s)
  in number_of_covered_squares checkerboard_size square_side_length r disc_center = 24 :=
sorry

end covered_squares_l213_213954


namespace minimally_intersecting_remainder_l213_213136

noncomputable def count_minimally_intersecting_triples (s : Finset (Fin 7)) : Nat := sorry

theorem minimally_intersecting_remainder (s : Finset (Fin 7)) :
  let N := count_minimally_intersecting_triples s
  N % 1000 = 760 := sorry

end minimally_intersecting_remainder_l213_213136


namespace sum_of_primes_no_solution_congruence_l213_213603

theorem sum_of_primes_no_solution_congruence :
  2 + 5 = 7 :=
by
  sorry

end sum_of_primes_no_solution_congruence_l213_213603


namespace ceil_square_neg_seven_over_four_l213_213209

theorem ceil_square_neg_seven_over_four : 
  let x := - (7 / 4 : ℚ) in
  ⌈x^2⌉ = 4 :=
by
  let x := - (7 / 4 : ℚ)
  sorry

end ceil_square_neg_seven_over_four_l213_213209


namespace trapezoid_area_l213_213083

-- Define the problem statement
theorem trapezoid_area 
  (a b h: ℝ)
  (b₁ b₂: ℝ)
  (θ: ℝ) 
  (h₃: θ = Real.arcsin 0.6)
  (h₄: a = 20)
  (h₅: b = a - 2 * b₁ * Real.sin θ) 
  (h₆: h = b₁ * Real.cos θ) 
  (h₇: θ = Real.arcsin (3/5)) 
  (circum: isosceles_trapezoid_circumscribed a b₁ b₂) :
  ((1 / 2) * (a + b₂) * h = 2000 / 27) :=
by sorry

end trapezoid_area_l213_213083


namespace caleb_trip_duration_l213_213113

-- Define the times when the clock hands meet
def startTime := 7 * 60 + 38 -- 7:38 a.m. in minutes from midnight
def endTime := 13 * 60 + 5 -- 1:05 p.m. in minutes from midnight

def duration := endTime - startTime

theorem caleb_trip_duration :
  duration = 5 * 60 + 27 := by
sorry

end caleb_trip_duration_l213_213113


namespace find_lower_rate_l213_213065

-- Definitions
def total_investment : ℝ := 20000
def total_interest : ℝ := 1440
def higher_rate : ℝ := 0.09
def fraction_higher : ℝ := 0.55

-- The amount invested at the higher rate
def x := fraction_higher * total_investment
-- The amount invested at the lower rate
def y := total_investment - x

-- The interest contributions
def interest_higher := x * higher_rate
def interest_lower (r : ℝ) := y * r

-- The equation we need to solve to find the lower interest rate
theorem find_lower_rate (r : ℝ) : interest_higher + interest_lower r = total_interest → r = 0.05 :=
by
  sorry

end find_lower_rate_l213_213065


namespace constant_term_in_binomial_expansion_l213_213035

theorem constant_term_in_binomial_expansion :
  (∃ (t : ℚ), t = -55 / 2 ∧ 
  (∀ x : ℚ, ((x / 2 - 1 / real.cbrt (x))^12) 
    = t + (polynomial.of_fun (λ k, if k ≠ 0 then polynomial.coeff ((x / 2 - 1 / real.cbrt (x))^12) k else 0)))) :=
sorry

end constant_term_in_binomial_expansion_l213_213035


namespace students_received_B_l213_213741

/-!
# Problem Statement

Given:
1. In Mr. Johnson's class, 18 out of 30 students received a B.
2. Ms. Smith has 45 students in total, and the ratio of students receiving a B is the same as in Mr. Johnson's class.
Prove:
27 students in Ms. Smith's class received a B.
-/

theorem students_received_B (s1 s2 b1 : ℕ) (r1 : ℚ) (r2 : ℕ) (h₁ : s1 = 30) (h₂ : b1 = 18) (h₃ : s2 = 45) (h₄ : r1 = 3/5) 
(H : (b1 : ℚ) / s1 = r1) : r2 = 27 :=
by
  -- Conditions provided
  -- h₁ : s1 = 30
  -- h₂ : b1 = 18
  -- h₃ : s2 = 45
  -- h₄ : r1 = 3/5
  -- H : (b1 : ℚ) / s1 = r1
  sorry

end students_received_B_l213_213741


namespace asymptotes_of_hyperbola_l213_213282

-- Define the hyperbola structure and its properties
structure Hyperbola :=
(center : ℝ × ℝ)
(foci_axis : String)
(eccentricity : ℝ)

-- Given conditions
def hyperbola_C : Hyperbola := {
  center := (0, 0),
  foci_axis := "coordinate",
  eccentricity := Real.sqrt 2
}

-- Definition of the theorem statement
theorem asymptotes_of_hyperbola (C : Hyperbola) (h_center : C.center = (0, 0))
  (h_foci_axis : C.foci_axis = "coordinate") (h_eccentricity : C.eccentricity = Real.sqrt 2) :
  ∀ x y : ℝ, (y = x) ∨ (y = -x) :=
by
  -- Sorry is used to indicate that proof steps are skipped.
  sorry

end asymptotes_of_hyperbola_l213_213282


namespace range_of_x_l213_213289

variable {f : ℝ → ℝ}
variable {x : ℝ}

-- The conditions
axiom odd_f : ∀ x, f (-x) = -f x
axiom derivative_f : ∀ x ∈ Ioo (-1 : ℝ) 1, deriv f x = 1 - cos x

-- The proof statement
theorem range_of_x :
  (∀ x ∈ Ioo (-1 : ℝ) 1, deriv f x = 1 - cos x) →
  (∀ x, f (-x) = -f x) →
  (∀ x ∈ Ioo (-1 : ℝ) 1, f (1 - x^2) + f (1 - x) < 0 → 1 < x ∧ x < sqrt 2) :=
by
  intro h1 h2 x hx
  sorry

end range_of_x_l213_213289


namespace quadratic_roots_diff_l213_213467

theorem quadratic_roots_diff (k : ℝ) :
  (∃ r1 r2 : ℝ, r1 ≠ r2 ∧ r1 * r2 = 8 ∧ r1 + r2 = -k ∧ |r1 - r2| = sqrt 89) → 
  k = 11 :=
by
  sorry

end quadratic_roots_diff_l213_213467


namespace intersect_condition_l213_213292

variable (A B P : Point)
variable (a b k : ℝ)

def line_l (k : ℝ) : Set Point := {p | p.y = k * p.x - 1}
def point_A : Point := ⟨2, 0⟩
def point_B : Point := ⟨-2, 4⟩
def point_P : Point := ⟨0, -1⟩

def segment_AB : Set Point :=
  {p | ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ p = (1 - t) • point_A + t • point_B}

theorem intersect_condition (k : ℝ) :
  (∃ p ∈ segment_AB, p ∈ line_l k) → k ≤ -5/2 ∨ k ≥ 1/2 :=
sorry

end intersect_condition_l213_213292


namespace find_x_through_point_and_cos_l213_213314

theorem find_x_through_point_and_cos (x : ℝ) :
  (let α := real.angle (real.cosα)
  (P := (-x, -6 : ℝ × ℝ))
  (hyp_angle : cos α = 4 / 5)
  (hyp_distance : dist (0,0) P = real.sqrt (x^2 + 36)))
  (cos α = 4 / 5)
  -x / (real.sqrt (x^2 + 36)) = 4 / 5
  → x = -8 :=
begin
  sorry
end

end find_x_through_point_and_cos_l213_213314


namespace ceiling_of_square_of_neg_7_over_4_is_4_l213_213163

theorem ceiling_of_square_of_neg_7_over_4_is_4 : 
  Real.ceil ((-7 / 4 : Real) ^ 2) = 4 := by
  sorry

end ceiling_of_square_of_neg_7_over_4_is_4_l213_213163


namespace distinct_four_digit_arithmetic_sequence_digits_l213_213717

theorem distinct_four_digit_arithmetic_sequence_digits :
  ∃ (n : ℕ), ( ∀ (d : ℕ),
    1000 ≤ d ∧ d < 10000 ∧
    (∀ k, (d / 10^k) % 10 < 10 ∧ (d / 10^k) % 10 >= 0) ∧
    (∃ a b c, (d / 1000) % 10 = a ∧ (d / 100) % 10 = b ∧ (d / 10) % 10 = c ∧ 
              b = (a + c) / 2 ∧ b - a = c - b) ∧ 
    (∀ i j, i ≠ j → (d / 10^i) % 10 ≠ (d / 10^j) % 10)) → 
  n = 504 :=
begin
  sorry
end

end distinct_four_digit_arithmetic_sequence_digits_l213_213717


namespace average_paychecks_l213_213396

def first_paychecks : Nat := 6
def remaining_paychecks : Nat := 20
def total_paychecks : Nat := 26
def amount_first : Nat := 750
def amount_remaining : Nat := 770

theorem average_paychecks : 
  (first_paychecks * amount_first + remaining_paychecks * amount_remaining) / total_paychecks = 765 :=
by
  sorry

end average_paychecks_l213_213396


namespace trapezoid_area_l213_213086

-- Given definitions and conditions for the problem
def isosceles_trapezoid_circumscribed_around_circle (a b h : ℝ) : Prop :=
  a > b ∧ h > 0 ∧ ∀ (x y : ℝ), x = h / 0.6 ∧ y = (2 * x - h) / 8 → a = b + 2 * √((h^2 - ((a - b) / 2)^2))

-- Definitions derived from conditions
def longer_base := 20
def base_angle := Real.arcsin 0.6

-- The proposition we need to prove (area == 74)
theorem trapezoid_area : 
  ∀ (a b h : ℝ), isosceles_trapezoid_circumscribed_around_circle a b h → base_angle = Real.arcsin 0.6 → 
  a = 20 → (1 / 2) * (b + 20) * h = 74 :=
sorry

end trapezoid_area_l213_213086


namespace problem_1_area_triangle_problem_2_range_a_l213_213321

theorem problem_1_area_triangle (e : ℝ) (f : ℝ → ℝ) (f' : ℝ → ℝ) (a : ℝ) :
  (∀ x, f(x) = Real.exp x + x - 1) →
  (f(1) = Real.exp 1) →
  (∀ x, f'(x) = Real.exp x + 1) →
  (f'(1) = Real.exp 1 + 1) →
  a = 1 →
  let tangent_line := λ x, (Real.exp 1 + 1) * x - 1,
      point_A := (1 / (Real.exp 1 + 1), 0),
      point_B := (0, -1),
      area := (1 / 2) * (1 / (Real.exp 1 + 1)) * 1 in
  area = 1 / (2 * (Real.exp 1 + 1)) := by sorry

theorem problem_2_range_a (e : ℝ) (f : ℝ → ℝ) (a : ℝ) :
  (∀ x, f x = Real.exp x + a * x - 1) →
  (∀ x ∈ Ioo 0 1, f x ≥ x^2) →
  a ≥ 2 - Real.exp 1 := by sorry

end problem_1_area_triangle_problem_2_range_a_l213_213321


namespace current_speed_correct_l213_213531

noncomputable def speed_of_current : ℝ :=
  let rowing_speed_still_water := 10 -- speed of rowing in still water in kmph
  let distance_meters := 60 -- distance covered in meters
  let time_seconds := 17.998560115190788 -- time taken in seconds
  let distance_km := distance_meters / 1000 -- converting distance to kilometers
  let time_hours := time_seconds / 3600 -- converting time to hours
  let downstream_speed := distance_km / time_hours -- calculating downstream speed
  downstream_speed - rowing_speed_still_water -- calculating and returning the speed of the current

theorem current_speed_correct : speed_of_current = 2.00048 := by
  -- The proof is not provided in this statement as per the requirements.
  sorry

end current_speed_correct_l213_213531


namespace smallest_b_value_l213_213784

noncomputable def smallest_b (a b c : ℕ) (r : ℝ) :=
  ∃ (a b c : ℕ) (r : ℝ),
    a > 0 ∧ b = a * r ∧ c = a * r^2 ∧ a * b * c = 216

theorem smallest_b_value :
  ∀ (a b c : ℕ) (r : ℝ),
    smallest_b a b c r → b = 6 :=
by
  intros,
  sorry

end smallest_b_value_l213_213784


namespace simplified_expression_eq_seventeen_thirds_l213_213441

theorem simplified_expression_eq_seventeen_thirds :
  (√418 / √308 + √294 / √196) = (17 / 3) :=
by
  sorry

end simplified_expression_eq_seventeen_thirds_l213_213441


namespace solve_volume_of_water_l213_213953

noncomputable def volume_of_water (r : ℝ) (V : ℝ) : Prop :=
  let volume_ball := (4 / 3 * Real.pi * r^3)
  let base_area := Real.pi * r^2
  let height_water_after_ball := (volume_ball / base_area)
  V = height_water_after_ball * base_area - volume_ball

theorem solve_volume_of_water :
  volume_of_water 2 (16 * Real.pi / 3) :=
by
  have r : ℝ := 2
  have V : ℝ := 16 * Real.pi / 3
  change volume_of_water r V
  sorry

end solve_volume_of_water_l213_213953


namespace set_area_spherical_distances_l213_213127

def tetrahedron (X Y Z T : Type) : Prop :=
  ∃ (XY TZ : ℝ) (angle_YZT angle_XTZ angle_YTZ angle_XZT : ℝ),
  XY = 6 ∧
  TZ = 8 ∧
  angle_YZT = 25 ∧ 
  angle_XTZ = 25 ∧ 
  angle_YTZ = 65 ∧ 
  angle_XZT = 65

def sphere_circumscribed (X Y Z T : Type) (R : ℝ) : Prop :=
  ∀ (M : Type),
  (d_M_X + d_M_Y + d_M_Z + d_M_T) ≥ 8 * π →
  surface_area M = 32 * π

theorem set_area_spherical_distances (X Y Z T : Type):
  tetrahedron X Y Z T → 
  sphere_circumscribed X Y Z T 4 → 
  ∀ (M : Type), 
  (d_M_X + d_M_Y + d_M_Z + d_M_T) ≥ 8 * π → 
  surface_area M = 32 * π :=  sorry

end set_area_spherical_distances_l213_213127


namespace max_area_triangle_l213_213475

open Real
open Geometry

theorem max_area_triangle (A B C : Point ℝ ℝ) (hAB : 0 ≤ dist A B ∧ dist A B ≤ 1)
  (hBC : 1 ≤ dist B C ∧ dist B C ≤ 2) (hCA : 2 ≤ dist C A ∧ dist C A ≤ 3) : ∃ K ≤ 1, area A B C = K :=
by
  sorry

end max_area_triangle_l213_213475


namespace problem_translation_l213_213291

-- Definitions based on given conditions
def parametric_line (t : ℝ) : ℝ × ℝ :=
  (sqrt 2 * t * sin (π / 6), t * cos (7 * π / 4) - 6 * sqrt 2)

def polar_circle (θ : ℝ) : ℝ :=
  4 * cos (θ + π / 4)

-- The proof goal in Lean 4 statement

theorem problem_translation :
  (∀ t, parametric_line t.2 = parametric_line t.1 + -6 * sqrt 2) ∧
  (x^2 + y^2 - 2 * sqrt 2 * x + 2 * sqrt 2 * y = 0) ∧
  (min_distance_center_line = 2) :=
by
  sorry

end problem_translation_l213_213291


namespace find_angleB_find_maxArea_l213_213302

noncomputable def angleB (a b c : ℝ) (C : ℝ) :=
  (a + c) / b = Real.cos C + Real.sqrt 3 * Real.sin C

noncomputable def maxArea (a b c : ℝ) (B : ℝ) :=
  b = 2

theorem find_angleB (a b c : ℝ) (C : ℝ) (h : angleB a b c C) : 
  ∃ B, B = 60 ∧ angleB a b c C :=
sorry

theorem find_maxArea (a b c : ℝ) (B : ℝ) (hB : B = 60) (hb : maxArea a b c B) :
  ∃ S, S = Real.sqrt 3 :=
sorry

end find_angleB_find_maxArea_l213_213302


namespace strictly_increasing_interval_l213_213320

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (x + Real.pi / 3)

theorem strictly_increasing_interval :
  (∀ k : ℤ, ∀ x : ℝ, 
    (2 * k * Real.pi - 5 * Real.pi / 6 ≤ x ∧ x ≤ 2 * k * Real.pi + Real.pi / 6) 
    → (f x) < (f (x + 1))) :=
by 
  sorry

end strictly_increasing_interval_l213_213320


namespace ceil_square_eq_four_l213_213154

theorem ceil_square_eq_four : (⌈(-7 / 4: ℚ)^2⌉ : ℤ) = 4 := by
  sorry

end ceil_square_eq_four_l213_213154


namespace ceil_square_eq_four_l213_213158

theorem ceil_square_eq_four : (⌈(-7 / 4: ℚ)^2⌉ : ℤ) = 4 := by
  sorry

end ceil_square_eq_four_l213_213158


namespace probability_of_blue_ball_l213_213372

theorem probability_of_blue_ball 
(P_red P_yellow P_blue : ℝ) 
(h_red : P_red = 0.48)
(h_yellow : P_yellow = 0.35) 
(h_prob : P_red + P_yellow + P_blue = 1) 
: P_blue = 0.17 := 
sorry

end probability_of_blue_ball_l213_213372


namespace max_members_in_band_l213_213476

theorem max_members_in_band (m : ℤ) (h1 : 30 * m % 31 = 6) (h2 : 30 * m < 1200) : 30 * m = 360 :=
by {
  sorry -- Proof steps are not required according to the procedure
}

end max_members_in_band_l213_213476


namespace chairs_made_after_tables_l213_213272

def pieces_of_wood : Nat := 672
def wood_per_table : Nat := 12
def wood_per_chair : Nat := 8
def number_of_tables : Nat := 24

theorem chairs_made_after_tables (pieces_of_wood wood_per_table wood_per_chair number_of_tables : Nat) :
  wood_per_table * number_of_tables <= pieces_of_wood ->
  (pieces_of_wood - wood_per_table * number_of_tables) / wood_per_chair = 48 :=
by
  sorry

end chairs_made_after_tables_l213_213272


namespace profession_odd_one_out_l213_213565

/-- 
Three professions are provided: 
1. Dentist
2. Elementary school teacher
3. Programmer 

Under modern Russian pension legislation, some professions have special retirement benefits. 
For the given professions, the elementary school teacher and dentist typically have special considerations, whereas the programmer does not.

Prove that the programmer is the odd one out under these conditions.
-/
theorem profession_odd_one_out
  (dentist_special: Bool)
  (teacher_special: Bool)
  (programmer_special: Bool)
  (h1: dentist_special = true)
  (h2: teacher_special = true)
  (h3: programmer_special = false) :
  ∃ profession, profession = "Programmer" :=
by
  use "Programmer"
  sorry

end profession_odd_one_out_l213_213565


namespace area_ratio_l213_213760

noncomputable def area (a b c : ℝ) : ℝ := sqrt ((a + b + c) / 2 * ((a + b + c) / 2 - a) * ((a + b + c) / 2 - b) * ((a + b + c) / 2 - c))

theorem area_ratio
  (PQ QR RP : ℝ)
  (hPQ : PQ = 14)
  (hQR : QR = 16)
  (hRP : RP = 18)
  (x y z : ℝ)
  (hx : 0 < x)
  (hy : 0 < y)
  (hz : 0 < z)
  (h_sum : x + y + z = 3 / 4)
  (h_sumsq : x^2 + y^2 + z^2 = 3 / 7) :
  let A_PQR := area PQ QR RP in
  let A_STU := A_PQR * (1 - (x * (1 - z) + y * (1 - x) + z * (1 - y))) in
  A_STU / A_PQR = 43 / 112 :=
sorry

end area_ratio_l213_213760


namespace cross_country_hours_l213_213483

-- Definitions based on the conditions from part a)
def total_hours_required : ℕ := 1500
def hours_day_flying : ℕ := 50
def hours_night_flying : ℕ := 9
def goal_months : ℕ := 6
def hours_per_month : ℕ := 220

-- Problem statement: prove she has already completed 1261 hours of cross-country flying
theorem cross_country_hours : 
  (goal_months * hours_per_month) - (hours_day_flying + hours_night_flying) = 1261 := 
by
  -- Proof omitted (using the solution steps)
  sorry

end cross_country_hours_l213_213483


namespace n_squared_d_values_l213_213474

theorem n_squared_d_values (n : ℕ) (d : ℤ)
    (a : ℕ → ℤ)
    (h1 : ∑ i in finset.range n, |a i| = 250)
    (h2 : ∑ i in finset.range n, |(a i) + 1| = 250)
    (h3 : ∑ i in finset.range n, |(a i) + 2| = 250) :
    n^2 * d = 1000 ∨ n^2 * d = -1000 := 
begin
    sorry
end

end n_squared_d_values_l213_213474


namespace no_such_permutation_l213_213413

open Nat
open List

noncomputable def is_bijection (f : ℕ → ℕ) : Prop :=
  (Function.Injective f) ∧ (Function.Surjective f)

theorem no_such_permutation (f : ℕ → ℕ) (H : is_bijection f) :
  ¬ ∃ n : ℕ, 
    n > 0 ∧ Perm (List.ofFn (λ i : Fin n, f i.succ)) (List.range n).map (λ i, i + 1) :=
sorry

end no_such_permutation_l213_213413


namespace ceil_square_neg_fraction_l213_213180

theorem ceil_square_neg_fraction :
  let x := (-7) / 4
  let y := x^2
  let z := Real.ceil y
  z = 4 := 
by
  sorry

end ceil_square_neg_fraction_l213_213180


namespace negation_of_proposition_l213_213498

variable (a b : ℝ) -- Define variables a and b in the real numbers

def original_proposition := (a ≥ 0) ∧ (b ≥ 0) → (a * b ≥ 0)
def negated_proposition := (a < 0) ∨ (b < 0) → (a * b < 0)

theorem negation_of_proposition : ¬original_proposition ↔ negated_proposition :=
by
  sorry

end negation_of_proposition_l213_213498


namespace find_smaller_part_l213_213934

noncomputable def smaller_part (x y : ℕ) : ℕ :=
  if x ≤ y then x else y

theorem find_smaller_part (x y : ℕ) (h1 : x + y = 24) (h2 : 7 * x + 5 * y = 146) : smaller_part x y = 11 :=
  sorry

end find_smaller_part_l213_213934


namespace find_lambda_l213_213648

variables (a b : ℝ^3) (λ : ℝ)
  (ha : ∥a∥ = 2)
  (hb : ∥b∥ = √2)
  (θ : ℝ) (hθ : θ = 45)
  (h_perp : (λ • b - a) ⬝ a = 0)

#check λ

theorem find_lambda
  (ha : ∥a∥ = 2)
  (hb : ∥b∥ = √2)
  (hθ : θ = real.pi / 4)
  (h_perp : (λ • b - a) ⬝ a = 0) :
  λ = 2 := sorry

end find_lambda_l213_213648


namespace cos_squared_identity_l213_213351

theorem cos_squared_identity (α : ℝ) (h : Real.sin (π / 6 - α) = 1 / 3) : 
  2 * Real.cos (π / 6 + α / 2) ^ 2 + 1 = 7 / 3 := 
by
    sorry

end cos_squared_identity_l213_213351


namespace smallest_n_with_perfect_square_pair_l213_213250

theorem smallest_n_with_perfect_square_pair : ∃ (n : ℕ), ∀ (a b : ℕ), a ≠ b → a ∈ (Finset.range (70 + n + 1)).filter (λ x, 70 ≤ x) → b ∈ (Finset.range (70 + n + 1)).filter (λ x, 70 ≤ x) → (∃ k : ℕ, a * b = k^2) ∧ n = 28 :=
by
  sorry

end smallest_n_with_perfect_square_pair_l213_213250


namespace trapezoid_area_l213_213090

-- Given definitions and conditions for the problem
def isosceles_trapezoid_circumscribed_around_circle (a b h : ℝ) : Prop :=
  a > b ∧ h > 0 ∧ ∀ (x y : ℝ), x = h / 0.6 ∧ y = (2 * x - h) / 8 → a = b + 2 * √((h^2 - ((a - b) / 2)^2))

-- Definitions derived from conditions
def longer_base := 20
def base_angle := Real.arcsin 0.6

-- The proposition we need to prove (area == 74)
theorem trapezoid_area : 
  ∀ (a b h : ℝ), isosceles_trapezoid_circumscribed_around_circle a b h → base_angle = Real.arcsin 0.6 → 
  a = 20 → (1 / 2) * (b + 20) * h = 74 :=
sorry

end trapezoid_area_l213_213090


namespace clock_angle_at_730_degrees_l213_213142

theorem clock_angle_at_730_degrees :
  let degrees_per_hour := 360 / 12
  let minute_hand_position := 180
  let hour_hand_position := (7.5 * degrees_per_hour : ℝ)
  degree_between_hands (minute_hand_position : ℝ) (hour_hand_position : ℝ) = 45 :=
by
  let degrees_per_hour := 30
  let minute_hand_position := 180
  let hour_hand_position := 225
  sorry

end clock_angle_at_730_degrees_l213_213142


namespace min_value_condition_solve_inequality_l213_213701

open Real

-- Define the function f(x) = |x - a| + |x + 2|
def f (x a : ℝ) : ℝ := abs (x - a) + abs (x + 2)

-- Part I: Proving the values of a for f(x) having minimum value of 2
theorem min_value_condition (a : ℝ) : 
  (∀ x : ℝ, f x a ≥ 2) → (∃ x : ℝ, f x a = 2) → (a = 0 ∨ a = -4) :=
by
  sorry

-- Part II: Solving inequality f(x) ≤ 6 when a = 2
theorem solve_inequality : 
  ∀ x : ℝ, f x 2 ≤ 6 ↔ (x ≥ -3 ∧ x ≤ 3) :=
by
  sorry

end min_value_condition_solve_inequality_l213_213701


namespace solve_for_b_l213_213728

theorem solve_for_b (b : ℚ) (h : b + b / 4 = 5 / 2) : b = 2 := 
sorry

end solve_for_b_l213_213728


namespace max_points_N_l213_213743

def number_of_teams : ℕ := 15
def points_win : ℕ := 3
def points_draw : ℕ := 1
def points_loss : ℕ := 0
def successful_teams : ℕ := 6

theorem max_points_N : ℕ :=
  ∃ N : ℕ, (∀ team ∈ list.range successful_teams, team.points ≥ N) ∧
  (N = 34) :=
begin
  sorry
end

end max_points_N_l213_213743


namespace cos_angle_BAD_eq_sqrt_0_45_l213_213386

-- Define the sides of the triangle ABC
def AB : ℝ := 5
def AC : ℝ := 7
def BC : ℝ := 9

-- Define the bisector property
def D_lies_on_BC_and_bisects (D : Point) (AD : Line) : Prop :=
  -- Here we would define that D lies on BC and AD bisects ∠BAC 
  -- Skipping precise geometric formalization 
  sorry

-- Traditional statement: Prove the condition.
theorem cos_angle_BAD_eq_sqrt_0_45
  (D : Point) (AD : Line)
  (h : D_lies_on_BC_and_bisects D AD)
  (cos_angle_BAD : ℝ) :
  cos_angle_BAD = real.sqrt 0.45 :=
by sorry

end cos_angle_BAD_eq_sqrt_0_45_l213_213386


namespace weight_conversion_l213_213936

theorem weight_conversion (a b : ℝ) (conversion_rate : ℝ) : a = 3600 → b = 600 → conversion_rate = 1000 → (a - b) / conversion_rate = 3 := 
by
  intros h₁ h₂ h₃
  rw [h₁, h₂, h₃]
  sorry

end weight_conversion_l213_213936


namespace find_z_l213_213308

open Complex

theorem find_z (z : ℂ) (h : (1 + 2 * z) / (1 - z) = Complex.I) : 
  z = -1 / 5 + 3 / 5 * Complex.I := 
sorry

end find_z_l213_213308


namespace ceil_square_of_neg_fraction_l213_213189

theorem ceil_square_of_neg_fraction : 
  (Int.ceil ((-7 / 4 : ℚ)^2 : ℚ)).toNat = 4 := by
  sorry

end ceil_square_of_neg_fraction_l213_213189


namespace count_distinct_squares_dodecagon_l213_213782

-- Definitions based on the conditions provided in the problem
def vertices_dodecagon : set (euclidean_space ℝ (fin 2)) :=
  {A_1, A_2, A_3, A_4, A_5, A_6, A_7, A_8, A_9, A_10, A_11, A_12}

def is_square_formed (s : set (euclidean_space ℝ (fin 2))) : Prop :=
  ∃ a b c d : euclidean_space ℝ (fin 2),
    a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ a ∧
    distance a b = distance b c ∧ distance b c = distance c d ∧ distance c d = distance d a ∧
    distance a c = distance b d
  
-- Main theorem statement
theorem count_distinct_squares_dodecagon :
  {s : set (euclidean_space ℝ (fin 2)) // is_square_formed s ∧ s ⊆ vertices_dodecagon ∧ 2 ≤ s.card}.to_finset.card = 6
  :=
by
  sorry

end count_distinct_squares_dodecagon_l213_213782


namespace ceil_square_of_neg_fraction_l213_213192

theorem ceil_square_of_neg_fraction : 
  (Int.ceil ((-7 / 4 : ℚ)^2 : ℚ)).toNat = 4 := by
  sorry

end ceil_square_of_neg_fraction_l213_213192


namespace ceiling_of_square_of_neg_7_over_4_is_4_l213_213166

theorem ceiling_of_square_of_neg_7_over_4_is_4 : 
  Real.ceil ((-7 / 4 : Real) ^ 2) = 4 := by
  sorry

end ceiling_of_square_of_neg_7_over_4_is_4_l213_213166


namespace ceil_square_eq_four_l213_213161

theorem ceil_square_eq_four : (⌈(-7 / 4: ℚ)^2⌉ : ℤ) = 4 := by
  sorry

end ceil_square_eq_four_l213_213161


namespace triangle_equivalence_l213_213739

theorem triangle_equivalence
  (A B C : ℝ)
  (a b c : ℝ)
  (ha : 0 < a)
  (hb : a < b)
  (hc : b < c)
  (h_sinA : sin A = (sqrt 3 * a) / (2 * b))
  (h_a : a = 2)
  (h_b : b = sqrt 7) :
  (B = π / 3) ∧ (c = 3) ∧ (1 / 2 * a * c * sin B = 3 * sqrt 3 / 2) := 
by
  sorry

end triangle_equivalence_l213_213739


namespace probability_of_at_least_eight_sixes_in_ten_rolls_l213_213957

theorem probability_of_at_least_eight_sixes_in_ten_rolls :
  let prob := (if m=8 then 45 * (1/6)^8 * (5/6)^2 else
               if m=9 then 10 * (1/6)^9 * (5/6) else
               if m=10 then (1/6)^10 else 0) in 
  ∑ m in {8, 9, 10}, prob = 1136 / 60466176 :=
by
  sorry

end probability_of_at_least_eight_sixes_in_ten_rolls_l213_213957


namespace true_composite_prop_l213_213698

noncomputable def p1 : Prop :=
  ∃ x > 0, log (x^2 + 1/4) ≤ log x

noncomputable def p2 : Prop :=
  ∀ x, sin x ≠ 0 → sin x + 1/sin x ≥ 2

noncomputable def p3 : Prop :=
  ∀ x y, x + y = 0 ↔ x / y = -1

theorem true_composite_prop : p1 ∨ ¬p2 :=
by {
  -- Conditions outlined in the problem
  -- Feel free to add any necessary axioms, definitions, or lemmas here
  sorry
}

end true_composite_prop_l213_213698


namespace product_divisible_by_4_probability_l213_213608

theorem product_divisible_by_4_probability :
  let chips := {1, 2, 4, 6}
  let box1 := chips
  let box2 := chips
  let total_outcomes := box1.product(box2)
  let favorable_outcomes := { (a, b) ∈ total_outcome | (a * b) % 4 == 0 }
  (favorable_outcomes.card / total_outcomes.card : ℚ) = 3 / 4 :=
by
  sorry

end product_divisible_by_4_probability_l213_213608


namespace min_value_geometric_seq_a6_l213_213659

theorem min_value_geometric_seq_a6 :
  ∃ a_1 q: ℚ, (1 < q ∧ q < 2) ∧ (∀ n : ℕ, 1 ≤ n ∧ n ≤ 6 → ∃ m : ℕ, a_1 * q^(n-1) = m) ∧ 
  (let a_6 := a_1 * q^5 in a_6 = 243) :=
sorry

end min_value_geometric_seq_a6_l213_213659


namespace plane_passing_through_A_perpendicular_to_BC_l213_213496

-- Define the points A, B, and C
structure Point3D :=
(x : ℝ)
(y : ℝ)
(z : ℝ)

def A : Point3D := { x := -3, y := 7, z := 2 }
def B : Point3D := { x := 3, y := 5, z := 1 }
def C : Point3D := { x := 4, y := 5, z := 3 }

-- Define the vector BC as the difference between points C and B
def vectorBC (B C : Point3D) : Point3D :=
{ x := C.x - B.x,
  y := C.y - B.y,
  z := C.z - B.z }

-- Define the equation of the plane passing through point A and 
-- perpendicular to vector BC
def plane_eq (A : Point3D) (n : Point3D) (x y z : ℝ) : Prop :=
n.x * (x - A.x) + n.y * (y - A.y) + n.z * (z - A.z) = 0

-- Define the proof problem
theorem plane_passing_through_A_perpendicular_to_BC :
  ∀ (x y z : ℝ), plane_eq A (vectorBC B C) x y z ↔ x + 2 * z - 1 = 0 :=
by
  -- the proof part
  sorry

end plane_passing_through_A_perpendicular_to_BC_l213_213496


namespace complex_number_first_quadrant_condition_l213_213014

theorem complex_number_first_quadrant_condition (z : ℂ) (h : z = 1 / Complex.I) : 
  let w := z^3 + 1 in
  0 < w.re ∧ 0 < w.im :=
by sorry

end complex_number_first_quadrant_condition_l213_213014


namespace similarity_of_inductive_and_analogical_reasoning_l213_213886

def inductive_reasoning : Prop :=
  ∀ (specific_facts : Set α) (general_principle : α → Prop), true

def analogical_reasoning : Prop :=
  ∀ (object1 object2 : α) (attributes_shared : α → Prop),
    (attributes_shared object1 → attributes_shared object2) → true

theorem similarity_of_inductive_and_analogical_reasoning :
  (∀ (specific_facts : Set α) (general_principle : α → Prop),
    (inductive_reasoning → ¬ necessarily_correct) ∧
    (analogical_reasoning → ¬ necessarily_correct)) :=
by
  sorry

end similarity_of_inductive_and_analogical_reasoning_l213_213886


namespace problem_range_of_g_l213_213138

noncomputable def g (x : ℝ) : ℝ := arctan x + arctan ((2 - 3 * x) / (2 + 3 * x))

theorem problem_range_of_g :
  ∃ S, (S = {-π / 2, π / 4}) ∧ ∀ x : ℝ, g x ∈ S := 
sorry

end problem_range_of_g_l213_213138


namespace postage_arrangements_l213_213605

-- Define the stamps and their quantities
def stamps : List (ℕ × ℕ) := [(1,1), (2,2), (3,3), (4,4), (5,5), (6,6), (7,7), (8,8), (9,9)]

-- Define the problem conditions 
def valid_arrangement (arrangement : List ℕ) : Prop :=
  arrangement.sum = 10 ∧
  arrangement.all (λ n, ∃ (k : ℕ), (k, n) ∈ stamps)

-- Prove the number of different arrangements (considering identical stamps) is 88
theorem postage_arrangements : ∃ (arrangements : List (List ℕ)), arrangements.length = 88 ∧ ∀ a ∈ arrangements, valid_arrangement a :=
by
  sorry

end postage_arrangements_l213_213605


namespace smallest_positive_value_of_a_minus_b_l213_213400

theorem smallest_positive_value_of_a_minus_b :
  ∃ (a b : ℤ), 17 * a + 6 * b = 13 ∧ a - b = 17 :=
by
  sorry

end smallest_positive_value_of_a_minus_b_l213_213400


namespace area_outside_of_circles_is_0_45_l213_213436

noncomputable def area_inside_outside_circles : ℝ :=
  let area_rectangle := 4 * 6
  let area_quarter_circles := (π * 2^2 / 4) + (π * 3^2 / 4) + (π * 4^2 / 4) + (π * 1^2 / 4)
  area_rectangle - area_quarter_circles

theorem area_outside_of_circles_is_0_45 : abs (area_inside_outside_circles - 0.45) < 0.01 :=
by
  sorry

end area_outside_of_circles_is_0_45_l213_213436


namespace circle_tangent_radius_l213_213519

theorem circle_tangent_radius (k : ℝ) (h : k > 3) :
  ∀ (r : ℝ), 
    (distance_from_center_to_line (0, k) (-2, 1, 0) = r) ∧
    (distance_from_center_to_line (0, k) (2, 1, 0) = r) ∧
    (r = |k - 3|) →
      r = 3*sqrt (5) + 3 := 
by
  sorry

def distance_from_center_to_line (P : ℝ × ℝ) (line : ℝ × ℝ × ℝ) : ℝ := 
  let (x, y) := P
  let (A, B, C) := line
  (abs (A * x + B * y + C)) / sqrt (A * A + B * B)

end circle_tangent_radius_l213_213519


namespace sum_of_primes_no_solution_l213_213599

def is_prime (n : ℕ) : Prop := Nat.Prime n

def no_solution (p : ℕ) : Prop :=
  is_prime p ∧ ¬ ∃ x : ℤ, 5 * (10 * x + 2) ≡ 3 [MOD p]

def gcd_condition (p : ℕ) : Prop :=
  p = 2 ∨ p = 5

theorem sum_of_primes_no_solution : (∑ p in {p | is_prime p ∧ gcd_condition p}, p) = 7 :=
by
  sorry

end sum_of_primes_no_solution_l213_213599


namespace remainder_of_large_number_l213_213490

theorem remainder_of_large_number :
  let number := 65985241545898754582556898522454889
  let last_four_digits := 4889
  last_four_digits % 16 = 9 := 
by
  let number := 65985241545898754582556898522454889
  let last_four_digits := 4889
  show last_four_digits % 16 = 9
  sorry

end remainder_of_large_number_l213_213490


namespace rectangular_garden_side_length_l213_213966

theorem rectangular_garden_side_length (a b : ℝ) (h1 : 2 * a + 2 * b = 60) (h2 : a * b = 200) (h3 : b = 10) : a = 20 :=
by
  sorry

end rectangular_garden_side_length_l213_213966


namespace expected_steps_from_initial_state_l213_213820

-- Define the initial and target states
def initial_state : List Int := [1, 0, 1, 0]
def target_state : List Int := [0, 1, 0, 1]

-- Define a function that models the transitions and expected number of steps
noncomputable def expected_steps (s : List Int) : ℚ :=
if s = target_state then
  0
else
  let t := [s.take 1 ++ [0] ++ s.drop 2,
            s.take 2 ++ [1] ++ s.drop 3,
            s.take 1 ++ [1] ++ s.drop 2,
            s.take 2 ++ [0] ++ s.drop 3] in
  1 + (1/4 : ℚ) * (expected_steps t.head) + (1/4 : ℚ) * (expected_steps t.tail.head) +
      (1/4 : ℚ) * (expected_steps t.tail.tail.head) + (1/4 : ℚ) * (expected_steps t.tail.tail.tail.head)

-- State the theorem about the expected number of steps starting from the initial state
theorem expected_steps_from_initial_state : expected_steps initial_state = 6 :=
sorry

end expected_steps_from_initial_state_l213_213820


namespace finger_cycle_2004th_l213_213797

def finger_sequence : List String :=
  ["Little finger", "Ring finger", "Middle finger", "Index finger", "Thumb", "Index finger", "Middle finger", "Ring finger"]

theorem finger_cycle_2004th : 
  finger_sequence.get! ((2004 - 1) % finger_sequence.length) = "Index finger" :=
by
  -- The proof is not required, so we use sorry
  sorry

end finger_cycle_2004th_l213_213797


namespace unique_three_digit_sum_27_l213_213643

/--
There is exactly one three-digit whole number such that the sum of its digits is 27.
-/
theorem unique_three_digit_sum_27 :
  ∃! (n : ℕ), 
    100 ≤ n ∧ n < 1000 ∧ 
    let a := n / 100, b := (n / 10) % 10, c := n % 10
    in a + b + c = 27 := sorry

end unique_three_digit_sum_27_l213_213643


namespace double_sum_eq_fraction_l213_213124

theorem double_sum_eq_fraction : 
  (\sum n in Finset.range (n+1).filter n ≥ 3).sum (λ n, (\sum k in Finset.range (n-1).filter k ≥ 1).sum (λ k, (2 * k) / (3 : ℝ)^(n+k))) = 729 / 18252 :=
by 
  sorry

end double_sum_eq_fraction_l213_213124


namespace closest_perfect_square_l213_213009

theorem closest_perfect_square (n : ℕ) (h : n = 315) : ∃ k : ℕ, k^2 = 324 ∧ ∀ m : ℕ, m^2 ≠ 315 ∨ abs (n - m^2) > abs (n - k^2) :=
by
  use 18
  sorry

end closest_perfect_square_l213_213009


namespace intersection_line_circle_l213_213999

theorem intersection_line_circle (a : ℝ) :
  let line : ℝ → ℝ → Prop := λ x y, x + a * y - 1 = 0
  let circle : ℝ → ℝ → Prop := λ x y, x^2 + y^2 - 4 * x = 0
  ∃ x1 x2 y1 y2, line x1 y1 ∧ line x2 y2 ∧ circle x1 y1 ∧ circle x2 y2 ∧ (x1 = x2 ∧ y1 ≠ y2) :=
sorry

end intersection_line_circle_l213_213999


namespace problem_2_8_3_4_7_2_2_l213_213117

theorem problem_2_8_3_4_7_2_2 : 2 + 8 * 3 - 4 + 7 * 2 / 2 = 29 := by
  sorry

end problem_2_8_3_4_7_2_2_l213_213117


namespace johns_number_l213_213392

theorem johns_number (n : ℕ) :
  64 ∣ n ∧ 45 ∣ n ∧ 1000 < n ∧ n < 3000 -> n = 2880 :=
by
  sorry

end johns_number_l213_213392


namespace exists_positive_2x2_subgrid_l213_213030

theorem exists_positive_2x2_subgrid (a : Fin 5 → Fin 5 → ℝ)
  (h : ∀ i j : Fin 3, 0 < ∑ x in Fin3, ∑ y in Fin3, a (i.val + x.val) (j.val + y.val)) :
  ∃ i j : Fin 4, 0 < (a i j + a i (j + 1) + a (i + 1) j + a (i + 1) (j + 1)) :=
sorry


end exists_positive_2x2_subgrid_l213_213030


namespace number_square_roots_l213_213361

theorem number_square_roots (a x : ℤ) (h1 : x = (2 * a + 3) ^ 2) (h2 : x = (a - 18) ^ 2) : x = 169 :=
by 
  sorry

end number_square_roots_l213_213361


namespace ceil_square_eq_four_l213_213157

theorem ceil_square_eq_four : (⌈(-7 / 4: ℚ)^2⌉ : ℤ) = 4 := by
  sorry

end ceil_square_eq_four_l213_213157


namespace simplify_sqrt_sum_l213_213447

theorem simplify_sqrt_sum : (Real.sqrt 72 + Real.sqrt 32) = 10 * Real.sqrt 2 := by
  sorry

end simplify_sqrt_sum_l213_213447


namespace ceil_of_neg_frac_squared_l213_213202

-- Define the negated fraction
def neg_frac : ℚ := -7 / 4

-- Define the squared value of the negated fraction
def squared_value : ℚ := neg_frac ^ 2

-- Define the ceiling function applied to the squared value
def ceil_squared_value : ℤ := Int.ceil squared_value

-- Prove that the ceiling of the squared value is 4
theorem ceil_of_neg_frac_squared : ceil_squared_value = 4 := 
by sorry

end ceil_of_neg_frac_squared_l213_213202


namespace trapezoid_area_l213_213092

theorem trapezoid_area 
  (a b h c : ℝ) 
  (ha : 2 * 0.8 * a + b = c)
  (hb : c = 20) 
  (hc : h = 0.6 * a) 
  (hd : b + 1.6 * a = 20)
  (angle_base : ∃ θ : ℝ, θ = arcsin 0.6)
  : 
  (1 / 2) * (b + c) * h = 72 :=
sorry

end trapezoid_area_l213_213092


namespace trapezoid_area_l213_213081

-- Define the problem statement
theorem trapezoid_area 
  (a b h: ℝ)
  (b₁ b₂: ℝ)
  (θ: ℝ) 
  (h₃: θ = Real.arcsin 0.6)
  (h₄: a = 20)
  (h₅: b = a - 2 * b₁ * Real.sin θ) 
  (h₆: h = b₁ * Real.cos θ) 
  (h₇: θ = Real.arcsin (3/5)) 
  (circum: isosceles_trapezoid_circumscribed a b₁ b₂) :
  ((1 / 2) * (a + b₂) * h = 2000 / 27) :=
by sorry

end trapezoid_area_l213_213081


namespace probability_stopping_in_C_l213_213036

noncomputable def probability_C : ℚ :=
  let P_A := 1 / 5
  let P_B := 1 / 5
  let x := (1 - (P_A + P_B)) / 3
  x

theorem probability_stopping_in_C :
  probability_C = 1 / 5 :=
by
  unfold probability_C
  sorry

end probability_stopping_in_C_l213_213036


namespace find_n_l213_213618

theorem find_n (n : ℕ) : 2 ^ 6 * 3 ^ 3 * n = 10.factorial → n = 350 := 
by intros; sorry

end find_n_l213_213618


namespace rhombus_area_l213_213862

theorem rhombus_area (x y : ℝ) (h : |x - 1| + |y - 1| = 1) : 
  ∃ (area : ℝ), area = 2 :=
by
  sorry

end rhombus_area_l213_213862


namespace total_students_l213_213715

theorem total_students (h1 : 4 = 4) (h2 : 9 = 9) (n_lines : 5 = 5)
  (same_number_of_students : ∀ α β : ℕ, α = β) : 
  let students_in_row := 4 + 9 - 1 in
  students_in_row * 5 = 60 :=
by
  let students_in_row := 4 + 9 - 1
  have h3 : students_in_row = 12 := 
    calc 
      students_in_row = 4 + 9 - 1 : by rfl
      _ = 12 : by norm_num
  have total := students_in_row * 5
  have h4 : total = 60 := 
    calc 
      total = 12 * 5 : by rw h3
      _ = 60 : by norm_num
  exact h4

end total_students_l213_213715


namespace ceil_square_of_neg_fraction_l213_213190

theorem ceil_square_of_neg_fraction : 
  (Int.ceil ((-7 / 4 : ℚ)^2 : ℚ)).toNat = 4 := by
  sorry

end ceil_square_of_neg_fraction_l213_213190


namespace calories_per_shake_l213_213391

theorem calories_per_shake (total_calories_per_day : ℕ) (breakfast_calories : ℕ)
  (lunch_percentage_increase : ℕ) (dinner_multiplier : ℕ) (number_of_shakes : ℕ)
  (daily_calories : ℕ) :
  total_calories_per_day = breakfast_calories +
                            (breakfast_calories + (lunch_percentage_increase * breakfast_calories / 100)) +
                            (2 * (breakfast_calories + (lunch_percentage_increase * breakfast_calories / 100))) →
  daily_calories = total_calories_per_day + number_of_shakes * (daily_calories - total_calories_per_day) / number_of_shakes →
  daily_calories = 3275 → breakfast_calories = 500 →
  lunch_percentage_increase = 25 →
  dinner_multiplier = 2 →
  number_of_shakes = 3 →
  (daily_calories - total_calories_per_day) / number_of_shakes = 300 := by 
  sorry

end calories_per_shake_l213_213391


namespace angle_bisector_construction_correct_l213_213422

noncomputable def point := ℝ × ℝ 
def line (p1 p2 : point) := { l : point × point // ∃ k, fst l = p1 + k * (p2 - p1) } 

-- Given conditions
variables (L1 L2 : line)
variables (P O : point)
variables (k : ℝ) (k_ne_zero : k ≠ 0) (k_pos : 0 < k)

-- Image of lines under a homothety with a small coefficient k
def homothety (O P : point) (k : ℝ) := (O + k * (P - O))

-- The transformed lines L1' and L2' intersect at P'
def P' := homothety O P k
def L1' := line O P'
def L2' := line O P'

-- Angle bisector construction between L1' and L2'
def angle_bisector (l1 l2 : line) (P' : point) := 
    sorry -- Placeholder for angle bisector construction

-- Reversing homothety with coefficient 1/k to map back to original scale
def homothety_inverse (O P : point) (k : ℝ) := (O + (1/k) * (P - O))

-- Final step to prove the constructed line segment is the required angle bisector
def required_segment (O : point) (L1 L2 : line) (k : ℝ) (P' : point) := 
    homothety_inverse O (angle_bisector L1' L2' P') k

-- Lean 4 statement asserting the correctness of the solution to the problem
theorem angle_bisector_construction_correct : 
  ∃ O P : point, ∃ k : ℝ, 
  let P' := homothety O P k in
  let L1' := line O P' in
  let L2' := line O P' in
  let bisector := angle_bisector L1' L2' P' in
  ∀ O L1 L2 k, 
  (required_segment O L1 L2 k P' = angle_bisector L1 L2 P') :=
sorry

end angle_bisector_construction_correct_l213_213422


namespace probability_two_dice_same_number_l213_213823

theorem probability_two_dice_same_number (n : ℕ) (sides : ℕ) (h_n : n = 8) (h_sides : sides = 6):
  (∃ (prob : ℝ), prob = 1) :=
by
  sorry

end probability_two_dice_same_number_l213_213823


namespace sum_of_roots_abs_quadratic_l213_213587

theorem sum_of_roots_abs_quadratic (x : ℝ) :
    let y := |x|
    y^2 - 5 * y + 6 = 0 → 2 * (x = 2 ∨ x = -2) + 2 * (x = 3 ∨ x = -3) = 0 :=
begin
  sorry
end

end sum_of_roots_abs_quadratic_l213_213587


namespace trapezoid_has_area_approx_74_14_l213_213078

-- Define the properties and conditions of the isosceles trapezoid
def trapezoid_area (longer_base : ℝ) (base_angle : ℝ) : ℝ :=
  let x := 20.0 / 1.8 in
  let y := 0.2 * x in
  let height := 0.6 * x in
  (1 / 2) * (y + longer_base) * height

-- Main statement
theorem trapezoid_has_area_approx_74_14 :
  let longer_base := 20
  let base_angle := Real.arcsin 0.6
  abs (trapezoid_area longer_base base_angle - 74.14) < 0.01 :=
by
  sorry

end trapezoid_has_area_approx_74_14_l213_213078


namespace trapezoid_has_area_approx_74_14_l213_213077

-- Define the properties and conditions of the isosceles trapezoid
def trapezoid_area (longer_base : ℝ) (base_angle : ℝ) : ℝ :=
  let x := 20.0 / 1.8 in
  let y := 0.2 * x in
  let height := 0.6 * x in
  (1 / 2) * (y + longer_base) * height

-- Main statement
theorem trapezoid_has_area_approx_74_14 :
  let longer_base := 20
  let base_angle := Real.arcsin 0.6
  abs (trapezoid_area longer_base base_angle - 74.14) < 0.01 :=
by
  sorry

end trapezoid_has_area_approx_74_14_l213_213077


namespace cost_of_graveling_l213_213930

/-
  A rectangular lawn of dimensions 100 m by 60 m has two roads, each 10 m wide,
  one running parallel to the length and the other parallel to the breadth.
  Prove that the cost of graveling the two roads at Rs. 3 per sq m is Rs. 4500.
-/

theorem cost_of_graveling
  (lawn_length : ℝ) (lawn_breadth : ℝ)
  (road_width : ℝ) (gravel_cost_rate : ℝ) :
  lawn_length = 100 ∧ lawn_breadth = 60 ∧ road_width = 10 ∧ gravel_cost_rate = 3 →
  let area_first_road := road_width * lawn_breadth
  let area_second_road := road_width * (lawn_length - road_width)
  let total_area := area_first_road + area_second_road
  let total_cost := total_area * gravel_cost_rate
  total_cost = 4500 :=
begin
  intro h,
  rcases h with ⟨h_ll, h_lb, h_rw, h_gc⟩,
  simp only,
  let area_first_road := 10 * 60,
  let area_second_road := 10 * (100 - 10),
  let total_area := area_first_road + area_second_road,
  let total_cost := total_area * 3,
  exact eq_refl 4500,
end

end cost_of_graveling_l213_213930


namespace total_volume_of_mixed_solutions_l213_213044

theorem total_volume_of_mixed_solutions :
  let v1 := 3.6
  let v2 := 1.4
  v1 + v2 = 5.0 := by
  sorry

end total_volume_of_mixed_solutions_l213_213044


namespace joseph_birth_year_l213_213460

theorem joseph_birth_year 
  (first_amc_year : ℕ) 
  (is_given_annually : ∀ n : ℕ, ∃! y : ℕ, y = first_amc_year + n) 
  (joseph_age_amc7 : ℕ) 
  (joseph_age : ℕ) 
  (joseph_amc7_year : ℕ) :
  first_amc_year = 1987 ∧ 
  joseph_age_amc7 = 7 ∧ 
  joseph_age = 15 ∧ 
  joseph_amc7_year = first_amc_year + joseph_age_amc7 - 1 ∧ 
  joseph_amc7_year - joseph_age = 1978 := 
by
  split
  exact 1987
  split
  exact 7
  split
  exact 15
  split
  simp [joseph_amc7_year]
  sorry

end joseph_birth_year_l213_213460


namespace roots_difference_of_quadratic_l213_213626

theorem roots_difference_of_quadratic (a b c : ℝ) (h_eq : a = 1 ∧ b = -9 ∧ c = 14) 
  (h_cond : -b = c) : 
  (let roots := (r1 r2 : ℝ)(h_eq₁ : r1 + r2 = -b / a)
  (h_eq₂ : r1 * r2 = c / a) in
  abs (r1 - r2) = 5) :=
begin
  sorry
end

end roots_difference_of_quadratic_l213_213626


namespace probability_at_least_two_same_l213_213828

theorem probability_at_least_two_same (n : ℕ) (H : n = 8) : 
  (∃ i j, (1 ≤ i ∧ i ≤ n) ∧ (1 ≤ j ∧ j ≤ n) ∧ i ≠ j ∧ ∀ (x : ℕ), x ∈ {i, j}) :=
by
  sorry

end probability_at_least_two_same_l213_213828


namespace cos_squared_alpha_plus_pi_div_four_l213_213297

theorem cos_squared_alpha_plus_pi_div_four (α : ℝ) (h : sin (2 * α) = 2 / 3) :
  cos^2 (α + π / 4) = 1 / 6 :=
sorry

end cos_squared_alpha_plus_pi_div_four_l213_213297


namespace hexagram_stone_arrangements_l213_213801

-- The main statement that follows directly from the problem definition.
theorem hexagram_stone_arrangements : 
  let total_stones := 12
  let dihedral_group_order := 24
  (fact total_stones) / dihedral_group_order = 19958400 :=
  by
    let total_stones := 12
    let dihedral_group_order := 24
    show (fact total_stones) / dihedral_group_order = 19958400
    sorry

end hexagram_stone_arrangements_l213_213801


namespace compound_contains_one_nitrogen_atom_l213_213042

theorem compound_contains_one_nitrogen_atom
  (h : ℝ := 1.01)  -- atomic weight of Hydrogen
  (br : ℝ := 79.90)  -- atomic weight of Bromine
  (n : ℝ := 14.01)  -- atomic weight of Nitrogen
  (h_count : ℕ := 4)  -- number of Hydrogen atoms
  (br_count : ℕ := 1)  -- number of Bromine atoms
  (molecular_weight : ℝ := 98)  -- molecular weight of the compound
  : (molecular_weight - (h_count * h + br_count * br)) / n ≈ 1 := 
by
  sorry

end compound_contains_one_nitrogen_atom_l213_213042


namespace ratio_depth_to_height_l213_213070

noncomputable def height_ron : ℝ := 12
noncomputable def depth_water : ℝ := 60

theorem ratio_depth_to_height : depth_water / height_ron = 5 := by
  sorry

end ratio_depth_to_height_l213_213070


namespace hyperbola_eccentricity_l213_213679

theorem hyperbola_eccentricity
  (a b : ℝ) 
  (ha : a > 0) 
  (hb : b > 0)
  (A P : ℝ × ℝ) 
  (B : ℝ × ℝ)
  (hA : A.fst = B.fst ∧ A.snd = -B.snd)
  (hPA_PB_prod : 
    let k_PA := (P.snd - A.snd) / (P.fst - A.fst),
        k_PB := (P.snd - B.snd) / (P.fst - B.fst)
    in k_PA * k_PB = (3 / 4))
  (hA_hyp : A.fst^2 / a^2 - A.snd^2 / b^2 = 1)
  (hP_hyp : P.fst^2 / a^2 - P.snd^2 / b^2 = 1) : 
  let e := Real.sqrt (1 + b^2 / a^2) in
  e = Real.sqrt(7) / 2 := 
sorry

end hyperbola_eccentricity_l213_213679


namespace range_of_m_l213_213794

-- Defining the function f and its properties
variables {f : ℝ → ℝ} [Differentiable ℝ f]

-- Condition: f(x) = 4x^2 - f(-x)
axiom axiom1 : ∀ x : ℝ, f(x) = 4 * x^2 - f(-x)

-- Condition: f'(x) < 4x when x < 0
axiom axiom2 : ∀ x : ℝ, x < 0 → deriv f x < 4 * x

-- Condition: f(m+1) ≤ f(-m) + 4m + 2
axiom axiom3 : ∀ m : ℝ, f(m + 1) ≤ f(-m) + 4 * m + 2

-- Statement: Prove that the range of m is [-1/2, +∞)
theorem range_of_m : ∀ m : ℝ, (f m) ≤ (f (-m) + 4 * m + 2) → m ∈ Ici (-1 / 2) :=
by 
  sorry

end range_of_m_l213_213794


namespace Clara_sells_third_type_boxes_l213_213114

variable (total_cookies boxes_first boxes_second boxes_third : ℕ)
variable (cookies_per_first cookies_per_second cookies_per_third : ℕ)

theorem Clara_sells_third_type_boxes (h1 : cookies_per_first = 12)
                                    (h2 : boxes_first = 50)
                                    (h3 : cookies_per_second = 20)
                                    (h4 : boxes_second = 80)
                                    (h5 : cookies_per_third = 16)
                                    (h6 : total_cookies = 3320) :
                                    boxes_third = 70 :=
by
  sorry

end Clara_sells_third_type_boxes_l213_213114


namespace cinema_seating_problem_l213_213982

noncomputable def count_valid_arrangements : ℕ :=
  let total_arrangements := 2 * (5!) -- total ways Anton and Boris are next to each other
  let invalid_arrangements := 2 * 2 * (4!) -- ways Anton/Boris and Vadim/Gena pairs are both next to each other
  total_arrangements - invalid_arrangements

theorem cinema_seating_problem :
  count_valid_arrangements = 144 :=
by
  let fact_5 := 5 * 4 * 3 * 2 * 1
  let fact_4 := 4 * 3 * 2 * 1
  have h1 : total_arrangements = 2 * fact_5 := rfl
  have h2 : invalid_arrangements = 2 * 2 * fact_4 := rfl
  have h3 : 2 * fact_5 - 2 * 2 * fact_4 = 144 :=
    by
      calc
        2 * fact_5 - 2 * 2 * fact_4
            = 2 * 120 - 2 * 2 * 24 : rfl
        ... = 240 - 96           : rfl
        ... = 144                : rfl
  show count_valid_arrangements = 144, from h3

#eval count_valid_arrangements  -- Evaluating this should return 144

end cinema_seating_problem_l213_213982


namespace round_nearest_hundredth_l213_213438

theorem round_nearest_hundredth (x : ℝ) (h : x = 42.1376) : Real.round (x * 100) / 100 = 42.14 :=
by
  rw [h]
  -- The condition and exact steps would normally follow here.
sorry

end round_nearest_hundredth_l213_213438


namespace martha_savings_l213_213815

def daily_allowance : ℝ := 12
def saving_days : ℕ := 6
def saving_half (amount : ℝ) : ℝ := (1/2) * amount
def saving_quarter (amount : ℝ) : ℝ := (1/4) * amount

theorem martha_savings : 
  saving_days * saving_half daily_allowance + saving_quarter daily_allowance = 39 := 
by
  sorry

end martha_savings_l213_213815


namespace ceil_square_eq_four_l213_213155

theorem ceil_square_eq_four : (⌈(-7 / 4: ℚ)^2⌉ : ℤ) = 4 := by
  sorry

end ceil_square_eq_four_l213_213155


namespace find_n_eq_1050_l213_213614

theorem find_n_eq_1050 : ∃ (n : ℕ), 2^6 * 3^3 * n = 10! ∧ n = 1050 :=
by
  use 1050
  split
  · calc
    2^6 * 3^3 * 1050 = 10! := sorry
  · rfl

end find_n_eq_1050_l213_213614


namespace unique_integer_solution_m_l213_213706

theorem unique_integer_solution_m {m : ℤ} (h : ∀ x : ℤ, |2 * x - m| ≤ 1 → x = 2) : m = 4 := 
sorry

end unique_integer_solution_m_l213_213706


namespace part_1_part_2_part_3_l213_213958

noncomputable def f : ℝ → ℝ := sorry

axiom functional_eq {x y : ℝ} : f(x + y) = f(x) + f(y)
axiom negative_condition {x : ℝ} (h : x < 0) : f(x) > 0

theorem part_1 : f(0) = 0 := sorry

theorem part_2 {x1 x2 : ℝ} (h : x1 < x2) : f(x1) > f(x2) := sorry

theorem part_3 (a : ℝ) (h : a > 0) :
  (∃ x : ℝ, a < x ∧ x < 2/a) ∨ 
  (a = real.sqrt 2 ∧ ∀ x : ℝ, false) ∨ 
  (a > real.sqrt 2 ∧ ∃ x : ℝ, 2/a < x ∧ x < a) := sorry

end part_1_part_2_part_3_l213_213958


namespace Max_Among_RealNumbers_l213_213978

theorem Max_Among_RealNumbers : 
  ∀ (a b c d : ℝ), {a, b, c, d} = { -2, 0, 3, Real.sqrt 6 } → 
  max (max (max a b) c) d = 3 := 
by
sory

end Max_Among_RealNumbers_l213_213978


namespace martha_black_butterflies_l213_213805

-- Define the hypotheses
variables (total_butterflies blue_butterflies yellow_butterflies black_butterflies : ℕ)

-- Given conditions
def martha_collection_conditions : Prop :=
  total_butterflies = 19 ∧
  blue_butterflies = 6 ∧
  blue_butterflies = 2 * yellow_butterflies

-- The statement we want to prove
theorem martha_black_butterflies : martha_collection_conditions total_butterflies blue_butterflies yellow_butterflies black_butterflies →
  black_butterflies = 10 :=
sorry

end martha_black_butterflies_l213_213805


namespace probability_two_dice_same_number_l213_213824

theorem probability_two_dice_same_number (n : ℕ) (sides : ℕ) (h_n : n = 8) (h_sides : sides = 6):
  (∃ (prob : ℝ), prob = 1) :=
by
  sorry

end probability_two_dice_same_number_l213_213824


namespace floor_width_l213_213437

theorem floor_width (tile_length tile_width floor_length max_tiles : ℕ) (h1 : tile_length = 25) (h2 : tile_width = 65) (h3 : floor_length = 150) (h4 : max_tiles = 36) :
  ∃ floor_width : ℕ, floor_width = 450 :=
by
  sorry

end floor_width_l213_213437


namespace find_n_eq_1050_l213_213613

theorem find_n_eq_1050 : ∃ (n : ℕ), 2^6 * 3^3 * n = 10! ∧ n = 1050 :=
by
  use 1050
  split
  · calc
    2^6 * 3^3 * 1050 = 10! := sorry
  · rfl

end find_n_eq_1050_l213_213613


namespace number_added_l213_213534

def initial_number : ℕ := 9
def final_resultant : ℕ := 93

theorem number_added : ∃ x : ℕ, 3 * (2 * initial_number + x) = final_resultant ∧ x = 13 := by
  sorry

end number_added_l213_213534


namespace unique_three_digit_sum_27_l213_213644

/--
There is exactly one three-digit whole number such that the sum of its digits is 27.
-/
theorem unique_three_digit_sum_27 :
  ∃! (n : ℕ), 
    100 ≤ n ∧ n < 1000 ∧ 
    let a := n / 100, b := (n / 10) % 10, c := n % 10
    in a + b + c = 27 := sorry

end unique_three_digit_sum_27_l213_213644


namespace sum_of_primes_no_solution_l213_213598

def is_prime (n : ℕ) : Prop := Nat.Prime n

def no_solution (p : ℕ) : Prop :=
  is_prime p ∧ ¬ ∃ x : ℤ, 5 * (10 * x + 2) ≡ 3 [MOD p]

def gcd_condition (p : ℕ) : Prop :=
  p = 2 ∨ p = 5

theorem sum_of_primes_no_solution : (∑ p in {p | is_prime p ∧ gcd_condition p}, p) = 7 :=
by
  sorry

end sum_of_primes_no_solution_l213_213598


namespace sum_of_primes_no_solution_l213_213596

def is_prime (n : ℕ) : Prop := Nat.Prime n

def no_solution (p : ℕ) : Prop :=
  is_prime p ∧ ¬ ∃ x : ℤ, 5 * (10 * x + 2) ≡ 3 [MOD p]

def gcd_condition (p : ℕ) : Prop :=
  p = 2 ∨ p = 5

theorem sum_of_primes_no_solution : (∑ p in {p | is_prime p ∧ gcd_condition p}, p) = 7 :=
by
  sorry

end sum_of_primes_no_solution_l213_213596


namespace remainder_sum_f_l213_213779

def f (n : ℕ) : ℕ :=
  Nat.min_fac (n ^ 4 + 1)

theorem remainder_sum_f (sum_f : ℕ) (h : sum_f = (Finset.range 2014).sum (λ n, f (n + 1))) : 
  sum_f % 8 = 5 :=
  sorry

end remainder_sum_f_l213_213779


namespace no_positive_rational_r_l213_213398

theorem no_positive_rational_r (P : ℤ[X]) (n : ℕ) (hP : P.coeff ∈ set.range (@Int.cast ℤ)) (hn : P.eval (n^2) = 2022) : 
  ¬ ∃ r : ℚ, r > 0 ∧ P.eval (r^2) = 2024 := 
sorry

end no_positive_rational_r_l213_213398


namespace decreasing_power_function_l213_213646

open Nat

/-- For the power function y = x^(m^2 + 2*m - 3) (where m : ℕ) 
    to be a decreasing function in the interval (0, +∞), prove that m = 0. -/
theorem decreasing_power_function (m : ℕ) (h : m^2 + 2 * m - 3 < 0) : m = 0 := 
by
  sorry

end decreasing_power_function_l213_213646


namespace expression_of_f_min_value_m_graph_below_line_l213_213658

-- Definitions given in the conditions
def f (x : ℝ) := x * Real.log x + a * x^2 - 3
def f' (x : ℝ) := (derivative f x)

-- Theorem statements
theorem expression_of_f (a : ℝ) (h_deriv : f'(1) = -1) : f(x) = x * Real.log x - x^2 - 3 :=
by
  -- Placeholder for proof
  sorry

theorem min_value_m (m : ℝ) (h_ineq : ∀ x > 0, f(x) - m * x ≤ -3) : m ≥ -1 :=
by
  -- Placeholder for proof
  sorry

theorem graph_below_line (x : ℝ) : (f(x) - x * Real.exp x + x^2 < -2 * x - 3) :=
by
  -- Placeholder for proof
  sorry

end expression_of_f_min_value_m_graph_below_line_l213_213658


namespace meteor_temperature_increase_ice_mass_melted_l213_213471

-- Mass of the meteor in kg and g
def mass_kg : ℝ := 1
def mass_g : ℝ := mass_kg * 1000

-- Specific heat of the meteor in cal/g°C
def specific_heat : ℝ := 0.114

-- Latent heat of fusion for ice in cal/g
def latent_heat_fusion_ice : ℝ := 80

-- Initial and final speeds in m/s
def initial_speed_m_s : ℝ := 50 * 1000
def final_speed_m_s : ℝ := 5 * 1000

-- Conversion factor from Joules to calories
def joules_to_calories : ℝ := 0.239

-- Calculations for kinetic energy
def initial_kinetic_energy : ℝ := 0.5 * mass_kg * initial_speed_m_s^2
def final_kinetic_energy : ℝ := 0.5 * mass_kg * final_speed_m_s^2

-- Change in kinetic energy in Joules
def delta_energy_joules : ℝ := initial_kinetic_energy - final_kinetic_energy

-- Change in kinetic energy in calories
def delta_energy_calories : ℝ := delta_energy_joules * joules_to_calories

-- Heat absorbed and temperature change
def heat_absorbed_calories : ℝ := delta_energy_calories
def temperature_increase : ℝ := heat_absorbed_calories / (mass_g * specific_heat)

-- Mass of ice melted
def mass_ice_melted_g : ℝ := heat_absorbed_calories / latent_heat_fusion_ice
def mass_ice_melted_kg : ℝ := mass_ice_melted_g / 1000

-- Theorem statements
theorem meteor_temperature_increase : 
  temperature_increase = 2594548.25 := sorry

theorem ice_mass_melted : 
  mass_ice_melted_kg = 3697.03 := sorry

end meteor_temperature_increase_ice_mass_melted_l213_213471


namespace problem1_solution_set_problem2_a_range_l213_213993

-- Define the function f
def f (x a : ℝ) := |x - a| + 5 * x

-- Problem Part 1: Prove for a = -1, the solution set for f(x) ≤ 5x + 3 is [-4, 2]
theorem problem1_solution_set :
  ∀ (x : ℝ), f x (-1) ≤ 5 * x + 3 ↔ -4 ≤ x ∧ x ≤ 2 := by
  sorry

-- Problem Part 2: Prove that if f(x) ≥ 0 for all x ≥ -1, then a ≥ 4 or a ≤ -6
theorem problem2_a_range (a : ℝ) :
  (∀ (x : ℝ), x ≥ -1 → f x a ≥ 0) ↔ a ≥ 4 ∨ a ≤ -6 := by
  sorry

end problem1_solution_set_problem2_a_range_l213_213993


namespace min_value_l213_213652

noncomputable def minimum_value_of_y (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : b = (1 - a) / 3) : ℝ :=
3^a + 27^b

theorem min_value (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : b = (1 - a) / 3) :
  minimum_value_of_y a b h1 h2 h3 = 2 * real.sqrt 3 :=
sorry

end min_value_l213_213652


namespace hiring_cost_l213_213481

theorem hiring_cost (days_per_floor : ℕ) (builders : ℕ) (pay_per_day : ℕ) (additional_builders : ℕ) (houses : ℕ) (floors_per_house : ℕ) :
  days_per_floor = 30 →
  builders = 3 →
  pay_per_day = 100 →
  additional_builders = 6 →
  houses = 5 →
  floors_per_house = 6 →
  (additional_builders * houses * floors_per_house * pay_per_day * (days_per_floor / (additional_builders / builders))) = 270000 := 
by
  intros h1 h2 h3 h4 h5 h6
  simp [h1, h2, h3, h4, h5, h6]
  sorry

end hiring_cost_l213_213481


namespace solve_equation_l213_213632

theorem solve_equation (x : ℝ) :
  (15 * x - x^2) / (x + 2) * (x + (15 - x) / (x + 2)) = 60 → x = 12 :=
by
  sorry

end solve_equation_l213_213632


namespace greatest_integer_n_l213_213488

theorem greatest_integer_n (n : ℤ) : n^2 - 9 * n + 20 ≤ 0 → n ≤ 5 := sorry

end greatest_integer_n_l213_213488


namespace polynomial_sum_zero_l213_213931

open BigOperators

theorem polynomial_sum_zero {f : ℕ → ℤ} {n : ℕ} (hf : ∀ k, Polynomial.degree (Polynomial.ofFinsupp (f k)) ≤ n - 1) :
  ∑ k in Finset.range (n + 1), (Nat.choose n k) * (-1)^k * f k = 0 :=
sorry

end polynomial_sum_zero_l213_213931


namespace sequence_2006_l213_213023

theorem sequence_2006 :
  ∀ (a : ℕ → ℚ), 
  a 1 = -1 → 
  a 2 = 2 → 
  (∀ n >= 3, a n = a (n - 1) / a (n - 2)) → 
  a 2006 = 2 :=
by
  intros a h1 h2 hrec
  sorry

end sequence_2006_l213_213023


namespace angle_measure_is_fifty_l213_213866

theorem angle_measure_is_fifty (x : ℝ) :
  (90 - x = (1 / 2) * (180 - x) - 25) → x = 50 := by
  intro h
  sorry

end angle_measure_is_fifty_l213_213866


namespace ones_digit_of_33_pow_33_mul_12_pow_12_l213_213631

noncomputable def ones_digit (n : ℕ) : ℕ :=
  n % 10

theorem ones_digit_of_33_pow_33_mul_12_pow_12 : ones_digit (33^(33*(12^12))) = 1 := by
  -- Definition of the periodic pattern of ones digit of powers of 3
  have periodic_pattern : 
    ∀ (k : ℕ), ones_digit ((3^k) % 10) = if k % 4 == 1 then 3 else if k % 4 == 2 then 9 else if k % 4 == 3 then 7 else 1 := by sorry
  -- Calculation of 33(12^12) % 4
  have exponent_mod_4 : (33 * (12^12)) % 4 = 0 := by sorry 
  -- Conclusion based on periodic pattern and the exponent result
  calc
    ones_digit (33^(33*(12^12))) = ones_digit (3^(33*(12^12))) := by sorry
    ... = ones_digit (3^(0)) := by rw exponent_mod_4; sorry
    ... = 1 := by sorry

#eval ones_digit (33^(33*(12^12)))

end ones_digit_of_33_pow_33_mul_12_pow_12_l213_213631


namespace nat_prime_p_and_5p_plus_1_is_prime_l213_213622

theorem nat_prime_p_and_5p_plus_1_is_prime (p : ℕ) (hp : Nat.Prime p) (h5p1 : Nat.Prime (5 * p + 1)) : p = 2 := 
by 
  -- Sorry is added to skip the proof
  sorry 

end nat_prime_p_and_5p_plus_1_is_prime_l213_213622


namespace solve_original_eq_l213_213852

-- Definitions for the conditions, translated directly into Lean
def original_eq (x : ℝ) : Prop := x^2 + sqrt 2 * x - sqrt 6 = sqrt 3 * x

-- Equality statement asserting the roots of the equation
theorem solve_original_eq : ∃ (x : ℝ), original_eq x ∧ (x = -sqrt 2 ∨ x = sqrt 3) :=
begin
  sorry
end

end solve_original_eq_l213_213852


namespace radius_of_circle_eq_l213_213884

theorem radius_of_circle_eq (x y : ℝ) : (x - 1)^2 + y^2 = 9 → (∃ r : ℝ, r = 3 ∧ r^2 = 9) :=
by
  intro h
  use 3
  constructor
  . rfl
  . exact h

end radius_of_circle_eq_l213_213884


namespace ceiling_of_square_of_neg_7_over_4_is_4_l213_213171

theorem ceiling_of_square_of_neg_7_over_4_is_4 : 
  Real.ceil ((-7 / 4 : Real) ^ 2) = 4 := by
  sorry

end ceiling_of_square_of_neg_7_over_4_is_4_l213_213171


namespace line_tangent_to_circle_l213_213736

theorem line_tangent_to_circle (k : ℝ) :
  (∀ x y : ℝ, k * x - y - 2 * k + 3 = 0 → x^2 + (y + 1)^2 = 4) → k = 3 / 4 :=
by 
  intro h
  sorry

end line_tangent_to_circle_l213_213736


namespace probability_winning_single_draw_prefer_lottery_or_cashback_most_likely_cash_prize_10_draws_l213_213062
open Probability

-- Definitions from the conditions
def number_white_balls := 6
def number_red_balls := 4
def number_total_balls := number_white_balls + number_red_balls
def probability_draw_red_ball := (number_red_balls : ℝ) / (number_total_balls : ℝ)

def expected_cashback_lottery_draws (n : ℕ) (p : ℝ) := n * p * 100

-- Ⅰ) Prove probability of winning a 100 yuan cash prize when participating in the lottery once
theorem probability_winning_single_draw : probability_draw_red_ball = 2 / 5 :=
by sorry

-- Ⅱ) Customer who spends 1500 yuan should prefer lottery or cashback
theorem prefer_lottery_or_cashback (n : ℕ) (p : ℝ) : expected_cashback_lottery_draws 3 0.4 < 150 :=
by sorry

-- Ⅲ) Most likely amount of cash prize in 10 lottery draws
def binomial_distribution (n : ℕ) (p : ℝ) := λ k : ℕ, Nat.choose n k * (p ^ k) * ((1 - p) ^ (n - k))

def most_likely_cash_prize (n : ℕ) (p : ℝ) : ℕ :=
  let mode_approx := p * n;
  if floor mode_approx * 100 < nat.floor (mode_approx) * 100 then nat.floor mode_approx * 100 else ceiling mode_approx * 100

theorem most_likely_cash_prize_10_draws : most_likely_cash_prize 10 0.4 = 400 :=
by sorry

end probability_winning_single_draw_prefer_lottery_or_cashback_most_likely_cash_prize_10_draws_l213_213062


namespace smaller_cube_side_length_l213_213525

noncomputable def side_length_of_smaller_cube : ℝ := 2 / 3

theorem smaller_cube_side_length (R : ℝ) (d1 d2 : ℝ) :
  ∀ x : ℝ, (R = real.sqrt 3) → 
  d1 = (1 + x) → 
  d2 = (x * real.sqrt 2 / 2) → 
  R * R = (d1 * d1 + d2 * d2) → 
  x = side_length_of_smaller_cube :=
by
  sorry

end smaller_cube_side_length_l213_213525


namespace closest_perfect_square_to_315_l213_213004

theorem closest_perfect_square_to_315 : ∃ n : ℤ, n^2 = 324 ∧
  (∀ m : ℤ, m ≠ n → (abs (315 - m^2) > abs (315 - n^2))) := 
sorry

end closest_perfect_square_to_315_l213_213004


namespace neither_sufficient_nor_necessary_l213_213349

theorem neither_sufficient_nor_necessary (α β : ℝ) :
  (α + β = 90) ↔ ¬((α + β = 90) ↔ (Real.sin α + Real.sin β > 1)) :=
sorry

end neither_sufficient_nor_necessary_l213_213349


namespace Grace_pool_water_capacity_l213_213338

theorem Grace_pool_water_capacity :
  let rate1 := 50 -- gallons per hour of the first hose
  let rate2 := 70 -- gallons per hour of the second hose
  let hours1 := 3 -- hours the first hose was used alone
  let hours2 := 2 -- hours both hoses were used together
  let water1 := rate1 * hours1 -- water from the first hose in the first period
  let water2 := rate2 * hours2 -- water from the second hose in the second period
  let water3 := rate1 * hours2 -- water from the first hose in the second period
  let total_water := water1 + water2 + water3 -- total water in the pool
  total_water = 390 :=
by
  sorry

end Grace_pool_water_capacity_l213_213338


namespace trapezoid_area_l213_213094

theorem trapezoid_area 
  (a b h c : ℝ) 
  (ha : 2 * 0.8 * a + b = c)
  (hb : c = 20) 
  (hc : h = 0.6 * a) 
  (hd : b + 1.6 * a = 20)
  (angle_base : ∃ θ : ℝ, θ = arcsin 0.6)
  : 
  (1 / 2) * (b + c) * h = 72 :=
sorry

end trapezoid_area_l213_213094


namespace kerosene_percentage_l213_213505

def L1_percentage : ℝ := 0.25
def L2_percentage : ℝ := 0.3
def L1_parts : ℝ := 6
def L2_parts : ℝ := 4

def total_parts : ℝ := L1_parts + L2_parts
def total_kerosene : ℝ := L1_percentage * L1_parts + L2_percentage * L2_parts

theorem kerosene_percentage : (total_kerosene / total_parts) * 100 = 27 :=
by
      
s sorry

end kerosene_percentage_l213_213505


namespace car_speed_proof_l213_213480

theorem car_speed_proof : 
  let v := 60
  let north_initial_dist := 300
  let westward_speed := 20
  let time := 5
  let final_distance := 500
  let north_westward_dist := westward_speed * time
  let south_eastward_dist := v * time
  let east_west_distance := north_westward_dist + south_eastward_dist
  (north_initial_dist^2 + east_west_distance^2 = final_distance^2) →
  v = 60 :=
by
  -- equations and conditions used
  have h1 : north_westward_dist = 100 := by sorry
  have h2 : north_initial_dist = 300 := by sorry
  have h3 : east_west_distance = 100 + 5 * v := by sorry
  have h4 : final_distance = 500 := by sorry
  show v = 60 from by sorry

end car_speed_proof_l213_213480


namespace range_of_a_range_of_b_l213_213290

def f (x a b : ℝ) := x^2 + a * x + b
def g (x : ℝ) := 2^x - 2^(2 - x) + 1

-- Part 1
theorem range_of_a (a b : ℝ) (h1 : ∀ t, (∃ x, t = f x a b) → f t a b ≤ 3)
  (h2 : ∃ x, f x a b ≤ 0) (h3 : ∀ x, f x a b ≤ 0 ↔ f (f x a b) a b ≤ 3) :
  2 * Real.sqrt 3 ≤ a ∧ a ≤ 6 :=
sorry

-- Part 2
def h (x a b : ℝ) := if 0 ≤ x ∧ x ≤ 1 then f x a b else sorry -- Symmetry definition
-- Additional properties would need to be encoded for a symmetric function about (1, 1).

theorem range_of_b (a b : ℝ)
  (h1 : h 1 1 = 1) -- Symmetry about (1, 1)
  (h2 : ∀ x ∈ Icc 0 2, ∃ y ∈ Icc 0 2, h x a b = g y) :
  -2 ≤ b ∧ b ≤ 4 :=
sorry

end range_of_a_range_of_b_l213_213290


namespace third_card_is_2_l213_213429

def sequence_cards (a : ℕ) (b : ℕ) : Prop :=
  a = 37 ∧ b = 1 ∧ ∃ c, c = 2 ∧
    (∀ k : ℕ, 2 ≤ k → k < 37 → ∃ m : ℕ, m ∣ (37 + 1 + ∑ i in finset.range (k - 1), i) ∧ m = k)

theorem third_card_is_2 : ∃ a b c, sequence_cards a b ∧ c = 2 :=
sorry

end third_card_is_2_l213_213429


namespace basic_astrophysics_degrees_l213_213039

-- Define the given percentages
def microphotonics_percentage : ℝ := 14
def home_electronics_percentage : ℝ := 24
def food_additives_percentage : ℝ := 10
def gmo_percentage : ℝ := 29
def industrial_lubricants_percentage : ℝ := 8
def total_circle_degrees : ℝ := 360

-- Define a proof problem to show that basic astrophysics research occupies 54 degrees in the circle
theorem basic_astrophysics_degrees :
  total_circle_degrees - (microphotonics_percentage + home_electronics_percentage + food_additives_percentage + gmo_percentage + industrial_lubricants_percentage) = 15 ∧
  0.15 * total_circle_degrees = 54 :=
by
  sorry

end basic_astrophysics_degrees_l213_213039


namespace water_level_lowered_approx_6_02_l213_213892

noncomputable def lower_water_level 
  (length : ℝ) (width : ℝ) (removed_gallons : ℝ) (gallons_per_cubic_foot : ℝ) : ℝ :=
let volume_cubic_feet := removed_gallons / gallons_per_cubic_foot in
let surface_area_square_feet := length * width in
let height_feet := volume_cubic_feet / surface_area_square_feet in
height_feet * 12

theorem water_level_lowered_approx_6_02 
  (length : ℝ) (width : ℝ) (removed_gallons : ℝ) (gallons_per_cubic_foot : ℝ) 
  (h_length : length = 40) (h_width : width = 25) 
  (h_removed_gallons : removed_gallons = 3750) 
  (h_gallons_per_cubic_foot : gallons_per_cubic_foot = 7.48052) : 
  abs (lower_water_level length width removed_gallons gallons_per_cubic_foot - 6.02) < 0.01 :=
by {
  -- Definitions and simplifications will be done here
  sorry
}

end water_level_lowered_approx_6_02_l213_213892


namespace sum_of_three_integers_with_product_of_5_cubed_l213_213882

theorem sum_of_three_integers_with_product_of_5_cubed :
  ∃ (a b c : ℕ), 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ 
  a ≠ b ∧ a ≠ c ∧ b ≠ c ∧ 
  a * b * c = 5^3 ∧ 
  a + b + c = 31 :=
by
  sorry

end sum_of_three_integers_with_product_of_5_cubed_l213_213882


namespace find_BP_l213_213509

-- Define the points and segments
variables (A B C D P : Type) 
variables (AP PC BP PD BD : ℝ)

-- Define the conditions as assumptions
variables (h1 : AP = 5) (h2 : PC = 4) (h3 : BD = 10) 
variables (h4 : 0 < BP) (h5 : PD < BP)

-- Define the condition from Power of a Point theorem
theorem find_BP (h : AP * PC = BP * PD) : BP = 7.24 :=
by
  have h_ap_pc : AP * PC = 20 := by rw [h1, h2]; norm_num
  have h_bd : PD = BD - BP := by norm_num
  have h_equation : BP * (BD - BP) = 20 := by rw [h_ap_pc, h, h3]; norm_num
  sorry

end find_BP_l213_213509


namespace factorial_expression_l213_213109

noncomputable def fac : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * fac n

theorem factorial_expression :
  7 * fac 7 + 5 * fac 5 + 3 * fac 3 + fac 3 = 35904 :=
by sorry

end factorial_expression_l213_213109


namespace find_angle_ACB_l213_213757

-- Define the angles
variables {α β γ δ ω : ℝ}

-- Given conditions
def parallel {A B C D : Type} (AB : ℝ) (DC : ℝ):
  parallel_lines' DC AB :=
  sorry

def angle_DCA : ℝ := 50
def angle_ABC : ℝ := 80
def angle_DAC : ℝ := 30

-- Question: Prove ∠ACB = 30°
theorem find_angle_ACB (h1 : ∀ (A B C D : Type) (AB : ℝ) (DC : ℝ), parallel_lines' DC AB)
    (h2 : angle_DCA = 50)
    (h3 : angle_ABC = 80)
    (h4 : angle_DAC = 30) :
    angle_ACB = 30 :=
begin
  sorry
end

end find_angle_ACB_l213_213757


namespace bananas_added_l213_213894

variable (initial_bananas final_bananas added_bananas : ℕ)

-- Initial condition: There are 2 bananas initially
def initial_bananas_def : Prop := initial_bananas = 2

-- Final condition: There are 9 bananas finally
def final_bananas_def : Prop := final_bananas = 9

-- The number of bananas added to the pile
def added_bananas_def : Prop := final_bananas = initial_bananas + added_bananas

-- Proof statement: Prove that the number of bananas added is 7
theorem bananas_added (h1 : initial_bananas = 2) (h2 : final_bananas = 9) : added_bananas = 7 := by
  sorry

end bananas_added_l213_213894


namespace hyperbrick_hyperbox_probability_l213_213635

theorem hyperbrick_hyperbox_probability :
  let a_nums := {1, 2, 3, ... 500}.to_finset
  let a_sample := a_nums.sample_without_replacement 5
  let b_nums := a_nums \ a_sample
  let b_sample := b_nums.sample_without_replacement 4
  let q := (32 : ℚ) / 126
  let reduced_q := q.num.gcd q.denom
  ((32/126).num / (32/126).denom).num + ((32/126).num / (32/126).denom).denom = 79 :=
sorry

end hyperbrick_hyperbox_probability_l213_213635


namespace tank_destruction_minimum_shots_l213_213021

theorem tank_destruction_minimum_shots : 
  ∀ (grid : Type) (cells : grid → grid → Prop) 
  (tank : grid) (adjacent : grid → grid → Prop) 
  (hit : grid → Prop),
  (∃ (hits_needed : ℕ), ∀ shots : list grid, 
  (length shots ≥ hits_needed) → 
  (∀ t ∈ shots, hit t) → 
  ∃ (final_hit_count : ℕ), final_hit_count ≥ 2) :=
sorry

end tank_destruction_minimum_shots_l213_213021


namespace math_proof_problem_l213_213705

noncomputable def problem_statement : Prop :=
  ∃ (f : ℝ → ℝ) (α β : ℝ), (∀ x : ℝ, f x = |cos x + α * cos 2 * x + β * cos 3 * x|) ∧
                            (∃ M : ℝ, M = (√3) / 2 ∧ M = min (λ α β : ℝ, max (λ x : ℝ, f x)))

theorem math_proof_problem : problem_statement :=
sorry

end math_proof_problem_l213_213705


namespace trapezoid_has_area_approx_74_14_l213_213074

-- Define the properties and conditions of the isosceles trapezoid
def trapezoid_area (longer_base : ℝ) (base_angle : ℝ) : ℝ :=
  let x := 20.0 / 1.8 in
  let y := 0.2 * x in
  let height := 0.6 * x in
  (1 / 2) * (y + longer_base) * height

-- Main statement
theorem trapezoid_has_area_approx_74_14 :
  let longer_base := 20
  let base_angle := Real.arcsin 0.6
  abs (trapezoid_area longer_base base_angle - 74.14) < 0.01 :=
by
  sorry

end trapezoid_has_area_approx_74_14_l213_213074


namespace trapezoid_area_l213_213093

theorem trapezoid_area 
  (a b h c : ℝ) 
  (ha : 2 * 0.8 * a + b = c)
  (hb : c = 20) 
  (hc : h = 0.6 * a) 
  (hd : b + 1.6 * a = 20)
  (angle_base : ∃ θ : ℝ, θ = arcsin 0.6)
  : 
  (1 / 2) * (b + c) * h = 72 :=
sorry

end trapezoid_area_l213_213093


namespace seq_20_l213_213385

noncomputable def seq (n : ℕ) : ℝ := 
  if n = 0 then 0
  else if n = 1 then 1
  else if n = 2 then 1/2
  else sorry -- The actual function definition based on the recurrence relation is omitted for brevity

lemma seq_recurrence (n : ℕ) (hn : n ≥ 1) :
  2 / seq (n + 1) = (seq n + seq (n + 2)) / (seq n * seq (n + 2)) :=
sorry

theorem seq_20 : seq 20 = 1/20 :=
sorry

end seq_20_l213_213385


namespace apple_juice_percentage_in_blend_l213_213816

theorem apple_juice_percentage_in_blend :
  ∀ (a_total o_total : ℕ) (a_juice o_juice n : ℝ),
    a_total = 15 →
    o_total = 15 →
    a_juice = 2.5 →
    o_juice = 10 / 3 →
    n = 6 →
    let total_apple_juice := n * a_juice in
    let total_orange_juice := n * o_juice in
    let total_juice := total_apple_juice + total_orange_juice in
    (total_apple_juice / total_juice) * 100 ≈ 43 := by
  sorry

end apple_juice_percentage_in_blend_l213_213816


namespace distance_between_alex_and_bella_l213_213796

theorem distance_between_alex_and_bella :
  let Alex := (3 : ℂ) + (4 : ℂ) * complex.I
  let Bella := (1 : ℂ) - (1 : ℂ) * complex.I
  complex.abs (Alex - Bella) = real.sqrt 29 := by
  sorry

end distance_between_alex_and_bella_l213_213796


namespace ceil_square_neg_fraction_l213_213177

theorem ceil_square_neg_fraction :
  let x := (-7) / 4
  let y := x^2
  let z := Real.ceil y
  z = 4 := 
by
  sorry

end ceil_square_neg_fraction_l213_213177


namespace joey_return_speed_l213_213764

theorem joey_return_speed
    (h1: 1 = (2 : ℝ) / u)
    (h2: (4 : ℝ) / (1 + t) = 3)
    (h3: u = 2)
    (h4: t = 1 / 3) :
    (2 : ℝ) / t = 6 :=
by
  sorry

end joey_return_speed_l213_213764


namespace ceil_square_of_neg_seven_fourths_l213_213235

/-- Evaluate the ceiling of the square of -7/4 --/
theorem ceil_square_of_neg_seven_fourths : (Int.ceil ((-7/4 : ℚ)^2 : ℚ) = 4) :=
sorry

end ceil_square_of_neg_seven_fourths_l213_213235


namespace countThreeDigitNumbersWithPerfectCubeDigitSums_l213_213347

def isThreeDigitNumber (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

def digitSum (n : ℕ) : ℕ := 
  (n / 100) + ((n % 100) / 10) + (n % 10)

def isPerfectCube (n : ℕ) : Prop :=
  ∃ (k : ℕ), k*k*k = n

theorem countThreeDigitNumbersWithPerfectCubeDigitSums : 
  (Finset.filter (λ n, isPerfectCube (digitSum n)) (Finset.range' 100 900)).card = 10 := 
  sorry

end countThreeDigitNumbersWithPerfectCubeDigitSums_l213_213347


namespace radius_of_circle_l213_213375

variable {O : Type*} [MetricSpace O]

def distance_near : ℝ := 1
def distance_far : ℝ := 7
def diameter : ℝ := distance_near + distance_far

theorem radius_of_circle (P : O) (r : ℝ) (h1 : distance_near = 1) (h2 : distance_far = 7) :
  r = diameter / 2 :=
by
  -- Proof would go here 
  sorry

end radius_of_circle_l213_213375


namespace jacks_walking_rate_l213_213018

def jackDistance : ℝ := 4
def jackTimeHours : ℝ := 1 + (15 / 60)

theorem jacks_walking_rate :
  (jackDistance / jackTimeHours) = 3.2 := by
  -- proof skipped
  sorry

end jacks_walking_rate_l213_213018


namespace perpendicular_bisector_AC_circumcircle_eqn_l213_213694

/-- Given vertices of triangle ABC, prove the equation of the perpendicular bisector of side AC --/
theorem perpendicular_bisector_AC (A B C D : ℝ×ℝ) (hA: A = (0, 2)) (hC: C = (4, 0)) (hD: D = (2, 1)) :
  ∃ k b, (k = 2) ∧ (b = -3) ∧ (∀ x y, y = k * x + b ↔ 2 * x - y - 3 = 0) :=
sorry

/-- Given vertices of triangle ABC, prove the equation of the circumcircle --/
theorem circumcircle_eqn (A B C D E F : ℝ×ℝ) (hA: A = (0, 2)) (hB: B = (6, 4)) (hC: C = (4, 0)) :
  ∃ k, k = 10 ∧ 
  (∀ x y, (x - 3) ^ 2 + (y - 3) ^ 2 = k ↔ x ^ 2 + y ^ 2 - 6 * x - 2 * y + 8 = 0) :=
sorry

end perpendicular_bisector_AC_circumcircle_eqn_l213_213694


namespace solve_x_l213_213027

theorem solve_x
  (x : ℝ)
  (h : 3500 - (1000 / x) = 3451.2195121951218) :
  x ≈ 20.5 := 
by sorry

end solve_x_l213_213027


namespace find_coordinates_of_point_C_l213_213673

-- Define points A and B
def A : ℝ × ℝ := (-1, -1)
def B : ℝ × ℝ := (2, 5)

-- Define the coordinates of point C
def is_point_on_line_AB_with_relationship (x y : ℝ) : Prop :=
  let C : ℝ × ℝ := (x, y) in
  let AC := (C.1 - A.1, C.2 - A.2) in
  let CB := (B.1 - C.1, B.2 - C.2) in
  AC = (5 * CB.1, 5 * CB.2)

-- The theorem to prove
theorem find_coordinates_of_point_C : 
  ∃ (x y : ℝ), is_point_on_line_AB_with_relationship x y ∧ x = 3 / 2 ∧ y = 4 := 
by
  sorry

end find_coordinates_of_point_C_l213_213673


namespace star_expression_l213_213256

def star : ℕ → ℕ := sorry

axiom star_base : star 1 = 2

axiom star_recurrence : ∀ n : ℕ, star (n+1) = star n + 2^(n+1)

theorem star_expression (n : ℕ) : star n = 2^(n+1) - 2 :=
by sorry

end star_expression_l213_213256


namespace condition_A_neq_condition_B_l213_213579

variable {θ a : ℝ}

theorem condition_A_neq_condition_B
  (hA : sqrt (1 + sin θ) = a)
  (hB : sin (θ / 2) + cos (θ / 2) = a) :
  ∀ (θ a : ℝ), 
  hA → ¬ (a < 0)
  ∧ hB ∧ a ∈ set.Icc (-sqrt 2) sqrt 2 :=
by sorry

end condition_A_neq_condition_B_l213_213579


namespace max_not_greatest_l213_213319

noncomputable def f (x : ℝ) : ℝ := (2 * x - x^2) * exp x

theorem max_not_greatest : 
  f (real.sqrt 2) = (2 * real.sqrt 2 - 2) * exp (real.sqrt 2) ∧ 
  (∀ y, f y ≤ f (real.sqrt 2)) ∧ 
  (∃ z, z ≠ real.sqrt 2 ∧ f z = f (real.sqrt 2)) :=
begin
  sorry
end

end max_not_greatest_l213_213319


namespace curves_intersect_exactly_three_points_l213_213494

theorem curves_intersect_exactly_three_points (a : ℝ) :
  (∃! (p : ℝ × ℝ), p.1 ^ 2 + p.2 ^ 2 = a ^ 2 ∧ p.2 = p.1 ^ 2 - a) ↔ a > (1 / 2) :=
by sorry

end curves_intersect_exactly_three_points_l213_213494


namespace correct_propositions_l213_213141

section

variable (x : ℝ) (a b : ℝ) (m : ℤ)

-- Proposition 1: Negation understanding
def P1 : Prop := ¬ ∃ x_0 : ℝ, x_0^2 + 1 > 3 * x_0 = ∀ x : ℝ, x + 1 < 3 * x

-- Proposition 2: Complex number handling
def P2 : Prop := 
  let z := complex.abs (1 - complex.I) * complex.I ^ 2017
  ∃ (part_im : ℝ), part_im =  sqrt 2 ∧ complex.im (complex.conj z) = part_im

-- Proposition 3: Necessary and sufficient condition
def P3 : Prop := 
  let p := a < b
  let q := 1 / b < 1 / a ∧ 1 / a < 0
  p ∧ ¬ (p → q)

-- Proposition 4: Evaluation of power function
def even (m : ℤ) : Prop := ∃ k : ℤ, m = 2 * k
def P4 : Prop := (even m) ∧ ((m^2 - 3*m + 3) * (2:ℝ)^m = 4)

-- Now, let's state the main theorem
theorem correct_propositions : (prop_count : ℕ), prop_count = count_correct_propositions (P1, P2, P3, P4) := 
by 
  sorry

end

end correct_propositions_l213_213141


namespace initial_speeds_l213_213267

noncomputable def father_speed : ℝ :=
  classical.some $ exists_solution (λ x, 
    let daughter_speed := 2 * x in
    let meeting_distance_father := 20 in
    let meeting_distance_daughter := 60 - meeting_distance_father in
    let father_speed_post_meeting := x + 2 in
    let daughter_speed_post_meeting := 2 * x + 2 in
    let time_difference := (40 / daughter_speed_post_meeting) - (20 / father_speed_post_meeting) in
    time_difference = 1 / 12)

theorem initial_speeds :
  let father_initial_speed := father_speed in
  let daughter_initial_speed := 2 * father_initial_speed in
  father_initial_speed = 14 ∧ daughter_initial_speed = 28 := sorry

end initial_speeds_l213_213267


namespace parallelogram_side_equality_l213_213746

theorem parallelogram_side_equality (x y : ℚ) : 
  (3 * x + 4 = 9) ∧ (6 * y - 2 = 12) → x + y = 4 :=
by 
  intro h,
  cases h with h1 h2,
  sorry

end parallelogram_side_equality_l213_213746


namespace sum_primes_no_solution_congruence_l213_213588

theorem sum_primes_no_solution_congruence :
  ∑ p in {p | Nat.Prime p ∧ ¬ (∃ x : ℤ, 5 * (10 * x + 2) ≡ 3 [ZMOD p])}, p = 7 :=
sorry

end sum_primes_no_solution_congruence_l213_213588


namespace find_difference_l213_213271

def f : ℕ → ℤ
def g : ℕ → ℤ

axiom f_initial : f 1 = 4
axiom g_initial : g 1 = 9
axiom f_recurrence : ∀ n, n ≥ 1 → f (n + 1) = 2 * f n + 3 * g n + 2 * n
axiom g_recurrence : ∀ n, n ≥ 1 → g (n + 1) = 2 * g n + 3 * f n + 5

noncomputable def d (n : ℕ) : ℕ := f n - g n

theorem find_difference : f 2005 - g 2005 = 1004 :=
by
  sorry

end find_difference_l213_213271


namespace angle_XYZ_of_hexagon_and_equilateral_triangle_l213_213744

theorem angle_XYZ_of_hexagon_and_equilateral_triangle 
    (internal_angle_hexagon : ∀ (X Y : ℝ), angle X Y (next_vertex X Y) = 120)
    (angle_equilateral_triangle : ∀ (A B C : ℝ), angle A B C = 60)
    (XY_eq_YZ : ∀ (X Y Z : ℝ), distance X Y = distance Y Z)
    (isosceles_XYZ : ∀ (X Y Z : ℝ), isosceles_triangle X Y Z)
    : ∀ (X Y Z : ℝ), angle X Y Z = 60 := 
by
  intros
  sorry

end angle_XYZ_of_hexagon_and_equilateral_triangle_l213_213744


namespace trapezoid_perimeter_l213_213974

structure Trapezoid (Point : Type) :=
  (EF GH EG FH : ℝ)
  (hEF : EF = 90)
  (hGH : GH = 40)
  (hEG : EG = 53)
  (hFH : FH = 45)

def perimeter (t : Trapezoid) : ℝ :=
  t.EF + t.GH + t.EG + t.FH

theorem trapezoid_perimeter : ∀ (t : Trapezoid), t.EF = 90 → t.GH = 40 → t.EG = 53 → t.FH = 45 → 
  perimeter t = 228 :=
by
  intros
  simp [perimeter, *]
  sorry

end trapezoid_perimeter_l213_213974


namespace train_length_is_240_l213_213973

-- Define the given parameters
def v_train : ℝ := 60 -- speed of train in kmph
def v_man : ℝ := 6 -- speed of man in kmph
def t : ℝ := 13.090909090909092 -- time in seconds

-- Conversion factor from kmph to m/s
def kmph_to_mps (v : ℝ) : ℝ := v * (5 / 18)

-- Relative speed in m/s
def relative_speed : ℝ := kmph_to_mps (v_train + v_man)

def length_of_train : ℝ := relative_speed * t

-- The theorem to be proven
theorem train_length_is_240 : length_of_train = 240 :=
by
  sorry

end train_length_is_240_l213_213973


namespace find_triples_l213_213996

theorem find_triples :
  { (a, b, c) : ℕ × ℕ × ℕ | (c-1) * (a * b - b - a) = a + b - 2 } =
  { (2, 1, 0), (1, 2, 0), (3, 4, 2), (4, 3, 2), (1, 0, 2), (0, 1, 2), (2, 4, 3), (4, 2, 3) } :=
by
  sorry

end find_triples_l213_213996


namespace expected_teachers_with_masters_degree_l213_213425

theorem expected_teachers_with_masters_degree
  (prob: ℚ) (teachers: ℕ) (h_prob: prob = 1/4) (h_teachers: teachers = 320) :
  prob * teachers = 80 :=
by
  sorry

end expected_teachers_with_masters_degree_l213_213425


namespace trapezoid_has_area_approx_74_14_l213_213073

-- Define the properties and conditions of the isosceles trapezoid
def trapezoid_area (longer_base : ℝ) (base_angle : ℝ) : ℝ :=
  let x := 20.0 / 1.8 in
  let y := 0.2 * x in
  let height := 0.6 * x in
  (1 / 2) * (y + longer_base) * height

-- Main statement
theorem trapezoid_has_area_approx_74_14 :
  let longer_base := 20
  let base_angle := Real.arcsin 0.6
  abs (trapezoid_area longer_base base_angle - 74.14) < 0.01 :=
by
  sorry

end trapezoid_has_area_approx_74_14_l213_213073


namespace correct_matrix_l213_213628

open Matrix

noncomputable def cross_product_matrix : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![0, 8, 2], 
    ![-8, 0, -7], 
    ![-2, 7, 0]]

def given_vector : Fin 3 → ℝ :=
  ![7, 2, -8]

theorem correct_matrix :
  ∀ v : Fin 3 → ℝ, 
    cross_product_matrix.mulVec v = !![!(2 * v 2 + 8 * v 1), 
                                       8 * v 0 - 7 * v 2, 
                                       7 * v 1 - 2 * v 0] :=
by
  intros v
  sorry

end correct_matrix_l213_213628


namespace a_n_correct_b_n_correct_l213_213508

noncomputable def f1 (x : ℝ) : ℝ := 4 * (x - x^2)

noncomputable def fn (n : ℕ) (x : ℝ) : ℝ :=
  match n with
  | 0     => f1 x
  | n + 1 => fn n (f1 x)

def a_n (n : ℕ) : ℕ := 2 ^ (n - 1)
def b_n (n : ℕ) : ℕ := 2 ^ (n - 1) + 1

theorem a_n_correct (n : ℕ) : 
  ∀ x ∈ Icc 0 1, (fn n x = fn n (1/2)) → x = (1/2) := sorry

theorem b_n_correct (n : ℕ) : 
  ∀ x ∈ Icc 0 1, (fn n x = 0) → (x = 0 ∨ x = 1) := sorry

end a_n_correct_b_n_correct_l213_213508


namespace no_bijection_exists_l213_213960

def is_bijection {X Y : Type} (f : X → Y) : Prop :=
  (∀ y : Y, ∃! x : X, f x = y) ∧ (∀ x₁ x₂ : X, f x₁ = f x₂ → x₁ = x₂)

def nat_pos := { n : ℕ // n > 0 }
def nat_nonneg := { n : ℕ // n ≥ 0 }

theorem no_bijection_exists (f : nat_pos → nat_nonneg):
  is_bijection f →
  ¬ (∀ (m n : nat_pos), f ⟨m.1 * n.1, mul_pos m.2 n.2⟩ = ⟨f m.1 + f n.1 + 3 * f m.1 * f n.1, _⟩) := 
sorry

end no_bijection_exists_l213_213960


namespace total_pages_proof_l213_213393

/-
Conditions:
1. Johnny's essay has 150 words.
2. Madeline's essay is double the length of Johnny's essay.
3. Timothy's essay has 30 more words than Madeline's essay.
4. One page contains 260 words.

Question:
Prove that the total number of pages do Johnny, Madeline, and Timothy's essays fill is 5.
-/

def johnny_words : ℕ := 150
def words_per_page : ℕ := 260

def madeline_words : ℕ := 2 * johnny_words
def timothy_words : ℕ := madeline_words + 30

def pages (words : ℕ) : ℕ := (words + words_per_page - 1) / words_per_page  -- division rounding up

def johnny_pages : ℕ := pages johnny_words
def madeline_pages : ℕ := pages madeline_words
def timothy_pages : ℕ := pages timothy_words

def total_pages : ℕ := johnny_pages + madeline_pages + timothy_pages

theorem total_pages_proof : total_pages = 5 :=
by sorry

end total_pages_proof_l213_213393


namespace goods_train_speed_l213_213532

/-- Define all given conditions -/
def train_speed := 55 -- speed of the man's train in km/h
def goods_train_length := 0.32 -- length of the goods train in km
def passing_time := 10 / 3600 -- passing time in hours

/-- Calculate the relative speed -/
def relative_speed := goods_train_length / passing_time

/-- Define the proof problem statement -/
theorem goods_train_speed :
  let V_r := relative_speed in
  let V_g := V_r - train_speed in
  V_g = 60.2 :=
by
  sorry

end goods_train_speed_l213_213532


namespace unique_point_graph_eq_l213_213145

theorem unique_point_graph_eq (c : ℝ) : 
  (∀ x y : ℝ, 3 * x^2 + y^2 + 6 * x - 12 * y + c = 0 → x = -1 ∧ y = 6) ↔ c = 39 :=
sorry

end unique_point_graph_eq_l213_213145


namespace pete_miles_walked_l213_213840

-- Define the conditions
def maxSteps := 99999
def numFlips := 50
def finalReading := 25000
def stepsPerMile := 1500

-- Proof statement that Pete walked 3350 miles
theorem pete_miles_walked : 
  (numFlips * (maxSteps + 1) + finalReading) / stepsPerMile = 3350 := 
by 
  sorry

end pete_miles_walked_l213_213840


namespace ceil_of_neg_frac_squared_l213_213203

-- Define the negated fraction
def neg_frac : ℚ := -7 / 4

-- Define the squared value of the negated fraction
def squared_value : ℚ := neg_frac ^ 2

-- Define the ceiling function applied to the squared value
def ceil_squared_value : ℤ := Int.ceil squared_value

-- Prove that the ceiling of the squared value is 4
theorem ceil_of_neg_frac_squared : ceil_squared_value = 4 := 
by sorry

end ceil_of_neg_frac_squared_l213_213203


namespace ceil_square_eq_four_l213_213153

theorem ceil_square_eq_four : (⌈(-7 / 4: ℚ)^2⌉ : ℤ) = 4 := by
  sorry

end ceil_square_eq_four_l213_213153


namespace probability_same_number_l213_213836

theorem probability_same_number (n k : ℕ) (h₁ : n = 8) (h₂ : k = 6) : 
  (∃ m : ℝ, 0 ≤ m ∧ m ≤ 1 ∧ m = 1) := by
  sorry

end probability_same_number_l213_213836


namespace countThreeDigitNumbersWithPerfectCubeDigitSums_l213_213348

def isThreeDigitNumber (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

def digitSum (n : ℕ) : ℕ := 
  (n / 100) + ((n % 100) / 10) + (n % 10)

def isPerfectCube (n : ℕ) : Prop :=
  ∃ (k : ℕ), k*k*k = n

theorem countThreeDigitNumbersWithPerfectCubeDigitSums : 
  (Finset.filter (λ n, isPerfectCube (digitSum n)) (Finset.range' 100 900)).card = 10 := 
  sorry

end countThreeDigitNumbersWithPerfectCubeDigitSums_l213_213348


namespace total_students_in_class_l213_213897

-- Define the variables and conditions
variables {S : ℝ}

-- Given conditions
def received_A (S : ℝ) := 0.25 * S
def received_B_or_C (S : ℝ) := 0.1875 * S
def failed (S : ℝ) := 18
def remaining_students (S : ℝ) := 0.75 * S

theorem total_students_in_class : S = 32 :=
  have h : remaining_students S = received_B_or_C S + failed S, from sorry,
  -- Provided the given equation matches the conditions, we solve for S
  have eq1: 0.75 * S = 0.1875 * S + 18, from sorry,
  -- Solving the equation above will result in S = 32
  calc
    S = 32 : sorry

end total_students_in_class_l213_213897


namespace value_of_f_at_2_l213_213003

def f (x : ℝ) : ℝ := x^2 - 2 * x + 3

theorem value_of_f_at_2 : f 2 = 3 := sorry

end value_of_f_at_2_l213_213003


namespace integer_modulo_solution_l213_213243

theorem integer_modulo_solution (a : ℤ) : 
  (5 ∣ a^3 + 3 * a + 1) ↔ (a % 5 = 1 ∨ a % 5 = 2) := 
by
  exact sorry

end integer_modulo_solution_l213_213243


namespace largest_divisor_8_l213_213793

theorem largest_divisor_8 (p q : ℤ) (hp : p % 2 = 1) (hq : q % 2 = 1) (h : q < p) : 
  8 ∣ (p^2 - q^2 + 2*p - 2*q) := 
sorry

end largest_divisor_8_l213_213793


namespace chord_segments_equal_l213_213434

theorem chord_segments_equal
    (A B C : Point)
    (circle: Circle)
    (H₁ : A ∈ circle)
    (H₂ : B ∈ circle)
    (H₃ : C ∈ circle)
    (M N : Point)
    (H₄ : midpoint_arc A B circle M)
    (H₅ : midpoint_arc A C circle N)
    (P Q : Point)
    (H₆ : intersect_segment_chord M N A B P)
    (H₇ : intersect_segment_chord M N A C Q)
    (H₈ : ∠APQ = ∠AQP) :
  dist A P = dist A Q := 
begin
  sorry
end

end chord_segments_equal_l213_213434


namespace find_a_l213_213681

variables (a : ℝ)
def z : ℂ := (a - complex.i) / (3 + complex.i)

theorem find_a (h : z.re = 1 / 2) : a = 2 :=
by 
  sorry

end find_a_l213_213681


namespace one_appears_iff_not_divisible_by_5_l213_213647

def sequence (k : ℕ) : ℕ → ℕ 
| 1 => k
| (n + 1) => if sequence n % 2 = 0 then sequence n / 2 else sequence n + 5

theorem one_appears_iff_not_divisible_by_5 (k : ℕ) (hk : 0 < k) : 
  (∃ n, sequence k n = 1) ↔ ¬ (5 ∣ k) :=
sorry

end one_appears_iff_not_divisible_by_5_l213_213647


namespace minimum_area_two_equilateral_triangles_l213_213514

noncomputable def minimum_sum_of_areas (L : ℝ) : ℝ :=
  let x := L / 2 in
  let area (s : ℝ) : ℝ := (sqrt 3 / 4) * (s / 3)^2 in
  2 * area x

theorem minimum_area_two_equilateral_triangles : 
  minimum_sum_of_areas 12 = 2 * sqrt 3 :=
by
  sorry

end minimum_area_two_equilateral_triangles_l213_213514


namespace det_power_matrix_l213_213722

theorem det_power_matrix (M : Matrix ℕ ℕ ℝ) (h_det: det M = 3) : det (M^3) = 27 :=
  sorry

end det_power_matrix_l213_213722


namespace trigonometric_inequalities_l213_213785

theorem trigonometric_inequalities :
  let a := (real.sqrt 2 / 2) * (real.sin (real.pi * 17 / 180) + real.cos (real.pi * 17 / 180))
  let b := 2 * (real.cos (real.pi * 13 / 180))^2 - 1
  let c := (real.sqrt 3 / 2)
  in c < a ∧ a < b :=
by {
  let a := (real.sqrt 2 / 2) * (real.sin (real.pi * 17 / 180) + real.cos (real.pi * 17 / 180)),
  let b := 2 * (real.cos (real.pi * 13 / 180))^2 - 1,
  let c := real.sqrt 3 / 2,
  sorry
}

end trigonometric_inequalities_l213_213785


namespace area_of_triangle_ABC_l213_213487

theorem area_of_triangle_ABC :
  ∀ (A B C L : ℝ × ℝ),
  let AC := 15
  let BL := 9
  let BC := 17
  ∃ (AL : ℝ), 
  (BC = BL + 17 - BL) ∧ (ACLis_right_triangle AL AC BC) →
  area_of_triangle_ABC = (17 * Real.sqrt 161) / 2 := 
sorry

end area_of_triangle_ABC_l213_213487


namespace part1_part2_l213_213700

noncomputable def f (x m : ℝ) : ℝ := x^2 - 2*m*x + 2 - m

theorem part1 (m : ℝ) : (∀ x : ℝ, f x m ≥ x - m*x) → -7 ≤ m ∧ m ≤ 1 :=
by
  sorry

theorem part2 (m : ℝ) : (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → 0 ≤ f x m) → m ≤ 1 :=
by
  sorry

end part1_part2_l213_213700


namespace greenfield_basketball_l213_213752

theorem greenfield_basketball :
  let socks_cost := 6
  let t_shirt_cost := socks_cost + 7
  let total_cost_home_game := 2 * socks_cost + t_shirt_cost
  let total_cost_away_game := socks_cost + t_shirt_cost
  let total_cost_per_player := total_cost_home_game + total_cost_away_game
  let total_equipment_cost := 3100
  total_cost_per_player * 72 = total_equipment_cost :=
by
  let socks_cost := 6
  let t_shirt_cost := socks_cost + 7
  let total_cost_home_game := 2 * socks_cost + t_shirt_cost
  let total_cost_away_game := socks_cost + t_shirt_cost
  let total_cost_per_player := total_cost_home_game + total_cost_away_game
  let total_equipment_cost := 3100
  show total_cost_per_player * 72 = total_equipment_cost from sorry

end greenfield_basketball_l213_213752


namespace alberto_biked_more_l213_213106

theorem alberto_biked_more (dist_bjorn dist_alberto : ℕ) (h_bjorn : dist_bjorn = 75) (h_alberto : dist_alberto = 105) : dist_alberto - dist_bjorn = 30 :=
by
  -- We simply apply the information given in the hypotheses.
  rw [h_bjorn, h_alberto]
  -- Calculate the difference.
  exact rfl

end alberto_biked_more_l213_213106


namespace initial_quantity_of_A_l213_213962

-- Definitions based on the conditions.
variables (A B C : ℕ) (x : ℕ)
def initial_state := (4 * x, x, 3 * x)
def after_removal := (4 * x, x + 60, 3 * x - 60)
def ratio_condition (A B C : ℕ) : Prop := A / C = 2 / 1 ∧ B / C = 5 / 1

-- The main statement to prove.
theorem initial_quantity_of_A (init_A : ℕ) (init_B : ℕ) (init_C : ℕ) :
  initial_state A B C ->
  ratio_condition A (B + 60) (C - 60) ->
  init_A = 240 :=
by
  sorry

end initial_quantity_of_A_l213_213962


namespace ceil_square_of_neg_fraction_l213_213184

theorem ceil_square_of_neg_fraction : 
  (Int.ceil ((-7 / 4 : ℚ)^2 : ℚ)).toNat = 4 := by
  sorry

end ceil_square_of_neg_fraction_l213_213184


namespace ceil_of_neg_frac_squared_l213_213196

-- Define the negated fraction
def neg_frac : ℚ := -7 / 4

-- Define the squared value of the negated fraction
def squared_value : ℚ := neg_frac ^ 2

-- Define the ceiling function applied to the squared value
def ceil_squared_value : ℤ := Int.ceil squared_value

-- Prove that the ceiling of the squared value is 4
theorem ceil_of_neg_frac_squared : ceil_squared_value = 4 := 
by sorry

end ceil_of_neg_frac_squared_l213_213196


namespace sector_radius_l213_213540

open Real

-- Define given conditions and statement
theorem sector_radius (θ : ℝ) (hθ : 0 < θ ∧ θ < π / 2) :
  let r := 9 in 
  let R := r / cos θ in 
  R = 4.5 * sec θ := 
by
  -- Proof goes here.
  sorry

end sector_radius_l213_213540


namespace ceiling_of_square_frac_l213_213223

theorem ceiling_of_square_frac : 
  let x : ℚ := -7 / 4
  let y : ℚ := x^2
  in Int.ceil y = 4 := 
by 
  let x : ℚ := -7 / 4
  let y : ℚ := x^2
  exact sorry

end ceiling_of_square_frac_l213_213223


namespace max_value_of_function_l213_213879

noncomputable def f (x : ℝ) : ℝ := x^3 - 3 * x^2 - 9 * x + 6

theorem max_value_of_function : 
  ∃ (x : ℝ), x ∈ set.Icc (-4) 4 ∧ f x = 11 ∧ ∀ (y : ℝ), y ∈ set.Icc (-4) 4 → f y ≤ 11 :=
by 
  sorry

end max_value_of_function_l213_213879


namespace solve_fraction_eq_l213_213450

theorem solve_fraction_eq (x : ℝ) (h : (x + 6) / (x - 3) = 4) : x = 6 :=
sorry

end solve_fraction_eq_l213_213450


namespace valid_call_time_at_15_l213_213889

def time_difference := 5 -- Beijing is 5 hours ahead of Moscow

def beijing_start_time := 14 -- Start time in Beijing corresponding to 9:00 in Moscow
def beijing_end_time := 17  -- End time in Beijing corresponding to 17:00 in Beijing

-- Define the call time in Beijing
def call_time_beijing := 15

-- The time window during which they can start the call in Beijing
def valid_call_time (t : ℕ) : Prop :=
  beijing_start_time <= t ∧ t <= beijing_end_time

-- The theorem to prove that 15:00 is a valid call time in Beijing
theorem valid_call_time_at_15 : valid_call_time call_time_beijing :=
by
  sorry

end valid_call_time_at_15_l213_213889


namespace ceiling_of_square_of_neg_7_over_4_is_4_l213_213172

theorem ceiling_of_square_of_neg_7_over_4_is_4 : 
  Real.ceil ((-7 / 4 : Real) ^ 2) = 4 := by
  sorry

end ceiling_of_square_of_neg_7_over_4_is_4_l213_213172


namespace trapezoid_area_l213_213084

-- Define the problem statement
theorem trapezoid_area 
  (a b h: ℝ)
  (b₁ b₂: ℝ)
  (θ: ℝ) 
  (h₃: θ = Real.arcsin 0.6)
  (h₄: a = 20)
  (h₅: b = a - 2 * b₁ * Real.sin θ) 
  (h₆: h = b₁ * Real.cos θ) 
  (h₇: θ = Real.arcsin (3/5)) 
  (circum: isosceles_trapezoid_circumscribed a b₁ b₂) :
  ((1 / 2) * (a + b₂) * h = 2000 / 27) :=
by sorry

end trapezoid_area_l213_213084


namespace max_M_eq_8_l213_213790

def A := Fin 17

/-- Define the function f and its iterations -/
def f : A → A := λ x, (3 * x - 2) % 17

def f_iter (k : ℕ) (x : A) : A :=
  Nat.iterate k f x

def maximum_M (M : ℕ) : Prop :=
  (∀ m, 1 ≤ m → m < M → ∀ i, 1 ≤ i → i ≤ 16 →
    (f_iter m i.succ - f_iter m i) % 17 ≠ 1 ∧ (f_iter m i.succ - f_iter m i) % 17 ≠ 16) ∧
  (1 ≤ i → i ≤ 16 →
    ((f_iter M i.succ - f_iter M i) % 17 = 1 ∨ (f_iter M i.succ - f_iter M i) % 17 = 16) ∧
    ((f_iter M 1 - f_iter M 17) % 17 = 1 ∨ (f_iter M 1 - f_iter M 17) % 17 = 16))

theorem max_M_eq_8 : maximum_M 8 :=
sorry

end max_M_eq_8_l213_213790


namespace marta_total_spent_l213_213416

theorem marta_total_spent :
  let sale_book_cost := 5 * 10
  let online_book_cost := 40
  let bookstore_book_cost := 3 * online_book_cost
  let total_spent := sale_book_cost + online_book_cost + bookstore_book_cost
  total_spent = 210 := sorry

end marta_total_spent_l213_213416


namespace ceil_square_of_neg_fraction_l213_213194

theorem ceil_square_of_neg_fraction : 
  (Int.ceil ((-7 / 4 : ℚ)^2 : ℚ)).toNat = 4 := by
  sorry

end ceil_square_of_neg_fraction_l213_213194


namespace sum_largest_smallest_ABC_l213_213383

def hundreds (n : ℕ) : ℕ := n / 100
def units (n : ℕ) : ℕ := n % 10
def tens (n : ℕ) : ℕ := (n / 10) % 10

theorem sum_largest_smallest_ABC : 
  (hundreds 297 = 2) ∧ (units 297 = 7) ∧ (hundreds 207 = 2) ∧ (units 207 = 7) →
  (297 + 207 = 504) :=
by
  sorry

end sum_largest_smallest_ABC_l213_213383


namespace ceil_square_of_neg_seven_fourths_l213_213228

/-- Evaluate the ceiling of the square of -7/4 --/
theorem ceil_square_of_neg_seven_fourths : (Int.ceil ((-7/4 : ℚ)^2 : ℚ) = 4) :=
sorry

end ceil_square_of_neg_seven_fourths_l213_213228


namespace max_attempts_to_find_QQ_number_l213_213543

theorem max_attempts_to_find_QQ_number :
  (∃ l : List ℕ, l.length = 6 ∧
                l.count 1 = 1 ∧
                l.count 2 = 1 ∧
                l.count 5 = 2 ∧
                l.count 8 = 2 ∧
                (multiset.card (multiset.list l).permutations = 180)) :=
sorry

end max_attempts_to_find_QQ_number_l213_213543


namespace at_least_two_dice_same_number_probability_l213_213833

theorem at_least_two_dice_same_number_probability :
  let total_outcomes := 6^8
  let favorable_outcomes := 28 * 6! * 6^2
  let probability_all_different := favorable_outcomes / total_outcomes
  let required_probability := 1 - probability_all_different
  required_probability = 191 / 336
:= by
  sorry

end at_least_two_dice_same_number_probability_l213_213833


namespace radius_of_table_l213_213053

theorem radius_of_table (r1 r2 r3 : ℝ) (r_table : ℝ) 
  (h1 : r1 = 2) (h2 : r2 = 3) (h3 : r3 = 10)
  (h4 : r_table = 15) :
  ∃ (C A B : point),
  dist A C = r1 + r3 ∧
  dist B C = r2 + r3 ∧
  dist A B = r1 + r2 ∧
  right_triangle (triangle A B C) ∧
  ∀ (O : point), center_of_table O r_table (doilies C A B r1 r2 r3) := sorry

end radius_of_table_l213_213053


namespace constant_term_binomial_expansion_l213_213625

theorem constant_term_binomial_expansion : ∃ T, (∀ x : ℝ, T = (2 * x - 1 / (2 * x)) ^ 6) ∧ T = -20 := 
by
  sorry

end constant_term_binomial_expansion_l213_213625


namespace area_of_triangle_min_value_of_a_l213_213368

variables {A B C a b c : ℝ}
variables (α β γ : ℝ) -- angles

-- Definitions and assumptions from the conditions
def condition_1 : Prop := sqrt(3) * c * cos A = a * sin C
def condition_2 : Prop := 4 * sin C = c^2 * sin B
def condition_3 : Prop := (∥AB∥ * ∥AC∥) * cos 60 = 4

-- 1. Prove the area of triangle ABC is sqrt(3)
theorem area_of_triangle (h1 : condition_1) (h2 : condition_2) : 
  (1 / 2) * b * c * sin A = sqrt(3) :=
sorry

-- 2. Prove the minimum value of a is 2sqrt(2)
theorem min_value_of_a (h1 : condition_1) (h3 : condition_3) : 
  a ≥ 2 * sqrt(2) :=
sorry

end area_of_triangle_min_value_of_a_l213_213368


namespace ceil_square_neg_seven_over_four_l213_213214

theorem ceil_square_neg_seven_over_four : 
  let x := - (7 / 4 : ℚ) in
  ⌈x^2⌉ = 4 :=
by
  let x := - (7 / 4 : ℚ)
  sorry

end ceil_square_neg_seven_over_four_l213_213214


namespace compute_expression_l213_213119

theorem compute_expression : 2 + 8 * 3 - 4 + 7 * 2 / 2 = 29 := by
  sorry

end compute_expression_l213_213119


namespace avg_speed_trip_l213_213733

theorem avg_speed_trip (D : ℝ) (hD : D > 0) :
  let time_first := (D/3) / 80
  let time_second := (D/3) / 24
  let time_third := (D/3) / 60
  let total_time := time_first + time_second + time_third
  (D / total_time) = (240 / 17) :=
by
  let time_first := (D/3) / 80
  let time_second := (D/3) / 24
  let time_third := (D/3) / 60
  let total_time := time_first + time_second + time_third
  have h1 : time_first = D / 240 := by sorry
  have h2 : time_second = 10 * D / 240 := by sorry
  have h3 : time_third = 4 * D / 240 := by sorry
  have total_time_calc : total_time = 17 * D / 240 := by
    calc total_time = (time_first + time_second + time_third) : by sorry
               ... = ((1 * D / 240) + (10 * D / 240) + (4 * D / 240)) : by sorry
               ... = (17 * D / 240) : by sorry
  calc D / total_time = D / (17 * D / 240) : by rw [total_time_calc]
                 ... = 240 / 17 : by sorry

end avg_speed_trip_l213_213733


namespace third_bakery_sacks_per_week_l213_213560

theorem third_bakery_sacks_per_week 
  (first_bakery_sacks_per_week : ℕ)
  (second_bakery_sacks_per_week : ℕ)
  (total_sacks_in_4_weeks : ℕ) 
  (first_bakery_sacks_per_week = 2)
  (second_bakery_sacks_per_week = 4)
  (total_sacks_in_4_weeks = 72)
  : (72 - (2 * 4 + 4 * 4)) / 4 = 12 := 
by
  sorry

end third_bakery_sacks_per_week_l213_213560


namespace sqrt_and_cbrt_sum_l213_213988

theorem sqrt_and_cbrt_sum : Real.sqrt ((-3)^2) + Real.cbrt 8 = 5 := by
  sorry

end sqrt_and_cbrt_sum_l213_213988


namespace minimize_expression_l213_213265

theorem minimize_expression (a b : ℝ) :
  let p := 2 * a^2 - 8 * a * b + 17 * b^2 - 16 * a - 4 * b + 2044
    in p = 2 * (a - 2 * b - 4) ^ 2 + 9 * (b - 2) ^ 2 + 1976 := 
sorry

end minimize_expression_l213_213265


namespace at_least_two_dice_same_number_probability_l213_213834

theorem at_least_two_dice_same_number_probability :
  let total_outcomes := 6^8
  let favorable_outcomes := 28 * 6! * 6^2
  let probability_all_different := favorable_outcomes / total_outcomes
  let required_probability := 1 - probability_all_different
  required_probability = 191 / 336
:= by
  sorry

end at_least_two_dice_same_number_probability_l213_213834


namespace simplify_expression_l213_213851

theorem simplify_expression (x : ℝ) (h : x ≥ 2) : 
    |2 - x| + (sqrt (x - 2))^2 - sqrt (4 * x^2 - 4 * x + 1) = -3 := 
by
    sorry

end simplify_expression_l213_213851


namespace arithmetic_sequence_sum_l213_213382

theorem arithmetic_sequence_sum :
  ∀ (a : ℕ → ℝ),
  a 1 = 3 →
  a 3 = 2 →
  let d := (a 3 - a 1) / 2 in
  let S := λ n : ℕ, n / 2 * (2 * a 1 + (n - 1) * d) in
  S 10 = 7.5 :=
by
  intro a h₁ h₃ d S,
  sorry

end arithmetic_sequence_sum_l213_213382


namespace lower_bound_of_sum_of_squares_of_roots_l213_213963

noncomputable def lower_bound_sum_of_squares_of_roots (a2 : ℝ) : ℝ :=
  4 * a2^2 - 2 * a2

theorem lower_bound_of_sum_of_squares_of_roots : ∃ a2 : ℝ, 
  ∀ (a3 : ℝ), (a3 = 2 * a2) → ∀ (roots : list ℝ), (roots.length = 4) →
  let sum_roots := roots.sum in
  let sum_pairwise_products := (roots.map_with_index (λ _ r_i, 
    roots.map_with_index (λ j r_j, if i < j then r_i * r_j else 0)).sum) in
  sum_roots^2 = -2 * a2 * sum_roots →
  sum_pairwise_products = binom 4 2 * a2 →
  abs (lower_bound_sum_of_squares_of_roots a2) = 1 / 4 :=
begin
  sorry

end lower_bound_of_sum_of_squares_of_roots_l213_213963


namespace martha_black_butterflies_l213_213810

theorem martha_black_butterflies
    (total_butterflies : ℕ)
    (total_blue_butterflies : ℕ)
    (total_yellow_butterflies : ℕ)
    (total_black_butterflies : ℕ)
    (h1 : total_butterflies = 19)
    (h2 : total_blue_butterflies = 6)
    (h3 : total_blue_butterflies = 2 * total_yellow_butterflies)
    (h4 : total_black_butterflies = total_butterflies - (total_blue_butterflies + total_yellow_butterflies))
    : total_black_butterflies = 10 :=
  sorry

end martha_black_butterflies_l213_213810


namespace find_n_l213_213621

theorem find_n (n : ℕ) (h : 2^6 * 3^3 * n = 10!) : n = 2100 :=
by 
  sorry

end find_n_l213_213621


namespace min_value_a_l213_213735

theorem min_value_a (a : ℝ) : (∀ x : ℝ, 0 < x ∧ x ≤ (Real.sqrt 2) / 2 → x^3 - 2 * x * Real.log x / Real.log a ≤ 0) ↔ a ≥ 1 / 4 := 
sorry

end min_value_a_l213_213735


namespace new_person_weight_l213_213020

theorem new_person_weight (avg_increase : ℝ) (num_people : ℕ) (weight_replaced : ℝ) (new_weight : ℝ) : 
    num_people = 8 → avg_increase = 1.5 → weight_replaced = 65 → 
    new_weight = weight_replaced + num_people * avg_increase → 
    new_weight = 77 :=
by
  intros h1 h2 h3 h4
  sorry

end new_person_weight_l213_213020


namespace henry_walks_total_distance_l213_213340

theorem henry_walks_total_distance :
  let total_distance := (infinite_series_sum (λ n, (if n % 2 = 0 then 
                                                      (4 / 5) * (3 : ℝ) * (2 / 3) ^ (n / 2) 
                                                    else 
                                                      -(4 / 5) * (3 : ℝ) * (2 / 3) ^ ((n - 1) / 2)) :
                                                    ℝ)) in
  total_distance = 45 / 7 :=
sorry

end henry_walks_total_distance_l213_213340


namespace base7_to_base10_expression_l213_213951

theorem base7_to_base10_expression :
  let encode : ℕ → ℕ := λ n, 
    let f := list.zip ['V,'W,'X,'Y,'Z,'A,'B] [0,1,2,3,4,5,6] in 
    Σ n,

  let encode (digit : ℕ) : ℕ := ... -- Encoding function for the base-7 digit
  let decode (digit : ℕ) : ℕ := ... -- Decoding function for the base-7 digit
    
  -- let the conditions of the problem describing the relationships between encoded integers in base-7.
  let VXY := 7^2 * encode 'V' + 7 * encode 'X' + encode 'Y' in
  let VYB := 7^2 * encode 'V' + 7 * encode 'Y' + encode 'B' in
  let VZA := 7^2 * encode 'V' + 7 * encode 'Z' + encode 'A' in
  
  -- The given condition: VXY + 3 = VYB
  VXY + 3 = VYB ∧ 
  
  -- The given condition: VYB - 1 = VZA
  VYB - 1 = VZA →
  
  -- conclusion being validated
  decode (base10_expr XYZ) = 288

:= 
begin
  -- sorry as placeholder
  sorry
end

end base7_to_base10_expression_l213_213951


namespace gcd_calculation_l213_213913

def gcd_36_45_495 : ℕ :=
  Int.gcd 36 (Int.gcd 45 495)

theorem gcd_calculation : gcd_36_45_495 = 9 := by
  sorry

end gcd_calculation_l213_213913


namespace arithmetic_seq_condition_l213_213693

def sum_first_n_terms (a d : ℕ) (n : ℕ) : ℕ := 
  n * a + (n * (n - 1) / 2) * d

theorem arithmetic_seq_condition (a2 : ℕ) (S3 S9 : ℕ) :
  a2 = 1 → 
  (∃ d, (d > 4 ∧ S3 = 3 * a2 + (3 * (3 - 1) / 2) * d ∧ S9 = 9 * a2 + (9 * (9 - 1) / 2) * d) → (S3 + S9) > 93) ↔ 
  (∃ d, (S3 + S9 = sum_first_n_terms a2 d 3 + sum_first_n_terms a2 d 9 ∧ (sum_first_n_terms a2 d 3 + sum_first_n_terms a2 d 9) > 93 → d > 3 ∧ a2 + d > 5)) :=
by 
  sorry

end arithmetic_seq_condition_l213_213693


namespace abc_not_8_l213_213305

theorem abc_not_8 (a b c : ℕ) (h : 2^a * 3^b * 4^c = 192) : a + b + c ≠ 8 :=
sorry

end abc_not_8_l213_213305


namespace shakes_sold_l213_213530

variable (s : ℕ) -- the number of shakes sold

-- conditions
def shakes_ounces := 4 * s
def cone_ounces := 6
def total_ounces := 14

-- the theorem to prove
theorem shakes_sold : shakes_ounces + cone_ounces = total_ounces → s = 2 := by
  intros h
  -- proof can be filled in here
  sorry

end shakes_sold_l213_213530


namespace c_seq_formula_l213_213469

def x_seq (n : ℕ) : ℕ := 2 * n - 1
def y_seq (n : ℕ) : ℕ := n ^ 2
def c_seq (n : ℕ) : ℕ := (2 * n - 1) ^ 2

theorem c_seq_formula (n : ℕ) : ∀ k, (c_seq k) = (2 * k - 1) ^ 2 :=
by
  sorry

end c_seq_formula_l213_213469


namespace scientific_notation_26_billion_l213_213068

theorem scientific_notation_26_billion :
  ∃ (a : ℝ) (n : ℤ), (26 * 10^8 : ℝ) = a * 10^n ∧ 1 ≤ |a| ∧ |a| < 10 ∧ a = 2.6 ∧ n = 9 :=
sorry

end scientific_notation_26_billion_l213_213068


namespace find_b_l213_213245

theorem find_b (b : ℝ) (h_floor : b + ⌊b⌋ = 22.6) : b = 11.6 :=
sorry

end find_b_l213_213245


namespace triangle_angles_l213_213761

theorem triangle_angles (A B C : ℝ) 
  (h1 : B = 4 * A)
  (h2 : C - B = 27)
  (h3 : A + B + C = 180) : 
  A = 17 ∧ B = 68 ∧ C = 95 :=
by {
  -- Sorry will be replaced once the actual proof is provided
  sorry 
}

end triangle_angles_l213_213761


namespace ceil_square_of_neg_seven_fourths_l213_213233

/-- Evaluate the ceiling of the square of -7/4 --/
theorem ceil_square_of_neg_seven_fourths : (Int.ceil ((-7/4 : ℚ)^2 : ℚ) = 4) :=
sorry

end ceil_square_of_neg_seven_fourths_l213_213233


namespace sum_of_fourth_powers_l213_213895

theorem sum_of_fourth_powers (a : Fin 88 -> ℝ) 
  (h1 : ∀ i, a i = -3 ∨ a i = -1) 
  (h2 : (∑ i, (a i)^2) = 280) : 
  (∑ i, (a i)^4) = 2008 := 
sorry

end sum_of_fourth_powers_l213_213895


namespace parabola_directrix_l213_213358

-- Defining the given condition
def given_parabola_equation (x y : ℝ) : Prop := y = 2 * x^2

-- Defining the expected directrix equation for the parabola
def directrix_equation (y : ℝ) : Prop := y = -1 / 8

-- The theorem we aim to prove
theorem parabola_directrix :
  (∀ x y : ℝ, given_parabola_equation x y) → (directrix_equation (-1 / 8)) :=
by
  -- Using 'sorry' here since the proof is not required
  sorry

end parabola_directrix_l213_213358


namespace ab_minus_anb_eq_2sqrt2_l213_213654

variable (a b : ℝ)

-- Definitions of conditions
axiom a_gt_one : a > 1
axiom b_pos_rational : b ∈ ℚ
axiom abp : a^b + a^(-b) = 2 * Real.sqrt 3

-- Statement to prove that given the above conditions, a^b - a^(-b) = 2√2
theorem ab_minus_anb_eq_2sqrt2 : a > 1 → b ∈ ℚ → a^b + a^(-b) = 2 * Real.sqrt 3 → a^b - a^(-b) = 2 * Real.sqrt 2 :=
by
  intro a_gt_one b_pos_rational abp
  sorry

end ab_minus_anb_eq_2sqrt2_l213_213654


namespace parabola_focus_l213_213248

theorem parabola_focus : 
  ∀ x y : ℝ, y = - (1 / 16) * x^2 → ∃ f : ℝ × ℝ, f = (0, -4) := 
by
  sorry

end parabola_focus_l213_213248


namespace energy_saving_devices_count_l213_213849

theorem energy_saving_devices_count :
  ∀ (total_weight : ℕ) (lightest_three_weight : ℕ) (heaviest_three_weight : ℕ),
  total_weight = 120 → 
  lightest_three_weight = 31 → 
  heaviest_three_weight = 41 → 
  (∀ w₁ w₂ : ℕ, w₁ ≠ w₂) → -- weights of any two devices are different 
  (∃ n : ℕ, n = 10) :=
by
  intros total_weight lightest_three_weight heaviest_three_weight
  assume total_weight_eq lightest_three_weight_eq heaviest_three_weight_eq weights_diff
  sorry -- proof skipped

end energy_saving_devices_count_l213_213849


namespace sum_primes_no_solution_congruence_l213_213590

theorem sum_primes_no_solution_congruence :
  ∑ p in {p | Nat.Prime p ∧ ¬ (∃ x : ℤ, 5 * (10 * x + 2) ≡ 3 [ZMOD p])}, p = 7 :=
sorry

end sum_primes_no_solution_congruence_l213_213590


namespace sqrt_pi_decimal_expansion_l213_213577

-- Statement of the problem: Compute the first 23 digits of the decimal expansion of sqrt(pi)
theorem sqrt_pi_decimal_expansion : 
  ( ∀ n, n ≤ 22 → 
    (digits : List ℕ) = [d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15, d16, d17, d18, d19, d20, d21, d22, d23] →
      (d1 = 1 ∧ d2 = 7 ∧ d3 = 7 ∧ d4 = 2 ∧ d5 = 4 ∧ d6 = 5 ∧ d7 = 3 ∧ d8 = 8 ∧ d9 = 5 ∧ d10 = 0 ∧ d11 = 9 ∧ d12 = 0 ∧ d13 = 5 ∧ d14 = 5 ∧ d15 = 1 ∧ d16 = 6 ∧ d17 = 0 ∧ d18 = 2 ∧ d19 = 7 ∧ d20 = 2 ∧ d21 = 9 ∧ d22 = 8 ∧ d23 = 1)) → 
  True :=
by
  sorry
  -- Actual proof to be filled, this is just the statement showing that we expected the digits 
  -- of the decimal expansion of sqrt(pi) match the specified values up to the 23rd place.

end sqrt_pi_decimal_expansion_l213_213577


namespace find_smaller_cube_side_length_l213_213523

noncomputable theory

def radius_of_sphere_with_cube_side (a : ℝ) : ℝ := (a * real.sqrt 3) / 2

def smaller_cube_side (R : ℝ) (d : ℝ) : ℝ :=
  let x := (-4 + real.sqrt (4^2 - 4 * 3 * (-4))) / (2 * 3) in x

theorem find_smaller_cube_side_length :
  let a := 2 in
  let R := radius_of_sphere_with_cube_side a in
  let d := a in
  smaller_cube_side R d = 2 / 3 :=
by
  sorry

end find_smaller_cube_side_length_l213_213523


namespace find_n_l213_213619

theorem find_n (n : ℕ) (h : 2^6 * 3^3 * n = 10!) : n = 2100 :=
by 
  sorry

end find_n_l213_213619


namespace cd_perpendicular_ab_l213_213280

-- Definitions of points and circles
variables (A B C D : Point) (circle1 circle2 : Circle)

-- Conditions
def is_diameter (A B : Point) (circle : Circle) : Prop :=
  circle.diameter = LineSegment(A, B)

def center (circle : Circle) : Point :=
  circle.center

def intersect (circle : Circle) (line : LineSegment) (P : Point) : Prop :=
  circle.intersects(line) ∧ line.contains(P)

def common_tangent (circle1 circle2 : Circle) (line : Line) : Prop :=
  circle1.is_tangent(line) ∧ circle2.is_tangent(line)

-- The proof problem statement
theorem cd_perpendicular_ab
  (h1 : is_diameter A B circle1)
  (h2 : center circle2 = A)
  (h3 : intersect circle2 (LineSegment A B) C)
  (h4 : LineSegment.length (A, C) < LineSegment.length (A, B) / 2)
  (h5 : common_tangent circle1 circle2 (Line.through D))
  : LineSegment.perpendicular (LineSegment C D) (LineSegment A B) := sorry

end cd_perpendicular_ab_l213_213280


namespace evaluate_expression_l213_213611

theorem evaluate_expression :
    123 - (45 * (9 - 6) - 78) + (0 / 1994) = 66 :=
by
  sorry

end evaluate_expression_l213_213611


namespace hyperbola_intersections_l213_213738

theorem hyperbola_intersections (L1 L2 : Line) : 
  ∃ (n : ℕ), n ∈ {0, 1, 2, 3, 4} :=
by
  let hyperbola (x y : ℝ) := x^2 - y^2 = 1
  let is_tangent_or_intersecting (L : Line) :=
    ∃ (a b : ℝ), (L (a, b) ∧ hyperbola a b) ∨ (L (a, b) ∧ tangent_to_hyperbola L)
  sorry

end hyperbola_intersections_l213_213738


namespace range_of_c_l213_213306

def p (c : ℝ) : Prop :=
  ∀ (x : ℝ), (1 - c > 0) ∧ ((1 - c) * x - 1 > 0) → 
             deriv (λ x : ℝ, real.logb 10 ((1-c)*x - 1)) x > 0

def q (c : ℝ) : Prop :=
  ∀ (x : ℝ), x + |x - 2 * c| > 1

theorem range_of_c (c : ℝ) (h : c > 0) :
  ((p c) ∨ (q c)) ∧ ¬((p c) ∧ (q c)) ↔ c ∈ set.Ioo 0 (1/2) ∪ set.Ici 1 :=
sorry

end range_of_c_l213_213306


namespace problem_solution_l213_213573

noncomputable def problem_expr : ℝ :=
  (2 / 3)^0 + 3 * (9 / 4)^(-1 / 2) + (Real.log10 4 + Real.log10 25)

theorem problem_solution : problem_expr = 5 :=
by
  sorry

end problem_solution_l213_213573


namespace maximum_value_of_g_l213_213397

noncomputable def g (x : ℝ) : ℝ := real.sqrt (x * (100 - x)) + real.sqrt (x * (8 - x))

theorem maximum_value_of_g :
  let x1 := (200 : ℝ) / 27
  let M := 12 * real.sqrt 6
  0 ≤ x ∧ x ≤ 8 → g x ≤ M ∧ g x1 = M :=
by 
  let x1 := (200 : ℝ) / 27
  let M := 12 * real.sqrt 6
  sorry

end maximum_value_of_g_l213_213397


namespace ceil_square_of_neg_fraction_l213_213185

theorem ceil_square_of_neg_fraction : 
  (Int.ceil ((-7 / 4 : ℚ)^2 : ℚ)).toNat = 4 := by
  sorry

end ceil_square_of_neg_fraction_l213_213185


namespace A_B_have_same_acquaintances_l213_213479

open Finset

variable (Participants : Type)
variable [Fintype Participants]

variable (knows : Participants → Participants → Prop)
variable (A B : Participants)

-- Conditions:
variable (h1 : knows A B) -- A and B know each other
variable (h2 : ¬ ∃ C, knows A C ∧ knows B C) -- A and B have no mutual acquaintances
variable (h3 : ∀ (x y : Participants), ¬ knows x y → Fintype.card {z | knows x z ∧ knows y z} = 2) -- Any two who do not know each other have exactly two mutual acquaintances

-- Goal:
theorem A_B_have_same_acquaintances :
  card {x | knows A x} = card {x | knows B x} :=
sorry

end A_B_have_same_acquaintances_l213_213479


namespace maximal_colored_squares_l213_213293

theorem maximal_colored_squares (n k : ℕ) (h₀ : n > 0) (h₁ : k > 0) (h₂ : n > k^2) (h₃ : k^2 > 4) :
  ∃ N : ℕ, N = n * (k - 1)^2 ∧ 
  (∀ (squares : set (ℕ × ℕ)) (colors : (ℕ × ℕ) → ℕ), 
  (∀ (x y : ℕ × ℕ), x ∈ squares → y ∈ squares → x.1 ≠ y.1 → x.2 ≠ y.2 → colors x = colors y)
   ∧ 
  (∀ (squares' : set (ℕ × ℕ)), 
   squares' ⊆ squares → squares'.card = k → 
   ∃ x y, x ∈ squares' ∧ y ∈ squares' ∧ colors x ≠ colors y)) :=
sorry

end maximal_colored_squares_l213_213293


namespace police_speed_l213_213545

/-- 
A thief runs away from a location with a speed of 20 km/hr.
A police officer starts chasing him from a location 60 km away after 1 hour.
The police officer catches the thief after 4 hours.
Prove that the speed of the police officer is 40 km/hr.
-/
theorem police_speed
  (thief_speed : ℝ)
  (police_start_distance : ℝ)
  (police_chase_time : ℝ)
  (time_head_start : ℝ)
  (police_distance_to_thief : ℝ)
  (thief_distance_after_time : ℝ)
  (total_distance_police_officer : ℝ) :
  thief_speed = 20 ∧
  police_start_distance = 60 ∧
  police_chase_time = 4 ∧
  time_head_start = 1 ∧
  police_distance_to_thief = police_start_distance + 100 ∧
  thief_distance_after_time = thief_speed * police_chase_time + thief_speed * time_head_start ∧
  total_distance_police_officer = police_start_distance + (thief_speed * (police_chase_time + time_head_start)) →
  (total_distance_police_officer / police_chase_time) = 40 := by
  sorry

end police_speed_l213_213545


namespace find_n_l213_213620

theorem find_n (n : ℕ) (h : 2^6 * 3^3 * n = 10!) : n = 2100 :=
by 
  sorry

end find_n_l213_213620


namespace set_operations_l213_213510

def A : Set ℝ := { x | x < 2 }
def B : Set ℝ := { x | 0 < x ∧ x < 5 }
def U : Set ℝ := Set.univ  -- Universal set ℝ
def complement (s : Set ℝ) : Set ℝ := { x | x ∉ s }

theorem set_operations :
  (A ∩ B = { x | 0 < x ∧ x < 2 }) ∧ 
  (complement A ∪ B = { x | 0 < x }) :=
by {
  sorry
}

end set_operations_l213_213510


namespace largest_club_size_is_four_l213_213373

variable {Player : Type} -- Assume Player is a type

-- Definition of the lesson-taking relation
variable (takes_lessons_from : Player → Player → Prop)

-- Club conditions
def club_conditions (A B C : Player) : Prop :=
  (takes_lessons_from A B ∧ ¬takes_lessons_from B C ∧ ¬takes_lessons_from C A) ∨ 
  (¬takes_lessons_from A B ∧ takes_lessons_from B C ∧ ¬takes_lessons_from C A) ∨ 
  (¬takes_lessons_from A B ∧ ¬takes_lessons_from B C ∧ takes_lessons_from C A)

theorem largest_club_size_is_four :
  ∀ (club : Finset Player),
  (∀ (A B C : Player), A ≠ B → B ≠ C → C ≠ A → A ∈ club → B ∈ club → C ∈ club → club_conditions takes_lessons_from A B C) →
  club.card ≤ 4 :=
sorry

end largest_club_size_is_four_l213_213373


namespace trapezoid_area_l213_213087

-- Given definitions and conditions for the problem
def isosceles_trapezoid_circumscribed_around_circle (a b h : ℝ) : Prop :=
  a > b ∧ h > 0 ∧ ∀ (x y : ℝ), x = h / 0.6 ∧ y = (2 * x - h) / 8 → a = b + 2 * √((h^2 - ((a - b) / 2)^2))

-- Definitions derived from conditions
def longer_base := 20
def base_angle := Real.arcsin 0.6

-- The proposition we need to prove (area == 74)
theorem trapezoid_area : 
  ∀ (a b h : ℝ), isosceles_trapezoid_circumscribed_around_circle a b h → base_angle = Real.arcsin 0.6 → 
  a = 20 → (1 / 2) * (b + 20) * h = 74 :=
sorry

end trapezoid_area_l213_213087


namespace plane_region_area_max_value_f_values_bc_l213_213322

-- Problem (1)
theorem plane_region_area
    (a b : ℝ)
    (f : ℝ → ℝ)
    (h : ∀ x, f x = ax^2 + bx)
    (h1 : -1 ≤ f (-1) ∧ f (-1) ≤ 2)
    (h2 : 2 ≤ f 1 ∧ f 1 ≤ 4) :
    is_area_of_plane_region (a, b) (3) :=
by
  sorry

-- Problem (2)
theorem max_value_f
    (x : ℝ)
    (h1 : x < 1)
    (h2 : x ≠ 0)
    (t : ℝ := 2 + 1 / (x^2 - x))
    (f : ℝ → ℝ := λ x, t * x) :
    ∃ M, ∀ x, (x < 1 ∧ x ≠ 0) → f x ≤ M ∧ M = 2 - 2 * sqrt 2 :=
by
  sorry

-- Problem (3)
theorem values_bc
    (a : ℝ)
    (h : a = 1)
    (b c : ℝ)
    (f : ℝ → ℝ)
    (h1 : ∀ x, f x = x^2 - (a+3) * x)
    (solution_set : set ℝ := {x | x ∈ Icc (-1 : ℝ) 5})
    (ineq1 : ∀ x, x ∈ solution_set → b^2 + c^2 - b * c - 3 * b - 1 ≤ f x)
    (ineq2 : ∀ x, x ∈ solution_set → f x ≤ a + 4) :
    b = 2 ∧ c = 1 :=
by
  sorry

end plane_region_area_max_value_f_values_bc_l213_213322


namespace MKNL_is_rectangle_l213_213507

noncomputable theory

structure Point where
  x : ℝ
  y : ℝ

def midpoint (P Q : Point) : Point :=
  { x := (P.x + Q.x) / 2, y := (P.y + Q.y) / 2 }

def is_rectangle (M N K L : Point) : Prop :=
  let mk := { x := K.x - M.x, y := K.y - M.y }
  let kn := { x := N.x - K.x, y := N.y - K.y }
  let nl := { x := L.x - N.x, y := L.y - N.y }
  let lm := { x := M.x - L.x, y := M.y - L.y }
  mk.x * kn.x + mk.y * kn.y = 0 ∧ -- mk and kn are perpendicular
  kn.x * nl.x + kn.y * nl.y = 0 ∧ -- kn and nl are perpendicular
  nl.x * lm.x + nl.y * lm.y = 0 ∧ -- nl and lm are perpendicular
  lm.x * mk.x + lm.y * mk.y = 0 ∧ -- lm and mk are perpendicular
  mk.x^2 + mk.y^2 = kn.x^2 + kn.y^2 ∧ -- lengths mk and kn are equal
  kn.x^2 + kn.y^2 = nl.x^2 + nl.y^2 ∧ -- lengths kn and nl are equal
  nl.x^2 + nl.y^2 = lm.x^2 + lm.y^2   -- lengths nl and lm are equal

theorem MKNL_is_rectangle
  (A B C D M N K L : Point)
  (hM: M = midpoint A B)
  (hN: N = midpoint C D)
  (hK: K = midpoint A C)
  (hL: L = midpoint B D) :
  is_rectangle M N K L := by
  sorry

end MKNL_is_rectangle_l213_213507


namespace continuous_stripe_probability_l213_213149

-- Define the cube properties and the specific conditions
variable {Face : Type} [Fintype Face] [DecidableEq Face]
variable [Fintype (Configuration : Face → bool)]

-- Define the probability being calculated
def probability_continuous_stripe_encircles_cube (configurations : List (Face → bool)) : ℚ :=
  6 / (2^6)

-- Statement of the proof problem
theorem continuous_stripe_probability :
  probability_continuous_stripe_encircles_cube configurations = 3 / 32 :=
sorry

end continuous_stripe_probability_l213_213149


namespace sum_of_possible_a_values_l213_213251

variable (a : ℤ)

def quadratic_function (x : ℤ) : Prop :=
  ∃ p q : ℤ, 
    (x² - a * x + a - 3 = 0 ∧ p + q = a ∧ p * q = a - 3)

theorem sum_of_possible_a_values :
  (∀ a : ℤ, quadratic_function a → sum_of_possible_a_values = 4) :=
  sorry

end sum_of_possible_a_values_l213_213251


namespace length_LD_l213_213663

open Real

variable (A B C D L K : Point)
variable (squareABCD : square A B C D)
variable (onCD : On L CD)
variable (onExtDA : On K (extension DA))
variable (angleKBL : angle K B L = π / 2)
variable (KD : dist K D = 19)
variable (CL : dist C L = 6)

theorem length_LD : dist L D = 7 := by
  sorry

end length_LD_l213_213663


namespace ceil_square_eq_four_l213_213152

theorem ceil_square_eq_four : (⌈(-7 / 4: ℚ)^2⌉ : ℤ) = 4 := by
  sorry

end ceil_square_eq_four_l213_213152


namespace remainder_proof_l213_213012

-- Definitions and conditions
variables {x y u v : ℕ}
variables (hx : x = u * y + v)

-- Problem statement in Lean 4
theorem remainder_proof (hx : x = u * y + v) : ((x + 3 * u * y + y) % y) = v :=
sorry

end remainder_proof_l213_213012


namespace third_function_symmetry_l213_213712

theorem third_function_symmetry (f : ℝ → ℝ) (x y : ℝ) 
    (h1 : ∀ x, f (f⁻¹ x) = x) 
    (h2 : ∀ x, f⁻¹ (f x) = x) 
    (h3 : ∀ x y, (x, y) on graph of third function ↔ (y, -x) on graph of inverse function) :
    y = -f(-x) :=
by
  sorry

end third_function_symmetry_l213_213712


namespace distinct_ways_to_divide_books_l213_213478

theorem distinct_ways_to_divide_books : 
  ∃ (ways : ℕ), ways = 5 := sorry

end distinct_ways_to_divide_books_l213_213478


namespace find_angle_C_find_max_area_l213_213364

noncomputable def triangle_properties (A B C : ℝ) (R : ℝ) (S : ℝ) : Prop :=
  (2 * sin (A + B) / 2 * sin (A + B) / 2 - cos (2 * C) = 1) ∧ (R = 2) ∧ (C = 2 * π / 3)

noncomputable def maximum_triangle_area (a b c S : ℝ) : Prop :=
  (2 * sin (A + B) / 2 * sin (A + B) / 2 - cos (2 * C) = 1) ∧ (R = 2) ∧ (S = sqrt 3)

theorem find_angle_C (A B C : ℝ) (R : ℝ) (h : triangle_properties A B C R) : C = 2 * π / 3 := 
sorry

theorem find_max_area (A B C a b c S : ℝ) (R : ℝ) (h : maximum_triangle_area a b c S) : S = sqrt 3 := 
sorry

end find_angle_C_find_max_area_l213_213364


namespace interest_cents_correct_l213_213033

noncomputable def interest_cents (A : ℝ) (rate : ℝ) (time : ℝ) : ℕ :=
  let P := A / (1 + rate * time)
  let interest := A - P
  let cents := interest - interest.to_int
  (cents * 100).to_nat

theorem interest_cents_correct : 
  interest_cents 307.80 0.06 (1 / 4) = 43 :=
by
  sorry

end interest_cents_correct_l213_213033


namespace constant_term_is_80_l213_213867

open Real
open Finset

noncomputable def constant_term_in_expansion : ℝ :=
  (x^2 - 2 / sqrt x)^5.coefficients.sum (λ r, if 10 - 5 * r / 2 = 0 then (-2)^4 * binomial 5 4 else 0)

theorem constant_term_is_80 :
  constant_term_in_expansion = 80 := by
  sorry

end constant_term_is_80_l213_213867


namespace empty_cell_exists_l213_213379

-- Definitions of the grid and conditions as hypotheses.
open List

def is_valid_boundary_arrow {n : ℕ} (boundary_arrows : list (ℕ × ℕ × direction)) : Prop :=
  ∀ (i j : ℕ), (i = 0 ∨ j = 0 ∨ i = n-1 ∨ j = n-1) → 
  (i,j, get_direction(i,j)) ∈ boundary_arrows 

def not_opposite_direction (dir1 dir2 : direction) : Prop :=
  match dir1, dir2 with
  | direction.up, direction.down => false
  | direction.down, direction.up => false
  | direction.left, direction.right => false
  | direction.right, direction.left => false
  | _, _ => true

def adjacent_cells_not_opposite {n : ℕ} (arrows : list (ℕ × ℕ × direction)) : Prop :=
  ∀ (i1 j1 i2 j2 : ℕ), 
  ((i1 = i2 ∧ |j1 - j2| = 1) ∨ (j1 = j2 ∧ |i1 - i2| = 1) ∨ (|i1-i2| = 1 ∧ |j1-j2| = 1 )) → 
  (i1, j1, get_direction(i1,j1)) ∈ arrows ∧ (i2, j2, get_direction(i2,j2)) ∈ arrows → 
  not_opposite_direction (get_direction(i1,j1)) (get_direction(i2,j2))

theorem empty_cell_exists {n : ℕ} (h : n = 20)
  (boundary_arrows : list (ℕ × ℕ × direction))
  (arrows : list (ℕ × ℕ × direction)) 
  (valid_boundary : is_valid_boundary_arrow boundary_arrows)
  (adjacent_not_opposite : adjacent_cells_not_opposite arrows) :
  ∃ (i j : ℕ), i < n ∧ j < n ∧ (get_direction(i,j), get_direction(i,j)) ∉ arrows :=
by
  sorry

end empty_cell_exists_l213_213379


namespace asymptote_slope_range_l213_213295

noncomputable def hyperbola_asymptote_slope (a b : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (GF1 GF2 : ℝ) (h₂ : |GF1| - 7 * |GF2| = 0) : set ℝ :=
{ m : ℝ | 0 < m ∧ m ≤ (Real.sqrt 7) / 3 }

theorem asymptote_slope_range
  (a b : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (GF1 GF2 : ℝ) 
  (h₂ : |GF1| - 7 * |GF2| = 0) :
  ∃ m, m ∈ hyperbola_asymptote_slope a b h₀ h₁ GF1 GF2 := sorry

end asymptote_slope_range_l213_213295


namespace find_x_l213_213786

-- Define the custom operation a star b
def star (a b : ℝ) : ℝ := (Real.sqrt (a + b)) / (Real.sqrt (a - b))

-- Define the given conditions
theorem find_x (x : ℝ) (h : star x 16 = 2) : x = 80 / 3 := by
  rw [star] at h
  have h₁ : Real.sqrt (x + 16) = 2 * Real.sqrt (x - 16) := by
    sorry
  have h₂ : (Real.sqrt (x + 16))^2 = (2 * Real.sqrt (x - 16))^2 := by
    sorry
  have h₃ : x + 16 = 4 * (x - 16) := by
    sorry
  have h₄ : x + 16 = 4 * x - 64 := by
    sorry
  have h₅ : 80 = 3 * x := by
    sorry
  exact (80 / 3)


end find_x_l213_213786


namespace composite_solid_volume_l213_213333

theorem composite_solid_volume :
  let V_prism := 2 * 2 * 1
  let V_cylinder := Real.pi * 1^2 * 3
  let V_overlap := Real.pi / 2
  V_prism + V_cylinder - V_overlap = 4 + 5 * Real.pi / 2 :=
by
  sorry

end composite_solid_volume_l213_213333


namespace probability_product_multiple_of_4_l213_213427

-- Definitions based on conditions
def spinner_paco : ℕ → Prop := λ n, 1 ≤ n ∧ n ≤ 5
def spinner_manu : ℕ → Prop := λ n, 1 ≤ n ∧ n ≤ 8

def multiple_of_4 (n : ℕ) : Prop := n % 4 = 0

-- The theorem statement
theorem probability_product_multiple_of_4 :
  (∑ p in finset.filter spinner_paco (finset.range 6), 
     ∑ m in finset.filter spinner_manu (finset.range 9), 
     if multiple_of_4 (p * m) then (1 / 5) * (1 / 8) else 0) = 2 / 5 :=
by {
 sorry
}

end probability_product_multiple_of_4_l213_213427


namespace boxes_left_l213_213855

theorem boxes_left (received : ℕ) (brother : ℕ) (sister : ℕ) (cousin : ℕ)
  (h_received : received = 45)
  (h_brother : brother = 12)
  (h_sister : sister = 9)
  (h_cousin : cousin = 7) :
  received - (brother + sister + cousin) = 17 :=
by
  rw [h_received, h_brother, h_sister, h_cousin]
  norm_num
  sorry

end boxes_left_l213_213855


namespace positive_integer_sixk_l213_213995

theorem positive_integer_sixk (n : ℕ) :
  (∃ d1 d2 d3 : ℕ, d1 < d2 ∧ d2 < d3 ∧ d1 + d2 + d3 = n ∧ d1 ∣ n ∧ d2 ∣ n ∧ d3 ∣ n) ↔ (∃ k : ℕ, n = 6 * k) :=
by
  sorry

end positive_integer_sixk_l213_213995


namespace volume_of_scaled_cube_l213_213922

theorem volume_of_scaled_cube (V : ℝ) (S : ℝ) (V' : ℝ) :
  (V = 8) →
  (S = 6 * (V ^ (1 / 3)) ^ 2) →
  (V' := (3 * S / 6) ^ (3 / 2)) →
  V' = 24 * Real.sqrt 3 :=
by
  intro hV hS hV'
  sorry

end volume_of_scaled_cube_l213_213922


namespace isosceles_triangle_angle_bisector_ratio_l213_213749

theorem isosceles_triangle_angle_bisector_ratio
  (A B C O F D: Type)
  [is_point A B C F D O]
  [triangle ABC]
  (h : ℝ)
  (isosceles : AB = BC)
  (altitude_A : altitude AF BC)
  (altitude_B : altitude BD AC)
  (intersection : altitudes_intersect_at AF BD O)
  (ratio : BO / OD = h) :
  angle_bisector_divides_ratio AE BD = sqrt (h + 2) :=
sorry

end isosceles_triangle_angle_bisector_ratio_l213_213749


namespace select_eleven_from_twenty_l213_213268

theorem select_eleven_from_twenty {s : Finset ℕ} (h : s ⊆ Finset.range 21)
  (hs_card : s.card = 11) : 
  ∃ a b ∈ s, a + b = 21 := 
begin
  sorry
end

end select_eleven_from_twenty_l213_213268


namespace area_of_triangle_ABD_l213_213022

-- Assuming appropriate definitions and constructs for the problem
noncomputable def area_triangle (A B C : Point) (AB BC : ℝ) (angleB : ℝ) :=
  (1 / 2) * AB * BC * Real.sin angleB

noncomputable def AD : ℝ := 2 / 5
noncomputable def total_area_ABC := area_triangle A B C 4 6 (30 * Real.pi / 180)

noncomputable def area_triangle_ABD := (AD / (AD + 3 / 5)) * total_area_ABC

theorem area_of_triangle_ABD (A B C : Point) (D : Point)
  (h1 : ∠BAC = 30 * Real.pi / 180)
  (h2 : segment_length AB = 4)
  (h3 : segment_length BC = 6)
  (h4 : AngleBisector B A C D) :
  area_triangle_ABD = 2.4 := 
sorry

end area_of_triangle_ABD_l213_213022


namespace ellipse_focal_length_l213_213667

theorem ellipse_focal_length (a : ℝ) (h1 : a > 0) (h2 : 2 * real.sqrt (a^2 - 8) = 4) :
  a = 2 * real.sqrt 3 ∨ a = 2 :=
begin
  -- Initialization of the proof structure
  sorry
end

end ellipse_focal_length_l213_213667


namespace simplify_sqrt_sum_l213_213445

theorem simplify_sqrt_sum : (Real.sqrt 72 + Real.sqrt 32) = 10 * Real.sqrt 2 := by
  sorry

end simplify_sqrt_sum_l213_213445


namespace expanded_ohara_triple_value_l213_213899

theorem expanded_ohara_triple_value :
  ∀ (a b c x : ℕ),
    a = 49 → b = 64 → c = 16 →
    (sqrt a + sqrt b + sqrt c = x) → x = 19 :=
by 
  sorry

end expanded_ohara_triple_value_l213_213899


namespace johns_raise_percent_increase_l213_213504

theorem johns_raise_percent_increase (original_earnings new_earnings : ℝ) 
  (h₀ : original_earnings = 60) (h₁ : new_earnings = 110) : 
  ((new_earnings - original_earnings) / original_earnings) * 100 = 83.33 :=
by
  sorry

end johns_raise_percent_increase_l213_213504


namespace martha_black_butterflies_l213_213804

-- Define the hypotheses
variables (total_butterflies blue_butterflies yellow_butterflies black_butterflies : ℕ)

-- Given conditions
def martha_collection_conditions : Prop :=
  total_butterflies = 19 ∧
  blue_butterflies = 6 ∧
  blue_butterflies = 2 * yellow_butterflies

-- The statement we want to prove
theorem martha_black_butterflies : martha_collection_conditions total_butterflies blue_butterflies yellow_butterflies black_butterflies →
  black_butterflies = 10 :=
sorry

end martha_black_butterflies_l213_213804


namespace gumballs_ensure_four_same_color_l213_213959

-- Define the total number of gumballs in each color
def red_gumballs : ℕ := 10
def white_gumballs : ℕ := 9
def blue_gumballs : ℕ := 8
def green_gumballs : ℕ := 7

-- Define the minimum number of gumballs to ensure four of the same color
def min_gumballs_to_ensure_four_same_color : ℕ := 13

-- Prove that the minimum number of gumballs to ensure four of the same color is 13
theorem gumballs_ensure_four_same_color (n : ℕ) 
  (h₁ : red_gumballs ≥ 3)
  (h₂ : white_gumballs ≥ 3)
  (h₃ : blue_gumballs ≥ 3)
  (h₄ : green_gumballs ≥ 3)
  : n ≥ min_gumballs_to_ensure_four_same_color := 
sorry

end gumballs_ensure_four_same_color_l213_213959


namespace river_current_speed_l213_213515

variable (c : ℝ)

def boat_speed_still_water : ℝ := 20
def round_trip_distance : ℝ := 182
def round_trip_time : ℝ := 10

theorem river_current_speed (h : (91 / (boat_speed_still_water - c)) + (91 / (boat_speed_still_water + c)) = round_trip_time) : c = 6 :=
sorry

end river_current_speed_l213_213515


namespace calculate_expression_l213_213574

theorem calculate_expression :
    - (1:ℝ)^6 - (Real.sqrt 3 - 2)^0 + Real.sqrt 3 * Real.tan (Real.pi / 6) - (Real.cos (Real.pi / 4))^2 + (-1/2 : ℝ)^(-2) = 5 / 2 := by
  sorry

end calculate_expression_l213_213574


namespace part_cost_l213_213802

theorem part_cost (hours : ℕ) (hourly_rate total_paid : ℕ) 
  (h1 : hours = 2)
  (h2 : hourly_rate = 75)
  (h3 : total_paid = 300) : 
  total_paid - (hours * hourly_rate) = 150 := 
by
  sorry

end part_cost_l213_213802


namespace sum_of_special_primes_l213_213595

theorem sum_of_special_primes : 
  let primes_with_no_solution := {p : ℕ | prime p ∧ ¬ ∃ x : ℤ, 5 * (10 * x + 2) ≡ 3 [MOD p]} in
  ∃ p1 p2, p1 ∈ primes_with_no_solution ∧ p2 ∈ primes_with_no_solution ∧ p1 ≠ p2 ∧ p1 + p2 = 7 :=
by
  sorry

end sum_of_special_primes_l213_213595


namespace find_f_neg2_l213_213788

def f (x : ℝ) (b : ℝ) : ℝ := if x >= 0 then 2^x + 2*x + b else -(2^(-x) + 2*(-x) + b)

theorem find_f_neg2 (b : ℝ) (h : 1 + b = 0) : f (-2) b = -7 :=
by
  sorry

end find_f_neg2_l213_213788


namespace Anne_speed_l213_213981

def distance : ℝ := 3.0
def time : ℝ := 1.5
def speed (d : ℝ) (t : ℝ) : ℝ := d / t

theorem Anne_speed (d t : ℝ) (h_d : d = distance) (h_t : t = time) : speed d t = 2.0 :=
by
  rw [h_d, h_t]
  dsimp [speed]
  norm_num

end Anne_speed_l213_213981


namespace complex_number_in_second_quadrant_l213_213356

theorem complex_number_in_second_quadrant 
  (a b : ℝ) 
  (h : ¬ (a ≥ 0 ∨ b ≤ 0)) : 
  (a < 0 ∧ b > 0) :=
sorry

end complex_number_in_second_quadrant_l213_213356


namespace soccer_team_wins_l213_213501

theorem soccer_team_wins : 
  ∀ (total_games won_percentage : ℕ), total_games = 140 → won_percentage = 50 → (won_percentage * total_games / 100) = 70 :=
by
  intros total_games won_percentage h_total_games h_won_percentage
  rw [h_total_games, h_won_percentage]
  sorry

end soccer_team_wins_l213_213501


namespace compute_a_to_the_fourth_l213_213724

theorem compute_a_to_the_fourth (a : ℝ) (h : (a + 1/a)^3 = 2) : 
  a^4 + 1/a^4 = (Real.cbrt 4 - 2)^2 - 2 :=
by
  sorry

end compute_a_to_the_fourth_l213_213724


namespace ceil_of_neg_frac_squared_l213_213204

-- Define the negated fraction
def neg_frac : ℚ := -7 / 4

-- Define the squared value of the negated fraction
def squared_value : ℚ := neg_frac ^ 2

-- Define the ceiling function applied to the squared value
def ceil_squared_value : ℤ := Int.ceil squared_value

-- Prove that the ceiling of the squared value is 4
theorem ceil_of_neg_frac_squared : ceil_squared_value = 4 := 
by sorry

end ceil_of_neg_frac_squared_l213_213204


namespace number_of_people_wearing_hats_l213_213607

/-- Total number of adults -/
def total_adults : ℕ := 2400

/-- Fraction of women among the adults -/
def fraction_of_women : ℚ := 2/3

/-- Percentage of women wearing hats -/
def percentage_of_women_wearing_hats : ℚ := 30/100

/-- Percentage of men wearing hats -/
def percentage_of_men_wearing_hats : ℚ := 12/100

/-- Number of people wearing hats -/
theorem number_of_people_wearing_hats : 
  let num_women := (fraction_of_women * total_adults).toNat,
      num_men := total_adults - num_women,
      women_wearing_hats := (percentage_of_women_wearing_hats * num_women).toNat,
      men_wearing_hats := (percentage_of_men_wearing_hats * num_men).toNat
  in women_wearing_hats + men_wearing_hats = 576 :=
by {
  sorry -- Proof is omitted
}

end number_of_people_wearing_hats_l213_213607


namespace sum_of_coordinates_of_D_l213_213431

def Point := (ℝ × ℝ)

def isRectangle (A B C D : Point) : Prop :=
  let M1 := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  let M2 := ((B.1 + C.1) / 2, (B.2 + C.2) / 2)
  let M3 := ((C.1 + D.1) / 2, (C.2 + D.2) / 2)
  let M4 := ((D.1 + A.1) / 2, (D.2 + A.2) / 2)
  (M1.1 - M3.1)^2 + (M1.2 - M3.2)^2 = (M2.1 - M4.1)^2 + (M2.2 - M4.2)^2 ∧ 
  (M1.1 - M2.1) * (M1.1 - M3.1) + (M1.2 - M2.2) * (M1.2 - M3.2) = 0

theorem sum_of_coordinates_of_D
  (a b : ℕ)
  (A B C D : Point)
  (hA : A = (2,8))
  (hB : B = (1,2))
  (hC : C = (5,4))
  (hD : D = (a,b))
  (h : isRectangle A B C D) :
  a + b = 10 := 
  sorry

end sum_of_coordinates_of_D_l213_213431


namespace sum_of_special_primes_l213_213594

theorem sum_of_special_primes : 
  let primes_with_no_solution := {p : ℕ | prime p ∧ ¬ ∃ x : ℤ, 5 * (10 * x + 2) ≡ 3 [MOD p]} in
  ∃ p1 p2, p1 ∈ primes_with_no_solution ∧ p2 ∈ primes_with_no_solution ∧ p1 ≠ p2 ∧ p1 + p2 = 7 :=
by
  sorry

end sum_of_special_primes_l213_213594


namespace area_ratio_l213_213756

variables (A B C D E F P N1 N2 N3 : Type)
variable [field A] -- Assuming this represents the areas for simplicity

-- Given conditions
def is_division_ratio (X Y : Type) (r : ℝ) : Prop := sorry

def ratios (D E F P : Type) :=
  (is_division_ratio D A 0.25) ∧
  (is_division_ratio E A 0.25) ∧
  (is_division_ratio F A 0.25) ∧
  (is_division_ratio P B (1/4))

-- The question we want to answer and the proof statement
theorem area_ratio (h : ratios D E F P) : 
  let K := (A B C) in
    (N1 N2 N3) = (14/25) * K :=
sorry

end area_ratio_l213_213756


namespace intersection_M_N_is_correct_l213_213711

def M : Set ℝ := { x | Real.ln (x + 1) > 0 }
def N : Set ℝ := { x | -2 ≤ x ∧ x ≤ 2 }
def intersect_M_N : Set ℝ := { x | 0 < x ∧ x ≤ 2 }

theorem intersection_M_N_is_correct : (M ∩ N) = intersect_M_N := 
  sorry

end intersection_M_N_is_correct_l213_213711


namespace sum_of_primes_no_solution_congruence_l213_213600

theorem sum_of_primes_no_solution_congruence :
  2 + 5 = 7 :=
by
  sorry

end sum_of_primes_no_solution_congruence_l213_213600


namespace ceil_square_of_neg_fraction_l213_213187

theorem ceil_square_of_neg_fraction : 
  (Int.ceil ((-7 / 4 : ℚ)^2 : ℚ)).toNat = 4 := by
  sorry

end ceil_square_of_neg_fraction_l213_213187


namespace profit_percentage_is_correct_l213_213950

-- Definitions for the given conditions
def SP : ℝ := 850
def Profit : ℝ := 255
def CP : ℝ := SP - Profit

-- The target proof statement
theorem profit_percentage_is_correct : 
  (Profit / CP) * 100 = 42.86 := by
  sorry

end profit_percentage_is_correct_l213_213950


namespace boxes_left_l213_213856

theorem boxes_left (received : ℕ) (brother : ℕ) (sister : ℕ) (cousin : ℕ)
  (h_received : received = 45)
  (h_brother : brother = 12)
  (h_sister : sister = 9)
  (h_cousin : cousin = 7) :
  received - (brother + sister + cousin) = 17 :=
by
  rw [h_received, h_brother, h_sister, h_cousin]
  norm_num
  sorry

end boxes_left_l213_213856


namespace repeating_block_length_five_sevenths_l213_213914

theorem repeating_block_length_five_sevenths : 
  ∃ n : ℕ, (∃ k : ℕ, (5 * 10^k - 5) % 7 = 0) ∧ n = 6 :=
sorry

end repeating_block_length_five_sevenths_l213_213914


namespace johns_remaining_money_l213_213772

theorem johns_remaining_money (q : ℝ) : 
  let cost_of_drinks := 4 * q,
      cost_of_small_pizzas := 2 * q,
      cost_of_large_pizza := 4 * q,
      total_cost := cost_of_drinks + cost_of_small_pizzas + cost_of_large_pizza,
      initial_money := 50 in
  initial_money - total_cost = 50 - 10 * q :=
by
  sorry

end johns_remaining_money_l213_213772


namespace height_difference_percentage_l213_213550

theorem height_difference_percentage (H_A H_B : ℝ) (h : H_B = H_A * 1.8181818181818183) :
  (H_A < H_B) → ((H_B - H_A) / H_B) * 100 = 45 := 
by 
  sorry

end height_difference_percentage_l213_213550


namespace ceil_square_of_neg_seven_fourths_l213_213237

/-- Evaluate the ceiling of the square of -7/4 --/
theorem ceil_square_of_neg_seven_fourths : (Int.ceil ((-7/4 : ℚ)^2 : ℚ) = 4) :=
sorry

end ceil_square_of_neg_seven_fourths_l213_213237


namespace geoboard_quadrilateral_area_l213_213998

open Real

theorem geoboard_quadrilateral_area :
  let vertices := [(7, 1), (2, 6), (4, 5), (11, 11)]
  let shoelace_area (vertices : List (ℝ × ℝ)) : ℝ :=
    0.5 * abs (
      vertices.zip (vertices.tail ++ [vertices.head])
      |>.map (λ ⟨(x1, y1), (x2, y2)⟩ => x1 * y2 - x2 * y1)
      |>.sum)
  shoelace_area vertices = 25.5 := by sorry

end geoboard_quadrilateral_area_l213_213998


namespace prime_check_for_d1_prime_check_for_d2_l213_213057

-- Define d1 and d2
def d1 : ℕ := 9^4 - 9^3 + 9^2 - 9 + 1
def d2 : ℕ := 9^4 - 9^2 + 1

-- Prime checking function
def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Stating the conditions and proofs
theorem prime_check_for_d1 : ¬ is_prime d1 :=
by {
  -- condition: ten 8's in base nine is divisible by d1 (5905) is not used here directly
  sorry
}

theorem prime_check_for_d2 : is_prime d2 :=
by {
  -- condition: twelve 8's in base nine is divisible by d2 (6481) is not used here directly
  sorry
}

end prime_check_for_d1_prime_check_for_d2_l213_213057


namespace x_approx_l213_213254

noncomputable def solve_for_x : ℝ :=
  let numerator : ℝ := 3.6 * 0.48 * 2.5
  let denominator : ℝ := 0.12 * 0.09
  let fraction : ℝ := numerator / denominator
  let lhs : ℝ := 2.5 * (fraction / x)
  let rhs : ℝ := 2000.0000000000002
in fraction

theorem x_approx :
  ∃ x : ℝ, 2.5 * (3.6 * 0.48 * 2.5 / (0.12 * 0.09 * x)) = 2000.0000000000002 ∧ x ≈ 1.25 :=
begin
  use 1.25,
  split,
  sorry,
  sorry,
end

end x_approx_l213_213254


namespace find_angleB_find_maxArea_l213_213301

noncomputable def angleB (a b c : ℝ) (C : ℝ) :=
  (a + c) / b = Real.cos C + Real.sqrt 3 * Real.sin C

noncomputable def maxArea (a b c : ℝ) (B : ℝ) :=
  b = 2

theorem find_angleB (a b c : ℝ) (C : ℝ) (h : angleB a b c C) : 
  ∃ B, B = 60 ∧ angleB a b c C :=
sorry

theorem find_maxArea (a b c : ℝ) (B : ℝ) (hB : B = 60) (hb : maxArea a b c B) :
  ∃ S, S = Real.sqrt 3 :=
sorry

end find_angleB_find_maxArea_l213_213301


namespace simplify_radicals_l213_213442

open Real

theorem simplify_radicals : sqrt 72 + sqrt 32 = 10 * sqrt 2 := by
  sorry

end simplify_radicals_l213_213442


namespace domain_of_ln_sinx_minus_one_l213_213869

noncomputable def domain_of_ln_func : set ℝ :=
{ x : ℝ | ∃ (k : ℤ), (π/6) + 2 * k * π < x ∧ x < (5 * π / 6) + 2 * k * π }

theorem domain_of_ln_sinx_minus_one :
  ∀ (x : ℝ), (∃ (k : ℤ), (π/6) + 2 * k * π < x ∧ x < (5 * π / 6) + 2 * k * π) ↔ 
    2 * sin x - 1 > 0 :=
by
  sorry

end domain_of_ln_sinx_minus_one_l213_213869


namespace union_A_B_inter_compl_A_B_subset_intersection_l213_213710

open Set

variable {R : Set Real}
def A : Set ℝ := {x : ℝ | -5 < x ∧ x < 1}
def B : Set ℝ := {x : ℝ | -2 < x ∧ x < 8}
def C (a : ℝ) : Set ℝ := {x : ℝ | x < a}
def A_c : Set ℝ := compl A

-- First part: Prove A ∪ B = {x | -5 < x < 8}
theorem union_A_B : A ∪ B = {x : ℝ | -5 < x ∧ x < 8} :=
sorry

-- Second part: Prove Aᶜ ∩ B = {x | 1 ≤ x ∧ x < 8}
theorem inter_compl_A_B : A_c ∩ B = {x : ℝ | 1 ≤ x ∧ x < 8} :=
sorry

-- Third part: Given A ∩ B ⊆ C, prove a ≥ 1
theorem subset_intersection (a : ℝ) (h : A ∩ B ⊆ C a) : a ≥ 1 :=
sorry

end union_A_B_inter_compl_A_B_subset_intersection_l213_213710


namespace ab_max_am_gm_min_max_expression_min_fraction_l213_213726

theorem ab_max (a b : ℝ) (h : a + b = 1) (ha : a > 0) (hb : b > 0) : ab ≤ 1 / 4 :=
sorry

theorem am_gm_min (a b : ℝ) (h : a + b = 1) (ha : a > 0) (hb : b > 0) : 
  (a + 1 / a) * (b + 1 / b) ≥ 4 :=
sorry

theorem max_expression (a b : ℝ) (h : a + b = 1) (ha : a > 0) (hb : b > 0) : 
  4 * a - 1 / (4 * b) ≤ 2 :=
sorry

theorem min_fraction (a b : ℝ) (h : a + b = 1) (ha : a > 0) (hb : b > 0) : 
  1 / a + 2 / b ≥ 3 + 2 * real.sqrt 2 :=
sorry

end ab_max_am_gm_min_max_expression_min_fraction_l213_213726


namespace problem_l213_213261

section
variable (a b c : ℝ)

-- Define the operation \spadesuit
def spadesuit (a b : ℝ) := (a + b) * (a - b)

-- State the problem
theorem problem (h : spadesuit 2 (spadesuit 3 (spadesuit 1 2)) = 4) : 
  spadesuit 2 (spadesuit 3 (spadesuit 1 2)) = 4 := 
begin
  sorry
end

end

end problem_l213_213261


namespace second_box_weight_l213_213037

theorem second_box_weight (height1 width1 length1 height2 width2 length2 : ℝ) (material_density : ℝ) (packing_efficiency : ℝ)
  (V1 : height1 * width1 * length1 = 48) (material_weight1 : material_density * V1 = 24) :
  height2 = 3 * height1 → width2 = 2 * width1 → length2 = length1 → packing_efficiency = 0.8 →
  (0.5 * (packing_efficiency * (height2 * width2 * length2)) = 115.2) := 
begin
  intros,
  sorry
end

end second_box_weight_l213_213037


namespace find_smallest_n_l213_213050

noncomputable def smallest_n (c : ℕ) (n : ℕ) : Prop :=
  (∀ k : ℕ, 1 ≤ k ∧ k ≤ c → n + 2 - 2*k ≥ 0) ∧ c * (n - c + 1) = 2009

theorem find_smallest_n : ∃ n c : ℕ, smallest_n c n ∧ n = 89 :=
sorry

end find_smallest_n_l213_213050


namespace five_numbers_satisfy_conditions_l213_213927

theorem five_numbers_satisfy_conditions :
  ∃ (a1 a2 a3 a4 a5 : ℤ),
    a1 = 3 ∧ a2 = -4 ∧ a3 = 3 ∧ a4 = -4 ∧ a5 = 3 ∧
    a1 + a2 < 0 ∧ a2 + a3 < 0 ∧ a3 + a4 < 0 ∧ a4 + a5 < 0 ∧
    a1 + a2 + a3 + a4 + a5 > 0 :=
by
  use 3, -4, 3, -4, 3
  split ; norm_num
  split ; norm_num
  split ; norm_num
  split ; norm_num
  split ; norm_num
  sorry -- end of proof

end five_numbers_satisfy_conditions_l213_213927


namespace sums_equal_l213_213580

def grid := matrix (fin 3) (fin 3) ℤ

def valid_grid (g : grid) : Prop :=
  ∀ (i j : fin 3), g i j = -1 ∨ g i j = 0 ∨ g i j = 1

def row_sum (g : grid) (i : fin 3) : ℤ := g i 0 + g i 1 + g i 2
def column_sum (g : grid) (j : fin 3) : ℤ := g 0 j + g 1 j + g 2 j
def main_diag_sum (g : grid) : ℤ := g 0 0 + g 1 1 + g 2 2
def anti_diag_sum (g : grid) : ℤ := g 0 2 + g 1 1 + g 2 0

def grid_sums (g : grid) : list ℤ :=
  [row_sum g 0, row_sum g 1, row_sum g 2,
   column_sum g 0, column_sum g 1, column_sum g 2,
   main_diag_sum g, anti_diag_sum g]

theorem sums_equal (g : grid) (h : valid_grid g) : 
  ∃ (x y : ℤ), x ≠ y ∧ x ∈ grid_sums g ∧ y ∈ grid_sums g ∧ x = y :=
sorry

end sums_equal_l213_213580


namespace sum_of_minimal_area_triangle_integer_k_l213_213464

theorem sum_of_minimal_area_triangle_integer_k :
  let p1 := (2, 5)
  let p2 := (10, 20)
  let is_vertex_of_triangle (k : ℤ) := (7, k)
  let minimum_area_k := [14, 15]
  sum minimum_area_k = 29 :=  
by
  let k1 := 14
  let k2 := 15
  have k_integers : k1 = 14 ∧ k2 = 15 := by 
    split;
    rfl
  have sum_k : k1 + k2 = 29 := by
    rw [k_integers.left, k_integers.right]
    norm_num
  exact sum_k

end sum_of_minimal_area_triangle_integer_k_l213_213464


namespace find_a_l213_213758

noncomputable def pole_to_cartesian (ρ θ : ℝ) (a : ℝ) : Prop := 
  (ρ * (sin θ)^2 = 2 * a * cos θ)

noncomputable def line_parametric_eq (t : ℝ) : (ℝ × ℝ) :=
  ( -2 + (sqrt 2 / 2) * t, -4 + (sqrt 2 / 2) * t )

noncomputable def curve_cartesian (x y a: ℝ) : Prop :=
  (y^2 = 2 * a * x)

noncomputable def line_cartesian (x y : ℝ) : Prop :=
  (y = x - 2)

noncomputable def intersects_curve_and_line (a t : ℝ) : Prop :=
  let x := -2 + (sqrt 2 / 2) * t in
  let y := -4 + (sqrt 2 / 2) * t in
  curve_cartesian x y a

noncomputable def geometric_sequence_cond (t1 t2 a : ℝ) : Prop :=
  let s := 2 * sqrt 2 * (4 + a) in
  let p := 8 * (4 + a) in
  ((t1 - t2)^2 = (s^2 - 4 * p)) ∧ (s = 2 * sqrt 2 * (4 + a)) ∧ (p = 8 * (4 + a))

theorem find_a (a : ℝ) (h₁ : 0 < a)
  (h₂ : ∀ t, intersects_curve_and_line a t)
  (h₃ : ∃ t1 t2, geometric_sequence_cond t1 t2 a) :
  a = 1 :=
sorry

end find_a_l213_213758


namespace trapezoid_area_l213_213088

-- Given definitions and conditions for the problem
def isosceles_trapezoid_circumscribed_around_circle (a b h : ℝ) : Prop :=
  a > b ∧ h > 0 ∧ ∀ (x y : ℝ), x = h / 0.6 ∧ y = (2 * x - h) / 8 → a = b + 2 * √((h^2 - ((a - b) / 2)^2))

-- Definitions derived from conditions
def longer_base := 20
def base_angle := Real.arcsin 0.6

-- The proposition we need to prove (area == 74)
theorem trapezoid_area : 
  ∀ (a b h : ℝ), isosceles_trapezoid_circumscribed_around_circle a b h → base_angle = Real.arcsin 0.6 → 
  a = 20 → (1 / 2) * (b + 20) * h = 74 :=
sorry

end trapezoid_area_l213_213088


namespace average_salary_of_employees_l213_213456

theorem average_salary_of_employees (A : ℝ)
  (h1 : 24 * A + 11500 = 25 * (A + 400)) :
  A = 1500 := 
by
  sorry

end average_salary_of_employees_l213_213456


namespace ceiling_of_square_frac_l213_213222

theorem ceiling_of_square_frac : 
  let x : ℚ := -7 / 4
  let y : ℚ := x^2
  in Int.ceil y = 4 := 
by 
  let x : ℚ := -7 / 4
  let y : ℚ := x^2
  exact sorry

end ceiling_of_square_frac_l213_213222


namespace bus_catch_up_time_l213_213942

theorem bus_catch_up_time
  (l_bus : ℝ) (v_bus_kph : ℝ) (r_bus : ℝ)
  (v_skate_kph : ℝ) (r_skate : ℝ) (start_opposite : bool)
  (h_l_bus : l_bus = 15)
  (h_v_bus_kph : v_bus_kph = 40)
  (h_r_bus : r_bus = 500)
  (h_v_skate_kph : v_skate_kph = 8)
  (h_r_skate : r_skate = 250)
  (h_start_opposite : start_opposite = true) :
  let v_bus := (v_bus_kph * 1000) / 3600,
      v_skate := (v_skate_kph * 1000) / 3600,
      ω_bus := v_bus / r_bus,
      ω_skate := v_skate / r_skate,
      ω_relative := ω_bus + ω_skate,
      angular_distance := real.pi,
      t := angular_distance / ω_relative
  in
  t ≈ 101.0 :=
by
  sorry

end bus_catch_up_time_l213_213942


namespace similar_polygons_perimeter_ratio_l213_213359

-- Define the main function to assert the proportional relationship
theorem similar_polygons_perimeter_ratio (x y : ℕ) (h1 : 9 * y^2 = 64 * x^2) : x * 8 = y * 3 :=
by sorry

-- noncomputable if needed (only necessary when computation is involved, otherwise omit)

end similar_polygons_perimeter_ratio_l213_213359


namespace compute_expression_l213_213638

-- Definition of M and m
def M(a b : ℝ) : ℝ := max a b
def m(a b : ℝ) : ℝ := min a b

-- The theorem stating the problem and solution
theorem compute_expression (p q r s t : ℝ) (h : p < q ∧ q < r ∧ r < s ∧ s < t) :
  M(m(M(p, q), r), M(s, m(t, p))) = s :=
sorry

end compute_expression_l213_213638


namespace find_larger_number_l213_213503

theorem find_larger_number (L S : ℕ) (h1 : L - S = 1375) (h2 : L = 6 * S + 15) : L = 1647 :=
by
  -- proof to be filled
  sorry

end find_larger_number_l213_213503


namespace num_solutions_cos_eq_0_5_l213_213719

-- Define the interval and condition
def interval (x : ℝ) : Prop := 0 ≤ x ∧ x < 360
def cos_condition (x : ℝ) : Prop := Real.cos (x * (Float.pi / 180)) = 0.5

-- Define the main theorem
theorem num_solutions_cos_eq_0_5 : 
  (setOf (λ x, interval x ∧ cos_condition x)).card = 2 :=
begin
  sorry
end

end num_solutions_cos_eq_0_5_l213_213719


namespace ceil_square_of_neg_fraction_l213_213191

theorem ceil_square_of_neg_fraction : 
  (Int.ceil ((-7 / 4 : ℚ)^2 : ℚ)).toNat = 4 := by
  sorry

end ceil_square_of_neg_fraction_l213_213191


namespace initial_loss_percentage_l213_213108

theorem initial_loss_percentage 
  (C : ℝ) 
  (h1 : selling_price_one_pencil_20 = 1 / 20)
  (h2 : selling_price_one_pencil_10 = 1 / 10)
  (h3 : C = 1 / (10 * 1.30)) :
  (C - selling_price_one_pencil_20) / C * 100 = 35 :=
by
  sorry

end initial_loss_percentage_l213_213108


namespace pyramid_congruent_faces_exists_l213_213459

-- Definition of a regular polygon
structure RegularPolygon (n : ℕ) :=
  (vertices : Fin n → ℝ × ℝ)
  (sides_eq : ∀ i j, dist (vertices i) (vertices (i + 1) % n) = dist (vertices j) (vertices (j + 1) % n))

-- Definition of a pyramid with an equilateral base and equal plane angles at the apex
structure Pyramid (n : ℕ) :=
  (base : RegularPolygon n)
  (apex : ℝ × ℝ × ℝ)
  (plane_angles_eq : ∀ i j, angle_at_apex (base.vertices i) (apex) (base.vertices (i + 1) % n) = angle_at_apex (base.vertices j) (apex) (base.vertices (j + 1) % n))

-- Angles formed by two sides of the apex with the base edges
def angle_at_apex (p1 p2 p3 : ℝ × ℝ × ℝ) : ℝ := sorry -- Assume this function is defined

-- Statement: There exist at least two congruent lateral triangular faces
theorem pyramid_congruent_faces_exists {n : ℕ} (pyramid : Pyramid n) :
 ∃ (i j : Fin n), i ≠ j ∧ 
  congruent (triangle (pyramid.apex) (pyramid.base.vertices i) (pyramid.base.vertices (i + 1) % n))
            (triangle (pyramid.apex) (pyramid.base.vertices j) (pyramid.base.vertices (j + 1) % n)) :=
sorry

end pyramid_congruent_faces_exists_l213_213459


namespace vec_addition_l213_213110

namespace VectorCalculation

open Real

def v1 : ℤ × ℤ := (3, -8)
def v2 : ℤ × ℤ := (2, -6)
def scalar : ℤ := 5

def scaled_v2 : ℤ × ℤ := (scalar * v2.1, scalar * v2.2)
def result : ℤ × ℤ := (v1.1 + scaled_v2.1, v1.2 + scaled_v2.2)

theorem vec_addition : result = (13, -38) := by
  sorry

end VectorCalculation

end vec_addition_l213_213110


namespace probability_two_digit_multiple_of_5_probability_two_digit_even_probability_three_digit_greater_than_234_l213_213269

theorem probability_two_digit_multiple_of_5 :
  let numbers := {1, 2, 3, 4, 5}
  let all_two_digit_combinations := 20
  let favorable_combinations := 4
  (favorable_combinations / all_two_digit_combinations) = 1 / 5 := by
  sorry

theorem probability_two_digit_even :
  let numbers := {1, 2, 3, 4, 5}
  let all_two_digit_combinations := 20
  let favorable_combinations := 8
  (favorable_combinations / all_two_digit_combinations) = 2 / 5 := by
  sorry

theorem probability_three_digit_greater_than_234 :
  let numbers := {1, 2, 3, 4, 5}
  let all_three_digit_combinations := 60
  let favorable_combinations := 28
  (favorable_combinations / all_three_digit_combinations) = 7 / 15 := by
  sorry

end probability_two_digit_multiple_of_5_probability_two_digit_even_probability_three_digit_greater_than_234_l213_213269


namespace f_at_7_l213_213689

-- Define the function f and its properties
axiom f : ℝ → ℝ
axiom odd_f : ∀ x : ℝ, f (-x) = -f x
axiom periodic_f : ∀ x : ℝ, f (x + 4) = f x
axiom values_f : ∀ x : ℝ, 0 < x ∧ x < 2 → f x = 2 * x^2

-- Prove that f(7) = -2
theorem f_at_7 : f 7 = -2 :=
by
  sorry

end f_at_7_l213_213689


namespace collinear_O1_A_P_l213_213484

variables (O1 O2 A B P : Type*)

-- Definition of the two circles intersecting at points A and B
def circles_intersect_at (circ1 circ2 : Set (Set Type*)) (O1 O2 A B : Type*) : Prop :=
  A ∈ circ1 ∧ A ∈ circ2 ∧ B ∈ circ1 ∧ B ∈ circ2

-- Definition of the circle passing through O1, B, O2 intersecting the second circle at P
def circle_O1BO2_intersects_second_at_P (circ1 circ2 : Set (Set Type*)) (O1 O2 B P : Type*) : Prop :=
  (O1 ∈ circ1 ∧ B ∈ circ1 ∧ O2 ∈ circ1) ∧ P ∈ circ2 ∧ (O1 ∈ circ2 ∨ B ∈ circ2 ∨ O2 ∈ circ2)

-- The statement of collinearity to prove
theorem collinear_O1_A_P (circ1 circ2 : Set (Set Type*)) (O1 O2 A B P : Type*)
  (h1 : circles_intersect_at circ1 circ2 O1 O2 A B)
  (h2 : circle_O1BO2_intersects_second_at_P circ1 circ2 O1 O2 B P) : collinear O1 A P :=
sorry

end collinear_O1_A_P_l213_213484


namespace maximum_bags_of_milk_l213_213928

theorem maximum_bags_of_milk (bag_cost : ℚ) (promotion : ℕ → ℕ) (total_money : ℚ) 
  (h1 : bag_cost = 2.5) 
  (h2 : promotion 2 = 3) 
  (h3 : total_money = 30) : 
  ∃ n, n = 18 ∧ (total_money >= n * bag_cost - (n / 3) * bag_cost) :=
by
  sorry

end maximum_bags_of_milk_l213_213928


namespace digit_2_count_correct_l213_213466

-- Define the conditions provided in the problem
def room_numbers := {n : ℕ // 100 ≤ n ∧ n < 600 ∧ (n % 100) ≤ 35 ∧ n % 100 ≠ 0}

-- Define a function to count occurrences of the digit '2'
def count_digit_2 (n : ℕ) : ℕ :=
  (toString n).data.count (λ c => c = '2')

-- A total count of digit '2' in all room numbers
def total_count_of_digit_2 : ℕ :=
  room_numbers.to_finset.sum (λ ⟨n, _⟩ => count_digit_2 n)

-- The theorem to be proved
theorem digit_2_count_correct : total_count_of_digit_2 = 105 :=
by
  sorry

end digit_2_count_correct_l213_213466


namespace trajectory_point_M_l213_213381

noncomputable def trajectory_equation (x : ℝ): ℝ := (1/4) * x^2 - 2

theorem trajectory_point_M (A B M : ℝ × ℝ)
    (hA : A = (0, -1))
    (hB : ∃ x : ℝ, B = (x, -3))
    (hM_parallel : ∃ x y : ℝ, (M = (x, y) ∧ (0, -3 - y) = (0, -3) ∧ ∥(0, -3)∥ * ∥(0, -3 - y)∥ = ∥M∥ * ∥M∥))
    (hDot : (λ (M : ℝ × ℝ), (-M.1, -1 - M.2)) M.1 M.2 • λ (M : ℝ × ℝ), (M.1, -2) = (λ (M : ℝ × ℝ), (0, -3 - M.2)) M.1 M.2 • (λ (M : ℝ × ℝ), (M.1, -2))) :
    M.snd = trajectory_equation M.fst := 
begin
    sorry
end

end trajectory_point_M_l213_213381


namespace find_angle_l213_213249

theorem find_angle : ∃ n : ℕ, 0 ≤ n ∧ n ≤ 180 ∧ (Real.cos (n * Real.pi / 180) = Real.cos (331 * Real.pi / 180)) ∧ n = 29 := 
by 
  use 29
  split
  · exact Nat.zero_le _
  split
  · exact Nat.le_of_lt_add_one (by norm_num)
  split
  · rw [Real.cos_of_nat_degree] at *
    rfl
  · rfl

end find_angle_l213_213249


namespace custom_op_4_3_l213_213907

-- Define the custom operation a * b
def custom_op (a b : ℤ) : ℤ := a^2 + a * b - b^2

-- State the theorem to be proven
theorem custom_op_4_3 : custom_op 4 3 = 19 := 
by
sorry

end custom_op_4_3_l213_213907


namespace roundness_of_250000_l213_213112

def roundness (n : ℕ) : ℕ :=
  let prime_factors := n.factorization
  prime_factors.values.sum

theorem roundness_of_250000 : roundness 250000 = 10 :=
by
  sorry

end roundness_of_250000_l213_213112


namespace five_digit_unique_combinations_l213_213343

theorem five_digit_unique_combinations : 
  ∃ n : ℕ, n = nat.factorial 5 / (nat.factorial 2 * nat.factorial 2) ∧ n = 30 :=
by
  use 30
  split
  · rw [nat.factorial, nat.factorial, nat.factorial]
    norm_num
  · rfl

end five_digit_unique_combinations_l213_213343


namespace three_digit_integers_from_set_l213_213342

theorem three_digit_integers_from_set :
  let digits := {2, 2, 3, 3, 4, 5, 6}
  let count_all_diff := 5 * 4 * 3 * 3.factorial
  let count_two_same_one_diff := 2 * 4 * (3.factorial / 2.factorial)
  let total := count_all_diff + count_two_same_one_diff
  count_all_diff + count_two_same_one_diff = 384 :=
by
  let digits := {2, 2, 3, 3, 4, 5, 6}
  let count_all_diff := 5 * 4 * 3 * 3.factorial
  let count_two_same_one_diff := 2 * 4 * (3.factorial / 2.factorial)
  let total := count_all_diff + count_two_same_one_diff
  show count_all_diff + count_two_same_one_diff = 384 from sorry

end three_digit_integers_from_set_l213_213342


namespace complex_power_problem_l213_213010

theorem complex_power_problem
  (z : ℂ)
  (h : z = -((1 - complex.i) / real.sqrt 2)) :
  z^100 + z^50 + 1 = -complex.i :=
sorry

end complex_power_problem_l213_213010


namespace min_of_quadratic_l213_213923

theorem min_of_quadratic :
  ∃ x : ℝ, (∀ y : ℝ, x^2 + 7 * x + 3 ≤ y^2 + 7 * y + 3) ∧ x = -7 / 2 :=
by
  sorry

end min_of_quadratic_l213_213923


namespace complex_distance_problem_l213_213754

-- Define the sets A and B
def A := {2, 2 * complex.i, -2, -2 * complex.i}
def B := {4}

-- Greatest distance calculation
noncomputable def greatest_distance (A B : set ℂ) : ℝ :=
  Sup { complex.abs (a - b) | a ∈ A, b ∈ B }

theorem complex_distance_problem : greatest_distance A B = 2 * real.sqrt 5 :=
by
  -- We add "sorry" to skip the proof.
  sorry

end complex_distance_problem_l213_213754


namespace parabola_directrix_l213_213247

theorem parabola_directrix (x : ℝ) (y : ℝ) (h : y = -4 * x ^ 2 - 3) : y = - 49 / 16 := sorry

end parabola_directrix_l213_213247


namespace fraction_doubled_l213_213737

variable (x y : ℝ)

theorem fraction_doubled (x y : ℝ) : 
  (x + y) ≠ 0 → (2 * x * 2 * y) / (2 * x + 2 * y) = 2 * (x * y / (x + y)) := 
by
  intro h
  sorry

end fraction_doubled_l213_213737


namespace min_a2_b2_l213_213324

theorem min_a2_b2 (a b : ℝ) (f : ℝ → ℝ) (h : ∃ (x ∈ set.Icc (real.exp 1) (real.exp 3)), a * (real.log x - 1) + (b + 1) * x = 0) :
  a^2 + b^2 ≥ real.exp 4 / (1 + real.exp 4) :=
sorry

end min_a2_b2_l213_213324


namespace composite_odd_nines_form_l213_213841

theorem composite_odd_nines_form (k : ℕ) : Nat.isComposite (10^(2 * k) - 9) := 
by
  sorry

end composite_odd_nines_form_l213_213841


namespace solve_quadratic_solve_inequality_system_l213_213932

theorem solve_quadratic :
  ∀ x : ℝ, x^2 - 6 * x + 5 = 0 ↔ x = 1 ∨ x = 5 :=
sorry

theorem solve_inequality_system :
  ∀ x : ℝ, (x + 3 > 0 ∧ 2 * (x + 1) < 4) ↔ (-3 < x ∧ x < 1) :=
sorry

end solve_quadratic_solve_inequality_system_l213_213932


namespace ceil_of_neg_frac_squared_l213_213201

-- Define the negated fraction
def neg_frac : ℚ := -7 / 4

-- Define the squared value of the negated fraction
def squared_value : ℚ := neg_frac ^ 2

-- Define the ceiling function applied to the squared value
def ceil_squared_value : ℤ := Int.ceil squared_value

-- Prove that the ceiling of the squared value is 4
theorem ceil_of_neg_frac_squared : ceil_squared_value = 4 := 
by sorry

end ceil_of_neg_frac_squared_l213_213201


namespace ceiling_of_square_of_neg_7_over_4_is_4_l213_213165

theorem ceiling_of_square_of_neg_7_over_4_is_4 : 
  Real.ceil ((-7 / 4 : Real) ^ 2) = 4 := by
  sorry

end ceiling_of_square_of_neg_7_over_4_is_4_l213_213165


namespace find_smaller_cube_side_length_l213_213524

noncomputable theory

def radius_of_sphere_with_cube_side (a : ℝ) : ℝ := (a * real.sqrt 3) / 2

def smaller_cube_side (R : ℝ) (d : ℝ) : ℝ :=
  let x := (-4 + real.sqrt (4^2 - 4 * 3 * (-4))) / (2 * 3) in x

theorem find_smaller_cube_side_length :
  let a := 2 in
  let R := radius_of_sphere_with_cube_side a in
  let d := a in
  smaller_cube_side R d = 2 / 3 :=
by
  sorry

end find_smaller_cube_side_length_l213_213524


namespace no_infinite_sequence_of_positive_integers_exists_infinite_sequence_of_positive_irrational_numbers_l213_213026

variables {a_n : ℕ → ℕ} {n : ℕ}

theorem no_infinite_sequence_of_positive_integers (h : ∀ n : ℕ, (a_n > 0) → ((a_{n+1} * a_{n+1}) ≥ (2 * a_n * a_{n+2}))) : 
  ¬ ∃ a_n : ℕ → ℕ, ∀ n : ℕ, (a_n > 0) ∧ ((a_{n+1} * a_{n+1}) ≥ (2 * a_n * a_{n+2})) := 
sorry

variables {a_n : ℕ → ℝ} {n : ℕ}

theorem exists_infinite_sequence_of_positive_irrational_numbers : 
  ∃ a_n : ℕ → ℝ, (∀ n : ℕ, ((0 < a_n) ∧ (a_n ≠ ⌊a_n⌋) ∧ (a_{n+1} * a_{n+1}) ≥ (2 * a_n * a_{n+2}))) :=
sorry

end no_infinite_sequence_of_positive_integers_exists_infinite_sequence_of_positive_irrational_numbers_l213_213026


namespace rectangular_prism_width_l213_213125

theorem rectangular_prism_width 
  (l : ℝ) (h : ℝ) (d : ℝ) (w : ℝ)
  (hl : l = 5) (hh : h = 7) (hd : d = 14) :
  d = Real.sqrt (l^2 + w^2 + h^2) → w = Real.sqrt 122 :=
by 
  sorry

end rectangular_prism_width_l213_213125


namespace circles_intersect_l213_213465

def circle1_eq (x y : ℝ) : Prop := x^2 + y^2 - 2 * x = 0
def circle2_eq (x y : ℝ) : Prop := x^2 + y^2 + 4 * y = 0

theorem circles_intersect :
  ∃ x y : ℝ, circle1_eq x y ∧ circle2_eq x y := by
  sorry

end circles_intersect_l213_213465


namespace natural_numbers_solution_l213_213244

theorem natural_numbers_solution (n : ℕ) (p q : ℕ)
    (prime_p : nat.prime p)
    (prime_q : nat.prime q)
    (eqn : p * (p + 1) + q * (q + 1) = n * (n + 1)) :
    n = 3 ∨ n = 6 :=
sorry

end natural_numbers_solution_l213_213244


namespace ceil_square_neg_fraction_l213_213176

theorem ceil_square_neg_fraction :
  let x := (-7) / 4
  let y := x^2
  let z := Real.ceil y
  z = 4 := 
by
  sorry

end ceil_square_neg_fraction_l213_213176


namespace ceil_square_of_neg_seven_fourths_l213_213231

/-- Evaluate the ceiling of the square of -7/4 --/
theorem ceil_square_of_neg_seven_fourths : (Int.ceil ((-7/4 : ℚ)^2 : ℚ) = 4) :=
sorry

end ceil_square_of_neg_seven_fourths_l213_213231


namespace total_number_of_subsets_of_A_l213_213334

variable (U : Set ℕ) (A : Set ℕ)
variable (h1 : U = {1, 2, 3, 4})
variable (h2 : U \ A = {2})

theorem total_number_of_subsets_of_A :
  (A = {1, 3, 4}) → ∃ n : ℕ, (number_of_subsets A = 8) :=
by
  sorry

end total_number_of_subsets_of_A_l213_213334


namespace nublian_total_words_l213_213423

-- Define the problem's constants and conditions
def nublian_alphabet_size := 6
def word_length_one := nublian_alphabet_size
def word_length_two := nublian_alphabet_size * nublian_alphabet_size
def word_length_three := nublian_alphabet_size * nublian_alphabet_size * nublian_alphabet_size

-- Define the total number of words
def total_words := word_length_one + word_length_two + word_length_three

-- Main theorem statement
theorem nublian_total_words : total_words = 258 := by
  sorry

end nublian_total_words_l213_213423


namespace servings_in_container_l213_213947

def convert_to_improper_fraction (whole : ℕ) (num : ℕ) (denom : ℕ) : ℚ :=
  whole + (num / denom)

def servings (container : ℚ) (serving_size : ℚ) : ℚ :=
  container / serving_size

def mixed_number (whole : ℕ) (num : ℕ) (denom : ℕ) : ℚ :=
  whole + (num / denom)

theorem servings_in_container : 
  let container := convert_to_improper_fraction 37 2 3
  let serving_size := convert_to_improper_fraction 1 1 2
  let expected_servings := mixed_number 25 1 9
  servings container serving_size = expected_servings :=
by 
  let container := convert_to_improper_fraction 37 2 3
  let serving_size := convert_to_improper_fraction 1 1 2
  let expected_servings := mixed_number 25 1 9
  sorry

end servings_in_container_l213_213947


namespace ceiling_of_square_of_neg_7_over_4_is_4_l213_213164

theorem ceiling_of_square_of_neg_7_over_4_is_4 : 
  Real.ceil ((-7 / 4 : Real) ^ 2) = 4 := by
  sorry

end ceiling_of_square_of_neg_7_over_4_is_4_l213_213164


namespace ellipse_focal_length_l213_213666

theorem ellipse_focal_length (a : ℝ) (h1 : a > 0) (h2 : 2 * real.sqrt (a^2 - 8) = 4) :
  a = 2 * real.sqrt 3 ∨ a = 2 :=
begin
  -- Initialization of the proof structure
  sorry
end

end ellipse_focal_length_l213_213666


namespace integral_value_l213_213634

theorem integral_value :
  ∫ x in -1..1, (sqrt (1 - x^2) + cos (2 * x - (π / 2))) = π / 2 := by
sory

end integral_value_l213_213634


namespace garage_sale_items_count_l213_213562

theorem garage_sale_items_count (prices : List ℕ) (radio_price : ℕ) :
  (∀ i j, i ≠ j → prices[i] ≠ prices[j]) →
  (∃ n, prices[n] = radio_price ∧ n = 9) →
  (∃ m, prices[m] = radio_price ∧ m = 35) →
  prices.length = 43 := 
by
  intros h1 h2 h3
  sorry

end garage_sale_items_count_l213_213562


namespace sheila_attends_picnic_l213_213850

theorem sheila_attends_picnic :
  let probRain := 0.30
  let probSunny := 0.50
  let probCloudy := 0.20
  let probAttendIfRain := 0.15
  let probAttendIfSunny := 0.85
  let probAttendIfCloudy := 0.40
  (probRain * probAttendIfRain + probSunny * probAttendIfSunny + probCloudy * probAttendIfCloudy) = 0.55 :=
by sorry

end sheila_attends_picnic_l213_213850


namespace bags_sold_on_third_day_l213_213546

theorem bags_sold_on_third_day 
  (initial_stock : ℕ)
  (day1_sold_percent : ℚ)
  (day1_restock_times : ℚ)
  (day2_sold_percent : ℚ)
  (day2_restock_increase_percent : ℚ)
  (day3_decrease_percent : ℚ)
  (day1_stock : ℚ)
  (day2_stock : ℚ)
  (day3_stock : ℚ)
  (bags_sold_day3 : ℚ)
  (h1 : initial_stock = 500)
  (h2 : day1_stock = initial_stock - (day1_sold_percent * initial_stock) + (day1_restock_times * (day1_sold_percent * initial_stock)))
  (h3 : day1_sold_percent = 25 / 100)
  (h4 : day1_restock_times = 2)
  (h5 : day2_stock = day1_stock - (day2_sold_percent * day1_stock) + ((1 + day2_restock_increase_percent) * (day2_sold_percent * day1_stock)))
  (h6 : day2_sold_percent = 30 / 100)
  (h7 : day2_restock_increase_percent = 50 / 100)
  (h8 : day3_stock = day2_stock - (day2_stock * day3_decrease_percent))
  (h9 : day3_decrease_percent = 20 / 100)
  (h10 : day3_stock = day2_stock - bags_sold_day3)
  (h11 : bags_sold_day3 = 144) : 
  bags_sold_day3 = 144 :=
begin
  sorry
end

end bags_sold_on_third_day_l213_213546


namespace inequality_proof_l213_213411

theorem inequality_proof (x y z t : ℝ) (h₁ : x >= 0) (h₂ : y >= 0) (h₃ : z >= 0) (h₄ : t >= 0) (h₅ : x + y + z + t = 2) :
  √(x^2 + z^2) + √(x^2 + 1) + √(z^2 + y^2) + √(y^2 + t^2) + √(t^2 + 4) ≥ 5 :=
begin
  sorry
end

end inequality_proof_l213_213411


namespace probability_at_least_two_same_l213_213827

theorem probability_at_least_two_same (n : ℕ) (H : n = 8) : 
  (∃ i j, (1 ≤ i ∧ i ≤ n) ∧ (1 ≤ j ∧ j ≤ n) ∧ i ≠ j ∧ ∀ (x : ℕ), x ∈ {i, j}) :=
by
  sorry

end probability_at_least_two_same_l213_213827


namespace inscribed_circle_in_convex_polygon_l213_213948

theorem inscribed_circle_in_convex_polygon
  (P : Polygon)
  (h_convex : P.is_convex)
  (h_similar : ∀ Q : Polygon, Q.is_similar (P.translate_outward 1 P)) :
  ∃ O : Point, inscribed_circle O P :=
sorry

end inscribed_circle_in_convex_polygon_l213_213948


namespace pizza_circumference_ratio_l213_213535

/-- Given a pizza of diameter 30 cm and circumference 94.2 cm,
prove that the circumference is 3.14 times greater than the diameter. -/
theorem pizza_circumference_ratio:
  ∀ (d c : ℝ), d = 30 → c = 94.2 → (c / d = 3.14) :=
by
  intros d c h₁ h₂
  rw [h₁, h₂]
  norm_num
  sorry

end pizza_circumference_ratio_l213_213535


namespace initial_weight_of_fish_l213_213940

theorem initial_weight_of_fish (B F : ℝ) 
  (h1 : B + F = 54) 
  (h2 : B + F / 2 = 29) : 
  F = 50 := 
sorry

end initial_weight_of_fish_l213_213940


namespace num_factors_2012_l213_213718

theorem num_factors_2012 : (Nat.factors 2012).length = 6 := by
  sorry

end num_factors_2012_l213_213718


namespace ceiling_of_square_frac_l213_213218

theorem ceiling_of_square_frac : 
  let x : ℚ := -7 / 4
  let y : ℚ := x^2
  in Int.ceil y = 4 := 
by 
  let x : ℚ := -7 / 4
  let y : ℚ := x^2
  exact sorry

end ceiling_of_square_frac_l213_213218


namespace boxes_left_for_Sonny_l213_213853

def initial_boxes : ℕ := 45
def boxes_given_to_brother : ℕ := 12
def boxes_given_to_sister : ℕ := 9
def boxes_given_to_cousin : ℕ := 7

def total_given_away : ℕ := boxes_given_to_brother + boxes_given_to_sister + boxes_given_to_cousin

def remaining_boxes : ℕ := initial_boxes - total_given_away

theorem boxes_left_for_Sonny : remaining_boxes = 17 := by
  sorry

end boxes_left_for_Sonny_l213_213853


namespace coefficient_x3_expansion_l213_213624

theorem coefficient_x3_expansion :
  let T_r (n r : ℕ) := (nat.choose n r) * ((-1)^r : ℤ)
  ∃ c : ℤ, c = -26 ∧ 
    ∀ (n : ℕ), 
      n = 6 → 
      let f := (1 + x^(-2)) * (1 - x)^n in
      coeff (expansion f) x 3 = c :=
by sorry

end coefficient_x3_expansion_l213_213624


namespace circumference_base_of_cone_l213_213513

-- Define the given conditions
def radius_circle : ℝ := 6
def angle_sector : ℝ := 300

-- Define the problem to prove the circumference of the base of the resulting cone in terms of π
theorem circumference_base_of_cone :
  (angle_sector / 360) * (2 * π * radius_circle) = 10 * π := by
sorry

end circumference_base_of_cone_l213_213513


namespace boxes_amount_l213_213528

/-- 
  A food company has 777 kilograms of food to put into boxes. 
  If each box gets a certain amount of kilograms, they will have 388 full boxes.
  Prove that each box gets 2 kilograms of food.
-/
theorem boxes_amount (total_food : ℕ) (boxes : ℕ) (kilograms_per_box : ℕ) 
  (h_total : total_food = 777)
  (h_boxes : boxes = 388) :
  total_food / boxes = kilograms_per_box :=
by {
  -- Skipped proof
  sorry 
}

end boxes_amount_l213_213528


namespace enclosed_area_of_curve_l213_213865

def radius_of_arc (arc_length : ℝ) : ℝ :=
  arc_length / (2 * π)

def area_of_pentagon (side_length : ℝ) : ℝ :=
  (1 / 4) * sqrt (5 * (5 + 2 * sqrt 5)) * side_length^2

def area_of_sector (radius : ℝ) (sector_angle : ℝ) : ℝ :=
  (sector_angle / (2 * π)) * π * radius^2

def total_area (num_sectors : ℕ) (sector_area : ℝ) : ℝ :=
  num_sectors * sector_area

theorem enclosed_area_of_curve :
  let arc_length := π / 2
  let pentagon_side_length := 1
  let radius := radius_of_arc arc_length
  let pentagon_area := area_of_pentagon pentagon_side_length
  let sector_area := area_of_sector radius arc_length
  let sectors_area := total_area 10 sector_area
  (pentagon_area + sectors_area) = (1 / 4 * sqrt (5 * (5 + 2 * sqrt 5)) + 5 * π^2 / 4) :=
by
  sorry

end enclosed_area_of_curve_l213_213865


namespace minimum_floor_x_l213_213665

-- Define the edge lengths of the tetrahedron
def edge_lengths : set ℝ := {4, 7, 20, 22, 28, x}

-- Define the ranges coming from triangle inequalities
def range_x := { x : ℝ | 8 < x ∧ x < 11 }

-- State the goal to prove the smallest integer for the floor of x
theorem minimum_floor_x (x : ℝ) (h : 8 < x ∧ x < 11) : int.floor x = 8 :=
by
  sorry

end minimum_floor_x_l213_213665


namespace sqrt_cos_product_l213_213139

theorem sqrt_cos_product :
  sqrt ((2 - (Real.cos (π / 9))^2) * (2 - (Real.cos (2 * π / 9))^2) * (2 - (Real.cos (3 * π / 9))^2)) = Real.sqrt 377 / 8 :=
by
  sorry

end sqrt_cos_product_l213_213139


namespace nat_numbers_l213_213137

theorem nat_numbers (n : ℕ) (h1 : n ≥ 2) (h2 : ∃a b : ℕ, a * b = n ∧ ∀ c : ℕ, 1 < c ∧ c ∣ n → a ≤ c ∧ n = a^2 + b^2) : 
  n = 5 ∨ n = 8 ∨ n = 20 :=
by
  sorry

end nat_numbers_l213_213137


namespace find_area_l213_213421

noncomputable def area_figure (A B : ℝ) (r : ℝ) : ℝ :=
  let d := 4 in
  let radius := 2 in
  let semicircle_area := (π * radius^2) / 2 in
  let circle_area := π * radius^2 in
  let rectangle_area := d * radius in
  rectangle_area

theorem find_area:
  ∀ (A B : ℝ), (B - A = 4) → (area_figure A B 2 = 8) :=
by
  intros A B h
  rw area_figure
  sorry

end find_area_l213_213421


namespace number_of_robots_l213_213955

-- Define the constants involved in the problem
def minutes_per_battery : Nat := 6 + 9
def total_minutes : Nat := 5 * 60
def total_batteries : Nat := 200
def batteries_per_robot (R : Nat) : Nat := total_minutes / minutes_per_battery * R

-- The theorem statement
theorem number_of_robots : ∃ R : Nat, batteries_per_robot R = total_batteries ∧ R = 10 := by
  -- setting up the equation based on the given solution steps
  exists.intro 10
  sorry

end number_of_robots_l213_213955


namespace nursery_school_students_l213_213426

theorem nursery_school_students (S : ℕ)
  (h1 : ∃ x, x = S / 10)
  (h2 : 20 + (S / 10) = 25) : S = 50 :=
by
  sorry

end nursery_school_students_l213_213426


namespace ceil_square_of_neg_seven_fourths_l213_213230

/-- Evaluate the ceiling of the square of -7/4 --/
theorem ceil_square_of_neg_seven_fourths : (Int.ceil ((-7/4 : ℚ)^2 : ℚ) = 4) :=
sorry

end ceil_square_of_neg_seven_fourths_l213_213230


namespace trapezoid_area_l213_213096

theorem trapezoid_area 
  (a b h c : ℝ) 
  (ha : 2 * 0.8 * a + b = c)
  (hb : c = 20) 
  (hc : h = 0.6 * a) 
  (hd : b + 1.6 * a = 20)
  (angle_base : ∃ θ : ℝ, θ = arcsin 0.6)
  : 
  (1 / 2) * (b + c) * h = 72 :=
sorry

end trapezoid_area_l213_213096


namespace product_of_real_roots_l213_213240

theorem product_of_real_roots (x : ℝ) (h : x^5 = 100) : x = 10^(2/5) := by
  sorry

end product_of_real_roots_l213_213240


namespace find_OP_squared_l213_213041

noncomputable theory

def circle_radius := 24
def chord_AB_length := 40
def chord_CD_length := 18
def midpoint_distance := 15

theorem find_OP_squared
  (O P E F : Type)
  (radius : ℝ := circle_radius)
  (AB CD : ℝ := chord_AB_length) (mid_dist : ℝ := midpoint_distance)
  (OE OF : ℝ)
  (A B C D : O) (P : O)
  (EP FP : ℝ)
  (intersect : AB ∩ CD = {P})
  (mid1 : midpoint A B = E)
  (mid2 : midpoint C D = F)
  (dist_midpoints : dist (midpoint A B) (midpoint C D) = mid_dist) :
  OP^2 = 448 :=
sorry

end find_OP_squared_l213_213041


namespace ellipse_problem_l213_213686

noncomputable def A : ℝ × ℝ := (0, -1)
def ellipse_eq (x y : ℝ) : Prop := x^2 / 3 + y^2 = 1

def right_focus := sqrt 2 -- from the equation |c + 2√2| / √2 = 3

def verify_focus_distance : Prop := abs (right_focus + 2 * sqrt 2) / sqrt 2 = 3

def no_such_m (m : ℝ) : Prop :=
  ¬ ∃ x1 x2 y1 y2 : ℝ, 
    (x1 * x1 / 3 + y1 * y1 = 1) ∧ (x2 * x2 / 3 + y2 * y2 = 1) ∧ 
    (y1 = x1 + m) ∧ (y2 = x2 + m) ∧ 
    (x1^2 + (y1 + 1)^2 = x2^2 + (y2 + 1)^2)

theorem ellipse_problem :
  (∀ x y : ℝ, ellipse_eq x y ↔ (x = 0 ∧ y = -1) ∨ x^2 / 3 + y^2 = 1) ∧
  verify_focus_distance ∧
  ∀ m : ℝ, no_such_m m :=
by sorry

end ellipse_problem_l213_213686


namespace find_a_for_direction_vector_l213_213529

/-- The direction vector for a line passing through two points and scaling it to get a specific component. -/
theorem find_a_for_direction_vector 
  (a : ℚ)
  (p₁ p₂ : ℚ × ℚ)
  (v : ℚ × ℚ)
  (h1 : p₁ = (-3, 7))
  (h2 : p₂ = (2, -1))
  (direction : v = (p₂.1 - p₁.1, p₂.2 - p₁.2))
  (scaling : ∃ k : ℚ, k * v = (a, -2)) :
  a = 5 / 4 :=
by
  sorry

end find_a_for_direction_vector_l213_213529


namespace area_geometric_mean_l213_213132

-- Definitions
def triangles_geometry (A B C D M C1: Type) [triangle : Triangle A B C] : Prop :=
  (AB as hypotenuse) -- AB is the hypotenuse
  ∧ (right_triangle : RightTriangle A B D) -- ABD forms a right triangle
  ∧ (D on altitude : On D C C1) -- D is on the altitude CC1 of ABC
  ∧ (orthocenter M : Orthocenter M A B C). -- M is the orthocenter of ABC

-- Problem statement
theorem area_geometric_mean (A B C D M C1: Type) [triangle : Triangle A B C]
  [right_triangle : RightTriangle A B D]
  [D_on_altitude : On D C C1]
  [orthocenter M : Orthocenter M A B C] :
  area (triangle_of_sides A B D) = sqrt (area (triangle_of_sides A B C) * area (triangle_of_sides A B M)) :=
sorry

end area_geometric_mean_l213_213132


namespace savings_percentage_correct_l213_213961

theorem savings_percentage_correct :
  let original_price_jacket := 120
  let original_price_shirt := 60
  let original_price_shoes := 90
  let discount_jacket := 0.30
  let discount_shirt := 0.50
  let discount_shoes := 0.25
  let total_original_price := original_price_jacket + original_price_shirt + original_price_shoes
  let savings_jacket := original_price_jacket * discount_jacket
  let savings_shirt := original_price_shirt * discount_shirt
  let savings_shoes := original_price_shoes * discount_shoes
  let total_savings := savings_jacket + savings_shirt + savings_shoes
  let percentage_savings := (total_savings / total_original_price) * 100
  percentage_savings = 32.8 := 
by 
  sorry

end savings_percentage_correct_l213_213961


namespace solution_set_inequality_l213_213683

variable {f : ℝ → ℝ}

-- Given conditions
axiom decreasing_f : ∀ x y : ℝ, x < y → f(x) > f(y)
axiom point_A : f 3 = -1
axiom point_B : f 0 = 1

-- Definition of the solution set
def solution_set : set ℝ := {x : ℝ | x > 1/e ∧ x < e^2}

-- The proof problem
theorem solution_set_inequality : {x : ℝ | |f (1 + real.log x)| < 1} = solution_set := 
sorry

end solution_set_inequality_l213_213683


namespace no_linear_factor_with_integer_coefficients_l213_213997

def expression (x y z : ℤ) : ℤ :=
  x^2 - y^2 - z^2 + 3 * y * z + x + 2 * y - z

theorem no_linear_factor_with_integer_coefficients:
  ¬ ∃ (a b c d : ℤ), a ≠ 0 ∧ 
                      ∀ (x y z : ℤ), 
                        expression x y z = a * x + b * y + c * z + d := by
  sorry

end no_linear_factor_with_integer_coefficients_l213_213997


namespace profit_percentage_approx_l213_213949

-- Definitions as per conditions in the problem
def SP := 850
def Profit := 205
def CP := SP - Profit

-- Statement of the proof problem
theorem profit_percentage_approx : 
  approx ((Profit : ℝ) / (CP : ℝ) * 100) 31.78 :=
by
  sorry

end profit_percentage_approx_l213_213949


namespace probability_two_dice_same_number_l213_213826

theorem probability_two_dice_same_number (n : ℕ) (sides : ℕ) (h_n : n = 8) (h_sides : sides = 6):
  (∃ (prob : ℝ), prob = 1) :=
by
  sorry

end probability_two_dice_same_number_l213_213826


namespace film_radius_l213_213799

theorem film_radius 
  (thickness : ℝ)
  (container_volume : ℝ)
  (r : ℝ)
  (H1 : thickness = 0.25)
  (H2 : container_volume = 128) :
  r = Real.sqrt (512 / Real.pi) :=
by
  -- Placeholder for proof
  sorry

end film_radius_l213_213799


namespace no_20_digit_number_starting_with_11111111111_is_a_perfect_square_l213_213147

theorem no_20_digit_number_starting_with_11111111111_is_a_perfect_square :
  ¬ ∃ (n : ℤ), (10^19 ≤ n ∧ n < 10^20 ∧ (11111111111 * 10^9 ≤ n ∧ n < 11111111112 * 10^9) ∧ (∃ k : ℤ, n = k^2)) :=
by
  sorry

end no_20_digit_number_starting_with_11111111111_is_a_perfect_square_l213_213147


namespace probability_same_number_l213_213838

theorem probability_same_number (n k : ℕ) (h₁ : n = 8) (h₂ : k = 6) : 
  (∃ m : ℝ, 0 ≤ m ∧ m ≤ 1 ∧ m = 1) := by
  sorry

end probability_same_number_l213_213838


namespace ceiling_of_square_of_neg_7_over_4_is_4_l213_213162

theorem ceiling_of_square_of_neg_7_over_4_is_4 : 
  Real.ceil ((-7 / 4 : Real) ^ 2) = 4 := by
  sorry

end ceiling_of_square_of_neg_7_over_4_is_4_l213_213162


namespace deepak_age_l213_213017

variable (A D : ℕ)

theorem deepak_age (h1 : A / D = 2 / 3) (h2 : A + 5 = 25) : D = 30 :=
sorry

end deepak_age_l213_213017


namespace final_cost_and_gain_percentage_l213_213965

noncomputable def tea_costs := 
  (A B C D : ℝ) (H_A : A = 18) (H_B : B = 20) (H_C : C = 22) (H_D : D = 24)

noncomputable def blend_ratios := 
  (H1 : (5 / 8) * A + (3 / 8) * B = 18.75)
  (H2 : (3 / 5) * A + (2 / 5) * C = 19.60)
  (H3 : (2 / 3) * B + (1 / 3) * D = 21.33)
  (H_final : (3 / 4) * 18.75 + (1 / 4) * 21.33 = 19.395)

theorem final_cost_and_gain_percentage 
    (A B C D : ℝ) (H_A : A = 18) (H_B : B = 20) (H_C : C = 22) (H_D : D = 24)
    (H1 : (5 / 8) * A + (3 / 8) * B = 18.75)
    (H2 : (3 / 5) * A + (2 / 5) * C = 19.60)
    (H3 : (2 / 3) * B + (1 / 3) * D = 21.33)
    (H_final : (3 / 4) * 18.75 + (1 / 4) * 21.33 = 19.395) :
    let selling_price := 30 in
    let cost_price := 19.395 in
    let gain := selling_price - cost_price in
    let gain_percentage := (gain / cost_price) * 100 in
  gain_percentage = 54.67 := 
by
  sorry

end final_cost_and_gain_percentage_l213_213965


namespace area_of_N1N2N3_l213_213384

variables (A B C D E F N1 N2 N3 : Type)

-- Define the segment lengths as given in the conditions
variables (CD AE BF : ℝ)
variable (K : ℝ) -- Area of triangle ABC

-- Conditions
axiom h1 : CD = 0.5 * (distance B C)
axiom h2 : AE = 0.5 * (distance A E)
axiom h3 : BF = 0.5 * (distance B F)

-- Define areas 
def area_ABC : ℝ := K
def area_N1N2N3 : ℝ := area_ABC / 4

-- Statement of the theorem
theorem area_of_N1N2N3 : area (triangle N1 N2 N3) = area_ABC / 4 :=
sorry

end area_of_N1N2N3_l213_213384


namespace ceil_square_of_neg_seven_fourths_l213_213234

/-- Evaluate the ceiling of the square of -7/4 --/
theorem ceil_square_of_neg_seven_fourths : (Int.ceil ((-7/4 : ℚ)^2 : ℚ) = 4) :=
sorry

end ceil_square_of_neg_seven_fourths_l213_213234


namespace closest_perfect_square_to_315_l213_213005

theorem closest_perfect_square_to_315 : ∃ n : ℤ, n^2 = 324 ∧
  (∀ m : ℤ, m ≠ n → (abs (315 - m^2) > abs (315 - n^2))) := 
sorry

end closest_perfect_square_to_315_l213_213005


namespace eccentricity_range_l213_213678

noncomputable def ellipse (a b : ℝ) := {p : ℝ × ℝ | (p.1 ^ 2) / (a ^ 2) + (p.2 ^ 2) / (b ^ 2) = 1}

def foci (a b : ℝ) : ℝ × ℝ := (sqrt (a ^ 2 - b ^ 2), 0) -- Assuming standard positioning on the x-axis

def distance (p1 p2 : ℝ × ℝ) : ℝ := sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

theorem eccentricity_range (a b : ℝ) (h : a > b > 0) (A : ℝ × ℝ)
  (hA : A ∈ ellipse a b) (h_dist : 2 * distance A (foci a b) - 3 * distance A ((-foci a b).fst, foci a b).snd = a) :
  ∃ e, e ∈ set.Ico (2 / 5 : ℝ) 1 :=
sorry

end eccentricity_range_l213_213678


namespace largest_difference_33000_l213_213583

-- Definitions and conditions
def S_estimate : ℕ := 75000
def C_estimate : ℕ := 85000
def pct1 : ℝ := 0.15
def pct2 : ℝ := 0.12

def S_min := S_estimate * (1 - pct1)
def S_max := S_estimate * (1 + pct1)
def C_min := C_estimate / (1 + pct2)
def C_max := C_estimate / (1 - pct2)

def largest_possible_difference := C_max - S_min

-- Statement of the problem
theorem largest_difference_33000 :
  largest_possible_difference.to_nat = 33000 :=
sorry

end largest_difference_33000_l213_213583


namespace meaningful_fraction_range_l213_213350

theorem meaningful_fraction_range (x : ℝ) : 
  (∃ y : ℝ, y = (sqrt x) / (sqrt (1 - x))) → (0 ≤ x ∧ x < 1) :=
by
  sorry

end meaningful_fraction_range_l213_213350


namespace find_hair_color_count_l213_213568

-- Define the given conditions
variables (total_items : ℕ) (slippers_count : ℕ) (slippers_cost : ℝ)
          (lipstick_count : ℕ) (lipstick_cost : ℝ) (total_paid : ℝ)
          (hair_color_cost : ℝ)

-- Assign values to the conditions
def betty_conditions : Prop :=
  total_items = 18 ∧
  slippers_count = 6 ∧
  slippers_cost = 2.5 ∧
  lipstick_count = 4 ∧
  lipstick_cost = 1.25 ∧
  total_paid = 44 ∧
  hair_color_cost = 3

-- The proof statement to determine the number of hair colors ordered
def betty_ordered_hair_color (hair_color_count : ℕ) : Prop :=
  betty_conditions ∧
  (6 * slippers_cost + 4 * lipstick_cost + hair_color_count * hair_color_cost = total_paid) ∧
  (total_items = slippers_count + lipstick_count + hair_color_count)

-- The correct answer
theorem find_hair_color_count : ∃ (hair_color_count : ℕ), betty_ordered_hair_color hair_color_count ∧ hair_color_count = 8 :=
by {
  -- Proof would be filled in here, but we use sorry to focus on the setup
  sorry
}

end find_hair_color_count_l213_213568


namespace part_one_part_two_l213_213795

noncomputable def f (x a : ℝ) : ℝ := |x - 5/2| + |x - a|

theorem part_one (x : ℝ) : (∀ x : ℝ, a = -1/2 → log (f x a) > 1) := by
  intros
  sorry

theorem part_two (x : ℝ) : (∀ x : ℝ, ∀ a : ℝ, f x a ≥ a) → a ≤ 5/4 := by
  intros
  sorry

end part_one_part_two_l213_213795


namespace intersection_of_circles_l213_213432

-- Given conditions as definitions in Lean 4
def radius_of_small_circles : ℝ := 1 / 14

-- We need to define the existence of an intersection between a given large circle and one of the small circles centered at integer points
theorem intersection_of_circles (O : ℝ × ℝ) (R : ℝ) 
  (hR : R = 100) :
  ∃ (m n : ℤ), 
    let center_int_pt := ((m : ℝ), (n : ℝ)) in
    (dist O center_int_pt) ≤ R + radius_of_small_circles :=
begin
  sorry
end

end intersection_of_circles_l213_213432


namespace probability_each_team_loses_and_wins_at_least_one_l213_213609

theorem probability_each_team_loses_and_wins_at_least_one (n : ℕ) (p : ℚ)
  (h1 : n = 8)
  (h2 : p = 1 / 2) : 
  (∃ k : ℚ, k = 903 / 1024 ∧
  k = 1 - (8 * (2^22 - 7 * 2^15) / 2^28)) :=
by {
  use 903 / 1024,
  split;
  sorry
}

end probability_each_team_loses_and_wins_at_least_one_l213_213609


namespace edge_labeling_possible_l213_213452

-- Assume a connected graph G with k edges.
noncomputable def connected_graph (G : Type) [graph G] : Prop :=
  ∀ v w : G, ∃ (path : list (Edge G)), path.is_path v w

noncomputable def has_k_edges (G : Type) [graph G] (k : ℕ) : Prop :=
  Fintype.card (Edge G) = k

-- Define the labeling of edges such that the gcd condition at each vertex holds.
def label_gcd_condition {G : Type} [graph G] (label : Edge G → ℕ) : Prop :=
  ∀ v : G, (∃ e₁ e₂ : Edge G, e₁ ≠ e₂ ∧ e₁ ∈ adj v ∧ e₂ ∈ adj v) → 
  ∃ e₁ e₂ : Edge G, e₁ ∈ adj v ∧ e₂ ∈ adj v ∧ Int.gcd (label e₁) (label e₂) = 1

-- The main theorem.
theorem edge_labeling_possible (G : Type) [graph G] (k : ℕ) 
  (h1 : connected_graph G) (h2 : has_k_edges G k) :
  ∃ label : Edge G → ℕ, 
  (∀ e : Edge G, label e ∈ finset.range k) ∧ label_gcd_condition label :=
sorry

end edge_labeling_possible_l213_213452


namespace find_h4_l213_213409

noncomputable def f (x : ℝ) : ℝ := x^3 - 2 * x - 5
noncomputable def h (x : ℝ) : ℝ := sorry  -- Placeholder for the cubic polynomial h

axiom root_condition (a b c : ℝ) : 
  (f(x) = (x - a) * (x - b) * (x - c)) ∧ 
  (h(x) = B*(x - a^2)*(x - b^2)*(x - c^2))

axiom h_at_1 : h 1 = 2

theorem find_h4 (a b c B : ℝ): 
  f(1) = 0 → 
  h(1) = 2 → 
  (h(x) = B*(x - a^2)*(x - b^2)*(x - c^2)) →
  (a^2 * b^2 * c^2 = 25) → 
  h 4 = 9 :=
by
  sorry

end find_h4_l213_213409


namespace problem_solution_l213_213697

theorem problem_solution :
    (∀ (a : ℕ → ℝ),
        ({(x^2 + 2x + 2)}^5 = a 0 + a 1 * (x + 1) + a 2 * (x + 1)^2 + a 3 * (x + 1)^3
        + a 4 * (x + 1)^4 + a 5 * (x + 1)^5 + a 6 * (x + 1)^6 + a 7 * (x + 1)^7
        + a 8 * (x + 1)^8 + a 9 * (x + 1)^9 + a 10 * (x + 1)^10) →
        (∑ n in Icc 1 10, a n = 31 ∧ ∑ n in Icc 1 10, n * a n = 160)) :=
by
  sorry

end problem_solution_l213_213697


namespace product_sequence_equals_8_l213_213571

theorem product_sequence_equals_8 :
  (1 + 1/2) * (1 + 1/3) * (1 + 1/4) * (1 + 1/5) * (1 + 1/6) * (1 + 1/7) = 8 := 
by
  sorry

end product_sequence_equals_8_l213_213571


namespace cost_of_bananas_l213_213902

theorem cost_of_bananas (A B : ℕ) (h1 : 2 * A + B = 7) (h2 : A + B = 5) : B = 3 :=
by
  sorry

end cost_of_bananas_l213_213902


namespace perpendicular_vectors_collinear_tan_alpha_l213_213713

-- Definitions based on given conditions
def a : ℝ × ℝ := (2, 1)
def b (α : ℝ) : ℝ × ℝ := (Real.sin α, 2 * Real.cos α)

-- Proof for question (1)
theorem perpendicular_vectors (α : ℝ) (hα : α = 3 * Real.pi / 4) : a.1 * (b α).1 + a.2 * (b α).2 = 0 := 
by 
  have hb : b α = (Real.sin (3 * Real.pi / 4), 2 * Real.cos (3 * Real.pi / 4)), by rw hα
  sorry

-- Proof for question (2)
theorem collinear_tan_alpha (α : ℝ) (hcoll : ∃ k : ℝ, a = k • (b α)) : Real.tan α = 4 := 
by 
  sorry

end perpendicular_vectors_collinear_tan_alpha_l213_213713


namespace count_primes_f_l213_213404

def sum_of_divisors (n : ℕ) : ℕ :=
  (Finset.range (n + 1)).filter (λ d => n % d = 0).sum

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m ∈ Finset.range (n - 2) + 2, m ∣ n → m = 1 ∨ m = n

theorem count_primes_f (N : ℕ) (h : 1 ≤ N ∧ N ≤ 50) : 
  (Nat.card { n | 1 ≤ n ∧ n ≤ N ∧ is_prime (sum_of_divisors n) }) = 5 := 
by
  sorry

end count_primes_f_l213_213404


namespace correct_props_l213_213316

-- Definitions based on the conditions
def proposition1 (f : ℝ → ℝ) (a : ℝ) : Prop := 
  ∀ x₁ x₂ : ℝ, f(x₁) = f(x₂) → x₁ = x₂ ∧ x₁ = a ∧ x₂ = a

def proposition2 (x : ℝ) : Prop := 
  ∀ x : ℝ, (log x^2) = 2 * (log x)

def proposition3 (x y : ℝ) : Prop := 
  ∀ x : ℝ, x > 4 → 2^x > x^2

def proposition4 (f : ℝ → ℝ) (a b : ℝ) : Prop := 
  ∀ f : ℝ → ℝ, interval (a b) → f(a) * f(b) < 0 → continuous_on f [a, b] → ∃ c, c ∈ (a, b) ∧ f(c) = 0

def proposition5 (x₁ x₂ : ℝ) : Prop := 
  x₁ + log10 x₁ = 5 ∧ x₂ + 10^x₂ = 5 → x₁ + x₂ = 5

-- Proposition list
def correct_propositions : list Prop :=
  [proposition3, proposition5]

theorem correct_props :
  correct_propositions = [proposition3, proposition5] :=
by sorry

end correct_props_l213_213316


namespace length_LN_l213_213378

def right_triangle_XYZ (XY XZ : ℝ) : Prop :=
  XY = 7 ∧ XZ = 4 ∧ XY ^ 2 = (XZ ^ 2) + YZ ^ 2

def angle_bisector_X1 (XZ YZ X1Z X1Y : ℝ) : Prop :=
  X1Z + X1Y = YZ ∧ (X1Z / X1Y) = (XZ / YZ)

def triangle_LMN (X1Y X1Z LM LN : ℝ) : Prop :=
  LM = X1Z ∧ X1Y = sqrt (LM ^ 2 + LN ^ 2)

theorem length_LN : ∃ (LN : ℝ), 
  right_triangle_XYZ 7 4 ∧ 
  ∀ YZ, YZ = sqrt (7 ^ 2 - 4 ^ 2) →
  ∀ X1Z X1Y, angle_bisector_X1 4 YZ X1Z X1Y →
  ∀ LM, triangle_LMN X1Y X1Z LM LN →
  LN = 2 * sqrt 5 :=
  sorry

end length_LN_l213_213378


namespace closest_perfect_square_l213_213007

theorem closest_perfect_square (n : ℕ) (h : n = 315) : ∃ k : ℕ, k^2 = 324 ∧ ∀ m : ℕ, m^2 ≠ 315 ∨ abs (n - m^2) > abs (n - k^2) :=
by
  use 18
  sorry

end closest_perfect_square_l213_213007


namespace log_8_512_eq_3_l213_213239

noncomputable def log_8_512 : ℝ :=
  log 8 512

theorem log_8_512_eq_3 : log_8_512 = 3 :=
by
  have h : 8^3 = 512 := by norm_num
  rw [←logb_pow 512 8 3]
  exact log_b_eq_log_8_512 h
  sorry

end log_8_512_eq_3_l213_213239


namespace enclosed_area_is_correct_l213_213874

-- Given conditions
def length_of_arc : ℝ := π / 2
def side_length_of_octagon : ℝ := 3
def number_of_arcs : ℕ := 16

-- Define the radius calculated from the arc length
def radius := 1  -- Based on the solution step calculation

-- Define the area of one circular sector
def area_of_sector := (π * radius^2) / 4  -- Each sector is π/2 radians of a circle with radius 1

-- Total area of the circular sectors
def total_area_of_sectors := area_of_sector * number_of_arcs

-- Area of a regular octagon with side length 3
def area_of_octagon := 2 * (1 + Real.sqrt 2) * (side_length_of_octagon ^ 2)

-- The final combined area, taking overlaps into consideration
def enclosed_area := π + 54 * Real.sqrt 2

-- The theorem stating the problem
theorem enclosed_area_is_correct :
  enclosed_area = π + 54 * Real.sqrt 2 :=
by
  -- This is the statement only, the proof is not required
  sorry

end enclosed_area_is_correct_l213_213874


namespace ceil_square_neg_seven_over_four_l213_213212

theorem ceil_square_neg_seven_over_four : 
  let x := - (7 / 4 : ℚ) in
  ⌈x^2⌉ = 4 :=
by
  let x := - (7 / 4 : ℚ)
  sorry

end ceil_square_neg_seven_over_four_l213_213212


namespace eccentricity_of_ellipse_l213_213288

noncomputable def ellipse_eccentricity (a b : ℝ) (h1 : a > b) (h2 : b > 0)
  (h3 : ∃ P Q : ℝ × ℝ, P ≠ Q ∧ (P.1 / a)^2 + (P.2 / b)^2 = 1 ∧ (Q.1 / a)^2 + (Q.2 / b)^2 = 1 ∧
  (P.1 = 0) ∧ (Q.1^2 + Q.2^2 = a^2) ∧ (∃ A : ℝ × ℝ, A.1 = a ∧ P ≠ A ∧ (P.1 - A.1)^2 + P.2^2 = (a/2)^2)) : ℝ :=
  let b2_over_a2 := (b^2) / (a^2) in
  let ecc := Real.sqrt(1 - b2_over_a2) in
  ecc

theorem eccentricity_of_ellipse (a b : ℝ) (h1 : a > b) (h2 : b > 0)
  (h3 : ∃ P Q : ℝ × ℝ, P ≠ Q ∧ (P.1 / a)^2 + (P.2 / b)^2 = 1 ∧ (Q.1 / a)^2 + (Q.2 / b)^2 = 1 ∧
  (P.1 = 0) ∧ (Q.1^2 + Q.2^2 = a^2) ∧ (∃ A : ℝ × ℝ, A.1 = a ∧ P ≠ A ∧ (P.1 - A.1)^2 + P.2^2 = (a/2)^2)) :
  ellipse_eccentricity a b h1 h2 h3 = (2 * Real.sqrt(5)) / 5 :=
begin
  -- proof would go here
  sorry
end

end eccentricity_of_ellipse_l213_213288


namespace single_elimination_matches_l213_213935

theorem single_elimination_matches (n : ℕ) (h : n = 30) : (n - 1) = 29 :=
by 
  rw h
  simp
  exact nat.sub_self 1

end single_elimination_matches_l213_213935


namespace strictly_increasing_0_to_e_l213_213586

noncomputable def ln (x : ℝ) : ℝ := Real.log x

noncomputable def f (x : ℝ) : ℝ := ln x / x

theorem strictly_increasing_0_to_e :
  ∀ x : ℝ, 0 < x ∧ x < Real.exp 1 → 0 < (1 - ln x) / (x^2) :=
by
  sorry

end strictly_increasing_0_to_e_l213_213586


namespace keiko_speed_l213_213394

theorem keiko_speed {length width time_inner time_outer perimeter_inner perimeter_outer diff_time s: ℝ} 
  (h1 : length = 100) 
  (h2 : width = 50) 
  (h3 : time_inner = 200) 
  (h4 : perimeter_inner = 2 * (length + width)) 
  (h5 : perimeter_outer = 2 * (length + 10 + (width + 10))) 
  (h6 : diff_time = time_outer - time_inner) 
  (h7 : time_outer = 420) 
  (h8 : diff_time = 220) 
  (h9 : perimeter_outer = perimeter_inner + 40) 
  : s = 2 / 11 := 
begin
  -- Given conditions
  sorry
end

end keiko_speed_l213_213394


namespace sum_of_primes_no_solution_congruence_l213_213601

theorem sum_of_primes_no_solution_congruence :
  2 + 5 = 7 :=
by
  sorry

end sum_of_primes_no_solution_congruence_l213_213601


namespace lambda_value_perpendicular_l213_213650

def a := (1, -3, λ)
def b := (2, 4, -5)

theorem lambda_value_perpendicular (λ : ℝ) (h : a.1 * b.1 + a.2 * b.2 + λ * b.3 = 0) : λ = -2 := by
  sorry

end lambda_value_perpendicular_l213_213650


namespace isosceles_trapezoid_has_ratio_l213_213781

noncomputable def isosceles_trapezoid_ratio (AB CD : ℝ) (areas : List ℝ) : ℝ :=
  if (areas = [9, 5, 7, 3] ∨ areas = [3, 7, 5, 9]) ∧ AB > CD then 3 else 0

theorem isosceles_trapezoid_has_ratio (AB CD : ℝ) (h1 : AB > CD) 
  (h2 : ∃ (areas : List ℝ), areas = [9, 5, 7, 3] ∨ areas = [3, 7, 5, 9]) :
  isosceles_trapezoid_ratio AB CD (classical.some h2) = 3 :=
by
  sorry

end isosceles_trapezoid_has_ratio_l213_213781


namespace johns_remaining_money_l213_213771

theorem johns_remaining_money (q : ℝ) : 
  let cost_of_drinks := 4 * q,
      cost_of_small_pizzas := 2 * q,
      cost_of_large_pizza := 4 * q,
      total_cost := cost_of_drinks + cost_of_small_pizzas + cost_of_large_pizza,
      initial_money := 50 in
  initial_money - total_cost = 50 - 10 * q :=
by
  sorry

end johns_remaining_money_l213_213771


namespace john_remaining_money_l213_213770

variable (q : ℝ)
variable (number_of_small_pizzas number_of_large_pizzas number_of_drinks : ℕ)
variable (cost_of_drink cost_of_small_pizza cost_of_large_pizza dollars_left : ℝ)

def john_purchases := number_of_small_pizzas = 2 ∧
                      number_of_large_pizzas = 1 ∧
                      number_of_drinks = 4 ∧
                      cost_of_drink = q ∧
                      cost_of_small_pizza = q ∧
                      cost_of_large_pizza = 4 * q ∧
                      dollars_left = 50 - (4 * q + 2 * q + 4 * q)

theorem john_remaining_money : john_purchases q 2 1 4 q q (4 * q) (50 - 10 * q) :=
by
  sorry

end john_remaining_money_l213_213770


namespace valid_license_plate_number_is_1440_l213_213861

-- Define the alphabet
def alphabet := ['A', 'E', 'G', 'I', 'K', 'O', 'P', 'R', 'S', 'T', 'U', 'V']

-- Define constraints on the license plates
def is_valid_license_plate (plate : List Char) : Prop :=
  plate.length = 5 ∧
  plate.head ∈ ['A', 'E'] ∧
  plate.last = 'V' ∧
  'I' ∉ plate ∧
  plate.nodup

-- The total number of valid license plates
def valid_license_plate_count : ℕ :=
  2 * 10 * 9 * 8

theorem valid_license_plate_number_is_1440 :
  valid_license_plate_count = 1440 :=
by
  simp only [valid_license_plate_count]
  norm_num -- check that the count is correct
  sorry

end valid_license_plate_number_is_1440_l213_213861


namespace mean_median_difference_l213_213891

def roller_coaster_drops : List ℕ := [165, 119, 138, 300, 198]

def mean (l : List ℕ) : ℕ :=
  l.sum / l.length

def median (l : List ℕ) : ℕ :=
  let sorted := l.sort
  sorted.get! (sorted.length / 2)

theorem mean_median_difference : 
  let drops := roller_coaster_drops
  let drops_mean := mean drops
  let drops_median := median drops
  abs (drops_mean - drops_median) = 19 :=
by
  sorry

end mean_median_difference_l213_213891


namespace relationship_abc_l213_213276

noncomputable def a : ℝ := Real.sqrt 3
noncomputable def b : ℝ := Real.sqrt 15 - Real.sqrt 7
noncomputable def c : ℝ := Real.sqrt 11 - Real.sqrt 3

theorem relationship_abc : a > c ∧ c > b := 
by
  unfold a b c
  sorry

end relationship_abc_l213_213276


namespace tea_blend_gain_l213_213016

-- Definitions of the cost of each variety of tea, the quantities used, and the selling price
def cost_18 : ℝ := 18
def cost_20 : ℝ := 20
def ratio_18 : ℝ := 5
def ratio_20 : ℝ := 3
def selling_price : ℝ := 21

-- Define the formula for gain percent
def gain_percent (cost_price : ℝ) (selling_price : ℝ) : ℝ :=
  ((selling_price - cost_price) / cost_price) * 100

-- Prove that the gain percent is 12%
theorem tea_blend_gain :
  let total_weight := ratio_18 + ratio_20 in
  let total_cost := ratio_18 * cost_18 + ratio_20 * cost_20 in
  let cost_price := total_cost / total_weight in
  gain_percent cost_price selling_price = 12 :=
by
  let total_weight := ratio_18 + ratio_20
  let total_cost := ratio_18 * cost_18 + ratio_20 * cost_20
  let cost_price := total_cost / total_weight
  show gain_percent cost_price selling_price = 12
  sorry

end tea_blend_gain_l213_213016


namespace compute_expression_l213_213120

theorem compute_expression : 2 + 8 * 3 - 4 + 7 * 2 / 2 = 29 := by
  sorry

end compute_expression_l213_213120


namespace circles_intersect_l213_213671

def C1 (x y : ℝ) : Prop := x^2 + (y-1)^2 = 1
def C2 (x y a : ℝ) : Prop := (x-a)^2 + (y-1)^2 = 16

theorem circles_intersect (a : ℝ) (h : a > 0) :
  (∀ x y : ℝ, C1 x y → ∃ x' y' : ℝ, C2 x' y' a) ↔ 3 < a ∧ a < 4 :=
sorry

end circles_intersect_l213_213671


namespace base4_division_correct_l213_213612

def base4_division (x y : ℕ) : ℕ :=
  let x_base4 := Nat.to_digits 4 x
  let y_base4 := Nat.to_digits 4 y
  let quotient := (x / y)
  Nat.of_digits 4 (Nat.to_digits 4 quotient)

theorem base4_division_correct : base4_division 67 5 = 15 := 
by
  sorry

end base4_division_correct_l213_213612


namespace trapezoid_has_area_approx_74_14_l213_213076

-- Define the properties and conditions of the isosceles trapezoid
def trapezoid_area (longer_base : ℝ) (base_angle : ℝ) : ℝ :=
  let x := 20.0 / 1.8 in
  let y := 0.2 * x in
  let height := 0.6 * x in
  (1 / 2) * (y + longer_base) * height

-- Main statement
theorem trapezoid_has_area_approx_74_14 :
  let longer_base := 20
  let base_angle := Real.arcsin 0.6
  abs (trapezoid_area longer_base base_angle - 74.14) < 0.01 :=
by
  sorry

end trapezoid_has_area_approx_74_14_l213_213076


namespace BC_together_time_l213_213517

def work_done (W: ℝ) := W

def A_work_rate (W: ℝ) := W / 4
def B_work_rate (W: ℝ) := W / 4
def AC_work_rate (W: ℝ) := W / 2

def C_work_rate (W: ℝ) := AC_work_rate W - A_work_rate W
def BC_work_rate (W: ℝ) := B_work_rate W + C_work_rate W

theorem BC_together_time (W: ℝ) : (work_done W) / (BC_work_rate W) = 2 := by
    sorry

end BC_together_time_l213_213517


namespace inequality_does_not_hold_l213_213352

theorem inequality_does_not_hold (a b c : ℝ) (h : a < b) : ¬ (ac < bc) := by
  have h1 : c = 0 := sorry
  have h2 : ac = a * c := sorry
  have h3 : bc = b * c := sorry
  rw [h1, mul_zero, mul_zero] at h2 h3
  exact not_lt_of_le (le_refl (0 : ℝ))

end inequality_does_not_hold_l213_213352


namespace range_of_m_l213_213336

theorem range_of_m (m : ℝ) :
  (∀ (λ μ : ℝ), ∃ k : ℝ * ℝ, k = (λ * (3, -2 * m) + μ * (1, m - 2))) ↔ m ≠ 6 / 5 :=
by
  sorry

end range_of_m_l213_213336


namespace range_of_m_l213_213363

theorem range_of_m (m : ℝ) (h : ∃ x : ℝ, abs (x - 3) + abs (x - m) < 5) : -2 < m ∧ m < 8 :=
  sorry

end range_of_m_l213_213363


namespace janina_needs_78_pancakes_l213_213766

variable (rent supplies taxes_wages price_per_pancake : ℝ)
variable (daily_expenses pancakes_needed : ℝ)

def daily_expenses_eq : daily_expenses = rent + supplies + taxes_wages := by
  sorry

def pancakes_needed_eq : pancakes_needed = daily_expenses / price_per_pancake := by
  sorry

axiom rent_value : rent = 75.50
axiom supplies_value : supplies = 28.40
axiom taxes_wages_value : taxes_wages = 32.10
axiom price_per_pancake_value : price_per_pancake = 1.75
axiom daily_expenses_value : daily_expenses = 136.00
axiom pancakes_needed_value : pancakes_needed ≈ 78

theorem janina_needs_78_pancakes : pancakes_needed ≈ 78 := by
  sorry

end janina_needs_78_pancakes_l213_213766


namespace ceil_square_of_neg_seven_fourths_l213_213232

/-- Evaluate the ceiling of the square of -7/4 --/
theorem ceil_square_of_neg_seven_fourths : (Int.ceil ((-7/4 : ℚ)^2 : ℚ) = 4) :=
sorry

end ceil_square_of_neg_seven_fourths_l213_213232


namespace prop1_neg_prop1_conv_prop2_neg_prop2_conv_prop3_neg_prop3_conv_prop4_neg_prop4_conv_l213_213497

-- Definition of Proposition 1 and its negation and converse
def prop1 (x y : ℝ) : Prop := (x * y = 0) → (x = 0 ∨ y = 0)
def neg1 (x y : ℝ) : Prop := (x * y = 0) → ¬(x = 0 ∨ y = 0)
def conv1 (x y : ℝ) : Prop := (x * y ≠ 0) → ¬(x = 0 ∨ y = 0)

-- Definition of Proposition 2 and its negation and converse
def prop2 (a b : ℝ) : Prop := (a + b = 0) → ¬(a > 0 ∧ b > 0)
def neg2 (a b : ℝ) : Prop := (a + b = 0) → (a > 0 ∧ b > 0)
def conv2 (a b : ℝ) : Prop := (a + b ≠ 0) → (a > 0 ∧ b > 0)

-- Definition of Proposition 3 and its negation and converse
def isParallelogram (Q : Type) [Nonempty Q] : Prop := sorry -- A definition for a parallelogram
def adjAnglesEqual (Q : Type) [Nonempty Q] : Prop := sorry  -- A definition for the equality of adjacent angles
def prop3 (Q : Type) [Nonempty Q] : Prop := (isParallelogram Q) → (adjAnglesEqual Q)
def neg3 (Q : Type) [Nonempty Q] : Prop := (isParallelogram Q) → ¬(adjAnglesEqual Q)
def conv3 (Q : Type) [Nonempty Q] : Prop := ¬(isParallelogram Q) → ¬(adjAnglesEqual Q)

-- Definition of Proposition 4 and its negation and converse
def allRationalsAreFractions : Prop := ∀ (q : ℚ), ∃ (a b : ℤ), b ≠ 0 ∧ q = a / b
def neg4 : Prop := ¬(∀ (q : ℚ), ∃ (a b : ℤ), b ≠ 0 ∧ q = a / b)
def conv4 : Prop := ∀ (n : ℝ), (¬(∃ (a b : ℤ), b ≠ 0 ∧ (n = a / b))) → (¬isRational n)

-- Statements to be proven
theorem prop1_neg (x y : ℝ) : prop1 x y ↔ neg1 x y := sorry
theorem prop1_conv (x y : ℝ) : prop1 x y ↔ conv1 x y := sorry

theorem prop2_neg (a b : ℝ) : prop2 a b ↔ neg2 a b := sorry
theorem prop2_conv (a b : ℝ) : prop2 a b ↔ conv2 a b := sorry

theorem prop3_neg (Q : Type) [Nonempty Q] : prop3 Q ↔ neg3 Q := sorry
theorem prop3_conv (Q : Type) [Nonempty Q] : prop3 Q ↔ conv3 Q := sorry

theorem prop4_neg : allRationalsAreFractions ↔ neg4 := sorry
theorem prop4_conv : allRationalsAreFractions ↔ conv4 := sorry

end prop1_neg_prop1_conv_prop2_neg_prop2_conv_prop3_neg_prop3_conv_prop4_neg_prop4_conv_l213_213497


namespace probability_same_number_l213_213837

theorem probability_same_number (n k : ℕ) (h₁ : n = 8) (h₂ : k = 6) : 
  (∃ m : ℝ, 0 ≤ m ∧ m ≤ 1 ∧ m = 1) := by
  sorry

end probability_same_number_l213_213837


namespace fareFor85Miles_l213_213064

-- Define the base fare
def baseFare : ℝ := 20

-- Define the function for fare, given d (distance) and k (proportional constant)
def fare (d k : ℝ) : ℝ := baseFare + k * d

-- Define the given condition that for 60 miles, the fare is $140
def fareCondition : ∃ k : ℝ, fare 60 k = 140 :=
  Exists.intro (2 : ℝ) (
    by 
    have h : fare 60 2 = 140 := by
    dsimp [fare, baseFare]
    norm_num
    exact rfl
  )

-- Theorem stating that the fare for 85 miles is $190
theorem fareFor85Miles : ∃ k : ℝ, fare 85 k = 190 :=
  Exists.intro (2 : ℝ) (
    by 
    have h : fare 85 2 = 190 := by
    dsimp [fare, baseFare]
    norm_num
    exact rfl
  )

end fareFor85Miles_l213_213064


namespace adults_wearing_hats_l213_213512

theorem adults_wearing_hats (total_adults : ℕ) (percent_men : ℝ) (percent_men_hats : ℝ) 
  (percent_women_hats : ℝ) (num_hats : ℕ) 
  (h1 : total_adults = 3600) 
  (h2 : percent_men = 0.40) 
  (h3 : percent_men_hats = 0.15) 
  (h4 : percent_women_hats = 0.25) 
  (h5 : num_hats = 756) : 
  (percent_men * total_adults) * percent_men_hats + (total_adults - (percent_men * total_adults)) * percent_women_hats = num_hats := 
sorry

end adults_wearing_hats_l213_213512


namespace initial_sand_amount_l213_213975

theorem initial_sand_amount (lost_sand : ℝ) (arrived_sand : ℝ)
  (h1 : lost_sand = 2.4) (h2 : arrived_sand = 1.7) :
  lost_sand + arrived_sand = 4.1 :=
by
  rw [h1, h2]
  norm_num

end initial_sand_amount_l213_213975


namespace roots_of_quadratic_l213_213885

theorem roots_of_quadratic (x : ℝ) : (x * (x - 2) = 2 - x) ↔ (x = 2 ∨ x = -1) :=
by
  sorry

end roots_of_quadratic_l213_213885


namespace second_movie_duration_proof_l213_213424

-- initial duration for the first movie (in minutes)
def first_movie_duration_minutes : ℕ := 1 * 60 + 48

-- additional duration for the second movie (in minutes)
def additional_duration_minutes : ℕ := 25

-- total duration for the second movie (in minutes)
def second_movie_duration_minutes : ℕ := first_movie_duration_minutes + additional_duration_minutes

-- convert total minutes to hours and minutes
def duration_in_hours_and_minutes (total_minutes : ℕ) : ℕ × ℕ :=
  (total_minutes / 60, total_minutes % 60)

theorem second_movie_duration_proof :
  duration_in_hours_and_minutes second_movie_duration_minutes = (2, 13) :=
by
  -- proof would go here
  sorry

end second_movie_duration_proof_l213_213424


namespace convex_quadrilaterals_with_arithmetic_angles_l213_213345

theorem convex_quadrilaterals_with_arithmetic_angles :
  { d : ℕ // d > 0 ∧ d < 30 }.card = 29 :=
by 
  sorry

end convex_quadrilaterals_with_arithmetic_angles_l213_213345


namespace prob_heart_certain_event_diamond_random_event_diamond_l213_213270

theorem prob_heart : 
  let total_cards := 9 + 10 + 11
  let num_hearts := 9
  P_heart = num_hearts / total_cards := 
by sorry

theorem certain_event_diamond 
  (m : ℕ) (h_m_gt_6 : m > 6) (h_m_eq_10: m = 10) : 
  let hearts := 9
  let spades := 10
  let diamonds := 11
  let cards_after_removal := diamonds
  (cards_after_removal = 11 := 
by sorry

theorem random_event_diamond 
  (m : ℕ) (h_m_gt_6 : m > 6) (h_m_eq_7: m = 7) : 
  let hearts := 9
  let spades := 10
  let diamonds := 11
  let remaining_spades := spades - 7
  let total_remaining := diamonds + remaining_spades
  P_diamond = diamonds / total_remaining :=
by sorry

end prob_heart_certain_event_diamond_random_event_diamond_l213_213270


namespace proof_solution_l213_213370

noncomputable def proof_problem (a b c : ℝ) (α β γ : ℝ) : Prop :=
triangle a b c α β γ ∧
cos α = sqrt 10 / 10 ∧
a * sin α + b * sin β - c * sin γ = 2 * sqrt 5 / 5 * a * sin β ∧
(b = 10 → S = 60) 

theorem proof_solution (a b c : ℝ) (α β γ : ℝ) :
  proof_problem a b c α β γ → β = π / 4 ∧ b = 10 → S = 60 :=
by
  sorry

end proof_solution_l213_213370


namespace ceil_square_neg_seven_over_four_l213_213211

theorem ceil_square_neg_seven_over_four : 
  let x := - (7 / 4 : ℚ) in
  ⌈x^2⌉ = 4 :=
by
  let x := - (7 / 4 : ℚ)
  sorry

end ceil_square_neg_seven_over_four_l213_213211


namespace sum_primes_no_solution_congruence_l213_213591

theorem sum_primes_no_solution_congruence :
  ∑ p in {p | Nat.Prime p ∧ ¬ (∃ x : ℤ, 5 * (10 * x + 2) ≡ 3 [ZMOD p])}, p = 7 :=
sorry

end sum_primes_no_solution_congruence_l213_213591


namespace ceiling_of_square_frac_l213_213227

theorem ceiling_of_square_frac : 
  let x : ℚ := -7 / 4
  let y : ℚ := x^2
  in Int.ceil y = 4 := 
by 
  let x : ℚ := -7 / 4
  let y : ℚ := x^2
  exact sorry

end ceiling_of_square_frac_l213_213227


namespace abs_expression_not_positive_l213_213920

theorem abs_expression_not_positive (x : ℝ) (h : |2 * x - 7| = 0) : x = 7 / 2 :=
by
  sorry

end abs_expression_not_positive_l213_213920


namespace ceil_square_of_neg_fraction_l213_213193

theorem ceil_square_of_neg_fraction : 
  (Int.ceil ((-7 / 4 : ℚ)^2 : ℚ)).toNat = 4 := by
  sorry

end ceil_square_of_neg_fraction_l213_213193


namespace ceil_square_neg_fraction_l213_213175

theorem ceil_square_neg_fraction :
  let x := (-7) / 4
  let y := x^2
  let z := Real.ceil y
  z = 4 := 
by
  sorry

end ceil_square_neg_fraction_l213_213175


namespace isosceles_trapezoid_area_l213_213102

theorem isosceles_trapezoid_area 
  (a b x y : ℝ) 
  (h1 : a = 20) 
  (h2 : x = 11.11)
  (h3 : y = 2.22)
  (h4 : b = y + 20)
  (h5 : h = 0.6 * x)
  (h6 : arcsin 0.6 = θ) : 
  trapezoid_area a b h = 74.07 := 
  sorry

end isosceles_trapezoid_area_l213_213102


namespace martha_black_butterflies_l213_213809

theorem martha_black_butterflies
    (total_butterflies : ℕ)
    (total_blue_butterflies : ℕ)
    (total_yellow_butterflies : ℕ)
    (total_black_butterflies : ℕ)
    (h1 : total_butterflies = 19)
    (h2 : total_blue_butterflies = 6)
    (h3 : total_blue_butterflies = 2 * total_yellow_butterflies)
    (h4 : total_black_butterflies = total_butterflies - (total_blue_butterflies + total_yellow_butterflies))
    : total_black_butterflies = 10 :=
  sorry

end martha_black_butterflies_l213_213809


namespace parabola_intersect_l213_213285

-- Define a parabola
def parabola (x y : ℝ) : Prop := y ^ 2 = 4 * x

-- Define points A and B on the parabola
def is_point_on_parabola (x y : ℝ) : Prop := parabola x y

-- Focus point F
def focus : ℝ × ℝ := (1, 0)

-- Define line passing through the focus with points A and B
def line_through_focus (x₁ y₁ x₂ y₂ : ℝ) (F : ℝ × ℝ) : Prop :=
  let (Fx, Fy) := F
  Fx = 1 ∧ Fy = 0

-- Midpoint condition
def midpoint_x_condition (x₁ x₂ : ℝ) : Prop := 
  (x₁ + x₂) / 2 = 3

-- Main theorem to prove the length of AB
theorem parabola_intersect 
(F : ℝ × ℝ) (x₁ y₁ x₂ y₂ : ℝ) 
(m_cond : midpoint_x_condition x₁ x₂)
(l_focus : line_through_focus x₁ y₁ x₂ y₂ F)
(p1 : is_point_on_parabola x₁ y₁)
(p2 : is_point_on_parabola x₂ y₂) :
  sqrt ((x₂ - x₁) ^ 2 + (y₂ - y₁) ^ 2) = 8 :=
begin
  sorry
end

end parabola_intersect_l213_213285


namespace arithmetic_example_l213_213121

theorem arithmetic_example : 2 + 8 * 3 - 4 + 7 * 2 / 2 = 29 :=
by
  sorry

end arithmetic_example_l213_213121


namespace suff_but_not_nec_l213_213674

variable (E F G H : Type) [Nonempty E] [Nonempty F] [Nonempty G] [Nonempty H]

-- Proposition A: Points E, F, G, H are not coplanar
def not_coplanar : Prop := sorry -- define non-coplanar condition here

-- Proposition B: Line EF and GH do not intersect
def not_intersect : Prop := sorry -- define non-intersect condition here

-- Prove that A is a sufficient condition for B but not a necessary condition
theorem suff_but_not_nec (h : not_coplanar E F G H) : not_intersect E F G H :=
sorry -- formal proof goes here

end suff_but_not_nec_l213_213674


namespace three_digit_sum_27_l213_213642

theorem three_digit_sum_27 {a b c : ℕ} (h1 : 1 ≤ a ∧ a ≤ 9) (h2 : 0 ≤ b ∧ b ≤ 9) (h3 : 0 ≤ c ∧ c ≤ 9) :
  a + b + c = 27 → (a, b, c) = (9, 9, 9) :=
by
  sorry

end three_digit_sum_27_l213_213642


namespace exists_infinitely_many_N_l213_213399

open Set

-- Conditions: Definition of the initial set S_0 and recursive sets S_n
variable {S_0 : Set ℕ} (h0 : Set.Finite S_0) -- S_0 is a finite set of positive integers
variable (S : ℕ → Set ℕ) 
(has_S : ∀ n, ∀ a, a ∈ S (n+1) ↔ (a-1 ∈ S n ∧ a ∉ S n ∨ a-1 ∉ S n ∧ a ∈ S n))

-- Main theorem: Proving the existence of infinitely many integers N such that 
-- S_N = S_0 ∪ {N + a : a ∈ S_0}
theorem exists_infinitely_many_N : 
  ∃ᶠ N in at_top, S N = S_0 ∪ {n | ∃ a ∈ S_0, n = N + a} := 
sorry

end exists_infinitely_many_N_l213_213399


namespace unripe_oranges_after_days_l213_213339

-- Definitions and Conditions
def sacks_per_day := 65
def days := 6

-- Statement to prove
theorem unripe_oranges_after_days : sacks_per_day * days = 390 := by
  sorry

end unripe_oranges_after_days_l213_213339


namespace hyperbola_eccentricity_l213_213283

theorem hyperbola_eccentricity (a b : ℝ) (h : ∃ P : ℝ × ℝ, ∃ A : ℝ × ℝ, ∃ F : ℝ × ℝ, 
  (∃ c : ℝ, F = (c, 0) ∧ A = (-a, 0) ∧ P.1 ^ 2 / a ^ 2 - P.2 ^ 2 / b ^ 2 = 1 ∧ 
  (F.fst - P.fst) ^ 2 + P.snd ^ 2 = (F.fst + a) ^ 2 ∧ (F.fst - A.fst) ^ 2 + (F.snd - A.snd) ^ 2 = (F.fst + a) ^ 2 ∧ 
  (P.snd = F.snd) ∧ (abs (F.fst - A.fst) = abs (F.fst - P.fst)))) : 
∃ e : ℝ, e = 2 :=
by
  sorry

end hyperbola_eccentricity_l213_213283


namespace ratio_of_average_speeds_l213_213502

/--
Eddy and Freddy start simultaneously from city A and travel to city B and city C respectively.
Eddy takes 3 hours and Freddy takes 4 hours to complete their journeys.
The distance between city A and city B is 600 kms and between city A and city C is 360 kms.
Prove that the ratio of their average speeds (Eddy : Freddy) is 20:9.
-/
theorem ratio_of_average_speeds (tEddy tFreddy dAB dAC : ℕ)
  (h1: tEddy = 3) 
  (h2: tFreddy = 4) 
  (h3: dAB = 600) 
  (h4: dAC = 360) : 
  (dAB / tEddy : dAC / tFreddy) = 20 : 9 :=
by
  sorry

end ratio_of_average_speeds_l213_213502


namespace length_midpoints_of_medians_l213_213470

noncomputable section

variables {A B C M N K L G : Type}

-- Define triangle
def Triangle (A B C : Type) (BC : ℝ) : Prop :=
  ∃ (M N : Type), (median B M) ∧ (median C N) ∧
  (midpoint M L) ∧ (midpoint N K) ∧
  (length(B, C) = BC)

-- Define the length of a segment joining the midpoints of the medians.
theorem length_midpoints_of_medians (a : ℝ) :
  ∀ (A B C M N K L G : Type),
  Triangle A B C a → 
  length(K, L) = a / 4 :=
begin
  sorry
end

end length_midpoints_of_medians_l213_213470


namespace count_x_intercepts_l213_213344

theorem count_x_intercepts : 
  let y := λ x : ℝ, (x - 4) * (x^2 + 4 * x + 3) in
  (∃! x₁ x₂ x₃ : ℝ, y x₁ = 0 ∧ y x₂ = 0 ∧ y x₃ = 0 ∧ x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃) :=
sorry

end count_x_intercepts_l213_213344


namespace fraction_of_orange_juice_in_large_container_l213_213485

def total_capacity := 800 -- mL for each pitcher
def orange_juice_first_pitcher := total_capacity / 2 -- 400 mL
def orange_juice_second_pitcher := total_capacity / 4 -- 200 mL
def total_orange_juice := orange_juice_first_pitcher + orange_juice_second_pitcher -- 600 mL
def total_volume := total_capacity + total_capacity -- 1600 mL

theorem fraction_of_orange_juice_in_large_container :
  (total_orange_juice / total_volume) = 3 / 8 :=
by
  sorry

end fraction_of_orange_juice_in_large_container_l213_213485


namespace other_solution_of_quadratic_l213_213684

theorem other_solution_of_quadratic (x : ℚ) 
  (hx1 : 77 * x^2 - 125 * x + 49 = 0) (hx2 : x = 8/11) : 
  77 * (1 : ℚ)^2 - 125 * (1 : ℚ) + 49 = 0 :=
by sorry

end other_solution_of_quadratic_l213_213684


namespace four_possible_values_for_x_l213_213992

noncomputable def sequence (x : ℝ) : ℕ → ℝ
| 0     => x
| 1     => 1500
| (n+2) => (sequence (n+1) + 1) / sequence n

theorem four_possible_values_for_x :
  { x : ℝ // (sequence x 0 = 1501 ∨ sequence x 1 = 1501 ∨ sequence x 2 = 1501 ∨ sequence x 3 = 1501 ∨ sequence x 4 = 1501) }.to_finset.card = 4 :=
sorry

end four_possible_values_for_x_l213_213992


namespace solve_for_b_l213_213729

theorem solve_for_b (b : ℚ) (h : b + b / 4 = 5 / 2) : b = 2 := 
sorry

end solve_for_b_l213_213729


namespace area_of_polygon_leq_l213_213964

theorem area_of_polygon_leq (L S : ℝ) (hL_nonneg : L ≥ 0)
  (h_polygon : ∃ (broken_line : set (ℝ × ℝ)),
     (non_self_intersecting broken_line)
     ∧ (situated_in_halfplane broken_line)
     ∧ (ends_on_boundary broken_line)
     ∧ (length_of_broken_line broken_line = L)
     ∧ (area_of_polygon broken_line = S)) :
  S ≤ L^2 / (2 * Real.pi) :=
sorry

end area_of_polygon_leq_l213_213964


namespace problem_l213_213258

theorem problem 
  {a1 a2 : ℝ}
  (h1 : 0 ≤ a1)
  (h2 : 0 ≤ a2)
  (h3 : a1 + a2 = 1) :
  ∃ (b1 b2 : ℝ), 0 ≤ b1 ∧ 0 ≤ b2 ∧ b1 + b2 = 1 ∧ ((5/4 - a1) * b1 + 3 * (5/4 - a2) * b2 > 1) :=
by
  sorry

end problem_l213_213258


namespace meals_for_children_l213_213049

-- Define the conditions and the final proof statement

theorem meals_for_children (C : ℕ) 
  (H1 : ∃ A C : ℕ, A = 55 ∧ C = 70) 
  (H2 : ∃ a : ℕ, a = 14) 
  (H3 : ∀ a c : ℕ, a = 56 → c = 72 → 56 * C = 70 * 72) 
  : C = 90 := 
begin
  sorry
end

end meals_for_children_l213_213049


namespace triangle_properties_l213_213762

noncomputable def π : Real := Real.pi -- Define π, since it's noncomputable

-- Definitions of given values
def B : Real := π / 4
def c : Real := Real.sqrt 6
def C : Real := π / 3

-- Calculation to find A based on given conditions
def A : Real := π - B - C

-- Calculation of the Law of Sines ratio
def sine_ratio : Real := c / Real.sin C

-- Calculation of b based on the Law of Sines
def b : Real := sine_ratio * Real.sin B

-- Correct answer for a using the calculated A and Law of Sines
def a : Real := sine_ratio * Real.sin A

-- Statement to prove the questions are equivalent to the answers found
theorem triangle_properties :
  A = 5 * π / 12 ∧
  a = 1 + Real.sqrt 3 ∧
  b = 2 :=
by
  sorry

end triangle_properties_l213_213762


namespace sum_first_13_l213_213748

-- Define an arithmetic sequence
def arithmetic_seq (a d : ℕ → ℕ) : Prop :=
∀ n : ℕ, a (n + 1) = a n + d n

-- Given that we have the arithmetic sequence property
variable (a d : ℕ → ℕ) (h_arith_seq : arithmetic_seq a d)

-- Given the condition in the problem
variable (h_condition : a 2 + a 4 + a 10 + a 12 = 40)

-- Prove that the sum of the first 13 terms S_13 is 130
theorem sum_first_13 (a d : ℕ → ℕ) (h_arith_seq : arithmetic_seq a d) (h_condition : a 2 + a 4 + a 10 + a 12 = 40) :
    let S_13 := (13 * (a 1 + a 13)) / 2 in
    S_13 = 130 :=
by
  sorry

end sum_first_13_l213_213748


namespace range_of_m_l213_213353

noncomputable def m : ℝ := 5 * real.sqrt(1 / 5) - real.sqrt(45)

theorem range_of_m : -5 < m ∧ m < -4 :=
by
  sorry

end range_of_m_l213_213353


namespace find_discount_l213_213985

noncomputable def children_ticket_cost : ℝ := 4.25
noncomputable def adult_ticket_cost : ℝ := children_ticket_cost + 3.25
noncomputable def total_cost_without_discount : ℝ := 2 * adult_ticket_cost + 4 * children_ticket_cost
noncomputable def total_spent : ℝ := 30
noncomputable def discount_received : ℝ := total_cost_without_discount - total_spent

theorem find_discount :
  discount_received = 2 := by
  sorry

end find_discount_l213_213985


namespace xiao_ming_stones_l213_213499

theorem xiao_ming_stones : 
  (∑ i in Finset.range 9, if i = 0 then 0 else i) = 36 →
  (∑ i in Finset.range 9, if i = 0 then 0 else 2^i) = 510 :=
by
  intros h_sum
  sorry

end xiao_ming_stones_l213_213499


namespace big_white_toys_l213_213542

/-- A store has two types of toys, Big White and Little Yellow, with a total of 60 toys.
    The price ratio of Big White to Little Yellow is 6:5.
    Selling all of them results in a total of 2016 yuan.
    We want to determine how many Big Whites there are. -/
theorem big_white_toys (x k : ℕ) (h1 : 6 * x + 5 * (60 - x) = 2016) (h2 : k = 6) : x = 36 :=
by
  sorry

end big_white_toys_l213_213542


namespace revenue_increase_l213_213819

variables (P Q : ℝ)

def original_revenue := P * Q
def new_price := 1.5 * P
def new_quantity := 0.8 * Q
def new_revenue := new_price * new_quantity

theorem revenue_increase : new_revenue P Q = 1.2 * original_revenue P Q := 
by 
  sorry

end revenue_increase_l213_213819


namespace ceil_square_neg_seven_over_four_l213_213206

theorem ceil_square_neg_seven_over_four : 
  let x := - (7 / 4 : ℚ) in
  ⌈x^2⌉ = 4 :=
by
  let x := - (7 / 4 : ℚ)
  sorry

end ceil_square_neg_seven_over_four_l213_213206


namespace larger_group_men_count_l213_213028

-- Define the conditions
def total_man_days (men : ℕ) (days : ℕ) : ℕ := men * days

-- Define the total work for 36 men in 18 days
def work_by_36_men_in_18_days : ℕ := total_man_days 36 18

-- Define the number of days the larger group takes
def days_for_larger_group : ℕ := 8

-- Problem Statement: Prove that if 36 men take 18 days to complete the work, and a larger group takes 8 days, then the larger group consists of 81 men.
theorem larger_group_men_count : 
  ∃ (M : ℕ), total_man_days M days_for_larger_group = work_by_36_men_in_18_days ∧ M = 81 := 
by
  -- Here would go the proof steps
  sorry

end larger_group_men_count_l213_213028


namespace prove_k_range_prove_k_m_values_l213_213326

noncomputable theory
open Real

-- Definitions and conditions
def hyperbola (a : ℝ) (x y : ℝ) : Prop := x^2 / a^2 - y^2 = 1
def eccentricity (e : ℝ) : Prop := e = sqrt 2
def line (k x y : ℝ) : Prop := y = k * x - 1
def intersection_points (x1 y1 x2 y2 : ℝ) (k : ℝ) : Prop :=
    (line k x1 y1) ∧ (line k x2 y2) ∧ (hyperbola 1 x1 y1) ∧ (hyperbola 1 x2 y2)

-- Proofs
theorem prove_k_range (k : ℝ) (h_intersect : ∃ (x1 y1 x2 y2 : ℝ), intersection_points x1 y1 x2 y2 k) :
  1 < k ∧ k < sqrt 2 :=
sorry

theorem prove_k_m_values (k m : ℝ) 
  (h_intersect : ∃ (x1 y1 x2 y2 : ℝ), intersection_points x1 y1 x2 y2 k)
  (h_length : (sqrt (1 + k^2) * sqrt (2 - k^2) / (abs (k^2 - 1))) = 6 * sqrt 3)
  (C : ℝ × ℝ) 
  (h_C : ∃ (x1 y1 x2 y2 x0 y0 : ℝ), 
         intersection_points x1 y1 x2 y2 k ∧ C = (x0, y0) ∧ 
         (x0, y0) = (4 * sqrt 5 * m, 8 * m)) :
  k = sqrt 5 / 2 ∧ (m = 1 / 4 ∨ m = -1 / 4) :=
sorry

end prove_k_range_prove_k_m_values_l213_213326


namespace square_shadow_cannot_be_trapezoid_l213_213541

-- Definitions based on conditions
def is_square (shape : Type) : Prop := ∃ s : ℝ, shape = rectangle s s
def is_parallelogram (shape : Type) : Prop := ∃ a b : ℝ, a ≠ b ∧ shape = parallelogram a b
def is_rectangle (shape : Type) : Prop := ∃ l w : ℝ, shape = rectangle l w
def is_line_segment (shape : Type) : Prop := ∃ l : ℝ, shape = line_segment l
def is_trapezoid (shape : Type) : Prop := ∃ a b c d : ℝ, a ≠ c ∧ shape = trapezoid a b c d

-- Condition: The shadow shape of a square piece of paper with various light sources
def shadow_shape (light_source_orientation : Type) (shape : Type) : Prop :=
  (light_source_orientation = "overhead" → is_square shape) ∧
  (light_source_orientation = "angled" → (is_parallelogram shape ∨ is_rectangle shape)) ∧
  (light_source_orientation = "low_parallel" → is_line_segment shape)

-- Theorem stating the problem
theorem square_shadow_cannot_be_trapezoid (shape : Type) (light_source_orientation: Type) :
  shadow_shape light_source_orientation shape → ¬ is_trapezoid shape :=
sorry

end square_shadow_cannot_be_trapezoid_l213_213541


namespace cost_of_bananas_l213_213903

theorem cost_of_bananas (A B : ℕ) (h1 : 2 * A + B = 7) (h2 : A + B = 5) : B = 3 :=
by
  sorry

end cost_of_bananas_l213_213903


namespace final_color_all_blue_l213_213818

-- Definitions based on the problem's initial conditions
def initial_blue_sheep : ℕ := 22
def initial_red_sheep : ℕ := 18
def initial_green_sheep : ℕ := 15

-- The final problem statement: prove that all sheep end up being blue
theorem final_color_all_blue (B R G : ℕ) 
  (hB : B = initial_blue_sheep) 
  (hR : R = initial_red_sheep) 
  (hG : G = initial_green_sheep) 
  (interaction : ∀ (B R G : ℕ), (B > 0 ∨ R > 0 ∨ G > 0) → (R ≡ G [MOD 3])) :
  ∃ b, b = B + R + G ∧ R = 0 ∧ G = 0 ∧ b % 3 = 1 ∧ B = b :=
by
  -- Proof to be provided
  sorry

end final_color_all_blue_l213_213818


namespace range_of_omega_l213_213275

theorem range_of_omega (ω : ℝ) (hω : ω > 0) :
  (∀ x : ℝ, x ∈ set.Icc (π / 2) π → deriv (fun x => 3 * real.sin (ω * x + π / 4) - 2) x ≤ 0) ↔ (ω ∈ set.Icc (1 / 2) (5 / 4)) :=
by
  sorry

end range_of_omega_l213_213275


namespace jed_receives_five_10_bills_l213_213390

theorem jed_receives_five_10_bills
  (num_board_games : ℕ := 8)
  (cost_per_game : ℕ := 18)
  (total_payment : ℕ := 200)
  (bill_denomination : ℕ := 10) :
  let total_cost := num_board_games * cost_per_game
  let change := total_payment - total_cost
  let num_10_bills := change / bill_denomination
  num_10_bills = 5 := by
sory

end jed_receives_five_10_bills_l213_213390


namespace coordinate_system_restored_l213_213522

structure Point :=
  (x : ℝ)
  (y : ℝ)

def A : Point := ⟨1, 2⟩
def B : Point := ⟨3, 1⟩

def midpoint (p1 p2 : Point) : Point :=
  ⟨(p1.x + p2.x) / 2, (p1.y + p2.y) / 2⟩

theorem coordinate_system_restored :
  ∃ O : Point, ∃ M : Point, 
  M = midpoint A B ∧ 
  M = midpoint O B ∧ 
  (O.x ≠ B.x) ∧ (O.y ≠ B.y) ∧ 
  (A.x - O.x) * (A.y - O.y) = -((B.x - O.x) * (B.y - O.y)) :=
by
  sorry

end coordinate_system_restored_l213_213522


namespace tom_speed_RB_l213_213901

/-- Let d be the distance between B and C (in miles).
    Let 2d be the distance between R and B (in miles).
    Let v be Tom’s speed driving from R to B (in mph).
    Given conditions:
    1. Tom's speed from B to C = 20 mph.
    2. Total average speed of the whole journey = 36 mph.
    Prove that Tom's speed driving from R to B is 60 mph. -/
theorem tom_speed_RB
  (d : ℝ) (v : ℝ)
  (h1 : 20 ≠ 0)
  (h2 : 36 ≠ 0)
  (avg_speed : 3 * d / (2 * d / v + d / 20) = 36) :
  v = 60 := 
sorry

end tom_speed_RB_l213_213901


namespace sum_fractions_l213_213253

def f (x : ℝ) : ℝ := x^2 / (1 + x^2)

theorem sum_fractions :
  (∑ k in Finset.range 2000, f (k.succ / 2000)) + (∑ k in Finset.range 1999, f (2000 / (k.succ))) = 1999.5 := by
sorry

end sum_fractions_l213_213253


namespace cross_horizontal_asymptote_at_2_l213_213264

def g (x : ℝ) : ℝ := (3*x^2 - 7*x - 10) / (x^2 - 6*x + 4)

theorem cross_horizontal_asymptote_at_2 :
  ∃ x : ℝ, g x = 3 ∧ x = 2 :=
by
  sorry

end cross_horizontal_asymptote_at_2_l213_213264


namespace martha_savings_l213_213813

theorem martha_savings :
  let daily_allowance := 12
  let days_in_week := 7
  let save_half_daily := daily_allowance / 2
  let save_quarter_daily := daily_allowance / 4
  let days_saving_half := 6
  let day_saving_quarter := 1
  let total_savings := (days_saving_half * save_half_daily) + (day_saving_quarter * save_quarter_daily)
  in total_savings = 39 :=
by
  sorry

end martha_savings_l213_213813


namespace problem1_problem2_l213_213365

-- This file contains the Lean translation of the given math problem statement.

noncomputable def OA_length (AB BC OB OA : ℝ) := sqrt ((AB)^2 + (OB)^2 - 2 * AB * OB * cos (π/6)) = sqrt (7)/2

noncomputable def OB_OC_ratio (BC OB OC : ℝ) := OB / OC = sqrt (3) / 4

theorem problem1 :
  ∀ (AB BC OC OB OA : ℝ) 
  (h1 : AC = 2 * BC = 2) 
  (h2 : AB ⊥ BC)
  (h3 : OB ⊥ OC)
  (h4 : OC = sqrt (3) * OB), 
  OA = sqrt(7)/2 :=
begin
  sorry -- proof omitted
end

theorem problem2 :
  ∀ (AB BC OC OB : ℝ) 
  (h1 : AC = 2 * BC = 2) 
  (h2 : AB ⊥ BC)
  (h3 : OB ⊥ OC)
  (h4 : ∡ AOC = 2 / 3 * π),
  OB / OC = sqrt(3) / 4 := 
begin
  sorry -- proof omitted
end

end problem1_problem2_l213_213365


namespace average_speed_remainder_l213_213767

/--
Jason has to drive home which is 120 miles away.
If he drives at 60 miles per hour for 30 minutes,
prove that he has to average 90 miles per hour for the remainder of the drive
to get there in exactly 1 hour 30 minutes.
-/
theorem average_speed_remainder (total_distance : ℕ) (initial_speed : ℕ) (initial_time : ℕ) (total_time : ℕ) :
  total_distance = 120 →
  initial_speed = 60 →
  initial_time = 30 →
  total_time = 90 →
  average_speed_remainder = 90 :=
by
  sorry

end average_speed_remainder_l213_213767


namespace triangle_ratio_sum_l213_213759

theorem triangle_ratio_sum
  (A B C D E F G : Type)
  (h1 : (A ≠ B) ∧ (B ≠ C) ∧ (A ≠ C))
  (h2 : ∃ r : ℝ, r = 1/3 ∧ ratio A B B C = r)
  (h3 : midpoint D B C)
  (h4 : midpoint E A B)
  (h5 : ∃ s : ℝ, s = 1/2 ∧ ratio A G G B = s)
  (h6 : ∃ t : ℝ, t = 2 ∧ ratio D F F C = t) :
  ratio E F F C + ratio A F F D = 86 / 85 :=
sorry

end triangle_ratio_sum_l213_213759


namespace angle_b_is_sixty_max_area_triangle_l213_213304

variables (a b c A B C : Real) (A_pos B_pos C_pos : Prop)
variables (A_sum : A + B + C = π) -- angles of a triangle
variables (triangle_sides : a = 2 * sin(A) ∨ b = 2 * sin(B) ∨ c = 2 * sin(C)) -- Law of Sines
variables (condition : (a + c) / b = cos(C) + sqrt(3) * sin(C))

noncomputable theory

-- Part 1: Show B is 60 degrees
theorem angle_b_is_sixty (hA : A_pos A) (hB : B_pos B) (hC : C_pos C) :
  B = π / 3 :=
  sorry

-- Part 2: Given b = 2, show the maximum area of the triangle is sqrt(3)
theorem max_area_triangle (hb : b = 2) :
  (∃ a c, (a + c) / b = cos(C) + sqrt(3) * sin(C) ∧
          let S := 1 / 2 * a * c * sin(B) in
          ∀ a' c', (a' + c') / b = cos(C) + sqrt(3) * sin(C) → 
                   1 / 2 * a' * c' * sin(B) ≤ S ∧ S = sqrt(3)) :=
  sorry

end angle_b_is_sixty_max_area_triangle_l213_213304


namespace proof_d_e_f_value_l213_213685

theorem proof_d_e_f_value
  (a b c d e f : ℝ)
  (h1 : a * b * c = 195)
  (h2 : b * c * d = 65)
  (h3 : c * d * e = 1000)
  (h4 : (a * f) / (c * d) = 0.75) :
  d * e * f = 250 :=
sorry

end proof_d_e_f_value_l213_213685


namespace primary_school_teacher_is_odd_one_out_l213_213566

def profession := {stomatologist : bool, primary_school_teacher : bool, programmer : bool}

noncomputable def current_russian_pension_legislation (p : profession) := 
  if p.stomatologist then 
    true 
  else if p.primary_school_teacher then 
    true 
  else 
    false

theorem primary_school_teacher_is_odd_one_out
  (p : profession) 
  (hp : p = ⟨true, true, false⟩) : 
  ¬ current_russian_pension_legislation p := 
sorry

end primary_school_teacher_is_odd_one_out_l213_213566


namespace closest_perfect_square_to_315_l213_213006

theorem closest_perfect_square_to_315 : ∃ n : ℤ, n^2 = 324 ∧
  (∀ m : ℤ, m ≠ n → (abs (315 - m^2) > abs (315 - n^2))) := 
sorry

end closest_perfect_square_to_315_l213_213006


namespace chord_length_of_intersection_l213_213708

noncomputable def polar_to_cartesian_equation (ρ θ : ℝ) : Prop :=
  ρ * (Real.sin θ)^2 = 8 * Real.cos θ

noncomputable def cartesian_curve (x y : ℝ) : Prop :=
  y^2 = 8 * x

noncomputable def parametric_line (t : ℝ) (x y : ℝ) : Prop :=
  x = 2 + 0.5 * t ∧ y = (Real.sqrt 3) / 2 * t

theorem chord_length_of_intersection :
  (∀ (ρ θ : ℝ), polar_to_cartesian_equation ρ θ → ∃ x y, cartesian_curve x y)
  ∧
  (∀ (t : ℝ), ∃ x y, parametric_line t x y) →
  ∃ A B : ℝ × ℝ, let d := (A.1 - B.1)^2 + (A.2 - B.2)^2 in Real.sqrt d = 32 / 3 :=
sorry

end chord_length_of_intersection_l213_213708


namespace mari_buttons_l213_213800

/-- 
Given that:
1. Sue made 6 buttons
2. Sue made half as many buttons as Kendra.
3. Mari made 4 more than five times as many buttons as Kendra.

We are to prove that Mari made 64 buttons.
-/
theorem mari_buttons (sue_buttons : ℕ) (kendra_buttons : ℕ) (mari_buttons : ℕ) 
  (h1 : sue_buttons = 6)
  (h2 : sue_buttons = kendra_buttons / 2)
  (h3 : mari_buttons = 5 * kendra_buttons + 4) :
  mari_buttons = 64 :=
  sorry

end mari_buttons_l213_213800


namespace john_remaining_money_l213_213769

variable (q : ℝ)
variable (number_of_small_pizzas number_of_large_pizzas number_of_drinks : ℕ)
variable (cost_of_drink cost_of_small_pizza cost_of_large_pizza dollars_left : ℝ)

def john_purchases := number_of_small_pizzas = 2 ∧
                      number_of_large_pizzas = 1 ∧
                      number_of_drinks = 4 ∧
                      cost_of_drink = q ∧
                      cost_of_small_pizza = q ∧
                      cost_of_large_pizza = 4 * q ∧
                      dollars_left = 50 - (4 * q + 2 * q + 4 * q)

theorem john_remaining_money : john_purchases q 2 1 4 q q (4 * q) (50 - 10 * q) :=
by
  sorry

end john_remaining_money_l213_213769


namespace surface_area_ratio_l213_213330

-- Given condition
def radius_ratio (R R' : ℝ) : Prop := R / R' = 1 / 2

-- Define the surface area of a sphere
def surface_area (R : ℝ) : ℝ := 4 * Real.pi * R^2

-- Statement to prove: The ratio of the surface areas is 1:4 given the ratio of radii is 1:2
theorem surface_area_ratio (R R' : ℝ) (h : radius_ratio R R') : 
  surface_area R / surface_area R' = 1 / 4 := by
  sorry

end surface_area_ratio_l213_213330


namespace smallest_nine_digit_times_smallest_seven_digit_l213_213887

theorem smallest_nine_digit_times_smallest_seven_digit :
  let smallest_nine_digit := 100000000
  let smallest_seven_digit := 1000000
  smallest_nine_digit = 100 * smallest_seven_digit :=
by
  sorry

end smallest_nine_digit_times_smallest_seven_digit_l213_213887


namespace prove_circumcircle_area_l213_213677

-- Definitions and given conditions
def hyperbola (x y : ℝ) : Prop := (x^2 / 4) - (y^2 / 5) = 1

def F1 : ℝ × ℝ := (-3, 0)
def F2 : ℝ × ℝ := (3, 0)

def distance (a b : ℝ × ℝ) : ℝ :=
  real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2)

def P_on_hyperbola (P : ℝ × ℝ) : Prop := hyperbola P.1 P.2 ∧ distance P F1 = 2 * distance P F2

theorem prove_circumcircle_area (P : ℝ × ℝ)
  (hP : P_on_hyperbola P)
  (h_dist_F1F2 : distance F1 F2 = 6) :
  ∃ A : ℝ, A = (256 * real.pi) / 15 :=
sorry

end prove_circumcircle_area_l213_213677


namespace math_problem_proof_l213_213692

noncomputable def ordinary_eq_C1 : Prop :=
  ∀ (t : ℝ), (y = ∃ x, x = 1 + (1/2) * t ∧ y = (sqrt 3 / 2) * t) →
    ∀ x, y = sqrt 3 * (x - 1)

noncomputable def rectangular_eq_C2 : Prop := 
  ρ^2 = ∀ θ, 12 / (3 + sin(θ)^2) →
    ∀ x y, ((x / 2) ^ 2) + ((y / sqrt 3) ^ 2) = 1

noncomputable def intersection_value : Prop :=
  ∀ (t1 t2 : ℝ),
    (t1 + t2 = -4/5) ∧ (t1 * t2 = -12/5) ∧
    (C1_inter_C2 : ∀ x y, y = sqrt 3 * (x - 1) ∧ ((x / 2) ^ 2 + (y / sqrt 3) ^ 2 = 1)) →
    ((1 / abs(1 + (1/2) * t1) + (1 / abs(1 + (1/2) * t2)) = 4 / 3)

theorem math_problem_proof : ordinary_eq_C1 ∧ rectangular_eq_C2 ∧ intersection_value :=
sorry

end math_problem_proof_l213_213692


namespace product_correlation_function_l213_213845

open ProbabilityTheory

/-
Theorem: Given two centered and uncorrelated random functions \( \dot{X}(t) \) and \( \dot{Y}(t) \),
the correlation function of their product \( Z(t) = \dot{X}(t) \dot{Y}(t) \) is the product of their correlation functions.
-/
theorem product_correlation_function 
  (X Y : ℝ → ℝ)
  (hX_centered : ∀ t, (∫ x, X t ∂x) = 0) 
  (hY_centered : ∀ t, (∫ y, Y t ∂y) = 0)
  (h_uncorrelated : ∀ t1 t2, ∫ x, X t1 * Y t2 ∂x = (∫ x, X t1 ∂x) * (∫ y, Y t2 ∂y)) :
  ∀ t1 t2, 
  (∫ x, (X t1 * Y t1) * (X t2 * Y t2) ∂x) = 
  (∫ x, X t1 * X t2 ∂x) * (∫ y, Y t1 * Y t2 ∂y) :=
by
  sorry

end product_correlation_function_l213_213845


namespace shaded_square_cover_columns_l213_213536

def triangular_number (n : Nat) : Nat := n * (n + 1) / 2

theorem shaded_square_cover_columns :
  ∃ n : Nat, 
    triangular_number n = 136 ∧ 
    ∀ i : Fin 10, ∃ k ≤ n, (triangular_number k) % 10 = i.val :=
sorry

end shaded_square_cover_columns_l213_213536


namespace unique_real_solution_l213_213143

-- Define the variables
variables (x y : ℝ)

-- State the condition
def equation (x y : ℝ) : Prop :=
  (2^(4*x + 2)) * (4^(2*x + 3)) = (8^(3*x + 4)) * y

-- State the theorem
theorem unique_real_solution (y : ℝ) (h_y : 0 < y) : ∃! x : ℝ, equation x y :=
sorry

end unique_real_solution_l213_213143


namespace sum_of_primes_no_solution_l213_213597

def is_prime (n : ℕ) : Prop := Nat.Prime n

def no_solution (p : ℕ) : Prop :=
  is_prime p ∧ ¬ ∃ x : ℤ, 5 * (10 * x + 2) ≡ 3 [MOD p]

def gcd_condition (p : ℕ) : Prop :=
  p = 2 ∨ p = 5

theorem sum_of_primes_no_solution : (∑ p in {p | is_prime p ∧ gcd_condition p}, p) = 7 :=
by
  sorry

end sum_of_primes_no_solution_l213_213597


namespace problem_statement_l213_213747

-- Define the arithmetic sequence and required terms
def arithmetic_seq (a : ℕ → ℝ) (d : ℝ) := ∀ n : ℕ, a (n + 1) = a n + d

-- Conditions
variables (a : ℕ → ℝ) (d : ℝ)
axiom seq_is_arithmetic : arithmetic_seq a d
axiom sum_of_a2_a4_a6_is_3 : a 2 + a 4 + a 6 = 3

-- Goal: Prove a1 + a3 + a5 + a7 = 4
theorem problem_statement : a 1 + a 3 + a 5 + a 7 = 4 :=
by 
  sorry

end problem_statement_l213_213747


namespace correct_propositions_l213_213636

def prop_B (x y : ℝ) : Prop := ⌊x⌋ + ⌊y⌋ ≤ ⌊x + y⌋

def prop_C (y : ℝ) : Prop := 0 ≤ y ∧ y < 1

def prop_D (t : ℝ) (n : ℕ) : Prop :=
  let satisfies_condition := ∀ k : ℕ, 3 ≤ k → k ≤ n → ⌊t^k⌋ = k - 2 in
  satisfies_condition t n ∧ n ≤ 5

theorem correct_propositions :
  (∀ x y, prop_B x y) ∧
  (∀ x : ℝ, prop_C (x - ⌊x⌋)) ∧
  (∃ t : ℝ, ∃ n : ℕ, prop_D t n) :=
by
  sorry

end correct_propositions_l213_213636


namespace sin_810_eq_one_l213_213111

theorem sin_810_eq_one : Real.sin (810 * Real.pi / 180) = 1 :=
by
  -- You can add the proof here
  sorry

end sin_810_eq_one_l213_213111


namespace n_th_time_divide_angle_l213_213420

theorem n_th_time_divide_angle (n k : ℕ) :
  ∃ t : ℕ,
    let s := n * 60 in
    let x := t - s in
    let M := n + x / 60 in
    let H := n / 12 + x / 720 in
    (x - H) / (M - x) = k → t = (43200 * (1 + k) * n) / (719 + 708 * k) :=
sorry

end n_th_time_divide_angle_l213_213420


namespace max_value_log_a_log_b_sq_l213_213791

theorem max_value_log_a_log_b_sq (a b : ℝ) (ha : 1 < a) (hb : 1 < b) (hab : a * b = 100) :
  ∃ (x : ℝ), x = 32 / 27 ∧ log 10 (a ^ (log 10 b)^2) = x := by
  sorry

end max_value_log_a_log_b_sq_l213_213791


namespace wetsuit_chest_size_l213_213415

-- Conditions
def inches_to_feet (inches : ℚ) : ℚ := inches / 12

def feet_to_cm (feet : ℚ) : ℚ := feet * 31

def adjusted_chest_size_in_inches (chest_size : ℚ) (adjustment : ℚ) := chest_size + adjustment

-- Given Problem transformed to Lean 4 statement
theorem wetsuit_chest_size 
  (chest_size : ℚ := 36) 
  (adjustment : ℚ := 2) 
  (ft_to_cm_const : ℚ := 31) 
  (inch_to_ft_const : ℚ := 12) :
  let inches := adjusted_chest_size_in_inches chest_size adjustment
  let feet := inches_to_feet inches
  let centimeters := feet_to_cm feet
  (centimeters ≈ 98.2) := -- Here ≈ means approximately equal, since the problem states rounding to the nearest tenth
begin
  sorry,
end

end wetsuit_chest_size_l213_213415


namespace determine_f_l213_213410

def f (x : ℝ) : ℝ := (x ^ 3 + 7 * x) / (2 - 2 * x ^ 2)

theorem determine_f :
  (∀ (x : ℝ), (|x| ≠ 1) → f ((x - 2) / (x + 1)) + f ((3 + x) / (1 - x)) = x) :=
by
  intro x hx
  simp [f]
  sorry

end determine_f_l213_213410


namespace exists_zero_in_interval_l213_213893

noncomputable def f (x : ℝ) : ℝ := Real.log x + 2 * x - 6

theorem exists_zero_in_interval : 
  (f 2) * (f 3) < 0 := by
  sorry

end exists_zero_in_interval_l213_213893


namespace find_positive_integer_solutions_l213_213623

theorem find_positive_integer_solutions : 
  {s : (ℕ × ℕ × ℕ × ℕ) // s.1 ≤ s.2 ∧ s.2 ≤ s.3 ∧ s.3 ≤ s.4 ∧ 0 < s.1 ∧ 0 < s.2 ∧ 0 < s.3 ∧ 0 < s.4 ∧
    (1 : ℚ) / (s.1 : ℚ) + (1 : ℚ) / (s.2) + (1 : ℚ) / (s.3) + (1 : ℚ) / (s.4) = 1 } = 
  { (2, 3, 7, 42), (2, 3, 8, 24), (2, 3, 9, 18), (2, 3, 10, 15), (2, 3, 12, 12), 
    (2, 4, 5, 20), (2, 4, 6, 12), (2, 4, 8, 8), (2, 5, 5, 10), (2, 6, 6, 6), 
    (3, 3, 4, 12), (3, 3, 6, 6), (3, 4, 4, 6), (4, 4, 4, 4) } :=
by
  sorry

end find_positive_integer_solutions_l213_213623


namespace cost_of_fencing_l213_213455

theorem cost_of_fencing :
  ∀ (total_length : ℝ) (ratio_A ratio_B ratio_C : ℝ) 
  (cost_A cost_B cost_C : ℝ) (correct_total_cost : ℝ),
  total_length = 2400 →
  ratio_A = 3 → ratio_B = 4 → ratio_C = 5 →
  cost_A = 6.25 → cost_B = 4.90 → cost_C = 7.35 →
  correct_total_cost = 15020 →
  let x := total_length / (ratio_A + ratio_B + ratio_C),
      length_A := ratio_A * x,
      length_B := ratio_B * x,
      length_C := ratio_C * x,
      total_cost := length_A * cost_A + length_B * cost_B + length_C * cost_C in
  total_cost = correct_total_cost :=
by
  intros total_length ratio_A ratio_B ratio_C cost_A cost_B cost_C correct_total_cost
  intro h_total_length
  intro h_ratio_A
  intro h_ratio_B
  intro h_ratio_C
  intro h_cost_A
  intro h_cost_B
  intro h_cost_C
  intro h_correct_total_cost
  let x := total_length / (ratio_A + ratio_B + ratio_C)
  let length_A := ratio_A * x
  let length_B := ratio_B * x
  let length_C := ratio_C * x
  let total_cost := length_A * cost_A + length_B * cost_B + length_C * cost_C
  sorry

end cost_of_fencing_l213_213455


namespace simplify_expression_l213_213493

theorem simplify_expression (x y : ℝ) (h : x ≠ y) : 
  (x - y)⁻¹ * (x⁻² + y⁻²) = (x² + y²) * x⁻² * y⁻² * (x - y)⁻¹ :=
by 
  sorry

end simplify_expression_l213_213493


namespace digit_222_of_fraction_41_777_is_0_l213_213911

-- Define the decimal sequence for the fraction 41 / 777 as a list of digits
def decimal_sequence : List ℕ := [0, 5, 2, 7, 8, 5, 9, 2, 3, 7, 5, 3, 6, 6, 5, 6, 8, 9, 1, 4, 9, 5, 6, 0, 1, 1, 7, 3, 0, 2, 0, 5, 2, 7, 8, 5, 9, 2, 3, 7, 5, 3, 6, 6, 5, 6, 8, 9, 1, 4, 9, 5, 6, 0, 1, 1, 7, 3, 0, 2]

-- Prove that the 222nd digit after the decimal point of 41/777 is 0
theorem digit_222_of_fraction_41_777_is_0 : (decimal_sequence.nth (222 % 42 - 1)).iget = 0 := by {
  -- Skip the proof
  sorry
}

end digit_222_of_fraction_41_777_is_0_l213_213911


namespace complex_conjugate_product_eq_two_l213_213657

theorem complex_conjugate_product_eq_two (z : ℂ) (hz : z = 1 + complex.I) : z * complex.conj z = 2 :=
by
  rw [hz, complex.conj, complex.norm_sq, abs_z]
  sorry

end complex_conjugate_product_eq_two_l213_213657


namespace ceiling_of_square_of_neg_7_over_4_is_4_l213_213168

theorem ceiling_of_square_of_neg_7_over_4_is_4 : 
  Real.ceil ((-7 / 4 : Real) ^ 2) = 4 := by
  sorry

end ceiling_of_square_of_neg_7_over_4_is_4_l213_213168


namespace intersection_A_B_l213_213294

def A := {x : ℝ | x > 1}
def B := {0, 1, 2, 3}

theorem intersection_A_B : A ∩ B = {2, 3} :=
by
  -- Proof is omitted
  sorry

end intersection_A_B_l213_213294


namespace problem_1_problem_2_l213_213325

noncomputable def f (x : ℝ) : ℝ := log (x / 8) / log 2 * (log (x / 2) / log 4) + 1/2

theorem problem_1 (x : ℝ) (hx : x = 4^(2/3)) : f x = 2/9 :=
by sorry

theorem problem_2 (m : ℝ) (hm : m > 1) : 
  ∃ y_min, ∀ x, 2 ≤ x ∧ x ≤ 2^m → 
    f x ≥ y_min ∧ 
    (y_min = if 1 < m ∧ m ≤ 2 then 1/2 * m^2 - 2 * m + 2 else 0) :=
by sorry

end problem_1_problem_2_l213_213325


namespace ceil_of_neg_frac_squared_l213_213195

-- Define the negated fraction
def neg_frac : ℚ := -7 / 4

-- Define the squared value of the negated fraction
def squared_value : ℚ := neg_frac ^ 2

-- Define the ceiling function applied to the squared value
def ceil_squared_value : ℤ := Int.ceil squared_value

-- Prove that the ceiling of the squared value is 4
theorem ceil_of_neg_frac_squared : ceil_squared_value = 4 := 
by sorry

end ceil_of_neg_frac_squared_l213_213195


namespace range_of_m_l213_213695

variable {x y m : ℝ}

theorem range_of_m (hx : 0 < x) (hy : 0 < y) (h : 4 / x + 1 / y = 1) :
  x + y ≥ m^2 + m + 3 ↔ -3 ≤ m ∧ m ≤ 2 := sorry

end range_of_m_l213_213695


namespace isosceles_trapezoid_area_l213_213099

theorem isosceles_trapezoid_area 
  (a b x y : ℝ) 
  (h1 : a = 20) 
  (h2 : x = 11.11)
  (h3 : y = 2.22)
  (h4 : b = y + 20)
  (h5 : h = 0.6 * x)
  (h6 : arcsin 0.6 = θ) : 
  trapezoid_area a b h = 74.07 := 
  sorry

end isosceles_trapezoid_area_l213_213099


namespace sin_double_angle_l213_213298

variable (α : ℝ)

theorem sin_double_angle (h : sin α + cos α = 2 / 3) : sin (2 * α) = -5 / 9 :=
by sorry

end sin_double_angle_l213_213298


namespace ratio_of_groups_l213_213060

variable (x : ℚ)

-- The total number of people in the calligraphy group
def calligraphy_group (x : ℚ) := x + (2 / 7) * x

-- The total number of people in the recitation group
def recitation_group (x : ℚ) := x + (1 / 5) * x

theorem ratio_of_groups (x : ℚ) (hx : x ≠ 0) : 
    (calligraphy_group x) / (recitation_group x) = (3 : ℚ) / (4 : ℚ) := by
  sorry

end ratio_of_groups_l213_213060


namespace trapezoid_area_l213_213080

-- Define the problem statement
theorem trapezoid_area 
  (a b h: ℝ)
  (b₁ b₂: ℝ)
  (θ: ℝ) 
  (h₃: θ = Real.arcsin 0.6)
  (h₄: a = 20)
  (h₅: b = a - 2 * b₁ * Real.sin θ) 
  (h₆: h = b₁ * Real.cos θ) 
  (h₇: θ = Real.arcsin (3/5)) 
  (circum: isosceles_trapezoid_circumscribed a b₁ b₂) :
  ((1 / 2) * (a + b₂) * h = 2000 / 27) :=
by sorry

end trapezoid_area_l213_213080


namespace hannah_spent_on_dessert_l213_213714

theorem hannah_spent_on_dessert
  (initial_money : ℕ)
  (money_left : ℕ)
  (half_spent_on_rides : ℕ)
  (total_spent : ℕ)
  (spent_on_dessert : ℕ)
  (H1 : initial_money = 30)
  (H2 : money_left = 10)
  (H3 : half_spent_on_rides = initial_money / 2)
  (H4 : total_spent = initial_money - money_left)
  (H5 : spent_on_dessert = total_spent - half_spent_on_rides) : spent_on_dessert = 5 :=
by
  sorry

end hannah_spent_on_dessert_l213_213714


namespace ceil_square_neg_fraction_l213_213183

theorem ceil_square_neg_fraction :
  let x := (-7) / 4
  let y := x^2
  let z := Real.ceil y
  z = 4 := 
by
  sorry

end ceil_square_neg_fraction_l213_213183


namespace hilary_corn_shucking_l213_213341

def total_ears (stalks: ℕ) (ears_per_stalk: ℕ) : ℕ :=
  stalks * ears_per_stalk

def bad_ears (total_ears: ℕ) (bad_percentage: ℕ) : ℕ :=
  (total_ears * bad_percentage) / 100

def good_ears (total_ears: ℕ) (bad_ears: ℕ) : ℕ :=
  total_ears - bad_ears

def kernel_count (good_ears: ℕ) (perc: ℕ) (kernels_per_ear: ℕ) : ℕ :=
  (good_ears * perc / 100) * kernels_per_ear

def total_kernels 
  (good_ears: ℕ) 
  (perc_500: ℕ) (kernels_500: ℕ) 
  (perc_600: ℕ) (kernels_600: ℕ) 
  (perc_700: ℕ) (kernels_700: ℕ) : ℕ :=
  kernel_count (good_ears id perc_500 kernels_500) +
  kernel_count (good_ears id perc_600 kernels_600) +
  kernel_count (good_ears id perc_700 kernels_700)

theorem hilary_corn_shucking : 
  total_kernels (good_ears (total_ears 108 4) (bad_ears (total_ears 108 4) 20))
                60 500 
                30 600 
                10 700 = 189100 :=
sorry

end hilary_corn_shucking_l213_213341


namespace compute_area_l213_213576

noncomputable def area_of_bounded_figure : ℝ :=
  let parametric_x (t : ℝ) := 24 * (Real.cos t) ^ 3
  let parametric_y (t : ℝ) := 2 * (Real.sin t) ^ 3
  2 * (∫ t in 0..π / 6, parametric_y t * (Deriv.dt (parametric_x t) t))

theorem compute_area : area_of_bounded_figure = 3 * π - 9 * Real.sqrt 3 := 
by
  -- Proof is omitted
  sorry

end compute_area_l213_213576


namespace high_school_nine_total_games_l213_213860

theorem high_school_nine_total_games :
  ∀ (n : ℕ), 
  ∀ (non_league_games_per_team : ℕ), 
  (n = 9) → (non_league_games_per_team = 6) → 
  (∑ i in finset.range n, (n - i - 1) / 2 + non_league_games_per_team * n = 90) :=
begin
  intros n non_league_games_per_team h1 h2,
  rw [h1, h2],
  sorry
end

end high_school_nine_total_games_l213_213860


namespace ceiling_of_square_frac_l213_213225

theorem ceiling_of_square_frac : 
  let x : ℚ := -7 / 4
  let y : ℚ := x^2
  in Int.ceil y = 4 := 
by 
  let x : ℚ := -7 / 4
  let y : ℚ := x^2
  exact sorry

end ceiling_of_square_frac_l213_213225


namespace min_value_of_reciprocal_sum_l213_213299

theorem min_value_of_reciprocal_sum (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : 2^(a + b) = 16) : 
  (∃ (x : ℝ), (2^(x) = 4) ∧ (x = a / b)) → 
  ∃ x : ℝ, ∀ (y z : ℝ), (y = 1 / a) ∧ (z = 1 / b) ∧ (y + z = 1) :=
sorry

end min_value_of_reciprocal_sum_l213_213299


namespace unpainted_cubes_count_l213_213937

noncomputable def num_unpainted_cubes : ℕ :=
  let total_cubes := 216
  let painted_on_faces := 16 * 6 / 1  -- Central 4x4 areas on each face
  let shared_edges := ((4 * 4) * 6) / 2  -- Shared edges among faces
  let shared_corners := (4 * 6) / 3  -- Shared corners among faces
  let total_painted := painted_on_faces - shared_edges - shared_corners
  total_cubes - total_painted

theorem unpainted_cubes_count : num_unpainted_cubes = 160 := sorry

end unpainted_cubes_count_l213_213937


namespace less_than_k_times_perimeter_l213_213976

open_locale big_operators

variables {T : Type*} [metric_space T] (A B C A' B' C' : T)
variables (k : ℝ) (h₁ : 1 / 2 < k) (h₂ : k < 1)
variables (ABC_perimeter A'B'C'_perimeter : ℝ)

noncomputable def Triangle (A B C : T) : Prop := true

axiom same_side_ratio (ABC_triangle : Triangle A B C) :
  0 < k ∧ k < 1 →
  dist A B * dist B C = k * dist A' B' ∧
  dist A' B' * dist B' C' = k * dist ABC_perimeter

theorem less_than_k_times_perimeter (ABC_triangle : Triangle A B C) :
  (dist A' B' + dist B' C' + dist C' A') < k * (dist A B + dist B C + dist C A) :=
sorry

end less_than_k_times_perimeter_l213_213976


namespace graph_shift_right_by_pi_over_6_l213_213702

theorem graph_shift_right_by_pi_over_6 (ω : ℝ) (hω : ω > 0) (h_period : cos (ω * x - (ω * π / 6)) = cos ((ω * x) - (2 * π))) :
  ∃ c : ℝ, ∀ x : ℝ, cos (2 * x - π / 3) = cos (2 * (x - c)) ∧ c = π / 6 :=
by { sorry }

end graph_shift_right_by_pi_over_6_l213_213702


namespace ceil_square_of_neg_seven_fourths_l213_213238

/-- Evaluate the ceiling of the square of -7/4 --/
theorem ceil_square_of_neg_seven_fourths : (Int.ceil ((-7/4 : ℚ)^2 : ℚ) = 4) :=
sorry

end ceil_square_of_neg_seven_fourths_l213_213238


namespace sum_is_zero_l213_213977

def ai : ℕ → ℤ
def S (n : ℕ) : ℤ := ∑ i in Finset.range 25, ai (n + 1 + i)

theorem sum_is_zero (h : ∀ i, 1 ≤ i ∧ i ≤ 100 → ai i = 1 ∨ ai i = -1)
  (h_sum : ∑ i in Finset.range 100, ai (i + 1) = 0) :
  ∃ n, 0 ≤ n ∧ n ≤ 75 ∧ S n = 1 :=
sorry

end sum_is_zero_l213_213977


namespace find_f_21_l213_213653

def f : ℝ → ℝ := sorry

lemma f_condition (x : ℝ) : f (2 / x + 1) = Real.log x := sorry

theorem find_f_21 : f 21 = -1 := sorry

end find_f_21_l213_213653


namespace arithmetic_sequence_common_difference_l213_213662

theorem arithmetic_sequence_common_difference (a : ℕ → ℝ) (d : ℝ)
  (h1 : a 1 = 1)
  (h3 : a 3 = 11)
  (h_arith : ∀ n : ℕ, a (n + 1) - a n = d) : d = 5 :=
sorry

end arithmetic_sequence_common_difference_l213_213662


namespace ceil_square_of_neg_seven_fourths_l213_213229

/-- Evaluate the ceiling of the square of -7/4 --/
theorem ceil_square_of_neg_seven_fourths : (Int.ceil ((-7/4 : ℚ)^2 : ℚ) = 4) :=
sorry

end ceil_square_of_neg_seven_fourths_l213_213229


namespace trigonometric_identity_l213_213454

theorem trigonometric_identity (α : ℝ) :
  (sin (real.pi - α) / cos (real.pi + α)) * (cos (-α) * cos (2 * real.pi - α) / sin (real.pi / 2 + α)) = -sin α := by
  -- Using trigonometric identities
  -- sin (real.pi - α) = sin α
  -- cos (real.pi + α) = -cos α
  -- cos (-α) = cos α
  -- cos (2 * real.pi - α) = cos α
  -- sin (real.pi / 2 + α) = cos α
  sorry

end trigonometric_identity_l213_213454


namespace arrangement_count_l213_213561

/-- There are 48 different ways to arrange 5 different Winter Olympics
    publicity works into two rows with at least 2 works in each row,
    and works A and B must be placed in the front row. -/
theorem arrangement_count (works : Fin 5 → Prop) (front_row back_row : Fin 5 → Prop)
  (A B : Fin 5) 
  (h1 : works.sum = 5) 
  (h2 : ∀ (x : Fin 5), works x → (front_row x ∨ back_row x))
  (h3 : front_row A)
  (h4 : front_row B)
  (h5 : (front_row.sum ≥ 2) ∧ (back_row.sum ≥ 2)) :
  ∃ n : ℕ, n = 48 :=
begin
  -- Begin proof by counting the number of valid arrangements.
  -- Placeholder for actual proof steps.
  sorry
end

end arrangement_count_l213_213561


namespace original_bacteria_count_l213_213058

-- Define the initial conditions
variable (original current increase : ℕ)

-- Define the given conditions
axiom current_eq : current = 8917
axiom increase_eq : increase = 8317
axiom bacteria_growth : current = original + increase

-- The main goal is to prove that the original number of bacteria is 600
theorem original_bacteria_count : original = 600 :=
by
  rw [← bacteria_growth, current_eq, increase_eq]
  exact Nat.sub_eq_of_eq_add (Nat.add_sub_eq_of_eq_add bacteria_growth)
  sorry

end original_bacteria_count_l213_213058


namespace trapezoid_area_l213_213095

theorem trapezoid_area 
  (a b h c : ℝ) 
  (ha : 2 * 0.8 * a + b = c)
  (hb : c = 20) 
  (hc : h = 0.6 * a) 
  (hd : b + 1.6 * a = 20)
  (angle_base : ∃ θ : ℝ, θ = arcsin 0.6)
  : 
  (1 / 2) * (b + c) * h = 72 :=
sorry

end trapezoid_area_l213_213095


namespace sector_angle_l213_213883

theorem sector_angle (r L : ℝ) (h1 : r = 1) (h2 : L = 4) : abs (L - 2 * r) = 2 :=
by 
  -- This is the statement of our proof problem
  -- and does not include the proof itself.
  sorry

end sector_angle_l213_213883


namespace ceil_square_of_neg_fraction_l213_213186

theorem ceil_square_of_neg_fraction : 
  (Int.ceil ((-7 / 4 : ℚ)^2 : ℚ)).toNat = 4 := by
  sorry

end ceil_square_of_neg_fraction_l213_213186


namespace find_h_l213_213876

theorem find_h (j k h : ℕ) (h₁ : 2013 = 3 * h^2 + j) (h₂ : 2014 = 2 * h^2 + k)
  (pos_int_x_intercepts_1 : ∃ x1 x2 : ℕ, x1 ≠ x2 ∧ x1 > 0 ∧ x2 > 0 ∧ (3 * (x1 - h)^2 + j = 0 ∧ 3 * (x2 - h)^2 + j = 0))
  (pos_int_x_intercepts_2 : ∃ y1 y2 : ℕ, y1 ≠ y2 ∧ y1 > 0 ∧ y2 > 0 ∧ (2 * (y1 - h)^2 + k = 0 ∧ 2 * (y2 - h)^2 + k = 0)):
  h = 36 :=
by
  sorry

end find_h_l213_213876


namespace equiangular_quad_is_rectangle_l213_213045

/-- A figure is an equiangular quadrilateral with each interior angle measuring 90 degrees
    is necessarily a rectangle. -/
theorem equiangular_quad_is_rectangle
  (Q : Type) [quasiGroup Q] [involutiveGhd Q]
  [subfield Q] (equiv_quad : eq Q ≘ qua_group Q)
  [gEq : Q → Q → Prop] (angle : ∠ Q = 90) : 
  is_rectangle equiv_quad :=
sorry

end equiangular_quad_is_rectangle_l213_213045


namespace max_single_student_books_l213_213374

-- Definitions and conditions
variable (total_students : ℕ) (no_books : ℕ) (one_book : ℕ) (two_books : ℕ)
variable (total_avg_books_per_student : ℕ)

-- Given data
def given_data : Prop :=
  total_students = 20 ∧ no_books = 2 ∧ one_book = 8 ∧
  two_books = 3 ∧ total_avg_books_per_student = 2

-- Maximum number of books any single student could borrow
theorem max_single_student_books (total_students no_books one_book two_books total_avg_books_per_student : ℕ) 
  (h : given_data total_students no_books one_book two_books total_avg_books_per_student) : 
  ∃ max_books_borrowed, max_books_borrowed = 8 :=
by
  sorry

end max_single_student_books_l213_213374


namespace solve_for_x_l213_213585

-- Define the operation "※" within the range of positive numbers.
def op (a b : ℝ) : ℝ := (1 / a) + (1 / b)

theorem solve_for_x (x : ℝ) : op 3 (x - 1) = 1 -> x = 5 / 2 :=
by
  intro h
  -- The proof will be filled in here.
  sorry

end solve_for_x_l213_213585


namespace find_x_l213_213732

theorem find_x (x y : ℝ) (h₁ : 0.60 * x = 0.30 * y + 27) (h₂ : y = real.cbrt 125) : x = 47.5 :=
by sorry

end find_x_l213_213732


namespace line_through_A_and_intersecting_circle_l213_213606

-- Define the point A
def A : ℝ × ℝ := (-3, 0)

-- Define the circle equation
def circle (x y : ℝ) : Prop := x^2 + y^2 - 6 * y - 16 = 0

-- Define the distance condition of MN
def chord_length_8 (M N : ℝ × ℝ) : Prop := (M.1 - N.1)^2 + (M.2 - N.2)^2 = 64

theorem line_through_A_and_intersecting_circle (l : ℝ → ℝ → Prop) :
  (∃ y : ℝ, l (-3) y) ∧ (∀ x y : ℝ, l x y → circle x y) ∧
  (∃ M N : ℝ × ℝ, M ≠ N ∧ l M.1 M.2 ∧ l N.1 N.2 ∧ chord_length_8 M N) →
  (∀ x y : ℝ, l x y ↔ (x = -3 ∨ y = 0)) :=
sorry

end line_through_A_and_intersecting_circle_l213_213606


namespace arc_length_intercepted_by_octagon_side_l213_213537

noncomputable def radius : ℝ := 5
noncomputable def central_angle : ℝ := 45
def circumference (r : ℝ) : ℝ := 2 * Real.pi * r
def arc_fraction (angle : ℝ) : ℝ := angle / 360
def arc_length (C : ℝ) (frac : ℝ) : ℝ := C * frac

theorem arc_length_intercepted_by_octagon_side :
  arc_length (circumference radius) (arc_fraction central_angle) = (5 * Real.pi) / 4 :=
  by
    sorry

end arc_length_intercepted_by_octagon_side_l213_213537


namespace AM_GM_inequality_l213_213440

theorem AM_GM_inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : (a + b) / 2 ≥ real.sqrt (a * b) := by
  sorry

end AM_GM_inequality_l213_213440


namespace binomial_odd_sum_l213_213307

theorem binomial_odd_sum (n : ℕ) (hn : binomial n 3 = binomial n 7) : 
  (finset.range (n + 1)).filter (λ k, k % 2 = 1).sum (λ k, binomial n k) = 512 :=
by sorry

end binomial_odd_sum_l213_213307


namespace trapezoid_has_area_approx_74_14_l213_213075

-- Define the properties and conditions of the isosceles trapezoid
def trapezoid_area (longer_base : ℝ) (base_angle : ℝ) : ℝ :=
  let x := 20.0 / 1.8 in
  let y := 0.2 * x in
  let height := 0.6 * x in
  (1 / 2) * (y + longer_base) * height

-- Main statement
theorem trapezoid_has_area_approx_74_14 :
  let longer_base := 20
  let base_angle := Real.arcsin 0.6
  abs (trapezoid_area longer_base base_angle - 74.14) < 0.01 :=
by
  sorry

end trapezoid_has_area_approx_74_14_l213_213075


namespace ceiling_of_square_frac_l213_213226

theorem ceiling_of_square_frac : 
  let x : ℚ := -7 / 4
  let y : ℚ := x^2
  in Int.ceil y = 4 := 
by 
  let x : ℚ := -7 / 4
  let y : ℚ := x^2
  exact sorry

end ceiling_of_square_frac_l213_213226


namespace average_distance_from_sides_l213_213054

noncomputable def lemming_average_distance 
(side_length : ℝ) 
(move_diag : ℝ) 
(move_right_1 : ℝ) 
(move_right_2 : ℝ) : ℝ :=
let diagonal := real.sqrt (side_length ^ 2 + side_length ^ 2) in
let x_after_diag := move_diag * side_length / diagonal in
let y_after_diag := move_diag * side_length / diagonal in
let x_after_first_turn := x_after_diag + move_right_1 in
let y_after_first_turn := y_after_diag in
let x_after_second_turn := x_after_first_turn in
let y_after_second_turn := y_after_first_turn - move_right_2 in
let dist_left := x_after_second_turn in
let dist_bottom := y_after_second_turn in
let dist_right := side_length - x_after_second_turn in
let dist_top := side_length - y_after_second_turn in
(dist_left + dist_bottom + dist_right + dist_top) / 4

theorem average_distance_from_sides 
: lemming_average_distance 12 8 3 2 = 6.25 := 
by
  -- Proof is omitted.
  sorry

end average_distance_from_sides_l213_213054


namespace same_remainder_permuted_digits_l213_213051

noncomputable def four_digit_number (m c d u : ℕ) : ℕ :=
  1000 * m + 100 * c + 10 * d + u

theorem same_remainder_permuted_digits (m₁ c₁ d₁ u₁ m₂ c₂ d₂ u₂ : ℕ) :
  four_digit_number m₁ c₁ d₁ u₁ < 10000 →
  four_digit_number m₂ c₂ d₂ u₂ < 10000 →
  multiset.card (multiset.of_list [m₁, c₁, d₁, u₁]) = multiset.card (multiset.of_list [m₂, c₂, d₂, u₂]) →
  (m₁ + c₁ + d₁ + u₁) % 9 = (m₂ + c₂ + d₂ + u₂) % 9 :=
by 
  intros hN1 hN2 hperm
  sorry

end same_remainder_permuted_digits_l213_213051


namespace primary_school_teacher_is_odd_one_out_l213_213567

def profession := {stomatologist : bool, primary_school_teacher : bool, programmer : bool}

noncomputable def current_russian_pension_legislation (p : profession) := 
  if p.stomatologist then 
    true 
  else if p.primary_school_teacher then 
    true 
  else 
    false

theorem primary_school_teacher_is_odd_one_out
  (p : profession) 
  (hp : p = ⟨true, true, false⟩) : 
  ¬ current_russian_pension_legislation p := 
sorry

end primary_school_teacher_is_odd_one_out_l213_213567


namespace eccentricity_of_ellipse_equation_of_ellipse_l213_213670

variable {a b : ℝ}
variable {x y : ℝ}

/-- Problem 1: Eccentricity of the given ellipse --/
theorem eccentricity_of_ellipse (ha : a = 2 * b) (hb0 : 0 < b) :
  ∃ e : ℝ, e = Real.sqrt (1 - (b / a) ^ 2) ∧ e = Real.sqrt 3 / 2 := by
  sorry

/-- Problem 2: Equation of the ellipse with respect to maximizing the area of triangle OMN --/
theorem equation_of_ellipse (ha : a = 2 * b) (hb0 : 0 < b) :
  ∃ l : ℝ → ℝ, (∃ k : ℝ, ∀ x, l x = k * x + 2) →
  ∀ x y : ℝ, (x^2 / (a^2) + y^2 / (b^2) = 1) →
  (∀ x' y' : ℝ, (x'^2 + 4 * y'^2 = 4 * b^2) ∧ y' = k * x' + 2) →
  (∃ a b : ℝ, a = 8 ∧ b = 2 ∧ x^2 / a + y^2 / b = 1) := by
  sorry

end eccentricity_of_ellipse_equation_of_ellipse_l213_213670


namespace find_g_l213_213682

noncomputable def f (x : ℝ) : ℝ :=
if x < 0 then g x else x^2 - 2 * x

def is_odd (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f (-x) = -f x

variable (g : ℝ → ℝ)

theorem find_g (H1 : ∀ x, f x = (if x < 0 then g x else x^2 - 2 * x))
               (H2 : is_odd f) :
  ∀ x < 0, g x = -x^2 - 2 * x :=
sorry

end find_g_l213_213682


namespace ceil_square_neg_seven_over_four_l213_213213

theorem ceil_square_neg_seven_over_four : 
  let x := - (7 / 4 : ℚ) in
  ⌈x^2⌉ = 4 :=
by
  let x := - (7 / 4 : ℚ)
  sorry

end ceil_square_neg_seven_over_four_l213_213213


namespace equilateral_triangle_l213_213776

variable {α : Type*}
variables {A B C L H M : α}
variables [HasSmul α α] [HasSub α] [Add α]

noncomputable def is_midpoint (X Y Z : α) : Prop := 2 • X = Y + Z

noncomputable def is_angle_bisector (B C L : α) : Prop := sorry -- Needs further definitions 

noncomputable def is_altitude (A H B C : α) : Prop := sorry -- Needs further definitions 

theorem equilateral_triangle {A B C L H M : α} 
  (h1 : ∃ (A B C : α), True) 
  (h2 : is_angle_bisector B C L)
  (h3 : is_altitude A H B C)
  (h4 : is_midpoint M A B)
  (h5 : ∃ (X : α), is_midpoint X B L ∧ is_midpoint X M H) :
  ∠A B C = 60 ∧ ∠B A C = 60 ∧ ∠A C B = 60 := 
sorry

end equilateral_triangle_l213_213776


namespace quadrilateral_midpoints_intersect_l213_213453

noncomputable def midpoint (A B : Point) : Point := 
  ⟨(A.1 + B.1) / 2, (A.2 + B.2) / 2⟩

theorem quadrilateral_midpoints_intersect 
  (A B C D M N P Q E F : Point) 
  (hM : M = midpoint A B)
  (hN : N = midpoint B C) 
  (hP : P = midpoint C D) 
  (hQ : Q = midpoint D A) 
  (hE : E = midpoint A C)
  (hF : F = midpoint B D) : 
  ∃ O : Point, 
    (O = midpoint M P) ∧ (O = midpoint N Q) ∧ (O = midpoint E F) :=
sorry

end quadrilateral_midpoints_intersect_l213_213453


namespace petya_can_win_l213_213908

def is_divisible_by_9 (n : ℕ) : Prop := n % 9 = 0

def valid_placement (d : ℕ) (positions : list ℕ) : Prop :=
  d ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ ∀ p, p ∈ positions → 0 ≤ p ∧ p < 10

theorem petya_can_win :
  ∀ (digits: list ℕ) (positions: list (list ℕ)),
  (∀ d ∈ digits, valid_placement d (positions.nth.least [])) →
  ∃ numbers: list ℕ, 
  (∀ n ∈ numbers, is_divisible_by_3 n) →
  is_divisible_by_9 (numbers.prod) :=
by
  sorry

end petya_can_win_l213_213908


namespace part1_part2_l213_213323

-- Definition of the function
def f (x : ℝ) (m : ℝ) : ℝ := x^2 + m * x - 1

-- Theorem for part (1)
theorem part1 
  (m n : ℝ)
  (h1 : ∀ x : ℝ, f x m < 0 ↔ -2 < x ∧ x < n) : 
  m = 3 / 2 ∧ n = 1 / 2 :=
sorry

-- Theorem for part (2)
theorem part2 
  (m : ℝ)
  (h2 : ∀ x : ℝ, m ≤ x ∧ x ≤ m + 1 → f x m < 0) : 
  -Real.sqrt 2 / 2 < m ∧ m < 0 :=
sorry

end part1_part2_l213_213323


namespace sum_of_fractions_solve_equation_l213_213842

theorem sum_of_fractions (n : ℕ) : 
  (Finset.range (n + 1)).sum (λ k, 1 / ((k + 1) * (k + 2) : ℚ)) = (n + 1) / (n + 2) :=
sorry

theorem solve_equation : 
  let lhs := (Finset.range 17).sum (λ k, (2 * k + 1)) / 
            ((Finset.range 18).sum (λ k, 1 / ((k + 1) * (k + 2) : ℚ))) in
     lhs = 342 → 17 = 17 := 
sorry

end sum_of_fractions_solve_equation_l213_213842


namespace estimate_sqrt_expr_l213_213150

theorem estimate_sqrt_expr :
  5 < sqrt (1 / 3) * sqrt 27 + sqrt 7 ∧ sqrt (1 / 3) * sqrt 27 + sqrt 7 < 6 := 
by 
  sorry

end estimate_sqrt_expr_l213_213150


namespace scientific_notation_of_12000_l213_213461

theorem scientific_notation_of_12000 : 12000 = 1.2 * 10^4 := 
by sorry

end scientific_notation_of_12000_l213_213461


namespace retirement_age_total_l213_213038

theorem retirement_age_total (hire_year : ℕ) (hire_age : ℕ) (retirement_year : ℕ) (age_plus_years : ℕ) :
  hire_year = 1988 → hire_age = 32 → retirement_year = 2007 → age_plus_years = 70 :=
by
  intros h1 h2 h3
  have employment_years : ℕ := retirement_year - hire_year
  have age_at_retirement : ℕ := hire_age + employment_years
  calc 
    age_plus_years = age_at_retirement + employment_years :
      by sorry -- The actual steps relevant to the proof will go here.

end retirement_age_total_l213_213038


namespace vect_sum_magnitude_l213_213337

noncomputable def dot_product (u v : ℝ × ℝ) : ℝ :=
u.1 * v.1 + u.2 * v.2

noncomputable def magnitude (u : ℝ × ℝ) : ℝ :=
Real.sqrt (u.1 * u.1 + u.2 * u.2)

theorem vect_sum_magnitude :
  ∀ (x : ℝ),
    let a := (x, 1)
    let b := (1, -2)
    (dot_product a b = 0) → (magnitude (a.1 + b.1, a.2 + b.2) = Real.sqrt 10) :=
by {
  intros x a b ha_perpendicular,
  have hx : x = 2,
  calc
    dot_product a b = x * 1 + 1 * -2 : by simp [dot_product]
    ... = x - 2 : by ring
    ... = 0 : by exact ha_perpendicular,
  sorry
}

end vect_sum_magnitude_l213_213337


namespace alice_spent_19_percent_l213_213107

variable (A : ℝ) (x : ℝ)
variable (h1 : ∃ (B : ℝ), B = 0.9 * A) -- Bob's initial amount in terms of Alice's initial amount
variable (h2 : A - x = 0.81 * A) -- Alice's remaining amount after spending x

theorem alice_spent_19_percent (h1 : ∃ (B : ℝ), B = 0.9 * A) (h2 : A - x = 0.81 * A) : (x / A) * 100 = 19 := by
  sorry

end alice_spent_19_percent_l213_213107


namespace length_of_train_l213_213067

-- Define the conditions as separate constants
def speed_train := 60 -- in kmph
def speed_man := 6 -- in kmph
def time_to_pass_man := 21 -- in seconds

-- Conversion factor from kmph to m/s
def kmph_to_mps (speed_kmph : ℕ) : ℕ := speed_kmph * 5 / 18

-- Combine the above definitions to specify the problem statement
theorem length_of_train :
  let relative_speed := kmph_to_mps (speed_train + speed_man) in
  let length_train := relative_speed * time_to_pass_man in
  length_train = 385 :=
by
  -- This is a placeholder as we assume a later filling of the proof steps.
  sorry

end length_of_train_l213_213067


namespace simplify_expression_l213_213917

variable (w : ℝ)

theorem simplify_expression : 3 * w + 5 - 6 * w^2 + 4 * w - 7 + 9 * w^2 = 3 * w^2 + 7 * w - 2 := by
  sorry

end simplify_expression_l213_213917


namespace dihedral_angle_between_planes_l213_213755

noncomputable def midpoint (a b : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  ((a.1 + b.1) / 2, (a.2 + b.2) / 2, (a.3 + b.3) / 2)

noncomputable def point_F (a a1 : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (2 * a1.1 / 3 + a.1 / 3, 2 * a1.2 / 3 + a.2 / 3, 2 * a1.3 / 3 + a.3 / 3)

theorem dihedral_angle_between_planes :
  let A := (0, 0, 0)
  let B := (6, 0, 0)
  let C := (6, 6, 0)
  let B1 := (6, 0, 6)
  let A1 := (0, 0, 6)
  let E := midpoint B C
  let F := point_F A A1
  arctan (real.sqrt 37 / 3) = arctan (real.sqrt 37 / 3) :=
by
  let A := (0, 0, 0)
  let B := (6, 0, 0)
  let C := (6, 6, 0)
  let B1 := (6, 0, 6)
  let A1 := (0, 0, 6)
  let E := midpoint B C
  let F := point_F A A1
  sorry

end dihedral_angle_between_planes_l213_213755


namespace book_pages_read_l213_213451

theorem book_pages_read (pages_per_day : ℕ) (days_per_week : ℕ) (weeks : ℕ) (total_pages : ℕ) :
  (pages_per_day = 100) →
  (days_per_week = 3) →
  (weeks = 7) →
  total_pages = pages_per_day * days_per_week * weeks →
  total_pages = 2100 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  exact h4

end book_pages_read_l213_213451


namespace nature_of_roots_l213_213645

noncomputable def custom_operation (a b : ℝ) : ℝ := a^2 - a * b

theorem nature_of_roots :
  ∀ x : ℝ, custom_operation (x + 1) 3 = -2 → ∃ x1 x2 : ℝ, x1 ≠ x2 :=
by
  intro x h
  sorry

end nature_of_roots_l213_213645


namespace triangle_is_right_isosceles_l213_213367

noncomputable def right_isosceles_triangle (a b c : ℝ) (A B C : ℝ) : Prop :=
  B ∈ Ioo 0 (Real.pi / 2) ∧ Real.log10 a - Real.log10 c = Real.log10 (Real.sin B) ∧ Real.log10 (Real.sin B) = -Real.log10 (Real.sqrt 2) ∧ 
  A = Real.pi - B - C ∧ C = Real.pi / 2

theorem triangle_is_right_isosceles {a b c A B C : ℝ} (h : right_isosceles_triangle a b c A B C) :
  (A = Real.pi / 4 ∧ B = Real.pi / 4 ∧ C = Real.pi / 2) :=
by sorry

end triangle_is_right_isosceles_l213_213367


namespace circle_radius_tangent_rel_l213_213990

theorem circle_radius_tangent_rel (C D E : Type) [metric_space C] [metric_space D] [metric_space E]
    (radius_C : ℝ) (radius_D : ℝ) (radius_E : ℝ) (AB : set C) (A : C) 
    (hC : ∀ (x y : C), dist x y = 2 * radius_C)
    (hD : ∀ (x y : D), dist x y = 2 * radius_D)
    (hE : ∀ (x y : E), dist x y = 2 * radius_E)
    (tangent_CD : ∀ {a : C}, a ∈ metric.ball D radius_D ↔ a ∈ metric.ball C radius_C)
    (tangent_DE : ∀ {a : D}, a ∈ metric.ball E radius_E ↔ a ∈ metric.ball D radius_D)
    (tangent_CE : ∀ {a : C}, a ∈ metric.ball E radius_E ↔ a ∈ metric.ball C radius_C)
    (radii_relation : radius_D = 2 * radius_E)
    (radius_form : radius_D = 2 * real.sqrt 32 - 16) : 32 + 16 = 48 :=
by sorry

end circle_radius_tangent_rel_l213_213990


namespace inequality_solution_range_of_a_l213_213317

noncomputable def f (x : ℝ) : ℝ := |1 - 2 * x| - |1 + x| 

theorem inequality_solution (x : ℝ) : f x ≥ 4 ↔ x ≤ -2 ∨ x ≥ 6 := 
by sorry

theorem range_of_a (a x : ℝ) (h : a^2 + 2 * a + |1 + x| < f x) : -3 < a ∧ a < 1 :=
by sorry

end inequality_solution_range_of_a_l213_213317


namespace greatest_difference_between_set_A_set_B_l213_213848

theorem greatest_difference_between_set_A_set_B 
  (A : Finset ℕ)
  (B : Finset ℕ)
  (hA_card : A.card = 8)
  (hB_card : B.card = 8)
  (hA_sum : A.sum = 39)
  (hB_sum : B.sum = 39)
  (hB_distinct : ∀ x ∈ B, ∀ y ∈ B, x ≠ y → x ≠ y) :
  let M := 39 - (Finset.erase A (Finset.max' A (Finset.nonempty_of_card_eq_succ hA_card)).erase (Finset.max' A (Finset.nonempty_of_card_eq_succ hA_card))).sum
  let N := 39 - (Finset.erase B (Finset.max' B (Finset.nonempty_of_card_eq_succ hB_card)).erase (Finset.max' B (Finset.nonempty_of_card_eq_succ hB_card))).sum
  M - N = 8 :=
begin
  sorry
end

end greatest_difference_between_set_A_set_B_l213_213848


namespace ceiling_of_square_frac_l213_213217

theorem ceiling_of_square_frac : 
  let x : ℚ := -7 / 4
  let y : ℚ := x^2
  in Int.ceil y = 4 := 
by 
  let x : ℚ := -7 / 4
  let y : ℚ := x^2
  exact sorry

end ceiling_of_square_frac_l213_213217


namespace simplify_sqrt_sum_l213_213446

theorem simplify_sqrt_sum : (Real.sqrt 72 + Real.sqrt 32) = 10 * Real.sqrt 2 := by
  sorry

end simplify_sqrt_sum_l213_213446


namespace sum_of_slopes_of_tangents_l213_213255

open Real

noncomputable def eccentricity (a b : ℝ) : ℝ := √(1 - (b^2 / a^2))

theorem sum_of_slopes_of_tangents 
  (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : b < a) 
  (k : ℝ) (hE : eccentricity a b = sqrt 2 / 2)
  (hx1n : x1 ≠ 0) :
  let E := λ x y, (x^2 / a^2) + (y^2 / b^2) = 1 in
  let A := (0, -b) in
  let P := (x1, k * (x1 - 1) + 1) in
  let Q := (x2, k * (x2 - 1) + 1) in
  x1 ≠ 1 ∧ x2 ≠ 1 ∧
  sum_of_roots := (λ x1 x2 k, x1 + x2) in
  ∃ x1 x2, (E x1 (k * (x1 - 1) + 1)) ∧ (E x2 (k * (x2 - 1) + 1))
           ∧ x1 ≠ x2 
           ∧ (k + sum_of_roots x1 x2 k) = 2 :=
begin
  sorry
end

end sum_of_slopes_of_tangents_l213_213255


namespace arithmetic_example_l213_213123

theorem arithmetic_example : 2 + 8 * 3 - 4 + 7 * 2 / 2 = 29 :=
by
  sorry

end arithmetic_example_l213_213123


namespace isosceles_trapezoid_area_l213_213098

theorem isosceles_trapezoid_area 
  (a b x y : ℝ) 
  (h1 : a = 20) 
  (h2 : x = 11.11)
  (h3 : y = 2.22)
  (h4 : b = y + 20)
  (h5 : h = 0.6 * x)
  (h6 : arcsin 0.6 = θ) : 
  trapezoid_area a b h = 74.07 := 
  sorry

end isosceles_trapezoid_area_l213_213098


namespace compote_fractional_decrease_l213_213552

def initial_volume (V : ℝ) := V
def reduced_volume_first_step (V : ℝ) := (2 / 3) * V
def reduced_volume_second_step (V1 : ℝ) := (3 / 4) * V1
def fractional_decrease (V1 V2 : ℝ) := (V1 - V2) / V1

theorem compote_fractional_decrease (V : ℝ) :
  fractional_decrease (reduced_volume_first_step V) (reduced_volume_second_step (reduced_volume_first_step V)) = 1 / 4 :=
by
  sorry

end compote_fractional_decrease_l213_213552


namespace perpendicularity_of_intersection_lines_l213_213103

theorem perpendicularity_of_intersection_lines
  (A B C D E F G H I J K L : Point)
  (sq1 : Square ABCD)
  (sq2 : Square EFGH)
  (h1 : Line EF ∩ Line AB = {J})
  (h2 : Line FG ∩ Line BC = {K})
  (h3 : Line GH ∩ Line CD = {L})
  (h4 : Line HE ∩ Line DA = {I}) :
  Perpendicular (Line IK) (Line JL) :=
sorry

end perpendicularity_of_intersection_lines_l213_213103


namespace janet_cat_litter_cost_l213_213765

def cat_litter_cost (days : ℕ) (weekdays_usage weekend_usage : ℝ) (container_weight container_price : ℝ) 
  (discount1 discount2 : ℝ) (threshold1 threshold2 : ℕ) : ℝ := 
  let weeks := days / 7
  let total_usage := weeks * (5 * weekdays_usage + 2 * weekend_usage)
  let containers := ⌈total_usage / container_weight⌉
  let price_per_container := if containers >= threshold1 then container_price * discount1 else container_price
  let total_price := price_per_container * containers
  if containers > threshold2 then total_price * discount2 else total_price

theorem janet_cat_litter_cost :
  cat_litter_cost 210 0.3 0.8 45 21 0.9 1 3 5 = 56.7 :=
by sorry

end janet_cat_litter_cost_l213_213765


namespace min_value_of_k_is_sqrt2_l213_213311

noncomputable def min_k_satisfies_inequality : ℝ :=
  Inf { k : ℝ | ∀ (x y : ℝ), sqrt x + sqrt y ≤ k * sqrt (x + y) }

theorem min_value_of_k_is_sqrt2 : min_k_satisfies_inequality = sqrt 2 :=
sorry

end min_value_of_k_is_sqrt2_l213_213311


namespace music_workshop_average_age_l213_213745

theorem music_workshop_average_age:
  (avg_females : ℕ) (avg_males : ℕ) (avg_elderly : ℕ) (num_females : ℕ) (num_males : ℕ) (num_elderly : ℕ)
  (h1 : avg_females = 34) (h2 : avg_males = 32) (h3 : avg_elderly = 60)
  (h4 : num_females = 8) (h5 : num_males = 12) (h6 : num_elderly = 5) :
  ( (num_females * avg_females + num_males * avg_males + num_elderly * avg_elderly) / (num_females + num_males + num_elderly) ) = 38.24 :=
by
  sorry

end music_workshop_average_age_l213_213745


namespace angle_at_6_15_is_obtuse_l213_213912

/-- Define the clock angle function -/
noncomputable def clock_angle (hours : ℕ) (minutes : ℕ) : ℝ :=
  let hour_position := ((hours % 12) + (minutes / 60.0)) * 30
  let minute_position := (minutes / 60.0) * 360
  let angle := abs (hour_position - minute_position)
  if angle > 180 then 360 - angle else angle 

/-- Problem Statement: The angle at 6:15 is obtuse -/
theorem angle_at_6_15_is_obtuse : 90 < clock_angle 6 15 ∧ clock_angle 6 15 < 180 :=
by
  sorry

end angle_at_6_15_is_obtuse_l213_213912


namespace smaller_cube_side_length_l213_213526

noncomputable def side_length_of_smaller_cube : ℝ := 2 / 3

theorem smaller_cube_side_length (R : ℝ) (d1 d2 : ℝ) :
  ∀ x : ℝ, (R = real.sqrt 3) → 
  d1 = (1 + x) → 
  d2 = (x * real.sqrt 2 / 2) → 
  R * R = (d1 * d1 + d2 * d2) → 
  x = side_length_of_smaller_cube :=
by
  sorry

end smaller_cube_side_length_l213_213526


namespace part_1_part_2_part_3_l213_213556

-- Problem conditions and requirements as Lean definitions and statements
def y_A (x : ℝ) : ℝ := (2 / 5) * x
def y_B (x : ℝ) : ℝ := -(1 / 5) * x^2 + 2 * x

-- Part (1)
theorem part_1 : y_A 10 = 4 := sorry

-- Part (2)
theorem part_2 (m : ℝ) (h : m > 0) : y_A m = y_B m → m = 3 := sorry

-- Part (3)
def W (x_A x_B : ℝ) : ℝ := y_A x_A + y_B x_B

theorem part_3 (x_A x_B : ℝ) (h : x_A + x_B = 32) : 
  (∀ x_A' x_B', x_A' + x_B' = 32 → W x_A x_B ≥ W x_A' x_B') ∧ W x_A x_B = 16 :=
begin
  sorry
end

end part_1_part_2_part_3_l213_213556


namespace perpendicular_condition_parallel_condition_l213_213651

-- Vector definition and conditions
def vector_a : ℝ × ℝ := (1, 2)
def vector_b : ℝ × ℝ := (-3, 2)

-- First problem: perpendicular condition
theorem perpendicular_condition (k : ℝ) : 
  let lhs := (k - 3, 2 * k + 2) in
  let rhs := (10, -4) in
  (vector_a, vector_b) = ((1, 2), (-3, 2)) →
  10 * lhs.1 - 4 * lhs.2 = 0 →
  k = 19 := 
by tautology

-- Second problem: parallel condition
theorem parallel_condition (k : ℝ) : 
  let lhs := (k - 3, 2 * k + 2) in
  let rhs := (10, -4) in
  (vector_a, vector_b) = ((1, 2), (-3, 2)) →
  -4 * lhs.1 - 10 * lhs.2 = 0 →
  k = -(1/3) :=
by tautology

end perpendicular_condition_parallel_condition_l213_213651


namespace problem_equivalence_l213_213312

noncomputable def S (n : ℕ) : ℝ := 2 ^ n

noncomputable def a : ℕ → ℝ
| 1       := 2
| (n + 1) := 2 ^ n

noncomputable def b (n : ℕ) : ℝ := a n * Real.log2 (a n)

noncomputable def T (n : ℕ) : ℝ :=
  ∑ k in finset.range n, b (k + 1)

theorem problem_equivalence (n : ℕ) : T n = (n - 2) * 2 ^ n + 4 := sorry

end problem_equivalence_l213_213312


namespace max_m_with_triangle_property_l213_213581

def is_triangle_property (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def has_triangle_property (S : Finset ℕ) : Prop :=
  ∀ T : Finset ℕ, T.card = 3 → T.to_list.sorted_comb.includes (is_triangle_property T)

def all_seven_subsets_have_triangle_property (S : Finset ℕ) : Prop :=
  ∀ T : Finset ℕ, T.card = 7 → has_triangle_property T

theorem max_m_with_triangle_property :
  all_seven_subsets_have_triangle_property (Finset.range 27).insert 1.insert 3.insert 5.insert 7.insert 9.insert 11.insert 13.insert 15.insert 17.insert 19.insert 21.insert 23.insert 25 :=
  all_seven_subsets_have_triangle_property (Finset.range 28).insert 1.insert 3.insert 5.insert 7.insert 9.insert 11.insert 13.insert 15.insert 17.insert 19.insert 21.insert 23.insert 25 → false :=
sorry

end max_m_with_triangle_property_l213_213581


namespace ceil_of_neg_frac_squared_l213_213200

-- Define the negated fraction
def neg_frac : ℚ := -7 / 4

-- Define the squared value of the negated fraction
def squared_value : ℚ := neg_frac ^ 2

-- Define the ceiling function applied to the squared value
def ceil_squared_value : ℤ := Int.ceil squared_value

-- Prove that the ceiling of the squared value is 4
theorem ceil_of_neg_frac_squared : ceil_squared_value = 4 := 
by sorry

end ceil_of_neg_frac_squared_l213_213200


namespace math_problem_proof_l213_213890

def eight_to_zero : ℝ := 1
def log_base_10_of_100 : ℝ := 2

theorem math_problem_proof : eight_to_zero - log_base_10_of_100 = -1 :=
by sorry

end math_problem_proof_l213_213890


namespace maximize_trig_expression_not_among_choices_l213_213246

theorem maximize_trig_expression_not_among_choices :
  let B := Float.pi / 6 in
  let A1 := 30 * Float.pi / 180 in
  let A2 := 90 * Float.pi / 180 in
  let A3 := 150 * Float.pi / 180 in
  let A4 := 210 * Float.pi / 180 in
  let target := 2 * (Math.sin ((Float.ofNat 120) * Float.pi / 360 + Float.pi / 6)) in
  ∀ A,
    (sin (A / 2) + sqrt 3 * cos (A / 2) = target) ->
    (A ≠ A1 ∧ A ≠ A2 ∧ A ≠ A3 ∧ A ≠ A4) :=
by
  sorry

end maximize_trig_expression_not_among_choices_l213_213246


namespace isosceles_trapezoid_area_l213_213101

theorem isosceles_trapezoid_area 
  (a b x y : ℝ) 
  (h1 : a = 20) 
  (h2 : x = 11.11)
  (h3 : y = 2.22)
  (h4 : b = y + 20)
  (h5 : h = 0.6 * x)
  (h6 : arcsin 0.6 = θ) : 
  trapezoid_area a b h = 74.07 := 
  sorry

end isosceles_trapezoid_area_l213_213101


namespace ceiling_of_square_of_neg_7_over_4_is_4_l213_213167

theorem ceiling_of_square_of_neg_7_over_4_is_4 : 
  Real.ceil ((-7 / 4 : Real) ^ 2) = 4 := by
  sorry

end ceiling_of_square_of_neg_7_over_4_is_4_l213_213167


namespace buses_passed_on_highway_l213_213986

-- Definitions for conditions in the problem.
def Austin_to_SanAntonio_departure_interval : ℕ := 45  -- in minutes
def SanAntonio_to_Austin_departure_interval : ℕ := 30 -- in minutes
def trip_duration : ℕ := 360                       -- in minutes (6 hours)

theorem buses_passed_on_highway :
  -- Start time and departures adjusted to minutes since 12:00 PM for simplicity
  (∃ (n : ℕ), n * 30 + 15 = t ∧ ∀ t ∈ (12 * 60 + 0) + (umin (12 * 60 + 0 + 45 * m for m in range(497/45))),
  7 * 60 + 30 <= t ∧ t <= 18 * 60) →
  -- Some logical construct relating the number of Austin-bound buses.
  -- Proof should establish that the San Antonio-bound bus (departing at 12:15 PM) passes 9 Austin-bound buses
  -- on the highway during its 6-hour trip from 12:15 PM to 6:15 PM.
  (∃ n : ℕ, t = 5515 + 30 * n ∧ n = 9) :=
by
  sorry

end buses_passed_on_highway_l213_213986


namespace pi_is_irrational_l213_213430

theorem pi_is_irrational : Irrational real.pi := 
sorry

end pi_is_irrational_l213_213430


namespace no_intersecting_segments_l213_213933

-- Assume we have n red points and n blue points
def n : ℕ := sorry
def RedPoint : Type := sorry  -- Type representing Red points
def BluePoint : Type := sorry -- Type representing Blue points

-- Assume we have sets of red and blue points
axiom red_points : finset RedPoint
axiom blue_points : finset BluePoint

-- Condition: no three points are collinear
axiom no_three_collinear : ∀ (p1 p2 p3 : RedPoint ∪ BluePoint), 
  p1 ≠ p2 → p2 ≠ p3 → p1 ≠ p3 → ¬ collinear p1 p2 p3

-- To prove: It is possible to draw n non-intersecting line segments connecting each red point to a blue point
theorem no_intersecting_segments :
  ∃ (match_points : RedPoint → BluePoint), 
    (∀ r1 r2 : RedPoint, r1 ≠ r2 → 
     ∀ b1 b2 : BluePoint, b1 ≠ b2 → ¬ segments_intersect (r1, match_points r1) (r2, match_points r2)) :=
sorry

end no_intersecting_segments_l213_213933


namespace square_neg_2x_squared_l213_213024

theorem square_neg_2x_squared (x : ℝ) : (-2 * x ^ 2) ^ 2 = 4 * x ^ 4 :=
by
  sorry

end square_neg_2x_squared_l213_213024


namespace find_n_eq_1050_l213_213615

theorem find_n_eq_1050 : ∃ (n : ℕ), 2^6 * 3^3 * n = 10! ∧ n = 1050 :=
by
  use 1050
  split
  · calc
    2^6 * 3^3 * 1050 = 10! := sorry
  · rfl

end find_n_eq_1050_l213_213615


namespace length_of_CE_l213_213402

-- Define the points C, D, B, A, E and their conditions
variables (A E C D B : Type)
variables [metric_space A] [metric_space E] [metric_space C] [metric_space D] [metric_space B]
variables {AE_line : A × E -> Prop}
variables {CE_line : C × E -> Prop}

-- Define the lengths 
variables (AB : ℝ) (CD : ℝ) (AE : ℝ) (CE : ℝ)
variables {x : ℝ}

-- Conditions
axiom C_not_on_AE : ¬ AE_line (C, D)
axiom CD_perp_AE : C ≠ D ∧ AE_line (D, E) ∧ (∀ x, AE_line (D, x) → CD ⊥ AE)
axiom B_on_CE : CE_line (C, E)
axiom AB_perp_CE : A ≠ B ∧ CE_line (B, E) ∧ (∀ x, CE_line (B, x) → AB ⊥ CE)

axiom AB_length : AB = 6
axiom CD_length : CD = 10
axiom AE_length : AE = 7

-- Prove the length of CE
theorem length_of_CE : CE = 35 / 3 :=
by
sory -- Placeholder for the proof

end length_of_CE_l213_213402


namespace integer_solution_n_l213_213910

theorem integer_solution_n 
  (n : Int) 
  (h1 : n + 13 > 15) 
  (h2 : -6 * n > -18) : 
  n = 2 := 
sorry

end integer_solution_n_l213_213910


namespace ceil_of_neg_frac_squared_l213_213199

-- Define the negated fraction
def neg_frac : ℚ := -7 / 4

-- Define the squared value of the negated fraction
def squared_value : ℚ := neg_frac ^ 2

-- Define the ceiling function applied to the squared value
def ceil_squared_value : ℤ := Int.ceil squared_value

-- Prove that the ceiling of the squared value is 4
theorem ceil_of_neg_frac_squared : ceil_squared_value = 4 := 
by sorry

end ceil_of_neg_frac_squared_l213_213199


namespace nearest_integer_to_power_l213_213915

theorem nearest_integer_to_power (x y : ℝ) (h₁ : x = 3) (h₂ : y = sqrt 5) :
  Int.round ((x + y)^5) = 3936 := 
by 
  sorry

end nearest_integer_to_power_l213_213915


namespace find_k_l213_213649

noncomputable section

variables {a b k : ℝ}

theorem find_k 
  (h1 : 4^a = k) 
  (h2 : 9^b = k)
  (h3 : 1 / a + 1 / b = 2) : 
  k = 6 :=
sorry

end find_k_l213_213649


namespace isosceles_trapezoid_area_l213_213097

theorem isosceles_trapezoid_area 
  (a b x y : ℝ) 
  (h1 : a = 20) 
  (h2 : x = 11.11)
  (h3 : y = 2.22)
  (h4 : b = y + 20)
  (h5 : h = 0.6 * x)
  (h6 : arcsin 0.6 = θ) : 
  trapezoid_area a b h = 74.07 := 
  sorry

end isosceles_trapezoid_area_l213_213097


namespace power_series_expansion_property_l213_213130

noncomputable def a : ℕ → ℤ
| 0     := 1
| 1     := 2
| (n+2) := 2 * a (n + 1) + a n

theorem power_series_expansion_property (n : ℕ) : ∃ m : ℕ, a n ^ 2 + a (n + 1) ^ 2 = a m := 
sorry

end power_series_expansion_property_l213_213130


namespace ceil_square_neg_fraction_l213_213181

theorem ceil_square_neg_fraction :
  let x := (-7) / 4
  let y := x^2
  let z := Real.ceil y
  z = 4 := 
by
  sorry

end ceil_square_neg_fraction_l213_213181


namespace ceil_square_neg_fraction_l213_213173

theorem ceil_square_neg_fraction :
  let x := (-7) / 4
  let y := x^2
  let z := Real.ceil y
  z = 4 := 
by
  sorry

end ceil_square_neg_fraction_l213_213173


namespace probability_two_dice_same_number_l213_213825

theorem probability_two_dice_same_number (n : ℕ) (sides : ℕ) (h_n : n = 8) (h_sides : sides = 6):
  (∃ (prob : ℝ), prob = 1) :=
by
  sorry

end probability_two_dice_same_number_l213_213825


namespace angle_b_is_sixty_max_area_triangle_l213_213303

variables (a b c A B C : Real) (A_pos B_pos C_pos : Prop)
variables (A_sum : A + B + C = π) -- angles of a triangle
variables (triangle_sides : a = 2 * sin(A) ∨ b = 2 * sin(B) ∨ c = 2 * sin(C)) -- Law of Sines
variables (condition : (a + c) / b = cos(C) + sqrt(3) * sin(C))

noncomputable theory

-- Part 1: Show B is 60 degrees
theorem angle_b_is_sixty (hA : A_pos A) (hB : B_pos B) (hC : C_pos C) :
  B = π / 3 :=
  sorry

-- Part 2: Given b = 2, show the maximum area of the triangle is sqrt(3)
theorem max_area_triangle (hb : b = 2) :
  (∃ a c, (a + c) / b = cos(C) + sqrt(3) * sin(C) ∧
          let S := 1 / 2 * a * c * sin(B) in
          ∀ a' c', (a' + c') / b = cos(C) + sqrt(3) * sin(C) → 
                   1 / 2 * a' * c' * sin(B) ≤ S ∧ S = sqrt(3)) :=
  sorry

end angle_b_is_sixty_max_area_triangle_l213_213303


namespace handrail_length_correct_l213_213968

noncomputable def handrail_length (radius height total_turns_in_degrees : ℝ) : ℝ :=
  let perimeter := 2 * real.pi * radius
  let arc_length := (total_turns_in_degrees / 360) * perimeter
  real.sqrt (height^2 + arc_length^2)

theorem handrail_length_correct :
  handrail_length 4 15 450 ≈ 17.4 :=
by sorry

end handrail_length_correct_l213_213968


namespace superabundant_count_l213_213406

noncomputable def g (n : ℕ) : ℕ :=
  if n = 1 then 1
  else (List.prod (List.filter (fun d => n % d = 0) (List.range (n + 1))))

def is_superabundant (n : ℕ) : Prop :=
  g (g n) = n^2 + 2 * n

theorem superabundant_count : Finset.card (Finset.filter is_superabundant (Finset.range (1000))) = 1 :=
by
  sorry

end superabundant_count_l213_213406


namespace right_triangle_ratio_l213_213286

theorem right_triangle_ratio (x y : ℝ) (h1 : x > y) (h2 : y > 0) (h3 : (x - y)^2 + x^2 = (x + y)^2) : x / y = 4 :=
by
  sorry

end right_triangle_ratio_l213_213286


namespace problem_2_8_3_4_7_2_2_l213_213116

theorem problem_2_8_3_4_7_2_2 : 2 + 8 * 3 - 4 + 7 * 2 / 2 = 29 := by
  sorry

end problem_2_8_3_4_7_2_2_l213_213116


namespace parabola_intersection_sum_l213_213878

theorem parabola_intersection_sum (a1 a2 a3 a4 b1 b2 b3 b4 : ℝ)
  (h1 : ∀ x, x^2 + a1*x + b1 = x^2 + a2*x + b2 → x ∈ {α | ∃ β, x = β})
  (h2 : ∀ x, x^2 + a1*x + b1 = x^2 + a3*x + b3 → x ∈ {α | ∃ β, x = β})
  (h3 : ∀ x, x^2 + a4*x + b4 = x^2 + a2*x + b2 → x ∈ {γ | ∃ δ, x = δ})
  (h4 : ∀ x, x^2 + a4*x + b4 = x^2 + a3*x + b3 → x ∈ {γ | ∃ δ, x = δ})
  (h_order : (α < β) ∧ (δ < γ)) :
  α + δ = β + γ := 
begin
  sorry
end

end parabola_intersection_sum_l213_213878


namespace sum_first_10_log_a_l213_213313

-- Given sum of the first n terms of the sequence
def S (n : ℕ) : ℕ := 2^n - 1

-- Function to get general term log_2 a_n
def log_a (n : ℕ) : ℕ := n - 1

-- The statement to prove
theorem sum_first_10_log_a : (List.range 10).sum = 45 := by 
  sorry

end sum_first_10_log_a_l213_213313


namespace no_integers_satisfy_eq_except_2_and_6_l213_213792

theorem no_integers_satisfy_eq_except_2_and_6 (p : ℕ) (hp : ∃ n : ℕ, p = n * (n + 1) ∧ n > 2) :
  (¬ (∃ (x : fin p → ℤ), (∑ i, x i ^ 2) - (4 / (4 * p + 1) * (∑ i, x i) ^ 2) = 1)) :=
by
  intro h
  have : p ∈ {2, 6} ∨ p ∉ {2, 6} := em (p ∈ {2, 6})
  cases this
  · cases this; contradiction
  · sorry

end no_integers_satisfy_eq_except_2_and_6_l213_213792


namespace part_one_part_two_l213_213309

-- Definitions based on given conditions
def point := (ℝ, ℝ)

def A : point := (-2*Real.sqrt 2, 0)
def B : point := (2*Real.sqrt 2, 0)
def distance (P Q : point) : ℝ := Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

-- We start by defining the curve C
def curve_C (x y : ℝ) : Prop := (x^2 / 9) + y^2 = 1

-- The first property to prove: Given the conditions, point P lies on the ellipse
theorem part_one (P : point) (hP_dist: distance P A + distance P B = 6) : curve_C P.1 P.2 := 
sorry

-- The second property: The existence of point T on the x-axis such that k_MT * k_NT is constant
theorem part_two (m t : ℝ) (h_t: t = 3 ∨ t = -3) :
  ∃ T : point, T.1 = t ∧ T.2 = 0 ∧ ∀ M N : point, 
  (curve_C M.1 M.2) → (curve_C N.1 N.2) →
  M ≠ N → M ≠ T → N ≠ T →
  let k_MT := (M.2 - T.2) / (M.1 - T.1),
      k_NT := (N.2 - T.2) / (N.1 - T.1)
  in k_MT * k_NT = (-2 / 9 : ℚ) ∨ k_MT * k_NT = (-1 / 18 : ℚ) :=
sorry

end part_one_part_two_l213_213309


namespace trapezoid_of_segment_mean_l213_213843

open Real

structure Quadrilateral :=
(A B C D : Point)
(AD BC AB DC : ℝ)

def midpoint (p q : Point) : Point := (p + q) / 2

noncomputable def is_trapezoid (ABCD : Quadrilateral) : Prop :=
let E := midpoint ABCD.A ABCD.D in
let F := midpoint ABCD.B ABCD.C in
let EF := dist E F in
EF = (ABCD.AB + ABCD.DC) / 2 → parallel ABCD.AB ABCD.DC

axiom dist (p q : Point) : ℝ

axiom parallel (l m : Line) : Prop

axiom Point_on_line (l : Line) (p : Point) : Prop

instance : add_comm_group Point :=
{ add := λ x y, sorry,
  add_assoc := sorry,
  zero := sorry,
  zero_add := sorry,
  add_zero := sorry,
  neg := sorry,
  add_left_neg := sorry,
  add_comm := sorry }

variable quadrilateral : Quadrilateral

theorem trapezoid_of_segment_mean :
  is_trapezoid quadrilateral :=
by
  sorry

end trapezoid_of_segment_mean_l213_213843


namespace number_of_meetings_l213_213419

def odell_speed : ℝ := 260  -- meters per minute
def odell_radius : ℝ := 52  -- meters
def kershaw_speed : ℝ := 310  -- meters per minute
def kershaw_radius : ℝ := 62  -- meters
def total_time : ℝ := 30  -- minutes

theorem number_of_meetings : 
  let co := 2 * Real.pi * odell_radius   -- Odell's circumference
  let ck := 2 * Real.pi * kershaw_radius -- Kershaw's circumference
  let wo := (odell_speed / co) * 2 * Real.pi  -- Odell's angular speed
  let wk := (kershaw_speed / ck) * 2 * Real.pi -- Kershaw's angular speed
  let relative_angular_speed := wo + wk  -- relative angular speed
  let time_to_meet := 2 * Real.pi / relative_angular_speed  -- time to meet once
  let total_meetings := (total_time / time_to_meet : ℝ).toInt
  total_meetings = 47 :=
by
  sorry

end number_of_meetings_l213_213419


namespace sequence_floor_sum_l213_213680

noncomputable def a_seq : ℕ → ℕ
| 0 := 1
| (n+1) := a_seq n ^ 2 + a_seq n

def series_sum (n : ℕ) : ℝ :=
(∑ i in Finset.range n, (a_seq i) / (a_seq i + 1))

theorem sequence_floor_sum :
  (⌊series_sum 2016⌋ = 2015) :=
sorry

end sequence_floor_sum_l213_213680


namespace melissa_initial_oranges_l213_213417

theorem melissa_initial_oranges (initially remaining taken : ℕ) (h₁ : remaining = 51) (h₂ : taken = 19) (h₃ : remaining = initially - taken) :
  initially = 70 :=
by
  rw [h₁, h₂] at h₃
  linarith

end melissa_initial_oranges_l213_213417


namespace binomial_sum_mod_l213_213578

def primitive_root_of_unity (ζ : ℂ) : Prop :=
  ζ^4 = 1 ∧ ζ ≠ 1 ∧ ζ^2 ≠ 1 ∧ ζ^3 ≠ 1

theorem binomial_sum_mod : 
  (∑ k in Finset.range 502, Nat.choose 2005 (4 * k)) % 1000 = 64 := 
by {
  -- Given conditions
  let ζ := complex.exp (2 * π * I / 4),
  have hζ : primitive_root_of_unity ζ := by sorry,

  -- The actual proof goes here, but it's omitted as required.
  sorry
}

end binomial_sum_mod_l213_213578


namespace complex_expression_ab_l213_213273

open Complex

theorem complex_expression_ab :
  ∀ (a b : ℝ), (2 + 3 * I) / I = a + b * I → a * b = 6 :=
by
  intros a b h
  sorry

end complex_expression_ab_l213_213273


namespace part_a_correct_part_b_correct_l213_213991

-- Define the alphabet and mapping
inductive Letter
| C | H | M | O
deriving DecidableEq, Inhabited

open Letter

def letter_to_base4 (ch : Letter) : ℕ :=
  match ch with
  | C => 0
  | H => 1
  | M => 2
  | O => 3

def word_to_base4 (word : List Letter) : ℕ :=
  word.foldl (fun acc ch => acc * 4 + letter_to_base4 ch) 0

def base4_to_letter (n : ℕ) : Letter :=
  match n with
  | 0 => C
  | 1 => H
  | 2 => M
  | 3 => O
  | _ => C -- This should not occur if input is in valid base-4 range

def base4_to_word (n : ℕ) (size : ℕ) : List Letter :=
  if size = 0 then []
  else
    let quotient := n / 4
    let remainder := n % 4
    base4_to_letter remainder :: base4_to_word quotient (size - 1)

-- The size of the words is fixed at 8
def word_size : ℕ := 8

noncomputable def part_a : List Letter :=
  base4_to_word 2017 word_size

theorem part_a_correct :
  part_a = [H, O, O, H, M, C] := by
  sorry

def given_word : List Letter :=
  [H, O, M, C, H, O, M, C]

noncomputable def part_b : ℕ :=
  word_to_base4 given_word + 1 -- Adjust for zero-based indexing

theorem part_b_correct :
  part_b = 29299 := by
  sorry

end part_a_correct_part_b_correct_l213_213991


namespace inequality_solution_exists_at_minimum_a_l213_213257

theorem inequality_solution_exists_at_minimum_a :
  ∃ x : ℝ, x ≥ -2 ∧ (x^3 - 3 * x + 3 - x / Real.exp x) - (1 - 1 / Real.exp 1) ≤ 0 :=
sorry

end inequality_solution_exists_at_minimum_a_l213_213257


namespace sum_of_special_primes_l213_213592

theorem sum_of_special_primes : 
  let primes_with_no_solution := {p : ℕ | prime p ∧ ¬ ∃ x : ℤ, 5 * (10 * x + 2) ≡ 3 [MOD p]} in
  ∃ p1 p2, p1 ∈ primes_with_no_solution ∧ p2 ∈ primes_with_no_solution ∧ p1 ≠ p2 ∧ p1 + p2 = 7 :=
by
  sorry

end sum_of_special_primes_l213_213592


namespace find_m_l213_213048

def g (n : ℤ) : ℤ :=
  if Int.Odd n then n + 5 else n / 2

theorem find_m (m : ℤ) (h_odd : Int.Odd m) (h_eq : g (g (g m)) = 39) : m = 63 :=
sorry

end find_m_l213_213048


namespace arithmetic_sequence_a13_l213_213753

variable (a1 d : ℤ)

theorem arithmetic_sequence_a13 (h : a1 + 2 * d + a1 + 8 * d + a1 + 26 * d = 12) : a1 + 12 * d = 4 :=
by
  sorry

end arithmetic_sequence_a13_l213_213753


namespace martha_black_butterflies_l213_213803

-- Define the hypotheses
variables (total_butterflies blue_butterflies yellow_butterflies black_butterflies : ℕ)

-- Given conditions
def martha_collection_conditions : Prop :=
  total_butterflies = 19 ∧
  blue_butterflies = 6 ∧
  blue_butterflies = 2 * yellow_butterflies

-- The statement we want to prove
theorem martha_black_butterflies : martha_collection_conditions total_butterflies blue_butterflies yellow_butterflies black_butterflies →
  black_butterflies = 10 :=
sorry

end martha_black_butterflies_l213_213803


namespace trapezoid_area_l213_213079

-- Define the problem statement
theorem trapezoid_area 
  (a b h: ℝ)
  (b₁ b₂: ℝ)
  (θ: ℝ) 
  (h₃: θ = Real.arcsin 0.6)
  (h₄: a = 20)
  (h₅: b = a - 2 * b₁ * Real.sin θ) 
  (h₆: h = b₁ * Real.cos θ) 
  (h₇: θ = Real.arcsin (3/5)) 
  (circum: isosceles_trapezoid_circumscribed a b₁ b₂) :
  ((1 / 2) * (a + b₂) * h = 2000 / 27) :=
by sorry

end trapezoid_area_l213_213079


namespace karlson_maximum_candies_l213_213821

theorem karlson_maximum_candies (n : ℕ) (h : n = 30) : 
  max_candies n = (n * (n - 1)) / 2 := 
by
  sorry

-- Additional definitions required to support max_candies function can be added as needed.
-- Definitions should follow the conditions in a) and avoid steps from the solution in b).

def max_candies : ℕ → ℕ
| 0 := 0
| n := (n * (n - 1)) / 2

end karlson_maximum_candies_l213_213821


namespace probability_sum_leq_4_is_one_third_l213_213939

theorem probability_sum_leq_4_is_one_third :
  let balls := {1, 2, 3, 4}
  let draws := {x | ∃ a b ∈ balls, a < b ∧ x = {a, b}}
  let favorable_draws := {x | x ∈ draws ∧ (x.sum ≤ 4)}
  ∃ (p : ℚ), p = (favorable_draws.to_finset.card / draws.to_finset.card) →
  p = 1 / 3 :=
by
  sorry

end probability_sum_leq_4_is_one_third_l213_213939


namespace find_w_l213_213994

noncomputable def line_p(t : ℝ) : (ℝ × ℝ) := (2 + 3 * t, 5 + 2 * t)
noncomputable def line_q(u : ℝ) : (ℝ × ℝ) := (-3 + 3 * u, 7 + 2 * u)

def vector_DC(t u : ℝ) : ℝ × ℝ := ((2 + 3 * t) - (-3 + 3 * u), (5 + 2 * t) - (7 + 2 * u))

def w_condition (w1 w2 : ℝ) : Prop := w1 + w2 = 3

theorem find_w (t u : ℝ) :
  ∃ w1 w2 : ℝ, 
    w_condition w1 w2 ∧ 
    (∃ k : ℝ, 
      sorry -- This is a placeholder for the projection calculation
    )
    :=
  sorry -- This is a placeholder for the final proof

end find_w_l213_213994


namespace length_BC_l213_213331

-- Definition of points, lengths and right triangles.
variable (A B C D : Point)

-- Triangle ABD is a right triangle.
axiom ABD_right : RightTriangle A B D

-- Triangle ABC is a right triangle.
axiom ABC_right : RightTriangle A B C

-- Given lengths for specific segments.
axiom AD_length : distance A D = 20 + 25
axiom BD_length : distance B D = 53
axiom AC_length : distance A C = 20
axiom AB_height : distance A B = 15

-- Prove the length of segment BC.
theorem length_BC : distance B C = 25 :=
sorry

end length_BC_l213_213331


namespace shortest_distance_l213_213675

-- Define the function f(x) = ln x - x^2
def f (x : ℝ) : ℝ := Real.log x - x^2

-- Define the line equation x + y - 3 = 0
def line (x y : ℝ) : Prop := x + y - 3 = 0

-- Define the point A on the graph of function f
def pointA (x y : ℝ) : Prop := y = f x

-- The shortest distance between the arbitrary point on the graph and a point on the line
theorem shortest_distance :
  ∀ x y, pointA x y → ∃ b₁ b₂, line b₁ b₂ ∧ 
  (Real.dist (x, y) (b₁, b₂) = 3 * Real.sqrt 2 / 2) :=
sorry

end shortest_distance_l213_213675


namespace probability_two_balls_red_l213_213938

variables (total_balls red_balls blue_balls green_balls picked_balls : ℕ)

def probability_of_both_red
  (h_total_balls : total_balls = 8)
  (h_red_balls : red_balls = 3)
  (h_blue_balls : blue_balls = 2)
  (h_green_balls : green_balls = 3)
  (h_picked_balls : picked_balls = 2) : ℚ :=
  (red_balls / total_balls) * ((red_balls - 1) / (total_balls - 1))

theorem probability_two_balls_red (h_total_balls : total_balls = 8)
  (h_red_balls : red_balls = 3)
  (h_blue_balls : blue_balls = 2)
  (h_green_balls : green_balls = 3)
  (h_picked_balls : picked_balls = 2)
  (h_prob : probability_of_both_red total_balls red_balls blue_balls green_balls picked_balls 
    h_total_balls h_red_balls h_blue_balls h_green_balls h_picked_balls = 3 / 28) : 
  probability_of_both_red total_balls red_balls blue_balls green_balls picked_balls 
    h_total_balls h_red_balls h_blue_balls h_green_balls h_picked_balls = 3 / 28 := 
sorry

end probability_two_balls_red_l213_213938


namespace equivalent_single_percentage_change_l213_213970

theorem equivalent_single_percentage_change :
  let original_price : ℝ := 250
  let num_items : ℕ := 400
  let first_increase : ℝ := 0.15
  let second_increase : ℝ := 0.20
  let discount : ℝ := -0.10
  let third_increase : ℝ := 0.25

  -- Calculations
  let price_after_first_increase := original_price * (1 + first_increase)
  let price_after_second_increase := price_after_first_increase * (1 + second_increase)
  let price_after_discount := price_after_second_increase * (1 + discount)
  let final_price := price_after_discount * (1 + third_increase)

  -- Calculate percentage change
  let percentage_change := ((final_price - original_price) / original_price) * 100

  percentage_change = 55.25 :=
by
  sorry

end equivalent_single_percentage_change_l213_213970


namespace final_weight_is_correct_l213_213395

def initial_weight : ℝ := 1.5 + 0.5 + 2
def weight_after_chocolate_bars : ℝ := initial_weight + (initial_weight * 0.5)
def weight_after_popcorn : ℝ := weight_after_chocolate_bars - 0.25 + 0.75
def weight_after_cookies : ℝ := weight_after_popcorn * 2
def final_weight : ℝ := weight_after_cookies - (1.5 / 2)

theorem final_weight_is_correct : final_weight = 12.25 := by
  sorry

end final_weight_is_correct_l213_213395


namespace area_triangle_MOI_is_11_over_4_l213_213740

-- Definitions for the vertices of triangle ABC
def triangle_ABC := {A B C : ℝ × ℝ // dist A B = 15 ∧ dist A C = 14 ∧ dist B C = 13}

-- Definitions for the circumcenter O and incenter I of the triangle ABC
def circumcenter (t : triangle_ABC) : ℝ × ℝ := sorry
def incenter (t : triangle_ABC) : ℝ × ℝ := sorry

-- Definitions for the circle center M tangent to legs AC, BC and incircle of ABC
def circle_center_M (t : triangle_ABC) : ℝ × ℝ := sorry

-- Calculate the area of triangle MOI
def area_triangle (O I M : ℝ × ℝ) : ℝ := 
(abs (fst O * snd I + fst I * snd M + fst M * snd O - snd O * fst I - snd I * fst M - snd M * fst O)) / 2

theorem area_triangle_MOI_is_11_over_4 (t : triangle_ABC) :
    let O := circumcenter t
    let I := incenter t
    let M := circle_center_M t
    area_triangle O I M = 11 / 4 := 
by
  sorry

end area_triangle_MOI_is_11_over_4_l213_213740


namespace find_a_parallel_find_a_perpendicular_l213_213335

open Real

def line_parallel (p1 p2 q1 q2 : (ℝ × ℝ)) : Prop :=
  let k1 := (q2.2 - q1.2) / (q2.1 - q1.1)
  let k2 := (p2.2 - p1.2) / (p2.1 - p1.1)
  k1 = k2

def line_perpendicular (p1 p2 q1 q2 : (ℝ × ℝ)) : Prop :=
  let k1 := (q2.2 - q1.2) / (q2.1 - q1.1)
  let k2 := (p2.2 - p1.2) / (p2.1 - p1.1)
  k1 * k2 = -1

theorem find_a_parallel (a : ℝ) :
  line_parallel (3, a) (a-1, 2) (1, 2) (-2, a+2) ↔ a = 1 ∨ a = 6 :=
by sorry

theorem find_a_perpendicular (a : ℝ) :
  line_perpendicular (3, a) (a-1, 2) (1, 2) (-2, a+2) ↔ a = 3 ∨ a = -4 :=
by sorry

end find_a_parallel_find_a_perpendicular_l213_213335


namespace tiffany_daily_miles_l213_213569

-- Definitions for running schedule
def billy_sunday_miles := 1
def billy_monday_miles := 1
def billy_tuesday_miles := 1
def billy_wednesday_miles := 1
def billy_thursday_miles := 1
def billy_friday_miles := 1
def billy_saturday_miles := 1

def tiffany_wednesday_miles := 1 / 3
def tiffany_thursday_miles := 1 / 3
def tiffany_friday_miles := 1 / 3

-- Total miles is the sum of miles for the week
def billy_total_miles := billy_sunday_miles + billy_monday_miles + billy_tuesday_miles +
                         billy_wednesday_miles + billy_thursday_miles + billy_friday_miles +
                         billy_saturday_miles

def tiffany_total_miles (T : ℝ) := T * 3 + 
                                   tiffany_wednesday_miles + tiffany_thursday_miles + tiffany_friday_miles

-- Proof problem: show that Tiffany runs 2 miles each day on Sunday, Monday, and Tuesday
theorem tiffany_daily_miles : ∃ T : ℝ, (tiffany_total_miles T = billy_total_miles) ∧ T = 2 :=
by
  sorry

end tiffany_daily_miles_l213_213569


namespace buratino_solved_16_problems_l213_213492

-- Defining the conditions given in the problem
def total_kopeks_received : ℕ := 655 * 100 + 35

def geometric_sum (n : ℕ) : ℕ := 2^n - 1

-- The goal is to prove that Buratino solved 16 problems
theorem buratino_solved_16_problems (n : ℕ) (h : geometric_sum n = total_kopeks_received) : n = 16 := by
  sorry

end buratino_solved_16_problems_l213_213492


namespace max_gcd_of_13n_plus_3_and_7n_plus_1_l213_213983

theorem max_gcd_of_13n_plus_3_and_7n_plus_1 (n : ℕ) (hn : 0 < n) :
  ∃ d, d = Nat.gcd (13 * n + 3) (7 * n + 1) ∧ ∀ m, m = Nat.gcd (13 * n + 3) (7 * n + 1) → m ≤ 8 := 
sorry

end max_gcd_of_13n_plus_3_and_7n_plus_1_l213_213983


namespace arithmetic_example_l213_213122

theorem arithmetic_example : 2 + 8 * 3 - 4 + 7 * 2 / 2 = 29 :=
by
  sorry

end arithmetic_example_l213_213122


namespace ts_parallel_ak_l213_213655

variable (A B C M K D S L T : Point)
variable (h1 : Circle A C)
variable (h2 : ∃ M, M ≠ A ∧ M ∈ Circle A C)
variable (h3 : ∃ K, K ≠ C ∧ K ∈ Circle A C)
variable (h4 : ∃ D, D ∈ Circle B K M ∧ D ∈ Line A K)
variable (h5 : ∃ S L, Line B D ∩ Line M K = {S} ∧ Line B D ∩ Line A C = {L})
variable (h6 : ∃ T, T ∈ Line A B ∧ ∠ A L T = ∠ C B L)

theorem ts_parallel_ak (h1 h2 h3 h4 h5 h6 : Prop) : Parallel (Line T S) (Line A K) := 
sorry

end ts_parallel_ak_l213_213655


namespace two_digit_inserts_zero_divisibility_count_valid_two_digit_numbers_l213_213069

theorem two_digit_inserts_zero_divisibility :
  ∃ n : ℕ, 10 ≤ n ∧ n < 100 ∧ ∀ a b : ℕ,
  1 ≤ a ∧ a ≤ 9 ∧ 1 ≤ b ∧ b ≤ 9 ∧ n = 10 * a + b → 
  ∃ c : ℕ, 2 ≤ c ∧ c < 10 ∧ 100 * a + b = c * n :=
by sorry

-- To count how many such two-digit numbers there are
theorem count_valid_two_digit_numbers :
  finset.card {n : ℕ | 10 ≤ n ∧ n < 100 ∧ 
    ∀ a b : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ 1 ≤ b ∧ b ≤ 9 ∧ n = 10 * a + b →
    ∃ c : ℕ, 2 ≤ c ∧ c < 10 ∧ 100 * a + b = c * n} = 3 :=
by sorry

end two_digit_inserts_zero_divisibility_count_valid_two_digit_numbers_l213_213069


namespace remaining_mean_correct_l213_213822

def remaining_mean_is_2005 {α : Type*} [linear_ordered_field α] 
  (numbers : set α) 
  (mean_five : α) : Prop :=
  ∃ (a b c d e f : α), 
    numbers = {a, b, c, d, e, f} ∧ 
    mean_five = (a + b + c + d + e) / 5 ∧ 
    (a + b + c + d + e + f = 11765) → 
    f = 2005

theorem remaining_mean_correct :
  remaining_mean_is_2005 {1742, 1865, 1907, 2003, 2091, 2157} 1952 :=
by
  sorry

end remaining_mean_correct_l213_213822


namespace ellipse_a_value_l213_213669

-- Define the conditions
def ellipse_equation (x y a : ℝ) : Prop := (x^2)/(a^2) + (y^2)/8 = 1
def focal_length (c : ℝ) : Prop := 2 * c = 4

-- Define the proof statement
theorem ellipse_a_value (a : ℝ) (h1 : focal_length 2) (h2 : ∀ x y, ellipse_equation x y a) : a = 2 * real.sqrt 3 :=
sorry

end ellipse_a_value_l213_213669


namespace range_of_a_l213_213859

variable (a : ℝ)

/-- Proposition p: The function f(x) = x^3 - ax - 1 is monotonically decreasing on the interval [-1, 1] -/
def prop_p := ∀ x ∈ Icc (-1 : ℝ) 1, deriv (λ x, x^3 - a * x - 1) x ≤ 0

/-- Proposition q: The range of the function y = ln(x^2 + ax + 1) is ℝ -/
def prop_q := ∀ y : ℝ, ∃ x : ℝ, ln (x^2 + a * x + 1) = y

/-- Function that returns whether either prop_p or prop_q is true, but not both -/
def exclusive_or : Prop := (prop_p a ∨ prop_q a) ∧ ¬ (prop_p a ∧ prop_q a)

/-- Prove the range of values for the real number a is (-∞, -2] ∪ [2, 3) -/
theorem range_of_a : (exclusive_or a) → (a ∈ Iic (-2) ∨ a ∈ Ico 2 3) :=
by
  sorry

end range_of_a_l213_213859


namespace fraction_calls_B_correct_l213_213516

-- Defining constants and the conditions.
constant C : ℝ  -- calls processed by each member of team B
constant N : ℝ  -- number of agents in team B

-- Condition: Each member of team A processes 6/5 times calls compared to one member of team B
constant calls_per_member_A : ℝ := (6 / 5) * C

-- Condition: Team A has 5/8 as many agents as team B
constant agents_team_A : ℝ := (5 / 8) * N

-- Total calls processed by team A
def total_calls_A : ℝ := calls_per_member_A * agents_team_A

-- Total calls processed by team B
def total_calls_B : ℝ := C * N

-- Total calls processed by both teams
def total_calls_both : ℝ := total_calls_A + total_calls_B

-- The fraction of the total calls processed by team B
def fraction_calls_B : ℝ := total_calls_B / total_calls_both

-- Main theorem: the fraction of total calls processed by team B is 4/7
theorem fraction_calls_B_correct : fraction_calls_B = 4 / 7 := 
  sorry

end fraction_calls_B_correct_l213_213516


namespace arithmetic_mean_of_a_and_c_value_of_a_l213_213369

-- Define the problem space and conditions
variables (a b c : ℝ)
variables (A B C : ℝ)

-- Given conditions for part 1
axiom cos_B_eq_neg5_13 : cos B = -5 / 13
axiom geometric_sequence : 2 * sin A = sin B * (sin C / sin B)
axiom area_eq_6_13: (1/2) * a * c * sin B = 6 / 13

-- Part 1: Proving arithmetic mean of a and c
theorem arithmetic_mean_of_a_and_c :
  (a + c) / 2 = sqrt 221 / 13 := sorry

-- Given conditions for part 2
axiom cos_C_eq_4_5 : cos C = 4 / 5
axiom dot_product_eq_14 : b * c * cos A = 14

-- Part 2: Proving value of a
theorem value_of_a :
  a = 11 / 4 := sorry

end arithmetic_mean_of_a_and_c_value_of_a_l213_213369


namespace year_price_diff_65_cents_l213_213881

variable (Y : ℝ) -- price of commodity Y in 2001

def price_X (n : ℕ) : ℝ := 4.20 + 0.45 * n -- price of X after n years
def price_Y (n : ℕ) : ℝ := Y + 0.20 * n -- price of Y after n years

theorem year_price_diff_65_cents :
  ∃ n : ℕ, price_X Y n = price_Y Y n + 0.65 :=
sorry

end year_price_diff_65_cents_l213_213881


namespace unique_three_digit_numbers_l213_213486

theorem unique_three_digit_numbers (d1 d2 d3 : ℕ) :
  (d1 = 3 ∧ d2 = 0 ∧ d3 = 8) →
  ∃ nums : Finset ℕ, 
  (∀ n ∈ nums, (∃ h t u : ℕ, n = 100 * h + 10 * t + u ∧ 
                h ≠ 0 ∧ (h = d1 ∨ h = d2 ∨ h = d3) ∧ 
                (t = d1 ∨ t = d2 ∨ t = d3) ∧ (u = d1 ∨ u = d2 ∨ u = d3) ∧ 
                h ≠ t ∧ t ≠ u ∧ u ≠ h)) ∧ nums.card = 4 :=
by
  sorry

end unique_three_digit_numbers_l213_213486


namespace correct_probability_reassemble_black_cube_l213_213043

noncomputable def probability_reassemble_black_cube : ℝ := (3^8 * nat.factorial 8 * 2^12 * nat.factorial 12 * 4^6 * nat.factorial 6 : ℝ) / (24^27 * nat.factorial 27 : ℝ)

theorem correct_probability_reassemble_black_cube :
  abs (probability_reassemble_black_cube - (1.8 * 10^(-37))) < 10^(-38) := sorry

end correct_probability_reassemble_black_cube_l213_213043


namespace convert_base_10_to_base_8_l213_213133

theorem convert_base_10_to_base_8 (n : ℕ) (h : n = 1632) : ∃ m : ℕ, m = 3140 ∧ (n₁ n = 8) (m₁ m = 8) :=
by
  sorry

end convert_base_10_to_base_8_l213_213133


namespace max_ants_collisions_l213_213401

theorem max_ants_collisions (n : ℕ) (hpos : 0 < n) :
  ∃ (ants : Fin n → ℝ) (speeds: Fin n → ℝ) (finite_collisions : Prop)
    (collisions_bound : ℕ),
  (∀ i : Fin n, speeds i ≠ 0) →
  finite_collisions →
  collisions_bound = (n * (n - 1)) / 2 :=
by
  sorry

end max_ants_collisions_l213_213401


namespace intersection_A_B_l213_213355

def A : Set ℝ := {y | ∃ x : ℝ, y = Real.cos x}
def B : Set ℝ := {x | x * (x + 1) ≥ 0}

theorem intersection_A_B :
  (A ∩ B) = {x | (0 ≤ x ∧ x ≤ 1) ∨ x = -1} :=
  sorry

end intersection_A_B_l213_213355


namespace problem_I_problem_II_l213_213703

open Real

noncomputable def f (a : ℝ) (x : ℝ) := sin x - a * x
noncomputable def h (x : ℝ) := x * log x - x - cos x

theorem problem_I {a : ℝ} (h1 : log 2 > sin (1 / 2)) (h2 : log (4 / π) < sqrt 2 / 2) : 
  (∀ x : ℝ, 0 < x ∧ x < 1 → f a x > 0) → 0 < a ∧ a ≤ sin 1 :=
sorry

theorem problem_II (h1 : log 2 > sin (1 / 2)) (h2 : log (4 / π) < sqrt 2 / 2) :
  ∃ x0 : ℝ, 1 / 2 < x0 ∧ x0 < π / 4 ∧ (∀ x : ℝ, h_deriv x0 x = 0) :=
sorry

end problem_I_problem_II_l213_213703


namespace ceil_square_neg_seven_over_four_l213_213215

theorem ceil_square_neg_seven_over_four : 
  let x := - (7 / 4 : ℚ) in
  ⌈x^2⌉ = 4 :=
by
  let x := - (7 / 4 : ℚ)
  sorry

end ceil_square_neg_seven_over_four_l213_213215


namespace central_angle_radian_measure_l213_213287

-- Define the radius and area of the sector as constants
def radius : ℝ := 2
def area : ℝ := 4

-- Prove the radian measure of the central angle
theorem central_angle_radian_measure : ∃ α : ℝ, α = 2 ∧ (2 * α * radius = 2 * area) :=
by
  -- Use the definition of area for a sector and given conditions
  use 2
  split
  · rfl
  sorry

end central_angle_radian_measure_l213_213287


namespace complex_magnitude_equality_l213_213610

open Complex Real

theorem complex_magnitude_equality :
  abs ((Complex.mk (5 * sqrt 2) (-5)) * (Complex.mk (2 * sqrt 3) 6)) = 60 :=
by
  sorry

end complex_magnitude_equality_l213_213610


namespace initial_momentum_eq_2Fx_div_v_l213_213549

variable (m v F x t : ℝ)
variable (H_initial_conditions : v ≠ 0)
variable (H_force : F > 0)
variable (H_distance : x > 0)
variable (H_time : t > 0)
variable (H_stopping_distance : x = (m * v^2) / (2 * F))
variable (H_stopping_time : t = (m * v) / F)

theorem initial_momentum_eq_2Fx_div_v :
  m * v = (2 * F * x) / v :=
sorry

end initial_momentum_eq_2Fx_div_v_l213_213549


namespace sum_proper_divisors_360_l213_213919

theorem sum_proper_divisors_360 : 
  let divisors_360 := {d ∈ finset.range 361 | 360 % d = 0}
  let proper_divisors_360 := divisors_360 \ {360}
  finset.sum proper_divisors_360 (λ x, x) = 810 :=
by
  let divisors_360 := {d ∈ finset.range 361 | 360 % d = 0}
  let proper_divisors_360 := divisors_360 \ {360}
  have h1 : ∑ d in proper_divisors_360, d = 810 := sorry
  exact h1

end sum_proper_divisors_360_l213_213919


namespace trapezoid_area_l213_213082

-- Define the problem statement
theorem trapezoid_area 
  (a b h: ℝ)
  (b₁ b₂: ℝ)
  (θ: ℝ) 
  (h₃: θ = Real.arcsin 0.6)
  (h₄: a = 20)
  (h₅: b = a - 2 * b₁ * Real.sin θ) 
  (h₆: h = b₁ * Real.cos θ) 
  (h₇: θ = Real.arcsin (3/5)) 
  (circum: isosceles_trapezoid_circumscribed a b₁ b₂) :
  ((1 / 2) * (a + b₂) * h = 2000 / 27) :=
by sorry

end trapezoid_area_l213_213082


namespace max_height_l213_213034

noncomputable def height (t : ℝ) : ℝ := -5 * t^2 + 20 * t + 10

theorem max_height : ∃ t : ℝ, height t = 30 :=
by
  use 2
  rw [height]
  linarith

end max_height_l213_213034


namespace given_equation_roots_sum_cubes_l213_213407

theorem given_equation_roots_sum_cubes (r s t : ℝ) 
    (h1 : 6 * r ^ 3 + 1506 * r + 3009 = 0)
    (h2 : 6 * s ^ 3 + 1506 * s + 3009 = 0)
    (h3 : 6 * t ^ 3 + 1506 * t + 3009 = 0)
    (sum_roots : r + s + t = 0) :
    (r + s) ^ 3 + (s + t) ^ 3 + (t + r) ^ 3 = 1504.5 := 
by 
  -- proof omitted
  sorry

end given_equation_roots_sum_cubes_l213_213407


namespace time_to_pass_jogger_l213_213052

noncomputable def speed_of_jogger_kmph : ℝ := 9
noncomputable def speed_of_train_kmph : ℝ := 45
noncomputable def initial_distance_m : ℝ := 240
noncomputable def length_of_train_m : ℝ := 110

theorem time_to_pass_jogger : 
  let relative_speed_mps := (speed_of_train_kmph - speed_of_jogger_kmph) * (1000 / 3600),
      total_distance_m := initial_distance_m + length_of_train_m,
      time_seconds := total_distance_m / relative_speed_mps
  in time_seconds = 35 :=
by
  sorry

end time_to_pass_jogger_l213_213052


namespace ellipse_intersection_l213_213555

open Real

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem ellipse_intersection (f1 f2 : ℝ × ℝ)
    (h1 : f1 = (0, 5))
    (h2 : f2 = (4, 0))
    (origin_intersection : distance (0, 0) f1 + distance (0, 0) f2 = 5) :
    ∃ x : ℝ, (distance (x, 0) f1 + distance (x, 0) f2 = 5 ∧ x > 0 ∧ x ≠ 0 → x = 28 / 9) :=
by 
  sorry

end ellipse_intersection_l213_213555


namespace ceil_square_neg_fraction_l213_213179

theorem ceil_square_neg_fraction :
  let x := (-7) / 4
  let y := x^2
  let z := Real.ceil y
  z = 4 := 
by
  sorry

end ceil_square_neg_fraction_l213_213179


namespace closest_perfect_square_l213_213008

theorem closest_perfect_square (n : ℕ) (h : n = 315) : ∃ k : ℕ, k^2 = 324 ∧ ∀ m : ℕ, m^2 ≠ 315 ∨ abs (n - m^2) > abs (n - k^2) :=
by
  use 18
  sorry

end closest_perfect_square_l213_213008


namespace find_width_of_cistern_l213_213945

noncomputable def width_of_cistern (length : ℝ) (water_depth : ℝ) (total_wet_surface_area : ℝ) : ℝ :=
  (total_wet_surface_area - 2 * length * water_depth - water_depth * length) / (length + water_depth)

theorem find_width_of_cistern : width_of_cistern 6 1.25 49 ≈ 4.69 :=
begin
  sorry
end

end find_width_of_cistern_l213_213945


namespace adeline_hours_per_day_l213_213551

theorem adeline_hours_per_day :
  ∀ (total_earnings weekly_days weeks daily_rate earned_in_7_weeks : ℝ), 
    weekly_days = 5 →
    weeks = 7 →
    daily_rate = 12 →
    earned_in_7_weeks = 3780 →
    total_earnings = earned_in_7_weeks / weeks →
    total_earnings / weekly_days / daily_rate = 9 :=
by 
  intros total_earnings weekly_days weeks daily_rate earned_in_7_weeks 
         h_days h_weeks h_rate h_earn h_total_earnings,
  sorry

end adeline_hours_per_day_l213_213551


namespace monster_ratio_l213_213527

theorem monster_ratio (r : ℝ) :
  (121 + 121 * r + 121 * r^2 = 847) → r = 2 :=
by
  intros h
  sorry

end monster_ratio_l213_213527


namespace probability_of_arithmetic_sequences_l213_213146

noncomputable def arithmetic_sequence_probability : ℚ :=
  have total_ways := (nat.choose 9 3) * (nat.choose 6 3) * (nat.choose 3 3) / (3!),
  have valid_groups := 5,
  (valid_groups : ℚ) / (total_ways : ℚ)

theorem probability_of_arithmetic_sequences : arithmetic_sequence_probability = 1 / 56 :=
  sorry

end probability_of_arithmetic_sequences_l213_213146


namespace smallest_n_for_pairwise_coprime_contains_prime_l213_213403
-- Import necessary libraries

-- Define the set S
def S := {1, 2, ... , 2005}

-- Define the problem statement in Lean
theorem smallest_n_for_pairwise_coprime_contains_prime :
  ∀ (A : finset ℕ), A ⊆ S ∧ A.card = 16 → (∃ x ∈ A, nat.prime x) :=
by
  sorry

end smallest_n_for_pairwise_coprime_contains_prime_l213_213403


namespace x_intercept_of_line_l213_213019

theorem x_intercept_of_line (x1 y1 x2 y2 : ℝ) (hx1 : x1 = 10) (hy1 : y1 = 3) (hx2 : x2 = -8) (hy2 : y2 = -6) :
  ∃ x0 : ℝ, (∀ y : ℝ, y = 0 → (∃ m : ℝ, y = m * (x0 - x1) + y1)) ∧ x0 = 4 :=
by
  sorry

end x_intercept_of_line_l213_213019


namespace trapezoid_area_l213_213085

-- Given definitions and conditions for the problem
def isosceles_trapezoid_circumscribed_around_circle (a b h : ℝ) : Prop :=
  a > b ∧ h > 0 ∧ ∀ (x y : ℝ), x = h / 0.6 ∧ y = (2 * x - h) / 8 → a = b + 2 * √((h^2 - ((a - b) / 2)^2))

-- Definitions derived from conditions
def longer_base := 20
def base_angle := Real.arcsin 0.6

-- The proposition we need to prove (area == 74)
theorem trapezoid_area : 
  ∀ (a b h : ℝ), isosceles_trapezoid_circumscribed_around_circle a b h → base_angle = Real.arcsin 0.6 → 
  a = 20 → (1 / 2) * (b + 20) * h = 74 :=
sorry

end trapezoid_area_l213_213085


namespace martha_savings_l213_213812

theorem martha_savings :
  let daily_allowance := 12
  let days_in_week := 7
  let save_half_daily := daily_allowance / 2
  let save_quarter_daily := daily_allowance / 4
  let days_saving_half := 6
  let day_saving_quarter := 1
  let total_savings := (days_saving_half * save_half_daily) + (day_saving_quarter * save_quarter_daily)
  in total_savings = 39 :=
by
  sorry

end martha_savings_l213_213812


namespace ceil_square_eq_four_l213_213159

theorem ceil_square_eq_four : (⌈(-7 / 4: ℚ)^2⌉ : ℤ) = 4 := by
  sorry

end ceil_square_eq_four_l213_213159


namespace twisted_fraction_eq_l213_213072

namespace AlinaFraction

noncomputable def twist_periodic_fraction (n d : ℕ) : ℚ :=
  let decimal := (n : ℚ) / (d : ℚ)
  let periodic_part := "2482678983833718244803695150118819566759000999000999000999"
  let twisted_periodic_part := "9248267898383371824480369515011881956675900099900099900099"
  let twisted_decimal := "0." ++ repeated_twisted_periodic_part
  let fraction := 
     /-- Converts the periodic decimal to a fraction
       We use the mathematical formula for converting repeating decimals to fractions.
     --/
      sorry
  fraction

theorem twisted_fraction_eq :
  twist_periodic_fraction 503 2022 =
    (9248267898383371824480369515011881956675900099900099900099 : ℚ) /
    ((10 : ℚ) ^ 336 - 1) :=
sorry

end AlinaFraction

end twisted_fraction_eq_l213_213072


namespace solve_system_of_equations_l213_213025

theorem solve_system_of_equations (m b : ℤ) 
  (h1 : 3 * m + b = 11)
  (h2 : -4 * m - b = 11) : 
  m = -22 ∧ b = 77 :=
  sorry

end solve_system_of_equations_l213_213025


namespace total_trophies_after_five_years_l213_213388

theorem total_trophies_after_five_years (michael_current_trophies : ℕ) (michael_increase : ℕ) (jack_multiplier : ℕ) (h1 : michael_current_trophies = 50) (h2 : michael_increase = 150) (h3 : jack_multiplier = 15) :
  let michael_five_years : ℕ := michael_current_trophies + michael_increase
  let jack_five_years : ℕ := jack_multiplier * michael_current_trophies
  michael_five_years + jack_five_years = 950 :=
by
  sorry

end total_trophies_after_five_years_l213_213388


namespace sum_primes_no_solution_congruence_l213_213589

theorem sum_primes_no_solution_congruence :
  ∑ p in {p | Nat.Prime p ∧ ¬ (∃ x : ℤ, 5 * (10 * x + 2) ≡ 3 [ZMOD p])}, p = 7 :=
sorry

end sum_primes_no_solution_congruence_l213_213589


namespace ceil_square_neg_seven_over_four_l213_213216

theorem ceil_square_neg_seven_over_four : 
  let x := - (7 / 4 : ℚ) in
  ⌈x^2⌉ = 4 :=
by
  let x := - (7 / 4 : ℚ)
  sorry

end ceil_square_neg_seven_over_four_l213_213216


namespace cylinder_surface_area_l213_213863

noncomputable def total_surface_area_cylinder (r h : ℝ) : ℝ :=
  let base_area := 64 * Real.pi
  let lateral_surface_area := 2 * Real.pi * r * h
  let total_surface_area := 2 * base_area + lateral_surface_area
  total_surface_area

theorem cylinder_surface_area (r h : ℝ) (hr : Real.pi * r^2 = 64 * Real.pi) (hh : h = 2 * r) : 
  total_surface_area_cylinder r h = 384 * Real.pi := by
  sorry

end cylinder_surface_area_l213_213863


namespace sum_minimum_values_l213_213414

def P (x : ℝ) (a b c : ℝ) : ℝ := x^3 + a * x^2 + b * x + c
def Q (x : ℝ) (d e f : ℝ) : ℝ := x^3 + d * x^2 + e * x + f

theorem sum_minimum_values (a b c d e f : ℝ)
  (hPQ : ∀ x, P (Q x d e f) a b c = 0 → x = -4 ∨ x = -2 ∨ x = 0 ∨ x = 2 ∨ x = 4)
  (hQP : ∀ x, Q (P x a b c) d e f = 0 → x = -3 ∨ x = -1 ∨ x = 1 ∨ x = 3) :
  P 0 a b c + Q 0 d e f = -20 := sorry

end sum_minimum_values_l213_213414


namespace find_a3_a4_a5_a6_a7_a8_l213_213660

-- Let {a_n} be a geometric sequence
variables {a : ℕ → ℚ}

-- Conditions given
axiom a_seq_geom (n : ℕ) : a (n + 1) = q * a n
axiom sum_a1_a2_a3 : a 1 + a 2 + a 3 = 4
axiom sum_a2_a3_a4 : a 2 + a 3 + a 4 = -2

-- The target question: Find the value of a3 + a4 + a5 + a6 + a7 + a8
theorem find_a3_a4_a5_a6_a7_a8 :
  a 3 + a 4 + a 5 + a 6 + a 7 + a 8 = 7 / 8 :=
sorry

end find_a3_a4_a5_a6_a7_a8_l213_213660


namespace distance_between_points_on_quadratic_curve_l213_213709

theorem distance_between_points_on_quadratic_curve
  (a c m k : ℝ) :
  let p1 := (a, m * a^2 + k) in
  let p2 := (c, m * c^2 + k) in
  dist p1 p2 = abs (a - c) * real.sqrt (1 + m^2 * (c + a)^2) :=
by
  let p1 := (a, m * a^2 + k)
  let p2 := (c, m * c^2 + k)
  sorry

end distance_between_points_on_quadratic_curve_l213_213709


namespace increasing_on_interval_l213_213734

def f (x : ℝ) (a : ℝ) : ℝ := 2 * Real.log x + 1 - a / x

theorem increasing_on_interval (a : ℝ) :
  (∀ x : ℝ, 1 < x → 0 ≤ deriv (λ x, f x a) x) ↔ a ≥ -2 :=
by
  sorry

end increasing_on_interval_l213_213734


namespace simplify_expression_l213_213241

theorem simplify_expression (x : ℝ) : 
  6 * (x - 7) * (2 * x + 15) + (3 * x - 4) * (x + 5) = 15 * x^2 + 17 * x - 650 :=
by
  sorry

end simplify_expression_l213_213241


namespace compute_expression_l213_213118

theorem compute_expression : 2 + 8 * 3 - 4 + 7 * 2 / 2 = 29 := by
  sorry

end compute_expression_l213_213118


namespace find_C_l213_213482

theorem find_C (C : ℤ) (h : 4 * C + 3 = 31) : C = 7 := by
  sorry

end find_C_l213_213482


namespace solve_system_of_equations_l213_213259

noncomputable def system_solution (n : ℕ) (x : ℕ → ℤ) : Prop :=
  ∀ (i : ℕ), i < n →
  (n^2 - n) * x i + (∏ j in finset.univ.erase i, x j) * (∑ k in finset.range n, x k ^ 2) = n^3 - n^2

theorem solve_system_of_equations (n : ℕ) (h : 2 ≤ n) (x : ℕ → ℤ) :
  system_solution n x →
  ∃ i : ℕ, i < n ∧ x i = 1 ∧ (∀ j : ℕ, j ≠ i → j < n → x j = n - 1) :=
sorry

end solve_system_of_equations_l213_213259


namespace minimum_moves_l213_213105

theorem minimum_moves (n : ℕ) : 
  let M := if odd n then (n^2 + 1)/2 else n^2/2 + n
  in ∀ (grid : ℕ → ℕ → 𝔹) (playable : ℕ → ℕ → 𝔹),
    (∀ i j, playable i j → 
      (∀ k, k ≠ i → ¬grid k j) ∧ 
      (∀ l, l ≠ j → ¬grid i l)) →
    (∀ k l, ¬playable k l → ∃ i j, playable i j ∧ grid i j) →
    M
:= by
  sorry

end minimum_moves_l213_213105


namespace find_n_l213_213616

theorem find_n (n : ℕ) : 2 ^ 6 * 3 ^ 3 * n = 10.factorial → n = 350 := 
by intros; sorry

end find_n_l213_213616


namespace cos_5_2x_l213_213277

theorem cos_5_2x (θ x : ℝ) (h1 : sin x = sin θ + cos θ) (h2 : cos x = sin θ * cos θ) : cos (2 * x)^5 = -1 := 
sorry

end cos_5_2x_l213_213277


namespace find_a_l213_213707

-- Definitions for conditions
def line_equation (a : ℝ) (x y : ℝ) := a * x - y - 1 = 0
def angle_of_inclination (θ : ℝ) := θ = Real.pi / 3

-- The main theorem statement
theorem find_a (a : ℝ) (θ : ℝ) (h1 : angle_of_inclination θ) (h2 : a = Real.tan θ) : a = Real.sqrt 3 :=
 by
   -- skipping the proof
   sorry

end find_a_l213_213707


namespace probability_five_digit_palindrome_divisible_by_11_l213_213046

def is_five_digit_palindrome (n : ℕ) : Prop :=
  let a := n / 10000
  let b := (n / 1000) % 10
  let c := (n / 100) % 10
  n % 100 = 100*a + 10*b + c

def divisible_by_11 (n : ℕ) : Prop :=
  n % 11 = 0

theorem probability_five_digit_palindrome_divisible_by_11 :
  let count_palindromes := 9 * 10 * 10
  let count_divisible_by_11 := 165
  (count_divisible_by_11 : ℚ) / count_palindromes = 11 / 60 :=
by
  sorry

end probability_five_digit_palindrome_divisible_by_11_l213_213046


namespace problem1_problem2_problem3_l213_213558

-- Problem 1
theorem problem1 (x : ℝ) (h : x = 10) : (2 / 5) * x = 4 :=
by sorry

-- Problem 2
theorem problem2 (m : ℝ) (h1 : m > 0) (h2 : (2 / 5) * m = (- (1 / 5) * m^2) + 2 * m) : m = 8 :=
by sorry

-- Problem 3
theorem problem3 (t : ℝ) (h1 : ∃ t, (2 / 5) * (32 - t) + (- (1 / 5) * t^2) + 2 * t = 16) : true :=
by sorry

end problem1_problem2_problem3_l213_213558


namespace football_team_progress_l213_213015

theorem football_team_progress (loss gain : ℤ) (h_loss : loss = -5) (h_gain : gain = 8) :
  (loss + gain = 3) :=
by
  sorry

end football_team_progress_l213_213015


namespace unique_ordered_triple_l213_213346

open Real

theorem unique_ordered_triple : 
  (∃! (a b c : ℤ), 
    2 ≤ a ∧ 
    1 ≤ b ∧ 
    0 ≤ c ∧ 
    log a.toReal b.toReal = (c ^ 2).toReal ∧ 
    a + b + c = 100) := 
sorry

end unique_ordered_triple_l213_213346


namespace sum_of_nine_terms_l213_213783

theorem sum_of_nine_terms (a : ℕ → ℝ) (d : ℝ) : 
  (∀ n, a (n + 1) = a n + d) →
  (a 4 + a 5 + a 6 = 21) →
  (9 / 2) * (a 1 + a 9) = 63 :=
by
  assume h_arithmetic_seq hn_condition,
  sorry

end sum_of_nine_terms_l213_213783


namespace minimize_total_cost_l213_213943

open Real

noncomputable def total_cost (x : ℝ) (h : 50 ≤ x ∧ x ≤ 100) : ℝ :=
  (130 / x) * 2 * (2 + (x^2 / 360)) + (14 * 130 / x)

theorem minimize_total_cost :
  ∀ (x : ℝ) (h : 50 ≤ x ∧ x ≤ 100),
  total_cost x h = (2340 / x) + (13 * x / 18)
  ∧ (x = 18 * sqrt 10 → total_cost x h = 26 * sqrt 10) :=
by
  sorry

end minimize_total_cost_l213_213943


namespace probability_at_least_two_same_l213_213830

theorem probability_at_least_two_same (n : ℕ) (H : n = 8) : 
  (∃ i j, (1 ≤ i ∧ i ≤ n) ∧ (1 ≤ j ∧ j ≤ n) ∧ i ≠ j ∧ ∀ (x : ℕ), x ∈ {i, j}) :=
by
  sorry

end probability_at_least_two_same_l213_213830


namespace parametric_equation_solution_l213_213448

noncomputable def solve_parametric_equation (a b : ℝ) (ha2b : a ≠ 2 * b) (ha3b : a ≠ -3 * b) : ℝ :=
  (5 / (a - 2 * b))

theorem parametric_equation_solution (a b x : ℝ) (ha2b : a ≠ 2 * b) (ha3b : a ≠ -3 * b) 
  (h : (a * x - 3) / (b * x + 1) = 2) : 
  x = solve_parametric_equation a b ha2b ha3b :=
sorry

end parametric_equation_solution_l213_213448


namespace martha_black_butterflies_l213_213807

theorem martha_black_butterflies (total_butterflies blue_butterflies yellow_butterflies black_butterflies : ℕ) 
    (h1 : total_butterflies = 19)
    (h2 : blue_butterflies = 2 * yellow_butterflies)
    (h3 : blue_butterflies = 6) :
    black_butterflies = 10 :=
by
  -- Prove the theorem assuming the conditions are met
  sorry

end martha_black_butterflies_l213_213807


namespace factorize_polynomial_l213_213242

theorem factorize_polynomial (x y : ℝ) :
  3 * x ^ 2 + 6 * x * y + 3 * y ^ 2 = 3 * (x + y) ^ 2 :=
by
  sorry

end factorize_polynomial_l213_213242


namespace batsman_average_after_10th_inning_l213_213941

theorem batsman_average_after_10th_inning 
  (A : ℕ) 
  (h1 : ∑ (i : ℕ) in (range 9), (A : ℕ) = 9 * A)
  (h2 : 60)
  (h3 : (9 * A + 60) / 10 = A + 3) :
  (9 * A + 60) / 10 = 33 :=
by 
  sorry

end batsman_average_after_10th_inning_l213_213941


namespace supermarket_sales_l213_213544

theorem supermarket_sales (S_Dec : ℝ) (S_Jan : ℝ) (S_Feb : ℝ) (S_Jan_eq : S_Jan = S_Dec * (1 + x))
  (S_Feb_eq : S_Feb = S_Jan * (1 + x))
  (inc_eq : S_Feb = S_Dec + 0.24 * S_Dec) :
  x = 0.2 ∧ S_Feb = S_Dec * (1 + 0.2)^2 := by
sorry

end supermarket_sales_l213_213544


namespace sum_of_special_primes_l213_213593

theorem sum_of_special_primes : 
  let primes_with_no_solution := {p : ℕ | prime p ∧ ¬ ∃ x : ℤ, 5 * (10 * x + 2) ≡ 3 [MOD p]} in
  ∃ p1 p2, p1 ∈ primes_with_no_solution ∧ p2 ∈ primes_with_no_solution ∧ p1 ≠ p2 ∧ p1 + p2 = 7 :=
by
  sorry

end sum_of_special_primes_l213_213593


namespace imaginary_part_of_z_l213_213281

theorem imaginary_part_of_z (z : ℂ) (hz : (1 + complex.i) * z = 1 - complex.i) : z.im = -1 := 
sorry

end imaginary_part_of_z_l213_213281


namespace x_is_integer_l213_213511

theorem x_is_integer
  (x : ℝ)
  (h_pos : 0 < x)
  (h1 : ∃ k1 : ℤ, x^2012 = x^2001 + k1)
  (h2 : ∃ k2 : ℤ, x^2001 = x^1990 + k2) : 
  ∃ n : ℤ, x = n :=
sorry

end x_is_integer_l213_213511


namespace cone_volume_is_3_6_l213_213952

-- Define the given conditions
def is_maximum_volume_cone_with_cutoff (cone_volume cutoff_volume : ℝ) : Prop :=
  cutoff_volume = 2 * cone_volume

def volume_difference (cutoff_volume cone_volume difference : ℝ) : Prop :=
  cutoff_volume - cone_volume = difference

-- The theorem to prove the volume of the cone
theorem cone_volume_is_3_6 
  (cone_volume cutoff_volume difference: ℝ)  
  (h1: is_maximum_volume_cone_with_cutoff cone_volume cutoff_volume)
  (h2: volume_difference cutoff_volume cone_volume 3.6) 
  : cone_volume = 3.6 :=
sorry

end cone_volume_is_3_6_l213_213952


namespace remainder_when_A_divided_by_9_l213_213262

theorem remainder_when_A_divided_by_9 (A B : ℕ) (h1 : A = B * 9 + 13) : A % 9 = 4 := 
by {
  sorry
}

end remainder_when_A_divided_by_9_l213_213262


namespace martha_black_butterflies_l213_213806

theorem martha_black_butterflies (total_butterflies blue_butterflies yellow_butterflies black_butterflies : ℕ) 
    (h1 : total_butterflies = 19)
    (h2 : blue_butterflies = 2 * yellow_butterflies)
    (h3 : blue_butterflies = 6) :
    black_butterflies = 10 :=
by
  -- Prove the theorem assuming the conditions are met
  sorry

end martha_black_butterflies_l213_213806


namespace arithmetic_sequence_ratio_l213_213637

theorem arithmetic_sequence_ratio (a_n : ℕ → ℝ) (S : ℕ → ℝ) (h_arith : ∀ n, a_n = a_n 0 + n * a_n 1) :
  (S 6 / S 3 = 4) → (S 5 / S 6 = 25 / 36) :=
by
  sorry

end arithmetic_sequence_ratio_l213_213637


namespace find_b_c_and_sin_B_minus_C_l213_213763

theorem find_b_c_and_sin_B_minus_C
  (a b c : ℝ) 
  (B C : ℝ) 
  (cos_B : ℝ)
  (cos_B_neg_half : cos_B = -1/2)
  (a_eq : a = 3)
  (b_minus_c_eq : b - c = 2)
  (cos_B_eq : ∀ B, cos B = cos_B)
  : b = 7 ∧ c = 5 ∧ sin (B - C) = 4 * real.sqrt 3 / 7 := 
by
  sorry

end find_b_c_and_sin_B_minus_C_l213_213763


namespace monotonicity_of_f_distinct_x_sum_lt_e_l213_213699

section part1

variables {R : Type*} [linear_ordered_field R]

def f (a : R) (x : R) : R := x * (1 - a * log x)

theorem monotonicity_of_f {a : R} (h : a ≠ 0) :
  (∀ x : R, 0 < x → x < exp ((1 - a) / a) → deriv (f a) x > 0) ∧
  (∀ x : R, x > exp ((1 - a) / a) → deriv (f a) x < 0) ∧
  (∀ x : R, x < exp ((1 - a) / a) → deriv (f a) x < 0) ∧
  (∀ x : R, exp ((1 - a) / a) < x → deriv (f a) x > 0) :=
sorry

end part1

section part2

variables {R : Type*} [linear_ordered_field R]

def f (x : R) := x * (1 - log x)

theorem distinct_x_sum_lt_e (x1 x2 : R) (hx1 : 0 < x1) (hx2 : 0 < x2) (hx : x1 ≠ x2) 
  (h_eq : f x1 = f x2) : x1 + x2 < exp 1 :=
sorry

end part2

end monotonicity_of_f_distinct_x_sum_lt_e_l213_213699


namespace julia_cookies_l213_213011

theorem julia_cookies (N : ℕ) 
  (h1 : N % 6 = 5) 
  (h2 : N % 8 = 7) 
  (h3 : N < 100) : 
  N = 17 ∨ N = 41 ∨ N = 65 ∨ N = 89 → 17 + 41 + 65 + 89 = 212 :=
sorry

end julia_cookies_l213_213011


namespace average_temperature_assumption_l213_213458

noncomputable def average_temperature_at_noontime 
  (temps : Fin 5 → ℝ) : ℝ :=
  (Array.sum temps.toList) / 5

def lowest_temperature (temps : Fin 5 → ℝ) : ℝ :=
  Array.foldl min temps[0] temps.toList

def temperature_range (temps : Fin 5 → ℝ) : ℝ :=
  (Array.foldl max temps[0] temps.toList - lowest_temperature temps)

-- Statement of the proof problem
theorem average_temperature_assumption 
  (temps : Fin 5 → ℝ) 
  (h1 : lowest_temperature temps = 42)
  (h2 : temperature_range temps = 15) :
  average_temperature_at_noontime temps = 49.5 :=
sorry

end average_temperature_assumption_l213_213458


namespace constant_term_in_expansion_l213_213327

noncomputable def integral_solution : ℝ :=
  ∫ x in 1..Real.exp 1, 6 / x

theorem constant_term_in_expansion :
  integral_solution = 6 →
  let n := integral_solution in
  (x^2 - x⁻¹)^n = ∑ r in Finset.range n.succ, (C n r * (- 1)^r * x^(n - 3*r)) →
  C 6 2 = 15 :=
begin
  sorry
end

end constant_term_in_expansion_l213_213327


namespace equal_triangles_by_perpendiculars_l213_213798

-- Given definitions and conditions
variable {Point : Type*} [EuclideanGeometry Point]
variable (A B C : Point) -- Vertices of the triangle
variable (A' B' C' : Point) -- Intersection points of the perpendiculars from each vertex to the opposite side
variable (hA : isPerpendicular (line_through A A') (side BC))
variable (hB : isPerpendicular (line_through B B') (side AC))
variable (hC : isPerpendicular (line_through C C') (side AB))

-- Proof target
theorem equal_triangles_by_perpendiculars :
  congruent (triangle A B' C') (triangle B C' A') :=
sorry

end equal_triangles_by_perpendiculars_l213_213798


namespace slope_of_line_through_points_l213_213918

theorem slope_of_line_through_points :
  let x1 := 1
  let y1 := 3
  let x2 := 5
  let y2 := 7
  let m := (y2 - y1) / (x2 - x1)
  m = 1 := by
  sorry

end slope_of_line_through_points_l213_213918


namespace permutation_modulo_1000_l213_213817

/-- 
  Prove that the number of valid permutations of the string "AAAABBBBBCCCCCDD"
  such that:
  - None of the first five letters is an 'A'.
  - None of the next five letters is a 'B'.
  - None of the last six letters is a 'C'.
  is congruent to 625 modulo 1000.
-/
theorem permutation_modulo_1000 : 
  let N := 74625 in
  N % 1000 = 625 :=
by
  rfl

end permutation_modulo_1000_l213_213817


namespace circle_rational_points_l213_213433

theorem circle_rational_points :
  ( ∃ B : ℚ × ℚ, ∀ k : ℚ, B ∈ {p | p.1 ^ 2 + 2 * p.1 + p.2 ^ 2 = 1992} ) ∧ 
  ( (42 : ℤ)^2 + 2 * 42 + 12^2 = 1992 ) :=
by
  sorry

end circle_rational_points_l213_213433


namespace dog_total_bones_l213_213418

-- Define the number of original bones and dug up bones as constants
def original_bones : ℕ := 493
def dug_up_bones : ℕ := 367

-- Define the total bones the dog has now
def total_bones : ℕ := original_bones + dug_up_bones

-- State and prove the theorem
theorem dog_total_bones : total_bones = 860 := by
  -- placeholder for the proof
  sorry

end dog_total_bones_l213_213418


namespace smallest_value_of_diff_l213_213387

-- Definitions of the side lengths from the conditions
def XY (x : ℝ) := x + 6
def XZ (x : ℝ) := 4 * x - 1
def YZ (x : ℝ) := x + 10

-- Conditions derived from the problem
noncomputable def valid_x (x : ℝ) := x > 5 / 3 ∧ x < 11 / 3

-- The proof statement
theorem smallest_value_of_diff : 
  ∀ (x : ℝ), valid_x x → (YZ x - XY x) = 4 :=
by
  intros x hx
  -- Proof goes here
  sorry

end smallest_value_of_diff_l213_213387


namespace equation_equivalence_product_uvw_l213_213871

variable {a x y b : ℕ}

theorem equation_equivalence (u v w : ℕ) (h1 : u = 4) (h2 : v = 2) (h3 : w = 6) :
  a ^ 10 * x * y - a ^ 9 * y - a ^ 8 * x = a ^ 6 * (b ^ 5 - 1) ↔
  (a ^ u * x - a ^ v) * (a ^ w * y - a ^ 3) = a ^ 6 * b ^ 5 :=
by
  sorry

theorem product_uvw (u v w : ℕ) (h1 : u = 4) (h2 : v = 2) (h3 : w = 6) :
  u * v * w = 48 :=
by
  calc
    u * v * w = 4 * 2 * 6 : by rw [h1, h2, h3]
          ... = 48        : by norm_num

end equation_equivalence_product_uvw_l213_213871


namespace impossibility_of_quadratic_conditions_l213_213604

open Real

theorem impossibility_of_quadratic_conditions :
  ∀ (a b c t : ℝ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a ≠ t ∧ b ≠ t ∧ c ≠ t →
  (b * t) ^ 2 - 4 * a * c > 0 →
  c ^ 2 - 4 * b * a > 0 →
  (a * t) ^ 2 - 4 * b * c > 0 →
  false :=
by sorry

end impossibility_of_quadratic_conditions_l213_213604


namespace mars_mission_cost_per_person_l213_213858

theorem mars_mission_cost_per_person
  (total_cost : ℕ) (number_of_people : ℕ)
  (h1 : total_cost = 50000000000) (h2 : number_of_people = 500000000) :
  (total_cost / number_of_people) = 100 := 
by
  sorry

end mars_mission_cost_per_person_l213_213858


namespace solution_set_of_f_l213_213310

-- Define f(x) to reflect the given conditions
def f (x : ℝ) : ℝ :=
  if 0 ≤ x then real.log (x + 1) / real.log 2 else -real.log (1 - x) / real.log 2

-- Prove that the solution set of |f(x)| ≤ 2 is [-3, 3]
theorem solution_set_of_f :
  {x : ℝ | abs (f x) ≤ 2} = set.Icc (-3 : ℝ) 3 :=
by
  sorry

end solution_set_of_f_l213_213310


namespace ceil_square_of_neg_fraction_l213_213188

theorem ceil_square_of_neg_fraction : 
  (Int.ceil ((-7 / 4 : ℚ)^2 : ℚ)).toNat = 4 := by
  sorry

end ceil_square_of_neg_fraction_l213_213188


namespace cos_alpha_minus_beta_l213_213274

theorem cos_alpha_minus_beta (α β : ℝ) (hα : 0 < α ∧ α < π / 2)
  (hβ : 0 < β ∧ β < π / 2) (h_cos_add : Real.cos (α + β) = -5 / 13)
  (h_tan_sum : Real.tan α + Real.tan β = 3) :
  Real.cos (α - β) = 1 :=
by
  sorry

end cos_alpha_minus_beta_l213_213274


namespace lecture_hall_rows_l213_213462

-- We define the total number of seats
def total_seats (n : ℕ) : ℕ := n * (n + 11)

-- We state the problem with the given conditions
theorem lecture_hall_rows : 
  (400 ≤ total_seats n) ∧ (total_seats n ≤ 440) → n = 16 :=
by
  sorry

end lecture_hall_rows_l213_213462


namespace find_k_no_solution_l213_213252

-- Conditions
def vector1 : ℝ × ℝ := (1, 3)
def direction1 : ℝ × ℝ := (5, -8)
def vector2 : ℝ × ℝ := (0, -1)
def direction2 (k : ℝ) : ℝ × ℝ := (-2, k)

-- Statement
theorem find_k_no_solution (k : ℝ) : 
  (∀ t s : ℝ, vector1 + t • direction1 ≠ vector2 + s • direction2 k) ↔ k = 16 / 5 :=
sorry

end find_k_no_solution_l213_213252


namespace probability_same_number_l213_213835

theorem probability_same_number (n k : ℕ) (h₁ : n = 8) (h₂ : k = 6) : 
  (∃ m : ℝ, 0 ≤ m ∧ m ≤ 1 ∧ m = 1) := by
  sorry

end probability_same_number_l213_213835


namespace problem1_problem2_problem3_l213_213559

-- Problem 1
theorem problem1 (x : ℝ) (h : x = 10) : (2 / 5) * x = 4 :=
by sorry

-- Problem 2
theorem problem2 (m : ℝ) (h1 : m > 0) (h2 : (2 / 5) * m = (- (1 / 5) * m^2) + 2 * m) : m = 8 :=
by sorry

-- Problem 3
theorem problem3 (t : ℝ) (h1 : ∃ t, (2 / 5) * (32 - t) + (- (1 / 5) * t^2) + 2 * t = 16) : true :=
by sorry

end problem1_problem2_problem3_l213_213559


namespace carlos_finishes_first_l213_213980

-- Define the areas of the lawns and the mowing rates
def andy_lawn_area (x : ℝ) : ℝ := 3 * x
def beth_lawn_area (x : ℝ) : ℝ := x
def carlos_lawn_area (x : ℝ) : ℝ := (3 * x) / 4

def andy_mowing_rate (y : ℝ) : ℝ := y
def carlos_mowing_rate (y : ℝ) : ℝ := y / 6
def beth_mowing_rate (y : ℝ) : ℝ := y / 4

-- Define the mowing times based on area and rate
def andy_mowing_time (x y : ℝ) : ℝ := (andy_lawn_area x) / (andy_mowing_rate y)
def beth_mowing_time (x y : ℝ) : ℝ := (beth_lawn_area x) / (beth_mowing_rate y)
def carlos_mowing_time (x y : ℝ) : ℝ := (carlos_lawn_area x) / (carlos_mowing_rate y)

-- Proof goal
theorem carlos_finishes_first (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) :
  carlos_mowing_time x y < andy_mowing_time x y ∧ 
  carlos_mowing_time x y < beth_mowing_time x y :=
by {
  -- Calculate times
  have t_andy := andy_mowing_time x y,
  have t_beth := beth_mowing_time x y,
  have t_carlos := carlos_mowing_time x y,
  -- Prove that carlos finishes first
  sorry
}

end carlos_finishes_first_l213_213980


namespace sum_of_digits_of_511_l213_213491

noncomputable def sum_of_digits_base_2 (n : ℕ) : ℕ :=
  (nat.digits 2 n).sum

theorem sum_of_digits_of_511 : sum_of_digits_base_2 511 = 9 :=
by 
  sorry

end sum_of_digits_of_511_l213_213491


namespace decreasing_interval_implies_positive_a_l213_213357

theorem decreasing_interval_implies_positive_a 
  (a : ℝ)
  (h : ∀ x ∈ (Ioo (- (sqrt 3 / 3)) (sqrt 3 / 3)), deriv (fun y => a * (y^3 - y)) x < 0 ) : a > 0 := 
sorry

end decreasing_interval_implies_positive_a_l213_213357


namespace chocolate_unclaimed_is_zero_l213_213071

theorem chocolate_unclaimed_is_zero (x : ℝ) : 
  let al_share := (2 / 5) * x
      remaining_after_al := x - al_share
      bert_share := (3 / 10) * x
      remaining_after_bert := remaining_after_al - bert_share
      carl_share := (1 / 5) * x
      remaining_after_carl := remaining_after_bert - carl_share
      dana_share := (1 / 10) * x
      remaining_after_dana := remaining_after_carl - dana_share
  in remaining_after_dana = 0 := 
by 
  sorry

end chocolate_unclaimed_is_zero_l213_213071


namespace find_first_number_l213_213457

theorem find_first_number (x : ℝ) : (10 + 70 + 28) / 3 = 36 →
  (x + 40 + 60) / 3 = 40 →
  x = 20 := 
by
  intros h_avg_old h_avg_new
  sorry

end find_first_number_l213_213457


namespace probability_divisible_by_5_expectation_distribution_l213_213731

section problem1

-- Define three-digit increasing number
def is_three_digit_increasing (n : ℕ) : Prop :=
  let units := n % 10
  let tens := (n / 10) % 10
  let hundreds := n / 100
  units > tens ∧ tens > hundreds

-- Condition: Choose any three numbers from {1, 2, 3, 4, 5}
def forming_three_digit_numbers (nums : List ℕ) : List ℕ :=
  nums.permutations.map (λ l, 100 * l.head! + 10 * l.tail!.head! + l.tail!.tail!.head!)

-- Condition: Find numbers divisible by 5
def divisible_by_5 (n : ℕ) : Prop := n % 5 = 0

-- Theorem: Probability of the number being divisible by 5 is 3/5
theorem probability_divisible_by_5 :
  let numbers := forming_three_digit_numbers [1, 2, 3, 4, 5]
  let increasing_numbers := numbers.filter is_three_digit_increasing
  let valid_numbers := increasing_numbers.filter divisible_by_5
  valid_numbers.length / increasing_numbers.length = 3 / 5 := sorry

end problem1

section problem2

-- Define three-digit increasing number based on different set
def forming_three_digit_numbers_full_set (nums : List ℕ) : List ℕ :=
  nums.permutations.map (λ l, 100 * l.head! + 10 * l.tail!.head! + l.tail!.tail!.head!)

-- Redefine the necessary properties
def divisible_by_3 (n : ℕ) : Prop := n % 3 = 0
def divisible_by_15 (n : ℕ) : Prop := n % 15 = 0

-- Scoring rules
def score (n : ℕ) : ℕ :=
  if divisible_by_15 n then 2 else if divisible_by_3 n ∨ divisible_by_5 n then 1 else 0

-- Theorem: Distribution table and mathematical expectation
theorem expectation_distribution :
  let numbers := forming_three_digit_numbers_full_set (List.range 1 10)
  let increasing_numbers := numbers.filter is_three_digit_increasing
  let scores := increasing_numbers.map score
  let count_0 := scores.count (λ s, s = 0)
  let count_1 := scores.count (λ s, s = 1)
  let count_2 := scores.count (λ s, s = 2)
  count_0 / increasing_numbers.length = 5 / 42 ∧
  count_1 / increasing_numbers.length = 2 / 3 ∧
  count_2 / increasing_numbers.length = 3 / 14 ∧
  (count_0 * 0 + count_1 * 1 + count_2 * 2) / increasing_numbers.length = 23 / 21 := sorry

end problem2

end probability_divisible_by_5_expectation_distribution_l213_213731


namespace ceiling_of_square_of_neg_7_over_4_is_4_l213_213170

theorem ceiling_of_square_of_neg_7_over_4_is_4 : 
  Real.ceil ((-7 / 4 : Real) ^ 2) = 4 := by
  sorry

end ceiling_of_square_of_neg_7_over_4_is_4_l213_213170


namespace ceiling_of_square_frac_l213_213219

theorem ceiling_of_square_frac : 
  let x : ℚ := -7 / 4
  let y : ℚ := x^2
  in Int.ceil y = 4 := 
by 
  let x : ℚ := -7 / 4
  let y : ℚ := x^2
  exact sorry

end ceiling_of_square_frac_l213_213219


namespace centroid_distance_sum_l213_213778

noncomputable def sum_of_distances_to_sides (A B C G : Point) (AB BC CA : ℝ) : ℝ :=
  if AB = 13 ∧ BC = 14 ∧ CA = 15 ∧ G = centroid A B C then
    ∑ d in (distance_to_side G ABC), d
  else 
    0

theorem centroid_distance_sum (A B C G : Point) :
  G = centroid A B C → 
    AB = 13 → 
    BC = 14 → 
    CA = 15 → 
    sum_of_distances_to_sides A B C G AB BC CA = 2348 / 195 :=
sorry

end centroid_distance_sum_l213_213778


namespace area_of_one_trapezoid_l213_213063

theorem area_of_one_trapezoid (
  (large_square_area : ℝ) (h1 : large_square_area = 400) 
  (smaller_square_vertices_midpoints : Prop) 
  (h2 : smaller_square_vertices_midpoints) 
  (trapezoids_congruent : Prop) 
  (h3 : trapezoids_congruent)
  ) : 
  ∃ (trapezoid_area : ℝ), trapezoid_area = 50 := 
by 
  sorry

end area_of_one_trapezoid_l213_213063


namespace equivalent_sets_A_equivalent_sets_B_equivalent_sets_C_not_equivalent_sets_D_solution_equivalent_sets_l213_213925

def Set_A1 : Set Int := {x | ∃ n: Int, x = 2 * n}
def Set_A2 : Set Int := {x | ∃ n: Int, x = 2 * (n + 1)}

def Set_B1 : Set Real := {y | ∃ x: Real, y = x^2 + 1}
def Set_B2 : Set Real := {x | ∃ t: Real, x = t^2 + 1}

def Set_C1 : Set Nat := {x | x ∈ Nat ∧ ∃ k: Nat, 1 ≤ k ∧ k ≤ 4 ∧ x = 2 * k}
def Set_C2 : Set Nat := {x | ∃ n: Nat, 1 ≤ n ∧ n ≤ 4 ∧ 2 * n = x }

def Set_D1 : Set Real := {y | ∃ x: Real, y = x^2 - 1}
def Set_D2 : Set (Real × Real) := {(x, y) | ∃ x: Real, y = x^2 - 1}

theorem equivalent_sets_A : Set_A1 = Set_A2 := sorry
theorem equivalent_sets_B : Set_B1 = Set_B2 := sorry
theorem equivalent_sets_C : Set_C1 = Set_C2 := sorry
theorem not_equivalent_sets_D : Set_D1 ≠ Set_D2 := sorry

theorem solution_equivalent_sets : equivalent_sets_A ∧ equivalent_sets_B ∧ equivalent_sets_C :=
  by exact ⟨equivalent_sets_A, equivalent_sets_B, equivalent_sets_C⟩

end equivalent_sets_A_equivalent_sets_B_equivalent_sets_C_not_equivalent_sets_D_solution_equivalent_sets_l213_213925


namespace max_min_of_f_tangent_lines_at_point_area_of_closed_shape_l213_213318

noncomputable def f (x : ℝ) : ℝ := (1/3) * x^3 - (1/2) * x^2 + 1

theorem max_min_of_f :
  f 0 = 1 ∧ f 1 = 5/6 :=
by
  sorry

theorem tangent_lines_at_point:
  (∀ x, f x = 1 → (x = 0 ∨ x = 3/2) → 
  (∀ x = 0, y = 1) ∧ 
  (∀ x = 3/2, y = (3/4) * x - 1/8)) :=
by
  sorry

theorem area_of_closed_shape :
  ∫ x in 0..3/2, 1 - f x = 9/64 :=
by
  sorry

end max_min_of_f_tangent_lines_at_point_area_of_closed_shape_l213_213318


namespace det_power_matrix_l213_213721

theorem det_power_matrix (M : Matrix ℕ ℕ ℝ) (h_det: det M = 3) : det (M^3) = 27 :=
  sorry

end det_power_matrix_l213_213721


namespace profession_odd_one_out_l213_213564

/-- 
Three professions are provided: 
1. Dentist
2. Elementary school teacher
3. Programmer 

Under modern Russian pension legislation, some professions have special retirement benefits. 
For the given professions, the elementary school teacher and dentist typically have special considerations, whereas the programmer does not.

Prove that the programmer is the odd one out under these conditions.
-/
theorem profession_odd_one_out
  (dentist_special: Bool)
  (teacher_special: Bool)
  (programmer_special: Bool)
  (h1: dentist_special = true)
  (h2: teacher_special = true)
  (h3: programmer_special = false) :
  ∃ profession, profession = "Programmer" :=
by
  use "Programmer"
  sorry

end profession_odd_one_out_l213_213564


namespace ceil_square_neg_seven_over_four_l213_213208

theorem ceil_square_neg_seven_over_four : 
  let x := - (7 / 4 : ℚ) in
  ⌈x^2⌉ = 4 :=
by
  let x := - (7 / 4 : ℚ)
  sorry

end ceil_square_neg_seven_over_four_l213_213208


namespace largest_common_value_l213_213864

/-- The largest value less than 300 that appears in both sequences 
    {7, 14, 21, 28, ...} and {5, 15, 25, 35, ...} -/
theorem largest_common_value (a : ℕ) (n m k : ℕ) :
  (a = 7 * (1 + n)) ∧ (a = 5 + 10 * m) ∧ (a < 300) ∧ (∀ k, (55 + 70 * k < 300) → (55 + 70 * k) ≤ a) 
  → a = 265 :=
by
  sorry

end largest_common_value_l213_213864


namespace find_n_l213_213617

theorem find_n (n : ℕ) : 2 ^ 6 * 3 ^ 3 * n = 10.factorial → n = 350 := 
by intros; sorry

end find_n_l213_213617


namespace string_length_l213_213520

theorem string_length (C n H : ℝ) (hC : C = 6) (hn : n = 3) (hH : H = 18) : 
  n * (real.sqrt (C^2 + (H/n)^2)) = 18 * real.sqrt 2 :=
by
  -- Sorry, proof not provided as per instructions
  sorry

end string_length_l213_213520


namespace ceil_square_eq_four_l213_213156

theorem ceil_square_eq_four : (⌈(-7 / 4: ℚ)^2⌉ : ℤ) = 4 := by
  sorry

end ceil_square_eq_four_l213_213156


namespace grade_eight_students_unique_l213_213906

noncomputable def num_grade_eight_students (n k : ℕ) : Prop :=
  let total_students := n + 2 in
  let total_games := total_students * (total_students - 1) / 2 in
  let total_points := total_games in
  let points_grade_eight := total_points - 8 in
  k * n = points_grade_eight ∧ 
  ∃ k1 k2, 2 * k1 = (k1 - n)^2 + 56 ∧ 2 * k2 = (k2 - n)^2 + 56

theorem grade_eight_students_unique : ∃ n, (num_grade_eight_students n 4) ∨ (num_grade_eight_students n 8) :=
begin
  sorry
end

end grade_eight_students_unique_l213_213906


namespace ellipse_eq_proof_l213_213687

open Real

def circle_radius (x y : ℝ) : ℝ :=
  sqrt ((x - 1)^2 + y^2)

def ellipse_eccentricity : ℝ := 1 / 2

def major_axis_radius : ℝ := 4

def a (major_axis_radius : ℝ) : ℝ := major_axis_radius / 2

def c (a e : ℝ) : ℝ := a * e

def b_sq (a c : ℝ) : ℝ := a^2 - c^2

def ellipse_standard_eq (a b_sq: ℝ) (x y : ℝ) : Prop :=
  (x^2 / a^2) + (y^2 / b_sq) = 1

theorem ellipse_eq_proof :
  (circle_radius 1 0 = major_axis_radius) →
  (e = 1 / 2) →
  (a major_axis_radius = 2) →
  (c 2 1/2 = 1) → 
  (b_sq 2 1 = 3) →
  ellipse_standard_eq 2 3 = λ (x y: ℝ), (x^2 / 4) + (y^2 / 3) = 1 :=
by {
  intros,
  sorry
}

end ellipse_eq_proof_l213_213687


namespace count_three_digit_perfect_square_numbers_with_perfect_square_digit_sum_l213_213971

theorem count_three_digit_perfect_square_numbers_with_perfect_square_digit_sum : 
  {n : ℕ // n >= 100 ∧ n < 1000 ∧ ∃ k : ℕ, n = k^2 ∧ ∃ S : ℕ, S = (n / 100) + (n / 10 % 10) + (n % 10) ∧ ∃ m : ℕ, S = m^2}.toFinset.card = 13 := 
by 
  sorry

end count_three_digit_perfect_square_numbers_with_perfect_square_digit_sum_l213_213971


namespace ceil_of_neg_frac_squared_l213_213205

-- Define the negated fraction
def neg_frac : ℚ := -7 / 4

-- Define the squared value of the negated fraction
def squared_value : ℚ := neg_frac ^ 2

-- Define the ceiling function applied to the squared value
def ceil_squared_value : ℤ := Int.ceil squared_value

-- Prove that the ceiling of the squared value is 4
theorem ceil_of_neg_frac_squared : ceil_squared_value = 4 := 
by sorry

end ceil_of_neg_frac_squared_l213_213205


namespace angle_between_a_c_l213_213672

noncomputable def angle_between (a b : ℝ → ℝ → ℝ) : ℝ :=
  let dot_product := λ x y, x.1 * y.1 + x.2 * y.2 in
  real.acos ((dot_product a b) / (real.sqrt (dot_product a a) * real.sqrt (dot_product b b)))

theorem angle_between_a_c (a b c : ℝ → ℝ → ℝ) 
  (ha : a ≠ 0)
  (hb : b ≠ 0)
  (hc : c ≠ 0)
  (h_sum : a + b + c = (0, 0))
  (h_angle_ab : angle_between a b = real.pi / 3)
  (h_norm_a : real.sqrt (a.1 * a.1 + a.2 * a.2) = 1)
  (h_norm_b : real.sqrt (b.1 * b.1 + b.2 * b.2) = 1) :
  angle_between a c = 5 * real.pi / 6 :=
sorry

end angle_between_a_c_l213_213672


namespace value_of_g7_l213_213730

def g (x : ℝ) : ℝ := (2 * x + 3) / (4 * x - 5)

theorem value_of_g7 : g 7 = 17 / 23 :=
by sorry

end value_of_g7_l213_213730


namespace ceil_square_neg_seven_over_four_l213_213207

theorem ceil_square_neg_seven_over_four : 
  let x := - (7 / 4 : ℚ) in
  ⌈x^2⌉ = 4 :=
by
  let x := - (7 / 4 : ℚ)
  sorry

end ceil_square_neg_seven_over_four_l213_213207


namespace ceil_of_neg_frac_squared_l213_213197

-- Define the negated fraction
def neg_frac : ℚ := -7 / 4

-- Define the squared value of the negated fraction
def squared_value : ℚ := neg_frac ^ 2

-- Define the ceiling function applied to the squared value
def ceil_squared_value : ℤ := Int.ceil squared_value

-- Prove that the ceiling of the squared value is 4
theorem ceil_of_neg_frac_squared : ceil_squared_value = 4 := 
by sorry

end ceil_of_neg_frac_squared_l213_213197


namespace solve_for_b_l213_213727

theorem solve_for_b (b : ℚ) (h : b + b / 4 = 5 / 2) : b = 2 := 
sorry

end solve_for_b_l213_213727


namespace frac_series_simplification_l213_213575

theorem frac_series_simplification :
  (1 - 2 + 4 - 8 + 16 - 32 + 64 - 128 : ℚ) / (1^2 + 2^2 - 4^2 + 8^2 + 16^2 - 32^2 + 64^2 - 128^2 : ℚ) = 1 / 113 := 
by
  sorry

end frac_series_simplification_l213_213575


namespace max_apples_emmy_gerry_l213_213047

def price_of_apples := 2
def price_of_bananas := 1
def price_of_oranges := 3
def emmy_budget := 200
def gerry_budget := 100
def minimum_bananas := 5
def minimum_oranges := 5
def apple_discount_threshold := 10
def apple_discount_rate := 0.2

theorem max_apples_emmy_gerry : 
    let discounted_price_of_apples := price_of_apples * (1 - apple_discount_rate),
        minimum_banana_cost := minimum_bananas * price_of_bananas,
        minimum_orange_cost := minimum_oranges * price_of_oranges,
        emmy_remaining_budget := emmy_budget - (minimum_banana_cost + minimum_orange_cost),
        gerry_remaining_budget := gerry_budget - (minimum_banana_cost + minimum_orange_cost),
        emmy_apples := (emmy_remaining_budget / discounted_price_of_apples).toInt,
        gerry_apples := (gerry_remaining_budget / discounted_price_of_apples).toInt
    in emmy_apples + gerry_apples = 160 :=
by
  sorry

end max_apples_emmy_gerry_l213_213047


namespace ceiling_of_square_frac_l213_213220

theorem ceiling_of_square_frac : 
  let x : ℚ := -7 / 4
  let y : ℚ := x^2
  in Int.ceil y = 4 := 
by 
  let x : ℚ := -7 / 4
  let y : ℚ := x^2
  exact sorry

end ceiling_of_square_frac_l213_213220


namespace ceil_square_neg_seven_over_four_l213_213210

theorem ceil_square_neg_seven_over_four : 
  let x := - (7 / 4 : ℚ) in
  ⌈x^2⌉ = 4 :=
by
  let x := - (7 / 4 : ℚ)
  sorry

end ceil_square_neg_seven_over_four_l213_213210


namespace martha_savings_l213_213814

def daily_allowance : ℝ := 12
def saving_days : ℕ := 6
def saving_half (amount : ℝ) : ℝ := (1/2) * amount
def saving_quarter (amount : ℝ) : ℝ := (1/4) * amount

theorem martha_savings : 
  saving_days * saving_half daily_allowance + saving_quarter daily_allowance = 39 := 
by
  sorry

end martha_savings_l213_213814


namespace period_tan_transformed_l213_213916

theorem period_tan_transformed :
  (∃ p : ℝ, ∀ θ : ℝ, tan (θ + p) = tan θ) → (∃ q : ℝ, ∀ x : ℝ, tan (3/4 * x + q) = tan (3/4 * x)) :=
by
  sorry

end period_tan_transformed_l213_213916


namespace at_least_two_dice_same_number_probability_l213_213832

theorem at_least_two_dice_same_number_probability :
  let total_outcomes := 6^8
  let favorable_outcomes := 28 * 6! * 6^2
  let probability_all_different := favorable_outcomes / total_outcomes
  let required_probability := 1 - probability_all_different
  required_probability = 191 / 336
:= by
  sorry

end at_least_two_dice_same_number_probability_l213_213832


namespace largest_4digit_div_by_35_l213_213506

theorem largest_4digit_div_by_35 : ∃ n, (1000 ≤ n) ∧ (n ≤ 9999) ∧ (35 ∣ n) ∧ (∀ m, (1000 ≤ m) ∧ (m ≤ 9999) ∧ (35 ∣ m) → m ≤ n) ∧ n = 9985 :=
by
  sorry

end largest_4digit_div_by_35_l213_213506


namespace trajectory_passes_quadrants_l213_213380

-- Define the equation of the circle
def circle_equation (x y : ℝ) : Prop :=
  (x - 1)^2 + (y - 3)^2 = 4

-- Define the condition for a point to belong to the first quadrant
def in_first_quadrant (x y : ℝ) : Prop :=
  x > 0 ∧ y > 0

-- Define the condition for a point to belong to the second quadrant
def in_second_quadrant (x y : ℝ) : Prop :=
  x < 0 ∧ y > 0

-- State the theorem that the trajectory of point P passes through the first and second quadrants
theorem trajectory_passes_quadrants :
  (∃ x y : ℝ, circle_equation x y ∧ in_first_quadrant x y) ∧
  (∃ x y : ℝ, circle_equation x y ∧ in_second_quadrant x y) :=
sorry

end trajectory_passes_quadrants_l213_213380


namespace exists_non_parallel_diagonal_convex_polygon_l213_213844

theorem exists_non_parallel_diagonal_convex_polygon (k : ℕ) (h : ∀ (P : list (ℝ × ℝ)), convex_polygon P → (even (length P) ∧ length P = 2 * k)) :
  ∃ (d : (ℝ × ℝ) × (ℝ × ℝ)), d ∈ diagonals P ∧ ∀ (s : ℝ × ℝ), ¬parallel d s :=
begin
  sorry
end

end exists_non_parallel_diagonal_convex_polygon_l213_213844


namespace derivative_at_one_l213_213690

-- Define the function f and the given condition
variable {f : ℝ → ℝ}
axiom func_condition : ∀ x, f(x) = 2*f(2 - x) - x^2 + 8*x - 8

-- State the proof goal
theorem derivative_at_one : f'(1) = 2 :=
by
  sorry

end derivative_at_one_l213_213690


namespace floor_identity_frac_part_identity_l213_213846

namespace Proofs

  def def_floor (x : ℝ) : ℤ := int.floor x
  def frac_part (x : ℝ) : ℝ := x - def_floor x

  theorem floor_identity (x : ℝ) :
    def_floor (3 * x) = def_floor x + def_floor (x + 1/3) + def_floor (x + 2/3) :=
  by
    sorry

  theorem frac_part_identity (x : ℝ) :
    frac_part (3 * x) = frac_part x + frac_part (x + 1/3) + frac_part (x + 2/3) :=
  by
    sorry

end Proofs

end floor_identity_frac_part_identity_l213_213846


namespace parallelogram_count_in_triangle_l213_213904

open Classical

theorem parallelogram_count_in_triangle
  (A B C O A' B' C' : Point)
  (h_non_eq : ¬(A = B ∨ B = C ∨ C = A))
  (h_acute : ∀ (a b c : ℝ), a^2 + b^2 > c^2)
  (h_O_eq : dist O A = dist O B ∧ dist O B = dist O C)
  (h_symmetry_A' : is_symmetric O (line_segment B C) A')
  (h_symmetry_B' : is_symmetric O (line_segment A C) B')
  (h_symmetry_C' : is_symmetric O (line_segment A B) C') :
  count_parallelograms [A, B, C, O, A', B', C'] = 6 := sorry

axiom Point : Type
axiom dist : Point → Point → ℝ
axiom line_segment : Point → Point → Set Point
axiom is_symmetric : Point → Set Point → Point → Prop
axiom count_parallelograms : List Point → ℕ

end parallelogram_count_in_triangle_l213_213904


namespace three_digit_sum_27_l213_213641

theorem three_digit_sum_27 {a b c : ℕ} (h1 : 1 ≤ a ∧ a ≤ 9) (h2 : 0 ≤ b ∧ b ≤ 9) (h3 : 0 ≤ c ∧ c ≤ 9) :
  a + b + c = 27 → (a, b, c) = (9, 9, 9) :=
by
  sorry

end three_digit_sum_27_l213_213641


namespace correlation_coefficient_correct_l213_213979

theorem correlation_coefficient_correct (r : ℝ) (h₁ : |r| ≤ 1) :
  (∀ r, |r| ≤ 1 → (|r| = 1 → "greater correlation") ∧ (|r| = 0 → "smaller correlation")) :=
by
  sorry

end correlation_coefficient_correct_l213_213979


namespace kylie_father_coins_l213_213775

theorem kylie_father_coins (F : ℕ) : 
  (let coins_piggy_bank := 15 in 
   let coins_brother := 13 in 
   let coins_given_to_Laura := 21 in 
   let coins_left := 15 in 
   coins_piggy_bank + coins_brother + F - coins_given_to_Laura = coins_left) 
   → F = 8 :=
by
  intro h
  sorry

end kylie_father_coins_l213_213775


namespace median_is_71_l213_213131

-- Define the sequence where each integer n appears n times for 1 ≤ n ≤ 100
def sequence : List ℕ := List.join (List.map (fun n => List.replicate n n) (List.range' 1 101))

-- Total number of elements in the list
def N : ℕ := 5050

-- Define the median as the average of the 2525-th and 2526-th numbers in the sequence
def median (l : List ℕ) : ℕ :=
  let sorted := l.qsort (· ≤ ·)
  (sorted.get! 2524 + sorted.get! 2525) / 2

theorem median_is_71 : median sequence = 71 := by
  sorry

end median_is_71_l213_213131


namespace dot_product_BC_CA_l213_213366

-- Definitions based on conditions
def AC : ℝ := 8
def BC : ℝ := 5
def area_ABC : ℝ := 10 * Real.sqrt 3

-- Main theorem to prove the dot product
theorem dot_product_BC_CA :
  ∃ cos_ACB : ℝ, (cos_ACB = 1 / 2 ∨ cos_ACB = -1 / 2) → (BC * AC * cos_ACB = 20 ∨ BC * AC * cos_ACB = -20) :=
by
  sorry

end dot_product_BC_CA_l213_213366


namespace arc_containment_l213_213944

theorem arc_containment (n : ℕ) (h : n ≥ 1) :
  ∃ (i j : ℕ) (h_l_i : 1 ≤ i ∧ i ≤ (n + 1)) (h_l_j : 1 ≤ j ∧ j ≤ (n + 1)), (i ≠ j) ∧ (arc_length i) ⊆ (arc_length j) :=
by
  sorry


end arc_containment_l213_213944


namespace average_score_boys_combined_is_67_l213_213898

-- Definitions for the average scores at Carson and Dylan High Schools and combined schools
variables 
  (C c D d : ℕ) -- The number of boys and girls at Carson (C, c) and Dylan (D, d)
  (BoysAvgCarson GirlsAvgCarson CombinedAvgCarson : ℝ)
  (BoysAvgDylan GirlsAvgDylan CombinedAvgDylan : ℝ)
  (CombinedGirlsAvg : ℝ)

-- Conditions given in the problem
axiom CarsonAvgs : BoysAvgCarson = 65 ∧ GirlsAvgCarson = 75 ∧ CombinedAvgCarson = 68
axiom DylanAvgs : BoysAvgDylan = 70 ∧ GirlsAvgDylan = 85 ∧ CombinedAvgDylan = 75
axiom CombinedGirlsAvgCond : CombinedGirlsAvg = 80

-- Definition of average calculation for both schools combined
noncomputable def CombinedBoysAvg := (65 * (7/3 * c) + 70 * (2 * c)) / ((7/3 * c) + 2 * c)

-- The statement to be proven
theorem average_score_boys_combined_is_67 :
  CarsonAvgs ∧ DylanAvgs ∧ CombinedGirlsAvgCond → CombinedBoysAvg = 67 := 
by
  sorry

end average_score_boys_combined_is_67_l213_213898


namespace selling_price_l213_213554

/-- 
Prove that the selling price (S) of an article with a cost price (C) of 180 sold at a 15% profit (P) is 207.
-/
theorem selling_price (C P S : ℝ) (hC : C = 180) (hP : P = 15) (hS : S = 207) :
  S = C + (P / 100 * C) :=
by
  -- here we rely on sorry to skip the proof details
  sorry

end selling_price_l213_213554


namespace prob_same_color_is_correct_prob_diff_colors_two_days_is_correct_l213_213742

noncomputable def prob_same_color (black: ℕ) (red: ℕ) : ℚ :=
  let total := black + red
  (black / total) * (black / total) + (red / total) * (red / total)

noncomputable def prob_diff_colors_two_days (black: ℕ) (red: ℕ) : ℚ :=
  let same_color := prob_same_color black red
  let diff_color := 1 - same_color
  (nat.choose 4 2) * (same_color * same_color) * (diff_color * diff_color)

theorem prob_same_color_is_correct :
  prob_same_color 4 2 = 5 / 9 :=
by { sorry }

theorem prob_diff_colors_two_days_is_correct :
  prob_diff_colors_two_days 4 2 = 800 / 2187 :=
by { sorry }

end prob_same_color_is_correct_prob_diff_colors_two_days_is_correct_l213_213742


namespace part_1_part_2_part_3_l213_213557

-- Problem conditions and requirements as Lean definitions and statements
def y_A (x : ℝ) : ℝ := (2 / 5) * x
def y_B (x : ℝ) : ℝ := -(1 / 5) * x^2 + 2 * x

-- Part (1)
theorem part_1 : y_A 10 = 4 := sorry

-- Part (2)
theorem part_2 (m : ℝ) (h : m > 0) : y_A m = y_B m → m = 3 := sorry

-- Part (3)
def W (x_A x_B : ℝ) : ℝ := y_A x_A + y_B x_B

theorem part_3 (x_A x_B : ℝ) (h : x_A + x_B = 32) : 
  (∀ x_A' x_B', x_A' + x_B' = 32 → W x_A x_B ≥ W x_A' x_B') ∧ W x_A x_B = 16 :=
begin
  sorry
end

end part_1_part_2_part_3_l213_213557


namespace line_eq_parallel_asymptotes_l213_213873

def is_parallel (L1 L2 : ℝ → ℝ → Prop) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ ∀ x y, L1 x y = 0 → L2 x (k * y) = 0

def distance_between_lines (a b c1 c2 : ℝ) : ℝ :=
  abs (c1 - c2) / sqrt (a ^ 2 + b ^ 2)

def line_parallel_to_asymptote (t : ℝ) : ℝ → ℝ → Prop :=
  λ x y, sqrt 5 * x + 2 * y + t = 0

theorem line_eq_parallel_asymptotes 
  (x y : ℝ)
  (h1 : ∃ k : ℝ, k ≠ 0 ∧ (λ x y, sqrt 5 * x + 2 * y = 0) x (k * y))
  (h2 : ((λ t, abs t / sqrt (5 + 4) = 1) t)) :
  line_parallel_to_asymptote t x y → t = 3 ∨ t = -3 :=
sorry

end line_eq_parallel_asymptotes_l213_213873


namespace eighth_term_l213_213888

noncomputable def S (n : ℕ) (a : ℕ → ℤ) : ℤ := (n * (a 1 + a n)) / 2

variables {a : ℕ → ℤ} {d : ℤ}

-- Conditions
axiom sum_of_first_n_terms : ∀ n : ℕ, S n a = (n * (a 1 + a n)) / 2
axiom second_term : a 2 = 3
axiom sum_of_first_five_terms : S 5 a = 25

-- Question
theorem eighth_term : a 8 = 15 :=
sorry

end eighth_term_l213_213888


namespace incorrect_statement_B_l213_213926

-- Definitions related to the problem
def frequency_event (freq : ℕ → ℝ) := 
  ∀ n, (freq n) ≥ 0 ∧ (freq n) ≤ 1

def mutually_exclusive (A B : Prop) := 
  ¬ (A ∧ B)

def is_geometric_distribution (f : ℕ → ℝ) := 
  ∃ p : ℝ, 0 < p ∧ p ≤ 1 ∧ ∀ k, f k = (1 - p) ^ (k - 1) * p

-- Events for the newspaper subscription
def event_B (subscribed_A subscribed_B : Prop) := 
  subscribed_A ∨ subscribed_B

def event_C (subscribed_A subscribed_B : Prop) := 
  ¬ (subscribed_A ∧ subscribed_B)

-- Main theorem statement to prove
theorem incorrect_statement_B : 
  ∃ A B : Prop, event_B A B ∧ event_C A B ∧ ∃ P C : Prop, mutually_exclusive P C ∧ ∀ x : ℤ, -10 < x ∧ x < 10 → (x > 1 ∧ x < 5) ↔ (∃ p : ℝ, is_geometric_distribution (λ n : ℕ, p^n)) := 
sorry

end incorrect_statement_B_l213_213926


namespace smallest_x_value_l213_213002

theorem smallest_x_value : ∃ x : ℝ, (x = 0) ∧ (∀ y : ℝ, (left_side y x) = 0 → x ≤ y)
where
  left_side y x : ℝ := (y^2 + y - 20)

limit smaller

end smallest_x_value_l213_213002


namespace find_total_kids_l213_213896

-- Given conditions
def total_kids_in_camp (X : ℕ) : Prop :=
  let soccer_kids := X / 2
  let morning_soccer_kids := soccer_kids / 4
  let afternoon_soccer_kids := soccer_kids - morning_soccer_kids
  afternoon_soccer_kids = 750

-- Theorem statement
theorem find_total_kids (X : ℕ) (h : total_kids_in_camp X) : X = 2000 :=
by
  sorry

end find_total_kids_l213_213896


namespace count_non_carrying_pairs_in_range_l213_213639

def no_carrying_in_addition (a b : Nat) : Prop :=
  ∀ (n : ℕ), ((a / 10^n) % 10) + ((b / 10^n) % 10) < 10

def consecutive_pairs (n m : Nat) : List (Nat × Nat) :=
  List.zip (List.range' n (m - n + 1)) (List.range' (n + 1) (m - n + 1))

theorem count_non_carrying_pairs_in_range (count : Nat) :
  count = List.length (List.filter (λ (p: Nat × Nat), no_carrying_in_addition p.fst p.snd) (consecutive_pairs 950 2050)) → count = 988 :=
by sorry

end count_non_carrying_pairs_in_range_l213_213639


namespace sqrt_36_l213_213472

theorem sqrt_36 : {x : ℝ // x^2 = 36} = {6, -6} :=
by
  sorry

end sqrt_36_l213_213472


namespace determine_range_m_l213_213780

noncomputable def quadratic_root_set (m : ℝ) : Set ℝ :=
  {x | x^2 - m * x + 3 = 0}

theorem determine_range_m (m : ℝ) :
  (∀ x, x ∈ quadratic_root_set(m) → x = 1 ∨ x = 3) ↔ (m ∈ Ioo (-2 * Real.sqrt 3) (2 * Real.sqrt 3) ∨ m = 4) := 
sorry

end determine_range_m_l213_213780


namespace ellipse_problem_l213_213676

noncomputable def distance_to_line (c: ℝ) : Prop :=
  real.sqrt 3 * c = 2 * real.sqrt 3

noncomputable def ellipse_focal_length (a b c: ℝ) : Prop :=
  2 * c = 4

noncomputable def ellipse_equation (a b: ℝ) : Prop :=
  (a = 3) ∧ (b = real.sqrt 5)

theorem ellipse_problem (a b c : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : distance_to_line c) :
  ellipse_focal_length a b c ∧ ellipse_equation a b := 
sorry

end ellipse_problem_l213_213676


namespace find_t2_l213_213428

variable {P A1 A2 t1 r t2 : ℝ}
def conditions (P A1 A2 t1 r t2 : ℝ) :=
  P = 650 ∧
  A1 = 815 ∧
  A2 = 870 ∧
  t1 = 3 ∧
  A1 = P + (P * r * t1) / 100 ∧
  A2 = P + (P * r * t2) / 100

theorem find_t2
  (P A1 A2 t1 r t2 : ℝ)
  (hc : conditions P A1 A2 t1 r t2) :
  t2 = 4 :=
by
  sorry

end find_t2_l213_213428


namespace solving_histogram_issue_l213_213924

theorem solving_histogram_issue (data_points : List ℚ) (dividing_points : List ℚ)
  (h : ∃ x ∈ data_points, x ∈ dividing_points) :
  (∃ d : ℚ, x = d ∧ d ∈ dividing_points ∧ ∀ dp ∈ dividing_points, (∃ n : ℕ, d = dp + 10^(-n))) :=
sorry

end solving_histogram_issue_l213_213924


namespace arc_length_correct_l213_213570

-- Define the function ρ(φ) = 2 * cos(φ)
def rho (φ : ℝ) : ℝ := 2 * Real.cos φ

-- Define the derivative of ρ with respect to φ
def drho (φ : ℝ) : ℝ := -2 * Real.sin φ

-- Define the integrand for the arc length integral
def integrand (φ : ℝ) : ℝ := Real.sqrt ((rho φ)^2 + (drho φ)^2)

-- Define the arc length as the integral of the integrand from 0 to π/6
def arc_length : ℝ := ∫ φ in (0 : ℝ)..(Real.pi / 6), integrand φ

-- Proof obligation: The arc length is equal to π / 3
theorem arc_length_correct : arc_length = Real.pi / 3 :=
by
  sorry

end arc_length_correct_l213_213570


namespace smallest_x_value_l213_213000

theorem smallest_x_value : ∃ x : ℝ, (x = 0) ∧ (∀ y : ℝ, (left_side y x) = 0 → x ≤ y)
where
  left_side y x : ℝ := (y^2 + y - 20)

limit smaller

end smallest_x_value_l213_213000


namespace greatest_visible_unit_cubes_from_corner_l213_213029

theorem greatest_visible_unit_cubes_from_corner
  (n : ℕ) (units : ℕ) 
  (cube_volume : ∀ x, x = 1000)
  (face_size : ∀ x, x = 10) :
  (units = 274) :=
by sorry

end greatest_visible_unit_cubes_from_corner_l213_213029


namespace student_arrangement_count_l213_213376

theorem student_arrangement_count : 
  ∃ (students : Fin 5 → ℕ), 
  (∀ (i j : Fin 5), (students i = students j → i = j)) ∧
  (∃ (A B : Fin 5), A ≠ B ∧ 
   (∀ (k : Fin 5), k ≠ A ∧ k ≠ B → students k ≠ students A ∧ students k ≠ students B)) ∧ 
  (students 0 = 5 ∨ students 4 = 5) ∧ 
  let valid_arrangements := 24 
  in 
  valid_arrangements = 24 := sorry

end student_arrangement_count_l213_213376


namespace axis_of_symmetry_sine_function_l213_213872

theorem axis_of_symmetry_sine_function :
  ∃ k : ℤ, x = k * (π / 2) := sorry

end axis_of_symmetry_sine_function_l213_213872


namespace number_of_ways_to_place_letters_l213_213839

-- Define the number of letters and mailboxes
def num_letters : Nat := 3
def num_mailboxes : Nat := 5

-- Define the function to calculate the number of ways to place the letters into mailboxes
def count_ways (n : Nat) (m : Nat) : Nat := m ^ n

-- The theorem to prove
theorem number_of_ways_to_place_letters :
  count_ways num_letters num_mailboxes = 5 ^ 3 :=
by
  sorry

end number_of_ways_to_place_letters_l213_213839


namespace trapezoid_area_l213_213091

theorem trapezoid_area 
  (a b h c : ℝ) 
  (ha : 2 * 0.8 * a + b = c)
  (hb : c = 20) 
  (hc : h = 0.6 * a) 
  (hd : b + 1.6 * a = 20)
  (angle_base : ∃ θ : ℝ, θ = arcsin 0.6)
  : 
  (1 / 2) * (b + c) * h = 72 :=
sorry

end trapezoid_area_l213_213091


namespace cuboid_first_edge_length_l213_213870

theorem cuboid_first_edge_length (x : ℝ) (hx : 180 = x * 5 * 6) : x = 6 :=
by
  sorry

end cuboid_first_edge_length_l213_213870


namespace at_least_two_dice_same_number_probability_l213_213831

theorem at_least_two_dice_same_number_probability :
  let total_outcomes := 6^8
  let favorable_outcomes := 28 * 6! * 6^2
  let probability_all_different := favorable_outcomes / total_outcomes
  let required_probability := 1 - probability_all_different
  required_probability = 191 / 336
:= by
  sorry

end at_least_two_dice_same_number_probability_l213_213831


namespace digit_is_9_if_divisible_by_11_l213_213263

theorem digit_is_9_if_divisible_by_11 (d : ℕ) : 
  (678000 + 9000 + 800 + 90 + d) % 11 = 0 -> d = 9 := by
  sorry

end digit_is_9_if_divisible_by_11_l213_213263


namespace monotonic_decreasing_interval_l213_213463

def f (x : ℝ) : ℝ := Real.logb (1/2) (x^2 - 6 * x + 5)

theorem monotonic_decreasing_interval : 
  (∀ x, (x ∈ Set.Ioc 5 ∞) → x^2 - 6 * x + 5 > 0) → 
  (∀ x y, 5 < x ∧ x < y → f(x) ≥ f(y)) := 
by
  intro h x y hxy
  sorry

end monotonic_decreasing_interval_l213_213463


namespace no_such_abc_exists_l213_213875

theorem no_such_abc_exists :
  ¬ ∃ (a b c : ℝ), 
      ((a > 0 ∧ b > 0 ∧ c < 0) ∨ (a > 0 ∧ c > 0 ∧ b < 0) ∨ (b > 0 ∧ c > 0 ∧ a < 0) ∨
       (a < 0 ∧ b < 0 ∧ c > 0) ∨ (a < 0 ∧ c < 0 ∧ b > 0) ∨ (b < 0 ∧ c < 0 ∧ a > 0)) ∧
      ((a < 0 ∧ b < 0 ∧ c > 0) ∨ (a < 0 ∧ c < 0 ∧ b > 0) ∨ (b < 0 ∨ c < 0 ∧ a > 0) ∨
       (a > 0 ∧ b > 0 ∧ c < 0) ∨ (a > 0 ∧ c > 0 ∧ b < 0) ∨ (b > 0 ∧ c > 0 ∧ a < 0)) :=
by {
  sorry
}

end no_such_abc_exists_l213_213875


namespace Xiaobing_jumps_189_ropes_per_minute_l213_213148

-- Define conditions and variables
variable (x : ℕ) -- The number of ropes Xiaohan jumps per minute

-- Conditions:
-- 1. Xiaobing jumps x + 21 ropes per minute
-- 2. Time taken for Xiaobing to jump 135 ropes is the same as the time taken for Xiaohan to jump 120 ropes

theorem Xiaobing_jumps_189_ropes_per_minute (h : 135 * x = 120 * (x + 21)) :
    x + 21 = 189 :=
by
  sorry -- Proof is not required as per instructions

end Xiaobing_jumps_189_ropes_per_minute_l213_213148


namespace train_cross_signal_pole_time_l213_213032

theorem train_cross_signal_pole_time :
  ∀ (l_t l_p t_p : ℕ), l_t = 450 → l_p = 525 → t_p = 39 → 
  (l_t * t_p) / (l_t + l_p) = 18 := by
  sorry

end train_cross_signal_pole_time_l213_213032


namespace B_C_H_on_circle_with_center_E_l213_213377

-- Define the given conditions

variables {α : Type} [Euclidean α]

variables {B C H E : Point α}

def equal_segments (BE CE : Segment α) : Prop := 
  length BE = length CE

def angle_BHC (β : Real) : angle α := 
  measure_angle B H C = β

def angle_BEC (φ β : Real) : angle α := 
  measure_angle B E C = φ ∧ 
  φ = 2 * β

-- Prove that points B, C, and H lie on a circle with center E

theorem B_C_H_on_circle_with_center_E
( BE CE : Segment α )
( φ β : Real )
( h1 : equal_segments BE CE )
( h2 : angle_BHC β )
( h3 : angle_BEC φ β )
: OnCircle B C H E :=
sorry

end B_C_H_on_circle_with_center_E_l213_213377


namespace number_of_correct_statements_l213_213880

def statement1_condition : Prop :=
∀ a b : ℝ, (a - b > 0) → (a > 0 ∧ b > 0)

def statement2_condition : Prop :=
∀ a b : ℝ, a - b = a + (-b)

def statement3_condition : Prop :=
∀ a : ℝ, (a - (-a) = 0)

def statement4_condition : Prop :=
∀ a : ℝ, 0 - a = -a

theorem number_of_correct_statements : 
  (¬ statement1_condition ∧ statement2_condition ∧ ¬ statement3_condition ∧ statement4_condition) →
  (2 = 2) :=
by
  intros
  trivial

end number_of_correct_statements_l213_213880


namespace injective_function_count_non_injective_function_count_no_surjective_function_l213_213408

theorem injective_function_count (A B : Type) (hA : A = {1, 2, 3}) (hB : B = {1, 2, 3, 4}) :
  ∃ (f : A → B), function.injective f ∧ (finset.univ).card = 24 :=
sorry

theorem non_injective_function_count (A B : Type) (hA : A = {1, 2, 3}) (hB : B = {1, 2, 3, 4}) :
  ∃ (f : A → B), ¬ function.injective f ∧ (finset.univ).card = 40 :=
sorry

theorem no_surjective_function (A B : Type) (hA : A = {1, 2, 3}) (hB : B = {1, 2, 3, 4}) :
  ¬ ∃ (f : A → B), function.surjective f :=
sorry

end injective_function_count_non_injective_function_count_no_surjective_function_l213_213408


namespace find_range_of_num_on_die_l213_213521

-- Defining the conditions
def coin_toss {α : Type*} [Fintype α] :=
  (8 : Fin 8 → α)

axiom first_toss_tail {α : Type*} [Fintype α] :
  Π (s : set α), (coin_toss s).head = False

axiom equally_likely_events {α : Type*} [Fintype α] :
  ∀ s : set α, card s = 8

axiom die_roll_prob {α : Type*} [Fintype α] [Nonempty α] (r : fin 6 → Prop) :
  (1/3 : ℝ)

-- Defining the problem in Lean statement
theorem find_range_of_num_on_die {α : Type*} [Fintype α] [decidable_eq (fin 6)] :
  ∃ ranges : list (finset (fin 6)), 
    (∀ range : finset (fin 6), range.card = 2 → range ∈ ranges ∧ 
    (die_roll_prob (λ x, x ∈ range) = 1/3)) :=
sorry

end find_range_of_num_on_die_l213_213521


namespace length_of_AC_l213_213040

theorem length_of_AC (O A B C D : Point) (r : ℝ)
    (h_circle : (dist O A = r) ∧ (dist O B = r))
    (h_angle : angle A O B = real.pi / 2)
    (h_cd : dist C D = 3 * r)
    (h_parallel : parallel (line_through A B) (line_through C D)) :
    dist O C = dist O D ∧ dist C D = 3 * r ∧ dist A C = r + (dist C D) / real.sqrt 2 :=
sorry

end length_of_AC_l213_213040


namespace smallest_books_for_students_l213_213946

theorem smallest_books_for_students : 
  let n := 168 in
  ∀ k, (k = 3 ∨ k = 4 ∨ k = 6 ∨ k = 7 ∨ k = 8) → n % k = 0 :=
by
  sorry

end smallest_books_for_students_l213_213946


namespace min_marked_cells_l213_213284

theorem min_marked_cells (k : ℕ) :
  let f := (⌊(k + 1) / 2⌋ * ⌊(k + 2) / 2⌋ : ℕ)
  f = ⌊(k + 1) / 2⌋ * ⌊(k + 2) / 2⌋ 
:= sorry

end min_marked_cells_l213_213284


namespace solution_system_linear_eqns_l213_213664

theorem solution_system_linear_eqns
    (a1 b1 c1 a2 b2 c2 : ℝ)
    (h1: a1 * 6 + b1 * 3 = c1)
    (h2: a2 * 6 + b2 * 3 = c2) :
    (4 * a1 * 22 + 3 * b1 * 33 = 11 * c1) ∧
    (4 * a2 * 22 + 3 * b2 * 33 = 11 * c2) :=
by
    sorry

end solution_system_linear_eqns_l213_213664


namespace ceil_square_neg_fraction_l213_213182

theorem ceil_square_neg_fraction :
  let x := (-7) / 4
  let y := x^2
  let z := Real.ceil y
  z = 4 := 
by
  sorry

end ceil_square_neg_fraction_l213_213182


namespace simplify_radicals_l213_213444

open Real

theorem simplify_radicals : sqrt 72 + sqrt 32 = 10 * sqrt 2 := by
  sorry

end simplify_radicals_l213_213444


namespace no_single_x_for_doughnut_and_syrup_l213_213371

theorem no_single_x_for_doughnut_and_syrup :
  ¬ ∃ x : ℝ, (x^2 - 9 * x + 13 < 0) ∧ (x^2 + x - 5 < 0) :=
sorry

end no_single_x_for_doughnut_and_syrup_l213_213371


namespace average_speed_remainder_l213_213768

/--
Jason has to drive home which is 120 miles away.
If he drives at 60 miles per hour for 30 minutes,
prove that he has to average 90 miles per hour for the remainder of the drive
to get there in exactly 1 hour 30 minutes.
-/
theorem average_speed_remainder (total_distance : ℕ) (initial_speed : ℕ) (initial_time : ℕ) (total_time : ℕ) :
  total_distance = 120 →
  initial_speed = 60 →
  initial_time = 30 →
  total_time = 90 →
  average_speed_remainder = 90 :=
by
  sorry

end average_speed_remainder_l213_213768


namespace ratio_HD_HA_zero_l213_213128

noncomputable def triangle_ratio : ℝ × ℝ × ℝ → ℝ × ℝ × ℝ → Prop :=
λ sides altitudes, 
  let s₁ := sides.1, s₂ := sides.2, s₃ := sides.3 in
  let h₁ := altitudes.1, h₂ := altitudes.2, h₃ := altitudes.3 in
  (s₁ = 8 ∧ s₂ = 15 ∧ s₃ = 17) ∧
  h₃ = (0 : ℝ) / (h₁ : ℝ)

theorem ratio_HD_HA_zero :
  ∀ (sides : ℝ × ℝ × ℝ) (altitudes : ℝ × ℝ × ℝ),
  triangle_ratio sides altitudes → 
    (altitudes.3 / altitudes.1 = 0) :=
by {
  intros sides altitudes h,
  cases sides with s₁ sides, cases sides with s₂ s₃,
  cases altitudes with h₁ altitudes, cases altitudes with h₂ h₃,
  simp [triangle_ratio] at h,
  cases h with hs a_ratio,
  cases hs with hs₁ hs, cases hs with hs₂ hs₃,
  sorry
}

end ratio_HD_HA_zero_l213_213128


namespace find_floor_abs_S_l213_213129

-- Conditions
-- For integers from 1 to 1500, x_1 + 2 = x_2 + 4 = x_3 + 6 = ... = x_1500 + 3000 = ∑(n=1 to 1500) x_n + 3001
def condition (x : ℕ → ℤ) (S : ℤ) : Prop :=
  ∀ a : ℕ, 1 ≤ a ∧ a ≤ 1500 →
    x a + 2 * a = S + 3001

-- Problem statement
theorem find_floor_abs_S (x : ℕ → ℤ) (S : ℤ)
  (h : condition x S) :
  (⌊|S|⌋ : ℤ) = 1500 :=
sorry

end find_floor_abs_S_l213_213129


namespace complex_coordinates_l213_213656

theorem complex_coordinates (z : ℂ) (i_unit : ℂ) (h : i_unit = complex.I) 
  (hz : z = (1 + 2 * i_unit^3) / (2 + i_unit)) : z = 0 - complex.I :=
by sorry

end complex_coordinates_l213_213656


namespace number_of_tetrahedrons_number_of_planes_number_of_lines_l213_213278

/-- Given 5 points, any 4 of which do not lie in the same plane, 
    prove that the number of tetrahedrons formed by choosing 4 out of these 5 points is 5. -/
theorem number_of_tetrahedrons (n : ℕ) (h_n : n = 5) : (nat.choose 5 4) = 5 := by
  sorry

/-- Given 5 points, any 4 of which do not lie in the same plane, 
    prove that the number of planes determined by choosing 3 out of these 5 points is 10. -/
theorem number_of_planes (n : ℕ) (h_n : n = 5) : (nat.choose 5 3) = 10 := by
  sorry

/-- Given 5 points, any 4 of which do not lie in the same plane, 
    prove that the number of lines determined by choosing 2 out of these 5 points is 10. -/
theorem number_of_lines (n : ℕ) (h_n : n = 5) : (nat.choose 5 2) = 10 := by
  sorry

end number_of_tetrahedrons_number_of_planes_number_of_lines_l213_213278


namespace ceil_of_neg_frac_squared_l213_213198

-- Define the negated fraction
def neg_frac : ℚ := -7 / 4

-- Define the squared value of the negated fraction
def squared_value : ℚ := neg_frac ^ 2

-- Define the ceiling function applied to the squared value
def ceil_squared_value : ℤ := Int.ceil squared_value

-- Prove that the ceiling of the squared value is 4
theorem ceil_of_neg_frac_squared : ceil_squared_value = 4 := 
by sorry

end ceil_of_neg_frac_squared_l213_213198


namespace ceil_square_neg_fraction_l213_213178

theorem ceil_square_neg_fraction :
  let x := (-7) / 4
  let y := x^2
  let z := Real.ceil y
  z = 4 := 
by
  sorry

end ceil_square_neg_fraction_l213_213178


namespace train_speed_80_kmh_l213_213031

def speed_of_train (length_of_train_m : ℝ) (time_to_cross_s : ℝ) (speed_of_man_kmh : ℝ) : ℝ :=
  let speed_of_man_ms := speed_of_man_kmh * (1000 / 3600)
  let relative_speed_ms := length_of_train_m / time_to_cross_s
  let speed_of_train_ms := relative_speed_ms + speed_of_man_ms
  speed_of_train_ms * (3600 / 1000)

theorem train_speed_80_kmh :
  speed_of_train 220 (10.999120070394369) 8 = 80 :=
by
  sorry

end train_speed_80_kmh_l213_213031


namespace revenue_highest_visitors_is_48_thousand_l213_213518

-- Define the frequencies for each day
def freq_Oct_1 : ℝ := 0.05
def freq_Oct_2 : ℝ := 0.08
def freq_Oct_3 : ℝ := 0.09
def freq_Oct_4 : ℝ := 0.13
def freq_Oct_5 : ℝ := 0.30
def freq_Oct_6 : ℝ := 0.15
def freq_Oct_7 : ℝ := 0.20

-- Define the revenue on October 1st
def revenue_Oct_1 : ℝ := 80000

-- Define the revenue is directly proportional to the frequency of visitors
def avg_daily_visitor_spending_is_constant := true

-- The goal is to prove that the revenue on the day with the highest frequency is 48 thousand yuan
theorem revenue_highest_visitors_is_48_thousand :
  avg_daily_visitor_spending_is_constant →
  revenue_Oct_1 / freq_Oct_1 = x / freq_Oct_5 →
  x = 48000 :=
by
  sorry

end revenue_highest_visitors_is_48_thousand_l213_213518


namespace product_equals_16896_l213_213266

theorem product_equals_16896 (A B C D : ℕ) (h1 : A + B + C + D = 70)
  (h2 : A = 3 * C + 1) (h3 : B = 3 * C + 5) (h4 : C = C) (h5 : D = 3 * C^2) :
  A * B * C * D = 16896 :=
by
  sorry

end product_equals_16896_l213_213266


namespace ratio_of_new_to_initial_bales_l213_213989

noncomputable def initial_bales : ℕ := 10
noncomputable def initial_cost_per_bale : ℕ := 15
noncomputable def additional_money : ℕ := 210
noncomputable def new_cost_per_bale : ℕ := 18

theorem ratio_of_new_to_initial_bales :
  let initial_total_cost := initial_bales * initial_cost_per_bale in
  let additional_cost := initial_total_cost + additional_money in
  let new_bales := additional_cost / new_cost_per_bale in
  (new_bales / initial_bales) = 2 :=
by
  sorry

end ratio_of_new_to_initial_bales_l213_213989


namespace sum_of_primes_no_solution_congruence_l213_213602

theorem sum_of_primes_no_solution_congruence :
  2 + 5 = 7 :=
by
  sorry

end sum_of_primes_no_solution_congruence_l213_213602


namespace cone_base_circumference_l213_213538

theorem cone_base_circumference (V : ℝ) (h : ℝ) (C : ℝ) (r : ℝ) :
  V = 18 * Real.pi →
  h = 6 →
  (V = (1 / 3) * Real.pi * r^2 * h) →
  C = 2 * Real.pi * r →
  C = 6 * Real.pi :=
by
  intros h1 h2 h3 h4
  sorry

end cone_base_circumference_l213_213538


namespace C2_rectangular_form_OP_max_distance_EA_plus_EB_l213_213328

-- Definitions given in the problem
def curve_C1 (alpha : ℝ) : ℝ × ℝ :=
  (3 * Real.cos alpha, Real.sin alpha)

def polar_to_rectangular (rho theta : ℝ) : ℝ × ℝ :=
  (rho * Real.cos theta, rho * Real.sin theta)

def curve_C2_rectangular : Prop :=
  ∀ (x y : ℝ), x - y - 2 = 0 ↔ ∃ (rho θ : ℝ), rho * Real.cos (θ + π / 4) = sqrt 2 ∧
                    (x, y) = polar_to_rectangular rho θ

-- Proof statement for first part
theorem C2_rectangular_form : curve_C2_rectangular := 
  sorry

def OP_distance (alpha : ℝ) : ℝ :=
  let (x, y) := curve_C1 alpha in
  Real.sqrt (x^2 + y^2)

-- Proof for maximum distance of OP
theorem OP_max_distance : ∀ (P : ℝ × ℝ), P ∈ (curve_C1) → 
    OP_distance (Real.arg P) ≤ 3 := 
  sorry

-- Proof statement for second part
theorem EA_plus_EB (t1 t2 : ℝ) (h1 : t1 + t2 = -2 * Real.sqrt 2 / 5)
                   (h2 : t1 * t2 = -1) : 
  Real.abs t1 + Real.abs t2 = 6 * Real.sqrt 3 / 5 := 
  sorry

end C2_rectangular_form_OP_max_distance_EA_plus_EB_l213_213328


namespace problem_2_8_3_4_7_2_2_l213_213115

theorem problem_2_8_3_4_7_2_2 : 2 + 8 * 3 - 4 + 7 * 2 / 2 = 29 := by
  sorry

end problem_2_8_3_4_7_2_2_l213_213115


namespace number_of_functions_l213_213630

theorem number_of_functions :
  (∃ f : ℝ → ℝ , ∀ x y : ℝ, f (x + y) * f (x - y) = (f x + f y) ^ 2 - 4 * x ^ 2 * y ^ 2 * f y) →
  nat.card {f : ℝ → ℝ // ∀ x y : ℝ, f (x + y) * f (x - y) = (f x + f y) ^ 2 - 4 * x ^ 2 * y ^ 2 * f y} = 2 :=
by
  sorry

end number_of_functions_l213_213630


namespace find_all_n_l213_213260

def S (n : ℕ) : ℕ :=
  n.digits.sum

def satisfies_condition (n : ℕ) : Prop :=
  n = 2 * S(n)^3 + 8

theorem find_all_n (n : ℕ) : 
  satisfies_condition n → n = 10 ∨ n = 2008 ∨ n = 13726 :=
sorry

end find_all_n_l213_213260


namespace students_in_each_group_l213_213468

theorem students_in_each_group (num_boys : ℕ) (num_girls : ℕ) (num_groups : ℕ) 
  (h_boys : num_boys = 26) (h_girls : num_girls = 46) (h_groups : num_groups = 8) : 
  (num_boys + num_girls) / num_groups = 9 := 
by 
  sorry

end students_in_each_group_l213_213468


namespace total_discount_l213_213969

theorem total_discount (original_price : ℝ) : 
  let sale_price := 0.5 * original_price in
  let final_price := 0.75 * sale_price in
  let total_discount := 1 - final_price / original_price in
  total_discount = 0.625 := 
by
  sorry

end total_discount_l213_213969


namespace initial_members_in_d_l213_213500

variable a : ℕ 
variable b : ℕ 
variable c : ℕ 
variable d : ℕ 
variable e : ℕ 
variable f : ℕ

-- Initial number of members in the families
def families_initial : List ℕ := [7, 8, 10, d, 6, 10]

-- Number of families
def num_families : ℕ := families_initial.length

-- The sum of members in families after one member leaves each family
def sum_after_leaving : ℕ := 7 - 1 + 8 - 1 + 10 - 1 + d - 1 + 6 - 1 + 10 - 1

-- We know the new average is 8 members per family
def new_average : ℕ := 8

-- Therefore, we need to prove
theorem initial_members_in_d (h : sum_after_leaving = num_families * new_average) : d = 13 :=
by {
  sorry
}

end initial_members_in_d_l213_213500


namespace det_power_5_eq_243_l213_213723

variable (N : Matrix ℝ) (detN : det N = 3)

theorem det_power_5_eq_243 : det (N * N * N * N * N) = 243 := by
  sorry

end det_power_5_eq_243_l213_213723


namespace exists_unique_pieces_l213_213909

structure Chessboard (α : Type) :=
(pieces : α → α → Prop)
(num_pieces_row : ∀ r : α, fin 8, ∑ c : fin 8, if pieces r c then 1 else 0 = 4)
(num_pieces_col : ∀ c : α, fin 8, ∑ r : fin 8, if pieces r c then 1 else 0 = 4)

theorem exists_unique_pieces {α : Type} (board : Chessboard α) :
  ∃ (selected : fin 8 → fin 8), function.injective selected ∧ function.injective (function.inv_fun selected) :=
sorry

end exists_unique_pieces_l213_213909
